"""
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Single-GPU BF16 MoK kernel checks: contracts, determinism and the host tables.

Covers the fused forward/backward (ring replay, empty experts, variable widths, CUDA Graph replay), clamped SwiGLU,
FP32 weight-gradient accumulation, 36 and 9 local experts at top-8, context-only recompute, the context-free forward,
every exported scheduler layout and the epilogues, plus the host-only communication-split tables, resolvers and
configuration switches.  The router weight before the down projection (``router_weight_position="before_down"``):
its launchers' arguments against the registry, its refusals before device work and its targets (no GPU: every target
has its builds), and on SM100a, SM103a and SM107a its fused training at top-2 and top-6 (6 to 96 local experts; also
with another communication-SM split), zero weights, recompute and the context-free forward, the exact limit, the
384-expert scheduler layouts, and the scheduler and the adapter at the requested worst-case schedule capacity (the
routed weight gradients from ring rows past 2^31 bytes exactly).
The explicit modes (``router_weight_gradient``, ``shared_expert``, ``activation``): their refusals before device work
and their launchers' arguments against the registry (no GPU), and on SM100a, SM103a and SM107a the 512-expert top-10
group (``MODE_GROUPS["scaled_top10"]``) on inputs drawn on the CPU: the shared-scale helpers (``BF16(s * v)`` bit for
bit, the row dots repeatable at any grid size), the scaled epilogues (the exact backward sum, ``s = 1`` against the
top-8 epilogues), the direct-gradient backward against the quotient one (every shared output and the routed
down-projection weight gradient bit for bit, zero weights, the shared-scale operands at ``s = 1`` and ``s = 0``), and
the adapter at EP8 ``(8, 64)`` emulated on one GPU (nine backward items, checkpointed step, CUDA Graph, where ``s``
enters, changing source counts without kernel loads, no host synchronization in a steady-state step,
``torch.utils.checkpoint`` with ``use_reentrant=False``).
Every kernel output is checked against its contract (finite; exact integer and table outputs; documented bitwise
route forms such as recompute == saved context and graph replay == eager launch) and for run-to-run repeatability
(five launches on the same inputs give the same bytes).  The accuracy of every floating output against the float64
oracle is judged in ``test_mok_accuracy.py``.

Backend suite: the package is imported only through
:func:`tests.cake.test_helpers.cake_package.cake_package`; the :mod:`cake_pplx.cake_mok_det_train`
facade is covered by ``test_cake_mok_det_train_api.py``.
"""

import contextlib
import dataclasses

import pytest
import torch

from tests.experimental._mok_one_gpu import (
    CHANGED_COUNTS,
    OneGpuGroup,
    bound_multiplier,
    one_gpu_source_workspace,
    one_gpu_workspace,
    one_rank_ids,
    one_rank_multiplier,
    two_rank_skew_ids,
)
from tests.experimental._mok_reference import (
    normal_,
    production_inputs,
    rand,
    randint,
    randn,
    randn_like,
    require_gpu,
    single_rank_schedule,
    uniform_,
)
from tests.cake.test_helpers.cake_package import cake_package

kernels = cake_package("cake_mok_bf16", "_kernels")
api = cake_package("cake_mok_bf16", "api")
backend = cake_package("cake_mok_bf16", "backend")
jit = cake_package("cake_mok_bf16", "jit")
workspace_module = cake_package("cake_mok_bf16", "workspace")
MoKConfig = workspace_module.MoKConfig
SCHEDULER_LAYOUTS = kernels.SCHEDULER_LAYOUTS
MoKBackward = kernels.MoKBackward
MoKEpilogues = kernels.MoKEpilogues
MoKForward = kernels.MoKForward
MoKRecompute = kernels.MoKRecompute
MoKScheduler = kernels.MoKScheduler

GEOMETRIES = ((256, 256, 768), (512, 512, 768), (512, 256, 1536))


REPEATS = 5


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _require_role(role):
    """Skip a role of :data:`BEFORE_DOWN_ROLES` (the router-weight-before-down kernels, top-k 6 and the 384-expert
    scheduler layouts: builds for the targets of ``BEFORE_DOWN_TARGETS``) or of :data:`MODE_GROUP_ROLES` (SM100a, SM103a
    and SM107a) on a device whose target has no build of it; a group role absent from the registry skips too, and
    :func:`test_mode_group_roles_are_registry_records` fails for it.  Every other role is built for every target
    (:func:`test_before_down_roles_are_registry_records`), so a missing build fails at load instead of skipping."""
    require_gpu()
    arch = jit.target_arch()
    if role in MODE_GROUP_ROLES and role not in jit._registry():
        pytest.skip(f"{role} is not in this package's registry")
    if (
        role in BEFORE_DOWN_ROLES or role in MODE_GROUP_ROLES
    ) and arch not in jit._registry()[role]["arches"]:
        pytest.skip(f"{role} has no {arch} build")


def _scheduler_topk(ep, experts):
    """The top-k a layout's scheduler tests route with: 6 on the 384-expert layouts, the group's top-k on the layouts of
    a :data:`MODE_GROUPS` group (10 on the 512-expert layouts), else 2 on the diagnostic layouts and 8."""
    if (ep, experts) in kernels.SCHEDULER_LAYOUTS_384:
        return 6
    for group in kernels.MODE_GROUPS.values():
        if (ep, experts) in group.layouts:
            return group.topk
    return 2 if experts * ep <= 16 else 8


def _check_fused_training(
    *,
    swiglu_limit=None,
    fp32_wgrad=False,
    experts=4,
    topk=2,
    tokens=512,
    geometries=GEOMETRIES,
    empty_cases=(False, True),
    mini=256,
    comm=4,
):
    """Contract and repeatability of the fused forward + backward: every output finite, :data:`REPEATS` eager
    launches and :data:`REPEATS` CUDA Graph replays on the same inputs give the same bytes, a replay gives the eager
    launch's bytes (also after the scores and the inputs change), and zero scores give exactly zero score
    gradients."""
    clamped = swiglu_limit is not None
    forward, backward = MoKForward(clamped), MoKBackward(clamped, fp32_wgrad)
    records = []
    for empty in empty_cases:
        for hidden, intermediate, macro in geometries:
            torch.manual_seed(9173 + hidden)
            options = dict(device="cuda", dtype=torch.bfloat16)
            x = randn(tokens, hidden, **options) * 0.125
            dy = randn_like(x) * 0.125
            shared = [
                randn(intermediate, hidden, **options) / hidden**0.5,
                randn(intermediate, hidden, **options) / hidden**0.5,
                randn(hidden, intermediate, **options) / intermediate**0.5,
            ]
            routed = [
                randn(experts, intermediate, hidden, **options) / hidden**0.5,
                randn(experts, intermediate, hidden, **options) / hidden**0.5,
                randn(experts, hidden, intermediate, **options) / intermediate**0.5,
            ]
            scores = rand(tokens, topk, device="cuda") + 0.1
            scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
            row = torch.arange(tokens, device="cuda")
            routes = torch.stack(
                [(row + slot) % (topk if empty else experts) for slot in range(topk)],
                -1,
            )
            s = single_rank_schedule(routes, experts)
            actual = s["actual"]
            combine = torch.empty(tokens * topk, hidden, **options)
            dx_peer = torch.empty(tokens * topk, hidden, **options)
            ds_peer = torch.empty_like(scores)
            accumulators = initial = None
            if fp32_wgrad:
                initial = [
                    randn(w.shape, device="cuda") * 0.01 for w in (*shared, *routed)
                ]
                accumulators = [value.clone() for value in initial]

            def sequence():
                if fp32_wgrad:
                    for accumulator, start in zip(accumulators, initial, strict=True):
                        accumulator.copy_(start)
                values = forward(
                    x, [x.data_ptr()], combine, [combine.data_ptr()],
                    shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                    s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                    topk, swiglu_limit, comm, macro, mini,
                )  # fmt: skip
                result = backward(
                    dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
                    scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
                    shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                    *values[:7], x, [x.data_ptr()],
                    s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                    topk, swiglu_limit, comm, macro, mini,
                    weight_grad_accumulators=accumulators,
                )  # fmt: skip
                return result, values

            # Shared dx / d_gate / d_up / dh, the six weight gradients, the routed dx and the score gradient, the
            # shared and routed forward outputs.
            def outputs(launch):
                result, values = launch
                every = (*result, dx_peer, ds_peer, values[7], combine)
                picked = tuple(every[index] for index in (0, 2, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
                for value in picked:
                    assert torch.isfinite(value).all().item(), (empty, hidden)
                return tuple(value.clone() for value in picked)

            eager = [outputs(sequence()) for _ in range(REPEATS)]
            assert all(_same(eager[0], other) for other in eager[1:])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = sequence()

            def replay():
                graph.replay()
                return outputs(captured)

            replays = [replay() for _ in range(REPEATS)]
            assert all(_same(replays[0], other) for other in replays[1:])
            assert _same(replays[0], eager[0])  # no bit depends on graph replay versus eager launches
            # The score gradient is fused into SwiGLU backward as dot(d_hidden, hidden) / score; changed positive
            # scores replay the eager bytes, and a zero score returns an exactly zero score gradient.
            scores.add_(0.03125)
            assert _same(replay(), outputs(sequence()))
            scores.zero_()
            replay()
            assert torch.count_nonzero(ds_peer).item() == 0
            for _ in range(2):
                normal_(x, std=0.125)
                normal_(dy, std=0.125)
                uniform_(scores, 0.1, 1.0)
                scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
                assert _same(replay(), outputs(sequence()))
            records.append(dict(empty=empty, hidden=hidden, macro=macro, routed=actual))
    return records


@pytest.fixture(autouse=True)
def _exact_matmul(cuda_matmul_flags):
    cuda_matmul_flags(allow_tf32=False, allow_bf16_reduced_precision_reduction=False)


def test_fused_training():
    require_gpu()
    assert len(_check_fused_training()) == 6


def test_fused_training_clamped_swiglu():
    require_gpu()
    # Activations have standard deviation near 0.125, so this limit clamps a
    # substantial fraction of both gate and up values.
    _check_fused_training(
        swiglu_limit=0.125, geometries=GEOMETRIES[:1], empty_cases=(False,)
    )


@pytest.mark.parametrize("geometries", [GEOMETRIES[:1], GEOMETRIES], ids=["one", "all"])
def test_fused_training_fp32_wgrad(geometries):
    require_gpu()
    _check_fused_training(fp32_wgrad=True, geometries=geometries, empty_cases=(False,))


@pytest.mark.parametrize("experts", [36, 9], ids=["ep8-local36", "ep32-local9"])
def test_fused_training_local_experts(experts):
    """288 routed experts at EP8 / EP32 (36 or 9 local experts), top-8, clamped."""
    require_gpu()
    _check_fused_training(
        swiglu_limit=0.125,
        experts=experts,
        topk=8,
        tokens=512,
        geometries=((512, 256, 2048),),
        empty_cases=(False,),
    )


@pytest.mark.parametrize("swiglu_limit", [None, 0.125], ids=["plain", "clamped"])
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_recompute_context(swiglu_limit, macro):
    """Context-only recompute equals the saved forward context; backward agrees bitwise."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, recompute = MoKForward(clamped), MoKRecompute(clamped)
    backward = MoKBackward(clamped)
    tokens, hidden, intermediate, experts, topk, mini, comm = (
        512,
        512,
        256,
        4,
        2,
        256,
        4,
    )
    torch.manual_seed(4111)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = randn(tokens, hidden, **options) * 0.125
    dy = randn_like(x) * 0.125
    shared = [
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = rand(tokens, topk, device="cuda") + 0.1
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    combine = torch.zeros(tokens * topk, hidden, **options)
    saved = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()],
        shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rebuilt = recompute(
        x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rows = min(s["actual"], macro)
    for index in (0, 2, 4, 6):  # routed x, gate, up, hidden of the resident macrobatch
        assert torch.equal(saved[index][:rows], rebuilt[index][:rows]), index
    for index in (1, 3, 5):  # shared gate, up, hidden
        assert torch.equal(saved[index], rebuilt[index]), index

    def gradients(context):
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        values = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        return (dx_peer, ds_peer, values[0], *values[9:])

    for a, b in zip(gradients(saved), gradients(rebuilt), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("swiglu_limit", [None, 0.125], ids=["plain", "clamped"])
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_context_free_forward(swiglu_limit, macro):
    """The context-free forward writes the context forward's y, combine rows,
    hidden and routed x bitwise, in every launch, without gate/up storage; the
    context recomputed after it gives backward the saved context's bits."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, context_free = MoKForward(clamped), MoKForward(clamped, context=False)
    recompute, backward = MoKRecompute(clamped), MoKBackward(clamped)
    tokens, hidden, intermediate, experts, topk, mini, comm = (
        512,
        512,
        256,
        4,
        2,
        256,
        4,
    )
    torch.manual_seed(4127)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = randn(tokens, hidden, **options) * 0.125
    dy = randn_like(x) * 0.125
    shared = [
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = rand(tokens, topk, device="cuda") + 0.1
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    rows = min(s["actual"], macro)

    def run(kernel):
        combine = torch.zeros(tokens * topk, hidden, **options)
        values = kernel(
            x, [x.data_ptr()], combine, [combine.data_ptr()], *weights,
            *schedule, topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        torch.cuda.synchronize()
        return values, combine

    saved, combine = run(forward)
    assert torch.count_nonzero(combine).item() > 0
    assert torch.count_nonzero(saved[7]).item() > 0
    for _ in range(3):
        values, free_combine = run(context_free)
        assert values[1:5] == (None,) * 4
        assert torch.equal(free_combine, combine)
        assert torch.equal(values[7], saved[7])  # y_shared
        assert torch.equal(values[5], saved[5])  # hidden_shared
        for index in (0, 6):  # routed x, hidden of the resident macrobatch
            assert torch.equal(values[index][:rows], saved[index][:rows]), index
    rebuilt = recompute(
        x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip

    def gradients(context):
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        values = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            *weights, *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        return (dx_peer, ds_peer, values[0], *values[9:])

    expected = gradients(saved)
    assert all(torch.count_nonzero(value).item() > 0 for value in expected)
    for a, b in zip(expected, gradients(rebuilt), strict=True):
        assert torch.equal(a, b)


def test_context_free_forward_arguments_are_checked_before_device_work(monkeypatch):
    """``save_context`` and ``context`` must be booleans; MXFP8 routed experts
    have no context-free forward. Each refusal comes before any copy, barrier
    or launch (host only)."""
    with pytest.raises(TypeError, match="context must be a boolean"):
        MoKForward(False, context=1)
    functional = object.__new__(backend.MoKFunctional)
    functional.local_experts = 4
    functional._validated = {}  # the weight-validation memo __init__ creates
    events = []
    monkeypatch.setattr(
        backend.MoKFunctional, "_inputs", staticmethod(lambda *args: (None, 256))
    )
    monkeypatch.setattr(functional, "_check_device", lambda workspace: None)
    monkeypatch.setattr(functional, "_barrier", lambda w: events.append("barrier"))
    monkeypatch.setattr(functional, "_kernels", lambda *a: events.append("kernels"))
    monkeypatch.setattr(
        backend.MoKFunctional,
        "_copy_source",
        staticmethod(lambda *args: events.append("copy")),
    )
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(256, 256, **options)
    shared = [torch.zeros(256, 256, **options) for _ in range(3)]
    data = torch.zeros(4, 256, 256, dtype=torch.uint8)
    mxfp8 = [(data, torch.zeros(64, dtype=torch.uint8)) for _ in range(3)]
    with pytest.raises(TypeError, match="save_context must be a boolean"):
        functional.forward(None, None, None, x, None, *shared, *mxfp8, save_context=0)
    with pytest.raises(ValueError, match="save_context=False requires BF16"):
        functional.forward(
            None, None, None, x, None, *shared, *mxfp8, save_context=False
        )
    with pytest.raises(TypeError, match="save_context must be a boolean"):
        functional.prepare(None, save_context="no")
    with pytest.raises(ValueError, match="save_context=False requires BF16"):
        functional.prepare(None, True, save_context=False)
    assert events == ["kernels"]  # prepare compiles the MXFP8 pair, then refuses


@pytest.mark.parametrize("ep,experts", SCHEDULER_LAYOUTS)
def test_scheduler(ep, experts):
    require_gpu()
    _require_role(f"count_{ep}_{experts}")
    topk = _scheduler_topk(ep, experts)
    scheduler = MoKScheduler(ep, experts)
    tokens = 512
    generator = torch.Generator(device="cuda").manual_seed(1107)
    ids = (
        rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    # A masked tail (expert -1) models a rank with fewer real source rows.
    ids[0, tokens - 37 :] = -1
    capacity = ep * tokens * topk + experts * 256
    owner_loads = []
    for rank in {0, ep - 1}:
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(
            ids.flatten().long().clamp(min=0), minlength=ep * experts
        )
        expected_counts[0] -= (ids == -1).sum()
        expected_counts = expected_counts[rank * experts : (rank + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        torch.testing.assert_close(counts, expected_counts.int(), rtol=0, atol=0)
        assert counts.shape == (experts,) and total.shape == (1,)
        assert total.item() == expected_counts.sum().item()
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == rank * experts + expert).nonzero().flatten()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            offset += count
        assert (peers[offset:] == -1).all()
        # The count kernel also bins every valid gathered id by owning rank
        # (the tail of its counts buffer): exactly the host reference count.
        owner = scheduler.owner_loads()
        assert owner.dtype == torch.int32 and owner.shape == (ep,)
        assert torch.equal(owner.long(), backend.count_routed_load(ids, ep, experts))
        owner_loads.append(owner.clone())
    assert all(torch.equal(owner_loads[0], other) for other in owner_loads[1:])
    scheduler.release_counts()
    with pytest.raises(RuntimeError, match="scheduler call"):
        scheduler.owner_loads()


@pytest.mark.parametrize("ep,experts", SCHEDULER_LAYOUTS)
def test_planned_scheduler_matches_the_unplanned_launches(ep, experts):
    """The scheduler's plan (one int32 buffer per schedule whose sentinel and counters one template copy writes) gives
    the unplanned launches' schedule bit for bit, twice from one plan: peer ranks, the peer token of every real row,
    the row count, the padded counts, the owner-rank bins and the written layout prefix.  Every view lies in one
    allocation, the counters keep the zero slab's alignment and the layout starts 512-B aligned."""
    require_gpu()
    _require_role(f"count_{ep}_{experts}")
    topk = _scheduler_topk(ep, experts)
    scheduler = MoKScheduler(ep, experts)
    tokens = 512
    generator = torch.Generator(device="cuda").manual_seed(2211)
    ids = (
        rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    ids[0, tokens - 37 :] = -1
    capacity = ep * tokens * topk + experts * 256
    for rank in sorted({0, ep - 1}):
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        owner = scheduler.owner_loads().clone()
        layout = scheduler.layout(counts, capacity, peers)
        scheduler.release_counts()
        written = 3 * experts + int(total.item()) // 256
        real = peers >= 0
        plan = scheduler.plan(ids, capacity, rank)
        for _ in range(2):
            with kernels._ffi_stream():
                planned = scheduler.launch_planned(plan)
                planned_owner = scheduler.owner_loads().clone()
                scheduler.layout_planned(plan, planned[3], planned[0], planned[4])
            scheduler.release_counts()
            p_peers, p_slots, p_total, p_counts, p_layout = planned
            assert torch.equal(p_peers, peers) and torch.equal(p_total, total)
            assert torch.equal(p_counts, counts) and torch.equal(planned_owner, owner)
            assert torch.equal(p_slots[real], slots[real])
            assert torch.equal(p_layout[:written], layout[:written])
            base = p_peers.untyped_storage().data_ptr()
            assert all(view.untyped_storage().data_ptr() == base for view in planned)
            assert p_total.data_ptr() % 128 == 0 and p_counts.data_ptr() == p_total.data_ptr() + 4
            assert p_layout.data_ptr() % 512 == 0 and p_layout.numel() == 3 * experts + capacity // 256


def _schedule_fields(schedule, experts):
    """The defined fields of a schedule: peer ranks, the peer token of every real row, the row count, the padded
    counts and the written layout prefix."""
    written = 3 * experts + int(schedule.num_tokens.item()) // 256
    real = schedule.peer_rank >= 0
    return (
        schedule.peer_rank,
        schedule.peer_token_idx[real],
        schedule.num_tokens,
        schedule.tokens_per_expert,
        schedule.layout[:written],
    )


@pytest.mark.parametrize("count", [1024, 777], ids=["full", "partial"])
def test_build_schedule_plan_matches_the_unplanned_launches(count):
    """``build_schedule`` through the workspace's schedule plan gives the unplanned launches' schedule bit for bit on
    full and partial source counts (one-GPU emulation of rank 0 of EP 4, :class:`OneGpuGroup`), and builds the plan
    once per workspace and adapter.  A workspace's first ``build_schedule`` during CUDA Graph capture runs the
    unplanned launches and builds no plan; its replay gives the same schedule."""
    require_gpu()
    ep, experts, topk, hidden = 4, 4, 2, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(ep, experts, topk)
    functional.communication = OneGpuGroup()
    config = MoKConfig(fwd_num_comm_sms=4, bwd_num_comm_sms=4)
    generator = torch.Generator(device=device).manual_seed(3307 + count)
    ids = (
        rand(count, ep * experts, device=device, generator=generator)
        .argsort(-1)[:, :topk]
        .contiguous()
    )

    def fields(schedule):
        return _schedule_fields(schedule, experts)

    workspace, _ = one_gpu_source_workspace(ep, experts, (count,), hidden, topk, device)
    key = ("schedule", functional.scheduler, functional.communication)
    planned = [
        fields(functional.build_schedule(workspace, config, ids, num_local_experts=experts))
        for _ in range(2)
    ]
    plan = workspace.storage.launch_plans[key]
    assert len([k for k in workspace.storage.launch_plans if k[0] == "schedule"]) == 1
    rank, token, total, counts, layout, _ = functional._schedule_launches(
        workspace.storage, config, ids, count
    )
    unplanned = fields(
        workspace_module.MoKSchedule(
            peer_rank=rank, peer_token_idx=token, num_tokens=total, tokens_per_expert=counts, layout=layout
        )
    )
    assert _same(planned[0], unplanned) and _same(planned[1], unplanned)
    assert workspace.storage.launch_plans[key] is plan

    fresh, _ = one_gpu_source_workspace(ep, experts, (count,), hidden, topk, device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = functional.build_schedule(fresh, config, ids, num_local_experts=experts)
    assert key not in fresh.storage.launch_plans
    graph.replay()
    torch.cuda.synchronize()
    assert _same(fields(captured), unplanned)


def test_build_schedule_plan_built_under_inference_mode():
    """A workspace's first ``build_schedule`` under ``torch.inference_mode()`` builds the schedule plan with normal
    tensors, so the next ``build_schedule`` outside inference mode can overwrite the plan's route buffer, and a third
    one inside inference mode uses the same plan; all three give the unplanned launches' schedule bit for bit
    (one-GPU emulation of rank 0 of EP 4, :class:`OneGpuGroup`)."""
    require_gpu()
    ep, experts, topk, hidden, count = 4, 4, 2, 256, 1024
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(ep, experts, topk)
    functional.communication = OneGpuGroup()
    config = MoKConfig(fwd_num_comm_sms=4, bwd_num_comm_sms=4)
    generator = torch.Generator(device=device).manual_seed(3391)
    ids = (
        rand(count, ep * experts, device=device, generator=generator)
        .argsort(-1)[:, :topk]
        .contiguous()
    )
    workspace, _ = one_gpu_source_workspace(ep, experts, (count,), hidden, topk, device)
    key = ("schedule", functional.scheduler, functional.communication)
    with torch.inference_mode():
        first = _schedule_fields(
            functional.build_schedule(workspace, config, ids, num_local_experts=experts), experts
        )
    plan = workspace.storage.launch_plans[key]
    assert not plan.routes.is_inference() and not plan.scheduler.template.is_inference()
    second = _schedule_fields(
        functional.build_schedule(workspace, config, ids, num_local_experts=experts), experts
    )
    with torch.inference_mode():
        third = _schedule_fields(
            functional.build_schedule(workspace, config, ids, num_local_experts=experts), experts
        )
    assert workspace.storage.launch_plans[key] is plan
    rank, token, total, counts, layout, _ = functional._schedule_launches(
        workspace.storage, config, ids, count
    )
    unplanned = _schedule_fields(
        workspace_module.MoKSchedule(
            peer_rank=rank, peer_token_idx=token, num_tokens=total, tokens_per_expert=counts, layout=layout
        ),
        experts,
    )
    assert _same(first, unplanned) and _same(second, unplanned) and _same(third, unplanned)


#: int64 expert ids outside the int32 range and at its edges: ``gather_i64`` keeps their low 32 bits, as the route
#: buffer's int64 -> int32 copy does.
FAR_IDS = (2**32 + 5, -(2**33) + 7, 2**31, -(2**31) - 1, 2**31 - 1, -(2**31), 2**40 - 1, -(2**40), 2**63 - 1, -(2**63))


@pytest.mark.parametrize("count", [512, 333, 0], ids=["full", "partial", "empty"])
def test_gather_i64_plan_matches_the_route_buffer_gather(count):
    """The ``gather_i64`` plan stores the words of the route buffer's int64 -> int32 copy (and ``-1`` fill past the
    source count) followed by the int32 gather plan, bit for bit: ids outside the int32 range included, a partial and
    an empty source count, into the gathered slice of its rank only (one GPU: a unicast address in place of the
    multicast alias, at ranks 0 and EP - 1). EP 1 has no plan (its gather is a copy)."""
    _require_role("gather_i64")
    ep, rows, topk = 4, 512, 6
    communication = kernels.MoKCommunication()
    generator = torch.Generator(device="cuda").manual_seed(4421 + count)
    ids = randint(-(2**40), 2**40, (count, topk), device="cuda", generator=generator)
    far = torch.tensor(FAR_IDS, device="cuda", dtype=torch.int64)
    n = min(far.numel(), ids.numel())
    ids.view(-1)[torch.arange(n, device="cuda") * max(1, ids.numel() // max(1, n))] = far[:n]
    routes = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
    routes[:count].copy_(ids)
    for rank in (0, ep - 1):
        expected = torch.full((ep, rows, topk), 0x5A5A, dtype=torch.int32, device="cuda")
        base = communication.gather_plan(routes, expected, expected.data_ptr(), rank)
        gathered = torch.full_like(expected, 0x5A5A)
        plan = communication.gather_i64_plan(gathered, gathered.data_ptr(), rank)
        for _ in range(2):
            with kernels._ffi_stream():
                base.launch(())
                plan.launch((ids, count * topk))
            torch.cuda.synchronize()
            assert torch.equal(gathered, expected), rank
            assert torch.equal(gathered[rank], routes)
    single = torch.zeros((1, rows, topk), dtype=torch.int32, device="cuda")
    assert communication.gather_i64_plan(single, single.data_ptr(), 0) is None


#: Route patterns of :func:`test_fused_scheduler_plan_matches_the_unplanned_launches` (the geometries of the
#: ``count_fused`` kernel tests): a uniform top-6 router with a masked tail and below -1 and out-of-range ids, a hot
#: rank at 1.5 and 3 times its share, ragged source ranks with an empty one, top-k 1 routes all masked, all to one
#: expert, in multiples of 256 rows per expert and one row per expert, and 256 x 6 = 1536 slots per source rank (not a
#: multiple of the kernels' 1024 threads).
FUSED_ROUTE_PATTERNS = (
    "uniform",
    "hot1.5",
    "hot3",
    "ragged",
    "all_masked",
    "one_expert",
    "multiple_256",
    "one_row",
    "odd_slots",
)
#: The real rows of each source rank of the ``ragged`` pattern (512 rows per source rank), repeated over the ranks.
FUSED_RAGGED = (0, 512, 37, 300, 255, 1, 511, 256)


def _fused_routes(pattern, ep, experts, rank):
    """The gathered routes (contiguous int32 ``[EP, tokens, top-k]``) of a :data:`FUSED_ROUTE_PATTERNS` entry, as seen
    by ``rank``, from a fixed seed."""
    total, own = ep * experts, rank * experts
    if pattern in ("all_masked", "one_expert", "multiple_256", "one_row"):
        # Top-k 1 route patterns: every route masked; every route to this rank's local expert 1; 256-row token blocks
        # to one local expert each (block b of every source rank to local expert b % local experts); local expert
        # e's one route from token e of source rank 0, every other route masked.
        tokens = 256 * experts if pattern == "multiple_256" else 256
        ids = torch.full((ep, tokens, 1), -1, dtype=torch.int32, device="cuda")
        if pattern == "one_expert":
            ids.fill_(own + 1)
        elif pattern == "multiple_256":
            blocks = torch.arange(tokens, device="cuda", dtype=torch.int32) // 256 % experts
            ids.copy_((own + blocks).view(1, tokens, 1).expand_as(ids))
        elif pattern == "one_row":
            ids[0, :experts, 0] = own + torch.arange(experts, device="cuda", dtype=torch.int32)
        return ids
    topk = _scheduler_topk(ep, experts)
    tokens = 256 if pattern == "odd_slots" else 512
    generator = torch.Generator(device="cuda").manual_seed(2297)
    keys = rand(ep * tokens, total, device="cuda", generator=generator)
    if pattern.startswith("hot"):
        # Draws without replacement weighted by expert (the top-k largest u ** (1 / weight)): this rank's experts
        # weigh 1.5 or 3 times the others', so the rank receives about that many times its share of the routes.
        weight = torch.ones(total, device="cuda")
        weight[own : own + experts] = float(pattern[3:])
        ids = (keys ** (1 / weight)).argsort(-1, descending=True)[:, :topk]
    else:
        ids = keys.argsort(-1)[:, :topk]
    ids = ids.int().reshape(ep, tokens, topk).contiguous()
    if pattern == "uniform":
        ids[0, tokens - 37 :] = -1
        ids[ep - 1, :8, 0] = torch.tensor(
            [-2, -1000, total, total + experts - 1, -3, total + 7, -1, total + 1], device="cuda", dtype=torch.int32
        )
    elif pattern == "ragged":
        for peer in range(ep):
            ids[peer, FUSED_RAGGED[peer % len(FUSED_RAGGED)] :] = -1
    return ids


@pytest.mark.parametrize("pattern", FUSED_ROUTE_PATTERNS)
@pytest.mark.parametrize("ep,experts", kernels.SCHEDULER_LAYOUTS_384)
def test_fused_scheduler_plan_matches_the_unplanned_launches(ep, experts, pattern):
    """On the 384-expert layouts the scheduler's fused plan (``count_fused`` then rows, no template copy, no pad or
    layout launch) gives the unplanned launches' schedule bit for bit, twice from one plan: peer ranks on every row
    (the ``-1`` sentinel included), the peer token of every real row, the row count, the padded counts, the owner-rank
    bins and the written layout prefix, on every route pattern of :data:`FUSED_ROUTE_PATTERNS` (masked, below -1 and
    out-of-range ids included) at ranks 0, EP / 2 and EP - 1. Its ``done`` word is 0 after every launch, and its views
    sit where the unfused plan's do."""
    _require_role(f"count_fused_{ep}_{experts}")
    scheduler = MoKScheduler(ep, experts)
    assert scheduler.count_fused is not None
    for rank in sorted({0, ep // 2, ep - 1}):
        ids = _fused_routes(pattern, ep, experts, rank)
        _, tokens, topk = ids.shape
        capacity = ep * tokens * topk + experts * 256
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        owner = scheduler.owner_loads().clone()
        layout = scheduler.layout(counts, capacity, peers)
        scheduler.release_counts()
        written = 3 * experts + int(total.item()) // 256
        real = peers >= 0
        plan = scheduler.plan(ids, capacity, rank, fused=True)
        unfused = scheduler.plan(ids, capacity, rank)
        assert plan.template is None and plan.count is None and plan.pad is None and plan.layout is None
        assert (plan.slab, plan.layout_range, plan.total) == (unfused.slab, unfused.layout_range, unfused.total)
        for _ in range(2):
            with kernels._ffi_stream():
                planned = scheduler.launch_planned(plan)
                planned_owner = scheduler.owner_loads().clone()
            scheduler.release_counts()
            p_peers, p_slots, p_total, p_counts, p_layout = planned
            assert torch.equal(p_peers, peers) and torch.equal(p_total, total)
            assert torch.equal(p_counts, counts) and torch.equal(planned_owner, owner)
            assert torch.equal(p_slots[real], slots[real])
            assert torch.equal(p_layout[:written], layout[:written])
            assert plan.done.item() == 0
            base = p_peers.untyped_storage().data_ptr()
            assert all(view.untyped_storage().data_ptr() == base for view in planned)
            assert p_total.data_ptr() % 128 == 0 and p_counts.data_ptr() == p_total.data_ptr() + 4
            assert p_layout.data_ptr() % 512 == 0 and p_layout.numel() == 3 * experts + capacity // 256


class _OneGpuGroupI64(OneGpuGroup):
    """:class:`OneGpuGroup` with the package's ``gather_i64`` plan: the kernel stores this rank's gathered slice through
    a unicast address (one GPU), then every other slice copies it (every rank's ids are this rank's, as
    :class:`OneGpuGroup` gathers them)."""

    def __init__(self):
        self.communication = kernels.MoKCommunication()

    def gather_i64_plan(self, gathered, multicast_address, rank):
        plan = self.communication.gather_i64_plan(gathered, gathered.data_ptr(), rank)

        class Launch:
            @staticmethod
            def launch(values):
                plan.launch(values)
                own = gathered[rank : rank + 1]
                gathered[:rank].copy_(own.expand_as(gathered[:rank]))
                gathered[rank + 1 :].copy_(own.expand_as(gathered[rank + 1 :]))

        return Launch()


def _schedule_words(schedule, experts):
    """:func:`_schedule_fields` and the owner-rank route totals (``None`` with the source split off)."""
    return (*_schedule_fields(schedule, experts), schedule.owner_routes)


def _same_words(a, b):
    """Each pair of words equal, or both ``None``."""
    return all(
        (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
        for x, y in zip(a, b, strict=True)
    )


@pytest.mark.parametrize("split", [False, True], ids=["split-off", "split-on"])
@pytest.mark.parametrize("count", [1024, 777], ids=["full", "partial"])
@pytest.mark.parametrize("ep,experts", kernels.SCHEDULER_LAYOUTS_384)
def test_build_schedule_fused_plan_matches_the_unplanned_launches(ep, experts, count, split):
    """``build_schedule`` of a 384-expert layout through the workspace's fused schedule plan (``gather_i64`` on the
    caller's int64 ids, the barrier, ``count_fused`` and rows) gives the unplanned launches' schedule bit for bit,
    ``owner_routes`` included, with the source split on and off, on full and partial source counts (one-GPU emulation
    of rank 0, :class:`_OneGpuGroupI64`): the first build under ``torch.inference_mode()`` (the plan's tensors stay
    normal), the next outside it and a third inside it, and two replays of a CUDA Graph that captured a fused build. A
    fresh workspace's first build during capture runs the unplanned launches and builds no plan; its replay gives the
    same schedule."""
    _require_role(f"count_fused_{ep}_{experts}")
    _require_role("gather_i64")
    topk, hidden = 6, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep, experts, topk, clamped_swiglu=True, fp32_wgrad=True, router_weight_position="before_down"
    )
    assert functional.scheduler.count_fused is not None
    functional.communication = _OneGpuGroupI64()
    config = MoKConfig(fwd_num_comm_sms=4, bwd_num_comm_sms=4, source_split_comm_sms=4 if split else None)
    generator = torch.Generator(device=device).manual_seed(5501 + count + ep)
    ids = rand(count, ep * experts, device=device, generator=generator).argsort(-1)[:, :topk].contiguous()
    ids[: count // 7, topk - 1] = -1  # masked routes
    workspace, _ = one_gpu_source_workspace(ep, experts, (count,), hidden, topk, device)
    key = ("schedule", functional.scheduler, functional.communication)

    def build(ws):
        return _schedule_words(functional.build_schedule(ws, config, ids, num_local_experts=experts), experts)

    with torch.inference_mode():
        first = build(workspace)
    plan = workspace.storage.launch_plans[key]
    assert plan.fused and plan.routes is None
    assert not plan.scheduler.partials.is_inference() and not plan.scheduler.done.is_inference()
    second = build(workspace)
    with torch.inference_mode():
        third = build(workspace)
    assert workspace.storage.launch_plans[key] is plan
    rank, token, total, counts, layout, owner = functional._schedule_launches(
        workspace.storage, config, ids, count, split
    )
    unplanned = _schedule_words(
        workspace_module.MoKSchedule(
            peer_rank=rank,
            peer_token_idx=token,
            num_tokens=total,
            tokens_per_expert=counts,
            layout=layout,
            owner_routes=owner,
        ),
        experts,
    )
    assert (owner is not None) == split
    assert _same_words(first, unplanned) and _same_words(second, unplanned) and _same_words(third, unplanned)
    assert plan.scheduler.done.item() == 0

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert _same_words(_schedule_words(captured, experts), unplanned)
        assert plan.scheduler.done.item() == 0

    fresh, _ = one_gpu_source_workspace(ep, experts, (count,), hidden, topk, device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = functional.build_schedule(fresh, config, ids, num_local_experts=experts)
    assert key not in fresh.storage.launch_plans
    graph.replay()
    torch.cuda.synchronize()
    assert _same_words(_schedule_words(captured, experts), unplanned)


@pytest.mark.parametrize("ep,experts", [(4, 4), (8, 32)])
def test_scheduler_owner_loads_match_count_routed_load_with_invalid_ids(ep, experts):
    """The owner-rank bins drop padding (-1), more negative ids and ids at or
    past the limit exactly as ``count_routed_load`` does, on device and on the
    CPU reference."""
    require_gpu()
    topk = 2 if ep * experts <= 16 else 8
    limit = ep * experts
    generator = torch.Generator(device="cuda").manual_seed(2203)
    ids = randint(
        -1, limit + 8, (ep, 256, topk), device="cuda", dtype=torch.int32,
        generator=generator,
    )  # fmt: skip
    edges = torch.tensor(
        [-2, -1000, limit, limit + experts - 1], device="cuda", dtype=torch.int32
    )
    ids[1, :16, 0] = edges.repeat(4)
    scheduler = MoKScheduler(ep, experts)
    scheduler(ids, ep * 256 * topk + experts * 256, 0)
    owner = scheduler.owner_loads()
    expected = backend.count_routed_load(ids, ep, experts)
    assert owner.dtype == torch.int32 and owner.shape == (ep,)
    assert torch.equal(owner.long(), expected)
    assert torch.equal(
        expected.cpu(), backend.count_routed_load(ids.cpu(), ep, experts)
    )


@pytest.mark.parametrize(
    "ep,experts,tokens,rank,real",
    [
        (8, 32, 16384, 5, (8191, 16172, 16231, 0, 8191, 16172, 16231, 0)),
        (4, 64, 32768, 3, None),
    ],
    ids=["ep8-local32-ragged", "ep4-local64-full"],
)
def test_scheduler_at_training_geometries(ep, experts, tokens, rank, real):
    """The scheduler assertions of ``test_scheduler`` and the owner-rank bins at
    full training geometries: 256 routed experts, top-8, 16384 or 32768 source
    rows per rank, ragged ranks (empty ones included) or full ranks."""
    require_gpu()
    topk = 8
    generator = torch.Generator(device="cuda").manual_seed(5309 + ep)
    ids = (
        rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    if real is not None:
        for source, rows in enumerate(real):
            ids[source, rows:] = -1
    capacity = tokens * topk * 2
    scheduler = MoKScheduler(ep, experts)
    owner_loads = []
    for r in (0, rank):
        peers, slots, total, counts = scheduler(ids, capacity, r)
        expected_counts = torch.bincount(
            ids.flatten().long().clamp(min=0), minlength=ep * experts
        )
        expected_counts[0] -= (ids == -1).sum()
        expected_counts = expected_counts[r * experts : (r + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        torch.testing.assert_close(counts, expected_counts.int(), rtol=0, atol=0)
        assert total.item() == expected_counts.sum().item() <= capacity
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == r * experts + expert).nonzero().flatten()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            offset += count
        assert (peers[offset:] == -1).all()
        owner = scheduler.owner_loads()
        assert torch.equal(owner.long(), backend.count_routed_load(ids, ep, experts))
        owner_loads.append(owner.clone())
    assert torch.equal(owner_loads[0], owner_loads[1])
    if real is not None:
        assert owner_loads[0].sum().item() == sum(real) * topk


@pytest.mark.parametrize("topk", [2, 6, 8])
@pytest.mark.parametrize("tokens,hidden", [(512, 256), (37, 6144), (1, 1032)])
def test_epilogues(topk, tokens, hidden):
    """The weighted forward epilogue (top-k 2 and 8) is finite and repeatable; the unweighted sum (every top-k, the
    router-weight-before-down forward's too) is exact and repeatable."""
    require_gpu()
    _require_role(f"epilogue_backward_{topk}")
    epilogues = MoKEpilogues(topk)
    shared = randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
    routed = randn(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
    scores = rand(tokens, topk, device="cuda")
    # Native order: FP32 accumulation from the shared row, then ascending slots.
    expected_dx = shared.float()
    for k in range(topk):
        expected_dx = expected_dx + routed.float().view(tokens, topk, hidden)[:, k]
    # The weighted sum may contract multiply-adds (its accuracy is judged in test_mok_accuracy.py): finite and
    # repeatable.  The unweighted sum is exact.
    if topk in kernels.WEIGHTED_EPILOGUE_TOPKS:
        y = [epilogues.forward(shared, routed, scores) for _ in range(REPEATS)]
        assert torch.isfinite(y[0]).all().item()
        assert all(torch.equal(y[0], other) for other in y[1:])
    else:
        with pytest.raises(ValueError, match="no weighted form"):
            epilogues.forward(shared, routed, scores)
    sums = [epilogues.backward(shared, routed) for _ in range(REPEATS)]
    assert all(torch.equal(value, expected_dx.bfloat16()) for value in sums)


def test_count_routed_load_matches_a_reference_count():
    """Host-only: the per-rank load histogram counts the valid gathered ids by
    owning rank and ignores padding (-1) and out-of-range ids."""
    count_routed_load = backend.count_routed_load

    ep_size, local_experts = 4, 3
    routes = torch.tensor(
        [[0, 5, 11, -1], [3, 3, 12, -7], [8, 9, 10, 11], [-1, -1, -1, 2]],
        dtype=torch.int32,
    )
    expected = [0] * ep_size
    for e in routes.reshape(-1).tolist():
        if 0 <= e < ep_size * local_experts:
            expected[e // local_experts] += 1
    loads = count_routed_load(routes, ep_size, local_experts)
    assert loads.dtype == torch.int64
    assert loads.tolist() == expected == [2, 3, 1, 4]
    empty = torch.full((6,), -1, dtype=torch.int32)
    assert count_routed_load(empty, ep_size, local_experts).tolist() == [0] * ep_size
    # Ids exactly at the limit and one owner beyond it (the last histogram bin
    # and the first value outside the range), more negative padding, and a large
    # random gather per rank count: all match the reference count exactly.
    limit = ep_size * local_experts
    edges = torch.tensor(
        [limit, limit + local_experts - 1, limit + local_experts, -2, -1000, 0],
        dtype=torch.int32,
    )
    assert count_routed_load(edges, ep_size, local_experts).tolist() == [1, 0, 0, 0]
    generator = torch.Generator().manual_seed(0)
    ep_size, local_experts = 64, 4
    random_routes = randint(
        -1, ep_size * local_experts + 8, (ep_size, 2048, 8), generator=generator
    ).to(torch.int32)
    reference = torch.zeros(ep_size, dtype=torch.int64)
    valid = (random_routes >= 0) & (random_routes < ep_size * local_experts)
    reference.index_add_(
        0,
        torch.div(random_routes[valid], local_experts, rounding_mode="floor").long(),
        torch.ones(int(valid.sum()), dtype=torch.int64),
    )
    assert torch.equal(
        count_routed_load(random_routes, ep_size, local_experts), reference
    )


class _AtenRecorder:
    """Names of the aten ops dispatched inside the block (``TorchDispatchMode``)."""

    def __init__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        recorder = self

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                recorder.ops.append(func.overloadpacket.__name__)
                return func(*args, **(kwargs or {}))

        self.ops = []
        self._mode = Mode()

    def __enter__(self):
        self._mode.__enter__()
        return self

    def __exit__(self, *args):
        self._mode.__exit__(*args)


def test_observe_load_issues_one_copy_and_no_histogram_ops():
    """Once the workspace's load entry exists, an observation copies the count
    kernel's owner-rank tail into the pinned int32 slot through one view and
    one copy: no histogram, cast, division, fill or allocation is dispatched."""
    require_gpu()
    from collections import Counter
    from types import SimpleNamespace

    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    config = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        imbalance_comm_sms=(6, 6),
        imbalance_threshold=1.5,
    )
    routes = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32)
        % (ep_size * experts)
    ).view(ep_size, 256, 2)
    workspace = SimpleNamespace(ep_rank=0, ep_size=ep_size)
    functional.scheduler(routes, ep_size * 256 * 2 + experts * 256, 0)
    # The first observation of a workspace creates its entry (two pinned
    # slots, two ``empty`` calls); every later observation is the steady state.
    functional._observe_load(workspace, config)
    with _AtenRecorder() as recorder:
        functional._observe_load(workspace, config)
    ops = Counter(recorder.ops)
    assert ops["copy_"] == 1, ops
    assert sum(ops[name] for name in ("slice", "narrow", "select")) <= 1, ops
    forbidden = {
        "histc",
        "div",
        "floor_divide",
        "_to_copy",
        "to",
        "fill_",
        "zeros",
        "empty",
    }
    assert not (set(ops) & forbidden), ops
    entry = functional._load_entry(workspace, create=False)
    host = entry["host"][entry["pending"]]
    assert host.dtype == torch.int32 and host.is_pinned() and host.numel() == ep_size
    torch.cuda.synchronize()
    assert host.tolist() == backend.count_routed_load(routes, ep_size, experts).tolist()


def test_observe_load_keeps_a_completed_observation():
    """A host running ahead of the device schedules the next step before any
    launch of the previous one saw its observation complete; the next
    observation adopts the completed one before replacing it, so the split
    still follows the routed load."""
    require_gpu()
    from types import SimpleNamespace

    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    config = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        imbalance_comm_sms=(6, 6),
        imbalance_threshold=1.5,
    )
    skewed = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32) % experts
    ).view(ep_size, 256, 2)
    capacity = ep_size * 256 * 2 + experts * 256
    workspace = SimpleNamespace(ep_rank=0, ep_size=ep_size)
    functional.scheduler(skewed, capacity, 0)
    functional._observe_load(workspace, config)
    torch.cuda.synchronize()
    functional.scheduler(skewed, capacity, 0)
    functional._observe_load(workspace, config)  # no launch in between
    entry = functional._load_entry(workspace, create=False)
    expected = backend.count_routed_load(skewed, ep_size, experts).tolist()
    assert list(entry["loads"]) == expected
    assert functional._comm_split(config, workspace) == (6, 6)


def test_observe_load_first_observation_under_inference_mode():
    """A workspace's first observation under ``torch.inference_mode()`` creates
    its pinned slots as normal tensors, so the next observation outside
    inference mode copies into them in place, a third one inside it does too,
    and the adopted loads follow the routed load."""
    require_gpu()
    from types import SimpleNamespace

    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    config = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        imbalance_comm_sms=(6, 6),
        imbalance_threshold=1.5,
    )
    routes = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32)
        % (ep_size * experts)
    ).view(ep_size, 256, 2)
    capacity = ep_size * 256 * 2 + experts * 256
    workspace = SimpleNamespace(ep_rank=0, ep_size=ep_size)
    functional.scheduler(routes, capacity, 0)
    with torch.inference_mode():
        functional._observe_load(workspace, config)
    entry = functional._load_entry(workspace, create=False)
    assert not any(t.is_inference() for t in entry["host"] + entry["tokens_host"])
    torch.cuda.synchronize()
    functional._observe_load(workspace, config)
    torch.cuda.synchronize()
    with torch.inference_mode():
        functional._observe_load(workspace, config)
    torch.cuda.synchronize()
    functional._observe_load(workspace, config)
    expected = backend.count_routed_load(routes, ep_size, experts).tolist()
    assert list(entry["loads"]) == expected


def _layout_reference(counts, peers):
    """Torch reference of the expert layout's written prefix: padded counts,
    exclusive row offsets, real rows per expert and the expert of each
    256-row block."""
    padded = counts.long()
    offsets = torch.cumsum(padded, 0) - padded
    real = torch.stack(
        [
            (peers[start : start + count] >= 0).sum()
            for start, count in zip(offsets.tolist(), padded.tolist(), strict=True)
        ]
    )
    blocks = torch.empty(
        int(padded.sum()) // 256, dtype=torch.int64, device=peers.device
    )
    for expert, (start, count) in enumerate(
        zip(offsets.tolist(), padded.tolist(), strict=True)
    ):
        blocks[start // 256 : (start + count) // 256] = expert
    return torch.cat([padded, offsets, real, blocks]).int()


@pytest.mark.parametrize("ep,experts", SCHEDULER_LAYOUTS)
def test_expert_layout_matches_torch_reference(ep, experts):
    """The layout kernel's written prefix equals the torch reference on the
    scheduler's own outputs (masked ranks and empty experts included), through
    the module helper and the scheduler's ``layout`` method alike."""
    require_gpu()
    _require_role(f"count_{ep}_{experts}")
    topk = _scheduler_topk(ep, experts)
    scheduler = MoKScheduler(ep, experts)
    tokens = 512
    generator = torch.Generator(device="cuda").manual_seed(1107)
    ids = (
        rand(ep * tokens, ep * experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    ids[0, tokens - 37 :] = -1
    if experts >= 4:
        ids[ids == experts - 1] = -1  # an empty local expert on rank 0
    capacity = ep * tokens * topk + experts * 256
    peers, _, total, counts = scheduler(ids, capacity, 0)
    layout = kernels.expert_layout(scheduler.layout_module, counts, capacity, peers)
    assert (
        layout.dtype == torch.int32 and layout.numel() == 3 * experts + capacity // 256
    )
    written = 3 * experts + total.item() // 256
    assert torch.equal(layout[:written], _layout_reference(counts, peers))
    again = scheduler.layout(counts, capacity, peers)
    assert torch.equal(again[:written], layout[:written])


def test_launchers_accept_a_precomputed_layout(monkeypatch):
    """The fused launchers derive the expert layout themselves (``layout=None``)
    or read the one passed to them; both paths are bitwise equal, eager and
    under graph replay, and a layout of another geometry is rejected."""
    require_gpu()
    forward, recompute, backward = MoKForward(), MoKRecompute(), MoKBackward()
    tokens, hidden, intermediate, experts, topk, mini, comm, macro = (
        512, 512, 256, 4, 2, 256, 4, 768,
    )  # fmt: skip
    torch.manual_seed(4111)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = randn(tokens, hidden, **options) * 0.125
    dy = randn_like(x) * 0.125
    shared = [
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = rand(tokens, topk, device="cuda") + 0.1
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    capacity = s["peers"].numel()
    layout = kernels.expert_layout(
        forward.layout_module, s["counts"], capacity, s["peers"]
    )
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer = torch.zeros(tokens * topk, hidden, **options)
    ds_peer = torch.zeros_like(scores)
    rows = min(s["actual"], macro)
    calls = []
    original = kernels.expert_layout

    def counting(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(kernels, "expert_layout", counting)

    def sequence(layout_arg):
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
        saved = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *schedule, topk, None, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        rebuilt = recompute(
            x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
            *schedule, topk, None, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        grads = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *saved[:7], x, [x.data_ptr()], *schedule,
            topk, None, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        return saved, rebuilt, grads

    def outputs(result):
        saved, rebuilt, grads = result
        return (
            *(saved[index][:rows] for index in (0, 2, 4, 6)),
            *(saved[index] for index in (1, 3, 5, 7)),
            *(rebuilt[index][:rows] for index in (0, 2, 4, 6)),
            *(rebuilt[index] for index in (1, 3, 5)),
            grads[0], *grads[9:],
            combine, dx_peer, ds_peer,
        )  # fmt: skip

    runs = []
    for layout_arg in (None, None, None, layout, layout, layout):
        before = len(calls)
        runs.append(tuple(v.clone() for v in outputs(sequence(layout_arg))))
        assert len(calls) - before == (3 if layout_arg is None else 0)
    assert all(
        all(torch.equal(a, b) for a, b in zip(runs[0], other, strict=True))
        for other in runs[1:]
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = sequence(layout)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert all(
            torch.equal(a, b) for a, b in zip(runs[0], outputs(captured), strict=True)
        )
    short = layout[:-1]
    with pytest.raises(ValueError, match="schedule geometry"):
        forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
            *schedule, topk, None, comm, macro, mini, layout=short,
        )  # fmt: skip


def _slab_layout(sizes):
    """Expected (offset, size) of every kept piece of ``_zero_slab(..., *sizes)``
    and the total length, in int32 elements."""
    pieces, total = [], 0
    for size in sizes:
        total += -total % 32
        for n in size if isinstance(size, tuple) else (size,):
            pieces.append((total, n))
            total += n
    return pieces, total


@pytest.mark.parametrize(
    "sizes",
    [
        (1, 64, 8704, 1088, 64, 2048, 1),
        (64, 64, 64, 8704, 1088, 8192, 1024, 1, 1),
        (1,),
        (33, 1, 31, 32),
        (264, (1, 32)),
    ],
)
def test_zero_slab_views_are_disjoint_aligned_and_zero(sizes):
    """One zero slab per call: every counter a contiguous 1-D int32 view with a
    128-byte aligned start, pairwise disjoint on one storage, zero, tuple
    groups adjacent; fresh per call and re-zeroed on every graph replay."""
    require_gpu()
    expected, total = _slab_layout(sizes)
    views = kernels._zero_slab("cuda", *sizes)
    assert len(views) == len(expected)
    base = views[0].untyped_storage().data_ptr()
    for view, (offset, size) in zip(views, expected, strict=True):
        assert view.dtype == torch.int32 and view.is_cuda and view.dim() == 1
        assert view.numel() == size and view.is_contiguous()
        assert view.untyped_storage().data_ptr() == base
        assert view.data_ptr() == base + 4 * offset
        assert torch.count_nonzero(view).item() == 0
    assert views[0].untyped_storage().nbytes() == 4 * total
    for view in views:
        if (
            view.data_ptr() % 128 != 0
        ):  # a tuple-group member: adjacent to its predecessor
            previous = [
                v for v in views if v.data_ptr() + 4 * v.numel() == view.data_ptr()
            ]
            assert len(previous) == 1
    ranges = sorted((v.data_ptr(), v.data_ptr() + 4 * v.numel()) for v in views)
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))
    views[0].fill_(7)
    assert all(torch.count_nonzero(v).item() == 0 for v in views[1:])
    again = kernels._zero_slab("cuda", *sizes)
    assert again[0].data_ptr() != views[0].data_ptr()
    assert all(torch.count_nonzero(v).item() == 0 for v in again)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernels._zero_slab("cuda", *sizes)
        captured[0].fill_(5)
    for _ in range(2):
        captured[-1].fill_(9)
        graph.replay()
        torch.cuda.synchronize()
        assert (captured[0] == 5).all().item()
        assert all(torch.count_nonzero(v).item() == 0 for v in captured[1:])


def test_zero_slab_issues_one_zeros_and_one_split():
    """The slab is one ``zeros`` and one ``split_with_sizes``: no per-counter
    slicing, filling or copying."""
    require_gpu()
    from collections import Counter

    with _AtenRecorder() as recorder:
        kernels._zero_slab("cuda", 1, 64, 8704, 1088, 64, 2048, 1)
    assert Counter(recorder.ops) == {"zeros": 1, "split_with_sizes": 1}, recorder.ops


class _LaunchRecorder:
    """Stands in for a prepared kernel: records the launch keywords, launches nothing."""

    def __init__(self):
        self.calls = []

    def launch(self, *, grid, **kwargs):
        self.calls.append((grid, kwargs))

    def last(self, name):
        return next(kwargs for _, kwargs in reversed(self.calls) if name in kwargs)


def _check_counters(kwargs, expected):
    """Counter keywords ``expected`` (name -> numel) of one recorded fused launch:
    CUDA, contiguous, 1-D, int32 (uint32 for ``combine_next``), the exact
    numel, zero, 128-byte aligned, pairwise disjoint on one storage."""
    views = {name: kwargs[name] for name in expected}
    assert len({v.untyped_storage().data_ptr() for v in views.values()}) == 1
    for name, numel in expected.items():
        view = views[name]
        assert view.is_cuda and view.is_contiguous() and view.dim() == 1, name
        assert view.dtype == (torch.uint32 if name == "combine_next" else torch.int32)
        assert view.numel() == numel, (name, view.numel(), numel)
        assert torch.count_nonzero(view.view(torch.int32)).item() == 0, name
        assert view.data_ptr() % 128 == 0, name
    ranges = sorted(
        (v.data_ptr(), v.data_ptr() + 4 * v.numel()) for v in views.values()
    )
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))


def _bf16_backward_clusters(
    local_tokens, hidden, intermediate, capacity, macro, mini, experts
):
    """The BF16 backward grid's cluster count, as the launcher derived it per call."""
    minis = (capacity + mini - 1) // mini
    macros = (capacity + macro - 1) // macro
    shared_rows = local_tokens // 256
    ibw, hbw = (intermediate + 511) // 512, (hidden + 511) // 512
    shared_tasks = (
        shared_rows * (ibw + hbw)
        + kernels.swiglu_backward_tasks((local_tokens // 128) * (intermediate // 128))
        + 3 * kernels.wgrad_tiles(hidden, intermediate)
    )
    mini_bwd = (mini // 256) * (ibw + hbw) + kernels.swiglu_backward_tasks(
        (mini // 128) * (intermediate // 128)
    )
    mini_replay = (
        2 * (mini // 256) * ibw + ((mini // 128) * (intermediate // 128) + 5) // 6
    )
    return (
        shared_tasks
        + minis * mini_bwd
        + max(0, minis - macro // mini) * mini_replay
        + macros * 3 * experts * kernels.wgrad_tiles(hidden, intermediate)
    )


def test_launch_counters_are_one_slab_per_launch():
    """At the launch boundary the scheduler and the BF16 launchers hand the
    kernels the counter names, dtypes and extents they always did, now as
    views of one zero slab per launch; the schedule sentinel stays separate,
    and one launcher memoises its geometry per distinct expert count."""
    require_gpu()
    ep, experts, tokens, topk = 4, 4, 512, 2
    scheduler = MoKScheduler(ep, experts)
    recorder = _LaunchRecorder()
    scheduler.count = scheduler.pad = scheduler.rows = recorder
    generator = torch.Generator(device="cuda").manual_seed(77)
    ids = randint(
        0, ep * experts, (ep, tokens, topk), device="cuda", dtype=torch.int32,
        generator=generator,
    )  # fmt: skip
    capacity = ep * tokens * topk + experts * 256
    peers, _, total, counts = scheduler(ids, capacity, 0)
    count_kw, pad_kw, rows_kw = (kwargs for _, kwargs in recorder.calls)
    assert count_kw["counts"] is pad_kw["counts"] is rows_kw["counts"]
    slab = count_kw["counts"]
    assert slab.numel() == experts * ep + ep and slab.dtype == torch.int32
    assert slab.data_ptr() % 128 == 0 and torch.count_nonzero(slab).item() == 0
    assert pad_kw["num_tokens"] is total and pad_kw["tokens_per_expert"] is counts
    assert total.numel() == 1 and counts.numel() == experts
    assert total.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr()
    assert total.data_ptr() % 128 == 0 and counts.data_ptr() == total.data_ptr() + 4
    assert total.data_ptr() >= slab.data_ptr() + 4 * slab.numel()
    owner = scheduler.owner_loads()
    assert (
        owner.numel() == ep and owner.data_ptr() == slab.data_ptr() + 4 * experts * ep
    )
    assert rows_kw["schedule_peer_rank"] is peers and (peers == -1).all().item()
    assert peers.untyped_storage().data_ptr() != slab.untyped_storage().data_ptr()

    hidden, intermediate, macro, mini, comm = 256, 256, 768, 256, 4
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = randn(tokens, hidden, **options)
    dy = randn_like(x)
    shared = [randn(intermediate, hidden, **options) for _ in range(2)] + [
        randn(hidden, intermediate, **options)
    ]
    scores = rand(tokens, topk, device="cuda") + 0.1
    routes = torch.stack(
        [
            (torch.arange(tokens, device="cuda") + slot) % experts
            for slot in range(topk)
        ],
        -1,
    )
    combine = torch.empty(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.empty_like(combine), torch.empty_like(scores)
    forward, recompute, backward = MoKForward(), MoKRecompute(), MoKBackward()
    for launcher in (forward, recompute, backward):
        launcher.module = launcher.layout_module = _LaunchRecorder()
    backward.zero_module = backward.module
    for count in (experts, 2 * experts):
        routed = [
            randn(count, intermediate, hidden, **options),
            randn(count, intermediate, hidden, **options),
            randn(count, hidden, intermediate, **options),
        ]
        s = single_rank_schedule(routes % count, count)
        schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
        cap = s["peers"].numel()
        minis, macros = (cap + mini - 1) // mini, (cap + macro - 1) // macro
        shared_rows, routed_rows = tokens // 256, cap // 256
        ib = intermediate // 256
        for _ in range(2):
            saved = forward(
                x, [x.data_ptr()], combine, [combine.data_ptr()],
                shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                *schedule, topk, None, comm, macro, mini,
            )  # fmt: skip
            recompute(
                x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
                *schedule, topk, None, comm, macro, mini,
            )  # fmt: skip
            backward(
                dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
                scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
                shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                *saved[:7], x, [x.data_ptr()], *schedule,
                topk, None, comm, macro, mini,
            )  # fmt: skip
        _check_counters(
            forward.module.last("x_ready"),
            dict(
                x_ready=minis,
                hidden_ready=shared_rows + routed_rows,
                y_ready=minis,
                y_done=cap // 128,
            ),
        )
        routed_capacity = min(cap, macro)
        _check_counters(
            recompute.module.last("x_ready"),
            dict(
                x_ready=(routed_capacity + mini - 1) // mini,
                hidden_ready=shared_rows + routed_capacity // 256,
            ),
        )
        _check_counters(
            backward.module.last("dy_ready"),
            dict(
                dy_ready=minis,
                dh_ready=(shared_rows + routed_rows) * ib,
                dg_ready=shared_rows + routed_rows,
                dx_ready=minis,
                replay_x=minis,
                replay_gu=routed_rows * ib,
                replay_h=routed_rows,
                buffers_done=macros,
                weight_ready=macros,
            ),
        )
        grid, _ = next(
            call for call in reversed(backward.module.calls) if "dy_ready" in call[1]
        )
        clusters = _bf16_backward_clusters(
            tokens, hidden, intermediate, cap, macro, mini, count
        )
        assert grid == (comm + 2 * clusters, 1, 1)
    # One geometry entry per expert count on every launcher; the backward
    # geometry depends on it.
    launchers = (forward, recompute, backward)
    assert [len(launcher._geometry) for launcher in launchers] == [2, 2, 2]
    first, second = backward._geometry.values()
    assert first.clusters != second.clusters
    assert first.weight_tasks != second.weight_tasks


def _mxfp8_forward_geometry_reference(
    local_tokens, hidden, intermediate, capacity, macro, mini, tile
):
    """The MXFP8 forward launcher's per-call task arithmetic, as derived before memoisation."""
    shared_rows = local_tokens // 256
    shared_tasks = shared_rows * (intermediate // 256) + shared_rows * (
        (hidden + 511) // 512
    )
    mini_swiglu = ((mini // 128) * (intermediate // 128) + 5) // 6
    gate_passes = 1 if tile == 512 else 2
    mini_tasks = (
        gate_passes * (mini // 256) * (intermediate // 256)
        + mini_swiglu
        + (mini // 256) * ((hidden + tile - 1) // tile)
    )
    minis = (capacity + mini - 1) // mini
    return dict(
        m_tiles=macro // 128,
        h_tiles=hidden // 128,
        i_tiles=intermediate // 128,
        shared_rows=shared_rows,
        routed_rows=capacity // 256,
        shared_tasks=shared_tasks,
        mini_swiglu=mini_swiglu,
        mini_tasks=mini_tasks,
        minis=minis,
        mini_tasks_div=tuple(kernels._fast_divmod(mini_tasks)),
        minis_per_macro_div=tuple(kernels._fast_divmod(macro // mini)),
        gate_col_blocks_div=tuple(kernels._fast_divmod((intermediate + 255) // 256)),
        down_col_blocks_div=tuple(kernels._fast_divmod((hidden + tile - 1) // tile)),
    )


def _mxfp8_backward_geometry_reference(
    local_tokens, hidden, intermediate, capacity, macro, mini, experts, tile
):
    """The MXFP8 backward launcher's per-call task arithmetic, as derived before memoisation."""
    minis = (capacity + mini - 1) // mini
    macros = (capacity + macro - 1) // macro
    shared_rows, routed_rows = local_tokens // 256, capacity // 256
    ib = intermediate // 256
    ibw, hbw = (intermediate + 511) // 512, (hidden + 511) // 512
    ibr, hbr = (intermediate + tile - 1) // tile, (hidden + tile - 1) // tile
    shared_tasks = (
        shared_rows * (ibw + hbw)
        + ((local_tokens // 128) * (intermediate // 128) + 3) // 4
        + 3 * kernels.wgrad_tiles(hidden, intermediate)
    )
    mini_bwd = (mini // 256) * (ibr + hbr) + (
        (mini // 128) * (intermediate // 128) + 3
    ) // 4
    mini_replay = (
        2 * (mini // 256) * ib + ((mini // 128) * (intermediate // 128) + 5) // 6
    )
    clusters = (
        shared_tasks
        + minis * mini_bwd
        + max(0, minis - macro // mini) * mini_replay
        + macros
        * 3
        * experts
        * kernels.wgrad_tiles(hidden, intermediate, tile_cols=tile)
    )
    shared_wgrad = kernels.wgrad_tiles(hidden, intermediate)
    routed_wgrad = kernels.wgrad_tiles(hidden, intermediate, tile_cols=tile)
    weight_tasks = experts * routed_wgrad
    replay_macro_tasks = (macro // mini) * (mini_replay + mini_bwd) + 3 * weight_tasks
    return dict(
        minis=minis,
        macros=macros,
        shared_rows=shared_rows,
        routed_rows=routed_rows,
        ib=ib,
        clusters=clusters,
        weight_tasks=weight_tasks,
        replay_macro_tasks=replay_macro_tasks,
        mini_size_div=tuple(kernels._fast_divmod(mini)),
        mini_bwd_div=tuple(kernels._fast_divmod(mini_bwd)),
        mini_replay_div=tuple(kernels._fast_divmod(mini_replay)),
        replay_macro_tasks_div=tuple(kernels._fast_divmod(replay_macro_tasks)),
        weight_tasks_div=tuple(kernels._fast_divmod(weight_tasks)),
        shared_wgrad_div=tuple(kernels._fast_divmod(shared_wgrad)),
        routed_wgrad_div=tuple(kernels._fast_divmod(routed_wgrad)),
        dh_col_blocks_div=tuple(kernels._fast_divmod(ibr)),
        dx_col_blocks_div=tuple(kernels._fast_divmod(hbr)),
        replay_col_blocks_div=tuple(kernels._fast_divmod((intermediate + 255) // 256)),
    )


@pytest.mark.parametrize("tile", [256, 512])
@pytest.mark.parametrize(
    "local_tokens,capacity,macro,experts",
    [(16384, 262144, 262144, 32), (32768, 524288, 262144, 64)],
    ids=["ep8-local32", "ep4-local64"],
)
def test_launcher_geometry_memo_is_pure(tile, local_tokens, capacity, macro, experts):
    """The memoised launch geometry of the MXFP8 launchers equals a fresh
    evaluation of the per-call arithmetic at training geometries and both tile
    widths, including the expert-dependent backward cluster and task counts; a
    full memo drops its least recently used entry and still serves correct
    values."""
    hidden, intermediate, mini = 6144, 2048, 4096
    forward = kernels._forward_mxfp8_geometry(
        local_tokens, hidden, intermediate, capacity, macro, mini, tile
    )
    for name, value in _mxfp8_forward_geometry_reference(
        local_tokens, hidden, intermediate, capacity, macro, mini, tile
    ).items():
        actual = getattr(forward, name)
        assert (tuple(actual) if name.endswith("_div") else actual) == value, name
    assert forward.tile == tile and forward.gate_passes == (1 if tile == 512 else 2)
    for count in (experts, 9):
        backward = kernels._backward_mxfp8_geometry(
            local_tokens, hidden, intermediate, capacity, macro, mini, count, tile
        )
        for name, value in _mxfp8_backward_geometry_reference(
            local_tokens, hidden, intermediate, capacity, macro, mini, count, tile
        ).items():
            actual = getattr(backward, name)
            assert (tuple(actual) if name.endswith("_div") else actual) == value, name
    cache = {}
    keys = [(i, tile) for i in range(kernels._GEOMETRY_LIMIT + 1)]
    for key in keys[:-1]:
        kernels._remember(cache, key, key[0])
    # A hit refreshes keys[0]; keys[1] becomes the least recently used entry.
    assert kernels._recall(cache, keys[0]) == 0
    kernels._remember(cache, keys[-1], keys[-1][0])
    assert len(cache) == kernels._GEOMETRY_LIMIT and keys[1] not in cache
    assert all(cache[key] == key[0] for key in (keys[0], *keys[2:]))
    assert kernels._recall(cache, keys[1]) is None


def test_ffi_stream_context_follows_torch_current_stream():
    """The launchers' FFI stream context sets the environment stream of the
    current device to the current torch stream and restores it, exactly as
    ``tvm_ffi.use_torch_stream()`` does, and launches capture and replay
    bitwise under it."""
    require_gpu()
    import tvm_ffi

    device = tvm_ffi.device("cuda", torch.cuda.current_device())
    before = tvm_ffi.get_raw_stream(device)
    with kernels._ffi_stream():
        assert tvm_ffi.get_raw_stream(device) == torch.cuda.current_stream().cuda_stream
    assert tvm_ffi.get_raw_stream(device) == before
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        with tvm_ffi.use_torch_stream():
            reference = tvm_ffi.get_raw_stream(device)
        with kernels._ffi_stream():
            assert tvm_ffi.get_raw_stream(device) == reference == side.cuda_stream
        assert tvm_ffi.get_raw_stream(device) == before
    epilogues = MoKEpilogues(2)
    shared = randn(64, 256, device="cuda", dtype=torch.bfloat16)
    routed = randn(128, 256, device="cuda", dtype=torch.bfloat16)
    scores = rand(64, 2, device="cuda")
    expected = epilogues.forward(shared, routed, scores)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = epilogues.forward(shared, routed, scores)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, expected)


def test_kernel_launch_keeps_the_exported_abi_guard():
    """``_Kernel.launch`` rejects a missing, extra or renamed keyword with the
    ABI error and passes a correct set positionally in the exported order."""
    calls = []

    class Stub:
        def run(self, *args):
            calls.append(args)
            return "ran"

    kernel = jit._Kernel(Stub(), ("a", "b", "grid_x", "grid_y", "grid_z"))
    for kwargs in (dict(a=1), dict(a=1, b=2, c=3), dict(a=1, c=2)):
        with pytest.raises(ValueError, match="exported ABI"):
            kernel.launch(grid=(1, 1, 1), **kwargs)
    assert kernel.launch(grid=(2, 3, 4), b=2, a=1) == "ran"
    assert calls == [(1, 2, 2, 3, 4)]
    with pytest.raises(ValueError):
        kernel.launch(grid=(1, 1), a=1, b=2)


def test_select_comm_split_decision_table():
    """Host-only: the imbalance-aware split is a pure function of the observed
    per-rank loads, ``hot`` when max load >= threshold x mean load, else ``base``."""
    select_comm_split = backend.select_comm_split

    base, hot = (20, 24), (24, 28)
    assert select_comm_split([100, 100, 100, 100], base, hot, 1.5) == base  # 1.0
    assert select_comm_split([150, 100, 100, 100], base, hot, 1.5) == base  # 1.33
    assert select_comm_split([200, 100, 100, 100], base, hot, 1.5) == hot  # 1.6
    assert select_comm_split([150, 50, 50, 50], base, hot, 1.5) == hot  # 2.0
    assert select_comm_split([3, 1], base, hot, 1.5) == hot  # exactly 1.5
    assert select_comm_split([0, 0, 0, 0], base, hot, 1.5) == base  # every rank empty
    assert select_comm_split((), base, hot, 1.5) == base  # no observation yet
    assert select_comm_split([200, 100, 100, 100], base, None, 1.5) == base
    # The shipped threshold: 3x keeps the fixed split, 5.6x switches.
    assert (
        select_comm_split([300, 100, 100, 100, 100, 100, 0, 0], base, hot, 4.0) == base
    )  # 3.0
    assert (
        select_comm_split([560, 100, 100, 40, 0, 0, 0, 0], base, hot, 4.0) == hot
    )  # 5.6
    assert (
        select_comm_split([100, 0, 0, 0], base, hot, 4.0) == hot
    )  # exactly 4 with 4 ranks
    loads = torch.tensor([300, 100, 100, 100], dtype=torch.int32)
    assert select_comm_split(loads, base, hot, 1.5) == hot
    assert select_comm_split(loads, base, [24, 28], 1.5) == (24, 28)


@pytest.mark.parametrize(
    "capability,sm_count,precision,expected",
    [
        ((10, 0), 148, "bf16", (24, 28)),  # B200
        ((10, 0), 148, "mxfp8", (44, 36)),
        ((10, 0), 152, "bf16", None),  # GB200: same capability, no measured row
        ((10, 0), 152, "mxfp8", None),
        ((10, 3), 148, "bf16", None),  # B300: no imbalance row
        ((10, 3), 148, "mxfp8", None),
        ((10, 3), 152, "bf16", None),  # GB300
        ((10, 3), 152, "mxfp8", None),
        ((10, 7), 160, "bf16", None),
        ((10, 7), 212, "mxfp8", (72, 60)),  # Rubin
        ((10, 7), 212, "bf16", None),
        ((10, 7), 216, "mxfp8", None),  # other Rubin SM count
        ((12, 0), 170, "bf16", None),
    ],
)
def test_imbalance_comm_sms_auto_follows_the_device(
    monkeypatch, capability, sm_count, precision, expected
):
    """Host-only: ``"auto"`` resolves through the per-device table (compute
    capability and SM count); devices and precisions without a measured row keep
    the fixed split (``None``)."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_imbalance_comm_sms(precision) == expected


@pytest.mark.parametrize(
    "capability,sm_count,precision,expected",
    [
        ((10, 7), 212, "mxfp8", 1.5),  # Rubin: measured device threshold
        ((10, 7), 212, "bf16", 4.0),
        ((10, 7), 216, "mxfp8", 2.7),  # other Rubin SM count: precision default
        ((10, 0), 148, "mxfp8", 2.7),  # B200
        ((10, 0), 148, "bf16", 4.0),
        ((10, 0), 152, "bf16", 4.0),  # GB200: precision default
        ((10, 3), 148, "bf16", 4.0),  # B300: precision default, no device threshold
        ((10, 3), 148, "mxfp8", 2.7),
        ((10, 3), 152, "bf16", 4.0),  # GB300: precision default
    ],
)
def test_imbalance_threshold_follows_the_device(
    monkeypatch, capability, sm_count, precision, expected
):
    """Host-only: with a device, the threshold is the device's measured row of
    ``IMBALANCE_DEVICE_THRESHOLDS`` where one exists, else the precision's
    ``IMBALANCE_THRESHOLDS`` value; without a device it is always the latter."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_imbalance_threshold(precision, 0) == expected
    assert (
        api.resolve_imbalance_threshold(precision)
        == api.IMBALANCE_THRESHOLDS[precision]
    )


def test_imbalance_comm_sms_defaults_keep_the_fixed_split():
    """Host-only: the config default disables the imbalance split, the facade
    default ``"auto"`` resolves the measured row of the device generation, and
    malformed arguments are rejected before any collective."""
    import inspect

    config = MoKConfig()
    assert config.imbalance_comm_sms is None
    assert config.imbalance_threshold == 4.0
    facade = inspect.signature(api.create_mok_bf16_workspace).parameters
    assert facade["imbalance_comm_sms"].default == "auto"
    assert facade["imbalance_threshold"].default == "auto"
    # The measured rows and thresholds (backend README); a change here is a
    # shipping change. The backend default is the BF16 threshold.
    assert api.IMBALANCE_COMM_SMS == {
        (10, 0): {148: {"bf16": (24, 28), "mxfp8": (44, 36)}},
        (10, 7): {212: {"mxfp8": (72, 60)}},
    }
    assert api.IMBALANCE_THRESHOLDS == {"bf16": 4.0, "mxfp8": 2.7}
    assert api.IMBALANCE_DEVICE_THRESHOLDS == {(10, 7): {212: {"mxfp8": 1.5}}}
    assert api.IMBALANCE_THRESHOLDS["bf16"] == api.IMBALANCE_THRESHOLD
    assert api.resolve_imbalance_threshold("bf16") == config.imbalance_threshold
    assert api.resolve_imbalance_threshold("mxfp8") == 2.7
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_imbalance_comm_sms("fp8")
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_imbalance_threshold("fp8")
    kwargs = dict(
        group=None, device=None, num_local_tokens=512, hidden_size=256, topk=2
    )
    with pytest.raises(ValueError, match='"auto"'):
        api.create_mok_bf16_workspace(imbalance_comm_sms="fast", **kwargs)
    with pytest.raises(ValueError, match='"auto"'):
        api.create_mok_bf16_workspace(imbalance_threshold="fast", **kwargs)
    with pytest.raises(ValueError, match="pair"):
        api.create_mok_bf16_workspace(imbalance_comm_sms=(24,), **kwargs)
    with pytest.raises(ValueError, match="precision must be one of"):
        api.create_mok_bf16_workspace(precision="fp8", **kwargs)


@pytest.mark.parametrize(
    "capability,sm_count,precision,ep_size,expected",
    [
        ((10, 3), 152, "mxfp8", 8, (44, 36)),  # GB300, EP8 across two nodes
        ((10, 3), 152, "mxfp8", 32, (44, 36)),
        ((10, 3), 152, "mxfp8", 64, (44, 36)),
        ((10, 3), 152, "mxfp8", 4, None),  # GB300 EP4 / EP16: no gain
        ((10, 3), 152, "mxfp8", 16, None),
        ((10, 3), 152, "bf16", 4, (20, 24)),
        ((10, 3), 152, "bf16", 8, (20, 24)),
        ((10, 3), 152, "bf16", 16, (20, 24)),
        ((10, 3), 152, "bf16", 32, (20, 24)),
        ((10, 3), 152, "bf16", 64, (20, 24)),
        ((10, 3), 152, "bf16", 1, None),  # unmeasured group size: factory split
        ((10, 3), 160, "mxfp8", 8, None),  # other GB300 SM count: factory split
        ((10, 0), 148, "mxfp8", 8, None),  # B200: the imbalance switch instead
        ((10, 0), 152, "mxfp8", 8, None),  # GB200
        ((10, 0), 148, "bf16", 8, None),  # B200 EP8: slower from 2.0x skew
        ((10, 0), 148, "bf16", 4, None),
        ((10, 0), 148, "bf16", 16, None),
        ((10, 0), 152, "bf16", 4, None),  # GB200 EP4: no measured gain
        ((10, 0), 152, "bf16", 8, None),
        ((10, 0), 152, "bf16", 16, None),
        ((10, 0), 152, "mxfp8", 4, None),  # the MXFP8 key: factory split
        ((10, 3), 148, "bf16", 8, (24, 40)),  # B300 EP8: fixed, no imbalance row
        ((10, 3), 148, "bf16", 4, None),
        ((10, 3), 148, "bf16", 16, None),
        ((10, 3), 148, "mxfp8", 8, None),  # the MXFP8 key: factory split
        ((10, 7), 160, "mxfp8", 8, None),
        ((10, 7), 212, "mxfp8", 8, (60, 60)),  # Rubin, EP8 across two nodes
        ((10, 7), 212, "mxfp8", 16, None),
        ((10, 7), 212, "bf16", 8, None),
        ((10, 7), 216, "mxfp8", 8, None),  # other Rubin SM count: factory split
    ],
)
def test_comm_sms_auto_follows_the_device_and_group(
    monkeypatch, capability, sm_count, precision, ep_size, expected
):
    """Host-only: ``"auto"`` for the fixed split resolves through the per-device,
    per-group-size table; devices, precisions and group sizes without a measured
    row keep the factory split (``None``)."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_comm_sms(precision, ep_size) == expected


def test_comm_sms_defaults_resolve_the_measured_rows():
    """Host-only: the facade defaults are ``"auto"`` (the measured row, else the
    factory split 40 / 40), the measured rows are the shipped table (a change
    here is a shipping change) and malformed arguments are rejected before any
    collective."""
    import inspect

    facade = inspect.signature(api.create_mok_bf16_workspace).parameters
    assert facade["fwd_num_comm_sms"].default == "auto"
    assert facade["bwd_num_comm_sms"].default == "auto"
    assert api._FACTORY_COMM_SMS == (40, 40)
    assert api.COMM_SMS == {
        (10, 3): {
            148: {"bf16": {8: (24, 40)}},
            152: {
                "bf16": {
                    4: (20, 24),
                    8: (20, 24),
                    16: (20, 24),
                    32: (20, 24),
                    64: (20, 24),
                },
                "mxfp8": {8: (44, 36), 32: (44, 36), 64: (44, 36)},
            },
        },
        (10, 7): {212: {"mxfp8": {8: (60, 60)}}},
    }
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_comm_sms("fp8", 8)
    with pytest.raises(ValueError, match="ep_size must be a positive"):
        api.resolve_comm_sms("mxfp8", 0)
    with pytest.raises(ValueError, match='both be "auto"'):
        api.create_mok_bf16_workspace(
            group=None,
            device=None,
            num_local_tokens=512,
            hidden_size=512,
            topk=2,
            fwd_num_comm_sms="auto",
            bwd_num_comm_sms=40,
        )


@pytest.mark.parametrize(
    "capability,sm_count,ep_size,precision,expected",
    [
        # (fwd_num_comm_sms / bwd_num_comm_sms, bf16_comm_sms, mxfp8_comm_sms)
        ((10, 0), 148, 8, "bf16", ((40, 40), (40, 40), (40, 40))),  # B200 EP8
        ((10, 3), 148, 8, "mxfp8", ((40, 40), (24, 40), (40, 40))),  # B300 EP8
        ((10, 0), 148, 4, "bf16", ((40, 40), (40, 40), (40, 40))),
        ((10, 0), 152, 4, "bf16", ((40, 40), (40, 40), (40, 40))),  # GB200 EP4
        ((10, 3), 148, 8, "bf16", ((24, 40), (24, 40), (40, 40))),  # B300 EP8
        ((10, 3), 152, 4, "bf16", ((20, 24), (20, 24), (40, 40))),  # GB300 EP4
        ((10, 3), 152, 8, "bf16", ((20, 24), (20, 24), (44, 36))),  # GB300 EP8
        ((10, 3), 152, 8, "mxfp8", ((44, 36), (20, 24), (44, 36))),
        ((10, 7), 212, 8, "bf16", ((40, 40), (40, 40), (60, 60))),  # Rubin EP8
        ((10, 7), 212, 4, "mxfp8", ((40, 40), (40, 40), (40, 40))),
    ],
)
def test_comm_sms_auto_gives_each_routed_precision_its_row(
    monkeypatch, capability, sm_count, ep_size, precision, expected
):
    """Host-only: ``"auto"`` resolves the row of each routed precision for the
    device and group size (the factory split where there is none) into
    ``bf16_comm_sms`` / ``mxfp8_comm_sms``, whatever the workspace's
    ``precision``; ``fwd_num_comm_sms`` / ``bwd_num_comm_sms`` hold the row of
    ``precision``. Integer counts leave both per-precision splits unset."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: ep_size)
    created = []

    def create_source_workspace(config, group, **kwargs):
        created.append(config)
        return "workspace"

    monkeypatch.setattr(backend, "create_source_workspace", create_source_workspace)
    kwargs = dict(
        group=None, device=None, num_local_tokens=512, hidden_size=512, topk=2
    )
    config, workspace = api.create_mok_bf16_workspace(precision=precision, **kwargs)
    assert workspace == "workspace" and created == [config]
    assert (config.fwd_num_comm_sms, config.bwd_num_comm_sms) == expected[0]
    assert (config.bf16_comm_sms, config.mxfp8_comm_sms) == expected[1:]
    config, _ = api.create_mok_bf16_workspace(
        precision=precision, fwd_num_comm_sms=24, bwd_num_comm_sms=28, **kwargs
    )
    assert (config.fwd_num_comm_sms, config.bwd_num_comm_sms) == (24, 28)
    assert config.bf16_comm_sms is None and config.mxfp8_comm_sms is None


def test_comm_split_follows_the_routed_precision():
    """Host-only: without an imbalance split, a launch's fixed split is the row of
    its routed precision (``mxfp8_comm_sms`` for MXFP8 routed weights,
    ``bf16_comm_sms`` for BF16), else ``fwd_num_comm_sms`` /
    ``bwd_num_comm_sms``, and the config defaults leave both rows unset."""
    split = backend.MoKFunctional._comm_split
    assert MoKConfig().bf16_comm_sms is None and MoKConfig().mxfp8_comm_sms is None
    rows = MoKConfig(
        fwd_num_comm_sms=16,
        bwd_num_comm_sms=40,
        bf16_comm_sms=(16, 40),
        mxfp8_comm_sms=(40, 40),
    )
    assert split(None, rows, None) == (16, 40)
    assert split(None, rows, None, False) == (16, 40)
    assert split(None, rows, None, True) == (40, 40)
    # A workspace created with precision "mxfp8": its BF16 launches keep the BF16 row.
    mxfp8 = dataclasses.replace(rows, fwd_num_comm_sms=40, bwd_num_comm_sms=40)
    assert split(None, mxfp8, None, False) == (16, 40)
    assert split(None, mxfp8, None, True) == (40, 40)
    fixed = MoKConfig(fwd_num_comm_sms=20, bwd_num_comm_sms=24)
    assert split(None, fixed, None, False) == (20, 24)
    assert split(None, fixed, None, True) == (20, 24)


@pytest.mark.parametrize(
    "capability,sm_count,precision,ep_size,expected",
    [
        ((10, 3), 152, "bf16", 8, {(4096, 1024, 10): (56, 48)}),  # GB300 EP8
        ((10, 3), 148, "bf16", 8, {(4096, 1024, 10): (56, 40)}),  # B300 EP8
        ((10, 0), 148, "bf16", 8, {(5120, 2304, 6): (20, 40)}),  # B200 EP8
        ((10, 0), 152, "bf16", 8, {(4096, 1024, 10): (56, 48)}),  # GB200 EP8
        ((10, 3), 152, "mxfp8", 8, None),  # no MXFP8 shape row
        ((10, 3), 148, "mxfp8", 8, None),
        ((10, 0), 148, "mxfp8", 8, None),
        ((10, 0), 152, "mxfp8", 8, None),
        ((10, 3), 152, "bf16", 4, None),  # other group sizes keep COMM_SMS
        ((10, 3), 152, "bf16", 16, None),
        ((10, 3), 148, "bf16", 4, None),
        ((10, 3), 148, "bf16", 16, None),
        ((10, 0), 148, "bf16", 4, None),
        ((10, 0), 148, "bf16", 16, None),
        ((10, 0), 152, "bf16", 4, None),
        ((10, 0), 152, "bf16", 16, None),
        ((10, 3), 160, "bf16", 8, None),  # other SM count
        ((10, 0), 132, "bf16", 8, None),
        ((10, 7), 212, "bf16", 8, None),
        ((10, 7), 212, "mxfp8", 8, None),
    ],
)
def test_shape_comm_sms_auto_follows_the_device_and_group(
    monkeypatch, capability, sm_count, precision, ep_size, expected
):
    """Host-only: the shape rows of ``"auto"`` resolve through the per-device,
    per-precision, per-group-size table; everywhere else there is none
    (``None``) and every launch keeps the COMM_SMS row or the factory split."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_shape_comm_sms(precision, ep_size) == expected


def test_shape_comm_sms_rows_are_the_measured_table(monkeypatch):
    """Host-only: the shape rows are the shipped table (a change here is a
    shipping change), the resolver returns a new dict and rejects malformed
    arguments."""
    from types import SimpleNamespace

    assert api.SHAPE_COMM_SMS == {
        (10, 0): {
            148: {"bf16": {8: {(5120, 2304, 6): (20, 40)}}},
            152: {"bf16": {8: {(4096, 1024, 10): (56, 48)}}},
        },
        (10, 3): {
            148: {"bf16": {8: {(4096, 1024, 10): (56, 40)}}},
            152: {"bf16": {8: {(4096, 1024, 10): (56, 48)}}},
        },
    }
    with pytest.raises(ValueError, match="precision must be one of"):
        api.resolve_shape_comm_sms("fp8", 8)
    with pytest.raises(ValueError, match="ep_size must be a positive"):
        api.resolve_shape_comm_sms("bf16", 0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=152),
    )
    rows = api.resolve_shape_comm_sms("bf16", 8)
    rows[(4096, 1024, 10)] = (2, 2)
    assert api.resolve_shape_comm_sms("bf16", 8) == {(4096, 1024, 10): (56, 48)}


@pytest.mark.parametrize(
    "capability,sm_count,ep_size,expected",
    [
        ((10, 3), 152, 8, (("bf16", 4096, 1024, 10, 56, 48),)),  # GB300 EP8
        ((10, 3), 148, 8, (("bf16", 4096, 1024, 10, 56, 40),)),  # B300 EP8
        ((10, 3), 152, 4, None),
        ((10, 3), 148, 4, None),
        ((10, 0), 148, 8, (("bf16", 5120, 2304, 6, 20, 40),)),  # B200 EP8
        ((10, 0), 152, 8, (("bf16", 4096, 1024, 10, 56, 48),)),  # GB200 EP8
        ((10, 0), 148, 4, None),
        ((10, 0), 152, 4, None),
        ((10, 7), 212, 8, None),
    ],
)
def test_comm_sms_auto_sets_the_shape_rows(
    monkeypatch, capability, sm_count, ep_size, expected
):
    """Host-only: ``"auto"`` puts the device's shape rows of every routed
    precision into ``MoKConfig.shape_comm_sms`` (``None`` without a row) and
    keeps the per-precision rows; integer counts leave it ``None``."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: ep_size)
    monkeypatch.setattr(
        backend, "create_source_workspace", lambda config, group, **kwargs: "workspace"
    )
    kwargs = dict(
        group=None, device=None, num_local_tokens=512, hidden_size=4096, topk=10
    )
    for precision in ("bf16", "mxfp8"):
        config, _ = api.create_mok_bf16_workspace(precision=precision, **kwargs)
        assert config.shape_comm_sms == expected
        assert config.bf16_comm_sms == (
            api.resolve_comm_sms("bf16", ep_size) or (40, 40)
        )
    config, _ = api.create_mok_bf16_workspace(
        fwd_num_comm_sms=24, bwd_num_comm_sms=28, **kwargs
    )
    assert config.shape_comm_sms is None


def test_comm_split_follows_the_launch_shape():
    """Host-only: without an imbalance split, a launch whose routed precision,
    workspace hidden width and top-k and routed intermediate width match a row
    of ``shape_comm_sms`` takes that row; any other launch (another width,
    top-k or precision, or no width given) keeps the routed-precision row."""
    from types import SimpleNamespace

    split = backend.MoKFunctional._comm_split
    assert MoKConfig().shape_comm_sms is None
    rows = MoKConfig(
        fwd_num_comm_sms=20,
        bwd_num_comm_sms=24,
        bf16_comm_sms=(20, 24),
        mxfp8_comm_sms=(44, 36),
        shape_comm_sms=(("bf16", 4096, 1024, 10, 56, 48),),
    )
    ws = SimpleNamespace(hidden_size=4096, topk=10)
    assert split(None, rows, ws, False, 1024) == (56, 48)
    assert split(None, rows, ws, False, 2048) == (20, 24)
    assert split(None, rows, ws, False) == (20, 24)
    assert split(None, rows, ws, True, 1024) == (44, 36)
    assert split(None, rows, SimpleNamespace(hidden_size=4096, topk=8), False, 1024) == (20, 24)
    assert split(None, rows, SimpleNamespace(hidden_size=6144, topk=10), False, 1024) == (20, 24)
    fixed = MoKConfig(fwd_num_comm_sms=40, bwd_num_comm_sms=40)
    assert split(None, fixed, ws, False, 1024) == (40, 40)


def test_comm_split_takes_each_shape_row_and_the_imbalance_split(monkeypatch):
    """Host-only: with two shape rows at EP8 (the 512-expert layout 56 / 40 and
    the 384-expert layout 20 / 40, a constructed config), each forward or
    backward launch takes the row of its own shape, and the context recompute
    launch (no width given), an MXFP8 launch and another width keep the
    routed-precision row. With an imbalance split (B200 BF16 24 / 28 from
    4.0x), a skewed observation replaces the shape row; no observation and a
    balanced one keep it."""
    from types import SimpleNamespace

    split = backend.MoKFunctional._comm_split
    rows = MoKConfig(
        fwd_num_comm_sms=24,
        bwd_num_comm_sms=40,
        bf16_comm_sms=(24, 40),
        mxfp8_comm_sms=(40, 40),
        shape_comm_sms=(
            ("bf16", 4096, 1024, 10, 56, 40),
            ("bf16", 5120, 2304, 6, 20, 40),
        ),
    )
    p512 = SimpleNamespace(hidden_size=4096, topk=10)
    p384 = SimpleNamespace(hidden_size=5120, topk=6)
    assert split(None, rows, p512, False, 1024) == (56, 40)
    assert split(None, rows, p384, False, 2304) == (20, 40)
    assert split(None, rows, p512, False) == (24, 40)
    assert split(None, rows, p384, False) == (24, 40)
    assert split(None, rows, p384, True, 2304) == (40, 40)
    assert split(None, rows, p384, False, 1024) == (24, 40)
    assert split(None, rows, p512, False, 2304) == (24, 40)
    b200 = MoKConfig(
        fwd_num_comm_sms=40,
        bwd_num_comm_sms=40,
        bf16_comm_sms=(40, 40),
        mxfp8_comm_sms=(40, 40),
        shape_comm_sms=(("bf16", 5120, 2304, 6, 20, 40),),
        imbalance_comm_sms=(24, 28),
        imbalance_threshold=4.0,
    )
    entry = {"loads": (), "split": None}
    functional = SimpleNamespace(
        _load_entry=lambda workspace, create: entry,
        _collect_load=lambda observed: None,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert split(functional, b200, p384, False, 2304) == (20, 40)
    entry["loads"] = [100] * 8
    assert split(functional, b200, p384, False, 2304) == (20, 40)
    assert split(functional, b200, p384, False) == (40, 40)
    entry["loads"] = [560, 100, 100, 40, 0, 0, 0, 0]  # 5.6x the mean
    assert split(functional, b200, p384, False, 2304) == (24, 28)
    assert split(functional, b200, p384, False) == (24, 28)


def test_routed_precision_splits_are_validated():
    """``bf16_comm_sms`` / ``mxfp8_comm_sms`` must each be ``None`` or a
    (forward, backward) tuple of positive, even counts that leave compute SMs;
    ``build_schedule`` rejects any other value before it launches anything."""
    require_gpu()
    ep, experts, topk, tokens = 4, 4, 2, 256
    device = torch.device("cuda", torch.cuda.current_device())
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    functional = backend.MoKFunctional(ep, experts, topk)
    functional.communication = OneGpuGroup()
    capacity = ep * tokens * topk + experts * 256
    workspace = one_gpu_workspace(ep, tokens, 256, topk, capacity, device)
    ids = torch.zeros(tokens, topk, dtype=torch.int32, device=device)
    for name, bad in (
        ("bf16_comm_sms", (16,)),
        ("bf16_comm_sms", [16, 40]),
        ("mxfp8_comm_sms", (15, 40)),
        ("mxfp8_comm_sms", (40, sms)),
        ("mxfp8_comm_sms", (40, 0)),
        ("shape_comm_sms", [("bf16", 4096, 1024, 10, 56, 48)]),
        ("shape_comm_sms", (["bf16", 4096, 1024, 10, 56, 48],)),
        ("shape_comm_sms", (("bf16", 4096, 1024, 10, 56),)),
        ("shape_comm_sms", (("fp8", 4096, 1024, 10, 56, 48),)),
        ("shape_comm_sms", (("bf16", 4096, 0, 10, 56, 48),)),
        ("shape_comm_sms", (("bf16", 4096, 1024, 10, 55, 48),)),
        ("shape_comm_sms", (("bf16", 4096, 1024, 10, 56, sms),)),
    ):
        config = MoKConfig(fwd_num_comm_sms=8, bwd_num_comm_sms=8, **{name: bad})
        with pytest.raises(ValueError, match=name):
            functional.build_schedule(workspace, config, ids, num_local_experts=experts)


def test_comm_split_log_prints_each_split_change(capsys):
    """``MoKConfig.comm_split_log`` prints one stderr line per rank each time the
    imbalance-aware split of the next launch changes; the default is silent and
    the switch never changes the decision itself."""
    require_gpu()
    import dataclasses
    from types import SimpleNamespace

    assert MoKConfig().comm_split_log is False
    quiet = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        imbalance_comm_sms=(6, 6),
        imbalance_threshold=1.5,
    )
    loud = dataclasses.replace(quiet, comm_split_log=True)
    assert loud.comm_split_log is True
    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    # Gathered expert ids for the (4, 4) diagnostic layout, [EP, 256, 2]: every
    # expert receives the same number of routes (every rank owns 512 routed
    # rows, max/mean load 1), then the same ids all on rank 0's experts
    # (max/mean load 4, above the threshold). The observation reads the
    # owner-rank bins the scheduler's count kernel writes.
    balanced = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32)
        % (ep_size * experts)
    ).view(ep_size, 256, 2)
    skewed = balanced % experts
    capacity = ep_size * 256 * 2 + experts * 256
    prefix = "[cake_mok_bf16 rank 0] communication split -> "
    expected_lines = [
        prefix + "(4, 4) (max/mean routed load 1.000)",
        prefix + "(6, 6) (max/mean routed load 4.000)",
    ]
    workspaces = []  # keep every stand-in alive: the adapter keys its state by id
    for config, expected in ((quiet, []), (loud, expected_lines)):
        workspace = SimpleNamespace(ep_rank=0, ep_size=ep_size)
        workspaces.append(workspace)
        capsys.readouterr()
        splits = []
        for routes in (balanced, skewed):
            functional.scheduler(routes, capacity, 0)
            functional._observe_load(workspace, config)
            torch.cuda.synchronize()
            splits.append(functional._comm_split(config, workspace))
            # The same observation again: the split is unchanged, so no line.
            splits.append(functional._comm_split(config, workspace))
        assert splits == [(4, 4), (4, 4), (6, 6), (6, 6)]
        lines = [
            line
            for line in capsys.readouterr().err.splitlines()
            if "communication split" in line
        ]
        assert lines == expected


# ---------------------------------------------------------------------------------------------------------------------
# The router weight before the down projection: host checks (no GPU), then the kernels on SM100a, SM103a and SM107a
# ---------------------------------------------------------------------------------------------------------------------

#: The roles only the router-weight-before-down mode, top-k 6 and the 384-expert layouts load (the fused schedule's
#: ``gather_i64`` and ``count_fused`` included).
FUSED_SCHEDULE_ROLES = (
    "gather_i64",
    *(f"count_fused_{ep}_{experts}" for ep, experts in kernels.SCHEDULER_LAYOUTS_384),
)
BEFORE_DOWN_ROLES = (
    "forward_clamped_wbd",
    "forward_clamped_noctx_wbd",
    "recompute_clamped_wbd",
    "backward_clamped_fp32_wbd",
    "epilogue_backward_6",
    *(
        f"{kind}_{ep}_{experts}"
        for ep, experts in kernels.SCHEDULER_LAYOUTS_384
        for kind in ("count", "rows")
    ),
    *FUSED_SCHEDULE_ROLES,
)
#: The two layouts the requester runs and the worst-case geometry it asked to fit.
BOUND_LAYOUTS = [(64, 6), (8, 48)]
BOUND_IDS = ["ep64-local6", "ep8-local48"]


def _mode_group_roles(group, layouts=None):
    """The roles only the :data:`MODE_GROUPS` group ``group`` loads: its fused kernels (the direct-gradient backward;
    with SiTU-GLU also the forward, the context-free forward and the recompute, all without a shared expert), its
    epilogues, its shared-scale helpers and its layouts' scheduler kernels (only those of ``layouts`` when given)."""
    situ = group.activation == "situ_glu"
    roles = []
    if situ:
        roles += [
            kernels._role("forward", situ_glu=True),
            kernels._role("forward", noctx=True, situ_glu=True),
            kernels._role("recompute", situ_glu=True),
        ]
    roles.append(
        kernels._role(
            "backward",
            False,
            group.fp32_wgrad,
            direct=group.router_weight_gradient == "direct",
            situ_glu=situ,
        )
    )
    if group.shared_expert == "scaled":
        roles += [
            f"epilogue_forward_scaled_{group.topk}",
            f"epilogue_backward_scaled_{group.topk}",
            "shared_scale_rows",
            "shared_scale_dot",
        ]
    elif group.shared_expert == "none":
        roles += [
            f"epilogue_forward_none_{group.topk}",
            f"epilogue_backward_none_{group.topk}",
        ]
    for ep, experts in group.layouts if layouts is None else layouts:
        roles += [f"count_{ep}_{experts}", f"rows_{ep}_{experts}"]
    return tuple(roles)


#: The roles only a :data:`MODE_GROUPS` group loads (the 512-expert top-10 group: the direct-gradient backward, the
#: scaled top-10 epilogues, the two shared-scale helpers and the (8, 64) and (16, 32) scheduler kernels; the
#: 896-expert top-16 group: the SiTU-GLU forward, context-free forward, recompute and direct-gradient backward without
#: a shared expert, the top-16 epilogues without a shared row and the (64, 14) and (32, 28) scheduler kernels).
MODE_GROUP_ROLES = tuple(
    role for group in kernels.MODE_GROUPS.values() for role in _mode_group_roles(group)
)

#: The arguments of a sibling launch that a build without a shared expert does not take: every shared map of the
#: forward, the recompute and the backward, and ``local_tokens``.
NO_SHARED_ARGUMENTS = frozenset(
    {
        *("x_shared", "wg_shared", "wu_shared", "wd_shared", "gate_shared_out", "up_shared_out"),
        *("hidden_shared_out", "hidden_shared_in", "y_shared", "local_tokens"),
        *("dy_s", "dg_s", "du_s", "dy_atb_s", "dg_atb_s", "du_atb_s", "x_atb_s", "h_atb_s", "wg_s", "wu_s"),
        *("wd_s", "dh_s", "dx_s", "dwg_s", "dwu_s", "dwd_s", "dh_sw_s", "gate_sw_s", "up_sw_s", "dg_sw_s"),
        "du_sw_s",
    }
)

#: Per group role with a sibling: ``(sibling role, (argument, added argument after it) or None, arguments dropped,
#: SiTU-GLU)``; a SiTU-GLU role takes ``situ_beta``, ``situ_lb`` in the place of ``swiglu_limit``.
GROUP_SIBLINGS = {
    "backward_fp32_direct": ("backward_fp32", ("dy_routed_ptr", "dy_scaled_ptr"), (), False),
    "epilogue_forward_scaled_10": ("epilogue_forward_8", ("scores", "shared_scale"), (), False),
    "epilogue_backward_scaled_10": ("epilogue_backward_8", ("scores", "shared_scale"), (), False),
    "forward_situ_noshared": ("forward", None, NO_SHARED_ARGUMENTS, True),
    "forward_noctx_situ_noshared": ("forward_noctx", None, NO_SHARED_ARGUMENTS, True),
    "recompute_situ_noshared": ("recompute", None, NO_SHARED_ARGUMENTS, True),
    "backward_fp32_direct_situ_noshared": (
        "backward_fp32",
        ("dy_routed_ptr", "dy_scaled_ptr"),
        NO_SHARED_ARGUMENTS,
        True,
    ),
    "epilogue_forward_none_16": ("epilogue_forward_8", None, ("shared",), False),
    "epilogue_backward_none_16": ("epilogue_backward_8", None, ("shared",), False),
}


def _group_records(registry):
    """Stand-in registry records of the :data:`MODE_GROUP_ROLES` a registry does not hold yet, for the host-only
    tests: the arguments of the sibling role (:data:`GROUP_SIBLINGS`) with the group's additions (``dy_scaled_ptr``
    after ``dy_routed_ptr``, ``shared_scale`` after ``scores``) and removals (the shared maps and ``local_tokens``,
    :data:`NO_SHARED_ARGUMENTS`; the epilogues' ``shared`` row), ``situ_beta``, ``situ_lb`` in the place of
    ``swiglu_limit`` for SiTU-GLU, the scheduler kernels those of the 256-expert ones, the helpers' own lists, each
    group's targets.  A registry that holds a role keeps its record."""
    records = dict(registry)
    own = {
        "shared_scale_rows": ["rows", "scales", "scaled", "tokens", "width"],
        "shared_scale_dot": ["left", "right", "dots", "tokens", "width"],
    }
    for group in kernels.MODE_GROUPS.values():
        targets = list(group.targets)
        for role in _mode_group_roles(group):
            if role in records:
                continue
            if role in own:
                arguments = [*own[role], "grid_x", "grid_y", "grid_z"]
            elif role.startswith(("count_", "rows_")):
                arguments = list(registry[f"{role.split('_')[0]}_8_32"]["arguments"])
            else:
                base, added, dropped, situ = GROUP_SIBLINGS[role]
                arguments = list(registry[base]["arguments"])
                if added is not None:
                    arguments.insert(arguments.index(added[0]) + 1, added[1])
                arguments = [name for name in arguments if name not in dropped]
                if situ:
                    at = arguments.index("swiglu_limit")
                    arguments[at : at + 1] = ["situ_beta", "situ_lb"]
            records[role] = {
                "library": f"cake_mok_{role}",
                "arches": targets,
                "arguments": arguments,
            }
    return records


def _stub_loader(monkeypatch, loaded=None):
    """``jit.load_kernel`` stand-in: a kernel with its registry record's argument names whose ``run`` records the
    role and the arguments by name; nothing is compiled or launched.  Returns the record list."""
    registry = jit._registry()
    calls = []

    class Stub:
        def __init__(self, role):
            self.role = role

        def run(self, *args):
            names = registry[self.role]["arguments"]
            calls.append((self.role, dict(zip(names, args, strict=True))))

    def load(role):
        if loaded is not None:
            loaded.append(role)
        return jit._Kernel(Stub(role), registry[role]["arguments"])

    monkeypatch.setattr(jit, "load_kernel", load)
    return calls


def test_before_down_roles_are_registry_records():
    """Every role a launcher of either router-weight position can load is a registry record, with the suffixes in
    the registry's order; the router-weight-before-down, top-k 6 and 384-expert roles are built for exactly the
    targets of ``BEFORE_DOWN_TARGETS`` (SM100a, SM103a and SM107a), every other such role for exactly SM100a, SM103a
    and SM107a, and top-k 6 has no weighted forward epilogue."""
    registry = jit._registry()
    role = kernels._role
    assert role("forward", True, noctx=True) == "forward_clamped_noctx"
    assert role("forward", True, noctx=True, weight_before_down=True) == "forward_clamped_noctx_wbd"
    assert role("backward", True, True, weight_before_down=True) == "backward_clamped_fp32_wbd"
    names = {"expert_layout", "gather", "barrier", "zero"}
    for clamped in (False, True):
        names |= {
            role("forward", clamped),
            role("forward", clamped, noctx=True),
            role("recompute", clamped),
            role("backward", clamped),
            role("backward", clamped, True),
        }
    names |= {
        role("forward", True, weight_before_down=True),
        role("forward", True, noctx=True, weight_before_down=True),
        role("recompute", True, weight_before_down=True),
        role("backward", True, True, weight_before_down=True),
    }
    for ep, experts in SCHEDULER_LAYOUTS:
        if not any(
            (ep, experts) in group.layouts for group in kernels.MODE_GROUPS.values()
        ):
            names |= {f"count_{ep}_{experts}", f"rows_{ep}_{experts}", f"pad_{ep}"}
    names |= {f"epilogue_backward_{k}" for k in kernels.EPILOGUE_TOPKS}
    names |= {f"epilogue_forward_{k}" for k in kernels.WEIGHTED_EPILOGUE_TOPKS}
    names |= set(FUSED_SCHEDULE_ROLES)
    assert names <= set(registry), sorted(names - set(registry))
    assert set(BEFORE_DOWN_ROLES) <= names
    for name in BEFORE_DOWN_ROLES:
        assert registry[name]["arches"] == list(kernels.BEFORE_DOWN_TARGETS), name
    for name in sorted(names - set(BEFORE_DOWN_ROLES)):
        assert registry[name]["arches"] == ["sm_100a", "sm_103a", "sm_107a"], name
    assert "epilogue_forward_6" not in registry
    assert set(kernels.BEFORE_DOWN_LAYOUTS) <= set(SCHEDULER_LAYOUTS)
    assert registry["backward_clamped_fp32_wbd"]["arguments"] == registry["backward_clamped_fp32"]["arguments"]
    assert registry["epilogue_backward_6"]["arguments"] == registry["epilogue_backward_8"]["arguments"]


def test_fused_schedule_launches_pass_exactly_the_registry_arguments(monkeypatch):
    """Host only (stub kernels, no launch): a 384-expert layout's scheduler loads its ``count_fused`` kernel, and a
    fused plan hands ``count_fused`` and rows exactly their registry records' arguments over one schedule buffer split
    at the unfused plan's views: no template, the ``partials`` scratch (int32 ``EP * 32 * (local experts + EP)``) and
    the zeroed ``done`` word as a uint32 view; the owner-rank bins are the counts' tail. ``gather_i64_plan`` binds the
    gathered buffer's geometry and takes the caller's int64 ids and their valid slot count per call; EP 1 has no
    plan. A layout without a fused kernel refuses a fused plan."""
    calls = _stub_loader(monkeypatch)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    monkeypatch.setattr(jit, "target_arch", lambda device=None: "sm_103a")
    ep, experts, topk, tokens, rank = 8, 48, 6, 512, 3
    capacity = ep * tokens * topk + experts * 256
    gathered = torch.zeros(ep, tokens, topk, dtype=torch.int32)
    monkeypatch.setattr(MoKScheduler, "_check", lambda self, ids, size, r: (ep, tokens, topk))
    scheduler = MoKScheduler(ep, experts)
    assert scheduler.count_fused is not None
    plan = scheduler.plan(gathered, capacity, rank, fused=True)
    unfused = scheduler.plan(gathered, capacity, rank)
    assert plan.template is None and (plan.count, plan.pad, plan.layout) == (None, None, None)
    assert (plan.slab, plan.layout_range, plan.total) == (unfused.slab, unfused.layout_range, unfused.total)
    assert plan.partials.dtype == torch.int32 and plan.partials.numel() == ep * 32 * (experts + ep)
    assert plan.done.dtype == torch.int32 and plan.done.tolist() == [0]
    calls.clear()
    peer_rank, peer_token, num_tokens, per_expert, layout = scheduler.launch_planned(plan)
    assert [role for role, _ in calls] == [f"count_fused_{ep}_{experts}", f"rows_{ep}_{experts}"]
    fused, rows = calls[0][1], calls[1][1]
    (c0, c1), (n0, n1), (e0, e1) = plan.slab
    start, end = plan.layout_range
    base = peer_rank.untyped_storage().data_ptr()
    for view, at, size in (
        (peer_token, 0, capacity),
        (peer_rank, capacity, capacity),
        (fused["counts"], c0, c1 - c0),
        (num_tokens, n0, n1 - n0),
        (per_expert, e0, e1 - e0),
        (layout, start, end - start),
    ):
        assert view.untyped_storage().data_ptr() == base
        assert (view.storage_offset(), view.numel()) == (at, size)
    assert fused["topk_all"] is gathered and fused["partials"] is plan.partials
    assert fused["done"].dtype == torch.uint32 and fused["done"].data_ptr() == plan.done.data_ptr()
    assert fused["schedule_peer_rank"] is peer_rank and fused["layout"] is layout
    assert fused["tokens_per_expert"] is per_expert and fused["num_tokens"] is num_tokens
    assert (fused["local_tokens"], fused["top_k"], fused["capacity"], fused["rank"]) == (tokens, topk, capacity, rank)
    assert (fused["grid_x"], fused["grid_y"], fused["grid_z"]) == (ep * 32, 1, 1)
    assert rows["counts"] is fused["counts"] and rows["schedule_peer_rank"] is peer_rank
    assert rows["schedule_peer_token_idx"] is peer_token and rows["tokens_per_expert"] is per_expert
    assert scheduler.owner_loads().data_ptr() == fused["counts"][experts * ep :].data_ptr()
    scheduler.release_counts()

    communication = kernels.MoKCommunication()
    monkeypatch.setattr(kernels.MoKCommunication, "_check_gather", lambda self, routes, out, address, r: len(out))
    gather = communication.gather_i64_plan(gathered, 4096, rank)
    ids = torch.zeros(100, topk, dtype=torch.int64)
    gather.launch((ids, 100 * topk))
    role, arguments = calls[-1]
    assert role == "gather_i64" and arguments["local_ids"] is ids
    assert {name: value for name, value in arguments.items() if name != "local_ids"} == {
        "multicast_address": 4096,
        "rank": rank,
        "numel": tokens * topk,
        "valid": 100 * topk,
        "grid_x": tokens * topk * 4 // 2048,
        "grid_y": 1,
        "grid_z": 1,
    }
    assert communication.gather_i64_plan(torch.zeros(1, tokens, topk, dtype=torch.int32), 4096, 0) is None
    other = MoKScheduler(8, 32)
    assert other.count_fused is None
    with pytest.raises(ValueError, match="no fused count kernel"):
        other.plan(gathered, capacity, rank, fused=True)


def test_before_down_launchers_pass_exactly_the_registry_arguments(monkeypatch):
    """Host only (stub kernels, no launch): the router-weight-before-down forward (context and context-free),
    recompute and backward hand the kernel exactly its registry record's arguments through ``__call__`` and through
    a plan.  ``weight_peers`` is the uint64 table of the ranks' router-weight buffers and ``row_weights`` an FP32
    ``[macro_size]`` ring per call; ``weight_ptrs`` is required there and refused by the other launchers."""
    calls = _stub_loader(monkeypatch)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    tokens, hidden, intermediate, experts, topk, macro, mini, comm, ep, limit = (
        512, 256, 256, 4, 2, 768, 256, 4, 4, 10.0,
    )  # fmt: skip
    capacity = 4096
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(tokens, hidden, **options)
    combine = torch.zeros(tokens * topk, hidden, **options)
    ptrs = [4096 * (rank + 1) for rank in range(ep)]  # never dereferenced
    weight_ptrs = [65536 * (rank + 1) for rank in range(ep)]
    shared = [
        torch.zeros(intermediate, hidden, **options),
        torch.zeros(intermediate, hidden, **options),
        torch.zeros(hidden, intermediate, **options),
    ]
    routed = [
        torch.zeros(experts, intermediate, hidden, **options),
        torch.zeros(experts, intermediate, hidden, **options),
        torch.zeros(experts, hidden, intermediate, **options),
    ]
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    schedule = (
        torch.full((capacity,), -1, dtype=torch.int32),
        torch.zeros(capacity, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(experts, dtype=torch.int32),
    )
    layout = torch.zeros(3 * experts + capacity // 256, dtype=torch.int32)

    def check_router_weights(expected_calls):
        assert len(calls) == expected_calls
        for role, arguments in calls:
            assert role.endswith("_wbd"), role
            peers, ring = arguments["weight_peers"], arguments["row_weights"]
            assert peers.dtype == torch.uint64, role
            assert peers.view(torch.int64).tolist() == weight_ptrs, role
            assert ring.dtype == torch.float32 and tuple(ring.shape) == (macro,), role
        calls.clear()

    for context in (True, False):
        forward = MoKForward(True, context=context, weight_before_down=True)
        with pytest.raises(ValueError, match="need weight_ptrs"):
            forward(
                x, ptrs, combine, ptrs, *weights, *schedule, topk, limit, comm, macro, mini, layout=layout,
            )  # fmt: skip
        forward(
            x, ptrs, combine, ptrs, *weights, *schedule, topk, limit, comm, macro, mini, layout=layout,
            weight_ptrs=weight_ptrs,
        )  # fmt: skip
        plan = forward.plan(
            x, ptrs, ptrs, intermediate, experts, capacity, topk, limit, comm, macro, mini, weight_ptrs=weight_ptrs,
        )  # fmt: skip
        MoKForward.launch_planned(plan, *weights, *schedule[:3], layout)
        check_router_weights(2)
    recompute = MoKRecompute(True, weight_before_down=True)
    recompute(
        x, ptrs, shared[0], routed[0], shared[1], routed[1], *schedule, topk, limit, comm, macro, mini,
        layout=layout, weight_ptrs=weight_ptrs,
    )  # fmt: skip
    plan = recompute.plan(
        x, ptrs, intermediate, experts, capacity, topk, limit, comm, macro, mini, weight_ptrs=weight_ptrs,
    )  # fmt: skip
    MoKRecompute.launch_planned(plan, shared[0], routed[0], shared[1], routed[1], *schedule[:3], layout)
    check_router_weights(2)
    backward = MoKBackward(True, True, weight_before_down=True)
    accumulators = [torch.zeros(w.shape) for w in (*shared, *routed)]
    scores = torch.zeros(tokens, topk)
    ring = torch.zeros(macro, intermediate, **options)
    saved = (torch.zeros(macro, hidden, **options), x[:, :intermediate].clone(), ring, x[:, :intermediate].clone(),
             ring, x[:, :intermediate].clone(), ring)  # fmt: skip
    backward(
        x, ptrs, combine, ptrs, scores, weight_ptrs, scores, ptrs, *weights, *saved, x, ptrs, *schedule,
        topk, limit, comm, macro, mini, weight_grad_accumulators=accumulators, layout=layout,
    )  # fmt: skip
    plan = backward.plan(
        x, ptrs, ptrs, weight_ptrs, ptrs, ptrs, intermediate, experts, capacity, topk, limit, comm, macro, mini,
    )  # fmt: skip
    backward.launch_planned(
        plan, *weights, *saved, x, *schedule, layout, weight_grad_accumulators=accumulators,
    )  # fmt: skip
    assert [role for role, _ in calls] == ["backward_clamped_fp32_wbd"] * 2
    for _, arguments in calls:
        assert arguments["weight_peers"].view(torch.int64).tolist() == weight_ptrs
    calls.clear()
    # Every other launcher refuses the router-weight pointers; the mode needs the clamped (and FP32) builds.
    with pytest.raises(ValueError, match="read only by"):
        MoKForward(True)(
            x, ptrs, combine, ptrs, *weights, *schedule, topk, limit, comm, macro, mini, layout=layout,
            weight_ptrs=weight_ptrs,
        )  # fmt: skip
    with pytest.raises(ValueError, match="read only by"):
        MoKRecompute(True).plan(
            x, ptrs, intermediate, experts, capacity, topk, limit, comm, macro, mini, weight_ptrs=weight_ptrs,
        )  # fmt: skip
    assert calls == []
    with pytest.raises(ValueError, match="clamped SwiGLU only"):
        MoKForward(False, weight_before_down=True)
    with pytest.raises(ValueError, match="clamped SwiGLU only"):
        MoKRecompute(False, weight_before_down=True)
    with pytest.raises(ValueError, match="FP32 weight-gradient accumulation only"):
        MoKBackward(True, False, weight_before_down=True)
    with pytest.raises(TypeError, match="weight_before_down must be a boolean"):
        MoKForward(True, weight_before_down=1)


def _source_split_launcher_setup(monkeypatch):
    """Stub kernels (no launch) and the arguments of a router-weight-before-down forward at EP4, 4 forward
    communication SMs."""
    calls = _stub_loader(monkeypatch)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    tokens, hidden, intermediate, experts, topk, macro, mini, comm, ep, limit = (
        512, 256, 256, 4, 2, 768, 256, 4, 4, 10.0,
    )  # fmt: skip
    capacity = 4096
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(tokens, hidden, **options)
    combine = torch.zeros(tokens * topk, hidden, **options)
    ptrs = [4096 * (rank + 1) for rank in range(ep)]  # never dereferenced
    weight_ptrs = [65536 * (rank + 1) for rank in range(ep)]
    shared = [
        torch.zeros(intermediate, hidden, **options),
        torch.zeros(intermediate, hidden, **options),
        torch.zeros(hidden, intermediate, **options),
    ]
    routed = [torch.zeros(experts, intermediate, hidden, **options) for _ in range(2)]
    routed.append(torch.zeros(experts, hidden, intermediate, **options))
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    schedule = (
        torch.full((capacity,), -1, dtype=torch.int32),
        torch.zeros(capacity, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(experts, dtype=torch.int32),
    )
    layout = torch.zeros(3 * experts + capacity // 256, dtype=torch.int32)
    call = (x, ptrs, combine, ptrs, *weights, *schedule, topk, limit, comm, macro, mini)
    plan = (x, ptrs, ptrs, intermediate, experts, capacity, topk, limit, comm, macro, mini)
    return calls, call, plan, weights, schedule, layout, weight_ptrs


@pytest.mark.parametrize("context", [True, False], ids=["context", "context-free"])
def test_source_split_launcher_arguments(monkeypatch, context):
    """Host only (stub kernels, no launch): the router-weight-before-down forward hands the kernel ``ep_size``,
    ``rich_comm_sms``, ``rich_threshold`` (1/256 steps) and ``owner_routes``.  Without a split the count in force
    goes in as the rich count and ``num_tokens`` stands in for the route totals (never read); an unplanned call
    with a split but without the totals keeps the count in force; a plan with a split needs the schedule's int32
    ``[ep_size]`` totals at every launch.  Malformed splits and totals are refused before any launch, and the other
    launchers refuse the split."""
    calls, call, plan_args, weights, schedule, layout, weight_ptrs = _source_split_launcher_setup(monkeypatch)
    forward = MoKForward(True, context=context, weight_before_down=True)
    routes = torch.tensor([7, 0, 3, 5], dtype=torch.int32)

    def launched(**expected):
        ((role, arguments),) = calls
        calls.clear()
        assert role.endswith("_wbd"), role
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                got = arguments[name]
                assert got.data_ptr() == value.data_ptr() and torch.equal(got, value), name
            else:
                assert arguments[name] == value, (name, arguments[name])

    forward(*call, layout=layout, weight_ptrs=weight_ptrs)
    launched(ep_size=4, rich_comm_sms=4, rich_threshold=384, owner_routes=schedule[2])
    forward(*call, layout=layout, weight_ptrs=weight_ptrs, source_split=(2, 300))
    launched(ep_size=4, rich_comm_sms=4, rich_threshold=300, owner_routes=schedule[2])
    forward(*call, layout=layout, weight_ptrs=weight_ptrs, source_split=(2, 300), owner_routes=routes)
    launched(ep_size=4, rich_comm_sms=2, rich_threshold=300, owner_routes=routes)
    plan = forward.plan(*plan_args, weight_ptrs=weight_ptrs)
    MoKForward.launch_planned(plan, *weights, *schedule[:3], layout)
    launched(ep_size=4, rich_comm_sms=4, rich_threshold=384, owner_routes=schedule[2])
    plan = forward.plan(*plan_args, weight_ptrs=weight_ptrs, source_split=(2, 257))
    with pytest.raises(ValueError, match="owner_routes must be"):
        MoKForward.launch_planned(plan, *weights, *schedule[:3], layout)
    with pytest.raises(ValueError, match="owner_routes must be"):
        MoKForward.launch_planned(plan, *weights, *schedule[:3], layout, routes.long())
    MoKForward.launch_planned(plan, *weights, *schedule[:3], layout, routes)
    launched(ep_size=4, rich_comm_sms=2, rich_threshold=257, owner_routes=routes)
    for split in ((3, 384), (6, 384), (0, 384), (2, 256), (2, 2**31), (2.0, 384), (2, 384.0)):
        with pytest.raises(ValueError, match="source_split must be"):
            forward(*call, layout=layout, weight_ptrs=weight_ptrs, source_split=split)
        with pytest.raises(ValueError, match="source_split must be"):
            forward.plan(*plan_args, weight_ptrs=weight_ptrs, source_split=split)
    with pytest.raises(ValueError, match="owner_routes must be"):
        forward(*call, layout=layout, weight_ptrs=weight_ptrs, source_split=(2, 384), owner_routes=routes[:3])
    with pytest.raises(ValueError, match="read only by"):
        MoKForward(True, context=context)(*call, layout=layout, source_split=(2, 384))
    with pytest.raises(ValueError, match="read only by"):
        MoKForward(True, context=context)(*call, layout=layout, owner_routes=routes)
    with pytest.raises(ValueError, match="read only by"):
        MoKForward(True, context=context).plan(*plan_args, source_split=(2, 384))
    assert calls == []


def test_source_split_follows_the_config():
    """Host only: the adapter's source split of a forward launch: off (the forward count in force, the threshold,
    ``num_tokens`` in place of the totals) for the router weight after the down projection, without a count, with a
    count not below the forward count in force and for a schedule without totals; on, the count, the threshold in
    1/256 steps (rounded) and the schedule's totals; totals of another shape, type or device are refused."""
    from types import SimpleNamespace

    functional = object.__new__(backend.MoKFunctional)
    device = torch.device("cpu")
    workspace = SimpleNamespace(ep_size=4, device=SimpleNamespace(index=-1))
    routes = torch.tensor([7, 0, 3, 5], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, dtype=torch.int32, device=device)
    schedule = SimpleNamespace(owner_routes=routes, num_tokens=tokens)
    on = dataclasses.replace(MoKConfig(), source_split_comm_sms=16, source_split_threshold=1.25)
    functional.weight_before_down = False
    assert functional._source_split(on, workspace, schedule, 24) == (24, 320, tokens)
    functional.weight_before_down = True
    off = dataclasses.replace(on, source_split_comm_sms=None)
    assert functional._source_split(off, workspace, schedule, 24) == (24, 320, tokens)
    assert functional._source_split(on, workspace, schedule, 16) == (16, 320, tokens)
    no_routes = SimpleNamespace(owner_routes=None, num_tokens=tokens)
    assert functional._source_split(on, workspace, no_routes, 24) == (24, 320, tokens)
    rich, threshold, totals = functional._source_split(on, workspace, schedule, 24)
    assert (rich, threshold) == (16, 320) and totals is routes
    assert functional._source_split(
        dataclasses.replace(on, source_split_threshold=1.5009), workspace, schedule, 24
    )[1] == 384
    for bad in (routes[:3], routes.long(), routes.float()):
        with pytest.raises(ValueError, match="owner_routes must be"):
            functional._source_split(
                on, workspace, SimpleNamespace(owner_routes=bad, num_tokens=tokens), 24
            )
    with pytest.raises(ValueError, match="owner_routes must be"):
        functional._source_split(on, SimpleNamespace(ep_size=4, device=SimpleNamespace(index=0)), schedule, 24)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_split_comm_sms", 0),
        ("source_split_comm_sms", 3),
        ("source_split_comm_sms", -2),
        ("source_split_comm_sms", 148),
        ("source_split_comm_sms", 16.0),
        ("source_split_comm_sms", True),
        ("source_split_threshold", 1.0),
        ("source_split_threshold", 1.001),
        ("source_split_threshold", 0.5),
        ("source_split_threshold", 65536.5),
        ("source_split_threshold", float("inf")),
        ("source_split_threshold", float("nan")),
        ("source_split_threshold", "1.5"),
    ],
)
def test_source_split_config_is_validated(field, value):
    """Host only: the schedule check refuses a source-split count that is not a positive, even count leaving
    compute SMs and a threshold that is not finite, above 1.0 in 1/256 steps and at most 65536; the default and
    in-range values pass."""
    functional = object.__new__(backend.MoKFunctional)
    functional._schedule_configs, functional._sm_count = {}, {0: 148}
    device = torch.device("cuda", 0)
    with pytest.raises(ValueError, match=field):
        functional._check_schedule_config(dataclasses.replace(MoKConfig(), **{field: value}), device)
    for good in (
        MoKConfig(),
        dataclasses.replace(MoKConfig(), source_split_comm_sms=16, source_split_threshold=1.002),
        dataclasses.replace(MoKConfig(), source_split_comm_sms=2, source_split_threshold=65536),
        dataclasses.replace(MoKConfig(), source_split_comm_sms=146, source_split_threshold=4),
    ):
        functional._check_schedule_config(good, device)


@pytest.mark.parametrize(
    "capability,sm_count,ep_size,expected",
    [
        ((10, 0), 148, 8, 16),  # B200, EP8
        ((10, 0), 148, 4, None),
        ((10, 0), 152, 4, 16),  # GB200, EP4
        ((10, 0), 152, 8, None),
        ((10, 3), 148, 8, None),  # B300, EP8: measured, no gain (README)
        ((10, 3), 148, 4, None),
        ((10, 3), 152, 4, 16),  # GB300, EP4
        ((10, 3), 152, 8, None),
        ((10, 3), 160, 4, None),  # other SM count
        ((10, 7), 212, 8, None),  # Rubin: no row
        ((12, 0), 170, 4, None),
    ],
)
def test_source_split_auto_follows_the_device_and_group(monkeypatch, capability, sm_count, ep_size, expected):
    """Host-only: ``"auto"`` for the source split resolves through the per-device, per-group-size table; devices and
    group sizes without a measured row keep the split off (``None``)."""
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sm_count),
    )
    assert api.resolve_source_split_comm_sms(ep_size) == expected


def test_source_split_defaults_resolve_the_measured_rows():
    """Host-only: the config default keeps the source split off, the facade default ``"auto"`` resolves the
    measured row (threshold 1.5), the measured rows are the shipped table (a change here is a shipping change) and
    malformed arguments are rejected before any collective."""
    import inspect

    config = MoKConfig()
    assert config.source_split_comm_sms is None
    assert config.source_split_threshold == api.SOURCE_SPLIT_THRESHOLD == 1.5
    facade = inspect.signature(api.create_mok_bf16_workspace).parameters
    assert facade["source_split_comm_sms"].default == "auto"
    assert facade["source_split_threshold"].default == 1.5
    assert kernels.SOURCE_SPLIT_UNIT == 256
    assert api.SOURCE_SPLIT_COMM_SMS == {
        (10, 0): {148: {8: 16}, 152: {4: 16}},
        (10, 3): {152: {4: 16}},
    }
    with pytest.raises(ValueError, match="ep_size must be a positive"):
        api.resolve_source_split_comm_sms(0)
    kwargs = dict(group=None, device=None, num_local_tokens=512, hidden_size=256, topk=2)
    with pytest.raises(ValueError, match='"auto"'):
        api.create_mok_bf16_workspace(source_split_comm_sms="fast", **kwargs)


def test_before_down_call_refusals_come_before_device_work(monkeypatch):
    """Host only: with ``router_weight_position="before_down"`` plain SwiGLU (``swiglu_limit=None``), MXFP8 routed
    weights and a recompute without the router weights are refused by every operation and ``prepare`` before any
    copy, barrier or kernel load."""
    functional = object.__new__(backend.MoKFunctional)
    functional.local_experts = 4
    functional._validated = {}
    functional.router_weight_position = "before_down"
    functional.weight_before_down = True
    functional._forward_kernels, functional._backward_kernels = {}, {}
    functional._recompute_kernels, functional._context_free_forward_kernels = {}, {}
    events = []
    _stub_loader(monkeypatch, events)
    monkeypatch.setattr(
        backend.MoKFunctional, "_inputs", staticmethod(lambda *args: (None, 256))
    )
    monkeypatch.setattr(functional, "_check_device", lambda workspace: None)
    monkeypatch.setattr(functional, "_barrier", lambda w: events.append("barrier"))
    monkeypatch.setattr(
        backend.MoKFunctional,
        "_copy_source",
        staticmethod(lambda *args: events.append("copy")),
    )
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(256, 256, **options)
    shared = [torch.zeros(256, 256, **options) for _ in range(3)]
    routed = [torch.zeros(4, 256, 256, **options) for _ in range(3)]
    data = torch.zeros(4, 256, 256, dtype=torch.uint8)
    mxfp8 = [(data, torch.zeros(64, dtype=torch.uint8)) for _ in range(3)]
    scores = torch.zeros(256, 2)
    schedule = object()
    context = backend.MoKForwardContext(
        x_routed=x, gate_shared=x, gate_routed=x, up_shared=x, up_routed=x, hidden_shared=x, hidden_routed=x,
        schedule=schedule,
    )  # fmt: skip
    plain, needs_bf16 = "requires swiglu_limit", "takes BF16 routed weights"
    with pytest.raises(ValueError, match=plain):
        functional.forward(None, None, schedule, x, scores, *shared, *routed)
    with pytest.raises(ValueError, match=needs_bf16):
        functional.forward(None, None, schedule, x, scores, *shared, *mxfp8, swiglu_limit=10.0)
    with pytest.raises(ValueError, match=plain):
        functional.recompute_forward_context(
            None, None, schedule, x, shared[0], shared[1], routed[0], routed[1], router_weights=scores
        )
    with pytest.raises(ValueError, match="needs router_weights"):
        functional.recompute_forward_context(
            None, None, schedule, x, shared[0], shared[1], routed[0], routed[1], 10.0
        )
    with pytest.raises(ValueError, match=plain):
        functional.backward(None, None, schedule, context, x, x, scores, *shared, *routed)
    with pytest.raises(ValueError, match=needs_bf16):
        functional.backward(
            None, None, schedule, context, x, x, scores, *shared, *mxfp8, swiglu_limit=10.0
        )
    for prepare in (
        lambda: functional.prepare(None),
        lambda: functional.prepare(10.0, True),
        lambda: functional.prepare(None, save_context=False),
        lambda: functional.prepare_recompute(None),
        lambda: functional.prepare_recompute(10.0, True),
    ):
        with pytest.raises(ValueError, match="router_weight_position='before_down'"):
            prepare()
    assert events == []


def test_before_down_is_accepted_on_sm107a(monkeypatch):
    """Host only: ``"before_down"`` has SM100a, SM103a and SM107a builds; ``check_target`` accepts both positions on
    each of them and refuses ``"before_down"`` on any other target with a message that names the three.  On an SM107a
    device the adapter prepares the requester's EP64 top-6 layout, an EP8 384-expert layout and a diagnostic top-2
    layout, and every kernel it loads has an sm_107a build."""
    assert kernels.BEFORE_DOWN_TARGETS == ("sm_100a", "sm_103a", "sm_107a")
    for arch in kernels.BEFORE_DOWN_TARGETS:
        kernels.check_target("before_down", arch)
        kernels.check_target("after_down", arch)
    with pytest.raises(ValueError, match="no sm_90a build .* use an sm_100a, sm_103a or sm_107a device"):
        kernels.check_target("before_down", "sm_90a")
    registry = jit._registry()
    loaded = []
    _stub_loader(monkeypatch, loaded)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(jit, "target_arch", lambda device=None: "sm_107a")
    for ep, experts, topk in ((64, 6, 6), (8, 48, 6), (1, 4, 2)):
        loaded.clear()
        functional = backend.MoKFunctional(
            ep, experts, topk, clamped_swiglu=True, fp32_wgrad=True, router_weight_position="before_down"
        )
        assert functional.weight_before_down
        expected = {"forward_clamped_wbd", "backward_clamped_fp32_wbd", f"count_{ep}_{experts}", f"rows_{ep}_{experts}"}
        assert expected | {f"epilogue_backward_{topk}"} <= set(loaded), sorted(loaded)
        assert [role for role in loaded if "sm_107a" not in registry[role]["arches"]] == []


def test_top6_epilogues_have_the_unweighted_sum_only(monkeypatch):
    """Host only: top-k 6 loads the unweighted sum alone and refuses a weighted forward; ``weighted=False`` loads the
    sum alone for every top-k; top-k 2 and 8 still load both."""
    loaded = []
    _stub_loader(monkeypatch, loaded)
    epilogues = MoKEpilogues(6)
    assert loaded == ["epilogue_backward_6"] and epilogues.forward_module is None
    shared = torch.zeros(4, 256, dtype=torch.bfloat16)
    routed = torch.zeros(24, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="no weighted form"):
        epilogues.forward(shared, routed, torch.zeros(4, 6))
    with pytest.raises(ValueError, match="no weighted form"):
        epilogues.plan(routed, torch.zeros(4, 6))
    with pytest.raises(ValueError, match="only the unweighted epilogue"):
        MoKEpilogues(6, weighted=True)
    with pytest.raises(ValueError, match="top_k=2, 6 or 8"):
        MoKEpilogues(4)
    for topk in (2, 8):
        loaded.clear()
        MoKEpilogues(topk, weighted=False)
        assert loaded == [f"epilogue_backward_{topk}"]
        loaded.clear()
        MoKEpilogues(topk)
        assert loaded == [f"epilogue_forward_{topk}", f"epilogue_backward_{topk}"]


def _group_stub_loader(monkeypatch, loaded=None):
    """:func:`_stub_loader` over the registry with the :func:`_group_records` stand-ins of the group roles it does not
    hold yet."""
    records = _group_records(jit._registry())
    monkeypatch.setattr(jit, "_registry", lambda: records)
    return _stub_loader(monkeypatch, loaded)


def test_mode_group_roles_are_registry_records():
    """Every role of a :data:`MODE_GROUPS` group is a registry record built for exactly the group's targets (SM100a,
    SM103a and SM107a), with the arguments the launchers pass: the direct-gradient backward takes those of
    ``backward_fp32`` and ``dy_scaled_ptr``, the scaled epilogues those of the top-8 epilogues and ``shared_scale``, the
    SiTU-GLU kernels those of their SwiGLU siblings without the shared maps and ``local_tokens`` and with ``situ_beta``,
    ``situ_lb`` in the place of ``swiglu_limit``, the top-16 epilogues those of the top-8 ones without the ``shared``
    row, the 512- and 896-expert scheduler kernels those of the 256-expert ones, and the two shared-scale helpers their
    own lists."""
    registry = jit._registry()
    missing = [role for role in MODE_GROUP_ROLES if role not in registry]
    assert missing == [], f"group roles not in this package's registry: {missing}"
    expected = _group_records(
        {
            role: record
            for role, record in registry.items()
            if role not in MODE_GROUP_ROLES
        }
    )
    for group in kernels.MODE_GROUPS.values():
        for role in _mode_group_roles(group):
            assert registry[role]["arches"] == list(group.targets), role
            assert registry[role]["arguments"] == expected[role]["arguments"], role


GROUP_OK = dict(
    clamped_swiglu=False,
    fp32_wgrad=True,
    router_weight_position="after_down",
    router_weight_gradient="direct",
    shared_expert="scaled",
    activation="swiglu",
)


@pytest.mark.parametrize(
    "layout,topk,change",
    [
        ((8, 64), 8, {}),
        ((8, 32), 10, {}),
        ((8, 64), 10, dict(router_weight_gradient="quotient")),
        ((8, 64), 10, dict(shared_expert="unscaled")),
        ((8, 64), 10, dict(fp32_wgrad=False)),
        ((8, 64), 10, dict(clamped_swiglu=True)),
        ((8, 64), 10, dict(activation="clamped_swiglu")),
        ((8, 64), 10, dict(router_weight_position="before_down")),
        ((8, 32), 8, {}),
        ((8, 32), 8, dict(shared_expert="unscaled")),
        ((8, 32), 8, dict(router_weight_gradient="quotient")),
        (
            (64, 6),
            6,
            dict(
                router_weight_position="before_down",
                clamped_swiglu=True,
                shared_expert="unscaled",
            ),
        ),
    ],
    ids=[
        "512-top8",
        "256-top10",
        "512-quotient",
        "512-unscaled",
        "512-bf16-wgrad",
        "512-clamped",
        "512-clamped-act",
        "512-before-down",
        "256-direct-scaled",
        "256-direct",
        "256-scaled",
        "384-before-down-direct",
    ],
)
def test_mode_group_configurations_outside_the_group_are_refused(layout, topk, change):
    """Host only: a configuration that uses a layout, the top-k or a mode of the 512-expert top-10 group but is not
    the group's configuration raises ``ValueError`` naming the group's configuration and the one given."""
    with pytest.raises(
        ValueError,
        match=r"512 routed experts at top-k 10 .* runs \(EP, local experts\) \(8, 64\), \(16, 32\) at topk=10",
    ):
        kernels.check_configuration(*layout, topk, **{**GROUP_OK, **change})


def test_mode_group_configuration_is_checked_before_device_work(monkeypatch):
    """Host only: ``check_configuration`` admits the group's configuration and returns its name, and the default
    modes keep every existing configuration (returning None); ``check_target`` accepts the group on SM100a, SM103a
    and SM107a and refuses it elsewhere with a message naming the three.  On each of the three, at each of the
    group's layouts, the adapter loads the group's roles (the direct-gradient backward, the scaled top-10 epilogues,
    the shared-scale helpers, the scheduler kernels of that layout) and none of the quotient backward, the unscaled
    epilogues or another layout's scheduler kernels."""
    group = kernels.MODE_GROUPS["scaled_top10"]
    assert group.targets == ("sm_100a", "sm_103a", "sm_107a")
    assert group.layouts == ((8, 64), (16, 32)) and group.topk == 10
    for layout in group.layouts:
        assert kernels.check_configuration(*layout, 10, **GROUP_OK) == "scaled_top10"
    defaults = dict(
        clamped_swiglu=False, fp32_wgrad=False, router_weight_position="after_down"
    )
    assert kernels.check_configuration(8, 32, 8, **defaults) is None
    assert (
        kernels.check_configuration(8, 32, 8, **defaults, activation="clamped_swiglu")
        is None
    )
    for arch in group.targets:
        kernels.check_target("after_down", arch, "scaled_top10")
    with pytest.raises(
        ValueError,
        match="no sm_90a build in this package; use an sm_100a, sm_103a or sm_107a device",
    ):
        kernels.check_target("after_down", "sm_90a", "scaled_top10")
    loaded = []
    _group_stub_loader(monkeypatch, loaded)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    keywords = dict(
        fp32_wgrad=True, router_weight_gradient="direct", shared_expert="scaled"
    )
    for arch in group.targets:
        monkeypatch.setattr(jit, "target_arch", lambda device=None, arch=arch: arch)
        for ep, experts in group.layouts:
            loaded.clear()
            functional = backend.MoKFunctional(ep, experts, 10, **keywords)
            assert (
                functional.group == "scaled_top10" and functional.shared_expert == "scaled"
            )
            own = set(_mode_group_roles(group, ((ep, experts),)))
            assert own <= set(loaded), sorted(loaded)
            assert not (set(_mode_group_roles(group)) - own) & set(loaded), sorted(loaded)
            assert {"forward", f"pad_{ep}", "gather", "barrier", "expert_layout"} <= set(loaded)
            assert not {"backward_fp32", "epilogue_forward_8", "epilogue_backward_8"} & set(
                loaded
            )


def test_mode_values_and_mode_types_are_refused():
    """Host only: an unknown mode value raises ``ValueError`` naming the allowed values; a mode that is not a ``str``
    raises ``TypeError``, before any value check (also of another argument)."""
    for name, value in (
        ("router_weight_gradient", "dot"),
        ("shared_expert", "half"),
        ("activation", "gelu"),
    ):
        with pytest.raises(ValueError, match=f"{name} must be one of"):
            kernels.check_configuration(8, 32, 8, **{**GROUP_OK, name: value})
    for name in ("router_weight_gradient", "shared_expert", "activation"):
        with pytest.raises(TypeError, match=f"{name} must be one of"):
            kernels.check_configuration(
                8, 32, 8, **{**GROUP_OK, name: None, "router_weight_position": "middle"}
            )


def test_situ_parameters_are_checked():
    """Host only: the per-call SiTU-GLU scalars are finite positive Python numbers whose FP32 values are positive and
    give finite FP32 kernel constants ``2.88539 / beta`` and ``2.88539 / lb`` and a finite FP32 ``beta lb`` (above about
    ``2^129`` the kernels' ``kk``, about ``8.33 / (beta lb)``, is subnormal): ``TypeError`` for a ``bool``, a string or
    a tensor, ``ValueError`` for zero, a negative, an infinite or a NaN value, a value past the FP32 range or rounding
    to 0 there, and a pair whose constants or product are past the FP32 range.  The check returns the FP32 values."""
    assert kernels.check_situ_parameters(4.0, 25.0) == (4.0, 25.0)
    assert kernels.check_situ_parameters(4, 25) == (4.0, 25.0)
    assert kernels.check_situ_parameters(0.1, 3.0) == (
        float(torch.tensor(0.1, dtype=torch.float32)),
        3.0,
    )
    for bad in (True, "4", torch.tensor(4.0), None):
        with pytest.raises(
            TypeError, match="situ_beta must be a finite positive float"
        ):
            kernels.check_situ_parameters(bad, 25.0)
        with pytest.raises(TypeError, match="situ_lb must be a finite positive float"):
            kernels.check_situ_parameters(4.0, bad)
    for bad in (0.0, -4.0, float("inf"), float("nan"), 1e39, 1e-50):
        with pytest.raises(
            ValueError, match="situ_beta must be a finite positive float"
        ):
            kernels.check_situ_parameters(bad, 25.0)
        with pytest.raises(ValueError, match="situ_lb must be a finite positive float"):
            kernels.check_situ_parameters(4.0, bad)
    for beta, lb, constant in (
        (1e-39, 25.0, "2.88539 / situ_beta"),
        (4.0, 1e-39, "2.88539 / situ_lb"),
        (1e20, 1e20, "situ_beta \\* situ_lb"),
    ):
        with pytest.raises(ValueError, match=f"{constant} is past the FP32 range"):
            kernels.check_situ_parameters(beta, lb)


def test_situ_parameters_refuse_a_product_under_the_bound():
    """Host only: ``situ_beta * situ_lb`` of the FP32 values is at least ``2^-119`` (``SITU_BETA_LB_MIN``; under about
    ``2^-119.94`` the kernels' hidden value of tiny gates and ups would be 0).  A product of exactly ``2^-119`` passes,
    also from a value that FP32 rounds up to it; one FP32 step down in either scalar, and ``1e-20`` with ``1e-17``,
    raise ``ValueError``."""
    assert kernels.SITU_BETA_LB_MIN == 2.0**-119
    under = "situ_beta \\* situ_lb is under 2\\^-119"
    # One FP32 ulp under a power of two (exact in FP32).
    below = 1.0 - 2.0**-24
    for beta, lb in ((2.0**-60, 2.0**-59), (4.0, 2.0**-121), (2.0**-121, 4.0)):
        assert kernels.check_situ_parameters(beta, lb) == (beta, lb)
        with pytest.raises(ValueError, match=under):
            kernels.check_situ_parameters(beta * below, lb)
        with pytest.raises(ValueError, match=under):
            kernels.check_situ_parameters(beta, lb * below)
    # Rounds up to 2^-121 in FP32.
    rounds_up = 2.0**-121 * (1.0 - 2.0**-30)
    assert kernels.check_situ_parameters(4.0, rounds_up) == (4.0, 2.0**-121)
    with pytest.raises(ValueError, match=under):
        kernels.check_situ_parameters(1e-20, 1e-17)


def test_situ_parameters_refuse_values_the_kernels_flush():
    """Host only: the SiTU-GLU kernels are built with flush to zero, so their ``-2.88539 / beta`` and
    ``-2.88539 / lb`` read an FP32-subnormal ``beta`` or ``lb`` as 0 and give 0 for a quotient under ``2^-126``
    (``SITU_FP32_MIN_NORMAL``).  ``2^-126`` passes and the largest FP32 subnormal value and ``1e-38`` raise
    ``ValueError``; ``2.88539 * 2^126`` in FP32 (quotient exactly ``2^-126``) passes and the next FP32 value raises.
    The partner keeps ``beta * lb`` at least ``2^-119`` and finite in FP32."""
    assert kernels.SITU_FP32_MIN_NORMAL == 2.0**-126
    subnormal = "{} is an FP32 subnormal value, under 2\\^-126"
    flushed = "2.88539 / {} is under 2\\^-126"
    largest_subnormal = 2.0**-126 - 2.0**-149
    assert kernels.check_situ_parameters(2.0**-126, 256.0) == (2.0**-126, 256.0)
    assert kernels.check_situ_parameters(256.0, 2.0**-126) == (256.0, 2.0**-126)
    for value in (largest_subnormal, 1e-38):
        with pytest.raises(ValueError, match=subnormal.format("situ_beta")):
            kernels.check_situ_parameters(value, 256.0)
        with pytest.raises(ValueError, match=subnormal.format("situ_lb")):
            kernels.check_situ_parameters(256.0, value)
    numerator = float(torch.tensor(2.88539, dtype=torch.float32))
    # Exact in FP32: an FP32 value times a power of two.
    largest = numerator * 2.0**126
    assert float(torch.tensor(largest, dtype=torch.float32)) == largest
    # The next FP32 value: the spacing in [2^127, 2^128) is 2^104.
    above = largest + 2.0**104
    assert 2.0**127 <= largest < above < 2.0**128
    assert kernels.check_situ_parameters(largest, 1.0) == (largest, 1.0)
    assert kernels.check_situ_parameters(1.0, largest) == (1.0, largest)
    with pytest.raises(ValueError, match=flushed.format("situ_beta")):
        kernels.check_situ_parameters(above, 1.0)
    with pytest.raises(ValueError, match=flushed.format("situ_lb")):
        kernels.check_situ_parameters(1.0, above)


#: The 896-expert top-16 group (``MODE_GROUPS["situ_top16"]``): SiTU-GLU routed experts without a shared expert.
SITU_GROUP = kernels.MODE_GROUPS["situ_top16"]
SITU_OK = dict(
    clamped_swiglu=False,
    fp32_wgrad=True,
    router_weight_position="after_down",
    router_weight_gradient="direct",
    shared_expert="none",
    activation="situ_glu",
)
#: SiTU-GLU scalars of the GPU tests (``situ_beta``, ``situ_lb``).
SITU = dict(situ_beta=4.0, situ_lb=25.0)


@pytest.mark.parametrize(
    "layout,topk,change",
    [
        ((64, 14), 8, {}),
        ((8, 32), 16, {}),
        ((64, 14), 16, dict(activation="swiglu")),
        ((64, 14), 16, dict(activation="clamped_swiglu")),
        ((64, 14), 16, dict(clamped_swiglu=True)),
        ((64, 14), 16, dict(shared_expert="unscaled")),
        ((64, 14), 16, dict(shared_expert="scaled")),
        ((64, 14), 16, dict(router_weight_gradient="quotient")),
        ((64, 14), 16, dict(fp32_wgrad=False)),
        ((64, 14), 16, dict(router_weight_position="before_down")),
        ((8, 32), 8, dict(activation="situ_glu", shared_expert="unscaled", router_weight_gradient="quotient")),
        ((8, 32), 8, dict(shared_expert="none", activation="swiglu", router_weight_gradient="quotient")),
        ((8, 64), 10, dict(activation="situ_glu")),
    ],
    ids=[
        "896-top8",
        "256-top16",
        "896-swiglu",
        "896-clamped-act",
        "896-clamped",
        "896-unscaled",
        "896-scaled",
        "896-quotient",
        "896-bf16-wgrad",
        "896-before-down",
        "256-situ",
        "256-none",
        "512-situ",
    ],
)
def test_situ_group_configurations_outside_the_group_are_refused(layout, topk, change):
    """Host only: a configuration that uses the layout, the top-k or a mode of the 896-expert top-16 group (SiTU-GLU,
    ``shared_expert="none"``) but is not the group's configuration raises ``ValueError`` naming the group's
    configuration and the one given."""
    with pytest.raises(
        ValueError,
        match=(
            r"896 routed experts at top-k 16 .* runs \(EP, local experts\) "
            r"\(64, 14\), \(32, 28\) at topk=16"
        ),
    ):
        kernels.check_configuration(*layout, topk, **{**SITU_OK, **change})


def test_situ_group_configuration_is_checked_before_device_work(monkeypatch):
    """Host only: ``check_configuration`` admits the 896-expert top-16 group's configuration and returns its name;
    ``check_target`` accepts the group on SM100a, SM103a and SM107a and refuses it elsewhere with a message naming the
    three.  On each of them, at each of the group's layouts, the adapter (prepared for the context-free forward and
    the recompute) loads the group's roles (the SiTU-GLU forward, context-free forward, recompute and backward
    without a shared expert, the top-16 epilogues without a shared row, the scheduler kernels of that layout) and
    none of the SwiGLU fused kernels, the top-8 epilogues or another layout's scheduler kernels."""
    group = SITU_GROUP
    assert group.targets == ("sm_100a", "sm_103a", "sm_107a")
    assert group.layouts == ((64, 14), (32, 28)) and group.topk == 16
    for layout in group.layouts:
        assert kernels.check_configuration(*layout, 16, **SITU_OK) == "situ_top16"
    for arch in group.targets:
        kernels.check_target("after_down", arch, "situ_top16")
    with pytest.raises(
        ValueError,
        match="896 routed experts .* has no sm_90a build in this package; use an sm_100a, sm_103a or sm_107a device",
    ):
        kernels.check_target("after_down", "sm_90a", "situ_top16")
    loaded = []
    _group_stub_loader(monkeypatch, loaded)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    keywords = {key: SITU_OK[key] for key in ("fp32_wgrad", "router_weight_gradient", "shared_expert", "activation")}
    for arch in group.targets:
        monkeypatch.setattr(jit, "target_arch", lambda device=None, arch=arch: arch)
        for ep, experts in group.layouts:
            loaded.clear()
            functional = backend.MoKFunctional(ep, experts, 16, **keywords)
            functional.prepare(None, save_context=False)
            functional.prepare_recompute(None)
            assert functional.group == "situ_top16"
            assert functional.shared_expert == "none" and functional.activation == "situ_glu"
            own = set(_mode_group_roles(group, ((ep, experts),)))
            assert own <= set(loaded), sorted(loaded)
            assert not (set(_mode_group_roles(group)) - own) & set(loaded), sorted(loaded)
            assert {f"pad_{ep}", "gather", "barrier", "expert_layout"} <= set(loaded)
            swiglu = {"forward", "forward_noctx", "recompute", "backward_fp32", "backward_fp32_direct"}
            assert not (swiglu | {"epilogue_forward_8", "epilogue_backward_8"}) & set(loaded)


def _situ_launcher_problem(topk=16):
    """:func:`_launcher_problem` without a shared expert: ``None`` in the three shared-weight places and in the shared
    places of the saved context, the three routed FP32 accumulators."""
    p = _launcher_problem(topk=topk)
    shared, routed = p["weights"][0::2], p["weights"][1::2]
    p["weights"] = (None, routed[0], None, routed[1], None, routed[2])
    saved = p["saved"]
    p["saved"] = (saved[0], None, saved[2], None, saved[4], None, saved[6])
    p["accumulators"] = p["accumulators"][3:]
    p["shared"] = shared
    return p


def test_situ_launchers_pass_exactly_the_registry_arguments(monkeypatch):
    """Host only (stub kernels, no launch): the SiTU-GLU forward (context and context-free), recompute and backward
    without a shared expert hand their kernels exactly the registry arguments through ``__call__`` and through a plan:
    ``situ_beta`` and ``situ_lb`` as Python floats, no shared map and no ``local_tokens``.  They return ``None`` in
    every shared place (and the backward the three routed accumulators); the backward's unscaled and scaled rings are
    two buffers.  Each refuses a ``swiglu_limit``, missing SiTU-GLU scalars, a shared weight, the shared context and an
    accumulator count other than three before any launch, and the SwiGLU launchers refuse the SiTU-GLU scalars."""
    calls = _group_stub_loader(monkeypatch)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    p = _situ_launcher_problem()
    x, dy, combine, scores, weights = p["x"], p["dy"], p["combine"], p["scores"], p["weights"]
    topk, macro, mini, comm, ep = 16, 768, 256, 4, 4
    ptrs = [4096 * (rank + 1) for rank in range(ep)]  # never dereferenced
    schedule, layout, intermediate, experts, capacity = (
        p["schedule"], p["layout"], p["intermediate"], p["experts"], p["capacity"],
    )  # fmt: skip
    situ = dict(situ_beta=4.0, situ_lb=25)
    routed = weights[1::2]

    def check(role, count):
        assert [name for name, _ in calls] == [role] * count
        for _, arguments in calls:
            assert arguments["situ_beta"] == 4.0 and arguments["situ_lb"] == 25.0
            assert type(arguments["situ_beta"]) is float and type(arguments["situ_lb"]) is float
            assert not NO_SHARED_ARGUMENTS & set(arguments) and "swiglu_limit" not in arguments
        calls.clear()

    for context in (True, False):
        forward = MoKForward(False, context=context, situ_glu=True)
        out = forward(x, ptrs, combine, ptrs, *weights, *schedule, topk, None, comm, macro, mini, layout=layout, **situ)
        assert out[1] is None and out[3] is None and out[5] is None and out[7] is None
        plan = forward.plan(x, ptrs, ptrs, intermediate, experts, capacity, topk, None, comm, macro, mini, **situ)
        planned = MoKForward.launch_planned(plan, *weights, *schedule[:3], layout)
        assert len(planned) == 9 and all(planned[i] is None for i in (1, 3, 5, 7))
        check("forward_situ_noshared" if context else "forward_noctx_situ_noshared", 2)
    recompute = MoKRecompute(False, situ_glu=True)
    out = recompute(x, ptrs, None, routed[0], None, routed[1], *schedule, topk, None, comm, macro, mini, layout=layout,
                    **situ)  # fmt: skip
    assert len(out) == 7 and all(out[i] is None for i in (1, 3, 5))
    plan = recompute.plan(x, ptrs, intermediate, experts, capacity, topk, None, comm, macro, mini, **situ)
    planned = MoKRecompute.launch_planned(plan, None, routed[0], None, routed[1], *schedule[:3], layout)
    assert len(planned) == 7 and all(planned[i] is None for i in (1, 3, 5))
    check("recompute_situ_noshared", 2)
    backward = MoKBackward(False, True, router_weight_gradient="direct", situ_glu=True)
    accumulators = p["accumulators"]
    args = (dy, ptrs, combine, ptrs, scores, ptrs, scores, ptrs, *weights, *p["saved"], x, ptrs, *schedule, topk, None,
            comm, macro, mini)  # fmt: skip
    out = backward(*args, weight_grad_accumulators=accumulators, layout=layout, **situ)
    assert all(out[i] is None for i in (0, 2, 4, 6, 9, 11, 13))
    assert all(out[i] is a for i, a in zip((10, 12, 14), accumulators, strict=True))
    plan = backward.plan(dy, ptrs, ptrs, ptrs, ptrs, ptrs, intermediate, experts, capacity, topk, None, comm, macro,
                         mini, **situ)  # fmt: skip
    planned = backward.launch_planned(plan, *weights, *p["saved"], x, *schedule, layout,
                                      weight_grad_accumulators=accumulators)  # fmt: skip
    assert all(planned[i] is None for i in (0, 2, 4, 6, 9, 11, 13))
    for _, arguments in calls:
        assert arguments["dy_r"] is arguments["dy_routed_ptr"]
        assert arguments["dy_atb_r"] is arguments["dy_scaled_ptr"]
        assert arguments["dy_r"].data_ptr() != arguments["dy_atb_r"].data_ptr()
        for name, accumulator in zip(("dwg_r", "dwu_r", "dwd_r"), accumulators, strict=True):
            assert arguments[name] is accumulator, name
    check("backward_fp32_direct_situ_noshared", 2)
    # Refusals before any launch.
    shared = p["shared"]
    with pytest.raises(ValueError, match="take situ_beta and situ_lb, not a swiglu_limit"):
        forward(x, ptrs, combine, ptrs, *weights, *schedule, topk, 10.0, comm, macro, mini, layout=layout, **situ)
    with pytest.raises(TypeError, match="situ_beta must be a finite positive float"):
        forward(x, ptrs, combine, ptrs, *weights, *schedule, topk, None, comm, macro, mini, layout=layout)
    with pytest.raises(ValueError, match="situ_lb must be a finite positive float"):
        recompute.plan(x, ptrs, intermediate, experts, capacity, topk, None, comm, macro, mini, situ_beta=4.0,
                       situ_lb=float("inf"))  # fmt: skip
    with pytest.raises(ValueError, match="have no shared expert: pass None"):
        forward(x, ptrs, combine, ptrs, shared[0], routed[0], None, routed[1], None, routed[2], *schedule, topk, None,
                comm, macro, mini, layout=layout, **situ)  # fmt: skip
    with pytest.raises(ValueError, match="have no shared expert: pass None"):
        recompute(x, ptrs, None, routed[0], shared[1], routed[1], *schedule, topk, None, comm, macro, mini,
                  layout=layout, **situ)  # fmt: skip
    saved = list(p["saved"])
    saved[1] = saved[2]
    with pytest.raises(ValueError, match="have no shared expert: pass None"):
        backward(dy, ptrs, combine, ptrs, scores, ptrs, scores, ptrs, *weights, *saved, x, ptrs, *schedule, topk, None,
                 comm, macro, mini, weight_grad_accumulators=accumulators, layout=layout, **situ)  # fmt: skip
    with pytest.raises(ValueError, match="exactly three FP32 accumulators"):
        backward(*args, layout=layout, **situ)
    with pytest.raises(ValueError, match="Accumulators must be contiguous FP32 tensors shaped like the weights"):
        backward(*args, weight_grad_accumulators=[*accumulators, accumulators[0]], layout=layout, **situ)
    with pytest.raises(ValueError, match="read only by the SiTU-GLU kernels"):
        MoKForward(False)(x, ptrs, combine, ptrs, shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                          *schedule, topk, None, comm, macro, mini, layout=layout, **situ)  # fmt: skip
    assert calls == []
    with pytest.raises(ValueError, match="SiTU-GLU forward is built without a SwiGLU limit"):
        MoKForward(True, situ_glu=True)
    with pytest.raises(ValueError, match="SiTU-GLU recompute is built without a SwiGLU limit"):
        MoKRecompute(True, situ_glu=True)
    with pytest.raises(ValueError, match="SiTU-GLU backward is built with the direct router-weight gradient"):
        MoKBackward(False, True, situ_glu=True)
    with pytest.raises(ValueError, match="SiTU-GLU backward is built with the direct router-weight gradient"):
        MoKBackward(False, True, router_weight_gradient="direct", shared_operands=True, situ_glu=True)
    with pytest.raises(TypeError, match="situ_glu must be a boolean"):
        MoKForward(False, situ_glu=1)


def test_none_epilogues_take_no_shared_row(monkeypatch):
    """Host only (stub kernels): ``shared_row="none"`` loads the top-16 forward and backward epilogues without a
    shared row, whose every launch (direct and planned) passes no ``shared`` argument and the token count and width of
    the ``[tokens * 16, hidden]`` routed rows; a launch with a shared row raises before any launch, and a planned
    launch takes ``shared=None`` exactly for such a plan."""
    loaded = []
    calls = _group_stub_loader(monkeypatch, loaded)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    epilogues = _host_epilogues(monkeypatch, MoKEpilogues(16, shared_row="none"))
    assert loaded == ["epilogue_forward_none_16", "epilogue_backward_none_16"]
    tokens, hidden = 4, 256
    shared = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    routed = torch.zeros(tokens * 16, hidden, dtype=torch.bfloat16)
    scores = torch.zeros(tokens, 16)
    with pytest.raises(ValueError, match="take shared=None"):
        epilogues.forward(shared, routed, scores)
    with pytest.raises(ValueError, match="take shared=None"):
        epilogues.backward(shared, routed)
    monkeypatch.setattr(
        MoKEpilogues, "_rows", lambda self, shared, routed: (routed.shape[0] // self.top_k, routed.shape[1])
    )
    with pytest.raises(ValueError, match="shared_scale is required exactly"):
        epilogues.backward(None, routed, torch.ones(tokens))
    assert calls == []
    outputs = [
        epilogues.forward(None, routed, scores),
        epilogues.backward(None, routed),
        MoKEpilogues.launch_planned(epilogues.plan(routed, scores), None, scores),
        MoKEpilogues.launch_planned(epilogues.plan(routed, False), None),
    ]
    for output in outputs:
        assert tuple(output.shape) == (tokens, hidden) and output.dtype == torch.bfloat16
    roles = ["epilogue_forward_none_16", "epilogue_backward_none_16"] * 2
    assert [role for role, _ in calls] == roles
    for _, arguments in calls:
        assert "shared" not in arguments and arguments["routed"] is routed
        assert arguments["tokens"] == tokens and arguments["hidden"] == hidden
    with pytest.raises(ValueError, match="shared is None exactly for a plan of the shared-row-free epilogues"):
        MoKEpilogues.launch_planned(epilogues.plan(routed, False), shared)
    with pytest.raises(ValueError, match="shared is None exactly for a plan of the shared-row-free epilogues"):
        MoKEpilogues.launch_planned(MoKEpilogues(8).plan(routed[: tokens * 8], False), None)


def test_situ_call_refusals_come_before_device_work(monkeypatch):
    """Host only: an adapter of the 896-expert top-16 group refuses, before any copy, barrier or kernel load, a forward,
    recompute or backward without ``situ_beta`` and ``situ_lb``, with values that are not finite positive numbers or
    with values the kernels flush (an FP32-subnormal ``beta``, an ``lb`` whose quotient ``2.88539 / lb`` is under
    ``2^-126``), a shared weight, a ``swiglu_limit``, a ``shared_scale`` and MXFP8 routed weights; a SwiGLU adapter
    refuses the SiTU-GLU scalars."""
    functional = object.__new__(backend.MoKFunctional)
    functional.local_experts = 4
    functional._validated = {}
    functional.group, functional.shared_expert, functional.router_weight_gradient = "situ_top16", "none", "direct"
    functional.activation, functional.weight_before_down = "situ_glu", False
    functional._forward_kernels, functional._backward_kernels = {}, {}
    functional._recompute_kernels, functional._context_free_forward_kernels = {}, {}
    events = []
    _group_stub_loader(monkeypatch, events)
    workspace = dataclasses.make_dataclass("W", ["device"])(torch.device("cuda", 0))
    monkeypatch.setattr(backend.MoKFunctional, "_inputs", staticmethod(lambda *args: (workspace, 256)))
    monkeypatch.setattr(functional, "_check_device", lambda workspace: None)
    monkeypatch.setattr(functional, "_barrier", lambda w: events.append("barrier"))
    monkeypatch.setattr(backend.MoKFunctional, "_copy_source", staticmethod(lambda *args: events.append("copy")))
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(256, 256, **options)
    shared = [torch.zeros(256, 256, **options) for _ in range(3)]
    routed = [torch.zeros(4, 256, 256, **options) for _ in range(3)]
    data = torch.zeros(4, 256, 256, dtype=torch.uint8)
    mxfp8 = [(data, torch.zeros(64, dtype=torch.uint8)) for _ in range(3)]
    scores = torch.zeros(256, 16)
    schedule = object()
    context = backend.MoKForwardContext(
        x_routed=x, gate_shared=None, gate_routed=x, up_shared=None, up_routed=x, hidden_shared=None, hidden_routed=x,
        schedule=schedule,
    )  # fmt: skip
    none = (None, None, None)
    forward = lambda **kw: functional.forward(  # noqa: E731
        None, None, schedule, x, scores, *kw.pop("s", none), *kw.pop("r", routed), **kw
    )
    recompute = lambda **kw: functional.recompute_forward_context(  # noqa: E731
        None, None, schedule, x, *kw.pop("s", none)[:2], *kw.pop("r", routed)[:2], **kw
    )
    backward = lambda **kw: functional.backward(  # noqa: E731
        None, None, schedule, context, x, x, scores, *kw.pop("s", none), *kw.pop("r", routed), **kw
    )
    for run in (forward, recompute, backward):
        with pytest.raises(ValueError, match="requires situ_beta and situ_lb"):
            run()
        with pytest.raises(ValueError, match="requires situ_beta and situ_lb"):
            run(situ_beta=4.0)
        with pytest.raises(TypeError, match="situ_lb must be a finite positive float"):
            run(situ_beta=4.0, situ_lb=torch.tensor(25.0))
        with pytest.raises(ValueError, match="situ_beta must be a finite positive float"):
            run(situ_beta=0.0, situ_lb=25.0)
        with pytest.raises(ValueError, match="situ_beta is an FP32 subnormal value"):
            run(situ_beta=1e-38, situ_lb=256.0)
        with pytest.raises(ValueError, match="2.88539 / situ_lb is under 2\\^-126"):
            run(situ_beta=1.0, situ_lb=3e38)
        with pytest.raises(ValueError, match="has no shared expert: pass None"):
            run(s=shared, **SITU)
        with pytest.raises(ValueError, match=r"runs SiTU-GLU \(situ_beta, situ_lb\): pass swiglu_limit=None"):
            run(swiglu_limit=10.0, **SITU)
    for run in (forward, backward):
        with pytest.raises(ValueError, match="takes BF16 routed weights"):
            run(r=mxfp8, **SITU)
        with pytest.raises(ValueError, match="taken only with shared_expert='scaled'"):
            run(shared_scale=torch.ones(256), **SITU)
    functional.group, functional.shared_expert, functional.router_weight_gradient = None, "unscaled", "quotient"
    functional.activation = "swiglu"
    for run in (forward, backward):
        with pytest.raises(ValueError, match="taken only with activation='situ_glu'"):
            run(s=shared, **SITU)
    assert events == []


def _launcher_problem(
    tokens=512,
    hidden=256,
    intermediate=256,
    experts=4,
    topk=10,
    macro=768,
    capacity=8192,
):
    """Host-only launcher arguments (no device memory is read): BF16 zeros for x, a separate dy, the combine rows and
    the saved context, the weights, FP32 accumulators and scores, a schedule and its layout."""
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(tokens, hidden, **options)
    dy = torch.zeros(tokens, hidden, **options)
    combine = torch.zeros(tokens * topk, hidden, **options)
    shared = [
        torch.zeros(intermediate, hidden, **options),
        torch.zeros(intermediate, hidden, **options),
        torch.zeros(hidden, intermediate, **options),
    ]
    routed = [
        torch.zeros(experts, intermediate, hidden, **options),
        torch.zeros(experts, intermediate, hidden, **options),
        torch.zeros(experts, hidden, intermediate, **options),
    ]
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    schedule = (
        torch.full((capacity,), -1, dtype=torch.int32),
        torch.zeros(capacity, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(experts, dtype=torch.int32),
    )
    ring = torch.zeros(macro, intermediate, **options)
    saved = (torch.zeros(macro, hidden, **options), x[:, :intermediate].clone(), ring, x[:, :intermediate].clone(),
             ring, x[:, :intermediate].clone(), ring)  # fmt: skip
    return dict(
        x=x, dy=dy, combine=combine, weights=weights, schedule=schedule, saved=saved,
        layout=torch.zeros(3 * experts + capacity // 256, dtype=torch.int32),
        accumulators=[torch.zeros(w.shape) for w in (*shared, *routed)],
        scores=torch.zeros(tokens, topk), capacity=capacity, intermediate=intermediate, experts=experts,
    )  # fmt: skip


@pytest.mark.parametrize(
    "operands", [False, True], ids=["staged-operands", "shared-operands"]
)
def test_direct_backward_binds_the_unscaled_and_the_scaled_ring(monkeypatch, operands):
    """Host only (stub kernels): the direct-gradient backward hands its kernel exactly the registry arguments through
    ``__call__`` and through a plan; the down projection's input gradient (``dy_r``) and the dispatch's unscaled store
    (``dy_routed_ptr``) are one ``[macro, hidden]`` BF16 ring, the down-projection weight gradient (``dy_atb_r``) and the
    scaled store (``dy_scaled_ptr``) a second one.  With ``shared_operands=True`` the shared weight gradients read the
    given ``(dy_atb_s, x_atb_s)``; without it the staged ``dy`` and ``x``."""
    calls = _group_stub_loader(monkeypatch)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    p = _launcher_problem()
    x, dy, combine, scores = p["x"], p["dy"], p["combine"], p["scores"]
    topk, macro, mini, comm, ep = 10, 768, 256, 4, 4
    ptrs = [4096 * (rank + 1) for rank in range(ep)]  # never dereferenced
    shared_operands = (torch.zeros_like(x), torch.zeros_like(x)) if operands else None
    backward = MoKBackward(
        False, True, router_weight_gradient="direct", shared_operands=operands
    )
    backward(
        dy, ptrs, combine, ptrs, scores, ptrs, scores, ptrs, *p["weights"], *p["saved"], x, ptrs, *p["schedule"],
        topk, None, comm, macro, mini, weight_grad_accumulators=p["accumulators"], layout=p["layout"],
        shared_operands=shared_operands,
    )  # fmt: skip
    plan = backward.plan(
        dy, ptrs, ptrs, ptrs, ptrs, ptrs, p["intermediate"], p["experts"], p["capacity"], topk, None, comm, macro,
        mini,
    )  # fmt: skip
    backward.launch_planned(
        plan, *p["weights"], *p["saved"], x, *p["schedule"], p["layout"], weight_grad_accumulators=p["accumulators"],
        shared_operands=shared_operands,
    )  # fmt: skip
    assert [role for role, _ in calls] == ["backward_fp32_direct"] * 2
    for _, arguments in calls:
        unscaled, scaled = arguments["dy_r"], arguments["dy_atb_r"]
        assert (
            unscaled is arguments["dy_routed_ptr"]
            and scaled is arguments["dy_scaled_ptr"]
        )
        assert unscaled is not scaled and unscaled.data_ptr() != scaled.data_ptr()
        for ring in (unscaled, scaled):
            assert (
                tuple(ring.shape) == (macro, x.shape[1])
                and ring.dtype == torch.bfloat16
            )
        expected = shared_operands if operands else (dy, x)
        assert (
            arguments["dy_atb_s"] is expected[0] and arguments["x_atb_s"] is expected[1]
        )
        assert arguments["dy_s"] is dy
    # The launcher's shared_operands must match how it was built, before any launch.
    calls.clear()
    with pytest.raises(ValueError, match="shared_operands is required exactly"):
        backward(
            dy, ptrs, combine, ptrs, scores, ptrs, scores, ptrs, *p["weights"], *p["saved"], x, ptrs, *p["schedule"],
            topk, None, comm, macro, mini, weight_grad_accumulators=p["accumulators"], layout=p["layout"],
            shared_operands=None if operands else (x, x),
        )  # fmt: skip
    if operands:
        with pytest.raises(
            ValueError, match="shared_operands must be two contiguous BF16"
        ):
            backward.launch_planned(
                plan, *p["weights"], *p["saved"], x, *p["schedule"], p["layout"],
                weight_grad_accumulators=p["accumulators"], shared_operands=(x, x[:256]),
            )  # fmt: skip
    assert calls == []


def test_direct_backward_is_refused_outside_its_build(monkeypatch):
    """Host only: the direct-gradient backward exists for plain SwiGLU with FP32 weight gradients and the router weight
    after the down projection; any other combination, or an unknown gradient name, raises before a kernel loads."""
    loaded = []
    _group_stub_loader(monkeypatch, loaded)
    for args in ((True, True), (False, False), (True, True, True)):
        with pytest.raises(
            ValueError,
            match="direct router-weight gradient backward is built for plain SwiGLU",
        ):
            MoKBackward(*args, router_weight_gradient="direct")
    with pytest.raises(ValueError, match="router_weight_gradient must be one of"):
        MoKBackward(False, True, router_weight_gradient="dot")
    with pytest.raises(TypeError, match="shared_operands must be booleans"):
        MoKBackward(False, True, shared_operands=1)
    assert loaded == []
    MoKBackward(False, True)
    assert loaded[0] == "backward_fp32"


def _host_epilogues(monkeypatch, epilogues):
    """Let host-only tests launch ``epilogues`` with CPU tensors: no CUDA checks, three CTAs."""
    monkeypatch.setattr(MoKEpilogues, "_check", lambda self, shared, routed: None)

    def ctas(self, tokens, device):
        self._no_scores.setdefault(device, torch.zeros(1))
        return 3

    monkeypatch.setattr(MoKEpilogues, "_ctas", ctas)
    return epilogues


def test_scaled_epilogues_take_the_shared_scale(monkeypatch):
    """Host only (stub kernels): ``shared_row="scaled"`` loads the scaled top-10 forward and backward epilogues, whose
    every launch (direct and planned) passes the caller's FP32 ``[tokens]`` ``shared_scale``; a scaled launch without
    them, or with another shape, raises before any launch, and an unscaled epilogue refuses them.  Top-k 10 has the
    scaled epilogues only and they have top-k 10 only."""
    loaded = []
    calls = _group_stub_loader(monkeypatch, loaded)
    monkeypatch.setattr(kernels, "_ffi_stream", contextlib.nullcontext)
    epilogues = _host_epilogues(monkeypatch, MoKEpilogues(10, shared_row="scaled"))
    assert loaded == ["epilogue_forward_scaled_10", "epilogue_backward_scaled_10"]
    tokens, hidden = 4, 256
    shared = torch.zeros(tokens, hidden, dtype=torch.bfloat16)
    routed = torch.zeros(tokens * 10, hidden, dtype=torch.bfloat16)
    scores, scales = torch.zeros(tokens, 10), torch.ones(tokens)
    with pytest.raises(ValueError, match="shared_scale is required exactly"):
        epilogues.forward(shared, routed, scores)
    with pytest.raises(ValueError, match="shared_scale is required exactly"):
        epilogues.backward(shared, routed)
    with pytest.raises(
        ValueError, match=r"shared_scale must be contiguous CUDA FP32 \[tokens\]"
    ):
        epilogues.forward(shared, routed, scores, torch.ones(tokens + 1))
    assert calls == []
    epilogues.forward(shared, routed, scores, scales)
    epilogues.backward(shared, routed, scales)
    MoKEpilogues.launch_planned(epilogues.plan(routed, scores), shared, scores, scales)
    MoKEpilogues.launch_planned(
        epilogues.plan(routed, False), shared, shared_scale=scales
    )
    roles = ["epilogue_forward_scaled_10", "epilogue_backward_scaled_10"] * 2
    assert [role for role, _ in calls] == roles
    for _, arguments in calls:
        assert arguments["shared_scale"] is scales and arguments["shared"] is shared
        assert arguments["tokens"] == tokens and arguments["hidden"] == hidden
    with pytest.raises(ValueError, match="top_k=2, 6 or 8"):
        MoKEpilogues(10)
    with pytest.raises(ValueError, match="The scaled epilogues support top_k=10"):
        MoKEpilogues(8, shared_row="scaled")
    with pytest.raises(ValueError, match="shared_row='unscaled', 'scaled' or 'none'"):
        MoKEpilogues(8, shared_row="half")
    with pytest.raises(
        ValueError, match="The shared-row-free epilogues support top_k=16"
    ):
        MoKEpilogues(8, shared_row="none")
    unscaled = MoKEpilogues(8)
    with pytest.raises(ValueError, match="shared_scale is required exactly"):
        unscaled.forward(
            shared,
            torch.zeros(tokens * 8, hidden, dtype=torch.bfloat16),
            torch.zeros(tokens, 8),
            scales,
        )


def test_shared_scale_helpers_pass_the_registry_arguments(monkeypatch):
    """Host only (stub kernels): ``rows`` hands its kernel the BF16 input, the FP32 scales, a new BF16 output of the
    input's shape and the row count and width on one CTA per row, at most eight per SM; ``dot`` the two BF16 operands
    and a new FP32 ``[tokens]`` output on one CTA per eight rows, at most eight per SM.  A scale tensor of another shape
    raises before any launch; zero rows launch nothing."""
    calls = _group_stub_loader(monkeypatch)
    monkeypatch.setattr(
        kernels.MoKSharedScale, "_check_rows", staticmethod(lambda *args: None)
    )
    monkeypatch.setattr(kernels.MoKSharedScale, "_sm_count", lambda self, device: 2)
    helpers = kernels.MoKSharedScale()
    values, scales = torch.zeros(3, 16, dtype=torch.bfloat16), torch.ones(3)
    with pytest.raises(
        ValueError, match=r"scales must be contiguous CUDA FP32 \[tokens\]"
    ):
        helpers.rows(values, torch.ones(4))
    assert calls == []
    out = helpers.rows(values, scales)
    dots = helpers.dot(values, out)
    assert out.shape == values.shape and out.dtype == torch.bfloat16
    assert tuple(dots.shape) == (3,) and dots.dtype == torch.float32
    assert [role for role, _ in calls] == ["shared_scale_rows", "shared_scale_dot"]
    rows, dot = calls[0][1], calls[1][1]
    assert rows["rows"] is values and rows["scales"] is scales and rows["scaled"] is out
    assert dot["left"] is values and dot["right"] is out and dot["dots"] is dots
    assert (rows["tokens"], rows["width"], rows["grid_x"]) == (3, 16, 3)
    assert (dot["tokens"], dot["width"], dot["grid_x"]) == (3, 16, 1)
    many = torch.zeros(200, 16, dtype=torch.bfloat16)
    calls.clear()
    helpers.rows(many, torch.ones(200))
    helpers.dot(many, many)
    assert [arguments["grid_x"] for _, arguments in calls] == [16, 16]
    calls.clear()
    empty = torch.zeros(0, 16, dtype=torch.bfloat16)
    assert helpers.rows(empty, torch.ones(0)).shape == (0, 16) and helpers.dot(
        empty, empty
    ).shape == (0,)
    assert calls == []


def test_scaled_call_refusals_come_before_device_work(monkeypatch):
    """Host only: an adapter of the 512-expert top-10 group refuses, before any copy, barrier or kernel load, a forward
    or backward without ``shared_scale``, a ``shared_scale`` that is not a tensor (``TypeError``) or not FP32
    ``[tokens]`` on the device, a clamped SwiGLU limit and MXFP8 routed weights; an unscaled adapter refuses a
    ``shared_scale``."""
    functional = object.__new__(backend.MoKFunctional)
    functional.local_experts = 4
    functional._validated = {}
    functional.group, functional.shared_expert, functional.router_weight_gradient = (
        "scaled_top10",
        "scaled",
        "direct",
    )
    functional._forward_kernels, functional._backward_kernels = {}, {}
    functional._recompute_kernels, functional._context_free_forward_kernels = {}, {}
    events = []
    _group_stub_loader(monkeypatch, events)
    workspace = dataclasses.make_dataclass("W", ["device"])(torch.device("cuda", 0))
    monkeypatch.setattr(
        backend.MoKFunctional, "_inputs", staticmethod(lambda *args: (workspace, 256))
    )
    monkeypatch.setattr(functional, "_check_device", lambda workspace: None)
    monkeypatch.setattr(functional, "_barrier", lambda w: events.append("barrier"))
    monkeypatch.setattr(
        backend.MoKFunctional,
        "_copy_source",
        staticmethod(lambda *args: events.append("copy")),
    )
    options = dict(dtype=torch.bfloat16)
    x = torch.zeros(256, 256, **options)
    shared = [torch.zeros(256, 256, **options) for _ in range(3)]
    routed = [torch.zeros(4, 256, 256, **options) for _ in range(3)]
    data = torch.zeros(4, 256, 256, dtype=torch.uint8)
    mxfp8 = [(data, torch.zeros(64, dtype=torch.uint8)) for _ in range(3)]
    scores = torch.zeros(256, 10)
    schedule = object()
    context = backend.MoKForwardContext(
        x_routed=x, gate_shared=x, gate_routed=x, up_shared=x, up_routed=x, hidden_shared=x, hidden_routed=x,
        schedule=schedule,
    )  # fmt: skip
    forward = lambda **kw: functional.forward(
        None, None, schedule, x, scores, *shared, *kw.pop("r", routed), **kw
    )  # noqa: E731
    backward = lambda **kw: functional.backward(  # noqa: E731
        None, None, schedule, context, x, x, scores, *shared, *kw.pop("r", routed), **kw
    )
    for run in (forward, backward):
        with pytest.raises(ValueError, match="requires shared_scale"):
            run()
        with pytest.raises(TypeError, match="shared_scale must be a torch.Tensor"):
            run(shared_scale=[1.0] * 256)
        with pytest.raises(
            ValueError, match="shared_scale must be contiguous torch.float32"
        ):
            run(shared_scale=torch.ones(256))  # a CPU tensor
        with pytest.raises(ValueError, match="runs plain SwiGLU"):
            run(swiglu_limit=10.0, shared_scale=torch.ones(256))
        with pytest.raises(ValueError, match="takes BF16 routed weights"):
            run(r=mxfp8, shared_scale=torch.ones(256))
    functional.group, functional.shared_expert, functional.router_weight_gradient = (
        None,
        "unscaled",
        "quotient",
    )
    for run in (forward, backward):
        with pytest.raises(ValueError, match="taken only with shared_expert='scaled'"):
            run(shared_scale=torch.ones(256))
    assert events == []


def test_memory_report_counts_each_storage_once():
    """Host only: the memory report helper (``report_mok_scaled.py``) counts each tensor storage once, in the first
    part that reaches it (views, and a workspace a schedule or a context refers to, included); the retained state is
    the schedule, with the context added for a layer without checkpointing."""
    from tests.experimental.report_mok_scaled import memory_report

    @dataclasses.dataclass
    class Store:
        rows: torch.Tensor
        more: tuple

    @dataclasses.dataclass
    class Plan:
        rows: torch.Tensor
        store: Store

    base = torch.zeros(64)
    store = Store(base, (base[:8], torch.zeros(4, dtype=torch.bfloat16), None))
    plan = Plan(torch.zeros(10, dtype=torch.int32), store)
    context = (torch.zeros(3, dtype=torch.float64), plan, {"rows": torch.zeros(2)})
    assert memory_report(store, plan, context) == dict(
        workspace=256 + 8,
        schedule=40,
        context=24 + 8,
        retained_checkpointed=40,
        retained_with_context=40 + 32,
    )


# The 512-expert top-10 group (``MODE_GROUPS["scaled_top10"]``) on SM100a, SM103a and SM107a: the helpers, the scaled
# epilogues, the direct-gradient backward and the adapter.  Every input is drawn on the CPU and copied to the device.

#: The group's (EP, local experts) layout and top-k.
GROUP_LAYOUT, GROUP_TOPK = (
    kernels.MODE_GROUPS["scaled_top10"].layouts[0],
    kernels.MODE_GROUPS["scaled_top10"].topk,
)
#: The group's roles beside the scheduler layout's.
GROUP_KERNELS = (
    "backward_fp32_direct",
    "epilogue_forward_scaled_10",
    "shared_scale_rows",
    "shared_scale_dot",
)


def _bits(value):
    """``value``'s bytes as integers (BF16 / FP32), so that signed zeros differ."""
    return value.view({2: torch.int16, 4: torch.int32}[value.element_size()])


def _same_bits(a, b):
    return all(torch.equal(_bits(x), _bits(y)) for x, y in zip(a, b, strict=True))


def _cpu_problem(tokens, hidden, intermediate, experts, topk, seed, device):
    """Inputs drawn on the CPU, then copied to ``device``: BF16 ``x`` and ``dY``, weights scaled to unit-variance
    outputs (shared gate, up, down, routed gate, up, down), FP32 router weights in [0, 1.5) with every 5th token's
    first weight and every weight of token 3 exactly 0, FP32 shared scales in [0, 2) with every 4th token's exactly 0
    and the next token's exactly 1, and int64 expert ids (``topk`` distinct experts per token)."""
    g = torch.Generator().manual_seed(seed)
    x = (randn(tokens, hidden, generator=g) * 0.125).bfloat16()
    dy = (randn(tokens, hidden, generator=g) * 0.125).bfloat16()
    weights = (
        (randn(intermediate, hidden, generator=g) / hidden**0.5).bfloat16(),
        (randn(intermediate, hidden, generator=g) / hidden**0.5).bfloat16(),
        (randn(hidden, intermediate, generator=g) / intermediate**0.5).bfloat16(),
        (randn(experts, intermediate, hidden, generator=g) / hidden**0.5).bfloat16(),
        (randn(experts, intermediate, hidden, generator=g) / hidden**0.5).bfloat16(),
        (
            randn(experts, hidden, intermediate, generator=g) / intermediate**0.5
        ).bfloat16(),
    )
    scores = rand(tokens, topk, generator=g) * 1.5
    scores[::5, 0] = 0.0
    scores[3:4] = 0.0
    scale = rand(tokens, generator=g) * 2.0
    scale[::4] = 0.0
    scale[1::4] = 1.0
    ids = rand(tokens, experts, generator=g).argsort(-1)[:, :topk].contiguous()
    return dict(
        x=x.to(device), dy=dy.to(device), weights=tuple(w.to(device) for w in weights), scores=scores.to(device),
        scale=scale.to(device), ids=ids.to(device),
    )  # fmt: skip


@pytest.mark.parametrize(
    "tokens,width", [(0, 4096), (1, 8), (37, 1032), (4099, 4096), (4099, 1024)]
)
def test_shared_scale_helpers_on_the_gpu(monkeypatch, tokens, width):
    """``rows`` gives ``BF16(s * v)`` bit for bit (one FP32 product, one rounding: the CPU's FP32 product rounded to
    BF16), so ``s = 1`` keeps the input bits and ``s = 0`` gives signed zeros.  ``dot`` gives the same FP32 bytes run
    to run and at another grid size; a row with one nonzero product gives that product exactly, a zero row zero.  The
    accuracy of ``dot`` against the float64 oracle is judged in ``test_mok_accuracy.py``."""
    for role in ("shared_scale_rows", "shared_scale_dot"):
        _require_role(role)
    device = torch.device("cuda", torch.cuda.current_device())
    helpers = kernels.MoKSharedScale()
    g = torch.Generator().manual_seed(7001 + tokens + width)
    values = (randn(tokens, width, generator=g) * 0.125).bfloat16()
    other = (randn(tokens, width, generator=g) * 0.125).bfloat16()
    scales = rand(tokens, generator=g) * 4.0 - 2.0
    scales[::4] = 0.0
    scales[1::4] = 1.0
    single = torch.zeros_like(values)
    columns = randint(0, width, (tokens,), generator=g)
    single[torch.arange(tokens), columns] = values[torch.arange(tokens), columns]
    expected_rows = (scales[:, None] * values.float()).bfloat16()
    expected_single = (
        single.float()[torch.arange(tokens), columns]
        * other.float()[torch.arange(tokens), columns]
    )
    d_values, d_other, d_scales, d_single = (
        v.to(device) for v in (values, other, scales, single)
    )
    with kernels._ffi_stream():
        scaled = [helpers.rows(d_values, d_scales) for _ in range(REPEATS)]
        dots = [helpers.dot(d_values, d_other) for _ in range(REPEATS)]
        single_dot = helpers.dot(d_single, d_other)
        zero_dot = helpers.dot(torch.zeros_like(d_values), d_other)
    monkeypatch.setitem(helpers._sms, device, 1)  # one SM's grid: at most 8 CTAs
    with kernels._ffi_stream():
        small_grid = (helpers.rows(d_values, d_scales), helpers.dot(d_values, d_other))
    torch.cuda.synchronize()
    assert scaled[0].shape == (tokens, width) and dots[0].shape == (tokens,)
    assert all(
        torch.equal(_bits(value.cpu()), _bits(expected_rows))
        for value in (*scaled, small_grid[0])
    )
    assert torch.equal(
        _bits(scaled[0][1::4].cpu()), _bits(values[1::4])
    )  # s = 1: the input bits
    assert all(
        torch.equal(_bits(value), _bits(dots[0]))
        for value in (*dots[1:], small_grid[1])
    )
    assert torch.isfinite(dots[0]).all().item()
    nonzero = expected_single != 0
    assert torch.equal(single_dot.cpu()[nonzero], expected_single[nonzero])
    assert torch.count_nonzero(zero_dot).item() == 0


@pytest.mark.parametrize("tokens,hidden", [(1, 1032), (37, 4096), (512, 256)])
def test_scaled_epilogues_on_the_gpu(tokens, hidden):
    """The scaled top-10 epilogues.  The backward sum is exact: ``BF16(s * shared + routed_0 + ... + routed_9)`` in
    FP32 in that order, bit for bit the CPU's.  The forward with ``s = 1`` and the last two slots zero (zero rows and
    zero weights) has the bytes of the top-8 epilogue on the first eight slots (both start from ``s * shared``, then
    add the slots in ascending order, and ``fma(0, 0, acc) = acc``); a token's row with ``s = 1`` has the bytes of an
    all-ones call (each token reads its own scale).  Five launches and a CUDA Graph replay give the same bytes.  The
    forward's accuracy against the float64 oracle is judged in ``test_mok_accuracy.py``."""
    _require_role("epilogue_forward_scaled_10")
    _require_role("epilogue_backward_8")
    device = torch.device("cuda", torch.cuda.current_device())
    scaled, top8 = MoKEpilogues(10, shared_row="scaled"), MoKEpilogues(8)
    g = torch.Generator().manual_seed(7101 + tokens + hidden)
    shared = (randn(tokens, hidden, generator=g) * 0.125).bfloat16()
    routed = (randn(tokens, 10, hidden, generator=g) * 0.125).bfloat16()
    scores = rand(tokens, 10, generator=g) + 0.01
    mixed = rand(tokens, generator=g) * 2.0
    mixed[::3] = 1.0
    expected_dx = mixed[:, None] * shared.float()
    for slot in range(10):
        expected_dx = expected_dx + routed[:, slot].float()
    padded, padded_scores = routed.clone(), scores.clone()
    padded[:, 8:], padded_scores[:, 8:] = 0.0, 0.0
    d = {name: value.to(device) for name, value in dict(
        shared=shared, routed=routed.view(-1, hidden), scores=scores, mixed=mixed, ones=torch.ones(tokens),
        padded=padded.view(-1, hidden), padded_scores=padded_scores,
        routed8=routed[:, :8].reshape(-1, hidden), scores8=scores[:, :8].contiguous(),
    ).items()}  # fmt: skip

    def launch():
        return (
            scaled.forward(d["shared"], d["routed"], d["scores"], d["mixed"]),
            scaled.backward(d["shared"], d["routed"], d["mixed"]),
        )

    eager = [tuple(value.clone() for value in launch()) for _ in range(REPEATS)]
    assert all(_same_bits(eager[0], other) for other in eager[1:])
    assert eager[0][0].shape == (tokens, hidden)
    assert torch.equal(_bits(eager[0][1].cpu()), _bits(expected_dx.bfloat16()))
    ones = scaled.forward(d["shared"], d["routed"], d["scores"], d["ones"])
    unit = mixed == 1.0
    assert torch.equal(
        _bits(eager[0][0][unit.to(device)]), _bits(ones[unit.to(device)])
    )
    padded_y = scaled.forward(d["shared"], d["padded"], d["padded_scores"], d["ones"])
    assert torch.equal(
        _bits(padded_y), _bits(top8.forward(d["shared"], d["routed8"], d["scores8"]))
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = launch()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(captured, eager[0])


#: The direct-gradient launcher check's outputs, by index of ``(*backward result, dx_peer, ds_peer, forward shared
#: output, combine rows)``.
DIRECT_OUTPUTS = {
    "dx_shared": 0, "d_gate_shared": 2, "d_up_shared": 4, "dh_shared": 6, "dw_shared_gate": 9, "dw_routed_gate": 10,
    "dw_shared_up": 11, "dw_routed_up": 12, "dw_shared_down": 13, "dw_routed_down": 14, "dx_routed": 15,
    "d_weights": 16, "y_shared": 17, "y_routed": 18,
}  # fmt: skip
#: What the direct and the quotient backward write alike (every shared output and the routed down-projection weight
#: gradient, which reads ``BF16(w * dy)`` in both), and what only the shared scale's operands change.
DIRECT_EQUAL_TO_QUOTIENT = ("dx_shared", "d_gate_shared", "d_up_shared", "dh_shared", "dw_shared_gate", "dw_shared_up",
                            "dw_shared_down", "dw_routed_down", "y_shared", "y_routed")  # fmt: skip
SHARED_OPERAND_OUTPUTS = ("dw_shared_gate", "dw_shared_up", "dw_shared_down")


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: "h{}-i{}-m{}".format(*g))
def test_direct_backward_on_the_gpu(geometry):
    """The direct-gradient backward at top-10 over 16 local experts (plain SwiGLU, FP32 accumulators from nonzero
    starts), against the quotient backward on the same forward: the same bytes in every shared output and in the
    routed down-projection weight gradient; a zero router weight has a nonzero gradient and exactly zero routed input
    gradients.  With ``shared_operands=True`` and the helpers' ``BF16(1 * dy)`` / ``BF16(1 * x)`` every output has the
    bytes of the staged-operand launch; with ``s = 0`` the shared weight-gradient accumulators keep their starts bit
    for bit and every other output keeps its bytes.  :data:`REPEATS` launches, a CUDA Graph replay, and a replay after
    the inputs change give the eager bytes."""
    for role in GROUP_KERNELS:
        _require_role(role)
    hidden, intermediate, macro = geometry
    experts, topk, tokens, mini, comm = 16, GROUP_TOPK, 512, 256, 4
    device = torch.device("cuda", torch.cuda.current_device())
    p = _cpu_problem(
        tokens, hidden, intermediate, experts, topk, 7201 + hidden + macro, device
    )
    x, dy, scores = p["x"], p["dy"], p["scores"]
    shared_gate, shared_up, shared_down, routed_gate, routed_up, routed_down = p[
        "weights"
    ]
    forward = MoKForward(False)
    launchers = dict(
        quotient=MoKBackward(False, True),
        direct=MoKBackward(False, True, router_weight_gradient="direct"),
        operands=MoKBackward(
            False, True, router_weight_gradient="direct", shared_operands=True
        ),
    )
    helpers = kernels.MoKSharedScale()
    row = torch.arange(tokens, device=device)
    s = single_rank_schedule(
        torch.stack([(row + slot) % experts for slot in range(topk)], -1), experts
    )
    options = dict(device=device, dtype=torch.bfloat16)
    combine = torch.empty(tokens * topk, hidden, **options)
    dx_peer = torch.empty(tokens * topk, hidden, **options)
    ds_peer = torch.empty_like(scores)
    g = torch.Generator().manual_seed(7301 + hidden)
    initial = [(randn(w.shape, generator=g) * 0.01).to(device) for w in p["weights"]]
    accumulators = [value.clone() for value in initial]
    ones, zeros = torch.ones(tokens, device=device), torch.zeros(tokens, device=device)

    def sequence(name, scale=None):
        for accumulator, start in zip(accumulators, initial, strict=True):
            accumulator.copy_(start)
        values = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared_gate, routed_gate, shared_up, routed_up, shared_down, routed_down,
            s["peers"], s["schedule"], s["num_tokens"], s["counts"],
            topk, None, comm, macro, mini,
        )  # fmt: skip
        operands = None
        if scale is not None:
            with kernels._ffi_stream():
                operands = (helpers.rows(dy, scale), helpers.rows(x, scale))
        result = launchers[name](
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared_gate, routed_gate, shared_up, routed_up, shared_down, routed_down,
            *values[:7], x, [x.data_ptr()],
            s["peers"], s["schedule"], s["num_tokens"], s["counts"],
            topk, None, comm, macro, mini,
            weight_grad_accumulators=accumulators, shared_operands=operands,
        )  # fmt: skip
        return result, values

    def outputs(launch):
        result, values = launch
        every = (*result, dx_peer, ds_peer, values[7], combine)
        out = {name: every[index].clone() for name, index in DIRECT_OUTPUTS.items()}
        for name, value in out.items():
            assert torch.isfinite(value).all().item(), name
        return out

    def same(a, b, names=DIRECT_OUTPUTS):
        return [
            name for name in names if not torch.equal(_bits(a[name]), _bits(b[name]))
        ]

    eager = [outputs(sequence("direct")) for _ in range(REPEATS)]
    assert all(same(eager[0], other) == [] for other in eager[1:])
    quotient = outputs(sequence("quotient"))
    assert same(eager[0], quotient, DIRECT_EQUAL_TO_QUOTIENT) == []
    zero = scores == 0
    assert torch.count_nonzero(eager[0]["dx_routed"][zero.flatten()]).item() == 0
    assert (
        torch.count_nonzero(eager[0]["d_weights"][zero]).item() > 0
    )  # defined at a zero weight
    assert torch.count_nonzero(quotient["d_weights"][zero]).item() == 0
    assert same(outputs(sequence("operands", ones)), eager[0]) == []
    scaled_zero = outputs(sequence("operands", zeros))
    assert same(scaled_zero, eager[0]) == list(SHARED_OPERAND_OUTPUTS)
    for name, start in zip(
        ("dw_shared_gate", "dw_shared_up", "dw_shared_down"), initial[:3], strict=True
    ):
        assert torch.equal(_bits(scaled_zero[name]), _bits(start)), name
    for name, scale in (("direct", None), ("operands", p["scale"])):
        reference = outputs(sequence(name, scale))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            sequence(name, scale)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = sequence(name, scale)

        def replay():
            graph.replay()
            torch.cuda.synchronize()
            return outputs(captured)

        assert same(replay(), reference) == [] and same(replay(), reference) == []
        g = torch.Generator().manual_seed(7401 + hidden)
        for value in (x, dy):
            value.copy_((randn(value.shape, generator=g) * 0.125).bfloat16())
        scores.copy_(rand(scores.shape, generator=g) * 1.5)
        assert same(replay(), outputs(sequence(name, scale))) == []


def _group_adapter(ep, experts, tokens, hidden, macro, mini, device, counts=None):
    """The group's adapter as rank 0 of its EP group emulated on one GPU (:class:`OneGpuGroup`, every route on this
    rank), prepared for the context and the context-free forward and the recompute; its workspace (at the worst-case
    multiplier; with ``counts`` a source workspace for those counts) and its config."""
    functional = backend.MoKFunctional(ep, experts, GROUP_TOPK, **GROUP_OK)
    functional.communication = OneGpuGroup()
    functional.prepare(None, save_context=False)
    functional.prepare(None)
    functional.prepare_recompute(None)
    if counts is None:
        multiplier = bound_multiplier(tokens, GROUP_TOPK, ep, experts)
        capacity = workspace_module.schedule_capacity_for(
            tokens, GROUP_TOPK, ep, multiplier
        )
        workspace = one_gpu_workspace(ep, tokens, hidden, GROUP_TOPK, capacity, device)
    else:
        workspace, multiplier = one_gpu_source_workspace(
            ep, experts, counts, hidden, GROUP_TOPK, device
        )
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    return functional, workspace, config


def _group_step(
    functional,
    config,
    workspace,
    p,
    accumulators,
    *,
    checkpointed,
    shared_scale,
    schedule=None,
):
    """One training step of the group's adapter on the inputs ``p`` of :func:`_cpu_problem`: ``build_schedule`` (unless
    ``schedule`` is given), the context forward (or the context-free forward and the recompute), ``backward`` into the
    zeroed accumulators.  Returns ``(y, dx, d_weights, routed gate, up, down, shared gate, up, down weight gradients,
    d_shared_scale)`` and the schedule."""
    experts = p["weights"][3].shape[0]
    if schedule is None:
        schedule = functional.build_schedule(
            workspace, config, p["ids"], num_local_experts=experts
        )
    for accumulator in accumulators:
        accumulator.zero_()
    weights = p["weights"]
    y, context = functional.forward(
        config, workspace, schedule, p["x"], p["scores"], *weights, save_context=not checkpointed,
        shared_scale=shared_scale,
    )  # fmt: skip
    if checkpointed:
        context = functional.recompute_forward_context(
            config, workspace, schedule, p["x"], weights[0], weights[1], weights[3], weights[4]
        )  # fmt: skip
    grads = functional.backward(
        config, workspace, schedule, context, p["dy"], p["x"], p["scores"], *weights,
        weight_grad_accumulators=accumulators, shared_scale=shared_scale,
    )  # fmt: skip
    assert len(grads) == 9
    return (y, *grads), schedule


def _snapshot(values):
    torch.cuda.synchronize()
    out = tuple(value.clone() for value in values)
    for index, value in enumerate(out):
        assert torch.isfinite(value).all().item(), index
    return out


#: Indexes of ``_group_step``'s values.
Y, DX, D_WEIGHTS, ROUTED_DW, SHARED_DW_INDEX, D_SHARED_SCALE = (
    0,
    1,
    2,
    (3, 4, 5),
    (6, 7, 8),
    9,
)


def test_scaled_group_adapter_step():
    """The group's adapter step at EP8 ``(8, 64)``, top-10, emulated on one GPU as rank 0 (512 source rows, several
    macrobatches): nine backward items, ``d_shared_scale`` FP32 ``[tokens]``; the context step and the checkpointed step
    (context-free forward, recompute, backward) give the same bytes, run to run and in a CUDA Graph replay; a zero
    router weight has a nonzero gradient.  The scale enters where the contract puts it: ``d_weights``, the routed weight
    gradients and ``d_shared_scale`` do not depend on ``s``; with ``s = 0`` everywhere the shared weight gradients are
    zero; a token's ``y`` and ``dx`` rows with ``s = 0`` (``s = 1``) have the bytes of an all-zero (all-one) step's.  A
    steady-state step (``build_schedule``, context-free forward, recompute, backward) makes no host synchronization
    (``torch.cuda.set_sync_debug_mode("error")``).  Accuracy: ``test_mok_accuracy.py``."""
    ep, experts = GROUP_LAYOUT
    for role in (*GROUP_KERNELS, f"count_{ep}_{experts}"):
        _require_role(role)
    tokens, hidden, intermediate, macro, mini = 512, 512, 256, 1536, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional, workspace, config = _group_adapter(
        ep, experts, tokens, hidden, macro, mini, device
    )
    p = _cpu_problem(tokens, hidden, intermediate, experts, GROUP_TOPK, 7501, device)
    accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"]]
    scale = p["scale"]

    def step(checkpointed=False, shared_scale=scale, schedule=None):
        values, schedule = _group_step(
            functional, config, workspace, p, accumulators, checkpointed=checkpointed, shared_scale=shared_scale,
            schedule=schedule,
        )  # fmt: skip
        return values, schedule

    values, schedule = step()
    first = _snapshot(values)
    assert (
        int(schedule.num_tokens.item()) > macro
    )  # several macrobatches: backward replays the later ones
    assert (
        first[D_SHARED_SCALE].shape == (tokens,)
        and first[D_SHARED_SCALE].dtype == torch.float32
    )
    assert _same_bits(first, _snapshot(step(schedule=schedule)[0]))
    assert _same_bits(first, _snapshot(step(True, schedule=schedule)[0]))
    zero_weight = p["scores"] == 0
    assert torch.count_nonzero(first[D_WEIGHTS][zero_weight]).item() > 0
    assert (
        torch.count_nonzero(first[D_SHARED_SCALE][scale == 0]).item() > 0
    )  # defined at s = 0
    zero = _snapshot(step(shared_scale=torch.zeros_like(scale), schedule=schedule)[0])
    one = _snapshot(step(shared_scale=torch.ones_like(scale), schedule=schedule)[0])
    for index in (D_WEIGHTS, *ROUTED_DW, D_SHARED_SCALE):
        assert torch.equal(_bits(zero[index]), _bits(first[index])), index
        assert torch.equal(_bits(one[index]), _bits(first[index])), index
    for index in SHARED_DW_INDEX:
        assert torch.count_nonzero(zero[index]).item() == 0, index
    for index in (Y, DX):
        for value, rows in ((zero, scale == 0), (one, scale == 1)):
            assert torch.equal(_bits(first[index][rows]), _bits(value[index][rows])), (
                index
            )
    # Steady state: every kernel loaded, the schedule's load observation in flight.
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        steady, _ = step(True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert _same_bits(first, _snapshot(steady))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(True, schedule=schedule)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, _ = step(True, schedule=schedule)
    for _ in range(2):
        graph.replay()
        assert _same_bits(first, _snapshot(captured))


def test_scaled_group_adapter_changed_counts():
    """The group's adapter over changing source counts on one source workspace (:data:`CHANGED_COUNTS`: 1031, 768, 257
    and 513 rows), as rank 0 of EP8 emulated on one GPU: per count a new schedule, nine backward items with the
    count's rows, and before each step every workspace row past the count holds other values, so the staging alone
    decides those rows: the context step and the checkpointed step give exactly the bytes of the context step on a
    fresh zeroed workspace; no count after the first loads a kernel."""
    ep, experts = GROUP_LAYOUT
    for role in (*GROUP_KERNELS, f"count_{ep}_{experts}"):
        _require_role(role)
    hidden, intermediate, macro, mini = 512, 256, 1536, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional, workspace, config = _group_adapter(
        ep, experts, None, hidden, macro, mini, device, CHANGED_COUNTS
    )
    fresh_workspaces = []  # alive to the end: no later workspace takes a freed one's address
    loads = None
    for index, count in enumerate(CHANGED_COUNTS):
        p = _cpu_problem(
            count, hidden, intermediate, experts, GROUP_TOPK, 7601 + index, device
        )
        accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"]]

        def step(ws, checkpointed):
            values, schedule = _group_step(
                functional, config, ws, p, accumulators, checkpointed=checkpointed, shared_scale=p["scale"]
            )  # fmt: skip
            assert (
                schedule.num_source_tokens == count
                and int(schedule.num_tokens.item()) > macro
            )
            return _snapshot(values)

        def poison(ws):
            storage = ws.storage
            for buffer, rows, value in (
                (storage.x_buffer, count, 3.0),
                (storage.d_y_buffer, count, -5.0),
                (storage.router_weight_buffer, count, 0.75),
                (storage.d_router_weight_buffer, count, 9.0),
                (storage.combine_buffer, count * GROUP_TOPK, 7.0),
                (storage.d_x_routed_buffer, count * GROUP_TOPK, -7.0),
            ):
                buffer[rows:].fill_(value)

        fresh_workspaces.append(
            one_gpu_source_workspace(
                ep, experts, CHANGED_COUNTS, hidden, GROUP_TOPK, device
            )[0]
        )
        reference = step(fresh_workspaces[-1], False)
        shapes = [tuple(reference[i].shape) for i in (Y, DX, D_WEIGHTS, D_SHARED_SCALE)]
        assert shapes == [
            (count, hidden),
            (count, hidden),
            (count, GROUP_TOPK),
            (count,),
        ]
        for checkpointed in (False, True):
            poison(workspace)
            values = step(workspace, checkpointed)
            differs = [
                i
                for i, (a, b) in enumerate(zip(reference, values, strict=True))
                if not _same_bits((a,), (b,))
            ]
            assert not differs, (
                count,
                "checkpointed" if checkpointed else "context",
                differs,
            )
        misses = jit._load.cache_info().misses
        if loads is None:
            loads = misses
        assert misses == loads, (count, "kernel loads", loads, misses)


def test_scaled_group_checkpoint_without_reentrant():
    """A caller's ``torch.autograd.Function`` (the package ships none) around the context-free forward that saves only
    its inputs and whose backward recomputes the context and runs ``backward``, under
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` (which reruns the forward in the backward pass):
    ``x.grad``, the router-weight gradient, ``d_shared_scale`` and the FP32 weight-gradient accumulators have the bytes
    of the adapter's context step, and ``y`` its ``y``."""
    from torch.utils.checkpoint import checkpoint

    ep, experts = GROUP_LAYOUT
    for role in (*GROUP_KERNELS, f"count_{ep}_{experts}"):
        _require_role(role)
    tokens, hidden, intermediate, macro, mini = 512, 512, 256, 1536, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional, workspace, config = _group_adapter(
        ep, experts, tokens, hidden, macro, mini, device
    )
    p = _cpu_problem(tokens, hidden, intermediate, experts, GROUP_TOPK, 7701, device)
    accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"]]
    values, schedule = _group_step(
        functional, config, workspace, p, accumulators, checkpointed=False, shared_scale=p["scale"]
    )  # fmt: skip
    reference = _snapshot(values)
    weights = p["weights"]

    class Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scores, shared_scale):
            y, _ = functional.forward(
                config, workspace, schedule, x, scores, *weights, save_context=False, shared_scale=shared_scale
            )  # fmt: skip
            ctx.save_for_backward(x, scores, shared_scale)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, scores, shared_scale = ctx.saved_tensors
            context = functional.recompute_forward_context(
                config, workspace, schedule, x, weights[0], weights[1], weights[3], weights[4]
            )  # fmt: skip
            grads = functional.backward(
                config, workspace, schedule, context, dy.contiguous(), x, scores, *weights,
                weight_grad_accumulators=accumulators, shared_scale=shared_scale,
            )  # fmt: skip
            return grads[0].clone(), grads[1].clone(), grads[8].clone()

    leaves = [
        p[name].detach().clone().requires_grad_(True)
        for name in ("x", "scores", "scale")
    ]
    for accumulator in accumulators:
        accumulator.zero_()
    y = checkpoint(Layer.apply, *leaves, use_reentrant=False)
    y_bits = y.detach().clone()
    y.backward(p["dy"])
    torch.cuda.synchronize()
    assert torch.equal(_bits(y_bits), _bits(reference[Y]))
    for leaf, index in zip(leaves, (DX, D_WEIGHTS, D_SHARED_SCALE), strict=True):
        assert torch.equal(_bits(leaf.grad), _bits(reference[index])), index
    order = (
        *SHARED_DW_INDEX,
        *ROUTED_DW,
    )  # accumulators: shared gate, up, down, routed gate, up, down
    for accumulator, index in zip(accumulators, order, strict=True):
        assert torch.equal(_bits(accumulator), _bits(reference[index])), index


#: The kernels the 896-expert top-16 group's GPU tests launch.
SITU_KERNELS = (
    "forward_situ_noshared",
    "forward_noctx_situ_noshared",
    "recompute_situ_noshared",
    "backward_fp32_direct_situ_noshared",
    "epilogue_forward_none_16",
    "epilogue_backward_none_16",
)


def _require_situ_group():
    ep, experts = SITU_GROUP.layouts[0]
    for role in (*SITU_KERNELS, f"count_{ep}_{experts}"):
        _require_role(role)


@pytest.mark.parametrize("tokens,hidden", [(1, 1032), (37, 4096), (512, 256)])
def test_none_epilogues_on_the_gpu(tokens, hidden):
    """The top-16 epilogues without a shared row.  The backward sum is exact: ``BF16(routed_0 + ... + routed_15)`` in
    FP32 in that order, bit for bit the CPU's.  The forward with the last eight slots zero (zero rows and zero weights)
    has the bytes of the top-8 epilogue on the first eight slots and a zero shared row (both start from zero, then add
    the slots in ascending order, and ``fma(0, 0, acc) = acc``).  Five launches and a CUDA Graph replay of the direct
    and the planned launches give the same bytes.  The forward's accuracy against the float64 oracle is judged in
    ``test_mok_accuracy.py``."""
    _require_role("epilogue_forward_none_16")
    _require_role("epilogue_backward_8")
    device = torch.device("cuda", torch.cuda.current_device())
    none, top8 = MoKEpilogues(16, shared_row="none"), MoKEpilogues(8)
    g = torch.Generator().manual_seed(7151 + tokens + hidden)
    routed = (randn(tokens, 16, hidden, generator=g) * 0.125).bfloat16()
    scores = rand(tokens, 16, generator=g) + 0.01
    expected_dx = torch.zeros(tokens, hidden)
    for slot in range(16):
        expected_dx = expected_dx + routed[:, slot].float()
    padded, padded_scores = routed.clone(), scores.clone()
    padded[:, 8:], padded_scores[:, 8:] = 0.0, 0.0
    d = {name: value.to(device) for name, value in dict(
        routed=routed.view(-1, hidden), scores=scores, padded=padded.view(-1, hidden), padded_scores=padded_scores,
        routed8=routed[:, :8].reshape(-1, hidden), scores8=scores[:, :8].contiguous(), zeros=torch.zeros(tokens, hidden),
    ).items()}  # fmt: skip
    zeros = d["zeros"].bfloat16()
    forward_plan, backward_plan = none.plan(d["routed"], d["scores"]), none.plan(d["routed"], False)

    def launch():
        with kernels._ffi_stream():
            planned = (
                MoKEpilogues.launch_planned(forward_plan, None, d["scores"]),
                MoKEpilogues.launch_planned(backward_plan, None),
            )
        return (none.forward(None, d["routed"], d["scores"]), none.backward(None, d["routed"]), *planned)

    eager = [tuple(value.clone() for value in launch()) for _ in range(REPEATS)]
    assert all(_same_bits(eager[0], other) for other in eager[1:])
    assert _same_bits(eager[0][:2], eager[0][2:])
    assert eager[0][0].shape == (tokens, hidden) and eager[0][1].shape == (tokens, hidden)
    assert torch.equal(_bits(eager[0][1].cpu()), _bits(expected_dx.bfloat16()))
    padded_y = none.forward(None, d["padded"], d["padded_scores"])
    assert torch.equal(_bits(padded_y), _bits(top8.forward(zeros, d["routed8"], d["scores8"])))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = launch()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert _same_bits(captured, eager[0])


def _situ_problem(tokens, hidden, intermediate, experts, topk, seed, device):
    """:func:`_cpu_problem` without a shared expert: ``None`` for the three shared weights and every route on this
    rank's ``experts`` (:func:`one_rank_ids`; at top-16 over 14 experts two routes of each token repeat an expert)."""
    p = _cpu_problem(tokens, hidden, intermediate, experts, topk, seed, device)
    p["weights"] = (None, None, None, *p["weights"][3:])
    p["ids"] = one_rank_ids(tokens, experts, topk, torch.Generator().manual_seed(seed + 1)).to(device)
    return p


def _situ_adapter(tokens, hidden, macro, mini, device, counts=None, layout=None):
    """The 896-expert top-16 group's adapter as rank 0 of EP64 ``(64, 14)`` (or of the group's ``layout``) emulated on
    one GPU (:class:`OneGpuGroup`, every route on this rank), prepared for the context and the context-free forward and
    the recompute; its workspace (at the multiplier of every route on one rank, :func:`one_rank_multiplier`; with
    ``counts`` a source workspace for those counts) and its config."""
    ep, experts = SITU_GROUP.layouts[0] if layout is None else layout
    topk = SITU_GROUP.topk
    functional = backend.MoKFunctional(ep, experts, topk, **SITU_OK)
    functional.communication = OneGpuGroup()
    functional.prepare(None, save_context=False)
    functional.prepare(None)
    functional.prepare_recompute(None)
    if counts is None:
        multiplier = one_rank_multiplier(tokens, topk, ep, experts)
        capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, multiplier)
        workspace = one_gpu_workspace(ep, tokens, hidden, topk, capacity, device)
    else:
        workspace, multiplier = one_gpu_source_workspace(
            ep, experts, counts, hidden, topk, device, multiplier_of=one_rank_multiplier
        )
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    return functional, workspace, config


def _situ_step(functional, config, workspace, p, accumulators, *, checkpointed, situ=None, schedule=None):
    """One training step of the SiTU-GLU group's adapter on the inputs ``p`` of :func:`_situ_problem`:
    ``build_schedule`` (unless ``schedule`` is given), the context forward (or the context-free forward and the
    recompute), ``backward`` into the three zeroed routed accumulators, with the SiTU-GLU scalars ``situ`` (default
    :data:`SITU`).  Returns ``(y, dx, d_weights, routed gate, up, down weight gradients)`` and the schedule; the
    context's shared fields and the backward's three shared weight gradients are ``None``."""
    situ = SITU if situ is None else situ
    routed = p["weights"][3:]
    if schedule is None:
        schedule = functional.build_schedule(workspace, config, p["ids"], num_local_experts=routed[0].shape[0])
    for accumulator in accumulators:
        accumulator.zero_()
    y, context = functional.forward(
        config, workspace, schedule, p["x"], p["scores"], None, None, None, *routed, save_context=not checkpointed,
        **situ,
    )  # fmt: skip
    if checkpointed:
        assert context is None
        context = functional.recompute_forward_context(
            config, workspace, schedule, p["x"], None, None, routed[0], routed[1], **situ
        )  # fmt: skip
    assert context.gate_shared is None and context.up_shared is None and context.hidden_shared is None
    grads = functional.backward(
        config, workspace, schedule, context, p["dy"], p["x"], p["scores"], None, None, None, *routed,
        weight_grad_accumulators=accumulators, **situ,
    )  # fmt: skip
    assert len(grads) == 8 and all(value is None for value in grads[5:])
    assert all(grads[2 + i] is accumulator for i, accumulator in enumerate(accumulators))
    return (y, *grads[:5]), schedule


#: Indexes of ``_situ_step``'s values.
SITU_Y, SITU_DX, SITU_D_WEIGHTS, SITU_ROUTED_DW = 0, 1, 2, (3, 4, 5)


def test_situ_group_adapter_step():
    """The 896-expert top-16 group's adapter step (SiTU-GLU, no shared expert) at EP64 ``(64, 14)``, emulated on one
    GPU as rank 0 (every route on this rank, two routes of each token repeating an expert; 256 source rows, several
    macrobatches): ``y``, ``dx``, ``d_weights`` and the three routed FP32 weight gradients, ``None`` in the shared
    places; the context step and the checkpointed step (context-free forward, recompute, backward) give the same bytes,
    run to run and in a CUDA Graph replay; a zero router weight has a nonzero gradient.  ``situ_beta`` and ``situ_lb``
    are run-time values: other values change ``y``, ``dx``, ``d_weights`` and every weight gradient, and a graph
    captured with :data:`SITU` replays those values after a step with other ones.  A steady-state step
    (``build_schedule``, context-free forward, recompute, backward) makes no host synchronization
    (``torch.cuda.set_sync_debug_mode("error")``).  Accuracy: ``test_mok_accuracy.py``."""
    _require_situ_group()
    topk = SITU_GROUP.topk
    tokens, hidden, intermediate, macro, mini = 256, 512, 256, 8192, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional, workspace, config = _situ_adapter(tokens, hidden, macro, mini, device)
    experts = SITU_GROUP.layouts[0][1]
    p = _situ_problem(tokens, hidden, intermediate, experts, topk, 7801, device)
    accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"][3:]]

    def step(checkpointed=False, situ=None, schedule=None):
        return _situ_step(
            functional, config, workspace, p, accumulators, checkpointed=checkpointed, situ=situ, schedule=schedule
        )  # fmt: skip

    values, schedule = step()
    first = _snapshot(values)
    assert int(schedule.num_tokens.item()) > macro  # several macrobatches: backward replays the later ones
    assert first[SITU_Y].shape == (tokens, hidden) and first[SITU_D_WEIGHTS].shape == (tokens, topk)
    assert _same_bits(first, _snapshot(step(schedule=schedule)[0]))
    assert _same_bits(first, _snapshot(step(True, schedule=schedule)[0]))
    zero_weight = p["scores"] == 0
    assert torch.count_nonzero(first[SITU_D_WEIGHTS][zero_weight]).item() > 0
    other = _snapshot(step(situ=dict(situ_beta=3.0, situ_lb=12.0), schedule=schedule)[0])
    for index in (SITU_Y, SITU_DX, SITU_D_WEIGHTS, *SITU_ROUTED_DW):
        assert not torch.equal(_bits(other[index]), _bits(first[index])), index
    # Steady state: every kernel loaded, the schedule's load observation in flight.
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        steady, _ = step(True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert _same_bits(first, _snapshot(steady))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(True, schedule=schedule)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, _ = step(True, schedule=schedule)
    step(True, situ=dict(situ_beta=3.0, situ_lb=12.0), schedule=schedule)
    for _ in range(2):
        graph.replay()
        assert _same_bits(first, _snapshot(captured))


def test_situ_group_adapter_changed_counts():
    """The SiTU-GLU group's adapter over changing source counts on one source workspace (:data:`CHANGED_COUNTS`: 1031,
    768, 257 and 513 rows), as rank 0 of EP64 emulated on one GPU: per count a new schedule, the six step values with
    the count's rows, and before each step every workspace row past the count holds other values, so the staging alone
    decides those rows: the context step and the checkpointed step give exactly the bytes of the context step on a
    fresh zeroed workspace; no count after the first loads a kernel."""
    _require_situ_group()
    topk = SITU_GROUP.topk
    experts = SITU_GROUP.layouts[0][1]
    hidden, intermediate, macro, mini = 512, 256, 16384, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional, workspace, config = _situ_adapter(None, hidden, macro, mini, device, CHANGED_COUNTS)
    fresh_workspaces = []  # alive to the end: no later workspace takes a freed one's address
    loads = None
    for index, count in enumerate(CHANGED_COUNTS):
        p = _situ_problem(count, hidden, intermediate, experts, topk, 7901 + index, device)
        accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"][3:]]

        def step(ws, checkpointed):
            values, schedule = _situ_step(functional, config, ws, p, accumulators, checkpointed=checkpointed)
            assert schedule.num_source_tokens == count and int(schedule.num_tokens.item()) > macro
            return _snapshot(values)

        def poison(ws):
            storage = ws.storage
            for buffer, rows, value in (
                (storage.x_buffer, count, 3.0),
                (storage.d_y_buffer, count, -5.0),
                (storage.router_weight_buffer, count, 0.75),
                (storage.d_router_weight_buffer, count, 9.0),
                (storage.combine_buffer, count * topk, 7.0),
                (storage.d_x_routed_buffer, count * topk, -7.0),
            ):
                buffer[rows:].fill_(value)

        fresh_workspaces.append(
            one_gpu_source_workspace(
                *SITU_GROUP.layouts[0], CHANGED_COUNTS, hidden, topk, device, multiplier_of=one_rank_multiplier
            )[0]
        )
        reference = step(fresh_workspaces[-1], False)
        shapes = [tuple(reference[i].shape) for i in (SITU_Y, SITU_DX, SITU_D_WEIGHTS)]
        assert shapes == [(count, hidden), (count, hidden), (count, topk)]
        for checkpointed in (False, True):
            poison(workspace)
            values = step(workspace, checkpointed)
            differs = [
                i for i, (a, b) in enumerate(zip(reference, values, strict=True)) if not _same_bits((a,), (b,))
            ]
            assert not differs, (count, "checkpointed" if checkpointed else "context", differs)
        misses = jit._load.cache_info().misses
        if loads is None:
            loads = misses
        assert misses == loads, (count, "kernel loads", loads, misses)


def test_situ_group_checkpoint_without_reentrant():
    """A caller's ``torch.autograd.Function`` around the SiTU-GLU group's context-free forward that saves only its
    inputs and whose backward recomputes the context and runs ``backward``, under
    ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``: ``x.grad``, the router-weight gradient and the
    three routed FP32 weight-gradient accumulators have the bytes of the adapter's context step, and ``y`` its
    ``y``."""
    from torch.utils.checkpoint import checkpoint

    _require_situ_group()
    topk = SITU_GROUP.topk
    experts = SITU_GROUP.layouts[0][1]
    tokens, hidden, intermediate, macro, mini = 256, 512, 256, 8192, 256
    device = torch.device("cuda", torch.cuda.current_device())
    functional, workspace, config = _situ_adapter(tokens, hidden, macro, mini, device)
    p = _situ_problem(tokens, hidden, intermediate, experts, topk, 8001, device)
    accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"][3:]]
    values, schedule = _situ_step(functional, config, workspace, p, accumulators, checkpointed=False)
    reference = _snapshot(values)
    routed = p["weights"][3:]

    class Layer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scores):
            y, _ = functional.forward(
                config, workspace, schedule, x, scores, None, None, None, *routed, save_context=False, **SITU
            )  # fmt: skip
            ctx.save_for_backward(x, scores)
            return y

        @staticmethod
        def backward(ctx, dy):
            x, scores = ctx.saved_tensors
            context = functional.recompute_forward_context(
                config, workspace, schedule, x, None, None, routed[0], routed[1], **SITU
            )  # fmt: skip
            grads = functional.backward(
                config, workspace, schedule, context, dy.contiguous(), x, scores, None, None, None, *routed,
                weight_grad_accumulators=accumulators, **SITU,
            )  # fmt: skip
            return grads[0].clone(), grads[1].clone()

    leaves = [p[name].detach().clone().requires_grad_(True) for name in ("x", "scores")]
    for accumulator in accumulators:
        accumulator.zero_()
    y = checkpoint(Layer.apply, *leaves, use_reentrant=False)
    y_bits = y.detach().clone()
    y.backward(p["dy"])
    torch.cuda.synchronize()
    assert torch.equal(_bits(y_bits), _bits(reference[SITU_Y]))
    for leaf, index in zip(leaves, (SITU_DX, SITU_D_WEIGHTS), strict=True):
        assert torch.equal(_bits(leaf.grad), _bits(reference[index])), index
    for accumulator, index in zip(accumulators, SITU_ROUTED_DW, strict=True):
        assert torch.equal(_bits(accumulator), _bits(reference[index])), index


@pytest.mark.parametrize("tokens", [4096, 16384, 32768])
@pytest.mark.parametrize("ep,experts", BOUND_LAYOUTS, ids=BOUND_IDS)
def test_capacity_at_the_requested_bound(ep, experts, tokens):
    """Host only: at the requester's worst-case multiplier (top-6, macrobatch 262,144, minibatch 4,096) the schedule
    capacity holds every padded route of any routing and is a multiple of 256, and every fast-divmod divisor and grid
    of the forward, recompute and backward launches fits its 31-bit field (the geometry helpers refuse a divisor
    outside it)."""
    topk, hidden, intermediate, macro, mini, comm = 6, 5120, 2304, 262144, 4096, 40
    multiplier = bound_multiplier(tokens, topk, ep, experts)
    assert multiplier * ep == ep + 1  # (EP + 1) / EP at these shapes
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, multiplier)
    assert capacity == tokens * topk * (ep + 1) and capacity % 256 == 0
    assert capacity >= min(topk, experts) * tokens * ep + experts * 255
    limit = 2**31 - 1
    forward = kernels._forward_geometry(tokens, hidden, intermediate, capacity, macro, mini)
    recompute = kernels._recompute_geometry(tokens, hidden, intermediate, capacity, macro, mini)
    backward = kernels._backward_geometry(
        tokens, hidden, intermediate, capacity, macro, mini, experts
    )
    assert 2 * (forward.shared_tasks + forward.minis * forward.mini_tasks) + comm <= limit
    assert 2 * (recompute.shared_tasks + recompute.minis * recompute.mini_tasks) + comm <= limit
    assert comm + 2 * backward.clusters <= limit
    assert macro * hidden <= limit  # elements of the routed x ring
    # The 3/16 test default overflows under full skew at the 384-expert layouts.
    assert workspace_module.schedule_capacity_for(tokens, topk, ep, 3 / 16) < min(topk, experts) * tokens * ep


def _before_down_problem(tokens, hidden, intermediate, experts, topk, seed):
    """BF16 ``x`` and ``dY``, weights scaled to unit-variance outputs, FP32 router weights in [0, 1.5) with every 5th
    token's first weight and every weight of token 3 exactly 0."""
    torch.manual_seed(seed)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = randn(tokens, hidden, **options) * 0.125
    dy = randn_like(x) * 0.125
    shared = [
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    scores = rand(tokens, topk, device="cuda") * 1.5
    scores[::5, 0] = 0.0
    scores[3] = 0.0
    return x, dy, shared, routed, scores


def _check_before_down_training(
    *,
    experts=4,
    topk=2,
    tokens=512,
    geometries=GEOMETRIES,
    empty_cases=(False, True),
    limit=0.125,
    mini=256,
    comm=4,
):
    """Contract and repeatability of the router-weight-before-down forward + backward (clamped SwiGLU, FP32
    accumulators from nonzero starts): every output finite, :data:`REPEATS` eager launches and graph replays give
    the same bytes, a launch with another communication-SM split gives the same bytes, a replay gives the eager
    launch's bytes (also after the weights and the inputs change), a zero
    router weight gives its route exactly zero combine rows and input gradients, and the router-weight gradient
    reads no weight: zeroing every weight leaves its bits, zeroes every routed output, input gradient and
    routed weight-gradient increment, and leaves the shared expert's outputs."""
    forward = MoKForward(True, weight_before_down=True)
    backward = MoKBackward(True, True, weight_before_down=True)
    records = []
    for empty in empty_cases:
        for hidden, intermediate, macro in geometries:
            x, dy, shared, routed, scores = _before_down_problem(
                tokens, hidden, intermediate, experts, topk, 9173 + hidden
            )
            options = dict(device="cuda", dtype=torch.bfloat16)
            row = torch.arange(tokens, device="cuda")
            routes = torch.stack(
                [(row + slot) % (topk if empty else experts) for slot in range(topk)],
                -1,
            )
            s = single_rank_schedule(routes, experts)
            combine = torch.empty(tokens * topk, hidden, **options)
            dx_peer = torch.empty(tokens * topk, hidden, **options)
            ds_peer = torch.empty_like(scores)
            initial = [randn(w.shape, device="cuda") * 0.01 for w in (*shared, *routed)]
            accumulators = [value.clone() for value in initial]

            def sequence(split=comm):
                for accumulator, start in zip(accumulators, initial, strict=True):
                    accumulator.copy_(start)
                values = forward(
                    x, [x.data_ptr()], combine, [combine.data_ptr()],
                    shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                    s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                    topk, limit, split, macro, mini, weight_ptrs=[scores.data_ptr()],
                )  # fmt: skip
                result = backward(
                    dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
                    scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
                    shared[0], routed[0], shared[1], routed[1], shared[2], routed[2],
                    *values[:7], x, [x.data_ptr()],
                    s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                    topk, limit, split, macro, mini,
                    weight_grad_accumulators=accumulators,
                )  # fmt: skip
                return result, values

            # Shared dx / d_gate / d_up / dh, the six FP32 accumulators (shared gate, routed gate, shared up, routed
            # up, shared down, routed down), the routed dx and the router-weight gradient, the shared and routed
            # forward outputs.
            def outputs(launch):
                result, values = launch
                every = (*result, dx_peer, ds_peer, values[7], combine)
                picked = tuple(every[index] for index in (0, 2, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
                for value in picked:
                    assert torch.isfinite(value).all().item(), (empty, hidden)
                return tuple(value.clone() for value in picked)

            zero = scores == 0
            zero_rows = zero.flatten()
            eager = [outputs(sequence()) for _ in range(REPEATS)]
            assert all(_same(eager[0], other) for other in eager[1:])
            # No bit depends on the communication-SM split; the dispatch's communication CTAs gather the router
            # weights into the row ring.
            assert _same(outputs(sequence(comm + 4)), eager[0])
            assert torch.count_nonzero(eager[0][13][zero_rows]).item() == 0  # combine rows of zero weights
            assert torch.count_nonzero(eager[0][10][zero_rows]).item() == 0  # their input gradients
            assert torch.count_nonzero(eager[0][11][zero]).item() > 0  # their weight gradients: no division
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = sequence()

            def replay():
                graph.replay()
                return outputs(captured)

            replays = [replay() for _ in range(REPEATS)]
            assert all(_same(replays[0], other) for other in replays[1:])
            assert _same(replays[0], eager[0])  # no bit depends on graph replay versus eager launches
            scores.mul_(0.5)
            weighted = replay()
            assert _same(weighted, outputs(sequence()))
            scores.zero_()
            unweighted = replay()
            assert torch.equal(unweighted[11], weighted[11])  # the weight gradient reads no weight
            for index in (5, 7, 9, 10, 13):  # routed increments, routed dx, combine rows
                expected = initial[3 + (index - 5) // 2] if index < 10 else None
                if expected is None:
                    assert torch.count_nonzero(unweighted[index]).item() == 0, index
                else:
                    assert torch.equal(unweighted[index], expected), index
            for index in (0, 1, 2, 3, 4, 6, 8, 12):  # the shared expert's outputs
                assert torch.equal(unweighted[index], weighted[index]), index
            for _ in range(2):
                normal_(x, std=0.125)
                normal_(dy, std=0.125)
                uniform_(scores, 0.0, 1.5)
                assert _same(replay(), outputs(sequence()))
            records.append(dict(empty=empty, hidden=hidden, macro=macro, routed=s["actual"]))
    return records


@pytest.mark.parametrize("geometries", [GEOMETRIES[:1], GEOMETRIES], ids=["one", "all"])
def test_fused_training_before_down(geometries):
    """The router weight before the down projection at top-2 over 4 local experts (the diagnostic layouts)."""
    _require_role("forward_clamped_wbd")
    _check_before_down_training(geometries=geometries)


@pytest.mark.parametrize(
    "experts,geometry,empty",
    [
        (6, (512, 256, 1536), False),
        (6, (2048, 256, 768), False),
        (12, (512, 256, 1536), False),
        (12, (512, 256, 1536), True),
        (24, (512, 256, 1536), False),
        (48, (512, 256, 1536), False),
        (96, (512, 256, 1536), False),
    ],
    ids=["local6", "local6-h2048", "local12", "local12-empty-experts", "local24", "local48", "local96"],
)
def test_fused_training_before_down_top6(experts, geometry, empty):
    """Top-6 over the local expert counts of the 384-expert layouts (EP 64, 32, 16, 8, 4), several macrobatches;
    ``empty`` routes every token to the first 6 experts."""
    _require_role("forward_clamped_wbd")
    _check_before_down_training(
        experts=experts, topk=6, geometries=(geometry,), empty_cases=(empty,)
    )


@pytest.mark.parametrize("experts,topk", [(4, 2), (6, 6)], ids=["top2", "top6"])
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_before_down_recompute_and_context_free_forward(macro, experts, topk):
    """The router-weight-before-down recompute (which gathers the router weights itself) rebuilds the saved context
    bitwise; the context-free forward writes the context forward's y, combine rows, hidden and routed x bitwise in
    every launch; backward from the recomputed context gives the saved context's bits."""
    _require_role("forward_clamped_wbd")
    forward = MoKForward(True, weight_before_down=True)
    context_free = MoKForward(True, context=False, weight_before_down=True)
    recompute = MoKRecompute(True, weight_before_down=True)
    backward = MoKBackward(True, True, weight_before_down=True)
    tokens, hidden, intermediate, mini, comm, limit = 512, 512, 256, 256, 4, 0.125
    x, dy, shared, routed, scores = _before_down_problem(tokens, hidden, intermediate, experts, topk, 4111)
    options = dict(device="cuda", dtype=torch.bfloat16)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    pointers = [scores.data_ptr()]
    rows = min(s["actual"], macro)

    def run(kernel):
        combine = torch.zeros(tokens * topk, hidden, **options)
        values = kernel(
            x, [x.data_ptr()], combine, [combine.data_ptr()], *weights,
            *schedule, topk, limit, comm, macro, mini, weight_ptrs=pointers,
        )  # fmt: skip
        torch.cuda.synchronize()
        return values, combine

    saved, combine = run(forward)
    assert torch.count_nonzero(combine).item() > 0
    for _ in range(3):
        values, free_combine = run(context_free)
        assert values[1:5] == (None,) * 4
        assert torch.equal(free_combine, combine)
        assert torch.equal(values[7], saved[7])  # y_shared
        assert torch.equal(values[5], saved[5])  # hidden_shared
        for index in (0, 6):  # routed x, hidden of the resident macrobatch
            assert torch.equal(values[index][:rows], saved[index][:rows]), index
    rebuilt = recompute(
        x, [x.data_ptr()], shared[0], routed[0], shared[1], routed[1],
        *schedule, topk, limit, comm, macro, mini, weight_ptrs=pointers,
    )  # fmt: skip
    for index in (0, 2, 4, 6):  # routed x, gate, up, hidden of the resident macrobatch
        assert torch.equal(saved[index][:rows], rebuilt[index][:rows]), index
    for index in (1, 3, 5):  # shared gate, up, hidden
        assert torch.equal(saved[index], rebuilt[index]), index

    def gradients(context):
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        accumulators = [torch.zeros(w.shape, device="cuda") for w in (*shared, *routed)]
        values = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            *weights, *context[:7], x, [x.data_ptr()], *schedule,
            topk, limit, comm, macro, mini, weight_grad_accumulators=accumulators,
        )  # fmt: skip
        return (dx_peer, ds_peer, values[0], *values[9:])

    expected = gradients(saved)
    assert all(torch.count_nonzero(value).item() > 0 for value in expected)
    for a, b in zip(expected, gradients(rebuilt), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("experts,topk", [(4, 2), (6, 6)], ids=["top2", "top6"])
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_before_down_source_split_is_bitwise(macro, experts, topk):
    """The router-weight-before-down forward (context and context-free) with the source split taken on one GPU
    (route totals of one route, so the rank's 1024 source rows count as rich; 2, 4 and 6 of 8 communication SMs) and
    held (the schedule's real totals): every launch writes the split-off bytes of y, the combine rows, the shared
    gate, up and hidden and, on the resident macrobatch, the routed x, gate, up and hidden."""
    _require_role("forward_clamped_wbd")
    tokens, hidden, intermediate, mini, comm, limit = 1024, 512, 256, 256, 8, 0.125
    x, _, shared, routed, scores = _before_down_problem(tokens, hidden, intermediate, experts, topk, 4127)
    options = dict(device="cuda", dtype=torch.bfloat16)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 5 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    pointers = [scores.data_ptr()]
    rows = min(s["actual"], macro)
    taken = torch.ones(1, dtype=torch.int32, device="cuda")
    held = torch.full((1,), tokens * topk, dtype=torch.int32, device="cuda")
    for context in (True, False):
        kernel = MoKForward(True, context=context, weight_before_down=True)

        def run(**split):
            combine = torch.zeros(tokens * topk, hidden, **options)
            values = kernel(
                x, [x.data_ptr()], combine, [combine.data_ptr()], *weights,
                *schedule, topk, limit, comm, macro, mini, weight_ptrs=pointers, **split,
            )  # fmt: skip
            torch.cuda.synchronize()
            return values, combine

        base, base_combine = run()
        assert torch.count_nonzero(base_combine).item() > 0
        for split in (
            dict(source_split=(2, 257), owner_routes=taken),
            dict(source_split=(4, 300), owner_routes=taken),
            dict(source_split=(6, 384), owner_routes=taken),
            dict(source_split=(2, 257), owner_routes=held),
        ):
            values, combine = run(**split)
            assert torch.equal(combine, base_combine), split
            for index in (1, 3, 5, 7):  # shared gate, up, hidden, y
                if base[index] is not None:
                    assert torch.equal(values[index], base[index]), (split, index)
            for index in (0, 2, 4, 6):  # routed x, gate, up, hidden of the resident macrobatch
                if base[index] is not None:
                    assert torch.equal(values[index][:rows], base[index][:rows]), (split, index)


def test_before_down_clamp_at_the_limit():
    """The clamped SwiGLU of the router-weight-before-down kernels at exactly the limit 10, on the requester's input
    distribution (:func:`production_inputs`, hidden 2048, 6 local experts, top-6), with one macrobatch holding every
    route, so the backward's routed gate and up gradient rows are the saved context's rows.  Token 1's first expert
    has gate exactly 10 and up exactly +10 and -10, and the forward saves them exactly.  The gradients pass where the
    saved gate is at most the limit and the saved up within it, both ends included: they are nonzero at the pinned
    elements and exactly zero wherever the saved gate is above the limit or the saved up beyond it."""
    _require_role("forward_clamped_wbd")
    tokens, hidden, intermediate, experts, topk, macro, mini, comm, limit = 512, 2048, 256, 6, 6, 4096, 256, 4, 10.0
    p = production_inputs(tokens, hidden, intermediate, experts, topk, limit, seed=8377)
    assert p["pinned"] is not None
    x, dy, scores, shared, routed = p["x"], p["dy"], p["weights"], p["shared"], p["routed"]
    s = single_rank_schedule(p["ids"], experts)
    assert s["peers"].numel() <= macro  # one macrobatch: the backward replays nothing
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.zeros_like(combine), torch.zeros_like(scores)
    accumulators = [torch.zeros(w.shape, device="cuda") for w in (*shared, *routed)]
    context = MoKForward(True, weight_before_down=True)(
        x, [x.data_ptr()], combine, [combine.data_ptr()], *weights,
        *schedule, topk, limit, comm, macro, mini, weight_ptrs=[scores.data_ptr()],
    )  # fmt: skip
    torch.cuda.synchronize()
    slots = s["schedule"][: s["actual"]]
    defined = slots >= 0
    gate, up = context[2][: s["actual"]][defined].clone(), context[4][: s["actual"]][defined].clone()
    token, expert = p["pinned"]
    slot = token * topk + int((p["ids"][token] == expert).nonzero()[0, 0])
    (position,) = (slots[defined] == slot).nonzero().flatten().tolist()
    assert gate[position, 0].item() == limit
    assert up[position, 0].item() == limit and up[position, 1].item() == -limit
    above, beyond = gate > limit, up.abs() > limit
    assert above.any().item() and beyond.any().item()
    result = MoKBackward(True, True, weight_before_down=True)(
        dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
        scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
        *weights, *context[:7], x, [x.data_ptr()], *schedule,
        topk, limit, comm, macro, mini, weight_grad_accumulators=accumulators,
    )  # fmt: skip
    torch.cuda.synchronize()
    d_gate, d_up = result[3][: s["actual"]][defined], result[5][: s["actual"]][defined]
    assert torch.count_nonzero(d_gate[above]).item() == 0
    assert torch.count_nonzero(d_up[beyond]).item() == 0
    assert torch.count_nonzero(d_gate[~above]).item() > 0 and torch.count_nonzero(d_up[~beyond]).item() > 0
    assert d_gate[position, 0].item() != 0
    assert d_up[position, 0].item() != 0 and d_up[position, 1].item() != 0


@pytest.mark.parametrize("ep,experts", BOUND_LAYOUTS, ids=BOUND_IDS)
def test_scheduler_at_the_capacity_bound(ep, experts):
    """At the requester's worst-case multiplier, 16,384 source rows per rank and top-6, full skew (every token of every
    rank routes to rank 0's experts) fits: the scheduler's per-expert counts and rows are exact, the total is within
    the capacity, and another rank receives nothing."""
    _require_role(f"count_{ep}_{experts}")
    tokens, topk = 16384, 6
    capacity = workspace_module.schedule_capacity_for(
        tokens, topk, ep, bound_multiplier(tokens, topk, ep, experts)
    )
    generator = torch.Generator(device="cuda").manual_seed(4409 + ep)
    ids = (
        rand(ep * tokens, experts, device="cuda", generator=generator)
        .argsort(-1)[:, :topk]
        .int()
        .reshape(ep, tokens, topk)
        .contiguous()
    )
    scheduler = MoKScheduler(ep, experts)
    for rank in (0, 1):
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(ids.flatten().long(), minlength=ep * experts)
        expected_counts = expected_counts[rank * experts : (rank + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        assert torch.equal(counts, expected_counts.int())
        assert total.item() == expected_counts.sum().item() <= capacity
        if rank:
            assert total.item() == 0
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == rank * experts + expert).nonzero().flatten()
            assert torch.equal(actual, expected), (rank, expert)
            offset += count
        assert (peers[offset:] == -1).all()
        assert torch.equal(
            scheduler.owner_loads().long(), backend.count_routed_load(ids, ep, experts)
        )
    scheduler.release_counts()


@pytest.mark.parametrize("tokens", [4096, 16384, 32768])
def test_situ_capacity_at_the_requested_bound(tokens):
    """Host only: at the 896-expert layout (64, 14) and top-16 (more routes per token than local experts) the
    worst-case multiplier counts at most 14 routes of a token on one rank: ``57 / 64`` at 16,384 source rows; the
    schedule capacity holds every padded route of any routing with distinct experts per token and is a multiple of
    256, and every fast-divmod divisor and grid of the forward, recompute and backward launches without a shared
    expert (``local_tokens`` 0, no shared task) fits its 31-bit field at hidden 3584, intermediate 3072, macrobatch
    262,144 and minibatch 4,096."""
    (ep, experts), topk = SITU_GROUP.layouts[0], SITU_GROUP.topk
    hidden, intermediate, macro, mini, comm = 3584, 3072, 262144, 4096, 40
    multiplier = bound_multiplier(tokens, topk, ep, experts)
    if tokens == 16384:
        assert multiplier == 57 / 64
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, multiplier)
    assert capacity % 256 == 0
    assert capacity >= experts * tokens * ep + experts * 255
    limit = 2**31 - 1
    forward = kernels._forward_geometry(0, hidden, intermediate, capacity, macro, mini)
    recompute = kernels._recompute_geometry(0, hidden, intermediate, capacity, macro, mini)
    backward = kernels._backward_geometry(0, hidden, intermediate, capacity, macro, mini, experts, False)
    assert forward.shared_tasks == recompute.shared_tasks == backward.shared_tasks == 0
    assert 2 * forward.minis * forward.mini_tasks + comm <= limit
    assert 2 * recompute.minis * recompute.mini_tasks + comm <= limit
    assert comm + 2 * backward.clusters <= limit
    assert macro * hidden <= limit  # elements of the routed x ring


def test_situ_scheduler_at_the_capacity_bound():
    """At the 896-expert layout (64, 14), top-16, 16,384 source rows per rank and the worst-case multiplier (14 of a
    token's 16 routes on one rank), the two-rank full skew (:func:`two_rank_skew_ids`: every token of every rank routes
    to all 14 experts of rank 0 and to two distinct experts of rank 1) fits: the scheduler's per-expert counts and rows
    are exact on rank 0 (the bound) and rank 1 (the rest), each total within the capacity, and rank 2 receives
    nothing."""
    (ep, experts), topk = SITU_GROUP.layouts[0], SITU_GROUP.topk
    _require_role(f"count_{ep}_{experts}")
    tokens = 16384
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, bound_multiplier(tokens, topk, ep, experts))
    ids = two_rank_skew_ids(ep * tokens, experts, topk, torch.Generator().manual_seed(4421), "cuda")
    ids = ids.reshape(ep, tokens, topk).contiguous()
    assert bool((ids.sort(-1).values.diff(dim=-1) != 0).all())  # distinct experts per token
    scheduler = MoKScheduler(ep, experts)
    for rank in (0, 1, 2):
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(ids.flatten().long(), minlength=ep * experts)
        expected_counts = expected_counts[rank * experts : (rank + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        assert torch.equal(counts, expected_counts.int())
        assert total.item() == expected_counts.sum().item() <= capacity
        if rank == 0:
            assert total.item() >= experts * tokens * ep
        if rank == 2:
            assert total.item() == 0
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == rank * experts + expert).nonzero().flatten()
            assert torch.equal(actual, expected), (rank, expert)
            offset += count
        assert (peers[offset:] == -1).all()
        assert torch.equal(scheduler.owner_loads().long(), backend.count_routed_load(ids, ep, experts))
    scheduler.release_counts()


#: The layouts of each :data:`MODE_GROUPS` group after its first, whose tests above run at the first layout:
#: ``(group name, (EP, local experts))``.  Each has at least as many local experts as the group's top-k.
ADDED_GROUP_LAYOUTS = [
    (name, layout) for name, group in kernels.MODE_GROUPS.items() for layout in group.layouts[1:]
]
ADDED_GROUP_IDS = [f"{name}-ep{ep}-local{experts}" for name, (ep, experts) in ADDED_GROUP_LAYOUTS]


def _require_group_layout(name, layout):
    """Skip on a device without the group's kernels or the layout's scheduler kernels (:func:`_require_role`)."""
    ep, experts = layout
    roles = SITU_KERNELS if kernels.MODE_GROUPS[name].activation == "situ_glu" else GROUP_KERNELS
    for role in (*roles, f"count_{ep}_{experts}", f"rows_{ep}_{experts}"):
        _require_role(role)


@pytest.mark.parametrize("name,layout", ADDED_GROUP_LAYOUTS, ids=ADDED_GROUP_IDS)
def test_group_layout_adapter_step(name, layout):
    """The adapter step of a group at one of its later layouts, emulated on one GPU as rank 0 (every route on this
    rank, distinct experts per token, several macrobatches): the values of :func:`_group_step` (the 512-expert group,
    512 source rows) or of :func:`_situ_step` (the 896-expert group, 256 source rows); the context step and the
    checkpointed step (context-free forward, recompute, backward) give the same bytes, run to run and in a CUDA Graph
    replay; a zero router weight has a nonzero gradient; a steady-state step (``build_schedule``, context-free forward,
    recompute, backward) makes no host synchronization (``torch.cuda.set_sync_debug_mode("error")``).  Accuracy:
    ``test_mok_accuracy.py``."""
    _require_group_layout(name, layout)
    ep, experts = layout
    device = torch.device("cuda", torch.cuda.current_device())
    if name == "scaled_top10":
        tokens, hidden, intermediate, macro, mini = 512, 512, 256, 1536, 256
        functional, workspace, config = _group_adapter(ep, experts, tokens, hidden, macro, mini, device)
        p = _cpu_problem(tokens, hidden, intermediate, experts, GROUP_TOPK, 7501, device)
        accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"]]
        d_weights = D_WEIGHTS

        def step(checkpointed=False, schedule=None):
            return _group_step(
                functional, config, workspace, p, accumulators, checkpointed=checkpointed, shared_scale=p["scale"],
                schedule=schedule,
            )  # fmt: skip

    else:
        tokens, hidden, intermediate, macro, mini = 256, 512, 256, 8192, 256
        functional, workspace, config = _situ_adapter(tokens, hidden, macro, mini, device, layout=layout)
        p = _situ_problem(tokens, hidden, intermediate, experts, SITU_GROUP.topk, 7801, device)
        accumulators = [torch.zeros(w.shape, device=device) for w in p["weights"][3:]]
        d_weights = SITU_D_WEIGHTS

        def step(checkpointed=False, schedule=None):
            return _situ_step(
                functional, config, workspace, p, accumulators, checkpointed=checkpointed, schedule=schedule
            )  # fmt: skip

    values, schedule = step()
    first = _snapshot(values)
    assert int(schedule.num_tokens.item()) > macro  # several macrobatches: backward replays the later ones
    assert _same_bits(first, _snapshot(step(schedule=schedule)[0]))
    assert _same_bits(first, _snapshot(step(True, schedule=schedule)[0]))
    assert torch.count_nonzero(first[d_weights][p["scores"] == 0]).item() > 0
    # Steady state: every kernel loaded, the schedule's load observation in flight.
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        steady, _ = step(True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert _same_bits(first, _snapshot(steady))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(True, schedule=schedule)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured, _ = step(True, schedule=schedule)
    for _ in range(2):
        graph.replay()
        assert _same_bits(first, _snapshot(captured))


@pytest.mark.parametrize("name,layout", ADDED_GROUP_LAYOUTS, ids=ADDED_GROUP_IDS)
def test_group_layout_scheduler_at_the_capacity_bound(name, layout):
    """At a group's later layout, its top-k, 16,384 source rows per rank and the worst-case multiplier
    (:func:`bound_multiplier`), the full skew to rank 0 (every token of every rank routes to top-k distinct experts of
    rank 0) fits: the scheduler's per-expert counts and rows are exact on rank 0 (every route) and on rank 1
    (nothing), each total within the capacity."""
    _require_group_layout(name, layout)
    (ep, experts), topk = layout, kernels.MODE_GROUPS[name].topk
    tokens = 16384
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, bound_multiplier(tokens, topk, ep, experts))
    order = rand(ep * tokens, experts, generator=torch.Generator().manual_seed(4423)).argsort(-1)
    ids = order[:, :topk].int().reshape(ep, tokens, topk).to("cuda").contiguous()
    scheduler = MoKScheduler(ep, experts)
    for rank in (0, 1):
        peers, slots, total, counts = scheduler(ids, capacity, rank)
        expected_counts = torch.bincount(ids.flatten().long(), minlength=ep * experts)
        expected_counts = expected_counts[rank * experts : (rank + 1) * experts]
        expected_counts = ((expected_counts + 255) // 256) * 256
        assert torch.equal(counts, expected_counts.int())
        assert total.item() == expected_counts.sum().item() <= capacity
        assert total.item() >= topk * tokens * ep if rank == 0 else total.item() == 0
        offset = 0
        for expert, count in enumerate(expected_counts.tolist()):
            p = peers[offset : offset + count].long()
            t = slots[offset : offset + count].long()
            valid = p >= 0
            actual = (p[valid] * tokens * topk + t[valid]).sort().values
            expected = (ids.flatten() == rank * experts + expert).nonzero().flatten()
            assert torch.equal(actual, expected), (rank, expert)
            offset += count
        assert (peers[offset:] == -1).all()
        assert torch.equal(scheduler.owner_loads().long(), backend.count_routed_load(ids, ep, experts))
    scheduler.release_counts()


def _exact_outer(gradient, known, copies, known_on_rows):
    """``gradient`` (FP32 ``[m, n]``) is exactly ``copies`` times the outer product of a BF16 vector with the BF16
    vector ``known`` (along its columns, or its rows with ``known_on_rows``), checked in float64 (``copies`` a power
    of 2)."""
    value = gradient.double() / copies
    if known_on_rows:
        value = value.T
    known = known.double()
    column = int(known.abs().argmax())
    vector = value[:, column] / known[column]
    assert torch.count_nonzero(vector).item() > 0
    assert torch.equal(vector, vector.bfloat16().double())
    assert torch.equal(value, torch.outer(vector, known))


def _check_routed_weight_gradients(run, configs, schedule, ids, scores, x, dy, high, macro, copies, experts):
    """The routed weight gradients of the capacity-bound step, from ring rows past 2^31 bytes, exactly.

    Only tokens that share no expert keep their router weights; every other weight is 0, so its routes have
    ``h = BF16(0 * a) = 0`` and ``d_gate = d_up = 0`` and add exactly 0 to every routed weight gradient.  The first
    chosen token has routes in the resident macrobatch's ring rows from byte offset 2^31 on (``high`` .. ``macro``).
    Every element of a chosen expert's gradient is then ``copies`` (the EP size: every emulated peer sends the same
    route) times one BF16 x BF16 product, exact in FP32 in any order and macrobatch split.  ``run(config, router)``
    returns the routed gate, up and down weight gradients of one step: the two ``configs`` (macrobatches of 262,144
    and 32,768 rows) give equal values, an expert no chosen token routes to has a zero gradient, and a chosen expert's
    gradient is exactly ``copies`` times the outer product of a BF16 vector with its token's ``x`` (gate, up) or
    ``dy`` (down)."""
    rows = slice(high, macro)
    above = schedule.peer_token_idx[rows][schedule.peer_rank[rows] >= 0].long() // ids.shape[1]
    routes = ids.tolist()
    owner = {}
    for token in [int(above.min().item()), *range(len(routes))]:
        if len(owner) == experts:
            break
        if not owner.keys() & set(routes[token]):
            owner.update(dict.fromkeys(routes[token], token))
    chosen = sorted(set(owner.values()))
    router = torch.zeros_like(scores)
    router[chosen] = scores[chosen]
    gradients = run(configs[0], router)
    for config in configs[1:]:
        assert _same(gradients, run(config, router))
    touched = torch.zeros(experts, dtype=torch.bool)
    touched[list(owner)] = True
    for gradient in gradients:
        assert gradient.shape[0] == experts
        assert torch.equal(gradient.flatten(1).count_nonzero(-1).cpu() > 0, touched)
    for expert, token in owner.items():
        _exact_outer(gradients[0][expert], x[token], copies, False)  # gate: d_gate rows x x rows
        _exact_outer(gradients[1][expert], x[token], copies, False)  # up
        _exact_outer(gradients[2][expert], dy[token], copies, True)  # down: dy rows x h rows


@pytest.mark.parametrize("ep,experts", BOUND_LAYOUTS, ids=BOUND_IDS)
def test_before_down_adapter_at_the_capacity_bound(ep, experts):
    """The adapter's router-weight-before-down training step at the requester's worst-case geometry of one rank:
    16,384 source rows, H 5120, I 2304, top-6, macrobatch 262,144, minibatch 4,096, the worst-case multiplier and full
    skew (every route of the group lands on this rank), emulated on one GPU (:class:`OneGpuGroup`).  The context
    step, the checkpointed step (context-free forward, recompute with the router weights, backward; another router
    weight in the workspace before the recompute and before the backward) and its CUDA Graph replay complete with
    finite outputs and the same bytes, run to run.  The routed x ring's rows at byte offsets of 2^31 and beyond hold
    their tokens' x exactly, and ``y``, ``dx`` and the router-weight gradients are the bytes of the same step with
    32,768-row macrobatches, whose ring rows all lie below 2^31 bytes (per-row outputs).  The routed weight gradients,
    which read the x, dy, gate-gradient and up-gradient rings, are checked exactly on a step where only tokens that
    share no expert keep their router weights (:func:`_check_routed_weight_gradients`).  Accuracy is judged at the
    smaller shapes (``test_mok_accuracy.py``, the adapter's top-6 layouts included)."""
    _require_role("forward_clamped_wbd")
    _require_role(f"count_{ep}_{experts}")
    tokens, hidden, intermediate, topk, macro, mini, limit = 16384, 5120, 2304, 6, 262144, 4096, 10.0
    device = torch.device("cuda", torch.cuda.current_device())
    multiplier = bound_multiplier(tokens, topk, ep, experts)
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, multiplier)
    functional = backend.MoKFunctional(
        ep, experts, topk, clamped_swiglu=True, fp32_wgrad=True, router_weight_position="before_down"
    )
    functional.communication = OneGpuGroup()
    functional.prepare(limit, save_context=False)
    functional.prepare_recompute(limit)
    workspace = one_gpu_workspace(ep, tokens, hidden, topk, capacity, device)
    config = MoKConfig(
        fwd_num_comm_sms=40,
        bwd_num_comm_sms=40,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    generator = torch.Generator(device=device).manual_seed(7211 + ep)
    options = dict(device=device, dtype=torch.bfloat16)
    x = randn(tokens, hidden, device=device, generator=generator)
    x = (x / x.square().mean(-1, keepdim=True).sqrt()).bfloat16()
    dy = randn(tokens, hidden, device=device, generator=generator).bfloat16()
    ids = rand(tokens, experts, device=device, generator=generator).argsort(-1)[:, :topk].contiguous()
    scores = rand(tokens, topk, device=device, generator=generator)
    scores = (scores / scores.sum(-1, keepdim=True) * 1.5).contiguous()
    scale = 4.0 / hidden**0.5
    weights = (
        (randn(intermediate, hidden, **options) * scale),
        (randn(intermediate, hidden, **options) * scale),
        (randn(hidden, intermediate, **options) / intermediate**0.5),
        (randn(experts, intermediate, hidden, **options) * scale),
        (randn(experts, intermediate, hidden, **options) * scale),
        (randn(experts, hidden, intermediate, **options) / intermediate**0.5),
    )
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    schedule = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
    routed_rows = int(schedule.num_tokens.item())
    assert ep * tokens * topk <= routed_rows <= capacity
    assert routed_rows > macro  # several macrobatches: backward replays the later ones
    # The first routed x ring row at a byte offset of 2^31 or more.
    high = -(-(2**31) // (2 * hidden))
    assert high < macro

    def check_high_ring_rows(context):
        """The resident macrobatch's routed x rows from byte offset 2^31 on: each holds its token's x exactly (every
        peer's x is this rank's)."""
        torch.cuda.synchronize()
        rows = slice(high, macro)
        defined = schedule.peer_rank[rows] >= 0
        tokens_of = schedule.peer_token_idx[rows][defined].long() // topk
        assert tokens_of.numel() > 0
        assert torch.equal(context.x_routed[rows][defined], x[tokens_of])

    def step(checkpointed, cfg=config, check=False, router=scores):
        for accumulator in accumulators:
            accumulator.zero_()
        y, context = functional.forward(
            cfg, workspace, schedule, x, router, *weights, swiglu_limit=limit,
            save_context=not checkpointed,
        )  # fmt: skip
        if check:
            check_high_ring_rows(context)
        if checkpointed:
            # Another layer's router weights in the workspace: the recompute and the backward stage their own.
            workspace.router_weight_buffer.fill_(0.75)
            context = functional.recompute_forward_context(
                cfg, workspace, schedule, x, weights[0], weights[1], weights[3], weights[4],
                swiglu_limit=limit, router_weights=router,
            )  # fmt: skip
            workspace.router_weight_buffer.fill_(0.75)
        grads = functional.backward(
            cfg, workspace, schedule, context, dy, x, router, *weights, swiglu_limit=limit,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        return (y, *grads)

    def snapshot(values):
        torch.cuda.synchronize()
        out = tuple(value.clone() for value in values)
        for value in out:
            assert torch.isfinite(value).all().item()
        return out

    first = snapshot(step(False, check=True))
    assert all(torch.count_nonzero(value).item() > 0 for value in first)
    assert _same(first, snapshot(step(False)))
    assert _same(first, snapshot(step(True)))
    small = dataclasses.replace(config, macrobatch_size=32768)
    assert small.macrobatch_size * hidden * 2 < 2**31
    assert _same(first[:3], snapshot(step(False, small))[:3])  # y, dx, router-weight gradients
    _check_routed_weight_gradients(
        lambda cfg, router: snapshot(step(False, cfg, router=router)[3:6]),
        (config, small), schedule, ids, scores, x, dy, high, macro, ep, experts,
    )  # fmt: skip
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step(True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step(True)
    for _ in range(2):
        graph.replay()
        assert _same(first, snapshot(captured))


@pytest.mark.parametrize("ep,experts", BOUND_LAYOUTS, ids=BOUND_IDS)
def test_before_down_adapter_top6_changed_counts(ep, experts):
    """The adapter's top-6 router-weight-before-down step over changing source counts on one source workspace, as rank
    0 of its EP group emulated on one GPU (:class:`OneGpuGroup`, every emulated peer at this rank's count): the counts
    of :data:`CHANGED_COUNTS` in order (1031, 257 and 513 rows: padded staging of rows ``[count, shared_rows)``; 768:
    unpadded), a new schedule per count, several macrobatches.  Before each step every workspace row past the count
    holds other values (as after a larger count or another layer), so the staging alone decides those rows: the
    context step and the checkpointed step (context-free forward, recompute with the router weights, backward, another
    router weight in the workspace between them) give exactly the bytes of the context step on a fresh zeroed
    workspace, the outputs have the count's rows, and no count after the first loads a kernel.  The accuracy of these
    steps is judged in ``test_mok_accuracy.py`` (``test_before_down_adapter_top6_changed_counts_accuracy``)."""
    _require_role("forward_clamped_wbd")
    _require_role(f"count_{ep}_{experts}")
    topk, hidden, intermediate, macro, mini, limit = 6, 512, 256, 1536, 256, 0.125
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep, experts, topk, clamped_swiglu=True, fp32_wgrad=True, router_weight_position="before_down"
    )
    functional.communication = OneGpuGroup()
    functional.prepare(limit, save_context=False)
    functional.prepare_recompute(limit)
    workspace, multiplier = one_gpu_source_workspace(ep, experts, CHANGED_COUNTS, hidden, topk, device)
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    fresh_workspaces = []  # alive to the end: no later workspace takes a freed one's address
    loads = None
    for index, count in enumerate(CHANGED_COUNTS):
        x, dy, shared, routed, scores = _before_down_problem(count, hidden, intermediate, experts, topk, 6151 + index)
        ids = rand(count, experts, device=device).argsort(-1)[:, :topk].contiguous()
        weights = (*shared, *routed)

        def step(ws, checkpointed):
            schedule = functional.build_schedule(ws, config, ids, num_local_experts=experts)
            assert schedule.num_source_tokens == count and int(schedule.num_tokens.item()) > macro
            accumulators = [torch.zeros(w.shape, device=device) for w in weights]
            y, context = functional.forward(
                config, ws, schedule, x, scores, *weights, swiglu_limit=limit, save_context=not checkpointed
            )  # fmt: skip
            if checkpointed:
                # Another layer's router weights in the workspace: the recompute and the backward stage their own.
                ws.storage.router_weight_buffer.fill_(0.75)
                context = functional.recompute_forward_context(
                    config, ws, schedule, x, weights[0], weights[1], weights[3], weights[4], swiglu_limit=limit,
                    router_weights=scores,
                )  # fmt: skip
                ws.storage.router_weight_buffer.fill_(0.75)
            grads = functional.backward(
                config, ws, schedule, context, dy, x, scores, *weights, swiglu_limit=limit,
                weight_grad_accumulators=accumulators,
            )  # fmt: skip
            torch.cuda.synchronize()
            return tuple(value.clone() for value in (y, *grads))

        def poison(ws):
            """Every workspace row past the count: values no staging of this count writes."""
            storage = ws.storage
            for buffer, rows, value in (
                (storage.x_buffer, count, 3.0),
                (storage.d_y_buffer, count, -5.0),
                (storage.router_weight_buffer, count, 0.75),
                (storage.d_router_weight_buffer, count, 9.0),
                (storage.combine_buffer, count * topk, 7.0),
                (storage.d_x_routed_buffer, count * topk, -7.0),
            ):
                buffer[rows:].fill_(value)

        fresh_workspaces.append(one_gpu_source_workspace(ep, experts, CHANGED_COUNTS, hidden, topk, device)[0])
        reference = step(fresh_workspaces[-1], False)
        assert [tuple(v.shape) for v in reference[:3]] == [(count, hidden), (count, hidden), (count, topk)]
        assert all(torch.isfinite(value).all().item() for value in reference)
        assert all(torch.count_nonzero(value).item() > 0 for value in reference)
        for checkpointed in (False, True):
            poison(workspace)
            values = step(workspace, checkpointed)
            differs = [i for i, (a, b) in enumerate(zip(reference, values, strict=True)) if not torch.equal(a, b)]
            assert not differs, (count, "checkpointed" if checkpointed else "context", differs)
        # A changed source count builds a new schedule and launch plan on the workspace; it loads no kernel.
        misses = jit._load.cache_info().misses
        if loads is None:
            loads = misses
        assert misses == loads, (count, "kernel loads", loads, misses)


def test_cat_rows_matches_copy_source_and_rejects_what_cat_rejects():
    """Host only: ``_cat_rows`` writes a destination prefix with the bytes of ``_copy_source`` (the source rows, then
    the padding rows), leaves the rows past the prefix alone, and returns False when ``torch.cat`` rejects a staging,
    so the caller stages by ``_copy_source``: a source that autograd records under grad mode, and a source that
    overlaps its destination."""
    functional = backend.MoKFunctional
    source = torch.arange(20.0).reshape(5, 4)
    for value, padding in ((0.0, torch.zeros(3, 4)), (1.0, torch.ones(3, 4))):
        expected = torch.full((10, 4), 7.0)
        functional._copy_source(expected, source, value, 8)
        destination = torch.full((10, 4), 7.0)
        assert functional._cat_rows((source, padding, destination[:8]), None)
        assert torch.equal(destination, expected)
    destination = torch.full((10, 4), 7.0)
    with torch.enable_grad():
        recorded = source.clone().requires_grad_(True)
        assert not functional._cat_rows((recorded, torch.zeros(3, 4), destination[:8]))
    buffer = torch.full((10, 4), 7.0)
    buffer[:5] = source
    assert not functional._cat_rows((buffer[:5], torch.zeros(3, 4), buffer[:8]))


def test_padding_rows_are_constant_normal_tensors_built_outside_capture(monkeypatch):
    """Host only: a workspace's padding rows are BF16 zeros ``[256, hidden]`` and FP32 ones ``[256, topk]`` with the
    launch plans, built once (normal tensors also under ``torch.inference_mode()``), and not built during CUDA Graph
    capture (the caller then stages by ``_copy_source``)."""
    import types

    workspace = types.SimpleNamespace(launch_plans={}, hidden_size=512, topk=6, device=torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert backend.MoKFunctional._padding_rows(workspace) is None and workspace.launch_plans == {}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with torch.inference_mode():
        rows = backend.MoKFunctional._padding_rows(workspace)
    assert workspace.launch_plans == {("padding_rows",): rows}
    assert rows.zeros.dtype == torch.bfloat16 and tuple(rows.zeros.shape) == (256, 512) and not rows.zeros.any()
    assert rows.ones.dtype == torch.float32 and tuple(rows.ones.shape) == (256, 6) and bool((rows.ones == 1).all())
    assert not rows.zeros.is_inference() and not rows.ones.is_inference()
    assert backend.MoKFunctional._padding_rows(workspace) is rows


@pytest.mark.parametrize(
    "position,ep,experts,topk",
    [("before_down", 8, 48, 6), ("after_down", 4, 4, 2)],
    ids=["before_down-ep8-local48", "after_down-ep4-local4"],
)
def test_padded_staging_matches_the_copy_and_fill_form(position, ep, experts, topk, monkeypatch):
    """The padded staging of the planned forward, recompute and backward (``_cat_rows``: one ``torch.cat`` per source
    buffer from the workspace's padding rows) over the changing source counts of :data:`CHANGED_COUNTS` on one source
    workspace, as rank 0 of its EP group emulated on one GPU (:class:`OneGpuGroup`).  Before each step every
    workspace buffer holds other values.  After the context step and after the checkpointed step the workspace rows
    ``[0, shared_rows)`` of ``x``, the router weights and ``dy`` hold the source rows and the padding values, the
    padding rows of the combine and gradient buffers hold 0, and the rows past ``shared_rows`` keep their values;
    every output (``y``, ``dx``, ``dscores``, the six FP32 weight-gradient accumulators) equals bit for bit the same
    step with every padded staging in the copy-and-fill form (``_copy_source``), and so does a step whose ``x`` is a
    view of the workspace's own ``x`` rows (``cat`` rejects it, the copy-and-fill form stages it)."""
    before = position == "before_down"
    if before:
        _require_role("forward_clamped_wbd")
        _require_role(f"count_{ep}_{experts}")
    else:
        require_gpu()
    hidden, intermediate, macro, mini = 512, 256, 1536, 256
    limit = 0.125 if before else None
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep, experts, topk, clamped_swiglu=before, fp32_wgrad=True, router_weight_position=position
    )
    functional.communication = OneGpuGroup()
    functional.prepare(limit, save_context=False)
    functional.prepare_recompute(limit)
    workspace, multiplier = one_gpu_source_workspace(ep, experts, CHANGED_COUNTS, hidden, topk, device)
    storage = workspace.storage
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    original = backend.MoKFunctional._cat_rows
    staged = []

    def spy(*stagings):
        staged.append(original(*stagings))
        return staged[-1]

    sources = (
        (storage.x_buffer, 3.0, 0.0),
        (storage.d_y_buffer, -5.0, 0.0),
        (storage.router_weight_buffer, 0.75, 1.0),
    )
    targets = (
        (storage.combine_buffer, topk, 7.0),
        (storage.d_x_routed_buffer, topk, -7.0),
        (storage.d_router_weight_buffer, 1, 9.0),
    )
    for index, count in enumerate(CHANGED_COUNTS):
        rows = backend.MoKFunctional._shared_rows(count)
        x, dy, shared, routed, scores = _before_down_problem(count, hidden, intermediate, experts, topk, 6301 + index)
        ids = rand(count, experts, device=device).argsort(-1)[:, :topk].contiguous()
        weights = (*shared, *routed)

        def step(checkpointed, x=x):
            schedule = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
            accumulators = [torch.zeros(w.shape, device=device) for w in weights]
            y, context = functional.forward(
                config, workspace, schedule, x, scores, *weights, swiglu_limit=limit, save_context=not checkpointed
            )  # fmt: skip
            if checkpointed:
                context = functional.recompute_forward_context(
                    config, workspace, schedule, x, weights[0], weights[1], weights[3], weights[4], swiglu_limit=limit,
                    router_weights=scores,
                )  # fmt: skip
            grads = functional.backward(
                config, workspace, schedule, context, dy, x, scores, *weights, swiglu_limit=limit,
                weight_grad_accumulators=accumulators,
            )  # fmt: skip
            torch.cuda.synchronize()
            return tuple(value.clone() for value in (y, *grads))

        def poison():
            for buffer, value, _ in sources:
                buffer.fill_(value)
            for buffer, _, value in targets:
                buffer.fill_(value)

        def check_staged():
            torch.cuda.synchronize()
            pad = rows - count
            for (buffer, rest, value), source in zip(sources, (x, dy, scores), strict=True):
                filler = source.new_full((pad, source.shape[1]), value)
                assert torch.equal(buffer[:rows], torch.cat((source, filler))), (count, value)
                assert bool((buffer[rows:] == rest).all()), (count, rest)
            for buffer, width, rest in targets:
                assert not buffer[count * width : rows * width].any(), (count, width)
                assert bool((buffer[rows * width :] == rest).all()), (count, rest)

        values = {}
        for form in ("cat", "copy"):
            if form == "cat":
                monkeypatch.setattr(backend.MoKFunctional, "_cat_rows", staticmethod(spy))
            else:
                monkeypatch.setattr(backend.MoKFunctional, "_cat_rows", staticmethod(lambda *stagings: False))
            for checkpointed in (False, True):
                staged.clear()
                poison()
                values[form, checkpointed] = step(checkpointed)
                check_staged()
                if form == "cat":
                    padded = rows > count
                    assert staged == ([True] * (3 if checkpointed else 2) if padded else []), (count, staged)
            monkeypatch.undo()
        for checkpointed in (False, True):
            assert _same(values["cat", checkpointed], values["copy", checkpointed]), (count, checkpointed)
        poison()
        storage.x_buffer[:count].copy_(x)
        assert _same(values["cat", True], step(True, x=storage.x_buffer[:count])), count


def test_build_schedule_route_buffer_over_changing_counts():
    """``build_schedule`` through the schedule plan over changing source counts on one source workspace (one-GPU
    emulation of rank 0 of EP 4, :class:`OneGpuGroup`): after each eager build the plan's route buffer holds the
    count's expert IDs and ``-1`` in every other row (repeated, growing and shrinking counts), and the schedule equals
    the unplanned launches' bit for bit.  Replays of a build captured in a CUDA Graph, which write IDs past a later
    eager count, leave every later eager schedule exact, and the captured schedule replays exactly."""
    require_gpu()
    ep, experts, topk, hidden = 4, 4, 2, 256
    counts = (777, 777, 900, 1024, 300, 300, 901, 513)
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(ep, experts, topk)
    functional.communication = OneGpuGroup()
    config = MoKConfig(fwd_num_comm_sms=4, bwd_num_comm_sms=4)
    workspace, _ = one_gpu_source_workspace(ep, experts, counts, hidden, topk, device)
    storage = workspace.storage
    rows = storage.num_local_tokens
    assert rows == max(counts)
    key = ("schedule", functional.scheduler, functional.communication)
    generator = torch.Generator(device=device).manual_seed(5501)

    def problem(count):
        return rand(count, ep * experts, device=device, generator=generator).argsort(-1)[:, :topk].contiguous()

    def unplanned(ids, count):
        rank, token, total, tokens, layout, _ = functional._schedule_launches(storage, config, ids, count)
        return _schedule_fields(
            workspace_module.MoKSchedule(
                peer_rank=rank, peer_token_idx=token, num_tokens=total, tokens_per_expert=tokens, layout=layout
            ),
            experts,
        )

    def eager(count):
        ids = problem(count)
        schedule = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
        torch.cuda.synchronize()
        expected = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
        expected[:count] = ids
        assert torch.equal(storage.launch_plans[key].routes, expected), count
        assert _same(_schedule_fields(schedule, experts), unplanned(ids, count)), count

    for count in counts:
        eager(count)
    plan = storage.launch_plans[key]
    eager(300)
    graph_ids = problem(901)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = functional.build_schedule(workspace, config, graph_ids, num_local_experts=experts)
    assert storage.launch_plans[key] is plan
    for count in (600, 600, 1024, 450):
        graph.replay()
        torch.cuda.synchronize()
        assert _same(_schedule_fields(captured, experts), unplanned(graph_ids, 901))
        graph.replay()
        eager(count)
