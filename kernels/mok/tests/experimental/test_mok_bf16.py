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
    scheduler layouts: builds for the targets of ``BEFORE_DOWN_TARGETS``) on a device whose target has no build of
    it.  Every other role is built for every target (:func:`test_before_down_roles_are_registry_records`), so a
    missing build fails at load instead of skipping."""
    require_gpu()
    arch = jit.target_arch()
    if role in BEFORE_DOWN_ROLES and arch not in jit._registry()[role]["arches"]:
        pytest.skip(f"{role} has no {arch} build")


def _scheduler_topk(ep, experts):
    """The top-k a layout's scheduler tests route with: 6 on the 384-expert layouts, else 2 on the diagnostic layouts
    and 8."""
    if (ep, experts) in kernels.SCHEDULER_LAYOUTS_384:
        return 6
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
    rank, token, total, counts, layout = functional._schedule_launches(
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
    rank, token, total, counts, layout = functional._schedule_launches(
        workspace.storage, config, ids, count
    )
    unplanned = _schedule_fields(
        workspace_module.MoKSchedule(
            peer_rank=rank, peer_token_idx=token, num_tokens=total, tokens_per_expert=counts, layout=layout
        ),
        experts,
    )
    assert _same(first, unplanned) and _same(second, unplanned) and _same(third, unplanned)


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
            152: {
                "bf16": {
                    4: (20, 24),
                    8: (20, 24),
                    16: (20, 24),
                    32: (20, 24),
                    64: (20, 24),
                },
                "mxfp8": {8: (44, 36), 32: (44, 36), 64: (44, 36)},
            }
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

#: The roles only the router-weight-before-down mode, top-k 6 and the 384-expert layouts load.
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
)
#: The two layouts the requester runs and the worst-case geometry it asked to fit.
BOUND_LAYOUTS = [(64, 6), (8, 48)]
BOUND_IDS = ["ep64-local6", "ep8-local48"]


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
        names |= {f"count_{ep}_{experts}", f"rows_{ep}_{experts}", f"pad_{ep}"}
    names |= {f"epilogue_backward_{k}" for k in kernels.EPILOGUE_TOPKS}
    names |= {f"epilogue_forward_{k}" for k in kernels.WEIGHTED_EPILOGUE_TOPKS}
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
