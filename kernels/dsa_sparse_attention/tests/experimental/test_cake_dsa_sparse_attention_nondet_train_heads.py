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

The RoPE form at 96 and 128 query heads (and its 64-head launches).

The shipped kernels compute 64 query heads per work item.  A call with ``H`` = 96 or 128 heads runs, where the program
ships them for the device's architecture, the head-window builds (``*_hw``): one launch per stage over the ``2 T``
(token, window) items, item ``2 t + w`` the 64-head window ``w`` of token ``t`` (heads ``[64 w, 64 w + 64)``, the
rows past ``H`` zero-filled and never stored), reading and writing the ``[T, H]`` tensors directly.  Where the
program does not ship them, it runs the two-window bridge: the 64-head kernels on ``[H - 64, H)`` first, then
``[0, 64)``.  The bridge's forward second window rewrites the heads both cover; its backward runs one delta launch
over the ``T * H`` (token, head) rows, then the main stage per window into one shared dK/dV accumulator, the first
window with ``lse = +inf`` and ``delta = 0`` on the heads the second covers, so those heads add exactly 0 there and
the second window writes their ``dq``.  A 64-head call launches what it launched before.

Host tests (CPU, no device): the launches of the eager entries and the prepared runner, recorded and simulated by
``test_helpers/cake_dsa_sparse_attention_host_launches.py`` -- the 64-head launch list against the golden capture
(with and without the head-window builds registered), the head-window plans (their launches, item order, key-pass
chunks, workspace, operands and fallbacks: the harness registers the builds where the package's registry lacks them)
and the bridge's windows, their order, the window statistics and the accumulator at 64, 96 and 128 heads -- plus the
retained state of the autograd entry, validation and the calls without query rows.  Device tests judge every floating output
against the FP64 oracle at the naive BF16 tier (``tests/cake_naive.py``) on sm_100a / sm_103a (96 and 128 heads ship
there first; sm_107a runs the 64-head cases); a call with no valid key in any row has no accuracy to judge and
equals the oracle exactly instead (:func:`test_naive_tier_heads`).  ``CAKE_PPLX_DSA_LARGE_TESTS=1`` opts in to the
T = 16380 case (:func:`test_large_t_heads`), which a default run skips: the GPU validation of this form sets it and
reports that case on its own.
"""

import dataclasses
import itertools
import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import DRAWS, contract, draw_seed
from tests.experimental.test_helpers import (
    cake_dsa_sparse_attention_host_launches as hl,
)
from tests.experimental.test_helpers import cake_dsa_sparse_attention_naive as nv
from tests.experimental.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (
    derive_topk_length,
    make_inputs,
)

cake_backend = cake_package(
    "cake.cake_dsa_sparse_attention_nondet_train", "cake_backend"
)
cake_jit = cake_package("cake.cake_dsa_sparse_attention_nondet_train", "cake_jit")
cake_launch = cake_package("cake.cake_dsa_sparse_attention_nondet_train", "cake_launch")
det_backend = cake_package("cake.cake_dsa_sparse_attention_det_train", "cake_backend")
D_LATENT = cake_backend.D_LATENT
D_QK = cake_backend.D_QK
D_ROPE = cake_backend.D_ROPE
KEY_PASS_STAGES = cake_backend.KEY_PASS_STAGES
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
generated_program_available = cake_backend.generated_program_available
offset_gather_kv_indices = cake_backend.offset_gather_kv_indices
prepare_dsa_train = cake_backend.prepare_dsa_train
validate_dsa_train_inputs = cake_backend.validate_dsa_train_inputs

SEED = 20261124
#: The head counts every judged case runs at; 128 heads run the reduced cases (:data:`REDUCED_HEADS`).
HEADS = (96, 64)
REDUCED_HEADS = (128,)
#: The softmax scale of the issue's model (192-wide query-key head dimension).
SCALE = 192**-0.5
#: Run-time softmax scales of :func:`test_heads_and_counts_load_no_new_module` (192-, 576- and 128-wide heads).
SCALES = (SCALE, 576**-0.5, 128**-0.5)
TOPK = 2048
LARGE_TESTS_ENV = "CAKE_PPLX_DSA_LARGE_TESTS"
MIB = 2**20
#: The most a checkpointed forward may keep alive beyond its output.
CKPT_RETAINED_LIMIT_MIB = 2.0


# ---------------------------------------------------------------------------
# Host layer (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, arch", hl.h64_case_names(), ids=[f"{n}@{a}" for n, a in hl.h64_case_names()]
)
def test_h64_plan_is_unchanged(monkeypatch, name, arch):
    """A 64-head call launches what the entries launched before the head windows: every stage launch (stage, grid
    and every argument its kernel receives: the scalar values, each tensor operand's role, shape, strides, dtype and
    offset) and every allocation of the backend equal the golden capture
    (``cake_dsa_sparse_attention_h64_launches.json``) of the same case."""
    golden = json.loads(hl.GOLDEN.read_text())[f"{name}@{arch}"]
    found = json.loads(
        json.dumps(
            hl.capture_case(
                monkeypatch, cake_backend, cake_jit, cake_launch, name, arch
            )
        )
    )
    assert found["allocs"] == golden["allocs"], (found["allocs"], golden["allocs"])
    stages = [row[0] for row in found["launches"]]
    assert stages == [row[0] for row in golden["launches"]], stages
    for i, (got, want) in enumerate(
        zip(found["launches"], golden["launches"], strict=True)
    ):
        assert got == want, f"launch {i} ({got[0]}): {got} != {want}"


_HW_H64_CASES = [(n, a) for n, a in hl.h64_case_names() if a != "sm_107a"]


@pytest.mark.parametrize(
    "name, arch", _HW_H64_CASES, ids=[f"{n}@{a}" for n, a in _HW_H64_CASES]
)
def test_h64_plan_is_unchanged_with_head_window_builds(monkeypatch, name, arch):
    """With the head-window builds registered (``hl.register_head_window_builds``, as the export registers them for
    sm_100a / sm_103a), a 64-head call still launches exactly the golden capture of the same case: the builds serve
    only calls over more than 64 heads."""
    golden = json.loads(hl.GOLDEN.read_text())[f"{name}@{arch}"]
    found = json.loads(
        json.dumps(
            hl.capture_case(
                monkeypatch,
                cake_backend,
                cake_jit,
                cake_launch,
                name,
                arch,
                head_windows=True,
            )
        )
    )
    assert cake_backend.head_window_stage("fwd", arch) == "fwd_hw"
    assert found == golden


@pytest.mark.parametrize(
    "name, arch", _HW_H64_CASES, ids=[f"{n}@{a}" for n, a in _HW_H64_CASES]
)
def test_h64_plan_is_unchanged_with_single_pass_forward(monkeypatch, name, arch):
    """With the single-pass forward builds registered as well (``hl.register_single_pass_builds``, as the export
    registers them for sm_100a / sm_103a), a 64-head call still launches exactly the golden capture of the same
    case: the single-pass forward serves only calls over more than 64 heads."""
    golden = json.loads(hl.GOLDEN.read_text())[f"{name}@{arch}"]
    found = json.loads(
        json.dumps(
            hl.capture_case(
                monkeypatch,
                cake_backend,
                cake_jit,
                cake_launch,
                name,
                arch,
                head_windows=True,
                single_pass=True,
            )
        )
    )
    assert cake_backend.single_pass_stage("fwd", arch) == "fwd_h128"
    assert cake_backend.single_pass_stage("fwd_no_olo", arch) == "fwd_h128_no_olo"
    assert cake_backend.head_window_plan(64, arch) == (None, None)
    assert found == golden


#: The host cases of the head-window tests (keys of :func:`hl.run_case`).
HOST_CASES = {
    "fwd": dict(entry="forward", lengths=False),
    "fwd_packed_no_olo": dict(
        entry="forward", packed=True, with_o_lo=False, derive=True
    ),
    "bwd": dict(entry="backward"),
    "bwd_head_major_fp32_no_olo": dict(
        entry="backward", head_major=True, with_o_lo=False, dkv_fp32=True, lengths=False
    ),
    "bwd_passes3_chunks": dict(
        entry="backward", T=40, S=96, key_passes=3, token_chunk=16
    ),
    "bwd_acc_direct_map": dict(
        entry="backward", acc=True, dst_map=True, dkv_direct=True
    ),
    "bwd_acc_direct_passes2": dict(
        entry="backward", acc=True, dkv_direct=True, key_passes=2
    ),
    "bwd_acc_cast_passes2_map": dict(
        entry="backward", acc=True, dst_map=True, dkv_direct=False, key_passes=2
    ),
    "runner": dict(entry="runner"),
    "runner_acc_no_olo": dict(entry="runner", acc=True, dst_map=True, with_o_lo=False),
    "runner_passes2_chunks": dict(
        entry="runner", T=40, S=96, key_passes=2, token_chunk=16
    ),
    # the head-major dout a trainer hands back: above 64 heads the runner's "dout_copy" step fills its workspace copy
    "runner_head_major": dict(entry="runner", head_major=True),
    "runner_head_major_passes2_chunks": dict(
        entry="runner", head_major=True, T=40, S=96, key_passes=2, token_chunk=16
    ),
    "runner_head_major_passes2_no_olo": dict(
        entry="runner", head_major=True, key_passes=2, with_o_lo=False
    ),
    # a packed-slice dout (head stride 576, dense over its rows): read in place at every head count
    "bwd_packed_dout": dict(entry="backward", packed_dout=True),
    "bwd_packed_dout_passes3_chunks": dict(
        entry="backward", packed_dout=True, T=40, S=96, key_passes=3, token_chunk=16
    ),
    "runner_packed_dout_no_olo": dict(
        entry="runner", packed_dout=True, with_o_lo=False
    ),
}


def _host_case_inputs(case, heads):
    return hl.host_inputs(
        case.get("T", 8),
        case.get("S", 64),
        case.get("K", 16),
        heads,
        seed=heads,
        packed=case.get("packed", False),
        head_major=case.get("head_major", False),
        packed_dout=case.get("packed_dout", False),
    )


def _check_dkv(case, roles, total, *, direct):
    """The key gradients of a host backward: the simulated accumulator's two cells reach the caller's ``dkv_acc``
    (directly or through the cast, at the row the map gives source row 0) or the cast's output, nothing else."""
    want = [total["count"], total["sum"]]
    acc = roles.get("dkv_acc")
    if acc is not None:
        dst_map = roles.get("dkv_dst_map")
        row = int(dst_map[0]) if dst_map is not None else 0
        assert acc[row, :2].tolist() == want, (acc[row, :2].tolist(), want)
        rest = acc.clone()
        rest[row, :2] = 0
        assert not rest.any(), "the accumulator changed outside the simulated cells"
        assert roles["dkv_latent"] is None and roles["dk_rope"] is None
        return
    assert not direct
    dkv = roles["dkv_latent"]
    if case.get("dkv_fp32"):
        assert dkv.dtype == torch.float32 and dkv.view(-1)[:2].tolist() == want
    else:
        assert dkv.dtype == torch.bfloat16
        assert torch.equal(dkv.view(-1)[:2], torch.tensor(want).to(torch.bfloat16))


def _check_dout_operand(launches, inp):
    """The ``dout`` the delta launch of a host backward reads: at 64 heads the caller's own at its own strides;
    above 64 heads the caller's own at the virtual strides ``(64 * hs, hs)`` when its token stride is ``H`` head
    strides (``hs``: its head stride), else -- the head-major view -- the dense ``[T, H, 512]`` copy holding the
    caller's values at ``(64 * 512, 512)`` (the main launches read the same buffer: :func:`hl.check_backward`)."""
    H, dout = inp.heads, inp.dout
    delta = next(rec for rec in launches if hl.family(rec.stage) == "delta")
    got = delta.values["dout"]
    strides = (
        int(delta.values["dout_token_stride"]),
        int(delta.values["dout_head_stride"]),
    )
    own = got.untyped_storage().data_ptr() == dout.untyped_storage().data_ptr()
    if H == 64:
        assert own and strides == (dout.stride(0), dout.stride(1)), strides
        return
    in_place = dout.stride(0) == H * dout.stride(1)
    head = dout.stride(1) if in_place else D_LATENT
    assert own == in_place, (
        "above 64 heads a dout dense over its rows is read in place and any other is copied"
    )
    assert strides == (64 * head, head), strides
    if not own:
        assert got.is_contiguous() and tuple(got.shape) == tuple(dout.shape)
        assert torch.equal(got, dout), "the dout copy does not hold the caller's dout"


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("name", list(HOST_CASES))
def test_head_window_launches(monkeypatch, name, arch, heads):
    """The two-window bridge (the plan of a program without the head-window builds; the harness hides them): the
    launches of one host case at ``heads`` query heads, simulated (``hl``): the forward runs the windows
    ``[H - 64, H)`` then ``[0, 64)`` (one window at 64 heads) and every query head's ``out`` / ``lse`` / ``o_lo`` is
    its own, the heads both windows cover from the later one; the backward runs one delta launch over the ``T * H``
    rows, the main stage per (token chunk, window, key pass) with the window statistics of the contract, one shared
    accumulator to which every query head adds once per pass, and one cast (none when direct); every query head's
    ``dq`` is its own.  The delta reads ``dout`` in place, or above 64 heads a head-major ``dout`` from the dense copy
    the eager backward or the runner's ``"dout_copy"`` step makes (:func:`_check_dout_operand`)."""
    case = HOST_CASES[name]
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        arch,
        token_chunk=case.get("token_chunk"),
        head_windows=False,
    )
    inp = _host_case_inputs(case, heads)
    roles = hl.run_case(h, cake_backend, case, inp)
    with_o_lo = case.get("with_o_lo", True)
    entry = case["entry"]
    passes = case.get("key_passes") or 1
    chunks = (
        -(-inp.total_q // case["token_chunk"])
        if passes > 1 and case.get("token_chunk")
        else 1
    )
    direct = bool(case.get("dkv_direct"))
    if entry == "runner":
        direct = roles[
            "runner"
        ].dkv_direct  # the record's rule decides a runner without dkv_direct
    if entry == "forward":
        hl.check_forward(
            h.launches,
            inp,
            roles["out"],
            roles["lse"],
            roles["o_lo"],
            with_o_lo=with_o_lo,
        )
        return
    launches = h.launches
    if entry == "runner":
        first_bwd = next(
            i for i, rec in enumerate(launches) if hl.family(rec.stage) != "fwd"
        )
        hl.check_forward(
            launches[:first_bwd],
            inp,
            roles["out"],
            roles["lse"],
            roles["o_lo"],
            with_o_lo=with_o_lo,
        )
        launches = launches[first_bwd:]
        lse = roles["lse"]
    else:
        lse = inp.lse
    total = hl.check_backward(
        launches,
        inp,
        roles["out"],
        roles["o_lo"],
        lse,
        roles["dq_latent"],
        roles["dq_rope"],
        passes=passes,
        chunks=chunks,
        direct=direct,
    )
    _check_dout_operand(launches, inp)
    _check_dkv(case, roles, total, direct=direct)


@pytest.mark.parametrize("forward", ("head_window", "single_pass"))
@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("name", list(HOST_CASES))
def test_head_window_build_launches(monkeypatch, name, arch, heads, forward):
    """The head-window plan (the builds registered for sm_100a / sm_103a, ``hl.register_head_window_builds``): the
    launches of one host case at ``heads`` query heads, simulated (``hl``): the forward is one launch of ``fwd_hw``
    (``fwd_hw_no_olo`` without ``o_lo``) over the ``2 T`` items or -- with the single-pass forward builds registered
    as well (``forward="single_pass"``, ``hl.register_single_pass_builds``) -- one launch of ``fwd_h128``
    (``fwd_h128_no_olo``) over the ``T`` tokens on its persistent grid, writing every query head's ``out`` / ``lse`` /
    ``o_lo`` in the call's ``[T, H]`` tensors; the backward runs the ``"dout_copy"`` aside, one delta launch over the
    ``T * H`` rows, then the main stage's head-window build -- one launch over the ``2 T`` items (descending from
    item ``2 T - 1`` on sm_103a), or per token chunk and pass a compaction over the chunk's tokens and the pass
    build over its items (item-space first item ``2 * token_base``) --, reading the call's ``lse`` and the delta's
    rows, every query head adding once per pass into one accumulator, then one cast (none when direct); every
    query head's ``dq`` is its own.  No window launch, window scratch or window statistics."""
    case = HOST_CASES[name]
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        arch,
        token_chunk=case.get("token_chunk"),
        head_windows=True,
        single_pass=forward == "single_pass",
    )
    inp = _host_case_inputs(case, heads)
    roles = hl.run_case(h, cake_backend, case, inp)
    with_o_lo = case.get("with_o_lo", True)
    entry = case["entry"]
    passes = case.get("key_passes") or 1
    chunks = (
        -(-inp.total_q // case["token_chunk"])
        if passes > 1 and case.get("token_chunk")
        else 1
    )
    direct = bool(case.get("dkv_direct"))
    single = forward == "single_pass"
    assert all(
        (rec.sp if single and hl.family(rec.stage) == "fwd" else rec.hw)
        for rec in h.launches
        if hl.family(rec.stage) in ("fwd", "main", "main_pass")
    )
    assert all(rec.window is None for rec in h.launches)
    forward_stage = {
        (False, True): "fwd_hw",
        (False, False): "fwd_hw_no_olo",
        (True, True): "fwd_h128",
        (True, False): "fwd_h128_no_olo",
    }[single, with_o_lo]

    def check_forward(launches):
        if single:
            hl.check_forward_sp(
                launches,
                inp,
                roles["out"],
                roles["lse"],
                roles["o_lo"],
                with_o_lo=with_o_lo,
                k_rope=inp.k_rope,
            )
        else:
            hl.check_forward_hw(
                launches,
                inp,
                roles["out"],
                roles["lse"],
                roles["o_lo"],
                with_o_lo=with_o_lo,
            )

    if entry == "runner":
        runner = roles["runner"]
        direct = runner.dkv_direct
        assert tuple(runner.forward_stages) == (forward_stage,)
        assert not any(
            isinstance(key, cake_backend.WindowLaunch) for key in runner.backward_order
        )
        steps = [
            key
            for key in runner.backward_order
            if isinstance(key, cake_backend.HeadStep)
        ]
        assert steps == (
            [cake_backend.HeadStep("dout_copy")] if "dout_copy" in runner.layout else []
        )
        assert not {"fwd_lse", "fwd_o_lo", "window_lse", "window_delta"} & set(
            runner.layout
        ), runner.layout
    if entry == "forward":
        check_forward(h.launches)
        return
    launches = h.launches
    if entry == "runner":
        first_bwd = next(
            i for i, rec in enumerate(launches) if hl.family(rec.stage) != "fwd"
        )
        check_forward(launches[:first_bwd])
        launches = launches[first_bwd:]
        lse = roles["lse"]
    else:
        lse = inp.lse
    total = hl.check_backward_hw(
        launches,
        inp,
        roles["out"],
        roles["o_lo"],
        lse,
        roles["dq_latent"],
        roles["dq_rope"],
        passes=passes,
        chunks=chunks,
        direct=direct,
        arch=arch,
    )
    _check_dout_operand(launches, inp)
    _check_dkv(case, roles, total, direct=direct)


def _capture_plans(monkeypatch) -> list:
    """The list the plans (``_Plan``) the backend builds from now on are appended to, in call order."""
    made = []
    real_plan = cake_backend._plan

    def plan(*args, **kwargs):
        made.append(real_plan(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(cake_backend, "_plan", plan)
    return made


#: The registrations of :func:`test_head_window_fallbacks`: ``(builds registered, their arches, hidden tables)``.
HEAD_WINDOW_REGISTRATIONS = {
    "forward_only": (("fwd_hw", "fwd_hw_no_olo"), None, ()),
    "backward_only": (
        (
            "bwd_main_hw",
            "bwd_main_natural_hw",
            "bwd_main_pass_hw",
            "bwd_main_pass_natural_hw",
        ),
        None,
        (),
    ),
    "sm_100a_only": (None, ["sm_100a"], ()),
    "no_launcher": (None, None, ("LAUNCH",)),
    "no_grid": (None, None, ("GRID",)),
    "not_in_order": (None, None, ("STAGES_OF",)),
}


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("registration", list(HEAD_WINDOW_REGISTRATIONS))
def test_head_window_fallbacks(monkeypatch, registration, arch):
    """A phase whose head-window build the registry does not register for the architecture runs the two-window
    bridge, the other phase its head-window build where registered: builds of the forward only, of the backward
    only, builds shipping for sm_100a only (sm_103a: the bridge), a build without its generated launcher or grid
    (``cake_launch``), and a record entry outside the program's ``STAGES_OF`` order (not a registered stage).  The
    host case (a prepared runner over 96 heads, two forced key passes over token chunks, a head-major ``dout``) is
    simulated through either plan per phase; the workspace is the mixed plan's, as ``dsa_train_workspace_size``
    reports it."""
    builds, arches, hidden = HEAD_WINDOW_REGISTRATIONS[registration]
    hl.strip_single_pass_builds(monkeypatch, cake_jit, cake_launch)
    hl.strip_head_window_builds(monkeypatch, cake_jit, cake_launch)
    order = cake_jit.STAGES_OF[cake_jit.PROGRAM]
    hl.register_head_window_builds(
        monkeypatch, cake_jit, cake_launch, arches=arches, stages=builds
    )
    for name in hidden:
        if name == "STAGES_OF":
            monkeypatch.setattr(
                cake_jit,
                "STAGES_OF",
                dict(cake_jit.STAGES_OF, **{cake_jit.PROGRAM: order}),
            )
            continue
        table = {
            stage: value
            for stage, value in getattr(cake_launch, name).items()
            if not hl.is_head_window(stage)
        }
        monkeypatch.setattr(cake_launch, name, table)
    served = arches is None or arch in arches
    registered = served and not hidden
    fused_forward = registered and (builds is None or "fwd_hw" in builds)
    fused_backward = registered and (builds is None or "bwd_main_pass_hw" in builds)
    assert cake_backend.head_window_plan(96, arch, backward=True, key_passes=2) == (
        "fwd_hw" if fused_forward else None,
        "bwd_main_pass_hw" if fused_backward else None,
    )
    case = dict(
        entry="runner", head_major=True, T=40, S=96, key_passes=2, token_chunk=16
    )
    h = hl.HostLaunches(
        monkeypatch, cake_backend, cake_jit, cake_launch, arch, token_chunk=16
    )
    inp = _host_case_inputs(case, 96)
    roles = hl.run_case(h, cake_backend, case, inp)
    runner = roles["runner"]
    first_bwd = next(
        i for i, rec in enumerate(h.launches) if hl.family(rec.stage) != "fwd"
    )
    fwd, bwd = h.launches[:first_bwd], h.launches[first_bwd:]
    check_fwd = hl.check_forward_hw if fused_forward else hl.check_forward
    check_fwd(fwd, inp, roles["out"], roles["lse"], roles["o_lo"], with_o_lo=True)
    check_bwd = hl.check_backward_hw if fused_backward else hl.check_backward
    kw = dict(arch=arch) if fused_backward else {}
    total = check_bwd(
        bwd,
        inp,
        roles["out"],
        roles["o_lo"],
        roles["lse"],
        roles["dq_latent"],
        roles["dq_rope"],
        passes=2,
        chunks=3,
        direct=runner.dkv_direct,
        **kw,
    )
    _check_dout_operand(bwd, inp)
    _check_dkv(case, roles, total, direct=runner.dkv_direct)
    assert ("fwd_lse" in runner.layout) == (not fused_forward)
    assert ("window_lse" in runner.layout) == (not fused_backward)
    want = cake_backend.workspace_layout(
        40,
        96,
        16,
        backward=True,
        key_passes=2,
        token_chunk=16,
        num_heads=96,
        dout_copy=True,
        fused_forward=fused_forward,
        fused_backward=fused_backward,
    )
    assert runner.layout == want
    assert (
        runner.workspace.numel()
        == want["total"]
        == cake_backend.dsa_train_workspace_size(
            40, 96, 16, key_passes=2, num_heads=96, dout_copy=True
        )
    )


@pytest.mark.parametrize("heads", (64, 96, 128))
def test_head_window_plan_selection(monkeypatch, heads):
    """:func:`head_window_plan` by registration, architecture and head count: without the builds registered every
    phase runs the bridge; registered (sm_100a / sm_103a), a call over more than 64 heads runs ``fwd_hw`` /
    ``fwd_hw_no_olo`` and the main stage's build by key passes and the direct path (``bwd_main_hw``,
    ``bwd_main_natural_hw``, ``bwd_main_pass_hw``, ``bwd_main_pass_natural_hw``); a 64-head call never, sm_107a
    never (even with builds registered for it), and the stages without a build (``bwd_delta``, ``bwd_compact``,
    ``bwd_cast``) have none."""
    arches = ("sm_100a", "sm_103a", "sm_107a")
    plans = [
        dict(backward=False),
        dict(backward=True),
        dict(backward=True, dkv_direct=True),
        dict(backward=True, key_passes=2),
        dict(backward=True, key_passes=3, dkv_direct=True),
    ]
    # the single-pass forward, which replaces fwd_hw where it ships: test_single_pass_plan_selection
    hl.strip_single_pass_builds(monkeypatch, cake_jit, cake_launch)
    if not set(cake_backend.HEAD_WINDOW_STAGES.values()) & set(
        cake_jit.registered_stages()
    ):
        for arch in arches:
            for kw in plans:
                assert cake_backend.head_window_plan(heads, arch, **kw) == (None, None)
    hl.register_head_window_builds(
        monkeypatch,
        cake_jit,
        cake_launch,
        arches=["sm_100a", "sm_103a", "sm_107a"],
    )
    assert cake_backend.HEAD_WINDOW_STAGES == hl.HEAD_WINDOW_STAGES
    for stage in ("bwd_delta", "bwd_compact", "bwd_cast", "fwd_scan"):
        assert cake_backend.head_window_stage(stage, "sm_100a") is None
    for arch in arches:
        served = heads != 64 and arch != "sm_107a"
        for with_o_lo in (True, False):
            for kw in plans:
                forward, main = cake_backend.head_window_plan(
                    heads, arch, with_o_lo=with_o_lo, **kw
                )
                if not served:
                    assert (forward, main) == (None, None), (arch, kw)
                    continue
                assert forward == ("fwd_hw" if with_o_lo else "fwd_hw_no_olo")
                if not kw["backward"]:
                    assert main is None
                    continue
                want = (
                    "bwd_main"
                    + ("_pass" if kw.get("key_passes", 1) > 1 else "")
                    + ("_natural" if kw.get("dkv_direct") else "")
                )
                assert main == f"{want}_hw", (arch, kw, main)


@pytest.mark.parametrize("heads", (64, 96, 128))
def test_single_pass_plan_selection(monkeypatch, heads):
    """The forward :func:`head_window_plan` picks with the single-pass forward builds registered
    (``hl.register_single_pass_builds``; declared for sm_107a too in this test): above 64 heads on sm_100a / sm_103a
    the single-pass build (``fwd_h128``, ``fwd_h128_no_olo`` without ``o_lo``) in place of ``fwd_hw`` /
    ``fwd_hw_no_olo`` -- with or without the head-window builds registered --, and the backward's main stage as
    before; a 64-head call never, sm_107a never; :func:`single_pass_stage` maps only the two forward stages."""
    plans = [
        dict(backward=False),
        dict(backward=True),
        dict(backward=True, key_passes=2, dkv_direct=True),
    ]
    hl.strip_single_pass_builds(monkeypatch, cake_jit, cake_launch)
    hl.strip_head_window_builds(monkeypatch, cake_jit, cake_launch)
    hl.register_single_pass_builds(
        monkeypatch, cake_jit, cake_launch, arches=["sm_100a", "sm_103a", "sm_107a"]
    )
    assert cake_backend.SINGLE_PASS_STAGES == hl.SINGLE_PASS_STAGES
    for stage in ("bwd_main", "bwd_delta", "fwd_scan", "fwd_hw", "fwd_h128"):
        assert cake_backend.single_pass_stage(stage, "sm_100a") is None, stage
    for registered in (False, True):
        if registered:
            hl.register_head_window_builds(monkeypatch, cake_jit, cake_launch)
        for arch in ("sm_100a", "sm_103a", "sm_107a"):
            served = heads != 64 and arch != "sm_107a"
            for with_o_lo in (True, False):
                single = "fwd_h128" if with_o_lo else "fwd_h128_no_olo"
                stage = "fwd" if with_o_lo else "fwd_no_olo"
                assert cake_backend.single_pass_stage(stage, arch) == (
                    single if arch != "sm_107a" else None
                )
                for kw in plans:
                    forward, main = cake_backend.head_window_plan(
                        heads, arch, with_o_lo=with_o_lo, **kw
                    )
                    if not served:
                        assert (forward, main) == (None, None), (arch, kw)
                        continue
                    assert forward == single, (arch, kw, forward)
                    if not kw["backward"] or not registered:
                        assert main is None, (arch, kw, main)
                    else:
                        want = (
                            "bwd_main_pass_natural_hw"
                            if kw.get("key_passes")
                            else "bwd_main_hw"
                        )
                        assert main == want, (arch, kw, main)


#: The registrations of :func:`test_single_pass_forward_fallbacks`: ``(arches, hidden tables)``.
SINGLE_PASS_REGISTRATIONS = {
    "sm_100a_only": (["sm_100a"], ()),
    "no_launcher": (None, ("LAUNCH",)),
    "no_grid": (None, ("GRID",)),
    "not_in_order": (None, ("STAGES_OF",)),
}


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("registration", list(SINGLE_PASS_REGISTRATIONS))
def test_single_pass_forward_fallbacks(monkeypatch, registration, arch):
    """A single-pass forward build the registry does not register for the architecture -- shipping for sm_100a only
    (sm_103a), without its generated launcher or grid, or outside the program's ``STAGES_OF`` order -- leaves the
    forward to the head-window build (``fwd_hw``), simulated through a host forward over 96 heads; registered, the
    forward is the single-pass build."""
    arches, hidden = SINGLE_PASS_REGISTRATIONS[registration]
    hl.strip_single_pass_builds(monkeypatch, cake_jit, cake_launch)
    hl.strip_head_window_builds(monkeypatch, cake_jit, cake_launch)
    hl.register_head_window_builds(monkeypatch, cake_jit, cake_launch)
    order = cake_jit.STAGES_OF[cake_jit.PROGRAM]
    hl.register_single_pass_builds(monkeypatch, cake_jit, cake_launch, arches=arches)
    for name in hidden:
        if name == "STAGES_OF":
            monkeypatch.setattr(
                cake_jit,
                "STAGES_OF",
                dict(cake_jit.STAGES_OF, **{cake_jit.PROGRAM: order}),
            )
            continue
        table = {
            stage: value
            for stage, value in getattr(cake_launch, name).items()
            if not hl.is_single_pass(stage)
        }
        monkeypatch.setattr(cake_launch, name, table)
    served = (arches is None or arch in arches) and not hidden
    forward = "fwd_h128" if served else "fwd_hw"
    assert cake_backend.head_window_plan(96, arch) == (forward, None)
    h = hl.HostLaunches(
        monkeypatch, cake_backend, cake_jit, cake_launch, arch, single_pass=None
    )
    case = dict(entry="forward")
    inp = _host_case_inputs(case, 96)
    roles = hl.run_case(h, cake_backend, case, inp)
    check = hl.check_forward_sp if served else hl.check_forward_hw
    check(h.launches, inp, roles["out"], roles["lse"], roles["o_lo"], with_o_lo=True)


@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
def test_single_pass_forward_plan(monkeypatch, arch, heads):
    """A plan whose forward is the single-pass build: its launch values ``num_queries = T`` and ``num_heads = H`` (no
    window selector, no item count) next to the head-window main stage's ``2 T`` items; its grid the persistent
    :func:`cake_backend.single_pass_grid` over the device's SM count (:data:`hl.SM_COUNT` here: 74 clusters at most);
    the same workspace as the plan with the head-window forward (no window scratch either); the host operand contract
    (:func:`cake_backend._check_head_window_operands`) refuses other launch values for it; and
    :func:`cake_backend._stage_entry` refuses a grid that is not a multiple of its ``(2, 1, 1)`` cluster."""
    made = _capture_plans(monkeypatch)
    layouts = {}
    for single in (False, True):
        with monkeypatch.context() as mp:
            h = hl.HostLaunches(
                mp,
                cake_backend,
                cake_jit,
                cake_launch,
                arch,
                simulate=False,
                head_windows=True,
                single_pass=single,
            )
            for T in (1, 40, 300):
                inp = hl.host_inputs(T, 64, 16, heads)
                h.bind(inp.q_latent, inp.q_rope, inp.dout)
                made.clear()
                hl.run_case(h, cake_backend, dict(entry="backward"), inp)
                hl.run_case(h, cake_backend, dict(entry="runner"), inp)
                plan = made[-1]
                layouts[single, T] = plan.layout
                assert plan.fused_forward and plan.fused_backward
                fwd = plan.forward_stages[0]
                if not single:
                    assert fwd == "fwd_hw"
                    continue
                assert tuple(plan.forward_stages) == ("fwd_h128",)
                assert plan.stage_values["fwd_h128"] == dict(
                    num_queries=T, num_heads=heads
                )
                assert plan.stage_values["bwd_main_hw"]["num_queries"] == 2 * T
                assert (
                    plan.entries["fwd_h128"].grid
                    == cake_backend.single_pass_grid(T, hl.SM_COUNT)
                    == (2 * min(T, hl.SM_COUNT // 2), 1, 1)
                )
                (rec,) = [r for r in h.launches if r.stage == "fwd_h128"][-1:]
                assert rec.grid == plan.entries["fwd_h128"].grid
                check = cake_backend._check_head_window_operands
                good = _hw_plan_tensors(plan, inp)
                check(plan, good)
                for own in (
                    dict(num_queries=2 * T, num_heads=heads),
                    dict(num_queries=T, num_heads=64),
                    dict(num_queries=T, num_heads=heads, wsel=-1),
                ):
                    broken = dataclasses.replace(
                        plan, stage_values=dict(plan.stage_values, fwd_h128=own)
                    )
                    with pytest.raises(ValueError, match="fwd_h128"):
                        check(broken, good)
    for T in (1, 40, 300):
        assert layouts[True, T] == layouts[False, T], (
            "the single-pass forward changes the workspace"
        )
        assert not {"fwd_lse", "fwd_o_lo"} & set(layouts[True, T])
    # the cluster-multiple check of the stage entry (before any module load)
    with monkeypatch.context() as mp:
        hl.register_single_pass_builds(mp, cake_jit, cake_launch)
        for grid in ((1, 1, 1), (3, 1, 1), (149, 1, 1)):
            with pytest.raises(ValueError, match="not a multiple of the cluster shape"):
                cake_backend._stage_entry(
                    "fwd_h128",
                    arch,
                    dict(num_queries=8, num_kv=64, topk=16),
                    grid=grid,
                )


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
def test_head_window_item_order(monkeypatch, arch):
    """The item order of the head-window main launches (``token_base`` / ``token_step`` in item space): both items
    of a token stay adjacent -- ascending from an even item or descending from an odd one
    (:func:`head_window_token_order` refuses anything else) --; a single-pass plan walks the ``2 T`` items as the
    64-head plan walks its tokens (descending from ``2 T - 1`` on sm_103a), a key-pass plan ascending from item
    ``2 * token_base`` of every token chunk."""
    for base, step in ((0, 1), (2, 1), (40, 1), (1, -1), (79, -1)):
        cake_backend.head_window_token_order(base, step)
    for base, step in ((1, 1), (3, 1), (0, -1), (78, -1), (0, 2), (0, 0)):
        with pytest.raises(ValueError, match="head-window main launch"):
            cake_backend.head_window_token_order(base, step)
    made = _capture_plans(monkeypatch)
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        arch,
        token_chunk=16,
        simulate=False,
        head_windows=True,
    )
    T = 40
    inp = hl.host_inputs(T, 96, 16, 96)
    h.bind(inp.q_latent, inp.q_rope, inp.dout)
    for passes in (1, 2):
        made.clear()
        hl.run_case(h, cake_backend, dict(entry="backward", key_passes=passes), inp)
        (plan,) = made
        assert plan.fused_forward and plan.fused_backward
        main = "bwd_main_hw" if passes == 1 else "bwd_main_pass_hw"
        own = plan.stage_values[main]
        assert (own["num_queries"], own["num_heads"], own["wsel"]) == (2 * T, 96, -1)
        if passes == 1:
            want = cake_backend.token_order(2 * T, arch, True, "rope")
            assert want == (
                dict(token_base=2 * T - 1, token_step=-1)
                if arch == "sm_103a"
                else dict(token_base=0, token_step=1)
            )
            assert {k: own[k] for k in want} == want
            continue
        bases = [
            int(over[0]["token_base"]) for over in plan.pass_overrides
        ]  # token space
        assert bases == [0, 16, 32]
        for chunk, base in enumerate(bases):
            for index in range(passes):
                values = cake_backend._key_values(plan, {}, main, chunk, index)
                assert values["token_base"] == 2 * base and values["token_step"] == 1
                assert values["num_tokens"] == 2 * min(16, T - base)
                assert values["num_queries"] == 2 * T
                compact = cake_backend._key_values(
                    plan, {}, "bwd_compact", chunk, index
                )
                assert compact["token_base"] == base
                assert compact["num_tokens"] == min(16, T - base)


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
def test_head_window_key_pass_rule(monkeypatch, arch):
    """The key passes of a head-window backward: the record's rule of the architecture with the overrides of
    ``HEAD_WINDOW_KEY_PASS_RULES`` -- on sm_103a no tail rows, so the issue's context-parallel rows (T = 16,380,
    S = 524,160 / 1,048,320, top-k 2048) plan one pass where the 64-head rule plans three --, every other row and
    architecture as at 64 heads, a forced count and a multi-segment row unchanged.  The override holds above 64 heads
    where the single-pass head-window build of the binding's accumulation ships
    (:func:`head_window_key_passes_rule`; never at 64 heads or on the two-window bridge), and both planners pass it:
    ``_plan`` and ``dsa_train_workspace_size`` (its size is the one-pass head-window layout on sm_103a, the three-pass
    64-head layout at 64 heads)."""
    assert cake_backend.HEAD_WINDOW_KEY_PASS_RULES == {"sm_103a": {"tail_rows": 0}}
    record = cake_jit.record()
    stages = tuple(cake_jit.registered_stages())
    assert set(KEY_PASS_STAGES) <= set(stages)
    rows = (
        (16380, 16380, 1),
        (16380, 524160, 1),
        (16380, 1048320, 1),
        (16380, 16380, 4),
        (16380, 524160, 4),
        (4096, 65536, 1),
        (4096, 131072, 1),
        (4096, 1048320, 1),
        (8448, 131072, 1),
        (32768, 131072, 1),
        (65536, 1048320, 1),
    )
    for T, S, seg in rows:
        kw = {"num_segments": seg, "arch": arch}
        base = cake_backend.plan_key_passes(record, stages, T, S, TOPK, **kw)
        hw = cake_backend.plan_key_passes(
            record, stages, T, S, TOPK, head_windows=True, **kw
        )
        tail_only = arch == "sm_103a" and seg == 1 and T > 4224 and 2 * T <= S
        assert hw == (1 if tail_only else base), (arch, T, S, seg, base, hw)
        for forced in (1, 2, 3):
            assert (
                cake_backend.plan_key_passes(
                    record, stages, T, S, TOPK, forced, head_windows=True, **kw
                )
                == forced
            )
    if arch == "sm_103a":
        for S in (524160, 1048320):
            assert (
                cake_backend.plan_key_passes(record, stages, 16380, S, TOPK, arch=arch)
                == 3
            )
            assert (
                cake_backend.plan_key_passes(
                    record, stages, 16380, S, TOPK, arch=arch, head_windows=True
                )
                == 1
            )
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        arch,
        simulate=False,
        head_windows=arch != "sm_107a",
    )
    for heads in (64, 96, 128):
        for direct in (False, True):
            want = heads != 64 and arch != "sm_107a"
            assert cake_backend.head_window_key_passes_rule(heads, arch, direct) == want
    with monkeypatch.context() as m:  # the two-window bridge
        m.setattr(cake_backend, "head_window_stage", lambda stage, arch: None)
        for direct in (False, True):
            assert not cake_backend.head_window_key_passes_rule(96, arch, direct)
    if arch == "sm_107a":
        return
    seen = []
    real = cake_backend.plan_key_passes

    def recorded(*args, **kwargs):
        seen.append(bool(kwargs.get("head_windows", False)))
        return real(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "plan_key_passes", recorded)
    for heads in (64, 96):
        inp = hl.host_inputs(40, 96, 16, heads)
        h.bind(inp.q_latent, inp.q_rope, inp.dout)
        seen.clear()
        hl.run_case(h, cake_backend, {"entry": "backward"}, inp)
        assert seen and set(seen) == {heads != 64}, (heads, seen)
    T, S = 16380, 524160
    for heads in (64, 96, 128):
        seen.clear()
        size = cake_backend.dsa_train_workspace_size(
            T, S, TOPK, num_heads=heads, dkv_acc=True, dout_copy=False
        )
        assert seen == [heads != 64]
        passes = 3 if (arch == "sm_103a" and heads == 64) else 1
        items = 1 if heads == 64 else cake_backend.HEAD_WINDOW_ITEMS
        policy = cake_backend.KeyPassPolicy.from_record(record)
        layout = cake_backend.workspace_layout(
            T,
            S,
            TOPK,
            backward=True,
            key_passes=passes,
            dkv_direct=True,
            token_chunk=policy.token_chunk(TOPK, items),
            num_heads=heads,
            dout_copy=False,
            fused_forward=heads != 64,
            fused_backward=heads != 64,
        )
        assert size == layout["total"], (arch, heads, size, layout["total"])


def _hw_plan_tensors(plan, inp):
    """The call tensors ``t`` of a head-window backward plan over ``inp`` (host, the shapes its contract wants)."""
    T, H = inp.total_q, inp.heads
    return dict(
        q_latent=inp.q_latent,
        q_rope=inp.q_rope,
        out=inp.q_latent.clone(),
        o_lo=-inp.q_latent,
        lse=inp.lse.clone(),
        dout=inp.dout,
        dq_latent=torch.zeros(T, H, D_LATENT, dtype=torch.bfloat16),
        dq_rope=torch.zeros(T, H, D_ROPE, dtype=torch.bfloat16),
        delta=torch.zeros(T, H),
    )


def test_head_window_operand_contract(monkeypatch):
    """The host contract a head-window plan checks where its launch values are bound
    (:func:`cake_backend._check_head_window_operands`): every head operand holds exactly ``num_heads`` heads of the
    ``T`` tokens and ``lse`` / ``o_lo`` / ``delta`` are contiguous ``[T, H]`` rows, and every head-window stage
    runs ``2 T`` items of ``H`` heads with ``wsel = -1``; any other operand or launch value is refused with
    ``ValueError`` before a launch."""
    made = _capture_plans(monkeypatch)
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        "sm_100a",
        simulate=False,
        head_windows=True,
    )
    inp = hl.host_inputs(8, 64, 16, 96)
    h.bind(inp.q_latent, inp.q_rope, inp.dout)
    hl.run_case(h, cake_backend, dict(entry="backward"), inp)
    (plan,) = made
    assert plan.fused_forward and plan.fused_backward
    assert set(plan.stage_values) == {"fwd_hw", "bwd_main_hw"}
    check = cake_backend._check_head_window_operands
    good = _hw_plan_tensors(plan, inp)
    check(plan, good)
    other = hl.host_inputs(8, 64, 16, 128)
    bad = {
        "q_latent": other.q_latent,
        "q_rope": other.q_rope,
        "out": other.q_latent,
        "dout": other.dout,
        "dq_latent": torch.zeros(8, 128, D_LATENT, dtype=torch.bfloat16),
        "dq_rope": torch.zeros(8, 64, D_ROPE, dtype=torch.bfloat16),
        "lse": torch.zeros(96, 8).t(),  # [T, H], not contiguous
        "o_lo": torch.zeros(8, 64, D_LATENT, dtype=torch.bfloat16),
        "delta": torch.zeros(8, 128),
    }
    for name, value in bad.items():
        with pytest.raises(ValueError, match=name):
            check(plan, dict(good, **{name: value}))
    for stage in plan.stage_values:
        for field_name, value in (("num_queries", 8), ("num_heads", 64), ("wsel", 0)):
            own = dict(plan.stage_values[stage], **{field_name: value})
            broken = dataclasses.replace(
                plan, stage_values=dict(plan.stage_values, **{stage: own})
            )
            with pytest.raises(ValueError, match=stage):
                check(broken, good)
    # the bound values check it: a backward whose dq would be carved for other heads is refused before a launch
    launched = len(h.launches)
    broken = dataclasses.replace(plan, num_heads=128)
    with pytest.raises(ValueError, match="head-window launch"):
        cake_backend._bound_values(broken, good)
    assert len(h.launches) == launched


@pytest.mark.parametrize("heads", (96, 128))
def test_head_window_workspace(monkeypatch, heads):
    """The workspace of the head-window plans: the forward has no window scratch (``fwd_lse`` / ``fwd_o_lo``), the
    backward no window statistics (``window_lse`` / ``window_delta``); a key-pass backward holds one dQ partial row
    per (token, window) item -- ``2 x`` the chunk's tokens -- and one key-scratch / count row per token, its token
    chunk the record policy's chunk at two items per token (top-k 2048: 2,176 tokens, against 4,224 for 64 heads);
    ``delta`` and the ``dout`` copy as on the bridge.  A 64-head layout refuses the head-window flags.  A prepared
    runner (host) carves exactly that layout, of ``dsa_train_workspace_size`` bytes."""
    T, S, K, chunk = 40, 96, 16, 16
    for backward, passes, direct, with_o_lo, dout_copy in itertools.product(
        (False, True), (1, 3), (False, True), (True, False), (False, True)
    ):
        kw = dict(
            backward=backward,
            key_passes=passes,
            dkv_direct=direct,
            token_chunk=chunk,
            num_heads=heads,
            with_o_lo=with_o_lo,
            dout_copy=dout_copy,
        )
        bridge = cake_backend.workspace_layout(T, S, K, **kw)
        fused = cake_backend.workspace_layout(
            T, S, K, fused_forward=True, fused_backward=backward, **kw
        )
        want = {
            name: entry[1]
            for name, entry in bridge.items()
            if name
            not in ("total", "fwd_lse", "fwd_o_lo", "window_lse", "window_delta")
        }
        if backward and passes > 1:
            want["dq_partial"] = 2 * chunk * cake_backend.DQ_PARTIAL_BYTES_PER_TOKEN
            assert want["key_scratch"] == chunk * K * 4
            assert want["pass_counts"] == chunk * 4
        regions = {name: entry for name, entry in fused.items() if name != "total"}
        assert {name: entry[1] for name, entry in regions.items()} == want
        names = [name for name in bridge if name in regions]
        assert list(regions) == names, "the head-window regions keep the bridge's order"
        spans = sorted(regions.values())
        assert all(off % cake_backend.WORKSPACE_ALIGN == 0 for off, _ in spans)
        assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        only_forward = cake_backend.workspace_layout(T, S, K, fused_forward=True, **kw)
        assert "fwd_lse" not in only_forward
        assert ("window_lse" in only_forward) == backward
    for flags in (dict(fused_forward=True), dict(fused_backward=True)):
        with pytest.raises(ValueError, match="more than 64 query heads"):
            cake_backend.workspace_layout(T, S, K, num_heads=64, **flags)
    policy = cake_backend.KeyPassPolicy.from_record(cake_jit.record())
    assert policy.dq_partial_bytes_per_token == cake_backend.DQ_PARTIAL_BYTES_PER_TOKEN
    assert policy.token_chunk(TOPK) == 4224
    assert policy.token_chunk(TOPK, cake_backend.HEAD_WINDOW_ITEMS) == 2176
    assert policy.token_chunks(5000, TOPK, 2) == ((0, 2176), (2176, 2176), (4352, 648))
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        "sm_100a",
        token_chunk=chunk,
        head_windows=True,
    )
    for head_major, passes in itertools.product((False, True), (1, 2)):
        inp = hl.host_inputs(T, S, K, heads, head_major=head_major)
        h.bind(inp.q_latent, inp.q_rope, inp.dout)
        runner = prepare_dsa_train(
            inp.q_latent,
            inp.q_rope,
            inp.kv_latent,
            inp.k_rope,
            inp.indices,
            dout=inp.dout,
            key_passes=passes,
        )
        want = cake_backend.workspace_layout(
            T,
            S,
            K,
            backward=True,
            key_passes=passes,
            token_chunk=chunk,
            num_heads=heads,
            dout_copy=head_major,
            fused_forward=True,
            fused_backward=True,
        )
        assert runner.layout == want
        assert (
            runner.workspace.numel()
            == want["total"]
            == cake_backend.dsa_train_workspace_size(
                T, S, K, key_passes=passes, num_heads=heads, dout_copy=head_major
            )
        )
    upper = cake_backend.workspace_layout(
        T,
        S,
        K,
        backward=True,
        key_passes=2,
        token_chunk=chunk,
        num_heads=heads,
        dout_copy=True,
        fused_forward=True,
        fused_backward=True,
    )
    assert (
        cake_backend.dsa_train_workspace_size(T, S, K, key_passes=2, num_heads=heads)
        == upper["total"]
    )


@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
def test_head_window_builds_zero_rows_and_sm107a(monkeypatch, arch, heads):
    """With the head-window builds registered (and, for this test, declared for sm_107a too): a call without query
    rows returns its empty outputs without planning or launching on every architecture that serves its head count
    (sm_107a refuses it as a call with rows); on sm_107a a call with rows is refused (``NotImplementedError``) by the
    eager entries, ``prepare_dsa_train`` and the workspace size -- the head-window builds serve sm_100a / sm_103a
    only -- and nothing launches."""
    hl.register_head_window_builds(
        monkeypatch,
        cake_jit,
        cake_launch,
        arches=["sm_100a", "sm_103a", "sm_107a"],
    )
    h = hl.HostLaunches(monkeypatch, cake_backend, cake_jit, cake_launch, arch)
    assert cake_backend.head_window_stage("fwd", "sm_107a") is None
    refused = "sm_107a serves 64 query heads only"
    real_plan = cake_backend._plan

    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither plan nor launch")

    monkeypatch.setattr(cake_backend, "_plan", refuse)
    empty = hl.host_inputs(0, 16, 5, heads)
    args = (empty.q_latent, empty.q_rope, empty.kv_latent, empty.k_rope, empty.indices)
    dout = torch.zeros(0, heads, D_LATENT, dtype=torch.bfloat16)
    if arch == "sm_107a":
        with pytest.raises(NotImplementedError, match=refused):
            cake_backend.forward(*args)
    else:
        out, lse, o_lo = cake_backend.forward(*args)
        assert tuple(out.shape) == (0, heads, D_LATENT) and tuple(lse.shape) == (
            0,
            heads,
        )
        grads = cake_backend.backward(*args, out, o_lo, lse, dout)
        assert [tuple(g.shape) for g in grads] == [
            (0, heads, D_LATENT),
            (0, heads, D_ROPE),
            (16, D_LATENT),
            (16, D_ROPE),
        ]
        assert not grads[2].any() and not grads[3].any()
    assert not h.launches
    if arch != "sm_107a":
        return
    monkeypatch.setattr(cake_backend, "_plan", real_plan)
    inp = hl.host_inputs(8, 64, 16, heads)
    h.bind(inp.q_latent, inp.q_rope, inp.dout)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.forward(*args)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.backward(*args, inp.q_latent, -inp.q_latent, inp.lse, inp.dout)
    with pytest.raises(NotImplementedError, match=refused):
        prepare_dsa_train(*args, dout=inp.dout)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.dsa_train_workspace_size(8, 64, 16, num_heads=heads)
    assert not h.launches


def _saved_by(call):
    """The tensors ``call()`` saves for its backward (``saved_tensors_hooks``) and its result."""
    saved = []

    def pack(t):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = call()
    return saved, result


def _retained_bytes(saved, held) -> int:
    """Bytes of the storages of ``saved`` that no tensor of ``held`` (the call's inputs and its returned ``out``)
    owns, each storage counted once."""
    owned = {t.untyped_storage().data_ptr() for t in held if t.numel()}
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in saved
        if t.numel() and t.untyped_storage().data_ptr() not in owned
    }
    return sum(storages.values())


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("with_o_lo", (True, False))
def test_autograd_entry_retains_o_lo_lse_and_lengths_per_head(
    monkeypatch, heads, with_o_lo
):
    """The autograd entry at ``heads`` query heads (host, simulated): the forward saves the nine tensors of its
    contract, of which the call's own state is ``lse [T, H]`` FP32, ``o_lo [T, H, 512]`` BF16 (with ``with_o_lo``;
    the zero-size placeholder without) and the derived ``topk_length [T]`` int32 -- exactly
    ``T * (H * (4 + 1024 with_o_lo) + 4)`` bytes beside the inputs and ``out``, no window scratch -- and its backward
    gives every query head its own ``dq`` (the backward launches as :func:`test_head_window_launches`, or
    :func:`test_head_window_build_launches` where the registry ships the head-window builds).  The packed entry (its
    torch index route) saves the nine as well; beside its inputs and ``out`` it keeps exactly
    ``dsa_train_retained_size(T, num_heads=H, with_o_lo=..., varlen_topk=K_slots)`` bytes: ``lse``, ``o_lo``, the
    compacted rows ``[T, K_slots]`` and their counts ``[T]``."""
    _autograd_entry_case(monkeypatch, heads, with_o_lo, head_windows=None)


@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("with_o_lo", (True, False))
def test_autograd_entry_with_head_window_builds(monkeypatch, heads, with_o_lo):
    """:func:`test_autograd_entry_retains_o_lo_lse_and_lengths_per_head` with the head-window builds registered: the
    same nine saved tensors and retained bytes (the head-window forward writes ``lse`` / ``o_lo`` in place, with no
    window scratch either), the forward one ``fwd_hw`` launch and the backward the head-window main build."""
    _autograd_entry_case(monkeypatch, heads, with_o_lo, head_windows=True)


def _autograd_entry_case(monkeypatch, heads, with_o_lo, *, head_windows):
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        "sm_100a",
        head_windows=head_windows,
    )
    fused_forward, fused_backward = cake_backend.head_window_plan(
        heads, "sm_100a", with_o_lo=with_o_lo, backward=True
    )
    assert (fused_forward is not None) == (fused_backward is not None)
    fused = fused_forward is not None
    assert fused or not head_windows or heads == 64
    check_forward = hl.check_forward_hw if fused else hl.check_forward
    check_backward = hl.check_backward_hw if fused else hl.check_backward
    inp = hl.host_inputs(8, 64, 16, heads, seed=heads)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    h.bind(leaves[0], leaves[1], inp.dout)
    saved, (out, lse) = _saved_by(
        lambda: cake_backend.cake_dsa_sparse_attention_nondet(
            *leaves, inp.indices, return_lse=True, with_o_lo=with_o_lo
        )
    )
    assert len(saved) == 9
    T, H = inp.total_q, heads
    retained = _retained_bytes(saved, (*leaves, inp.indices, out))
    assert retained == T * (H * (4 + (1024 if with_o_lo else 0)) + 4), retained
    fwd_launches = h.mark()
    check_forward(
        h.launches[:fwd_launches],
        hl.HostInputs(leaves[0].detach(), leaves[1].detach(), *(None,) * 6),
        out.detach(),
        lse,
        saved[7] if with_o_lo else None,
        with_o_lo=with_o_lo,
    )
    grads = torch.autograd.grad(out, leaves, inp.dout)
    bwd = hl.HostInputs(
        leaves[0].detach(),
        leaves[1].detach(),
        inp.kv_latent,
        inp.k_rope,
        inp.indices,
        None,
        inp.dout,
        lse,
    )
    check_backward(
        h.launches[fwd_launches:],
        bwd,
        out.detach(),
        saved[7] if with_o_lo else None,
        lse,
        grads[0],
        grads[1],
    )
    # the packed entry on its torch index route (the index-preparation stage needs a device)
    monkeypatch.setattr(
        cake_backend, "nope_index_prep_available", lambda device=None: False
    )
    K = int(inp.indices.shape[1])
    local = inp.indices % 24
    cu_q = torch.tensor([0, 3, T], dtype=torch.int32)
    cu_k = torch.tensor([0, 24, inp.total_k], dtype=torch.int32)
    leaves = [t.detach().clone().requires_grad_() for t in leaves]
    h.bind(leaves[0], leaves[1], inp.dout)
    saved, out = _saved_by(
        lambda: cake_backend.cake_dsa_sparse_attention_nondet_varlen(
            *leaves, local, cu_q, cu_k, causal=True, with_o_lo=with_o_lo
        )
    )
    assert len(saved) == 9
    want = cake_backend.dsa_train_retained_size(
        T, num_heads=H, with_o_lo=with_o_lo, varlen_topk=K
    )
    assert want == T * (H * (4 + (1024 if with_o_lo else 0)) + 4 * K + 4)
    retained = _retained_bytes(saved, (*leaves, local, cu_q, cu_k, out))
    assert retained == want, (retained, want)
    grads = torch.autograd.grad(out, leaves, inp.dout)
    assert [tuple(g.shape) for g in grads] == [tuple(t.shape) for t in leaves]


#: The ``dout`` layouts of :func:`test_every_rope_plan_carries_dout_copy`: dense, the head-major view a trainer hands
#: back and a packed slice (channels ``0:512`` of ``[T, H, 576]``, dense over its rows).
DOUT_LAYOUTS = ("dense", "head_major", "packed")


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("with_o_lo", (True, False))
def test_every_rope_plan_carries_dout_copy(monkeypatch, heads, with_o_lo):
    """Every plan the backend builds for the rope form carries ``dout_copy`` (host): the eager forward and backward,
    a prepared runner and the flat and packed (varlen) autograd entries at ``heads`` query heads, each with every
    ``dout`` layout of :data:`DOUT_LAYOUTS` (the autograd entries get it from ``autograd.grad``).  The bound values
    read the field with a ``False`` fallback for duck-typed plans, so a real plan must hold it: a ``_Plan`` field,
    ``True`` exactly for a backward over more than 64 heads whose ``dout`` is not dense over its rows (the
    head-major view).  The plan's three uses of it agree: the ``"dout_copy"`` step is in its backward order, its
    workspace layout has the ``dout_copy`` region, and the delta reads at the head stride 512 of the copy (else at
    the ``dout``'s own) with the virtual token stride of 64 heads above 64 heads."""
    _dout_copy_case(monkeypatch, heads, with_o_lo, head_windows=None)


@pytest.mark.parametrize("heads", (96, 128))
@pytest.mark.parametrize("with_o_lo", (True, False))
def test_every_rope_plan_carries_dout_copy_with_head_window_builds(
    monkeypatch, heads, with_o_lo
):
    """:func:`test_every_rope_plan_carries_dout_copy` with the head-window builds registered: every plan is a
    head-window plan (``fused_forward``, and ``fused_backward`` for a backward) and carries ``dout_copy`` by the same
    rule -- the head-window main build reads the same dense copy the delta reads."""
    found = _dout_copy_case(monkeypatch, heads, with_o_lo, head_windows=True)
    for entry, layout, _, made in found:
        assert made.fused_forward and made.fused_backward == made.backward, (
            entry,
            layout,
        )


def _dout_copy_case(monkeypatch, heads, with_o_lo, *, head_windows):
    assert "dout_copy" in {f.name for f in dataclasses.fields(cake_backend._Plan)}
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        "sm_100a",
        head_windows=head_windows,
    )
    real_plan = cake_backend._plan
    found = []
    call = {}

    def plan(*args, **kwargs):
        made = real_plan(*args, **kwargs)
        found.append((call["entry"], call["layout"], call["dout"], made))
        return made

    monkeypatch.setattr(cake_backend, "_plan", plan)
    # the packed entry takes its torch route on the host (the index-preparation stage needs a device)
    monkeypatch.setattr(
        cake_backend, "nope_index_prep_available", lambda device=None: False
    )
    T, S, K = 8, 64, 16

    def start(entry, layout):
        cake_backend.BINDING_CACHE._entries.clear()  # every entry plans afresh
        inp = hl.host_inputs(
            T,
            S,
            K,
            heads,
            seed=heads,
            head_major=layout == "head_major",
            packed_dout=layout == "packed",
        )
        call.update(entry=entry, layout=layout, dout=inp.dout)
        h.bind(inp.q_latent, inp.q_rope, inp.dout)
        return inp

    for layout in DOUT_LAYOUTS:
        for entry in ("forward", "backward", "runner"):
            inp = start(entry, layout)
            hl.run_case(h, cake_backend, dict(entry=entry, with_o_lo=with_o_lo), inp)
        for entry in ("autograd", "varlen"):
            inp = start(entry, layout)
            leaves = [
                t.detach().clone().requires_grad_()
                for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
            ]
            h.bind(leaves[0], leaves[1], inp.dout)
            if entry == "autograd":
                out = cake_backend.cake_dsa_sparse_attention_nondet(
                    *leaves, inp.indices, with_o_lo=with_o_lo
                )
            else:
                cu_q = torch.tensor([0, 3, T], dtype=torch.int32)
                cu_k = torch.tensor([0, 24, S], dtype=torch.int32)
                local = inp.indices % 24
                local[:, -1] = -1
                out = cake_backend.cake_dsa_sparse_attention_nondet_varlen(
                    *leaves, local, cu_q, cu_k, causal=True, with_o_lo=with_o_lo
                )
            torch.autograd.grad(out, leaves, inp.dout)
    assert {(entry, layout) for entry, layout, _, _ in found} == {
        (entry, layout)
        for entry in ("forward", "backward", "runner", "autograd", "varlen")
        for layout in DOUT_LAYOUTS
    }
    for entry in ("autograd", "varlen"):
        for layout in DOUT_LAYOUTS:
            assert any(
                made.backward
                for name, lay, _, made in found
                if (name, lay) == (entry, layout)
            ), (entry, layout)
    for entry, layout, dout, made in found:
        assert made.num_heads == heads and made.with_o_lo == with_o_lo
        want = made.backward and heads != 64 and layout == "head_major"
        assert isinstance(made.dout_copy, bool) and made.dout_copy == want, (
            entry,
            layout,
            made.backward,
            made.dout_copy,
        )
        steps = [
            key for key in made.backward_order if isinstance(key, cake_backend.HeadStep)
        ]
        assert (cake_backend.HeadStep("dout_copy") in steps) == made.dout_copy, (
            entry,
            layout,
        )
        assert ("dout_copy" in made.layout) == made.dout_copy, (entry, layout)
        if not made.backward:
            continue
        strides = (
            made.constants["dout_token_stride"],
            made.constants["dout_head_stride"],
        )
        if heads == 64:
            assert strides == (dout.stride(0), dout.stride(1)), (entry, layout, strides)
        else:
            head = D_LATENT if made.dout_copy else dout.stride(1)
            assert strides == (64 * head, head), (entry, layout, strides)
    assert any(made.dout_copy for _, _, _, made in found) == (heads != 64)
    return found


@pytest.mark.parametrize("heads", (64, 96, 128))
def test_workspace_and_retained_accounting_heads(monkeypatch, heads):
    """The per-call scratch (workspace) and the state a step keeps from its forward to its backward (retained) are
    reported apart.  Workspace: a 64-head layout is 0.6.1's, region for region; above 64 heads the
    forward adds ``fwd_lse`` ``[T, 64]`` FP32 and (``with_o_lo``) ``fwd_o_lo`` ``[T, 64, 512]`` BF16, the backward
    ``delta`` ``[T, H]`` FP32, ``window_lse`` / ``window_delta`` ``[2, T, 64]`` FP32 and (a ``dout`` that is not
    dense over its rows) ``dout_copy`` ``[T, H, 512]`` BF16, the dQ partials staying 64 heads per token; every
    region aligned and apart.  A prepared runner (host) carves exactly that layout, of
    ``dsa_train_workspace_size`` bytes; that size's default ``dout_copy=None`` is the upper bound (the copy
    included above 64 heads).  Retained: ``lse`` ``[T, H]`` FP32, ``o_lo`` ``[T, H, 512]`` BF16 with
    ``with_o_lo``, the derived ``topk_length`` ``[T]`` int32 without caller lengths; the packed entry
    (``varlen_topk = K_slots``) keeps its compacted rows ``indices`` ``[T, K_slots]`` int32 and their counts
    ``topk_length`` ``[T]`` int32 on both of its routes; ``out`` is the caller's."""
    T, S, K, chunk = 40, 96, 16, 16
    windows = heads != 64
    for backward, passes, direct, with_o_lo, dout_copy in itertools.product(
        (False, True), (1, 3), (False, True), (True, False), (False, True)
    ):
        kw = dict(
            backward=backward, key_passes=passes, dkv_direct=direct, token_chunk=chunk
        )
        layout = cake_backend.workspace_layout(
            T, S, K, num_heads=heads, with_o_lo=with_o_lo, dout_copy=dout_copy, **kw
        )
        base = cake_backend.workspace_layout(T, S, K, **kw)
        if not windows:
            assert layout == base
        want = {name: entry[1] for name, entry in base.items() if name != "total"}
        if backward:
            want["delta"] = T * heads * 4
        if windows:
            want["fwd_lse"] = T * 64 * 4
            if with_o_lo:
                want["fwd_o_lo"] = T * 64 * D_LATENT * 2
            if backward:
                want["window_lse"] = want["window_delta"] = 2 * T * 64 * 4
                if dout_copy:
                    want["dout_copy"] = T * heads * D_LATENT * 2
        regions = {name: entry for name, entry in layout.items() if name != "total"}
        assert {name: entry[1] for name, entry in regions.items()} == want
        spans = sorted(regions.values())
        assert all(off % cake_backend.WORKSPACE_ALIGN == 0 for off, _ in spans)
        assert all(a[0] + a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        assert layout["total"] >= spans[-1][0] + spans[-1][1]
    for with_o_lo, derive in itertools.product((True, False), (True, False)):
        retained = cake_backend.retained_layout(
            T, num_heads=heads, with_o_lo=with_o_lo, derive_topk_length=derive
        )
        want = {"lse": T * heads * 4}
        if with_o_lo:
            want["o_lo"] = T * heads * D_LATENT * 2
        if derive:
            want["topk_length"] = T * 4
        want["total"] = sum(want.values())
        assert retained == want
        assert (
            cake_backend.dsa_train_retained_size(
                T, num_heads=heads, with_o_lo=with_o_lo, derive_topk_length=derive
            )
            == want["total"]
        )
    # the packed entry: its compacted rows and their counts, whatever derive_topk_length says
    for with_o_lo, derive in itertools.product((True, False), (True, False)):
        kw = dict(
            num_heads=heads,
            with_o_lo=with_o_lo,
            derive_topk_length=derive,
            varlen_topk=K,
        )
        retained = cake_backend.retained_layout(T, **kw)
        want = {"lse": T * heads * 4}
        if with_o_lo:
            want["o_lo"] = T * heads * D_LATENT * 2
        want["indices"] = T * K * 4
        want["topk_length"] = T * 4
        want["total"] = sum(want.values())
        assert retained == want and list(retained) == list(want)
        assert cake_backend.dsa_train_retained_size(T, **kw) == want["total"]
    # the two-window bridge's layout (the head-window plans' own: :func:`test_head_window_workspace`)
    h = hl.HostLaunches(
        monkeypatch,
        cake_backend,
        cake_jit,
        cake_launch,
        "sm_100a",
        token_chunk=chunk,
        head_windows=False,
    )
    for head_major in (False, True):
        inp = hl.host_inputs(T, S, K, heads, head_major=head_major)
        h.bind(inp.q_latent, inp.q_rope, inp.dout)
        runner = prepare_dsa_train(
            inp.q_latent,
            inp.q_rope,
            inp.kv_latent,
            inp.k_rope,
            inp.indices,
            dout=inp.dout,
            key_passes=2,
        )
        copy = windows and head_major
        want = cake_backend.workspace_layout(
            T,
            S,
            K,
            backward=True,
            key_passes=2,
            token_chunk=chunk,
            num_heads=heads,
            dout_copy=copy,
        )
        assert runner.layout == want
        assert (
            runner.workspace.numel()
            == want["total"]
            == cake_backend.dsa_train_workspace_size(
                T, S, K, key_passes=2, num_heads=heads, dout_copy=copy
            )
        )
    upper = cake_backend.workspace_layout(
        T,
        S,
        K,
        backward=True,
        key_passes=2,
        token_chunk=chunk,
        num_heads=heads,
        dout_copy=windows,
    )
    assert (
        cake_backend.dsa_train_workspace_size(T, S, K, key_passes=2, num_heads=heads)
        == upper["total"]
    )


@pytest.mark.parametrize("heads", (96, 128))
def test_zero_query_rows_at_more_heads_return_empty_outputs_without_launching(
    monkeypatch, heads
):
    """``T == 0`` at ``heads`` query heads: the eager entries return ``[0, H, ...]`` outputs and zero key gradients
    without planning, loading or launching (host tensors; a CUDA call checks the architecture first,
    :func:`test_zero_query_rows_of_a_cuda_call_check_the_architecture`)."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither plan nor launch")

    monkeypatch.setattr(cake_backend, "_plan", refuse)
    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    inp = hl.host_inputs(0, 16, 5, heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
    for with_o_lo in (True, False):
        out, lse, o_lo = cake_backend.forward(*args, with_o_lo=with_o_lo)
        assert tuple(out.shape) == (0, heads, D_LATENT) and out.dtype == torch.bfloat16
        assert tuple(lse.shape) == (0, heads) and lse.dtype == torch.float32
        assert (o_lo is None) == (not with_o_lo)
        if o_lo is not None:
            assert tuple(o_lo.shape) == (0, heads, D_LATENT)
        dout = torch.zeros(0, heads, D_LATENT, dtype=torch.bfloat16)
        grads = cake_backend.backward(*args, out, o_lo, lse, dout, with_o_lo=with_o_lo)
        assert [tuple(g.shape) for g in grads] == [
            (0, heads, D_LATENT),
            (0, heads, D_ROPE),
            (16, D_LATENT),
            (16, D_ROPE),
        ]
        assert not grads[2].any() and not grads[3].any()
    leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
    out_pub, lse_pub = cake_backend.cake_dsa_sparse_attention_nondet(
        *leaves, inp.indices, return_lse=True
    )
    assert tuple(out_pub.shape) == (0, heads, D_LATENT) and tuple(lse_pub.shape) == (
        0,
        heads,
    )
    g = torch.autograd.grad(
        out_pub, leaves, torch.zeros(0, heads, D_LATENT, dtype=torch.bfloat16)
    )
    assert [tuple(t.shape) for t in g] == [tuple(leaf.shape) for leaf in leaves]


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a", "sm_107a"))
def test_zero_query_rows_of_a_cuda_call_check_the_architecture(
    monkeypatch, arch, heads
):
    """``T == 0`` on a CUDA call (host, the architecture and the device stood in): on ``sm_107a`` a call over more
    than 64 heads is refused with the ``NotImplementedError`` of a call with query rows, by the eager forward and
    backward alike; 64 heads there and every head count on ``sm_100a`` / ``sm_103a`` return the empty outputs.
    Either way nothing is planned, loaded or launched."""
    h = hl.HostLaunches(monkeypatch, cake_backend, cake_jit, cake_launch, arch)

    def refuse(*args, **kwargs):
        raise AssertionError("a call without query rows must neither plan nor launch")

    monkeypatch.setattr(cake_backend, "_plan", refuse)
    monkeypatch.setattr(cake_jit, "load_cake_dsa_train_module", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    inp = hl.host_inputs(0, 16, 5, heads)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
    dout = torch.zeros(0, heads, D_LATENT, dtype=torch.bfloat16)
    if heads != 64 and arch == "sm_107a":
        refused = "sm_107a serves 64 query heads only"
        with pytest.raises(NotImplementedError, match=refused):
            cake_backend.forward(*args)
        lse = torch.zeros(0, heads, dtype=torch.float32)
        with pytest.raises(NotImplementedError, match=refused):
            cake_backend.backward(*args, dout, dout, lse, dout)
    else:
        out, lse, o_lo = cake_backend.forward(*args)
        assert tuple(out.shape) == tuple(o_lo.shape) == (0, heads, D_LATENT)
        assert tuple(lse.shape) == (0, heads)
        grads = cake_backend.backward(*args, out, o_lo, lse, dout)
        assert [tuple(g.shape) for g in grads] == [
            (0, heads, D_LATENT),
            (0, heads, D_ROPE),
            (16, D_LATENT),
            (16, D_ROPE),
        ]
        assert not grads[2].any() and not grads[3].any()
    assert not h.launches


def _validate(q_heads, *, rope_heads=None, dout_heads=None, packed=False):
    inp = hl.host_inputs(8, 16, 5, q_heads, packed=packed)
    q_rope = inp.q_rope
    if rope_heads is not None:
        q_rope = hl.host_inputs(8, 16, 5, rope_heads).q_rope
    dout = None
    if dout_heads is not None:
        dout = torch.zeros(8, dout_heads, D_LATENT, dtype=torch.bfloat16)
    return validate_dsa_train_inputs(
        inp.q_latent, q_rope, inp.kv_latent, inp.k_rope, inp.indices, dout=dout
    )


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("packed", (False, True))
def test_validate_accepts_the_supported_head_counts(heads, packed):
    """64, 96 and 128 query heads validate (split tensors and views of a packed ``[T, H, 576]`` query), with a
    ``dout`` of the same head count."""
    found = _validate(heads, dout_heads=heads, packed=packed)
    assert tuple(found[:3]) == (8, 16, 5)


@pytest.mark.parametrize("heads", (1, 32, 63, 65, 80, 160, 192))
def test_validate_rejects_other_head_counts(heads):
    """Any other query head count is refused with ``ValueError`` naming the supported counts."""
    with pytest.raises(ValueError, match=r"\(64, 96, 128\)"):
        _validate(heads)


@pytest.mark.parametrize(
    "q_heads, rope_heads, dout_heads",
    [
        (96, 64, None),
        (64, 96, None),
        (96, 128, None),
        (96, 96, 64),
        (64, 64, 96),
        (128, 128, 96),
    ],
)
def test_validate_rejects_mismatched_head_counts(q_heads, rope_heads, dout_heads):
    """``q_rope`` and ``dout`` carry the head count of ``q_latent``."""
    with pytest.raises(ValueError):
        _validate(q_heads, rope_heads=rope_heads, dout_heads=dout_heads)


@pytest.mark.parametrize("heads", (96, 128))
def test_nope_and_det_forms_reject_more_heads(heads):
    """The NoPE form and the deterministic package keep 64 query heads: 96 and 128 are refused with
    ``ValueError`` saying so (host validation), while the same tensors at 64 heads validate."""
    kv = torch.zeros(16, D_LATENT, dtype=torch.bfloat16)
    idx = torch.zeros(8, 5, dtype=torch.int32)
    assert tuple(
        cake_backend.validate_dsa_nope_inputs(
            torch.zeros(8, 64, D_LATENT, dtype=torch.bfloat16), kv, idx
        )
    ) == (8, 16, 5)
    with pytest.raises(ValueError, match="64 query heads only"):
        cake_backend.validate_dsa_nope_inputs(
            torch.zeros(8, heads, D_LATENT, dtype=torch.bfloat16), kv, idx
        )
    for h in (64, heads):
        inp = hl.host_inputs(8, 16, 5, h)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
        if h == 64:
            assert tuple(det_backend.validate_dsa_train_inputs(*args)[:3]) == (8, 16, 5)
            continue
        with pytest.raises(ValueError, match="64 query heads only"):
            det_backend.validate_dsa_train_inputs(*args)


@pytest.mark.parametrize("heads", (96, 128))
def test_sm107a_refuses_more_than_64_heads(monkeypatch, heads):
    """On sm_107a the windows do not ship yet: a 96- or 128-head call is refused before any kernel of the package
    launches (host, the architecture stood in) -- the eager forward and backward, the flat and packed (varlen, its
    torch index route) autograd entries, ``prepare_dsa_train`` and the workspace size; 64 heads launch as before.
    The public step entries: ``test_cake_dsa_sparse_attention_nondet_train_api.py``."""
    h = hl.HostLaunches(monkeypatch, cake_backend, cake_jit, cake_launch, "sm_107a")
    monkeypatch.setattr(
        cake_backend, "nope_index_prep_available", lambda device=None: False
    )
    inp = hl.host_inputs(8, 64, 16, heads)
    h.bind(inp.q_latent, inp.q_rope, inp.dout)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
    refused = "sm_107a serves 64 query heads only"
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.forward(*args)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.backward(*args, inp.q_latent, -inp.q_latent, inp.lse, inp.dout)
    leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.cake_dsa_sparse_attention_nondet(*leaves, inp.indices)
    cu_q = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 24, 64], dtype=torch.int32)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.cake_dsa_sparse_attention_nondet_varlen(
            *leaves, inp.indices % 24, cu_q, cu_k, causal=True
        )
    with pytest.raises(NotImplementedError, match=refused):
        prepare_dsa_train(*args, dout=inp.dout)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.dsa_train_workspace_size(8, 64, 16, num_heads=heads)
    assert not h.launches
    small = hl.host_inputs(8, 64, 16, 64)
    h.bind(small.q_latent, small.q_rope, small.dout)
    out, lse, o_lo = cake_backend.forward(
        small.q_latent, small.q_rope, small.kv_latent, small.k_rope, small.indices
    )
    hl.check_forward(h.launches, small, out, lse, o_lo, with_o_lo=True)


# ---------------------------------------------------------------------------
# Device tests (sm_100a / sm_103a at 96 and 128 heads; every architecture at 64)
# ---------------------------------------------------------------------------


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_heads(heads: int, *, backward: bool = False) -> None:
    """A device and a registered program that serve ``heads`` query heads (skip otherwise): 64 heads on every
    supported architecture, 96 and 128 on sm_100a / sm_103a."""
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda"), backward=backward):
        pytest.skip("generated DSA training program not registered for this device")
    if heads != 64 and cake_backend.arch_for(torch.device("cuda")) == "sm_107a":
        pytest.skip("96 and 128 query heads ship on sm_100a / sm_103a; sm_107a follows")


def _require_key_passes_and_direct() -> None:
    _, record = cake_backend.record_for(torch.device("cuda"))
    stages = cake_jit.registered_stages()
    if not set(KEY_PASS_STAGES) <= set(stages):
        pytest.skip("the registered program has no key-range-pass stages")
    if not cake_backend.record_direct_stages(record, stages):
        pytest.skip("the registered program has no natural-layout (direct) main stage")


@pytest.fixture(**nv.WITH_O_LO_FIXTURE)
def with_o_lo(request, monkeypatch):
    """Each judged test runs as shipped (``True``) and with ``with_o_lo=False`` (marker ``no_olo``)."""
    yield from nv.with_o_lo_case(request.param, monkeypatch, cake_backend, cake_jit)


def _rope_outputs(forward=None, grads=None, **more) -> dict:
    """One call's outputs by name for ``nv.judged`` (see the backend suite's helper of the same name)."""
    found = {}
    if forward is not None:
        found.update(zip(("out", "lse", "o_lo"), forward, strict=True))
    if grads is not None:
        names = ("dq_latent", "dq_rope", "dkv_latent", "dk_rope")
        found.update((n, g) for n, g in zip(names, grads, strict=True) if g is not None)
    found.update(more)
    return found


VALIDITY = ("all", "some", "none")


def _validity_rows(inp, validity: str) -> torch.Tensor:
    """The global rows of a draw: ``all`` -- every slot valid (the queries are the tail of a document of at least
    ``TOPK`` keys); ``some`` -- ``-1`` and ``>= S`` slots anywhere, a run of ``-1`` in the middle and rows without a
    valid slot; ``none`` -- no valid slot in any row (``-1`` and ``>= S``)."""
    S = inp.total_k
    idx = inp.idx_global.clone()
    if validity == "all":
        assert bool((idx >= 0).all())
    elif validity == "some":
        idx[::4, ::5] = -1
        idx[1::4, 3::7] = S + 5
        idx[2::4, 64:192] = -1
        idx[3::16] = -1
        idx[11::16, 1::2] = S
        idx[11::16, ::2] = -1
    else:
        assert validity == "none"
        idx[::2] = -1
        idx[1::2] = S + 1
        assert not bool(((idx >= 0) & (idx < S)).any())
    return idx


def _cases_naive_tier():
    cases = [(h, v) for h in HEADS for v in VALIDITY]
    cases += [(h, "some") for h in REDUCED_HEADS]
    return cases


@pytest.mark.parametrize(
    "heads, validity",
    _cases_naive_tier(),
    ids=[f"H{h}-{v}" for h, v in _cases_naive_tier()],
)
def test_naive_tier_heads(with_o_lo, heads, validity, record_property):
    """``out``, ``lse``, ``o_sum``, ``dq`` and the key gradients at ``heads`` query heads, scale ``192**-0.5``, top-k
    2048, with the rows of ``validity`` (:func:`_validity_rows`), judged against the FP64 oracle at the naive BF16
    tier on every draw: the forward; the backward with fresh key gradients, with the caller's ``dkv_acc`` through
    the natural-layout main stage (``dkv_direct=True``) and through the cast (``dkv_direct=False``), each over 1, 2
    and 3 forced key passes (1 and 3 at 128 heads).

    ``none`` has no accuracy to judge.  No row has a valid key, so on every draw the FP64 oracle is the same
    constant whatever the inputs: ``out``, ``o_sum`` and ``dq`` are 0 and ``lse`` is ``-inf`` on every row (a row
    without a valid slot), the fresh key gradients and the ``dkv_acc`` increment are 0 on every key row (no slot
    names one).  The naive tier is exact there too, and the judge refuses draws whose oracle values are the same
    (``tests/cake_naive.py`` holds such a case to its contract only).  So ``none`` makes the same calls over the
    same :data:`DRAWS` draws, holds each call to its contract (``nv.judged``) and each output to the oracle
    exactly (``contract(..., exact=True)``: equal values, no tolerance).  Rows without a valid key beside rows
    with one are judged at the tier in ``some``."""
    _require_heads(heads, backward=True)
    _require_key_passes_and_direct()
    passes = (1, 3) if heads in REDUCED_HEADS else (1, 2, 3)

    def case(s, j):
        inp = make_inputs([256], [4096], seed=s, topk=TOPK, heads=heads)
        idx = _validity_rows(inp, validity)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, idx)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        bwd = args + (out, o_lo, lse, inp.dout)
        runs = {"forward": _rope_outputs(fwd)}
        for p in passes:
            grads = cake_backend.backward(*bwd, softmax_scale=SCALE, key_passes=p)
            runs[f"{p} pass"] = _rope_outputs(grads=grads)
            for direct in (True, False):
                acc = torch.zeros(inp.total_k, D_QK, device="cuda")
                grads = cake_backend.backward(
                    *bwd,
                    softmax_scale=SCALE,
                    key_passes=p,
                    dkv_acc=acc,
                    dkv_direct=direct,
                )
                assert grads[2] is None and grads[3] is None
                label = f"{p} pass dkv_acc {'direct' if direct else 'cast'}"
                runs[label] = _rope_outputs(
                    grads=grads, dkv_acc=nv.Acc(before=torch.zeros_like(acc), after=acc)
                )
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, indices=idx, scale=SCALE), runs)

    seed = SEED + 10 * heads + VALIDITY.index(validity)
    if validity != "none":
        nv.judge_draws(record_property, case, seed=seed)
        return
    for j in range(DRAWS):
        found, oracle, _ = case(draw_seed(seed, j), j)
        for name, want in oracle.items():
            constant = float("-inf") if name.endswith("lse") else 0.0
            assert bool((want == constant).all()), (
                f"draw {j}: the oracle's {name} is not {constant}"
            )
        wrong = [
            message
            for name, value in found.items()
            if (message := contract(name, value, oracle[name], exact=True))
        ]
        assert not wrong, f"draw {j}: " + "; ".join(wrong)


@pytest.mark.parametrize("heads", (96,) + REDUCED_HEADS)
def test_varlen_causal_heads(with_o_lo, heads, record_property):
    """A packed batch of four documents (query segments the tail of their key prefixes, one full document, one
    shorter than the top-k width) with ``causal=True`` through the varlen autograd entry, and the same rows through
    the flat autograd entry: ``out`` and every gradient of both judged at ``heads`` query heads."""
    _require_heads(heads, backward=True)
    seq_q, seq_k = [300, 77, 512, 129], [300, 1000, 1024, 4000]

    def case(s, j):
        inp = make_inputs(seq_q, seq_k, seed=s, topk=TOPK, heads=heads)
        cu_q, cu_k = inp.cu_seqlens_q, inp.cu_seqlens_k
        rows = offset_gather_kv_indices(inp.idx_local, cu_q, cu_k, causal=True)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
        runs = {}
        for label in ("varlen", "flat"):
            leaves = [t.detach().clone().requires_grad_() for t in args]
            if label == "varlen":
                out = cake_backend.cake_dsa_sparse_attention_nondet_varlen(
                    *leaves, inp.idx_local, cu_q, cu_k, causal=True, softmax_scale=SCALE
                )
            else:
                out = cake_backend.cake_dsa_sparse_attention_nondet(
                    *leaves, rows, softmax_scale=SCALE, num_segments=len(seq_q)
                )
            grads = torch.autograd.grad(out, leaves, inp.dout)
            runs[label] = _rope_outputs(grads=grads, out=out.detach())
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, indices=rows, scale=SCALE), runs)

    nv.judge_draws(record_property, case, seed=SEED + 2000 + heads)


#: ``(top-k, key rows, destination rows)`` of :func:`test_dkv_acc_repeated_dst_heads`: a short row, and the issue's
#: top-k 2048 over 4096 keys.
DST_GEOMETRIES = {"topk512": (512, 1024, 300), "topk2048": (TOPK, 4096, 1200)}


@pytest.mark.parametrize("geometry", list(DST_GEOMETRIES))
@pytest.mark.parametrize("heads", (96,) + REDUCED_HEADS)
def test_dkv_acc_repeated_dst_heads(with_o_lo, heads, geometry, record_property):
    """``dkv_acc`` through a many-to-few destination map (``geometry``: top-k 512 with 1024 key rows into 300
    destination rows, or top-k 2048 with 4096 key rows into 1200; repeated destinations summing as ``index_add_``)
    at ``heads`` query heads: the natural-layout main stage and the cast's accumulate path, one pass and two forced
    key passes, through the eager backward and through a prepared runner's step (two forced key passes: the window
    launches per pass reuse the runner's carved window statistics and dQ partial); each call's ``dq`` and buffer
    increment judged (unmapped destination rows and the columns past 576 keep their bits), the runner's forward
    outputs too.  The same draw also judges the eager backward's natural FP32 key gradients (``dkv_fp32=True``, one
    and two key passes)."""
    _require_heads(heads, backward=True)
    _require_key_passes_and_direct()
    topk, num_kv, num_dst = DST_GEOMETRIES[geometry]

    def case(s, j):
        inp = make_inputs([256], [num_kv], seed=s, topk=topk, heads=heads)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        gen = torch.Generator(device="cuda").manual_seed(s + 3)
        dst_map = torch.randint(
            0, num_dst, (inp.total_k,), device="cuda", generator=gen
        ).to(torch.int32)
        runs = {"forward": _rope_outputs(fwd)}
        for direct in (True, False):
            for p in (1, 2):
                acc = torch.randn(num_dst, 704, device="cuda", generator=gen)
                before = acc.clone()
                grads = cake_backend.backward(
                    *args,
                    out,
                    o_lo,
                    lse,
                    inp.dout,
                    softmax_scale=SCALE,
                    key_passes=p,
                    dkv_acc=acc,
                    dkv_dst_map=dst_map,
                    dkv_direct=direct,
                )
                runs[f"{'direct' if direct else 'cast'} {p} pass"] = _rope_outputs(
                    grads=grads,
                    dkv_acc=nv.Acc(before=before, after=acc, dst_map=dst_map),
                )
            acc = torch.randn(num_dst, 704, device="cuda", generator=gen)
            before = acc.clone()
            runner = prepare_dsa_train(
                *args,
                dout=inp.dout,
                softmax_scale=SCALE,
                key_passes=2,
                dkv_acc=acc,
                dkv_dst_map=dst_map,
                dkv_direct=direct,
            )
            assert runner.num_heads == heads and runner.dkv_direct == direct
            _, hw_main = cake_backend.head_window_plan(
                heads,
                cake_backend.arch_for(torch.device("cuda")),
                backward=True,
                key_passes=2,
                dkv_direct=direct,
            )
            if hw_main is None:
                # the two-window bridge: per pass, the main launches of both windows
                windows = {
                    (key.head0, key.key[2])
                    for key in runner.backward_order
                    if isinstance(key, cake_backend.WindowLaunch)
                    and isinstance(key.key, tuple)
                }
                assert windows == {(b, p) for b in (heads - 64, 0) for p in (0, 1)}, (
                    windows
                )
            else:
                # the head-window build: per pass, one launch over the chunk's items
                passes = [
                    key[2]
                    for key in runner.backward_order
                    if isinstance(key, tuple) and key[0] == hw_main
                ]
                assert passes == [0, 1], runner.backward_order
                assert not any(
                    isinstance(key, cake_backend.WindowLaunch)
                    for key in runner.backward_order
                )
            grads = runner.step()
            assert grads[2] is None and grads[3] is None
            runs[f"runner {'direct' if direct else 'cast'} 2 passes"] = _rope_outputs(
                (runner.out, runner.lse, runner.o_lo),
                grads,
                dkv_acc=nv.Acc(before=before, after=acc, dst_map=dst_map),
            )
        for p in (1, 2):
            grads = cake_backend.backward(
                *args,
                out,
                o_lo,
                lse,
                inp.dout,
                softmax_scale=SCALE,
                key_passes=p,
                dkv_fp32=True,
            )
            assert grads[2].dtype == torch.float32 and grads[3].dtype == torch.float32
            runs[f"fp32 dkv {p} pass"] = _rope_outputs(grads=grads)
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, scale=SCALE), runs)

    nv.judge_draws(
        record_property,
        case,
        seed=SEED + 3000 + heads + 500 * list(DST_GEOMETRIES).index(geometry),
    )


@pytest.mark.parametrize("heads", HEADS + REDUCED_HEADS)
def test_one_token_heads(with_o_lo, heads, record_property):
    """``T == 1`` (every grid clamps to one CTA) at ``heads`` query heads: the eager forward and backward, the
    autograd entry and the prepared runner's step, judged."""
    _require_heads(heads, backward=True)

    def case(s, j):
        inp = make_inputs([1], [3000], seed=s, topk=TOPK, heads=heads)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        grads = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, softmax_scale=SCALE
        )
        leaves = [t.detach().clone().requires_grad_() for t in args[:4]]
        out_a = cake_backend.cake_dsa_sparse_attention_nondet(
            *leaves, inp.idx_global, softmax_scale=SCALE
        )
        grads_a = torch.autograd.grad(out_a, leaves, inp.dout)
        runner = prepare_dsa_train(*args, dout=inp.dout, softmax_scale=SCALE)
        grads_r = runner.step()
        torch.cuda.synchronize()
        runs = {
            "eager": _rope_outputs(fwd, grads),
            "autograd": _rope_outputs(grads=grads_a, out=out_a.detach()),
            "runner": _rope_outputs((runner.out, runner.lse, runner.o_lo), grads_r),
        }
        return nv.judged(nv.rope_problem(inp, scale=SCALE), runs)

    nv.judge_draws(record_property, case, seed=SEED + 4000 + heads)


@pytest.mark.parametrize("heads", HEADS)
def test_large_t_heads(with_o_lo, heads, record_property):
    """T = 16380 queries, one causal document of 16380 keys, top-k 2048, at ``heads`` query heads (the issue's
    per-rank row): the forward and the backward (the key-pass policy's plan, then two forced passes over the token
    chunks) judged over the draws.  The oracle runs in blocks of query rows (``nv.BLOCK_SLOTS`` (row, slot) pairs),
    about 2e13 FP64 operations per draw at 96 heads: opt in with ``CAKE_PPLX_DSA_LARGE_TESTS=1``.  A default run
    skips it; the GPU validation runs it on sm_100a and sm_103a with that variable and reports its four cases
    (heads x ``with_o_lo``) as their own gate, with a step time limit sized for the FP64 oracle of the device."""
    if os.environ.get(LARGE_TESTS_ENV, "") != "1":
        pytest.skip(
            f"set {LARGE_TESTS_ENV}=1 to run the T = 16380 case (minutes of FP64 oracle per draw)"
        )
    _require_heads(heads, backward=True)
    T = 16380

    def case(s, j):
        inp = make_inputs([T], [T], seed=s, topk=TOPK, heads=heads)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        grads = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, softmax_scale=SCALE
        )
        runs = {"policy": _rope_outputs(fwd, grads)}
        if set(KEY_PASS_STAGES) <= set(cake_jit.registered_stages()):
            del grads
            grads = cake_backend.backward(
                *args, out, o_lo, lse, inp.dout, softmax_scale=SCALE, key_passes=2
            )
            runs["2 passes"] = _rope_outputs(grads=grads)
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, scale=SCALE), runs)

    nv.judge_draws(record_property, case, seed=SEED + 5000 + heads)


def test_long_keys_heads(with_o_lo, record_property):
    """128 queries at the tail of one causal document of S = 1,048,320 keys (the issue's last context-parallel
    rank), top-k 2048, 96 query heads: the forward and the backward with the policy's plan and with two and three
    forced key passes (fresh key gradients, then the caller's ``dkv_acc`` through the natural-layout stage),
    judged."""
    _require_heads(96, backward=True)
    _require_key_passes_and_direct()
    S = 1_048_320

    def case(s, j):
        inp = make_inputs([128], [S], seed=s, topk=TOPK, heads=96)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        bwd = args + (out, o_lo, lse, inp.dout)
        runs = {
            "policy": _rope_outputs(
                fwd, cake_backend.backward(*bwd, softmax_scale=SCALE)
            )
        }
        for p in (2, 3):
            runs[f"{p} passes"] = _rope_outputs(
                grads=cake_backend.backward(*bwd, softmax_scale=SCALE, key_passes=p)
            )
        acc = torch.zeros(S, D_QK, device="cuda")
        grads = cake_backend.backward(
            *bwd, softmax_scale=SCALE, key_passes=2, dkv_acc=acc, dkv_direct=True
        )
        runs["2 passes dkv_acc"] = _rope_outputs(
            grads=grads, dkv_acc=nv.Acc(before=torch.zeros_like(acc), after=acc)
        )
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, scale=SCALE), runs)

    nv.judge_draws(record_property, case, seed=SEED + 6000)


def test_packed_views_head_major_dout_and_lengths_heads(with_o_lo, record_property):
    """At 96 query heads: ``q_latent`` / ``q_rope`` as views of a packed ``[T, 96, 576]`` query and ``kv`` of a
    packed ``[S, 576]`` row; the head-major ``dout`` an output projection hands back (``[T, 96, 512]`` view of
    ``[96, T, 512]``), which the eager backward and a prepared runner's step (one pass, and two forced key passes
    where the program registers them) copy into their dense workspace ``dout`` first, and a packed-slice ``dout``
    (channels ``0:512`` of a ``[T, 96, 576]`` buffer, NaN past them), which both read in place; caller lengths that
    cut rows (0 and 3 included) and the forward deriving the lengths into the caller's vector -- every call judged
    against the problem of the lengths it used."""
    _require_heads(96, backward=True)
    passes = (
        (1, 2) if set(KEY_PASS_STAGES) <= set(cake_jit.registered_stages()) else (1,)
    )

    def case(s, j):
        inp = make_inputs(
            [256], [2048], seed=s, topk=1024, heads=96, dout_layout="head_major"
        )
        assert inp.dout.stride() == (512, 256 * 512, 1)
        q = torch.cat([inp.q_latent, inp.q_rope], -1).contiguous()
        kv = torch.cat([inp.kv_latent, inp.k_rope], -1).contiguous()
        views = (
            q[..., :D_LATENT],
            q[..., D_LATENT:],
            kv[:, :D_LATENT],
            kv[:, D_LATENT:],
        )
        lengths = inp.topk_length.clone()
        lengths[5] = 0
        lengths[9] = 3
        fwd = cake_backend.forward(
            *views, inp.idx_global, topk_length=lengths, softmax_scale=SCALE
        )
        out, lse, o_lo = fwd
        grads = cake_backend.backward(
            *views,
            inp.idx_global,
            out,
            o_lo,
            lse,
            inp.dout,
            topk_length=lengths,
            softmax_scale=SCALE,
        )
        runs = {"eager": _rope_outputs(fwd, grads)}
        for p in passes:
            runner = prepare_dsa_train(
                *views,
                inp.idx_global,
                dout=inp.dout,
                topk_length=lengths,
                softmax_scale=SCALE,
                key_passes=p,
            )
            assert "dout_copy" in runner.layout
            assert cake_backend.HeadStep("dout_copy") in runner.backward_order
            runs[f"runner {p} pass"] = _rope_outputs(
                (runner.out, runner.lse, runner.o_lo), runner.step()
            )
        # the packed-slice dout: dense over its rows (token stride 96 head strides), read in place at head stride 576
        wide = torch.full(
            (inp.total_q, 96, D_QK), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        wide[..., :D_LATENT].copy_(inp.dout)
        packed_dout = wide[..., :D_LATENT]
        runs["packed dout eager"] = _rope_outputs(
            grads=cake_backend.backward(
                *views,
                inp.idx_global,
                out,
                o_lo,
                lse,
                packed_dout,
                topk_length=lengths,
                softmax_scale=SCALE,
            )
        )
        runner = prepare_dsa_train(
            *views,
            inp.idx_global,
            dout=packed_dout,
            topk_length=lengths,
            softmax_scale=SCALE,
        )
        assert "dout_copy" not in runner.layout
        runs["packed dout runner"] = _rope_outputs(
            (runner.out, runner.lse, runner.o_lo), runner.step()
        )
        filled = torch.empty(inp.total_q, dtype=torch.int32, device="cuda")
        derived = cake_backend.forward(
            *views,
            inp.idx_global,
            topk_length=filled,
            derive_topk_length=True,
            softmax_scale=SCALE,
        )
        torch.cuda.synchronize()
        assert torch.equal(filled, derive_topk_length(inp.idx_global, inp.total_k))
        return nv.merge(
            nv.judged(
                nv.rope_problem(inp, topk_length=lengths, scale=SCALE),
                runs,
                prefix="lengths ",
            ),
            nv.judged(
                nv.rope_problem(inp, dout=None, scale=SCALE),
                {"": _rope_outputs(derived)},
                prefix="derived ",
            ),
        )

    nv.judge_draws(record_property, case, seed=SEED + 7000)


def test_heads_and_counts_load_no_new_module(with_o_lo, record_property):
    """64, 96 and 128 query heads, new token / key counts and run-time softmax scales run on the modules already
    loaded (no head count, shape or scale in a build): after one warm-up step per head count at scale
    ``192**-0.5``, the step of every draw -- its own head count, ``T``, ``S`` and scale (:data:`SCALES`, every head
    count meeting each scale) -- adds no entry and no miss to the module loader's cache; its outputs are judged."""
    _require_heads(96, backward=True)

    def step(inp, scale):
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=scale)
        out, lse, o_lo = fwd
        grads = cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, softmax_scale=scale
        )
        torch.cuda.synchronize()
        return _rope_outputs(fwd, grads)

    for heads in (64, 96, 128):
        step(make_inputs([64], [512], seed=SEED + 8000, topk=256, heads=heads), SCALE)
    loaded = cake_jit.load_cake_dsa_train_module.cache_info()

    def draw(s, j):
        heads = (64, 96, 128)[j % 3]
        scale = SCALES[(j // 3) % len(SCALES)]
        T = 100 + 37 * j
        inp = make_inputs([T], [T + 211 * j], seed=s, topk=256, heads=heads)
        run = step(inp, scale)
        after = cake_jit.load_cake_dsa_train_module.cache_info()
        assert (after.misses, after.currsize) == (loaded.misses, loaded.currsize), (
            heads,
            T,
            scale,
            loaded,
            after,
        )
        return nv.judged(nv.rope_problem(inp, scale=scale), {"": run})

    nv.judge_draws(record_property, draw, seed=SEED + 8000)


def _graph_capture(fn):
    """``fn`` eagerly on a side stream (detached copies of its outputs), warmed twice there, captured into a CUDA
    graph on that stream and replayed twice: ``(eager, captured)``."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = tuple(None if t is None else t.detach().clone() for t in fn())
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = fn()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    return eager, captured


def test_step_at_96_heads_captures_into_a_cuda_graph_and_does_not_synchronize(
    with_o_lo, record_property
):
    """A training step at 96 query heads through the autograd entry (forward, ``autograd.grad`` to the query
    leaves, the key gradients into ``dkv_acc`` through a destination map with repeated rows) captures into a CUDA
    graph and replays, judged beside the eager step, and so does a prepared runner's step over a head-major ``dout``
    (its ``"dout_copy"`` step included) into the same kind of ``dkv_acc``; in steady state the step makes no pageable
    host-to-device copy and no synchronizing call and returns while queued device work still runs, as do the eager
    forward / backward pair and the prepared runner's step, which also allocates nothing."""
    from tests.experimental.test_helpers.cake_host_sync import (
        census,
        nonblocking,
        summary,
    )

    _require_heads(96, backward=True)

    def stepper(inp):
        S = inp.total_k
        dst_map = (torch.arange(S, device="cuda", dtype=torch.int32) // 2).contiguous()
        acc = torch.zeros((S + 1) // 2, D_QK, device="cuda")
        q_leaf = inp.q_latent.detach().clone().requires_grad_()
        r_leaf = inp.q_rope.detach().clone().requires_grad_()

        def step():
            acc.zero_()
            out, lse = cake_backend.cake_dsa_sparse_attention_nondet(
                q_leaf,
                r_leaf,
                inp.kv_latent,
                inp.k_rope,
                inp.idx_global,
                softmax_scale=SCALE,
                return_lse=True,
                dkv_acc=acc,
                dkv_dst_map=dst_map,
            )
            dq, dr = torch.autograd.grad(out, (q_leaf, r_leaf), inp.dout)
            return out, lse, dq, dr, acc

        return step, dst_map

    def runner_stepper(inp):
        S = inp.total_k
        dst_map = (torch.arange(S, device="cuda", dtype=torch.int32) // 2).contiguous()
        acc = torch.zeros((S + 1) // 2, D_QK, device="cuda")
        # the same values as the head-major view of a [96, T, 512] buffer: the step copies it first
        dout = inp.dout.transpose(0, 1).contiguous().transpose(0, 1)
        runner = prepare_dsa_train(
            inp.q_latent,
            inp.q_rope,
            inp.kv_latent,
            inp.k_rope,
            inp.idx_global,
            dout=dout,
            softmax_scale=SCALE,
            dkv_acc=acc,
            dkv_dst_map=dst_map,
        )
        assert cake_backend.HeadStep("dout_copy") in runner.backward_order

        def step():
            acc.zero_()
            dq, dr, _, _ = runner.step()
            return runner.out, runner.lse, dq, dr, acc

        return step, dst_map

    def case(s, j):
        inp = make_inputs([512], [1024], seed=s, topk=512, heads=96)
        runs = {}
        for name, make in (("autograd", stepper), ("runner", runner_stepper)):
            step, dst_map = make(inp)
            eager, captured = _graph_capture(step)
            replayed = [t.detach().clone() for t in captured]
            for label, got in (("eager", eager), ("graph", replayed)):
                run = dict(
                    zip(("out", "lse", "dq_latent", "dq_rope"), got[:4], strict=True)
                )
                run["dkv_acc"] = nv.Acc(
                    before=torch.zeros_like(got[4]), after=got[4], dst_map=dst_map
                )
                runs[f"{name} {label}"] = run
        return nv.judged(nv.rope_problem(inp, scale=SCALE), runs)

    nv.judge_draws(record_property, case, seed=SEED + 9000)
    inp = make_inputs([512], [1024], seed=SEED + 9001, topk=512, heads=96)
    device = torch.device("cuda")
    step, dst_map = stepper(inp)
    step()
    torch.cuda.synchronize()
    c = census(step, device)
    assert nonblocking(c), summary(c)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)

    def eager_pair():
        out, lse, o_lo = cake_backend.forward(*args, softmax_scale=SCALE)
        return cake_backend.backward(
            *args, out, o_lo, lse, inp.dout, softmax_scale=SCALE
        )

    eager_pair()
    torch.cuda.synchronize()
    c = census(eager_pair, device)
    assert nonblocking(c), summary(c)
    acc = torch.zeros(dst_map.numel() // 2 + 1, D_QK, device="cuda")
    runner = prepare_dsa_train(
        *args, dout=inp.dout, softmax_scale=SCALE, dkv_acc=acc, dkv_dst_map=dst_map
    )
    runner.step()
    torch.cuda.synchronize()
    c = census(runner.step, device)
    assert nonblocking(c), summary(c)
    assert c["device_alloc"] == 0, summary(c)


CKPT_T, CKPT_W = 2048, 256


def _ckpt_inputs(heads: int, seed: int = 0):
    """A checkpointing construction at ``heads`` query heads: sliding-window rows over one document of
    ``CKPT_T`` tokens (``CKPT_W`` keys per row), the leaves, ``dout`` and the prefix sums."""
    T, W = CKPT_T, CKPT_W
    g = torch.Generator(device="cuda").manual_seed(seed)

    def leaf(*shape):
        return (
            torch.randn(*shape, device="cuda", generator=g).bfloat16().requires_grad_()
        )

    rows = torch.arange(T, device="cuda", dtype=torch.int32)[:, None] - torch.arange(
        W, device="cuda", dtype=torch.int32
    )
    return SimpleNamespace(
        q=leaf(T, heads, D_LATENT),
        q_rope=leaf(T, heads, D_ROPE),
        kv=leaf(T, D_LATENT),
        k_rope=leaf(T, D_ROPE),
        dout=torch.randn(T, heads, D_LATENT, device="cuda", generator=g).bfloat16(),
        idx=torch.where(rows >= 0, rows, -1).to(torch.int32),
        cu=torch.tensor([0, T], device="cuda", dtype=torch.int32),
    )


def _ckpt_fn(name, inp, acc=None):
    leaves = (inp.q, inp.q_rope, inp.kv, inp.k_rope)
    kw = dict(softmax_scale=SCALE)
    if acc is not None:
        kw["dkv_acc"] = acc
    if name == "rope":

        def fn(q, q_rope, kv, k_rope):
            return cake_backend.cake_dsa_sparse_attention_nondet(
                q, q_rope, kv, k_rope, inp.idx, **kw
            )

    else:

        def fn(q, q_rope, kv, k_rope):
            return cake_backend.cake_dsa_sparse_attention_nondet_varlen(
                q, q_rope, kv, k_rope, inp.idx, inp.cu, inp.cu, causal=True, **kw
            )

    return fn, leaves


@pytest.mark.parametrize("acc", (False, True), ids=("dkv", "dkv_acc"))
@pytest.mark.parametrize("name", ("rope", "rope_varlen"))
def test_ckpt_at_96_heads_keeps_nothing_alive(name, acc):
    """``checkpoint(fn, ..., use_reentrant=False)`` of the rotary flat and varlen autograd entries at 96 query heads
    keeps at most ``CKPT_RETAINED_LIMIT_MIB`` beyond its output (no ``o_lo``, ``lse``, lengths or window scratch)."""
    _require_heads(96, backward=True)
    inp = _ckpt_inputs(96)
    buf = torch.zeros(CKPT_T, D_QK, device="cuda") if acc else None
    fn, leaves = _ckpt_fn(name, inp, buf)
    fn(*leaves)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = checkpoint(fn, *leaves, use_reentrant=False)
    torch.cuda.synchronize()
    kept = (
        torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
    ) / MIB
    assert kept <= CKPT_RETAINED_LIMIT_MIB, (
        f"{name}: {kept:.1f} MiB kept alive after a checkpointed forward"
    )


@pytest.mark.parametrize("name", ("rope", "rope_varlen"))
def test_ckpt_at_96_heads_outputs_and_gradients(with_o_lo, name, record_property):
    """The checkpointed step and the eager step of the rotary flat and varlen autograd entries at 96 query heads
    (fresh key gradients, then the caller's ``dkv_acc`` zero-filled between the checkpointed forward and its
    backward): ``out`` and every gradient or buffer increment of both steps judged on the same draw."""
    _require_heads(96, backward=True)

    def draw(s, j):
        inp = _ckpt_inputs(96, seed=s)
        rows = inp.idx
        if name == "rope_varlen":
            rows = offset_gather_kv_indices(inp.idx, inp.cu, inp.cu, causal=True)
        runs = {}
        for with_acc in (False, True):
            acc = torch.zeros(CKPT_T, D_QK, device="cuda") if with_acc else None
            fn, leaves = _ckpt_fn(name, inp, acc)
            for label in ("eager", "checkpointed"):
                if acc is not None:
                    acc.zero_()
                out = (
                    checkpoint(fn, *leaves, use_reentrant=False)
                    if label == "checkpointed"
                    else fn(*leaves)
                )
                if acc is not None:
                    acc.zero_()
                grads = torch.autograd.grad(out, leaves, inp.dout, allow_unused=True)
                torch.cuda.synchronize()
                more = {}
                if acc is not None:
                    assert grads[2] is None and grads[3] is None
                    more["dkv_acc"] = nv.Acc(
                        before=torch.zeros_like(acc), after=acc.clone()
                    )
                runs[f"{label}{' dkv_acc' if with_acc else ''}"] = _rope_outputs(
                    grads=grads, out=out.detach(), **more
                )
        problem = nv.Problem(
            q_latent=inp.q,
            kv_latent=inp.kv,
            indices=rows,
            q_rope=inp.q_rope,
            k_rope=inp.k_rope,
            dout=inp.dout,
            scale=SCALE,
        )
        return nv.judged(problem, runs)

    nv.judge_draws(record_property, draw, seed=SEED + 9500)


#: The per-head outputs the head-window builds and the two-window bridge give bit for bit alike.
PER_HEAD_OUTPUTS = ("out", "lse", "o_lo", "dq_latent", "dq_rope")


def _same_per_head_outputs(found: dict, bridge: dict) -> None:
    for label, got in found.items():
        for name in PER_HEAD_OUTPUTS:
            if got.get(name) is None:
                continue
            want = bridge[label][name]
            assert got[name].dtype == want.dtype and got[name].shape == want.shape
            diff = int((got[name] != want).sum())
            assert torch.equal(got[name], want), (
                f"{label} {name}: {diff} elements differ from the two-window bridge"
            )


@pytest.mark.parametrize("heads", (96,) + REDUCED_HEADS)
def test_head_window_builds_match_the_bridge(
    with_o_lo, heads, monkeypatch, record_property
):
    """Where the program ships the head-window builds for the device (sm_100a / sm_103a; skip where it does not), a
    call over ``heads`` query heads runs them (the forward one head-window launch, the backward the main stage's
    build, no window launch), and its per-head outputs equal bit for bit the two-window bridge's on the same inputs
    (every head's rows go through the same 64-head kernel either way): ``out``, ``lse``, ``o_lo`` and ``dq`` of the
    eager forward, of the eager backward at one key pass (permuted accumulation and, into ``dkv_acc``, the
    natural layout) and at two forced key passes, and of a prepared runner's step -- also at T = 2,304, where two
    forced passes run the head-window build over two token chunks (2,176 tokens at two items per token) and the
    bridge over one.  The head-window calls' outputs, key gradients included, are judged at the naive tier.  The
    single-pass forward, which replaces the head-window forward where it ships (another kernel, not the bridge's
    bits: :func:`test_single_pass_forward_through_the_loader`), is set aside for this test."""
    _require_heads(heads, backward=True)
    _require_key_passes_and_direct()
    monkeypatch.setattr(cake_backend, "single_pass_stage", lambda stage, arch: None)
    arch = cake_backend.arch_for(torch.device("cuda"))
    forward, main = cake_backend.head_window_plan(
        heads, arch, with_o_lo=with_o_lo, backward=True
    )
    if forward is None or main is None:
        pytest.skip(
            "the registered program ships no head-window builds for this device"
        )
    head_window_stage = cake_backend.head_window_stage

    def calls(inp, *, bridge: bool, passes=(1, 2), direct=True, runner=True):
        monkeypatch.setattr(
            cake_backend,
            "head_window_stage",
            (lambda stage, arch: None) if bridge else head_window_stage,
        )
        cake_backend.BINDING_CACHE._entries.clear()  # every arm plans afresh
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        runs = {"forward": _rope_outputs(fwd)}
        for p in passes:
            grads = cake_backend.backward(
                *args,
                out,
                o_lo,
                lse,
                inp.dout,
                softmax_scale=SCALE,
                key_passes=p,
                dkv_direct=False,
            )
            runs[f"{p} pass"] = _rope_outputs(grads=grads)
        if direct:
            acc = torch.zeros(inp.total_k, 704, device="cuda")
            grads = cake_backend.backward(
                *args,
                out,
                o_lo,
                lse,
                inp.dout,
                softmax_scale=SCALE,
                key_passes=1,
                dkv_acc=acc,
                dkv_direct=True,
            )
            runs["direct"] = _rope_outputs(
                grads=grads, dkv_acc=nv.Acc(before=torch.zeros_like(acc), after=acc)
            )
        prepared = None
        if runner:
            prepared = prepare_dsa_train(*args, dout=inp.dout, softmax_scale=SCALE)
            grads = prepared.step()
            runs["runner"] = _rope_outputs(
                (prepared.out, prepared.lse, prepared.o_lo), grads
            )
            windowed = any(
                isinstance(key, cake_backend.WindowLaunch)
                for key in (*prepared.forward_stages, *prepared.backward_order)
            )
            assert windowed == bridge, (
                prepared.forward_stages,
                prepared.backward_order,
            )
            if not bridge:
                assert tuple(prepared.forward_stages) == (forward,)
                assert main in prepared.backward_order
        torch.cuda.synchronize()
        return runs, prepared

    # two forced key passes over two head-window token chunks (one bridge chunk)
    big = make_inputs([2304], [4096], seed=SEED + 9700 + heads, topk=512, heads=heads)
    chunks = cake_backend.KeyPassPolicy.from_record(cake_jit.record()).token_chunks(
        2304, 512, cake_backend.HEAD_WINDOW_ITEMS
    )
    assert chunks == ((0, 2176), (2176, 128)), chunks
    found, _ = calls(big, bridge=False, passes=(2,), direct=False, runner=False)
    bridge, _ = calls(big, bridge=True, passes=(2,), direct=False, runner=False)
    _same_per_head_outputs(found, bridge)
    del big, found, bridge

    def case(s, j):
        inp = make_inputs([256], [1024], seed=s, topk=512, heads=heads)
        found, _ = calls(inp, bridge=False)
        bridge, _ = calls(inp, bridge=True)
        _same_per_head_outputs(found, bridge)
        monkeypatch.setattr(cake_backend, "head_window_stage", head_window_stage)
        return nv.judged(nv.rope_problem(inp, scale=SCALE), found)

    nv.judge_draws(record_property, case, seed=SEED + 9800 + heads)


@pytest.mark.parametrize("heads", (96,) + REDUCED_HEADS)
def test_single_pass_forward_through_the_loader(heads, monkeypatch, record_property):
    """Where the program ships the single-pass forward for the device (sm_100a / sm_103a; skip where it does not), the
    forward of a call over ``heads`` query heads is one launch of it through the package's loader -- the stage's
    module, its generated launcher and the persistent grid :func:`cake_backend.single_pass_grid` over the device's
    SMs -- for T = 1, a T below the device's cluster count and one above it, with and without ``o_lo``:

    * a prepared runner's forward is that one launch; its ``out`` / ``lse`` / ``o_lo`` repeat bit for bit run to run,
      equal bit for bit the same launch over the generated grid (one cluster per token: a token's result does not
      depend on the cluster that runs it) and the eager forward's; the ``with_o_lo=False`` build's ``out`` / ``lse``
      equal the default build's bit for bit;
    * without ``topk_length`` the forward derives the row lengths (``derive_topk_length=True`` writes them: equal to
      the reference's) and its outputs equal those of the call given the lengths, bit for bit;
    * an odd grid is refused (``ValueError``) by the stage's binding (the cluster launch) and by
      :func:`cake_backend._stage_entry`, before any launch.

    The forward outputs are judged at the naive tier over the draws (T = 300)."""
    _require_heads(heads)
    arch = cake_backend.arch_for(torch.device("cuda"))
    stage = cake_backend.single_pass_stage("fwd", arch)
    if stage is None:
        pytest.skip(
            "the registered program ships no single-pass forward for this device"
        )
    assert cake_backend.single_pass_stage("fwd_no_olo", arch) == "fwd_h128_no_olo"
    sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    clusters = sms // cake_backend.SINGLE_PASS_CLUSTER

    def fresh(t):
        return torch.full_like(t, float("nan")) if t.is_floating_point() else t

    def run_launch(runner, key, grid):
        launch = runner.launches[key]
        with cake_backend._ffi_stream_context(runner.device_index):
            launch.launcher(launch.run, launch.values, grid)

    for T in (1, min(5, clusters - 1), clusters + 37):
        inp = make_inputs(
            [T], [2048], seed=SEED + 9900 + T + heads, topk=512, heads=heads
        )
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        outs = {}
        for with_o_lo in (True, False):
            runner = prepare_dsa_train(*args, softmax_scale=SCALE, with_o_lo=with_o_lo)
            want_stage = "fwd_h128" if with_o_lo else "fwd_h128_no_olo"
            assert tuple(runner.forward_stages) == (want_stage,), runner.forward_stages
            launch = runner.launches[want_stage]
            grid = cake_backend.single_pass_grid(T, sms)
            assert launch.grid == grid == (2 * min(T, clusters), 1, 1), (
                launch.grid,
                grid,
            )
            assert (launch.values["num_queries"], launch.values["num_heads"]) == (
                T,
                heads,
            )
            names = ("out", "lse", "o_lo") if with_o_lo else ("out", "lse")
            got = []
            for _ in range(2):
                for name in names:
                    runner.tensors[name].copy_(fresh(runner.tensors[name]))
                first = runner.forward()
                torch.cuda.synchronize()
                got.append({n: first[i].clone() for i, n in enumerate(names)})
            for name in names:
                assert torch.equal(got[0][name], got[1][name]), (
                    f"T {T} {name}: not run-to-run bitwise"
                )
            # the generated grid: one cluster per token
            per_token = cake_launch.GRID[want_stage](T, inp.total_k, 512)
            assert per_token == (2 * max(1, T), 1, 1)
            for name in names:
                runner.tensors[name].copy_(fresh(runner.tensors[name]))
            run_launch(runner, want_stage, per_token)
            torch.cuda.synchronize()
            for name in names:
                assert torch.equal(runner.tensors[name], got[0][name]), (
                    f"T {T} {name}: the one-cluster-per-token grid differs from the persistent grid"
                )
            # odd grids: the binding refuses the cluster launch, the stage entry refuses the grid
            for odd in ((grid[0] + 1, 1, 1), (1, 1, 1)):
                with pytest.raises(ValueError, match="grid"):
                    run_launch(runner, want_stage, odd)
                with pytest.raises(
                    ValueError, match="not a multiple of the cluster shape"
                ):
                    cake_backend._stage_entry(
                        want_stage,
                        arch,
                        dict(num_queries=T, num_kv=inp.total_k, topk=512),
                        grid=odd,
                    )
            torch.cuda.synchronize()
            eager = cake_backend.forward(
                *args, softmax_scale=SCALE, with_o_lo=with_o_lo
            )
            for i, name in enumerate(names):
                assert torch.equal(eager[i], got[0][name]), (
                    f"T {T} {name}: eager != runner"
                )
            outs[with_o_lo] = got[0]
            del runner
        for name in ("out", "lse"):
            assert torch.equal(outs[True][name], outs[False][name]), (
                f"T {T} {name}: the with_o_lo=False build differs from the default build"
            )
        # derived lengths: written when asked, outputs equal to the given-length call
        lengths = torch.full((T,), -7, dtype=torch.int32, device="cuda")
        derived = cake_backend.forward(
            *args, topk_length=lengths, derive_topk_length=True, softmax_scale=SCALE
        )
        assert torch.equal(lengths, derive_topk_length(inp.idx_global, inp.total_k)), (
            f"T {T}: derived lengths"
        )
        given = cake_backend.forward(
            *args, topk_length=lengths.clone(), softmax_scale=SCALE
        )
        for i, name in enumerate(("out", "lse", "o_lo")):
            assert torch.equal(derived[i], given[i]), (
                f"T {T} {name}: derived != given lengths"
            )
            assert torch.equal(derived[i], outs[True][name]), (
                f"T {T} {name}: derived != no lengths"
            )

    def case(s, j):
        inp = make_inputs([300], [4096], seed=s, topk=512, heads=heads)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        out, lse, o_lo = cake_backend.forward(
            *args, softmax_scale=SCALE, with_o_lo=False
        )
        assert o_lo is None
        runs = {
            "forward": _rope_outputs(cake_backend.forward(*args, softmax_scale=SCALE)),
            # the judge reads an o_lo as the default mode's: the with_o_lo=False call is judged on out and lse
            "forward no_olo": dict(out=out, lse=lse),
        }
        torch.cuda.synchronize()
        return nv.judged(nv.rope_problem(inp, scale=SCALE), runs)

    nv.judge_draws(record_property, case, seed=SEED + 9950 + heads)


def test_sm107a_device_refuses_more_than_64_heads():
    """On an sm_107a device a 96-head call is refused before any launch (the windows ship on sm_100a / sm_103a
    first)."""
    if (
        not _device_supported()
        or cake_backend.arch_for(torch.device("cuda")) != "sm_107a"
    ):
        pytest.skip("an sm_107a device")
    inp = make_inputs([64], [512], seed=SEED, topk=128, heads=96)
    with pytest.raises(NotImplementedError, match="sm_107a serves 64 query heads only"):
        cake_backend.forward(
            inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global
        )
