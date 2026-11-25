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

The shipped kernels compute 64 query heads per launch.  A call with ``H`` = 96 or 128 heads runs them on two head
windows: ``[H - 64, H)`` first, then ``[0, 64)``.  The forward's second window rewrites the heads both cover; the
backward runs one delta launch over the ``T * H`` (token, head) rows, then the main stage per window into one shared
dK/dV accumulator, the first window with ``lse = +inf`` and ``delta = 0`` on the heads the second covers, so those
heads add exactly 0 there and the second window writes their ``dq``.  A 64-head call launches what it launched
before.

Host tests (CPU, no device): the launches of the eager entries and the prepared runner, recorded and simulated by
``test_helpers/cake_dsa_sparse_attention_host_launches.py`` -- the 64-head launch list against the golden capture,
the windows, their order, the window statistics and the accumulator at 64, 96 and 128 heads -- plus the retained
state of the autograd entry, validation and the calls without query rows.  Device tests judge every floating output
against the FP64 oracle at the naive BF16 tier (``tests/cake_naive.py``) on sm_100a / sm_103a (96 and 128 heads ship
there first; sm_107a runs the 64-head cases).  ``CAKE_PPLX_DSA_LARGE_TESTS=1`` opts in to the T = 16380 case
(:func:`test_large_t_heads`), which a default run skips: the GPU validation of this form sets it and reports that
case on its own.
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
    """The launches of one host case at ``heads`` query heads, simulated (``hl``): the forward runs the windows
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


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("with_o_lo", (True, False))
def test_autograd_entry_retains_o_lo_lse_and_lengths_per_head(
    monkeypatch, heads, with_o_lo
):
    """The autograd entry at ``heads`` query heads (host, simulated): the forward saves the nine tensors of its
    contract, of which the call's own state is ``lse [T, H]`` FP32, ``o_lo [T, H, 512]`` BF16 (with ``with_o_lo``;
    the zero-size placeholder without) and the derived ``topk_length [T]`` int32 -- exactly
    ``T * (H * (4 + 1024 with_o_lo) + 4)`` bytes beside the inputs and ``out``, no window scratch -- and its backward
    gives every query head its own ``dq`` (the backward launches as :func:`test_head_window_launches`)."""
    h = hl.HostLaunches(monkeypatch, cake_backend, cake_jit, cake_launch, "sm_100a")
    inp = hl.host_inputs(8, 64, 16, heads, seed=heads)
    leaves = [
        t.detach().clone().requires_grad_()
        for t in (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope)
    ]
    h.bind(leaves[0], leaves[1], inp.dout)
    saved = []

    def pack(t):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out, lse = cake_backend.cake_dsa_sparse_attention_nondet(
            *leaves, inp.indices, return_lse=True, with_o_lo=with_o_lo
        )
    assert len(saved) == 9
    T, H = inp.total_q, heads
    held = {
        t.untyped_storage().data_ptr() for t in (*leaves, inp.indices, out) if t.numel()
    }
    retained = sum(
        t.untyped_storage().nbytes()
        for t in saved
        if t.numel() and t.untyped_storage().data_ptr() not in held
    )
    assert retained == T * (H * (4 + (1024 if with_o_lo else 0)) + 4), retained
    fwd_launches = h.mark()
    hl.check_forward(
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
    hl.check_backward(
        h.launches[fwd_launches:],
        bwd,
        out.detach(),
        saved[7] if with_o_lo else None,
        lse,
        grads[0],
        grads[1],
    )


@pytest.mark.parametrize("heads", (64, 96, 128))
@pytest.mark.parametrize("with_o_lo", (True, False))
def test_every_rope_plan_carries_dout_copy(monkeypatch, heads, with_o_lo):
    """Every plan the backend builds for the rope form carries ``dout_copy`` (host): the eager forward and backward,
    a prepared runner and the flat and packed (varlen) autograd entries at ``heads`` query heads.  The bound values
    read the field with a ``False`` fallback for duck-typed plans, so a real plan must hold it: a ``_Plan`` field,
    ``True`` exactly for a backward over more than 64 heads whose ``dout`` is not dense over its rows (the
    head-major view)."""
    assert "dout_copy" in {f.name for f in dataclasses.fields(cake_backend._Plan)}
    h = hl.HostLaunches(monkeypatch, cake_backend, cake_jit, cake_launch, "sm_100a")
    real_plan = cake_backend._plan
    found = []
    call = {}

    def plan(*args, **kwargs):
        made = real_plan(*args, **kwargs)
        found.append((call["entry"], call["head_major"], made))
        return made

    monkeypatch.setattr(cake_backend, "_plan", plan)
    # the packed entry takes its torch route on the host (the index-preparation stage needs a device)
    monkeypatch.setattr(
        cake_backend, "nope_index_prep_available", lambda device=None: False
    )
    T, S, K = 8, 64, 16

    def start(entry, head_major=False):
        cake_backend.BINDING_CACHE._entries.clear()  # every entry plans afresh
        call.update(entry=entry, head_major=head_major)
        inp = hl.host_inputs(T, S, K, heads, seed=heads, head_major=head_major)
        h.bind(inp.q_latent, inp.q_rope, inp.dout)
        return inp

    for head_major in (False, True):
        for entry in ("forward", "backward", "runner"):
            inp = start(entry, head_major)
            hl.run_case(h, cake_backend, dict(entry=entry, with_o_lo=with_o_lo), inp)
    for entry in ("autograd", "varlen"):
        inp = start(entry)
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
    assert {entry for entry, _, _ in found} == {
        "forward",
        "backward",
        "runner",
        "autograd",
        "varlen",
    }
    for entry in ("autograd", "varlen"):
        assert any(made.backward for name, _, made in found if name == entry), entry
    for entry, head_major, made in found:
        assert made.num_heads == heads and made.with_o_lo == with_o_lo
        want = made.backward and heads != 64 and head_major
        assert isinstance(made.dout_copy, bool) and made.dout_copy == want, (
            entry,
            head_major,
            made.backward,
            made.dout_copy,
        )
    assert any(made.dout_copy for _, _, made in found) == (heads != 64)


@pytest.mark.parametrize("heads", (64, 96, 128))
def test_workspace_and_retained_accounting_heads(monkeypatch, heads):
    """The per-call scratch (workspace) and the state a step keeps from its forward to its backward (retained) are
    reported apart.  Workspace: a 64-head layout is the first release's, region for region; above 64 heads the
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
    h = hl.HostLaunches(
        monkeypatch, cake_backend, cake_jit, cake_launch, "sm_100a", token_chunk=chunk
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
    """On sm_107a the windows do not ship yet: a 96- or 128-head call is refused before any launch (host, the
    architecture stood in); 64 heads launch as before."""
    h = hl.HostLaunches(monkeypatch, cake_backend, cake_jit, cake_launch, "sm_107a")
    inp = hl.host_inputs(8, 64, 16, heads)
    h.bind(inp.q_latent, inp.q_rope, inp.dout)
    args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.indices)
    refused = "sm_107a serves 64 query heads only"
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.forward(*args)
    with pytest.raises(NotImplementedError, match=refused):
        cake_backend.backward(*args, inp.q_latent, -inp.q_latent, inp.lse, inp.dout)
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
    and 3 forced key passes (1 and 3 at 128 heads)."""
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

    nv.judge_draws(
        record_property, case, seed=SEED + 10 * heads + VALIDITY.index(validity)
    )


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


@pytest.mark.parametrize("heads", (96,) + REDUCED_HEADS)
def test_dkv_acc_repeated_dst_heads(with_o_lo, heads, record_property):
    """``dkv_acc`` through a many-to-few destination map (1024 key rows into 300 destination rows, repeated
    destinations summing as ``index_add_``) at ``heads`` query heads: the natural-layout main stage and the cast's
    accumulate path, one pass and two forced key passes, through the eager backward and through a prepared runner's
    step (two forced key passes: the window launches per pass reuse the runner's carved window statistics and dQ
    partial); each call's ``dq`` and buffer increment judged (unmapped destination rows and the columns past 576
    keep their bits), the runner's forward outputs too.  The same draw also judges the eager backward's natural FP32
    key gradients (``dkv_fp32=True``, one and two key passes)."""
    _require_heads(heads, backward=True)
    _require_key_passes_and_direct()

    def case(s, j):
        inp = make_inputs([256], [1024], seed=s, topk=512, heads=heads)
        args = (inp.q_latent, inp.q_rope, inp.kv_latent, inp.k_rope, inp.idx_global)
        fwd = cake_backend.forward(*args, softmax_scale=SCALE)
        out, lse, o_lo = fwd
        gen = torch.Generator(device="cuda").manual_seed(s + 3)
        dst_map = torch.randint(
            0, 300, (inp.total_k,), device="cuda", generator=gen
        ).to(torch.int32)
        runs = {"forward": _rope_outputs(fwd)}
        for direct in (True, False):
            for p in (1, 2):
                acc = torch.randn(300, 704, device="cuda", generator=gen)
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
            acc = torch.randn(300, 704, device="cuda", generator=gen)
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
            windows = {
                (key.head0, key.key[2])
                for key in runner.backward_order
                if isinstance(key, cake_backend.WindowLaunch)
                and isinstance(key.key, tuple)
            }
            assert windows == {(b, p) for b in (heads - 64, 0) for p in (0, 1)}, windows
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

    nv.judge_draws(record_property, case, seed=SEED + 3000 + heads)


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
