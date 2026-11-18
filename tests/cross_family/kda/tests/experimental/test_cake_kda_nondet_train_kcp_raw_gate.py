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

The raw gate mode of the non-deterministic context-parallel (KCP) KDA training route (``use_gate_in_kernel=True`` of
``cake_kcp.cake_kda_nondet_kcp`` / ``cake_kda_nondet_train_kcp_forward`` / ``cake_kda_nondet_train_kcp_backward``, issue
#7816), the mirror of ``test_cake_kda_det_train_kcp_raw_gate.py``.

CPU rows: the keywords and their refusals, the gate-aware workspace and its byte query (one context or a list of
contexts; the deterministic package's answer for the same partition) and their refusal of the lean local arena, the
saved set's metadata (the single-rank mismatch rule of both packages), the recompute handle's fingerprint, the
cross-rank reduction helper on one rank (fresh copies, its refusals), and the local stages' value dicts against the
raw records of the non-deterministic single-rank registry (the deterministic pair ``glm_intra_raw`` /
``param_finalize`` on every architecture).

Accuracy: every judged output is judged by its error against the exact FP64 reference beside the naive BF16 tier
(``tests/cake_naive.py``) over the judge's 32 draws, each path, mode or call under its own arm name.
This package has no bit contract: no output of one call or path is compared with another's, and no output with the
deterministic package's.

GPU rows at W = 1: four raw-mode packs on one reserved raw workspace and on one workspace per pack, each pack's
results on each form judged (the reserved bytes, the byte count and the explicit-mode refusal exact); under
``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) a raw backward raises no alert and gives no
such warning, eagerly and on a captured graph's replay, and its results are judged.

GPU rows (one process per rank; ``W = 1`` in the pytest process, ``W > 1`` spawned or one process per Slurm task as in
``test_cake_kda_nondet_train_kcp_dist.py``), each over the judge's draws in one process group: on every draw every rank
runs the raw-mode KCP forward / backward in every mode (saved, replay, repeat -- and on the first draw ``REPEATS``
more saved passes --, autograd, checkpoint, checkpoint with a recompute handle of both tiers, checkpoint with
``checkpoint_handoff="on"``, save_transfer, ``workspace=None``, a strided raw view, ring where it runs, graph replay);
rank 0 gathers each mode's outputs (``dA_log`` / ``ddt_bias`` summed over the ranks by the helper) and judges them
under the mode's name, and the single-rank raw path on the whole pack (arm ``single``), against the FP64 reference
of the unsplit pack.  On the first draw, exactly: a recompute handle's recompute issues no collective (the call's
counter and sequence, and no torch.distributed call in the recompute phase), the hand-off counters, the graph
events, the mode refusals, the explicit-gate path on the same activated gate runs the same stages in the same order
with the same collectives (``gate_epilogue`` -> ``param_finalize``), and the reduction helper's documented bits:
one call's result is the same on every rank, the FP32 rank-order sum of the partials.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import tempfile
import traceback
import types
import warnings
import zlib

import pytest
import torch
import torch.distributed as dist

from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed
from tests.experimental.test_cake_kda_det_train_raw_gate_accuracy import torch_gate
from tests.experimental.test_cake_kda_nondet_train_kcp_dist import (
    _Counter,
    _bwd_sequence_first_main,
    _handle_recompute_flags,
    _init_group,
    _local_slices,
    _partition,
    _programs_exported,
    _start_ranks,
    _to_global,
)
from tests.experimental.test_cake_kda_nondet_train_kcp_handoff import _handoff_delta
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    raw_gate,
)
from tests.experimental.test_helpers.cake_kda_known_gaps import settle
from tests.experimental.test_helpers.cake_kda_naive import (
    draw_case,
    judge_arms,
    naive_of,
    oracle_with_grads,
)
from tests.cake.test_helpers.cake_package import cake_package

cake_kcp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp")
cake_kcp_plan = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_plan")
cake_backend = cake_package("cake.cake_kda_nondet_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")
cake_kcp_jit = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_jit")
# the deterministic package: its byte queries, saved sets and gate rules, the mode contracts the CPU rows compare
# with this package's (imported by the test, never by the package; no output of it is run or compared here)
det_kcp = cake_package("cake.cake_kda_det_train_kcp", "cake_kcp")
det_kcp_plan = cake_package("cake.cake_kda_det_train_kcp", "cake_kcp_plan")
det_backend = cake_package("cake.cake_kda_det_train", "cake_backend")

D = cake_kcp.D
SCALE = cake_kcp.SCALE
LB = LOWER_BOUND  # -5.0
TOKEN_GRADS = ("dq", "dk", "dv", "dbeta", "dg")  # dg: the raw gate's gradient (BF16)
PARAM_GRADS = ("dA_log", "ddt_bias")  # this rank's partial sums
TENSORS = ("out",) + TOKEN_GRADS + PARAM_GRADS
TOKEN_NAMES = ("out",) + TOKEN_GRADS  # the outputs with a token axis
REPEATS = 20  # more saved passes of every row on the first draw (arm "repeat": the worst launch kept)
GRAPH_EVENTS = [
    ["eager", "eager"],
    ["capture", "eager"],
    ["replay", "capture"],
    ["replay", "replay"],
]


def _graph_events_ok(events: list) -> bool:
    """The forward follows GRAPH_EVENTS exactly.  The backward key of a rank holds its inputs and its saved tensors
    (``S_in``, ``c1_recv``, ``M_seg``, ``carry``).  When those addresses repeat on the second call, the backward
    captures one call early: at W = 1 the allocator may give the captured forward's ``S_in`` the freed eager
    address; a rank with one token keeps empty saved tensors (address 0).  The explicit mode does the same (the
    W8 ragged row, rank 6).  A capture replays at once, so the collectives stay aligned across ranks.  The rule is
    the single-rank rule of test_cake_kda_nondet_train_kcp: one capture, eager calls before it, replays after it."""
    if [e[0] for e in events] != [e[0] for e in GRAPH_EVENTS]:
        return False
    bwd = [e[1] for e in events]
    if bwd.count("capture") != 1:
        return False
    i = bwd.index("capture")
    return i >= 1 and set(bwd[:i]) == {"eager"} and set(bwd[i + 1 :]) == {"replay"}


# --------------------------------------------------------------------------- CPU rows


def _cpu_ctx(
    W: int = 2, rank: int = 0, lengths=(700, 1, 1300, 999), sm: int = 148, plan=None
):
    """A CPU context of this package (``plan=det_kcp_plan``: of the deterministic package, the same partition)."""
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if plan is det_kcp_plan:
        part = det_kcp_plan.cake_kda_det_train_kcp_partition(
            cu, W, layout="zigzag_per_document"
        )
        return det_kcp_plan.cake_kda_det_train_kcp_build_context(
            part, None, rank, "cpu", 4, sm_count=sm
        )
    part = cake_kcp_plan.cake_kda_nondet_train_kcp_partition(
        cu, W, layout="zigzag_per_document"
    )
    return cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, None, rank, "cpu", 4, sm_count=sm
    )


def _cpu_ops(ctx, H: int, *, raw: bool = True) -> dict:
    T = int(ctx.T_r)
    gen = torch.Generator().manual_seed(7816)
    ops = {n: torch.randn(T, H, D, generator=gen).to(torch.bfloat16) for n in "qkv"}
    ops["g"] = (
        torch.randn(T, H, D, generator=gen).to(torch.bfloat16)
        if raw
        else -torch.rand(T, H, D, generator=gen)
    )
    ops["beta"] = torch.rand(T, H, generator=gen).to(torch.bfloat16)
    return ops


def _gate_kw(H: int, device="cpu", lb=LB) -> dict:
    return {
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(H, dtype=torch.float32, device=device),
        "dt_bias": torch.zeros(H * D, dtype=torch.float32, device=device),
        "lower_bound": lb,
    }


def test_raw_keywords_are_checked_before_any_launch() -> None:
    """The raw gate keywords follow the single-rank rules (``cake_backend._gate_mode`` / ``_check_raw`` /
    ``_gate_params``) and are checked before the workspace is sized or anything launches: the parameters without
    ``use_gate_in_kernel``, a raw call without ``lower_bound`` or out of range, an FP32 raw gate, a non-bool mode,
    a workspace of the other mode in both directions, and a backend without local stages are refused."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    kw = _gate_kw(H)
    g_ops = _cpu_ops(ctx, H, raw=False)
    g_args = tuple(g_ops[n] for n in ("q", "k", "v", "g", "beta")) + (ctx,)
    with pytest.raises(ValueError, match="use_gate_in_kernel=True only"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(*g_args, A_log=kw["A_log"])
    with pytest.raises(ValueError, match="use_gate_in_kernel=True only"):
        cake_kcp.cake_kda_nondet_train_kcp_backward(*g_args, g_ops["q"], lower_bound=LB)
    with pytest.raises(ValueError, match="lower_bound"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(*args, **{**kw, "lower_bound": None})
    for bad in (0.0, -5.5, float("nan"), True):
        with pytest.raises(ValueError, match="lower_bound"):
            cake_kcp.cake_kda_nondet_train_kcp_forward(
                *args, **{**kw, "lower_bound": bad}
            )
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(
            *args, **{**kw, "use_gate_in_kernel": 1}
        )
    fp32 = (*args[:3], ops["g"].float(), *args[4:])
    with pytest.raises(ValueError, match="torch.bfloat16"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(*fp32, **kw)
    # an overlapping view is refused (the strided-pointer bindings check no layout of their own)
    over = ops["g"].as_strided(ops["g"].shape, (8, 8, 1))
    with pytest.raises(ValueError, match="strides"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(*args[:3], over, *args[4:], **kw)
    with pytest.raises(ValueError, match="A_log"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(
            *args, **{**kw, "A_log": torch.zeros(H + 1)}
        )
    with pytest.raises(ValueError, match="dt_bias"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(
            *args, **{**kw, "dt_bias": torch.zeros(H, D - 1)}
        )
    # the workspace's mode: a g workspace refuses a raw call and a raw workspace a g call (both directions)
    ws_g = cake_kcp.CakeKDANondetKcpWorkspace()
    ws_r = cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel=True)
    assert (ws_g.gate, ws_r.gate) == ("g", "raw")
    with pytest.raises(ValueError, match="use_gate_in_kernel=True"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(*args, workspace=ws_g, **kw)
    with pytest.raises(ValueError, match="use_gate_in_kernel=False"):
        cake_kcp.cake_kda_nondet_train_kcp_forward(*g_args, workspace=ws_r)
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel="raw")
    # a backend without the local kernel stages (an emulation) has no raw gate
    stub = type("Stub", (cake_kcp.Backend,), {"name": "stub", "local_stages": False})()
    prev = cake_kcp.set_backend(stub)
    try:
        with pytest.raises(NotImplementedError, match="local kernel stages"):
            cake_kcp.cake_kda_nondet_train_kcp_forward(*args, **kw)
    finally:
        cake_kcp.set_backend(prev)


def test_raw_saved_set_metadata_round_trips() -> None:
    """The saved set records the gate mode and the FP32 lower bound as metadata (same tensors, same version); the
    autograd node's ``meta`` / ``from_tensors`` round trip keeps them; the single-rank mismatch rule refuses the
    other mode and another lower bound.  The rule and the field names are the deterministic package's: each
    package's rule reads the other package's KCP saved set alike (the KCP backward still reads its own package's
    set only)."""
    t = torch.zeros(4)
    base = dict(
        version=cake_kcp.KCP_SAVED_VERSION, S_in=t, c1_recv=None, M_seg=None, T_r=1, H=1,
        n_frag=1, P=1, Hg=1, scale=0.5, save_mode="light", save_transfer=False,
        offset_class="fla", exchange="a2a", precision_mode=0, ctx=None,
    )  # fmt: skip
    g = cake_kcp.CakeKDANondetKcpSaved(**base)
    raw = cake_kcp.CakeKDANondetKcpSaved(**base, gate="raw", lower_bound=-5.0)
    assert (g.gate, g.lower_bound) == ("g", None)
    assert raw.tensors() == g.tensors() and raw.flags() == g.flags()
    back = cake_kcp.CakeKDANondetKcpSaved.from_tensors(
        raw.tensors(), raw.flags(), raw.meta(), None
    )
    assert back.meta() == raw.meta() and (back.gate, back.lower_bound) == (
        "raw",
        -5.0,
    )
    gp = cake_backend._gate_params(
        "raw", torch.zeros(1), torch.zeros(D), -5.0, 1, torch.device("cpu")
    )
    cake_backend._check_saved_gate(raw, gp)
    cake_backend._check_saved_gate(g, None)
    with pytest.raises(ValueError, match="gate mode"):
        cake_backend._check_saved_gate(raw, None)
    with pytest.raises(ValueError, match="gate mode"):
        cake_backend._check_saved_gate(g, gp)
    other = cake_backend._gate_params(
        "raw", torch.zeros(1), torch.zeros(D), -4.0, 1, torch.device("cpu")
    )
    with pytest.raises(ValueError, match="lower_bound"):
        cake_backend._check_saved_gate(raw, other)
    det_raw = det_kcp.CakeKDADetKcpSaved(**base, gate="raw", lower_bound=-5.0)
    det_g = det_kcp.CakeKDADetKcpSaved(**base)
    assert det_raw.meta() == raw.meta() and det_g.meta() == g.meta()
    det_gp = det_backend._gate_params(
        "raw", torch.zeros(1), torch.zeros(D), -5.0, 1, torch.device("cpu")
    )
    det_other = det_backend._gate_params(
        "raw", torch.zeros(1), torch.zeros(D), -4.0, 1, torch.device("cpu")
    )
    for rule, own_gp, own_other in (
        (cake_backend._check_saved_gate, gp, other),
        (det_backend._check_saved_gate, det_gp, det_other),
    ):
        for saved_raw, saved_g in ((raw, g), (det_raw, det_g)):
            rule(saved_raw, own_gp)
            rule(saved_g, None)
            with pytest.raises(ValueError, match="gate mode"):
                rule(saved_raw, None)
            with pytest.raises(ValueError, match="gate mode"):
                rule(saved_g, own_gp)
            with pytest.raises(ValueError, match="lower_bound"):
                rule(saved_raw, own_other)


def test_raw_workspace_bytes_and_local_plan() -> None:
    """``cake_kda_nondet_train_kcp_workspace_bytes(..., use_gate_in_kernel=True)`` is the raw plan's local arena (the existing backend's
    ``workspace_bytes(gate="raw")`` plus the scratch table set of the raw plan's intra record) plus the same KCP
    pools as the explicit mode; the raw arena is never larger (the gate epilogue's ``dg_int`` / ``db_int`` leave
    the backward layouts); the forward-only workspace does not depend on the mode; every answer is the deterministic
    package's for the same partition."""

    def local(ctx, H: int, backward: bool, gate: str) -> int:
        arena = cake_backend.workspace_bytes(
            ctx.T_r, ctx.n_frag, H, backward=backward, save_mode="light", sm_count=ctx.sm_count,
            fwd_layout=cake_kcp_plan.local_fwd_layout(backward, "light"), gate=gate,
        )  # fmt: skip
        plan = cake_backend.plan(ctx.T_r, ctx.n_frag, H, ctx.sm_count, gate=gate)
        if cake_kcp_plan.intra_writes_tables(plan):
            arena += cake_backend.tables_bytes(
                ctx.T_r, ctx.n_frag, H, sm_count=ctx.sm_count
            )
        return int(arena)

    for H in (4, 64):
        for rank in (0, 1):
            ctx = _cpu_ctx(rank=rank)
            for backward in (True, False):
                g = cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                    ctx, H=H, backward=backward
                )
                r = cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                    ctx, H=H, backward=backward, use_gate_in_kernel=True
                )
                lr, lg = local(ctx, H, backward, "raw"), local(ctx, H, backward, "g")
                assert r - lr == g - lg > 0, (H, rank, backward)  # the same KCP pools
                assert lr <= lg and r <= g, (H, rank, backward, lr, lg)
                if not backward:
                    assert r == g, "the forward layouts do not depend on the gate mode"
                det_ctx = _cpu_ctx(rank=rank, plan=det_kcp_plan)
                for gate_on, mine in ((True, r), (False, g)):
                    assert mine == det_kcp_plan.cake_kda_det_train_kcp_workspace_bytes(
                        det_ctx, H=H, backward=backward, use_gate_in_kernel=gate_on
                    ), (H, rank, backward, gate_on)
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
            _cpu_ctx(), H=4, use_gate_in_kernel="raw"
        )


def test_raw_workspace_refuses_the_lean_arena() -> None:
    """The raw gate mode runs on the replay local arena: ``CakeKDANondetKcpWorkspace`` and ``cake_kda_nondet_train_kcp_workspace_bytes``
    (backward and forward-only) refuse ``local_arena="lean"`` with ``use_gate_in_kernel=True``."""
    with pytest.raises(NotImplementedError, match='local_arena="lean"'):
        cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel=True, local_arena="lean")
    ctx = _cpu_ctx()
    for backward in (True, False):
        with pytest.raises(NotImplementedError, match='local_arena="lean"'):
            cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                ctx, H=4, backward=backward, use_gate_in_kernel=True, local_arena="lean"
            )


SHARED_PACKS = ((1000, 577), (50, 640, 77, 1000, 129), (1450,), (300, 64, 700))


def test_raw_workspace_bytes_of_a_list_of_contexts() -> None:
    """A list of contexts in raw mode: the largest raw local arena, the raw plans' scratch table sets of every
    distinct ``(T_r, n_frag)`` and the same KCP pools as the explicit mode (the pools do not depend on the mode);
    a list of one context gives the bytes of the context; the raw answer is the deterministic package's for the same
    partitions."""

    def local(ctxs, H: int, backward: bool, gate: str) -> int:
        arena, scratch = 0, {}
        for c in ctxs:
            arena = max(arena, int(cake_backend.workspace_bytes(
                c.T_r, c.n_frag, H, backward=backward, save_mode="light", sm_count=c.sm_count,
                fwd_layout=cake_kcp_plan.local_fwd_layout(backward, "light"), gate=gate,
            )))  # fmt: skip
            plan = cake_backend.plan(c.T_r, c.n_frag, H, c.sm_count, gate=gate)
            if cake_kcp_plan.intra_writes_tables(plan):
                scratch[(c.T_r, c.n_frag)] = int(
                    cake_backend.tables_bytes(c.T_r, c.n_frag, H, sm_count=c.sm_count)
                )
        return arena + sum(scratch.values())

    ctxs = [_cpu_ctx(W=1, lengths=lengths) for lengths in SHARED_PACKS]
    det_ctxs = [
        _cpu_ctx(W=1, lengths=lengths, plan=det_kcp_plan) for lengths in SHARED_PACKS
    ]
    for H in (4, 64):
        for backward in (True, False):
            g = cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                ctxs, H=H, backward=backward
            )
            r = cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                ctxs, H=H, backward=backward, use_gate_in_kernel=True
            )
            lr, lg = local(ctxs, H, backward, "raw"), local(ctxs, H, backward, "g")
            assert r - lr == g - lg > 0, (H, backward)  # the same KCP pools
            assert r <= g, (H, backward, r, g)
            assert r == det_kcp_plan.cake_kda_det_train_kcp_workspace_bytes(
                det_ctxs, H=H, backward=backward, use_gate_in_kernel=True
            ), (H, backward)
            for c in ctxs:
                assert cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                    [c], H=H, backward=backward, use_gate_in_kernel=True
                ) == cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                    c, H=H, backward=backward, use_gate_in_kernel=True
                )


def _shared_schedule(ctxs: list, ops: list, wss: list) -> list[dict]:
    """The packs ``ops`` interleaved on the workspaces ``wss`` (one per pack, or one shared): the saving forwards in
    pack order, their backwards in reverse order, then the replay-mode forwards and backwards in reverse order.  Per
    pack its ``"saved"`` and ``"replay"`` results by name, as they stand after the whole schedule."""
    names = TOKEN_GRADS + PARAM_GRADS
    n, res, saved = len(ctxs), [{} for _ in ctxs], [None] * len(ctxs)
    for i in range(n):
        o = ops[i]
        out, saved[i] = cake_kcp.cake_kda_nondet_train_kcp_forward(
            o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], workspace=wss[i], **o["GK"]
        )  # fmt: skip
        res[i]["saved"] = {"out": out}
    for i in reversed(range(n)):
        o = ops[i]
        grads = cake_kcp.cake_kda_nondet_train_kcp_backward(
            o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], o["dout"], saved=saved[i],
            workspace=wss[i], **o["GK"],
        )  # fmt: skip
        res[i]["saved"].update(zip(names, grads, strict=True))
    for i in reversed(range(n)):
        o = ops[i]
        out, none = cake_kcp.cake_kda_nondet_train_kcp_forward(
            o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], workspace=wss[i],
            boundary_mode="replay", **o["GK"],
        )  # fmt: skip
        assert none is None
        grads = cake_kcp.cake_kda_nondet_train_kcp_backward(
            o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], o["dout"], saved=None,
            workspace=wss[i], **o["GK"],
        )  # fmt: skip
        res[i]["replay"] = {"out": out, **dict(zip(names, grads, strict=True))}
    return res


def _cloned(res: list[dict]) -> list[dict]:
    """A copy of :func:`_shared_schedule`'s results taken when the schedule ends (the next schedule on a shared
    workspace may reuse its storage)."""
    return [
        {form: {n: t.clone() for n, t in out.items()} for form, out in pack.items()}
        for pack in res
    ]


def _shared_draw(H: int, ctxs: list, parts: list, device, p: int):
    """``draw(j, s)`` of pack ``p``'s judge in
    :func:`test_gpu_raw_reserved_shared_workspace_and_one_workspace_per_pack_are_judged`: every pack's operands drawn
    from the draw's seed, the schedule on one raw workspace per pack, then twice on one reserved raw workspace; pack
    ``p``'s results as the arms and its operands as the draw's inputs.  On the first draw, exactly: ``reserve``
    returns the packs' byte query, each round leaves the byte count, and an explicit-mode call on the raw workspace
    is refused."""

    def draw(j: int, s: int):
        globs, ops = [], []
        for i, ((part, cu), lengths) in enumerate(
            zip(parts, SHARED_PACKS, strict=True)
        ):
            glob = draw_case(lengths, H, "raw", False, s + 101 * i, device)
            o = _local_slices(
                part, 0, cu, {n: glob[n] for n in ("q", "k", "v", "g", "beta", "dout")}
            )
            o["GK"] = dict(
                use_gate_in_kernel=True,
                A_log=glob["A_log"],
                dt_bias=glob["dt_bias"],
                lower_bound=LB,
            )
            globs.append(glob)
            ops.append(o)
        own = _cloned(
            _shared_schedule(
                ctxs,
                ops,
                [
                    cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel=True)
                    for _ in ctxs
                ],
            )
        )
        ws = cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel=True)
        got = ws.reserve(ctxs, H)
        if j == 0:
            assert (
                got
                == ws.bytes
                == cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
                    ctxs, H=H, use_gate_in_kernel=True
                )
            ), H
        rounds = []
        for rnd in range(2):
            rounds.append(_cloned(_shared_schedule(ctxs, ops, [ws] * len(ctxs))))
            if j == 0:
                assert ws.bytes == got, (H, rnd)
        if j == 0:
            o = ops[0]
            with pytest.raises(ValueError, match="gate mode"):
                cake_kcp.cake_kda_nondet_train_kcp_forward(
                    o["q"], o["k"], o["v"], torch_gate(o["g"], o["GK"]["A_log"], o["GK"]["dt_bias"]),
                    o["beta"], ctxs[0], workspace=ws,
                )  # fmt: skip
        part, cu = parts[p]

        def in_global_order(res: dict) -> dict:
            return {
                n: _to_global(part, [t], cu, cu[-1]) if n in TOKEN_NAMES else t
                for n, t in res.items()
            }

        arm = f"h{H}_p{p}"
        arms = {
            arm: in_global_order(own[p]["saved"]),
            f"{arm}_replay": in_global_order(own[p]["replay"]),
            f"{arm}_shared": [in_global_order(r[p]["saved"]) for r in rounds],
            f"{arm}_shared_replay": [in_global_order(r[p]["replay"]) for r in rounds],
        }
        return arms, globs[p]

    return draw


def test_gpu_raw_reserved_shared_workspace_and_one_workspace_per_pack_are_judged(
    record_property,
) -> None:
    """Four raw-mode packs of one rank (different ``T_r`` / ``n_frag``) at H = 4 and 64, interleaved (saving forwards
    in pack order, their backwards in reverse order, replay-mode forwards and backwards) on one raw workspace per
    pack and, twice, on ONE reserved raw workspace.  Each pack's results on each workspace form are judged against
    the FP64 reference beside the naive BF16 tier over the judge's draws, one judge per pack and head count (arms
    ``h<H>_p<i>`` and ``h<H>_p<i>_replay`` on the pack's own workspace, ``..._shared`` and ``..._shared_replay`` on
    the shared one, its two rounds judged as repeats: the worst kept; every output, the raw gate's gradient and the
    parameter partial sums); no form is compared with another.  On the first draw, exactly: ``reserve`` returns
    ``cake_kda_nondet_train_kcp_workspace_bytes(ctxs, H=H, use_gate_in_kernel=True)``, the schedule changes no byte
    count, and an explicit-mode call on the raw workspace is refused."""
    _skip_unless_runnable(1)
    device = torch.device("cuda", torch.cuda.current_device())
    for H in (4, 64):
        parts, ctxs = [], []
        for lengths in SHARED_PACKS:
            part, cu = _partition(lengths, 1, "contiguous")
            parts.append((part, cu))
            ctxs.append(
                cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
                    part, None, 0, device
                )
            )
        for p, lengths in enumerate(SHARED_PACKS):
            judge_arms(
                record_property,
                _shared_draw(H, ctxs, parts, device, p),
                seed=_row_seed("shared_ws", (lengths, H, 1, "contiguous")),
                scale=SCALE,
                lower_bound=LB,
            )
        del ctxs, parts
        gc.collect()


def test_raw_handle_fingerprint_records_the_gate() -> None:
    """A recompute handle's fingerprint carries the gate mode and the FP32 lower bound (``modes``) and lists only
    tensors under ``operands`` (the raw mode's strides and lower bound are host scalars): a raw entry never binds a
    recompute in the explicit mode or with another lower bound."""
    H = 4
    ctx = _cpu_ctx()
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
    ops = _cpu_ops(ctx, H)

    def call(gate: str, lb: float | None):
        c = types.SimpleNamespace(
            tb=tb, T_r=int(ctx.T_r), H=H, scale=0.5, exchange="a2a",
            opts=cake_kcp._options({}), gate=gate, gate_params=None, ops=dict(ops),
        )  # fmt: skip
        if gate == "raw":
            c.gate_params = cake_backend._gate_params(
                "raw", torch.zeros(H), torch.zeros(H * D), lb, H, torch.device("cpu")
            )
            g = c.ops.pop("g")
            c.ops.update(cake_backend._raw_values(g, c.gate_params))
        return c

    def fp(c):
        return dict(
            cake_kcp._handle_fingerprint(
                c, save_mode="light", keep="states", **cake_kcp._gate_modes(c)
            )
        )

    f_g, f_r, f_r4 = fp(call("g", None)), fp(call("raw", -5.0)), fp(call("raw", -4.0))
    assert dict(f_g["modes"]) == {"use_gate_in_kernel": False, "lower_bound": None}
    assert dict(f_r["modes"]) == {"use_gate_in_kernel": True, "lower_bound": -5.0}
    assert f_r["modes"] != f_r4["modes"] and f_r["modes"] != f_g["modes"]
    names = [n for n, *_ in f_r["operands"]]
    assert names == ["q", "k", "v", "beta", "g_raw", "A_log", "dt_bias"], names
    assert [n for n, *_ in f_g["operands"]] == ["q", "k", "v", "g", "beta"]


def test_reduce_gate_param_grads_one_rank_and_checks() -> None:
    """On one rank the helper returns fresh copies of the inputs in their shapes (byte copies); it refuses other
    dtypes and shapes, and several ranks without an initialized process group (it never returns a partial sum)."""
    ctx = types.SimpleNamespace(world_size=1, group=None)
    a = torch.arange(4, dtype=torch.float32)
    for b in (torch.ones(4 * D), torch.ones(4, D)):
        ra, rb = cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(ctx, a, b)
        assert torch.equal(ra, a) and torch.equal(rb, b) and rb.shape == b.shape
        assert ra.data_ptr() != a.data_ptr() and rb.data_ptr() != b.data_ptr()
    with pytest.raises(ValueError, match="float32"):
        cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(
            ctx, a.double(), torch.ones(4 * D)
        )
    with pytest.raises(ValueError, match="ddt_bias"):
        cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(
            ctx, a, torch.ones(4 * D + 1)
        )
    with pytest.raises(RuntimeError, match="process group"):
        cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(
            types.SimpleNamespace(world_size=2, group=None), a, torch.ones(4 * D)
        )


LOCAL_RECORD_CASES = [
    (4, "sm_100a", 148),
    (64, "sm_100a", 148),
    (4, "sm_103a", 152),
    (64, "sm_103a", 152),
    (64, "sm_107a", 216),
]


@pytest.mark.parametrize(
    "H,arch,sm",
    LOCAL_RECORD_CASES,
    ids=[f"h{h}-{a}-{s}" for h, a, s in LOCAL_RECORD_CASES],
)
def test_raw_local_stage_values_cover_the_records(H, arch, sm) -> None:
    """The kernel backend's local stages on a raw plan (``cake_backend.plan(..., gate="raw")``): the forward, a
    saved-mode and a replay backward bind every logical name of every record they launch (the raw intra / prep /
    GLM intra records read ``g_raw``, its strides, ``A_log``, ``dt_bias``, ``lower_bound``; the GLM intra program
    writes ``dg_raw``; no gate epilogue launches).  The parameter-gradient pair, the deterministic package's on every
    architecture: ``glm_intra_raw`` writes the partials ``dA_part`` / ``db_part`` and ``param_finalize`` reads
    ``seq_chunk_start`` bounded at ``num_seqs`` = the rank's fragments and writes ``dA_log`` / ``ddt_bias`` on its
    ``(H, seg_cap, 1)`` grid with the per-call ``part2`` / ``ticket`` scratch.  Every stage outside
    ``cake_jit.GATE_STAGES`` launches exactly the explicit plan's record on the same grid, and the raw records
    replace the explicit ones stage for stage."""
    ctx = _cpu_ctx(sm=sm)
    assert ctx.T_r > 0
    plans = {
        gate: cake_backend.plan(ctx.T_r, ctx.n_frag, H, sm, arch, gate=gate)
        for gate in ("g", "raw")
    }
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
    launches: list = []

    class Arena:
        def __init__(self, plan) -> None:
            self.plan = plan
            self.device = torch.device("cpu")
            self.device_index = 0
            self.arch = arch
            self.launcher = self._launch
            self.fwd_layout = cake_kcp.LOCAL_LAYOUT_FWD
            self._views: dict = {}

        def views(self, mode: str):
            ns = self._views.get(mode)
            if ns is None:
                ns = types.SimpleNamespace()
                for r in self.plan.layout(mode):
                    setattr(
                        ns,
                        r.name,
                        torch.zeros(2, D) if r.name in ("h", "dh") else torch.zeros(1),
                    )
                self._views[mode] = ns
            return ns

        def _launch(self, record, values, *, grid, device_index) -> None:
            launches.append(("backend", record, grid, dict(values)))

    class Provider:
        def run_record(self, record, values, *, grid, device_index) -> None:
            launches.append(("kcp", record, grid, dict(values)))

    backend = cake_kcp.KernelBackend(programs=Provider())
    T = int(ctx.T_r)
    raw_g = torch.zeros(T, H, D, dtype=torch.bfloat16)
    gp = cake_backend._gate_params(
        "raw", torch.zeros(H), torch.zeros(H * D), LB, H, torch.device("cpu")
    )
    ops = {
        "g": {n: torch.zeros(1) for n in ("q", "k", "v", "g", "beta")},
        "raw": {
            **{n: torch.zeros(1) for n in ("q", "k", "v", "beta")},
            **cake_backend._raw_values(raw_g, gp),
        },
    }
    # param_finalize's per-call scratch, as the raw backward of a rank with tokens allocates it
    fin_scratch = cake_backend._raw_finalize_scratch(
        int(plans["raw"].NC_pad), H, torch.device("cpu")
    )
    seg_cap = int(fin_scratch["seg_cap"])
    grads = {
        "g": {n: torch.zeros(1) for n in ("dq", "dk", "dv", "dg", "dbeta")},
        "raw": {
            **{
                n: torch.zeros(1)
                for n in ("dq", "dk", "dv", "dbeta", "dg_raw", "dA_part", "db_part", "dA_log", "ddt_bias")
            },
            "part2": fin_scratch["part2"],
            "ticket": fin_scratch["ticket"],
        },
    }  # fmt: skip

    def run(gate: str, backward: bool, saved) -> list:
        call = types.SimpleNamespace(
            local=Arena(plans[gate]), ctx=ctx, tb=tb, T_r=T, H=H, backward=backward, host_trim=1,
            ops={**ops[gate], **({"dout": torch.zeros(1)} if backward else {})}, scale=0.5, scratch={},
            local_values=None, out=None if backward else torch.zeros(1), grads=grads[gate] if backward else None,
            device=torch.device("cpu"), opts=cake_kcp._options({"spec3_probe": "off"}),
        )  # fmt: skip
        backend.begin_call(call)
        if backward:
            stages = [("bwd_prep", {"saved": saved}), ("fwd_intra", {}), ("wy", {}), ("dav", {}),
                      ("dqkg", {}), ("glm_intra", {})]  # fmt: skip
            stages.append(
                ("param_finalize", {"seg_cap": seg_cap})
                if gate == "raw"
                else ("gate_epilogue", {})
            )
        else:
            stages = [("fwd_intra", {}), ("wy_fwd", {}), ("fwd_o", {})]
        for name, extra in stages:
            backend.stage(name, call, **extra)
        done = list(launches)
        launches.clear()
        return done

    saved = types.SimpleNamespace(S_in=torch.zeros(1), c1_recv=None, M_seg=None)
    res = {
        (gate, kind): run(gate, kind != "fwd", saved if kind == "bwd" else None)
        for gate in ("g", "raw")
        for kind in ("fwd", "bwd", "replay")
    }
    pr = plans["raw"]
    b = cake_kcp.LOCAL_PATH_BWD
    for kind in ("fwd", "bwd", "replay"):
        for src, record, _grid, values in res[("raw", kind)]:
            mod = (cake_jit if src == "backend" else cake_kcp_jit).MODULES[record]
            missing = set(mod["logical"].values()) - set(values)
            assert not missing, (kind, record, sorted(missing))
            assert "g" not in values, (kind, record)  # no FP32 gate on the raw route
            if src == "backend":
                assert (
                    mod.get("gate", "g") == "raw"
                    or mod["stage"] not in cake_jit.GATE_STAGES
                )
    assert [(k, r, g) for k, r, g, _ in res[("raw", "bwd")]] == [
        (k, r, g) for k, r, g, _ in res[("raw", "replay")]
    ]
    # stage for stage: the same records outside the gate stages, the raw class inside them
    for kind in ("fwd", "bwd"):
        g_l = [
            (k, r, gr)
            for k, r, gr, _ in res[("g", kind)]
            if not (k == "backend" and cake_jit.MODULES[r]["stage"] == "gate_epilogue")
        ]
        r_l = [
            (k, r, gr)
            for k, r, gr, _ in res[("raw", kind)]
            if not (k == "backend" and cake_jit.MODULES[r]["stage"] == "param_finalize")
        ]
        assert len(g_l) == len(r_l), (kind, g_l, r_l)
        for (kg, rg, gg), (kr, rr, gr) in zip(g_l, r_l, strict=True):
            stage = cake_jit.MODULES[rg]["stage"] if kg == "backend" else rg
            if kg == "backend" and stage in cake_jit.GATE_STAGES:
                assert cake_jit.MODULES[rr]["stage"] == stage and gr == gg, (rg, rr)
                assert cake_jit.MODULES[rr].get("gate") == "raw", rr
            else:
                assert (kr, rr, gr) == (kg, rg, gg), (kind, rg, rr)
    bwd = res[("raw", "bwd")]
    records = [r for _, r, _, _ in bwd]
    assert records[-2:] == [
        pr.record("glm_intra", b),
        pr.record("param_finalize", b),
    ], records
    assert records[-2:] == ["glm_intra_raw", "param_finalize"], records
    assert records[0] == pr.record("bwd_prep", b) == "bwd_prep_g8_replay_raw"
    assert not any(
        cake_jit.MODULES[r]["stage"] == "gate_epilogue"
        for _, r, _, _ in bwd
        if r in cake_jit.MODULES
    )
    fin = bwd[-1][3]
    assert (
        fin["dA_log"] is grads["raw"]["dA_log"]
        and fin["ddt_bias"] is grads["raw"]["ddt_bias"]
    )
    gi = bwd[-2][3]
    assert bwd[-1][2] == pr.grid("param_finalize", b) == (H, seg_cap, 1)
    assert (
        fin["num_seqs"] == ctx.n_frag
        and fin["seq_chunk_start"] is ctx.tables["seq_chunk_start"]
    )
    assert fin["dA_part"] is grads["raw"]["dA_part"]
    assert gi["dA_part"] is grads["raw"]["dA_part"]
    assert gi["db_part"] is grads["raw"]["db_part"]
    assert (
        fin["part2"] is fin_scratch["part2"]
        and fin["ticket"] is fin_scratch["ticket"]
        and fin["seg_cap"] == seg_cap
    )
    assert (
        gi["g_raw"] is raw_g
        and gi["lower_bound"] == LB
        and gi["dg_raw"] is grads["raw"]["dg_raw"]
    )
    assert (gi["g_raw_row_stride"], gi["g_raw_head_stride"]) == (H * D, D)
    fwd = res[("raw", "fwd")]
    assert fwd[0][1] == pr.record("fwd_intra", cake_kcp.LOCAL_PATH_FWD)
    assert cake_jit.MODULES[fwd[0][1]].get("gate") == "raw"
    assert (
        fwd[0][3]["write_ext"],
        fwd[0][3]["write_int"],
        fwd[0][3]["write_prep"],
    ) == (0, 1, 1)


# --------------------------------------------------------------------------- GPU rows (one process per rank)

# (lengths, H, W, layout): W = 1 (in-process), CP2 / CP4 (one node), CP8 (two nodes, Slurm per-task); boundaries
# inside chunks, a document over >= 3 ranks (zigzag, per-document zigzag), one-token documents and fragments,
# explicit ragged tables with a zero-token rank and a one-token rank, a 64-head row.
ROWS = [
    ((100, 640, 77, 1000, 129), 4, 1, "contiguous"),
    ((700, 1, 1300, 999), 64, 1, "contiguous"),
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((3000,), 4, 2, "zigzag_per_document"),
    ((100, 640, 77, 1000, 129), 4, 4, "contiguous"),
    ((3000,), 4, 4, "zigzag"),
    ((3000,), 4, 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"),
    ((5000, 1003), 1, 4, "ragged:1,2,4997,1003"),
    ((700, 1, 1300, 999), 64, 4, "zigzag_per_document"),
    ((3000, 2048, 5, 1), 4, 8, "zigzag_per_document"),
    ((4000, 1003), 4, 8, "ragged:1,0,999,1000,1000,1000,1,1002"),
    ((700, 1, 1300, 999), 64, 8, "zigzag"),
]


def _row_id(row) -> str:
    lengths, H, W, layout = row
    return "s" + "_".join(map(str, lengths)) + f"-h{H}-w{W}-{layout.split(':')[0]}"


def _collective_seq(rec: dict) -> list:
    return [s for s in rec["sequence"] if ":" in s]


def _row_seed(test: str, row) -> int:
    """The base seed of a judged row (draw ``j`` draws every input from ``draw_seed(base, j)``): a CRC of the test's
    name and the row."""
    return zlib.crc32(f"{test}:{_row_id(row)}".encode())


def _like(t: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """``t`` in ``actual``'s shape where only the shape differs (``ddt_bias`` ``[H * 128]`` or ``[H, 128]``)."""
    if t.shape != actual.shape and t.numel() == actual.numel():
        return t.reshape(actual.shape)
    return t


def _judge_draw(
    judge, failures: list, j: int, arms: dict, ref: dict, naive: dict, device
) -> None:
    """Draw ``j`` of every arm (``{arm: {output: tensor}}``, an output judged as ``"<arm>.<output>"``): each output
    held to its contract against the FP64 reference ``ref`` and added to ``judge`` beside the naive tier ``naive``
    (an arm added again on the same draw keeps its worst launch)."""
    for arm, results in arms.items():
        for name, actual in results.items():
            actual = actual.to(device)
            key = f"{arm}.{name}"
            want = _like(ref[name], actual)
            if message := contract(key, actual, want):
                failures.append(f"draw {j}: {message}")
                if actual.shape != want.shape:
                    continue
            judge.add(key, j, actual, want, {"bf16": _like(naive[name], actual)})


def _finish(record_property, report: dict, lengths) -> None:
    """Record rank 0's verdicts (``naive_tier``), hold every rank's exact checks (``pass``), then settle rank 0's
    verdicts against the owner-accepted gaps (the row's documents are the gaps' scope).  Another rank of a
    one-process-per-task run holds no verdicts."""
    verdicts = report.get("verdicts") or {}
    for verdict in verdicts.values():
        record_property("naive_tier", verdict)
    assert report["pass"], {
        k: report.get(k) for k in ("row", "flags", "reduce_eq_rank_order_sum")
    }
    if report["rank"] == 0:
        assert verdicts, report.get("failures")
        settle(verdicts, report["failures"], lengths=lengths)


def _run_rank(rank: int, W: int, row, rdzv: str | None, result_dir: str | None) -> dict:
    """One rank of a row over the judge's draws, all in one process group.  On every draw every rank runs the
    raw-mode KCP forward / backward in every mode; each mode's outputs (the parameter gradients summed by the
    reduction helper) are gathered to rank 0, which judges them under the mode's arm name, and the single-rank raw
    path on the whole pack (arm ``single``), against the FP64 reference of the unsplit pack beside the naive BF16
    tier.  On the first draw also: :data:`REPEATS` more saved passes (arm ``repeat``, the worst launch kept), the
    recompute handles' collectives, the hand-off counters, the graph events, the mode refusals, the explicit-mode
    launch comparison and the reduction helper's bits (one call's result on every rank; the FP32 rank-order sum of
    the partials)."""
    from torch.utils.checkpoint import checkpoint

    lengths, H, _w, layout = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    total = cu[-1]
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    ws = cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel=True)
    report: dict = {
        "rank": rank,
        "world": W,
        "row": _row_id(row),
        "pass": True,
        "flags": {},
        "device": torch.cuda.get_device_name(device),
        "arch": cake_jit.device_arch(device.index),
    }
    judge, failures = TierJudge(DRAWS), []

    def gather(x) -> list:
        if W == 1:
            return [x]
        objs: list = [None] * W
        dist.all_gather_object(objs, x)
        return objs

    def one_draw(j: int, glob: dict) -> None:
        first = j == 0
        A_log, dt_bias = glob["A_log"], glob["dt_bias"]
        tokens = {n: glob[n] for n in ("q", "k", "v", "g", "beta", "dout")}
        ops = _local_slices(part, rank, cu, tokens)
        q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
        GK = dict(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=LB)
        recs: dict = {}
        handle_runs: dict = {}
        flags: dict = {}
        ref = naive = None
        if rank == 0:
            # the exact FP64 reference of the unsplit pack and the naive tier (on this rank's device)
            ref = oracle_with_grads(glob, SCALE, LB, False)
            naive = naive_of(glob, SCALE, LB, False)

        def saved_pass(raw=g, workspace=ws, tag=None, **opts):
            out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q,
                k,
                v,
                raw,
                beta,
                ctx,
                workspace=workspace,
                **GK,
                **opts,
            )
            rec_f = dict(cake_kcp.LAST_CALL["fwd"])
            bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
            grads = cake_kcp.cake_kda_nondet_train_kcp_backward(
                q, k, v, raw, beta, ctx, dout, saved=saved, workspace=workspace, **GK, **bwd_opts
            )  # fmt: skip
            if tag is not None:
                recs[tag] = (rec_f, dict(cake_kcp.LAST_CALL["bwd"]))
            return {
                "out": out,
                **dict(zip(TOKEN_GRADS + PARAM_GRADS, grads, strict=True)),
            }

        def autograd_pass(
            ckpt: bool = False, handle=None, workspace=ws, handoff="auto", keep=None
        ):
            """With a handle (under the checkpoint) also records in ``handle_runs[keep]`` the original forward's and
            the recompute's LAST_CALL records, the backward's, and the torch.distributed calls per phase
            (``_Counter``; the recompute phase from the checkpoint's ``context_fn``)."""
            leaves = [
                x.detach().clone().requires_grad_(True)
                for x in (q, k, v, g, beta, A_log, dt_bias)
            ]
            keep_kw = {} if keep is None else {"checkpoint_keep": keep}

            def fn(q_, k_, v_, g_, b_, a_, d_, *h):
                return cake_kcp.cake_kda_nondet_kcp(
                    q_, k_, v_, g_, b_, ctx, None, "light", "saved", workspace,
                    recompute_handle=h[0] if h else None, checkpoint_handoff=handoff, saved_rows="tape",
                    use_gate_in_kernel=True, A_log=a_, dt_bias=d_, lower_bound=LB, **keep_kw,
                )  # fmt: skip

            extra = (handle,) if handle is not None else ()
            if handle is not None and ckpt:
                with _Counter() as cc:
                    out = checkpoint(
                        fn,
                        *leaves,
                        *extra,
                        use_reentrant=False,
                        context_fn=lambda: (
                            contextlib.nullcontext(),
                            cc.recompute_cm(),
                        ),
                    )
                    orig = cake_kcp.LAST_CALL.pop("fwd", None)
                    cc.phase = "bwd"
                    out.backward(dout)
                    rep = cake_kcp.LAST_CALL.pop("fwd", None)
                handle_runs[keep or "states"] = (
                    orig,
                    rep,
                    dict(cake_kcp.LAST_CALL["bwd"]),
                    cc.record(),
                )
            else:
                out = (
                    checkpoint(fn, *leaves, *extra, use_reentrant=False)
                    if ckpt
                    else fn(*leaves, *extra)
                )
                out.backward(dout)
            names = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
            return {
                "out": out.detach(),
                **{n: lf.grad for n, lf in zip(names, leaves, strict=True)},
            }

        def reduced(res: dict) -> tuple[torch.Tensor, torch.Tensor]:
            return cake_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads(
                ctx, res["dA_log"], res["ddt_bias"]
            )

        def judged(arm: str, res: dict) -> tuple[torch.Tensor, torch.Tensor]:
            """Every rank: reduce ``res``'s parameter partials (a collective) and gather its token outputs to rank 0,
            which judges them as arm ``arm``.  On the first draw also: the reduced sums are the same bits on every
            rank (one call's result).  Returns the reduced sums."""
            red = reduced(res)
            gathered = {n: gather(res[n].cpu()) for n in ("out",) + TOKEN_GRADS}
            if first:
                every = gather((red[0].cpu(), red[1].cpu()))
                flags["reduce_eq_ranks"] = flags.get("reduce_eq_ranks", True) and all(
                    torch.equal(e[0], every[0][0]) and torch.equal(e[1], every[0][1])
                    for e in every
                )
            if rank == 0:
                got = {n: _to_global(part, gathered[n], cu, total) for n in gathered}
                got["dA_log"], got["ddt_bias"] = red[0].cpu(), red[1].cpu()
                _judge_draw(judge, failures, j, {arm: got}, ref, naive, device)
            return red

        if first:
            pending = bool(getattr(cake_kcp.get_backend(), "_spec3_pending", False))
            res = saved_pass(tag="raw", spec3_probe="record")
            spec3 = (cake_kcp.LAST_CALL.get("bwd") or {}).get("spec3")
            handoff_stats = cake_kcp.cake_kda_nondet_train_kcp_handoff_stats()
        else:
            res = saved_pass()
        red = judged("kcp", res)
        if first:
            # the helper's sum: FP32, in rank order (rank 0 first), from the per-rank partials
            parts = gather((res["dA_log"].cpu(), res["ddt_bias"].cpu()))
            if rank == 0:
                sums = [torch.zeros_like(parts[0][0]), torch.zeros_like(parts[0][1])]
                for pa, pb in parts:
                    sums[0] += pa
                    sums[1] += pb
                report["reduce_eq_rank_order_sum"] = bool(
                    torch.equal(sums[0], red[0].cpu())
                    and torch.equal(sums[1], red[1].cpu())
                )
                report["pass"] &= report["reduce_eq_rank_order_sum"]
        del res
        # every mode on the same operands, each judged under its own name
        judged("replay", saved_pass(boundary_mode="replay"))
        judged("repeat", saved_pass())
        judged("autograd", autograd_pass())
        judged("ckpt", autograd_pass(ckpt=True))
        judged(
            "handle",
            autograd_pass(ckpt=True, handle=cake_kcp.CakeKDANondetKcpRecomputeHandle()),
        )
        judged(
            "handle_transfer",
            autograd_pass(
                ckpt=True,
                handle=cake_kcp.CakeKDANondetKcpRecomputeHandle(),
                keep="transfer",
            ),
        )
        # the checkpoint hand-off forced on: the set kept by both passes and read by the backward (a rank with tokens)
        handoff_on = autograd_pass(ckpt=True, handoff="on")
        handoff_on_bwd = cake_kcp.LAST_CALL["bwd"].get("checkpoint_handoff")
        judged("handoff_on", handoff_on)
        del handoff_on
        judged("save_transfer", saved_pass(save_transfer=False))
        judged("ws_none", autograd_pass(workspace=None))
        if first:
            # the counters: the three checkpointed "auto" calls (plain, and a recompute handle of each tier) decided
            # by the raw mode's rule, the forced one engaged
            flags["handoff_counts"] = _handoff_delta(
                handoff_stats, ctx, H, "raw", False, int(ctx.T_r), auto_calls=3
            ) and handoff_on_bwd == ("bound" if int(ctx.T_r) > 0 else None)
            flags.update(_handle_recompute_flags(handle_runs, W))
            # run to run: REPEATS more saved passes on the first draw (every rank runs all of them: they exchange)
            for _ in range(REPEATS):
                judged("repeat", saved_pass())
        # a strided raw view (a column slice of a wider fused projection, the single-rank rule's example): no copy
        T = int(ctx.T_r)
        wide = torch.zeros((T, (H + 2) * D), dtype=torch.bfloat16, device=device)
        wide[:, 8 : 8 + H * D] = g.reshape(T, H * D)
        view = wide[:, 8 : 8 + H * D].view(T, H, D)
        judged("strided", saved_pass(raw=view))
        del wide, view
        if W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS:
            judged("ring", saved_pass(exchange="ring", save_transfer=False))
        # graph replay: every rank captures and replays the same calls (the last one judged: a replay)
        cake_kcp.graph_cache_clear()
        events = []
        try:
            for _ in range(len(GRAPH_EVENTS)):
                res_graph = {
                    n: t.clone() for n, t in saved_pass(graph="replay").items()
                }
                events.append(
                    [
                        str(cake_kcp.LAST_CALL[x].get("graph_event"))
                        for x in ("fwd", "bwd")
                    ]
                )
            torch.cuda.synchronize(device)
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            cake_kcp.graph_cache_clear()
        judged("graph", res_graph)
        del res_graph
        if first:
            flags["graph_events"] = _graph_events_ok(events)
            # the mode is checked against the saved set and the workspace: symmetric on every rank, before any
            # collective
            _, saved_r = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, **GK
            )
            refusals = []
            for kw, match in (
                ({**GK, "lower_bound": -4.0}, "lower_bound"),
                ({}, "gate mode"),
            ):
                try:
                    cake_kcp.cake_kda_nondet_train_kcp_backward(
                        q, k, v, (g if kw else g.float()), beta, ctx, dout, saved=saved_r,
                        workspace=ws if kw else cake_kcp.CakeKDANondetKcpWorkspace(), **kw,
                    )  # fmt: skip
                    refusals.append(False)
                except ValueError as exc:
                    refusals.append(match in str(exc))
            flags["mismatch_refused"] = all(refusals)
            del saved_r
            # the explicit mode on the same activated gate (torch FP32): the same launches in the same order with
            # the same collectives -- gate_epilogue (a stage that launches only on sm_100a) in place of
            # param_finalize; the backwards in the "first" / "main" order (_bwd_sequence_first_main: each decides
            # its K1 tape replay's place and its K3's stream itself, from its own plan's fused-gradient record,
            # arena and the table set's first K3)
            g_exp = raw_gate(g, A_log, dt_bias, LB, torch.float32).contiguous()
            ws_g = cake_kcp.CakeKDANondetKcpWorkspace()
            _, saved_e = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g_exp, beta, ctx, workspace=ws_g
            )
            rec_fe = dict(cake_kcp.LAST_CALL["fwd"])
            cake_kcp.cake_kda_nondet_train_kcp_backward(
                q, k, v, g_exp, beta, ctx, dout, saved=saved_e, workspace=ws_g
            )
            rec_be = dict(cake_kcp.LAST_CALL["bwd"])
            rec_fr, rec_br = recs["raw"]
            mapped = [
                "param_finalize" if s == "gate_epilogue" else s
                for s in _bwd_sequence_first_main(rec_be)
            ]
            launch_cmp = {
                "fwd_collectives": [rec_fr["n_collectives"], rec_fe["n_collectives"]],
                "bwd_collectives": [rec_br["n_collectives"], rec_be["n_collectives"]],
                "fwd_launches": [rec_fr["n_launches"], rec_fe["n_launches"]],
                "bwd_launches": [rec_br["n_launches"], rec_be["n_launches"]],
                "gates": [
                    rec_fr["gate"],
                    rec_br["gate"],
                    rec_fe["gate"],
                    rec_be["gate"],
                ],
            }
            flags["collectives_eq_explicit"] = (
                _collective_seq(rec_fr) == _collective_seq(rec_fe)
                and _collective_seq(rec_br) == _collective_seq(rec_be)
                and rec_fr["n_collectives"] == rec_fe["n_collectives"]
                and rec_br["n_collectives"] == rec_be["n_collectives"]
            )
            flags["launches_eq_explicit"] = (
                rec_fr["sequence"] == rec_fe["sequence"] and _bwd_sequence_first_main(rec_br) == mapped
                and launch_cmp["gates"] == ["raw", "raw", "g", "g"]
            )  # fmt: skip
            del ws_g, saved_e
            # a backend with local stages whose probe was pending must have run it on the first pass (reported,
            # not gated beyond 'checked': the forward's intra program writes gk row 63 alone)
            flags["spec3"] = (
                not pending
                or (spec3 is None and T == 0)
                or bool(spec3 and spec3.get("checked"))
            )
            flag_list = gather(flags)
            report.update(
                flags_by_rank=flag_list,
                spec3_by_rank=gather(spec3),
                launches_by_rank=gather(launch_cmp),
                graph_events=events,
            )
            for key in flags:
                report["flags"][key] = all(f[key] for f in flag_list)
                report["pass"] &= report["flags"][key]
        if rank != 0:
            return
        # the single-rank raw path on the whole pack
        out_s, saved_s = cake_backend.cake_kda_nondet_train_forward(
            glob["q"],
            glob["k"],
            glob["v"],
            glob["g"],
            glob["beta"],
            cu_t,
            **GK,
        )
        grads_s = cake_backend.cake_kda_nondet_train_backward(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], cu_t, glob["dout"],
            saved=saved_s, **GK,
        )  # fmt: skip
        single = {
            "out": out_s,
            **dict(zip(TOKEN_GRADS + PARAM_GRADS, grads_s, strict=True)),
        }
        del saved_s
        _judge_draw(judge, failures, j, {"single": single}, ref, naive, device)

    seed = _row_seed("rows", row)
    for j in range(DRAWS):
        one_draw(j, draw_case(lengths, H, "raw", False, draw_seed(seed, j), device))
        gc.collect()
    if rank == 0:
        failures += judge.failures()
        report["verdicts"], report["failures"] = judge.verdicts(), failures
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"])
        report["pass"] = all(verdict)
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "raw-" + _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    gc.collect()
    return report


def _spawn_entry(rank: int, W: int, row, rdzv: str | None, result_dir: str) -> None:
    try:
        report = _run_rank(rank, W, row, rdzv, result_dir)
        if not report["pass"]:
            raise AssertionError(
                f"rank {rank}: {report['row']} exact checks failed: {report['flags']}"
            )
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


def _skip_unless_runnable(W: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    capability = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip("the current device's architecture is not covered by the package")


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_raw_gate_rows_match_single_rank_and_fp64(row, record_property) -> None:
    """Every mode of the row's KCP raw route (arms ``kcp``, ``replay``, ``repeat``, ``autograd``, ``ckpt``,
    ``handle``, ``handle_transfer``, ``handoff_on``, ``save_transfer``, ``ws_none``, ``strided``, ``ring`` where it
    runs, ``graph``) and the single-rank raw path on the whole pack (arm ``single``), every output of each judged
    against the FP64 reference of the unsplit pack beside the naive BF16 tier over the judge's draws (no arm is
    compared with another), and the first draw's exact checks on every rank (see the module doc)."""
    lengths, H, W, layout = row
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1")) == W
        and W > 1
    ):
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]), W, row, None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )  # fmt: skip
        _finish(record_property, report, lengths)
        return
    _skip_unless_runnable(W)
    if W == 1:
        report = _run_rank(0, 1, row, None, os.environ.get("KCP_TEST_RESULT_DIR"))
        _finish(record_property, report, lengths)
        return
    with tempfile.TemporaryDirectory(prefix="cake_kcp_raw_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry, W, row, tmp, result_dir)
        with open(os.path.join(result_dir, "raw-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    _finish(record_property, report, lengths)


@pytest.mark.parametrize("l2n", ("off", "fused"))
def test_gpu_raw_raises_no_alert_under_torch_deterministic_mode(
    l2n, record_property
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict and with ``warn_only=True``, a raw KCP backward
    raises no non-determinism alert and gives no such warning on any architecture, with the in-kernel Q/K L2 norm off
    (``glm_intra_raw`` / ``param_finalize``) and on (``l2n="fused"``, ``use_qk_l2norm_in_kernel=True``:
    ``param_finalize_l2n``): eagerly, on a captured graph's replay, and when its graph capture fails and the call
    prepares again to run eagerly, on every draw.  The results are judged against the FP64 reference beside the
    naive BF16 tier over the judge's draws, each call under its own name: the forward and the eager backward with the
    mode off (arm ``""``), the backwards under the mode (arms ``strict`` / ``warn_only``, ``strict_replay`` /
    ``warn_only_replay`` on the captured graph, ``capture_fails`` its two calls, the worst kept); no call is compared
    with another.  On the first draw, exactly: the captured backward's graph events, a new backward record per call,
    the replay's graph event and the one failed capture."""
    _skip_unless_runnable(1)
    if l2n != "off":
        missing = cake_kcp_plan.qk_l2norm_missing(
            "fused", "raw", cake_kcp_plan.qk_l2norm_arch()
        )
        if missing is not None:
            pytest.skip(
                f"use_qk_l2norm_in_kernel=True cannot run on the KCP route yet: {missing}"
            )
    device = torch.device("cuda", torch.cuda.current_device())
    L2 = {"use_qk_l2norm_in_kernel": True} if l2n != "off" else {}
    # PyTorch's alert text (a RuntimeError, a UserWarning with warn_only=True), whichever call raises it
    alert = "does not have a deterministic implementation"
    lengths, H = (100, 640, 77), 4
    part, cu = _partition(lengths, 1, "contiguous")
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(part, None, 0, device)
    ws = cake_kcp.CakeKDANondetKcpWorkspace(use_gate_in_kernel=True, **L2)
    mode = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    names = TOKEN_GRADS + PARAM_GRADS

    def draw(j: int, s: int):
        glob = draw_case(lengths, H, "raw", bool(L2), s, device)
        o = _local_slices(
            part, 0, cu, {n: glob[n] for n in ("q", "k", "v", "g", "beta", "dout")}
        )
        GK = dict(
            use_gate_in_kernel=True,
            A_log=glob["A_log"],
            dt_bias=glob["dt_bias"],
            lower_bound=LB,
        )
        ops = tuple(o[n] for n in ("q", "k", "v", "g", "beta"))

        def bwd(saved, **kw):
            return cake_kcp.cake_kda_nondet_train_kcp_backward(
                *ops, ctx, o["dout"], saved=saved, workspace=ws, **GK, **L2, **kw
            )  # fmt: skip

        def quiet(saved, tag, **kw) -> dict:
            """One backward with every warning recorded: none is PyTorch's non-determinism alert; its results by
            name, cloned (a replay's outputs are the graph's)."""
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                grads = tuple(t.clone() for t in bwd(saved, **kw))
            torch.cuda.synchronize(device)
            ours = [str(w.message) for w in caught if alert in str(w.message)]
            assert not ours, (tag, ours)
            return dict(zip(names, grads, strict=True))

        arms: dict = {}
        cake_kcp.graph_cache_clear()
        try:
            torch.use_deterministic_algorithms(False)
            # the forward and the eager backward with the mode off
            out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
                *ops, ctx, workspace=ws, **GK, **L2
            )
            arms[""] = {
                "out": out.clone(),
                **{n: t.clone() for n, t in zip(names, bwd(saved), strict=True)},
            }
            # a captured backward to replay below
            _, saved_g = cake_kcp.cake_kda_nondet_train_kcp_forward(
                *ops, ctx, workspace=ws, graph="replay", **GK, **L2
            )
            bwd(saved_g)  # the backward arena: a graph key sees the workspace's pools
            events = []
            for _ in range(2):
                bwd(saved_g, graph="replay")
                events.append(cake_kcp.LAST_CALL["bwd"].get("graph_event"))
            torch.cuda.synchronize(device)
            if j == 0:
                assert events == ["eager", "capture"], events
            for warn_only in (False, True):
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
                tag = "warn_only" if warn_only else "strict"
                before = cake_kcp.LAST_CALL.get("bwd")
                arms[tag] = quiet(saved, f"{tag} eager")
                if j == 0:
                    assert cake_kcp.LAST_CALL.get("bwd") is not before, tag
                # the captured graph's replay (its own preparation) raises none either
                arms[f"{tag}_replay"] = quiet(saved_g, f"{tag} replay", graph="replay")
                if j == 0:
                    assert cake_kcp.LAST_CALL["bwd"].get("graph_event") == "replay", tag
            # a new key: its first call runs eagerly, its second fails to capture (as _graph_capture records a
            # failure) and runs eagerly after a second preparation: no warning on either preparation
            failed: list = []

            def capture_fails(kind, entry, call, body):
                failed.append(kind)
                entry.disabled = "test: the capture failed"
                return None

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(cake_kcp, "_graph_capture", capture_fails)
                arms["capture_fails"] = [
                    quiet(saved, f"capture fails {i}", graph="replay") for i in range(2)
                ]
            if j == 0:
                assert failed == ["bwd"], failed
        finally:
            torch.use_deterministic_algorithms(mode[0], warn_only=mode[1])
            cake_kcp.graph_cache_clear()
        return arms, glob

    judge_arms(
        record_property,
        draw,
        seed=_row_seed(f"no_alert_{l2n}", (lengths, H, 1, "contiguous")),
        scale=SCALE,
        l2n=bool(L2),
        lower_bound=LB,
    )
