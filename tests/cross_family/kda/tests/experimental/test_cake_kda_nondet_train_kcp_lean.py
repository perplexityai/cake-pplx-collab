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

The lean local arena of the KCP backward of the non-deterministic package (``CakeKDANondetKcpWorkspace(local_arena="lean")``,
opt-in; ``"replay"`` is the default): this route's ``LOCAL_LAYOUT_BWD_LEAN`` layout registered on the existing backend's arenas (3332 B per
internal token-head; no ``kcp_dav``, the tail stages once per chunk range on per-pass buffers;
two tail passes unless ``lean_passes`` says otherwise) against the ``"replay"`` arena (4488 B).  Host rows: the
layout's bytes, lifetimes and per-pass packing, the single-rank layouts and bytes unchanged (with and without the
registered layout, line for line), the registration rules, the fused-gradient records the layout relies on (this
package's ``S1`` forms among them), every logical name of every launched
record bound (the tail ranges' item bounds from the arena's tail-count table) and the byte prediction.  GPU rows: the
unit rows (H 1-64, P > 1, ragged packs, both boundary modes, checkpointed autograd, recompute handles, graph replay, the
overlap schedule) on the lean arena NaN-filled, the per-pass buffers NaN-filled before every write, beside the replay
arena; the workspace bytes equal to ``cake_kda_nondet_train_kcp_workspace_bytes``, no allocation or blocking in steady
calls, and the multi-GPU rows (W = 2 / 4 / 8) in the launch forms of ``test_cake_kda_nondet_train_kcp_shared_ws.py``.

Accuracy: this package has no bit contract, so the lean arena's results are not compared with the replay arena's.
Out and every gradient of each arena and path are judged under their own names per pack against the FP64 reference
beside the naive BF16 tier of ``tests/cake_naive.py`` over 32 draws (``NaiveJudge`` of
``test_cake_kda_nondet_train_kcp_shared_ws.py``; the owner-accepted gaps only).  The proxy pack (16380 tokens) is
judged at H = 4 (the production-size case of each lean route); at H = 64 it keeps its exact checks only (the packs of
the same rows judge the route).  A written saved set is an internal of the package: not compared; its backward's
gradients are judged.
"""

import gc
import json
import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.experimental.test_cake_kda_nondet_train_kcp_dist import _rendezvous
from tests.cake_naive import draw_seed
from tests.experimental.test_cake_kda_nondet_train_kcp_shared_ws import (
    BWD_GRADS,
    DIST_ROWS,
    HANDLE_LAYERS,
    JUDGE_SEED,
    PACKS,
    NaiveJudge,
    _bwd,
    _cu,
    _device_or_skip,
    _dist_row_id,
    _draw_ops,
    _finish_rank_report,
    _fwd,
    _handle_pass,
    _judge_packs,
    _judge_rank_arms,
    _part,
    _pointers,
    _rank_ops,
    _rank_packs,
    _schedule,
    draws_of,
    handle_arms,
    host,
    judged,
    pack_inputs,
    schedule_arms,
    settle_naive,
    stack_fla,
    stack_refs,
)
from tests.experimental.test_helpers.cake_host_sync import (
    allocator_counts,
    census,
    nonblocking,
    summary,
)
from tests.cake.test_helpers.cake_package import cake_package

kp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_plan")
cake_kcp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp")
cake_backend = cake_package("cake.cake_kda_nondet_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")
cake_kcp_jit = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_jit")

LEAN = kp.LOCAL_LAYOUT_BWD_LEAN
CONTRACT_HEADS = (1, 2, 4, 8, 16, 32, 64)
ARCH_SMS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 216}
# a pack of the CP16 proxy's geometry (16380 tokens in 12 fragments: NC_pad = 268, R * H = 1097728 at H = 64): the
# smallest arena at four tail passes on sm_100a, three where GLM intra fuses the gate epilogue, the per-pass set inside
# the dead whole-sequence slots
PROXY_PACK = (3000, 1000, 2500, 64, 1, 4000, 2000, 1500, 700, 600, 500, 515)
# the per-pass buffers the tail writes before it reads them (the fused gradient's, then GLM intra's)
PASS_BUFFERS = ("dq_f", "dk_f", "dg_f", "db_f", "dAkk", "dAqk", "dg_int", "db_int")


def _lean(**kw) -> int:
    """``cake_kda_nondet_train_kcp_workspace_bytes`` of the lean arena."""
    return kp.cake_kda_nondet_train_kcp_workspace_bytes(local_arena="lean", **kw)


def _passes_of(local) -> int | None:
    """The lean tail passes of a local arena or plan view (``None``: the replay layout)."""
    return cake_kcp._lean_passes(local)


def _ws(local_arena: str = "lean", lean_passes: int | None = None):
    backend = cake_kcp.get_backend()
    return cake_kcp.CakeKDANondetKcpWorkspace(
        state_dtype=backend.dtypes["state"],
        local=backend.local_stages,
        local_arena=local_arena,
        lean_passes=lean_passes,
    )


def _whole(regions) -> int:
    """Bytes of the whole-sequence part of a layout (the slots up to and including ``beta_int``'s)."""
    by = {r.name: r for r in regions}
    return by["beta_int"].offset + by["beta_int"].nbytes


# --------------------------------------------------------------------------- CPU


def test_lean_layout_bytes_lifetimes_and_packing() -> None:
    """``KCP_LEAN_MODE`` on the host for every architecture, contract head count and several geometries: the layout
    checker accepts every pass count (no two occupants overlap in bytes and lifetime), the whole-sequence part is
    3332 B per internal token-head with no state block, no whole ``dAqk``, no ``dv1`` and no sink, ``kb`` beside ``u``
    under the ``dh`` tape, ``aqk_int`` live through the adjoint walks (step 5) in a slot of its own, the per-pass set
    sized for one range in the bytes of qg / kg, w and aqk_int (or an overflow slot after them), the gate buffers only where the gate epilogue launches; the chosen passes give the
    smallest arena, below the replay arena, and the default two passes; on the proxy geometry the smallest arena is
    exactly 3332 B per internal token-head at four passes (sm_100a) / three (sm_103a, sm_107a)."""
    seen = 0
    for arch, sm in ARCH_SMS.items():
        for H in CONTRACT_HEADS:
            for lengths in ((1000, 577), (50, 640, 77, 1000, 129), (1450,), PROXY_PACK):
                S, N = sum(lengths), len(lengths)
                p = cake_backend.plan(S, N, H, sm, arch)
                assert kp.local_lean_serves(p), (arch, H, N)
                unit = p.R * p.H
                sizes = []
                for b in range(1, kp.LEAN_MAX_PASSES + 1):
                    regions = p.layout(LEAN, b)
                    by = {r.name: r for r in regions}
                    names = set(by)
                    assert not names & {"seg_b", "seg_m", "hseg", "dseg", "dhseg"}
                    assert _whole(regions) == 3332 * unit, (arch, H, N, b)
                    assert not names & {"dAqk_sink", "dv1"}
                    u, kb, dh, aqk = by["u"], by["kb"], by["dh"], by["aqk_int"]
                    assert u.slot == kb.slot == dh.slot != aqk.slot
                    assert u.offset + u.nbytes <= kb.offset < dh.offset + dh.nbytes
                    assert (kb.lifetime, u.lifetime, dh.lifetime) == (
                        (1, 2),
                        (2, 3),
                        (5, 6),
                    )
                    assert aqk.lifetime == (1, 5) and aqk.offset > by["w"].offset
                    R_b = 64 * -(-p.NC_pad // b)
                    for name in PASS_BUFFERS:
                        if name in _GATE and p.gate_fused:
                            assert name not in names
                            continue
                        r = by[name]
                        assert r.shape[0] == R_b and r.lifetime[0] >= 6, (name, r)
                        assert r.offset >= by["qg"].offset, (name, r)
                    assert by["dAqk"].shape == (R_b, H, 64)
                    sizes.append(p.workspace_bytes(LEAN, b))
                    seen += 1
                best = 1 + sizes.index(min(sizes))
                assert kp.local_lean_passes(p) == kp.LEAN_PASSES_DEFAULT == 2
                fwd = p.workspace_bytes("fwd_kcp")
                arena = kp.local_arena_bytes(p, backward=True, local_arena="lean")
                assert arena == max(fwd, sizes[1]) == kp.local_lean_bytes(p, 2)
                replay = p.arena_bytes(backward=True)
                assert kp.local_arena_bytes(p, backward=True) == replay
                assert (
                    kp.local_arena_bytes(p, backward=True, local_arena="replay")
                    == replay
                )
                assert max(fwd, min(sizes)) <= arena < replay
                for b in range(1, kp.LEAN_MAX_PASSES + 1):
                    assert kp.local_arena_bytes(
                        p, backward=True, local_arena="lean", lean_passes=b
                    ) == max(fwd, sizes[b - 1])
                assert kp.local_arena_bytes(p, backward=False) == p.arena_bytes(
                    backward=False, fwd_layout="fwd_kcp"
                )
                if lengths is PROXY_PACK:
                    assert p.NC_pad == 268 and max(fwd, min(sizes)) == 3332 * unit, (
                        arch,
                        H,
                    )
                    if H == 64:
                        assert best == (3 if p.gate_fused else 4), (arch, best)
                        assert p.workspace_bytes("replay") == 4488 * unit
    assert seen == len(ARCH_SMS) * len(CONTRACT_HEADS) * 4 * kp.LEAN_MAX_PASSES


_GATE = ("dg_int", "db_int")


def test_single_rank_layouts_keep_their_bytes() -> None:
    """The single-rank layouts are untouched by the KCP mode: on the serial route (H = 64, sm_100a) the bytes per
    internal token-head of record -- forward 2560, KCP forward 2432, saved-set backward 4360, replay 4488, lean
    one pass 4108, lean four passes 2820 (tape: 3340 / 2052) -- and the default / lean arenas of a workspace; the
    KCP mode is in none of the single-rank mode tuples, and a single-rank workspace refuses ``layouts`` with an arena
    option, ``doc_lengths``, an unknown forward layout or an empty sequence."""
    S, N, H, sm = sum(PROXY_PACK), len(PROXY_PACK), 64, 148
    p = cake_backend.plan(S, N, H, sm, "sm_100a")
    unit = p.R * p.H
    assert p.NSEG == 0
    want = {
        ("fwd", 1): 2560,
        ("fwd_kcp", 1): 2432,
        ("bwd", 1): 4360,
        ("replay", 1): 4488,
        ("bwd_lean", 1): 4108,
        ("bwd_lean", 4): 2820,
        ("bwd_tape_lean", 1): 3340,
        ("bwd_tape_lean", 4): 2052,
    }
    for (mode, b), per in want.items():
        assert p.workspace_bytes(mode, b) == per * unit, (mode, b)
    assert LEAN not in cake_backend.MODES + cake_backend.LEAN_MODES
    assert LEAN not in cake_backend.SLOTS and LEAN not in cake_backend.FWD_LAYOUTS
    assert "_ROUTE_SLOTS" not in cake_backend.__all__
    assert p.arena_bytes(backward=True) == 4488 * unit
    assert (
        p.arena_bytes(
            backward=True, arena=cake_backend.CakeKDANondetArena(tail_passes=4)
        )
        == 2820 * unit
    )
    for bad in (
        {"arena": cake_backend.CakeKDANondetArena(lean=True)},
        {"arena": cake_backend.CakeKDANondetArena()},
        {"doc_lengths": PROXY_PACK},
        {"fwd_layout": "tiny"},
    ):
        with pytest.raises(ValueError):
            cake_backend.CakeKDANondetWorkspace(p, "cpu", layouts=((LEAN, 4),), **bad)
    with pytest.raises(ValueError):
        cake_backend.CakeKDANondetWorkspace(p, "cpu", layouts=())


def test_single_rank_workspace_refuses_its_lean_modes_in_layouts() -> None:
    """``layouts`` holds another route's modes only: a single-rank lean mode (its tail passes come from the arena
    option, not from ``layouts``) is refused at any passes, alone or beside a route layout."""
    p = cake_backend.plan(sum(PROXY_PACK), len(PROXY_PACK), 64, 148, "sm_100a")
    for mode in cake_backend.LEAN_MODES:
        for passes in (1, 2, 4):
            for layouts in (((mode, passes),), ((LEAN, 2), (mode, passes))):
                with pytest.raises(ValueError, match="take their tail passes"):
                    cake_backend.CakeKDANondetWorkspace(p, "cpu", layouts=layouts)


def test_route_layout_registration_rules() -> None:
    """``cake_backend._register_route_slots``: importing the KCP module registered the lean layout with this module's
    slot function (registering it again changes nothing); a mode of the single-rank module (:data:`MODES`,
    :data:`LEAN_MODES`, :data:`FWD_LAYOUTS`) and a registered mode with another function are refused; a mode looked up
    before its registration raises and that miss is not cached (``_layout``'s cache keys on the mode and the passes,
    and a mode keeps its one function, so a cached layout never goes stale); the default arena option passes a
    registered mode through to the layout."""
    assert cake_backend._ROUTE_SLOTS[LEAN] is kp._lean_slots
    assert kp.register_lean_layout() is cake_backend
    cake_backend._register_route_slots(LEAN, kp._lean_slots)
    assert cake_backend._ROUTE_SLOTS[LEAN] is kp._lean_slots
    for own in cake_backend.MODES + cake_backend.LEAN_MODES + cake_backend.FWD_LAYOUTS:
        with pytest.raises(ValueError):
            cake_backend._register_route_slots(own, kp._lean_slots)
        assert own not in cake_backend._ROUTE_SLOTS
    with pytest.raises(ValueError):
        cake_backend._register_route_slots(LEAN, lambda *a: kp._lean_slots(*a))
    assert cake_backend._ROUTE_SLOTS[LEAN] is kp._lean_slots
    p = cake_backend.plan(sum(PROXY_PACK), len(PROXY_PACK), 64, 148, "sm_100a")
    probe = "kcp_test_probe_layout"
    with pytest.raises(ValueError):
        p.layout(probe, 2)
    try:
        cake_backend._register_route_slots(probe, kp._lean_slots)
        assert p.layout(probe, 2) == p.layout(LEAN, 2)
        assert p.workspace_bytes(probe, 3) == p.workspace_bytes(LEAN, 3)
    finally:
        cake_backend._ROUTE_SLOTS.pop(probe, None)
        cake_backend._layout.cache_clear()
    for b in (1, 2, 4):
        assert p.layout(LEAN, b) == p.layout(LEAN, b)
    assert p.layout(LEAN, 2) != p.layout(LEAN, 4)
    assert cake_backend.CakeKDANondetArena().mode(LEAN) == LEAN


# The single-rank parity rows: every layout mode of the single-rank module at every tail-pass count it takes and the
# arena bytes of every arena option, over the architectures, head counts and geometries below (the serial and the
# segmented routes, the dav rows H 9-15), one text line each; an error is part of the record (its type).  The same
# rows of this package before the KCP lean layout compare equal line for line (a base-versus-branch run of this
# function on both trees).
_PARITY_HEADS = (1, 2, 4, 8, 12, 16, 32, 64)
_PARITY_PACKS = (
    (1000, 577),
    (50, 640, 77, 1000, 129),
    (1450,),
    (3000, 1000, 2500, 64, 1, 4000, 2000, 1500, 700, 600, 500, 515),
    (64,) * 40,
    (8192, 8192),
)
_PARITY_SMS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 216}


def _single_rank_record(arch: str) -> list[str]:
    """The parity rows of ``arch`` as text lines (see :data:`_PARITY_SMS`)."""

    def regions(p, mode, b):
        try:
            return repr(
                [
                    (
                        r.name,
                        r.slot,
                        r.offset,
                        r.nbytes,
                        r.shape,
                        str(r.dtype),
                        r.lifetime,
                    )
                    for r in p.layout(mode, b)
                ]
            )
        except (ValueError, KeyError) as e:
            return type(e).__name__

    def arena(p, **kw):
        try:
            return str(p.arena_bytes(**kw))
        except (ValueError, KeyError) as e:
            return type(e).__name__

    Arena = cake_backend.CakeKDANondetArena
    options = (
        ("none", None),
        ("default", Arena()),
        ("lean", Arena(lean=True)),
        ("b2", Arena(tail_passes=2)),
        ("b3", Arena(tail_passes=3)),
        ("b4", Arena(tail_passes=4)),
        ("b8", Arena(tail_passes=8)),
        ("batched", Arena(tail_passes=4, batch_tokens=2048)),
    )
    lines = []
    sm = _PARITY_SMS[arch]
    for lengths in _PARITY_PACKS:
        S, N = sum(lengths), len(lengths)
        for H in _PARITY_HEADS:
            p = cake_backend.plan(S, N, H, sm, arch)
            row = f"{arch}|{S}|{N}|{H}|R={p.R}|NC={p.NC_pad}|NSEG={p.NSEG}|gf={p.gate_fused}"
            for mode in cake_backend.MODES + cake_backend.FWD_LAYOUTS[1:]:
                lines.append(f"{row}|{mode}|1|{regions(p, mode, 1)}")
            for mode in cake_backend.LEAN_MODES:
                for b in range(1, 9):
                    lines.append(f"{row}|{mode}|{b}|{regions(p, mode, b)}")
            for backward in (True, False):
                for save_mode in cake_backend.SAVE_MODES:
                    for name, option in options:
                        for docs in (None, lengths):
                            lines.append(
                                f"{row}|arena|{backward}|{save_mode}|{name}|{docs is not None}|"
                                + arena(
                                    p,
                                    backward=backward,
                                    save_mode=save_mode,
                                    arena=option,
                                    doc_lengths=docs,
                                )
                            )
            for fwd_layout in cake_backend.FWD_LAYOUTS:
                lines.append(
                    f"{row}|arena_fwd|{fwd_layout}|"
                    + arena(p, backward=False, fwd_layout=fwd_layout)
                )
    return lines


@pytest.mark.parametrize("arch", tuple(_PARITY_SMS))
def test_single_rank_layouts_and_bytes_ignore_the_route_registry(arch) -> None:
    """Every single-rank layout (default, lean and tail-pass forms, tape, the KCP forward layout) and every arena byte
    count (default, lean, tail-pass and batched options, forward-only, tape) on the parity rows is the same, line for
    line, with the KCP lean layout registered (this module imported the KCP module) and with the registry emptied
    (each computed on a cleared layout cache); the record holds the serial and the segmented routes."""
    assert LEAN in cake_backend._ROUTE_SLOTS
    saved = dict(cake_backend._ROUTE_SLOTS)
    try:
        cake_backend._layout.cache_clear()
        with_lean = _single_rank_record(arch)
        cake_backend._ROUTE_SLOTS.clear()
        cake_backend._layout.cache_clear()
        without = _single_rank_record(arch)
    finally:
        cake_backend._ROUTE_SLOTS.clear()
        cake_backend._ROUTE_SLOTS.update(saved)
        cake_backend._layout.cache_clear()
    assert with_lean == without
    assert any("NSEG=0" in line for line in with_lean)
    assert any("NSEG=0" not in line for line in with_lean)
    print(f"SINGLE_RANK_RECORD {arch} rows={len(with_lean)}")


def test_lean_layout_rests_on_the_davq_records() -> None:
    """The replay fused-gradient record of every contract head count (1, 2, 4, 8, 16, 32, 64) on every architecture
    and document count is one of the records that store the whole ``dAqk`` themselves (the lean layout's premise);
    H 9-15 selects ``dqkg_int``, which leaves ``dAqk`` to ``kcp_dav``: the lean layout does not serve those rows and
    the workspace keeps the replay arena there."""
    for arch, sm in ARCH_SMS.items():
        for N in (1, 2, 3, 12, 64, 300):
            S = 64 * 4 * N
            for H in CONTRACT_HEADS:
                p = cake_backend.plan(S, N, H, sm, arch)
                assert p.record("dqkg", kp.LOCAL_PATH_BWD) in kp.LEAN_DQKG_RECORDS, (
                    arch,
                    N,
                    H,
                )
                assert kp.local_lean_passes(p) is not None
            for H in range(9, 16):
                p = cake_backend.plan(S, N, H, sm, arch)
                assert p.record("dqkg", kp.LOCAL_PATH_BWD) == "dqkg_int"
                assert not kp.local_lean_serves(p) and kp.local_lean_passes(p) is None
                assert kp.local_arena_bytes(
                    p, backward=True, local_arena="lean"
                ) == p.arena_bytes(backward=True)
    for name in kp.LEAN_DQKG_RECORDS:
        logical = cake_jit.MODULES[name]["logical"]
        assert logical["dAqk_out"] == "dAqk" and logical["seq_chunk_start"] == (
            "seq_chunk_start"
        )


@pytest.mark.parametrize(
    "H,arch", [(64, "sm_100a"), (64, "sm_103a"), (64, "sm_107a"), (4, "sm_100a")]
)
def test_lean_stage_values_cover_the_records(H, arch) -> None:
    """The kernel backend's value dicts of a backward on the lean layout (an arena of meta tensors): every logical
    name of every launched record bound; the ``dav`` stage launches nothing (the lean layout serves only plans whose
    fused gradient stores every dAqk row, so ``kcp_dav`` has no reader: ``kcp_dav_needed``); the tail stages
    launch once per range of ``ceil(NC_pad / passes)`` chunks with the whole-sequence rows offset by the range's
    first chunk, the chunk tables sliced, ``num_seqs = 0`` and the item bound the tail-count table's entry of the
    range's real chunk count (their sum: the context's ``NC``), on the per-pass buffers of the layout and the plan's
    grids at the range's chunk count; the gate epilogue only where it launches; the replay layout's launch list
    with one tail range is the earlier one."""
    sm = ARCH_SMS[arch]
    ctx = kp.cake_kda_nondet_train_kcp_build_context(
        _part(PROXY_PACK), None, 0, "cpu", 4, sm_count=sm
    )
    plan = cake_backend.plan(ctx.T_r, ctx.n_frag, H, sm, arch)
    passes = kp.local_lean_passes(plan)
    assert passes > 1
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
    launches: list = []
    words = kp.local_tail_count_words(
        kp.local_arena_bytes(plan, backward=True, local_arena="lean"), H
    )

    class Arena:  # the lean arena's surface used by begin_call / stage, on meta tensors
        def __init__(self) -> None:
            self.plan = plan
            self.device = torch.device("cpu")
            self.device_index = 0
            self.arch = arch
            self.launcher = self._launch
            self.fwd_layout = kp.LOCAL_LAYOUT_FWD_ONLY
            self.layout_passes = {LEAN: passes}
            setattr(
                self,
                cake_kcp.TAIL_COUNTS_ATTR,
                torch.arange(words, dtype=torch.int32, device="meta"),
            )
            self._views: dict = {}

        def views(self, mode: str):
            ns = self._views.get(mode)
            if ns is None:
                ns = types.SimpleNamespace()
                tail = passes if mode == LEAN else 1
                for r in plan.layout(mode, tail):
                    setattr(
                        ns, r.name, torch.empty(r.shape, dtype=r.dtype, device="meta")
                    )
                self._views[mode] = ns
            return ns

        def _launch(self, record, values, *, grid, device_index) -> None:
            launches.append(("backend", record, grid, dict(values)))

    class Provider:
        def run_record(self, record, values, *, grid, device_index) -> None:
            launches.append(("kcp", record, grid, dict(values)))

    arena = Arena()
    backend = cake_kcp.KernelBackend(programs=Provider())
    ops = {n: torch.zeros(1) for n in ("q", "k", "v", "g", "beta", "dout")}
    grads = {n: torch.zeros(1) for n in ("dq", "dk", "dv", "dg", "dbeta")}
    call = types.SimpleNamespace(
        local=arena,
        ctx=ctx,
        tb=tb,
        T_r=int(ctx.T_r),
        H=H,
        backward=True,
        host_trim=1,
        ops=ops,
        scale=0.5,
        scratch={},
        local_values=None,
        out=None,
        grads=grads,
        device=torch.device("cpu"),
        opts=cake_kcp._options({"spec3_probe": "off"}),
    )
    backend.begin_call(call)
    views = arena.views(LEAN)
    assert call.scratch["views"] is views
    for name, extra in (
        ("bwd_prep", {"saved": None}),
        ("fwd_intra", {}),
        ("wy", {}),
        ("dav", {}),
    ):
        backend.stage(name, call, **extra)
    tail = call.local_values["_kcp_tail"]
    ranges = cake_backend._tail_ranges(plan.NC_pad, passes)
    assert len(tail) == len(ranges) > 1
    head = len(launches)
    for i in range(len(tail)):
        for name in cake_kcp.TAIL_STAGES:
            backend.stage(name, call, tail_pass=i)
    for kind, record, _grid, values in launches:
        logical = set(
            (cake_jit if kind == "backend" else cake_kcp_jit)
            .MODULES[record]["logical"]
            .values()
        )
        assert not logical - set(values), (record, sorted(logical - set(values)))
    assert not kp.kcp_dav_needed(plan) and call.scratch["kcp_dav"] == "skipped"
    assert [r for _, r, _, _ in launches[:head]].count("kcp_dav") == 0
    stages = [s for s in cake_kcp.TAIL_STAGES if plan.dispatched(s, kp.LOCAL_PATH_BWD)]
    assert plan.dispatched("gate_epilogue", kp.LOCAL_PATH_BWD) == (arch == "sm_100a")
    body = launches[head:]
    assert len(body) == len(ranges) * len(stages)
    counts = getattr(arena, cake_kcp.TAIL_COUNTS_ATTR)
    total = 0
    for i, (c0, length) in enumerate(ranges):
        rows = 64 * c0
        real = min(max(int(ctx.NC) - c0, 0), length)
        total += real
        for j, stage in enumerate(stages):
            kind, record, grid, values = body[i * len(stages) + j]
            assert record == plan.record(stage, kp.LOCAL_PATH_BWD)
            assert grid == cake_jit.launch_grid(
                record,
                num_chunks_pad=length,
                num_seqs=plan.N,
                num_heads=H,
                sm_count=sm,
                num_segments=plan.NSEG,
            )
            for name in cake_kcp.TAIL_ROW_BUFFERS:
                v, whole = values[name], getattr(views, name)
                assert v.shape[0] == whole.shape[0] - rows, (name, i)
                assert v.storage_offset() == whole.storage_offset() + rows * (
                    whole.stride(0)
                ), (name, i)
            assert values["h2d"].storage_offset() == views.h.storage_offset() + c0 * (
                views.h.stride(0)
            )
            assert values["dh2d"].storage_offset() == views.dh.storage_offset() + c0 * (
                views.dh.stride(0)
            )
            assert values["num_seqs"] == 0
            sc = values["seq_chunk_start"]
            assert (
                sc.numel() == 1
                and sc.storage_offset() == counts.storage_offset() + real
            )
            assert torch.equal(
                values["chunk_bos"], ctx.tables["chunk_bos"][c0 : c0 + length]
            )
            assert torch.equal(
                values["chunk_len"], ctx.tables["chunk_len"][c0 : c0 + length]
            )
            for name in PASS_BUFFERS:
                if hasattr(views, name):
                    assert values[name] is getattr(views, name), (name, i)
    assert total == int(ctx.NC)
    assert ranges[-1][0] + ranges[-1][1] == plan.NC_pad


def test_lean_byte_prediction_on_the_host() -> None:
    """``cake_kda_nondet_train_kcp_workspace_bytes`` (host only) under the lean arena: one context's local arena is
    ``local_arena_bytes`` plus the tail-count table, the list form takes the first of the largest arenas with its
    table, ``local_arena="replay"`` (and the default, ``None``) gives the existing backend's arena, a forward-only
    workspace is the same under either; the lean bytes are below the replay ones on every row."""
    seen = 0
    for W in (1, 2, 4):
        for layout in ("contiguous", "zigzag_per_document"):
            parts = [_part(lengths, W, layout) for lengths in (*PACKS, PROXY_PACK)]
            for r in range(W):
                ctxs = [
                    kp.cake_kda_nondet_train_kcp_build_context(
                        p, None, r, "cpu", 4, sm_count=152
                    )
                    for p in parts
                ]
                for H in (4, 12, 64):
                    replay = kp.cake_kda_nondet_train_kcp_workspace_bytes(
                        ctxs, H=H, local_arena="replay"
                    )
                    assert (
                        kp.cake_kda_nondet_train_kcp_workspace_bytes(ctxs, H=H)
                        == replay
                    )
                    # the pools and the scratch tables: the same under either arena
                    rest = replay - max(
                        cake_backend.workspace_bytes(
                            c.T_r,
                            c.n_frag,
                            H,
                            sm_count=c.sm_count,
                            fwd_layout=kp.local_fwd_layout(True),
                            segment_rule=kp.LOCAL_SEGMENT_RULE,
                        )
                        if c.T_r
                        else 0
                        for c in ctxs
                    )
                    local = []
                    for c in ctxs:
                        if not c.T_r:
                            local.append(0)
                            continue
                        p = cake_backend.plan(
                            c.T_r,
                            c.n_frag,
                            H,
                            c.sm_count,
                            segment_rule=kp.LOCAL_SEGMENT_RULE,
                        )
                        a = kp.local_arena_bytes(p, backward=True, local_arena="lean")
                        if kp.local_lean_serves(p):
                            assert a == kp.local_lean_bytes(p, kp.local_lean_passes(p))
                            a += 4 * kp.local_tail_count_words(a, H)
                        local.append(a)
                    got = _lean(ctx=ctxs, H=H)
                    assert got == max(local) + rest, (W, layout, r, H)
                    for c, a in zip(ctxs, local, strict=True):
                        single = _lean(ctx=c, H=H)
                        assert (
                            single
                            == a
                            + kp.cake_kda_nondet_train_kcp_workspace_bytes(
                                c, H=H, local_arena="replay"
                            )
                            - (
                                cake_backend.workspace_bytes(
                                    c.T_r,
                                    c.n_frag,
                                    H,
                                    sm_count=c.sm_count,
                                    fwd_layout=kp.local_fwd_layout(True),
                                    segment_rule=kp.LOCAL_SEGMENT_RULE,
                                )
                                if c.T_r
                                else 0
                            )
                        )
                        assert _lean(ctx=[c], H=H) == single
                    if H == 12:  # dqkg_int: the replay arena under either option
                        assert got == replay
                    elif any(c.T_r for c in ctxs):
                        assert got < replay
                    for c in ctxs:
                        assert _lean(
                            ctx=c, H=H, backward=False
                        ) == kp.cake_kda_nondet_train_kcp_workspace_bytes(
                            c, H=H, backward=False, local_arena="replay"
                        )
                    seen += 1
    assert seen > 0
    with pytest.raises(ValueError):
        kp.cake_kda_nondet_train_kcp_workspace_bytes(ctxs, H=64, local_arena="tiny")
    with pytest.raises(ValueError):
        cake_kcp.CakeKDANondetKcpWorkspace(local_arena="tiny")
    assert (
        cake_kcp.CakeKDANondetKcpWorkspace().local_arena
        == kp.LOCAL_ARENA_DEFAULT
        == "replay"
    )


# the explicit pass-count variants of the lean arena (``lean_passes``; two is the default): fewer tail launches for
# more bytes, or more launches for fewer
LEAN_PASS_VARIANTS = (1, 3, 4)


def test_lean_passes_knob_bytes_on_the_host() -> None:
    """``lean_passes`` (host only): a lean arena sized for ``b`` passes is ``local_lean_bytes(plan, b)`` with its
    tail-count table; ``None`` is two passes; on the proxy geometry the bytes fall with the pass count down to the
    smallest arena; a forward-only workspace ignores it; the knob is validated and needs the lean arena (the
    public byte query forwards both arguments)."""
    part = _part(PROXY_PACK)
    ctx = kp.cake_kda_nondet_train_kcp_build_context(
        part, None, 0, "cpu", 4, sm_count=152
    )
    kcp_module = pytest.importorskip("cake_pplx.cake_kda_nondet_train_kcp")
    for H in (4, 64):
        p = cake_backend.plan(
            ctx.T_r, ctx.n_frag, H, ctx.sm_count, segment_rule=kp.LOCAL_SEGMENT_RULE
        )
        replay = kp.cake_kda_nondet_train_kcp_workspace_bytes(
            ctx, H=H, local_arena="replay"
        )
        default = _lean(ctx=ctx, H=H)
        assert default == _lean(ctx=ctx, H=H, lean_passes=2) < replay
        sizes = [kp.local_lean_bytes(p, b) for b in range(1, kp.LEAN_MAX_PASSES + 1)]
        best = 1 + sizes.index(min(sizes))
        a2 = kp.local_lean_bytes(p, 2)
        lean2 = a2 + 4 * kp.local_tail_count_words(a2, H)
        prev = None
        for b in range(1, best + 1):
            a = kp.local_lean_bytes(p, b)
            assert kp.local_lean_passes(p, passes=b) == b
            assert (
                kp.local_arena_bytes(
                    p, backward=True, local_arena="lean", lean_passes=b
                )
                == a
            )
            got = _lean(ctx=ctx, H=H, lean_passes=b)
            lean = a + 4 * kp.local_tail_count_words(a, H)
            assert got == default - lean2 + lean, (H, b)
            assert prev is None or got <= prev, (H, b, prev, got)
            assert (
                kcp_module.cake_kda_nondet_train_kcp_workspace_bytes(
                    ctx, H=H, local_arena="lean", lean_passes=b
                )
                == got
            )
            prev = got
            print(f"LEAN_PASSES_HOST H={H} b={b} bytes={got} replay={replay}")
        assert kcp_module.cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=H) == replay
        # no effect off the lean arena's rows: forward-only workspaces
        assert _lean(
            ctx=ctx, H=H, backward=False, lean_passes=1
        ) == kp.cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=H, backward=False)
    for bad in (0, kp.LEAN_MAX_PASSES + 1, 2.0, True, "2"):
        with pytest.raises(ValueError):
            kp.check_lean_passes(bad)
        with pytest.raises(ValueError):
            _lean(ctx=ctx, H=64, lean_passes=bad)
    for arena in ("replay", None):
        with pytest.raises(ValueError):
            cake_kcp.CakeKDANondetKcpWorkspace(local_arena=arena, lean_passes=2)
    assert (
        cake_kcp.CakeKDANondetKcpWorkspace(
            local_arena="lean", lean_passes=3
        ).lean_passes
        == 3
    )
    assert cake_kcp.CakeKDANondetKcpWorkspace(local_arena="lean").lean_passes is None


# --------------------------------------------------------------------------- GPU


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def test_gpu_single_rank_workspace_views_and_route_layouts(device) -> None:
    """The single-rank workspace on the device: without ``layouts`` (the default, lean and tail-pass options) its
    bytes are the plan's ``arena_bytes``, ``layout_passes`` is empty and every view of its layouts lies at the
    plan's region offsets (the option's tail passes for the lean forms, one pass otherwise); with
    ``layouts=((fwd_kcp, 1), (LEAN, b))`` its bytes are exactly the larger of the two, ``views(LEAN)`` resolves at
    ``b`` passes on the plan's regions and ``views("fwd_kcp")`` too, and a layout larger than the arena is
    refused."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    sm, arch = cake_jit.sm_count(index), cake_jit.device_arch(index)

    def check(ws, mode, b) -> None:
        ns = ws.views(mode)
        base = ws.arena.data_ptr()
        for r in ws.plan.layout(ws.arena_option.mode(mode), b):
            t = getattr(ns, r.name)
            assert t.data_ptr() - base == r.offset, (mode, b, r.name)
            assert tuple(t.shape) == r.shape and t.dtype == r.dtype, (mode, r.name)

    for H in (4, 64):
        p = cake_backend.plan(sum(PROXY_PACK), len(PROXY_PACK), H, sm, arch)
        for option in (
            None,
            cake_backend.CakeKDANondetArena(lean=True),
            cake_backend.CakeKDANondetArena(tail_passes=4),
        ):
            ws = cake_backend.CakeKDANondetWorkspace(p, device, arena=option)
            assert ws.layout_passes == {}
            assert ws.bytes == p.arena_bytes(backward=True, arena=option)
            if option is None:
                for mode in ("fwd", "bwd", "replay"):
                    check(ws, mode, 1)
            else:
                check(ws, "bwd", option.tail_passes)  # mapped to bwd_lean
            del ws
        fwd = cake_backend.CakeKDANondetWorkspace(
            p, device, backward=False, fwd_layout="fwd_kcp"
        )
        assert fwd.layout_passes == {} and fwd.bytes == p.workspace_bytes("fwd_kcp")
        check(fwd, "fwd_kcp", 1)
        del fwd
        for b in (1, 2, 4):
            ws = cake_backend.CakeKDANondetWorkspace(
                p,
                device,
                fwd_layout="fwd_kcp",
                layouts=(("fwd_kcp", 1), (LEAN, b)),
            )
            assert ws.layout_passes == {"fwd_kcp": 1, LEAN: b}
            assert (
                ws.bytes
                == max(p.workspace_bytes("fwd_kcp"), p.workspace_bytes(LEAN, b))
                == kp.local_lean_bytes(p, b)
            )
            check(ws, LEAN, b)
            check(ws, "fwd_kcp", 1)
            if p.workspace_bytes("replay") > ws.bytes:
                with pytest.raises(ValueError):
                    ws.views("replay")
            del ws
    torch.cuda.empty_cache()


def _packs_on(device, H: int, packs, seed: int = JUDGE_SEED) -> tuple[list, list]:
    ctxs = [
        kp.cake_kda_nondet_train_kcp_build_context(_part(lengths), None, 0, device)
        for lengths in packs
    ]
    return ctxs, _draw_ops(ctxs, H, device, seed)


def _poison(ws) -> None:
    if ws.local is not None:
        ws.local.arena.fill_(0xFF)


def _lean_used(ws, ctxs) -> list:
    """The tail passes of every pack's arena or view on ``ws`` (``None``: the replay layout)."""
    out = []
    for c in ctxs:
        if not c.T_r:
            continue
        key = (int(c.T_r), int(c.n_frag))
        local = ws.local
        if key != (local.plan.S, local.plan.N):
            local = next(v for k, v in ws._views.items() if k[:2] == key)
        out.append(_passes_of(local))
    return out


UNIT_ROWS = [(H, "packs") for H in CONTRACT_HEADS] + [(64, "proxy"), (4, "proxy")]


@pytest.mark.parametrize("H,which", UNIT_ROWS, ids=[f"h{h}-{w}" for h, w in UNIT_ROWS])
def test_gpu_lean_arena_beside_the_replay_arena_judged(
    device, H, which, record_property
) -> None:
    """The interleaved schedule of the shared-workspace tests (saving and replay-mode forwards, saved and
    replay-mode backwards, forwards without a saved set, checkpointed autograd through all packs) on the replay
    arena, one workspace per pack (``replay``), on a lean workspace per pack (``lean_per``) and on ONE reserved lean
    workspace (``lean_shared``), the lean arenas NaN-filled before every forward: out and every gradient of each
    judged per pack (case ``p<i>``; :func:`judged`).  Every pack ran the lean layout, two tail passes (the default) on
    a pack's own workspace and on the reserved workspace's largest pack (a smaller pack's view takes the fewest passes
    that fit the reserved arena, one where it fits whole); the lean bytes equal
    ``cake_kda_nondet_train_kcp_workspace_bytes`` and are below the replay ones; no pointer moved (every draw)."""
    packs = PACKS if which == "packs" else (PROXY_PACK,)
    ctxs, _ = _packs_on(device, H, packs)
    judge = NaiveJudge()
    ws = ptrs = None
    for j in range(draws_of(packs, H)):
        ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        ref_ws = [_ws("replay") for _ in ctxs]
        ref = _schedule(ctxs, ops, ref_ws)
        replay_bytes = [w.bytes for w in ref_ws]
        assert all(_passes_of(w.local) is None for w in ref_ws)
        del ref_ws
        gc.collect()
        per = [_ws() for _ in ctxs]
        got_per = _schedule(ctxs, ops, per, poison=_poison)
        for c, w in zip(ctxs, per, strict=True):
            assert w.bytes == _lean(ctx=c, H=H)
            assert _passes_of(w.local) == kp.LEAN_PASSES_DEFAULT, (
                H,
                _passes_of(w.local),
            )
        assert all(w.bytes < b for w, b in zip(per, replay_bytes, strict=True)), (
            [w.bytes for w in per],
            replay_bytes,
        )
        del per
        gc.collect()
        if ws is None:
            ws = _ws()
            assert ws.reserve(ctxs, H) == _lean(ctx=ctxs, H=H) == ws.bytes
            ptrs = _pointers(ws)
        used = _lean_used(ws, ctxs)
        assert all(b is not None for b in used) and max(used) == 2, used
        got = _schedule(ctxs, ops, ws, poison=_poison)
        if j == 0:
            print(
                f"LEAN_UNIT H={H} {which} passes={used} lean={ws.bytes} replay_max={max(replay_bytes)} "
                f"NC_pad={[int(c.NC_pad) for c in ctxs]}"
            )
        assert _pointers(ws) == ptrs and ws.bytes == _lean(ctx=ctxs, H=H), j
        arms = [
            {
                **schedule_arms(ref[i], "replay"),
                **schedule_arms(got_per[i], "lean_per"),
                **schedule_arms(got[i], "lean_shared"),
            }
            for i in range(len(ctxs))
        ]
        _judge_packs(judge, j, "", packs, ops, arms, device)
        del ref, got_per, got, arms
    judge.settle(record_property)


@pytest.mark.parametrize("b", LEAN_PASS_VARIANTS, ids=lambda b: f"b{b}")
@pytest.mark.parametrize(
    "H,which", [(64, "proxy"), (4, "proxy"), (4, "packs"), (64, "packs")]
)
def test_gpu_lean_pass_variants_judged(device, H, which, b, record_property) -> None:
    """A lean arena sized for ``b`` tail passes (``lean_passes``), one workspace per pack (``lean_per``) and ONE
    reserved for all packs (``lean_shared``), the arena NaN-filled before every forward, beside the replay arena
    (``replay``): out and every gradient of each judged per pack (case ``p<i>``; :func:`judged`); the largest pack runs
    ``b`` passes; the bytes equal ``cake_kda_nondet_train_kcp_workspace_bytes(..., local_arena="lean",
    lean_passes=b)``; no pointer moved (every draw)."""
    packs = PACKS if which == "packs" else (PROXY_PACK,)
    ctxs, _ = _packs_on(device, H, packs)
    judge = NaiveJudge()
    ws = ptrs = None
    for j in range(draws_of(packs, H)):
        ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        ref = _schedule(ctxs, ops, [_ws("replay") for _ in ctxs])
        gc.collect()
        per = [_ws(lean_passes=b) for _ in ctxs]
        got_per = _schedule(ctxs, ops, per, poison=_poison)
        for c, w in zip(ctxs, per, strict=True):
            assert _passes_of(w.local) == b, (H, which, b, _passes_of(w.local))
            assert w.bytes == _lean(ctx=c, H=H, lean_passes=b)
        del per
        gc.collect()
        if ws is None:
            ws = _ws(lean_passes=b)
            assert (
                ws.reserve(ctxs, H) == _lean(ctx=ctxs, H=H, lean_passes=b) == ws.bytes
            )
            ptrs = _pointers(ws)
        used = _lean_used(ws, ctxs)
        assert all(u is not None for u in used) and max(used) == b, used
        got = _schedule(ctxs, ops, ws, poison=_poison)
        if j == 0:
            print(f"LEAN_PASSES H={H} {which} b={b} passes={used} bytes={ws.bytes}")
        assert _pointers(ws) == ptrs, j
        arms = [
            {
                **schedule_arms(ref[i], "replay"),
                **schedule_arms(got_per[i], "lean_per"),
                **schedule_arms(got[i], "lean_shared"),
            }
            for i in range(len(ctxs))
        ]
        _judge_packs(judge, j, "", packs, ops, arms, device)
        del ref, got_per, got, arms
    judge.settle(record_property)


def _nan_stage(monkeypatch, log: list) -> None:
    """Wrap the kernel backend's stages: before each tail range's fused gradient fill every per-pass buffer with NaN
    (each must be written before it is read); log the ``dav`` stage's record (the lean layout launches no
    ``kcp_dav``)."""
    real = cake_kcp.KernelBackend.stage

    def stage(self, name, call, **extra):
        views = call.scratch.get("views")
        if extra.get("tail_pass") is not None and name == "dqkg":
            for n in PASS_BUFFERS:
                if hasattr(views, n):
                    getattr(views, n).fill_(float("nan"))
                    log.append(n)
        real(self, name, call, **extra)
        if name == "dav":
            log.append("dav_" + str(call.scratch.get("kcp_dav")))

    monkeypatch.setattr(cake_kcp.KernelBackend, "stage", stage)


@pytest.mark.parametrize("H", (4, 16, 64))
def test_gpu_lean_nan_arena_rows(device, H, monkeypatch, record_property) -> None:
    """The lean layout's dead and per-pass bytes carry nothing: the arena NaN-filled before every forward and
    backward, every per-pass buffer NaN-filled before each tail range's fused gradient, no ``kcp_dav`` launch; no NaN
    reaches a gradient of a saved or replay-mode backward (every pack, every draw).  Out and the saved and replay-mode
    gradients on the replay arena (``replay``: ``out`` + ``saved``, ``replay_bwd``) and on the poisoned lean arena
    (``lean_nan``: the same) are judged per pack (case ``p<i>``; :func:`judged`)."""
    packs = PACKS + (PROXY_PACK,) if H == 64 else PACKS
    ctxs, _ = _packs_on(device, H, packs)

    def run(wss, poison, ops) -> list:
        out = []
        for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
            poison(wss[i])
            y, s = _fwd(c, o, wss[i])
            poison(wss[i])
            g = _bwd(c, o, wss[i], s)
            poison(wss[i])
            g_replay = _bwd(c, o, wss[i], None)
            out.append((y, g, g_replay))
        return out

    def arms(prefix: str, r) -> dict:
        y, g, g_replay = r
        return {
            f"{prefix}.saved": {"out": y, **dict(zip(BWD_GRADS, g, strict=True))},
            f"{prefix}.replay_bwd": dict(zip(BWD_GRADS, g_replay, strict=True)),
        }

    log: list = []
    ws = None
    judge = NaiveJudge()
    for j in range(draws_of(packs, H)):
        ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        if ws is not None:
            monkeypatch.undo()  # the replay arena's reference runs without the NaN stage
        ref = run([_ws("replay") for _ in ctxs], lambda w: None, ops)
        log.clear()
        _nan_stage(monkeypatch, log)
        if ws is None:
            ws = _ws()
            ws.reserve(ctxs, H)
        got = run([ws] * len(ctxs), _poison, ops)
        assert "dAqk" in log and "dq_f" in log, sorted(set(log))
        assert "dav_skipped" in log and "dav_launched" not in log, sorted(set(log))
        for i in range(len(ctxs)):
            for g in got[i][1] + got[i][2]:
                if isinstance(g, torch.Tensor) and g.is_floating_point():
                    assert not torch.isnan(g).any(), (H, i, j)
        all_arms = [
            {**arms("replay", ref[i]), **arms("lean_nan", got[i])}
            for i in range(len(ctxs))
        ]
        _judge_packs(judge, j, "", packs, ops, all_arms, device)
        del ref, got, all_arms
    judge.settle(record_property)


@pytest.mark.parametrize("H", (4, 64))
def test_gpu_lean_handles_graph_replay_and_overlap_schedule(
    device, H, record_property
) -> None:
    """The knobs around the backward on the lean arena, each beside the replay arena: recompute handles of both
    tiers through two checkpointed layers per pack in both orders (``direct``: direct autograd on the replay arena;
    ``<tier>.<order>``: the lean arena poisoned; the stored tensors unchanged and outside the workspace, each recompute
    run once per layer), ``graph="replay"`` (eager, captured and replayed calls; ``graph_replay_arena`` and
    ``graph_lean``, repeats) and the overlap schedule (the document-end adjoint tape walk under b2; ``overlap_replay``
    and ``overlap_lean``) through ``_kcp_test_knobs``.  Out and every gradient of each judged per pack (case
    ``p<i>``; the handle arms against the FP64 reference of the two-layer stack, :func:`stack_refs`, beside FLA's
    stack, :func:`stack_fla`; :func:`judged`)."""
    packs = PACKS + (PROXY_PACK,) if H == 64 else PACKS
    ctxs, _ = _packs_on(device, H, packs)

    def graph(wss, ops) -> list:
        res = [[] for _ in ctxs]
        for _ in range(3):  # eager, capture, replay
            for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
                y, s = _fwd(c, o, wss[i], graph="replay")
                g = _bwd(c, o, wss[i], s, graph="replay")
                res[i].append(host({"out": y, **dict(zip(BWD_GRADS, g, strict=True))}))
        return res

    def overlap(wss, ops) -> list:
        knobs = {"_kcp_test_knobs": {"schedule": "overlap"}}
        res = []
        for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
            y, s = _fwd(c, o, wss[i], **knobs)
            g = _bwd(c, o, wss[i], s, **knobs)
            res.append({"out": y, **dict(zip(BWD_GRADS, g, strict=True))})
        return res

    judge = NaiveJudge()
    previous = kp.enable_test_knobs(True)
    try:
        for j in range(draws_of(packs, H)):
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
            ref, _ = _handle_pass(ctxs, ops, [_ws("replay") for _ in ctxs])
            stack = [handle_arms(ref[i], "direct") for i in range(len(ctxs))]
            del ref
            for keep in ("states", "transfer"):
                for order in ("interleaved", "sequential"):
                    ws = _ws()
                    ws.reserve(ctxs, H)
                    got, log = _handle_pass(
                        ctxs, ops, ws, keep=keep, order=order, poison=_poison
                    )
                    assert not (
                        log["changed"] or log["overlap"] or log["recompute_out"]
                    ), (H, keep, order, log)
                    for i in range(len(ctxs)):
                        stack[i].update(handle_arms(got[i], f"{keep}.{order}"))
                    del got, log, ws
            cake_kcp.graph_cache_clear()
            graph_ref = graph([_ws("replay") for _ in ctxs], ops)
            ws = _ws()
            ws.reserve(ctxs, H)
            graph_lean = graph([ws] * len(ctxs), ops)
            cake_kcp.graph_cache_clear()
            overlap_ref = overlap([_ws("replay") for _ in ctxs], ops)
            ws = _ws()
            ws.reserve(ctxs, H)
            overlap_lean = overlap([ws] * len(ctxs), ops)
            del ws
            arms = [
                {
                    "graph_replay_arena": graph_ref[i],
                    "graph_lean": graph_lean[i],
                    "overlap_replay": overlap_ref[i],
                    "overlap_lean": overlap_lean[i],
                }
                for i in range(len(ctxs))
            ]
            _judge_packs(judge, j, "", packs, ops, arms, device)
            for i, lengths in enumerate(packs):
                if judged(lengths, H):
                    inputs = pack_inputs(_part(lengths), _cu(lengths), ops[i], device)
                    r, t = stack_refs(judge, inputs, HANDLE_LAYERS)
                    judge.add_refs(
                        j, f"p{i}", stack[i], r, t, stack_fla(judge, inputs, HANDLE_LAYERS)
                    )
            del stack, graph_ref, graph_lean, overlap_ref, overlap_lean, arms
    finally:
        kp.enable_test_knobs(previous)
        cake_kcp.graph_cache_clear()
    judge.settle(record_property)


def test_gpu_lean_steady_calls_do_not_block_or_allocate(
    device, record_property
) -> None:
    """Census of a steady round (saving forwards, then saved-mode backwards in reverse order) of the packs and the
    proxy-geometry pack at H = 64 on one reserved lean workspace (two tail passes on the largest pack; the first
    draw): no pageable or pinned copy, no synchronizing operation, no device allocation or free (the tail ranges'
    item bounds come from the arena's tail-count table).  Out and the five gradients of the first round (``first``)
    and of the steady round (``steady``) are judged per pack on every draw (case ``p<i>``; the proxy pack keeps its
    exact checks only, :func:`judged`)."""
    alloc_ok = allocator_counts(device)
    H = 64
    packs = PACKS + (PROXY_PACK,)
    ctxs, ops = _packs_on(device, H, packs)
    ws = _ws()
    ws.reserve(ctxs, H)
    used = _lean_used(ws, ctxs)
    assert all(b is not None for b in used) and max(used) > 1, used

    def rnd():
        saved, res = [], [{} for _ in ctxs]
        for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
            out, s = _fwd(c, o, ws)
            saved.append(s)
            res[i]["out"] = out
        for i in reversed(range(len(ctxs))):
            grads = _bwd(ctxs[i], ops[i], ws, saved[i])
            res[i].update(zip(BWD_GRADS, grads, strict=True))
        return res

    judge = NaiveJudge()
    for j in range(draws_of(packs, H)):
        if j:
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        first = rnd()
        rnd()
        if j == 0:
            c = census(rnd, device)
            print(f"CENSUS_LEAN {summary(c)}")
            assert nonblocking(c), summary(c)
            assert c["pinned"] == 0, summary(c)
            if alloc_ok:
                assert (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (
                    0,
                    0,
                    0,
                ), summary(c)
                assert c["prof_malloc"] in (None, 0), summary(c)
            steady = c["out"]
            del c
        else:
            steady = rnd()
        arms = [{"first": a, "steady": b} for a, b in zip(first, steady, strict=True)]
        _judge_packs(judge, j, "", packs, ops, arms, device)
        del first, steady, arms
    judge.settle(record_property)
    if not alloc_ok:
        pytest.skip(
            "the caching allocator's statistics do not count device allocations here"
        )


# --------------------------------------------------------------------------- multi-GPU

# (packs, H, W, layout, exchange): two-rank rows (P > 1 at H = 4, P = 1 at H = 64, the ring) and the four- and
# eight-rank rows of the shared-workspace tests (the eight-rank rows on two nodes as one process per Slurm task)
LEAN_DIST_ROWS = [
    (
        ((3000,), (2048, 5, 3000, 1), (700, 1, 1300, 999)),
        4,
        2,
        "zigzag_per_document",
        "a2a",
    ),
    (((100, 640, 77, 1000, 129), (2000, 1000), (3000,)), 64, 2, "contiguous", "a2a"),
    (((1000, 1000), (1500, 1500), (500, 2500)), 4, 2, "contiguous", "ring"),
    *DIST_ROWS,
]


def _run_lean_rank(
    rank: int,
    W: int,
    row,
    rdzv: str | None,
    result_dir: str | None,
    lean_passes: int | None = None,
) -> dict:
    """One rank, over the judge's draws: the row's packs on the replay arena, one workspace per pack (``replay``),
    then on ONE reserved lean workspace (two rounds of the interleaved schedule, the arena NaN-filled before every
    forward of the first: ``lean_poisoned``, ``lean``); the bytes equal ``cake_kda_nondet_train_kcp_workspace_bytes``
    and below the replay ones, no pointer moved, every pack of the rank on the lean layout.  Out and the five gradients
    of every arm of every pack are gathered to rank 0 and judged there (``report["naive"]``).  The flags are gathered
    to every rank."""
    packs, H, _w, _layout, exchange = row
    device, ctxs, parts = _rank_packs(rank, W, row, rdzv)
    kw = {"exchange": exchange}
    ring = exchange == "ring"
    rid = (
        "lean-"
        + _dist_row_id(row)
        + ("" if lean_passes is None else f"-b{lean_passes}")
    )
    report: dict = {
        "rank": rank,
        "row": rid,
        "lean_passes": lean_passes,
        "pass": True,
        "fail": [],
    }
    judge = NaiveJudge() if rank == 0 else None
    ws = ptrs = None
    for j in range(draws_of(packs, H)):
        ops, globs = _rank_ops(packs, parts, rank, H, device, draw_seed(JUDGE_SEED, j))
        ref_ws = [_ws("replay") for _ in ctxs]
        ref = _schedule(ctxs, ops, ref_ws, **kw)
        if ws is None:
            report["replay_bytes"] = [w.bytes for w in ref_ws]
        del ref_ws
        gc.collect()
        if ws is None:
            ws = _ws(lean_passes=lean_passes)
            ws.reserve(ctxs, H, ring=ring)
            report["lean_bytes"] = ws.bytes
            report["pred_bytes"] = _lean(ctx=ctxs, H=H, lean_passes=lean_passes) + (
                sum(
                    max(kp.kcp_arena_layout(c, H=H)[f"pool.{name}"] for c in ctxs)
                    for name in kp.LAZY_POOLS
                )
                if ring
                else 0
            )
            report["passes"] = _lean_used(ws, ctxs) if ws.local is not None else []
            ptrs = _pointers(ws)
        got = []
        for rnd in range(2):
            got.append(
                _schedule(ctxs, ops, ws, poison=_poison if rnd == 0 else None, **kw)
            )
            if _pointers(ws) != ptrs:
                report["fail"].append(f"pointers moved d{j} r{rnd}")
        for i, (lengths, (part, cu)) in enumerate(zip(packs, parts, strict=True)):
            if not judged(lengths, H):
                continue
            arms = {
                **schedule_arms(ref[i], "replay"),
                **schedule_arms(got[0][i], "lean_poisoned"),
                **schedule_arms(got[1][i], "lean"),
            }
            glob = globs[i] if globs is not None else None
            _judge_rank_arms(judge, j, f"p{i}", arms, part, cu, glob, W, rank)
        del ref, got, ops, globs
    if report["lean_bytes"] != report["pred_bytes"]:
        report["fail"].append(
            "bytes != cake_kda_nondet_train_kcp_workspace_bytes(ctxs)"
        )
    if ws.local is not None and not report["lean_bytes"] < max(report["replay_bytes"]):
        report["fail"].append("lean bytes not below the replay workspace")
    if any(b is None for b in report["passes"]):
        report["fail"].append(f"a pack off the lean layout: {report['passes']}")
    return _finish_rank_report(report, judge, W, rank, result_dir, rid)


def _bound_rank_threads(W: int) -> None:
    """Give a spawned rank its share of the host's CPUs: each of the ``W`` ranks would
    otherwise start as many threads as the host has, and rank 0's FP64 reference then
    ran past the process group's ten-minute collective timeout while the other ranks
    waited (813 s on a four-rank row that takes 22 s with the share)."""
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // W))


def _spawn_lean_entry(
    rank: int,
    W: int,
    row,
    rdzv: str | None,
    result_dir: str,
    lean_passes: int | None = None,
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_lean_rank(rank, W, row, rdzv, result_dir, lean_passes)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("row", LEAN_DIST_ROWS, ids=_dist_row_id)
def test_multi_gpu_lean_rows_judged(row, record_property) -> None:
    """The rows of :func:`_run_lean_rank`: one process per GPU (``spawn``, rows up to the visible GPU count) or one
    process per Slurm task (a step with ``SLURM_NTASKS`` > 1 runs the rows of its own world size and skips the
    others); rank 0's verdicts recorded and settled here."""
    _run_lean_row(row, None, record_property)


# the explicit pass-count variants on a two-rank row each of a2a and the ring, a four-rank row and an eight-rank row
LEAN_PASS_DIST_ROWS = [
    (row, b)
    for b in (3, 4)
    for row in (LEAN_DIST_ROWS[0], LEAN_DIST_ROWS[2], DIST_ROWS[0], DIST_ROWS[-1])
]


@pytest.mark.parametrize(
    "row,b",
    LEAN_PASS_DIST_ROWS,
    ids=[f"b{b}-{_dist_row_id(r)}" for r, b in LEAN_PASS_DIST_ROWS],
)
def test_multi_gpu_lean_pass_variant_rows_judged(row, b, record_property) -> None:
    """:func:`_run_lean_rank` on a lean arena sized for ``b`` tail passes (``lean_passes``)."""
    _run_lean_row(row, b, record_property)


def _run_lean_row(row, lean_passes: int | None, record_property) -> None:
    _packs_, _H, W, _layout, _exchange = row
    rid = (
        "lean-"
        + _dist_row_id(row)
        + ("" if lean_passes is None else f"-b{lean_passes}")
    )
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"this Slurm step runs the W = {ntasks} rows")
        report = _run_lean_rank(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
            lean_passes,
        )
        assert report["pass"], report
        if "naive" in report:  # rank 0
            settle_naive(record_property, report["naive"])
        return
    _device_or_skip()
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_kcp_lean_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_lean_entry,
            args=(W, row, _rendezvous(tmp), result_dir, lean_passes),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, rid + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], {k: v for k, v in report.items() if k != "naive"}
    settle_naive(record_property, report["naive"])
