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

Multi-GPU rows of the context-parallel (KCP) packed GDN training route (``cake_gdn_det_train_kcp.cake_kcp``): ``W`` ranks over
NCCL, each holding the fragments of one partition of a packed batch, judged against the exact FP64 per-document
recurrence of the same operands and compared with one another bit for bit.  Every row runs ``DRAWS`` (32) independent
draws of its operands (the seeded generator at ``draw_seed(SEED, j)``; draw 0 is the generator's own seed) inside one
process group.  Per row the test checks

* accuracy (rank 0, the outputs reassembled to the global order): ``out, dq, dk, dv`` (BF16) and ``dg, dbeta`` (FP32)
  judged by the naive BF16 tier of ``tests/cake_naive.py``: their error against the FP64 oracle
  (``gdn_exact_with_grads``) beside the error of the PyTorch BF16 recurrence on the same operands
  (``gdn_naive_with_grads``), median and largest over the draws, L2 and max norm; the forward-only rows judge ``out``
  and the crossing-in fragments' carried states ``S_in`` (``gdn_exact_forward`` / ``cut_states`` and their naive
  tiers); rank 0 records one ``naive_tier`` verdict per output; the ``p4`` rows also judge the forced ``P = 4`` plan
  (``segments=4``) as ``p4.<output>``, its ``P`` read from the launch records;
* route (the production-size row PROD-C4, every draw): the shipped plan rule segments every rank's fragment
  (``P > 1``) and the forward and backward launch records ran that plan;
* bitwise on every draw: repeat; saved == replay (``boundary_mode``); ``save_mode`` light == full; autograd ==
  explicit; ``torch.utils.checkpoint(use_reentrant=False)`` == direct; ``save_transfer`` True == False; ring == a2a
  where the partition allows the ring; strided packed-QKV views == contiguous operands;
* bitwise on draw 0 (G3): the checkpoint's full / replay / kept-tape / ``"auto"`` forms and the recompute handle ==
  direct; ``save_mode="full"`` with ``save_transfer=False`` (a2a and, where allowed, ring) == the
  light form with every package allocation filled with NaN bytes, at the plan's ``P`` and at the forced ``P = 2``
  (``segments=2``, where every rank's fragments take it; the full saved set holds ``M_seg`` there); the copy layout
  of strided packed-QKV views == contiguous operands (under tma the backward writes one packed dQKV row, and
  ``cake_gdn_det_train_kcp_packed_qkv_views`` hands it to a packed leaf without a concatenation; plain ``split``
  views of the leaf, direct and checkpointed, keep torch's split backward and their hooks, issue #8367);
  ``.backward()`` twice accumulates exactly twice the gradient;
* semantics (G4, draw 0): zero-gradient leak (``dout`` zero outside one document -> exactly zero gradients on the
  others) and independence (the other documents' operands replaced -> one document's outputs bitwise unchanged);
* contracts (G10, draw 0): #7039 retained bytes after a checkpointed forward (``graph="off"``), the exchange byte counts
  against ``cake_gdn_det_train_kcp_comm_bytes``, CUDA-graph capture of forward + backward (no steady-state D2H / sync),
  the #7038 side-stream ratio of the backward (recorded, never a gate).

Two launch forms: ``torch.multiprocessing.spawn`` (``pytest`` on a node with >= ``W`` GPUs) and one process per Slurm
task (``SLURM_PROCID`` / ``SLURM_NTASKS``: the WORLD group is initialised once per process, every row runs on a
sub-group of its ``W`` lowest ranks; the other ranks skip the row).  ``KCP_TEST_BACKEND`` selects ``kernels`` |
``fake`` (default: ``kernels`` when the package's ``build.json`` lists the programs, else ``fake``);
``KCP_TEST_ROWS`` selects rows by name (comma list; ``all``) or by stage (``int1`` CP <= 4, ``int3`` CP8, ``int4`` CP16,
``prod`` the production-size row); ``KCP_TEST_RESULT_DIR`` receives one JSON per row.
Operands come from the seeded CPU generator (bitwise identical on every rank and host).  The row shapes are sized
for 32 draws of the FP64 oracle and its naive tier on rank 0 (at most about 4k tokens per row; the production-size
row PROD-C4 holds 8,191 tokens, see the rows block).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import gc
import importlib
import json
import os
import socket
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed
from tests.experimental.test_helpers.kcp_collective_shims import (
    CollectiveCounter,
    RecomputePhase,
)

PACKAGE = "cake_pplx._backends.cake_gdn_det_train_kcp"


def _mod(sub: str | None = None):
    try:
        from tests.cake.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_det_train_kcp", sub)
    except (ImportError, KeyError):
        return importlib.import_module(PACKAGE if sub is None else f"{PACKAGE}.{sub}")


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
PACKAGE_DIR = Path(kcp.__file__).resolve().parent
TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta")
GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd order of the leaves (q, k, v, g, beta)
WORLD_TIMEOUT_S = 300  # NCCL / rendezvous timeout of the WORLD group (printed at init)
ROW_TIMEOUT_S = 3600  # gloo timeout of a row's verdict exchange (``_Slurm.cpu``), which covers the longest row
SCALE = kcp.SCALE
C_CONV = 10240
ALLOC_GRANULE = 512  # the caching allocator rounds every block to 512 B
SIDE_STREAM_RATIO_MAX = 1.10  # #7038; the per-row ratio is recorded, never a gate (the module-timing protocol decides)
SIDE_STREAM_ITERS = 7


def _reference():
    """The FP64 oracle and its naive BF16 tier (the in-tree test helper)."""
    from tests.experimental.test_helpers import cake_gdn_det_train_kcp_reference as ref

    return ref


def _inputs():
    """The seeded operand generator (``make_inputs(cu, H_qk, HV, *, regime, seed, token_range)``)."""
    try:
        from tests.experimental.test_helpers import cake_gdn_det_train_kcp_inputs as gen
    except ImportError as exc:
        pytest.skip(f"the seeded GDN KCP inputs helper is unavailable: {exc}")
    return gen


def _draw_seed(j: int) -> int:
    """The operand seed of draw ``j``: the generator's own seed for draw 0, far apart for the others."""
    return draw_seed(int(_inputs().SEED), j)


# ----------------------------------------------------------------------------------------------------------------
# the judge (rank 0): every floating output against the FP64 oracle beside the naive BF16 tier (tests/cake_naive.py)


def _judge_add(
    judge: TierJudge,
    failures: list[str],
    name: str,
    j: int,
    actual: torch.Tensor,
    oracle: torch.Tensor,
    naive: torch.Tensor,
) -> None:
    """Draw ``j`` of output ``name``: held to its contract against the oracle and added to the judge with the naive
    tier's value.  The oracle and the naive value are laid out as ``actual`` (the reassembled candidate) when they
    hold as many elements; a shape mismatch is a contract failure and is not judged."""
    if oracle.numel() == actual.numel():
        oracle = oracle.reshape(actual.shape)
        naive = naive.reshape(actual.shape)
    if (m := contract(name, actual, oracle)) is not None:
        failures.append(f"draw {j}: {m}")
        if actual.shape != oracle.shape:
            return
    judge.add(name, j, actual, oracle, {"bf16": naive})


def _judge_report(report: dict, judge: TierJudge | None, failures: list[str]) -> None:
    """Rank 0: the row's verdicts (``naive_tier``, one per judged output) and its failures (contracts, then the
    judge's), folded into ``report["pass"]``."""
    if judge is None:
        return
    failures = failures + judge.failures()
    report["naive_tier"] = list(judge.verdicts().values())
    report["failures"] = failures
    report["pass"] &= not failures


def _record_verdicts(report: dict | None, record_property) -> None:
    """The test process records rank 0's verdicts (pytest ``record_property``, read by the conftest's ``TIER``
    lines)."""
    if record_property is None:
        return
    for verdict in (report or {}).get("naive_tier", []):
        record_property("naive_tier", verdict)


def _gather(obj, W: int, group) -> list:
    """``obj`` of every rank of the group in rank order (``all_gather_object``; ``[obj]`` at ``W = 1``)."""
    if W == 1:
        return [obj]
    objs: list = [None] * W
    dist.all_gather_object(objs, obj, group=group)
    return objs


# ----------------------------------------------------------------------------------------------------------------
# rows (C1-C9 layout rows, A1-A4 adversarial-regime rows, the no-leak rows; the int4 stage = the CP16 rows of G4: at
# least one row at W = 16, here the C6 zigzag row and the A1 contiguous row at the (2, 6) head pair).  Every row is
# judged the same way (the naive BF16 tier over 32 draws); the shapes keep each row's layout features at a size the
# 32 draws of the FP64 oracle allow: several documents of odd and chunk-aligned lengths, fragments of more than one
# 64-token chunk with a partial last chunk, a 1-token rank and an empty rank where the row had them, the same
# document chains (hop counts, so the same rows take the ring), grouped heads HV / H_qk = 3, and (16, 48) heads.
# C6-cp16: 32 zigzag pieces of 66 / 67 tokens, so every rank's piece of the document holds one full 64-token chunk.
#
# The prod stage: the production-size row PROD-C4 on its production route.  At the production heads (16, 48) the shipped
# plan rule (``small_one``, from ``row_cost_lpt``) segments the fragment of a one-fragment rank into the one-wave fill
# ``P = floor(SM / HV)`` (3 on 148 / 152 SMs, 4 on 212 SMs: the segment transfer walk, the carry merges and the compose
# chain run) once that beats ``P = 1`` in its cost model, i.e. from about 17 (148 / 152 SMs) or 23 (212 SMs) chunks per
# fragment; below that it plans ``P = 1`` (as on the two-fragment ranks of a zigzag partition at 148 / 152 SMs, where
# ``P = 2`` needs a second wave).  The ~1k-token (16, 48) rows C6 / C8 therefore run ``P = 1``.  PROD-C4 is one
# 8,191-token document over 4 contiguous ranks (a node of 4 GPUs): a 32-chunk fragment per rank (the last one 2,047
# tokens, a partial last chunk), clear of both thresholds, the document crossing every rank (a2a exchange).  Its route
# is gated on every draw (``prod_route_segmented``): the shipped rule's plan has ``P > 1`` and the forward and backward
# launch records ran that ``P`` on every rank.  The FP64 oracle and its naive tier cost about T x 16 ms for the 32 draws
# (about 2.2 minutes on rank 0; more at HV = 48 if the per-token ops stop being launch bound).
#
# The ``p4`` rows also run the forced ``P = 4`` plan (``segments=4``, a split plan of more than two segments; the
# shipped rule plans ``P = 1`` on the (4, 12) and (2, 6) rows and 3 on PROD-C4 at 148 / 152 SMs) on every draw,
# judged against the same oracle as ``p4.<output>``; the forward and backward launch records must report ``P = 4``
# (``p4_launch_P``), and every rank's longest fragment must allow it (``p4_allowed``: at least 4 chunks).


@dataclasses.dataclass(frozen=True)
class Row:
    name: str
    lengths: tuple[int, ...]
    heads: tuple[int, int]  # (H_qk, HV)
    W: int
    layout: str  # contiguous | zigzag | zigzag_per_document | ragged:<T_r list> | explicit:<name>
    regime: str = "normal"
    leak: bool = False  # run the zero-gradient leak / independence checks
    stage: str = "int1"  # int1 (CP <= 4, small), int3 (CP8 / large), int4 (CP16), prod (production size)
    p4: bool = False  # also judge the forced P = 4 plan (segments=4)


def _c5_fragments() -> list[list[tuple[int, int, int]]]:
    """C5: fragments of exactly 63 / 64 / 65 tokens at cut points with offsets ``1`` and ``63 (mod 64)``: document 0 =
    [63, 64, 65, 66] (offsets 0, 63, 127, 192), document 1 = [1, 64, 63, 65] (offsets 0, 1, 65, 128); rank ``r`` holds
    piece ``r`` of both documents."""
    pieces = [(63, 64, 65, 66), (1, 64, 63, 65)]
    frags: list[list[tuple[int, int, int]]] = [[] for _ in range(4)]
    for doc, lens in enumerate(pieces):
        off = 0
        for r, n in enumerate(lens):
            frags[r].append((doc, off, n))
            off += n
    return frags


EXPLICIT = {"c5": (_c5_fragments, (63 + 64 + 65 + 66, 1 + 64 + 63 + 65))}

ROWS = [
    Row("C1-cp2", (100, 290, 77, 420, 129), (2, 6), 2, "contiguous", leak=True),
    Row("C1-cp4", (100, 290, 77, 420, 129), (2, 6), 4, "contiguous"),
    Row("C2-C", (1021,), (4, 12), 4, "contiguous", p4=True),
    Row("C2-Z", (1021,), (4, 12), 4, "zigzag"),
    Row("C2-D", (1021,), (4, 12), 4, "zigzag_per_document"),
    Row("C3", (1024, 5, 1001, 1), (1, 3), 4, "zigzag_per_document", leak=True),
    Row("C4a", (500, 520), (8, 24), 4, "ragged:350,1,370,299"),
    Row("C4b", (500, 520), (8, 24), 4, "ragged:600,0,210,210"),
    Row("C5", EXPLICIT["c5"][1], (4, 4), 4, "explicit:c5"),
    Row("C9a", (1,), (2, 6), 2, "contiguous"),
    Row("C9b", (0,), (2, 6), 2, "contiguous"),
    Row("leak-C", (300, 500, 200), (2, 6), 2, "contiguous", leak=True),
    Row("leak-D", (300, 500, 200), (2, 6), 2, "zigzag_per_document", leak=True),
    Row(
        "A3",
        (2017, 1985),
        (2, 6),
        4,
        "zigzag_per_document",
        regime="A3",
        stage="int3",
    ),
    Row("C6", (1041,), (16, 48), 8, "zigzag", stage="int3"),
    Row("C8", (512, 448, 101), (16, 48), 8, "contiguous", stage="int3"),
    Row("A1", (4065,), (2, 6), 8, "contiguous", regime="A1", stage="int3", p4=True),
    Row("A2", (4000,), (2, 6), 8, "contiguous", regime="A2", stage="int3"),
    Row("A4", (4095,), (2, 6), 8, "contiguous", regime="A4", stage="int3"),
    Row("C6-cp16", (2113,), (16, 48), 16, "zigzag", stage="int4"),
    Row("A1-cp16", (4065,), (2, 6), 16, "contiguous", regime="A1", stage="int4"),
    Row("PROD-C4", (8191,), (16, 48), 4, "contiguous", stage="prod", p4=True),
]
# the forced segment count of the p4 rows
P4 = 4


def _selected_rows() -> list[Row]:
    sel = os.environ.get("KCP_TEST_ROWS", "int1")
    if sel == "all":
        return list(ROWS)
    if sel in ("int1", "int3", "int4", "prod"):
        return [r for r in ROWS if r.stage == sel]
    names = {s.strip() for s in sel.split(",") if s.strip()}
    rows = [r for r in ROWS if r.name in names]
    missing = names - {r.name for r in rows}
    if missing:
        raise ValueError(f"KCP_TEST_ROWS names unknown rows {sorted(missing)}")
    return rows


def _row_id(row: Row) -> str:
    return (
        f"{row.name}-w{row.W}-h{row.heads[0]}x{row.heads[1]}-{row.layout.split(':')[0]}"
    )


# ----------------------------------------------------------------------------------------------------------------
# partition / slicing helpers


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


def _ragged(cu: list[int], t_r: list[int]) -> list[list[tuple[int, int, int]]]:
    """Contiguous ragged split of the global buffer into per-rank token counts, cut at document boundaries."""
    nested: list[list[tuple[int, int, int]]] = [[] for _ in t_r]
    pos = 0
    for r, n in enumerate(t_r):
        end = pos + n
        d = 0
        while d < len(cu) - 1 and cu[d + 1] <= pos:
            d += 1
        p = pos
        while p < end and d < len(cu) - 1:
            e = min(end, cu[d + 1])
            if e > p:
                nested[r].append((d, p - cu[d], e - p))
            p = e
            d += 1
        pos = end
    return nested


def _partition(row: Row):
    cu = _cu(row.lengths)
    if row.layout.startswith("ragged:"):
        t_r = [int(x) for x in row.layout.split(":")[1].split(",")]
        assert len(t_r) == row.W and sum(t_r) == cu[-1]
        return plan.cake_gdn_det_train_kcp_partition(
            cu, row.W, fragments=_ragged(cu, t_r)
        ), cu
    if row.layout.startswith("explicit:"):
        make, lengths = EXPLICIT[row.layout.split(":")[1]]
        assert tuple(lengths) == tuple(row.lengths)
        return plan.cake_gdn_det_train_kcp_partition(cu, row.W, fragments=make()), cu
    return plan.cake_gdn_det_train_kcp_partition(cu, row.W, layout=row.layout), cu


def _programs_exported() -> bool:
    """Every shipped record of the registry has a library in the package's ``build.json`` -- else the kernel path
    cannot run."""
    manifest = PACKAGE_DIR / "build.json"
    if not manifest.exists():
        return False
    try:
        with open(manifest) as fh:
            libraries = {lib["name"] for lib in json.load(fh)["libraries"]}
        jit = _mod("cake_jit")
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False
    if not libraries or not hasattr(jit, "MODULES"):
        return False
    try:
        return all(
            any(jit.library_name(name, arch) in libraries for arch in record["arches"])
            for name, record in jit.MODULES.items()
        )
    except Exception:  # noqa: BLE001 - a registry without library_name: the manifest's names decide
        return bool(libraries)


def _forward_arena_tapes(backend: str, device, T_r: int) -> tuple[str, ...]:
    """The forward tapes an eager light, saved-boundary forward leaves in a caller's arena: the output walk's two
    (``FWD_TAPES_ARENA_AFTER_FUSED``) where the forward folds #5 into its tape walks (``cake_kcp._fuse_o``: tokens and
    the registry's output-walk records for this backend's architecture), else the four #5 reads."""
    arch = None if backend == "fake" else _mod("cake_backend").device_arch(device.index)
    if T_r > 0 and kcp.fused_o_shipped(arch):
        return kcp.FWD_TAPES_ARENA_AFTER_FUSED
    return kcp.FWD_TAPES_ARENA_AFTER_FWD


def _backend_name() -> str:
    name = os.environ.get("KCP_TEST_BACKEND")
    if name:
        return name
    return "kernels" if _programs_exported() else "fake"


def _local_inputs(
    row: Row, part, cu: list[int], rank: int, device, *, seed=None, doc_override=None
):
    """This rank's operands (fragments in slot order) generated per fragment range (bitwise the global slice);
    ``doc_override = (doc, other_seed)`` takes every document except ``doc`` from ``other_seed`` (independence)."""
    gen = _inputs()
    H_qk, HV = row.heads
    parts: dict[str, list[torch.Tensor]] = {
        n: [] for n in ("q", "k", "v", "g", "beta", "dout")
    }
    for f in part.fragments[rank]:
        if f.length == 0:
            continue
        a = cu[f.doc] + f.off
        kw = {"regime": row.regime, "token_range": (a, a + f.length)}
        if seed is not None:
            kw["seed"] = seed
        if doc_override is not None and f.doc != doc_override[0]:
            kw["seed"] = doc_override[1]
        got = gen.make_inputs(cu, H_qk, HV, **kw)
        for n in parts:
            parts[n].append(got[n].reshape((-1,) + tuple(got[n].shape[2:])))
    out = {}
    for n, lst in parts.items():
        if lst:
            out[n] = torch.cat(lst, 0).to(device)
        else:
            shape = {
                "q": (0, H_qk, 128),
                "k": (0, H_qk, 128),
                "v": (0, HV, 128),
                "g": (0, HV),
                "beta": (0, HV),
                "dout": (0, HV, 128),
            }[n]
            dtype = torch.float32 if n in ("g", "beta") else torch.bfloat16
            out[n] = torch.empty(shape, dtype=dtype, device=device)
    return out


def _global_inputs(row: Row, cu: list[int], device, *, seed=None):
    gen = _inputs()
    H_qk, HV = row.heads
    kw = {"regime": row.regime}
    if seed is not None:
        kw["seed"] = seed
    got = gen.make_inputs(cu, H_qk, HV, **kw)
    return {
        n: got[n].reshape((-1,) + tuple(got[n].shape[2:])).to(device)
        for n in ("q", "k", "v", "g", "beta", "dout")
    }


def _to_global(
    part, rank_tensors: list[torch.Tensor], cu: list[int], total: int
) -> torch.Tensor:
    ref = rank_tensors[0]
    g = torch.zeros((total,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)
    for r, t in enumerate(rank_tensors):
        pos = 0
        for f in part.fragments[r]:
            if f.length:
                g[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] = t[
                    pos : pos + f.length
                ]
            pos += f.length
    return g


def _local_to_global(part, cu: list[int], rank: int) -> torch.Tensor:
    """Global token index of every local token of ``rank`` (fragment order), CPU int64."""
    pieces = [
        torch.arange(cu[f.doc] + f.off, cu[f.doc] + f.off + f.length)
        for f in part.fragments[rank]
        if f.length
    ]
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.int64)


def _doc_mask_local(part, cu: list[int], rank: int, doc: int, device) -> torch.Tensor:
    """Boolean ``[T_r]``: the local tokens of document ``doc``."""
    pieces = []
    for f in part.fragments[rank]:
        pieces.append(torch.full((f.length,), f.doc == doc, dtype=torch.bool))
    m = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.bool)
    return m.to(device)


def _packed_views(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Strided packed-QKV views ``[T, ., 128]`` of one ``[T, (2 H_qk + HV) * 128]`` buffer holding ``q | k | v``."""
    T, H_qk = q.shape[0], q.shape[1]
    HV = v.shape[1]
    buf = torch.empty((T, (2 * H_qk + HV) * 128), dtype=q.dtype, device=q.device)
    qv = buf[:, : H_qk * 128].view(T, H_qk, 128)
    kv = buf[:, H_qk * 128 : 2 * H_qk * 128].view(T, H_qk, 128)
    vv = buf[:, 2 * H_qk * 128 :].view(T, HV, 128)
    qv.copy_(q)
    kv.copy_(k)
    vv.copy_(v)
    return buf, qv, kv, vv


def _eq(a: dict, b: dict, names=TENSORS) -> bool:
    return all(torch.equal(a[n], b[n]) for n in names)


@contextlib.contextmanager
def _nan_allocations():
    """Every tensor the package allocates (its ``_alloc``: outputs, saved set, pools, tapes) is filled with NaN
    bytes while the block runs: a program that reads a plane no program of the call wrote gives non-finite results
    instead of the bytes a previous call left in the caching allocator."""
    real = kcp._alloc

    def nan_alloc(shape, dtype, device):
        t = real(shape, dtype, device)
        if t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    kcp._alloc = nan_alloc
    try:
        yield
    finally:
        kcp._alloc = real


def _m_seg_rule(saved) -> dict:
    """The ``M_seg`` record of a ``save_mode="full"`` saved set: present exactly at ``P > 1`` (the full backward runs no
    transfer walk, so its merges read the saved ``M_seg^T`` plane)."""
    if saved is None:
        return {"ok": False, "P": None, "m_seg_bytes": None}
    m = saved.M_seg
    ok = (m is not None) == (int(saved.P) > 1) and saved.save_mode == "full"
    return {
        "ok": bool(ok),
        "P": int(saved.P),
        "m_seg_bytes": None if m is None else int(m.numel() * m.element_size()),
    }


def _p_group_ok(ctx, HV: int, P: int, group, W: int, device: torch.device) -> bool:
    """Every rank's fragments take the segments override ``P`` (the longest fragment of each rank has at least ``P``
    chunks).  Many rows plan ``P = 1``; the forced ``P = 2`` forms put the full backward's read of the saved
    ``M_seg^T`` plane on them, the forced ``P = 4`` arm a split plan of more than two segments.  A group decision (MIN
    over the ranks): every rank runs the same calls."""
    try:
        ok = int(ctx.heads(HV, P=P).P) == P
    except ValueError:
        ok = False
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    if W > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _p2_full_flags(
    saved_pass, info, flags, *, ring: bool, prefix: str = "", **extra
) -> None:
    """The forced ``P = 2`` forms without ``save_transfer`` under NaN-filled allocations: ``save_mode="full"`` against
    the light form at ``P = 2`` (bitwise, ``M_seg`` held at ``P = 2``), over a2a and, where the partition allows it,
    over the ring (and the ring's light form bitwise the a2a's)."""
    o = {"segments": 2, "save_transfer": False, **extra}
    light = {}
    for name, ex in (("p2_full_no_transfer_eq", "a2a"), ("p2_full_ring_eq", "ring")):
        if ex == "ring" and not ring:
            continue
        with _nan_allocations():
            res_full = saved_pass(save_mode="full", exchange=ex, **o)
            light[ex] = saved_pass(exchange=ex, **o)
        rule = _m_seg_rule(res_full["saved"])
        info[prefix + name] = rule
        flags[prefix + name] = (
            _eq(res_full, light[ex]) and rule["ok"] and rule["P"] == 2
        )
        del res_full
    if ring:
        flags[prefix + "p2_ring_eq_a2a"] = _eq(light["ring"], light["a2a"])
    del light


def _plain_split_pass(buf, g, beta, dout, call, H_qk: int, HV: int, *, ckpt: bool):
    """Issue #8367: one autograd pass with q / k / v as plain ``split`` + ``view`` head views of a packed leaf holding
    ``buf`` (torch's own split backward concatenates the gradients).  Returns the results (``dq`` / ``dk`` / ``dv``
    cut from the leaf's gradient) and whether the tensor hooks and ``retain_grad`` set on the views and on the split
    outputs of the original forward fired once each with the gradient the leaf received (vacuous at ``T_r = 0``)."""
    T = int(buf.shape[0])
    sizes = (H_qk * 128, H_qk * 128, HV * 128)
    row = buf.detach().clone().requires_grad_(True)
    g_leaf, b_leaf = (x.detach().clone().requires_grad_(True) for x in (g, beta))
    fired: dict[str, list] = {}
    views: dict[str, torch.Tensor] = {}

    def fn(row_, g_, b_):
        parts = row_.split(sizes, dim=-1)
        q_, k_, v_ = (
            p.view(T, n, 128) for p, n in zip(parts, (H_qk, H_qk, HV), strict=True)
        )
        if not views:  # the original forward (a checkpoint recompute builds a graph no backward runs)
            views.update(
                q=q_, k=k_, v=v_, **{f"part{i}": p for i, p in enumerate(parts)}
            )
            for name, t in views.items():
                t.register_hook(
                    lambda gr, name=name: fired.setdefault(name, []).append(gr.clone())
                )
                t.retain_grad()
        return call(q_, k_, v_, g_, b_)

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, row, g_leaf, b_leaf, use_reentrant=False)
    else:
        out = fn(row, g_leaf, b_leaf)
    out.backward(dout)
    ga = row.grad
    cols = {
        "q": (0, sizes[0]),
        "k": (sizes[0], 2 * sizes[0]),
        "v": (2 * sizes[0], None),
    }
    res = {
        "out": out.detach(),
        **{
            "d" + n: ga[:, a:b].reshape(T, h, 128)
            for (n, (a, b)), h in zip(cols.items(), (H_qk, H_qk, HV), strict=True)
        },
        "dg": g_leaf.grad,
        "dbeta": b_leaf.grad,
    }
    hooks = T == 0 or (
        sorted(fired) == sorted(views)
        and all(len(x) == 1 for x in fired.values())
        and all(
            torch.equal(fired[n][0].reshape(T, -1), ga[:, a:b])
            and torch.equal(views[n].grad.reshape(T, -1), ga[:, a:b])
            and torch.equal(fired[f"part{i}"][0], ga[:, a:b])
            for i, (n, (a, b)) in enumerate(cols.items())
        )
    )
    return res, hooks


def _rounded(nbytes: int) -> int:
    return -(-nbytes // ALLOC_GRANULE) * ALLOC_GRANULE if nbytes else 0


# ----------------------------------------------------------------------------------------------------------------
# one rank


def _live_cuda_tensors(device) -> dict[int, tuple]:
    """``data_ptr -> (bytes, shape, dtype)`` of every live CUDA tensor on ``device`` (diagnostic only)."""
    out: dict[int, tuple] = {}
    for obj in gc.get_objects():
        try:
            if (
                torch.is_tensor(obj)
                and obj.is_cuda
                and obj.device == device
                and obj.numel()
            ):
                out[obj.data_ptr()] = (
                    obj.numel() * obj.element_size(),
                    tuple(obj.shape),
                    str(obj.dtype),
                )
        except Exception:  # noqa: BLE001 - foreign objects may refuse attribute access
            continue
    return out


def _new_cuda_tensors(before: dict[int, tuple], device, *, skip=()) -> list[dict]:
    """The live CUDA tensors that are new since ``before`` with the type names of their referrers (bounded)."""
    found: list[dict] = []
    for obj in gc.get_objects():
        try:
            if not (
                torch.is_tensor(obj)
                and obj.is_cuda
                and obj.device == device
                and obj.numel()
            ):
                continue
            ptr = obj.data_ptr()
            if ptr in before or ptr in skip:
                continue
            refs = [type(r).__name__ for r in gc.get_referrers(obj)[:6]]
            found.append(
                {
                    "bytes": obj.numel() * obj.element_size(),
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "referrers": refs,
                }
            )
        except Exception:  # noqa: BLE001
            continue
        if len(found) >= 40:
            break
    return found


def _alloc_stats(device) -> dict[str, int]:
    """``requested`` (bytes the callers asked for) and ``allocated`` (block bytes) of the caching allocator."""
    stats = torch.cuda.memory_stats(device)
    return {
        "requested": int(stats.get("requested_bytes.all.current", 0)),
        "allocated": int(stats.get("allocated_bytes.all.current", 0)),
    }


def _block_of(t: torch.Tensor) -> dict | None:
    """Size / requested size of the allocator block that holds ``t`` (from the snapshot's segments; diagnostic)."""
    ptr = t.data_ptr()
    for seg in torch.cuda.memory._snapshot().get("segments", []):
        addr = int(seg.get("address", 0))
        for blk in seg.get("blocks", []):
            size = int(blk.get("size", 0))
            if addr <= ptr < addr + size:
                return {
                    "size": size,
                    "requested_size": int(blk.get("requested_size", 0)),
                    "state": str(blk.get("state", "")),
                    "segment_size": int(seg.get("total_size", 0)),
                }
            addr += size
    return None


def _trace_mark(device) -> int:
    """Start the allocator's history (diagnostic only) and return the current trace length of ``device``."""
    torch.cuda.memory._record_memory_history(max_entries=200000)
    snap = torch.cuda.memory._snapshot()
    traces = snap.get("device_traces") or []
    return len(traces[device.index]) if device.index < len(traces) else 0


def _live_blocks_since(mark: int, device, *, skip=()) -> list[dict]:
    """Allocator blocks allocated after ``mark`` and not freed, with sizes and their non-torch Python frames."""
    snap = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    traces = snap.get("device_traces") or []
    events = traces[device.index][mark:] if device.index < len(traces) else []
    live: dict[int, dict] = {}
    for ev in events:
        action = ev.get("action")
        if action == "alloc":
            live[ev["addr"]] = ev
        elif action in ("free_requested", "free_completed"):
            live.pop(ev["addr"], None)
    rows = []
    for ev in live.values():
        if ev["addr"] in skip:
            continue
        frames = []
        for f in ev.get("frames", []):
            fn = str(f.get("filename", ""))
            if (
                "/torch/" in fn
                and "checkpoint" not in fn
                and "cake_gdn_det_train_kcp" not in fn
            ):
                continue
            frames.append(f"{fn.rsplit('/', 1)[-1]}:{f.get('line')} {f.get('name')}")
            if len(frames) >= 10:
                break
        rows.append({"size": int(ev["size"]), "frames": frames})
        if len(rows) >= 40:
            break
    return rows


def _test_mode() -> str:
    """``KCP_TEST_MODE``: ``full`` (forward + backward rows, default) or ``forward`` (the INT1 forward-only rows:
    #1-#5, ``out`` and the carried states against FP64, no backward program)."""
    mode = os.environ.get("KCP_TEST_MODE", "full")
    if mode not in ("full", "forward"):
        raise ValueError(f"KCP_TEST_MODE must be 'full' or 'forward', got {mode!r}")
    return mode


def _rank_fragments(ctx, rank: int) -> tuple:
    """This rank's fragments from ``ctx.fragments``: the local tuple of fragments (empty on a rank that holds no
    tokens, e.g. ``ragged:1200,0,900,900``) or a per-rank table (a tuple of such tuples), indexed by ``rank``."""
    frs = ctx.fragments
    if len(frs) > 0 and isinstance(frs[0], tuple):
        return frs[rank]
    return frs


def _fwd_draw(
    ctx,
    ws,
    base: dict,
    ops: dict,
    W: int,
    rank: int,
    cu: list[int],
    HV: int,
    info: dict,
) -> tuple[torch.Tensor, torch.Tensor, list, dict[str, bool]]:
    """One draw of the forward-only body: ``out``, this rank's carried states ``S_in`` (``[n_in, HV, 128, 128]``,
    ``in_slot`` order) with the global cut of each, and the bitwise flags of the forward's forms (repeat,
    ``save_transfer``, light == full, saved == replay, ring == a2a, strided views and the copy layout == contiguous);
    ``info`` receives the launch record."""
    q, k, v, g, beta = (ops[n] for n in ("q", "k", "v", "g", "beta"))

    def fwd(qq=q, kk=k, vv=v, **opts):
        return kcp.cake_gdn_det_train_kcp_forward(
            qq, kk, vv, g, beta, ctx, workspace=ws, **{**base, **opts}
        )

    flags: dict[str, bool] = {}
    out, saved = fwd()
    info["fwd_record"] = {
        k_: v_
        for k_, v_ in kcp.LAST_CALL["fwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
    }
    info["fwd_sequence"] = list(kcp.LAST_CALL["fwd"]["sequence"])
    flags["repeat"] = torch.equal(out, fwd()[0])
    flags["save_transfer_eq"] = torch.equal(out, fwd(save_transfer=False)[0])
    flags["light_eq_full"] = torch.equal(out, fwd(save_mode="full")[0])
    flags["saved_eq_replay"] = torch.equal(out, fwd(boundary_mode="replay")[0])
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_eligible
    flags["ring_eq_a2a"] = (not ring_eligible) or torch.equal(
        out, fwd(exchange="ring", save_transfer=False)[0]
    )
    _buf, qs, ks, vs = _packed_views(q, k, v)
    flags["strided_eq_contiguous"] = torch.equal(out, fwd(qs, ks, vs)[0])
    flags["copy_layout_eq_contiguous"] = torch.equal(
        out, fwd(qs, ks, vs, layout="copy")[0]
    )
    # the carried states: S_in per crossing-in fragment (in_slot order) with its global cut
    s_in = (
        kcp.saved_states(saved)
        if saved is not None
        else out.new_zeros((0, HV, 128, 128), dtype=torch.float32)
    )
    cuts = [None] * int(ctx.n_in)
    for f in _rank_fragments(ctx, rank):
        if f.length > 0 and f.off > 0:
            cuts[int(ctx.in_slot[f.slot])] = int(cu[f.doc] + f.off)
    info["s_in_cuts"] = cuts
    return out, s_in, cuts, flags


def _judge_fwd_draw(
    judge: TierJudge,
    failures: list[str],
    j: int,
    row: Row,
    part,
    cu: list[int],
    device,
    seed: int,
    objs_out: list,
    objs_s: list,
    objs_cuts: list,
) -> None:
    """Rank 0, draw ``j`` of the forward-only body: ``out`` reassembled to the global order against
    ``gdn_exact_forward``, and every rank's crossing-in ``S_in`` (stacked in rank, then slot order) against
    ``cut_states``, each beside its naive BF16 tier (``gdn_naive_forward``, ``naive_cut_states``)."""
    ref_mod = _reference()
    glob = _global_inputs(row, cu, device, seed=seed)
    ops = (
        glob["q"],
        glob["k"],
        glob["v"],
        glob["g"],
        glob["beta"],
        torch.tensor(cu, dtype=torch.int64),
    )
    ref_out, _ = ref_mod.gdn_exact_forward(*ops, scale=SCALE)
    naive_out, _ = ref_mod.gdn_naive_forward(*ops, scale=SCALE)
    cand = _to_global(part, [t.to(device) for t in objs_out], cu, cu[-1])
    _judge_add(judge, failures, "out", j, cand, ref_out, naive_out.to(torch.bfloat16))
    picked = [
        (st[slot], c)
        for cs, st in zip(objs_cuts, objs_s, strict=True)
        for slot, c in enumerate(cs)
        if c is not None
    ]
    if not picked:
        return
    cuts = sorted({c for _, c in picked})
    states = ref_mod.cut_states(*ops, cuts, scale=SCALE)
    naive_states = ref_mod.naive_cut_states(*ops, cuts, scale=SCALE)
    _judge_add(
        judge,
        failures,
        "s_in",
        j,
        torch.stack([s_ for s_, _ in picked]).to(device),
        torch.stack([states[c].to(device).reshape(s_.shape) for s_, c in picked]),
        torch.stack([naive_states[c].to(device).reshape(s_.shape) for s_, c in picked]),
    )


def _run_rank_fwd(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """Forward-only body of one rank (INT1) over the row's draws: the forward's bitwise forms on every draw
    (:func:`_fwd_draw`); on rank 0 ``out`` and the crossing-in fragments' ``S_in`` judged against the FP64 forward
    and cut states beside the naive BF16 tier; nothing of the backward runs."""
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    ws = kcp.CakeGDNDetKcpWorkspace()
    base = {"backend": backend}
    t_start = time.time()
    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    judge, failures = (TierJudge(), []) if rank == 0 and total > 0 else (None, [])
    # an empty pack has nothing to judge: one draw of its bitwise forms
    for j in range(DRAWS if total > 0 else 1):
        seed = _draw_seed(j)
        ops = _local_inputs(row, part, cu, rank, device, seed=seed)
        out, s_in, cuts, fl = _fwd_draw(
            ctx, ws, base, ops, W, rank, cu, HV, info if j == 0 else {}
        )
        for key, ok in fl.items():
            flags[key] = flags.get(key, True) and bool(ok)
        objs_out = _gather(out.cpu(), W, group)
        objs_s = _gather(s_in.cpu(), W, group)
        objs_cuts = _gather(cuts, W, group)
        del out, s_in, ops
        if judge is not None:
            _judge_fwd_draw(
                judge,
                failures,
                j,
                row,
                part,
                cu,
                device,
                seed,
                objs_out,
                objs_s,
                objs_cuts,
            )
        del objs_out, objs_s
    flag_list = _gather(flags, W, group)
    info_list = _gather(info, W, group)
    report = {
        "row": dataclasses.asdict(row),
        "mode": "forward",
        "rank": rank,
        "world": W,
        "backend": backend,
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "nccl": ".".join(str(x) for x in torch.cuda.nccl.version())
        if torch.cuda.is_available()
        else None,
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": True,
        "flags": {},
        "gated": [],
        "reported": [],
        "info_by_rank": info_list,
    }
    for key in flag_list[0]:
        ok = all(bool(fl[key]) for fl in flag_list)
        report["flags"][key] = ok
        report["gated"].append(key)
        report["pass"] &= ok
    _judge_report(report, judge, failures)
    report["seconds"] = round(time.time() - t_start, 2)
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"], group=group)
        report["pass"] = all(verdict)
        dist.barrier(group=group)
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + "-fwd.json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _kept_record(T_r: int) -> dict:
    """The last backward's launch inventory under ``reuse_recompute_tapes``: where its forward tapes came from and
    whether anything ran before #6.  With tokens the kept path must have been taken (no silent fallback to the
    replay); a rank without tokens has no tapes and runs the replay form as before."""
    bwd = kcp.LAST_CALL["bwd"]
    seq = list(bwd["sequence"])
    forward_programs = {
        "fwd_prep",
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_fwdh_out",
        "fwd_o",
    }
    first_local = next(
        (s_ for s_ in seq if s_ in kcp.LOCAL_PROGRAMS or s_ in forward_programs), None
    )
    source = bwd.get("tapes_source")
    if T_r > 0:
        path_ok = (
            source == "kept"
            and first_local == "bwd_prep"
            and not (forward_programs & set(seq))
            and bool(kcp.LAST_CALL["fwd"].get("hand_off"))
        )
    else:
        path_ok = source == "replay" and not bool(kcp.LAST_CALL["fwd"].get("hand_off"))
    return {
        "tapes_source": source,
        "first_local_program": first_local,
        "forward_programs_in_bwd": sorted(forward_programs & set(seq)),
        "fwd_hand_off": bool(kcp.LAST_CALL["fwd"].get("hand_off")),
        "P": bwd.get("P"),
        "path_ok": bool(path_ok),
        "bwd_sequence": seq,
    }


def _rank_draw(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    backend: str,
    part,
    cu: list[int],
    ctx,
    ws,
    ops: dict,
    seed: int,
    info: dict,
    *,
    full: bool,
) -> tuple[dict, dict[str, bool]]:
    """One draw of the body of one rank on the draw's operands ``ops`` (generated at ``seed``): the primary saved
    forward + backward (``res``, the judged outputs) and the bitwise flags.  Every draw runs the core forms (repeat,
    saved == replay, light == full, autograd == explicit, checkpoint == direct, ``save_transfer``, ring == a2a,
    strided views == contiguous) and on the prod row the route check; ``full`` (draw 0) runs every other mode of the
    KCP forward / backward and the contracts as well.  ``info`` receives the records."""
    H_qk, HV = row.heads
    total = cu[-1]
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    base = {"backend": backend}

    def saved_pass(qq=q, kk=k, vv=v, gg=g, bb=beta, dd=dout, **opts) -> dict:
        o = {**base, **opts}
        fwd_opts = dict(o)
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            qq, kk, vv, gg, bb, ctx, workspace=ws, **fwd_opts
        )
        dq, dk, dv, dbeta, dg = kcp.cake_gdn_det_train_kcp_backward(
            qq, kk, vv, gg, bb, ctx, dd, saved=saved, workspace=ws, **o
        )
        return {
            "out": out,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dg": dg,
            "dbeta": dbeta,
            "saved": saved,
        }

    def autograd_pass(ckpt: bool, *, twice: bool = False, workspace=ws, **opts) -> dict:
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o = {**base, **opts}

        def fn(*a):
            return kcp.cake_gdn_det_kcp(*a, ctx, workspace=workspace, **o)

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
        else:
            out = fn(*leaves)
        out.backward(dout, retain_graph=twice)
        if twice:
            out.backward(dout)
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
        }

    flags: dict[str, bool] = {}
    res = saved_pass()
    info["fwd_record"] = {
        k_: v_
        for k_, v_ in kcp.LAST_CALL["fwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
    }
    info["bwd_record"] = {
        k_: v_
        for k_, v_ in kcp.LAST_CALL["bwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
    }
    info["fwd_sequence"] = list(kcp.LAST_CALL["fwd"]["sequence"])
    info["bwd_sequence"] = list(kcp.LAST_CALL["bwd"]["sequence"])
    if row.stage == "prod":
        # the production route: the shipped rule's plan segments this rank's fragments, and the primary forward and
        # backward ran that plan (their launch records)
        seg = ctx.heads(HV).segments
        info["plan"] = {
            "P": int(seg.P),
            "p_rule": seg.p_rule,
            "n_frag": int(seg.n_frag),
            "NC_max_frag": int(seg.NC_max_frag),
        }
        flags["prod_route_segmented"] = (
            seg.p_rule == plan.P_RULE_DEFAULT
            and int(seg.P) > 1
            and int(kcp.LAST_CALL["fwd"]["P"]) == int(seg.P)
            and int(kcp.LAST_CALL["bwd"]["P"]) == int(seg.P)
        )
    flags["repeat"] = _eq(res, saved_pass())
    flags["saved_eq_replay"] = _eq(res, saved_pass(boundary_mode="replay"))
    flags["light_eq_full"] = _eq(res, saved_pass(save_mode="full"))
    res_auto = autograd_pass(False)
    flags["autograd_eq_saved"] = _eq(res, res_auto)
    res_ckpt = autograd_pass(True)
    flags["ckpt_eq_direct"] = _eq(res_auto, res_ckpt)
    res_recompute = saved_pass(save_transfer=False)
    flags["save_transfer_eq"] = _eq(res, res_recompute)
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_eligible
    flags["ring_eq_a2a"] = (not ring_eligible) or _eq(
        res_recompute, saved_pass(exchange="ring", save_transfer=False)
    )
    # strided packed-QKV views, zero-copy (layout="tma"): bitwise the contiguous call
    _buf, qs, ks, vs = _packed_views(q, k, v)
    flags["strided_eq_contiguous"] = _eq(res, saved_pass(qs, ks, vs))
    # issue #7817: packed operands -> #8 writes dq / dk / dv as the same views of one packed dQKV row
    flags["packed_grad_layout"] = (
        int(q.shape[0]) == 0 or kcp.LAST_CALL["bwd"]["grad_layout"] == "packed"
    )  # T_r = 0: nothing to pack
    if not full:  # the draws after the first: the core forms only
        return res, flags
    flags["ckpt_full_eq_direct"] = _eq(res_auto, autograd_pass(True, save_mode="full"))
    flags["ckpt_replay_eq_direct"] = _eq(
        res_auto, autograd_pass(True, boundary_mode="replay")
    )
    # reuse_recompute_tapes: the checkpoint backward binds the forward tapes torch's recompute just wrote instead
    # of replaying #1 and the tape walk; bitwise the direct and the switch-off checkpoint gradients; the kept path
    # is proven from the backward's launch inventory (a rank without tokens has no tapes and runs the replay form);
    # the registry is empty afterwards; the per-rank peak of the checkpointed call with the switch is not above the
    # switch-off peak (G8 evidence; the row of record is the memory trace)
    KEPT = {"reuse_recompute_tapes": True}
    # both states are named (res_ckpt above is the default's)
    OFF = {"reuse_recompute_tapes": False}
    T_r_here = int(q.shape[0])
    res_ckpt_off = autograd_pass(True, **OFF)
    kept_stats0 = dict(kcp.KEPT_STATS)
    res_kept = autograd_pass(True, **KEPT)
    info["kept"] = _kept_record(T_r_here)
    flags["kept_eq_direct"] = _eq(res_auto, res_kept)
    flags["kept_eq_ckpt_light"] = _eq(res_ckpt_off, res_kept) and _eq(
        res_ckpt, res_kept
    )
    flags["kept_path_taken"] = bool(info["kept"]["path_ok"])
    res_kept_nt = autograd_pass(True, save_transfer=False, **KEPT)
    info["kept_no_transfer"] = _kept_record(T_r_here)
    # the direct reference inside the stats window names the replay form: under the shipped default a direct
    # backward looks the registry up and counts a "miss" (by design), which would break the miss = 0 expectation
    flags["kept_no_transfer_eq_direct"] = _eq(
        autograd_pass(False, save_transfer=False, **OFF), res_kept_nt
    ) and bool(info["kept_no_transfer"]["path_ok"])
    res_kept_twice = autograd_pass(True, twice=True, **KEPT)
    flags["kept_twice_accumulates"] = all(
        torch.equal(res_kept_twice[n], res_auto[n] + res_auto[n]) for n in GRADS
    )
    kept_delta = {k: kcp.KEPT_STATS[k] - kept_stats0[k] for k in kept_stats0}
    info["kept_stats_delta"] = kept_delta
    n_kept_bwd = 4  # three kept passes, one of them with two backwards
    expect = (
        {"put": n_kept_bwd, "take": n_kept_bwd, "miss": 0}
        if T_r_here > 0
        else {"put": 0, "take": 0, "miss": n_kept_bwd}
    )
    flags["kept_registry_empty"] = kcp.kept_registry_size() == 0 and all(
        kept_delta[k] == v_ for k, v_ in expect.items()
    )

    # issue #7074 (2) + G10: the "auto" form (the shipped default) in a region the GDN node closes takes the kept
    # path on a rank with tokens (the frame proof "dead"; a rank without tokens replays as before), is bitwise the
    # direct result, and its checkpointed forward + backward -- the recompute forward, the frame proof over the
    # autograd graph (structure only, never tensor contents), the kept-tape take -- make no synchronizing CUDA call
    # (outside the kept-stats window above: the "auto" pass has its own accounting below)
    AUTO = {"reuse_recompute_tapes": "auto"}
    auto_stats0 = dict(kcp.KEPT_STATS)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        res_auto_form = autograd_pass(True, **AUTO)
        flags["auto_no_sync"] = True
    except RuntimeError as exc:  # a synchronizing call in the "auto" pass
        info["auto_no_sync_error"] = str(exc)[:200]
        flags["auto_no_sync"] = False
        res_auto_form = None
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if res_auto_form is None:
        res_auto_form = autograd_pass(True, **AUTO)
    info["auto_form"] = _kept_record(T_r_here)
    info["auto_form"]["hand_off_reason"] = kcp.LAST_CALL["fwd"].get("hand_off_reason")
    flags["auto_eq_direct"] = _eq(res_auto, res_auto_form)
    # a rank with tokens hands off on the frame proof ("dead"); a rank without tokens has no tapes and switches the
    # hand-off off before the "auto" decision, as every form does there (recorded "off", the replay form)
    flags["auto_path_taken"] = bool(info["auto_form"]["path_ok"]) and (
        info["auto_form"]["hand_off_reason"] == ("dead" if T_r_here > 0 else "off")
        or not kcp.torch_release_verified()
    )
    auto_delta = {k: kcp.KEPT_STATS[k] - auto_stats0[k] for k in auto_stats0}
    info["auto_stats_delta"] = auto_delta
    # one hand-off on a rank with tokens (the proof "dead"); none without tokens or on an unverified release -- and no
    # miss is counted for an "auto" forward that did not hand off
    n_auto = 1 if (T_r_here > 0 and kcp.torch_release_verified()) else 0
    flags["auto_registry_empty"] = kcp.kept_registry_size() == 0 and all(
        auto_delta[k] == v_
        for k, v_ in {"put": n_auto, "take": n_auto, "miss": 0}.items()
    )

    # issue #7818: a CakeGDNDetKcpRecomputeHandle carries this rank's incoming states S_in from the checkpointed forward to
    # torch's recompute, which then issues no collective and waits on none (counted by phase through shims on
    # torch.distributed; the recompute phase is marked by the checkpoint's context_fn).  Out and the five gradients
    # are bitwise the plain checkpoint's (same options) and the direct pass's; the original forward and the backward
    # exchange exactly as the plain checkpoint does (A4); a second backward reuses the entry; the handle retains
    # exactly S_in (the light saved set without the transfer and the convolution rows).  The row's own plan P is
    # used (a forced segment count is not valid on every rank of every row); info records it.  The plain
    # checkpoint's per-phase counts are recorded as the reproduction of the issue.
    def handle_pass(handle: bool, *, twice: bool = False, workspace=ws, **opts):
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o = {**base, "save_transfer": False, **opts}
        hd = kcp.CakeGDNDetKcpRecomputeHandle() if handle else None

        def fn(q_, k_, v_, g_, b_, h_):
            return kcp.cake_gdn_det_kcp(
                q_, k_, v_, g_, b_, ctx, workspace=workspace, recompute_handle=h_, **o
            )

        from torch.utils.checkpoint import checkpoint

        with CollectiveCounter() as cc:
            rphase = RecomputePhase(cc)
            out = checkpoint(
                fn,
                *leaves,
                hd,
                use_reentrant=False,
                context_fn=lambda: (contextlib.nullcontext(), rphase),
            )
            # torch's early stop ends the recompute inside the node's apply, after the forward recorded itself
            modes = [kcp.LAST_CALL["fwd"].get("recompute_handle")]
            P_used = int(kcp.LAST_CALL["fwd"].get("P", 0) or 0)
            cc.phase = "bwd"
            out.backward(dout, retain_graph=twice)
            modes.append(kcp.LAST_CALL["fwd"].get("recompute_handle"))
            if twice:
                out.backward(dout)
                modes.append(kcp.LAST_CALL["fwd"].get("recompute_handle"))
        rec = {
            **cc.record(),
            "modes": modes,
            "P": P_used,
            "handle_bytes": hd.nbytes() if hd is not None else 0,
        }
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
        }, rec

    handle_info: dict[str, object] = {}
    res_plain, rec_plain = handle_pass(False)
    res_h, rec_h = handle_pass(True)
    handle_info["plain"], handle_info["handle"] = rec_plain, rec_h
    flags["handle_eq_ckpt"] = _eq(res_plain, res_h)
    flags["handle_eq_direct"] = _eq(res_auto, res_h)
    flags["handle_recompute_no_comm"] = (
        rec_h["calls"].get("recompute", 0) == 0
        and rec_h["waits"].get("recompute", 0) == 0
    )
    flags["handle_orig_bwd_comm_unchanged"] = all(
        rec_h[t].get(ph, 0) == rec_plain[t].get(ph, 0)
        for t in ("calls", "waits")
        for ph in ("orig", "bwd")
    )
    flags["handle_modes"] = rec_h["modes"] == ["capture", "replay"]
    # the ring exchange (where the partition allows it): its recompute skips the hops as well
    if W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS:
        res_pr, rec_pr = handle_pass(False, exchange="ring")
        res_hr, rec_hr = handle_pass(True, exchange="ring")
        handle_info["ring"] = {"plain": rec_pr, "handle": rec_hr}
        flags["handle_ring_eq_ckpt"] = _eq(res_pr, res_hr) and _eq(res_auto, res_hr)
        flags["handle_ring_recompute_no_comm"] = (
            rec_hr["calls"].get("recompute", 0) == 0
            and rec_hr["waits"].get("recompute", 0) == 0
            and rec_hr["modes"] == ["capture", "replay"]
        )
    res_h2, rec_h2 = handle_pass(True, twice=True)
    handle_info["twice"] = rec_h2
    flags["handle_twice_accumulates"] = (
        all(torch.equal(res_h2[n], res_auto[n] + res_auto[n]) for n in GRADS)
        and rec_h2["calls"].get("recompute", 0) == 0
        and rec_h2["modes"] == ["capture", "replay", "replay"]
    )
    s_in_query = int(
        kcp.cake_gdn_det_train_kcp_saved_bytes(
            ctx,
            H_qk=H_qk,
            HV=HV,
            save_mode="light",
            save_transfer=False,
            with_conv=False,
        )
    )
    handle_info["s_in_query_bytes"] = s_in_query
    flags["handle_bytes_eq_query"] = rec_h["handle_bytes"] == s_in_query
    # retained after the checkpointed forward (per-call workspace, after a warm-up of the same form): exactly the
    # handle's S_in, where the plain checkpoint retains nothing (retained_zero below)
    handle_pass(True, workspace=None)
    torch.cuda.synchronize(device)
    gc.collect()
    leaves_h = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    hd_r = kcp.CakeGDNDetKcpRecomputeHandle()

    def fn_r(q_, k_, v_, g_, b_, h_):
        return kcp.cake_gdn_det_kcp(
            q_, k_, v_, g_, b_, ctx, recompute_handle=h_, save_transfer=False, **base
        )

    from torch.utils.checkpoint import checkpoint as _checkpoint

    before_h = _alloc_stats(device)
    out_h = _checkpoint(fn_r, *leaves_h, hd_r, use_reentrant=False)
    torch.cuda.synchronize(device)
    after_h = _alloc_stats(device)
    retained_h = int(
        after_h["requested"]
        - before_h["requested"]
        - out_h.numel() * out_h.element_size()
    )
    handle_info["retained_bytes"] = retained_h
    flags["handle_retained_eq_s_in"] = retained_h == hd_r.nbytes() == s_in_query
    out_h.backward(dout)
    del out_h, leaves_h, hd_r
    # the reproduction of the issue: the plain checkpoint's recompute exchanges (W > 1 with a crossing fragment)
    handle_info["plain_recompute_calls"] = rec_plain["calls"].get("recompute", 0)
    handle_info["plain_recompute_waits"] = rec_plain["waits"].get("recompute", 0)
    info["recompute_handle"] = handle_info

    def _ckpt_peak(**o) -> int:
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        live = torch.cuda.memory_allocated(device)
        autograd_pass(True, **o)
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device) - live)

    peaks = {"off": _ckpt_peak(**OFF), "on": _ckpt_peak(**KEPT)}
    info["ckpt_peak_bytes"] = peaks
    flags["kept_peak_le_off"] = peaks["on"] <= peaks["off"]
    # save_mode="full" without save_transfer (the keyword, or the ring exchange, which needs it): the full backward
    # runs no transfer walk, so at P > 1 its merges read the saved M_seg^T plane; on NaN-filled allocations a read of
    # an unwritten plane cannot pass on the bytes the forward left in the caching allocator
    with _nan_allocations():
        flags["full_no_transfer_eq"] = _eq(
            res, saved_pass(save_mode="full", save_transfer=False)
        )
        flags["full_ring_eq_a2a"] = (not ring_eligible) or _eq(
            res, saved_pass(save_mode="full", exchange="ring", save_transfer=False)
        )
    # many rows plan P = 1: the same full / ring + full forms again at the forced P = 2 (segments override, where every
    # rank's fragments take it), against the light form at P = 2
    info["p2_eligible"] = _p_group_ok(ctx, HV, 2, group, W, device)
    if info["p2_eligible"]:
        _p2_full_flags(saved_pass, info, flags, ring=ring_eligible)
    res_host = saved_pass(group_reduce="host")
    info["host_reduce_eq_fused"] = _eq(
        res, res_host
    )  # reported: both forms are fixed-order FP32 + one rounding
    # the copy adapter of the strided packed-QKV views: bitwise the contiguous call
    flags["copy_layout_eq_contiguous"] = _eq(res, saved_pass(qs, ks, vs, layout="copy"))
    # the copy adapter copies strided views; a one-token view is already contiguous, stays a packed view and keeps
    # the packed gradient row
    copy_noop = int(q.shape[0]) > 0 and all(x.is_contiguous() for x in (qs, ks, vs))
    flags["copy_grad_layout_separate"] = kcp.LAST_CALL["bwd"]["grad_layout"] == (
        "packed" if copy_noop else "separate"
    )
    # ... and the split hands that row to the packed leaf without a concatenation (autograd, bitwise)
    split_before = dict(kcp.PACKED_SPLIT_STATS)
    qkv_leaf = _buf.detach().clone().requires_grad_(True)
    g_leaf, b_leaf = (x.detach().clone().requires_grad_(True) for x in (g, beta))
    pq, pk, pv = kcp.cake_gdn_det_train_kcp_packed_qkv_views(qkv_leaf, H_qk=H_qk, HV=HV)
    out_split = kcp.cake_gdn_det_kcp(
        pq, pk, pv, g_leaf, b_leaf, ctx, workspace=ws, **base
    )
    out_split.backward(dout)
    ga = qkv_leaf.grad
    res_split = {
        "out": out_split.detach(),
        "dq": ga[:, : H_qk * 128].reshape(-1, H_qk, 128),
        "dk": ga[:, H_qk * 128 : 2 * H_qk * 128].reshape(-1, H_qk, 128),
        "dv": ga[:, 2 * H_qk * 128 :].reshape(-1, HV, 128),
        "dg": g_leaf.grad,
        "dbeta": b_leaf.grad,
    }
    flags["packed_split_eq_contiguous"] = _eq(res_auto, res_split)
    flags["packed_split_aliased"] = (
        int(ga.shape[0]) == 0
        or kcp.PACKED_SPLIT_STATS["aliased"] - split_before["aliased"] == 1
    )
    del qkv_leaf, g_leaf, b_leaf, out_split, ga, res_split

    # issue #8367: plain split views of the packed leaf keep torch's autograd path (its split backward concatenates):
    # bitwise the contiguous call, direct and checkpointed, with the hooks and retain_grad on the views firing
    def gdn_call(*a):
        return kcp.cake_gdn_det_kcp(*a, ctx, workspace=ws, **base)

    for ck, tag in ((False, "direct"), (True, "ckpt")):
        res_plain, hooks_ok = _plain_split_pass(
            _buf, g, beta, dout, gdn_call, H_qk, HV, ckpt=ck
        )
        flags[f"plain_split_{tag}_eq_contiguous"] = _eq(res_auto, res_plain)
        flags[f"plain_split_{tag}_hooks_fire"] = hooks_ok
        del res_plain
    # gradient accumulation: .backward() twice == exactly twice the gradient (x + x is exact in every float format)
    res_twice = autograd_pass(False, twice=True)
    flags["backward_twice_accumulates"] = all(
        torch.equal(res_twice[n], res_auto[n] + res_auto[n]) for n in GRADS
    )
    # G10 (issue #7074): preparing a fresh context of this rank's layout -- the host plan, its table uploads and the
    # per-key device tables -- makes no synchronizing CUDA call (pinned asynchronous copies), so a pipeline can prepare
    # the next batch's metadata behind queued device work
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        fresh_ctx = plan.cake_gdn_det_train_kcp_build_context(part, group, rank, device)
        fresh_ctx.prepare(H_qk, HV)
        flags["prepare_no_sync"] = True
    except RuntimeError as exc:  # "called a synchronizing CUDA operation"
        info["prepare_no_sync_error"] = str(exc)[:200]
        flags["prepare_no_sync"] = False
    finally:
        torch.cuda.set_sync_debug_mode("default")
    del fresh_ctx
    torch.cuda.synchronize(device)
    # #7039: retained bytes after a checkpointed forward (graph off), per-call and caller-held workspace forms.
    # the plan's per-key device tables are uploaded first (ctx.prepare + one warm-up call of the
    # same form) so they are not counted; graph="replay" keeps a static pool and is reported, not asserted.
    prepare = getattr(ctx, "prepare", None)
    if callable(prepare):
        prepare(H_qk, HV)
    retained: dict[str, int] = {}
    retained_block: dict[str, int] = {}
    for form, wsp, form_opts in (
        ("per_call", None, {**base, **OFF}),
        ("caller_ws", ws, {**base, **OFF}),
        ("per_call_kept", None, {**base, **KEPT}),
        ("caller_ws_kept", ws, {**base, **KEPT}),
    ):
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

        def fn(*a, _wsp=wsp, _o=form_opts):
            return kcp.cake_gdn_det_kcp(*a, ctx, workspace=_wsp, **_o)

        from torch.utils.checkpoint import checkpoint

        warm = checkpoint(
            fn, *leaves, use_reentrant=False
        )  # warm-up of the same form (tables, allocator)
        warm.backward(dout)
        del warm
        for leaf in leaves:
            leaf.grad = None
        torch.cuda.synchronize(device)
        gc.collect()
        arena_after = _forward_arena_tapes(backend, device, int(q.shape[0]))
        if wsp is not None:
            # the warm-up backward's replay re-bound the arena tapes an eager forward drops before its output
            # (FWD_DEAD_BEFORE_O; a fused forward binds no h / v_new either); start from the state a forward
            # leaves so the delta is the output alone
            for name in kcp.FWD_TAPES:
                if name not in arena_after:
                    wsp.release_tape(name)
            for name in kcp.FWD_DEAD_POOLS:
                wsp.release_pool(name)
            torch.cuda.synchronize(device)
            gc.collect()
        debug = bool(os.environ.get("KCP_TEST_DEBUG_RETAINED"))
        live_before = _live_cuda_tensors(device) if debug else None
        trace_from = _trace_mark(device) if debug else None
        before = _alloc_stats(device)
        out = checkpoint(fn, *leaves, use_reentrant=False)
        torch.cuda.synchronize(device)
        after = _alloc_stats(device)
        # The gate counts REQUESTED bytes: ``memory_allocated`` counts block sizes, and the caching allocator hands
        # out an unsplit cached block when its remainder is below the split threshold (1 MiB for the large pool),
        # so the block holding ``out`` may be larger than ``out`` without any tensor being retained.
        out_bytes = out.numel() * out.element_size()
        retained[form] = int(after["requested"] - before["requested"] - out_bytes)
        # the block delta minus out's rounded block stays under the allocator's 1 MiB split threshold
        retained_block[form] = int(
            after["allocated"] - before["allocated"] - _rounded(out_bytes)
        )
        dbg = info.setdefault("retained_debug", {}).setdefault(form, {})
        dbg["requested_delta"] = int(after["requested"] - before["requested"])
        dbg["allocated_delta"] = int(after["allocated"] - before["allocated"])
        dbg["out_bytes"] = int(out_bytes)
        dbg["out_block"] = _block_of(out)
        if wsp is not None:
            # an eager forward leaves no dead forward tape in the caller's arena (its output took their place)
            dbg["arena_tapes_after_forward"] = sorted(wsp._tapes)
            dead_tapes = set(kcp.FWD_DEAD_BEFORE_O + kcp.FWD_DEAD_AFTER_PREP)
            no_dead_tape = not (set(wsp._tapes) & dead_tapes)
            flags["fwd_drops_dead_tapes"] = (
                flags.get("fwd_drops_dead_tapes", True) and no_dead_tape
            )
            dbg["arena_pools_after_forward"] = sorted(wsp._pools)
            no_dead_pool = not (set(wsp._pools) & set(kcp.FWD_DEAD_POOLS))
            flags["fwd_drops_dead_pools"] = (
                flags.get("fwd_drops_dead_pools", True) and no_dead_pool
            )
        if debug:
            dbg["tensors"] = _new_cuda_tensors(
                live_before, device, skip={out.data_ptr()}
            )
            dbg["blocks"] = _live_blocks_since(
                trace_from, device, skip={out.data_ptr()}
            )
        out.backward(dout)
        if wsp is not None and form == "caller_ws_kept":
            # a kept backward leaves the caller's arena as a forward leaves it: of the forward tapes only the ones
            # the forward leaves are bound (the four #5 reads; a fused forward's two), no dead tape, no kept tape
            # beyond them, no entry survives the backward
            dbg["arena_tapes_after_backward"] = sorted(wsp._tapes)
            flags["kept_ws_as_forward_leaves"] = (
                set(wsp._tapes) & set(kcp.FWD_TAPES)
            ) <= set(arena_after) and kcp.kept_registry_size() == 0
        del out, leaves
    info["retained_bytes"] = retained
    info["retained_block_bytes"] = retained_block
    # INT3 save_mode decision inputs (G8, M1): this rank's saved-set bytes per save_mode -- the public byte query and
    # the measured bytes of the saved objects (recorded, not gated: the default stays "light")
    try:
        sb: dict[str, int] = {"T_r": int(q.shape[0])}
        for mode in ("light", "full"):
            sb[f"{mode}_query"] = int(
                kcp.cake_gdn_det_train_kcp_saved_bytes(
                    ctx, H_qk=H_qk, HV=HV, save_mode=mode
                )
            )
            _, s_obj = kcp.cake_gdn_det_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, save_mode=mode, **base
            )
            sb[f"{mode}_actual"] = int(s_obj.bytes())
            del s_obj
        info["saved_bytes"] = sb
    except Exception as exc:  # noqa: BLE001 - recorded for the decision table
        info["saved_bytes"] = f"unavailable: {type(exc).__name__}: {str(exc)[:200]}"
    flags["retained_zero"] = all(v_ == 0 for v_ in retained.values())
    flags["retained_block_lt_1mib"] = all(v_ < 2**20 for v_ in retained_block.values())
    # exchange bytes: the tables' split sums (what the collectives move) against the public byte query
    tb = ws.tables(ctx, HV)
    moved = {}
    for coll in ("c1", "c2", "b2"):
        s, r = tb.split_elems(coll)
        moved[coll] = {"send": 4 * sum(s), "recv": 4 * sum(r)}
    s, r = tb.split_elems("b1_d")
    moved["b1"] = {"send": 4 * sum(s), "recv": 4 * sum(r)}
    s, r = tb.split_elems("b1_dm")
    moved["b1_dm"] = {"send": 4 * sum(s), "recv": 4 * sum(r)}
    info["moved_bytes"] = moved
    try:
        query = kcp.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        info["query_bytes"] = {
            c: query[c] for c in ("c1", "c2", "b1", "b2", "b1_design") if c in query
        }
        # save_transfer=False: "b1" = the plan's split tables (maps paired with the crossing-IN slots; what the
        # collective moves) must equal the b1_dm split sums; "b1_design" is the per-rank lower bound
        query_dm = kcp.cake_gdn_det_train_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False
        )
        info["query_bytes_dm"] = {
            c: query_dm[c] for c in ("b1", "b1_design") if c in query_dm
        }
        flags["bytes_match_query"] = (
            all(
                int(query[c]["send"]) == moved[c]["send"]
                and int(query[c]["recv"]) == moved[c]["recv"]
                for c in ("c1", "c2", "b2")
            )
            and int(query["b1"]["send"]) == moved["b1"]["send"]
            and int(query["b1"]["recv"]) == moved["b1"]["recv"]
            and int(query_dm["b1"]["send"]) == moved["b1_dm"]["send"]
            and int(query_dm["b1"]["recv"]) == moved["b1_dm"]["recv"]
        )
    except (
        ImportError,
        AttributeError,
        KeyError,
        TypeError,
    ) as exc:  # the byte module is another lane's deliverable
        info["query_bytes"] = f"unavailable: {type(exc).__name__}: {exc}"
        flags["bytes_match_query"] = False
    # G10: CUDA-graph capture of forward + backward (pointer-stable operands, caller-held workspace): the third call
    # of a key replays the captured graph and must be bitwise the eager result.  The backward's key includes the
    # saved tensors, which change once (the eager forward's fresh tensors -> the captured forward's static tensors),
    # so the backward reaches its replay one call after the forward: four forward + backward calls are made and the
    # per-call events are recorded (forward: eager, capture, replay, replay; backward: eager, eager, capture, replay).
    graph_info: dict[str, object] = {}
    try:
        kcp.graph_cache_clear()
        g_res = None
        events: list[list[str]] = []
        for _ in range(4):
            out, saved = kcp.cake_gdn_det_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, graph="replay", **base
            )
            grads = kcp.cake_gdn_det_train_kcp_backward(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                dout,
                saved=saved,
                workspace=ws,
                graph="replay",
                **base,
            )
            g_res = {
                "out": out.clone(),
                **{
                    n: t.clone()
                    for n, t in zip(
                        ("dq", "dk", "dv", "dbeta", "dg"), grads, strict=True
                    )
                },
            }
            events.append(
                [
                    str(kcp.LAST_CALL["fwd"]["graph_event"]),
                    str(kcp.LAST_CALL["bwd"]["graph_event"]),
                ]
            )
        graph_info["events"] = events
        graph_info["fwd_event"] = kcp.LAST_CALL["fwd"]["graph_event"]
        graph_info["bwd_event"] = kcp.LAST_CALL["bwd"]["graph_event"]
        graph_info["fwd_fallback"] = kcp.LAST_CALL["fwd"].get("graph_fallback")
        graph_info["bwd_fallback"] = kcp.LAST_CALL["bwd"].get("graph_fallback")
        graph_info["stats"] = dict(kcp.GRAPH_STATS)
        graph_ok = (
            graph_info["fwd_event"] == "replay"
            and graph_info["bwd_event"] == "replay"
            and _eq(res, g_res)
        )
        torch.cuda.synchronize(device)
        kcp.graph_cache_clear()
    except Exception as exc:  # noqa: BLE001 - recorded; the gate decides below
        graph_info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        graph_ok = False
        kcp.graph_cache_clear()
    info["graph"] = graph_info
    flags["graph_capture"] = bool(graph_ok)
    # #7038: the backward's time with a resident one-thread kernel on a side stream, over its time alone
    ratio = _side_stream_ratio(
        lambda: kcp.cake_gdn_det_train_kcp_backward(
            q, k, v, g, beta, ctx, dout, saved=res["saved"], workspace=ws, **base
        ),
        device,
    )
    info["side_stream"] = ratio
    flags["side_stream_ratio"] = ratio["ratio"] <= SIDE_STREAM_RATIO_MAX
    # no-leak rows: zero gradients outside the selected document and independence of its outputs
    if row.leak and total > 0:
        n_docs = len(cu) - 1
        doc = 1 if n_docs > 1 else 0
        mask = _doc_mask_local(part, cu, rank, doc, device)
        dout_sel = dout * mask.view(-1, 1, 1).to(dout.dtype)
        res_sel = saved_pass(dd=dout_sel)
        others = ~mask
        flags["zero_grad_leak"] = all(
            bool(torch.equal(res_sel[n][others], torch.zeros_like(res_sel[n][others])))
            for n in GRADS
        )
        alt = _local_inputs(
            row, part, cu, rank, device, seed=seed, doc_override=(doc, seed + 1)
        )
        res_alt = saved_pass(
            alt["q"], alt["k"], alt["v"], alt["g"], alt["beta"], alt["dout"]
        )
        flags["independence"] = all(
            torch.equal(res_alt[n][mask], res[n][mask]) for n in TENSORS
        )
        flags["independence_changed_others"] = (
            bool(others.any())
            and any(
                not torch.equal(res_alt[n][others], res[n][others]) for n in ("out",)
            )
            or not bool(others.any())
        )
    return res, flags


def _p4_draw(ctx, ws, ops: dict, backend: str) -> tuple[dict, bool]:
    """The forced ``P = 4`` plan (``segments=4``) on one draw's operands, on its own workspace (the row's other forms
    and contracts never see its pools): the six outputs and whether the forward and the backward launch records ran
    ``P = 4``."""
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    o = {"backend": backend, "segments": P4}
    out, saved = kcp.cake_gdn_det_train_kcp_forward(
        q, k, v, g, beta, ctx, workspace=ws, **o
    )
    p_fwd = int(kcp.LAST_CALL["fwd"]["P"])
    dq, dk, dv, dbeta, dg = kcp.cake_gdn_det_train_kcp_backward(
        q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, **o
    )
    ok = p_fwd == P4 and int(kcp.LAST_CALL["bwd"]["P"]) == P4
    return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}, ok


def _judge_full_draw(
    judge: TierJudge,
    failures: list[str],
    j: int,
    row: Row,
    part,
    cu: list[int],
    device,
    seed: int,
    gathered: dict[str, list],
) -> None:
    """Rank 0, draw ``j``: the six outputs reassembled to the global order (and those of the forced ``P = 4`` pass as
    ``p4.<output>``) against the FP64 oracle ``gdn_exact_with_grads`` beside its naive BF16 tier
    ``gdn_naive_with_grads`` on the same operands."""
    ref_mod = _reference()
    glob = _global_inputs(row, cu, device, seed=seed)
    ops = (
        glob["q"],
        glob["k"],
        glob["v"],
        glob["g"],
        glob["beta"],
        glob["dout"],
        torch.tensor(cu, dtype=torch.int64),
    )
    refs = ref_mod.gdn_exact_with_grads(*ops, scale=SCALE)
    naive = ref_mod.gdn_naive_with_grads(*ops, scale=SCALE)
    del glob, ops
    for name, tensors in gathered.items():
        n = name.rsplit(".", 1)[-1]  # "p4.dq" -> "dq"
        cand = _to_global(part, [t.to(device) for t in tensors], cu, cu[-1])
        _judge_add(judge, failures, name, j, cand, refs[n], naive[n])


def _run_rank(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """The body of one rank over the row's draws (:func:`_rank_draw`: every mode of the KCP forward / backward on
    draw 0, the core forms on every draw; on a ``p4`` row the forced ``P = 4`` pass of :func:`_p4_draw` on every
    draw); each draw's outputs are gathered to rank 0 and judged there against the FP64 oracle beside the naive BF16
    tier."""
    part, cu = _partition(row)
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    ws = kcp.CakeGDNDetKcpWorkspace()
    t_start = time.time()
    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    p4 = False
    if row.p4:
        p4 = _p_group_ok(ctx, row.heads[1], P4, group, W, device)
        flags["p4_allowed"] = p4
    ws4 = kcp.CakeGDNDetKcpWorkspace() if p4 else None
    judge, failures = (TierJudge(), []) if rank == 0 and total > 0 else (None, [])
    # an empty pack has nothing to judge: one draw of its bitwise forms
    for j in range(DRAWS if total > 0 else 1):
        seed = _draw_seed(j)
        ops = _local_inputs(row, part, cu, rank, device, seed=seed)
        res, fl = _rank_draw(
            rank,
            W,
            row,
            group,
            device,
            backend,
            part,
            cu,
            ctx,
            ws,
            ops,
            seed,
            info if j == 0 else {},
            full=j == 0,
        )
        gathered = {n: _gather(res[n].cpu(), W, group) for n in TENSORS}
        del res
        if p4:
            res, fl["p4_launch_P"] = _p4_draw(ctx, ws4, ops, backend)
            for n in TENSORS:
                gathered[f"p4.{n}"] = _gather(res[n].cpu(), W, group)
            del res
        del ops
        for key, ok in fl.items():
            flags[key] = flags.get(key, True) and bool(ok)
        if judge is not None:
            _judge_full_draw(judge, failures, j, row, part, cu, device, seed, gathered)
        del gathered
    flag_list = _gather(flags, W, group)
    info_list = _gather(info, W, group)
    report: dict = {
        "row": dataclasses.asdict(row),
        "rank": rank,
        "world": W,
        "backend": backend,
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "nccl": ".".join(map(str, torch.cuda.nccl.version())) if W > 1 else None,
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": True,
        "flags": {},
        "gated": [],
        "reported": [],
        "info_by_rank": info_list,
        "seconds": None,
    }
    # gates: bitwise / semantics / retained / bytes on every backend; graph capture is a gate of the kernel backend
    # (the torch emulation reports it); the per-row side-stream ratio is recorded on every backend -- the #7038 gate
    # of record is the module-timing protocol (slowest-rank median over >= 3 groups on an exclusive node)
    kernel_only = {"graph_capture"}
    info_only = {"side_stream_ratio"}
    for key in flags:
        ok = all(f[key] for f in flag_list)
        report["flags"][key] = ok
        if key in info_only or (key in kernel_only and backend != "kernels"):
            report["reported"].append(key)
        else:
            report["gated"].append(key)
            report["pass"] &= ok
    _judge_report(report, judge, failures)
    report["seconds"] = round(time.time() - t_start, 2)
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"], group=group)
        report["pass"] = all(verdict)
        dist.barrier(group=group)
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _side_stream_ratio(bwd, device: torch.device) -> dict:
    """#7038 repro form: the backward's wall time (CUDA events, median of ``SIDE_STREAM_ITERS``) alone and with a
    one-thread ``torch.cuda._sleep`` kernel resident on a side stream for the whole call."""

    def timed() -> float:
        torch.cuda.synchronize(device)
        e0, e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        bwd()
        e1.record()
        torch.cuda.synchronize(device)
        return float(e0.elapsed_time(e1))

    bwd()  # warm
    alone = [timed() for _ in range(SIDE_STREAM_ITERS)]
    med_alone = statistics.median(alone)
    clock_khz = (
        torch.cuda.get_device_properties(device).clock_rate
        if hasattr(torch.cuda.get_device_properties(device), "clock_rate")
        else 1_900_000
    )
    cycles = int(
        max(med_alone, 1.0) * 3.0 * clock_khz
    )  # ~3x the backward's duration in SM cycles
    side = torch.cuda.Stream(device=device)
    with_side = []
    for _ in range(SIDE_STREAM_ITERS):
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(cycles)
        with_side.append(timed())
        side.synchronize()
    med_side = statistics.median(with_side)
    return {
        "alone_ms": alone,
        "with_side_ms": with_side,
        "median_alone_ms": med_alone,
        "median_with_side_ms": med_side,
        "ratio": med_side / max(med_alone, 1e-9),
        "sleep_cycles": cycles,
    }


# ----------------------------------------------------------------------------------------------------------------
# launch forms


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_entry(
    rank: int, W: int, row: Row, port: int, result_dir: str, backend: str
) -> None:
    """One spawned rank (the verdict and rank 0's naive-tier verdicts travel through rank 0's report file; the
    parent test records them and asserts)."""
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    try:
        if W > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
            dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
        group = dist.group.WORLD if W > 1 else None
        _run_rank(rank, W, row, group, device, result_dir, backend)
    finally:
        kcp.graph_cache_clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def _expand_hostlist(s: str) -> list[str]:
    import re

    out: list[str] = []
    for item in re.findall(r"[^,\[\]]+(?:\[[^\]]*\])?", s):
        m = re.match(r"^([^\[]*)\[([^\]]*)\](.*)$", item)
        if not m:
            if item:
                out.append(item)
            continue
        pre, body, post = m.groups()
        for part in body.split(","):
            if "-" in part:
                a, b = part.split("-")
                for i in range(int(a), int(b) + 1):
                    out.append(f"{pre}{str(i).zfill(len(a))}{post}")
            elif part:
                out.append(f"{pre}{part}{post}")
    return out


def _slurm_rendezvous(rank: int, world: int) -> tuple[str, int]:
    """MASTER_ADDR from the STEP's nodelist; the port published per Slurm step by rank 0 in ``KCP_TEST_PORT_DIR``
    (shared filesystem) unless ``MASTER_PORT`` is set."""
    addr = os.environ.get("MASTER_ADDR")
    if not addr:
        hl = (
            os.environ.get("SLURM_STEP_NODELIST")
            or os.environ.get("SLURM_NODELIST")
            or socket.gethostname()
        )
        addr = _expand_hostlist(hl)[0]
    port_env = os.environ.get("MASTER_PORT")
    if port_env:
        return addr, int(port_env)
    job, step = (
        os.environ.get("SLURM_JOB_ID", "0"),
        os.environ.get("SLURM_STEP_ID", "0"),
    )
    d = os.environ.get("KCP_TEST_PORT_DIR") or os.path.join(
        os.environ.get("W", tempfile.gettempdir()), "tmp"
    )
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"kcp_test_port.{job}.{step}")
    if rank == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", 0))
            port = s.getsockname()[1]
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(f"{port}\n")
        os.replace(tmp, path)
        return addr, port
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            with open(path) as fh:
                txt = fh.read().strip()
            if txt:
                return addr, int(txt)
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"rank {rank}: no rendezvous port file {path} within 600 s")


class _Slurm:
    """The one-process-per-task form: WORLD initialised once; sub-groups of the ``W`` lowest ranks per row."""

    def __init__(self) -> None:
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world = int(os.environ.get("SLURM_NTASKS", "1"))
        local = int(os.environ.get("SLURM_LOCALID", self.rank))
        n = torch.cuda.device_count()
        self.device = torch.device("cuda", 0 if n == 1 else local % max(n, 1))
        torch.cuda.set_device(self.device)
        self.groups: dict[int, object] = {}
        if self.world > 1 and not dist.is_initialized():
            addr, port = _slurm_rendezvous(self.rank, self.world)
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = addr, str(port)
            # bounded: a rendezvous that never completes (a port collision with a live socket) or a failed row
            # must fail the step instead of holding its block; the longest legitimate collective wait (rank 0's
            # FP64 oracle of the largest row) stays well below it
            dist.init_process_group(
                "nccl",
                rank=self.rank,
                world_size=self.world,
                device_id=self.device,
                timeout=_dt.timedelta(seconds=WORLD_TIMEOUT_S),
            )
        # the ranks outside a row's group wait in the row's verdict exchange (``_slurm_row``) while the row runs.  A
        # production-stage row outlasts WORLD_TIMEOUT_S where the FP64 oracle is slow, and the NCCL watchdog would then
        # abort the waiting ranks, so the exchange runs on a CPU group of its own whose timeout covers the longest row.
        self.cpu = (
            dist.new_group(backend="gloo", timeout=_dt.timedelta(seconds=ROW_TIMEOUT_S))
            if self.world > 1
            else None
        )
        print(
            f"[kcp_dist] init timeout {WORLD_TIMEOUT_S} s; "
            f"rank {self.rank}/{self.world} local {local} host {socket.gethostname()} device {self.device} "
            f"{torch.cuda.get_device_name(self.device)} package={_mod().__file__} MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
            flush=True,
        )

    def group(self, W: int):
        """The sub-group of ranks ``0..W-1`` (every rank of WORLD must call this in the same order)."""
        if self.world == W:
            return dist.group.WORLD if self.world > 1 else None
        g = self.groups.get(W)
        if g is None:
            g = self.groups[W] = dist.new_group(ranks=list(range(W)))
        return g

    def close(self) -> None:
        kcp.graph_cache_clear()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


_SLURM: _Slurm | None = None


@pytest.fixture(scope="session")
def slurm():
    global _SLURM
    if os.environ.get("SLURM_PROCID") is None:
        yield None
        return
    if _SLURM is None:
        _SLURM = _Slurm()
    yield _SLURM
    _SLURM.close()


def _skip_unless_runnable(W: int, backend: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if backend == "kernels" and not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")


def _slurm_row(row: Row, slurm, backend: str, body, record_property=None) -> None:
    """One row under Slurm: sub-group of the row's ranks, WORLD-wide verdict: every rank (member or not) passes or
    fails the same row, so ``-x`` and the session teardown keep the collectives aligned across ranks.  Rank 0 records
    the row's naive-tier verdicts with ``record_property`` before the verdict is asserted."""
    W = row.W
    if slurm.world < W:
        pytest.skip(f"row needs {W} ranks, the step has {slurm.world}")
    group = slurm.group(
        W
    )  # collective over WORLD: every rank calls it, members and non-members alike
    if slurm.world > 1:
        dist.barrier()
    if backend == "kernels" and not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    report = None
    if slurm.rank < W:
        try:
            report = body(
                slurm.rank,
                W,
                row,
                group,
                slurm.device,
                os.environ.get("KCP_TEST_RESULT_DIR"),
                backend,
            )
        except Exception:
            # print now: the other ranks wait in a collective until its timeout, which would hide this rank's error
            print(
                f"rank {slurm.rank}: row {_row_id(row)} raised:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
    ok = report is None or bool(report["pass"])
    if slurm.world > 1:
        verdicts: list = [None] * slurm.world
        dist.all_gather_object(verdicts, ok, group=slurm.cpu)
        ok = all(verdicts)
    if slurm.rank >= W:
        if not ok:
            pytest.fail(f"rank {slurm.rank}: row failed on its {W} member ranks")
        pytest.skip(f"rank {slurm.rank} is not part of the row's {W}-rank group")
    _record_verdicts(report, record_property)
    assert ok, _summary(report)


def _summary(report: dict) -> str:
    """A row report without the per-rank records and the verdict list (the failures stay), for an assert message."""
    return json.dumps(
        {k: v for k, v in report.items() if k not in ("info_by_rank", "naive_tier")},
        default=str,
    )[:6000]


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_forward_rows(row: Row, slurm, record_property) -> None:
    """INT1 forward-only rows (``KCP_TEST_MODE=forward``): #1-#5 bitwise, ``out`` and ``S_in`` judged against the FP64
    forward and cut states (naive BF16 tier, 32 draws)."""
    if _test_mode() != "forward":
        pytest.skip("KCP_TEST_MODE=forward selects the forward-only rows")
    backend = _backend_name()
    if slurm is not None:
        _slurm_row(row, slurm, backend, _run_rank_fwd, record_property)
        return
    _skip_unless_runnable(row.W, backend)
    if row.W != 1:
        pytest.skip("forward-only rows outside Slurm run at W = 1 only")
    report = _run_rank_fwd(
        0,
        1,
        row,
        None,
        torch.device("cuda", 0),
        os.environ.get("KCP_TEST_RESULT_DIR"),
        backend,
    )
    _record_verdicts(report, record_property)
    assert report["pass"], _summary(report)


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_multi_gpu_rows(row: Row, slurm, record_property) -> None:
    """Forward + backward rows: the bitwise forms and contracts of :func:`_rank_draw`, and the six outputs judged
    against the FP64 oracle (naive BF16 tier, 32 draws)."""
    if _test_mode() != "full":
        pytest.skip("KCP_TEST_MODE=forward: the forward-only rows run instead")
    backend = _backend_name()
    if slurm is not None:
        _slurm_row(row, slurm, backend, _run_rank, record_property)
        return
    _skip_unless_runnable(row.W, backend)
    with tempfile.TemporaryDirectory(prefix="cake_gdn_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(row.W, row, _free_port(), result_dir, backend),
            nprocs=row.W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    _record_verdicts(report, record_property)
    assert report["pass"], _summary(report)


def test_public_api_cannot_select_test_programs() -> None:
    """The precision policy: the shipped record is ``precision_mode=0`` / ``transfer_walk="tf32_hilo"``;
    the test-only records (1 ``_three_mma``, 2 ``_plain_tf32``, ``"bf16x3"``) are refused unless ``test_knobs`` names
    the same value; nothing in the environment changes that (there is no environment hook)."""
    from cake_pplx._backends.cake_gdn_det_train_kcp import cake_options

    for mode in (1, 2):
        with pytest.raises(ValueError):
            cake_options.make_options(precision_mode=mode)
    with pytest.raises(ValueError):
        cake_options.make_options(transfer_walk="bf16x3")
    with pytest.raises(TypeError):
        cake_options.make_options(_kcp_test_knobs={"precision_mode": 1})
    o = cake_options.make_options()
    assert o.precision_mode == 0 and o.transfer_walk == "tf32_hilo"
    o = cake_options.make_options(precision_mode=2, test_knobs={"precision_mode": 2})
    assert o.precision_mode == 2


# ----------------------------------------------------------------------------------------------------------------
# C7: the convolution halo row.  The user calls ``cake_causal_conv1d_det_kcp(x_local, weight, ctx)`` with
# THIS package's context: the conv package reads the halo plan (``ctx.halo``, ``ctx.halo_tables``), the local
# ``cu_seqlens`` and the process group by duck typing.  The row's partition: document 0 = [1, 2, 300] on ranks 0,
# 1 and 3 (the three history rows of rank 3's first outputs come from TWO predecessor ranks), rank 2 holds no token
# (an empty predecessor), document 1 = [200] on rank 3 (a document start inside the continuing rank); C = 10,240.

CONV_C = 10240
CONV_WIDTH = 4
# the seed of the C7 operands: draw 0; draw j at draw_seed(CONV_SEED, j)
CONV_SEED = 43567


def _c7_fragments() -> list[list[tuple[int, int, int]]]:
    return [[(0, 0, 1)], [(0, 1, 2)], [], [(0, 3, 300), (1, 0, 200)]]


EXPLICIT["c7"] = (_c7_fragments, (303, 200))
C7_ROW = Row("C7", (303, 200), (2, 6), 4, "explicit:c7", stage="conv")


def _doc_shifts(cu: list[int], S: int, device):
    """``shifted(t, k)`` (``t[i - k]`` of the same document, else 0) and ``shifted_up(t, k)`` (``t[i + k]`` of the
    same document, else 0) over the ``S`` packed tokens of ``cu``: exact copies in the dtype of ``t``."""
    doc = torch.zeros(S, dtype=torch.int64, device=device)
    for i in range(len(cu) - 1):
        doc[cu[i] : cu[i + 1]] = i

    def shifted(t, k):  # t[i - k] of the same document, else 0
        if k == 0:
            return t.clone()
        out = torch.zeros_like(t)
        if k < S:
            same = doc[k:] == doc[:-k]
            out[k:] = torch.where(
                same[:, None], t[: S - k], torch.zeros_like(t[: S - k])
            )
        return out

    def shifted_up(t, k):  # t[i + k] of the same document, else 0
        if k == 0:
            return t.clone()
        out = torch.zeros_like(t)
        if k < S:
            same = doc[k:] == doc[:-k]
            out[: S - k] = torch.where(same[:, None], t[k:], torch.zeros_like(t[k:]))
        return out

    return shifted, shifted_up


def _conv_reference(x: torch.Tensor, w: torch.Tensor, cu: list[int], dy: torch.Tensor):
    """FP64 oracle of the unpartitioned width-4 depthwise causal convolution + SiLU with the conv package's KCP
    semantics (``weight[c, 3]`` multiplies the current token; zero history at a document start; the SiLU
    derivative on the unrounded preactivation): FP64 ``y``, ``dx`` and ``dpre``, and the taps for the per-rank
    ``dweight`` partials."""
    shifted, shifted_up = _doc_shifts(cu, x.shape[0], x.device)
    xf, wd = x.double(), w.double()
    halo = CONV_WIDTH - 1
    pre = sum(wd[:, k][None, :] * shifted(xf, halo - k) for k in range(CONV_WIDTH))
    s = torch.sigmoid(pre)
    y = pre * s
    dpre = dy.double() * s * (1 + pre * (1 - s))
    dx = sum(wd[:, k][None, :] * shifted_up(dpre, halo - k) for k in range(CONV_WIDTH))
    taps = [shifted(xf, halo - k) for k in range(CONV_WIDTH)]
    return y, dx, dpre, taps


def _stored(t: torch.Tensor) -> torch.Tensor:
    """``t`` as the naive BF16 tier reads it: BF16 when stored no wider, else as stored (the FP32 weight)."""
    return t.to(torch.bfloat16) if t.dtype.itemsize <= 2 else t


def _conv_naive(x: torch.Tensor, w: torch.Tensor, cu: list[int], dy: torch.Tensor):
    """The naive tier of :func:`_conv_reference`: the same ops on the same inputs in PyTorch BF16 -- ``x`` and ``dy``
    read as BF16, the FP32 weight as stored (its products with BF16 rows run in FP32 by torch's type promotion) --
    with ``y`` and ``dx`` rounded to BF16; ``dpre`` and the BF16 taps for the per-rank ``dweight`` partials (rounded
    to BF16 by :func:`_dw_partials`)."""
    shifted, shifted_up = _doc_shifts(cu, x.shape[0], x.device)
    xb, wb, dyb = _stored(x), _stored(w), _stored(dy)
    halo = CONV_WIDTH - 1
    pre = sum(wb[:, k][None, :] * shifted(xb, halo - k) for k in range(CONV_WIDTH))
    s = torch.sigmoid(pre)
    y = (pre * s).to(torch.bfloat16)
    dpre = dyb * s * (1 + pre * (1 - s))
    dx = sum(
        wb[:, k][None, :] * shifted_up(dpre, halo - k) for k in range(CONV_WIDTH)
    ).to(torch.bfloat16)
    taps = [shifted(xb, halo - k) for k in range(CONV_WIDTH)]
    return y, dx, dpre, taps


def _dw_partials(dpre, taps, sels, dtype=None) -> list[torch.Tensor]:
    """Every rank's ``dweight`` partial ``[C, 4]`` (the framework reduces the ranks): the sum over the rank's
    tokens ``sel`` of ``dpre`` times each tap, rounded to ``dtype`` when given (the naive tier)."""
    out = []
    for sel in sels:
        dw = torch.stack([(dpre[sel] * tap[sel]).sum(0) for tap in taps], dim=1)
        out.append(dw if dtype is None else dw.to(dtype))
    return out


def _conv_operands(seed: int, total: int, C: int = CONV_C):
    """The convolution operands of one draw (CPU, the same on every rank): ``x`` and ``dy`` ``[total, C]`` BF16 and
    the FP32 weight ``[C, 4]``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x_all = (torch.randn(total, C, generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(C, CONV_WIDTH, generator=gen) * 0.5).float()
    dy_all = (torch.randn(total, C, generator=gen) * 0.5).to(torch.bfloat16)
    return x_all, w, dy_all


def _judge_conv_draw(
    judge: TierJudge,
    failures: list[str],
    j: int,
    part,
    cu: list[int],
    W: int,
    device,
    x_all: torch.Tensor,
    w: torch.Tensor,
    dy_all: torch.Tensor,
    got: list[dict],
) -> None:
    """Rank 0, draw ``j`` of C7: ``y`` / ``dx`` reassembled to the global order and every rank's ``dweight`` partial
    (``dw.r<rank>``) against the FP64 oracle of the whole pack beside its naive BF16 tier."""
    x, wd, dy = x_all.to(device), w.to(device), dy_all.to(device)
    y_ref, dx_ref, dpre, taps = _conv_reference(x, wd, cu, dy)
    y_nv, dx_nv, dpre_nv, taps_nv = _conv_naive(x, wd, cu, dy)
    for n, ref, nv in (("y", y_ref, y_nv), ("dx", dx_ref, dx_nv)):
        cand = _to_global(part, [r_[n].to(device) for r_ in got], cu, cu[-1])
        _judge_add(judge, failures, n, j, cand, ref, nv)
    sels = [_local_to_global(part, cu, r).to(device) for r in range(W)]
    refs = _dw_partials(dpre, taps, sels)
    naive = _dw_partials(dpre_nv, taps_nv, sels, torch.bfloat16)
    for r in range(W):
        _judge_add(
            judge, failures, f"dw.r{r}", j, got[r]["dw"].to(device), refs[r], naive[r]
        )


def _run_rank_conv(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """C7 on one rank over 32 draws: on every draw ``y`` bitwise the single-rank conv of the whole pack on the rank's
    tokens, its repeat bitwise, the shapes, the halo-row gradients delivered to the predecessor ranks (rank 0 / 1: 1 /
    2 tokens of document 0, non-zero) and the empty rank; on rank 0 ``y`` / ``dx`` / every rank's ``dweight`` partial
    judged against the FP64 oracle beside the naive BF16 tier (:func:`_judge_conv_draw`)."""
    from cake_pplx import cake_causal_conv1d_det_train as conv
    from cake_pplx import cake_causal_conv1d_det_train_kcp as conv_kcp

    part, cu = _partition(row)
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    l2g = _local_to_global(part, cu, rank)
    sel = l2g.to(device)
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    flags: dict[str, bool] = {}
    judge, failures = (TierJudge(), []) if rank == 0 else (None, [])
    t0 = time.time()
    for j in range(DRAWS):
        x_all, w, dy_all = _conv_operands(draw_seed(CONV_SEED, j), total)
        x_local = x_all[l2g].to(device).requires_grad_(True)
        w_dev = w.to(device).requires_grad_(True)
        y = conv_kcp.cake_causal_conv1d_det_kcp(x_local, w_dev, ctx)
        y.backward(dy_all[l2g].to(device))
        dx, dw = x_local.grad, w_dev.grad
        y2 = conv_kcp.cake_causal_conv1d_det_kcp(x_local.detach(), w_dev.detach(), ctx)
        # the single-rank conv of the whole pack
        y_single = conv.cake_causal_conv1d_det(x_all.to(device), w.to(device), cu_dev)
        torch.cuda.synchronize(device)
        fl = {
            "y_eq_single_rank": torch.equal(y, y_single[sel]),
            "repeat": torch.equal(y, y2),
            "shapes": tuple(y.shape) == tuple(x_local.shape)
            and tuple(dx.shape) == tuple(x_local.shape),
        }
        # the halo rows: the predecessor ranks' tokens feed rank 3's first outputs, so their dx must carry that term
        # (non-zero), and a rank without tokens has empty y / dx
        if rank in (0, 1):
            fl["halo_grad_nonzero"] = bool((dx.float().abs() > 0).any().item())
        if rank == 2:
            fl["empty_rank"] = y.numel() == 0 and dx.numel() == 0
        for key, ok in fl.items():
            flags[key] = flags.get(key, True) and bool(ok)
        got = _gather({"y": y.detach().cpu(), "dx": dx.cpu(), "dw": dw.cpu()}, W, group)
        del x_local, w_dev, y, y2, dx, dw, y_single
        if judge is not None:
            _judge_conv_draw(
                judge, failures, j, part, cu, W, device, x_all, w, dy_all, got
            )
        del got
    report = {
        "row": dataclasses.asdict(row),
        "rank": rank,
        "world": W,
        "backend": "conv",
        "mode": "conv",
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": all(flags.values()),
        "flags": flags,
        "gated": sorted(flags),
        "reported": [],
        "info_by_rank": [
            {
                "rank": rank,
                "T_r": int(l2g.numel()),
                "l2g_first": int(l2g[0]) if len(l2g) else None,
            }
        ],
        "seconds": None,
        "C": CONV_C,
    }
    _judge_report(report, judge, failures)
    report["seconds"] = round(time.time() - t0, 2)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, f"{row.name}-conv-rank{rank}.json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def test_conv_halo_row(slurm, record_property) -> None:
    """C7: the causal_conv1d family's ``cake_causal_conv1d_det_kcp`` through this package's context (duck typing):
    ``y`` / ``dx`` / ``dweight`` judged against FP64 (naive BF16 tier, 32 draws), ``y`` bitwise the single-rank conv,
    history rows from two predecessor ranks, an empty predecessor and a document start.  Runs under Slurm with >= 4
    ranks."""
    if slurm is None:
        pytest.skip("the conv halo row runs under Slurm (4 ranks)")
    conv = pytest.importorskip("cake_pplx.cake_causal_conv1d_det_train")
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(C7_ROW, slurm, "conv", _run_rank_conv, record_property)


# ----------------------------------------------------------------------------------------------------------------
# Whole-layer checkpoint rows (issue #7818): KCP convolution -> packed q | k | v views -> caller L2 norm of q, k ->
# cake_gdn_det_kcp, one checkpointed region per layer with one CakeGDNDetKcpRecomputeHandle.  Gated: out / dx / dweight / dg /
# dbeta bitwise across the direct pass, the plain checkpoint and the handle checkpoint; the GDN part of the
# recompute exchanges nothing (with a convolution that takes the handle the whole recompute exchanges nothing;
# before that, exactly the convolution's own forward collectives remain, the whole-layer zero is reported as
# pending); the original forward and the backward exchange as in the plain checkpoint.  Runs under Slurm.

LAYER_ROWS = [
    Row("L4", (303, 200), (2, 6), 4, "explicit:c7", stage="layer"),
    Row(
        "L8",
        (10240, 8192, 2085, 333),
        (16, 48),
        8,
        "zigzag_per_document",
        stage="layer",
    ),
    Row(
        "L8r",
        (10240, 8192, 2085, 333),
        (16, 48),
        8,
        "ragged:5000,0,1,3000,2849,4000,3000,3000",
        stage="layer",
    ),
]


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """The caller's per-head L2 norm of ``[T, H, 128]`` BF16 (FP32 math, one rounding): deterministic, so the
    checkpoint recompute reproduces it bit for bit."""
    xf = x.float()
    return (xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


def _run_rank_layer(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
    *,
    l2n: bool = False,
) -> dict:
    """One whole-layer row (``l2n``: #7817, the in-kernel L2 norm on the raw convolution output views instead of the
    caller's norm)."""
    import inspect

    from cake_pplx import cake_causal_conv1d_det_train_kcp as conv_kcp

    H_qk, HV = row.heads
    C = (2 * H_qk + HV) * 128
    part, cu = _partition(row)
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    gen = torch.Generator(device="cpu").manual_seed(78180)
    x_all = (torch.randn(total, C, generator=gen) * 0.5).to(torch.bfloat16)
    w_all = (torch.randn(C, CONV_WIDTH, generator=gen) * 0.5).float()
    pieces = [
        torch.arange(cu[f.doc] + f.off, cu[f.doc] + f.off + f.length)
        for f in part.fragments[rank]
        if f.length
    ]
    l2g = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.int64)
    x_local = x_all[l2g].to(device)
    ops = _local_inputs(row, part, cu, rank, device)
    g, beta, dout = ops["g"], ops["beta"], ops["dout"]
    conv_takes_handle = (
        "recompute_handle"
        in inspect.signature(conv_kcp.cake_causal_conv1d_det_kcp).parameters
    )

    def layer(x, w, g_, b_, h):
        ckw = {"recompute_handle": h} if (h is not None and conv_takes_handle) else {}
        y = conv_kcp.cake_causal_conv1d_det_kcp(x, w, ctx, **ckw)
        qkv = y.view(y.shape[0], 2 * H_qk + HV, 128)
        if l2n:  # #7817: the raw views; the recurrence normalizes q / k itself
            q, k = qkv[:, :H_qk], qkv[:, H_qk : 2 * H_qk]
        else:
            q = _l2_normalize(qkv[:, :H_qk])
            k = _l2_normalize(qkv[:, H_qk : 2 * H_qk])
        v = qkv[:, 2 * H_qk :]
        return kcp.cake_gdn_det_kcp(
            q,
            k,
            v,
            g_,
            b_,
            ctx,
            recompute_handle=h,
            save_transfer=False,
            backend=backend,
            **({"use_qk_l2norm_in_kernel": True} if l2n else {}),
        )

    def run(mode: str):
        leaves = [
            x_local.detach().clone().requires_grad_(True),
            w_all.to(device).requires_grad_(True),
            g.detach().clone().requires_grad_(True),
            beta.detach().clone().requires_grad_(True),
        ]
        hd = kcp.CakeGDNDetKcpRecomputeHandle() if mode == "handle" else None
        from torch.utils.checkpoint import checkpoint

        with CollectiveCounter() as cc:
            if mode == "direct":
                out = layer(*leaves, None)
            else:
                rphase = RecomputePhase(cc)
                out = checkpoint(
                    layer,
                    *leaves,
                    hd,
                    use_reentrant=False,
                    context_fn=lambda: (contextlib.nullcontext(), rphase),
                )
            cc.phase = "bwd"
            out.backward(dout)
        names = ("out", "dx", "dw", "dg", "dbeta")
        tensors = [out.detach()] + [lf.grad for lf in leaves]
        return dict(zip(names, tensors, strict=True)), {
            **cc.record(),
            "handle_bytes": hd.nbytes() if hd is not None else 0,
            "handle_keys": sorted(hd.entries) if hd is not None else [],
        }

    # the convolution's own forward collectives (its halo exchange), for the pending whole-layer count
    with CollectiveCounter() as cc_conv:
        conv_kcp.cake_causal_conv1d_det_kcp(x_local, w_all.to(device), ctx)
    conv_fwd = cc_conv.record()
    t0 = time.time()
    res_d, rec_d = run("direct")
    res_p, rec_p = run("plain")
    res_h, rec_h = run("handle")
    eq = lambda a, b: all(torch.equal(a[n], b[n]) for n in a)  # noqa: E731
    flags: dict[str, bool] = {
        "layer_handle_eq_ckpt": eq(res_p, res_h),
        "layer_handle_eq_direct": eq(res_d, res_h),
        "layer_orig_bwd_comm_unchanged": all(
            rec_h[t].get(ph, 0) == rec_p[t].get(ph, 0)
            for t in ("calls", "waits")
            for ph in ("orig", "bwd")
        ),
    }
    rc_calls = rec_h["calls"].get("recompute", 0)
    rc_waits = rec_h["waits"].get("recompute", 0)
    if conv_takes_handle:
        flags["layer_recompute_no_comm"] = rc_calls == 0 and rc_waits == 0
    else:
        # only the convolution's own halo exchange is left in the recompute: the GDN part exchanges nothing
        flags["layer_gdn_recompute_no_comm"] = rc_calls == conv_fwd["calls"].get(
            "orig", 0
        ) and rc_waits == conv_fwd["waits"].get("orig", 0)
    report = {
        "row": dataclasses.asdict(row),
        "rank": rank,
        "world": W,
        "backend": backend,
        "mode": "layer-l2n" if l2n else "layer",
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": all(flags.values()),
        "flags": flags,
        "gated": sorted(flags),
        "reported": []
        if conv_takes_handle
        else ["layer_recompute_no_comm (pending: conv recompute_handle)"],
        "info_by_rank": [
            {
                "rank": rank,
                "T_r": int(x_local.shape[0]),
                "conv_takes_handle": conv_takes_handle,
                "conv_fwd_collectives": conv_fwd,
                "direct": rec_d,
                "plain": rec_p,
                "handle": rec_h,
                "plain_recompute_calls": rec_p["calls"].get("recompute", 0),
                "handle_recompute_calls": rc_calls,
                "handle_recompute_waits": rc_waits,
            }
        ],
        "seconds": round(time.time() - t0, 2),
        "C": C,
    }
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(
                result_dir, f"{row.name}-layer{'-l2n' if l2n else ''}-rank{rank}.json"
            ),
            "w",
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


@pytest.mark.parametrize("row", LAYER_ROWS, ids=_row_id)
def test_layer_checkpoint_rows(row: Row, slurm) -> None:
    """#7818 whole-layer rows: KCP convolution + L2 norm + cake_gdn_det_kcp in one checkpointed region with one
    recompute handle (see the block comment above).  Runs under Slurm with >= the row's ranks."""
    if slurm is None:
        pytest.skip("the layer rows run under Slurm")
    conv = pytest.importorskip("cake_pplx.cake_causal_conv1d_det_train")
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(row, slurm, _backend_name(), _run_rank_layer)


# ----------------------------------------------------------------------------------------------------------------
# #7817: the in-kernel Q / K L2 norm (``use_qk_l2norm_in_kernel=True``).  ``q`` / ``k`` are RAW rows here (the
# generator's rows times a per-token, per-head power of two drawn from the global token index, so every rank sees
# the bits of the global pack).  Per row and rank:
#
# * mode equalities with the norm (bitwise): repeat, light == full (saved ``qn`` / ``kn``) == full with test knob
#   ``full_qk="apply"`` (``qk_apply``) == boundary replay, autograd ==
#   saved, checkpoint (replay form, kept tapes, "auto") == direct, recompute handle == direct, ``save_transfer``,
#   ring == a2a (where the partition allows it), CUDA-graph replay == eager, ``.backward()`` twice;
# * the composed chain: out / dv / dbeta / dg == the norm-off route on the materialized ``(qn, kn)`` (the forward's
#   own rows, ``record_qk``); dq / dk == ``bf16(l2n_pullback(x, r, D))`` with ``D`` the norm-off route's
#   ``group_reduce="host"`` partials summed in head order (the oracle of #8_l2n's fused group sum + pullback);
# * #1_l2n: its eight tapes == the norm-off #1 on the materialized ``(qn, kn)``; its ``qn`` / ``kn`` / ``r`` == the
#   ``cake_l2n`` reference (CPU, FTZ off); ``qk_apply`` on its ``r`` == its ``qn`` / ``kn``; the backward's rows and
#   regenerated ``r`` == the forward's;
# * the norm-off route launches no record of the in-kernel norm's programs (``L2N_PROGRAMS``);
# * packed QKV views with the norm: bitwise the contiguous call, the split still aliases the packed dQKV row;
# * memory: no ``qn`` / ``kn`` / ``qk_rstd`` in the arena after a forward or a backward; the bytes a forward retains
#   with the norm == without + ``8 T_r H_qk`` (the saved r); saved bytes on - off == the query's ``qk_rstd`` term;
#   with ``save_mode="full"`` both grow by the saved ``qn`` / ``kn`` (``512 T_r H_qk``) as well;
#   the peaks and a profiler count of copy ops on the packed views are recorded.
#
# The rows skip with a clear reason while the registry does not ship the three records (the kernel backend refuses
# the option then); ``KCP_TEST_BACKEND=fake`` runs them on the torch emulation.

L2N_SKIP = (
    "the in-kernel Q/K L2 norm records (fwd_prep_l2n, bwd_dqkg_l2n, qk_apply) are not in the shipped registry "
    "yet: use_qk_l2norm_in_kernel cannot run on the kernel backend"
)
L2N = {"use_qk_l2norm_in_kernel": True}


def _l2n_shipped() -> bool:
    shipped = getattr(kcp, "l2n_records_shipped", None)
    return bool(shipped()) if callable(shipped) else False


def _skip_unless_l2n(backend: str) -> None:
    if backend == "kernels" and not _l2n_shipped():
        pytest.skip(L2N_SKIP)


def _raw_qk(
    row: Row, part, cu: list[int], rank: int, x: torch.Tensor, salt: int
) -> torch.Tensor:
    """``x`` (this rank's L2-normalized generator rows) times ``2^e``, ``e`` in [-6, 6] per (global token, head)."""
    total, H = cu[-1], x.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(78170 + salt)
    e = torch.randint(-6, 7, (max(total, 1), H, 1), generator=gen).float()
    pieces = [
        torch.arange(cu[f.doc] + f.off, cu[f.doc] + f.off + f.length)
        for f in part.fragments[rank]
        if f.length
    ]
    l2g = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.int64)
    return (x.float() * torch.exp2(e[l2g]).to(x.device)).to(torch.bfloat16)


def _prep_tapes(
    ctx, q, k, v, g, beta, *, l2n: bool, backend: str, rstd=None, apply=False
) -> dict:
    """One #1 (``l2n``: #1_l2n) -- or with ``apply`` one ``qk_apply`` on ``rstd`` -- on zero-initialised tapes of a
    fresh workspace; returns copies of the eight #1 tapes and the norm's rows."""
    opts = kcp.make_options(backend=backend, use_qk_l2norm_in_kernel=l2n)
    call, _ = kcp._prepare(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        scale=None,
        opts=opts,
        workspace=kcp.CakeGDNDetKcpWorkspace(),
        backward=apply,
        dout=(torch.zeros(v.shape, dtype=v.dtype, device=v.device) if apply else None),
    )
    names = ("w", "u", "aqk", "akk", "g_cum", "rowscale", "qscale", "gamma")
    for n in names:
        call.tapes[n] = call.tape(n).zero_()
    if l2n:
        for n in (*kcp.QK_TAPES, kcp.QK_RSTD):
            call.tapes[n] = call.tape(n).zero_()
    if apply:
        call.bufs[kcp.QK_RSTD] = rstd
        call.local("qk_apply")
    else:
        call.local("fwd_prep")
    torch.cuda.synchronize(q.device)
    out = {} if apply else {n: call.tape(n).clone() for n in names}
    if l2n:
        out.update({n: call.tape(n).clone() for n in kcp.QK_TAPES})
        out["rstd"] = call.rstd_in().clone() if apply else call.rstd_out().clone()
    return out


def _run_rank_l2n(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    l2n = _mod("cake_l2n")
    gb = _mod("cake_bytes")
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.cake_gdn_det_train_kcp_build_context(part, group, rank, device)
    ops = _local_inputs(row, part, cu, rank, device)
    q = _raw_qk(row, part, cu, rank, ops["q"], 1)
    k = _raw_qk(row, part, cu, rank, ops["k"], 2)
    v, g, beta, dout = (ops[n] for n in ("v", "g", "beta", "dout"))
    T_r = int(q.shape[0])
    ws = kcp.CakeGDNDetKcpWorkspace()
    base = {"backend": backend}
    t_start = time.time()
    flags: dict[str, bool] = {}
    info: dict[str, object] = {"T_r": T_r}

    def saved_pass(qq=q, kk=k, workspace=ws, **opts) -> dict:
        o = {**base, **opts}
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            qq, kk, v, g, beta, ctx, workspace=workspace, **o
        )
        rec_f = dict(kcp.LAST_CALL["fwd"])
        dq, dk, dv, dbeta, dg = kcp.cake_gdn_det_train_kcp_backward(
            qq, kk, v, g, beta, ctx, dout, saved=saved, workspace=workspace, **o
        )
        return {
            "out": out,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dg": dg,
            "dbeta": dbeta,
            "saved": saved,
            "fwd": rec_f,
            "bwd": dict(kcp.LAST_CALL["bwd"]),
        }

    def autograd_pass(ckpt: bool, *, handle=False, twice=False, **opts) -> dict:
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o = {**base, **opts}
        hd = kcp.CakeGDNDetKcpRecomputeHandle() if handle else None

        def fn(q_, k_, v_, g_, b_, h_):
            return kcp.cake_gdn_det_kcp(
                q_, k_, v_, g_, b_, ctx, workspace=ws, recompute_handle=h_, **o
            )

        if ckpt:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, hd, use_reentrant=False)
        else:
            out = fn(*leaves, hd)
        out.backward(dout, retain_graph=twice)
        if twice:
            out.backward(dout)
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(GRADS, leaves, strict=True)},
        }

    # the norm-off route launches no record of the in-kernel norm's programs
    jit = _mod("cake_jit")
    l2n_records = {
        name
        for name, rec in {**jit.MODULES, **jit.TEST_MODULES}.items()
        if rec.get("program") in kcp.L2N_PROGRAMS
    }
    res_off = saved_pass()
    used = {r for r, _g in res_off["fwd"]["records"]} | {
        r for r, _g in res_off["bwd"]["records"]
    }
    info["norm_off_records"] = sorted(used)
    flags["norm_off_launches_no_l2n_record"] = (
        not (used & l2n_records) and not res_off["fwd"]["qk_l2norm"]
    )

    # the norm on: mode equalities
    res = saved_pass(**L2N, test_knobs={"record_qk": True})
    rec_fwd_qk, rec_bwd_qk = res["fwd"].get("qk"), res["bwd"].get("qk")
    info["fwd_record"] = {
        k_: v_
        for k_, v_ in res["fwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges", "qk")
    }
    info["bwd_record"] = {
        k_: v_
        for k_, v_ in res["bwd"].items()
        if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges", "qk")
    }
    info["fwd_sequence"], info["bwd_sequence"] = (
        list(res["fwd"]["sequence"]),
        list(res["bwd"]["sequence"]),
    )
    flags["l2n_repeat"] = _eq(res, saved_pass(**L2N))
    full = saved_pass(save_mode="full", **L2N, test_knobs={"record_qk": True})
    flags["l2n_light_eq_full"] = _eq(res, full) and (
        T_r == 0
        or (
            full["bwd"]["qk_source"] == "saved"
            and "qk_apply" not in full["bwd"]["sequence"]
        )
    )
    full_apply = saved_pass(
        save_mode="full", **L2N, test_knobs={"record_qk": True, "full_qk": "apply"}
    )
    flags["l2n_light_eq_full_apply"] = _eq(res, full_apply) and (
        T_r == 0 or full_apply["bwd"]["qk_source"] == "qk_apply"
    )
    flags["l2n_saved_eq_replay"] = _eq(res, saved_pass(boundary_mode="replay", **L2N))
    flags["l2n_save_transfer_eq"] = _eq(res, saved_pass(save_transfer=False, **L2N))
    res_auto = autograd_pass(False, **L2N)
    flags["l2n_autograd_eq_saved"] = _eq(res, res_auto)
    for name, opts in (
        ("ckpt_replay_form", {"reuse_recompute_tapes": False}),
        ("ckpt_kept", {"reuse_recompute_tapes": True}),
        ("ckpt_auto", {"reuse_recompute_tapes": "auto"}),
        ("ckpt_full", {"save_mode": "full"}),
        ("ckpt_boundary_replay", {"boundary_mode": "replay"}),
    ):
        flags[f"l2n_{name}_eq_direct"] = _eq(
            res_auto, autograd_pass(True, **opts, **L2N)
        )
        if name == "ckpt_kept":
            info["kept_tapes_source"] = kcp.LAST_CALL["bwd"].get("tapes_source")
            info["kept_qk_source"] = kcp.LAST_CALL["bwd"].get("qk_source")
    flags["l2n_handle_eq_direct"] = _eq(
        res_auto, autograd_pass(True, handle=True, save_transfer=False, **L2N)
    )
    twice = autograd_pass(True, handle=True, twice=True, save_transfer=False, **L2N)
    flags["l2n_twice_accumulates"] = all(
        torch.equal(twice[n], res_auto[n] + res_auto[n]) for n in GRADS
    )
    if W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS:
        flags["l2n_ring_eq_a2a"] = _eq(
            res, saved_pass(exchange="ring", save_transfer=False, **L2N)
        )
    # save_mode="full" without save_transfer (the keyword, or the ring exchange) on NaN-filled allocations, as in the
    # norm-off row: at P > 1 the full backward's merges read the saved M_seg^T plane
    with _nan_allocations():
        flags["l2n_full_no_transfer_eq"] = _eq(
            res, saved_pass(save_mode="full", save_transfer=False, **L2N)
        )
        if W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS:
            flags["l2n_full_ring_eq_a2a"] = _eq(
                res,
                saved_pass(
                    save_mode="full", exchange="ring", save_transfer=False, **L2N
                ),
            )
    # the forced P = 2 forms (see _run_rank)
    info["p2_eligible"] = _p_group_ok(ctx, HV, 2, group, W, device)
    if info["p2_eligible"]:
        _p2_full_flags(
            saved_pass,
            info,
            flags,
            ring=W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS,
            prefix="l2n_",
            **L2N,
        )
    res_host = saved_pass(group_reduce="host", **L2N)
    info["l2n_host_eq_fused"] = _eq(
        res, res_host
    )  # reported, as the norm-off host == fused

    # the composed chain on the forward's own rows.  The reference pass exchanges states, so every rank issues it (a
    # rank without tokens passes its empty rows); skipping it there would put the ranks' collectives out of step.
    ws0 = kcp.CakeGDNDetKcpWorkspace()
    ref = saved_pass(
        qq=(rec_fwd_qk["qn"] if T_r > 0 else q).clone(),
        kk=(rec_fwd_qk["kn"] if T_r > 0 else k).clone(),
        workspace=ws0,
        group_reduce="host",
    )
    if T_r > 0:
        qn, kn, rstd = rec_fwd_qk["qn"], rec_fwd_qk["kn"], rec_fwd_qk["rstd"]
        flags["l2n_bwd_rows_eq_fwd"] = all(
            torch.equal(rec_bwd_qk[n], rec_fwd_qk[n]) for n in ("qn", "kn", "rstd")
        )
        flags["l2n_regen_rstd_eq_saved"] = bool(rec_bwd_qk.get("rstd_regen_equal"))
        flags["l2n_full_rows_eq_fwd"] = all(
            torch.equal(full["bwd"]["qk"][n], rec_fwd_qk[n])
            for n in ("qn", "kn", "rstd")
        )
        qh, rq = l2n.qk_l2norm_ref(q.cpu())
        kh, rk = l2n.qk_l2norm_ref(k.cpu())
        flags["l2n_qk_eq_helper"] = (
            torch.equal(qn.cpu(), qh)
            and torch.equal(kn.cpu(), kh)
            and torch.equal(rstd.cpu(), torch.stack([rq, rk]))
        )
        flags["l2n_eq_reference_out_dv_dbeta_dg"] = _eq(
            res, ref, ("out", "dv", "dbeta", "dg")
        )
        # the host form's D is the head-order sum of the partials; the fused #8_l2n's FP32 group sum is that sum
        # only at one value head per Q/K head (r > 1: the fused form is reported against it)
        r_ = HV // H_qk
        ok_host = ok_fused = True
        for name, part, x, plane in (("dq", "dq_part", q, 0), ("dk", "dk_part", k, 1)):
            p = ws0.tape(part).view(T_r, H_qk, r_, 128)
            acc = p[:, :, 0]
            for j in range(1, r_):
                acc = acc + p[:, :, j]
            want = l2n.l2n_pullback(x, rstd[plane], acc).to(torch.bfloat16)
            ok_host = ok_host and torch.equal(res_host[name], want)
            ok_fused = ok_fused and torch.equal(res[name], want)
        flags["l2n_host_dqdk_eq_partials_pullback"] = ok_host
        if r_ == 1:
            flags["l2n_fused_dqdk_eq_partials_pullback"] = ok_fused
        else:
            info["fused_dqdk_eq_partials_pullback"] = ok_fused
        # #1_l2n against the norm-off #1 on the materialized rows, the reference, and qk_apply
        on = _prep_tapes(ctx, q, k, v, g, beta, l2n=True, backend=backend)
        off = _prep_tapes(
            ctx, on["qn"], on["kn"], v, g, beta, l2n=False, backend=backend
        )
        flags["l2n_prep_tapes_eq_norm_off"] = all(
            torch.equal(on[n], off[n]) for n in off
        )
        flags["l2n_prep_rows_eq_forward"] = torch.equal(on["qn"], qn) and torch.equal(
            on["rstd"], rstd
        )
        ap = _prep_tapes(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            l2n=True,
            backend=backend,
            rstd=on["rstd"].clone(),
            apply=True,
        )
        flags["l2n_qk_apply_eq_prep"] = torch.equal(ap["qn"], on["qn"]) and torch.equal(
            ap["kn"], on["kn"]
        )

    # packed QKV views with the norm: bitwise, and the split still aliases the packed dQKV row
    buf, qs, ks, vs = _packed_views(q, k, v)
    flags["l2n_strided_eq_contiguous"] = _eq(res, saved_pass(qq=qs, kk=ks, **L2N))
    split_before = dict(kcp.PACKED_SPLIT_STATS)
    qkv_leaf = buf.detach().clone().requires_grad_(True)
    g_leaf, b_leaf = (x.detach().clone().requires_grad_(True) for x in (g, beta))
    pq, pk, pv = kcp.cake_gdn_det_train_kcp_packed_qkv_views(qkv_leaf, H_qk=H_qk, HV=HV)
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            out_split = kcp.cake_gdn_det_kcp(
                pq, pk, pv, g_leaf, b_leaf, ctx, workspace=ws, **base, **L2N
            )
            out_split.backward(dout)
        names = [e.name for e in prof.events()]
        info["profiler_copy_ops"] = {
            n: names.count(n)
            for n in ("aten::copy_", "aten::contiguous", "aten::clone", "aten::cat")
            if n in names
        }
    except Exception as exc:  # noqa: BLE001 - the profiler is evidence only
        info["profiler_copy_ops"] = (
            f"unavailable: {type(exc).__name__}: {str(exc)[:120]}"
        )
        out_split = kcp.cake_gdn_det_kcp(
            pq, pk, pv, g_leaf, b_leaf, ctx, workspace=ws, **base, **L2N
        )
        out_split.backward(dout)
    ga = qkv_leaf.grad
    res_split = {
        "out": out_split.detach(),
        "dq": ga[:, : H_qk * 128].reshape(-1, H_qk, 128),
        "dk": ga[:, H_qk * 128 : 2 * H_qk * 128].reshape(-1, H_qk, 128),
        "dv": ga[:, 2 * H_qk * 128 :].reshape(-1, HV, 128),
        "dg": g_leaf.grad,
        "dbeta": b_leaf.grad,
    }
    flags["l2n_packed_split_eq_contiguous"] = _eq(res_auto, res_split)
    flags["l2n_packed_split_aliased"] = (
        T_r == 0 or kcp.PACKED_SPLIT_STATS["aliased"] - split_before["aliased"] == 1
    )
    del qkv_leaf, g_leaf, b_leaf, out_split, ga, res_split

    # issue #8367 with the norm: plain split views, direct and checkpointed (see _run_rank)
    def gdn_call(*a):
        return kcp.cake_gdn_det_kcp(*a, ctx, workspace=ws, **base, **L2N)

    for ck, tag in ((False, "direct"), (True, "ckpt")):
        res_plain, hooks_ok = _plain_split_pass(
            buf, g, beta, dout, gdn_call, H_qk, HV, ckpt=ck
        )
        flags[f"l2n_plain_split_{tag}_eq_contiguous"] = _eq(res_auto, res_plain)
        flags[f"l2n_plain_split_{tag}_hooks_fire"] = hooks_ok
        del res_plain

    # CUDA-graph replay with the norm: the fourth forward + backward replay, bitwise the eager result
    try:
        kcp.graph_cache_clear()
        events = []
        for _ in range(4):
            out_g, saved_g = kcp.cake_gdn_det_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, graph="replay", **base, **L2N
            )
            grads_g = kcp.cake_gdn_det_train_kcp_backward(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                dout,
                saved=saved_g,
                workspace=ws,
                graph="replay",
                **base,
                **L2N,
            )
            g_res = {
                "out": out_g.clone(),
                **{
                    n: t.clone()
                    for n, t in zip(
                        ("dq", "dk", "dv", "dbeta", "dg"), grads_g, strict=True
                    )
                },
            }
            events.append(
                [
                    kcp.LAST_CALL["fwd"]["graph_event"],
                    kcp.LAST_CALL["bwd"]["graph_event"],
                ]
            )
        info["graph_events"] = events
        graph_ok = events[-1] == ["replay", "replay"] and _eq(res, g_res)
        torch.cuda.synchronize(device)
        kcp.graph_cache_clear()
    except Exception as exc:  # noqa: BLE001 - recorded; the gate decides
        info["graph_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        graph_ok = False
        kcp.graph_cache_clear()
    flags["l2n_graph_eq_eager"] = bool(graph_ok)

    # memory: nothing of the norm in the arena after a forward / a backward; retained after a forward = off + r
    wsm = kcp.CakeGDNDetKcpWorkspace()
    bad = {*kcp.QK_TAPES, kcp.QK_RSTD}
    out_m, saved_m = kcp.cake_gdn_det_train_kcp_forward(
        q, k, v, g, beta, ctx, workspace=wsm, **base, **L2N
    )
    flags["l2n_absent_after_forward"] = not (set(wsm._tapes) & bad)
    kcp.cake_gdn_det_train_kcp_backward(
        q, k, v, g, beta, ctx, dout, saved=saved_m, workspace=wsm, **base, **L2N
    )
    flags["l2n_absent_after_backward"] = not (set(wsm._tapes) & bad)
    del out_m, saved_m, wsm

    def retained(**opts) -> tuple[int, int, int]:
        kcp.cake_gdn_det_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=None, **base, **opts
        )  # warm-up of the form
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        before = _alloc_stats(device)
        live = torch.cuda.memory_allocated(device)
        out_r, saved_r = kcp.cake_gdn_det_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=None, **base, **opts
        )
        torch.cuda.synchronize(device)
        after = _alloc_stats(device)
        peak = int(torch.cuda.max_memory_allocated(device) - live)
        kept = int(
            after["requested"]
            - before["requested"]
            - out_r.numel() * out_r.element_size()
        )
        nbytes = int(saved_r.bytes)
        del out_r, saved_r
        return kept, nbytes, peak

    kept_off, saved_off_b, peak_off = retained()
    kept_on, saved_on_b, peak_on = retained(**L2N)
    rstd_bytes = 8 * T_r * H_qk
    info["memory"] = {
        "retained_off": kept_off,
        "retained_on": kept_on,
        "saved_off": saved_off_b,
        "saved_on": saved_on_b,
        "fwd_peak_off": peak_off,
        "fwd_peak_on": peak_on,
        "qn_kn_bytes": 2 * T_r * H_qk * 256,
    }
    flags["l2n_post_forward_eq_off_plus_rstd"] = kept_on - kept_off == rstd_bytes
    q_on = int(
        gb.cake_gdn_det_train_kcp_saved_bytes(
            ctx, H_qk=H_qk, HV=HV, use_qk_l2norm_in_kernel=True
        )
    )
    q_off = int(gb.cake_gdn_det_train_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV))
    info["saved_query"] = {"on": q_on, "off": q_off}
    flags["l2n_saved_bytes_on_minus_off"] = (
        saved_on_b - saved_off_b == rstd_bytes == q_on - q_off
    )
    # save_mode="full" keeps the forward's qn / kn as well: retained and saved grow by r and the two rows
    kept_full_off, saved_full_off_b, _p = retained(save_mode="full")
    kept_full_on, saved_full_on_b, _p = retained(save_mode="full", **L2N)
    qk_bytes = 2 * T_r * H_qk * 256
    qf_on = int(
        gb.cake_gdn_det_train_kcp_saved_bytes(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", use_qk_l2norm_in_kernel=True
        )
    )
    qf_off = int(
        gb.cake_gdn_det_train_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV, save_mode="full")
    )
    info["memory"].update(
        retained_full_off=kept_full_off,
        retained_full_on=kept_full_on,
        saved_full_off=saved_full_off_b,
        saved_full_on=saved_full_on_b,
    )
    flags["l2n_full_post_forward_eq_off_plus_rstd_rows"] = (
        kept_full_on - kept_full_off == rstd_bytes + qk_bytes
    )
    flags["l2n_full_saved_bytes_on_minus_off"] = (
        saved_full_on_b - saved_full_off_b == rstd_bytes + qk_bytes == qf_on - qf_off
    )

    # gather the flags; the verdict is WORLD-wide
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        flag_list, info_list = [flags], [info]
    report: dict = {
        "row": dataclasses.asdict(row),
        "rank": rank,
        "world": W,
        "backend": backend,
        "mode": "l2n",
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pass": True,
        "flags": {},
        "gated": [],
        "reported": [],
        "info_by_rank": info_list,
        "seconds": None,
    }
    # a rank without tokens has no rows to compare: the per-row flags exist on the ranks with tokens only
    keys = sorted({k_ for f in flag_list for k_ in f})
    for key in keys:
        ok = all(f[key] for f in flag_list if key in f)
        report["flags"][key] = ok
        report["gated"].append(key)
        report["pass"] &= ok
    report["seconds"] = round(time.time() - t_start, 2)
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"], group=group)
        report["pass"] = all(verdict)
        dist.barrier(group=group)
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + "-l2n.json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_l2n(
    rank: int, W: int, row: Row, port: int, result_dir: str, backend: str
) -> None:
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    try:
        if W > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
            dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
        group = dist.group.WORLD if W > 1 else None
        report = _run_rank_l2n(rank, W, row, group, device, result_dir, backend)
        if not report["pass"]:
            raise AssertionError(
                f"rank {rank}: {json.dumps(report, default=str)[:4000]}"
            )
    finally:
        kcp.graph_cache_clear()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_l2n_rows(row: Row, slurm) -> None:
    """#7817 rows: the in-kernel Q / K L2 norm on the row's partition (see the block comment above).  Skips with a
    clear reason while the registry does not ship the three records (kernel backend)."""
    if _test_mode() != "full":
        pytest.skip("KCP_TEST_MODE=forward: the forward-only rows run instead")
    backend = _backend_name()
    _skip_unless_l2n(backend)
    if slurm is not None:
        _slurm_row(row, slurm, backend, _run_rank_l2n)
        return
    _skip_unless_runnable(row.W, backend)
    with tempfile.TemporaryDirectory(prefix="cake_gdn_kcp_l2n_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry_l2n,
            args=(row.W, row, _free_port(), result_dir, backend),
            nprocs=row.W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, _row_id(row) + "-l2n.json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


@pytest.mark.parametrize("row", LAYER_ROWS, ids=_row_id)
def test_l2n_layer_checkpoint_rows(row: Row, slurm) -> None:
    """#7817 whole-layer rows: KCP convolution + ``cake_gdn_det_kcp(use_qk_l2norm_in_kernel=True)`` on the RAW
    convolution output views in one checkpointed region (the B3 boundary), with and without a recompute handle:
    out / dx / dw / dg / dbeta bitwise the direct pass; L8 is the ``C = 10240`` (16, 48) row."""
    if slurm is None:
        pytest.skip("the layer rows run under Slurm")
    conv = pytest.importorskip("cake_pplx.cake_causal_conv1d_det_train")
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    backend = _backend_name()
    _skip_unless_l2n(backend)
    import functools

    _slurm_row(row, slurm, backend, functools.partial(_run_rank_layer, l2n=True))
