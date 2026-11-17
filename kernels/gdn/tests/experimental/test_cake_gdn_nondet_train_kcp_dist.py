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

Multi-GPU rows of the non-deterministic context-parallel (KCP) packed GDN training package
(``cake_pplx._backends.cake_gdn_nondet_train_kcp``, ``cake_kcp``; the mode is in the package name):
``W`` ranks over NCCL, each holding the fragments of one partition of a packed batch.

Accuracy is judged by ``tests/cake_naive.py``: every row runs :data:`DRAWS` independent draws of its operands inside
one process group.  On every draw each arm's outputs (``out``, ``dq``, ``dk``, ``dv`` as stored BF16, ``dg``,
``dbeta`` FP32) are reassembled into the global pack on rank 0 (:class:`_GroupJudge`), held to their contract and
judged as ``"<arm>.<output>"`` against the exact FP64 per-document recurrence of the same operands
(``reference_pack``) beside the naive BF16 tier of that recurrence (``naive_pack``).  The package has no bitwise
contract, so every arm is judged on its own against the FP64 reference; the repeated runs of one draw are judged
under one name (the judge keeps each output's largest error of the draw).  Per row:

* the arms: ``N_REPEAT`` forward + backward runs, saved / boundary replay, light / full, the full form's split twin
  (test knob ``fused_o_tape=0``; each full forward's launch log names the walk it ran), autograd / explicit,
  checkpoint / direct with its kept-tape, ``"auto"`` and recompute-handle forms, ``save_transfer``, ring / a2a,
  strided / contiguous, the packed-QKV split and plain split views, CUDA-graph replay (kernel backend) and the
  checkpointed CUDA-graph replay steps of the recompute's tape hand-off (the shipped ``"auto"``, ``"auto"`` with the
  size rule between the ranks, ``True``), the package root's entries; ``.backward()`` twice: half the accumulated
  gradient;
* ``save_mode="full"`` with ``save_transfer=False`` (a2a and, where the partition allows it, ring) runs with every
  package allocation filled with NaN bytes, and its saved set carries ``M_seg`` exactly at ``P > 1`` (the full
  backward's merges read that plane; it runs no transfer walk); a row whose ranks all plan ``P = 1`` reads no
  ``M_seg`` at its plan, so the same forms run again at the forced ``P = 2`` (``segments=2``, where every rank's
  fragments take it) with the light form at ``P = 2``, all under NaN-filled allocations; where every rank's longest
  fragment has at least four chunks the same forms run at the forced ``P = 4`` (the split plan beyond two segments),
  both launch records naming the forced ``P``;
* semantics (G4): the zero-gradient leak stays exact (``dout`` zero outside one document -> exactly zero gradients on
  the others); with the other documents' operands replaced, the selected document's outputs are judged on its tokens;
* contracts (G10): retained bytes after a checkpointed forward, the exchange bytes against
  ``cake_gdn_nondet_train_kcp_comm_bytes``, CUDA-graph capture (replay events; asserted for the kernel backend), no
  synchronizing call in context preparation or the ``"auto"`` checkpoint form, the side-stream ratio (reported);
* the collectives of one backward (counted by ``torch.distributed`` name): ``b1`` and ``b2`` as
  ``all_to_all_single`` where any fragment crosses, and no ``all_reduce`` / ``reduce_scatter`` of parameters;
* the host of every rank (``socket.gethostname``); under Slurm the group spans the hosts the step's task layout gives
  its ranks (the cross-host rows of a multi-node step).

The exact checks run on every draw where they cost only the launches the draw makes anyway; the memory, byte,
collective and timing records run on the first draw.  A row without tokens is held to the shapes of its outputs.
No check compares two launches of this package bit for bit (it has no bitwise contract, not even run to run).

The ``prod`` stage holds one production-size row (``PROD-C2``, the production heads ``(16, 48)``): every rank's
fragment is longer than ``MAX_CHAIN_STEPS`` chunks, so the shipped plan rule segments it (``P >= P_chain >= 2``) on
any SM count; every draw asserts that plan and the default exchange from the launch records (``production_route``).

``test_poisoned_workspace_pairs``: one ``CakeGDNNondetKcpWorkspace`` reused across two rows of different token layouts
on one group, every pool and tape of the workspace filled with NaN in between, on every draw; both rows judged, each
also in the ``save_mode="full"`` / ``save_transfer=False`` form (and its ring form where eligible), at the plan's
``P`` and at the forced ``P = 2`` where every rank plans ``P = 1`` and its fragments take it, under NaN-filled
allocations.  The rows that also need the causal_conv1d family -- the convolution halo row C7 and the whole-layer
checkpoint rows (``test_layer_checkpoint_rows``, ``test_l2n_layer_checkpoint_rows``) -- live in
``tests/cross_family/gdn/tests/experimental/test_cake_gdn_nondet_train_kcp_dist.py``.

Packed QKV (#7817): under ``layout="tma"`` the backward of strided packed views writes one packed dQKV row, and
``cake_gdn_nondet_train_kcp_packed_qkv_views`` hands it to a packed leaf without a concatenation; plain ``split``
views of the leaf, direct and checkpointed, get three contiguous gradients and keep torch's split backward and their
hooks and ``retain_grad`` (issue #8367).  ``test_l2n_rows``: the in-kernel Q / K L2 norm
(``use_qk_l2norm_in_kernel=True``) on raw rows, every form judged against the FP64 reference of the norm, the norm's
rows judged against the FP64 norm, and ``save_mode="light_akk"`` (see the block comment before ``L2N_SKIP``).

Two launch forms: ``torch.multiprocessing.spawn`` (``pytest`` on a node with >= ``W`` GPUs) and one process per Slurm
task (``SLURM_PROCID`` / ``SLURM_NTASKS``: the WORLD group is initialised once per process, every row runs on a
sub-group of its ``W`` lowest ranks; the other ranks skip the row).  Rank 0 hands the verdicts to the test, which
records one ``TIER`` line per judged output.  Environment: ``KCP_TEST_BACKEND`` (``kernels`` | ``fake``; default
``kernels`` when the package's ``build.json`` lists the programs, else ``fake``), ``KCP_TEST_ROWS`` (row or pair names,
comma list; ``all``; or a stage: ``int1`` CP <= 4, ``int3`` CP8, ``int4`` CP16, ``prod`` the production-size row),
``KCP_TEST_RESULT_DIR`` (one JSON per row), ``KCP_TEST_MODE`` (``full`` | ``forward``), ``KCP_TEST_REPEATS`` (raises
``N_REPEAT`` above 3), ``KCP_TEST_PORT_DIR`` (the Slurm rendezvous port file), ``KCP_TEST_DEBUG_RETAINED``.  Operands
come from the seeded
CPU generator (bitwise identical on every rank and host; draw ``j`` takes the seed ``draw_seed(SEED, j)``).  The suite
imports this package through ``tests/cake/test_helpers/cake_package.py`` and nothing else from cake_pplx (the
facade's tests live in ``test_cake_gdn_nondet_train_kcp_api.py``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import gc
import json
import os
import re
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
from tests.experimental.test_helpers import (
    cake_gdn_det_train_kcp_inputs as kcp_inputs,
)
from tests.experimental.test_helpers import (
    cake_gdn_det_train_kcp_reference as kcp_reference,
)
from tests.experimental.test_helpers.kcp_collective_shims import (
    CollectiveCounter,
    RecomputePhase,
)

PACKAGE = "cake_pplx._backends.cake_gdn_nondet_train_kcp"  # the in-tree path (report metadata)


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (see ``test_cake_gdn_nondet_train_kcp.py``), so this module runs with at most 8
    threads and oneDNN off, as that suite does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _mod(sub: str | None = None):
    """The package (or one of its modules) through the test-helper registry: a stale entry fails here, never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_nondet_train_kcp", sub)


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
SCALE = kcp.SCALE
C_CONV = 10240
ALLOC_GRANULE = 512  # the caching allocator rounds every block to 512 B
SIDE_STREAM_RATIO_MAX = 1.10  # #7038; the per-row ratio is recorded, never a gate (the module-timing protocol decides)
SIDE_STREAM_ITERS = 7
F64 = torch.float64
# the seed of every row's draw 0 (the generator's own seed); draw j takes draw_seed(SEED, j)
SEED = kcp_inputs.SEED
# (a) forward + backward runs per draw, every one judged under one name (the judge keeps each output's largest error
# of the draw; KCP_TEST_REPEATS may raise the count)
N_REPEAT = max(3, int(os.environ.get("KCP_TEST_REPEATS") or 3))
# owner chains (the exchange families of the host tables) of the long-chain rows: longer than this on every owning rank
OWNER_FAMILIES = ("fwd", "adj_b1", "adj_c1")
LONG_CHAIN_MIN = 3
# (b) collectives that would reduce parameters across the group: never issued by the route
PARAM_REDUCTIONS = ("all_reduce", "reduce_scatter_tensor")
# (d) the identity tile is a constant written once when the pool is sized (not scratch): it is not poisoned
POISON_EXEMPT_POOLS = ("eye",)
STAGES = ("int1", "int3", "int4", "prod")


def _reference():
    """The FP64 reference and its naive BF16 tier (the in-tree test helper, read only)."""
    return kcp_reference


def _inputs():
    """The seeded operand generator (``make_inputs(cu, H_qk, HV, *, regime, seed, token_range)``; the same generator
    as the reference helper's, bitwise)."""
    return kcp_inputs


# ----------------------------------------------------------------------------------------------------------------
# the judge over the group: rank 0 reassembles every arm's outputs and judges them against the naive BF16 tier


def _shapes(H_qk: int, HV: int) -> dict[str, tuple[int, ...]]:
    """The per-token shape of every output (the rows of the global pack and of a rank's slice), with the in-kernel
    norm's rows (:data:`L2N_ROWS`: ``q_hat`` / ``k_hat`` and their FP32 ``rstd``)."""
    return {
        "out": (HV, 128),
        "dq": (H_qk, 128),
        "dk": (H_qk, 128),
        "dv": (HV, 128),
        "dg": (HV,),
        "dbeta": (HV,),
        "qn": (H_qk, 128),
        "kn": (H_qk, 128),
        "rq": (H_qk,),
        "rk": (H_qk,),
    }


def _flag(flags: dict, key: str, ok) -> None:
    """``flags[key]``: ``ok`` held on every draw so far (an exact check repeated on every draw)."""
    flags[key] = bool(flags.get(key, True) and ok)


def _keep(info: dict, key: str, draw: int, record, ok: bool = True) -> None:
    """``info[key]``: the first draw's record; the first draw whose check failed adds its own record as
    ``info[key + "_failed"]``."""
    if draw == 0:
        info[key] = record
    if not ok:
        info.setdefault(f"{key}_failed", {"draw": draw, "record": record})


class _GroupJudge:
    """The accuracy of one row (or poison pair) over the group, judged against the naive BF16 tier
    (:class:`tests.cake_naive.TierJudge` on rank 0).

    :meth:`begin` opens a draw of a row: rank 0 computes the FP64 oracle and the naive tier of the global pack
    (``oracle_fn``).  :meth:`add` is a collective (every rank calls it in the same order): each rank puts its local
    slice of an arm's output (its tokens in fragment order) at their global tokens of a zero pack and the packs are
    summed onto rank 0 -- the ranks' tokens are disjoint, so the sum is the reassembled tensor -- where it is held to
    its contract and judged as ``"<arm>.<output>"``; a second launch of an arm on one draw keeps its larger error.
    :meth:`finish` returns rank 0's verdicts and every failure of the group: the ranks' shapes and the contracts,
    then the judge's."""

    def __init__(self, rank: int, W: int, group, device: torch.device) -> None:
        self.rank, self.W, self.group, self.device = rank, W, group, device
        self.judge = TierJudge() if rank == 0 else None
        self.failures: list[str] = []
        self.draw = 0
        self.total = 0
        self.l2g = torch.zeros(0, dtype=torch.int64)
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.oracle: dict[str, torch.Tensor] = {}
        self.naive: dict[str, torch.Tensor] = {}

    def begin(
        self, draw: int, part, cu: list[int], heads: tuple[int, int], oracle_fn
    ) -> None:
        """Draw ``draw`` of the row ``(part, cu)``; ``oracle_fn()`` -> ``(oracle, naive)`` of the global pack (dicts
        of CPU tensors without the batch axis), called on rank 0 only (not for a row without tokens)."""
        self.draw = draw
        self.total = cu[-1]
        self.l2g = _local_to_global(part, cu, self.rank).to(self.device)
        self.shapes = _shapes(*heads)
        self.oracle, self.naive = {}, {}
        if self.rank == 0 and self.total > 0:
            oracle, naive = oracle_fn()
            self.oracle = {n: t.to(self.device) for n, t in oracle.items()}
            self.naive = {n: t.to(self.device) for n, t in naive.items()}

    def add(
        self, arm: str, res: dict, names=TENSORS, *, half: bool = False, rows=None
    ) -> None:
        """Every rank: ``res[n]`` (this rank's tokens in fragment order) of each output ``n`` of ``names``.  ``half``:
        half of an accumulated (two-backward) gradient, exact in every float format; ``rows=(a, b)``: rank 0 judges
        the global tokens ``[a, b)`` only.  A row without tokens checks the local shapes and judges nothing."""
        for n in names:
            name = f"{arm}.{n}"
            local = res[n].detach()
            if half:
                local = local * 0.5
            shape = (int(self.l2g.numel()), *self.shapes[n])
            if tuple(local.shape) != shape:
                self.failures.append(
                    f"draw {self.draw}: rank {self.rank}: {name}: shape {tuple(local.shape)}, expected {shape}"
                )
                local = torch.zeros(shape, dtype=local.dtype, device=self.device)
            if self.total == 0:
                continue
            pack = torch.zeros(
                (self.total, *self.shapes[n]), dtype=local.dtype, device=self.device
            )
            pack[self.l2g] = local.to(self.device)
            if self.W > 1:
                dist.reduce(pack, dst=0, op=dist.ReduceOp.SUM, group=self.group)
            if self.rank != 0:
                continue
            want, naive = self.oracle[n], self.naive[n]
            if rows is not None:
                pack, want, naive = (t[rows[0] : rows[1]] for t in (pack, want, naive))
            self.add_rank0(name, pack, want, naive)

    def add_rank0(
        self,
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        naive: torch.Tensor,
    ) -> None:
        """Rank 0 only: an output it holds whole (``actual`` against the oracle ``expected``; ``naive`` the tier)."""
        if (msg := contract(name, actual, expected)) is not None:
            self.failures.append(f"draw {self.draw}: {msg}")
            if actual.shape != expected.shape:
                return
        self.judge.add(name, self.draw, actual, expected, {"bf16": naive})

    def finish(self) -> tuple[dict, list[str]]:
        """Rank 0's verdicts (``{}`` on the other ranks) and every failure of the group."""
        every: list = [None] * self.W
        if self.W > 1:
            dist.all_gather_object(every, self.failures, group=self.group)
        else:
            every = [self.failures]
        failures = [m for part in every for m in part]
        if self.judge is None:
            return {}, failures
        return self.judge.verdicts(), failures + self.judge.failures()


# ----------------------------------------------------------------------------------------------------------------
# collectives by name (b), hosts (c)


class _NamedCounter(CollectiveCounter):
    """:class:`CollectiveCounter` that also counts the calls per phase by ``torch.distributed`` name
    (``record()["named"][phase][name]``)."""

    def __init__(self) -> None:
        super().__init__()
        self.named: dict[str, dict[str, int]] = {}

    def __enter__(self) -> "_NamedCounter":
        super().__enter__()
        for name in list(self._saved):
            inner = getattr(dist, name)

            def shim(*a, _inner=inner, _name=name, **kw):
                table = self.named.setdefault(self.phase, {})
                table[_name] = table.get(_name, 0) + 1
                return _inner(*a, **kw)

            setattr(dist, name, shim)
        return self

    def record(self) -> dict:
        return {
            **super().record(),
            "named": {p: dict(t) for p, t in self.named.items()},
        }


def _coll_tags(sequence) -> list[str]:
    """The collective entries of a call's issue sequence (``a2a:<tag>`` / ``ring:<tag>``), in order."""
    return [str(s) for s in sequence if str(s).startswith(("a2a:", "ring:"))]


def _no_param_reduction(*records: dict) -> bool:
    return all(
        not table.get(name, 0)
        for rec in records
        for table in rec.get("named", {}).values()
        for name in PARAM_REDUCTIONS
    )


def _expand_tasks_per_node(s: str) -> list[int]:
    """``SLURM_*TASKS_PER_NODE`` (``4(x2),3``) -> ``[4, 4, 3]`` (``[]`` when unparsable)."""
    out: list[int] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        m = re.match(r"^(\d+)(?:\(x(\d+)\))?$", item)
        if not m:
            return []
        out += [int(m.group(1))] * int(m.group(2) or 1)
    return out


def _expected_node_index(W: int) -> list[int] | None:
    """The step node of each rank ``0 .. W-1`` under the step's task layout (``block``, Slurm's default, or
    ``cyclic``); ``None`` outside Slurm or for another distribution."""
    if os.environ.get("SLURM_PROCID") is None:
        return None
    tpn = _expand_tasks_per_node(
        os.environ.get("SLURM_STEP_TASKS_PER_NODE")
        or os.environ.get("SLURM_TASKS_PER_NODE")
        or ""
    )
    if not tpn or sum(tpn) < W:
        return None
    mode = re.split(r"[:,]", os.environ.get("SLURM_DISTRIBUTION") or "block")[0]
    idx: list[int] = []
    if mode == "block":
        for node, n in enumerate(tpn):
            idx += [node] * n
    elif mode == "cyclic":
        left, node = list(tpn), 0
        while len(idx) < sum(tpn):
            if left[node] > 0:
                idx.append(node)
                left[node] -= 1
            node = (node + 1) % len(tpn)
    else:
        return None
    return idx[:W]


def _host_map(group, W: int) -> dict:
    """(c) ``socket.gethostname()`` of every rank of the group (one ``all_gather_object``).  Under Slurm the ranks
    share a host exactly when the step's task layout puts them on one node (a multi-node step with an unknown layout
    fails: a cross-host row must be shown to span its hosts); outside Slurm the group is one host."""
    host = socket.gethostname()
    hosts: list = [None] * W
    if W > 1:
        dist.all_gather_object(hosts, host, group=group)
    else:
        hosts = [host]
    nnodes = int(
        os.environ.get("SLURM_STEP_NUM_NODES") or os.environ.get("SLURM_NNODES") or 1
    )
    idx = _expected_node_index(W)
    rec: dict = {
        "hosts": hosts,
        "distinct_hosts": len(set(hosts)),
        "slurm_nnodes": nnodes,
        "expected_node_index": idx,
    }
    if idx is not None:
        ok = all(
            (idx[a] == idx[b]) == (hosts[a] == hosts[b])
            for a in range(W)
            for b in range(W)
        )
        rec["expected_distinct_hosts"] = len(set(idx))
        nodes = _expand_hostlist(
            os.environ.get("SLURM_STEP_NODELIST")
            or os.environ.get("SLURM_NODELIST")
            or ""
        )
        # reported: the node names of the step against the host names (naming may differ by domain suffix)
        rec["nodelist_names_match"] = (
            all(
                str(hosts[r]).split(".")[0] == nodes[idx[r]].split(".")[0]
                for r in range(W)
            )
            if nodes and max(idx) < len(nodes)
            else None
        )
    elif os.environ.get("SLURM_PROCID") is not None and nnodes > 1:
        ok = False
        rec["reason"] = (
            "multi-node step with an unknown task layout (SLURM_STEP_TASKS_PER_NODE / SLURM_DISTRIBUTION)"
        )
    else:
        ok = len(set(hosts)) == 1
    rec["cross_host"] = len(set(hosts)) > 1
    rec["ok"] = bool(ok)
    return rec


# ----------------------------------------------------------------------------------------------------------------
# rows (C1-C9 rows in the normal regime, A1-A4 rows in the adversarial regimes, the no-leak rows, the long-chain rows;
# the int4 stage = the CP16 rows: the C6 zigzag row, the A1 contiguous row at the (2, 6) head pair and the long-chain
# two-document row).  Every row is judged over DRAWS draws: its documents are at most a few thousand tokens (a token
# x value-head product of at most about 50k), each keeping the layout features the row exists for -- multi-chunk
# fragments with a partial last chunk, odd lengths, one-token and empty documents and ranks where the row has them,
# ring-eligible chains, several documents per rank, every rank holding tokens unless the row tests an empty rank.


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
    long_chains: bool = False  # long owner chains (checked on the CPU: test_long_chain_rows_have_long_owner_chains)
    production: bool = False  # the production route asserted per draw


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
    # five documents (odd lengths, several per rank); CP2: the 418-token document crosses the cut, CP4: it spans three
    # ranks (a ring-eligible chain) and the 300-token one two
    Row("C1-cp2", (100, 300, 77, 418, 129), (2, 6), 2, "contiguous", leak=True),
    Row("C1-cp4", (100, 300, 77, 418, 129), (2, 6), 4, "contiguous"),
    # one document over four ranks: fragments of 250 / 251 (contiguous) and 125 / 126 tokens (zigzag)
    Row("C2-C", (1001,), (4, 12), 4, "contiguous"),
    Row("C2-Z", (1001,), (4, 12), 4, "zigzag"),
    Row("C2-D", (1001,), (4, 12), 4, "zigzag_per_document"),
    # eight pieces per document: one chunk each at chunk-aligned cuts (512), one chunk + 1 token (520), the 5-token and
    # the 1-token documents with empty pieces
    Row("C3", (512, 5, 520, 1), (1, 3), 4, "zigzag_per_document", leak=True),
    # ragged: a one-token rank, a rank holding the end of one document and the start of the other (C4a); an empty
    # rank (C4b)
    Row("C4a", (400, 600), (8, 24), 4, "ragged:280,1,420,299"),
    Row("C4b", (400, 600), (8, 24), 4, "ragged:480,0,260,260"),
    Row("C5", EXPLICIT["c5"][1], (4, 4), 4, "explicit:c5"),
    Row("C9a", (1,), (2, 6), 2, "contiguous"),
    Row("C9b", (0,), (2, 6), 2, "contiguous"),
    Row("leak-C", (300, 500, 200), (2, 6), 2, "contiguous", leak=True),
    Row("leak-D", (300, 500, 200), (2, 6), 2, "zigzag_per_document", leak=True),
    # A3: document 0 in the A1 regime, document 1 normal; eight pieces of about 100 tokens each
    Row(
        "A3",
        (801, 769),
        (2, 6),
        4,
        "zigzag_per_document",
        regime="A3",
        stage="int3",
    ),
    # the (16, 48) head pair: 1041 x 48 value heads, 16 zigzag fragments of 65 / 66 tokens
    Row("C6", (1041,), (16, 48), 8, "zigzag", stage="int3", long_chains=True),
    Row("C8", (512, 416, 105), (16, 48), 8, "contiguous", stage="int3"),
    # A1 / A1-cp16: the adversarial regime's mild gates carry state across every rank of the owner chains
    Row(
        "A1",
        (1601,),
        (2, 6),
        8,
        "contiguous",
        regime="A1",
        stage="int3",
        long_chains=True,
    ),
    Row("A2", (2001,), (2, 6), 8, "contiguous", regime="A2", stage="int3"),
    # A4: the longest row (2^12 - 1 tokens, 512 per rank)
    Row("A4", (4095,), (2, 6), 8, "contiguous", regime="A4", stage="int3"),
    # long-chain rows: two documents, every one cut into 2 W pieces (owner chains of 2 W - 1 crossing fragments)
    Row(
        "LC-D8",
        (1301, 1041),
        (2, 6),
        8,
        "zigzag_per_document",
        stage="int3",
        long_chains=True,
    ),
    # CP16 at the (16, 48) head pair: 32 zigzag fragments of 65 / 66 tokens (one full chunk and a partial one each)
    Row("C6-cp16", (2081,), (16, 48), 16, "zigzag", stage="int4", long_chains=True),
    Row(
        "A1-cp16",
        (1601,),
        (2, 6),
        16,
        "contiguous",
        regime="A1",
        stage="int4",
        long_chains=True,
    ),
    # 32 pieces per document of 65 / 66 and 64 / 65 tokens: every piece holds at least one full chunk
    Row(
        "LC-D16",
        (2081, 2049),
        (2, 6),
        16,
        "zigzag_per_document",
        stage="int4",
        long_chains=True,
    ),
    # the production-size row: the production heads (16, 48), one document over two ranks, each rank's fragment 16385
    # tokens = 257 chunks (the last one a single token), one chunk past the chain cap (MAX_CHAIN_STEPS = 256 chunks):
    # the shipped plan rule segments every fragment (P >= P_chain = 2) on any SM count, the smallest layout that does
    Row(
        "PROD-C2",
        (32770,),
        (16, 48),
        2,
        "contiguous",
        stage="prod",
        production=True,
    ),
]


def _row_by_name(name: str) -> Row:
    for r in ROWS:
        if r.name == name:
            return r
    raise KeyError(name)


@dataclasses.dataclass(frozen=True)
class PoisonPair:
    """(d) two rows of one group size and different token layouts that share one workspace, the second after the
    first (a pool of the first is kept where it is large enough for the second and regrown elsewhere; the kept pools
    are recorded)."""

    name: str
    first: str
    second: str
    stage: str

    @property
    def rows(self) -> tuple[Row, Row]:
        return _row_by_name(self.first), _row_by_name(self.second)

    @property
    def W(self) -> int:
        return self.rows[0].W


POISON_PAIRS = [
    PoisonPair(
        "P2", "C1-cp2", "leak-D", "int1"
    ),  # contiguous 5 docs -> zigzag_per_document 3 docs
    PoisonPair(
        "P4z", "C2-C", "C2-Z", "int1"
    ),  # contiguous -> zigzag, one document (zigzag and zigzag_per_document coincide for one document)
    PoisonPair("P4r", "C4a", "C4b", "int1"),  # ragged -> ragged with an empty rank
    PoisonPair(
        "P8", "A2", "LC-D8", "int3"
    ),  # contiguous (A2 regime) -> zigzag_per_document
    PoisonPair(
        "P16", "A1-cp16", "LC-D16", "int4"
    ),  # contiguous (A1 regime) -> zigzag_per_document
]


def _selected_rows() -> list[Row]:
    sel = os.environ.get("KCP_TEST_ROWS", "int1")
    if sel == "all":
        return list(ROWS)
    if sel in STAGES:
        return [r for r in ROWS if r.stage == sel]
    names = {s.strip() for s in sel.split(",") if s.strip()}
    rows = [r for r in ROWS if r.name in names]
    missing = names - {r.name for r in rows} - {p.name for p in POISON_PAIRS}
    if missing:
        raise ValueError(f"KCP_TEST_ROWS names unknown rows {sorted(missing)}")
    return rows


def _selected_poison_pairs() -> list[PoisonPair]:
    sel = os.environ.get("KCP_TEST_ROWS", "int1")
    if sel == "all":
        return list(POISON_PAIRS)
    if sel in STAGES:
        return [p for p in POISON_PAIRS if p.stage == sel]
    names = {s.strip() for s in sel.split(",") if s.strip()}
    return [p for p in POISON_PAIRS if p.name in names or {p.first, p.second} <= names]


def _row_id(row: Row) -> str:
    return (
        f"{row.name}-w{row.W}-h{row.heads[0]}x{row.heads[1]}-{row.layout.split(':')[0]}"
    )


def _pair_id(pair: PoisonPair) -> str:
    return f"{pair.name}-{pair.first}+{pair.second}-w{pair.W}"


def _report_file(obj, kind: str) -> str:
    ident = _pair_id(obj) if isinstance(obj, PoisonPair) else _row_id(obj)
    return (
        ident
        + {
            "full": ".json",
            "fwd": "-fwd.json",
            "poison": "-poison.json",
            "l2n": "-l2n.json",
        }[kind]
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


def _partition_with(fn, row: Row):
    """The partition of the row by ``fn(cu, W, layout=... | fragments=...)`` (this package's
    ``cake_gdn_nondet_train_kcp_partition``) and its ``cu_seqlens``."""
    cu = _cu(row.lengths)
    if row.layout.startswith("ragged:"):
        t_r = [int(x) for x in row.layout.split(":")[1].split(",")]
        assert len(t_r) == row.W and sum(t_r) == cu[-1]
        return fn(cu, row.W, fragments=_ragged(cu, t_r)), cu
    if row.layout.startswith("explicit:"):
        make, lengths = EXPLICIT[row.layout.split(":")[1]]
        assert tuple(lengths) == tuple(row.lengths)
        return fn(cu, row.W, fragments=make()), cu
    return fn(cu, row.W, layout=row.layout), cu


def _partition(row: Row):
    return _partition_with(plan.cake_gdn_nondet_train_kcp_partition, row)


def _launched_P() -> tuple:
    """``P`` of the last forward's and the last backward's launch records."""
    return kcp.LAST_CALL["fwd"]["P"], kcp.LAST_CALL["bwd"]["P"]


def _production_route(ctx, HV: int, *records: dict) -> dict:
    """The production route of this rank (a ``production`` row): the context's plan rule is the shipped one, the
    longest fragment is past the chain cap (``P_chain >= 2``: more than ``MAX_CHAIN_STEPS`` chunks), the plan
    segments it (``P >= P_chain``), and every given launch record ran that ``P`` over the default exchange
    (``"a2a"``)."""
    seg = ctx.heads(HV).segments
    ok = (
        ctx.p_rule == plan.P_RULE_DEFAULT
        and seg.P_chain >= 2
        and seg.P >= seg.P_chain
        and all(r["P"] == seg.P and r["exchange"] == "a2a" for r in records)
    )
    return {
        "ok": bool(ok),
        "p_rule": ctx.p_rule,
        "sm_count": int(ctx.sm_count),
        "NC_max_frag": int(seg.NC_max_frag),
        "P_chain": int(seg.P_chain),
        "P": int(seg.P),
        "launched": [(r["P"], r["exchange"]) for r in records],
    }


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


def _draw_inputs(row: Row, cu: list[int], seed: int) -> dict:
    """The global pack of one draw (CPU; the generator's layout: a batch axis of 1 on the operands, ``cu_seqlens``)."""
    H_qk, HV = row.heads
    return _inputs().make_inputs(cu, H_qk, HV, regime=row.regime, seed=seed)


def _oracle_pack(row: Row, cu: list[int], device, seed: int) -> tuple[dict, dict]:
    """Rank 0, one draw: the FP64 reference of the global pack (``reference_pack``: the exact recurrence of every
    document from the zero state and its autograd) and its naive BF16 tier (``naive_pack``), ``out`` and the five
    gradients as CPU tensors without the batch axis."""
    ref_mod = _reference()
    glob = _draw_inputs(row, cu, seed)
    oracle, _ = ref_mod.reference_pack(glob, device, scale=SCALE)
    return oracle, ref_mod.naive_pack(glob, device, scale=SCALE)


def _oracle_out(row: Row, cu: list[int], device, seed: int) -> tuple[dict, dict]:
    """Rank 0, one draw: ``out`` of the global pack from the FP64 forward of every document (``gdn_exact_forward``)
    and its naive BF16 tier (``gdn_naive_forward``, rounded to BF16), CPU tensors without the batch axis."""
    ref_mod = _reference()
    glob = _draw_inputs(row, cu, seed)
    q, k, v, g, beta = (glob[n][0] for n in ("q", "k", "v", "g", "beta"))
    exact, naive = [], []
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        ops = [t[s0:s1].to(device) for t in (q, k, v, g, beta)]
        cu_d = [0, s1 - s0]
        exact.append(ref_mod.gdn_exact_forward(*ops, cu_d, scale=SCALE)[0].cpu())
        naive.append(
            ref_mod.gdn_naive_forward(*ops, cu_d, scale=SCALE)[0]
            .to(torch.bfloat16)
            .cpu()
        )
        del ops
    return {"out": torch.cat(exact, 0)}, {"out": torch.cat(naive, 0)}


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


def _plain_split_pass(buf, g, beta, dout, call, H_qk: int, HV: int, *, ckpt: bool):
    """Issue #8367: one autograd pass with q / k / v as plain ``split`` + ``view`` head views of a packed leaf holding
    ``buf`` (torch's own split backward concatenates the gradients).  Returns the results (``dq`` / ``dk`` / ``dv``
    cut from the leaf's gradient) and whether the tensor hooks and ``retain_grad`` set on the views and on the split
    outputs of the original forward fired once each with the gradient the leaf received in the same pass (exact:
    structural; vacuous at ``T_r = 0``)."""
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
            and torch.equal(views[f"part{i}"].grad, ga[:, a:b])
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
                and "cake_gdn_nondet_train_kcp" not in fn
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
    #1-#5, ``out`` and the carried states judged against FP64, no backward program)."""
    mode = os.environ.get("KCP_TEST_MODE", "full")
    if mode not in ("full", "forward"):
        raise ValueError(f"KCP_TEST_MODE must be 'full' or 'forward', got {mode!r}")
    return mode


def _new_report(obj, rank: int, W: int, backend: str, device, mode: str) -> dict:
    return {
        "row": dataclasses.asdict(obj),
        "pass": True,
        "failing": [],
        "mode": mode,
        "package": PACKAGE,
        "rank": rank,
        "world": W,
        "backend": backend,
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "nccl": ".".join(str(x) for x in torch.cuda.nccl.version())
        if W > 1 and torch.cuda.is_available()
        else None,
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "draws": DRAWS,
        "n_repeat": N_REPEAT,
        "flags": {},
        "gated": [],
        "reported": [],
        "verdicts": {},
        "judge_failures": [],
        "info_by_rank": None,
        "seconds": None,
    }


def _fold(
    report: dict,
    flag_list: list,
    *,
    backend: str = "kernels",
    kernel_only=(),
    info_only=(),
) -> None:
    """Fold the ranks' exact flags (all ranks must hold) into ``report["pass"]``."""
    keys: list[str] = []
    for fl in flag_list:
        for key in fl:
            if key not in keys:
                keys.append(key)
    for key in keys:
        ok = all(bool(fl.get(key, False)) for fl in flag_list)
        report["flags"][key] = ok
        if key in info_only or (key in kernel_only and backend != "kernels"):
            report["reported"].append(key)
        else:
            report["gated"].append(key)
            report["pass"] &= ok
            if not ok:
                report["failing"].append(key)


def _fold_judge(report: dict, verdicts: dict, failures: list[str]) -> None:
    """Fold the judge into ``report["pass"]``: rank 0's verdicts (one per judged output) and every failure of the group
    (the ranks' shapes, the contracts, the outputs not judged on every draw, the draws with the same oracle values).
    A missed naive tier is in the verdicts only (a ``FAIL`` ``TIER`` line); it does not fail the row.  The failing label
    stays ``naive_tier``: it marks a contract or judge failure, not a missed tier."""
    report["verdicts"] = verdicts
    report["judge_failures"] = failures
    if failures:
        report["pass"] = False
        report["failing"].append("naive_tier")


def _finish_report(report: dict, group, W: int, rank: int, result_dir, kind, obj):
    """The group-wide verdict and rank 0's JSON."""
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"], group=group)
        report["pass"] = all(verdict)
        dist.barrier(group=group)
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _report_file(obj, kind)), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _record_tiers(report: dict | None, record_property) -> None:
    """One ``naive_tier`` property per verdict of rank 0's report (the conftest prints each as a ``TIER`` line); the
    other ranks' reports carry none."""
    if report is None or record_property is None:
        return
    for verdict in (report.get("verdicts") or {}).values():
        record_property("naive_tier", verdict)


def _summary(report: dict) -> str:
    """The failing checks first (the judge's failures bounded), then the report without the per-rank records and the
    verdicts (the ``TIER`` lines show those)."""
    head = {
        "failing": report.get("failing"),
        "judge_failures": (report.get("judge_failures") or [])[:40],
    }
    rest = {
        k: v
        for k, v in report.items()
        if k not in ("info_by_rank", "verdicts", "failing", "judge_failures")
    }
    return json.dumps({**head, **rest}, default=str)[:6000]


def _run_rank_fwd(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """Forward-only body of one rank (INT1) over :data:`DRAWS` draws: every forward arm's ``out`` judged against the
    FP64 forward, and the crossing-in fragments' ``S_in`` against the FP64 cut states, each beside its naive tier;
    nothing of the backward runs."""
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    ws = kcp.CakeGDNNondetKcpWorkspace()
    base = {"backend": backend}
    t_start = time.time()
    ref_mod = _reference()
    gj = _GroupJudge(rank, W, group, device)
    OUT = ("out",)
    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_eligible
    # the carried states: the global cut of every crossing-in fragment's S_in (in_slot order); a rank without tokens has
    # no fragments (an empty tuple): it must still reach the collectives below
    cuts = [None] * int(ctx.n_in)
    frags = ctx.fragments
    if frags and isinstance(frags[0], tuple):
        frags = frags[rank]
    for f in frags:
        if f.length > 0 and f.off > 0:
            cuts[int(ctx.in_slot[f.slot])] = int(cu[f.doc] + f.off)
    info["s_in_cuts"] = cuts
    objs_cuts: list = [None] * W
    if W > 1:
        dist.all_gather_object(objs_cuts, cuts, group=group)
    else:
        objs_cuts = [cuts]
    all_cuts = sorted({c for cs in objs_cuts for c in cs if c is not None})
    for j in range(DRAWS if total > 0 else 1):
        s = draw_seed(SEED, j)
        gj.begin(j, part, cu, row.heads, lambda s=s: _oracle_out(row, cu, device, s))
        ops = _local_inputs(row, part, cu, rank, device, seed=s)
        q, k, v, g, beta = (ops[n] for n in ("q", "k", "v", "g", "beta"))

        def fwd(qq=q, kk=k, vv=v, gg=g, bb=beta, **opts):
            return kcp.cake_gdn_nondet_train_kcp_forward(
                qq,
                kk,
                vv,
                gg,
                bb,
                ctx,
                workspace=ws,
                **{**base, **opts},
            )

        out, saved = fwd()
        if row.production:
            route = _production_route(ctx, HV, kcp.LAST_CALL["fwd"])
            _flag(flags, "production_route", route["ok"])
            _keep(info, "production_route", j, route, route["ok"])
        if j == 0:
            info["fwd_record"] = {
                k_: v_
                for k_, v_ in kcp.LAST_CALL["fwd"].items()
                if k_ not in ("sequence", "pool_bytes", "fresh_bytes", "merges")
            }
            info["fwd_sequence"] = list(kcp.LAST_CALL["fwd"]["sequence"])
        # the N_REPEAT runs under one name (the judge keeps the largest error of the draw), then every other form
        gj.add("run", {"out": out}, OUT)
        for _ in range(1, N_REPEAT):
            gj.add("run", {"out": fwd()[0]}, OUT)
        gj.add("no_save_transfer", {"out": fwd(save_transfer=False)[0]}, OUT)
        gj.add("full", {"out": fwd(save_mode="full")[0]}, OUT)
        gj.add("replay", {"out": fwd(boundary_mode="replay")[0]}, OUT)
        if ring_eligible:
            gj.add("ring", {"out": fwd(exchange="ring", save_transfer=False)[0]}, OUT)
        _buf, qs, ks, vs = _packed_views(q, k, v)
        gj.add("strided", {"out": fwd(qs, ks, vs)[0]}, OUT)
        gj.add("copy_layout", {"out": fwd(qs, ks, vs, layout="copy")[0]}, OUT)
        del _buf, qs, ks, vs
        # S_in per crossing-in fragment (in_slot order), judged on rank 0 against the FP64 state carried out of the
        # tokens before its global cut (cut_states) beside the naive tier (naive_cut_states)
        s_in = (
            kcp.saved_states(saved)
            if saved is not None
            else out.new_zeros((0, HV, 128, 128), dtype=torch.float32)
        )
        objs_s: list = [None] * W
        if W > 1:
            dist.all_gather_object(objs_s, s_in.cpu(), group=group)
        else:
            objs_s = [s_in.cpu()]
        if rank == 0 and all_cuts:
            glob = _draw_inputs(row, cu, s)
            ops_g = [glob[n][0].to(device) for n in ("q", "k", "v", "g", "beta")]
            cu_t = torch.tensor(cu, dtype=torch.int64)
            states = ref_mod.cut_states(*ops_g, cu_t, all_cuts, scale=SCALE)
            naive_states = ref_mod.naive_cut_states(*ops_g, cu_t, all_cuts, scale=SCALE)
            for cs, st in zip(objs_cuts, objs_s, strict=True):
                for slot, c in enumerate(cs):
                    if c is None:
                        continue
                    cand = st[slot].to(device)
                    gj.add_rank0(
                        f"s_in.{c}",
                        cand,
                        states[c].reshape(cand.shape),
                        naive_states[c].reshape(cand.shape),
                    )
            del glob, ops_g, states, naive_states
        del ops, out, saved, s_in, objs_s
    host = _host_map(group, W)
    flags["host_map"] = host["ok"]
    verdicts, failures = gj.finish()
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        flag_list, info_list = [flags], [info]
    report = _new_report(row, rank, W, backend, device, "forward")
    report["hosts"] = host
    report["info_by_rank"] = info_list
    _fold(report, flag_list)
    _fold_judge(report, verdicts, failures)
    report["seconds"] = round(time.time() - t_start, 2)
    return _finish_report(report, group, W, rank, result_dir, "fwd", row)


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
        "kcp_fwdh_out_tape",
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


def _run_rank(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """The body of one rank over :data:`DRAWS` draws: every mode of the KCP forward / backward judged against the FP64
    reference with its naive tier (rank 0 judges the reassembled global pack), the exact contracts, the collectives,
    the hosts."""
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    ws = kcp.CakeGDNNondetKcpWorkspace()
    base = {"backend": backend}
    t_start = time.time()
    kernels = backend == "kernels"
    gj = _GroupJudge(rank, W, group, device)
    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    info["ring_eligible"] = ring_eligible
    # rows whose ranks all plan P = 1: the full / ring + full forms again at the forced P = 2 (segments override)
    info["p2_eligible"] = _p2_group_ok(ctx, HV, group, W, device)
    # rows whose every rank takes the segments override P = 4 (at least four chunks in its longest fragment)
    info["p4_eligible"] = _forced_group_ok(ctx, HV, 4, group, W, device)
    # the bytes a recompute handle holds: exactly this rank's S_in (the light, save_transfer=False byte query)
    s_in_query = int(
        kcp.cake_gdn_nondet_train_kcp_saved_bytes(
            ctx,
            H_qk=H_qk,
            HV=HV,
            save_mode="light",
            save_transfer=False,
            with_conv=False,
        )
    )
    KEPT = {"reuse_recompute_tapes": True}
    # both states are named (the plain checkpoint runs the default's)
    OFF = {"reuse_recompute_tapes": False}
    AUTO = {"reuse_recompute_tapes": "auto"}

    def draw(j: int, seed: int) -> None:
        """Draw ``j`` (operand seed ``seed``): every judged arm; the exact checks of the draw; the memory, byte,
        collective and timing records on the first draw."""
        first = j == 0
        gj.begin(j, part, cu, row.heads, lambda: _oracle_pack(row, cu, device, seed))
        ops = _local_inputs(row, part, cu, rank, device, seed=seed)
        q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
        T_r_here = int(q.shape[0])

        def saved_pass(qq=q, kk=k, vv=v, gg=g, bb=beta, dd=dout, **opts) -> dict:
            o = {**base, **opts}
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                qq, kk, vv, gg, bb, ctx, workspace=ws, **o
            )
            dq, dk, dv, dbeta, dg = kcp.cake_gdn_nondet_train_kcp_backward(
                qq,
                kk,
                vv,
                gg,
                bb,
                ctx,
                dd,
                saved=saved,
                workspace=ws,
                **o,
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

        def autograd_pass(
            ckpt: bool, *, twice: bool = False, workspace=ws, **opts
        ) -> dict:
            leaves = [
                x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
            ]
            o = {**base, **opts}

            def fn(*a):
                return kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=workspace, **o)

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

        res = saved_pass()
        if row.production:
            route = _production_route(
                ctx, HV, kcp.LAST_CALL["fwd"], kcp.LAST_CALL["bwd"]
            )
            _flag(flags, "production_route", route["ok"])
            _keep(info, "production_route", j, route, route["ok"])
        if first:
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
        # (a) the worst case over N_REPEAT forward + backward runs: every run judged under one name (the judge keeps
        # each output's largest error of the draw)
        gj.add("run", res)
        for _ in range(1, N_REPEAT):
            gj.add("run", saved_pass())
        gj.add("replay", saved_pass(boundary_mode="replay"))
        res_full = saved_pass(save_mode="full")
        full_fwd = dict(kcp.LAST_CALL["fwd"])
        gj.add("full", res_full)
        # the full forward writes its output and the saved h / v_new tapes in one walk per launch list
        # (kcp_fwdh_out_tape) where the registry ships it for this architecture; test knob fused_o_tape = 0 is the
        # split form (the tape walk, then #5): judged on its own, each forward's launch log and fwd_o evidence naming
        # the form it ran (a rank without tokens runs neither walk)
        res_full_split = saved_pass(save_mode="full", test_knobs={"fused_o_tape": 0})
        split_fwd = dict(kcp.LAST_CALL["fwd"])
        gj.add("full_split", res_full_split)
        full_arch = (
            None
            if backend == "fake"
            else _mod("cake_backend").device_arch(device.index)
        )
        fused_tape = T_r_here > 0 and kcp.fused_o_tape_shipped(full_arch)
        full_seq, split_seq = list(full_fwd["sequence"]), list(split_fwd["sequence"])
        if first:
            info["full_fused_tape"] = {
                "expected": bool(fused_tape),
                "fwd_o": full_fwd.get("fwd_o"),
                "split_fwd_o": split_fwd.get("fwd_o"),
            }
        want_fwd_o = "fused_tape" if fused_tape else "launched"
        full_ok = (
            full_fwd.get("fwd_o") == want_fwd_o
            and ("kcp_fwdh_out_tape" in full_seq) == fused_tape
            and ("fwd_o" in full_seq) == (not fused_tape)
        )
        split_ok = "kcp_fwdh_out_tape" not in split_seq
        _flag(
            flags,
            "full_forward_runs_the_tape_and_output_walk",
            (T_r_here == 0 or full_ok) and split_ok,
        )
        del res_full, res_full_split
        gj.add("autograd", autograd_pass(False))
        gj.add("ckpt", autograd_pass(True))
        gj.add("ckpt_full", autograd_pass(True, save_mode="full"))
        gj.add("ckpt_replay", autograd_pass(True, boundary_mode="replay"))
        # reuse_recompute_tapes: the checkpoint backward binds the forward tapes torch's recompute just wrote instead
        # of replaying #1 and the tape walk; the kept path is proven from the backward's launch inventory (a rank
        # without tokens has no tapes and runs the replay form); the registry is empty afterwards; the per-rank peak
        # with the switch is not above the switch-off peak (first draw)
        gj.add("ckpt_off", autograd_pass(True, **OFF))
        kept_stats0 = dict(kcp.KEPT_STATS)
        res_kept = autograd_pass(True, **KEPT)
        kept_rec = _kept_record(T_r_here)
        gj.add("kept", res_kept)
        del res_kept
        _flag(flags, "kept_path_taken", kept_rec["path_ok"])
        res_kept_nt = autograd_pass(True, save_transfer=False, **KEPT)
        kept_nt_rec = _kept_record(T_r_here)
        gj.add("kept_no_transfer", res_kept_nt)
        del res_kept_nt
        _flag(flags, "kept_no_transfer_path_taken", kept_nt_rec["path_ok"])
        # the direct form inside the stats window names the replay form: under the shipped default a direct backward
        # looks the registry up and counts a "miss" (by design), which would break the miss = 0 expectation
        gj.add("direct_no_transfer", autograd_pass(False, save_transfer=False, **OFF))
        # two backwards: half the accumulated gradient
        gj.add(
            "kept_twice_half",
            autograd_pass(True, twice=True, **KEPT),
            GRADS,
            half=True,
        )
        kept_delta = {k_: kcp.KEPT_STATS[k_] - kept_stats0[k_] for k_ in kept_stats0}
        n_kept_bwd = 4  # three kept passes, one of them with two backwards
        expect = (
            {"put": n_kept_bwd, "take": n_kept_bwd, "miss": 0}
            if T_r_here > 0
            else {"put": 0, "take": 0, "miss": n_kept_bwd}
        )
        _flag(
            flags,
            "kept_registry_empty",
            kcp.kept_registry_size() == 0
            and all(kept_delta[k_] == v_ for k_, v_ in expect.items()),
        )
        _keep(info, "kept", j, kept_rec, kept_rec["path_ok"])
        _keep(info, "kept_no_transfer", j, kept_nt_rec, kept_nt_rec["path_ok"])
        _keep(info, "kept_stats_delta", j, kept_delta, flags["kept_registry_empty"])

        # issue #7074 (2) + G10: the "auto" form (the shipped default) in a region the GDN node closes takes the kept
        # path on a rank with tokens (the frame proof "dead"; a rank without tokens replays as before), and its
        # checkpointed forward + backward make no synchronizing CUDA call
        auto_stats0 = dict(kcp.KEPT_STATS)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            res_auto_form = autograd_pass(True, **AUTO)
            no_sync = True
        except RuntimeError as exc:  # a synchronizing call in the "auto" pass
            info.setdefault("auto_no_sync_error", str(exc)[:200])
            no_sync = False
            res_auto_form = None
        finally:
            torch.cuda.set_sync_debug_mode("default")
        _flag(flags, "auto_no_sync", no_sync)
        if res_auto_form is None:
            res_auto_form = autograd_pass(True, **AUTO)
        auto_rec = _kept_record(T_r_here)
        auto_rec["hand_off_reason"] = kcp.LAST_CALL["fwd"].get("hand_off_reason")
        gj.add("auto_form", res_auto_form)
        del res_auto_form
        # a rank with tokens hands off on the frame proof ("dead"); a rank without tokens has no tapes and switches the
        # hand-off off before the "auto" decision, as every form does there (recorded "off", the replay form)
        _flag(
            flags,
            "auto_path_taken",
            auto_rec["path_ok"]
            and (
                auto_rec["hand_off_reason"] == ("dead" if T_r_here > 0 else "off")
                or not kcp.torch_release_verified()
            ),
        )
        auto_delta = {k_: kcp.KEPT_STATS[k_] - auto_stats0[k_] for k_ in auto_stats0}
        n_auto = 1 if (T_r_here > 0 and kcp.torch_release_verified()) else 0
        _flag(
            flags,
            "auto_registry_empty",
            kcp.kept_registry_size() == 0
            and all(
                auto_delta[k_] == v_
                for k_, v_ in {"put": n_auto, "take": n_auto, "miss": 0}.items()
            ),
        )
        _keep(info, "auto_form", j, auto_rec, flags["auto_path_taken"])
        _keep(info, "auto_stats_delta", j, auto_delta, flags["auto_registry_empty"])

        # issue #7818: a CakeGDNNondetKcpRecomputeHandle carries this rank's incoming states S_in from the checkpointed
        # forward to torch's recompute, which then issues no collective and waits on none (counted by phase through
        # shims on torch.distributed; the recompute phase is marked by the checkpoint's context_fn).  The original
        # forward and the backward exchange exactly as the plain checkpoint does; a second backward reuses the entry;
        # the handle holds exactly S_in (and retains it after the forward: first draw).  The plain checkpoint's
        # per-phase counts are recorded as the reproduction of the issue.
        def handle_pass(handle: bool, *, twice: bool = False, workspace=ws, **opts):
            leaves = [
                x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
            ]
            o = {**base, "save_transfer": False, **opts}
            hd = kcp.CakeGDNNondetKcpRecomputeHandle() if handle else None

            def fn(q_, k_, v_, g_, b_, h_):
                return kcp.cake_gdn_nondet_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctx,
                    workspace=workspace,
                    recompute_handle=h_,
                    **o,
                )

            from torch.utils.checkpoint import checkpoint

            with _NamedCounter() as cc:
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

        handle_info: dict[str, object] = {"s_in_query_bytes": s_in_query}
        res_plain, rec_plain = handle_pass(False)
        res_h, rec_h = handle_pass(True)
        handle_info["plain"], handle_info["handle"] = rec_plain, rec_h
        gj.add("plain_ckpt", res_plain)
        gj.add("handle", res_h)
        del res_plain, res_h
        _flag(
            flags,
            "handle_recompute_no_comm",
            rec_h["calls"].get("recompute", 0) == 0
            and rec_h["waits"].get("recompute", 0) == 0,
        )
        _flag(
            flags,
            "handle_orig_bwd_comm_unchanged",
            all(
                rec_h[t].get(ph, 0) == rec_plain[t].get(ph, 0)
                for t in ("calls", "waits")
                for ph in ("orig", "bwd")
            ),
        )
        _flag(flags, "handle_modes", rec_h["modes"] == ["capture", "replay"])
        _flag(flags, "handle_bytes_eq_query", rec_h["handle_bytes"] == s_in_query)
        # the ring exchange (where the partition allows it): its recompute skips the hops as well
        if ring_eligible:
            res_pr, rec_pr = handle_pass(False, exchange="ring")
            res_hr, rec_hr = handle_pass(True, exchange="ring")
            handle_info["ring"] = {"plain": rec_pr, "handle": rec_hr}
            gj.add("ring_plain_ckpt", res_pr)
            gj.add("ring_handle", res_hr)
            del res_pr, res_hr
            _flag(
                flags,
                "handle_ring_recompute_no_comm",
                rec_hr["calls"].get("recompute", 0) == 0
                and rec_hr["waits"].get("recompute", 0) == 0
                and rec_hr["modes"] == ["capture", "replay"],
            )
        res_h2, rec_h2 = handle_pass(True, twice=True)
        handle_info["twice"] = rec_h2
        gj.add("handle_twice_half", res_h2, GRADS, half=True)
        del res_h2
        _flag(
            flags,
            "handle_twice_counts",
            rec_h2["calls"].get("recompute", 0) == 0
            and rec_h2["modes"] == ["capture", "replay", "replay"],
        )
        handle_info["plain_recompute_calls"] = rec_plain["calls"].get("recompute", 0)
        handle_info["plain_recompute_waits"] = rec_plain["waits"].get("recompute", 0)
        if first:
            # retained after the checkpointed forward (per-call workspace, after a warm-up of the same form): exactly
            # the handle's S_in, where the plain checkpoint retains nothing (retained_zero below)
            handle_pass(True, workspace=None)
            torch.cuda.synchronize(device)
            gc.collect()
            leaves_h = [
                x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
            ]
            hd_r = kcp.CakeGDNNondetKcpRecomputeHandle()

            def fn_r(q_, k_, v_, g_, b_, h_):
                return kcp.cake_gdn_nondet_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctx,
                    recompute_handle=h_,
                    save_transfer=False,
                    **base,
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
        res_recompute = saved_pass(save_transfer=False)
        gj.add("no_save_transfer", res_recompute)
        # save_mode="full" without save_transfer (the ring's only form): at P > 1 the backward's merges read the
        # forward's M_seg^T plane, which the full backward does not rebuild, so the saved set carries it.  Run under
        # NaN-filled allocations: a read of a plane no program of the call wrote is non-finite and breaks the output's
        # contract.
        res_full_nt = _nan_allocations_pass(
            lambda: saved_pass(save_mode="full", save_transfer=False)
        )
        gj.add("full_no_save_transfer", res_full_nt)
        rule = _m_seg_rule(res_full_nt["saved"])
        _flag(flags, "full_no_save_transfer_saves_m_seg", rule["ok"])
        if first:
            info["full_no_save_transfer"] = rule
        del res_full_nt
        if ring_eligible:
            res_ring = saved_pass(exchange="ring", save_transfer=False)
            gj.add("ring", res_ring)
            res_ring_full = _nan_allocations_pass(
                lambda: saved_pass(
                    exchange="ring", save_transfer=False, save_mode="full"
                )
            )
            gj.add("ring_full", res_ring_full)
            rule = _m_seg_rule(res_ring_full["saved"])
            _flag(flags, "ring_full_saves_m_seg", rule["ok"])
            if first:
                info["ring_full"] = rule
            del res_ring, res_ring_full
        del res_recompute
        # rows whose ranks all plan P = 1: the same full / ring + full forms again at the forced P = 2
        if info["p2_eligible"]:
            _forced_arms(
                saved_pass, gj, info, flags, P=2, ring=ring_eligible, first=first
            )
        # rows whose every rank's longest fragment has at least four chunks: the same forms at the forced P = 4 (the
        # split plan beyond two segments: three segment merges per fragment), whatever P the rule plans
        if info["p4_eligible"]:
            _forced_arms(
                saved_pass, gj, info, flags, P=4, ring=ring_eligible, first=first
            )
        # strided packed-QKV views: zero-copy (layout="tma") and the copy adapter
        _buf, qs, ks, vs = _packed_views(q, k, v)
        res_strided = saved_pass(qs, ks, vs)
        # issue #7817: packed operands -> #8 writes dq / dk / dv as the same views of one packed dQKV row
        _flag(
            flags,
            "packed_grad_layout",
            T_r_here == 0 or kcp.LAST_CALL["bwd"]["grad_layout"] == "packed",
        )  # T_r = 0: nothing to pack
        gj.add("strided", res_strided)
        del res_strided
        res_copy = saved_pass(qs, ks, vs, layout="copy")
        # the copy adapter copies strided views; a one-token view is already contiguous, stays a packed view and keeps
        # the packed gradient row
        copy_noop = T_r_here > 0 and all(x.is_contiguous() for x in (qs, ks, vs))
        _flag(
            flags,
            "copy_grad_layout_separate",
            kcp.LAST_CALL["bwd"]["grad_layout"]
            == ("packed" if copy_noop else "separate"),
        )
        gj.add("copy_layout", res_copy)
        del res_copy
        # ... and the split hands that row to the packed leaf without a concatenation
        split_before = dict(kcp.PACKED_SPLIT_STATS)
        qkv_leaf = _buf.detach().clone().requires_grad_(True)
        g_leaf, b_leaf = (x.detach().clone().requires_grad_(True) for x in (g, beta))
        pq, pk, pv = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(
            qkv_leaf, H_qk=H_qk, HV=HV
        )
        out_split = kcp.cake_gdn_nondet_kcp(
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
        gj.add("packed_split", res_split)
        _flag(
            flags,
            "packed_split_aliased",
            int(ga.shape[0]) == 0
            or kcp.PACKED_SPLIT_STATS["aliased"] - split_before["aliased"] == 1,
        )
        del qkv_leaf, g_leaf, b_leaf, out_split, ga, res_split

        # issue #8367: plain split views of the packed leaf keep torch's autograd path (its split backward
        # concatenates), direct and checkpointed, with the hooks and retain_grad on the views firing
        def gdn_call(*a):
            return kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=ws, **base)

        for ck, tag in ((False, "direct"), (True, "ckpt")):
            res_plain, hooks_ok = _plain_split_pass(
                _buf, g, beta, dout, gdn_call, H_qk, HV, ckpt=ck
            )
            # three contiguous gradients (one packed row only for cake_gdn_nondet_train_kcp_packed_qkv_views's split)
            layout_ok = kcp.LAST_CALL["bwd"]["grad_layout"] == "separate"
            gj.add(f"plain_split_{tag}", res_plain)
            _flag(flags, f"plain_split_{tag}_hooks_fire", hooks_ok)
            _flag(flags, f"plain_split_{tag}_grad_layout_separate", layout_ok)
            del res_plain
        del _buf, qs, ks, vs
        # gradient accumulation: .backward() twice -> half the accumulated gradient
        gj.add(
            "backward_twice_half", autograd_pass(False, twice=True), GRADS, half=True
        )
        if first:
            _first_draw_contracts(
                part,
                ctx,
                group,
                rank,
                device,
                backend,
                ws,
                (q, k, v, g, beta, dout),
                (H_qk, HV),
                flags,
                info,
                kept_opts=KEPT,
                off_opts=OFF,
            )
        # G10: CUDA-graph capture of forward + backward (pointer-stable operands, caller-held workspace): the third
        # call of a key replays the captured graph.  The backward's key includes the saved tensors, which change once
        # (the eager forward's fresh tensors -> the captured forward's static tensors), so the backward reaches its
        # replay one call after the forward: four forward + backward calls are made and the per-call events are
        # recorded (forward: eager, capture, replay, replay; backward: eager, eager, capture, replay).  The replayed
        # result is judged on the kernel backend (with the capture; the torch emulation reports both).
        graph_info: dict[str, object] = {}
        g_res = None
        try:
            kcp.graph_cache_clear()
            events: list[list[str]] = []
            for _ in range(4):
                out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    workspace=ws,
                    graph="replay",
                    **base,
                )
                grads = kcp.cake_gdn_nondet_train_kcp_backward(
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
            )
            torch.cuda.synchronize(device)
            kcp.graph_cache_clear()
        except Exception as exc:  # noqa: BLE001 - recorded; the gate decides below
            graph_info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            graph_ok = False
            g_res = None
            kcp.graph_cache_clear()
        # every rank adds the same keys (a failed capture adds zeros: the judge fails instead of a missing key)
        g_cmp = (
            g_res
            if g_res is not None
            else {n: torch.zeros_like(res[n]) for n in TENSORS}
        )
        if kernels:
            gj.add("graph_replay", g_cmp)
        del g_res, g_cmp
        _keep(info, "graph", j, graph_info, graph_ok)
        _flag(flags, "graph_capture", graph_ok)
        # graph="replay" + torch.utils.checkpoint (the recompute's tape hand-off): five checkpointed steps per arm on
        # a caller-held workspace with the same leaves -- the shipped "auto" (this architecture's size rule), "auto"
        # with the size rule set to the group's largest token x value-head rows (where the ranks' rows differ: the
        # ranks below it replay the original forward's graph while the others capture the hand-off form, one group
        # mixing both forms) and True.  Per rank every step's recompute decision and backward form are the expected
        # ones (gated on every backend); the last step replays the forward and the backward graphs (gated with the
        # capture on the kernel backend); the last step's out and gradients are judged on the kernel backend.
        from torch.utils.checkpoint import checkpoint

        ckpt_graph: dict[str, object] = {}
        rows_by_rank = [part.rank_tokens(r) * HV for r in range(W)]
        rows_here = T_r_here * HV
        verified = kcp.torch_release_verified()
        rule_arch = kcp._graph_recompute_arch(kcp.get_backend(backend), device)
        rule_table = dict(kcp.GRAPH_RECOMPUTE_MIN_ROWS)
        rule_default = kcp.GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT
        split_rule = (
            max(rows_by_rank) if min(rows_by_rank) < max(rows_by_rank) else None
        )
        ckpt_graph["rows_by_rank"] = rows_by_rank
        ckpt_graph["arch"] = rule_arch
        forms_ok = capture_ok = True
        for name, reuse, rule in (
            ("auto", "auto", None),
            ("auto_split", "auto", split_rule),
            ("kept", True, None),
        ):
            if name == "auto_split" and rule is None:
                ckpt_graph[name] = "not run: every rank holds the same rows"
                continue
            g_res = None
            try:
                if rule is not None:
                    for a_ in kcp.GRAPH_RECOMPUTE_MIN_ROWS:
                        kcp.GRAPH_RECOMPUTE_MIN_ROWS[a_] = rule
                    kcp.GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT = rule
                eff = kcp.GRAPH_RECOMPUTE_MIN_ROWS.get(
                    rule_arch, kcp.GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT
                )
                if reuse == "auto" and rows_here < eff:
                    expect = (False, "small", "replay")
                elif T_r_here == 0:
                    expect = (False, "off", "replay")
                elif reuse is True:
                    expect = (True, "always", "kept")
                elif verified:
                    expect = (True, "dead", "kept")
                else:  # no frame proof on this torch: the replay form (its reason names the torch build)
                    expect = (False, None, "replay")
                kcp.graph_cache_clear()
                wsg = kcp.CakeGDNNondetKcpWorkspace()
                leaves = [
                    x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
                ]
                o = {**base, "graph": "replay", "reuse_recompute_tapes": reuse}

                def fn(*a, o=o, wsg=wsg):
                    return kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=wsg, **o)

                steps: list[list] = []
                for _ in range(5):
                    for lf in leaves:
                        lf.grad = None
                    out = checkpoint(fn, *leaves, use_reentrant=False)
                    out.backward(dout)
                    f_, b_ = kcp.LAST_CALL["fwd"], kcp.LAST_CALL["bwd"]
                    steps.append(
                        [
                            str(f_["graph_event"]),
                            str(b_["graph_event"]),
                            bool(f_.get("hand_off")),
                            f_.get("hand_off_reason"),
                            b_.get("tapes_source"),
                        ]
                    )
                g_res = {
                    "out": out.detach().clone(),
                    **{n: lf.grad.clone() for n, lf in zip(GRADS, leaves, strict=True)},
                }
                forms = all(
                    (st[2], st[4]) == (expect[0], expect[2])
                    and (expect[1] is None or st[3] == expect[1])
                    for st in steps
                )
                replayed = steps[-1][0] == steps[-1][1] == "replay"
                ckpt_graph[name] = {
                    "rule": eff,
                    "expect": list(expect),
                    "steps": steps,
                    "forms_ok": forms,
                }
                forms_ok &= forms
                capture_ok &= replayed
                torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001 - recorded; the gates decide below
                ckpt_graph[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
                forms_ok = capture_ok = False
                g_res = None
            finally:
                kcp.GRAPH_RECOMPUTE_MIN_ROWS.clear()
                kcp.GRAPH_RECOMPUTE_MIN_ROWS.update(rule_table)
                kcp.GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT = rule_default
                kcp.graph_cache_clear()
            g_cmp = (
                g_res
                if g_res is not None
                else {n: torch.zeros_like(res[n]) for n in TENSORS}
            )
            if kernels:
                gj.add(f"graph_ckpt_{name}", g_cmp)
            del g_res, g_cmp
        _keep(info, "graph_ckpt", j, ckpt_graph, forms_ok and capture_ok)
        _flag(flags, "graph_ckpt_forms", forms_ok)
        _flag(flags, "graph_ckpt_capture", capture_ok)
        if first:
            # #7038: the backward's time with a resident one-thread kernel on a side stream, over its time alone
            ratio = _side_stream_ratio(
                lambda: kcp.cake_gdn_nondet_train_kcp_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    dout,
                    saved=res["saved"],
                    workspace=ws,
                    **base,
                ),
                device,
            )
            info["side_stream"] = ratio
            flags["side_stream_ratio"] = ratio["ratio"] <= SIDE_STREAM_RATIO_MAX
        # no-leak rows: exactly zero gradients outside the selected document; with the other documents' operands
        # replaced (drawn from the seed + 1, never another draw's seed) the selected document's outputs judged on its
        # tokens
        if row.leak and total > 0:
            n_docs = len(cu) - 1
            doc = 1 if n_docs > 1 else 0
            mask = _doc_mask_local(part, cu, rank, doc, device)
            dout_sel = dout * mask.view(-1, 1, 1).to(dout.dtype)
            res_sel = saved_pass(dd=dout_sel)
            others = ~mask
            _flag(
                flags,
                "zero_grad_leak",
                all(
                    bool(
                        torch.equal(
                            res_sel[n][others], torch.zeros_like(res_sel[n][others])
                        )
                    )
                    for n in GRADS
                ),
            )
            del res_sel
            alt = _local_inputs(
                row, part, cu, rank, device, seed=seed, doc_override=(doc, seed + 1)
            )
            res_alt = saved_pass(
                alt["q"], alt["k"], alt["v"], alt["g"], alt["beta"], alt["dout"]
            )
            gj.add("independence", res_alt, rows=(cu[doc], cu[doc + 1]))
            _flag(
                flags,
                "independence_changed_others",
                bool(others.any())
                and any(
                    not torch.equal(res_alt[n][others], res[n][others])
                    for n in ("out",)
                )
                or not bool(others.any()),
            )
            del res_alt, alt
        if first:
            # (b) the collectives of one backward (saved set of run0), by torch.distributed name and by issue tag: b1,
            # b2 as all_to_all_single where any fragment crosses, no parameter reduction
            with _NamedCounter() as cc_nd:
                cc_nd.phase = "bwd"
                kcp.cake_gdn_nondet_train_kcp_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    dout,
                    saved=res["saved"],
                    workspace=ws,
                    **base,
                )
            coll_nd = {
                "named": cc_nd.record()["named"].get("bwd", {}),
                "tags": _coll_tags(kcp.LAST_CALL["bwd"]["sequence"]),
            }
            info["collectives"] = coll_nd
            flags["bwd_collectives_b1_b2"] = coll_nd["tags"] in (
                [],
                ["a2a:b1", "a2a:b2"],
            ) and set(coll_nd["named"]) <= {"all_to_all_single"}
            flags["bwd_collectives_issued"] = W == 1 or coll_nd["named"].get(
                "all_to_all_single", 0
            ) == len(coll_nd["tags"])
            flags["bwd_no_param_reduction"] = _no_param_reduction(
                {"named": {"bwd": coll_nd["named"]}}
            )
        # (f) the package root's entries on the same operands route to this package
        pkg = _mod()
        f_out, f_saved = pkg.cake_gdn_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=ws, **base
        )
        f_grads = pkg.cake_gdn_nondet_train_kcp_backward(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=f_saved,
            workspace=ws,
            **base,
        )
        gj.add(
            "root",
            {
                "out": f_out,
                **dict(zip(("dq", "dk", "dv", "dbeta", "dg"), f_grads, strict=True)),
            },
        )
        _flag(
            flags,
            "root_routes_nondet",
            type(f_saved).__name__ == "CakeGDNNondetKcpSaved",
        )
        if first:
            info["routes"] = {"root_saved": type(f_saved).__name__}
        del f_out, f_saved, f_grads

    for j in range(DRAWS if total > 0 else 1):
        draw(j, draw_seed(SEED, j))
    # (c) the hosts of the group
    host = _host_map(group, W)
    flags["host_map"] = host["ok"]
    verdicts, failures = gj.finish()
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        flag_list, info_list = [flags], [info]
    report = _new_report(row, rank, W, backend, device, "full")
    report["hosts"] = host
    report["info_by_rank"] = info_list
    # gates: exact flags and the judge on every backend; graph capture is a gate of the kernel backend (the torch
    # emulation reports it); the per-row side-stream ratio is recorded on every backend -- the #7038 gate of record
    # is the module-timing protocol (slowest-rank median over >= 3 groups on an exclusive node)
    _fold(
        report,
        flag_list,
        backend=backend,
        kernel_only={"graph_capture", "graph_ckpt_capture"},
        info_only={"side_stream_ratio"},
    )
    _fold_judge(report, verdicts, failures)
    report["seconds"] = round(time.time() - t_start, 2)
    return _finish_report(report, group, W, rank, result_dir, "full", row)


def _first_draw_contracts(
    part,
    ctx,
    group,
    rank: int,
    device: torch.device,
    backend: str,
    ws,
    operands: tuple,
    heads: tuple[int, int],
    flags: dict,
    info: dict,
    *,
    kept_opts: dict,
    off_opts: dict,
) -> None:
    """The first draw's memory and byte contracts of :func:`_run_rank`: no synchronizing call while a fresh context
    prepares, the bytes a checkpointed forward retains, the saved-set bytes per ``save_mode`` (recorded) and the
    exchange bytes against the public byte query."""
    q, k, v, g, beta, dout = operands
    H_qk, HV = heads
    base = {"backend": backend}
    # G10 (issue #7074): preparing a fresh context of this rank's layout -- the host plan, its table uploads and the
    # per-key device tables -- makes no synchronizing CUDA call (pinned asynchronous copies)
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        fresh_ctx = plan.cake_gdn_nondet_train_kcp_build_context(
            part, group, rank, device
        )
        fresh_ctx.prepare(H_qk, HV)
        flags["prepare_no_sync"] = True
    except RuntimeError as exc:  # "called a synchronizing CUDA operation"
        info["prepare_no_sync_error"] = str(exc)[:200]
        flags["prepare_no_sync"] = False
        fresh_ctx = None
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
        ("per_call", None, {**base, **off_opts}),
        ("caller_ws", ws, {**base, **off_opts}),
        ("per_call_kept", None, {**base, **kept_opts}),
        ("caller_ws_kept", ws, {**base, **kept_opts}),
    ):
        leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

        def fn(*a, _wsp=wsp, _o=form_opts):
            return kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=_wsp, **_o)

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
    # the measured bytes of the saved objects (recorded, not gated: the default without the norm stays "light")
    try:
        sb: dict[str, int] = {"T_r": int(q.shape[0])}
        for mode in ("light", "full"):
            sb[f"{mode}_query"] = int(
                kcp.cake_gdn_nondet_train_kcp_saved_bytes(
                    ctx, H_qk=H_qk, HV=HV, save_mode=mode
                )
            )
            _, s_obj = kcp.cake_gdn_nondet_train_kcp_forward(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                workspace=ws,
                save_mode=mode,
                **base,
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
        query = kcp.cake_gdn_nondet_train_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV
        )
        info["query_bytes"] = {
            c: query[c] for c in ("c1", "c2", "b1", "b2", "b1_design") if c in query
        }
        # save_transfer=False: "b1" = the plan's split tables (maps paired with the crossing-IN slots; what the
        # collective moves) must equal the b1_dm split sums; "b1_design" is the per-rank lower bound
        query_dm = kcp.cake_gdn_nondet_train_kcp_comm_bytes(
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
    except (ImportError, AttributeError, KeyError, TypeError) as exc:
        info["query_bytes"] = f"unavailable: {type(exc).__name__}: {exc}"
        flags["bytes_match_query"] = False


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
# (d) a poisoned workspace reused across changed token layouts


def _poison(ws) -> dict:
    """Fill every floating tensor the workspace holds (its pools and its tape arena) with NaN; the identity tile
    (``POISON_EXEMPT_POOLS``) and the ticket words (u32, zero by the kernels' contract) are not scratch."""
    names: list[str] = []
    nbytes = 0
    for kind, table in (("pool", ws._pools), ("tape", ws._tapes)):
        for name, t in table.items():
            if kind == "pool" and name in POISON_EXEMPT_POOLS:
                continue
            if not t.is_floating_point() or t.numel() == 0:
                continue
            t.fill_(float("nan"))
            names.append(f"{kind}:{name}")
            nbytes += t.numel() * t.element_size()
    torch.cuda.synchronize()
    return {"tensors": len(names), "bytes": nbytes, "names": sorted(names)}


def _nan_allocations_pass(fn):
    """``fn()`` with every tensor the package allocates through ``_alloc`` (``torch.empty``: buffers a program or a
    collective of the same call writes before any read) filled with NaN bytes (0xFF): a read of a buffer no program
    of the call wrote is non-finite, so the output breaks its contract (finite where the FP64 reference is).  The
    identity tile is copied after its allocation and the ticket words are zero-filled outside ``_alloc``."""
    real = kcp._alloc

    def poisoned(shape, dtype, device):
        t = real(shape, dtype, device)
        if t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    kcp._alloc = poisoned
    try:
        return fn()
    finally:
        kcp._alloc = real


def _m_seg_rule(saved) -> dict:
    """The ``M_seg`` record of a ``save_mode="full"`` saved set: present exactly at ``P > 1`` (the full backward runs no
    transfer walk, so its merges read the saved ``M_seg^T`` plane), of the plan's ``stage_m`` size."""
    if saved is None:
        return {"ok": False, "P": None, "m_seg_bytes": None}
    m = saved.M_seg
    ok = (m is not None) == (int(saved.P) > 1) and saved.save_mode == "full"
    return {
        "ok": bool(ok),
        "P": int(saved.P),
        "m_seg_bytes": None if m is None else int(m.numel() * m.element_size()),
    }


def _forced_group_ok(
    ctx, HV: int, P: int, group, W: int, device: torch.device, *, from_one=False
) -> bool:
    """Every rank's fragments take the segments override ``P`` (the longest fragment of each rank has at least ``P``
    chunks; a rank without tokens takes none) and, with ``from_one``, every rank plans ``P = 1``.  A group decision
    (MIN over the ranks): every rank runs the same calls."""
    try:
        ok = int(ctx.heads(HV, P=P).P) == P and (
            not from_one or int(ctx.heads(HV).P) == 1
        )
    except ValueError:
        ok = False
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    if W > 1:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def _p2_group_ok(ctx, HV: int, group, W: int, device: torch.device) -> bool:
    """Every rank plans ``P = 1`` and its fragments take the segments override ``P = 2`` (the longest fragment of
    each rank has at least two chunks).  Such a row reads no saved ``M_seg^T`` plane at its plan; the forced
    ``P = 2`` arms put the full form's read of that plane on it.  A row that plans ``P > 1`` on some rank already
    reads the plane in its planned forms, and forcing ``P = 2`` there lowers its plan, so it runs no forced form."""
    return _forced_group_ok(ctx, HV, 2, group, W, device, from_one=True)


def _forced_arms(
    saved_pass, gj, info, flags, *, P: int, ring: bool, first: bool, **extra
) -> None:
    """The forced ``P`` forms (``segments=P``) without ``save_transfer`` under NaN-filled allocations:
    ``save_mode="full"`` (its backward reads the saved ``M_seg^T`` plane) and the light form, and the same pair over
    the ring when the partition allows it.  Every arm judged (``gj``, ``p<P>_...``); on every draw ``M_seg`` held at
    ``P`` and the forward's and the backward's launch records name ``P``."""
    pairs = [(f"p{P}_full_no_save_transfer", f"p{P}_no_save_transfer", {})]
    if ring:
        pairs.append((f"p{P}_ring_full", f"p{P}_ring", {"exchange": "ring"}))
    for full_name, light_name, opts in pairs:
        o = {"segments": P, "save_transfer": False, **opts, **extra}
        res_full = _nan_allocations_pass(lambda o=o: saved_pass(save_mode="full", **o))
        full_P = _launched_P()
        res_light = _nan_allocations_pass(
            lambda o=o: saved_pass(save_mode="light", **o)
        )
        light_P = _launched_P()
        gj.add(light_name, res_light)
        gj.add(full_name, res_full)
        rule = _m_seg_rule(res_full["saved"])
        _flag(flags, f"{full_name}_saves_m_seg", rule["ok"] and rule["P"] == P)
        _flag(flags, f"{full_name}_launched_P", full_P == (P, P))
        _flag(flags, f"{light_name}_launched_P", light_P == (P, P))
        if first:
            info[full_name] = {
                **rule,
                "launched_P": full_P,
                "light_launched_P": light_P,
            }
        del res_full, res_light


def _run_rank_poison(
    rank: int,
    W: int,
    pair: PoisonPair,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """(d) one ``CakeGDNNondetKcpWorkspace`` for both rows of the pair, on every draw: the first row's forward +
    backward, every pool and tape it left NaN, the workspace sized for the second row (pools kept where large enough,
    the second row's tapes bound), every pool and tape NaN again, then the second row's forward + backward.  Every
    arm of both rows judged (``<first|second>_<row>[_<form>].<output>``)."""
    ws = kcp.CakeGDNNondetKcpWorkspace()
    t_start = time.time()
    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    gj = _GroupJudge(rank, W, group, device)
    rows = []
    for row in pair.rows:
        part, cu = _partition(row)
        ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
        ring_ok = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
        # the forced P = 2 forms where every rank plans P = 1 and its fragments take P = 2 (_p2_group_ok)
        p2 = _p2_group_ok(ctx, row.heads[1], group, W, device)
        rows.append((row, part, cu, ctx, ring_ok, p2))
    draws = DRAWS if max(r[2][-1] for r in rows) > 0 else 1
    for j in range(draws):
        seed = draw_seed(SEED, j)
        for i, (row, part, cu, ctx, ring_ok, p2) in enumerate(rows):
            H_qk, HV = row.heads
            gj.begin(
                j,
                part,
                cu,
                row.heads,
                lambda row=row, cu=cu: _oracle_pack(row, cu, device, seed),
            )
            ops = _local_inputs(row, part, cu, rank, device, seed=seed)
            q, k, v, g, beta, dout = (
                ops[n] for n in ("q", "k", "v", "g", "beta", "dout")
            )
            if i == 1:
                p_left = _poison(ws)
                held = {name: t.data_ptr() for name, t in ws._pools.items()}
                ws.ensure(ctx, H_qk, HV, device=device)
                for name in kcp.FWD_TAPES + kcp.BWD_TAPES:
                    ws.tape(name)
                p_bound = _poison(ws)
                if j == 0:
                    info["poison"] = {
                        "left_by_first": p_left,
                        "before_second": p_bound,
                        "pools_kept_from_first": sorted(
                            n
                            for n, t in ws._pools.items()
                            if held.get(n) == t.data_ptr()
                        ),
                    }
                _flag(flags, "poisoned", p_bound["bytes"] > 0)

            def one_pass(**opts) -> dict:
                out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    workspace=ws,
                    backend=backend,
                    **opts,
                )
                dq, dk, dv, dbeta, dg = kcp.cake_gdn_nondet_train_kcp_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    dout,
                    saved=saved,
                    workspace=ws,
                    backend=backend,
                    **opts,
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

            tag = f"{'second' if i else 'first'}_{row.name}"
            gj.add(tag, one_pass())
            # save_mode="full" without save_transfer on the same workspace under NaN-filled allocations (and its ring
            # form where the partition allows it): at P > 1 its backward reads the saved M_seg^T plane
            # (+ the forced P = 2 forms where every rank plans P = 1 and its fragments take P = 2: _p2_group_ok)
            forms = [("full_no_save_transfer", {})]
            if ring_ok:
                forms.append(("ring_full", {"exchange": "ring"}))
            if j == 0:
                info[f"{tag}_p2_eligible"] = p2
            if p2:
                forms.append(("p2_full_no_save_transfer", {"segments": 2}))
                if ring_ok:
                    forms.append(("p2_ring_full", {"exchange": "ring", "segments": 2}))
            for form, extra in forms:
                res = _nan_allocations_pass(
                    lambda extra=extra: one_pass(
                        save_mode="full", save_transfer=False, **extra
                    )
                )
                gj.add(f"{tag}_{form}", res)
                rule = _m_seg_rule(res["saved"])
                if j == 0:
                    info[f"{tag}_{form}"] = rule
                forced = "segments" not in extra or rule["P"] == int(extra["segments"])
                _flag(flags, f"{form}_saves_m_seg", rule["ok"] and forced)
                del res
            del ops
    host = _host_map(group, W)
    flags["host_map"] = host["ok"]
    verdicts, failures = gj.finish()
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        flag_list, info_list = [flags], [info]
    report = _new_report(pair, rank, W, backend, device, "poison")
    report["rows"] = [dataclasses.asdict(r) for r in pair.rows]
    report["hosts"] = host
    report["info_by_rank"] = info_list
    _fold(report, flag_list)
    _fold_judge(report, verdicts, failures)
    report["seconds"] = round(time.time() - t_start, 2)
    return _finish_report(report, group, W, rank, result_dir, "poison", pair)


_BODIES = {"full": _run_rank, "fwd": _run_rank_fwd, "poison": _run_rank_poison}


# ----------------------------------------------------------------------------------------------------------------
# launch forms


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_entry(
    rank: int, W: int, obj, port: int, result_dir: str, backend: str, kind: str
) -> None:
    """One spawned rank.  The verdict travels through rank 0's report file (the group-wide pass, the judge's verdicts
    and failures), which the test reads and records before it asserts; a rank that raises fails the start."""
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    try:
        if W > 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
            dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
        group = dist.group.WORLD if W > 1 else None
        _BODIES[kind](rank, W, obj, group, device, result_dir, backend)
    finally:
        kcp.graph_cache_clear()
        if dist.is_initialized():
            dist.destroy_process_group()


# _free_port releases the port it finds before rank 0's rendezvous store binds it, so another socket of the node can
# take it in between: rank 0 then fails in init_process_group with EADDRINUSE before the row runs.  Such a start is
# retried on a fresh port, at most this many times in all.
RENDEZVOUS_TRIES = 3


def _rendezvous_port_taken(err: BaseException) -> bool:
    """Whether a spawned start failed because rank 0's rendezvous port was taken (``EADDRINUSE`` from its store)."""
    return "EADDRINUSE" in str(err)


def _spawn_row(obj, backend: str, kind: str, record_property=None) -> None:
    """``obj`` (a row or a poison pair) on ``obj.W`` spawned ranks of this node; rank 0's report decides, its verdicts
    recorded first (``record_property``).  A start whose rendezvous port was taken is retried on a fresh port
    (:data:`RENDEZVOUS_TRIES`)."""
    with tempfile.TemporaryDirectory(prefix="cake_gdn_nondet_train_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        for attempt in range(RENDEZVOUS_TRIES):
            try:
                mp.start_processes(
                    _spawn_entry,
                    args=(obj.W, obj, _free_port(), result_dir, backend, kind),
                    nprocs=obj.W,
                    join=True,
                    start_method="spawn",
                )
                break
            except mp.ProcessRaisedException as err:
                if attempt + 1 == RENDEZVOUS_TRIES or not _rendezvous_port_taken(err):
                    raise
        with open(os.path.join(result_dir, _report_file(obj, kind))) as fh:
            report = json.load(fh)
    _record_tiers(report, record_property)
    assert report["pass"], _summary(report)


def _expand_hostlist(s: str) -> list[str]:
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
    # a name of its own: the deterministic suite's port file of the same step is never read
    path = os.path.join(d, f"kcp_train_test_port.{job}.{step}")
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
            # must fail the step instead of holding its block; the longest legitimate collective wait (the FP64
            # oracle of the largest row on the rank holding most of it) stays well below it
            dist.init_process_group(
                "nccl",
                rank=self.rank,
                world_size=self.world,
                device_id=self.device,
                timeout=_dt.timedelta(seconds=WORLD_TIMEOUT_S),
            )
        print(
            f"[kcp_train_dist] init timeout {WORLD_TIMEOUT_S} s; "
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
    # no CUDA device (a CPU-only Slurm step): the spawn path's skips apply instead of a WORLD init that needs a device
    if os.environ.get("SLURM_PROCID") is None or not torch.cuda.is_available():
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


def _slurm_row(row, slurm, backend: str, body, record_property=None) -> None:
    """One row (or poison pair) under Slurm: sub-group of the row's ranks; the verdict is WORLD-wide, so every rank
    (member or not) passes or fails the same row and ``-x`` and the session teardown keep the collectives aligned.
    Rank 0 records the verdicts of its judge (``record_property``) before the row's verdict is asserted."""
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
            # a body that raises leaves the row's collectives on the other ranks unmatched; the step then ends at
            # the WORLD timeout before pytest's summary, so the cause is printed here
            traceback.print_exc()
            sys.stderr.flush()
            raise
    _record_tiers(report, record_property)
    ok = report is None or bool(report["pass"])
    if slurm.world > 1:
        verdicts: list = [None] * slurm.world
        dist.all_gather_object(verdicts, ok)
        ok = all(verdicts)
    if slurm.rank >= W:
        if not ok:
            pytest.fail(f"rank {slurm.rank}: row failed on its {W} member ranks")
        pytest.skip(f"rank {slurm.rank} is not part of the row's {W}-rank group")
    assert ok, _summary(report)


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_forward_rows(row: Row, slurm, record_property) -> None:
    """INT1 forward-only rows (``KCP_TEST_MODE=forward``): #1-#5, every forward arm's ``out`` judged against the FP64
    forward and ``S_in`` against the FP64 cut states, each beside its naive BF16 tier over :data:`DRAWS` draws."""
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
    _record_tiers(report, record_property)
    assert report["pass"], _summary(report)


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_multi_gpu_rows(row: Row, slurm, record_property) -> None:
    """Every forward / backward mode of the row judged against the FP64 reference with its naive BF16 tier over
    :data:`DRAWS` draws (the ``N_REPEAT`` runs of a draw under one name), the exact contracts, the backward's
    collectives, the hosts (see the module docstring)."""
    if _test_mode() != "full":
        pytest.skip("KCP_TEST_MODE=forward: the forward-only rows run instead")
    backend = _backend_name()
    if slurm is not None:
        _slurm_row(row, slurm, backend, _run_rank, record_property)
        return
    _skip_unless_runnable(row.W, backend)
    _spawn_row(row, backend, "full", record_property)


@pytest.mark.parametrize("pair", _selected_poison_pairs(), ids=_pair_id)
def test_poisoned_workspace_pairs(pair: PoisonPair, slurm, record_property) -> None:
    """(d) one ``CakeGDNNondetKcpWorkspace`` reused across the pair's two token layouts on one group, every pool and tape NaN in
    between: every arm of both rows judged against the FP64 reference with its naive BF16 tier over :data:`DRAWS`
    draws."""
    if _test_mode() != "full":
        pytest.skip("KCP_TEST_MODE=forward: the forward-only rows run instead")
    backend = _backend_name()
    if slurm is not None:
        _slurm_row(pair, slurm, backend, _run_rank_poison, record_property)
        return
    _skip_unless_runnable(pair.W, backend)
    _spawn_row(pair, backend, "poison", record_property)


def _fake_starts(monkeypatch, tmp_path, row: Row, failures: list[str]) -> list[int]:
    """Replace the spawned start with one that raises ``failures`` in turn (a rank's error text) and then writes rank
    0's passing report; returns the rendezvous port of every start."""
    ports: list[int] = []
    free = iter(range(41000, 41100))
    monkeypatch.setattr(sys.modules[__name__], "_free_port", lambda: next(free))
    monkeypatch.setenv("KCP_TEST_RESULT_DIR", str(tmp_path))

    def start(fn, *, args, nprocs, join, start_method) -> None:
        ports.append(args[2])
        if len(ports) <= len(failures):
            raise mp.ProcessRaisedException(failures[len(ports) - 1], 0, 0)
        (tmp_path / _report_file(row, "full")).write_text(json.dumps({"pass": True}))

    monkeypatch.setattr(mp, "start_processes", start)
    return ports


TAKEN = (
    "DistNetworkError: The server socket has failed to listen on any local network address. port: 48409, "
    "useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use"
)


def test_spawned_start_retries_a_taken_rendezvous_port(monkeypatch, tmp_path) -> None:
    """A start whose rank 0 finds its rendezvous port taken (EADDRINUSE) runs again on a fresh port.  CPU-only."""
    row = Row("Zz", (1,), (2, 6), 2, "contiguous")
    ports = _fake_starts(monkeypatch, tmp_path, row, [TAKEN])
    _spawn_row(row, "fake", "full")
    assert ports == [41000, 41001]


def test_spawned_start_does_not_retry_other_failures(monkeypatch, tmp_path) -> None:
    """Any other rank failure is raised at once, and a port taken on every try is raised after
    :data:`RENDEZVOUS_TRIES` starts.  CPU-only."""
    row = Row("Zz", (1,), (2, 6), 2, "contiguous")
    ports = _fake_starts(
        monkeypatch, tmp_path, row, ["AssertionError: rank 0: out differs"]
    )
    with pytest.raises(mp.ProcessRaisedException, match="out differs"):
        _spawn_row(row, "fake", "full")
    assert ports == [41000]
    ports = _fake_starts(monkeypatch, tmp_path, row, [TAKEN] * RENDEZVOUS_TRIES)
    with pytest.raises(mp.ProcessRaisedException, match="EADDRINUSE"):
        _spawn_row(row, "fake", "full")
    assert len(ports) == RENDEZVOUS_TRIES


def test_public_api_cannot_select_test_programs() -> None:
    """The precision policy: the shipped record is ``precision_mode=0`` / ``transfer_walk="tf32_hilo"``;
    the test-only records (1 ``_three_mma``, 2 ``_plain_tf32``, ``"bf16x3"``) are refused unless ``test_knobs`` names
    the same value; nothing in the environment changes that (there is no environment hook)."""
    cake_options = _mod("cake_options")
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


def test_long_chain_rows_have_long_owner_chains() -> None:
    """The long-chain rows cover ``zigzag``, ``zigzag_per_document`` and ``contiguous`` at CP 8 and 16, and their owner
    chains (families ``OWNER_FAMILIES`` of the host tables) are longer than ``LONG_CHAIN_MIN`` on every rank that owns
    one (some rank does).  CPU-only."""
    rows = [r for r in ROWS if r.long_chains]
    assert {r.layout for r in rows} == {"zigzag", "zigzag_per_document", "contiguous"}
    assert {r.regime for r in rows} >= {"normal", "A1"}  # A1: state crosses the ranks
    assert {r.W for r in rows} == {8, 16}
    assert all(len(r.lengths) in (1, 2) for r in rows)
    for row in rows:
        H_qk, HV = row.heads
        part, _ = _partition(row)
        owned = 0
        for r in range(row.W):
            ctx = plan.cake_gdn_nondet_train_kcp_build_context(
                part, None, r, torch.device("cpu"), 4, sm_count=148
            )
            tb = kcp.CakeGDNNondetKcpWorkspace().tables(ctx, HV)
            for fam in OWNER_FAMILIES:
                ch = tb.chains[fam]
                lengths = [
                    ch.chain_start[i + 1] - ch.chain_start[i]
                    for i in range(ch.n_chains)
                ]
                if not lengths:
                    continue
                owned += 1
                assert max(lengths) > LONG_CHAIN_MIN, (row.name, r, fam, lengths)
        assert owned > 0, row.name


def test_poison_pairs_change_the_layout() -> None:
    """(d) the rows of every poison pair share a group size and differ in their fragment tables.  CPU-only."""
    for pair in POISON_PAIRS:
        a, b = pair.rows
        assert a.W == b.W, pair.name
        assert pair.stage in STAGES, pair.name
        pa, pb = _partition(a)[0], _partition(b)[0]
        frags = lambda p: [  # noqa: E731
            [(f.doc, f.off, f.length) for f in fr] for fr in p.fragments
        ]
        assert frags(pa) != frags(pb), pair.name


# ----------------------------------------------------------------------------------------------------------------
# #7817 / #8063: the in-kernel Q / K L2 norm (``use_qk_l2norm_in_kernel=True``) of this package.  ``q`` / ``k`` are RAW
# rows here (the generator's rows times a per-token, per-head power of two drawn from the global token index and the
# draw's seed, so every rank sees the bits of the global pack).  Per row, rank and draw:
#
# * the oracle: the op computed in FP64 from its raw inputs (``gdn_exact_l2n_with_grads``: the raw rows normalized in
#   FP64, ``x / sqrt(sum x^2 + 1e-6)``, with no rounding of the normalized rows, the exact recurrence on them and its
#   autograd, the q / k gradients carried back through the FP64 norm); its naive tier (``gdn_naive_l2n_with_grads``):
#   the same chain with every op in BF16 (the norm, the recurrence and its autograd, the pullback);
# * every form with the norm judged (``N_REPEAT`` runs under one name, light / full (``qk_apply``) / boundary replay,
#   autograd / saved, checkpoint in its replay / kept-tape / "auto" / full / boundary-replay forms, recompute handle,
#   ``save_transfer``, ring / a2a where the partition allows it, the forced ``P = 2`` forms where every rank plans
#   ``P = 1``, CUDA-graph replay on the kernel backend, ``.backward()`` twice);
# * the composed chain, each part judged: the norm-off route on the forward's own ``(qn, kn)`` (out / dv / dbeta /
#   dg) and ``bf16(l2n_pullback(x, r, D))`` with ``D`` that route's ``group_reduce="host"`` partials summed in head
#   order (dq / dk);
# * the norm's rows: the forward's ``qn`` / ``kn`` and their FP32 ``r`` (``record_qk``) judged against the FP64 norm
#   of the raw rows (``l2n_exact``: ``x / n``, ``1 / n``) beside its BF16 tier (``l2n_naive``);
# * packed QKV views with the norm: judged; the backward writes one packed dQKV row and
#   ``cake_gdn_nondet_train_kcp_packed_qkv_views`` hands it to the packed leaf without a concatenation;
# * memory (first draw): no ``qn`` / ``kn`` / ``qk_rstd`` in the arena after a forward or a backward; the bytes a
#   forward retains with the norm (``save_mode="light"``) == without + ``8 T_r H_qk`` (the saved r); saved bytes on -
#   off == the query's ``qk_rstd`` term; the default with the norm (``"light_akk"``) recorded with its ``Akk`` tape (not
#   gated);
# * ``save_mode="light_akk"`` (the default with the norm: the light set plus #1_l2n's ``Akk`` tape; the light replay
#   runs ``fwd_prep_replay_l2n``, #1_l2n reading that tape): judged in every form (a2a / ring, ``save_transfer`` on /
#   off, the plan's ``P`` and the forced ``P = 2``, autograd, checkpoint, CUDA-graph replay steps on the light steps'
#   workspace); exact: the launch census (the backward's ``fwd_prep_l2n`` becomes the replay, once per light replay;
#   the forward is light's), the saved ``Akk`` bytes (``128 R HV``, the byte query's difference), the checkpoint
#   saving as ``"light"``, and on the kernel backend every form beside its light form (run twice) under NaN-filled
#   allocations, every one judged, with the replay run.
#
# The rows skip with a clear reason while the registry does not ship the three records (the kernel backend refuses
# the option then); ``KCP_TEST_BACKEND=fake`` runs them on the torch emulation.

L2N_SKIP = (
    "the in-kernel Q/K L2 norm records (fwd_prep_l2n, bwd_dqkg_l2n, qk_apply) are not in the shipped registry "
    "yet: use_qk_l2norm_in_kernel cannot run on the kernel backend"
)
L2N = {"use_qk_l2norm_in_kernel": True}
# the norm's rows of a forward (record_qk), judged: q_hat, k_hat (BF16) and their rstd (FP32)
L2N_ROWS = ("qn", "kn", "rq", "rk")


def _l2n_shipped() -> bool:
    shipped = getattr(kcp, "l2n_records_shipped", None)
    return bool(shipped()) if callable(shipped) else False


def _skip_unless_l2n(backend: str) -> None:
    if backend == "kernels" and not _l2n_shipped():
        pytest.skip(L2N_SKIP)


def _raw_exponents(total: int, H: int, salt: int, seed: int) -> torch.Tensor:
    """The per-(global token, head) power-of-two exponents in [-6, 6] of the raw rows of one draw (CPU, float), drawn
    from the draw's operand seed and ``salt`` (1: q, 2: k)."""
    gen = torch.Generator(device="cpu").manual_seed(78170 + salt + seed)
    return torch.randint(-6, 7, (max(total, 1), H, 1), generator=gen).float()


def _raw_qk(
    row: Row, part, cu: list[int], rank: int, x: torch.Tensor, salt: int, seed: int
) -> torch.Tensor:
    """``x`` (this rank's L2-normalized generator rows) times ``2^e``, ``e`` in [-6, 6] per (global token, head)."""
    e = _raw_exponents(cu[-1], x.shape[1], salt, seed)
    l2g = _local_to_global(part, cu, rank)
    return (x.float() * torch.exp2(e[l2g]).to(x.device)).to(torch.bfloat16)


def _oracle_l2n(row: Row, cu: list[int], device, seed: int) -> tuple[dict, dict]:
    """Rank 0, one draw: the FP64 reference of the op with the norm on the global pack and its naive BF16 tier, per
    document, from the RAW rows (the bits :func:`_raw_qk` gives every rank).  The oracle: ``gdn_exact_l2n_with_grads``
    (the raw rows normalized in FP64 with no rounding of the normalized rows, the exact recurrence and its autograd,
    ``dq`` / ``dk`` the gradients of the raw rows) and the norm's rows from ``l2n_exact`` (``qn`` / ``kn`` = ``x / n``,
    ``rq`` / ``rk`` = ``1 / n``).  The naive tier: ``gdn_naive_l2n_with_grads`` and ``l2n_naive`` (every op in
    BF16)."""
    ref_mod = _reference()
    H_qk, HV = row.heads
    total = cu[-1]
    glob = _draw_inputs(row, cu, seed)
    pack = {n: glob[n][0] for n in ("q", "k", "v", "g", "beta", "dout")}
    for name, salt in (("q", 1), ("k", 2)):
        e = _raw_exponents(total, H_qk, salt, seed)[:total]
        pack[name] = (pack[name].float() * torch.exp2(e)).to(torch.bfloat16)
    names = TENSORS + L2N_ROWS
    exact: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    naive: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for d in range(len(cu) - 1):
        s0, s1 = cu[d], cu[d + 1]
        if s1 <= s0:
            continue
        ops = [pack[n][s0:s1].to(device) for n in ("q", "k", "v", "g", "beta", "dout")]
        r = ref_mod.gdn_exact_l2n_with_grads(*ops, [0, s1 - s0], scale=SCALE)
        rn = ref_mod.gdn_naive_l2n_with_grads(*ops, [0, s1 - s0], scale=SCALE)
        qh, nq = ref_mod.l2n_exact(ops[0])
        kh, nk = ref_mod.l2n_exact(ops[1])
        qb, mq = ref_mod.l2n_naive(ops[0])
        kb, mk = ref_mod.l2n_naive(ops[1])
        doc = {n: r[n] for n in TENSORS}
        doc.update(qn=qh, kn=kh, rq=1.0 / nq[..., 0], rk=1.0 / nk[..., 0])
        doc_naive = {n: rn[n] for n in TENSORS}
        doc_naive.update(qn=qb, kn=kb, rq=1.0 / mq[..., 0], rk=1.0 / mk[..., 0])
        for n in names:
            exact[n].append(doc[n].to(F64).cpu())
            naive[n].append(doc_naive[n].to(torch.bfloat16).cpu())
        del ops, r, rn, doc, doc_naive
    return (
        {n: torch.cat(ts, 0) for n, ts in exact.items()},
        {n: torch.cat(ts, 0) for n, ts in naive.items()},
    )


def _run_rank_l2n(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """One rank of a #7817 row of this package over :data:`DRAWS` draws (see the block comment above)."""
    l2n = _mod("cake_l2n")
    gb = _mod("cake_bytes")
    jit = _mod("cake_jit")
    H_qk, HV = row.heads
    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    ws = kcp.CakeGDNNondetKcpWorkspace()
    base = {"backend": backend}
    kernels = backend == "kernels"
    t_start = time.time()
    gj = _GroupJudge(rank, W, group, device)
    flags: dict[str, bool] = {}
    info: dict[str, object] = {}
    ring_eligible = W > 1 and int(ctx.n_d_max) - 1 <= kcp.RING_MAX_HOPS
    # the forced P = 2 forms (see _run_rank)
    info["p2_eligible"] = _p2_group_ok(ctx, HV, group, W, device)
    akk_replay = kcp.AKK_REPLAY_OF["fwd_prep_l2n"]
    akk_info: dict[str, object] = {"shipped": bool(jit.records_of_program(akk_replay))}
    # the replay not rendered (kernel backend): nothing of the light_akk mode runs
    run_akk = not kernels or bool(akk_info["shipped"])

    def draw(j: int, seed: int) -> None:
        """Draw ``j`` (operand seed ``seed``): every judged form with the norm; the exact checks of the draw; the
        memory records on the first draw."""
        first = j == 0
        gj.begin(j, part, cu, row.heads, lambda: _oracle_l2n(row, cu, device, seed))
        ops = _local_inputs(row, part, cu, rank, device, seed=seed)
        q = _raw_qk(row, part, cu, rank, ops["q"], 1, seed)
        k = _raw_qk(row, part, cu, rank, ops["k"], 2, seed)
        v, g, beta, dout = (ops[n] for n in ("v", "g", "beta", "dout"))
        del ops
        T_r = int(q.shape[0])
        if first:
            info["T_r"] = T_r

        def saved_pass(qq=q, kk=k, workspace=ws, **opts) -> dict:
            o = {**base, **opts}
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                qq, kk, v, g, beta, ctx, workspace=workspace, **o
            )
            rec_f = dict(kcp.LAST_CALL["fwd"])
            dq, dk, dv, dbeta, dg = kcp.cake_gdn_nondet_train_kcp_backward(
                qq,
                kk,
                v,
                g,
                beta,
                ctx,
                dout,
                saved=saved,
                workspace=workspace,
                **o,
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
            leaves = [
                x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
            ]
            o = {**base, **opts}
            hd = kcp.CakeGDNNondetKcpRecomputeHandle() if handle else None

            def fn(q_, k_, v_, g_, b_, h_):
                return kcp.cake_gdn_nondet_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctx,
                    workspace=ws,
                    recompute_handle=h_,
                    **o,
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

        # (a) the norm on: N_REPEAT runs under one name; every form judged.  The light arms name save_mode="light"
        # (the default with the norm is "light_akk"; the arms without a save mode run it)
        res = saved_pass(save_mode="light", **L2N, test_knobs={"record_qk": True})
        rec_fwd_qk = res["fwd"].get("qk")
        if row.production:
            route = _production_route(ctx, HV, res["fwd"], res["bwd"])
            _flag(flags, "production_route", route["ok"])
            _keep(info, "production_route", j, route, route["ok"])
        if first:
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
        _flag(
            flags,
            "l2n_forward_ran_the_norm",
            bool(res["fwd"]["qk_l2norm"])
            and (
                T_r == 0
                or (
                    "fwd_prep_l2n" in res["fwd"]["sequence"]
                    and "bwd_dqkg_l2n" in res["bwd"]["sequence"]
                )
            ),
        )
        gj.add("run", res)
        for _ in range(1, N_REPEAT):
            gj.add("run", saved_pass(save_mode="light", **L2N))
        full = saved_pass(save_mode="full", **L2N, test_knobs={"record_qk": True})
        gj.add("full", full)
        _flag(
            flags,
            "l2n_full_rebuilds_with_qk_apply",
            T_r == 0 or full["bwd"]["qk_source"] == "qk_apply",
        )
        gj.add("replay", saved_pass(boundary_mode="replay", **L2N))
        res_nt = saved_pass(save_mode="light", save_transfer=False, **L2N)
        gj.add("no_save_transfer", res_nt)
        # save_mode="full" without save_transfer under NaN-filled allocations (the M_seg^T plane at P > 1; _run_rank)
        res_full_nt = _nan_allocations_pass(
            lambda: saved_pass(save_mode="full", save_transfer=False, **L2N)
        )
        gj.add("full_no_save_transfer", res_full_nt)
        rule = _m_seg_rule(res_full_nt["saved"])
        _flag(flags, "full_no_save_transfer_saves_m_seg", rule["ok"])
        if first:
            info["full_no_save_transfer"] = rule
        del res_full_nt, res_nt
        gj.add("autograd", autograd_pass(False, save_mode="light", **L2N))
        for name, opts in (
            ("ckpt_replay_form", {"reuse_recompute_tapes": False}),
            ("ckpt_kept", {"reuse_recompute_tapes": True}),
            ("ckpt_auto", {"reuse_recompute_tapes": "auto"}),
            ("ckpt_full", {"save_mode": "full"}),
            ("ckpt_boundary_replay", {"boundary_mode": "replay"}),
        ):
            res_ck = autograd_pass(True, **opts, **L2N)
            if name == "ckpt_kept" and first:
                info["kept_tapes_source"] = kcp.LAST_CALL["bwd"].get("tapes_source")
                info["kept_qk_source"] = kcp.LAST_CALL["bwd"].get("qk_source")
            gj.add(name, res_ck)
            del res_ck
        gj.add("handle", autograd_pass(True, handle=True, save_transfer=False, **L2N))
        gj.add(
            "backward_twice_half",
            autograd_pass(True, handle=True, twice=True, save_transfer=False, **L2N),
            GRADS,
            half=True,
        )
        if ring_eligible:
            res_ring = saved_pass(
                exchange="ring", save_mode="light", save_transfer=False, **L2N
            )
            gj.add("ring", res_ring)
            res_ring_full = _nan_allocations_pass(
                lambda: saved_pass(
                    exchange="ring", save_transfer=False, save_mode="full", **L2N
                )
            )
            gj.add("ring_full", res_ring_full)
            rule = _m_seg_rule(res_ring_full["saved"])
            _flag(flags, "ring_full_saves_m_seg", rule["ok"])
            if first:
                info["ring_full"] = rule
            del res_ring, res_ring_full
        # the forced P = 2 forms (see _run_rank)
        if info["p2_eligible"]:
            _forced_arms(
                saved_pass,
                gj,
                info,
                flags,
                P=2,
                ring=ring_eligible,
                first=first,
                **L2N,
            )
        # (a2) save_mode="light_akk": the light set plus #1_l2n's BF16 Akk tape, whose light replay runs
        # fwd_prep_replay_l2n (#1_l2n reading that tape).  Every form judged; the launch census (the backward's
        # fwd_prep_l2n becomes the replay, once per light replay; the forward is light's), the saved Akk bytes (128 R
        # HV, the byte query's difference); on the kernel backend each form beside the light form of the same options
        # (run twice, one name) under NaN-filled allocations, every form judged and the replay run gated; autograd
        # (the saved set through save_for_backward) and a non-reentrant checkpoint (the mode saves as "light" there).
        if run_akk:
            res_akk = saved_pass(save_mode="light_akk", **L2N)
            gj.add("light_akk", res_akk)
            lb, ab = res["bwd"], res_akk["bwd"]
            n_l2n = lb["sequence"].count("fwd_prep_l2n")
            _flag(
                flags,
                "light_akk_census",
                res_akk["fwd"]["sequence"] == res["fwd"]["sequence"]
                and ab["sequence"]
                == [akk_replay if p_ == "fwd_prep_l2n" else p_ for p_ in lb["sequence"]]
                and ab["sequence"].count(akk_replay) == n_l2n
                and "fwd_prep_l2n" not in ab["sequence"]
                and (T_r == 0 or (n_l2n >= 1 and ab["akk_source"] == "saved")),
            )
            if kernels:
                replay_records = set(jit.records_of_program(akk_replay))
                _flag(
                    flags,
                    "light_akk_replay_records",
                    sum(r_ in replay_records for r_, _g in ab["records"])
                    == ab["sequence"].count(akk_replay)
                    and [g_ for _r, g_ in ab["records"]]
                    == [g_ for _r, g_ in lb["records"]],
                )
            tape = int(ctx.R) * HV * 128
            q_light = int(
                gb.cake_gdn_nondet_train_kcp_saved_bytes(
                    ctx, H_qk=H_qk, HV=HV, save_mode="light", with_conv=False, **L2N
                )
            )
            q_akk = int(
                gb.cake_gdn_nondet_train_kcp_saved_bytes(
                    ctx,
                    H_qk=H_qk,
                    HV=HV,
                    save_mode="light_akk",
                    with_conv=False,
                    **L2N,
                )
            )
            _flag(
                flags,
                "light_akk_saved_bytes",
                int(res_akk["saved"].bytes) - int(res["saved"].bytes)
                == tape
                == q_akk - q_light,
            )
            if first:
                akk_info["census"] = {
                    "light_bwd": list(lb["sequence"]),
                    "light_akk_bwd": list(ab["sequence"]),
                    "akk_source": ab["akk_source"],
                }
                akk_info["saved_bytes"] = {
                    "light": int(res["saved"].bytes),
                    "light_akk": int(res_akk["saved"].bytes),
                    "tape": tape,
                    "query_difference": q_akk - q_light,
                }
            del res_akk
            if kernels:
                forms = [
                    ("akk", {}),
                    ("akk_no_save_transfer", {"save_transfer": False}),
                ]
                if info["p2_eligible"]:
                    forms += [
                        ("p2_akk", {"segments": 2}),
                        (
                            "p2_akk_no_save_transfer",
                            {"segments": 2, "save_transfer": False},
                        ),
                    ]
                if ring_eligible:
                    forms.append(
                        ("ring_akk", {"exchange": "ring", "save_transfer": False})
                    )
                for name, opts in forms:
                    o = {**opts, **L2N}
                    lt = _nan_allocations_pass(
                        lambda o=o: saved_pass(save_mode="light", **o)
                    )
                    lt2 = _nan_allocations_pass(
                        lambda o=o: saved_pass(save_mode="light", **o)
                    )
                    ak = _nan_allocations_pass(
                        lambda o=o: saved_pass(save_mode="light_akk", **o)
                    )
                    # the light form's two runs under one name (the judge keeps the larger error of the draw)
                    gj.add(f"{name}_light", lt)
                    gj.add(f"{name}_light", lt2)
                    gj.add(f"{name}_light_akk", ak)
                    if first:
                        akk_info[name] = {
                            "P": None if ak["saved"] is None else int(ak["saved"].P),
                            "akk_source": ak["bwd"]["akk_source"],
                        }
                    _flag(
                        flags,
                        f"{name}_replay_ran",
                        T_r == 0 or ak["bwd"]["akk_source"] == "saved",
                    )
                    del lt, lt2, ak
            res_aa = autograd_pass(False, save_mode="light_akk", **L2N)
            _flag(
                flags,
                "light_akk_autograd_replay_ran",
                T_r == 0 or kcp.LAST_CALL["bwd"]["akk_source"] == "saved",
            )
            gj.add("light_akk_autograd", res_aa)
            res_ck = autograd_pass(True, save_mode="light_akk", **L2N)
            _flag(
                flags,
                "light_akk_ckpt_saves_light",
                kcp.LAST_CALL["bwd"]["save_mode"] == "light"
                and akk_replay not in kcp.LAST_CALL["bwd"]["sequence"],
            )
            gj.add("light_akk_ckpt", res_ck)
            del res_aa, res_ck

        # (c) the composed chain on the forward's own rows, each part judged: the norm-off route on (qn, kn) and the
        # reference pullback of that route's group_reduce="host" partials.  The chain's pass exchanges states, so every
        # rank issues it (a rank without tokens passes its empty rows); skipping it there would put the ranks'
        # collectives out of step.
        ws0 = kcp.CakeGDNNondetKcpWorkspace()
        chain = saved_pass(
            qq=(rec_fwd_qk["qn"] if T_r > 0 else q).clone(),
            kk=(rec_fwd_qk["kn"] if T_r > 0 else k).clone(),
            workspace=ws0,
            group_reduce="host",
        )
        gj.add("chain_norm_off", chain, ("out", "dv", "dbeta", "dg"))
        want = {}
        if T_r > 0:
            rstd = rec_fwd_qk["rstd"]
            r_ = HV // H_qk
            for name, part_name, x, plane in (
                ("dq", "dq_part", q, 0),
                ("dk", "dk_part", k, 1),
            ):
                p = ws0.tape(part_name).view(T_r, H_qk, r_, 128)
                acc = p[:, :, 0]
                for jj in range(1, r_):
                    acc = acc + p[:, :, jj]
                want[name] = l2n.l2n_pullback(x, rstd[plane], acc).to(torch.bfloat16)
        else:
            want = {n: torch.zeros_like(res[n]) for n in ("dq", "dk")}
        gj.add("chain_pullback", want, ("dq", "dk"))
        del chain, ws0, want
        # (c2) the norm's rows of the forward (record_qk: fwd_prep_l2n's q_hat / k_hat and their FP32 rstd) judged
        # against the FP64 norm of the raw rows (x / n, 1 / n) beside its BF16 tier; the other launches of the norm
        # (the backward's regenerated rows, qk_apply of the full form, a standalone #1_l2n) feed the outputs judged
        # above (run, full, chain_norm_off)
        if T_r > 0:
            rows_ = {
                "qn": rec_fwd_qk["qn"],
                "kn": rec_fwd_qk["kn"],
                "rq": rec_fwd_qk["rstd"][0],
                "rk": rec_fwd_qk["rstd"][1],
            }
        else:
            rows_ = {
                "qn": q.new_zeros((0, H_qk, 128)),
                "kn": k.new_zeros((0, H_qk, 128)),
                "rq": torch.zeros((0, H_qk), dtype=torch.float32, device=device),
                "rk": torch.zeros((0, H_qk), dtype=torch.float32, device=device),
            }
        gj.add("l2n_rows", rows_, L2N_ROWS)
        del full, rec_fwd_qk, rows_

        # (d) packed QKV views with the norm: judged; one packed dQKV row, and the split aliases it
        buf, qs, ks, vs = _packed_views(q, k, v)
        gj.add("strided", saved_pass(qq=qs, kk=ks, **L2N))
        split_before = dict(kcp.PACKED_SPLIT_STATS)
        qkv_leaf = buf.detach().clone().requires_grad_(True)
        g_leaf, b_leaf = (x.detach().clone().requires_grad_(True) for x in (g, beta))
        pq, pk, pv = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(
            qkv_leaf, H_qk=H_qk, HV=HV
        )

        def split_call():
            out_ = kcp.cake_gdn_nondet_kcp(
                pq,
                pk,
                pv,
                g_leaf,
                b_leaf,
                ctx,
                workspace=ws,
                **base,
                **L2N,
            )
            out_.backward(dout)
            return out_

        out_split = None
        if first:
            # the copy ops of one packed call (the profiler is evidence only)
            try:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    out_split = split_call()
                names = [e.name for e in prof.events()]
                info["profiler_copy_ops"] = {
                    n: names.count(n)
                    for n in (
                        "aten::copy_",
                        "aten::contiguous",
                        "aten::clone",
                        "aten::cat",
                    )
                    if n in names
                }
            except Exception as exc:  # noqa: BLE001 - the profiler is evidence only
                info["profiler_copy_ops"] = (
                    f"unavailable: {type(exc).__name__}: {str(exc)[:120]}"
                )
        if out_split is None:
            out_split = split_call()
        _flag(
            flags,
            "l2n_packed_grad_layout",
            T_r == 0 or kcp.LAST_CALL["bwd"]["grad_layout"] == "packed",
        )
        ga = qkv_leaf.grad
        res_split = {
            "out": out_split.detach(),
            "dq": ga[:, : H_qk * 128].reshape(-1, H_qk, 128),
            "dk": ga[:, H_qk * 128 : 2 * H_qk * 128].reshape(-1, H_qk, 128),
            "dv": ga[:, 2 * H_qk * 128 :].reshape(-1, HV, 128),
            "dg": g_leaf.grad,
            "dbeta": b_leaf.grad,
        }
        gj.add("packed_split", res_split)
        _flag(
            flags,
            "l2n_packed_split_aliased",
            T_r == 0
            or kcp.PACKED_SPLIT_STATS["aliased"] - split_before["aliased"] == 1,
        )
        del qkv_leaf, out_split, ga, res_split

        # issue #8367 with the norm: plain split views, direct and checkpointed (see _run_rank)
        def gdn_call(*a):
            return kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=ws, **base, **L2N)

        for ck, tag in ((False, "direct"), (True, "ckpt")):
            res_plain, hooks_ok = _plain_split_pass(
                buf, g, beta, dout, gdn_call, H_qk, HV, ckpt=ck
            )
            layout_ok = kcp.LAST_CALL["bwd"]["grad_layout"] == "separate"
            gj.add(f"plain_split_{tag}", res_plain)
            _flag(flags, f"l2n_plain_split_{tag}_hooks_fire", hooks_ok)
            _flag(flags, f"l2n_plain_split_{tag}_grad_layout_separate", layout_ok)
            del res_plain
        del buf, qs, ks, vs

        # (e) CUDA-graph replay with the norm: the fourth forward + backward replays; judged on the kernel backend
        graph_info: dict[str, object] = {}
        g_res = None
        try:
            kcp.graph_cache_clear()
            events = []
            for _ in range(4):
                out_g, saved_g = kcp.cake_gdn_nondet_train_kcp_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    workspace=ws,
                    graph="replay",
                    save_mode="light",
                    **base,
                    **L2N,
                )
                grads_g = kcp.cake_gdn_nondet_train_kcp_backward(
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
                    save_mode="light",
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
                        str(kcp.LAST_CALL["fwd"]["graph_event"]),
                        str(kcp.LAST_CALL["bwd"]["graph_event"]),
                    ]
                )
            graph_info["events"] = events
            graph_ok = events[-1] == ["replay", "replay"]
            torch.cuda.synchronize(device)
            kcp.graph_cache_clear()
        except Exception as exc:  # noqa: BLE001 - recorded; the gate decides
            graph_info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            graph_ok = False
            g_res = None
            kcp.graph_cache_clear()
        g_cmp = (
            g_res
            if g_res is not None
            else {n: torch.zeros_like(res[n]) for n in TENSORS}
        )
        if kernels:
            gj.add("graph_replay", g_cmp)
        _keep(info, "graph", j, graph_info, graph_ok)
        _flag(flags, "l2n_graph_capture", graph_ok)
        del g_res, g_cmp

        # (e2) the same four steps with save_mode="light_akk" on the same workspace (its arena holds the light steps'
        # tapes): the fourth replays both graphs and launches the replay in place of fwd_prep_l2n; judged on the
        # kernel backend
        akk_graph: dict[str, object] = {}
        a_res = None
        akk_graph_ok = (
            True  # the replay not rendered (kernel backend): nothing of the mode runs
        )
        if run_akk:
            try:
                kcp.graph_cache_clear()
                events, sources = [], []
                for _ in range(4):
                    out_g, saved_g = kcp.cake_gdn_nondet_train_kcp_forward(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        ctx,
                        workspace=ws,
                        graph="replay",
                        save_mode="light_akk",
                        **base,
                        **L2N,
                    )
                    grads_g = kcp.cake_gdn_nondet_train_kcp_backward(
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
                        save_mode="light_akk",
                        **base,
                        **L2N,
                    )
                    a_res = {
                        "out": out_g.clone(),
                        **{
                            n: t.clone()
                            for n, t in zip(
                                ("dq", "dk", "dv", "dbeta", "dg"),
                                grads_g,
                                strict=True,
                            )
                        },
                    }
                    events.append(
                        [
                            str(kcp.LAST_CALL["fwd"]["graph_event"]),
                            str(kcp.LAST_CALL["bwd"]["graph_event"]),
                        ]
                    )
                    sources.append(kcp.LAST_CALL["bwd"].get("akk_source"))
                akk_graph["events"], akk_graph["akk_source"] = events, sources
                akk_graph_ok = events[-1] == ["replay", "replay"] and (
                    T_r == 0 or all(src == "saved" for src in sources)
                )
                torch.cuda.synchronize(device)
                kcp.graph_cache_clear()
            except Exception as exc:  # noqa: BLE001 - recorded; the gate decides
                akk_graph["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                akk_graph_ok = False
                a_res = None
                kcp.graph_cache_clear()
            a_cmp = (
                a_res
                if a_res is not None
                else {n: torch.zeros_like(res[n]) for n in TENSORS}
            )
            if kernels:
                gj.add("graph_replay_light_akk", a_cmp)
            del a_res, a_cmp
        _keep(info, "graph_light_akk", j, akk_graph, akk_graph_ok)
        _flag(flags, "l2n_graph_capture_light_akk", akk_graph_ok)

        if first:
            _l2n_memory(ctx, (q, k, v, g, beta, dout), row.heads, base, gb, flags, info)

    for j in range(DRAWS if total > 0 else 1):
        draw(j, draw_seed(SEED, j))
    info["light_akk"] = akk_info
    verdicts, failures = gj.finish()
    flag_list: list = [None] * W
    info_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags, group=group)
        dist.all_gather_object(info_list, info, group=group)
    else:
        flag_list, info_list = [flags], [info]
    report = _new_report(row, rank, W, backend, device, "l2n")
    report["info_by_rank"] = info_list
    keys: list[str] = []
    for fl in flag_list:
        for key in fl:
            if key not in keys:
                keys.append(key)
    # a rank without tokens has no rows to compare: the token-local flags exist on the ranks with tokens only
    padded = [{key: fl.get(key, True) for key in keys} for fl in flag_list]
    _fold(
        report,
        padded,
        backend=backend,
        kernel_only={"l2n_graph_capture", "l2n_graph_capture_light_akk"},
    )
    _fold_judge(report, verdicts, failures)
    report["seconds"] = round(time.time() - t_start, 2)
    return _finish_report(report, group, W, rank, result_dir, "l2n", row)


def _l2n_memory(
    ctx, operands: tuple, heads: tuple[int, int], base: dict, gb, flags, info
) -> None:
    """(f) The first draw's memory records of :func:`_run_rank_l2n`: nothing of the norm in the arena after a forward
    or a backward; the bytes a forward retains with the norm (``save_mode="light"``) = without + the saved r; saved
    bytes on - off = the byte query's ``qk_rstd`` term; the default with the norm recorded."""
    q, k, v, g, beta, dout = operands
    H_qk, HV = heads
    T_r = int(q.shape[0])
    device = q.device
    wsm = kcp.CakeGDNNondetKcpWorkspace()
    bad = {*kcp.QK_TAPES, kcp.QK_RSTD}
    out_m, saved_m = kcp.cake_gdn_nondet_train_kcp_forward(
        q, k, v, g, beta, ctx, workspace=wsm, **base, **L2N
    )
    flags["l2n_absent_after_forward"] = not (set(wsm._tapes) & bad)
    kcp.cake_gdn_nondet_train_kcp_backward(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        dout,
        saved=saved_m,
        workspace=wsm,
        **base,
        **L2N,
    )
    flags["l2n_absent_after_backward"] = not (set(wsm._tapes) & bad)
    del out_m, saved_m, wsm

    def retained(**opts) -> tuple[int, int, int]:
        kcp.cake_gdn_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=None, **base, **opts
        )  # warm-up of the form
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        before = _alloc_stats(device)
        live = torch.cuda.memory_allocated(device)
        out_r, saved_r = kcp.cake_gdn_nondet_train_kcp_forward(
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
    kept_on, saved_on_b, peak_on = retained(save_mode="light", **L2N)
    # the default with the norm ("light_akk"): recorded, not gated (+ the Akk tape, 128 R HV bytes, against "light")
    kept_def, saved_def_b, peak_def = retained(**L2N)
    rstd_bytes = 8 * T_r * H_qk
    info["memory"] = {
        "retained_off": kept_off,
        "retained_on": kept_on,
        "saved_off": saved_off_b,
        "saved_on": saved_on_b,
        "fwd_peak_off": peak_off,
        "fwd_peak_on": peak_on,
        "qn_kn_bytes": 2 * T_r * H_qk * 256,
        "retained_on_default": kept_def,
        "saved_on_default": saved_def_b,
        "fwd_peak_on_default": peak_def,
        "akk_tape_bytes": int(ctx.R) * HV * 128,
    }
    flags["l2n_post_forward_eq_off_plus_rstd"] = kept_on - kept_off == rstd_bytes
    q_on = int(
        gb.cake_gdn_nondet_train_kcp_saved_bytes(
            ctx, H_qk=H_qk, HV=HV, save_mode="light", use_qk_l2norm_in_kernel=True
        )
    )
    q_off = int(gb.cake_gdn_nondet_train_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV))
    info["saved_query"] = {"on": q_on, "off": q_off}
    flags["l2n_saved_bytes_on_minus_off"] = (
        saved_on_b - saved_off_b == rstd_bytes == q_on - q_off
    )


@pytest.mark.parametrize("row", _selected_rows(), ids=_row_id)
def test_l2n_rows(row: Row, slurm, record_property) -> None:
    """#7817 rows: the in-kernel Q / K L2 norm on the row's partition, every form judged against the FP64 reference of
    the norm with its naive BF16 tier over :data:`DRAWS` draws (see the block comment above).  Skips with a clear
    reason while the registry does not ship the three records (kernel backend)."""
    if _test_mode() != "full":
        pytest.skip("KCP_TEST_MODE=forward: the forward-only rows run instead")
    backend = _backend_name()
    _skip_unless_l2n(backend)
    if slurm is not None:
        _slurm_row(row, slurm, backend, _run_rank_l2n, record_property)
        return
    _skip_unless_runnable(row.W, backend)
    _spawn_row(row, backend, "l2n", record_property)


_BODIES["l2n"] = _run_rank_l2n
