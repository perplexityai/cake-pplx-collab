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

Cross-family rows of the non-deterministic context-parallel (KCP) packed GDN training package
(``cake_pplx._backends.cake_gdn_nondet_train_kcp``): the rows that also need the causal_conv1d family
(``cake-pplx-causal-conv1d``), moved here from ``kernels/gdn/tests/experimental/
test_cake_gdn_nondet_train_kcp_dist.py`` (the in-family rows stay there).

* ``test_conv_halo_row`` (C7): the conv package's ``cake_causal_conv1d_det_kcp`` with THIS package's context;
* ``test_layer_checkpoint_rows`` (L4 / L8 / L8r): KCP convolution + caller L2 norm + ``cake_gdn_nondet_kcp`` in one
  checkpointed region with one ``CakeGDNNondetKcpRecomputeHandle`` (the recompute exchanges nothing; the original
  forward and the backward exchange as the plain checkpoint does; the backward of every form issues exactly the
  fixed collective list ``LAYER_BWD_COLLECTIVES``; no parameter reduction); out / dx / dweight / dg / dbeta of the
  direct pass, the plain checkpoint and the handle checkpoint, each judged against the FP64 layer oracle;
* ``test_l2n_layer_checkpoint_rows``: the same layer with ``use_qk_l2norm_in_kernel=True`` on the raw convolution
  output views, judged against the FP64 oracle of the in-kernel norm (``gdn_exact_l2n_with_grads``: the raw rows
  normalized in FP64; only its naive tier normalizes in BF16).

Accuracy is the naive BF16 tier of ``tests/cake_naive.py``: every floating output's error against the FP64 oracle
beside the error of a PyTorch BF16 implementation of the same ops on the same inputs, median and largest over
``DRAWS`` (32) independent draws of the operands, L2 and max norm.  Each row runs its 32 draws inside one process
group; rank 0 gathers every draw's outputs, judges them and records one ``naive_tier`` verdict per output.

All of them run under Slurm (one process per task: ``SLURM_PROCID`` / ``SLURM_NTASKS``; the WORLD group is initialised
once per process and every row runs on a sub-group of its ``W`` lowest ranks).  Environment: ``KCP_TEST_BACKEND``
(``kernels`` | ``fake``), ``KCP_TEST_RESULT_DIR`` (one JSON per row and rank), ``KCP_TEST_PORT_DIR`` (the Slurm
rendezvous port file).  The suite imports this package and the causal-convolution package through
``tests/cake/test_helpers/cake_package.py``; a missing distribution fails the row, never skips.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import importlib
import inspect
import json
import os
import re
import socket
import sys
import tempfile
import time
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed
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


def _conv_mod(sub: str | None = None):
    """The deterministic causal-convolution package (or its ``kcp`` module) through the test-helper registry: the
    entries ``cake_causal_conv1d_det`` / ``is_cake_causal_conv1d_det_train_supported`` and
    ``kcp.cake_causal_conv1d_det_kcp`` that its facades forward to."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_causal_conv1d_det_train", sub)


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
PACKAGE_DIR = Path(kcp.__file__).resolve().parent
LAYER_TENSORS = ("out", "dx", "dw", "dg", "dbeta")
WORLD_TIMEOUT_S = 300  # NCCL / rendezvous timeout of the WORLD group (printed at init)
SCALE = kcp.SCALE
F64 = torch.float64
# (b) collectives that would reduce parameters across the group: never issued by the route
PARAM_REDUCTIONS = ("all_reduce", "reduce_scatter_tensor")
# (b) the collectives of the layer's backward by torch.distributed name, the same on every rank and in every form
# (direct, plain checkpoint, recompute handle: the handle removes the recompute's exchanges only), for a partition
# with a document across ranks (every layer row): the GDN backward's two all_to_all_single exchanges (b1: the adjoint
# offsets and maps to the head owners; b2: the boundary adjoints back) and the convolution backward's halo exchange
# (one all_to_all_single); nothing else
LAYER_BWD_COLLECTIVES = {"all_to_all_single": 3}


def _reference():
    """The FP64 oracle and its naive BF16 tier (the in-tree test helper, read only)."""
    from tests.experimental.test_helpers import cake_gdn_det_train_kcp_reference as ref

    return ref


def _inputs():
    """The seeded operand generator (``make_inputs(cu, H_qk, HV, *, regime, seed, token_range)``): the inputs helper,
    else the reference helper (the same generator, bitwise), else the session harness on PYTHONPATH."""
    for name in (
        "tests.experimental.test_helpers.cake_gdn_det_train_kcp_inputs",
        "tests.experimental.test_helpers.cake_gdn_det_train_kcp_reference",
    ):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "make_inputs"):
            return mod
    return importlib.import_module("gdn_inputs")


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
    if failures:
        report["failing"].append("naive_tier")


def _record_verdicts(report: dict | None, record_property) -> None:
    """The test process records rank 0's verdicts (pytest ``record_property``, read by the conftest's ``TIER``
    lines)."""
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
# rows (C1-C9 correctness rows, A1-A4 accuracy rows, the no-leak rows, the long-chain rows; the int4 stage = the
# CP16 rows: the C6 zigzag row, the A1 contiguous row at the (2, 6) head pair and the long-chain two-document row)


@dataclasses.dataclass(frozen=True)
class Row:
    name: str
    lengths: tuple[int, ...]
    heads: tuple[int, int]  # (H_qk, HV)
    W: int
    layout: str  # contiguous | zigzag | zigzag_per_document | ragged:<T_r list> | explicit:<name>
    regime: str = "normal"
    leak: bool = False  # run the zero-gradient leak / independence checks
    stage: str = "int1"  # int1 (CP <= 4, small), int3 (CP8 / large), int4 (CP16)
    long_chains: bool = False  # long owner chains (checked on the CPU: test_long_chain_rows_have_long_owner_chains)


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
    """The row's partition (``cake_gdn_nondet_train_kcp_partition``) and its global ``cu_seqlens``."""
    fn = plan.cake_gdn_nondet_train_kcp_partition
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


def _local_to_global(part, cu: list[int], rank: int) -> torch.Tensor:
    """Global token index of every local token of ``rank`` (fragment order), CPU int64."""
    pieces = [
        torch.arange(cu[f.doc] + f.off, cu[f.doc] + f.off + f.length)
        for f in part.fragments[rank]
        if f.length
    ]
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.int64)


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


def _slurm_row(row, slurm, backend: str, body, record_property) -> None:
    """One row under Slurm: sub-group of the row's ranks; the verdict is WORLD-wide, so every rank (member or not)
    passes or fails the same row and ``-x`` and the session teardown keep the collectives aligned.  Rank 0 records
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
            # a body that raises leaves the row's collectives on the other ranks unmatched; the step then ends at
            # the WORLD timeout before pytest's summary, so the cause is printed here
            traceback.print_exc()
            sys.stderr.flush()
            raise
    ok = report is None or bool(report["pass"])
    if slurm.world > 1:
        verdicts: list = [None] * slurm.world
        dist.all_gather_object(verdicts, ok)
        ok = all(verdicts)
    if slurm.rank >= W:
        if not ok:
            pytest.fail(f"rank {slurm.rank}: row failed on its {W} member ranks")
        pytest.skip(f"rank {slurm.rank} is not part of the row's {W}-rank group")
    _record_verdicts(report, record_property)
    assert ok, json.dumps(
        {k: v for k, v in report.items() if k not in ("info_by_rank", "naive_tier")},
        default=str,
    )[:6000]


# ----------------------------------------------------------------------------------------------------------------
# C7: the convolution halo row.  The user calls ``cake_causal_conv1d_det_kcp(x_local, weight, ctx)`` with
# THIS package's context: the conv package reads the halo plan (``ctx.halo``, ``ctx.halo_tables``), the local
# ``cu_seqlens`` and the process group by duck typing.  The row's partition: document 0 = [1, 2, 300] on ranks 0,
# 1 and 3 (the three history rows of rank 3's first outputs come from TWO predecessor ranks), rank 2 holds no token
# (an empty predecessor), document 1 = [200] on rank 3 (a document start inside the continuing rank); C = 10,240.
# The bitwise checks of this row are the convolution package's own (its single-rank form and its repeat).

CONV_C = 10240
CONV_WIDTH = 4
# the seed of the C7 operands: draw 0; draw j at draw_seed(CONV_SEED, j)
CONV_SEED = 43567


def _c7_fragments() -> list[list[tuple[int, int, int]]]:
    return [[(0, 0, 1)], [(0, 1, 2)], [], [(0, 3, 300), (1, 0, 200)]]


EXPLICIT["c7"] = (_c7_fragments, (303, 200))
C7_ROW = Row("C7", (303, 200), (2, 6), 4, "explicit:c7", stage="conv")


def _to_global(part, rank_tensors: list[torch.Tensor], cu: list[int]) -> torch.Tensor:
    """The ranks' local tensors (fragment order) reassembled to the global token order (on the first's device)."""
    ref = rank_tensors[0]
    out = torch.zeros(
        (cu[-1],) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device
    )
    for r, t in enumerate(rank_tensors):
        out[_local_to_global(part, cu, r).to(ref.device)] = t.to(ref.device)
    return out


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
        cand = _to_global(part, [r_[n].to(device) for r_ in got], cu)
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
    """C7 on one rank over 32 draws: on every draw ``y`` bitwise the conv package's single-rank conv of the whole
    pack on the rank's tokens, its repeat bitwise, the shapes, the halo-row gradients delivered to the predecessor
    ranks (rank 0 / 1: 1 / 2 tokens of document 0, non-zero) and the empty rank; on rank 0 ``y`` / ``dx`` / every
    rank's ``dweight`` partial judged against the FP64 oracle beside the naive BF16 tier
    (:func:`_judge_conv_draw`)."""
    conv = _conv_mod()
    conv_kcp = _conv_mod("kcp")

    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
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
        "pass": all(flags.values()),
        "failing": [k for k, ok in flags.items() if not ok],
        "rank": rank,
        "world": W,
        "backend": "conv",
        "mode": "conv",
        "package": PACKAGE,
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
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
    """C7: ``cake_causal_conv1d_det_kcp`` through this package's context (duck typing): ``y`` / ``dx`` / ``dweight``
    judged against FP64 (naive BF16 tier, 32 draws), ``y`` bitwise the conv package's single-rank conv, history rows
    from two predecessor ranks, an empty predecessor and a document start.  Runs under Slurm with >= 4 ranks."""
    if slurm is None:
        pytest.skip("the conv halo row runs under Slurm (4 ranks)")
    conv = _conv_mod()
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(C7_ROW, slurm, "conv", _run_rank_conv, record_property)


# ----------------------------------------------------------------------------------------------------------------
# Whole-layer checkpoint rows (issue #7818): KCP convolution -> packed q | k | v views -> caller L2 norm of q, k ->
# cake_gdn_nondet_kcp, one checkpointed region per layer with one CakeGDNNondetKcpRecomputeHandle, over 32 draws of
# the operands.  Gated on every draw: the GDN part of the recompute exchanges nothing (with a convolution that takes
# the handle the whole recompute exchanges nothing; before that, exactly the convolution's own forward collectives
# remain, the whole-layer zero is reported as pending); the original forward and the backward exchange as in the
# plain checkpoint; the backward of every form issues exactly ``LAYER_BWD_COLLECTIVES``; no parameter reduction.
# Judged on rank 0: out / dx / dweight / dg / dbeta of the direct pass, the plain checkpoint and the handle checkpoint,
# each against the FP64 layer oracle beside its naive BF16 tier (:func:`_layer_reference`).  Runs under Slurm.  The L8
# rows keep the (16, 48) heads (C = 10240 convolution channels) at about 1k tokens (the 32 draws of the FP64 oracle);
# L8r keeps the empty rank, the 1-token rank and the document chains of its ragged layout.

LAYER_ROWS = [
    Row("L4", (303, 200), (2, 6), 4, "explicit:c7", stage="layer"),
    Row(
        "L8",
        (512, 320, 165, 33),
        (16, 48),
        8,
        "zigzag_per_document",
        stage="layer",
    ),
    Row(
        "L8r",
        (512, 320, 165, 33),
        (16, 48),
        8,
        "ragged:246,0,1,148,141,148,118,228",
        stage="layer",
    ),
]
# the seed of the layer's convolution operands (draw j at draw_seed(LAYER_SEED, j)); its GDN operands come from the
# seeded generator at draw_seed(SEED, j)
LAYER_SEED = 78180
# (run mode, judged form): the direct pass, the plain checkpoint, the checkpoint with the recompute handle
LAYER_ARMS = (("direct", "direct"), ("plain", "plain_ckpt"), ("handle", "handle"))


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """The caller's per-head L2 norm of ``[T, H, 128]`` BF16 (FP32 math, one rounding): deterministic, so the
    checkpoint recompute reproduces it bit for bit."""
    xf = x.float()
    return (xf / torch.sqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


def _l2_normalize_exact(x: torch.Tensor) -> torch.Tensor:
    """The same L2 norm without the FP32 math and the rounding, in the dtype of ``x``: FP64 for the oracle's
    derivative, BF16 for its naive tier."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _layer_reference(
    row: Row,
    part,
    cu: list[int],
    W: int,
    device,
    x_all,
    w_all,
    conv,
    seed: int,
    *,
    l2n: bool = False,
) -> tuple[dict, dict]:
    """FP64 oracle of the whole layer and its naive BF16 tier, on the layer's own BF16 operands: the conv package's
    single-rank convolution of the whole pack (bitwise the KCP convolution on every rank's tokens, the C7 row's gate)
    feeds the exact GDN of the whole pack (the GDN operands of the generator at ``seed``), whose FP64 gradients are
    carried back to the convolution output and through the FP64 convolution + SiLU.  Without ``l2n`` the GDN reads
    the caller's L2 norm of the q / k views (the layer's own BF16 rows) and ``dq`` / ``dk`` go back through that norm
    in FP64; with ``l2n`` (the in-kernel norm) it reads the raw views: ``gdn_exact_l2n_with_grads`` normalizes them in
    FP64 and returns the gradients of the raw rows.  The naive tier runs the same chain on the same operands in
    PyTorch BF16 (``gdn_naive_with_grads`` with the norm's autograd on the BF16 rows, or ``gdn_naive_l2n_with_grads``;
    :func:`_conv_naive`).  ``out`` / ``dx`` / ``dg`` / ``dbeta`` of the whole pack; ``dw`` the list of every rank's
    partial (the weight leaf is per rank)."""
    ref_mod = _reference()
    H_qk, HV = row.heads
    total = cu[-1]
    x_dev, w_dev = x_all.to(device), w_all.to(device)
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    with torch.no_grad():
        y = conv.cake_causal_conv1d_det(x_dev, w_dev, cu_dev)
    qkv = y.view(total, 2 * H_qk + HV, 128)
    if l2n:  # the raw views: the oracle's norm is FP64, the naive tier's BF16
        q, k = qkv[:, :H_qk], qkv[:, H_qk : 2 * H_qk]
        exact, naive_fn = (
            ref_mod.gdn_exact_l2n_with_grads,
            ref_mod.gdn_naive_l2n_with_grads,
        )
    else:
        q = _l2_normalize(qkv[:, :H_qk])
        k = _l2_normalize(qkv[:, H_qk : 2 * H_qk])
        exact, naive_fn = ref_mod.gdn_exact_with_grads, ref_mod.gdn_naive_with_grads
    v = qkv[:, 2 * H_qk :].contiguous()
    glob = _global_inputs(row, cu, device, seed=seed)
    args = (
        q,
        k,
        v,
        glob["g"],
        glob["beta"],
        glob["dout"],
        torch.tensor(cu, dtype=torch.int64),
    )
    gdn = exact(*args, scale=SCALE)
    gdn_nv = naive_fn(*args, scale=SCALE)
    del glob, args, q, k, v
    # dq / dk / dv onto the convolution output: FP64 (oracle) and BF16 (naive)
    dy = {}
    for tier, dtype, grads in (("oracle", F64, gdn), ("naive", torch.bfloat16, gdn_nv)):
        if l2n:  # dq / dk are already the gradients of the raw rows
            dy[tier] = torch.cat(
                [grads[n].to(dtype).reshape(total, -1) for n in ("dq", "dk", "dv")],
                dim=1,
            )
            continue
        yl = y.detach().to(dtype).requires_grad_(True)
        y3 = yl.view(total, 2 * H_qk + HV, 128)
        outs = (
            _l2_normalize_exact(y3[:, :H_qk]),
            _l2_normalize_exact(y3[:, H_qk : 2 * H_qk]),
            y3[:, 2 * H_qk :],
        )
        (dy[tier],) = torch.autograd.grad(
            outs, (yl,), tuple(grads[n].to(dtype) for n in ("dq", "dk", "dv"))
        )
        del yl, y3, outs
    _, dx64, dpre, taps = _conv_reference(x_dev, w_dev, cu, dy["oracle"])
    _, dx_nv, dpre_nv, taps_nv = _conv_naive(x_dev, w_dev, cu, dy["naive"])
    sels = [_local_to_global(part, cu, r).to(device) for r in range(W)]
    refs = {
        "out": gdn["out"].to(F64),
        "dx": dx64,
        "dg": gdn["dg"].to(F64),
        "dbeta": gdn["dbeta"].to(F64),
        "dw": _dw_partials(dpre, taps, sels),
    }
    naive = {
        "out": gdn_nv["out"],
        "dx": dx_nv,
        "dg": gdn_nv["dg"],
        "dbeta": gdn_nv["dbeta"],
        "dw": _dw_partials(dpre_nv, taps_nv, sels, torch.bfloat16),
    }
    return refs, naive


def _judge_layer_draw(
    judge: TierJudge,
    failures: list[str],
    j: int,
    row: Row,
    part,
    cu: list[int],
    W: int,
    device,
    x_all: torch.Tensor,
    w_all: torch.Tensor,
    conv,
    seed: int,
    got: list[dict],
    *,
    l2n: bool = False,
) -> None:
    """Rank 0, draw ``j`` of a layer row: per form (``direct``, ``plain_ckpt``, ``handle``) ``out`` / ``dx`` / ``dg``
    / ``dbeta`` reassembled to the global order and every rank's ``dw`` partial (``<form>.dw.r<rank>``) against the
    FP64 layer oracle beside its naive BF16 tier (``l2n``: the oracle of the in-kernel norm)."""
    refs, naive = _layer_reference(
        row, part, cu, W, device, x_all, w_all, conv, seed, l2n=l2n
    )
    for _mode, arm in LAYER_ARMS:
        for n in ("out", "dx", "dg", "dbeta"):
            cand = _to_global(part, [r_[arm][n].to(device) for r_ in got], cu)
            _judge_add(judge, failures, f"{arm}.{n}", j, cand, refs[n], naive[n])
        for r in range(W):
            _judge_add(
                judge,
                failures,
                f"{arm}.dw.r{r}",
                j,
                got[r][arm]["dw"].to(device),
                refs["dw"][r],
                naive["dw"][r],
            )


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
    """One whole-layer row over 32 draws (``l2n``: #7817, the in-kernel L2 norm on the raw convolution output views
    instead of the caller's norm): the collective counts of the three forms on every draw (the backward's against
    the fixed list ``LAYER_BWD_COLLECTIVES``), and on rank 0 their outputs judged against the FP64 layer oracle
    (:func:`_judge_layer_draw`)."""
    conv = _conv_mod()
    conv_kcp = _conv_mod("kcp")

    H_qk, HV = row.heads
    C = (2 * H_qk + HV) * 128
    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
    l2g = _local_to_global(part, cu, rank)
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
        return kcp.cake_gdn_nondet_kcp(
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

    def run(mode: str, x_local, w_all, g, beta, dout):
        leaves = [
            x_local.detach().clone().requires_grad_(True),
            w_all.to(device).requires_grad_(True),
            g.detach().clone().requires_grad_(True),
            beta.detach().clone().requires_grad_(True),
        ]
        hd = kcp.CakeGDNNondetKcpRecomputeHandle() if mode == "handle" else None
        from torch.utils.checkpoint import checkpoint

        with _NamedCounter() as cc:
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
        tensors = [out.detach()] + [lf.grad for lf in leaves]
        return dict(zip(LAYER_TENSORS, tensors, strict=True)), {
            **cc.record(),
            "handle_bytes": hd.nbytes() if hd is not None else 0,
            "handle_keys": sorted(hd.entries) if hd is not None else [],
        }

    flags: dict[str, bool] = {}
    first: dict[str, object] = {}
    conv_fwd: dict | None = None
    judge, failures = (TierJudge(), []) if rank == 0 else (None, [])
    t0 = time.time()
    for j in range(DRAWS):
        x_all, w_all, _ = _conv_operands(draw_seed(LAYER_SEED, j), cu[-1], C)
        x_local = x_all[l2g].to(device)
        gdn_seed = draw_seed(int(_inputs().SEED), j)
        ops = _local_inputs(row, part, cu, rank, device, seed=gdn_seed)
        if conv_fwd is None:
            # the convolution's own forward collectives (its halo exchange), for the pending whole-layer count
            with CollectiveCounter() as cc_conv:
                conv_kcp.cake_causal_conv1d_det_kcp(x_local, w_all.to(device), ctx)
            conv_fwd = cc_conv.record()
        res, rec = {}, {}
        for mode, _arm in LAYER_ARMS:
            res[mode], rec[mode] = run(
                mode, x_local, w_all, ops["g"], ops["beta"], ops["dout"]
            )
        rec_d, rec_p, rec_h = rec["direct"], rec["plain"], rec["handle"]
        rc_calls = rec_h["calls"].get("recompute", 0)
        rc_waits = rec_h["waits"].get("recompute", 0)
        fl = {
            "layer_orig_bwd_comm_unchanged": all(
                rec_h[t].get(ph, 0) == rec_p[t].get(ph, 0)
                for t in ("calls", "waits")
                for ph in ("orig", "bwd")
            ),
            "layer_no_param_reduction": _no_param_reduction(rec_d, rec_p, rec_h),
            "layer_bwd_collectives_fixed": all(
                r_["named"].get("bwd", {}) == LAYER_BWD_COLLECTIVES
                for r_ in (rec_d, rec_p, rec_h)
            ),
        }
        if conv_takes_handle:
            fl["layer_recompute_no_comm"] = rc_calls == 0 and rc_waits == 0
        else:
            # only the convolution's own halo exchange is left in the recompute: the GDN part exchanges nothing
            fl["layer_gdn_recompute_no_comm"] = rc_calls == conv_fwd["calls"].get(
                "orig", 0
            ) and rc_waits == conv_fwd["waits"].get("orig", 0)
        for key, ok in fl.items():
            flags[key] = flags.get(key, True) and bool(ok)
        if j == 0:
            first = {
                "direct": rec_d,
                "plain": rec_p,
                "handle": rec_h,
                "plain_recompute_calls": rec_p["calls"].get("recompute", 0),
                "handle_recompute_calls": rc_calls,
                "handle_recompute_waits": rc_waits,
            }
        got = _gather(
            {
                arm: {n: t.detach().cpu() for n, t in res[mode].items()}
                for mode, arm in LAYER_ARMS
            },
            W,
            group,
        )
        del res, ops, x_local
        if judge is not None:
            _judge_layer_draw(
                judge,
                failures,
                j,
                row,
                part,
                cu,
                W,
                device,
                x_all,
                w_all,
                conv,
                gdn_seed,
                got,
                l2n=l2n,
            )
        del got
    host = _host_map(group, W)
    flags["host_map"] = host["ok"]
    report = {
        "row": dataclasses.asdict(row),
        "pass": True,
        "failing": [],
        "rank": rank,
        "world": W,
        "backend": backend,
        "mode": "layer-l2n" if l2n else "layer",
        "package": PACKAGE,
        "host": socket.gethostname(),
        "hosts": host,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flags": {},
        "gated": [],
        "reported": []
        if conv_takes_handle
        else ["layer_recompute_no_comm (pending: conv recompute_handle)"],
        "info_by_rank": [
            {
                "rank": rank,
                "T_r": int(l2g.numel()),
                "conv_takes_handle": conv_takes_handle,
                "conv_fwd_collectives": conv_fwd,
                "N_out": int(ctx.N_out),
                **first,
            }
        ],
        "seconds": None,
        "C": C,
    }
    _fold(report, [flags])
    _judge_report(report, judge, failures)
    report["seconds"] = round(time.time() - t0, 2)
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
def test_layer_checkpoint_rows(row: Row, slurm, record_property) -> None:
    """#7818 whole-layer rows: KCP convolution + L2 norm + cake_gdn_nondet_kcp in one checkpointed
    region with one recompute handle (see the block comment above).  Runs under Slurm with >= the row's ranks."""
    if slurm is None:
        pytest.skip("the layer rows run under Slurm")
    conv = _conv_mod()
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(row, slurm, _backend_name(), _run_rank_layer, record_property)


L2N_SKIP = (
    "the in-kernel Q/K L2 norm records (fwd_prep_l2n, bwd_dqkg_l2n, qk_apply) are not in the shipped registry "
    "yet: use_qk_l2norm_in_kernel cannot run on the kernel backend"
)


def _l2n_shipped() -> bool:
    shipped = getattr(kcp, "l2n_records_shipped", None)
    return bool(shipped()) if callable(shipped) else False


def _skip_unless_l2n(backend: str) -> None:
    if backend == "kernels" and not _l2n_shipped():
        pytest.skip(L2N_SKIP)


@pytest.mark.parametrize("row", LAYER_ROWS, ids=_row_id)
def test_l2n_layer_checkpoint_rows(row: Row, slurm, record_property) -> None:
    """#7817 whole-layer rows: KCP convolution + ``cake_gdn_nondet_kcp(use_qk_l2norm_in_kernel=True)``
    on the RAW convolution output views in one checkpointed region (the B3 boundary), with and without a recompute
    handle, each form judged against the FP64 layer oracle (naive BF16 tier, 32 draws)."""
    if slurm is None:
        pytest.skip("the layer rows run under Slurm")
    conv = _conv_mod()
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    backend = _backend_name()
    _skip_unless_l2n(backend)
    _slurm_row(
        row,
        slurm,
        backend,
        lambda *a: _run_rank_layer(*a, l2n=True),
        record_property,
    )
