"""
Copyright (c) 2026 by FlashInfer team.
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
(``cake-pplx-causal-conv1d``), moved here unchanged from ``kernels/gdn/tests/experimental/
test_cake_gdn_nondet_train_kcp_dist.py`` (the in-family rows stay there).

* ``test_conv_halo_row`` (C7): the conv package's ``cake_causal_conv1d_det_kcp`` with THIS package's context;
* ``test_layer_checkpoint_rows`` (L4 / L8 / L8r): KCP convolution + caller L2 norm + ``cake_gdn_nondet_kcp`` in one
  checkpointed region with one ``CakeGDNNondetKcpRecomputeHandle`` (the recompute exchanges nothing; out / dx /
  dweight / dg / dbeta within the budget constants of the plain checkpoint and the direct pass and within the
  correctness clause of the FP64 layer oracle; the backward's collectives are the deterministic layer's);
* ``test_l2n_layer_checkpoint_rows``: the same layer with ``use_qk_l2norm_in_kernel=True`` on the raw convolution
  output views.

All of them run under Slurm (one process per task: ``SLURM_PROCID`` / ``SLURM_NTASKS``; the WORLD group is initialised
once per process and every row runs on a sub-group of its ``W`` lowest ranks).  Environment: ``KCP_TEST_BACKEND``
(``kernels`` | ``fake``), ``KCP_TEST_RESULT_DIR`` (one JSON per row and rank), ``KCP_TEST_PORT_DIR`` (the Slurm
rendezvous port file).  The suite imports this package, the deterministic GDN package and the causal-convolution
package through ``tests/cake/test_helpers/cake_package.py``; a missing distribution fails the row, never skips.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import importlib
import inspect
import json
import math
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


def _det_mod(sub: str | None = None):
    """A module of the deterministic package through the test-helper registry (its entries and its ``LAST_CALL`` for
    the collective comparison)."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_det_train_kcp", sub)


def _conv_mod(sub: str | None = None):
    """The deterministic causal-convolution package (or its ``kcp`` module) through the test-helper registry: the
    entries ``cake_causal_conv1d_det`` / ``is_cake_causal_conv1d_det_train_supported`` and
    ``kcp.cake_causal_conv1d_det_kcp`` that its facades forward to."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_causal_conv1d_det_train", sub)


kcp = _mod("cake_kcp")
plan = _mod("cake_kcp_plan")
PACKAGE_DIR = Path(kcp.__file__).resolve().parent
TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta")
LAYER_TENSORS = ("out", "dx", "dw", "dg", "dbeta")
ATOL = RTOL = 1e-2
WORLD_TIMEOUT_S = 300  # NCCL / rendezvous timeout of the WORLD group (printed at init)
SCALE = kcp.SCALE
F64 = torch.float64
# (b) collectives that would reduce parameters across the group: never issued by the route
PARAM_REDUCTIONS = ("all_reduce", "reduce_scatter_tensor")


def _reference():
    """The FP64 reference: the FlashInfer test helper (read only), else the session harness on PYTHONPATH."""
    try:
        from tests.experimental.test_helpers import (
            cake_gdn_det_train_kcp_reference as ref,
        )  # type: ignore[attr-defined]

        return ref
    except ImportError:
        return importlib.import_module("gdn_reference")


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
# the distributed judge: every rank compares its local tokens, the partial sums combine over the group


def _partial(ref_mod, cand: torch.Tensor, ref: torch.Tensor) -> dict:
    """This rank's share of one comparison: the reference module's error metrics and elementwise count on the local
    elements (FP64), as sums that combine across ranks."""
    m = ref_mod.error_metrics(cand, ref)
    e = ref_mod.elementwise_mismatch(cand, ref, ATOL, RTOL)
    n = int(m["numel"])
    return {
        "n": n,
        "sq_err": float(m["rms"]) ** 2 * n,
        "sq_ref": float(m["ref_rms"]) ** 2 * n,
        "max_abs": float(m["max_abs"]),
        "ref_max_abs": float(m["ref_max_abs"]),
        "nonfinite": int(m["nonfinite"]),
        "mismatch": int(e["mismatch"]),
        "worst": m["worst"],
    }


def _combine(parts: list) -> dict:
    """The metrics of the whole tensor from the ranks' partials (``error_metrics`` keys; ``worst`` carries its rank)."""
    ps = [p for p in parts if p is not None]
    n = sum(p["n"] for p in ps)
    sq_err = sum(p["sq_err"] for p in ps)
    sq_ref = sum(p["sq_ref"] for p in ps)
    rms = math.sqrt(sq_err / n) if n else 0.0
    ref_rms = math.sqrt(sq_ref / n) if n else 0.0
    worst = None
    for r, p in enumerate(parts):
        w = None if p is None else p["worst"]
        if w is None:
            continue
        if worst is None or math.isnan(w["err"]) or w["err"] > worst["err"]:
            worst = {**w, "rank": r}
    return {
        "rel_rms": rms / ref_rms if ref_rms > 0 else (math.inf if rms > 0 else 0.0),
        "max_abs": max((p["max_abs"] for p in ps), default=0.0),
        "rms": rms,
        "ref_rms": ref_rms,
        "ref_max_abs": max((p["ref_max_abs"] for p in ps), default=0.0),
        "numel": n,
        "nonfinite": sum(p["nonfinite"] for p in ps),
        "mismatch": sum(p["mismatch"] for p in ps),
        "worst": worst,
    }


def _verdict(row: Row, name: str, met: dict, budget, ref_mod) -> dict:
    """The verdict of one combined comparison.  Correctness rows (regime ``normal``): the elementwise clause, zero
    elements beyond ``ATOL + RTOL |ref|``.  Accuracy rows: the row's budget cell (``_budget_cell``): rel-RMS within the
    budget and max-abs within the budget plus one storage ulp of |ref| at the worst element (the reference module's
    ``judge_tensor``); the elementwise count is recorded.  Finiteness gates every row; an accuracy row without a
    budget cell cannot pass.  The same rule judges a comparison against FP64 and one between two arms (the partner arm
    in the place of the reference)."""
    v = {
        k: met[k]
        for k in ("rel_rms", "max_abs", "ref_rms", "numel", "nonfinite", "mismatch")
    }
    v["worst"] = met["worst"]
    if met["numel"] == 0:
        v["verdict"] = "empty"
        ok = True
    elif row.regime == "normal":
        v["verdict"] = "elementwise"
        ok = met["nonfinite"] == 0 and met["mismatch"] == 0
    elif budget is None or name not in budget[0]["tensors"]:
        v["verdict"] = "budget"
        v["budget"] = None
        ok = False
    else:
        cell, cp = budget
        b = cell["tensors"][name]
        j = ref_mod.judge_tensor(
            met, b["budget_rel_rms"], b["budget_max_abs"], b["storage"]
        )
        j["cp_of_cell"] = cp
        v["verdict"] = "budget"
        v["budget"] = j
        ok = met["nonfinite"] == 0 and bool(j["pass_u1"])
    v["ok"] = bool(ok)
    return v


def _rel_rms_verdict(met: dict, max_rel_rms: float) -> dict:
    """Finite and rel-RMS <= ``max_rel_rms``; the elementwise count (``ATOL + RTOL |ref|``) and the worst element are
    recorded, not judged."""
    v = {
        k: met[k]
        for k in ("rel_rms", "max_abs", "ref_rms", "numel", "nonfinite", "mismatch")
    }
    v["worst"] = met["worst"]
    v["verdict"] = "empty" if met["numel"] == 0 else "rel_rms"
    v["rel_rms_max"] = max_rel_rms
    v["ok"] = bool(
        met["numel"] == 0 or (met["nonfinite"] == 0 and met["rel_rms"] <= max_rel_rms)
    )
    return v


class _Judge:
    """FP64 and cross-arm comparisons of one row: each ``fp64`` / ``cross`` call adds this rank's partials at once
    (the arm's tensors can go), ``finish`` gathers them once over the group and every rank derives the same verdicts.
    Keys: ``fp64:<arm>:<tensor>`` and ``cross:<arm>:<tensor>``.  A comparison added with ``forced=True`` (an arm at
    the forced ``P = 2``) is judged by ``p2_budget`` (:func:`_p2_budget_cell`), every other one by ``budget``; each
    verdict names its cell (``"cell"``: ``"p2"`` or ``"g1"``)."""

    def __init__(
        self, row: Row, ref_mod, budget, fp64_rel_rms=None, p2_budget=None
    ) -> None:
        """``fp64_rel_rms``: ``{tensor: max}``, the ``fp64`` comparisons of these tensors judged by finiteness and
        rel-RMS alone (the elementwise count and the worst element recorded).  ``p2_budget``: the cell of the
        ``forced`` comparisons."""
        self.row, self.ref_mod, self.budget = row, ref_mod, budget
        self.p2_budget = p2_budget
        self.fp64_rel_rms = dict(fp64_rel_rms or {})
        self.parts: dict[str, dict] = {}
        self.gated: dict[str, bool] = {}
        self.forced: dict[str, bool] = {}

    def _add(
        self,
        key: str,
        cand: torch.Tensor,
        ref: torch.Tensor,
        gate: bool,
        forced: bool,
    ):
        self.parts[key] = _partial(self.ref_mod, cand, ref)
        self.gated[key] = bool(gate)
        self.forced[key] = bool(forced)

    def fp64(
        self,
        arm: str,
        res: dict,
        refs: dict,
        names=TENSORS,
        *,
        half=False,
        gate=True,
        forced=False,
    ):
        """``res`` against the FP64 reference; ``half``: half of an accumulated (two-backward) gradient, exact in
        every float format, against the FP64 gradient; ``forced``: an arm at the forced ``P = 2``."""
        for n in names:
            cand = res[n].detach().to(F64) * 0.5 if half else res[n]
            self._add(f"fp64:{arm}:{n}", cand, refs[n], gate, forced)

    def cross(
        self, arm: str, a: dict, b: dict, names=TENSORS, *, gate=True, forced=False
    ):
        """Arm ``a`` against its partner ``b`` within the same budget constants (``b`` in the reference's place;
        ``forced``: both arms at the forced ``P = 2``)."""
        for n in names:
            self._add(f"cross:{arm}:{n}", a[n], b[n], gate, forced)

    def finish(self, group, W: int) -> dict[str, dict]:
        mine = dict(self.parts)
        if W > 1:
            every: list = [None] * W
            dist.all_gather_object(every, (mine, self.gated, self.forced), group=group)
        else:
            every = [(mine, self.gated, self.forced)]
        keys: list[str] = []
        gated: dict[str, bool] = {}
        forced: dict[str, bool] = {}
        for p, gt, fc in every:
            for key in p:
                if key not in gated:
                    keys.append(key)
                gated[key] = gated.get(key, True) and gt[key]
                forced[key] = forced.get(key, False) or fc[key]
        out: dict[str, dict] = {}
        for key in keys:
            met = _combine([p.get(key) for p, _, _ in every])
            name = key.rsplit(":", 1)[1]
            budget = self.p2_budget if forced[key] else self.budget
            if key.startswith("fp64:") and name in self.fp64_rel_rms:
                v = _rel_rms_verdict(met, self.fp64_rel_rms[name])
            else:
                v = _verdict(self.row, name, met, budget, self.ref_mod)
            v["cell"] = "p2" if forced[key] else "g1"
            v["gated"] = gated[key]
            # a key missing on a rank: that rank skipped the comparison (fails the verdict)
            v["ranks"] = sum(1 for p, _, _ in every if key in p)
            if v["ranks"] != len(every):
                v["ok"] = False
            out[key] = v
        return out


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


def _partition_with(fn, row: Row):
    """``fn(cu, W, layout=... | fragments=...)``: this package's ``cake_gdn_nondet_train_kcp_partition`` or the deterministic route's
    ``cake_gdn_det_train_kcp_partition`` (the same partition of the row)."""
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
    verdicts: dict,
    *,
    backend: str = "kernels",
    kernel_only=(),
    info_only=(),
) -> None:
    """Fold the ranks' exact flags (all ranks must hold) and the judged verdicts into ``report["pass"]``."""
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
    report["judged"] = verdicts
    for key, v in verdicts.items():
        if v["gated"]:
            report["pass"] &= bool(v["ok"])
            if not v["ok"]:
                report["failing"].append(key)
        else:
            report["reported"].append(key)


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
            f"{torch.cuda.get_device_name(self.device)} flashinfer={_mod().__file__} MASTER={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
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


def _slurm_row(row, slurm, backend: str, body) -> None:
    """One row (or poison pair) under Slurm: sub-group of the row's ranks; the verdict is WORLD-wide, so every rank
    (member or not) passes or fails the same row and ``-x`` and the session teardown keep the collectives aligned."""
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
    assert ok, json.dumps(
        {k: v for k, v in report.items() if k != "info_by_rank"}, default=str
    )[:6000]


# ----------------------------------------------------------------------------------------------------------------
# C7: the convolution halo row.  The user calls FlashInfer's ``cake_causal_conv1d_det_kcp(x_local, weight, ctx)`` with
# THIS package's context: the conv package reads the halo plan (``ctx.halo``, ``ctx.halo_tables``), the local
# ``cu_seqlens`` and the process group by duck typing.  The row's partition: document 0 = [1, 2, 300] on ranks 0,
# 1 and 3 (the three history rows of rank 3's first outputs come from TWO predecessor ranks), rank 2 holds no token
# (an empty predecessor), document 1 = [200] on rank 3 (a document start inside the continuing rank); C = 10,240.
# The bitwise checks of this row are the convolution package's own (its single-rank form and its repeat).

CONV_C = 10240
CONV_WIDTH = 4
CONV_Y_ATOL, CONV_Y_RTOL = (
    1e-3,
    2.0**-7,
)  # one BF16 ulp, no element outside (the conv package's own gate)
CONV_DW_ATOL, CONV_DW_RTOL = 5e-2, 1e-4  # FP32 dweight against the FP64 oracle


def _c7_fragments() -> list[list[tuple[int, int, int]]]:
    return [[(0, 0, 1)], [(0, 1, 2)], [], [(0, 3, 300), (1, 0, 200)]]


EXPLICIT["c7"] = (_c7_fragments, (303, 200))
C7_ROW = Row("C7", (303, 200), (2, 6), 4, "explicit:c7", stage="conv")


def _conv_reference(
    x: torch.Tensor,
    w: torch.Tensor,
    cu: list[int],
    dy: torch.Tensor,
    *,
    round_dx: bool = True,
):
    """FP64 oracle of the unpartitioned width-4 depthwise causal convolution + SiLU with the conv package's KCP
    semantics (``weight[c, 3]`` multiplies the current token; zero history at a document start; the SiLU
    derivative on the unrounded preactivation): BF16 ``y``, ``dx`` (BF16, one rounding; FP64 without
    ``round_dx``), FP64 ``dpre`` and the taps for the per-rank ``dweight`` partials."""
    S, _ = x.shape
    xf, wd = x.double(), w.double()
    doc = torch.zeros(S, dtype=torch.int64, device=x.device)
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

    halo = CONV_WIDTH - 1
    pre = sum(wd[:, k][None, :] * shifted(xf, halo - k) for k in range(CONV_WIDTH))
    s = torch.sigmoid(pre)
    y = (pre * s).to(torch.bfloat16)
    dpre = dy.double() * s * (1 + pre * (1 - s))
    dx = sum(wd[:, k][None, :] * shifted_up(dpre, halo - k) for k in range(CONV_WIDTH))
    if round_dx:
        dx = dx.to(torch.bfloat16)
    taps = [shifted(xf, halo - k) for k in range(CONV_WIDTH)]
    return y, dx, dpre, taps


def _outside(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float
) -> dict:
    """Elements of ``actual`` beyond ``atol + rtol |expected|`` and the comparison's statistics."""
    a = actual.detach().reshape(-1).double()
    b = expected.detach().reshape(-1).double()
    if a.numel() == 0:
        return {"count_outside": 0, "max_abs_diff": 0.0, "finite": True, "n": 0}
    diff = (a - b).abs()
    return {
        "count_outside": int((diff > atol + rtol * b.abs()).sum().item()),
        "max_abs_diff": float(diff.max().item()),
        "finite": bool(torch.isfinite(a).all().item()),
        "n": int(a.numel()),
    }


def _run_rank_conv(
    rank: int,
    W: int,
    row: Row,
    group,
    device: torch.device,
    result_dir: str | None,
    backend: str,
) -> dict:
    """C7 on one rank: ``y`` / ``dx`` / ``dweight`` of the rank's tokens against the FP64 oracle of the whole pack,
    ``y`` bitwise equal to the conv package's single-rank conv of the whole pack on the rank's tokens, its repeat
    bitwise, and the halo-row gradients delivered to the predecessor ranks (rank 0 / 1: 1 / 2 tokens of document 0)."""
    conv = _conv_mod()
    conv_kcp = _conv_mod("kcp")

    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
    total = cu[-1]
    gen = torch.Generator(device="cpu").manual_seed(43567)
    x_all = (torch.randn(total, CONV_C, generator=gen) * 0.5).to(torch.bfloat16)
    w = (torch.randn(CONV_C, CONV_WIDTH, generator=gen) * 0.5).float()
    dy_all = (torch.randn(total, CONV_C, generator=gen) * 0.5).to(torch.bfloat16)
    l2g = _local_to_global(part, cu, rank)
    x_local = x_all[l2g].to(device).requires_grad_(True)
    w_dev = w.to(device).requires_grad_(True)
    dy_local = dy_all[l2g].to(device)
    t0 = time.time()
    y = conv_kcp.cake_causal_conv1d_det_kcp(x_local, w_dev, ctx)
    y.backward(dy_local)
    dx, dw = x_local.grad, w_dev.grad
    y2 = conv_kcp.cake_causal_conv1d_det_kcp(x_local.detach(), w_dev.detach(), ctx)
    torch.cuda.synchronize(device)
    # the oracle (FP64 on this rank's device, the whole pack) and the single-rank conv of the whole pack
    y_ref, dx_ref, dpre, taps = _conv_reference(
        x_all.to(device), w.to(device), cu, dy_all.to(device)
    )
    sel = l2g.to(device)
    dw_ref = torch.stack(
        [(dpre[sel] * tap[sel]).sum(0) for tap in taps], dim=1
    )  # this rank's partial of dweight (the framework reduces the ranks)
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    y_single = conv.cake_causal_conv1d_det(x_all.to(device), w.to(device), cu_dev)
    flags: dict[str, bool] = {}
    metrics: dict[str, dict] = {}
    metrics["y"] = _outside(y, y_ref[sel], atol=CONV_Y_ATOL, rtol=CONV_Y_RTOL)
    metrics["dx"] = _outside(dx, dx_ref[sel], atol=CONV_Y_ATOL, rtol=CONV_Y_RTOL)
    metrics["dw"] = _outside(dw, dw_ref, atol=CONV_DW_ATOL, rtol=CONV_DW_RTOL)
    flags["y_within"] = metrics["y"]["count_outside"] == 0 and metrics["y"]["finite"]
    flags["dx_within"] = metrics["dx"]["count_outside"] == 0 and metrics["dx"]["finite"]
    flags["dw_within"] = metrics["dw"]["count_outside"] == 0 and metrics["dw"]["finite"]
    flags["y_eq_single_rank"] = torch.equal(y, y_single[sel])
    flags["repeat"] = torch.equal(y, y2)
    flags["shapes"] = tuple(y.shape) == tuple(x_local.shape) and tuple(
        dx.shape
    ) == tuple(x_local.shape)
    # the halo rows: the predecessor ranks' tokens feed rank 3's first outputs, so their dx must carry that term
    # (non-zero), and a rank without tokens has empty y / dx
    if rank in (0, 1):
        flags["halo_grad_nonzero"] = bool((dx.float().abs() > 0).any().item())
    if rank == 2:
        flags["empty_rank"] = y.numel() == 0 and dx.numel() == 0
    report = {
        "row": dataclasses.asdict(row),
        "pass": all(flags.values()),
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
        "metrics": metrics,
        "info_by_rank": [
            {
                "rank": rank,
                "T_r": int(x_local.shape[0]),
                "l2g_first": int(l2g[0]) if len(l2g) else None,
            }
        ],
        "seconds": round(time.time() - t0, 2),
        "C": CONV_C,
    }
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, f"{row.name}-conv-rank{rank}.json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def test_conv_halo_row(slurm) -> None:
    """C7: FlashInfer's ``cake_causal_conv1d_det_kcp`` through this package's context (duck typing): ``y`` / ``dx`` /
    ``dweight`` vs FP64 on every rank, ``y`` bitwise the conv package's single-rank conv, history rows from two
    predecessor ranks, an empty predecessor and a document start.  Runs under Slurm with >= 4 ranks."""
    if slurm is None:
        pytest.skip("the conv halo row runs under Slurm (4 ranks)")
    conv = _conv_mod()
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(C7_ROW, slurm, "conv", _run_rank_conv)


# ----------------------------------------------------------------------------------------------------------------
# Whole-layer checkpoint rows (issue #7818): KCP convolution -> packed q | k | v views -> caller L2 norm of q, k ->
# cake_gdn_nondet_kcp, one checkpointed region per layer with one CakeGDNNondetKcpRecomputeHandle.  Gated: the GDN
# part of the recompute exchanges nothing (with a convolution that takes the handle the whole recompute exchanges
# nothing; before that, exactly the convolution's own forward collectives remain, the whole-layer zero is reported as
# pending); the original forward and the backward exchange as in the plain checkpoint; the direct backward's
# collectives by name are the deterministic layer's (b1, b2 and the convolution's halo), no parameter reduction;
# out / dx / dweight / dg / dbeta of the handle checkpoint within the budget constants of the plain checkpoint and
# of the direct pass, and every form within the correctness clause of the FP64 layer oracle.  Runs under Slurm.

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


def _l2_normalize_f64(x: torch.Tensor) -> torch.Tensor:
    """The same L2 norm in FP64 without the rounding (the oracle's derivative)."""
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _layer_reference(
    row: Row, part, cu: list[int], rank: int, device, x_all, w_all, conv
) -> dict[str, torch.Tensor]:
    """FP64 oracle of the whole layer on this rank's tokens: the exact GDN of the layer's own BF16 operands (the conv
    package's single-rank convolution of the whole pack -- bitwise the KCP convolution on every rank's tokens, the C7
    row's gate -- through the caller's L2 norm), its FP64 gradients carried back through the FP64 L2 norm and the FP64
    convolution + SiLU; ``dw`` is this rank's partial (the weight leaf is per rank)."""
    ref_mod = _reference()
    H_qk, HV = row.heads
    total = cu[-1]
    sel = _local_to_global(part, cu, rank).to(device)
    x_dev, w_dev = x_all.to(device), w_all.to(device)
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    with torch.no_grad():
        y = conv.cake_causal_conv1d_det(x_dev, w_dev, cu_dev)
    qkv = y.view(total, 2 * H_qk + HV, 128)
    q = _l2_normalize(qkv[:, :H_qk])
    k = _l2_normalize(qkv[:, H_qk : 2 * H_qk])
    v = qkv[:, 2 * H_qk :].contiguous()
    glob = _global_inputs(row, cu, device)
    gdn = ref_mod.gdn_exact_with_grads(
        q,
        k,
        v,
        glob["g"],
        glob["beta"],
        glob["dout"],
        torch.tensor(cu, dtype=torch.int64),
        scale=SCALE,
    )
    del glob, q, k, v
    y64 = y.detach().to(F64).requires_grad_(True)
    y3 = y64.view(total, 2 * H_qk + HV, 128)
    outs = (
        _l2_normalize_f64(y3[:, :H_qk]),
        _l2_normalize_f64(y3[:, H_qk : 2 * H_qk]),
        y3[:, 2 * H_qk :],
    )
    (dy64,) = torch.autograd.grad(
        outs, (y64,), (gdn["dq"].to(F64), gdn["dk"].to(F64), gdn["dv"].to(F64))
    )
    del outs, y3, y64
    _, dx64, dpre, taps = _conv_reference(x_dev, w_dev, cu, dy64, round_dx=False)
    dw = torch.stack([(dpre[sel] * tap[sel]).sum(0) for tap in taps], dim=1)
    out = {
        "out": gdn["out"][sel].to(F64),
        "dx": dx64[sel],
        "dw": dw,
        "dg": gdn["dg"][sel].to(F64),
        "dbeta": gdn["dbeta"][sel].to(F64),
    }
    del gdn, dx64, dpre, taps, dy64
    return out


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
    conv = _conv_mod()
    conv_kcp = _conv_mod("kcp")

    H_qk, HV = row.heads
    C = (2 * H_qk + HV) * 128
    part, cu = _partition(row)
    ctx = plan.cake_gdn_nondet_train_kcp_build_context(part, group, rank, device)
    gen = torch.Generator(device="cpu").manual_seed(78180)
    x_all = (torch.randn(cu[-1], C, generator=gen) * 0.5).to(torch.bfloat16)
    w_all = (torch.randn(C, CONV_WIDTH, generator=gen) * 0.5).float()
    l2g = _local_to_global(part, cu, rank)
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

    def leaves_of():
        return [
            x_local.detach().clone().requires_grad_(True),
            w_all.to(device).requires_grad_(True),
            g.detach().clone().requires_grad_(True),
            beta.detach().clone().requires_grad_(True),
        ]

    def run(mode: str):
        leaves = leaves_of()
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

    # the convolution's own forward collectives (its halo exchange), for the pending whole-layer count
    with CollectiveCounter() as cc_conv:
        conv_kcp.cake_causal_conv1d_det_kcp(x_local, w_all.to(device), ctx)
    conv_fwd = cc_conv.record()
    t0 = time.time()
    res_d, rec_d = run("direct")
    res_p, rec_p = run("plain")
    res_h, rec_h = run("handle")
    # (b) the deterministic layer (its own context of the same partition; the convolution reads it by duck typing):
    # the direct backward's collectives by name
    det_kcp = _det_mod("cake_kcp")
    det_plan = _det_mod("cake_kcp_plan")
    det_part, _ = _partition_with(det_plan.cake_gdn_det_train_kcp_partition, row)
    det_ctx = det_plan.cake_gdn_det_train_kcp_build_context(
        det_part, group, rank, device
    )
    det_leaves = leaves_of()
    with _NamedCounter() as cc_det:
        y_d = conv_kcp.cake_causal_conv1d_det_kcp(det_leaves[0], det_leaves[1], det_ctx)
        qkv_d = y_d.view(y_d.shape[0], 2 * H_qk + HV, 128)
        if l2n:  # the deterministic layer with the norm on the same raw views
            qd, kd = qkv_d[:, :H_qk], qkv_d[:, H_qk : 2 * H_qk]
        else:
            qd = _l2_normalize(qkv_d[:, :H_qk])
            kd = _l2_normalize(qkv_d[:, H_qk : 2 * H_qk])
        out_det = det_kcp.cake_gdn_det_kcp(
            qd,
            kd,
            qkv_d[:, 2 * H_qk :],
            det_leaves[2],
            det_leaves[3],
            det_ctx,
            save_transfer=False,
            backend=backend,
            **({"use_qk_l2norm_in_kernel": True} if l2n else {}),
        )
        cc_det.phase = "bwd"
        out_det.backward(dout)
    rec_det = cc_det.record()
    del det_leaves, y_d, qkv_d, qd, kd, out_det
    flags: dict[str, bool] = {
        "layer_orig_bwd_comm_unchanged": all(
            rec_h[t].get(ph, 0) == rec_p[t].get(ph, 0)
            for t in ("calls", "waits")
            for ph in ("orig", "bwd")
        ),
        "layer_bwd_collectives_eq_det": rec_d["named"].get("bwd", {})
        == rec_det["named"].get("bwd", {}),
        "layer_no_param_reduction": _no_param_reduction(rec_d, rec_p, rec_h),
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
    # (g) the FP64 layer oracle and the cross-form budget checks
    ref_mod = _reference()
    refs = _layer_reference(row, part, cu, rank, device, x_all, w_all, conv)
    judge = _Judge(row, ref_mod, None)
    for name, r_ in (("direct", res_d), ("plain_ckpt", res_p), ("handle", res_h)):
        judge.fp64(name, r_, refs, LAYER_TENSORS)
    judge.cross("handle_vs_plain_ckpt", res_h, res_p, LAYER_TENSORS)
    judge.cross("handle_vs_direct", res_h, res_d, LAYER_TENSORS)
    del refs, res_d, res_p, res_h
    host = _host_map(group, W)
    flags["host_map"] = host["ok"]
    verdicts = judge.finish(group, W)
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
        "judged": {},
        "info_by_rank": [
            {
                "rank": rank,
                "T_r": int(x_local.shape[0]),
                "conv_takes_handle": conv_takes_handle,
                "conv_fwd_collectives": conv_fwd,
                "direct": rec_d,
                "plain": rec_p,
                "handle": rec_h,
                "det_direct": rec_det,
                "plain_recompute_calls": rec_p["calls"].get("recompute", 0),
                "handle_recompute_calls": rc_calls,
                "handle_recompute_waits": rc_waits,
            }
        ],
        "seconds": None,
        "C": C,
    }
    _fold(report, [flags], verdicts)
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
def test_layer_checkpoint_rows(row: Row, slurm) -> None:
    """#7818 whole-layer rows: KCP convolution + L2 norm + cake_gdn_nondet_kcp in one checkpointed
    region with one recompute handle (see the block comment above).  Runs under Slurm with >= the row's ranks."""
    if slurm is None:
        pytest.skip("the layer rows run under Slurm")
    conv = _conv_mod()
    if not conv.is_cake_causal_conv1d_det_train_supported(slurm.device):
        pytest.skip("the conv package does not support this device")
    _slurm_row(row, slurm, _backend_name(), _run_rank_layer)
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
def test_l2n_layer_checkpoint_rows(row: Row, slurm) -> None:
    """#7817 whole-layer rows: KCP convolution + ``cake_gdn_nondet_kcp(use_qk_l2norm_in_kernel=True)``
    on the RAW convolution output views in one checkpointed region (the B3 boundary), with
    and without a recompute handle, against the FP64 layer oracle and one another within the budgets."""
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
    )
