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

Cross-family CPU check of the GDN KCP metadata layer of the NON-DETERMINISTIC package
``cake_pplx._backends.cake_gdn_nondet_train_kcp``, moved here unchanged from
``kernels/gdn/tests/experimental/test_cake_gdn_nondet_train_kcp.py`` (the in-family checks stay there): the
package-local plan module against the KDA family's ``cake_kda_det_train_kcp.cake_kcp_plan`` (``cake-pplx-kda``;
field-identical partitions and contexts on the step-0 rows with ``H := HV``), and the per-rank crossing counts and
exchange bytes against the rows of record.  The KDA package is imported by this test only, never by the package, and
through ``tests/cake/test_helpers/cake_package.py``: a missing distribution fails, never skips.

Environment of the test (never of the package): ``GDN_ROWS_DIR`` (the step-0 row JSONs; the row test skips without
it), ``GDN_EVIDENCE_DIR`` (JSONL evidence records).
"""

from __future__ import annotations

import collections.abc
import glob
import hashlib
import json
import os

import pytest
import torch


def _package(submodule: str):
    """A module of the package through the test-helper registry: a stale entry fails here, never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_nondet_train_kcp", submodule)


gp = _package("cake_kcp_plan")
gb = _package("cake_bytes")
ROWS_DIR = os.environ.get("GDN_ROWS_DIR")
EVIDENCE_DIR = os.environ.get("GDN_EVIDENCE_DIR")
SM_COUNT = 152  # GB300
C_CONV = 10_240  # the rows' convolution width (2 * 16 + 48) * 128
CPU = torch.device("cpu")


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (one test of this suite on a 352-CPU host: 272 s by default, 6 s with at
    most 8 threads and oneDNN off), so this module runs that way."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _size_rule_at(kcp, monkeypatch, rows: int) -> None:
    """The size rule of ``reuse_recompute_tapes="auto"`` under ``graph="replay"`` at ``rows`` on every architecture
    (``cake_kcp.GRAPH_RECOMPUTE_MIN_ROWS`` and its default), with an empty architecture cache."""
    table = {arch: rows for arch in kcp.GRAPH_RECOMPUTE_MIN_ROWS}
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS", table)
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT", rows)
    monkeypatch.setattr(kcp, "_GRAPH_RECOMPUTE_ARCH", {})


@pytest.fixture(autouse=True)
def _graph_recompute_at_any_size(monkeypatch):
    """The test problems hold far fewer rows than the size rule of ``reuse_recompute_tapes="auto"`` under
    ``graph="replay"``, so every test runs with the rule at 0 rows (the recompute decides at any size); the rule's own
    test sets it around its problem."""
    _size_rule_at(_package("cake_kcp"), monkeypatch, 0)


def _evidence(name: str, record: dict) -> None:
    if not EVIDENCE_DIR:
        return
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, f"{name}.jsonl"), "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- rows of record


def _rows() -> list[str]:
    if not ROWS_DIR:
        return []
    return sorted(
        p
        for p in glob.glob(os.path.join(ROWS_DIR, "*.json"))
        if not os.path.basename(p).startswith("rows_")
    )


ROW_FILES = _rows()
ROW_IDS = [os.path.basename(p)[: -len(".json")] for p in ROW_FILES]
ROWS = pytest.mark.skipif(
    not ROW_FILES, reason="GDN_ROWS_DIR with the step-0 row JSONs not set"
)


def _ragged_fragments(
    cu: list[int], t_r: list[int]
) -> list[list[tuple[int, int, int]]]:
    """Explicit per-rank fragment tables of a contiguous ragged split (harness ``gdn_shapes.ragged_fragments``,
    copied: the rows' ``fragments=`` input)."""
    out: list[list[tuple[int, int, int]]] = []
    pos = 0
    num_docs = len(cu) - 1
    for n in t_r:
        gs, ge = pos, pos + n
        frs: list[tuple[int, int, int]] = []
        d = 0
        while d < num_docs and cu[d + 1] <= gs:
            d += 1
        p = gs
        while p < ge and d < num_docs:
            e = min(ge, cu[d + 1])
            if e > p:
                frs.append((d, p - cu[d], e - p))
            p = e
            d += 1
        out.append(frs)
        pos = ge
    return out


def _partition_inputs(row: dict) -> dict:
    """The ``cake_gdn_nondet_train_kcp_partition`` call of a row JSON (layout C / Z / D / R)."""
    cu = torch.tensor(row["cu_seqlens"], dtype=torch.int64)
    W = int(row["cp"])
    if row["layout"] == "R":
        return {
            "cu_seqlens_global_cpu": cu,
            "world_size": W,
            "fragments": _ragged_fragments(row["cu_seqlens"], row["ragged_table"]),
        }
    if row["layout"] == "D":
        return {
            "cu_seqlens_global_cpu": cu,
            "world_size": W,
            "layout": "zigzag_per_document",
            "cut_points": row["cut_points"],
        }
    return {"cu_seqlens_global_cpu": cu, "world_size": W, "layout": row["fi_layout"]}


def _plain(x):
    """Dataclass / tensor / tuple -> JSON-like plain values (tensors as lists with their dtype)."""
    if isinstance(x, torch.Tensor):
        return {"dtype": str(x.dtype), "shape": list(x.shape), "values": x.tolist()}
    # the contexts are plain classes listing their fields in ``_FIELDS``; the other plan objects are dataclasses
    fields = getattr(type(x), "_FIELDS", None) or getattr(
        x, "__dataclass_fields__", None
    )
    if fields is not None:
        return {
            f: _plain(getattr(x, f))
            for f in fields
            if not f.startswith("_") and f != "sequence"
        }
    if isinstance(
        x, collections.abc.Mapping
    ):  # dicts and the KDA module's lazily uploaded table mapping
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _rename_h(x):
    """The KDA module's ``H`` keys -> ``HV`` (the only naming difference between the two modules)."""
    if isinstance(x, dict):
        return {("HV" if k == "H" else k): _rename_h(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rename_h(v) for v in x]
    return x


def _context_record(ctx, HV: int) -> dict:
    """Every field of a context and its head plan as plain values (the field-identity record)."""
    rec = _plain(ctx)
    rec.pop("partition", None)
    rec.pop("group", None)
    # the KDA module keeps the fragment table as host rows plus a lazily uploaded property and records its upload
    # knobs; compare the table itself on both sides
    for name in ("frag_rows", "upload", "table_cache"):
        rec.pop(name, None)
    rec["frag_table"] = _plain(ctx.frag_table)
    rec["device"] = str(ctx.device)
    rec["heads"] = _plain(ctx.heads(HV))
    rec["describe"] = ctx.describe(HV)
    return rec


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_plan_parity_with_kda_on_rows(path, monkeypatch) -> None:
    """Result 1: ``cake_gdn_nondet_train_kcp.cake_kcp_plan`` is field-identical to ``cake_kda_det_train_kcp.cake_kcp_plan`` on
    every step-0 row (partition, every rank's context and head plan at ``H := HV``) and reproduces the row's recorded
    fragment table.  Result 2: ``n_in`` / ``n_out`` / ``max_chain_fragments`` and the exchange bytes of
    ``cake_gdn_nondet_train_kcp_comm_bytes`` equal the row JSON.  Both modules plan under the per-rank rule they share,
    ``"wave_fill1"`` (the GDN default is the row rule, the KDA default its own tie rule; each is selected explicitly
    here: the keyword-only ``p_rule`` on the GDN side, the private test hook on the KDA side)."""
    from tests.cake.test_helpers.cake_package import cake_package

    kp = cake_package("cake_kda_det_train_kcp", "cake_kcp_plan")  # the test only
    # the KDA module accepts its private test hook only after its opt-in (enable_test_knobs)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    with open(path) as fh:
        row = json.load(fh)
    H_qk, HV = (int(x) for x in row["heads"])
    W = int(row["cp"])
    args = _partition_inputs(row)
    part = gp.cake_gdn_nondet_train_kcp_partition(**args)
    kpart = kp.cake_kda_det_train_kcp_partition(**args)
    assert part.to_dict() == kpart.to_dict(), "partition differs from the KDA module"
    assert _plain(part.to_dict()["fragments"]) == row["fragments"], (
        "partition differs from the row of record"
    )  # JSON round trip: the row stores (prev, next) tuples as lists
    assert [list(part.local_cu_seqlens(r)) for r in range(W)] == row["local_cu_seqlens"]
    n_in, n_out = part.crossing_counts()
    assert (
        list(n_in) == row["crossing_in_per_rank"]
        and list(n_out) == row["crossing_out_per_rank"]
    )
    assert part.max_chain_fragments() == row["max_chain_fragments"]
    assert [len(c) for c in part.chains()] == row["chain_fragments_per_doc"]
    kda_bytes = row.get("kda_plan_comm_bytes_H=HV")
    if isinstance(kda_bytes, dict) and "error" in kda_bytes:
        kda_bytes = None
    per_rank, records = [], []
    for r in range(W):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )  # the per-rank rule both modules share; the GDN default "row_cost" is tested on the rows below
        kctx = kp.cake_kda_det_train_kcp_build_context(
            kpart,
            None,
            r,
            CPU,
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1"},
        )
        mine, theirs = _context_record(ctx, HV), _rename_h(_context_record(kctx, HV))
        records.append(mine)
        assert mine == theirs, (
            f"rank {r}: context / head plan differ from the KDA module (HV={HV})"
        )
        assert (ctx.n_in, ctx.n_out) == (n_in[r], n_out[r])
        cb = gb.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        entry = {
            "rank": r,
            "n_in": ctx.n_in,
            "n_out": ctx.n_out,
            "T_r": ctx.T_r,
            "n_frag": ctx.n_frag,
            "P": ctx.heads(HV).P,
        }
        entry.update({k: cb[k] for k in ("B1", "B2", "B3", "halo_fwd", "halo_bwd")})
        if kda_bytes is not None:
            # the rows were built with the KDA module's context at sm_count=1; the exchange bytes do not depend on P
            assert {
                k: cb[k] for k in ("B1", "B2", "B3", "halo_fwd", "halo_bwd")
            } == kda_bytes[str(r)], (
                f"rank {r}: exchange bytes differ from the row of record"
            )
            entry["rows_match"] = True
        per_rank.append(entry)
    _evidence(
        "parity",
        {
            "row": row["row"],
            "cp": W,
            "layout": row["layout"],
            "fla_padded": row["fla_padded"],
            "H_qk": H_qk,
            "HV": HV,
            "total": row["padded_total"],
            "fragments": sum(len(frs) for frs in part.fragments),
            "n_in": list(n_in),
            "n_out": list(n_out),
            "max_chain_fragments": part.max_chain_fragments(),
            "bytes_rows_match": kda_bytes is not None,
            "per_rank": per_rank,
            "context_sha256": _sha(records),
            "equal": True,
        },
    )
