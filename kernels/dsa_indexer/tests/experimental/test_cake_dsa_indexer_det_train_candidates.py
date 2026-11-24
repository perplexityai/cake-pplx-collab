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

Backend tests of the candidate-block call of the deterministic Cake DSA indexer
(``cake_dsa_indexer_det_train_topk_with_candidates``, ``prepare_dsa_indexer_det_topk_with_candidates``,
``plan_dsa_indexer_det_topk_with_candidates`` and the workspace bound): the registry's ``scan_blocks`` programs, the
host plan at the request's shapes and on a dense grid of call geometries, the call's argument checks and the refusal
of an architecture whose registry lacks the programs (CPU), then on an SM100 / SM103 / SM107 device the exact sets
against the FP64 oracle (``tests/experimental/test_helpers/cake_dsa_indexer_candidates.py``) on exact MXFP4-grid inputs, the Gaussian rule,
the containment of every id in a returned block, bitwise repetition and the equality with the plain top-k call's bits (also at
``top_k`` 2044 and 2047), a prepared runner that draws its work units on a poisoned caller-owned workspace, the caller-owned
workspace and outputs, and the absence of host synchronization; where every row keeps all its blocks (``iota``),
the plain call's plan, the FP64 oracle's rows and the ``scan_blocks`` programs' result, a CUDA graph replaying new
segment boundaries, and no allocation or host synchronization in a run.
"""

import contextlib
import functools
import json
import math
import random
import subprocess
import sys
import warnings

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers import cake_dsa_indexer_candidates as cb
from tests.experimental.test_helpers import cake_dsa_indexer_naive as naive
from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref

cake_backend = cake_package("cake.cake_dsa_indexer_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_indexer_det_train", "cake_jit")
cake_policy = cake_package("cake.cake_dsa_indexer_det_train", "cake_policy")
CB_ROLE = cake_policy.CB_ROLE
CB_SLOTS = cake_policy.CB_SLOTS
CB_MAX_TOP_K = cake_policy.CB_MAX_TOP_K
TICKET_TAIL_BYTES = cake_policy.TICKET_TAIL_BYTES
CandidateBlocksChoice = cake_policy.CandidateBlocksChoice
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
generated_program_available = cake_backend.generated_program_available
arch_for = cake_backend.arch_for
topk = cake_backend.cake_dsa_indexer_det_train_topk
topk_with_candidates = cake_backend.cake_dsa_indexer_det_train_topk_with_candidates
plan_with_candidates = cake_backend.plan_dsa_indexer_det_topk_with_candidates
prepare_with_candidates = cake_backend.prepare_dsa_indexer_det_topk_with_candidates
with_candidates_workspace_size = (
    cake_backend.cake_dsa_indexer_det_train_topk_with_candidates_workspace_size
)

ORDERED = True
TOP_K = 512
CB_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# the persistent grids (SM counts) of the devices of each architecture
GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160), "sm_107a": (212, 216)}
CB_SLOTS_PLAN = [
    ["buffer", "CandBlk"],
    ["buffer", "CandBlocks"],
    ["parameter", "blk_cap"],
]
_CB_KEYS_U1 = [
    f"{CB_ROLE}:{kind}:u1:s0:f{fit}" for kind in ("l6", "narrow") for fit in (0, 1)
]
# the ``scan_blocks`` keys of each architecture: the unpaired, unsplit narrow and l6 units of both fit forms at tile
# unroll 1; sm_107a also registers the narrow unit at tile unroll 2 (unsampled fit)
CB_KEYS = {
    "sm_100a": sorted(_CB_KEYS_U1),
    "sm_103a": sorted(_CB_KEYS_U1),
    "sm_107a": sorted(_CB_KEYS_U1 + [f"{CB_ROLE}:narrow:u2:s0:f0"]),
}
# the request's shapes: T = 16,384 queries at the tail of one document of 16,384 / 65,536 / 262,144 keys, and the
# packed layout of eight segments (T = 16,231, Tkv = 268,757)
P1_SEG_Q = (1777, 1755, 2123, 2110, 2123, 2110, 2123, 2110)
P1_SEG_K = (1777, 56842, 2123, 67923, 2123, 67923, 2123, 67923)
SHAPES = {
    "tail16k": ((16384,), (16384,)),
    "tail64k": ((16384,), (65536,)),
    "tail256k": ((16384,), (262144,)),
    "packed8": (P1_SEG_Q, P1_SEG_K),
}
# the ``scan_blocks`` program of each request shape on sm_107a, on both grids, with and without ``max_seqlen_k``
# (tail16k keeps every block of every row: the plain call's plan, ``iota``)
CB_KEYS_AT_THE_REQUEST_SHAPES = {
    "sm_107a": {
        "tail64k": f"{CB_ROLE}:l6:u1:s0:f0",
        "tail256k": f"{CB_ROLE}:l6:u1:s0:f0",
        "packed8": f"{CB_ROLE}:l6:u1:s0:f0",
    },
}
# the refusal of a plan whose unit no registered ``scan_blocks`` program serves
CB_REFUSAL = "no registered scan_blocks program serves"
# more work units than any grid of the architectures (8,192 queries: 2,048 narrow or 1,366 l6 units), so the scan
# draws its units from a ticket word
DRAWING_SHAPE = ([8192], [32768])
# the architectures whose ``scan_blocks`` programs draw from their TicketWord argument (registry ``ticket_arg_arches``)
TICKET_WORD_ARCHES = ("sm_107a",)
# the architectures whose staged rank finalize runs its persistent program (key tag ``:persist``) where it moves the
# rows in bulk: ``top_k`` a multiple of 4, as ``PERSIST_TOP_K``, up to the policy's ``rank_persist_max_k``
PERSIST_ARCHES = ("sm_107a",)
PERSIST_TOP_K = 2044
# the exact layouts: default offsets (rows with fewer than, exactly and more than 2048 visible blocks, n % 8 != 0,
# unequal query and key lengths, an empty query segment, a segment without keys) and explicit offsets (positive,
# beyond the keys, negative: rows without visible keys)
LAYOUTS = {
    "default_offsets": ((64, 40, 0, 16, 24), (16420, 300, 100, 0, 17000), None),
    "explicit_offsets": (
        (48, 24, 16, 16),
        (20000, 30000, 500, 64),
        (16370, 25000, 600, -8),
    ),
}


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_cb_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")
    if arch_for(torch.device("cuda")) not in CB_ARCHES:
        pytest.skip(
            "the candidate-block programs are registered for sm_100a, sm_103a and sm_107a only"
        )


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool):
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)


def _offsets_tensor(offsets, device):
    return (
        None
        if offsets is None
        else torch.tensor(offsets, dtype=torch.int64, device=device)
    )


def _call(q, k, w, cq, ck, *, offsets=None, scale=cb.EXACT_SCALE, **extra):
    out = topk_with_candidates(
        q,
        k,
        w,
        cq,
        ck,
        top_k=extra.pop("top_k", TOP_K),
        softmax_scale=scale,
        q_causal_offsets=offsets,
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 3
    return tuple(t.clone() for t in out)


def _structure_failures(indices, scores, blocks, ns) -> list[str]:
    """The structure every row keeps on any input: ``min(top_k, n)`` unique visible ids and ``min(2048, nb)`` unique
    blocks in ``[0, nb)`` with the newest present, ``-1`` elsewhere, ``-inf`` scores on the padding, rows in ascending id order with a ``-1`` tail."""
    out = []
    ns_t = torch.tensor(ns, dtype=torch.int64, device=indices.device)
    nbs = (ns_t + 7) // 8
    for name, t, cap, count in (
        ("indices", indices.long(), ns_t, ns_t.clamp(max=indices.shape[1])),
        ("candidate_blocks", blocks.long(), nbs, nbs.clamp(max=CB_SLOTS)),
    ):
        valid = t >= 0
        if not torch.equal(valid.sum(1), count):
            out.append(f"{name}: valid counts differ from min(slots, visible)")
        if bool(((t < -1) | (valid & (t >= cap[:, None]))).any()):
            out.append(f"{name}: an id outside [-1, visible)")
        s = cb.ascending(t)
        if bool(((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any()):
            out.append(f"{name}: a repeated id")
        if ORDERED and not torch.equal(s, t):
            out.append(f"{name}: a row not in ascending order with a -1 tail")
    if bool(((indices < 0) != torch.isneginf(scores)).any()):
        out.append("scores: -inf exactly on the padding slots")
    has = (blocks.long() == (nbs - 1)[:, None]).any(1) | (ns_t == 0)
    if not bool(has.all()):
        out.append("candidate_blocks: a row without its newest block")
    if not bool(cb.contained(indices, blocks).all()):
        out.append("an id of indices outside the row's candidate blocks")
    return out


def _visible_counts(cq, ck, offsets) -> list[int]:
    segs = cb.segments(cq.cpu(), ck.cpu(), None if offsets is None else offsets.cpu())
    return [cb.visible(u, off, Lk) for _, Lq, _, Lk, off in segs for u in range(Lq)]


def _exact_failures(q, k, w, cq, ck, offsets, got, *, rows=None) -> list[str]:
    """``got = (indices, scores, candidate_blocks)`` of exact inputs against the FP64 oracle on ``rows`` (default all):
    the sets exactly (rows in ascending id order with a ``-1`` tail), the scores equal to the FP64 scores of the ids (exact on these inputs)."""
    indices, scores, blocks = got
    g, ids64, cand64 = cb.with_candidates(
        q,
        k,
        w,
        cq,
        ck,
        softmax_scale=cb.EXACT_SCALE,
        top_k=indices.shape[1],
        q_causal_offsets=offsets,
        rows=rows,
    )
    gi, gb, gs = indices[g].long(), blocks[g].long(), scores[g]
    if not ORDERED:
        order = torch.where(gi >= 0, gi, torch.iinfo(torch.int64).max).argsort(1)
        gi, gs, gb = gi.gather(1, order), gs.gather(1, order), cb.ascending(gb)
        gi = torch.where(gi == torch.iinfo(torch.int64).max, -1, gi)
    out = []
    if not torch.equal(gi, ids64):
        bad = (gi != ids64).any(1).nonzero().flatten()[:5].tolist()
        out.append(
            f"indices differ from the FP64 oracle in rows {[int(g[r]) for r in bad]}"
        )
    if not torch.equal(gb, cand64):
        bad = (gb != cand64).any(1).nonzero().flatten()[:5].tolist()
        out.append(
            f"candidate_blocks differ from the FP64 oracle in rows {[int(g[r]) for r in bad]}"
        )
    want = ref_scores64(q, k, w, cq, ck, offsets, g, gi)
    if not torch.equal(gs.double(), want):
        out.append("scores differ from the FP64 scores of the ids")
    return out


def ref_scores64(q, k, w, cq, ck, offsets, rows, ids) -> torch.Tensor:
    """The FP64 scores of the ids ``ids [R, K]`` of the global query rows ``rows`` (``-inf`` at ``-1``)."""
    out = torch.full(ids.shape, -math.inf, dtype=torch.float64, device=ids.device)
    for q0, Lq, k0, Lk, _ in cb.segments(cq.cpu(), ck.cpu()):
        sel = ((rows >= q0) & (rows < q0 + Lq)).nonzero().flatten()
        if not sel.numel() or Lk == 0:
            continue
        r, j = rows[sel], ids[sel]
        x = torch.einsum("thd,tkd->thk", q[r].double(), k[k0 + j.clamp(min=0)].double())
        s = (w[r].double()[:, :, None] * torch.relu(cb.EXACT_SCALE * x)).sum(1) + 0.0
        out[sel] = torch.where(j >= 0, s, -math.inf)
    return out


# ---------------------------------------------------------------------------
# Host only: registry, plan, argument checks
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_registry_ships_the_candidate_block_programs():
    """``scan_blocks`` is a role of its own after the family's roles; its argument plan is the scan's with ``CandBlk``,
    ``CandBlocks`` and ``blk_cap`` right before ``K_scale``; every architecture registers the unpaired, unsplit BF16
    narrow and l6 programs of both fit forms at tile unroll 1 under ``scan_blocks:`` keys, and sm_107a also the
    narrow program at tile unroll 2 of the unsampled fit (``CB_KEYS``), each a program of its own beside the scan
    program of the same unit."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    assert cake_jit.STAGES.count(CB_ROLE) == 1
    # after the family's roles; only the candidate-restricted top-k (``cand``) follows it
    assert cake_jit.STAGES[cake_jit.STAGES.index(CB_ROLE) + 1 :] == ["cand"]
    scan = cake_jit.ARG_PLANS["scan"]
    at = [name for _, name in scan].index("K_scale")
    assert cake_jit.ARG_PLANS[CB_ROLE] == scan[:at] + CB_SLOTS_PLAN + scan[at:]
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        cb_keys = sorted(key for key in keys if key.startswith(f"{CB_ROLE}:"))
        if arch not in CB_ARCHES:
            assert cb_keys == [], arch
            continue
        assert cb_keys == CB_KEYS[arch], arch
        for key in cb_keys:
            name = keys[key]
            record = cake_jit.PROGRAMS[name]
            assert record["role"] == CB_ROLE and arch in record["arches"], (arch, key)
            twin = "scan:" + key.split(":", 1)[1]
            assert twin in keys and keys[twin] != name, (arch, key)
    roles = {record["role"] for record in cake_jit.PROGRAMS.values()}
    assert CB_ROLE in roles
    for name, record in cake_jit.PROGRAMS.items():
        if record["role"] == CB_ROLE:
            assert set(record["arches"]) <= set(CB_ARCHES), name


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CB_ARCHES)
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("bound", ["max_seqlen_k", "num_keys"])
def test_plan_at_the_request_shapes(arch, shape, bound):
    """Per grid of the architecture, where ``ceil(W / 8) > 2048`` (``W``: ``max_seqlen_k`` when given, else ``Tkv``):
    the plan runs a registered ``scan_blocks`` program unsplit and unpaired (on sm_107a the program of
    ``CB_KEYS_AT_THE_REQUEST_SHAPES``), the block entries are ``ceil(W / 8) - 1`` per query slot and the call's
    workspace is the scan's (candidate buffers and ticket tail) plus the block entries.  Elsewhere (tail16k) every row
    keeps all its blocks: the plan is the plain call's (``iota``) with its workspace and no block entries.  The bound
    is the larger plan of the two sampling modes."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    seg_q, seg_k = SHAPES[shape]
    T, Tkv, S = sum(seg_q), sum(seg_k), len(seg_q)
    msk = max(seg_k) if bound == "max_seqlen_k" else None
    W = Tkv if msk is None else msk
    nb = -(-W // 8)
    expected = CB_KEYS_AT_THE_REQUEST_SHAPES.get(arch, {}).get(shape)
    for grid in GRIDS[arch]:
        kw = dict(top_k=TOP_K, arch=arch, num_sms=grid, max_seqlen_k=msk)
        plan = plan_with_candidates(T, Tkv, S, **kw)
        assert isinstance(plan, CandidateBlocksChoice)
        c = plan.choice
        assert plan.iota == (nb <= CB_SLOTS) == (shape == "tail16k"), grid
        if plan.iota:
            assert c == cake_backend.plan_dsa_indexer_det_topk(T, Tkv, S, **kw), grid
            assert (plan.blk_cap, plan.blk_bytes) == (0, 0), grid
            assert plan.workspace_bytes == c.workspace_bytes, grid
        else:
            assert (
                c.scan_key in CB_KEYS[arch]
                and c.scan_key in cake_jit.PROGRAM_KEYS[arch]
            ), (grid, c.scan_key)
            assert expected is None or c.scan_key == expected, (grid, c.scan_key)
            assert (
                c.n_split == 1 and not c.split and not c.pair and c.merge_key is None
            ), grid
            assert (
                c.workspace_bytes
                == c.grid_ctas * c.block_q * c.cand_cap * 8 + TICKET_TAIL_BYTES
            ), grid
            assert plan.blk_cap == nb - 1, grid
            assert plan.blk_bytes == c.grid_ctas * c.block_q * plan.blk_cap * 8, grid
            assert plan.workspace_bytes == c.workspace_bytes + plan.blk_bytes, grid
            assert plan.blocks_at == c.workspace_bytes, grid
        unsampled = plan_with_candidates(T, Tkv, S, sample_tiles_max=0, **kw)
        bound_bytes = with_candidates_workspace_size(T, Tkv, S, **kw)
        assert bound_bytes == max(plan.workspace_bytes, unsampled.workspace_bytes), grid


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CB_ARCHES)
def test_candidate_block_plans_run_registered_programs_on_a_dense_grid(arch):
    """Every program a candidate-block plan names is registered for the architecture, on a dense grid of call
    geometries: 1 to 2^17 queries in 1, 8 or 64 segments, 0 to 2^13 mean key tiles per segment, ``top_k`` 1, 512,
    1024, 1025 and 2047, without ``max_seqlen_k`` and with it at the mean key segment and at ``Tkv`` (one long
    segment), both sampling modes and every grid of the architecture.
    The host fails closed by name on an unregistered program, so a selectable but unregistered ``scan_blocks`` or
    finalize program would fail in production (the backend suite's dense-grid test does the same for the plain
    call).  Every plan with ``ceil(W / 8) > 2048`` runs one ``scan_blocks`` program unsplit and without a merge;
    every other plan is the plain call's (``iota``)."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    keys = cake_jit.PROGRAM_KEYS[arch]
    tile = cake_backend.policy_for(arch).tile_keys
    queries = sorted({1, 3, 6018} | {int(round(2 ** (17 * i / 10))) for i in range(11)})
    tiles = sorted({0, 1} | {int(round(2 ** (13 * i / 8))) for i in range(9)})
    chosen, iota = set(), 0
    for grid in GRIDS[arch]:
        for S in (1, 8, 64):
            for T in queries:
                for m in tiles:
                    Tkv = max(1, int(round(m * tile * S + T / 2)))
                    for top_k in (1, 512, 1024, 1025, CB_MAX_TOP_K):
                        for bound in [None, *sorted({-(-Tkv // S), Tkv})]:
                            for sample_tiles_max in (None, 0):
                                plan = plan_with_candidates(
                                    T,
                                    Tkv,
                                    S,
                                    top_k=top_k,
                                    arch=arch,
                                    num_sms=grid,
                                    max_seqlen_k=bound,
                                    sample_tiles_max=sample_tiles_max,
                                )
                                c = plan.choice
                                where = (
                                    grid,
                                    S,
                                    T,
                                    Tkv,
                                    top_k,
                                    bound,
                                    sample_tiles_max,
                                )
                                W = Tkv if bound is None else bound
                                assert plan.iota == (-(-W // 8) <= CB_SLOTS), where
                                for key in plan.keys:
                                    assert key in keys, (*where, key)
                                if plan.iota:
                                    assert not c.scan_key.startswith(f"{CB_ROLE}:")
                                    assert plan.blk_cap == 0, where
                                    iota += 1
                                    continue
                                assert c.scan_key.startswith(f"{CB_ROLE}:"), where
                                assert c.n_split == 1 and c.merge_key is None, where
                                chosen.add(c.scan_key)
    assert chosen <= set(CB_KEYS[arch]), sorted(chosen)
    assert iota > 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CB_ARCHES)
def test_architecture_without_candidate_block_programs_is_refused(arch, monkeypatch):
    """An architecture whose registry has no ``scan_blocks`` program (here the registry of ``arch`` without its
    ``scan_blocks:`` keys): the plan and the workspace bound raise the candidate-block dispatch's
    ``NotImplementedError`` (also where every row keeps all its blocks and the plain programs would run), and the
    plain call's plan is unaffected."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    keys = cake_backend.PROGRAM_KEYS[arch]
    stripped = {
        key: name for key, name in keys.items() if not key.startswith(f"{CB_ROLE}:")
    }
    assert len(stripped) < len(keys)
    monkeypatch.setitem(cake_backend.PROGRAM_KEYS, arch, stripped)
    grid = GRIDS[arch][0]
    for fn in (plan_with_candidates, with_candidates_workspace_size):
        for Tkv in (262144, 16384):
            with pytest.raises(NotImplementedError, match=CB_REFUSAL):
                fn(16384, Tkv, 1, top_k=TOP_K, arch=arch, num_sms=grid)
    plain = cake_backend.plan_dsa_indexer_det_topk(
        16384, 262144, 1, top_k=TOP_K, arch=arch, num_sms=grid
    )
    assert not plain.scan_key.startswith(f"{CB_ROLE}:")


@pytest.mark.cpu_only
def test_plan_arguments_are_checked():
    """``1 <= top_k <= 2047`` (with at most 2047 ids every id lies in a returned block) and ``ratio == 1`` (blocks of
    8 keys), before any architecture is consulted."""
    for top_k in (0, CB_MAX_TOP_K + 1, 4096):
        with pytest.raises(ValueError):
            plan_with_candidates(64, 20000, 1, top_k=top_k, arch="sm_100a", num_sms=148)
    for ratio in (2, 4):
        with pytest.raises(ValueError, match="ratio == 1"):
            plan_with_candidates(
                64, 20000, 1, top_k=TOP_K, ratio=ratio, arch="sm_100a", num_sms=148
            )
    assert CB_MAX_TOP_K == CB_SLOTS - 1 == 2047


# ---------------------------------------------------------------------------
# Device: exact sets, Gaussian rule, repetition, workspace and outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("case", cb.EXACT_CASES)
def test_exact_sets_match_the_fp64_oracle(case, layout):
    """On exact inputs (MXFP4-grid ``q`` / ``k``, ``w`` in ``{-4..4} / 8``, ``softmax_scale`` 1/16) every row's
    ``indices`` and ``candidate_blocks`` equal the FP64 oracle's sets (rows in ascending id order with a ``-1`` tail) and the scores equal the FP64
    scores of the ids: distinct scores, ties at the block and key cutoffs and equal block maxima (duplicated keys),
    all-equal and signed-zero scores (the newest 2048 blocks and ``top_k`` keys win), negative scores, a newest block
    with the lowest block score; rows with fewer than, exactly and more than 2048 visible blocks, ``n % 8 != 0``,
    unequal query and key lengths, positive / negative offsets, empty segments and rows without visible keys."""
    _require_cb_program()
    seg_q, seg_k, offsets = LAYOUTS[layout]
    seed = 7700 + 31 * cb.EXACT_CASES.index(case) + (0 if offsets is None else 1)
    q, k, w, cq, ck = cb.exact_inputs(case, seg_q, seg_k, seed=seed, device="cuda")
    off = _offsets_tensor(offsets, q.device)
    got = _call(q, k, w, cq, ck, offsets=off)
    ns = _visible_counts(cq, ck, off)
    nbs = [(n + 7) // 8 for n in ns]
    assert min(nbs) <= CB_SLOTS < max(nbs)
    failures = _structure_failures(*got, ns)
    failures += _exact_failures(q, k, w, cq, ck, off, got)
    assert not failures, f"{case} / {layout}:\n" + "\n".join(failures[:20])
    if case in ("duplicated keys", "all scores zero", "signed zeros"):
        # the case reaches ties at the block cutoff of a row with more than 2048 blocks
        rows = [r for r, nb in enumerate(nbs) if nb > CB_SLOTS][:4]
        ties = 0
        for r in rows:
            s, n = cb.row_scores64(
                q, k, w, cq, ck, r, softmax_scale=cb.EXACT_SCALE, q_causal_offsets=off
            )
            bmax = cb.block_maxima(s.tolist(), n)
            kept = {int(b) for b in got[2][r].tolist() if b >= 0} - {len(bmax) - 1}
            ties += cb.cut_tie(bmax, list(range(len(bmax) - 1)), kept)
        assert ties > 0, case
    if case == "newest block lowest" and offsets is None:
        # the last rows of the 16,420- and 17,000-key segments see the zero block last
        lowest = 0
        for r, nb in enumerate(nbs):
            if nb <= CB_SLOTS:
                continue
            s, n = cb.row_scores64(
                q, k, w, cq, ck, r, softmax_scale=cb.EXACT_SCALE, q_causal_offsets=off
            )
            bmax = cb.block_maxima(s.tolist(), n)
            lowest += bmax[-1] < min(bmax[:-1])
        assert lowest > 0


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_request_shapes_on_sampled_rows(shape):
    """The request's shapes on exact inputs: the sets of sampled rows (the first and last rows of every segment and
    random ones) equal the FP64 oracle's, every row keeps the structure and the containment, and the call with
    ``max_seqlen_k`` (the longest key segment) returns the sets of the call without it."""
    _require_cb_program()
    seg_q, seg_k = SHAPES[shape]
    q, k, w, cq, ck = cb.exact_inputs(
        "MXFP4-grid", seg_q, seg_k, seed=7790, device="cuda"
    )
    got = _call(q, k, w, cq, ck)
    bounded = _call(q, k, w, cq, ck, max_seqlen_k=max(seg_k))
    ns = _visible_counts(cq, ck, None)
    failures = _structure_failures(*got, ns)
    rng = random.Random(7791)
    rows = set()
    for q0, Lq, _, _, _ in cb.segments(cq.cpu(), ck.cpu()):
        rows |= {q0, q0 + 1, q0 + Lq - 2, q0 + Lq - 1} if Lq > 1 else {q0}
        rows |= {q0 + rng.randrange(Lq) for _ in range(4)} if Lq else set()
    failures += _exact_failures(q, k, w, cq, ck, None, got, rows=sorted(rows))
    for a, b, name in (
        (got[0], bounded[0], "indices"),
        (got[2], bounded[2], "candidate_blocks"),
    ):
        if not torch.equal(cb.ascending(a), cb.ascending(b)):
            failures.append(f"{name}: the call with max_seqlen_k returns other sets")
    assert not failures, f"{shape}:\n" + "\n".join(failures[:20])


def test_gaussian_scores_and_sets(record_property):
    """Gaussian inputs (rows with more than 2048 visible blocks among them): the scores are judged against FP64
    beside the BF16 naive tier (the family's rule), and each row's sets may differ from the FP64 sets only in members
    whose FP64 value lies within twice the draw's largest score error of the FP64 cutoff; every id lies in a returned
    block."""
    _require_cb_program()
    state = {}

    def make(seed):
        return ref.make_random_inputs(
            [24, 8], [17000, 900], top_k=TOP_K, seed=seed, device="cuda"
        )

    def run(inputs):
        out = topk_with_candidates(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            **inputs.kwargs(),
        )
        state["blocks"] = out[2].clone()
        return out[0], out[1]

    def check(j, inputs, result, _):
        indices, scores = result
        blocks = state["blocks"]
        ns = ref.visible_rows(inputs).tolist()
        failures = _structure_failures(indices, scores, blocks, ns)
        e = 0.0
        rows = []
        for r, n in enumerate(ns):
            s, n = cb.row_scores64(
                inputs.q,
                inputs.k,
                inputs.w,
                inputs.cu_seqlens_q,
                inputs.cu_seqlens_k,
                r,
                softmax_scale=inputs.softmax_scale,
            )
            ids = indices[r].long()
            valid = ids >= 0
            if valid.any():
                e = max(
                    e, float((scores[r][valid].double() - s[ids[valid]]).abs().max())
                )
            rows.append((r, s.tolist(), n))
        for r, vals, n in rows:
            got = {int(x) for x in indices[r].tolist() if x >= 0}
            if not cb.near_cutoff_only(vals, list(range(n)), got, TOP_K, e):
                failures.append(
                    f"draw {j} row {r}: indices beyond the near-cutoff rule"
                )
            bmax = cb.block_maxima(vals, n)
            nb = len(bmax)
            kept = {int(b) for b in blocks[r].tolist() if b >= 0} - {nb - 1}
            if nb and not cb.near_cutoff_only(
                bmax, list(range(nb - 1)), kept, CB_SLOTS - 1, e
            ):
                failures.append(
                    f"draw {j} row {r}: candidate_blocks beyond the near-cutoff rule"
                )
        assert not failures, "\n".join(failures[:20])

    naive.judge_gaussian(
        record_property, make, run, seed=7795, ordered=ORDERED, check=check
    )


def test_indices_and_scores_are_the_plain_calls():
    """The call returns the ``indices`` / ``scores`` of the family's top-k call with the same ``top_k``
    (bit for bit), and its candidate blocks repeat (bit for bit, in ascending block id order with a ``-1`` tail)."""
    _require_cb_program()
    inputs = ref.make_random_inputs(
        [48, 300], [20000, 4000], top_k=TOP_K, seed=7796, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    first = tuple(t.clone() for t in topk_with_candidates(*args, **inputs.kwargs()))
    second = tuple(t.clone() for t in topk_with_candidates(*args, **inputs.kwargs()))
    plain = naive.cloned(topk(*args, **inputs.kwargs()))
    if ORDERED:
        for a, b in zip(first, second):
            assert torch.equal(a, b)
        assert ref.same_bits(first[:2], plain)
    else:
        assert naive.same_ids(first[:2], second[:2]) and naive.same_ids(
            first[:2], plain
        )
        assert torch.equal(cb.ascending(first[2]), cb.ascending(second[2]))
    assert bool(cb.contained(first[0], first[2]).all())


def _same_result(a, b) -> bool:
    """Two results of the call on one input: the same bits (det) or the same sets (nondet)."""
    if ORDERED:
        return ref.same_bits(a[:2], b[:2]) and torch.equal(a[2], b[2])
    if not naive.same_ids(a[:2], b[:2]):
        return False
    return torch.equal(cb.ascending(a[2]), cb.ascending(b[2]))


def _same_as_plain(got, plain) -> bool:
    """The call's ``indices`` / ``scores`` against the plain call's: the same bits (det) or the same sets (nondet)."""
    if ORDERED:
        return ref.same_bits(got[:2], plain)
    return naive.same_ids(got[:2], plain)


def test_prepared_runner_draws_its_tickets_on_a_poisoned_caller_workspace():
    """A call with more work units than the grid draws them from a ticket word: on the architectures of
    ``TICKET_WORD_ARCHES`` from a 16-byte block of the prepared runner's own, outside the caller-owned workspace,
    elsewhere from a ticket tail: the caller-owned workspace's, which the scan binding zeroes before every scan launch,
    or the backend-owned runner's, zeroed once at prepare.  With the whole caller-owned workspace filled with 0xFF
    before each of three runs, every run of that runner and of a backend-owned one returns the eager call's result, the
    word reads 0 after every run, and ``indices`` / ``scores`` are the plain call's."""
    _require_cb_program()
    inputs = ref.make_random_inputs(
        *DRAWING_SHAPE, top_k=TOP_K, seed=7801, device="cuda"
    )
    device = inputs.device
    arch = arch_for(device)
    T, Tkv, S = inputs.num_queries, inputs.num_keys, inputs.num_segments
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    eager = tuple(t.clone() for t in topk_with_candidates(*args, **inputs.kwargs()))
    plain = naive.cloned(topk(*args, **inputs.kwargs()))
    assert not _structure_failures(*eager, ref.visible_rows(inputs).tolist())
    assert _same_as_plain(eager, plain)
    nbytes = with_candidates_workspace_size(T, Tkv, S, top_k=TOP_K, device=device)
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    owned = prepare_with_candidates(*args, **inputs.kwargs())
    shared = prepare_with_candidates(
        *args, **inputs.kwargs(), workspace_buffer=workspace
    )
    assert not cake_policy.ticket_static_path(shared.choice)
    reads = cake_backend.reads_ticket_word(shared.programs[CB_ROLE], arch)
    assert reads == (arch in TICKET_WORD_ARCHES), (arch, shared.choice.scan_key)
    for runner in (owned, shared):
        if reads:
            assert runner.ticket_tail is None and runner.ticket_block is not None
            assert runner.ticket_block.numel() == cake_backend.TICKET_BLOCK_BYTES
        else:
            assert runner.ticket_block is None and runner.ticket_tail is not None
    if reads:
        at = shared.ticket_block.data_ptr()
        assert not workspace.data_ptr() <= at < workspace.data_ptr() + nbytes
    for step in range(3):
        workspace.fill_(0xFF)
        for runner in (shared, owned):
            out = runner()
            torch.cuda.synchronize()
            word = runner.ticket_block if reads else runner.ticket_tail
            assert not word[: cake_backend.TICKET_WORD_ZERO_BYTES].any(), step
            assert _same_result(out, eager), step
            assert _same_as_plain(out, plain), step


@pytest.mark.parametrize("top_k", [PERSIST_TOP_K, CB_MAX_TOP_K])
def test_indices_and_scores_are_the_plain_calls_at_the_largest_top_k(top_k):
    """At ``top_k`` 2047 and 2044, on a call that draws its work units: ``indices`` / ``scores`` are the plain call's,
    a prepared runner returns the eager call's result and every id lies in a returned block.  On the architectures of
    ``PERSIST_ARCHES`` the call at 2044 runs the persistent rank finalize (key tag ``:persist``), which needs the
    bulk row copies of a ``top_k`` that is a multiple of 4; 2047 is not one."""
    _require_cb_program()
    inputs = ref.make_random_inputs(
        *DRAWING_SHAPE, top_k=top_k, seed=7802, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    runner = prepare_with_candidates(*args, **inputs.kwargs())
    key = runner.choice.finalize_key
    if arch_for(inputs.device) in PERSIST_ARCHES and top_k == PERSIST_TOP_K:
        assert ":persist" in key, key
    eager = tuple(t.clone() for t in topk_with_candidates(*args, **inputs.kwargs()))
    plain = naive.cloned(topk(*args, **inputs.kwargs()))
    prepared = runner()
    assert not _structure_failures(*eager, ref.visible_rows(inputs).tolist())
    assert _same_as_plain(eager, plain), key
    assert _same_result(prepared, eager), key


def test_caller_owned_workspace_and_outputs():
    """A prepared runner on a caller-owned workspace of exactly the bound and caller-owned ``indices``, ``scores``
    and ``candidate_blocks`` runs without an allocation and returns the caller's tensors; a workspace one byte short
    of the call's plan is refused before anything runs, by the eager entry and by prepare."""
    _require_cb_program()
    inputs = ref.make_random_inputs(
        [48, 16], [20000, 900], top_k=TOP_K, seed=7797, device="cuda"
    )
    device = inputs.device
    T, Tkv, S = inputs.num_queries, inputs.num_keys, inputs.num_segments
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    nbytes = with_candidates_workspace_size(T, Tkv, S, top_k=TOP_K, device=device)
    need = plan_with_candidates(T, Tkv, S, top_k=TOP_K, device=device).workspace_bytes
    assert nbytes >= need
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    scores = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    blocks = torch.empty((T, CB_SLOTS), dtype=torch.int32, device=device)
    runner = prepare_with_candidates(
        *args,
        **inputs.kwargs(),
        workspace_buffer=workspace,
        indices=indices,
        scores=scores,
        candidate_blocks=blocks,
    )
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    out = runner()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, (
        "the prepared launch allocated"
    )
    assert [t.data_ptr() for t in out] == [
        indices.data_ptr(),
        scores.data_ptr(),
        blocks.data_ptr(),
    ]
    assert runner.candidate_blocks is blocks
    assert not _structure_failures(*out, ref.visible_rows(inputs).tolist())
    short = workspace[: need - 1]
    for call in (
        lambda: prepare_with_candidates(
            *args, **inputs.kwargs(), workspace_buffer=short
        ),
        lambda: topk_with_candidates(*args, **inputs.kwargs(), workspace_buffer=short),
    ):
        with pytest.raises(ValueError, match="workspace_buffer needs"):
            call()


def test_candidate_blocks_output_is_checked():
    """A caller-owned ``candidate_blocks`` must be an int32 ``[T, 2048]`` contiguous tensor on the device of ``q``;
    ``top_k`` above 2047, ``ratio != 1`` and the FP8 form are refused; all before any launch."""
    _require_cb_program()
    inputs = ref.make_random_inputs([32], [3000], top_k=TOP_K, seed=7798, device="cuda")
    T, device = inputs.num_queries, inputs.device
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    wide = torch.empty((T, 2 * CB_SLOTS), dtype=torch.int32, device=device)
    for bad, match in (
        (
            torch.empty((T, CB_SLOTS), dtype=torch.int64, device=device),
            "candidate_blocks must be",
        ),
        (
            torch.empty((T, CB_SLOTS - 1), dtype=torch.int32, device=device),
            "candidate_blocks must be",
        ),
        (
            torch.empty((T + 1, CB_SLOTS), dtype=torch.int32, device=device),
            "candidate_blocks must be",
        ),
        (wide[:, ::2], "candidate_blocks must be contiguous"),
        (torch.empty((T, CB_SLOTS), dtype=torch.int32), "candidate_blocks must be on"),
    ):
        for fn in (topk_with_candidates, prepare_with_candidates):
            with pytest.raises(ValueError, match=match):
                fn(*args, **inputs.kwargs(), candidate_blocks=bad)
    kwargs = inputs.kwargs()
    for change in ({"top_k": CB_MAX_TOP_K + 1}, {"ratio": 2}):
        with pytest.raises(ValueError):
            topk_with_candidates(*args, **{**kwargs, **change})
    with pytest.raises(ValueError, match="bfloat16"):
        topk_with_candidates(
            inputs.q.to(torch.float8_e4m3fn),
            inputs.k.to(torch.float8_e4m3fn),
            *args[2:],
            top_k=TOP_K,
        )
    good = topk_with_candidates(*args, **kwargs)
    assert good[2].dtype == torch.int32 and tuple(good[2].shape) == (T, CB_SLOTS)


def test_no_host_synchronization():
    """After a warm-up, one eager call and one run of a prepared runner complete under
    ``torch.cuda.set_sync_debug_mode("error")`` and return the warm-up's sets."""
    _require_cb_program()
    inputs = ref.make_random_inputs(
        [48, 300], [20000, 4000], top_k=TOP_K, seed=7799, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    expected = tuple(t.clone() for t in topk_with_candidates(*args, **inputs.kwargs()))
    runner = prepare_with_candidates(*args, **inputs.kwargs())
    runner()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = topk_with_candidates(*args, **inputs.kwargs())
        prepared = runner()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    for got in (eager, prepared):
        assert naive.same_ids(got[:2], expected[:2])
        assert torch.equal(cb.ascending(got[2]), cb.ascending(expected[2]))


def test_torch_deterministic_mode_is_not_consulted():
    """The deterministic package runs under ``torch.use_deterministic_algorithms(True)`` without a warning."""
    _require_cb_program()
    inputs = ref.make_random_inputs(
        [32], [17000], top_k=TOP_K, seed=7800, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    with _torch_deterministic(warn_only=False), warnings.catch_warnings():
        warnings.simplefilter("error")
        out = topk_with_candidates(*args, **inputs.kwargs())
        prepare_with_candidates(*args, **inputs.kwargs())()
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Rows that keep all their blocks (``iota``): the plain plan and the rows written after it
# ---------------------------------------------------------------------------

# the longest key segment whose rows keep all their blocks (2048 blocks of 8 keys)
IOTA_W = CB_SLOTS * cake_policy.CB_BLOCK
# exact layouts whose longest key segment holds at most 2048 blocks: default offsets (rows with 2041 .. 2048 visible
# blocks, n % 8 != 0, unequal query and key lengths, an empty query segment, a segment without keys) and explicit
# offsets (positive, beyond the keys, negative: rows without visible keys), packed beyond 16384 keys with
# ``max_seqlen_k`` 16384, and a call of 16384 keys without ``max_seqlen_k`` (an empty segment, a negative default
# offset)
IOTA_LAYOUTS = {
    "default_offsets": (
        (64, 40, 0, 16, 24),
        (IOTA_W, 300, 100, 0, 16376),
        None,
        IOTA_W,
    ),
    "explicit_offsets": (
        (48, 24, 16, 16),
        (IOTA_W, 9000, 500, 64),
        (16370, 9000, 600, -8),
        IOTA_W,
    ),
    "unbounded": ((40, 0, 8), (16377, 0, 7), None, None),
}
# plan geometries on both sides of the bound: (seg_q, seg_k, max_seqlen_k)
IOTA_PLAN_SHAPES = (
    ((16384,), (IOTA_W,), None),
    ((16384,), (IOTA_W,), IOTA_W),
    ((1,), (8,), None),
    ((64, 40, 0, 16, 24), (IOTA_W, 300, 100, 0, 16376), IOTA_W),
    ((4096,) * 4, (IOTA_W,) * 4, IOTA_W),
    ((2048,), (IOTA_W + 1,), None),
    ((2048, 2048), (IOTA_W + 1, 100), IOTA_W + 1),
    ((2048, 2048), (IOTA_W, 100), None),
)


def _scan_blocks_only(monkeypatch):
    """The dispatch of the package before ``iota``: every candidate-block call runs its ``scan_blocks`` program."""
    monkeypatch.setattr(
        cake_backend,
        "select_cb_program",
        functools.partial(cake_policy.select_cb_program, iota=False),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CB_ARCHES)
def test_iota_plan_is_the_plain_plan(arch):
    """Where the longest key segment ``W`` (``max_seqlen_k`` when given, else ``Tkv``) holds at most 2048 blocks the
    plan is the plain call's plan of the same arguments (its programs, split, pair, order, finalize and workspace) with
    ``iota`` set, no block entries and the plain workspace, at every ``top_k`` and in both sampling modes, and the
    workspace bound is the plain plans'; one block more runs the ``scan_blocks`` program with ``ceil(W / 8) - 1``
    entries per query slot."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for grid in GRIDS[arch]:
        for seg_q, seg_k, bound in IOTA_PLAN_SHAPES:
            T, Tkv, S = sum(seg_q), sum(seg_k), len(seg_q)
            nb = -(-(Tkv if bound is None else bound) // 8)
            where = (grid, seg_q, seg_k, bound)
            for top_k in (1, TOP_K, 1025, CB_MAX_TOP_K):
                kw = dict(top_k=top_k, arch=arch, num_sms=grid, max_seqlen_k=bound)
                plain = [
                    cake_backend.plan_dsa_indexer_det_topk(
                        T, Tkv, S, sample_tiles_max=sample, **kw
                    )
                    for sample in (None, 0)
                ]
                for sample, want in zip((None, 0), plain):
                    plan = plan_with_candidates(
                        T, Tkv, S, sample_tiles_max=sample, **kw
                    )
                    assert plan.iota == (nb <= CB_SLOTS), (*where, top_k)
                    assert plan.as_dict()["cb_iota"] == plan.iota
                    if plan.iota:
                        assert plan.choice == want, (*where, top_k, sample)
                        assert plan.keys == want.keys
                        assert (plan.blk_cap, plan.blk_bytes) == (0, 0)
                        assert plan.workspace_bytes == want.workspace_bytes
                    else:
                        assert plan.choice.scan_key.startswith(f"{CB_ROLE}:"), where
                        assert plan.blk_cap == nb - 1, where
                bound_bytes = with_candidates_workspace_size(T, Tkv, S, **kw)
                if nb <= CB_SLOTS:
                    assert bound_bytes == max(p.workspace_bytes for p in plain), where


@pytest.mark.parametrize("layout", sorted(IOTA_LAYOUTS))
def test_iota_rows_are_the_scan_blocks_programs_rows(layout, monkeypatch):
    """Exact inputs on layouts whose longest key segment holds at most 2048 blocks: the call runs the plain programs
    and then the rows (stage ``cb_iota``, no ``scan_blocks`` program); every row's ``indices`` and
    ``candidate_blocks`` equal the FP64 oracle's sets with the FP64 scores of the ids, a prepared runner returns the
    eager result, ``indices`` / ``scores`` are the plain call's, and the ``scan_blocks`` program of the same call (the
    dispatch before ``iota``) returns the same bits of all three outputs."""
    _require_cb_program()
    seg_q, seg_k, offsets, bound = IOTA_LAYOUTS[layout]
    seed = 7810 + sorted(IOTA_LAYOUTS).index(layout)
    q, k, w, cq, ck = cb.exact_inputs(
        "MXFP4-grid", seg_q, seg_k, seed=seed, device="cuda"
    )
    off = _offsets_tensor(offsets, q.device)
    extra = {} if bound is None else {"max_seqlen_k": bound}
    args = (q, k, w, cq, ck)
    kw = dict(top_k=TOP_K, softmax_scale=cb.EXACT_SCALE, q_causal_offsets=off, **extra)
    ns = _visible_counts(cq, ck, off)
    assert max((n + 7) // 8 for n in ns) == CB_SLOTS and min(ns) == 0
    runner = prepare_with_candidates(*args, **kw)
    assert runner.stages[0] == cake_policy.CB_IOTA_STAGE, runner.stages
    assert CB_ROLE not in runner.programs, runner.programs
    got = _call(q, k, w, cq, ck, offsets=off, **extra)
    failures = _structure_failures(*got, ns)
    failures += _exact_failures(q, k, w, cq, ck, off, got)
    if not _same_result(tuple(t.clone() for t in runner()), got):
        failures.append(
            "the prepared runner returns another result than the eager call"
        )
    if not _same_as_plain(got, naive.cloned(topk(*args, **kw))):
        failures.append("indices / scores are not the plain call's")
    _scan_blocks_only(monkeypatch)
    scan_runner = prepare_with_candidates(*args, **kw)
    assert CB_ROLE in scan_runner.programs, scan_runner.programs
    assert cake_policy.CB_IOTA_STAGE not in scan_runner.stages
    scan = _call(q, k, w, cq, ck, offsets=off, **extra)
    if not _same_result(scan, got):
        failures.append("the scan_blocks program returns another result")
    assert not failures, f"{layout}:\n" + "\n".join(failures[:20])


def test_iota_runner_follows_new_segments_without_allocation_or_sync():
    """A runner whose rows keep all their blocks, on a caller-owned workspace of the bound and caller-owned outputs: a
    run allocates nothing and completes under ``torch.cuda.set_sync_debug_mode("error")``; captured in a CUDA graph,
    a replay rewrites the rows, and after new segment boundaries and new ``q`` / ``k`` / ``w`` values are written into
    the bound tensors (the same ``T``, ``Tkv`` and ``S``) a replay returns the eager call's result for the new
    values."""
    _require_cb_program()
    seg_q, seg_k, _, bound = IOTA_LAYOUTS["default_offsets"]
    new_q, new_k = (24, 0, 80, 16, 24), (16376, 100, 300, IOTA_W, 0)
    assert sum(new_q) == sum(seg_q) and sum(new_k) == sum(seg_k)
    q, k, w, cq, ck = cb.exact_inputs(
        "MXFP4-grid", seg_q, seg_k, seed=7820, device="cuda"
    )
    T, Tkv, S = int(q.shape[0]), int(k.shape[0]), len(seg_q)
    device = q.device
    kw = dict(top_k=TOP_K, softmax_scale=cb.EXACT_SCALE, max_seqlen_k=bound)
    nbytes = with_candidates_workspace_size(
        T, Tkv, S, top_k=TOP_K, max_seqlen_k=bound, device=device
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    scores = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    blocks = torch.full((T, CB_SLOTS), 7, dtype=torch.int32, device=device)
    runner = prepare_with_candidates(
        q,
        k,
        w,
        cq,
        ck,
        **kw,
        workspace_buffer=workspace,
        indices=indices,
        scores=scores,
        candidate_blocks=blocks,
    )
    assert runner.stages[0] == cake_policy.CB_IOTA_STAGE, runner.stages
    expected = _call(q, k, w, cq, ck, max_seqlen_k=bound)
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = runner()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, (
        "the prepared launch allocated"
    )
    assert [t.data_ptr() for t in out] == [
        indices.data_ptr(),
        scores.data_ptr(),
        blocks.data_ptr(),
    ]
    assert _same_result(tuple(t.clone() for t in out), expected)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        runner()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner()
    blocks.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    assert _same_result((indices.clone(), scores.clone(), blocks.clone()), expected)
    fresh = cb.exact_inputs("MXFP4-grid", new_q, new_k, seed=7821, device="cuda")
    for dst, src in zip((q, k, w, cq, ck), fresh):
        dst.copy_(src)
    blocks.fill_(7)
    graph.replay()
    torch.cuda.synchronize()
    got = (indices.clone(), scores.clone(), blocks.clone())
    want = _call(*fresh, max_seqlen_k=bound)
    failures = _structure_failures(*got, _visible_counts(cq, ck, None))
    if not _same_result(got, want):
        failures.append(
            "the replay after new segments returns another result than the eager call"
        )
    assert not failures, "\n".join(failures[:20])


def _iota_reference(T, cq, ck, offsets) -> torch.Tensor:
    """The candidate-block rows of the kernel rule on the host: row ``t`` of segment ``s`` (query position ``u``, ``L`` keys,
    offset ``offset`` or ``L - Lq``) sees ``n = clamp(offset + u + 1, 0, L)`` keys and holds ``0 .. ceil(n / 8) - 1`` and a
    ``-1`` tail; a row outside every query segment is all ``-1``."""
    cq, ck = [int(v) for v in cq.tolist()], [int(v) for v in ck.tolist()]
    off = None if offsets is None else [int(v) for v in offsets.tolist()]
    rows = torch.full((T, CB_SLOTS), -1, dtype=torch.int32)
    for s in range(len(cq) - 1):
        lq, lk = cq[s + 1] - cq[s], ck[s + 1] - ck[s]
        base = lk - lq if off is None else off[s]
        for u in range(lq):
            n = min(max(base + u + 1, 0), lk)
            nb = (n + 7) // 8
            rows[cq[s] + u, :nb] = torch.arange(nb, dtype=torch.int32)
    return rows


def test_iota_fill_rows_outside_every_query_segment():
    """The ``cb_iota`` fill on its own: rows before ``cu_seqlens_q[0]`` and from ``cu_seqlens_q[S]`` on are all ``-1``
    (with and without causal offsets, an empty query segment and a segment without keys), the rows of the segments
    follow the kernel rule, and a call without segments fills every row with ``-1``."""
    _require_cb_program()
    device = torch.device("cuda", torch.cuda.current_device())
    T = 64
    cq = torch.tensor([5, 12, 12, 30, 41], dtype=torch.int32, device=device)
    ck = torch.tensor([0, 100, 140, 16000, 16000], dtype=torch.int32, device=device)
    for offsets in (
        None,
        torch.tensor([90, -3, 15900, 4], dtype=torch.int64, device=device),
    ):
        blocks = torch.full((T, CB_SLOTS), 7, dtype=torch.int32, device=device)
        fill = cake_backend._IotaBlocks(blocks, cq, ck, offsets, 4, device.index)
        fill()
        torch.cuda.synchronize()
        want = _iota_reference(T, cq, ck, offsets)
        assert torch.equal(blocks.cpu(), want), offsets
        assert (blocks[:5] == -1).all() and (blocks[41:] == -1).all()
    blocks = torch.full((T, CB_SLOTS), 7, dtype=torch.int32, device=device)
    empty = torch.zeros(1, dtype=torch.int32, device=device)
    cake_backend._IotaBlocks(blocks, empty, empty, None, 0, device.index)()
    torch.cuda.synchronize()
    assert (blocks == -1).all()


def test_iota_table_made_on_a_busy_stream_is_waited_for():
    """The per-device row table is made on the stream of its first prepare; a fill prepared and run on another stream
    while that stream still holds the table's writes (behind a long-running kernel) waits for them, so its rows are the
    kernel rule's; the table's event is dropped once it has completed at prepare."""
    _require_cb_program()
    device = torch.device("cuda", torch.cuda.current_device())
    T = 4096
    cq = torch.tensor([0, T], dtype=torch.int32, device=device)
    ck = torch.tensor([0, 16384], dtype=torch.int32, device=device)
    want = _iota_reference(T, cq, ck, None)
    saved = cake_backend._IOTA_ROWS.pop(device.index, None)
    try:
        busy, other = torch.cuda.Stream(device), torch.cuda.Stream(device)
        a = torch.full((T, CB_SLOTS), 7, dtype=torch.int32, device=device)
        b = torch.full((T, CB_SLOTS), 7, dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        with torch.cuda.stream(busy):
            torch.cuda._sleep(1 << 30)
            first = cake_backend._IotaBlocks(a, cq, ck, None, 1, device.index)
        with torch.cuda.stream(other):
            second = cake_backend._IotaBlocks(b, cq, ck, None, 1, device.index)
            assert (
                second.ready is not None
            )  # the table's writes were still queued behind the busy stream
            second()
        with torch.cuda.stream(busy):
            first()
        torch.cuda.synchronize()
        assert torch.equal(b.cpu(), want) and torch.equal(a.cpu(), want)
        third = cake_backend._IotaBlocks(a, cq, ck, None, 1, device.index)
        assert third.ready is None
    finally:
        if saved is not None:
            cake_backend._IOTA_ROWS[device.index] = saved


def test_iota_call_with_a_max_seqlen_k_below_a_key_segment_stops_before_its_programs():
    """A call whose ``max_seqlen_k`` holds at most 2048 blocks takes the plain plan with the ``cb_iota`` fill first;
    a key segment longer than the bound gives rows past 2048 blocks, and the fill stops the call with a device-side
    assert before its programs run (a fresh interpreter: the CUDA context is unusable afterwards)."""
    _require_cb_program()
    seg_q, seg_k = (16, 16), (20000, 100)
    T, Tkv = sum(seg_q), sum(seg_k)
    plan = plan_with_candidates(
        T,
        Tkv,
        len(seg_q),
        top_k=TOP_K,
        max_seqlen_k=IOTA_W,
        device=torch.device("cuda"),
    )
    assert plan.iota
    code = f"""
import importlib, json, sys
import torch
backend = importlib.import_module({cake_backend.__name__!r})
fn = getattr(backend, {topk_with_candidates.__name__!r})
g = torch.Generator(device="cuda").manual_seed(7830)
q = torch.randn(({T}, 32, 128), generator=g, device="cuda").to(torch.bfloat16)
k = torch.randn(({Tkv}, 128), generator=g, device="cuda").to(torch.bfloat16)
w = torch.randn(({T}, 32), generator=g, device="cuda")
cq = torch.tensor([0, {seg_q[0]}, {T}], dtype=torch.int32, device="cuda")
ck = torch.tensor([0, {seg_k[0]}, {Tkv}], dtype=torch.int32, device="cuda")
runner = backend.{prepare_with_candidates.__name__}(q, k, w, cq, ck, top_k={TOP_K}, max_seqlen_k={IOTA_W})
print(json.dumps({{"stages": list(runner.stages)}}), flush=True)
try:
    fn(q, k, w, cq, ck, top_k={TOP_K}, max_seqlen_k={IOTA_W})
    torch.cuda.synchronize()
except RuntimeError as exc:
    print("RAISED", type(exc).__name__, str(exc).splitlines()[0], flush=True)
    sys.exit(3)
print("COMPLETED", flush=True)
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    lines = out.stdout.strip().splitlines()
    assert lines and lines[0].startswith("{"), (
        out.returncode,
        out.stdout[-2000:],
        out.stderr[-2000:],
    )
    assert json.loads(lines[0])["stages"][0] == cake_policy.CB_IOTA_STAGE
    assert out.returncode != 0 and "COMPLETED" not in out.stdout, (
        out.stdout[-2000:],
        out.stderr[-2000:],
    )
    text = out.stdout + out.stderr
    assert "device-side assert" in text or "srcIndex < srcSelectDimSize" in text, text[
        -3000:
    ]
