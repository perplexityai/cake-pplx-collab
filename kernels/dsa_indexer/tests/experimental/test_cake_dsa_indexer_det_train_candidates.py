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
host plan at the request's shapes, the call's argument checks and the refusal of an architecture without the programs
(CPU), then on an SM100 / SM103 device the exact sets against the FP64 oracle
(``tests/experimental/test_helpers/cake_dsa_indexer_candidates.py``) on exact MXFP4-grid inputs, the Gaussian rule,
the containment of every id in a returned block, bitwise repetition and the equality with the plain top-k call's bits, the caller-owned workspace and outputs, and the absence of
host synchronization.
"""

import contextlib
import math
import random
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
CB_ARCHES = ("sm_100a", "sm_103a")
# the persistent grids (SM counts) of the devices of each architecture
GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160)}
CB_SLOTS_PLAN = [
    ["buffer", "CandBlk"],
    ["buffer", "CandBlocks"],
    ["parameter", "blk_cap"],
]
CB_KEYS = sorted(
    f"{CB_ROLE}:{kind}:u1:s0:f{fit}" for kind in ("l6", "narrow") for fit in (0, 1)
)
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
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")
    if arch_for(torch.device("cuda")) not in CB_ARCHES:
        pytest.skip(
            "the candidate-block programs are registered for sm_100a and sm_103a only"
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
    ``CandBlocks`` and ``blk_cap`` right before ``K_scale``; sm_100a and sm_103a register the unpaired, unsplit BF16
    narrow and l6 programs of both fit forms under ``scan_blocks:`` keys, each a program of its own beside the scan
    program of the same unit; sm_107a registers none."""
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
        assert cb_keys == CB_KEYS, arch
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
    """Per grid of the architecture: the plan runs a registered ``scan_blocks`` program unsplit and unpaired; the
    block entries are ``ceil(W / 8) - 1`` per query slot when ``ceil(W / 8) > 2048`` (``W``: ``max_seqlen_k`` when
    given, else ``Tkv``), else none; the call's workspace is the scan's (candidate buffers and ticket tail) plus the
    block entries; the bound is the larger plan of the two sampling modes."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    seg_q, seg_k = SHAPES[shape]
    T, Tkv, S = sum(seg_q), sum(seg_k), len(seg_q)
    msk = max(seg_k) if bound == "max_seqlen_k" else None
    W = Tkv if msk is None else msk
    for grid in GRIDS[arch]:
        kw = dict(top_k=TOP_K, arch=arch, num_sms=grid, max_seqlen_k=msk)
        plan = plan_with_candidates(T, Tkv, S, **kw)
        assert isinstance(plan, CandidateBlocksChoice)
        c = plan.choice
        assert c.scan_key in CB_KEYS and c.scan_key in cake_jit.PROGRAM_KEYS[arch], (
            grid,
            c.scan_key,
        )
        assert c.n_split == 1 and not c.split and not c.pair and c.merge_key is None, (
            grid
        )
        assert (
            c.workspace_bytes
            == c.grid_ctas * c.block_q * c.cand_cap * 8 + TICKET_TAIL_BYTES
        ), grid
        nb = -(-W // 8)
        assert plan.blk_cap == (nb - 1 if nb > CB_SLOTS else 0), grid
        assert plan.blk_bytes == c.grid_ctas * c.block_q * plan.blk_cap * 8, grid
        assert plan.workspace_bytes == c.workspace_bytes + plan.blk_bytes, grid
        assert plan.blocks_at == c.workspace_bytes, grid
        unsampled = plan_with_candidates(T, Tkv, S, sample_tiles_max=0, **kw)
        bound_bytes = with_candidates_workspace_size(T, Tkv, S, **kw)
        assert bound_bytes == max(plan.workspace_bytes, unsampled.workspace_bytes), grid


@pytest.mark.cpu_only
def test_architecture_without_candidate_block_programs_is_refused():
    """sm_107a registers no ``scan_blocks`` program: the plan and the workspace bound raise ``NotImplementedError``
    (the plain call's plan is unaffected)."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for fn in (plan_with_candidates, with_candidates_workspace_size):
        with pytest.raises(NotImplementedError):
            fn(16384, 262144, 1, top_k=TOP_K, arch="sm_107a", num_sms=212)
    plain = cake_backend.plan_dsa_indexer_det_topk(
        16384, 262144, 1, top_k=TOP_K, arch="sm_107a", num_sms=212
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
