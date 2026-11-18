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

Backend tests of the candidate-restricted top-k of the non-deterministic Cake DSA indexer
(``cake_dsa_indexer_nondet_train_topk_in_candidates``, ``prepare_dsa_indexer_nondet_topk_in_candidates``,
``plan_dsa_indexer_nondet_topk_in_candidates`` and the workspace bound): the registry's ``cand`` program, the host plan
and its workspace bound, the argument checks and the refusal of an architecture without the program (CPU), then on an
SM100 / SM103 device the exact sets against the FP64 oracle
(``tests/experimental/test_helpers/cake_dsa_indexer_candidates.py``) on exact MXFP4-grid inputs with source-layer and
synthetic candidate rows, the unrestricted sets from the source layer's own candidates and from every visible block,
keys outside the pool that never count, the Gaussian rule, the repetition of the sets and the alert under torch's deterministic algorithms, the caller-owned workspace and outputs, and the
absence of host synchronization.
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

cake_backend = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_backend")
cake_jit = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_jit")
cake_policy = cake_package("cake.cake_dsa_indexer_nondet_train", "cake_policy")
CAND_ROLE = cake_policy.CAND_ROLE
CAND_KEY = cake_policy.CAND_KEY
CAND_WIDTH_MAX = cake_policy.CAND_WIDTH_MAX
CAND_MAX_TOP_K = cake_policy.CAND_MAX_TOP_K
CAND_LIST_BYTES = cake_policy.CAND_LIST_BYTES
CAND_TICKET_BYTES = cake_policy.CAND_TICKET_BYTES
CandidateTopKChoice = cake_policy.CandidateTopKChoice
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
generated_program_available = cake_backend.generated_program_available
arch_for = cake_backend.arch_for
topk = cake_backend.cake_dsa_indexer_nondet_train_topk
topk_with_candidates = cake_backend.cake_dsa_indexer_nondet_train_topk_with_candidates
in_candidates = cake_backend.cake_dsa_indexer_nondet_train_topk_in_candidates
plan_in_candidates = cake_backend.plan_dsa_indexer_nondet_topk_in_candidates
prepare_in_candidates = cake_backend.prepare_dsa_indexer_nondet_topk_in_candidates
in_candidates_workspace_size = (
    cake_backend.cake_dsa_indexer_nondet_train_topk_in_candidates_workspace_size
)

ORDERED = False
TOP_K = 512
CAND_ARCHES = ("sm_100a", "sm_103a")
# the persistent grids (SM counts) of the devices of each architecture
GRIDS = {"sm_100a": (148,), "sm_103a": (148, 152, 160)}
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
# the synthetic candidate rows (``cb.synthetic_rows``): random subsets in shuffled order with -1 scattered, ids past
# the visible blocks and past the segment's keys, the newest (maybe partial) block, fewer candidate keys than top_k,
# no candidates, every visible block
KINDS = ("subset", "past", "newest", "few", "empty", "all")


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_cand_program():
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not generated_program_available(torch.device("cuda")):
        pytest.skip("generated DSA indexer programs not registered for this device")
    if arch_for(torch.device("cuda")) not in CAND_ARCHES:
        pytest.skip(
            "the candidate-restricted program is registered for sm_100a and sm_103a only"
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


def _visible_counts(cq, ck, offsets) -> list[int]:
    segs = cb.segments(cq.cpu(), ck.cpu(), None if offsets is None else offsets.cpu())
    return [cb.visible(u, off, Lk) for _, Lq, _, Lk, off in segs for u in range(Lq)]


def _call(q, k, w, cq, ck, cand, *, offsets=None, scale=cb.EXACT_SCALE, **extra):
    out = in_candidates(
        q,
        k,
        w,
        cq,
        ck,
        cand,
        top_k=extra.pop("top_k", TOP_K),
        softmax_scale=scale,
        q_causal_offsets=offsets,
        **extra,
    )
    assert isinstance(out, tuple) and len(out) == 2
    return tuple(t.clone() for t in out)


def _pool_sizes(cand, ns) -> torch.Tensor:
    """``|P_t|`` per row: the visible keys of the distinct blocks ``0 <= b < ceil(n / 8)`` of ``cand[t]``."""
    n = torch.tensor(ns, dtype=torch.int64, device=cand.device)[:, None]
    b = (
        torch.where(cand >= 0, cand.long(), torch.iinfo(torch.int64).max)
        .sort(dim=1)
        .values
    )
    first = torch.ones_like(b, dtype=torch.bool)
    first[:, 1:] = b[:, 1:] != b[:, :-1]
    valid = first & (b < (n + 7) // 8)
    keys = (n - 8 * torch.where(valid, b, 0)).clamp(max=8)
    return torch.where(valid, keys, 0).sum(dim=1)


def _structure_failures(indices, scores, cand, ns) -> list[str]:
    """The structure every row keeps on any input: ``min(top_k, |P_t|)`` unique ids, each a visible key of a block
    the row names, ``-1`` elsewhere, ``-inf`` scores exactly on the padding, as sets: slot order free."""
    out = []
    ids = indices.long()
    valid = ids >= 0
    want = _pool_sizes(cand, ns).to(ids.device).clamp(max=ids.shape[1])
    if not torch.equal(valid.sum(1), want):
        out.append("indices: valid counts differ from min(top_k, |pool|)")
    ns_t = torch.tensor(ns, dtype=torch.int64, device=ids.device)
    if bool(((ids < -1) | (valid & (ids >= ns_t[:, None]))).any()):
        out.append("indices: an id outside [-1, visible)")
    s = cb.ascending(ids)
    if bool(((s[:, 1:] == s[:, :-1]) & (s[:, 1:] >= 0)).any()):
        out.append("indices: a repeated id")
    if ORDERED and not torch.equal(s, ids):
        out.append("indices: a row not in ascending order with a -1 tail")
    if bool((valid != ~torch.isneginf(scores)).any()):
        out.append("scores: -inf exactly on the padding slots")
    if not bool(cb.contained(indices, cand).all()):
        out.append("an id outside the row's candidate blocks")
    return out


def _exact_failures(q, k, w, cq, ck, offsets, cand, got, *, rows=None) -> list[str]:
    """``got = (indices, scores)`` of exact inputs against the FP64 oracle on ``rows`` (default all): the sets
    exactly (as sets: slot order free), the scores equal to the FP64 scores of the ids (exact on these inputs)."""
    indices, scores = got
    g, ids64 = cb.in_candidates(
        q,
        k,
        w,
        cq,
        ck,
        cand,
        softmax_scale=cb.EXACT_SCALE,
        top_k=indices.shape[1],
        q_causal_offsets=offsets,
        rows=rows,
    )
    gi, gs = indices[g].long(), scores[g]
    if not ORDERED:
        big = torch.iinfo(torch.int64).max
        order = torch.where(gi >= 0, gi, big).argsort(1)
        gi, gs = gi.gather(1, order), gs.gather(1, order)
    out = []
    if not torch.equal(gi, ids64):
        bad = (gi != ids64).any(1).nonzero().flatten()[:5].tolist()
        out.append(
            f"indices differ from the FP64 oracle in rows {[int(g[r]) for r in bad]}"
        )
    want = _scores64_of_ids(q, k, w, cq, ck, g, gi)
    if not torch.equal(gs.double(), want):
        out.append("scores differ from the FP64 scores of the ids")
    return out


def _scores64_of_ids(q, k, w, cq, ck, rows, ids) -> torch.Tensor:
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
def test_registry_ships_the_candidate_restricted_program():
    """``cand`` is the last role, after the family's roles and ``scan_blocks``; its argument plan takes ``Kraw``, the
    caller's ``CandBlocks``, the block lists ``Lists``, the candidate buffers ``Cand`` and its own ``TicketWord``
    once, and neither ``K_scale`` nor a zeroed region; sm_100a and sm_103a route the key ``cand`` to one program of
    role ``cand``; sm_107a routes none."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    assert cake_jit.STAGES[-1] == CAND_ROLE and cake_jit.STAGES.count(CAND_ROLE) == 1
    plan = cake_jit.ARG_PLANS[CAND_ROLE]
    names = [name for _, name in plan]
    assert names.count("TicketWord") == 1 and "K_scale" not in names
    assert not any(kind == "zero" for kind, _ in plan)
    for slot in (
        ["buffer", "Kraw"],
        ["buffer", "CandBlocks"],
        ["buffer", "Lists"],
        ["buffer", "Cand"],
    ):
        assert slot in plan, slot
    assert plan[-3:] == [["grid", f"grid_{axis}"] for axis in "xyz"]
    for arch, keys in cake_jit.PROGRAM_KEYS.items():
        routed = sorted(
            key for key in keys if key == CAND_KEY or key.startswith(f"{CAND_ROLE}:")
        )
        if arch not in CAND_ARCHES:
            assert routed == [], arch
            continue
        assert routed == [CAND_KEY], arch
        record = cake_jit.PROGRAMS[keys[CAND_KEY]]
        assert record["role"] == CAND_ROLE and arch in record["arches"], arch

    for name, record in cake_jit.PROGRAMS.items():
        if record["role"] == CAND_ROLE:
            assert set(record["arches"]) <= set(CAND_ARCHES), name


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", CAND_ARCHES)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_plan_and_workspace_bound(arch, shape):
    """Per grid of the architecture: one ``cand`` program on ``grid`` persistent CTAs for every shape, ``cand_cap =
    max(4096, 4 top_k)`` entries per unit query and a check period that keeps ``cand_cap >= top_k + 128 x
    check_period``; the workspace is the candidate buffers, the block lists and the 16-byte ticket word, the same
    for every ``T``, ``Tkv`` and segment layout; the bound equals the plan's."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    seg_q, seg_k = SHAPES[shape]
    T, Tkv, S = sum(seg_q), sum(seg_k), len(seg_q)
    for grid in GRIDS[arch]:
        plan = plan_in_candidates(T, Tkv, S, top_k=TOP_K, arch=arch, num_sms=grid)
        assert isinstance(plan, CandidateTopKChoice)
        assert plan.cand_key == CAND_KEY and plan.grid_ctas == grid
        assert (
            plan.cand_cap == max(4096, 4 * TOP_K)
            and plan.cand_cap >= TOP_K + 128 * plan.check_period >= TOP_K + 128
        )
        assert (
            plan.cand_bytes == grid * 4 * plan.cand_cap * 8
            and plan.lists_at == plan.cand_bytes
        )
        assert plan.lists_bytes == grid * CAND_LIST_BYTES == grid * 65536
        assert plan.ticket_at == plan.cand_bytes + plan.lists_bytes
        assert plan.workspace_bytes == plan.ticket_at + CAND_TICKET_BYTES
        other = plan_in_candidates(1, 8, 1, top_k=TOP_K, arch=arch, num_sms=grid)
        assert other.workspace_bytes == plan.workspace_bytes
        assert (
            in_candidates_workspace_size(
                T, Tkv, S, top_k=TOP_K, arch=arch, num_sms=grid
            )
            == plan.workspace_bytes
        )
        assert plan.keys == (CAND_KEY,)


@pytest.mark.cpu_only
def test_architecture_without_the_program_is_refused():
    """sm_107a registers no ``cand`` program: the plan and the workspace bound raise ``NotImplementedError``."""
    if not cake_jit.PROGRAM_KEYS:
        pytest.skip("placeholder registry")
    for fn in (plan_in_candidates, in_candidates_workspace_size):
        with pytest.raises(NotImplementedError):
            fn(16384, 262144, 1, top_k=TOP_K, arch="sm_107a", num_sms=212)


@pytest.mark.cpu_only
def test_plan_arguments_are_checked():
    """``1 <= top_k <= 2048``, ``ratio == 1``, ``0 <= candidate_width <= 2048`` and an explicit ``cand_cap`` /
    ``check_period`` pair that keeps ``cand_cap >= top_k + 128 x check_period``, before any program is chosen."""
    kw = dict(arch="sm_100a", num_sms=148)
    for top_k in (0, CAND_MAX_TOP_K + 1, 4096):
        with pytest.raises(ValueError):
            plan_in_candidates(64, 20000, 1, top_k=top_k, **kw)
    for ratio in (2, 4):
        with pytest.raises(ValueError, match="ratio == 1"):
            plan_in_candidates(64, 20000, 1, top_k=TOP_K, ratio=ratio, **kw)
    for width in (-1, CAND_WIDTH_MAX + 1):
        with pytest.raises(ValueError):
            plan_in_candidates(64, 20000, 1, top_k=TOP_K, candidate_width=width, **kw)
    if not cake_jit.PROGRAM_KEYS:
        return
    with pytest.raises(ValueError, match="cand_cap"):
        plan_in_candidates(
            64, 20000, 1, top_k=TOP_K, cand_cap=TOP_K + 127, check_period=1, **kw
        )
    plan = plan_in_candidates(
        64, 20000, 1, top_k=TOP_K, cand_cap=TOP_K + 128, check_period=1, **kw
    )
    assert (plan.cand_cap, plan.check_period) == (TOP_K + 128, 1)
    assert CAND_MAX_TOP_K == CAND_WIDTH_MAX == 2048


# ---------------------------------------------------------------------------
# Device: exact sets, unrestricted equivalence, Gaussian rule, sets, workspace and outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", cb.EXACT_CASES)
def test_exact_sets_match_the_fp64_oracle(case, kind, layout):
    """On exact inputs (MXFP4-grid ``q`` / ``k``, ``w`` in ``{-4..4} / 8``, ``softmax_scale`` 1/16) every row's
    ``indices`` equal the FP64 oracle's top ``min(top_k, |P_t|)`` of its pool (as sets: slot order free) and the scores equal the
    FP64 scores of the ids, for every kind of candidate row (shuffled subsets with ``-1`` scattered, ids past the
    visible blocks and past the segment's keys, the partial newest block, fewer than ``top_k`` candidate keys, no
    candidates, every visible block) and every case (distinct scores, ties at the cutoff, all-equal and signed-zero
    scores where the newest keys win, negative scores); rows with fewer than and more than 2048 visible blocks,
    ``n % 8 != 0``, unequal query and key lengths, positive / negative offsets, empty segments and rows without
    visible keys."""
    _require_cand_program()
    seg_q, seg_k, offsets = LAYOUTS[layout]
    seed = (
        7820
        + 31 * cb.EXACT_CASES.index(case)
        + 7 * KINDS.index(kind)
        + (0 if offsets is None else 1)
    )
    q, k, w, cq, ck = cb.exact_inputs(case, seg_q, seg_k, seed=seed, device="cuda")
    off = _offsets_tensor(offsets, q.device)
    ns = _visible_counts(cq, ck, off)
    cand = cb.synthetic_rows(ns, CAND_WIDTH_MAX, kind, seed=seed, device=q.device)
    got = _call(q, k, w, cq, ck, cand, offsets=off)
    failures = _structure_failures(*got, cand, ns)
    failures += _exact_failures(q, k, w, cq, ck, off, cand, got)
    assert not failures, f"{case} / {kind} / {layout}:\n" + "\n".join(failures[:20])


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("case", ("MXFP4-grid", "duplicated keys", "all scores zero"))
def test_source_candidates_and_every_block_give_the_unrestricted_sets(case, layout):
    """On exact inputs: the call with the source layer's own ``q``, ``w`` and the candidate blocks of
    ``*_topk_with_candidates`` returns that layer's unrestricted top-512 sets (every row), and with every visible
    block as a candidate (rows of at most 2048 blocks) the unrestricted sets too; another layer's ``q`` and ``w``
    against the source layer's ``k`` and candidates give the FP64 oracle's restricted sets."""
    _require_cand_program()
    seg_q, seg_k, offsets = LAYOUTS[layout]
    seed = (
        7900
        + 11 * ("MXFP4-grid", "duplicated keys", "all scores zero").index(case)
        + (0 if offsets is None else 1)
    )
    q, k, w, cq, ck = cb.exact_inputs(case, seg_q, seg_k, seed=seed, device="cuda")
    off = _offsets_tensor(offsets, q.device)
    ns = _visible_counts(cq, ck, off)
    plain_ids, plain_scores, source = (
        t.clone()
        for t in topk_with_candidates(
            q,
            k,
            w,
            cq,
            ck,
            top_k=TOP_K,
            softmax_scale=cb.EXACT_SCALE,
            q_causal_offsets=off,
        )
    )
    failures = []
    got = _call(q, k, w, cq, ck, source, offsets=off)
    if not torch.equal(cb.ascending(got[0].long()), cb.ascending(plain_ids.long())):
        failures.append(
            "the source layer's own candidates: indices differ from the unrestricted sets"
        )
    every = cb.synthetic_rows(ns, CAND_WIDTH_MAX, "all", seed=seed, device=q.device)
    small = torch.tensor([(n + 7) // 8 <= CAND_WIDTH_MAX for n in ns], device=q.device)
    got_all = _call(q, k, w, cq, ck, every, offsets=off)
    same = (cb.ascending(got_all[0].long()) == cb.ascending(plain_ids.long())).all(1)
    if not bool(same[small].all()):
        failures.append(
            "every visible block: indices differ from the unrestricted sets"
        )
    q2, _, w2, _, _ = cb.exact_inputs(case, seg_q, seg_k, seed=seed + 5, device="cuda")
    other = _call(q2, k, w2, cq, ck, source, offsets=off)
    failures += _structure_failures(*other, source, ns)
    failures += _exact_failures(q2, k, w2, cq, ck, off, source, other)
    assert not failures, f"{case} / {layout}:\n" + "\n".join(failures[:20])


def test_keys_outside_the_pool_never_count():
    """Keys a row does not see count for nothing even where its candidate row names their blocks: one query per
    segment whose offset leaves the newest named block partial and later named blocks invisible, the invisible keys
    set to 6.0 (scores above most visible keys if they were scored), pools larger than ``top_k`` where the segment
    allows; the sets equal the FP64 oracle's."""
    _require_cand_program()
    seg_q, seg_k = (1, 1, 1, 1), (4099, 70, 20000, 13)
    offsets = (3001, 60, 17003, 4)
    q, k, w, cq, ck = cb.exact_inputs(
        "MXFP4-grid", seg_q, seg_k, seed=7950, device="cuda"
    )
    w = w.abs()
    off = _offsets_tensor(offsets, q.device)
    ns = _visible_counts(cq, ck, off)
    k = k.clone()
    for (q0, Lq, k0, Lk, _), n in zip(cb.segments(cq.cpu(), ck.cpu(), off.cpu()), ns):
        k[k0 + n : k0 + Lk] = 6.0
    cand = torch.full((len(ns), CAND_WIDTH_MAX), -1, dtype=torch.int32)
    for r, n in enumerate(ns):
        nb, past = (n + 7) // 8, (seg_k[r] + 7) // 8
        ids = list(range(max(0, nb - 300), past))[:CAND_WIDTH_MAX]
        cand[r, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
    cand = cand.to(q.device)
    got = _call(q, k, w, cq, ck, cand, offsets=off)
    failures = _structure_failures(*got, cand, ns)
    failures += _exact_failures(q, k, w, cq, ck, off, cand, got)
    assert not failures, "\n".join(failures[:20])


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_request_shapes_on_sampled_rows(shape):
    """The request's shapes on exact inputs with full candidate rows from the source layer (``*_topk_with_candidates``
    of another ``q`` / ``w`` over the same ``k``): every row keeps the structure, the sets of sampled rows (the first
    and last rows of every segment and random ones) equal the FP64 oracle's, and the call with ``max_seqlen_k``
    returns the same sets."""
    _require_cand_program()
    seg_q, seg_k = SHAPES[shape]
    q, k, w, cq, ck = cb.exact_inputs(
        "MXFP4-grid", seg_q, seg_k, seed=7960, device="cuda"
    )
    qs, _, ws, _, _ = cb.exact_inputs(
        "MXFP4-grid", seg_q, seg_k, seed=7961, device="cuda"
    )
    source = topk_with_candidates(
        qs, k, ws, cq, ck, top_k=TOP_K, softmax_scale=cb.EXACT_SCALE
    )[2].clone()
    del qs, ws
    got = _call(q, k, w, cq, ck, source)
    bounded = _call(q, k, w, cq, ck, source, max_seqlen_k=max(seg_k))
    ns = _visible_counts(cq, ck, None)
    failures = _structure_failures(*got, source, ns)
    rng = random.Random(7962)
    rows = set()
    for q0, Lq, _, _, _ in cb.segments(cq.cpu(), ck.cpu()):
        rows |= {q0, q0 + 1, q0 + Lq - 2, q0 + Lq - 1} if Lq > 1 else {q0}
        rows |= {q0 + rng.randrange(Lq) for _ in range(4)} if Lq else set()
    failures += _exact_failures(q, k, w, cq, ck, None, source, got, rows=sorted(rows))
    if not torch.equal(cb.ascending(got[0].long()), cb.ascending(bounded[0].long())):
        failures.append("the call with max_seqlen_k returns other sets")
    assert not failures, f"{shape}:\n" + "\n".join(failures[:20])


def test_gaussian_scores_and_sets(record_property):
    """Gaussian inputs with synthetic candidate rows (random subsets of the visible blocks in shuffled order): the
    scores are judged against FP64 beside the BF16 naive tier (the family's rule), and each row's set may differ from
    the FP64 restricted set only in members whose FP64 score lies within twice the draw's largest score error of the
    FP64 cutoff of its pool."""
    _require_cand_program()
    state = {}

    def make(seed):
        return ref.make_random_inputs(
            [24, 8], [17000, 900], top_k=TOP_K, seed=seed, device="cuda"
        )

    def run(inputs):
        ns = ref.visible_rows(inputs).tolist()
        cand = cb.synthetic_rows(
            ns, CAND_WIDTH_MAX, "subset", seed=7970, device=inputs.device
        )
        state["cand"], state["ns"] = cand, ns
        return in_candidates(
            inputs.q,
            inputs.k,
            inputs.w,
            inputs.cu_seqlens_q,
            inputs.cu_seqlens_k,
            cand,
            **inputs.kwargs(),
        )

    def check(j, inputs, result, _):
        indices, scores = result
        cand, ns = state["cand"], state["ns"]
        failures = _structure_failures(indices, scores, cand, ns)
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
            pool = sorted(
                {
                    8 * int(b) + i
                    for b in cand[r].tolist()
                    if b >= 0
                    for i in range(8)
                    if 8 * int(b) + i < n
                }
            )
            rows.append((r, s.tolist(), pool))
        for r, vals, pool in rows:
            got = {int(x) for x in indices[r].tolist() if x >= 0}
            if not cb.near_cutoff_only(vals, pool, got, TOP_K, e):
                failures.append(
                    f"draw {j} row {r}: indices beyond the near-cutoff rule"
                )
        assert not failures, "\n".join(failures[:20])

    naive.judge_gaussian(
        record_property, make, run, seed=7975, ordered=ORDERED, check=check
    )


def test_repeated_calls():
    """Eager calls and runs of a prepared runner repeat the sets."""
    _require_cand_program()
    inputs = ref.make_random_inputs(
        [48, 300], [20000, 4000], top_k=TOP_K, seed=7980, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    ns = ref.visible_rows(inputs).tolist()
    cand = cb.synthetic_rows(
        ns, CAND_WIDTH_MAX, "subset", seed=7981, device=inputs.device
    )
    first = naive.cloned(in_candidates(*args, cand, **inputs.kwargs()))
    runner = prepare_in_candidates(*args, cand, **inputs.kwargs())
    again = [
        naive.cloned(in_candidates(*args, cand, **inputs.kwargs())) for _ in range(2)
    ]
    again += [naive.cloned(runner()) for _ in range(2)]
    for got in again:
        if ORDERED:
            assert ref.same_bits(got, first)
        else:
            assert naive.same_ids(got, first)
    assert not _structure_failures(*first, cand, ns)


def test_caller_owned_workspace_and_outputs():
    """Prepared runners on one caller-owned workspace of exactly the bound (run in turns, so each run finds the
    ticket word the other left) and caller-owned ``indices`` / ``scores`` run without an allocation and return the
    caller's tensors; a workspace one byte short of the bound is refused before anything runs, by the eager entry
    and by prepare."""
    _require_cand_program()
    inputs = ref.make_random_inputs(
        [48, 16], [20000, 900], top_k=TOP_K, seed=7985, device="cuda"
    )
    device = inputs.device
    T, Tkv, S = inputs.num_queries, inputs.num_keys, inputs.num_segments
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    ns = ref.visible_rows(inputs).tolist()
    cand = cb.synthetic_rows(ns, CAND_WIDTH_MAX, "subset", seed=7986, device=device)
    other = cb.synthetic_rows(ns, CAND_WIDTH_MAX, "newest", seed=7987, device=device)
    nbytes = in_candidates_workspace_size(T, Tkv, S, top_k=TOP_K, device=device)
    assert (
        nbytes
        == plan_in_candidates(T, Tkv, S, top_k=TOP_K, device=device).workspace_bytes
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    indices = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    scores = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    runner = prepare_in_candidates(
        *args,
        cand,
        **inputs.kwargs(),
        workspace_buffer=workspace,
        indices=indices,
        scores=scores,
    )
    second = prepare_in_candidates(
        *args, other, **inputs.kwargs(), workspace_buffer=workspace
    )
    expected = naive.cloned(in_candidates(*args, cand, **inputs.kwargs()))
    expected2 = naive.cloned(in_candidates(*args, other, **inputs.kwargs()))
    runner()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    for _ in range(2):
        out = runner()
        out2 = second()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before, (
        "the prepared launches allocated"
    )
    assert [t.data_ptr() for t in out] == [indices.data_ptr(), scores.data_ptr()]
    assert naive.same_ids(naive.cloned(out), expected) and naive.same_ids(
        naive.cloned(out2), expected2
    )
    assert not _structure_failures(*out, cand, ns)
    short = workspace[: nbytes - 1]
    for call in (
        lambda: prepare_in_candidates(
            *args, cand, **inputs.kwargs(), workspace_buffer=short
        ),
        lambda: in_candidates(*args, cand, **inputs.kwargs(), workspace_buffer=short),
    ):
        with pytest.raises(ValueError, match="workspace_buffer needs"):
            call()


def test_arguments_are_checked():
    """``candidate_blocks`` must be an int32 ``[T, W]`` contiguous tensor on the device of ``q`` with ``W <= 2048``;
    ``k`` contiguous ``[Tkv, 128]`` rows; ``top_k`` in ``[1, 2048]``, ``ratio == 1`` and the BF16 form; all refused
    before any launch."""
    _require_cand_program()
    inputs = ref.make_random_inputs([32], [3000], top_k=TOP_K, seed=7990, device="cuda")
    T, device = inputs.num_queries, inputs.device
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    good = torch.full((T, 64), -1, dtype=torch.int32, device=device)
    good[:, 0] = 0
    wide = torch.zeros((T, 2 * 64), dtype=torch.int32, device=device)
    for bad, match in (
        (good.long(), "candidate_blocks must be"),
        (
            torch.zeros((T + 1, 64), dtype=torch.int32, device=device),
            "candidate_blocks must be",
        ),
        (
            torch.zeros((T, CAND_WIDTH_MAX + 1), dtype=torch.int32, device=device),
            "candidate_blocks must be",
        ),
        (
            torch.zeros((T,), dtype=torch.int32, device=device),
            "candidate_blocks must be",
        ),
        (wide[:, ::2], "candidate_blocks must be contiguous"),
        (good.cpu(), "candidate_blocks must be on"),
        (None, "candidate_blocks must be"),
    ):
        for fn in (in_candidates, prepare_in_candidates):
            with pytest.raises(ValueError, match=match):
                fn(*args, bad, **inputs.kwargs())
    packed = torch.zeros((inputs.num_keys, 704), dtype=inputs.k.dtype, device=device)
    packed[:, :128] = inputs.k
    with pytest.raises(ValueError, match="contiguous"):
        in_candidates(inputs.q, packed[:, :128], *args[2:], good, **inputs.kwargs())
    kwargs = inputs.kwargs()
    for change in ({"top_k": 0}, {"top_k": CAND_MAX_TOP_K + 1}, {"ratio": 2}):
        with pytest.raises(ValueError):
            in_candidates(*args, good, **{**kwargs, **change})
    with pytest.raises(ValueError, match="bfloat16"):
        in_candidates(
            inputs.q.to(torch.float8_e4m3fn),
            inputs.k.to(torch.float8_e4m3fn),
            *args[2:],
            good,
            top_k=TOP_K,
        )
    out = in_candidates(*args, good, **kwargs)
    assert out[0].dtype == torch.int32 and tuple(out[0].shape) == (T, TOP_K)
    empty = torch.empty((T, 0), dtype=torch.int32, device=device)
    ids, scores = in_candidates(*args, empty, **kwargs)
    assert bool((ids == -1).all()) and bool(torch.isneginf(scores).all())


def test_no_host_synchronization():
    """After a warm-up, one eager call and one run of a prepared runner complete under
    ``torch.cuda.set_sync_debug_mode("error")`` and return the warm-up's sets."""
    _require_cand_program()
    inputs = ref.make_random_inputs(
        [48, 300], [20000, 4000], top_k=TOP_K, seed=7995, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    ns = ref.visible_rows(inputs).tolist()
    cand = cb.synthetic_rows(
        ns, CAND_WIDTH_MAX, "subset", seed=7996, device=inputs.device
    )
    expected = naive.cloned(in_candidates(*args, cand, **inputs.kwargs()))
    runner = prepare_in_candidates(*args, cand, **inputs.kwargs())
    runner()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = in_candidates(*args, cand, **inputs.kwargs())
        prepared = runner()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    for got in (eager, prepared):
        assert naive.same_ids(naive.cloned(got), expected)


ALERT = "does not have a deterministic implementation"


@pytest.mark.parametrize("warn_only", [False, True])
def test_deterministic_algorithms_raise_or_warn(warn_only):
    """Under ``torch.use_deterministic_algorithms(True)`` the eager call with queries and keys raises PyTorch's
    ``RuntimeError`` before it allocates or launches anything, and every run of a prepared runner raises before its
    first launch; with ``warn_only=True`` each warns and runs.  Preparing a runner, the plan and the workspace bound
    never alert."""
    _require_cand_program()
    inputs = ref.make_random_inputs(
        [32], [17000], top_k=TOP_K, seed=7998, device="cuda"
    )
    args = (inputs.q, inputs.k, inputs.w, inputs.cu_seqlens_q, inputs.cu_seqlens_k)
    T, Tkv, S = inputs.num_queries, inputs.num_keys, inputs.num_segments
    cand = cb.synthetic_rows(
        ref.visible_rows(inputs).tolist(),
        CAND_WIDTH_MAX,
        "subset",
        seed=7999,
        device=inputs.device,
    )
    with _torch_deterministic(warn_only=warn_only):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            runner = prepare_in_candidates(*args, cand, **inputs.kwargs())
            plan_in_candidates(T, Tkv, S, top_k=TOP_K, device=inputs.device)
            in_candidates_workspace_size(T, Tkv, S, top_k=TOP_K, device=inputs.device)
        if warn_only:
            for call in (lambda: in_candidates(*args, cand, **inputs.kwargs()), runner):
                with pytest.warns(UserWarning, match=ALERT):
                    out = call()
                assert len(out) == 2
        else:
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()["allocation.all.allocated"]
            with pytest.raises(RuntimeError, match=ALERT):
                in_candidates(*args, cand, **inputs.kwargs())
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
            with pytest.raises(RuntimeError, match=ALERT):
                runner()
