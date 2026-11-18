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

FP64 oracle of the indexer's candidate blocks (``*_topk_with_candidates``) and the checks of its sets.

Query ``u`` of a segment of ``Lq`` queries and ``Lk`` keys sees the keys ``j < n = clamp(offset + u + 1, 0, Lk)``
(ratio 1; ``offset`` is ``q_causal_offsets[s]`` when given, else ``Lk - Lq``) and scores them
``s = sum_h w[u, h] relu(softmax_scale q[u, h] . k[j])`` in float64 (:func:`scores64`).  Keys rank by score
descending, then by id descending, ``+0 == -0`` (:func:`rank`).  Block ``b`` holds the keys ``8 b .. 8 b + 7``
(:data:`BLOCK`); a row sees ``nb = ceil(n / 8)`` blocks, the newest (``nb - 1``) maybe partial, and a block's score is
the largest score of its visible keys.  ``candidate_blocks`` (:data:`NUM_CANDIDATES` = 2048 slots) holds every visible
block when ``nb <= 2048``, else the newest block and the 2047 older blocks of the largest scores, ranked as keys are;
``indices`` holds the top ``min(top_k, n)`` keys.  Both are returned ascending with a ``-1`` tail
(:func:`with_candidates`).  With ``top_k <= 2047`` every kept key lies in a kept block (:func:`contained`).

Exact inputs (:func:`exact_inputs`): ``q`` and ``k`` on the MXFP4 grid (E2M1 values times ``2^0`` or ``2^-1`` per 32
elements, :func:`mx`), ``w`` in ``{-4..4} / 8`` and ``softmax_scale`` 1/16; every partial sum of every score is then an
FP32 value in any summation order, so the FP32 kernel's scores equal the FP64 ones (up to the sign of a zero, which
the order ignores) and its sets equal the oracle's.  Gaussian inputs round: :func:`near_cutoff_only` is their set rule.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

HEADS = 32
DIM = 128
BLOCK = 8
NUM_CANDIDATES = 2048
INF = float("inf")
EXACT_SCALE = 1.0 / 16
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # the magnitudes of an MXFP4 element
EXACT_CASES = (
    "MXFP4-grid",
    "duplicated keys",
    "all scores zero",
    "signed zeros",
    "negative scores",
    "newest block lowest",
)
CHUNK_ELEMENTS = (
    1 << 25
)  # the [rows, heads, keys] float64 logits of one scoring step (256 MiB)


def scores64(
    q: torch.Tensor, k: torch.Tensor, w: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """The ``[T, L]`` float64 scores of the queries ``q [T, H, D]`` with weights ``w [T, H]`` against ``k [L, D]``."""
    x = torch.einsum("thd,jd->thj", q.double(), k.double())
    return (w.double()[:, :, None] * torch.relu(softmax_scale * x)).sum(1)


def visible(u: int, offset: int, L: int) -> int:
    """The number of keys that query ``u`` of a segment of ``L`` keys and causal offset ``offset`` sees (ratio 1)."""
    return max(0, min(L, offset + u + 1))


def segments(
    cu_seqlens_q, cu_seqlens_k, q_causal_offsets=None
) -> list[tuple[int, int, int, int, int]]:
    """``(q0, Lq, k0, Lk, offset)`` of every segment."""
    cq, ck = (
        torch.as_tensor(cu_seqlens_q).tolist(),
        torch.as_tensor(cu_seqlens_k).tolist(),
    )
    offsets = (
        None if q_causal_offsets is None else torch.as_tensor(q_causal_offsets).tolist()
    )
    out = []
    for s in range(len(cq) - 1):
        Lq, Lk = cq[s + 1] - cq[s], ck[s + 1] - ck[s]
        out.append((cq[s], Lq, ck[s], Lk, Lk - Lq if offsets is None else offsets[s]))
    return out


def rank(vals: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``ids`` by ``vals`` descending, then by id descending; ``+0 == -0``."""
    order = torch.argsort(ids, descending=True)
    _, o2 = torch.sort(vals[order] + 0.0, descending=True, stable=True)
    return ids[order][o2]


def source_pass(
    s_rows: torch.Tensor,
    ns: Sequence[int],
    top_k: int,
    num_blocks: int = NUM_CANDIDATES,
    block: int = BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indices [R, top_k], candidate_blocks [R, num_blocks])`` of rows with scores ``s_rows [R, L]`` that see
    ``ns`` keys; ascending, ``-1`` tail."""
    dev = s_rows.device
    ids = torch.full((len(ns), top_k), -1, dtype=torch.long, device=dev)
    cand = torch.full((len(ns), num_blocks), -1, dtype=torch.long, device=dev)
    for r, n in enumerate(map(int, ns)):
        if n == 0:
            continue
        s = s_rows[r, :n]
        top = rank(s, torch.arange(n, device=dev))[:top_k].sort().values
        ids[r, : len(top)] = top
        nb = (n - 1) // block + 1
        bmax = F.pad(s, (0, nb * block - n), value=-INF).view(nb, block).amax(1)
        older = rank(bmax[:-1], torch.arange(nb - 1, device=dev))[: num_blocks - 1]
        c = torch.cat([older, torch.tensor([nb - 1], device=dev)]).sort().values
        cand[r, : len(c)] = c
    return ids, cand


def _chunks(Lq: int, Lk: int, rows=None) -> list[list[int]]:
    """Lists of segment-local query rows whose float64 logits stay within :data:`CHUNK_ELEMENTS`."""
    rows = list(range(Lq)) if rows is None else [int(u) for u in rows]
    step = max(1, CHUNK_ELEMENTS // max(1, HEADS * Lk))
    return [rows[i : i + step] for i in range(0, len(rows), step)]


def with_candidates(
    q,
    k,
    w,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    softmax_scale: float,
    top_k: int,
    q_causal_offsets=None,
    rows=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The oracle over a packed batch: ``(rows [R], indices [R, top_k], candidate_blocks [R, 2048])``, int64,
    segment-local, for the global query rows ``rows`` (default: all, in order)."""
    want = None if rows is None else sorted({int(r) for r in rows})
    out_rows, out_ids, out_cand = [], [], []
    for q0, Lq, k0, Lk, offset in segments(
        cu_seqlens_q, cu_seqlens_k, q_causal_offsets
    ):
        local = None if want is None else [r - q0 for r in want if q0 <= r < q0 + Lq]
        for chunk in _chunks(Lq, Lk, local):
            if not chunk:
                continue
            g = torch.tensor([q0 + u for u in chunk], device=q.device)
            s = scores64(q[g], k[k0 : k0 + Lk], w[g], softmax_scale)
            ids, cand = source_pass(s, [visible(u, offset, Lk) for u in chunk], top_k)
            out_rows.append(g)
            out_ids.append(ids)
            out_cand.append(cand)
    if not out_rows:
        empty = torch.empty(0, dtype=torch.long, device=q.device)
        return empty, empty.view(0, top_k), empty.view(0, NUM_CANDIDATES)
    return torch.cat(out_rows), torch.cat(out_ids), torch.cat(out_cand)


def row_scores64(
    q,
    k,
    w,
    cu_seqlens_q,
    cu_seqlens_k,
    row: int,
    *,
    softmax_scale: float,
    q_causal_offsets=None,
):
    """``(scores [n], n)``: the float64 scores of the visible keys of global query row ``row``."""
    for q0, Lq, k0, Lk, offset in segments(
        cu_seqlens_q, cu_seqlens_k, q_causal_offsets
    ):
        if q0 <= row < q0 + Lq:
            n = visible(row - q0, offset, Lk)
            return scores64(
                q[row : row + 1], k[k0 : k0 + n], w[row : row + 1], softmax_scale
            )[0], n
    raise ValueError(f"row {row} is in no segment")


def mx(
    *shape: int, generator: torch.Generator | None = None, device=None
) -> torch.Tensor:
    """float64 values on the MXFP4 grid: E2M1 values with random signs, times ``2^0`` or ``2^-1`` per 32 elements of
    the last dimension."""
    grid = torch.tensor(E2M1, dtype=torch.float64, device=device)
    v = grid[torch.randint(0, 8, shape, generator=generator, device=device)] * (
        torch.randint(0, 2, shape, generator=generator, device=device) * 2 - 1
    )
    e = torch.randint(
        0, 2, shape[:-1] + (shape[-1] // 32, 1), generator=generator, device=device
    )
    return (v.unflatten(-1, (-1, 32)) * 2.0**-e).flatten(-2)


def exact_segment(case: str, Tq: int, Lk: int, *, generator=None, device=None):
    """``(q [Tq, 32, 128], k [Lk, 128], w [Tq, 32])``, float64, of one segment of the case ``case``: MXFP4-grid ``q``
    and ``k``, ``w`` in ``{-4..4} / 8``; "duplicated keys": every key a copy of one of the first three (score ties,
    equal block maxima: the id decides); "all scores zero" / "signed zeros": zero keys (every score a zero of either
    sign); "negative scores": ``w`` in ``-{1..4} / 8``, ``|q|`` and positive keys, so every score is negative;
    "newest block lowest": ``w`` in ``{1..4} / 8`` and the last 8 keys zero, so the newest block of the segment's
    last rows has the lowest block score (every older block scores above 0)."""
    if case not in EXACT_CASES:
        raise ValueError(f"case {case!r} is none of {EXACT_CASES}")
    q, k = (
        mx(Tq, HEADS, DIM, generator=generator, device=device),
        mx(Lk, DIM, generator=generator, device=device),
    )
    w = (
        torch.randint(-4, 5, (Tq, HEADS), generator=generator, device=device).double()
        / 8
    )
    if case == "duplicated keys" and Lk:
        k = k[torch.randint(0, min(3, Lk), (Lk,), generator=generator, device=device)]
    if case in ("all scores zero", "signed zeros"):
        k = torch.zeros(Lk, DIM, dtype=torch.float64, device=device)
    if case == "negative scores":
        w, q = (
            -torch.randint(
                1, 5, (Tq, HEADS), generator=generator, device=device
            ).double()
            / 8,
            q.abs(),
        )
        grid = torch.tensor(E2M1, dtype=torch.float64, device=device)
        magnitudes = grid[
            torch.randint(1, 8, (Lk, DIM), generator=generator, device=device)
        ]
        e = torch.randint(0, 2, (Lk, DIM // 32), generator=generator, device=device)
        k = magnitudes * 2.0 ** -e.repeat_interleave(32, 1).double()
    if case == "newest block lowest":
        w = (
            torch.randint(
                1, 5, (Tq, HEADS), generator=generator, device=device
            ).double()
            / 8
        )
        k[max(0, Lk - 8) :] = 0
    return q, k, w


def exact_inputs(
    case: str, seg_q: Sequence[int], seg_k: Sequence[int], *, seed: int, device
):
    """``(q bf16, k bf16, w fp32, cu_seqlens_q, cu_seqlens_k)`` of a packed batch of the case ``case``, one
    :func:`exact_segment` per segment (MXFP4-grid values are BF16 values; ``w`` an FP32 value)."""
    gen = torch.Generator(device=device).manual_seed(int(seed))
    parts = [
        exact_segment(case, Tq, Lk, generator=gen, device=device)
        for Tq, Lk in zip(seg_q, seg_k)
    ]
    q, k, w = (torch.cat([p[i] for p in parts]) for i in range(3))

    def cumulative(lengths: Sequence[int]) -> torch.Tensor:
        return torch.tensor((0, *lengths), device=device).cumsum(0).int()

    return q.bfloat16(), k.bfloat16(), w.float(), cumulative(seg_q), cumulative(seg_k)


def ascending(t: torch.Tensor) -> torch.Tensor:
    """Each row of an id table in ascending order with its ``-1`` slots last (the det order; a set compare)."""
    key = torch.where(t >= 0, t.long(), torch.iinfo(torch.int64).max)
    s = key.sort(dim=-1).values
    return torch.where(s == torch.iinfo(torch.int64).max, -1, s)


def contained(
    indices: torch.Tensor, candidate_blocks: torch.Tensor, block: int = BLOCK
) -> torch.Tensor:
    """``[T]`` bool: whether every non-negative ``indices[t, i] // block`` is a block of ``candidate_blocks[t]``."""
    ids, cand = indices.long(), candidate_blocks.long()
    want = torch.where(ids >= 0, ids // block, 0)
    have = (
        torch.where(cand >= 0, cand, torch.iinfo(torch.int64).max)
        .sort(dim=-1)
        .values.contiguous()
    )
    pos = torch.searchsorted(have, want.contiguous()).clamp(
        max=max(0, have.shape[1] - 1)
    )
    found = (
        have.gather(-1, pos) == want
        if have.shape[1]
        else torch.zeros_like(want, dtype=torch.bool)
    )
    return (found | (ids < 0)).all(dim=-1)


def block_maxima(vals: Sequence[float], n: int, block: int = BLOCK) -> list[float]:
    """The score of each of the ``ceil(n / block)`` visible blocks of a row of scores ``vals``."""
    return [
        max(vals[block * b : min(block * b + block, n)])
        for b in range((n + block - 1) // block)
    ]


def beats_rest(vals: Sequence[float], pool: Sequence[int], got: set, top: int) -> bool:
    """Whether ``got`` is the top ``min(top, |pool|)`` of ``pool`` by ``(vals[j], j)`` (independent of :func:`rank`;
    ``-0.0 == +0.0`` in Python)."""
    if len(got) != min(top, len(pool)) or not got <= set(pool):
        return False
    rest = [(vals[j], j) for j in pool if j not in got]
    return not got or not rest or min((vals[j], j) for j in got) > max(rest)


def cut_tie(vals: Sequence[float], pool: Sequence[int], got: set) -> bool:
    """Whether equal values sit on both sides of the cutoff of ``got`` in ``pool``."""
    rest = [vals[j] for j in pool if j not in got]
    return bool(got and rest and min(vals[j] for j in got) == max(rest))


def near_cutoff_only(
    vals: Sequence[float], pool: Sequence[int], got: set, top: int, e: float
) -> bool:
    """The set rule for Gaussian inputs: ``got`` holds ``min(top, |pool|)`` members of ``pool``, and each member of
    its symmetric difference with the FP64 top (``vals`` by ``(value, id)``) has an FP64 value within ``2 e`` of the
    FP64 cutoff (the value of that top's last member); ``e`` is the largest ``|kernel score - FP64 score|`` of the
    draw.  A pool of at most ``top`` members is matched exactly."""
    if len(got) != min(top, len(pool)) or not got <= set(pool):
        return False
    best = sorted(pool, key=lambda j: (vals[j], j), reverse=True)[:top]
    if not best:
        return True
    cutoff = vals[best[-1]]
    return all(abs(vals[j] - cutoff) <= 2 * e for j in got.symmetric_difference(best))


def restricted_topk(
    s_rows: torch.Tensor,
    ns: Sequence[int],
    cand: torch.Tensor,
    top_k: int,
    block: int = BLOCK,
) -> torch.Tensor:
    """The candidate-restricted top-k of rows with scores ``s_rows [R, L]`` that see ``ns`` keys: the top
    ``min(top_k, |pool|)`` keys ``j < n`` of the blocks of ``cand[r]`` (any order, ``-1`` anywhere; an id ``>= nb``
    names no key), ascending, ``-1`` tail."""
    dev = s_rows.device
    ids = torch.full((len(ns), top_k), -1, dtype=torch.long, device=dev)
    lanes = torch.arange(block, device=dev)
    for r, n in enumerate(map(int, ns)):
        b = cand[r].long()
        b = b[b >= 0]
        pos = (b[:, None] * block + lanes).flatten()
        pos = pos[pos < n]
        top = rank(s_rows[r, pos], pos)[:top_k].sort().values
        ids[r, : len(top)] = top
    return ids


def in_candidates(
    q,
    k,
    w,
    cu_seqlens_q,
    cu_seqlens_k,
    candidate_blocks,
    *,
    softmax_scale: float,
    top_k: int,
    q_causal_offsets=None,
    rows=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The oracle of the candidate-restricted top-k over a packed batch: ``(rows [R], indices [R, top_k])``, int64,
    segment-local, for the global query rows ``rows`` (default: all, in order)."""
    want = None if rows is None else sorted({int(r) for r in rows})
    out_rows, out_ids = [], []
    for q0, Lq, k0, Lk, offset in segments(
        cu_seqlens_q, cu_seqlens_k, q_causal_offsets
    ):
        local = None if want is None else [r - q0 for r in want if q0 <= r < q0 + Lq]
        for chunk in _chunks(Lq, Lk, local):
            if not chunk:
                continue
            g = torch.tensor([q0 + u for u in chunk], device=q.device)
            s = scores64(q[g], k[k0 : k0 + Lk], w[g], softmax_scale)
            out_rows.append(g)
            out_ids.append(
                restricted_topk(
                    s,
                    [visible(u, offset, Lk) for u in chunk],
                    candidate_blocks[g],
                    top_k,
                )
            )
    if not out_rows:
        empty = torch.empty(0, dtype=torch.long, device=q.device)
        return empty, empty.view(0, top_k)
    return torch.cat(out_rows), torch.cat(out_ids)


def synthetic_rows(
    ns: Sequence[int], width: int, kind: str, *, seed: int, device
) -> torch.Tensor:
    """``[R, width]`` int32 candidate rows for rows that see ``ns`` keys: ``"subset"`` a random subset of the visible
    blocks in shuffled order with ``-1`` scattered; ``"past"`` the same plus ids at and past the visible blocks (and
    past any segment's keys); ``"newest"`` the newest (maybe partial) block and a few older ones; ``"few"`` at most
    three blocks (fewer than ``top_k`` candidate keys); ``"empty"`` all ``-1``; ``"all"`` every visible block when it
    fits, else the newest ``width``."""
    gen = torch.Generator().manual_seed(int(seed))
    out = torch.full((len(ns), width), -1, dtype=torch.int32)
    for r, n in enumerate(map(int, ns)):
        nb = (n + BLOCK - 1) // BLOCK
        if kind == "empty" or width == 0:
            continue
        if kind == "all":
            ids = torch.arange(max(0, nb - width), nb)
        else:
            take = {"subset": width // 2, "past": width // 2, "newest": 4, "few": 3}[
                kind
            ]
            ids = (
                torch.randperm(max(nb, 1), generator=gen)[: min(take, nb)]
                if nb
                else torch.empty(0, dtype=torch.long)
            )
            if kind == "newest" and nb:
                ids = torch.unique(torch.cat([ids[:3], torch.tensor([nb - 1])]))
            if kind == "past":
                extra = torch.tensor([nb, nb + 1, nb + 7, 1 << 20, (1 << 31) - 1])
                ids = torch.cat([ids[: max(0, width - len(extra))], extra])
        slots = torch.randperm(width, generator=gen)[: len(ids)]
        out[r, slots] = ids.to(torch.int32)
    return out.to(device)


__all__ = [
    "BLOCK",
    "EXACT_CASES",
    "EXACT_SCALE",
    "NUM_CANDIDATES",
    "ascending",
    "beats_rest",
    "block_maxima",
    "contained",
    "cut_tie",
    "exact_inputs",
    "exact_segment",
    "in_candidates",
    "mx",
    "near_cutoff_only",
    "rank",
    "restricted_topk",
    "row_scores64",
    "scores64",
    "segments",
    "source_pass",
    "synthetic_rows",
    "visible",
    "with_candidates",
]
