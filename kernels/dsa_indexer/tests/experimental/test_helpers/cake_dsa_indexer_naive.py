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

The DSA indexer's outputs held to their contract and judged against the naive tier (``tests/cake_naive.py``).

The operator returns ``indices`` (int32 ``[T, top_k]``: segment-local key ids, ``-1`` padding) and ``scores``
(FP32 ``[T, top_k]``: the score of the id in the same slot, ``-inf`` padding).  The ids and the count of valid slots
of every row are integers and compare exactly; the scores are floating and are judged.

Exact draws (:func:`judge_exact`, :func:`assert_exact`) take inputs whose scores are exact in FP32 under any
summation order (``ref.exact_inputs``: the integer generators of the reference module).  On them the top
``min(top_k, visible)`` keys of a row, by score descending and then by segment-local id descending (``+0 == -0``),
are one set (``ref.exact_selection``) every program must return: ``indices`` and ``counts`` equal it exactly, and
``scores`` holds the oracle's ``-inf`` on every padding slot and a finite value on every other.  Over
:data:`~tests.cake_naive.DRAWS` draws (:func:`judge_exact`) the scores are judged against their FP64 values
(``ref.scores_at``) beside the naive tier's: the same scores of the same keys computed by torch in BF16 (the einsum
in BF16, the FP32 ``w`` and ``k_scale`` as stored, the result rounded to BF16).

Gaussian draws (:func:`judge_gaussian`) take a serving stack's inputs (``ref.make_random_inputs``,
``ref.make_fp8_inputs``), on which the FP32 summation order rounds: keys whose FP64 scores are near-tied may change
places, so no exact selection exists to compare the ids with.  There the ids keep their structure exactly
(``ref.structure_failures``: unique, ascending, visible, ``-1`` padding) and their count (``min(top_k, visible)``),
and the scores are judged at the kernel's own keys against the FP64 and the naive tier's BF16 scores of those keys.

The deterministic package writes every row in ascending id order, padding last (``ordered=True``: compared as
written); the non-deterministic package writes a row's slots in any order (``ordered=False``: :func:`canonical` puts
each row in ascending id order with its scores before anything is compared).
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from tests.cake_naive import contract, judge_case
from tests.experimental.test_helpers import cake_dsa_indexer_reference as ref

#: Exact outputs of every operator result: the ids and the count of valid slots per row.
EXACT_OUTPUTS = frozenset({"indices", "counts"})
#: The judged output of an exact draw and of a Gaussian draw.
SCORES = "scores"
SCORES_GAUSSIAN = "scores@randn"
# sorts after every int32 id: the padding of a row goes behind its valid slots
_PAD_SORT_KEY = 1 << 32


def cloned(out) -> tuple[torch.Tensor, torch.Tensor]:
    """Copies of an ``(indices, scores)`` result (a prepared runner writes into the same outputs on every run)."""
    return out[0].clone(), out[1].clone()


def canonical(result) -> tuple[torch.Tensor, torch.Tensor]:
    """``result`` with every row in ascending id order with its scores, then the padding (any negative id keeps its
    score and goes behind them): a stable per-row permutation of the slots, so every row keeps its (id, score) pairs
    and its valid count, and a duplicate, a missing entry or a bad padding slot stays visible to the checks."""
    ids, scores = result
    key = torch.where(ids >= 0, ids.to(torch.int64), _PAD_SORT_KEY)
    order = key.sort(dim=1, stable=True).indices
    return ids.gather(1, order), scores.gather(1, order)


def same_ids(a, b) -> bool:
    """Per row the same valid count and the same set of ids (slot order free; scores not compared)."""
    if tuple(a[0].shape) != tuple(b[0].shape):
        return False
    return bool(torch.equal((a[0] >= 0).sum(dim=1), (b[0] >= 0).sum(dim=1))) and bool(
        torch.equal(a[0].sort(dim=1).values, b[0].sort(dim=1).values)
    )


def _counts(indices: torch.Tensor) -> torch.Tensor:
    return (indices >= 0).sum(dim=1).to(torch.int64)


def exact_failures(
    result,
    inputs: ref.IndexerInputs,
    reference: ref.ReferenceSelection | None = None,
    *,
    ordered: bool = True,
    rows: torch.Tensor | None = None,
) -> list[str]:
    """Every way ``result = (indices, scores)`` breaks its exact contract on ``inputs`` (exact in FP32): the
    structure checks, ``indices`` and ``counts`` against ``reference`` (default ``ref.exact_selection(inputs)``)
    exactly, ``scores`` finite where the oracle is and ``-inf`` where it is.  ``rows``: compare those rows only
    (``reference`` then covers the problem they belong to, row for row)."""
    if reference is None:
        reference = ref.exact_selection(inputs)
    got = result if ordered else canonical(result)
    failures = list(ref.structure_failures(inputs, got[0], got[1], rows=rows))
    if failures:
        return failures
    want_ids, want_scores, want_counts = (
        reference.indices,
        reference.scores_fp64,
        reference.selected_count,
    )
    got_ids, got_scores = got
    if rows is not None:
        rows = rows.to(device=got_ids.device, dtype=torch.int64)
        got_ids, got_scores = got_ids.index_select(0, rows), got_scores.index_select(0, rows)
        want_ids, want_scores, want_counts = (
            want_ids.index_select(0, rows),
            want_scores.index_select(0, rows),
            want_counts.index_select(0, rows),
        )
    for name, actual, expected, exact in (
        ("indices", got_ids, want_ids, True),
        ("counts", _counts(got_ids), want_counts, True),
        (SCORES, got_scores, want_scores, False),
    ):
        if message := contract(name, actual, expected, exact):
            failures.append(message)
    return failures


def assert_exact(
    result,
    inputs: ref.IndexerInputs,
    reference: ref.ReferenceSelection | None = None,
    *,
    ordered: bool = True,
    rows: torch.Tensor | None = None,
) -> ref.ReferenceSelection:
    """:func:`exact_failures` must be empty; returns the reference used."""
    if reference is None:
        reference = ref.exact_selection(inputs)
    failures = exact_failures(result, inputs, reference, ordered=ordered, rows=rows)
    assert not failures, f"{inputs.label or 'call'}:\n" + "\n".join(failures[:20])
    return reference


def judge_exact(
    record_property,
    make: Callable[[int], ref.IndexerInputs],
    run: Callable[[ref.IndexerInputs], tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
    ordered: bool = True,
    check: Callable[[int, ref.IndexerInputs, tuple, ref.ReferenceSelection], None] | None = None,
) -> None:
    """Judge ``run`` over :data:`~tests.cake_naive.DRAWS` exact draws: draw ``j`` makes ``make(s)`` (``s`` the draw's
    seed, ``tests.cake_naive.draw_seed(seed, j)``; the inputs must be exact in FP32), runs ``run`` on it once and holds
    ``indices`` and ``counts`` to ``ref.exact_selection`` exactly and ``scores`` to the structure and the oracle's
    ``-inf`` / finite pattern; the scores are judged against their FP64 values beside the naive tier's BF16 scores of
    the same keys.  ``check(j, inputs, result, reference)`` adds the test's own checks of a draw.  Every verdict is
    recorded with ``record_property`` (the ``TIER`` lines of the terminal summary); any contract failure or a missed
    tier fails the test with its numbers."""
    state: dict = {}
    structure: list[str] = []

    def draw(j: int, s: int):
        inputs = make(s)
        result = cloned(run(inputs))
        reference = ref.exact_selection(inputs)
        if check is not None:
            check(j, inputs, result, reference)
        got = result if ordered else canonical(result)
        structure.extend(f"draw {j}: {m}" for m in ref.structure_failures(inputs, *got))
        state.update(inputs=inputs, reference=reference)
        outputs = {"indices": got[0], "counts": _counts(got[0]), SCORES: got[1]}
        oracle = {
            "indices": reference.indices,
            "counts": reference.selected_count,
            SCORES: reference.scores_fp64,
        }
        return outputs, oracle

    def naive(j: int, tier: str):
        return {SCORES: ref.scores_at(state["inputs"], state["reference"].indices, torch.bfloat16)}

    judge, failures = judge_case(draw, naive, seed=seed, exact=EXACT_OUTPUTS)
    judge.record(record_property)
    failures = structure + failures
    assert not failures, "\n".join(failures[:30])


def judge_gaussian(
    record_property,
    make: Callable[[int], ref.IndexerInputs],
    run: Callable[[ref.IndexerInputs], tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int,
    ordered: bool = True,
    check: Callable[[int, ref.IndexerInputs, tuple, None], None] | None = None,
) -> None:
    """Judge ``run`` over :data:`~tests.cake_naive.DRAWS` Gaussian draws (``make(s)``): the ids hold their structure
    and their count ``min(top_k, visible)`` exactly, and the scores are judged at the kernel's own keys against their
    FP64 values beside the naive tier's BF16 scores of those keys (output ``scores@randn``).  ``check(j, inputs,
    result, None)`` adds the test's own checks of a draw."""
    state: dict = {}
    structure: list[str] = []

    def draw(j: int, s: int):
        inputs = make(s)
        got = cloned(run(inputs))
        if check is not None:
            check(j, inputs, got, None)
        if not ordered:
            got = canonical(got)
        structure.extend(f"draw {j}: {m}" for m in ref.structure_failures(inputs, *got))
        state.update(inputs=inputs, ids=got[0])
        outputs = {"counts": _counts(got[0]), SCORES_GAUSSIAN: got[1]}
        oracle = {
            "counts": ref.visible_rows(inputs).clamp(max=inputs.top_k),
            SCORES_GAUSSIAN: ref.scores_at(inputs, got[0], torch.float64),
        }
        return outputs, oracle

    def naive(j: int, tier: str):
        return {SCORES_GAUSSIAN: ref.scores_at(state["inputs"], state["ids"], torch.bfloat16)}

    judge, failures = judge_case(draw, naive, seed=seed, exact=frozenset({"counts"}))
    judge.record(record_property)
    failures = structure + failures
    assert not failures, "\n".join(failures[:30])


__all__ = [
    "EXACT_OUTPUTS",
    "SCORES",
    "SCORES_GAUSSIAN",
    "assert_exact",
    "canonical",
    "cloned",
    "exact_failures",
    "judge_exact",
    "judge_gaussian",
    "same_ids",
]
