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

The outputs of the KDA tests that the owner accepted to miss the naive BF16 tier, each only within its scope and up
to its caps (:data:`KNOWN_GAPS`), and the test items where such a miss is listed (:data:`KNOWN_GAP_ITEMS`).

The judge (``tests/cake_naive.py``) is unchanged: every output is judged against the naive BF16 tier and every
verdict prints its ``TIER`` line.  After a judge, a test calls :func:`settle` instead of asserting that the judge
found no failure:

- a failure that is not a tier miss (a contract break, an output judged on fewer draws than the judge's, two draws
  with the same oracle values) fails the test with the judge's text;
- an output that misses the tier and FLA and is listed for the running test in the owner-accepted misses of
  ``tests/cake_naive_accepted.json`` passes when it is within the caps listed for it, and fails the test with the
  reasons and the judge's text when it is beyond them (``tests.cake_naive.accepted_check``): the test runs on, it is
  not reported xfailed, and its ``TIER`` line reads ``accepted``;
- an output that misses the tier and is not listed for the running test fails the test with the judge's text;
- a listed output that meets the tier fails the test: the listing is stale (a strict xfail);
- a listed output that misses beyond its gap fails the test: the case is outside the gap's scope (:attr:`KnownGap.
  scope`), the output is not one of the gap's (:attr:`KnownGap.outputs`), the gap has no cap, a norm it misses is
  judged draw by draw (no cap covers a draw-by-draw norm) or has no cap, the ratio of its median or of its largest
  error over the tier's is above the cap, or the zero rule or a tier error that is not finite (no cap covers them);
- a listed output that misses within its caps is recorded, and the test runs on.

The family's ``conftest.py`` reports a test that passed with recorded gaps as xfailed, its reason naming each gap's
key, the output, its ratios and caps (:func:`finish`); a test that passed without judging an output listed for it
fails, and a test that failed stays failed.  So a gap that widens beyond its cap fails, and a gap that closes fails
until its listing is removed.

A cap is the owner's: ``{norm: cap}`` for a norm judged by its median and largest error over the draws (a ratio of
the output's error over the tier's), a finite float of at least :data:`NEAR_TIE` with at most four decimals
(:func:`check_registry`).  A gap whose ``caps`` are empty is accepted without a cap: every miss of it fails, with its
measured ratios, until the owner gives a cap.

The test id is pytest's node id from the ``tests/experimental/`` directory of the family's tests
(``tests/experimental/<file>::<test>[<params>]``), set by the conftest around each test's call (:func:`running`).
This file is the same in the family's own tests and in its cross-family tests.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from tests.cake_naive import NEAR_TIE, NORMS, TIERS, accepted_cap, accepted_check

#: The scopes a gap may have: ``"one_token"``, a batch of exactly one document of one token (``cu_seqlens`` ``[0,
#: 1]``); ``"internal"``, an internal tensor of the kernels that no test of this family judges (never applies).
SCOPES = ("one_token", "internal")


@dataclass(frozen=True)
class KnownGap:
    """A miss of the naive tier the owner accepted: its key, what it covers (``outputs``: the output names, the
    part after an arm's ``"<arm>."`` prefix; ``scope``: the cases), and its caps ``{norm: cap}`` (empty: no cap
    accepted yet, so every miss fails)."""

    key: str
    what: str
    outputs: tuple[str, ...]
    scope: str
    caps: Mapping[str, float] = field(default_factory=dict)


#: The owner-accepted gaps of the KDA family.  Only these four exist; anything else that misses the tier is a failure.
KNOWN_GAPS: dict[str, KnownGap] = {
    "one_token_out": KnownGap(
        "one_token_out",
        "the forward output of a batch of one one-token document",
        ("out",),
        "one_token",
        {"l2": 1.0168},
    ),
    "one_token_dbeta": KnownGap(
        "one_token_dbeta",
        "dbeta of a batch of one one-token document; accepted without a cap: a miss fails with its ratios until the "
        "owner caps it",
        ("dbeta",),
        "one_token",
        {},
    ),
    "kcp_t2k_one_chunk_v_new": KnownGap(
        "kcp_t2k_one_chunk_v_new",
        "the v_new tape of the KCP T2K route on one-chunk documents: an internal tape that no test here judges",
        (),
        "internal",
        {"l2": 1.0060},
    ),
    "out_of_domain_gate_forms": KnownGap(
        "out_of_domain_gate_forms",
        "out-of-domain gate forms u20 / negzero (Aqk / Akk not judged) and tiny (gk exact on the FTZ prefix): internal "
        "stage outputs that no test here judges",
        (),
        "internal",
        {},
    ),
}

#: The test items where an owner-accepted gap is listed: node id -> {output: gap key}.  An item is listed only after a
#: recorded run shows the output missing within the gap's scope and caps; a listed output that meets the tier fails.
KNOWN_GAP_ITEMS: dict[str, dict[str, str]] = {}

_current: str | None = None
# test id -> {"judged": set of outputs, "gaps": {output: (gap key, ratios, caps)}}
_records: dict[str, dict] = {}


def canonical(node_id: str) -> str:
    """``node_id`` from its ``tests/experimental/`` directory on (pytest's node id depends on the rootdir)."""
    marker = "tests/experimental/"
    at = node_id.rfind(marker)
    return node_id[at:] if at >= 0 else node_id


@contextlib.contextmanager
def running(test_id: str | None) -> Iterator[None]:
    """Make ``test_id`` (pytest's node id) the running test of :func:`settle` while the block runs."""
    global _current
    previous, _current = _current, None if test_id is None else canonical(test_id)
    try:
        yield
    finally:
        _current = previous


def current_test() -> str:
    """The running test's id: the one :func:`running` set, else pytest's ``PYTEST_CURRENT_TEST``."""
    if _current is not None:
        return _current
    found = os.environ.get("PYTEST_CURRENT_TEST")
    if not found:
        raise RuntimeError("no running test: call settle() inside a pytest test")
    return canonical(found.rsplit(" (", 1)[0])


def in_scope(gap: KnownGap, lengths: Sequence[int] | None) -> bool:
    """Whether a case of documents ``lengths`` is in ``gap``'s scope."""
    if gap.scope == "one_token":
        return lengths is not None and tuple(int(n) for n in lengths) == (1,)
    return False


def ratios(verdict: Mapping) -> tuple[float, float, float, float]:
    """The output's error over the tier's, (L2 median, L2 largest, max median, max largest), of a verdict (inf where
    the tier's error is 0 or not finite and the output's is not 0)."""
    (tier,) = TIERS
    found = []
    for norm in NORMS:
        ours, theirs = verdict["norms"][norm]["ours"], verdict["norms"][norm][tier]
        for i in (0, 1):
            if theirs is None or theirs[i] == 0:
                found.append(float("inf") if ours[i] > 0 else 1.0)
            else:
                found.append(ours[i] / theirs[i])
    return tuple(found)


def missed_norms(verdict: Mapping) -> list[str]:
    """The norms in which a verdict misses the tier, by the judge's rule: a norm judged draw by draw (``by_draw``)
    with fewer than half of the draws within :data:`NEAR_TIE` of the tier; another norm with its median or its largest
    error above :data:`NEAR_TIE` times the tier's; a tier error that is not finite."""
    (tier,) = TIERS
    found = []
    for norm in NORMS:
        ours, bound = verdict["norms"][norm]["ours"], verdict["norms"][norm][tier]
        if bound is None:
            found.append(norm)
        elif norm in verdict.get("by_draw", ()):
            if 2 * verdict["wins"][norm] < verdict["draws"]:
                found.append(norm)
        elif not (ours[0] <= NEAR_TIE * bound[0] and ours[1] <= NEAR_TIE * bound[1]):
            found.append(norm)
    return found


def beyond(
    name: str, gap: KnownGap, verdict: Mapping, lengths: Sequence[int] | None
) -> list[str]:
    """Why the listed output ``name`` misses the tier beyond ``gap`` on a case of documents ``lengths``, one message per
    reason, or none."""
    (tier,) = TIERS
    head = f"{name}: misses the naive tier beyond {gap.key}"
    found = []
    base = name.rpartition(".")[2]
    if base not in gap.outputs:
        found.append(f"{head}: {base} is not an output of the gap {gap.outputs}")
    if not in_scope(gap, lengths):
        found.append(
            f"{head}: the case's documents {None if lengths is None else tuple(lengths)} are outside its scope {gap.scope}"
        )
    if verdict["zero"][tier]:
        found.append(
            f"{head}: tier {tier} error 0 on draws {verdict['zero'][tier]}, ours not (the zero rule; no cap covers it)"
        )
    measured = ratios(verdict)
    for i, norm in enumerate(NORMS):
        if norm not in missed_norms(verdict):
            continue
        pair = measured[2 * i : 2 * i + 2]
        if verdict["norms"][norm][tier] is None:
            found.append(
                f"{head}: {norm}: the tier's error is not finite on a draw (no cap covers it)"
            )
        elif norm in verdict.get("by_draw", ()):
            found.append(
                f"{head}: {norm} is judged draw by draw, ours within {NEAR_TIE} x the tier on {verdict['wins'][norm]} of "
                f"{verdict['draws']} draws (no cap covers a draw-by-draw norm)"
            )
        elif not gap.caps:
            found.append(
                f"{head}: {norm} median / largest ours x{pair[0]:.9g} / x{pair[1]:.9g} the tier's, and the gap has no "
                "cap yet (the owner's cap is needed)"
            )
        elif norm not in gap.caps:
            found.append(
                f"{head}: {norm} median / largest ours x{pair[0]:.9g} / x{pair[1]:.9g} the tier's, no {norm} cap"
            )
        else:
            found += [
                f"{head}: {norm} {statistic} ours x{ratio:.9g} the tier's, above its cap x{gap.caps[norm]!r}"
                for statistic, ratio in zip(("median", "largest"), pair, strict=True)
                if not ratio <= gap.caps[norm]
            ]
    return found


def settle(
    verdicts: Mapping[str, Mapping],
    failures: Sequence[str],
    *,
    lengths: Sequence[int] | None,
) -> None:
    """Settle a judge's ``failures`` against the owner-accepted misses (``tests.cake_naive.accepted_check``) and
    :data:`KNOWN_GAP_ITEMS` for the running test: pass an owner-accepted miss within its caps; raise an
    ``AssertionError`` for a failure that is not a tier miss, an owner-accepted miss beyond its caps, a tier miss that
    is not listed, a listed output that met the tier, and a listed output that misses beyond its gap (:func:`beyond`);
    record the listed misses within their gaps (the conftest reports the test xfailed).  ``verdicts`` are
    :meth:`TierJudge.verdicts` (or their JSON form), ``lengths`` the case's document lengths (the gaps' scope)."""
    misses, others = {}, []
    for failure in failures:
        name = next(
            (
                n
                for n, v in verdicts.items()
                if v["tier"] is None
                and failure.startswith(f"{n}: misses the naive tier ")
            ),
            None,
        )
        if name is None:
            others.append(failure)
        else:
            misses[name] = failure
    unexplained = [
        n for n, v in verdicts.items() if v["tier"] is None and n not in misses
    ]
    if others or unexplained:
        raise AssertionError(
            "\n".join(
                [*failures, *(f"{n}: misses the naive tier" for n in unexplained)]
            )
        )
    test = current_test()
    problems = []
    for name in list(misses):
        entry, why = accepted_check(test, verdicts[name])
        if entry is not None:
            failure = misses.pop(name)
            if why:
                problems += [
                    *(f"{name}: an owner-accepted miss beyond its cap {accepted_cap(entry)}: {r}" for r in why),
                    failure,
                ]
    listed = KNOWN_GAP_ITEMS.get(test, {})
    problems += [failure for name, failure in misses.items() if name not in listed]
    problems += [
        f"{name}: listed as a known gap ({listed[name]}) but meets the naive tier now: remove it from KNOWN_GAP_ITEMS"
        for name in listed
        if name in verdicts and verdicts[name]["tier"] is not None
    ]
    for name, failure in misses.items():
        if name in listed:
            gap = KNOWN_GAPS[listed[name]]
            if reasons := beyond(name, gap, verdicts[name], lengths):
                problems += [*reasons, failure]
    if problems:
        raise AssertionError("\n".join(problems))
    record = _records.setdefault(test, {"judged": set(), "gaps": {}})
    record["judged"].update(verdicts)
    for name in misses:
        gap = KNOWN_GAPS[listed[name]]
        record["gaps"][name] = (gap.key, ratios(verdicts[name]), dict(gap.caps))


def _shown(r: tuple[float, ...], capped: Mapping[str, float]) -> str:
    cap = ", ".join(f"{norm} x{value!r}" for norm, value in capped.items())
    return f"l2 x{r[0]:.6f} / x{r[1]:.6f}, max x{r[2]:.6f} / x{r[3]:.6f} (cap {cap})"


def finish(test_id: str, report) -> None:
    """Report a passed call of ``test_id`` (pytest's call-phase ``report``) with recorded gaps as xfailed (the reason
    names each gap's key, the output, its ratios and caps), and one that did not judge an output listed for it as
    failed; a failed or skipped call is left as it is.  Clears the test's records."""
    test = canonical(test_id)
    record = _records.pop(test, None)
    if report.when != "call" or not report.passed:
        return
    listed = KNOWN_GAP_ITEMS.get(test, {})
    judged = record["judged"] if record else set()
    absent = [
        f"{name}: listed as a known gap ({key}) but not judged by the test: remove it from KNOWN_GAP_ITEMS"
        for name, key in listed.items()
        if name not in judged
    ]
    if absent:
        report.outcome = "failed"
        report.longrepr = "\n".join(absent)
    elif record and record["gaps"]:
        report.outcome = "skipped"
        report.wasxfail = (
            "owner-accepted naive-tier gaps within their caps: "
            + "; ".join(
                f"{name} {key}: {_shown(r, capped)}"
                for name, (key, r, capped) in sorted(record["gaps"].items())
            )
        )


def _cap_form(cap: object) -> bool:
    return (
        type(cap) is float
        and math.isfinite(cap)
        and cap >= NEAR_TIE
        and round(cap, 4) == cap
    )


def check_registry() -> list[str]:
    """Problems of the registry's form: a gap keyed under another key than its own, a scope outside :data:`SCOPES`, a cap of a norm
    outside :data:`NORMS` or not a finite float of at least :data:`NEAR_TIE` with at most four decimals; an item that
    is not a node id of ``tests/experimental/``, a key that is not a gap, or an output that is not one of its gap's."""
    problems = []
    for key, gap in KNOWN_GAPS.items():
        if gap.key != key:
            problems.append(f"{key}: keyed under another key than its own {gap.key}")
        if gap.scope not in SCOPES:
            problems.append(f"{key}: scope {gap.scope!r} is not one of {SCOPES}")
        for norm, cap in gap.caps.items():
            if norm not in NORMS:
                problems.append(
                    f"{key}: a cap of {norm!r}, not a norm of the judge {NORMS}"
                )
            elif not _cap_form(cap):
                problems.append(
                    f"{key}: {norm} cap {cap!r} is not a finite float >= {NEAR_TIE} with at most four decimals"
                )
    for test, outputs in KNOWN_GAP_ITEMS.items():
        if not test.startswith("tests/experimental/") or "::" not in test:
            problems.append(f"{test}: not a node id of tests/experimental/")
        for output, key in outputs.items():
            gap = KNOWN_GAPS.get(key)
            if gap is None:
                problems.append(
                    f"{test} {output}: {key!r} is not an owner-accepted gap {sorted(KNOWN_GAPS)}"
                )
            elif output.rpartition(".")[2] not in gap.outputs:
                problems.append(
                    f"{test} {output}: not an output of {key} {gap.outputs}"
                )
    return problems


__all__ = [
    "KNOWN_GAPS",
    "KNOWN_GAP_ITEMS",
    "SCOPES",
    "KnownGap",
    "beyond",
    "canonical",
    "check_registry",
    "current_test",
    "finish",
    "in_scope",
    "missed_norms",
    "ratios",
    "running",
    "settle",
]
