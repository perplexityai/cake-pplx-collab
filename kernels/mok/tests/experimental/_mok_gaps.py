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

The known accuracy gaps of the MoK kernels: outputs that miss the naive BF16 tier by a measured margin and that a
follow-up kernel change fixes.

A case with known-gap outputs is split into items that share one judged run (:func:`judged`): the main item
(:data:`OUTPUTS`) judges every other output of the case, and each known-gap output is judged in an item of its own.
On an architecture where the output is a known gap (:data:`CAPS`), the item carries a strict expected-failure marker
that expects only the tier miss (an ``AssertionError``); on the others it is judged like any output and must meet the
tier.  A known gap has caps per architecture, one for each norm the owner accepts the output's miss in: the largest
ratio of its median and of its largest error over the naive tier's (kernel error / naive error against the float64
oracle, over the draws the verdict counts in that norm) measured on that architecture, rounded up to 4 decimals.  A
known-gap output that misses a norm without a cap, misses beyond a cap, fails the zero rule, or misses where it has no
caps fails its item (:class:`GapCapError`, not the expected miss).  When the fix lands the item meets the tier, the
strict marker turns that into a failure, and the marker and its caps go with the fix.  Every output, a known gap or not, is
judged by the same rule (:class:`tests.cake_naive.TierJudge`, its two-stage max-norm rule included): an output that
meets the tier on an architecture is no known gap there.
"""

from __future__ import annotations

import decimal
import functools
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence

import pytest
import torch

from tests.cake_naive import NEAR_TIE, NORMS, PAIRED_NORMS, TIERS

#: The id of a case's main item.
OUTPUTS = "outputs"
#: The architectures the package targets, by compute capability.
TARGETS = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}

#: The caps of the known gaps: per known-gap item (``<test function>[<id>]``) and architecture, ``{norm: cap}`` for each
#: norm the owner accepts the output's miss in, the largest ratio of its median and of its largest error over the
#: tier's measured there, rounded up to 4 decimals (a float, at least the near tie).  An item is marked on exactly the
#: architectures it lists.
CAPS: dict[str, dict[str, dict[str, float]]] = {}

MXFP8_D_SCORES_WHAT = (
    "the MXFP8 kernels' score gradient (the fused form sum(dh * hidden) / score) misses the naive tier "
    "(dY * y_route).sum(-1)"
)
MXFP8_D_SCORES_FIX = "A follow-up kernel change fixes it."
BF16_STRADDLE_DW_WHAT = (
    "with BF16 weight gradients the kernel rounds a routed expert's dW to BF16 per macrobatch and adds the parts in "
    "BF16, so an expert whose rows cross a macrobatch boundary misses the naive tier (one GEMM per expert)"
)
BF16_STRADDLE_DW_FIX = "A follow-up kernel change (an FP32 carry for straddling experts) fixes it."


class GapCapError(Exception):
    """A known-gap output that misses the tier beyond its cap, or where it has no cap: a failure, never the expected
    miss of the item's marker."""


@functools.cache
def arch() -> str | None:
    """The generated target of GPU 0 (``sm_100a``, ``sm_103a`` or ``sm_107a``), or None without a supported GPU."""
    if not torch.cuda.is_available():
        return None
    return TARGETS.get(torch.cuda.get_device_capability(0))


def _cap_form(cap: object) -> bool:
    return type(cap) is float and math.isfinite(cap) and cap >= NEAR_TIE and round(cap, 4) == cap


def _checked(item: str, caps: Mapping[str, Mapping[str, float]]) -> Mapping[str, Mapping[str, float]]:
    """``caps`` of ``item`` (:data:`CAPS`), refused unless each architecture names a supported target and a nonempty
    ``{norm: cap}`` of norms of :data:`tests.cake_naive.NORMS` and caps of the cap form."""
    for target, norms in caps.items():
        if target not in TARGETS.values() or not norms or not all(
            norm in NORMS and _cap_form(cap) for norm, cap in norms.items()
        ):
            raise ValueError(f"{item}: caps {target}: {norms!r}: not a target and a nonempty {{norm: cap}}")
    return caps


def _caps_text(caps: Mapping[str, Mapping[str, float]]) -> str:
    return "; ".join(
        f"{target} " + ", ".join(f"{norm} {cap:.4f}" for norm, cap in norms.items())
        for target, norms in sorted(caps.items())
    )


def marker(item: str, what: str, fix: str) -> pytest.MarkDecorator:
    """The strict expected-failure marker of known-gap item ``item``: active on the architectures :data:`CAPS` lists
    for it, its reason naming each with its caps."""
    caps = _checked(item, CAPS.get(item, {}))
    reason = (
        f"known gap on {', '.join(sorted(caps)) or 'no architecture'}: {what}; caps (largest ratio of the median and of "
        f"the largest error, ours / naive, in each norm the miss is accepted in): {_caps_text(caps) or 'none'}.  {fix}"
    )
    return pytest.mark.xfail(arch() in caps, strict=True, raises=AssertionError, reason=reason)


def mxfp8_d_scores(item: str) -> pytest.MarkDecorator:
    """The marker of an MXFP8 score-gradient item."""
    return marker(item, MXFP8_D_SCORES_WHAT, MXFP8_D_SCORES_FIX)


def mxfp8_d_scores_of(function: str) -> pytest.MarkDecorator:
    """The marker of the score-gradient items of MXFP8 test ``function``, whose ``part`` parameter is shared by its
    cases: active on the architectures :data:`CAPS` lists for any of them (``<function>[...-d_scores]``), its reason
    naming each with the range of the items' caps.  Each item is still held to its own cap (:func:`check`)."""
    caps: dict[str, dict[str, list[float]]] = {}
    for item, item_caps in CAPS.items():
        if item.startswith(f"{function}[") and item.endswith("-d_scores]"):
            for target, norms in _checked(item, item_caps).items():
                for norm, cap in norms.items():
                    caps.setdefault(target, {}).setdefault(norm, []).append(cap)
    text = "; ".join(
        f"{target} "
        + ", ".join(
            f"{norm} {min(found):.4f}" + (f"-{max(found):.4f}" if max(found) > min(found) else "")
            for norm, found in norms.items()
        )
        for target, norms in sorted(caps.items())
    )
    reason = (
        f"known gap on {', '.join(sorted(caps)) or 'no architecture'}: {MXFP8_D_SCORES_WHAT}; caps of the items "
        f"(largest ratio of the median and of the largest error, ours / naive, in each norm the miss is accepted in): "
        f"{text or 'none'}.  {MXFP8_D_SCORES_FIX}"
    )
    return pytest.mark.xfail(arch() in caps, strict=True, raises=AssertionError, reason=reason)


def bf16_straddle_dw(item: str) -> pytest.MarkDecorator:
    """The marker of a BF16 straddle routed dW item."""
    return marker(item, BF16_STRADDLE_DW_WHAT, BF16_STRADDLE_DW_FIX)


def item_id(output: str) -> str:
    """The id of a known-gap item: the output's name (``:`` spelled ``.``)."""
    return output.replace(":", ".")


def parts(
    function: str, gaps: Iterable[tuple[str, Callable[[str], pytest.MarkDecorator]]], prefix: str = ""
) -> list:
    """The values of a case's ``part`` parameter for test ``function``: :data:`OUTPUTS`, then each known-gap output
    under its marker (``gaps``: ``(output, marker factory)``); ``prefix`` is the id of the case's other parameters
    (``<prefix>-<output>``), empty when ``part`` is the item's only parameter."""
    found = [pytest.param(OUTPUTS, id=f"{prefix}-{OUTPUTS}" if prefix else OUTPUTS)]
    for name, mark in gaps:
        ident = f"{prefix}-{item_id(name)}" if prefix else item_id(name)
        found.append(pytest.param(name, marks=mark(f"{function}[{ident}]"), id=ident))
    return found


_JUDGED: dict[tuple, object] = {}


def judged(key: tuple, run: Callable[[], object]) -> object:
    """``run()``'s result for ``key``, computed once per process: the items of one case share it."""
    if key not in _JUDGED:
        _JUDGED[key] = run()
    return _JUDGED[key]


def _miss(output: str) -> str:
    return f"{output}: misses the naive tier"


def miss_ratios(verdict: Mapping) -> dict[str, list[float] | None]:
    """For each norm the output does not meet (missed, or undecided), the ratios of its median and of its largest error
    over the tier's, over the draws the verdict counts in that norm (None where the tier has no error on every one of
    them): what a known gap's caps hold."""
    (tier,) = TIERS
    ratios: dict[str, list[float] | None] = {}
    for norm in NORMS:
        if verdict["met"][norm]:
            continue
        ours, theirs = verdict["norms"][norm]["ours"], verdict["norms"][norm][tier]
        ratios[norm] = (
            None
            if theirs is None
            else [
                mine / bound if bound > 0 else math.inf if mine > 0 else 1.0
                for mine, bound in zip(ours, theirs, strict=True)
            ]
        )
    return ratios


def beyond_caps(verdict: Mapping, caps: Mapping[str, float]) -> list[str]:
    """Why a known gap's output misses the tier beyond its ``caps`` (``{norm: cap}``), or nothing: a norm it does not
    meet (missed, or undecided) without a cap; in a norm with a cap, the ratio of its median or of its largest error
    over the tier's above the cap (:func:`miss_ratios`), or a tier without an error on every draw counted; a draw where
    the tier's error is 0 and its own is not; or a norm left undecided without stage 2."""
    (tier,) = TIERS
    beyond = []
    for norm, ratios in miss_ratios(verdict).items():
        if norm not in caps:
            beyond.append(f"{norm}: misses the tier, and the known gap has no cap in it")
        elif ratios is None:
            beyond.append(f"{norm}: the tier has no error on every draw counted")
        else:
            beyond += [
                f"{norm} {statistic} ours x{ratio!r} the tier's, above the cap x{caps[norm]!r}"
                for statistic, ratio in zip(("median", "largest"), ratios, strict=True)
                if ratio > caps[norm]
            ]
    if draws := verdict["zero"][tier]:
        beyond.append(f"tier {tier}: error 0 on draws {draws}, ours not")
    if verdict["undecided"]:
        beyond.append(f"{', '.join(verdict['undecided'])} undecided: stage 2 was not run")
    return beyond


def cap_of(ratio: float) -> float:
    """The cap of a measured ratio: rounded up to 4 decimals (never below the ratio), at least the near tie."""
    return max(NEAR_TIE, float(decimal.Decimal(ratio).quantize(decimal.Decimal("0.0001"), decimal.ROUND_CEILING)))


def current_item() -> str:
    """The running item's ``<test function>[<id>]`` (pytest's ``PYTEST_CURRENT_TEST``)."""
    current = os.environ.get("PYTEST_CURRENT_TEST", "")
    return current.rsplit(" (", 1)[0].rsplit("::", 1)[-1]


def check(record_property, part: str, verdicts: Sequence[dict], failures: Sequence[str], gaps: Iterable[str]) -> None:
    """Record and judge one item of a case: :data:`OUTPUTS` takes every verdict and failure except the known-gap
    outputs' tier misses; a known-gap item takes its own output's verdict and miss.  A known-gap output that misses the
    tier records its ratios and caps (``known_gap``: :func:`miss_ratios`) and raises :class:`GapCapError` where it has
    no caps on this architecture or misses beyond them (:func:`beyond_caps`)."""
    gaps = tuple(gaps)
    if part == OUTPUTS:
        for verdict in verdicts:
            if verdict["output"] not in gaps:
                record_property("naive_tier", verdict)
        found = [f for f in failures if not f.startswith(tuple(_miss(name) for name in gaps))]
    else:
        assert part in gaps, (part, gaps)
        (verdict,) = (v for v in verdicts if v["output"] == part)
        record_property("naive_tier", verdict)
        found = [f for f in failures if f.startswith(_miss(part))]
        if found:
            item, target = current_item(), arch()
            caps = _checked(item, CAPS.get(item, {})).get(target)
            beyond = ["no caps on this architecture"] if caps is None else beyond_caps(verdict, caps)
            record_property(
                "known_gap",
                {
                    "item": item,
                    "arch": target,
                    "caps": None if caps is None else dict(caps),
                    "ratios": miss_ratios(verdict),
                    "beyond": beyond,
                },
            )
            if beyond:
                raise GapCapError(
                    f"{part}: misses the naive tier on {target} beyond its known gap: {'; '.join(beyond)}\n"
                    + "\n".join(found)
                )
    assert not found, "\n".join(found)
