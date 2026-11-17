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
tier.  Each known gap has a cap per architecture: the largest error ratio (kernel error / naive error against the
float64 oracle, median or largest over the draws, in each norm the output misses) measured on that architecture,
rounded up to 4 decimals.  A known-gap output whose ratio exceeds its cap, or that misses the tier where it has no cap,
fails its item (:class:`GapCapError`, not the expected miss).  When the fix lands the item meets the tier, the strict
marker turns that into a failure, and the marker and its caps go with the fix.  Every output, a known gap or not, is
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

#: The caps of the known gaps: per known-gap item (``<test function>[<id>]``) and architecture, the largest error ratio
#: measured there in the norms the output misses, rounded up to 4 decimals (at least the near tie).  An item is marked
#: on exactly the architectures it lists.
CAPS: dict[str, dict[str, float]] = {}

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


def _caps_text(caps: Mapping[str, float]) -> str:
    return ", ".join(f"{target} {cap:.4f}" for target, cap in sorted(caps.items()))


def marker(item: str, what: str, fix: str) -> pytest.MarkDecorator:
    """The strict expected-failure marker of known-gap item ``item``: active on the architectures :data:`CAPS` lists
    for it, its reason naming each with its cap."""
    caps = CAPS.get(item, {})
    reason = (
        f"known gap on {', '.join(sorted(caps)) or 'no architecture'}: {what}; cap (largest error ratio, ours / "
        f"naive, in the norms it misses): {_caps_text(caps) or 'none'}.  {fix}"
    )
    return pytest.mark.xfail(arch() in caps, strict=True, raises=AssertionError, reason=reason)


def mxfp8_d_scores(item: str) -> pytest.MarkDecorator:
    """The marker of an MXFP8 score-gradient item."""
    return marker(item, MXFP8_D_SCORES_WHAT, MXFP8_D_SCORES_FIX)


def mxfp8_d_scores_of(function: str) -> pytest.MarkDecorator:
    """The marker of the score-gradient items of MXFP8 test ``function``, whose ``part`` parameter is shared by its
    cases: active on the architectures :data:`CAPS` lists for any of them (``<function>[...-d_scores]``), its reason
    naming each with the range of the items' caps.  Each item is still held to its own cap (:func:`check`)."""
    caps: dict[str, list[float]] = {}
    for item, item_caps in CAPS.items():
        if item.startswith(f"{function}[") and item.endswith("-d_scores]"):
            for target, cap in item_caps.items():
                caps.setdefault(target, []).append(cap)
    text = ", ".join(
        f"{target} {min(found):.4f}" + (f"-{max(found):.4f}" if max(found) > min(found) else "")
        for target, found in sorted(caps.items())
    )
    reason = (
        f"known gap on {', '.join(sorted(caps)) or 'no architecture'}: {MXFP8_D_SCORES_WHAT}; caps of the items "
        f"(largest error ratio, ours / naive, in the norms they miss): {text or 'none'}.  {MXFP8_D_SCORES_FIX}"
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


def miss_ratio(verdict: Mapping) -> float:
    """The largest error ratio of a verdict that misses the tier, ours / naive, median or largest, over the norms it
    misses (the L2 norm by its median and largest, a norm judged draw by draw by its wins); inf where a tier error is 0
    or not finite, or where the zero rule fails."""
    (tier,) = TIERS
    if any(verdict["zero"].values()):
        return math.inf
    ratio = 0.0
    for norm in NORMS:
        errors = verdict["norms"][norm]
        bound, ours = errors[tier], errors["ours"]
        if norm in PAIRED_NORMS:
            missed = bound is None or verdict["wins"][norm] < verdict["needed"][norm] or verdict["undecided"]
        else:
            missed = bound is None or ours[0] > NEAR_TIE * bound[0] or ours[1] > NEAR_TIE * bound[1]
        if not missed:
            continue
        if bound is None or not (bound[0] > 0 and bound[1] > 0):
            return math.inf
        ratio = max(ratio, ours[0] / bound[0], ours[1] / bound[1])
    return ratio


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
    tier records its ratio (``known_gap``, :func:`miss_ratio`) and raises :class:`GapCapError` beyond its cap on this
    architecture or without one."""
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
            ratio = miss_ratio(verdict)
            cap = CAPS.get(item, {}).get(target)
            record_property("known_gap", {"item": item, "arch": target, "ratio": ratio, "cap": cap})
            if cap is None or not ratio <= cap:
                why = "no cap there" if cap is None else f"its cap there is {cap:.4f}"
                raise GapCapError(
                    f"{part}: misses the naive tier on {target} with error ratio {ratio:.9g}, and {why}:\n"
                    + "\n".join(found)
                )
    assert not found, "\n".join(found)
