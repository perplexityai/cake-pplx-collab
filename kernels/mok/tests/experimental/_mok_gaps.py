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
(:data:`OUTPUTS`) judges every other output of the case, and each known-gap output is judged in an item of its own
under a strict expected-failure marker that expects only the tier miss (an ``AssertionError``; any other error
fails the item).  The marker states the measured ratios (kernel error / naive error against the
float64 oracle over :data:`tests.cake_naive.DRAWS` draws, median / largest).  When the fix lands the item passes,
the strict marker turns that into a failure, and the marker goes with the fix.  Near ties are not listed here: they
stay judged in the main item.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import pytest

#: The id of a case's main item.
OUTPUTS = "outputs"

MXFP8_D_SCORES = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "known gap: the MXFP8 kernels' score gradient (the fused form sum(dh * hidden) / score) misses the naive "
        "tier (dY * y_route).sum(-1) on every architecture; L2 error ratio, median / largest: single GPU "
        "1.46-1.84 / 1.45-1.86 (sm_100a, sm_103a) and 1.44-1.84 / 1.46-1.84 (sm_107a), EP1 and EP4 1.71-1.72 / "
        "1.70-1.73 (sm_107a).  A follow-up kernel change fixes it."
    ),
)

BF16_STRADDLE_DW = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "known gap: with BF16 weight gradients the kernel rounds a routed expert's dW to BF16 per macrobatch and "
        "adds the parts in BF16, so an expert whose rows cross a macrobatch boundary misses the naive tier (one "
        "GEMM per expert): in the L2 norm, error ratio median / largest 1.033-1.038 / 1.033-1.072, or with its L2 "
        "below the tier's in the max norm, median ratio 1.05-1.14 (sm_103a, sm_107a); its L2 error is 0.92-0.97 "
        "times a per-macrobatch naive's.  A follow-up kernel change (an FP32 carry for straddling experts) fixes it."
    ),
)


def item_id(output: str) -> str:
    """The id of a known-gap item: the output's name (``:`` spelled ``.``)."""
    return output.replace(":", ".")


def parts(gaps: Iterable[tuple[str, pytest.MarkDecorator]]) -> list:
    """The values of a case's ``part`` parameter: :data:`OUTPUTS`, then each known-gap output under its marker."""
    return [OUTPUTS, *(pytest.param(name, marks=mark, id=item_id(name)) for name, mark in gaps)]


_JUDGED: dict[tuple, object] = {}


def judged(key: tuple, run: Callable[[], object]) -> object:
    """``run()``'s result for ``key``, computed once per process: the items of one case share it."""
    if key not in _JUDGED:
        _JUDGED[key] = run()
    return _JUDGED[key]


def _miss(output: str) -> str:
    return f"{output}: misses the naive tier"


def check(record_property, part: str, verdicts: Sequence[dict], failures: Sequence[str], gaps: Iterable[str]) -> None:
    """Record and judge one item of a case: :data:`OUTPUTS` takes every verdict and failure except the known-gap
    outputs' tier misses; a known-gap item takes its own output's verdict and miss."""
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
    assert not found, "\n".join(found)
