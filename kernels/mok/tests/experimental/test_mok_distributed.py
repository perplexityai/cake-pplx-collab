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

Independent single-GPU fixtures for the MoK kernel tests.
Multi-GPU MoK tests: EP1 and EP4 groups on one node, each test starting its own processes
(:func:`tests.experimental._mok_spawn.run_ranks`; run with the ``SLURM_*`` task variables unset).

The workers live in :mod:`tests.experimental._mok_dist`: the complete training step eager and under CUDA Graph
replay, unequal source counts per rank (empty ranks included), the training features (clamped SwiGLU, recompute and
the context-free forward, FP32 weight-gradient accumulation, native MXFP8) and the adapter's source-length guards and
scratch lifetime.  Each worker checks the contract and repeatability in every rank; the accuracy verdicts of each case
(the whole group's outputs against the float64 oracle beside the naive BF16 implementation) are recorded here and
printed as ``TIER`` lines.  The known accuracy gaps (:mod:`tests.experimental._mok_gaps`) are judged in items of their
own under a strict expected-failure marker, from the same group run: the MXFP8 score gradient of the MXFP8 feature
cases, and the BF16 routed weight gradients of the cases whose experts' rows cross a macrobatch boundary and miss the
tier.

Backend suite: the package is imported only through :func:`tests.cake.test_helpers.cake_package.cake_package`.
"""

import pytest

from tests.experimental import _mok_dist as workers
from tests.experimental import _mok_gaps as gaps
from tests.experimental._mok_spawn import run_ranks


def _run(world, worker):
    """The group run of ``worker`` over ``world`` ranks, shared by the items of one case."""
    return gaps.judged((worker.__name__, world), lambda: run_ranks(world, worker))


def _record(results, record_property, part=gaps.OUTPUTS, known=()):
    """Record and judge rank 0's accuracy verdicts (every rank judged the same gathered values), output names
    ``<case>:<output>``: item ``part`` of :func:`tests.experimental._mok_gaps.check`."""
    verdicts, failures = [], []
    for record in results[0]:
        if "judged" not in record:
            continue
        case_verdicts, alternatives, misses = record["judged"]
        verdicts += [dict(verdict, output=f"{record['case']}:{verdict['output']}") for verdict in case_verdicts]
        if part == gaps.OUTPUTS:
            for verdict in alternatives:
                record_property("naive_alternative", dict(verdict, output=f"{record['case']}:{verdict['output']}"))
        failures += [f"{record['case']}:{miss}" for miss in misses]
    gaps.check(record_property, part, verdicts, failures, known)


def _cases(world_gaps):
    """``(world, part)`` parameters: per world the main item, then each known-gap output under its marker."""
    cases = []
    for world, known in world_gaps.items():
        cases.append(pytest.param(world, gaps.OUTPUTS, id=f"ep{world}-{gaps.OUTPUTS}"))
        cases += [pytest.param(world, name, marks=mark, id=f"ep{world}-{gaps.item_id(name)}") for name, mark in known]
    return cases


@pytest.mark.parametrize("world", [1, 4], ids=["ep1", "ep4"])
def test_training_graph(record_property, world):
    _record(_run(world, workers.training_graph), record_property)


#: The known-gap outputs of the unequal-source cases: the 0 / 513 / 769 / 8193 case's experts cross macrobatch
#: boundaries (BF16 routed dW).
UNEQUAL_GAPS = tuple(
    (f"counts-0-513-769-8193:{name}", gaps.BF16_STRADDLE_DW)
    for name in ("dw_routed_gate", "dw_routed_up", "dw_routed_down")
)


@pytest.mark.parametrize("part", gaps.parts(UNEQUAL_GAPS))
def test_unequal_sources(record_property, part):
    _record(_run(4, workers.unequal_sources), record_property, part, [name for name, _ in UNEQUAL_GAPS])


_MXFP8_FEATURE_GAPS = tuple(
    (f"{case}:d_scores", gaps.MXFP8_D_SCORES) for case in ("mxfp8", "mxfp8-fp32-wgrad", "mxfp8-clamped-recompute")
)
#: The known-gap outputs of the training features per world: the MXFP8 score gradient, and at EP1 the routed down
#: projection's BF16 dW of the clamped cases (their experts cross macrobatch boundaries).
FEATURE_GAPS = {
    1: (
        *_MXFP8_FEATURE_GAPS,
        ("bf16-clamped:dw_routed_down", gaps.BF16_STRADDLE_DW),
        ("bf16-clamped-recompute:dw_routed_down", gaps.BF16_STRADDLE_DW),
    ),
    4: _MXFP8_FEATURE_GAPS,
}


@pytest.mark.parametrize("world,part", _cases(FEATURE_GAPS))
def test_training_features(record_property, world, part):
    results = _run(world, workers.training_features)
    if part == gaps.OUTPUTS:
        assert {r["case"] for r in results[0]} == {c["name"] for c in workers.FEATURE_CASES}
    _record(results, record_property, part, [name for name, _ in FEATURE_GAPS[world]])


def test_source_capacity_and_context_ownership():
    run_ranks(4, workers.source_capacity)


def test_backward_releases_launcher_scratch_before_epilogue():
    run_ranks(4, workers.backward_scratch)
