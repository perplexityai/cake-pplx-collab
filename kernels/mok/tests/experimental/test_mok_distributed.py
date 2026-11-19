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
the context-free forward, FP32 weight-gradient accumulation, native MXFP8), the router weight before the down
projection through the adapter (EP1 on SM100a / SM103a) and the adapter's source-length guards and scratch lifetime.
Each worker checks the contract and repeatability in every rank; the accuracy verdicts of each case (the whole
group's outputs against the float64 oracle beside the naive BF16 implementation) are recorded here and
printed as ``TIER`` lines.  The known accuracy gaps (:mod:`tests.experimental._mok_gaps`) are judged in items of their
own under a strict expected-failure marker, from the same group run: the BF16 routed weight gradients of the cases
whose experts' rows cross a macrobatch boundary and miss the tier.

Backend suite: the package is imported only through :func:`tests.cake.test_helpers.cake_package.cake_package`.
"""

import pytest
import torch

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


def _cases(function, world_gaps):
    """``(world, part)`` parameters of test ``function``: per world the main item, then each known-gap output under its
    marker."""
    return [
        pytest.param(world, *p.values, marks=p.marks, id=p.id)
        for world, known in world_gaps.items()
        for p in gaps.parts(function, known, f"ep{world}")
    ]


@pytest.mark.parametrize("world", [1, 4], ids=["ep1", "ep4"])
def test_training_graph(record_property, world):
    _record(_run(world, workers.training_graph), record_property)


#: The known-gap outputs of the unequal-source cases: the 0 / 513 / 769 / 8193 case's experts cross macrobatch
#: boundaries (BF16 routed dW).
UNEQUAL_GAPS = tuple(
    (f"counts-0-513-769-8193:{name}", gaps.bf16_straddle_dw)
    for name in ("dw_routed_gate", "dw_routed_up", "dw_routed_down")
)


@pytest.mark.parametrize("part", gaps.parts("test_unequal_sources", UNEQUAL_GAPS))
def test_unequal_sources(record_property, part):
    _record(_run(4, workers.unequal_sources), record_property, part, [name for name, _ in UNEQUAL_GAPS])


#: The known-gap outputs of the training features per world: at EP1 (1031 tokens, 768-row macrobatches: experts cross a
#: macrobatch boundary) the BF16 routed dW that misses the tier, the down projection of the clamped cases and all three
#: projections of the plain recompute case.  The MXFP8 score gradient of these cases meets the tier.
FEATURE_GAPS = {
    1: (
        ("bf16-clamped:dw_routed_down", gaps.bf16_straddle_dw),
        ("bf16-recompute:dw_routed_gate", gaps.bf16_straddle_dw),
        ("bf16-recompute:dw_routed_up", gaps.bf16_straddle_dw),
        ("bf16-recompute:dw_routed_down", gaps.bf16_straddle_dw),
        ("bf16-clamped-recompute:dw_routed_down", gaps.bf16_straddle_dw),
    ),
    4: (),
}


@pytest.mark.parametrize("world,part", _cases("test_training_features", FEATURE_GAPS))
def test_training_features(record_property, world, part):
    results = _run(world, workers.training_features)
    if part == gaps.OUTPUTS:
        assert {r["case"] for r in results[0]} == {c["name"] for c in workers.FEATURE_CASES}
    _record(results, record_property, part, [name for name, _ in FEATURE_GAPS[world]])


def test_source_capacity_and_context_ownership():
    run_ranks(4, workers.source_capacity)


def test_backward_releases_launcher_scratch_before_epilogue():
    run_ranks(4, workers.backward_scratch)


@pytest.mark.parametrize("world", [1], ids=["ep1"])
def test_before_down_training(record_property, world):
    """``router_weight_position="before_down"`` through the adapter: contract, repeatability, graph replay, recompute,
    the context-free forward and a test-local autograd function with the saved context's bytes; judged."""
    if torch.cuda.is_available() and torch.cuda.device_count() >= world:
        if any(torch.cuda.get_device_capability(index) == (10, 7) for index in range(world)):
            pytest.skip("The router weight before the down projection has SM100a and SM103a builds only")
    results = _run(world, workers.before_down_training)
    assert [r["case"] for r in results[0]] == ["bf16-clamped-before-down"]
    _record(results, record_property)
