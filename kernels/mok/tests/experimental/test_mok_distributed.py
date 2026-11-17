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
printed as ``TIER`` lines.

Backend suite: the package is imported only through :func:`tests.cake.test_helpers.cake_package.cake_package`.
"""

import pytest

from tests.experimental import _mok_dist as workers
from tests.experimental._mok_spawn import run_ranks


def _record(results, record_property):
    """Record rank 0's accuracy verdicts (every rank judged the same gathered values) and fail on any miss."""
    failures = []
    for record in results[0]:
        if "judged" not in record:
            continue
        verdicts, alternatives, misses = record["judged"]
        for verdict in verdicts:
            record_property("naive_tier", dict(verdict, output=f"{record['case']}:{verdict['output']}"))
        for verdict in alternatives:
            record_property("naive_alternative", dict(verdict, output=f"{record['case']}:{verdict['output']}"))
        failures += [f"{record['case']}: {miss}" for miss in misses]
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("world", [1, 4], ids=["ep1", "ep4"])
def test_training_graph(record_property, world):
    _record(run_ranks(world, workers.training_graph), record_property)


def test_unequal_sources(record_property):
    _record(run_ranks(4, workers.unequal_sources), record_property)


@pytest.mark.parametrize("world", [1, 4], ids=["ep1", "ep4"])
def test_training_features(record_property, world):
    results = run_ranks(world, workers.training_features)
    assert {r["case"] for r in results[0]} == {c["name"] for c in workers.FEATURE_CASES}
    _record(results, record_property)


def test_source_capacity_and_context_ownership():
    run_ranks(4, workers.source_capacity)


def test_backward_releases_launcher_scratch_before_epilogue():
    run_ranks(4, workers.backward_scratch)
