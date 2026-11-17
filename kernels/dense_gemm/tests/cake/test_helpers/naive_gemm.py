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

The accuracy of a dense_gemm case, judged against the naive BF16 tier (``tests/cake_naive.py``).

A GEMM output has no tolerance.  :func:`judge_gemm` draws the case's inputs :data:`DRAWS` times, launches the kernel
on each draw and judges every output against the float64 product of the same operands (the oracle), beside the naive
tier: the same product in BF16 through torch (``A.bfloat16() @ B.bfloat16()``; FP32 router operands are rounded to
BF16 first).  Each output is held to its contract on every draw: the oracle's shape, finite where the oracle is.

The case supplies three callables:

- ``draw(seed)`` makes the draw's inputs from ``seed`` (a ``torch.Generator`` seeded with it), as fresh tensors or
  written in place into the tensors a prepared launch is bound to.  It returns them in any form ``launch`` and
  ``product`` read, usually a dict or a ``SimpleNamespace``.
- ``launch(inputs)`` runs the kernel once and returns its outputs by name, each in the orientation ``product`` gives
  it (``out.t()`` for a transposed store, a view without the padding of a padded output).
- ``product(inputs, dtype)`` returns the same names: the GEMM of each output on the operands converted to ``dtype``.
  float64 gives the oracle and bfloat16 the naive tier, so one formula serves both.  It is called on the draw's
  inputs before the draw's first launch, so it reads only operands that the launches do not write; a chain of
  GEMMs is the product of the whole chain on the draw's inputs, not of a kernel's intermediate output.

``launches`` launches the kernel that many times on the first draw (an output is written into NaN again before each
launch after the first when ``poison`` is given); the judge keeps the largest error of a draw's launches.

The judge's verdicts are recorded through pytest's ``record_property``, and ``tests/conftest.py`` prints one ``TIER``
line per output in the terminal summary.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from tests.cake_naive import DRAWS, TIERS, TierJudge, contract, draw_seed, seed_all

__all__ = ["DRAWS", "judge_gemm", "nan_fill"]


def nan_fill(outputs: Iterable[torch.Tensor]) -> None:
    """Write NaN into every floating output, so an element a launch leaves unwritten fails its contract."""
    for t in outputs:
        t.fill_(float("nan"))


def judge_gemm(
    record_property: Callable[[str, object], None] | None,
    *,
    seed: int,
    draw: Callable[[int], Any],
    launch: Callable[[Any], Mapping[str, torch.Tensor]],
    product: Callable[[Any, torch.dtype], Mapping[str, torch.Tensor]],
    launches: int = 1,
    poison: Callable[[Any], Iterable[torch.Tensor]] | None = None,
    draws: int = DRAWS,
) -> Any:
    """Judge one case over ``draws`` inputs against the naive BF16 tier and assert it met the tier.

    Draw ``j`` uses the seed :func:`~tests.cake_naive.draw_seed` of ``seed`` and ``j``; :func:`seed_all` seeds the
    global random state with it too.  The Python, torch CPU and every CUDA device's random state are restored after
    the last draw.  ``record_property`` (pytest's fixture, or None) records each output's verdict.  Fails with every
    contract and judge failure.  Returns the last draw's inputs, so the test can run its other checks on them."""
    if launches < 1:
        raise ValueError(f"launches is at least 1, got {launches}")
    judge = TierJudge(draws)
    failures: list[str] = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    inputs = None
    try:
        for j in range(draws):
            s = draw_seed(seed, j)
            seed_all(s)
            inputs = draw(s)
            oracle = dict(product(inputs, torch.float64))
            naive = dict(product(inputs, torch.bfloat16))
            for k in range(launches if j == 0 else 1):
                if k and poison is not None:
                    nan_fill(poison(inputs))
                outputs = dict(launch(inputs))
                if set(outputs) != set(oracle):
                    raise ValueError(f"launch gave outputs {sorted(outputs)}, product gave {sorted(oracle)}")
                for name, actual in outputs.items():
                    want = oracle[name]
                    if message := contract(name, actual, want):
                        failures.append(f"draw {j} launch {k}: {message}")
                        if actual.shape != want.shape:
                            continue
                    judge.add(name, j, actual, want, {tier: naive[name] for tier in TIERS})
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    if record_property is not None:
        judge.record(record_property)
    failures += judge.failures()
    assert not failures, "\n".join(failures)
    return inputs
