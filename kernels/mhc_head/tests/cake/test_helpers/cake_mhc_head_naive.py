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

The float64 oracle and the BF16 naive tier of the mHC training head tests, and the judge of a case over its draws.

The head is the stream mean ``Y = X.mean(dim=1)`` of BF16 ``X`` ``[T, 4, H]`` and its gradient ``dX[:, i, :] = dY / 4``
(with ``accumulate``, ``dX0 + dY / 4``).  The oracle is the exact value in float64; where PyTorch's arithmetic (the
FP32 sum of the streams from zero in stream order, one scaling by 0.25, one BF16 rounding; ``BF16(dY / 4)`` and the
BF16 add of the accumulation) is not finite, it is that value.  The naive tier is PyTorch on the same BF16 inputs:
``X.mean(dim=1)``, ``dY * 0.25`` on each stream and ``dX0 + dY * 0.25``, in BF16.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping

import torch

from tests.cake_naive import DRAWS, TIERS, TierJudge, contract, draw_seed, seed_all

STREAMS = 4
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _where_finite(exact: torch.Tensor, torch_value: torch.Tensor) -> torch.Tensor:
    """``exact`` where PyTorch's value is finite, else PyTorch's value (an infinity or a NaN), in float64."""
    return torch.where(torch_value.isfinite(), exact, torch_value.to(F64))


def forward_oracle(X: torch.Tensor) -> torch.Tensor:
    """``Y`` in float64: the exact mean of the four streams."""
    acc = torch.zeros(X.shape[0], X.shape[2], device=X.device, dtype=F32)
    for s in range(STREAMS):
        acc = acc + X[:, s].float()
    return _where_finite(X.double().sum(dim=1) * 0.25, (acc * 0.25).to(BF16))


def forward_naive(X: torch.Tensor) -> torch.Tensor:
    """``Y`` of the naive tier: PyTorch's mean over the streams of the BF16 ``X``."""
    return X.mean(dim=1)


def backward_oracle(dY: torch.Tensor, dX0: torch.Tensor | None = None) -> torch.Tensor:
    """``dX`` in float64: ``dY / 4`` on each stream, plus ``dX0`` when accumulating."""
    T, H = dY.shape
    exact = (dY.double() * 0.25)[:, None, :].expand(T, STREAMS, H)
    q = (dY.float() * 0.25).to(BF16)
    if dX0 is None:
        torch_value = q[:, None, :].expand(T, STREAMS, H)
    else:
        exact = dX0.double() + exact
        torch_value = (dX0.float() + q.float()[:, None, :]).to(BF16)
    return _where_finite(exact, torch_value).contiguous()


def backward_naive(dY: torch.Tensor, dX0: torch.Tensor | None = None) -> torch.Tensor:
    """``dX`` of the naive tier: ``dY * 0.25`` on each stream, plus ``dX0`` when accumulating, in BF16."""
    T, H = dY.shape
    grad = (dY * 0.25)[:, None, :].expand(T, STREAMS, H)
    if dX0 is not None:
        grad = dX0 + grad
    return grad.contiguous()


def judge_draws(
    case: Callable[
        [int, int], tuple | Iterable[tuple[Iterable[Mapping], Mapping, Mapping]]
    ],
    record_property: Callable[[str, object], None] | None,
    *,
    seed: int,
    draws: int = DRAWS,
    prefix: str = "",
) -> TierJudge:
    """Judge one case over ``draws`` inputs with ``tests/cake_naive.py``.

    For each draw ``j``, :func:`~tests.cake_naive.seed_all` seeds the random state with
    :func:`~tests.cake_naive.draw_seed` of ``seed`` and ``j``; ``case(j, s)`` (``s`` that seed) makes the draw's
    inputs, launches the package and returns ``(launches, expected, naive)``: the outputs of every launch on those
    inputs (a list of name -> tensor), the oracle's value of each judged output and the naive tier's (BF16), or
    yields such triples one after another.  Every launch's every output is held to its contract and recorded in the
    judge (a second launch of a draw keeps the larger of the errors).  The random state is restored after the last
    draw, the verdicts are recorded with ``record_property`` (one ``TIER`` line each in the terminal summary) and
    every failure is asserted.  Output names take ``prefix``."""
    judge = TierJudge(draws)
    failures: list[str] = []
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = (random.getstate(), torch.get_rng_state())
    try:
        for j in range(draws):
            s = draw_seed(seed, j)
            seed_all(s)
            result = case(j, s)
            for launches, expected, naive in (
                [result] if isinstance(result, tuple) else result
            ):
                for launch in launches:
                    for name, actual in launch.items():
                        want = expected[name]
                        if message := contract(prefix + name, actual, want):
                            failures.append(f"draw {j}: {message}")
                            if actual.shape != want.shape:
                                continue
                        judge.add(
                            prefix + name,
                            j,
                            actual,
                            want,
                            {tier: naive[name] for tier in TIERS},
                        )
                del launches, expected, naive
    finally:
        random.setstate(state[0])
        torch.set_rng_state(state[1])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    failures += judge.failures()
    if record_property is not None:
        judge.record(record_property)
    assert not failures, "\n".join(failures)
    return judge
