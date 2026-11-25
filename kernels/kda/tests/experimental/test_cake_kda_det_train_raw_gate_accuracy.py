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

Accuracy of the raw forget-gate mode of the KDA chunked training backend (``use_gate_in_kernel=True``) and of
the explicit path a training stack runs in its place, each judged against the naive BF16 tier
(``tests/cake_naive.py``).

Arms, each judged by ``judge_arms`` (``test_helpers/cake_kda_naive.py``) against the exact FP64 recurrence whose
gate is activated from the same BF16 raw gate and FP32 parameters in FP64, beside the error of the naive BF16 tier
(PyTorch running the same ops in BF16, the gate activated in the same graph) on the same inputs, over the judge's
32 draws:

* ``""`` (outputs ``out``, ``dq``, ...): the raw path, ``cake_kda_det(raw, use_gate_in_kernel=True, A_log=...,
  dt_bias=..., lower_bound=-5.0)``;
* ``explicit`` (outputs ``explicit.out``, ...): the explicit path with a torch FP32 gate, autograd on the BF16
  raw gate and the FP32 parameters, then ``cake_kda_det(g FP32)``.

Outputs: ``out``, ``dq``, ``dk``, ``dv``, ``dbeta``, the raw gate's ``dg``, ``dA_log`` and ``ddt_bias``.  The arms
are not compared with each other: each meets the tier, or the test fails (unless the owner listed the miss as a
known gap, ``test_helpers/cake_kda_known_gaps.py``).

Rows (lengths, heads, ``dt_bias`` shift): A, B, D and E run by default; C (16384 tokens at 4 heads) and F (64
heads) run with ``CAKE_PPLX_KDA_TRAIN_RAW_GATE_ACCURACY=1``.  The default draw's mean activated gate is about
-2.5; D shifts ``dt_bias`` by +6 (``s -> 1``, the cancellation case of the explicit sigmoid's derivative); E by -6
(slow decay, ``g ~ -0.012``).
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    raw_gate,
)
from tests.experimental.test_helpers.cake_kda_naive import draw_case, judge_arms

cake_backend = cake_package("cake.cake_kda_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_det_train", "cake_jit")

ENV_RUN = "CAKE_PPLX_KDA_TRAIN_RAW_GATE_ACCURACY"


def _flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no")


RUN = _flag(ENV_RUN)

# rows: (lengths, heads, dt_bias shift); default gate: mean activated gate about -2.5; D: s -> 1 (dt_bias ~ +6 +
# 0.5 N, the cancellation case of the explicit sigma'); E: slow decay (dt_bias ~ -6 + 0.5 N, g ~ -0.012)
ROWS: dict[str, tuple[tuple[int, ...], int, float]] = {
    "A": ((100, 640, 77, 1000, 129), 2, 0.0),
    "B": ((63, 1, 65), 4, 0.0),
    "C": ((5000, 7001, 4383), 4, 0.0),
    "D": ((4096,), 4, 6.0),
    "E": ((1500, 2596), 4, -6.0),
    "F": ((1000, 1100, 900, 1096), 64, 0.0),
}
# the rows that run only with ENV_RUN set: the production-size pack C and the 64-head row F
OPT_IN_ROWS = ("C", "F")
assert sum(ROWS["C"][0]) == 16384 and sum(ROWS["D"][0]) == sum(ROWS["E"][0]) == 4096
assert sum(ROWS["F"][0]) == 4096
# the base seed of row r's draws (judge_arms draws every input of draw j from draw_seed(base, j))
SEED_BASE = 3000
SEEDS = {row: SEED_BASE + 100 * i for i, row in enumerate(ROWS)}


# ---- the arms ---------------------------------------------------------------------------------------------------
RAW_LEAVES = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
LEAF_RESULTS = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")


def _leaves(inputs: dict) -> list[torch.Tensor]:
    return [inputs[n].detach().clone().requires_grad_(True) for n in RAW_LEAVES]


def _results(out: torch.Tensor, leaves) -> dict:
    res = {"out": out.detach()}
    for name, leaf in zip(LEAF_RESULTS, leaves, strict=True):
        res[name] = leaf.grad.detach()
    return res


def arm_raw(inputs: dict) -> dict:
    """The raw path through the autograd entry."""
    leaves = _leaves(inputs)
    q, k, v, raw, beta, a_log, dt_bias = leaves
    out = cake_backend.cake_kda_det(
        q,
        k,
        v,
        raw,
        beta,
        inputs["cu_seqlens"],
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
    )
    out.backward(inputs["dout"])
    return _results(out, leaves)


def torch_gate(raw: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor):
    """The explicit arm's gate: torch FP32 on the BF16 raw gate and the FP32 parameters."""
    return raw_gate(raw, a_log, dt_bias, LOWER_BOUND, torch.float32)


def arm_explicit(inputs: dict) -> dict:
    """The explicit arm: :func:`torch_gate` activates the gate (FP32, differentiable in the raw gate and the
    parameters), then ``cake_kda_det(g)``; autograd carries ``dg`` back through the gate."""
    leaves = _leaves(inputs)
    q, k, v, raw, beta, a_log, dt_bias = leaves
    g = torch_gate(raw, a_log, dt_bias)
    assert g.dtype == torch.float32 and g.shape == raw.shape, (g.dtype, g.shape)
    out = cake_backend.cake_kda_det(q, k, v, g.contiguous(), beta, inputs["cu_seqlens"])
    out.backward(inputs["dout"])
    return _results(out, leaves)


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(f"compute capability {capability} is not an exported target")
    return torch.device("cuda", index)


@pytest.mark.parametrize("row", tuple(ROWS))
def test_raw_gate_and_the_explicit_gate_meet_the_naive_tier(
    row: str, device, record_property
) -> None:
    """Row ``row``: the raw path (arm ``""``) and the explicit torch-gate path (arm ``explicit``), every output of
    each judged by ``judge_arms`` against the exact FP64 recurrence beside the naive BF16 tier over the judge's draws
    (the draws' ``dt_bias`` shifted by the row's ``dt_bias`` shift).  Rows C and F run only with ``ENV_RUN`` set."""
    if row in OPT_IN_ROWS and not RUN:
        pytest.skip(f"row {row} runs with {ENV_RUN}=1 (its size)")
    lengths, heads, shift = ROWS[row]

    def draw(j, s):
        inputs = draw_case(lengths, heads, "raw", False, s, device, dt_bias_mean=shift)
        return {"": arm_raw(inputs), "explicit": arm_explicit(inputs)}, inputs

    judge_arms(
        record_property,
        draw,
        seed=SEEDS[row],
        scale=cake_backend.SCALE,
        lower_bound=LOWER_BOUND,
    )
