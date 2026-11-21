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

The accuracy acceptance of issue #82 for both modules (``det`` and ``nondet``), through the autograd entry with each
forward sum of the dot product (``dot_sum="fp32"``, the default, and the opt-in ``"compensated"``; the backward is the
same for both), on operands as the issue's script draws them (``T = 16``, BF16 operands, FP32 weights rounded from
BF16).

* random draws (:data:`DRAWS`; ``x`` scaled 0.25, 1, 2, 4 per stream, ``key`` 4, 2, 1, 0.25, every fourth token
  masked): for each of ``out``, ``dx``, ``dkey``, ``dvalue``, ``dq_weight`` and ``dk_weight``, the median and the
  maximum relative L2 error against the FP64 reference are at most twice those of the FP32 evaluation of the issue's
  reference (PyTorch's FP32 autograd) on the same draws;
* near-cancelling draws (:data:`DRAWS`; ``key`` moved so that ``2e-6 <= |dot| <= 2e-5``): ``dx``, ``dkey``,
  ``dq_weight`` and ``dk_weight``, measured per stream, meet twice the error of the issue's FP32 evaluation of these
  rows (the FP32 formula with an accurately summed dot product), per stream likewise.

Every row is printed and recorded (``record_property``) with the kernel's and the FP32 evaluation's median and
maximum and the bounds.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_engram_gate_naive import (
    STREAMS,
    acceptance_rows,
    issue_fp32_near,
    issue_fp32_random,
    issue_fp64,
    make_inputs,
    rel_l2,
)

MODES = ("det", "nondet")
# The ``dot_sum`` choices, the default first.
DOT_SUMS = ("fp32", "compensated")
#: Draws of each kind (the issue asks for 25 or more).
DRAWS = 32
T = 16
RANDOM_OUTPUTS = ("out", "dx", "dkey", "dvalue", "dq", "dk")
NEAR_OUTPUTS = ("dx", "dkey", "dq", "dk")


def _api(mode: str):
    return importlib.import_module(f"cake_pplx.cake_engram_gate_{mode}_train")


def _device_or_skip(mode: str) -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake Engram gate ships for SM100 / SM103 / SM107")
    jit = importlib.import_module(f"cake_pplx._backends.cake_engram_gate_{mode}_train.cake_jit")
    if not (Path(jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return torch.device("cuda:0")


def _kernel(mode: str, ns, dot_sum: str) -> dict[str, torch.Tensor]:
    """Output and gradients of the module's autograd entry with ``dot_sum``."""
    api = _api(mode)
    entry = getattr(api, f"cake_engram_gate_{mode}")
    assert inspect.signature(inspect.unwrap(entry)).parameters["dot_sum"].default == DOT_SUMS[0]
    x = ns.x.detach().clone().requires_grad_(True)
    key = ns.key.detach().clone().requires_grad_(True)
    value = ns.value.detach().clone().requires_grad_(True)
    qw = ns.q_weight.detach().clone().requires_grad_(True)
    kw = ns.k_weight.detach().clone().requires_grad_(True)
    out = entry(x, key, value, qw, kw, ns.token_mask, dot_sum=dot_sum)
    out.backward(ns.dout)
    return {"out": out.detach(), "dx": x.grad, "dkey": key.grad, "dvalue": value.grad, "dq": qw.grad, "dk": kw.grad}


def _report(rows, label: str, record_property) -> None:
    for row in rows:
        line = (
            f"{label} {row['output']}: n={row['n']} ours med/max={row['ours'][0]:.3e}/{row['ours'][1]:.3e} "
            f"fp32 med/max={row['fp32'][0]:.3e}/{row['fp32'][1]:.3e} bound={row['bound'][0]:.3e}/{row['bound'][1]:.3e} "
            f"{'ok' if row['ok'] else 'FAIL'}"
        )
        print(line)
        record_property(f"{label}-{row['output']}", line)


def _per_stream(t: torch.Tensor, s: int) -> torch.Tensor:
    """Stream ``s`` of a ``[T, 4, H]`` gradient or a ``[4, H]`` weight gradient."""
    return t[:, s] if t.dim() == 3 else t[s]


@pytest.mark.parametrize("dot_sum", DOT_SUMS)
@pytest.mark.parametrize("mode", MODES)
def test_random_draws_meet_twice_the_fp32_evaluation(mode, dot_sum, record_property) -> None:
    device = _device_or_skip(mode)
    ours = {name: [] for name in RANDOM_OUTPUTS}
    fp32 = {name: [] for name in RANDOM_OUTPUTS}
    for draw in range(DRAWS):
        ns = make_inputs(T, seed=8200 + draw, device=device, weights="fp32", mask="some")
        want = issue_fp64(ns)
        got = _kernel(mode, ns, dot_sum)
        evaluation = issue_fp32_random(ns)
        for name in RANDOM_OUTPUTS:
            ours[name].append(rel_l2(got[name], want[name]))
            fp32[name].append(rel_l2(evaluation[name], want[name]))
    rows = acceptance_rows(ours, fp32)
    _report(rows, f"{mode}-{dot_sum}-random", record_property)
    assert all(row["ok"] for row in rows), [row["output"] for row in rows if not row["ok"]]


@pytest.mark.parametrize("dot_sum", DOT_SUMS)
@pytest.mark.parametrize("mode", MODES)
def test_near_cancelling_draws_meet_twice_the_fp32_evaluation_per_stream(mode, dot_sum, record_property) -> None:
    device = _device_or_skip(mode)
    ours = {name: [] for name in NEAR_OUTPUTS}
    fp32 = {name: [] for name in NEAR_OUTPUTS}
    for draw in range(DRAWS):
        ns = make_inputs(T, seed=8300 + draw, device=device, weights="fp32", near_cancel=True)
        want = issue_fp64(ns)
        got = _kernel(mode, ns, dot_sum)
        evaluation = issue_fp32_near(ns)
        for name in NEAR_OUTPUTS:
            for s in range(STREAMS):
                ours[name].append(rel_l2(_per_stream(got[name], s), _per_stream(want[name], s)))
                fp32[name].append(rel_l2(_per_stream(evaluation[name], s), _per_stream(want[name], s)))
    rows = acceptance_rows(ours, fp32)
    _report(rows, f"{mode}-{dot_sum}-near", record_property)
    assert all(row["ok"] for row in rows), [row["output"] for row in rows if not row["ok"]]
