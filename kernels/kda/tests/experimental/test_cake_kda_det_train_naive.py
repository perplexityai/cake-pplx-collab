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

Accuracy of the deterministic packed KDA chunked training package (``cake_pplx.cake_kda_det_train``) against the
naive BF16 tier (``tests/cake_naive.py``).

Each case runs one training step through the autograd entry ``cake_kda_det`` and judges every floating output
(``out`` and the gradients of ``q``, ``k``, ``v``, ``g``, ``beta`` and, in the raw gate mode, of ``A_log`` and
``dt_bias``) by its error against the FP64 token recurrence (``test_helpers/cake_kda_det_train_reference.py`` and
``..._reference_l2n.py``), beside the error of PyTorch running the same ops in BF16 on the same inputs
(``test_helpers/cake_kda_naive.py``), over 32 random draws.  On the first draw the step runs twice and every output
must repeat bit for bit.

Cases: four documents with odd lengths (a one-token document, partial last chunks, a document of several chunks)
with 2 heads, which take the segmented state recurrence, in the four forms of the gate and the Q/K norm (the
activated FP32 gate, the raw gate activated in the kernels, the in-kernel Q/K L2 norm, both), the activated-gate
form also under ``torch.utils.checkpoint`` (outputs ``ckpt.*``); and two documents with 12 heads, which take the
serial walk.  Above 64 heads (issue #264): the raw gate with the in-kernel Q/K L2 norm and ``dt_bias`` ``[H, 128]`` at
96 heads on the 4,096-token pack the raw gate is judged on at 64 heads, also under ``torch.utils.checkpoint``; the
activated gate at 96 heads; the raw gate with the L2 norm at 128 heads, the top of the range.
"""

from __future__ import annotations

import pytest
import torch

from cake_pplx.cake_kda_det_train import DEFAULT_SCALE, cake_kda_det
from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers.cake_kda_det_train_reference import LOWER_BOUND
from tests.experimental.test_helpers.cake_kda_naive import judge_single_rank

cake_jit = cake_package("cake.cake_kda_det_train", "cake_jit")

SEG = (300, 1, 130, 89)
# name: (lengths, H, gate, in-kernel Q/K L2 norm, also under torch.utils.checkpoint)
CASES = {
    "seg": (SEG, 2, "g", False, True),
    "seg_raw": (SEG, 2, "raw", False, False),
    "seg_l2n": (SEG, 2, "g", True, False),
    "seg_raw_l2n": (SEG, 2, "raw", True, False),
    "serial": ((200, 77), 12, "g", False, False),
    "h96_raw_l2n": ((1000, 1100, 900, 1096), 96, "raw", True, True),
    "h96": ((63, 1, 65), 96, "g", False, False),
    "h128_raw_l2n": ((63, 1, 65), 128, "raw", True, False),
}
# the cases that draw ``dt_bias`` as ``[H, 128]`` (the issue's form)
DT_BIAS_2D = {"h96_raw_l2n", "h128_raw_l2n"}
SEEDS = {name: 4100 + 10 * i for i, name in enumerate(CASES)}


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {cake_jit.ARCHES}"
        )
    return torch.device("cuda", index)


@pytest.mark.parametrize("name", list(CASES))
def test_cake_kda_det_meets_the_naive_tier(name, device, record_property) -> None:
    judge_single_rank(
        record_property,
        cake_kda_det,
        CASES[name],
        SEEDS[name],
        device,
        DEFAULT_SCALE,
        LOWER_BOUND,
        det=True,
        dt_bias_2d=name in DT_BIAS_2D,
    )
