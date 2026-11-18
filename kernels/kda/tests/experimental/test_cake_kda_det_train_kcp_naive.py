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

Accuracy of the deterministic context-parallel (KCP) packed KDA chunked training package
(``cake_pplx.cake_kda_det_train_kcp``) against the naive BF16 tier (``tests/cake_naive.py``).

Each case splits one packed batch over ``W`` ranks, one process and one GPU per rank
(``test_helpers/cake_kda_naive_kcp.py``), runs one training step of every shard through the autograd entry
``cake_kda_det_kcp`` (in the raw gate mode the parameter gradients summed over the ranks by
``cake_kda_det_train_kcp_reduce_gate_param_grads``) and judges ``out`` and every gradient of the whole batch by its
error against the FP64 token recurrence of the unsplit batch, beside the error of PyTorch running the same ops in
BF16 (``test_helpers/cake_kda_naive.py``), over 32 random draws.  A case needs ``W`` visible GPUs; it is skipped
otherwise.

Cases: four documents with odd lengths over two ranks in the contiguous layout (the first document crosses the
rank boundary), with the activated FP32 gate and with the raw gate and the in-kernel Q/K L2 norm; one document over
four ranks in the zigzag layout.
"""

from __future__ import annotations

import pytest
import torch

from cake_pplx.cake_kda_det_train import DEFAULT_SCALE
from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers.cake_kda_det_train_reference import LOWER_BOUND
from tests.experimental.test_helpers.cake_kda_naive_kcp import judge_kcp_case

cake_jit = cake_package("cake.cake_kda_det_train", "cake_jit")

MODE = "det"
SEG = (300, 1, 130, 89)
# name: (lengths, H, gate, in-kernel Q/K L2 norm, layout, ranks)
CASES = {
    "w2_contiguous": (SEG, 2, "g", False, "contiguous", 2),
    "w2_contiguous_raw_l2n": (SEG, 2, "raw", True, "contiguous", 2),
    "w4_zigzag": ((520,), 2, "g", False, "zigzag", 4),
}
SEEDS = {name: 4300 + 10 * i for i, name in enumerate(CASES)}


def _skip_unless_runnable(W: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    capability = tuple(torch.cuda.get_device_capability())
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {cake_jit.ARCHES}"
        )


@pytest.mark.parametrize("name", list(CASES))
def test_cake_kda_det_kcp_meets_the_naive_tier(name, record_property) -> None:
    lengths, H, gate, l2n, layout, W = CASES[name]
    _skip_unless_runnable(W)
    judge_kcp_case(
        record_property,
        MODE,
        (lengths, H, gate, l2n, layout, DEFAULT_SCALE, LOWER_BOUND),
        SEEDS[name],
        W,
    )
