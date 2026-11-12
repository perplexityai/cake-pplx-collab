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
"""

import pytest
import torch

from cake_pplx.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_backward,
    cake_gated_act_mxfp8_det_train_forward,
)


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in (
        (10, 0),
        (10, 3),
        (10, 7),
    )


def _make_inputs(m: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(20260820)
    gated_input = torch.randn(
        (m, 2 * k), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(
        (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    return gated_input, grad_output


def _run(
    direction: str,
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    rowwise: bool,
    colwise: bool,
):
    if direction == "forward":
        return cake_gated_act_mxfp8_det_train_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        )
    return cake_gated_act_mxfp8_det_train_backward(
        gated_input, grad_output, rowwise=rowwise, colwise=colwise
    )


@pytest.mark.parametrize("direction", ["forward", "backward"])
@torch.inference_mode()
def test_gated_act_mxfp8_mode_consistency(direction):
    if not _supported():
        pytest.skip("fused gated MXFP8 quantization requires SM100, SM103 or SM107")
    gated_input, grad_output = _make_inputs(256, 512)
    row = _run(direction, gated_input, grad_output, True, False)
    col = _run(direction, gated_input, grad_output, False, True)
    both = _run(direction, gated_input, grad_output, True, True)
    for actual, expected in (
        (both[0], row[0]),
        (both[2], row[2]),
        (both[1], col[1]),
        (both[3], col[3]),
    ):
        torch.testing.assert_close(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )


