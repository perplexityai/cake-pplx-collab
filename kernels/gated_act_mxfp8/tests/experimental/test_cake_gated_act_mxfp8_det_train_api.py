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

Tests of the cake_pplx public surface of the Cake fused SwiGLU + MXFP8
quantization training kernels (SM100 / SM103 / SM107): the
``cake_pplx.cake_gated_act_mxfp8_det_train`` entry points, the facade bound on
the ``cake_pplx`` namespace package (which re-exports no entry point) and the
custom-op wrappers.
The entry points are pass-throughs to
``cake_pplx._backends.cake_gated_act_mxfp8_det_train``;
this module checks that identity (same objects at the top level, the backend's
exact signatures, same bytes as the backend) and leaves the numerical contract
to ``test_cake_gated_act_mxfp8_det_train.py``, which imports only the backend package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import cake_pplx
import cake_pplx.cake_gated_act_mxfp8_det_train
import cake_pplx._backends.cake_gated_act_mxfp8_det_train as backend
from cake_pplx.cake_gated_act_mxfp8_det_train import (
    cake_gated_act_mxfp8_det_train_register_custom_ops,
    cake_gated_act_mxfp8_det_train_forward,
    cake_gated_act_mxfp8_det_train_backward,
)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
MODES = ((True, False), (False, True), (True, True))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(0) in SUPPORTED_CAPABILITIES
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_gated_act_mxfp8_det_train needs an SM100 / SM103 / SM107 device",
)


def _inputs(M: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1365)
    gated_input = torch.randn(
        (M, 2 * K), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(
        (M, K), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    return gated_input, grad_output


def _same(a, b) -> None:
    assert len(a) == len(b) == 4
    for x, y in zip(a, b, strict=True):
        assert x.dtype == y.dtype and x.shape == y.shape and x.stride() == y.stride()
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


def test_entry_points_are_exported() -> None:
    # The facade is the ``cake_pplx`` submodule; the namespace package itself
    # re-exports no entry point (it has no ``__init__.py``).
    facade = cake_pplx.cake_gated_act_mxfp8_det_train
    assert (
        facade.cake_gated_act_mxfp8_det_train_forward
        is cake_gated_act_mxfp8_det_train_forward
    )
    assert (
        facade.cake_gated_act_mxfp8_det_train_backward
        is cake_gated_act_mxfp8_det_train_backward
    )
    assert not hasattr(cake_pplx, "cake_gated_act_mxfp8_det_train_forward")
    assert not hasattr(cake_pplx, "cake_gated_act_mxfp8_det_train_backward")


def test_entry_point_signatures_mirror_the_backend() -> None:
    for name in (
        "cake_gated_act_mxfp8_det_train_forward",
        "cake_gated_act_mxfp8_det_train_backward",
    ):
        # The backend postpones its annotations; compare the evaluated forms.
        facade = inspect.signature(
            getattr(cake_pplx.cake_gated_act_mxfp8_det_train, name), eval_str=True
        )
        backend_signature = inspect.signature(getattr(backend, name), eval_str=True)
        assert facade == backend_signature, (name, facade, backend_signature)
        parameters = facade.parameters
        assert parameters["rowwise"].default is True
        assert parameters["colwise"].default is False
        assert parameters["rowwise"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("rowwise,colwise", MODES)
def test_entry_points_match_the_backend(rowwise, colwise) -> None:
    gated_input, grad_output = _inputs(256, 384)
    _same(
        cake_gated_act_mxfp8_det_train_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        ),
        backend.cake_gated_act_mxfp8_det_train_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        ),
    )
    _same(
        cake_gated_act_mxfp8_det_train_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        ),
        backend.cake_gated_act_mxfp8_det_train_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        ),
    )


@pytest.mark.parametrize("rowwise,colwise", MODES)
def test_custom_ops_match_the_entry_points(rowwise, colwise) -> None:
    gated_input, grad_output = _inputs(128, 256)
    module = cake_gated_act_mxfp8_det_train_register_custom_ops()
    _same(
        module.forward(gated_input, rowwise, colwise),
        cake_gated_act_mxfp8_det_train_forward(
            gated_input, rowwise=rowwise, colwise=colwise
        ),
    )
    _same(
        module.backward(gated_input, grad_output, rowwise, colwise),
        cake_gated_act_mxfp8_det_train_backward(
            gated_input, grad_output, rowwise=rowwise, colwise=colwise
        ),
    )
