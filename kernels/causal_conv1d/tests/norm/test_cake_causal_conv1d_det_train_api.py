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

Tests of the public surface of the Cake packed width-4 depthwise
causal convolution + SiLU training kernels (SM100 / SM103 / SM107): the
``cake_pplx.cake_causal_conv1d_det_train`` facade.  The facade's three entry points are
pass-throughs to ``cake_pplx._backends.cake_causal_conv1d_det_train``; this module
checks that identity (same objects on the facade, the backend's exact
signatures, same bits as the backend), that the opt-in
``dweight_sum="fp32_pair"`` reaches the pair program through the facade, and
that it raises ``ValueError`` before any launch under the ``"bf16"`` policy.  It
leaves the numerical contract to ``test_cake_causal_conv1d_det_train.py`` and
``test_cake_causal_conv1d_det_train_pair.py``, which import only the backend
package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import cake_pplx.cake_causal_conv1d_det_train
import cake_pplx._backends.cake_causal_conv1d_det_train as backend
from cake_pplx.cake_causal_conv1d_det_train import (
    CakeCausalConv1dDetFunction,
    cake_causal_conv1d_det,
    cake_causal_conv1d_det_train_backward,
    cake_causal_conv1d_det_train_backward_workspace_bytes,
    cake_causal_conv1d_det_train_forward,
    cake_causal_conv1d_det_train_forward_workspace,
    cake_causal_conv1d_det_train_forward_workspace_bytes,
    is_cake_causal_conv1d_det_train_supported,
)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
SEED = 2033
PAIR = "fp32_pair"
loader = backend.cake_jit


def _device() -> torch.device:
    return torch.device("cuda:0")


def _inputs(S: int, C: int, *, bias: bool = False):
    """``(x, w, cu, dy, b)``: BF16 ``x`` / ``dy`` of N(0, 1), FP32 taps of N(0, 0.25),
    four documents and, with ``bias``, an FP32 bias of N(0, 0.25) (``None`` without)."""

    generator = torch.Generator(device=_device()).manual_seed(SEED)
    x = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    dy = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)
    w = 0.5 * torch.randn(
        C, 4, dtype=torch.float32, device=_device(), generator=generator
    )
    b = None
    if bias:
        b = 0.5 * torch.randn(
            C, dtype=torch.float32, device=_device(), generator=generator
        )
    cu = torch.tensor([0, 369, 659, 868, S], dtype=torch.int32, device=_device())
    return x, w, cu, dy, b


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_causal_conv1d_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_causal_conv1d_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


def test_function_signature_is_exported() -> None:
    assert cake_pplx.cake_causal_conv1d_det_train.cake_causal_conv1d_det is cake_causal_conv1d_det
    assert (
        cake_pplx.cake_causal_conv1d_det_train.cake_causal_conv1d_det_train_forward
        is cake_causal_conv1d_det_train_forward
    )
    assert (
        cake_pplx.cake_causal_conv1d_det_train.cake_causal_conv1d_det_train_backward
        is cake_causal_conv1d_det_train_backward
    )
    assert (
        cake_pplx.cake_causal_conv1d_det_train.cake_causal_conv1d_det_train_backward_workspace_bytes
        is cake_causal_conv1d_det_train_backward_workspace_bytes
    )
    assert (
        cake_pplx.cake_causal_conv1d_det_train.cake_causal_conv1d_det_train_forward_workspace_bytes
        is cake_causal_conv1d_det_train_forward_workspace_bytes
        is backend.cake_causal_conv1d_det_train_forward_workspace_bytes
    )
    assert (
        cake_causal_conv1d_det_train_forward_workspace
        is backend.cake_causal_conv1d_det_train_forward_workspace
    )
    assert cake_pplx.cake_causal_conv1d_det_train.CakeCausalConv1dDetFunction is CakeCausalConv1dDetFunction


def test_facade_signatures_mirror_the_backend() -> None:
    """Every keyword of the backend's entry points (``bias``, ``silu_grad_preactivation``, ``dbias``, ...) is
    spelled identically on the ``@cake_api`` facade, which public-API tooling reads."""

    for name in (
        "cake_causal_conv1d_det_train_forward",
        "cake_causal_conv1d_det_train_backward",
        "cake_causal_conv1d_det",
    ):
        facade = inspect.signature(
            getattr(cake_pplx.cake_causal_conv1d_det_train, name)
        )
        backend_signature = inspect.signature(getattr(backend, name))
        assert facade == backend_signature, (name, facade, backend_signature)
    backward = inspect.signature(cake_causal_conv1d_det_train_backward).parameters
    assert backward["bias"].default is None
    assert backward["silu_grad_preactivation"].default == "bf16"
    assert backward["dbias"].default is None
    assert backward["bias"].kind is inspect.Parameter.KEYWORD_ONLY
    autograd = inspect.signature(cake_causal_conv1d_det).parameters
    assert autograd["bias"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert autograd["silu_grad_preactivation"].kind is inspect.Parameter.KEYWORD_ONLY
    assert backend.SILU_GRAD_PREACTIVATIONS == ("bf16", "fp32")
    # The opt-in backward grid multiple: keyword-only, default 1 (the default grid), on the backward, the
    # autograd entry point and both backward workspace helpers.
    # The allowed multiples are the registry's rendered ``GRID_MULT_CHOICES``.
    assert backend.BWD_GRID_MULTS == (1, 2, 3, 4)
    assert backend.BWD_GRID_MULTS == backend.cake_jit.GRID_MULT_CHOICES
    for function in (
        cake_causal_conv1d_det_train_backward,
        cake_causal_conv1d_det,
        cake_causal_conv1d_det_train_backward_workspace_bytes,
        cake_pplx.cake_causal_conv1d_det_train.cake_causal_conv1d_det_train_backward_workspace,
    ):
        parameter = inspect.signature(function).parameters["bwd_grid_mult"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function
        assert parameter.default == 1, function


def test_facade_entry_points_match_the_backend() -> None:
    """The facade forwards every argument: its forward, backward and autograd
    entry points return the same bits as the backend package's."""

    S, C = 1000, 512
    x, w, cu, dy, _ = _inputs(S, C)
    y = cake_causal_conv1d_det_train_forward(x, w, cu)
    assert torch.equal(y, backend.cake_causal_conv1d_det_train_forward(x, w, cu))
    workspace = cake_causal_conv1d_det_train_forward_workspace(S, C, _device())
    assert torch.equal(
        y, cake_causal_conv1d_det_train_forward(x, w, cu, workspace=workspace)
    )
    assert int(torch.count_nonzero(workspace)) == 0
    dx, dw = cake_causal_conv1d_det_train_backward(x, w, cu, dy)
    dx_backend, dw_backend = backend.cake_causal_conv1d_det_train_backward(x, w, cu, dy)
    assert torch.equal(dx, dx_backend) and torch.equal(dw, dw_backend)
    dx3, dw3 = cake_causal_conv1d_det_train_backward(x, w, cu, dy, bwd_grid_mult=3)
    dx3_backend, dw3_backend = backend.cake_causal_conv1d_det_train_backward(
        x, w, cu, dy, bwd_grid_mult=3
    )
    assert torch.equal(dx3, dx3_backend) and torch.equal(dw3, dw3_backend)
    assert torch.equal(dx3, dx)
    assert torch.equal(
        cake_causal_conv1d_det(x, w, cu), backend.cake_causal_conv1d_det(x, w, cu)
    )


def test_facade_pair_sum_launches_the_pair_program(monkeypatch) -> None:
    """``dweight_sum="fp32_pair"`` with ``silu_grad_preactivation="fp32"`` passes
    through the facade: its backward and autograd entry points launch the pair
    route's program for ``C`` then its reduce program, return finite outputs of
    the right shapes and dtypes, and give the same bits as the backend."""

    S, C = 1000, 512
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(_device()))
    launched: list[str] = []
    run = backend.run_modules

    def record(launches, values, *, device_index):
        launched.extend(name for name, _ in launches)
        return run(launches, values, device_index=device_index)

    monkeypatch.setattr(backend, "run_modules", record)
    for bias, kind in ((False, "bwd_fp32g_pair"), (True, "bwd_bias_fp32g_pair")):
        x, w, cu, dy, b = _inputs(S, C, bias=bias)
        main = loader.selected_modules(arch, kind, C=C, S=S)[0]
        pair_launches = [main, loader.MODULES[main]["reduce"]]
        launched.clear()
        grads = cake_causal_conv1d_det_train_backward(
            x, w, cu, dy, bias=b, silu_grad_preactivation="fp32", dweight_sum=PAIR
        )
        assert launched == pair_launches, (kind, launched)
        expected = [(torch.bfloat16, tuple(x.shape)), (w.dtype, (C, 4))]
        if b is not None:
            expected.append((b.dtype, (C,)))
        assert len(grads) == len(expected), kind
        for grad, (dtype, shape) in zip(grads, expected):
            assert grad.dtype == dtype and tuple(grad.shape) == shape, kind
            assert torch.isfinite(grad.float()).all(), kind
        backend_grads = backend.cake_causal_conv1d_det_train_backward(
            x, w, cu, dy, bias=b, silu_grad_preactivation="fp32", dweight_sum=PAIR
        )
        for got, expected_bits in zip(grads, backend_grads):
            assert torch.equal(got, expected_bits), kind
        xg = x.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        bg = None if b is None else b.clone().requires_grad_(True)
        y = cake_causal_conv1d_det(
            xg, wg, cu, bg, silu_grad_preactivation="fp32", dweight_sum=PAIR
        )
        assert y.dtype == torch.bfloat16 and tuple(y.shape) == tuple(x.shape), kind
        assert torch.isfinite(y.detach().float()).all(), kind
        launched.clear()
        y.backward(dy)
        assert launched == pair_launches, (kind, launched)
        assert torch.equal(xg.grad, grads[0]) and torch.equal(wg.grad, grads[1]), kind
        if bg is not None:
            assert torch.equal(bg.grad, grads[2]), kind


def test_facade_pair_sum_needs_the_fp32_policy(monkeypatch) -> None:
    """``dweight_sum="fp32_pair"`` under the ``"bf16"`` policy (passed or by
    default) raises ``ValueError`` on the facade's backward and autograd entry
    points, with and without a bias, before any launch."""

    S, C = 1000, 512
    launches = []
    monkeypatch.setattr(backend, "run_modules", lambda *a, **k: launches.append(a))
    monkeypatch.setattr(backend, "run_module", lambda *a, **k: launches.append(a))
    for bias in (False, True):
        x, w, cu, dy, b = _inputs(S, C, bias=bias)
        for policy in ({"silu_grad_preactivation": "bf16"}, {}):
            with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
                cake_causal_conv1d_det_train_backward(
                    x, w, cu, dy, bias=b, dweight_sum=PAIR, **policy
                )
            with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
                cake_causal_conv1d_det(x, w, cu, b, dweight_sum=PAIR, **policy)
    assert launches == []
