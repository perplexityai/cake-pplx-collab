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

Tests of the public surface of the Cake route-weight row scaling training kernels (SM100 / SM103): the
``cake_pplx.cake_route_weight_scale_det_train`` facade.  Its entry points are pass-throughs to
``cake_pplx._backends.cake_route_weight_scale_det_train``: the same signatures (``out``, ``dx`` and ``dw`` keyword-only),
the same errors and the same bits; its autograd class and constants are the backend's objects, resolved on first
access, and importing the facade does not import the backend.  The mode rules of the deterministic package: no entry
point takes a ``deterministic`` keyword, and ``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``)
neither raises nor warns in any call, and changes no bit.  The facade reads no environment variable.  The numerical
contract is ``test_cake_route_weight_scale_det_train.py``'s.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import cake_pplx._backends.cake_route_weight_scale_det_train as backend
import cake_pplx.cake_route_weight_scale_det_train as facade
from cake_pplx.cake_route_weight_scale_det_train import (
    cake_route_weight_scale_det,
    cake_route_weight_scale_det_train_backward,
    cake_route_weight_scale_det_train_forward,
    is_cake_route_weight_scale_det_train_supported,
)

requires_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_route_weight_scale_det_train_supported(torch.device("cuda"))
    or not (Path(backend.__file__).resolve().parent / "build.json").is_file(),
    reason="Cake route-weight row scaling needs an SM100 / SM103 GPU and an exported package",
)
ENTRY_NAMES = (
    "cake_route_weight_scale_det",
    "cake_route_weight_scale_det_train_backward",
    "cake_route_weight_scale_det_train_forward",
    "is_cake_route_weight_scale_det_train_supported",
)


def _inputs(rows: int, width: int, *, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, width, device="cuda", generator=g).bfloat16()
    dy = torch.randn(rows, width, device="cuda", generator=g).bfloat16()
    w = torch.randn(rows, device="cuda", generator=g)
    return x, dy, w


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[a.dtype]
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(view), b.view(view))


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_facade_signatures_are_the_backends(name: str) -> None:
    assert inspect.signature(getattr(facade, name)) == inspect.signature(getattr(backend, name)), name


@pytest.mark.cpu_only
def test_outputs_are_keyword_only() -> None:
    forward = inspect.signature(cake_route_weight_scale_det_train_forward).parameters
    assert list(forward) == ["x", "w", "out"] and forward["out"].kind is inspect.Parameter.KEYWORD_ONLY
    backward = inspect.signature(cake_route_weight_scale_det_train_backward).parameters
    assert list(backward) == ["x", "dy", "w", "dx", "dw"]
    assert backward["dx"].kind is backward["dw"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(cake_route_weight_scale_det).parameters) == ["x", "w"]


@pytest.mark.cpu_only
def test_lazy_objects_are_the_backends() -> None:
    for name in ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "WIDTHS", "CakeRouteWeightScaleDetFunction"):
        assert getattr(facade, name) is getattr(backend, name), name
    assert tuple(facade.WIDTHS) == (4096, 3584)
    assert tuple(facade.ARCHES) == ("sm_100a", "sm_103a")
    with pytest.raises(AttributeError):
        facade.not_a_name  # noqa: B018


@pytest.mark.cpu_only
def test_importing_the_facade_does_not_import_the_backend() -> None:
    """In a fresh interpreter: the facade imports without its backend package; the first entry call imports it."""
    code = (
        "import json, sys\n"
        "import cake_pplx.cake_route_weight_scale_det_train as facade\n"
        "name = 'cake_pplx._backends.cake_route_weight_scale_det_train'\n"
        "before = name in sys.modules\n"
        "facade.is_cake_route_weight_scale_det_train_supported('cpu')\n"
        "print(json.dumps(dict(before=before, after=name in sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == dict(before=False, after=True)


@pytest.mark.cpu_only
def test_supported_query_without_a_device() -> None:
    assert is_cake_route_weight_scale_det_train_supported(torch.device("cpu")) is False


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", [True, False])
def test_a_mode_keyword_is_a_type_error(mode) -> None:
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    w = torch.zeros(4)
    for call in (
        lambda: cake_route_weight_scale_det(x, w, deterministic=mode),
        lambda: cake_route_weight_scale_det_train_forward(x, w, deterministic=mode),
        lambda: cake_route_weight_scale_det_train_backward(x, x, w, deterministic=mode),
    ):
        with pytest.raises(TypeError, match="deterministic"):
            call()


@pytest.mark.cpu_only
def test_facade_refuses_what_the_backend_refuses() -> None:
    """The facade passes its arguments through: a CPU call, a wrong width and a strided weight raise the backend's
    errors."""
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    for call, message in (
        (lambda: cake_route_weight_scale_det_train_forward(x, torch.zeros(4)), "requires CUDA tensors"),
        (lambda: cake_route_weight_scale_det_train_forward(x[:, :2048], torch.zeros(4)), "H in"),
        (lambda: cake_route_weight_scale_det_train_backward(x, x, torch.zeros(8)[::2]), "contiguous"),
        (lambda: cake_route_weight_scale_det(x, torch.zeros(4)), "requires CUDA tensors"),
    ):
        with pytest.raises(ValueError, match=message):
            call()


@pytest.mark.cpu_only
def test_the_facade_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(facade.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("environ", "getenv"), node.lineno


@requires_device
@pytest.mark.parametrize("width", [4096, 3584])
def test_facade_matches_backend_bits(width: int) -> None:
    x, dy, w = _inputs(1027, width, seed=width)
    y_f = cake_route_weight_scale_det_train_forward(x, w)
    y_b = backend.cake_route_weight_scale_det_train_forward(x, w)
    dx_f, dw_f = cake_route_weight_scale_det_train_backward(x, dy, w)
    dx_b, dw_b = backend.cake_route_weight_scale_det_train_backward(x, dy, w)
    assert _same_bits(y_f, y_b) and _same_bits(dx_f, dx_b) and _same_bits(dw_f, dw_b)
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = cake_route_weight_scale_det(xa, wa)
    y.backward(dy)
    assert _same_bits(y.detach(), y_b) and _same_bits(xa.grad, dx_b) and _same_bits(wa.grad, dw_b)


@requires_device
@pytest.mark.parametrize("warn_only", [False, True])
def test_deterministic_algorithms_neither_raise_nor_warn(warn_only: bool) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (with PyTorch's NaN fill of new tensors) every call runs
    without an error or a warning and gives the bits of a call without it."""
    x, dy, w = _inputs(515, 4096, seed=3)
    y0 = cake_route_weight_scale_det_train_forward(x, w)
    dx0, dw0 = cake_route_weight_scale_det_train_backward(x, dy, w)
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y = cake_route_weight_scale_det_train_forward(x, w)
            dx, dw = cake_route_weight_scale_det_train_backward(x, dy, w)
            xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
            cake_route_weight_scale_det(xa, wa).backward(dy)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    assert _same_bits(y, y0) and _same_bits(dx, dx0) and _same_bits(dw, dw0)
    assert _same_bits(xa.grad, dx0) and _same_bits(wa.grad, dw0)
