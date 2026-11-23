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

Tests of the cake_pplx public surface of the Cake activation
quantize-dequantize forward (SM100 / SM103 / SM107): the
``cake_pplx.cake_act_qdq_det_train`` entry points and the facade bound on the
``cake_pplx`` namespace package (which re-exports no entry point).  The entry
points are pass-throughs to ``cake_pplx._backends.cake_act_qdq_det_train``;
this module checks that identity (the facade's own functions, the backend's
exact signatures, the same bytes as the backend) and leaves the numerical
contract to ``test_cake_act_qdq_det_train.py``, which imports only the backend
package.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import cake_pplx
import cake_pplx.cake_act_qdq_det_train
import cake_pplx._backends.cake_act_qdq_det_train as backend
from cake_pplx.cake_act_qdq_det_train import (
    cake_act_qdq_det_train_domain_empty,
    cake_act_qdq_det_train_domain_plan,
    cake_act_qdq_det_train_forward,
    cake_act_qdq_det_train_forward_workspace_bytes,
    is_cake_act_qdq_det_train_supported,
)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
FORMATS = ("fp4_e4m3_g16", "mxfp4_e8m0_g32", "mxfp8_e8m0_g32")
ENTRIES = (
    "cake_act_qdq_det_train_domain_empty",
    "cake_act_qdq_det_train_domain_plan",
    "cake_act_qdq_det_train_forward",
    "cake_act_qdq_det_train_forward_workspace_bytes",
    "is_cake_act_qdq_det_train_supported",
)


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) in SUPPORTED_CAPABILITIES


gpu = pytest.mark.skipif(
    not _supported(),
    reason="cake_act_qdq_det_train needs an SM100 / SM103 / SM107 device",
)


def _input(shape: tuple[int, ...]) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(83)
    return torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)


def _same(a: torch.Tensor, b: torch.Tensor) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.cpu_only
def test_entry_points_are_exported() -> None:
    # The facade is the ``cake_pplx`` submodule; the namespace package itself
    # re-exports no entry point (it has no ``__init__.py``).
    facade = cake_pplx.cake_act_qdq_det_train
    assert facade.cake_act_qdq_det_train_forward is cake_act_qdq_det_train_forward
    assert facade.cake_act_qdq_det_train_forward_workspace_bytes is cake_act_qdq_det_train_forward_workspace_bytes
    assert facade.is_cake_act_qdq_det_train_supported is is_cake_act_qdq_det_train_supported
    assert facade.cake_act_qdq_det_train_domain_empty is cake_act_qdq_det_train_domain_empty
    assert facade.cake_act_qdq_det_train_domain_plan is cake_act_qdq_det_train_domain_plan
    for name in ENTRIES:
        assert not hasattr(cake_pplx, name), name


@pytest.mark.cpu_only
def test_entry_point_signatures_mirror_the_backend() -> None:
    for name in ENTRIES:
        # Both modules postpone their annotations; compare the evaluated forms.
        facade = inspect.signature(getattr(cake_pplx.cake_act_qdq_det_train, name), eval_str=True)
        backend_signature = inspect.signature(getattr(backend, name), eval_str=True)
        assert facade == backend_signature, (name, facade, backend_signature)
    parameters = inspect.signature(cake_act_qdq_det_train_forward).parameters
    assert list(parameters) == ["x", "format", "out", "scales", "return_scales", "locality"]
    assert parameters["x"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for keyword in ("format", "out", "scales", "return_scales", "locality"):
        assert parameters[keyword].kind is inspect.Parameter.KEYWORD_ONLY, keyword
    assert parameters["format"].default is inspect.Parameter.empty
    assert parameters["out"].default is None
    assert parameters["scales"].default is None
    assert parameters["return_scales"].default is False
    assert parameters["locality"].default == "off"
    # the plan takes the forward's arguments but ``locality``
    plan = inspect.signature(cake_act_qdq_det_train_domain_plan).parameters
    assert list(plan) == ["x", "format", "out", "scales", "return_scales"]
    for keyword in ("format", "out", "scales", "return_scales"):
        assert plan[keyword].kind is inspect.Parameter.KEYWORD_ONLY, keyword
        assert plan[keyword].default == parameters[keyword].default, keyword
    empty = inspect.signature(cake_act_qdq_det_train_domain_empty).parameters
    assert list(empty) == ["shape", "dtype", "device"]
    assert empty["shape"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert empty["dtype"].kind is inspect.Parameter.KEYWORD_ONLY and empty["dtype"].default is torch.bfloat16
    assert empty["device"].kind is inspect.Parameter.KEYWORD_ONLY and empty["device"].default is None


@pytest.mark.cpu_only
def test_locality_is_checked_first() -> None:
    """``locality`` is checked before every other argument, so a bad value fails even on a CPU tensor or an unknown
    format; a good value reaches the other checks."""
    x = torch.zeros(4, 32, dtype=torch.bfloat16)
    for bad in (None, 1, True, ("domain",)):
        with pytest.raises(TypeError, match="locality"):
            cake_act_qdq_det_train_forward(x, format="nvfp4", locality=bad)
    for bad in ("", "on", "Domain", "domain "):
        with pytest.raises(ValueError, match="locality"):
            cake_act_qdq_det_train_forward(x, format="nvfp4", locality=bad)
    for good in ("off", "domain"):
        with pytest.raises(ValueError, match="format"):
            cake_act_qdq_det_train_forward(x, format="nvfp4", locality=good)
        with pytest.raises(ValueError, match="CUDA tensor"):
            cake_act_qdq_det_train_forward(x, format="mxfp8_e8m0_g32", locality=good)


@pytest.mark.cpu_only
def test_plan_checks_the_arguments_as_the_forward() -> None:
    x = torch.zeros(4, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="format"):
        cake_act_qdq_det_train_domain_plan(x, format="nvfp4")
    with pytest.raises(ValueError, match="CUDA tensor"):
        cake_act_qdq_det_train_domain_plan(x, format="mxfp8_e8m0_g32")


@pytest.mark.cpu_only
def test_domain_empty_refuses_a_cpu_device() -> None:
    with pytest.raises(ValueError, match="CUDA device"):
        cake_act_qdq_det_train_domain_empty((4, 32), device="cpu")


@pytest.mark.cpu_only
def test_domain_empty_checks_dtype_and_shape_before_the_device() -> None:
    with pytest.raises(TypeError, match="dtype"):
        cake_act_qdq_det_train_domain_empty((4, 32), dtype="bfloat16")
    with pytest.raises(ValueError, match="negative"):
        cake_act_qdq_det_train_domain_empty((-1, 32))


@pytest.mark.cpu_only
def test_constants_are_the_backend_objects() -> None:
    facade = cake_pplx.cake_act_qdq_det_train
    for name in ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "FORMATS", "GROUP_SIZES"):
        assert getattr(facade, name) is getattr(backend, name), name
    assert tuple(facade.FORMATS) == FORMATS
    assert dict(facade.GROUP_SIZES) == {"fp4_e4m3_g16": 16, "mxfp4_e8m0_g32": 32, "mxfp8_e8m0_g32": 32}
    assert tuple(facade.SUPPORTED_COMPUTE_CAPABILITIES) == SUPPORTED_CAPABILITIES


@pytest.mark.cpu_only
def test_workspace_query_matches_the_backend() -> None:
    for fmt in FORMATS:
        for shape in ((16384, 512), (8192, 32, 128), (0, 128)):
            assert cake_act_qdq_det_train_forward_workspace_bytes(shape, format=fmt) == 0
            assert backend.cake_act_qdq_det_train_forward_workspace_bytes(shape, format=fmt) == 0


@pytest.mark.cpu_only
def test_support_query_off_device() -> None:
    assert is_cake_act_qdq_det_train_supported(torch.device("cpu")) is False


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_entry_points_match_the_backend(fmt: str) -> None:
    assert is_cake_act_qdq_det_train_supported(torch.device("cuda", 0)) is True
    x = _input((129, 32, 128))
    _same(
        cake_act_qdq_det_train_forward(x, format=fmt),
        backend.cake_act_qdq_det_train_forward(x, format=fmt),
    )
    y_facade, s_facade = cake_act_qdq_det_train_forward(x, format=fmt, return_scales=True)
    y_backend, s_backend = backend.cake_act_qdq_det_train_forward(x, format=fmt, return_scales=True)
    _same(y_facade, y_backend)
    _same(s_facade, s_backend)
    inplace = x.clone()
    assert cake_act_qdq_det_train_forward(inplace, format=fmt, out=inplace) is inplace
    _same(inplace, y_backend)


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_domain_locality_on_default_memory_matches_the_backend(fmt: str) -> None:
    """On default memory ``locality="domain"`` runs the ``"off"`` program: the same bytes, through the facade and the
    backend, and the plan of the facade is the backend's."""
    x = _input((129, 32, 128))
    out = torch.empty_like(x)
    want = backend.cake_act_qdq_det_train_forward(x, format=fmt)
    _same(cake_act_qdq_det_train_forward(x, format=fmt, out=out, locality="domain"), want)
    assert cake_act_qdq_det_train_domain_plan(x, format=fmt, out=out) == backend.cake_act_qdq_det_train_domain_plan(
        x, format=fmt, out=out
    )
    plan = cake_act_qdq_det_train_domain_plan(x, format=fmt, out=out)
    assert plan["route"] == "plain" and plan["first_tile"] is None and plan["first_die"] is None, plan
