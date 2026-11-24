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

Tests of the public surface of the Cake SiTU-GLU training kernels (SM100 / SM103): the
``cake_pplx.cake_situ_glu_det_train`` facade.  The facade's entry points are pass-throughs to
``cake_pplx._backends.cake_situ_glu_det_train``: the same signatures (``beta`` and ``lb`` keywords with the defaults
4.0 and 25.0 everywhere) and the same bits as the backend; its autograd class and constants are the backend's
objects.  The mode rules of the deterministic package: no entry point takes a ``deterministic`` keyword
(``TypeError``, checked without a device), and ``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``)
neither raises nor warns in any call.  The facade reads no environment variable.  The numerical contract is
``test_cake_situ_glu_det_train.py``'s.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

import cake_pplx._backends.cake_situ_glu_det_train as backend
import cake_pplx.cake_situ_glu_det_train as facade
from cake_pplx.cake_situ_glu_det_train import (
    CakeSiTUGLUDetFunction,
    cake_situ_glu_det,
    cake_situ_glu_det_train_backward,
    cake_situ_glu_det_train_forward,
    is_cake_situ_glu_det_train_supported,
)

WIDTH = 3072
requires_device = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cake_situ_glu_det_train_supported(torch.device("cuda")),
    reason="Cake SiTU-GLU training needs an SM100 / SM103 GPU",
)
ENTRY_NAMES = (
    "cake_situ_glu_det",
    "cake_situ_glu_det_train_backward",
    "cake_situ_glu_det_train_forward",
    "is_cake_situ_glu_det_train_supported",
)
NAMESPACES = {"facade": facade, "backend": backend}


@pytest.fixture
def determinism_setting():
    """Restore the caller's ``torch.use_deterministic_algorithms`` setting after the test; the test starts with it
    off."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _inputs(rows: int, *, seed: int, merged: bool = True, width: int = WIDTH):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (2.0 * torch.randn(rows, 2 * width, device="cuda", generator=g)).bfloat16()
    dh = torch.randn(rows, width, device="cuda", generator=g).bfloat16()
    if merged:
        return gate_up, None, dh
    return gate_up[:, :width].contiguous(), gate_up[:, width:].contiguous(), dh


# --------------------------------------------------------------------------- the surface (no device)


@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_facade_signatures_are_the_backends(name: str) -> None:
    """Each facade entry has its backend function's signature (names, kinds, defaults and annotations)."""
    assert inspect.signature(getattr(facade, name), eval_str=True) == inspect.signature(
        getattr(backend, name), eval_str=True
    )


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
@pytest.mark.parametrize("name", ENTRY_NAMES)
def test_det_entry_points_take_no_mode_keyword(namespace: str, name: str) -> None:
    entry = getattr(NAMESPACES[namespace], name)
    signature = inspect.signature(entry)
    assert "deterministic" not in signature.parameters, name
    assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values()), name
    with pytest.raises(TypeError):
        signature.bind_partial(deterministic=False)


@pytest.mark.parametrize("mode", [True, False, None], ids=["true", "false", "none"])
def test_a_mode_keyword_is_a_type_error(mode) -> None:
    """Every entry point of both namespaces raises ``TypeError`` for a ``deterministic`` keyword before it reads any
    argument (no device needed), and the autograd function takes no fifth input."""
    x = torch.zeros(4, 2 * WIDTH, dtype=torch.bfloat16)
    dh = torch.zeros(4, WIDTH, dtype=torch.bfloat16)
    for module in NAMESPACES.values():
        runs = {
            "autograd": lambda: module.cake_situ_glu_det(x, deterministic=mode),
            "forward": lambda: module.cake_situ_glu_det_train_forward(x, deterministic=mode),
            "backward": lambda: module.cake_situ_glu_det_train_backward(dh, x, deterministic=mode),
            "supported": lambda: module.is_cake_situ_glu_det_train_supported(torch.device("cpu"), deterministic=mode),
        }
        for what, run in runs.items():
            with pytest.raises(TypeError, match="deterministic"):
                run()
    forward_params = list(inspect.signature(CakeSiTUGLUDetFunction.forward).parameters)
    assert forward_params == ["ctx", "gate_up_or_gate", "up", "beta", "lb"]


def test_scalars_are_keywords_with_the_model_defaults_everywhere() -> None:
    for module in NAMESPACES.values():
        for name in ENTRY_NAMES[:3]:
            parameters = inspect.signature(getattr(module, name)).parameters
            for scalar, default in (("beta", 4.0), ("lb", 25.0)):
                parameter = parameters[scalar]
                assert parameter.kind is parameter.KEYWORD_ONLY and parameter.default == default, (name, scalar)


def test_lazy_objects_are_the_backends() -> None:
    assert facade.CakeSiTUGLUDetFunction is CakeSiTUGLUDetFunction is backend.CakeSiTUGLUDetFunction
    assert facade.ARCHES is backend.ARCHES == ("sm_100a", "sm_103a")
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES is backend.SUPPORTED_COMPUTE_CAPABILITIES
    assert facade.WIDTH_MULTIPLE == backend.WIDTH_MULTIPLE == 256
    assert sorted(dir(facade)) == dir(facade)
    assert set(facade.__all__) <= set(dir(facade))


def test_supported_query_without_a_device() -> None:
    assert is_cake_situ_glu_det_train_supported(torch.device("cpu")) is False
    assert is_cake_situ_glu_det_train_supported("cpu") is False


def test_the_facade_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(facade.__file__).read_text())
    uses = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"))
        or (isinstance(node, ast.Name) and node.id in ("environ", "getenv"))
    ]
    assert uses == []


# --------------------------------------------------------------------------- on a device


@requires_device
@pytest.mark.parametrize(("merged", "width"), [(True, 3072), (False, 3072), (True, 6144), (False, 33792)])
def test_facade_matches_backend_bits(merged: bool, width: int) -> None:
    first, second, dh = _inputs(65, seed=11, merged=merged, width=width)
    for scalars in ({}, {"beta": 3.0, "lb": 40.0}):
        h = cake_situ_glu_det_train_forward(first, second, **scalars)
        assert torch.equal(h, backend.cake_situ_glu_det_train_forward(first, second, **scalars))
        got = cake_situ_glu_det_train_backward(dh, first, second, **scalars)
        want = backend.cake_situ_glu_det_train_backward(dh, first, second, **scalars)
        for a, b in zip(got, want, strict=True):
            assert (a is None and b is None) or torch.equal(a, b)
        leaves = [None if t is None else t.clone().requires_grad_(True) for t in (first, second)]
        out = cake_situ_glu_det(*leaves, **scalars)
        out.backward(dh)
        assert torch.equal(out, h)
        for leaf, grad in zip(leaves, got, strict=True):
            if leaf is not None:
                assert torch.equal(leaf.grad, grad)


@requires_device
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_deterministic_algorithms_neither_raise_nor_warn(warn_only: bool, determinism_setting) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) with every warning an error, the
    steps (merged, two tensors, no rows), the support query, autograd and a non-reentrant checkpoint run as with the
    mode off: the same bits."""
    merged = _inputs(257, seed=13)
    split = _inputs(65, seed=14, merged=False, width=6144)
    empty = _inputs(0, seed=15)

    def run():
        out = []
        for first, second, dh in (merged, split, empty):
            out.append(cake_situ_glu_det_train_forward(first, second))
            out += [t for t in cake_situ_glu_det_train_backward(dh, first, second) if t is not None]
        assert is_cake_situ_glu_det_train_supported(merged[0].device)
        first, _, dh = merged
        for checkpoint in (False, True):
            leaf = first.clone().requires_grad_(True)
            if checkpoint:
                y = torch.utils.checkpoint.checkpoint(cake_situ_glu_det, leaf, use_reentrant=False)
            else:
                y = cake_situ_glu_det(leaf)
            y.backward(dh)
            out += [y, leaf.grad]
        return out

    want = run()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run()
    torch.cuda.synchronize()
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(a, b), i
