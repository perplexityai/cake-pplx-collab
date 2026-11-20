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

Tests of the public surface of the Cake routed-expert clamped SwiGLU training kernels (SM100 / SM103 / SM107): the
``cake_pplx.cake_clamped_swiglu_det_train`` facade.  The facade's entry points are pass-throughs to
``cake_pplx._backends.cake_clamped_swiglu_det_train``: the same signatures (``limit`` a required keyword everywhere)
and the same bits as the backend; its autograd class and constants are the backend's objects, and its entries call the
backend's step functions and autograd entry (``apply``, then the forward launch), bound at their first call.  The
mode rules of the deterministic package: no entry point takes a ``deterministic`` keyword (``TypeError``, checked
without a device), and ``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``) neither raises nor warns
in any call.  The facade reads no environment variable.  The numerical contract is
``test_cake_clamped_swiglu_det_train.py``'s.
"""

from __future__ import annotations

import ast
import inspect
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

import cake_pplx._backends.cake_clamped_swiglu_det_train as backend
import cake_pplx.cake_clamped_swiglu_det_train as facade
from cake_pplx.cake_clamped_swiglu_det_train import (
    CakeClampedSwiGLUDetFunction,
    cake_clamped_swiglu_det,
    cake_clamped_swiglu_det_train_backward,
    cake_clamped_swiglu_det_train_forward,
    is_cake_clamped_swiglu_det_train_supported,
)

I = 2304
L = 10.0
requires_device = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cake_clamped_swiglu_det_train_supported(torch.device("cuda")),
    reason="Cake clamped SwiGLU training needs an SM100 / SM103 / SM107 GPU",
)
ENTRY_NAMES = (
    "cake_clamped_swiglu_det",
    "cake_clamped_swiglu_det_train_backward",
    "cake_clamped_swiglu_det_train_forward",
    "is_cake_clamped_swiglu_det_train_supported",
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


def _inputs(rows: int, *, seed: int, merged: bool = True, weighted: bool = True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (6.0 * torch.randn(rows, 2 * I, device="cuda", generator=g)).bfloat16()
    w = 1.5 * torch.rand(rows, device="cuda", generator=g) if weighted else None
    dh = torch.randn(rows, I, device="cuda", generator=g).bfloat16()
    if merged:
        return gate_up, None, w, dh
    return gate_up[:, :I].contiguous(), gate_up[:, I:].contiguous(), w, dh


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
    argument (no device needed), and the autograd function takes no fifth input but its internal launch slot: a mode
    there raises ``TypeError`` before any argument is read."""
    x = torch.zeros(4, 2 * I, dtype=torch.bfloat16)
    dh = torch.zeros(4, I, dtype=torch.bfloat16)
    for module in NAMESPACES.values():
        runs = {
            "autograd": lambda: module.cake_clamped_swiglu_det(x, limit=L, deterministic=mode),
            "forward": lambda: module.cake_clamped_swiglu_det_train_forward(x, limit=L, deterministic=mode),
            "backward": lambda: module.cake_clamped_swiglu_det_train_backward(dh, x, limit=L, deterministic=mode),
            "supported": lambda: module.is_cake_clamped_swiglu_det_train_supported(
                torch.device("cpu"), deterministic=mode
            ),
        }
        for what, run in runs.items():
            with pytest.raises(TypeError, match="deterministic"):
                run()
    forward_params = list(inspect.signature(CakeClampedSwiGLUDetFunction.forward).parameters)
    assert forward_params == ["ctx", "gate_up_or_gate", "up", "weight", "limit", "_launch"]
    assert inspect.signature(CakeClampedSwiGLUDetFunction.forward).parameters["_launch"].default is None
    if mode is not None:
        with pytest.raises(TypeError, match="fifth argument"):
            CakeClampedSwiGLUDetFunction.apply(x, None, None, L, mode)


def test_limit_is_required_everywhere() -> None:
    for module in NAMESPACES.values():
        for name in ENTRY_NAMES[:3]:
            parameter = inspect.signature(getattr(module, name)).parameters["limit"]
            assert parameter.kind is parameter.KEYWORD_ONLY and parameter.default is parameter.empty, name


def test_lazy_objects_are_the_backends() -> None:
    assert facade.CakeClampedSwiGLUDetFunction is CakeClampedSwiGLUDetFunction is backend.CakeClampedSwiGLUDetFunction
    assert facade.ARCHES is backend.ARCHES == ("sm_100a", "sm_103a", "sm_107a")
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES is backend.SUPPORTED_COMPUTE_CAPABILITIES
    assert facade.INTERMEDIATE_SIZE == backend.INTERMEDIATE_SIZE == I
    assert sorted(dir(facade)) == dir(facade)
    assert set(facade.__all__) <= set(dir(facade))


def test_supported_query_without_a_device() -> None:
    assert is_cake_clamped_swiglu_det_train_supported(torch.device("cpu")) is False
    assert is_cake_clamped_swiglu_det_train_supported("cpu") is False


@pytest.mark.parametrize("name", ENTRY_NAMES[:3])
def test_facade_refuses_what_the_backend_refuses(name: str) -> None:
    """Without a device: each facade entry (which runs the backend's step or autograd class directly) raises the
    backend entry's exception, type and message, for CPU operands and for an operand that is not a tensor."""
    dh = torch.zeros(4, I, dtype=torch.bfloat16)
    calls = {
        "cake_clamped_swiglu_det": lambda module, x: module.cake_clamped_swiglu_det(x, limit=L),
        "cake_clamped_swiglu_det_train_forward": lambda module, x: module.cake_clamped_swiglu_det_train_forward(
            x, limit=L
        ),
        "cake_clamped_swiglu_det_train_backward": lambda module, x: module.cake_clamped_swiglu_det_train_backward(
            dh, x, limit=L
        ),
    }
    for operand in (torch.zeros(4, 2 * I, dtype=torch.bfloat16), [1.0]):
        raised = []
        for module in (facade, backend):
            with pytest.raises((TypeError, ValueError)) as info:
                calls[name](module, operand)
            raised.append((type(info.value), str(info.value)))
        assert raised[0] == raised[1], (name, raised)


def test_the_facade_binds_the_backend_functions_at_their_first_call() -> None:
    """Without a device: after a first call of each facade entry (each refused for a CPU operand), the facade's steps
    and autograd entry call the backend's own step functions and positional autograd entry (``apply``, then the
    forward launch), bound to globals of the facade (no cache lookup per call)."""
    dh = torch.zeros(4, I, dtype=torch.bfloat16)
    x = torch.zeros(4, 2 * I, dtype=torch.bfloat16)
    for call in (
        lambda: facade.cake_clamped_swiglu_det_train_forward(x, limit=L),
        lambda: facade.cake_clamped_swiglu_det_train_backward(dh, x, limit=L),
        lambda: facade.cake_clamped_swiglu_det(x, limit=L),
    ):
        with pytest.raises(ValueError):
            call()
    assert facade._step_forward is backend._forward_step
    assert facade._step_backward is backend._backward_step
    assert facade._autograd is backend._autograd


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
@pytest.mark.parametrize(("merged", "weighted"), [(True, True), (False, True), (True, False)])
def test_facade_matches_backend_bits(merged: bool, weighted: bool) -> None:
    first, second, w, dh = _inputs(257, seed=11, merged=merged, weighted=weighted)
    h = cake_clamped_swiglu_det_train_forward(first, second, w, limit=L)
    assert torch.equal(h, backend.cake_clamped_swiglu_det_train_forward(first, second, w, limit=L))
    got = cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L)
    want = backend.cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L)
    for a, b in zip(got, want, strict=True):
        assert (a is None and b is None) or torch.equal(a, b)
    leaves = [None if t is None else t.clone().requires_grad_(True) for t in (first, second, w)]
    out = cake_clamped_swiglu_det(*leaves, limit=L)
    out.backward(dh)
    assert torch.equal(out, h)
    for leaf, grad in zip(leaves, got, strict=True):
        if leaf is not None:
            assert torch.equal(leaf.grad, grad)


@requires_device
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_deterministic_algorithms_neither_raise_nor_warn(warn_only: bool, determinism_setting) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) with every warning an error, the
    steps (merged, two tensors, unweighted, no rows), the support query, autograd and a non-reentrant checkpoint run
    as with the mode off: the same bits."""
    merged = _inputs(257, seed=13)
    split = _inputs(65, seed=14, merged=False, weighted=False)
    empty = _inputs(0, seed=15)

    def run():
        out = []
        for first, second, w, dh in (merged, split, empty):
            out.append(cake_clamped_swiglu_det_train_forward(first, second, w, limit=L))
            out += [t for t in cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L) if t is not None]
        assert is_cake_clamped_swiglu_det_train_supported(merged[0].device)
        first, _, w, dh = merged
        for checkpoint in (False, True):
            leaves = [first.clone().requires_grad_(True), None, w.clone().requires_grad_(True)]
            if checkpoint:
                y = torch.utils.checkpoint.checkpoint(cake_clamped_swiglu_det, *leaves, limit=L, use_reentrant=False)
            else:
                y = cake_clamped_swiglu_det(*leaves, limit=L)
            y.backward(dh)
            out += [y, leaves[0].grad, leaves[2].grad]
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
