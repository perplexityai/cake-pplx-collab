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
backend's step functions, ``apply`` and, in a backward with saved-tensors hooks set, the held-launch entry, bound at
their first call.  Through the facade, the forward of a checkpoint's recomputation launches after autograd saved its
inputs (one held launch; merged, two-tensor, strided and copied inputs, weighted and unweighted), a real non-reentrant
checkpoint of the op alone launches no forward in the backward, and with no rows no call (the steps, autograd, the held
call, a checkpoint) looks up, loads or launches a kernel.  ``locality`` (the last keyword of the steps and the
autograd entry, ``"off"`` by default) is checked by the backend for every value but ``"off"``, with the same errors
through the facade, whose ``"domain"`` entries bind the backend's at their first call and give the ``"off"`` bits; the
locality-domain helpers and plans have their backend functions' signatures.  The mode rules of the deterministic
package: no entry point takes a ``deterministic`` keyword (``TypeError``, checked without a device), and
``torch.use_deterministic_algorithms(True)`` (strict or ``warn_only``) neither raises nor warns in any call.  The
facade reads no environment variable.  The numerical contract is ``test_cake_clamped_swiglu_det_train.py``'s.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import warnings
from collections.abc import Callable
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
    "cake_clamped_swiglu_det_train_domain_empty",
    "cake_clamped_swiglu_det_train_domain_release",
    "cake_clamped_swiglu_det_train_forward_domain_plan",
    "cake_clamped_swiglu_det_train_backward_domain_plan",
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


@pytest.fixture
def events(monkeypatch) -> list:
    """The stage of every backend launch (``fwd``, ``bwd`` or ``bwd_unweighted``) in order, beside the tests' own
    ``"pack"`` marks: each FFI entry is wrapped as a launcher looks it up, and the launchers are made afresh."""
    found: list = []
    entry = backend._entry

    def spy(name, arch):
        run, stage = entry(name, arch), backend.cake_jit.MODULES[name]["stage"]

        def recorded(*args):
            found.append(stage)
            return run(*args)

        return recorded

    monkeypatch.setattr(backend, "_entry", spy)
    monkeypatch.setattr(backend, "_launcher", functools.cache(backend._launcher.__wrapped__))
    return found


@pytest.fixture
def held(monkeypatch) -> list:
    """Every held launch the backend makes (one per call that keeps its forward launch), in order."""
    made: list = []

    class Counted(backend._HeldLaunch):
        __slots__ = ()

        def __init__(self) -> None:
            made.append(self)
            super().__init__()

    monkeypatch.setattr(backend, "_HeldLaunch", Counted)
    return made


def _inputs(rows: int, *, seed: int, merged: bool = True, weighted: bool = True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (6.0 * torch.randn(rows, 2 * I, device="cuda", generator=g)).bfloat16()
    w = 1.5 * torch.rand(rows, device="cuda", generator=g) if weighted else None
    dh = torch.randn(rows, I, device="cuda", generator=g).bfloat16()
    if merged:
        return gate_up, None, w, dh
    return gate_up[:, :I].contiguous(), gate_up[:, I:].contiguous(), w, dh


# NaN columns on each side of the gate and the up in a wider buffer: "strided" keeps row strides that are multiples of
# 8 elements, "copied" makes ones that are not (the backend copies those inputs once).
PADS = {"strided": (32, 8), "copied": (3, 1)}


def _laid_out(rows: int, *, seed: int, layout: str, weighted: bool):
    """:func:`_inputs` (the same values) in ``layout``: merged, split, or the gate and the up as row slices of wider
    buffers (``strided``, ``copied``)."""
    if layout == "merged":
        return _inputs(rows, seed=seed, weighted=weighted)
    gate, up, w, dh = _inputs(rows, seed=seed, merged=False, weighted=weighted)
    if layout == "split":
        return gate, up, w, dh
    placed = []
    for values, pad in zip((gate, up), PADS[layout], strict=True):
        buffer = torch.full((rows, I + 2 * pad), float("nan"), dtype=values.dtype, device=values.device)
        buffer[:, pad : pad + I] = values
        placed.append(buffer[:, pad : pad + I])
    return placed[0], placed[1], w, dh


def _fresh(operands) -> tuple[list, Callable[[], list]]:
    """Fresh leaves in the layout of ``operands`` (a row slice of a wider buffer is the same slice of a fresh leaf of
    its whole buffer; None stays None), and a function giving each operand's gradient."""
    out: list = []
    parts: list = []
    for x in operands:
        if x is None:
            out.append(None)
            parts.append(None)
        elif x._base is not None and not x.is_contiguous():
            base = x._base.detach().clone().requires_grad_(True)
            out.append(base.as_strided(x.shape, x.stride(), x.storage_offset()))
            parts.append((base, x))
        else:
            leaf = x.detach().clone().requires_grad_(True)
            out.append(leaf)
            parts.append((leaf, None))

    def gradients() -> list:
        return [
            None
            if part is None
            else part[0].grad
            if part[1] is None
            else part[0].grad.as_strided(part[1].shape, part[1].stride(), part[1].storage_offset())
            for part in parts
        ]

    return out, gradients


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
    argument (no device needed), and the autograd function takes no fifth input: a mode there raises ``TypeError``
    before any argument is read."""
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
    assert forward_params == ["ctx", "gate_up_or_gate", "up", "weight", "limit"]
    with pytest.raises(TypeError, match="positional argument"):
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


def test_the_facade_binds_the_backend_functions_at_their_first_call(monkeypatch) -> None:
    """Without a device: after a first call of each facade entry (each refused for a CPU operand; the autograd entry
    as it is and as in a backward with saved-tensors hooks set), the facade's steps and autograd entry call the
    backend's own step functions, ``CakeClampedSwiGLUDetFunction.apply`` and the held-launch entry ``_held`` (in a
    backward with hooks set), bound to globals of the facade (no cache lookup per call)."""
    dh = torch.zeros(4, I, dtype=torch.bfloat16)
    x = torch.zeros(4, 2 * I, dtype=torch.bfloat16)

    def hooked():
        with monkeypatch.context() as patch, torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            patch.setattr(facade, "_graph_task", lambda: 0)  # as in a backward (a checkpoint's recomputation)
            facade.cake_clamped_swiglu_det(x, limit=L)

    for call in (
        lambda: facade.cake_clamped_swiglu_det_train_forward(x, limit=L),
        lambda: facade.cake_clamped_swiglu_det_train_backward(dh, x, limit=L),
        lambda: facade.cake_clamped_swiglu_det(x, limit=L),
        hooked,
    ):
        with pytest.raises(ValueError):
            call()
    assert facade._step_forward is backend._forward_step
    assert facade._step_backward is backend._backward_step
    assert facade._apply == backend.CakeClampedSwiGLUDetFunction.apply  # a new bound method at each attribute read
    assert facade._held is backend._held


@pytest.mark.parametrize("locality", [None, 1, b"domain", "on", "Domain", ""])
def test_the_facade_checks_locality_as_the_backend_does(locality) -> None:
    """Without a device: a ``locality`` that is not a ``str`` (``TypeError``) or not ``"off"`` / ``"domain"``
    (``ValueError``) raises the same exception, type and message, through the facade and the backend, before any other
    argument is read."""
    dh = torch.zeros(4, I, dtype=torch.bfloat16)
    x = torch.zeros(4, 2 * I, dtype=torch.bfloat16)
    calls = (
        lambda module: module.cake_clamped_swiglu_det_train_forward(x, limit=L, locality=locality),
        lambda module: module.cake_clamped_swiglu_det_train_backward(dh, x, limit=L, locality=locality),
        lambda module: module.cake_clamped_swiglu_det(x, limit=L, locality=locality),
    )
    for call in calls:
        raised = []
        for module in (facade, backend):
            with pytest.raises((TypeError, ValueError), match="locality") as info:
                call(module)
            raised.append((type(info.value), str(info.value)))
        assert raised[0] == raised[1], raised
        assert raised[0][0] is (ValueError if isinstance(locality, str) else TypeError)


def test_the_facade_binds_the_locality_entries_at_their_first_call() -> None:
    """Without a device: after a first ``locality="domain"`` call of each facade entry (each refused for a CPU
    operand), the facade calls the backend's own locality entries, bound to globals of the facade."""
    dh = torch.zeros(4, I, dtype=torch.bfloat16)
    x = torch.zeros(4, 2 * I, dtype=torch.bfloat16)
    for call in (
        lambda: facade.cake_clamped_swiglu_det_train_forward(x, limit=L, locality="domain"),
        lambda: facade.cake_clamped_swiglu_det_train_backward(dh, x, limit=L, locality="domain"),
        lambda: facade.cake_clamped_swiglu_det(x, limit=L, locality="domain"),
    ):
        with pytest.raises(ValueError):
            call()
    assert facade._forward_locality is backend._forward_locality
    assert facade._backward_locality is backend._backward_locality
    assert facade._autograd_locality is backend._autograd_locality


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
@pytest.mark.parametrize(("merged", "weighted"), [(True, True), (False, True), (True, False)])
def test_facade_domain_calls_on_ordinary_memory_give_the_off_bits(merged: bool, weighted: bool) -> None:
    """Through the facade, ``locality="domain"`` on ordinary memory (the steps and autograd) gives the backend's
    ``"off"`` bits on every architecture, and the plans name a program of the ``"off"`` route (the backward's for
    reason ``"arch"``: it has no twin)."""
    first, second, w, dh = _inputs(257, seed=12, merged=merged, weighted=weighted)
    h = backend.cake_clamped_swiglu_det_train_forward(first, second, w, limit=L)
    want = backend.cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L)
    assert torch.equal(cake_clamped_swiglu_det_train_forward(first, second, w, limit=L, locality="domain"), h)
    got = cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L, locality="domain")
    for a, b in zip(got, want, strict=True):
        assert (a is None and b is None) or torch.equal(a, b)
    backward_plan = facade.cake_clamped_swiglu_det_train_backward_domain_plan(dh, first, second, w, limit=L)
    for plan in (facade.cake_clamped_swiglu_det_train_forward_domain_plan(first, second, w, limit=L), backward_plan):
        assert plan["route"] == "plain" and plan["program"] in backend.cake_jit.MODULES, plan
    assert backward_plan["reason"] == "arch", backward_plan
    leaves = [None if t is None else t.clone().requires_grad_(True) for t in (first, second, w)]
    out = cake_clamped_swiglu_det(*leaves, limit=L, locality="domain")
    out.backward(dh)
    assert torch.equal(out, h)
    for leaf, grad in zip(leaves, want, strict=True):
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


@requires_device
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("layout", ["merged", "split", "strided", "copied"])
def test_the_facade_launches_a_recomputed_forward_after_autograd_saved_the_inputs(
    layout: str, weighted: bool, events, held, monkeypatch
) -> None:
    """Through the facade, as in a backward with saved-tensors hooks set (a non-reentrant checkpoint's recomputation;
    the facade's graph-task query answers "a backward runs"), the forward launches after the pack hooks of its inputs,
    through one held launch, in every layout (``strided`` and ``copied`` take the backend's full checks); with hooks
    outside a backward, and in a backward without hooks, it launches inside ``forward`` (before the pack hooks), with no
    held launch.  Every way gives the steps' bits and one backward launch."""
    first, second, w, dh = _laid_out(129, seed=21, layout=layout, weighted=weighted)
    want_h = cake_clamped_swiglu_det_train_forward(first, second, w, limit=L)
    want = cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L)
    packs = sum(t is not None for t in (first, second, w))
    bwd = "bwd" if weighted else "bwd_unweighted"

    def pack(x: torch.Tensor) -> torch.Tensor:
        events.append("pack")
        return x

    def run(operands, grad, *, in_backward: bool, hooks: bool):
        leaves, gradients = _fresh(operands)
        events.clear()
        with monkeypatch.context() as patch:
            if in_backward:
                patch.setattr(facade, "_graph_task", lambda: 0)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x) if hooks else contextlib.nullcontext():
                h = cake_clamped_swiglu_det(*leaves, limit=L)
        order = list(events)
        h.backward(grad)
        return order, events[len(order) :], h, gradients()

    cases = (
        (True, True, ["pack"] * packs + ["fwd"], 1),
        (False, True, ["fwd"] + ["pack"] * packs, 0),
        (True, False, ["fwd"], 0),
    )
    for in_backward, hooks, expected, holds in cases:
        held.clear()
        order, after, h, grads = run((first, second, w), dh, in_backward=in_backward, hooks=hooks)
        assert order == expected, (in_backward, hooks, order)
        assert after == [bwd], (in_backward, hooks, after)
        assert len(held) == holds, (in_backward, hooks)
        assert torch.equal(h, want_h)
        for got, ref in zip(grads, want, strict=True):
            assert (got is None and ref is None) or torch.equal(got, ref)


@requires_device
@pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("merged", [True, False], ids=["merged", "split"])
def test_no_rows_through_the_facade_load_and_launch_nothing(merged: bool, weighted: bool, held, monkeypatch) -> None:
    """``R = 0`` through the facade, with every loader and launcher of the backend refusing: the steps, autograd, the
    held call (in a backward with saved-tensors hooks set; the facade's graph-task query answers "a backward runs") and
    a real non-reentrant checkpoint return empty outputs and gradients of the right shapes.  The held call runs only the
    pack hooks of its inputs; it and the checkpoint's recomputation each make one held launch and keep nothing in it."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call with no rows looked up, loaded or launched a kernel")

    for name in ("run_module", "_launcher", "load", "_entry"):
        monkeypatch.setattr(backend, name, refuse)
    first, second, w, dh = _inputs(0, seed=22, merged=merged, weighted=weighted)
    h = cake_clamped_swiglu_det_train_forward(first, second, w, limit=L)
    assert h.shape == (0, I) and h.dtype == torch.bfloat16
    grads = cake_clamped_swiglu_det_train_backward(dh, first, second, w, limit=L)
    for grad, operand in zip(grads, (first, second, w), strict=True):
        assert (grad is None and operand is None) or grad.shape == operand.shape
    packs = sum(t is not None for t in (first, second, w))
    marks: list = []

    def pack(x: torch.Tensor) -> torch.Tensor:
        marks.append("pack")
        return x

    def held_call(*leaves):
        with monkeypatch.context() as patch:
            patch.setattr(facade, "_graph_task", lambda: 0)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
                return cake_clamped_swiglu_det(*leaves, limit=L)

    calls = {
        "autograd": (lambda *leaves: cake_clamped_swiglu_det(*leaves, limit=L), [], 0),
        "held": (held_call, ["pack"] * packs, 1),
        "checkpoint": (
            lambda *leaves: torch.utils.checkpoint.checkpoint(
                cake_clamped_swiglu_det, *leaves, limit=L, use_reentrant=False
            ),
            [],
            1,
        ),
    }
    for what, (call, packed, holds) in calls.items():
        leaves = [None if t is None else t.clone().requires_grad_(True) for t in (first, second, w)]
        held.clear()
        marks.clear()
        out = call(*leaves)
        assert marks == packed, what
        assert out.shape == (0, I) and out.grad_fn is not None, what
        out.backward(dh)
        for leaf in leaves:
            assert leaf is None or leaf.grad.shape == leaf.shape, what
        assert len(held) == holds and not any(held), what


@requires_device
@pytest.mark.parametrize(("merged", "weighted"), [(True, True), (False, True), (True, False)])
def test_a_checkpoint_through_the_facade_launches_the_forward_again_only_when_needed(
    merged: bool, weighted: bool, events
) -> None:
    """Through the facade under ``torch.utils.checkpoint`` (early stop on): the op alone, its down projection outside,
    launches no forward in the backward (``use_reentrant=False``); the op with the down projection, which saves ``h``,
    launches one, and so does the op alone in a reentrant region.  Each launches one forward before the backward and
    one backward launch, with the bits of the same step without a checkpoint."""
    first, second, w, _ = _inputs(129, seed=23, merged=merged, weighted=weighted)
    g = torch.Generator(device="cuda").manual_seed(24)
    wd = (torch.randn(256, I, device="cuda", generator=g) / I**0.5).bfloat16()
    dy = torch.randn(129, 256, device="cuda", generator=g).bfloat16()
    checkpoint = torch.utils.checkpoint.checkpoint
    op = functools.partial(cake_clamped_swiglu_det, limit=L)
    down = torch.nn.functional.linear
    bwd = "bwd" if weighted else "bwd_unweighted"
    regions = {
        "no checkpoint": (lambda f, s, u, d: down(op(f, s, u), d), 0),
        "op alone": (lambda f, s, u, d: down(checkpoint(op, f, s, u, use_reentrant=False), d), 0),
        "op and the down projection": (
            lambda f, s, u, d: checkpoint(lambda *a: down(op(*a[:3]), a[3]), f, s, u, d, use_reentrant=False),
            1,
        ),
        "op alone, reentrant": (lambda f, s, u, d: down(checkpoint(op, f, s, u, use_reentrant=True), d), 1),
    }
    want = None
    for what, (region, recomputes) in regions.items():
        leaves = [None if t is None else t.clone().requires_grad_(True) for t in (first, second, w, wd)]
        events.clear()
        y = region(*leaves)
        before = list(events)
        y.backward(dy)
        after = events[len(before) :]
        assert before == ["fwd"], (what, before)
        assert sorted(after) == sorted(["fwd"] * recomputes + [bwd]), (what, after)
        got = [y.detach()] + [t.grad for t in leaves if t is not None]
        if want is None:
            want = got
        for a, b in zip(got, want, strict=True):
            assert torch.equal(a, b), what
