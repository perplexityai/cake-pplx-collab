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

Tests of the public surface of the Cake SiLU-gated RMSNorm training
package with the non-deterministic weight gradient (``nondet``): the
``cake_pplx.cake_rmsnorm_silu_gate_nondet_train`` facade and its interplay with
``torch.use_deterministic_algorithms``.

The mode is part of every name.  Every entry point of the facade and of the
standalone backend ``cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train``
runs only the non-deterministic kernels; no entry point takes a
``deterministic`` keyword (passing one is a ``TypeError``), and no call runs
the deterministic package ``cake_pplx.cake_rmsnorm_silu_gate_det_train``.

Every output of the package repeats bit for bit, except a BF16 weight
gradient of the fused backward (``bwd_bf16_fused``, a band of
``BACKWARD_BANDS`` past the small-row bands on every architecture, on sm_107a
between two repeatable bands; an FP32 weight gradient runs no fused
backward, and past its small-row band the repeatable backward,
``bwd_f32_rep``, or on sm_100a and sm_103a past 64 and up to 128 rows per SM
the static backward, ``bwd_f32``) and of the row-claims backwards
(``bwd_bf16_rc``, ``bwd_bf16_rc_mid``, where a band runs them), whose CTAs
claim 16-row blocks at run time and add the FP32 sums of their rows into
FP32 replicas with FP32 atomic adds (grid kind ``row_claims``), so its last
bits may vary from run to run; there every launch's ``dw`` is judged instead
of compared with the first launch's bits.

Every finite training output (``y``, ``dx``, ``dz`` and the weight gradient
of either dtype, accumulated or not) on the paths checked here is judged
against the PyTorch BF16 naive implementation of the operation, each against
a float64 oracle of the BF16 inputs, over 32 random inputs of a case
(:func:`tests.cake.test_helpers.silu_reference.judge`, the ``TIER`` lines of
the terminal summary).  Inputs that hold an infinite or a ``NaN`` value are
held to the oracle's classes instead: ``NaN``, ``+inf`` and ``-inf`` exactly
where the oracle has them (:func:`assert_same_classes`).  No test compares
this package's outputs with the deterministic package's.  This module
checks:

* the surface without a device: the exports (one autograd function, the
  backend's class), the facade's sources (no import of the deterministic
  package), the signature of every entry point (no ``deterministic``
  parameter), the ``TypeError`` for a ``deterministic`` keyword of any value
  from every facade and backend entry point under each
  ``torch.use_deterministic_algorithms`` setting (nothing launched), that the
  facade runs the backend with the same arguments (the backend replaced by
  recorders), the autograd function in both directions (its backward naming
  itself for the alert), the support query and the
  workspace reset, the backend's standalone form (no import of another
  ``cake_pplx`` module; a copy under another name imports), and the strides
  the backend passes for one-head and other inputs (a single head's stride as
  128, so packed one-head rows pass the kernels' packed-rows test), and the
  forward's host plan against version 0.2.17's tables and classes, by class
  and by input form (the module differs only in the bands version 0.2.18
  moved: a gate of more than one head sliced from a projection and
  whole-token rows);
* on a device: the alert of an autograd node whose backward runs the fused
  backward after ``torch.use_deterministic_algorithms(True)`` was
  enabled (exact texts: a ``RuntimeError`` before the launch, or a
  ``UserWarning`` under ``warn_only=True`` and the launch; none for a node
  whose weight needs no gradient or in any other band, the small-row bands
  among them); no call warns or
  raises in the bands whose outputs repeat their bits, strict or with
  ``warn_only=True`` (the forward, the backward into a BF16 and an FP32
  weight gradient, autograd and non-reentrant checkpointing); the
  ``TypeError`` for a mode keyword on real tensors before
  any launch; the entry points launch only the backend's modules; the
  non-deterministic backward refuses an FP32 workspace (the deterministic
  package's) naming both modes' workspaces, also in a call with no rows with
  the error of a call with rows, and the deterministic backward refuses the
  ``torch.uint8`` workspace of this package with its own FP32 workspace
  requirement; on the paths of the facade (accumulation,
  ``compute_grad_weight=False``, a reset workspace with nonzero bytes reused
  across shapes, two streams, a side stream, CUDA Graph replays, autograd
  and non-reentrant checkpointing) every output repeats bit for bit (an
  accumulated ``dw`` from the same starting value too) but a BF16 ``dw`` in
  the fused and the row-claims bands, and every output is judged; so are
  the forward around the ends of its ``FORWARD_BANDS`` bands (packed rows;
  and of ``FORWARD_TINY_GAP``), ``PROJECTION_FORWARD_BANDS`` bands (``z``
  the gate columns of a wider projection), ``HEAD_MAJOR_FORWARD_BANDS``
  bands (head-major rows; ``STRIDED_FORWARD_BANDS`` where the architecture
  has no such table) and ``SLICED_FORWARD_BANDS`` bands (``x`` and ``z`` both
  sliced from wider projections, whole-token rows; ``STRIDED_FORWARD_BANDS``
  where the architecture has no such table), each call's table that of its
  rows' class (``forward_layout``), the forward of ``x``
  or ``z`` broadcast over more than one token (token stride 0: the band of
  other layouts where the band of packed rows runs a program that reads them
  at their linear offsets, ``fwd_pipe`` where it runs ``fwd_keep_pipe``,
  else the program of packed rows with the token stride 0) and the
  backward at
  the first and the last rows of each ``BACKWARD_BANDS`` band of this device
  (the small-row backwards, one cluster each but the solo backward, one
  CTA, the static backward of an
  FP32 weight gradient on sm_100a and sm_103a, the fused backward and the
  repeatable backward of either weight gradient), each call launching
  ``module_for_rows``' module, and, where its band holds such rows, where a
  chunked backward's chunks grow by a block (the static band, up to 128 rows
  per SM, holds none on fewer than 512 SMs: those cases skip); so are
  strided inputs (``dy``, ``x`` and ``z`` not of one packed row stride) in
  the bands of the static backward of an FP32 weight gradient and of the
  fused and the repeatable backwards, the repeatable ones of either weight
  gradient (``LAYOUT_BAND_CASES``; its row-claims and claimed cases skip: no
  band runs them), launching the strides of their layout; finite gate
  inputs at and past the packed gate's range (around
  FP32 ``expf``'s overflow and in the gate's tail too), in the forward and
  the backward, are judged, and with infinite and ``NaN`` gates beside them
  and infinite upstream gradients (in the last row too, with one row, half a
  block and one row short of a whole block in the last row block of the
  band's program) the outputs have the oracle's ``NaN``, ``+inf`` and
  ``-inf`` positions in every band; one-head inputs of several layouts one
  row past the start of each module that warms rows in L2 and of the fused
  backward give the same bits for every layout (a BF16 ``dw`` of the fused
  and the row-claims backwards judged instead), passing the packed-rows test
  exactly when their rows follow one another, and are judged.

The data generators and the module tables of the non-deterministic kernels
are ``test_cake_rmsnorm_silu_gate_nondet_train.py``'s, reused here.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy
import pytest
import torch
import torch.utils.checkpoint

import cake_pplx
import cake_pplx.cake_rmsnorm_silu_gate_det_train as deterministic_facade
import cake_pplx.cake_rmsnorm_silu_gate_nondet_train as facade
import cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train as backend
from cake_pplx.cake_rmsnorm_silu_gate_nondet_train import (
    CakeRMSNormSiLUGateNondetFunction,
    cake_rmsnorm_silu_gate_nondet,
    cake_rmsnorm_silu_gate_nondet_train_backward,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace,
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes,
    cake_rmsnorm_silu_gate_nondet_train_forward,
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace,
    is_cake_rmsnorm_silu_gate_nondet_train_supported,
)
from cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train import (
    cake_jit as loader,
)
from tests.cake.test_helpers.cake_naive import DRAWS
from tests.cake.test_helpers.silu_reference import Draw, judge, oracle, silu_draw
from tests.norm import test_cake_rmsnorm_silu_gate_nondet_train as suite

D = 128
EPS = 1e-6
# The entry points of the facade and of the backend package (the same names;
# the autograd function is the backend's class).
ENTRY_NAMES = (
    "cake_rmsnorm_silu_gate_nondet_train_forward",
    "cake_rmsnorm_silu_gate_nondet_train_backward",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
    "cake_rmsnorm_silu_gate_nondet",
    "is_cake_rmsnorm_silu_gate_nondet_train_supported",
)
# Values a caller might pass as ``deterministic``.
MODES = (True, False, None, 0, 1, "False", numpy.bool_(False))
MODE_IDS = ["true", "false", "none", "zero", "one", "string", "numpy_false"]
BACKEND_DIR = Path(backend.__file__).parent
# The non-deterministic backward's refusal of an FP32 workspace (the
# deterministic package's), naming both modes' workspace dtypes and factories
# (the backend suite asserts the same text).
REFUSAL = (
    "workspace is FP32, a workspace of the deterministic mode "
    "(cake_pplx.cake_rmsnorm_silu_gate_det_train."
    "cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
    "non-deterministic backward needs its own zero-filled uint8 workspace "
    f"({backend.__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
)

nd_forward = cake_rmsnorm_silu_gate_nondet_train_forward
nd_backward = cake_rmsnorm_silu_gate_nondet_train_backward
nd_workspace = cake_rmsnorm_silu_gate_nondet_train_backward_workspace
nd_workspace_bytes = cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
nd_reset = cake_rmsnorm_silu_gate_nondet_train_reset_workspace
nd_gated_rmsnorm = cake_rmsnorm_silu_gate_nondet
# The deterministic package's backward, through its public entry point only
# (its refusal of this package's workspace).
det_backward = deterministic_facade.cake_rmsnorm_silu_gate_det_train_backward

def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture
def determinism_setting():
    """Restore the caller's ``torch.use_deterministic_algorithms`` setting
    after the test; the test starts with it off."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _cpu_tensors():
    x = torch.zeros(4, 2, D, dtype=torch.bfloat16)
    w = torch.ones(D, dtype=torch.bfloat16)
    return x, w


def _inputs(T: int, H: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x, z, dy = (
        torch.randn(T, H, D, device="cuda", generator=g).bfloat16() for _ in range(3)
    )
    w = (1.0 + 0.5 * torch.randn(D, device="cuda", generator=g)).bfloat16()
    return x, z, w, dy


def _stub_kernels(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace the backend's entry points and its forward and backward bodies
    by recorders (no device needed)."""
    calls: list = []
    for name in (*ENTRY_NAMES, "_forward", "_backward"):
        monkeypatch.setattr(
            backend, name, lambda *a, _name=name, **k: calls.append((_name, a, k))
        )
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: calls.append(("run", name))
    )
    return calls


def _mode_calls(module, ws) -> dict:
    """One call of every entry point of ``module`` (the facade or the backend)
    on CPU operands, taking extra keywords: ``entry -> call``."""
    x, w = _cpu_tensors()
    return {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, w, EPS, **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward(x, x, x, w, EPS, workspace=ws, **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(8, "cuda:0", **k),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(8, "cuda:0", **k),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **k: module.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **k),
        "cake_rmsnorm_silu_gate_nondet": lambda **k: module.cake_rmsnorm_silu_gate_nondet(x, x, w, EPS, **k),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **k: module.is_cake_rmsnorm_silu_gate_nondet_train_supported("cuda:0", **k),
    }  # fmt: skip


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s bits as integers (a ``NaN`` equals its own bits)."""
    ints = {2: torch.int16, 4: torch.int32, 8: torch.int64}
    return t.detach().view(ints[t.element_size()])


def assert_same_bits(what, got, want) -> None:
    """Run to run: ``got`` is ``want``'s bits."""
    assert got.dtype == want.dtype and got.shape == want.shape, (
        what,
        got.dtype,
        want.dtype,
        tuple(got.shape),
        tuple(want.shape),
    )
    assert torch.equal(_bits(got), _bits(want)), f"{what}: not the same bits"


def assert_same_classes(what, got, ref) -> None:
    """``got`` has the float64 oracle's classes: ``NaN`` exactly where it is
    ``NaN`` (any sign and payload), ``+inf`` and ``-inf`` exactly where it is."""
    got = got.detach().double().reshape(-1)
    ref = ref.double().reshape(-1)
    for name, cls in (
        ("NaN", torch.isnan),
        ("+inf", torch.isposinf),
        ("-inf", torch.isneginf),
    ):
        assert torch.equal(cls(got), cls(ref)), f"{what}: {name} positions vs FP64"


def _assert_refusal(error: BaseException) -> None:
    """The refusal of an FP32 workspace is :data:`REFUSAL`, naming both
    modes' workspaces."""
    assert str(error) == REFUSAL, str(error)


# --------------------------------------------------------------------------- the surface (no device)


def test_top_level_exports_are_the_facade() -> None:
    assert cake_pplx.cake_rmsnorm_silu_gate_nondet_train.cake_rmsnorm_silu_gate_nondet is cake_rmsnorm_silu_gate_nondet
    assert (
        cake_pplx.cake_rmsnorm_silu_gate_nondet_train.cake_rmsnorm_silu_gate_nondet_train_forward
        is cake_rmsnorm_silu_gate_nondet_train_forward
    )
    assert (
        cake_pplx.cake_rmsnorm_silu_gate_nondet_train.cake_rmsnorm_silu_gate_nondet_train_backward
        is cake_rmsnorm_silu_gate_nondet_train_backward
    )
    assert (
        cake_pplx.cake_rmsnorm_silu_gate_nondet_train.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
        is cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
    )
    # one non-deterministic autograd function: the backend's class, reached
    # through the facade and the top level
    assert (
        CakeRMSNormSiLUGateNondetFunction is backend.CakeRMSNormSiLUGateNondetFunction
    )
    assert (
        cake_pplx.cake_rmsnorm_silu_gate_nondet_train.CakeRMSNormSiLUGateNondetFunction
        is CakeRMSNormSiLUGateNondetFunction
    )
    # the det names stay the det package's objects
    assert (
        cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det
        is deterministic_facade.cake_rmsnorm_silu_gate_det
    )
    assert (
        cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_forward
        is deterministic_facade.cake_rmsnorm_silu_gate_det_train_forward
    )


def test_facade_imports_no_deterministic_package() -> None:
    """The facade never runs the deterministic package: its source imports
    nothing from it."""
    tree = ast.parse(Path(facade.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        assert not any("silu_gate_det" in name for name in names), (node.lineno, names)


@pytest.mark.parametrize("namespace", ["facade", "backend"])
def test_no_entry_point_takes_a_mode_keyword(namespace: str) -> None:
    """The mode is in the name (``nondet``): no entry point of the facade or
    the backend package has a ``deterministic`` parameter, and the autograd
    function takes no fifth input."""
    module = {"facade": facade, "backend": backend}[namespace]
    assert set(ENTRY_NAMES) | {"CakeRMSNormSiLUGateNondetFunction"} == set(
        module.__all__
    ) - {"ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"}
    for name in ENTRY_NAMES:
        parameters = inspect.signature(getattr(module, name)).parameters
        assert "deterministic" not in parameters, (namespace, name)
    assert list(
        inspect.signature(CakeRMSNormSiLUGateNondetFunction.forward).parameters
    ) == ["ctx", "x", "z", "weight", "eps"]


@pytest.mark.parametrize("value", MODES, ids=MODE_IDS)
def test_a_mode_keyword_is_a_type_error(
    value, monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """``deterministic=<any value>`` is a ``TypeError`` from every facade and
    backend entry point before any call reaches a kernel, under each
    ``torch.use_deterministic_algorithms`` setting; so is a fifth input of the
    autograd function (the backend's bodies recorded: no device needed)."""
    x, w = _cpu_tensors()
    ws = torch.zeros(64, dtype=torch.uint8)
    launched: list = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: launched.append(name)
    )
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for module in (facade, backend):
            for call in _mode_calls(module, ws).values():
                with pytest.raises(TypeError, match="deterministic"):
                    call(deterministic=value)
        with pytest.raises(TypeError):
            CakeRMSNormSiLUGateNondetFunction.apply(x, x, w, EPS, value)
    assert launched == []
    assert not bool(ws.any())


def test_facade_runs_the_backend_with_the_same_arguments(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """Each facade entry point calls the backend entry point of the same name
    with the same arguments and nothing else (stubbed)."""
    calls = _stub_kernels(monkeypatch)
    x, w = _cpu_tensors()
    ws = torch.zeros(64, dtype=torch.uint8)
    cake_rmsnorm_silu_gate_nondet_train_forward(x, x, w, EPS, out=x)
    cake_rmsnorm_silu_gate_nondet_train_backward(
        x, x, x, w, EPS, workspace=ws, compute_grad_weight=False
    )
    cake_rmsnorm_silu_gate_nondet(x, x, w, EPS)
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace(7, "cuda:0")
    cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(7, "cuda:0")
    cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)
    is_cake_rmsnorm_silu_gate_nondet_train_supported("cuda:0")
    names = [c[0] for c in calls]
    assert names == [
        "cake_rmsnorm_silu_gate_nondet_train_forward",
        "cake_rmsnorm_silu_gate_nondet_train_backward",
        "cake_rmsnorm_silu_gate_nondet",
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
        "is_cake_rmsnorm_silu_gate_nondet_train_supported",
    ], names
    assert calls[0][1:] == ((x, x, w, EPS), {"out": x})
    assert calls[1][1] == (x, x, x, w, EPS)
    assert calls[1][2] == {
        "grad_x": None,
        "grad_z": None,
        "grad_weight": None,
        "accumulate_grad_weight": False,
        "compute_grad_weight": False,
        "workspace": ws,
    }
    assert calls[2][1:] == ((x, x, w, EPS), {})
    assert calls[3][1:] == ((7, "cuda:0"), {})
    assert calls[4][1:] == ((7, "cuda:0"), {})
    assert calls[5][1:] == ((ws,), {})
    assert calls[6][1:] == (("cuda:0",), {})


def test_autograd_function_runs_the_backend_in_both_directions(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """:class:`CakeRMSNormSiLUGateNondetFunction` (the backend's class, also an
    attribute of the facade) runs the backend's forward and backward bodies;
    its backward names the autograd backward for the alert (``entry``).
    Neither stubbed body warns under ``torch.use_deterministic_algorithms(True)``
    (bodies stubbed: no device needed)."""
    calls: list = []

    def forward(x, z, w, eps, *, out):
        calls.append(("fwd", out))
        return x * 1

    def backward(dy, x, z, w, eps, **kwargs):
        calls.append(("bwd", kwargs))
        return dy * 1, dy * 1, w * 1

    monkeypatch.setattr(backend, "_forward", forward)
    monkeypatch.setattr(backend, "_backward", backward)
    x, w = _cpu_tensors()
    leaves = [t.clone().requires_grad_(True) for t in (x, x, w)]
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = CakeRMSNormSiLUGateNondetFunction.apply(*leaves, EPS)
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
        y.backward(torch.ones_like(y))
    assert calls == [
        ("fwd", None),
        (
            "bwd",
            {
                "grad_x": None,
                "grad_z": None,
                "grad_weight": None,
                "accumulate_grad_weight": False,
                "compute_grad_weight": True,
                "workspace": None,
                "entry": "CakeRMSNormSiLUGateNondetFunction.backward",
            },
        ),
    ]


def test_support_query_ignores_torch_determinism(determinism_setting) -> None:
    torch.use_deterministic_algorithms(True)
    cpu = torch.device("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert is_cake_rmsnorm_silu_gate_nondet_train_supported(cpu) is False
        assert backend.is_cake_rmsnorm_silu_gate_nondet_train_supported(cpu) is False


def test_reset_workspace_passes_through(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """The facade's reset resets through the backend whatever
    ``torch.use_deterministic_algorithms`` says (a fill has no reduction); the
    backend refuses an FP32 buffer (checked on the device)."""
    resets: list = []
    monkeypatch.setattr(
        backend,
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
        lambda workspace, **k: resets.append((workspace, k)),
    )
    ws = torch.ones(8, dtype=torch.uint8)
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)
    assert resets == [(ws, {})] * 3


def test_backend_package_imports_no_other_package() -> None:
    """The backend package is standalone: no source file imports from a parent
    package or another ``cake_pplx`` module, and every
    relative import names a module of the package.  The shared loader
    ``cake_build.py`` reaches its JIT fallback, the package's own JIT shim
    ``_cake_pplx_jit.py``, through a relative import inside a function when no
    prebuilt library exists."""
    sources = sorted(BACKEND_DIR.glob("*.py"))
    assert {p.name for p in sources} == {
        "__init__.py",
        "_cake_pplx_jit.py",
        "cake_build.py",
        "cake_jit.py",
    }
    jit_fallback = []
    for path in sources:
        text = path.read_text()
        assert "import_module(" not in text and "__import__(" not in text, path.name
        tree = ast.parse(text)
        in_function = {
            id(node)
            for function in ast.walk(tree)
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(function)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                assert node.level == 1, (path.name, node.lineno, node.level)
                names = (
                    [node.module.split(".")[0]]
                    if node.module
                    else [alias.name for alias in node.names]
                )
                for name in names:
                    assert (BACKEND_DIR / f"{name}.py").is_file(), (path.name, name)
                if node.module == "_cake_pplx_jit":
                    assert path.name == "cake_build.py" and id(node) in in_function, (
                        path.name,
                        node.lineno,
                    )
                    jit_fallback.append(node.module)
                continue
            if isinstance(node, ast.ImportFrom):
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                assert module != "cake_pplx" and not module.startswith("cake_pplx."), (
                    path.name,
                    node.lineno,
                    module,
                )
    assert jit_fallback, "the cake_build JIT fallback import was not found"


_VENDORED_CHILD = r"""
import importlib, json, sys

sys.path.insert(0, sys.argv[1])
import torch

package = importlib.import_module(sys.argv[2])
importlib.import_module(sys.argv[2] + ".cake_jit")
importlib.import_module(sys.argv[2] + ".cake_build")
x = torch.zeros(2, 128, dtype=torch.bfloat16)
out = {
    "supported_cpu": package.is_cake_rmsnorm_silu_gate_nondet_train_supported(
        torch.device("cpu")
    ),
    "all": sorted(package.__all__),
}
try:
    package.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, x[0], deterministic=False)
except TypeError as error:
    out["mode_keyword"] = str(error)
try:
    package.cake_rmsnorm_silu_gate_nondet_train_forward(x, x, x[0])
except ValueError as error:
    out["cpu"] = str(error)
print(json.dumps(out))
"""


def test_vendored_copy_imports(tmp_path: Path) -> None:
    """The backend package copied under another top-level name (the vendored
    gate's first step, without its sources) imports, answers its support query,
    takes no mode keyword and refuses a CPU tensor."""
    neutral = "vendored_cake_rmsnorm_silu_gate_nondet_train"
    shutil.copytree(
        BACKEND_DIR,
        tmp_path / neutral,
        ignore=shutil.ignore_patterns("aot", "__pycache__", "csrc", "build.json"),
    )
    run = subprocess.run(
        [sys.executable, "-c", _VENDORED_CHILD, str(tmp_path), neutral],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-8:])
    assert run.returncode == 0, tail
    out = json.loads(run.stdout.strip().splitlines()[-1])
    assert out["supported_cpu"] is False
    assert out["all"] == sorted(backend.__all__)
    assert "deterministic" in out["mode_keyword"], out["mode_keyword"]
    assert "CUDA" in out["cpu"], out["cpu"]


# --------------------------------------------------------------------------- on the device


@requires_device
def test_autograd_backward_alerts_once_deterministic_algorithms_are_enabled(
    monkeypatch: pytest.MonkeyPatch, determinism_setting
) -> None:
    """A node of :func:`cake_rmsnorm_silu_gate_nondet` is the backend's
    :class:`CakeRMSNormSiLUGateNondetFunction`.  When
    ``torch.use_deterministic_algorithms(True)`` is enabled between its forward
    and its backward, the backward of a node in the first rows of the fused
    band ``bwd_bf16_fused`` (``suite._band_shape``; every architecture)
    raises PyTorch's alert, naming the autograd backward and the
    deterministic autograd function, immediately before the launch of the
    fused backward, whose weight-gradient sum's order depends on scheduling;
    under ``warn_only=True`` it warns and runs it.  A node there whose weight
    needs no gradient (which runs the band's program without the weight
    gradient, ``nodw_module``), and a node in the first rows of every other band of a
    BF16 weight gradient on this device whose module repeats its bits
    (``suite._varies``: the small-row bands of ``BAND_ROLES``, the short
    backward on every architecture and the exact tiny backward on sm_100a
    and sm_107a, each summing on chip in a fixed order; a static or a claimed
    backward where a band runs one, adding its partials exactly), does not
    alert.  The launches are recorded and skipped (``run_module``
    stubbed)."""
    launched: list[str] = []
    monkeypatch.setattr(
        loader, "run_module", lambda name, *a, **k: launched.append(name)
    )
    arch = loader.device_arch(torch.cuda.current_device())
    q = f"{backend.__name__}.CakeRMSNormSiLUGateNondetFunction.backward"
    d = f"{deterministic_facade.__name__}.CakeRMSNormSiLUGateDetFunction"
    alerting = suite._band_shape("bwd_bf16_fused")
    # every architecture runs the fused backward in a band of its own
    assert alerting is not None, arch
    bands = suite._device_bands("bwd_bf16")
    quiet_bands = [
        band
        for band, _, _ in bands
        if not suite._varies(loader.route_module(arch, band))
    ]
    quiet = [suite._band_shape(band) for band in quiet_bands]
    assert None not in quiet, (arch, quiet_bands, quiet)
    assert any(band in loader.BAND_ROLES for band in quiet_bands), quiet_bands

    def node(shape, weight_grad=True):
        x, z, w, _ = _inputs(*shape, 21)
        y = cake_rmsnorm_silu_gate_nondet(
            x.requires_grad_(True),
            z.requires_grad_(True),
            w.requires_grad_(weight_grad),
            EPS,
        )
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
        return y

    def module(shape):
        return loader.module_for_rows(
            arch, "bwd_bf16", shape[0] * shape[1], _sm_count()
        )

    for band, shape in zip(quiet_bands, quiet, strict=True):
        assert module(shape) == loader.route_module(arch, band), (band, shape)
    assert module(alerting) == loader.route_module(arch, "bwd_bf16_fused")
    y = node(alerting)
    launched.clear()
    torch.use_deterministic_algorithms(True)
    with pytest.raises(RuntimeError) as info:
        y.backward(torch.ones_like(y))
    assert str(info.value) == (
        f"{q} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'. You can call {d} instead, "
        "or use the 'warn_only=True' option, if that's acceptable for your "
        "application."
    )
    assert launched == []
    torch.use_deterministic_algorithms(False)
    y = node(alerting)
    launched.clear()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        y.backward(torch.ones_like(y))
    assert [str(r.message) for r in record if backend.__name__ in str(r.message)] == [
        f"{q} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True, warn_only=True)'; the "
        f"non-deterministic kernels run. {d} is the deterministic one."
    ]
    assert launched == [module(alerting)]
    no_alert = [(alerting, False)] + [(shape, True) for shape in quiet]
    for shape, weight_grad in no_alert:
        torch.use_deterministic_algorithms(False)
        y = node(shape, weight_grad)
        launched.clear()
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y.backward(torch.ones_like(y))
        want = module(shape)
        if not weight_grad:
            # the band's program without the weight gradient (contiguous
            # inputs: packed rows, and a single token too where the build of
            # packed rows only has a cut)
            rows = shape[0] * shape[1]
            packed = suite._nodw_packed(*shape)
            want = loader.nodw_module(arch, want, rows, _sm_count(), packed=packed)
        assert launched == [want], (shape, weight_grad, launched)


@requires_device
@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_no_call_alerts_under_deterministic_algorithms(
    warn_only, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict or with
    ``warn_only=True``, every call in the bands whose outputs repeat their
    bits runs with no warning and no error: the forward and the backward
    into a BF16 and an FP32 weight gradient at the first rows of each
    ``BACKWARD_BANDS`` band and of each long role's packed rows of this
    device (``_band_modules``, ``_long_band_modules``; for a BF16
    weight gradient every band but the fused band, whose backward alerts:
    :func:`test_autograd_backward_alerts_once_deterministic_algorithms_are_enabled`),
    and the autograd backward, with the weight gradient, of
    :func:`cake_rmsnorm_silu_gate_nondet`, of
    :class:`CakeRMSNormSiLUGateNondetFunction` and of a non-reentrant
    checkpoint at the most rows of those BF16 calls.  The functional calls
    launch ``module_for_rows``' modules."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    calls = []
    varying = set()
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        bands = _band_modules(arch, role, sms) + _long_band_modules(arch, role, sms)
        for band, name, first, _ in bands:
            if suite._varies(name):
                varying.add(name)
                continue
            assert loader.module_for_rows(arch, role, first, sms) == name, (role, band)
            calls.append((first, dtype, name))
    # the fused band of this device (a BF16 weight gradient on every
    # architecture; an FP32 weight gradient runs none) alerts
    routed = {loader.ROUTES[arch][role] for role in suite.FUSED_ROLES[arch]}
    assert routed <= varying, (arch, varying, routed)
    autograd_rows = max(rows for rows, dtype, _ in calls if dtype == torch.bfloat16)
    inputs = {rows: suite.make(rows, 1, layout="flat", seed=21) for rows, _, _ in calls}
    x, z, w, dy = inputs[autograd_rows]
    leaves = [[t.clone().requires_grad_(True) for t in (x, z, w)] for _ in range(3)]
    launched = _spy_launches(monkeypatch)
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for rows, dtype, _ in calls:
            xr, zr, wr, dyr = inputs[rows]
            nd_forward(xr, zr, wr, EPS)
            nd_backward(
                dyr,
                xr,
                zr,
                wr,
                EPS,
                grad_weight=torch.empty(D, dtype=dtype, device=xr.device),
            )
        nd_gated_rmsnorm(*leaves[0], EPS).backward(dy)
        CakeRMSNormSiLUGateNondetFunction.apply(*leaves[1], EPS).backward(dy)
        torch.utils.checkpoint.checkpoint(
            nd_gated_rmsnorm, *leaves[2], EPS, use_reentrant=False
        ).backward(dy)
        torch.cuda.synchronize()
    assert [str(c.message) for c in caught] == []
    for leaf_set in leaves:
        assert all(t.grad is not None for t in leaf_set)
    want = []
    for rows, _, name in calls:
        want += [loader.module_for_rows(arch, "fwd", rows, sms), name]
    assert launched[: len(want)] == want, launched


@requires_device
def test_a_mode_keyword_is_a_type_error_on_the_device(
    determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a device too, ``deterministic=<any value>`` is a ``TypeError`` from
    every facade entry point before any launch or change of the workspace; so
    is a fifth input of the autograd function.  The same calls without the
    keyword run."""
    launched = _spy_launches(monkeypatch)
    T, H = 129, 48
    x, z, w, dy = _inputs(T, H, 11)
    device = x.device
    ws = nd_workspace(T * H, device)
    runs = {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS, workspace=ws, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(T * H, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: cake_rmsnorm_silu_gate_nondet(x, z, w, EPS, **m),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m),
    }  # fmt: skip
    assert set(runs) == set(ENTRY_NAMES)
    for run in runs.values():
        for value in (True, False):
            with pytest.raises(TypeError, match="deterministic"):
                run(deterministic=value)
    for value in (True, False):
        with pytest.raises(TypeError):
            CakeRMSNormSiLUGateNondetFunction.apply(x, z, w, EPS, value)
    assert launched == []
    torch.cuda.synchronize()
    assert not bool(ws.any())
    for run in runs.values():
        run()
    assert launched, "the entry points launched nothing"


@requires_device
def test_nondet_entry_points_run_this_package(
    determinism_setting, monkeypatch: pytest.MonkeyPatch, record_property
) -> None:
    """The ``nondet`` entry points launch the backend's forward and backward
    modules that ``module_for_rows`` picks: ``y``, ``dx``, ``dz`` and ``dw``
    repeat bit for bit (``dw`` where its band repeats it,
    ``suite.assert_dw_repeats``), and on the inputs of every draw of
    ``judge`` every call's outputs are judged against the naive
    implementation.  The two workspace kinds do not substitute, and a
    refusal launches nothing: this backward refuses an FP32 tensor (the
    deterministic package's workspace), naming both modes' workspaces, and
    the deterministic backward refuses this package's ``torch.uint8``
    workspace (``ValueError``)."""
    launched = _spy_launches(monkeypatch)
    T, H = 257, 48
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    modules = [
        loader.module_for_rows(arch, "fwd", T * H, sms),
        loader.module_for_rows(arch, "bwd_bf16", T * H, sms),
    ]

    def run(j, s):
        x, z, w, dy = _inputs(T, H, s)
        launched.clear()
        y = cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS)
        dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS)
        assert launched == modules, launched
        assert_same_bits("y", nd_forward(x, z, w, EPS), y)
        again = nd_backward(dy, x, z, w, EPS)
        for what, got, want in zip(("dx", "dz"), again[:2], (dx, dz), strict=True):
            assert_same_bits(what, got, want)
        suite.assert_dw_repeats(
            again[2], dw, varies=suite._varies(modules[1]), what="dw"
        )
        return silu_draw(
            x,
            z,
            w,
            dy,
            EPS,
            [{"y": y, "dx": dx, "dz": dz, "dw": dw}, {"dw": again[2]}],
            {"y": "y", "dx": "dx", "dz": "dz", "dw": "dw"},
        )

    judge(record_property, run, seed=12)
    x, z, w, dy = _inputs(T, H, 12)
    # an FP32 buffer (the deterministic package's workspace) is not a
    # workspace of the non-deterministic backward, nor the other way round
    launched.clear()
    fp32_workspace = torch.zeros(T * H, dtype=torch.float32, device=x.device)
    with pytest.raises(ValueError) as info:
        cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, workspace=fp32_workspace
        )
    _assert_refusal(info.value)
    # the deterministic backward refuses this package's workspace, naming both
    # modes' workspaces
    with pytest.raises(
        ValueError,
        match=re.escape(
            "workspace is uint8, a workspace of the non-deterministic mode "
            "(cake_pplx.cake_rmsnorm_silu_gate_nondet_train."
            "cake_rmsnorm_silu_gate_nondet_train_backward_workspace); the "
            "deterministic backward needs its own zero-filled FP32 workspace "
            "(cake_pplx.cake_rmsnorm_silu_gate_det_train."
            "cake_rmsnorm_silu_gate_det_train_backward_workspace)"
        ),
    ):
        det_backward(dy, x, z, w, EPS, workspace=nd_workspace(T * H, x.device))
    assert launched == []


@requires_device
@pytest.mark.parametrize("shape", [(0, D), (0, 48, D)], ids=["0x128", "0x48x128"])
def test_backward_with_no_rows_refuses_the_deterministic_workspace(
    shape: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call with no rows refuses an FP32 tensor (the deterministic package's
    workspace) with the ``ValueError`` of a call with rows, naming both modes'
    workspaces, before it returns (``grad_weight`` keeps its value).  With
    this package's ``torch.uint8`` workspace it returns zero-sized ``dx`` and
    ``dz`` and a zero ``dw``, or keeps ``grad_weight`` when accumulating;
    nothing is launched and the workspace stays zero."""
    launched = _spy_launches(monkeypatch)
    empty = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    x, z, w, dy = _inputs(65, 48, 14)
    fp32_workspace = torch.zeros(65 * 48, dtype=torch.float32, device=x.device)
    ws = nd_workspace(0, x.device)
    assert ws.dtype == torch.uint8
    assert ws.numel() == nd_workspace_bytes(0, x.device)
    for module in (facade, backend):
        backward = module.cake_rmsnorm_silu_gate_nondet_train_backward
        with pytest.raises(ValueError) as with_rows:
            backward(dy, x, z, w, EPS, workspace=fp32_workspace)
        _assert_refusal(with_rows.value)
        main = torch.full((D,), 1.5, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError) as no_rows:
            backward(
                empty,
                empty,
                empty,
                w,
                EPS,
                grad_weight=main,
                workspace=fp32_workspace,
            )
        assert str(no_rows.value) == str(with_rows.value), str(no_rows.value)
        assert torch.equal(main, torch.full_like(main, 1.5))
        dx, dz, dw = backward(empty, empty, empty, w, EPS, workspace=ws)
        assert dx.shape == dz.shape == empty.shape and dx.numel() == dz.numel() == 0
        assert dw.dtype == torch.bfloat16 and torch.equal(dw, torch.zeros_like(dw))
        _, _, dw = backward(
            empty,
            empty,
            empty,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        assert dw is main and torch.equal(main, torch.full_like(main, 1.5))
        _, _, dw = backward(empty, empty, empty, w, EPS, grad_weight=main, workspace=ws)
        assert dw is main and torch.equal(main, torch.zeros_like(main))
    torch.cuda.synchronize()
    assert launched == []
    assert not bool(ws.any())


# --------------------------------------------------------------------------- judged against the PyTorch BF16 naive implementation


@requires_device
@pytest.mark.parametrize(
    "band",
    [
        None,
        "bwd_bf16_fused",
        "bwd_bf16_rep",
        "bwd_bf16_rep_long",
        "bwd_f32",
        "bwd_f32_rep",
        "bwd_f32_rep_long",
    ],
    ids=[
        "1031x48",
        "bwd_bf16_fused",
        "bwd_bf16_rep",
        "bwd_bf16_rep_long",
        "bwd_f32",
        "bwd_f32_rep",
        "bwd_f32_rep_long",
    ],
)
def test_accumulation_meets_the_naive_tier(
    band, determinism_setting, record_property
) -> None:
    """Two microbatches of ``T x H`` rows (1031 tokens of 48 heads; for
    ``band`` the fewest whole tokens of 48 heads in that band, of a BF16
    weight gradient the fused, the repeatable or the long role's packed
    rows, of an FP32 one the static (sm_100a and sm_103a; skipped elsewhere),
    the repeatable or the long role's packed rows, ``suite._band_shape``, on
    every architecture; the weight gradient of the other dtype runs its own
    band there) accumulated into an FP32 and a BF16
    buffer starting at 0.25: the same starting buffer gives the same bits (a
    BF16 buffer where its band repeats a BF16 ``dw``).  Autograd accumulation into
    ``.grad`` over two microbatches gives the sum of the two nodes' weight
    gradients (recorded by a hook), each the functional backward's bits
    where its band repeats them (``suite.assert_dw_repeats``), and twice the
    functional backward's ``dx`` and ``dz`` bits.  On the inputs of every
    draw of ``judge``, every accumulator (against the naive implementation's
    BF16 sum of the starting value and the two gradients), every node's
    weight gradient and ``x.grad`` and ``z.grad`` (against the naive
    implementation's BF16 sums of two gradients) are judged."""
    shape = (1031, 48) if band is None else suite._band_shape(band)
    if shape is None:
        pytest.skip(f"this device routes no {band} band")
    T, H = shape
    rows = T * H

    def run(j, s):
        x, z, w, dy = suite.make(T, H, seed=s)
        x2, z2, _, dy2 = suite.make(T, H, seed=s + 1)
        launches = []
        for dtype in (torch.float32, torch.bfloat16):
            name = f"accumulated {str(dtype).removeprefix('torch.')} dw"
            varies = suite._dw_varies(rows, dtype)
            mains = []
            for _ in range(2):
                main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
                for args in ((dy, x, z), (dy2, x2, z2)):
                    nd_backward(
                        *args, w, EPS, grad_weight=main, accumulate_grad_weight=True
                    )
                mains.append(main)
            if not varies:
                assert_same_bits(name, mains[1], mains[0])
            launches += [{name: main} for main in (mains if varies else mains[:1])]
        # autograd accumulation into .grad over two microbatches
        xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
        node_dws: list[torch.Tensor] = []
        wa.register_hook(lambda grad: node_dws.append(grad.detach().clone()))
        nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        nd_gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        assert len(node_dws) == 2, len(node_dws)
        dx, dz, dw = nd_backward(dy, x, z, w, EPS)
        assert_same_bits("x.grad", xa.grad, dx + dx)
        assert_same_bits("z.grad", za.grad, dz + dz)
        assert_same_bits("w.grad", wa.grad, node_dws[0] + node_dws[1])
        for k, node_dw in enumerate(node_dws):
            suite.assert_dw_repeats(
                node_dw, dw, varies=suite._dw_varies(rows), what=f"node {k}: dw"
            )
        launches += [{"node dw": node_dw} for node_dw in node_dws]
        launches.append({"x.grad": xa.grad, "z.grad": za.grad})
        first = silu_draw(x, z, w, dy, EPS, [], {"dx": "dx", "dz": "dz", "dw": "dw"})
        second = silu_draw(x2, z2, w, dy2, EPS, [], {"dw": "dw"})
        both = 0.25 + first.expected["dw"] + second.expected["dw"]
        expected = {
            "accumulated float32 dw": both,
            "accumulated bfloat16 dw": both,
            "node dw": first.expected["dw"],
            "x.grad": 2 * first.expected["dx"],
            "z.grad": 2 * first.expected["dz"],
        }

        def naive_values():
            one = first.naive()
            accumulated = (
                torch.full_like(one["dw"], 0.25) + one["dw"]
            ) + second.naive()["dw"]
            return {
                "accumulated float32 dw": accumulated,
                "accumulated bfloat16 dw": accumulated,
                "node dw": one["dw"],
                "x.grad": one["dx"] + one["dx"],
                "z.grad": one["dz"] + one["dz"],
            }

        return Draw(launches, expected, naive_values)

    judge(record_property, run, seed=6)


@requires_device
@pytest.mark.parametrize(
    "band",
    [None, "bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long"],
    ids=["257x48", "bwd_bf16_fused", "bwd_bf16_rep", "bwd_bf16_rep_long"],
)
def test_compute_grad_weight_false_meets_the_naive_tier(
    band, determinism_setting, record_property
) -> None:
    """Without the weight gradient (with this package's workspace or none,
    and through autograd with a weight that needs no gradient), ``dx`` and
    ``dz`` are the bits of a backward with the weight gradient (of the
    repeatable backward where the build of packed rows only runs the rows:
    packed rows up to its cut on sm_100a and sm_103a, ``suite._nodw_bits``),
    at 257 tokens
    of 48 heads and, for ``band``, at the fewest whole tokens of 48 heads in
    that BF16 band, the fused, the repeatable or the long role's packed rows
    (``suite._band_shape``; a call without the weight gradient takes the
    BF16 bands, so no band of an FP32 weight gradient is a case; every
    architecture); on the inputs of every draw of ``judge`` they are judged
    against the naive implementation."""
    shape = (257, 48) if band is None else suite._band_shape(band)
    if shape is None:
        pytest.skip(f"this device routes no {band} band")
    T, H = shape
    rows = T * H
    nodw = suite._nodw_module(
        suite._backward_module(rows), rows, packed=suite._nodw_packed(T, H)
    )

    def run(j, s):
        x, z, w, dy = suite.make(T, H, seed=s)
        want_dx, want_dz = suite._nodw_bits(nodw, dy, x, z, w)
        ws = nd_workspace(T * H, x.device)
        launches = [{"dx": want_dx, "dz": want_dz}]
        for workspace in (ws, None):
            dx, dz, dw = nd_backward(
                dy, x, z, w, EPS, compute_grad_weight=False, workspace=workspace
            )
            assert dw is None
            assert_same_bits("dx", dx, want_dx)
            assert_same_bits("dz", dz, want_dz)
            launches.append({"dx": dx, "dz": dz})
        # autograd with a weight that needs no gradient
        xa, za = (t.clone().requires_grad_(True) for t in (x, z))
        nd_gated_rmsnorm(xa, za, w, EPS).backward(dy)
        assert_same_bits("x.grad", xa.grad, want_dx)
        assert_same_bits("z.grad", za.grad, want_dz)
        launches.append({"dx": xa.grad, "dz": za.grad})
        return silu_draw(x, z, w, dy, EPS, launches, {"dx": "dx", "dz": "dz"})

    judge(record_property, run, seed=12)


@requires_device
def test_reset_workspace_then_reuse_across_shapes(
    determinism_setting, record_property
) -> None:
    """A workspace of the architecture's bytes (``suite.WORKSPACE_BYTES``)
    with nonzero bytes in every buffer (the integer windows, the FP32
    replicas of the column sums, the class words, the arrival ticket and the
    claim counter) zeroed by
    ``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``, then reused
    without a reset over the backend suite's sequence of shapes, then the
    first rows of each ``BACKWARD_BANDS`` band and of each long role's packed
    rows of this device none of them runs (``suite._range_shape``; so every
    banded backward program of the architecture among them, the fused
    backward and the repeatable backward's long role too), weight-gradient
    dtypes and accumulation: every output repeats bit for bit (an
    accumulated ``dw`` from the same starting value too; a BF16 ``dw`` in the
    fused and the row-claims bands is judged on each call instead), and the
    workspace is zero after every call.  The bytes, the reset and the
    sequence run on the inputs of every draw of ``judge``, every call's
    ``dx``, ``dz`` and ``dw`` judged against the naive implementation."""
    device = torch.device("cuda")
    ws = nd_workspace(1, device)
    assert ws.dtype == torch.uint8
    assert ws.numel() == nd_workspace_bytes(1, device)
    assert nd_workspace_bytes(1, device) == nd_workspace_bytes(10**7, device)
    arch = loader.device_arch(torch.cuda.current_device())
    assert ws.numel() == suite.WORKSPACE_BYTES[arch], (arch, ws.numel())
    cases = [
        (1031, 48, torch.bfloat16, False),
        (7, 48, torch.float32, False),
        (4097, 16, torch.bfloat16, False),
        (1, 1, torch.float32, True),
        (30193, 48, torch.bfloat16, False),
        (33, 7, torch.float32, True),
        (129, 48, torch.float32, False),
    ]
    sms = _sm_count()
    ran = {
        (_role_of(dtype), _band_of(arch, _role_of(dtype), T * H, sms))
        for T, H, dtype, _ in cases
    }
    for dtype in (torch.bfloat16, torch.float32):
        role = _role_of(dtype)
        bands = _band_modules(arch, role, sms) + _long_band_modules(arch, role, sms)
        for band, _, first, last in bands:
            if (role, (band, first)) not in ran:
                T, H = suite._range_shape(first, last)
                cases.append((T, H, dtype, False))

    def run(j, s):
        generator = torch.Generator(device="cuda").manual_seed(s)
        ws.copy_(
            torch.randint(
                1, 256, ws.shape, dtype=torch.uint8, device=device, generator=generator
            )
        )
        assert bool(ws.ne(0).all())
        nd_reset(ws)
        torch.cuda.synchronize()
        assert not bool(ws.any())
        draws = []
        for i, (T, H, dtype, accumulate) in enumerate(cases):
            x, z, w, dy = suite.make(T, H, seed=s + 1 + i)
            start = 0.5 if accumulate else 0.0
            case = f"T={T} H={H} {dtype}"
            varies = suite._dw_varies(T * H, dtype)

            def gw(start=start, dtype=dtype, accumulate=accumulate):
                if accumulate or dtype != torch.bfloat16:
                    return torch.full((D,), start, dtype=dtype, device=device)
                return None

            runs = [
                nd_backward(
                    dy,
                    x,
                    z,
                    w,
                    EPS,
                    grad_weight=gw(),
                    accumulate_grad_weight=accumulate,
                    workspace=ws,
                )
                for _ in range(2)
            ]
            torch.cuda.synchronize()
            assert not bool(ws.any()), case
            for what, first, again in zip(("dx", "dz", "dw"), *runs, strict=True):
                if what != "dw" or not varies:
                    assert_same_bits(f"{what} {case}", again, first)
            assert runs[0][2].dtype == dtype
            label = f"{i} {T}x{H} {str(dtype).removeprefix('torch.')} "
            launches = [
                {label + "dx": dx, label + "dz": dz, label + "dw": dw}
                for dx, dz, dw in (runs if varies else runs[:1])
            ]
            draws.append(
                silu_draw(
                    x,
                    z,
                    w,
                    dy,
                    EPS,
                    launches,
                    {label + "dx": "dx", label + "dz": "dz", label + "dw": "dw"},
                    starts={label + "dw": torch.full((D,), start, dtype=dtype)},
                )
            )
        return suite.merge_draws(draws)

    judge(record_property, run, seed=70)


@requires_device
def test_two_streams_repeat_the_current_stream_bits(
    determinism_setting, record_property
) -> None:
    """Backwards on two streams, each with its own workspace, five times over:
    every ``dx``, ``dz`` and ``dw`` is the bits of the same backward on the
    current stream (``dw`` where its band repeats it,
    ``suite.assert_dw_repeats``); on the inputs of every draw of ``judge``
    every call's ``dx``, ``dz`` and ``dw`` are judged against the naive
    implementation."""
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [nd_workspace(1, torch.device("cuda")) for _ in streams]

    def run(j, s):
        cases = [suite.make(4097, 48, seed=s), suite.make(1031, 16, seed=s + 1)]
        refs = [nd_backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
        outputs: list[list] = [[], []]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for _ in range(5):
            for i, stream in enumerate(streams):
                x, z, w, dy = cases[i]
                with torch.cuda.stream(stream):
                    outputs[i].append(
                        nd_backward(dy, x, z, w, EPS, workspace=workspaces[i])
                    )
        torch.cuda.synchronize()
        draws = []
        for i in range(2):
            varies = suite._dw_varies(cases[i][0].numel() // D)
            for got in outputs[i]:
                for what, g, want in zip(
                    ("dx", "dz"), got[:2], refs[i][:2], strict=True
                ):
                    assert_same_bits(f"stream {i}: {what}", g, want)
                suite.assert_dw_repeats(
                    got[2], refs[i][2], varies=varies, what=f"stream {i}: dw"
                )
            label = f"stream {i}: "
            names = [label + what for what in ("dx", "dz", "dw")]
            draws.append(
                silu_draw(
                    *cases[i],
                    EPS,
                    [
                        dict(zip(names, got, strict=True))
                        for got in [refs[i], *outputs[i]]
                    ],
                    dict(zip(names, ("dx", "dz", "dw"), strict=True)),
                )
            )
        return suite.merge_draws(draws)

    judge(record_property, run, seed=30)


@requires_device
def test_side_stream_repeats_the_current_stream_bits(
    determinism_setting, record_property
) -> None:
    """The forward and the backward (with this package's workspace and
    without one) on a side stream give the bits of the same calls on the
    current stream (``dw`` where its band repeats it,
    ``suite.assert_dw_repeats``); on the inputs of every draw of ``judge``
    every call's outputs are judged against the naive implementation."""

    def run(j, s):
        x, z, w, dy = suite.make(2049, 48, seed=s)
        want_y = nd_forward(x, z, w, EPS)
        want = nd_backward(dy, x, z, w, EPS)
        ws = nd_workspace(x.shape[0] * x.shape[1], x.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            y = nd_forward(x, z, w, EPS)
            got = nd_backward(dy, x, z, w, EPS, workspace=ws)
            fresh = nd_backward(dy, x, z, w, EPS)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert_same_bits("y", y, want_y)
        varies = suite._dw_varies(x.numel() // D)
        for outputs in (got, fresh):
            for what, g, w_ in zip(("dx", "dz"), outputs[:2], want[:2], strict=True):
                assert_same_bits(what, g, w_)
            suite.assert_dw_repeats(outputs[2], want[2], varies=varies, what="dw")
        names = ("dx", "dz", "dw")
        return silu_draw(
            x,
            z,
            w,
            dy,
            EPS,
            [{"y": want_y}, {"y": y}]
            + [dict(zip(names, out, strict=True)) for out in (want, got, fresh)],
            {"y": "y", "dx": "dx", "dz": "dz", "dw": "dw"},
        )

    judge(record_property, run, seed=9)


@requires_device
def test_cuda_graph_replay_repeats_the_eager_bits(
    determinism_setting, record_property
) -> None:
    """Replays of a captured forward and backward (this package's workspace,
    a BF16 ``dw``), one on the inputs of each draw of ``judge`` (copied into
    the captured inputs), give the bits of eager calls on those inputs
    (``dw`` where its band repeats it, ``suite.assert_dw_repeats``); every
    replay's outputs are judged against the naive implementation."""
    T, H = 300, 48
    x, z, w, dy = suite.make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    ws = nd_workspace(T * H, x.device)

    def step():
        nd_forward(xs, zs, w, EPS, out=y)
        nd_backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    names = ("y", "dx", "dz", "dw")
    varies = suite._dw_varies(T * H)

    def run(k, s):
        xn, zn, _, dyn = suite.make(T, H, seed=s)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        graph.replay()
        torch.cuda.synchronize()
        eager = (nd_forward(xn, zn, w, EPS), *nd_backward(dyn, xn, zn, w, EPS))
        for what, got, want in zip(names[:3], (y, dx, dz), eager[:3], strict=True):
            assert_same_bits(f"replay {k}: {what}", got, want)
        suite.assert_dw_repeats(dw, eager[3], varies=varies, what=f"replay {k}: dw")
        replayed = (y.clone(), dx.clone(), dz.clone(), dw.clone())
        return silu_draw(
            xn,
            zn,
            w,
            dyn,
            EPS,
            [dict(zip(names, replayed, strict=True)), {"dw": eager[3]}],
            dict(zip(names, names, strict=True)),
        )

    judge(record_property, run, seed=40)


@requires_device
def test_autograd_and_checkpoint_repeat_the_functional_bits(
    determinism_setting, record_property
) -> None:
    """Autograd through :func:`cake_rmsnorm_silu_gate_nondet`, through
    :class:`CakeRMSNormSiLUGateNondetFunction` and through a non-reentrant
    checkpoint gives the functional entry points' bits (``y``, ``dx``, ``dz``
    and a BF16 ``dw``; ``dw`` where its band repeats it,
    ``suite.assert_dw_repeats``); on the inputs of every draw of ``judge``
    every path's outputs are judged against the naive implementation."""
    names = ("y", "dx", "dz", "dw")
    varies = suite._dw_varies(513 * 48)

    def run(j, s):
        x, z, w, dy = suite.make(513, 48, seed=s)
        want = (nd_forward(x, z, w, EPS), *nd_backward(dy, x, z, w, EPS))
        launches = [dict(zip(names, want, strict=True))]

        def check(y, xg, zg, wg, what):
            assert wg.dtype == torch.bfloat16, what
            for name, got, w_ in zip(names[:3], (y, xg, zg), want[:3], strict=True):
                assert_same_bits(f"{what}: {name}", got, w_)
            suite.assert_dw_repeats(wg, want[3], varies=varies, what=f"{what}: dw")
            launches.append(dict(zip(names, (y.detach(), xg, zg, wg), strict=True)))

        xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
        y = nd_gated_rmsnorm(xa, za, wa, EPS)
        assert y.grad_fn._forward_cls is backend.CakeRMSNormSiLUGateNondetFunction
        y.backward(dy)
        check(y, xa.grad, za.grad, wa.grad, "autograd")
        xf, zf, wf = (t.clone().requires_grad_(True) for t in (x, z, w))
        yf = CakeRMSNormSiLUGateNondetFunction.apply(xf, zf, wf, EPS)
        yf.backward(dy)
        check(yf, xf.grad, zf.grad, wf.grad, "autograd function")
        xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
        yc = torch.utils.checkpoint.checkpoint(
            nd_gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
        )
        yc.backward(dy)
        check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")
        return silu_draw(
            x, z, w, dy, EPS, launches, dict(zip(names, names, strict=True))
        )

    judge(record_property, run, seed=8)


# --------------------------------------------------------------------------- the forward's bands and the backward's programs


def _spy_launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def _sm_count() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _role_of(dtype: torch.dtype) -> str:
    """The backward role of a ``dtype`` weight gradient: ``bwd_f32`` for
    FP32, ``bwd_bf16`` for BF16."""
    return "bwd_f32" if dtype == torch.float32 else "bwd_bf16"


def _band_role_dtype(band: str) -> torch.dtype:
    """The weight-gradient dtype of the backward band role ``band``
    (``bwd_f32...`` FP32, ``bwd_bf16...`` BF16)."""
    return torch.float32 if band.startswith("bwd_f32") else torch.bfloat16


def _band_modules(
    arch: str, role: str, sms: int
) -> list[tuple[str, str, int, int | None]]:
    """``(band role, module, first rows, last rows or None)`` of each
    ``loader.BACKWARD_BANDS`` band of the backward ``role`` (``bwd_bf16`` or
    ``bwd_f32``) on ``arch`` with ``sms`` SMs that runs at least one row
    count, in order (``suite._backward_bands``): the band's routed module
    over the rows past the ends of the bands before it up to its own end
    (``suite._band_end``); the last band has no last row.  A band role may
    hold more than one band (on sm_107a the BF16 repeatable role runs on both
    sides of the fused band, on sm_100a and sm_103a the FP32 one on both sides
    of the static band)."""
    return [
        (band, loader.route_module(arch, band), first, last)
        for band, first, last in suite._backward_bands(arch, role, sms)
    ]


def _long_band_modules(
    arch: str, role: str, sms: int
) -> list[tuple[str, str, int, int | None]]:
    """``(long role, module, first rows, last rows or None)`` of each long
    role of the backward ``role`` on ``arch`` with ``sms`` SMs
    (``suite._long_bands``): the routed module of the long role over the
    packed rows of its band from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM
    on."""
    return [
        (band, loader.route_module(arch, band), first, last)
        for band, first, last in suite._long_bands(arch, role, sms)
    ]


def _band_occurrence(
    arch: str, role: str, sms: int, band: str, nth: int = 1
) -> tuple[str, int, int | None] | None:
    """``(module, first rows, last rows or None)`` of the ``nth`` band of the
    band role ``band`` among :func:`_band_modules`, or ``None`` where the
    role has fewer such bands on ``arch`` with ``sms`` SMs."""
    found = [
        (module, first, last)
        for name, module, first, last in _band_modules(arch, role, sms)
        if name == band
    ]
    return found[nth - 1] if len(found) >= nth else None


def _band_of(arch: str, role: str, rows: int, sms: int) -> tuple[str, int]:
    """``(band role, first rows)`` of the band that runs ``rows`` packed rows
    of the backward ``role`` on ``arch`` with ``sms`` SMs: a long role's
    (:func:`_long_band_modules`) where it holds them, else the
    ``loader.BACKWARD_BANDS`` band (:func:`_band_modules`) that holds them."""
    bands = _long_band_modules(arch, role, sms) + _band_modules(arch, role, sms)
    for band, _, first, last in bands:
        if first <= rows and (last is None or rows <= last):
            return band, first
    raise AssertionError(f"no {role} band of {arch} holds {rows} rows")


def _start_rows(arch: str, name: str, sms: int, packed: bool = True) -> int:
    """The fewest rows the routed module ``name`` runs for on ``arch`` with
    ``sms`` SMs: for a forward module the first row of its first
    ``FORWARD_BANDS`` band (``STRIDED_FORWARD_BANDS``, the bands of rows of
    no class of their own, where the rows are not ``packed``; one more than
    the previous band's ``last * SMs``), for a
    backward module the first row of its ``BACKWARD_BANDS`` band
    (:func:`_band_modules`), for a long role's module the first of its
    packed rows (:func:`_long_band_modules`)."""
    role = loader.MODULES[name]["role"]
    if role.startswith("bwd"):
        call_role = "bwd_f32" if role.startswith("bwd_f32") else "bwd_bf16"
        bands = _band_modules(arch, call_role, sms)
        for _, module, first, _ in bands + _long_band_modules(arch, call_role, sms):
            if module == name:
                return first
        raise AssertionError(f"{arch} has no backward band of {role}")
    first = 1
    table = loader.FORWARD_BANDS if packed else loader.STRIDED_FORWARD_BANDS
    for last, band in table[arch]:
        if band == role:
            return first
        assert last is not None, (arch, role)
        first = int(last) * sms + 1
    raise AssertionError(f"{arch} has no forward band of {role}")


def _forward_class(x: torch.Tensor, z: torch.Tensor) -> tuple[bool, str | None]:
    """``(packed, layout)`` of a forward of ``x`` and ``z`` as the host routes
    it, from the strides the kernels get (``backend._kernel_view``: a single
    token's passed as 0, a single head's as 128): packed rows (both head
    strides 128, both token strides 0 or ``H * 128``) have no class, other
    rows the class of ``loader.forward_layout``."""
    shape = x.shape
    tokens, heads = backend._rows_shape(shape)
    _, x_tok, x_head = backend._kernel_view(x, "x", shape=shape, device=x.device)
    _, z_tok, z_head = backend._kernel_view(z, "z", shape=shape, device=z.device)
    row = heads * D
    if x_head == D and z_head == D and x_tok in (0, row) and z_tok in (0, row):
        return True, None
    return False, loader.forward_layout(tokens, heads, x_tok, x_head, z_tok, z_head)


def _forward_table(arch: str, rows: int, packed: bool, layout: str | None):
    """The forward bands of ``arch`` that run ``rows`` rows of ``packed`` rows
    or of the class ``layout`` (:func:`_forward_class`): ``FORWARD_BANDS`` for
    packed rows fewer than ``PACKED_ROWS``, else the class's table
    (``LAYOUT_FORWARD_BANDS``) where the architecture has one, else
    ``STRIDED_FORWARD_BANDS``."""
    if packed and rows < loader.PACKED_ROWS:
        return loader.FORWARD_BANDS[arch]
    by_arch = loader.LAYOUT_FORWARD_BANDS.get(layout, {})
    return by_arch.get(arch) or loader.STRIDED_FORWARD_BANDS[arch]


def _made_class(tokens: int, heads: int, layout: str) -> tuple[bool, str | None]:
    """``(packed, layout)`` (:func:`_forward_class`) of the inputs of
    ``suite.make(tokens, heads, layout=layout)`` by their layout
    (``suite._packed``, ``suite._layout_class``)."""
    return (
        suite._packed(tokens, heads, layout),
        suite._layout_class(tokens, heads, layout),
    )


# The band kinds of ``FORWARD_CASES``: ``(heads, layout)`` of their inputs
# (``suite.make``'s layout; ``None`` heads: one token of the rows as heads)
# and ``(packed, layout)`` of their rows, whose table (:func:`_forward_table`)
# holds the band; ``other``: ``x`` and ``z`` both sliced, whole-token rows
# (``SLICED_FORWARD_BANDS`` where the architecture has it, else
# ``STRIDED_FORWARD_BANDS``).
FORWARD_KINDS = {
    "flat": (1, "flat", True, None),
    "token": (None, "contiguous", True, None),
    "projection": (4, "projection", False, loader.PROJECTION_LAYOUT),
    "headmajor": (4, "head_major", False, loader.HEAD_MAJOR_LAYOUT),
    "other": (1, "both_sliced", False, loader.SLICED_LAYOUT),
}


def _v0218_moved(
    arch: str, rows: int, sm_count: int, packed: bool, layout: str | None
) -> tuple[str, str] | None:
    """``(0.2.17 role, 0.2.18 role)`` of the forward of ``rows`` rows on
    ``arch`` with ``sm_count`` SMs where version 0.2.18 moved its module
    (``suite.V0218_MOVED_BANDS``: by the class ``layout`` of rows that are not
    ``packed``, no class for packed rows of ``PACKED_ROWS`` or more), else
    None."""
    if packed and rows < loader.PACKED_ROWS:
        return None
    band = suite.V0218_MOVED_BANDS.get((arch, None if packed else layout))
    if band is None:
        return None
    lo, hi, old, new = band
    return (old, new) if lo * sm_count < rows <= hi * sm_count else None


# SM counts of the comparison of the forward's plan with version 0.2.17's.
PLAN_SM_COUNTS = (1, 2, 7, 148, 152, 212, 216)


def _put_back_0_2_17_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put version 0.2.17's bands back in the registry's tables
    (``suite.V0217_FORWARD_BANDS``: an entry ``None`` is removed, so whole-token
    rows run ``STRIDED_FORWARD_BANDS`` as in 0.2.17), restored by
    ``monkeypatch``; each entry differs from the shipped one."""
    for table, by_arch in suite.V0217_FORWARD_BANDS.items():
        current = getattr(loader, table)
        assert set(by_arch) <= set(current), table
        for arch, bands in by_arch.items():
            assert bands != current[arch], (table, arch)
            if bands is None:
                monkeypatch.delitem(current, arch)
            else:
                monkeypatch.setitem(current, arch, bands)


@pytest.mark.parametrize("sm_count", PLAN_SM_COUNTS)
def test_forward_plan_differs_from_0_2_17_only_in_the_moved_bands(
    sm_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host plan only, no device: the registry's forward tables are the
    suite's mirrors, and ``module_for_rows`` of the forward on every
    architecture, of packed rows and of rows that are not packed of every
    class (none, ``PROJECTION_LAYOUT``, ``HEAD_MAJOR_LAYOUT``,
    ``SLICED_LAYOUT``), at every whole rows per SM up to 3200, at every cut of
    either version's tables and a row either side and at ``PACKED_ROWS`` and
    ``INT32_MAX``, gives the module it gives with version 0.2.17's tables put
    back (:func:`_put_back_0_2_17_tables`: no ``SLICED_FORWARD_BANDS`` entry)
    but in the bands of ``suite.V0218_MOVED_BANDS``, where it gives
    ``fwd_layout_pipe`` in the place of 0.2.17's module: on sm_100a and
    sm_103a a gate of more than one head sliced from a projection past 599
    and 629 rows per SM up to 650 and 699 (0.2.17's extra-long forward), and
    whole-token rows from 701 to 1500 rows per SM (0.2.17's plain forward).
    Every moved band holds rows of the comparison; packed rows, rows of no
    class, head-major rows, sm_107a and every other band run what 0.2.17
    ran: rows of no class and head-major rows on sm_100a and sm_103a from 701
    to 1500 rows per SM the plain forward, beside whole-token rows on
    ``fwd_layout_pipe``."""
    assert loader.PROJECTION_FORWARD_BANDS == suite.PROJECTION_FORWARD_BANDS
    assert loader.STRIDED_FORWARD_BANDS == suite.STRIDED_FORWARD_BANDS
    assert loader.HEAD_MAJOR_FORWARD_BANDS == suite.HEAD_MAJOR_FORWARD_BANDS
    assert loader.SLICED_FORWARD_BANDS == suite.SLICED_FORWARD_BANDS
    assert loader.LAYOUT_FORWARD_BANDS[loader.PROJECTION_LAYOUT] is (
        loader.PROJECTION_FORWARD_BANDS
    )
    assert loader.LAYOUT_FORWARD_BANDS[loader.HEAD_MAJOR_LAYOUT] is (
        loader.HEAD_MAJOR_FORWARD_BANDS
    )
    assert loader.LAYOUT_FORWARD_BANDS[loader.SLICED_LAYOUT] is (
        loader.SLICED_FORWARD_BANDS
    )
    rows_by_arch = {}
    for arch in loader.ARCHES:
        rows = {k * sm_count for k in range(1, 3201)}
        rows |= {1, loader.PACKED_ROWS - 1, loader.PACKED_ROWS}
        rows |= {loader.PACKED_ROWS + 1, loader.INT32_MAX}
        tables = [loader.FORWARD_BANDS[arch], loader.STRIDED_FORWARD_BANDS[arch]]
        tables += [t[arch] for t in loader.LAYOUT_FORWARD_BANDS.values() if arch in t]
        tables += [t[arch] for t in suite.V0217_FORWARD_BANDS.values() if t.get(arch)]
        for bands in tables:
            for last, _ in bands[:-1]:
                rows |= {last * sm_count + d for d in (-1, 0, 1)}
        rows_by_arch[arch] = sorted(r for r in rows if r >= 1)

    def plan() -> dict:
        got = {}
        for arch, all_rows in rows_by_arch.items():
            for rows in all_rows:
                for packed in (True, False):
                    for layout in (None, *loader.LAYOUT_FORWARD_BANDS):
                        got[arch, rows, packed, layout] = loader.module_for_rows(
                            arch, "fwd", rows, sm_count, packed=packed, layout=layout
                        )
        return got

    shipped = plan()
    _put_back_0_2_17_tables(monkeypatch)
    old = plan()
    assert shipped.keys() == old.keys()
    moved = set()
    for key, name in shipped.items():
        arch, rows, packed, layout = key
        want = _v0218_moved(arch, rows, sm_count, packed, layout)
        if want is None:
            assert name == old[key], key
            continue
        routes = loader.ROUTES[arch]
        assert (old[key], name) == (routes[want[0]], routes[want[1]]), key
        moved.add((arch, None if packed else layout))
    assert moved == set(suite.V0218_MOVED_BANDS), moved
    # rows of no class and head-major rows where whole-token rows moved keep
    # the plain forward
    kept = 0
    for (arch, rows, packed, layout), name in shipped.items():
        if packed or layout not in (None, loader.HEAD_MAJOR_LAYOUT):
            continue
        if _v0218_moved(arch, rows, sm_count, False, loader.SLICED_LAYOUT) is None:
            continue
        routes = loader.ROUTES[arch]
        assert name == old[arch, rows, packed, layout] == routes["fwd"], arch
        sliced = shipped[arch, rows, packed, loader.SLICED_LAYOUT]
        assert sliced == routes["fwd_layout_pipe"], (arch, rows)
        kept += 1
    assert kept, sm_count


def _forward_layout_0_2_17(
    tokens: int, heads: int, x_tok: int, x_head: int, z_tok: int, z_head: int
) -> str | None:
    """Version 0.2.17's ``forward_layout``: the projection class (more than
    one head, packed ``x``, ``z`` of head stride 128 and a token stride past
    ``H * 128``), the head-major class, else None (no whole-token class)."""
    row = heads * D
    if heads > 1 and x_head == z_head == D and x_tok == row and z_tok > row:
        return loader.PROJECTION_LAYOUT
    plane = tokens * D
    if x_tok == z_tok == D and x_head == z_head == plane:
        return loader.HEAD_MAJOR_LAYOUT
    return None


def _form_strides(form: str, tokens: int, heads: int) -> tuple[int, int, int, int]:
    """``(x_tok, x_head, z_tok, z_head)`` the host passes for the inputs of
    ``form`` of ``tokens`` tokens (more than one) and ``heads`` heads (a single
    head's stride passed as 128): ``projection`` (packed ``x``, ``z`` the gate
    columns of a projection ``2 * H * 128 + 16`` wide), ``projw4`` (``z`` of
    one ``4 * H * 128`` wide), ``both_sliced`` (``x`` and ``z`` both sliced
    from such projections), ``padded`` (``x`` and ``z`` the first ``H * 128``
    columns of ``[T, H * 128 + 16]``), ``x_sliced`` (``x`` sliced, ``z``
    packed), ``head_major``, ``padded_heads`` (head stride 144),
    ``padded_head_major`` (head stride ``2 * T * 128``), ``mixed``
    (head-major ``x``, ``z`` of a projection), ``x_broadcast_heads``,
    ``x_broadcast_projection`` (``x`` broadcast over tokens, ``z`` of a
    projection) and ``z_broadcast_x_sliced``."""
    hd = heads * D
    wide = 2 * hd + 16
    x, z = {
        "projection": ((hd, D), (wide, D)),
        "projw4": ((hd, D), (4 * hd, D)),
        "both_sliced": ((wide, D), (wide, D)),
        "padded": ((hd + 16, D), (hd + 16, D)),
        "x_sliced": ((wide, D), (hd, D)),
        "head_major": ((D, tokens * D), (D, tokens * D)),
        "padded_heads": ((heads * (D + 16), D + 16), (heads * (D + 16), D + 16)),
        "padded_head_major": ((D, 2 * tokens * D), (D, 2 * tokens * D)),
        "mixed": ((D, tokens * D), (wide, D)),
        "x_broadcast_heads": ((D, 0), (hd, D)),
        "x_broadcast_projection": ((0, D), (wide, D)),
        "z_broadcast_x_sliced": ((wide, D), (0, D)),
    }[form]
    if heads == 1:
        x, z = (x[0], D), (z[0], D)
    return x[0], x[1], z[0], z[1]


# The class ``forward_layout`` gives each form of :func:`_form_strides` of 16
# heads and of one head: "packed" for rows the host reads as packed (no class).
FORM_CLASSES = {
    "projection": (loader.PROJECTION_LAYOUT, loader.SLICED_LAYOUT),
    "projw4": (loader.PROJECTION_LAYOUT, loader.SLICED_LAYOUT),
    "both_sliced": (loader.SLICED_LAYOUT, loader.SLICED_LAYOUT),
    "padded": (loader.SLICED_LAYOUT, loader.SLICED_LAYOUT),
    "x_sliced": (loader.SLICED_LAYOUT, loader.SLICED_LAYOUT),
    "head_major": (loader.HEAD_MAJOR_LAYOUT, "packed"),
    "padded_heads": (None, loader.SLICED_LAYOUT),
    "padded_head_major": (None, "packed"),
    "mixed": (None, loader.SLICED_LAYOUT),
    "x_broadcast_heads": (None, "packed"),
    "x_broadcast_projection": (None, None),
    "z_broadcast_x_sliced": (None, None),
}


@pytest.mark.parametrize("sm_count", PLAN_SM_COUNTS)
def test_forward_plan_of_every_form_differs_from_0_2_17_only_in_the_moved_bands(
    sm_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host plan only, no device: for every form of ``FORM_CLASSES`` of 16
    heads and of one head, at the whole tokens of every whole rows per SM up
    to 3200 and of every cut of either version's tables and a row either
    side, the host's packed-rows test and ``forward_layout`` give the form's
    class, and ``module_for_rows`` of that class gives the module version
    0.2.17 gives (its tables put back, :func:`_put_back_0_2_17_tables`, and
    its class, :func:`_forward_layout_0_2_17`) but where 0.2.18 moved it
    (``suite.V0218_MOVED_BANDS``): a gate of more than one head sliced from a
    projection and whole-token rows (a gate of one head, ``x`` and ``z``
    both sliced, padded rows, ``x`` sliced beside packed ``z``, padded heads
    and a head-major ``x`` beside such a gate of one head) on sm_100a and
    sm_103a, each of which moves; broadcasts, padded heads of 16 heads,
    padded head-major planes, mixed layouts, head-major rows and sm_107a
    run what 0.2.17 ran."""
    cases = []
    for form, classes in FORM_CLASSES.items():
        for heads, cls in zip((16, 1), classes, strict=True):
            for arch in loader.ARCHES:
                rows = {k * sm_count for k in range(1, 3201)}
                tables = [loader.STRIDED_FORWARD_BANDS[arch]]
                tables += [
                    t[arch] for t in loader.LAYOUT_FORWARD_BANDS.values() if arch in t
                ]
                tables += [
                    t[arch] for t in suite.V0217_FORWARD_BANDS.values() if t.get(arch)
                ]
                for bands in tables:
                    for last, _ in bands[:-1]:
                        rows |= {last * sm_count + d for d in (-1, 0, 1)}
                for tokens in sorted({max(2, -(-r // heads)) for r in rows if r >= 1}):
                    strides = _form_strides(form, tokens, heads)
                    x_tok, x_head, z_tok, z_head = strides
                    row = heads * D
                    packed = x_head == z_head == D and {x_tok, z_tok} <= {0, row}
                    got = (
                        "packed"
                        if packed
                        else loader.forward_layout(tokens, heads, *strides)
                    )
                    assert got == cls, (form, heads, tokens, got)
                    if packed:
                        continue
                    old_cls = _forward_layout_0_2_17(tokens, heads, *strides)
                    cases.append((arch, form, heads, tokens * heads, cls, old_cls))
    assert cases, sm_count

    def plan(use_old: bool) -> list:
        return [
            loader.module_for_rows(
                arch,
                "fwd",
                rows,
                sm_count,
                packed=False,
                layout=old_cls if use_old else cls,
            )
            for arch, _, _, rows, cls, old_cls in cases
        ]

    shipped = plan(False)
    _put_back_0_2_17_tables(monkeypatch)
    old = plan(True)
    moved = set()
    for case, name, was in zip(cases, shipped, old, strict=True):
        arch, form, heads, rows, cls, _ = case
        want = _v0218_moved(arch, rows, sm_count, False, cls)
        if want is None:
            assert name == was, case
            continue
        routes = loader.ROUTES[arch]
        assert (was, name) == (routes[want[0]], routes[want[1]]), case
        moved.add((arch, form, heads))
    want_moved = {
        (arch, form, heads)
        for form, classes in FORM_CLASSES.items()
        for heads, cls in zip((16, 1), classes, strict=True)
        for arch in ("sm_100a", "sm_103a")
        if cls in (loader.PROJECTION_LAYOUT, loader.SLICED_LAYOUT)
    }
    assert moved == want_moved, sorted(moved ^ want_moved)


# Forward inputs around and past the ends of the forward's bands: (tokens,
# heads, layout); ``heads=None`` takes, with ``end`` the last row of band ``i``
# of a ``FORWARD_KINDS`` kind (``last * SMs``), ``end // h + tokens`` tokens of
# ``h`` heads: of one head for ``flat<i>`` (the architecture's
# ``FORWARD_BANDS``; packed rows: one flat [rows, 128] tensor) and ``other<i>``
# (its ``STRIDED_FORWARD_BANDS``; ``x`` and ``z`` both sliced from wider
# projections), of 4 heads for ``projection<i>`` (its
# ``PROJECTION_FORWARD_BANDS``; ``z`` the gate columns of a wider projection, a
# class of more than one head) and ``headmajor<i>`` (its
# ``HEAD_MAJOR_FORWARD_BANDS``; 4 divides 148, 152 and 212, so ``end`` is whole
# tokens there); ``token<i>`` takes one token of ``end + 1`` heads (packed rows
# whose token stride is passed as 0); skipped where the architecture has no
# such band with an end.
FORWARD_CASES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(16, 1, "contiguous", id="16x1"),
    pytest.param(17, 1, "contiguous", id="17x1"),
    pytest.param(2, 16, "contiguous", id="2x16"),
    pytest.param(33, 1, "contiguous", id="33x1"),
    pytest.param(13, 7, "contiguous", id="13x7"),
    pytest.param(13, 48, "projection", id="13x48-projection"),
    pytest.param(21, 48, "head_major", id="21x48-head_major"),
    pytest.param(-1, None, "flat0", id="band0-end-1"),
    pytest.param(0, None, "flat0", id="band0-end"),
    pytest.param(1, None, "flat0", id="band0-end+1"),
    pytest.param(0, None, "flat1", id="band1-end"),
    pytest.param(1, None, "flat1", id="band1-end+1"),
    pytest.param(0, None, "projection0", id="projection-band0-end"),
    pytest.param(1, None, "projection0", id="projection-band0-end+1"),
    pytest.param(0, None, "projection1", id="projection-band1-end"),
    pytest.param(1, None, "projection1", id="projection-band1-end+1"),
    pytest.param(0, None, "headmajor0", id="headmajor-band0-end"),
    pytest.param(1, None, "headmajor0", id="headmajor-band0-end+1"),
    pytest.param(0, None, "headmajor1", id="headmajor-band1-end"),
    pytest.param(1, None, "headmajor1", id="headmajor-band1-end+1"),
    pytest.param(0, None, "other0", id="other-band0-end"),
    pytest.param(1, None, "other0", id="other-band0-end+1"),
    pytest.param(0, None, "other1", id="other-band1-end"),
    pytest.param(1, None, "other1", id="other-band1-end+1"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(4097, 48, "contiguous", id="4097x48"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(6000, 32, "contiguous", id="6000x32"),
    pytest.param(0, None, "flat2", id="band2-end"),
    pytest.param(1, None, "flat2", id="band2-end+1"),
    pytest.param(0, None, "flat3", id="band3-end"),
    pytest.param(1, None, "flat3", id="band3-end+1"),
    pytest.param(0, None, "flat4", id="band4-end"),
    pytest.param(1, None, "flat4", id="band4-end+1"),
    pytest.param(0, None, "flat5", id="band5-end"),
    pytest.param(1, None, "flat5", id="band5-end+1"),
    pytest.param(1, None, "token0", id="one-token-band0-end+1"),
    pytest.param(1, None, "token1", id="one-token-band1-end+1"),
    pytest.param(1, None, "token2", id="one-token-band2-end+1"),
    pytest.param(1, None, "token3", id="one-token-band3-end+1"),
    pytest.param(1, None, "token4", id="one-token-band4-end+1"),
    pytest.param(1, None, "token5", id="one-token-band5-end+1"),
    pytest.param(0, None, "projection2", id="projection-band2-end"),
    pytest.param(1, None, "projection2", id="projection-band2-end+1"),
    pytest.param(0, None, "projection3", id="projection-band3-end"),
    pytest.param(1, None, "projection3", id="projection-band3-end+1"),
    pytest.param(0, None, "projection4", id="projection-band4-end"),
    pytest.param(1, None, "projection4", id="projection-band4-end+1"),
    pytest.param(0, None, "projection5", id="projection-band5-end"),
    pytest.param(1, None, "projection5", id="projection-band5-end+1"),
    pytest.param(0, None, "headmajor2", id="headmajor-band2-end"),
    pytest.param(1, None, "headmajor2", id="headmajor-band2-end+1"),
    pytest.param(0, None, "headmajor3", id="headmajor-band3-end"),
    pytest.param(1, None, "headmajor3", id="headmajor-band3-end+1"),
    pytest.param(0, None, "other2", id="other-band2-end"),
    pytest.param(1, None, "other2", id="other-band2-end+1"),
    pytest.param(0, None, "other3", id="other-band3-end"),
    pytest.param(1, None, "other3", id="other-band3-end+1"),
    pytest.param(0, None, "other4", id="other-band4-end"),
    pytest.param(1, None, "other4", id="other-band4-end+1"),
    pytest.param(7685, 48, "contiguous", id="7685x48"),
    pytest.param(7685, 48, "projection", id="7685x48-projection"),
    pytest.param(7685, 48, "head_major", id="7685x48-head_major"),
    pytest.param(9500, 48, "contiguous", id="9500x48"),
    pytest.param(9500, 48, "projection", id="9500x48-projection"),
    pytest.param(9500, 48, "head_major", id="9500x48-head_major"),
    pytest.param(16384, 32, "contiguous", id="16384x32"),
    pytest.param(8192, 48, "contiguous", id="8192x48"),
    pytest.param(8192, 48, "projection", id="8192x48-projection"),
    pytest.param(8192, 48, "head_major", id="8192x48-head_major"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
    pytest.param(32737, 32, "contiguous", id="32737x32"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", FORWARD_CASES)
def test_forward_around_its_bands_meets_the_naive_tier(
    tokens,
    heads,
    layout,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """On both sides of the ends of the forward's bands and past them, the
    forward launches ``module_for_rows``' module for the class of its rows
    (:func:`_forward_class`, from the strides of the inputs as the host gets
    them): the module of the first band of the class's table
    (:func:`_forward_table`: ``FORWARD_BANDS`` for packed rows,
    ``PROJECTION_FORWARD_BANDS`` or ``HEAD_MAJOR_FORWARD_BANDS`` for rows of
    those classes where the architecture has the table, else
    ``STRIDED_FORWARD_BANDS``) with ``rows <= last * SMs`` (the tiny, short,
    pipelined, mid, long, extra-long and plain forwards, each where the
    architecture's bands name it), but on packed rows of
    ``FORWARD_TINY_GAP`` (16, 17, 32 and 33 rows: its edges) the module of
    its role; the inputs' class is their layout's (:func:`_made_class`).
    ``y`` repeats bit for bit and, on the inputs of every draw of ``judge``,
    is judged against the naive implementation."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    routes = loader.ROUTES[arch]
    if heads is None:
        kind = next(k for k in FORWARD_KINDS if layout.startswith(k))
        i = int(layout[len(kind) :])
        kind_heads, layout, kind_packed, kind_layout = FORWARD_KINDS[kind]
        table = _forward_table(arch, 0, kind_packed, kind_layout)
        if i >= len(table) or table[i][0] is None:
            pytest.skip(f"{arch} has no {kind} forward band {i} with an end")
        end = int(table[i][0]) * sms
        if kind_heads is None:
            tokens, heads = 1, end + tokens
        else:
            tokens, heads = end // kind_heads + tokens, kind_heads
    rows = tokens * heads
    want_class = _made_class(tokens, heads, layout)
    launched = _spy_launches(monkeypatch)

    def run(j, s):
        x, z, w, _ = suite.make(tokens, heads, layout=layout, seed=s)
        packed, layout_class = _forward_class(x, z)
        assert (packed, layout_class) == want_class, (layout, tokens, heads)
        bands = _forward_table(arch, rows, packed, layout_class)
        want = next(
            routes[band] for last, band in bands if last is None or rows <= last * sms
        )
        gap = loader.FORWARD_TINY_GAP.get(arch)
        if packed and gap is not None and gap[0] < rows <= gap[1]:
            want = routes[gap[2]]
        name = loader.module_for_rows(
            arch, "fwd", rows, sms, packed=packed, layout=layout_class
        )
        assert name == want, (arch, rows, sms, layout_class)
        launched.clear()
        y = nd_forward(x, z, w, EPS)
        assert_same_bits("y", nd_forward(x, z, w, EPS), y)
        assert launched == [name] * 2, launched
        return silu_draw(x, z, w, None, EPS, [{"y": y}], {"y": "y"})

    judge(record_property, run, seed=21)


# Gate inputs from about -87.34 down to above about -88.72, where FP32
# ``expf(-z)`` does not overflow but the sigmoid is a nonzero FP32 subnormal,
# and ``y`` a nonzero BF16 value of ordinary rows: a forward whose gate
# flushed to 0 there would give 0.
GATE_TAIL = (-87.5, -88.0, -88.5)


@requires_device
@pytest.mark.parametrize("band", range(7))
@pytest.mark.parametrize("kind", ["flat", "projection", "headmajor", "other"])
def test_forward_keeps_the_gate_tail_in_every_band(
    kind, band, determinism_setting, monkeypatch: pytest.MonkeyPatch, record_property
) -> None:
    """At the first whole tokens of every forward band of each class's table
    (``FORWARD_KINDS``: ``FORWARD_BANDS`` of packed rows,
    ``PROJECTION_FORWARD_BANDS`` and ``HEAD_MAJOR_FORWARD_BANDS`` of 4 heads,
    and ``STRIDED_FORWARD_BANDS`` of ``x`` and ``z`` both sliced from wider
    projections), with every ``z`` in :data:`GATE_TAIL`, the inputs are of the
    kind's class (:func:`_forward_class`), the forward launches the
    band's module (``module_for_rows``' module for that class) and, on the
    inputs of every draw of ``judge`` (``x`` and ``w`` of the draw), ``y`` is
    judged against the naive implementation: no forward program flushes the
    gate to 0 where FP32 keeps it."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    heads, layout, packed, layout_class = FORWARD_KINDS[kind]
    table = _forward_table(arch, 0, packed, layout_class)
    if band >= len(table):
        pytest.skip(f"{arch} has {len(table)} {kind} forward bands")
    # The band's first whole tokens past the previous band's end; in the first
    # band one token of packed rows, two of other rows, as one token is packed
    # whatever its strides.
    if band:
        tokens = int(table[band - 1][0]) * sms // heads + 1
    else:
        tokens = 1 if packed else 2
    rows = tokens * heads
    name = loader.ROUTES[arch][table[band][1]]
    routed = loader.module_for_rows(
        arch, "fwd", rows, sms, packed=packed, layout=layout_class
    )
    assert routed == name, (arch, kind, rows, routed)
    launched = _spy_launches(monkeypatch)

    def run(j, s):
        x, z, w, _ = suite.make(tokens, heads, layout=layout, seed=s)
        assert _forward_class(x, z) == (packed, layout_class), (kind, tokens)
        tail = torch.tensor(GATE_TAIL, dtype=z.dtype, device=z.device)
        index = torch.arange(z.numel(), device=z.device) % len(GATE_TAIL)
        z.copy_(tail[index].view(z.shape))
        launched.clear()
        y = nd_forward(x, z, w, EPS)
        assert launched == [name], (arch, rows, launched)
        return silu_draw(x, z, w, None, EPS, [{"y": y}], {"y": "y"})

    judge(record_property, run, seed=23)


# Forward inputs whose ``x`` and ``z`` (``both``), or one of them, are one
# token of its own storage broadcast over more tokens (``expand``: token
# stride 0), at row counts at which packed rows run a program that reads
# packed rows at their linear offsets: a tiny forward (no stride; on 148 SMs
# or more) or, on sm_107a past 280 rows per SM, a pipelined forward of packed
# rows (passed the token stride ``H * 128``: ``fwd_pipe_packed`` up to 8192
# rows per SM, at 1300 x 48 rows, and ``fwd_keep_pipe_packed`` past it, at
# 36865 x 48 rows on 212 and 216 SMs).  Such a program cannot read them (the
# tiny forward takes contiguous ``x`` and ``z`` only, and the pipelined
# forwards of packed rows would read rows past the broadcast token), so they
# run the forward of rows of no class of their own (``module_for_rows`` with
# ``packed=False`` and no layout: ``STRIDED_FORWARD_BANDS``), which reads the
# token stride.
BROADCAST_CASES = [
    pytest.param(2, 1, "both", id="2x1"),
    pytest.param(3, 48, "both", id="3x48"),
    pytest.param(37, 48, "both", id="37x48"),
    pytest.param(100, 48, "both", id="100x48"),
    pytest.param(1300, 48, "both", id="1300x48"),
    pytest.param(36865, 48, "both", id="36865x48"),
    pytest.param(3, 48, "x", id="3x48-x"),
    pytest.param(3, 48, "z", id="3x48-z"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,broadcast", BROADCAST_CASES)
def test_forward_of_inputs_broadcast_over_tokens_runs_the_forward_of_other_layouts(
    tokens,
    heads,
    broadcast,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """``BROADCAST_CASES``: the forward passes a broadcast input's token
    stride 0 and launches ``module_for_rows``' module of other layouts
    (``packed=False``, no layout), no program of packed rows at their linear
    offsets, where packed rows of the same count run one (skipped elsewhere:
    ``BROADCAST_STRIDED_CASES``, ``BROADCAST_KEEP_PIPE_CASES``); ``y``
    repeats bit for bit and, on the
    inputs of every draw of ``judge``, is judged against the naive
    implementation."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    rows = tokens * heads
    linear = _linear_forwards(arch)
    if loader.module_for_rows(arch, "fwd", rows, sms) not in linear:
        pytest.skip(f"{rows} rows on {sms} SMs: packed rows run no such program")
    name = loader.module_for_rows(arch, "fwd", rows, sms, packed=False)
    assert name not in linear, (arch, rows, name)
    _judge_broadcast(tokens, heads, broadcast, name, monkeypatch, record_property, 27)


def _linear_forwards(arch):
    """The forward modules of ``arch`` that read packed rows at their linear
    offsets: the tiny forwards and the pipelined forwards of packed rows
    (``fwd_pipe_packed`` and ``fwd_keep_pipe_packed``)."""
    return {
        name
        for role, name in loader.ROUTES[arch].items()
        if role.startswith("fwd") and role.endswith(("_tiny", "_packed"))
    }


def _judge_broadcast(
    tokens, heads, broadcast, name, monkeypatch, record_property, seed
):
    """Two forwards of ``tokens`` x ``heads`` rows with ``x``, ``z`` or both
    (``broadcast``) one token broadcast over the others: each launches
    ``name`` with the inputs' token strides (0 for a broadcast input), ``y``
    repeats bit for bit and, on the inputs of every draw of ``judge``, is
    judged against the naive implementation."""
    arch = loader.device_arch(torch.cuda.current_device())
    rows = tokens * heads
    launched = _spy_strides(monkeypatch)

    def run(j, s):
        x, z, w, _ = suite.make(tokens, heads, seed=s)
        if broadcast in ("both", "x"):
            x = x[:1].clone().expand(tokens, heads, D)
        if broadcast in ("both", "z"):
            z = z[:1].clone().expand(tokens, heads, D)
        launched.clear()
        y = nd_forward(x, z, w, EPS)
        assert_same_bits("y", nd_forward(x, z, w, EPS), y)
        assert [n for n, _ in launched] == [name] * 2, (arch, rows, launched)
        for _, strides in launched:
            for which, t in (("x", x), ("z", z)):
                want = 0 if t.stride(0) == 0 else heads * D
                assert strides[f"{which}_tok"] == want, (which, strides)
        return silu_draw(x, z, w, None, EPS, [{"y": y}], {"y": "y"})

    judge(record_property, run, seed=seed)


# Broadcast inputs as in ``BROADCAST_CASES`` at row counts at which packed
# rows run a program that reads the token stride (on 148 SMs or more: the
# mid forward at 441 x 48 rows on every architecture and at 185 x 48 on
# sm_103a, the long forward of sm_100a and the extra-long forward of sm_103a
# and sm_107a at 883 x 48, the extra-long forward of sm_100a and sm_103a at
# 1300 x 48, the pipelined forward of sm_100a and sm_103a at 4000 x 48, whole
# waves of such rows): such a call runs the program of packed rows of the
# same count, as 0.2.11 did, with the token stride 0.  A count in a band of
# ``fwd_keep_pipe`` (185 x 48 on sm_100a) skips here
# (``BROADCAST_KEEP_PIPE_CASES``).
BROADCAST_STRIDED_CASES = [
    pytest.param(185, 48, "both", id="185x48"),
    pytest.param(441, 48, "both", id="441x48"),
    pytest.param(883, 48, "both", id="883x48"),
    pytest.param(1300, 48, "both", id="1300x48"),
    pytest.param(4000, 48, "both", id="4000x48"),
    pytest.param(883, 48, "x", id="883x48-x"),
    pytest.param(883, 48, "z", id="883x48-z"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,broadcast", BROADCAST_STRIDED_CASES)
def test_forward_of_inputs_broadcast_over_tokens_elsewhere_runs_the_program_of_packed_rows(
    tokens,
    heads,
    broadcast,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """``BROADCAST_STRIDED_CASES``: where packed rows of the same count run
    a program that reads the token stride, the forward of a broadcast input
    launches that program (``module_for_rows``' module of packed rows) with
    the token stride 0 (skipped where packed rows run a program that reads
    them at their linear offsets, ``BROADCAST_CASES``, or ``fwd_keep_pipe``,
    ``BROADCAST_KEEP_PIPE_CASES``); ``y`` repeats bit for bit and, on the
    inputs of every draw of ``judge``, is judged against the naive
    implementation."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    rows = tokens * heads
    name = loader.module_for_rows(arch, "fwd", rows, sms)
    if name in _linear_forwards(arch):
        pytest.skip(
            f"{rows} rows on {sms} SMs: packed rows run {name} (BROADCAST_CASES)"
        )
    if name == loader.ROUTES[arch].get("fwd_keep_pipe"):
        pytest.skip(
            f"{rows} rows on {sms} SMs: packed rows run fwd_keep_pipe "
            "(BROADCAST_KEEP_PIPE_CASES)"
        )
    _judge_broadcast(tokens, heads, broadcast, name, monkeypatch, record_property, 29)


# Broadcast inputs as in ``BROADCAST_CASES`` (``x``, ``z`` or both) of
# ``BROADCAST_KEEP_PIPE_HEADS`` heads at rows per SM in the band of
# ``fwd_keep_pipe`` (``FORWARD_BANDS``: 59 on sm_100a, in its band from 37 to
# 85; the tokens ``rows_per_sm * SMs // BROADCAST_KEEP_PIPE_HEADS``, so the
# rows are exactly that many rows per SM on 148 and 152 SMs).  The pipelined
# forward with ``x`` and ``z`` loaded with the L2's default policy was measured
# on packed rows only, so such a call runs ``fwd_pipe`` (the program of 0.2.16
# there, the same bits) with the token stride 0; a case where packed rows of
# its count run another program skips.
BROADCAST_KEEP_PIPE_HEADS = 4
BROADCAST_KEEP_PIPE_CASES = [
    pytest.param(rows_per_sm, broadcast, id=f"{rows_per_sm}-per-SM-{broadcast}")
    for rows_per_sm in (59,)
    for broadcast in ("both", "x", "z")
]


@requires_device
@pytest.mark.parametrize("rows_per_sm,broadcast", BROADCAST_KEEP_PIPE_CASES)
def test_broadcast_forward_in_a_keep_pipe_band_runs_fwd_pipe(
    rows_per_sm,
    broadcast,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """``BROADCAST_KEEP_PIPE_CASES``: where packed rows of the same count run
    ``fwd_keep_pipe``, the forward of a broadcast input launches ``fwd_pipe``
    with the token stride 0 (skipped where packed rows run another program);
    ``y`` repeats bit for bit and, on the inputs of every draw of ``judge``,
    is judged against the naive implementation."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    heads = BROADCAST_KEEP_PIPE_HEADS
    tokens = rows_per_sm * sms // heads
    rows = tokens * heads
    routes = loader.ROUTES[arch]
    if loader.module_for_rows(arch, "fwd", rows, sms) != routes.get("fwd_keep_pipe"):
        pytest.skip(f"{rows} rows on {sms} SMs: packed rows run no fwd_keep_pipe")
    name = routes["fwd_pipe"]
    _judge_broadcast(tokens, heads, broadcast, name, monkeypatch, record_property, 31)


# Gate inputs at and past the edges of the packed gate's range: below -41,
# around FP32 expf's overflow (|z| from 87.5 to 89), subnormal and zero,
# infinite and NaN, beside saturated and ordinary ones.
GATE_EDGE_VALUES = [
    -1e4,
    -100.0,
    -89.0,
    -88.5,
    -88.0,
    -87.5,
    -42.0,
    -41.0,
    -40.0,
    -1e-30,
    -1e-39,
    -0.0,
    0.0,
    1e-39,
    1e-30,
    40.0,
    87.5,
    88.0,
    89.0,
    100.0,
    1e4,
    float("inf"),
    float("-inf"),
    float("nan"),
]
# Forward inputs (``x`` and ``z`` packed rows, or ``z`` the gate columns of a
# wider projection) the tiny or the short forward runs, and those at the end
# of the short forward's band on sm_100a and sm_103a, 12 rows per SM, on 148
# SMs (37x48, 253x7, 55x32) and on 152 SMs (38x48, 260x7, 57x32), and on
# sm_107a, 59 rows per SM, on 212 SMs (260x48, 1786x7, 390x32), and one token
# past it (261x48, 1787x7, 391x32; of the projection layout
# ``fwd_layout_keep_pipe`` runs there), with forward inputs at and past 48
# rows per SM on 148 and 152 SMs (148x48, 1014x7, 222x32, 152x48, 1042x7,
# 228x32 and one token more), whose packed rows run ``fwd_keep_pipe`` on
# sm_100a and the mid forward on sm_103a, and the projection layout's the mid
# forward.
GATE_EDGE_CASES = [
    pytest.param(1, 48, id="1x48"),
    pytest.param(13, 48, id="13x48"),
    pytest.param(30, 48, id="30x48"),
    pytest.param(31, 48, id="31x48"),
    pytest.param(33, 48, id="33x48"),
    pytest.param(37, 48, id="37x48"),
    pytest.param(38, 48, id="38x48"),
    pytest.param(13, 7, id="13x7"),
    pytest.param(217, 7, id="217x7"),
    pytest.param(253, 7, id="253x7"),
    pytest.param(260, 7, id="260x7"),
    pytest.param(55, 32, id="55x32"),
    pytest.param(57, 32, id="57x32"),
    pytest.param(129, 48, id="129x48"),
    pytest.param(148, 48, id="148x48"),
    pytest.param(149, 48, id="149x48"),
    pytest.param(152, 48, id="152x48"),
    pytest.param(153, 48, id="153x48"),
    pytest.param(1014, 7, id="1014x7"),
    pytest.param(1015, 7, id="1015x7"),
    pytest.param(1042, 7, id="1042x7"),
    pytest.param(1043, 7, id="1043x7"),
    pytest.param(222, 32, id="222x32"),
    pytest.param(223, 32, id="223x32"),
    pytest.param(228, 32, id="228x32"),
    pytest.param(229, 32, id="229x32"),
    pytest.param(260, 48, id="260x48"),
    pytest.param(261, 48, id="261x48"),
    pytest.param(1786, 7, id="1786x7"),
    pytest.param(1787, 7, id="1787x7"),
    pytest.param(390, 32, id="390x32"),
    pytest.param(391, 32, id="391x32"),
]


def _gate_edge_inputs(
    tokens: int, heads: int, columns: int = D, *, seed: int = 23, finite=False
):
    """``(x, z, w, dy)`` of ``suite.make`` (seed ``seed``) with gate inputs
    from ``GATE_EDGE_VALUES`` (its finite ones where ``finite``) in a quarter
    of the elements of the first ``columns`` columns of two of every three
    rows (so the packed gate's range check fails for some threads of a row and
    holds for others) and ordinary ones in the rest."""
    x, z, w, dy = suite.make(tokens, heads, seed=seed)
    g = torch.Generator(device=z.device).manual_seed(seed + 1)
    edges = [v for v in GATE_EDGE_VALUES if math.isfinite(v) or not finite]
    values = torch.tensor(edges, device=z.device).bfloat16()
    pick = torch.randint(0, len(edges), z.shape, device=z.device, generator=g)
    edge = torch.rand(z.shape, device=z.device, generator=g) < 0.25
    rows = torch.arange(tokens * heads, device=z.device).view(tokens, heads, 1)
    edge &= rows % 3 != 0
    edge &= torch.arange(D, device=z.device) < columns
    z.copy_(torch.where(edge, values[pick], z))
    return x, z, w, dy


def _projected(z: torch.Tensor) -> torch.Tensor:
    """``z`` copied into the gate columns of a wider projection, read in
    place."""
    tokens, heads, _ = z.shape
    wide = torch.zeros(tokens, 2 * heads * D + 16, dtype=z.dtype, device=z.device)
    view = wide[:, heads * D : 2 * heads * D].view(tokens, heads, D)
    view.copy_(z)
    return view


@requires_device
@pytest.mark.parametrize("layout", ["contiguous", "projection"])
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_CASES)
def test_forward_gate_edges_keep_the_fp64_classes(
    tokens,
    heads,
    layout,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """With gate inputs at and past the packed gate's range in two of every
    three rows (a quarter of their elements, so the range check fails for some
    threads of a row and holds for others) and ordinary ones in the rest, the
    forward launches ``module_for_rows``' module for the class of its rows
    (:func:`_forward_class`: packed rows, or rows of the projection class
    with more than one token) and ``y`` repeats bit for bit
    and has the float64 oracle's ``NaN``, ``+inf`` and ``-inf`` positions;
    with the finite gate edges alone, on the inputs of every draw of
    ``judge``, ``y`` is judged against the naive implementation."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    rows = tokens * heads
    want_class = _made_class(tokens, heads, layout)
    launched = _spy_launches(monkeypatch)

    def forward(finite, seed):
        x, z, w, dy = _gate_edge_inputs(tokens, heads, seed=seed, finite=finite)
        if layout == "projection":
            z = _projected(z)
        packed, layout_class = _forward_class(x, z)
        assert (packed, layout_class) == want_class, (layout, tokens, heads)
        name = loader.module_for_rows(
            arch, "fwd", rows, sms, packed=packed, layout=layout_class
        )
        launched.clear()
        y = nd_forward(x, z, w, EPS)
        assert_same_bits("y", nd_forward(x, z, w, EPS), y)
        assert launched == [name] * 2, launched
        return x, z, w, y

    x, z, w, y = forward(False, 23)
    assert_same_classes("y", y, oracle(x, z, w, None, EPS, ("y",))["y"])

    def run(j, s):
        x, z, w, y = forward(True, s)
        return silu_draw(x, z, w, None, EPS, [{"y": y}], {"y": "y"})

    judge(record_property, run, seed=23)


# Backward inputs with the forward's gate edges (``_gate_edge_inputs``) in the
# first GATE_EDGE_BACKWARD_COLUMNS columns, so every output keeps finite
# elements (``dw`` in the other columns): 48 and 240 rows (the first
# small-row band of each role: for a BF16 weight gradient the short
# backward), 288 rows (the exact tiny backward of a BF16
# weight gradient on sm_100a and sm_107a, the fused backward on sm_103a),
# then 1008, 6192, 14208, 49488, 49536 and 245808 rows in the bands past them
# (the fused or the repeatable backward of a BF16 weight gradient, the
# repeatable backward of an FP32 one or, on sm_100a and sm_103a past 64 and up
# to 128 rows per SM, its static backward, as the architecture's
# ``BACKWARD_BANDS`` and the SM count place them: 14208 rows, 96 rows per SM
# on 148 SMs and 93.5 on 152 SMs, in the static band).
GATE_EDGE_BACKWARD_COLUMNS = 96
GATE_EDGE_BACKWARD_CASES = [
    pytest.param(1, 48, id="1x48"),
    pytest.param(5, 48, id="5x48"),
    pytest.param(6, 48, id="6x48"),
    pytest.param(21, 48, id="21x48"),
    pytest.param(129, 48, id="129x48"),
    pytest.param(296, 48, id="296x48"),
    pytest.param(1031, 48, id="1031x48"),
    pytest.param(1032, 48, id="1032x48"),
    pytest.param(5121, 48, id="5121x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads", GATE_EDGE_BACKWARD_CASES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_gate_edges_keep_the_fp64_classes(
    tokens,
    heads,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """With the forward's gate edges in the first
    ``GATE_EDGE_BACKWARD_COLUMNS`` columns (``_gate_edge_inputs``) the
    backward launches ``module_for_rows``' module, every output repeats bit
    for bit (but a BF16 ``dw`` in the fused and the row-claims bands), the
    workspace is zero after every call, and ``dx``, ``dz`` and ``dw`` (of
    every call) have the float64 oracle's ``NaN``, ``+inf`` and ``-inf``
    positions in every band; with the finite gate edges alone, on the inputs
    of every draw of ``judge``, every call's ``dx``, ``dz`` and ``dw`` are
    judged against the naive implementation."""
    sms = _sm_count()
    arch = loader.device_arch(torch.cuda.current_device())
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    rows = tokens * heads
    name = loader.module_for_rows(arch, role, rows, sms)
    varies = suite._varies(name)
    ws = nd_workspace(rows, torch.device("cuda"))
    launched = _spy_launches(monkeypatch)

    def calls(finite, seed):
        x, z, w, dy = _gate_edge_inputs(
            tokens, heads, GATE_EDGE_BACKWARD_COLUMNS, seed=seed, finite=finite
        )
        launched.clear()
        got = []
        for _ in range(2):
            dw = torch.empty(D, dtype=dw_dtype, device=x.device)
            got.append(nd_backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws))
            assert not bool(ws.any()), "the workspace is not zero"
        assert launched == [name] * 2, launched
        assert got[0][2].dtype == dw_dtype
        for what, g, a in zip(("dx", "dz", "dw"), *got, strict=True):
            if what != "dw" or not varies:
                assert_same_bits(what, a, g)
        return (x, z, w, dy), got if varies else got[:1]

    (x, z, w, dy), got = calls(False, 23)
    ref = oracle(x, z, w, dy, EPS, ("dx", "dz", "dw"))
    for k, outputs in enumerate(got):
        for what, g in zip(("dx", "dz", "dw"), outputs, strict=True):
            assert_same_classes(f"{what} of call {k}", g, ref[what])
            assert bool(ref[what].isfinite().any()), f"{what}: no finite element"

    def run(j, s):
        inputs, got = calls(True, s)
        names = ("dx", "dz", "dw")
        return silu_draw(
            *inputs,
            EPS,
            [dict(zip(names, outputs, strict=True)) for outputs in got],
            dict(zip(names, names, strict=True)),
        )

    judge(record_property, run, seed=23)


def _check_backward(
    record_property, inputs, dw_dtype, name, monkeypatch, *, seed
) -> None:
    """Over the draws of ``judge`` (``inputs(s)`` the draw's ``(x, z, w,
    dy)``): ``suite.REPEATS`` backward launches of module ``name`` on the
    first draw's inputs, one on each other draw's, into a ``dw_dtype``
    weight gradient with this package's workspace, then two accumulating
    ones from the same starting value (0.25): every output is the first
    launch's bits (the accumulated ``dw`` too; where a BF16 ``dw`` may vary,
    in the fused and the row-claims bands, ``dx`` and ``dz`` only), the
    workspace is zero after every call, and ``dx``, ``dz``, ``dw`` and the
    accumulated ``dw`` (of every launch where they may vary) are judged
    against the naive implementation."""
    varies = suite._varies(name)
    launched = _spy_launches(monkeypatch)
    names = ("dx", "dz", "dw")

    def run(j, s):
        x, z, w, dy = inputs(s)
        rows = x.numel() // D

        def out(value=None):
            if value is None:
                return torch.empty(D, dtype=dw_dtype, device=x.device)
            return torch.full((D,), value, dtype=dw_dtype, device=x.device)

        ws = nd_workspace(rows, x.device)
        launches = []
        first = None
        for _ in range(suite.REPEATS if j == 0 else 1):
            got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
            assert got[2].dtype == dw_dtype
            assert not bool(ws.any()), "the workspace is not zero"
            if first is None:
                first = got
                launches.append(dict(zip(names, got, strict=True)))
                continue
            for what, g, f in zip(names[:2], got[:2], first[:2], strict=True):
                assert_same_bits(what, g, f)
            suite.assert_dw_repeats(got[2], first[2], varies=varies, what="dw")
            if varies:
                launches.append({"dw": got[2]})
        acc = [
            nd_backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=out(0.25),
                accumulate_grad_weight=True,
                workspace=ws,
            )[2]
            for _ in range(2)
        ]
        assert not bool(ws.any()), "the workspace is not zero"
        suite.assert_dw_repeats(acc[1], acc[0], varies=varies, what="accumulated dw")
        launches += [{"accumulated dw": a} for a in (acc if varies else acc[:1])]
        return silu_draw(
            x,
            z,
            w,
            dy,
            EPS,
            launches,
            {"dx": "dx", "dz": "dz", "dw": "dw", "accumulated dw": "dw"},
            starts={"accumulated dw": torch.full((D,), 0.25, dtype=dw_dtype)},
        )

    judge(record_property, run, seed=seed)
    assert launched == [name] * (suite.REPEATS + 2 + 3 * (DRAWS - 1)), launched


# Every backward band role of ``loader.BACKWARD_BANDS`` on any architecture,
# in table order (the weight-gradient dtype from the role,
# :func:`_band_role_dtype`); a test of a band this device does not route
# skips.
BACKWARD_BAND_ROLES = list(
    dict.fromkeys(
        band
        for roles in loader.BACKWARD_BANDS.values()
        for bands in roles.values()
        for _, band in bands
    )
)
# The grid kinds of the modules each band role routes on the architectures
# that route it.
_BAND_KINDS = {
    band: {
        loader.MODULES[routes[band]]["grid_rule"]["kind"]
        for routes in loader.ROUTES.values()
        if band in routes
    }
    for band in BACKWARD_BAND_ROLES
}


def _band_nths() -> dict[tuple[str, int], bool]:
    """``(band role, n) -> whether it has an end`` of the ``n``-th band of each
    band role in a role's table on some architecture (a band role may hold
    more than one band: on sm_107a the BF16 repeatable role runs on both sides
    of the fused band, on sm_100a and sm_103a the FP32 one on both sides of the
    static band); a band has an end (``last``) where some architecture gives it
    one."""
    nths: dict[tuple[str, int], bool] = {}
    for roles in loader.BACKWARD_BANDS.values():
        for bands in roles.values():
            seen: dict[str, int] = {}
            for last, band in bands:
                seen[band] = seen.get(band, 0) + 1
                key = (band, seen[band])
                nths[key] = nths.get(key, False) or last is not None
    return nths


_BAND_NTHS = _band_nths()
# The ``(band role, n)`` bands in table order, and those with an end (``last``)
# on some architecture (every band but the last one of a role).
BACKWARD_BAND_NTHS = [
    (band, nth) for band in BACKWARD_BAND_ROLES for b, nth in _BAND_NTHS if b == band
]
_BANDS_WITH_AN_END = {key for key, has_end in _BAND_NTHS.items() if has_end}


def _band_id(band: str, nth: int) -> str:
    """The test id of the ``nth`` band of the band role ``band``: the role
    for the first one, the role and ``nth`` for a later one."""
    return band if nth == 1 else f"{band}{nth}"


# The edges of the backward's bands on this device (:func:`_band_modules`):
# (band role, nth, edge).  ``first`` and ``last`` are a band's first and last rows
# (``last`` where the band has an end); for a band of a chunked backward
# (grid kind ``chunks``) ``chunks`` is the program's last rows of chunks of
# ``k`` blocks, ``rows_per_block * chunk_limit * k`` for the fewest ``k`` in
# its band, and ``chunks+1`` its first rows of ``k + 1`` blocks, where its
# band holds them (the static band of an FP32 weight gradient, up to 128 rows
# per SM on sm_100a and sm_103a, holds none on fewer than 512 SMs: there those
# cases skip).
BACKWARD_BAND_EDGES = [
    pytest.param(band, nth, edge, id=f"{_band_id(band, nth)}-{edge}")
    for band, nth in BACKWARD_BAND_NTHS
    for edge in ("first", "last", "chunks", "chunks+1")
    if (edge != "last" or (band, nth) in _BANDS_WITH_AN_END)
    and (not edge.startswith("chunks") or "chunks" in _BAND_KINDS[band])
]
# Each band of each band role (``BACKWARD_BAND_NTHS``) as ``(band, nth)``.
BACKWARD_BAND_CASES = [
    pytest.param(band, nth, id=_band_id(band, nth)) for band, nth in BACKWARD_BAND_NTHS
]
# Each long role (a band role with a cut in ``LONG_BACKWARD_ROWS_PER_SM`` on
# some architecture, and ``LONG``) as ``(long role, 1)``: the packed rows of
# its band from the cut on (:func:`_long_band_modules`).
LONG_BACKWARD_CASES = [
    pytest.param(band + loader.LONG, 1, id=band + loader.LONG)
    for band in sorted(
        {band for cuts in loader.LONG_BACKWARD_ROWS_PER_SM.values() for band in cuts}
    )
]


def _case_occurrence(
    arch: str, role: str, sms: int, band: str, nth: int
) -> tuple[str, int, int | None] | None:
    """``(module, first rows, last rows or None)`` of the case ``(band,
    nth)`` of ``BACKWARD_BAND_CASES`` or ``LONG_BACKWARD_CASES`` on ``arch``
    with ``sms`` SMs: the ``nth`` band of a band role (:func:`_band_occurrence`)
    or the long role ``band``'s packed rows (:func:`_long_band_modules`), or
    ``None`` where ``arch`` with ``sms`` SMs has no such band."""
    if not band.endswith(loader.LONG):
        return _band_occurrence(arch, role, sms, band, nth)
    found = [
        (module, first, last)
        for name, module, first, last in _long_band_modules(arch, role, sms)
        if name == band
    ]
    return found[nth - 1] if len(found) >= nth else None


@requires_device
@pytest.mark.parametrize("band,nth,edge", BACKWARD_BAND_EDGES)
def test_backward_band_edges_meet_the_naive_tier(
    band,
    nth,
    edge,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """At the first and the last rows of each ``BACKWARD_BANDS`` band on this
    device's SM count (:func:`_band_modules`: the small-row backwards, the
    static backward of an FP32 weight gradient, the fused backward, the
    row-claims backwards and the repeatable backward of either weight
    gradient, each where the architecture routes it) and where a chunked
    program's chunks grow by a block (``backward_chunks``; skipped where its
    band holds no such rows): ``module_for_rows`` runs the
    band's program from its first to its last rows only, and
    ``_check_backward`` holds on flat [rows, 128] inputs into the band's
    weight-gradient dtype (each draw's)."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = _band_role_dtype(band)
    role = _role_of(dw_dtype)
    found = _band_occurrence(arch, role, sms, band, nth)
    if found is None:
        pytest.skip(f"{arch} routes no {_band_id(band, nth)} band on {sms} SMs")
    name, first, last = found
    if edge == "first":
        rows = first
    elif edge == "last":
        if last is None:
            pytest.skip(f"the {band} band of {arch} has no end")
        rows = last
    else:
        rule = loader.MODULES[name]["grid_rule"]
        if rule["kind"] != "chunks":
            pytest.skip(f"the {band} program of {arch} has no chunks")
        span = int(rule["rows_per_block"]) * int(rule["chunk_limit"])
        k = -(-first // span)
        rows = k * span + (edge == "chunks+1")
        if last is not None and rows > last:
            pytest.skip(f"the {band} band ({first}-{last} rows) has no such edge")
        assert loader.backward_chunks(name, rows)[0] == k + (edge == "chunks+1")
    assert loader.module_for_rows(arch, role, rows, sms) == name, (rows, sms)
    if edge == "first" and first > 1:
        assert loader.module_for_rows(arch, role, rows - 1, sms) != name, rows
    if edge == "last":
        assert loader.module_for_rows(arch, role, rows + 1, sms) != name, rows
    _check_backward(
        record_property,
        lambda s: suite.make(rows, 1, layout="flat", seed=s),
        dw_dtype,
        name,
        monkeypatch,
        seed=22,
    )


# Backward inputs of every program and layout: (tokens, heads, layout).
BACKWARD_SHAPES = [
    pytest.param(1, 48, "contiguous", id="1x48"),
    pytest.param(5, 48, "projection", id="5x48-projection"),
    pytest.param(13, 7, "head_major", id="13x7-head_major"),
    pytest.param(255, 1, "flat", id="255x1"),
    pytest.param(256, 1, "flat", id="256x1"),
    pytest.param(257, 1, "flat", id="257x1"),
    pytest.param(1031, 48, "contiguous", id="1031x48"),
    pytest.param(2048, 48, "contiguous", id="2048x48"),
    pytest.param(2701, 48, "contiguous", id="2701x48"),
    pytest.param(3091, 48, "projection", id="3091x48-projection"),
    pytest.param(3157, 48, "head_major", id="3157x48-head_major"),
    pytest.param(4097, 48, "projection", id="4097x48-projection"),
    pytest.param(4097, 48, "head_major", id="4097x48-head_major"),
    pytest.param(5300, 32, "contiguous", id="5300x32"),
    pytest.param(9600, 48, "contiguous", id="9600x48"),
    pytest.param(25000, 48, "contiguous", id="25000x48"),
    pytest.param(32737, 48, "contiguous", id="32737x48"),
]


@requires_device
@pytest.mark.parametrize("tokens,heads,layout", BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_shapes_meet_the_naive_tier(
    tokens,
    heads,
    layout,
    dw_dtype,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """Inputs of several shapes and layouts: ``module_for_rows`` runs one of
    the role's programs (the modules of its bands, :func:`_band_modules`, or
    on packed rows of its long roles, :func:`_long_band_modules`) and
    ``_check_backward`` holds."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    rows = tokens * heads
    packed = suite._backward_packed(tokens, heads, layout)
    name = loader.module_for_rows(arch, role, rows, sms, packed=packed)
    bands = _band_modules(arch, role, sms)
    if packed:
        bands += _long_band_modules(arch, role, sms)
    programs = {module for _, module, _, _ in bands}
    assert name in programs, (name, programs)
    _check_backward(
        record_property,
        lambda s: suite.make(tokens, heads, layout=layout, seed=s),
        dw_dtype,
        name,
        monkeypatch,
        seed=24,
    )


# The backend suite's ``BULK_BAND_CASES`` (strided and packed inputs in the
# bands of the static backward of an FP32 weight gradient and of the
# row-claims, the claimed, the fused and the repeatable backwards, the
# repeatable ones of either weight gradient) and the same
# layouts in the fused band (``bwd_bf16_fused``, every architecture) where the
# suite does not list it; a case of a band this device does not route skips.
FUSED_BANDS = ("bwd_bf16_fused",)
BULK_LAYOUTS = ("projection", "head_major", "one_token", "contiguous")
LAYOUT_BAND_CASES = [
    *suite.BULK_BAND_CASES,
    *(
        pytest.param(band, layout, id=f"{band}-{layout}")
        for band in FUSED_BANDS
        if band not in {case.values[0] for case in suite.BULK_BAND_CASES}
        for layout in BULK_LAYOUTS
    ),
]


@requires_device
@pytest.mark.parametrize("band,layout", LAYOUT_BAND_CASES)
def test_layouts_in_the_row_claims_and_claimed_backward_bands_meet_the_naive_tier(
    band,
    layout,
    determinism_setting,
    monkeypatch: pytest.MonkeyPatch,
    record_property,
) -> None:
    """``LAYOUT_BAND_CASES`` (strided and packed inputs in the bands of the
    FP32 static, the row-claims, the claimed, the fused and the repeatable
    backwards, ``suite._bulk_band_shape``): ``module_for_rows`` runs the
    band's program, the launch passes the strides of the layout, which pass
    the kernels' packed-rows test (``_packed``: ``dy``, ``x`` and ``z`` of one
    packed row stride, the rows the sm_100a and sm_103a programs warm with a
    bulk prefetch) for packed rows of more than one token only, and
    ``_check_backward`` holds (on the other layouts those programs warm the
    rows with a prefetch per thread, and so does the fused program, which
    warms packed rows with a bulk prefetch on every architecture)."""
    shape = suite._bulk_band_shape(band, layout)
    if shape is None:
        pytest.skip(f"this device routes no {band} band with a {layout} input")
    tokens, heads, layout = shape
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = torch.float32 if band.startswith("bwd_f32") else torch.bfloat16
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    x, z, w, dy = suite.make(tokens, heads, layout=layout, seed=26)
    rows = x.numel() // D
    packed = layout == "contiguous" and tokens > 1
    assert suite._backward_packed(tokens, heads, layout) == packed, layout
    name = loader.module_for_rows(arch, role, rows, sms, packed=packed)
    assert name == loader.ROUTES[arch][band], (rows, name)
    ws = nd_workspace(rows, x.device)
    launched = _spy_strides(monkeypatch)
    nd_backward(
        dy,
        x,
        z,
        w,
        EPS,
        grad_weight=torch.empty(D, dtype=dw_dtype, device=x.device),
        workspace=ws,
    )
    assert [n for n, _ in launched] == [name], launched
    assert _packed(launched[0][1]) == packed, (layout, launched[0][1])
    assert not bool(ws.any()), "the workspace is not zero"
    _check_backward(
        record_property,
        lambda s: suite.make(tokens, heads, layout=layout, seed=s),
        dw_dtype,
        name,
        monkeypatch,
        seed=26,
    )


@requires_device
@pytest.mark.parametrize("band,nth", BACKWARD_BAND_CASES + LONG_BACKWARD_CASES)
def test_backward_with_infinite_grad_out_keeps_the_fp64_classes(
    band, nth, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in rows that are not the last, in each
    ``BACKWARD_BANDS`` band of this device (:func:`_band_modules`: a band's
    last rows, 4097 rows past the first of the last band, which has no end)
    and in each long role's packed rows (:func:`_long_band_modules`: 4097
    rows past its cut), into the band's weight-gradient dtype: the weight
    gradient of the columns holding them is not finite and every launch's
    has the float64 oracle's classes (an infinity its sign, a ``NaN`` a
    ``NaN``, every other column finite), and ``dx`` and ``dz`` have the
    oracle's ``NaN``, ``+inf`` and ``-inf`` positions, on every repeat the
    same bits (but a BF16 ``dw`` in a fused or row-claims band), with the
    workspace zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = _band_role_dtype(band)
    role = _role_of(dw_dtype)
    found = _case_occurrence(arch, role, sms, band, nth)
    if found is None:
        pytest.skip(f"{arch} routes no {_band_id(band, nth)} band on {sms} SMs")
    name, first, last = found
    rows = first + 4097 if last is None else last
    assert loader.module_for_rows(arch, role, rows, sms) == name
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=27)
    # (row, column, sign): +inf twice in column 3, -inf in column 40 (row 17,
    # or two rows before the last of the solo band's 16), +inf and -inf in
    # column 77; no infinite term in the last row
    for r, c, sign in (
        (0, 3, 1.0),
        (rows // 2, 3, 1.0),
        (min(17, rows - 2), 40, -1.0),
        (rows // 3, 77, 1.0),
        (rows // 3 + 5, 77, -1.0),
    ):
        dy[r, c] = sign * float("inf")
    cols = torch.zeros(D, dtype=torch.bool, device=x.device)
    cols[[3, 40, 77]] = True
    ref = oracle(x, z, w, dy, EPS, ("dx", "dz", "dw"))
    assert torch.equal(~torch.isfinite(ref["dw"]), cols), "FP64 dw columns"

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    varies = suite._varies(name)
    first_run = None
    dws = []
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first_run is None:
            first_run = got
        for what, g, f in zip(("dx", "dz"), got[:2], first_run[:2], strict=True):
            assert_same_bits(what, g, f)
        if not varies:
            assert_same_bits("dw", got[2], first_run[2])
        dws.append(got[2])
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got in zip(("dx", "dz"), first_run[:2], strict=True):
        assert_same_classes(what, got, ref[what])
    for dw in dws if varies else dws[:1]:
        assert_same_classes("dw", dw, ref["dw"])
    assert launched == [name] * suite.REPEATS, launched

def _block_rows(name: str) -> int:
    """The rows of one row block of the backward module ``name``, from its
    record: ``rows_per_block`` of its grid rule for a chunked backward (grid
    kind ``chunks``: the static and the claimed backwards, 8) and for the
    repeatable backward (``claimed_row_blocks``, 16), ``rows_per_cta`` for
    the fused and a row-claims backward (``row_claims``, 16), and for a
    small-row backward (``cluster``) its launch block's threads over the
    teams of its CTA over 16 threads a row (``suite._cluster_block_rows``):
    16 rows for the solo and the short backward (one 256-thread team a CTA),
    8 for the exact tiny backwards (4 teams of the deterministic backward's
    128-thread CTA in a 512-thread CTA, each team its own 8-row block)."""
    rule = loader.MODULES[name]["grid_rule"]
    if rule["kind"] in ("chunks", "claimed_row_blocks"):
        return int(rule["rows_per_block"])
    if rule["kind"] == "row_claims":
        return int(rule["rows_per_cta"])
    assert rule["kind"] == "cluster", (name, rule)
    return suite._cluster_block_rows(name)


def _last_block_rows(block: int, first: int, last: int | None) -> dict[str, int]:
    """Three row counts of a program of ``block``-row blocks over ``first``
    to ``last`` rows that are not a multiple of its block: ``rem1`` one row
    in the last block and ``half`` half a block in it (``block // 2`` rows: 8
    rows of a 16-row block, whose padded rows must not turn an infinite
    ``grad_out`` in the last row into a ``NaN`` weight gradient), each the
    smallest such count from ``2 * block + 1`` rows and ``first`` (from 2
    rows and ``first`` where ``last`` is below that: the solo band, 1 to 16
    rows, one block), and ``remmax`` one row short of a whole block (the
    largest up to ``last``, or 4096 rows past ``rem1`` for the last band,
    which has no last row)."""
    lo = max(first, 2 * block + 1)
    if last is not None and lo > last:
        lo = max(first, 2)
    low = lo + (1 - lo) % block
    half = lo + (block // 2 - lo) % block
    high = low + 4096 + block - 2 if last is None else last - (last + 1) % block
    return {"rem1": low, "half": half, "remmax": high}


@requires_device
@pytest.mark.parametrize("count", ["rem1", "half", "remmax"])
@pytest.mark.parametrize("band,nth", BACKWARD_BAND_CASES + LONG_BACKWARD_CASES)
def test_backward_with_infinite_grad_out_in_the_last_row_keeps_the_fp64_classes(
    count, band, nth, determinism_setting, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Infinite ``grad_out`` in the last row, at row counts that are not a
    multiple of the program's row block (``_last_block_rows``: one row, half
    a block and one row short of a whole block in the last block), in each
    ``BACKWARD_BANDS`` band of this device (:func:`_band_modules`) and each
    long role's packed rows (:func:`_long_band_modules`) into the band's
    weight-gradient dtype: ``+inf`` in column 3, ``-inf`` in
    column 40, and ``+inf`` in column 77 beside ``-inf`` in the row before.
    The weight gradient of those columns has the float64 oracle's class
    (``+inf``, ``-inf`` and ``NaN``) in every band and every other column is
    finite (every launch's), and ``dx`` and ``dz`` have the oracle's
    ``NaN``, ``+inf`` and ``-inf`` positions, on every repeat the same bits
    (but a BF16 ``dw`` in a fused or row-claims band), with the workspace
    zero after every call."""
    arch = loader.device_arch(torch.cuda.current_device())
    sms = _sm_count()
    dw_dtype = _band_role_dtype(band)
    role = _role_of(dw_dtype)
    found = _case_occurrence(arch, role, sms, band, nth)
    if found is None:
        pytest.skip(f"{arch} routes no {_band_id(band, nth)} band on {sms} SMs")
    name, first, last = found
    block = _block_rows(name)
    rows = _last_block_rows(block, first, last)[count]
    if not first <= rows <= (last if last is not None else rows):
        pytest.skip(f"{band} ({first}-{last} rows) has no such row count")
    assert rows % block != 0, rows
    assert loader.module_for_rows(arch, role, rows, sms) == name, (rows, sms)
    x, z, w, dy = suite.make(rows, 1, layout="flat", seed=28)
    # (row, column, sign of the weight-gradient term): a term dy * silu(z) * x
    # * rstd has dy's sign times that of z * x, so dy takes the product
    for r, c, sign in (
        (rows - 1, 3, 1.0),
        (rows - 1, 40, -1.0),
        (rows - 1, 77, 1.0),
        (rows - 2, 77, -1.0),
    ):
        zx = float(z[r, c]) * float(x[r, c])
        assert zx != 0.0, (r, c)
        dy[r, c] = math.copysign(float("inf"), sign * zx)
    cols = torch.zeros(D, dtype=torch.bool, device=x.device)
    cols[[3, 40, 77]] = True
    ref = oracle(x, z, w, dy, EPS, ("dx", "dz", "dw"))
    dw64 = ref["dw"]
    assert torch.equal(~torch.isfinite(dw64), cols), "FP64 dw columns"
    assert bool(torch.isposinf(dw64[3])), dw64[3]
    assert bool(torch.isneginf(dw64[40])), dw64[40]
    assert bool(torch.isnan(dw64[77])), dw64[77]

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    ws = nd_workspace(rows, x.device)
    launched = _spy_launches(monkeypatch)
    varies = suite._varies(name)
    first_run = None
    dws = []
    for _ in range(suite.REPEATS):
        got = nd_backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
        assert got[2].dtype == dw_dtype
        if first_run is None:
            first_run = got
        for what, g, f in zip(("dx", "dz"), got[:2], first_run[:2], strict=True):
            assert_same_bits(what, g, f)
        if not varies:
            assert_same_bits("dw", got[2], first_run[2])
        dws.append(got[2])
        assert not bool(ws.any()), "the workspace is not zero"
    for what, got in zip(("dx", "dz"), first_run[:2], strict=True):
        assert_same_classes(f"{what} rows={rows}", got, ref[what])
    for dw in dws if varies else dws[:1]:
        assert_same_classes(f"dw rows={rows}", dw, dw64)
    assert launched == [name] * suite.REPEATS, launched

# --------------------------------------------------------------------------- one head: packed rows


def _packed(strides: dict) -> bool:
    """The kernels' packed-rows test over the strides of one launch: every
    input's head stride is 128 and its token stride ``heads * 128``.  The mid
    and the extra-long forward warm row blocks in L2 with a bulk prefetch only
    when it holds, and so do the row-claims mid and the claimed backward of
    sm_100a and sm_103a (``dy``, ``x`` and ``z`` of one packed row stride)."""
    heads = int(strides["heads"])
    inputs = [k[: -len("_head")] for k in strides if k.endswith("_head")]
    return bool(inputs) and all(
        strides[f"{n}_head"] == D and strides[f"{n}_tok"] == heads * D for n in inputs
    )


def _spy_strides(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """``(module, {"heads", "<input>_tok", "<input>_head"})`` of every launch."""
    launched: list[tuple[str, dict]] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        strides = {
            k: int(v)
            for k, v in values.items()
            if k == "heads" or k.endswith(("_tok", "_head"))
        }
        launched.append((name, strides))
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


# One-head layouts of a [rows, 128] input: (passes the packed-rows test, read
# in place).  ``three_d`` [rows, 1, 128], ``flat`` [rows, 128], ``transposed``
# the transpose of [1, rows, 128] (token stride 128, head stride rows * 128),
# ``first_of_two`` the first head of [rows, 2, 128] (token stride 256: rows
# that do not follow one another), ``misaligned`` [rows, 128] at a storage
# offset of 8 elements (copied once by the host).
ONE_HEAD_LAYOUTS = {
    "three_d": (True, True),
    "flat": (True, True),
    "transposed": (True, True),
    "first_of_two": (False, True),
    "misaligned": (True, False),
}


def _one_head(base: torch.Tensor, layout: str) -> torch.Tensor:
    """``base`` ([rows, 128]) as a one-head input of ``layout``."""
    rows = base.shape[0]
    if layout == "three_d":
        return base.clone().view(rows, 1, D)
    if layout == "flat":
        return base.clone()
    if layout == "transposed":
        return base.clone().view(1, rows, D).transpose(0, 1)
    if layout == "first_of_two":
        wide = torch.zeros(rows, 2, D, dtype=base.dtype, device=base.device)
        wide[:, 0] = base
        return wide[:, :1]
    if layout == "misaligned":
        buf = torch.zeros(rows * D + 8, dtype=base.dtype, device=base.device)
        out = buf[8:].view(rows, D)
        out.copy_(base)
        return out
    raise ValueError(layout)


@pytest.mark.parametrize("layout", list(ONE_HEAD_LAYOUTS))
def test_one_head_input_passes_the_packed_rows_test_when_its_rows_follow_one_another(
    layout,
) -> None:
    """Without a device: a one-head input's head stride is passed as 128 (it is
    never dereferenced), so the kernels see packed rows exactly when its token
    stride is 128 (or the host copied it), as for ``[T, H, 128]`` inputs."""
    base = torch.randn(7, D).bfloat16()
    t = _one_head(base, layout)
    view, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=t.device)
    packed, in_place = ONE_HEAD_LAYOUTS[layout]
    assert head == D
    assert tok == (2 * D if layout == "first_of_two" else D)
    assert _packed({"heads": 1, "x_tok": tok, "x_head": head}) == packed
    assert (view.data_ptr() == t.data_ptr()) == in_place
    assert torch.equal(view.reshape(-1, D), base)


@pytest.mark.parametrize(
    "shape,strides,want",
    [
        pytest.param((1, 1, D), None, (0, D), id="1x1"),
        pytest.param((1, D), None, (0, D), id="1-flat"),
        pytest.param((D,), None, (0, D), id="one-row"),
        pytest.param((1, 2, D), None, (0, D), id="1x2"),
        pytest.param((5, 2, D), None, (2 * D, D), id="5x2"),
        pytest.param((5, 3, D), None, (3 * D, D), id="5x3"),
        pytest.param((5, 2, D), (4 * D, D, 1), (4 * D, D), id="5x2-projection"),
        pytest.param((5, 2, D), (D, 5 * D, 1), (D, 5 * D), id="5x2-head_major"),
    ],
)
def test_kernel_view_strides(shape, strides, want) -> None:
    """Without a device: a single token's stride is passed as 0, a single
    head's as 128, any other dimension's as it is."""
    if strides is None:
        t = torch.zeros(shape, dtype=torch.bfloat16)
    else:
        t = torch.empty_strided(shape, strides, dtype=torch.bfloat16)
    _, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=t.device)
    assert (tok, head) == want


class _FunctionTensor(torch.Tensor):
    """A ``torch.Tensor`` subclass with its own ``__torch_function__``."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs or {})


# Whether this torch build has dimension names (named tensors; torch 2.14 has
# none): the cases with names skip without them.
NAMED_TENSORS = hasattr(torch.Tensor, "refine_names")


def _named(t: torch.Tensor) -> torch.Tensor:
    """``t`` with the dimension names ``T``, ``H`` and ``D``."""
    if not NAMED_TENSORS:
        pytest.skip("this torch build has no dimension names")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return t.refine_names("T", "H", "D")


def _view_layout(layout: str, device: torch.device) -> torch.Tensor:
    """A ``[T, H, 128]`` input of ``layout`` (``[5, 3, 128]`` but for
    ``one_token``, ``one_head`` and ``flat``; any dtype other than BF16 by
    name: ``float32``) on ``device``."""
    T, H = 5, 3
    if layout == "contiguous":
        return torch.randn(T, H, D, device=device).bfloat16()
    if layout == "projection":
        return torch.randn(T, H, 2 * D, device=device).bfloat16()[:, :, :D]
    if layout == "head_major":
        return torch.randn(H, T, D, device=device).bfloat16().transpose(0, 1)
    if layout == "one_token":
        return torch.randn(1, H, D, device=device).bfloat16()
    if layout == "one_head":
        return torch.randn(T, 2, D, device=device).bfloat16()[:, :1]
    if layout == "offset_16":
        return torch.randn(T * H * D + 16, device=device).bfloat16()[16:].view(T, H, D)
    if layout == "misaligned":
        return torch.randn(T * H * D + 8, device=device).bfloat16()[8:].view(T, H, D)
    if layout == "token_stride_8":
        return (
            torch.randn(T, H * D + 8, device=device)
            .bfloat16()[:, : H * D]
            .view(T, H, D)
        )
    if layout == "head_stride_8":
        return torch.randn(T, H + 1, D + 8, device=device).bfloat16()[:, :H, :D]
    if layout == "last_stride_2":
        return torch.randn(T, H, 2 * D, device=device).bfloat16()[:, :, ::2]
    if layout == "named":
        return _named(_view_layout("contiguous", device))
    if layout == "named_misaligned":
        return _named(_view_layout("misaligned", device))
    if layout == "function_subclass":
        return _view_layout("contiguous", device).as_subclass(_FunctionTensor)
    if layout == "flat":
        return torch.randn(T * H, D, device=device).bfloat16()
    if layout == "float32":
        return torch.randn(T, H, D, device=device)
    raise ValueError(layout)


# Plain BF16 ``[T, H, 128]`` CUDA inputs: the short path of ``_kernel_view``
# reads each in place or copies it (each failed check of the in-place read
# alone: ``misaligned``, ``token_stride_8``, ``head_stride_8``,
# ``last_stride_2``).
SHORT_PATH_LAYOUTS = [
    "contiguous",
    "projection",
    "head_major",
    "one_token",
    "one_head",
    "offset_16",
    "misaligned",
    "token_stride_8",
    "head_stride_8",
    "last_stride_2",
]


@requires_device
@pytest.mark.parametrize("layout", SHORT_PATH_LAYOUTS + ["flat"])
def test_kernel_view_short_path_matches_every_check(layout) -> None:
    """The short path of ``_kernel_view`` (a plain BF16 ``[T, H, 128]`` tensor
    on its device, read in place) returns what every check
    (``_kernel_view_checked``) returns: the same tensor read in place, or a
    copy with the same values, and the same strides."""
    device = torch.device("cuda", torch.cuda.current_device())
    t = _view_layout(layout, device)
    view, tok, head = backend._kernel_view(t, "x", shape=t.shape, device=device)
    want, want_tok, want_head = backend._kernel_view_checked(
        t, "x", shape=t.shape, device=device
    )
    assert (tok, head) == (want_tok, want_head)
    assert (view.data_ptr() == t.data_ptr()) == (want.data_ptr() == t.data_ptr())
    assert view.shape == want.shape and view.stride() == want.stride()
    assert torch.equal(view, want)


# ``_kernel_view`` inputs every check refuses (a ValueError): another dtype,
# shape, device type or device index, a short last dimension.
KERNEL_VIEW_REFUSALS = ["float32", "other_shape", "cpu", "other_index", "one_dim_short"]


def _kernel_view_case(
    case: str, device: torch.device
) -> tuple[torch.Tensor, torch.Size, torch.device]:
    """``(input, shape, device)`` of a ``_kernel_view`` case off the short path
    (or ``named``, a contiguous input with dimension names)."""
    t = _view_layout("contiguous", device)
    if case == "other_shape":
        return t, torch.Size((5, 4, D)), device
    if case == "cpu":
        return t.cpu(), t.shape, device
    if case == "other_index":
        # The checks only compare the device: no tensor lives on it.
        return t, t.shape, torch.device("cuda", device.index + 1)
    if case == "one_dim_short":
        t = t[:, :, : D // 2]
    elif case == "subclass":
        t = torch.nn.Parameter(t, requires_grad=False)
    elif case != "contiguous":
        t = _view_layout(case, device)
    return t, t.shape, device


@requires_device
@pytest.mark.parametrize(
    "case",
    KERNEL_VIEW_REFUSALS
    + ["subclass", "function_subclass", "flat", "named", "named_misaligned"],
)
def test_kernel_view_short_path_refuses_as_every_check(case) -> None:
    """An input the short path refuses gets the error of every check
    (``_kernel_view_checked``); a subclass, a flat input and an input with
    dimension names get its result or its error (an input with names that
    needs a copy raises as in 0.2.8)."""
    t, shape, device = _kernel_view_case(
        case, torch.device("cuda", torch.cuda.current_device())
    )
    outcomes = []
    for view_of in (backend._kernel_view, backend._kernel_view_checked):
        try:
            view, tok, head = view_of(t, "x", shape=shape, device=device)
        except (TypeError, ValueError, RuntimeError) as error:
            outcomes.append((type(error), str(error)))
        else:
            outcomes.append(
                (
                    view.data_ptr() == t.data_ptr(),
                    tok,
                    head,
                    type(view),
                    getattr(view, "names", None),
                    view.stride(),
                )
            )
    assert outcomes[0] == outcomes[1]
    if case in KERNEL_VIEW_REFUSALS:
        assert outcomes[0][0] is ValueError


@requires_device
def test_kernel_view_runs_every_check_only_off_the_short_path(monkeypatch) -> None:
    """``_kernel_view`` runs every check (``_kernel_view_checked``) for no
    plain BF16 ``[T, H, 128]`` CUDA input on its device, read in place (with
    dimension names or without) or copied, and for every other input:
    another dtype, shape, device or type, a flat input, and an input with
    dimension names that needs a copy."""
    device = torch.device("cuda", torch.cuda.current_device())
    checked = backend._kernel_view_checked
    calls = []

    def record(tensor, name, *, shape, device):
        calls.append(name)
        return checked(tensor, name, shape=shape, device=device)

    monkeypatch.setattr(backend, "_kernel_view_checked", record)
    named = ["named"] if NAMED_TENSORS else []
    for layout in SHORT_PATH_LAYOUTS + named:
        t = _view_layout(layout, device)
        backend._kernel_view(t, layout, shape=t.shape, device=device)
    assert calls == []
    others = KERNEL_VIEW_REFUSALS + ["subclass", "function_subclass", "flat"]
    others += ["named_misaligned"] if NAMED_TENSORS else []
    for case in others:
        t, shape, on = _kernel_view_case(case, device)
        try:
            backend._kernel_view(t, case, shape=shape, device=on)
        except (TypeError, ValueError, RuntimeError):
            pass
    assert calls == others


@requires_device
@pytest.mark.parametrize(
    "like",
    [
        "bfloat16",
        "float32",
        "none",
        "projection",
        "head_major",
        "function_subclass",
        "named",
    ],
)
def test_new_output_buffer_is_contiguous_and_aligned(like) -> None:
    """A new output buffer is a plain contiguous 32-byte aligned BF16 tensor of
    the input's shape on its device without dimension names (as
    ``torch.empty`` gives it), whatever ``like`` is."""
    device = torch.device("cuda", torch.cuda.current_device())
    x = _view_layout("contiguous", device)
    if like == "bfloat16":
        ref = x
    elif like == "float32":
        ref = x.float()
    elif like == "none":
        ref = None
    else:
        ref = _view_layout(like, device)
    out, user = backend._output(None, "grad_x", shape=x.shape, device=device, like=ref)
    assert user is None
    assert type(out) is torch.Tensor and out.dtype == torch.bfloat16
    assert out.shape == x.shape and out.device == device and out.is_contiguous()
    assert out.data_ptr() % (2 * loader.ALIGNMENT_ELEMENTS) == 0
    if NAMED_TENSORS:
        assert out.names == (None,) * out.dim()


@requires_device
def test_launch_placeholder_converts_a_legacy_dlpack_capsule(monkeypatch) -> None:
    """The empty buffer a launch does not read is converted once to a
    ``tvm_ffi.Tensor`` from a legacy DLPack capsule (no stream exchange), not
    from the torch tensor, and reused."""
    index = torch.cuda.current_device()
    convert = backend.tvm_ffi.from_dlpack
    seen = []

    def record(value, *args, **kwargs):
        seen.append(type(value).__name__)
        return convert(value, *args, **kwargs)

    monkeypatch.setattr(backend.tvm_ffi, "from_dlpack", record)
    backend._launch_placeholder.cache_clear()
    try:
        held = backend._launch_placeholder(index, torch.bfloat16)
        assert backend._launch_placeholder(index, torch.bfloat16) is held
    finally:
        backend._launch_placeholder.cache_clear()
    assert seen == ["PyCapsule"]
    assert isinstance(held, backend.tvm_ffi.Tensor)


# The modules that warm rows in L2, by role: the mid and the extra-long
# forward (row blocks of packed rows; one row past the start of their
# ``STRIDED_FORWARD_BANDS`` band, where the one-head layouts with packed rows
# run the forward of that row count's ``FORWARD_BANDS`` band), the
# row-claims mid backward (sm_100a and sm_103a: thread 0 warms the next block
# of packed rows), the claimed backward (on sm_100a and sm_103a a CTA's
# next block, with a bulk prefetch per tensor where the rows are packed; on
# sm_107a the rows of the block two ahead); and the fused backward (every
# architecture: thread 0 warms the next block of a claim with a bulk prefetch
# per tensor where the rows are packed, else each thread its own row), whose
# bits must not depend on the layout either; so do the repeatable backward
# and its long role (which runs the packed layouts only; a layout whose rows
# do not follow one another runs the repeatable backward, the same bits),
# of either weight gradient.
# The one-head inputs take one row past the start of each one's band
# (``_start_rows``); a module that no band of this device runs (a forward
# module: no ``STRIDED_FORWARD_BANDS`` band, the table of other layouts; the
# one-head layout whose rows do not follow one another is of the whole-token
# class, whose ``SLICED_FORWARD_BANDS`` run the same modules as
# ``STRIDED_FORWARD_BANDS`` up to 700 rows per SM) skips.
ONE_HEAD_ROUTES = [
    pytest.param("fwd", "fwd_mid", id="fwd_mid"),
    pytest.param("fwd", "fwd_xlong", id="fwd_xlong"),
    pytest.param("bwd_bf16", "bwd_bf16_rc_mid", id="bwd_bf16_rc_mid"),
    pytest.param("bwd_bf16", "bwd_bf16_mid", id="bwd_bf16_claimed"),
    pytest.param("bwd_bf16", "bwd_bf16_fused", id="bwd_bf16_fused"),
    pytest.param("bwd_bf16", "bwd_bf16_rep", id="bwd_bf16_rep"),
    pytest.param("bwd_bf16", "bwd_bf16_rep_long", id="bwd_bf16_rep_long"),
    pytest.param("bwd_f32", "bwd_f32_rep", id="bwd_f32_rep"),
    pytest.param("bwd_f32", "bwd_f32_rep_long", id="bwd_f32_rep_long"),
]


@requires_device
@pytest.mark.parametrize("role,band", ONE_HEAD_ROUTES)
def test_one_head_inputs_give_the_same_bits_for_every_layout(
    role, band, determinism_setting, monkeypatch: pytest.MonkeyPatch, record_property
) -> None:
    """One row past the start of the mid and the extra-long forward (their
    strided bands) and of the row-claims mid, the claimed, the fused and the
    repeatable backwards (of either weight gradient) and the long roles'
    packed rows: a one-head input of
    every layout (``ONE_HEAD_LAYOUTS``) launches ``module_for_rows``' module
    (the module of its layout's packing and, for the forward, its rows'
    class, :func:`_forward_class`) with head stride 128, passes the
    packed-rows test exactly when its rows follow one another (or the host
    copied it), and returns the same ``y``, or ``dx``, ``dz`` and ``dw``,
    bits for every layout launching the same module and every repeat (but a
    BF16 ``dw`` of the fused and the row-claims mid backwards,
    ``suite.assert_dw_repeats``), with the workspace zero after every call.
    On the inputs of every draw of ``judge`` (the first draw's repeated, the
    others' once), every layout's outputs are judged against the naive
    implementation."""
    arch = loader.device_arch(torch.cuda.current_device())
    name = loader.ROUTES[arch].get(band)
    if name is None:
        pytest.skip(f"{arch} routes no {band} module")
    sms = _sm_count()
    if role != "fwd" and name not in {
        module
        for _, module, _, _ in _band_modules(arch, role, sms)
        + _long_band_modules(arch, role, sms)
    }:
        pytest.skip(f"no band of {arch} on {sms} SMs runs {band}")
    strided_bands = {b for _, b in loader.STRIDED_FORWARD_BANDS[arch]}
    if role == "fwd" and band not in strided_bands:
        pytest.skip(f"no band of {arch} of other layouts runs {band}")
    strided = role == "fwd"
    rows = _start_rows(arch, name, sms, packed=not strided) + 1
    assert loader.module_for_rows(arch, role, rows, sms, packed=not strided) == name
    dw_dtype = torch.float32 if role == "bwd_f32" else torch.bfloat16
    names = ("y",) if role == "fwd" else ("dx", "dz", "dw")
    varies = role != "fwd" and suite._varies(name)
    ws = nd_workspace(rows, torch.device("cuda"))
    launched = _spy_strides(monkeypatch)
    modules = set()

    def run(j, s):
        x0, z0, w, dy0 = suite.make(rows, 1, layout="flat", seed=s)
        calls = (2 if role == "fwd" else suite.REPEATS) if j == 0 else 1
        wants: dict[str, tuple] = {}
        launches = []
        for layout, (packed, _) in ONE_HEAD_LAYOUTS.items():
            x, z, dy = (_one_head(t, layout) for t in (x0, z0, dy0))
            # the forward's class of the rows (a backward reads no class)
            rows_packed, layout_class = _forward_class(x, z)
            assert rows_packed == packed, (layout, layout_class)
            module = loader.module_for_rows(
                arch, role, rows, sms, packed=packed, layout=layout_class
            )
            modules.add(module)
            want = wants.get(module)
            launched.clear()
            for _ in range(calls):
                if role == "fwd":
                    got = (nd_forward(x, z, w, EPS).reshape(rows, D),)
                else:
                    dw = torch.empty(D, dtype=dw_dtype, device=x0.device)
                    dx, dz, dw = nd_backward(
                        dy, x, z, w, EPS, grad_weight=dw, workspace=ws
                    )
                    got = (dx.reshape(rows, D), dz.reshape(rows, D), dw)
                    assert not bool(ws.any()), (layout, "the workspace is not zero")
                launches.append(dict(zip(names, got, strict=True)))
                if want is None:
                    want = wants[module] = got
                    continue
                for what, g, w_ in zip(names, got, want, strict=True):
                    if what == "dw":
                        suite.assert_dw_repeats(
                            g, w_, varies=varies, what=f"{layout}: dw"
                        )
                    else:
                        assert_same_bits(f"{layout}: {what}", g, w_)
            assert [n for n, _ in launched] == [module] * calls, (layout, launched)
            for _, strides in launched:
                assert strides["heads"] == 1, (layout, strides)
                heads = [v for k, v in strides.items() if k.endswith("_head")]
                assert heads and all(v == D for v in heads), (layout, strides)
                assert _packed(strides) == packed, (layout, strides)
        flat = [t.reshape(rows, D) for t in (x0, z0, dy0)]
        return silu_draw(
            flat[0],
            flat[1],
            w,
            None if role == "fwd" else flat[2],
            EPS,
            launches,
            dict(zip(names, names, strict=True)),
        )

    judge(record_property, run, seed=25)
    assert name in modules, (name, sorted(modules))
