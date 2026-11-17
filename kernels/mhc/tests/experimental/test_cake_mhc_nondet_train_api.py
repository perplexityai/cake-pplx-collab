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

Tests of ``cake_pplx.cake_mhc_nondet_train``, the stable top-level module of
the Cake four-stream mHC training kernels' package
``cake_pplx._backends.cake_mhc_nondet_train`` (SM100 / SM103 / SM107):
its entry points emit no warning (the ``cake_api`` decorator emits none).

The mode is part of the names: every entry point runs the package
``cake_pplx._backends.cake_mhc_nondet_train`` (imported on the first
call or attribute access), no entry point takes a ``deterministic`` keyword
(passing one is a ``TypeError``), and nothing falls back to the deterministic
package.  With PyTorch's deterministic algorithms on, every entry point, the
backward included, runs without an alert.  The package makes no bit promise:
its outputs are judged with ``tests/cake_naive.py`` against the naive BF16 tier
over 32 inputs (the error against the float64 oracle of
``tests/cake/test_helpers/cake_mhc_naive.py``, in the L2 and the max norm,
median and max, at most 1.005 times the tier's), never compared bit for bit
with another call or with the deterministic package.  This module checks:

* the surface without a device: the exports (no head name; every name but the
  ALL_CAPS constants carries the ``nondet`` mode token and no other), the
  backend objects resolved on first access, the signatures (the deterministic
  package's; no ``deterministic`` parameter on any public callable of the
  module or of the backend, the autograd ``Function.forward`` included), the
  ``TypeError`` of a ``deterministic`` keyword on every entry point, the
  workspace helpers' ``hidden`` default equal to the backend's
  ``cake_jit.HIDDEN``, the lazy import (in a fresh interpreter ``import
  cake_pplx`` and ``import cake_pplx.cake_mhc_nondet_train`` load no backend,
  and the namespace exposes no ``nondet`` name), the former module paths
  failing at import, no import of the deterministic package by this module and
  none of the former fallback's state, each entry point's forwarding to the
  backend's (replaced by a recorder) with every argument unchanged, and the
  ``TypeError`` of ``cake_mhc_nondet_train_finish`` for any other handle
  (naming the deterministic finish);
* on a device, at every hidden size the device's architecture routes (4096,
  5120 and 7168 on SM100 / SM103 / SM107): the module, the
  deterministic package's launches refused, gives the backend's workspace sizes
  and judged results (the forward entry points, the backward, the autograd
  functions);
  under ``torch.use_deterministic_algorithms(True)``
  ``is_cake_mhc_nondet_train_supported``, the workspace helpers, the forward
  entry points, ``backward_prepare``, ``backward_finish`` and the prepare
  node's backward run without an alert or a warning and give judged results
  (with ``warn_only``, ``backward_finish`` gives no alert); a deterministic
  backward workspace (smaller than this package's layout) is refused; a
  deterministic prepare handle is refused by ``cake_mhc_nondet_train_finish``
  (the deterministic finish then gives the deterministic package's bits, run
  to run); on every architecture a hidden size runs exactly where the
  deterministic package runs it.

Training with this package composed with the model's deterministic head
(``cake_pplx.cake_mhc_head_det_train``, family ``mhc_head``) is checked in
``tests/cross_family/mhc/tests/experimental/test_cake_mhc_nondet_train_head.py``.

The numerical contract of the package's backward is
``test_cake_mhc_nondet_train.py``'s.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import cake_pplx.cake_mhc_det_train as det_facade
import cake_pplx.cake_mhc_nondet_train as facade
import cake_pplx._backends.cake_mhc_det_train as det_backend
import cake_pplx._backends.cake_mhc_nondet_train as backend
from cake_pplx._backends.cake_mhc_det_train import cake_jit as det_loader
from cake_pplx._backends.cake_mhc_nondet_train import cake_jit as loader
from tests.cake.test_helpers.cake_mhc_naive import (
    OUTPUTS,
    expected_and_naive,
    judge_draws,
    package_outputs,
    stats_outputs,
)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# The hidden sizes of both packages; a device case of a size runs where the device's architecture routes it.
HIDDEN_SIZES = (4096, 5120, 7168)
REPO = Path(__file__).resolve().parents[2]
SEED = 2032
BF16 = torch.bfloat16
# The zero prefix every backward workspace starts with: the ticket region and the column slab's arrival counts (zero
# again after every completed launch).
ZERO_PREFIX_BYTES = 4 * (loader.TICKET_WORDS + loader.SLAB_CNT_WORDS)
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
_LEAVES = ("X", "fn", "scale", "base")
# PyTorch's wording of the alert of a non-deterministic operation (no call of this package raises it).
_ALERT = "does not have a deterministic implementation"
# The interpreter's refusal of a ``deterministic`` keyword: no entry point has that parameter.
_NO_MODE_KEYWORD = "unexpected keyword argument 'deterministic'"
# facade entry point -> (the deterministic package's entry point of the same verb, positional stand-ins: a keyword the
# function does not take is refused before its body runs, so no stand-in is read)
_ENTRIES = {
    "is_cake_mhc_nondet_train_supported": ("is_cake_mhc_det_train_supported", ("device", 5120)),
    "cake_mhc_nondet_train_forward_workspace_bytes": ("cake_mhc_det_train_forward_workspace_bytes", (8, "device")),
    "cake_mhc_nondet_train_backward_workspace_bytes": ("cake_mhc_det_train_backward_workspace_bytes", (8, "device")),
    "cake_mhc_nondet_train_forward_workspace": ("cake_mhc_det_train_forward_workspace", (8, "device")),
    "cake_mhc_nondet_train_backward_workspace": ("cake_mhc_det_train_backward_workspace", (8, "device")),
    "cake_mhc_nondet_train_forward_prepare": ("cake_mhc_det_train_forward_prepare", ("X", "fn", "scale", "base")),
    "cake_mhc_nondet_train_forward_stats": ("cake_mhc_det_train_forward_stats", ("X", "fn", "scale", "base")),
    "cake_mhc_nondet_train_forward_finish": ("cake_mhc_det_train_forward_finish", ("X", "B", "stats")),
    "cake_mhc_nondet_train_backward_prepare": ("cake_mhc_det_train_backward_prepare", ("dY", "B", "X", "stats")),
    "cake_mhc_nondet_train_backward_finish": (
        "cake_mhc_det_train_backward_finish",
        ("X", "dY", "dU", "B", "fn", "scale", "base", "stats", "grad_stats"),
    ),
    "cake_mhc_nondet_train_prepare": ("cake_mhc_det_train_prepare", ("X", "fn", "scale", "base")),
    "cake_mhc_nondet_train_finish": ("cake_mhc_det_train_finish", ("X", "B", "handle")),
}  # fmt: skip
# The backend objects the facade resolves on first access (the backend's own objects).
_REEXPORTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCNondetFinishFunction",
    "CakeMHCNondetPrepareFunction",
    "cake_mhc_nondet_train_reset_workspace",
    "cake_mhc_nondet_train_route_key",
)
_WORKSPACE_HELPERS = (
    "cake_mhc_nondet_train_forward_workspace_bytes",
    "cake_mhc_nondet_train_backward_workspace_bytes",
    "cake_mhc_nondet_train_forward_workspace",
    "cake_mhc_nondet_train_backward_workspace",
)
# PyTorch's deterministic algorithms as ``(mode, warn_only)``: off, strict, warn_only.
_TORCH_MODES = ((False, False), (True, False), (True, True))
# The state of the former mode keyword and of the former fallback to the deterministic package: the mode check, the
# once-per-process warning, the mode switch and the companion backward workspaces.
_REMOVED = (
    "_require_nondeterministic",
    "_DET_FALLBACK_WARNED",
    "_deterministic",
    "_COMPANIONS",
    "_forget_companions",
    "_fallback_backward_workspace",
)
_HANDLE_ERROR = (
    "handle must be the handle returned by cake_mhc_nondet_train_prepare (a cake_mhc_det_train_prepare "
    "handle goes to cake_pplx.cake_mhc_det_train.cake_mhc_det_train_finish)"
)
_BACKEND = "cake_pplx._backends.cake_mhc_nondet_train"
# The former module paths of the non-deterministic mode (and of the earlier names): removed, with no alias.
_FORMER_MODULES = (
    "cake_pplx._backends.cake_mhc_train",
    "cake_pplx._backends.cake_mhc_train_backend",
    "cake_pplx.cake_mhc_train",
    "cake_pplx.cake_mhc_train_backend",
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and det_facade.is_cake_mhc_det_train_supported(_device())
        and facade.is_cake_mhc_nondet_train_supported(_device())
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_nondet_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries of both packages that have no prebuilt file through the spec objects
    # the loaders build on first use; a prebuilt or cached library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
            spec.build_and_load()
    yield


@contextlib.contextmanager
def _torch_deterministic(mode: bool, *, warn_only: bool = False):
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


@pytest.fixture
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms``; the process-wide flags are restored however the test ends."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield torch.use_deterministic_algorithms
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _shape(parameters) -> list:
    """Name, kind and default of each parameter (the annotations are strings under postponed evaluation)."""
    return [(p.name, p.kind, p.default) for p in parameters]


def _exactly(message: str) -> str:
    """A ``pytest.raises`` pattern that matches ``message`` and nothing else."""
    return f"^{re.escape(message)}$"


def _mode_tokens(name: str) -> set[str]:
    """The mode words of ``name``: ``det`` / ``nondet`` as ``_``-separated tokens of a snake_case name, ``Det`` /
    ``Nondet`` as words of a CamelCase one (a substring test would read ``nondet`` as ``det`` too)."""
    if name[:1].isupper():
        return {word.lower() for word in re.findall(r"(Nondet|Det)(?=[A-Z]|$)", name)}
    return set(re.findall(r"(?:^|_)(det|nondet)(?=_|$)", name))


def _public_callables(module) -> dict:
    """``name -> callable`` of every public callable of ``module.__all__``: its functions, and the ``forward`` /
    ``backward`` of its autograd ``Function`` classes."""
    found = {}
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj):
            for method in ("forward", "backward"):
                found[f"{name}.{method}"] = getattr(obj, method)
        elif callable(obj):
            found[name] = obj
    return found


def _refuse_deterministic_package(m) -> None:
    """Every entry point of the deterministic package (its facade's and its backend's) fails the test when called."""
    for det_name, _stand_ins in _ENTRIES.values():

        def refuse(*_args, _name=det_name, **_kwargs):
            raise AssertionError(f"the deterministic package's {_name} was called")

        for module in (det_facade, det_backend):
            m.setattr(module, det_name, refuse)


def _run_child(code: str, *args: str) -> dict:
    """``code`` in a fresh interpreter (whatever this session imported does not count); its last line of output as
    JSON."""
    proc = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ----------------------------------------------------------------------------------------------------------------
# The surface (no device)


@pytest.mark.cpu_only
def test_exports() -> None:
    """``__all__``: the backend objects resolved on first access (the backend's own objects, listed by ``dir``) and
    the module's own entry points (``def``s of this module, not the backend's objects); no public head name; every
    name but the ALL_CAPS constants carries the mode token ``nondet`` (``Nondet`` in a class name) and no ``det``
    token."""
    expected = set(_ENTRIES) | set(_REEXPORTS)
    assert set(facade.__all__) == expected, sorted(set(facade.__all__) ^ expected)
    for name in _REEXPORTS:
        assert getattr(facade, name) is getattr(backend, name), name
        assert name in dir(facade), name
    for name in _ENTRIES:
        function = getattr(facade, name)
        assert inspect.unwrap(function).__module__ == facade.__name__, name
        assert function is not getattr(backend, name), name
    for name in facade.__all__:
        assert "head" not in name.lower(), name
        if not name.isupper():
            assert _mode_tokens(name) == {"nondet"}, name
    for name in (n for n in dir(facade) if not n.startswith("_")):
        assert "det" not in _mode_tokens(name), name
    with pytest.raises(AttributeError):
        facade.route_key  # noqa: B018 - the former name is gone
    assert facade.SUPPORTED_COMPUTE_CAPABILITIES == SUPPORTED_CAPABILITIES


@pytest.mark.cpu_only
def test_signatures_mirror_the_deterministic_facade() -> None:
    """Every entry point, ``cake_mhc_nondet_train_finish`` included, has the parameters of the deterministic entry
    point of the same verb (names, kinds and defaults, in order; an exported name of
    ``cake_pplx.cake_mhc_det_train``), and the backend's entry point of its name has them too.  No public callable
    of the module or of the backend has a ``deterministic`` parameter, the autograd ``Function.forward`` /
    ``backward`` of the backend's nodes included."""
    for name, (det_name, _stand_ins) in _ENTRIES.items():
        assert det_name in det_facade.__all__, det_name
        parameters = _shape(
            inspect.signature(getattr(facade, name)).parameters.values()
        )
        det_parameters = _shape(
            inspect.signature(getattr(det_facade, det_name)).parameters.values()
        )
        backend_parameters = _shape(
            inspect.signature(getattr(backend, name)).parameters.values()
        )
        assert parameters == det_parameters, name
        assert backend_parameters == parameters, name
    for module in (facade, backend):
        callables = _public_callables(module)
        assert "CakeMHCNondetPrepareFunction.forward" in callables, module.__name__
        for name, function in callables.items():
            parameters = inspect.signature(function).parameters
            assert "deterministic" not in parameters, (module.__name__, name)


@pytest.mark.cpu_only
@pytest.mark.parametrize("value", (False, True, None), ids=repr)
def test_deterministic_keyword_is_a_type_error(value, monkeypatch) -> None:
    """A ``deterministic`` keyword, whatever its value, is the interpreter's ``TypeError`` (an unexpected keyword
    argument) on every entry point of the module and of the backend, ``cake_mhc_nondet_train_reset_workspace``
    included, before anything else (no stand-in argument is read, the deterministic package is not called), with
    PyTorch's deterministic algorithms off, strict and ``warn_only`` alike (no warning, no alert)."""
    _refuse_deterministic_package(monkeypatch)
    for torch_mode, warn_only in _TORCH_MODES:
        with (
            _torch_deterministic(torch_mode, warn_only=warn_only),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            for name, (_det_name, args) in _ENTRIES.items():
                for module in (facade, backend):
                    with pytest.raises(TypeError, match=_NO_MODE_KEYWORD):
                        getattr(module, name)(*args, deterministic=value)
            with pytest.raises(TypeError, match=_NO_MODE_KEYWORD):
                facade.cake_mhc_nondet_train_reset_workspace(
                    "workspace", deterministic=value
                )


@pytest.mark.cpu_only
def test_hidden_default_is_the_backend_hidden() -> None:
    """The workspace helpers' ``hidden`` default, the module's ``_HIDDEN`` (spelled there so that defining the entry
    points imports no backend), is the backend's ``cake_jit.HIDDEN``."""
    assert facade._HIDDEN == loader.HIDDEN == 4096
    for name in _WORKSPACE_HELPERS:
        for module in (facade, backend):
            default = inspect.signature(getattr(module, name)).parameters["hidden"]
            assert default.default == loader.HIDDEN, (module.__name__, name)


def _imports(path: Path) -> list[str]:
    """The modules the imports of ``path`` name, and each imported name qualified by its module (a relative module
    without its leading dots)."""
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


@pytest.mark.cpu_only
def test_top_level_does_not_import_the_module() -> None:
    """``cake_pplx`` and ``cake_pplx._backends`` are PEP 420 namespace packages: no ``__init__.py`` of this family
    imports this module or a backend package, and the imported namespace has no file."""
    import cake_pplx
    import cake_pplx._backends

    for level in ("cake_pplx", "cake_pplx/_backends"):
        assert not (REPO / level / "__init__.py").exists(), level
    for namespace in (cake_pplx, cake_pplx._backends):
        assert getattr(namespace, "__file__", None) is None, namespace


_LAZY_CHILD = r"""
import json, sys

backend = sys.argv[1]


def loaded():
    return sorted(m for m in sys.modules if m.startswith("cake_pplx._backends."))


import cake_pplx

out = {
    "namespace": loaded(),
    "facade_loaded": "cake_pplx.cake_mhc_nondet_train" in sys.modules,
    "mhc_nondet_names": sorted(n for n in dir(cake_pplx) if "mhc" in n.lower() and "nondet" in n.lower()),
}
import cake_pplx.cake_mhc_nondet_train as facade

out["facade"] = loaded()
facade.cake_mhc_nondet_train_backward_finish
out["entry_point"] = loaded()
facade.ARCHES
out["lazy_attribute"] = loaded()
out["backend"] = backend in sys.modules
print(json.dumps(out))
"""


@pytest.mark.cpu_only
def test_the_backend_is_imported_on_first_use() -> None:
    """In a fresh interpreter, ``import cake_pplx`` loads no backend package and not this module, and the namespace
    exposes no mHC ``nondet`` name; ``import
    cake_pplx.cake_mhc_nondet_train`` and reading an entry point load no backend either; the first lazily resolved
    attribute (``ARCHES``) imports the backend."""
    out = _run_child(_LAZY_CHILD, _BACKEND)
    assert out["namespace"] == [], out
    assert out["facade_loaded"] is False, out
    assert out["mhc_nondet_names"] == [], out
    assert out["facade"] == [] and out["entry_point"] == [], out
    assert _BACKEND in out["lazy_attribute"] and out["backend"] is True, out


_FORMER_CHILD = r"""
import importlib, json, sys

out = {}
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
    except ImportError:
        out[name] = None
    else:
        # a namespace package left by a stale __pycache__ directory has no origin: it holds no code
        spec = getattr(module, "__spec__", None)
        out[name] = getattr(spec, "origin", None)
print(json.dumps(out))
"""


@pytest.mark.cpu_only
def test_former_module_paths_fail_at_import() -> None:
    """The former module paths of the non-deterministic mode (``cake_pplx._backends.cake_mhc_train`` and its
    ``_backend``) and of the earlier names are gone, with no alias: importing each in a fresh interpreter raises
    ``ImportError`` (a namespace package without code, left by a stale ``__pycache__``, counts as gone)."""
    out = _run_child(_FORMER_CHILD, *_FORMER_MODULES)
    assert out == dict.fromkeys(_FORMER_MODULES), out


@pytest.mark.cpu_only
def test_module_imports_only_the_backend_on_first_use() -> None:
    """The module's source imports nothing of the deterministic package (no imported module or name contains
    ``cake_mhc_det_train``), imports the backend only inside a function (no module-level import of
    ``cake_pplx._backends``), and defines and reads none of the former mode keyword's or fallback's names (the
    mode check, the once-per-process warning flag, the mode switch, the companion workspaces)."""
    path = Path(facade.__file__)
    det_imports = [m for m in _imports(path) if "cake_mhc_det_train" in m]
    assert not det_imports, det_imports
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert "experimental" not in (node.module or ""), node.lineno
        elif isinstance(node, ast.Import):
            assert not [a for a in node.names if "experimental" in a.name], node.lineno
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)
    left = sorted(names & set(_REMOVED))
    assert not left, left
    for name in _REMOVED:
        assert not hasattr(backend, name), name


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(_ENTRIES))
def test_forwards_to_the_backend(name: str, monkeypatch) -> None:
    """With the backend's entry point replaced by a recorder and the deterministic package's refused, a call
    (stand-ins for every positional and keyword-only argument) calls the backend's entry point of the same name once
    with every argument unchanged, with PyTorch's deterministic algorithms off, strict and ``warn_only`` alike (no
    warning, no alert)."""
    function = getattr(facade, name)
    target = inspect.signature(getattr(backend, name))
    _det_name, args = _ENTRIES[name]
    keywords = {
        p.name: f"<{p.name}>"
        for p in inspect.signature(function).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    want = target.bind(*args, **keywords)
    want.apply_defaults()
    calls: list = []

    def record(*a, **k):
        calls.append(target.bind(*a, **k))
        return "nd"

    monkeypatch.setattr(backend, name, record)
    _refuse_deterministic_package(monkeypatch)
    for torch_mode, warn_only in _TORCH_MODES:
        calls.clear()
        with (
            _torch_deterministic(torch_mode, warn_only=warn_only),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            assert function(*args, **keywords) == "nd"
        assert len(calls) == 1, (torch_mode, warn_only)
        calls[0].apply_defaults()
        assert calls[0].arguments == want.arguments, (torch_mode, warn_only)


@pytest.mark.cpu_only
def test_finish_refuses_other_handles() -> None:
    """``cake_mhc_nondet_train_finish`` refuses every handle but its prepare's -- the deterministic package's handle
    included -- with one ``TypeError`` naming the deterministic finish."""
    det_handle = det_backend._MHCTrainHandle(None, ())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for handle in (det_handle, object(), None, (None, ())):
            with pytest.raises(TypeError, match=_exactly(_HANDLE_ERROR)):
                facade.cake_mhc_nondet_train_finish("X", "B", handle)


# ----------------------------------------------------------------------------------------------------------------
# On a device


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture routes every kind of (this package's sizes)."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a device case (skipped where the architecture has no route of it)."""
    if request.param not in _hidden_sizes():
        pytest.skip(
            f"hidden {request.param} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return request.param


def _inputs(T: int, seed: int = SEED, hidden: int = 4096) -> dict[str, torch.Tensor]:
    """A case's inputs and this package's forward records of them (``stats``, ``gstats``)."""
    generator = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=_device(), dtype=torch.float32
        ).to(dtype)

    i = {
        "X": randn(T, 4, hidden, dtype=BF16),
        "fn": (randn(24, 4 * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(24),
        "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, 4, hidden, dtype=BF16),
        "dU": randn(T, hidden, dtype=BF16),
    }
    _U, i["stats"] = facade.cake_mhc_nondet_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, i["gstats"] = facade.cake_mhc_nondet_train_backward_prepare(
        i["dY"], i["B"], i["X"], i["stats"]
    )
    return i


_ARG_KEYS = ("X", "dY", "dU", "B", "fn", "scale", "base", "stats", "gstats")


def _args(i: dict) -> tuple:
    return tuple(i[k] for k in _ARG_KEYS)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8),
        b.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _assert_zero_prefix(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:ZERO_PREFIX_BYTES]))
    assert nonzero == 0, (
        f"{what}: {nonzero} non-zero bytes in the ticket region and the arrival counts"
    )


_FORWARD_ENTRIES = (
    "forward_prepare",
    "forward_stats",
    "forward_finish",
    "backward_prepare",
)


def _entries(module, prefix: str = "cake_mhc_nondet_train") -> dict:
    """The forward entry points and ``backward_prepare`` of ``module`` by short name."""
    return {k: getattr(module, f"{prefix}_{k}") for k in _FORWARD_ENTRIES}


def _forward_outputs(entries: dict, i: dict) -> dict:
    X, fn, scale, base = (i[k] for k in _LEAVES)
    U, stats = entries["forward_prepare"](X, fn, scale, base)
    dB, gstats = entries["backward_prepare"](i["dY"], i["B"], X, stats)
    return {
        "U": U.clone(),
        "stats": stats.clone(),
        "replay": entries["forward_stats"](X, fn, scale, base).clone(),
        "Y": entries["forward_finish"](X, i["B"], stats).clone(),
        "dB": dB.clone(),
        "gstats": gstats.clone(),
    }


def _nd_prepare(*args, **kwargs):
    return facade.cake_mhc_nondet_train_prepare(*args, **kwargs)


def _nd_finish(*args, **kwargs):
    return facade.cake_mhc_nondet_train_finish(*args, **kwargs)


def _graph(prepare, finish, i: dict, **kwargs) -> tuple:
    """``(leaves, handle, Y, loss)`` of ``Y = finish(X, U + B, prepare(X, fn, scale, base))`` and the loss
    ``sum(Y * dY)``, not yet differentiated; ``kwargs`` go to ``prepare``."""
    leaves = [i[k].clone().requires_grad_(True) for k in _LEAVES]
    U, handle = prepare(*leaves, **kwargs)
    Y = finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
    return leaves, handle, Y, (Y.float() * i["dY"].float()).sum()


def _autograd(prepare, finish, i: dict, **kwargs) -> tuple:
    """``(handle, Y, [X.grad, fn.grad, scale.grad, base.grad])`` of :func:`_graph` after its loss's backward."""
    leaves, handle, Y, loss = _graph(prepare, finish, i, **kwargs)
    loss.backward()
    return handle, Y.detach(), [t.grad for t in leaves]


def _connection(i: dict, **kwargs) -> dict:
    """This package's residual connection ``Y = finish(X, U + B, prepare(X, fn, scale, base))`` and its loss
    ``sum(Y * dY)``, not yet differentiated (``U``'s gradient retained); ``kwargs`` go to the prepare."""
    leaves = [i[k].clone().requires_grad_(True) for k in _LEAVES]
    U, handle = _nd_prepare(*leaves, **kwargs)
    U.retain_grad()
    B = (U.float() + i["B"].float()).to(BF16)
    Y = _nd_finish(leaves[0], B, handle)
    return {
        "leaves": leaves,
        "handle": handle,
        "U": U,
        "B": B,
        "Y": Y,
        "loss": (Y.float() * i["dY"].float()).sum(),
    }


def _connection_part(prefix: str, i: dict, c: dict) -> tuple[dict, dict, dict]:
    """The judged outputs of a differentiated :func:`_connection` ``c`` under ``prefix``: ``U``, ``Y``, ``U``'s
    gradient as ``dB`` (the residual branch hands ``dB`` on as ``dU``) and the gradients of the leaves, with the
    oracle's and the naive tier's values on the branch output and the ``dU`` the graph produced."""
    U, leaves = c["U"], c["leaves"]
    got = {
        "U": U.detach(), "Y": c["Y"].detach(), "dB": U.grad, "dX": leaves[0].grad, "dfn": leaves[1].grad,
        "dscale": leaves[2].grad, "dbase": leaves[3].grad,
    }  # fmt: skip
    e, t = expected_and_naive({**i, "B": c["B"].detach(), "dU": U.grad.detach()}, got)
    return _part(prefix, got, e, t)


def _forward_judged(got: dict) -> dict:
    """The judged outputs of :func:`_forward_outputs`' results (the statistics replay's under ``replay``)."""
    out = package_outputs(got)
    out.update({"replay " + k: v for k, v in stats_outputs(got["replay"]).items()})
    return out


def _backward_judged(got) -> dict:
    """``backward_finish``'s outputs by name."""
    return {k: t.clone() for k, t in zip(_OUTPUTS, got, strict=True)}


def _part(
    prefix: str, got: dict, expected: dict, naive: dict
) -> tuple[dict, dict, dict]:
    """``got`` under the name prefix ``prefix`` with the oracle's and the naive tier's values of its outputs (a
    ``replay <statistic>`` output: the statistic's)."""
    base = {k: k.removeprefix("replay ") for k in got}
    return (
        {prefix + k: v for k, v in got.items()},
        {prefix + k: expected[b] for k, b in base.items()},
        {prefix + k: naive[b] for k, b in base.items()},
    )


def _merge(*parts) -> tuple[list, dict, dict]:
    """One launch of the parts' outputs, with their oracle and naive values (a triple of ``judge_draws``)."""
    outputs, expected, naive = {}, {}, {}
    for o, e, t in parts:
        outputs.update(o)
        expected.update(e)
        naive.update(t)
    return [outputs], expected, naive


def _refuse_launches(m, module) -> None:
    """Every launch of ``module`` (all go through its ``_launch_route`` / ``_run_route``) fails the test."""

    def refuse(*_args, **_kwargs):
        raise AssertionError(f"{module.__name__} launched a kernel")

    m.setattr(module, "_launch_route", refuse)
    m.setattr(module, "_run_route", refuse)


def _workspace_sizes(T: int, hidden: int) -> dict[str, int]:
    """The facade's forward and backward workspace bytes for ``T`` tokens at ``hidden``."""
    return {
        kind: getattr(facade, f"cake_mhc_nondet_train_{kind}_workspace_bytes")(
            T, _device(), hidden=hidden
        )
        for kind in ("forward", "backward")
    }


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
def test_module_runs_the_nondeterministic_backend(
    T: int, hidden: int, monkeypatch, record_property
) -> None:
    """The module, with the deterministic package's launches refused: the backend's workspace sizes (the
    forward one the deterministic package's, the backward one larger: it adds the column slab's arrival counts and
    per-chunk ``dfn`` slots) and zeroed buffers; the autograd prepare makes the backend's handle.  Over 32 inputs,
    the outputs of the forward entry points (the statistics replay's under ``replay``), of ``backward_finish`` on
    the default and on a caller workspace (under ``backward workspace=<bool>``) and of the autograd functions (under
    ``autograd``) are judged against the naive BF16 tier."""
    dev = _device()
    nd_bytes = loader.workspace_bytes_upto("bwd", T, _sm_count(), _arch(), hidden)
    what = f"T={T} H={hidden}"
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        sizes = _workspace_sizes(T, hidden)
        assert (
            sizes["backward"]
            == nd_bytes
            == backend.cake_mhc_nondet_train_backward_workspace_bytes(
                T, dev, hidden=hidden
            )
        ), what
        assert nd_bytes > det_facade.cake_mhc_det_train_backward_workspace_bytes(
            T, dev, hidden=hidden
        ), what
        assert (
            sizes["forward"]
            == loader.workspace_bytes_upto("fwd", T, _sm_count(), _arch(), hidden)
            == det_facade.cake_mhc_det_train_forward_workspace_bytes(
                T, dev, hidden=hidden
            )
        ), what
        ws = facade.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=hidden)
        assert ws.dtype == torch.uint8 and ws.numel() == nd_bytes, what
        assert int(torch.count_nonzero(ws)) == 0, what

        def case(j: int, s: int):
            i = _inputs(T, seed=s, hidden=hidden)
            e, t = expected_and_naive(i, OUTPUTS)
            parts = [
                _part("", _forward_judged(_forward_outputs(_entries(facade), i)), e, t)
            ]
            for workspace in (None, ws):
                got = facade.cake_mhc_nondet_train_backward_finish(
                    *_args(i), workspace=workspace
                )
                parts.append(
                    _part(
                        f"backward workspace={workspace is not None} ",
                        _backward_judged(got),
                        e,
                        t,
                    )
                )
            c = _connection(i)
            c["loss"].backward()
            assert isinstance(c["handle"], backend._MHCTrainHandle), what
            parts.append(_connection_part("autograd ", i, c))
            return _merge(*parts)

        judge_draws(case, record_property, seed=SEED + T)


@requires_device
def test_strict_mode_runs_every_entry_point(
    hidden: int, deterministic_algorithms, monkeypatch, record_property
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, with the deterministic package's launches refused (nothing
    falls back to it), warnings are errors: ``is_cake_mhc_nondet_train_supported``, the workspace helpers, the
    forward entry points, ``backward_prepare``, the prepare / finish forwards, ``backward_finish`` (on a caller
    workspace whose bytes past the zero prefix are 0xAB, with caller outputs) and the backward of a prepare node,
    recorded with the mode on or off, run without an alert or a warning; the support answer and the sizes are those
    of the mode off, the zero prefix is zero after the backward, and over 32 inputs every output and gradient is
    judged against the naive BF16 tier (the oracle and the tier evaluated with the mode off)."""
    T = 129
    dev = _device()
    want_supported = facade.is_cake_mhc_nondet_train_supported(dev, hidden)
    want_sizes = _workspace_sizes(T, hidden)
    ws = facade.cake_mhc_nondet_train_backward_workspace(T, dev, hidden=hidden)
    outputs = {
        "dX": torch.empty((T, 4, hidden), dtype=BF16, device=dev),
        "dfn": torch.empty((24, 4 * hidden), device=dev),
        "dscale": torch.empty((3,), device=dev),
        "dbase": torch.empty((24,), device=dev),
    }
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            supported = facade.is_cake_mhc_nondet_train_supported(dev, hidden)
            sizes = _workspace_sizes(T, hidden)
            buffers = {
                kind: getattr(facade, f"cake_mhc_nondet_train_{kind}_workspace")(
                    T, dev, hidden=hidden
                )
                for kind in ("forward", "backward")
            }
        deterministic_algorithms(False)
    assert supported is True and want_supported is True
    assert sizes == want_sizes, (sizes, want_sizes)
    for kind, buffer in buffers.items():
        assert buffer.dtype == torch.uint8 and buffer.numel() == sizes[kind], kind
        assert int(torch.count_nonzero(buffer)) == 0, kind

    def case(j: int, s: int):
        i = _inputs(T, seed=s, hidden=hidden)
        recorded_off = _connection(i)
        ws[ZERO_PREFIX_BYTES:].fill_(0xAB)
        for v in outputs.values():
            v.fill_(3.0)
        with monkeypatch.context() as m:
            _refuse_launches(m, det_backend)
            deterministic_algorithms(True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    fwd = _forward_outputs(_entries(facade), i)
                    backward = facade.cake_mhc_nondet_train_backward_finish(
                        *_args(i), workspace=ws, **outputs
                    )
                    recorded_on = _connection(i)
                    for c in (recorded_on, recorded_off):
                        c["loss"].backward()
            finally:
                deterministic_algorithms(False)
        _assert_zero_prefix(ws, f"backward_finish under the strict mode, draw {j}")
        e, t = expected_and_naive(i, OUTPUTS)
        return _merge(
            _part("", _forward_judged(fwd), e, t),
            _part("backward ", _backward_judged(backward), e, t),
            _connection_part("recorded on ", i, recorded_on),
            _connection_part("recorded off, strict at backward ", i, recorded_off),
        )

    judge_draws(case, record_property, seed=SEED + 1)


@requires_device
def test_warn_only_runs_the_backend_without_a_warning(
    hidden: int, deterministic_algorithms, monkeypatch, record_property
) -> None:
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)``, with the deterministic package's launches
    refused, ``backward_finish`` gives no warning that carries PyTorch's alert and runs the backend's backward: over
    32 inputs its outputs are judged against the naive BF16 tier (evaluated with the mode off)."""
    T = 129

    def case(j: int, s: int):
        i = _inputs(T, seed=s, hidden=hidden)
        with monkeypatch.context() as m:
            _refuse_launches(m, det_backend)
            deterministic_algorithms(True, warn_only=True)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    got = _backward_judged(
                        facade.cake_mhc_nondet_train_backward_finish(*_args(i))
                    )
            finally:
                deterministic_algorithms(False)
        alerts = [str(w.message) for w in caught if _ALERT in str(w.message)]
        assert alerts == [], alerts
        expected, naive = expected_and_naive(i, _OUTPUTS)
        return [got], expected, naive

    judge_draws(case, record_property, seed=SEED + 2)


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
def test_deterministic_backward_workspace_is_refused(
    T: int, hidden: int, monkeypatch, record_property
) -> None:
    """A backward workspace of the deterministic package
    (``cake_pplx.cake_mhc_det_train.cake_mhc_det_train_backward_workspace``) is smaller than this package's layout
    of the call (which adds the column slab's arrival counts and per-chunk ``dfn`` slots), so ``backward_finish`` and
    a prepare node's ``backward_workspace`` refuse it (``ValueError``) with the deterministic package's launches
    refused; this package's own workspace of the call serves both: the zero prefix is zero after every call, and over
    32 inputs the outputs of ``backward_finish`` (under ``backward``) and of the autograd functions (under
    ``autograd``) are judged against the naive BF16 tier."""
    i = _inputs(T, hidden=hidden)
    what = f"T={T} H={hidden}"
    det_ws = det_facade.cake_mhc_det_train_backward_workspace(
        T, _device(), hidden=hidden
    )
    need = loader.workspace_bytes("bwd", T, _sm_count(), _arch(), hidden)
    assert det_ws.numel() < need, (what, det_ws.numel(), need)
    ws = facade.cake_mhc_nondet_train_backward_workspace(T, _device(), hidden=hidden)
    with monkeypatch.context() as m:
        _refuse_launches(m, det_backend)
        with pytest.raises(ValueError, match="workspace holds"):
            facade.cake_mhc_nondet_train_backward_finish(*_args(i), workspace=det_ws)
        _leaves, _handle, _Y, loss = _graph(
            _nd_prepare, _nd_finish, i, backward_workspace=det_ws
        )
        with pytest.raises(ValueError, match="workspace holds"):
            loss.backward()

        def case(j: int, s: int):
            c_i = _inputs(T, seed=s, hidden=hidden)
            got = _backward_judged(
                facade.cake_mhc_nondet_train_backward_finish(*_args(c_i), workspace=ws)
            )
            _assert_zero_prefix(ws, f"backward_finish {what} draw {j}")
            c = _connection(c_i, backward_workspace=ws)
            c["loss"].backward()
            _assert_zero_prefix(ws, f"prepare node {what} draw {j}")
            assert isinstance(c["handle"], backend._MHCTrainHandle), what
            e, t = expected_and_naive(c_i, _OUTPUTS)
            return _merge(
                _part("backward ", got, e, t), _connection_part("autograd ", c_i, c)
            )

        judge_draws(case, record_property, seed=SEED + 3 + T)


@requires_device
def test_finish_refuses_a_deterministic_handle() -> None:
    """``cake_mhc_nondet_train_finish`` given the handle of the deterministic package's prepare raises the handle's
    ``TypeError`` (naming the deterministic finish), and with a ``deterministic`` keyword the interpreter's
    ``TypeError``; neither call touches the handle, whose deterministic finish then gives the deterministic ``Y`` and
    gradients bit for bit."""
    i = _inputs(7, seed=SEED + 10)

    def finish(X, B, handle):
        with pytest.raises(TypeError, match=_NO_MODE_KEYWORD):
            facade.cake_mhc_nondet_train_finish(X, B, handle, deterministic=False)
        with pytest.raises(TypeError, match=_exactly(_HANDLE_ERROR)):
            facade.cake_mhc_nondet_train_finish(X, B, handle)
        return det_facade.cake_mhc_det_train_finish(X, B, handle)

    handle, Y, grads = _autograd(det_facade.cake_mhc_det_train_prepare, finish, i)
    _h, want_Y, want = _autograd(
        det_facade.cake_mhc_det_train_prepare, det_facade.cake_mhc_det_train_finish, i
    )
    assert isinstance(handle, det_backend._MHCTrainHandle)
    assert _same_bits(Y, want_Y)
    for name, g, w in zip(_LEAVES, grads, want, strict=True):
        assert _same_bits(g, w), f"grad {name}"


# ----------------------------------------------------------------------------------------------------------------
# Hidden sizes


@pytest.mark.cpu_only
def test_nondeterministic_workspace_helpers_reject_other_hidden_sizes() -> None:
    """This package lists the hidden sizes 4096, 5120 and 7168 (``cake_jit.HIDDEN_SIZES``): its workspace
    helpers raise ``ValueError`` for another hidden size or a non-``int`` one before any device query, in the facade
    and in the backend (the sizes of the device's architecture are ``test_hidden_sizes_follow_the_device``'s)."""
    assert tuple(loader.HIDDEN_SIZES) == HIDDEN_SIZES
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in _WORKSPACE_HELPERS:
            for module in (facade, backend):
                for hidden in (6144, 2048, 4097, 0, -4096, True, 4096.0, "4096"):
                    with pytest.raises(ValueError, match="hidden"):
                        getattr(module, name)(1, torch.device("cuda"), hidden=hidden)


@requires_device
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_hidden_sizes_follow_the_device(hidden: int, record_property) -> None:
    """On every architecture (sm_100a, sm_103a and sm_107a) this package takes a hidden size exactly where the
    deterministic package does: 5120 and 7168 size their workspaces (the forward one the deterministic package's, the
    backward one larger than its, both other than the H 4096 sizes) and the entry points run on them
    (``backward_finish``'s outputs judged against the naive BF16 tier over 32 inputs)."""
    dev = _device()
    exported = facade.is_cake_mhc_nondet_train_supported(dev, hidden)
    assert exported and hidden in _hidden_sizes(), (_arch(), hidden)
    assert backend.is_cake_mhc_nondet_train_supported(dev, hidden) == exported
    assert det_facade.is_cake_mhc_det_train_supported(dev, hidden)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for kind in ("forward", "backward"):
            nbytes = getattr(facade, f"cake_mhc_nondet_train_{kind}_workspace_bytes")
            reference_nbytes = getattr(
                det_facade, f"cake_mhc_det_train_{kind}_workspace_bytes"
            )
            got = nbytes(300, dev, hidden=hidden)
            want = reference_nbytes(300, dev, hidden=hidden)
            assert got != nbytes(300, dev), kind
            assert got == want if kind == "forward" else got > want, (kind, got, want)
            ws = getattr(facade, f"cake_mhc_nondet_train_{kind}_workspace")(
                300, dev, hidden=hidden
            )
            assert ws.numel() == got and int(torch.count_nonzero(ws)) == 0, kind

    def case(j: int, s: int):
        c = _inputs(7, seed=s, hidden=hidden)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _backward_judged(
                facade.cake_mhc_nondet_train_backward_finish(*_args(c))
            )
        expected, naive = expected_and_naive(c, _OUTPUTS)
        return [got], expected, naive

    judge_draws(case, record_property, seed=SEED + hidden)
