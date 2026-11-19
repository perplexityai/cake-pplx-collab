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

Names of the Cake routed-expert clamped SwiGLU training family.

The family has one mode, ``det``: the facade ``cake_pplx.cake_clamped_swiglu_det_train`` (top level) and its backend
package ``cake_pplx._backends.cake_clamped_swiglu_det_train``.  Every public name but the module constants
(``ARCHES``, ``INTERMEDIATE_SIZE``, ``SUPPORTED_COMPUTE_CAPABILITIES``) carries ``det`` right after the op: the
autograd entry ``cake_clamped_swiglu_det``, the steps ``cake_clamped_swiglu_det_train_forward`` / ``_backward``, the
support query ``is_cake_clamped_swiglu_det_train_supported`` and the autograd class ``CakeClampedSwiGLUDetFunction``.
No entry point has a ``deterministic`` parameter.  The family has no non-deterministic module (every output is
elementwise or a sum within one row) and no former names.  The facade imports its backend package on first use, so
``import cake_pplx`` loads no module of it; ``cake_pplx`` is a PEP 420 namespace package (no ``__init__.py``) that
binds no name of the family itself.  The shared checks of the mode-in-name rule
(``tests/cake/test_helpers/cake_mode_names.py``) run over both namespaces; the checks here add what is particular to
this family.  The library names (``cake_clamped_swiglu_det_train_<hex>``) are build identities, not Python names.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

import cake_pplx
import cake_pplx._backends
import cake_pplx._backends.cake_clamped_swiglu_det_train as backend
import cake_pplx.cake_clamped_swiglu_det_train as facade
from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module, assert_mode_names

pytestmark = pytest.mark.cpu_only

MODULE = "cake_clamped_swiglu_det_train"
FACADE = f"cake_pplx.{MODULE}"
BACKEND = f"cake_pplx._backends.{MODULE}"
AUTOGRAD = "CakeClampedSwiGLUDetFunction"
# The facade's functions (its own ``cake_api`` defs), each running what the backend function of the same name runs.
FACADE_FUNCTIONS = {
    "cake_clamped_swiglu_det",
    f"{MODULE}_forward",
    f"{MODULE}_backward",
    f"is_{MODULE}_supported",
}
CONSTANTS = {"ARCHES", "INTERMEDIATE_SIZE", "SUPPORTED_COMPUTE_CAPABILITIES"}
PUBLIC = FACADE_FUNCTIONS | {AUTOGRAD} | CONSTANTS
NAMESPACES = {"facade": facade, "backend": backend}
# The module paths of a non-deterministic package, which this family does not have.
NO_NONDET_MODULES = (
    "cake_pplx.cake_clamped_swiglu_nondet_train",
    "cake_pplx._backends.cake_clamped_swiglu_nondet_train",
)
# A name of the family.
FAMILY = re.compile(r"clamped_swiglu|ClampedSwiGLU|ClampedSwiglu")


def test_all_is_the_mode_names_and_the_constants() -> None:
    for kind, module in NAMESPACES.items():
        assert set(module.__all__) == PUBLIC, kind
        assert len(module.__all__) == len(PUBLIC), kind
        for name in module.__all__:
            assert hasattr(module, name), (kind, name)


def test_shared_mode_name_checks() -> None:
    """The shared checks of the mode-in-name rule (``assert_cake_mode_module``): the mode word and its place in every
    public name; no entry point binds a ``deterministic`` keyword; the facade, the ``cake_pplx`` namespace package and
    the backend package bind each object only under its own name.  The backend's public names follow the same rule."""
    assert_cake_mode_module(
        facade,
        namespaces=[cake_pplx, cake_pplx._backends, backend],
        forwards=[(name, getattr(backend, name)) for name in sorted(FACADE_FUNCTIONS)],
    )
    assert_cake_mode_module(backend, namespaces=[cake_pplx, cake_pplx._backends])
    assert_mode_names(backend, names=sorted(PUBLIC - CONSTANTS))


def test_the_autograd_class_carries_the_mode() -> None:
    cls = getattr(facade, AUTOGRAD)
    assert cls is getattr(backend, AUTOGRAD)
    assert cls.__name__ == AUTOGRAD
    assert cls.__name__.lower().startswith("cakeclampedswigludet")


@pytest.mark.parametrize("name", NO_NONDET_MODULES)
def test_there_is_no_nondeterministic_module(name: str) -> None:
    assert importlib.util.find_spec(name) is None, name
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_no_name_shim() -> None:
    """The backend package has no module ``__getattr__``; the facade's resolves exactly the backend's constants and
    autograd class and raises ``AttributeError`` for anything else."""
    assert "__getattr__" not in vars(backend)
    assert set(facade._BACKEND_OBJECTS) == CONSTANTS | {AUTOGRAD}
    for name in facade._BACKEND_OBJECTS:
        assert facade.__getattr__(name) is getattr(backend, name), name
    for name in ("deterministic", "cake_clamped_swiglu_nondet", "clamped_swiglu", "run_module", "_run"):
        with pytest.raises(AttributeError):
            facade.__getattr__(name)


def test_facade_objects() -> None:
    """The facade's functions are its own ``cake_api`` defs, named as exported; its constants and autograd class are
    the backend's objects."""
    for name in FACADE_FUNCTIONS:
        obj = getattr(facade, name)
        assert obj.__name__ == name, name
        assert obj is not getattr(backend, name), name
    for name in CONSTANTS | {AUTOGRAD}:
        assert getattr(facade, name) is getattr(backend, name), name


def test_modules_sit_at_their_places() -> None:
    """The facade sits at the ``cake_pplx`` top level, the backend package at ``cake_pplx/_backends/<module>/`` with its
    registry and JIT shim, and neither namespace level has an ``__init__.py`` (PEP 420)."""
    root = Path(facade.__file__).parent
    assert root.name == "cake_pplx"
    assert not (root / "__init__.py").exists()
    assert not (root / "_backends" / "__init__.py").exists()
    assert getattr(cake_pplx, "__file__", None) is None
    assert getattr(cake_pplx._backends, "__file__", None) is None
    package = root / "_backends" / MODULE
    assert Path(backend.__file__) == package / "__init__.py"
    files = {p.name for p in package.iterdir() if p.name != "__pycache__"}
    assert {"__init__.py", "README.md", "cake_jit.py", "_cake_pplx_jit.py", "csrc"} <= files
    loaded = importlib.import_module(f"{BACKEND}.cake_jit")
    assert Path(loaded.__file__).parent == package
    assert loaded.MODULES and all(
        re.fullmatch(r"cake_clamped_swiglu_det_train_[0-9a-f]{20}", record["library"])
        for record in loaded.MODULES.values()
    )


def test_top_level_binds_only_mode_names() -> None:
    """The ``cake_pplx`` namespace binds no entry point of the family: its only family name is the facade submodule
    itself (bound by the import system), and ``cake_pplx._backends`` binds only the backend package."""
    public = {name for name in vars(cake_pplx) if not name.startswith("_")}
    assert {name for name in public if FAMILY.search(name)} == {MODULE}
    assert getattr(cake_pplx, MODULE) is facade
    for name in FACADE_FUNCTIONS | {AUTOGRAD}:
        assert not hasattr(cake_pplx, name), name
    assert {name for name in vars(cake_pplx._backends) if FAMILY.search(name)} == {MODULE}
    assert getattr(cake_pplx._backends, MODULE) is backend


# A plain ``import cake_pplx`` binds none of the family's names and loads neither the facade nor the backend package;
# importing the facade still loads no backend module, and its first backend access does.
FRESH = (
    "import re, sys, cake_pplx\n"
    "def family(names):\n"
    f"    return sorted(n for n in names if re.search({FAMILY.pattern!r}, n))\n"
    "def loaded(prefix):\n"
    "    return sorted(m for m in sys.modules if m.startswith(prefix))\n"
    "assert family(vars(cake_pplx)) == [], family(vars(cake_pplx))\n"
    f"assert loaded({FACADE!r}) == [] and loaded({BACKEND!r}) == []\n"
    "assert family(dir(cake_pplx)) == [], family(dir(cake_pplx))\n"
    f"import {FACADE} as facade\n"
    f"assert loaded({BACKEND!r}) == [], loaded({BACKEND!r})\n"
    "facade.ARCHES\n"
    f"assert {BACKEND!r} in loaded({BACKEND!r}), loaded({BACKEND!r})\n"
)


def test_import_forms_in_a_fresh_interpreter() -> None:
    r = subprocess.run(
        [sys.executable, "-c", FRESH],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
