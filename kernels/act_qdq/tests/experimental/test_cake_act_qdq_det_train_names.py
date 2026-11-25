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

Names of the Cake activation quantize-dequantize family.

The family has one mode, ``det``: the facade ``cake_pplx.cake_act_qdq_det_train``
(top level) and its backend package ``cake_pplx._backends.cake_act_qdq_det_train``.
There is no ``nondet`` module (no step of the op has an order to relax) and no KCP
module (the op works on each token alone).  Every public name but the module
constants (``ARCHES``, ``SUPPORTED_COMPUTE_CAPABILITIES``, ``FORMATS``,
``GROUP_SIZES``, and in the backend ``ALIGNMENT_BYTES`` and ``MAX_ELEMENTS``)
carries ``det`` right after the op: the functions are named by the module plus a
verb, and the support query is ``is_cake_act_qdq_det_train_supported``.  No entry
point has a ``deterministic`` parameter.  The family is new, so it has no former
names.  The facade imports its backend package on first use, so ``import
cake_pplx`` loads no module of it; ``cake_pplx`` is a PEP 420 namespace package
(no ``__init__.py``) that binds no name of the family itself.  The shared checks
of the mode-in-name rule (``tests/cake/test_helpers/cake_mode_names.py``) run over
both namespaces; the checks here add what is particular to this family.  The JIT
library names (``cake_act_qdq_<hash>``) are build identities, not Python names.
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
import cake_pplx.cake_act_qdq_det_train as facade
import cake_pplx._backends.cake_act_qdq_det_train as backend
from tests.cake.test_helpers.cake_mode_names import assert_cake_mode_module

pytestmark = pytest.mark.cpu_only

MODULE = "cake_act_qdq_det_train"
FACADE = f"cake_pplx.{MODULE}"
BACKEND = f"cake_pplx._backends.{MODULE}"
# The facade's functions (its own ``cake_api`` defs); each calls the backend
# function of the same name.
FACADE_FUNCTIONS = {
    f"{MODULE}_domain_empty",
    f"{MODULE}_domain_plan",
    f"{MODULE}_forward",
    f"{MODULE}_forward_workspace_bytes",
    f"is_{MODULE}_supported",
}
FORWARDED = sorted(FACADE_FUNCTIONS)
PUBLIC = {
    "facade": FACADE_FUNCTIONS,
    "backend": FACADE_FUNCTIONS,
}
CONSTANTS = {
    "facade": {"ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "FORMATS", "GROUP_SIZES"},
    "backend": {
        "ALIGNMENT_BYTES",
        "ARCHES",
        "FORMATS",
        "GROUP_SIZES",
        "MAX_ELEMENTS",
        "SUPPORTED_COMPUTE_CAPABILITIES",
    },
}
NAMESPACES = {"facade": facade, "backend": backend}
# Module paths of the modes and routes this family does not have.
ABSENT_MODULES = (
    "cake_pplx.cake_act_qdq_nondet_train",
    "cake_pplx._backends.cake_act_qdq_nondet_train",
    "cake_pplx.cake_act_qdq_det_train_kcp",
    "cake_pplx._backends.cake_act_qdq_det_train_kcp",
)
# A name of the family.
FAMILY = re.compile(r"act_qdq|ActQDQ|ActQdq")


def test_all_is_the_mode_names_and_the_constants() -> None:
    for kind, module in NAMESPACES.items():
        assert set(module.__all__) == PUBLIC[kind] | CONSTANTS[kind], kind
        for name in module.__all__:
            assert hasattr(module, name), (kind, name)


@pytest.mark.parametrize("kind", sorted(NAMESPACES))
def test_shared_mode_name_checks(kind: str) -> None:
    """The shared checks of the mode-in-name rule (``assert_cake_mode_module``):
    the mode word and its place in every public name; no entry point binds a
    ``deterministic`` keyword; the namespace, the ``cake_pplx`` namespace
    package and, for the facade, its backend package bind each object only under
    its own name."""
    if kind == "facade":
        assert_cake_mode_module(
            facade,
            namespaces=[cake_pplx, backend],
            forwards=[(name, getattr(backend, name)) for name in FORWARDED],
        )
    else:
        assert_cake_mode_module(
            backend,
            namespaces=[cake_pplx, cake_pplx._backends],
        )


@pytest.mark.parametrize("module", ABSENT_MODULES)
def test_absent_modes_do_not_import(module: str) -> None:
    assert importlib.util.find_spec(module) is None, module
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
    assert not hasattr(cake_pplx, module.rpartition(".")[2]), module


def test_no_name_shim() -> None:
    """The backend package has no module ``__getattr__``; the facade's resolves
    exactly the four constants of the backend and raises ``AttributeError`` for
    anything else."""
    assert "__getattr__" not in vars(backend)
    assert facade._BACKEND_OBJECTS == ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "FORMATS", "GROUP_SIZES")
    for name in facade._BACKEND_OBJECTS:
        assert facade.__getattr__(name) is getattr(backend, name), name
    for name in ("deterministic", "MAX_ELEMENTS", "ALIGNMENT_BYTES", "cake_jit", "_check_shape"):
        with pytest.raises(AttributeError):
            facade.__getattr__(name)


def test_facade_objects() -> None:
    """The facade's functions are its own ``cake_api`` defs, named as exported;
    its constants are the backend's objects."""
    for name in FACADE_FUNCTIONS:
        obj = getattr(facade, name)
        assert obj.__name__ == name, name
        assert obj is not getattr(backend, name), name
    for name in CONSTANTS["facade"]:
        assert getattr(facade, name) is getattr(backend, name), name


def test_modules_sit_at_their_places() -> None:
    """The facade sits at the ``cake_pplx`` top level, the backend package at
    ``cake_pplx/_backends/<module>/`` with its own build files, and neither
    namespace level has an ``__init__.py`` (PEP 420)."""
    root = Path(facade.__file__).parent
    assert root.name == "cake_pplx"
    assert not (root / "__init__.py").exists()
    assert not (root / "_backends" / "__init__.py").exists()
    assert getattr(cake_pplx, "__file__", None) is None
    assert getattr(cake_pplx._backends, "__file__", None) is None
    package = root / "_backends" / MODULE
    assert Path(backend.__file__) == package / "__init__.py"
    files = {p.name for p in package.iterdir() if p.name != "__pycache__"}
    assert {
        "__init__.py",
        "README.md",
        "_cake_pplx_jit.py",
        "build.json",
        "cake_build.py",
        "cake_domain.py",
        "cake_jit.py",
        "csrc",
    } <= files
    for submodule in ("cake_build", "cake_domain", "cake_jit"):
        loaded = importlib.import_module(f"{BACKEND}.{submodule}")
        assert Path(loaded.__file__).parent == package, submodule
    # the registry keys are the programs the routes and the interleaved-domain routes name; the library names are
    # build identities
    cake_jit = importlib.import_module(f"{BACKEND}.cake_jit")
    routed = {name for routes in cake_jit.ROUTES.values() for names in routes.values() for name in names}
    domain = {name for routes in cake_jit.DOMAIN_ROUTES.values() for name in routes.values()}
    assert not routed & domain
    assert set(cake_jit.MODULES) == routed | domain
    assert set(cake_jit.KINDS) <= routed
    assert all(
        re.fullmatch(r"cake_act_qdq_[0-9a-f]{20}", record["library"]) for record in cake_jit.MODULES.values()
    )


def test_top_level_binds_only_mode_names() -> None:
    """The ``cake_pplx`` namespace binds no entry point of the family: its only
    family name is the facade submodule itself (bound by the import system), and
    ``cake_pplx._backends`` binds only the backend package."""
    # private helper modules of the family (``_cake_pplx_api_<family>``) aside
    public = {name for name in vars(cake_pplx) if not name.startswith("_")}
    assert {name for name in public if FAMILY.search(name)} == {MODULE}
    assert getattr(cake_pplx, MODULE) is facade
    for name in FORWARDED:
        assert not hasattr(cake_pplx, name), name
    assert {name for name in vars(cake_pplx._backends) if FAMILY.search(name)} == {MODULE}
    assert getattr(cake_pplx._backends, MODULE) is backend


# A plain ``import cake_pplx`` binds none of the family's names and loads
# neither the facade nor the backend package; importing the facade still loads
# no backend module, and its first backend access does.
FRESH = (
    "import re, sys, cake_pplx\n"
    "def family(names):\n"
    f"    return sorted(n for n in names if re.search({FAMILY.pattern!r}, n))\n"
    "def loaded(prefix):\n"
    "    return sorted(m for m in sys.modules if m.startswith(prefix))\n"
    f"assert family(vars(cake_pplx)) == [], family(vars(cake_pplx))\n"
    f"assert loaded({FACADE!r}) == [] and loaded({BACKEND!r}) == []\n"
    f"assert family(dir(cake_pplx)) == [], family(dir(cake_pplx))\n"
    f"import {FACADE} as facade\n"
    f"assert loaded({BACKEND!r}) == [], loaded({BACKEND!r})\n"
    "facade.FORMATS\n"
    f"assert {BACKEND!r} in loaded({BACKEND!r}), loaded({BACKEND!r})\n"
)


def test_import_forms_in_a_fresh_interpreter() -> None:
    r = subprocess.run([sys.executable, "-c", FRESH], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (r.stdout, r.stderr)
