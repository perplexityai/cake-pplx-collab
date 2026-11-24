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

Names of the deterministic route-weight row scaling module ``cake_pplx.cake_route_weight_scale_det_train``.

The family has one mode, ``det``: nothing sums across rows and the one sum, ``dw`` within a row, has one fixed order,
so no non-deterministic package exists (no ``cake_route_weight_scale_nondet_train`` module, facade or backend), and no
KCP form (the op works on each row alone).  The public names are the facade's ``__all__`` and the backend objects it
forwards to (``cake_pplx._backends.cake_route_weight_scale_det_train``): each carries the mode (``det``) as a token
right after the op, and no entry point takes a ``deterministic`` keyword.  The rules shared by every family are checked
with ``tests.cake.test_helpers.cake_mode_names``; the checks here are this family's own: the backend never reads
torch's deterministic-algorithms setting, and the build manifest names the package, one
``cake_route_weight_scale_det_train_<hex>`` library per record and architecture, and the numerics flags (no fast
math, no flush to zero).
"""

import importlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import cake_pplx
import cake_pplx._backends
import cake_pplx.cake_route_weight_scale_det_train as public
from tests.cake.test_helpers.cake_mode_names import (
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
)

pytestmark = pytest.mark.cpu_only

BACKEND = "cake_pplx._backends.cake_route_weight_scale_det_train"
ENTRIES = (
    "cake_route_weight_scale_det",
    "cake_route_weight_scale_det_train_backward",
    "cake_route_weight_scale_det_train_forward",
    "is_cake_route_weight_scale_det_train_supported",
)
CONSTANTS = ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES", "WIDTHS")
AUTOGRAD = "CakeRouteWeightScaleDetFunction"
# Module paths of a non-deterministic or KCP package, which this family does not have.
NO_OTHER_MODULES = (
    "cake_pplx.cake_route_weight_scale_nondet_train",
    "cake_pplx._backends.cake_route_weight_scale_nondet_train",
    "cake_pplx.cake_route_weight_scale_det_train_kcp",
    "cake_pplx._backends.cake_route_weight_scale_det_train_kcp",
    "cake_pplx.cake_route_weight_scale_nondet_train_kcp",
)
# torch's deterministic-algorithms entry points: the deterministic backend reads none of them.
TORCH_DETERMINISTIC_TOKENS = (
    "are_deterministic_algorithms_enabled",
    "is_deterministic_algorithms_warn_only_enabled",
    "_alert_not_deterministic",
    "use_deterministic_algorithms",
)


def _backend(module_name=None):
    return importlib.import_module(BACKEND if module_name is None else f"{BACKEND}.{module_name}")


def _fresh(code: str):
    """``code`` run in a fresh interpreter; returns the JSON it prints last."""
    repo = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=repo, timeout=600, check=False
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _manifest_or_skip() -> dict:
    path = Path(_backend("cake_jit").__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    return json.loads(path.read_text())


def test_facade_all_is_the_entries_and_the_backend_objects():
    """The facade's ``__all__`` is its four entries, the autograd class and the constants; the entries are named after
    the module, and the lazy names are the backend's objects."""
    assert sorted(public.__all__) == sorted((*ENTRIES, AUTOGRAD, *CONSTANTS))
    package = _backend()
    for name in ENTRIES:
        fn = getattr(public, name)
        assert fn.__name__ == name
        assert callable(getattr(package, name)), name
    for name in (AUTOGRAD, *CONSTANTS):
        assert getattr(public, name) is getattr(package, name), name


@pytest.mark.parametrize("name", NO_OTHER_MODULES)
def test_there_is_no_other_mode_or_kcp_module(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_the_namespaces_list_the_det_package_only():
    """In a fresh interpreter: ``cake_pplx`` and ``cake_pplx._backends`` list ``cake_route_weight_scale_det_train`` and
    no other ``cake_route_weight_scale`` module; the other module paths do not import."""
    code = (
        "import json, pkgutil\n"
        "import cake_pplx, cake_pplx._backends\n"
        "spaces = (cake_pplx, cake_pplx._backends)\n"
        "names = {m.name for p in spaces for m in pkgutil.iter_modules(p.__path__)}\n"
        "names |= {n for p in spaces for n in dir(p)}\n"
        "mine = 'cake_route_weight_scale_det_train'\n"
        "others = sorted(n for n in names if n.startswith('cake_route_weight_scale') and n != mine)\n"
        "print(json.dumps(dict(others=others, listed=mine in names)))\n"
    )
    assert _fresh(code) == dict(others=[], listed=True)
    assert_old_names_fail(NO_OTHER_MODULES, cwd=Path(__file__).resolve().parents[2])


def test_the_shared_mode_rules_hold():
    """The rules shared by every family (``tests.cake.test_helpers.cake_mode_names``): the facade's public names carry
    ``det`` after the op, no entry point takes ``deterministic``, and the ``cake_pplx`` namespaces and the backend
    package bind the facade's entries and the backend objects they call only under the facade's names."""
    package = _backend()
    assert_mode_names(public)
    assert_no_deterministic_keyword(public)
    assert_same_object_same_name(
        public,
        (cake_pplx, cake_pplx._backends, package),
        forwards=[(name, getattr(package, name)) for name in ENTRIES],
    )


def test_backend_names_carry_the_mode():
    """Every backend ``__all__`` name is an ALL_CAPS constant or carries ``det`` after the op."""
    package = _backend()
    assert_mode_names(package, "det", stem=False)
    for name in package.__all__:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            assert name.startswith(("cake_route_weight_scale_det", "CakeRouteWeightScaleDet", "is_cake_route_weight_scale_det")), name


def test_no_entry_takes_a_mode_keyword():
    """The facade entries, the backend functions they call and the autograd class accept no ``deterministic`` keyword;
    passing it to the facade is a ``TypeError`` from argument binding, before any work."""
    package = _backend()
    for name in ENTRIES:
        for fn in (getattr(public, name), getattr(package, name)):
            signature = inspect.signature(fn)
            assert "deterministic" not in signature.parameters, name
            assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values()), name
            with pytest.raises(TypeError):
                signature.bind_partial(deterministic=False)
    x = torch.zeros(4, 4096, dtype=torch.bfloat16)
    w = torch.zeros(4)
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            public.cake_route_weight_scale_det_train_forward(x, w, deterministic=value)
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            public.cake_route_weight_scale_det_train_backward(x, x, w, deterministic=value)
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            public.cake_route_weight_scale_det(x, w, deterministic=value)


def test_the_backend_never_reads_the_torch_deterministic_setting():
    """The module is the mode: no module of the backend package names torch's deterministic-algorithms entry points.
    The facade documents the behaviour."""
    root = Path(_backend("cake_jit").__file__).parent
    paths = sorted(root.glob("*.py"))
    assert {"cake_jit.py", "__init__.py"} <= {p.name for p in paths}
    for path in paths:
        text = path.read_text()
        for token in TORCH_DETERMINISTIC_TOKENS:
            assert token not in text, (path.name, token)
    assert "use_deterministic_algorithms" in public.__doc__


def test_package_identity_and_build_identity():
    """``build.json`` and the registry name the package; each of the four records is one
    ``cake_route_weight_scale_det_train`` program built for ``sm_100a`` and ``sm_103a``."""
    jit = _backend("cake_jit")
    assert jit.PACKAGE == BACKEND
    assert jit.LIBRARY_PREFIX == "cake_route_weight_scale_det_train"
    manifest = _manifest_or_skip()
    assert manifest["package"] == BACKEND
    names = {entry["name"] for entry in manifest["libraries"]}
    assert names == {f"{rec['library']}_{arch}" for rec in jit.MODULES.values() for arch in rec["arches"]}
    assert all(re.fullmatch(r"cake_route_weight_scale_det_train_[0-9a-f]{20}_sm_10[03]a", n) for n in names), names
    assert len(names) == len(jit.MODULES) * len(jit.ARCHES) == 8


def test_manifest_numerics_flags():
    """No library is built with fast math or flush to zero: every product and sum rounds to nearest without
    flushing."""
    manifest = _manifest_or_skip()
    common = list(manifest["common"]["nvcc_flags"])
    for item in manifest["common"].get("nvcc_flags_if", []):
        common += list(item.get("flags", []))
    for entry in manifest["libraries"]:
        flags = common + list(entry["nvcc_flags"])
        for item in entry.get("nvcc_flags_if", []):
            flags += list(item.get("flags", []))
        assert "--use_fast_math" not in flags and "-use_fast_math" not in flags, (entry["name"], flags)
        assert "--ftz=true" not in flags, (entry["name"], flags)
