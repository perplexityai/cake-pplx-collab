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

Standalone-import tests of the Cake non-deterministic GDN chunked-training package
(``cake_pplx._backends.cake_gdn_nondet_train``).

A copy of the package directory must import and reach its host-side checks on its own: under another top-level
name, in a ``python -I`` subprocess (the form of a vendoring gate,
``scripts/check_cake_build_manifests.py --vendored-gate``), with its manifest read by its own loader.  The package
imports only itself, ``torch`` and ``tvm_ffi``: the JIT fallback of ``cake_build.py``
imports the package's own JIT shim ``_cake_pplx_jit.py``, the package reads no
environment variable outside that shim (every control is a keyword argument), and the backend test
files of the package import it through ``tests.cake.test_helpers.cake_package`` only, so a vendoring build runs them
against its own copy.  The facade ``cake_pplx.cake_gdn_nondet_train`` is covered by
``test_cake_gdn_nondet_train_api.py``.  CPU only: no GPU, no compiled library and no JIT build is needed.
"""

from __future__ import annotations

import ast
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

tvm_ffi = pytest.importorskip("tvm_ffi")  # cake_backend imports it at module level

PACKAGE_DIR = Path(cake_package("cake_gdn_nondet_train").__file__).resolve().parent
TESTS_DIR = Path(__file__).resolve().parent
# The backend test files of the package: each imports the package through the shared alias helper only.
BACKEND_TESTS = (
    "test_cake_gdn_nondet_train.py",
    "test_cake_gdn_nondet_train_plan.py",
    "test_cake_gdn_nondet_train_host.py",
    "test_cake_gdn_nondet_train_standalone.py",
)
VENDORED_NAME = (
    "vendored_cake_gdn_nondet_train"  # the copy's top-level name in the subprocess
)


# ---------------------------------------------------------------------------------------------------------------------
# A vendored copy


def _dependency_paths() -> list[str]:
    """Where ``torch`` and ``tvm_ffi`` live, plus the interpreter's site-packages (``python -I`` drops
    ``PYTHONPATH`` and the user site)."""
    paths = list(site.getsitepackages())
    for module in (torch, tvm_ffi):
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(module.__file__))))
    return list(dict.fromkeys(paths))


_ISOLATED = """
import importlib
import os
import sys
import warnings

PKG_ROOT, PKG_NAME = sys.argv[1], sys.argv[2]
for p in sys.argv[3:]:
    if p not in sys.path:
        sys.path.append(p)
sys.path.insert(0, PKG_ROOT)

import torch

pkg = importlib.import_module(PKG_NAME)
assert os.path.realpath(pkg.__file__).startswith(os.path.realpath(PKG_ROOT) + os.sep), pkg.__file__
assert sorted(pkg.__all__) == sorted(
    ["CakeGDNNondetFunction", "cake_gdn_nondet", "cake_gdn_nondet_train_backward", "cake_gdn_nondet_train_forward"]
), pkg.__all__
for name in pkg.__all__:
    getattr(pkg, name)
for sub in ("cake_backend", "cake_build", "cake_chain", "cake_jit", "cake_plan"):
    importlib.import_module(PKG_NAME + "." + sub)
build = importlib.import_module(PKG_NAME + ".cake_build")
manifest = build.manifest()
assert manifest["schema"] == "cake-build/1", manifest["schema"]
assert manifest["libraries"], "the manifest lists no library"
for entry in manifest["libraries"]:
    assert entry["path"] == "aot/" + entry["name"] + ".so", entry["name"]

# host operands: the API checks refuse them with ValueError before any device access, twice (nothing warns)
q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
v = torch.zeros(1, 64, 3, 128, dtype=torch.bfloat16)
g, beta = torch.zeros(1, 64, 3), torch.zeros(1, 64, 3)
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    for _ in range(2):
        try:
            pkg.cake_gdn_nondet(q, q, v, g, beta)
        except ValueError as exc:
            assert "CUDA tensor" in str(exc), exc
        else:
            raise AssertionError("host operands must be refused")
        try:
            pkg.cake_gdn_nondet(q, q, v, g, beta, deterministic=False)
        except TypeError as exc:
            assert "deterministic" in str(exc), exc
        else:
            raise AssertionError("the entry took deterministic=")
assert not rec, [f"{w.category.__name__}: {w.message}" for w in rec]
print("OK")
"""


def test_vendored_copy_imports_standalone(tmp_path) -> None:
    """A copy of the package under another top-level name imports in a ``python -I`` subprocess, resolves its four
    names and its modules, reads its ``cake-build/1`` manifest with its own loader, and refuses host operands with
    ``ValueError`` and a ``deterministic`` keyword with ``TypeError`` without a warning of any kind."""
    root = tmp_path / "site"
    shutil.copytree(
        PACKAGE_DIR,
        root / VENDORED_NAME,
        ignore=shutil.ignore_patterns("__pycache__", "aot"),
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-I", "-c", _ISOLATED, str(root), VENDORED_NAME]
        + _dependency_paths(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert proc.stdout.strip() == "OK", proc.stdout


# ---------------------------------------------------------------------------------------------------------------------
# The package's import surface (source only)


def _exception_names(expr) -> list[str]:
    """Exception class names of an ``except`` clause: a Name or a tuple of Names."""
    if isinstance(expr, ast.Tuple):
        return [e.id for e in expr.elts if isinstance(e, ast.Name)]
    return [expr.id] if isinstance(expr, ast.Name) else []


def _python_files() -> list[Path]:
    return sorted(PACKAGE_DIR.glob("*.py"))


# The package's JIT shim.
JIT_SHIM = "_cake_pplx_jit"


def _jit_shim_import(node) -> bool:
    return isinstance(node, ast.ImportFrom) and node.level == 1 and node.module == JIT_SHIM


def test_package_imports_its_jit_shim_only_in_the_jit_fallback() -> None:
    """No relative import climbs above the package, and every import of the JIT shim sits in the body of a ``try``
    with an ``except ImportError`` handler: exactly one such block, the JIT fallback of ``cake_build.py``."""
    guards: dict[str, set[int]] = {}
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        guarded = {}  # import node -> line of the enclosing guarded try
        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and any(
                "ImportError" in _exception_names(h.type) for h in node.handlers
            ):
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        guarded[sub] = node.lineno
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level < 2, (
                    f"{path.name}:{node.lineno}: import from outside the package"
                )
            if _jit_shim_import(node):
                assert node in guarded, (
                    f"{path.name}:{node.lineno}: JIT shim import outside a try/except ImportError"
                )
                guards.setdefault(path.name, set()).add(guarded[node])
    assert {name: len(lines) for name, lines in guards.items()} == {
        "cake_build.py": 1
    }, guards


def test_the_package_reads_no_environment_variable() -> None:
    """Every control is a keyword argument: no module of the package reads ``os.environ`` or ``os.getenv``.  The JIT
    shim is the exception: it reads the toolkit and cache locations (``CUDA_HOME``,
    ``CAKE_PPLX_CACHE_DIR``), never a kernel control."""
    offenders = []
    for path in _python_files():
        if path.stem == JIT_SHIM:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("environ", "getenv", "putenv")
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module == "os":
                if any(a.name in ("environ", "getenv", "putenv") for a in node.names):
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


@pytest.mark.parametrize("name", BACKEND_TESTS)
def test_backend_tests_import_the_package_through_the_alias_helper(name: str) -> None:
    """A backend test file names its package through ``tests.cake.test_helpers.cake_package``, so a vendoring build
    that points the helper at its own copy runs it unchanged (the facade's tests live in
    ``test_cake_gdn_nondet_train_api.py``)."""
    path = TESTS_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} is not in this tree")
    tree = ast.parse(path.read_text(), filename=str(path))
    helper = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "tests.cake.test_helpers.cake_package"
    ]
    assert helper, f"{name} does not import tests.cake.test_helpers.cake_package"
