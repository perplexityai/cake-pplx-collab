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

Names of the non-deterministic dense multi-head attention backward module ``cake_pplx.cake_fmha_nondet_train``.

The family has one mode: the main stage adds the FP32 dQ tiles of every key block in arrival order, and no
deterministic package exists (no ``cake_fmha_det_train`` module, facade or backend).  The public names are the
facade's ``__all__`` and the backend object it forwards to (``cake_pplx._backends.cake_fmha_nondet_train``): each
carries the mode (``nondet``) as a token right after the op, and no entry point takes a ``deterministic`` keyword.
The rules shared by every family are checked with ``tests.cake.test_helpers.cake_mode_names``; the checks here are this
family's own.  The package raises PyTorch's deterministic-algorithms alert (the API suite checks it on a device).
Build identities (program and library stems ``cake_fmha_train_<hex>``, kernel symbols, csrc file names) keep their
names.  The package serves capabilities (10, 0), (10, 3) and (10, 7) only.
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
import cake_pplx.cake_fmha_nondet_train as public
from tests.cake.test_helpers.cake_mode_names import (
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
)

pytestmark = pytest.mark.cpu_only

BACKEND = "cake_pplx._backends.cake_fmha_nondet_train"
ENTRY = "cake_fmha_nondet_train_backward"
ENTRIES = (ENTRY,)
# The backend objects the facade forwards to (``cake_backend``, also exported by the package).
FORWARDED = ENTRIES
# Module paths of a deterministic package, which this family does not have.
NO_DET_MODULES = (
    "cake_pplx.cake_fmha_det_train",
    "cake_pplx._backends.cake_fmha_det_train",
    "cake_pplx._backends.cake_fmha_det_train.cake_backend",
)
# Internal helpers of the backend (the stage launchers and the device helpers): not package attributes.
INTERNAL = (
    "launch_bwd_pre",
    "launch_bwd_main",
    "launch_bwd_post",
    "launch_bwd_fused",
    "device_arch",
    "arch_for_capability",
    "library_name",
)
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}


def _backend(module_name=None):
    return importlib.import_module(BACKEND if module_name is None else f"{BACKEND}.{module_name}")


def _fresh(code: str):
    """``code`` run in a fresh interpreter; returns the JSON it prints last."""
    repo = str(Path(__file__).resolve().parents[2])
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_facade_all_is_the_entry():
    """The facade's ``__all__`` is exactly its one entry, a stable API (no experimental marking) named after the
    module; the backend package exports the same object."""
    assert list(public.__all__) == list(ENTRIES)
    fn = getattr(public, ENTRY)
    assert getattr(fn, "is_experimental", False) is False
    assert fn.__name__ == ENTRY
    assert getattr(_backend(), ENTRY) is getattr(_backend("cake_backend"), ENTRY)


@pytest.mark.parametrize("name", NO_DET_MODULES)
def test_there_is_no_deterministic_module(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_the_namespaces_list_the_nondet_package_only():
    """In a fresh interpreter: ``cake_pplx`` and ``cake_pplx._backends`` list ``cake_fmha_nondet_train`` and no
    ``cake_fmha_det_train``; the deterministic module paths do not import."""
    code = (
        "import json, pkgutil\n"
        "import cake_pplx, cake_pplx._backends\n"
        "spaces = (cake_pplx, cake_pplx._backends)\n"
        "names = {m.name for p in spaces for m in pkgutil.iter_modules(p.__path__)}\n"
        "names |= {n for p in spaces for n in dir(p)}\n"
        "print(json.dumps(dict(det=sorted(n for n in names if n.startswith('cake_fmha_det')),\n"
        "    listed='cake_fmha_nondet_train' in names)))\n"
    )
    assert _fresh(code) == dict(det=[], listed=True)
    assert_old_names_fail(NO_DET_MODULES, cwd=Path(__file__).resolve().parents[2])


def test_the_shared_mode_rules_hold():
    """The rules shared by every family (``tests.cake.test_helpers.cake_mode_names``): the facade's public names carry
    ``nondet`` after the op, no entry point takes ``deterministic``, and the ``cake_pplx`` namespaces, the backend
    package and ``cake_backend`` bind the facade's entry and the backend object it calls only under the facade's
    name."""
    package, backend = _backend(), _backend("cake_backend")
    assert_mode_names(public)
    assert_no_deterministic_keyword([getattr(public, name) for name in public.__all__])
    assert_same_object_same_name(
        public,
        (cake_pplx, cake_pplx._backends, package, backend),
        forwards=[(name, getattr(backend, name)) for name in FORWARDED],
    )


def test_public_names_carry_the_mode():
    """The backend object the facade forwards to carries ``nondet`` after the op (the shared rule, applied to the
    backend's names); every package ``__all__`` name is such an object or an ALL_CAPS constant, and the backend's
    internal helpers (the stage launchers, the device helpers) are not package attributes."""
    backend = _backend("cake_backend")
    package = _backend()
    assert_mode_names(package, names=FORWARDED)
    for name in FORWARDED:
        assert name in backend.__all__ and name in package.__all__, name
        assert getattr(package, name) is getattr(backend, name), name
    for name in package.__all__:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in FORWARDED, name
    internal = [
        name for name in backend.__all__ if name not in FORWARDED and not re.fullmatch(r"[A-Z][A-Z0-9_]*", name)
    ]
    for name in INTERNAL:
        assert hasattr(backend, name), name
    for name in set(internal) | set(INTERNAL):
        assert not hasattr(package, name), name


def test_no_entry_takes_a_mode_keyword():
    """The facade entry and the backend function it calls accept no ``deterministic`` keyword (no such parameter, no
    ``**kwargs``); passing it to the facade is a ``TypeError`` from argument binding, before any work."""
    backend = _backend("cake_backend")
    fns = {f"facade.{ENTRY}": getattr(public, ENTRY), f"backend.{ENTRY}": getattr(backend, ENTRY)}
    autograd = [
        cls
        for cls in vars(backend).values()
        if inspect.isclass(cls) and issubclass(cls, torch.autograd.Function) and cls.__module__ == backend.__name__
    ]
    assert autograd == []  # a backward step only: no autograd entry
    for label, fn in fns.items():
        signature = inspect.signature(fn)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)
        assert "deterministic" not in signature.parameters, label
        assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values()), label
    t = torch.zeros(1, 128, 2, 128, dtype=torch.bfloat16)
    lse = torch.zeros(1, 2, 128)
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            getattr(public, ENTRY)(t, t, t, t, lse, t, deterministic=value)


def test_the_package_follows_torch_deterministic_mode():
    """The backend reads torch's deterministic mode (its alert); the entry documents it."""
    source = inspect.getsource(_backend("cake_backend"))
    assert "torch.are_deterministic_algorithms_enabled(" in source
    assert "torch.is_deterministic_algorithms_warn_only_enabled(" in source
    assert "use_deterministic_algorithms" in getattr(public, ENTRY).__doc__


def test_capabilities_map_to_the_three_architectures(monkeypatch):
    """Capabilities (10, 0), (10, 3) and (10, 7) map to ``sm_100a``, ``sm_103a`` and ``sm_107a``; every other
    capability maps to no target.  A device of capability (10, 7) is served as ``sm_107a``, and a device of another
    capability is a ``RuntimeError`` naming the delivered architectures."""
    jit, backend = _backend("cake_jit"), _backend("cake_backend")
    assert tuple(jit.ARCHES) == ARCHES and tuple(backend.ARCHES) == ARCHES
    assert dict(jit.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    assert dict(backend.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    for capability, arch in ARCH_BY_CAPABILITY.items():
        assert backend.arch_for_capability(capability) == arch
    for capability in ((10, 1), (9, 0), (11, 0), (12, 0), (12, 1)):
        assert backend.arch_for_capability(capability) is None, capability
    for rec in jit.MODULES.values():
        assert tuple(rec["arches"]) == ARCHES
    clear = getattr(backend.device_arch, "cache_clear", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    clear()
    try:
        assert backend.device_arch(0) == "sm_107a"
    finally:
        monkeypatch.undo()
        clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            backend.device_arch(0)
    finally:
        monkeypatch.undo()
        clear()


def test_package_identity_and_build_identity():
    """``build.json`` and the registry name the package; every library stem is a ``cake_fmha_train`` program or host
    sequence built for ``sm_100a``, ``sm_103a`` and ``sm_107a``, and the registry's library names are the stems of
    ``build.json``."""
    jit = _backend("cake_jit")
    assert jit.PACKAGE == BACKEND
    assert jit.LIBRARY_PREFIX == "cake_fmha_train"
    path = Path(jit.__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    manifest = json.loads(path.read_text())
    assert manifest["package"] == BACKEND
    names = {entry["name"] for entry in manifest["libraries"]}
    libraries = (*jit.MODULES.values(), *jit.SEQUENCES.values())
    assert names == {f"{rec['library']}_{arch}" for rec in libraries for arch in rec["arches"]}
    assert all(re.fullmatch(r"cake_fmha_train_[0-9a-f]{20}_sm_10[037]a", name) for name in names), sorted(names)
    assert {entry["arch"] for entry in manifest["libraries"]} == set(ARCHES)
    stems = {rec["library"] for rec in libraries}
    assert len(stems) == len(libraries) and len(names) == len(stems) * len(ARCHES)
