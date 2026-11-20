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

Names of the deterministic Engram row lookup module ``cake_pplx.cake_engram_lookup_det_train``.

The family has one mode, ``det``: a gather has no sums, so the output repeats bit for bit and no non-deterministic
package exists (no ``cake_engram_lookup_nondet_train`` module, facade or backend), and no KCP form (the lookup is
single-rank).  The public names are the facade's ``__all__`` and the backend objects it forwards to
(``cake_pplx._backends.cake_engram_lookup_det_train``): each carries the mode (``det``) as a token right after the
op, and no entry point takes a ``deterministic`` keyword.  The rules shared by every family are checked with
``tests.cake.test_helpers.cake_mode_names``; the checks here are this family's own: the backend never reads torch's
deterministic-algorithms setting, the package serves capabilities (10, 0), (10, 3) and (10, 7) only (every other
capability a ``RuntimeError``), and the build manifest names the package, two ``cake_engram_lookup_det_train_<hex>``
libraries per architecture (the main program and the small one), the minimum CUDA toolkit of each architecture and
numerics flags that keep FP32 subnormals (no fast math, ``--ftz=false``).  Build identities (program and library stems, kernel symbols, csrc file
names) keep their names.
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
import cake_pplx.cake_engram_lookup_det_train as public
from tests.cake.test_helpers.cake_mode_names import (
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
)

pytestmark = pytest.mark.cpu_only

BACKEND = "cake_pplx._backends.cake_engram_lookup_det_train"
ENTRIES = ("cake_engram_lookup_det_train_forward", "cake_engram_lookup_det_train_split_packed")
# The backend objects the facade forwards to (``cake_backend``, also exported by the package).
FORWARDED = ENTRIES
# Module paths of a non-deterministic or KCP package, which this family does not have.
NO_OTHER_MODULES = (
    "cake_pplx.cake_engram_lookup_nondet_train",
    "cake_pplx._backends.cake_engram_lookup_nondet_train",
    "cake_pplx._backends.cake_engram_lookup_nondet_train.cake_backend",
    "cake_pplx.cake_engram_lookup_det_train_kcp",
    "cake_pplx._backends.cake_engram_lookup_det_train_kcp",
)
# Internal helpers of the backend (the launch and the device helpers): not package attributes.
INTERNAL = (
    "device_arch",
    "arch_for_capability",
    "library_name",
    "load",
    "sm_count",
    "_launch",
    "_check",
)
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
# The minimum CUDA toolkit of every library of each architecture in ``build.json``.
MIN_CUDA = {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.5"}
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
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=repo,
        timeout=600,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _manifest_or_skip() -> dict:
    path = Path(_backend("cake_jit").__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    return json.loads(path.read_text())


def test_facade_all_is_the_entries():
    """The facade's ``__all__`` is exactly its two entries, stable APIs (no experimental marking) named after the
    module; the backend package exports the same objects as ``cake_backend``."""
    assert list(public.__all__) == list(ENTRIES)
    for name in ENTRIES:
        fn = getattr(public, name)
        assert getattr(fn, "is_experimental", False) is False, name
        assert fn.__name__ == name
        assert getattr(_backend(), name) is getattr(_backend("cake_backend"), name), name


@pytest.mark.parametrize("name", NO_OTHER_MODULES)
def test_there_is_no_other_mode_or_kcp_module(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_the_namespaces_list_the_det_package_only():
    """In a fresh interpreter: ``cake_pplx`` and ``cake_pplx._backends`` list ``cake_engram_lookup_det_train`` and no
    other ``cake_engram_lookup`` module; the other module paths do not import."""
    code = (
        "import json, pkgutil\n"
        "import cake_pplx, cake_pplx._backends\n"
        "spaces = (cake_pplx, cake_pplx._backends)\n"
        "names = {m.name for p in spaces for m in pkgutil.iter_modules(p.__path__)}\n"
        "names |= {n for p in spaces for n in dir(p)}\n"
        "mine = 'cake_engram_lookup_det_train'\n"
        "others = sorted(n for n in names if n.startswith('cake_engram_lookup') and n != mine)\n"
        "print(json.dumps(dict(others=others, listed=mine in names)))\n"
    )
    assert _fresh(code) == dict(others=[], listed=True)
    assert_old_names_fail(NO_OTHER_MODULES, cwd=Path(__file__).resolve().parents[2])


def test_the_shared_mode_rules_hold():
    """The rules shared by every family (``tests.cake.test_helpers.cake_mode_names``): the facade's public names carry
    ``det`` after the op, no entry point takes ``deterministic``, and the ``cake_pplx`` namespaces, the backend
    package and ``cake_backend`` bind the facade's entries and the backend objects they call only under the facade's
    names."""
    package, backend = _backend(), _backend("cake_backend")
    assert_mode_names(public)
    assert_no_deterministic_keyword([getattr(public, name) for name in public.__all__])
    assert_same_object_same_name(
        public,
        (cake_pplx, cake_pplx._backends, package, backend),
        forwards=[(name, getattr(backend, name)) for name in FORWARDED],
    )


def test_public_names_carry_the_mode():
    """The backend objects the facade forwards to carry ``det`` after the op (the shared rule, applied to the backend's
    names); every package ``__all__`` name is such an object or an ALL_CAPS constant, and the backend's internal
    helpers (the launch, the checks, the device helpers) are not package attributes."""
    backend = _backend("cake_backend")
    package = _backend()
    assert_mode_names(package, names=FORWARDED)
    for name in FORWARDED:
        assert name.startswith("cake_engram_lookup_det_"), name
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
    """The facade entries and the backend functions they call accept no ``deterministic`` keyword (no such parameter,
    no ``**kwargs``); passing it to the facade is a ``TypeError`` from argument binding, before any work.  The family
    has no autograd entry."""
    backend = _backend("cake_backend")
    fns = {}
    for name in ENTRIES:
        fns[f"facade.{name}"] = getattr(public, name)
        fns[f"backend.{name}"] = getattr(backend, name)
    autograd = [
        cls
        for cls in vars(backend).values()
        if inspect.isclass(cls) and issubclass(cls, torch.autograd.Function) and cls.__module__ == backend.__name__
    ]
    assert autograd == []  # a forward step only: no autograd entry
    for label, fn in fns.items():
        signature = inspect.signature(fn)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)
        assert "deterministic" not in signature.parameters, label
        assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values()), label
    weight = torch.zeros(4, 256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scale = torch.zeros(4, 8, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    ids = torch.zeros(3, dtype=torch.int64)
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            public.cake_engram_lookup_det_train_forward(weight, scale, ids, deterministic=value)
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            public.cake_engram_lookup_det_train_split_packed(
                torch.zeros(4, 264, dtype=torch.uint8), deterministic=value
            )


def test_the_backend_never_reads_the_torch_deterministic_setting():
    """The module is the mode: the lookup neither alerts nor changes path under ``torch.use_deterministic_algorithms``
    (strict or ``warn_only``): no module of the backend package names torch's deterministic-algorithms entry points.
    The facade documents the behaviour."""
    root = Path(_backend("cake_jit").__file__).parent
    paths = sorted(root.glob("*.py"))
    assert {"cake_backend.py", "cake_jit.py", "__init__.py"} <= {p.name for p in paths}
    for path in paths:
        text = path.read_text()
        for token in TORCH_DETERMINISTIC_TOKENS:
            assert token not in text, (path.name, token)
    assert "use_deterministic_algorithms" in public.cake_engram_lookup_det_train_forward.__doc__


def test_capabilities_map_to_the_three_architectures(monkeypatch):
    """Capabilities (10, 0), (10, 3) and (10, 7) map to ``sm_100a``, ``sm_103a`` and ``sm_107a``; every other
    capability maps to no target.  A device of capability (10, 7) is served as ``sm_107a``."""
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
    clear = backend.device_arch.cache_clear
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    clear()
    try:
        assert backend.device_arch(0) == "sm_107a"
    finally:
        monkeypatch.undo()
        clear()


@pytest.mark.parametrize("capability", ((12, 0), (12, 1), (11, 0), (9, 0), (10, 1)), ids=lambda c: f"cc{c[0]}{c[1]}")
def test_an_unsupported_capability_is_refused(capability, monkeypatch):
    """A device of a capability without a build is a ``RuntimeError`` naming the delivered architectures."""
    backend = _backend("cake_backend")
    clear = backend.device_arch.cache_clear
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            backend.device_arch(0)
    finally:
        monkeypatch.undo()
        clear()


def test_package_identity_and_build_identity():
    """``build.json`` and the registry name the package; each of the registry's two libraries is a
    ``cake_engram_lookup_det_train`` program built for ``sm_100a``, ``sm_103a`` and ``sm_107a``, and their names are
    the stems of ``build.json``."""
    jit = _backend("cake_jit")
    assert jit.PACKAGE == BACKEND
    assert jit.LIBRARY_PREFIX == "cake_engram_lookup_det_train"
    manifest = _manifest_or_skip()
    assert manifest["package"] == BACKEND
    names = {entry["name"] for entry in manifest["libraries"]}
    assert names == {f"{rec['library']}_{arch}" for rec in jit.MODULES.values() for arch in rec["arches"]}
    pattern = r"cake_engram_lookup_det_train_[0-9a-f]{20}_sm_10[037]a"
    assert all(re.fullmatch(pattern, name) for name in names), sorted(names)
    assert {entry["arch"] for entry in manifest["libraries"]} == set(ARCHES)
    assert len({rec["library"] for rec in jit.MODULES.values()}) == len(jit.MODULES) == 2
    assert len(names) == len(jit.MODULES) * len(ARCHES) == 2 * len(ARCHES)


def test_manifest_min_cuda():
    """Every library of an architecture needs the toolkit of :data:`MIN_CUDA`: 12.8 for ``sm_100a``, 12.9 for
    ``sm_103a``, 13.5 for ``sm_107a``."""
    manifest = _manifest_or_skip()
    floors = {}
    for entry in manifest["libraries"]:
        floors.setdefault(entry["arch"], set()).add(entry["min_cuda"])
    assert floors == {arch: {floor} for arch, floor in MIN_CUDA.items()}, floors


def test_manifest_numerics_flags_keep_fp32_subnormals():
    """The build flags of every library keep FP32 subnormals (the 568 subnormal products and the scale 2^-127): no
    ``--use_fast_math``, no ``--ftz=true``, and ``--ftz=false``; division and square root stay IEEE
    (``--prec-div=true``, ``--prec-sqrt=true``)."""
    manifest = _manifest_or_skip()
    common = list(manifest["common"]["nvcc_flags"])
    for entry in manifest["libraries"]:
        flags = common + list(entry["nvcc_flags"])
        for item in entry.get("nvcc_flags_if", []):
            flags += list(item.get("flags", []))
        for item in manifest["common"].get("nvcc_flags_if", []):
            flags += list(item.get("flags", []))
        assert "--use_fast_math" not in flags, (entry["name"], flags)
        assert "--ftz=true" not in flags, (entry["name"], flags)
        for wanted in ("--ftz=false", "--prec-div=true", "--prec-sqrt=true"):
            assert wanted in flags, (entry["name"], wanted, flags)
