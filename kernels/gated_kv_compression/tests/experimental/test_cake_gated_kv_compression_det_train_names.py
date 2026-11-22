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

Names of the deterministic ratio-2 gated KV compression module ``cake_pplx.cake_gated_kv_compression_det_train``.

The family has one mode: every output element depends on its own pair or token only, with no reduction and no
atomic, so no non-deterministic package exists (no ``cake_gated_kv_compression_nondet_train`` module, facade or
backend), and the kernel is single-rank (no ``_kcp`` module).  The public names are the facade's ``__all__`` and the
backend objects it forwards to (``cake_pplx._backends.cake_gated_kv_compression_det_train``): each carries the mode
(``det``) as a token right after the op, and no entry point takes a ``deterministic`` keyword.  The rules shared by
every family are checked with ``tests.cake.test_helpers.cake_mode_names``; the checks here are this family's own.
Build identities (program and library stems ``cake_gated_kv_compression_det_train_<hex>``, kernel symbols, csrc file
names) keep their names.  The package serves capabilities (10, 0), (10, 3) and (10, 7) only; each role has one record
per architecture, and the forwards' records follow the architecture (``fwd`` and ``fwd_unfused``, 128-bit lanes, on
sm_100a / sm_103a; ``fwd_sm_107a`` and ``fwd_unfused_sm_107a``, 256-bit lanes, on sm_107a).
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
import cake_pplx.cake_gated_kv_compression_det_train as public
from tests.cake.test_helpers.cake_mode_names import (
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
)

pytestmark = pytest.mark.cpu_only

BACKEND = "cake_pplx._backends.cake_gated_kv_compression_det_train"
NAMES = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeGatedKvCompressionDetCzFunction",
    "CakeGatedKvCompressionDetFunction",
    "cake_gated_kv_compression_det",
    "cake_gated_kv_compression_det_cz",
    "cake_gated_kv_compression_det_train_backward",
    "cake_gated_kv_compression_det_train_forward",
    "cake_gated_kv_compression_det_train_memory_bytes",
    "is_cake_gated_kv_compression_det_train_supported",
)
# The facade functions that call a backend function of the same name.
FORWARDED = (
    "cake_gated_kv_compression_det",
    "cake_gated_kv_compression_det_cz",
    "cake_gated_kv_compression_det_train_backward",
    "cake_gated_kv_compression_det_train_forward",
    "cake_gated_kv_compression_det_train_memory_bytes",
)
# The backend objects the facade serves lazily under the same name.
LAZY = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeGatedKvCompressionDetCzFunction",
    "CakeGatedKvCompressionDetFunction",
    "is_cake_gated_kv_compression_det_train_supported",
)
# Module paths of the modes and routes this family does not have.
ABSENT_MODULES = (
    "cake_pplx.cake_gated_kv_compression_nondet_train",
    "cake_pplx._backends.cake_gated_kv_compression_nondet_train",
    "cake_pplx.cake_gated_kv_compression_det_train_kcp",
    "cake_pplx._backends.cake_gated_kv_compression_det_train_kcp",
    "cake_pplx.cake_gated_kv_compression_nondet_train_kcp",
)
# Internal helpers of the backend (the launchers and the device helpers): not package attributes.
INTERNAL = ("launch_pair_table", "launch_fwd", "launch_bwd", "device_arch", "arch_for_capability")
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


def test_the_facade_lists_the_documented_names():
    """The facade's ``__all__`` is exactly the documented names, each a stable API (no experimental marking); the
    backend package lists the same objects."""
    assert sorted(public.__all__) == sorted(NAMES)
    package = _backend()
    assert sorted(package.__all__) == sorted(NAMES)
    for name in FORWARDED:
        fn = getattr(public, name)
        assert getattr(fn, "is_experimental", False) is False, name
        assert fn.__name__ == name
        assert getattr(package, name) is getattr(_backend("cake_backend"), name), name
    for name in LAZY:
        assert getattr(public, name) is getattr(package, name), name


@pytest.mark.parametrize("name", ABSENT_MODULES)
def test_there_is_no_nondet_and_no_kcp_module(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_the_namespaces_list_the_det_package_only():
    """In a fresh interpreter: ``cake_pplx`` and ``cake_pplx._backends`` list ``cake_gated_kv_compression_det_train``
    and no ``nondet`` or ``kcp`` module of the family; those module paths do not import."""
    code = (
        "import json, pkgutil\n"
        "import cake_pplx, cake_pplx._backends\n"
        "spaces = (cake_pplx, cake_pplx._backends)\n"
        "names = {m.name for p in spaces for m in pkgutil.iter_modules(p.__path__)}\n"
        "names |= {n for p in spaces for n in dir(p)}\n"
        "mine = sorted(n for n in names if n.startswith('cake_gated_kv_compression'))\n"
        "print(json.dumps(dict(mine=mine)))\n"
    )
    assert _fresh(code) == dict(mine=["cake_gated_kv_compression_det_train"])
    assert_old_names_fail(ABSENT_MODULES, cwd=Path(__file__).resolve().parents[2])


def test_the_shared_mode_rules_hold():
    """The rules shared by every family (``tests.cake.test_helpers.cake_mode_names``): the facade's public names carry
    ``det`` after the op, no entry point takes ``deterministic``, and the ``cake_pplx`` namespaces, the backend
    package and ``cake_backend`` bind the facade's objects and the backend objects it calls only under the facade's
    names."""
    package, backend = _backend(), _backend("cake_backend")
    assert_mode_names(public)
    assert_no_deterministic_keyword([getattr(public, name) for name in public.__all__])
    assert_same_object_same_name(
        public,
        (cake_pplx, cake_pplx._backends, package, backend),
        forwards=[(name, getattr(backend, name)) for name in FORWARDED],
    )


def test_backend_names_carry_the_mode():
    """The backend objects the facade forwards to carry ``det`` after the op; every package ``__all__`` name is such an
    object or an ALL_CAPS constant, and the backend's internal helpers are not package attributes."""
    backend = _backend("cake_backend")
    package = _backend()
    assert_mode_names(package)
    for name in package.__all__:
        assert name in backend.__all__, name
        assert getattr(package, name) is getattr(backend, name), name
    for name in INTERNAL:
        assert hasattr(backend, name), name
        assert not hasattr(package, name), name


def test_no_entry_takes_a_mode_keyword():
    """The facade entries, the backend functions they call and the autograd nodes accept no ``deterministic`` keyword
    (no such parameter, no ``**kwargs``)."""
    backend = _backend("cake_backend")
    fns = {}
    for name in FORWARDED:
        fns[f"facade.{name}"] = getattr(public, name)
        fns[f"backend.{name}"] = getattr(backend, name)
    for label, fn in fns.items():
        signature = inspect.signature(fn)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)
        assert "deterministic" not in signature.parameters, label
        assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values()), label
    autograd = sorted(
        name
        for name, cls in vars(backend).items()
        if inspect.isclass(cls) and issubclass(cls, torch.autograd.Function) and cls.__module__ == backend.__name__
    )
    assert autograd == ["CakeGatedKvCompressionDetCzFunction", "CakeGatedKvCompressionDetFunction"]
    for name in autograd:
        for method in ("forward", "backward"):
            assert "deterministic" not in inspect.signature(getattr(getattr(backend, name), method)).parameters


def test_the_package_raises_no_deterministic_alert():
    """A ``det`` package: no module of the backend reads PyTorch's deterministic-algorithms switches (there is no
    alert to raise), and the forward documents its behaviour under ``torch.use_deterministic_algorithms``."""
    for module in ("cake_backend", "cake_jit"):
        source = inspect.getsource(_backend(module))
        for switch in ("are_deterministic_algorithms_enabled", "is_deterministic_algorithms_warn_only_enabled"):
            assert switch not in source, (module, switch)
    assert "use_deterministic_algorithms" in public.cake_gated_kv_compression_det_train_forward.__doc__


def test_capabilities_map_to_the_three_architectures(monkeypatch):
    """Capabilities (10, 0), (10, 3) and (10, 7) map to ``sm_100a``, ``sm_103a`` and ``sm_107a``; every other
    capability maps to no target.  A device of capability (10, 7) is served as ``sm_107a``, and a device of another
    capability is a ``RuntimeError`` naming the delivered architectures."""
    jit, backend = _backend("cake_jit"), _backend("cake_backend")
    assert tuple(jit.ARCHES) == ARCHES and tuple(backend.ARCHES) == ARCHES and tuple(public.ARCHES) == ARCHES
    assert dict(jit.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    assert tuple(public.SUPPORTED_COMPUTE_CAPABILITIES) == tuple(sorted(ARCH_BY_CAPABILITY))
    for capability, arch in ARCH_BY_CAPABILITY.items():
        assert backend.arch_for_capability(capability) == arch
    for capability in ((10, 1), (9, 0), (11, 0), (12, 0), (12, 1)):
        assert backend.arch_for_capability(capability) is None, capability
    for role in jit.ROLES:
        for arch in ARCHES:
            serving = [name for name, rec in jit.MODULES.items() if rec["role"] == role and arch in rec["arches"]]
            assert len(serving) == (1 if arch in jit.ROLE_ARCHES.get(role, ARCHES) else 0), (role, arch, serving)
    assert {arch for rec in jit.MODULES.values() for arch in rec["arches"]} == set(ARCHES)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    jit.device_arch.cache_clear()
    try:
        assert backend.device_arch(0) == "sm_107a"
    finally:
        monkeypatch.undo()
        jit.device_arch.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (11, 0))
    jit.device_arch.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            backend.device_arch(0)
    finally:
        monkeypatch.undo()
        jit.device_arch.cache_clear()


def test_package_identity_and_build_identity():
    """``build.json`` and the registry name the package; every library stem is a
    ``cake_gated_kv_compression_det_train`` program built for ``sm_100a``, ``sm_103a`` and ``sm_107a``, and the
    registry's library names are the stems of ``build.json``.  Each record names its own program, but the records of
    each backward role (``bwd`` and ``bwd_sm_107a``, ``bwd_small`` and ``bwd_small_sm_107a``, ``bwd_warp``,
    ``bwd_warp_sm_103a`` and ``bwd_warp_sm_107a``), each one program on disjoint architectures, and no library is built
    twice for one architecture."""
    jit = _backend("cake_jit")
    assert jit.PACKAGE == BACKEND
    assert jit.LIBRARY_PREFIX == "cake_gated_kv_compression_det_train"
    path = Path(jit.__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    manifest = json.loads(path.read_text())
    assert manifest["package"] == BACKEND
    names = {entry["name"] for entry in manifest["libraries"]}
    assert names == {f"{rec['library']}_{arch}" for rec in jit.MODULES.values() for arch in rec["arches"]}
    pattern = r"cake_gated_kv_compression_det_train_[0-9a-f]{20}_sm_10[037]a"
    assert all(re.fullmatch(pattern, name) for name in names), sorted(names)
    assert {entry["arch"] for entry in manifest["libraries"]} == set(ARCHES)
    stems = {rec["library"] for rec in jit.MODULES.values()}
    assert len(stems) == len(jit.MODULES) - 4 and len(names) == sum(len(rec["arches"]) for rec in jit.MODULES.values())
    for first, second in (
        ("bwd", "bwd_sm_107a"),
        ("bwd_small", "bwd_small_sm_107a"),
        ("bwd_warp", "bwd_warp_sm_103a"),
        ("bwd_warp", "bwd_warp_sm_107a"),
    ):
        assert jit.MODULES[first]["library"] == jit.MODULES[second]["library"], first


# The forward record of each architecture and the width of its C / Z loads: 128-bit (4 FP32 channels a lane, 128 lanes
# a row) on sm_100a and sm_103a, 256-bit (8 channels, 64 lanes) on sm_107a, the faster schedule of each.
FORWARD = {"sm_100a": ("fwd", 128, "v4"), "sm_103a": ("fwd", 128, "v4"), "sm_107a": ("fwd_sm_107a", 64, "v8")}
# The same for the forward of more than fuse_docs documents, which builds no pair table.
UNFUSED = {
    "sm_100a": ("fwd_unfused", 128, "v4"),
    "sm_103a": ("fwd_unfused", 128, "v4"),
    "sm_107a": ("fwd_unfused_sm_107a", 64, "v8"),
}


def test_the_forward_record_follows_the_architecture(monkeypatch):
    """The ``fwd`` role resolves to ``fwd`` on sm_100a and sm_103a and to ``fwd_sm_107a`` on sm_107a (also from the
    device capability), and the ``fwd_unfused`` role to ``fwd_unfused`` and ``fwd_unfused_sm_107a``; each pair is two
    distinct programs whose rendered kernels read ``C`` and ``Z`` with 128-bit and 256-bit loads, and whose grid rules
    give a row 128 and 64 lanes (``rows_capped`` for the ``fwd_unfused`` records)."""
    jit = _backend("cake_jit")
    for role, records in (("fwd", FORWARD), ("fwd_unfused", UNFUSED)):
        for arch, (name, lanes, _) in records.items():
            assert jit.role_record(role, arch) == name, (role, arch)
            kind = "rows_capped" if role == "fwd_unfused" else "rows"
            assert jit.MODULES[name]["grid_rule"] == {"kind": kind, "lanes_per_row": lanes, "rows_per_lane_group": 1}
    assert jit.MODULES["fwd"]["library"] != jit.MODULES["fwd_sm_107a"]["library"]
    assert jit.MODULES["fwd_unfused"]["library"] != jit.MODULES["fwd_unfused_sm_107a"]["library"]
    for capability, arch in ARCH_BY_CAPABILITY.items():
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, capability=capability: capability)
        jit.device_arch.cache_clear()
        try:
            assert jit.role_record("fwd", jit.device_arch(0)) == FORWARD[arch][0], capability
        finally:
            monkeypatch.undo()
            jit.device_arch.cache_clear()
    csrc = Path(jit.__file__).resolve().parent / "csrc"
    if not (csrc.parent / "build.json").is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    for arch, (name, _, width) in (*FORWARD.items(), *UNFUSED.items()):
        source = (csrc / f"{jit.MODULES[name]['library']}_kernel.cu").read_text()
        loads = set(re.findall(r"ld\.global[\w:.]*\.(v[48])\.b32", source))
        assert loads == {width}, (arch, name, loads)
