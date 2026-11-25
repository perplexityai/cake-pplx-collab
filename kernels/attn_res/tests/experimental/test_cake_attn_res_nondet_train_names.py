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

Names of the non-deterministic attention-residual module ``cake_pplx.cake_attn_res_nondet_train``.

The public names are the facade's ``__all__`` and the backend objects it forwards to or re-exports
(``cake_pplx._backends.cake_attn_res_nondet_train``): each carries the mode (``nondet``) as a token, the other
package's names are not bound (no alias), and no entry point takes a ``deterministic`` keyword.  The rules shared by
every family are checked with ``tests.cake.test_helpers.cake_mode_names``; the checks here are this family's own.
Only the backend reads torch's deterministic mode, for the alert its backward raises; the facade and the registry never
read it.
Build identities (library stems ``cake_attn_res_train_<hex>``, kernel symbols, csrc file names) keep their names.
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
import cake_pplx.cake_attn_res_nondet_train as public
from tests.cake.test_helpers.cake_mode_names import (
    assert_mode_names,
    assert_no_deterministic_keyword,
    assert_old_names_fail,
    assert_same_object_same_name,
)

pytestmark = pytest.mark.cpu_only

BACKEND = "cake_pplx._backends.cake_attn_res_nondet_train"
OTHER_BACKEND = "cake_pplx._backends.cake_attn_res_det_train"
ENTRIES = (
    "cake_attn_res_nondet",
    "cake_attn_res_nondet_train_backward",
    "cake_attn_res_nondet_train_forward",
    "cake_attn_res_nondet_train_retained_bytes",
    "cake_attn_res_nondet_train_workspace_bytes",
    "is_cake_attn_res_nondet_train_supported",
)
# Backend classes the facade re-exports (resolved on first access).
REEXPORTED = ("CakeAttnResNondetFunction",)
CONSTANTS = ("ARCHES", "HIDDEN", "MAX_VALUES", "SUPPORTED_COMPUTE_CAPABILITIES")
# The backend objects the facade forwards to or re-exports (``cake_backend``, also exported by the package).
FORWARDED = ENTRIES + REEXPORTED
# The other package's public names: never bound by this module or its backend.
OTHER_NAMES = (
    "cake_attn_res_det",
    "cake_attn_res_det_train_backward",
    "cake_attn_res_det_train_forward",
    "cake_attn_res_det_train_retained_bytes",
    "cake_attn_res_det_train_workspace_bytes",
    "is_cake_attn_res_det_train_supported",
    "CakeAttnResDetFunction",
)
# torch's deterministic-mode queries.
READS_TORCH_MODE = re.compile(r"torch\.(?:are|is)_deterministic_algorithms_\w*enabled\(")


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


def test_other_mode_names_are_not_bound():
    """No name of the other package is an attribute of the facade, the backend package or ``cake_backend``; the
    ``cake_pplx`` namespaces list both modules of the family (fresh interpreter)."""
    for name in OTHER_NAMES:
        assert not hasattr(public, name), name
        for module in (_backend(), _backend("cake_backend")):
            assert not hasattr(module, name), (module.__name__, name)
            assert name not in module.__all__, (module.__name__, name)
    code = (
        "import json, pkgutil\n"
        "import cake_pplx, cake_pplx._backends\n"
        "names = {m.name for m in pkgutil.iter_modules(cake_pplx.__path__)}\n"
        "backends = {m.name for m in pkgutil.iter_modules(cake_pplx._backends.__path__)}\n"
        "print(json.dumps(dict(facade='cake_attn_res_nondet_train' in names,\n"
        "    backend='cake_attn_res_nondet_train' in backends)))\n"
    )
    assert _fresh(code) == dict(facade=True, backend=True)


def test_facade_all_is_the_entries_reexports_and_constants():
    """The facade's ``__all__`` is exactly its six entries, one re-export and four constants; each entry is a stable
    API (no experimental marking) named after the module; the constants equal the backend's."""
    assert sorted(public.__all__) == sorted(ENTRIES + REEXPORTED + CONSTANTS)
    backend = _backend("cake_backend")
    for name in ENTRIES:
        fn = getattr(public, name)
        assert name.startswith("cake_attn_res_nondet") or name.startswith("is_cake_attn_res_nondet_train")
        assert getattr(fn, "is_experimental", False) is False, name
        assert fn.__name__ == name, name
    for name in REEXPORTED:
        assert getattr(public, name) is getattr(backend, name), name
    for name in CONSTANTS:
        assert getattr(public, name) == getattr(backend, name), name
    assert public.HIDDEN == 7168 and public.MAX_VALUES == 9
    assert public.SUPPORTED_COMPUTE_CAPABILITIES == ((10, 0), (10, 3), (10, 7))
    assert public.ARCHES == ("sm_100a", "sm_103a", "sm_107a")


def test_the_shared_mode_rules_hold():
    """The rules shared by every family (``tests.cake.test_helpers.cake_mode_names``): the facade's public names
    carry ``nondet`` after the op, no entry point takes ``deterministic``, the ``cake_pplx`` namespaces, the backend
    package and ``cake_backend`` bind each facade object and each backend object a facade entry calls only under the
    facade's name, and the other mode's names fail at import in a fresh interpreter."""
    package, backend = _backend(), _backend("cake_backend")
    old_names = [("cake_pplx.cake_attn_res_nondet_train", name) for name in OTHER_NAMES]
    old_names += [(module.__name__, name) for module in (package, backend) for name in OTHER_NAMES]
    assert_mode_names(public)
    assert_no_deterministic_keyword([getattr(public, name) for name in public.__all__ if name not in CONSTANTS])
    assert_same_object_same_name(
        public,
        (cake_pplx, cake_pplx._backends, package, backend),
        forwards=[(name, getattr(backend, name)) for name in ENTRIES],
    )
    assert_old_names_fail((), old_names, cwd=Path(__file__).resolve().parents[2])


def test_public_names_carry_the_mode():
    """Every backend object the facade forwards to or re-exports carries ``nondet`` after the op; every package
    ``__all__`` name is such an object or an ALL_CAPS constant, and the backend's internal helpers (``load``,
    ``record``, the stage launchers, ...) are not package attributes."""
    backend = _backend("cake_backend")
    package = _backend()
    assert_mode_names(package, "nondet", names=FORWARDED)
    for name in FORWARDED:
        assert name in backend.__all__ and name in package.__all__, name
        assert getattr(package, name) is getattr(backend, name), name
    for name in package.__all__:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", name) or name in FORWARDED, name
    internal = [n for n in backend.__all__ if n not in FORWARDED and not re.fullmatch(r"[A-Z][A-Z0-9_]*", n)]
    assert {"load", "record", "launch_fwd", "launch_bwd"} <= set(internal)
    for name in internal:
        assert not hasattr(package, name), name


def test_no_entry_takes_a_mode_keyword():
    """No facade entry, forwarded backend callable or autograd ``Function.forward`` accepts a ``deterministic``
    keyword; passing it to a facade entry is a ``TypeError`` from argument binding, before any work."""
    backend = _backend("cake_backend")
    fns = {f"facade.{name}": getattr(public, name) for name in ENTRIES}
    for name in ENTRIES:
        fns[f"backend.{name}"] = getattr(backend, name)
    for name in REEXPORTED:
        fns[f"backend.{name}.forward"] = getattr(backend, name).forward
    autograd = sorted(
        cls.__name__
        for cls in vars(backend).values()
        if inspect.isclass(cls) and issubclass(cls, torch.autograd.Function) and cls.__module__ == backend.__name__
    )
    assert autograd == sorted(REEXPORTED)
    for label, fn in fns.items():
        signature = inspect.signature(fn)
        with pytest.raises(TypeError):
            signature.bind_partial(deterministic=False)
        assert "deterministic" not in signature.parameters, label
        assert all(p.kind is not p.VAR_KEYWORD for p in signature.parameters.values()), label
    v = torch.zeros(1, 7168, dtype=torch.bfloat16)
    w = torch.ones(7168)
    state = torch.zeros(1, 1)
    calls = {
        "cake_attn_res_nondet": ([v], w),
        "cake_attn_res_nondet_train_forward": ([v], w),
        "cake_attn_res_nondet_train_backward": (v, [v], w, state, state, state),
        "cake_attn_res_nondet_train_retained_bytes": (1, 1),
        "cake_attn_res_nondet_train_workspace_bytes": (1, 1),
        "is_cake_attn_res_nondet_train_supported": (),
    }
    for name, args in calls.items():
        for value in (True, False):
            with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
                getattr(public, name)(*args, deterministic=value)


def test_only_the_backend_reads_torch_mode():
    """The facade, the backend package and the registry never read torch's deterministic mode; ``cake_backend`` reads
    it for the alert of its backward (``... does not have a deterministic implementation ...``; the API suite checks
    the strict and ``warn_only`` modes on a device)."""
    assert not READS_TORCH_MODE.search(inspect.getsource(public))
    for module_name in ("cake_jit", None):
        assert not READS_TORCH_MODE.search(inspect.getsource(_backend(module_name))), module_name
    source = inspect.getsource(_backend("cake_backend"))
    assert READS_TORCH_MODE.search(source)
    assert "does not have a deterministic implementation" in source


def test_package_identity_and_build_identity():
    """``build.json`` and the registry name the package; every library stem carries the package's own prefix
    ``cake_attn_res_train_<20 hex>`` and is not a library of the other package."""
    jit = _backend("cake_jit")
    assert jit.PACKAGE == BACKEND
    assert jit.LIBRARY_PREFIX == "cake_attn_res_train"
    libraries = {rec["library"] for rec in jit.MODULES.values()}
    for library in libraries:
        assert re.fullmatch(r"cake_attn_res_train_[0-9a-f]{20}", library), library
    other = importlib.import_module(f"{OTHER_BACKEND}.cake_jit")
    assert not libraries & {rec["library"] for rec in other.MODULES.values()}
    path = Path(jit.__file__).resolve().parent / "build.json"
    if not path.is_file():
        pytest.skip("the package's programs are not rendered in this checkout")
    manifest = json.loads(path.read_text())
    assert manifest["package"] == BACKEND
    assert all(entry["name"].startswith("cake_attn_res_train_") for entry in manifest["libraries"])


def test_capability_map():
    """The backend maps (10, 0), (10, 3) and (10, 7) to their architectures and nothing else."""
    backend = _backend("cake_backend")
    assert backend.arch_for_capability((10, 0)) == "sm_100a"
    assert backend.arch_for_capability((10, 3)) == "sm_103a"
    assert backend.arch_for_capability((10, 7)) == "sm_107a"
    for capability in ((9, 0), (10, 1), (11, 0), (12, 0)):
        assert backend.arch_for_capability(capability) is None, capability
