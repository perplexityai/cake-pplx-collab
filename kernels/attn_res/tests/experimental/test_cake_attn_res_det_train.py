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

Backend suite of ``cake_pplx._backends.cake_attn_res_det_train``, the deterministic package of the attention
residual (values of width 7168).

Without a device: the registry's records, stages, architectures and launch facts; the persistent grid sizes; every
stage launcher passes every kernel parameter of its records in plan order, the first value in the slots past ``n``
(the FFI entry replaced by a recorder); the workspace and retained-state sizes; ``build.json`` covers every record on
every architecture and its source digests; the loader imports standalone; a prebuilt library is preferred; the package
reads no environment variable but the JIT shim's.

On SM100 / SM103 / SM107: every record loads; the forward record writes every element of NaN-filled ``out``, ``p``,
``r`` and ``d`` and meets the naive BF16 tier against an FP64 oracle (``tests/cake_naive.py``, 32 draws); the backward
from the kernel's own forward state writes every element of NaN-filled ``dv_j`` and meets the tier for every ``dv_j``
and ``dw``; ``dw_reduce`` alone gives the fixed-order sum of random rows; every output repeats bit for bit run to run;
``n`` from 1 to 9, scaled, peaked and all-zero-row values.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_attn_res_naive import HIDDEN, MAX_VALUES, judge_draws, make_inputs
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import judge_case

PACKAGE = "cake.cake_attn_res_det_train"
pkg = cake_package(PACKAGE)
cake_backend = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STAGES = ("fwd", "bwd", "dw_reduce")
RECORDS = ("bwd", "dw_reduce", "fwd")
PERSISTENT_RECORDS = ("bwd", "fwd")
ENTRIES = (
    "cake_attn_res_det",
    "cake_attn_res_det_train_backward",
    "cake_attn_res_det_train_forward",
    "cake_attn_res_det_train_retained_bytes",
    "cake_attn_res_det_train_workspace_bytes",
    "is_cake_attn_res_det_train_supported",
)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the package ships {ARCHES}; this device is {capability}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------------------------
# Without a device: the registry and the host plan


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert cake_jit.ARCH_BY_CAPABILITY == {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
    assert cake_jit.STAGES == STAGES
    assert cake_jit.HIDDEN == HIDDEN and cake_jit.MAX_VALUES == MAX_VALUES == 9
    assert sorted(cake_jit.MODULES) == sorted(RECORDS)
    for name, rec in cake_jit.MODULES.items():
        assert rec["arches"] == list(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert rec["stage"] == name, name
        assert cake_jit.record(name) == name
        assert rec["library"].startswith(cake_jit.LIBRARY_PREFIX + "_"), name
        assert rec["kernel_symbol"] == f"kernel_{rec['library']}", name
        launch = rec["launch"]
        assert set(launch) == {"block", "cluster", "cooperative", "dynamic_smem_bytes", "use_pdl"}, name
        assert launch["cooperative"] is False, name
        kinds = [kind for kind, _ in rec["arg_plan"]]
        assert kinds[-3:] == ["grid", "grid", "grid"], name
        assert cake_backend.parameters(name), name
        cake_backend._plan(name, rec["stage"])
    with pytest.raises(ValueError):
        cake_jit.record("main")


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name in cake_jit.MODULES:
        for arch in ARCHES:
            assert cake_backend.library_name(name, arch) == f"{cake_jit.MODULES[name]['library']}_{arch}"
        with pytest.raises(RuntimeError):
            cake_backend.library_name(name, "sm_90a")
    with pytest.raises(KeyError):
        cake_backend.record("bwd_main")


def test_the_resident_group_table_is_bounded() -> None:
    """Every persistent record has its CTAs per SM; every table entry is at most CTAs per SM * SMs / cluster CTAs
    groups; a device the table does not hold takes the fallback fraction."""
    assert sorted(cake_jit.CTAS_PER_SM) == sorted(PERSISTENT_RECORDS)
    fraction = cake_jit.UNMEASURED_FRACTION
    for (arch, sms), table in cake_jit.RESIDENT_GROUPS.items():
        assert arch in ARCHES
        for per_sm, held in table.items():
            for name in PERSISTENT_RECORDS:
                assert 1 <= held <= per_sm * sms // cake_jit.cluster_ctas(name), (arch, sms, per_sm)
    for name in PERSISTENT_RECORDS:
        ctas = cake_jit.cluster_ctas(name)
        per_sm = cake_jit.CTAS_PER_SM[name]
        if ("sm_103a", 160) not in cake_jit.RESIDENT_GROUPS:
            assert cake_jit.resident_groups(name, "sm_103a", 160) == max(1, int(fraction * per_sm * 160) // ctas)
        assert cake_jit.resident_groups(name, "sm_107a", 1) >= 1


@pytest.mark.parametrize("sms", (1, 148, 152, 160, 212))
@pytest.mark.parametrize("T", (1, 2, 3, 129, 4097, 16380, 32768, 2**31 - 1))
def test_grids_are_persistent_groups(T, sms) -> None:
    """``fwd`` and ``bwd`` run ``min(T, resident groups)`` groups (at least one) of whole clusters."""
    for arch in ARCHES:
        for name in PERSISTENT_RECORDS:
            resident = cake_jit.resident_groups(name, arch, sms)
            groups = cake_jit.groups(name, T, arch, sms)
            assert 1 <= groups <= T
            assert groups == max(1, min(T, resident))
            assert cake_jit.grid(name, T, arch, sms) == (cake_jit.cluster_ctas(name) * groups, 1, 1)
        assert cake_jit.grid("dw_reduce", T, arch, sms) == cake_jit.dw_reduce_grid()
    assert cake_jit.dw_reduce_grid() == (HIDDEN // cake_jit.DW_REDUCE_COLUMNS_PER_CTA, 1, 1)
    assert HIDDEN % cake_jit.DW_REDUCE_COLUMNS_PER_CTA == 0


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def entry(self, name: str, arch: str):
        return lambda *args: self.calls.append((name, args))


@pytest.mark.parametrize("n", (1, 2, 5, 9))
def test_stage_launchers_pass_every_kernel_parameter(n, monkeypatch) -> None:
    """Each stage launcher passes every parameter of its record's argument plan, in order, then the grid (``_entry``
    replaced by a recorder; nothing is launched): the values in their slots and the first value in every slot past
    ``n``, the token count, ``n`` and ``eps``."""
    recorder = _Recorder()
    monkeypatch.setattr(cake_backend, "_entry", recorder.entry)
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: "sm_100a")
    T, groups = 5, 2
    bf16 = dict(dtype=torch.bfloat16)
    values = [torch.zeros(T, HIDDEN, **bf16) for _ in range(n)]
    dv = [torch.zeros(T, HIDDEN, **bf16) for _ in range(n)]
    w = torch.zeros(HIDDEN)
    out = torch.zeros(T, HIDDEN, **bf16)
    p, r, d = (torch.zeros(T, n) for _ in range(3))
    dout = torch.zeros(T, HIDDEN, **bf16)
    dw_rows = torch.zeros(groups, HIDDEN)
    dw = torch.zeros(HIDDEN)
    cake_backend.launch_fwd("fwd", values=values, w=w, out=out, p=p, r=r, d=d, eps=1e-5, groups=groups)
    cake_backend.launch_bwd("bwd", dout=dout, values=values, w=w, p=p, r=r, d=d, dv=dv, dw_rows=dw_rows, groups=groups)
    cake_backend.launch_dw_reduce("dw_reduce", dw_rows=dw_rows, dw=dw)
    assert [name for name, _ in recorder.calls] == ["fwd", "bwd", "dw_reduce"]
    for name, args in recorder.calls:
        params = cake_backend.parameters(name)
        assert len(args) == len(params) + 3, (name, params)
        named = dict(zip(params, args, strict=False))
        if name == "dw_reduce":
            assert args[-3:] == cake_jit.dw_reduce_grid()
            assert named["dw_rows"] is dw_rows and named["dw"] is dw and named["G"] == groups
            continue
        assert args[-3:] == (cake_jit.cluster_ctas(name) * groups, 1, 1)
        assert named["T"] == T and named["n"] == n
        assert named["w"] is w and named["p"] is p and named["r"] is r and named["d"] is d
        for j in range(MAX_VALUES):
            assert named[f"v{j}"] is values[j if j < n else 0], (name, j)
        if name == "fwd":
            assert named["out"] is out and named["eps"] == 1e-5
        else:
            assert named["dout"] is dout and named["dw_rows"] is dw_rows
            for j in range(MAX_VALUES):
                assert named[f"dv{j}"] is dv[j if j < n else 0], (name, j)


def test_workspace_and_retained_bytes(monkeypatch) -> None:
    """The backward's workspace is one FP32 ``[7168]`` row per launch group of its grid, whatever ``n``; the forward
    keeps ``p``, ``r``, ``d``: ``12 T n`` bytes."""
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: "sm_103a")
    monkeypatch.setattr(cake_backend, "sm_count", lambda index: 152)
    for T in (0, 1, 16380):
        for n in (1, 2, 5, 9):
            assert pkg.cake_attn_res_det_train_retained_bytes(T, n) == 12 * T * n
            rows = cake_jit.groups("bwd", T, "sm_103a", 152) if T else 0
            assert pkg.cake_attn_res_det_train_workspace_bytes(T, n, "cuda:0") == 4 * HIDDEN * rows
    for bad in ((-1, 2), (1, 0), (1, 10)):
        with pytest.raises(ValueError):
            pkg.cake_attn_res_det_train_retained_bytes(*bad)
        with pytest.raises(ValueError):
            pkg.cake_attn_res_det_train_workspace_bytes(*bad)
    with pytest.raises(ValueError, match="CUDA"):
        pkg.cake_attn_res_det_train_workspace_bytes(1, 2, "cpu")


def test_the_package_exports_the_backend_entries() -> None:
    for name in ENTRIES:
        assert getattr(pkg, name) is getattr(cake_backend, name), name
        assert name in pkg.__all__ and name in cake_backend.__all__, name


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]}
    assert set(entries) == wanted
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert name.endswith("_" + entry["arch"]), name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    sm107 = [entry for entry in manifest["libraries"] if entry["arch"] == "sm_107a"]
    assert all(entry["gencode"] == "-gencode=arch=compute_107a,code=sm_107a" for entry in sm107)


def test_the_loader_imports_standalone_by_path() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a fresh process reads the manifest."""
    _manifest_or_skip()
    loader = str(PACKAGE_DIR / "cake_build.py")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {loader!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    libraries = len({f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]})
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {libraries}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path."""
    _manifest_or_skip()
    cake_build = cake_package(PACKAGE, "cake_build")
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_backend.library_name(name, arch)
            assert cake_build.prebuilt_path(library) == PACKAGE_DIR / "aot" / f"{library}.so"
            assert cake_backend.is_prebuilt(name, arch) == cake_build.prebuilt_path(library).is_file()


# The variables the JIT shim (``_cake_pplx_jit.py``) reads: the toolchain, cache and JIT switches.
JIT_SHIM_ENVIRONMENT_VARIABLES = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CAKE_PPLX_CACHE_DIR",
        "CAKE_PPLX_NVCC",
        "CXX",
        "CAKE_PPLX_JIT_WORKERS",
        "CAKE_PPLX_DISABLE_JIT",
    }
)


def _environment_reads(tree: ast.AST) -> tuple[set[str], list[int]]:
    """``(names, unresolved)``: the variable names ``tree`` reads through ``os.environ.get(...)``,
    ``os.environ[...]``, ``... in os.environ`` or ``os.getenv(...)`` by a string literal or a loop variable over a
    tuple / list of string literals, and the lines of every other ``environ`` / ``getenv`` use."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def keys(expr: ast.AST, at: ast.AST):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.Name):
            node = parents.get(at)
            while node is not None:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == expr.id
                    and isinstance(node.iter, (ast.Tuple, ast.List))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)
                ):
                    return {e.value for e in node.iter.elts}
                node = parents.get(node)
        return None

    names: set[str] = set()
    unresolved: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")):
            continue
        parent = parents.get(node)
        found = None
        if node.attr == "getenv":
            if isinstance(parent, ast.Call) and parent.func is node and parent.args:
                found = keys(parent.args[0], parent)
        elif isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and call.args:
                found = keys(call.args[0], call)
        elif isinstance(parent, ast.Subscript) and parent.value is node:
            found = keys(parent.slice, parent)
        elif (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is node
        ):
            found = keys(parent.left, parent)
        if found is None:
            unresolved.append(node.lineno)
        else:
            names |= found
    return names, unresolved


def test_the_package_reads_no_environment_variable() -> None:
    """No module of the package reads an environment variable, except the JIT shim, which reads exactly
    :data:`JIT_SHIM_ENVIRONMENT_VARIABLES`."""
    offenders = []
    shim = None
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        names, unresolved = _environment_reads(ast.parse(path.read_text()))
        if path.name == "_cake_pplx_jit.py":
            shim = names
            offenders += [f"{path.name}:{line} (variable not resolved)" for line in unresolved]
        else:
            offenders += [f"{path.name}:{line}" for line in unresolved]
            offenders += [f"{path.name}: {name}" for name in sorted(names)]
    assert not offenders, offenders
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107


def test_every_record_loads(device) -> None:
    arch = cake_backend.device_arch(device.index)
    for name in cake_jit.MODULES:
        assert callable(cake_backend.load(name, arch).run), name


# (n, T, data, weights): one value to nine; one token, a few, past one wave of groups, an odd count.
CASES = (
    (1, 7, "scaled", "small"),
    (2, 1, "scaled", "small"),
    (2, 129, "peaked", "normproj"),
    (3, 1025, "zero_rows", "small"),
    (5, 129, "scaled", "normproj"),
    (5, 1025, "peaked", "small"),
    (8, 7, "zero_rows", "normproj"),
    (9, 129, "zero_rows", "small"),
    (9, 1025, "scaled", "small"),
)
CASE_IDS = [f"n{n}-T{T}-{data}-{w}" for n, T, data, w in CASES]


def _record_failures(judge, failures, record_property) -> None:
    judge.record(record_property)
    assert not failures, failures


def _forward_state(ns, device):
    """The forward record launched directly into NaN-filled outputs: ``(out, p, r, d)``."""
    T, n = ns.T, ns.n
    out = torch.full((T, HIDDEN), math.nan, dtype=torch.bfloat16, device=device)
    p, r, d = (torch.full((T, n), math.nan, device=device) for _ in range(3))
    arch = cake_backend.device_arch(device.index)
    groups = cake_jit.groups("fwd", T, arch, cake_backend.sm_count(device.index))
    cake_backend.launch_fwd("fwd", values=ns.values, w=ns.w, out=out, p=p, r=r, d=d, eps=1e-5, groups=groups)
    return out, p, r, d


@pytest.mark.parametrize(("n", "T", "data", "weights"), CASES, ids=CASE_IDS)
def test_the_forward_record_matches_the_reference(n, T, data, weights, device, record_property) -> None:
    """The forward record launched directly writes every element of NaN-filled ``out``, ``p``, ``r`` and ``d`` and
    meets the tier for each."""

    def run(ns):
        out, p, r, d = _forward_state(ns, device)
        torch.cuda.synchronize(device)
        return {"out": out, "p": p, "r": r, "d": d}

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data=data, weights=weights)

    _record_failures(*judge_draws(run, make, seed=11 + T + 100 * n), record_property)


@pytest.mark.parametrize(("n", "T", "data", "weights"), CASES, ids=CASE_IDS)
def test_the_backward_matches_the_reference(n, T, data, weights, device, record_property) -> None:
    """The backward (``bwd`` then ``dw_reduce``) from the kernel's own forward state, with NaN-filled ``dv_j``, meets
    the tier for every ``dv_j`` and ``dw``."""

    def run(ns):
        _, p, r, d = _forward_state(ns, device)
        dv = [torch.full((T, HIDDEN), math.nan, dtype=torch.bfloat16, device=device) for _ in range(n)]
        got_dv, dw = cake_backend.cake_attn_res_det_train_backward(ns.dout, ns.values, ns.w, p, r, d, dv=dv)
        torch.cuda.synchronize(device)
        assert all(a is b for a, b in zip(got_dv, dv, strict=True))
        return {**{f"dv{j}": g for j, g in enumerate(got_dv)}, "dw": dw}

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data=data, weights=weights)

    _record_failures(*judge_draws(run, make, seed=23 + T + 100 * n), record_property)


@pytest.mark.parametrize("G", (1, 7, 296))
def test_dw_reduce_matches_the_reference(G, device, record_property) -> None:
    """``dw_reduce`` alone on random FP32 rows writes every element of a NaN-filled ``dw``, meets the naive tier (the
    rows summed by torch in BF16) against the FP64 sum of the rows, and gives the same bytes on a second launch."""
    cache = {}

    def run(j, s):
        gen = torch.Generator(device=device).manual_seed(s)
        rows = torch.randn(G, HIDDEN, generator=gen, device=device) * (1.0 + 3.0 * torch.rand(G, 1, device=device))
        cache[j] = rows
        dw = torch.full((HIDDEN,), math.nan, device=device)
        again = torch.full_like(dw, math.nan)
        cake_backend.launch_dw_reduce("dw_reduce", dw_rows=rows, dw=dw)
        cake_backend.launch_dw_reduce("dw_reduce", dw_rows=rows, dw=again)
        torch.cuda.synchronize(device)
        assert torch.equal(dw.view(torch.int32), again.view(torch.int32))
        return {"dw": dw}, {"dw": rows.double().sum(0)}

    def naive(j, tier):
        return {"dw": cache.pop(j).bfloat16().sum(0)}

    _record_failures(*judge_case(run, naive, seed=G), record_property)


@pytest.mark.parametrize(("n", "T"), ((1, 129), (5, 1025), (9, 4097)))
def test_every_output_repeats_bit_for_bit(n, T, device) -> None:
    """The forward and the backward launched twice on the same inputs give the same bytes, ``dw`` included."""
    ns = make_inputs(T, n, seed=97 + n, device=device, data="peaked")
    runs = []
    for _ in range(2):
        out, p, r, d = cake_backend.cake_attn_res_det_train_forward(ns.values, ns.w)
        dv, dw = cake_backend.cake_attn_res_det_train_backward(ns.dout, ns.values, ns.w, p, r, d)
        runs.append([out, p, r, d, *dv, dw])
    torch.cuda.synchronize(device)
    for k, (a, b) in enumerate(zip(*runs, strict=True)):
        assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), k
