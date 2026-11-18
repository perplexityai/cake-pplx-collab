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

Backend suite of ``cake_pplx._backends.cake_engram_gate_det_train``, the deterministic package of the Engram
context gate (four streams of width 5120).

Without a device: the registry's records, stages, architectures and launch facts; the record of each stage and dot
product summation, and the record suffix of each ``dot_sum`` choice; the persistent grid sizes and the resident
cluster table they come from; every stage launcher passes every kernel parameter of its records by name (the FFI
entry replaced by a recorder); ``build.json`` covers every record on every architecture and its source
digests; the loader imports standalone; a prebuilt library is preferred; the package reads no environment variable
but the JIT shim's.

On SM100 / SM103 / SM107: every record loads; the forward of each summation record writes every element of a
NaN-filled ``out`` and meets the naive BF16 tier against an FP64 oracle (``tests/cake_naive.py``, 32 draws); the
backward of each ``dot_sum`` choice meets the tier for ``dx``, ``dkey``, ``dvalue``, ``dq_weight`` and
``dk_weight``;
``dw_reduce`` alone meets the tier on random partial rows.
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

from tests.cake.test_helpers.cake_engram_gate_naive import HIDDEN, STREAMS, judge_draws, make_inputs
from tests.cake.test_helpers.cake_package import cake_package

PACKAGE = "cake.cake_engram_gate_det_train"
pkg = cake_package(PACKAGE)
cake_backend = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
RECORDS = ("bwd_dot2", "bwd_f32", "dw_reduce", "fwd_dot2", "fwd_f32")
ACC_MODES = ("f32", "dot2")
# The public ``dot_sum`` choices and the record suffix each launches; the first is the default.
DOT_SUMS = {"compensated": "dot2", "fp32": "f32"}
ENTRIES = (
    "cake_engram_gate_det",
    "cake_engram_gate_det_kv",
    "cake_engram_gate_det_train_backward",
    "cake_engram_gate_det_train_forward",
    "cake_engram_gate_det_train_workspace_bytes",
    "is_cake_engram_gate_det_train_supported",
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
    assert cake_jit.STAGES == ("fwd", "bwd", "dw_reduce")
    assert tuple(cake_jit.ACC_MODES) == ACC_MODES
    assert sorted(cake_jit.MODULES) == sorted(RECORDS)
    for name, rec in cake_jit.MODULES.items():
        assert rec["arches"] == list(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert rec["stage"] in cake_jit.STAGES, name
        assert name == rec["stage"] or name.startswith(rec["stage"] + "_"), name
        assert rec["library"].startswith(cake_jit.LIBRARY_PREFIX + "_"), name
        assert rec["kernel_symbol"] == f"kernel_{rec['library']}", name
        launch = rec["launch"]
        assert set(launch) == {"block", "cluster", "cooperative", "dynamic_smem_bytes", "use_pdl"}, name
        assert launch["cooperative"] is False, name
        if rec["stage"] != "dw_reduce":
            assert tuple(launch["cluster"]) == (cake_jit.CLUSTER_CTAS, 1, 1), name
        kinds = [kind for kind, _ in rec["arg_plan"]]
        assert kinds[-3:] == ["grid", "grid", "grid"], name
        assert cake_backend.parameters(name), name


def test_every_summation_has_its_records() -> None:
    """Both summations of the dot product ship a forward and a backward record, and each ``dot_sum`` choice names
    one of them (``"compensated"``: ``dot2``, ``"fp32"``: ``f32``); another choice is refused."""
    assert cake_backend._DOT_SUMS == DOT_SUMS
    assert sorted(DOT_SUMS.values()) == sorted(ACC_MODES)
    for choice, acc in DOT_SUMS.items():
        assert cake_backend._dot_acc(choice) == acc
    for bad in ("dot2", "f32", "FP32", "", "compensated "):
        with pytest.raises(ValueError, match=r"dot_sum must be one of \['compensated', 'fp32'\]"):
            cake_backend._dot_acc(bad)
    for bad in (None, 1, b"fp32", ("fp32",)):
        with pytest.raises(TypeError, match=r"dot_sum must be a str, one of \['compensated', 'fp32'\]"):
            cake_backend._dot_acc(bad)
    for acc in ACC_MODES:
        for stage in ("fwd", "bwd"):
            assert cake_jit.record(stage, acc) == f"{stage}_{acc}"
            assert cake_jit.MODULES[f"{stage}_{acc}"]["stage"] == stage
        assert cake_jit.record("dw_reduce", acc) == "dw_reduce"
    with pytest.raises(ValueError):
        cake_jit.record("fwd", "f64")
    with pytest.raises(ValueError):
        cake_jit.record("main", "f32")


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name in cake_jit.MODULES:
        for arch in ARCHES:
            assert cake_backend.library_name(name, arch) == f"{cake_jit.MODULES[name]['library']}_{arch}"
        with pytest.raises(RuntimeError):
            cake_backend.library_name(name, "sm_90a")
    with pytest.raises(KeyError):
        cake_backend.record("bwd_main")


# The measured resident four-CTA clusters of the shipped programs: (architecture, SM count, record) -> clusters.
MEASURED = {
    ("sm_100a", 148, "fwd_dot2"): 142,
    ("sm_100a", 148, "fwd_f32"): 142,
    ("sm_103a", 152, "fwd_dot2"): 148,
    ("sm_103a", 152, "fwd_f32"): 148,
    ("sm_103a", 152, "bwd_dot2"): 110,
    ("sm_103a", 152, "bwd_f32"): 110,
    ("sm_107a", 212, "fwd_dot2"): 184,
    ("sm_107a", 212, "fwd_f32"): 184,
    ("sm_107a", 212, "bwd_dot2"): 138,
    ("sm_107a", 212, "bwd_f32"): 138,
}
PERSISTENT_RECORDS = ("bwd_dot2", "bwd_f32", "fwd_dot2", "fwd_f32")


def test_the_resident_cluster_table_holds_the_measured_devices() -> None:
    """Every ``fwd`` / ``bwd`` record has its CTAs per SM; the measured devices give their measured clusters; every
    table entry is at most CTAs per SM * SMs / 4 clusters, and the fallback stays below every table entry (a grid
    never exceeds what a measured device holds)."""
    assert sorted(cake_jit.CTAS_PER_SM) == sorted(PERSISTENT_RECORDS)
    assert sorted(name for name, rec in cake_jit.MODULES.items() if rec["stage"] != "dw_reduce") == sorted(
        PERSISTENT_RECORDS
    )
    for (arch, sms, name), want in MEASURED.items():
        assert cake_jit.resident_clusters(name, arch, sms) == want, (arch, sms, name)
    fraction = cake_jit.UNMEASURED_CLUSTER_FRACTION
    for (arch, sms), table in cake_jit.RESIDENT_CLUSTERS.items():
        assert arch in ARCHES
        for per_sm, held in table.items():
            assert 1 <= held <= per_sm * sms // cake_jit.CLUSTER_CTAS, (arch, sms, per_sm)
            assert int(fraction * per_sm * sms) // cake_jit.CLUSTER_CTAS <= held, (arch, sms, per_sm)
    # A device or a CTA count the table does not hold: the fallback fraction.
    assert cake_jit.resident_clusters("fwd_dot2", "sm_100a", 152) == int(fraction * 4 * 152) // 4
    assert cake_jit.resident_clusters("bwd_dot2", "sm_103a", 160) == int(fraction * 3 * 160) // 4
    assert cake_jit.resident_clusters("fwd_f32", "sm_107a", 1) == 1


@pytest.mark.parametrize("sms", (1, 148, 152, 160, 212))
@pytest.mark.parametrize("T", (1, 2, 3, 129, 4096, 16384, 65536, 2**31 - 1))
def test_grids_are_persistent_clusters(T, sms) -> None:
    """``fwd`` and ``bwd`` run ``min(T, resident clusters)`` clusters (at least one) of four CTAs; ``dw_reduce`` covers
    the 20480 weight-gradient columns on any device."""
    for arch in ARCHES:
        for name in PERSISTENT_RECORDS:
            resident = cake_jit.resident_clusters(name, arch, sms)
            clusters = cake_jit.clusters(name, T, arch, sms)
            assert 1 <= clusters <= T
            assert clusters == max(1, min(T, resident))
            assert resident <= max(1, cake_jit.CTAS_PER_SM[name] * sms // cake_jit.CLUSTER_CTAS)
            assert cake_jit.grid(name, T, arch, sms) == (cake_jit.CLUSTER_CTAS * clusters, 1, 1)
        assert cake_jit.grid("dw_reduce", T, arch, sms) == cake_jit.dw_reduce_grid()
    assert cake_jit.dw_reduce_grid() == (STREAMS * HIDDEN // cake_jit.DW_REDUCE_COLUMNS_PER_CTA, 1, 1)
    assert STREAMS * HIDDEN % cake_jit.DW_REDUCE_COLUMNS_PER_CTA == 0


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def entry(self, name: str, arch: str):
        return lambda *args: self.calls.append((name, args))


def _cpu_operands(T: int) -> dict:
    bf16 = dict(dtype=torch.bfloat16)
    return {
        "x": torch.zeros(T, STREAMS, HIDDEN, **bf16),
        "key": torch.zeros(T, STREAMS, HIDDEN, **bf16),
        "value": torch.zeros(T, HIDDEN, **bf16),
        "w": torch.zeros(STREAMS, HIDDEN),
        "dout": torch.zeros(T, STREAMS, HIDDEN, **bf16),
    }


@pytest.mark.parametrize("masked", (False, True), ids=("no-mask", "mask"))
@pytest.mark.parametrize("acc", ACC_MODES)
def test_stage_launchers_pass_every_kernel_parameter(acc, masked, monkeypatch) -> None:
    """Each stage launcher passes every parameter of its records' argument plans, in order, then the grid
    (``_entry`` replaced by a recorder; nothing is launched)."""
    recorder = _Recorder()
    monkeypatch.setattr(cake_backend, "_entry", recorder.entry)
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: "sm_100a")
    T, clusters = 5, 2
    ops = _cpu_operands(T)
    mask = torch.ones(T, dtype=torch.bool) if masked else None
    out = torch.empty_like(ops["x"])
    cake_backend.launch_fwd(
        cake_jit.record("fwd", acc),
        x=ops["x"],
        key=ops["key"],
        value=ops["value"],
        w=ops["w"],
        mask=mask,
        out=out,
        eps=1e-20,
        clamp_value=1e-6,
        clusters=clusters,
    )
    rows = clusters  # one partial row per cluster
    dw = torch.zeros(rows, STREAMS, HIDDEN)
    cake_backend.launch_bwd(
        cake_jit.record("bwd", acc),
        dout=ops["dout"],
        x=ops["x"],
        key=ops["key"],
        value=ops["value"],
        w=ops["w"],
        mask=mask,
        dx=torch.empty_like(ops["x"]),
        dkey=torch.empty_like(ops["x"]),
        dvalue=torch.empty_like(ops["value"]),
        dw=dw,
        eps=1e-20,
        clamp_value=1e-6,
        clusters=clusters,
    )
    vec = torch.zeros(STREAMS, HIDDEN)
    cake_backend.launch_dw_reduce("dw_reduce", dw=dw, qw=vec, kw=vec, dq=vec.clone(), dk=vec.clone())
    assert [name for name, _ in recorder.calls] == [f"fwd_{acc}", f"bwd_{acc}", "dw_reduce"]
    for name, args in recorder.calls:
        params = cake_backend.parameters(name)
        assert len(args) == len(params) + 3, (name, params)
        grid = args[-3:]
        if name == "dw_reduce":
            assert grid == cake_jit.dw_reduce_grid()
        else:
            assert grid == (cake_jit.CLUSTER_CTAS * clusters, 1, 1)
        named = dict(zip(params, args, strict=False))
        if "T" in named:
            assert named["T"] == T
        if "mask" in named and masked:
            assert named["mask"] is mask
        if "has_mask" in named:
            assert named["has_mask"] == int(masked)


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


# (T, weights, mask, layout): one token, two, a token count past one wave of clusters, an odd count.
CASES = (
    (1, "fp32", "none", "contiguous"),
    (2, "bf16", "some", "kv"),
    (129, "mixed", "some", "kv"),
    (1031, "fp32", "none", "contiguous"),
)
CASE_IDS = [f"T{T}-{w}-{m}-{lay}" for T, w, m, lay in CASES]


def _record_failures(judge, failures, record_property) -> None:
    judge.record(record_property)
    assert not failures, failures


@pytest.mark.parametrize("acc", ACC_MODES)
@pytest.mark.parametrize(("T", "weights", "mask", "layout"), CASES, ids=CASE_IDS)
def test_the_forward_record_matches_the_reference(acc, T, weights, mask, layout, device, record_property) -> None:
    """The forward record of ``acc`` launched directly writes every element of a NaN-filled ``out`` and meets the
    tier."""
    name, arch = cake_jit.record("fwd", acc), cake_backend.device_arch(device.index)

    def run(ns):
        out = torch.full_like(ns.x, math.nan)
        w = (ns.q_weight.float() * ns.k_weight.float()).contiguous()
        cake_backend.launch_fwd(
            name,
            x=ns.x,
            key=ns.key,
            value=ns.value,
            w=w,
            mask=ns.token_mask,
            out=out,
            eps=1e-20,
            clamp_value=1e-6,
            clusters=cake_jit.clusters(name, T, arch, cake_backend.sm_count(device.index)),
        )
        torch.cuda.synchronize(device)
        return {"out": out}

    def make(seed):
        return make_inputs(T, seed=seed, device=device, weights=weights, mask=mask, layout=layout)

    _record_failures(*judge_draws(run, make, seed=11 + T), record_property)


@pytest.mark.parametrize("dot_sum", tuple(DOT_SUMS))
@pytest.mark.parametrize(("T", "weights", "mask", "layout"), CASES, ids=CASE_IDS)
def test_the_backward_records_match_the_reference(dot_sum, T, weights, mask, layout, device, record_property):
    """The backward of ``dot_sum`` (``bwd_<acc>`` then ``dw_reduce``), with NaN-filled ``dx``, ``dkey`` and
    ``dvalue``, meets the tier for every gradient."""

    def run(ns):
        dx = torch.full_like(ns.x, math.nan)
        dkey = torch.full_like(ns.x, math.nan)
        dvalue = torch.full_like(ns.value.contiguous(), math.nan)
        got = cake_backend.cake_engram_gate_det_train_backward(
            ns.dout,
            ns.x,
            ns.key,
            ns.value,
            ns.q_weight,
            ns.k_weight,
            ns.token_mask,
            dot_sum=dot_sum,
            dx=dx,
            dkey=dkey,
            dvalue=dvalue,
        )
        torch.cuda.synchronize(device)
        return dict(zip(("dx", "dkey", "dvalue", "dq", "dk"), got, strict=True))

    def make(seed):
        return make_inputs(T, seed=seed, device=device, weights=weights, mask=mask, layout=layout)

    _record_failures(*judge_draws(run, make, seed=23 + T), record_property)


@pytest.mark.parametrize("parts", (1, 7, 296))
def test_dw_reduce_matches_the_reference(parts, device, record_property) -> None:
    """``dw_reduce`` alone: ``dq = (sum_p dw[p]) * kw32`` and ``dk = (sum_p dw[p]) * qw32`` (FP32) from random partial
    rows meet the tier (BF16 tier: the rows, weights and sum in BF16)."""
    from tests.cake_naive import judge_case

    cache = {}

    def run(j, s):
        gen = torch.Generator(device=device).manual_seed(s)
        dw = torch.randn(parts, STREAMS, HIDDEN, device=device, generator=gen)
        qw = 1.0 + 0.1 * torch.randn(STREAMS, HIDDEN, device=device, generator=gen)
        kw = 1.0 + 0.1 * torch.randn(STREAMS, HIDDEN, device=device, generator=gen)
        dq = torch.full((STREAMS, HIDDEN), math.nan, device=device)
        dk = torch.full_like(dq, math.nan)
        cake_backend.launch_dw_reduce("dw_reduce", dw=dw, qw=qw, kw=kw, dq=dq, dk=dk)
        torch.cuda.synchronize(device)
        cache[j] = (dw, qw, kw)
        total = dw.double().sum(0)
        return {"dq": dq, "dk": dk}, {"dq": total * kw.double(), "dk": total * qw.double()}

    def naive(j, tier):
        dw, qw, kw = cache.pop(j)
        total = dw.bfloat16().sum(0)
        return {"dq": total * kw.bfloat16(), "dk": total * qw.bfloat16()}

    judge, failures = judge_case(run, naive, seed=31 + parts)
    _record_failures(judge, failures, record_property)
