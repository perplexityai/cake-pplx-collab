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

Backend suite of the Cake deterministic (``det``) ratio-2 gated KV compression package
(``cake_pplx._backends.cake_gated_kv_compression_det_train``, the backend of
``cake_pplx.cake_gated_kv_compression_det_train``): width 512, packed documents bounded by a device ``cu_seqlens``, on
SM100, SM103 and SM107.

Without a device: the registry (the pair-table, forward and backward roles, each served on each of the three
architectures by exactly one record and delivered for no other: the forward by ``fwd`` on sm_100a / sm_103a and
``fwd_sm_107a`` on sm_107a; the launch facts, the kernel parameters each record takes), the record each role launches on
each architecture, the launch grids (every row covered, the pair table on one CTA), the launchers' keywords and the
kernel arguments they pass, the exact overlap rule of two row-strided views (the two column halves of one buffer share
no element), the memory report, the build manifest (every record's library, the recorded source digests), the standalone
loader, the prebuilt libraries served first, and no environment read outside the JIT shim.

On SM100 / SM103 / SM107: the pair table equal to the exclusive prefix of the pair counts (an integer table: exactly);
each kernel judged on its own against the naive BF16 tier (``tests/cake_naive.py``: the error against an FP64 oracle
of the same inputs over 32 independent draws, in the L2 and the max norm, median and largest, beside the error of the
same math in BF16), every output buffer filled with NaN before the launch so every element must be written: the
forward (``out``) and the backward (``dC``, ``dZ``, the tail rows exactly +0.0) on contiguous ``C`` / ``Z`` and on the
column halves of one ``[T, 1024]`` tensor (the gradients written into the halves of one ``[T, 1024]`` buffer), on
small packs with empty and one-token documents, close pairs, gate extremes, many small odd documents and the two
performance packs; the forward rows past the pair count stored as +0.0.  Then the documented invariances, bit for
bit: repeated calls, the two layouts, each document alone against the pack, the tail tokens' values, and the records
each call launches.

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import random
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_gated_kv_compression_naive import (
    PERF_PACKS,
    WIDTH,
    Pack,
    assert_backward_contract,
    assert_bitwise,
    assert_forward_contract,
    assert_zero_rows,
    bounds,
    judge_draws,
    judged,
    make_inputs,
    nan_halves,
    nan_like_rows,
    pair_table_of,
    tail_rows,
)
from tests.cake.test_helpers.cake_package import cake_package

PACKAGE_KEY = "cake.cake_gated_kv_compression_det_train"
pkg = cake_package(PACKAGE_KEY)
cake_backend = cake_package(PACKAGE_KEY, "cake_backend")
cake_jit = cake_package(PACKAGE_KEY, "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ROLES = ("pair_table", "fwd", "bwd")
# Record -> (its role, the architectures it serves, the lanes of a 512-wide row): the forward ships 128-bit lanes
# (4 FP32 channels) on sm_100a / sm_103a and 256-bit lanes (8 channels) on sm_107a.
RECORDS = {
    "pair_table": ("pair_table", ARCHES, None),
    "fwd": ("fwd", ("sm_100a", "sm_103a"), 128),
    "fwd_sm_107a": ("fwd", ("sm_107a",), 64),
    "bwd": ("bwd", ARCHES, 64),
}
# The kernel parameters of each role (the argument plan without the grid), in any order: the host passes them by name.
PARAMETERS = {
    "pair_table": {"cu_seqlens", "cu_pairs", "num_docs"},
    "fwd": {"c", "z", "cu_seqlens", "cu_pairs", "out", "num_docs", "G", "c_ld", "z_ld"},
    "bwd": {
        "dout",
        "c",
        "z",
        "cu_seqlens",
        "cu_pairs",
        "dc",
        "dz",
        "num_docs",
        "G",
        "c_ld",
        "z_ld",
        "dc_ld",
        "dz_ld",
    },
}
LAUNCHER_KEYWORDS = {
    "launch_pair_table": ("cu_seqlens", "cu_pairs"),
    "launch_fwd": ("c", "z", "cu_seqlens", "cu_pairs", "out", "c_ld", "z_ld"),
    "launch_bwd": ("dout", "c", "z", "cu_seqlens", "cu_pairs", "dc", "dz", "c_ld", "z_ld", "dc_ld", "dz_ld"),
}
LAYOUTS = ("contiguous", "halves")

# Packs with at least one pair: one document, empty and one-token documents, odd tails, the reference's mixed pack, many
# small odd documents (lengths 1, 3, 5, 7 repeated), and the two performance packs.
PAIR_PACKS = (
    Pack((2,)),
    Pack((0, 1, 2, 3)),
    Pack((0, 1, 3, 2, 7, 4)),
    Pack((127, 256, 1, 641)),
    Pack((1, 3, 5, 7) * 75, "small-odd-S300"),
    *PERF_PACKS,
)
# Input variants judged on the reference's mixed pack: close pairs and gate extremes.
VARIANTS = (
    pytest.param({"close": 12}, id="close-2^-12"),
    pytest.param({"close": 16}, id="close-2^-16"),
    pytest.param({"gates": True}, id="gate-extremes"),
)
MIXED = Pack((127, 256, 1, 641))
# Pair-table packs: one document, empty documents only, odd and even lengths, and more documents than one tile.
TABLE_PACKS = (
    Pack((0,)),
    Pack((2,)),
    Pack((0, 0, 0)),
    Pack((0, 1, 3, 2, 7, 4)),
    Pack(tuple((i * 7919) % 13 for i in range(5000)), "S5000"),
    *PERF_PACKS,
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
    for spec in cake_jit.specs_for_arch(cake_jit.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return json.loads(path.read_text())


def _lds(x) -> dict:
    return {"c_ld": x.c.stride(0), "z_ld": x.z.stride(0)}


def _grad_buffers(x, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    """NaN ``dc`` and ``dz``: two tensors, or the two halves of one ``[T, 1024]`` buffer."""
    if layout == "halves":
        _, dc, dz = nan_halves(x.T, x.c.device)
        return dc, dz
    return nan_like_rows(x.T, torch.float32, x.c.device), nan_like_rows(x.T, torch.float32, x.c.device)


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.ROLES) == ROLES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert cake_jit.WIDTH == WIDTH and cake_jit.CZ_WIDTH == 2 * WIDTH
    assert cake_jit.LIBRARY_PREFIX == "cake_gated_kv_compression_det_train"
    assert list(cake_jit.MODULES) == list(RECORDS)
    rendered = (PACKAGE_DIR / "build.json").is_file()
    libraries = set()
    for name, rec in cake_jit.MODULES.items():
        role, arches, lanes = RECORDS[name]
        assert rec["role"] == role, name
        assert tuple(rec["arches"]) == arches, (name, rec["arches"])
        if lanes is not None:
            assert rec["grid_rule"]["lanes_per_row"] == lanes, name
        assert rec["ffi_entry"] == "run", name
        assert rec["grid_rule"]["kind"] in cake_jit.GRID_KINDS, name
        launch = rec["launch"]
        assert launch["cooperative"] is False, name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert int(launch["block"][1]) == int(launch["block"][2]) == 1, name
        if rendered:
            assert re.fullmatch(r"cake_gated_kv_compression_det_train_[0-9a-f]{20}", rec["library"]), rec["library"]
            assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        libraries.add(rec["library"])
    assert cake_jit.MODULES["pair_table"]["grid_rule"]["kind"] == "one_cta"
    for name in ("fwd", "fwd_sm_107a", "bwd"):
        assert cake_jit.MODULES[name]["grid_rule"]["kind"] == "rows", name
    if rendered:
        assert len(libraries) == len(cake_jit.MODULES)


def test_each_role_has_one_record_per_architecture() -> None:
    """Every role is served on every architecture by exactly one record: the forward by ``fwd`` (128-bit lanes) on
    sm_100a and sm_103a and by ``fwd_sm_107a`` (256-bit lanes) on sm_107a; no record serves another architecture."""
    for role in ROLES:
        for arch in ARCHES:
            serving = [name for name, rec in cake_jit.MODULES.items() if rec["role"] == role and arch in rec["arches"]]
            assert serving == [cake_jit.role_record(role, arch)], (role, arch, serving)
    assert [cake_jit.role_record("fwd", arch) for arch in ARCHES] == ["fwd", "fwd", "fwd_sm_107a"]
    assert {cake_jit.role_record(role, arch) for role in ("pair_table", "bwd") for arch in ARCHES} == {
        "pair_table",
        "bwd",
    }
    for arch in ("sm_110a", "sm_90a", "sm_120a"):
        with pytest.raises(KeyError):
            cake_jit.role_record("fwd", arch)
    with pytest.raises(KeyError):
        cake_jit.role_record("pool", "sm_100a")
    # 256 threads a CTA: two forward rows on sm_100a / sm_103a, four on sm_107a
    assert cake_jit.rows_per_cta("fwd") * 2 == cake_jit.rows_per_cta("fwd_sm_107a")


@pytest.mark.parametrize("name", list(RECORDS))
def test_each_record_takes_its_kernel_parameters(name) -> None:
    """The argument plan of each record names the kernel parameters of its role (the host passes them by name), then
    the grid."""
    plan = cake_jit.MODULES[name]["arg_plan"]
    if not plan:
        pytest.skip("the record's argument plan is not rendered yet")
    found = set(cake_jit.parameters(name)) - {"enable_pdl"}
    logical = cake_jit.MODULES[name].get("logical", {})
    assert {logical.get(key, key) for key in found} == PARAMETERS[RECORDS[name][0]]


def test_no_record_is_delivered_outside_its_architectures() -> None:
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            assert cake_jit.library_name(name, arch) == f"{rec['library']}_{arch}"
        for arch in ("sm_110a", "sm_90a", "sm_120a", *(arch for arch in ARCHES if arch not in rec["arches"])):
            with pytest.raises(RuntimeError):
                cake_jit.library_name(name, arch)


def test_grids_cover_every_row() -> None:
    """The pair table runs on one CTA; the forward covers ``G`` rows and the backward ``G + S`` virtual rows with the
    fewest CTAs of :func:`rows_per_cta` rows each."""
    assert cake_jit.launch_grid("pair_table", 1) == (1, 1, 1)
    assert cake_jit.launch_grid("pair_table", 10**6) == (1, 1, 1)
    rng = random.Random(7091)
    for name in ("fwd", "fwd_sm_107a", "bwd"):
        per = cake_jit.rows_per_cta(name)
        assert per >= 1
        for rows in [1, 2, per - 1, per, per + 1, *[rng.randint(1, 1 << 22) for _ in range(200)]]:
            if rows < 1:
                continue
            grid = cake_jit.launch_grid(name, rows)
            assert grid[1:] == (1, 1), (name, rows, grid)
            assert grid[0] * per >= rows > (grid[0] - 1) * per, (name, rows, grid, per)


def test_launchers_take_named_buffers() -> None:
    """Each launcher takes its buffers and strides by keyword only."""
    for launcher, keywords in LAUNCHER_KEYWORDS.items():
        params = list(inspect.signature(getattr(cake_backend, launcher)).parameters.values())
        assert all(p.kind is p.KEYWORD_ONLY for p in params), launcher
        assert sorted(p.name for p in params) == sorted(keywords), launcher


def test_launchers_pass_every_kernel_argument(monkeypatch) -> None:
    """Each launcher hands ``run_module`` exactly the kernel parameters of its record, with ``num_docs = S``, the
    rows of ``out`` / ``dout`` as ``G``, the strides it was given and the rows of its grid (no device: the launch is a
    recorder)."""
    calls = []
    monkeypatch.setattr(
        cake_jit, "run_module", lambda name, values, *, rows, device_index: calls.append((name, dict(values), rows))
    )
    S, G, T = 5, 9, 23
    cu = torch.zeros(S + 1, dtype=torch.int32)
    table = torch.zeros(S + 1, dtype=torch.int32)
    cz = torch.zeros(T, 2 * WIDTH)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    out = torch.zeros(G, WIDTH, dtype=torch.bfloat16)
    dc, dz = torch.zeros(T, WIDTH), torch.zeros(T, WIDTH)
    cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table)
    cake_backend.launch_fwd(c=c, z=z, cu_seqlens=cu, cu_pairs=table, out=out, c_ld=1024, z_ld=1024)
    cake_backend.launch_bwd(
        dout=out, c=c, z=z, cu_seqlens=cu, cu_pairs=table, dc=dc, dz=dz, c_ld=1024, z_ld=1024, dc_ld=512, dz_ld=512
    )
    assert [name for name, _, _ in calls] == list(ROLES)
    for name, values, rows in calls:
        assert set(values) == PARAMETERS[name], name
        assert values["num_docs"] == S and values["cu_seqlens"] is cu and values["cu_pairs"] is table, name
    (_, _, rows_table), (_, fwd, rows_fwd), (_, bwd, rows_bwd) = calls
    assert rows_table == 1 and rows_fwd == G and rows_bwd == G + S
    assert fwd["G"] == bwd["G"] == G
    assert (fwd["c_ld"], fwd["z_ld"], bwd["dc_ld"], bwd["dz_ld"]) == (1024, 1024, 512, 512)
    assert fwd["out"] is out and bwd["dout"] is out and bwd["dc"] is dc and bwd["dz"] is dz


def test_each_role_launches_the_record_of_the_device_architecture(monkeypatch) -> None:
    """``run_module`` takes a role and launches the record that serves it on the device's architecture, on that
    record's grid: the forward of ``G`` rows runs ``fwd`` (two rows a CTA) on sm_100a / sm_103a and ``fwd_sm_107a``
    (four rows a CTA) on sm_107a; the pair table and the backward run their one record everywhere (no device: the
    architecture is given and the binding entry is a recorder)."""
    if not all(rec["arg_plan"] for rec in cake_jit.MODULES.values()):
        pytest.skip("the records' argument plans are not rendered yet")
    launched = []

    def entry(name, arch):
        return lambda *args: launched.append((name, arch, len(args), tuple(args[-3:])))

    monkeypatch.setattr(cake_jit, "_entry", entry)
    tvm_ffi = types.SimpleNamespace(use_raw_stream=lambda device, stream: contextlib.nullcontext())
    monkeypatch.setattr(cake_jit, "_tvm_ffi", lambda: tvm_ffi)
    monkeypatch.setattr(cake_jit, "_tvm_device", lambda index: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda index=None: types.SimpleNamespace(cuda_stream=0))
    S, G, T = 5, 9, 23
    cu = torch.zeros(S + 1, dtype=torch.int32)
    table = torch.zeros(S + 1, dtype=torch.int32)
    cz = torch.zeros(T, 2 * WIDTH)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    out = torch.zeros(G, WIDTH, dtype=torch.bfloat16)
    dc, dz = torch.zeros(T, WIDTH), torch.zeros(T, WIDTH)
    for arch in ARCHES:
        monkeypatch.setattr(cake_jit, "device_arch", lambda index, arch=arch: arch)
        cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table)
        cake_backend.launch_fwd(c=c, z=z, cu_seqlens=cu, cu_pairs=table, out=out, c_ld=1024, z_ld=1024)
        cake_backend.launch_bwd(
            dout=out, c=c, z=z, cu_seqlens=cu, cu_pairs=table, dc=dc, dz=dz, c_ld=1024, z_ld=1024, dc_ld=512, dz_ld=512
        )
    expected = []
    for arch in ARCHES:
        forward = "fwd_sm_107a" if arch == "sm_107a" else "fwd"
        for name, rows in (("pair_table", 1), (forward, G), ("bwd", G + S)):
            grid = cake_jit.launch_grid(name, rows)
            expected.append((name, arch, len(cake_jit.MODULES[name]["arg_plan"]), grid))
    assert launched == expected
    # nine forward rows: five CTAs of two rows on sm_100a / sm_103a, three CTAs of four rows on sm_107a
    assert [grid for name, _, _, grid in launched if name.startswith("fwd")] == [(5, 1, 1), (5, 1, 1), (3, 1, 1)]


def test_the_overlap_rule_is_exact_for_column_halves() -> None:
    """The two column halves of one ``[T, 1024]`` buffer share no element; a view overlaps itself, a shifted window,
    a row of the other half's buffer region, and a dense alias; separate allocations never overlap."""
    overlaps = cake_backend._overlaps
    buffer = torch.zeros(7, 2 * WIDTH)
    left, right = buffer[:, :WIDTH], buffer[:, WIDTH:]
    assert not overlaps(left, right) and not overlaps(right, left)
    assert overlaps(left, left) and overlaps(right, right)
    assert overlaps(left, buffer[:, 256:768]) and overlaps(buffer[:, 256:768], right)
    assert overlaps(left, buffer[3:4, :WIDTH]) and not overlaps(left[:3], buffer[3:, :WIDTH])
    assert overlaps(left, buffer.view(-1)[5 * 2 * WIDTH : 5 * 2 * WIDTH + 8])
    assert not overlaps(left, buffer.view(-1)[WIDTH : WIDTH + 8])  # inside row 0 of the right half
    assert not overlaps(left, torch.zeros(7, WIDTH)) and not overlaps(left, torch.zeros(0, WIDTH))


def test_the_memory_report_separates_workspace_and_retained_state() -> None:
    for T, S in ((0, 0), (16384, 27), (65536, 68), (7, 1)):
        report = pkg.cake_gated_kv_compression_det_train_memory_bytes(T, S)
        table = 4 * (S + 1)
        assert report == {
            "forward_workspace": table,
            "backward_workspace": table,
            "retained_for_backward": table,
            "retained_inputs": 4096 * T + table,
        }
    with pytest.raises(ValueError):
        pkg.cake_gated_kv_compression_det_train_memory_bytes(-1, 2)
    with pytest.raises(TypeError):
        pkg.cake_gated_kv_compression_det_train_memory_bytes(torch.tensor(3), 2)


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]}
    assert set(entries) == wanted
    assert len(entries) == sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert sorted(entry["arch"] for entry in entries.values()) == sorted(
        arch for _, arches, _ in RECORDS.values() for arch in arches
    )
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert name.endswith("_" + entry["arch"]), name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


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
    records = sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path."""
    _manifest_or_skip()
    cake_build = cake_package(PACKAGE_KEY, "cake_build")
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_jit.library_name(name, arch)
            assert cake_build.prebuilt_path(library).resolve() == PACKAGE_DIR / "aot" / f"{library}.so"


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
    ``os.environ[...]``, ``... in os.environ`` or ``os.getenv(...)`` -- a string literal, or a loop variable over a
    tuple / list of string literals -- and the lines of every other ``environ`` / ``getenv`` use."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def keys(expr: ast.AST, at: ast.AST) -> set[str] | None:
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
    """No module of the package reads an environment variable, except the JIT shim (``_cake_pplx_jit.py``), which
    reads exactly :data:`JIT_SHIM_ENVIRONMENT_VARIABLES`, each by a literal name."""
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


def test_the_host_never_reads_device_values() -> None:
    """No module of the package (the JIT shim aside) calls a tensor-to-host conversion: ``.item()``, ``.tolist()``,
    ``.cpu()``, ``.numpy()``, ``.to("cpu")`` or ``.synchronize()``."""
    banned = {"item", "tolist", "cpu", "numpy", "synchronize"}
    offenders = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name in ("_cake_pplx_jit.py", "cake_build.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in banned:
                    offenders.append(f"{path.name}:{node.lineno} .{node.func.attr}()")
                if node.func.attr == "to" and any(isinstance(a, ast.Constant) and a.value == "cpu" for a in node.args):
                    offenders.append(f"{path.name}:{node.lineno} .to('cpu')")
    assert not offenders, offenders


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: every record loads; the pair table


def test_every_record_loads(device) -> None:
    """Every record serving the device's architecture loads: one per role."""
    arch = cake_jit.device_arch(device.index)
    assert arch in ARCHES
    names = [cake_jit.role_record(role, arch) for role in ROLES]
    for name in names:
        assert callable(getattr(cake_jit.load(name, arch), cake_jit.MODULES[name]["ffi_entry"])), name


@pytest.mark.parametrize("pack", TABLE_PACKS, ids=[p.id for p in TABLE_PACKS])
def test_the_pair_table_is_the_prefix_of_the_pair_counts(pack, device) -> None:
    """``cu_pairs`` (filled with -7 first) equals the exclusive prefix of ``floor(length / 2)`` exactly, also with
    empty documents and more documents than one tile."""
    cu = torch.tensor(bounds(pack.lengths), dtype=torch.int32, device=device)
    table = torch.full((pack.S + 1,), -7, dtype=torch.int32, device=device)
    cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table)
    torch.cuda.synchronize(device)
    assert torch.equal(table, pair_table_of(pack.lengths, device))


# ----------------------------------------------------------------------------------------------------------------
# Each kernel on its own, judged against the naive BF16 tier


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("pack", PAIR_PACKS, ids=[p.id for p in PAIR_PACKS])
def test_the_forward_kernel_matches_the_reference(pack, layout, device, record_property) -> None:
    """``out`` (NaN first) of the forward kernel alone, on the pair table of the pack, meets the tier."""

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout=layout)
        out = nan_like_rows(x.G, torch.bfloat16, device)
        table = pair_table_of(pack.lengths, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=out, **_lds(x))
        assert_forward_contract("fwd", x, out)
        return judged(x, {"fwd": {"out": out}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_forward_kernel_matches_the_reference_on_extreme_inputs(variant, device, record_property) -> None:
    def draw(seed: int):
        x = make_inputs(MIXED, seed=seed, device=device, layout="halves", **variant)
        out = nan_like_rows(x.G, torch.bfloat16, device)
        table = pair_table_of(MIXED.lengths, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=out, **_lds(x))
        assert_forward_contract("fwd", x, out)
        return judged(x, {"fwd": {"out": out}})

    judge_draws(record_property, draw)


def test_forward_rows_past_the_pair_count_are_zero(device) -> None:
    """A caller ``G`` above the pair count: the rows past it are +0.0 (NaN first), the rows below are the bits of
    the call with the exact ``G``."""
    for pack in (MIXED, Pack((0, 1, 3, 2, 7, 4))):
        x = make_inputs(pack, seed=5, device=device)
        table = pair_table_of(pack.lengths, device)
        exact = nan_like_rows(pack.G, torch.bfloat16, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=exact, **_lds(x))
        wide = nan_like_rows(pack.G + 37, torch.bfloat16, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=wide, **_lds(x))
        assert_bitwise(wide[: pack.G], exact, f"{pack.id}: the rows below the pair count")
        assert_zero_rows(wide, range(pack.G, pack.G + 37), f"{pack.id}: the rows past the pair count")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("pack", PAIR_PACKS, ids=[p.id for p in PAIR_PACKS])
def test_the_backward_kernel_matches_the_reference(pack, layout, device, record_property) -> None:
    """``dC`` and ``dZ`` (NaN first; with ``halves``, the two halves of one NaN ``[T, 1024]`` buffer) of the backward
    kernel alone meet the tier, the tail rows exactly +0.0."""

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout=layout)
        dc, dz = _grad_buffers(x, layout)
        table = pair_table_of(pack.lengths, device)
        cake_backend.launch_bwd(
            dout=x.dout,
            c=x.c,
            z=x.z,
            cu_seqlens=x.cu,
            cu_pairs=table,
            dc=dc,
            dz=dz,
            dc_ld=dc.stride(0),
            dz_ld=dz.stride(0),
            **_lds(x),
        )
        assert_backward_contract("bwd", x, dc, dz)
        return judged(x, {"bwd": {"dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_backward_kernel_matches_the_reference_on_extreme_inputs(variant, device, record_property) -> None:
    def draw(seed: int):
        x = make_inputs(MIXED, seed=seed, device=device, layout="halves", **variant)
        dc, dz = _grad_buffers(x, "halves")
        table = pair_table_of(MIXED.lengths, device)
        cake_backend.launch_bwd(
            dout=x.dout,
            c=x.c,
            z=x.z,
            cu_seqlens=x.cu,
            cu_pairs=table,
            dc=dc,
            dz=dz,
            dc_ld=dc.stride(0),
            dz_ld=dz.stride(0),
            **_lds(x),
        )
        assert_backward_contract("bwd", x, dc, dz)
        return judged(x, {"bwd": {"dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


# ----------------------------------------------------------------------------------------------------------------
# The composed calls and their invariances, bit for bit


@pytest.mark.parametrize("pack", PAIR_PACKS, ids=[p.id for p in PAIR_PACKS])
def test_the_composed_calls_match_the_reference(pack, device, record_property) -> None:
    """The backend's forward and backward entries (pair table, then the kernel), on the column halves, meet the
    tier."""

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout="halves")
        out = pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
        dc, dz = pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
        assert_forward_contract("forward", x, out)
        assert_backward_contract("backward", x, dc, dz)
        return judged(x, {"composed": {"out": out, "dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


def _run(x, layout: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = nan_like_rows(x.G, torch.bfloat16, x.c.device)
    pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, out=out)
    dc, dz = _grad_buffers(x, layout)
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    return out, dc, dz


@pytest.mark.parametrize("pack", (MIXED, *PERF_PACKS), ids=[p.id for p in (MIXED, *PERF_PACKS)])
def test_repeated_calls_give_the_same_bits(pack, device) -> None:
    """Three calls on the same inputs (outputs NaN first each time) give the same bits: the deterministic contract."""
    for layout in LAYOUTS:
        x = make_inputs(pack, seed=17, device=device, layout=layout, gates=True)
        first = _run(x, layout)
        for _ in range(2):
            for name, got, want in zip(("out", "dc", "dz"), _run(x, layout), first, strict=True):
                assert_bitwise(got, want, f"{pack.id} {layout}: {name} run to run")


@pytest.mark.parametrize("pack", (MIXED, *PERF_PACKS), ids=[p.id for p in (MIXED, *PERF_PACKS)])
def test_column_halves_give_the_bits_of_contiguous_operands(pack, device) -> None:
    """``C`` / ``Z`` as the halves of one ``[T, 1024]`` tensor and the gradients written into the halves of one
    buffer give the bits of contiguous operands and gradients."""
    halves = make_inputs(pack, seed=23, device=device, layout="halves")
    dense = make_inputs(pack, seed=23, device=device, layout="contiguous")
    assert torch.equal(halves.c, dense.c) and torch.equal(halves.z, dense.z)
    for name, got, want in zip(("out", "dc", "dz"), _run(halves, "halves"), _run(dense, "contiguous"), strict=True):
        assert_bitwise(got, want, f"{pack.id}: {name} of the halves against contiguous")


@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED), ids=["L0-1-3-2-7-4", MIXED.id])
def test_each_document_alone_gives_the_bits_of_the_pack(pack, device) -> None:
    """Each document run alone (its own rows of ``C`` / ``Z``, ``cu_seqlens = [0, L]``, its rows of ``dout``) gives
    the bits of its rows in the pack's ``out``, ``dC`` and ``dZ``."""
    x = make_inputs(pack, seed=29, device=device)
    out, dc, dz = _run(x, "contiguous")
    b = bounds(pack.lengths)
    row = 0
    for s, n in enumerate(pack.lengths):
        pairs = n // 2
        doc = Pack((n,))
        y = make_inputs(doc, seed=0, device=device)
        y.c, y.z = x.c[b[s] : b[s + 1]].contiguous(), x.z[b[s] : b[s + 1]].contiguous()
        y.dout = x.dout[row : row + pairs].contiguous()
        got_out, got_dc, got_dz = _run(y, "contiguous")
        assert_bitwise(got_out, out[row : row + pairs], f"document {s} (length {n}): out")
        assert_bitwise(got_dc, dc[b[s] : b[s + 1]], f"document {s} (length {n}): dc")
        assert_bitwise(got_dz, dz[b[s] : b[s + 1]], f"document {s} (length {n}): dz")
        row += pairs


@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED, *PERF_PACKS), ids=lambda p: p.id)
def test_the_tail_tokens_change_no_output(pack, device) -> None:
    """New values of ``C`` and ``Z`` on every odd-length document's last token (``+1e6`` / ``-1e6``, then NaN) change
    no byte of ``out``, ``dC`` or ``dZ``; the tail rows of ``dC`` and ``dZ`` stay +0.0."""
    tails = tail_rows(pack.lengths)
    if not tails:
        pytest.skip("the pack has no odd-length document")
    x = make_inputs(pack, seed=31, device=device, layout="halves")
    want = _run(x, "halves")
    index = torch.tensor(tails, dtype=torch.long, device=device)
    for c_value, z_value in ((1e6, -1e6), (float("nan"), float("nan"))):
        x.cz[index, :WIDTH] = c_value
        x.cz[index, WIDTH:] = z_value
        for name, got, ref in zip(("out", "dc", "dz"), _run(x, "halves"), want, strict=True):
            assert_bitwise(got, ref, f"{pack.id}: {name} after the tail tokens changed to {c_value}")


def test_the_calls_launch_their_records_in_order(device, monkeypatch) -> None:
    """The forward launches ``pair_table`` then ``fwd`` (the record of the device's architecture: ``fwd`` on sm_100a /
    sm_103a, ``fwd_sm_107a`` on sm_107a); the backward ``pair_table`` then ``bwd``, or ``bwd`` alone with
    ``pair_offsets``; the autograd entries launch no pair table in the backward.  No call writes its inputs."""
    launched = []
    entries = []
    original = cake_jit.run_module
    entry_of = cake_jit._entry

    def spy(role, values, *, rows, device_index):
        launched.append(role)
        return original(role, values, rows=rows, device_index=device_index)

    def entry_spy(name, arch):
        entries.append((name, arch))
        return entry_of(name, arch)

    monkeypatch.setattr(cake_jit, "run_module", spy)
    monkeypatch.setattr(cake_jit, "_entry", entry_spy)
    arch = cake_jit.device_arch(device.index)
    x = make_inputs(MIXED, seed=37, device=device, layout="halves")
    before = [t.clone() for t in (x.cz, x.cu, x.dout)]
    pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
    assert launched == ["pair_table", "fwd"]
    assert entries == [("pair_table", arch), ("fwd_sm_107a" if arch == "sm_107a" else "fwd", arch)]
    launched.clear()
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
    assert launched == ["pair_table", "bwd"]
    launched.clear()
    table = pair_table_of(MIXED.lengths, device)
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, pair_offsets=table)
    assert launched == ["bwd"]
    for entry, args in (
        (pkg.cake_gated_kv_compression_det_cz, lambda cz: (cz, x.cu, x.G)),
        (pkg.cake_gated_kv_compression_det, lambda cz: (cz[:, :WIDTH], cz[:, WIDTH:], x.cu, x.G)),
    ):
        launched.clear()
        cz = x.cz.detach().clone().requires_grad_(True)
        entry(*args(cz)).backward(x.dout)
        assert launched == ["pair_table", "fwd", "bwd"], entry.__name__
    for old, new in zip(before, (x.cz, x.cu, x.dout), strict=True):
        assert torch.equal(old, new), "a call wrote its input"
