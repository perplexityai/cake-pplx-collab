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

Backend suite of the Cake deterministic (``det``) Engram row lookup package
(``cake_pplx._backends.cake_engram_lookup_det_train``, the backend of ``cake_pplx.cake_engram_lookup_det_train``):
FP8 E4M3 rows ``[R, 256]`` with one E8M0 scale per 32 columns ``[R, 8]``, gathered by int64 row ids of any shape
into BF16 ``[*row_ids.shape, 256]``, on SM100, SM103 and SM107.

Every comparison is bit for bit (NaN payloads aside, NaN at the same elements) against the FP64 reference of the
feature request (``tests/cake/test_helpers/cake_engram_lookup_ref.py``: the exact product rounded once to BF16).  There
is no error budget anywhere: the lookup rounds once, so the error against FP64 is 0 for every element, which is
stricter than the naive BF16 tier the other families use (this family has no ``cake_naive.py``).

The package has two programs per architecture: the main one (record ``lookup``) and the small one (record
``lookup_small``), which a call launches for at most ``cake_jit.SMALL_MAX_POSITIONS`` looked-up rows.

Without a device: the registry (the two records, each delivered for the three architectures and no other, their
launch facts, their argument plans in the launch order of the host code, the threshold), the record a row count
picks (the small one up to the threshold, the main one above), the launch grid, the launch's arguments (byte views
of the table operands with their strides, then the ids, the output, the row count and the grid), the first launch on
a device (both records' entries looked up, the other program launched first with the same arguments; later launches
there launch one program, the first launch on another device both again), the build manifest
(each record's library per architecture, the recorded source digests), the standalone loader, the prebuilt
libraries served first, no environment read outside the JIT shim, and a one-row table launched with its compact row
strides.

On SM100 / SM103 / SM107, each request acceptance item below runs through each program (the ``program`` fixture
moves the threshold the entry reads, so every call of the item takes that program), in both table layouts (two
contiguous tensors; two views of one packed ``uint8 [R, 264]`` buffer):

* the exhaustive table (every one of the 65,536 E4M3 code / E8M0 byte pairs) through ``[M]`` and ``[T, 24]`` ids:
  the 64,770 non-NaN pairs bitwise, the 60,198 pairs of the scales 2^-117 .. 2^119 exact, the 568 FP32-subnormal
  products not flushed, the 560 products past the FP32 range ``±inf``, NaN exactly at the pairs of a NaN code or the
  NaN scale byte; signed zeros, 2^-9 and ±448 as literal BF16 bits;
* the NaN scale byte ``0xFF`` makes its own 32 columns NaN and no other;
* contiguous and packed tables of the same bytes give the same output bytes;
* random tables and ids of the shapes ``[T, 24]``, ``[M]`` (tails that are not a multiple of the CTA's rows) and a
  rank-3 shape; ``T = 0`` and ``M = 0`` (an empty output, no launch, no library load); ``M = 1``; repeated ids;
  one-row tables (``R = 1``): two contiguous tensors, a packed ``[1, 264]`` buffer and one-row slices of tables with
  the row strides 264 and 272;
* every ``T`` and ``M`` without a rebuild (the loader's cache misses do not move), and the same bits over five runs;
* row byte offsets past 2^31 (a 9,000,000-row table) and output offsets past 2^31 elements (8,388,616 looked-up rows,
  every row checked).  The second test skips only when the device lacks the memory, with the reason; the
  kernel-level suite of the kernel source checks the same offsets on its own.

And the threshold through the entry's own choice: ``SMALL_MAX_POSITIONS`` looked-up rows launch the small program
and one more the main one; both calls have the reference's bits, both programs give the same bytes on both id sets,
and a row id (the last id repeats the first) gives the same bytes on both sides of the threshold.

The public-API suite (``test_cake_engram_lookup_det_train_api.py``) holds the operand checks, the stream, CUDA graph,
checkpoint, workspace and deterministic-mode items.  The suite names its package once, through the shared alias
helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_engram_lookup_ref import (
    EXHAUSTIVE_COUNTS,
    LAYOUTS,
    N_COLS,
    NUM_GROUPS,
    PACKED_ROW_BYTES,
    ROW_DIM,
    Table,
    assert_same_bits,
    empty_table,
    exhaustive_bytes,
    exhaustive_report,
    make_table,
    nan_scale_cells,
    oracle,
    oracle_bits,
    random_bytes,
    random_ids,
    table_bytes,
    write_rows,
)
from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake.cake_engram_lookup_det_train")
cake_backend = cake_package("cake.cake_engram_lookup_det_train", "cake_backend")
cake_build = cake_package("cake.cake_engram_lookup_det_train", "cake_build")
cake_jit = cake_package("cake.cake_engram_lookup_det_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STAGES = ("lookup",)
RECORD = "lookup"
SMALL_RECORD = "lookup_small"
# The registry's records, in registry order: the main program, then the small one.
RECORDS = (RECORD, SMALL_RECORD)
# The programs a device item runs through (the ``program`` fixture): the threshold each sets, so that every call
# takes that program.
PROGRAMS = {"main": 0, "small": 2**31}
ENTRIES = ("cake_engram_lookup_det_train_forward", "cake_engram_lookup_det_train_split_packed")
# The index forms of the exhaustive table: ``[M]`` = arange(256); ``[T, 24]`` = arange(264) % 256 as [11, 24] (the
# last 8 positions repeat rows 0 .. 7).
FORMS = ("M", "T24")
# Random cases: (table rows, row_ids shape, seed); tails off a multiple of the CTA's 32 rows, a rank-3 shape.
RANDOM_CASES = (
    (512, (64, N_COLS), 11),
    (96, (7, N_COLS), 12),
    (4096, (1001,), 13),
    (7, (33,), 14),
    (300, (2, 3, N_COLS), 15),
)
RANDOM_CASE_IDS = [f"R{rows}-{'x'.join(map(str, shape))}" for rows, shape, _ in RANDOM_CASES]
# Tables past 2^31 bytes: 9,000,000 rows (packed 2.38 GB, contiguous 2.30 GB); the high ids start past the row whose
# offset is 2^31 bytes in either layout (8,134,442 packed, 8,388,608 contiguous).
BIG_ROWS = 9_000_000
BIG_HIGH_IDS = (8_400_000, BIG_ROWS)
BIG_LOW_IDS = (0, 8_100_000)
# Outputs past 2^31 elements: 8,388,616 looked-up rows (2,147,485,696 elements, 4.29 GB).
BIG_POSITIONS = 8_388_616

_EXHAUSTIVE: dict[tuple[str, str, str], torch.Tensor] = {}


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the package ships {ARCHES}; this device is {capability}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's library before the first call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


@pytest.fixture(params=tuple(PROGRAMS))
def program(request, monkeypatch) -> str:
    """The program every call of the item takes: the threshold the entry reads at each call, moved to 0 (every call
    the main program) or past any row count (every call the small one)."""
    monkeypatch.setattr(cake_jit, "SMALL_MAX_POSITIONS", PROGRAMS[request.param])
    return request.param


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


def _memory_or_skip(device: torch.device, need: int, what: str) -> None:
    free, _ = torch.cuda.mem_get_info(device)
    if free < need:
        pytest.skip(f"{what} needs {need / 2**30:.1f} GiB of free device memory; {free / 2**30:.1f} GiB are free")


def _lookup(table, row_ids: torch.Tensor) -> torch.Tensor:
    return pkg.cake_engram_lookup_det_train_forward(table.weight, table.scale, row_ids)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


# ----------------------------------------------------------------------------------------------------------------
# Without a device


@pytest.mark.parametrize("name", RECORDS)
def test_the_registry_records_are_well_formed(name) -> None:
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert cake_jit.LIBRARY_PREFIX == "cake_engram_lookup_det_train"
    assert (cake_jit.ROW_DIM, cake_jit.GROUP, cake_jit.GROUPS, cake_jit.PACKED_ROW_BYTES) == (256, 32, 8, 264)
    assert tuple(cake_jit.MODULES) == RECORDS == tuple(cake_backend.RECORDS)
    assert (cake_backend.RECORD, cake_backend.SMALL_RECORD) == RECORDS
    rec = cake_jit.MODULES[name]
    assert rec["stage"] == RECORD
    assert tuple(rec["arches"]) == ARCHES
    assert rec["ffi_entry"] == "run"
    assert re.fullmatch(re.escape(cake_jit.LIBRARY_PREFIX + "_") + "[0-9a-f]{20}", rec["library"]), rec["library"]
    assert rec["kernel_symbol"] == "kernel_" + rec["library"]
    launch = rec["launch"]
    assert launch["cooperative"] is False and launch["use_pdl"] is False
    assert tuple(launch["cluster"]) == (1, 1, 1)
    # The grid's rows per CTA assume the programs' block of WARPS_PER_CTA warps; the lookup uses no shared memory.
    assert tuple(launch["block"]) == (32 * cake_jit.WARPS_PER_CTA, 1, 1)
    assert launch["dynamic_smem_bytes"] == 0


def test_the_two_programs_share_the_launch() -> None:
    """The two records are two programs (two libraries) of one launch: the same block, cluster, launch flags and
    argument plan, so one grid and one argument list serve both."""
    main, small = (cake_jit.MODULES[name] for name in RECORDS)
    assert main["library"] != small["library"]
    assert main["launch"] == small["launch"]
    assert main["arg_plan"] == small["arg_plan"]


@pytest.mark.parametrize("name", RECORDS)
def test_the_argument_plan_is_the_launch_order(name) -> None:
    """The registry's argument plan (the rendered binding's parameters) is the list the host launch passes, then the
    grid."""
    assert cake_backend.parameters(name) == cake_backend._PARAMETERS
    kinds = [kind for kind, _ in cake_jit.MODULES[name]["arg_plan"]]
    assert kinds[-3:] == ["grid", "grid", "grid"]


@pytest.mark.parametrize("name", RECORDS)
def test_no_record_is_delivered_outside_the_architectures(name) -> None:
    rec = cake_jit.MODULES[name]
    for arch in ARCHES:
        assert cake_backend.library_name(name, arch) == f"{rec['library']}_{arch}"
    for arch in ("sm_110a", "sm_90a", "sm_120a", "sm_100f"):
        with pytest.raises(RuntimeError):
            cake_backend.library_name(name, arch)
    with pytest.raises(KeyError):
        cake_backend.record("bwd")


def test_the_row_count_picks_the_program() -> None:
    """At most ``SMALL_MAX_POSITIONS`` looked-up rows take the small program, more the main one; the threshold is a
    positive row count below the 32-bit row limit."""
    limit = cake_jit.SMALL_MAX_POSITIONS
    assert type(limit) is int and 1 <= limit < cake_backend._MAX_POSITIONS, limit
    for positions in (1, limit // 2 + 1, limit):
        assert cake_backend._record_for(positions) == SMALL_RECORD, positions
    for positions in (limit + 1, 2 * limit + 1, 393_216, cake_backend._MAX_POSITIONS - 1):
        assert cake_backend._record_for(positions) == RECORD, positions


def test_the_grid_covers_every_row() -> None:
    """One CTA per ``WARPS_PER_CTA * ROWS_PER_WARP`` rows, capped at ``CTAS_PER_SM[arch]`` CTAs per SM (8 on SM100
    and SM103, 4 on SM107), at least one: a one-dimensional grid the kernel walks grid-stride."""
    assert cake_jit.CTAS_PER_SM == {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4}
    per_cta = cake_jit.WARPS_PER_CTA * cake_jit.ROWS_PER_WARP
    rng = random.Random(8379)
    sizes = [1, 2, per_cta - 1, per_cta, per_cta + 1, 393_216, 1_572_864, 2**31 - 9]
    sizes += [rng.randint(1, 1 << 24) for _ in range(200)]
    for arch in ARCHES:
        for sms in (1, 132, 148, 160, 224):
            cap = sms * cake_jit.CTAS_PER_SM[arch]
            for positions in sizes:
                gx, gy, gz = cake_jit.grid(positions, sms, arch)
                assert (gy, gz) == (1, 1)
                assert gx == min(-(-positions // per_cta), cap), (arch, positions, sms, gx)
                assert 1 <= gx <= cap
                assert gx * per_cta >= positions or gx == cap


def _recorders(monkeypatch) -> list:
    """The FFI entries replaced by recorders (no device): the entry of a record appends ``(record, arch, args)`` to the
    returned list at each launch, and no launch has run on any device yet."""
    calls = []
    monkeypatch.setattr(cake_backend, "_entry", lambda name, arch: lambda *args: calls.append((name, arch, args)))
    monkeypatch.setattr(cake_backend, "_ENTRIES", {})
    return calls


def test_the_launch_passes_byte_views_and_the_grid(monkeypatch) -> None:
    """The launch passes ``weight`` and ``scale`` as ``uint8`` views of the same storage, shape and strides (the
    binding reads the row strides), then ``row_ids``, ``out`` and the row count as given, then the grid ``(gx, 1,
    1)``: one argument per parameter of the argument plan, three grid values.  The second launch on a device, which
    launches its own program only (no device: the FFI entries are recorders)."""
    calls = _recorders(monkeypatch)
    buf = torch.zeros(5, PACKED_ROW_BYTES, dtype=torch.uint8)
    weight, scale = buf[:, :ROW_DIM].view(torch.float8_e4m3fn), buf[:, ROW_DIM:].view(torch.float8_e8m0fnu)
    row_ids = torch.zeros(3, N_COLS, dtype=torch.int64)
    out = torch.empty(3, N_COLS, ROW_DIM, dtype=torch.bfloat16)
    for _ in range(2):
        cake_backend._launch("sm_103a", RECORD, weight, scale, row_ids, out, 72, 3, 0)
    [(name, arch, args)] = calls[2:]
    assert (name, arch) == (RECORD, "sm_103a")
    assert len(args) == len(cake_backend._PARAMETERS) + 3
    w, s, ids, o, positions, gx, gy, gz = args
    for view, src in ((w, weight), (s, scale)):
        assert view.dtype == torch.uint8
        assert view.data_ptr() == src.data_ptr() and view.shape == src.shape and view.stride() == src.stride()
    assert ids is row_ids and o is out
    assert (positions, gx, gy, gz) == (72, 3, 1, 1)
    assert all(type(v) is int for v in (positions, gx, gy, gz))


def test_a_one_row_table_goes_in_with_compact_strides(monkeypatch) -> None:
    """A one-row table goes in as byte views of the same storage with the compact row strides 256 and 8, not its own
    (the packed 264 here): a DLPack exporter that keeps a compact view's strides may still rewrite another size-1
    stride, and the binding requires multiples of 8 (no device: the FFI entry is a recorder)."""
    calls = _recorders(monkeypatch)
    buf = torch.zeros(1, PACKED_ROW_BYTES, dtype=torch.uint8)
    weight, scale = buf[:, :ROW_DIM].view(torch.float8_e4m3fn), buf[:, ROW_DIM:].view(torch.float8_e8m0fnu)
    assert weight.stride() == (PACKED_ROW_BYTES, 1) and scale.stride() == (PACKED_ROW_BYTES, 1)
    ids, out = torch.zeros(4, dtype=torch.int64), torch.empty(4, ROW_DIM, dtype=torch.bfloat16)
    cake_backend._launch("sm_100a", SMALL_RECORD, weight, scale, ids, out, 4, 1, 0)
    assert [name for name, _, _ in calls] == [RECORD, SMALL_RECORD]  # the first launch on the device: both programs
    for _, _, args in calls:
        w, s = args[:2]
        assert w.dtype == torch.uint8 and s.dtype == torch.uint8
        assert (w.data_ptr(), tuple(w.shape), w.stride()) == (weight.data_ptr(), (1, ROW_DIM), (ROW_DIM, 1))
        assert (s.data_ptr(), tuple(s.shape), s.stride()) == (scale.data_ptr(), (1, NUM_GROUPS), (NUM_GROUPS, 1))


def test_the_first_launch_on_a_device_launches_both_programs(monkeypatch) -> None:
    """The first launch on a device looks up the entries of both records and launches the program the call did not
    pick first, with the same arguments, then the call's own (the programs write the same bytes), so the first call
    on a device runs each kernel once there and a later call, one captured in a CUDA graph included, loads nothing.
    Later launches on that device launch their own program only; the first launch on another device launches both
    again (no device: the FFI entries are recorders)."""
    calls = _recorders(monkeypatch)
    buf = torch.zeros(5, PACKED_ROW_BYTES, dtype=torch.uint8)
    weight, scale = buf[:, :ROW_DIM].view(torch.float8_e4m3fn), buf[:, ROW_DIM:].view(torch.float8_e8m0fnu)
    ids, out = torch.zeros(4, dtype=torch.int64), torch.empty(4, ROW_DIM, dtype=torch.bfloat16)
    for name, other in ((SMALL_RECORD, RECORD), (RECORD, SMALL_RECORD)):
        calls.clear()
        monkeypatch.setattr(cake_backend, "_ENTRIES", {})
        cake_backend._launch("sm_107a", name, weight, scale, ids, out, 4, 1, 0)
        assert [(record, arch) for record, arch, _ in calls] == [(other, "sm_107a"), (name, "sm_107a")]
        (_, _, first), (_, _, own) = calls
        assert len(own) == len(cake_backend._PARAMETERS) + 3
        assert all(a is b for a, b in zip(first, own, strict=True))
        assert sorted(cake_backend._ENTRIES) == sorted((record, 0) for record in RECORDS)
    calls.clear()
    for name in (SMALL_RECORD, RECORD, RECORD, SMALL_RECORD):
        cake_backend._launch("sm_107a", name, weight, scale, ids, out, 4, 1, 0)
    assert [record for record, _, _ in calls] == [SMALL_RECORD, RECORD, RECORD, SMALL_RECORD]
    calls.clear()
    cake_backend._launch("sm_107a", SMALL_RECORD, weight, scale, ids, out, 4, 1, 1)
    assert [record for record, _, _ in calls] == [RECORD, SMALL_RECORD]
    assert sorted(cake_backend._ENTRIES) == sorted((record, index) for record in RECORDS for index in (0, 1))


def test_the_package_exports_the_backend_entries() -> None:
    for name in ENTRIES:
        assert getattr(pkg, name) is getattr(cake_backend, name), name
        assert name in pkg.__all__ and name in cake_backend.__all__, name


def test_manifest_covers_the_records_and_their_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    libraries = [cake_jit.MODULES[name]["library"] for name in RECORDS]
    assert set(entries) == {f"{library}_{arch}" for library in libraries for arch in ARCHES}
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
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {len(RECORDS) * len(ARCHES)}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path; without one the JIT spec names the entry."""
    _manifest_or_skip()
    for name in RECORDS:
        for arch in ARCHES:
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
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES, (
        sorted(shim - JIT_SHIM_ENVIRONMENT_VARIABLES) if shim is not None else "no _cake_pplx_jit.py",
        sorted(JIT_SHIM_ENVIRONMENT_VARIABLES - shim) if shim is not None else None,
    )


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: the exhaustive table


@pytest.mark.parametrize("name", RECORDS)
def test_the_record_loads(name, device) -> None:
    arch = cake_backend.device_arch(device.index)
    assert arch in ARCHES
    assert callable(getattr(cake_backend.load(name, arch), cake_jit.MODULES[name]["ffi_entry"]))


def _exhaustive_out(device: torch.device, layout: str, form: str, program: str) -> torch.Tensor:
    """The lookup of the exhaustive table (``layout``) through the ``form`` ids by ``program``, as BF16 ``[256, 256]``
    on the CPU (row ``b``, column ``c``: E4M3 code ``c`` times E8M0 byte ``b``); computed once per process."""
    key = (layout, form, program)
    if key not in _EXHAUSTIVE:
        table = make_table(*exhaustive_bytes(), layout, device)
        if form == "M":
            ids = torch.arange(256, device=device)
        else:
            ids = (torch.arange(11 * N_COLS, device=device) % 256).view(11, N_COLS)
        out = _lookup(table, ids)
        assert (tuple(out.shape), out.dtype, out.is_contiguous()) == ((*ids.shape, ROW_DIM), torch.bfloat16, True)
        flat = out.reshape(-1, ROW_DIM)
        if form == "T24":
            # The last 8 positions look rows 0 .. 7 up again: the same bits.
            assert torch.equal(_bits(flat[256:]), _bits(flat[:8]))
        _EXHAUSTIVE[key] = flat[:256].cpu()
    return _EXHAUSTIVE[key]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_exhaustive_pairs_bitwise(layout, form, program, device) -> None:
    """Every (E4M3 code, E8M0 byte) pair: the 64,770 non-NaN pairs carry the bits of the FP64 product rounded once to
    BF16, and NaN is exactly at the pairs of a NaN code (``0x7F``, ``0xFF``) or the NaN byte ``0xFF``."""
    out = _exhaustive_out(device, layout, form, program)
    report = exhaustive_report(out)
    assert report["non_nan_pairs"] == EXHAUSTIVE_COUNTS["non_nan_pairs"] == 64_770
    assert report["non_nan_equal"] == report["non_nan_pairs"], report
    assert report["nan_mismatches"] == 0, report
    assert report["nan_pairs"] == 65_536 - 64_770, report


@pytest.mark.parametrize("layout", LAYOUTS)
def test_mid_range_products_exact(layout, program, device) -> None:
    """For the scales 2^-117 .. 2^119 (60,198 pairs) every product is exactly representable in BF16, and the output
    is the exact product."""
    report = exhaustive_report(_exhaustive_out(device, layout, "M", program))
    assert report["mid_range_pairs"] == EXHAUSTIVE_COUNTS["mid_range_pairs"] == 60_198
    assert report["mid_range_exact"] == report["mid_range_pairs"], report


@pytest.mark.parametrize("layout", LAYOUTS)
def test_fp32_subnormal_products_not_flushed(layout, program, device) -> None:
    """The 568 products below 2^-126 in magnitude (FP32 subnormals) carry the oracle's bits: the multiply flushes
    neither the subnormal scale 2^-127 nor a subnormal product.  Some of them round to a non-zero BF16, so a flush to
    zero would change their bits."""
    report = exhaustive_report(_exhaustive_out(device, layout, "M", program))
    assert report["fp32_subnormal_pairs"] == EXHAUSTIVE_COUNTS["fp32_subnormal_pairs"] == 568
    assert report["fp32_subnormal_equal"] == report["fp32_subnormal_pairs"], report
    assert report["fp32_subnormal_nonzero"] > 0, report


@pytest.mark.parametrize("layout", LAYOUTS)
def test_overflow_products_are_inf(layout, program, device) -> None:
    """The 560 products of magnitude 2^128 or more are ``inf`` with the product's sign, as in the release."""
    report = exhaustive_report(_exhaustive_out(device, layout, "M", program))
    assert report["fp32_overflow_pairs"] == EXHAUSTIVE_COUNTS["fp32_overflow_pairs"] == 560
    assert report["fp32_overflow_inf"] == report["fp32_overflow_pairs"], report


# (E8M0 byte, E4M3 code) -> the output's BF16 bits.
SPECIAL_BITS = {
    (127, 0x00): 0x0000,  # +0 * 1
    (127, 0x80): 0x8000,  # -0 * 1 = -0
    (0, 0x80): 0x8000,  # -0 * 2^-127 = -0
    (254, 0x00): 0x0000,  # +0 * 2^127
    (127, 0x01): 0x3B00,  # 2^-9 (the smallest E4M3 subnormal) * 1
    (127, 0x81): 0xBB00,  # -2^-9 * 1
    (9, 0x01): 0x0040,  # 2^-9 * 2^-118 = 2^-127: a BF16 subnormal
    (127, 0x7E): 0x43E0,  # 448 (the largest E4M3 value) * 1
    (127, 0xFE): 0xC3E0,  # -448 * 1
    (0, 0x7E): 0x0460,  # 448 * 2^-127 (an FP32-subnormal scale) = 1.75 * 2^-119
    (0, 0xFE): 0x8460,  # -448 * 2^-127
    (254, 0x7E): 0x7F80,  # 448 * 2^127: +inf
    (254, 0xFE): 0xFF80,  # -448 * 2^127: -inf
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_special_values_bits(layout, program, device) -> None:
    """Signed zeros, the smallest subnormal 2^-9 and the largest value ±448 of E4M3, times scales 1, 2^-127, 2^-118
    and 2^127: the literal BF16 bits (the oracle gives the same, :func:`test_exhaustive_pairs_bitwise`)."""
    bits = _bits(_exhaustive_out(device, layout, "M", program))
    got = {key: int(bits[key[0], key[1]]) & 0xFFFF for key in SPECIAL_BITS}
    assert got == SPECIAL_BITS


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: random tables, layouts and id shapes


@pytest.mark.parametrize("layout", LAYOUTS)
def test_nan_scale_byte_poisons_only_its_group(layout, program, device) -> None:
    """A scale byte ``0xFF`` makes the 32 columns of its group NaN; the other 224 columns of the row are NaN only where
    the E4M3 code is NaN, and every element has the oracle's bits."""
    rows, every = 96, 3
    table = make_table(*random_bytes(rows, seed=81, nan_every=every), layout, device)
    ids = torch.arange(rows, device=device)
    out = _lookup(table, ids)
    assert_same_bits(out, oracle(table, ids), f"nan groups {layout}")
    bits = _bits(out).cpu()
    nan = (bits & 0x7FFF) > 0x7F80
    weight_bytes, _ = table_bytes(table)
    code_nan = (weight_bytes.cpu() & 0x7F) == 0x7F
    cells = nan_scale_cells(rows, every)
    assert len(cells) == rows // every
    for r, g in cells:
        group = slice(32 * g, 32 * g + 32)
        assert bool(nan[r, group].all()), (r, g)
        others = torch.ones(ROW_DIM, dtype=torch.bool)
        others[group] = False
        assert torch.equal(nan[r, others], code_nan[r, others]), (r, g)


@pytest.mark.parametrize("form", ("M", "T24"))
def test_contiguous_equals_packed(form, program, device) -> None:
    """The same table bytes as two contiguous tensors and as the views of one packed ``[R, 264]`` buffer, the same
    ids: the same output bytes, and both are the oracle's."""
    rows = 512
    weight_bytes, scale_bytes = random_bytes(rows, seed=21, nan_every=11)
    shape = (1001,) if form == "M" else (64, N_COLS)
    ids = random_ids(rows, shape, seed=22, device=device)
    outs = {layout: _lookup(make_table(weight_bytes, scale_bytes, layout, device), ids) for layout in LAYOUTS}
    assert torch.equal(_bits(outs["contiguous"]), _bits(outs["packed"]))
    table = make_table(weight_bytes, scale_bytes, "contiguous", device)
    assert_same_bits(outs["contiguous"], oracle(table, ids), f"contiguous {form}")


@pytest.mark.parametrize(("rows", "shape", "seed"), RANDOM_CASES, ids=RANDOM_CASE_IDS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_random_tables_match_the_oracle(layout, rows, shape, seed, program, device) -> None:
    """Random tables (every E4M3 code and finite E8M0 byte, some NaN scale bytes) and random ids of the shapes
    ``[T, 24]``, ``[M]`` and rank 3: the output has the shape ``(*row_ids.shape, 256)`` and the oracle's bits."""
    table = make_table(*random_bytes(rows, seed=seed, nan_every=13), layout, device)
    ids = random_ids(rows, shape, seed=seed + 100, device=device)
    out = _lookup(table, ids)
    assert (tuple(out.shape), out.dtype, out.device, out.is_contiguous()) == (
        (*shape, ROW_DIM),
        torch.bfloat16,
        device,
        True,
    )
    assert_same_bits(out, oracle(table, ids), f"{layout} R={rows} {shape}")


@pytest.mark.parametrize("rows", (64, 0), ids=("R64", "R0"))
@pytest.mark.parametrize("shape", ((0, N_COLS), (0,)), ids=("T0", "M0"))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_empty_ids_return_empty_without_launch(layout, shape, rows, device, monkeypatch) -> None:
    """``T = 0`` or ``M = 0`` (also with an empty table): an empty contiguous BF16 output of shape
    ``(*row_ids.shape, 256)`` on the device; nothing is launched and no library is loaded."""
    table = make_table(*random_bytes(rows, seed=31), layout, device)
    misses = cake_backend.load.cache_info().misses
    launched = []
    monkeypatch.setattr(cake_backend, "_launch", lambda *args: launched.append(args))
    out = _lookup(table, torch.empty(shape, dtype=torch.int64, device=device))
    assert launched == []
    assert (tuple(out.shape), out.dtype, out.device, out.is_contiguous()) == (
        (*shape, ROW_DIM),
        torch.bfloat16,
        device,
        True,
    )
    assert cake_backend.load.cache_info().misses == misses


@pytest.mark.parametrize("shape", ((1,), (1, N_COLS), ()), ids=("M1", "T1", "scalar"))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_single_row(layout, shape, program, device) -> None:
    """``M = 1``, ``T = 1`` and a 0-dimensional id: the oracle's bits."""
    table = make_table(*random_bytes(40, seed=41, nan_every=9), layout, device)
    ids = random_ids(40, shape, seed=42, device=device)
    out = _lookup(table, ids)
    assert tuple(out.shape) == (*shape, ROW_DIM)
    assert_same_bits(out, oracle(table, ids), f"single {layout} {shape}")


def _one_row_table(form: str, device: torch.device) -> Table:
    """Row 5 of a random 16-row table as a one-row table (``R = 1``) in ``form``: the two contiguous tensors, the
    views of a packed ``[1, 264]`` buffer, a one-row slice of a packed ``[16, 264]`` table, or a one-row slice of a
    table with a 272-byte row stride."""
    weight_bytes, scale_bytes = random_bytes(16, seed=71, nan_every=7)
    if form in LAYOUTS:
        return make_table(weight_bytes[5:6], scale_bytes[5:6], form, device)
    if form == "packed-slice":
        full = make_table(weight_bytes, scale_bytes, "packed", device)
        return Table("packed", full.weight[5:6], full.scale[5:6])
    buf = torch.zeros(16, 272, dtype=torch.uint8, device=device)
    buf[:, :ROW_DIM] = weight_bytes.to(device)
    buf[:, ROW_DIM:PACKED_ROW_BYTES] = scale_bytes.to(device)
    weight = buf[5:6, :ROW_DIM].view(torch.float8_e4m3fn)
    return Table("packed", weight, buf[5:6, ROW_DIM:PACKED_ROW_BYTES].view(torch.float8_e8m0fnu))


@pytest.mark.parametrize("shape", ((5,), (3, N_COLS)), ids=("M5", "T3"))
@pytest.mark.parametrize(
    ("form", "row_stride"),
    (("contiguous", ROW_DIM), ("packed", PACKED_ROW_BYTES), ("packed-slice", PACKED_ROW_BYTES), ("stride-272", 272)),
)
def test_one_row_tables(form, row_stride, shape, program, device) -> None:
    """A one-row table (``R = 1``) in each form, its one id repeated through ``[M]`` and ``[T, 24]`` ids: the oracle's
    bits.  The packed and strided forms have a row stride other than the compact 256; the launch passes the compact
    strides, which change no address (the one valid id is 0)."""
    table = _one_row_table(form, device)
    assert table.rows == 1 and table.weight.stride(0) == row_stride
    assert table.scale.stride(0) == (NUM_GROUPS if form == "contiguous" else row_stride)
    ids = torch.zeros(shape, dtype=torch.int64, device=device)
    out = _lookup(table, ids)
    assert tuple(out.shape) == (*shape, ROW_DIM)
    assert_same_bits(out, oracle(table, ids), f"one row {form} {shape}")


@pytest.mark.parametrize("layout", LAYOUTS)
def test_repeated_row_ids(layout, program, device) -> None:
    """Repeated ids (8 rows looked up 4096 times; one row 32 times): every position has the oracle's bits, and the
    positions of one id carry the same bits."""
    table = make_table(*random_bytes(8, seed=51, nan_every=5), layout, device)
    ids = random_ids(8, (4096,), seed=52, device=device)
    out = _lookup(table, ids)
    assert_same_bits(out, oracle(table, ids), f"repeats {layout}")
    bits = _bits(out)
    for r in range(8):
        rows = bits[ids == r]
        assert rows.shape[0] > 0 and bool((rows == rows[0]).all()), r
    same = _lookup(table, torch.full((32,), 5, dtype=torch.int64, device=device))
    assert bool((_bits(same) == _bits(same)[0]).all())
    assert_same_bits(same, oracle(table, torch.full((32,), 5, dtype=torch.int64, device=device)), "one id")


def test_no_rebuild_across_shapes(device) -> None:
    """Every ``T`` and ``M``, in both layouts, runs one of the two loaded programs: the loader's and the entry's cache
    misses do not move after the first call of each."""
    tables = {layout: make_table(*random_bytes(4096, seed=61), layout, device) for layout in LAYOUTS}
    for positions in (1, cake_jit.SMALL_MAX_POSITIONS + 1):  # the first call of each program
        _lookup(tables["contiguous"], random_ids(4096, (positions,), seed=62, device=device))
    before = (cake_backend.load.cache_info().misses, cake_backend._entry.cache_info().misses)
    for shape in ((1, N_COLS), (7, N_COLS), (16384, N_COLS), (1,), (1001,), (393_216,)):
        ids = random_ids(4096, shape, seed=len(shape) + shape[0], device=device)
        for table in tables.values():
            _lookup(table, ids)
    torch.cuda.synchronize(device)
    assert (cake_backend.load.cache_info().misses, cake_backend._entry.cache_info().misses) == before


@pytest.mark.parametrize("layout", LAYOUTS)
def test_run_to_run_bitwise(layout, program, device) -> None:
    """Five calls on the same table and ids give the same output bytes (NaN payloads included)."""
    table = make_table(*random_bytes(4096, seed=71, nan_every=5), layout, device)
    ids = random_ids(4096, (2048, N_COLS), seed=72, device=device)
    first = _lookup(table, ids)
    for _ in range(4):
        assert torch.equal(_bits(_lookup(table, ids)), _bits(first))


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: 64-bit offsets


@pytest.mark.parametrize("layout", LAYOUTS)
def test_row_offsets_past_2_31_bytes(layout, program, device) -> None:
    """A 9,000,000-row table (2.3 to 2.4 GB): half the ids lie past the row at byte offset 2^31, and every looked-up
    row has the oracle's bits."""
    _memory_or_skip(device, BIG_ROWS * PACKED_ROW_BYTES + (1 << 30), f"a {BIG_ROWS}-row {layout} table")
    table = empty_table(BIG_ROWS, layout, device)
    gen = torch.Generator().manual_seed(81)
    low = torch.randint(*BIG_LOW_IDS, (2048,), generator=gen, dtype=torch.int64)
    high = torch.randint(*BIG_HIGH_IDS, (2048,), generator=gen, dtype=torch.int64)
    ids = torch.stack((low, high), dim=1).reshape(-1).to(device)  # interleaved: low, high, low, ...
    assert int(high.min()) * table.weight.stride(0) >= 2**31
    written = ids.unique()
    weight_bytes, scale_bytes = random_bytes(written.numel(), seed=82, nan_every=17)
    write_rows(table, written, weight_bytes, scale_bytes)
    out = _lookup(table, ids)
    assert_same_bits(out, oracle(table, ids), f"rows past 2^31 bytes {layout}")
    del table
    torch.cuda.empty_cache()


def test_output_offsets_past_2_31_elements(program, device) -> None:
    """8,388,616 looked-up rows (``out`` holds 2,147,485,696 elements, 4.29 GB, past 2^31 elements and 2^32 bytes),
    not a multiple of the CTA's rows: every row of the output has the oracle's bits.  Skips only when the device
    lacks the memory."""
    positions = BIG_POSITIONS
    assert positions * ROW_DIM > 2**31 and positions * ROW_DIM * 2 > 2**32
    _memory_or_skip(device, positions * (ROW_DIM * 2 + 8) + (3 << 30), f"{positions} looked-up rows")
    table = make_table(*random_bytes(64, seed=91, nan_every=7), "packed", device)
    ids = random_ids(64, (positions,), seed=92, device=device)
    out = _lookup(table, ids)
    assert tuple(out.shape) == (positions, ROW_DIM)
    step = 1 << 17
    for start in range(0, positions, step):
        stop = min(start + step, positions)
        assert_same_bits(out[start:stop], oracle_bits(table, ids[start:stop]), f"rows {start}..{stop}")
    del out, ids
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: the threshold between the two programs


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_threshold_keeps_the_bytes(layout, device, monkeypatch) -> None:
    """``SMALL_MAX_POSITIONS`` looked-up rows launch the small program and one more the main one (the entry's own
    choice); the ids of the first call lead those of the second, and the last id repeats the first.  Both calls have
    the reference's bits, both programs give the same bytes on both id sets, and a row id gives the same bytes on
    both sides of the threshold."""
    limit = cake_jit.SMALL_MAX_POSITIONS
    table = make_table(*random_bytes(4096, seed=171, nan_every=11), layout, device)
    ids = random_ids(4096, (limit + 1,), seed=172, device=device)
    ids[limit] = ids[0]
    launched = []
    launch = cake_backend._launch

    def recorded(arch, name, *args):
        launched.append(name)
        return launch(arch, name, *args)

    monkeypatch.setattr(cake_backend, "_launch", recorded)
    small = _lookup(table, ids[:limit])
    main = _lookup(table, ids)
    assert launched == [SMALL_RECORD, RECORD]
    assert_same_bits(small, oracle(table, ids[:limit]), f"{limit} rows {layout}")
    assert_same_bits(main, oracle(table, ids), f"{limit + 1} rows {layout}")
    assert torch.equal(_bits(main[:limit]), _bits(small)), "a row id gave other bytes on the two sides"
    assert torch.equal(_bits(main[limit]), _bits(small[0])), "the repeated id gave other bytes on the two sides"
    for threshold, program in ((0, RECORD), (2**31, SMALL_RECORD)):
        monkeypatch.setattr(cake_jit, "SMALL_MAX_POSITIONS", threshold)
        for got, want in ((_lookup(table, ids[:limit]), small), (_lookup(table, ids), main)):
            assert torch.equal(_bits(got), _bits(want)), (program, tuple(got.shape))
        assert launched[-2:] == [program, program]
