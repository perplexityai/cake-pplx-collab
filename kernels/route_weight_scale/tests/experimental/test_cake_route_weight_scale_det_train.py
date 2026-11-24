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

Backend suite of the Cake deterministic route-weight row scaling training package
(``cake_pplx._backends.cake_route_weight_scale_det_train``, the backend of
``cake_pplx.cake_route_weight_scale_det_train``): ``y = bf16(w x)``, ``dx = bf16(w dy)`` and the FP32
``dw = sum over c of dy x``, ``H`` 4096 and 3584, on SM100 and SM103.

Without a device: the registry (five records, ``<stage>_h<width>`` and ``fwd_h4096_t32``, each delivered for the two
architectures, its launch facts, its rows per CTA and its argument plan in the order the host passes it; literal tables
the exporter reads), the record of each call (``fwd_h4096_t32`` below the small-row bound), the launch grid
(``nblocks = ceil(rows / rows_per_cta)`` CTAs, at most ``2**31 - nblocks``), the refusal of a record that takes its
parameters in another order, the capability map, the operand checks (dtype, shape, width, strides, alignment, ``w``
and ``dw`` layout, CPU tensors), the build manifest (every record's library, the recorded source digests) and no
environment read outside the JIT shim.

On SM100 / SM103, the contract of issue #259, each without a tolerance where the contract is exact:

* ``y`` and ``dx`` equal ``(w[:, None] * x.float()).bfloat16()`` and ``(w[:, None] * dy.float()).bfloat16()`` bit for
  bit, and ``dw`` equals the stated summation order (:func:`tree_row_sum`) bit for bit, at row counts 1 to 65537,
  both widths and signed weights.
* ``dw`` against an FP64 reference of the same inputs (rel-L2 over the whole vector) is no worse than the FP32 row
  sum ``(dy.float() * x.float()).sum(1)``: on the issue's own draws (65536 rows, both widths) against the issue's CPU
  FP32 sum, and beside the same sum on the GPU (reported); both numbers are printed.
* Rows with ``w = 0`` (``y`` and ``dx`` signed zeros, ``dw`` the row sum, independent of ``w``), all-zero ``x`` rows
  (``dw`` 0), ``rows`` 0 (empty outputs, no launch) and 1, ``rows`` 1,048,576 at ``H`` 4096 (more than 2^31
  elements; skipped with a reason when the device has less than 30 GiB free).
* Row slices of wider and longer buffers as inputs and outputs, read and written in place (the bytes around them
  untouched); operands of another layout refused before any allocation or launch.
* Run-to-run bits: repeated calls, a side stream, a CUDA graph replay, another current device.
* One library per record and architecture serves every row count (no library loaded after the first call of each
  record); at ``H`` 4096 the row counts on both sides of the small-row bound give the same bits, ``y`` the contract's;
  no host synchronization (``torch.cuda.set_sync_debug_mode("error")``), the autograd function saves only
  ``x`` and ``w`` and gives the steps' bits, ``torch.utils.checkpoint(..., use_reentrant=False)`` gives the plain
  call's gradients bit for bit, an expanded upstream gradient is read, no double backward.
* Peak extra device memory beyond the outputs, measured and reported, is ``O(rows)`` (0 here, up to the caching
  allocator's rounding), beside the eager PyTorch formula's.

FP64 appears only in the references.  The suite names its package once, through the shared alias helper, and imports
nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.route_weight_scale_reference import (
    bits,
    fp32_row_sum,
    fp64_row_sum,
    issue_inputs,
    make_inputs,
    rel_l2,
    report,
    same_bits,
    scaled,
    tree_row_sum,
)

pkg = cake_package("cake_route_weight_scale_det_train")
cake_jit = cake_package("cake_route_weight_scale_det_train", "cake_jit")

forward = pkg.cake_route_weight_scale_det_train_forward
backward = pkg.cake_route_weight_scale_det_train_backward
route_weight_scale = pkg.cake_route_weight_scale_det

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
WIDTHS = (4096, 3584)
STAGES = ("fwd", "bwd")
RECORDS = ("fwd_h4096", "fwd_h4096_t32", "fwd_h3584", "bwd_h4096", "bwd_h3584")
PARAMETERS = {
    "fwd": ["x", "w", "y", "rows", "x_stride", "y_stride"],
    "bwd": ["x", "dy", "w", "dx", "dw", "rows", "x_stride", "dy_stride", "dx_stride"],
}
LAUNCH = {"block": (128, 1, 1), "cluster": (1, 1, 1), "cooperative": False, "dynamic_smem_bytes": 0, "use_pdl": False}
# Each record's launch facts: ``LAUNCH``, with a CTA of 32 threads for ``fwd_h4096_t32``.
LAUNCHES = {name: dict(LAUNCH, block=(32, 1, 1) if name == "fwd_h4096_t32" else (128, 1, 1)) for name in RECORDS}
# The rows one CTA of each record owns.
ROWS_PER_CTA = {"fwd_h4096": 4, "fwd_h4096_t32": 1, "fwd_h3584": 4, "bwd_h4096": 1, "bwd_h3584": 1}
# The row count from which the H 4096 forward takes ``fwd_h4096`` (``fwd_h4096_t32`` below it).
SMALL_ROWS_BOUND = cake_jit.SMALL_ROWS["fwd_h4096"][0]
# Row counts of the bit tests: 1 to 8 (one partial block of 4 rows and two), odd counts, and counts past a block.
ROW_COUNTS = (1, 2, 3, 4, 5, 7, 8, 129, 1000, 4097, 65537)
# The issue's largest row count: at H 4096 more than 2^31 elements.
MAX_ISSUE_ROWS = 1_048_576
# Free device memory the 1,048,576-row test needs: x, dy and dx (24 GiB) and the references of one chunk.
LARGE_FREE_BYTES = 30 << 30


# ----------------------------------------------------------------------------------------------------------------
# Fixtures


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not pkg.is_cake_route_weight_scale_det_train_supported(torch.device("cuda")):
        pytest.skip(f"the package ships {ARCHES}; this device is {torch.cuda.get_device_capability()}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first call.
    for spec in pkg.specs_for_arch(pkg.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture
def launches(monkeypatch) -> list:
    """Every kernel launch of the package, as ``(stage, width, rows)``, in order: ``_launch`` (the one launch point)
    wrapped."""
    found: list = []
    launch = pkg._launch

    def spy(stage, width, device_index, rows, *arguments):
        found.append((stage, width, rows))
        return launch(stage, width, device_index, rows, *arguments)

    monkeypatch.setattr(pkg, "_launch", spy)
    return found


def _step(x, dy, w) -> dict[str, torch.Tensor]:
    y = forward(x, w)
    dx, dw = backward(x, dy, w)
    return {"y": y, "dx": dx, "dw": dw}


def _expect_contract(x, dy, w, found: dict[str, torch.Tensor]) -> None:
    """``y`` and ``dx`` are the FP32 product rounded once, and ``dw`` the stated order, bit for bit."""
    assert same_bits(found["y"], scaled(w, x)), "y"
    assert same_bits(found["dx"], scaled(w, dy)), "dx"
    assert same_bits(found["dw"], tree_row_sum(x, dy)), "dw"


# ----------------------------------------------------------------------------------------------------------------
# Without a device: the registry, the grid, the capability map, the operand checks, the manifest


pytestmark_cpu = pytest.mark.cpu_only


@pytestmark_cpu
def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert dict(cake_jit.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.WIDTHS) == WIDTHS
    assert cake_jit.PACKAGE == "cake_pplx._backends.cake_route_weight_scale_det_train"
    assert cake_jit.LIBRARY_PREFIX == "cake_route_weight_scale_det_train"
    assert sorted(cake_jit.MODULES) == sorted(RECORDS)
    libraries = set()
    for name, record in cake_jit.MODULES.items():
        stage, width = record["stage"], record["width"]
        assert name in (f"{stage}_h{width}", f"{stage}_h{width}_t32"), name
        assert record["rows_per_cta"] == ROWS_PER_CTA[name], name
        assert list(record["arches"]) == list(ARCHES), name
        assert record["ffi_entry"] == "run", name
        assert dict(record["launch"]) == LAUNCHES[name], name
        assert record["library"].startswith(cake_jit.LIBRARY_PREFIX + "_"), name
        assert record["kernel_symbol"] == "kernel_" + record["library"], name
        assert list(pkg.parameters(name)) == PARAMETERS[stage], name
        kinds = [kind for kind, _ in record["arg_plan"]]
        assert kinds[-3:] == ["grid", "grid", "grid"], name
        buffers = [key for kind, key in record["arg_plan"] if kind == "buffer"]
        assert buffers == [key for key in PARAMETERS[stage] if key in {"x", "dy", "w", "y", "dx", "dw"}], name
        libraries.add(record["library"])
    assert len(libraries) == len(RECORDS)


@pytestmark_cpu
def test_record_name_takes_the_small_row_record_below_its_bound() -> None:
    """A call's record is ``<stage>_h<width>``, except the H 4096 forward below the bound of ``SMALL_ROWS``, which
    takes ``fwd_h4096_t32``."""
    assert cake_jit.SMALL_ROWS == {"fwd_h4096": (SMALL_ROWS_BOUND, "fwd_h4096_t32")}
    assert isinstance(SMALL_ROWS_BOUND, int) and 1 < SMALL_ROWS_BOUND < cake_jit.MAX_ROWS
    for stage in STAGES:
        for width in WIDTHS:
            for rows in (1, SMALL_ROWS_BOUND - 1, SMALL_ROWS_BOUND, 4097, 65536):
                small = (stage, width) == ("fwd", 4096) and rows < SMALL_ROWS_BOUND
                expected = "fwd_h4096_t32" if small else f"{stage}_h{width}"
                assert cake_jit.record_name(stage, width, rows) == expected, (stage, width, rows)
    with pytest.raises(KeyError):
        cake_jit.record_name("fwd", 2048, 1)
    with pytest.raises(KeyError):
        cake_jit.record_name("bwd_unweighted", 4096, 1)
    with pytest.raises(ValueError):
        cake_jit.record_name("fwd", 4096, 0)


@pytestmark_cpu
def test_registry_is_a_literal_table() -> None:
    """The exporter and the manifest checker read ``MODULES`` and the host-plan constants with
    ``ast.literal_eval``."""
    tree = ast.parse((PACKAGE_DIR / "cake_jit.py").read_text())
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0]
        else:
            continue
        if isinstance(target, ast.Name) and node.value is not None:
            try:
                literals[target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    for name in ("MODULES", "SMALL_ROWS", "STAGES", "WIDTHS", "ARCHES", "PACKAGE", "LIBRARY_PREFIX"):
        assert name in literals, name
    assert literals["MODULES"] == cake_jit.MODULES
    assert literals["SMALL_ROWS"] == cake_jit.SMALL_ROWS


@pytestmark_cpu
def test_grid_is_one_cta_per_block_of_the_records_rows() -> None:
    """One CTA per block of the record's ``rows_per_cta`` rows, at most ``2**31 - nblocks`` CTAs: the programs stride
    over the blocks with a 32-bit counter."""
    for name in RECORDS:
        if ROWS_PER_CTA[name] == 4:
            cases = ((1, 1), (3, 1), (4, 1), (5, 2), (8, 2), (4097, 1025), (MAX_ISSUE_ROWS, MAX_ISSUE_ROWS // 4))
        else:
            cases = tuple((rows, rows) for rows in (1, 3, 4, 5, 8, 4097, MAX_ISSUE_ROWS))
        for rows, ctas in cases:
            assert cake_jit.grid(rows, name) == (ctas, 1, 1), (name, rows)
        with pytest.raises(ValueError):
            cake_jit.grid(0, name)
    # Past 2**30 blocks the grid shrinks so that the block counter stays below 2**31.
    assert cake_jit.grid(2**30, "bwd_h4096") == (2**30, 1, 1)
    assert cake_jit.grid(2**30 + 1, "bwd_h4096") == (2**30 - 1, 1, 1)
    assert cake_jit.grid(2**31 - 256, "bwd_h4096") == (256, 1, 1)
    assert cake_jit.grid(cake_jit.MAX_ROWS - 1, "fwd_h4096") == (2**29 - 64, 1, 1)


@pytestmark_cpu
def test_a_record_that_takes_another_order_is_refused(monkeypatch) -> None:
    """``_entry`` checks the argument plan against the order the host passes, before it loads a library."""
    modules = {name: dict(record) for name, record in cake_jit.MODULES.items()}
    plan = list(modules["fwd_h4096"]["arg_plan"])
    plan[0], plan[1] = plan[1], plan[0]
    modules["fwd_h4096"]["arg_plan"] = plan
    monkeypatch.setattr(cake_jit, "MODULES", modules)
    pkg.parameters.cache_clear()
    pkg._entry.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="not in the order the host passes them"):
            pkg._entry("fwd_h4096", "sm_100a")
    finally:
        monkeypatch.undo()
        pkg.parameters.cache_clear()
        pkg._entry.cache_clear()


@pytestmark_cpu
def test_capabilities_map_to_the_two_architectures(monkeypatch) -> None:
    for capability, arch in ARCH_BY_CAPABILITY.items():
        assert pkg.arch_for_capability(capability) == arch
    for capability in ((9, 0), (10, 1), (10, 7), (11, 0), (12, 0), (12, 1)):
        assert pkg.arch_for_capability(capability) is None, capability
    assert tuple(pkg.SUPPORTED_COMPUTE_CAPABILITIES) == ((10, 0), (10, 3))
    assert pkg.is_cake_route_weight_scale_det_train_supported(torch.device("cpu")) is False
    clear = pkg._arch_of_index.cache_clear
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a"):
            pkg.device_arch(0)
    finally:
        monkeypatch.undo()
        clear()


def _cpu(rows: int = 4, width: int = 4096, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.zeros(rows, width, dtype=dtype)


# (case id, the call on CPU tensors, the exception, its message): every operand check runs before the device check.
OPERAND_CASES = [
    ("x not a tensor", lambda: forward([[0.0]], torch.zeros(1)), TypeError, "x must be a torch.Tensor"),
    ("x float32", lambda: forward(_cpu(dtype=torch.float32), torch.zeros(4)), ValueError, "x must be torch.bfloat16"),
    ("x 1-D", lambda: forward(torch.zeros(4096, dtype=torch.bfloat16), torch.zeros(1)), ValueError, "2-D"),
    ("x width 2048", lambda: forward(_cpu(width=2048), torch.zeros(4)), ValueError, r"H in \(4096, 3584\)"),
    ("x column stride", lambda: forward(_cpu(width=8192)[:, ::2], torch.zeros(4)), ValueError, "unit column stride"),
    (
        "x row stride",
        lambda: forward(torch.zeros(4, 4100, dtype=torch.bfloat16)[:, :4096], torch.zeros(4)),
        ValueError,
        "multiple of 8",
    ),
    (
        "x misaligned",
        lambda: forward(torch.zeros(4, 4104, dtype=torch.bfloat16)[:, 4:4100], torch.zeros(4)),
        ValueError,
        "16-byte boundary",
    ),
    ("w float64", lambda: forward(_cpu(), torch.zeros(4, dtype=torch.float64)), ValueError, "w must be torch.float32"),
    ("w rows", lambda: forward(_cpu(), torch.zeros(5)), ValueError, r"w must have shape \[4\]"),
    ("w strided", lambda: forward(_cpu(), torch.zeros(8)[::2]), ValueError, "w must be contiguous"),
    (
        "out width",
        lambda: forward(_cpu(), torch.zeros(4), out=_cpu(width=3584)),
        ValueError,
        "out must have H = 4096",
    ),
    (
        "out overlapping rows",
        lambda: forward(
            _cpu(), torch.zeros(4), out=torch.zeros(8192, dtype=torch.bfloat16).as_strided((4, 4096), (8, 1))
        ),
        ValueError,
        "must not overlap",
    ),
    ("dy rows", lambda: backward(_cpu(), _cpu(rows=5), torch.zeros(4)), ValueError, "dy must have 4 rows"),
    ("dy width", lambda: backward(_cpu(), _cpu(width=3584), torch.zeros(4)), ValueError, "dy must have H = 4096"),
    (
        "dw float16",
        lambda: backward(_cpu(), _cpu(), torch.zeros(4), dw=torch.zeros(4, dtype=torch.float16)),
        ValueError,
        "dw must be torch.float32",
    ),
    (
        "dx rows",
        lambda: backward(_cpu(), _cpu(), torch.zeros(4), dx=_cpu(rows=3)),
        ValueError,
        "dx must have 4 rows",
    ),
    ("CPU forward", lambda: forward(_cpu(), torch.zeros(4)), ValueError, "requires CUDA tensors"),
    ("CPU backward", lambda: backward(_cpu(), _cpu(), torch.zeros(4)), ValueError, "requires CUDA tensors"),
    ("CPU no rows", lambda: forward(_cpu(rows=0), torch.zeros(0)), ValueError, "requires CUDA tensors"),
]


@pytestmark_cpu
@pytest.mark.parametrize("case", OPERAND_CASES, ids=[case[0] for case in OPERAND_CASES])
def test_operands_are_checked_before_anything_runs(case, launches) -> None:
    _, call, error, message = case
    with pytest.raises(error, match=message):
        call()
    assert launches == []


@pytestmark_cpu
def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == cake_jit.PACKAGE
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{record['library']}_{arch}" for record in cake_jit.MODULES.values() for arch in record["arches"]}
    assert set(entries) == wanted
    assert len(entries) == len(RECORDS) * len(ARCHES)
    pattern = r"cake_route_weight_scale_det_train_[0-9a-f]{20}_sm_10[03]a"
    for name, entry in entries.items():
        assert re.fullmatch(pattern, name), name
        assert entry["arch"] in ARCHES and name.endswith("_" + entry["arch"]), name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


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


@pytestmark_cpu
def test_only_the_jit_shim_reads_environment_variables() -> None:
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
# On SM100 / SM103: the exact contract


def test_the_records_load(device) -> None:
    arch = pkg.device_arch(device.index)
    for name, record in cake_jit.MODULES.items():
        assert callable(getattr(pkg.load(name, arch), record["ffi_entry"])), name


@pytest.mark.parametrize("rows", ROW_COUNTS)
@pytest.mark.parametrize("width", WIDTHS)
def test_outputs_are_the_contract_bit_for_bit(width, rows, device) -> None:
    """``y`` and ``dx`` are the FP32 product rounded once and ``dw`` the stated order, bit for bit, with signed route
    weights (the kernels take any FP32 value)."""
    x, dy, w = make_inputs(rows, width, seed=rows + width, device=device, weight="signed")
    _expect_contract(x, dy, w, _step(x, dy, w))


@pytest.mark.parametrize("width", WIDTHS)
def test_dw_is_no_worse_than_the_fp32_row_sum_against_fp64(width, device) -> None:
    """On the issue's draws (65536 rows): ``dw``'s rel-L2 error against the FP64 reference is at most the error of the
    FP32 row sum ``(dy.float() * x.float()).sum(1)`` of the issue's CPU reference on the same inputs.  The same sum on
    the GPU is reported beside it.  The issue prints 8.85e-08 (H 4096) and 8.72e-08 (H 3584) for the CPU sum."""
    x_cpu, dy_cpu, w_cpu = _issue()[width]
    cpu_fp32 = fp32_row_sum(x_cpu, dy_cpu)
    x, dy, w = x_cpu.to(device), dy_cpu.to(device), w_cpu.to(device)
    dx, dw = backward(x, dy, w)
    reference = fp64_row_sum(x, dy)
    cake_error = rel_l2(dw, reference)
    cpu_error = rel_l2(cpu_fp32.to(device), reference)
    gpu_error = rel_l2(fp32_row_sum(x, dy), reference)
    report(
        f"dw rel-L2 vs FP64, rows 65536, H {width}: cake {cake_error:.3e}; FP32 row sum (issue reference, CPU) "
        f"{cpu_error:.3e}; FP32 row sum (torch, GPU) {gpu_error:.3e}"
    )
    assert same_bits(dw, tree_row_sum(x, dy))
    assert same_bits(dx, scaled(w, dy))
    assert cake_error <= cpu_error, (cake_error, cpu_error)


_ISSUE: dict = {}


def _issue() -> dict:
    """The issue's draws, made once per process."""
    if not _ISSUE:
        _ISSUE.update(issue_inputs())
    return _ISSUE


@pytest.mark.parametrize("width", WIDTHS)
def test_zero_weight_rows(width, device) -> None:
    """``w = +0`` and ``-0`` rows: ``y`` and ``dx`` are the signed zeros of the FP32 product, and ``dw`` is the row sum,
    the same bits as with any other weight (``dw`` never reads ``w``)."""
    rows = 1027
    x, dy, w = make_inputs(rows, width, seed=7, device=device, weight="signed")
    w[::3] = 0.0
    w[1::7] = -0.0
    found = _step(x, dy, w)
    _expect_contract(x, dy, w, found)
    zero = (w == 0)
    assert torch.all(found["y"][zero] == 0) and torch.all(found["dx"][zero] == 0)
    _, dw_ones = backward(x, dy, torch.ones_like(w))
    assert same_bits(found["dw"], dw_ones)


@pytest.mark.parametrize("width", WIDTHS)
def test_all_zero_x_rows(width, device) -> None:
    """All-zero ``x`` rows: ``y`` is zero there and ``dw`` is exactly 0 (a signed zero)."""
    rows = 515
    x, dy, w = make_inputs(rows, width, seed=8, device=device)
    x[::4] = 0.0
    found = _step(x, dy, w)
    _expect_contract(x, dy, w, found)
    assert torch.all(found["dw"][::4] == 0)
    assert torch.all(found["y"][::4] == 0)


@pytest.mark.parametrize("width", WIDTHS)
def test_no_rows_return_empty_outputs_and_launch_nothing(width, device, launches) -> None:
    x = torch.empty(0, width, dtype=torch.bfloat16, device=device)
    dy = torch.empty(0, width, dtype=torch.bfloat16, device=device)
    w = torch.empty(0, dtype=torch.float32, device=device)
    y = forward(x, w)
    dx, dw = backward(x, dy, w)
    assert y.shape == (0, width) and y.dtype == torch.bfloat16 and y.device == device
    assert dx.shape == (0, width) and dx.dtype == torch.bfloat16
    assert dw.shape == (0,) and dw.dtype == torch.float32
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    out = route_weight_scale(xa, wa)
    out.backward(torch.empty_like(out))
    assert xa.grad.shape == (0, width) and wa.grad.shape == (0,)
    # A row slice with no rows of a wider buffer, as a caller's last empty expert.
    buffer = torch.empty(4, width + 64, dtype=torch.bfloat16, device=device)
    forward(buffer[4:, :width], w)
    assert launches == []


@pytest.mark.parametrize("width", WIDTHS)
def test_one_row(width, device) -> None:
    """One row, also as a row of a buffer whose row stride is not a multiple of 8 (a single row's stride is not
    read)."""
    x, dy, w = make_inputs(1, width, seed=9, device=device, weight="signed")
    _expect_contract(x, dy, w, _step(x, dy, w))
    buffer = torch.zeros(3, width + 4, dtype=torch.bfloat16, device=device)  # row stride H + 4: not a multiple of 8
    row = buffer[0:1, :width]
    row.copy_(x)
    _expect_contract(row, dy, w, _step(row, dy, w))


def test_rows_past_two_to_the_31_elements(device) -> None:
    """1,048,576 rows at ``H`` 4096 (2^32 elements): ``y`` and ``dx`` bit for bit, ``dw`` the stated order bit for bit,
    compared in chunks of 65536 rows (the rows past 2^31 elements included)."""
    rows, width = MAX_ISSUE_ROWS, 4096
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    if free < LARGE_FREE_BYTES:
        pytest.skip(f"needs {LARGE_FREE_BYTES / 2**30:.0f} GiB free device memory, {free / 2**30:.1f} GiB free")
    g = torch.Generator(device=device).manual_seed(11)
    x = torch.randn(rows, width, device=device, dtype=torch.bfloat16, generator=g)
    w = torch.randn(rows, device=device, generator=g)
    chunk = 65536
    y = forward(x, w)
    for start in range(0, rows, chunk):
        part = slice(start, start + chunk)
        assert same_bits(y[part], scaled(w[part], x[part])), start
    del y
    dy = torch.randn(rows, width, device=device, dtype=torch.bfloat16, generator=g)
    dx, dw = backward(x, dy, w)
    for start in range(0, rows, chunk):
        part = slice(start, start + chunk)
        assert same_bits(dx[part], scaled(w[part], dy[part])), start
        assert same_bits(dw[part], tree_row_sum(x[part], dy[part])), start
    report(f"rows {rows} x H {width} ({rows * width} elements): y, dx and dw bit for bit")


@pytest.mark.parametrize("width", WIDTHS)
def test_row_slices_are_read_and_written_in_place(width, device) -> None:
    """Inputs as row slices of a wider and longer buffer (column offset 32, row stride ``H + 64``), outputs as row
    slices of preallocated buffers: the contract's bits, and the bytes around the slices untouched."""
    rows, offset = 1029, 3
    x0, dy0, w0 = make_inputs(rows, width, seed=12, device=device, weight="signed")
    wide = torch.full((rows + 8, width + 64), float("nan"), dtype=torch.bfloat16, device=device)
    x = wide[offset : offset + rows, 32 : 32 + width]
    x.copy_(x0)
    dy_wide = torch.full_like(wide, float("nan"))
    dy = dy_wide[offset : offset + rows, 16 : 16 + width]
    dy.copy_(dy0)
    w_long = torch.full((rows + 16,), float("nan"), device=device)
    w = w_long[5 : 5 + rows]
    w.copy_(w0)
    sentinel = float("inf")
    y_buffer = torch.full((rows + 8, width + 8), sentinel, dtype=torch.bfloat16, device=device)
    dx_buffer = torch.full((rows + 8, width), sentinel, dtype=torch.bfloat16, device=device)
    dw_buffer = torch.full((rows + 8,), sentinel, device=device)
    y = forward(x, w, out=y_buffer[2 : 2 + rows, 8 : 8 + width])
    dx, dw = backward(x, dy, w, dx=dx_buffer[4 : 4 + rows], dw=dw_buffer[1 : 1 + rows])
    assert y.data_ptr() == y_buffer[2, 8].data_ptr() and dx.data_ptr() == dx_buffer[4].data_ptr()
    assert dw.data_ptr() == dw_buffer[1].data_ptr()
    _expect_contract(x0, dy0, w0, {"y": y, "dx": dx, "dw": dw})
    for buffer, written in ((y_buffer, (slice(2, 2 + rows), slice(8, 8 + width))), (dx_buffer, (slice(4, 4 + rows),))):
        mask = torch.ones_like(buffer, dtype=torch.bool)
        mask[written] = False
        assert torch.all(buffer[mask] == sentinel)
    assert torch.all(dw_buffer[:1] == sentinel) and torch.all(dw_buffer[1 + rows :] == sentinel)


@pytest.mark.parametrize(
    "case",
    ["misaligned x", "row stride x", "column stride x", "overlapping out", "strided w", "strided dw", "dy device"],
)
def test_unfit_operands_are_refused_before_any_launch(case, device, launches) -> None:
    """Each unfit operand (built before the call) raises ``ValueError``; the call allocates and launches nothing."""
    rows, width = 64, 4096
    x, dy, w = make_inputs(rows, width, seed=13, device=device)
    wide = torch.zeros(rows, width + 64, dtype=torch.bfloat16, device=device)
    odd = torch.zeros(rows, width + 4, dtype=torch.bfloat16, device=device)
    double = torch.zeros(rows, 2 * width, dtype=torch.bfloat16, device=device)
    flat = torch.zeros(rows * 8 + width, dtype=torch.bfloat16, device=device)
    pairs = torch.zeros(2 * rows, device=device)
    dy_cpu = dy.cpu()
    calls = {
        "misaligned x": lambda: forward(wide[:, 4 : 4 + width], w),
        "row stride x": lambda: forward(odd[:, :width], w),
        "column stride x": lambda: forward(double[:, ::2], w),
        "overlapping out": lambda: forward(x, w, out=flat.as_strided((rows, width), (8, 1))),
        "strided w": lambda: forward(x, pairs[::2]),
        "strided dw": lambda: backward(x, dy, w, dw=pairs[::2]),
        "dy device": lambda: backward(x, dy_cpu, w),
    }
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    with pytest.raises(ValueError):
        calls[case]()
    assert launches == []
    assert torch.cuda.memory_allocated(device) == before


def test_an_unsupported_device_is_refused_before_any_allocation(device, monkeypatch, launches) -> None:
    x, dy, w = make_inputs(8, 4096, seed=14, device=device)
    monkeypatch.setattr(pkg, "_arch_of_index", lambda index: None)
    for call in (lambda: forward(x, w), lambda: backward(x, dy, w), lambda: forward(x[:0], w[:0])):
        before = torch.cuda.memory_allocated(device)
        with pytest.raises(RuntimeError, match="not an exported"):
            call()
        assert torch.cuda.memory_allocated(device) == before
    assert launches == []


def test_one_library_per_record_serves_every_row_count(device, monkeypatch) -> None:
    """After the first call of each record, no call at another row count loads or builds a library, and each row's
    outputs do not depend on the other rows (the first rows of a longer call give the same bits).  The row counts
    include both sides of the small-row bound, where the H 4096 forward changes programs (one row a CTA of 32 threads
    below it, 4 rows a CTA of 128 threads from it on): there ``y`` is also the contract's bits."""
    for width in WIDTHS:
        for rows in (4, SMALL_ROWS_BOUND):  # every record: both sides of the small-row bound
            x, dy, w = make_inputs(rows, width, seed=15, device=device)
            _step(x, dy, w)
    loaded = pkg.load.cache_info().currsize
    cake_build = cake_package("cake_route_weight_scale_det_train", "cake_build")

    def refuse(name):
        raise AssertionError(f"library {name} loaded after the first call")

    monkeypatch.setattr(cake_build, "load_library", refuse)
    bound = (SMALL_ROWS_BOUND - 1, SMALL_ROWS_BOUND, SMALL_ROWS_BOUND + 1)
    counts = sorted({*ROW_COUNTS, *bound})
    for width in WIDTHS:
        x, dy, w = make_inputs(max(counts), width, seed=16, device=device, weight="signed")
        full = _step(x, dy, w)
        for rows in counts:
            part = _step(x[:rows], dy[:rows], w[:rows])
            for key in ("y", "dx", "dw"):
                assert same_bits(part[key], full[key][:rows]), (width, rows, key)
            if rows in bound:
                assert same_bits(part["y"], scaled(w[:rows], x[:rows])), (width, rows)
    assert pkg.load.cache_info().currsize == loaded


@pytest.mark.parametrize("width", WIDTHS)
def test_outputs_repeat_bit_for_bit(width, device) -> None:
    """Repeated calls, a side stream and a CUDA graph replay give the same bits."""
    x, dy, w = make_inputs(4099, width, seed=17, device=device, weight="signed")
    first = _step(x, dy, w)
    for _ in range(3):
        again = _step(x, dy, w)
        assert all(same_bits(again[key], first[key]) for key in first)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        on_side = _step(x, dy, w)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert all(same_bits(on_side[key], first[key]) for key in first)
    y = torch.full_like(first["y"], float("nan"))
    dx = torch.full_like(first["dx"], float("nan"))
    dw = torch.full_like(first["dw"], float("nan"))
    graph = torch.cuda.CUDAGraph()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward(x, w, out=y)
        backward(x, dy, w, dx=dx, dw=dw)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        forward(x, w, out=y)
        backward(x, dy, w, dx=dx, dw=dw)
    for buffer in (y, dx, dw):
        buffer.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert same_bits(y, first["y"]) and same_bits(dx, first["dx"]) and same_bits(dw, first["dw"])


def test_another_current_device(device) -> None:
    """Each call runs on its operands' device, whatever the current device."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", 1 if device.index == 0 else 0)
    if not pkg.is_cake_route_weight_scale_det_train_supported(other):
        pytest.skip(f"{other} is not an exported architecture")
    x, dy, w = make_inputs(1025, 4096, seed=18, device=other, weight="signed")
    with torch.cuda.device(device):
        found = _step(x, dy, w)
    torch.cuda.synchronize(other)
    assert found["y"].device == other
    _expect_contract(x, dy, w, found)


def test_no_host_synchronization(device) -> None:
    """Under ``torch.cuda.set_sync_debug_mode("error")`` the steps and an autograd step (forward and backward) run:
    nothing synchronizes the host."""
    x, dy, w = make_inputs(4097, 4096, seed=19, device=device)
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    for _ in range(2):  # warm-up: libraries, allocator
        _step(x, dy, w)
        route_weight_scale(xa, wa).backward(dy)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        found = _step(x, dy, w)
        route_weight_scale(xa, wa).backward(dy)
        found_rows = _step(x[:1000], dy[:1000], w[:1000])
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    _expect_contract(x, dy, w, found)
    assert same_bits(found_rows["dw"], found["dw"][:1000])


@pytest.mark.parametrize("width", WIDTHS)
def test_autograd_matches_the_steps_and_saves_only_the_inputs(width, device) -> None:
    x, dy, w = make_inputs(2053, width, seed=20, device=device, weight="signed")
    steps = _step(x, dy, w)
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = route_weight_scale(xa, wa)
    saved = y.grad_fn.saved_tensors
    assert len(saved) == 2 and saved[0] is xa and saved[1] is wa
    y.backward(dy)
    assert same_bits(y.detach(), steps["y"])
    assert same_bits(xa.grad, steps["dx"]) and same_bits(wa.grad, steps["dw"])
    # Only one input needs a gradient.
    xb = x.clone().requires_grad_()
    dx_only = torch.autograd.grad(route_weight_scale(xb, w), xb, dy)[0]
    assert same_bits(dx_only, steps["dx"])
    wb = w.clone().requires_grad_()
    dw_only = torch.autograd.grad(route_weight_scale(x, wb), wb, dy)[0]
    assert same_bits(dw_only, steps["dw"])


@pytest.mark.parametrize("width", WIDTHS)
def test_checkpoint_gives_the_plain_gradients(width, device, launches) -> None:
    """``torch.utils.checkpoint(..., use_reentrant=False)``: the gradients of the plain call bit for bit; the
    recomputation launches the forward once and the backward launches once."""
    x, dy, w = make_inputs(1031, width, seed=21, device=device, weight="signed")
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    route_weight_scale(xa, wa).backward(dy)
    xc, wc = x.clone().requires_grad_(), w.clone().requires_grad_()
    launches.clear()
    y = torch.utils.checkpoint.checkpoint(route_weight_scale, xc, wc, use_reentrant=False)
    y.backward(dy)
    assert same_bits(xc.grad, xa.grad) and same_bits(wc.grad, wa.grad)
    assert [stage for stage, _, _ in launches] == ["fwd", "fwd", "bwd"]


def test_an_expanded_upstream_gradient_is_read(device) -> None:
    """``y.sum().backward()`` hands over an expanded gradient (strides 0, ones): it is copied once and gives the
    gradients of a contiguous one."""
    x, _, w = make_inputs(129, 4096, seed=22, device=device, weight="signed")
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    route_weight_scale(xa, wa).sum().backward()
    dx, dw = backward(x, torch.ones_like(x), w)
    assert same_bits(xa.grad, dx) and same_bits(wa.grad, dw)


def test_no_double_backward(device) -> None:
    x, dy, w = make_inputs(16, 4096, seed=23, device=device)
    xa, wa = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = route_weight_scale(xa, wa)
    dx, dw = torch.autograd.grad(y, (xa, wa), dy, create_graph=True)
    steps = _step(x, dy, w)
    assert same_bits(dx.detach(), steps["dx"]) and same_bits(dw.detach(), steps["dw"])
    with pytest.raises(RuntimeError):
        dw.sum().backward()


@pytest.mark.parametrize("width", WIDTHS)
def test_peak_extra_memory_is_o_rows(width, device) -> None:
    """Peak device memory a call allocates beyond its outputs: 0 up to the caching allocator's rounding (at most 2 MiB
    a call here), so ``O(rows)``; reported beside the eager PyTorch formula's at 65536 rows."""
    rows = 65536
    x, dy, w = make_inputs(rows, width, seed=24, device=device)
    _step(x, dy, w)  # warm-up

    def peak(call) -> tuple[int, int]:
        torch.cuda.synchronize()
        start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        outputs = call()
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated(device) - start
        extra = torch.cuda.max_memory_allocated(device) - start - kept
        del outputs
        return extra, kept

    fwd_extra, fwd_kept = peak(lambda: forward(x, w))
    bwd_extra, bwd_kept = peak(lambda: backward(x, dy, w))
    eager_fwd, _ = peak(lambda: scaled(w, x))
    eager_bwd, _ = peak(lambda: (scaled(w, dy), fp32_row_sum(x, dy)))
    report(
        f"peak extra device memory beyond the outputs, rows {rows}, H {width}: cake forward {fwd_extra} B "
        f"(outputs {fwd_kept} B), backward {bwd_extra} B (outputs {bwd_kept} B); eager PyTorch forward {eager_fwd} B, "
        f"backward {eager_bwd} B"
    )
    slack = 2 << 20
    assert fwd_extra <= slack and bwd_extra <= slack, (fwd_extra, bwd_extra)
    assert fwd_kept <= rows * width * 2 + slack and bwd_kept <= rows * width * 2 + rows * 4 + slack


@pytestmark_cpu
def test_bits_are_raw_integer_views() -> None:
    """The bit helper reads ``+0`` and ``-0`` as different values (the zero-weight test relies on it)."""
    a = torch.tensor([0.0, -0.0], dtype=torch.bfloat16)
    assert not torch.equal(bits(a[:1]), bits(a[1:]))
