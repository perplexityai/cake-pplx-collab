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

Public-API suite of ``cake_pplx.cake_engram_lookup_det_train``, the Engram row lookup of the Cake deterministic
training package: ``cake_engram_lookup_det_train_forward(weight, scale, row_ids) -> out`` and
``cake_engram_lookup_det_train_split_packed(buf) -> (weight, scale)``.

Without a device: both entries are stable APIs (no ``ExperimentalWarning``) and the module's only public names,
importing the module imports no backend, the module reads no environment variable, the entries take the backend's
parameters, the lookup documents its behaviour under ``torch.use_deterministic_algorithms``, a ``deterministic``
keyword is a ``TypeError`` before any work, operands on the CPU are a ``ValueError``, and the packed-table views share
the buffer's storage and row stride (every malformed buffer a ``ValueError``).

With any CUDA device (the checks run before the architecture is looked up): every documented ``ValueError`` (dtype,
device, rank, shape, column and row strides, alignment, ``row_ids`` layout, the row-count limit, an empty table with
ids); a device outside the exported architectures is a ``RuntimeError`` and an empty lookup there still returns an
empty output; with two devices, operands on two devices are a ``ValueError``.

On SM100 / SM103 / SM107, every comparison bit for bit against the FP64 reference of the feature request: the lookup
of ``[T, 24]`` and ``[M]`` ids in both layouts (the packed one through the public views), operands on an 8-byte (not
16-byte) boundary; the launch runs on the current stream without synchronizing (behind a sleep on a side stream,
under ``torch.cuda.set_sync_debug_mode("error")``); a call captured in a CUDA graph replays to the eager bits and reads
the table and ids it is replayed on; under ``torch.utils.checkpoint(..., use_reentrant=False)`` the lookup runs twice
(once without) and a BF16 linear layer after it has the same outputs and weight gradients, under
``torch.use_deterministic_algorithms(True)`` with ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, in a fresh process; an id outside
the table (``R``, ``-1``, ``2**62``; first, middle and tail-step positions) raises, in a fresh process, and ``R - 1``
does not; the call
allocates exactly its output (the workspace is 0 bytes); under ``torch.use_deterministic_algorithms(True)``, strict or
``warn_only``, no alert, no warning and the same bits (the output allocated in the mode is NaN-filled first, so every
element is written); with two devices, operands on the device that is not current give the reference's bits there.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import torch

import cake_pplx.cake_engram_lookup_det_train as api
from tests.cake.test_helpers.cake_engram_lookup_ref import (
    LAYOUTS,
    N_COLS,
    PACKED_ROW_BYTES,
    ROW_DIM,
    assert_same_bits,
    make_table,
    oracle,
    random_bytes,
    random_ids,
)

BACKEND = "cake_pplx._backends.cake_engram_lookup_det_train"
ENTRY = "cake_engram_lookup_det_train_forward"
SPLIT = "cake_engram_lookup_det_train_split_packed"
ENTRIES = (ENTRY, SPLIT)
CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# Cycles of the sleep kernel queued before a call on a side stream (about 0.1 s): the call returns long before it ends.
SLEEP_CYCLES = 200_000_000


def _backend():
    from cake_pplx._backends.cake_engram_lookup_det_train import cake_backend

    return cake_backend


def _cuda_or_skip() -> torch.device:
    """A CUDA device of any architecture: the operand checks run before the architecture is looked up."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda:0")


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in CAPABILITIES:
        pytest.skip("the Cake Engram row lookup ships for SM100 / SM103 / SM107")
    from cake_pplx._backends.cake_engram_lookup_det_train import cake_jit

    if not (Path(cake_jit.__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    _build_libraries(0)
    return torch.device("cuda:0")


@functools.cache
def _build_libraries(index: int) -> None:
    """Build the device's library once, before the first call."""
    backend = _backend()
    for spec in backend.specs_for_arch(backend.device_arch(index)):
        spec.build_and_load()


@contextlib.contextmanager
def _torch_deterministic(*, warn_only: bool):
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _lookup(table, row_ids: torch.Tensor) -> torch.Tensor:
    return api.cake_engram_lookup_det_train_forward(table.weight, table.scale, row_ids)


def _public_table(weight_bytes: torch.Tensor, scale_bytes: torch.Tensor, layout: str, device):
    """The table in ``layout``; a packed table's views come from the public split of its ``uint8 [R, 264]`` buffer."""
    table = make_table(weight_bytes, scale_bytes, layout, device)
    if layout == "packed":
        table.weight, table.scale = api.cake_engram_lookup_det_train_split_packed(table.packed)
    return table


def _operands(device, *, rows: int = 16, shape: tuple[int, ...] = (4, N_COLS)) -> dict:
    """Well-formed operands of zeros (a call that should fail never reads them)."""
    return {
        "weight": torch.zeros(rows, ROW_DIM, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn),
        "scale": torch.zeros(rows, 8, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu),
        "row_ids": torch.zeros(shape, dtype=torch.int64, device=device),
    }


def _call(ops: dict, **kwargs):
    return api.cake_engram_lookup_det_train_forward(ops["weight"], ops["scale"], ops["row_ids"], **kwargs)


def _at_offset(rows: int, width: int, offset: int, dtype: torch.dtype, device) -> torch.Tensor:
    """A ``[rows, width]`` tensor of ``dtype`` (one byte per element) with a row stride of ``width``, starting
    ``offset`` bytes past the start of its allocation (the caching allocator aligns an allocation to 512 bytes)."""
    flat = torch.zeros(rows * width + 64, dtype=torch.uint8, device=device)
    view = flat[offset : offset + rows * width].view(rows, width).view(dtype)
    assert view.data_ptr() % 512 == offset
    return view


def _strided_bytes(rows: int, width: int, row_stride: int, column_stride: int, dtype: torch.dtype, device):
    """A ``[rows, width]`` view of ``dtype`` with the given element strides."""
    base = torch.zeros(rows, row_stride, dtype=torch.uint8, device=device)
    view = base[:, : width * column_stride : column_stride].view(dtype)
    assert view.stride() == (row_stride, column_stride) and tuple(view.shape) == (rows, width)
    return view


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_the_entries_are_a_stable_api() -> None:
    assert list(api.__all__) == list(ENTRIES)
    for name in ENTRIES:
        entry = getattr(api, name)
        assert getattr(entry, "is_experimental", False) is False, name
        assert entry.__name__ == name


def test_importing_the_module_imports_no_backend() -> None:
    code = (
        "import sys\n"
        "import cake_pplx.cake_engram_lookup_det_train as api\n"
        "names = dir(api)\n"
        f"print(any(m.startswith({BACKEND!r}) for m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_the_module_reads_no_environment_variable() -> None:
    tree = ast.parse(Path(api.__file__).read_text())
    hits = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")
    ]
    assert not hits, hits


def test_parameters_match_the_backend() -> None:
    wanted = {ENTRY: ["weight", "scale", "row_ids"], SPLIT: ["buf"]}
    for name, names in wanted.items():
        ours = inspect.signature(inspect.unwrap(getattr(api, name))).parameters
        theirs = inspect.signature(getattr(_backend(), name)).parameters
        assert list(ours) == list(theirs) == names, name
        for key, param in ours.items():
            assert param.kind == theirs[key].kind == param.POSITIONAL_OR_KEYWORD, (name, key)
            assert param.default is theirs[key].default is param.empty, (name, key)


def test_the_entry_documents_its_determinism() -> None:
    """The lookup states its behaviour under ``torch.use_deterministic_algorithms``, its workspace and its stream."""
    doc = getattr(api, ENTRY).__doc__
    for phrase in ("use_deterministic_algorithms", "Workspace", "current stream", "use_reentrant=False"):
        assert phrase in doc, phrase


def test_a_deterministic_keyword_is_a_type_error() -> None:
    ops = _operands("cpu")
    for value in (True, False):
        with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
            _call(ops, deterministic=value)


def test_cpu_operands_are_refused() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        _call(_operands("cpu"))


def test_the_row_count_limit_is_2_31_minus_8() -> None:
    assert _backend()._MAX_POSITIONS == 2**31 - 8


@pytest.mark.parametrize("contiguous", (True, False), ids=("contiguous", "row_strided"))
def test_split_packed_views_share_the_buffer(contiguous) -> None:
    """The views are ``buf[:, :256]`` as E4M3 and ``buf[:, 256:]`` as E8M0: the same storage, the buffer's row
    stride (264 for a contiguous buffer, 528 for every other row of one), nothing copied; a write into the buffer shows
    through them."""
    rows = 6
    base = torch.arange(rows * 2 * PACKED_ROW_BYTES, dtype=torch.int64).remainder(251).to(torch.uint8)
    base = base.view(rows * 2, PACKED_ROW_BYTES)
    buf = base[:rows] if contiguous else base[::2]
    weight, scale = api.cake_engram_lookup_det_train_split_packed(buf)
    assert (weight.dtype, tuple(weight.shape)) == (torch.float8_e4m3fn, (rows, ROW_DIM))
    assert (scale.dtype, tuple(scale.shape)) == (torch.float8_e8m0fnu, (rows, 8))
    assert weight.stride() == scale.stride() == (buf.stride(0), 1)
    assert weight.data_ptr() == buf.data_ptr() and scale.data_ptr() == buf.data_ptr() + ROW_DIM
    assert torch.equal(weight.view(torch.uint8), buf[:, :ROW_DIM])
    assert torch.equal(scale.view(torch.uint8), buf[:, ROW_DIM:])
    buf[2, 7] = 0x38
    buf[2, ROW_DIM + 1] = 0x81
    assert int(weight.view(torch.uint8)[2, 7]) == 0x38 and int(scale.view(torch.uint8)[2, 1]) == 0x81


@pytest.mark.parametrize(
    "buf",
    (
        pytest.param(lambda: torch.zeros(4, PACKED_ROW_BYTES, dtype=torch.int8), id="dtype-int8"),
        pytest.param(
            lambda: torch.zeros(4, PACKED_ROW_BYTES, dtype=torch.uint8).view(torch.float8_e4m3fn), id="dtype-e4m3"
        ),
        pytest.param(lambda: torch.zeros(4, PACKED_ROW_BYTES - 1, dtype=torch.uint8), id="width-263"),
        pytest.param(lambda: torch.zeros(4, ROW_DIM, dtype=torch.uint8), id="width-256"),
        pytest.param(lambda: torch.zeros(4 * PACKED_ROW_BYTES, dtype=torch.uint8), id="rank-1"),
        pytest.param(lambda: torch.zeros(2, 4, PACKED_ROW_BYTES, dtype=torch.uint8), id="rank-3"),
        pytest.param(lambda: torch.zeros(4, 2 * PACKED_ROW_BYTES, dtype=torch.uint8)[:, ::2], id="column-stride-2"),
        pytest.param(lambda: [[0] * PACKED_ROW_BYTES], id="not-a-tensor"),
    ),
)
def test_split_packed_refuses_a_malformed_buffer(buf) -> None:
    with pytest.raises(ValueError):
        api.cake_engram_lookup_det_train_split_packed(buf())


# ----------------------------------------------------------------------------------------------------------------
# Any CUDA device: refused calls


def _set(name, make):
    def edit(device):
        ops = _operands(device)
        ops[name] = make(ops[name], device)
        return ops

    return edit


VALUE_ERRORS = [
    pytest.param(_set("weight", lambda t, d: t.view(torch.uint8)), id="dtype-weight-uint8"),
    pytest.param(_set("weight", lambda t, d: t.view(torch.float8_e5m2)), id="dtype-weight-e5m2"),
    pytest.param(
        _set("weight", lambda t, d: torch.zeros(t.shape, dtype=torch.bfloat16, device=d)), id="dtype-weight-bf16"
    ),
    pytest.param(_set("scale", lambda t, d: t.view(torch.uint8)), id="dtype-scale-uint8"),
    pytest.param(_set("scale", lambda t, d: t.view(torch.float8_e4m3fn)), id="dtype-scale-e4m3"),
    pytest.param(_set("row_ids", lambda t, d: t.to(torch.int32)), id="dtype-row_ids-int32"),
    pytest.param(_set("row_ids", lambda t, d: t.float()), id="dtype-row_ids-fp32"),
    pytest.param(_set("weight", lambda t, d: t.cpu()), id="device-weight"),
    pytest.param(_set("scale", lambda t, d: t.cpu()), id="device-scale"),
    pytest.param(_set("row_ids", lambda t, d: t.cpu()), id="device-row_ids"),
    pytest.param(_set("weight", lambda t, d: t.reshape(-1)), id="rank-weight-1"),
    pytest.param(_set("weight", lambda t, d: t.unsqueeze(0)), id="rank-weight-3"),
    pytest.param(_set("scale", lambda t, d: t.reshape(-1)), id="rank-scale-1"),
    pytest.param(_set("weight", lambda t, d: t[:, :128]), id="shape-weight-128"),
    pytest.param(_set("scale", lambda t, d: t[:, :4]), id="shape-scale-4"),
    pytest.param(_set("scale", lambda t, d: t[:-1]), id="shape-scale-rows"),
    pytest.param(_set("weight", lambda t, d: t[:-1]), id="shape-weight-rows"),
    pytest.param(
        _set("weight", lambda t, d: _strided_bytes(t.shape[0], ROW_DIM, 2 * ROW_DIM, 2, t.dtype, d)),
        id="stride-weight-columns",
    ),
    pytest.param(
        _set("weight", lambda t, d: _strided_bytes(t.shape[0], ROW_DIM, ROW_DIM + 4, 1, t.dtype, d)),
        id="stride-weight-rows-260",
    ),
    pytest.param(
        _set("scale", lambda t, d: _strided_bytes(t.shape[0], 8, 16, 2, t.dtype, d)), id="stride-scale-columns"
    ),
    pytest.param(
        _set("scale", lambda t, d: _strided_bytes(t.shape[0], 8, 12, 1, t.dtype, d)), id="stride-scale-rows-12"
    ),
    pytest.param(_set("row_ids", lambda t, d: t.t()), id="layout-row_ids-transposed"),
    pytest.param(
        _set("row_ids", lambda t, d: torch.zeros(4, 2 * N_COLS, dtype=torch.int64, device=d)[:, ::2]),
        id="layout-row_ids-strided",
    ),
    pytest.param(_set("row_ids", lambda t, d: t.tolist()), id="not-a-tensor-row_ids"),
    pytest.param(_set("weight", lambda t, d: t.view(torch.uint8).cpu().numpy()), id="not-a-tensor-weight"),
]


@pytest.mark.parametrize("edit", VALUE_ERRORS)
def test_malformed_operands_are_a_value_error(edit) -> None:
    device = _cuda_or_skip()
    with pytest.raises(ValueError):
        _call(edit(device))


@pytest.mark.parametrize("offset", (1, 4), ids=("1-byte", "4-bytes"))
@pytest.mark.parametrize("name", ("weight", "scale"))
def test_a_base_off_an_8_byte_boundary_is_a_value_error(name, offset) -> None:
    device = _cuda_or_skip()
    ops = _operands(device)
    width = ROW_DIM if name == "weight" else 8
    ops[name] = _at_offset(ops[name].shape[0], width, offset, ops[name].dtype, device)
    with pytest.raises(ValueError, match="8-byte boundary"):
        _call(ops)


def test_the_row_count_limit_is_a_value_error(monkeypatch) -> None:
    """``row_ids`` with ``2^31 - 8`` or more elements is a ``ValueError`` (the limit lowered to 96 here, so the test
    needs no 16 GB id tensor; the limit itself is checked without a device)."""
    device = _cuda_or_skip()
    monkeypatch.setattr(_backend(), "_MAX_POSITIONS", 96)
    ops = _operands(device, shape=(4, N_COLS))
    with pytest.raises(ValueError, match="fewer than"):
        _call(ops)


def test_ids_into_an_empty_table_are_a_value_error() -> None:
    device = _cuda_or_skip()
    with pytest.raises(ValueError, match="no rows"):
        _call(_operands(device, rows=0))


@pytest.mark.parametrize("capability", ((12, 0), (9, 0), (11, 0), (10, 1)), ids=lambda c: f"cc{c[0]}{c[1]}")
def test_a_device_outside_the_architectures_is_refused(capability, monkeypatch) -> None:
    """A device of another capability is a ``RuntimeError`` naming the exported architectures; an empty lookup there
    returns an empty output (it is not refused: nothing runs)."""
    device = _cuda_or_skip()
    ops = _operands(device)
    device_arch = _backend().device_arch
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    device_arch.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
            _call(ops)
        empty = api.cake_engram_lookup_det_train_forward(
            ops["weight"], ops["scale"], torch.empty(0, N_COLS, dtype=torch.int64, device=device)
        )
        assert (tuple(empty.shape), empty.dtype) == ((0, N_COLS, ROW_DIM), torch.bfloat16)
    finally:
        monkeypatch.undo()
        device_arch.cache_clear()


def test_operands_on_two_devices_are_a_value_error() -> None:
    _cuda_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")
    for name in ("scale", "row_ids"):
        ops = _operands(torch.device("cuda:0"))
        ops[name] = ops[name].to("cuda:1")
        with pytest.raises(ValueError, match="device"):
            _call(ops)


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: results


@pytest.mark.parametrize("shape", ((16, N_COLS), (1001,)), ids=("T24", "M"))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_entry_matches_the_reference(layout, shape) -> None:
    """The lookup through the public entry (a packed table through the public views): the output's shape, dtype and
    layout, and the reference's bits; no warning."""
    device = _device_or_skip()
    table = _public_table(*random_bytes(777, seed=101, nan_every=19), layout, device)
    ids = random_ids(777, shape, seed=102, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _lookup(table, ids)
    assert (tuple(out.shape), out.dtype, out.device, out.is_contiguous()) == (
        (*shape, ROW_DIM),
        torch.bfloat16,
        device,
        True,
    )
    assert out.requires_grad is False
    assert_same_bits(out, oracle(table, ids), f"entry {layout} {shape}")


def test_operands_on_an_8_byte_boundary_match_the_reference() -> None:
    """``weight`` and ``scale`` starting 8 bytes past a 16-byte boundary (as the odd rows of a packed table do):
    accepted, and the reference's bits."""
    device = _device_or_skip()
    rows = 300
    weight_bytes, scale_bytes = random_bytes(rows, seed=111)
    weight = _at_offset(rows, ROW_DIM, 8, torch.float8_e4m3fn, device)
    scale = _at_offset(rows, 8, 8, torch.float8_e8m0fnu, device)
    weight.view(torch.uint8).copy_(weight_bytes.to(device))
    scale.view(torch.uint8).copy_(scale_bytes.to(device))
    ids = random_ids(rows, (5, N_COLS), seed=112, device=device)
    out = api.cake_engram_lookup_det_train_forward(weight, scale, ids)
    assert_same_bits(out, oracle(make_table(weight_bytes, scale_bytes, "contiguous", device), ids), "8-byte boundary")


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: streams, graphs, checkpointing, memory, deterministic mode and devices


def test_runs_on_the_current_stream_without_synchronizing() -> None:
    """On a side stream, behind a long sleep kernel that precedes the copy of the table's bytes into a buffer of NaN
    codes: the call returns while the sleep still runs (no synchronization; under
    ``torch.cuda.set_sync_debug_mode("error")`` a synchronizing torch call would raise), and its output has the bits
    of the copied table (the launch read the table after the copy, on the current stream)."""
    device = _device_or_skip()
    rows = 1024
    table = _public_table(*random_bytes(rows, seed=121), "packed", device)
    ids = random_ids(rows, (64, N_COLS), seed=122, device=device)
    want = _lookup(table, ids)  # loads the library
    nan_weight = torch.full((rows, ROW_DIM), 0x7F, dtype=torch.uint8)
    nan_scale = torch.full((rows, 8), 0xFF, dtype=torch.uint8)
    poisoned = make_table(nan_weight, nan_scale, "packed", device)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        poisoned.packed.copy_(table.packed)
        torch.cuda.set_sync_debug_mode("error")
        try:
            got = _lookup(poisoned, ids)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        pending = not side.query()
    side.synchronize()
    assert pending, "the call synchronized: the sleep kernel queued before it had finished"
    assert torch.equal(_bits(got), _bits(want))


def test_a_call_captured_in_a_cuda_graph_replays() -> None:
    """A call captured in a CUDA graph: after its output is filled with NaN, the replay writes the eager bits; after
    new table bytes and new ids are copied into the captured operands, the replay gives the lookup of those."""
    device = _device_or_skip()
    rows = 2048
    table = _public_table(*random_bytes(rows, seed=131, nan_every=7), "packed", device)
    ids = random_ids(rows, (128, N_COLS), seed=132, device=device)
    eager = _lookup(table, ids)  # loads the library before the capture
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        _lookup(table, ids)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _lookup(table, ids)
    _bits(captured).fill_(-1)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(_bits(captured), _bits(eager))
    weight_bytes, scale_bytes = random_bytes(rows, seed=133)
    table.packed[:, :ROW_DIM].copy_(weight_bytes.to(device))
    table.packed[:, ROW_DIM:].copy_(scale_bytes.to(device))
    ids.copy_(random_ids(rows, (128, N_COLS), seed=134, device=device))
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(_bits(captured), _bits(_lookup(table, ids)))
    assert_same_bits(captured, oracle(table, ids), "replayed on new bytes")


# The checkpoint run, in a fresh process (``CUBLAS_WORKSPACE_CONFIG`` takes effect before cuBLAS starts): the lookup
# of ``[64, 24]`` ids, concatenated to width 6144, feeds a BF16 linear layer; the step runs once plainly and once
# under ``torch.utils.checkpoint(..., use_reentrant=False)``, each under ``torch.use_deterministic_algorithms(True)``,
# counting the lookup's launches.  argv: the backend module, the facade module.
CHECKPOINT_SCRIPT = r"""
import importlib, json, sys
import torch
import torch.utils.checkpoint

backend = importlib.import_module(sys.argv[1])
facade = importlib.import_module(sys.argv[2])
torch.use_deterministic_algorithms(True)
device = torch.device("cuda:0")
gen = torch.Generator().manual_seed(8379)
rows, tokens, width, features = 4096, 64, 24 * 256, 128
weight_bytes = torch.randint(0, 256, (rows, 256), generator=gen, dtype=torch.uint8)
weight_bytes[(weight_bytes & 0x7F) == 0x7F] -= 1  # no NaN code: finite gradients
scale_bytes = torch.randint(120, 134, (rows, 8), generator=gen, dtype=torch.uint8)
weight = weight_bytes.to(device).view(torch.float8_e4m3fn)
scale = scale_bytes.to(device).view(torch.float8_e8m0fnu)
ids = torch.randint(0, rows, (tokens, 24), generator=gen).to(device)
proj_init = (torch.randn(features, width, generator=gen) / 64).to(torch.bfloat16).to(device)
grad_out = torch.randn(tokens, features, generator=gen).to(torch.bfloat16).to(device)
launches = [0]
launch = backend._launch


def counted(*args):
    launches[0] += 1
    return launch(*args)


backend._launch = counted


def step(checkpointed):
    proj = proj_init.clone().requires_grad_(True)

    def layer(p):
        found = facade.cake_engram_lookup_det_train_forward(weight, scale, ids)
        return torch.nn.functional.linear(found.flatten(-2), p)

    launches[0] = 0
    if checkpointed:
        y = torch.utils.checkpoint.checkpoint(layer, proj, use_reentrant=False)
    else:
        y = layer(proj)
    y.backward(grad_out)
    torch.cuda.synchronize(device)
    return launches[0], y.detach(), proj.grad


plain, ckpt = step(False), step(True)
print(json.dumps({
    "launches": [plain[0], ckpt[0]],
    "out_equal": torch.equal(plain[1].view(torch.int16), ckpt[1].view(torch.int16)),
    "grad_equal": torch.equal(plain[2].view(torch.int16), ckpt[2].view(torch.int16)),
    "grad_nonzero": bool((plain[2] != 0).any()),
    "grad_finite": bool(plain[2].isfinite().all()),
}))
"""


def test_checkpoint_non_reentrant_repeats_the_lookup_bitwise() -> None:
    """Under ``torch.utils.checkpoint(..., use_reentrant=False)`` the lookup runs twice (the forward and the recompute
    of the backward; once without the checkpoint), and the linear layer after it has the same output and the same
    weight gradient bits as without the checkpoint."""
    _device_or_skip()
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    result = subprocess.run(
        [sys.executable, "-c", CHECKPOINT_SCRIPT, _backend().__name__, api.__name__],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {
        "launches": [1, 2],
        "out_equal": True,
        "grad_equal": True,
        "grad_nonzero": True,
        "grad_finite": True,
    }, report


# An id outside the table, in a fresh process (the trap leaves the process's CUDA context unusable): the lookup of
# 1001 ids into a 64-row table with the id at position ``at`` replaced by ``bad`` (position 1000 sits in the tail
# step), then a synchronize.  argv: the facade module, the id, its position.
TRAP_SCRIPT = r"""
import importlib, json, sys
import torch

facade = importlib.import_module(sys.argv[1])
bad, at = int(sys.argv[2]), int(sys.argv[3])
device = torch.device("cuda:0")
rows = 64
weight = torch.zeros(rows, 256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
scale = torch.full((rows, 8), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
ids = torch.arange(1001, device=device) % rows
ids[at] = bad
try:
    facade.cake_engram_lookup_det_train_forward(weight, scale, ids)
    torch.cuda.synchronize(device)
except Exception as error:  # noqa: BLE001
    print(json.dumps({"raised": True, "error": f"{type(error).__name__}: {str(error)[:300]}"}))
else:
    print(json.dumps({"raised": False, "error": ""}))
"""


@pytest.mark.parametrize(
    ("bad", "at", "raised"),
    ((64, 0, True), (64, 1000, True), (-1, 500, True), (2**62, 7, True), (63, 1000, False)),
    ids=("R-first", "R-tail-step", "negative", "huge", "R-minus-1-in-range"),
)
def test_an_id_outside_the_table_traps(bad, at, raised) -> None:
    """An id outside ``[0, R)`` (a negative id included) stops the launch with a device-side trap, so the call or the
    next synchronize raises instead of returning rows read from outside the table; the last row, ``R - 1``, does
    not."""
    _device_or_skip()
    result = subprocess.run(
        [sys.executable, "-c", TRAP_SCRIPT, api.__name__, str(bad), str(at)],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["raised"] is raised, report


@pytest.mark.parametrize("positions", (1, 1001, 393_216))
def test_the_workspace_is_the_output_only(positions) -> None:
    """The call allocates exactly its output: the peak and the retained allocation both grow by
    ``round_up(P * 512, 512)`` bytes (the caching allocator's 512-byte rounding) and by nothing else."""
    device = _device_or_skip()
    rows = 4096
    table = _public_table(*random_bytes(rows, seed=141), "packed", device)
    ids = random_ids(rows, (positions,), seed=142, device=device)
    _lookup(table, ids)  # loads the library
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    out = _lookup(table, ids)
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    after = torch.cuda.memory_allocated(device)
    want = -(-positions * ROW_DIM * 2 // 512) * 512
    assert (peak - before, after - before) == (want, want)
    assert out.numel() * out.element_size() == positions * ROW_DIM * 2


@pytest.mark.parametrize("warn_only", (False, True), ids=("strict", "warn_only"))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_deterministic_mode_raises_no_alert_and_keeps_the_bits(layout, warn_only) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict or ``warn_only=True``: no alert, no warning, and the
    bits of the run outside the mode.  The output of the run in the mode is allocated in it, so PyTorch fills it with
    NaN first: the same bits show that every element is written."""
    device = _device_or_skip()
    table = _public_table(*random_bytes(512, seed=151, nan_every=9), layout, device)
    ids = random_ids(512, (33, N_COLS), seed=152, device=device)
    base = _lookup(table, ids)
    torch.cuda.synchronize(device)
    with _torch_deterministic(warn_only=warn_only), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = _lookup(table, ids)
        torch.cuda.synchronize(device)
    assert not caught, [f"{w.category.__name__}: {w.message}" for w in caught]
    assert torch.equal(_bits(got), _bits(base))


def test_operands_on_the_device_that_is_not_current() -> None:
    """With two devices: operands on the device that is not current give the reference's bits there."""
    _device_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")
    other = torch.device("cuda:1")
    if torch.cuda.get_device_capability(other) not in CAPABILITIES:
        pytest.skip("the second device is not an exported architecture")
    _build_libraries(1)
    previous = torch.cuda.current_device()
    torch.cuda.set_device(0)
    try:
        table = _public_table(*random_bytes(640, seed=161, nan_every=11), "packed", other)
        ids = random_ids(640, (40, N_COLS), seed=162, device=other)
        assert torch.cuda.current_device() == 0
        out = _lookup(table, ids)
        assert out.device == other
        torch.cuda.synchronize(other)
        assert_same_bits(out, oracle(table, ids), "other device")
    finally:
        torch.cuda.set_device(previous)
