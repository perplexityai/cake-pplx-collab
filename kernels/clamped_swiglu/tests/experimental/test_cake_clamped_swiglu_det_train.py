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

Backend suite of the Cake deterministic routed-expert clamped SwiGLU training package
(``cake_pplx._backends.cake_clamped_swiglu_det_train``, the backend of ``cake_pplx.cake_clamped_swiglu_det_train``):
``h = bf16(w silu(min(g, L)) clamp(u, -L, L))``, its gradients ``d_gate`` and ``d_up`` with the inclusive clamp masks,
and the FP32 route-weight gradient ``d_weight = sum dh a``, intermediate width 2304, on SM100, SM103 and SM107.

Without a device: the registry (per stage a small-R program on every architecture and, for ``fwd`` and ``bwd``, a
large-R program with its windows of rows per SM on some; the unweighted backward ``bwd_unweighted`` one program with no
weight, weight flag or weight gradient; each record delivered for its architectures and for no other, the launch facts,
the argument plans), the route of a launch by its rows per SM (1 row, one row below, at and above each edge of the
windows and the most rows, on the SM counts of the delivered GPUs; windows that are not ascending, untouching lists of
integers refused; the small-R program at every row count on an architecture without a large-R one; the SM count read
once per stage and device, no library loaded), the launch grid of each program (``ceil(3 R / 4)`` CTAs of
``bwd_unweighted`` up to the most rows, where ``3 R`` passes 32 bits) and the arguments of a launch in its
plan's order, the entry points' launcher of a stage and device (made once per stage and device, no library loaded; the
program, arguments and grid of ``run_module`` at every edge row count, each program's FFI entry looked up once; refused
for a program that takes its parameters in another order), the cached FP32 rounding of the last limit (used only for an
equal float or int; every other limit takes the full checks and their errors), the build manifest (every record's
library, the recorded source digests), the capability map, the required ``limit`` keyword, the refusal of CPU tensors
and no environment read outside the JIT shim.

On SM100 / SM103 / SM107, every training output (``h``, ``d_gate``, ``d_up`` and ``d_weight``) of each stage's small-R
and large-R program (each forced through the route) judged against the naive BF16 tier
(``tests/cake/test_helpers/cake_naive.py``: the error against an FP64 oracle of the same inputs over 32 independent
draws, in the L2 and the max norm, median and largest, beside the error of the same operation in BF16,
:mod:`tests.cake.test_helpers.clamped_swiglu_reference`) over row counts 1 to 4097, the merged, two-tensor, strided and
copied layouts, weighted and unweighted calls (the unweighted backward runs ``bwd_unweighted``), values on the clamp
boundary, deep negative gates, the edge of the FP32 sigmoid, a gate below -87 in each third of every row beside ordinary
and boundary values, every BF16 gate, a limit between two BF16 values and a small limit.  Beside the judge, exact facts
of the contract (no tolerance): the inclusive masks (a gradient is 0 exactly where its mask is 0 and passes at ``g = L``
and ``u = +-L``); zero-weight rows give exactly 0 in ``h``, ``d_gate`` and ``d_up`` and the weight gradient of any other
weight; the sigmoid's exact path (taken by a warp's run of loads with a clamped gate at or below -87) gives the bits of
its fast path wherever the clamped gate is above -87, in each program; ``d_weight`` sums the FP32 ``a``, and ``h`` is
the FP32 product ``w a`` rounded once to BF16; ``R = 0`` returns empty outputs and launches nothing; one library per
program serves every row count, limit, weight flag and layout; the small-R and the large-R programs of a stage give the
same bits at and one row above each edge of the device's windows, where the route runs the large-R program inside the
windows and the small-R one outside; ``bwd_unweighted`` stores the ``d_gate`` and ``d_up`` bits of both ``bwd`` programs
launched with the weight flag 0 and of a weighted call with weights of 1 (every data kind, NaN and infinities, a gate of
-88 in one third of a row, three layouts, three limits, 1 to 5 rows, 129, 4097 and the ``bwd`` window edges), and an
unweighted backward allocates no weight gradient; the same bits at every call, in every layout, at every row count (a
row's outputs do not depend on the other rows), on a side stream with no host synchronization, in CUDA graph replays of
the small-R and of the large-R programs, under ``torch.use_deterministic_algorithms`` (where every new output holds
PyTorch's NaN fill when its kernel launches) and with the current device switched to another GPU (each call runs on its
operands' device); the autograd function matches the steps bit for bit, saves only its inputs and gives the same bits
under non-reentrant checkpointing, its backward checks the saved operands again (unpacked by a saved-tensors hook as
rows to copy: the steps' bits; one row short: ``ValueError``), and gradients computed with ``create_graph=True`` have
the steps' bits and raise when differentiated; inputs the kernels can read are read in place, others copied once; unfit
outputs, operands and limits are refused, before any copy, allocation or launch; the one-pass checks of the common call
give the launches of the full checks; every unweighted forward launch of a device passes one cached FP32 buffer, set to
1, as its ignored weight, also when the process's first unweighted call is a CUDA graph capture.  On one fixed draw, the
L2 error against the FP64 oracle of ``h``, ``d_gate`` and ``d_up`` is below that of the weight applied after a BF16
rounding of each output, and the L2 error of ``d_weight`` below that of the sum over the BF16-rounded ``a``.

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import fractions
import functools
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
import types
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_naive import norm_errors
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.clamped_swiglu_reference import (
    DATA,
    SLOW_COLUMNS,
    SLOW_GATE,
    WIDTH,
    Case,
    bf16_edges,
    fp32,
    judge,
    make_inputs,
    oracle,
)
from tests.cake.test_helpers.clamped_swiglu_reference import outputs as step_outputs

pkg = cake_package("cake_clamped_swiglu_det_train")
cake_jit = cake_package("cake_clamped_swiglu_det_train", "cake_jit")

forward = pkg.cake_clamped_swiglu_det_train_forward
backward = pkg.cake_clamped_swiglu_det_train_backward
clamped_swiglu = pkg.cake_clamped_swiglu_det
# What makes a launcher of a stage and device, without the package's cache (:func:`_fresh_launchers`).
LAUNCHER = pkg._launcher.__wrapped__

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
STAGES = ("fwd", "bwd", "bwd_unweighted")
# The stages with a large-R program on some architecture (``bwd_unweighted`` has one program at every row count).
WINDOWED = ("fwd", "bwd")
# Each stage's small-R role (the stage itself) and its large-R role.
LARGE = "_large"
ROLES = ("fwd", "fwd_large", "bwd", "bwd_large", "bwd_unweighted", "bwd_unweighted_large")
# The SM counts of the delivered GPUs the routes are checked on.
SM_COUNTS = (148, 152, 212)
# The windows of rows per SM in which each stage's large-R program runs (the measured routes; ``None``: no upper
# bound), and the lane layout every program keeps: literal copies, so a record that drifts from them fails here, and
# the route tests expect these windows.
WINDOWS = {
    "fwd_large": {"sm_100a": [[112, None]], "sm_103a": [[208, None]], "sm_107a": [[384, None]]},
    "bwd_large": {
        "sm_100a": [[12, 21], [32, 41], [48, None]],
        "sm_103a": [[12, 21], [32, 41], [48, None]],
        "sm_107a": [[12, 21], [35, 41], [48, None]],
    },
}
LANE_LAYOUT = {"lanes": 32, "rows_per_thread": 1, "width_bits": 128}
# The build facts of each stage: the lane layout, and in ``bwd_unweighted`` 3 tasks a row (a warp per (row, run)).
BUILDS = {"fwd": LANE_LAYOUT, "bwd": LANE_LAYOUT, "bwd_unweighted": {**LANE_LAYOUT, "tasks_per_row": 3}}
# The program a forced launch runs: the small-R or the large-R program of its stage (:func:`_force`).
PROGRAMS = ("small", "large")
# The kernel parameters of each stage, in launch order (the kernels' signatures).
PARAMETERS = {
    "fwd": ("gate", "up", "w", "h", "rows", "gate_stride", "up_stride", "h_stride", "limit", "weighted"),
    "bwd": (
        "gate",
        "up",
        "dh",
        "w",
        "d_gate",
        "d_up",
        "d_weight",
        "rows",
        "gate_stride",
        "up_stride",
        "dh_stride",
        "d_gate_stride",
        "d_up_stride",
        "limit",
        "weighted",
    ),
    "bwd_unweighted": (
        "gate",
        "up",
        "dh",
        "d_gate",
        "d_up",
        "rows",
        "gate_stride",
        "up_stride",
        "dh_stride",
        "d_gate_stride",
        "d_up_stride",
        "limit",
    ),
}
BUFFERS = {"gate", "up", "w", "h", "dh", "d_gate", "d_up", "d_weight"}
L = 10.0

# The judged cases: row counts 1 to 4097, every layout, weighted and unweighted, the data kinds and three limits.
CASES = [
    Case(1),
    Case(3, "merged"),
    Case(129),
    Case(129, "strided"),
    Case(129, "copied"),
    Case(4097),
    Case(4097, "merged"),
    Case(4097, w="none"),
    Case(129, "merged", w="none"),
    Case(129, data="boundary"),
    Case(129, "merged", data="boundary"),
    Case(129, data="boundary", L=7.3),  # a limit between two BF16 values
    Case(129, "merged", data="boundary", L=1.5),  # most elements clamped
    Case(129, L=1.5),
    Case(129, data="deep_negative"),
    Case(129, data="newton_edge"),
    Case(129, data="sigmoid_edge"),
    Case(129, "merged", data="mixed"),
    Case(129, data="slow_mixed"),  # a gate of -88 in each third of every row, beside ordinary and boundary values
    Case(257, "merged", data="slow_mixed"),
    Case(129, "merged", w="none", data="slow_mixed"),
    Case(257, w="none", data="slow_mixed"),
    Case(29, data="all_gates"),  # 29 x 2304 >= 65536 gate patterns
]


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not pkg.is_cake_clamped_swiglu_det_train_supported(torch.device("cuda")):
        pytest.skip(f"the package ships {ARCHES}; this device is {torch.cuda.get_device_capability()}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in pkg.specs_for_arch(pkg.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


def _fresh_launchers(patch: pytest.MonkeyPatch) -> None:
    """The test (or the ``patch`` context) gets its own cache of launchers (``pkg._launcher``): the launchers it makes
    use its patched route, FFI entries or device, and none of them outlives it or reaches another test."""
    patch.setattr(pkg, "_launcher", functools.cache(LAUNCHER))


def _spy_entries(patch: pytest.MonkeyPatch, record) -> None:
    """Every launch calls ``record(name, values)`` before the FFI entry of its program ``name``, ``values`` naming
    the program's kernel parameters (the grid left out): each entry is wrapped as it is looked up, and the launchers
    are made afresh so that they bind the wrapped entries."""
    entry = pkg._entry

    def spy(name, arch):
        run, keys = entry(name, arch), pkg.parameters(name)

        def recorded(*args):
            record(name, dict(zip(keys, args[: len(keys)], strict=True)))
            return run(*args)

        return recorded

    patch.setattr(pkg, "_entry", spy)
    _fresh_launchers(patch)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list:
    """``(stage, rows, values)`` of every launch, in order: ``values`` names the kernel parameters of the launch."""
    launched: list = []
    _spy_entries(
        monkeypatch, lambda name, values: launched.append((cake_jit.MODULES[name]["stage"], values["rows"], values))
    )
    return launched


@pytest.fixture
def programs_run(monkeypatch: pytest.MonkeyPatch) -> list:
    """The program name of every launch, in order."""
    names: list = []
    _spy_entries(monkeypatch, lambda name, values: names.append(name))
    return names


@pytest.fixture
def host_device(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.SimpleNamespace]:
    """With no GPU: ``place(arch, sm_count)`` makes every device an ``arch`` device of ``sm_count`` SMs for the
    route, and each launch is recorded in ``calls`` as ``(name, arch, args)`` instead of loading a library.  The route
    cache is cleared at each placement and after the test, and each placement makes its launchers afresh."""
    calls: list = []

    def place(arch: str, sm_count: int) -> None:
        properties = types.SimpleNamespace(multi_processor_count=sm_count)
        monkeypatch.setattr(pkg, "device_arch", lambda index: arch)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
        pkg._route.cache_clear()
        _fresh_launchers(monkeypatch)

    monkeypatch.setattr(pkg, "_entry", lambda name, arch: lambda *args: calls.append((name, arch, args)))
    _fresh_launchers(monkeypatch)
    pkg._route.cache_clear()
    yield types.SimpleNamespace(place=place, calls=calls)
    pkg._route.cache_clear()


def _force(patch: pytest.MonkeyPatch, program: str) -> None:
    """Every launch runs the ``program`` (``"small"`` or ``"large"``) of its stage on its device, whatever its row
    count: the route's limit becomes 0, so every launch runs its tail, the small-R or the large-R plan.  An
    architecture without a large-R program of a stage runs the small-R one either way.  The launchers are made afresh
    on the forced route."""
    route = pkg._route

    def forced(stage: str, device_index: int) -> tuple:
        _, _, small, large, _ = route(stage, device_index)
        return 0, b"", small, large, large if program == "large" else small

    patch.setattr(pkg, "_route", forced)
    _fresh_launchers(patch)


def _in_windows(windows: list[list[int | None]], sm_count: int, rows: int) -> bool:
    """Whether ``rows`` lies in one of ``windows`` of rows per SM: ``lo * sm_count <= rows <= hi * sm_count``."""
    return any(lo * sm_count <= rows and (hi is None or rows <= hi * sm_count) for lo, hi in windows)


def _stages(weighted: bool) -> tuple[str, str]:
    """The stages of one step, in launch order: the forward, then the weighted backward ``bwd`` or the unweighted
    backward ``bwd_unweighted``."""
    return ("fwd", "bwd") if weighted else ("fwd", "bwd_unweighted")


def _routed(stage: str, rows: int, device_index: int) -> str:
    """The program a launch of ``stage`` for ``rows`` rows runs on device ``device_index``."""
    limit, table, small, large, tail = pkg._route(stage, device_index)
    return ((large if table[rows] else small) if rows < limit else tail)[0]


def _program(stage: str, device_index: int, program: str) -> str:
    """The ``program`` (``"small"`` or ``"large"``) of ``stage`` on device ``device_index``."""
    _, _, small, large, _ = pkg._route(stage, device_index)
    return (small if program == "small" else large)[0]


def _device_edges(stage: str, device_index: int) -> tuple[int, ...]:
    """The edges of the windows of ``stage`` on device ``device_index`` (:func:`cake_jit.route`)."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return cake_jit.route(stage, pkg.device_arch(device_index), sm_count)[0]


def _expected_route(stage: str, arch: str, sm_count: int) -> tuple[list[list[int | None]], str, str]:
    """``(windows, small, large)`` of ``stage`` on ``arch``: the windows of rows per SM of its large-R role on ``arch``
    (``WINDOWS``), the record of the stage's role and the one of its large-R role (none: no window and the small-R
    record)."""
    on_arch = {name: record["role"] for name, record in cake_jit.MODULES.items() if arch in record["arches"]}
    small = [name for name, role in on_arch.items() if role == stage]
    large = [name for name, role in on_arch.items() if role == stage + LARGE]
    assert len(small) == 1 and len(large) <= 1, (stage, arch, small, large)
    if not large:
        return [], small[0], small[0]
    return WINDOWS[stage + LARGE][arch], small[0], large[0]


def _edges(windows: list[list[int | None]], sm_count: int) -> list[int]:
    """The last row count of each run of one program on ``sm_count`` SMs, ascending: ``lo * sm_count - 1`` and
    ``hi * sm_count`` of each window (an open window's ``lo * sm_count - 1`` alone)."""
    return [bound * sm_count - (k == 0) for window in windows for k, bound in enumerate(window) if bound is not None]


def _edge_rows(edges: list[int]) -> list[int]:
    """1 row, one row below, at and above each edge, and the most rows a launch takes: those that are launches."""
    near = {edge + step for edge in edges for step in (-1, 0, 1)}
    return sorted({rows for rows in {1, cake_jit.MAX_ROWS - 1} | near if 1 <= rows < cake_jit.MAX_ROWS})


def assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape, (what, got.dtype, want.dtype, got.shape, want.shape)
    view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.contiguous().view(view), want.contiguous().view(view)), what


def assert_same_outputs(got: dict, want: dict, what: str) -> None:
    assert set(got) == set(want), what
    for name in want:
        assert_same_bits(got[name], want[name], f"{what}: {name}")


def _relayout(t: types.SimpleNamespace, layout: str) -> types.SimpleNamespace:
    """The inputs ``t`` (same values) in ``layout``: merged, split, strided (row slices of wider, aligned buffers) or
    copied (row strides that are not multiples of 8 elements)."""
    pads = {"strided": (32, 8, 16), "copied": (3, 1, 5)}

    def place(values: torch.Tensor, pad: int) -> torch.Tensor:
        buffer = torch.full(
            (values.shape[0], values.shape[1] + 2 * pad), float("nan"), dtype=values.dtype, device=values.device
        )
        buffer[:, pad : pad + values.shape[1]] = values
        return buffer[:, pad : pad + values.shape[1]]

    out = types.SimpleNamespace(R=t.R, L=t.L, layout=layout, weight=t.weight)
    gate, up, dh = t.gate.contiguous(), t.up.contiguous(), t.dh.contiguous()
    if layout == "merged":
        out.gate_up = torch.cat([gate, up], dim=1)
        out.gate, out.up = out.gate_up[:, :WIDTH], out.gate_up[:, WIDTH:]
        out.first, out.second, out.dh = out.gate_up, None, dh
    elif layout == "split":
        out.gate, out.up, out.dh = gate, up, dh
        out.first, out.second = gate, up
    else:
        pg, pu, pd = pads[layout]
        out.gate, out.up, out.dh = place(gate, pg), place(up, pu), place(dh, pd)
        out.first, out.second = out.gate, out.up
    return out


def _rows(t: types.SimpleNamespace, rows: int, offset: int = 0) -> types.SimpleNamespace:
    """Rows ``[offset, offset + rows)`` of the inputs ``t`` (views)."""
    span = slice(offset, offset + rows)
    out = types.SimpleNamespace(R=rows, L=t.L, layout=t.layout)
    out.gate, out.up, out.dh = t.gate[span], t.up[span], t.dh[span]
    out.weight = None if t.weight is None else t.weight[span]
    if t.second is None:
        out.first, out.second = t.first[span], None
    else:
        out.first, out.second = out.gate, out.up
    return out


def _weights(rows: int, *, seed: int, device: torch.device) -> torch.Tensor:
    """``rows`` FP32 route weights that are not powers of two: 0.7, 1.3, 2.2 and 2.9, then ``3 x U[0, 1)``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    w = 3.0 * torch.rand(rows, device=device, generator=gen)
    w[:4] = torch.tensor([0.7, 1.3, 2.2, 2.9], device=device)
    return w


def _leaves(t: types.SimpleNamespace) -> tuple:
    """Fresh leaves ``(first, second, weight)`` of the inputs ``t`` that require grad (None stays None)."""
    return tuple(None if x is None else x.detach().clone().requires_grad_(True) for x in (t.first, t.second, t.weight))


def _autograd_step(t: types.SimpleNamespace, *, checkpoint: bool = False) -> dict:
    """``h`` and the gradients of the leaves of one autograd step on ``t`` (the forward under non-reentrant
    checkpointing when asked)."""
    leaves = _leaves(t)
    if checkpoint:
        h = torch.utils.checkpoint.checkpoint(clamped_swiglu, *leaves, limit=t.L, use_reentrant=False)
    else:
        h = clamped_swiglu(*leaves, limit=t.L)
    h.backward(t.dh)
    found = {"h": h.detach(), "d_first": leaves[0].grad}
    if leaves[1] is not None:
        found["d_up"] = leaves[1].grad
    if leaves[2] is not None:
        found["d_weight"] = leaves[2].grad
    return found


def _functional(t: types.SimpleNamespace) -> dict:
    """The steps' ``h`` and gradients on ``t``, keyed as :func:`_autograd_step` keys them."""
    h = forward(t.first, t.second, t.weight, limit=t.L)
    d_first, d_up, d_weight = backward(t.dh, t.first, t.second, t.weight, limit=t.L)
    found = {"h": h, "d_first": d_first}
    if d_up is not None:
        found["d_up"] = d_up
    if d_weight is not None:
        found["d_weight"] = d_weight
    return found


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert dict(cake_jit.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    assert cake_jit.WIDTH == WIDTH == 2304
    assert cake_jit.PACKAGE == "cake_pplx._backends.cake_clamped_swiglu_det_train"
    assert tuple(cake_jit.ROLES) == ROLES and cake_jit.LARGE == LARGE
    # Records keyed by their program names (their libraries), each in one role of its stage.  A role's records serve
    # disjoint architectures: the stage's role every architecture, its large-R role some, each large-R record with the
    # rows per SM from which it runs on each of its architectures.  The host reaches them through PROGRAM_OF.
    programs: dict = {role: {} for role in ROLES}
    for name, record in cake_jit.MODULES.items():
        role = record["role"]
        assert role in ROLES, name
        assert record["stage"] == role.removesuffix(LARGE), name
        arches = list(record["arches"])
        assert arches and arches == [arch for arch in ARCHES if arch in arches], name
        for arch in arches:
            assert arch not in programs[role], (name, role, arch)
            programs[role][arch] = name
        if role.endswith(LARGE):
            windows = record["rows_per_sm_windows"]
            assert set(windows) == set(arches), name
            for arch, row in windows.items():
                bounds = [bound for window in row for bound in window]
                assert row and all(type(window) is list and len(window) == 2 for window in row), (name, arch)
                assert all(type(bound) is int for bound in bounds[:-1]), (name, arch)
                assert bounds[-1] is None or type(bounds[-1]) is int, (name, arch)
                bounds = [bound for bound in bounds if bound is not None]
                assert bounds[0] >= 1 and bounds == sorted(set(bounds)), (name, arch)
        else:
            assert "rows_per_sm_windows" not in record, name
    for stage in STAGES:
        assert set(programs[stage]) == set(ARCHES), stage
    assert cake_jit.PROGRAM_OF == programs
    assert set(cake_jit.BUILD_OF) == set(STAGES)
    prefix = cake_jit.LIBRARY_PREFIX + "_"
    for name, record in cake_jit.MODULES.items():
        assert record["library"] == name
        assert re.fullmatch(re.escape(prefix) + "[0-9a-f]{20}", name), name
        assert record["ffi_entry"] == "run", name
        assert record["kernel_symbol"] == "kernel_" + record["library"], name
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["dynamic_smem_bytes"] == 0, name
        threads = launch["block"][0]
        build = cake_jit.BUILD_OF[record["stage"]]
        assert tuple(launch["block"]) == (threads, 1, 1) and threads % 32 == 0, name
        assert threads % build["lanes"] == 0, name
        per_cta = threads // build["lanes"] * build["rows_per_thread"]
        assert cake_jit.cta_tasks(name) == (per_cta, build.get("tasks_per_row", 1)), name
    # The BF16 stride and alignment rule is one load of the widest build: 8 values and 16 bytes for 128-bit loads.
    widest = max(build["width_bits"] for build in cake_jit.BUILD_OF.values())
    assert cake_jit.VECTOR_ELEMENTS == widest // 16 and cake_jit.ALIGNMENT_BYTES == widest // 8


def test_the_windows_and_the_lane_layout_are_the_measured_ones() -> None:
    windows: dict = {role: {} for role in WINDOWS}
    for name, record in cake_jit.MODULES.items():
        if record["role"] in windows:
            for arch, row in record["rows_per_sm_windows"].items():
                assert arch not in windows[record["role"]], (name, arch)
                windows[record["role"]][arch] = row
    assert windows == WINDOWS
    layouts = {stage: dict(build) for stage, build in cake_jit.BUILD_OF.items()}
    assert layouts == BUILDS


def test_argument_plans_are_the_kernel_parameters() -> None:
    """Each record takes its kernel's parameters in order (tensors as buffers, scalars as parameters), then the
    grid; the host names every one of them."""
    for name, record in cake_jit.MODULES.items():
        assert pkg.parameters(name) == PARAMETERS[record["stage"]], name
        for kind, key in record["arg_plan"]:
            if kind == "grid":
                continue
            assert kind == ("buffer" if key in BUFFERS else "parameter"), (name, kind, key)
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == ["grid_x", "grid_y", "grid_z"]


def test_grid_is_one_cta_per_block_of_tasks() -> None:
    """A CTA owns ``threads / lanes * rows_per_thread`` tasks and a row is ``tasks_per_row`` tasks (a row in ``fwd``
    and ``bwd``, 3 (row, run) tasks in ``bwd_unweighted``): the grid is ``ceil(rows * tasks_per_row / per_cta)`` CTAs,
    at most ``2**31 - 1`` at the most rows."""
    for name in cake_jit.MODULES:
        per_cta, per_row = cake_jit.cta_tasks(name)
        for rows in (1, 2, per_cta - 1, per_cta, per_cta + 1, 129, 4097, 98304, 196608, cake_jit.MAX_ROWS - 1):
            if rows >= 1:
                assert tuple(cake_jit.grid(name, rows)) == (-(-rows * per_row // per_cta), 1, 1), (name, rows)
        assert cake_jit.grid(name, cake_jit.MAX_ROWS - 1)[0] <= 2**31 - 1
        if per_row == 1:
            # The fwd and bwd kernels round the row count up to whole blocks in 32 bits.
            assert cake_jit.MAX_ROWS - 1 + per_cta - 1 < 2**31
        with pytest.raises(ValueError):
            cake_jit.grid(name, 0)


# The grid of the unweighted backward, ``ceil(3 R / 4)`` CTAs: row counts with a partial last CTA, the last row counts
# at which ``3 R`` fits 32 bits and the first past it (the kernel counts its tasks in 64 bits), and the most rows.
UNWEIGHTED_GRIDS = {
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    129: 97,
    4097: 3073,
    715_827_881: 536_870_911,
    715_827_882: 536_870_912,
    715_827_883: 536_870_913,
    2**31 - 257: 1_610_612_544,
}


def test_the_unweighted_backward_program_and_its_grid(host_device) -> None:
    """``bwd_unweighted`` has one record, on all three architectures, with no windows and no large-R program, and takes
    the 12 kernel parameters of :data:`PARAMETERS` (no weight, weight flag or weight gradient), in the order the entry
    points pass them.  Its grid and the grid of every launch through ``run_module`` and the launcher, on every
    architecture and SM count, is ``ceil(3 R / 4)`` CTAs (:data:`UNWEIGHTED_GRIDS`): below ``2**31 - 1`` at the most
    rows, where ``3 R`` is past 32 bits."""
    names = [name for name, record in cake_jit.MODULES.items() if record["stage"] == "bwd_unweighted"]
    assert len(names) == 1, names
    (name,) = names
    record = cake_jit.MODULES[name]
    assert record["role"] == "bwd_unweighted" and list(record["arches"]) == list(ARCHES)
    assert "rows_per_sm_windows" not in record
    assert cake_jit.PROGRAM_OF["bwd_unweighted"] == dict.fromkeys(ARCHES, name)
    assert cake_jit.PROGRAM_OF["bwd_unweighted_large"] == {}
    assert pkg.parameters(name) == PARAMETERS["bwd_unweighted"] == pkg._ARGUMENTS["bwd_unweighted"]
    assert tuple(record["launch"]["block"]) == (128, 1, 1)
    assert cake_jit.cta_tasks(name) == (4, 3)
    assert 3 * (cake_jit.MAX_ROWS - 1) > 2**31 - 1
    for rows, want in UNWEIGHTED_GRIDS.items():
        assert want == -(-3 * rows // 4) <= 2**31 - 1, rows
        assert cake_jit.grid(name, rows) == (want, 1, 1), rows
    values = {key: f"<{key}>" for key in PARAMETERS["bwd_unweighted"]}
    for arch in ARCHES:
        for sm_count in SM_COUNTS:
            assert cake_jit.route("bwd_unweighted", arch, sm_count) == ((), name, name)
            host_device.place(arch, sm_count)
            launch = pkg._launcher("bwd_unweighted", 0)
            for rows, want in UNWEIGHTED_GRIDS.items():
                host_device.calls.clear()
                pkg.run_module("bwd_unweighted", values, rows=rows, device_index=0)
                launch(rows, *(values[key] for key in PARAMETERS["bwd_unweighted"]))
                args = (*(values[key] for key in PARAMETERS["bwd_unweighted"]), want, 1, 1)
                assert host_device.calls == [(name, arch, args)] * 2, (arch, sm_count, rows)


def test_launches_pass_the_argument_plan_and_the_grid(host_device) -> None:
    """``run_module`` passes the routed program's kernel parameters in its argument plan's order, then
    ``cake_jit.grid(name, rows)``; a missing parameter and a launch of no rows are ``ValueError``s naming them, raised
    before any library is looked up."""
    host_device.place("sm_100a", 148)
    calls = host_device.calls
    for stage in STAGES:
        windows, small, large = _expected_route(stage, "sm_100a", 148)
        values = {key: f"<{key}>" for key in PARAMETERS[stage]}
        for rows in (1, 2, 3, 4, 5, 7, 8, 9, 129, 1777, 4097, cake_jit.MAX_ROWS - 1):
            name = large if _in_windows(windows, 148, rows) else small
            calls.clear()
            pkg.run_module(stage, values, rows=rows, device_index=0)
            want = (*(values[key] for key in PARAMETERS[stage]), *cake_jit.grid(name, rows))
            assert calls == [(name, "sm_100a", want)], (stage, rows)
        calls.clear()
        with pytest.raises(ValueError, match="at least one row"):
            pkg.run_module(stage, values, rows=0, device_index=0)
        missing = {key: value for key, value in values.items() if key != "limit"}
        with pytest.raises(ValueError, match=f"{small} needs argument 'limit'"):
            pkg.run_module(stage, missing, rows=1, device_index=0)
        assert calls == []


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
def test_rows_route_to_the_small_or_the_large_program(stage, arch, sm_count, host_device) -> None:
    """A launch whose rows lie in a window of the large-R record (``lo * SMs <= rows <= hi * SMs`` for a window
    ``[lo, hi]`` of its ``rows_per_sm_windows`` on the architecture) runs the stage's large-R program, any other its
    small-R program, on that program's grid: 1 row, one row below, at and above each edge of the windows, and
    ``MAX_ROWS - 1`` rows.  The package's route holds a byte per row count below one above the last edge, 1 exactly
    in a window, and runs the large-R program above it when the last window has no upper bound."""
    windows, small, large = _expected_route(stage, arch, sm_count)
    edges = _edges(windows, sm_count)
    assert cake_jit.route(stage, arch, sm_count) == (tuple(edges), small, large)
    host_device.place(arch, sm_count)
    limit, table, got_small, got_large, tail = pkg._route(stage, 0)
    assert (got_small[0], got_large[0]) == (small, large)
    assert limit == (edges[-1] + 1 if edges else 0)
    assert table == bytes(_in_windows(windows, sm_count, rows) for rows in range(limit))
    assert tail[0] == (large if windows and windows[-1][1] is None else small)
    values = {key: f"<{key}>" for key in PARAMETERS[stage]}
    ran = set()
    for rows in _edge_rows(edges):
        name = large if _in_windows(windows, sm_count, rows) else small
        ran.add(name)
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, device_index=0)
        want = (*(values[key] for key in PARAMETERS[stage]), *cake_jit.grid(name, rows))
        assert host_device.calls == [(name, arch, want)], (rows, windows)
    assert ran == {small, large}


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
def test_an_architecture_without_a_large_program_runs_the_small_one(stage, arch, host_device, monkeypatch) -> None:
    """With no large-R program of the stage on the architecture (the registry's table without it), every row count
    runs the small-R program: the route has no edge."""
    sm_count = SM_COUNTS[-1]
    windows, small, _ = _expected_route(stage, arch, sm_count)
    table = {role: dict(programs) for role, programs in cake_jit.PROGRAM_OF.items()}
    table[stage + LARGE].pop(arch, None)
    monkeypatch.setattr(cake_jit, "PROGRAM_OF", table)
    assert cake_jit.route(stage, arch, sm_count) == ((), small, small)
    host_device.place(arch, sm_count)
    values = {key: f"<{key}>" for key in PARAMETERS[stage]}
    for rows in _edge_rows(_edges(windows, sm_count)):
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, device_index=0)
        want = (*(values[key] for key in PARAMETERS[stage]), *cake_jit.grid(small, rows))
        assert host_device.calls == [(small, arch, want)], rows


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ([], "are not \\[lo, hi\\] lists of integers"),
        ([[12]], "are not \\[lo, hi\\] lists of integers"),
        ([[12, 20, 30]], "are not \\[lo, hi\\] lists of integers"),
        ([[None, 20]], "are not \\[lo, hi\\] lists of integers"),
        ([[12, None], [36, 40]], "are not \\[lo, hi\\] lists of integers"),
        ([[12.0, 20]], "are not \\[lo, hi\\] lists of integers"),
        ([[True, 20]], "are not \\[lo, hi\\] lists of integers"),
        ([[-1, 20]], "are not \\[lo, hi\\] lists of integers"),
        ([[12, 12]], "do not ascend without touching"),
        ([[20, 12]], "do not ascend without touching"),
        ([[12, 20], [20, 40]], "do not ascend without touching"),
        ([[36, 40], [12, 20]], "do not ascend without touching"),
    ],
)
def test_the_route_refuses_windows_that_are_no_route(row, message, monkeypatch) -> None:
    """A large-R record's windows on an architecture must be ascending ``[lo, hi]`` lists of integers that do not
    touch, ``0 <= lo``, only the last ``hi`` None: any other row is a ``RuntimeError`` naming the record."""
    large = cake_jit.PROGRAM_OF["bwd_large"]["sm_100a"]
    record = {**cake_jit.MODULES[large], "rows_per_sm_windows": {"sm_100a": row}}
    monkeypatch.setitem(cake_jit.MODULES, large, record)
    with pytest.raises(RuntimeError, match=f"{large}: rows_per_sm_windows\\['sm_100a'\\] .* {message}"):
        cake_jit.route("bwd", "sm_100a", 148)


def test_the_route_reads_the_device_once_and_loads_no_library(host_device, monkeypatch) -> None:
    """The route of a stage reads the device's SM count once per stage and device and loads no library (a program's
    library loads at its first launch)."""

    def refuse(*args, **kwargs):
        raise AssertionError("the route loaded a library")

    host_device.place("sm_103a", SM_COUNTS[1])
    reads: list = []
    properties = torch.cuda.get_device_properties
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: reads.append(index) or properties(index))
    monkeypatch.setattr(pkg, "load", refuse)
    monkeypatch.setattr(pkg, "_entry", refuse)
    for _ in range(3):
        for stage in STAGES:
            pkg._route(stage, 0)
            pkg._route(stage, 1)
    assert sorted(reads) == [0] * len(STAGES) + [1] * len(STAGES)


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
def test_the_launcher_launches_what_run_module_launches(stage, arch, sm_count, host_device, monkeypatch) -> None:
    """The launcher of a stage and device (the entry points' launch) is made once per stage and device and loads no
    library when made; at 1 row, one row below, at and above each edge of the windows and ``MAX_ROWS - 1`` rows it
    runs the program ``run_module`` runs, with the same arguments (the kernel parameters in the plan's order) and grid,
    and it looks up each program's FFI entry once, at that program's first launch."""
    host_device.place(arch, sm_count)
    bound: list = []
    entry = pkg._entry
    monkeypatch.setattr(pkg, "_entry", lambda name, on: bound.append((name, on)) or entry(name, on))
    launch = pkg._launcher(stage, 0)
    assert pkg._launcher(stage, 0) is launch
    assert pkg._launcher(stage, 1) is not launch
    assert bound == [] and host_device.calls == []
    windows, small, large = _expected_route(stage, arch, sm_count)
    values = {key: f"<{key}>" for key in PARAMETERS[stage]}
    rows_list = _edge_rows(_edges(windows, sm_count))
    launched = []
    for rows in rows_list:
        host_device.calls.clear()
        launch(rows, *(values[key] for key in PARAMETERS[stage]))
        launched.append(list(host_device.calls))
    assert sorted(bound) == sorted((name, arch) for name in {small, large}), bound
    for rows, got in zip(rows_list, launched, strict=True):
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, device_index=0)
        assert got == host_device.calls, rows
        assert got[0][0] == (large if _in_windows(windows, sm_count, rows) else small), rows


def test_the_launcher_refuses_programs_that_take_another_order(host_device, monkeypatch) -> None:
    """A launcher passes the kernel parameters by position, so it is not made (``RuntimeError`` naming the record) for
    a stage whose program takes them in another order than the entry points pass them; nothing is launched."""
    host_device.place("sm_100a", SM_COUNTS[0])
    name = cake_jit.PROGRAM_OF["bwd_large"]["sm_100a"]
    plan = [list(item) for item in cake_jit.MODULES[name]["arg_plan"]]
    plan[0], plan[1] = plan[1], plan[0]  # the up before the gate
    monkeypatch.setitem(cake_jit.MODULES, name, {**cake_jit.MODULES[name], "arg_plan": plan})
    pkg.parameters.cache_clear()
    pkg._plan.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=f"record {name}: kernel parameters"):
            pkg._launcher("bwd", 0)
        assert host_device.calls == []
        pkg._launcher("fwd", 0)  # the other stage's programs keep the order
    finally:
        pkg.parameters.cache_clear()
        pkg._plan.cache_clear()
        pkg._route.cache_clear()


def test_the_limit_check_and_its_cached_rounding(monkeypatch) -> None:
    """``limit`` is the same FP32 value whatever the calls before passed: a float or an int equal to the last
    accepted limit gets its cached rounding, and every other value (another number, a bool, a float subclass, a
    fraction, a tensor, a string, a non-finite, non-positive or too large number) takes the full checks, with their
    errors, also right after an equal valid limit was cached."""
    monkeypatch.setattr(pkg, "_last_limit", (math.nan, math.nan))

    class Limit(float):
        pass

    valid = (10.0, 10, 10.0, 7.3, 7.3, 1.5, 3, 2**53 + 1, 2**53 + 1, 1e30, 10**30, fractions.Fraction(3, 2), Limit(7.3))
    for limit in valid + valid[::-1]:
        got = pkg._limit(limit)
        assert type(got) is float and got == fp32(float(limit)), limit
        assert pkg._last_limit == (float(limit), got), limit
    invalid = (
        (True, TypeError),
        (False, TypeError),
        (torch.tensor(1.0), TypeError),
        ("1", TypeError),
        (None, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (0.0, ValueError),
        (-0.0, ValueError),
        (-1.0, ValueError),
        (-1, ValueError),
        (1e39, ValueError),
        (1e-50, ValueError),
        (10**400, ValueError),
        (fractions.Fraction(10**400), ValueError),
    )
    for cached in (1.0, 1, 0.5):
        for bad, kind in invalid:
            pkg._limit(cached)
            with pytest.raises(kind):
                pkg._limit(bad)
            assert pkg._last_limit == (float(cached), float(cached)), (cached, bad)


def test_no_record_is_delivered_outside_its_architectures() -> None:
    for name, record in cake_jit.MODULES.items():
        for arch in record["arches"]:
            assert pkg.library_name(name, arch) == f"{record['library']}_{arch}"
        for arch in (*(arch for arch in ARCHES if arch not in record["arches"]), "sm_90a", "sm_110a", "sm_120a"):
            with pytest.raises(RuntimeError):
                pkg.library_name(name, arch)


def test_capabilities_map_to_the_three_architectures(monkeypatch) -> None:
    """(10, 0), (10, 3) and (10, 7) map to ``sm_100a``, ``sm_103a`` and ``sm_107a``; any other capability is no target,
    and a device of one is a ``RuntimeError`` naming the delivered architectures."""
    assert tuple(pkg.ARCHES) == ARCHES
    assert tuple(pkg.SUPPORTED_COMPUTE_CAPABILITIES) == ((10, 0), (10, 3), (10, 7))
    assert pkg.INTERMEDIATE_SIZE == WIDTH
    for capability, arch in ARCH_BY_CAPABILITY.items():
        assert pkg.arch_for_capability(capability) == arch
    for capability in ((9, 0), (10, 1), (11, 0), (12, 0), (12, 1)):
        assert pkg.arch_for_capability(capability) is None, capability
    for capability, want in (((10, 7), "sm_107a"), ((9, 0), None)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        pkg._arch_of_index.cache_clear()
        try:
            if want is None:
                with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
                    pkg.device_arch(0)
            else:
                assert pkg.device_arch(0) == want
        finally:
            monkeypatch.undo()
            pkg._arch_of_index.cache_clear()


def test_limit_is_a_required_keyword() -> None:
    """``limit`` has no default in any entry: a silent mismatch with the model's limit cannot happen."""
    for entry in (forward, backward, clamped_swiglu):
        parameter = inspect.signature(entry).parameters["limit"]
        assert parameter.kind is parameter.KEYWORD_ONLY, entry.__name__
        assert parameter.default is parameter.empty, entry.__name__
    x = torch.zeros(2, 2 * WIDTH, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="limit"):
        forward(x)
    with pytest.raises(TypeError, match="limit"):
        clamped_swiglu(x)


def test_cpu_tensors_are_refused() -> None:
    x = torch.zeros(2, 2 * WIDTH, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        forward(x, limit=L)
    with pytest.raises(ValueError, match="CUDA"):
        backward(x[:, :WIDTH], x, limit=L)


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{record['library']}_{arch}" for record in cake_jit.MODULES.values() for arch in record["arches"]}
    assert set(entries) == wanted
    assert len(entries) == sum(len(record["arches"]) for record in cake_jit.MODULES.values())
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
    records = sum(len(record["arches"]) for record in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


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
# On SM100 / SM103 / SM107: accuracy, judged against the naive BF16 tier


def test_the_records_load(device) -> None:
    arch = pkg.device_arch(device.index)
    assert arch in ARCHES
    for name, record in cake_jit.MODULES.items():
        if arch in record["arches"]:
            assert callable(getattr(pkg.load(name, arch), record["ffi_entry"])), name


@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_outputs_meet_the_naive_tier(case, program, device, record_property, monkeypatch) -> None:
    """``h``, ``d_gate``, ``d_up`` and (weighted) ``d_weight`` of each stage's small-R and large-R program (forced
    through the route) hold their contract (the oracle's shape, finite where it is finite: every input here is finite)
    and meet the naive BF16 tier over 32 draws."""
    _force(monkeypatch, program)
    judge(record_property, case, forward, backward, device=device)


# ----------------------------------------------------------------------------------------------------------------
# Exact facts of the contract


@pytest.mark.parametrize("limit", [10.0, 7.3, 1.5], ids=["L10", "L7.3", "L1.5"])
def test_clamp_masks_are_inclusive(limit, device) -> None:
    """With ``dh = 1`` and ``w = 1``: at ``up = 1``, ``d_gate`` is nonzero if and only if ``gate <= L``; at
    ``gate = 1``, ``d_up`` is nonzero if and only if ``-L <= up <= L``; for the issue's probes (``L - 0.0625``,
    ``L``, ``L + 0.0625``, ``-L``, ``-50`` and ``+-L``, ``+-(L + 0.0625)``, as BF16) and the BF16 values on each side of
    ``L`` and ``-L``."""
    Lf = fp32(limit)
    edges = bf16_edges(Lf, device)
    issue_gates = torch.tensor([Lf - 0.0625, Lf, Lf + 0.0625, -Lf, -50.0], device=device).bfloat16().float()
    issue_ups = torch.tensor([Lf, -Lf, Lf + 0.0625, -Lf - 0.0625], device=device).bfloat16().float()
    gates = torch.cat([edges, issue_gates]).unique()
    ups = torch.cat([edges, issue_ups]).unique()
    rows = len(gates) + len(ups)
    g = torch.ones(rows, WIDTH, device=device)
    u = torch.ones(rows, WIDTH, device=device)
    g[: len(gates)] = gates[:, None]
    u[len(gates) :] = ups[:, None]
    gate, up = g.bfloat16(), u.bfloat16()
    dh = torch.ones(rows, WIDTH, dtype=torch.bfloat16, device=device)
    w = torch.ones(rows, device=device)
    for first, second in ((gate, up), (torch.cat([gate, up], dim=1), None)):
        d_first, d_up, d_weight = backward(dh, first, second, w, limit=limit)
        if second is None:
            d_gate, d_up = d_first[:, :WIDTH], d_first[:, WIDTH:]
        else:
            d_gate = d_first
        for i, value in enumerate(gates.tolist()):
            passes = value <= Lf
            assert bool((d_gate[i] != 0).all()) if passes else bool((d_gate[i] == 0).all()), ("gate", value, Lf)
            assert bool((d_up[i] != 0).all()), ("up = 1 at gate", value)
        for k, value in enumerate(ups.tolist()):
            i = len(gates) + k
            passes = -Lf <= value <= Lf
            assert bool((d_up[i] != 0).all()) if passes else bool((d_up[i] == 0).all()), ("up", value, Lf)
            assert bool((d_gate[i] != 0).all()), ("gate = 1 at up", value)
        assert bool(d_weight.isfinite().all())


@pytest.mark.parametrize("data", ["boundary", "slow_mixed"])
@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize("limit", [10.0, 7.3])
def test_masked_gradients_are_exactly_zero(data, layout, limit, device) -> None:
    """On boundary data, also with a gate of -88 in each third of every row (the sigmoid's exact path), ``d_gate`` is
    exactly 0 wherever ``gate > L`` and ``d_up`` wherever ``|up| > L``, and the gradient passes on the boundary:
    nonzero at ``gate <= L`` (``up = +-L``) wherever its other factors are."""
    t = make_inputs(Case(257, layout, data=data, L=limit), seed=3, device=device)
    found = step_outputs(forward, backward, t)
    g, u, d = t.gate.float(), t.up.float(), t.dh.float()
    assert bool((found["d_gate"][g > t.L] == 0).all())
    assert bool((found["d_up"][(u > t.L) | (u < -t.L)] == 0).all())
    live = (t.weight != 0)[:, None] & (d != 0)
    on_gate = (g == float(torch.tensor(t.L).bfloat16())) & (g <= t.L) & live & (u != 0)
    on_up = (u.abs() == float(torch.tensor(t.L).bfloat16())) & (u.abs() <= t.L) & live & (g != 0) & (g > -80)
    near_gate = (g < t.L) & (g >= t.L - 1) & live & (u != 0)
    assert bool(on_gate.any() or near_gate.any())
    assert bool((found["d_gate"][on_gate | near_gate] != 0).all())
    assert bool((found["d_up"][on_up] != 0).all())


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_zero_weight_rows_are_exactly_zero(layout, device) -> None:
    """A zero-weight row gives exactly 0 in ``h``, ``d_gate`` and ``d_up``, and a nonzero ``d_weight``; ``d_weight``
    does not depend on the weight (no division by it): the same bits with every weight replaced, and with all of
    them 0."""
    t = make_inputs(Case(129, layout), seed=7, device=device)
    found = step_outputs(forward, backward, t)
    zero = t.weight == 0
    assert int(zero.sum()) == 43
    for name in ("h", "d_gate", "d_up"):
        assert bool((found[name][zero] == 0).all()), name
        assert bool((found[name][~zero] != 0).any()), name
    assert bool((found["d_weight"][zero] != 0).all())
    for other in (1.0 + torch.rand(t.R, device=device), torch.zeros(t.R, device=device)):
        _, _, d_weight = backward(t.dh, t.first, t.second, other, limit=t.L)
        assert_same_bits(d_weight, found["d_weight"], "d_weight with other weights")
    h = forward(t.first, t.second, torch.zeros(t.R, device=device), limit=t.L)
    assert bool((h == 0).all())


@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize("w", ["random", "none"])
def test_no_rows_return_empty_outputs_and_launch_nothing(layout, w, device, monkeypatch) -> None:
    """``R = 0``: the steps and the autograd function return empty outputs of the right shapes and dtypes, and
    nothing is launched or loaded."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call with no rows launched or loaded a kernel")

    monkeypatch.setattr(pkg, "run_module", refuse)
    monkeypatch.setattr(pkg, "_launcher", refuse)
    monkeypatch.setattr(pkg, "load", refuse)
    monkeypatch.setattr(pkg, "_entry", refuse)
    t = make_inputs(Case(0, layout, w), seed=1, device=device)
    h = forward(t.first, t.second, t.weight, limit=L)
    assert h.shape == (0, WIDTH) and h.dtype == torch.bfloat16 and h.device == device
    d_first, d_up, d_weight = backward(t.dh, t.first, t.second, t.weight, limit=L)
    if layout == "merged":
        assert d_first.shape == (0, 2 * WIDTH) and d_up is None
    else:
        assert d_first.shape == (0, WIDTH) and d_up.shape == (0, WIDTH)
    assert d_first.dtype == torch.bfloat16
    if w == "none":
        assert d_weight is None
    else:
        assert d_weight.shape == (0,) and d_weight.dtype == torch.float32
    leaves = _leaves(t)
    out = clamped_swiglu(*leaves, limit=L)
    assert out.shape == (0, WIDTH)
    out.sum().backward()
    assert leaves[0].grad.shape == leaves[0].shape
    for leaf in leaves[1:]:
        if leaf is not None:
            assert leaf.grad.shape == leaf.shape


def test_one_library_per_program_serves_every_call(device, monkeypatch) -> None:
    """Row counts 1 to 4097 and each edge of the device's windows of each stage and one row past it, both layouts, a
    strided layout, weighted and unweighted calls and three limits load one library per program they run, once: no
    recompilation (``rows``, ``limit``, the weight flag and the strides are launch arguments)."""
    edges = sorted({edge for stage in WINDOWED for edge in _device_edges(stage, device.index)})
    cases = [Case(1), Case(4097), Case(3, "merged", "none", L=7.3), Case(129, "strided", L=1.5), Case(2)]
    cases += [Case(rows, "merged") for edge in edges for rows in (edge, edge + 1)]
    cake_build = cake_package("cake_clamped_swiglu_det_train", "cake_build")
    loaded: list[str] = []
    load_library = cake_build.load_library

    def spy(name):
        loaded.append(name)
        return load_library(name)

    monkeypatch.setattr(cake_build, "load_library", spy)
    pkg.load.cache_clear()
    pkg._entry.cache_clear()
    _fresh_launchers(monkeypatch)  # launchers that bind their entries anew, through the spied loader
    try:
        for case in cases:
            step_outputs(forward, backward, make_inputs(case, seed=5, device=device))
    finally:
        pkg.load.cache_clear()
        pkg._entry.cache_clear()
    arch = pkg.device_arch(device.index)
    run = {_routed(stage, case.R, device.index) for case in cases for stage in _stages(case.w != "none")}
    assert sorted(loaded) == sorted(pkg.library_name(name, arch) for name in run)


@pytest.mark.parametrize("step", [0, 1], ids=["edge", "edge+1"])
@pytest.mark.parametrize("stage", WINDOWED)
def test_the_programs_of_a_stage_give_the_same_bits(stage, step, device, programs_run, monkeypatch) -> None:
    """At and one row above each edge of this device's windows of ``stage`` (the row counts where its route changes
    program), weighted on rows at the edge of the FP32 sigmoid, and on gates of -88 beside boundary values (unweighted
    for ``fwd``; weighted for ``bwd``, which runs weighted calls only): the routed launches run the large-R program of
    ``stage`` inside its windows (one row above the edge below a window, at its ``hi`` edge) and the small-R program
    outside, and every launch forced to the small-R programs and every launch forced to the large-R programs give the
    routed call's bits of ``h``, ``d_gate``, ``d_up`` and ``d_weight``."""
    edges = _device_edges(stage, device.index)
    small, large = (_program(stage, device.index, program) for program in PROGRAMS)
    if not edges:
        pytest.skip(f"no large-R {stage} program on {pkg.device_arch(device.index)}")
    for index, edge in enumerate(edges):
        rows = edge + step
        # an even edge is the last row below a window (which starts one row above it), an odd edge the last row in it
        inside = (index % 2 == 0) == (step == 1)
        for w, data in (("random", "mixed"), ("none" if stage == "fwd" else "random", "slow_mixed")):
            t = make_inputs(Case(rows, "merged", w, data=data), seed=43, device=device)
            stages = _stages(w != "none")
            programs_run.clear()
            want = step_outputs(forward, backward, t)
            assert programs_run == [_routed(s, rows, device.index) for s in stages], (rows, w)
            assert programs_run[stages.index(stage)] == (large if inside else small), (rows, w)
            for program in PROGRAMS:
                with monkeypatch.context() as patch:
                    _force(patch, program)
                    programs_run.clear()
                    got = step_outputs(forward, backward, t)
                assert programs_run == [_program(s, device.index, program) for s in stages], (program, rows, w)
                assert_same_outputs(got, want, f"{program} programs, {rows} rows, w={w}")


# The inputs of the unweighted backward's bit checks: every judged data kind, NaN and infinities in the gate, the up and
# ``dh`` (``"nonfinite"``: about 1 % of each, no oracle judges them), and a gate of -88 in one third of each row only, a
# different third in consecutive rows (``"one_run"``: that run of 768 columns takes the deep-gate sigmoid, the other two
# the fast one).
BITS_DATA = (*DATA, "nonfinite", "one_run")


def _bits_inputs(data: str, layout: str, rows: int, limit: float, *, seed: int, device) -> types.SimpleNamespace:
    """Unweighted inputs of ``data`` (:data:`BITS_DATA`) in ``layout``."""
    base = {"nonfinite": "normal", "one_run": "boundary"}.get(data, data)
    t = make_inputs(Case(rows, "split", "none", data=base, L=limit), seed=seed, device=device)
    gate, up, dh = t.gate.clone(), t.up.clone(), t.dh.clone()
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    if data == "nonfinite":
        special = torch.tensor([float("nan"), float("inf"), float("-inf")], device=device)
        for x in (gate, up, dh):
            where = torch.rand(x.shape, device=device, generator=gen) < 0.01
            pick = special[torch.randint(0, 3, x.shape, device=device, generator=gen)].to(x.dtype)
            x.copy_(torch.where(where, pick, x))
    elif data == "one_run":
        r = torch.arange(rows, device=device)
        gate[r, (r % 3) * (WIDTH // 3) + r % 7] = SLOW_GATE
    plain = types.SimpleNamespace(R=rows, L=t.L, weight=None, gate=gate, up=up, dh=dh)
    return _relayout(plain, layout)


def _bwd_with_weight_flag_0(t: types.SimpleNamespace, name: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(d_gate, d_up)`` of the ``bwd`` program ``name`` itself (no route) launched as 0.1.2 launched an unweighted
    backward: the weight flag 0, the device's cached one-element weight, and a weight gradient it writes and nobody
    reads."""
    merged = t.second is None
    if merged:
        d_first = torch.empty(t.R, 2 * WIDTH, dtype=torch.bfloat16, device=device)
        d_gate, d_up = d_first[:, :WIDTH], d_first[:, WIDTH:]
    else:
        d_gate = torch.empty(t.R, WIDTH, dtype=torch.bfloat16, device=device)
        d_up = torch.empty(t.R, WIDTH, dtype=torch.bfloat16, device=device)
    values = {
        "gate": t.gate,
        "up": t.up,
        "dh": t.dh,
        "w": pkg._ignored_weight(device.index),
        "d_gate": d_gate,
        "d_up": d_up,
        "d_weight": torch.empty(t.R, dtype=torch.float32, device=device),
        "rows": t.R,
        "gate_stride": t.gate.stride(0),
        "up_stride": t.up.stride(0),
        "dh_stride": t.dh.stride(0),
        "d_gate_stride": d_gate.stride(0),
        "d_up_stride": d_up.stride(0),
        "limit": t.L,
        "weighted": 0,
    }
    entry = pkg._entry(name, pkg.device_arch(device.index))
    entry(*(values[key] for key in pkg.parameters(name)), *cake_jit.grid(name, t.R))
    return d_gate, d_up


def _gradients(d_first: torch.Tensor, d_up) -> tuple[torch.Tensor, torch.Tensor]:
    """``(d_gate, d_up)`` of a backward's result (a merged gradient split into its halves)."""
    return (d_first[:, :WIDTH], d_first[:, WIDTH:]) if d_up is None else (d_first, d_up)


@pytest.mark.parametrize("layout", ["split", "merged", "strided"])
@pytest.mark.parametrize("data", BITS_DATA)
def test_the_unweighted_backward_stores_the_bits_of_bwd(data, layout, device, programs_run) -> None:
    """An unweighted backward runs ``bwd_unweighted``, and its ``d_gate`` and ``d_up`` are, bit for bit (BF16 compared
    as int16, so NaN payloads and signed zeros count), those of the small-R and of the large-R ``bwd`` program each
    launched itself with the weight flag 0 (0.1.2's unweighted backward), and those of a weighted backward with weights
    of 1: every data kind, NaN and infinities, a gate of -88 in one third of a row only, limits 10, 7.3 and 1.5, 1 to 5
    rows (a partial last CTA of tasks), 129 and 4097 rows, and at and one row above each edge of the device's ``bwd``
    windows."""
    arch = pkg.device_arch(device.index)
    unweighted = cake_jit.PROGRAM_OF["bwd_unweighted"][arch]
    bwd_programs = list(dict.fromkeys(_program("bwd", device.index, program) for program in PROGRAMS))
    edges = _device_edges("bwd", device.index)
    row_counts = sorted({1, 2, 3, 4, 5, 129, 4097} | {edge + step for edge in edges for step in (0, 1)})
    for rows in row_counts:
        for limit in (10.0, 7.3, 1.5):
            t = _bits_inputs(data, layout, rows, limit, seed=rows, device=device)
            programs_run.clear()
            d_gate, d_up = _gradients(*backward(t.dh, t.first, t.second, None, limit=limit)[:2])
            assert programs_run == [unweighted], (rows, programs_run)
            what = f"{data}, {layout}, {rows} rows, L={limit}"
            for name in bwd_programs:
                got = _bwd_with_weight_flag_0(t, name, device)
                assert_same_bits(d_gate, got[0], f"d_gate of {name} with the weight flag 0: {what}")
                assert_same_bits(d_up, got[1], f"d_up of {name} with the weight flag 0: {what}")
            ones = torch.ones(rows, dtype=torch.float32, device=device)
            programs_run.clear()
            got = _gradients(*backward(t.dh, t.first, t.second, ones, limit=limit)[:2])
            assert programs_run == [_routed("bwd", rows, device.index)], (rows, programs_run)
            assert_same_bits(d_gate, got[0], f"d_gate of a weighted call with weights of 1: {what}")
            assert_same_bits(d_up, got[1], f"d_up of a weighted call with weights of 1: {what}")


@pytest.mark.parametrize("layout", ["merged", "split"])
def test_an_unweighted_backward_allocates_no_weight_gradient(layout, device, launches) -> None:
    """An unweighted backward launches ``bwd_unweighted`` with its 12 kernel parameters (no weight, weight flag or
    weight gradient) and allocates only its new gradients: none with preallocated gradients, one new merged gradient or
    two new split ones, with the same bits.  A weighted backward allocates its weight gradient as well."""
    t = make_inputs(Case(129, layout, "none"), seed=47, device=device)
    merged = t.second is None
    d_first = torch.empty(t.R, 2 * WIDTH if merged else WIDTH, dtype=torch.bfloat16, device=device)
    d_up = None if merged else torch.empty(t.R, WIDTH, dtype=torch.bfloat16, device=device)
    ones = torch.ones(t.R, dtype=torch.float32, device=device)
    want = backward(t.dh, t.first, t.second, None, limit=t.L)  # makes the launcher and loads the library
    backward(t.dh, t.first, t.second, ones, limit=t.L)

    def allocations() -> int:
        return torch.cuda.memory_stats(device)["allocation.all.allocated"]

    for preallocated in (True, False):
        launches.clear()
        torch.cuda.synchronize(device)
        before = allocations()
        if preallocated:
            got = backward(t.dh, t.first, t.second, None, limit=t.L, grad_gate_up_or_gate=d_first, grad_up=d_up)
        else:
            got = backward(t.dh, t.first, t.second, None, limit=t.L)
        allocated = allocations() - before
        assert allocated == (0 if preallocated else 1 if merged else 2), (preallocated, allocated)
        assert got[2] is None
        assert [(stage, rows, tuple(values)) for stage, rows, values in launches] == [
            ("bwd_unweighted", t.R, PARAMETERS["bwd_unweighted"])
        ]
        for name, found, expected in zip(("d_first", "d_up"), got[:2], want[:2], strict=True):
            assert (found is None) == (expected is None), name
            if found is not None:
                assert_same_bits(found, expected, f"{name}, preallocated={preallocated}")
    torch.cuda.synchronize(device)
    before = allocations()
    backward(t.dh, t.first, t.second, ones, limit=t.L)
    assert allocations() - before == (2 if merged else 3)


@pytest.mark.parametrize(
    "case", [Case(1), Case(129, "merged", data="mixed"), Case(4097, w="none")], ids=lambda case: case.id
)
def test_outputs_are_the_same_bits_at_every_call(case, device) -> None:
    t = make_inputs(case, seed=21, device=device)
    want = step_outputs(forward, backward, t)
    for k in range(4):
        assert_same_outputs(step_outputs(forward, backward, t), want, f"call {k + 1}")


@pytest.mark.parametrize("w", ["random", "none"])
def test_layouts_give_the_same_bits(w, device) -> None:
    """The same values as two contiguous tensors, one merged tensor, row slices of wider buffers (read in place) and
    rows the host copies give the same bits."""
    t = make_inputs(Case(257, "split", w, data="mixed"), seed=4, device=device)
    want = step_outputs(forward, backward, t)
    for layout in ("merged", "strided", "copied"):
        assert_same_outputs(step_outputs(forward, backward, _relayout(t, layout)), want, layout)


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("layout", ["split", "merged"])
def test_rows_do_not_depend_on_the_row_count(layout, w, device) -> None:
    """A row's outputs are the same bits whatever the other rows and the row count (each row is computed alone, on
    the sigmoid path its own gates select), weighted and unweighted."""
    t = make_inputs(Case(4097, layout, w, data="mixed"), seed=9, device=device)
    full = step_outputs(forward, backward, t)
    for rows in (1, 7, 129, 1000):
        part = step_outputs(forward, backward, _rows(t, rows))
        for name, value in part.items():
            assert_same_bits(value, full[name][:rows], f"{name} of the first {rows} rows")


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("rows", [98304, 196608])
def test_large_row_counts(rows, w, device) -> None:
    """The issue's row count and twice it, weighted and unweighted: finite outputs, the same bits at a second call, and
    the first and the last 4097 rows the bits of a call on those 4097 rows (views of the inputs, the last ones at a
    large row offset)."""
    t = make_inputs(Case(rows, "merged", w), seed=31, device=device)
    want = step_outputs(forward, backward, t)
    for name, value in want.items():
        assert bool(value.isfinite().all()), name
    assert_same_outputs(step_outputs(forward, backward, t), want, "second call")
    part = step_outputs(forward, backward, _rows(t, 4097))
    for name, value in part.items():
        assert_same_bits(value, want[name][:4097], f"{name} of the first 4097 rows")
    part = step_outputs(forward, backward, _rows(t, 4097, offset=rows - 4097))
    for name, value in part.items():
        assert_same_bits(value, want[name][rows - 4097 :], f"{name} of the last 4097 rows")


@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize("w", ["random", "none"])
def test_the_exact_sigmoid_gives_the_bits_of_the_fast_one(layout, w, program, device, monkeypatch) -> None:
    """In each stage's small-R and large-R program (forced through the route), on ordinary and boundary values whose
    clamped gates are all above -87 (the sigmoid's fast path everywhere), a gate of -88 at columns 0, 768 and 1536 of
    every row (one in each third: every run takes the exact path, the row of a one-run program and each run of 768
    columns of a three-run one) keeps the bits of ``h``, ``d_gate`` and ``d_up`` at every other column, and the value
    of ``d_weight`` (``dh`` is 0 at those columns in both inputs).  The exact path ran: at the gates of -88, ``h`` and
    ``d_gate`` (``dh`` 1 there) are nonzero in every row with a nonzero weight, where the fast path's sigmoid is 0 (its
    approximate reciprocal of ``1 + e^88`` flushes to 0)."""
    _force(monkeypatch, program)
    columns = list(SLOW_COLUMNS)
    t = make_inputs(Case(257, "split", w, data="boundary"), seed=17, device=device)
    assert bool((t.gate.float().clamp(max=t.L) > -87).all())
    t.dh[:, columns] = 0
    slow = types.SimpleNamespace(R=t.R, L=t.L, weight=t.weight, gate=t.gate.clone(), up=t.up, dh=t.dh)
    slow.gate[:, columns] = SLOW_GATE
    want = step_outputs(forward, backward, _relayout(t, layout))
    got = step_outputs(forward, backward, _relayout(slow, layout))
    other = torch.ones(WIDTH, dtype=torch.bool, device=device)
    other[columns] = False
    for name in ("h", "d_gate", "d_up"):
        assert_same_bits(got[name][:, other], want[name][:, other], f"{name} beside the gates of {SLOW_GATE}")
    if w != "none":
        assert bool((got["d_weight"] == want["d_weight"]).all())
    live = slice(None) if t.weight is None else t.weight != 0
    probe = types.SimpleNamespace(R=t.R, L=t.L, weight=t.weight, gate=slow.gate, up=t.up, dh=torch.zeros_like(t.dh))
    probe.dh[:, columns] = 1.0
    d_gate = step_outputs(forward, backward, _relayout(probe, layout))["d_gate"]
    assert bool((got["h"][:, columns][live] != 0).all()), "h at the gates of -88"
    assert bool((d_gate[:, columns][live] != 0).all()), "d_gate at the gates of -88"


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_h_is_the_fp32_weighted_activation_rounded_once(layout, device) -> None:
    """With ``dh`` 1 at one column ``c`` of each row and 0 elsewhere (``c`` in each third of the row, even and odd),
    ``d_weight`` is the FP32 activation ``a[r, c]`` (one product with 1, the others with 0; not every value of it a
    BF16 value), and ``h[r, c]`` is ``bf16(w[r] d_weight[r])`` bit for bit: the FP32 product of the weight and ``a``,
    rounded once, for weights that are not powers of two."""
    t = make_inputs(Case(257, layout, data="boundary"), seed=19, device=device)
    t.weight = _weights(t.R, seed=19, device=device)
    rows = torch.arange(t.R, device=device)
    columns = rows * 37 % WIDTH
    t.dh = torch.zeros(t.R, WIDTH, dtype=torch.bfloat16, device=device)
    t.dh[rows, columns] = 1.0
    found = step_outputs(forward, backward, t)
    assert bool((found["d_weight"].bfloat16().float() != found["d_weight"]).any()), "d_weight holds BF16 values only"
    want = (t.weight * found["d_weight"]).bfloat16()
    assert_same_bits(found["h"][rows, columns], want, "h at the column of dh = 1")


def test_no_bf16_intermediate_before_the_weight_or_the_sum(device) -> None:
    """On one draw of 2049 rows with weights that are not powers of two, the L2 error against the FP64 oracle of
    ``h``, ``d_gate`` and ``d_up`` is strictly below that of the weight applied after a BF16 rounding,
    ``bf16(w x)`` (the product in FP32), for ``x`` the unweighted call's output on the same inputs (the bits of a
    kernel that applies the weight after its BF16 rounding) and for ``x`` the unweighted FP64 value rounded to BF16;
    the L2 error of ``d_weight`` is strictly below that of ``sum over c of dh bf16(a)`` (``a`` in FP64)."""
    t = make_inputs(Case(2049, "split"), seed=23, device=device)
    t.weight = _weights(t.R, seed=23, device=device)
    found = step_outputs(forward, backward, t)
    want = oracle(t)
    unweighted = types.SimpleNamespace(
        R=t.R, L=t.L, first=t.first, second=t.second, gate=t.gate, up=t.up, dh=t.dh, weight=None
    )
    plain = step_outputs(forward, backward, unweighted)
    plain64 = oracle(unweighted)
    for name in ("h", "d_gate", "d_up"):
        for source, rounded in (("kernel", plain[name]), ("FP64", plain64[name].bfloat16())):
            late = (t.weight[:, None] * rounded.float()).bfloat16()
            (ours, theirs), _ = norm_errors(want[name], (found[name], late), "l2")
            assert ours < theirs, (name, source, ours, theirs)
    late = (t.dh.double() * plain64["h"].bfloat16().double()).sum(dim=1)
    (ours, theirs), _ = norm_errors(want["d_weight"], (found["d_weight"], late), "l2")
    assert ours < theirs, ("d_weight", ours, theirs)


@pytest.mark.parametrize(
    ("layout", "w"), [("merged", "random"), ("split", "random"), ("merged", "none"), ("split", "none")]
)
def test_autograd_matches_the_steps_and_checkpoint(layout, w, device) -> None:
    """The autograd function gives the steps' bits (a merged input's gradient is one ``[R, 4608]`` tensor), saves
    only its inputs (no activation, no FP32 intermediate), and gives the same bits under non-reentrant
    checkpointing."""
    t = make_inputs(Case(513, layout, w), seed=8, device=device)
    want = _functional(t)
    if layout == "merged":
        assert want["d_first"].shape == (t.R, 2 * WIDTH)
    assert_same_outputs(_autograd_step(t), want, "autograd")
    assert_same_outputs(_autograd_step(t, checkpoint=True), want, "checkpoint")

    saved = []

    def pack(x):
        saved.append((str(x.dtype), x.numel(), x.data_ptr()))
        return x

    leaves = _leaves(t)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        clamped_swiglu(*leaves, limit=t.L)
    expected = [(str(x.dtype), x.numel(), x.data_ptr()) for x in leaves if x is not None]
    assert sorted(saved) == sorted(expected)


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_the_autograd_backward_checks_the_saved_operands_again(layout, device) -> None:
    """The autograd backward reuses the forward's FP32 limit but checks the saved operands again, as a saved-tensors
    hook (or a ``.data`` assignment) can change them after the forward: unpacked as rows the kernels cannot read in
    place (a row stride that is not a multiple of 8 elements), they are copied and give the steps' bits; unpacked one
    row short, the backward raises ``ValueError`` before it launches."""
    t = make_inputs(Case(129, layout), seed=29, device=device)
    want = _functional(t)

    def relaid(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            return x
        buffer = torch.full((x.shape[0], x.shape[1] + 3), float("nan"), dtype=x.dtype, device=x.device)
        buffer[:, : x.shape[1]] = x
        return buffer[:, : x.shape[1]]

    def found(leaves: tuple, h: torch.Tensor) -> dict:
        got = {"h": h.detach(), "d_first": leaves[0].grad}
        if leaves[1] is not None:
            got["d_up"] = leaves[1].grad
        if leaves[2] is not None:
            got["d_weight"] = leaves[2].grad
        return got

    leaves = _leaves(t)
    with torch.autograd.graph.saved_tensors_hooks(lambda x: x, relaid):
        h = clamped_swiglu(*leaves, limit=t.L)
    h.backward(t.dh)
    assert_same_outputs(found(leaves, h), want, "saved operands unpacked as rows to copy")

    leaves = _leaves(t)
    with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x[:-1]):
        h = clamped_swiglu(*leaves, limit=t.L)
    with pytest.raises(ValueError):
        h.backward(t.dh)
    assert all(leaf is None or leaf.grad is None for leaf in leaves)


def test_autograd_returns_the_gradients_that_are_needed(device) -> None:
    """Only the inputs that require grad get one; ``limit`` gets none; an expanded upstream gradient (``h.sum()``)
    gives the bits of the steps with a gradient of ones."""
    t = make_inputs(Case(129, "split"), seed=12, device=device)
    want = _functional(t)
    gate = t.gate.detach().clone().requires_grad_(True)
    up, weight = t.up.detach().clone(), t.weight.detach().clone()
    clamped_swiglu(gate, up, weight, limit=t.L).backward(t.dh)
    assert_same_bits(gate.grad, want["d_first"], "d_gate alone")
    assert up.grad is None and weight.grad is None

    m = make_inputs(Case(129, "merged"), seed=13, device=device)
    leaves = _leaves(m)
    clamped_swiglu(*leaves, limit=m.L).sum().backward()
    ones = torch.ones(m.R, WIDTH, dtype=torch.bfloat16, device=device)
    d_first, _, d_weight = backward(ones, m.first, None, m.weight, limit=m.L)
    assert_same_bits(leaves[0].grad, d_first, "merged gradient of h.sum()")
    assert_same_bits(leaves[2].grad, d_weight, "weight gradient of h.sum()")


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_double_backward_raises(layout, device) -> None:
    """Gradients computed with ``create_graph=True`` have the steps' bits, and differentiating them raises
    ``RuntimeError``: with an upstream gradient that requires grad, and with a constant upstream gradient beside a
    second use of the input (where a check of the upstream gradient alone would let this operation's second-order
    terms drop silently)."""
    t = make_inputs(Case(33, layout), seed=3, device=device)
    want = _functional(t)
    leaves = _leaves(t)
    wants = [leaf for leaf in leaves if leaf is not None]
    grads = torch.autograd.grad(clamped_swiglu(*leaves, limit=t.L), wants, t.dh, create_graph=True)
    names = ["d_first", "d_weight"] if t.second is None else ["d_first", "d_up", "d_weight"]
    for name, grad in zip(names, grads, strict=True):
        assert grad.requires_grad, name
        assert_same_bits(grad.detach(), want[name], f"{name} with create_graph=True")

    leaves = _leaves(t)
    dh = t.dh.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(clamped_swiglu(*leaves, limit=t.L), leaves[0], dh, create_graph=True)
    with pytest.raises(RuntimeError, match="no double backward"):
        grad.float().sum().backward()

    leaves = _leaves(t)
    loss = clamped_swiglu(*leaves, limit=t.L).float().sum() + leaves[0].float().square().sum()
    (grad,) = torch.autograd.grad(loss, leaves[0], create_graph=True)
    with pytest.raises(RuntimeError, match="no double backward"):
        grad.float().sum().backward()


def test_side_stream_and_no_host_sync(device) -> None:
    """On a side stream under ``torch.cuda.set_sync_debug_mode("error")``: the steps and an autograd step run without
    synchronizing and give the bits of the default stream."""
    t = make_inputs(Case(2049, "merged"), seed=9, device=device)
    want = step_outputs(forward, backward, t)
    want_autograd = _autograd_step(t)
    leaves = _leaves(t)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            got = step_outputs(forward, backward, t)
            h = clamped_swiglu(*leaves, limit=t.L)
            h.backward(t.dh)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    assert_same_outputs(got, want, "side stream")
    assert_same_bits(h.detach(), want_autograd["h"], "autograd h on a side stream")
    assert_same_bits(leaves[0].grad, want_autograd["d_first"], "autograd gradient on a side stream")
    assert_same_bits(leaves[2].grad, want_autograd["d_weight"], "autograd weight gradient on a side stream")


@pytest.mark.parametrize("program", PROGRAMS)
def test_cuda_graph_capture_and_replay(program, device, programs_run, monkeypatch) -> None:
    """A weighted merged step and an unweighted two-tensor step (another limit), with preallocated outputs, on the
    small-R or the large-R program of each stage (forced through the route; ``bwd_unweighted`` has one), run once
    eagerly at the captured row count, captured once and replayed three times on new inputs copied into the captured
    tensors: the capture records the forced programs, and each replay gives the bits of eager calls on those
    inputs."""
    _force(monkeypatch, program)
    R = 300
    a = make_inputs(Case(R, "merged"), seed=40, device=device)
    b = make_inputs(Case(R, "split", "none", L=7.3), seed=41, device=device)
    gu, w, dh_a = a.gate_up.clone(), a.weight.clone(), a.dh.clone()
    gate, up, dh_b = b.gate.clone(), b.up.clone(), b.dh.clone()
    h_a = torch.empty(R, WIDTH, dtype=torch.bfloat16, device=device)
    d_gu = torch.empty(R, 2 * WIDTH, dtype=torch.bfloat16, device=device)
    d_w = torch.empty(R, dtype=torch.float32, device=device)
    h_b = torch.empty_like(h_a)
    d_gate, d_up = torch.empty_like(h_a), torch.empty_like(h_a)

    def step():
        forward(gu, None, w, limit=a.L, out=h_a)
        backward(dh_a, gu, None, w, limit=a.L, grad_gate_up_or_gate=d_gu, grad_weight=d_w)
        forward(gate, up, None, limit=b.L, out=h_b)
        backward(dh_b, gate, up, None, limit=b.L, grad_gate_up_or_gate=d_gate, grad_up=d_up)

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        step()  # loads the forced programs' libraries before the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    programs_run.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    stages = (*_stages(True), *_stages(False))
    assert programs_run == [_program(stage, device.index, program) for stage in stages], program
    for k in range(3):
        na = make_inputs(Case(R, "merged"), seed=50 + k, device=device)
        nb = make_inputs(Case(R, "split", "none", L=7.3), seed=60 + k, device=device)
        gu.copy_(na.gate_up)
        w.copy_(na.weight)
        dh_a.copy_(na.dh)
        gate.copy_(nb.gate)
        up.copy_(nb.up)
        dh_b.copy_(nb.dh)
        for buffer in (h_a, d_gu, d_w, h_b, d_gate, d_up):
            buffer.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        want_a = _functional(na)
        want_b = _functional(nb)
        assert_same_bits(h_a, want_a["h"], f"replay {k}: h (merged)")
        assert_same_bits(d_gu, want_a["d_first"], f"replay {k}: d_gate_up")
        assert_same_bits(d_w, want_a["d_weight"], f"replay {k}: d_weight")
        assert_same_bits(h_b, want_b["h"], f"replay {k}: h (unweighted)")
        assert_same_bits(d_gate, want_b["d_first"], f"replay {k}: d_gate")
        assert_same_bits(d_up, want_b["d_up"], f"replay {k}: d_up")


@pytest.mark.parametrize("layout", ["merged", "split"])
def test_an_unweighted_step_captured_before_any_eager_unweighted_call(layout, device, launches) -> None:
    """An unweighted forward and backward with preallocated outputs, captured in a CUDA graph with no unweighted call
    in the process before (only a weighted call and an unweighted backward, which load the libraries; the ignored
    weight's cache is cleared): the capture makes the device's cached FP32 buffer, the captured forward passes it as
    its ignored weight and the captured backward runs ``bwd_unweighted``, which takes no weight, weight flag or weight
    gradient; later eager unweighted forwards pass the same buffer, and each replay on new inputs copied into the
    captured tensors gives the bits of eager calls."""
    R = 129
    step_outputs(forward, backward, make_inputs(Case(R, layout), seed=70, device=device))
    warm = make_inputs(Case(R, layout, "none"), seed=69, device=device)
    backward(warm.dh, warm.first, warm.second, None, limit=warm.L)  # loads the unweighted backward's library
    a = make_inputs(Case(R, layout, "none"), seed=71, device=device)
    first = a.first.clone()
    second = None if a.second is None else a.second.clone()
    dh = a.dh.clone()
    h = torch.empty(R, WIDTH, dtype=torch.bfloat16, device=device)
    d_first = torch.empty(R, WIDTH if second is not None else 2 * WIDTH, dtype=torch.bfloat16, device=device)
    d_up = None if second is None else torch.empty(R, WIDTH, dtype=torch.bfloat16, device=device)
    pkg._ignored_weight.cache_clear()
    launches.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(first, second, None, limit=a.L, out=h)
        backward(dh, first, second, None, limit=a.L, grad_gate_up_or_gate=d_first, grad_up=d_up)
    assert [stage for stage, _, _ in launches] == ["fwd", "bwd_unweighted"]
    cached = pkg._ignored_weight(first.get_device())
    assert cached.dtype == torch.float32 and tuple(cached.shape) == (1,) and cached.device == first.device
    (_, _, fwd_values), (_, _, bwd_values) = launches
    assert fwd_values["weighted"] == 0 and fwd_values["w"] is cached
    assert tuple(bwd_values) == PARAMETERS["bwd_unweighted"]
    launches.clear()
    forward(first, second, None, limit=a.L)
    backward(dh, first, second, None, limit=a.L)
    assert [stage for stage, _, _ in launches] == ["fwd", "bwd_unweighted"]
    assert launches[0][2]["w"] is cached and "w" not in launches[1][2]
    for k in range(3):
        n = make_inputs(Case(R, layout, "none"), seed=80 + k, device=device)
        first.copy_(n.first)
        if second is not None:
            second.copy_(n.second)
        dh.copy_(n.dh)
        for buffer in (h, d_first, d_up):
            if buffer is not None:
                buffer.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
        want = _functional(n)
        assert_same_bits(h, want["h"], f"replay {k}: h")
        assert_same_bits(d_first, want["d_first"], f"replay {k}: d_first")
        if d_up is not None:
            assert_same_bits(d_up, want["d_up"], f"replay {k}: d_up")


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_steps_neither_raise_nor_warn_under_deterministic_algorithms(warn_only, device) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) with warnings as errors, the
    steps, an autograd step and a non-reentrant checkpointed step raise nothing, warn nothing and give the bits of the
    same calls with the setting off."""
    t = make_inputs(Case(1031, "merged"), seed=35, device=device)
    u = make_inputs(Case(77, "split", "none"), seed=36, device=device)

    def run():
        return [
            step_outputs(forward, backward, t),
            step_outputs(forward, backward, u),
            _autograd_step(t),
            _autograd_step(t, checkpoint=True),
        ]

    want = run()
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = run()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_outputs(a, b, f"warn_only={warn_only} call {i}")


@pytest.mark.parametrize("layout", ["merged", "split", "copied"])
@pytest.mark.parametrize("w", ["random", "none"])
def test_new_outputs_are_filled_under_deterministic_algorithms(layout, w, device, monkeypatch) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` with ``torch.utils.deterministic.fill_uninitialized_memory``,
    every output a step allocates holds NaN when its kernel launches (PyTorch filled it): ``h``; ``d_gate_up`` or
    ``d_gate`` and ``d_up``; a weighted call's ``d_weight`` (an unweighted backward has none).  Every call allocates its
    outputs with ``torch.empty`` on the operands' device; the outputs have the bits of the same calls with the setting
    off."""
    outputs = {"fwd": ("h",), "bwd": ("d_gate", "d_up", "d_weight"), "bwd_unweighted": ("d_gate", "d_up")}
    seen: list = []

    def record(name, values):
        stage = cake_jit.MODULES[name]["stage"]
        for key in outputs[stage]:
            seen.append((stage, key, bool(torch.isnan(values[key]).all())))

    t = make_inputs(Case(77, layout, w), seed=37, device=device)
    want = step_outputs(forward, backward, t)
    _spy_entries(monkeypatch, record)
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        got = step_outputs(forward, backward, t)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    assert seen == [(stage, key, True) for stage in _stages(w != "none") for key in outputs[stage]], seen
    assert_same_outputs(got, want, f"{layout}, w={w}")


def test_fit_inputs_are_read_in_place_and_others_copied_once(device, launches) -> None:
    """The halves of a merged input and row slices of wider buffers reach the kernels as views of the caller's memory
    (their row strides); rows the kernels cannot read are copied once to contiguous rows; a single row's stride is
    passed as 0."""
    m = make_inputs(Case(33, "merged"), seed=2, device=device)
    forward(m.first, None, m.weight, limit=L)
    _, rows, values = launches[-1]
    assert rows == 33
    assert values["gate"].data_ptr() == m.gate_up.data_ptr()
    assert values["up"].data_ptr() == m.gate_up.data_ptr() + 2 * WIDTH
    assert values["gate_stride"] == values["up_stride"] == 2 * WIDTH
    assert values["w"] is m.weight and values["weighted"] == 1
    assert values["limit"] == L and values["h_stride"] == WIDTH

    s = make_inputs(Case(33, "strided"), seed=2, device=device)
    backward(s.dh, s.gate, s.up, s.weight, limit=L)
    _, _, values = launches[-1]
    for name, tensor in (("gate", s.gate), ("up", s.up), ("dh", s.dh)):
        assert values[name].data_ptr() == tensor.data_ptr(), name
        assert values[f"{name}_stride"] == tensor.stride(0), name

    c = make_inputs(Case(33, "copied", "none", L=7.3), seed=2, device=device)
    backward(c.dh, c.gate, c.up, None, limit=7.3)
    _, _, values = launches[-1]
    for name, tensor in (("gate", c.gate), ("up", c.up), ("dh", c.dh)):
        assert values[name].data_ptr() != tensor.data_ptr(), name
        assert values[name].is_contiguous() and values[f"{name}_stride"] == WIDTH, name
    assert tuple(values) == PARAMETERS["bwd_unweighted"]  # no weight, weight flag or weight gradient
    assert values["limit"] == fp32(7.3)

    one = make_inputs(Case(1, "merged"), seed=2, device=device)
    backward(one.dh, one.first, None, one.weight, limit=L)
    _, _, values = launches[-1]
    for name in ("gate", "up", "dh", "d_gate", "d_up"):
        assert values[f"{name}_stride"] == 0, name


def _launch_facts(launched: list, caller: set[int]) -> list:
    """The launches ``(stage, rows, values)`` with each tensor value as its dtype, shape, strides, storage offset and
    memory: the caller's storage (its address, one of ``caller``) or ``"new"`` (a copy or a new output)."""
    facts = []
    for stage, rows, values in launched:
        found = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage().data_ptr()
                value = (
                    value.dtype,
                    tuple(value.shape),
                    value.stride(),
                    value.storage_offset(),
                    storage if storage in caller else "new",
                )
            found[name] = value
        facts.append((stage, rows, found))
    return facts


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("rows", [1, 33, 4097])
def test_one_pass_checks_give_the_launches_of_the_full_checks(rows, w, device, launches, monkeypatch) -> None:
    """With the one-pass checks of the common call on and off (``_FAST_CHECKS``), every layout (merged, two tensors,
    row slices of wider buffers, rows the host copies), with new and with preallocated outputs, gives the same launches
    (each tensor's dtype, shape, strides, storage offset and memory, the caller's or new; every scalar) and the same
    output bits.  With them on, a merged or two-tensor call of contiguous tensors runs no full check of any tensor."""
    checked: list = []
    check_tensor = pkg._check_tensor

    def counted(tensor, name, *args):
        checked.append(name)
        return check_tensor(tensor, name, *args)

    monkeypatch.setattr(pkg, "_check_tensor", counted)
    t = make_inputs(Case(rows, "split", w, data="mixed"), seed=27, device=device)
    for layout in ("merged", "split", "strided", "copied"):
        x = _relayout(t, layout)
        merged = x.second is None
        for preallocated in (False, True):
            h = d_first = d_up = d_weight = None
            if preallocated:
                h = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
                d_first = torch.empty(rows, 2 * WIDTH if merged else WIDTH, dtype=torch.bfloat16, device=device)
                d_up = None if merged else torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
                d_weight = None if x.weight is None else torch.empty(rows, device=device)
            tensors = (x.first, x.second, x.weight, x.dh, h, d_first, d_up, d_weight)
            caller = {tensor.untyped_storage().data_ptr() for tensor in tensors if tensor is not None}
            runs = []
            for fast in (True, False):
                monkeypatch.setattr(pkg, "_FAST_CHECKS", fast)
                launches.clear()
                checked.clear()
                found = [forward(x.first, x.second, x.weight, limit=t.L, out=h)]
                found += backward(
                    x.dh,
                    x.first,
                    x.second,
                    x.weight,
                    limit=t.L,
                    grad_gate_up_or_gate=d_first,
                    grad_up=d_up,
                    grad_weight=d_weight,
                )
                outputs = [None if tensor is None else tensor.clone() for tensor in found]
                runs.append((_launch_facts(launches, caller), outputs))
                if fast and layout in ("merged", "split"):
                    assert not checked, (layout, preallocated, checked)
                elif not fast:
                    assert checked, (layout, preallocated)
            what = f"{layout}, preallocated={preallocated}"
            (fast_launches, fast_outputs), (full_launches, full_outputs) = runs
            assert [stage for stage, _, _ in fast_launches] == list(_stages(w != "none")), what
            assert fast_launches == full_launches, what
            for got, want in zip(fast_outputs, full_outputs, strict=True):
                assert (got is None) == (want is None), what
                if got is not None:
                    assert_same_bits(got, want, what)


def test_preallocated_outputs(device) -> None:
    """Preallocated outputs, contiguous or row slices of wider aligned buffers, receive the bits of new outputs and are
    returned; outputs the kernels cannot write in place are refused, never copied."""
    t = make_inputs(Case(65, "merged"), seed=14, device=device)
    want = _functional(t)

    def wide(rows, cols, pad=8):
        return torch.empty(rows, cols + 2 * pad, dtype=torch.bfloat16, device=device)[:, pad : pad + cols]

    def new(rows, cols):
        return torch.empty(rows, cols, dtype=torch.bfloat16, device=device)

    for h, d_gu in ((new(t.R, WIDTH), new(t.R, 2 * WIDTH)), (wide(t.R, WIDTH), wide(t.R, 2 * WIDTH))):
        d_w = torch.empty(t.R, device=device)
        assert forward(t.first, None, t.weight, limit=L, out=h) is h
        result = backward(t.dh, t.first, None, t.weight, limit=L, grad_gate_up_or_gate=d_gu, grad_weight=d_w)
        assert result[0] is d_gu and result[1] is None and result[2] is d_w
        assert_same_bits(h, want["h"], "out")
        assert_same_bits(d_gu, want["d_first"], "grad_gate_up_or_gate")
        assert_same_bits(d_w, want["d_weight"], "grad_weight")

    s = make_inputs(Case(65, "split", "none"), seed=15, device=device)
    want_s = _functional(s)
    d_gate, d_up = wide(s.R, WIDTH), wide(s.R, WIDTH)
    result = backward(s.dh, s.gate, s.up, None, limit=L, grad_gate_up_or_gate=d_gate, grad_up=d_up)
    assert result[0] is d_gate and result[1] is d_up and result[2] is None
    assert_same_bits(d_gate, want_s["d_first"], "grad_gate")
    assert_same_bits(d_up, want_s["d_up"], "grad_up")

    raw = torch.empty(t.R * 2 * WIDTH + 64, dtype=torch.bfloat16, device=device)
    unfit = {
        "an FP32 output": torch.empty(t.R, WIDTH, device=device),
        "another shape": torch.empty(t.R + 1, WIDTH, dtype=torch.bfloat16, device=device),
        "a base that is not 16-byte aligned": raw[1 : 1 + t.R * WIDTH].view(t.R, WIDTH),
        "a row stride that is not a multiple of 8": raw[: t.R * (WIDTH + 2)].view(t.R, WIDTH + 2)[:, :WIDTH],
        "overlapping rows": raw.as_strided((t.R, WIDTH), (8, 1)),
        "a non-unit last stride": raw[: t.R * WIDTH * 2].view(t.R, WIDTH, 2)[:, :, 0],
    }
    for what, out in unfit.items():
        with pytest.raises(ValueError):
            forward(t.first, None, t.weight, limit=L, out=out)
        with pytest.raises(ValueError):
            backward(s.dh, s.gate, s.up, None, limit=L, grad_gate_up_or_gate=out)
        with pytest.raises(ValueError):
            backward(s.dh, s.gate, s.up, None, limit=L, grad_up=out)
    with pytest.raises(ValueError, match="grad_up"):
        backward(t.dh, t.first, None, t.weight, limit=L, grad_up=torch.empty_like(t.dh))
    with pytest.raises(ValueError, match="grad_weight"):
        backward(s.dh, s.gate, s.up, None, limit=L, grad_weight=torch.empty(s.R, device=device))
    for bad in (
        torch.empty(t.R, dtype=torch.bfloat16, device=device),
        torch.empty(t.R + 1, device=device),
        torch.empty(t.R, 2, device=device)[:, 0],
    ):
        with pytest.raises(ValueError):
            backward(t.dh, t.first, None, t.weight, limit=L, grad_weight=bad)


def test_operand_and_limit_errors(device) -> None:
    """Wrong types, dtypes, shapes and devices, and limits that are not finite positive Python numbers, are refused
    before any launch."""
    t = make_inputs(Case(8, "split"), seed=1, device=device)
    gate, up, weight, dh = t.gate, t.up, t.weight, t.dh
    value_errors = {
        "an FP16 gate": lambda: forward(gate.half(), up, weight, limit=L),
        "an FP32 up": lambda: forward(gate, up.float(), weight, limit=L),
        "a BF16 weight": lambda: forward(gate, up, weight.bfloat16(), limit=L),
        "a weight of another length": lambda: forward(gate, up, weight[:7], limit=L),
        "a 2-D weight": lambda: forward(gate, up, weight[:, None], limit=L),
        "another width": lambda: forward(gate[:, :2048], up[:, :2048], weight, limit=L),
        "a [R, 2304] tensor as the merged input": lambda: forward(gate, None, weight, limit=L),
        "another row count of up": lambda: forward(gate, up[:7], weight, limit=L),
        "a 3-D gate": lambda: forward(gate[None], up[None], weight, limit=L),
        "up on the CPU": lambda: forward(gate, up.cpu(), weight, limit=L),
        "a weight on the CPU": lambda: forward(gate, up, weight.cpu(), limit=L),
        "dh of another shape": lambda: backward(dh[:, :100], gate, up, weight, limit=L),
        "an FP32 dh": lambda: backward(dh.float(), gate, up, weight, limit=L),
        "a NaN limit": lambda: forward(gate, up, weight, limit=float("nan")),
        "an infinite limit": lambda: forward(gate, up, weight, limit=float("inf")),
        "a zero limit": lambda: forward(gate, up, weight, limit=0.0),
        "a negative limit": lambda: forward(gate, up, weight, limit=-1.0),
        "a limit past FP32": lambda: forward(gate, up, weight, limit=1e39),
        "a limit that rounds to 0 in FP32": lambda: forward(gate, up, weight, limit=1e-50),
        "a negative limit in the backward": lambda: backward(dh, gate, up, weight, limit=-10.0),
        "an integer limit past every float": lambda: forward(gate, up, weight, limit=10**400),
        "a fraction limit past every float": lambda: forward(gate, up, weight, limit=fractions.Fraction(10**400)),
        "an integer limit past every float in the backward": lambda: backward(dh, gate, up, weight, limit=10**400),
    }
    type_errors = {
        "a tensor limit": lambda: forward(gate, up, weight, limit=torch.tensor(10.0, device=device)),
        "a boolean limit": lambda: forward(gate, up, weight, limit=True),
        "a string limit": lambda: forward(gate, up, weight, limit="10"),
        "no limit": lambda: forward(gate, up, weight),
        "a list gate": lambda: forward([1.0], up, weight, limit=L),
        "a tensor limit in the autograd entry": lambda: clamped_swiglu(gate, up, weight, limit=torch.tensor(10.0)),
    }
    misses = []
    for kind, calls in ((ValueError, value_errors), (TypeError, type_errors)):
        for what, call in calls.items():
            try:
                call()
            except kind:
                continue
            except Exception as e:
                misses.append(f"{what}: {type(e).__name__} ({e}), not {kind.__name__}")
            else:
                misses.append(f"{what}: no {kind.__name__}")
    assert not misses, misses
    # An integer limit is a Python number: accepted, the same bits as the float.
    assert_same_bits(forward(gate, up, weight, limit=10), forward(gate, up, weight, limit=10.0), "integer limit")


def test_refused_calls_copy_allocate_and_launch_nothing(device, launches, monkeypatch) -> None:
    """Beside inputs the kernels cannot read (row strides that are not multiples of 8 elements, copied when the call is
    valid), a bad argument of each kind (the limit, a dtype, a shape, a device, an output buffer, the ``grad_up`` and
    ``grad_weight`` rules, the row count, the architecture) is refused before the call copies an input, allocates an
    output or launches a kernel: the caching allocator counts no allocation and nothing launches.  The same inputs in a
    valid call are copied and launched."""
    R = 33
    t = make_inputs(Case(R, "copied"), seed=3, device=device)
    gate, up, weight, dh = t.gate, t.up, t.weight, t.dh
    merged = torch.zeros(R, 2 * WIDTH + 3, dtype=torch.bfloat16, device=device)[:, : 2 * WIDTH]
    up32 = torch.zeros(R, WIDTH, device=device)
    weight_cpu = weight.cpu()
    out = torch.empty(R + 1, WIDTH, dtype=torch.bfloat16, device=device)
    grad_up = torch.empty(R, WIDTH, dtype=torch.bfloat16, device=device)
    grad_weight = torch.empty(R, device=device)
    tensor_limit = torch.tensor(L)
    # R past the kernels' row count, as stride-0 rows: a gate read in place, and an up and a merged input that start
    # 2 bytes past an aligned allocation (copied if the row count were valid).
    many = cake_jit.MAX_ROWS
    huge_gate = torch.zeros(1, WIDTH, dtype=torch.bfloat16, device=device).expand(many, WIDTH)
    huge_up = torch.zeros(1, WIDTH + 1, dtype=torch.bfloat16, device=device)[:, 1:].expand(many, WIDTH)
    huge_merged = torch.zeros(1, 2 * WIDTH + 1, dtype=torch.bfloat16, device=device)[:, 1:].expand(many, 2 * WIDTH)
    calls = {
        "a negative limit": (ValueError, lambda: forward(gate, up, weight, limit=-1.0)),
        "a NaN limit in the backward": (ValueError, lambda: backward(dh, gate, up, weight, limit=float("nan"))),
        "an integer limit past every float": (ValueError, lambda: forward(merged, None, weight, limit=10**400)),
        "a tensor limit": (TypeError, lambda: backward(dh, merged, None, weight, limit=tensor_limit)),
        "an FP32 up": (ValueError, lambda: forward(gate, up32, weight, limit=L)),
        "a weight of another length": (ValueError, lambda: forward(gate, up, weight[:-1], limit=L)),
        "a weight on the CPU": (ValueError, lambda: forward(merged, None, weight_cpu, limit=L)),
        "dh of another shape": (ValueError, lambda: backward(dh[:, :100], gate, up, weight, limit=L)),
        "an out of another shape": (ValueError, lambda: forward(gate, up, weight, limit=L, out=out)),
        "grad_up with a merged input": (
            ValueError,
            lambda: backward(dh, merged, None, weight, limit=L, grad_up=grad_up),
        ),
        "grad_weight without a weight": (
            ValueError,
            lambda: backward(dh, gate, up, None, limit=L, grad_weight=grad_weight),
        ),
        "a grad_weight of another length": (
            ValueError,
            lambda: backward(dh, merged, None, weight, limit=L, grad_weight=grad_weight[:-1]),
        ),
        "too many rows": (ValueError, lambda: forward(huge_gate, huge_up, None, limit=L)),
        "too many rows, merged": (ValueError, lambda: backward(dh, huge_merged, None, None, limit=L)),
    }

    def allocations() -> int:
        return torch.cuda.memory_stats(device)["allocation.all.allocated"]

    def refuse(calls: dict) -> None:
        misses = []
        for what, (kind, call) in calls.items():
            try:
                call()
            except kind:
                continue
            except Exception as e:  # noqa: BLE001 - reported with the other misses
                misses.append(f"{what}: {type(e).__name__} ({e}), not {kind.__name__}")
            else:
                misses.append(f"{what}: no {kind.__name__}")
        assert not misses, misses

    torch.cuda.synchronize(device)
    before = allocations()
    refuse(calls)
    with monkeypatch.context() as patch:
        patch.setattr(pkg, "_arch_of_index", lambda index: None)
        refuse(
            {
                "another architecture": (RuntimeError, lambda: forward(gate, up, weight, limit=L)),
                "another architecture, merged": (RuntimeError, lambda: backward(dh, merged, None, None, limit=L)),
            }
        )
    assert allocations() == before, "a refused call allocated a copy or an output"
    assert launches == []
    forward(gate, up, weight, limit=L)
    backward(dh, merged, None, None, limit=L)
    assert allocations() > before
    assert [stage for stage, _, _ in launches] == ["fwd", "bwd_unweighted"]


def test_unsupported_devices_raise(device, monkeypatch) -> None:
    """A device of another architecture is a ``RuntimeError`` naming the delivered architectures, also with no rows,
    and is reported unsupported."""
    t = make_inputs(Case(5, "merged"), seed=1, device=device)
    monkeypatch.setattr(pkg, "_arch_of_index", lambda index: None)
    assert not pkg.is_cake_clamped_swiglu_det_train_supported(device)
    for rows in (0, 5):
        with pytest.raises(RuntimeError, match="sm_100a"):
            forward(t.first[:rows], None, t.weight[:rows], limit=L)
        with pytest.raises(RuntimeError, match="sm_100a"):
            backward(t.dh[:rows], t.first[:rows], None, t.weight[:rows], limit=L)


def test_operands_on_another_device(device) -> None:
    """Operands on a second GPU run there and give the bits of the first; mixed devices are refused."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", 1 if device.index != 1 else 0)
    if not pkg.is_cake_clamped_swiglu_det_train_supported(other):
        pytest.skip(f"{other} is not a supported device")
    t = make_inputs(Case(129, "merged"), seed=6, device=device)
    want = step_outputs(forward, backward, t)
    o = make_inputs(Case(129, "merged"), seed=6, device=other)
    got = step_outputs(forward, backward, o)
    for name, value in got.items():
        assert value.device == other, name
        assert_same_bits(value.to(device), want[name], f"{name} on {other}")
    with pytest.raises(ValueError):
        forward(t.first, None, o.weight, limit=L)


@pytest.mark.parametrize("case", [Case(129, "merged"), Case(129, "split", "none")], ids=lambda case: case.id)
def test_calls_after_set_device_run_on_their_operands_device(case, device) -> None:
    """The launchers are per stage and device, and the current device picks none of them: with the current device
    switched to the other GPU by ``torch.cuda.set_device`` after calls on both, the steps and an autograd step on the
    first GPU's operands run there (new outputs on that device, the bits of the calls before the switch), and the steps
    on the other GPU's operands run on the other GPU, before and after switching back.  Weighted and unweighted (the
    ignored weight of each device)."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", 1 if device.index != 1 else 0)
    if not pkg.is_cake_clamped_swiglu_det_train_supported(other):
        pytest.skip(f"{other} is not a supported device")
    t = make_inputs(case, seed=6, device=device)
    o = make_inputs(case, seed=6, device=other)
    want = step_outputs(forward, backward, t)
    want_autograd = _autograd_step(t)
    want_other = step_outputs(forward, backward, o)
    for stage in STAGES:
        assert pkg._launcher(stage, device.index) is not pkg._launcher(stage, other.index), stage
    current = torch.cuda.current_device()
    try:
        for switched in (other, device):
            torch.cuda.set_device(switched)
            got = step_outputs(forward, backward, t)
            got_autograd = _autograd_step(t)
            got_other = step_outputs(forward, backward, o)
            for found, expected, on in (
                (got, want, device),
                (got_autograd, want_autograd, device),
                (got_other, want_other, other),
            ):
                for name, value in found.items():
                    assert value.device == on, (switched, name)
                assert_same_outputs(found, expected, f"current device {switched}, operands on {on}")
    finally:
        torch.cuda.set_device(current)
