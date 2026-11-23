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

Without a device: the registry (per stage a small-R program on every architecture, a fine program with one window
``[0, hi]`` of rows per SM on every architecture, for ``fwd`` and ``bwd`` a large-R program with its windows of rows
per SM on some and for ``bwd`` a relay program with its windows; the unweighted backward ``bwd_unweighted`` with no
weight, weight flag or weight gradient; each record delivered for its architectures and for no other, the launch facts,
the argument plans), the route of a launch by its rows per SM (1 row, one row below, at and above each edge of the
windows and the most rows, on the SM counts of the delivered GPUs: the fine, the small-R, the large-R and the relay
program each running the row counts its windows hold, and the route of 0.1.9's code for the same records on an
architecture without a relay record; with synthetic windows of the large-R and the relay record of ``bwd``, the same;
windows that are not ascending, untouching lists of integers refused, windows of the large-R and the relay record that
overlap or touch refused, and a fine window that is not one window from 0, that reaches the first large-R or relay
window, that passes 64 rows per SM or whose grid passes ``2**31 - 1``; the small-R program at every row count on an
architecture without a fine, a large-R or a relay one; the SM count
read once per stage and device, no library loaded), the launch grid of each program (``ceil(3 R / 4)`` CTAs of
``bwd_unweighted`` up to the most rows, where ``3 R`` passes 32 bits; ``ceil(9 R / 4)`` CTAs of ``fwd_fine`` and
``bwd_unweighted_fine``, at most ``2**31 - 1``; ``R`` CTAs of 288 threads of ``bwd_fine`` and of 96 threads of a relay
program) and the arguments of a launch in its plan's order, the
entry points' launcher of a stage and device (made once per stage and device, no library loaded; the program, arguments
and grid of ``run_module`` at every edge row count, each program's FFI entry looked up once; refused for a program
that takes its parameters in another order), the cached FP32 rounding of the last limit (used only for a
float, or an int that a float holds exactly, equal to its float value; every other limit takes the full checks and their
errors), the build manifest (every record's library, the recorded source digests), the capability map, the required
``limit`` keyword, the refusal of CPU tensors and no environment read outside the JIT shim.

On SM100 / SM103 / SM107, every training output (``h``, ``d_gate``, ``d_up`` and ``d_weight``) of each stage's small-R,
large-R and fine program (each forced through the route) judged against the naive BF16 tier
(``tests/cake/test_helpers/cake_naive.py``: the error against an FP64 oracle of the same inputs over 32 independent
draws, in the L2 and the max norm, median and largest, beside the error of the same operation in BF16,
:mod:`tests.cake.test_helpers.clamped_swiglu_reference`) over row counts 1 to 4097, the merged, two-tensor, strided and
copied layouts, weighted and unweighted calls (the unweighted backward runs ``bwd_unweighted``), values on the clamp
boundary, deep negative gates, the edge of the FP32 sigmoid, a gate below -87 in each third of every row beside ordinary
and boundary values, every BF16 gate, a limit between two BF16 values and a small limit.  Beside the judge, exact facts
of the contract (no tolerance): the inclusive masks (a gradient is 0 exactly where its mask is 0 and passes at ``g = L``
and ``u = +-L``); zero-weight rows give exactly 0 in ``h``, ``d_gate`` and ``d_up`` and the weight gradient of any other
weight; the sigmoid's deep path (taken by a warp's run of loads with a clamped gate at or below -87) gives the bits of
its fast path wherever the clamped gate is above -87, in each program; ``d_weight`` sums the FP32 ``a``, and ``h`` is
the FP32 product ``w a`` rounded once to BF16; ``R = 0`` returns empty outputs and launches nothing; one library per
program serves every row count, limit, weight flag and layout; the programs of a stage give the same bits at and one row
above each edge of the device's windows, where the route runs the fine program up to its last row count, the large-R
program inside its windows and the small-R one elsewhere; each ``bwd_unweighted`` program stores the ``d_gate`` and
``d_up`` bits of every ``bwd`` program launched with the weight flag 0 and of a weighted call with weights of 1 (every
data kind, NaN and infinities, a gate of -88 in one third of a row, three layouts, three limits, 1 to 5 rows, 129, 4097
and the window edges of ``bwd`` and ``bwd_unweighted``), and an unweighted backward allocates no weight gradient; the
same bits at every call, in every layout, at every row count (a row's outputs do not depend on the other rows), on a
side stream with no host synchronization, in CUDA graph replays of the small-R, the large-R and the fine programs, under
``torch.use_deterministic_algorithms`` (where every new output holds PyTorch's NaN fill when its kernel launches) and
with the current device switched to another GPU (each call runs on its operands' device); the autograd function
matches the steps bit for bit, saves only its inputs and gives the same bits under non-reentrant checkpointing; in a
backward with saved-tensors hooks set (a non-reentrant checkpoint's
recomputation), ``cake_clamped_swiglu_det`` launches its forward after autograd saved the inputs (their pack hooks run
first; anywhere else, for a direct ``apply`` and while compiling or tracing, it launches inside ``forward``), so with
early stop on a non-reentrant region launches the forward again only when a later operation of the region saves a
tensor: none for the op alone or the gate/up projection and the op, one for the op with the down projection or the
whole block (gate/up projection, op, down projection), and one in every region with early stop off, a reentrant
region or a direct ``apply``, also in a captured CUDA graph, all with the bits of the steps without a checkpoint; the
autograd backward checks the saved operands again (unpacked by a saved-tensors hook as rows to copy: the steps' bits;
one row short: ``ValueError``), and gradients computed with ``create_graph=True`` have the steps' bits and raise when
differentiated; inputs the kernels can read are read in place (a merged input and its gradient passed as both
halves' tensors with the up's column offset 2304, no view), others copied once; every binding refuses a column offset
that is negative, not a multiple of 8 or past the row; unfit outputs, operands and limits are
refused, before any copy, allocation or launch, also when a step refuses a common call itself (its limit or its
architecture); every spied launch has the grid ``cake_jit.grid`` of its program; the one-pass checks of the common
call give the launches of the full checks (the program, the grid and every argument), and every call they leave to the
full checks (each refusal class, a tensor subclass, inputs to copy, no rows, a call that fails only one check of the
step's own, a tensor on another GPU) gives what it gives without them; a call with two faults raises the error of the
one the full checks find first; the steps and the autograd function (its backward also with ``create_graph=True``)
check and launch the common call themselves (no call of the full checks) and pass the row strides of its contiguous
tensors as constants; one launcher per stage and device, made once by the package's own cache of launchers, serves
every call; every unweighted forward launch of a device passes one cached FP32
buffer, set to 1, as its ignored weight, made by the first unweighted forward (no weighted call or backward makes it),
also when that forward is a CUDA graph capture.  On one fixed draw, the L2 error against the
FP64 oracle of ``h``, ``d_gate`` and ``d_up`` is below that of the weight applied after a BF16 rounding of each output,
and the L2 error of ``d_weight`` below that of the sum over the BF16-rounded ``a``.

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import contextlib
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
from collections.abc import Callable, Iterator
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
# The stages with a large-R program on some architecture (``bwd_unweighted`` has none).
WINDOWED = ("fwd", "bwd")
# Each stage's small-R role (the stage itself), its large-R role, its fine role and its relay role.
LARGE = "_large"
FINE = "_fine"
RELAY = "_relay"
# The suffixes of a stage's windowed roles, whose programs run in their windows above the fine window.
WINDOWED_ROLES = (LARGE, RELAY)
ROLES = (
    "fwd",
    "fwd_large",
    "fwd_fine",
    "fwd_relay",
    "bwd",
    "bwd_large",
    "bwd_fine",
    "bwd_relay",
    "bwd_unweighted",
    "bwd_unweighted_large",
    "bwd_unweighted_fine",
    "bwd_unweighted_relay",
)
# The SM counts of the delivered GPUs the routes are checked on.
SM_COUNTS = (148, 152, 212)
# The windows of rows per SM in which each stage's large-R, relay and fine programs run (the measured routes; ``None``:
# no upper bound), and the lane layout every program keeps: literal copies, so a record that drifts from them fails
# here, and the route tests expect these windows.
WINDOWS = {
    "fwd_large": {"sm_100a": [[112, None]], "sm_103a": [[208, None]], "sm_107a": [[384, None]]},
    "bwd_large": {"sm_100a": [[13, 21]], "sm_103a": [[14, 21]], "sm_107a": [[16, 21]]},
    "bwd_relay": {"sm_100a": [[3, 5], [25, None]], "sm_103a": [[12, 13], [25, 647]], "sm_107a": [[3, 6], [31, None]]},
    "fwd_fine": {"sm_100a": [[0, 8]], "sm_103a": [[0, 4]], "sm_107a": [[0, 4]]},
    "bwd_fine": {"sm_100a": [[0, 2]], "sm_103a": [[0, 2]], "sm_107a": [[0, 2]]},
    "bwd_unweighted_fine": {"sm_100a": [[0, 4]], "sm_103a": [[0, 4]], "sm_107a": [[0, 2]]},
}
LANE_LAYOUT = {"lanes": 32, "rows_per_thread": 1, "width_bits": 128}
# The build facts of each role with records: the lane layout, in ``bwd_unweighted`` and ``bwd_relay`` 3 tasks a row (a warp per (row, run)) and in a fine role 9 tasks a row (one load of each input a task).
BUILDS = {
    "fwd": LANE_LAYOUT,
    "fwd_large": LANE_LAYOUT,
    "fwd_fine": {**LANE_LAYOUT, "tasks_per_row": 9},
    "bwd": LANE_LAYOUT,
    "bwd_large": LANE_LAYOUT,
    "bwd_fine": {**LANE_LAYOUT, "tasks_per_row": 9},
    "bwd_relay": {**LANE_LAYOUT, "tasks_per_row": 3},
    "bwd_unweighted": {**LANE_LAYOUT, "tasks_per_row": 3},
    "bwd_unweighted_fine": {**LANE_LAYOUT, "tasks_per_row": 9},
}
# The program a forced launch runs: the small-R, the large-R or the fine program of its stage and, when a record
# ships one, its relay program (:func:`_force`).
PROGRAMS = ("small", "large", "fine", *(("relay",) if any(cake_jit.PROGRAM_OF[s + RELAY] for s in STAGES) else ()))
# The parameters of each stage's programs, in launch order: the kernel parameters (the kernels' signatures), then the
# column offsets of ``up`` (and ``d_up``) that the bindings add to those pointers.
PARAMETERS = {
    "fwd": ("gate", "up", "w", "h", "rows", "gate_stride", "up_stride", "h_stride", "limit", "weighted", "up_col"),
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
        "up_col",
        "d_up_col",
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
        "up_col",
        "d_up_col",
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
    """The test (or the ``patch`` context) gets its own cache of launchers (``pkg._launcher``, which every launch of
    the entry points reads, the common call's too): the launchers it makes use its patched route, FFI entries or
    device, hold their own FFI entries and ignored weight, and none of them outlives it or reaches another test."""
    patch.setattr(pkg, "_launcher", functools.cache(LAUNCHER))


def _spy_entries(patch: pytest.MonkeyPatch, record) -> None:
    """Every launch calls ``record(name, values, grid)`` before the FFI entry of its program ``name``, ``values``
    naming the program's parameters (the kernel's, then the column offsets) and ``grid`` its grid, which must be
    ``cake_jit.grid(name, rows)`` (an ``AssertionError`` out of the launch otherwise): each entry is wrapped as it is
    looked up, and the launchers are made afresh so that they bind the wrapped entries."""
    entry = pkg._entry

    def spy(name, arch):
        run, keys = entry(name, arch), pkg.parameters(name)

        def recorded(*args):
            values, grid = dict(zip(keys, args[: len(keys)], strict=True)), args[len(keys) :]
            assert grid == cake_jit.grid(name, values["rows"]), (name, values["rows"], grid)
            record(name, values, grid)
            return run(*args)

        return recorded

    patch.setattr(pkg, "_entry", spy)
    _fresh_launchers(patch)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list:
    """``(stage, rows, values)`` of every launch, in order: ``values`` names the parameters of the launch (the kernel's,
    then the column offsets)."""
    launched: list = []
    _spy_entries(
        monkeypatch,
        lambda name, values, grid: launched.append((cake_jit.MODULES[name]["stage"], values["rows"], values)),
    )
    return launched


@pytest.fixture
def launch_records(monkeypatch: pytest.MonkeyPatch) -> list:
    """``(name, rows, values, grid)`` of every launch, in order: the program that runs (the route), the parameters of
    the launch (``values``, as for ``launches``) and its grid."""
    launched: list = []
    _spy_entries(monkeypatch, lambda name, values, grid: launched.append((name, values["rows"], values, grid)))
    return launched


@pytest.fixture
def programs_run(monkeypatch: pytest.MonkeyPatch) -> list:
    """The program name of every launch, in order."""
    names: list = []
    _spy_entries(monkeypatch, lambda name, values, grid: names.append(name))
    return names


def _full_checks(patch: pytest.MonkeyPatch) -> list:
    """The names of the full-check functions each call runs, in order (each still runs): :func:`_forward` and
    :func:`_backward`, which a step calls for any call it does not check and launch itself, and :func:`_check_tensor`,
    which the full checks run on every tensor they check.  Empty after a call: the step checked and launched it."""
    called: list = []

    def counted(name: str):
        run = getattr(pkg, name)

        def call(*args, **kwargs):
            called.append(name)
            return run(*args, **kwargs)

        return call

    for name in ("_forward", "_backward", "_check_tensor"):
        patch.setattr(pkg, name, counted(name))
    return called


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
        _clear_routes()
        _fresh_launchers(monkeypatch)

    monkeypatch.setattr(pkg, "_entry", lambda name, arch: lambda *args: calls.append((name, arch, args)))
    _fresh_launchers(monkeypatch)
    _clear_routes()
    yield types.SimpleNamespace(place=place, calls=calls)
    _clear_routes()


def _clear_routes() -> None:
    """Empties the package's caches of the programs (``pkg._programs``) and the routes (``pkg._route``) of each stage
    and device."""
    pkg._programs.cache_clear()
    pkg._route.cache_clear()


def _plans(stage: str, device_index: int) -> dict[str, tuple]:
    """The plan of each program of ``stage`` on device ``device_index`` (``pkg._programs``) by kind: ``"small"``,
    ``"large"``, ``"relay"`` and ``"fine"``.  A kind without a program on the device's architecture is the small-R
    plan."""
    _, _, runs, fine = pkg._programs(stage, device_index)
    arch = pkg.device_arch(device_index)
    found = {run[0]: run for run in runs}
    plans = {"small": runs[0], "fine": fine}
    for kind, suffix in (("large", LARGE), ("relay", RELAY)):
        name = cake_jit.PROGRAM_OF[stage + suffix].get(arch)
        plans[kind] = runs[0] if name is None else found[name]
    return plans


def _force(patch: pytest.MonkeyPatch, program: str) -> None:
    """Every launch runs the ``program`` (``"small"``, ``"large"``, ``"relay"`` or ``"fine"``) of its stage on its
    device, whatever its row count: the route's limit becomes 0, so every launch runs its tail, that plan.  An
    architecture without such a program of a stage runs the small-R one either way.  The launchers are made afresh on
    the forced route."""

    def forced(stage: str, device_index: int) -> tuple:
        return 0, (), _plans(stage, device_index)[program]

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
    limit, plans, tail = pkg._route(stage, device_index)
    return (plans[rows] if rows < limit else tail)[0]


def _program(stage: str, device_index: int, program: str) -> str:
    """The ``program`` (``"small"``, ``"large"``, ``"relay"`` or ``"fine"``) of ``stage`` on device
    ``device_index``."""
    return _plans(stage, device_index)[program][0]


def _device_edges(stage: str, device_index: int) -> tuple[int, ...]:
    """The edges of the windows of the windowed programs of ``stage`` on device ``device_index``
    (:func:`cake_jit.route`)."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return cake_jit.route(stage, pkg.device_arch(device_index), sm_count)[0]


def _device_fine(stage: str, device_index: int) -> int:
    """The last row count of the fine window of ``stage`` on device ``device_index`` (:func:`cake_jit.fine`; 0
    without a fine program)."""
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    return cake_jit.fine(stage, pkg.device_arch(device_index), sm_count)[0]


def _record(role: str, arch: str) -> str | None:
    """The record of ``role`` delivered for ``arch`` (None: none)."""
    found = [name for name, record in cake_jit.MODULES.items() if record["role"] == role and arch in record["arches"]]
    assert len(found) <= 1, (role, arch, found)
    return found[0] if found else None


def _expected_windowed(
    stage: str, arch: str, table: dict | None = None, without: tuple[str, ...] = ()
) -> list[tuple[str, list[list[int | None]]]]:
    """``(record, windows)`` of each windowed role of ``stage`` (``WINDOWED_ROLES`` order, but the suffixes in
    ``without``) with a record on ``arch``: the role's windows of rows per SM on ``arch`` from ``table`` (role ->
    architecture -> windows; ``WINDOWS``, the measured routes, by default)."""
    table = WINDOWS if table is None else table
    found = []
    for suffix in WINDOWED_ROLES:
        name = _record(stage + suffix, arch)
        if name is not None and suffix not in without:
            found.append((name, table[stage + suffix][arch]))
    return found


def _expected_route(
    stage: str, arch: str, sm_count: int, table: dict | None = None, without: tuple[str, ...] = ()
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """``(edges, names)`` of ``stage`` on ``arch`` with ``sm_count`` SMs (:func:`_expected_windowed`): every window of
    the windowed records, in the order of their ``lo``; ``edges`` holds ``lo * sm_count - 1`` and ``hi * sm_count`` of
    each (an open window's first alone) and ``names`` the small-R record, then per window its record and, after a
    closed window, the small-R record again."""
    small = _record(stage, arch)
    spans = sorted(
        ((window, name) for name, windows in _expected_windowed(stage, arch, table, without) for window in windows),
        key=lambda span: span[0][0],
    )
    edges, names = [], [small]
    for (lo, hi), name in spans:
        edges.append(lo * sm_count - 1)
        names.append(name)
        if hi is not None:
            edges.append(hi * sm_count)
            names.append(small)
    return tuple(edges), tuple(names)


def _expected_fine(stage: str, arch: str, sm_count: int, without: tuple[str, ...] = ()) -> tuple[int, str]:
    """``(last, fine)`` of ``stage`` on ``arch``: the last row count of the window ``[0, hi]`` of its fine role on
    ``arch`` (``WINDOWS``), ``hi * sm_count``, and the fine record (none, or ``FINE`` in ``without``: 0 and the
    small-R record)."""
    small, fine = _record(stage, arch), _record(stage + FINE, arch)
    assert small is not None, (stage, arch)
    if fine is None or FINE in without:
        return 0, small
    ((lo, hi),) = WINDOWS[stage + FINE][arch]
    assert lo == 0 and hi >= 1, (stage, arch, lo, hi)
    return hi * sm_count, fine


def _expected_program(
    stage: str, arch: str, sm_count: int, rows: int, table: dict | None = None, without: tuple[str, ...] = ()
) -> str:
    """The record a launch of ``stage`` for ``rows`` rows runs on ``arch`` with ``sm_count`` SMs: the fine record up
    to its last row count, the windowed record whose window holds ``rows`` (at most one), the small-R record at every
    other row count."""
    last, fine = _expected_fine(stage, arch, sm_count, without)
    if rows <= last:
        return fine
    held = [
        name
        for name, windows in _expected_windowed(stage, arch, table, without)
        if _in_windows(windows, sm_count, rows)
    ]
    assert len(held) <= 1, (stage, arch, rows, held)
    return held[0] if held else _record(stage, arch)


def _route_edges(stage: str, arch: str, sm_count: int, table: dict | None = None) -> list[int]:
    """The row counts at which the route of ``stage`` changes program on ``arch`` with ``sm_count`` SMs, ascending: the
    fine window's last row count and the edges of the windows of the windowed records."""
    last, _ = _expected_fine(stage, arch, sm_count)
    return sorted({last, *_expected_route(stage, arch, sm_count, table)[0]} - {0})


def _route_019(stage: str, arch: str, sm_count: int) -> tuple[int, list[str], str]:
    """``(limit, names, tail)`` of the route of ``stage`` on ``arch`` as the 0.1.9 package makes it from the records'
    small-R, large-R and fine programs (its ``cake_jit.route``, ``cake_jit.fine`` and ``_route``, restated; no relay
    role): the program of each row count below ``limit``, and of every larger one."""
    small = cake_jit.PROGRAM_OF[stage][arch]
    large = cake_jit.PROGRAM_OF[stage + LARGE].get(arch, small)
    edges: tuple[int, ...] = ()
    if large != small:
        windows = cake_jit.MODULES[large]["rows_per_sm_windows"][arch]
        bounds = [bound for window in windows for bound in window if bound is not None]
        edges = tuple(bound * sm_count - (index % 2 == 0) for index, bound in enumerate(bounds))
    fine = cake_jit.PROGRAM_OF[stage + FINE].get(arch)
    last = 0 if fine is None else cake_jit.MODULES[fine]["rows_per_sm_windows"][arch][0][1] * sm_count
    limit = max(edges[-1] + 1 if edges else 0, last + 1)
    names = [small if fine is None else fine] * (last + 1) + [small] * (limit - last - 1)
    for below, top in zip(edges[0::2], edges[1::2]):
        names[below + 1 : top + 1] = [large] * (top - below)
    return limit, names, large if len(edges) % 2 else small


def _edge_rows(edges: list[int]) -> list[int]:
    """1 row, one row below, at and above each edge, and the most rows a launch takes: those that are launches."""
    near = {edge + step for edge in edges for step in (-1, 0, 1)}
    return sorted({rows for rows in {1, cake_jit.MAX_ROWS - 1} | near if 1 <= rows < cake_jit.MAX_ROWS})


# The name of the relay record of the route tests: the shipped one, whose windows the synthetic routes replace.
RELAY_NAME = cake_jit.PROGRAM_OF["bwd_relay"].get("sm_100a", f"{cake_jit.LIBRARY_PREFIX}_{'0' * 20}")


def _on(arches: tuple[str, ...], windows: list[list[int | None]]) -> dict[str, list[list[int | None]]]:
    """The same ``windows`` on each of ``arches``."""
    return {arch: windows for arch in arches}


# Windows of rows per SM of the ``bwd`` windowed records for the synthetic routes (role -> architecture -> windows), by
# name; the fine window stays the shipped ``[0, 2]``.  The shipped windows.  Without a relay record: the shipped
# large-R windows, 0.1.9's three large-R windows and other large-R windows (0.1.9's route).  With one: relay windows
# below, between and above the large-R ones, a relay window one row per SM above a large-R one, the large-R window last
# and open, a relay record alone, and a relay record on two architectures only (``sm_107a`` keeps the shipped large-R
# windows).
ROUTES = {
    "shipped": {"bwd_large": WINDOWS["bwd_large"], "bwd_relay": WINDOWS["bwd_relay"]},
    "large_shipped": {"bwd_large": WINDOWS["bwd_large"]},
    "large_019": {
        "bwd_large": {**_on(ARCHES[:2], [[12, 21], [32, 41], [48, None]]), "sm_107a": [[12, 21], [35, 41], [48, None]]}
    },
    "large_two": {"bwd_large": {**_on(ARCHES[:2], [[16, 20], [36, 40]]), "sm_107a": [[16, 20], [35, 36]]}},
    "large_three": {"bwd_large": _on(ARCHES, [[3, 11], [16, 20], [48, None]])},
    "relay_around": {"bwd_large": _on(ARCHES, [[16, 20]]), "bwd_relay": _on(ARCHES, [[3, 15], [21, None]])},
    "relay_between": {"bwd_large": _on(ARCHES, [[16, 20], [36, 40]]), "bwd_relay": _on(ARCHES, [[24, 32], [48, None]])},
    "relay_adjacent": {"bwd_large": _on(ARCHES, [[16, 20]]), "bwd_relay": _on(ARCHES, [[21, 26]])},
    "large_last": {"bwd_large": _on(ARCHES, [[48, None]]), "bwd_relay": _on(ARCHES, [[12, 15], [21, 40]])},
    "relay_alone": {"bwd_relay": _on(ARCHES, [[3, None]])},
    "relay_two_arches": {
        "bwd_large": {**_on(ARCHES[:2], [[16, 20], [36, 40]]), "sm_107a": WINDOWS["bwd_large"]["sm_107a"]},
        "bwd_relay": _on(ARCHES[:2], [[3, 15], [21, 35], [41, None]]),
    },
}


@pytest.fixture
def bwd_windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[dict], None]]:
    """``place(table)``: the registry with the ``bwd`` windows of ``table`` (``"bwd_large"`` and ``"bwd_relay"`` ->
    architecture -> windows of rows per SM): the ``bwd_large`` record on the architectures of its row, with those
    windows, and the ``bwd_relay`` record ``RELAY_NAME`` on the architectures of its row (the shipped relay record's
    plan, or without one the ``bwd_large`` record's plan with a block of 96 threads, a CTA of 3 warps per row); a role
    without a row has no record.  ``PROGRAM_OF`` is derived again, and the package's caches of plans, parameters and
    routes are emptied at each placement and after the test."""
    large = cake_jit.PROGRAM_OF["bwd_large"]["sm_100a"]
    base = cake_jit.MODULES[large]
    relay_base = cake_jit.MODULES.get(RELAY_NAME)

    def clear() -> None:
        pkg._plan.cache_clear()
        pkg.parameters.cache_clear()
        _clear_routes()

    def place(table: dict) -> None:
        for name, role in ((large, "bwd_large"), (RELAY_NAME, "bwd_relay")):
            rows = table.get(role, {})
            if not rows:
                monkeypatch.delitem(cake_jit.MODULES, name, raising=False)
                continue
            arches = [arch for arch in ARCHES if arch in rows]
            windows = {arch: rows[arch] for arch in arches}
            shipped = relay_base if role == "bwd_relay" and relay_base is not None else base
            record = {**shipped, "arches": arches, "role": role, "rows_per_sm_windows": windows}
            if role == "bwd_relay" and relay_base is None:
                launch = {**base["launch"], "block": (96, 1, 1)}
                record.update(kernel_symbol=f"kernel_{name}", library=name, launch=launch)
            monkeypatch.setitem(cake_jit.MODULES, name, record)
        monkeypatch.setattr(cake_jit, "PROGRAM_OF", cake_jit._programs())
        clear()

    yield place
    clear()


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
    assert tuple(cake_jit.ROLES) == ROLES and cake_jit.LARGE == LARGE and cake_jit.FINE == FINE
    assert cake_jit.RELAY == RELAY and tuple(cake_jit.WINDOWED) == WINDOWED_ROLES
    # Records keyed by their program names (their libraries), each in one role of its stage.  A role's records serve
    # disjoint architectures: the stage's role every architecture, its large-R, relay and fine roles some (or none),
    # each large-R, relay or fine record with the windows of rows per SM in which it runs on each of its architectures
    # (a fine record one window from 0).  The host reaches them through PROGRAM_OF.
    programs: dict = {role: {} for role in ROLES}
    for name, record in cake_jit.MODULES.items():
        role = record["role"]
        assert role in ROLES, name
        assert record["stage"] == role.removesuffix(LARGE).removesuffix(FINE).removesuffix(RELAY), name
        arches = list(record["arches"])
        assert arches and arches == [arch for arch in ARCHES if arch in arches], name
        for arch in arches:
            assert arch not in programs[role], (name, role, arch)
            programs[role][arch] = name
        if role.endswith((LARGE, FINE, RELAY)):
            windows = record["rows_per_sm_windows"]
            assert set(windows) == set(arches), name
            for arch, row in windows.items():
                bounds = [bound for window in row for bound in window]
                assert row and all(type(window) is list and len(window) == 2 for window in row), (name, arch)
                assert all(type(bound) is int for bound in bounds[:-1]), (name, arch)
                assert bounds[-1] is None or type(bounds[-1]) is int, (name, arch)
                bounds = [bound for bound in bounds if bound is not None]
                assert bounds[0] >= 1 or role.endswith(FINE), (name, arch)
                assert bounds == sorted(set(bounds)), (name, arch)
                if role.endswith(FINE):
                    assert len(row) == 1 and row[0][0] == 0 and type(row[0][1]) is int and row[0][1] >= 1, (name, arch)
        else:
            assert "rows_per_sm_windows" not in record, name
    for stage in STAGES:
        assert set(programs[stage]) == set(ARCHES), stage
    assert cake_jit.PROGRAM_OF == programs
    # Every role with a record has its build facts.
    roles = {record["role"] for record in cake_jit.MODULES.values()}
    assert roles <= set(cake_jit.BUILD_OF) == set(BUILDS) <= set(ROLES)
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
        build = cake_jit.BUILD_OF[record["role"]]
        assert tuple(launch["block"]) == (threads, 1, 1) and threads % 32 == 0, name
        assert threads % build["lanes"] == 0, name
        per_cta = threads // build["lanes"] * build["rows_per_thread"]
        assert cake_jit.cta_tasks(name) == (per_cta, build.get("tasks_per_row", 1)), name
        # a relay program runs one row a CTA (its kernel's grid is one CTA per row)
        assert not record["role"].endswith(RELAY) or cake_jit.cta_tasks(name) == (3, 3), name
    # The BF16 stride and alignment rule is one load of the widest build: 8 values and 16 bytes for 128-bit loads.
    widest = max(build["width_bits"] for build in cake_jit.BUILD_OF.values())
    assert cake_jit.VECTOR_ELEMENTS == widest // 16 and cake_jit.ALIGNMENT_BYTES == widest // 8


def test_the_windows_and_the_lane_layout_are_the_measured_ones() -> None:
    # every record's windows, by role
    windows: dict = {}
    for name, record in cake_jit.MODULES.items():
        for arch, row in record.get("rows_per_sm_windows", {}).items():
            assert arch not in windows.setdefault(record["role"], {}), (name, arch)
            windows[record["role"]][arch] = row
    assert windows == WINDOWS
    layouts = {role: dict(build) for role, build in cake_jit.BUILD_OF.items()}
    assert layouts == BUILDS


def test_argument_plans_are_the_kernel_parameters_then_the_column_offsets() -> None:
    """Each record takes its kernel's parameters in order (tensors as buffers, scalars as parameters), then the
    column offsets of ``up`` (and ``d_up``) as parameters, then the grid; the host names every one of them."""
    for name, record in cake_jit.MODULES.items():
        assert pkg.parameters(name) == PARAMETERS[record["stage"]], name
        for kind, key in record["arg_plan"]:
            if kind == "grid":
                continue
            assert kind == ("buffer" if key in BUFFERS else "parameter"), (name, kind, key)
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == ["grid_x", "grid_y", "grid_z"]


def test_grid_is_one_cta_per_block_of_tasks() -> None:
    """A CTA owns ``threads / lanes * rows_per_thread`` tasks and a row is ``tasks_per_row`` tasks (a row in the
    small-R and large-R ``fwd`` and ``bwd`` programs, 3 (row, run) tasks in the small-R ``bwd_unweighted`` one, 9
    tasks in a fine program): the grid is ``ceil(rows * tasks_per_row / per_cta)`` CTAs, at most ``2**31 - 1``
    (``cake_jit.MAX_GRID``) at the most rows (a count only for ``bwd_fine``, which takes at most ``2**30`` rows on its
    grid of ``R`` CTAs)."""
    assert cake_jit.MAX_GRID == 2**31 - 1
    for name in cake_jit.MODULES:
        per_cta, per_row = cake_jit.cta_tasks(name)
        for rows in (1, 2, per_cta - 1, per_cta, per_cta + 1, 129, 4097, 98304, 196608, cake_jit.MAX_ROWS - 1):
            if rows >= 1:
                want = min(-(-rows * per_row // per_cta), 2**31 - 1)
                assert tuple(cake_jit.grid(name, rows)) == (want, 1, 1), (name, rows)
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
    """``bwd_unweighted`` has one small-R record, on all three architectures, with no windows and no large-R program,
    and takes the 12 kernel parameters and the 2 column offsets of :data:`PARAMETERS` (no weight, weight flag or
    weight gradient), in the order the entry points pass them.  Its grid is ``ceil(3 R / 4)`` CTAs
    (:data:`UNWEIGHTED_GRIDS`): below ``2**31 - 1`` at the most rows, where ``3 R`` is past 32 bits.  So is the grid
    of every launch through ``run_module`` and the launcher, on every architecture and SM count, above the fine
    window; in it the fine program runs on its grid."""
    names = [name for name, record in cake_jit.MODULES.items() if record["role"] == "bwd_unweighted"]
    assert len(names) == 1, names
    (name,) = names
    record = cake_jit.MODULES[name]
    assert record["role"] == "bwd_unweighted" and list(record["arches"]) == list(ARCHES)
    assert "rows_per_sm_windows" not in record
    assert cake_jit.PROGRAM_OF["bwd_unweighted"] == dict.fromkeys(ARCHES, name)
    assert cake_jit.PROGRAM_OF["bwd_unweighted_large"] == cake_jit.PROGRAM_OF["bwd_unweighted_relay"] == {}
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
            assert cake_jit.route("bwd_unweighted", arch, sm_count) == ((), (name,))
            last, fine = _expected_fine("bwd_unweighted", arch, sm_count)
            assert cake_jit.fine("bwd_unweighted", arch, sm_count) == (last, fine) and fine != name
            host_device.place(arch, sm_count)
            launch = pkg._launcher("bwd_unweighted", 0)
            for rows, want in UNWEIGHTED_GRIDS.items():
                host_device.calls.clear()
                pkg.run_module("bwd_unweighted", values, rows=rows, device_index=0)
                launch(rows, *(values[key] for key in PARAMETERS["bwd_unweighted"]))
                runs, grid = (fine, -(-9 * rows // 4)) if rows <= last else (name, want)
                args = (*(values[key] for key in PARAMETERS["bwd_unweighted"]), grid, 1, 1)
                assert host_device.calls == [(runs, arch, args)] * 2, (arch, sm_count, rows)


def test_launches_pass_the_argument_plan_and_the_grid(host_device) -> None:
    """``run_module`` passes the routed program's parameters (the kernel's, then the column offsets) in its argument
    plan's order, then ``cake_jit.grid(name, rows)``; a missing parameter and a launch of no rows are ``ValueError``s
    naming them, raised before any library is looked up."""
    host_device.place("sm_100a", 148)
    calls = host_device.calls
    for stage in STAGES:
        values = {key: f"<{key}>" for key in PARAMETERS[stage]}
        for rows in (1, 2, 3, 4, 5, 7, 8, 9, 129, 1777, 4097, cake_jit.MAX_ROWS - 1):
            name = _expected_program(stage, "sm_100a", 148, rows)
            calls.clear()
            pkg.run_module(stage, values, rows=rows, device_index=0)
            want = (*(values[key] for key in PARAMETERS[stage]), *cake_jit.grid(name, rows))
            assert calls == [(name, "sm_100a", want)], (stage, rows)
        calls.clear()
        with pytest.raises(ValueError, match="at least one row"):
            pkg.run_module(stage, values, rows=0, device_index=0)
        missing = {key: value for key, value in values.items() if key != "limit"}
        first = _expected_program(stage, "sm_100a", 148, 1)
        with pytest.raises(ValueError, match=f"{first} needs argument 'limit'"):
            pkg.run_module(stage, missing, rows=1, device_index=0)
        assert calls == []


def _check_route(stage: str, arch: str, sm_count: int, host_device, table: dict | None = None) -> set[str]:
    """The route of ``stage`` on ``arch`` with ``sm_count`` SMs is the one the windows give (``table``, role ->
    architecture -> windows of the windowed records; ``WINDOWS`` by default): ``cake_jit.route`` and ``cake_jit.fine``
    (:func:`_expected_route`, :func:`_expected_fine`), the package's plans (one per program, shared by its runs), its
    table at every row count below its limit and its tail (:func:`_expected_program`: by the windows that hold each
    row count, not by the edges), and the launches of ``run_module`` and of the launcher at 1 row, one row below, at
    and above each edge and ``MAX_ROWS - 1`` rows, each on its program's grid.  Returns the programs launched."""
    edges, names = _expected_route(stage, arch, sm_count, table)
    last, fine = _expected_fine(stage, arch, sm_count)
    assert cake_jit.route(stage, arch, sm_count) == (edges, names)
    assert cake_jit.fine(stage, arch, sm_count) == (last, fine)
    assert not edges or last < edges[0]
    host_device.place(arch, sm_count)
    got_last, got_edges, runs, got_fine = pkg._programs(stage, 0)
    assert (got_last, got_edges, tuple(run[0] for run in runs), got_fine[0]) == (last, edges, names, fine)
    assert len({id(run) for run in runs}) == len(set(names)), names
    limit, plans, tail = pkg._route(stage, 0)
    assert limit == max(edges[-1] + 1 if edges else 0, last + 1) == len(plans)
    assert {id(plan) for plan in (*plans, tail)} <= {id(run) for run in (*runs, got_fine)}
    want = [_expected_program(stage, arch, sm_count, rows, table) for rows in range(limit)]
    assert [plan[0] for plan in plans] == want
    assert tail[0] == names[-1] == _expected_program(stage, arch, sm_count, cake_jit.MAX_ROWS - 1, table)
    values = {key: f"<{key}>" for key in PARAMETERS[stage]}
    arguments = tuple(values[key] for key in PARAMETERS[stage])
    launch = pkg._launcher(stage, 0)
    ran = set()
    for rows in _edge_rows(_route_edges(stage, arch, sm_count, table)):
        name = _expected_program(stage, arch, sm_count, rows, table)
        ran.add(name)
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, device_index=0)
        launch(rows, *arguments)
        want = (*arguments, *cake_jit.grid(name, rows))
        assert host_device.calls == [(name, arch, want)] * 2, (rows, edges, last)
    assert ran == {*names, fine}
    return ran


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
def test_rows_route_to_the_fine_the_small_or_a_windowed_program(stage, arch, sm_count, host_device) -> None:
    """A launch of at most ``hi * SMs`` rows for the fine record's window ``[0, hi]`` on the architecture runs the
    stage's fine program; a launch of more rows that lie in a window of a windowed record (``lo * SMs <= rows <= hi *
    SMs`` for a window ``[lo, hi]`` of its ``rows_per_sm_windows``) runs that program, any other its small-R program,
    each on its grid: 1 row, one row below, at and above each edge of the windows, and ``MAX_ROWS - 1`` rows.  The
    package's route holds the plan of each row count up to the larger of the fine window's last row count and the last
    edge; above it, it runs the program of the last window when that window has no upper bound (:func:`_check_route`).
    On an architecture without a relay record of the stage, the route is the one 0.1.9's code makes from the same
    records at every row count."""
    ran = _check_route(stage, arch, sm_count, host_device)
    relay = cake_jit.PROGRAM_OF[stage + RELAY].get(arch)
    assert (relay in ran) == (relay is not None), (stage, arch, relay)
    if relay is None:
        limit, plans, tail = pkg._route(stage, 0)
        assert (limit, [plan[0] for plan in plans], tail[0]) == _route_019(stage, arch, sm_count)


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "without", [(LARGE, RELAY, FINE), (LARGE, FINE), (FINE,)], ids=["large_relay_fine", "large_fine", "fine"]
)
def test_an_architecture_without_a_program_runs_the_small_one(stage, arch, without, host_device, monkeypatch) -> None:
    """With no fine program of the stage on the architecture, with neither a fine nor a large-R one, and with none of
    the fine, the large-R and the relay one (the registry's table without them), every row count that no remaining
    program's window holds runs the small-R program: with none, the route has no edge and no fine bound."""
    sm_count = SM_COUNTS[-1]
    rows_list = _edge_rows(_route_edges(stage, arch, sm_count))
    edges, names = _expected_route(stage, arch, sm_count, without=without)
    table = {role: dict(programs) for role, programs in cake_jit.PROGRAM_OF.items()}
    for suffix in without:
        table[stage + suffix].pop(arch, None)
    monkeypatch.setattr(cake_jit, "PROGRAM_OF", table)
    small = cake_jit.PROGRAM_OF[stage][arch]
    assert cake_jit.route(stage, arch, sm_count) == (edges, names)
    assert not set(WINDOWED_ROLES) <= set(without) or (edges, names) == ((), (small,))
    assert cake_jit.fine(stage, arch, sm_count) == (0, small)
    host_device.place(arch, sm_count)
    values = {key: f"<{key}>" for key in PARAMETERS[stage]}
    for rows in rows_list:
        name = _expected_program(stage, arch, sm_count, rows, without=without)
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, device_index=0)
        want = (*(values[key] for key in PARAMETERS[stage]), *cake_jit.grid(name, rows))
        assert host_device.calls == [(name, arch, want)], rows
    with pytest.raises(ValueError, match="at least one row"):
        pkg.run_module(stage, values, rows=0, device_index=0)


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
@pytest.mark.parametrize("role", ["bwd_large", "bwd_relay"])
def test_the_route_refuses_windows_that_are_no_route(role, row, message, bwd_windows) -> None:
    """A large-R or a relay record's windows on an architecture must be ascending ``[lo, hi]`` lists of integers that
    do not touch, ``0 <= lo``, only the last ``hi`` None: any other row is a ``RuntimeError`` naming the record (the
    relay record beside the shipped large-R windows)."""
    large = cake_jit.PROGRAM_OF["bwd_large"]["sm_100a"]
    if role == "bwd_large":
        bwd_windows({"bwd_large": {**WINDOWS["bwd_large"], "sm_100a": row}})
    else:
        bwd_windows({"bwd_large": WINDOWS["bwd_large"], "bwd_relay": {"sm_100a": row}})
    name = large if role == "bwd_large" else RELAY_NAME
    with pytest.raises(RuntimeError, match=f"{name}: rows_per_sm_windows\\['sm_100a'\\] .* {message}"):
        cake_jit.route("bwd", "sm_100a", 148)


@pytest.mark.parametrize(
    ("stage", "row", "message"),
    [
        ("bwd", [], "is not one window \\[0, hi\\]"),
        ("bwd", [[0]], "is not one window \\[0, hi\\]"),
        ("bwd", [[0, 2], [4, 6]], "is not one window \\[0, hi\\]"),
        ("bwd", [[1, 2]], "is not one window \\[0, hi\\]"),
        ("bwd", [[0, 0]], "is not one window \\[0, hi\\]"),
        ("bwd", [[0, None]], "is not one window \\[0, hi\\]"),
        ("bwd", [[0, 2.0]], "is not one window \\[0, hi\\]"),
        ("bwd", [[0, True]], "is not one window \\[0, hi\\]"),
        ("bwd", [[False, 2]], "is not one window \\[0, hi\\]"),
        ("bwd", [[0, 3]], "reaches the first large-R or relay window of 'bwd', from 444 rows"),
        ("bwd", [[0, 40]], "reaches the first large-R or relay window of 'bwd', from 444 rows"),
        ("fwd", [[0, 112]], "reaches the first large-R or relay window of 'fwd', from 16576 rows"),
    ],
)
def test_the_route_refuses_fine_windows_that_are_no_route(stage, row, message, host_device, monkeypatch) -> None:
    """A fine record's windows on an architecture must be one ``[0, hi]`` list with an integer ``hi >= 1`` that ends
    below the first window of the stage's large-R and relay records (``hi`` below every ``lo``): any other row is a
    ``RuntimeError`` naming the record, from ``cake_jit.fine`` and from the package's route (the launcher is not made).
    ``hi`` one below the first ``lo``, or ``MAX_FINE_ROWS_PER_SM`` when that is smaller, is a route."""
    fine = cake_jit.PROGRAM_OF[stage + FINE]["sm_100a"]
    first = min(WINDOWS[stage + suffix]["sm_100a"][0][0] for suffix in WINDOWED_ROLES if stage + suffix in WINDOWS)
    below = min(first - 1, cake_jit.MAX_FINE_ROWS_PER_SM)
    for hi, accepted in ((below, True), (None, False)):
        windows = [[0, hi]] if accepted else row
        record = {**cake_jit.MODULES[fine], "rows_per_sm_windows": {"sm_100a": windows}}
        monkeypatch.setitem(cake_jit.MODULES, fine, record)
        host_device.place("sm_100a", 148)
        if accepted:
            assert cake_jit.fine(stage, "sm_100a", 148) == (hi * 148, fine)
            last, _, _, got = pkg._programs(stage, 0)
            assert (last, got) == (hi * 148, (fine, *pkg._plan(fine)))
            _, plans, _ = pkg._route(stage, 0)
            assert plans[last][0] == fine != plans[last + 1][0]
            continue
        pattern = f"{fine}: rows_per_sm_windows\\['sm_100a'\\] .* {message}"
        with pytest.raises(RuntimeError, match=pattern):
            cake_jit.fine(stage, "sm_100a", 148)
        with pytest.raises(RuntimeError, match=pattern):
            pkg._launcher(stage, 0)
        assert host_device.calls == []


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_the_windows_of_the_large_r_and_the_relay_programs_route_the_rows(
    route, arch, sm_count, host_device, bwd_windows
) -> None:
    """With the ``bwd`` windows of each synthetic route (``ROUTES``), a launch of ``bwd`` runs the fine program up to
    its last row count, the large-R or the relay program whose window holds its row count and the small-R one at every
    other row count, at every row count of the package's table, above it and at the edges (:func:`_check_route`).  On
    an architecture without a relay record the table, its limit and its tail are 0.1.9's for the same records
    (:func:`_route_019`)."""
    table = ROUTES[route]
    bwd_windows(table)
    _check_route("bwd", arch, sm_count, host_device, table)
    if arch not in table.get("bwd_relay", {}):
        limit, plans, tail = pkg._route("bwd", 0)
        assert (limit, [plan[0] for plan in plans], tail[0]) == _route_019("bwd", arch, sm_count)


def test_three_programs_split_the_rows_at_their_window_edges(host_device, bwd_windows) -> None:
    """The edges of a route of three programs above the fine window, one row below, at and above each: ``bwd`` on
    ``sm_100a`` with 148 SMs, the large-R window ``[16, 20]`` and the relay window ``[21, 26]`` one row per SM above
    it, then on ``sm_103a`` with 152 SMs the relay windows ``[3, 15]`` and ``[21, None]`` around the large-R window
    ``[16, 20]``.  The relay program runs one CTA per row (``rows`` CTAs), the others ``ceil(rows / 4)``."""
    small, fine = cake_jit.PROGRAM_OF["bwd"]["sm_100a"], cake_jit.PROGRAM_OF["bwd_fine"]["sm_100a"]
    large = cake_jit.PROGRAM_OF["bwd_large"]["sm_100a"]
    values = {key: f"<{key}>" for key in PARAMETERS["bwd"]}
    cases = {
        ("sm_100a", 148, "relay_adjacent"): (
            (2367, 2960, 3107, 3848),
            (small, large, small, RELAY_NAME, small),
            {296: fine, 297: small, 2367: small, 2368: large, 2960: large, 2961: small, 3107: small, 3108: RELAY_NAME},
        ),
        ("sm_103a", 152, "relay_around"): (
            (455, 2280, 2431, 3040, 3191),
            (small, RELAY_NAME, small, large, small, RELAY_NAME),
            {304: fine, 305: small, 455: small, 456: RELAY_NAME, 2280: RELAY_NAME, 2281: small, 2432: large},
        ),
    }
    for (arch, sm_count, route), (edges, names, rows_of) in cases.items():
        bwd_windows(ROUTES[route])
        assert cake_jit.route("bwd", arch, sm_count) == (edges, names)
        host_device.place(arch, sm_count)
        rows_of = {**rows_of, edges[-1]: names[-2], edges[-1] + 1: names[-1], cake_jit.MAX_ROWS - 1: names[-1]}
        for rows, name in sorted(rows_of.items()):
            host_device.calls.clear()
            pkg.run_module("bwd", values, rows=rows, device_index=0)
            # a CTA a row in the fine and the relay programs, 4 rows a CTA in the others
            grid = rows if name in (fine, RELAY_NAME) else -(-rows // 4)
            want = (*(values[key] for key in PARAMETERS["bwd"]), grid, 1, 1)
            assert host_device.calls == [(name, arch, want)], (arch, route, rows)


@pytest.mark.parametrize(
    ("large", "relay", "pair"),
    [
        ([[16, 20]], [[20, 26]], ("large", [16, 20], "relay", [20, 26])),  # touch
        ([[12, 21]], [[16, 18]], ("large", [12, 21], "relay", [16, 18])),  # inside
        ([[16, 20]], [[16, 18]], ("large", [16, 20], "relay", [16, 18])),  # the same lo
        ([[16, 20]], [[10, 16]], ("relay", [10, 16], "large", [16, 20])),  # touch, relay first
        ([[48, None]], [[56, 60]], ("large", [48, None], "relay", [56, 60])),  # an open window below another
        ([[16, 20]], [[3, None]], ("relay", [3, None], "large", [16, 20])),
        ([[12, 21], [32, 41]], [[22, 32]], ("relay", [22, 32], "large", [32, 41])),
        ([[12, 21], [32, 41], [48, None]], [[24, 30], [40, 44]], ("large", [32, 41], "relay", [40, 44])),
    ],
)
def test_the_route_refuses_windows_of_two_programs_that_overlap_or_touch(
    large, relay, pair, host_device, bwd_windows
) -> None:
    """The windows of a stage's large-R and relay records on an architecture, taken together in the order of their
    ``lo``, must be disjoint: a window that overlaps or touches the next one (or an open window below another) is a
    ``RuntimeError`` naming both records and windows, from ``cake_jit.route``, ``cake_jit.fine`` and the package's
    route (the launcher is not made).  The other architectures keep their routes."""
    bwd_windows({"bwd_large": {**WINDOWS["bwd_large"], "sm_100a": large}, "bwd_relay": {"sm_100a": relay}})
    names = {"large": cake_jit.PROGRAM_OF["bwd_large"]["sm_100a"], "relay": RELAY_NAME}
    first, window, second, other = pair
    pattern = re.escape(
        f"{names[first]}: rows_per_sm_windows['sm_100a'] window {window!r} and {names[second]}: window {other!r} of "
        "'bwd' overlap or touch"
    )
    with pytest.raises(RuntimeError, match=pattern):
        cake_jit.route("bwd", "sm_100a", 148)
    with pytest.raises(RuntimeError, match=pattern):
        cake_jit.fine("bwd", "sm_100a", 148)
    host_device.place("sm_100a", 148)
    with pytest.raises(RuntimeError, match=pattern):
        pkg._launcher("bwd", 0)
    assert host_device.calls == []
    assert cake_jit.route("bwd", "sm_103a", 152) == _expected_route("bwd", "sm_103a", 152)


def test_the_fine_window_ends_below_every_windowed_program(host_device, bwd_windows) -> None:
    """The fine window must end below the first window of the large-R and the relay records: the shipped ``bwd_fine``
    window ``[0, 2]`` beside a relay window ``[2, 10]`` is a ``RuntimeError`` naming the fine record, from
    ``cake_jit.fine`` and the package's route; beside ``[3, 10]`` it is a route, and the small-R program runs between
    the two windows."""
    fine = cake_jit.PROGRAM_OF["bwd_fine"]["sm_100a"]
    small = cake_jit.PROGRAM_OF["bwd"]["sm_100a"]
    bwd_windows({"bwd_large": _on(ARCHES, [[16, 20]]), "bwd_relay": _on(ARCHES, [[2, 10]])})
    host_device.place("sm_100a", 148)
    pattern = (
        f"{fine}: rows_per_sm_windows\\['sm_100a'\\] .* reaches the first large-R or relay window of 'bwd', "
        "from 296 rows on 148 SMs"
    )
    with pytest.raises(RuntimeError, match=pattern):
        cake_jit.fine("bwd", "sm_100a", 148)
    with pytest.raises(RuntimeError, match=pattern):
        pkg._launcher("bwd", 0)
    bwd_windows({"bwd_large": _on(ARCHES, [[16, 20]]), "bwd_relay": _on(ARCHES, [[3, 10]])})
    host_device.place("sm_100a", 148)
    assert cake_jit.fine("bwd", "sm_100a", 148) == (296, fine)
    _, plans, _ = pkg._route("bwd", 0)
    assert [plans[rows][0] for rows in (296, 297, 443, 444)] == [fine, small, small, RELAY_NAME]


def test_the_relay_program_runs_a_row_a_cta(bwd_windows) -> None:
    """``bwd_relay``'s build facts are the lane layout with 3 tasks a row (a warp per (row, run)); a relay record of
    96 threads owns 3 tasks a CTA, one row: its grid is ``rows`` CTAs, below ``MAX_GRID`` at the most rows."""
    assert cake_jit.BUILD_OF["bwd_relay"] == {**LANE_LAYOUT, "tasks_per_row": 3}
    assert [role for role in cake_jit.BUILD_OF if role.endswith(RELAY)] == ["bwd_relay"]
    bwd_windows(ROUTES["relay_around"])
    assert cake_jit.cta_tasks(RELAY_NAME) == (3, 3)
    for rows in (1, 2, 3, 4, 129, 4097, 16384, 98304, 196608, cake_jit.MAX_ROWS - 1):
        assert cake_jit.grid(RELAY_NAME, rows) == (rows, 1, 1), rows
    assert cake_jit.MAX_ROWS - 1 < cake_jit.MAX_GRID


def test_the_launcher_refuses_a_relay_program_that_takes_another_order(host_device, bwd_windows, monkeypatch) -> None:
    """The launcher of ``bwd`` is not made (``RuntimeError`` naming the record) when its relay program takes its
    parameters in another order than the entry points pass them; nothing is launched."""
    bwd_windows(ROUTES["relay_around"])
    plan = [list(item) for item in cake_jit.MODULES[RELAY_NAME]["arg_plan"]]
    plan[0], plan[1] = plan[1], plan[0]  # the up before the gate
    monkeypatch.setitem(cake_jit.MODULES, RELAY_NAME, {**cake_jit.MODULES[RELAY_NAME], "arg_plan": plan})
    pkg.parameters.cache_clear()
    pkg._plan.cache_clear()
    host_device.place("sm_100a", SM_COUNTS[0])
    with pytest.raises(RuntimeError, match=f"record {RELAY_NAME}: parameters"):
        pkg._launcher("bwd", 0)
    assert host_device.calls == []


def test_the_fine_programs_and_their_grids() -> None:
    """Each stage has one fine record, on all three architectures, with the parameters of the stage's other programs
    in the same order.  ``fwd_fine`` and ``bwd_unweighted_fine`` run a warp per (row, load) task, 4 tasks a CTA of 128
    threads, ``ceil(9 R / 4)`` CTAs, at most ``2**31 - 1`` (past it from about 954 M rows; the kernels count in 64 bits
    and stride over the blocks).  ``bwd_fine`` runs a CTA of 288 threads (9 warps) per row, ``R`` CTAs; it counts its
    rows in 32 bits, so it takes at most ``2**30`` rows (the grids below are counts only; the route sends it at most
    its window's ``hi * SMs`` rows: at most ``MAX_FINE_ROWS_PER_SM * SMs``, and ``2 * SMs`` as shipped)."""
    facts = {"fwd": ((128, 1, 1), (4, 9)), "bwd": ((288, 1, 1), (9, 9)), "bwd_unweighted": ((128, 1, 1), (4, 9))}
    for stage, (block, tasks) in facts.items():
        names = [name for name, record in cake_jit.MODULES.items() if record["role"] == stage + FINE]
        assert len(names) == 1, (stage, names)
        (name,) = names
        record = cake_jit.MODULES[name]
        assert record["stage"] == stage and list(record["arches"]) == list(ARCHES), name
        assert cake_jit.PROGRAM_OF[stage + FINE] == dict.fromkeys(ARCHES, name)
        assert pkg.parameters(name) == PARAMETERS[stage] == pkg._ARGUMENTS[stage], name
        assert tuple(record["launch"]["block"]) == block and cake_jit.cta_tasks(name) == tasks, name
        per_cta, per_row = tasks
        for rows in (1, 2, 3, 4, 5, 9, 33, 129, 1184, 4097):
            assert cake_jit.grid(name, rows) == (-(-rows * per_row // per_cta), 1, 1), (name, rows)
        most = cake_jit.MAX_ROWS - 1
        assert cake_jit.grid(name, most) == (min(-(-most * 9 // per_cta), 2**31 - 1), 1, 1), name
    first = -(-(2**31 - 1) * 4 // 9)  # the first row count whose ceil(9 R / 4) passes 2**31 - 1, about 954 M
    for stage in ("fwd", "bwd_unweighted"):
        name = cake_jit.PROGRAM_OF[stage + FINE]["sm_100a"]
        assert cake_jit.grid(name, first - 1)[0] <= 2**31 - 1 < -(-(first + 1) * 9 // 4)
        assert cake_jit.grid(name, first + 1) == (2**31 - 1, 1, 1)


@pytest.mark.parametrize("stage", ["fwd", "bwd_unweighted"])
def test_the_route_refuses_a_fine_window_past_the_most_rows_per_sm(stage, host_device, monkeypatch) -> None:
    """A fine record's window ``[0, hi]`` with ``hi`` past ``MAX_FINE_ROWS_PER_SM`` (64) is a ``RuntimeError`` naming
    the record, from ``cake_jit.fine`` and from the package's route (the launcher is not made), also where no large-R
    window bounds it (``bwd_unweighted``; ``fwd``, whose first large-R window is from 112 rows per SM).  The window
    ``[0, 64]`` is a route on every delivered SM count: the fine plan covers rows 1 to ``64 * SMs`` and the small-R
    plan runs at the next row count (in ``fwd`` up to the first large-R edge).  In ``bwd_unweighted`` the table ends
    there (``limit`` ``64 * SMs + 1``) and the tail is small-R.  On the delivered SM counts the bound keeps
    ``bwd_fine``, which counts its rows in 32 bits, far below its ``2**30`` rows."""
    most = cake_jit.MAX_FINE_ROWS_PER_SM
    assert most == 64 and 2 * most * max(SM_COUNTS) <= 2**31
    fine = cake_jit.PROGRAM_OF[stage + FINE]["sm_100a"]
    small = cake_jit.PROGRAM_OF[stage]["sm_100a"]
    for hi in (most, most + 1):
        record = {**cake_jit.MODULES[fine], "rows_per_sm_windows": {"sm_100a": [[0, hi]]}}
        monkeypatch.setitem(cake_jit.MODULES, fine, record)
        for sm_count in SM_COUNTS:
            host_device.place("sm_100a", sm_count)
            if hi == most:
                last = hi * sm_count
                assert cake_jit.fine(stage, "sm_100a", sm_count) == (last, fine)
                limit, plans, tail = pkg._route(stage, 0)
                assert plans[last][0] == fine and {plan[0] for plan in plans[1 : last + 1]} == {fine}
                if stage == "bwd_unweighted":
                    assert (limit, len(plans), tail[0]) == (last + 1, last + 1, small)
                else:
                    assert plans[last + 1][0] == small
                continue
            pattern = f"{fine}: rows_per_sm_windows\\['sm_100a'\\] .* passes {most} rows per SM"
            with pytest.raises(RuntimeError, match=pattern):
                cake_jit.fine(stage, "sm_100a", sm_count)
            with pytest.raises(RuntimeError, match=pattern):
                pkg._launcher(stage, 0)
            assert host_device.calls == []


def test_the_route_refuses_a_fine_window_whose_grid_passes_the_largest(monkeypatch) -> None:
    """A fine record's window whose last row count needs more than ``MAX_GRID`` CTAs is a ``RuntimeError`` naming the
    record, so the grid of a routed launch, ``ceil(9 R / 4)`` CTAs, is never capped.  Within ``MAX_FINE_ROWS_PER_SM``
    rows per SM only an SM count of about 14.9 M reaches it (``bwd_unweighted`` has no large-R window): the window
    ``[0, 64]`` is a route on the most SMs whose last row count fits the grid, and refused on one SM more."""
    fine = cake_jit.PROGRAM_OF["bwd_unweighted_fine"]["sm_100a"]
    per_cta, per_row = cake_jit.cta_tasks(fine)
    hi = cake_jit.MAX_FINE_ROWS_PER_SM
    record = {**cake_jit.MODULES[fine], "rows_per_sm_windows": {"sm_100a": [[0, hi]]}}
    monkeypatch.setitem(cake_jit.MODULES, fine, record)
    most = cake_jit.MAX_GRID * per_cta // (per_row * hi)  # the most SMs whose last row count fits the grid
    for sm_count in (most, most + 1):
        uncapped = -(-hi * sm_count * per_row // per_cta)
        if sm_count == most:
            assert cake_jit.fine("bwd_unweighted", "sm_100a", sm_count) == (hi * sm_count, fine)
            assert cake_jit.grid(fine, hi * sm_count) == (uncapped, 1, 1) and uncapped <= cake_jit.MAX_GRID
            continue
        assert uncapped > cake_jit.MAX_GRID
        pattern = f"{fine}: rows_per_sm_windows\\['sm_100a'\\] .* more than {cake_jit.MAX_GRID} CTAs"
        with pytest.raises(RuntimeError, match=pattern):
            cake_jit.fine("bwd_unweighted", "sm_100a", sm_count)


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
    library when made; at 1 row, one row below, at and above each edge of the windows and ``MAX_ROWS - 1`` rows it runs
    the program ``run_module`` runs, with the same arguments (the parameters in the plan's order) and grid, and it looks
    up each program's FFI entry once, at that program's first launch."""
    host_device.place(arch, sm_count)
    bound: list = []
    entry = pkg._entry
    monkeypatch.setattr(pkg, "_entry", lambda name, on: bound.append((name, on)) or entry(name, on))
    launch = pkg._launcher(stage, 0)
    assert pkg._launcher(stage, 0) is launch
    assert pkg._launcher(stage, 1) is not launch
    assert bound == [] and host_device.calls == []
    _, names = _expected_route(stage, arch, sm_count)
    _, fine = _expected_fine(stage, arch, sm_count)
    values = {key: f"<{key}>" for key in PARAMETERS[stage]}
    rows_list = _edge_rows(_route_edges(stage, arch, sm_count))
    launched = []
    for rows in rows_list:
        host_device.calls.clear()
        launch(rows, *(values[key] for key in PARAMETERS[stage]))
        launched.append(list(host_device.calls))
    assert sorted(bound) == sorted((name, arch) for name in {*names, fine}), bound
    for rows, got in zip(rows_list, launched, strict=True):
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, device_index=0)
        assert got == host_device.calls, rows
        assert got[0][0] == _expected_program(stage, arch, sm_count, rows), rows


@pytest.mark.parametrize("role", ["bwd_large", "bwd_fine"])
def test_the_launcher_refuses_programs_that_take_another_order(role, host_device, monkeypatch) -> None:
    """A launcher passes the parameters by position, so it is not made (``RuntimeError`` naming the record) for a stage
    whose program (here its large-R or its fine one) takes them in another order than the entry points pass them;
    nothing is launched."""
    host_device.place("sm_100a", SM_COUNTS[0])
    name = cake_jit.PROGRAM_OF[role]["sm_100a"]
    plan = [list(item) for item in cake_jit.MODULES[name]["arg_plan"]]
    plan[0], plan[1] = plan[1], plan[0]  # the up before the gate
    monkeypatch.setitem(cake_jit.MODULES, name, {**cake_jit.MODULES[name], "arg_plan": plan})
    pkg.parameters.cache_clear()
    pkg._plan.cache_clear()
    try:
        with pytest.raises(RuntimeError, match=f"record {name}: parameters"):
            pkg._launcher("bwd", 0)
        assert host_device.calls == []
        pkg._launcher("fwd", 0)  # the other stage's programs keep the order
    finally:
        pkg.parameters.cache_clear()
        pkg._plan.cache_clear()
        _clear_routes()


def test_the_limit_check_and_its_cached_rounding(monkeypatch) -> None:
    """``limit`` is the same FP32 value whatever the calls before passed: a float, or an int that a float holds exactly,
    equal to the float value of the last accepted limit gets its cached rounding, and every other value (another number,
    a bool, a float subclass, a fraction, a tensor, a string, a non-finite, non-positive or too large number) takes the
    full checks, with their errors, also right after an equal valid limit was cached."""
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
    """``h``, ``d_gate``, ``d_up`` and (weighted) ``d_weight`` of each stage's small-R, large-R and fine program (forced
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
    """On boundary data, also with a gate of -88 in each third of every row (the sigmoid's deep path), ``d_gate`` is
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


@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize("w", ["random", "none"])
def test_no_rows_return_empty_outputs_and_launch_nothing(layout, w, program, device, monkeypatch) -> None:
    """``R = 0``: the steps and the autograd function return empty outputs of the right shapes and dtypes, and
    nothing is launched or loaded, with the route forced to each program (0 rows is in every fine window)."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call with no rows launched or loaded a kernel")

    _force(monkeypatch, program)

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
    # The held launch (as in a backward, with hooks set) keeps no launch, and a checkpoint launches nothing.
    made = _held_launches(monkeypatch)
    with monkeypatch.context() as patch:
        _as_in_a_backward(patch)
        leaves = _leaves(t)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            out = clamped_swiglu(*leaves, limit=L)
        assert out.shape == (0, WIDTH) and out.grad_fn is not None
        out.sum().backward()
        assert leaves[0].grad.shape == leaves[0].shape
    assert len(made) == 1 and not any(made)
    leaves = _leaves(t)
    out = torch.utils.checkpoint.checkpoint(clamped_swiglu, *leaves, limit=L, use_reentrant=False)
    out.sum().backward()
    assert out.shape == (0, WIDTH) and leaves[0].grad.shape == leaves[0].shape
    assert len(made) == 2 and not any(made)  # the recomputation's held launch


def test_one_library_per_program_serves_every_call(device, monkeypatch) -> None:
    """Row counts 1 to 4097 and each edge of the device's windows of each stage (the fine windows' last row counts
    included) and one row past it, both layouts, a strided layout, weighted and unweighted calls and three limits load
    one library per program they run, once: no recompilation (``rows``, ``limit``, the weight flag, the strides and
    the column offsets are launch arguments)."""
    edges = sorted({edge for stage in WINDOWED for edge in _device_edges(stage, device.index)})
    edges = sorted({*edges, *(_device_fine(stage, device.index) for stage in STAGES)} - {0})
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
@pytest.mark.parametrize("stage", STAGES)
def test_the_programs_of_a_stage_give_the_same_bits(stage, step, device, programs_run, monkeypatch) -> None:
    """At and one row above each edge of this device's windows of ``stage`` (the row counts where its route changes
    program: the fine window's last row count and the edges of the large-R windows), on rows at the edge of the FP32
    sigmoid and on gates of -88 beside boundary values (weighted, then unweighted for ``fwd``; weighted for ``bwd``,
    which runs weighted calls only; unweighted for ``bwd_unweighted``): the routed launches run the fine program of
    ``stage`` up to its last row count, its large-R program inside its windows (one row above the edge below a window,
    at its ``hi`` edge) and the small-R program elsewhere, and every launch forced to the small-R, to the large-R and to
    the fine programs gives the routed call's bits of ``h``, ``d_gate``, ``d_up`` and ``d_weight``."""
    arch = pkg.device_arch(device.index)
    sm_count = torch.cuda.get_device_properties(device.index).multi_processor_count
    edges = _route_edges(stage, arch, sm_count)
    assert edges[0] == _device_fine(stage, device.index) and edges[1:] == list(_device_edges(stage, device.index))
    weights = {"fwd": ("random", "none"), "bwd": ("random", "random"), "bwd_unweighted": ("none", "none")}[stage]
    for edge in edges:
        rows = edge + step
        for w, data in zip(weights, ("mixed", "slow_mixed"), strict=True):
            t = make_inputs(Case(rows, "merged", w, data=data), seed=43, device=device)
            stages = _stages(w != "none")
            programs_run.clear()
            want = step_outputs(forward, backward, t)
            assert programs_run == [_routed(s, rows, device.index) for s in stages], (rows, w)
            assert programs_run[stages.index(stage)] == _expected_program(stage, arch, sm_count, rows), (rows, w)
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
    """``(d_gate, d_up)`` of the ``bwd`` program ``name`` itself (no route) launched as 0.1.4 and earlier launched an
    unweighted backward: the weight flag 0, the device's cached one-element weight, and a weight gradient it writes and
    nobody reads."""
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
        "w": pkg._launcher("fwd", device.index).ignored_weight(),
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
        "up_col": 0,
        "d_up_col": 0,
    }
    entry = pkg._entry(name, pkg.device_arch(device.index))
    entry(*(values[key] for key in pkg.parameters(name)), *cake_jit.grid(name, t.R))
    return d_gate, d_up


def _gradients(d_first: torch.Tensor, d_up) -> tuple[torch.Tensor, torch.Tensor]:
    """``(d_gate, d_up)`` of a backward's result (a merged gradient split into its halves)."""
    return (d_first[:, :WIDTH], d_first[:, WIDTH:]) if d_up is None else (d_first, d_up)


@pytest.mark.parametrize("layout", ["split", "merged", "strided"])
@pytest.mark.parametrize("data", BITS_DATA)
def test_the_unweighted_backward_stores_the_bits_of_bwd(data, layout, device, programs_run, monkeypatch) -> None:
    """An unweighted backward runs a ``bwd_unweighted`` program (its fine one up to the fine window's last row count,
    its small-R one at more rows), and its ``d_gate`` and ``d_up`` are, bit for bit (BF16 compared as int16, so NaN
    payloads and signed zeros count), those of the other ``bwd_unweighted`` program (forced through the route), those
    of the small-R, the large-R and the fine ``bwd`` program each launched itself with the weight flag 0 (the
    unweighted backward of 0.1.4 and earlier), and those of a weighted backward with weights of 1: every data kind, NaN
    and infinities, a gate of -88 in one third of a row only, limits 10, 7.3 and 1.5, 1 to 5 rows (a partial last CTA
    of tasks), 129 and 4097 rows, and at and one row above each edge of the device's ``bwd`` and ``bwd_unweighted``
    windows."""
    bwd_programs = list(dict.fromkeys(_program("bwd", device.index, program) for program in PROGRAMS))
    unweighted = list(dict.fromkeys(_program("bwd_unweighted", device.index, program) for program in PROGRAMS))
    assert len(unweighted) == 2, unweighted
    edges = {*_device_edges("bwd", device.index), _device_fine("bwd", device.index)}
    edges |= {_device_fine("bwd_unweighted", device.index)}
    row_counts = sorted({1, 2, 3, 4, 5, 129, 4097} | {edge + step for edge in edges - {0} for step in (0, 1)})
    for rows in row_counts:
        for limit in (10.0, 7.3, 1.5):
            t = _bits_inputs(data, layout, rows, limit, seed=rows, device=device)
            programs_run.clear()
            d_gate, d_up = _gradients(*backward(t.dh, t.first, t.second, None, limit=limit)[:2])
            assert programs_run == [_routed("bwd_unweighted", rows, device.index)], (rows, programs_run)
            what = f"{data}, {layout}, {rows} rows, L={limit}"
            for program in ("small", "fine"):
                with monkeypatch.context() as patch:
                    _force(patch, program)
                    programs_run.clear()
                    got = _gradients(*backward(t.dh, t.first, t.second, None, limit=limit)[:2])
                name = _program("bwd_unweighted", device.index, program)
                assert programs_run == [name], (program, rows, programs_run)
                assert_same_bits(d_gate, got[0], f"d_gate of the unweighted program {name}: {what}")
                assert_same_bits(d_up, got[1], f"d_up of the unweighted program {name}: {what}")
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
    """An unweighted backward launches ``bwd_unweighted`` with its 12 kernel parameters and 2 column offsets (no
    weight, weight flag or weight gradient) and allocates only its new gradients: none with preallocated gradients, one
    new merged gradient or two new split ones, with the same bits.  A weighted backward allocates its weight gradient as
    well."""
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
def test_the_deep_sigmoid_gives_the_bits_of_the_fast_one(layout, w, program, device, monkeypatch) -> None:
    """In each stage's small-R, large-R and fine program (forced through the route), on ordinary and boundary values
    whose clamped gates are all above -87 (the sigmoid's fast path everywhere), a gate of -88 at columns 0, 768 and
    1536 of every row (one in each third: every run takes the deep path, the row of a one-run program, each run of 768
    columns of a three-run one and the task of each of those loads in a fine one) keeps the bits of ``h``,
    ``d_gate`` and ``d_up`` at every other column, and the value of ``d_weight`` (``dh`` is 0 at those columns in both
    inputs).  The deep path ran: at the gates of -88, ``h`` and
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


# The forward launch of the autograd entry, and the checkpoint recompute.

# The width of the model rows that the gate/up projection reads and the down projection writes, in these tests.
HIDDEN = 256


def _projections(t: types.SimpleNamespace, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """BF16 operands of the projections around the clamped SwiGLU on ``t``'s rows: ``x [R, HIDDEN]``, the gate/up
    weights (one ``[4608, HIDDEN]`` weight for a merged layout, else ``[2304, HIDDEN]`` each, ``wu`` None when
    merged), the down weight ``wd [HIDDEN, 2304]`` and the upstream gradient ``dy [R, HIDDEN]``."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def draw(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (scale * torch.randn(*shape, device=device, generator=gen)).bfloat16()

    rows = 2 * WIDTH if t.layout == "merged" else WIDTH
    return types.SimpleNamespace(
        x=draw(t.R, HIDDEN),
        wg=draw(rows, HIDDEN, scale=6 / HIDDEN**0.5),
        wu=None if t.layout == "merged" else draw(rows, HIDDEN, scale=6 / HIDDEN**0.5),
        wd=draw(HIDDEN, WIDTH, scale=1 / WIDTH**0.5),
        dy=draw(t.R, HIDDEN),
    )


def _down(h: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(h, wd)


def _projected(L: float):
    """The gate/up projection (one merged GEMM or two), then the clamped SwiGLU."""

    def projected(x, wg, wu, weight):
        first = torch.nn.functional.linear(x, wg)
        second = None if wu is None else torch.nn.functional.linear(x, wu)
        return clamped_swiglu(first, second, weight, limit=L)

    return projected


def _block(L: float):
    """The expert's block: the gate/up projection (one merged GEMM or two), the clamped SwiGLU, the down projection."""
    projected = _projected(L)

    def block(x, wg, wu, weight, wd):
        return _down(projected(x, wg, wu, weight), wd)

    return block


def _kept_leaves(t: types.SimpleNamespace) -> tuple[list, Callable[[], dict]]:
    """Fresh operands of the inputs ``t`` that keep their layout, and a function giving their gradients keyed as
    :func:`_autograd_step` keys them: an operand that is a non-contiguous view of a wider buffer (``strided``,
    ``copied``) is the same view of a fresh leaf of its whole buffer, any other one a fresh leaf of its own (None stays
    None)."""
    operands: list = []
    parts: list = []
    for x in (t.first, t.second, t.weight):
        if x is None:
            operands.append(None)
            parts.append(None)
        elif x._base is not None and not x.is_contiguous():
            base = x._base.detach().clone().requires_grad_(True)
            operands.append(base.as_strided(x.shape, x.stride(), x.storage_offset()))
            parts.append((base, x))
        else:
            leaf = x.detach().clone().requires_grad_(True)
            operands.append(leaf)
            parts.append((leaf, None))

    def gradients() -> dict:
        found = {}
        for key, part in zip(("d_first", "d_up", "d_weight"), parts, strict=True):
            if part is not None:
                leaf, view = part
                grad = leaf.grad
                found[key] = grad if view is None else grad.as_strided(view.shape, view.stride(), view.storage_offset())
        return found

    return operands, gradients


def _held_launches(patch: pytest.MonkeyPatch) -> list:
    """Every :class:`_HeldLaunch` made from now on (one per call that takes the held launch), in order."""
    made: list = []

    class Counted(pkg._HeldLaunch):
        __slots__ = ()

        def __init__(self) -> None:
            made.append(self)
            super().__init__()

    patch.setattr(pkg, "_HeldLaunch", Counted)
    return made


def _as_in_a_backward(patch: pytest.MonkeyPatch) -> None:
    """The package's graph-task query answers "a backward runs" (as in a checkpoint's recomputation)."""
    patch.setattr(pkg, "_graph_task", lambda: 0)


def _in_a_backward(call: Callable[[], object]) -> object:
    """``call()`` run with grad mode on inside the backward of another autograd function (a backward graph task on this
    thread, as for a checkpoint's recomputation); its result."""
    found: list = []

    class Outer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            assert torch._C._current_graph_task_id() != -1
            with torch.enable_grad():
                found.append(call())
            return grad

    Outer.apply(torch.zeros(1, requires_grad=True)).backward()
    (result,) = found
    return result


def _recomputed(launched: list, run) -> tuple[int, int]:
    """``run(mark)`` runs a forward, calls ``mark()`` and runs its backward: the number of ``fwd`` launches before the
    mark and after it (a forward launched in the backward is a checkpoint recompute)."""
    marks: list = []
    run(lambda: marks.append(len(launched)))
    (mark,) = marks
    stages = [launch[0] for launch in launched]
    return stages[:mark].count("fwd"), stages[mark:].count("fwd")


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("layout", ["merged", "split", "strided", "copied"])
def test_a_recomputation_launches_the_forward_after_autograd_saved_the_inputs(layout, w, device, launches, monkeypatch):
    """In a backward with saved-tensors hooks set (where a non-reentrant checkpoint recomputes its region),
    ``cake_clamped_swiglu_det`` launches its forward kernel after ``_HeldFunction.apply`` returns: every pack hook of
    the inputs it saves runs before the launch, as for a built-in op, in a real backward too, in the common call and in
    the full checks' layouts (``strided``, ``copied``: the launch of
    :func:`_forward`).  Anywhere else it launches inside ``forward``, before the pack hooks, and keeps no launch:
    without hooks, with hooks outside a backward (a checkpoint's forward pass, activation offloading), for a direct
    ``apply`` (which takes no fifth argument) and while ``_is_compiling()`` or ``_is_tracing()`` is true.  Under
    ``torch.no_grad()`` either way launches once and runs no pack hook.  Every way gives the steps' bits, one forward
    launch and one backward launch."""
    t = make_inputs(Case(129, layout, w), seed=31, device=device)
    want = _functional(t)
    packs = sum(x is not None for x in (t.first, t.second, t.weight))
    made = _held_launches(monkeypatch)

    def pack(x: torch.Tensor) -> torch.Tensor:
        launches.append(("pack", None, None))
        return x

    def run(call, *, in_backward: bool = True, hooks: bool = True) -> list:
        """The launches and packs of ``call(operands)`` in order (as in a backward unless not asked, with a pack hook
        unless not asked); then its backward, with the steps' bits and one backward launch."""
        launches.clear()
        operands, gradients = _kept_leaves(t)
        with monkeypatch.context() as patch:
            if in_backward:
                _as_in_a_backward(patch)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x) if hooks else contextlib.nullcontext():
                h = call(operands)
        order = [launch[0] for launch in launches]
        h.backward(t.dh)
        assert_same_outputs({"h": h.detach(), **gradients()}, want, f"{layout}, {w}: {order}")
        assert [launch[0] for launch in launches[len(order) :]] == [_stages(t.weight is not None)[1]]
        return order

    def backend(operands):
        return clamped_swiglu(*operands, limit=t.L)

    held = ["pack"] * packs + ["fwd"]
    inside = ["fwd"] + ["pack"] * packs
    assert run(backend) == held
    assert _in_a_backward(lambda: run(backend, in_backward=False)) == held
    assert len(made) == 2 and all(len(kept) == 1 for kept in made)
    assert run(backend, in_backward=False) == inside  # hooks outside a backward
    assert run(backend, hooks=False) == ["fwd"]  # a backward without hooks
    assert run(lambda operands: pkg.CakeClampedSwiGLUDetFunction.apply(*operands, t.L)) == inside
    with pytest.raises(TypeError):
        pkg.CakeClampedSwiGLUDetFunction.apply(*_kept_leaves(t)[0], t.L, None)
    for name in ("_is_compiling", "_is_tracing"):
        with monkeypatch.context() as patch:
            patch.setattr(pkg, name, lambda: True)
            assert run(backend) == inside, name
    assert len(made) == 2

    for in_backward in (True, False):
        launches.clear()
        made.clear()
        with monkeypatch.context() as patch:
            if in_backward:
                _as_in_a_backward(patch)
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x), torch.no_grad():
                h = clamped_swiglu(*_kept_leaves(t)[0], limit=t.L)
        assert [launch[0] for launch in launches] == ["fwd"], in_backward  # no pack hook ran: nothing is saved
        assert len(made) == in_backward
        assert_same_bits(h, want["h"], f"h under no_grad ({in_backward})")
        assert not h.requires_grad and h.grad_fn is None


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("layout", ["merged", "split", "strided", "copied"])
def test_which_checkpoint_regions_launch_the_forward_again(layout, w, device, launches, monkeypatch):
    """Under ``torch.utils.checkpoint``, which regions launch the forward again in the backward.  Non-reentrant with
    early stop (the default), a region whose last saved tensors are the op's inputs launches none: the op alone (the
    down projection outside, or ``h`` itself backpropagated) and the gate/up projection with the op (the down
    projection outside); the recomputation stops when autograd saves the inputs, before the launch.  A region in which
    a later operation saves a tensor launches one: the op with the down projection (which saves ``h``) and the whole
    block (gate/up projection, op, down projection).  Early stop off (``early_stop=False`` or
    ``set_checkpoint_early_stop(False)``), a reentrant checkpoint and a direct ``CakeClampedSwiGLUDetFunction.apply``
    launch one in every region, also when its output is dropped.  The checkpoint's forward pass keeps no launch; a
    non-reentrant recomputation takes the held launch once.  Every way launches one forward before the backward and
    gives the bits of the same steps without a checkpoint, in every layout."""
    t = make_inputs(Case(129, layout, w), seed=32, device=device)
    p = _projections(t, seed=33, device=device)
    checkpoint = torch.utils.checkpoint.checkpoint
    op = functools.partial(clamped_swiglu, limit=t.L)
    block = _block(t.L)
    projected = _projected(t.L)
    made = _held_launches(monkeypatch)

    def gradients(out: torch.Tensor, leaves: list) -> dict:
        found = {"out": out.detach()}
        found.update({f"d{i}": leaf.grad for i, leaf in enumerate(leaves) if leaf is not None})
        return found

    def op_step(region):
        """``region(first, second, weight, wd)`` gives the down projection's output; its backward from ``dy``."""
        results: list = []

        def run(mark):
            operands, operand_gradients = _kept_leaves(t)
            wd = p.wd.detach().clone().requires_grad_(True)
            y = region(*operands, wd)
            mark()
            y.backward(p.dy)
            results.append({"out": y.detach(), **operand_gradients(), "d_wd": wd.grad})

        return run, results

    def counted(run) -> tuple[int, int, int, int]:
        """The forward launches before and after the mark (:func:`_recomputed`), then the held launches made before
        it and after it."""
        made.clear()
        held: list = []

        def marked(mark):
            def both():
                mark()
                held.append(len(made))

            run(both)

        before, after = _recomputed(launches, marked)
        (forward_held,) = held
        return before, after, forward_held, len(made) - forward_held

    def block_step(region):
        results: list = []

        def run(mark):
            leaves = [None if x is None else x.detach().clone().requires_grad_(True) for x in (p.x, p.wg, p.wu)]
            leaves = [*leaves, _leaves(t)[2], p.wd.detach().clone().requires_grad_(True)]
            y = region(*leaves)
            mark()
            y.backward(p.dy)
            results.append(gradients(y, leaves))

        return run, results

    def early_stop_off(*leaves):
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            return _down(checkpoint(op, *leaves[:3], use_reentrant=False), leaves[3])

    # (region, forward launches in the backward, held launches in the backward); none in the forward pass.
    op_cases = {
        "no checkpoint": (lambda f, s, w, wd: _down(op(f, s, w), wd), 0, 0),
        "op alone": (lambda f, s, w, wd: _down(checkpoint(op, f, s, w, use_reentrant=False), wd), 0, 1),
        "op and the down projection": (
            lambda f, s, w, wd: checkpoint(lambda *a: _down(op(*a[:3]), a[3]), f, s, w, wd, use_reentrant=False),
            1,
            1,
        ),
        "op alone, early_stop=False": (
            lambda f, s, w, wd: _down(checkpoint(op, f, s, w, use_reentrant=False, early_stop=False), wd),
            1,
            1,
        ),
        "op alone, set_checkpoint_early_stop(False)": (early_stop_off, 1, 1),
        "op alone, reentrant": (lambda f, s, w, wd: _down(checkpoint(op, f, s, w, use_reentrant=True), wd), 1, 0),
        "direct apply alone": (
            lambda f, s, w, wd: _down(
                checkpoint(pkg.CakeClampedSwiGLUDetFunction.apply, f, s, w, t.L, use_reentrant=False), wd
            ),
            1,
            0,
        ),
    }
    want = None
    for what, (region, recomputes, held) in op_cases.items():
        launches.clear()
        run, results = op_step(region)
        assert counted(run) == (1, recomputes, 0, held), what
        if want is None:
            want = results[0]
        assert_same_outputs(results[0], want, what)

    block_cases = {
        "block, no checkpoint": (block, 0, 0),
        "gate/up projection and the op": (
            lambda x, wg, wu, w, wd: _down(checkpoint(projected, x, wg, wu, w, use_reentrant=False), wd),
            0,
            1,
        ),
        "block": (lambda *a: checkpoint(block, *a, use_reentrant=False), 1, 1),
        "block, reentrant": (lambda *a: checkpoint(block, *a, use_reentrant=True), 1, 0),
        "block, early_stop=False": (lambda *a: checkpoint(block, *a, use_reentrant=False, early_stop=False), 1, 1),
    }
    want = None
    for what, (region, recomputes, held) in block_cases.items():
        launches.clear()
        run, results = block_step(region)
        assert counted(run) == (1, recomputes, 0, held), what
        if want is None:
            want = results[0]
        assert_same_outputs(results[0], want, what)

    launches.clear()
    operands, operand_gradients = _kept_leaves(t)

    def alone(mark):
        h = checkpoint(op, *operands, use_reentrant=False)
        mark()
        h.backward(t.dh)

    assert counted(alone) == (1, 0, 0, 1)
    found = operand_gradients()
    assert_same_outputs(found, {key: _functional(t)[key] for key in found}, "op alone, h backpropagated")


@pytest.mark.parametrize("layout", ["merged", "split"])
def test_a_cuda_graph_of_a_checkpointed_step_captures_the_recompute_only_when_needed(layout, device, launches):
    """A checkpointed step (forward, then ``torch.autograd.grad``) captured in one CUDA graph: with the op alone, or
    the gate/up projection and the op, in the region it captures one forward and one backward launch (no recompute),
    with the whole block one forward, the recompute and one backward; each replay gives the bits of the eager step
    without a checkpoint."""
    t = make_inputs(Case(257, layout), seed=34, device=device)
    p = _projections(t, seed=35, device=device)
    checkpoint = torch.utils.checkpoint.checkpoint
    op = functools.partial(clamped_swiglu, limit=t.L)
    block = _block(t.L)
    projected = _projected(t.L)
    regions = {
        "op alone": (lambda f, s, w, wd: _down(checkpoint(op, f, s, w, use_reentrant=False), wd), False, 0),
        "gate/up projection and the op": (
            lambda x, wg, wu, w, wd: _down(checkpoint(projected, x, wg, wu, w, use_reentrant=False), wd),
            True,
            0,
        ),
        "block": (lambda *a: checkpoint(block, *a, use_reentrant=False), True, 1),
    }

    def fresh(whole: bool) -> list:
        """New leaves of the region's operands (the eager reference and the captured step each take their own: a leaf
        used on the default stream before the capture cannot be captured)."""
        if whole:
            leaves = [None if x is None else x.detach().clone().requires_grad_(True) for x in (p.x, p.wg, p.wu)]
            return [*leaves, _leaves(t)[2], p.wd.detach().clone().requires_grad_(True)]
        return [*_leaves(t), p.wd.detach().clone().requires_grad_(True)]

    def op_down(f, s, w, wd):
        return _down(op(f, s, w), wd)

    for what, (region, whole, recomputes) in regions.items():
        reference = fresh(whole)
        y = (block if whole else op_down)(*reference)
        want = [y.detach(), *torch.autograd.grad(y, [x for x in reference if x is not None], p.dy)]
        leaves = fresh(whole)
        wants = [leaf for leaf in leaves if leaf is not None]
        marks: list = []

        def step(mark=lambda: None):
            y = region(*leaves)
            mark()
            return [y.detach(), *torch.autograd.grad(y, wants, p.dy)]

        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        launches.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = step(lambda: marks.append(len(launches)))
        stages = [launch[0] for launch in launches]
        assert stages[: marks[0]] == ["fwd"], what
        assert sorted(stages[marks[0] :]) == sorted(["fwd"] * recomputes + [_stages(True)[1]]), what
        for k in range(2):
            for out in outs:
                out.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            for i, (got, expected) in enumerate(zip(outs, want, strict=True)):
                assert_same_bits(got, expected, f"{what}: replay {k}, output {i}")
        marks.clear()


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
    small-R, the large-R or the fine program of each stage (forced through the route; ``bwd_unweighted`` has no large-R
    one and runs its small-R one there), run once
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
def test_an_unweighted_step_captured_before_any_eager_unweighted_forward(layout, device, launches) -> None:
    """An unweighted forward and backward with preallocated outputs, captured in a CUDA graph with no unweighted forward
    through the device's launchers before (fresh launchers; only a weighted call and an eager unweighted backward,
    which load the libraries and make no ignored weight): the capture makes the device's cached FP32 buffer, the
    captured forward passes it as its ignored weight and the captured backward runs ``bwd_unweighted``, which takes no
    weight, weight flag or weight gradient; later eager unweighted forwards pass the same buffer, and each replay on new
    inputs copied into the captured tensors gives the bits of eager calls."""
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
    assert pkg._launcher("fwd", first.get_device()).weight is None  # a weighted call makes no ignored weight
    launches.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(first, second, None, limit=a.L, out=h)
        backward(dh, first, second, None, limit=a.L, grad_gate_up_or_gate=d_first, grad_up=d_up)
    assert [stage for stage, _, _ in launches] == ["fwd", "bwd_unweighted"]
    cached = pkg._launcher("fwd", first.get_device()).weight
    assert cached is not None and cached.dtype == torch.float32 and tuple(cached.shape) == (1,)
    assert cached.device == first.device
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

    def record(name, values, grid):
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
    """A merged input reaches the kernels as the caller's tensor itself, passed as both the gate and the up with the
    up's column offset 2304 (the binding moves the up's pointer, so no view is made), and its new gradient as both
    ``d_gate`` and ``d_up`` with the same offset; row slices of wider buffers reach them as views of the caller's memory
    (their row strides, column offsets 0); rows the kernels cannot read are copied once to contiguous rows; a single
    row's stride is passed as 0."""
    m = make_inputs(Case(33, "merged"), seed=2, device=device)
    forward(m.first, None, m.weight, limit=L)
    _, rows, values = launches[-1]
    assert rows == 33
    assert values["gate"] is m.gate_up and values["up"] is m.gate_up and values["up_col"] == WIDTH
    assert values["gate_stride"] == values["up_stride"] == 2 * WIDTH
    assert values["w"] is m.weight and values["weighted"] == 1
    assert values["limit"] == L and values["h_stride"] == WIDTH
    for weight in (m.weight, None):
        d_gate_up = backward(m.dh, m.first, None, weight, limit=L)[0]
        _, _, values = launches[-1]
        assert values["gate"] is m.gate_up and values["up"] is m.gate_up and values["up_col"] == WIDTH
        assert values["d_gate"] is d_gate_up and values["d_up"] is d_gate_up and values["d_up_col"] == WIDTH
        assert values["d_gate_stride"] == values["d_up_stride"] == 2 * WIDTH

    s = make_inputs(Case(33, "strided"), seed=2, device=device)
    backward(s.dh, s.gate, s.up, s.weight, limit=L)
    _, _, values = launches[-1]
    for name, tensor in (("gate", s.gate), ("up", s.up), ("dh", s.dh)):
        assert values[name].data_ptr() == tensor.data_ptr(), name
        assert values[f"{name}_stride"] == tensor.stride(0), name
    assert values["up_col"] == values["d_up_col"] == 0

    c = make_inputs(Case(33, "copied", "none", L=7.3), seed=2, device=device)
    backward(c.dh, c.gate, c.up, None, limit=7.3)
    _, _, values = launches[-1]
    for name, tensor in (("gate", c.gate), ("up", c.up), ("dh", c.dh)):
        assert values[name].data_ptr() != tensor.data_ptr(), name
        assert values[name].is_contiguous() and values[f"{name}_stride"] == WIDTH, name
    assert tuple(values) == PARAMETERS["bwd_unweighted"]  # no weight, weight flag or weight gradient
    assert values["limit"] == fp32(7.3)
    assert values["up_col"] == values["d_up_col"] == 0

    one = make_inputs(Case(1, "merged"), seed=2, device=device)
    backward(one.dh, one.first, None, one.weight, limit=L)
    _, _, values = launches[-1]
    for name in ("gate", "up", "dh", "d_gate", "d_up"):
        assert values[f"{name}_stride"] == 0, name
    assert values["up_col"] == values["d_up_col"] == WIDTH


def test_the_bindings_refuse_column_offsets_outside_the_row(device) -> None:
    """Every program's binding takes the column offset of ``up`` (and of ``d_up``) that keeps the 2304 columns inside
    a row of the 2-D tensor and the moved address 16-byte aligned: the merged call's 2304 on a ``[R, 4608]`` tensor and
    0 on a ``[R, 2304]`` one run; a negative offset, one that is not a multiple of 8 and one past ``size(1) - 2304``
    raise ``ValueError`` naming the offset, and nothing is written."""
    R = 33
    arch = pkg.device_arch(device.index)
    m = make_inputs(Case(R, "merged"), seed=4, device=device)
    s = make_inputs(Case(R, "split"), seed=4, device=device)

    def values(t: types.SimpleNamespace) -> dict:
        merged = t.second is None
        gate, up, col = (t.gate_up, t.gate_up, WIDTH) if merged else (t.gate, t.up, 0)
        d_gate = torch.zeros(R, 2 * WIDTH if merged else WIDTH, dtype=torch.bfloat16, device=device)
        d_up = d_gate if merged else torch.zeros(R, WIDTH, dtype=torch.bfloat16, device=device)
        h = torch.zeros(R, WIDTH, dtype=torch.bfloat16, device=device)
        return {
            "gate": gate,
            "up": up,
            "w": t.weight,
            "h": h,
            "dh": t.dh,
            "d_gate": d_gate,
            "d_up": d_up,
            "d_weight": torch.zeros(R, dtype=torch.float32, device=device),
            "rows": R,
            "gate_stride": gate.stride(0),
            "up_stride": up.stride(0),
            "h_stride": h.stride(0),
            "dh_stride": t.dh.stride(0),
            "d_gate_stride": d_gate.stride(0),
            "d_up_stride": d_up.stride(0),
            "limit": t.L,
            "weighted": 1,
            "up_col": col,
            "d_up_col": col,
        }

    checked = 0
    for t in (m, s):
        base = values(t)
        past = base["up_col"] + 8
        for stage in STAGES:
            for name in dict.fromkeys(_program(stage, device.index, program) for program in PROGRAMS):
                entry, keys, grid = pkg._entry(name, arch), pkg.parameters(name), cake_jit.grid(name, R)
                entry(*(base[key] for key in keys), *grid)  # the offsets the entry points pass
                torch.cuda.synchronize(device)
                for key in ("up_col", "d_up_col"):
                    if key not in keys:
                        continue
                    for bad in (-8, 4, past):
                        refused = values(t)
                        refused[key] = bad
                        with pytest.raises(ValueError, match=f"'{key}' = {bad} must be"):
                            entry(*(refused[k] for k in keys), *grid)
                        for out in ("h", "d_gate", "d_up", "d_weight"):
                            assert int(torch.count_nonzero(refused[out])) == 0, (name, key, bad, out)
                        checked += 1
    # two layouts, three refused values, and per layout at least the up_col of fwd and both offsets of bwd and
    # bwd_unweighted
    assert checked >= 2 * 3 * (1 + 2 + 2), checked


def _launch_facts(launched: list, caller: set[int]) -> list:
    """The launches ``(name, rows, values, grid)`` (``launch_records``) with each tensor value as its dtype, shape,
    strides, storage offset and memory: the caller's storage (its address, one of ``caller``) or ``"new"`` (a copy or
    a new output).  The program (the route) and the grid are facts of the launch too."""
    facts = []
    for name, rows, values, grid in launched:
        found = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage().data_ptr()
                value = (
                    value.dtype,
                    tuple(value.shape),
                    value.stride(),
                    value.storage_offset(),
                    storage if storage in caller else "new",
                )
            found[key] = value
        facts.append((name, rows, found, tuple(grid)))
    return facts


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("rows", [1, 2, 33, 4097])
def test_one_pass_checks_give_the_launches_of_the_full_checks(rows, w, device, launch_records, monkeypatch) -> None:
    """With the one-pass checks of the common call on and off (``_FAST_CHECKS``), every layout (merged, two tensors,
    row slices of wider buffers, rows the host copies), with new and with preallocated outputs, gives the same launches
    (the program the route runs and the grid; each tensor's dtype, shape, strides, storage offset and memory, the
    caller's or new; every scalar) and the same output bits; each launch runs the program of its stage that the route
    gives for its rows.  With them on, a merged or two-tensor call of contiguous tensors runs none of the full checks:
    the forward and the backward step each check and launch it themselves (no call of :func:`_forward`,
    :func:`_backward` or :func:`_check_tensor`)."""
    checked = _full_checks(monkeypatch)
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
                launch_records.clear()
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
                runs.append((_launch_facts(launch_records, caller), outputs))
                if fast and layout in ("merged", "split"):
                    assert not checked, (layout, preallocated, checked)
                elif not fast:
                    assert "_check_tensor" in checked, (layout, preallocated, checked)
            what = f"{layout}, preallocated={preallocated}"
            (fast_launches, fast_outputs), (full_launches, full_outputs) = runs
            stages = _stages(w != "none")
            assert [cake_jit.MODULES[name]["stage"] for name, _, _, _ in fast_launches] == list(stages), what
            for stage, (name, _, _, grid) in zip(stages, fast_launches, strict=True):
                assert name == _routed(stage, rows, device.index), (what, stage, name)
                assert grid == cake_jit.grid(name, rows), (what, name, grid)
            assert fast_launches == full_launches, what
            for got, want in zip(fast_outputs, full_outputs, strict=True):
                assert (got is None) == (want is None), what
                if got is not None:
                    assert_same_bits(got, want, what)


class _Subclass(torch.Tensor):
    """A tensor subclass: the full checks take it (``isinstance``), the one-pass checks leave it to them (exact
    types)."""


def _odd_calls(t: types.SimpleNamespace, device) -> dict:
    """Calls the one-pass checks of the common call leave to the full checks, each the plain call ``t`` with one
    argument changed (``(first, second, weight, dh, out, grads)``, ``grads`` the backward's output keywords): refused
    for a type, a dtype, a device, a shape, an output rule or the row count, or valid and launched by the full checks
    (a tensor subclass, a non-contiguous or misaligned input, which is copied, a row slice of a wider output, no
    rows).  Some fail only one check of the step's own (a contiguous ``dh`` that is not 16-byte aligned beside an
    aligned, contiguous first operand; contiguous operands of another width; a contiguous, misaligned up or output),
    so that check alone sends them to the full checks."""
    first, second, weight, dh = t.first, t.second, t.weight, t.dh
    merged, rows, cols = second is None, t.R, t.first.shape[1]

    def misaligned(x: torch.Tensor) -> torch.Tensor:
        buffer = torch.empty(x.numel() + 8, dtype=x.dtype, device=x.device)
        view = buffer[1 : 1 + x.numel()].view(x.shape)
        view.copy_(x)
        return view

    def misaligned_empty(*shape: int) -> torch.Tensor:
        return misaligned(torch.zeros(*shape, dtype=torch.bfloat16, device=device))

    def many(x: torch.Tensor) -> torch.Tensor:
        return x[:1].expand(cake_jit.MAX_ROWS, *x.shape[1:])

    calls = {
        "plain": (first, second, weight, dh, None, {}),
        "an FP16 first operand": (first.half(), second, weight, dh, None, {}),
        "an FP32 dh": (first, second, weight, dh.float(), None, {}),
        "a 3-D first operand": (first[None], second, weight, dh, None, {}),
        "a first operand of another width": (first[:, :-8], second, weight, dh, None, {}),
        "a list as the first operand": ([1.0], second, weight, dh, None, {}),
        "dh one row short": (first, second, weight, dh[:-1], None, {}),
        "dh on the CPU": (first, second, weight, dh.cpu(), None, {}),
        "a subclass first operand": (first.as_subclass(_Subclass), second, weight, dh, None, {}),
        "a subclass dh": (first, second, weight, dh.as_subclass(_Subclass), None, {}),
        "a misaligned first operand and dh": (misaligned(first), second, weight, misaligned(dh), None, {}),
        "a contiguous, misaligned dh": (first, second, weight, misaligned(dh), None, {}),
        "contiguous operands of another width": (
            first[:, :-8].contiguous(),
            None if merged else second[:, :-8].contiguous(),
            weight,
            dh,
            None,
            {},
        ),
        "a contiguous, misaligned out and gradient": (
            first,
            second,
            weight,
            dh,
            misaligned_empty(rows, WIDTH),
            {"grad_gate_up_or_gate": misaligned_empty(rows, cols)},
        ),
        "no rows": (
            first[:0],
            None if merged else second[:0],
            None if weight is None else weight[:0],
            dh[:0],
            None,
            {},
        ),
        "too many rows": (
            many(first),
            None if merged else many(second),
            None if weight is None else many(weight),
            many(dh),
            None,
            {},
        ),
        "an out of another shape": (first, second, weight, dh, torch.empty_like(dh)[:-1], {}),
        "a row slice of a wider out": (
            first,
            second,
            weight,
            dh,
            torch.empty(rows, WIDTH + 16, dtype=torch.bfloat16, device=device)[:, 8 : 8 + WIDTH],
            {"grad_gate_up_or_gate": torch.empty(rows, cols + 16, dtype=torch.bfloat16, device=device)[:, 8:-8]},
        ),
        "a gradient of another shape": (
            first,
            second,
            weight,
            dh,
            None,
            {"grad_gate_up_or_gate": torch.empty(rows + 1, cols, dtype=torch.bfloat16, device=device)},
        ),
    }
    if merged:
        calls["a grad_up with a merged input"] = (first, second, weight, dh, None, {"grad_up": torch.empty_like(dh)})
    else:
        calls["an up one row short"] = (first, second[:-1], weight, dh, None, {})
        calls["an FP32 up"] = (first, second.float(), weight, dh, None, {})
        calls["a subclass up"] = (first, second.as_subclass(_Subclass), weight, dh, None, {})
        calls["a contiguous, misaligned up"] = (first, misaligned(second), weight, dh, None, {})
        grad_up = misaligned_empty(rows, WIDTH)
        calls["a contiguous, misaligned grad_up"] = (first, second, weight, dh, None, {"grad_up": grad_up})
    if weight is None:
        grad_weight = torch.empty(rows, device=device)
        calls["a grad_weight without a weight"] = (first, second, None, dh, None, {"grad_weight": grad_weight})
    else:
        calls["a weight on the CPU"] = (first, second, weight.cpu(), dh, None, {})
        calls["a BF16 weight"] = (first, second, weight.bfloat16(), dh, None, {})
        calls["a weight one row short"] = (first, second, weight[:-1], dh, None, {})
        calls["a subclass weight"] = (first, second, weight.as_subclass(_Subclass), dh, None, {})
        strided = torch.empty(2 * rows, device=device)[::2]
        strided.copy_(weight)
        calls["a strided weight"] = (first, second, strided, dh, None, {})
        calls["a grad_weight of another length"] = (
            first,
            second,
            weight,
            dh,
            None,
            {"grad_weight": torch.empty(rows + 1, device=device)},
        )
    return calls


def _calls_across_two_gpus(t: types.SimpleNamespace, other: torch.device) -> dict:
    """Calls of the inputs ``t`` with one tensor on the GPU ``other`` (as :func:`_odd_calls` gives them): ``dh``, the
    first operand, the up, the weight or an output buffer.  A step that takes that tensor with one on the first GPU
    refuses the call for mixed devices; a forward does not take ``dh``, and the forward of a merged unweighted input
    moved to ``other`` runs there."""
    first, second, weight, dh = t.first, t.second, t.weight, t.dh
    rows, cols = t.R, first.shape[1]

    def new(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.zeros(*shape, dtype=dtype, device=other)

    calls = {
        "dh on the other GPU": (first, second, weight, dh.to(other), None, {}),
        "the first operand on the other GPU": (first.to(other), second, weight, dh, None, {}),
        "an out and a gradient on the other GPU": (
            first,
            second,
            weight,
            dh,
            new(rows, WIDTH),
            {"grad_gate_up_or_gate": new(rows, cols)},
        ),
    }
    if second is not None:
        calls["an up on the other GPU"] = (first, second.to(other), weight, dh, None, {})
        calls["a grad_up on the other GPU"] = (first, second, weight, dh, None, {"grad_up": new(rows, WIDTH)})
    if weight is not None:
        calls["a weight on the other GPU"] = (first, second, weight.to(other), dh, None, {})
        grad_weight = new(rows, dtype=torch.float32)
        calls["a grad_weight on the other GPU"] = (first, second, weight, dh, None, {"grad_weight": grad_weight})
    return calls


def _same_with_and_without_one_pass_checks(calls: dict, limit: float, launch_records: list, patch) -> int:
    """Each call of ``calls`` (``(first, second, weight, dh, out, grads)``, as :func:`_odd_calls` gives them) at
    ``limit``, as a forward and as a backward step, gives with the one-pass checks on what it gives with them off:
    the same exception (type and message) and no launch, or the same launches (:func:`_launch_facts`: the program, the
    grid, each tensor's dtype, shape, strides, storage offset and memory; every scalar) and the same output bits.
    Returns the number of step calls compared."""

    def outcome(call, caller: set[int]) -> tuple:
        launch_records.clear()
        try:
            found = call()
        except (TypeError, ValueError, RuntimeError) as e:
            return ("raised", type(e), str(e), _launch_facts(launch_records, caller))
        found = found if isinstance(found, tuple) else (found,)
        return ("ran", _launch_facts(launch_records, caller), [None if x is None else x.clone() for x in found])

    checked = 0
    for what, (first, second, weight, dh, out, grads) in calls.items():
        tensors = [x for x in (first, second, weight, dh, out, *grads.values()) if isinstance(x, torch.Tensor)]
        caller = {x.untyped_storage().data_ptr() for x in tensors}
        steps = {
            "forward": functools.partial(forward, first, second, weight, limit=limit, out=out),
            "backward": functools.partial(backward, dh, first, second, weight, limit=limit, **grads),
        }
        for step, call in steps.items():
            runs = []
            for fast in (True, False):
                patch.setattr(pkg, "_FAST_CHECKS", fast)
                runs.append(outcome(call, caller))
            (kind, *got), (want_kind, *want) = runs
            assert kind == want_kind, (what, step, runs)
            if kind == "raised":
                assert got[:2] == want[:2] and got[2] == want[2] == [], (what, step, got, want)
            else:
                assert got[0] == want[0], (what, step)
                for a, b in zip(got[1], want[1], strict=True):
                    assert (a is None) == (b is None), (what, step)
                    if a is not None:
                        assert_same_bits(a, b, f"{what}, {step}")
            checked += 1
    return checked


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("layout", ["merged", "split"])
def test_one_pass_checks_leave_every_other_call_to_the_full_checks(
    layout, w, device, launch_records, monkeypatch
) -> None:
    """Each kind of call the one-pass checks do not take (:func:`_odd_calls`: a type, a dtype, a device, a shape, an
    output rule or ``R >= MAX_ROWS`` refused; a tensor subclass, a non-contiguous or misaligned input, a row slice of a
    wider output or no rows, valid; calls that fail one check of the step's own only), as a forward and as a backward
    step, gives with the one-pass checks on what it gives with them off: the same exception (type and message) and no
    launch, or the same launches (the program and the grid; each tensor's dtype, shape, strides, storage offset and
    memory; every scalar) and the same output bits."""
    t = make_inputs(Case(33, layout, w), seed=31, device=device)
    calls = _odd_calls(t, device)
    checked = _same_with_and_without_one_pass_checks(calls, t.L, launch_records, monkeypatch)
    assert checked == 2 * len(calls) >= 2 * 21, checked


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("layout", ["merged", "split"])
def test_one_pass_checks_leave_calls_across_two_gpus_to_the_full_checks(
    layout, w, device, launch_records, monkeypatch
) -> None:
    """With two GPUs, each call with one tensor on the other GPU (:func:`_calls_across_two_gpus`: ``dh``, the first
    operand, the up, the weight, an output buffer), as a forward and as a backward step, gives with the one-pass checks
    on what it gives with them off: the same exception (type and message) and no launch, or the same launches and
    bits."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", 1 if device.index != 1 else 0)
    t = make_inputs(Case(33, layout, w), seed=32, device=device)
    calls = _calls_across_two_gpus(t, other)
    checked = _same_with_and_without_one_pass_checks(calls, t.L, launch_records, monkeypatch)
    assert checked == 2 * len(calls) >= 2 * 3, checked


@pytest.mark.parametrize("layout", ["merged", "split"])
def test_the_one_pass_checks_refuse_in_the_order_of_the_full_checks(
    layout, device, launch_records, monkeypatch
) -> None:
    """A call with two faults, one of them an output buffer whose metadata check raises (:func:`_plain` patched to
    raise ``RuntimeError`` for it, as reading the metadata of a tensor of some layouts can), raises with the one-pass
    checks on and off the error of the fault the full checks find first, and launches nothing: a bad ``limit`` before
    the forward's ``out`` and the backward's output buffers, and the ``grad_up`` rule of a merged input before
    ``grad_gate_up_or_gate``.  With a valid limit, the step's own checks do reach that buffer."""
    t = make_inputs(Case(33, layout), seed=33, device=device)
    rows, cols = t.R, t.first.shape[1]
    h = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
    d_first = torch.empty(rows, cols, dtype=torch.bfloat16, device=device)
    plain = pkg._plain

    def unreadable(tensor, *args):
        if tensor is h or tensor is d_first:
            raise RuntimeError("the metadata of this output cannot be read")
        return plain(tensor, *args)

    monkeypatch.setattr(pkg, "_plain", unreadable)
    first, second, weight, dh = t.first, t.second, t.weight, t.dh
    with pytest.raises(RuntimeError, match="cannot be read"):
        forward(first, second, weight, limit=t.L, out=h)
    with pytest.raises(RuntimeError, match="cannot be read"):
        backward(dh, first, second, weight, limit=t.L, grad_gate_up_or_gate=d_first)
    calls = {
        "a negative limit and an unreadable out": (
            "limit",
            lambda: forward(first, second, weight, limit=-1.0, out=h),
        ),
        "an integer limit past every float and an unreadable out": (
            "limit",
            lambda: forward(first, second, None, limit=10**400, out=h),
        ),
        "a NaN limit and an unreadable gradient": (
            "limit",
            lambda: backward(dh, first, second, weight, limit=float("nan"), grad_gate_up_or_gate=d_first),
        ),
    }
    if second is None:
        calls["a grad_up with a merged input and an unreadable gradient"] = (
            "grad_up",
            lambda: backward(dh, first, None, weight, limit=t.L, grad_gate_up_or_gate=d_first, grad_up=dh),
        )
    for what, (match, call) in calls.items():
        raised = []
        for fast in (True, False):
            monkeypatch.setattr(pkg, "_FAST_CHECKS", fast)
            launch_records.clear()
            with pytest.raises(ValueError, match=match) as e:
                call()
            assert launch_records == [], (what, fast)
            raised.append(str(e.value))
        assert raised[0] == raised[1], (what, raised)


@pytest.mark.parametrize("w", ["random", "none"])
@pytest.mark.parametrize("layout", ["merged", "split"])
def test_the_common_call_passes_the_row_strides_of_its_tensors(layout, w, device, launches, monkeypatch) -> None:
    """The common call passes the row strides of its contiguous tensors as constants, without reading them: at 2, 33
    and 4097 rows, with new and with preallocated outputs, in the steps, in an autograd and a checkpointed autograd
    step and in gradients computed with ``create_graph=True``, every ``<name>_stride`` of every launch is ``stride(0)``
    of the tensor ``<name>`` it passes (the merged tensor's for its up half, the merged gradient's for ``d_up``), and at
    1 row each is 0.  Every one of these calls, the autograd function's forward and backward too (also the backward
    that records the gradients' graph, through ``_Gradients``), is checked by the step itself and launched there (the
    checkpoint's recomputation, in the backward, keeps its launch for :func:`_held` and stops before it): none of the
    full checks runs (no call of :func:`_forward`, :func:`_backward` or :func:`_check_tensor`)."""
    checked = _full_checks(monkeypatch)
    for rows in (1, 2, 33, 4097):
        t = make_inputs(Case(rows, layout, w), seed=19, device=device)
        merged = t.second is None
        launches.clear()
        step_outputs(forward, backward, t)
        h = torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
        d_first = torch.empty(rows, 2 * WIDTH if merged else WIDTH, dtype=torch.bfloat16, device=device)
        d_up = None if merged else torch.empty(rows, WIDTH, dtype=torch.bfloat16, device=device)
        d_weight = None if t.weight is None else torch.empty(rows, device=device)
        forward(t.first, t.second, t.weight, limit=t.L, out=h)
        grads = {"grad_gate_up_or_gate": d_first, "grad_up": d_up, "grad_weight": d_weight}
        backward(t.dh, t.first, t.second, t.weight, limit=t.L, **grads)
        assert not checked, (rows, "steps", checked)
        _autograd_step(t)
        assert not checked, (rows, "autograd step", checked)
        _autograd_step(t, checkpoint=True)
        assert not checked, (rows, "checkpointed autograd step", checked)
        leaves = _leaves(t)
        wants = [leaf for leaf in leaves if leaf is not None]
        grads = torch.autograd.grad(clamped_swiglu(*leaves, limit=t.L), wants, t.dh, create_graph=True)
        assert all(grad.requires_grad for grad in grads), rows  # the backward ran through ``_Gradients``
        assert not checked, (rows, "autograd step with create_graph=True", checked)
        # The checkpointed step launches no recompute: the op alone is the region (in the recomputation the forward
        # runs after autograd saved the inputs, so the recomputation stops before it).
        assert len(launches) == 2 + 2 + 2 + 2 + 2, (rows, [stage for stage, _, _ in launches])
        for stage, launched_rows, values in launches:
            assert launched_rows == rows
            strides = {key: value for key, value in values.items() if key.endswith("_stride")}
            assert len(strides) == {"fwd": 3, "bwd": 5, "bwd_unweighted": 5}[stage], (stage, strides)
            for key, value in strides.items():
                tensor = values[key.removesuffix("_stride")]
                assert value == (tensor.stride(0) if rows > 1 else 0), (rows, stage, key, value, tensor.stride())
                assert rows == 1 or value == tensor.shape[1], (rows, stage, key)  # contiguous rows


def test_one_launcher_per_stage_and_device_serves_every_call(device, monkeypatch, request) -> None:
    """Every call on a device, the one-pass checks' and the full checks', weighted and unweighted, the steps and the
    autograd function, launches through the device's one launcher per stage (``pkg._launcher``), made once, at the
    stage's first launch: the test counts every :class:`_Launcher` made (by the package's own cache of launchers,
    emptied before the calls and again after the test) and every FFI entry looked up (:func:`_entry`, which a launcher
    calls once per program), so a step that made its own launcher or looked up its own entry at a call, or a cache
    that dropped a launcher, would count more.  Only an unweighted forward makes the forward launcher's ignored weight:
    weighted calls and backward steps leave it None, and every later unweighted forward passes the same buffer."""
    made: list = []
    looked_up: list = []
    entry = pkg._entry

    class Counted(pkg._Launcher):
        def __init__(self, stage: str, index: int) -> None:
            made.append((stage, index))
            super().__init__(stage, index)

    def counted_entry(name: str, arch: str):
        looked_up.append(name)
        return entry(name, arch)

    monkeypatch.setattr(pkg, "_Launcher", Counted)
    monkeypatch.setattr(pkg, "_entry", counted_entry)
    # The package's own cache (``pkg._launcher``, whose body reads ``_Launcher`` from the module), not
    # :func:`_fresh_launchers`: emptied here, so it makes every launcher of the calls below, and after the test, so no
    # counted launcher outlives it.
    pkg._launcher.cache_clear()
    request.addfinalizer(pkg._launcher.cache_clear)
    index = device.index
    weighted = [make_inputs(Case(33, layout), seed=17, device=device) for layout in ("merged", "split", "strided")]
    for _ in range(2):
        for t in weighted:
            step_outputs(forward, backward, t)
            _autograd_step(t)
            _autograd_step(t, checkpoint=True)
            u = types.SimpleNamespace(**{**vars(t), "weight": None})
            backward(u.dh, u.first, u.second, None, limit=u.L)
    assert sorted(made) == [("bwd", index), ("bwd_unweighted", index), ("fwd", index)], made
    assert pkg._launcher("fwd", index).weight is None
    unweighted = [make_inputs(Case(33, layout, "none"), seed=18, device=device) for layout in ("merged", "copied")]
    buffers = set()
    for _ in range(2):
        for t in unweighted:
            step_outputs(forward, backward, t)
            _autograd_step(t, checkpoint=True)
            buffers.add(id(pkg._launcher("fwd", index).weight))
    assert sorted(made) == [("bwd", index), ("bwd_unweighted", index), ("fwd", index)], made
    assert looked_up and len(looked_up) == len(set(looked_up)), looked_up
    weight = pkg._launcher("fwd", index).weight
    assert len(buffers) == 1 and weight is not None and id(weight) in buffers
    assert weight.dtype == torch.float32 and tuple(weight.shape) == (1,) and float(weight) == 1.0


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
    output or launches a kernel: the caching allocator counts no allocation and nothing launches.  So are the limit and
    the architecture refusals of common calls (plain, contiguous, aligned merged and two-tensor operands, weighted and
    unweighted, forward and backward steps), which the steps raise themselves (no call of the full checks), and no
    refused call makes the forward launcher's ignored weight.  The same inputs in a valid call are copied and
    launched."""
    R = 33
    t = make_inputs(Case(R, "copied"), seed=3, device=device)
    p = make_inputs(Case(R, "split"), seed=4, device=device)
    m = make_inputs(Case(R, "merged"), seed=5, device=device)
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
    # Refused in the steps' own checks of the common call: the limit, then (below) the architecture.
    common = {
        "a negative limit, a common call": (ValueError, lambda: forward(p.gate, p.up, None, limit=-1.0)),
        "a NaN limit, a common merged call": (ValueError, lambda: forward(m.first, None, m.weight, limit=math.nan)),
        "a tensor limit, a common merged call": (TypeError, lambda: forward(m.first, None, None, limit=tensor_limit)),
        "an integer limit past every float, a common backward": (
            ValueError,
            lambda: backward(p.dh, p.gate, p.up, p.weight, limit=10**400),
        ),
        "a fraction limit past every float, a common merged backward": (
            ValueError,
            lambda: backward(m.dh, m.first, None, None, limit=fractions.Fraction(10**400)),
        ),
    }
    common_arch = {
        "another architecture, a common call": (RuntimeError, lambda: forward(p.gate, p.up, None, limit=L)),
        "another architecture, a common merged call": (RuntimeError, lambda: forward(m.first, None, m.weight, limit=L)),
        "another architecture, a common backward": (
            RuntimeError,
            lambda: backward(p.dh, p.gate, p.up, p.weight, limit=L),
        ),
        "another architecture, a common merged backward": (
            RuntimeError,
            lambda: backward(m.dh, m.first, None, None, limit=L),
        ),
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
        checked = _full_checks(patch)
        refuse(common)
        assert not checked, checked
    with monkeypatch.context() as patch:
        patch.setattr(pkg, "_arch_of_index", lambda index: None)
        refuse(
            {
                "another architecture": (RuntimeError, lambda: forward(gate, up, weight, limit=L)),
                "another architecture, merged": (RuntimeError, lambda: backward(dh, merged, None, None, limit=L)),
            }
        )
        checked = _full_checks(patch)
        refuse(common_arch)
        assert not checked, checked
    assert allocations() == before, "a refused call allocated a copy or an output"
    assert launches == []
    assert pkg._launcher("fwd", device.index).weight is None, "a refused call made the ignored weight"
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
