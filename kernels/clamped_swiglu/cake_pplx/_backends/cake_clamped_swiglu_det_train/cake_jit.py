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

Registry of the Cake clamped SwiGLU training forward and backward kernels of a
routed MoE expert (SM100 / SM103 / SM107), deterministic.

Every record of ``MODULES`` is one exported program: a CUDA device translation
unit plus a tvm-ffi binding, compiled once per architecture it is delivered
for.  A record is keyed by its program name
``cake_clamped_swiglu_det_train_<20 hex>`` and carries the host plan: the
``role`` it runs in, the ``stage`` it serves (``fwd``, ``bwd`` or
``bwd_unweighted``, its role without the ``_large`` or ``_fine`` suffix),
the architectures it is delivered for, the ``run`` entry and, in a large-R or
a fine role, ``rows_per_sm_windows``; ``library`` is its key.  The derived
leaves (``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding.  ``MODULES`` is a literal
table (the exporter and the manifest checker read it with
``ast.literal_eval``).  Every launch is a plain stream-ordered launch: nothing
cooperative, no PDL, no dynamic shared memory.

The stages:

- ``fwd``: per row ``r < rows`` and column ``c < 2304``, from the BF16 gate
  ``g`` and up ``u``, the row's FP32 route weight ``w`` and the FP32 limit
  ``L``: ``h = bf16(w * silu(min(g, L)) * clamp(u, -L, L))``, the route weight
  multiplied in FP32 before the one rounding (``w`` is 1 with ``weighted``
  0).
- ``bwd``: ``d_gate``, ``d_up`` (BF16, the inclusive clamp masks) and the FP32
  ``d_weight = sum over c of dh * a`` from the unweighted activation ``a``,
  one fixed-order FP32 sum per row (each lane's two FMA chains over its even
  and odd columns in load order, added, then XOR shuffles across the lanes).
  The host runs it for a weighted call.
- ``bwd_unweighted``: ``d_gate`` and ``d_up`` of an unweighted call (``w``
  1), and nothing else: no weight, no weight flag, no ``d_weight``.  It
  stores the bits ``bwd`` stores with ``weighted`` 0 (``dh * 1`` is ``dh``).

Each stage has three roles (``ROLES``).  The stage's own role (``fwd``,
``bwd``, ``bwd_unweighted``) is its small-R program, one on every
architecture.  Its large-R role (``fwd_large``, ``bwd_large``,
``bwd_unweighted_large``) and its fine role (``fwd_fine``, ``bwd_fine``,
``bwd_unweighted_fine``) have at most one program on each architecture
(``bwd_unweighted_large`` has none); the record's ``rows_per_sm_windows``
gives, per architecture, the windows of rows per SM in which it runs:
``[lo, hi]`` lists, ascending, each ``hi`` below the next ``lo``, only the
last ``hi`` None (no upper bound).  A fine record has one window ``[0, hi]``
on each of its architectures, below every large-R window of its stage: a
launch of at most ``hi * sm_count`` rows on a device of ``sm_count`` SMs runs
the fine program (:func:`fine`).  A fine program spreads each row over 9
tasks, one load of each input a task (a warp per task in ``fwd_fine`` and
``bwd_unweighted_fine``, a CTA of 9 warps per row in ``bwd_fine``), so a
row's serial chain is shorter when few rows share an SM.  A launch of more
rows runs the large-R program when ``lo * sm_count <= rows <= hi * sm_count``
for one of its windows (both bounds included) and the small-R program
otherwise (:func:`route`); an architecture without a large-R or a fine
program runs the small-R one in its place.  The programs of a stage differ in
their schedule only (the threads of a CTA, the tasks of a row and the loads
of a task, when the loads of a run are issued and when its values are
stored): they store the same bits for every input, so the route changes no
output bit.

The host finds a program through ``PROGRAM_OF`` (role -> architecture ->
program name, derived from the records) and never spells a program name:
retuning a kernel renames its program (the record's key and ``library``) and
leaves the host code as it is.  The host's own facts of each role's programs
are ``BUILD_OF``: the lanes that own a task, the tasks of a thread and the
tasks of a row (with the launch block's threads, the share of rows one CTA
owns) and the load width (the stride and alignment rule of every BF16
tensor).  Every kernel takes the row count, the limit, every BF16 row stride
(in elements) and, in ``fwd`` and ``bwd``, the weight flag as launch
arguments, so one library per program and architecture serves every row
count routed to it and both layouts.  The host launches nothing for
``rows == 0``.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
PACKAGE = "cake_pplx._backends.cake_clamped_swiglu_det_train"
LIBRARY_PREFIX = "cake_clamped_swiglu_det_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order.
STAGES = ("fwd", "bwd", "bwd_unweighted")
# The suffixes of a stage's large-R role and of its fine role.
LARGE = "_large"
FINE = "_fine"
# The roles of the records: each stage's small-R role (the stage itself), its large-R role and its fine role.
ROLES = tuple(stage + suffix for stage in STAGES for suffix in ("", LARGE, FINE))

MODULES: dict[str, dict[str, Any]] = {
    # The forward for small R: h = bf16(w * silu(min(g, L)) * clamp(u, -L, L)), 128 threads, 32 lanes a row, one row
    # a thread (4 rows a CTA), a lane's 9 loads in 3 runs, each run's loads issued while the run before it computes.
    "cake_clamped_swiglu_det_train_d38b827f821aba65855f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_d38b827f821aba65855f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_d38b827f821aba65855f",
        "role": "fwd",
        "stage": "fwd",
    },
    # The forward for large R on SM100 and SM103: 256 threads, 32 lanes a row, one row a thread (8 rows a CTA), a
    # lane's 9 loads in one run.
    "cake_clamped_swiglu_det_train_581a0b3b2e69bde65b1c": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_581a0b3b2e69bde65b1c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_581a0b3b2e69bde65b1c",
        "rows_per_sm_windows": {"sm_100a": [[112, None]], "sm_103a": [[208, None]]},
        "role": "fwd_large",
        "stage": "fwd",
    },
    # The forward for large R on SM107: 128 threads, 32 lanes a row, one row a thread (4 rows a CTA), a lane's 9 loads
    # in one run, the values of each load stored as soon as they are computed.
    "cake_clamped_swiglu_det_train_8e8ba1e7a3bd840b207b": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_8e8ba1e7a3bd840b207b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_8e8ba1e7a3bd840b207b",
        "rows_per_sm_windows": {"sm_107a": [[384, None]]},
        "role": "fwd_large",
        "stage": "fwd",
    },
    # The forward for the fewest rows: 128 threads, a warp per (row, load) task (9 tasks a row, 4 tasks a CTA), a
    # lane's one load of each input of its task.
    "cake_clamped_swiglu_det_train_afd0b9fcbf881e8b2cf3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_afd0b9fcbf881e8b2cf3",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_afd0b9fcbf881e8b2cf3",
        "rows_per_sm_windows": {"sm_100a": [[0, 8]], "sm_103a": [[0, 4]], "sm_107a": [[0, 4]]},
        "role": "fwd_fine",
        "stage": "fwd",
    },
    # The backward for small R: d_gate, d_up (BF16) and d_weight (FP32, one fixed-order warp sum a row), 128 threads,
    # 32 lanes a row, one row a thread (4 rows a CTA), a lane's 9 loads in 3 runs, each run's loads issued while the
    # run before it computes.
    "cake_clamped_swiglu_det_train_0f584d80b1575fdf3bc5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "w"], ["buffer", "d_gate"], ["buffer", "d_up"], ["buffer", "d_weight"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["parameter", "d_up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_0f584d80b1575fdf3bc5",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_0f584d80b1575fdf3bc5",
        "role": "bwd",
        "stage": "bwd",
    },
    # The backward for large R: 128 threads, 32 lanes a row, one row a thread (4 rows a CTA), a lane's 9 loads in 3
    # runs, one run at a time.
    "cake_clamped_swiglu_det_train_630f5af11e3ba9d58de9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "w"], ["buffer", "d_gate"], ["buffer", "d_up"], ["buffer", "d_weight"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["parameter", "d_up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_630f5af11e3ba9d58de9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_630f5af11e3ba9d58de9",
        "rows_per_sm_windows": {
            "sm_100a": [[12, 21], [32, 41], [48, None]],
            "sm_103a": [[12, 21], [32, 41], [48, None]],
            "sm_107a": [[12, 21], [35, 41], [48, None]],
        },
        "role": "bwd_large",
        "stage": "bwd",
    },
    # The weighted backward for the fewest rows: a CTA of 288 threads per row, warp r owns load r of each lane (9 tasks
    # a row); warps 1-8 hand their dh and activation words to warp 0 in static shared memory, and warp 0 runs the
    # d_weight chain of the other programs in load order.
    "cake_clamped_swiglu_det_train_c30f690752c972d064fa": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "w"], ["buffer", "d_gate"], ["buffer", "d_up"], ["buffer", "d_weight"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["parameter", "d_up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_c30f690752c972d064fa",
        "launch": {
            "block": (288, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_c30f690752c972d064fa",
        "rows_per_sm_windows": {"sm_100a": [[0, 2]], "sm_103a": [[0, 2]], "sm_107a": [[0, 2]]},
        "role": "bwd_fine",
        "stage": "bwd",
    },
    # The backward of an unweighted call: d_gate and d_up (BF16) only, no weight and no d_weight, 128 threads, 32 lanes
    # a (row, run) task (3 runs a row, 4 tasks a CTA), a lane's 3 loads of its run.
    "cake_clamped_swiglu_det_train_f8ece5d554866a83bfdb": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "d_gate"], ["buffer", "d_up"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "up_col"], ["parameter", "d_up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_f8ece5d554866a83bfdb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_f8ece5d554866a83bfdb",
        "role": "bwd_unweighted",
        "stage": "bwd_unweighted",
    },
    # The backward of an unweighted call for the fewest rows: 128 threads, a warp per (row, load) task (9 tasks a row,
    # 4 tasks a CTA), a lane's one load of each input of its task.
    "cake_clamped_swiglu_det_train_ebff32b92dd1308d1c58": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "d_gate"], ["buffer", "d_up"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "up_col"], ["parameter", "d_up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_ebff32b92dd1308d1c58",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_ebff32b92dd1308d1c58",
        "rows_per_sm_windows": {"sm_100a": [[0, 4]], "sm_103a": [[0, 4]], "sm_107a": [[0, 2]]},
        "role": "bwd_unweighted_fine",
        "stage": "bwd_unweighted",
    },
}


def _programs() -> dict[str, dict[str, str]]:
    """Role -> architecture -> the program name of the record of that role delivered for that architecture."""
    table: dict[str, dict[str, str]] = {role: {} for role in ROLES}
    for name, record in MODULES.items():
        for arch in record["arches"]:
            other = table[record["role"]].setdefault(arch, name)
            if other != name:
                raise RuntimeError(f"records {other} and {name} both run role {record['role']!r} on {arch}")
    return table


# Role -> architecture -> program name (host code reads this, never a spelled name; not a literal, so the exporter's
# reader skips it).
PROGRAM_OF: dict[str, dict[str, str]] = _programs()

# I: the intermediate width of the expert (the gate, up, h and gradient columns).
WIDTH = 2304

# Role -> the build facts of its programs that the host relies on.  Every program of a stage keeps one lane layout
# (``lanes`` and ``width_bits``: a lane's columns), so the programs store the same bits.  ``lanes`` lanes own a task
# and a thread owns ``rows_per_thread`` tasks, so a CTA of ``threads`` threads (the launch block) owns
# ``threads / lanes * rows_per_thread`` consecutive tasks.  A row is ``tasks_per_row`` consecutive tasks (1 when
# absent: a task is a row; ``bwd_unweighted``: 3, a task is one run of 768 columns of a row; a fine role: 9, a task is
# one load of 256 columns of a row), so the grid is ``ceil(rows * tasks_per_row / (threads / lanes *
# rows_per_thread))`` CTAs (:func:`grid`).  ``width_bits`` is the load width.  A program built with other values of
# these facts is another program (another name): change this table with the ``MODULES`` keys.
BUILD_OF: dict[str, dict[str, int]] = {
    "fwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "fwd_large": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "fwd_fine": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 9},
    "bwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd_large": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd_fine": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 9},
    "bwd_unweighted": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 3},
    "bwd_unweighted_fine": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 9},
}

# The kernels load ``width_bits`` at a time: every BF16 row stride is a multiple of one load (8 BF16 values for
# 128-bit loads) and every BF16 base address is aligned to one load (16 bytes).  One rule for every tensor of the
# family (the strictest of the stages), so a forward's output is a backward's input as it is.
VECTOR_ELEMENTS = max(build["width_bits"] for build in BUILD_OF.values()) // 16
ALIGNMENT_BYTES = 2 * VECTOR_ELEMENTS

# ``rows`` is a 32-bit kernel parameter, and the ``fwd`` and ``bwd`` kernels round it up to whole blocks of rows in 32
# bits: fewer rows than this.  The ``bwd_unweighted`` kernel counts its ``rows * 3`` tasks and their blocks in 64 bits
# (``rows * 3`` passes ``2**31 - 1`` from 715,827,883 rows), and its grid, ``ceil(rows * 3 / 4)`` CTAs, stays below
# ``2**31 - 1`` for every row count below this.
MAX_ROWS = 2**31 - 256
# The largest grid a launch takes (the bindings refuse a larger one).  Below ``MAX_ROWS`` rows, every small-R and
# large-R program and ``bwd_fine`` (``rows`` CTAs) stay below it.  A grid of ``ceil(9 rows / 4)`` CTAs (the
# ``fwd_fine`` and ``bwd_unweighted_fine`` programs) passes it from about 954 M rows.  Those two kernels count their
# tasks and blocks in 64 bits and stride over the blocks by the grid, so :func:`grid` caps a launch of more rows here.
# ``bwd_fine`` counts its rows in 32 bits: its row loop (``row += gridDim.x``) stays in range only while the row
# count plus the grid is at most ``2**31``, so on its grid of ``rows`` CTAs it takes at most ``2**30`` rows.  The route
# sends it at most its window's ``hi * sm_count`` rows: at most ``MAX_FINE_ROWS_PER_SM * sm_count``, and
# ``2 * sm_count`` with its shipped window ``[0, 2]``.  :func:`fine` refuses a fine window past
# :data:`MAX_FINE_ROWS_PER_SM` rows per SM and one whose last row count needs a grid past this cap, so a routed launch
# never reaches the cap.  That bound keeps ``bwd_fine`` within its ``2**30`` rows on a GPU of at most ``2**24`` SMs
# (about 16.8 M); :func:`fine` checks no SM count.  On the delivered SM counts (148 to 212) a fine program takes at
# most 13,568 rows, far below these 32-bit limits.
MAX_GRID = 2**31 - 1

# The most rows per SM a fine window may reach (its ``hi``).  The fine programs are for the fewest rows: the shipped
# windows end at 2 to 8 rows per SM.  64 is 8 times the largest of them, room for a retune.  The package's route table
# holds one entry per row count up to the fine window's last row count or the last large-R edge, whichever is larger,
# so the bound lets the fine window add at most ``64 * sm_count`` entries (13,568 on 212 SMs).  On the delivered SM
# counts (148 to 212) it also keeps a fine grid far below the 32-bit limits above (``bwd_fine``: at most
# ``64 * sm_count`` CTAs and rows).
MAX_FINE_ROWS_PER_SM = 64


def route(stage: str, arch: str, sm_count: int) -> tuple[tuple[int, ...], str, str]:
    """``(edges, small, large)`` of ``stage`` on ``arch`` with ``sm_count`` SMs: a launch of ``rows`` rows runs the
    program ``large`` (the stage's large-R role) when ``lo * sm_count <= rows <= hi * sm_count`` for a window
    ``[lo, hi]`` of its record's ``rows_per_sm_windows[arch]`` (``hi`` None: no upper bound), and the program
    ``small`` (the stage's role) otherwise.  ``edges`` are the last row count of each run of one program, ascending:
    ``lo * sm_count - 1`` and ``hi * sm_count`` of each window (an open window's ``lo * sm_count - 1`` alone).  A
    launch runs ``large`` exactly when the count of edges below ``rows``, ``bisect.bisect_left(edges, rows)``, is
    odd.  Without a large-R program on ``arch``, ``edges`` is empty and ``large`` is ``small``."""
    small = PROGRAM_OF[stage][arch]
    large = PROGRAM_OF[stage + LARGE].get(arch)
    if large is None:
        return (), small, small
    windows = MODULES[large]["rows_per_sm_windows"][arch]
    bounds = [bound for window in windows for bound in window]
    if (
        not windows
        or any(len(window) != 2 for window in windows)
        or any(bound is None for bound in bounds[:-1])
        or any(type(bound) is not int for bound in bounds if bound is not None)
        or bounds[0] < 0
    ):
        raise RuntimeError(f"{large}: rows_per_sm_windows[{arch!r}] {windows!r} are not [lo, hi] lists of integers")
    bounds = [bound for bound in bounds if bound is not None]
    if any(below >= above for below, above in zip(bounds, bounds[1:])):
        raise RuntimeError(f"{large}: rows_per_sm_windows[{arch!r}] {windows!r} do not ascend without touching")
    edges = tuple(bound * int(sm_count) - (index % 2 == 0) for index, bound in enumerate(bounds))
    return edges, small, large


def fine(stage: str, arch: str, sm_count: int) -> tuple[int, str]:
    """``(last, name)`` of ``stage`` on ``arch`` with ``sm_count`` SMs: a launch of ``1 <= rows <= last`` rows runs
    the program ``name``, the stage's fine role, ahead of :func:`route`.  Its record's ``rows_per_sm_windows[arch]``
    is one window ``[0, hi]`` with an integer ``hi >= 1``, so ``last = hi * sm_count``.  The window must end below
    the first window of the stage's large-R program (``hi`` below its ``lo``: ``last`` at most the first edge of
    :func:`route`), so the fine program never takes a row count that :func:`route` gives the large-R one.  ``hi``
    must be at most :data:`MAX_FINE_ROWS_PER_SM`, so the fine window adds at most ``MAX_FINE_ROWS_PER_SM * sm_count``
    entries to the package's route table.  Its grid at ``last`` rows must not pass :data:`MAX_GRID`, so a routed
    launch's grid is :func:`grid`'s, never capped.  ``bwd_fine`` counts its rows in 32 bits and takes at most
    ``2**30`` rows (see the :data:`MAX_GRID` comment); the route sends it at most ``last`` rows (``2 * sm_count``
    with its shipped window ``[0, 2]`` on every architecture).  Without a fine program on ``arch``, ``last`` is 0 (no
    launch runs it) and ``name`` is the stage's small-R program."""
    name = PROGRAM_OF[stage + FINE].get(arch)
    if name is None:
        return 0, PROGRAM_OF[stage][arch]
    windows = MODULES[name]["rows_per_sm_windows"][arch]
    if (
        len(windows) != 1
        or len(windows[0]) != 2
        or type(windows[0][0]) is not int
        or windows[0][0] != 0
        or type(windows[0][1]) is not int
        or windows[0][1] < 1
    ):
        raise RuntimeError(f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} is not one window [0, hi], hi >= 1")
    last = windows[0][1] * int(sm_count)
    edges = route(stage, arch, sm_count)[0]
    if edges and last > edges[0]:
        raise RuntimeError(
            f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} reaches the first large-R window of {stage!r}, "
            f"from {edges[0] + 1} rows on {sm_count} SMs"
        )
    if windows[0][1] > MAX_FINE_ROWS_PER_SM:
        raise RuntimeError(
            f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} passes {MAX_FINE_ROWS_PER_SM} rows per SM, the most a "
            "fine window may reach"
        )
    per_cta, per_row = cta_tasks(name)
    if -(-last * per_row // per_cta) > MAX_GRID:
        raise RuntimeError(
            f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} needs a grid of more than {MAX_GRID} CTAs at its last "
            f"row count, {last} rows on {sm_count} SMs"
        )
    return last, name


def cta_tasks(name: str) -> tuple[int, int]:
    """``(per_cta, per_row)`` of program ``name``: one CTA owns ``per_cta = threads / lanes * rows_per_thread``
    consecutive tasks and a row is ``per_row`` tasks (``BUILD_OF`` of its role, ``tasks_per_row`` 1 when absent; the
    threads of its launch block), so one CTA owns ``per_cta / per_row`` rows."""
    record = MODULES[name]
    build = BUILD_OF[record["role"]]
    threads = int(record["launch"]["block"][0])
    per_cta = threads // build["lanes"] * build["rows_per_thread"]
    per_row = build.get("tasks_per_row", 1)
    if per_cta < 1 or threads % build["lanes"] != 0 or per_row < 1:
        raise RuntimeError(
            f"{name}: a launch block of {threads} threads does not hold whole tasks of {build['lanes']} "
            "lanes (the registry's launch leaves are written by the exporter)"
        )
    return per_cta, per_row


def grid(name: str, rows: int) -> tuple[int, int, int]:
    """The launch grid of program ``name`` for ``rows >= 1`` rows: one CTA per block of ``per_cta`` of the
    ``rows * per_row`` tasks (:func:`cta_tasks`), ``ceil(rows * per_row / per_cta)`` CTAs on the x axis, at most
    :data:`MAX_GRID` (the kernels stride over the blocks from ``blockIdx.x``, so any row count fits one axis and a
    smaller grid runs every task).  Only a fine program launched at more rows than its window holds reaches the cap
    (:func:`fine`), so the package's launches pass the uncapped count."""
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    per_cta, per_row = cta_tasks(name)
    return (min(-(-rows * per_row // per_cta), MAX_GRID), 1, 1)
