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
``role`` it runs in, the ``stage`` it serves (``fwd`` or ``bwd``, its role
without the ``_large`` suffix), the architectures it is delivered for, the
``run`` entry and, in a large-R role, ``rows_per_sm_windows``; ``library`` is
its key.  The derived leaves (``kernel_symbol``, ``launch``, ``arg_plan``) are
written by the exporter from the rendered binding.  ``MODULES`` is a literal
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

Each stage has two roles (``ROLES``).  The stage's own role (``fwd``,
``bwd``) is its small-R program, one on every architecture.  Its large-R role
(``fwd_large``, ``bwd_large``) has at most one program on each architecture;
the record's ``rows_per_sm_windows`` gives, per architecture, the windows of
rows per SM in which it runs: ``[lo, hi]`` lists, ascending, each ``hi`` below
the next ``lo``, only the last ``hi`` None (no upper bound).  A launch of
``rows`` rows on a device of ``sm_count`` SMs runs the large-R program when
``lo * sm_count <= rows <= hi * sm_count`` for one of its windows (both bounds
included) and the small-R program otherwise (:func:`route`); an architecture
without a large-R program runs the small-R one at every row count.  The
programs of a stage differ in their schedule only (the threads of a CTA, the
runs of a lane's loads, when the loads of a run are issued and when its values
are stored): they store the same bits for every input, so the route changes no
output bit.

The host finds a program through ``PROGRAM_OF`` (role -> architecture ->
program name, derived from the records) and never spells a program name:
retuning a kernel renames its program (the record's key and ``library``) and
leaves the host code as it is.  The host's own facts of each stage's programs
are ``BUILD_OF``: the lanes that own a row and the rows of a thread (with the
launch block's threads, the rows one CTA owns) and the load width (the stride
and alignment rule of every BF16 tensor).  Both kernels take the row count,
the limit, the weight flag and every BF16 row stride (in elements) as launch
arguments, so one library per program and architecture serves every row
count routed to it and both layouts.  The host
launches nothing for ``rows == 0``.
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
# The suffix of a stage's large-R role.
LARGE = "_large"
# The roles of the records: each stage's small-R role (the stage itself) and its large-R role.
ROLES = tuple(stage + suffix for stage in STAGES for suffix in ("", LARGE))

MODULES: dict[str, dict[str, Any]] = {
    # The forward for small R: h = bf16(w * silu(min(g, L)) * clamp(u, -L, L)), 128 threads, 32 lanes a row, one row
    # a thread (4 rows a CTA), a lane's 9 loads in 3 runs, each run's loads issued while the run before it computes.
    "cake_clamped_swiglu_det_train_d38b827f821aba65855f": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
    # The backward for small R: d_gate, d_up (BF16) and d_weight (FP32, one fixed-order warp sum a row), 128 threads,
    # 32 lanes a row, one row a thread (4 rows a CTA), a lane's 9 loads in 3 runs, each run's loads issued while the
    # run before it computes.
    "cake_clamped_swiglu_det_train_0f584d80b1575fdf3bc5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "w"], ["buffer", "d_gate"], ["buffer", "d_up"], ["buffer", "d_weight"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "w"], ["buffer", "d_gate"], ["buffer", "d_up"], ["buffer", "d_weight"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
    # The backward of an unweighted call: d_gate and d_up (BF16) only, no weight and no d_weight, 128 threads, 32 lanes
    # a (row, run) task (3 runs a row, 4 tasks a CTA), a lane's 3 loads of its run.
    "cake_clamped_swiglu_det_train_f8ece5d554866a83bfdb": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
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

# Stage -> the build keywords of its programs that the host relies on, the same for every program of the stage (one
# lane layout, so the programs store the same bits).  ``lanes`` lanes own a row and a thread owns ``rows_per_thread``
# rows, so a CTA of ``threads`` threads (the launch block) owns ``threads / lanes * rows_per_thread`` consecutive rows;
# ``width_bits`` is the load width.  A program built with other values of these keywords is another program (another
# name): change this table with the ``MODULES`` keys.
BUILD_OF: dict[str, dict[str, int]] = {
    "fwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd_unweighted": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 3},
}

# The kernels load ``width_bits`` at a time: every BF16 row stride is a multiple of one load (8 BF16 values for
# 128-bit loads) and every BF16 base address is aligned to one load (16 bytes).  One rule for every tensor of the
# family (the strictest of the stages), so a forward's output is a backward's input as it is.
VECTOR_ELEMENTS = max(build["width_bits"] for build in BUILD_OF.values()) // 16
ALIGNMENT_BYTES = 2 * VECTOR_ELEMENTS

# ``rows`` is a 32-bit kernel parameter, and the kernels round it up to whole blocks of rows in 32 bits: fewer rows
# than this.
MAX_ROWS = 2**31 - 256


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


def rows_per_cta(name: str) -> int:
    """The rows one CTA of program ``name`` owns: ``threads / lanes * rows_per_thread`` (``BUILD_OF`` of its stage,
    the threads of its launch block)."""
    record = MODULES[name]
    build = BUILD_OF[record["stage"]]
    threads = int(record["launch"]["block"][0])
    per_cta = threads // build["lanes"] * build["rows_per_thread"]
    if per_cta < 1 or threads % build["lanes"] != 0:
        raise RuntimeError(
            f"{name}: a launch block of {threads} threads does not hold whole rows of {build['lanes']} "
            "lanes (the registry's launch leaves are written by the exporter)"
        )
    return per_cta


def grid(name: str, rows: int) -> tuple[int, int, int]:
    """The launch grid of program ``name`` for ``rows >= 1`` rows: one CTA per block of :func:`rows_per_cta` rows on
    the x axis (the kernels stride over the blocks from ``blockIdx.x``, so any row count fits one axis)."""
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    per_cta = rows_per_cta(name)
    return (-(-rows // per_cta), 1, 1)
