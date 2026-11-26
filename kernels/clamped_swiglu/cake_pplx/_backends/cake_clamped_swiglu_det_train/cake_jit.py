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
``bwd_unweighted``, its role without the ``_large``, ``_relay`` or ``_fine``
suffix), the architectures it is delivered for, the ``run`` entry and, in a
large-R, a relay or a fine role, ``rows_per_sm_windows``; ``library`` is its
key.  The derived
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

Each stage has four roles (``ROLES``).  The stage's own role (``fwd``,
``bwd``, ``bwd_unweighted``) is its small-R program, one on every
architecture.  Its large-R role (``fwd_large``, ``bwd_large``,
``bwd_unweighted_large``), its relay role (``fwd_relay``, ``bwd_relay``,
``bwd_unweighted_relay``) and its fine role (``fwd_fine``, ``bwd_fine``,
``bwd_unweighted_fine``) have at most one program on each architecture,
and may have none; the record's ``rows_per_sm_windows`` gives, per
architecture, the windows of rows per SM in which it runs: ``[lo, hi]``
lists of integers, ascending, ``lo`` below ``hi``, each ``hi`` below the next
``lo``, only the last ``hi`` None (no upper bound).  The large-R and the
relay roles are the stage's windowed roles (``WINDOWED``): on an
architecture, the windows of both, taken together and ordered by ``lo``,
are disjoint (each ``hi`` below the next ``lo``, only the last ``hi``
None).  A fine record has one window ``[0, hi]`` on each of its
architectures, below every window of its stage's windowed roles: a launch of
at most ``hi * sm_count`` rows on a device of ``sm_count`` SMs runs the fine
program (:func:`fine`).  A fine program spreads each row over 9 tasks, one
load of each input a task (a warp per task in ``fwd_fine`` and
``bwd_unweighted_fine``, a CTA of 9 warps per row in ``bwd_fine``), so a
row's serial chain is shorter when few rows share an SM.  A relay program
runs a row on a CTA of 3 warps, a warp per run of 768 columns (3 tasks a
row), so a row's loads take one memory trip.  A launch of more rows runs the windowed program of the window that holds
it, ``lo * sm_count <= rows <= hi * sm_count`` (both bounds included), and
the small-R program at every other row count (:func:`route`); an
architecture without a large-R, a relay or a fine program runs the small-R
one in its place.  The programs of a stage differ in their schedule only
(the threads of a CTA, the tasks of a row and the loads of a task, when the
loads of a run are issued and when its values are stored): they store the
same bits for every input, so the route changes no output bit.

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

A stage may also have a locality-domain role, the stage's name +
:data:`DOMAIN`: at most one program on an architecture, on ``sm_107a`` only,
its *twin*.  Only ``fwd`` has one (``fwd_domain``); ``bwd`` and
``bwd_unweighted`` have none.  :func:`route` and :func:`fine` never pick it:
only a call with ``locality="domain"`` runs it, where the host finds its
operands in memory whose chunks alternate between the two dies of an R200
(the backend's ``cake_domain``).  A twin reads row ``r`` of each operand at
slot ``first_slot + 9 r`` of chunks of ``2^chunk_shift`` slots (a slot is
512 bytes of a ``[R, 2304]`` tensor, 1,024 bytes of a merged ``[R, 4608]``
one; a row is 9 slots), chunk ``k`` in die ``first_die ^ (k & 1)``.  Its
CTAs claim items of at most ``item_rows`` rows of one chunk, their own die's
first, with one 32-bit counter in each die (``counters0``, ``counters1``),
and read their die from the SM table passed by value (``sm_table``).  The
record's ``grid_rule`` (kind :data:`DOMAIN_GRID_KIND`) gives the slots of a
row, ``chunk_shift``, its largest value, the most rows, ``item_rows`` and,
per architecture, the CTAs per SM; its ``counters`` give the claim workspace
(``words`` words a die, which every launch leaves zero); its
``first_rows_per_sm`` gives, per architecture and route class
(:data:`DOMAIN_CLASSES`), the rows per SM from which the twin runs, or None
where it never does.  :func:`domain_items` counts each die's items as the
kernel does, :func:`domain_item_rows` decodes one claim to its rows and
:func:`domain_grid` is the launch grid.  A twin runs each row through its
base program's instructions, so it stores the bits of the stage's other
programs for every input, grid and placement.
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
# The suffixes of a stage's large-R role, of its fine role and of its relay role.
LARGE = "_large"
FINE = "_fine"
RELAY = "_relay"
# The suffix of a stage's locality-domain role: its twin, run only by a ``locality="domain"`` call (never routed by
# row count).
DOMAIN = "_domain"
# The suffixes of a stage's windowed roles: their programs run in their windows of rows per SM above the fine window.
WINDOWED = (LARGE, RELAY)
# The roles of the records: each stage's small-R role (the stage itself), its large-R role, its fine role, its relay
# role and its locality-domain role.
ROLES = tuple(stage + suffix for stage in STAGES for suffix in ("", LARGE, FINE, RELAY, DOMAIN))

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
        "rows_per_sm_windows": {"sm_100a": [[13, 21]], "sm_103a": [[12, 21]], "sm_107a": [[16, 21]]},
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
    # The weighted backward for many rows per SM: a CTA of 96 threads per row, warp r owns run r of each lane's loads
    # (3 runs of 3 loads, 3 tasks a row), so a row's loads take one memory trip; registers sized for 9 CTAs an SM.
    # Warps 1 and 2 hand their dh and activation words to warp 0 in a static 9 KB shared pool, and warp 0 runs the
    # d_weight chain of the other programs in load order.
    "cake_clamped_swiglu_det_train_b161c9eed1166d9766b9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "w"], ["buffer", "d_gate"], ["buffer", "d_up"], ["buffer", "d_weight"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["parameter", "up_col"], ["parameter", "d_up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_b161c9eed1166d9766b9",
        "launch": {
            "block": (96, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_b161c9eed1166d9766b9",
        "rows_per_sm_windows": {
            "sm_100a": [[3, 5], [25, None]],
            "sm_103a": [[25, 647]],
            "sm_107a": [[3, 6], [31, None]],
        },
        "role": "bwd_relay",
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
    # The forward's locality-domain twin on SM107: the large-R forward's build (128 threads, stores as computed) at 5
    # CTAs an SM, each CTA claiming items of 4 rows from its die's half of the rows.
    "cake_clamped_swiglu_det_train_fbdbdca9a77e038ca415": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "w"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "limit"], ["parameter", "weighted"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "first_slot"], ["parameter", "first_die"], ["parameter", "chunk_shift"], ["parameter", "item_rows"], ["param_array", "sm_table"], ["parameter", "up_col"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "first_rows_per_sm": {"sm_107a": {"anchor": 128, "homed": 112}},
        "grid_rule": {
            "chunk_shift": 12,
            "ctas_per_sm": {"sm_107a": 5},
            "item_rows": 4,
            "kind": "domain_rows",
            "max_chunk_shift": 27,
            "max_rows": 134217728,
            "row_slots": 9,
        },
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_fbdbdca9a77e038ca415",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_fbdbdca9a77e038ca415",
        "role": "fwd_domain",
        "stage": "fwd",
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
# absent: a task is a row; ``bwd_unweighted`` and ``bwd_relay``: 3, a task is one run of 768 columns of a row; a fine
# role: 9, a task is one load of 256 columns of a row), so the grid is ``ceil(rows * tasks_per_row / (threads / lanes *
# rows_per_thread))`` CTAs (:func:`grid`).  A ``bwd_relay`` CTA of 96 threads owns 3 tasks, one row: its grid is
# ``rows`` CTAs.  ``width_bits`` is the load width.  A program built with other values of these facts is another program
# (another name): change this table with the ``MODULES`` keys.  A role with no program in ``MODULES`` may have an
# entry: the facts its programs keep.
BUILD_OF: dict[str, dict[str, int]] = {
    "fwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "fwd_large": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "fwd_fine": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 9},
    "bwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd_large": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd_fine": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 9},
    "bwd_relay": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 3},
    "bwd_unweighted": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 3},
    "bwd_unweighted_fine": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128, "tasks_per_row": 9},
    # The forward's twin keeps the forward's lane layout and loads (a row a warp).  Its grid is not ``cta_tasks``'s:
    # it is :func:`domain_grid`'s.
    "fwd_domain": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
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
# large-R program, ``bwd_relay`` and ``bwd_fine`` (``rows`` CTAs) stay below it.  ``bwd_relay`` counts its row loop
# (``row += gridDim.x``) in 64 bits, so any grid takes every row count below ``MAX_ROWS``.  A grid of
# ``ceil(9 rows / 4)`` CTAs (the ``fwd_fine`` and ``bwd_unweighted_fine`` programs) passes it from about 954 M rows.
# Those two kernels count their tasks and blocks in 64 bits and stride over the blocks by the grid, so :func:`grid`
# caps a launch of more rows here.
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
# holds one entry per row count up to the fine window's last row count or the last edge of the windowed programs (the
# large-R and the relay ones), whichever is larger, so the bound lets the fine window add at most ``64 * sm_count``
# entries (13,568 on 212 SMs).  On the delivered SM
# counts (148 to 212) it also keeps a fine grid far below the 32-bit limits above (``bwd_fine``: at most
# ``64 * sm_count`` CTAs and rows).
MAX_FINE_ROWS_PER_SM = 64


def _windows(name: str, arch: str) -> list[tuple[int, int | None]]:
    """``(lo, hi)`` of each window of the windowed record ``name`` on ``arch``, in its order.  Its
    ``rows_per_sm_windows[arch]`` must be ``[lo, hi]`` lists of integers, ``lo >= 0``, only the last ``hi`` None, that
    ascend without touching (``lo < hi < next lo``): any other row is a ``RuntimeError`` naming the record."""
    windows = MODULES[name]["rows_per_sm_windows"][arch]
    bounds = [bound for window in windows for bound in window]
    if (
        not windows
        or any(len(window) != 2 for window in windows)
        or any(bound is None for bound in bounds[:-1])
        or any(type(bound) is not int for bound in bounds if bound is not None)
        or bounds[0] < 0
    ):
        raise RuntimeError(f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} are not [lo, hi] lists of integers")
    bounds = [bound for bound in bounds if bound is not None]
    if any(below >= above for below, above in zip(bounds, bounds[1:])):
        raise RuntimeError(f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} do not ascend without touching")
    return [(window[0], window[1]) for window in windows]


def route(stage: str, arch: str, sm_count: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """``(edges, names)`` of ``stage`` on ``arch`` with ``sm_count`` SMs: a launch of ``rows`` rows runs the program
    ``names[bisect.bisect_left(edges, rows)]``, the index being the count of edges below ``rows``.  That is the
    stage's windowed program (its large-R or its relay one, ``WINDOWED``) when ``lo * sm_count <= rows <= hi *
    sm_count`` for a window ``[lo, hi]`` of the program's ``rows_per_sm_windows[arch]`` (``hi`` None: no upper bound),
    and the program of the stage's role, the small-R one, otherwise.  ``edges`` are the last row count of each run of
    one program, ascending: ``lo * sm_count - 1`` and ``hi * sm_count`` of each window of the windowed programs, in the
    order of their ``lo`` (an open window's ``lo * sm_count - 1`` alone).  ``names`` holds one program more: the
    small-R program, then per window its windowed program and, after a closed window, the small-R program again.  Each
    windowed program's windows are checked (:func:`_windows`), and the windows of both, in the order of their ``lo``,
    must be disjoint: each ``hi`` below the next ``lo`` and only the last ``hi`` None (else a ``RuntimeError`` naming
    both records).  Without a windowed program on ``arch``, ``edges`` is empty and ``names`` is the small-R program
    alone."""
    small = PROGRAM_OF[stage][arch]
    spans = sorted(
        (
            (lo, hi, name)
            for name in (PROGRAM_OF[stage + suffix].get(arch) for suffix in WINDOWED)
            if name is not None
            for lo, hi in _windows(name, arch)
        ),
        key=lambda span: span[0],
    )
    for (lo, hi, name), (after, top, other) in zip(spans, spans[1:]):
        if hi is None or hi >= after:
            raise RuntimeError(
                f"{name}: rows_per_sm_windows[{arch!r}] window {[lo, hi]!r} and {other}: window {[after, top]!r} of "
                f"{stage!r} overlap or touch"
            )
    edges: list[int] = []
    names = [small]
    for lo, hi, name in spans:
        edges.append(lo * int(sm_count) - 1)
        names.append(name)
        if hi is not None:
            edges.append(hi * int(sm_count))
            names.append(small)
    return tuple(edges), tuple(names)


def fine(stage: str, arch: str, sm_count: int) -> tuple[int, str]:
    """``(last, name)`` of ``stage`` on ``arch`` with ``sm_count`` SMs: a launch of ``1 <= rows <= last`` rows runs
    the program ``name``, the stage's fine role, ahead of :func:`route`.  Its record's ``rows_per_sm_windows[arch]``
    is one window ``[0, hi]`` with an integer ``hi >= 1``, so ``last = hi * sm_count``.  The window must end below
    the first window of the stage's windowed programs (``hi`` below every ``lo`` of its large-R and its relay program:
    ``last`` at most the first edge of :func:`route`), so the fine program never takes a row count that :func:`route`
    gives a windowed one.  ``hi`` must be at most :data:`MAX_FINE_ROWS_PER_SM`, so the fine window adds at most
    ``MAX_FINE_ROWS_PER_SM * sm_count`` entries to the package's route table.  Its grid at ``last`` rows must not pass
    :data:`MAX_GRID`, so a routed launch's grid is :func:`grid`'s, never capped.  ``bwd_fine`` counts its rows in 32
    bits and takes at most ``2**30`` rows (see the :data:`MAX_GRID` comment); the route sends it at most ``last`` rows
    (``2 * sm_count`` with its shipped window ``[0, 2]`` on every architecture).  Without a fine program on ``arch``,
    ``last`` is 0 (no launch runs it) and ``name`` is the stage's small-R program."""
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
            f"{name}: rows_per_sm_windows[{arch!r}] {windows!r} reaches the first large-R or relay window of "
            f"{stage!r}, from {edges[0] + 1} rows on {sm_count} SMs"
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
    if MODULES[name]["role"].endswith(DOMAIN):
        raise ValueError(f"{name} is a locality-domain twin: its grid is domain_grid's")
    per_cta, per_row = cta_tasks(name)
    return (min(-(-rows * per_row // per_cta), MAX_GRID), 1, 1)


# ----------------------------------------------------------------------------------------------------------------
# The locality-domain twin (sm_107a; only ``fwd`` has one): the host's facts of a twin, the item decode and the grid.

# The grid rule of a twin (``record["grid_rule"]["kind"]``).
DOMAIN_GRID_KIND = "domain_rows"
# The claim workspace of a twin (``record["counters"]``): its word type and who re-arms it to zero (``"kernel"``: every
# launch leaves it zero, so the host zeroes it once, when it makes it).
COUNTER_DTYPE = "uint32"
COUNTER_REARM = "kernel"
# The slots of a row: 2304 BF16 elements are 9 slots of 512 bytes, and a merged row of 4608 9 slots of 1,024 bytes.
DOMAIN_ROW_SLOTS = 9
# The bounds a twin's ``kernel_requires`` states: at most 2^27 rows, ``chunk_shift`` at most 27 and ``item_rows`` at
# most 1024, so no slot, row or item index leaves 32 bits.  A call of more rows runs its stage's other programs.
DOMAIN_MAX_ROWS = 2**27
DOMAIN_MAX_ITEM_ROWS = 1024
# The route classes of a twin's call: ``homed``, every BF16 operand (inputs, ``dh`` and outputs) in the anchor's row
# frame; ``anchor``, the anchor (the merged input or the gate) in its frame and some other BF16 operand not.  The class
# picks the twin's first rows per SM only: the twin stores the same bits in either.
DOMAIN_CLASSES = ("homed", "anchor")
# The reasons of a ``locality="domain"`` call, in the order the host checks them: ``"placed"`` for the twin, any other
# for the program the call runs with ``locality="off"`` (the same bits).
DOMAIN_REASONS = (
    "placed",
    "arch",
    "size",
    "sm_table",
    "input_memory",
    "input_layout",
    "overlap",
    "input_phase",
    "out_memory",
    "no_out",
)


def domain_program(stage: str, arch: str) -> str | None:
    """The twin of ``stage`` on ``arch`` (its locality-domain role's program), or None.  A twin whose ``grid_rule``,
    ``counters`` or ``first_rows_per_sm`` is outside the vocabulary is a registry error (``RuntimeError`` naming
    it)."""
    name = PROGRAM_OF[stage + DOMAIN].get(arch)
    if name is not None:
        domain_rule(name, arch)
    return name


def domain_rule(name: str, arch: str) -> tuple[int, int, int]:
    """``(chunk_shift, item_rows, ctas_per_sm)`` of twin ``name`` on ``arch``, its record checked: grid rule kind
    :data:`DOMAIN_GRID_KIND` with :data:`DOMAIN_ROW_SLOTS` slots a row, ``0 <= chunk_shift <= max_chunk_shift <= 27``,
    ``max_rows`` :data:`DOMAIN_MAX_ROWS`, ``1 <= item_rows <= 1024`` and, per architecture of the record, at least one
    CTA an SM (``ctas_per_sm[arch]``, the build's resident CTAs); a claim workspace of at least one ``uint32`` word a
    die that the kernel re-arms; and, per architecture of the record, a first row count per SM (an integer of at least
    1, or None) for each route class.  Anything else, or an ``arch`` the record does not ship, is a ``RuntimeError``
    naming the record."""
    record = MODULES[name]
    rule = record.get("grid_rule") or {}
    counters = record.get("counters") or {}
    firsts = record.get("first_rows_per_sm") or {}
    try:
        shift, top, most, rows = (int(rule[key]) for key in ("chunk_shift", "max_chunk_shift", "max_rows", "item_rows"))
        per_arch = rule["ctas_per_sm"]
        ctas = per_arch[arch]
        ok = (
            rule["kind"] == DOMAIN_GRID_KIND
            and int(rule["row_slots"]) == DOMAIN_ROW_SLOTS
            and 0 <= shift <= top <= 27
            and most == DOMAIN_MAX_ROWS
            and 1 <= rows <= DOMAIN_MAX_ITEM_ROWS
            and set(per_arch) == set(record["arches"])
            and all(type(value) is int and value >= 1 for value in per_arch.values())
            and counters["dtype"] == COUNTER_DTYPE
            and counters["rearm"] == COUNTER_REARM
            and int(counters["words"]) >= 1
            and set(firsts) == set(record["arches"])
            and all(
                set(row) == set(DOMAIN_CLASSES)
                and all(value is None or (type(value) is int and value >= 1) for value in row.values())
                for row in firsts.values()
            )
        )
    except (KeyError, TypeError, ValueError, AttributeError):
        ok = False
    if not ok:
        raise RuntimeError(
            f"{name}: the locality-domain record's grid_rule {rule!r}, counters {counters!r} or first_rows_per_sm "
            f"{firsts!r} is outside the vocabulary (on {arch})"
        )
    return shift, rows, ctas


def _first_rows() -> dict[str, dict[str, dict[str, int | None]]]:
    """Stage -> architecture -> route class -> the first rows per SM of the stage's twin (None: never), from the
    twin's record (``first_rows_per_sm``)."""
    table: dict[str, dict[str, dict[str, int | None]]] = {}
    for record in MODULES.values():
        if record["role"].endswith(DOMAIN):
            for arch, row in record.get("first_rows_per_sm", {}).items():
                table.setdefault(record["stage"], {})[arch] = {klass: row.get(klass) for klass in DOMAIN_CLASSES}
    return table


# Stage -> architecture -> route class -> the rows per SM from which a ``locality="domain"`` call of that class runs the
# stage's twin (a call of ``rows >= value * sm_count`` rows), or None where it never does; read from the twin's record
# (not a literal, so the exporter's reader skips it).  The values are the measured first row counts at which the twin
# on memory that alternates dies runs faster than the stage's other programs on the same memory.
DOMAIN_FIRST_ROWS_PER_SM: dict[str, dict[str, dict[str, int | None]]] = _first_rows()


def _chunk_row(chunk: int, first_slot: int, chunk_shift: int) -> int:
    """The first row whose first slot lies in chunk ``chunk`` or later."""
    return max(((chunk << chunk_shift) - first_slot + DOMAIN_ROW_SLOTS - 1) // DOMAIN_ROW_SLOTS, 0)


def domain_chunk_items(chunk_shift: int, item_rows: int) -> int:
    """``P``: the items of a whole chunk, ``ceil(ceil(2^chunk_shift / 9) / item_rows)``."""
    most = ((1 << chunk_shift) + DOMAIN_ROW_SLOTS - 1) // DOMAIN_ROW_SLOTS
    return (most + item_rows - 1) // item_rows


def domain_items(rows: int, first_slot: int, first_die: int, chunk_shift: int, item_rows: int) -> tuple[int, int]:
    """``(items0, items1)``: the items a twin's dies hold for ``rows >= 1`` rows whose row ``r`` starts at slot
    ``first_slot + 9 r`` of chunks of ``2^chunk_shift`` slots, chunk ``k`` in die ``first_die ^ (k & 1)``.  A die's
    chunks are ``k = die ^ first_die``, then every second one; it holds ``P`` items of each of them but its last
    (:func:`domain_chunk_items`), then the items its last chunk's rows need, ``item_rows`` rows an item.  The kernel's
    count, in Python integers."""
    last = (first_slot + DOMAIN_ROW_SLOTS * (rows - 1)) >> chunk_shift
    per_chunk = domain_chunk_items(chunk_shift, item_rows)
    counts = []
    for start in (first_die, first_die ^ 1):
        chunks = (last + 2 - start) >> 1
        if chunks == 0:
            counts.append(0)
            continue
        final = start + 2 * (chunks - 1)
        first = _chunk_row(final, first_slot, chunk_shift)
        end = min(_chunk_row(final + 1, first_slot, chunk_shift), rows)
        counts.append((chunks - 1) * per_chunk + (end - first + item_rows - 1) // item_rows)
    return counts[0], counts[1]


def domain_item_rows(
    word: int, rows: int, first_slot: int, first_die: int, chunk_shift: int, item_rows: int
) -> tuple[int, int]:
    """The rows ``[lo, hi)`` of claim word ``item * 2 + die`` (item ``item`` of die ``die``, :func:`domain_items`),
    as the kernel decodes it: item ``j`` of the die's ``pair``-th chunk (``item = pair * P + j``).  ``lo >= hi`` for an
    item past its chunk's rows."""
    per_chunk = domain_chunk_items(chunk_shift, item_rows)
    item = word >> 1
    pair = item // per_chunk
    chunk = ((word & 1) ^ first_die) + 2 * pair
    lo = _chunk_row(chunk, first_slot, chunk_shift) + (item - pair * per_chunk) * item_rows
    bound = min(_chunk_row(chunk + 1, first_slot, chunk_shift), rows)
    return lo, min(lo + item_rows, bound)


def domain_grid(
    name: str, arch: str, rows: int, first_slot: int, first_die: int, sm_count: int
) -> tuple[int, int, int]:
    """The launch grid of twin ``name`` on ``arch`` for ``1 <= rows <= DOMAIN_MAX_ROWS`` rows from slot ``first_slot``
    of a die ``first_die`` chunk on a device of ``sm_count`` SMs: ``min(items0 + items1, ctas_per_sm * sm_count)``
    CTAs, at least one (:func:`domain_items` at the record's ``chunk_shift`` and ``item_rows``; ``ctas_per_sm`` the
    record's for ``arch``).  Any grid of at least one CTA gives the same bits; this one keeps the walk's limit, a die's
    items plus the grid less one, below ``2**31``."""
    if not 1 <= rows <= DOMAIN_MAX_ROWS:
        raise ValueError(f"a locality-domain launch takes 1 to {DOMAIN_MAX_ROWS} rows, got {rows}")
    chunk_shift, item_rows, ctas = domain_rule(name, arch)
    if not 0 <= first_slot < 1 << chunk_shift or first_die not in (0, 1):
        raise ValueError(
            f"first_slot {first_slot} or first_die {first_die} is outside a chunk of 2^{chunk_shift} slots"
        )
    items0, items1 = domain_items(rows, first_slot, first_die, chunk_shift, item_rows)
    return (max(1, min(items0 + items1, ctas * int(sm_count))), 1, 1)
