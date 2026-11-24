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

Registry of the Cake route-weight row scaling training kernels of a routed MoE
expert (SM100 / SM103 / SM107), deterministic.

``MODULES`` has four records, one per stage and width, keyed by the record
name ``<stage>_h<width>``.  Each record is one exported program (a CUDA device
translation unit plus a tvm-ffi binding), compiled once per architecture.  A
record carries the host plan: the ``stage`` it serves (``fwd`` or ``bwd``),
its ``width`` (the hidden size H), the architectures it is delivered for and
the ``run`` entry.  The derived leaves (``library``, ``kernel_symbol``,
``launch``, ``arg_plan``) are written by the exporter from the rendered
binding; ``library`` is the program identity
(``cake_route_weight_scale_det_train_<20 hex>``), so a kernel change renames
it.  ``MODULES`` is a literal table (the exporter and the manifest checker
read it with ``ast.literal_eval``).

The stages, per row ``r < rows`` and column ``c < H``:

- ``fwd``: ``y = bf16(w[r] * x)``, the FP32 product rounded once.
- ``bwd``: ``dx = bf16(w[r] * dy)``, the FP32 product rounded once, and
  ``dw[r] = sum over c of dy * x``, one FP32 word per row.  The products of
  two BF16 values are exact in FP32; the sum adds them in a balanced binary
  tree in one fixed order (pairs within a load of 8 columns, then the loads
  of a lane pairwise, the even and odd halves, then five XOR shuffles across
  the row's 32 lanes).

Launch geometry: a CTA of 128 threads owns :data:`ROWS_PER_CTA` consecutive
rows (32 lanes a row) and the CTAs stride over the blocks of rows, so any grid
of at most ``ceil(rows / ROWS_PER_CTA)`` CTAs covers every row; the host
launches exactly that many (:func:`grid`).  Every kernel takes the row count
and every BF16 row stride (in elements) as launch arguments, so one library
per program and architecture serves every row count and every row slice.
Row offsets are 64-bit.  The launch is a plain stream-ordered launch: nothing
cooperative, no PDL, no shared memory, no workspace.  The host launches
nothing for ``rows == 0``.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
PACKAGE = "cake_pplx._backends.cake_route_weight_scale_det_train"
LIBRARY_PREFIX = "cake_route_weight_scale_det_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order.
STAGES = ("fwd", "bwd")
# The hidden sizes H the programs are built for: 4096 and 3584.
WIDTHS = (4096, 3584)
# Rows one CTA owns per block of rows (its 128 threads, 32 lanes a row).
ROWS_PER_CTA = 4
# A lane's load: 8 BF16 elements (16 bytes).  Every BF16 row stride is a multiple of it and every BF16 base is aligned
# to it.
VECTOR_ELEMENTS = 8
ALIGNMENT_BYTES = 16
# The row count is a 32-bit kernel parameter: the host refuses ``rows >= MAX_ROWS``.
MAX_ROWS = 2**31 - 256

MODULES: dict[str, dict[str, Any]] = {
    # The forward, H = 4096.
    "fwd_h4096": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_00000000000000000001",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_00000000000000000001",
        "stage": "fwd",
        "width": 4096,
    },
    # The forward, H = 3584.
    "fwd_h3584": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_00000000000000000002",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_00000000000000000002",
        "stage": "fwd",
        "width": 3584,
    },
    # The backward, H = 4096: dx and dw in one pass over x and dy.
    "bwd_h4096": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "dy"], ["buffer", "w"], ["buffer", "dx"], ["buffer", "dw"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "dy_stride"], ["parameter", "dx_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_00000000000000000003",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_00000000000000000003",
        "stage": "bwd",
        "width": 4096,
    },
    # The backward, H = 3584: dx and dw in one pass over x and dy.
    "bwd_h3584": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "dy"], ["buffer", "w"], ["buffer", "dx"], ["buffer", "dw"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "dy_stride"], ["parameter", "dx_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_00000000000000000004",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_00000000000000000004",
        "stage": "bwd",
        "width": 3584,
    },
}


def record_name(stage: str, width: int) -> str:
    """The record of ``stage`` at hidden size ``width``: ``<stage>_h<width>`` (a key of :data:`MODULES`)."""
    if stage not in STAGES:
        raise KeyError(f"{stage!r} is not a stage of {STAGES}")
    if width not in WIDTHS:
        raise KeyError(f"H = {width} is not one of the built widths {WIDTHS}")
    return f"{stage}_h{width}"


def grid(rows: int) -> tuple[int, int, int]:
    """The launch grid of ``rows >= 1`` rows: one CTA per block of :data:`ROWS_PER_CTA` rows."""
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    return (-(-int(rows) // ROWS_PER_CTA), 1, 1)
