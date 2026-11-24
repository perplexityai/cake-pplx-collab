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
expert (SM100 / SM103), deterministic.  The sm_107a builds follow once they
run on an SM107 device.

``MODULES`` has five records.  Four, keyed ``<stage>_h<width>``, serve one
stage and width each; the fifth, ``fwd_h4096_t32``, is a second forward
program at H = 4096 that the host takes for few rows (:data:`SMALL_ROWS`).
Each record is one exported program (a CUDA device translation unit plus a
tvm-ffi binding), compiled once per architecture.  A record carries the host
plan: the ``stage`` it serves (``fwd`` or ``bwd``), its ``width`` (the hidden
size H), the rows one CTA owns (``rows_per_cta``), the architectures it is
delivered for and the ``run`` entry.  The derived leaves (``library``,
``kernel_symbol``, ``launch``, ``arg_plan``) are written by the exporter from
the rendered binding; ``library`` is the program identity
(``cake_route_weight_scale_det_train_<20 hex>``), so a kernel change renames
it.  ``MODULES`` is a literal table (the exporter and the manifest checker
read it with ``ast.literal_eval``), and so is ``SMALL_ROWS``.

The stages, per row ``r < rows`` and column ``c < H``:

- ``fwd``: ``y = bf16(w[r] * x)``, the FP32 product rounded once.
- ``bwd``: ``dx = bf16(w[r] * dy)``, the FP32 product rounded once, and
  ``dw[r] = sum over c of dy * x``, one FP32 word per row.  The products of
  two BF16 values are exact in FP32; the sum adds them in a balanced binary
  tree in one fixed order (pairs within a load of 8 columns, then the loads
  of a lane pairwise, the even and odd halves, then five XOR shuffles across
  32 lanes).

Launch geometry.  A row's columns are spread over 32 lanes: lane ``l`` makes
``H / 256`` loads of 8 elements at columns ``(p * 32 + l) * 8``, so each load
of a warp is contiguous.  Three kinds of program:

- ``fwd_h4096`` and ``fwd_h3584``: a CTA of 128 threads owns 4 consecutive
  rows, one warp a row (``rows_per_cta`` 4).
- ``fwd_h4096_t32``: a CTA of 32 threads owns one row (``rows_per_cta`` 1)
  and runs the per-element instructions of ``fwd_h4096``, so it stores the
  same bits.  The host takes it at H = 4096 below the bound of
  :data:`SMALL_ROWS` (:func:`record_name`), ``fwd_h4096`` from the bound on.
- ``bwd_h4096`` and ``bwd_h3584``: a CTA of 4 warps (128 threads) owns one
  row (``rows_per_cta`` 1).  Each warp sums one run of each lane's loads
  (H = 4096: 4, 4, 4, 4 loads; H = 3584: 4, 4, 4, 2) as a subtree of the
  ``dw`` tree above and stores the subtree's root, two FP32 words a lane, to
  static shared memory (1 KiB a CTA).  After a CTA barrier, warp 0 joins the
  four roots pairwise in the same order, adds the even and odd halves and
  makes the five XOR shuffles across its 32 lanes.  So ``dw`` is summed in
  the order above, bit for bit.

Every program of one stage and width stores the same bits, whatever the row
count.  Every program strides over the blocks of ``rows_per_cta`` rows with a
32-bit block counter, so a grid of ``g`` CTAs covers ``nblocks = ceil(rows /
rows_per_cta)`` blocks only while ``nblocks + g <= 2**31``.  The host
launches ``min(nblocks, 2**31 - nblocks)`` CTAs (:func:`grid`), which is
``nblocks`` up to ``2**30`` blocks.  Every kernel takes the row count and
every BF16 row stride (in elements) as launch arguments, so one library per
program and architecture serves every row count and every row slice.  Row
offsets are 64-bit.  The launch is a plain stream-ordered launch: nothing
cooperative, no PDL, no dynamic shared memory (the backward uses 1 KiB of
static shared memory a CTA), no workspace.  The host launches nothing for
``rows == 0``.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
PACKAGE = "cake_pplx._backends.cake_route_weight_scale_det_train"
LIBRARY_PREFIX = "cake_route_weight_scale_det_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order.
STAGES = ("fwd", "bwd")
# The hidden sizes H the programs are built for: 4096 and 3584.
WIDTHS = (4096, 3584)
# A lane's load: 8 BF16 elements (16 bytes).  Every BF16 row stride is a multiple of it and every BF16 base is aligned
# to it.
VECTOR_ELEMENTS = 8
ALIGNMENT_BYTES = 16
# The row count is a 32-bit kernel parameter: the host refuses ``rows >= MAX_ROWS``.
MAX_ROWS = 2**31 - 256
# ``record: (bound, small)``: below ``bound`` rows the host launches record ``small`` in place of ``record`` (the same
# bits).  Here the H = 4096 forward at one row a CTA of 32 threads, the faster of the two up to 8192 rows on B200 and
# GB300; from 10240 rows on, the 128-thread program is as fast or faster.
SMALL_ROWS: dict[str, tuple[int, str]] = {"fwd_h4096": (10240, "fwd_h4096_t32")}

MODULES: dict[str, dict[str, Any]] = {
    # The forward, H = 4096: a CTA of 128 threads owns 4 rows, one warp a row.
    "fwd_h4096": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_cb5cdccb7cf7465f5128",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_cb5cdccb7cf7465f5128",
        "rows_per_cta": 4,
        "stage": "fwd",
        "width": 4096,
    },
    # The forward, H = 4096, for few rows: a CTA of 32 threads owns one row, with the instructions of fwd_h4096 (the
    # same bits).  The host takes it below the bound of SMALL_ROWS.
    "fwd_h4096_t32": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_00000000000000000005",
        "launch": {
            "block": (32, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_00000000000000000005",
        "rows_per_cta": 1,
        "stage": "fwd",
        "width": 4096,
    },
    # The forward, H = 3584: a CTA of 128 threads owns 4 rows, one warp a row.
    "fwd_h3584": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_83007fd313fa63aaab26",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_83007fd313fa63aaab26",
        "rows_per_cta": 4,
        "stage": "fwd",
        "width": 3584,
    },
    # The backward, H = 4096: dx and dw in one pass over x and dy; a CTA of 4 warps owns one row, each warp sums 4
    # loads of each lane and warp 0 joins the four subtrees through shared memory.
    "bwd_h4096": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "dy"], ["buffer", "w"], ["buffer", "dx"], ["buffer", "dw"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "dy_stride"], ["parameter", "dx_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_709321d157c7df3a2ec4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_709321d157c7df3a2ec4",
        "rows_per_cta": 1,
        "stage": "bwd",
        "width": 4096,
    },
    # The backward, H = 3584: dx and dw in one pass over x and dy; a CTA of 4 warps owns one row, the warps sum 4, 4,
    # 4 and 2 loads of each lane and warp 0 joins the four subtrees through shared memory.
    "bwd_h3584": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "dy"], ["buffer", "w"], ["buffer", "dx"], ["buffer", "dw"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "dy_stride"], ["parameter", "dx_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_0f7ee9ed09647f3c9653",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_0f7ee9ed09647f3c9653",
        "rows_per_cta": 1,
        "stage": "bwd",
        "width": 3584,
    },
}


def record_name(stage: str, width: int, rows: int) -> str:
    """The record (a key of :data:`MODULES`) that serves ``rows >= 1`` rows of ``stage`` at hidden size ``width``:
    ``<stage>_h<width>``, except below the bound :data:`SMALL_ROWS` sets for that record, which takes the record
    :data:`SMALL_ROWS` names in its place."""
    if stage not in STAGES:
        raise KeyError(f"{stage!r} is not a stage of {STAGES}")
    if width not in WIDTHS:
        raise KeyError(f"H = {width} is not one of the built widths {WIDTHS}")
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    name = f"{stage}_h{width}"
    if name in SMALL_ROWS and rows < SMALL_ROWS[name][0]:
        return SMALL_ROWS[name][1]
    return name


def grid(rows: int, name: str) -> tuple[int, int, int]:
    """The launch grid of record ``name`` for ``rows >= 1`` rows: one CTA per block of its ``rows_per_cta`` rows,
    ``nblocks = ceil(rows / rows_per_cta)`` CTAs, at most ``2**31 - nblocks``.  The program strides over the blocks
    with a 32-bit counter, so a grid of ``g`` CTAs covers at most ``2**31 - g`` blocks; the bound binds only above
    ``2**30`` blocks."""
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    blocks = -(-int(rows) // MODULES[name]["rows_per_cta"])
    return (min(blocks, 2**31 - blocks), 1, 1)
