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

``MODULES`` has eight records.  Four, keyed ``<stage>_h<width>``, are the
programs of one stage and width each (no band of :data:`ROUTES` takes
``fwd_h4096``; the tests launch it as the reference); the other four are
further programs that the host takes in a band of row counts (:data:`ROUTES`):
``fwd_h4096_t32``, a forward at H = 4096 for few rows, ``fwd_h4096_flat`` and
``fwd_h3584_flat``, the forward of each width for many rows, and
``bwd_h3584_warp``, a backward at H = 3584.
Each record is one
exported program (a CUDA device translation unit plus a tvm-ffi binding),
compiled once per architecture.  A record carries the host plan: the
``stage`` it serves (``fwd`` or ``bwd``), its ``width`` (the hidden size H),
the rows one CTA owns (``rows_per_cta``) or, for a flat program, the elements
one CTA owns (``elements_per_cta``), the architectures it is delivered for and
the ``run`` entry.  The derived leaves (``library``,
``kernel_symbol``, ``launch``, ``arg_plan``) are written by the exporter from
the rendered binding; ``library`` is the program identity
(``cake_route_weight_scale_det_train_<20 hex>``), so a kernel change renames
it.  ``MODULES`` is a literal table (the exporter and the manifest checker
read it with ``ast.literal_eval``), and so is ``ROUTES``.

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
of a warp is contiguous.  Five kinds of program:

- ``fwd_h4096`` and ``fwd_h3584``: a CTA of 128 threads owns 4 consecutive
  rows, one warp a row (``rows_per_cta`` 4).
- ``fwd_h4096_t32``: a CTA of 32 threads owns one row (``rows_per_cta`` 1)
  and runs the per-element instructions of ``fwd_h4096``, so it stores the
  same bits.  The host takes it at H = 4096 in its band of :data:`ROUTES`
  (:func:`route`).
- ``fwd_h4096_flat`` and ``fwd_h3584_flat``: the flat program, torch.compile's
  elementwise schedule.  The tensors are one row-major index of ``rows * H``
  elements; a CTA of 128 threads owns a chunk of 2048 consecutive elements
  (``elements_per_cta`` 2048), each thread two loads of 8 at ``(i * 128 + t) *
  8`` past the chunk's start.  A vector never crosses a row.  Each vector finds
  its row and column by a division by H, in 32-bit arithmetic where every
  element index and offset of the call fits, else in 64-bit; both loops make
  the same loads and stores.  It runs the per-element instructions of
  ``fwd_h<width>``, so it stores the same bits.  The host takes it in its band
  of :data:`ROUTES`, above the row programs' bands.
- ``bwd_h4096`` and ``bwd_h3584``: a CTA of 4 warps (128 threads) owns one
  row (``rows_per_cta`` 1).  Each warp sums one run of each lane's loads
  (H = 4096: 4, 4, 4, 4 loads; H = 3584: 4, 4, 4, 2) as a subtree of the
  ``dw`` tree above and stores the subtree's root, two FP32 words a lane, to
  static shared memory (1 KiB a CTA).  After a CTA barrier, warp 0 joins the
  four roots pairwise in the same order, adds the even and odd halves and
  makes the five XOR shuffles across its 32 lanes.  So ``dw`` is summed in
  the order above, bit for bit.
- ``bwd_h3584_warp``: a CTA of 128 threads owns 4 consecutive rows, one warp
  a row (``rows_per_cta`` 4).  Each lane sums its 14 loads in the order of
  the ``dw`` tree above and the warp makes the five XOR shuffles, so it
  stores the same ``dx`` and ``dw`` bits as ``bwd_h3584``.  The host takes it
  at H = 3584 in its band of :data:`ROUTES`.

Every program of one stage and width stores the same bits, whatever the row
count and the grid.  Every program strides over its blocks (``rows_per_cta``
rows, or ``elements_per_cta`` elements) with a 32-bit block counter, so a grid
of ``g`` CTAs covers ``nblocks = ceil(rows / rows_per_cta)`` (or ``ceil(rows *
H / elements_per_cta)``) blocks only while ``nblocks + g <= 2**31``.  The host launches ``min(nblocks, 2**31 -
nblocks)`` CTAs (:func:`grid`), which is ``nblocks`` up to ``2**30`` blocks;
in a band of :data:`ROUTES` with a cap it launches at most ``ctas_per_sm``
CTAs per SM of the device (a persistent grid, each CTA striding over several
blocks).  Every kernel takes the row count and every BF16 row stride (in
elements) as launch arguments, so one library per program and architecture
serves every row count and every row slice.  Row offsets are 64-bit.  The
launch is a plain stream-ordered launch: nothing cooperative, no PDL, no
dynamic shared memory (``bwd_h4096`` and ``bwd_h3584`` use 1 KiB of static
shared memory a CTA), no workspace.  The host launches nothing for
``rows == 0``.
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
# A lane's load: 8 BF16 elements (16 bytes).  Every BF16 row stride is a multiple of it and every BF16 base is aligned
# to it.
VECTOR_ELEMENTS = 8
ALIGNMENT_BYTES = 16
# The host refuses ``rows >= MAX_ROWS``: below it the flat programs' blocks stay below ``2**30`` (H = 4096: ``2 * rows``
# blocks), so their 32-bit block counter covers every block on the full grid.  One tensor of ``MAX_ROWS`` rows of
# H = 3584 already needs 3.5 TiB.
MAX_ROWS = 2**29
# The program and grid of each call, keyed by the call's ``<stage>_h<width>``: bands ``(first_row, record,
# ctas_per_sm)`` in strictly increasing ``first_row`` order, the first from 1.  A call of ``rows`` rows takes the last
# band whose ``first_row <= rows``; ``ctas_per_sm > 0`` caps its grid at that many CTAs per SM of the device, 0 leaves
# it uncapped.  Every record of one stage and width stores the same bits.  Measured on B200 and GB300: the H = 4096
# forward at one row a CTA of 32 threads is the faster for few rows, the flat forward of each width the faster for many
# rows, and the four-warp backward on a grid of 8 CTAs per SM is the faster while the call's three BF16 row tensors fit
# in L2, except in the band of the one-warp-a-row H = 3584 backward, the fastest there.
ROUTES: dict[str, tuple[tuple[int, str, int], ...]] = {
    "fwd_h4096": ((1, "fwd_h4096_t32", 0), (10240, "fwd_h4096_flat", 0)),
    "fwd_h3584": ((1, "fwd_h3584", 0), (16384, "fwd_h3584_flat", 0)),
    "bwd_h4096": ((1, "bwd_h4096", 8), (4864, "bwd_h4096", 0)),
    "bwd_h3584": (
        (1, "bwd_h3584", 8),
        (3328, "bwd_h3584_warp", 0),
        (4352, "bwd_h3584", 8),
        (5632, "bwd_h3584", 0),
    ),
}

MODULES: dict[str, dict[str, Any]] = {
    # The forward, H = 4096: a CTA of 128 threads owns 4 rows, one warp a row.
    "fwd_h4096": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_bd43b17800f7121c6434",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_bd43b17800f7121c6434",
        "rows_per_cta": 4,
        "stage": "fwd",
        "width": 4096,
    },
    # The forward, H = 4096, for few rows: a CTA of 32 threads owns one row, with the instructions of fwd_h4096 (the
    # same bits).  The host takes it in its band of ROUTES.
    "fwd_h4096_t32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_edfd720e5f7640b79cf4",
        "launch": {
            "block": (32, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_edfd720e5f7640b79cf4",
        "rows_per_cta": 1,
        "stage": "fwd",
        "width": 4096,
    },
    # The forward, H = 3584: a CTA of 128 threads owns 4 rows, one warp a row.
    "fwd_h3584": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_40ea8f7288b169790f4c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_40ea8f7288b169790f4c",
        "rows_per_cta": 4,
        "stage": "fwd",
        "width": 3584,
    },
    # The forward, H = 4096, for many rows: the flat program, a CTA of 128 threads owns a chunk of 2048 consecutive
    # elements of the row-major index (torch.compile's elementwise schedule), with the per-element instructions of
    # fwd_h4096 (the same bits).  The host takes it in its band of ROUTES.
    "fwd_h4096_flat": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "elements_per_cta": 2048,
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_c8a39889d6f827831ba5",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_c8a39889d6f827831ba5",
        "stage": "fwd",
        "width": 4096,
    },
    # The forward, H = 3584, for many rows: the flat program, a CTA of 128 threads owns a chunk of 2048 consecutive
    # elements of the row-major index (torch.compile's elementwise schedule), with the per-element instructions of
    # fwd_h3584 (the same bits).  The host takes it in its band of ROUTES.
    "fwd_h3584_flat": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "w"], ["buffer", "y"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "y_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "elements_per_cta": 2048,
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_76225253b414f853a74d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_76225253b414f853a74d",
        "stage": "fwd",
        "width": 3584,
    },
    # The backward, H = 4096: dx and dw in one pass over x and dy; a CTA of 4 warps owns one row, each warp sums 4
    # loads of each lane and warp 0 joins the four subtrees through shared memory.
    "bwd_h4096": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "dy"], ["buffer", "w"], ["buffer", "dx"], ["buffer", "dw"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "dy_stride"], ["parameter", "dx_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_fdcbf1098d5f29c13f73",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_fdcbf1098d5f29c13f73",
        "rows_per_cta": 1,
        "stage": "bwd",
        "width": 4096,
    },
    # The backward, H = 3584: dx and dw in one pass over x and dy; a CTA of 4 warps owns one row, the warps sum 4, 4,
    # 4 and 2 loads of each lane and warp 0 joins the four subtrees through shared memory.
    "bwd_h3584": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "dy"], ["buffer", "w"], ["buffer", "dx"], ["buffer", "dw"], ["parameter", "rows"], ["parameter", "x_stride"], ["parameter", "dy_stride"], ["parameter", "dx_stride"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_route_weight_scale_det_train_af754e6499762a11f5c9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_route_weight_scale_det_train_af754e6499762a11f5c9",
        "rows_per_cta": 1,
        "stage": "bwd",
        "width": 3584,
    },
    # The backward, H = 3584, one warp a row: a CTA of 128 threads owns 4 rows, each lane sums its 14 loads in the
    # order of the dw tree (the same bits as bwd_h3584).  The host takes it in its band of ROUTES.
    "bwd_h3584_warp": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
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
        "rows_per_cta": 4,
        "stage": "bwd",
        "width": 3584,
    },
}


def route(stage: str, width: int, rows: int) -> tuple[str, int]:
    """``(record, ctas_per_sm)`` of a call of ``rows >= 1`` rows of ``stage`` at hidden size ``width``: the record (a
    key of :data:`MODULES`) and the grid cap of the last band of :data:`ROUTES` under ``<stage>_h<width>`` whose first
    row is at most ``rows``; a cap of 0 leaves the grid uncapped."""
    if stage not in STAGES:
        raise KeyError(f"{stage!r} is not a stage of {STAGES}")
    if width not in WIDTHS:
        raise KeyError(f"H = {width} is not one of the built widths {WIDTHS}")
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    bands = ROUTES[f"{stage}_h{width}"]
    _, name, ctas_per_sm = bands[0]  # the first band starts at row 1
    for first_row, record, cap in bands[1:]:
        if first_row > rows:
            break
        name, ctas_per_sm = record, cap
    return name, ctas_per_sm


def grid(rows: int, name: str, ctas_per_sm: int = 0, sm_count: int = 0) -> tuple[int, int, int]:
    """The launch grid of record ``name`` for ``rows >= 1`` rows.  The program owns ``nblocks = ceil(rows /
    rows_per_cta)`` blocks of its ``rows_per_cta`` rows (a flat program: ``ceil(rows * width / elements_per_cta)``
    blocks of its ``elements_per_cta`` elements) and strides over them with a 32-bit counter, so any grid of
    ``g >= 1`` CTAs covers every block while ``nblocks + g <= 2**31``.  The grid is ``min(nblocks, 2**31 - nblocks)``
    CTAs (one CTA per block up to ``2**30`` blocks) and, with ``ctas_per_sm > 0``, at most ``ctas_per_sm *
    sm_count`` CTAs: a persistent grid over the ``sm_count >= 1`` SMs of the device, the same bits."""
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    record = MODULES[name]
    if "elements_per_cta" in record:
        blocks = -(-int(rows) * record["width"] // record["elements_per_cta"])
    else:
        blocks = -(-int(rows) // record["rows_per_cta"])
    ctas = min(blocks, 2**31 - blocks)
    if ctas_per_sm > 0:
        if sm_count < 1:
            raise ValueError(f"a grid capped at {ctas_per_sm} CTAs per SM needs the SM count, got {sm_count}")
        ctas = min(ctas, ctas_per_sm * sm_count)
    return (ctas, 1, 1)
