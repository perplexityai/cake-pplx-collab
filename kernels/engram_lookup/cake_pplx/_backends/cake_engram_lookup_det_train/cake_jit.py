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

Registry of the Cake Engram row lookup: FP8 E4M3 rows with one E8M0 scale per
32 columns, gathered by row id and written as BF16 (SM100 / SM103 / SM107).
The loader and the launch plan are :mod:`.cake_backend`.

``MODULES`` has one record, ``lookup``: one mechanically exported program (a
CUDA device translation unit plus a tvm-ffi binding), compiled once per
architecture.  The record carries the host plan: the ``stage`` it serves, the
architectures it is delivered for and the ``run`` entry.  The derived leaves
(``library``, ``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding; ``library`` is the program identity
(``cake_engram_lookup_det_train_<20 hex>``), so a kernel swap renames it.
``MODULES`` is a literal table (the exporter and the manifest checker read it
with ``ast.literal_eval``).

The one program serves both table layouts (two contiguous tensors, or two
views into one packed ``[R, 264]`` byte buffer) and every ``row_ids`` shape:
the binding reads the row strides of ``weight`` and ``scale`` from the tensors,
and the number of looked-up rows ``P`` is a run-time value.  A CTA has
:data:`WARPS_PER_CTA` warps; a warp takes :data:`ROWS_PER_WARP` consecutive
looked-up rows per step and walks the rows grid-stride, so any grid covers
every row.  The launch is a plain stream-ordered launch: nothing cooperative,
no PDL, no shared memory, no workspace.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
PACKAGE = "cake_pplx._backends.cake_engram_lookup_det_train"
LIBRARY_PREFIX = "cake_engram_lookup_det_train"
MANIFEST = "build.json"

# The stages of one lookup call, in launch order.
STAGES = ("lookup",)

# One table row: ROW_DIM FP8 E4M3 values and one E8M0 scale per GROUP columns (GROUPS scales); a packed row holds the
# ROW_DIM value bytes, then the GROUPS scale bytes.
ROW_DIM = 256
GROUP = 32
GROUPS = ROW_DIM // GROUP
PACKED_ROW_BYTES = ROW_DIM + GROUPS

# Launch geometry: warps per CTA (the program's block), looked-up rows per warp and step (the program's unroll), and
# the resident CTAs per SM the grid is capped at.
WARPS_PER_CTA = 8
ROWS_PER_WARP = 4
CTAS_PER_SM = 8

MODULES: dict[str, dict[str, Any]] = {
    "lookup": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "PLACEHOLDER",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "PLACEHOLDER",
        "stage": "lookup",
    },
}


def grid(positions: int, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of a lookup of ``positions`` rows on a device of ``sm_count`` SMs: one CTA per
    ``WARPS_PER_CTA * ROWS_PER_WARP`` rows, capped at ``CTAS_PER_SM`` CTAs per SM (the kernel walks the rows
    grid-stride, so the cap leaves no row out), at least one CTA."""
    per_cta = WARPS_PER_CTA * ROWS_PER_WARP
    return (max(1, min(-(-int(positions) // per_cta), int(sm_count) * CTAS_PER_SM)), 1, 1)
