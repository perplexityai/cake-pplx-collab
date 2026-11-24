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

Registry of the Cake SiTU-GLU training forward and backward kernels
(SM100 / SM103), deterministic.

Every record of ``MODULES`` is one exported program: a CUDA device translation
unit plus a tvm-ffi binding, compiled once per architecture.  A record is keyed
by its program name ``cake_situ_glu_det_train_<20 hex>`` and carries the host
plan: the ``stage`` it serves (``fwd`` or ``bwd``), the architectures it is
delivered for and the ``run`` entry; ``library`` is its key.  The derived
leaves (``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding.  ``MODULES`` is a literal table (the
exporter and the manifest checker read it with ``ast.literal_eval``).  Every
launch is a plain stream-ordered launch: nothing cooperative, no PDL, no
dynamic shared memory.

The stages, per row ``r < rows`` and column ``c < width``, from the BF16 gate
``g`` and up ``u`` and the FP32 scalars ``beta`` and ``lb``:

- ``fwd``: ``a = beta tanh(g / beta) sigmoid(g)``, ``u' = lb tanh(u / lb)``
  and ``h = bf16(a u')``, the one rounding.
- ``bwd``: from ``g``, ``u`` and the BF16 ``dh``, with ``s = sigmoid(g)``,
  ``tg = tanh(g / beta)`` and ``tu = tanh(u / lb)``:
  ``d_gate = bf16(dh u' ((1 - tg^2) s + a (1 - s)))`` and
  ``d_up = bf16(dh a (1 - tu^2))``; the activation is recomputed.

The host finds a stage's record through ``PROGRAM_OF`` and never spells a
program name: retuning a kernel renames its program (the record's key and
``library``) and leaves the host code as it is.  Both kernels take the row
count, the width, ``beta``, ``lb`` and every BF16 row stride (in elements) as
launch arguments, so one build serves every row count, every width that is a
multiple of :data:`SEGMENT` and every layout.  A CTA owns a block of
:data:`ROWS_PER_CTA` rows at a time (one in the forward, two in the backward)
and the CTAs stride over the blocks, so the grid is one CTA per block
(:func:`grid`).  The host launches nothing for ``rows == 0``.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
PACKAGE = "cake_pplx._backends.cake_situ_glu_det_train"
LIBRARY_PREFIX = "cake_situ_glu_det_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order.
STAGES = ("fwd", "bwd")

MODULES: dict[str, dict[str, Any]] = {
    # The forward: h = bf16(beta tanh(g / beta) sigmoid(g) lb tanh(u / lb)), one row a CTA, four warps a row, three
    # segments of 256 columns a warp per pass.
    "cake_situ_glu_det_train_0d8b30c1317b470593ea": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "width"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "beta"], ["parameter", "lb"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_0d8b30c1317b470593ea",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_0d8b30c1317b470593ea",
        "stage": "fwd",
    },
    # The backward: d_gate and d_up (BF16) from gate, up and dh, the activation recomputed, two rows a CTA, two warps
    # a row, six segments of 256 columns a warp per pass.
    "cake_situ_glu_det_train_9830620c58096d1142ec": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "d_gate"], ["buffer", "d_up"], ["parameter", "rows"], ["parameter", "width"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "beta"], ["parameter", "lb"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_9830620c58096d1142ec",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_9830620c58096d1142ec",
        "stage": "bwd",
    },
}

# Stage -> the program name of its record (host code reads this, never a spelled name; not a literal, so the
# exporter's reader skips it).
PROGRAM_OF: dict[str, str] = {record["stage"]: name for name, record in MODULES.items()}

# The width (I, the columns of gate, up, h and the gradients) is a run-time value: a positive multiple of one segment
# of 256 columns (a warp's 32 lanes of 8 BF16 values).  ``width`` is a 32-bit kernel parameter.
SEGMENT = 256
MAX_WIDTH = 2**31 - SEGMENT

# The kernels load 8 BF16 values (16 bytes) at a time: every BF16 row stride is a multiple of 8 elements and every
# BF16 base address is aligned to 16 bytes.  One rule for every tensor of the family, so a forward's output is a
# backward's input as it is.
VECTOR_ELEMENTS = 8
ALIGNMENT_BYTES = 16

# The rows a CTA of 128 threads owns at a time: the forward's four warps own one row, the backward's two warps a row
# own two rows.
ROWS_PER_CTA = {"fwd": 1, "bwd": 2}

# ``rows`` is a 32-bit kernel parameter, and each CTA steps over the rows by the grid size times its rows in 32 bits:
# with one CTA per block, up to this many rows no step passes 2**31 - 1.
MAX_ROWS = 2**30


def grid(stage: str, rows: int) -> tuple[int, int, int]:
    """The launch grid of ``stage`` for ``1 <= rows <= MAX_ROWS`` rows: one CTA per block of ``ROWS_PER_CTA[stage]``
    rows on the x axis (the kernels stride over the blocks from ``blockIdx.x``, so a smaller grid would also be
    correct)."""
    if stage not in STAGES:
        raise ValueError(f"stage is one of {STAGES}, got {stage!r}")
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"a launch needs 1 to {MAX_ROWS} rows, got {rows}")
    return (-(-rows // ROWS_PER_CTA[stage]), 1, 1)
