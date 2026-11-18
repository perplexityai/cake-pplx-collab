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
unit plus a tvm-ffi binding, compiled once per architecture.  A record is keyed
by its program name ``cake_clamped_swiglu_det_train_<20 hex>`` and carries the
host plan: the ``stage`` it serves (``fwd`` or ``bwd``), the architectures it
is delivered for and the ``run`` entry; ``library`` is its key.  The derived
leaves (``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding.  ``MODULES`` is a literal table (the
exporter and the manifest checker read it with ``ast.literal_eval``).  Every
launch is a plain stream-ordered launch: nothing cooperative, no PDL, no
dynamic shared memory.

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

The host finds a stage's record through ``PROGRAM_OF`` and never spells a
program name: retuning a kernel renames its program (the record's key and
``library``) and leaves the host code as it is.  The host's own facts of each
stage's build are ``BUILD_OF``: the lanes that own a row and the rows of a
thread (with the launch block's threads, the rows one CTA owns) and the load
width (the stride and alignment rule of every BF16 tensor).  Both kernels take
the row count, the limit, the weight flag and every BF16 row stride (in
elements) as launch arguments, so one build serves every row count and both
layouts.  The host launches nothing for ``rows == 0``.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
PACKAGE = "cake_pplx._backends.cake_clamped_swiglu_det_train"
LIBRARY_PREFIX = "cake_clamped_swiglu_det_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order.
STAGES = ("fwd", "bwd")

MODULES: dict[str, dict[str, Any]] = {
    # The forward: h = bf16(w * silu(min(g, L)) * clamp(u, -L, L)), 256 threads,
    # 32 lanes a row, one row a thread (8 rows a CTA).
    "cake_clamped_swiglu_det_train_0dddfba03f753508a937": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_0dddfba03f753508a937",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_0dddfba03f753508a937",
        "stage": "fwd",
    },
    # The backward: d_gate, d_up (BF16) and d_weight (FP32, one fixed-order
    # warp sum a row), 256 threads, 32 lanes a row, one row a thread.
    "cake_clamped_swiglu_det_train_815715cef4387daa0406": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_clamped_swiglu_det_train_815715cef4387daa0406",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_clamped_swiglu_det_train_815715cef4387daa0406",
        "stage": "bwd",
    },
}

# Stage -> the program name of its record (host code reads this, never a spelled name; not a literal, so the
# exporter's reader skips it).
PROGRAM_OF: dict[str, str] = {record["stage"]: name for name, record in MODULES.items()}

# I: the intermediate width of the expert (the gate, up, h and gradient columns).
WIDTH = 2304

# Stage -> the build keywords of its program that the host relies on.  ``lanes`` lanes own a row and a thread owns
# ``rows_per_thread`` rows, so a CTA of ``threads`` threads (the launch block) owns ``threads / lanes *
# rows_per_thread`` consecutive rows; ``width_bits`` is the load width.  A program built with other values of these
# keywords is another program (another name): change this table with the ``MODULES`` keys.
BUILD_OF: dict[str, dict[str, int]] = {
    "fwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
    "bwd": {"lanes": 32, "rows_per_thread": 1, "width_bits": 128},
}

# The kernels load ``width_bits`` at a time: every BF16 row stride is a multiple of one load (8 BF16 values for
# 128-bit loads) and every BF16 base address is aligned to one load (16 bytes).  One rule for every tensor of the
# family (the strictest of the stages), so a forward's output is a backward's input as it is.
VECTOR_ELEMENTS = max(build["width_bits"] for build in BUILD_OF.values()) // 16
ALIGNMENT_BYTES = 2 * VECTOR_ELEMENTS

# ``rows`` is a 32-bit kernel parameter, and the kernels round it up to whole blocks of rows in 32 bits: fewer rows
# than this.
MAX_ROWS = 2**31 - 256


def rows_per_cta(stage: str) -> int:
    """The rows one CTA of ``stage``'s program owns: ``threads / lanes * rows_per_thread`` (``BUILD_OF``, the threads
    of the record's launch block)."""
    build = BUILD_OF[stage]
    threads = int(MODULES[PROGRAM_OF[stage]]["launch"]["block"][0])
    per_cta = threads // build["lanes"] * build["rows_per_thread"]
    if per_cta < 1 or threads % build["lanes"] != 0:
        raise RuntimeError(
            f"{PROGRAM_OF[stage]}: a launch block of {threads} threads does not hold whole rows of {build['lanes']} "
            "lanes (the registry's launch leaves are written by the export)"
        )
    return per_cta


def grid(stage: str, rows: int) -> tuple[int, int, int]:
    """The launch grid of ``stage`` for ``rows >= 1`` rows: one CTA per block of :func:`rows_per_cta` rows on the x
    axis (the kernels stride over the blocks from ``blockIdx.x``, so any row count fits one axis)."""
    if rows < 1:
        raise ValueError(f"a launch needs at least one row, got {rows}")
    per_cta = rows_per_cta(stage)
    return (-(-rows // per_cta), 1, 1)
