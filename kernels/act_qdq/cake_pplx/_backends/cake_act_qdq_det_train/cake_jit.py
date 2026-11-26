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

Registry and host plan of the Cake activation quantize-dequantize forward
(BF16 in, BF16 out; SM100 / SM103 / SM107).

Every record of ``MODULES`` is one generated program: a CUDA device
translation unit plus a tvm-ffi binding, compiled once per architecture.  A
record is keyed by its route: the format (``fp4_e4m3_g16``,
``mxfp4_e8m0_g32`` or ``mxfp8_e8m0_g32``), with the suffix ``_scales`` for
the build that also stores the FP32 scale of each group, and the suffix
``_small``, ``_wide``, ``_large``, ``_mid`` or ``_domain`` for the route's
small, wide, large, mid or interleaved-domain build.  The host plan of a record
is ``format``, ``scale_out``, ``group`` (the elements of one scale group),
``arches``, ``ffi_entry``, ``grid_rule``; for a ranged record (small, wide,
large, mid or interleaved-domain) only, ``vectors``: per architecture and kind
of call (``out_of_place`` or ``in_place``), the inclusive ``n_vecs`` range its
route runs it for, or ``None`` for no size; and for an interleaved-domain
record only, ``counters`` (its claim workspace, below).
The derived leaves (``library``, ``kernel_symbol``,
``launch`` and ``arg_plan``) are written from the rendered program each time
the package is generated; ``library`` is the program identity
(``cake_act_qdq_<20 hex>``, over the kernel's module and build options), so it
names the same program across revisions of the kernel, and ``build.json``
records the digests of the sources it is compiled from.  ``MODULES`` and
``ROUTES`` are literal tables (they are read with ``ast.literal_eval``).

What to compile (sources, digests, nvcc flags) is recorded once in
``build.json`` next to this file.  :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it,
and otherwise through the package's JIT shim from the same sources and flags.

A program sees the input as ``n_vecs = numel / 8`` vectors of eight BF16
values.  One CTA of 64 threads takes a tile of 128 vectors (256 on SM107) per
trip of a CTA-uniform grid-stride loop, each thread loading its two (four)
vectors before it computes any, so any grid of at least one CTA is correct.
The host launches one CTA per tile once the tiles pass 2.5 waves of the CTAs
the SMs hold (32 per SM, 16 on SM107), else that many CTAs, at most one per
tile (:func:`launch_grid`).  A small program is the same kernel with CTAs of
256 threads, two vectors each (tiles of 512 vectors, 8 CTAs per SM, 4 on
SM107).  A call runs the small record of its route when ``n_vecs`` lies in
the record's ``vectors`` range for the call's kind, where the problem fills
at most about one wave and the larger tiles need fewer CTAs: out of place
65,537 to 524,288 on SM100 and SM103 (to 131,072 for MXFP8 without the
scales) and 1 to 262,144 on SM107, in place 1 to 262,144 on SM107 only (in
place on SM100 and SM103 the small program measured no faster).  The wide
program, MXFP8 without the scales on SM100 and SM103 only, is the same kernel
with CTAs of 128 threads, four vectors each (tiles of 512 vectors), launched
one CTA per tile at every size, and with 48 registers per thread, so 10 CTAs
fit on an SM.  A call runs it above the main and small programs' sizes: out
of place from 131,073 vectors to 2,147,483,136 (``2**31 - 512``, as it forms
``n_vecs + 511``; the main program runs above that), and in place from
262,145 to 33,554,431, below the large program's range; there it measured up
to 5 % faster than the main program.  The large program, MXFP4 and MXFP8
with and without the scales on SM100 and SM103 only, is the same kernel with
CTAs of 1024 threads, two vectors each (tiles of 2,048 vectors), launched one
CTA per tile at every size, and with 32 registers per thread, so 2 CTAs fit
on an SM.  A call runs it at the largest sizes: in place from 4,194,304
vectors (from 33,554,432 for MXFP8 without the scales, above the wide
program's range), and out of place from 8,388,608 for MXFP8 with the scales
only, all to 2,147,481,600 (``2**31 - 2048``, as it forms ``n_vecs + 2047``;
the main program runs above that).  On B200 and GB300, with L2 full of dirty
lines, full of clean lines or holding ``x`` just written before the call, it
measured faster than the program it replaces in every state inside those
ranges: by 0.2 to 3.3 % from 8,388,608 vectors on, and at 4,194,304 vectors
by up to 7 % with a clean L2 and up to 20 % with ``x`` in L2.  Below them it
measured up to 16 % slower, and in place at 16,777,216 vectors 1.2 % slower
than the wide program with a clean L2.  Out of place it measured from 0.5 %
slower to 0.7 % faster for MXFP4 and no faster for MXFP8 without the scales,
and FP4 measured slower at some sizes in both kinds of call, so those have
no large range.  The mid program, MXFP4 and MXFP8 with and without the
scales on SM107 only, is the same kernel with CTAs of 64 threads, eight
vectors each (tiles of 512 vectors), 16 CTAs per SM and the main program's
grid rule; it keeps each vector in packed BF16 pairs, so it fits in 64
registers per thread.  A call runs it in both kinds of call from 868,353
vectors, past one wave of the main program's CTAs on a VR200 (212 SMs), to
1,736,704, one wave of its own; MXFP4's range starts at 933,888.  On three
VR200 GPUs with L2 flushed before each call, the kernel measured 2.4 to 14 %
faster than the main program from 868,353 to 1,736,704 vectors (the median
over the GPUs), and from 7 % slower to 2.4 % faster outside that range.
Replayed in a CUDA graph, some MXFP4 calls measured up to 2 % slower than
with the main program below 933,888 vectors.  Every other call runs the main
record (:func:`selected_module`).  Every program gives the same bytes.
``n_vecs`` is a run-time scalar, so a library serves every shape; a call
builds or loads only its own program's library on first use.  Build every
library of the architecture (:func:`specs_for_arch`) before a CUDA graph
capture, to keep the JIT out of the capture.

The interleaved-domain records (``<route>_domain``, SM107 only) are not on any
route of ``ROUTES``: ``DOMAIN_ROUTES`` names one per route, and only a call
with ``locality="domain"`` runs it, out of place, when :func:`domain_route`
finds ``x`` and ``y`` in memory whose chunks of ``grid_rule["chunk_bytes"]``
alternate between the two dies (chunk ``k`` of a buffer in die ``k & 1``), at
the same offset in their chunks and in chunks of the same die, and ``n_vecs``
in the record's ``vectors`` range.  Such a program takes its own tiles of 128
threads, 1,024 vectors (8 a thread), and each CTA claims its next item,
``2^batch_shift`` tiles of one chunk (one tile), from its own die's counter
first and then from the other die's; the record's ``counters`` field gives the
claim workspace: ``words`` words a die of ``dtype`` ``"uint32"``, which every
launch leaves zero (``rearm`` ``"kernel"``), so a workspace is zeroed once,
when it is made.  Its grid is the ``domain_items`` rule of
:func:`domain_launch`; any grid of at least one CTA gives the same bytes.
Every domain program gives the bytes of the plain program of its route.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``library``, ``kernel_symbol``,
# ``launch.block``, ``launch.cluster``, ``launch.cooperative``,
# ``launch.dynamic_smem_bytes``, ``launch.use_pdl`` and ``arg_plan``) are
# rewritten from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    # FP4: E2M1 values, one E4M3 scale per 16 elements.
    "fp4_e4m3_g16": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "fp4_e4m3_g16",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 32, "sm_103a": 32, "sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 128, "sm_103a": 128, "sm_107a": 256},
            "vector_elements": 8,
        },
        "group": 16,
        "kernel_symbol": "kernel_cake_act_qdq_6c386eb171b622b07aa6",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_6c386eb171b622b07aa6",
        "scale_out": False,
    },
    "fp4_e4m3_g16_scales": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "fp4_e4m3_g16",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 32, "sm_103a": 32, "sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 128, "sm_103a": 128, "sm_107a": 256},
            "vector_elements": 8,
        },
        "group": 16,
        "kernel_symbol": "kernel_cake_act_qdq_3c9b6d6bcbaf8c45c904",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_3c9b6d6bcbaf8c45c904",
        "scale_out": True,
    },
    # MXFP4: E2M1 values, one E8M0 (power-of-two) scale per 32 elements.
    "mxfp4_e8m0_g32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 32, "sm_103a": 32, "sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 128, "sm_103a": 128, "sm_107a": 256},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_d3568612f1c725cff309",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_d3568612f1c725cff309",
        "scale_out": False,
    },
    "mxfp4_e8m0_g32_scales": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 32, "sm_103a": 32, "sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 128, "sm_103a": 128, "sm_107a": 256},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_79d268b9599a2036c54d",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_79d268b9599a2036c54d",
        "scale_out": True,
    },
    # MXFP8: E4M3 values, one E8M0 (power-of-two) scale per 32 elements.
    "mxfp8_e8m0_g32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 32, "sm_103a": 32, "sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 128, "sm_103a": 128, "sm_107a": 256},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_21b6556b9b84f1fd0731",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_21b6556b9b84f1fd0731",
        "scale_out": False,
    },
    "mxfp8_e8m0_g32_scales": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 32, "sm_103a": 32, "sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 128, "sm_103a": 128, "sm_107a": 256},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_e51efb4c536c4e1ed3a3",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_e51efb4c536c4e1ed3a3",
        "scale_out": True,
    },
    # The small build of each route: 256-thread CTAs, two vectors per thread, for the n_vecs of its ``vectors``.
    "fp4_e4m3_g16_small": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "fp4_e4m3_g16",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 16,
        "kernel_symbol": "kernel_cake_act_qdq_31fe4b142ec4ef695bc0",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_31fe4b142ec4ef695bc0",
        "scale_out": False,
        "vectors": {
            "sm_100a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_107a": {"in_place": (1, 262144), "out_of_place": (1, 262144)},
        },
    },
    "fp4_e4m3_g16_scales_small": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "fp4_e4m3_g16",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 16,
        "kernel_symbol": "kernel_cake_act_qdq_2023c9ed5353480a0770",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_2023c9ed5353480a0770",
        "scale_out": True,
        "vectors": {
            "sm_100a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_107a": {"in_place": (1, 262144), "out_of_place": (1, 262144)},
        },
    },
    "mxfp4_e8m0_g32_small": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_fd1904d6eabe30c83dd8",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_fd1904d6eabe30c83dd8",
        "scale_out": False,
        "vectors": {
            "sm_100a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_107a": {"in_place": (1, 262144), "out_of_place": (1, 262144)},
        },
    },
    "mxfp4_e8m0_g32_scales_small": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_fcf58c212757a8738714",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_fcf58c212757a8738714",
        "scale_out": True,
        "vectors": {
            "sm_100a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_107a": {"in_place": (1, 262144), "out_of_place": (1, 262144)},
        },
    },
    "mxfp8_e8m0_g32_small": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_d71450a083590e4d0ab9",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_d71450a083590e4d0ab9",
        "scale_out": False,
        "vectors": {
            "sm_100a": {"in_place": None, "out_of_place": (65537, 131072)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 131072)},
            "sm_107a": {"in_place": (1, 262144), "out_of_place": (1, 262144)},
        },
    },
    "mxfp8_e8m0_g32_scales_small": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 8, "sm_103a": 8, "sm_107a": 4},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512, "sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_bd4ec0bb720b07117a05",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_bd4ec0bb720b07117a05",
        "scale_out": True,
        "vectors": {
            "sm_100a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_107a": {"in_place": (1, 262144), "out_of_place": (1, 262144)},
        },
    },
    # The wide build of MXFP8 without the scales, SM100 and SM103 only:
    # 128-thread CTAs, four vectors per thread, one CTA per tile, for the
    # n_vecs of its ``vectors``.
    "mxfp8_e8m0_g32_wide": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 10, "sm_103a": 10},
            "kind": "tiles",
            "persist_waves": 0,
            "tile_vectors": {"sm_100a": 512, "sm_103a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_9b71c6472447855ead3b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_9b71c6472447855ead3b",
        "scale_out": False,
        "vectors": {
            "sm_100a": {"in_place": (262145, 33554431), "out_of_place": (131073, 2147483136)},
            "sm_103a": {"in_place": (262145, 33554431), "out_of_place": (131073, 2147483136)},
        },
    },
    # The large build of MXFP4 without the scales, SM100 and SM103 only:
    # 1024-thread CTAs, two vectors per thread, one CTA per tile, for the
    # n_vecs of its ``vectors`` (in place only).
    "mxfp4_e8m0_g32_large": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 2, "sm_103a": 2},
            "kind": "tiles",
            "persist_waves": 0,
            "tile_vectors": {"sm_100a": 2048, "sm_103a": 2048},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_4bdefd2badefabe8d780",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_4bdefd2badefabe8d780",
        "scale_out": False,
        "vectors": {
            "sm_100a": {"in_place": (4194304, 2147481600), "out_of_place": None},
            "sm_103a": {"in_place": (4194304, 2147481600), "out_of_place": None},
        },
    },
    # The large build of MXFP4 with the scales, SM100 and SM103 only:
    # 1024-thread CTAs, two vectors per thread, one CTA per tile, for the
    # n_vecs of its ``vectors`` (in place only).
    "mxfp4_e8m0_g32_scales_large": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 2, "sm_103a": 2},
            "kind": "tiles",
            "persist_waves": 0,
            "tile_vectors": {"sm_100a": 2048, "sm_103a": 2048},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_26369f0d3a4d2e709630",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_26369f0d3a4d2e709630",
        "scale_out": True,
        "vectors": {
            "sm_100a": {"in_place": (4194304, 2147481600), "out_of_place": None},
            "sm_103a": {"in_place": (4194304, 2147481600), "out_of_place": None},
        },
    },
    # The large build of MXFP8 without the scales, SM100 and SM103 only:
    # 1024-thread CTAs, two vectors per thread, one CTA per tile, for the
    # n_vecs of its ``vectors`` (in place only, above the wide range).
    "mxfp8_e8m0_g32_large": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 2, "sm_103a": 2},
            "kind": "tiles",
            "persist_waves": 0,
            "tile_vectors": {"sm_100a": 2048, "sm_103a": 2048},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_bcbfc5f217bc58e7c8a9",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_bcbfc5f217bc58e7c8a9",
        "scale_out": False,
        "vectors": {
            "sm_100a": {"in_place": (33554432, 2147481600), "out_of_place": None},
            "sm_103a": {"in_place": (33554432, 2147481600), "out_of_place": None},
        },
    },
    # The large build of MXFP8 with the scales, SM100 and SM103 only:
    # 1024-thread CTAs, two vectors per thread, one CTA per tile, for the
    # n_vecs of its ``vectors``.
    "mxfp8_e8m0_g32_scales_large": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_100a": 2, "sm_103a": 2},
            "kind": "tiles",
            "persist_waves": 0,
            "tile_vectors": {"sm_100a": 2048, "sm_103a": 2048},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_a1b2f2c224d421a29e22",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_a1b2f2c224d421a29e22",
        "scale_out": True,
        "vectors": {
            "sm_100a": {"in_place": (4194304, 2147481600), "out_of_place": (8388608, 2147481600)},
            "sm_103a": {"in_place": (4194304, 2147481600), "out_of_place": (8388608, 2147481600)},
        },
    },
    # The mid build of MXFP4 without the scales, SM107 only: 64-thread CTAs,
    # eight vectors per thread, for the n_vecs of its ``vectors``.
    "mxfp4_e8m0_g32_mid": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_5ddb470d9d818fe82b6b",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_5ddb470d9d818fe82b6b",
        "scale_out": False,
        "vectors": {
            "sm_107a": {"in_place": (933888, 1736704), "out_of_place": (933888, 1736704)},
        },
    },
    # The mid build of MXFP4 with the scales, SM107 only: 64-thread CTAs,
    # eight vectors per thread, for the n_vecs of its ``vectors``.
    "mxfp4_e8m0_g32_scales_mid": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_9f47fc0fe25e824b0f73",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_9f47fc0fe25e824b0f73",
        "scale_out": True,
        "vectors": {
            "sm_107a": {"in_place": (933888, 1736704), "out_of_place": (933888, 1736704)},
        },
    },
    # The mid build of MXFP8 without the scales, SM107 only: 64-thread CTAs,
    # eight vectors per thread, for the n_vecs of its ``vectors``.
    "mxfp8_e8m0_g32_mid": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_46ee6195467c78a3c8e3",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_46ee6195467c78a3c8e3",
        "scale_out": False,
        "vectors": {
            "sm_107a": {"in_place": (868353, 1736704), "out_of_place": (868353, 1736704)},
        },
    },
    # The mid build of MXFP8 with the scales, SM107 only: 64-thread CTAs,
    # eight vectors per thread, for the n_vecs of its ``vectors``.
    "mxfp8_e8m0_g32_scales_mid": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["parameter", "n_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "ctas_per_sm": {"sm_107a": 16},
            "kind": "tiles",
            "persist_waves": 2.5,
            "tile_vectors": {"sm_107a": 512},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_bdc991febcca0ecf2cf6",
        "launch": {
            "block": (64, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_bdc991febcca0ecf2cf6",
        "scale_out": True,
        "vectors": {
            "sm_107a": {"in_place": (868353, 1736704), "out_of_place": (868353, 1736704)},
        },
    },
    # The interleaved-domain build of each route, SM107 only (``DOMAIN_ROUTES``):
    # tiles of 128 threads (1,024 vectors), one tile an item, over memory whose
    # 2 MB chunks alternate between the two dies, each CTA claiming items of
    # its own die's chunks first, out of place, for the n_vecs of its
    # ``vectors``; a call with ``locality="domain"`` runs it when
    # :func:`domain_route` places ``x`` and ``y`` in such memory.
    "fp4_e4m3_g16_domain": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "n_vecs"], ["parameter", "first_tile"], ["parameter", "first_die"], ["parameter", "chunk_tile_shift"], ["parameter", "batch_shift"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "format": "fp4_e4m3_g16",
        "grid_rule": {
            "batch_shift": {"sm_107a": 0},
            "chunk_bytes": 2097152,
            "ctas_per_sm": {"sm_107a": 7},
            "kind": "domain_items",
            "tile_vectors": {"sm_107a": 1024},
            "vector_elements": 8,
        },
        "group": 16,
        "kernel_symbol": "kernel_cake_act_qdq_dfdafec075fee7302f3b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_dfdafec075fee7302f3b",
        "scale_out": False,
        "vectors": {"sm_107a": {"in_place": None, "out_of_place": (8388608, 2147482624)}},
    },
    "fp4_e4m3_g16_scales_domain": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "n_vecs"], ["parameter", "first_tile"], ["parameter", "first_die"], ["parameter", "chunk_tile_shift"], ["parameter", "batch_shift"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "format": "fp4_e4m3_g16",
        "grid_rule": {
            "batch_shift": {"sm_107a": 0},
            "chunk_bytes": 2097152,
            "ctas_per_sm": {"sm_107a": 7},
            "kind": "domain_items",
            "tile_vectors": {"sm_107a": 1024},
            "vector_elements": 8,
        },
        "group": 16,
        "kernel_symbol": "kernel_cake_act_qdq_851b2cd815bd9f70bd34",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_851b2cd815bd9f70bd34",
        "scale_out": True,
        "vectors": {"sm_107a": {"in_place": None, "out_of_place": (8388608, 2147482624)}},
    },
    "mxfp4_e8m0_g32_domain": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "n_vecs"], ["parameter", "first_tile"], ["parameter", "first_die"], ["parameter", "chunk_tile_shift"], ["parameter", "batch_shift"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "batch_shift": {"sm_107a": 0},
            "chunk_bytes": 2097152,
            "ctas_per_sm": {"sm_107a": 8},
            "kind": "domain_items",
            "tile_vectors": {"sm_107a": 1024},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_3fcef995964af95f49e7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_3fcef995964af95f49e7",
        "scale_out": False,
        "vectors": {"sm_107a": {"in_place": None, "out_of_place": (8388608, 2147482624)}},
    },
    "mxfp4_e8m0_g32_scales_domain": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "n_vecs"], ["parameter", "first_tile"], ["parameter", "first_die"], ["parameter", "chunk_tile_shift"], ["parameter", "batch_shift"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "format": "mxfp4_e8m0_g32",
        "grid_rule": {
            "batch_shift": {"sm_107a": 0},
            "chunk_bytes": 2097152,
            "ctas_per_sm": {"sm_107a": 8},
            "kind": "domain_items",
            "tile_vectors": {"sm_107a": 1024},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_db936c037a2c98a5db5c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_db936c037a2c98a5db5c",
        "scale_out": True,
        "vectors": {"sm_107a": {"in_place": None, "out_of_place": (8388608, 2147482624)}},
    },
    "mxfp8_e8m0_g32_domain": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "n_vecs"], ["parameter", "first_tile"], ["parameter", "first_die"], ["parameter", "chunk_tile_shift"], ["parameter", "batch_shift"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "batch_shift": {"sm_107a": 0},
            "chunk_bytes": 2097152,
            "ctas_per_sm": {"sm_107a": 8},
            "kind": "domain_items",
            "tile_vectors": {"sm_107a": 1024},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_c3252edeaabce416eab7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_c3252edeaabce416eab7",
        "scale_out": False,
        "vectors": {"sm_107a": {"in_place": None, "out_of_place": (8388608, 2147482624)}},
    },
    "mxfp8_e8m0_g32_scales_domain": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "x"], ["buffer", "y"], ["buffer", "scales"], ["buffer", "counters0"], ["buffer", "counters1"], ["parameter", "n_vecs"], ["parameter", "first_tile"], ["parameter", "first_die"], ["parameter", "chunk_tile_shift"], ["parameter", "batch_shift"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "counters": {"dtype": "uint32", "rearm": "kernel", "words": 1},
        "ffi_entry": "run",
        "format": "mxfp8_e8m0_g32",
        "grid_rule": {
            "batch_shift": {"sm_107a": 0},
            "chunk_bytes": 2097152,
            "ctas_per_sm": {"sm_107a": 8},
            "kind": "domain_items",
            "tile_vectors": {"sm_107a": 1024},
            "vector_elements": 8,
        },
        "group": 32,
        "kernel_symbol": "kernel_cake_act_qdq_48c0dbfaea5f3b9c07f7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": False,
        },
        "library": "cake_act_qdq_48c0dbfaea5f3b9c07f7",
        "scale_out": True,
        "vectors": {"sm_107a": {"in_place": None, "out_of_place": (8388608, 2147482624)}},
    },
}

# Per architecture, the records of each route: its main record and its ranged
# records (its small record and, on SM100 and SM103, MXFP8's wide record and
# the large record of MXFP4 and MXFP8), which a call runs for the ``n_vecs``
# of their disjoint ``vectors`` ranges for the call's kind
# (:func:`selected_module`).
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "fp4_e4m3_g16": ["fp4_e4m3_g16", "fp4_e4m3_g16_small"],
        "fp4_e4m3_g16_scales": ["fp4_e4m3_g16_scales", "fp4_e4m3_g16_scales_small"],
        "mxfp4_e8m0_g32": ["mxfp4_e8m0_g32", "mxfp4_e8m0_g32_small", "mxfp4_e8m0_g32_large"],
        "mxfp4_e8m0_g32_scales": [
            "mxfp4_e8m0_g32_scales",
            "mxfp4_e8m0_g32_scales_small",
            "mxfp4_e8m0_g32_scales_large",
        ],
        "mxfp8_e8m0_g32": ["mxfp8_e8m0_g32", "mxfp8_e8m0_g32_small", "mxfp8_e8m0_g32_wide", "mxfp8_e8m0_g32_large"],
        "mxfp8_e8m0_g32_scales": [
            "mxfp8_e8m0_g32_scales",
            "mxfp8_e8m0_g32_scales_small",
            "mxfp8_e8m0_g32_scales_large",
        ],
    },
    "sm_103a": {
        "fp4_e4m3_g16": ["fp4_e4m3_g16", "fp4_e4m3_g16_small"],
        "fp4_e4m3_g16_scales": ["fp4_e4m3_g16_scales", "fp4_e4m3_g16_scales_small"],
        "mxfp4_e8m0_g32": ["mxfp4_e8m0_g32", "mxfp4_e8m0_g32_small", "mxfp4_e8m0_g32_large"],
        "mxfp4_e8m0_g32_scales": [
            "mxfp4_e8m0_g32_scales",
            "mxfp4_e8m0_g32_scales_small",
            "mxfp4_e8m0_g32_scales_large",
        ],
        "mxfp8_e8m0_g32": ["mxfp8_e8m0_g32", "mxfp8_e8m0_g32_small", "mxfp8_e8m0_g32_wide", "mxfp8_e8m0_g32_large"],
        "mxfp8_e8m0_g32_scales": [
            "mxfp8_e8m0_g32_scales",
            "mxfp8_e8m0_g32_scales_small",
            "mxfp8_e8m0_g32_scales_large",
        ],
    },
    "sm_107a": {
        "fp4_e4m3_g16": ["fp4_e4m3_g16", "fp4_e4m3_g16_small"],
        "fp4_e4m3_g16_scales": ["fp4_e4m3_g16_scales", "fp4_e4m3_g16_scales_small"],
        "mxfp4_e8m0_g32": ["mxfp4_e8m0_g32", "mxfp4_e8m0_g32_small", "mxfp4_e8m0_g32_mid"],
        "mxfp4_e8m0_g32_scales": ["mxfp4_e8m0_g32_scales", "mxfp4_e8m0_g32_scales_small", "mxfp4_e8m0_g32_scales_mid"],
        "mxfp8_e8m0_g32": ["mxfp8_e8m0_g32", "mxfp8_e8m0_g32_small", "mxfp8_e8m0_g32_mid"],
        "mxfp8_e8m0_g32_scales": ["mxfp8_e8m0_g32_scales", "mxfp8_e8m0_g32_scales_small", "mxfp8_e8m0_g32_scales_mid"],
    },
}

# Per architecture, the interleaved-domain record of each route.  No route of
# ``ROUTES`` lists one: a call runs it only with ``locality="domain"`` and only
# where :func:`domain_route` places its memory.
DOMAIN_ROUTES: dict[str, dict[str, str]] = {
    "sm_107a": {
        "fp4_e4m3_g16": "fp4_e4m3_g16_domain",
        "fp4_e4m3_g16_scales": "fp4_e4m3_g16_scales_domain",
        "mxfp4_e8m0_g32": "mxfp4_e8m0_g32_domain",
        "mxfp4_e8m0_g32_scales": "mxfp4_e8m0_g32_scales_domain",
        "mxfp8_e8m0_g32": "mxfp8_e8m0_g32_domain",
        "mxfp8_e8m0_g32_scales": "mxfp8_e8m0_g32_scales_domain",
    },
}

# One build per architecture: B200 / GB200 (``sm_100a``), B300 / GB300
# (``sm_103a``) and R200 (``sm_107a``), each compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``).  No family fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The formats and the group size of each: one scale per ``G`` consecutive
# elements of the last dimension.
FORMATS = ("fp4_e4m3_g16", "mxfp4_e8m0_g32", "mxfp8_e8m0_g32")
GROUP_SIZES: dict[str, int] = {
    "fp4_e4m3_g16": 16,
    "mxfp4_e8m0_g32": 32,
    "mxfp8_e8m0_g32": 32,
}
# The routes (``ROUTES[arch]`` keys): a format, and ``_scales`` when the
# program also stores the FP32 scales.
KINDS = tuple(f"{fmt}{suffix}" for fmt in FORMATS for suffix in ("", "_scales"))
# The grid-rule vocabulary (``record["grid_rule"]["kind"]``): ``tiles`` takes
# ``tiles = ceil(n_vecs / tile_vectors[arch])`` and ``resident =
# ctas_per_sm[arch] * SM count``, and launches ``tiles`` CTAs once ``tiles >
# persist_waves * resident``, else ``min(tiles, resident)``, where ``n_vecs =
# numel / vector_elements``.
GRID_KINDS = ("tiles",)
# The grid rule of an interleaved-domain record (:func:`domain_launch`).
DOMAIN_GRID_KIND = "domain_items"
# The claim workspace of an interleaved-domain record (``record["counters"]``):
# its word type and who re-arms it to zero (``"kernel"``: every launch leaves
# it zero, so the host zeroes it once, when it makes it).
COUNTER_DTYPE = "uint32"
COUNTER_REARM = "kernel"
# The reasons of :func:`domain_route`: ``"placed"`` for the interleaved-domain
# program, any other for the plain program of the route (the same bytes).
DOMAIN_REASONS = (
    "placed",
    "arch",
    "in_place",
    "no_out",
    "size",
    "x_memory",
    "out_memory",
    "chunk_offset",
    "die_parity",
)
# The grid arguments of an argument plan, in order.
GRID_AXES = ("grid_x", "grid_y", "grid_z")
# BF16 elements per vector (16 bytes): a thread's load and store.
VECTOR = 8
# The kernels load and store 16-byte vectors.
ALIGNMENT_BYTES = 16
INT32_MAX = 2**31 - 1
# ``n_vecs`` is a signed 32-bit kernel argument, and the kernel forms
# ``n_vecs + 255`` (its tile count on SM107, ``+ 127`` on SM100 / SM103) and
# its vector indices below that in signed 32 bits.  A small or wide program
# forms ``n_vecs + 511``, a large one ``n_vecs + 2047`` and an
# interleaved-domain one ``n_vecs + 1023``; a call runs each only inside its
# ``vectors`` range.
MAX_VECTORS = INT32_MAX - 255


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """The architecture of one device capability, if the package has one."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_name(format: str, *, scale_out: bool) -> str:
    """The route of one call: the format, with ``_scales`` when the call stores
    the scales (``ValueError`` for an unknown format)."""

    if format not in GROUP_SIZES:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    return f"{format}_scales" if scale_out else format


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """The records of one route on ``arch``."""

    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake activation QDQ route {kind!r} on {arch!r}; "
            f"architectures: {ARCHES}"
        )
    return tuple(names)


def selected_module(
    arch: str, format: str, *, scale_out: bool, n_vecs: int, in_place: bool
) -> str:
    """The record a call of ``format`` with ``n_vecs`` vectors runs on ``arch``,
    in place (the output is the input) or out of place.

    A route lists one main record and its ranged records (the records with
    ``vectors``: a small record, and on some routes a wide, a large or a mid
    one, or a wide and a large one).  The call
    runs the ranged record whose inclusive ``vectors[arch]`` range for the
    call's kind (``"in_place"`` or ``"out_of_place"``; ``None`` is no size)
    holds ``n_vecs``, and the main record when none does.
    Every record of the route must serve the route's format, its scale output
    and ``arch``, and at most one range may hold ``n_vecs``; anything else is a
    registry error (``RuntimeError``).
    """

    kind = route_name(format, scale_out=scale_out)
    names = route_modules(arch, kind)
    main = [name for name in names if "vectors" not in MODULES[name]]
    ranged = [name for name in names if "vectors" in MODULES[name]]
    if len(main) != 1:
        raise RuntimeError(
            f"the route {kind!r} on {arch} lists {list(names)}; expected one main "
            f"record and its ranged records"
        )
    for name in names:
        record = MODULES[name]
        if (
            record["format"] != format
            or bool(record["scale_out"]) != bool(scale_out)
            or record["group"] != GROUP_SIZES[format]
            or arch not in record["arches"]
        ):
            raise RuntimeError(
                f"the record {name} of route {kind!r} on {arch} does not serve that route"
            )
    call = "in_place" if in_place else "out_of_place"
    held = []
    for name in ranged:
        span = MODULES[name]["vectors"].get(arch, {}).get(call)
        if span is not None and int(span[0]) <= int(n_vecs) <= int(span[1]):
            held.append(name)
    if len(held) > 1:
        raise RuntimeError(
            f"the ranges of {held} of route {kind!r} on {arch} all hold "
            f"n_vecs={n_vecs} for a {call} call"
        )
    return held[0] if held else main[0]


def domain_module(arch: str, format: str, *, scale_out: bool) -> str | None:
    """The interleaved-domain record of the route of ``format`` (with its scale
    output) on ``arch``, or None when ``arch`` has none (``ValueError`` for an
    unknown format).  A record that does not serve the route, or whose grid
    rule or claim workspace is outside the vocabulary, is a registry error
    (``RuntimeError``)."""

    kind = route_name(format, scale_out=scale_out)
    name = DOMAIN_ROUTES.get(arch, {}).get(kind)
    if name is None:
        return None
    record = MODULES.get(name)
    if (
        record is None
        or record["format"] != format
        or bool(record["scale_out"]) != bool(scale_out)
        or record["group"] != GROUP_SIZES[format]
        or arch not in record["arches"]
        or record["grid_rule"]["kind"] != DOMAIN_GRID_KIND
        or record["counters"]["rearm"] != COUNTER_REARM
        or record["counters"]["dtype"] != COUNTER_DTYPE
        or int(record["counters"]["words"]) < 1
    ):
        raise RuntimeError(
            f"the interleaved-domain record {name} of route {kind!r} on {arch} does not serve that route"
        )
    return name


def domain_chunk_bytes(arch: str) -> int | None:
    """The chunk of the interleaved-domain records of ``arch`` (the bytes of
    memory in one die before the next chunk in the other die), or None when
    ``arch`` has none.  Records that disagree are a registry error
    (``RuntimeError``)."""

    sizes = {int(MODULES[name]["grid_rule"]["chunk_bytes"]) for name in DOMAIN_ROUTES.get(arch, {}).values()}
    if not sizes:
        return None
    if len(sizes) != 1:
        raise RuntimeError(f"the interleaved-domain records of {arch} disagree on their chunk: {sorted(sizes)}")
    return sizes.pop()


def chunk_place(address: int, nbytes: int, base: int, size: int, chunk_bytes: int) -> tuple[int, int] | None:
    """Where ``nbytes`` bytes from ``address`` lie in a buffer of ``size`` bytes
    from ``base`` whose chunk ``k`` of ``chunk_bytes`` lies in die ``k & 1``:
    ``(offset of the first byte in its chunk, die of that chunk)``, or None when
    the bytes do not all lie in the buffer."""

    if not base <= address < base + size or address + nbytes > base + size:
        return None
    chunk, offset = divmod(address - base, chunk_bytes)
    return offset, chunk & 1


def domain_candidate(
    arch: str,
    format: str,
    *,
    scale_out: bool,
    n_vecs: int,
    in_place: bool,
    out_given: bool,
) -> tuple[str | None, str]:
    """The first four checks of :func:`domain_route`, which read no memory
    placement: ``(record, "candidate")`` for the interleaved-domain record of
    the route when ``arch`` has one, the call is out of place with an ``out``
    buffer and ``n_vecs`` lies in the record's ``vectors`` range, else
    ``(None, reason)`` with the first of ``"arch"``, ``"in_place"``,
    ``"no_out"`` and ``"size"`` that holds."""

    name = domain_module(arch, format, scale_out=scale_out)
    if name is None:
        return None, "arch"
    if in_place:
        return None, "in_place"
    if not out_given:
        return None, "no_out"
    span = MODULES[name]["vectors"].get(arch, {}).get("out_of_place")
    if span is None or not int(span[0]) <= int(n_vecs) <= int(span[1]):
        return None, "size"
    return name, "candidate"


def domain_placement(
    x_place: tuple[int, int] | None, y_place: tuple[int, int] | None
) -> str:
    """The last four checks of :func:`domain_route`, on the placements of a
    candidate's input and output (:func:`chunk_place` results): ``"placed"``,
    or the first of ``"x_memory"``, ``"out_memory"``, ``"chunk_offset"`` and
    ``"die_parity"`` that holds."""

    if x_place is None:
        return "x_memory"
    if y_place is None:
        return "out_memory"
    if x_place[0] != y_place[0]:
        return "chunk_offset"
    if x_place[1] != y_place[1]:
        return "die_parity"
    return "placed"


def domain_route(
    arch: str,
    format: str,
    *,
    scale_out: bool,
    n_vecs: int,
    in_place: bool,
    out_given: bool,
    x_place: tuple[int, int] | None,
    y_place: tuple[int, int] | None,
) -> tuple[str | None, str]:
    """The program of a ``locality="domain"`` call of ``format`` with
    ``n_vecs`` vectors on ``arch``: ``(record, "placed")`` for the
    interleaved-domain record of its route, or ``(None, reason)`` for the
    route's plain program (:func:`selected_module`, the same bytes), with the
    first reason that holds, in this order:

    - ``"arch"``: ``arch`` has no interleaved-domain record;
    - ``"in_place"``: the output is the input;
    - ``"no_out"``: the call has no ``out`` buffer (its output would be new
      memory that does not alternate dies);
    - ``"size"``: ``n_vecs`` lies outside the record's ``vectors`` range;
    - ``"x_memory"``: the input does not lie in one buffer whose chunks
      alternate dies (``x_place`` None);
    - ``"out_memory"``: the output does not (``y_place`` None);
    - ``"chunk_offset"``: the two start at different offsets in their chunks;
    - ``"die_parity"``: their chunks lie in different dies.

    ``x_place`` and ``y_place`` are :func:`chunk_place` results.  The first
    four are :func:`domain_candidate`, the last four
    :func:`domain_placement`.
    """

    name, reason = domain_candidate(
        arch,
        format,
        scale_out=scale_out,
        n_vecs=n_vecs,
        in_place=in_place,
        out_given=out_given,
    )
    if name is None:
        return None, reason
    reason = domain_placement(x_place, y_place)
    return (name, reason) if reason == "placed" else (None, reason)


def domain_launch(
    record: dict[str, Any], *, n_vecs: int, x_place: tuple[int, int], sm_count: int, arch: str
) -> dict[str, Any]:
    """The scalar arguments and the grid of interleaved-domain ``record`` for
    ``n_vecs`` vectors whose first byte lies at ``x_place``
    (:func:`chunk_place`) on a device of architecture ``arch`` with
    ``sm_count`` SMs: ``first_tile``, ``first_die``, ``chunk_tile_shift``,
    ``batch_shift`` and ``grid``.

    A tile is ``T = tile_vectors[arch]`` vectors (``16 T`` bytes), and a chunk
    of ``chunk_bytes`` holds ``2^chunk_tile_shift`` tiles.  Tile ``c`` of ``x``
    lies in slot ``first_tile + c``: ``first_tile`` is ``x``'s offset in its
    chunk in whole tiles, rounded down (an ``x`` that starts inside a tile
    moves less than one tile of bytes across the dies, never what it
    computes), and chunk ``k`` lies in die ``first_die ^ (k & 1)``.  An item
    is ``2^batch_shift`` slots of one chunk (``batch_shift`` at most
    ``chunk_tile_shift``).  The chunks ``0`` to ``last_chunk = (first_tile +
    tiles - 1) >> chunk_tile_shift`` hold the items, each die's chunks in turn,
    and the grid is one CTA per item, at most ``ctas_per_sm[arch]`` per SM; any
    grid of at least one CTA gives the same bytes.
    """

    rule = record["grid_rule"]
    if rule["kind"] != DOMAIN_GRID_KIND or rule["vector_elements"] != VECTOR:
        raise ValueError(
            f"not a Cake activation QDQ interleaved-domain grid rule: {rule!r} (kind {DOMAIN_GRID_KIND!r}, "
            f"{VECTOR}-element vectors)"
        )
    if arch not in rule["tile_vectors"]:
        raise ValueError(f"the Cake activation QDQ grid rule has no tile for {arch}: {rule!r}")
    tile = int(rule["tile_vectors"][arch])
    tile_bytes = tile * VECTOR * 2  # BF16
    slots, rest = divmod(int(rule["chunk_bytes"]), tile_bytes)
    if rest or slots < 1 or slots & (slots - 1):
        raise ValueError(f"the chunk of {rule!r} is not a power-of-two count of {tile_bytes}-byte tiles")
    chunk_tile_shift = slots.bit_length() - 1
    batch_shift = min(int(rule["batch_shift"][arch]), chunk_tile_shift)
    offset, first_die = int(x_place[0]), int(x_place[1])
    first_tile = offset // tile_bytes
    tiles = -(-int(n_vecs) // tile)
    last_chunk = (first_tile + tiles - 1) >> chunk_tile_shift
    items = sum(
        ((last_chunk - die + 2) >> 1) << (chunk_tile_shift - batch_shift) for die in (first_die, first_die ^ 1)
    )
    resident = int(rule["ctas_per_sm"][arch]) * int(sm_count)
    return {
        "first_tile": first_tile,
        "first_die": first_die,
        "chunk_tile_shift": chunk_tile_shift,
        "batch_shift": batch_shift,
        "grid": (max(1, min(items, resident)), 1, 1),
    }


@functools.cache
def device_arch(device_index: int) -> str:
    """The architecture of one CUDA device (``RuntimeError`` when the package has
    none for it); queried once per device."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not a Cake "
            f"activation QDQ target; architectures: {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """The SM count of one CUDA device, queried once per device (a host query,
    no device synchronization)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(record: dict[str, Any], *, n_vecs: int, sm_count: int, arch: str) -> tuple[int, int, int]:
    """The grid of one record for ``n_vecs`` vectors on a device of
    architecture ``arch`` with ``sm_count`` SMs."""

    rule = record["grid_rule"]
    if rule["kind"] not in GRID_KINDS or rule["vector_elements"] != VECTOR:
        raise ValueError(
            f"unknown Cake activation QDQ grid rule: {rule!r} (known kinds: {GRID_KINDS}, "
            f"{VECTOR}-element vectors)"
        )
    if arch not in rule["tile_vectors"]:
        raise ValueError(f"the Cake activation QDQ grid rule has no tile for {arch}: {rule!r}")
    tiles = -(-int(n_vecs) // int(rule["tile_vectors"][arch]))
    resident = int(rule["ctas_per_sm"][arch]) * int(sm_count)
    if tiles > float(rule["persist_waves"]) * resident:
        return (tiles, 1, 1)
    return (min(tiles, resident), 1, 1)


def library_name(name: str, arch: str) -> str:
    """The ``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake activation QDQ record {name} is not built for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else the JIT
    build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    a warm-up build)."""

    return cake_build.jit_specs(arch)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    device_index: int,
    grid: tuple[int, int, int] | None = None,
) -> None:
    """Launch record ``name`` with the kernel arguments ``values``.

    ``values`` maps the kernel parameter names of the record's ``arg_plan``
    (``x``, ``y``, ``scales``, ``n_vecs``; for an interleaved-domain record
    also ``counters0``, ``counters1``, ``first_tile``, ``first_die``,
    ``chunk_tile_shift`` and ``batch_shift``) to tensors and Python integers.
    The grid is the record's :func:`launch_grid` unless ``grid`` is given (any
    grid of at least one CTA is correct); an interleaved-domain record takes
    the grid of :func:`domain_launch`.  The kernel runs on the current torch
    stream.
    """

    record = MODULES[name]
    if grid is None:
        grid = launch_grid(
            record, n_vecs=values["n_vecs"], sm_count=sm_count(device_index), arch=device_arch(device_index)
        )
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid[GRID_AXES.index(key)]))
            continue
        if key not in values:
            raise ValueError(f"module {name} needs the kernel argument {key!r}")
        args.append(values[key])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ALIGNMENT_BYTES",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "COUNTER_DTYPE",
    "COUNTER_REARM",
    "DOMAIN_GRID_KIND",
    "DOMAIN_REASONS",
    "DOMAIN_ROUTES",
    "FORMATS",
    "GRID_AXES",
    "GRID_KINDS",
    "GROUP_SIZES",
    "INT32_MAX",
    "KINDS",
    "MAX_VECTORS",
    "MODULES",
    "ROUTES",
    "VECTOR",
    "arch_for_capability",
    "chunk_place",
    "device_arch",
    "domain_candidate",
    "domain_chunk_bytes",
    "domain_launch",
    "domain_module",
    "domain_placement",
    "domain_route",
    "launch_grid",
    "library_name",
    "load",
    "route_modules",
    "route_name",
    "run_module",
    "selected_module",
    "sm_count",
    "specs_for_arch",
]
