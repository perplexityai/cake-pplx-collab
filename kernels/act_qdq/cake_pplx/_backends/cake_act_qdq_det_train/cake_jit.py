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
``_small`` for the route's small build.  The host plan of a record is
``format``, ``scale_out``, ``group`` (the elements of one scale group),
``arches``, ``ffi_entry``, ``grid_rule`` and, for a small record only,
``vectors``: per architecture and kind of call (``out_of_place`` or
``in_place``), the inclusive ``n_vecs`` range its route runs it for, or
``None`` for no size.  The derived leaves (``library``, ``kernel_symbol``,
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
about one wave and the larger CTAs start sooner: out of place 65,537 to
524,288 on SM100 and SM103 and 1 to 262,144 on SM107, in place 1 to 262,144
on SM107 only (in place on SM100 and SM103 the small program measured no
faster).  Every other call runs the main record (:func:`selected_module`).
Both give the same bytes.  ``n_vecs`` is a run-time scalar, so a library
serves every shape; a call builds or loads only its own program's library on
first use.  Build every library of the architecture (:func:`specs_for_arch`)
before a CUDA graph capture, to keep the JIT out of the capture.
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
            "sm_100a": {"in_place": None, "out_of_place": (65537, 524288)},
            "sm_103a": {"in_place": None, "out_of_place": (65537, 524288)},
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
}

# Per architecture, the records of each route: its main record and its small
# record, which a call runs for the ``n_vecs`` of the small record's
# ``vectors`` range for the call's kind (:func:`selected_module`).
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "fp4_e4m3_g16": ["fp4_e4m3_g16", "fp4_e4m3_g16_small"],
        "fp4_e4m3_g16_scales": ["fp4_e4m3_g16_scales", "fp4_e4m3_g16_scales_small"],
        "mxfp4_e8m0_g32": ["mxfp4_e8m0_g32", "mxfp4_e8m0_g32_small"],
        "mxfp4_e8m0_g32_scales": ["mxfp4_e8m0_g32_scales", "mxfp4_e8m0_g32_scales_small"],
        "mxfp8_e8m0_g32": ["mxfp8_e8m0_g32", "mxfp8_e8m0_g32_small"],
        "mxfp8_e8m0_g32_scales": ["mxfp8_e8m0_g32_scales", "mxfp8_e8m0_g32_scales_small"],
    },
    "sm_103a": {
        "fp4_e4m3_g16": ["fp4_e4m3_g16", "fp4_e4m3_g16_small"],
        "fp4_e4m3_g16_scales": ["fp4_e4m3_g16_scales", "fp4_e4m3_g16_scales_small"],
        "mxfp4_e8m0_g32": ["mxfp4_e8m0_g32", "mxfp4_e8m0_g32_small"],
        "mxfp4_e8m0_g32_scales": ["mxfp4_e8m0_g32_scales", "mxfp4_e8m0_g32_scales_small"],
        "mxfp8_e8m0_g32": ["mxfp8_e8m0_g32", "mxfp8_e8m0_g32_small"],
        "mxfp8_e8m0_g32_scales": ["mxfp8_e8m0_g32_scales", "mxfp8_e8m0_g32_scales_small"],
    },
    "sm_107a": {
        "fp4_e4m3_g16": ["fp4_e4m3_g16", "fp4_e4m3_g16_small"],
        "fp4_e4m3_g16_scales": ["fp4_e4m3_g16_scales", "fp4_e4m3_g16_scales_small"],
        "mxfp4_e8m0_g32": ["mxfp4_e8m0_g32", "mxfp4_e8m0_g32_small"],
        "mxfp4_e8m0_g32_scales": ["mxfp4_e8m0_g32_scales", "mxfp4_e8m0_g32_scales_small"],
        "mxfp8_e8m0_g32": ["mxfp8_e8m0_g32", "mxfp8_e8m0_g32_small"],
        "mxfp8_e8m0_g32_scales": ["mxfp8_e8m0_g32_scales", "mxfp8_e8m0_g32_scales_small"],
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
# The grid arguments of an argument plan, in order.
GRID_AXES = ("grid_x", "grid_y", "grid_z")
# BF16 elements per vector (16 bytes): a thread's load and store.
VECTOR = 8
# The kernels load and store 16-byte vectors.
ALIGNMENT_BYTES = 16
INT32_MAX = 2**31 - 1
# ``n_vecs`` is a signed 32-bit kernel argument, and the kernel forms
# ``n_vecs + 255`` (its tile count on SM107, ``+ 127`` on SM100 / SM103) and
# its vector indices below that in signed 32 bits.  A small program forms
# ``n_vecs + 511``; a call runs it only inside its ``vectors`` range.
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

    A route lists one main record and at most one small record (the record
    with ``vectors``).  The call runs the small record when ``n_vecs`` lies in
    its inclusive ``vectors[arch]`` range for the call's kind
    (``"in_place"`` or ``"out_of_place"``; ``None`` is no size), and the main
    record otherwise.
    Every record of the route must serve the route's format, its scale output
    and ``arch``; anything else is a registry error (``RuntimeError``).
    """

    kind = route_name(format, scale_out=scale_out)
    names = route_modules(arch, kind)
    main = [name for name in names if "vectors" not in MODULES[name]]
    small = [name for name in names if "vectors" in MODULES[name]]
    if len(main) != 1 or len(small) > 1:
        raise RuntimeError(
            f"the route {kind!r} on {arch} lists {list(names)}; expected one main "
            f"record and at most one small record"
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
    for name in small:
        span = MODULES[name]["vectors"].get(arch, {}).get(call)
        if span is not None and int(span[0]) <= int(n_vecs) <= int(span[1]):
            return name
    return main[0]


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
    (``x``, ``y``, ``scales``, ``n_vecs``) to tensors and Python integers.
    The grid is the record's :func:`launch_grid` unless ``grid`` is given (any
    grid of at least one CTA is correct).  The kernel runs on the current
    torch stream.
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
    "device_arch",
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
