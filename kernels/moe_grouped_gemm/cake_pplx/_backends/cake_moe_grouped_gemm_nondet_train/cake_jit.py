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
"""

from __future__ import annotations

import functools
from typing import Any, Optional

from . import cake_build

# Registry of the generated ragged grouped GEMM programs of the
# non-deterministic package.  ``PROGRAMS`` and ``ROUTES`` are written by the
# Cake generated-program export; the host plan constants and the functions
# below are this package's.
#
# ``PROGRAMS``: one record per physical program -- one kernel source plus its
# host binding, shared by every architecture in ``arches`` (genuine
# per-architecture lowering differences live inside the source under exact
# ``__CUDA_ARCH__`` regions) -- with its architecture-free ``library`` stem, FFI
# entry, argument plan and launch shape (a lead twin's record also names, in
# ``twin_of``, the program it twins; below).  What to compile (the translation
# units, the compile flags, the per-architecture closure digest) is recorded in
# ``build.json`` (see ``cake_build``), whose entry for a program is
# ``<library>[<specialization tag>]_<arch>``; a program is built once per
# (architecture, compile-line specialization), prebuilt under ``aot/`` or by
# the package's JIT shim.
#
# ``ROUTES``: one record per route -- the public routes ``fwd``, ``dgrad`` and
# ``wgrad_{bf16,f32}_k{256,512}`` and the lead twin of each (below) -- naming
# its architectures and ordered stages.  Each stage binds a program and the compile-line specializations the
# loader defines for it (``WGRAD_OUT`` = 16 for bf16 or 32 for fp32 weight
# gradients, from the route's output dtype; ``WGRAD_TILE``, the k width of the
# partials the tail-reduce program sums, from the route's tile).  A stage
# serves the architectures its program is built for among the route's: a route
# may name two stages of one name whose programs serve disjoint architectures,
# so a stage is looked up per architecture (``stage_binding``).
#
# The forward, data-gradient and 256-wide k tile weight-gradient routes bind
# this package's copies of the deterministic package's programs
# (``cake_pplx._backends.cake_moe_grouped_gemm_det_train``: the same builds of the
# same kernels, so the same output bits) with the same stages: ``main``,
# ``main_wide``, ``main_wide_ef`` and ``main_wide_ts33`` of ``fwd`` / ``dgrad``,
# and ``main``, ``main_wave`` and the ordered fp32 ``tail_reduce`` of the
# 256-wide k tile.  The ``main_wide_ef`` programs are the deterministic
# package's L2 evict-first twins of the wide tile: on sm_107a the A-collector
# programs of ``fwd`` and ``dgrad``; on sm_100a and sm_103a the forward pass's
# domain-scheduled program with the X loads and Y stores evict-first, and on
# sm_100a the data gradient's with the G loads and dX stores evict-first.  The
# 512-wide k tile's weight-gradient routes bind the
# builds of the deterministic package's k512 forms -- ``main``, ``main_wave``,
# ``main_halves`` (with the planned tile walk and the slack tail placement),
# ``main_wave_halves`` (waves synchronised every eight) and, on sm_107a,
# ``main_halves_k256t`` (with the slack tail placement) -- whose split tail is
# finished in the same launch in a fixed order (the ordered one-launch tail):
# every unit of a split tail tile stores its fp32 partial to its slot
# ``j * tail_splits + s`` of ``tail_sums`` and counts its arrival on the tile's
# tickets in ``tail_tickets``; the tile's last unit to arrive reads the tile's
# slots back, adds them from 0.0 in slot order (the sum of the deterministic
# tail reduce), rounds once and stores (an accumulate build adds the sum to C
# once).  Each launch overwrites the slots it reads, so ``tail_sums`` needs no
# value between launches; only the tickets rest at zero (the last unit wraps
# them).  Those routes have no ``tail_reduce`` stage and launch one kernel.
# The k512 routes also name this package's queue-walk builds on sm_100a and
# sm_103a: ``main_halves_queue`` (both routes) and ``main_halves_queue_acc``
# (the fp32 route), the wave-free half-release builds with the planned split
# tail whose tiles before the tail are claimed from one global counter in tile
# order instead of strided over the clusters.  Each tile keeps its k-step order,
# so a queue build gives the bits of its strided twin (the same build on the
# strided walk).  It takes one more workspace, ``sched``: two u32 words, the
# counter and the retired-cluster count, at zero before and after a launch
# (the last cluster to retire sets them back to zero).  The deterministic
# package has no such stage; the host binds it in place of ``main_halves`` /
# ``main_wave_halves`` (and their accumulate stages) where ``queue_walk``
# names the call.
# Every whole tile is reduced in a fixed order by one CTA pair, as in the
# deterministic package, so the output repeats bit for bit for the same
# inputs, shapes, device SM count and ``tail_splits``.  With the slack tail
# placement the kernel plans each launch's tail on the device: when every tail
# tile fits in the room the full waves leave on the least-loaded clusters,
# each runs whole on one of them (no slot or ticket is touched), else the tail
# splits as above.
#
# The host chooses among the stages a route serves by ``HOST_PLAN_CONSTANTS``,
# with the deterministic package's rules and constants (``cake_backend``,
# program choice); ``ordered_tail_archs`` lists the architectures whose 512-wide k
# tile routes bind the ordered one-launch tail builds -- every architecture of
# those routes; ``unordered_tail_archs`` is empty (the registry holds no build
# that adds its split tail with unordered reductions); and
# ``atomic_tail_plan`` holds the cost-model constants of the one-launch split
# tail (``tail_plan`` stays the two-launch tail reduce's); ``queue_walk`` names
# the calls that bind the queue walk and ``queue_sched_words`` sizes its
# counters (this package's keys).
#
# The lead routes: beside each public route ``ROUTES`` holds its lead twin
# ``<route>_lead`` (``LEAD_SUFFIX``, ``lead_route``; not a public route) with
# the route's op, architectures, stage names and specializations, whose every
# stage binds the ``offs_lead`` twin of the program the route binds there
# (``lead_twin``: the ``PROGRAMS`` record ``<program>_lead`` whose ``twin_of``
# names the program).  A twin takes its program's argument
# plan with the ``m_begin`` buffer after ``offs``: the first entry of an
# ``m_indptr[E + 1]`` row table, read on the device, so that group 0 starts at
# row ``m_indptr[0]`` instead of row 0; on the rows ``[m_begin, sum_m)`` it
# gives the bits of its program launched over ``offs - m_begin`` with the same
# grid and scalars.  The host binds the lead route for an ``m_indptr`` row
# table and the route itself for ``[E]`` end offsets; it selects the stages and
# plans the launch on the route either way (``cake_backend``, row tables).
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_001f83d4fc1d538b8731": {
        "library": "cake_moe_grouped_gemm_001f83d4fc1d538b8731",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_09a336207c15c131f5dc": {
        "library": "cake_moe_grouped_gemm_09a336207c15c131f5dc",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["parameter", "lda"],
            ["parameter", "ldb"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_12a3b3c91ad6f91315e0": {
        "library": "cake_moe_grouped_gemm_12a3b3c91ad6f91315e0",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_1edcad995dcc790600cb": {
        "library": "cake_moe_grouped_gemm_1edcad995dcc790600cb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb": {
        "library": "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_27a75f32c7a0e39f50f3": {
        "library": "cake_moe_grouped_gemm_27a75f32c7a0e39f50f3",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_2803c50f00dd015bf1d4": {
        "library": "cake_moe_grouped_gemm_2803c50f00dd015bf1d4",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "sched"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_3edd5c53a1269a6095a7": {
        "library": "cake_moe_grouped_gemm_3edd5c53a1269a6095a7",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_4b5788aa552dab1c904b": {
        "library": "cake_moe_grouped_gemm_4b5788aa552dab1c904b",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_4f40b5f07baf6e086fdd": {
        "library": "cake_moe_grouped_gemm_4f40b5f07baf6e086fdd",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_5064b19de3716266ed03": {
        "library": "cake_moe_grouped_gemm_5064b19de3716266ed03",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_55833be10c8515562245": {
        "library": "cake_moe_grouped_gemm_55833be10c8515562245",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_57f04c87c1fcb31565c6": {
        "library": "cake_moe_grouped_gemm_57f04c87c1fcb31565c6",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_68b9978a6b64cc618fc3": {
        "library": "cake_moe_grouped_gemm_68b9978a6b64cc618fc3",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_6b2d4157a15035904205": {
        "library": "cake_moe_grouped_gemm_6b2d4157a15035904205",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "sched"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_6bd667b99e236d0c69df": {
        "library": "cake_moe_grouped_gemm_6bd667b99e236d0c69df",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_7e39db00fc7c99693f3e": {
        "library": "cake_moe_grouped_gemm_7e39db00fc7c99693f3e",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_932cb8a0d39a950ac28f": {
        "library": "cake_moe_grouped_gemm_932cb8a0d39a950ac28f",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a"],
    },
    "cake_moe_grouped_gemm_994f65a456ae68a92e84": {
        "library": "cake_moe_grouped_gemm_994f65a456ae68a92e84",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
    },
    "cake_moe_grouped_gemm_9df3c426f0afac83124c": {
        "library": "cake_moe_grouped_gemm_9df3c426f0afac83124c",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_a15e2364075bd5afd8e8": {
        "library": "cake_moe_grouped_gemm_a15e2364075bd5afd8e8",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_aba5ff762eb9355f540a": {
        "library": "cake_moe_grouped_gemm_aba5ff762eb9355f540a",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c": {
        "library": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partials"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["parameter", "num_clusters"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_ce655879551e583539e2": {
        "library": "cake_moe_grouped_gemm_ce655879551e583539e2",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["buffer", "sched"],
            ["parameter", "c0"],
            ["parameter", "c1"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_103a"],
    },
    "cake_moe_grouped_gemm_d8b4bc5530d29550347b": {
        "library": "cake_moe_grouped_gemm_d8b4bc5530d29550347b",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_107a"],
    },
    "cake_moe_grouped_gemm_ed9a74f36ba13a52183d": {
        "library": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_f2b304a9af5613f646f4": {
        "library": "cake_moe_grouped_gemm_f2b304a9af5613f646f4",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "tail_sums"],
            ["buffer", "tail_tickets"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [320, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb": {
        "library": "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb",
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["buffer", "C"],
            ["buffer", "offs"],
            ["buffer", "tensormap_workspace"],
            ["buffer", "partials"],
            ["parameter", "tail_splits"],
            ["parameter", "raster_rows"],
            ["parameter", "num_groups"],
            ["parameter", "sum_m"],
            ["parameter", "N"],
            ["parameter", "K"],
            ["parameter", "ldc"],
            ["parameter", "stride_e"],
            ["buffer", "wave_sync"],
            ["parameter", "wave_mode"],
            ["buffer", "A_rows"],
            ["buffer", "B_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        "arches": ["sm_100a", "sm_103a"],
    },
}
ROUTES: dict[str, dict[str, Any]] = {
    "cake_moe_grouped_gemm_dgrad": {
        "op": "dgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_7e39db00fc7c99693f3e",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_1edcad995dcc790600cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_6bd667b99e236d0c69df",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_aba5ff762eb9355f540a",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_5064b19de3716266ed03",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_gemm_994f65a456ae68a92e84",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_932cb8a0d39a950ac28f",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_fwd": {
        "op": "fwd",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_9df3c426f0afac83124c",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_276cec1f1d2c8ba1e4cb",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_55833be10c8515562245",
                "specializations": {},
            },
            {
                "name": "main_wide",
                "program": "cake_moe_grouped_gemm_3edd5c53a1269a6095a7",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_4f40b5f07baf6e086fdd",
                "specializations": {},
            },
            {
                "name": "main_wide_ts33",
                "program": "cake_moe_grouped_gemm_ce655879551e583539e2",
                "specializations": {},
            },
            {
                "name": "main_wide_ef",
                "program": "cake_moe_grouped_gemm_27a75f32c7a0e39f50f3",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k256": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
                "specializations": {},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_09a336207c15c131f5dc",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb",
                "specializations": {},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_bf16_k512": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_68b9978a6b64cc618fc3",
                "specializations": {},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_57f04c87c1fcb31565c6",
                "specializations": {},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_001f83d4fc1d538b8731",
                "specializations": {},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_gemm_4b5788aa552dab1c904b",
                "specializations": {},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_gemm_d8b4bc5530d29550347b",
                "specializations": {},
            },
            {
                "name": "main_halves_queue",
                "program": "cake_moe_grouped_gemm_2803c50f00dd015bf1d4",
                "specializations": {},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_f32_k256": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_ed9a74f36ba13a52183d",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_09a336207c15c131f5dc",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_fc0a5c2da4d4782f42eb",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "tail_reduce",
                "program": "cake_moe_grouped_gemm_bc774d802e4f0b3ac27c",
                "specializations": {"WGRAD_OUT": 32, "WGRAD_TILE": 256},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
    "cake_moe_grouped_gemm_wgrad_f32_k512": {
        "op": "wgrad",
        "stages": [
            {
                "name": "main",
                "program": "cake_moe_grouped_gemm_68b9978a6b64cc618fc3",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave",
                "program": "cake_moe_grouped_gemm_57f04c87c1fcb31565c6",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves",
                "program": "cake_moe_grouped_gemm_001f83d4fc1d538b8731",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves",
                "program": "cake_moe_grouped_gemm_4b5788aa552dab1c904b",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t",
                "program": "cake_moe_grouped_gemm_d8b4bc5530d29550347b",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_acc",
                "program": "cake_moe_grouped_gemm_12a3b3c91ad6f91315e0",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_wave_halves_acc",
                "program": "cake_moe_grouped_gemm_f2b304a9af5613f646f4",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_k256t_acc",
                "program": "cake_moe_grouped_gemm_a15e2364075bd5afd8e8",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_queue",
                "program": "cake_moe_grouped_gemm_2803c50f00dd015bf1d4",
                "specializations": {"WGRAD_OUT": 32},
            },
            {
                "name": "main_halves_queue_acc",
                "program": "cake_moe_grouped_gemm_6b2d4157a15035904205",
                "specializations": {"WGRAD_OUT": 32},
            },
        ],
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
    },
}
HOST_PLAN_CONSTANTS: dict[str, Any] = {
    "block_n": 256,
    "wide_block_n": 512,
    "wide_fwd_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "wide_dgrad_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "dgrad_evict_first_archs": ["sm_100a", "sm_107a"],
    "dgrad_evict_first_min_rows": {"sm_100a": 4096, "sm_103a": 16384, "sm_107a": 16384},
    "dgrad_evict_first_min_k": 4096,
    "fwd_lwh_ef_archs": ["sm_107a"],
    "fwd_lwh_ef_max_k": 2048,
    "fwd_evict_first_archs": ["sm_100a", "sm_103a"],
    "fwd_evict_first_min_k": 6144,
    "dgrad_wide_ts33_archs": ["sm_103a"],
    "dgrad_wide_ts33_min_rows": {"sm_103a": 8192},
    "dgrad_wide_ts33_min_k": 4096,
    "fwd_wide_ts33_archs": ["sm_103a"],
    "fwd_wide_ts33_max_k": 2048,
    "fwd_wide_ts33_max_rows": 4096,
    "block_k": 64,
    "cta_group": 2,
    "cluster_m": 256,
    "epi_chunk": 16,
    "tmap_slot_bytes": 128,
    "tmap_slots_per_cta": 18,
    "k512_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "k512_release_archs": ["sm_100a", "sm_103a", "sm_107a"],
    "k256_tail_archs": ["sm_107a"],
    "k256_tail_max_k": 2048,
    "k256_tail_min_rows": 8192,
    "k512_min_rows_per_group": {"sm_100a": 3072, "sm_103a": 3072, "sm_107a": 4096},
    "wave_sync_min_k": 4096,
    "wave_sync_archs": {
        256: ["sm_100a", "sm_103a"],
        512: ["sm_100a", "sm_103a"],
    },
    "wave_sync_mode": {"sm_100a": 2, "sm_103a": 2},
    "wave_sync_words": 2,
    "sched_words": 4224,
    "sched_rest": [0, 2147483648, 0],
    "domain_split_archs": ["sm_100a", "sm_103a"],
    "domain_split_ops": ["dgrad"],
    "domain_split_max_weight_elems": 12582912,
    "wgrad_tail_split": 1,
    "tail_plan": {
        "max_splits": 16,
        "sm_clock_ghz": 1.8,
        "dram_tbps": 6.0,
        "launch_us": 5.0,
        "cycles_per_step": {256: 512, 512: 1024},
    },
    # The architectures whose 512-wide k tile routes bind builds that add their
    # split tail in place with unordered reductions
    # (``cake_backend.wgrad_unordered_tail``): none, the registry holds no such
    # build.  An architecture listed here launches an order-varying weight
    # gradient, which alerts under torch's deterministic mode.
    "unordered_tail_archs": [],
    # The architectures whose 512-wide k tile routes bind the ordered one-launch
    # split-tail builds (``cake_backend.wgrad_ordered_tail``): every architecture
    # of the k512 routes.  No architecture may be listed in both lists.
    "ordered_tail_archs": ["sm_100a", "sm_103a", "sm_107a"],
    # The cost model of the one-launch split tail, ordered or unordered
    # (``cake_backend.wgrad_atomic_tail_plan``): the keys of ``tail_plan``.
    "atomic_tail_plan": {
        "max_splits": 16,
        "sm_clock_ghz": 1.8,
        "dram_tbps": 6.0,
        "launch_us": 5.0,
        "cycles_per_step": {256: 512, 512: 1024},
    },
    # Where a 512-wide k tile weight gradient binds the queue walk
    # (``cake_backend.wgrad_queue_walk``): the ``main_halves_queue`` stage, or
    # ``main_halves_queue_acc`` for a fused FP32 accumulate, in place of the
    # stage the host selects.  A call takes it when one rule names the device's
    # architecture and SM count, the selected stage, the kind of output
    # (``"bfloat16"`` / ``"float32"`` overwrite, ``"accumulate"`` for the
    # fused FP32 accumulate) and a range ``[min, max]`` holding the group count;
    # every other call keeps its stage.
    "queue_walk": [
        {
            "arch": "sm_100a",
            "sm_counts": [148],
            "stage": "main_wave_halves",
            "outputs": ["bfloat16", "accumulate"],
            "groups": [4, 32],
        },
        {
            "arch": "sm_100a",
            "sm_counts": [148],
            "stage": "main_halves",
            "outputs": ["bfloat16", "accumulate"],
            "groups": [8, 32],
        },
        {
            "arch": "sm_100a",
            "sm_counts": [152],
            "stage": "main_wave_halves",
            "outputs": ["accumulate"],
            "groups": [4, 32],
        },
        {
            "arch": "sm_100a",
            "sm_counts": [152],
            "stage": "main_halves",
            "outputs": ["bfloat16"],
            "groups": [16, 32],
        },
        {
            "arch": "sm_103a",
            "sm_counts": [152],
            "stage": "main_wave_halves",
            "outputs": ["bfloat16", "accumulate"],
            "groups": [4, 32],
        },
        {
            "arch": "sm_103a",
            "sm_counts": [152],
            "stage": "main_halves",
            "outputs": ["accumulate"],
            "groups": [8, 16],
        },
    ],
    # The u32 words of the queue walk's tile counters (``queue_sched``).
    "queue_sched_words": 2,
    "int32_elems": 2147483647,
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
OPS = ("fwd", "dgrad", "wgrad")
# The name suffix of a route's lead twin (the lead routes, above).
LEAD_SUFFIX = "_lead"


def lead_route(name: str) -> str:
    """The lead twin of route ``name``: the route whose stages bind the ``offs_lead`` twins of its
    programs (``lead_twin``), which the host binds for an ``m_indptr[E + 1]`` row table."""
    if is_lead_route(name):
        raise ValueError(f"route {name!r} is a lead route; it has no lead twin")
    return name + LEAD_SUFFIX


def is_lead_route(name: str) -> bool:
    """True for the name of a lead twin route (``lead_route``)."""
    return name.endswith(LEAD_SUFFIX)


def lead_route_served(name: str, arch: str) -> bool:
    """True when this checkout registers the lead twin of route ``name`` for ``arch``."""
    route = ROUTES.get(lead_route(name))
    return route is not None and arch in route["arches"]


def lead_twin(program: str) -> str:
    """The ``offs_lead`` twin of ``program``: the ``PROGRAMS`` record ``<program>_lead``
    whose ``twin_of`` names ``program`` (``KeyError`` for a program without one)."""
    twin = program + LEAD_SUFFIX
    if PROGRAMS.get(twin, {}).get("twin_of") != program:
        raise KeyError(f"program {program!r} has no lead twin in PROGRAMS")
    return twin


def record_name(
    op: str, *, out_dtype: Optional[str] = None, tile_k: Optional[int] = None
) -> str:
    """Route key of ``op``; weight-gradient routes are further keyed by output dtype and k tile."""
    if op not in OPS:
        raise ValueError(
            f"unknown grouped GEMM operation {op!r}; expected one of {OPS}"
        )
    base = f"cake_moe_grouped_gemm_{op}"
    if op == "wgrad":
        if out_dtype not in ("bfloat16", "float32") or tile_k not in (256, 512):
            raise ValueError(
                "weight-gradient routes are keyed by out_dtype ('bfloat16' / "
                f"'float32') and tile_k (256 / 512); got {out_dtype!r}, {tile_k!r}"
            )
        base += f"_{'f32' if out_dtype == 'float32' else 'bf16'}_k{tile_k}"
    return base


def program_registered(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
    lead: bool = False,
) -> bool:
    """True when this checkout registers a route of ``op`` for ``arch``.

    For the weight gradient with ``tile_k=None`` any k tile of ``out_dtype``
    counts; ``out_dtype=None`` accepts either output dtype.  With ``lead=True`` a
    route counts only when its lead twin (``lead_route``) is registered for
    ``arch`` too: then both row-table forms, ``[E]`` and ``m_indptr[E + 1]``, run.
    """
    if arch not in ARCHES:
        return False
    if op != "wgrad":
        names = [record_name(op)]
    else:
        dtypes = (out_dtype,) if out_dtype else ("bfloat16", "float32")
        tiles = (tile_k,) if tile_k else (256, 512)
        names = [
            record_name(op, out_dtype=dtype, tile_k=tile)
            for dtype in dtypes
            for tile in tiles
        ]
    for name in names:
        route = ROUTES.get(name)
        if route is None or arch not in route["arches"]:
            continue
        if not lead or lead_route_served(name, arch):
            return True
    return False


def select_module(
    op: str,
    arch: str,
    *,
    out_dtype: Optional[str] = None,
    tile_k: Optional[int] = None,
) -> str:
    """Return the registered route name serving ``op`` on ``arch`` or raise."""
    if arch not in ARCHES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHES}")
    name = record_name(op, out_dtype=out_dtype, tile_k=tile_k)
    route = ROUTES.get(name)
    if route is None or arch not in route["arches"]:
        raise NotImplementedError(
            f"The generated ragged grouped GEMM route {name!r} is not registered for "
            f"{arch} in this checkout (the registry of "
            "cake_pplx._backends.cake_moe_grouped_gemm_nondet_train.cake_jit is filled by the "
            "generated-program export)"
        )
    if route["op"] != op:
        raise RuntimeError(
            f"registered route {name!r} is a {route['op']} route, bound to {op}"
        )
    return name


def stage_binding(name: str, stage: str, arch: str) -> dict[str, Any]:
    """The ``{"name", "program", "specializations"}`` record of ``stage`` of route ``name`` on ``arch``."""
    route = ROUTES[name]
    for item in route["stages"]:
        if item["name"] == stage and arch in PROGRAMS[item["program"]]["arches"]:
            return item
    raise KeyError(f"route {name!r} has no {stage!r} stage on {arch}")


def stage_served(name: str, stage: str, arch: str) -> bool:
    """True when route ``name`` has a ``stage`` stage whose program is built for ``arch``."""
    return any(
        item["name"] == stage and arch in PROGRAMS[item["program"]]["arches"]
        for item in ROUTES[name]["stages"]
    )


def _specialization_items(
    specializations: dict[str, Any],
) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(specializations.items()))


def specialization_tag(specializations: tuple[tuple[str, Any], ...]) -> str:
    """Library-name suffix of a compile-line specialization set: ``_wgrad_tile256`` for
    ``{"WGRAD_TILE": 256}``, empty for none.  ``build.json`` names a specialized build
    ``<library><tag>_<arch>`` and carries its ``-D<KEY>=<value>`` defines in that entry's ``nvcc_flags``."""
    return "".join(f"_{key.lower()}{value}" for key, value in specializations)


def library_name(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
) -> str:
    """``build.json`` entry of ``program`` built for ``arch`` under ``specializations``."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not delivered for {arch}")
    tag = specialization_tag(_specialization_items(specializations or {}))
    return f"{record['library']}{tag}_{arch}"


@functools.cache
def _load_module(program: str, arch: str, specializations: tuple[tuple[str, Any], ...]):
    return cake_build.load_library(library_name(program, arch, dict(specializations)))


def load_module(
    program: str, arch: str, specializations: Optional[dict[str, Any]] = None
):
    """Load (once per program, exact target and specialization) one physical program: the prebuilt
    library under ``aot/`` when the package ships it, else the JIT shim's build of its ``build.json``
    entry."""
    return _load_module(program, arch, _specialization_items(specializations or {}))
