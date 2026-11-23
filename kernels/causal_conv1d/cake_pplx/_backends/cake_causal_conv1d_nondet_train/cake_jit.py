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

Host plan and loader glue of the Cake packed width-4 depthwise causal
convolution + SiLU training kernels of the non-deterministic package (SM100 /
SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch
record, grid rule, workspace rule and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it,
and otherwise through the package's JIT (``_cake_pplx_jit``) from the same sources and flags,
verifying every source digest before the first build.

The routes, their selection rules and the forward programs are those of the
deterministic package (``cake_pplx.cake_causal_conv1d_det_train``):
the forward routes ``fwd`` / ``fwd_dt`` / ``fwd_bias`` run this package's
copies of the det forward programs (no cross-CTA reduction, so ``y`` is the
det program's bits), and the backward routes ``bwd`` / ``bwd_dt`` /
``bwd_bias`` / ``bwd_fp32g`` / ``bwd_bias_fp32g``
select, by the same channel
(and token) thresholds, single-launch builds of the det backward bodies,
except at the rows of ``FALLBACK_ROWS`` (:func:`selected_modules`), where a
route runs its fallback program (the single-launch build of the det route's
fallback body).  Inside the token range of a forward row of
``FALLBACK_ROWS``, ``DIRECT_ROWS`` or ``PAIR_ROWS`` a forward call runs the det
static program (each CTA a fixed token range of its channel tile), the det
direct program (each CTA the next token blocks of its channel tile, on the
route's tiles of ``DIRECT_PLANS``) or the det pair direct program (the direct
program with each thread on a pair of channels, on the route's tiles of
``PAIR_PLANS``); none takes ``work``, and ``y`` is the same bits on every
forward program and plan.  Each row gives its own token range; a ``PAIR_ROWS``
range runs ahead of the other two tables where it holds the call.  Every
architecture and route has a row at each C = 512 to 1024.  For most C the first
range starts at S = 1.  Where a key has no ``DIRECT_ROWS`` range (for example
the sm_107a ``fwd_bias`` rows at C = 584, 600, 616 and 632, from S = 1 to
24576), the static program runs over the key's range outside the ``PAIR_ROWS``
range.  The sm_103a ``fwd_bias`` rows at C = 720 to 768 have no ``DIRECT_ROWS``
range: they run the pair direct program from S = 16 to 1536 and the static
program from S = 13824 (C = 736: 14336).  Outside a row's range, and where an
architecture, route and C have no row, a forward call runs the dynamic program,
which takes ``work``, as in 0.2.4.  Inside the token range of a backward row of
``DIRECT_ROWS`` (routes ``bwd``, ``bwd_dt`` and ``bwd_bias``; per architecture
and route, channel ranges within C = 512 to 1024, each with its own token range
within S = 1 to 4096) a backward call runs the route's direct backward program
instead, ahead of ``SM_WIDE_ROWS``, ``FALLBACK_ROWS``, ``TILE_OVERRIDES``,
``NARROW_TILES`` and the thresholds: one ordinary launch without TMA or a
staged pipeline, each CTA the next token blocks of ``n_sub x 4`` tokens of
its channel tile, on the route's tiles of ``BACKWARD_DIRECT_PLANS``.  It has
no det twin; its ``dx`` is bitwise the route's staged programs', and its
``dweight`` / ``dbias`` take the cross-CTA reduction below.  A backward call is
ONE ordinary launch: every per-thread and per-CTA FP32 step and ``dx`` of a
staged program are those of the det program it is built from, and only the
cross-CTA reduction of ``dweight`` / ``dbias``
differs.  Instead of storing one partial row per CTA for a reduce program to
sum in a fixed order, each CTA adds its CTA row into the accumulators (logical
tensor ``acc``, kernel parameter ``dw_acc``) with unordered global reductions
-- FP32 ``red.add`` (``v4``), one plain FP32 word per sum, as the det reduce
sums in FP32 on every route; the ``_fp32g`` routes spread a tile's CTAs over
``slots`` copies of the accumulators so that fewer adds queue on a word -- then
takes a ticket on its channel tile's ``work`` counter; the tile's last CTA
reads the tile's accumulators (with several copies every copy's, added in copy
order in FP32), stores ``dw`` (and ``db``) in the dtypes ``flags`` names, and
stores zeros back.  So ``acc`` and ``work`` are zero before and after every
launch (CUDA Graph replays included), the host never zeroes them again, and
nothing synchronizes with the host.  The summation order of the cross-CTA sum
depends on the CTAs' timing: ``dweight`` / ``dbias`` are not bitwise
reproducible run to run (the only relaxed property); they accumulate in plain
FP32 words, as the det reduction does.

The opt-in pair routes ``bwd_fp32g_pair`` / ``bwd_bias_fp32g_pair``
(:data:`PAIR_ROUTES`, selected only by ``dweight_sum="fp32_pair"``) are the
FP32 hi/lo pair forms of the two ``_fp32g`` routes: the CTA's fold is
Kahan-compensated and every accumulator word is an FP32 (hi, lo) pair (the
CTA's add into hi and its exact rounding error, TwoSum, into lo: FP32
operations only), merged and rounded once by the tile's last CTA.  They
select by the ``_fp32g`` routes' thresholds and plan their launches by those
routes' rules (:func:`record_route`); their accumulators are four pair copies
(``slots`` 8), within the public workspace.

The backward workspace (:func:`accumulator_layout`) is one ``uint8`` buffer:
``acc`` (``slots x taps x C`` FP32 words: ``slots`` copies of the ``[C, 4]``
taps, then ``[C]`` bias on the bias routes) at offset 0 and the uint32
``work`` counters (one per channel tile) at the next 16-byte boundary.  Its size depends on ``C`` and the
route, never on ``S`` or the grid; the public size
(:func:`backward_workspace_bytes`) is the largest over every backward program
of the architecture, so one zeroed buffer serves every route.

The program name of a forward copy is ``cake_causal_conv1d_train_<20 hex>``
(the det program's identity digits under this package's prefix); the backward
programs' names, the direct backward programs' included, are the identity
digests of their own kernel module key (so no backward name equals a name the
det package ships).

Every program but the pair routes' launches with ``launch.use_pdl`` of
``"runtime"``: its binding takes the ``enable_pdl`` launch scalar and sets the
programmatic dependent launch attribute when it is 1, which the host chooses
per architecture and per mode (outside or inside a CUDA Graph capture;
:data:`PDL_EAGER`, :data:`PDL_GRAPH`), except that a backward call at a row
of :data:`PDL_OFF_ROWS` (its route, ``C`` and ``S``) passes 0 in both modes
(:func:`pdl_enabled`).  Each program
waits for the stream's preceding kernel (``griddepcontrol.wait``) before it
reads or writes global memory, so the attribute changes only when a launch may
start, never what it computes.  The det package's forward programs, which this
package's forward copies run, launch without it (its backward reduce programs
set it on every launch); a forward copy differs from its det program in the
kernel symbol, the library name and this launch scalar.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.use_pdl`` is also host plan: the rendered binding takes it from the
# record (``"runtime"``: the ``enable_pdl`` launch scalar, ``("launch",
# "enable_pdl")`` in ``arg_plan``), and a record's ``logical`` names every
# host parameter, ``enable_pdl`` included.
MODULES: dict[str, dict[str, Any]] = {
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_f193c5247980992f4f81`` (the static program: built with ``sched="static"``; the
    # same kernel under this package's prefix, so ``y`` is bitwise the det program's); the route's fallback (no
    # ``select_rule``), which runs at the forward rows of ``FALLBACK_ROWS``.  Each CTA owns a fixed token range of
    # its channel tile, so it takes no ``work`` counters (``workspace_rule`` None).
    "cake_causal_conv1d_train_f193c5247980992f4f81": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_f193c5247980992f4f81",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f193c5247980992f4f81",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_f193c5247980992f4f81",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": None,
    },
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_f5497330bfd286211643`` (built with ``fast=True, sched="dynamic",
    # chunk_stages=1, pair_math=True``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's); it runs every call of at least 8 channels (``channels_at_least`` 8 on every architecture) that no
    # row of ``PAIR_ROWS``, ``DIRECT_ROWS`` or ``FALLBACK_ROWS`` and no record of a higher threshold takes.  ``work``
    # holds the int32 token-chunk counters (two per channel tile), zero before the first launch and left zero by
    # every launch.
    "cake_causal_conv1d_train_f5497330bfd286211643": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_f5497330bfd286211643",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f5497330bfd286211643",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_f5497330bfd286211643",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_100a": 8,
                "sm_103a": 8,
                "sm_107a": 8,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_a81632301ba92a887c46`` (the direct program: built with ``sched="direct",
    # pair_math=True``; the same kernel under this package's prefix, so ``y`` is bitwise the det program's).  Each
    # CTA takes the next token blocks of ``n_sub x 8`` tokens of its channel tile, with no shared memory and no
    # ``work`` counters (4 CTAs per SM).  It runs only at its rows of ``DIRECT_ROWS`` (``select_rule`` kind
    # ``rows``), on the route's tiles of ``DIRECT_PLANS``, and launches at most one CTA per token block (grid cap
    # ``token_blocks``).
    "cake_causal_conv1d_train_a81632301ba92a887c46": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_a81632301ba92a887c46",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a81632301ba92a887c46",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_a81632301ba92a887c46",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": None,
    },
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_d0b237e92b28348315d0`` (the pair direct program: built with ``sched="direct",
    # pair_math=True, lane_ch=2``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's).  The direct program with each thread on a pair of channels (4-byte row loads), so its geometry's
    # ``vec`` is 2 and ``ct_vec`` counts channel pairs (7 CTAs per SM).  It runs only at its rows of ``PAIR_ROWS``
    # (``select_rule`` kind ``rows``), on the route's tiles of ``PAIR_PLANS``, and launches at most one CTA per token
    # block (grid cap ``token_blocks``).
    "cake_causal_conv1d_train_d0b237e92b28348315d0": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 7,
            "geometry": {
                "threads": 128,
                "vec": 2,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_d0b237e92b28348315d0",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d0b237e92b28348315d0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_d0b237e92b28348315d0",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": None,
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_9bfa65707a49df6a48ce`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host
    # synchronization, no cooperative launch.  On sm_100a / sm_103a it also runs at the rows of ``FALLBACK_ROWS``
    # (C = 520 to 1016 up to a per-C token bound, above the route's ``DIRECT_ROWS`` range), in place of the program
    # the ``bwd`` thresholds pick there.
    "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_b4f7ff5a3dd9ddd2f41c`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host
    # synchronization, no cooperative launch.
    "cake_causal_conv1d_train_a79ed3e33712abf525ad": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_a79ed3e33712abf525ad",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_9a07bcaa0223632640ee`` (the same build keywords, body and grid; selected
    # from C = 256 on sm_107a and S >= 262144 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx``
    # are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the
    # FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce
    # program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_5ed71ee5908c07376183": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_5ed71ee5908c07376183",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_5ed71ee5908c07376183",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_5ed71ee5908c07376183",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 256,
            },
            "kind": "channels_tokens_at_least",
            "tokens": {
                "sm_107a": 262144,
            },
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_37e428406a0cc4a8ca1b`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 768 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and
    # ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into
    # the FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce
    # program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_461d358d7090642c65de": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_461d358d7090642c65de",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_461d358d7090642c65de",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_461d358d7090642c65de",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 768,
                "sm_107a": 1032,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    "cake_causal_conv1d_train_06cfd917a14c30ce8ebc": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_06cfd917a14c30ce8ebc",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_06cfd917a14c30ce8ebc",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_06cfd917a14c30ce8ebc",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the channel-band copy of the route's sm_107a fallback program
    # ``cake_causal_conv1d_train_a79ed3e33712abf525ad`` (the same build keywords and kernel with ``c_band=520``,
    # under its own name), selected from C = 520 on sm_107a in place of
    # ``cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab``, which keeps 512 <= C < 520 there.  ``dx`` and every
    # per-CTA FP32 step are the fallback's; ``acc`` / ``work`` are zero before and after every launch.
    "cake_causal_conv1d_train_e5d6d187103ab6e0cbbb": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_e5d6d187103ab6e0cbbb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_e5d6d187103ab6e0cbbb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_e5d6d187103ab6e0cbbb",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 520,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_d46036aa2bc5f90bb0f6`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_103a, 512 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise
    # the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32
    # accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce
    # program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_103a": 512,
                "sm_107a": 512,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_aeff6106e2822f313a0c`` (the same build keywords, body and grid; selected
    # from C = 2048 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator
    # ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``),
    # and the tile's last CTA stores FP32 ``dw`` once (one rounding from FP32) and zeroes its accumulators and
    # counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host
    # synchronization, no cooperative launch.
    "cake_causal_conv1d_train_c798878a383c652c428d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_c798878a383c652c428d",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_c798878a383c652c428d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_c798878a383c652c428d",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 2048,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_0f3982a5518f5a8696f6`` (the static program: built with ``dtype_flags=True,
    # sched="static"``; the same kernel under this package's prefix, so ``y`` is bitwise the det program's); the
    # route's fallback (no ``select_rule``), which runs at the forward rows of ``FALLBACK_ROWS``.  Each CTA owns a
    # fixed token range of its channel tile, so it takes no ``work`` counters (``workspace_rule`` None).
    "cake_causal_conv1d_train_0f3982a5518f5a8696f6": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_0f3982a5518f5a8696f6",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_0f3982a5518f5a8696f6",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_0f3982a5518f5a8696f6",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": None,
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_9c1e7ee9f37e3102660e`` (built with ``dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=1, pair_math=True``; the same kernel under this package's prefix, so ``y`` is
    # bitwise the det program's); it runs every call of at least 8 channels (``channels_at_least`` 8 on every
    # architecture) that no row of ``PAIR_ROWS``, ``DIRECT_ROWS`` or ``FALLBACK_ROWS`` and no record of a higher
    # threshold takes.  ``work`` holds the int32 token-chunk counters (two per channel tile), zero before the first
    # launch and left zero by every launch.
    "cake_causal_conv1d_train_9c1e7ee9f37e3102660e": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_100a": 8,
                "sm_103a": 8,
                "sm_107a": 8,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_21b9be3d1093abe38406`` (the direct program: built with ``dtype_flags=True,
    # sched="direct", pair_math=True``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's).  Each CTA takes the next token blocks of ``n_sub x 8`` tokens of its channel tile, with no shared
    # memory and no ``work`` counters (4 CTAs per SM).  It runs only at its rows of ``DIRECT_ROWS`` (``select_rule``
    # kind ``rows``), on the route's tiles of ``DIRECT_PLANS``, and launches at most one CTA per token block (grid
    # cap ``token_blocks``).
    "cake_causal_conv1d_train_21b9be3d1093abe38406": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_21b9be3d1093abe38406",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_21b9be3d1093abe38406",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_21b9be3d1093abe38406",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": None,
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_96bf2b8aa1132598f130`` (the pair direct program: built with ``dtype_flags=True,
    # sched="direct", pair_math=True, lane_ch=2``; the same kernel under this package's prefix, so ``y`` is bitwise
    # the det program's).  The ``fwd`` pair direct program's form with the ``flags`` scalar (7 CTAs per SM).  It runs
    # only at its rows of ``PAIR_ROWS`` (``select_rule`` kind ``rows``), on the route's tiles of ``PAIR_PLANS``, and
    # launches at most one CTA per token block (grid cap ``token_blocks``).
    "cake_causal_conv1d_train_96bf2b8aa1132598f130": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 7,
            "geometry": {
                "threads": 128,
                "vec": 2,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_96bf2b8aa1132598f130",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": None,
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_62cb38eb8aa667d46bc1`` (the static program: built with ``bias=True,
    # dtype_flags=True, sched="static"``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's); the route's fallback (no ``select_rule``), which runs at the forward rows of ``FALLBACK_ROWS``.
    # Each CTA owns a fixed token range of its channel tile, so it takes no ``work`` counters (``workspace_rule``
    # None).
    "cake_causal_conv1d_train_62cb38eb8aa667d46bc1": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_62cb38eb8aa667d46bc1",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_62cb38eb8aa667d46bc1",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_62cb38eb8aa667d46bc1",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": None,
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_08880ca12e4064bc8cdb`` (built with ``bias=True, dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=1, pair_math=True``; the same kernel under this package's prefix, so ``y`` is
    # bitwise the det program's); it runs every call of at least 8 channels (``channels_at_least`` 8 on every
    # architecture) that no row of ``PAIR_ROWS``, ``DIRECT_ROWS`` or ``FALLBACK_ROWS`` and no record of a higher
    # threshold takes.  ``work`` holds the int32 token-chunk counters (two per channel tile), zero before the first
    # launch and left zero by every launch.
    "cake_causal_conv1d_train_08880ca12e4064bc8cdb": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 46208,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_100a": 8,
                "sm_103a": 8,
                "sm_107a": 8,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_8ed855b04178bc5d90f9`` (the direct program: built with ``bias=True,
    # dtype_flags=True, sched="direct", pair_math=True, min_blocks=3``; the same kernel under this package's prefix,
    # so ``y`` is bitwise the det program's).  Each CTA takes the next token blocks of ``n_sub x 8`` tokens of its
    # channel tile, with no shared memory and no ``work`` counters (3 CTAs per SM).  It runs only at its rows of
    # ``DIRECT_ROWS`` (``select_rule`` kind ``rows``), on the route's tiles of ``DIRECT_PLANS``, and launches at
    # most one CTA per token block (grid cap ``token_blocks``).
    "cake_causal_conv1d_train_8ed855b04178bc5d90f9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": None,
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_6cc04dd65e493836b918`` (the pair direct program: built with ``bias=True,
    # dtype_flags=True, sched="direct", pair_math=True, lane_ch=2``; the same kernel under this package's prefix, so
    # ``y`` is bitwise the det program's).  The ``fwd`` pair direct program's form with the bias and the ``flags``
    # scalar (6 CTAs per SM, as the det record).  It runs only at its rows of ``PAIR_ROWS`` (``select_rule`` kind
    # ``rows``), on the route's tiles of ``PAIR_PLANS``, and launches at most one CTA per token block (grid cap
    # ``token_blocks``).
    "cake_causal_conv1d_train_6cc04dd65e493836b918": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 6,
            "geometry": {
                "threads": 128,
                "vec": 2,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_6cc04dd65e493836b918",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_6cc04dd65e493836b918",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_6cc04dd65e493836b918",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": None,
    },
    # Route ``fwd``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_8665d0756b8240ecb4d0`` (built with ``fast=True, sched="dynamic",
    # chunk_stages=2, pair_math=True``; the same kernel under this package's prefix, so ``y`` is bitwise the det
    # program's); selected from C = 256 and S = 65536 on sm_107a, and its channel band's copy from C = 2048 at every
    # S.  ``work`` holds the int32 token-chunk counters (two per channel tile), zero before the first
    # launch and left zero by every launch.
    "cake_causal_conv1d_train_8665d0756b8240ecb4d0": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 256,
            },
            "kind": "channels_tokens_at_least",
            "tokens": {
                "sm_107a": 65536,
            },
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_dt``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_176ebb3e14f7ad9c94dc`` (built with ``dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=2, pair_math=True``; the same kernel under this package's prefix, so ``y`` is
    # bitwise the det program's); selected from C = 512 and S = 65536 on sm_107a, and its channel band's copy from C
    # = 2048 at every S.  ``work`` holds the int32 token-chunk counters (two per channel tile), zero
    # before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 512,
            },
            "kind": "channels_tokens_at_least",
            "tokens": {
                "sm_107a": 65536,
            },
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``fwd_bias``: this package's copy of the det forward program
    # ``cake_causal_conv1d_det_train_d829c95e1d610bc14b23`` (built with ``bias=True, dtype_flags=True, fast=True,
    # sched="dynamic", chunk_stages=2, pair_math=True``; the same kernel under this package's prefix, so ``y`` is
    # bitwise the det program's); selected from C = 1024 and S = 65536 on sm_107a, and its channel band's copy from
    # C = 2048 at every S.  ``work`` holds the int32 token-chunk counters (two per channel tile), zero
    # before the first launch and left zero by every launch.
    "cake_causal_conv1d_train_d829c95e1d610bc14b23": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_d829c95e1d610bc14b23",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 1024,
            },
            "kind": "channels_tokens_at_least",
            "tokens": {
                "sm_107a": 65536,
            },
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # This package's copy of the det forward channel-band program: the forward
    # with fast, sched dynamic, chunk_stages 2, pair_math, c_band 2048
    # (4 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_d00a2892aa363564c673": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_d00a2892aa363564c673",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d00a2892aa363564c673",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_d00a2892aa363564c673",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 2048,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # This package's copy of the det forward channel-band program: the forward
    # with dtype_flags, fast, sched dynamic, chunk_stages 2, pair_math, c_band
    # 2048
    # (4 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_6211cda590057b12f1a2": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_6211cda590057b12f1a2",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_6211cda590057b12f1a2",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_6211cda590057b12f1a2",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 2048,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # This package's copy of the det forward channel-band program: the forward
    # with bias, dtype_flags, fast, sched dynamic, chunk_stages 2, pair_math,
    # c_band 2048
    # (4 CTAs per SM); it runs when C >= select_rule.channels[arch] and no
    # routed record of a higher threshold admits C; below every threshold the
    # fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_77af6383ff00b1cd047c": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["buffer", "b"],
            ["buffer", "work"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_77af6383ff00b1cd047c",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_77af6383ff00b1cd047c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33920,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_77af6383ff00b1cd047c",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "select_rule": {
            "channels": {
                "sm_107a": 2048,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "tile_counters",
            "words_per_tile": 2,
        },
    },
    # Route ``bwd_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_e31ee9ad746b7101fdca`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into one of the ``slots`` FP32 copies
    # of the accumulator ``acc`` (``[C x 4]`` each), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA adds the copies in copy order in FP32 and stores ``dw`` (and ``db``)
    # in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc``
    # / ``work`` are zero before and after every launch.  No reduce program, no host synchronization, no cooperative
    # launch.
    "cake_causal_conv1d_train_e3ff1615646de68a1aab": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_e3ff1615646de68a1aab",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_e3ff1615646de68a1aab",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_e3ff1615646de68a1aab",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_f5bf4ecc39d89fc55c1c`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into one of the ``slots`` FP32 copies
    # of the accumulator ``acc`` (``[C x 4]`` each), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA adds the copies in copy order in FP32 and stores ``dw`` (and ``db``)
    # in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc``
    # / ``work`` are zero before and after every launch.  No reduce program, no host synchronization, no cooperative
    # launch.
    "cake_causal_conv1d_train_c5abee01ba7440dd50b8": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_c5abee01ba7440dd50b8",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_c5abee01ba7440dd50b8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_c5abee01ba7440dd50b8",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_d9bf04014f9741f50f41`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into one of the ``slots`` FP32 copies
    # of the accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias each), takes a ticket on its channel tile's
    # ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA adds the copies in copy order in FP32 and
    # stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no
    # host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_8433ef0894e75c192d4b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_8433ef0894e75c192d4b",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_8433ef0894e75c192d4b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_8433ef0894e75c192d4b",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_9da2f80d080a8f4e7f3c`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into one of the ``slots`` FP32 copies
    # of the accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias each), takes a ticket on its channel tile's
    # ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA adds the copies in copy order in FP32 and
    # stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no
    # host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_286b95d7960d16de0849": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_286b95d7960d16de0849",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_286b95d7960d16de0849",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_286b95d7960d16de0849",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g_pair`` (opt-in; the route's fallback, sm_100a / sm_103a): the ``bwd_fp32g`` fallback
    # ``cake_causal_conv1d_train_e3ff1615646de68a1aab`` built with ``fold_kahan=True, acc_pairs=True``.
    # The CTA's two-level fold is Kahan-compensated (``fold_kahan``) and each accumulator word is an FP32
    # (hi, lo) pair (``acc_pairs``): a CTA adds its value into hi by an ``atomicAdd`` that returns the hi it
    # added to, and that add's exact rounding error (TwoSum, FP32 operations only) into lo, so hi + lo is the
    # exact sum of the rows up to lo's own rounding, whatever the order of the CTAs.  Four pair copies of
    # ``acc`` (``slots`` 8: eight FP32 words per sum); the tile's last CTA merges the copies' pairs in copy
    # order, rounds each sum once into ``dw`` (and ``db``) and zeroes them.  ``dx`` is the twin's.  The
    # exporter fills the derived leaves (shared memory included).
    "cake_causal_conv1d_train_9181be685ac105263751": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_9181be685ac105263751",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_9181be685ac105263751",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_9181be685ac105263751",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 8,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g_pair`` (opt-in; the route's fallback, sm_107a): the sm_107a ``bwd_fp32g`` fallback
    # ``cake_causal_conv1d_train_c5abee01ba7440dd50b8`` built with ``fold_kahan=True, acc_pairs=True``.
    # The CTA's two-level fold is Kahan-compensated (``fold_kahan``) and each accumulator word is an FP32
    # (hi, lo) pair (``acc_pairs``): a CTA adds its value into hi by an ``atomicAdd`` that returns the hi it
    # added to, and that add's exact rounding error (TwoSum, FP32 operations only) into lo, so hi + lo is the
    # exact sum of the rows up to lo's own rounding, whatever the order of the CTAs.  Four pair copies of
    # ``acc`` (``slots`` 8: eight FP32 words per sum); the tile's last CTA merges the copies' pairs in copy
    # order, rounds each sum once into ``dw`` (and ``db``) and zeroes them.  ``dx`` is the twin's.  The
    # exporter fills the derived leaves (shared memory included).
    "cake_causal_conv1d_train_4685f66f3b7ef380e22a": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_4685f66f3b7ef380e22a",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_4685f66f3b7ef380e22a",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_4685f66f3b7ef380e22a",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 8,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g_pair`` (opt-in; the route's fallback, sm_100a / sm_103a): the ``bwd_bias_fp32g``
    # fallback ``cake_causal_conv1d_train_8433ef0894e75c192d4b`` built with ``fold_kahan=True, acc_pairs=True``.
    # The CTA's two-level fold is Kahan-compensated (``fold_kahan``) and each accumulator word is an FP32
    # (hi, lo) pair (``acc_pairs``): a CTA adds its value into hi by an ``atomicAdd`` that returns the hi it
    # added to, and that add's exact rounding error (TwoSum, FP32 operations only) into lo, so hi + lo is the
    # exact sum of the rows up to lo's own rounding, whatever the order of the CTAs.  Four pair copies of
    # ``acc`` (``slots`` 8: eight FP32 words per sum); the tile's last CTA merges the copies' pairs in copy
    # order, rounds each sum once into ``dw`` (and ``db``) and zeroes them.  ``dx`` is the twin's.  The
    # exporter fills the derived leaves (shared memory included).
    "cake_causal_conv1d_train_b42a8b7cf83f2b508d99": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_b42a8b7cf83f2b508d99",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_b42a8b7cf83f2b508d99",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_b42a8b7cf83f2b508d99",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 8,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g_pair`` (opt-in; the route's fallback, sm_107a): the sm_107a ``bwd_bias_fp32g``
    # fallback ``cake_causal_conv1d_train_286b95d7960d16de0849`` built with ``fold_kahan=True, acc_pairs=True``.
    # The CTA's two-level fold is Kahan-compensated (``fold_kahan``) and each accumulator word is an FP32
    # (hi, lo) pair (``acc_pairs``): a CTA adds its value into hi by an ``atomicAdd`` that returns the hi it
    # added to, and that add's exact rounding error (TwoSum, FP32 operations only) into lo, so hi + lo is the
    # exact sum of the rows up to lo's own rounding, whatever the order of the CTAs.  Four pair copies of
    # ``acc`` (``slots`` 8: eight FP32 words per sum); the tile's last CTA merges the copies' pairs in copy
    # order, rounds each sum once into ``dw`` (and ``db``) and zeroes them.  ``dx`` is the twin's.  The
    # exporter fills the derived leaves (shared memory included).
    "cake_causal_conv1d_train_1212581b2f85eebc7d7f": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_1212581b2f85eebc7d7f",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_1212581b2f85eebc7d7f",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 74752,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_1212581b2f85eebc7d7f",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 8,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_d1ba933488aef5d736cc`` (the same build keywords, body and grid; selected from C =
    # 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise
    # the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into one of the ``slots``
    # FP32 copies of the accumulator ``acc`` (``[C x 4]`` each), takes a ticket on its channel tile's ``work`` counter
    # (uint32 ``[n_ch_tiles]``), and the tile's last CTA adds the copies in copy order in FP32 and stores ``dw`` (and
    # ``db``) in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its accumulators and counter, so
    # ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host synchronization, no
    # cooperative launch.
    "cake_causal_conv1d_train_41fcd5b4c3749107aba7": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_41fcd5b4c3749107aba7",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_41fcd5b4c3749107aba7",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_41fcd5b4c3749107aba7",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1032,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    "cake_causal_conv1d_train_3fdd49d8bc8a1048df1d": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_3fdd49d8bc8a1048df1d",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_3fdd49d8bc8a1048df1d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_3fdd49d8bc8a1048df1d",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_b2c01f2bee8bcdaec6f8`` (the same build keywords, body and grid; selected from C =
    # 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise
    # the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into one of the ``slots``
    # FP32 copies of the accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias each), takes a ticket on its channel
    # tile's ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA adds the copies in copy order in FP32
    # and stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its
    # accumulators and counter, so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no
    # host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_7dd30c23ccab6ab97182": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_7dd30c23ccab6ab97182",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_7dd30c23ccab6ab97182",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 87040,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_7dd30c23ccab6ab97182",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 16,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_fp32g_pair`` (opt-in; the channel-ruled program): ``cake_causal_conv1d_train_41fcd5b4c3749107aba7``
    # built with ``fold_kahan=True, acc_pairs=True``, selected where the ``bwd_fp32g`` route runs that program or
    # its sm_107a channel band (from C = 1024 on sm_107a).
    # The CTA's two-level fold is Kahan-compensated (``fold_kahan``) and each accumulator word is an FP32
    # (hi, lo) pair (``acc_pairs``): a CTA adds its value into hi by an ``atomicAdd`` that returns the hi it
    # added to, and that add's exact rounding error (TwoSum, FP32 operations only) into lo, so hi + lo is the
    # exact sum of the rows up to lo's own rounding, whatever the order of the CTAs.  Four pair copies of
    # ``acc`` (``slots`` 8: eight FP32 words per sum); the tile's last CTA merges the copies' pairs in copy
    # order, rounds each sum once into ``dw`` (and ``db``) and zeroes them.  ``dx`` is the twin's.  The
    # exporter fills the derived leaves (shared memory included).
    "cake_causal_conv1d_train_6857ed58b2a273da419c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_6857ed58b2a273da419c",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_6857ed58b2a273da419c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 99328,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_6857ed58b2a273da419c",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 8,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias_fp32g_pair`` (opt-in; the channel-ruled program):
    # ``cake_causal_conv1d_train_7dd30c23ccab6ab97182`` built with ``fold_kahan=True, acc_pairs=True``, selected
    # where the ``bwd_bias_fp32g`` route runs that program.
    # The CTA's two-level fold is Kahan-compensated (``fold_kahan``) and each accumulator word is an FP32
    # (hi, lo) pair (``acc_pairs``): a CTA adds its value into hi by an ``atomicAdd`` that returns the hi it
    # added to, and that add's exact rounding error (TwoSum, FP32 operations only) into lo, so hi + lo is the
    # exact sum of the rows up to lo's own rounding, whatever the order of the CTAs.  Four pair copies of
    # ``acc`` (``slots`` 8: eight FP32 words per sum); the tile's last CTA merges the copies' pairs in copy
    # order, rounds each sum once into ``dw`` (and ``db``) and zeroes them.  ``dx`` is the twin's.  The
    # exporter fills the derived leaves (shared memory included).
    "cake_causal_conv1d_train_7c3318fb3f7e750cb63b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_7c3318fb3f7e750cb63b",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_7c3318fb3f7e750cb63b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 107520,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_7c3318fb3f7e750cb63b",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 8,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_8f872e5b653a26ab365d`` (the same build keywords, body and grid; selected
    # from C = 264 on sm_100a and sm_103a: the single-tile channel counts 264 <= C < 512, whose channel vectors
    # are no multiple of 32, copy each sub-tile's rows in one bulk copy per row group).  Every per-thread and
    # per-CTA FP32 step and ``dx`` are bitwise the det program's; only the cross-CTA reduction differs: each CTA
    # adds its CTA row into the FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's
    # ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores FP32 ``dw`` once (one rounding
    # from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after every
    # launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_e8e950fb254d3376db52": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_e8e950fb254d3376db52",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_e8e950fb254d3376db52",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 66560,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_e8e950fb254d3376db52",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 264,
                "sm_103a": 264,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``, sm_100a only: the SM-wide build of the det program
    # ``cake_causal_conv1d_det_train_e057357c655024d4ff1d`` (its row-group body and build keywords: one CTA of 384
    # threads per SM, the single channel tile of the row, 5 token sub-tiles per CTA) with ``swizzled=True``: the
    # swizzled flush and the batched finish.  ``dx`` and every per-thread and per-CTA FP32 step are the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``,
    # takes a ticket on the tile's ``work`` counter, and the last CTA stores FP32 ``dw`` once and zeroes ``acc`` and
    # ``work``.  No reduce program, no cooperative launch.  It runs only at its rows of ``SM_WIDE_ROWS`` (C = 520 to
    # 608 at S = 24576 to 98303; ``select_rule`` kind ``rows``).
    "cake_causal_conv1d_train_82e874f30406be2334b6": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 384,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_82e874f30406be2334b6",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_82e874f30406be2334b6",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_82e874f30406be2334b6",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``, sm_100a only: the SM-wide build of the det program
    # ``cake_causal_conv1d_det_train_e057357c655024d4ff1d`` (its row-group body and build keywords: one CTA of 384
    # threads per SM, the single channel tile of the row, 5 token sub-tiles per CTA) with the rows flush and the
    # plain finish.  ``dx`` and every per-thread and per-CTA FP32 step are the det program's; only the cross-CTA
    # reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``, takes a ticket on the tile's
    # ``work`` counter, and the last CTA stores FP32 ``dw`` once and zeroes ``acc`` and ``work``.  No reduce
    # program, no cooperative launch.  It runs only at its rows of ``SM_WIDE_ROWS`` (C = 520 to 608 from S = 98304;
    # ``select_rule`` kind ``rows``).
    "cake_causal_conv1d_train_1ec222b2d908a7d6ea21": {
        "arches": [
            "sm_100a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 384,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_1ec222b2d908a7d6ea21",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_1ec222b2d908a7d6ea21",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_1ec222b2d908a7d6ea21",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``, sm_103a only: the SM-wide build of the det program
    # ``cake_causal_conv1d_det_train_baca4c969c783d07d81f`` (its row-group body and build keywords: one CTA of 384
    # threads per SM, the single channel tile of the row, 5 token sub-tiles per CTA) with ``swizzled=True``: the
    # swizzled flush and the batched finish.  ``dx`` and every per-thread and per-CTA FP32 step are the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``,
    # takes a ticket on the tile's ``work`` counter, and the last CTA stores FP32 ``dw`` once and zeroes ``acc`` and
    # ``work``.  No reduce program, no cooperative launch.  It runs only at its rows of ``SM_WIDE_ROWS`` (C = 520 to
    # 608 at S = 20480 to 31291; ``select_rule`` kind ``rows``).
    "cake_causal_conv1d_train_77692909592ff14be82f": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 384,
                "vec": 8,
            },
            "grid_mult": {
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_77692909592ff14be82f",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_77692909592ff14be82f",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_77692909592ff14be82f",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``, sm_103a only: the SM-wide build of the det program
    # ``cake_causal_conv1d_det_train_baca4c969c783d07d81f`` (its row-group body and build keywords: one CTA of 384
    # threads per SM, the single channel tile of the row, 5 token sub-tiles per CTA) with the rows flush and the
    # plain finish.  ``dx`` and every per-thread and per-CTA FP32 step are the det program's; only the cross-CTA
    # reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``, takes a ticket on the tile's
    # ``work`` counter, and the last CTA stores FP32 ``dw`` once and zeroes ``acc`` and ``work``.  No reduce
    # program, no cooperative launch.  It runs only at its rows of ``SM_WIDE_ROWS`` (C = 520 to 608 from S = 31292;
    # ``select_rule`` kind ``rows``).
    "cake_causal_conv1d_train_ccd45ec8667831b67344": {
        "arches": [
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 384,
                "vec": 8,
            },
            "grid_mult": {
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_ccd45ec8667831b67344",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_ccd45ec8667831b67344",
        "launch": {
            "block": (384, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 197632,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_ccd45ec8667831b67344",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``, sm_107a only: the SM-wide build of the det program
    # ``cake_causal_conv1d_det_train_5ca7515dd2118dd96241`` (its row-group body and build keywords: one CTA of 256
    # threads per SM, the single channel tile of the row, 3 token sub-tiles per CTA) with ``swizzled=True``: the
    # swizzled flush and the batched finish.  ``dx`` and every per-thread and per-CTA FP32 step are the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``,
    # takes a ticket on the tile's ``work`` counter, and the last CTA stores FP32 ``dw`` once and zeroes ``acc`` and
    # ``work``.  No reduce program, no cooperative launch.  It runs only at its rows of ``SM_WIDE_ROWS`` (C = 520 to
    # 680 at S = 32768 to 131071; ``select_rule`` kind ``rows``).
    "cake_causal_conv1d_train_175b82ccd2309f434dfb": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 256,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_175b82ccd2309f434dfb",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_175b82ccd2309f434dfb",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 132096,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_175b82ccd2309f434dfb",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd``, sm_107a only: the SM-wide build of the det program
    # ``cake_causal_conv1d_det_train_5ca7515dd2118dd96241`` (its row-group body and build keywords: one CTA of 256
    # threads per SM, the single channel tile of the row, 3 token sub-tiles per CTA) with ``swizzled=True,
    # acc_slots=2``: the swizzled flush and the batched finish over two accumulator copies (``slots`` 2; the CTAs
    # spread their adds over both).  ``dx`` and every per-thread and per-CTA FP32 step are the det program's; only
    # the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``, takes a ticket
    # on the tile's ``work`` counter, and the last CTA stores FP32 ``dw`` once and zeroes ``acc`` and ``work``.  No
    # reduce program, no cooperative launch.  It runs only at its rows of ``SM_WIDE_ROWS`` (C = 520 to 680 from S =
    # 131072; ``select_rule`` kind ``rows``).
    "cake_causal_conv1d_train_3293c0e34e70413fb4af": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 1,
            "geometry": {
                "threads": 256,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_3293c0e34e70413fb4af",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_3293c0e34e70413fb4af",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 132096,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_3293c0e34e70413fb4af",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 2,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_dt``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_af4c42c4bc5c9f00be94`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)), built with ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with
    # FP32 and with BF16 taps (``flags`` names the taps dtype).  Every per-thread and per-CTA FP32 step and ``dx``
    # are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32
    # accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one
    # rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after
    # every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_7035e23df8c9bf3c11e8": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_7035e23df8c9bf3c11e8",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_7035e23df8c9bf3c11e8",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_7035e23df8c9bf3c11e8",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_dt``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_ff7637fef45fa62da9d5`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)), built with ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with
    # FP32 and with BF16 taps (``flags`` names the taps dtype).  Every per-thread and per-CTA FP32 step and ``dx``
    # are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32
    # accumulator ``acc`` (``[C x 4]``), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one
    # rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after
    # every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_d3273135fc61ea150378": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_d3273135fc61ea150378",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_d3273135fc61ea150378",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 50176,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_d3273135fc61ea150378",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_dt``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_e5c5595c9beb345803b0`` (the same build keywords, body and grid; selected
    # from C = 512 on sm_100a, 512 on sm_103a, 1024 on sm_107a), built with ``exact_pre=True``: SiLU' at the
    # unrounded FP32 preactivation, with FP32 and with BF16 taps (``flags`` names the taps dtype).  Every
    # per-thread and per-CTA FP32 step and ``dx`` are bitwise the det program's; only the cross-CTA reduction
    # differs: each CTA adds its CTA row into the FP32 accumulator ``acc`` (``[C x 4]``), takes a ticket on its
    # channel tile's ``work`` counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and
    # ``db``) in the dtypes ``flags`` names once (one rounding from FP32) and zeroes its accumulators and counter,
    # so ``acc`` / ``work`` are zero before and after every launch.  No reduce program, no host synchronization,
    # no cooperative launch.
    "cake_causal_conv1d_train_092af77575f0fdbf3615": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_092af77575f0fdbf3615",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_092af77575f0fdbf3615",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 82944,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_092af77575f0fdbf3615",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_d8ce3d764c386135943a`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)), built with ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with
    # FP32 and with BF16 taps (``flags`` names the taps dtype).  Every per-thread and per-CTA FP32 step and ``dx``
    # are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32
    # accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work``
    # counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags``
    # names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero
    # before and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_93f7280de3dcb9ad7f49": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_93f7280de3dcb9ad7f49",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_93f7280de3dcb9ad7f49",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_93f7280de3dcb9ad7f49",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_4bf88567b50bfe45e241`` (the same build keywords, body and grid; the route's
    # fallback (no ``select_rule``)), built with ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with
    # FP32 and with BF16 taps (``flags`` names the taps dtype).  Every per-thread and per-CTA FP32 step and ``dx``
    # are bitwise the det program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32
    # accumulator ``acc`` (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work``
    # counter (uint32 ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags``
    # names once (one rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero
    # before and after every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_6a8201512a0992310513": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_6a8201512a0992310513",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_6a8201512a0992310513",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 54272,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_6a8201512a0992310513",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the single-launch build of the det program
    # ``cake_causal_conv1d_det_train_03799186845e6a7d8296`` (the same build keywords, body and grid; selected from
    # C = 512 on sm_100a and sm_103a and from C = 1032 on sm_107a, where the channel-band program below takes C =
    # 1024), built with ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with FP32 and with BF16 taps
    # (``flags`` names the taps dtype).  Every per-thread and per-CTA FP32 step and ``dx`` are bitwise the det
    # program's; only the cross-CTA reduction differs: each CTA adds its CTA row into the FP32 accumulator ``acc``
    # (``[C x 4]`` taps then ``[C]`` bias), takes a ticket on its channel tile's ``work`` counter (uint32
    # ``[n_ch_tiles]``), and the tile's last CTA stores ``dw`` (and ``db``) in the dtypes ``flags`` names once (one
    # rounding from FP32) and zeroes its accumulators and counter, so ``acc`` / ``work`` are zero before and after
    # every launch.  No reduce program, no host synchronization, no cooperative launch.
    "cake_causal_conv1d_train_92aeee41bc2e2d394937": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_100a": 1,
                "sm_103a": 1,
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_92aeee41bc2e2d394937",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_92aeee41bc2e2d394937",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 87040,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_92aeee41bc2e2d394937",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1032,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: sm_107a's channel-band program (selected from C = 1024), built with ``exact_pre=True``:
    # SiLU' at the unrounded FP32 preactivation, with FP32 and with BF16 taps (``flags`` names the taps dtype); the
    # accumulators, ``work`` counters and cross-CTA reduction of the route's programs above.
    "cake_causal_conv1d_train_4c8b21939791ace6e830": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "grid_mult": {
                "sm_107a": 1,
            },
            "kind": "occupancy",
        },
        "kernel": "cake_causal_conv1d_train_4c8b21939791ace6e830",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_4c8b21939791ace6e830",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 87040,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_4c8b21939791ace6e830",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
    # Route ``bwd``: the direct backward program (no det twin): one ordinary launch without TMA, mbarriers or a stage
    # ring.  A CTA of 128 threads is ``ct_vec`` channel vectors x ``n_sub`` token sub-tiles; a thread owns eight
    # channels and 4 consecutive tokens of a token block of ``n_sub x 4`` tokens, and CTA ``b`` takes the next blocks
    # of channel tile ``b % n_ch_tiles`` (2 CTAs per SM).  ``dx`` is bitwise the route's staged programs'; each CTA
    # adds its CTA row into the FP32 accumulator ``acc`` (``[C x 4]``) and takes a ticket on its channel tile's
    # ``work`` counter, and the tile's last CTA stores FP32 ``dw`` and zeroes its accumulators and counter, as the
    # route's programs above do.  It runs only at its rows of ``DIRECT_ROWS`` (``select_rule`` kind ``rows``), on the
    # route's tiles of ``BACKWARD_DIRECT_PLANS``, and launches at least one CTA per channel tile and at most one per
    # token block (grid cap ``token_blocks`` of ``block_tokens`` 4), times the call's ``bwd_grid_mult``.
    "cake_causal_conv1d_train_a9deef7249fb422bc28a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "block_tokens": 4,
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a9deef7249fb422bc28a",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_dt``: the route's direct backward program (the ``bwd`` direct program's body and launch), built with
    # ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with FP32 and with BF16 taps (``flags`` names the
    # taps and ``dw`` dtypes).  ``dx`` is bitwise the route's staged programs'; the accumulators, ``work`` counters
    # and cross-CTA reduction of the route's programs above.  It runs only at its rows of ``DIRECT_ROWS``, on the
    # route's tiles of ``BACKWARD_DIRECT_PLANS``, under the grid cap ``token_blocks`` of ``block_tokens`` 4.
    "cake_causal_conv1d_train_6a1e78e37349abd841ff": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "block_tokens": 4,
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_6a1e78e37349abd841ff",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16384,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        },
    },
    # Route ``bwd_bias``: the route's direct backward program (the ``bwd`` direct program's body and launch), built
    # with ``exact_pre=True``: SiLU' at the unrounded FP32 preactivation, with FP32 and with BF16 taps and bias
    # (``flags`` names the dtypes).  ``dx`` is bitwise the route's staged programs'; the accumulators (``[C x 4]``
    # taps then ``[C]`` bias), ``work`` counters and cross-CTA reduction of the route's programs above.  It runs only
    # at its rows of ``DIRECT_ROWS``, on the route's tiles of ``BACKWARD_DIRECT_PLANS``, under the grid cap
    # ``token_blocks`` of ``block_tokens`` 4.
    "cake_causal_conv1d_train_2a2c48dec0406daa9e81": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw_acc"],
            ["buffer", "work"],
            ["buffer", "dw"],
            ["buffer", "db"],
            ["buffer", "b"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["parameter", "flags"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "block_tokens": 4,
            "cap": "token_blocks",
            "cooperative": False,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20480,
            "use_pdl": "runtime",
        },
        "library": "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        "logical": {
            "C": "C",
            "S": "S",
            "b": "b",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "db": "db",
            "dw": "dw",
            "dw_acc": "acc",
            "dx": "dx",
            "dy": "dy",
            "enable_pdl": "enable_pdl",
            "flags": "flags",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "work": "work",
            "x": "x",
        },
        "role": "bwd",
        "select_rule": {
            "kind": "rows",
        },
        "workspace_rule": {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 5,
            "work": "uint32",
        },
    },
}

# architecture -> route -> the records the route may launch (fallback first); the
# opt-in pair routes (``_pair``, :data:`PAIR_ROUTES`) list the pair builds of their
# twin route's programs.
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd": [
            "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_e8e950fb254d3376db52",
            "cake_causal_conv1d_train_82e874f30406be2334b6",
            "cake_causal_conv1d_train_1ec222b2d908a7d6ea21",
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_93f7280de3dcb9ad7f49",
            "cake_causal_conv1d_train_92aeee41bc2e2d394937",
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_8433ef0894e75c192d4b",
            "cake_causal_conv1d_train_7dd30c23ccab6ab97182",
        ],
        "bwd_bias_fp32g_pair": [
            "cake_causal_conv1d_train_b42a8b7cf83f2b508d99",
            "cake_causal_conv1d_train_7c3318fb3f7e750cb63b",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_7035e23df8c9bf3c11e8",
            "cake_causal_conv1d_train_092af77575f0fdbf3615",
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_e3ff1615646de68a1aab",
            "cake_causal_conv1d_train_41fcd5b4c3749107aba7",
        ],
        "bwd_fp32g_pair": [
            "cake_causal_conv1d_train_9181be685ac105263751",
            "cake_causal_conv1d_train_6857ed58b2a273da419c",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
            "cake_causal_conv1d_train_f5497330bfd286211643",
            "cake_causal_conv1d_train_a81632301ba92a887c46",
            "cake_causal_conv1d_train_d0b237e92b28348315d0",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_62cb38eb8aa667d46bc1",
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_0f3982a5518f5a8696f6",
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ],
    },
    "sm_103a": {
        "bwd": [
            "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
            "cake_causal_conv1d_train_e8e950fb254d3376db52",
            "cake_causal_conv1d_train_77692909592ff14be82f",
            "cake_causal_conv1d_train_ccd45ec8667831b67344",
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_93f7280de3dcb9ad7f49",
            "cake_causal_conv1d_train_92aeee41bc2e2d394937",
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_8433ef0894e75c192d4b",
            "cake_causal_conv1d_train_7dd30c23ccab6ab97182",
        ],
        "bwd_bias_fp32g_pair": [
            "cake_causal_conv1d_train_b42a8b7cf83f2b508d99",
            "cake_causal_conv1d_train_7c3318fb3f7e750cb63b",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_7035e23df8c9bf3c11e8",
            "cake_causal_conv1d_train_092af77575f0fdbf3615",
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_e3ff1615646de68a1aab",
            "cake_causal_conv1d_train_41fcd5b4c3749107aba7",
        ],
        "bwd_fp32g_pair": [
            "cake_causal_conv1d_train_9181be685ac105263751",
            "cake_causal_conv1d_train_6857ed58b2a273da419c",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
            "cake_causal_conv1d_train_f5497330bfd286211643",
            "cake_causal_conv1d_train_a81632301ba92a887c46",
            "cake_causal_conv1d_train_d0b237e92b28348315d0",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_62cb38eb8aa667d46bc1",
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_0f3982a5518f5a8696f6",
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ],
    },
    "sm_107a": {
        "bwd": [
            "cake_causal_conv1d_train_a79ed3e33712abf525ad",
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
            "cake_causal_conv1d_train_c798878a383c652c428d",
            "cake_causal_conv1d_train_5ed71ee5908c07376183",
            "cake_causal_conv1d_train_06cfd917a14c30ce8ebc",
            "cake_causal_conv1d_train_e5d6d187103ab6e0cbbb",
            "cake_causal_conv1d_train_175b82ccd2309f434dfb",
            "cake_causal_conv1d_train_3293c0e34e70413fb4af",
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ],
        "bwd_bias": [
            "cake_causal_conv1d_train_6a8201512a0992310513",
            "cake_causal_conv1d_train_92aeee41bc2e2d394937",
            "cake_causal_conv1d_train_4c8b21939791ace6e830",
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ],
        "bwd_bias_fp32g": [
            "cake_causal_conv1d_train_286b95d7960d16de0849",
            "cake_causal_conv1d_train_7dd30c23ccab6ab97182",
        ],
        "bwd_bias_fp32g_pair": [
            "cake_causal_conv1d_train_1212581b2f85eebc7d7f",
            "cake_causal_conv1d_train_7c3318fb3f7e750cb63b",
        ],
        "bwd_dt": [
            "cake_causal_conv1d_train_d3273135fc61ea150378",
            "cake_causal_conv1d_train_092af77575f0fdbf3615",
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ],
        "bwd_fp32g": [
            "cake_causal_conv1d_train_c5abee01ba7440dd50b8",
            "cake_causal_conv1d_train_41fcd5b4c3749107aba7",
            "cake_causal_conv1d_train_3fdd49d8bc8a1048df1d",
        ],
        "bwd_fp32g_pair": [
            "cake_causal_conv1d_train_4685f66f3b7ef380e22a",
            "cake_causal_conv1d_train_6857ed58b2a273da419c",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
            "cake_causal_conv1d_train_f5497330bfd286211643",
            "cake_causal_conv1d_train_8665d0756b8240ecb4d0",
            "cake_causal_conv1d_train_d00a2892aa363564c673",
            "cake_causal_conv1d_train_a81632301ba92a887c46",
            "cake_causal_conv1d_train_d0b237e92b28348315d0",
        ],
        "fwd_bias": [
            "cake_causal_conv1d_train_62cb38eb8aa667d46bc1",
            "cake_causal_conv1d_train_08880ca12e4064bc8cdb",
            "cake_causal_conv1d_train_d829c95e1d610bc14b23",
            "cake_causal_conv1d_train_77af6383ff00b1cd047c",
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ],
        "fwd_dt": [
            "cake_causal_conv1d_train_0f3982a5518f5a8696f6",
            "cake_causal_conv1d_train_9c1e7ee9f37e3102660e",
            "cake_causal_conv1d_train_176ebb3e14f7ad9c94dc",
            "cake_causal_conv1d_train_6211cda590057b12f1a2",
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ],
    },
}


# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The routes (``ROUTES[arch]`` keys) a public call may resolve to.  ``fwd`` /
# ``bwd`` are the default routes (FP32 taps, no bias, SiLU' at the BF16-rounded
# preactivation); ``_dt`` routes take BF16 taps and / or store a BF16 gradient
# (the ``flags`` scalar says which), ``_bias`` routes add the per-channel bias
# and its gradient, ``_fp32g`` routes evaluate SiLU' at the unrounded FP32
# preactivation for the weight gradient, and ``bwd_dt`` and ``bwd_bias``
# evaluate SiLU' at the unrounded FP32 preactivation with FP32 and with BF16
# taps.
# The ``_pair`` routes are the opt-in FP32 hi/lo pair forms of the two
# ``_fp32g`` routes (:data:`PAIR_ROUTES`).
# :func:`forward_route` and :func:`backward_route` are the host's map from a
# call's arguments to a route.
KINDS = (
    "fwd",
    "fwd_dt",
    "fwd_bias",
    "bwd",
    "bwd_dt",
    "bwd_bias",
    "bwd_fp32g",
    "bwd_bias_fp32g",
    "bwd_fp32g_pair",
    "bwd_bias_fp32g_pair",
)
# The opt-in pair routes and the route each pairs (its twin): a call selects
# one only with ``dweight_sum="fp32_pair"`` (:func:`backward_route`), never by
# default.  A pair route's programs are its twin's programs built with
# ``fold_kahan`` and ``acc_pairs`` (each sum an FP32 (hi, lo) pair, rounded
# once); ``dx`` is the twin's.  A pair route selects its program by the twin
# route's thresholds and plans its launch (channel tiles, grid) by the twin
# route's rule (:func:`record_route`).
PAIR_ROUTES: dict[str, str] = {
    "bwd_fp32g_pair": "bwd_fp32g",
    "bwd_bias_fp32g_pair": "bwd_bias_fp32g",
}
# The ``dweight`` / ``dbias`` sum forms a backward call may ask for
# (``dweight_sum``): ``"fp32"`` (the default) is one plain FP32 word per sum,
# ``"fp32_pair"`` an FP32 (hi, lo) pair per sum (the pair routes).
DWEIGHT_SUMS = ("fp32", "fp32_pair")
# The routes every exported architecture ships: the base forward and backward.
# The other routes are optional per architecture; a call that needs one the
# architecture lacks is refused with a ``ValueError`` naming the route.
REQUIRED_KINDS = ("fwd", "bwd")
# The role of a module record: a forward program or a backward program (the
# one launch of a backward call; this package has no reduce programs).
ROLES = ("fwd", "bwd")
# The grid-rule vocabulary (``record["grid_rule"]["kind"]``, see
# :func:`launch_grid`): ``co_resident`` launches exactly ``ctas_per_sm x SM
# count`` CTAs, the co-resident count (the forward records, the direct
# backward records and the KCP forward and head programs of :mod:`.kcp_jit`,
# which share this host plan); ``occupancy`` launches ``grid_mult[arch] x
# ctas_per_sm x SM count`` CTAs normally (the other backward programs,
# single-rank and KCP; ``grid_mult`` names every architecture of its record).
GRID_KINDS = ("co_resident", "occupancy")
# The grid caps of a ``co_resident`` record (``record["grid_rule"]["cap"]``, see
# :func:`launch_grid`): ``token_blocks`` launches at most one CTA per token
# block of the call, ``n_ch_tiles x ceil(S / (block_tokens x n_sub))`` (the
# direct programs: each CTA takes the next blocks of ``n_sub x block_tokens``
# tokens of its channel tile, so a larger grid would only add CTAs without a
# block).  ``block_tokens`` is the record's ``grid_rule["block_tokens"]`` (4 on
# the direct backward records) or, where the record has none (the forward
# direct records), :data:`DIRECT_BLOCK_TOKENS`.  A backward record's capped
# grid keeps at least one CTA per channel tile.
GRID_CAPS = ("token_blocks",)
# The tokens of one token sub-tile of a forward direct program's block (a thread's
# rows).
DIRECT_BLOCK_TOKENS = 8
# The selection-rule vocabulary of a forward or backward record
# (``record["select_rule"]["kind"]``); of the records whose rule admits a call
# the one of the highest channel threshold runs, and a record without a rule is
# its route's fallback.  ``channels_tokens_at_least`` adds a token floor to the
# channel threshold.
SELECT_KINDS = ("channels_at_least", "channels_tokens_at_least")
# The selection-rule kind of a record that runs only at the rows of a table
# (``{"kind": "rows"}``): a backward program at its segments of
# :data:`SM_WIDE_ROWS`, a forward pair direct program at its ranges of
# :data:`PAIR_ROWS`, a forward or backward direct program at its ranges of
# :data:`DIRECT_ROWS`.  No channel threshold or token floor ever admits it
# (:func:`_admits` is ``False``), so the rules never pick it and it is no
# route's fallback.
ROWS_KIND = "rows"
# The ``flags`` bit mask of the ``dtype_flags`` programs (every program of the
# six other routes ``fwd_dt``, ``fwd_bias``, ``bwd_dt``, ``bwd_bias``,
# ``bwd_fp32g`` and ``bwd_bias_fp32g``): the host
# derives it from the tap tensor, the ACTUAL
# ``dweight`` out tensor, the bias tensor and the ACTUAL ``dbias`` out tensor;
# the kernel reads it once in its prologue (load form) and once in its epilogue
# (store form).  The default routes' programs take no ``flags``.
FLAG_W_BF16 = 1  # ``w`` holds BF16 taps (else FP32)
FLAG_DW_BF16 = 2  # ``dw`` is stored as BF16 (else FP32)
FLAG_B_BF16 = 4  # ``b`` holds BF16 (else FP32); bias programs
FLAG_DB_BF16 = 8  # ``db`` is stored as BF16 (else FP32); bias programs
# The convolution width: ``weight`` and ``dweight`` are ``[C, WIDTH]`` and tap
# ``WIDTH - 1`` multiplies the current token.
WIDTH = 4
# A forward thread and a backward thread each own one 16-byte vector of
# eight BF16 channels, so ``C`` is a multiple of eight; the ``vec`` of every
# record's geometry divides this.
CHANNEL_MULTIPLE = 8
# The kernels take ``S``, ``C`` and the plan scalars as 32-bit integers.
INT32_MAX = 2**31 - 1
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "cu",
    "y",
    "dy",
    "dx",
    "dw",
    "acc",
    "b",
    "db",
    "work",
)
# The workspace-rule vocabulary (``record["workspace_rule"]["kind"]``): a
# backward program's ``accumulators`` (``{"kind": "accumulators", "taps": 4 | 5,
# "acc": "float32" | "float64", "work": "uint32", "slots": K}``:
# ``slots x taps x C`` accumulator words of dtype ``acc``; every program of
# this package sums in FP32, one plain FP32 word per sum -- ``slots`` copies
# (the program's accumulator copies; required, a positive int) of the ``[C, 4]`` taps, then
# the ``[C]`` bias on the bias routes -- and one ``work`` ticket counter per
# channel tile,
# :func:`accumulator_layout`) and a forward program's ``tile_counters`` (its
# int32 ``work`` counters, ``words_per_tile`` per channel tile,
# :func:`counter_words`).  The backward's ``work`` and the forward's ``work`` are
# separate buffers of separate calls.
WORKSPACE_KINDS = ("accumulators", "tile_counters")
# The dtype names an ``accumulators`` rule may give ``acc`` and ``work``, with
# their element sizes in bytes.
ACCUMULATOR_DTYPES: dict[str, int] = {"float32": 4, "float64": 8}
COUNTER_DTYPES: dict[str, int] = {"uint32": 4}
# The byte alignment of the ``acc`` and ``work`` regions inside the backward
# workspace (the kernels add 16-byte ``v4`` vectors into ``acc``).
WORKSPACE_ALIGN = 16
# The scalars :func:`plan` produces, in kernel order; ``flags`` (the dtype bit
# mask of the ``dtype_flags`` programs) is set by the host beside them.
PLAN_SCALARS = (
    "S",
    "C",
    "num_docs",
    "search_steps",
    "ct_vec",
    "n_sub",
    "n_ch_tiles",
)
# ``enable_pdl`` (0 or 1) is the per-call launch scalar of a record whose
# ``launch.use_pdl`` is ``"runtime"``: its binding sets the programmatic
# dependent launch attribute only when it is 1 (:func:`pdl_enabled`).
LOGICAL_SCALARS = (*PLAN_SCALARS, "flags", "enable_pdl")


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_applies(*, device_capability: Sequence[int], kind: str) -> bool:
    """Whether the verified export owns one training call on a device."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return kind in ROUTES.get(arch, {})


def route_role(kind: str) -> str:
    """``"fwd"`` or ``"bwd"``: the kernel a route's programs are built from (``ValueError`` for an unknown route)."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return "fwd" if kind.startswith("fwd") else "bwd"


def forward_route(*, bias: bool, bf16_taps: bool) -> str:
    """The forward route of a call: ``fwd_bias`` with a bias, else ``fwd_dt`` for BF16 taps, else ``fwd``.

    A bias program always has a bias (no runtime switch: adding ``+0.0`` would
    flip the sign of ``-0.0`` outputs), so ``bias=None`` never selects one; the
    bias programs read FP32 or BF16 taps and bias through ``flags``.
    """

    if bias:
        return "fwd_bias"
    return "fwd_dt" if bf16_taps else "fwd"


def backward_route(
    *,
    bias: bool,
    fp32_grad: bool,
    bf16_taps: bool,
    bf16_dweight: bool,
    pair: bool = False,
) -> str:
    """The backward route of a call.

    ``bias`` selects ``bwd_bias_fp32g`` under ``fp32_grad`` (the
    ``silu_grad_preactivation="fp32"`` policy), else ``bwd_bias`` with FP32
    and with BF16 taps (its programs evaluate SiLU' at the unrounded FP32
    preactivation); without a bias ``fp32_grad`` selects ``bwd_fp32g``; else
    BF16 taps or a BF16 ``dweight`` out tensor select ``bwd_dt`` (its
    programs evaluate SiLU' at the unrounded FP32 preactivation, with FP32
    and with BF16 taps), and FP32 taps and ``dweight`` the default ``bwd``.
    Every route but ``bwd`` reads the tap / gradient dtypes
    from ``flags``.  ``pair`` (the ``dweight_sum="fp32_pair"`` opt-in)
    selects the ``_fp32g`` route's pair route (:data:`PAIR_ROUTES`); only the
    ``_fp32g`` routes have one, so ``pair`` without ``fp32_grad`` is a
    ``ValueError`` (never another route).
    """

    if pair and not fp32_grad:
        raise ValueError(
            'dweight_sum="fp32_pair" needs silu_grad_preactivation="fp32": only '
            "the FP32-SiLU' routes have an FP32 (hi, lo) pair form"
        )
    if bias:
        route = "bwd_bias_fp32g" if fp32_grad else "bwd_bias"
    elif fp32_grad:
        route = "bwd_fp32g"
    else:
        route = "bwd_dt" if bf16_taps or bf16_dweight else "bwd"
    return f"{route}_pair" if pair else route


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch).

    A call launches the one :func:`selected_modules` picks for its channel
    count.
    """

    route_role(kind)
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record, ``None`` when the record carries none.

    A record without a rule (no ``select_rule`` key, or ``None``) is its
    route's fallback program.  ``{"kind": "channels_at_least", "channels":
    {arch: C_min, ...}}`` owns the calls whose channel count is at least
    ``channels[arch]`` on the architectures it names and never hits on the
    others.  ``{"kind": "channels_tokens_at_least", "channels": {arch: C_min,
    ...}, "tokens": {arch: S_min, ...}}`` owns those calls only when the token
    count is also at least ``tokens[arch]``.  When several records own a call,
    the one of the highest channel threshold runs; at a row of
    :data:`FALLBACK_ROWS` the route's fallback runs whatever the rules.
    ``{"kind": "rows"}`` (:data:`ROWS_KIND`) owns no call by the rules: the
    record runs only at the rows of :data:`SM_WIDE_ROWS`, :data:`PAIR_ROWS` or
    :data:`DIRECT_ROWS` that name it, before :data:`FALLBACK_ROWS` and the
    rules.
    """

    return record.get("select_rule")


def _threshold(rule: dict[str, Any], *, arch: str) -> int | None:
    """The channel count from which a selection rule owns the calls on ``arch``, ``None`` where it never does
    (a rule of kind :data:`ROWS_KIND` never does; ``ValueError`` for an unknown kind)."""

    kind = rule["kind"]
    if kind == ROWS_KIND:
        return None
    if kind in SELECT_KINDS:
        channels = rule["channels"]
        return int(channels[arch]) if arch in channels else None
    raise ValueError(
        f"unknown Cake causal conv1d training selection rule: {kind!r} "
        f"(known: {(*SELECT_KINDS, ROWS_KIND)})"
    )


def _tokens(rule: dict[str, Any], *, arch: str) -> int | None:
    """The token count from which a selection rule owns the calls on ``arch`` (its token floor), ``None`` when the
    rule has none there (``ValueError`` for an unknown kind)."""

    _threshold(rule, arch=arch)
    if rule["kind"] != "channels_tokens_at_least":
        return None
    tokens = rule["tokens"]
    return int(tokens[arch]) if arch in tokens else None


def _admits(rule: dict[str, Any], *, arch: str, C: int, S: int | None = None) -> bool:
    """Whether a selection rule owns ``C`` channels (and ``S`` tokens) on ``arch``.

    ``ValueError`` for an unknown kind, or when the rule's token floor decides
    and ``S`` is ``None``.
    """

    threshold = _threshold(rule, arch=arch)
    if threshold is None or int(C) < threshold:
        return False
    floor = _tokens(rule, arch=arch)
    if floor is None:
        return True
    if S is None:
        raise ValueError(
            f"the Cake causal conv1d training selection rule {rule!r} on {arch} "
            f"for C={int(C)} runs from S {floor}; the selection needs S"
        )
    return int(S) >= floor


# The rows where a route runs its fallback program (the route's record without
# a ``select_rule``) in place of the program its selection rules pick,
# ``{arch: {(route, C_lo, C_hi): (S_min, S_max)}}``: a call of ``route`` on
# ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple of 8) and ``S_min <=
# S <= S_max`` tokens launches the fallback (:func:`selected_modules`).  A
# backward key's fallback keeps the route's tiles (:func:`plan`); a forward
# key's fallback is the route's static program on the route's plan.  The keys
# of one ``(arch, route)`` are disjoint, and a forward key shares no ``(C, S)``
# with a :data:`DIRECT_ROWS` range; a :data:`PAIR_ROWS` range runs ahead of a
# forward key where it covers its ``(C, S)``.  Inside the range of a backward
# key of :data:`DIRECT_ROWS` the route's direct program runs instead, so a
# backward key's fallback runs outside that range.  Rendered from the kernel
# generators' tables; every other channel count, route, ``S`` and architecture
# selects by :data:`SM_WIDE_ROWS`, :data:`PAIR_ROWS`, :data:`DIRECT_ROWS` and
# the rules.
FALLBACK_ROWS: dict[str, dict[tuple[str, int, int], tuple[int, int]]] = {
    "sm_100a": {
        ("bwd", 520, 608): (1, 24575),
        ("bwd", 616, 760): (1, 131072),
        ("bwd", 768, 768): (1, 262144),
        ("bwd", 776, 1016): (1, 62585),
        ("bwd_bias", 520, 568): (1, 16384),
        ("bwd_bias", 576, 640): (1, 24576),
        ("bwd_bias", 648, 1016): (1, 16384),
        ("bwd_dt", 520, 568): (1, 16384),
        ("bwd_dt", 576, 640): (1, 24576),
        ("bwd_dt", 648, 1016): (1, 16384),
        ("fwd", 512, 512): (1, 24576),
        ("fwd", 520, 544): (1, 12288),
        ("fwd", 552, 568): (1, 8192),
        ("fwd", 576, 576): (1, 4096),
        ("fwd", 584, 640): (1, 8192),
        ("fwd", 648, 704): (1, 12288),
        ("fwd", 712, 712): (1, 12288),
        ("fwd", 720, 768): (1, 8192),
        ("fwd", 776, 888): (1, 8192),
        ("fwd", 896, 896): (1, 12288),
        ("fwd", 904, 952): (1, 8192),
        ("fwd", 960, 1024): (1, 24576),
        ("fwd_dt", 512, 512): (1, 24576),
        ("fwd_dt", 520, 568): (1, 8192),
        ("fwd_dt", 576, 576): (1, 4096),
        ("fwd_dt", 584, 640): (1, 8192),
        ("fwd_dt", 648, 696): (1, 8192),
        ("fwd_dt", 704, 704): (1, 12288),
        ("fwd_dt", 712, 768): (1, 16384),
        ("fwd_dt", 776, 896): (1, 12288),
        ("fwd_dt", 904, 1024): (1, 12288),
        ("fwd_bias", 512, 512): (1, 24576),
        ("fwd_bias", 520, 544): (1, 12288),
        ("fwd_bias", 552, 568): (1, 8192),
        ("fwd_bias", 576, 576): (1, 4096),
        ("fwd_bias", 584, 640): (1, 4096),
        ("fwd_bias", 648, 696): (1, 8192),
        ("fwd_bias", 704, 704): (1, 17408),
        ("fwd_bias", 712, 768): (1, 16384),
        ("fwd_bias", 776, 776): (1, 8192),
        ("fwd_bias", 784, 896): (1, 8192),
        ("fwd_bias", 904, 1024): (1, 12288),
    },
    "sm_103a": {
        ("bwd", 520, 608): (1, 20479),
        ("bwd", 616, 760): (1, 31292),
        ("bwd", 768, 768): (1, 250340),
        ("bwd", 776, 1016): (1, 125170),
        ("bwd_bias", 520, 568): (1, 16384),
        ("bwd_bias", 576, 640): (1, 24576),
        ("bwd_bias", 648, 1016): (1, 16384),
        ("bwd_dt", 520, 568): (1, 16384),
        ("bwd_dt", 576, 640): (1, 24576),
        ("bwd_dt", 648, 1016): (1, 16384),
        ("fwd", 512, 512): (1, 2048),
        ("fwd", 520, 520): (1, 8192),
        ("fwd", 528, 568): (1, 4096),
        ("fwd", 576, 632): (1, 2048),
        ("fwd", 640, 688): (1, 4096),
        ("fwd", 696, 696): (1, 4096),
        ("fwd", 704, 704): (1, 8192),
        ("fwd", 712, 712): (1, 8192),
        ("fwd", 720, 768): (1, 8192),
        ("fwd", 776, 824): (1, 4096),
        ("fwd", 832, 832): (1, 8192),
        ("fwd", 840, 896): (1, 4096),
        ("fwd", 904, 952): (1, 2048),
        ("fwd", 960, 1024): (1, 24576),
        ("fwd_dt", 512, 512): (1, 20480),
        ("fwd_dt", 520, 568): (1, 8192),
        ("fwd_dt", 576, 640): (1, 8192),
        ("fwd_dt", 648, 704): (1, 8192),
        ("fwd_dt", 712, 752): (1, 12800),
        ("fwd_dt", 760, 760): (1, 12288),
        ("fwd_dt", 768, 768): (1, 12288),
        ("fwd_dt", 776, 776): (1, 12288),
        ("fwd_dt", 784, 832): (1, 12288),
        ("fwd_dt", 840, 888): (1, 8192),
        ("fwd_dt", 896, 896): (1, 8192),
        ("fwd_dt", 904, 952): (1, 8192),
        ("fwd_dt", 960, 1016): (1, 20480),
        ("fwd_dt", 1024, 1024): (1, 24576),
        ("fwd_bias", 512, 512): (1, 24576),
        ("fwd_bias", 520, 520): (1, 12288),
        ("fwd_bias", 528, 544): (1, 24576),
        ("fwd_bias", 552, 568): (1, 8192),
        ("fwd_bias", 576, 576): (1, 2048),
        ("fwd_bias", 584, 640): (1, 4104),
        ("fwd_bias", 648, 704): (1, 16384),
        ("fwd_bias", 712, 712): (1, 12800),
        ("fwd_bias", 720, 728): (13824, 16383),
        ("fwd_bias", 736, 736): (14336, 16383),
        ("fwd_bias", 744, 760): (13824, 16383),
        ("fwd_bias", 768, 768): (13824, 16384),
        ("fwd_bias", 776, 896): (1, 8192),
        ("fwd_bias", 904, 952): (1, 12288),
        ("fwd_bias", 960, 1024): (1, 20480),
    },
    "sm_107a": {
        ("fwd", 512, 512): (1, 8192),
        ("fwd", 520, 520): (1, 24575),
        ("fwd", 528, 528): (1, 16384),
        ("fwd", 536, 536): (1, 16384),
        ("fwd", 544, 568): (1, 16384),
        ("fwd", 576, 704): (1, 4096),
        ("fwd", 712, 760): (1, 4096),
        ("fwd", 768, 768): (1, 2048),
        ("fwd", 776, 824): (1, 32768),
        ("fwd", 832, 832): (1, 4096),
        ("fwd", 840, 888): (1, 8192),
        ("fwd", 896, 896): (1, 16384),
        ("fwd", 904, 1016): (1, 12288),
        ("fwd", 1024, 1024): (1, 32768),
        ("fwd_dt", 512, 512): (1, 32768),
        ("fwd_dt", 520, 528): (1, 16384),
        ("fwd_dt", 536, 544): (1, 16384),
        ("fwd_dt", 552, 568): (1, 16384),
        ("fwd_dt", 576, 640): (1, 12288),
        ("fwd_dt", 648, 704): (1, 12288),
        ("fwd_dt", 712, 712): (1, 32768),
        ("fwd_dt", 720, 768): (1, 12288),
        ("fwd_dt", 776, 824): (1, 24576),
        ("fwd_dt", 832, 832): (1, 20480),
        ("fwd_dt", 840, 888): (1, 20480),
        ("fwd_dt", 896, 896): (1, 20480),
        ("fwd_dt", 904, 1024): (1, 32768),
        ("fwd_bias", 512, 512): (1, 32768),
        ("fwd_bias", 520, 528): (1, 24575),
        ("fwd_bias", 536, 568): (1, 32768),
        ("fwd_bias", 576, 576): (1, 12296),
        ("fwd_bias", 584, 584): (1, 24576),
        ("fwd_bias", 592, 592): (1, 16384),
        ("fwd_bias", 600, 600): (1, 24576),
        ("fwd_bias", 608, 608): (1, 16384),
        ("fwd_bias", 616, 616): (1, 24576),
        ("fwd_bias", 624, 624): (1, 16384),
        ("fwd_bias", 632, 632): (1, 24576),
        ("fwd_bias", 640, 696): (1, 24576),
        ("fwd_bias", 704, 768): (1, 32768),
        ("fwd_bias", 776, 1024): (1, 32768),
    },
}


def fallback_row_range(arch: str, kind: str, C: int) -> tuple[int, int] | None:
    """The ``(S_min, S_max)`` of the :data:`FALLBACK_ROWS` key of ``arch`` on
    route ``kind`` covering ``C`` (``C_lo <= C <= C_hi``, ``C`` a multiple of
    8), ``None`` without one: inside that token range (both bounds included)
    the call launches the route's fallback program, but where a
    :data:`PAIR_ROWS` range holds a forward call."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    for (route, c_lo, c_hi), (s_min, s_max) in FALLBACK_ROWS.get(arch, {}).items():
        if route == kind and int(c_lo) <= C <= int(c_hi):
            return int(s_min), int(s_max)
    return None


# The rows where a forward or backward route runs its direct program, ``{arch:
# {(route, C_lo, C_hi): (S_lo, S_hi, module)}}``: a call of ``route`` on
# ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple of 8) and ``S_lo <=
# S <= S_hi`` tokens launches ``module`` (:func:`direct_module`), the route's
# direct record (its ``select_rule`` of kind :data:`ROWS_KIND`, its geometry's
# ``vec`` 8), on the architecture's direct tiles of the route
# (:func:`direct_split`, :func:`plan`).  The keys of one ``(arch, route)`` are
# disjoint.  The forward keys come first: they share no ``(C, S)`` with a
# :data:`FALLBACK_ROWS` range (where a key has both, the static range of
# :data:`FALLBACK_ROWS` ends where the direct range starts), and a
# :data:`PAIR_ROWS` range runs ahead of a forward key where it covers its
# ``(C, S)``.  Then, per architecture, the backward keys of the ``bwd``,
# ``bwd_dt`` and ``bwd_bias`` routes (channel ranges within C = 512 to 1024,
# each with its own token range within S = 1 to 4096): they share no ``(C,
# S)`` with an :data:`SM_WIDE_ROWS` segment, and inside their range the direct
# program runs ahead of :data:`FALLBACK_ROWS`, :data:`TILE_OVERRIDES`,
# :data:`NARROW_TILES` and the rules.  Rendered from the forward and the direct
# backward kernel generators' tables; outside a key's range the other tables
# and the rules select.
DIRECT_ROWS: dict[str, dict[tuple[str, int, int], tuple[int, int, str]]] = {
    "sm_100a": {
        ("fwd", 520, 544): (
            12289,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 552, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 576, 576): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 584, 640): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 648, 704): (
            12289,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 712, 712): (
            12289,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 720, 768): (
            8193,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 776, 888): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 896, 896): (
            12289,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 904, 952): (
            8193,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd_dt", 520, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 576, 576): (
            4097,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 584, 640): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 648, 696): (
            8193,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 704, 704): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 712, 768): (
            16385,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 776, 896): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 904, 1024): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_bias", 520, 544): (
            12289,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 552, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 576, 576): (
            4097,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 584, 640): (
            4097,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 648, 696): (
            8193,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 704, 704): (
            17409,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 776, 776): (
            8193,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 784, 896): (
            8193,
            14336,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("bwd", 512, 832): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 896, 1016): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 1024, 1024): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd_dt", 512, 616): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 640, 736): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 768, 768): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 800, 800): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 832, 832): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 864, 864): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 896, 896): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 960, 1016): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_bias", 512, 512): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 520, 576): (
            1000,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
    },
    "sm_103a": {
        ("fwd", 512, 512): (
            2049,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 520, 520): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 528, 568): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 576, 632): (
            2049,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 640, 688): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 696, 696): (
            4097,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 704, 704): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 712, 712): (
            8193,
            20479,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 720, 768): (
            8193,
            19456,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 776, 824): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 832, 832): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 840, 896): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 904, 952): (
            2049,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd_dt", 512, 512): (
            20481,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 520, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 576, 640): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 648, 704): (
            8193,
            18432,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 712, 752): (
            12801,
            19456,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 760, 760): (
            12289,
            19456,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 768, 768): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 776, 776): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 784, 832): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 840, 888): (
            8193,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 896, 896): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 904, 952): (
            8193,
            16384,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_bias", 520, 520): (
            12289,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 552, 568): (
            8193,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 576, 576): (
            2049,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 584, 640): (
            4105,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 712, 712): (
            12801,
            14592,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 776, 896): (
            8193,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("bwd", 512, 896): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 960, 1016): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 1024, 1024): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd_dt", 512, 640): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 672, 672): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 704, 768): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 800, 832): (
            128,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 864, 896): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 960, 1016): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_bias", 512, 512): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 520, 576): (
            1000,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 768, 768): (
            1000,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 960, 960): (
            128,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
    },
    "sm_107a": {
        ("fwd", 512, 512): (
            8193,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 528, 528): (
            16385,
            24575,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 536, 536): (
            16385,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 544, 568): (
            16385,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 576, 704): (
            4097,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 712, 760): (
            4097,
            16384,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 768, 768): (
            2049,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 832, 832): (
            4097,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 840, 888): (
            8193,
            16384,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 896, 896): (
            16385,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 904, 1016): (
            12289,
            16384,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd_dt", 520, 528): (
            16385,
            24575,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 536, 544): (
            16385,
            32768,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 552, 568): (
            16385,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 576, 640): (
            12289,
            32768,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 648, 704): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 720, 768): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 832, 832): (
            20481,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 896, 896): (
            20481,
            32768,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_bias", 576, 576): (
            12297,
            32768,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 592, 592): (
            16385,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 608, 608): (
            16385,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 624, 624): (
            16385,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("bwd", 512, 512): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 520, 576): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 600, 616): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 640, 640): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 704, 704): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 768, 768): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 832, 832): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 992, 992): (1, 1000, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 1024, 1024): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd_dt", 512, 512): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 520, 520): (
            1000,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 576, 576): (
            1000,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 600, 640): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 704, 704): (
            1000,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 768, 832): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 896, 896): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 928, 960): (
            1,
            1000,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1016, 1016): (
            1,
            1000,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_bias", 512, 512): (
            1,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 520, 832): (
            1000,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 1024, 1024): (
            1,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
    },
}
# The channel tiles of a forward call at a row of :data:`DIRECT_ROWS`, ``{arch:
# {route: (ct_vec, n_sub)}}``: ``ct_vec`` channel vectors per channel tile
# (8: 64 channels, 16: 128 channels) and ``n_sub`` token sub-tiles per CTA, so
# a call of ``C`` channels runs ``ceil(C / vec / ct_vec)`` tiles; the tiles each
# route's direct program was timed on, one form per architecture and route.
DIRECT_PLANS: dict[str, dict[str, tuple[int, int]]] = {
    "sm_100a": {"fwd": (8, 16), "fwd_dt": (8, 16), "fwd_bias": (8, 16)},
    "sm_103a": {"fwd": (8, 16), "fwd_dt": (16, 8), "fwd_bias": (8, 16)},
    "sm_107a": {"fwd": (8, 16), "fwd_dt": (16, 8), "fwd_bias": (8, 16)},
}
# The channel tiles of a backward call at a row of :data:`DIRECT_ROWS`, ``{arch:
# {route: ((ct_vec, n_sub), ...)}}``: the tile forms of the route's direct
# backward program, 128-channel tiles ``(16, 8)`` then 64-channel tiles ``(8,
# 16)`` (``ct_vec x n_sub`` = 128 threads).  A call of ``C`` channels runs the
# first form whose tiles pad ``C`` to the fewest channel vectors, in
# ``ceil(C / vec / ct_vec)`` tiles (:func:`direct_split`): a ``C`` that 128
# divides runs 128-channel tiles, and C = 576 nine 64-channel tiles rather than
# five 128-channel tiles with 64 idle channels.  Exactly the ``(arch, route)``
# pairs of the backward keys of :data:`DIRECT_ROWS` have forms.
BACKWARD_DIRECT_PLANS: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
    "sm_100a": {
        "bwd": ((16, 8), (8, 16)),
        "bwd_dt": ((16, 8), (8, 16)),
        "bwd_bias": ((16, 8), (8, 16)),
    },
    "sm_103a": {
        "bwd": ((16, 8), (8, 16)),
        "bwd_dt": ((16, 8), (8, 16)),
        "bwd_bias": ((16, 8), (8, 16)),
    },
    "sm_107a": {
        "bwd": ((16, 8), (8, 16)),
        "bwd_dt": ((16, 8), (8, 16)),
        "bwd_bias": ((16, 8), (8, 16)),
    },
}
# The build threads of the direct backward programs: a backward plan takes the
# direct tiles at a row of :data:`DIRECT_ROWS` at this thread count only (other
# geometries keep their plan, as at the rows of :data:`NARROW_TILES`).
BACKWARD_DIRECT_THREADS = 128


def direct_row_range(arch: str, kind: str, C: int) -> tuple[int, int] | None:
    """The ``(S_lo, S_hi)`` of the :data:`DIRECT_ROWS` key of ``arch`` on route ``kind`` covering ``C`` (``C_lo <= C
    <= C_hi``, ``C`` a multiple of 8), ``None`` without one: inside that token range (both bounds included) the call
    launches the key's direct program, but where a :data:`PAIR_ROWS` range holds the call."""

    return _row_range(DIRECT_ROWS, arch, kind, C)


def direct_row(arch: str | None, kind: str | None, C: int, S: int | None) -> bool:
    """Whether a call of route ``kind`` on ``arch`` with ``C`` channels and ``S`` tokens lies in a :data:`DIRECT_ROWS`
    range (both bounds included); ``False`` without ``arch``, ``kind`` or ``S``."""

    return _in_rows(direct_row_range, arch, kind, C, S)


def direct_module(arch: str, kind: str, C: int, S: int | None) -> str | None:
    """The direct program a call of route ``kind`` on ``arch`` with ``C`` channels and ``S`` tokens launches: the
    module of the :data:`DIRECT_ROWS` key covering ``C`` when its range holds ``S`` (both bounds included), ``None``
    outside it or without a key.  ``S`` of ``None`` where a key covers ``C`` is a ``ValueError``; a module the route
    does not list, or whose rule is not of kind :data:`ROWS_KIND`, is a registry error (``RuntimeError``)."""

    return _rows_module(DIRECT_ROWS, "DIRECT_ROWS", "direct", arch, kind, C, S)


# The rows where a forward route runs its pair direct program (the direct program
# with each thread on a pair of channels) in place of the program
# :data:`DIRECT_ROWS`, :data:`FALLBACK_ROWS` or its selection rules pick, ``{arch:
# {(route, C_lo, C_hi): (S_lo, S_hi, module)}}``: a forward call of ``route`` on
# ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple of 8) and ``S_lo <= S <=
# S_hi`` tokens launches ``module`` (:func:`pair_module`), the route's record whose
# ``select_rule`` is of kind :data:`ROWS_KIND` and whose geometry's ``vec`` is 2,
# on the architecture's pair tiles of the route (:data:`PAIR_PLANS`,
# :func:`plan`).  The keys of one ``(arch, route)`` are disjoint in C: one key,
# but two on sm_100a ``fwd_dt``, whose range ends at S 1024 from C 648 (at most
# 512 CTAs of the pair grid).  A range may cover the ``(C, S)`` of
# :data:`FALLBACK_ROWS` and :data:`DIRECT_ROWS` ranges and takes them: those
# tables decide only outside it.  The det package's table, under this package's
# prefix.  Rendered from the forward kernel generator's table.
PAIR_ROWS: dict[str, dict[tuple[str, int, int], tuple[int, int, str]]] = {
    "sm_100a": {
        ("fwd", 512, 1024): (64, 3072, "cake_causal_conv1d_train_d0b237e92b28348315d0"),
        ("fwd_dt", 512, 640): (
            64,
            1536,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_dt", 648, 1024): (
            64,
            1024,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_bias", 512, 1024): (
            128,
            1536,
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ),
    },
    "sm_103a": {
        ("fwd", 512, 1024): (64, 3072, "cake_causal_conv1d_train_d0b237e92b28348315d0"),
        ("fwd_dt", 512, 1024): (
            32,
            1792,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_bias", 512, 1024): (
            16,
            1536,
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ),
    },
    "sm_107a": {
        ("fwd", 512, 1024): (64, 2560, "cake_causal_conv1d_train_d0b237e92b28348315d0"),
        ("fwd_dt", 512, 1024): (
            16,
            1792,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_bias", 512, 1024): (
            16,
            1792,
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ),
    },
}
# The channel tiles of a forward call at a row of :data:`PAIR_ROWS`, the form of
# :data:`DIRECT_PLANS` with ``ct_vec`` counted in channel pairs (a pair record's
# ``vec`` is 2): ``(16, 8)`` gives each CTA 32 channels and 64 tokens, ``ceil(C /
# 32)`` tiles, at every row.  The plan never changes ``y``.  Kept in sync with the
# forward kernel generator's pair plans.
PAIR_PLANS: dict[str, dict[str, tuple[int, int]]] = {
    "sm_100a": {"fwd": (16, 8), "fwd_dt": (16, 8), "fwd_bias": (16, 8)},
    "sm_103a": {"fwd": (16, 8), "fwd_dt": (16, 8), "fwd_bias": (16, 8)},
    "sm_107a": {"fwd": (16, 8), "fwd_dt": (16, 8), "fwd_bias": (16, 8)},
}


def pair_row_range(arch: str, kind: str, C: int) -> tuple[int, int] | None:
    """The ``(S_lo, S_hi)`` of the :data:`PAIR_ROWS` key of ``arch`` on route ``kind`` covering ``C`` (``C_lo <= C <=
    C_hi``, ``C`` a multiple of 8), ``None`` without one: inside that token range (both bounds included) the call
    launches the key's pair direct program."""

    return _row_range(PAIR_ROWS, arch, kind, C)


def pair_row(arch: str | None, kind: str | None, C: int, S: int | None) -> bool:
    """Whether a call of route ``kind`` on ``arch`` with ``C`` channels and ``S`` tokens lies in a :data:`PAIR_ROWS`
    range (both bounds included); ``False`` without ``arch``, ``kind`` or ``S``."""

    return _in_rows(pair_row_range, arch, kind, C, S)


def pair_module(arch: str, kind: str, C: int, S: int | None) -> str | None:
    """The pair direct program a call of route ``kind`` on ``arch`` with ``C`` channels and ``S`` tokens launches: the
    module of the :data:`PAIR_ROWS` key covering ``C`` when its range holds ``S`` (both bounds included), ``None``
    outside it or without a key; the errors of :func:`direct_module`."""

    return _rows_module(PAIR_ROWS, "PAIR_ROWS", "pair direct", arch, kind, C, S)


def _row_range(
    table: dict[str, dict[tuple[str, int, int], tuple[int, int, str]]],
    arch: str,
    kind: str,
    C: int,
) -> tuple[int, int] | None:
    """The ``(S_lo, S_hi)`` of the key of ``table`` (:data:`DIRECT_ROWS` or :data:`PAIR_ROWS`) of ``arch`` on route
    ``kind`` covering ``C``."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    for (route, c_lo, c_hi), (s_lo, s_hi, _name) in table.get(arch, {}).items():
        if route == kind and int(c_lo) <= C <= int(c_hi):
            return int(s_lo), int(s_hi)
    return None


def _in_rows(
    row_range: Callable[[str, str, int], tuple[int, int] | None],
    arch: str | None,
    kind: str | None,
    C: int,
    S: int | None,
) -> bool:
    """Whether ``S`` lies in the range ``row_range`` gives (both bounds included); ``False`` without ``arch``,
    ``kind`` or ``S``."""

    if arch is None or kind is None or S is None:
        return False
    bounds = row_range(arch, kind, C)
    return bounds is not None and bounds[0] <= int(S) <= bounds[1]


def _rows_module(
    table: dict[str, dict[tuple[str, int, int], tuple[int, int, str]]],
    table_name: str,
    what: str,
    arch: str,
    kind: str,
    C: int,
    S: int | None,
) -> str | None:
    """The module of the key of ``table`` (named ``table_name``, its programs ``what``) covering ``C`` when its range
    holds ``S``; see :func:`direct_module`."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    for (route, c_lo, c_hi), (s_lo, s_hi, name) in table.get(arch, {}).items():
        if route != kind or not int(c_lo) <= C <= int(c_hi):
            continue
        if S is None:
            raise ValueError(
                f"the Cake causal conv1d training {kind!r} route on {arch} for C={C} runs its {what} program at S "
                f"{s_lo} to {s_hi}; the selection needs S"
            )
        if not int(s_lo) <= int(S) <= int(s_hi):
            return None
        if (
            name not in route_modules(arch, kind)
            or (select_rule(MODULES[name]) or {}).get("kind") != ROWS_KIND
        ):
            raise RuntimeError(
                f"the {table_name} key ({route!r}, {c_lo}, {c_hi}) on {arch} names {name}, which is no record of the "
                f"route with a {ROWS_KIND!r} selection rule"
            )
        return name
    return None


def direct_split(
    C: int, *, geometry: dict[str, int], route: str, arch: str, pair: bool = False
) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of the direct program of ``route`` on ``arch`` for ``C`` channels under a
    record's build geometry, whatever ``S`` and the taps: on a forward route the :data:`DIRECT_PLANS` form ``(ct_vec,
    n_sub)`` (with ``pair``, the pair direct program's :data:`PAIR_PLANS` form; the deterministic package's split), on
    a backward route the first form of :data:`BACKWARD_DIRECT_PLANS` whose tiles pad ``C`` to the fewest channel
    vectors (``ceil(C / vec / ct_vec) x ct_vec``), and ``ceil(C / vec / ct_vec)`` channel tiles of the geometry's
    ``vec`` (the last tile partial).  ``ValueError`` for an architecture or route without a form, a form beyond the
    geometry's threads, or ``C`` that is not a positive multiple of ``vec``."""

    if arch not in ARCHES:
        raise ValueError(f"arch must be one of {ARCHES}, got {arch!r}")
    if not pair and route in KINDS and route_role(route) == "bwd":
        what, table = "direct", BACKWARD_DIRECT_PLANS
        table_name = "BACKWARD_DIRECT_PLANS"
        forms = table.get(arch, {}).get(route)
    else:
        what, table = ("pair", PAIR_PLANS) if pair else ("direct", DIRECT_PLANS)
        table_name = f"{what.upper()}_PLANS"
        form = table.get(arch, {}).get(route)
        forms = None if form is None else (form,)
    if forms is None:
        raise ValueError(
            f"the {route!r} route on {arch} has no {what} plan ({table_name} names "
            f"{sorted(table.get(arch, {}))})"
        )
    vec, threads = int(geometry["vec"]), int(geometry["threads"])
    for form in forms:
        ct_vec, n_sub = (int(value) for value in form)
        if ct_vec < 1 or n_sub < 1 or ct_vec * n_sub > threads:
            raise ValueError(
                f"the {what} plan {form!r} of route {route!r} on {arch} does not fit {threads} threads"
            )
    C = int(C)
    if C <= 0 or C % vec:
        raise ValueError(f"C={C} must be a positive multiple of {vec}")
    channel_vectors = C // vec
    # The first form of the fewest padded channel vectors (``min`` keeps the first of equal keys).
    ct_vec, n_sub = min(
        ((int(form[0]), int(form[1])) for form in forms),
        key=lambda tile: _ceil_div(channel_vectors, tile[0]) * tile[0],
    )
    return ct_vec, n_sub, _ceil_div(channel_vectors, ct_vec)


def direct_tiles(arch: str, route: str, C: int, geometry: dict[str, int]) -> int:
    """The channel tiles of the backward :data:`DIRECT_ROWS` key covering ``(arch, route, C)`` under ``geometry``
    (:func:`direct_split`; 0 without a key or at a thread count other than :data:`BACKWARD_DIRECT_THREADS`),
    whatever ``S``: the workspace bound of the key's launches."""

    if (
        route_role(route) != "bwd"
        or int(geometry["threads"]) != BACKWARD_DIRECT_THREADS
        or direct_row_range(arch, route, C) is None
    ):
        return 0
    return direct_split(C, geometry=geometry, route=route, arch=arch)[2]


# The rows where a backward route runs one of its SM-wide programs (one CTA per SM
# for the whole launch, each CTA a whole token stream of every channel) in place
# of the program :data:`FALLBACK_ROWS` or its selection rules pick, ``{arch:
# {(route, C_lo, C_hi): ((S_lo, S_hi, module), ...)}}``: a backward call of
# ``route`` on ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple of 8) and
# ``S_lo <= S <= S_hi`` tokens launches ``module`` (:func:`sm_wide_module`), a
# record of the route whose ``select_rule`` is of kind :data:`ROWS_KIND`.  The
# segments of a key ascend, are contiguous and end at ``INT32_MAX``; the keys of
# one ``(arch, route)`` are disjoint and share no ``(C, S)`` with a
# :data:`FALLBACK_ROWS` range.  Rendered from the backward kernel generator's
# table; every other channel count, route, ``S`` and architecture selects by
# :data:`FALLBACK_ROWS` and the rules.
SM_WIDE_ROWS: dict[str, dict[tuple[str, int, int], tuple[tuple[int, int, str], ...]]] = {
    "sm_100a": {
        ("bwd", 520, 608): (
            (24576, 98303, "cake_causal_conv1d_train_82e874f30406be2334b6"),
            (98304, 2147483647, "cake_causal_conv1d_train_1ec222b2d908a7d6ea21"),
        ),
    },
    "sm_103a": {
        ("bwd", 520, 608): (
            (20480, 31291, "cake_causal_conv1d_train_77692909592ff14be82f"),
            (31292, 2147483647, "cake_causal_conv1d_train_ccd45ec8667831b67344"),
        ),
    },
    "sm_107a": {
        ("bwd", 520, 680): (
            (32768, 131071, "cake_causal_conv1d_train_175b82ccd2309f434dfb"),
            (131072, 2147483647, "cake_causal_conv1d_train_3293c0e34e70413fb4af"),
        ),
    },
}


def sm_wide_segments(
    arch: str, kind: str, C: int
) -> tuple[tuple[int, int, str], ...] | None:
    """The ``((S_lo, S_hi, module), ...)`` segments of the :data:`SM_WIDE_ROWS` key of ``arch`` on route ``kind``
    covering ``C`` (``C_lo <= C <= C_hi``, ``C`` a multiple of 8), ``None`` without one."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    for (route, c_lo, c_hi), segments in SM_WIDE_ROWS.get(arch, {}).items():
        if route == kind and int(c_lo) <= C <= int(c_hi):
            return tuple((int(s_lo), int(s_hi), str(name)) for s_lo, s_hi, name in segments)
    return None


def sm_wide_module(arch: str, kind: str, C: int, S: int | None) -> str | None:
    """The SM-wide program a call of route ``kind`` on ``arch`` with ``C`` channels and ``S`` tokens launches: the
    module of the :data:`SM_WIDE_ROWS` segment holding ``S`` (both bounds included) of the key covering ``C``,
    ``None`` outside every segment or without a key.  ``S`` of ``None`` where a key covers ``C`` is a
    ``ValueError``; a module the route does not list, or whose rule is not of kind :data:`ROWS_KIND`, is a registry
    error (``RuntimeError``)."""

    segments = sm_wide_segments(arch, kind, C)
    if segments is None:
        return None
    if S is None:
        raise ValueError(
            f"the Cake causal conv1d training {kind!r} route on {arch} for C={int(C)} runs its SM-wide programs at "
            f"S {segments[0][0]} to {segments[-1][1]}; the selection needs S"
        )
    for s_lo, s_hi, name in segments:
        if s_lo <= int(S) <= s_hi:
            if name not in route_modules(arch, kind) or (select_rule(MODULES[name]) or {}).get("kind") != ROWS_KIND:
                raise RuntimeError(
                    f"the SM_WIDE_ROWS segment ({s_lo}, {s_hi}) of route {kind!r} on {arch} names {name}, which is "
                    f"no record of the route with a {ROWS_KIND!r} selection rule"
                )
            return name
    return None


def row_sides(arch: str, kind: str, C: int) -> tuple[tuple[int, int], ...]:
    """The token ranges ``((s_lo, s_hi), ...)`` (ascending, disjoint, from 0 up, the last to ``sys.maxsize``) over
    each of which the calls of route ``kind`` on ``arch`` with ``C`` channels select by the :data:`PAIR_ROWS` range,
    the :data:`DIRECT_ROWS` range, one :data:`SM_WIDE_ROWS` segment, the :data:`FALLBACK_ROWS` range or the rules: the
    bounds of every segment of the key covering ``C`` and of the pair, direct and fallback ranges covering ``C`` cut
    the token axis (a pair range may lie across a direct or fallback range: its bounds cut that range; a backward
    direct range cuts the fallback range of the same ``C``).  One range ``((0, sys.maxsize),)`` without a key.  Inside
    a range :func:`selected_modules` picks one program and :func:`plan` one pair, direct or rule plan, but where a
    token floor of the rules decides the program outside the tables (:func:`tokens_select`)."""

    ranges = [] if (bounds := fallback_row_range(arch, kind, C)) is None else [bounds]
    ranges += [] if (direct := direct_row_range(arch, kind, C)) is None else [direct]
    ranges += [] if (pair := pair_row_range(arch, kind, C)) is None else [pair]
    ranges += [(s_lo, s_hi) for s_lo, s_hi, _ in sm_wide_segments(arch, kind, C) or ()]
    starts = {0}
    for s_lo, s_hi in ranges:
        starts |= {max(int(s_lo), 0), max(int(s_hi) + 1, 0)}
    ordered = sorted(starts)
    return tuple(zip(ordered, [*(start - 1 for start in ordered[1:]), sys.maxsize]))


def selected_modules(
    arch: str, kind: str, *, C: int, S: int | None = None
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``C`` channels and ``S`` tokens.

    A route may carry several programs of its kernel and launches exactly one:
    at a row of :data:`PAIR_ROWS` (its range holds ``S``) the range's pair
    direct program (:func:`pair_module`); else at a row of :data:`DIRECT_ROWS`
    the range's direct program (:func:`direct_module`); else at a row of
    :data:`SM_WIDE_ROWS` (a segment holds ``S``) the segment's SM-wide program
    (:func:`sm_wide_module`); else at a row of :data:`FALLBACK_ROWS`
    (:func:`fallback_row_range` holds ``S``) the route's single fallback
    record (the one without a rule); elsewhere, of the records whose
    :func:`select_rule` admits the call on ``arch`` the one of the highest
    channel threshold, else the fallback.  The choice is a pure function of
    the architecture, the route, ``C`` and ``S`` -- ``S`` enters only through
    a rule's token floor, an :data:`SM_WIDE_ROWS` segment, a
    :data:`PAIR_ROWS`, :data:`DIRECT_ROWS` or :data:`FALLBACK_ROWS` range, and
    the host never reads the data -- so a given ``(S, C, device)`` always runs
    the same program of a route (``y`` and ``dx`` are that program's bits;
    ``dweight`` / ``dbias`` are summed in an unordered cross-CTA order).  Two
    records of one channel threshold that both reach ``C``, zero or several
    fallbacks, or a segment or range naming no ``rows`` record of the route,
    are registry errors (``RuntimeError``); an unknown rule kind, or ``S`` of
    ``None`` where a token floor decides or an :data:`SM_WIDE_ROWS`,
    :data:`PAIR_ROWS`, :data:`DIRECT_ROWS` or :data:`FALLBACK_ROWS` key covers
    ``C``, is a ``ValueError``.
    """

    rows = pair_module(arch, kind, C, S) or direct_module(arch, kind, C, S)
    if rows is not None:
        return (rows,)
    sm_wide = sm_wide_module(arch, kind, C, S)
    if sm_wide is not None:
        return (sm_wide,)
    candidates, fallback = _ranked_candidates(arch, kind, C)
    bounds = fallback_row_range(arch, kind, C)
    if bounds is not None:
        if S is None:
            raise ValueError(
                f"the Cake causal conv1d training {kind!r} route on {arch} for C={int(C)} runs its fallback "
                f"program at S {bounds[0]} to {bounds[1]}; the selection needs S"
            )
        if bounds[0] <= int(S) <= bounds[1]:
            return (fallback,)
    for name in candidates:
        if _admits(select_rule(MODULES[name]), arch=arch, C=C, S=S):
            return (name,)
    return (fallback,)


def _ranked_candidates(arch: str, kind: str, C: int) -> tuple[tuple[str, ...], str]:
    """``(candidates, fallback)`` of :func:`selected_modules`: the records of ``kind`` on ``arch`` whose channel
    threshold ``C`` reaches, highest threshold first, and the route's fallback record (registry errors as there).
    A record of kind :data:`ROWS_KIND` has no threshold: it is neither a candidate nor the fallback."""

    names = route_modules(arch, kind)
    candidates: list[tuple[int, str]] = []
    fallback: list[str] = []
    for name in names:
        rule = select_rule(MODULES[name])
        if rule is None:
            fallback.append(name)
            continue
        threshold = _threshold(rule, arch=arch)
        if threshold is not None and int(C) >= threshold:
            candidates.append((threshold, name))
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} declares {len(fallback)} "
            "fallback programs (records without a select_rule); expected exactly one"
        )
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        tied = [name for threshold, name in candidates if threshold == candidates[0][0]]
        raise RuntimeError(
            f"the exported route {kind!r} on {arch} selects {len(tied)} "
            f"programs for C={int(C)} at one threshold {candidates[0][0]} ({tied}); "
            "thresholds must be distinct"
        )
    return tuple(name for _threshold_c, name in candidates), fallback[0]


def tokens_select(arch: str, kind: str, C: int) -> bool:
    """Whether a token floor of the selection rules decides :func:`selected_modules` for ``C`` channels of
    ``kind`` on ``arch``.

    A record whose channel threshold ``C`` reaches admits the call unless its
    rule has a token floor, so the rules read ``S`` exactly when the
    highest-threshold record ``C`` reaches has one (in this registry the
    sm_107a ``bwd`` route at ``256 <= C < 512``); every other row's rules
    select one program whatever ``S``.  An :data:`SM_WIDE_ROWS`, a
    :data:`PAIR_ROWS`, a :data:`DIRECT_ROWS` and a :data:`FALLBACK_ROWS` key
    read ``S`` too, as
    range compares (:func:`row_sides`; the row plan of each side is
    :func:`plan_row` ``at=`` a token count of the side).
    """

    candidates, _ = _ranked_candidates(arch, kind, C)
    return bool(candidates) and (
        _tokens(select_rule(MODULES[candidates[0]]), arch=arch) is not None
    )


def backward_kinds(arch: str) -> tuple[str, ...]:
    """The backward routes ``arch`` exports, in :data:`KINDS` order."""

    return tuple(
        kind
        for kind in KINDS
        if route_role(kind) == "bwd" and ROUTES.get(arch, {}).get(kind)
    )


def workspace_owners(arch: str) -> tuple[dict[str, Any], ...]:
    """The backward module records of ``arch`` (each owns an ``accumulators`` workspace), over every backward route
    in :data:`KINDS` order, each route in launch order.

    Every backward program declares an ``accumulators`` rule; an architecture
    without a backward program, or a backward record of another rule, is a
    registry error (``RuntimeError``).
    """

    owners = tuple(
        MODULES[name]
        for kind in backward_kinds(arch)
        for name in route_modules(arch, kind)
    )
    if not owners:
        raise RuntimeError(
            f"the exported backward routes on {arch} declare no workspace owner; "
            "expected at least one"
        )
    for owner in owners:
        rule = owner["workspace_rule"]
        if rule is None or rule["kind"] != "accumulators":
            raise RuntimeError(
                f"the backward program {owner['kernel']} on {arch} declares the workspace rule "
                f"{rule!r}; every backward program of this package owns an 'accumulators' workspace"
            )
    return owners


def workspace_geometry(arch: str) -> dict[str, int]:
    """The build geometry of the backward routes' fallback programs on ``arch``.

    Every backward route has one fallback program (the record without a
    ``select_rule``); fallbacks of different geometries are a registry error
    (``RuntimeError``).
    """

    geometries = []
    for kind in backward_kinds(arch):
        for name in route_modules(arch, kind):
            record = MODULES[name]
            if select_rule(record) is None and record["grid_rule"]["geometry"] not in geometries:
                geometries.append(record["grid_rule"]["geometry"])
    if len(geometries) != 1:
        raise RuntimeError(
            f"the backward fallback programs on {arch} declare {len(geometries)} "
            "build geometries; expected one"
        )
    return dict(geometries[0])


def workspace_record(arch: str) -> dict[str, Any]:
    """The capacity owner of the backward workspace on ``arch``.

    The public workspace serves every backward program of every backward route
    (:func:`workspace_owners`).  The owners of the :func:`workspace_geometry`
    share one build geometry, so for every ``C`` they plan one ``n_ch_tiles``
    and their workspaces differ only in the accumulator bytes per channel,
    ``slots x taps x sizeof(acc)``: the record with the most (a bias
    ``_fp32g`` route's sixteen copies of five FP32 words, 320 bytes per
    channel; :data:`KINDS` then route order break ties) bounds them all.  An
    owner of another geometry (an SM-wide program of :data:`SM_WIDE_ROWS`)
    carries a ``select_rule`` (a fallback of another geometry is a registry
    error, ``RuntimeError``) and is not the capacity owner; its own layout
    (one channel tile, a few accumulator copies) counts in
    :func:`backward_workspace_bytes`, the largest over every owner.
    """

    geometry = workspace_geometry(arch)
    owners = [
        owner
        for owner in workspace_owners(arch)
        if owner["grid_rule"]["geometry"] == geometry
    ]
    best = owners[0]
    for owner in owners[1:]:
        if _accumulator_bytes_per_channel(owner) > _accumulator_bytes_per_channel(best):
            best = owner
    return best


def grid_mult(record: dict[str, Any], arch: str) -> int:
    """The over-decomposition factor of an ``occupancy`` record on ``arch`` (1 under every other grid rule).

    ``grid_rule["grid_mult"]`` maps each architecture of the record to the
    factor its grid multiplies the co-resident CTA count by (1 for every
    program of this package: the det package's measured factor, kept).
    """

    rule = record["grid_rule"]
    if rule["kind"] != "occupancy":
        return 1
    factor = int(rule["grid_mult"][arch])
    if factor < 1:
        raise ValueError(
            f"module {record['kernel']}: grid_mult[{arch!r}] must be positive, got {factor}"
        )
    return factor


def grid_per_sm(record: dict[str, Any], arch: str) -> int:
    """CTAs per SM of a ``co_resident`` or ``occupancy`` record's grid on ``arch``:
    ``grid_mult x ctas_per_sm`` (the grid is this times the SM count)."""

    return grid_mult(record, arch) * int(record["grid_rule"]["ctas_per_sm"])


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


# The roles whose programs plan their channel tiles with :func:`channel_split`
# (the KCP programs plan with ``legacy=True``): each has its own rule.
SPLIT_ROLES = ("fwd", "bwd")
# The forward's narrow rows: the deterministic package's forward rule over the
# same forward programs, from a token floor of the call's architecture.  On a
# route of :data:`FORWARD_NARROW_ROUTES` the single-tile rows at 65 <= cv <=
# 96 (C = 520 to 768) run, from the row's token floor on the call's
# architecture (:func:`forward_narrow_min_s`), 32-vector tiles
# (``ct_vec`` 32, four token sub-tiles, the last tile partial) at
# :data:`FORWARD_NARROW_THREADS` threads, or, from the width floor of
# :data:`FORWARD_WIDTH_FLOORS` on the call's architecture, the whole tiles of
# :data:`FORWARD_NARROW_WIDTHS` (``cv -> ct_vec``, a divisor of ``cv``): C =
# 520 as (13, 9, 5), 528 as (11, 11, 6), 544 as (17, 7, 4), 552 as (23, 5,
# 3), 560 as (14, 9, 5), 576 as (18, 7, 4) and C = 640 as (40, 3, 2); C =
# 536 and 568 run the 32-vector tiles, (32, 4, 3).  One tile of cv vectors
# leaves 128 - cv lanes of every CTA idle over the whole token axis, which
# costs more than the narrower tiles' extra channel tiles only on longer token
# streams: below the row's floor, in a plan made without ``S`` or without
# the architecture, and on an architecture without a floor for the row, the
# row keeps its single tile of cv vectors (the plan of the package's 0.2.0
# release); a call without a route keeps it too.
# From the floor the plan is the deterministic forward's (which runs the
# 32-vector tiles of the rows from 72 vectors below their width floor at
# every ``S``, and the rows below 72 vectors from the same row floors).  The
# programs are unchanged and ``y`` keeps its bits in every plan.  The narrow
# tables are the deterministic package's forward rows
# (``NARROW_TILE_ROUTES["fwd"]``, ``NARROW_TILE_THREADS``,
# ``NARROW_TILE_WIDTHS["fwd"]``, ``NARROW_WIDTH_FLOORS["fwd"]`` and
# ``NARROW_ROW_FLOORS["fwd"]``, over the forward programs both packages run)
# and are kept in sync with them and with the forward kernel generator;
# :data:`FORWARD_NARROW_MIN_S`, :data:`FORWARD_NARROW_ROW_MIN_S`
# and :data:`FORWARD_SINGLE_TILE_ROWS` are this package's own tables.  A
# row's floor is :func:`forward_narrow_min_s`: for a row below 72 vectors its
# :data:`FORWARD_NARROW_ROW_FLOORS` floor on the architecture; else the
# architecture's floor, or the floor of :data:`FORWARD_NARROW_ROW_MIN_S` for
# the rows of its routes and tap dtypes from a channel-vector count upward.
# The rows below 72 vectors (C = 520 to 568): each row's floor is the smallest
# measured token count from which no measured cell -- any forward route; this
# package or the det package; the forward, the training step and the
# checkpointed step; eager and under CUDA Graph replay -- ran slower with the
# narrower tiles than with the single tile beyond the timing noise, with at
# least one faster cell at every measured token count from it; untimed token
# counts below it keep the single tile.  The measured cells are the tile
# probes and the release timings (this package's forward against the 0.2.3
# package's on the same GPU, and the same for the det package); GB200 and
# GB300 cells are pooled for sm_100a and sm_103a.  The slower cells next below
# the floors: on GB200 and GB300 C = 520, 528, 544 and 560 at 16384; on the
# VR200 C = 568 at 12288, C = 552 and 560 at 16384, C = 544 at 20480, C = 528
# at 31292 and C = 520 at 36864.  The release timings set the floors of
# C = 520 and 544, and of C = 528 on the VR200; their slower cells below those
# floors (the 0.2.3 time over this package's, eager / under CUDA Graph replay
# where both modes were slower, else the mode is named; the cells of this
# package unless det, the det package, is named): on GB200 and GB300, C = 520
# at S = 16384 (GB200, FP32 taps and bias: forward 0.9684 / 0.9743); floor
# 20480. C = 544 at S = 16384 (GB200, FP32 taps and bias: forward
# 0.9801 / 0.9754); floor 20480. On the VR200, C = 520 at S = 24576 (FP32 taps
# and bias: forward 0.9876 / 0.9732), S = 31292 (FP32 taps and bias: forward
# 0.9690 / 0.9632, det forward 0.9784 / 0.9736; FP32 taps, no bias: step
# 0.9796 / 0.9828), S = 32767 (FP32 taps and bias: forward 0.9885 / 0.9827)
# and S = 36864 (FP32 taps and bias: forward 0.9910 / 0.9864); floor 40960.
# C = 528 at S = 24576 (FP32 taps, no bias: forward graph 0.9855, step eager
# 0.9843, checkpointed step eager 0.9837) and S = 31292 (FP32 taps, no bias:
# forward 0.9524 / 0.9549, step 0.9725 / 0.9735, checkpointed step eager
# 0.9840, det forward 0.9611 / 0.9580, det step 0.9822 / 0.9820, det
# checkpointed step eager 0.9876); floor 62585. C = 544 at S = 20480 (FP32
# taps and bias: forward 0.9862 / 0.9775); floor 24576. The width floor of a
# row below 72 vectors is its row floor.
# On sm_103a the ``fwd_bias`` rows of 89 to 96 vectors (C = 712 to 768) read
# the dtype of the call's taps (``bf16_taps``; the ``fwd`` route's taps are
# FP32 and the ``fwd_dt`` route's BF16): with FP32 taps they take 20480 (on a
# GB300 at 16384 x 768 the 32-vector tiles of that route under CUDA Graph
# replay were slower than the single tile in two measurements each: FP32
# taps, BF16 bias 17.14 us single tile, 17.41 / 17.38 us narrow; FP32 taps,
# FP32 bias 17.12 vs 17.36 / 17.41 us), with BF16 taps 12288 but in the
# token ranges of :data:`FORWARD_SINGLE_TILE_ROWS`, where the single tile
# stays: on a GB300 with BF16 taps and bias the 32-vector tiles ran 1.02 to
# 1.10 times as fast as the single tile at 12288 and from 16384 to 20479 at
# C = 712, 736, 760 and 768, eager and under CUDA Graph replay, but not
# faster between them (from 13312 to 15360 at C = 712, from 13824 to 15360
# at C = 768, at 14336 at C = 736 and 760); each range runs from the token
# count after the last timed faster one to the one before the next.  The
# ``fwd`` and ``fwd_dt`` rows keep the architecture's 12288: under CUDA Graph
# replay the default route ran 1.025 / 1.028 times as fast at 12288 x 768
# with the 32-vector tiles (14.90 us single tile, 14.48 / 14.50 us narrow),
# and the BF16-taps calls 1.003 to 1.032 times as fast at 16384 x 768.  C =
# 704 (88 vectors) keeps 12288 on every route: on the default route its
# 32-vector tiles were not slower than the single tile in either measurement
# at any timed token count from 12288 to 250340.  C = 736 (92 vectors) was
# timed on the ``fwd`` and ``fwd_dt`` routes on the GB300: under CUDA Graph
# replay the 32-vector tiles ran 1.023 / 1.023 times as fast as the single
# tile at 12288 / 16384 x 736 on the default route (14.40 -> 14.08 us, 16.54
# -> 16.18 us) and 1.030 / 1.014 times as fast with BF16 taps (14.05 -> 13.63
# us, 16.00 -> 15.78 us).  The untimed channel counts from C = 712 to 760 take
# the floors and gaps of their measured neighbours.
# The backward plans its narrower rows by :data:`TILE_OVERRIDES` and
# :data:`NARROW_TILES` instead.
FORWARD_NARROW_ROUTES: tuple[str, ...] = ("fwd", "fwd_dt", "fwd_bias")
FORWARD_NARROW_THREADS = 128
FORWARD_NARROW_WIDTHS: dict[int, int] = {65: 13, 66: 11, 68: 17, 69: 23, 70: 14, 72: 18, 80: 40}
FORWARD_WIDTH_FLOORS: dict[str, dict[int, int]] = {
    "sm_100a": {65: 20480, 66: 24576, 68: 20480, 69: 16384, 70: 24576, 72: 32768, 80: 24576},
    "sm_103a": {65: 20480, 66: 24576, 68: 20480, 69: 16384, 70: 24576, 72: 32768, 80: 24576},
    "sm_107a": {65: 40960, 66: 62585, 68: 24576, 69: 24576, 70: 24576, 72: 40960, 80: 36864},
}
# ``{arch: {cv: S}}``: on ``arch`` the narrow forward row of ``cv`` channel
# vectors below 72 (C = 520 to 568) plans as the deterministic forward from
# ``S`` tokens, on every forward route; a row of fewer than 72 vectors that
# the architecture does not list keeps the single tile at every ``S``.
FORWARD_NARROW_ROW_FLOORS: dict[str, dict[int, int]] = {
    "sm_100a": {65: 20480, 66: 24576, 67: 16384, 68: 20480, 69: 16384, 70: 24576, 71: 12288},
    "sm_103a": {65: 20480, 66: 24576, 67: 16384, 68: 20480, 69: 16384, 70: 24576, 71: 12288},
    "sm_107a": {65: 40960, 66: 62585, 67: 16384, 68: 24576, 69: 24576, 70: 24576, 71: 16384},
}
# The rows below this many channel vectors take :data:`FORWARD_NARROW_ROW_FLOORS`.
FORWARD_ROW_FLOOR_VECTORS = 72
FORWARD_NARROW_MIN_S: dict[str, int] = {"sm_100a": 20480, "sm_103a": 12288, "sm_107a": 16384}
# The tap dtypes a forward row of :data:`FORWARD_NARROW_ROW_MIN_S` and
# :data:`FORWARD_SINGLE_TILE_ROWS` names: the ``fwd`` route's taps are FP32,
# the ``fwd_dt`` route's BF16, and a ``fwd_bias`` call's those of its tap
# tensor (``bf16_taps``).
FORWARD_TAPS: tuple[str, ...] = ("f32", "bf16")
# ``{arch: ((cv, S, routes, taps), ...)}``, ``cv`` non-decreasing: on ``arch``
# the narrow forward rows of at least ``cv`` channel vectors (and at least 72)
# on a forward route of ``routes`` whose taps (:func:`forward_taps`) are in
# ``taps`` take the token floor ``S`` instead of the architecture's
# :data:`FORWARD_NARROW_MIN_S` floor (of the entries a row reaches on its
# route and taps, the last wins); read through :func:`forward_narrow_min_s`.
# ``routes`` is a non-empty tuple of :data:`FORWARD_NARROW_ROUTES` and
# ``taps`` of :data:`FORWARD_TAPS`, each in that order.
FORWARD_NARROW_ROW_MIN_S: dict[str, tuple[tuple[int, int, tuple[str, ...], tuple[str, ...]], ...]] = {
    "sm_103a": ((89, 20480, ("fwd_bias",), ("f32",)), (89, 12288, ("fwd_bias",), ("bf16",))),
}
# ``{arch: ((cv_lo, cv_hi, routes, taps, S_lo, S_hi), ...)}``: on ``arch`` the
# narrow forward rows of ``cv_lo`` to ``cv_hi`` channel vectors on a route of
# ``routes`` whose taps are in ``taps`` keep the single tile at ``S_lo <= S <=
# S_hi`` tokens, at or above the row's floor (the first entry a row reaches
# wins); read through :func:`forward_single_tile_range`.  The entries of one
# architecture and route share no ``(cv, taps)``.
FORWARD_SINGLE_TILE_ROWS: dict[
    str, tuple[tuple[int, int, tuple[str, ...], tuple[str, ...], int, int], ...]
] = {
    "sm_103a": (
        (89, 89, ("fwd_bias",), ("bf16",), 12961, 16383),
        (90, 95, ("fwd_bias",), ("bf16",), 12289, 16383),
        (96, 96, ("fwd_bias",), ("bf16",), 13313, 16383),
    ),
}


def forward_taps(route: str, *, bf16_taps: bool = False) -> str:
    """The taps (one of :data:`FORWARD_TAPS`) of a forward call on route ``route`` (one of
    :data:`FORWARD_NARROW_ROUTES`, a ``ValueError`` otherwise): ``"f32"`` on ``fwd``, ``"bf16"`` on ``fwd_dt``, and on
    ``fwd_bias`` ``"bf16"`` with ``bf16_taps``, else ``"f32"``."""

    if route not in FORWARD_NARROW_ROUTES:
        raise ValueError(f"route must be one of {FORWARD_NARROW_ROUTES}, got {route!r}")
    if route == "fwd_bias":
        return "bf16" if bf16_taps else "f32"
    return "bf16" if route == "fwd_dt" else "f32"


def forward_narrow_min_s(
    arch: str | None, channel_vectors: int, *, route: str, bf16_taps: bool = False
) -> int | None:
    """The token floor from which the forward row of ``channel_vectors``
    vectors on route ``route`` (one of :data:`FORWARD_NARROW_ROUTES`, a
    ``ValueError`` otherwise) on ``arch`` plans as the deterministic forward:
    below 72 vectors (:data:`FORWARD_ROW_FLOOR_VECTORS`) the row's
    :data:`FORWARD_NARROW_ROW_FLOORS` floor on ``arch`` (``None`` where
    ``arch`` lists none); else the floor of the last
    :data:`FORWARD_NARROW_ROW_MIN_S` entry of ``arch`` that lists ``route``
    and the call's taps (:func:`forward_taps`; ``bf16_taps`` is read on
    ``fwd_bias`` only) and whose channel-vector count the row reaches, else
    the architecture's :data:`FORWARD_NARROW_MIN_S` floor; ``None`` (the
    single tile at every ``S``) on an architecture
    :data:`FORWARD_NARROW_MIN_S` does not list.  The per-call plan
    (:func:`_forward_tiles`), the row plan (:func:`plan_row`) and the tests
    read the floor here."""

    taps = forward_taps(route, bf16_taps=bf16_taps)
    cv = int(channel_vectors)
    if cv < FORWARD_ROW_FLOOR_VECTORS:
        floor = FORWARD_NARROW_ROW_FLOORS.get(arch, {}).get(cv)
        return None if floor is None else int(floor)
    floor = FORWARD_NARROW_MIN_S.get(arch)
    if floor is None:
        return None
    for cv_from, S, routes, row_taps in FORWARD_NARROW_ROW_MIN_S.get(arch, ()):
        if cv >= int(cv_from) and route in routes and taps in row_taps:
            floor = S
    return int(floor)


def forward_single_tile_range(
    arch: str | None, channel_vectors: int, *, route: str, bf16_taps: bool = False
) -> tuple[int, int] | None:
    """The ``(S_lo, S_hi)`` of the :data:`FORWARD_SINGLE_TILE_ROWS` entry of ``arch`` covering the forward row of
    ``channel_vectors`` vectors on route ``route`` (one of :data:`FORWARD_NARROW_ROUTES`, a ``ValueError``
    otherwise) and the call's taps (:func:`forward_taps`), ``None`` without one: inside that token range (both
    bounds included) the row keeps its single tile."""

    taps = forward_taps(route, bf16_taps=bf16_taps)
    cv = int(channel_vectors)
    for cv_lo, cv_hi, routes, row_taps, s_lo, s_hi in FORWARD_SINGLE_TILE_ROWS.get(arch, ()):
        if int(cv_lo) <= cv <= int(cv_hi) and route in routes and taps in row_taps:
            return int(s_lo), int(s_hi)
    return None


def channel_split(
    C: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
    route: str | None = None,
    S: int | None = None,
    arch: str | None = None,
    bf16_taps: bool = False,
) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of ``C`` channels under a record's build geometry, role, route, token count,
    architecture and (on ``fwd_bias``) tap dtype.

    ``ct_vec`` channel vectors per channel tile (one thread each), ``n_sub =
    threads // ct_vec`` token sub-tiles per CTA and ``n_ch_tiles = ceil(C / vec
    / ct_vec)`` channel tiles.  ``role`` is the record's, ``"fwd"`` or
    ``"bwd"`` (a ``ValueError`` without one unless ``legacy``): the two rules
    agree except in the fallback below.  With ``cv = C / vec`` channel
    vectors: ``ct_vec = cv`` when ``cv <= threads``; above that, the multiple
    of 32 dividing ``cv`` (at most ``threads``) with the highest lane use
    ``ct_vec x (threads // ct_vec)``, the larger ``ct_vec`` on a tie.  So C =
    1280 (160 vectors) runs five 32-vector tiles with four token sub-tiles
    each, (32, 4, 5), and C = 1536 (64, 2, 3) -- every lane busy -- instead of
    two 128-vector tiles at 62 % / 75 % lane use; C <= 1024 and 2048 / 4096 /
    8192 are unchanged.  When no multiple of 32 divides ``cv``, the role's
    fallback picks the ``ct_vec`` of the highest lane use over the launch,
    ``cv / (n_ch_tiles x ct_vec) x (ct_vec x (threads // ct_vec) / threads)``
    rounded to 9 decimals (the larger on a tie), among every multiple of 32 at
    most ``threads`` (the last tile partial) and, for the forward, every
    divisor of ``cv`` from 40 to ``threads`` (whole tiles).  At 128 x 8 the
    backward runs C = 1160 as (32, 4, 5), 1344 as (64, 2, 3) and 1600 as (32,
    4, 7), the forward 1160 as (32, 4, 5), 1344 as (42, 3, 4) and 1600 as (40,
    3, 5); both run 1920 as (128, 1, 2).  The per element arithmetic does not
    depend on the tiling, so ``y`` and ``dx`` are bitwise the legacy plan's
    (``dweight``'s CTA rows follow the tiling).  ``legacy=True`` is the
    rule an earlier revision shipped, ``ct_vec = min(cv, threads)``,
    for either role; the KCP path keeps it (its kernels pick the grouped path
    from ``ct_vec`` on the device and are measured under this rule).  Each
    role's rule is kept in sync with its copies (the kernel generator of
    that role and the test plan of that kind).

    ``route`` is the launch's route (one of :data:`KINDS` of the role, a
    ``ValueError`` otherwise; the host passes its record's).  On a route of
    :data:`FORWARD_NARROW_ROUTES` the forward's single-tile rows at C = 520 to
    768 run 32-vector tiles, ``(32, threads // 32, ceil(cv / 32))``, when the
    call names its token count ``S`` and its architecture ``arch`` (one of
    :data:`ARCHES`, a ``ValueError`` otherwise) and ``S`` is at least the
    row's floor on its route, ``arch`` and taps (:func:`forward_narrow_min_s`:
    below 72 vectors the row's :data:`FORWARD_NARROW_ROW_FLOORS` floor, else
    the architecture's :data:`FORWARD_NARROW_MIN_S` floor or a
    :data:`FORWARD_NARROW_ROW_MIN_S` floor of the route and taps) and outside
    the row's :data:`FORWARD_SINGLE_TILE_ROWS` range
    (:func:`forward_single_tile_range`): C = 536, 568, 576, 640 and 768 as
    (32, 4, 3) instead of one tile of 67, 71, 72, 80 and 96 vectors.  A row of
    :data:`FORWARD_NARROW_WIDTHS` runs whole tiles of its width instead from
    its floor on ``arch`` in :data:`FORWARD_WIDTH_FLOORS` (and the row's
    floor): C = 520 as (13, 9, 5), 576 as (18, 7, 4) and C = 640 as (40, 3,
    2).  ``bf16_taps`` (the call's taps are BF16) is read on ``fwd_bias``
    only: the ``fwd`` route's taps are FP32 and the ``fwd_dt`` route's BF16.
    Without ``S`` or ``arch``, below the row's floor on its route, ``arch``
    and taps, inside its single-tile range and on an architecture without a
    floor for the row, the single tile stays (the 0.2.0 plan); ``y`` keeps
    its bits in every plan.  Without a route, under
    ``legacy`` and for the backward role the split above stays (the
    backward's narrower rows are :func:`plan`'s :data:`TILE_OVERRIDES` and
    :data:`NARROW_TILES`).  From the row's floor on its route and ``arch`` (and on every
    other row) this is the deterministic package's ``channel_split`` for the
    forward role.
    """

    if arch is not None and arch not in ARCHES:
        raise ValueError(f"arch must be one of {ARCHES}, got {arch!r}")
    vec = int(geometry["vec"])
    return _channel_split(
        int(C),
        vec,
        int(geometry["threads"]),
        role,
        bool(legacy),
        route,
        _forward_tiles(int(C) // vec, role, bool(legacy), route, S, arch, bool(bf16_taps)),
    )


# The forward tiles of a narrow row (:func:`_forward_tiles`): one tile of cv
# vectors, the 32-vector tiles, or the row's :data:`FORWARD_NARROW_WIDTHS`.
FORWARD_TILES = ("one", "narrow", "wide")


def _forward_tiles(
    channel_vectors: int,
    role: str | None,
    legacy: bool,
    route: str | None,
    S: int | None,
    arch: str | None,
    bf16_taps: bool = False,
) -> str:
    """Which tiles of :data:`FORWARD_TILES` a narrow forward row runs: for a
    forward call on a route of :data:`FORWARD_NARROW_ROUTES` that names ``S``
    and ``arch``, with ``S`` at least the row's floor on that route, ``arch``
    and taps (:func:`forward_narrow_min_s`) and outside the row's single-tile
    range (:func:`forward_single_tile_range`), ``"wide"`` when the row has a
    width in :data:`FORWARD_NARROW_WIDTHS` and ``S`` is at least its floor on
    ``arch`` in :data:`FORWARD_WIDTH_FLOORS`, else ``"narrow"``; ``"one"``
    otherwise.  Only the rows :func:`_forward_narrow_tile` names read it.  The
    memoized plan takes this word, not ``S`` or the taps, so its keys stay at
    most three per row."""

    if legacy or S is None or arch is None or role != "fwd":
        return "one"
    if route not in FORWARD_NARROW_ROUTES:
        return "one"
    min_s = forward_narrow_min_s(arch, channel_vectors, route=route, bf16_taps=bf16_taps)
    if min_s is None or int(S) < min_s:
        return "one"
    gap = forward_single_tile_range(arch, channel_vectors, route=route, bf16_taps=bf16_taps)
    if gap is not None and gap[0] <= int(S) <= gap[1]:
        return "one"
    floor = FORWARD_WIDTH_FLOORS.get(arch, {}).get(channel_vectors)
    if channel_vectors in FORWARD_NARROW_WIDTHS and floor is not None and int(S) >= floor:
        return "wide"
    return "narrow"


@functools.lru_cache(maxsize=4096)
def _channel_split(
    C: int,
    vec: int,
    threads: int,
    role: str | None,
    legacy: bool,
    route: str | None = None,
    tiles: str = "one",
) -> tuple[int, int, int]:
    """:func:`channel_split` of plain arguments, memoized: a model's channel
    counts are few, so every call after a channel count's first reuses its
    plan (a backward call plans its row up to twice).  ``tiles`` is
    :func:`_forward_tiles`: the tiles a narrow forward row runs (one of
    :data:`FORWARD_TILES`; ``"wide"`` on a row without a width runs the
    32-vector tiles)."""

    if (role is None and not legacy) or (role is not None and role not in SPLIT_ROLES):
        raise ValueError(
            f"role must be one of {SPLIT_ROLES} (or legacy=True), got {role!r}"
        )
    if route is not None and (
        route not in KINDS or role is None or route_role(route) != role
    ):
        raise ValueError(f"route={route!r} is not a route of role {role!r}")
    if C <= 0 or C % vec:
        raise ValueError(f"C={C} must be a positive multiple of {vec}")
    channel_vectors = C // vec
    if legacy or channel_vectors <= threads:
        ct_vec = min(channel_vectors, threads)
        if (
            not legacy
            and role == "fwd"
            and route in FORWARD_NARROW_ROUTES
            and tiles != "one"
            and _forward_narrow_tile(channel_vectors, threads)
        ):
            ct_vec = (
                FORWARD_NARROW_WIDTHS.get(channel_vectors, 32) if tiles == "wide" else 32
            )
    else:
        candidates = [
            width
            for width in range(32, threads + 1, 32)
            if channel_vectors % width == 0
        ]
        ct_vec = (
            max(candidates, key=lambda width: (width * (threads // width), width))
            if candidates
            else _fallback_split(channel_vectors, threads, role)
        )
    return ct_vec, threads // ct_vec, _ceil_div(channel_vectors, ct_vec)


def _fallback_split(channel_vectors: int, threads: int, role: str) -> int:
    """``ct_vec`` of :func:`channel_split` above ``threads`` vectors with no 32-multiple divisor.

    The candidates are every multiple of 32 at most ``threads`` (the last tile
    partial: its copies and stores stop at channel ``C``) and, for the
    forward, every divisor of ``channel_vectors`` from 40 to ``threads`` (whole
    tiles); the one of the highest :func:`_lane_use`, rounded to 9 decimals,
    wins, the larger on a tie.  No tile is narrower than a warp's 32 vectors
    and no whole forward tile narrower than 40.
    """

    candidates = set(range(32, threads + 1, 32))
    if role == "fwd":
        candidates |= {
            width for width in range(40, threads + 1) if channel_vectors % width == 0
        }
    return max(
        candidates,
        key=lambda width: (
            round(_lane_use(channel_vectors, width, threads), 9),
            width,
        ),
    )


def _forward_narrow_tile(channel_vectors: int, threads: int) -> bool:
    """Whether a forward single-tile row runs narrower tiles on the routes of :data:`FORWARD_NARROW_ROUTES`."""

    return threads == FORWARD_NARROW_THREADS and 65 <= channel_vectors <= 96


def _lane_use(channel_vectors: int, width: int, threads: int) -> float:
    """The busy share of a CTA's lanes over ``ceil(channel_vectors / width)`` tiles of ``width``."""

    return (
        channel_vectors
        / (_ceil_div(channel_vectors, width) * width)
        * (width * (threads // width) / threads)
    )


# The narrower channel tiles of the backward programs, ``{arch: {(route, C):
# (ct_vec, S_min, S_max)}}``: at exactly these channel counts, on these routes
# and at ``S_min <= S <= S_max`` a backward call runs ``ct_vec``-vector tiles
# instead of :func:`channel_split`'s wider ones (:func:`tile_override`).  More
# tiles split the tail of a launch: each tile's accumulator words take the adds
# of fewer CTAs, and more tiles' last CTAs share the finish.  ``dx`` does not
# depend on the tiling; ``dweight`` / ``dbias`` add in another partition.  The
# workspace takes one ticket per tile (:func:`backward_workspace_bytes` covers
# the largest tile count of every row, whatever ``S``).  Written by the exporter
# from the backward kernel generator's table; every other channel count, route,
# ``S`` and architecture keeps :func:`channel_split` (or the tiles of a
# :data:`NARROW_TILES` row).  Inside a backward range of :data:`DIRECT_ROWS` the
# route's direct program runs its own tiles instead (:func:`direct_split`).
TILE_OVERRIDES: dict[str, dict[tuple[str, int], tuple[int, int, int]]] = {
    "sm_100a": {
        ("bwd_bias", 768): (32, 62585, 500680),
        ("bwd_bias_fp32g", 768): (32, 31292, 500680),
        ("bwd_dt", 768): (32, 62585, 500680),
        ("bwd_fp32g", 768): (32, 31292, 500680),
        ("bwd_bias_fp32g", 1536): (32, 31292, 500680),
        ("bwd_dt", 1536): (32, 31292, 125170),
        ("bwd_fp32g", 1536): (32, 31292, 500680),
        ("bwd_bias_fp32g", 2048): (32, 31292, 500680),
        ("bwd_fp32g", 2048): (32, 31292, 500680),
        ("bwd_bias_fp32g", 2560): (32, 31292, 125170),
        ("bwd_fp32g", 2560): (32, 31292, 125170),
        ("bwd", 3072): (32, 31292, 125170),
        ("bwd_bias", 3072): (64, 31292, 500680),
        ("bwd_bias_fp32g", 3072): (64, 31292, 500680),
        ("bwd_dt", 3072): (64, 31292, 500680),
        ("bwd_fp32g", 3072): (64, 31292, 500680),
    },
    "sm_103a": {
        ("bwd_bias", 768): (32, 62585, 500680),
        ("bwd_bias_fp32g", 768): (32, 31292, 500680),
        ("bwd_dt", 768): (32, 62585, 500680),
        ("bwd_fp32g", 768): (32, 62585, 500680),
        ("bwd", 1536): (32, 31292, 500680),
        ("bwd_bias_fp32g", 1536): (32, 31292, 500680),
        ("bwd_dt", 1536): (32, 31292, 500680),
        ("bwd_fp32g", 1536): (32, 31292, 500680),
        ("bwd_bias_fp32g", 2048): (32, 31292, 500680),
        ("bwd_fp32g", 2048): (32, 31292, 500680),
        ("bwd_bias_fp32g", 2560): (32, 31292, 250340),
        ("bwd_fp32g", 2560): (32, 31292, 250340),
        ("bwd", 3072): (32, 31292, 500680),
        ("bwd_bias", 3072): (64, 31292, 500680),
        ("bwd_bias_fp32g", 3072): (32, 31292, 500680),
        ("bwd_dt", 3072): (32, 31292, 500680),
        ("bwd_fp32g", 3072): (32, 31292, 500680),
        ("bwd_bias_fp32g", 4096): (32, 31292, 500680),
        ("bwd_fp32g", 4096): (32, 31292, 500680),
    },
    "sm_107a": {
        ("bwd_dt", 768): (32, 1, 2147483647),
        ("bwd_bias_fp32g", 1536): (32, 31292, 500680),
        ("bwd_dt", 1536): (32, 31292, 125170),
        ("bwd_fp32g", 1536): (32, 31292, 500680),
        ("bwd_bias_fp32g", 2048): (32, 31292, 500680),
        ("bwd_dt", 2048): (32, 31292, 62585),
        ("bwd_fp32g", 2048): (32, 31292, 500680),
        ("bwd_bias_fp32g", 2560): (32, 62585, 500680),
        ("bwd_dt", 2560): (32, 62585, 500680),
        ("bwd_fp32g", 2560): (32, 62585, 500680),
        ("bwd", 3072): (32, 31292, 62585),
        ("bwd_bias_fp32g", 3072): (32, 31292, 500680),
        ("bwd_dt", 3072): (64, 31292, 500680),
        ("bwd_fp32g", 3072): (32, 31292, 500680),
        ("bwd", 4096): (32, 31292, 62585),
        ("bwd_bias", 4096): (32, 62585, 125170),
        ("bwd_bias_fp32g", 4096): (64, 31292, 500680),
        ("bwd_dt", 4096): (32, 31292, 125170),
        ("bwd_fp32g", 4096): (64, 31292, 500680),
    },
}


def tile_override(arch: str, route: str, C: int, S: int) -> int | None:
    """The ``ct_vec`` of :data:`TILE_OVERRIDES` for a backward call of ``route``
    on ``arch`` with ``C`` channels and ``S`` tokens (``S_min <= S <= S_max``,
    both bounds included), else ``None`` (:func:`channel_split` plans it)."""

    entry = TILE_OVERRIDES.get(arch, {}).get((route, int(C)))
    if entry is None:
        return None
    ct_vec, s_min, s_max = entry
    return int(ct_vec) if int(s_min) <= int(S) <= int(s_max) else None


def _override_split(
    ct_vec: int, C: int, geometry: dict[str, int]
) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of an override's ``ct_vec``: ``threads //
    ct_vec`` token sub-tiles and ``ceil(C / vec / ct_vec)`` channel tiles."""

    threads, vec = int(geometry["threads"]), int(geometry["vec"])
    return ct_vec, threads // ct_vec, _ceil_div(int(C) // vec, ct_vec)


def override_tiles(arch: str, route: str, C: int, geometry: dict[str, int]) -> int:
    """The channel tiles of the :data:`TILE_OVERRIDES` row of ``(arch, route,
    C)`` under ``geometry`` (0 without a row), whatever ``S``: the workspace
    bound of the row's launches."""

    entry = TILE_OVERRIDES.get(arch, {}).get((route, int(C)))
    if entry is None:
        return 0
    return _override_split(int(entry[0]), C, geometry)[2]


# The 32-vector channel tiles of the narrow single-tile backward rows, ``{arch:
# {(route, C_lo, C_hi): (S_min, S_max)}}``: a backward call of ``route`` on
# ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple of 8) and ``S_min
# <= S <= S_max`` tokens, outside a :data:`TILE_OVERRIDES` row, runs ``(32, 4,
# ceil(C / 8 / 32))`` tiles at 128 threads where :func:`channel_split` plans
# one tile of every channel vector (:func:`narrow_tile_row`).  On sm_100a and
# sm_103a the ``bwd_dt`` and ``bwd_bias`` routes take them at C 768 only up to
# a token bound, below the S range of their C 768 :data:`TILE_OVERRIDES`
# rows; between the two the call keeps the one tile.  As at the rows of
# :data:`TILE_OVERRIDES`, each
# tile's accumulator words take the adds of fewer CTAs; ``dx`` does not depend
# on the tiling.  The workspace takes one ticket per tile
# (:func:`backward_workspace_bytes` covers the row's tiles, whatever ``S``).
# Written by the exporter from the backward kernel generator's table; every
# other channel count, route, ``S``, thread count and architecture keeps
# :func:`tile_override`'s or :func:`channel_split`'s plan.  Inside a backward
# range of :data:`DIRECT_ROWS` the route's direct program runs its own tiles
# instead (:func:`direct_split`).
NARROW_TILES: dict[str, dict[tuple[str, int, int], tuple[int, int]]] = {
    "sm_100a": {
        ("bwd", 224, 224): (1, 2147483647),
        ("bwd", 520, 1016): (1, 2147483647),
        ("bwd_dt", 224, 224): (1, 2147483647),
        ("bwd_dt", 520, 760): (1, 2147483647),
        ("bwd_dt", 768, 768): (1, 8192),
        ("bwd_dt", 776, 1016): (1, 2147483647),
        ("bwd_bias", 224, 224): (1, 2147483647),
        ("bwd_bias", 520, 760): (1, 2147483647),
        ("bwd_bias", 768, 768): (1, 16384),
        ("bwd_bias", 776, 1016): (1, 2147483647),
        ("bwd_fp32g", 224, 224): (1, 2147483647),
        ("bwd_fp32g", 520, 760): (1, 2147483647),
        ("bwd_fp32g", 776, 1016): (1, 2147483647),
        ("bwd_bias_fp32g", 224, 224): (1, 2147483647),
        ("bwd_bias_fp32g", 520, 760): (1, 2147483647),
        ("bwd_bias_fp32g", 776, 1016): (1, 2147483647),
    },
    "sm_103a": {
        ("bwd", 224, 224): (1, 2147483647),
        ("bwd", 520, 1016): (1, 2147483647),
        ("bwd_dt", 224, 224): (1, 2147483647),
        ("bwd_dt", 520, 760): (1, 2147483647),
        ("bwd_dt", 768, 768): (1, 16384),
        ("bwd_dt", 776, 1016): (1, 2147483647),
        ("bwd_bias", 224, 224): (1, 2147483647),
        ("bwd_bias", 520, 760): (1, 2147483647),
        ("bwd_bias", 768, 768): (1, 16384),
        ("bwd_bias", 776, 1016): (1, 2147483647),
        ("bwd_fp32g", 224, 224): (1, 2147483647),
        ("bwd_fp32g", 520, 760): (1, 2147483647),
        ("bwd_fp32g", 776, 1016): (1, 2147483647),
        ("bwd_bias_fp32g", 224, 224): (1, 2147483647),
        ("bwd_bias_fp32g", 520, 760): (1, 2147483647),
        ("bwd_bias_fp32g", 776, 1016): (1, 2147483647),
    },
    "sm_107a": {
        ("bwd", 224, 224): (1, 2147483647),
        ("bwd", 520, 1016): (1, 2147483647),
        ("bwd_dt", 224, 224): (1, 2147483647),
        ("bwd_dt", 520, 760): (1, 2147483647),
        ("bwd_dt", 776, 1016): (1, 2147483647),
        ("bwd_bias", 224, 224): (1, 2147483647),
        ("bwd_bias", 520, 1016): (1, 2147483647),
        ("bwd_fp32g", 224, 224): (1, 2147483647),
        ("bwd_fp32g", 520, 1016): (1, 2147483647),
        ("bwd_bias_fp32g", 224, 224): (1, 2147483647),
        ("bwd_bias_fp32g", 520, 1016): (1, 2147483647),
    },
}

# The narrow rows' tile: ``NARROW_CT_VEC`` channel vectors of a CTA of
# ``NARROW_THREADS`` threads (other geometries keep their plan).
NARROW_CT_VEC = 32
NARROW_THREADS = 128


def narrow_tile_range(arch: str, route: str, C: int) -> tuple[int, int] | None:
    """The ``(S_min, S_max)`` of the :data:`NARROW_TILES` key of ``arch`` on
    ``route`` covering ``C`` (``C_lo <= C <= C_hi``, ``C`` a multiple of 8,
    the first such key), ``None`` without one or where the token range of the
    architecture's :data:`TILE_OVERRIDES` row of ``(route, C)`` meets the
    key's (whose rows :func:`tile_override` plans).  A key whose tokens end
    below the row's ``S_min`` (the C 768 keys of the ``bwd_dt`` and
    ``bwd_bias`` routes on sm_100a and sm_103a) gives its range."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    entry = TILE_OVERRIDES.get(arch, {}).get((route, C))
    for (name, c_lo, c_hi), (s_min, s_max) in NARROW_TILES.get(arch, {}).items():
        if name == route and int(c_lo) <= C <= int(c_hi):
            if (
                entry is not None
                and int(s_min) <= int(entry[2])
                and int(entry[1]) <= int(s_max)
            ):
                return None
            return int(s_min), int(s_max)
    return None


def narrow_tile_row(arch: str, route: str, C: int, S: int) -> bool:
    """Whether a backward call of ``route`` on ``arch`` with ``C`` channels and
    ``S`` tokens is a row of :data:`NARROW_TILES` (:func:`narrow_tile_range`
    holds ``S``, both bounds included)."""

    bounds = narrow_tile_range(arch, route, C)
    return bounds is not None and bounds[0] <= int(S) <= bounds[1]


def _narrow_split(
    arch: str, route: str, C: int, geometry: dict[str, int]
) -> tuple[int, int, int] | None:
    """The ``(ct_vec, n_sub, n_ch_tiles)`` of the :data:`NARROW_TILES` key
    covering ``(arch, route, C)`` under ``geometry`` (``NARROW_CT_VEC``-vector
    tiles at ``NARROW_THREADS`` threads), ``None`` without a key or at another
    thread count."""

    if int(geometry["threads"]) != NARROW_THREADS:
        return None
    if narrow_tile_range(arch, route, C) is None:
        return None
    return _override_split(NARROW_CT_VEC, C, geometry)


def narrow_tiles(arch: str, route: str, C: int, geometry: dict[str, int]) -> int:
    """The channel tiles of the :data:`NARROW_TILES` key covering ``(arch,
    route, C)`` under ``geometry`` (0 without one), whatever ``S``: the
    workspace bound of the row's launches."""

    split = _narrow_split(arch, route, C, geometry)
    return 0 if split is None else split[2]


def plan(
    S: int,
    C: int,
    num_docs: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
    arch: str | None = None,
    route: str | None = None,
    bf16_taps: bool = False,
) -> dict[str, int]:
    """The kernel's launch scalars for one problem under a record's build geometry and role.

    ``geometry`` is ``record["grid_rule"]["geometry"]``: ``threads`` per CTA
    and ``vec`` BF16 channels per thread; ``role`` is ``record["role"]``
    (``"fwd"`` or ``"bwd"``).  The scalars are ``S``, ``C``, ``num_docs``,
    :func:`channel_split`'s ``ct_vec`` / ``n_sub`` / ``n_ch_tiles`` (``role``
    and ``legacy`` are passed on) and ``search_steps``, the number of halvings
    of the device-side binary search over ``cu_seqlens`` (``max(1, ceil(log2
    num_docs))``).  Every value is a pure function of ``(S, C, num_docs)``, the
    geometry and the role: the host never reads ``cu_seqlens``, and the kernel
    derives its token ranges from ``S`` and the grid.  The ``flags`` scalar of
    the ``dtype_flags`` programs is not a plan value: the host derives it from
    the tensors of the call.  Given the backward record's ``arch`` and
    ``route`` (:func:`record_route`), a range of :data:`DIRECT_ROWS`
    (:func:`direct_row`, where the route's direct program runs) replaces the
    split by the route's direct tiles (:func:`direct_split`, at
    :data:`BACKWARD_DIRECT_THREADS` threads), else a row of
    :data:`TILE_OVERRIDES` (:func:`tile_override`) by ``(ct_vec, threads //
    ct_vec, ceil(C / vec / ct_vec))``, else a row of :data:`NARROW_TILES`
    (:func:`narrow_tile_row`, at 128 threads) by ``(32, threads // 32, ceil(C
    / vec / 32))``; without them and under ``legacy`` the split is
    :func:`channel_split`'s.  For the forward role ``route``, ``S`` and
    ``arch`` are passed on to :func:`channel_split` (not under ``legacy``),
    with ``bf16_taps`` (the call's taps are BF16; read on ``fwd_bias`` only):
    from the row's floor on the record's route, ``arch`` and taps
    (:func:`forward_narrow_min_s`) the forward record's route runs the
    32-vector tiles of its narrow rows (but inside the row's
    :func:`forward_single_tile_range`), and a row of
    :data:`FORWARD_NARROW_WIDTHS` its width from its token floor on ``arch``,
    as the deterministic package's forward plans them; below that floor the
    row keeps its single tile.  Inside a :data:`PAIR_ROWS` or
    :data:`DIRECT_ROWS` range of the call's route and ``arch`` (a
    :data:`PAIR_ROWS` range first) the forward runs the route's pair or direct
    tiles instead (:func:`direct_split`, whatever the floors and the taps), as
    the deterministic package plans that range.  The backward ignores
    ``bf16_taps``.
    """

    S, C, num_docs = int(S), int(C), int(num_docs)
    if S < 0 or num_docs < 0:
        raise ValueError(f"S and num_docs must be non-negative, got {(S, num_docs)}")
    split = _plan_split(
        S, C, geometry=geometry, role=role, legacy=legacy, arch=arch, route=route, bf16_taps=bf16_taps
    )
    return _plan_scalars(S, C, num_docs, split)


def _plan_split(
    S: int,
    C: int,
    *,
    geometry: dict[str, int],
    role: str | None = None,
    legacy: bool = False,
    arch: str | None = None,
    route: str | None = None,
    bf16_taps: bool = False,
) -> tuple[int, int, int]:
    """:func:`plan`'s ``(ct_vec, n_sub, n_ch_tiles)`` of one call: for a
    backward call given ``arch`` and ``route`` (not ``legacy``), inside a
    :data:`DIRECT_ROWS` range of its route and ``arch`` (where the route's
    direct program runs) :func:`direct_split`'s at
    :data:`BACKWARD_DIRECT_THREADS` threads, else the
    :data:`TILE_OVERRIDES` row's split, else the :data:`NARROW_TILES` row's,
    else :func:`channel_split`'s; for a forward call (not ``legacy``), inside
    a :data:`PAIR_ROWS` range (the pair form) or else a :data:`DIRECT_ROWS`
    range of its route and ``arch`` :func:`direct_split`'s, else
    :func:`channel_split`'s of its route, ``S``, ``arch`` and taps (the
    :func:`forward_narrow_min_s` and :data:`FORWARD_WIDTH_FLOORS` floors and
    the :func:`forward_single_tile_range` range)."""

    if role == "bwd" and not legacy and arch is not None and route is not None:
        if (
            direct_row(arch, route, C, S)
            and route_role(route) == role
            and int(geometry["threads"]) == BACKWARD_DIRECT_THREADS
        ):
            return direct_split(C, geometry=geometry, route=route, arch=arch)
        override = tile_override(arch, route, C, S)
        if override is not None:
            return _override_split(override, C, geometry)
        if narrow_tile_row(arch, route, C, S):
            narrow = _narrow_split(arch, route, C, geometry)
            if narrow is not None:
                return narrow
    if role == "fwd" and not legacy:
        pair = pair_row(arch, route, C, S)
        if (pair or direct_row(arch, route, C, S)) and route_role(route) == role:
            return direct_split(C, geometry=geometry, route=route, arch=arch, pair=pair)
        return channel_split(
            C, geometry=geometry, role=role, route=route, S=S, arch=arch, bf16_taps=bf16_taps
        )
    return channel_split(C, geometry=geometry, role=role, legacy=legacy)


def search_steps(num_docs: int) -> int:
    """The ``search_steps`` scalar of :func:`plan`: the halvings of the device-side binary search over
    ``cu_seqlens`` for ``num_docs`` documents, ``max(1, ceil(log2 num_docs))``."""

    return max(1, (int(num_docs) - 1).bit_length())


def _plan_scalars(
    S: int, C: int, num_docs: int, split: tuple[int, int, int]
) -> dict[str, int]:
    """The :data:`PLAN_SCALARS` of one call from its ints and its ``(ct_vec, n_sub, n_ch_tiles)`` split."""

    ct_vec, n_sub, n_ch_tiles = split
    return {
        "S": S,
        "C": C,
        "num_docs": num_docs,
        "search_steps": search_steps(num_docs),
        "ct_vec": ct_vec,
        "n_sub": n_sub,
        "n_ch_tiles": n_ch_tiles,
    }


# Whole-rank trim of the backward's single-wave main grid (:func:`main_grid`).
# A channel tile ``ct`` of a ``grid``-CTA launch gets ``ceil((grid - ct) /
# n_ch_tiles)`` CTAs, so when ``n_ch_tiles`` does not divide the grid two pairs
# of neighbouring tiles stream other token ranges at any instant; a row that is
# not 64-byte aligned shares a cache line (and a 32-byte ``dx`` sector) between
# them, which beyond L2 is then fetched twice.  Dropping ``grid % n_ch_tiles``
# CTAs gives every tile the same CTA count.  ``TILE_ALIGNED_GRID[arch]``: the
# co-resident grid the rule was measured at (GB300's 304 CTAs; 0: none), on the
# routes of ``TILE_ALIGNED_ROUTES``, from ``TILE_ALIGNED_MIN_S`` tokens, with a
# last tile of at least ``TILE_ALIGNED_MIN_LAST[ct_vec]`` vectors and at most
# ``TILE_ALIGNED_MAX_TILES[ct_vec]`` tiles (a ``ct_vec`` without an entry:
# never).  ``TILE_ALIGNED_CLASSES[arch]`` holds the other devices' measured
# classes, ``(co_resident, ct_vec, n_ch_tiles) -> (min S, min last-tile
# vectors)`` on the routes of ``TILE_ALIGNED_CLASS_ROUTES[arch]`` (B200's 296
# CTAs; R200 none).  Every other launch keeps its grid.  ``dx`` does not depend
# on the grid; ``dweight`` / ``dbias`` sum other CTA rows.  The rule (the det
# package's, kept for the same bodies) is kept in sync with its copy in the
# backward kernel generator.
TILE_ALIGNED_GRID: dict[str, int] = {"sm_100a": 0, "sm_103a": 304, "sm_107a": 0}
TILE_ALIGNED_ROUTES: tuple[str, ...] = ("bwd",)
TILE_ALIGNED_MIN_S = 98304
TILE_ALIGNED_MIN_LAST: dict[int, int] = {32: 15, 64: 49}
TILE_ALIGNED_MAX_TILES: dict[int, int] = {32: 27, 64: 9}
TILE_ALIGNED_CLASSES: dict[str, dict[tuple[int, int, int], tuple[int, int]]] = {
    "sm_100a": {
        (296, 32, 5): (147456, 27),
        (296, 32, 7): (65536, 31),
        (296, 64, 3): (147456, 59),
    },
    "sm_103a": {},
    "sm_107a": {},
}
TILE_ALIGNED_CLASS_ROUTES: dict[str, tuple[str, ...]] = {
    "sm_100a": ("bwd", "bwd_dt", "bwd_fp32g"),
    "sm_103a": (),
    "sm_107a": (),
}
# The small-S rows where a single-wave backward main launch is capped at a token
# count per CTA, ``{arch: {(route, C_lo, C_hi): ((S_lo, S_hi, tokens), ...)}}``:
# a call of ``route`` on ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple
# of 8) and ``S_lo <= S <= S_hi`` tokens, at ``bwd_grid_mult`` 1, launches an
# ``occupancy`` record with :func:`small_s_grid` of its grid, at most
# ``n_ch_tiles x ceil(S / (n_sub x tokens))`` CTAs and at least ``n_ch_tiles``
# (:func:`launch_grid`), but at a backward :data:`DIRECT_ROWS` range.  The spans
# of a key ascend and are disjoint; the keys of one ``(arch, route)`` share no
# channel count and no ``(C, S)`` with an :data:`SM_WIDE_ROWS` segment, a
# :data:`FALLBACK_ROWS` range or a backward :data:`DIRECT_ROWS` range.  The
# program, its channel tiles and the workspace stay the row's; ``dx`` does not
# depend on the grid; ``dweight`` / ``dbias`` sum other CTA rows.  Rendered from
# the backward kernel generator's table; every other row keeps its grid.
GRID_CAP_ROWS: dict[
    str, dict[tuple[str, int, int], tuple[tuple[int, int, int], ...]]
] = {
    "sm_107a": {
        ("bwd", 584, 592): ((512, 1000, 3),),
        ("bwd", 600, 616): ((512, 999, 3),),
        ("bwd", 624, 632): ((512, 1000, 3),),
        ("bwd", 648, 696): ((512, 1000, 3),),
        ("bwd", 704, 704): ((512, 999, 3),),
        ("bwd", 712, 760): ((512, 1000, 3),),
        ("bwd", 768, 768): ((512, 512, 3),),
        ("bwd_bias", 512, 512): ((2049, 3072, 5),),
    },
}


def grid_cap_spans(
    arch: str, kind: str | None, C: int
) -> tuple[tuple[int, int, int], ...] | None:
    """The ``((S_lo, S_hi, tokens), ...)`` spans of the :data:`GRID_CAP_ROWS` key of ``arch`` on route ``kind``
    covering ``C`` (``C_lo <= C <= C_hi``, ``C`` a multiple of 8), ``None`` without one."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    for (route, c_lo, c_hi), spans in GRID_CAP_ROWS.get(arch, {}).items():
        if route == kind and int(c_lo) <= C <= int(c_hi):
            return tuple(
                (int(s_lo), int(s_hi), int(tokens)) for s_lo, s_hi, tokens in spans
            )
    return None


def grid_cap_tokens(arch: str, kind: str | None, C: int, S: int | None) -> int | None:
    """The tokens per CTA of the :data:`GRID_CAP_ROWS` span holding ``S`` (both bounds included) of the key of
    ``arch`` on route ``kind`` covering ``C``; ``None`` outside every span, without a key or for ``S`` of ``None``."""

    if S is None:
        return None
    for s_lo, s_hi, tokens in grid_cap_spans(arch, kind, C) or ():
        if s_lo <= int(S) <= s_hi:
            return tokens
    return None


def small_s_grid(
    grid: int,
    *,
    arch: str,
    route: str | None,
    C: int,
    S: int,
    n_ch_tiles: int,
    n_sub: int,
    grid_mult: int = 1,
) -> int:
    """``grid``, a backward main launch's CTA count (:func:`main_grid`), capped at a
    row of :data:`GRID_CAP_ROWS`: where :func:`grid_cap_tokens` of ``arch``,
    ``route``, ``C`` and ``S`` gives ``tokens``, at ``grid_mult`` 1 and ``S >= 1``,
    ``max(n_ch_tiles, min(grid, n_ch_tiles x ceil(S / (n_sub x tokens))))`` with
    the call's plan (``n_ch_tiles`` channel tiles of ``n_sub`` sub-tiles each);
    ``grid`` elsewhere.  The callers keep a backward :data:`DIRECT_ROWS` range out
    (:func:`launch_grid`)."""

    tokens = grid_cap_tokens(arch, route, C, S)
    if tokens is None or grid_mult != 1 or int(S) < 1:
        return int(grid)
    capped = int(n_ch_tiles) * _ceil_div(int(S), int(n_sub) * tokens)
    return max(int(n_ch_tiles), min(int(grid), capped))


def main_grid(
    co_resident: int,
    *,
    arch: str,
    grid_mult: int = 1,
    C: int | None = None,
    n_ch_tiles: int = 1,
    S: int | None = None,
    ct_vec: int | None = None,
    route: str | None = None,
) -> int:
    """The backward main launch's CTA count: ``grid_mult x co_resident``.

    ``co_resident`` is the program's co-resident CTA count on the device
    (``ctas_per_sm x SM count``).  A single-wave launch (``grid_mult`` 1) of
    a row the caller names (``C``, ``S``, the plan's ``n_ch_tiles`` and
    ``ct_vec``, the ``route``) is trimmed to whole channel-tile ranks:
    :func:`tile_aligned_grid` on a route of ``TILE_ALIGNED_ROUTES`` when
    ``co_resident`` is ``TILE_ALIGNED_GRID[arch]``, else
    :func:`class_aligned_grid` on a route of
    ``TILE_ALIGNED_CLASS_ROUTES[arch]``.  A caller that names no row keeps the
    untrimmed grid, the most CTAs any row launches.
    """

    if type(grid_mult) is not int or grid_mult < 1:
        raise ValueError(f"grid_mult must be a positive int, got {grid_mult!r}")
    grid = int(co_resident) * grid_mult
    named = None not in (C, S, ct_vec)
    if (
        grid_mult == 1
        and co_resident == TILE_ALIGNED_GRID[arch]
        and route in TILE_ALIGNED_ROUTES
        and named
    ):
        grid = tile_aligned_grid(grid, C, n_ch_tiles, S=S, ct_vec=ct_vec)
    elif grid_mult == 1 and route in TILE_ALIGNED_CLASS_ROUTES[arch] and named:
        grid = class_aligned_grid(grid, C, n_ch_tiles, arch=arch, S=S, ct_vec=ct_vec)
    return grid


def tile_aligned_grid(
    grid: int, C: int, n_ch_tiles: int, *, S: int, ct_vec: int, vec: int = 8
) -> int:
    """``grid`` less ``grid % n_ch_tiles`` CTAs when ``C % 32`` (a row of
    ``x`` / ``dy`` / ``dx`` is not 64-byte aligned), ``1 < n_ch_tiles <=
    TILE_ALIGNED_MAX_TILES[ct_vec]``, ``grid % n_ch_tiles``, ``grid >
    n_ch_tiles``, ``S >= TILE_ALIGNED_MIN_S`` and the last channel tile holds
    at least ``TILE_ALIGNED_MIN_LAST[ct_vec]`` vectors (``C / vec -
    (n_ch_tiles - 1) x ct_vec``); else ``grid``."""

    last = C // vec - (n_ch_tiles - 1) * ct_vec
    if (
        1 < n_ch_tiles <= TILE_ALIGNED_MAX_TILES.get(ct_vec, 0)
        and C % 32
        and grid % n_ch_tiles
        and grid > n_ch_tiles
        and S >= TILE_ALIGNED_MIN_S
        and last >= TILE_ALIGNED_MIN_LAST.get(ct_vec, ct_vec + 1)
    ):
        return grid - grid % n_ch_tiles
    return grid


def class_aligned_grid(
    grid: int,
    C: int,
    n_ch_tiles: int,
    *,
    arch: str,
    S: int,
    ct_vec: int,
    vec: int = 8,
) -> int:
    """``grid`` less ``grid % n_ch_tiles`` CTAs when ``C % 32``, ``grid %
    n_ch_tiles`` and ``(grid, ct_vec, n_ch_tiles)`` is a class of
    ``TILE_ALIGNED_CLASSES[arch]`` whose bounds the row meets (``S`` and the
    last tile's ``C / vec - (n_ch_tiles - 1) x ct_vec`` vectors at least the
    class's); else ``grid``."""

    bound = TILE_ALIGNED_CLASSES[arch].get((grid, ct_vec, n_ch_tiles))
    if bound is None or not C % 32 or not grid % n_ch_tiles or bound[0] > S:
        return grid
    if C // vec - (n_ch_tiles - 1) * ct_vec < bound[1]:
        return grid
    return grid - grid % n_ch_tiles


def row_may_trim(
    record: dict[str, Any], arch: str, co_resident: int, factor: int, C: int, S: int
) -> bool:
    """Whether :func:`main_grid` may trim ``record``'s launch for the row.

    Every trim needs a single wave (``factor`` 1), ``C % 32``, the record on a
    route of the rule and ``S`` at least the rule's smallest token bound for
    ``co_resident``; a call failing any of these launches the untrimmed grid
    without planning the row's channel tiles (no host cost beyond the
    untrimmed grid's)."""

    if factor != 1 or not C % 32:
        return False
    route = _plan_route(record, arch)
    if (
        co_resident == TILE_ALIGNED_GRID[arch]
        and S >= TILE_ALIGNED_MIN_S
        and route in TILE_ALIGNED_ROUTES
    ):
        return True
    return (
        any(
            grid == co_resident and bound[0] <= S
            for (grid, _, _), bound in TILE_ALIGNED_CLASSES[arch].items()
        )
        and route in TILE_ALIGNED_CLASS_ROUTES[arch]
    )


def _plan_route(record: dict[str, Any], arch: str) -> str | None:
    """:func:`record_route` of ``record``, or ``None`` for a record no route of ``ROUTES[arch]`` lists (a KCP
    program: never trimmed)."""

    try:
        return record_route(record, arch)
    except ValueError:
        return None


def record_route(record: dict[str, Any], arch: str) -> str:
    """The route whose host plan ``record`` follows: the route of ``ROUTES[arch]`` that lists it, or, for a record
    of a pair route, the route it pairs (:data:`PAIR_ROUTES`: the same channel tiles and grid) (``ValueError`` if
    no route lists it)."""

    route = _library_route(record["library"], arch)
    return PAIR_ROUTES.get(route, route)


@functools.cache
def _library_route(library: str, arch: str) -> str:
    for kind, names in ROUTES[arch].items():
        if library in names:
            return kind
    raise ValueError(f"module {library} is on no route of {arch}")


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(
    record: dict[str, Any],
    *,
    device_index: int,
    C: int | None = None,
    S: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one module on ``device_index`` under its recorded grid rule.

    ``co_resident`` modules launch exactly ``ctas_per_sm x SM count`` CTAs,
    the co-resident count.  ``occupancy`` modules (the backward programs, every
    one an ordinary launch) launch :func:`main_grid` of that count:
    ``grid_mult[arch] x ctas_per_sm x SM count`` CTAs normally
    (:func:`grid_per_sm`): ``ctas_per_sm`` co-resident CTAs per SM and
    ``grid_mult`` times finer token streams, so a CTA that finds no free slot
    beside another stream's kernel trails by a short stream in a later wave.
    Given the call's ``C`` and ``S``, a single-wave launch of the rows
    :func:`main_grid` names (its route, the call's plan: :func:`plan` with the
    architecture and the record's route, so a row of :data:`TILE_OVERRIDES`
    or :data:`NARROW_TILES` trims by its own tiles) drops CTAs to whole
    channel-tile ranks (the row is planned only where :func:`row_may_trim`
    holds).  At a row of :data:`GRID_CAP_ROWS` (the record's route, ``C`` and
    ``S``; a record of ``grid_mult`` 1; not at a backward :data:`DIRECT_ROWS`
    range) an ``occupancy`` record then launches :func:`small_s_grid` of that
    grid with the call's plan.  A ``co_resident`` record with the ``cap``
    ``token_blocks`` (the direct programs, :data:`GRID_CAPS`) launches at most
    one CTA per token block of the call: ``min(ctas_per_sm x SM count,
    n_ch_tiles x max(1, ceil(S / (block_tokens x n_sub))))`` with the call's
    plan (:func:`plan` with the architecture and the record's route),
    ``block_tokens`` the record's ``grid_rule["block_tokens"]``, else
    :data:`DIRECT_BLOCK_TOKENS`;
    a backward record's capped grid is at least ``n_ch_tiles`` CTAs (every
    channel tile needs a CTA to store its gradients).  So it needs the call's
    ``C`` and ``S`` (a ``ValueError`` without them).  The grid is a pure
    function of ``(S, C, device)`` and the route; ``y`` and ``dx`` do not
    depend on it, and the accumulator workspace does not either
    (:func:`accumulator_layout`).  A backward call's ``bwd_grid_mult``
    multiplies the grid this returns (:func:`plan_row`, :func:`row_launch`).
    """

    rule = record["grid_rule"]
    kind = rule["kind"]
    cap = rule.get("cap")
    if cap is not None and (kind != "co_resident" or cap not in GRID_CAPS):
        raise ValueError(
            f"module {record['kernel']}: unknown grid cap {cap!r} of a {kind!r} grid rule (known: co_resident "
            f"with {GRID_CAPS})"
        )
    if kind in ("co_resident", "occupancy"):
        arch = device_arch(device_index)
        grid_x = grid_per_sm(record, arch) * sm_count(device_index)
        if cap is not None:
            if C is None or S is None:
                raise ValueError(
                    f"module {record['kernel']}: a {cap!r} grid is planned per call; launch_grid needs C and S"
                )
            _, n_sub, n_ch_tiles = _plan_split(
                int(S),
                int(C),
                geometry=rule["geometry"],
                role=record["role"],
                arch=arch,
                route=record_route(record, arch),
            )
            block_tokens = int(rule.get("block_tokens", DIRECT_BLOCK_TOKENS))
            blocks = max(1, _ceil_div(int(S), block_tokens * n_sub))
            grid_x = min(grid_x, n_ch_tiles * blocks)
            if record["role"] == "bwd":
                grid_x = max(grid_x, n_ch_tiles)
        if kind == "occupancy" and C is not None and S is not None:
            co_resident = int(rule["ctas_per_sm"]) * sm_count(device_index)
            factor = grid_mult(record, arch)
            route = _plan_route(record, arch)
            tokens = None
            if factor == 1 and route is not None:
                tokens = grid_cap_tokens(arch, route, int(C), int(S))
                if tokens is not None and direct_row(arch, route, int(C), int(S)):
                    tokens = None
            trim = False
            if C % 32:
                trim = row_may_trim(record, arch, co_resident, factor, int(C), int(S))
            if trim or tokens is not None:
                ct_vec, n_sub, n_ch_tiles = _plan_split(
                    int(S),
                    int(C),
                    geometry=rule["geometry"],
                    role=record["role"],
                    arch=arch,
                    route=route,
                )
                if trim:
                    grid_x = main_grid(
                        co_resident,
                        arch=arch,
                        grid_mult=factor,
                        C=int(C),
                        n_ch_tiles=n_ch_tiles,
                        S=int(S),
                        ct_vec=ct_vec,
                        route=route,
                    )
                if tokens is not None:
                    grid_x = small_s_grid(
                        grid_x,
                        arch=arch,
                        route=route,
                        C=int(C),
                        S=int(S),
                        n_ch_tiles=n_ch_tiles,
                        n_sub=n_sub,
                    )
    else:
        raise ValueError(
            f"unknown Cake causal conv1d training grid rule: {kind!r} (known: {GRID_KINDS})"
        )
    return (max(1, grid_x), 1, 1)


class RowPlan(NamedTuple):
    """The host plan of one call row -- ``(arch, route, C, device, bwd_grid_mult)`` and, where an
    :data:`SM_WIDE_ROWS`, :data:`PAIR_ROWS`, :data:`DIRECT_ROWS` or :data:`FALLBACK_ROWS` key covers the row, one
    side of :func:`row_sides` -- for every ``S`` of that side and ``num_docs`` (:func:`plan_row`); :func:`row_launch`
    completes it for one call."""

    # The programs the row launches (:func:`selected_modules`) and the record of the first.
    names: tuple[str, ...]
    record: dict[str, Any]
    C: int
    # :func:`plan`'s ``(ct_vec, n_sub, n_ch_tiles)`` outside the overrides' token ranges.
    split: tuple[int, int, int]
    # Disjoint ``(S_min, S_max, split)`` token ranges, each with its split: the backward record's
    # :data:`TILE_OVERRIDES` row, then its :data:`NARROW_TILES` row (at most one of each; at a ``(route, C)`` of both
    # tables, the C 768 rows of sm_100a and sm_103a, the narrow key's tokens end below the override's), or the forward
    # record's narrow row: its 32-vector tiles from its :func:`forward_narrow_min_s` floor
    # up to its :data:`FORWARD_WIDTH_FLOORS` floor, and from the later of the two floors the width of a
    # :data:`FORWARD_NARROW_WIDTHS` row with no ceiling, but none inside its :func:`forward_single_tile_range` (the
    # last range's ``S_max`` is ``sys.maxsize``: the forward rule has none, so the row's plan is the per-call plan
    # for every ``S``); empty without one, and on a :data:`PAIR_ROWS` or :data:`DIRECT_ROWS` side (its pair or
    # direct tiles at every ``S``).
    overrides: tuple[tuple[int, int, tuple[int, int, int]], ...]
    # The launch grid, ``None`` where :func:`launch_grid` may trim or cap it by ``S`` (planned per call).
    grid: tuple[int, int, int] | None
    # The factor :func:`row_launch` multiplies a grid planned per call by: the call's ``bwd_grid_mult`` on a
    # backward record with a grid cap (a direct backward program), else 1 (a fixed grid already carries it).
    grid_mult: int = 1


def plan_row(
    arch: str,
    kind: str,
    C: int,
    *,
    device_index: int,
    bwd_grid_mult: int = 1,
    at: int | None = None,
    bf16_taps: bool = False,
) -> RowPlan | None:
    """The :class:`RowPlan` of the calls of route ``kind`` with ``C`` channels on device ``device_index`` (of
    architecture ``arch``) launching ``bwd_grid_mult`` times the untrimmed backward grid (1: the default grid; at a
    backward direct row ``bwd_grid_mult`` times the capped grid of the call), or ``None`` where a token floor of the
    selection rules decides the program (:func:`tokens_select`).

    Where an :data:`SM_WIDE_ROWS`, :data:`PAIR_ROWS`, :data:`DIRECT_ROWS` or :data:`FALLBACK_ROWS` key covers
    ``(kind, C)`` on ``arch`` the row has several sides (:func:`row_sides`) and ``at``, a token count, names the side
    the row plans (``ValueError`` when it is ``None`` or lies in no side): inside the :data:`PAIR_ROWS` range the
    range's pair direct program on the route's pair tiles, else inside the :data:`DIRECT_ROWS` range the range's
    direct program on the route's direct tiles (both :func:`direct_split`), else inside an :data:`SM_WIDE_ROWS`
    segment that segment's SM-wide program, else inside the :data:`FALLBACK_ROWS` range the route's fallback
    program, else
    the program the selection rules pick there (or ``None`` where a token floor decides); with one side ``at`` is not
    read.  ``S`` enters a call's host plan only through those sides, a token floor of the route's selection rules,
    the ``[S_min, S_max]`` range of the record's :data:`TILE_OVERRIDES` row or :data:`NARROW_TILES` row, the token
    floors and single-tile range of a narrow forward row (:func:`forward_narrow_min_s`, :data:`FORWARD_WIDTH_FLOORS`
    and :func:`forward_single_tile_range`, of the call's taps: ``bf16_taps``, read on ``fwd_bias`` only), the trim of
    :func:`launch_grid` (an ``occupancy`` record at ``C % 32`` on the default grid), its :data:`GRID_CAP_ROWS` cap
    (an ``occupancy`` record inside a span, on the default grid: every row of a key's channel counts plans its grid per
    call) and its ``token_blocks`` cap (the direct programs),
    and ``num_docs`` only through
    :func:`search_steps`; so with this row (of the call's side and taps) :func:`row_launch` returns what
    :func:`plan` and :func:`launch_grid` return for the call (the planning ``_launch`` does: ``plan(..., arch=,
    route=, bf16_taps=)`` with the record's route) without selecting, splitting or sizing the grid again.  Registry
    errors raise as in :func:`selected_modules`.
    """

    C = int(C)
    S: int | None = None
    sides = row_sides(arch, kind, C)
    if len(sides) > 1:
        if at is None or not any(s_lo <= int(at) <= s_hi for s_lo, s_hi in sides):
            raise ValueError(
                f"the {kind!r} route on {arch} for C={C} selects by S over the token ranges {sides}; plan_row needs "
                f"at= (a token count of the side the row plans), got {at!r}"
            )
        S = int(at)
    bounds = fallback_row_range(arch, kind, C)
    keyed = S is not None and (
        sm_wide_module(arch, kind, C, S) is not None
        or pair_row(arch, kind, C, S)
        or direct_row(arch, kind, C, S)
        or (bounds is not None and bounds[0] <= S <= bounds[1])
    )
    if not keyed and tokens_select(arch, kind, C):
        return None
    names = selected_modules(arch, kind, C=C, S=S)
    record = MODULES[names[0]]
    rule = record["grid_rule"]
    geometry, role = rule["geometry"], record["role"]
    overrides: tuple[tuple[int, int, tuple[int, int, int]], ...] = ()
    pair = pair_row(arch, kind, C, S)
    if pair or direct_row(arch, kind, C, S):
        # A side of PAIR_ROWS or DIRECT_ROWS (a forward or a backward route): the route's pair or direct tiles at every
        # S of the side.
        split = direct_split(C, geometry=geometry, route=kind, arch=arch, pair=pair)
    elif role == "fwd":
        # The route's plan below every token floor of the row (or without one), then one range per floor and
        # single-tile range bound, up to the next one (the last without a ceiling), wherever its plan differs from
        # the one below every floor.
        route = record_route(record, arch)
        taps = {"bf16_taps": bool(bf16_taps)}
        split = channel_split(C, geometry=geometry, role=role, route=route, S=0, arch=arch, **taps)
        cv = C // int(geometry["vec"])
        gap = forward_single_tile_range(arch, cv, route=route, **taps)
        floors = sorted(
            {
                int(S)
                for S in (
                    forward_narrow_min_s(arch, cv, route=route, **taps),
                    FORWARD_WIDTH_FLOORS.get(arch, {}).get(cv),
                    *(() if gap is None else (gap[0], gap[1] + 1)),
                )
                if S is not None
            }
        )
        ranges = []
        for floor, end in zip(floors, [*(f - 1 for f in floors[1:]), sys.maxsize]):
            tiles = channel_split(C, geometry=geometry, role=role, route=route, S=floor, arch=arch, **taps)
            if tiles != split:
                ranges.append((floor, end, tiles))
        overrides = tuple(ranges)
    else:
        split = channel_split(C, geometry=geometry, role=role)
        route = record_route(record, arch)
        ranges = []
        entry = TILE_OVERRIDES.get(arch, {}).get((route, C))
        if entry is not None:
            ct_vec, s_min, s_max = (int(value) for value in entry)
            ranges.append((s_min, s_max, _override_split(ct_vec, C, geometry)))
        narrow = _narrow_split(arch, route, C, geometry)
        if narrow is not None:
            s_min, s_max = narrow_tile_range(arch, route, C)
            ranges.append((s_min, s_max, narrow))
        overrides = tuple(ranges)
    factor = 1
    capped = grid_cap_spans(arch, kind, C) is not None
    if rule.get("cap") is not None:
        # Capped by S (the direct programs): planned per call; a backward call multiplies the capped grid.
        grid = None
        factor = int(bwd_grid_mult) if role == "bwd" else 1
    elif bwd_grid_mult != 1:
        grid = (launch_grid(record, device_index=device_index)[0] * bwd_grid_mult, 1, 1)
    elif rule["kind"] == "occupancy" and (C % 32 or capped):
        # Trimmed or capped by S: planned per call.
        grid = None
    else:
        grid = launch_grid(record, device_index=device_index)
    return RowPlan(names, record, C, split, overrides, grid, factor)


def row_launch(
    row: RowPlan, S: int, num_docs: int, *, device_index: int
) -> tuple[dict[str, int], tuple[int, int, int]]:
    """``(scalars, grid)`` of one call of ``row`` with ``S`` tokens and ``num_docs`` documents: :func:`plan`'s
    scalars (the split of the override range with ``S_min <= S <= S_max``, else the row's) and the row's grid, or
    where the row has none :func:`launch_grid` of the call times the row's ``grid_mult``."""

    split = row.split
    for s_min, s_max, tiles in row.overrides:
        if s_min <= S <= s_max:
            split = tiles
            break
    grid = row.grid
    if grid is None:
        grid = launch_grid(row.record, device_index=device_index, C=row.C, S=S)
        if row.grid_mult != 1:
            grid = (grid[0] * row.grid_mult, grid[1], grid[2])
    return _plan_scalars(S, row.C, num_docs, split), grid


def _align(nbytes: int) -> int:
    """``nbytes`` rounded up to :data:`WORKSPACE_ALIGN`."""

    return -(-int(nbytes) // WORKSPACE_ALIGN) * WORKSPACE_ALIGN


def _accumulator_rule(record: dict[str, Any]) -> dict[str, Any]:
    """The ``accumulators`` workspace rule of a backward record (``ValueError`` for any other record)."""

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "accumulators":
        raise ValueError(
            f"module {record['kernel']} owns no backward accumulator workspace"
        )
    if rule["acc"] not in ACCUMULATOR_DTYPES or rule["work"] not in COUNTER_DTYPES:
        raise ValueError(
            f"module {record['kernel']}: unknown accumulator dtypes {rule['acc']!r} / {rule['work']!r}"
        )
    # No default: a rule without its copies would undersize ``dw_acc`` for a program that keeps several.
    slots = rule.get("slots")
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        raise ValueError(
            f"module {record['kernel']}: an accumulators rule must name its accumulator copies as a positive int "
            f"'slots', got {slots!r}"
        )
    return rule


def accumulator_words_per_channel(record: dict[str, Any]) -> int:
    """Accumulator elements per channel of a backward record's rule: ``slots x taps`` (``slots`` copies of the
    ``taps`` per-channel sums)."""

    rule = _accumulator_rule(record)
    return int(rule["slots"]) * int(rule["taps"])


def _accumulator_bytes_per_channel(record: dict[str, Any]) -> int:
    return (
        accumulator_words_per_channel(record)
        * ACCUMULATOR_DTYPES[record["workspace_rule"]["acc"]]
    )


def accumulator_layout(
    record: dict[str, Any], C: int, n_ch_tiles: int
) -> dict[str, Any]:
    """Byte layout of one backward launch's ``acc`` / ``work`` inside the ``uint8`` backward workspace.

    ``acc``: ``slots x taps x C`` words of the rule's ``acc`` dtype (FP32 on
    every route) at offset 0 -- ``slots`` copies (the rule's
    required field; the program spreads its CTAs over them) of the
    ``[C, 4]`` tap accumulators, then the ``[C]`` bias accumulators on the bias
    routes, in the program's own order; ``work``: ``n_ch_tiles``
    uint32 ticket counters at the next :data:`WORKSPACE_ALIGN` boundary.  The
    kernels leave both zero after every launch, so a zeroed buffer stays zero.
    The layout depends on ``C``, the plan's ``n_ch_tiles`` and the route,
    never on ``S`` or the launch grid.
    """

    rule = _accumulator_rule(record)
    C, n_ch_tiles = int(C), int(n_ch_tiles)
    if C <= 0 or n_ch_tiles <= 0:
        raise ValueError(f"C and n_ch_tiles must be positive, got {(C, n_ch_tiles)}")
    acc_elems = accumulator_words_per_channel(record) * C
    acc_bytes = acc_elems * ACCUMULATOR_DTYPES[rule["acc"]]
    work_offset = _align(acc_bytes)
    work_bytes = n_ch_tiles * COUNTER_DTYPES[rule["work"]]
    return {
        "acc_dtype": rule["acc"],
        "acc_elems": acc_elems,
        "acc_bytes": acc_bytes,
        "work_dtype": rule["work"],
        "work_offset": work_offset,
        "work_words": n_ch_tiles,
        "total_bytes": _align(work_offset + work_bytes),
    }


def backward_workspace_bytes(arch: str, C: int) -> int:
    """Bytes of the public backward workspace for ``C`` channels on ``arch``.

    The largest :func:`accumulator_layout` over every backward program of every
    backward route of the architecture (each planned with its own geometry and
    role, at the largest of :func:`channel_split`'s tile count, its route's
    :func:`override_tiles`, its route's :func:`narrow_tiles` and its route's
    :func:`direct_tiles`), so one zeroed buffer serves whichever program and
    plan a call selects.  It depends on ``C`` and the architecture only (not
    on ``S``, the SM count or the grid).  The direct backward programs' layouts
    (one plain accumulator copy and at most 16 tickets at C <= 1024) lie far
    below the ``_fp32g`` programs' sixteen copies, so they never set the size.
    """

    nbytes = 0
    for owner in workspace_owners(arch):
        geometry = owner["grid_rule"]["geometry"]
        route = record_route(owner, arch)
        _, _, n_ch_tiles = channel_split(C, geometry=geometry, role=owner["role"])
        n_ch_tiles = max(
            n_ch_tiles,
            override_tiles(arch, route, C, geometry),
            narrow_tiles(arch, route, C, geometry),
            direct_tiles(arch, route, C, geometry),
        )
        nbytes = max(
            nbytes, int(accumulator_layout(owner, C, n_ch_tiles)["total_bytes"])
        )
    return nbytes


def counter_words(record: dict[str, Any], n_ch_tiles: int) -> int:
    """Int32 words of a forward record's ``work`` counters for a launch of ``n_ch_tiles`` channel tiles.

    A ``tile_counters`` rule holds ``words_per_tile`` words per channel tile
    (the dynamic forward's next-chunk counter and done-CTA counter); a record
    without a workspace rule takes no ``work`` (0 words).
    """

    rule = record["workspace_rule"]
    if rule is None:
        return 0
    if rule["kind"] != "tile_counters":
        raise ValueError(
            f"module {record['kernel']}: workspace rule {rule['kind']!r} holds no forward counters"
        )
    if n_ch_tiles <= 0:
        raise ValueError(f"n_ch_tiles must be positive, got {n_ch_tiles}")
    return int(rule["words_per_tile"]) * int(n_ch_tiles)


def forward_kinds(arch: str) -> tuple[str, ...]:
    """The forward routes ``arch`` exports, in :data:`KINDS` order."""

    return tuple(
        kind
        for kind in KINDS
        if route_role(kind) == "fwd" and ROUTES.get(arch, {}).get(kind)
    )


def forward_workspace_words(arch: str, C: int) -> int:
    """Int32 words of the forward workspace for ``C`` channels on ``arch``.

    The largest :func:`counter_words` over every forward program of every
    forward route of the architecture (each planned with its own geometry and
    role on its route), so one buffer serves every forward route.  A narrow
    row takes up to three plans over ``S`` (C = 576: one tile below the
    row's :func:`forward_narrow_min_s` floor, 3 tiles from it, 4 from its width
    floor), so each program is planned with each of the row's tiles of
    :data:`FORWARD_TILES` whatever the floors on the architecture, and the
    buffer serves every ``S``: it depends on ``C`` only (never on ``S``, the
    token floors or the device's SM count).  0 when no forward program of the
    architecture takes ``work``.  This is the deterministic package's size
    over the same programs (the single tile never holds more tiles than the
    others).
    """

    words = 0
    for kind in forward_kinds(arch):
        for name in route_modules(arch, kind):
            record = MODULES[name]
            if record["workspace_rule"] is None:
                continue
            geometry = record["grid_rule"]["geometry"]
            for tiles in FORWARD_TILES:
                _, _, n_ch_tiles = _channel_split(
                    int(C),
                    int(geometry["vec"]),
                    int(geometry["threads"]),
                    record["role"],
                    False,
                    kind,
                    tiles,
                )
                words = max(words, counter_words(record, n_ch_tiles))
    return words


def is_rendered() -> bool:
    """Whether the exporter has rendered this package: its ``build.json`` exists next to this file."""

    return (Path(__file__).parent / "build.json").is_file()


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake causal conv1d training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake causal conv1d training module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else the package's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up: ``spec.build()`` of each)."""

    return cake_build.jit_specs(arch)


_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


@functools.cache
def _arg_sources(name: str) -> tuple[tuple[bool, Any], ...]:
    """``(grid, source)`` per entry of ``MODULES[name]["arg_plan"]``, in
    order: a grid slot (``source`` is the axis, 0 / 1 / 2 for ``grid_x`` /
    ``grid_y`` / ``grid_z``) or the logical name the kernel parameter binds
    (the record's ``logical`` map); the registry is constant, so
    :func:`run_modules` reads it once per module."""

    record = MODULES[name]
    logical = record["logical"]
    return tuple(
        (True, _GRID_AXES[key]) if kind == "grid" else (False, logical[key])
        for kind, key in record["arg_plan"]
    )


def _missing_argument(name: str, values: dict[str, Any]) -> ValueError:
    """The error of a launch of module ``name`` whose ``values`` lack a
    logical name it binds (the first such name, with its kernel parameter)."""

    record = MODULES[name]
    logical, key = next(
        (record["logical"][key], key)
        for kind, key in record["arg_plan"]
        if kind != "grid" and record["logical"][key] not in values
    )
    return ValueError(f"module {name} needs {logical!r} (kernel parameter {key!r})")


@functools.cache
def _ffi_entry(name: str, device_index: int) -> Any:
    """The FFI entry of module ``name`` loaded for CUDA device
    ``device_index``'s architecture (:func:`load`), looked up once."""

    module = load(name, device_arch(device_index))
    return getattr(module, MODULES[name]["ffi_entry"])


@functools.cache
def _raw_stream_getter() -> Any:
    """``torch._C._cuda_getCurrentRawStream``: the handle of a device's
    current torch stream as an integer, without constructing a ``Stream``
    object; or the ``torch.cuda`` form on a torch build without it.  Bound on
    first use: this module imports torch lazily."""

    import torch

    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is not None:
        return getter

    def fallback(device_index: int) -> int:
        return torch.cuda.current_stream(device_index).cuda_stream

    return fallback


_FFI_DEVICES: dict[int, Any] = {}


def _stream_context(device_index: int) -> Any:
    """The tvm-ffi environment-stream context of torch's current stream on
    CUDA device ``device_index`` (a CUDA graph's capture stream during
    capture): the bindings launch on that stream.  The FFI device handle is
    kept per index, so entering the context costs a fraction of
    ``tvm_ffi.use_torch_stream()``, which builds a ``torch.cuda.Stream`` and
    parses a device string on every entry."""

    device = _FFI_DEVICES.get(device_index)
    if device is None:
        device = _FFI_DEVICES[device_index] = tvm_ffi.device(f"cuda:{device_index}")
    return tvm_ffi.use_raw_stream(device, _raw_stream_getter()(device_index))


# The programmatic dependent launch policy of the records whose ``launch.use_pdl`` is ``"runtime"`` (their binding
# takes the ``enable_pdl`` launch scalar, :data:`LOGICAL_SCALARS`), per architecture: ``PDL_EAGER[arch]`` for a call
# outside a CUDA Graph capture, ``PDL_GRAPH[arch]`` for a call inside one (read through :func:`pdl_enabled`), except
# at the backward rows of :data:`PDL_OFF_ROWS`.  With the attribute the launch may start while the stream's
# preceding kernel finishes; every program waits for that kernel (``griddepcontrol.wait``) before it reads or writes
# global memory, so the results are the same either way.
PDL_EAGER: dict[str, bool] = {"sm_100a": True, "sm_103a": True, "sm_107a": True}
PDL_GRAPH: dict[str, bool] = {"sm_100a": True, "sm_103a": True, "sm_107a": True}
# The backward rows that launch without programmatic dependent launch, outside and inside a CUDA Graph capture, whatever
# :data:`PDL_EAGER` and :data:`PDL_GRAPH` say: ``{arch: {(route, C_lo, C_hi): (S_min, S_max)}}`` (the form of
# :data:`FALLBACK_ROWS`), a backward call of ``route`` on ``arch`` with ``C_lo <= C <= C_hi`` channels (a multiple of 8)
# and ``S_min <= S <= S_max`` tokens passes ``enable_pdl`` 0 (:func:`pdl_off_row_range`, :func:`pdl_enabled`).  The keys
# of one ``(arch, route)`` are disjoint.  Measured on sm_100a (GB200) and sm_103a (GB300) in the training step and the
# checkpointed step (on one stream, the backward right after a forward): at large ``S`` on these channel counts the
# ``bwd`` program, launched with the attribute, starts before the forward ends and then runs longer, so the step is
# slower than when only that backward launch goes without the attribute.  ``S_min`` is the first measured ``S`` from
# which the backward without the attribute is not slower than with it in the training step and the checkpointed step,
# eager and in a CUDA Graph, at every ``C`` of the key and every larger measured ``S`` (slower: beyond the timing noise
# band in both of two timed copies of the package); ``S_max`` is ``INT32_MAX`` (every larger call the package takes).
# Token counts below ``S_min`` keep the two tables, also where they lie between measured points.  Every other route,
# row and architecture (sm_107a has no rows), and every forward launch, keep the two tables.
PDL_OFF_ROWS: dict[str, dict[tuple[str, int, int], tuple[int, int]]] = {
    "sm_100a": {
        ("bwd", 1000, 1008): (187755, 2147483647),
        ("bwd", 1016, 1016): (155648, 2147483647),
    },
    "sm_103a": {
        ("bwd", 1000, 1008): (131073, 2147483647),
        ("bwd", 1016, 1016): (125171, 2147483647),
    },
}


def pdl_off_row_range(arch: str, kind: str, C: int) -> tuple[int, int] | None:
    """The ``(S_min, S_max)`` of the :data:`PDL_OFF_ROWS` key of ``arch`` on
    route ``kind`` covering ``C`` (``C_lo <= C <= C_hi``, ``C`` a multiple of
    8), ``None`` without one: inside that token range (both bounds included)
    the call launches without programmatic dependent launch."""

    C = int(C)
    if C % CHANNEL_MULTIPLE:
        return None
    for (route, c_lo, c_hi), (s_min, s_max) in PDL_OFF_ROWS.get(arch, {}).items():
        if route == kind and int(c_lo) <= C <= int(c_hi):
            return int(s_min), int(s_max)
    return None


def pdl_enabled(
    arch: str,
    *,
    capturing: bool,
    kind: str | None = None,
    C: int | None = None,
    S: int | None = None,
) -> int:
    """The ``enable_pdl`` value, 0 or 1, of a launch on ``arch`` (one of :data:`ARCHES`, a ``ValueError``
    otherwise): 0 for a call of route ``kind`` with ``C`` channels and ``S`` tokens inside a :data:`PDL_OFF_ROWS`
    key (:func:`pdl_off_row_range` holds ``S``), in both modes; else :data:`PDL_GRAPH` inside a CUDA Graph capture
    (``capturing``) and :data:`PDL_EAGER` outside one.  The row is ``kind`` (one of :data:`KINDS`), ``C`` and ``S``
    together, or none of them (the two tables' value): a ``kind`` outside :data:`KINDS`, or one or two of the three
    alone, is a ``ValueError``.  The call reads the tables and the rows every time (nothing is planned or cached).
    A record whose ``launch.use_pdl`` is not ``"runtime"`` binds no ``enable_pdl`` and ignores the value."""

    if arch not in ARCHES:
        raise ValueError(f"arch must be one of {ARCHES}, got {arch!r}")
    if kind is not None or C is not None or S is not None:
        if kind not in KINDS or C is None or S is None:
            raise ValueError(
                f"the row of pdl_enabled is kind (one of {KINDS}), C and S together, got "
                f"kind={kind!r}, C={C!r}, S={S!r}"
            )
        bounds = pdl_off_row_range(arch, kind, C)
        if bounds is not None and bounds[0] <= int(S) <= bounds[1]:
            return 0
    return int(bool((PDL_GRAPH if capturing else PDL_EAGER)[arch]))


def run_modules(
    launches: Sequence[tuple[str, tuple[int, int, int] | None]],
    values: dict[str, Any],
    *,
    device_index: int,
) -> None:
    """Launch exported modules in order, back to back, with one set of logical
    argument ``values``.

    ``launches`` holds ``(name, grid)`` pairs, ``grid`` as in
    :func:`run_module`.  All of them are issued inside one stream context of
    torch's current stream on ``device_index``, each bound (grid and
    arguments) right before it is issued, so the first launch waits for no
    later one's binding.  ``ValueError`` names the first logical
    name a module binds that ``values`` lacks.
    """

    with _stream_context(device_index):
        for name, grid in launches:
            if grid is None:
                grid = launch_grid(
                    MODULES[name],
                    device_index=device_index,
                    C=values.get("C"),
                    S=values.get("S"),
                )
            sources = _arg_sources(name)
            try:
                args = [
                    grid[source] if is_grid else values[source]
                    for is_grid, source in sources
                ]
            except KeyError:
                raise _missing_argument(name, values) from None
            _ffi_entry(name, device_index)(*args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    device_index: int,
    grid: tuple[int, int, int] | None = None,
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is
    ``grid`` when the caller planned it (the module's :func:`launch_grid` for
    the call, as the public entry points pass it), else the module's
    :func:`launch_grid` on ``device_index`` for the call's ``C`` and ``S``,
    and the kernel runs on the current torch stream; a
    module whose launch record takes the programmatic dependent launch scalar
    (``use_pdl`` ``"runtime"``), launched with ``enable_pdl`` 1, overlaps its
    launch with the preceding kernel of that stream and waits for it
    in-kernel.
    """

    run_modules(((name, grid),), values, device_index=device_index)


__all__ = [
    "ACCUMULATOR_DTYPES",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHANNEL_MULTIPLE",
    "COUNTER_DTYPES",
    "DIRECT_BLOCK_TOKENS",
    "DWEIGHT_SUMS",
    "FLAG_B_BF16",
    "FLAG_DB_BF16",
    "FLAG_DW_BF16",
    "FLAG_W_BF16",
    "GRID_CAPS",
    "GRID_KINDS",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PAIR_PLANS",
    "PAIR_ROUTES",
    "PAIR_ROWS",
    "PDL_EAGER",
    "PDL_GRAPH",
    "PDL_OFF_ROWS",
    "PLAN_SCALARS",
    "REQUIRED_KINDS",
    "ROLES",
    "ROUTES",
    "ROWS_KIND",
    "SELECT_KINDS",
    "TILE_ALIGNED_CLASSES",
    "TILE_ALIGNED_CLASS_ROUTES",
    "TILE_ALIGNED_GRID",
    "TILE_ALIGNED_MAX_TILES",
    "TILE_ALIGNED_MIN_LAST",
    "TILE_ALIGNED_MIN_S",
    "TILE_ALIGNED_ROUTES",
    "WIDTH",
    "WORKSPACE_ALIGN",
    "WORKSPACE_KINDS",
    "accumulator_layout",
    "accumulator_words_per_channel",
    "arch_for_capability",
    "backward_kinds",
    "backward_route",
    "backward_workspace_bytes",
    "channel_split",
    "class_aligned_grid",
    "counter_words",
    "device_arch",
    "direct_module",
    "direct_row",
    "direct_row_range",
    "direct_split",
    "direct_tiles",
    "fallback_row_range",
    "forward_kinds",
    "forward_route",
    "forward_workspace_words",
    "grid_mult",
    "grid_per_sm",
    "is_rendered",
    "launch_grid",
    "library_name",
    "load",
    "main_grid",
    "pair_module",
    "pair_row",
    "pair_row_range",
    "pdl_enabled",
    "pdl_off_row_range",
    "plan",
    "record_route",
    "route_applies",
    "route_modules",
    "route_role",
    "row_may_trim",
    "row_sides",
    "run_module",
    "run_modules",
    "select_rule",
    "selected_modules",
    "sm_count",
    "sm_wide_module",
    "sm_wide_segments",
    "specs_for_arch",
    "tile_aligned_grid",
    "workspace_geometry",
    "workspace_owners",
    "workspace_record",
]
