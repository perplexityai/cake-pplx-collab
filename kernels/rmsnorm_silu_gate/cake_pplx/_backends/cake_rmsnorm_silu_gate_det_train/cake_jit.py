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

Host plan and loader glue of the Cake SiLU-gated RMSNorm training kernels
(forward and backward; SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its role, its
launch record, its grid rule, the backward's workspace and the
kernel-parameter -> logical-argument map.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through the package's
JIT shim from the same sources and flags, verifying every source digest before the
first build.

A role is one kernel of a call: ``fwd`` (the forward) and ``bwd_bf16`` /
``bwd_f32`` (the backward: ``dx``, ``dz`` and the weight gradient in one
kernel, the weight gradient rounded once to FP32, then to BF16 for a BF16
weight gradient).
``ROUTES[arch][role]`` names the module a role runs on an architecture.  An
architecture may also route ``fwd_packed`` (``PACKED_ROLES``): a forward of the
rows of contiguous ``[T, H, 128]`` inputs, on sm_107a one that addresses them
as one packed array, without the token and head strides, and on sm_100a and
sm_103a the forward of version 0.2.14, whose L2 prefetches warm only rows
packed in both ``x`` and ``z`` (below).  :func:`forward_module` runs it for
such inputs and ``fwd`` otherwise; both compute a row with the same FP32
operations in the same order, so ``y`` is the same bits on either route.  An
architecture may also route ``fwd_tiny`` (``TINY_ROLES``): a forward of
packed rows only (row ``n`` of ``x`` and ``z`` at element ``128 n``; it takes
no strides) that runs one block of its rule's ``rows_per_cta`` rows a CTA,
with no loop.  :func:`forward_module` runs it in the place of ``fwd_packed``
or ``fwd`` up to ``FWD_TINY_ROWS_PER_SM`` rows per SM of contiguous
``[T, H, 128]`` inputs or of one token of head stride 128, except the rows of
``FWD_TINY_GAP_ROWS``; it computes a row
with the row arithmetic of the architecture's ``fwd`` and ``fwd_packed``
modules, op for op, so ``y`` is their bits.  Each backward
role also routes a long role (``LONG_ROLES``: ``bwd_bf16_long``,
``bwd_f32_long``): the same backward in a loop that loads a block's rows while
the previous block's row math runs, faster where a CTA runs many blocks.
:func:`backward_module` runs it in the role's place when ``dy``, ``x`` and
``z`` hold packed rows and the rows are at least its rule's
``min_rows_per_sm`` times the SM count, on other rows (a call that computes
the weight gradient) from its rule's ``min_rows_per_sm_strided`` times the SM
count (the BF16 long module only: the FP32 one's rule says ``packed_rows``),
and the role's module otherwise; both compute the rows and the weight
gradient with the same FP32 operations in the same order, so ``dx``, ``dz``
and ``dw`` are the same bits on either route.
Each backward role also routes two small-row roles (``CLUSTER_ROLES``): the
same backward built for one CTA of at most 16 rows (``role + SOLO``:
``bwd_bf16_solo``, ``bwd_f32_solo``) and for one cluster of 16 CTAs of at
most 256 rows (``role + CLUSTER``: ``bwd_bf16_cluster``,
``bwd_f32_cluster``), CTA ``c`` running row block ``c`` as the role's module
runs a block it gives a CTA of its own, with no ticket, no claim and no
workspace word.  A call that computes the weight gradient runs them in the
place of the role's module up to their rules' ``max_rows``
(:func:`dw_module`; the cluster program only where
:func:`device_runs_clusters` holds): they form each block's FP32 column
words with the same FP32 operations in the same order and add the words
exactly (the cluster program's CTAs push their words to the CTA that owns
the column, through the cluster's shared memory), so ``dx``, ``dz`` and
``dw`` are the role's module's bits on a device of at least 4 SMs, where
that module gives every block of up to 256 rows a CTA of its own.
A backward that computes no weight gradient (``compute_grad_weight=False``)
runs a program built without it (``NODW_ROLES``, :func:`nodw_module`): the
BF16 backward's row arithmetic with all weight-gradient work removed at build
time, so ``dx`` and ``dz`` are the bits of the BF16 backward on the same rows
(at every grid).  The strided program (``bwd_bf16_nodw_strided``: the BF16
backward's 256-thread CTAs of 16-row blocks, CTA ``b`` running the blocks
``b``, ``b + grid``, ... with no claim, no ticket and no workspace) runs up
to ``NODW_STRIDED_ROWS_PER_SM`` rows per SM, the claimed program
(``bwd_bf16_nodw``: the BF16 backward's claims, ticket and workspace;
sm_100a and sm_103a) past it, and the long one (``bwd_bf16_nodw_long``: its
look-ahead loop) where :func:`backward_module` runs ``bwd_bf16_long`` for
such a call (packed rows only); sm_107a routes no claimed program, so there the strided program runs every
row count short of the long one.  Two 128-thread builds run in the strided
program's place (8-row blocks, a row to each 16 lanes as in the BF16
backward): ``bwd_bf16_nodw_strided_packed`` on packed rows and, up to
``NODW_ANY_LAYOUT_ROWS_PER_SM`` rows per SM, on rows of any layout (packed
rows read at their linear offsets, other layouts at their token and head
strides), and ``bwd_bf16_nodw_strided_packed_small`` on packed rows up to
``PACKED_SMALL_ROWS_PER_SM`` rows per SM (a build of packed rows only: no
stride test, every row read at its linear offset, a row's sum of squares and
its dot product reduced across its lanes together).  An architecture that
routes none of them runs the BF16 backward's module with ``weight_grad`` 0,
which gives the same bits.

Grids (:func:`launch_grid`): the forward launches one CTA per block of
``rows_per_block`` rows below a cut of rows per SM (below), and from there
the fewest whole waves of ``ctas_per_sm * SMs`` CTAs (the CTAs per SM its
launch bounds are compiled for) whose CTAs walk at most
``max_blocks_per_cta`` blocks each, and never more CTAs than blocks, CTA ``b``
walking the blocks ``b``, ``b + grid``, ``b + 2 * grid``, ... and loading each
next block while it computes the current one.  The forward's cut is keyed
by architecture (the record's ``arches``; :func:`launch_grid` takes
``arch``, and :func:`run_module` computes the same grid for the device's
with the fewest host operations).  The forward of rows of any layout has
two cuts: ``persistent_from_rows_per_sm`` for rows the kernel reads as
packed (:func:`forward_packed`, its own test) and
``persistent_from_rows_per_sm_strided`` for other rows (strided, broadcast
over tokens, or one token, whose token stride is passed as 0); a ``fwd_packed``
forward has the first only.  The forward warms in L2 the CTA's next
block and the first block of the CTA a wave later, each of ``x`` and ``z``
by its own layout: packed rows as one byte range, rows of head stride 128
and another token stride (a gate sliced from a projection, a broadcast over
tokens, one token) as one range a token, and rows of any other layout
(head-major) as one 256-byte range a row on sm_103a and sm_107a, not at all
on sm_100a (where those prefetches measured slower than none); a prefetch
writes no register and no memory, so ``y`` is the same bits.  On sm_100a and
sm_103a the ``fwd_packed`` forward warms only rows packed in both ``x`` and
``z``, as the forward of 0.2.14 did: next to the per-layout prefetch code,
packed rows measured slower at small and mid row counts.  On
packed rows the cut is 5600 on sm_103a, the fewest rows per SM measured
from which the whole waves are faster than one CTA per block every way:
with idle gaps between the calls (full clocks), back to back, and in each
call's GPU time with a cold L2; and 8192 on sm_100a and sm_107a,
unchanged.  On GB300 (sm_103a, 152 SMs) the whole waves are faster every
way from 5600 rows per SM (the step 1 to 2 %, the forward 1.4 to 5.5 %),
gain nothing from 2000 to 5000 rows per SM (within about 1 % either way;
at 5000 the forward's GPU time with a cold L2 is 1 % slower) and are
slower with idle gaps from 1000 rows per SM down (the forward up to 7 %
at 400).
On B200 (sm_100a, 148 SMs) the whole waves of packed rows are up to about
1 % slower in each call's GPU time with a cold L2 from 6500 to 8000 rows
per SM, so sm_100a keeps the cut of 8192.  On other rows the whole waves
gain from far fewer rows per SM, so their cut is 500 rows per SM on sm_100a
and 400 on sm_103a and sm_107a: from there strided rows and rows broadcast
over tokens are faster on whole waves, or within noise, in three
measurements of the forward without the L2 prefetches of these rows: back
to back, with idle gaps (cold L2), and each call's GPU time with a cold L2
(CUPTI).  On B200 strided rows
are faster from 300 rows per SM, broadcast rows faster or equal from 500 (at
300 they are 1.3 % slower in GPU time and 5.3 % slower with idle gaps); on
GB300 both are faster from 300 in GPU time and within 0.9 % with idle gaps
from 400; on VR200 (sm_107a) both are faster from 300 in GPU time, within
0.8 % with idle gaps at 300 and faster at 400.  Further up the gain grows
(GB300: 2 to 9 % with idle gaps from 1000 to 6600 rows per SM, 4 to 11 %
back to back; VR200: 3 to 13 % back to back).  The grid changes no bits: a
CTA runs each of its blocks' rows with the same FP32 operations in the same
order whatever the grid.  The tiny forward launches
``ceil(rows / rows_per_cta)`` CTAs whatever the SM count, CTA ``b`` running
block ``b`` alone.  The backward launches
``min(blocks, ctas_per_sm * SMs)`` CTAs of ``rows_per_block``-row blocks; past
one block per CTA its CTAs claim the blocks in chunks whose rows are fixed by
the row count and the grid, add each thread's FP32 runs of rows into plain
FP32 words and each CTA's FP32 column words exactly into fixed-point digits,
so the weight gradient's bits depend on the rows and
the SM count only, not on which CTA runs a chunk (up to 256 rows, on a
device of at least 4 SMs, on the rows alone).  A strided program
launches ``min(blocks, ctas_per_sm * SMs)`` CTAs of its own rule (the
256-thread one the BF16 backward's 16-row blocks and 4 CTAs per SM, the
128-thread ones 8-row blocks and 8 CTAs per SM:
``min(ceil(rows / 8), 8 * SMs)``), its CTAs walking their blocks with no
claim.  A small-row program launches its rule's ``cluster`` CTAs (one
cluster; one CTA for a solo program) whatever the rows, up to its rule's
``max_rows``.  The backward's workspace
(``workspace`` of its record: ``slots`` accumulator copies of ``limbs``
64-bit digits per column, then one 32-bit flag word per column and the two
32-bit words of the arrival ticket and the claim counter) starts zero and the
kernel leaves it zero; the records of a strided program and of a small-row
program have none, and they read and write no workspace word.  No module is
cooperative, and none waits for another CTA or grid inside a kernel but a
cluster program, whose CTAs wait for the words their peers push to them:
the cluster's 16 CTAs are co-scheduled (the GPU starts them together, once
one GPC has room for all of them), so none waits for a CTA that is not
running.  A kernel resident on another stream delays a
call roughly by its share of the SMs; a forward CTA whose slot such a kernel
holds walks all of its blocks once it starts, and a backward CTA that starts
late claims only the chunks left (a strided program's CTA, which claims
none, runs all of its own blocks once it starts; a cluster program starts
once one GPC has room for its 16 CTAs).

The program name ``cake_rmsnorm_silu_gate_det_train_<20 hex>`` is the first twenty
hex digits of the SHA-256 of the compact JSON ``{"build": <build keywords,
resolved and sorted>, "module": <dotted kernel module>}`` of the kernel the
program was rendered from (a keyword a kernel gained after its programs
shipped is not spelled at its default, so a program keeps its name when the
kernel gains a keyword); each record's comment names its build.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    # The forward: 128 threads, 8 lanes per row (16-row blocks), at most 6
    # CTAs per SM, each loading its next block while it computes the current
    # one, with the L2 prefetch of its next block and of the first block of
    # the CTA a wave later (x and z each by its own layout: packed rows one
    # range, head stride 128 one range a token, other layouts one row a lane
    # on sm_103a and none on sm_100a), the gate and rstd by the approximate
    # instructions (build {"min_blocks": 6, "prefetch_ahead": 1,
    # "prefetch_wave": 1, "threads": 128}); whole waves of at most 2 blocks
    # per CTA.  Packed rows below 2^24 rows run ``fwd_packed`` instead.
    "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": {
                "sm_100a": 8192,
                "sm_103a": 5600,
            },
            "persistent_from_rows_per_sm_strided": {
                "sm_100a": 500,
                "sm_103a": 400,
            },
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd",
    },
    # The sm_107a forward: the schedule above with the sum of squares as four
    # FMA chains, rsqrt.approx.ftz and w loaded behind the first block's loads
    # (build {"min_blocks": 6, "prefetch_ahead": 1, "prefetch_wave": 1,
    # "rsqrt_ftz": 1, "sum_pairs": 1, "threads": 128, "w_late": 1}); whole
    # waves of at most 2 blocks per CTA.
    "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": {
                "sm_107a": 8192,
            },
            "persistent_from_rows_per_sm_strided": {
                "sm_107a": 400,
            },
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd",
    },
    # The sm_107a forward of packed rows (contiguous ``[T, H, 128]`` x and z,
    # fewer than 2^24 rows): the forward above with each row's offset taken
    # from its index alone and one 32-byte store per lane (build
    # {"min_blocks": 6, "packed_rows": 1, "prefetch_ahead": 1,
    # "prefetch_wave": 1, "rsqrt_ftz": 1, "store_width": 256, "sum_pairs": 1,
    # "threads": 128, "w_late": 1}); whole waves of at most 2 blocks per CTA.
    "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": {
                "sm_107a": 8192,
            },
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_packed",
    },
    # The sm_100a / sm_103a forward of packed rows (contiguous ``[T, H, 128]``
    # x and z, fewer than 2^24 rows): the first forward with the L2 prefetches
    # of version 0.2.14, which warm only rows packed in both x and z (build
    # {"layout_prefetch": 0, "min_blocks": 6, "prefetch_ahead": 1,
    # "prefetch_wave": 1, "threads": 128}); whole waves of at most 2 blocks per
    # CTA.
    "cake_rmsnorm_silu_gate_det_train_3c42d6e202fd8d5c994b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "persistent_from_rows_per_sm": {
                "sm_100a": 8192,
                "sm_103a": 5600,
            },
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_3c42d6e202fd8d5c994b",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_3c42d6e202fd8d5c994b",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_3c42d6e202fd8d5c994b",
        "logical": {
            "eps": "eps",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "y": "y",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "fwd_packed",
    },
    # The tiny forward: 128 threads, 32 lanes per row (4-row blocks, a row
    # per warp), one block per CTA and no loop, packed rows only (row n of x
    # and z at element 128 n; no strides), with the row arithmetic of the
    # sm_100a and sm_103a forward op for op (each 16-element slice's sum of
    # squares on one f32x2 FMA chain, the slices folded in its order,
    # rsqrt.approx), so y is its bits (build {"chains": 1, "handoff": 0,
    # "lanes": 32, "min_blocks": 8, "rsqrt_ftz": 0, "threads": 128});
    # forward_module runs it up to FWD_TINY_ROWS_PER_SM rows per SM.
    "cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 4,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99",
        "logical": {
            "eps": "eps",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "y": "y",
            "z": "z",
        },
        "role": "fwd_tiny",
    },
    # The sm_107a tiny forward: 128 threads, 16 lanes per row (8-row blocks,
    # two rows per warp), one block per CTA and no loop, packed rows only,
    # with the row arithmetic of the sm_107a forwards op for op (four FMA
    # chains, rsqrt.approx.ftz), so y is their bits (build {"chains": 2,
    # "handoff": 0, "lanes": 16, "min_blocks": 8, "rsqrt_ftz": 1, "threads":
    # 128}); forward_module runs it up to FWD_TINY_ROWS_PER_SM rows per SM,
    # except 17 to 32 rows (FWD_TINY_GAP_ROWS).
    "cake_rmsnorm_silu_gate_det_train_1e611ed07e9f796d9857": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["parameter", "rows"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "row_blocks",
            "rows_per_cta": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_1e611ed07e9f796d9857",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_1e611ed07e9f796d9857",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_1e611ed07e9f796d9857",
        "logical": {
            "eps": "eps",
            "rows": "rows",
            "w": "w",
            "x": "x",
            "y": "y",
            "z": "z",
        },
        "role": "fwd_tiny",
    },
    # The backward, BF16 weight gradient: 256 threads, 16 lanes per row
    # (16-row blocks), at most 4 CTAs per SM, claimed chunks, the weight
    # gradient summed exactly in fixed point and rounded once (build
    # {"math": "ax", "min_blocks": 4, "out": "bf16",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The backward's long program, BF16 weight gradient: the BF16 backward in
    # the look-ahead loop (a block's rows load while the previous block's row
    # math runs; the same FP32 operations in the same order, so dx, dz and dw
    # are that backward's bits at every row count and grid), run in its place
    # from min_rows_per_sm rows per SM on packed rows and, by a call that
    # computes the weight gradient, from min_rows_per_sm_strided on other rows
    # (build {"lookahead": True, "math": "ax", "min_blocks": 4, "out": "bf16",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "min_rows_per_sm": {
                "sm_100a": 2560,
                "sm_103a": 2560,
                "sm_107a": 1536,
            },
            "min_rows_per_sm_strided": {
                "sm_100a": 6145,
                "sm_103a": 6145,
                "sm_107a": 4097,
            },
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_long",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The backward, FP32 weight gradient: 256 threads, 16 lanes per row
    # (16-row blocks), at most 4 CTAs per SM, claimed chunks, the weight
    # gradient summed exactly in fixed point and rounded once (build
    # {"math": "ax", "min_blocks": 4, "out": "f32",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The backward's long program, FP32 weight gradient: the FP32 backward in
    # the look-ahead loop (a block's rows load while the previous block's row
    # math runs; the same FP32 operations in the same order, so dx, dz and dw
    # are that backward's bits at every row count and grid), run in its place
    # on packed rows from min_rows_per_sm rows per SM (build
    # {"lookahead": True, "math": "ax", "min_blocks": 4, "out": "f32",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "min_rows_per_sm": {
                "sm_100a": 2560,
                "sm_103a": 2560,
                "sm_107a": 1536,
            },
            "packed_rows": True,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32_long",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The backward without the weight gradient, strided: the BF16 backward
    # built without the weight gradient (all of its work removed at build
    # time), CTA b running the 16-row blocks b, b + grid, ... with no claim,
    # no ticket and no workspace word, so dx and dz are the BF16 backward's
    # bits at any grid; the BF16 backward's kernel
    # parameters (the weight-gradient and workspace ones unread) and CTA
    # count; a call with compute_grad_weight=False runs it on rows that are
    # not packed past NODW_ANY_LAYOUT_ROWS_PER_SM rows per SM (packed rows,
    # and fewer rows of any layout, run the 128-thread builds) up to
    # NODW_STRIDED_ROWS_PER_SM rows per SM, on sm_107a at every row count
    # (build {"dw": False, "math": "ax", "min_blocks": 4, "out": "bf16",
    # "prefetch_share": None, "strided": True, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "strided_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_nodw_strided",
    },
    # The backward without the weight gradient, strided, for packed rows: the
    # BF16 backward's row arithmetic built without the weight gradient (all
    # of its work removed at build time) in 128-thread CTAs at most 8 per SM,
    # a row to each 16 lanes as in the BF16 backward, CTA b running the 8-row
    # blocks b, b + grid, ... with no claim, no ticket and no workspace word
    # (packed rows read at their linear offsets, other layouts at their token
    # and head strides through the fast divmod of heads, as the 256-thread
    # program reads them; a CTA of one block takes its block count and
    # prefetch test without an integer division), so dx and dz are the BF16
    # backward's bits at any grid; the BF16 backward's kernel parameters (the
    # weight-gradient and workspace ones unread) and its own grid,
    # min(ceil(rows / 8), 8 * SMs) CTAs; a call with compute_grad_weight=False
    # runs it in the strided program's place on packed rows (dy, x and z of
    # head stride 128 and token stride heads * 128, or, where
    # PACKED_SMALL_ROWS_PER_SM and NODW_STRIDED_ROWS_PER_SM have a cut, one
    # token of head stride 128) past PACKED_SMALL_ROWS_PER_SM rows per SM up to
    # NODW_STRIDED_ROWS_PER_SM rows per SM, on sm_107a at every row count but
    # the long program's, and on rows of any layout up to
    # NODW_ANY_LAYOUT_ROWS_PER_SM rows per SM (build {"dw": False, "math":
    # "ax", "min_blocks": 8, "out": "bf16", "prefetch_share": None,
    # "strided": True, "threads": 128}).
    "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 8,
            "kind": "strided_row_blocks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_nodw_strided_packed",
    },
    # The backward without the weight gradient, strided, for packed rows only,
    # at small row counts: the 128-thread strided build of packed rows (8-row
    # blocks, 128-thread CTAs at most 8 per SM, a row to each 16 lanes as in
    # the BF16 backward, CTA b running the blocks b, b + grid, ... with no
    # claim, no ticket and no workspace word) that takes no stride test and no
    # division by the head count, reads every row at its linear offset (row *
    # 128) and computes a row's sum of squares and its dot product before it
    # reduces both across the row's 16 lanes together, with the BF16 backward's
    # FP32 operations in the same order, so dx and dz are the BF16 backward's
    # bits at any grid; the BF16 backward's kernel parameters (the strides and
    # the weight-gradient and workspace ones unread) and the 128-thread build's
    # grid, min(ceil(rows / 8), 8 * SMs) CTAs; a call with
    # compute_grad_weight=False runs it in the strided program's place on
    # packed rows only (dy, x and z of head stride 128 and token stride heads *
    # 128, or one token of head stride 128) up to PACKED_SMALL_ROWS_PER_SM rows
    # per SM (build {"dw": False, "math": "ax", "min_blocks": 8, "out": "bf16",
    # "packed_rows": 1, "prefetch_share": None, "schedule": "joint", "strided":
    # True, "threads": 128}).
    "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 8,
            "kind": "strided_row_blocks",
            "rows_per_block": 8,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_nodw_strided_packed_small",
    },
    # The backward without the weight gradient, claimed: the BF16 backward
    # built without the weight gradient, with its claims, its arrival ticket
    # and its claim counter (its grid and workspace; the kernel leaves the
    # claim counter and the ticket words zero), so dx and dz are the BF16
    # backward's bits; a call with compute_grad_weight=False runs it past
    # NODW_STRIDED_ROWS_PER_SM rows per SM on sm_100a and sm_103a (build
    # {"dw": False, "math": "ax", "min_blocks": 4, "out": "bf16",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_nodw",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The long program without the weight gradient: the BF16 long program
    # built without the weight gradient (its look-ahead loop, claims, ticket,
    # grid and workspace), so dx and dz are the BF16 backward's bits; a call
    # with compute_grad_weight=False runs it where backward_module runs the
    # BF16 long program for such a call, on packed rows only: its rule has
    # the BF16 long program's min_rows_per_sm and packed_rows in place of its
    # min_rows_per_sm_strided (build {"dw": False, "lookahead": True, "math":
    # "ax", "min_blocks": 4, "out": "bf16", "prefetch_share": None,
    # "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "ctas_per_sm": 4,
            "kind": "claimed_row_blocks",
            "min_rows_per_sm": {
                "sm_100a": 2560,
                "sm_103a": 2560,
                "sm_107a": 1536,
            },
            "packed_rows": True,
            "rows_per_block": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 256,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_nodw_long",
        "workspace": {
            "limbs": 9,
            "slots": 2,
        },
    },
    # The backward of up to 16 rows, BF16 weight gradient: the BF16 backward
    # in one CTA of 256 threads that runs the call's one 16-row block as the
    # BF16 backward runs a block it gives a CTA of its own (the same row
    # arithmetic and FP32 column words), adds the words exactly into its own
    # fixed-point digits and rounds each column's sum once by the same
    # arithmetic, with no ticket, no claim and no workspace word (its
    # workspace parameters unread), so dx, dz and dw are the BF16 backward's
    # bits; a call that computes the weight gradient runs it in the BF16
    # backward's place up to its rule's max_rows (dw_module) (build
    # {"cluster": 1, "math": "ax", "min_blocks": 4, "out": "bf16",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "cluster",
            "max_rows": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_solo",
    },
    # The backward of up to 16 rows, FP32 weight gradient: the FP32 backward
    # in one CTA of 256 threads, as the BF16 one above, so dx, dz and dw are
    # the FP32 backward's bits; a call that computes the weight gradient runs
    # it in the FP32 backward's place up to its rule's max_rows (dw_module)
    # (build {"cluster": 1, "math": "ax", "min_blocks": 4, "out": "f32",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "cluster",
            "max_rows": 16,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 18176,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32_solo",
    },
    # The backward of up to 256 rows, BF16 weight gradient: the BF16 backward
    # in one non-portable cluster of 16 CTAs of 256 threads, CTA c running the
    # 16-row block c as the BF16 backward runs a block it gives a CTA of its
    # own (the same row arithmetic and FP32 column words; a CTA past the last
    # block has words of +0), with no ticket, no claim and no workspace word
    # (its workspace parameters unread); every CTA pushes its words of the
    # columns 8c to 8c + 7 to CTA c through the cluster's shared memory, and
    # CTA c adds them exactly into fixed-point digits and rounds each
    # column's sum once by the same arithmetic, so dx, dz and dw are the BF16
    # backward's bits on a device of at least 4 SMs; a CTA waits for its
    # peers' pushes (the cluster is co-scheduled); a call that computes the
    # weight gradient runs it in the BF16 backward's place past the solo
    # program's rows up to its rule's max_rows on a device that runs clusters
    # (dw_module, device_runs_clusters) (build {"cluster": 16, "math": "ax",
    # "min_blocks": 4, "out": "bf16", "prefetch_share": None, "threads":
    # 256}).
    "cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 256,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20224,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_bf16_cluster",
    },
    # The backward of up to 256 rows, FP32 weight gradient: the FP32 backward
    # in one non-portable cluster of 16 CTAs, as the BF16 one above, so dx, dz
    # and dw are the FP32 backward's bits on a device of at least 4 SMs; a
    # call that computes the weight gradient runs it in the FP32 backward's
    # place past the solo program's rows up to its rule's max_rows on a
    # device that runs clusters (dw_module, device_runs_clusters) (build
    # {"cluster": 16, "math": "ax", "min_blocks": 4, "out": "f32",
    # "prefetch_share": None, "threads": 256}).
    "cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "dy"],
            ["buffer", "x"],
            ["buffer", "z"],
            ["buffer", "w"],
            ["buffer", "dx"],
            ["buffer", "dz"],
            ["buffer", "acc"],
            ["buffer", "flags"],
            ["buffer", "done"],
            ["buffer", "dw"],
            ["parameter", "rows"],
            ["parameter", "heads"],
            ["parameter", "dy_tok"],
            ["parameter", "dy_head"],
            ["parameter", "x_tok"],
            ["parameter", "x_head"],
            ["parameter", "z_tok"],
            ["parameter", "z_head"],
            ["parameter", "eps"],
            ["parameter", "accumulate"],
            ["parameter", "weight_grad"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 16,
            "kind": "cluster",
            "max_rows": 256,
        },
        "kernel": "cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137",
        "kernel_symbol": "kernel_cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (16, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20224,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137",
        "logical": {
            "acc": "acc",
            "accumulate": "accumulate",
            "done": "done",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_head": "dy_head",
            "dy_tok": "dy_tok",
            "dz": "dz",
            "eps": "eps",
            "flags": "flags",
            "heads": "heads",
            "rows": "rows",
            "w": "w",
            "weight_grad": "weight_grad",
            "x": "x",
            "x_head": "x_head",
            "x_tok": "x_tok",
            "z": "z",
            "z_head": "z_head",
            "z_tok": "z_tok",
        },
        "role": "bwd_f32_cluster",
    },
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "fwd_packed": "cake_rmsnorm_silu_gate_det_train_3c42d6e202fd8d5c994b",
        "fwd_tiny": "cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_cluster": "cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "bwd_bf16_nodw": "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "bwd_bf16_nodw_long": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_nodw_strided": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "bwd_bf16_nodw_strided_packed": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "bwd_bf16_nodw_strided_packed_small": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "bwd_bf16_solo": "cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "bwd_f32_cluster": "cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
        "bwd_f32_solo": "cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "fwd_packed": "cake_rmsnorm_silu_gate_det_train_3c42d6e202fd8d5c994b",
        "fwd_tiny": "cake_rmsnorm_silu_gate_det_train_5a0540c2621ae3dbdb99",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_cluster": "cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "bwd_bf16_nodw": "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "bwd_bf16_nodw_long": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_nodw_strided": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "bwd_bf16_nodw_strided_packed": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "bwd_bf16_nodw_strided_packed_small": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "bwd_bf16_solo": "cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "bwd_f32_cluster": "cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
        "bwd_f32_solo": "cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "fwd_packed": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "fwd_tiny": "cake_rmsnorm_silu_gate_det_train_1e611ed07e9f796d9857",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_cluster": "cake_rmsnorm_silu_gate_det_train_1bc627a7e9eaa9aee52c",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "bwd_bf16_nodw_long": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_nodw_strided": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "bwd_bf16_nodw_strided_packed": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "bwd_bf16_nodw_strided_packed_small": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "bwd_bf16_solo": "cake_rmsnorm_silu_gate_det_train_36a3f270afc1a27290b7",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "bwd_f32_cluster": "cake_rmsnorm_silu_gate_det_train_de6420fe9e7e1fbc7137",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
        "bwd_f32_solo": "cake_rmsnorm_silu_gate_det_train_af462d9e22f9cca8b37d",
    },
}

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
# The kernels of a call, routed on every architecture.
ROLES = ("fwd", "bwd_bf16", "bwd_f32")
# The forwards of packed rows, routed on every architecture beside ``fwd``.
PACKED_ROLES = ("fwd_packed",)
# The tiny forward of packed rows, routed on the architectures with a cut in
# ``FWD_TINY_ROWS_PER_SM``: one block of its rule's ``rows_per_cta`` rows a
# CTA, no loop and no strides; it runs in the place of ``fwd_packed`` or
# ``fwd`` up to the cut, outside the rows of ``FWD_TINY_GAP_ROWS``
# (:func:`forward_module`), not at every row count, so it is not one of
# ``PACKED_ROLES``.
TINY_ROLES = ("fwd_tiny",)
# The suffix of a backward role's long role, and the long roles: a module that
# runs in its role's place on packed rows from its rule's ``min_rows_per_sm``
# rows per SM and, for a call that computes the weight gradient, on other rows
# from its ``min_rows_per_sm_strided`` (on packed rows only when its rule says
# ``packed_rows``; :func:`module_for_rows`).  The BF16 long module has both
# cuts; the FP32 one runs packed rows only.  The strided cuts are one row per
# SM past the default loop's L2 prefetch share (96 blocks per CTA on sm_100a
# and sm_103a, 64 on sm_107a) times 64 (16 rows a block, 4 CTAs per SM): 6145
# rows per SM on sm_100a and sm_103a, 4097 on sm_107a.  At the share and
# below it the default loop still warms each CTA's next block in L2 and is as
# fast as the look-ahead loop or faster; past it the default loop stops
# prefetching.  On GB300 (the package's programs, CUDA Graph replays, CUPTI
# kernel time with a cold L2) the look-ahead loop is 0.1 to 0.5 % slower at
# exactly 6144 rows per SM and 3.6 to 4.5 % faster from 6444 to 15000 on
# projection and head-major rows.  On a gate and ``grad_out`` sliced from
# projections (the same GPU, CUDA Graph replays, CUPTI kernel time with a
# cold L2, the same bits) the look-ahead loop is 4.4 to 5.3 % faster on B200
# and 4.2 to 4.7 % on GB300 from 7000 rows per SM and 3.5 to 4.0 % on VR200
# from 4500; it is at parity at 4500 on B200 and GB300 and 1.3 to 2.4 %
# slower at lower row counts below the share.  Only the BF16 weight gradient
# was measured.
LONG = "_long"
LONG_ROLES = ("bwd_bf16_long", "bwd_f32_long")
# The suffixes of the roles of a backward without the weight gradient, and
# those roles (:func:`nodw_module`): the BF16 backward built without the
# weight gradient, claimed (``bwd_bf16_nodw``), strided
# (``bwd_bf16_nodw_strided``: no claim, no ticket, no workspace), strided
# for packed rows (``bwd_bf16_nodw_strided_packed``: its 128-thread build,
# the strided role's program for packed rows and, at small row counts, for
# rows of any layout), strided for packed rows only at small row counts
# (``bwd_bf16_nodw_strided_packed_small``: the 128-thread build of packed
# rows only) and in the look-ahead loop (``bwd_bf16_nodw_long``); a call
# with ``compute_grad_weight=False`` runs one of them where its architecture
# routes it.
NODW = "_nodw"
STRIDED = "_strided"
PACKED = "_packed"
SMALL = "_small"
NODW_ROLES = (
    "bwd_bf16_nodw",
    "bwd_bf16_nodw_strided",
    "bwd_bf16_nodw_strided_packed",
    "bwd_bf16_nodw_strided_packed_small",
    "bwd_bf16_nodw_long",
)
# The suffixes of a backward role's small-row roles, and those roles
# (:func:`dw_module`): the role's backward in one CTA of at most 16 rows
# (``role + SOLO``) and in one cluster of 16 CTAs of at most 256 rows
# (``role + CLUSTER``), each with a ``cluster`` grid rule (its CTAs, whatever
# the rows, and the most rows it serves, ``max_rows``); a call that computes
# the weight gradient runs one of them in its role's module's place where
# its architecture routes it and the rows fit.  The cuts are the rules'
# ``max_rows``: measured as kernel time paired with the role's module (cold
# L2), the solo program takes 0.59 to 0.61 of its time at 16 rows (the
# cluster program 0.71 to 0.72) and the cluster program 0.66 to 0.75 of it
# from 16 to 256 rows.
SOLO = "_solo"
CLUSTER = "_cluster"
CLUSTER_ROLES = (
    "bwd_bf16_solo",
    "bwd_f32_solo",
    "bwd_bf16_cluster",
    "bwd_f32_cluster",
)
GRID_KINDS = (
    "persistent_row_blocks",
    "claimed_row_blocks",
    "strided_row_blocks",
    "cluster",
    "row_blocks",
)
# A call without the weight gradient runs the strided program (``role + NODW
# + STRIDED``, on packed rows ``role + NODW + STRIDED + PACKED`` where
# routed, and the 128-thread builds of the two tables below up to their
# cuts) up to this many rows per SM, by architecture and role, and the
# claimed program (``role + NODW``) past it: a role has a cut where its
# architecture routes both.  The cuts are measured per architecture (CUDA
# Graph replays, cold L2): up to them the strided program is the faster.
# sm_107a routes no claimed program (its strided program is as fast or
# faster at every row count measured), so it has no cut and the strided
# program runs every row count (:func:`nodw_module`).  The strided program
# launches the BF16 backward's ``min(ceil(rows / 16), 4 * SMs)`` 256-thread
# CTAs, the one of packed rows its own ``min(ceil(rows / 8), 8 * SMs)``
# 128-thread CTAs (no claim and no ticket): measured on sm_107a, the
# 128-thread build is the faster on packed rows (but on one token of head
# stride 128 with cold L2 the same or faster only up to 1024 rows per SM and
# slower from 1533), as fast or faster on other layouts up to
# ``NODW_ANY_LAYOUT_ROWS_PER_SM`` rows per SM (below), and the slower at
# about 128 rows per SM, where nearly every 128-thread CTA walks two blocks.
# Host plan only: a new cut needs no new program.
NODW_STRIDED_ROWS_PER_SM: dict[str, dict[str, int]] = {
    "sm_100a": {"bwd_bf16": 768},
    "sm_103a": {"bwd_bf16": 768},
    "sm_107a": {},
}
# A call without the weight gradient on packed rows runs the build of packed
# rows only (``role + NODW + STRIDED + PACKED + SMALL``) in the strided
# program's place up to this many rows per SM, by architecture and role,
# where the architecture routes it and the strided program; an architecture
# or role without a cut never runs it.  A cut lies at or below the strided
# cut.  Measured on sm_100a and sm_103a (CUDA Graph replays, cold L2): faster
# than the 128-thread strided build up to this cut, slower on sm_100a at 332
# rows per SM.  Measured on sm_107a (CUDA Graph replays, hot L2): faster at
# every measured row count up to 384 rows per SM but 96 (the same); the same
# from 512 to 1280, leaning slower (0.992 to 0.996), and faster at 1535
# (1.018).  Cold L2: up to this cut every row flagged slower read the same
# when timed again; past it rows from 224 to 384 rows per SM were flagged
# slower (by up to 5.1 %), and 256 rows per SM read 1.3 to 1.9 % slower in
# eager in six of eight timings on three GPUs, timed-again ones included
# (one, on the first full run, read 2.4 % faster), so the cut is the last
# measured point below them.  Host plan only.
PACKED_SMALL_ROWS_PER_SM: dict[str, dict[str, int]] = {
    "sm_100a": {"bwd_bf16": 128},
    "sm_103a": {"bwd_bf16": 128},
    "sm_107a": {"bwd_bf16": 192},
}
# A call without the weight gradient on rows that are not packed runs the
# 128-thread strided build (``role + NODW + STRIDED + PACKED``: other layouts
# read at their token and head strides, with the 256-thread program's
# addressing and row arithmetic) in the 256-thread one's place up to this
# many rows per SM, by architecture and role, where the architecture routes
# both; an architecture or role without a cut never runs it there.  Measured
# on sm_100a and sm_103a (CUDA Graph replays, cold L2): faster up to this
# cut, where each 128-thread CTA runs one 8-row block.  Measured on sm_107a
# (CUDA Graph replays, hot L2): faster up to 24 rows per SM and at this cut,
# the same from 32 to 64, slower at 128.  Host plan only.
NODW_ANY_LAYOUT_ROWS_PER_SM: dict[str, dict[str, int]] = {
    "sm_100a": {"bwd_bf16": 32},
    "sm_103a": {"bwd_bf16": 32},
    "sm_107a": {"bwd_bf16": 96},
}
# A forward of packed rows (head stride 128 and token stride ``H * 128`` on
# ``x`` and ``z``, or one token of head stride 128) runs the tiny forward
# (``fwd_tiny``) in the place of ``fwd_packed`` or ``fwd`` up to this many
# rows per SM, by architecture (:func:`forward_module`); an architecture with
# a cut of 0 routes no tiny forward and never runs one.  A cut is the largest
# measured rows per SM up to which the tiny forward is at no measured point
# slower than the forward it replaces: alone (cold L2, also after a delay
# kernel, and warm L2), in the checkpointed training step (two forwards and
# the backward) and beside kernels that hold some SMs.  Measured at 1 to 96
# rows per SM (48 heads): the tiny forward was first slower with warm L2, at
# 32 rows per SM on sm_100a, 36 on sm_103a and 48 on sm_107a.  Host plan
# only: a new cut above 0 needs no new program.
FWD_TINY_ROWS_PER_SM: dict[str, int] = {
    "sm_100a": 24,
    "sm_103a": 32,
    "sm_107a": 40,
}
# Rows ``(lo, hi]`` inside an architecture's tiny band that run the forward
# the tiny forward replaces (``fwd_packed`` or ``fwd``, as without the band;
# :func:`forward_module`); ``(0, 0)`` keeps none.  On sm_107a a call of 17 to
# 32 rows needs three or four CTAs of the tiny forward's 8-row blocks but
# two of the 16-row blocks of the forward it replaces, and with a cold L2 the
# third CTA costs more than the rows it takes.  Measured with a cold L2 at 8
# to 48 rows of one token and of packed rows, the replaced forward's time
# over the tiny forward's fell to 0.85 (the checkpointed training step's to
# 0.93) from 17 to 32 rows.  It was within noise of 1 or above from 8 to 16
# rows, and from 33 to 48 rows, where both forwards launch a third CTA.
FWD_TINY_GAP_ROWS: dict[str, tuple[int, int]] = {
    "sm_100a": (0, 0),
    "sm_103a": (0, 0),
    "sm_107a": (16, 32),
}
# Elements of one normalized row.
HEAD_DIM = 128
# The forward reads rows with 32-byte vector accesses (the backward with
# 16-byte ones): a tensor is read in place when its storage offset and its
# token and head strides are multiples of this many BF16 elements.
ALIGNMENT_ELEMENTS = 16
# The kernels index rows with 32-bit integers.
INT32_MAX = 2**31 - 1
# The ``fwd_packed`` forward serves fewer rows than this: the sm_107a one
# takes the offset of an element of row ``n`` as ``n * 128 + column`` in
# 32-bit arithmetic, and the sm_100a and sm_103a one keeps the same cut.
PACKED_ROWS = 2**31 // HEAD_DIM


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_module(arch: str, role: str) -> str:
    """The module of ``role`` on ``arch`` (``ValueError`` when none is exported)."""

    name = ROUTES.get(arch, {}).get(role)
    if name is None:
        raise ValueError(
            f"no exported Cake SiLU-gated RMSNorm {role!r} module on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return name


def min_rows_per_sm(name: str, arch: str) -> int:
    """The rows per SM from which long module ``name`` runs in its role's
    place on ``arch`` on packed rows (its rule's ``min_rows_per_sm``)."""

    return int(MODULES[name]["grid_rule"]["min_rows_per_sm"][arch])


def min_rows_per_sm_strided(name: str, arch: str) -> int | None:
    """The rows per SM from which long module ``name`` runs in its role's
    place on ``arch`` on rows that are not packed, for a call that computes
    the weight gradient: its rule's ``min_rows_per_sm_strided``, else None
    where its rule says ``packed_rows`` (packed rows only), else its
    :func:`min_rows_per_sm`."""

    rule = MODULES[name]["grid_rule"]
    if "min_rows_per_sm_strided" in rule:
        return int(rule["min_rows_per_sm_strided"][arch])
    if rule.get("packed_rows", False):
        return None
    return min_rows_per_sm(name, arch)


def module_for_rows(
    arch: str,
    role: str,
    rows: int,
    sm_count: int,
    packed: bool = False,
    compute_grad_weight: bool = True,
) -> str:
    """The module that runs ``role`` (one of ``ROLES``) for ``rows`` rows on
    ``arch`` with ``sm_count`` SMs, ``packed`` telling whether the kernel reads
    packed rows and ``compute_grad_weight`` whether the call computes the
    weight gradient: the role's long module (``role + LONG``) when ``arch``
    routes one and the rows are at least its cut times ``sm_count``, on packed
    rows its :func:`min_rows_per_sm`, on other rows its
    :func:`min_rows_per_sm_strided` (none for a module of packed rows only,
    and none for a call without the weight gradient, whose long program
    without it runs packed rows only); else :func:`route_module`.  Both give
    the same bits."""

    name = ROUTES.get(arch, {}).get(role + LONG)
    if name is not None:
        if packed:
            per_sm: int | None = min_rows_per_sm(name, arch)
        elif compute_grad_weight:
            per_sm = min_rows_per_sm_strided(name, arch)
        else:
            per_sm = None
        if per_sm is not None and int(rows) >= per_sm * int(sm_count):
            return name
    return route_module(arch, role)


def _fwd_plan(arch: str) -> tuple[str | None, int, str | None, str | None, int, int]:
    """``arch``'s forward plan: its ``fwd_tiny`` module and cut
    (``FWD_TINY_ROWS_PER_SM[arch]``) where it routes one and the cut is
    above 0, else None and 0; its ``fwd_packed`` module where it routes one,
    else None; its ``fwd`` module where it routes no long ``fwd`` module
    (then :func:`module_for_rows` of ``fwd`` names it at every row count),
    else None; and the rows ``(lo, hi]`` of its tiny band that run without
    it (``FWD_TINY_GAP_ROWS[arch]``, ``(0, 0)`` where none)."""

    routes = ROUTES[arch]
    cut = FWD_TINY_ROWS_PER_SM.get(arch, 0)
    base = routes.get("fwd") if "fwd" + LONG not in routes else None
    lo, hi = FWD_TINY_GAP_ROWS.get(arch, (0, 0))
    if "fwd_tiny" in routes and cut > 0:
        return routes["fwd_tiny"], cut, routes.get("fwd_packed"), base, lo, hi
    return None, 0, routes.get("fwd_packed"), base, 0, 0


# The forward plan of every routed architecture, read once at import: a
# forward looks its plan up once, tests its strides, compares ints and
# returns a module of the plan (its host path is most of the time of a small
# call).  Patching ``ROUTES``, ``FWD_TINY_ROWS_PER_SM`` or
# ``FWD_TINY_GAP_ROWS`` alone changes no forward route.
_FWD_PLANS = {arch: _fwd_plan(arch) for arch in ROUTES}


def forward_module(
    arch: str,
    rows: int,
    heads: int,
    sm_count: int,
    x_tok: int,
    x_head: int,
    z_tok: int,
    z_head: int,
) -> str:
    """The forward module for ``rows`` rows of ``heads`` heads on ``arch`` with
    ``sm_count`` SMs, given the token and head strides (in elements) the
    kernel reads ``x`` and ``z`` with:

    - the arch's ``fwd_tiny`` module up to ``FWD_TINY_ROWS_PER_SM[arch] *
      sm_count`` rows, outside the rows ``(lo, hi]`` of
      ``FWD_TINY_GAP_ROWS[arch]``, when it routes one and the rows are
      packed: both head strides 128 and both token strides ``heads * 128``,
      or one token (``rows == heads``) of token stride 0, which never enters
      an address (a broadcast of more tokens is not packed rows);
    - else the arch's ``fwd_packed`` module when it routes one and the rows
      are packed (both head strides 128, both token strides ``heads * 128``,
      fewer than ``PACKED_ROWS`` rows);
    - else :func:`module_for_rows` of ``fwd``.

    All give the same bits.  The tiny module, its cut, the packed module,
    the ``fwd`` module and the gap come from ``_FWD_PLANS``, read once at
    import; an architecture with no plan, or one that routes a long ``fwd``
    module, asks :func:`module_for_rows`."""

    plan = _FWD_PLANS.get(arch)
    if plan is None:
        return module_for_rows(arch, "fwd", rows, sm_count)
    if rows > 0 and x_head == HEAD_DIM and z_head == HEAD_DIM:
        row = heads * HEAD_DIM
        if x_tok == row and z_tok == row:
            if rows <= plan[1] * sm_count and not plan[4] < rows <= plan[5]:
                return plan[0]
            if plan[2] is not None and rows < PACKED_ROWS:
                return plan[2]
        elif (
            # one token: its token index is 0, so its stride is never read
            rows == heads
            and (x_tok == row or x_tok == 0)
            and (z_tok == row or z_tok == 0)
            and rows <= plan[1] * sm_count
            and not plan[4] < rows <= plan[5]
        ):
            return plan[0]
    if plan[3] is not None:
        return plan[3]
    return module_for_rows(arch, "fwd", rows, sm_count)


def forward_packed(
    heads: int, x_tok: int, x_head: int, z_tok: int, z_head: int
) -> bool:
    """Whether the forward kernels read ``x`` and ``z`` of ``heads`` heads,
    given their token and head strides (in elements), as packed rows: both
    head strides 128 and both token strides ``heads * 128`` (the kernel's test
    of each tensor by which it warms that tensor's rows in L2 as one byte
    range, here of both; a single token's stride, passed as 0, a broadcast
    over tokens, head-major rows and a gate sliced from a projection fail
    it).  :func:`launch_grid` takes it as ``packed``, and
    :func:`run_module` makes the same comparisons of a call's own arguments
    where a record's rule has a cut of rows not read as packed and the rows
    lie between its two cuts."""

    row = heads * HEAD_DIM
    return x_head == HEAD_DIM and z_head == HEAD_DIM and x_tok == row and z_tok == row


def backward_packed(
    heads: int,
    dy_tok: int,
    dy_head: int,
    x_tok: int,
    x_head: int,
    z_tok: int,
    z_head: int,
) -> bool:
    """Whether the backward kernels read ``dy``, ``x`` and ``z`` of ``heads``
    heads, given their token and head strides (in elements), as packed rows:
    all three head strides 128 and all three token strides ``heads * 128``
    (the kernel's own test)."""

    row = heads * HEAD_DIM
    return (
        dy_head == HEAD_DIM
        and x_head == HEAD_DIM
        and z_head == HEAD_DIM
        and dy_tok == row
        and x_tok == row
        and z_tok == row
    )


def backward_module(
    arch: str,
    role: str,
    rows: int,
    heads: int,
    sm_count: int,
    dy_tok: int,
    dy_head: int,
    x_tok: int,
    x_head: int,
    z_tok: int,
    z_head: int,
    compute_grad_weight: bool = True,
) -> str:
    """The module of backward ``role`` (``bwd_bf16`` or ``bwd_f32``) for
    ``rows`` rows of ``heads`` heads on ``arch`` with ``sm_count`` SMs, given
    the token and head strides (in elements) the kernel reads ``dy``, ``x`` and
    ``z`` with and whether the call computes the weight gradient:
    :func:`module_for_rows`, told the rows are packed when all three head
    strides are 128 and all three token strides ``heads * 128`` (the
    kernel's own test, :func:`backward_packed`, spelled out here: a call of
    it would cost a small backward's host path more than the test).  Every
    module of the role gives the same bits."""

    row = heads * HEAD_DIM
    packed = (
        dy_head == HEAD_DIM
        and x_head == HEAD_DIM
        and z_head == HEAD_DIM
        and dy_tok == row
        and x_tok == row
        and z_tok == row
    )
    return module_for_rows(arch, role, rows, sm_count, packed, compute_grad_weight)


def _nodw_keys(role: str) -> tuple[str | None, str, str, str, str]:
    """The ROUTES keys :func:`nodw_module` looks up for a module of backward
    ``role``: for a long role ``base + LONG`` the key ``base + NODW + LONG``
    (the other four empty), else ``None``, ``role + NODW + STRIDED``, that
    key ``+ PACKED``, that key ``+ SMALL`` and ``role + NODW``."""

    if role.endswith(LONG):
        return role[: -len(LONG)] + NODW + LONG, "", "", "", ""
    strided = role + NODW + STRIDED
    return None, strided, strided + PACKED, strided + PACKED + SMALL, role + NODW


# The keys of every role a module record names, spelled once at import: a
# call without the weight gradient then builds no string (its host path is
# most of the time of a small call).
_NODW_KEYS = {
    record["role"]: _nodw_keys(record["role"]) for record in MODULES.values()
}
# An empty table (read, never written): the default of a lookup that may
# miss, so :func:`nodw_module` makes no dict.
_EMPTY: dict[str, Any] = {}


def nodw_module(
    arch: str, name: str, rows: int, sm_count: int, packed: bool = False
) -> str:
    """The module of a backward without the weight gradient for ``rows``
    rows on ``arch`` with ``sm_count`` SMs, given ``name``, the module
    :func:`backward_module` names for the call's role (``bwd_bf16``), rows
    and layout with ``compute_grad_weight`` false (so a long module only on
    packed rows), and ``packed``, whether the kernel reads packed rows
    (:func:`backward_packed`; the backward also passes one token of head
    stride 128 as packed rows where ``PACKED_SMALL_ROWS_PER_SM[arch]`` has a
    cut, on an architecture without a strided cut only up to that cut):

    - for a long module (packed rows past its cut) the role's long module
      without the weight gradient (``role + NODW + LONG``) where ``arch``
      routes one, else ``name`` itself (run with ``weight_grad`` 0);
    - else, for packed rows up to ``PACKED_SMALL_ROWS_PER_SM[arch][role] *
      sm_count``, the module of packed rows only (``role + NODW + STRIDED +
      PACKED + SMALL``) where ``arch`` routes it and the strided module;
    - else the strided module (``role + NODW + STRIDED``; in its place
      ``role + NODW + STRIDED + PACKED`` where routed, for packed rows and
      for rows of any layout up to ``NODW_ANY_LAYOUT_ROWS_PER_SM[arch][role]
      * sm_count``) where ``arch`` routes one and ``rows <=
      NODW_STRIDED_ROWS_PER_SM[arch][role] * sm_count``;
    - else the claimed module (``role + NODW``) where routed, else the
      strided module where routed (an architecture without a claimed module
      has no cut: its strided module runs every row count), else ``name``.

    Every one gives the ``dx`` and ``dz`` bits of ``name``; the module of
    packed rows only never runs rows that are not packed.  The modules and
    cuts of ``arch`` and ``name`` come from :func:`_nodw_plan`, read once at
    import for every routed architecture and module (``_NODW_PLANS``)."""

    plan = _NODW_PLANS.get(arch, _EMPTY).get(name) or _nodw_plan(arch, name)
    done, strided, any_layout, small, small_cut, any_cut, cut, claimed = plan
    if done is not None:
        return done
    if packed:
        if rows <= small_cut * sm_count:
            return small
        strided = any_layout
    elif rows <= any_cut * sm_count:
        strided = any_layout
    if cut is not None and rows <= cut * sm_count:
        return strided
    if claimed is not None:
        return claimed
    return name if strided is None else strided


def _nodw_plan(
    arch: str, name: str
) -> tuple[
    str | None, str | None, str | None, str | None, int, int, int | None, str | None
]:
    """What :func:`nodw_module` reads for module ``name`` on ``arch``, from
    ``ROUTES`` and the three cut tables: ``(done, strided, any_layout, small,
    small_cut, any_cut, cut, claimed)``.  ``done`` is the answer for a long
    module (its long module without the weight gradient where routed, else
    ``name``), else ``None``; ``strided`` the strided module where routed;
    ``any_layout`` the module that reads any layout (``role + NODW + STRIDED +
    PACKED`` where routed, else ``strided``); ``small`` the module of packed
    rows only and ``small_cut`` its rows per SM (-1, which no row count is
    at most, where either the strided module or that module is not routed);
    ``any_cut`` the rows per SM of the module that reads any layout on other
    layouts (-1 without a strided module); ``cut`` the strided cut (``None``
    without a strided module or a cut); ``claimed`` the claimed module where
    routed."""

    routes = ROUTES.get(arch, _EMPTY)
    role = MODULES[name]["role"]
    keys = _NODW_KEYS.get(role) or _nodw_keys(role)
    long_key, strided_key, packed_key, small_key, claimed_key = keys
    if long_key is not None:
        return routes.get(long_key, name), None, None, None, -1, -1, None, None
    strided = routes.get(strided_key)
    claimed = routes.get(claimed_key)
    if strided is None:
        return None, None, None, None, -1, -1, None, claimed
    small = routes.get(small_key)
    small_cut = (
        -1 if small is None else PACKED_SMALL_ROWS_PER_SM.get(arch, _EMPTY).get(role, 0)
    )
    any_cut = NODW_ANY_LAYOUT_ROWS_PER_SM.get(arch, _EMPTY).get(role, 0)
    cut = NODW_STRIDED_ROWS_PER_SM.get(arch, _EMPTY).get(role)
    any_layout = routes.get(packed_key, strided)
    return None, strided, any_layout, small, small_cut, any_cut, cut, claimed


# The plan of every routed architecture and module, read once at import: a
# call without the weight gradient then makes two lookups and compares ints
# (its host path is most of the time of a small call).
_NODW_PLANS = {
    arch: {name: _nodw_plan(arch, name) for name in MODULES} for arch in ROUTES
}


def dw_module(arch: str, name: str, rows: int, clusters: bool = True) -> str:
    """The module of a backward that computes the weight gradient for
    ``rows`` rows on ``arch``, given ``name``, the module
    :func:`backward_module` names for the call's role (``bwd_bf16`` or
    ``bwd_f32``), rows and layout, and ``clusters``, whether the device runs
    the cluster programs (:func:`device_runs_clusters`; the backward passes
    it):

    - the solo program of ``name``'s role (``role + SOLO``: one CTA) up to
      its rule's ``max_rows`` where ``arch`` routes one;
    - else, where ``clusters`` holds, the cluster program (``role +
      CLUSTER``: one cluster of 16 CTAs) up to its rule's ``max_rows`` where
      ``arch`` routes one;
    - else ``name``.

    Only a role's module has these programs: any other module (a long one
    among them) is its own answer.  Each gives the ``dx``, ``dz`` and ``dw``
    bits of ``name`` on a device of at least 4 SMs, so ``clusters`` changes
    no bit there.  The programs and their rows come
    from :func:`_dw_plan`, read once at import for every routed
    architecture and module (``_DW_PLANS``)."""

    plan = _DW_PLANS.get(arch, _EMPTY).get(name)
    if plan is None:
        return name
    solo, solo_rows, cluster, cluster_rows = plan
    if rows <= solo_rows:
        return solo
    if clusters and rows <= cluster_rows:
        return cluster
    return name


def _dw_plan(
    arch: str, name: str
) -> tuple[str | None, int, str | None, int] | None:
    """What :func:`dw_module` reads for module ``name`` on ``arch``, from
    ``ROUTES`` and the small-row programs' rules: ``(solo, solo_rows,
    cluster, cluster_rows)``, the solo and the cluster program of ``name``'s
    role where routed and the most rows each serves (its rule's
    ``max_rows``; -1, which no row count is at most, where it is not
    routed), or ``None`` where neither is routed."""

    routes = ROUTES.get(arch, _EMPTY)
    role = MODULES[name]["role"]
    solo = routes.get(role + SOLO)
    cluster = routes.get(role + CLUSTER)
    if solo is None and cluster is None:
        return None
    return (
        solo,
        -1 if solo is None else int(MODULES[solo]["grid_rule"]["max_rows"]),
        cluster,
        -1 if cluster is None else int(MODULES[cluster]["grid_rule"]["max_rows"]),
    )


# The small-row plan of every routed architecture and module, read once at
# import: a call that computes the weight gradient then makes two lookups and
# compares ints (its host path is most of the time of a small call).
_DW_PLANS = {
    arch: {name: _dw_plan(arch, name) for name in MODULES} for arch in ROUTES
}

# The most rows a small-row program serves on any architecture (-1 where none
# is routed): a backward with the weight gradient past it keeps its module
# with one int compare and no lookup.
DW_SMALL_ROWS = max(
    (
        max(plan[1], plan[3])
        for plans in _DW_PLANS.values()
        for plan in plans.values()
        if plan is not None
    ),
    default=-1,
)


def workspace_elements(name: str) -> int:
    """FP32 elements of the workspace of backward module ``name``: its
    accumulator (``slots * limbs * HEAD_DIM`` 64-bit digits, two FP32 elements
    each), one 32-bit flag word per column, and the arrival ticket and the
    claim counter (a module whose record has a workspace: not a strided
    program)."""

    shape = MODULES[name]["workspace"]
    return 2 * int(shape["slots"]) * int(shape["limbs"]) * HEAD_DIM + HEAD_DIM + 2


def launch_grid(
    record: dict[str, Any],
    *,
    rows: int,
    sm_count: int,
    arch: str | None = None,
    packed: bool = True,
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device of
    architecture ``arch`` with ``sm_count`` SMs, ``packed`` telling whether
    the kernel reads the rows as packed (:func:`forward_packed`).  The forward
    launches one CTA per block of ``rows_per_block`` rows below its cut of
    rows per SM, and from there the fewest whole waves of ``ctas_per_sm *
    sm_count`` CTAs whose CTAs walk at most ``max_blocks_per_cta`` blocks
    each, and at most one CTA per block.  The cut is its rule's
    ``persistent_from_rows_per_sm``, or, with ``packed`` False, its
    ``persistent_from_rows_per_sm_strided`` where the rule has one (the
    forward of rows of any layout; the forward of packed rows only has
    none).  A cut keyed by architecture (every forward record's) gives
    ``arch``'s value, and with ``arch`` None or not one of its keys the call
    raises ``ValueError`` naming the record's kernel, ``arch`` and the keys (a
    single number serves every architecture).  The tiny forward launches
    ``ceil(rows / rows_per_cta)`` CTAs, one block each, whatever the SM
    count; the backward ``min(blocks, ctas_per_sm * sm_count)`` CTAs, and so
    do the strided programs, each of its own rule; a small-row program its
    rule's ``cluster`` CTAs (one cluster, or one CTA) whatever the rows and
    the SM count (its rule has no blocks or CTAs per SM).  Only the forward
    reads ``arch`` and ``packed``.  :func:`run_module` computes the same
    forward grid itself and calls this function for the other kinds."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    # The small-row kind is tested last, so the other kinds' launches pay
    # no extra compare; its rule has no blocks or CTAs per SM to read.  The
    # tiny forward's kind is tested right after the forward's: the
    # backward's launches pay one compare for it.  The forward's grid pays
    # a test of ``packed``, a type test and a lookup for its cut of ``arch``
    # here; run_module computes it for a launch of a cut keyed by the
    # device's architecture without calling this function.
    if kind == "persistent_row_blocks":
        blocks = -(-int(rows) // int(rule["rows_per_block"]))
        wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
        key = "persistent_from_rows_per_sm"
        if not packed and "persistent_from_rows_per_sm_strided" in rule:
            key = "persistent_from_rows_per_sm_strided"
        start = rule[key]
        if isinstance(start, dict):
            if arch not in start:
                raise ValueError(
                    f"Cake SiLU-gated RMSNorm module {record.get('kernel')!r}: "
                    f"its {key} is per architecture ({sorted(start)}); "
                    f"launch_grid got arch={arch!r}"
                )
            start = start[arch]
        if int(rows) < int(start) * int(sm_count):
            return (blocks, 1, 1)
        waves = max(1, -(-blocks // (wave * int(rule["max_blocks_per_cta"]))))
        return (min(blocks, waves * wave), 1, 1)
    if kind == "row_blocks":
        return (-(-int(rows) // int(rule["rows_per_cta"])), 1, 1)
    if kind in ("claimed_row_blocks", "strided_row_blocks"):
        blocks = -(-int(rows) // int(rule["rows_per_block"]))
        wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
        return (min(blocks, wave), 1, 1)
    if kind == "cluster":
        return (int(rule["cluster"]), 1, 1)
    raise ValueError(
        f"unknown Cake SiLU-gated RMSNorm grid rule: {kind!r} (known: {GRID_KINDS})"
    )


@functools.cache
def device_sm_count(device_index: int) -> int:
    """SM count of one CUDA device, read once per device (no device work)."""

    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


# The fewest SMs a device reports for :func:`device_runs_clusters` to run the
# cluster programs on it: an SM for each CTA of the cluster (the products
# report 148 and more; a device of fewer is a partition of one).
CLUSTER_MIN_SMS = 16


@functools.cache
def device_runs_clusters(device_index: int) -> bool:
    """Whether a backward on CUDA device ``device_index`` runs the cluster
    programs (one non-portable cluster of 16 CTAs; :func:`dw_module`), read
    once per device (no device work): the device reports at least
    ``CLUSTER_MIN_SMS`` SMs and is not a MIG instance (its name holds no
    ``MIG``).  A conservative test of what the host can read without a
    launch: elsewhere the role's module runs those rows, as before the
    cluster programs shipped, with the same bits on a device of at least 4
    SMs.  It does not query how many such clusters the device can hold (the
    package ships no occupancy query for these programs), so on a device it
    passes whose GPCs cannot hold the cluster's 16 CTAs at all (a partition
    its properties do not show) the launch fails with its CUDA error, which
    the call raises."""

    import torch

    properties = torch.cuda.get_device_properties(device_index)
    return (
        int(properties.multi_processor_count) >= CLUSTER_MIN_SMS
        and "MIG" not in str(properties.name)
    )


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            "Cake SiLU-gated RMSNorm target"
        )
    return arch


@functools.cache
def _sm_count_and_arch(device_index: int) -> tuple[int, str]:
    """``(device_sm_count(device_index), device_arch(device_index))``, read
    once per device: the one lookup of :func:`run_module`'s launches."""

    return device_sm_count(device_index), device_arch(device_index)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake SiLU-gated RMSNorm module {name} is not delivered for {arch}; "
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
    ahead-of-time warm-up)."""

    return cake_build.jit_specs(arch)


_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


@functools.cache
def _arguments(name: str) -> tuple[tuple[bool, Any, str], ...]:
    """Kernel arguments of module record ``name`` in launch order:
    ``(True, grid axis, parameter)`` for a grid dimension, else ``(False,
    logical name, parameter)``.  Cached: the module records are constant."""

    record = MODULES[name]
    logical = record["logical"]
    return tuple(
        (True, _GRID_AXES[key], key) if kind == "grid" else (False, logical[key], key)
        for kind, key in record["arg_plan"]
    )


@functools.cache
def _packed_slots(name: str) -> tuple[str, ...]:
    """The logical names of the ``heads``, ``x_tok``, ``x_head``, ``z_tok``
    and ``z_head`` arguments of module record ``name`` when its rule has a
    cut of rows not read as packed (``persistent_from_rows_per_sm_strided``),
    else ``()``: its grid does not depend on the layout.  Cached: the module
    records are constant."""

    record = MODULES[name]
    if "persistent_from_rows_per_sm_strided" not in record["grid_rule"]:
        return ()
    logical = record["logical"]
    return tuple(
        logical[key] for key in ("heads", "x_tok", "x_head", "z_tok", "z_head")
    )


# The records of the forward's grid kind (``persistent_row_blocks``), whose
# grid :func:`run_module` computes itself, each with the logical names of its
# ``heads``, ``x_tok``, ``x_head``, ``z_tok`` and ``z_head`` arguments (the
# kernel's test of packed rows, :func:`forward_packed`).  Built once: the
# records' kinds and arguments are constant (their cuts are read at each
# call).
_FORWARD_SLOTS = {
    name: tuple(
        record["logical"][key]
        for key in ("heads", "x_tok", "x_head", "z_tok", "z_head")
    )
    for name, record in MODULES.items()
    if record["grid_rule"]["kind"] == "persistent_row_blocks"
}


@functools.cache
def _entry(name: str, device_index: int) -> Any:
    """The tvm-ffi function of module record ``name`` on CUDA device ``device_index``."""

    return getattr(load(name, device_arch(device_index)), MODULES[name]["ffi_entry"])


@functools.cache
def _ffi_device(device_index: int) -> Any:
    """The tvm-ffi device of CUDA device ``device_index``."""

    return tvm_ffi.device(f"cuda:{device_index}")


@functools.cache
def current_raw_stream() -> Any:
    """A function of a CUDA device index that returns the handle of torch's
    current stream on that device (``torch.cuda.current_stream(index).cuda_stream``
    without making a stream object)."""

    import torch

    raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if raw is not None:
        return raw
    return lambda index: torch.cuda.current_stream(index).cuda_stream


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps logical names to tensors and Python scalars; the module
    record maps its kernel parameter names onto that vocabulary.  The launch
    grid is the module's :func:`launch_grid` on the device's SM count and
    architecture (:func:`_sm_count_and_arch`, read once per device).  The
    forward's grid (kind ``persistent_row_blocks``) is computed here with the
    fewest host operations, from the record's rule as it is at the call:
    its cuts of the device's architecture times the SMs, one CTA per block
    below both and the fewest whole waves from both (:func:`launch_grid`'s
    formula); between the two cuts of a rule with a cut of rows not read as
    packed (``persistent_from_rows_per_sm_strided``), the kernel's test of
    packed rows (:func:`forward_packed`) of the call's own ``heads``,
    ``x_tok``, ``x_head``, ``z_tok`` and ``z_head`` arguments picks the cut,
    so host and kernel agree on every call.  A cut of one number, a cut
    without the device's architecture (:func:`launch_grid` raises its
    ``ValueError``) and a missing argument take :func:`launch_grid` itself.
    The kernel runs on torch's current stream of CUDA device
    ``device_index``, the device of its tensors.
    """

    sm_count, arch = _sm_count_and_arch(device_index)
    record = MODULES[name]
    if name in _FORWARD_SLOTS:
        rule = record["grid_rule"]
        try:
            start = rule["persistent_from_rows_per_sm"][arch] * sm_count
            if "persistent_from_rows_per_sm_strided" in rule:
                strided = rule["persistent_from_rows_per_sm_strided"][arch] * sm_count
                # past one cut and below the other: the kernel's test of
                # packed rows (forward_packed) on this call's arguments
                # picks the cut
                if (rows < start) != (rows < strided):
                    heads, x_tok, x_head, z_tok, z_head = _FORWARD_SLOTS[name]
                    row = values[heads] * HEAD_DIM
                    if not (
                        values[x_head] == HEAD_DIM
                        and values[z_head] == HEAD_DIM
                        and values[x_tok] == row
                        and values[z_tok] == row
                    ):
                        start = strided
        except (KeyError, TypeError):
            # a cut of one number, a cut without this architecture (its
            # ValueError) or a missing argument: launch_grid itself
            grid = launch_grid(
                record,
                rows=rows,
                sm_count=sm_count,
                arch=arch,
                packed=forward_packed(
                    *[values.get(slot, 0) for slot in _FORWARD_SLOTS[name]]
                ),
            )
        else:
            blocks = -(-rows // rule["rows_per_block"])
            if rows < start:
                grid = (blocks, 1, 1)
            else:
                # launch_grid's whole waves: ``or 1`` is its floor of one CTA
                # a wave; its floor of one wave never binds (no rows, no CTAs)
                wave = rule["ctas_per_sm"] * sm_count or 1
                ctas = -(-blocks // (wave * rule["max_blocks_per_cta"])) * wave
                grid = (ctas if ctas < blocks else blocks, 1, 1)
    else:
        grid = launch_grid(record, rows=rows, sm_count=sm_count)
    arguments = _arguments(name)
    try:
        args = [grid[slot] if axis else values[slot] for axis, slot, _ in arguments]
    except KeyError:
        for axis, slot, key in arguments:
            if not axis and slot not in values:
                raise ValueError(
                    f"module {name} needs {slot!r} (kernel parameter {key!r})"
                ) from None
        raise
    entry = _entry(name, device_index)
    with tvm_ffi.use_raw_stream(
        _ffi_device(device_index), current_raw_stream()(device_index)
    ):
        entry(*args)


__all__ = [
    "ALIGNMENT_ELEMENTS",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CLUSTER",
    "CLUSTER_MIN_SMS",
    "CLUSTER_ROLES",
    "DW_SMALL_ROWS",
    "FWD_TINY_GAP_ROWS",
    "FWD_TINY_ROWS_PER_SM",
    "GRID_KINDS",
    "HEAD_DIM",
    "INT32_MAX",
    "LONG",
    "LONG_ROLES",
    "MODULES",
    "NODW",
    "NODW_ANY_LAYOUT_ROWS_PER_SM",
    "NODW_ROLES",
    "NODW_STRIDED_ROWS_PER_SM",
    "PACKED",
    "PACKED_ROLES",
    "PACKED_ROWS",
    "PACKED_SMALL_ROWS_PER_SM",
    "ROLES",
    "ROUTES",
    "SMALL",
    "SOLO",
    "STRIDED",
    "TINY_ROLES",
    "arch_for_capability",
    "backward_module",
    "backward_packed",
    "current_raw_stream",
    "device_arch",
    "device_runs_clusters",
    "device_sm_count",
    "dw_module",
    "forward_module",
    "forward_packed",
    "launch_grid",
    "library_name",
    "load",
    "min_rows_per_sm",
    "min_rows_per_sm_strided",
    "module_for_rows",
    "nodw_module",
    "route_module",
    "run_module",
    "specs_for_arch",
    "workspace_elements",
]
