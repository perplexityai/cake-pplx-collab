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
architecture may also route ``fwd_packed`` (``PACKED_ROLES``): a forward that
addresses the rows of contiguous ``[T, H, 128]`` inputs as one packed array,
without the token and head strides.  :func:`forward_module` runs it for such
inputs and ``fwd`` otherwise; both compute a row with the same FP32 operations
in the same order, so ``y`` is the same bits on either route.  Each backward
role also routes a long role (``LONG_ROLES``: ``bwd_bf16_long``,
``bwd_f32_long``): the same backward in a loop that loads a block's rows while
the previous block's row math runs, faster where a CTA runs many blocks.
:func:`backward_module` runs it in the role's place when ``dy``, ``x`` and
``z`` hold packed rows and the rows are at least its rule's
``min_rows_per_sm`` times the SM count, and the role's module otherwise; both
compute the rows and the weight gradient with the same FP32 operations in the
same order, so ``dx``, ``dz`` and ``dw`` are the same bits on either route.
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
look-ahead loop) where :func:`backward_module` runs ``bwd_bf16_long``;
sm_107a routes no claimed program, so there the strided program runs every
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
``rows_per_block`` rows below ``persistent_from_rows_per_sm`` rows per SM, and
from there the fewest whole waves of ``ctas_per_sm * SMs`` CTAs (the CTAs per
SM its launch bounds are compiled for) whose CTAs walk at most
``max_blocks_per_cta`` blocks each, and never more CTAs than blocks, CTA ``b``
walking the blocks ``b``, ``b + grid``, ``b + 2 * grid``, ... and loading each
next block while it computes the current one.  The backward launches
``min(blocks, ctas_per_sm * SMs)`` CTAs of ``rows_per_block``-row blocks; past
one block per CTA its CTAs claim the blocks in chunks whose rows are fixed by
the row count and the grid, add each thread's FP32 runs of rows into plain
FP32 words and each CTA's FP32 column words exactly into fixed-point digits,
so the weight gradient's bits depend on the rows and
the SM count only, not on which CTA runs a chunk.  A strided program
launches ``min(blocks, ctas_per_sm * SMs)`` CTAs of its own rule (the
256-thread one the BF16 backward's 16-row blocks and 4 CTAs per SM, the
128-thread ones 8-row blocks and 8 CTAs per SM:
``min(ceil(rows / 8), 8 * SMs)``), its CTAs walking their blocks with no
claim.  The backward's workspace
(``workspace`` of its record: ``slots`` accumulator copies of ``limbs``
64-bit digits per column, then one 32-bit flag word per column and the two
32-bit words of the arrival ticket and the claim counter) starts zero and the
kernel leaves it zero; a strided program's record has none, and it reads
and writes no workspace word.  No module is cooperative, and none waits for another
CTA or grid inside a kernel.  A kernel resident on another stream delays a
call roughly by its share of the SMs; a forward CTA whose slot such a kernel
holds walks all of its blocks once it starts, and a backward CTA that starts
late claims only the chunks left (a strided program's CTA, which claims
none, runs all of its own blocks once it starts).

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
    # the CTA a wave later, the gate and rstd by the approximate instructions
    # (build {"min_blocks": 6, "prefetch_ahead": 1, "prefetch_wave": 1,
    # "threads": 128}); whole waves of at most 2 blocks per CTA.
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
            "persistent_from_rows_per_sm": 8192,
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
            "persistent_from_rows_per_sm": 8192,
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
            "persistent_from_rows_per_sm": 8192,
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
    # on packed rows from min_rows_per_sm rows per SM (build
    # {"lookahead": True, "math": "ax", "min_blocks": 4, "out": "bf16",
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
            "packed_rows": True,
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
    # grid rule and workspace), so dx and dz are the BF16 backward's bits; a
    # call with compute_grad_weight=False runs it where backward_module runs
    # the BF16 long program (build {"dw": False, "lookahead": True, "math":
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
}

ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "bwd_bf16_nodw": "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "bwd_bf16_nodw_long": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_nodw_strided": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "bwd_bf16_nodw_strided_packed": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "bwd_bf16_nodw_strided_packed_small": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
    },
    "sm_103a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_36610c51a93167071e05",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "bwd_bf16_nodw": "cake_rmsnorm_silu_gate_det_train_cefc51faec88b7212f3c",
        "bwd_bf16_nodw_long": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_nodw_strided": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "bwd_bf16_nodw_strided_packed": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "bwd_bf16_nodw_strided_packed_small": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
    },
    "sm_107a": {
        "fwd": "cake_rmsnorm_silu_gate_det_train_5120481dd90c794f29e4",
        "fwd_packed": "cake_rmsnorm_silu_gate_det_train_8a371a49c065b60c88a9",
        "bwd_bf16": "cake_rmsnorm_silu_gate_det_train_8a6cb16beff15b6ffbfe",
        "bwd_bf16_long": "cake_rmsnorm_silu_gate_det_train_fef9d1cc2cc267219992",
        "bwd_bf16_nodw_long": "cake_rmsnorm_silu_gate_det_train_d8dd2e7dae0ca8233241",
        "bwd_bf16_nodw_strided": "cake_rmsnorm_silu_gate_det_train_cfe88bc164eeaa58821d",
        "bwd_bf16_nodw_strided_packed": "cake_rmsnorm_silu_gate_det_train_07de4b4c9a438a52ff79",
        "bwd_bf16_nodw_strided_packed_small": "cake_rmsnorm_silu_gate_det_train_38d04fbfa79b1461bc0c",
        "bwd_f32": "cake_rmsnorm_silu_gate_det_train_ae2073f31d0923e61736",
        "bwd_f32_long": "cake_rmsnorm_silu_gate_det_train_be861a132bbc76cb6433",
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
# The forwards of packed rows, routed on some architectures beside ``fwd``.
PACKED_ROLES = ("fwd_packed",)
# The suffix of a backward role's long role, and the long roles: a module that
# runs in its role's place from its rule's ``min_rows_per_sm`` rows per SM, on
# packed rows only when its rule says ``packed_rows`` (:func:`module_for_rows`).
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
GRID_KINDS = ("persistent_row_blocks", "claimed_row_blocks", "strided_row_blocks")
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
# 128-thread build is the faster on packed rows, as fast or faster on other
# layouts up to ``NODW_ANY_LAYOUT_ROWS_PER_SM`` rows per SM (below), and the
# slower at about 128 rows per SM, where nearly every 128-thread CTA walks
# two blocks.  Host plan only: a new cut needs no new program.
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
# every measured row count up to this cut but 96 rows per SM (the same); the
# same from 512 to 1280, leaning slower (0.992 to 0.996), and faster at 1535
# (1.018).  The cut is the last faster point below that band.  Cold L2:
# every row flagged slower read the same or faster when timed again (one of
# them only on another GPU).  Host plan only.
PACKED_SMALL_ROWS_PER_SM: dict[str, dict[str, int]] = {
    "sm_100a": {"bwd_bf16": 128},
    "sm_103a": {"bwd_bf16": 128},
    "sm_107a": {"bwd_bf16": 384},
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
# Elements of one normalized row.
HEAD_DIM = 128
# The forward reads rows with 32-byte vector accesses (the backward with
# 16-byte ones): a tensor is read in place when its storage offset and its
# token and head strides are multiples of this many BF16 elements.
ALIGNMENT_ELEMENTS = 16
# The kernels index rows with 32-bit integers.
INT32_MAX = 2**31 - 1
# A forward of packed rows takes the offset of an element of row ``n`` as
# ``n * 128 + column`` in 32-bit arithmetic: it serves fewer rows than this.
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
    place on ``arch`` (its rule's ``min_rows_per_sm``)."""

    return int(MODULES[name]["grid_rule"]["min_rows_per_sm"][arch])


def module_for_rows(
    arch: str, role: str, rows: int, sm_count: int, packed: bool = False
) -> str:
    """The module that runs ``role`` (one of ``ROLES``) for ``rows`` rows on
    ``arch`` with ``sm_count`` SMs, ``packed`` telling whether the kernel reads
    packed rows: the role's long module (``role + LONG``) when ``arch`` routes
    one, the rows are at least its :func:`min_rows_per_sm` times ``sm_count``
    and ``packed`` holds or its rule does not say ``packed_rows``; else
    :func:`route_module`.  Both give the same bits."""

    name = ROUTES.get(arch, {}).get(role + LONG)
    if (
        name is not None
        and (packed or not MODULES[name]["grid_rule"].get("packed_rows", False))
        and int(rows) >= min_rows_per_sm(name, arch) * int(sm_count)
    ):
        return name
    return route_module(arch, role)


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
    kernel reads ``x`` and ``z`` with: the arch's ``fwd_packed`` module when it
    routes one and the rows are packed (both head strides 128, both token
    strides ``heads * 128``, fewer than ``PACKED_ROWS`` rows), else
    :func:`module_for_rows` of ``fwd``.  Both give the same bits."""

    packed = ROUTES.get(arch, {}).get("fwd_packed")
    row = heads * HEAD_DIM
    if (
        packed is not None
        and 0 < rows < PACKED_ROWS
        and x_head == HEAD_DIM
        and z_head == HEAD_DIM
        and x_tok == row
        and z_tok == row
    ):
        return packed
    return module_for_rows(arch, "fwd", rows, sm_count)


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
) -> str:
    """The module of backward ``role`` (``bwd_bf16`` or ``bwd_f32``) for
    ``rows`` rows of ``heads`` heads on ``arch`` with ``sm_count`` SMs, given
    the token and head strides (in elements) the kernel reads ``dy``, ``x`` and
    ``z`` with: :func:`module_for_rows`, told the rows are packed when all
    three head strides are 128 and all three token strides ``heads * 128`` (the
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
    return module_for_rows(arch, role, rows, sm_count, packed)


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
    and layout, and ``packed``, whether the kernel reads packed rows
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


def workspace_elements(name: str) -> int:
    """FP32 elements of the workspace of backward module ``name``: its
    accumulator (``slots * limbs * HEAD_DIM`` 64-bit digits, two FP32 elements
    each), one 32-bit flag word per column, and the arrival ticket and the
    claim counter (a module whose record has a workspace: not a strided
    program)."""

    shape = MODULES[name]["workspace"]
    return 2 * int(shape["slots"]) * int(shape["limbs"]) * HEAD_DIM + HEAD_DIM + 2


def launch_grid(
    record: dict[str, Any], *, rows: int, sm_count: int
) -> tuple[int, int, int]:
    """The grid of one module record for ``rows`` rows on a device with
    ``sm_count`` SMs.  The forward launches one CTA per block of
    ``rows_per_block`` rows below ``persistent_from_rows_per_sm`` rows per SM,
    and from there the fewest whole waves of ``ctas_per_sm * sm_count`` CTAs
    whose CTAs walk at most ``max_blocks_per_cta`` blocks each, and at most one
    CTA per block; the backward ``min(blocks, ctas_per_sm * sm_count)`` CTAs,
    and so do the strided programs, each of its own rule."""

    rule = record["grid_rule"]
    kind = rule["kind"]
    blocks = -(-int(rows) // int(rule["rows_per_block"]))
    wave = max(1, int(rule["ctas_per_sm"]) * int(sm_count))
    if kind == "persistent_row_blocks":
        if int(rows) < int(rule["persistent_from_rows_per_sm"]) * int(sm_count):
            return (blocks, 1, 1)
        waves = max(1, -(-blocks // (wave * int(rule["max_blocks_per_cta"]))))
        return (min(blocks, waves * wave), 1, 1)
    if kind in ("claimed_row_blocks", "strided_row_blocks"):
        return (min(blocks, wave), 1, 1)
    raise ValueError(
        f"unknown Cake SiLU-gated RMSNorm grid rule: {kind!r} (known: {GRID_KINDS})"
    )


@functools.cache
def device_sm_count(device_index: int) -> int:
    """SM count of one CUDA device, read once per device (no device work)."""

    import torch

    return torch.cuda.get_device_properties(device_index).multi_processor_count


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
    grid is the module's :func:`launch_grid` on the device's SM count, and the
    kernel runs on torch's current stream of CUDA device ``device_index``, the
    device of its tensors.
    """

    grid = launch_grid(MODULES[name], rows=rows, sm_count=device_sm_count(device_index))
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
    "STRIDED",
    "arch_for_capability",
    "backward_module",
    "backward_packed",
    "current_raw_stream",
    "device_arch",
    "device_sm_count",
    "forward_module",
    "launch_grid",
    "library_name",
    "load",
    "min_rows_per_sm",
    "module_for_rows",
    "nodw_module",
    "route_module",
    "run_module",
    "specs_for_arch",
    "workspace_elements",
]
