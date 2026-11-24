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

Registry of the Cake dense multi-head attention backward kernels, head
dimension 64 or 128, non-causal or causal (SM100 / SM103 / SM107).  The
loader and the launch plan are :mod:`.cake_backend`.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record is keyed by its stage name for the head-dimension 128
build (non-causal for ``bwd_main``), or ``<stage>_<variant>`` for the other
builds of the stage (``bwd_main_2cta[_causal]`` and ``bwd_post_pair`` are the
pair route's), and carries the host plan: the ``stage`` it serves, the
architectures it is delivered for and the ``run`` entry.  The derived leaves
(``library``, ``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding; ``library`` is the program identity
(``cake_fmha_train_<20 hex>``), so a kernel swap renames it.  ``MODULES`` is a
literal table (the exporter and the manifest checker read it with
``ast.literal_eval``).

The stages of the three-kernel routes, in launch order: ``bwd_pre``
(``dPsum = rowsum(O * dO)``, ``LSE * log2(e)`` and the cleared FP32 dQ
accumulator), ``bwd_main`` (per key block: S, P, dP and dS against every
query block, dK and dV stored once, the dQ tiles added into the FP32
accumulator by bulk reductions) and ``bwd_post`` (``dQ = bf16(acc *
softmax_scale)`` in the layout of Q).  The dQ tiles of the
key blocks of one head reach the accumulator in scheduling order, so dQ is
not bitwise reproducible run to run; dK, dV and the preprocess outputs have
one fixed order.  Every launch is a plain stream-ordered launch: nothing
cooperative, no PDL.  Each of these kernels runs on the grid
``(S / 128, H, B)``: S is the query length for ``bwd_pre`` and ``bwd_post``
and the key length for ``bwd_main``.

The pair route (head dimension 128, a key length that is a multiple of 256, and
either at least as many queries as keys with ``B * H`` and the key length in a
region of ``PAIR_REGIONS`` for the architecture and mask, or no mask and fewer
queries than keys within the bounds of ``PAIR_FEWER_QUERIES`` for the
architecture): the main stage is the 2-CTA
kernel (``bwd_main_2cta`` / ``bwd_main_2cta_causal``, clusters of two CTAs
along x: one CTA pair per 256 keys), which adds the dQ tiles in the pair-row
order, and the postprocess is the build that reads that order
(``bwd_post_pair``).  The two change together: the other postprocess would
read the pair-row order as the tile order.  :func:`variant` picks the route of
a call.  Every record ships for the three architectures; where no region
routes a call (``sm_100a`` with the mask) the pair build ships and no call
takes it.

The one-launch route (head dimension 128 or 64, without the mask or with it):
one kernel, the ``bwd_fused`` stage (``bwd_fused`` / ``bwd_fused_hd64``
without the mask, ``bwd_fused_causal`` / ``bwd_fused_hd64_causal`` with it),
does the whole backward of a call on few CTAs (short sequences, or long ones
with few heads).  On the grid ``(Sq / 128 + Skv / 128, H, B)`` the first
``Sq / 128`` CTAs of a (head, batch) are Q-owners, each the dq of one
query block, and the others are KV-owners, each dk and dv of one key block;
the kernel makes ``LSE * log2(e)`` and ``dPsum`` itself, needs no workspace
and writes dq, dk and dv once each in one fixed order.  ``FUSED_OF`` names its
record per (head dimension, causal) and :func:`fused_record` picks the
one-launch route for a call without the mask inside one of the pieces of
``FUSED_REGION``, and for a causal call (``Sq == Skv``) inside one of the
pieces of ``FUSED_CAUSAL_REGION``; every other call takes the route of
:func:`variant`.  No ``RECORD_OF`` variant and no host sequence names a
``bwd_fused`` record.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
HEAD_DIMS = (64, 128)
# Rows of a query block and of a key block; the sequence lengths are multiples of it.
BLOCK = 128
PACKAGE = "cake_pplx._backends.cake_fmha_nondet_train"
LIBRARY_PREFIX = "cake_fmha_train"
MANIFEST = "build.json"

# The stages: the three of one three-kernel backward call, in launch order,
# then the one-launch backward.
STAGES = ("bwd_pre", "bwd_main", "bwd_post", "bwd_fused")

MODULES: dict[str, dict[str, Any]] = {
    "bwd_pre": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "O"], ["buffer", "dO"], ["buffer", "LSE"], ["buffer", "LSE_log2"], ["buffer", "dPsum"], ["buffer", "dQ_accum"], ["parameter", "seqlen_q"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_f0b639ee014a01d20db0",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_f0b639ee014a01d20db0",
        "stage": "bwd_pre",
    },
    "bwd_pre_hd64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "O"], ["buffer", "dO"], ["buffer", "LSE"], ["buffer", "LSE_log2"], ["buffer", "dPsum"], ["buffer", "dQ_accum"], ["parameter", "seqlen_q"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_25802c9246ef8b10ba9f",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_25802c9246ef8b10ba9f",
        "stage": "bwd_pre",
    },
    "bwd_main": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "LSE_log2_gmem"], ["buffer", "dPsum_gmem"], ["buffer", "dK_gmem"], ["buffer", "dV_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_32aa42a31904b8901ae7",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231936,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_32aa42a31904b8901ae7",
        "stage": "bwd_main",
    },
    "bwd_main_causal": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "LSE_log2_gmem"], ["buffer", "dPsum_gmem"], ["buffer", "dK_gmem"], ["buffer", "dV_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_d63e403fa023c1cfba55",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231936,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_d63e403fa023c1cfba55",
        "stage": "bwd_main",
    },
    "bwd_main_hd64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "LSE_log2_gmem"], ["buffer", "dPsum_gmem"], ["buffer", "dK_gmem"], ["buffer", "dV_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_18086b28fd0ceef0c320",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 182784,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_18086b28fd0ceef0c320",
        "stage": "bwd_main",
    },
    "bwd_main_hd64_causal": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "LSE_log2_gmem"], ["buffer", "dPsum_gmem"], ["buffer", "dK_gmem"], ["buffer", "dV_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_1f0c2f01d6f6b42a6b60",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 182784,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_1f0c2f01d6f6b42a6b60",
        "stage": "bwd_main",
    },
    "bwd_main_2cta": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "LSE_log2_gmem"], ["buffer", "dPsum_gmem"], ["buffer", "dK_gmem"], ["buffer", "dV_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_95e4e45dde43783fdf83",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_95e4e45dde43783fdf83",
        "stage": "bwd_main",
    },
    "bwd_main_2cta_causal": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "LSE_log2_gmem"], ["buffer", "dPsum_gmem"], ["buffer", "dK_gmem"], ["buffer", "dV_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_f52478391bbcd6308069",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (2, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 231424,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_f52478391bbcd6308069",
        "stage": "bwd_main",
    },
    "bwd_post": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "dQ"], ["parameter", "seqlen_q"], ["parameter", "softmax_scale"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_465f6043909255815e00",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 65536,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_465f6043909255815e00",
        "stage": "bwd_post",
    },
    "bwd_post_hd64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "dQ"], ["parameter", "seqlen_q"], ["parameter", "softmax_scale"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_7057c4c605e7b2cc2436",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 32768,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_7057c4c605e7b2cc2436",
        "stage": "bwd_post",
    },
    "bwd_post_pair": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ_accum"], ["buffer", "dQ"], ["parameter", "seqlen_q"], ["parameter", "softmax_scale"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_1a125639d68d30b57111",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 65536,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_1a125639d68d30b57111",
        "stage": "bwd_post",
    },
    "bwd_fused": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ"], ["buffer", "LSE"], ["buffer", "O_gmem"], ["buffer", "dO_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_n_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_956bb8b5689c91f2dafe",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 199168,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_956bb8b5689c91f2dafe",
        "stage": "bwd_fused",
    },
    "bwd_fused_hd64": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ"], ["buffer", "LSE"], ["buffer", "O_gmem"], ["buffer", "dO_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_n_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_535529048441929389ae",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 117248,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_535529048441929389ae",
        "stage": "bwd_fused",
    },
    "bwd_fused_causal": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ"], ["buffer", "LSE"], ["buffer", "O_gmem"], ["buffer", "dO_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_n_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_45471caca3efd9427760",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 199168,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_45471caca3efd9427760",
        "stage": "bwd_fused",
    },
    "bwd_fused_hd64_causal": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dQ"], ["buffer", "LSE"], ["buffer", "O_gmem"], ["buffer", "dO_gmem"], ["buffer", "Q"], ["buffer", "K"], ["buffer", "V"], ["buffer", "dO"], ["buffer", "dK"], ["buffer", "dV"], ["parameter", "seqlen_q"], ["parameter", "seqlen_kv"], ["parameter", "softmax_scale"], ["parameter", "softmax_scale_log2"], ["parameter", "num_m_blocks"], ["parameter", "num_n_blocks"], ["parameter", "num_heads"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_fmha_train_38ec1d22736060c57a93",
        "launch": {
            "block": (512, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 117248,
            "use_pdl": False,
        },
        "library": "cake_fmha_train_38ec1d22736060c57a93",
        "stage": "bwd_fused",
    },
}

# (head dim, causal, pair) -> the records of the three stages, in launch order.
# The preprocess depends on the head dimension only; the postprocess reads the
# dQ accumulator in the order of the variant's main stage: the pair-row order
# of the 2-CTA main kernels (``pair`` True, head dimension 128 only) or the
# tile order of the others.
RECORD_OF: dict[tuple[int, bool, bool], tuple[str, str, str]] = {
    (128, False, False): ("bwd_pre", "bwd_main", "bwd_post"),
    (128, True, False): ("bwd_pre", "bwd_main_causal", "bwd_post"),
    (64, False, False): ("bwd_pre_hd64", "bwd_main_hd64", "bwd_post_hd64"),
    (64, True, False): ("bwd_pre_hd64", "bwd_main_hd64_causal", "bwd_post_hd64"),
    (128, False, True): ("bwd_pre", "bwd_main_2cta", "bwd_post_pair"),
    (128, True, True): ("bwd_pre", "bwd_main_2cta_causal", "bwd_post_pair"),
}

# (head dim, causal, pair) -> the host sequence of the variant: one library whose ``run`` launches the three records
# RECORD_OF names for it, in order, in one host call.  Derived by the exporter.
SEQUENCES: dict[tuple[int, bool, bool], dict[str, Any]] = {
    (128, False, False): {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "library": "cake_fmha_train_495bb89b89eb8b41327e",
        "records": ("bwd_pre", "bwd_main", "bwd_post"),
    },
    (128, True, False): {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "library": "cake_fmha_train_5f9dc314733948320c47",
        "records": ("bwd_pre", "bwd_main_causal", "bwd_post"),
    },
    (64, False, False): {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "library": "cake_fmha_train_a69536754fc8eb40cf51",
        "records": ("bwd_pre_hd64", "bwd_main_hd64", "bwd_post_hd64"),
    },
    (64, True, False): {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "library": "cake_fmha_train_f4f0d4f37340ee62f928",
        "records": ("bwd_pre_hd64", "bwd_main_hd64_causal", "bwd_post_hd64"),
    },
    (128, False, True): {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "library": "cake_fmha_train_354c53aa1bd7a12eec41",
        "records": ("bwd_pre", "bwd_main_2cta", "bwd_post_pair"),
    },
    (128, True, True): {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "library": "cake_fmha_train_011ecf5b8485c428cdc9",
        "records": ("bwd_pre", "bwd_main_2cta_causal", "bwd_post_pair"),
    },
}

# (head dim, causal) -> the record of the one-launch backward (the ``bwd_fused``
# stage): the build without the mask, or the causal build (top-left mask,
# ``Sq == Skv``).
FUSED_OF: dict[tuple[int, bool], str] = {
    (128, False): "bwd_fused",
    (64, False): "bwd_fused_hd64",
    (128, True): "bwd_fused_causal",
    (64, True): "bwd_fused_hd64_causal",
}


def grid(stage: str, batch: int, heads: int, seqlen_q: int, seqlen_kv: int) -> tuple[int, int, int]:
    """The launch grid of ``stage``: one CTA per 128-row block of one (head,
    batch); query blocks for ``bwd_pre`` and ``bwd_post``, key blocks for
    ``bwd_main``, and for ``bwd_fused`` the query blocks (the Q-owners) then
    the key blocks (the KV-owners).  The kernels read the head from
    ``blockIdx.y`` and the batch from ``blockIdx.z``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {list(STAGES)}")
    if stage == "bwd_fused":
        return (seqlen_q // BLOCK + seqlen_kv // BLOCK, heads, batch)
    rows = seqlen_kv if stage == "bwd_main" else seqlen_q
    return (rows // BLOCK, heads, batch)


# Architecture -> mask (``causal``) -> the regions of the pair route (:func:`variant`).  A region ``(least, most,
# length)`` holds the calls with ``least <= B * H`` (and ``B * H <= most`` unless ``most`` is None) and at least
# ``length`` keys; the regions route only calls with at least as many queries as keys.  Every measured call inside
# a region was FASTER than 0.1.4 in every counted round of the crossover run, beyond the identical-kernel controls'
# single-round noise, and every bound is a measured value (``B * H`` 1, 4, 16, 32 and 128; lengths 256 to 8192).  The
# ``sm_103a`` causal region ``(32, None, 8192)`` comes from a later measurement against the 1-CTA route (``B * H`` 16,
# 32, 64, 128 and 256 with 4096, 8192 and 16384 keys): every measured call inside it was faster in every counted round
# of both replicate runs, in eager calls and in CUDA-graph replays, beyond the identical-kernel controls' noise; at
# ``B * H`` 128 and 256 with 4096 keys, and at ``B * H`` 16 with 8192 keys, not every run was, so calls with fewer
# keys or fewer heads keep the 1-CTA route.  An empty tuple routes no call: the ``sm_100a`` causal pair build ships
# and no call takes it.
PAIR_REGIONS: dict[str, dict[bool, tuple[tuple[int, int | None, int], ...]]] = {
    "sm_100a": {False: ((1, 1, 4096), (1, 4, 8192), (128, None, 8192)), True: ()},
    "sm_103a": {False: ((1, 4, 2048), (1, None, 8192)), True: ((1, 4, 2048), (32, None, 8192))},
    "sm_107a": {
        False: ((1, 4, 2048), (1, None, 4096)),
        True: ((1, 4, 2048), (1, 16, 4096), (1, None, 8192)),
    },
}
# Architecture -> mask -> the least ``length`` of its regions, or None (no region): a call with fewer keys takes the
# 1-CTA route before the regions are read.
PAIR_LEAST: dict[str, dict[bool, int | None]] = {
    arch: {causal: min((length for _, _, length in regions), default=None) for causal, regions in masks.items()}
    for arch, masks in PAIR_REGIONS.items()
}
# Architecture -> the bounds of the pair route for a call without the mask with fewer queries than keys, or None (no
# such call takes it): ``(least_count, least_q, least_kv, most_per_q)``.  :func:`variant` routes such a call (head
# dimension 128, a key length that is a multiple of 256) to the pair route when ``B * H >= least_count``, ``Sq >=
# least_q``, ``Skv >= least_kv`` and ``Skv <= most_per_q * Sq``.  Every measured call inside the bounds was faster than
# 0.1.8 (the 1-CTA route) in every counted round of both replicate runs of the CUDA-graph timing, beyond the
# identical-kernel controls' noise, and every bound is a measured value (``B * H`` 8, 4096 queries, 8192 keys, twice
# the queries).  ``sm_100a`` has none: no measured call was faster there in both replicate runs.
PAIR_FEWER_QUERIES: dict[str, tuple[int, int, int, int] | None] = {
    "sm_100a": None,
    "sm_103a": (8, 4096, 8192, 2),
    "sm_107a": (8, 4096, 8192, 2),
}


def variant(
    D: int, causal: bool, *, batch: int, heads: int, seqlen_q: int, seqlen_kv: int, arch: str
) -> tuple[int, bool, bool]:
    """The ``RECORD_OF`` key of a call on ``arch``: ``(D, causal, pair)``.
    ``pair`` is True for the pair route: head dimension 128, a key length that
    is a multiple of ``2 * BLOCK`` (a CTA pair of the 2-CTA main kernel covers
    two key blocks), and either at least as many queries as keys with
    ``batch * heads`` and ``seqlen_kv`` in a region of
    ``PAIR_REGIONS[arch][causal]``, or no mask and fewer queries than keys
    within the bounds ``PAIR_FEWER_QUERIES[arch]``.  ``arch``
    is one of ``ARCHES``; every argument after ``causal`` is keyword-only.  An
    unknown ``arch`` raises ``KeyError`` (the backend's ``device_arch`` raises
    ``RuntimeError`` for a device outside ``ARCHES`` before the call)."""
    causal = bool(causal)
    least = PAIR_LEAST[arch][causal]
    if seqlen_q < seqlen_kv:
        bounds = PAIR_FEWER_QUERIES[arch]
        if D != 128 or causal or bounds is None or seqlen_kv % (2 * BLOCK):
            return (D, causal, False)
        least_count, least_q, least_kv, most_per_q = bounds
        pair = batch * heads >= least_count and seqlen_q >= least_q and least_kv <= seqlen_kv <= most_per_q * seqlen_q
        return (D, causal, pair)
    if D != 128 or least is None or seqlen_kv < least or seqlen_kv % (2 * BLOCK):
        return (D, causal, False)
    count = batch * heads
    for low, high, length in PAIR_REGIONS[arch][causal]:
        if low <= count and (high is None or count <= high) and seqlen_kv >= length:
            return (D, causal, True)
    return (D, causal, False)


# (architecture, head dim) -> the region of the one-launch route there, or
# None (no call takes it): one or more pieces, each (max_ctas, s_max,
# skv_any, skv_per_sq).  A call without the mask is inside a piece when
# ``B * H * (Sq / 128 + Skv / 128) <= max_ctas`` (the CTAs of the one-launch
# grid), ``max(Sq, Skv) <= s_max`` and ``Skv <= skv_any`` or ``Skv <=
# skv_per_sq * Sq``, and inside the region when it is inside any of its
# pieces (:func:`fused_record`).  Each entry is the measured region of its
# architecture and head dimension, where the one-launch route measured faster
# than the three-kernel routes; pieces may overlap, and their order does not
# change the route.
FUSED_REGION: dict[tuple[str, int], tuple[tuple[int, int, int, int], ...] | None] = {
    ("sm_100a", 128): (
        (152, 2048, 512, 1),
        (190, 1024, 256, 0),
        (152, 1024, 0, 3),
        (640, 512, 256, 0),
        (301, 1024, 128, 0),
        (128, 8192, 0, 1),
    ),
    ("sm_100a", 64): (
        (152, 2048, 512, 2),
        (190, 1024, 256, 0),
        (152, 1024, 0, 6),
        (606, 256, 128, 0),
        (212, 256, 0, 1),
    ),
    ("sm_103a", 128): (
        (152, 2048, 512, 1),
        (190, 1024, 256, 0),
        (152, 1024, 0, 3),
        (303, 512, 256, 0),
        (301, 1024, 128, 0),
        (128, 8192, 0, 1),
    ),
    ("sm_103a", 64): (
        (204, 2048, 256, 0),
        (156, 512, 0, 2),
        (256, 1024, 128, 0),
        (152, 1024, 512, 3),
        (152, 2048, 0, 1),
    ),
    ("sm_107a", 128): (
        (256, 2048, 512, 0),
        (320, 1024, 256, 0),
        (423, 512, 256, 0),
        (210, 2048, 768, 2),
        (423, 1024, 128, 0),
        (128, 8192, 0, 1),
    ),
    ("sm_107a", 64): (
        (212, 2048, 512, 2),
        (210, 1024, 0, 6),
        (423, 512, 256, 0),
        (320, 1024, 256, 0),
        (423, 1024, 128, 0),
    ),
}

# (architecture, head dimension) -> the region of the one-launch route for a
# causal call there, or None (no causal call takes it): one or more pieces,
# each (least_ctas, most_ctas, least_s, most_s).  A causal call (top-left
# mask, ``Sq == Skv == S``) is inside a piece when ``least_ctas <= B * H *
# (Sq / 128 + Skv / 128) <= most_ctas`` (the CTAs of the one-launch grid) and
# ``least_s <= S <= most_s`` (a lower bound of 0 bounds nothing), and inside
# the region when it is inside any of its pieces (:func:`fused_record`).
# Every bound is a measured call.  Inside the pieces without a lower bound
# every measured causal call (``B * H`` 1 to 128, S 128 to 4096) was faster
# on the one-launch route than on the route of :func:`variant` in both
# replicate runs of the CUDA-graph timing, beyond the identical-kernel
# controls' noise; inside the pieces with a lower bound every measured call
# was faster than that route and than cuDNN's fastest measured plan in every
# counted round of both replicate runs, in CUDA-graph replays and in eager
# calls.  Every measured call that was not lies outside every piece (a call
# measured in both runs by the later one; some calls that were lie outside
# too: a piece holds no measured call that was not).  The pieces may
# overlap, and their order does not change the route.
FUSED_CAUSAL_REGION: dict[tuple[str, int], tuple[tuple[int, int, int, int], ...] | None] = {
    ("sm_100a", 128): ((0, 128, 0, 4096), (0, 512, 0, 256)),
    ("sm_100a", 64): ((0, 128, 0, 4096),),
    ("sm_103a", 128): ((0, 128, 0, 4096), (256, 256, 128, 256)),
    ("sm_103a", 64): ((0, 128, 0, 4096), (0, 256, 0, 256)),
    ("sm_107a", 128): ((0, 128, 0, 4096), (512, 1024, 256, 512), (256, 256, 4096, 4096)),
    ("sm_107a", 64): ((0, 128, 0, 4096), (512, 512, 256, 256)),
}


def fused_record(arch: str, B: int, H: int, Sq: int, Skv: int, D: int, causal: bool) -> str | None:
    """The ``bwd_fused`` record of a call on ``arch`` (one of ``ARCHES``), or
    None: the call then takes the route of :func:`variant`.  The record is
    ``FUSED_OF[(D, causal)]`` when the call is inside the region of its mask.
    Without the mask: the region ``FUSED_REGION[(arch, D)]`` is not None and
    the call is inside any of its pieces: with ``(max_ctas, s_max, skv_any,
    skv_per_sq)`` the bounds of a piece, ``B * H * (Sq // BLOCK + Skv //
    BLOCK) <= max_ctas`` (the CTAs of the one-launch grid), ``max(Sq, Skv) <=
    s_max`` and ``Skv <= skv_any`` or ``Skv <= skv_per_sq * Sq``.  With the
    mask: ``Sq == Skv``, the region ``FUSED_CAUSAL_REGION[(arch, D)]`` is not
    None and the call is inside any of its pieces: with ``(least_ctas,
    most_ctas, least_s, most_s)`` the bounds of a piece, ``least_ctas <= B * H
    * (Sq // BLOCK + Skv // BLOCK) <= most_ctas`` and ``least_s <= Sq <=
    most_s``.  Reads the shapes and the architecture only."""
    record = FUSED_OF.get((D, bool(causal)))
    if record is None:
        return None
    if causal:
        region = FUSED_CAUSAL_REGION.get((arch, D))
        if region is None or Sq != Skv:
            return None
        ctas = B * H * (Sq // BLOCK + Skv // BLOCK)
        for least_ctas, most_ctas, least_s, most_s in region:
            if least_ctas <= ctas <= most_ctas and least_s <= Sq <= most_s:
                return record
        return None
    region = FUSED_REGION.get((arch, D))
    if region is None:
        return None
    for max_ctas, s_max, skv_any, skv_per_sq in region:
        if B * H * (Sq // BLOCK + Skv // BLOCK) > max_ctas or max(Sq, Skv) > s_max:
            continue
        if Skv <= skv_any or Skv <= skv_per_sq * Sq:
            return record
    return None
