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
builds of the stage, and carries the host plan: the ``stage`` it serves, the
architectures it is delivered for and the ``run`` entry.  The derived leaves
(``library``, ``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding; ``library`` is the program identity
(``cake_fmha_train_<20 hex>``), so a kernel swap renames it.  ``MODULES`` is a
literal table (the exporter and the manifest checker read it with
``ast.literal_eval``).

The stages, in launch order: ``bwd_pre`` (``dPsum = rowsum(O * dO)``,
``LSE * log2(e)`` and the cleared FP32 dQ accumulator), ``bwd_main`` (per key
block: S, P, dP and dS against every query block, dK and dV stored once, the
dQ tiles added into the FP32 accumulator by bulk reductions) and ``bwd_post``
(``dQ = bf16(acc * softmax_scale)`` in the layout of Q).  The dQ tiles of the
key blocks of one head reach the accumulator in scheduling order, so dQ is
not bitwise reproducible run to run; dK, dV and the preprocess outputs have
one fixed order.  Every launch is a plain stream-ordered launch: nothing
cooperative, no PDL.  Every kernel runs on the grid ``(S / 128, H, B)``: S is
the query length for ``bwd_pre`` and ``bwd_post`` and the key length for
``bwd_main``.
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

# The stages of one backward call, in launch order.
STAGES = ("bwd_pre", "bwd_main", "bwd_post")

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
}

# (head dim, causal) -> the records of the three stages, in launch order.  The
# preprocess and the postprocess do not depend on the mask.
RECORD_OF: dict[tuple[int, bool], tuple[str, str, str]] = {
    (128, False): ("bwd_pre", "bwd_main", "bwd_post"),
    (128, True): ("bwd_pre", "bwd_main_causal", "bwd_post"),
    (64, False): ("bwd_pre_hd64", "bwd_main_hd64", "bwd_post_hd64"),
    (64, True): ("bwd_pre_hd64", "bwd_main_hd64_causal", "bwd_post_hd64"),
}


def grid(stage: str, batch: int, heads: int, seqlen_q: int, seqlen_kv: int) -> tuple[int, int, int]:
    """The launch grid of ``stage``: one CTA per 128-row block of one (head,
    batch); query blocks for ``bwd_pre`` and ``bwd_post``, key blocks for
    ``bwd_main``.  The kernels read the head from ``blockIdx.y`` and the batch
    from ``blockIdx.z``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {list(STAGES)}")
    rows = seqlen_kv if stage == "bwd_main" else seqlen_q
    return (rows // BLOCK, heads, batch)
