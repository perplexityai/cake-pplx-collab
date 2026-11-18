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

Registry of the Cake Engram context gate training kernels: four streams of
width 5120 (SM100 / SM103 / SM107), non-deterministic package.  The loader
and the launch plan are :mod:`.cake_backend`.

Every record of ``MODULES`` is one generated program: a CUDA device
translation unit plus a tvm-ffi binding, compiled once per architecture.  A
record is keyed by ``<stage>_<acc>`` for the two stages that sum the dot
product (``fwd`` and ``bwd``) and by the stage name for ``dw_reduce``, and
carries the host plan: the ``stage`` it serves, the architectures it is
delivered for and the ``run`` entry.  The derived leaves (``library``,
``kernel_symbol``, ``launch``, ``arg_plan``) are written by the code
generator from the rendered binding; ``library`` is the program identity
(``cake_engram_gate_train_<20 hex>``), so a kernel swap renames it.
``MODULES`` is a literal table (the generator and the manifest checker read
it with ``ast.literal_eval``).

The stages, in launch order: ``fwd`` (per token and stream, the RMS of the
stream and of its key, the weighted dot product, the signed square-root gate
and ``out = bf16(x + gate * value)``; nothing is saved for the backward),
``bwd`` (recomputes the forward sums from the inputs; writes dX, dK, dV
and adds each cluster's FP32 weight-gradient partial sums into one host-zeroed
``[4, 5120]`` accumulator with float atomics) and ``dw_reduce`` (reads the
accumulator as one partial row and writes ``dq_weight = dw * k_weight`` and
``dk_weight = dw * q_weight`` in FP32).  The clusters reach the accumulator in
arrival order, so the weight gradients are not bitwise reproducible run to run;
out, dX, dK and dV have one fixed order.

The ``fwd`` and ``bwd`` kernels give each token to one cluster of four CTAs
(each CTA owns 1280 of the 5120 columns of every stream) and walk the tokens
persistently: cluster ``c`` of ``n`` serves the tokens ``c, c + n, ...``.
The grid is ``(4 * n, 1, 1)`` with ``n = min(T, max clusters)``; any grid is
correct (the clusters never wait on each other).  The two programs of a stage
differ only in how the dot product ``P = sum_d w_d h_d k_d`` is summed:
``f32`` in one FP32 word, ``dot2`` as a compensated FP32 dot product (each
product split with an FMA, each running sum with an error-free two-sum, the
error words summed in FP32 and added once at the end).  Every other sum is one
FP32 word.  Nothing launches cooperatively.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
# The fixed operand geometry: four residual streams of width 5120.
STREAMS = 4
HIDDEN = 5120
PACKAGE = "cake_pplx._backends.cake_engram_gate_nondet_train"
LIBRARY_PREFIX = "cake_engram_gate_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order (``fwd`` in the forward,
# ``bwd`` then ``dw_reduce`` in the backward).
STAGES = ("fwd", "bwd", "dw_reduce")
# The two summations of the dot product P; a record of ``fwd`` or ``bwd`` is
# named ``<stage>_<acc>``.
ACC_MODES = ("f32", "dot2")

MODULES: dict[str, dict[str, Any]] = {
    "bwd_dot2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_train_00000000000000000021",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_train_00000000000000000021",
        "stage": "bwd",
    },
    "bwd_f32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_train_00000000000000000022",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_train_00000000000000000022",
        "stage": "bwd",
    },
    "dw_reduce": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_train_00000000000000000023",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_train_00000000000000000023",
        "stage": "dw_reduce",
    },
    "fwd_dot2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_train_00000000000000000024",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_train_00000000000000000024",
        "stage": "fwd",
    },
    "fwd_f32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_train_00000000000000000025",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_train_00000000000000000025",
        "stage": "fwd",
    },
}

# CTAs of one cluster; a token's columns are split over them.
CLUSTER_CTAS = 4
# CTAs per SM the persistent grid fills, per stage.  The grid is sized from the
# SM count only (no device query and no host synchronization per call).
CTAS_PER_SM = {"fwd": 4, "bwd": 2}
# Columns of the flattened [4 * 5120] weight gradient one CTA of ``dw_reduce``
# sums (32 lanes of four columns; its eight warps split the partial rows).
DW_REDUCE_COLUMNS_PER_CTA = 128


def record(stage: str, acc: str) -> str:
    """The ``MODULES`` record of ``stage`` for the dot-product summation ``acc``
    (``dw_reduce`` has one record)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {list(STAGES)}")
    if stage == "dw_reduce":
        return stage
    if acc not in ACC_MODES:
        raise ValueError(f"unknown acc {acc!r}; the summations are {list(ACC_MODES)}")
    return f"{stage}_{acc}"


def clusters(stage: str, tokens: int, sm_count: int) -> int:
    """The cluster count of the persistent ``fwd`` / ``bwd`` grid: about
    ``CTAS_PER_SM[stage]`` CTAs per SM, at least one cluster and never more
    clusters than tokens."""
    return max(1, min(tokens, CTAS_PER_SM[stage] * sm_count // CLUSTER_CTAS))


def grid(record_name: str, tokens: int, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of ``record_name`` for ``tokens >= 1`` tokens: ``(4 *
    clusters, 1, 1)`` for ``fwd`` and ``bwd``; for ``dw_reduce`` one CTA per
    ``DW_REDUCE_COLUMNS_PER_CTA`` columns of the ``[4, 5120]`` gradient."""
    stage = MODULES[record_name]["stage"]
    if stage == "dw_reduce":
        return (STREAMS * HIDDEN // DW_REDUCE_COLUMNS_PER_CTA, 1, 1)
    return (CLUSTER_CTAS * clusters(stage, tokens, sm_count), 1, 1)
