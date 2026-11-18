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
width 5120 (SM100 / SM103 / SM107), deterministic package.  The loader
and the launch plan are :mod:`.cake_backend`.

Every record of ``MODULES`` is one generated program: a CUDA device
translation unit plus a tvm-ffi binding, compiled once per architecture.  A
record is keyed by ``<stage>_<acc>`` for the two stages that sum the dot
product (``fwd`` and ``bwd``) and by the stage name for ``dw_reduce``, and
carries the host plan: the ``stage`` it serves, the architectures it is
delivered for and the ``run`` entry.  The derived leaves (``library``,
``kernel_symbol``, ``launch``, ``arg_plan``) are written by the code
generator from the rendered binding; ``library`` is the program identity
(``cake_engram_gate_det_train_<20 hex>``), so a kernel swap renames it.
``MODULES`` is a literal table (the generator and the manifest checker read
it with ``ast.literal_eval``).

The stages, in launch order: ``fwd`` (per token and stream, the RMS of the
stream and of its key, the weighted dot product, the signed square-root gate
and ``out = bf16(x + gate * value)``; nothing is saved for the backward),
``bwd`` (recomputes the forward sums from the inputs; writes dX, dK, dV
and one FP32 row of the weight-gradient partial sums per cluster,
``[clusters, 4, 5120]``, every row written in full) and ``dw_reduce`` (sums the
partial rows in one fixed order and writes ``dq_weight = dw * k_weight`` and
``dk_weight = dw * q_weight`` in FP32).  Every sum has one fixed order, so every
output is bitwise reproducible run to run on one device: the cluster count
follows the token count, the record and the device only.

The ``fwd`` and ``bwd`` kernels give each token to one cluster of four CTAs
(each CTA owns 1280 of the 5120 columns of every stream) and walk the tokens
persistently: cluster ``c`` of ``n`` serves the tokens ``c, c + n, ...``.
The grid is ``(4 * n, 1, 1)`` with ``n = min(T, resident clusters)``, the
clusters one device holds at once (:func:`resident_clusters`: measured per
architecture, SM count and CTAs per SM); any grid is correct (the clusters
never wait on each other).  The two programs of a stage differ only in how the
dot product ``P = sum_d w_d h_d k_d`` is summed: ``dot2`` as a compensated
FP32 dot product (each product split with an FMA, each running sum with an
error-free two-sum, the error words summed in FP32 and added once at the end;
the default), ``f32`` in one FP32 word (opt-in).  Every other sum is one FP32
word.  Nothing launches cooperatively.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
# The fixed operand geometry: four residual streams of width 5120.
STREAMS = 4
HIDDEN = 5120
PACKAGE = "cake_pplx._backends.cake_engram_gate_det_train"
LIBRARY_PREFIX = "cake_engram_gate_det_train"
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
        "kernel_symbol": "kernel_cake_engram_gate_det_train_00000000000000000011",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_det_train_00000000000000000011",
        "stage": "bwd",
    },
    "bwd_f32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_det_train_00000000000000000012",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_det_train_00000000000000000012",
        "stage": "bwd",
    },
    "dw_reduce": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_det_train_00000000000000000013",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_det_train_00000000000000000013",
        "stage": "dw_reduce",
    },
    "fwd_dot2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_det_train_00000000000000000014",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_det_train_00000000000000000014",
        "stage": "fwd",
    },
    "fwd_f32": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_engram_gate_det_train_00000000000000000015",
        "launch": {
            "block": (1, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_engram_gate_det_train_00000000000000000015",
        "stage": "fwd",
    },
}

# CTAs of one cluster; a token's columns are split over them.
CLUSTER_CTAS = 4
# CTAs per SM each ``fwd`` / ``bwd`` program holds at once: the launch bound of
# its build, which its register count reaches on every architecture (its
# threads and shared memory would allow more).
CTAS_PER_SM = {"fwd_f32": 4, "fwd_dot2": 4, "bwd_f32": 3, "bwd_dot2": 3}
# Whole four-CTA clusters one device holds at once for a count of CTAs per SM,
# as ``cudaOccupancyMaxActiveClusters`` reports it for this family's programs
# (spread cluster scheduling, as they launch): (architecture, SM count) ->
# {CTAs per SM: clusters}.  The four CTAs of a cluster share one GPC, so a device holds
# fewer clusters than CTAs per SM * SMs / 4: at four CTAs per SM, B200 (148
# SMs) holds 142, GB300 (152 SMs) 148 and VR200 (212 SMs) 184.
RESIDENT_CLUSTERS: dict[tuple[str, int], dict[int, int]] = {
    ("sm_100a", 148): {4: 142, 6: 213},
    ("sm_103a", 152): {2: 74, 3: 110, 4: 148},
    ("sm_107a", 212): {2: 92, 3: 138, 4: 184, 6: 276},
}
# On a device or a count of CTAs per SM the table does not hold, the grid takes
# this fraction of CTAs per SM * SMs / 4 clusters: less than every measured
# device holds (the least, VR200, holds 0.868 of it).  Any grid is correct;
# a grid past the resident clusters is slow (a cluster that is not resident
# starts its tokens only after a resident one has walked all of its own, about
# twice the time at a large token count), a smaller one leaves CTA slots idle.
UNMEASURED_CLUSTER_FRACTION = 0.85
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


def resident_clusters(record_name: str, arch: str, sm_count: int) -> int:
    """The four-CTA clusters of the ``fwd`` or ``bwd`` record ``record_name``
    one device of ``arch`` with ``sm_count`` SMs holds at once: the
    :data:`RESIDENT_CLUSTERS` entry of its ``CTAS_PER_SM``, else
    :data:`UNMEASURED_CLUSTER_FRACTION` of ``CTAS_PER_SM * sm_count / 4``; at
    least one."""
    per_sm = CTAS_PER_SM[record_name]
    measured = RESIDENT_CLUSTERS.get((arch, sm_count), {}).get(per_sm)
    if measured is not None:
        return measured
    return max(1, int(UNMEASURED_CLUSTER_FRACTION * per_sm * sm_count) // CLUSTER_CTAS)


def clusters(record_name: str, tokens: int, arch: str, sm_count: int) -> int:
    """The cluster count of the persistent grid of the ``fwd`` or ``bwd``
    record ``record_name``: the clusters the device holds at once
    (:func:`resident_clusters`), never more clusters than tokens, at least
    one."""
    return max(1, min(tokens, resident_clusters(record_name, arch, sm_count)))


def dw_reduce_grid() -> tuple[int, int, int]:
    """The ``dw_reduce`` grid: one CTA per ``DW_REDUCE_COLUMNS_PER_CTA``
    columns of the ``[4, 5120]`` gradient, whatever the token count and the
    device."""
    return (STREAMS * HIDDEN // DW_REDUCE_COLUMNS_PER_CTA, 1, 1)


def grid(record_name: str, tokens: int, arch: str, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of ``record_name`` for ``tokens >= 1`` tokens on a
    device of ``arch`` with ``sm_count`` SMs: ``(4 * clusters, 1, 1)`` for
    ``fwd`` and ``bwd``, :func:`dw_reduce_grid` for ``dw_reduce``."""
    if MODULES[record_name]["stage"] == "dw_reduce":
        return dw_reduce_grid()
    return (CLUSTER_CTAS * clusters(record_name, tokens, arch, sm_count), 1, 1)
