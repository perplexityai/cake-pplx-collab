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

Registry of the Cake attention-residual (AttnRes) training kernels: a
softmax-weighted sum of 1 to 9 BF16 values of width 7168 (SM100 / SM103 /
SM107), non-deterministic package.  The loader and the launch plan are
:mod:`.cake_backend`.

Every record of ``MODULES`` is one generated program: a CUDA device
translation unit plus a tvm-ffi binding, compiled once per architecture.  A
record is keyed by its stage name and carries the host plan: the ``stage`` it
serves, the architectures it is delivered for and the ``run`` entry.  The
derived leaves (``library``, ``kernel_symbol``, ``launch``, ``arg_plan``) are
written by the code generator from the rendered binding; ``library`` is the
program identity (``cake_attn_res_train_<20 hex>``), so a kernel swap renames
it.  ``MODULES`` is a literal table (the generator and the manifest checker
read it with ``ast.literal_eval``).

The stages, in launch order: ``fwd`` (per token, the FP32 norms ``r_j``, the
dot products ``d_j`` with ``w``, the softmax ``p`` of ``r_j d_j`` over the
values and ``out = bf16(sum_j p_j v_j)``; it writes ``p``, ``r`` and ``d``,
FP32 ``[T, n]``, the state the backward reads) and ``bwd`` (from that state,
``dout`` and the values: every ``dv_j``, and the FP32 weight gradient ``dw``
added into a zeroed ``[7168]`` row by ``red.global.add.f32``, so its bits may
change run to run).  The kernels take the values as nine pointers (the host
passes the first value for the slots past ``n``) and ``n`` and the token count
as run-time scalars: one program serves every ``n`` from 1 to 9 and every token
count.

Both kernels walk the tokens persistently: group ``g`` of ``G`` launch groups
(one cluster of :func:`cluster_ctas` CTAs) serves the tokens ``g, g + G,
...``.  The grid is ``(cluster_ctas * G, 1, 1)`` with ``G = min(T, resident
groups)``, the groups one device holds at once (:func:`resident_groups`); any
grid is correct (the groups never wait on each other).  Nothing launches
cooperatively.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
# The fixed operand geometry: values of width 7168, at most nine of them (the kernels' value slots).
HIDDEN = 7168
MAX_VALUES = 9
PACKAGE = "cake_pplx._backends.cake_attn_res_nondet_train"
LIBRARY_PREFIX = "cake_attn_res_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order (``fwd`` in the forward, ``bwd`` in the backward).
STAGES = ("fwd", "bwd")

MODULES: dict[str, dict[str, Any]] = {
    "bwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (0, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "",
        "stage": "bwd",
    },
    "fwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "",
        "launch": {
            "block": (0, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "",
        "stage": "fwd",
    },
}

# CTAs per SM each record holds at once: the launch bound of its build.
CTAS_PER_SM = {"fwd": 2, "bwd": 2}
# Launch groups one device holds at once, by the CTAs per SM of a record: (architecture, SM count) -> {CTAs per SM:
# groups}, as ``cudaOccupancyMaxActiveClusters`` reports them for the shipped programs.  Empty: no device measured.
RESIDENT_GROUPS: dict[tuple[str, int], dict[int, int]] = {}
# On a device or a count of CTAs per SM the table does not hold, the grid takes this fraction of CTAs per SM * SMs /
# cluster CTAs groups.  Any grid is correct; a grid past the resident groups is slow, a smaller one leaves SMs idle.
UNMEASURED_FRACTION = 0.85


def record(stage: str) -> str:
    """The ``MODULES`` record of ``stage``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {list(STAGES)}")
    return stage


def cluster_ctas(record_name: str) -> int:
    """The CTAs of one launch group (cluster) of record ``record_name``."""
    return int(MODULES[record_name]["launch"]["cluster"][0])


def resident_groups(record_name: str, arch: str, sm_count: int) -> int:
    """The launch groups of record ``record_name`` one device of ``arch`` with ``sm_count`` SMs holds at once: the
    :data:`RESIDENT_GROUPS` entry of its ``CTAS_PER_SM``, else :data:`UNMEASURED_FRACTION` of ``CTAS_PER_SM *
    sm_count / cluster_ctas``; at least one."""
    per_sm = CTAS_PER_SM[record_name]
    measured = RESIDENT_GROUPS.get((arch, sm_count), {}).get(per_sm)
    if measured is not None:
        return measured
    return max(1, int(UNMEASURED_FRACTION * per_sm * sm_count) // cluster_ctas(record_name))


def groups(record_name: str, tokens: int, arch: str, sm_count: int) -> int:
    """The launch groups of the persistent grid of record ``record_name``: the groups the device holds at once
    (:func:`resident_groups`), never more groups than tokens, at least one."""
    return max(1, min(tokens, resident_groups(record_name, arch, sm_count)))


def grid(record_name: str, tokens: int, arch: str, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of ``record_name`` for ``tokens >= 1`` tokens on a device of ``arch`` with ``sm_count`` SMs:
    ``(cluster_ctas * groups, 1, 1)``."""
    return (cluster_ctas(record_name) * groups(record_name, tokens, arch, sm_count), 1, 1)
