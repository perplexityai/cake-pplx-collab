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
record is keyed ``<stage>_n<nmax>`` (``dw_reduce`` alone for that stage) and carries
the host plan: the ``stage`` it serves, the architectures it is delivered for and the ``run`` entry.  The
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
as run-time scalars.  The ``fwd`` and ``bwd`` stages ship one program per value
bound ``nmax`` (2 to 9, :data:`NMAX`): program ``nmax`` keeps ``nmax`` values in
registers and serves every ``n <= nmax``; a call of ``n`` values runs the
program of the smallest bound at least ``n`` (:func:`nmax_for`, :func:`record`).
Every program serves every token count, so the set of libraries is fixed: no
shape compiles a new one.

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
# The value bounds of the ``fwd`` and ``bwd`` programs: program ``nmax`` serves ``1 <= n <= nmax``.
NMAX = (2, 3, 4, 5, 6, 7, 8, 9)

MODULES: dict[str, dict[str, Any]] = {
    "bwd_n2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_31b6129cc550ef82db2f",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12160,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_31b6129cc550ef82db2f",
        "stage": "bwd",
    },
    "bwd_n3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_d70ef24ea048f7e6a52a",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15744,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_d70ef24ea048f7e6a52a",
        "stage": "bwd",
    },
    "bwd_n4": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_8c57f9e0cf69473355f8",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 19328,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_8c57f9e0cf69473355f8",
        "stage": "bwd",
    },
    "bwd_n5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_a703c67e3a5c562472b6",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 23296,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_a703c67e3a5c562472b6",
        "stage": "bwd",
    },
    "bwd_n6": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_6209813f6f2f9260adc1",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 26880,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_6209813f6f2f9260adc1",
        "stage": "bwd",
    },
    "bwd_n7": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_a621fc4d9694beaa3114",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 30464,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_a621fc4d9694beaa3114",
        "stage": "bwd",
    },
    "bwd_n8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_839a85061c559c58a67c",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 34048,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_839a85061c559c58a67c",
        "stage": "bwd",
    },
    "bwd_n9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_02e4d1fdd0ece4985916",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 38016,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_02e4d1fdd0ece4985916",
        "stage": "bwd",
    },
    "fwd_n2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_69b0d3669ad1763980ad",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_69b0d3669ad1763980ad",
        "stage": "fwd",
    },
    "fwd_n3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_3c0cb6d1f9fe9ad455ef",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12544,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_3c0cb6d1f9fe9ad455ef",
        "stage": "fwd",
    },
    "fwd_n4": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_e8783a6a545412badc06",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16128,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_e8783a6a545412badc06",
        "stage": "fwd",
    },
    "fwd_n5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_5e01d41ec6de93183a63",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20096,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_5e01d41ec6de93183a63",
        "stage": "fwd",
    },
    "fwd_n6": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_e1377207638d0b08dbec",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 23680,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_e1377207638d0b08dbec",
        "stage": "fwd",
    },
    "fwd_n7": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_974529c3cba4307eece3",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 27520,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_974529c3cba4307eece3",
        "stage": "fwd",
    },
    "fwd_n8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_631827c004a7c72a4be2",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 31104,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_631827c004a7c72a4be2",
        "stage": "fwd",
    },
    "fwd_n9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_106c9acdde9f5ecbb8e8",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 35072,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_106c9acdde9f5ecbb8e8",
        "stage": "fwd",
    },
}

# CTAs per SM each ``fwd`` / ``bwd`` program holds at least: the launch bound (``min_blocks``) of its build.  Fewer
# value registers (a smaller ``nmax``) leave room for more CTAs per SM.
CTAS_PER_SM = {
    "fwd_n2": 5,
    "fwd_n3": 5,
    "fwd_n4": 5,
    "fwd_n5": 4,
    "fwd_n6": 4,
    "fwd_n7": 4,
    "fwd_n8": 4,
    "fwd_n9": 3,
    "bwd_n2": 4,
    "bwd_n3": 4,
    "bwd_n4": 4,
    "bwd_n5": 4,
    "bwd_n6": 3,
    "bwd_n7": 2,
    "bwd_n8": 2,
    "bwd_n9": 2,
}
# Launch groups one device holds at once, by the CTAs per SM of a program: (architecture, SM count) -> {CTAs per SM:
# groups}, as ``cudaOccupancyMaxActiveClusters`` reports them for the shipped programs.  Empty: no device measured.
RESIDENT_GROUPS: dict[tuple[str, int], dict[int, int]] = {}
# On a device or a count of CTAs per SM the table does not hold, the grid takes this fraction of CTAs per SM * SMs /
# cluster CTAs groups.  Any grid is correct; a grid past the resident groups is slow, a smaller one leaves SMs idle.
UNMEASURED_FRACTION = 0.85


def nmax_for(n: int) -> int:
    """The value bound of the program a call of ``n`` values runs: the smallest of :data:`NMAX` at least ``n``."""
    if not 1 <= n <= NMAX[-1]:
        raise ValueError(f"n must be 1 to {NMAX[-1]}, got {n}")
    return next(k for k in NMAX if k >= n)


def record(stage: str, n: int = NMAX[-1]) -> str:
    """The ``MODULES`` record of ``stage`` a call of ``n`` values launches: ``<stage>_n<nmax_for(n)>`` (the stage
    alone for ``dw_reduce``, whatever ``n``)."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {list(STAGES)}")
    if stage == "dw_reduce":
        return stage
    return f"{stage}_n{nmax_for(n)}"


def cluster_ctas(record_name: str) -> int:
    """The CTAs of one launch group (cluster) of record ``record_name``."""
    return int(MODULES[record_name]["launch"]["cluster"][0])


def resident_groups(record_name: str, arch: str, sm_count: int) -> int:
    """The launch groups of the ``fwd`` or ``bwd`` record ``record_name`` one device of ``arch`` with ``sm_count`` SMs
    holds at once: the :data:`RESIDENT_GROUPS` entry of its ``CTAS_PER_SM``, else :data:`UNMEASURED_FRACTION` of
    ``CTAS_PER_SM * sm_count / cluster_ctas``; at least one."""
    per_sm = CTAS_PER_SM[record_name]
    measured = RESIDENT_GROUPS.get((arch, sm_count), {}).get(per_sm)
    if measured is not None:
        return measured
    return max(1, int(UNMEASURED_FRACTION * per_sm * sm_count) // cluster_ctas(record_name))


def groups(record_name: str, tokens: int, arch: str, sm_count: int) -> int:
    """The launch groups of the persistent grid of the ``fwd`` or ``bwd`` record ``record_name``: the groups the
    device holds at once (:func:`resident_groups`), never more groups than tokens, at least one."""
    return max(1, min(tokens, resident_groups(record_name, arch, sm_count)))


def grid(record_name: str, tokens: int, arch: str, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of ``record_name`` for ``tokens >= 1`` tokens on a device of ``arch`` with ``sm_count`` SMs:
    ``(cluster_ctas * groups, 1, 1)``."""
    return (cluster_ctas(record_name) * groups(record_name, tokens, arch, sm_count), 1, 1)
