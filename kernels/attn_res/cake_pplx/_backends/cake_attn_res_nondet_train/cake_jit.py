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
``dout`` and the values: every ``dv_j``, and the FP32 weight gradient ``dw``:
each CTA keeps its tokens' ``dw`` sums in registers and adds them into a
zeroed ``[7168]`` row once, after its token loop, with ``red.global.add.v4.f32``,
so its bits may change run to run).  The kernels take the values as nine pointers (the host
passes the first value for the slots past ``n``) and ``n`` and the token count
as run-time scalars.  The ``fwd`` and ``bwd`` stages ship one program per value
bound ``nmax`` (2 to 9, :data:`NMAX`): program ``nmax`` keeps ``nmax`` values in
registers and serves every ``n <= nmax``; a call of ``n`` values runs the
program of the smallest bound at least ``n`` (:func:`nmax_for`, :func:`record`).
Every program serves every token count, so the set of libraries is fixed: no
shape compiles a new one.

Both kernels walk the tokens persistently over ``G`` launch groups (one
cluster of :func:`cluster_ctas` CTAs each).  Every ``bwd`` program splits
each token's columns over the four CTAs of a cluster: group ``g`` serves the
tokens ``g, g + G, ...``.  The ``fwd`` programs serve whole tokens per CTA
and exchange nothing.  Those of 2 to 4 values (``fwd_n2``, ``fwd_n3``,
``fwd_n4``, 224 threads) launch in clusters of four CTAs: CTA ``r`` of group
``g`` serves the tokens ``4 g + r, 4 g + r + 4 G, ...``.  Those of 5 to 9
values (``fwd_n5`` .. ``fwd_n9``) launch one 448-thread CTA per group (no
cluster), one CTA per SM: CTA ``g`` serves the tokens ``g, g + G, ...``, all
7168 columns of each.  The grid is ``(cluster_ctas * G, 1, 1)`` with
``G = min(T, resident groups)``, the groups one device holds at once,
or a measured cap below them for a program that runs faster with fewer groups
(:func:`resident_groups`, :data:`GROUP_CAPS`); any grid is correct (the groups
never wait on each other).  Nothing launches cooperatively.
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw"], ["parameter", "T"], ["parameter", "n"], ["zero", "zero_region"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
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
        "kernel_symbol": "kernel_cake_attn_res_train_8f0abbf6633012d9811f",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 28928,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_8f0abbf6633012d9811f",
        "stage": "fwd",
    },
    "fwd_n3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_df3f4e0082df6f7bc3f5",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 43520,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_df3f4e0082df6f7bc3f5",
        "stage": "fwd",
    },
    "fwd_n4": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_cdd21e6bc5cb6ddca3a7",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 57856,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_cdd21e6bc5cb6ddca3a7",
        "stage": "fwd",
    },
    "fwd_n5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_8ba012f8bc640ba0f589",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 73088,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_8ba012f8bc640ba0f589",
        "stage": "fwd",
    },
    "fwd_n6": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_00f9afa45eeb51811a77",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 87424,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_00f9afa45eeb51811a77",
        "stage": "fwd",
    },
    "fwd_n7": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_2d871eebb92640788a8f",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 102144,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_2d871eebb92640788a8f",
        "stage": "fwd",
    },
    "fwd_n8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_30ddc8b41cd3893406f3",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 116480,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_30ddc8b41cd3893406f3",
        "stage": "fwd",
    },
    "fwd_n9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_train_c0423d043d674720a691",
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 131328,
            "use_pdl": False,
        },
        "library": "cake_attn_res_train_c0423d043d674720a691",
        "stage": "fwd",
    },
}

# CTAs per SM each ``fwd`` / ``bwd`` program holds at least: the launch bound (``min_blocks``) of its build.  Fewer
# value registers (a smaller ``nmax``) leave room for more CTAs per SM.  The whole-token ``fwd`` programs hold every
# value's columns of a thread in registers: those of 2 to 4 values (224 threads) 32 columns per thread, two CTAs per
# SM; those of 5 to 9 values (448 threads) 16 columns per thread, one CTA per SM.
CTAS_PER_SM = {
    "fwd_n2": 2,
    "fwd_n3": 2,
    "fwd_n4": 2,
    "fwd_n5": 1,
    "fwd_n6": 1,
    "fwd_n7": 1,
    "fwd_n8": 1,
    "fwd_n9": 1,
    "bwd_n2": 4,
    "bwd_n3": 4,
    "bwd_n4": 4,
    "bwd_n5": 4,
    "bwd_n6": 3,
    "bwd_n7": 2,
    "bwd_n8": 2,
    "bwd_n9": 2,
}
# Launch groups one device holds at once of the programs that launch in four-CTA clusters (every ``bwd`` program and
# ``fwd_n2`` .. ``fwd_n4``), by the CTAs per SM of a program: (architecture, SM count) -> {CTAs per SM: clusters}, as
# ``cudaOccupancyMaxActiveClusters`` reports them (spread cluster scheduling, as the programs launch).  The count
# follows the CTAs per SM, not the program; every entry of GB300, GB200, B200, VR200 and R200 was measured on the
# four-CTA programs of 0.1.1 (all of them on GB300 and GB200), and the four-CTA programs of this version are those
# programs unchanged.  Boards of one SM count can differ by a cluster (one R200 board of 208 SMs holds 154 at 3 CTAs per
# SM, another 153): an entry holds the least count measured on the boards of its SM count.  The four CTAs of a cluster
# share one GPC, so a device holds at most CTAs per SM * SMs / 4 clusters (R200 exactly that many at 2 and 4 CTAs per
# SM, the other devices fewer), and on VR200 and R200 no more at 5 CTAs per SM than at 4.  GB200 and GB300 (both 152
# SMs) hold the same counts at 2, 3 and 4 CTAs per SM.  A program whose group is one CTA (``fwd_n5`` .. ``fwd_n9``)
# needs no entry: a lone CTA shares no GPC with another, so every device holds CTAs per SM * SMs of its groups
# (:func:`resident_groups`).  The GPU tests check that every program's entry, and the grid the host takes, do not
# exceed the driver's count (``cuOccupancyMaxActiveClusters``, with a cluster of one CTA for ``fwd_n5`` .. ``fwd_n9``)
# on the device they run on.
RESIDENT_GROUPS: dict[tuple[str, int], dict[int, int]] = {
    ("sm_100a", 148): {2: 71, 3: 104, 4: 142, 5: 175},
    ("sm_100a", 152): {2: 74, 3: 110, 4: 148},
    ("sm_103a", 152): {2: 74, 3: 110, 4: 148, 5: 184},
    ("sm_107a", 208): {2: 104, 3: 153, 4: 208, 5: 208},
    ("sm_107a", 212): {2: 92, 3: 138, 4: 184, 5: 184},
}
# Launch groups a program takes at most, below the groups the device holds: (architecture, SM count) -> {record:
# groups}.  ``fwd_n4`` (the whole-token forward of four values, two CTAs per SM) runs up to 7 % faster at 64 groups
# than at all 74 resident groups on GB200 and GB300 and all 71 on B200 (CUDA graph replays, cold L2, T = 4097, 16380
# and 32768, two to four GPUs of each, both packages); 62 to 66 groups measure within 3 % of 64.  ``fwd_n2``,
# ``fwd_n3`` and ``bwd_n7`` .. ``bwd_n9`` (the other two-CTA programs) and the spot-checked ``bwd_n5`` show no
# consistent gain below their resident groups on these devices; ``bwd_n2`` .. ``bwd_n4`` and ``bwd_n6`` were not
# measured below them, nor were ``fwd_n5`` .. ``fwd_n9`` of this version (one CTA on every SM; the spot checks of
# 0.1.1 ran on the four-CTA forwards they replace).  No program but ``fwd_n4`` has a cap.  A device without an entry
# here takes the resident groups.
GROUP_CAPS: dict[tuple[str, int], dict[str, int]] = {
    ("sm_100a", 148): {"fwd_n4": 64},
    ("sm_100a", 152): {"fwd_n4": 64},
    ("sm_103a", 152): {"fwd_n4": 64},
}
# A four-CTA program on a device or a count of CTAs per SM the table does not hold takes this fraction of min(CTAs per
# SM, UNMEASURED_MAX_CTAS_PER_SM) * SMs / cluster CTAs groups: less than every measured device holds (the least, VR200,
# holds 0.868 of CTAs per SM * SMs / 4 at 2 to 4 CTAs per SM, and at 5 the count of 4).  Any grid is correct; a grid
# past the resident groups is slow (a group that is not resident starts its tokens only after a resident one has
# walked all of its own), a smaller one leaves CTA slots idle.
UNMEASURED_FRACTION = 0.85
UNMEASURED_MAX_CTAS_PER_SM = 4


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
    """The launch groups the persistent grid of the ``fwd`` or ``bwd`` record ``record_name`` fills on one device of
    ``arch`` with ``sm_count`` SMs: the groups the device holds at once, at most the record's :data:`GROUP_CAPS`
    entry of the device.  A record whose group is one CTA (``fwd_n5`` .. ``fwd_n9``) holds ``CTAS_PER_SM *
    sm_count`` on every device; a four-CTA record the :data:`RESIDENT_GROUPS` entry of its ``CTAS_PER_SM``, else
    :data:`UNMEASURED_FRACTION` of ``min(CTAS_PER_SM, UNMEASURED_MAX_CTAS_PER_SM) * sm_count / cluster_ctas``; at
    least one."""
    per_sm = CTAS_PER_SM[record_name]
    if cluster_ctas(record_name) == 1:
        held = per_sm * sm_count
    else:
        held = RESIDENT_GROUPS.get((arch, sm_count), {}).get(per_sm)
    if held is not None:
        return min(held, GROUP_CAPS.get((arch, sm_count), {}).get(record_name, held))
    per_sm = min(per_sm, UNMEASURED_MAX_CTAS_PER_SM)
    return max(1, int(UNMEASURED_FRACTION * per_sm * sm_count) // cluster_ctas(record_name))


def groups(record_name: str, tokens: int, arch: str, sm_count: int) -> int:
    """The launch groups of the persistent grid of the ``fwd`` or ``bwd`` record ``record_name``: the groups the
    device holds at once, or the record's cap (:func:`resident_groups`), never more groups than tokens, at least
    one."""
    return max(1, min(tokens, resident_groups(record_name, arch, sm_count)))


def grid(record_name: str, tokens: int, arch: str, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of ``record_name`` for ``tokens >= 1`` tokens on a device of ``arch`` with ``sm_count`` SMs:
    ``(cluster_ctas * groups, 1, 1)``."""
    return (cluster_ctas(record_name) * groups(record_name, tokens, arch, sm_count), 1, 1)
