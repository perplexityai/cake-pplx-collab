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
SM107), deterministic package.  The loader and the launch plan are
:mod:`.cake_backend`.

Every record of ``MODULES`` is one generated program: a CUDA device
translation unit plus a tvm-ffi binding, compiled once per architecture.  A
record is keyed ``<stage>_n<nmax>`` (``dw_reduce`` alone for that stage) and carries
the host plan: the ``stage`` it serves, the architectures it is delivered for and the ``run`` entry.  The
derived leaves (``library``, ``kernel_symbol``, ``launch``, ``arg_plan``) are
written by the code generator from the rendered binding; ``library`` is the
program identity (``cake_attn_res_det_train_<20 hex>``), so a kernel swap renames
it.  ``MODULES`` is a literal table (the generator and the manifest checker
read it with ``ast.literal_eval``).

The stages, in launch order: ``fwd`` (per token, the FP32 norms ``r_j``, the
dot products ``d_j`` with ``w``, the softmax ``p`` of ``r_j d_j`` over the
values and ``out = bf16(sum_j p_j v_j)``; it writes ``p``, ``r`` and ``d``,
FP32 ``[T, n]``, the state the backward reads), ``bwd`` (from that state,
``dout`` and the values: every ``dv_j``, and one FP32 row of ``dw`` partial
sums per launch group into ``dw_rows [G, 7168]``, every row written in full)
and ``dw_reduce`` (the ``G`` rows summed in one fixed order into ``dw``).
Every sum has one fixed order.  The kernels take the values as nine pointers (the host
passes the first value for the slots past ``n``) and ``n`` and the token count
as run-time scalars.  The ``fwd`` and ``bwd`` stages ship one program per value
bound ``nmax`` (2 to 9, :data:`NMAX`): program ``nmax`` keeps ``nmax`` values in
registers and serves every ``n <= nmax``; a call of ``n`` values runs the
program of the smallest bound at least ``n`` (:func:`nmax_for`, :func:`record`).
Every program serves every token count, so the set of libraries is fixed: no
shape compiles a new one.

Both kernels walk the tokens persistently over ``G`` launch groups (one
cluster of :func:`cluster_ctas` CTAs each).  The ``fwd`` programs of 5 to 9
values and every ``bwd`` program split each token's columns over the CTAs of
a cluster: group ``g`` serves the tokens ``g, g + G, ...``.  The ``fwd``
programs of 2 to 4 values (``fwd_n2``, ``fwd_n3``, ``fwd_n4``) serve whole
tokens per CTA and exchange nothing: CTA ``r`` of group ``g`` serves the
tokens ``4 g + r, 4 g + r + 4 G, ...``.  The grid is ``(cluster_ctas * G, 1,
1)`` with ``G = min(T, resident groups)``, the groups one device holds at once
(:func:`resident_groups`); any grid is correct (the groups never wait on each
other).  Nothing launches cooperatively.
"""

from __future__ import annotations

from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
# The fixed operand geometry: values of width 7168, at most nine of them (the kernels' value slots).
HIDDEN = 7168
MAX_VALUES = 9
PACKAGE = "cake_pplx._backends.cake_attn_res_det_train"
LIBRARY_PREFIX = "cake_attn_res_det_train"
MANIFEST = "build.json"

# The stages of one training step, in launch order (``fwd`` in the forward, ``bwd`` then ``dw_reduce`` in the backward).
STAGES = ("fwd", "bwd", "dw_reduce")
# The value bounds of the ``fwd`` and ``bwd`` programs: program ``nmax`` serves ``1 <= n <= nmax``.
NMAX = (2, 3, 4, 5, 6, 7, 8, 9)

MODULES: dict[str, dict[str, Any]] = {
    "bwd_n2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_c94faaec10a30a5c18a5",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12160,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_c94faaec10a30a5c18a5",
        "stage": "bwd",
    },
    "bwd_n3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_15e69d4ac22203976212",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 15744,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_15e69d4ac22203976212",
        "stage": "bwd",
    },
    "bwd_n4": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_2959d7370da0e30f3ddf",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 19328,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_2959d7370da0e30f3ddf",
        "stage": "bwd",
    },
    "bwd_n5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_582051697bc9f2df6237",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 23296,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_582051697bc9f2df6237",
        "stage": "bwd",
    },
    "bwd_n6": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_001ec9ef8840ffd7a1cd",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 26880,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_001ec9ef8840ffd7a1cd",
        "stage": "bwd",
    },
    "bwd_n7": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_794c668a35c657982760",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 30464,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_794c668a35c657982760",
        "stage": "bwd",
    },
    "bwd_n8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_1be557182f9d49351f50",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 34048,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_1be557182f9d49351f50",
        "stage": "bwd",
    },
    "bwd_n9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dout"], ["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["buffer", "dv0"], ["buffer", "dv1"], ["buffer", "dv2"], ["buffer", "dv3"], ["buffer", "dv4"], ["buffer", "dv5"], ["buffer", "dv6"], ["buffer", "dv7"], ["buffer", "dv8"], ["buffer", "dw_rows"], ["parameter", "T"], ["parameter", "n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_bfc3816087be048f2ebe",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 38016,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_bfc3816087be048f2ebe",
        "stage": "bwd",
    },
    "dw_reduce": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "dw_rows"], ["buffer", "dw"], ["parameter", "G"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_f2e8cf286e7fa021968d",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 4096,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_f2e8cf286e7fa021968d",
        "stage": "dw_reduce",
    },
    "fwd_n2": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_69b0d3669ad1763980ad",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 8576,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_69b0d3669ad1763980ad",
        "stage": "fwd",
    },
    "fwd_n3": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_3c0cb6d1f9fe9ad455ef",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 12544,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_3c0cb6d1f9fe9ad455ef",
        "stage": "fwd",
    },
    "fwd_n4": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_e8783a6a545412badc06",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 16128,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_e8783a6a545412badc06",
        "stage": "fwd",
    },
    "fwd_n5": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_5e01d41ec6de93183a63",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 20096,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_5e01d41ec6de93183a63",
        "stage": "fwd",
    },
    "fwd_n6": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_e1377207638d0b08dbec",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 23680,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_e1377207638d0b08dbec",
        "stage": "fwd",
    },
    "fwd_n7": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_974529c3cba4307eece3",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 27520,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_974529c3cba4307eece3",
        "stage": "fwd",
    },
    "fwd_n8": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_631827c004a7c72a4be2",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 31104,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_631827c004a7c72a4be2",
        "stage": "fwd",
    },
    "fwd_n9": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [["buffer", "v0"], ["buffer", "v1"], ["buffer", "v2"], ["buffer", "v3"], ["buffer", "v4"], ["buffer", "v5"], ["buffer", "v6"], ["buffer", "v7"], ["buffer", "v8"], ["buffer", "w"], ["buffer", "out"], ["buffer", "p"], ["buffer", "r"], ["buffer", "d"], ["parameter", "T"], ["parameter", "n"], ["parameter", "eps"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_attn_res_det_train_106c9acdde9f5ecbb8e8",
        "launch": {
            "block": (224, 1, 1),
            "cluster": (4, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 35072,
            "use_pdl": False,
        },
        "library": "cake_attn_res_det_train_106c9acdde9f5ecbb8e8",
        "stage": "fwd",
    },
}

# CTAs per SM each ``fwd`` / ``bwd`` program holds at least: the launch bound (``min_blocks``) of its build.  Fewer
# value registers (a smaller ``nmax``) leave room for more CTAs per SM; the whole-token ``fwd`` programs of 2 to 4
# values hold every value's 32 columns per thread in registers, two CTAs per SM.
CTAS_PER_SM = {
    "fwd_n2": 2,
    "fwd_n3": 2,
    "fwd_n4": 2,
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
# Columns of the [7168] weight gradient one CTA of ``dw_reduce`` sums.
DW_REDUCE_COLUMNS_PER_CTA = 128
# Launch groups (four-CTA clusters) one device holds at once, by the CTAs per SM of a program: (architecture, SM
# count) -> {CTAs per SM: clusters}, as ``cudaOccupancyMaxActiveClusters`` reports them (spread cluster scheduling, as
# the programs launch).  The count follows the CTAs per SM, not the program; every entry of GB300, B200, VR200 and
# R200 was measured on the shipped programs (all of them on GB300).  The four CTAs of a cluster share one GPC, so a
# device holds fewer clusters than CTAs per SM * SMs / 4, and on VR200 and R200 no more at 5 CTAs per SM than at 4.
# The GB200 entries at 3 and 4 CTAs per SM are the counts measured on the four-CTA cluster programs of the
# engram_gate family, which give the same GB300 and B200 counts at 3 and 4.  The GPU tests check every program's
# entry, and the grid the host takes, against the driver's count (``cuOccupancyMaxActiveClusters``) on the device
# they run on.
RESIDENT_GROUPS: dict[tuple[str, int], dict[int, int]] = {
    ("sm_100a", 148): {2: 71, 3: 104, 4: 142, 5: 175},
    ("sm_100a", 152): {3: 110, 4: 148},
    ("sm_103a", 152): {2: 74, 3: 110, 4: 148, 5: 184},
    ("sm_107a", 208): {2: 104, 3: 153, 4: 208, 5: 208},
    ("sm_107a", 212): {2: 92, 3: 138, 4: 184, 5: 184},
}
# On a device or a count of CTAs per SM the table does not hold, the grid takes this fraction of min(CTAs per SM,
# UNMEASURED_MAX_CTAS_PER_SM) * SMs / cluster CTAs groups: less than every measured device holds (the least, VR200,
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
    """The launch groups of the ``fwd`` or ``bwd`` record ``record_name`` one device of ``arch`` with ``sm_count`` SMs
    holds at once: the :data:`RESIDENT_GROUPS` entry of its ``CTAS_PER_SM``, else :data:`UNMEASURED_FRACTION` of
    ``min(CTAS_PER_SM, UNMEASURED_MAX_CTAS_PER_SM) * sm_count / cluster_ctas``; at least one."""
    per_sm = CTAS_PER_SM[record_name]
    measured = RESIDENT_GROUPS.get((arch, sm_count), {}).get(per_sm)
    if measured is not None:
        return measured
    per_sm = min(per_sm, UNMEASURED_MAX_CTAS_PER_SM)
    return max(1, int(UNMEASURED_FRACTION * per_sm * sm_count) // cluster_ctas(record_name))


def groups(record_name: str, tokens: int, arch: str, sm_count: int) -> int:
    """The launch groups of the persistent grid of the ``fwd`` or ``bwd`` record ``record_name``: the groups the
    device holds at once (:func:`resident_groups`), never more groups than tokens, at least one."""
    return max(1, min(tokens, resident_groups(record_name, arch, sm_count)))


def dw_reduce_grid() -> tuple[int, int, int]:
    """The ``dw_reduce`` grid: one CTA per ``DW_REDUCE_COLUMNS_PER_CTA`` columns of ``dw``, whatever the token count
    and the device."""
    return (HIDDEN // DW_REDUCE_COLUMNS_PER_CTA, 1, 1)


def grid(record_name: str, tokens: int, arch: str, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of ``record_name`` for ``tokens >= 1`` tokens on a device of ``arch`` with ``sm_count`` SMs:
    ``(cluster_ctas * groups, 1, 1)`` for ``fwd`` and ``bwd``, :func:`dw_reduce_grid` for ``dw_reduce``."""
    if MODULES[record_name]["stage"] == "dw_reduce":
        return dw_reduce_grid()
    return (cluster_ctas(record_name) * groups(record_name, tokens, arch, sm_count), 1, 1)
