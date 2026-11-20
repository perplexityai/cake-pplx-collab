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

Host plan and loader glue of the Cake BF16 RMSNorm training kernels
(SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES``, ``ROUTES`` and ``HIDDEN_SIZES`` are filled by
the exporter from the complete verified program bundle and carry the host plan
only: the kernel's ``library`` stem, the architectures it is delivered for, its
launch / selection / chunking rules and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through its JIT shim from the same sources and flags,
verifying every source digest before the first build.

A route is the ordered list of module records one public call launches:
``fwd`` / ``fwd_residual`` launch one kernel; ``bwd`` / ``bwd_residual``
launch a pair -- the main kernel, whose persistent CTAs claim row chunks from
a counter in the workspace and write one FP32 partial row of ``dw`` per chunk
(two, a high and a low part, for a record whose workspace rule spells
``partial_rows`` 2), then the dw reduce kernel under programmatic dependent
launch, which folds the partial rows into ``dw`` in row order.  No launch is cooperative, so a
backward never waits for the whole device.  Token counts are runtime
scalars, so a changed number of rows never recompiles anything.

A module record is one (kernel, launch policy) pair.  A kernel launched under
several policies -- a different persistent grid or row range per stage -- has
one record per policy (``<kernel>__<stage>`` after the first), all naming the
kernel's ``library``, so it is compiled once.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Sequence

import tvm_ffi

from . import cake_build

# The three kernel-parameter plans (``arg_plan``: kernel parameter order as (kind, name);
# ``logical``: kernel parameter name -> logical argument name of the public API).
_CLASSES: dict[str, dict[str, Any]] = {
    "fwd": {
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "u"],
            ["buffer", "w"],
            ["buffer", "y"],
            ["buffer", "h_new"],
            ["buffer", "r"],
            ["parameter", "T"],
            ["parameter", "eps"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "T": "rows",
            "eps": "eps",
            "h_new": "h_new",
            "r": "r",
            "u": "u",
            "w": "w",
            "x": "x",
            "y": "y",
        },
    },
    "bwd_main": {
        "arg_plan": [
            ["buffer", "g"],
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "r"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["buffer", "counters"],
            ["buffer", "g_h"],
            ["parameter", "T"],
            ["parameter", "n_chunks"],
            ["parameter", "rows_per_chunk"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "T": "rows",
            "counters": "counters",
            "dw": "dw",
            "dx": "dx",
            "g": "g",
            "g_h": "g_h",
            "n_chunks": "n_chunks",
            "partial": "partial",
            "r": "r",
            "rows_per_chunk": "slots",
            "w": "w",
            "x": "x",
        },
    },
    "bwd_reduce": {
        "arg_plan": [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "n_chunks"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "logical": {
            "dw": "dw",
            "n_chunks": "n_chunks",
            "partial": "partial",
        },
    },
}

# Filled mechanically from the complete verified program bundle.
MODULES: dict[str, dict[str, Any]] = {
    "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_caf6978ed0a17fbf65b9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_07f9f2fbea7163639c27": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_ed36a8d998abba41f227__0_171": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_ed36a8d998abba41f227",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ed36a8d998abba41f227",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ed36a8d998abba41f227__0_171 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_978741b43372951cb5b9__0_171": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_978741b43372951cb5b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_978741b43372951cb5b9",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_a1c71cc355c7e9988ab2": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_a1c71cc355c7e9988ab2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a1c71cc355c7e9988ab2",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a1c71cc355c7e9988ab2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ab6b18375f52dcec73f7": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_ab6b18375f52dcec73f7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ab6b18375f52dcec73f7",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_ed36a8d998abba41f227__1_2": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_ed36a8d998abba41f227",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ed36a8d998abba41f227",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ed36a8d998abba41f227__1_2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_978741b43372951cb5b9__1_2": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_978741b43372951cb5b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_978741b43372951cb5b9",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_ed36a8d998abba41f227__2_171": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_ed36a8d998abba41f227",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ed36a8d998abba41f227",
        "role": "bwd_col",
        "select_rule": {
            "above": 2,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ed36a8d998abba41f227__2_171 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_978741b43372951cb5b9__2_171": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_978741b43372951cb5b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_978741b43372951cb5b9",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 2,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__171_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "role": "bwd_small",
        "select_rule": {
            "above": 171,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__171_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_07f9f2fbea7163639c27__171_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 171,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_cc26e316ae40d683a118__171_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 8,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "role": "bwd_large",
        "select_rule": {
            "above": 171,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_cc26e316ae40d683a118__171_16384 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c83c314c6dcf28ad7369__171_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 8,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c83c314c6dcf28ad7369",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c83c314c6dcf28ad7369",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 171,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b13b0c73ece62938899a": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_b13b0c73ece62938899a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b13b0c73ece62938899a",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b13b0c73ece62938899a (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_d22b3e8ce96e49997448": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d22b3e8ce96e49997448",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d22b3e8ce96e49997448",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_923e668c53f60f0b256b": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_923e668c53f60f0b256b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_923e668c53f60f0b256b",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_923e668c53f60f0b256b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9030dbeb948761ec2258": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_9030dbeb948761ec2258",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9030dbeb948761ec2258",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_64e6aa82bb2c6d4ea978__0_288": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_64e6aa82bb2c6d4ea978",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_64e6aa82bb2c6d4ea978",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_64e6aa82bb2c6d4ea978__0_288 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_17493054fed5409b6df9__0_288": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_17493054fed5409b6df9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_17493054fed5409b6df9",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__288_768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "role": "bwd_small",
        "select_rule": {
            "above": 288,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__288_768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_07f9f2fbea7163639c27__288_768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 288,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_cc26e316ae40d683a118__768_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "role": "bwd_small",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 768 and 1024 rows: 07f9f2fb, the FP32-weight twin of caf6978e.
    "cake_rmsnorm_det_train_07f9f2fbea7163639c27__ov100_768_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__171_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 171,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__171_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 171,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__288_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 288,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__288_768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 288,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__768_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__2047_22528": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2047,
            "at_most": 22528,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03bdf5667404141e1eaf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_03bdf5667404141e1eaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03bdf5667404141e1eaf",
        "role": "fwd_mid_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_03bdf5667404141e1eaf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485",
        "role": "fwd_mid_w32_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9ac8d21e46a658e7913f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_9ac8d21e46a658e7913f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9ac8d21e46a658e7913f",
        "role": "bwd_mid",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9ac8d21e46a658e7913f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_d96a14fe8882bbd9ec2e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d96a14fe8882bbd9ec2e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d96a14fe8882bbd9ec2e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_880768a0a264e04dbfb5__3072_4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 32,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d",
        "role": "fwd_large",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0ad4d53c4a0df501a99d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_595f506004038a65cc63": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 32,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_595f506004038a65cc63",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_595f506004038a65cc63",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d__wide": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 64,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d",
        "role": "fwd_wide",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 12288 and 16384 rows: 40210950, the FP32-weight twin of bde56bd5.
    "cake_rmsnorm_det_train_40210950f2e7ee48bc3e__ov107_12288_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_40210950f2e7ee48bc3e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_40210950f2e7ee48bc3e",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__4096_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1ea88523077fc340e55f": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1ea88523077fc340e55f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_21d3416baa877808371e": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d52385477d517c951e87": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_d52385477d517c951e87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d52385477d517c951e87",
        "role": "fwd_small",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_d52385477d517c951e87 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8e49f86a65049578dee1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_8e49f86a65049578dee1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8e49f86a65049578dee1",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1ea88523077fc340e55f__256_1536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1ea88523077fc340e55f__256_1536 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_21d3416baa877808371e__256_1536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1ea88523077fc340e55f__0_1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1ea88523077fc340e55f__0_1 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_21d3416baa877808371e__0_1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1ea88523077fc340e55f__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1ea88523077fc340e55f__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_21d3416baa877808371e__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_288d7b1d51c340236cb7": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_288d7b1d51c340236cb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_288d7b1d51c340236cb7",
        "role": "fwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1024 and 6144 rows: 3d619cd1, the FP32-weight twin of cf1d0554.
    "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d__ov100_1024_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, above 6144 rows: 5f89e5fd, the FP32-weight twin of 288d7b1d.
    "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e__ov100_6144_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1024 and 6144 rows: 2763cd5e, the FP32-weight twin of bad4db91.
    "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6__ov103_1024_6144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 6144 rows: 5f89e5fd, the FP32-weight twin of 288d7b1d.
    "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e__ov103_6144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2a42e5c02b240f8709b1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_2a42e5c02b240f8709b1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2a42e5c02b240f8709b1",
        "role": "fwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2a42e5c02b240f8709b1 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_79fecc7016e44e940733": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_79fecc7016e44e940733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_79fecc7016e44e940733",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2fdbab2af48db42cd2a9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9",
        "role": "fwd_main",
        "select_rule": {
            "above": 1536,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2fdbab2af48db42cd2a9__low (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_308da18230017dadc02d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_308da18230017dadc02d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_308da18230017dadc02d",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_308da18230017dadc02d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8e3407e059cfc005f421": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_8e3407e059cfc005f421",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8e3407e059cfc005f421",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8b4730338cac98ec272d": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_8b4730338cac98ec272d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8b4730338cac98ec272d",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8b4730338cac98ec272d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3e55e44ce21eff7d8931": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_3e55e44ce21eff7d8931",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3e55e44ce21eff7d8931",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8b4730338cac98ec272d__0_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_8b4730338cac98ec272d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8b4730338cac98ec272d",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8b4730338cac98ec272d__0_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3e55e44ce21eff7d8931__0_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_3e55e44ce21eff7d8931",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3e55e44ce21eff7d8931",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_0dcdc563be377247044e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": None,
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0dcdc563be377247044e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_56f988dde54db37f09d9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": None,
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_864e84f3d874f850c091__0_1152": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_864e84f3d874f850c091",
        "role": "bwd_col_residual",
        "select_rule": {
            "above": 0,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_864e84f3d874f850c091__0_1152 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_71d36617003ab7742a77__0_1152": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_71d36617003ab7742a77",
        "role": "bwd_col_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_864e84f3d874f850c091__0_1216": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_864e84f3d874f850c091",
        "role": "bwd_col_residual",
        "select_rule": {
            "above": 0,
            "at_most": 1216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_864e84f3d874f850c091__0_1216 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_71d36617003ab7742a77__0_1216": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_71d36617003ab7742a77",
        "role": "bwd_col_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 1216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_0dcdc563be377247044e__32_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 32,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0dcdc563be377247044e__32_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_56f988dde54db37f09d9__32_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 32,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_0dcdc563be377247044e__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": None,
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0dcdc563be377247044e__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_56f988dde54db37f09d9__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": None,
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_ee1be07a89a0482de45e__0_368": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 48,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_ee1be07a89a0482de45e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ee1be07a89a0482de45e",
        "role": "bwd_col_residual",
        "select_rule": {
            "above": 0,
            "at_most": 368,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ee1be07a89a0482de45e__0_368 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3bbd4f848b4847bff893__0_368": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 48,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_3bbd4f848b4847bff893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3bbd4f848b4847bff893",
        "role": "bwd_col_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 368,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_0dcdc563be377247044e__960_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 960,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0dcdc563be377247044e__960_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_56f988dde54db37f09d9__960_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 960,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c1a3be946e94e94233af__1152_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_c1a3be946e94e94233af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c1a3be946e94e94233af",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1152,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c1a3be946e94e94233af__1152_42496 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ab0e20feefa296ffff3c__1152_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_ab0e20feefa296ffff3c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ab0e20feefa296ffff3c",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1152,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c1a3be946e94e94233af__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_c1a3be946e94e94233af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c1a3be946e94e94233af",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c1a3be946e94e94233af__42496_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ab0e20feefa296ffff3c__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_ab0e20feefa296ffff3c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ab0e20feefa296ffff3c",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # cake_rmsnorm_det_train_84385468c8e8ef19f3dc over the window of cake_rmsnorm_det_train_c1a3be946e94e94233af__42496_inf: the sm_100a route bwd_residual_h6144 runs it.
    "cake_rmsnorm_det_train_84385468c8e8ef19f3dc__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_84385468c8e8ef19f3dc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_84385468c8e8ef19f3dc",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_84385468c8e8ef19f3dc__42496_inf (the effective weight read at FP32; every other operand and the host plan the owner's); no route selects this record: the FP32-weight route of sm_100a runs the twin of cake_rmsnorm_det_train_c1a3be946e94e94233af there.
    "cake_rmsnorm_det_train_486424b132fa3256e8ef__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_486424b132fa3256e8ef",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_486424b132fa3256e8ef",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_04ebec87cbee3f0ce093__1216_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_04ebec87cbee3f0ce093",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_04ebec87cbee3f0ce093",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1216,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_04ebec87cbee3f0ce093__1216_48640 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_47890c9ae28e3e4b6759__1216_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_47890c9ae28e3e4b6759",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_47890c9ae28e3e4b6759",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1216,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_04ebec87cbee3f0ce093__48640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_04ebec87cbee3f0ce093",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_04ebec87cbee3f0ce093",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 48640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_04ebec87cbee3f0ce093__48640_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_47890c9ae28e3e4b6759__48640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_47890c9ae28e3e4b6759",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_47890c9ae28e3e4b6759",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 48640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_2e8d2128a33361628509": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2e8d2128a33361628509__32_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2e8d2128a33361628509",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2e8d2128a33361628509",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_60bfcb24560bbae612c3": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_60bfcb24560bbae612c3",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_60bfcb24560bbae612c3",
        "role": "bwd_reduce",
        "select_rule": None,
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_60bfcb24560bbae612c3__960_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_60bfcb24560bbae612c3",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_60bfcb24560bbae612c3",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 960,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3430e5d9c053309b0323": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3430e5d9c053309b0323",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3430e5d9c053309b0323",
        "role": "fwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3430e5d9c053309b0323 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3430e5d9c053309b0323__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3430e5d9c053309b0323",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3430e5d9c053309b0323",
        "role": "fwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3430e5d9c053309b0323__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3feed5026e1156d2fba2": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3feed5026e1156d2fba2",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3feed5026e1156d2fba2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b143f616f21eeeaa021d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_b143f616f21eeeaa021d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b143f616f21eeeaa021d",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b143f616f21eeeaa021d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_740c03baae3da1ceb102": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_740c03baae3da1ceb102",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_740c03baae3da1ceb102",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_49af1e75185ddd637b76__0_768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_49af1e75185ddd637b76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_49af1e75185ddd637b76",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_49af1e75185ddd637b76__0_768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8f29656c0018a749a588__0_768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8f29656c0018a749a588",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8f29656c0018a749a588",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_3feed5026e1156d2fba2__768_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3feed5026e1156d2fba2",
        "role": "bwd_main",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3feed5026e1156d2fba2__768_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b__768_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_3feed5026e1156d2fba2__1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3feed5026e1156d2fba2",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3feed5026e1156d2fba2__1024_4096 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b__1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_383f709cd5c4fadd9436": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_383f709cd5c4fadd9436",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_383f709cd5c4fadd9436",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_03efa681eb38f59b00c2": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_03efa681eb38f59b00c2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03efa681eb38f59b00c2",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_f94861c36a14eb75e39c": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_f94861c36a14eb75e39c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_f94861c36a14eb75e39c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_f94861c36a14eb75e39c__768_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_f94861c36a14eb75e39c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_f94861c36a14eb75e39c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d75459d901441384398b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_d75459d901441384398b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d75459d901441384398b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8f032213ba6eeb5d959f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8f032213ba6eeb5d959f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_8f032213ba6eeb5d959f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8f032213ba6eeb5d959f__16384_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8f032213ba6eeb5d959f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_8f032213ba6eeb5d959f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8877322a3002e3a4fefd": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8877322a3002e3a4fefd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8877322a3002e3a4fefd",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8877322a3002e3a4fefd (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_659bfc53e5372850ae2e": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_659bfc53e5372850ae2e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_659bfc53e5372850ae2e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_122721bd6b4650866553": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3d20ab4a432bddf7349a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3d20ab4a432bddf7349a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3d20ab4a432bddf7349a",
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3d20ab4a432bddf7349a (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7da36ff557bb2bac8d89": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7da36ff557bb2bac8d89",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7da36ff557bb2bac8d89",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3d20ab4a432bddf7349a__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_3d20ab4a432bddf7349a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3d20ab4a432bddf7349a",
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3d20ab4a432bddf7349a__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7da36ff557bb2bac8d89__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7da36ff557bb2bac8d89",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7da36ff557bb2bac8d89",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_35a0b0ca6b2060313255": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_35a0b0ca6b2060313255 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_626aed743358c27513f1": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_35a0b0ca6b2060313255__44032_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 44032,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_35a0b0ca6b2060313255 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_626aed743358c27513f1__44032_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 44032,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_24643edc7e1143da24e0": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_24643edc7e1143da24e0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_24643edc7e1143da24e0",
        "role": "bwd_main",
        "select_rule": {
            "above": 48640,
            "at_most": 51200,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_24643edc7e1143da24e0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_78c26a53e83c9803bcbf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_78c26a53e83c9803bcbf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_78c26a53e83c9803bcbf",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 48640,
            "at_most": 51200,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # cake_rmsnorm_det_train_850eec3aff01f019a392 over the window of cake_rmsnorm_det_train_24643edc7e1143da24e0: no route selects this record; the FP32-weight route of sm_103a runs its twin there.
    "cake_rmsnorm_det_train_850eec3aff01f019a392": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_850eec3aff01f019a392",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_850eec3aff01f019a392",
        "role": "bwd_main",
        "select_rule": {
            "above": 48640,
            "at_most": 51200,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_850eec3aff01f019a392 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_cb164a27c0157ed0712d": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_cb164a27c0157ed0712d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cb164a27c0157ed0712d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 48640,
            "at_most": 51200,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_24643edc7e1143da24e0__51200_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_24643edc7e1143da24e0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_24643edc7e1143da24e0",
        "role": "bwd_main",
        "select_rule": {
            "above": 51200,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_24643edc7e1143da24e0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_78c26a53e83c9803bcbf__51200_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_78c26a53e83c9803bcbf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_78c26a53e83c9803bcbf",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 51200,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_35a0b0ca6b2060313255__103": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_35a0b0ca6b2060313255__103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_626aed743358c27513f1__103": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__32768_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__48640_51200": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 48640,
            "at_most": 51200,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__51200_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 51200,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__32768_44032": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__44032_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 44032,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4cdc4c495b3f59ff5069": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 8,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4cdc4c495b3f59ff5069",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4cdc4c495b3f59ff5069",
        "role": "fwd_large",
        "select_rule": {
            "above": 2048,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4cdc4c495b3f59ff5069 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_92cf7b87f1ebffb1b12b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 8,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_92cf7b87f1ebffb1b12b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_92cf7b87f1ebffb1b12b",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d292172df772adb17ae": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5d292172df772adb17ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5d292172df772adb17ae",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5d292172df772adb17ae (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_dffe53d5aae62879b794": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_dffe53d5aae62879b794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dffe53d5aae62879b794",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_dffe53d5aae62879b794 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_276474ee04bad8d3d764": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_276474ee04bad8d3d764",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_276474ee04bad8d3d764",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_0dafdf094615d080fcc9__256_1664": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_0dafdf094615d080fcc9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0dafdf094615d080fcc9",
        "role": "bwd_col",
        "select_rule": {
            "above": 256,
            "at_most": 1664,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0dafdf094615d080fcc9__256_1664 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_5cca2d0fa4553f992406__256_1664": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5cca2d0fa4553f992406",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5cca2d0fa4553f992406",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1664,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_0dafdf094615d080fcc9__0_1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_0dafdf094615d080fcc9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0dafdf094615d080fcc9",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0dafdf094615d080fcc9__0_1 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_5cca2d0fa4553f992406__0_1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5cca2d0fa4553f992406",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5cca2d0fa4553f992406",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_5d292172df772adb17ae__1664_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5d292172df772adb17ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5d292172df772adb17ae",
        "role": "bwd_main",
        "select_rule": {
            "above": 1664,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5d292172df772adb17ae__1664_19456 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb__1664_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1664,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_9cc51adfe9826056658b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_9cc51adfe9826056658b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9cc51adfe9826056658b",
        "role": "bwd_main",
        "select_rule": {
            "above": 19456,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9cc51adfe9826056658b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_bdfba721a43e0d093668": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_bdfba721a43e0d093668",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdfba721a43e0d093668",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19456,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_36cad444bc992ccc732d__368_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_36cad444bc992ccc732d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_36cad444bc992ccc732d",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 368,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_36cad444bc992ccc732d__368_19456 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b9d848cdbcfaeea0325c__368_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b9d848cdbcfaeea0325c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b9d848cdbcfaeea0325c",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 368,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_2f1499999535c6983213__19456_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2f1499999535c6983213",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2f1499999535c6983213",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 19456,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2f1499999535c6983213__19456_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_56d25337feb636c80b9a__19456_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_56d25337feb636c80b9a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56d25337feb636c80b9a",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 19456,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_302f64db6abe79e5259e": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_302f64db6abe79e5259e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_302f64db6abe79e5259e",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": 55296,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_302f64db6abe79e5259e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e98cb156abfe1666c76f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_e98cb156abfe1666c76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e98cb156abfe1666c76f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 55296,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_302f64db6abe79e5259e__55296_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_302f64db6abe79e5259e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_302f64db6abe79e5259e",
        "role": "bwd_main",
        "select_rule": {
            "above": 55296,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_302f64db6abe79e5259e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e98cb156abfe1666c76f__55296_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_e98cb156abfe1666c76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e98cb156abfe1666c76f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 55296,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_ceeef043ac64f37fc529": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_ceeef043ac64f37fc529",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ceeef043ac64f37fc529",
        "role": "bwd_main",
        "select_rule": {
            "above": 22528,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_ceeef043ac64f37fc529__58880_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_ceeef043ac64f37fc529",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ceeef043ac64f37fc529",
        "role": "bwd_main",
        "select_rule": {
            "above": 58880,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ceeef043ac64f37fc529 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9__58880_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 58880,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__16384_58880": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__58880_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 58880,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_596b457aaba073f8d69f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_596b457aaba073f8d69f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_596b457aaba073f8d69f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__32768_55296": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 55296,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__55296_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 55296,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_271082c56c222bf10d77": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_271082c56c222bf10d77",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_271082c56c222bf10d77",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c4d8be677fa70b841877": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c4d8be677fa70b841877__16384_50688": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c4d8be677fa70b841877__50688_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_60f47c5d00570ecf7740": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_60f47c5d00570ecf7740",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_60f47c5d00570ecf7740",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_60f47c5d00570ecf7740 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_61accbcc02891615f286": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_61accbcc02891615f286",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_61accbcc02891615f286",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_682675bda578f5b7a312": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_682675bda578f5b7a312",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_682675bda578f5b7a312",
        "role": "fwd_large",
        "select_rule": {
            "above": 12288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_682675bda578f5b7a312 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a30e5a3b69b7c183992f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 2,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_a30e5a3b69b7c183992f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a30e5a3b69b7c183992f",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 12288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bde56bd5a68c493f07a2": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_bde56bd5a68c493f07a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bde56bd5a68c493f07a2",
        "role": "fwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bde56bd5a68c493f07a2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_40210950f2e7ee48bc3e": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_40210950f2e7ee48bc3e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_40210950f2e7ee48bc3e",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23",
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_fdfa500c83d95df0f5c1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 1,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_fdfa500c83d95df0f5c1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_fdfa500c83d95df0f5c1",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__4096_14928": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 14928,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7673978dc698790054d2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7673978dc698790054d2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e2eabcb3b0915d916a13": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7673978dc698790054d2__1025_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 1025,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7673978dc698790054d2__1025_4096 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e2eabcb3b0915d916a13__1025_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 1025,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7673978dc698790054d2__quad": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7673978dc698790054d2__quad (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e2eabcb3b0915d916a13__quad": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_780b9833fd0a236ef0e4": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_780b9833fd0a236ef0e4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_780b9833fd0a236ef0e4",
        "role": "fwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_780b9833fd0a236ef0e4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_780b9833fd0a236ef0e4__cut": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_780b9833fd0a236ef0e4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_780b9833fd0a236ef0e4",
        "role": "fwd_cut",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_780b9833fd0a236ef0e4__cut (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b__cut": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b",
        "role": "fwd_cut_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7de1e51089d432dbc26b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7de1e51089d432dbc26b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7de1e51089d432dbc26b",
        "role": "fwd_mid",
        "select_rule": {
            "above": 3584,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7de1e51089d432dbc26b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_1974c5c86cee63f33370": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1974c5c86cee63f33370",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1974c5c86cee63f33370",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8392d1a008d58a006bba": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_8392d1a008d58a006bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8392d1a008d58a006bba",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8392d1a008d58a006bba (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_4371270e4db4b985998f": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_4371270e4db4b985998f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4371270e4db4b985998f",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8392d1a008d58a006bba__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_8392d1a008d58a006bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8392d1a008d58a006bba",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8392d1a008d58a006bba__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_4371270e4db4b985998f__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_4371270e4db4b985998f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4371270e4db4b985998f",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8757f6da86abf54363d5": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8757f6da86abf54363d5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8757f6da86abf54363d5",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8757f6da86abf54363d5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6769e393999bc2ba1926": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_6769e393999bc2ba1926",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6769e393999bc2ba1926",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_def532441f08b90a9903": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_def532441f08b90a9903",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_def532441f08b90a9903",
        "role": "fwd_quad",
        "select_rule": {
            "above": 4,
            "at_most": 1025,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_def532441f08b90a9903 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_20ebc686cc4099b9d026": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_20ebc686cc4099b9d026",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_20ebc686cc4099b9d026",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 4,
            "at_most": 1025,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_8757f6da86abf54363d5__0_4": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8757f6da86abf54363d5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8757f6da86abf54363d5",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8757f6da86abf54363d5__0_4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6769e393999bc2ba1926__0_4": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_6769e393999bc2ba1926",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6769e393999bc2ba1926",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_322574af704b51aef0bb": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_322574af704b51aef0bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_322574af704b51aef0bb",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_322574af704b51aef0bb (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9a8c0396edd85d4d545e": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_9a8c0396edd85d4d545e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9a8c0396edd85d4d545e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_322574af704b51aef0bb__48640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_322574af704b51aef0bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_322574af704b51aef0bb",
        "role": "bwd_mid",
        "select_rule": {
            "above": 48640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_322574af704b51aef0bb (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9a8c0396edd85d4d545e__48640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_9a8c0396edd85d4d545e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9a8c0396edd85d4d545e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 48640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_50044ee3bd1ca54bd821": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_50044ee3bd1ca54bd821",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_50044ee3bd1ca54bd821",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_50044ee3bd1ca54bd821 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_12d53fe5b85b212fc800__0_152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_12d53fe5b85b212fc800",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_12d53fe5b85b212fc800",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_12d53fe5b85b212fc800__0_152 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_27615d0e36c9f2fe0856__0_152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_27615d0e36c9f2fe0856",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_27615d0e36c9f2fe0856",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_5efbf78a48136634a5b3": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5efbf78a48136634a5b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5efbf78a48136634a5b3",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 1 rows: 27615d0e, the FP32-weight twin of 12d53fe5.
    "cake_rmsnorm_det_train_27615d0e36c9f2fe0856__ov103_0_1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_27615d0e36c9f2fe0856",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_27615d0e36c9f2fe0856",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1 and 512 rows: ff7de8d3, the FP32-weight twin of 5efbf78a.
    "cake_rmsnorm_det_train_ff7de8d3e4dcd036ea9a__ov103_1_512": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_ff7de8d3e4dcd036ea9a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ff7de8d3e4dcd036ea9a",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_50044ee3bd1ca54bd821__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_50044ee3bd1ca54bd821",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_50044ee3bd1ca54bd821",
        "role": "bwd_small",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_50044ee3bd1ca54bd821__152_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_ad2bb26ea5916baef963": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 6,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_ad2bb26ea5916baef963",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ad2bb26ea5916baef963",
        "role": "bwd_low",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ad2bb26ea5916baef963 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_16545056022785f2a7d8": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 6,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_16545056022785f2a7d8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_16545056022785f2a7d8",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3e644ea19b7d14e61ab9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_20c64c610d22c6db4c05": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_20c64c610d22c6db4c05",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_20c64c610d22c6db4c05",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_4822793ea0c954c11af6__0_203": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_4822793ea0c954c11af6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4822793ea0c954c11af6",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 203,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4822793ea0c954c11af6__0_203 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8616070567821eaa3e35__0_203": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 128,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_8616070567821eaa3e35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8616070567821eaa3e35",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 203,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9__203_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9",
        "role": "bwd_small",
        "select_rule": {
            "above": 203,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3e644ea19b7d14e61ab9__203_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_20c64c610d22c6db4c05__203_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_20c64c610d22c6db4c05",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_20c64c610d22c6db4c05",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 203,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_59270fb249dcba1ff679": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_59270fb249dcba1ff679",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_59270fb249dcba1ff679",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_59270fb249dcba1ff679 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__203_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 203,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__256_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__512_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_09601d42628de24f6f87": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_09601d42628de24f6f87",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_09601d42628de24f6f87 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9f07cea1d13b923ed91e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_9f07cea1d13b923ed91e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9f07cea1d13b923ed91e",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__0_227": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__0_227 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_2c60470d16e0a7fb6089__0_227": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_2c60470d16e0a7fb6089",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2c60470d16e0a7fb6089",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_f283bf47d2e3a49bc43c (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__1_4": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__1_4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_2c60470d16e0a7fb6089__1_4": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_2c60470d16e0a7fb6089",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2c60470d16e0a7fb6089",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__4_227": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb",
        "role": "bwd_col",
        "select_rule": {
            "above": 4,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__4_227 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_2c60470d16e0a7fb6089__4_227": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_2c60470d16e0a7fb6089",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2c60470d16e0a7fb6089",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 4,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_931757c844a9d257c08b__1_227": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_931757c844a9d257c08b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_931757c844a9d257c08b",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_931757c844a9d257c08b__1_227 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_404c8f54ab3129877350__1_227": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_404c8f54ab3129877350",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_404c8f54ab3129877350",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_09601d42628de24f6f87__227_1024": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_09601d42628de24f6f87",
        "role": "bwd_small",
        "select_rule": {
            "above": 227,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_09601d42628de24f6f87__227_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9f07cea1d13b923ed91e__227_1024": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_9f07cea1d13b923ed91e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9f07cea1d13b923ed91e",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 227,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_8e7f4d811e55a351a22e": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_8e7f4d811e55a351a22e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8e7f4d811e55a351a22e",
        "role": "bwd_low",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8e7f4d811e55a351a22e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_dfcbd32cdb8108643fba": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_dfcbd32cdb8108643fba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dfcbd32cdb8108643fba",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_123db0b2f7fb38cfab03": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_123db0b2f7fb38cfab03",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_123db0b2f7fb38cfab03",
        "role": "bwd_low",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_123db0b2f7fb38cfab03 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_632ca6ff01848a71f9f2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_632ca6ff01848a71f9f2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_632ca6ff01848a71f9f2",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_274b0055aace183f2367": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_274b0055aace183f2367",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_274b0055aace183f2367",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_274b0055aace183f2367 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_56ee8e66479625fd078a": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 6,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_56ee8e66479625fd078a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56ee8e66479625fd078a",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_7097069b8eea6ba44d5a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__227_1024": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 227,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fef03ce50a792c8704bd": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_fef03ce50a792c8704bd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fef03ce50a792c8704bd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9a830956447d4e304914": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_9a830956447d4e304914",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9a830956447d4e304914",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9a830956447d4e304914 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0c84999caf2b86c2e09e": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0c84999caf2b86c2e09e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0c84999caf2b86c2e09e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_24643edc7e1143da24e0__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_24643edc7e1143da24e0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_24643edc7e1143da24e0",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_24643edc7e1143da24e0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_78c26a53e83c9803bcbf__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_78c26a53e83c9803bcbf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_78c26a53e83c9803bcbf",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__1024_22528": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 22528,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__512_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__22528_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 22528,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5d75ea5f3c25579af496__24576_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_a4da85136fba4588239f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_a4da85136fba4588239f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a4da85136fba4588239f",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a4da85136fba4588239f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a72c6ece1900f748eaa2": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_a72c6ece1900f748eaa2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a72c6ece1900f748eaa2",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7dc29629ff21625607f8__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7dc29629ff21625607f8__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c3b1b46a2c6d1f9b3724": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_c3b1b46a2c6d1f9b3724",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c3b1b46a2c6d1f9b3724",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c3b1b46a2c6d1f9b3724 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_45057834f9e0e017b6ef": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_45057834f9e0e017b6ef",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_45057834f9e0e017b6ef",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_1207b05663ed2b4de14b__1_576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1207b05663ed2b4de14b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1207b05663ed2b4de14b",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1207b05663ed2b4de14b__1_576 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_137ba5328176fb07b37d__1_576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_137ba5328176fb07b37d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_137ba5328176fb07b37d",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_7dc29629ff21625607f8__576_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7dc29629ff21625607f8__576_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__576_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
        "role": "bwd_low",
        "select_rule": {
            "above": 384,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5c804aed7ad14b99e1d0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_41b80b6af97974feb181": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 384,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The dw reduce of the re-cut FP32-weight route of sm_100a, between 384 and 4096 rows.
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__384_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 384,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_61421bc72e57d14d8a32__384_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "free_sms": 8,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_61421bc72e57d14d8a32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_61421bc72e57d14d8a32",
        "role": "bwd_low",
        "select_rule": {
            "above": 384,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 384 and 4096 rows: 41b80b6a, the FP32-weight twin of 5c804aed.
    "cake_rmsnorm_det_train_41b80b6af97974feb181__ov100_384_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 384,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_61421bc72e57d14d8a32__256_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "free_sms": 8,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_61421bc72e57d14d8a32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_61421bc72e57d14d8a32",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 256 and 4096 rows: 41b80b6a, the FP32-weight twin of 5c804aed.
    "cake_rmsnorm_det_train_41b80b6af97974feb181__ov103_256_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__256_4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__384_767": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 384,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__767_4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 767,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0__103": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5c804aed7ad14b99e1d0__103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_41b80b6af97974feb181__103": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__0_4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__20_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 20,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__256_767": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c1a9efdf06c3087b4efe (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_21c8bb55c8d672b63952": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_21c8bb55c8d672b63952",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_21c8bb55c8d672b63952",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_21c8bb55c8d672b63952 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_4bf450b6af0314ed03be": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4bf450b6af0314ed03be",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4bf450b6af0314ed03be",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_1df7db17277d363b5168": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1df7db17277d363b5168",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1df7db17277d363b5168",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 14928,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1df7db17277d363b5168 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a99e2b380514c4d86d60": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_a99e2b380514c4d86d60",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a99e2b380514c4d86d60",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 14928,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_4df08ce51fb01ba1193f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4df08ce51fb01ba1193f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4df08ce51fb01ba1193f",
        "role": "bwd_large",
        "select_rule": {
            "above": 14928,
            "at_most": 15566,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4df08ce51fb01ba1193f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7a72e97bd6cb4dc0efc2": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7a72e97bd6cb4dc0efc2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7a72e97bd6cb4dc0efc2",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 14928,
            "at_most": 15566,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_f522f5b8d4343d96ca04": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_f522f5b8d4343d96ca04",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f522f5b8d4343d96ca04",
        "role": "bwd_large",
        "select_rule": {
            "above": 15566,
            "at_most": 40448,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_f522f5b8d4343d96ca04 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_109f19bb177c8f20cadc": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_109f19bb177c8f20cadc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_109f19bb177c8f20cadc",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 15566,
            "at_most": 40448,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_9cd3c766db4b70ed7213": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9cd3c766db4b70ed7213",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9cd3c766db4b70ed7213",
        "role": "bwd_large",
        "select_rule": {
            "above": 40448,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9cd3c766db4b70ed7213 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_51f46ff164b142ec8888": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_51f46ff164b142ec8888",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_51f46ff164b142ec8888",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 40448,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_1863320f3158d35eead6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1863320f3158d35eead6__4096_32768": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1863320f3158d35eead6__14928_15566": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 14928,
            "at_most": 15566,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1863320f3158d35eead6__15566_40448": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 15566,
            "at_most": 40448,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1863320f3158d35eead6__40448_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 40448,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_84efa2af7907877dfb56": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_84efa2af7907877dfb56",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_84efa2af7907877dfb56",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_84efa2af7907877dfb56 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6b13aa49e0c0414d9041": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_6b13aa49e0c0414d9041",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6b13aa49e0c0414d9041",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_92def3f97ddcec9165f0": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_92def3f97ddcec9165f0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_92def3f97ddcec9165f0",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_92def3f97ddcec9165f0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_5e3034bbb4032d8b7ad9": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_5e3034bbb4032d8b7ad9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e3034bbb4032d8b7ad9",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_6720b4a159be2b84b325__1_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_6720b4a159be2b84b325",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6720b4a159be2b84b325",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_6720b4a159be2b84b325__1_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_23c900134dc889302136__1_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_23c900134dc889302136",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_23c900134dc889302136",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_845cdcd8145ada631461__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_845cdcd8145ada631461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_845cdcd8145ada631461",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_845cdcd8145ada631461__256_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_838a9a4054731f7942af__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_838a9a4054731f7942af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_838a9a4054731f7942af",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_cc26e316ae40d683a118__1024_16384_107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_cc26e316ae40d683a118__1024_16384_107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c83c314c6dcf28ad7369__1024_16384_107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c83c314c6dcf28ad7369",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c83c314c6dcf28ad7369",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_845cdcd8145ada631461": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_845cdcd8145ada631461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_845cdcd8145ada631461",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_845cdcd8145ada631461 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_838a9a4054731f7942af": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_838a9a4054731f7942af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_838a9a4054731f7942af",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_21ee6fc93139c5e0cd41": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_21ee6fc93139c5e0cd41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_21ee6fc93139c5e0cd41",
        "role": "bwd_large",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_21ee6fc93139c5e0cd41 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3f270477a248188afe2a": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_3f270477a248188afe2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3f270477a248188afe2a",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_520453292492bd2725be": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_520453292492bd2725be",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_520453292492bd2725be",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b44107e05796c2922187": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b44107e05796c2922187 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0da8e18d4af432c6927a": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_a5f323db12acfb2cd565": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_a5f323db12acfb2cd565",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a5f323db12acfb2cd565",
        "role": "fwd_small",
        "select_rule": {
            "above": 1,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a5f323db12acfb2cd565 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_27fd540856828056457a": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_27fd540856828056457a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_27fd540856828056457a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 1,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b44107e05796c2922187__band": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "role": "fwd_band",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b44107e05796c2922187__band (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0da8e18d4af432c6927a__band": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "role": "fwd_band_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b44107e05796c2922187__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b44107e05796c2922187__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0da8e18d4af432c6927a__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b44107e05796c2922187__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b44107e05796c2922187",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b44107e05796c2922187__small (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0da8e18d4af432c6927a__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_0da5dd4b80ad9b5426ce": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0da5dd4b80ad9b5426ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0da5dd4b80ad9b5426ce",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0da5dd4b80ad9b5426ce (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0c0fa63d0ff2cdc484a4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0c0fa63d0ff2cdc484a4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0c0fa63d0ff2cdc484a4",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c1de7e41da4550cca0bc__0_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_c1de7e41da4550cca0bc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c1de7e41da4550cca0bc",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c1de7e41da4550cca0bc__0_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_30ec8bb68cf4fe0c817b__0_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 64,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_30ec8bb68cf4fe0c817b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_30ec8bb68cf4fe0c817b",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5572b2671862fe670ef4__1024_22528": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 8,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5572b2671862fe670ef4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5572b2671862fe670ef4",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 22528,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5572b2671862fe670ef4__1024_22528 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_47a01388444960da7aae__1024_22528": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 8,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_47a01388444960da7aae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_47a01388444960da7aae",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 22528,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5572b2671862fe670ef4__512_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 8,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5572b2671862fe670ef4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5572b2671862fe670ef4",
        "role": "bwd_mid",
        "select_rule": {
            "above": 512,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5572b2671862fe670ef4__512_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_47a01388444960da7aae__512_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 8,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_47a01388444960da7aae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_47a01388444960da7aae",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 512,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a__22528_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a",
        "role": "bwd_main",
        "select_rule": {
            "above": 22528,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5abcdc29d54ac214bd4a__22528_32768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a8eb754739a1589de3ec__22528_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_a8eb754739a1589de3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a8eb754739a1589de3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 22528,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a__24576_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a",
        "role": "bwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5abcdc29d54ac214bd4a__24576_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a8eb754739a1589de3ec__24576_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_a8eb754739a1589de3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a8eb754739a1589de3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_e9e21bbe4fa478075868": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_e9e21bbe4fa478075868",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e9e21bbe4fa478075868",
        "role": "bwd_main",
        "select_rule": {
            "above": 24064,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e9e21bbe4fa478075868 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0ff440daa6f9f304e544": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0ff440daa6f9f304e544",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0ff440daa6f9f304e544",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24064,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_e9e21bbe4fa478075868__50176_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_e9e21bbe4fa478075868",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e9e21bbe4fa478075868",
        "role": "bwd_main",
        "select_rule": {
            "above": 50176,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e9e21bbe4fa478075868 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_0ff440daa6f9f304e544__50176_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0ff440daa6f9f304e544",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0ff440daa6f9f304e544",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 50176,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_d41a514f5210e9adc39f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_d41a514f5210e9adc39f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d41a514f5210e9adc39f",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_d41a514f5210e9adc39f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b08633208e60b8ce94fd": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_b08633208e60b8ce94fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b08633208e60b8ce94fd",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a647fccc3d9fa9134733": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_a647fccc3d9fa9134733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a647fccc3d9fa9134733",
        "role": "bwd_low",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a647fccc3d9fa9134733 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_dc439f8335327e482025": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_dc439f8335327e482025",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dc439f8335327e482025",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H 4096 between 24064 and 50176 rows: the program of (256, 1024] at its host plan (A/B on VR200, 212 SMs).
    "cake_rmsnorm_det_train_a647fccc3d9fa9134733__24064_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_a647fccc3d9fa9134733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a647fccc3d9fa9134733",
        "role": "bwd_main",
        "select_rule": {
            "above": 24064,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 24064 and 50176 rows: 0ff440da, the FP32-weight twin of e9e21bbe.
    "cake_rmsnorm_det_train_0ff440daa6f9f304e544__ov107_24064_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_0ff440daa6f9f304e544",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0ff440daa6f9f304e544",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24064,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf__24064_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24064,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2030d80884deb8abcf7b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_2030d80884deb8abcf7b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2030d80884deb8abcf7b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_10e24516314c244771f4__4096_24064": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_10e24516314c244771f4__24064_50176": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24064,
            "at_most": 50176,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_10e24516314c244771f4__50176_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 50176,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_56d26996239835327501": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_56d26996239835327501",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_56d26996239835327501",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_56d26996239835327501 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_411e270c0575d26a77f4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_411e270c0575d26a77f4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_411e270c0575d26a77f4",
        "role": "bwd_main",
        "select_rule": {
            "above": 45056,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_411e270c0575d26a77f4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_64ffe19b041f5bfa34fb": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_64ffe19b041f5bfa34fb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_64ffe19b041f5bfa34fb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 45056,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_88510c609fd688aab969": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_88510c609fd688aab969",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_88510c609fd688aab969",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_88510c609fd688aab969 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6f2c9e90f551f35e99e6": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_6f2c9e90f551f35e99e6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6f2c9e90f551f35e99e6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_6a428da2227972ae18ae": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_6a428da2227972ae18ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6a428da2227972ae18ae",
        "role": "bwd_mid",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_6a428da2227972ae18ae (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e63beae98c236ce26335": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_e63beae98c236ce26335",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e63beae98c236ce26335",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_cfaae3cec98f8501832b": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_cfaae3cec98f8501832b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cfaae3cec98f8501832b",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_cfaae3cec98f8501832b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a546f410ec8cf03ef20d": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_a546f410ec8cf03ef20d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a546f410ec8cf03ef20d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_122721bd6b4650866553__4096_32768": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_det_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bad4db9153cc903b7ace": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bad4db9153cc903b7ace (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bad4db9153cc903b7ace__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bad4db9153cc903b7ace__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6__100": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bad4db9153cc903b7ace__ps5": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 5,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bad4db9153cc903b7ace__ps5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6__ps5": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 5,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bf933e1660cab8e5787b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_bf933e1660cab8e5787b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bf933e1660cab8e5787b",
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4096 and 20480 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_4096_20480": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bf933e1660cab8e5787b__wide": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_bf933e1660cab8e5787b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bf933e1660cab8e5787b",
        "role": "fwd_wide",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 20480 and 49152 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_20480_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b457c4d134e44476dc6f": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_b457c4d134e44476dc6f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b457c4d134e44476dc6f",
        "role": "bwd_large",
        "select_rule": {
            "above": 32768,
            "at_most": 36352,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b457c4d134e44476dc6f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8777b1c8538527c9aaf7": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_8777b1c8538527c9aaf7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8777b1c8538527c9aaf7",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 36352,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b457c4d134e44476dc6f__36352_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_b457c4d134e44476dc6f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b457c4d134e44476dc6f",
        "role": "bwd_large",
        "select_rule": {
            "above": 36352,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b457c4d134e44476dc6f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8777b1c8538527c9aaf7__36352_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_8777b1c8538527c9aaf7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8777b1c8538527c9aaf7",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 36352,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__32768_36352": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 36352,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__36352_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 36352,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5a8099e177e463daab9": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c5a8099e177e463daab9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c5a8099e177e463daab9",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c5a8099e177e463daab9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_01f0ebf3db9fa2c0884e": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_01f0ebf3db9fa2c0884e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_01f0ebf3db9fa2c0884e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_cf1d05546951bdbd9f5d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_cf1d05546951bdbd9f5d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cf1d05546951bdbd9f5d",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_cf1d05546951bdbd9f5d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_bdc524a233d2bc6e9571__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdc524a233d2bc6e9571",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_bdc524a233d2bc6e9571__mid103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdc524a233d2bc6e9571",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 1024 and 4096 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__wide103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 12,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d447b7f6398b60fe5daa",
        "role": "fwd_wide",
        "select_rule": {
            "above": 3072,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_d447b7f6398b60fe5daa__wide103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_bdc524a233d2bc6e9571__wide103": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 12,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bdc524a233d2bc6e9571",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 3072,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b93bf060d4d3defc8150": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b93bf060d4d3defc8150 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_62beb52fa50cc8624c9f": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_f2271417b1a596d0540f": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_f2271417b1a596d0540f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f2271417b1a596d0540f",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_f2271417b1a596d0540f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_44508df7dab2ab688f85__0_20": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_44508df7dab2ab688f85",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_44508df7dab2ab688f85",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 20,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_44508df7dab2ab688f85__0_20 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b3e6531e46667901ab91__0_20": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b3e6531e46667901ab91",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 20,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_f2271417b1a596d0540f__0_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_f2271417b1a596d0540f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f2271417b1a596d0540f",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 1 rows: b3e6531e, the FP32-weight twin of 44508df7.
    "cake_rmsnorm_det_train_b3e6531e46667901ab91__ov103_0_1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b3e6531e46667901ab91",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1 and 256 rows: 8a280a7e, the FP32-weight twin of f2271417.
    "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5__ov103_1_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0__256_767": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0",
        "role": "bwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5c804aed7ad14b99e1d0__256_767 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_41b80b6af97974feb181__256_767": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_41b80b6af97974feb181",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b93bf060d4d3defc8150__20_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "role": "bwd_small",
        "select_rule": {
            "above": 20,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b93bf060d4d3defc8150__20_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_62beb52fa50cc8624c9f__20_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 20,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__20_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 20,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b93bf060d4d3defc8150__mid": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b93bf060d4d3defc8150__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_62beb52fa50cc8624c9f__mid": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b93bf060d4d3defc8150__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b93bf060d4d3defc8150",
        "role": "bwd_mid",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b93bf060d4d3defc8150 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_62beb52fa50cc8624c9f__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62beb52fa50cc8624c9f",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__1216_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1216,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__48640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 48640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__4096_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__4096_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__1152_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__42496_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 42496,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__0_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__1664_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1664,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__19456_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19456,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__368_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 368,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e646bbdbd852884953da": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e646bbdbd852884953da (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6325c474ad9955a319e9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "role": "fwd_main_w32_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e646bbdbd852884953da__1024_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e646bbdbd852884953da__1024_4096 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6325c474ad9955a319e9__1024_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "role": "fwd_main_w32_residual",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e646bbdbd852884953da__4096_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e646bbdbd852884953da__4096_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6325c474ad9955a319e9__4096_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "role": "fwd_main_w32_residual",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_41805c5801f6956b4ce4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_41805c5801f6956b4ce4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_41805c5801f6956b4ce4",
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_41805c5801f6956b4ce4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_4be7a14e78ff59fa96e2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_4be7a14e78ff59fa96e2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4be7a14e78ff59fa96e2",
        "role": "fwd_main_w32_residual",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e646bbdbd852884953da__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e646bbdbd852884953da",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e646bbdbd852884953da__small (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6325c474ad9955a319e9__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__large": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "role": "bwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__large (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_07f9f2fbea7163639c27__large": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6",
        "role": "bwd_large",
        "select_rule": {
            "above": 1024,
            "at_most": 2047,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1024 and 16384 rows: b187e598, the FP32-weight twin of c73074cf.
    "cake_rmsnorm_det_train_b187e5989302def22b2d__ov100_1024_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_b187e5989302def22b2d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b187e5989302def22b2d",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_cc26e316ae40d683a118__2047_22528": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cc26e316ae40d683a118",
        "role": "bwd_large",
        "select_rule": {
            "above": 2047,
            "at_most": 22528,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 16384 and 58880 rows: 7e6a0b63, the FP32-weight twin of ceeef043.
    "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9__ov100_16384_58880": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__16384_32768": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__16384_32768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_07f9f2fbea7163639c27__16384_32768": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_07f9f2fbea7163639c27",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__1024_2047": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 2047,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__22528_58880": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 22528,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_d0960be986e58f449f50__4096_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7dc29629ff21625607f8": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7dc29629ff21625607f8 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_baff9ecbafe7de1e6030": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__0_214": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__0_214 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_698c3d36d67408394ccf__0_214": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_698c3d36d67408394ccf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_698c3d36d67408394ccf",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_e254714a110f4d45ea02": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_e254714a110f4d45ea02",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e254714a110f4d45ea02",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_e254714a110f4d45ea02 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9f139a52a93ccc69c3d1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9f139a52a93ccc69c3d1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9f139a52a93ccc69c3d1",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__1_31": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__1_31 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_698c3d36d67408394ccf__1_31": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_698c3d36d67408394ccf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_698c3d36d67408394ccf",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__31_214": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80",
        "role": "bwd_col",
        "select_rule": {
            "above": 31,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__31_214 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_698c3d36d67408394ccf__31_214": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_698c3d36d67408394ccf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_698c3d36d67408394ccf",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 31,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_7dc29629ff21625607f8__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7dc29629ff21625607f8__214_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_f6233ff84d0b93fe45db": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_f6233ff84d0b93fe45db",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f6233ff84d0b93fe45db",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_f6233ff84d0b93fe45db (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_5b177296918aca760fa2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_5b177296918aca760fa2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5b177296918aca760fa2",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_4176a24f8ee981dd2bb2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_4176a24f8ee981dd2bb2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4176a24f8ee981dd2bb2",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4176a24f8ee981dd2bb2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_06bbfeb2e313d41451fa": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_06bbfeb2e313d41451fa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_06bbfeb2e313d41451fa",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_dc478ac1195882aedee6__0_109": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_dc478ac1195882aedee6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dc478ac1195882aedee6",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 109,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_dc478ac1195882aedee6__0_109 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e83ee45f66352eaef755__0_109": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 96,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_e83ee45f66352eaef755",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e83ee45f66352eaef755",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 109,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_7dc29629ff21625607f8__768_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7dc29629ff21625607f8__768_1024 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__768_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_9b727c2c9933b2ca6724": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__768_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_628248febbdfe17c3325": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_628248febbdfe17c3325",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_628248febbdfe17c3325",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_628248febbdfe17c3325__576_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_628248febbdfe17c3325",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_628248febbdfe17c3325",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_db5c0b673c516bd0295d": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_db5c0b673c516bd0295d__12288_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_db5c0b673c516bd0295d__16384_45056": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_db5c0b673c516bd0295d__45056_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 45056,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fd913ecd254a6bba4b18": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_fd913ecd254a6bba4b18 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f0db3068d0d05442a59d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 49152 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_49152_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 640,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f0db3068d0d05442a59d__low": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 640,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f0db3068d0d05442a59d__low107": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_0489c784c13bb25499f6": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_0489c784c13bb25499f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0489c784c13bb25499f6",
        "role": "fwd_small",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0489c784c13bb25499f6 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f2e721db4c30900f12fe": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_f2e721db4c30900f12fe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f2e721db4c30900f12fe",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_76b9c06cc9ec083c24b8 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f8087de00e0409186d93": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_f8087de00e0409186d93",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f8087de00e0409186d93",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8__0_24": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 24,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_76b9c06cc9ec083c24b8__0_24 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f8087de00e0409186d93__0_24": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_f8087de00e0409186d93",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f8087de00e0409186d93",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 24,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_0489c784c13bb25499f6__24_8192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_0489c784c13bb25499f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0489c784c13bb25499f6",
        "role": "fwd_small",
        "select_rule": {
            "above": 24,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0489c784c13bb25499f6__24_8192 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f2e721db4c30900f12fe__24_8192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_f2e721db4c30900f12fe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f2e721db4c30900f12fe",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 24,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1b284baec9663d6de1b9": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_1b284baec9663d6de1b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1b284baec9663d6de1b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1b284baec9663d6de1b9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_77456435c9fbd1d18fcf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_77456435c9fbd1d18fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_77456435c9fbd1d18fcf",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0",
        "role": "fwd_rows",
        "select_rule": {
            "above": 40960,
            "at_most": 81920,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_86d51eff64fc9fe57cf0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_42d82d26e97c36651dae": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_42d82d26e97c36651dae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_42d82d26e97c36651dae",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 40960,
            "at_most": 81920,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1388ab779a35c4e41efb": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_1388ab779a35c4e41efb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1388ab779a35c4e41efb",
        "role": "fwd_rows",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1388ab779a35c4e41efb (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b0299f6e0881fb35f27b": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_b0299f6e0881fb35f27b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b0299f6e0881fb35f27b",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_466331ff3cd2d6c62738": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_466331ff3cd2d6c62738 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_32bebe5a7782311d60a0": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_ed9483d8c356affee55d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_ed9483d8c356affee55d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ed9483d8c356affee55d",
        "role": "fwd_rows",
        "select_rule": {
            "above": 196608,
            "at_most": 327680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_ed9483d8c356affee55d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_d0a656f053396a58c49a": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_d0a656f053396a58c49a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d0a656f053396a58c49a",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 196608,
            "at_most": 327680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 327680,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 327680,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bbb05314930d192dfe7b": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_bbb05314930d192dfe7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bbb05314930d192dfe7b",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bbb05314930d192dfe7b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a8fffd93682c5e382dc5": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a8fffd93682c5e382dc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a8fffd93682c5e382dc5",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_1b284baec9663d6de1b9__20480_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_1b284baec9663d6de1b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1b284baec9663d6de1b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 20480,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_1b284baec9663d6de1b9__20480_40960 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_77456435c9fbd1d18fcf__20480_40960": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_77456435c9fbd1d18fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_77456435c9fbd1d18fcf",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_466331ff3cd2d6c62738__81920_114688": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_466331ff3cd2d6c62738__81920_114688 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_32bebe5a7782311d60a0__81920_114688": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0__114688_196608": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0",
        "role": "fwd_rows",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_86d51eff64fc9fe57cf0__114688_196608 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_42d82d26e97c36651dae__114688_196608": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_42d82d26e97c36651dae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_42d82d26e97c36651dae",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_37f663183cbff23c2893": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_37f663183cbff23c2893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_37f663183cbff23c2893",
        "role": "fwd_rows",
        "select_rule": {
            "above": 327680,
            "at_most": 368640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_37f663183cbff23c2893 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_18e3c0a5204006b3a56f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_18e3c0a5204006b3a56f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_18e3c0a5204006b3a56f",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 327680,
            "at_most": 368640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5__368640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 368640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5__368640_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461__368640_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 368640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5ca409f44e169148301": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_c5ca409f44e169148301",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c5ca409f44e169148301",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c5ca409f44e169148301 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8f2756937eaac6b02502": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_8f2756937eaac6b02502",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8f2756937eaac6b02502",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_bbb05314930d192dfe7b__6144_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_bbb05314930d192dfe7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_bbb05314930d192dfe7b",
        "role": "fwd_rows",
        "select_rule": {
            "above": 6144,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_bbb05314930d192dfe7b__6144_32768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_a8fffd93682c5e382dc5__6144_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a8fffd93682c5e382dc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a8fffd93682c5e382dc5",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_466331ff3cd2d6c62738__32768_98304": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": 98304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_466331ff3cd2d6c62738__32768_98304 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_32bebe5a7782311d60a0__32768_98304": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 98304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_37f663183cbff23c2893__98304_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_37f663183cbff23c2893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_37f663183cbff23c2893",
        "role": "fwd_rows",
        "select_rule": {
            "above": 98304,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_37f663183cbff23c2893__98304_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_18e3c0a5204006b3a56f__98304_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_18e3c0a5204006b3a56f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_18e3c0a5204006b3a56f",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 98304,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_af005462cd6a301f31c4__0_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_af005462cd6a301f31c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_af005462cd6a301f31c4",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_af005462cd6a301f31c4__0_65536 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b7d9577ff884de98ea06__0_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__0_1680": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1680,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__0_1680 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__0_1680": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1680,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__1680_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 1680,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__1680_2048 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__1680_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1680,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_6959ea32516c82acae98": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_6959ea32516c82acae98",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6959ea32516c82acae98",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 87,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_6959ea32516c82acae98 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c484e2676d7b535ff396": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_c484e2676d7b535ff396",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c484e2676d7b535ff396",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 87,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__0_87": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 87,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__0_87 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__0_87": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 87,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__87_2048": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 87,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__87_2048 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__87_2048": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 87,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_af005462cd6a301f31c4": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_af005462cd6a301f31c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_af005462cd6a301f31c4",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_af005462cd6a301f31c4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b7d9577ff884de98ea06": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__5120_12288": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__5120_12288 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__5120_12288": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_af005462cd6a301f31c4__12288_40960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_af005462cd6a301f31c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_af005462cd6a301f31c4",
        "role": "bwd_main",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_af005462cd6a301f31c4__12288_40960 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b7d9577ff884de98ea06__12288_40960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__40960_80896": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 40960,
            "at_most": 80896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__40960_80896 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__40960_80896": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 40960,
            "at_most": 80896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__40960_99840": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 40960,
            "at_most": 99840,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__40960_99840 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__40960_99840": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 40960,
            "at_most": 99840,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__65536_70656": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 70656,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__65536_70656 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__65536_70656": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 70656,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__80896_1572864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 80896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__80896_1572864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 80896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__99840_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 99840,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__99840_1572864 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__99840_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 99840,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__99840_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 99840,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__99840_720896 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__99840_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 99840,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__720896_1572864 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__70656_1572864": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 70656,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__70656_1572864 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__70656_1572864": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 70656,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__0_1572864": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e__0_1572864 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__0_1572864": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_a385ae1681a2c078344e__1572864_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a385ae1681a2c078344e (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_420788e663a2c3a810f6__1572864_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 4,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__0_1680": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 1680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__1680_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1680,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__0_87": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 87,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__87_2048": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 87,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__2048_5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__2048_5120": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__5120_12288": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__5120_12288": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 5120,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__12288_40960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__12288_40960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 12288,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__40960_80896": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 40960,
            "at_most": 80896,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__40960_80896": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 40960,
            "at_most": 80896,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 40960,
            "at_most": 99840,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_312bbe084eac69d858c4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_312bbe084eac69d858c4__65536_70656": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 65536,
            "at_most": 70656,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__80896_1572864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 80896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__80896_1572864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 80896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__99840_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 99840,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__99840_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 99840,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__99840_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 99840,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_affa1b95f68bab8c411b__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_312bbe084eac69d858c4__70656_1572864": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 70656,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_312bbe084eac69d858c4__0_1572864": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_72138dce40d43da44aaf__1572864_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_312bbe084eac69d858c4__1572864_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b5e173f5ecc736d91109__mid": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b5e173f5ecc736d91109",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b5e173f5ecc736d91109",
        "role": "fwd_mid",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 640 and 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov100_640_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 640 and 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov103_640_6144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__small": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "role": "fwd_small",
        "select_rule": {
            "above": 64,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 64 and 640 rows: ba28d2c0, the FP32-weight twin of fd1097c4.
    "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e__ov100_64_640": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 64,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 64 and 640 rows: 1defffdf, the FP32-weight twin of 5eef28cd.
    "cake_rmsnorm_det_train_1defffdf2712cae36784__ov103_64_640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_1defffdf2712cae36784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1defffdf2712cae36784",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 64,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e12e07353014329ebad": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_6e12e07353014329ebad",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e12e07353014329ebad",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 0 and 64 rows: c1819da8, the FP32-weight twin of 6e12e073.
    "cake_rmsnorm_det_train_c1819da87ab45ebb8d56__ov100_0_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c1819da87ab45ebb8d56",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c1819da87ab45ebb8d56",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 64 rows: c1819da8, the FP32-weight twin of 6e12e073.
    "cake_rmsnorm_det_train_c1819da87ab45ebb8d56__ov103_0_64": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c1819da87ab45ebb8d56",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c1819da87ab45ebb8d56",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5eef28cd35498e9abfa5": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 6144 and 16384 rows: b49de9b4, the FP32-weight twin of 044d4a2a.
    "cake_rmsnorm_det_train_b49de9b47372fa4e1051__ov100_6144_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 16,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, above 16384 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov100_16384_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov103_6144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_fd1097c45c1e328871fd__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_fd1097c45c1e328871fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_fd1097c45c1e328871fd",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_fd1097c45c1e328871fd__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__low107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5eef28cd35498e9abfa5__low107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_1defffdf2712cae36784__low107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_1defffdf2712cae36784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1defffdf2712cae36784",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_b5e173f5ecc736d91109__mid107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b5e173f5ecc736d91109",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b5e173f5ecc736d91109",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1152,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b5e173f5ecc736d91109__mid107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__mid107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 6,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1152,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_044d4a2aa0553c16db06__large107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_044d4a2aa0553c16db06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_044d4a2aa0553c16db06",
        "role": "fwd_large",
        "select_rule": {
            "above": 6144,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_044d4a2aa0553c16db06__large107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b49de9b47372fa4e1051__large107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 22,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_044d4a2aa0553c16db06__wide107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_044d4a2aa0553c16db06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_044d4a2aa0553c16db06",
        "role": "fwd_wide",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_044d4a2aa0553c16db06__wide107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b49de9b47372fa4e1051__wide107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 48,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__107": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 49152 rows: b49de9b4, the FP32-weight twin of 044d4a2a.
    "cake_rmsnorm_det_train_b49de9b47372fa4e1051__ov107_49152_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 64,
            "kind": "rows_per_cta",
            "rows_per_cta": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b49de9b47372fa4e1051",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_db01cec1153cd7fb2759": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_db01cec1153cd7fb2759",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_db01cec1153cd7fb2759",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_db01cec1153cd7fb2759 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6042554b427634669739": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_6042554b427634669739",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6042554b427634669739",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_dd3a35731e2e1665dd71": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_dd3a35731e2e1665dd71 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_747d86f34cb58479c7ca": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_080e0a6c3b18e9bad790__0_1": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_080e0a6c3b18e9bad790",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_080e0a6c3b18e9bad790__0_1 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_139d162a8bb29d1b2d35__0_1": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_139d162a8bb29d1b2d35",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 0 and 1 rows: 747d86f3, the FP32-weight twin of dd3a3573.
    "cake_rmsnorm_det_train_747d86f34cb58479c7ca__ov100_0_1": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_db01cec1153cd7fb2759__121_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_db01cec1153cd7fb2759",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_db01cec1153cd7fb2759",
        "role": "bwd_small",
        "select_rule": {
            "above": 121,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_db01cec1153cd7fb2759__121_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_6042554b427634669739__121_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_6042554b427634669739",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6042554b427634669739",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 121,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "role": "bwd_main",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b482c84dbffcd75a4a4c (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_40227b2a2b2122628d58": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__6144_50688": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "role": "bwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b482c84dbffcd75a4a4c (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_40227b2a2b2122628d58__6144_50688": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # cake_rmsnorm_det_train_c145ef195d7aeb04f242 over the window of cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__6144_50688: no route selects this record; the FP32-weight route of sm_100a runs its twin there.
    "cake_rmsnorm_det_train_c145ef195d7aeb04f242__6144_50688": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c145ef195d7aeb04f242",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c145ef195d7aeb04f242",
        "role": "bwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c145ef195d7aeb04f242__6144_50688 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9479b1298a1b19d4d996__6144_50688": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9479b1298a1b19d4d996",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9479b1298a1b19d4d996",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__50688_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "role": "bwd_main",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__50688_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_40227b2a2b2122628d58__50688_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # cake_rmsnorm_det_train_c145ef195d7aeb04f242 over the window of cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__50688_inf: no route selects this record; the FP32-weight route of sm_100a runs its twin there.
    "cake_rmsnorm_det_train_c145ef195d7aeb04f242__50688_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c145ef195d7aeb04f242",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c145ef195d7aeb04f242",
        "role": "bwd_main",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c145ef195d7aeb04f242__50688_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_9479b1298a1b19d4d996__50688_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_9479b1298a1b19d4d996",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9479b1298a1b19d4d996",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_2fc4932fbde0bc848b32": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 0,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2fc4932fbde0bc848b32 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_79811a76205d7d7dd197": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_79811a76205d7d7dd197",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_79811a76205d7d7dd197",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 0,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_080e0a6c3b18e9bad790__0_25": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_080e0a6c3b18e9bad790",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 25,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_080e0a6c3b18e9bad790__0_25 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_139d162a8bb29d1b2d35__0_25": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_139d162a8bb29d1b2d35",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 25,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_dd3a35731e2e1665dd71__0_64": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_dd3a35731e2e1665dd71 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_747d86f34cb58479c7ca__0_64": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_2fc4932fbde0bc848b32__64_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2fc4932fbde0bc848b32__64_5120 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_79811a76205d7d7dd197__64_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_79811a76205d7d7dd197",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_79811a76205d7d7dd197",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_2fc4932fbde0bc848b32__5120_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 5120,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2fc4932fbde0bc848b32 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_79811a76205d7d7dd197__5120_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 2,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_79811a76205d7d7dd197",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_79811a76205d7d7dd197",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 5120,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_838c077df3a6d35d55b7": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_838c077df3a6d35d55b7 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_cd612de56c0985bb4caa": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_dd3a35731e2e1665dd71__1_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_dd3a35731e2e1665dd71__1_256 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_747d86f34cb58479c7ca__1_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_080e0a6c3b18e9bad790__256_1280": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_080e0a6c3b18e9bad790",
        "role": "bwd_col",
        "select_rule": {
            "above": 256,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_080e0a6c3b18e9bad790__256_1280 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_139d162a8bb29d1b2d35__256_1280": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 80,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_139d162a8bb29d1b2d35",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_838c077df3a6d35d55b7__1280_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 1280,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_838c077df3a6d35d55b7__1280_4096 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_cd612de56c0985bb4caa__1280_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 1280,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_6816fb80e97380f1bcbc": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_6816fb80e97380f1bcbc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6816fb80e97380f1bcbc",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_6816fb80e97380f1bcbc (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_468cc4d1b8128187b4b3": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_468cc4d1b8128187b4b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_468cc4d1b8128187b4b3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b8b0913b40f035f43b76": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b8b0913b40f035f43b76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b8b0913b40f035f43b76",
        "role": "bwd_main",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b8b0913b40f035f43b76 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_7b9e10b21b75d4615135": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_7b9e10b21b75d4615135",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7b9e10b21b75d4615135",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__19968_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
        "role": "bwd_main",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_b482c84dbffcd75a4a4c (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_40227b2a2b2122628d58__19968_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_40227b2a2b2122628d58",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_47da668e7e96652f840b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_47da668e7e96652f840b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_47da668e7e96652f840b",
        "role": "bwd_main",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_47da668e7e96652f840b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_de0cf457cd240f9c6422": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 4,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_de0cf457cd240f9c6422",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_de0cf457cd240f9c6422",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__121_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 121,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__256_6144": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__6144_50688": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 6144,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__50688_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__0_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__64_5120": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 5120,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__5120_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 5120,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2af4cf2f58ba58f78099": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 0,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__1280_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1280,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__4096_19968": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__19968_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c5831b29d17867842e5f__19968_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19968,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_2dc3fb268514e9182c6a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_2dc3fb268514e9182c6a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_2dc3fb268514e9182c6a",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_2dc3fb268514e9182c6a (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_153f811937bb0157224b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_153f811937bb0157224b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_153f811937bb0157224b",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1 and 1024 rows: 744d7429, the FP32-weight twin of 89fa3e96.
    "cake_rmsnorm_det_train_744d74299c2d45867c94__ov100_1_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_744d74299c2d45867c94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_744d74299c2d45867c94",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1 and 1024 rows: 744d7429, the FP32-weight twin of 89fa3e96.
    "cake_rmsnorm_det_train_744d74299c2d45867c94__ov103_1_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_744d74299c2d45867c94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_744d74299c2d45867c94",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac",
        "role": "bwd_col",
        "select_rule": {
            "above": 64,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2__128_1024": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_col",
        "select_rule": {
            "above": 128,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_0c581a133624ff396ab4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_0c581a133624ff396ab4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0c581a133624ff396ab4",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0c581a133624ff396ab4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_561bea88a25c2f245c29": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_561bea88a25c2f245c29",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_561bea88a25c2f245c29",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5e32dd889d72a75e348b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__12288_24576": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 12288,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5e32dd889d72a75e348b__12288_24576 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__12288_24576": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 16384 and 24576 rows: ba8feeec, the FP32-weight twin of 5e32dd88.
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_16384_24576": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The dw reduce of the re-cut FP32-weight route of sm_103a, between 16384 and 24576 rows.
    "cake_rmsnorm_det_train_e21352eeed287fc98d97__16384_24576": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4294c01f97e2459b646b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_4294c01f97e2459b646b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4294c01f97e2459b646b",
        "role": "bwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_4294c01f97e2459b646b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_58305d0f80e3e79161d4": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_58305d0f80e3e79161d4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_58305d0f80e3e79161d4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__1024_8192": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e21352eeed287fc98d97": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 12288,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__24576_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9346600882e53a589509": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 8,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 16,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_9346600882e53a589509",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9346600882e53a589509",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9346600882e53a589509 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 8,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 16,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_401a018c68bc082078b2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 16,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 16,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_401a018c68bc082078b2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_401a018c68bc082078b2",
        "role": "bwd_col",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_401a018c68bc082078b2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 16,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 16,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_8895b0224bb3d596a3a6": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 16,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_8895b0224bb3d596a3a6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8895b0224bb3d596a3a6",
        "role": "bwd_col",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_8895b0224bb3d596a3a6 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_103805d93149c8668264": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 16,
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_103805d93149c8668264",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_103805d93149c8668264",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_a5f2942d61e591f57a76": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 16,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_a5f2942d61e591f57a76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a5f2942d61e591f57a76",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a5f2942d61e591f57a76 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8d8ef930b7d650c5538a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 16,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_8d8ef930b7d650c5538a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8d8ef930b7d650c5538a",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_809235a3965d337a6877": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_809235a3965d337a6877",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_809235a3965d337a6877",
        "role": "bwd_col",
        "select_rule": {
            "above": 128,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_809235a3965d337a6877 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 128,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_12bbedff93fcaafa475d": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
        "role": "bwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_12bbedff93fcaafa475d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_80956d258172dfa76c2a": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_80956d258172dfa76c2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_80956d258172dfa76c2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__4096_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c646e0976e08c6fe8a91__4096_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__4096_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_59474336746fb32d4e51": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_59474336746fb32d4e51__4096_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__2_4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9e8a70075f28be558bba__2_4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__2_4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__4_512": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 4 and 32 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov100_4_32": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 32 and 64 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov100_32_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 64 and 512 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov100_64_512": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__512_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 512 and 1024 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov100_512_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__1024_2048": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1024 and 32768 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov100_1024_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__32768_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9e8a70075f28be558bba__32768_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__32768_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3808022dd402387eed16": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3808022dd402387eed16 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_628267fc724c2c8f0e07__2_4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_628267fc724c2c8f0e07__2_4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_1ecf9c260e418bc70046__2_4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3808022dd402387eed16__4_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 4 and 64 rows: 513f4ffb, the FP32-weight twin of 3808022d.
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863__ov100_4_64": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 64 and 128 rows: 1ecf9c26, the FP32-weight twin of 628267fc.
    "cake_rmsnorm_det_train_1ecf9c260e418bc70046__ov100_64_128": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 128 and 16384 rows: 513f4ffb, the FP32-weight twin of 3808022d.
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863__ov100_128_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 128,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_628267fc724c2c8f0e07": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_628267fc724c2c8f0e07 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_1ecf9c260e418bc70046": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 0 and 1 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_0_1": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1 and 4 rows: c43709ca, the FP32-weight twin of 03a74dfe.
    "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov100_1_4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 4 and 8 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_4_8": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 8 and 16 rows: c43709ca, the FP32-weight twin of 03a74dfe.
    "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov100_8_16": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752__16_32": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 16 and 256 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_16_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__32_256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752__256_512": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 256,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 256 and 1024 rows: c43709ca, the FP32-weight twin of 03a74dfe.
    "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov100_256_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__512_8192": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 512,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1024 and 2048 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_1024_2048": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 2048 and 32768 rows: 58c2fce1, the FP32-weight twin of 007e82b2.
    "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__ov100_2048_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_39b9b41b16769252153d": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_39b9b41b16769252153d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_39b9b41b16769252153d",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 8192 and 24576 rows: 62e8e275, the FP32-weight twin of 39b9b41b.
    "cake_rmsnorm_det_train_62e8e2756e015c39bf5c__ov103_8192_24576": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_62e8e2756e015c39bf5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62e8e2756e015c39bf5c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_007e82b24390c88156ac__24576_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03b5516039c533b453b9__32768_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03b5516039c533b453b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03b5516039c533b453b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_03b5516039c533b453b9__32768_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_99b209a37f0203af57ce__32768_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__1024_1536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1024 and 1536 rows: ba8feeec, the FP32-weight twin of 5e32dd88.
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_1024_1536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__1536_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1536 and 2048 rows: ba8feeec, the FP32-weight twin of 5e32dd88.
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_1536_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__2048_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 2048 and 16384 rows: ba8feeec, the FP32-weight twin of 5e32dd88.
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_2048_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The dw reduce of the re-cut FP32-weight route of sm_103a, between 2048 and 16384 rows.
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__4096_8192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__1024_1536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e21352eeed287fc98d97__1536_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4830a91c2471462a29c8__4096_8192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_4830a91c2471462a29c8",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_4830a91c2471462a29c8",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__16384_24576": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5e32dd889d72a75e348b__16384_24576 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__16384_24576": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_12bbedff93fcaafa475d__512_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
        "role": "bwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 512 and 4096 rows: 80956d25, the FP32-weight twin of 12bbedff.
    "cake_rmsnorm_det_train_80956d258172dfa76c2a__ov103_512_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_80956d258172dfa76c2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_80956d258172dfa76c2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c646e0976e08c6fe8a91": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 4096 rows: c66b8cbd, the FP32-weight twin of c646e097.
    "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__ov103_4096_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_59474336746fb32d4e51__512_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_59474336746fb32d4e51__2048_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 2 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_0_2": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 2 and 4 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov103_2_4": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 4 and 32 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_4_32": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 32 and 64 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov103_32_64": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 64 and 512 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_64_512": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 512 and 1024 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov103_512_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1024 and 32768 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_1024_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3808022dd402387eed16__0_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3808022dd402387eed16__0_16384 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863__0_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__0_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 512 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov103_0_512": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752__256_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 512 and 1024 rows: c43709ca, the FP32-weight twin of 03a74dfe.
    "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov103_512_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__1024_8192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1024 and 2048 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov103_1024_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 2048 and 8192 rows: 99b209a3, the FP32-weight twin of 03b55160.
    "cake_rmsnorm_det_train_99b209a37f0203af57ce__ov103_2048_8192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_007e82b24390c88156ac__24576_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 24576 and 32768 rows: 58c2fce1, the FP32-weight twin of 007e82b2.
    "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__ov103_24576_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 32768 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov103_32768_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac__1_8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac",
        "role": "bwd_col",
        "select_rule": {
            "above": 1,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac__1_8 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_cfd067433e487c414dbb__1_8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_cfd067433e487c414dbb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cfd067433e487c414dbb",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 1,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2__8_1536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_col",
        "select_rule": {
            "above": 8,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2__8_1536 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_744d74299c2d45867c94__8_1536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "col_tasks": 32,
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_744d74299c2d45867c94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_744d74299c2d45867c94",
        "role": "bwd_col_w32",
        "select_rule": {
            "above": 8,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_det_train_0c581a133624ff396ab4__1536_2048": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_0c581a133624ff396ab4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_0c581a133624ff396ab4",
        "role": "bwd_main",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_0c581a133624ff396ab4__1536_2048 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_561bea88a25c2f245c29__1536_2048": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_561bea88a25c2f245c29",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_561bea88a25c2f245c29",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_5e32dd889d72a75e348b__2048_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_5e32dd889d72a75e348b__2048_24576 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__2048_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_4830a91c2471462a29c8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_4830a91c2471462a29c8",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_4830a91c2471462a29c8",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_e21352eeed287fc98d97__24576_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 256,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_e21352eeed287fc98d97",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_12bbedff93fcaafa475d__512_8192": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
        "role": "bwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 512 and 4096 rows: 80956d25, the FP32-weight twin of 12bbedff.
    "cake_rmsnorm_det_train_80956d258172dfa76c2a__ov107_512_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_80956d258172dfa76c2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_80956d258172dfa76c2a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4096 and 16384 rows: c66b8cbd, the FP32-weight twin of c646e097.
    "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__ov107_4096_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # The dw reduce of the re-cut FP32-weight route of sm_107a, between 512 and 4096 rows.
    "cake_rmsnorm_det_train_849ee5d846ddc145e3a9__512_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_849ee5d846ddc145e3a9",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_849ee5d846ddc145e3a9",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The dw reduce of the re-cut FP32-weight route of sm_107a, between 4096 and 16384 rows.
    "cake_rmsnorm_det_train_382c8727d8b150884d7f__4096_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 320,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_382c8727d8b150884d7f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_382c8727d8b150884d7f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__8192_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "role": "bwd_main",
        "select_rule": {
            "above": 8192,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__16384_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c646e0976e08c6fe8a91__16384_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__16384_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_849ee5d846ddc145e3a9": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_849ee5d846ddc145e3a9",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_849ee5d846ddc145e3a9",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_382c8727d8b150884d7f__8192_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 320,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_382c8727d8b150884d7f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_382c8727d8b150884d7f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 8192,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_55c3530541ac1903b775": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_55c3530541ac1903b775",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_det_train_55c3530541ac1903b775",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__4096_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_c646e0976e08c6fe8a91__4096_16384 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__4096_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_8 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__0_8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__8_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_9e8a70075f28be558bba__8_16 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__8_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__16_256": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 16 and 512 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov107_16_512": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 512 and 1536 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov107_512_1536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__1024_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 1536 and 4096 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov107_1536_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__4096_8192": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4096 and 8192 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov107_4096_8192": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_cffbc5c8de9c874b207c": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_cffbc5c8de9c874b207c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cffbc5c8de9c874b207c",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 8192 and 12288 rows: f5110367, the FP32-weight twin of 141ce1cb.
    "cake_rmsnorm_det_train_f5110367098beea13130__ov107_8192_12288": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_f5110367098beea13130",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f5110367098beea13130",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_9e8a70075f28be558bba__12288_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 12288,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 12288 and 16384 rows: 364ed18c, the FP32-weight twin of 6e5ca1f4.
    "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov107_12288_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 16384 and 24576 rows: 8bd2d836, the FP32-weight twin of 91c090c0.
    "cake_rmsnorm_det_train_8bd2d83654da4780c48a__ov107_16384_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_8bd2d83654da4780c48a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8bd2d83654da4780c48a",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 24576 and 32768 rows: d99009bf, the FP32-weight twin of 9e8a7007.
    "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov107_24576_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_cffbc5c8de9c874b207c__32768_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_cffbc5c8de9c874b207c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_cffbc5c8de9c874b207c",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 32768 rows: 008fb179, the FP32-weight twin of cffbc5c8.
    "cake_rmsnorm_det_train_008fb179743126fb2d35__ov107_32768_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_008fb179743126fb2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_008fb179743126fb2d35",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_141ce1cb9005dddeee33": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_141ce1cb9005dddeee33",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_141ce1cb9005dddeee33",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_141ce1cb9005dddeee33 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_f5110367098beea13130": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_f5110367098beea13130",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_f5110367098beea13130",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_91c090c06a5add02f1fd": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_91c090c06a5add02f1fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_91c090c06a5add02f1fd",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_91c090c06a5add02f1fd (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_8bd2d83654da4780c48a": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_det_train_8bd2d83654da4780c48a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_8bd2d83654da4780c48a",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3808022dd402387eed16__0_32": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3808022dd402387eed16__0_32 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863__0_32": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_628267fc724c2c8f0e07__32_64": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_628267fc724c2c8f0e07__32_64 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_1ecf9c260e418bc70046__32_64": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3808022dd402387eed16__64_2048": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 64,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_3808022dd402387eed16__64_2048 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863__64_2048": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7221de56ded18727adaf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_7221de56ded18727adaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7221de56ded18727adaf",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_7221de56ded18727adaf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_04ea6484fdf4a9b14ee0": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_04ea6484fdf4a9b14ee0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_04ea6484fdf4a9b14ee0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_3808022dd402387eed16__4096_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4096 and 16384 rows: 513f4ffb, the FP32-weight twin of 3808022d.
    "cake_rmsnorm_det_train_513f4ffbfefd9d019863__ov107_4096_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 16,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 16384 and 32768 rows: 1ecf9c26, the FP32-weight twin of 628267fc.
    "cake_rmsnorm_det_train_1ecf9c260e418bc70046__ov107_16384_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_7221de56ded18727adaf__24576_32768": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_7221de56ded18727adaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_7221de56ded18727adaf",
        "role": "fwd_rows",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_a05a2262fd5fa6912442": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_a05a2262fd5fa6912442",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_a05a2262fd5fa6912442",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_a05a2262fd5fa6912442 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_67e0dfc12fbae0354e51": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 32,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_det_train_67e0dfc12fbae0354e51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_67e0dfc12fbae0354e51",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 0 and 4 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov107_0_4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__1_4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752__4_8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4 and 8 rows: c43709ca, the FP32-weight twin of 03a74dfe.
    "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov107_4_8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_007e82b24390c88156ac": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 8 and 128 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov107_8_128": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__16_128": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752__128_512": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 128,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 128 and 512 rows: c43709ca, the FP32-weight twin of 03a74dfe.
    "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov107_128_512": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 128,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_007e82b24390c88156ac__512_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 512 and 1024 rows: 58c2fce1, the FP32-weight twin of 007e82b2.
    "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__ov107_512_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03b5516039c533b453b9": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03b5516039c533b453b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03b5516039c533b453b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 1024 and 1536 rows: 62e8e275, the FP32-weight twin of 39b9b41b.
    "cake_rmsnorm_det_train_62e8e2756e015c39bf5c__ov107_1024_1536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_62e8e2756e015c39bf5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62e8e2756e015c39bf5c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__1536_2048": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 1536 and 2048 rows: 6258b6da, the FP32-weight twin of 4d9988d4.
    "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov107_1536_2048": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 2,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03a74dfe62df95c2a752__2048_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 4,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 2048 and 8192 rows: 62e8e275, the FP32-weight twin of 39b9b41b.
    "cake_rmsnorm_det_train_62e8e2756e015c39bf5c__ov107_2048_8192": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_62e8e2756e015c39bf5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_62e8e2756e015c39bf5c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_007e82b24390c88156ac__4096_12288": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 8192 and 16384 rows: 99b209a3, the FP32-weight twin of 03b55160.
    "cake_rmsnorm_det_train_99b209a37f0203af57ce__ov107_8192_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_03b5516039c533b453b9__12288_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_03b5516039c533b453b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_03b5516039c533b453b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_det_train_007e82b24390c88156ac__16384_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_det_train_007e82b24390c88156ac__16384_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__16384_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 0,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_det_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
}
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd_h1024": [
            "cake_rmsnorm_det_train_2dc3fb268514e9182c6a",
            "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
            "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac",
            "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2__128_1024",
            "cake_rmsnorm_det_train_0c581a133624ff396ab4",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__12288_24576",
            "cake_rmsnorm_det_train_4294c01f97e2459b646b",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__1024_8192",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__24576_inf",
        ],
        "bwd_w32_h1024": [
            "cake_rmsnorm_det_train_153f811937bb0157224b",
            "cake_rmsnorm_det_train_744d74299c2d45867c94__ov100_1_1024",
            "cake_rmsnorm_det_train_561bea88a25c2f245c29",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__12288_24576",
            "cake_rmsnorm_det_train_58305d0f80e3e79161d4",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__1024_8192",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__24576_inf",
        ],
        "bwd_h128": [
            "cake_rmsnorm_det_train_9346600882e53a589509",
            "cake_rmsnorm_det_train_401a018c68bc082078b2",
        ],
        "bwd_w32_h128": [
            "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e",
            "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea",
        ],
        "bwd_h1280": [
            "cake_rmsnorm_det_train_8895b0224bb3d596a3a6",
            "cake_rmsnorm_det_train_a5f2942d61e591f57a76",
            "cake_rmsnorm_det_train_809235a3965d337a6877",
            "cake_rmsnorm_det_train_12bbedff93fcaafa475d",
            "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__4096_inf",
            "cake_rmsnorm_det_train_59474336746fb32d4e51",
            "cake_rmsnorm_det_train_59474336746fb32d4e51__4096_inf",
        ],
        "bwd_w32_h1280": [
            "cake_rmsnorm_det_train_103805d93149c8668264",
            "cake_rmsnorm_det_train_8d8ef930b7d650c5538a",
            "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d",
            "cake_rmsnorm_det_train_80956d258172dfa76c2a",
            "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__4096_inf",
            "cake_rmsnorm_det_train_59474336746fb32d4e51",
            "cake_rmsnorm_det_train_59474336746fb32d4e51__4096_inf",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_det_train_f6233ff84d0b93fe45db",
            "cake_rmsnorm_det_train_4176a24f8ee981dd2bb2",
            "cake_rmsnorm_det_train_7dc29629ff21625607f8__768_1024",
            "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_det_train_21c8bb55c8d672b63952",
            "cake_rmsnorm_det_train_b457c4d134e44476dc6f",
            "cake_rmsnorm_det_train_b457c4d134e44476dc6f__36352_inf",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__768_1024",
            "cake_rmsnorm_det_train_1863320f3158d35eead6",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__32768_36352",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__36352_inf",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_det_train_5b177296918aca760fa2",
            "cake_rmsnorm_det_train_06bbfeb2e313d41451fa",
            "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__768_1024",
            "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b",
            "cake_rmsnorm_det_train_4bf450b6af0314ed03be",
            "cake_rmsnorm_det_train_8777b1c8538527c9aaf7",
            "cake_rmsnorm_det_train_8777b1c8538527c9aaf7__36352_inf",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__768_1024",
            "cake_rmsnorm_det_train_1863320f3158d35eead6",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__4096_32768",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__32768_36352",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__36352_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_det_train_b13b0c73ece62938899a",
            "cake_rmsnorm_det_train_923e668c53f60f0b256b",
            "cake_rmsnorm_det_train_cc26e316ae40d683a118__768_1024",
            "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6",
            "cake_rmsnorm_det_train_cc26e316ae40d683a118__2047_22528",
            "cake_rmsnorm_det_train_ceeef043ac64f37fc529",
            "cake_rmsnorm_det_train_ceeef043ac64f37fc529__58880_inf",
            "cake_rmsnorm_det_train_880768a0a264e04dbfb5__768_1024",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__1024_2047",
            "cake_rmsnorm_det_train_880768a0a264e04dbfb5__2047_22528",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__22528_58880",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__58880_inf",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_det_train_d22b3e8ce96e49997448",
            "cake_rmsnorm_det_train_9030dbeb948761ec2258",
            "cake_rmsnorm_det_train_07f9f2fbea7163639c27__ov100_768_1024",
            "cake_rmsnorm_det_train_b187e5989302def22b2d__ov100_1024_16384",
            "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9__ov100_16384_58880",
            "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9__58880_inf",
            "cake_rmsnorm_det_train_880768a0a264e04dbfb5__768_1024",
            "cake_rmsnorm_det_train_d0960be986e58f449f50",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__16384_58880",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__58880_inf",
        ],
        "bwd_h256": [
            "cake_rmsnorm_det_train_6959ea32516c82acae98",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__87_2048",
            "cake_rmsnorm_det_train_af005462cd6a301f31c4",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__5120_12288",
            "cake_rmsnorm_det_train_af005462cd6a301f31c4__12288_40960",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__40960_80896",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__80896_1572864",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__1572864_inf",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__87_2048",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__2048_5120",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__5120_12288",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__40960_80896",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__80896_1572864",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__1572864_inf",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_det_train_c484e2676d7b535ff396",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__87_2048",
            "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__5120_12288",
            "cake_rmsnorm_det_train_b7d9577ff884de98ea06__12288_40960",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__40960_80896",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__80896_1572864",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__1572864_inf",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__87_2048",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__2048_5120",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__5120_12288",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__40960_80896",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__80896_1572864",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__1572864_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_det_train_4822793ea0c954c11af6__0_203",
            "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9__203_256",
            "cake_rmsnorm_det_train_59270fb249dcba1ff679",
            "cake_rmsnorm_det_train_5572b2671862fe670ef4__1024_22528",
            "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a__22528_32768",
            "cake_rmsnorm_det_train_35a0b0ca6b2060313255",
            "cake_rmsnorm_det_train_35a0b0ca6b2060313255__44032_inf",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__203_256",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__1024_22528",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__22528_32768",
            "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__32768_44032",
            "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__44032_inf",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_det_train_8616070567821eaa3e35__0_203",
            "cake_rmsnorm_det_train_20c64c610d22c6db4c05__203_256",
            "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1",
            "cake_rmsnorm_det_train_47a01388444960da7aae__1024_22528",
            "cake_rmsnorm_det_train_a8eb754739a1589de3ec__22528_32768",
            "cake_rmsnorm_det_train_626aed743358c27513f1",
            "cake_rmsnorm_det_train_626aed743358c27513f1__44032_inf",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__203_256",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__1024_22528",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__22528_32768",
            "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__32768_44032",
            "cake_rmsnorm_det_train_fcbfa14d591a9babfd26__44032_inf",
        ],
        "bwd_h512": [
            "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c",
            "cake_rmsnorm_det_train_931757c844a9d257c08b__1_227",
            "cake_rmsnorm_det_train_09601d42628de24f6f87__227_1024",
            "cake_rmsnorm_det_train_123db0b2f7fb38cfab03",
            "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036",
            "cake_rmsnorm_det_train_8877322a3002e3a4fefd",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_det_train_fef03ce50a792c8704bd",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__4096_32768",
            "cake_rmsnorm_det_train_122721bd6b4650866553",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a",
            "cake_rmsnorm_det_train_404c8f54ab3129877350__1_227",
            "cake_rmsnorm_det_train_9f07cea1d13b923ed91e__227_1024",
            "cake_rmsnorm_det_train_632ca6ff01848a71f9f2",
            "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441",
            "cake_rmsnorm_det_train_659bfc53e5372850ae2e",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_det_train_fef03ce50a792c8704bd",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__4096_32768",
            "cake_rmsnorm_det_train_122721bd6b4650866553",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_det_train_080e0a6c3b18e9bad790__0_1",
            "cake_rmsnorm_det_train_dd3a35731e2e1665dd71",
            "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c",
            "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__6144_50688",
            "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c__50688_inf",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__256_6144",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__6144_50688",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__50688_inf",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_det_train_747d86f34cb58479c7ca__ov100_0_1",
            "cake_rmsnorm_det_train_747d86f34cb58479c7ca",
            "cake_rmsnorm_det_train_40227b2a2b2122628d58",
            "cake_rmsnorm_det_train_9479b1298a1b19d4d996__6144_50688",
            "cake_rmsnorm_det_train_9479b1298a1b19d4d996__50688_inf",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__256_6144",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__6144_50688",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__50688_inf",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_det_train_f2271417b1a596d0540f",
            "cake_rmsnorm_det_train_61421bc72e57d14d8a32__384_4096",
            "cake_rmsnorm_det_train_b93bf060d4d3defc8150__mid",
            "cake_rmsnorm_det_train_b93bf060d4d3defc8150__42496_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__384_4096",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__4096_42496",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__42496_inf",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5",
            "cake_rmsnorm_det_train_41b80b6af97974feb181__ov100_384_4096",
            "cake_rmsnorm_det_train_62beb52fa50cc8624c9f__mid",
            "cake_rmsnorm_det_train_62beb52fa50cc8624c9f__42496_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__384_4096",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__4096_42496",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__42496_inf",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_det_train_864e84f3d874f850c091__0_1152",
            "cake_rmsnorm_det_train_c1a3be946e94e94233af__1152_42496",
            "cake_rmsnorm_det_train_84385468c8e8ef19f3dc__42496_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__1152_42496",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__42496_inf",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_det_train_71d36617003ab7742a77__0_1152",
            "cake_rmsnorm_det_train_ab0e20feefa296ffff3c__1152_42496",
            "cake_rmsnorm_det_train_ab0e20feefa296ffff3c__42496_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__1152_42496",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__42496_inf",
        ],
        "fwd_h1024": [
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_2",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__2_4",
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__4_512",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__512_1024",
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__1024_2048",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba",
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__32768_inf",
        ],
        "fwd_w32_h1024": [
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__0_2",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__2_4",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov100_4_32",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov100_32_64",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov100_64_512",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov100_512_1024",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov100_1024_32768",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__32768_inf",
        ],
        "fwd_h128": [
            "cake_rmsnorm_det_train_3808022dd402387eed16",
            "cake_rmsnorm_det_train_628267fc724c2c8f0e07__2_4",
            "cake_rmsnorm_det_train_3808022dd402387eed16__4_16384",
            "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        ],
        "fwd_w32_h128": [
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863",
            "cake_rmsnorm_det_train_1ecf9c260e418bc70046__2_4",
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863__ov100_4_64",
            "cake_rmsnorm_det_train_1ecf9c260e418bc70046__ov100_64_128",
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863__ov100_128_16384",
            "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        ],
        "fwd_h1280": [
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5",
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752__16_32",
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__32_256",
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752__256_512",
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__512_8192",
            "cake_rmsnorm_det_train_39b9b41b16769252153d",
            "cake_rmsnorm_det_train_007e82b24390c88156ac__24576_32768",
            "cake_rmsnorm_det_train_03b5516039c533b453b9__32768_inf",
        ],
        "fwd_w32_h1280": [
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_0_1",
            "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov100_1_4",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_4_8",
            "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov100_8_16",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_16_256",
            "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov100_256_1024",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov100_1024_2048",
            "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__ov100_2048_32768",
            "cake_rmsnorm_det_train_99b209a37f0203af57ce__32768_inf",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_det_train_1ea88523077fc340e55f",
            "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_det_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_det_train_21d3416baa877808371e",
            "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41__low",
            "cake_rmsnorm_det_train_1974c5c86cee63f33370",
            "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_det_train_b44107e05796c2922187",
            "cake_rmsnorm_det_train_a5f323db12acfb2cd565",
            "cake_rmsnorm_det_train_780b9833fd0a236ef0e4__cut",
            "cake_rmsnorm_det_train_b44107e05796c2922187__band",
            "cake_rmsnorm_det_train_780b9833fd0a236ef0e4",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_det_train_0da8e18d4af432c6927a",
            "cake_rmsnorm_det_train_27fd540856828056457a",
            "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b__cut",
            "cake_rmsnorm_det_train_0da8e18d4af432c6927a__band",
            "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b",
        ],
        "fwd_h256": [
            "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8",
            "cake_rmsnorm_det_train_0489c784c13bb25499f6",
            "cake_rmsnorm_det_train_1b284baec9663d6de1b9",
            "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0",
            "cake_rmsnorm_det_train_1388ab779a35c4e41efb",
            "cake_rmsnorm_det_train_466331ff3cd2d6c62738",
            "cake_rmsnorm_det_train_ed9483d8c356affee55d",
            "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_det_train_f8087de00e0409186d93",
            "cake_rmsnorm_det_train_f2e721db4c30900f12fe",
            "cake_rmsnorm_det_train_77456435c9fbd1d18fcf",
            "cake_rmsnorm_det_train_42d82d26e97c36651dae",
            "cake_rmsnorm_det_train_b0299f6e0881fb35f27b",
            "cake_rmsnorm_det_train_32bebe5a7782311d60a0",
            "cake_rmsnorm_det_train_d0a656f053396a58c49a",
            "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_det_train_8392d1a008d58a006bba",
            "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid",
            "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_det_train_4371270e4db4b985998f",
            "cake_rmsnorm_det_train_f0db3068d0d05442a59d__low",
            "cake_rmsnorm_det_train_bdc524a233d2bc6e9571__mid",
            "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        ],
        "fwd_h512": [
            "cake_rmsnorm_det_train_8757f6da86abf54363d5",
            "cake_rmsnorm_det_train_7673978dc698790054d2",
            "cake_rmsnorm_det_train_3d20ab4a432bddf7349a__100",
            "cake_rmsnorm_det_train_3430e5d9c053309b0323__100",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_det_train_6769e393999bc2ba1926",
            "cake_rmsnorm_det_train_e2eabcb3b0915d916a13",
            "cake_rmsnorm_det_train_7da36ff557bb2bac8d89__100",
            "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57__100",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_det_train_6e12e07353014329ebad",
            "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__small",
            "cake_rmsnorm_det_train_b5e173f5ecc736d91109__mid",
            "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_det_train_c1819da87ab45ebb8d56__ov100_0_64",
            "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e__ov100_64_640",
            "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov100_640_6144",
            "cake_rmsnorm_det_train_b49de9b47372fa4e1051__ov100_6144_16384",
            "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov100_16384_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_det_train_bad4db9153cc903b7ace__100",
            "cake_rmsnorm_det_train_bad4db9153cc903b7ace__ps5",
            "cake_rmsnorm_det_train_288d7b1d51c340236cb7",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6__100",
            "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6__ps5",
            "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d__ov100_1024_6144",
            "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e__ov100_6144_inf",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_det_train_8b4730338cac98ec272d__0_256",
            "cake_rmsnorm_det_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_det_train_e646bbdbd852884953da__1024_4096",
            "cake_rmsnorm_det_train_41805c5801f6956b4ce4",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_det_train_3e55e44ce21eff7d8931__0_256",
            "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_det_train_6325c474ad9955a319e9__1024_4096",
            "cake_rmsnorm_det_train_4be7a14e78ff59fa96e2",
        ],
    },
    "sm_103a": {
        "bwd_h1024": [
            "cake_rmsnorm_det_train_2dc3fb268514e9182c6a",
            "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2",
            "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac",
            "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2__128_1024",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__1024_1536",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__1536_2048",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__2048_4096",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__4096_8192",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__12288_24576",
            "cake_rmsnorm_det_train_4294c01f97e2459b646b",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__1024_1536",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97__1536_2048",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_4096",
            "cake_rmsnorm_det_train_4830a91c2471462a29c8__4096_8192",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__24576_inf",
        ],
        "bwd_w32_h1024": [
            "cake_rmsnorm_det_train_153f811937bb0157224b",
            "cake_rmsnorm_det_train_744d74299c2d45867c94__ov103_1_1024",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_1024_1536",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_1536_2048",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_2048_16384",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__ov103_16384_24576",
            "cake_rmsnorm_det_train_58305d0f80e3e79161d4",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__1024_1536",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97__1536_2048",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_16384",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97__16384_24576",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__24576_inf",
        ],
        "bwd_h128": [
            "cake_rmsnorm_det_train_9346600882e53a589509",
            "cake_rmsnorm_det_train_401a018c68bc082078b2",
        ],
        "bwd_w32_h128": [
            "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e",
            "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea",
        ],
        "bwd_h1280": [
            "cake_rmsnorm_det_train_8895b0224bb3d596a3a6",
            "cake_rmsnorm_det_train_a5f2942d61e591f57a76",
            "cake_rmsnorm_det_train_809235a3965d337a6877",
            "cake_rmsnorm_det_train_12bbedff93fcaafa475d__512_2048",
            "cake_rmsnorm_det_train_c646e0976e08c6fe8a91",
            "cake_rmsnorm_det_train_59474336746fb32d4e51__512_2048",
            "cake_rmsnorm_det_train_59474336746fb32d4e51__2048_inf",
        ],
        "bwd_w32_h1280": [
            "cake_rmsnorm_det_train_103805d93149c8668264",
            "cake_rmsnorm_det_train_8d8ef930b7d650c5538a",
            "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d",
            "cake_rmsnorm_det_train_80956d258172dfa76c2a__ov103_512_4096",
            "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__ov103_4096_inf",
            "cake_rmsnorm_det_train_59474336746fb32d4e51",
            "cake_rmsnorm_det_train_59474336746fb32d4e51__4096_inf",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_det_train_e254714a110f4d45ea02",
            "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__1_31",
            "cake_rmsnorm_det_train_4aef8d1ff2c1c2af7b80__31_214",
            "cake_rmsnorm_det_train_7dc29629ff21625607f8__214_1024",
            "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe",
            "cake_rmsnorm_det_train_1df7db17277d363b5168",
            "cake_rmsnorm_det_train_4df08ce51fb01ba1193f",
            "cake_rmsnorm_det_train_f522f5b8d4343d96ca04",
            "cake_rmsnorm_det_train_9cd3c766db4b70ed7213",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__214_1024",
            "cake_rmsnorm_det_train_1863320f3158d35eead6",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__15566_40448",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__40448_inf",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_det_train_9f139a52a93ccc69c3d1",
            "cake_rmsnorm_det_train_698c3d36d67408394ccf__1_31",
            "cake_rmsnorm_det_train_698c3d36d67408394ccf__31_214",
            "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__214_1024",
            "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b",
            "cake_rmsnorm_det_train_a99e2b380514c4d86d60",
            "cake_rmsnorm_det_train_7a72e97bd6cb4dc0efc2",
            "cake_rmsnorm_det_train_109f19bb177c8f20cadc",
            "cake_rmsnorm_det_train_51f46ff164b142ec8888",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__214_1024",
            "cake_rmsnorm_det_train_1863320f3158d35eead6",
            "cake_rmsnorm_det_train_9b727c2c9933b2ca6724__4096_14928",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__14928_15566",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__15566_40448",
            "cake_rmsnorm_det_train_1863320f3158d35eead6__40448_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_det_train_a1c71cc355c7e9988ab2",
            "cake_rmsnorm_det_train_ed36a8d998abba41f227__1_2",
            "cake_rmsnorm_det_train_ed36a8d998abba41f227__2_171",
            "cake_rmsnorm_det_train_cc26e316ae40d683a118__171_16384",
            "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9__16384_32768",
            "cake_rmsnorm_det_train_302f64db6abe79e5259e",
            "cake_rmsnorm_det_train_302f64db6abe79e5259e__55296_inf",
            "cake_rmsnorm_det_train_880768a0a264e04dbfb5__171_16384",
            "cake_rmsnorm_det_train_596b457aaba073f8d69f",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__32768_55296",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__55296_inf",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_det_train_ab6b18375f52dcec73f7",
            "cake_rmsnorm_det_train_978741b43372951cb5b9__1_2",
            "cake_rmsnorm_det_train_978741b43372951cb5b9__2_171",
            "cake_rmsnorm_det_train_c83c314c6dcf28ad7369__171_16384",
            "cake_rmsnorm_det_train_07f9f2fbea7163639c27__16384_32768",
            "cake_rmsnorm_det_train_e98cb156abfe1666c76f",
            "cake_rmsnorm_det_train_e98cb156abfe1666c76f__55296_inf",
            "cake_rmsnorm_det_train_880768a0a264e04dbfb5__171_16384",
            "cake_rmsnorm_det_train_596b457aaba073f8d69f",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__32768_55296",
            "cake_rmsnorm_det_train_d0960be986e58f449f50__55296_inf",
        ],
        "bwd_h256": [
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__0_1680",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__1680_2048",
            "cake_rmsnorm_det_train_af005462cd6a301f31c4",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__5120_12288",
            "cake_rmsnorm_det_train_af005462cd6a301f31c4__12288_40960",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__40960_99840",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__99840_720896",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__720896_1572864",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__1572864_inf",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__0_1680",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__1680_2048",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__2048_5120",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__5120_12288",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__99840_720896",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__720896_1572864",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__1572864_inf",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__0_1680",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__1680_2048",
            "cake_rmsnorm_det_train_b7d9577ff884de98ea06",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__5120_12288",
            "cake_rmsnorm_det_train_b7d9577ff884de98ea06__12288_40960",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__40960_99840",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__99840_720896",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__720896_1572864",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__1572864_inf",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__0_1680",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__1680_2048",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__2048_5120",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__5120_12288",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__12288_40960",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__99840_720896",
            "cake_rmsnorm_det_train_affa1b95f68bab8c411b__720896_1572864",
            "cake_rmsnorm_det_train_72138dce40d43da44aaf__1572864_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_det_train_5efbf78a48136634a5b3",
            "cake_rmsnorm_det_train_5572b2671862fe670ef4__512_inf",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__512_inf",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_det_train_27615d0e36c9f2fe0856__ov103_0_1",
            "cake_rmsnorm_det_train_ff7de8d3e4dcd036ea9a__ov103_1_512",
            "cake_rmsnorm_det_train_47a01388444960da7aae__512_inf",
            "cake_rmsnorm_det_train_5d75ea5f3c25579af496__512_inf",
        ],
        "bwd_h512": [
            "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c",
            "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__1_4",
            "cake_rmsnorm_det_train_bdfd77893d8f9db3c9bb__4_227",
            "cake_rmsnorm_det_train_09601d42628de24f6f87__227_1024",
            "cake_rmsnorm_det_train_8e7f4d811e55a351a22e",
            "cake_rmsnorm_det_train_274b0055aace183f2367",
            "cake_rmsnorm_det_train_cfaae3cec98f8501832b",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_det_train_122721bd6b4650866553__4096_32768",
            "cake_rmsnorm_det_train_122721bd6b4650866553",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a",
            "cake_rmsnorm_det_train_2c60470d16e0a7fb6089__1_4",
            "cake_rmsnorm_det_train_2c60470d16e0a7fb6089__4_227",
            "cake_rmsnorm_det_train_9f07cea1d13b923ed91e__227_1024",
            "cake_rmsnorm_det_train_dfcbd32cdb8108643fba",
            "cake_rmsnorm_det_train_56ee8e66479625fd078a",
            "cake_rmsnorm_det_train_a546f410ec8cf03ef20d",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_det_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_det_train_122721bd6b4650866553__4096_32768",
            "cake_rmsnorm_det_train_122721bd6b4650866553",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_det_train_dd3a35731e2e1665dd71__0_64",
            "cake_rmsnorm_det_train_2fc4932fbde0bc848b32__64_5120",
            "cake_rmsnorm_det_train_2fc4932fbde0bc848b32__5120_inf",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__64_5120",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__5120_inf",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_det_train_747d86f34cb58479c7ca__0_64",
            "cake_rmsnorm_det_train_79811a76205d7d7dd197__64_5120",
            "cake_rmsnorm_det_train_79811a76205d7d7dd197__5120_inf",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__64_5120",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__5120_inf",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_det_train_f2271417b1a596d0540f__0_256",
            "cake_rmsnorm_det_train_61421bc72e57d14d8a32__256_4096",
            "cake_rmsnorm_det_train_322574af704b51aef0bb",
            "cake_rmsnorm_det_train_322574af704b51aef0bb__48640_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__256_4096",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__4096_48640",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__48640_inf",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_det_train_b3e6531e46667901ab91__ov103_0_1",
            "cake_rmsnorm_det_train_8a280a7ecafed4b9e9f5__ov103_1_256",
            "cake_rmsnorm_det_train_41b80b6af97974feb181__ov103_256_4096",
            "cake_rmsnorm_det_train_9a8c0396edd85d4d545e",
            "cake_rmsnorm_det_train_9a8c0396edd85d4d545e__48640_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__256_4096",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__4096_48640",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__48640_inf",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_det_train_864e84f3d874f850c091__0_1216",
            "cake_rmsnorm_det_train_04ebec87cbee3f0ce093__1216_48640",
            "cake_rmsnorm_det_train_04ebec87cbee3f0ce093__48640_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__1216_48640",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__48640_inf",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_det_train_71d36617003ab7742a77__0_1216",
            "cake_rmsnorm_det_train_47890c9ae28e3e4b6759__1216_48640",
            "cake_rmsnorm_det_train_47890c9ae28e3e4b6759__48640_inf",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__1216_48640",
            "cake_rmsnorm_det_train_5f5cdc5151e3c0620bcd__48640_inf",
        ],
        "fwd_h1024": [
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_2048",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba",
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__32768_inf",
        ],
        "fwd_w32_h1024": [
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_0_2",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov103_2_4",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_4_32",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov103_32_64",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_64_512",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov103_512_1024",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov103_1024_32768",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__32768_inf",
        ],
        "fwd_h128": [
            "cake_rmsnorm_det_train_3808022dd402387eed16__0_16384",
            "cake_rmsnorm_det_train_628267fc724c2c8f0e07",
        ],
        "fwd_w32_h128": [
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863__0_16384",
            "cake_rmsnorm_det_train_1ecf9c260e418bc70046",
        ],
        "fwd_h1280": [
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__0_256",
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752__256_1024",
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__1024_8192",
            "cake_rmsnorm_det_train_39b9b41b16769252153d",
            "cake_rmsnorm_det_train_007e82b24390c88156ac__24576_inf",
        ],
        "fwd_w32_h1280": [
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov103_0_512",
            "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov103_512_1024",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov103_1024_2048",
            "cake_rmsnorm_det_train_99b209a37f0203af57ce__ov103_2048_8192",
            "cake_rmsnorm_det_train_62e8e2756e015c39bf5c__ov103_8192_24576",
            "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__ov103_24576_32768",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov103_32768_inf",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_det_train_1ea88523077fc340e55f__0_1",
            "cake_rmsnorm_det_train_d52385477d517c951e87",
            "cake_rmsnorm_det_train_1ea88523077fc340e55f__256_1536",
            "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_det_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_det_train_21d3416baa877808371e__0_1",
            "cake_rmsnorm_det_train_8e49f86a65049578dee1",
            "cake_rmsnorm_det_train_21d3416baa877808371e__256_1536",
            "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41__low",
            "cake_rmsnorm_det_train_1974c5c86cee63f33370",
            "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_det_train_c5a8099e177e463daab9",
            "cake_rmsnorm_det_train_b44107e05796c2922187__mid",
            "cake_rmsnorm_det_train_780b9833fd0a236ef0e4",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_det_train_01f0ebf3db9fa2c0884e",
            "cake_rmsnorm_det_train_0da8e18d4af432c6927a__mid",
            "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b",
        ],
        "fwd_h256": [
            "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8__0_24",
            "cake_rmsnorm_det_train_0489c784c13bb25499f6__24_8192",
            "cake_rmsnorm_det_train_bbb05314930d192dfe7b",
            "cake_rmsnorm_det_train_1b284baec9663d6de1b9__20480_40960",
            "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0",
            "cake_rmsnorm_det_train_466331ff3cd2d6c62738__81920_114688",
            "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0__114688_196608",
            "cake_rmsnorm_det_train_ed9483d8c356affee55d",
            "cake_rmsnorm_det_train_37f663183cbff23c2893",
            "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5__368640_inf",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_det_train_f8087de00e0409186d93__0_24",
            "cake_rmsnorm_det_train_f2e721db4c30900f12fe__24_8192",
            "cake_rmsnorm_det_train_a8fffd93682c5e382dc5",
            "cake_rmsnorm_det_train_77456435c9fbd1d18fcf__20480_40960",
            "cake_rmsnorm_det_train_42d82d26e97c36651dae",
            "cake_rmsnorm_det_train_32bebe5a7782311d60a0__81920_114688",
            "cake_rmsnorm_det_train_42d82d26e97c36651dae__114688_196608",
            "cake_rmsnorm_det_train_d0a656f053396a58c49a",
            "cake_rmsnorm_det_train_18e3c0a5204006b3a56f",
            "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461__368640_inf",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_det_train_8392d1a008d58a006bba",
            "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid103",
            "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__wide103",
            "cake_rmsnorm_det_train_fd913ecd254a6bba4b18",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_det_train_4371270e4db4b985998f",
            "cake_rmsnorm_det_train_f0db3068d0d05442a59d__low",
            "cake_rmsnorm_det_train_bdc524a233d2bc6e9571__mid103",
            "cake_rmsnorm_det_train_bdc524a233d2bc6e9571__wide103",
            "cake_rmsnorm_det_train_f0db3068d0d05442a59d",
        ],
        "fwd_h512": [
            "cake_rmsnorm_det_train_8757f6da86abf54363d5__0_4",
            "cake_rmsnorm_det_train_def532441f08b90a9903",
            "cake_rmsnorm_det_train_7673978dc698790054d2__1025_4096",
            "cake_rmsnorm_det_train_3d20ab4a432bddf7349a",
            "cake_rmsnorm_det_train_3430e5d9c053309b0323",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_det_train_6769e393999bc2ba1926__0_4",
            "cake_rmsnorm_det_train_20ebc686cc4099b9d026",
            "cake_rmsnorm_det_train_e2eabcb3b0915d916a13__1025_4096",
            "cake_rmsnorm_det_train_7da36ff557bb2bac8d89",
            "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_det_train_6e12e07353014329ebad",
            "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__small",
            "cake_rmsnorm_det_train_b5e173f5ecc736d91109__mid",
            "cake_rmsnorm_det_train_5eef28cd35498e9abfa5",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_det_train_c1819da87ab45ebb8d56__ov103_0_64",
            "cake_rmsnorm_det_train_1defffdf2712cae36784__ov103_64_640",
            "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov103_640_6144",
            "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__ov103_6144_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_det_train_288d7b1d51c340236cb7",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
            "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6__ov103_1024_6144",
            "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e__ov103_6144_inf",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_det_train_8b4730338cac98ec272d",
            "cake_rmsnorm_det_train_e646bbdbd852884953da__4096_inf",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_det_train_3e55e44ce21eff7d8931",
            "cake_rmsnorm_det_train_6325c474ad9955a319e9__4096_inf",
        ],
    },
    "sm_107a": {
        "bwd_h1024": [
            "cake_rmsnorm_det_train_2dc3fb268514e9182c6a",
            "cake_rmsnorm_det_train_5ad5f0962d5c5e8f63ac__1_8",
            "cake_rmsnorm_det_train_89fa3e964f6ac26dc6a2__8_1536",
            "cake_rmsnorm_det_train_0c581a133624ff396ab4__1536_2048",
            "cake_rmsnorm_det_train_5e32dd889d72a75e348b__2048_24576",
            "cake_rmsnorm_det_train_4294c01f97e2459b646b",
            "cake_rmsnorm_det_train_4830a91c2471462a29c8",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_24576",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97__24576_inf",
        ],
        "bwd_w32_h1024": [
            "cake_rmsnorm_det_train_153f811937bb0157224b",
            "cake_rmsnorm_det_train_cfd067433e487c414dbb__1_8",
            "cake_rmsnorm_det_train_744d74299c2d45867c94__8_1536",
            "cake_rmsnorm_det_train_561bea88a25c2f245c29__1536_2048",
            "cake_rmsnorm_det_train_ba8feeec37d2bdde1fca__2048_24576",
            "cake_rmsnorm_det_train_58305d0f80e3e79161d4",
            "cake_rmsnorm_det_train_4830a91c2471462a29c8",
            "cake_rmsnorm_det_train_3b5dc80e2313fd64010e__2048_24576",
            "cake_rmsnorm_det_train_e21352eeed287fc98d97__24576_inf",
        ],
        "bwd_h128": [
            "cake_rmsnorm_det_train_9346600882e53a589509",
            "cake_rmsnorm_det_train_401a018c68bc082078b2",
        ],
        "bwd_w32_h128": [
            "cake_rmsnorm_det_train_ab8a320c36b909e7bf1e",
            "cake_rmsnorm_det_train_b5f8c6c9e6ae65a17cea",
        ],
        "bwd_h1280": [
            "cake_rmsnorm_det_train_8895b0224bb3d596a3a6",
            "cake_rmsnorm_det_train_a5f2942d61e591f57a76",
            "cake_rmsnorm_det_train_809235a3965d337a6877",
            "cake_rmsnorm_det_train_12bbedff93fcaafa475d__512_8192",
            "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__8192_16384",
            "cake_rmsnorm_det_train_c646e0976e08c6fe8a91__16384_inf",
            "cake_rmsnorm_det_train_849ee5d846ddc145e3a9",
            "cake_rmsnorm_det_train_382c8727d8b150884d7f__8192_16384",
            "cake_rmsnorm_det_train_55c3530541ac1903b775",
        ],
        "bwd_w32_h1280": [
            "cake_rmsnorm_det_train_103805d93149c8668264",
            "cake_rmsnorm_det_train_8d8ef930b7d650c5538a",
            "cake_rmsnorm_det_train_e476d0b2c0bd4285f16d",
            "cake_rmsnorm_det_train_80956d258172dfa76c2a__ov107_512_4096",
            "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__ov107_4096_16384",
            "cake_rmsnorm_det_train_c66b8cbd5626e2c1f2b4__16384_inf",
            "cake_rmsnorm_det_train_849ee5d846ddc145e3a9__512_4096",
            "cake_rmsnorm_det_train_382c8727d8b150884d7f__4096_16384",
            "cake_rmsnorm_det_train_55c3530541ac1903b775",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_det_train_c3b1b46a2c6d1f9b3724",
            "cake_rmsnorm_det_train_1207b05663ed2b4de14b__1_576",
            "cake_rmsnorm_det_train_7dc29629ff21625607f8__576_1024",
            "cake_rmsnorm_det_train_88510c609fd688aab969",
            "cake_rmsnorm_det_train_6a428da2227972ae18ae",
            "cake_rmsnorm_det_train_56d26996239835327501",
            "cake_rmsnorm_det_train_411e270c0575d26a77f4",
            "cake_rmsnorm_det_train_628248febbdfe17c3325__576_1024",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d__12288_16384",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d__16384_45056",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d__45056_inf",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_det_train_45057834f9e0e017b6ef",
            "cake_rmsnorm_det_train_137ba5328176fb07b37d__1_576",
            "cake_rmsnorm_det_train_baff9ecbafe7de1e6030__576_1024",
            "cake_rmsnorm_det_train_6f2c9e90f551f35e99e6",
            "cake_rmsnorm_det_train_e63beae98c236ce26335",
            "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4",
            "cake_rmsnorm_det_train_64ffe19b041f5bfa34fb",
            "cake_rmsnorm_det_train_628248febbdfe17c3325__576_1024",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d__12288_16384",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d__16384_45056",
            "cake_rmsnorm_det_train_db5c0b673c516bd0295d__45056_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_det_train_92def3f97ddcec9165f0",
            "cake_rmsnorm_det_train_6720b4a159be2b84b325__1_256",
            "cake_rmsnorm_det_train_845cdcd8145ada631461__256_1024",
            "cake_rmsnorm_det_train_cc26e316ae40d683a118__1024_16384_107",
            "cake_rmsnorm_det_train_845cdcd8145ada631461",
            "cake_rmsnorm_det_train_21ee6fc93139c5e0cd41",
            "cake_rmsnorm_det_train_271082c56c222bf10d77",
            "cake_rmsnorm_det_train_c4d8be677fa70b841877",
            "cake_rmsnorm_det_train_c4d8be677fa70b841877__16384_50688",
            "cake_rmsnorm_det_train_c4d8be677fa70b841877__50688_inf",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_det_train_5e3034bbb4032d8b7ad9",
            "cake_rmsnorm_det_train_23c900134dc889302136__1_256",
            "cake_rmsnorm_det_train_838a9a4054731f7942af__256_1024",
            "cake_rmsnorm_det_train_c83c314c6dcf28ad7369__1024_16384_107",
            "cake_rmsnorm_det_train_838a9a4054731f7942af",
            "cake_rmsnorm_det_train_3f270477a248188afe2a",
            "cake_rmsnorm_det_train_271082c56c222bf10d77",
            "cake_rmsnorm_det_train_c4d8be677fa70b841877",
            "cake_rmsnorm_det_train_c4d8be677fa70b841877__16384_50688",
            "cake_rmsnorm_det_train_c4d8be677fa70b841877__50688_inf",
        ],
        "bwd_h256": [
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__0_1572864",
            "cake_rmsnorm_det_train_a385ae1681a2c078344e__1572864_inf",
            "cake_rmsnorm_det_train_312bbe084eac69d858c4__0_1572864",
            "cake_rmsnorm_det_train_312bbe084eac69d858c4__1572864_inf",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__0_1572864",
            "cake_rmsnorm_det_train_420788e663a2c3a810f6__1572864_inf",
            "cake_rmsnorm_det_train_312bbe084eac69d858c4__0_1572864",
            "cake_rmsnorm_det_train_312bbe084eac69d858c4__1572864_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_det_train_c1de7e41da4550cca0bc__0_256",
            "cake_rmsnorm_det_train_a647fccc3d9fa9134733",
            "cake_rmsnorm_det_train_d41a514f5210e9adc39f",
            "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf",
            "cake_rmsnorm_det_train_a647fccc3d9fa9134733__24064_50176",
            "cake_rmsnorm_det_train_e9e21bbe4fa478075868__50176_inf",
            "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_det_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_det_train_10e24516314c244771f4__4096_24064",
            "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf__24064_50176",
            "cake_rmsnorm_det_train_10e24516314c244771f4__50176_inf",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_det_train_30ec8bb68cf4fe0c817b__0_256",
            "cake_rmsnorm_det_train_dc439f8335327e482025",
            "cake_rmsnorm_det_train_b08633208e60b8ce94fd",
            "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c",
            "cake_rmsnorm_det_train_0ff440daa6f9f304e544__ov107_24064_50176",
            "cake_rmsnorm_det_train_0ff440daa6f9f304e544__50176_inf",
            "cake_rmsnorm_det_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_det_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_det_train_10e24516314c244771f4__4096_24064",
            "cake_rmsnorm_det_train_10e24516314c244771f4__24064_50176",
            "cake_rmsnorm_det_train_10e24516314c244771f4__50176_inf",
        ],
        "bwd_h512": [
            "cake_rmsnorm_det_train_f283bf47d2e3a49bc43c",
            "cake_rmsnorm_det_train_b143f616f21eeeaa021d",
            "cake_rmsnorm_det_train_3feed5026e1156d2fba2__768_1024",
            "cake_rmsnorm_det_train_3feed5026e1156d2fba2__1024_4096",
            "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd",
            "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3",
            "cake_rmsnorm_det_train_f94861c36a14eb75e39c__768_1024",
            "cake_rmsnorm_det_train_d75459d901441384398b",
            "cake_rmsnorm_det_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_det_train_8f032213ba6eeb5d959f__16384_inf",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_det_train_78bf5eb3ee33f8dc220a",
            "cake_rmsnorm_det_train_740c03baae3da1ceb102",
            "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b__768_1024",
            "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b__1024_4096",
            "cake_rmsnorm_det_train_383f709cd5c4fadd9436",
            "cake_rmsnorm_det_train_03efa681eb38f59b00c2",
            "cake_rmsnorm_det_train_f94861c36a14eb75e39c__768_1024",
            "cake_rmsnorm_det_train_d75459d901441384398b",
            "cake_rmsnorm_det_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_det_train_8f032213ba6eeb5d959f__16384_inf",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_det_train_080e0a6c3b18e9bad790__0_1",
            "cake_rmsnorm_det_train_dd3a35731e2e1665dd71__1_256",
            "cake_rmsnorm_det_train_080e0a6c3b18e9bad790__256_1280",
            "cake_rmsnorm_det_train_838c077df3a6d35d55b7__1280_4096",
            "cake_rmsnorm_det_train_6816fb80e97380f1bcbc",
            "cake_rmsnorm_det_train_47da668e7e96652f840b",
            "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__1280_4096",
            "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__19968_inf",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_det_train_139d162a8bb29d1b2d35__0_1",
            "cake_rmsnorm_det_train_747d86f34cb58479c7ca__1_256",
            "cake_rmsnorm_det_train_139d162a8bb29d1b2d35__256_1280",
            "cake_rmsnorm_det_train_cd612de56c0985bb4caa__1280_4096",
            "cake_rmsnorm_det_train_468cc4d1b8128187b4b3",
            "cake_rmsnorm_det_train_de0cf457cd240f9c6422",
            "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__1280_4096",
            "cake_rmsnorm_det_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_det_train_c5831b29d17867842e5f__19968_inf",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_det_train_0dafdf094615d080fcc9__0_1",
            "cake_rmsnorm_det_train_dffe53d5aae62879b794",
            "cake_rmsnorm_det_train_0dafdf094615d080fcc9__256_1664",
            "cake_rmsnorm_det_train_5d292172df772adb17ae__1664_19456",
            "cake_rmsnorm_det_train_9cc51adfe9826056658b",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__1664_19456",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__19456_inf",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_det_train_5cca2d0fa4553f992406__0_1",
            "cake_rmsnorm_det_train_276474ee04bad8d3d764",
            "cake_rmsnorm_det_train_5cca2d0fa4553f992406__256_1664",
            "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb__1664_19456",
            "cake_rmsnorm_det_train_bdfba721a43e0d093668",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__1664_19456",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__19456_inf",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_det_train_ee1be07a89a0482de45e__0_368",
            "cake_rmsnorm_det_train_36cad444bc992ccc732d__368_19456",
            "cake_rmsnorm_det_train_2f1499999535c6983213__19456_inf",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__368_19456",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__19456_inf",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_det_train_3bbd4f848b4847bff893__0_368",
            "cake_rmsnorm_det_train_b9d848cdbcfaeea0325c__368_19456",
            "cake_rmsnorm_det_train_56d25337feb636c80b9a__19456_inf",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__368_19456",
            "cake_rmsnorm_det_train_fb3f2a96e954f0e241bc__19456_inf",
        ],
        "fwd_h1024": [
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__0_8",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__8_16",
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__16_256",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__256_1024",
            "cake_rmsnorm_det_train_6e5ca1f45eefb79bee7e__1024_4096",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__4096_8192",
            "cake_rmsnorm_det_train_cffbc5c8de9c874b207c",
            "cake_rmsnorm_det_train_9e8a70075f28be558bba__12288_32768",
            "cake_rmsnorm_det_train_cffbc5c8de9c874b207c__32768_inf",
        ],
        "fwd_w32_h1024": [
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__0_8",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__8_16",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov107_16_512",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov107_512_1536",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov107_1536_4096",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov107_4096_8192",
            "cake_rmsnorm_det_train_f5110367098beea13130__ov107_8192_12288",
            "cake_rmsnorm_det_train_364ed18ca31e5b754d4e__ov107_12288_16384",
            "cake_rmsnorm_det_train_8bd2d83654da4780c48a__ov107_16384_24576",
            "cake_rmsnorm_det_train_d99009bfce4043a5dffe__ov107_24576_32768",
            "cake_rmsnorm_det_train_008fb179743126fb2d35__ov107_32768_inf",
        ],
        "fwd_h128": [
            "cake_rmsnorm_det_train_3808022dd402387eed16__0_32",
            "cake_rmsnorm_det_train_628267fc724c2c8f0e07__32_64",
            "cake_rmsnorm_det_train_3808022dd402387eed16__64_2048",
            "cake_rmsnorm_det_train_7221de56ded18727adaf",
            "cake_rmsnorm_det_train_3808022dd402387eed16__4096_24576",
            "cake_rmsnorm_det_train_7221de56ded18727adaf__24576_32768",
            "cake_rmsnorm_det_train_a05a2262fd5fa6912442",
        ],
        "fwd_w32_h128": [
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863__0_32",
            "cake_rmsnorm_det_train_1ecf9c260e418bc70046__32_64",
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863__64_2048",
            "cake_rmsnorm_det_train_04ea6484fdf4a9b14ee0",
            "cake_rmsnorm_det_train_513f4ffbfefd9d019863__ov107_4096_16384",
            "cake_rmsnorm_det_train_1ecf9c260e418bc70046__ov107_16384_32768",
            "cake_rmsnorm_det_train_67e0dfc12fbae0354e51",
        ],
        "fwd_h1280": [
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752",
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__1_4",
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752__4_8",
            "cake_rmsnorm_det_train_007e82b24390c88156ac",
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__16_128",
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752__128_512",
            "cake_rmsnorm_det_train_007e82b24390c88156ac__512_1024",
            "cake_rmsnorm_det_train_03b5516039c533b453b9",
            "cake_rmsnorm_det_train_4d9988d4680c7c3c02a5__1536_2048",
            "cake_rmsnorm_det_train_03a74dfe62df95c2a752__2048_4096",
            "cake_rmsnorm_det_train_007e82b24390c88156ac__4096_12288",
            "cake_rmsnorm_det_train_03b5516039c533b453b9__12288_16384",
            "cake_rmsnorm_det_train_007e82b24390c88156ac__16384_inf",
        ],
        "fwd_w32_h1280": [
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov107_0_4",
            "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov107_4_8",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov107_8_128",
            "cake_rmsnorm_det_train_c43709cadf31ed04033c__ov107_128_512",
            "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__ov107_512_1024",
            "cake_rmsnorm_det_train_62e8e2756e015c39bf5c__ov107_1024_1536",
            "cake_rmsnorm_det_train_6258b6dac6840cc6bd58__ov107_1536_2048",
            "cake_rmsnorm_det_train_62e8e2756e015c39bf5c__ov107_2048_8192",
            "cake_rmsnorm_det_train_99b209a37f0203af57ce__ov107_8192_16384",
            "cake_rmsnorm_det_train_58c2fce1ea0b40758e91__16384_inf",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_det_train_1ea88523077fc340e55f__107",
            "cake_rmsnorm_det_train_4cdc4c495b3f59ff5069",
            "cake_rmsnorm_det_train_682675bda578f5b7a312",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_det_train_21d3416baa877808371e__107",
            "cake_rmsnorm_det_train_92cf7b87f1ebffb1b12b",
            "cake_rmsnorm_det_train_a30e5a3b69b7c183992f",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_det_train_b44107e05796c2922187__small",
            "cake_rmsnorm_det_train_a4da85136fba4588239f",
            "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d",
            "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d__wide",
            "cake_rmsnorm_det_train_bde56bd5a68c493f07a2",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_det_train_0da8e18d4af432c6927a__small",
            "cake_rmsnorm_det_train_a72c6ece1900f748eaa2",
            "cake_rmsnorm_det_train_595f506004038a65cc63",
            "cake_rmsnorm_det_train_40210950f2e7ee48bc3e__ov107_12288_16384",
            "cake_rmsnorm_det_train_40210950f2e7ee48bc3e",
        ],
        "fwd_h256": [
            "cake_rmsnorm_det_train_c5ca409f44e169148301",
            "cake_rmsnorm_det_train_bbb05314930d192dfe7b__6144_32768",
            "cake_rmsnorm_det_train_466331ff3cd2d6c62738__32768_98304",
            "cake_rmsnorm_det_train_37f663183cbff23c2893__98304_inf",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_det_train_8f2756937eaac6b02502",
            "cake_rmsnorm_det_train_a8fffd93682c5e382dc5__6144_32768",
            "cake_rmsnorm_det_train_32bebe5a7782311d60a0__32768_98304",
            "cake_rmsnorm_det_train_18e3c0a5204006b3a56f__98304_inf",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_det_train_8392d1a008d58a006bba__107",
            "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__low107",
            "cake_rmsnorm_det_train_d447b7f6398b60fe5daa__mid107",
            "cake_rmsnorm_det_train_bf933e1660cab8e5787b",
            "cake_rmsnorm_det_train_bf933e1660cab8e5787b__wide",
            "cake_rmsnorm_det_train_fd913ecd254a6bba4b18__107",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_det_train_4371270e4db4b985998f__107",
            "cake_rmsnorm_det_train_f0db3068d0d05442a59d__low107",
            "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_1024_4096",
            "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_4096_20480",
            "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_20480_49152",
            "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794__ov107_49152_inf",
        ],
        "fwd_h512": [
            "cake_rmsnorm_det_train_7673978dc698790054d2__quad",
            "cake_rmsnorm_det_train_2a42e5c02b240f8709b1",
            "cake_rmsnorm_det_train_60f47c5d00570ecf7740",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_det_train_e2eabcb3b0915d916a13__quad",
            "cake_rmsnorm_det_train_79fecc7016e44e940733",
            "cake_rmsnorm_det_train_61accbcc02891615f286",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_det_train_fd1097c45c1e328871fd__107",
            "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__low107",
            "cake_rmsnorm_det_train_b5e173f5ecc736d91109__mid107",
            "cake_rmsnorm_det_train_044d4a2aa0553c16db06__large107",
            "cake_rmsnorm_det_train_044d4a2aa0553c16db06__wide107",
            "cake_rmsnorm_det_train_5eef28cd35498e9abfa5__107",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e__107",
            "cake_rmsnorm_det_train_1defffdf2712cae36784__low107",
            "cake_rmsnorm_det_train_9ecb62c2640d571e50a8__mid107",
            "cake_rmsnorm_det_train_b49de9b47372fa4e1051__large107",
            "cake_rmsnorm_det_train_b49de9b47372fa4e1051__wide107",
            "cake_rmsnorm_det_train_b49de9b47372fa4e1051__ov107_49152_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_det_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_det_train_cf1d05546951bdbd9f5d",
            "cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6",
            "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d",
            "cake_rmsnorm_det_train_fdfa500c83d95df0f5c1",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_det_train_e646bbdbd852884953da__small",
            "cake_rmsnorm_det_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_det_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_det_train_6325c474ad9955a319e9__small",
            "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_det_train_6325c474ad9955a319e9",
        ],
    },
}


# The kernel-parameter plan of a record is one of three shapes -- the forward, the main backward and the
# dw reduce kernel -- shared by every record of that shape (``_CLASSES``); the shape follows from the
# record's role and workspace rule, so the records above carry only what differs between them.
def _plan_class(record: dict[str, Any]) -> str:
    if record["role"] == "bwd_reduce":
        return "bwd_reduce"
    if record["workspace_rule"] is not None:
        return "bwd_main"
    return "fwd"


for _record in MODULES.values():
    _record.update(_CLASSES[_plan_class(_record)])


HIDDEN_SIZES: tuple[int, ...] = (
    128,
    256,
    512,
    1024,
    1280,
    1536,
    2048,
    4096,
    5120,
    6144,
)

# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
KINDS = ("fwd", "bwd")
# Row-strided BF16 inputs are read with 16-byte vector accesses; the forward
# launcher additionally requires 32-byte aligned row storage (its 256-bit load
# configuration), the backward 16 bytes.  Row strides are multiples of eight
# elements.
ROW_ALIGNMENT_ELEMENTS = 8
FORWARD_ALIGNMENT_BYTES = 32
BACKWARD_ALIGNMENT_BYTES = 16
# The weight dtypes a route may ship for: BF16 (every route) and FP32 (the
# ``_w32`` routes: the effective weight ``1 + w`` of a zero-centred model,
# computed by the caller in FP32).  The FP32-weight programs load ``w`` with
# 32-byte vectors in both directions, so the host reads an FP32 ``w`` in place
# when contiguous and 32-byte aligned and copies it once otherwise (a BF16
# ``w`` keeps the forward's 32-byte and the backward's 16-byte rule).
WEIGHT_DTYPES = ("bf16", "f32")
FP32_WEIGHT_ALIGNMENT_BYTES = 32
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "y",
    "r",
    "u",
    "h_new",
    "g",
    "g_h",
    "dx",
    "dw",
    "partial",
    "counters",
)
LOGICAL_SCALARS = (
    "rows",
    "eps",
    "x_stride",
    "u_stride",
    "g_stride",
    "g_h_stride",
    "n_chunks",
    "slots",
)


def route_key(hidden: int, kind: str, residual: bool, w_dtype: str = "bf16") -> str:
    """Name the route of one public call: ``<kind>[_w32][_residual]_h<hidden>``."""

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if w_dtype not in WEIGHT_DTYPES:
        raise ValueError(f"w_dtype must be one of {WEIGHT_DTYPES}, got {w_dtype!r}")
    w32 = "_w32" if w_dtype == "f32" else ""
    return f"{kind}{w32}{'_residual' if residual else ''}_h{int(hidden)}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def supported_hidden_sizes(arch: str) -> tuple[int, ...]:
    """Hidden sizes with a complete fwd + bwd route on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        hidden
        for hidden in HIDDEN_SIZES
        if all(route_key(hidden, kind, False) in routes for kind in KINDS)
    )


def weight_dtypes(arch: str, hidden: int, kind: str, residual: bool) -> tuple[str, ...]:
    """The weight dtypes (tokens of ``WEIGHT_DTYPES``) the route ships on ``arch``."""

    routes = ROUTES.get(arch, {})
    return tuple(
        w_dtype
        for w_dtype in WEIGHT_DTYPES
        if route_key(hidden, kind, residual, w_dtype) in routes
    )


def route_applies(
    *,
    device_capability: Sequence[int],
    hidden: int,
    kind: str,
    residual: bool,
    w_dtype: str = "bf16",
) -> bool:
    """Whether the verified export owns one training call."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return route_key(hidden, kind, residual, w_dtype) in ROUTES.get(arch, {})


def route_modules(
    arch: str, hidden: int, kind: str, residual: bool, w_dtype: str = "bf16"
) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch)."""

    key = route_key(hidden, kind, residual, w_dtype)
    names = ROUTES.get(arch, {}).get(key)
    if not names:
        raise ValueError(
            f"unsupported Cake RMSNorm training route {key!r} on {arch!r}; exported "
            f"hidden sizes on this architecture: {supported_hidden_sizes(arch)}"
        )
    return tuple(names)


def _selects(rule: Optional[dict[str, Any]], rows: int) -> bool:
    if rule is None:
        return True
    kind = rule["kind"]
    if kind == "rows_between":
        at_most = rule["at_most"]
        return rows > int(rule["above"]) and (at_most is None or rows <= int(at_most))
    raise ValueError(f"unknown Cake RMSNorm training selection rule: {kind!r}")


def selected_modules(
    arch: str,
    hidden: int,
    kind: str,
    residual: bool,
    *,
    rows: int,
    w_dtype: str = "bf16",
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``rows`` tokens.

    A route may carry several programs of one kernel: the forward keeps
    latency programs for small and mid row counts next to its throughput
    program and switches on the row count.  Each module's recorded
    ``select_rule`` names the row range it owns (``rows_between``: ``above <
    rows <= at_most``, ``at_most`` ``None`` = unbounded; ``None`` = every call)
    and exactly the modules whose rule admits ``rows`` are launched, so the
    exported call runs the same program the source launcher selects.
    """

    rows = int(rows)
    names = tuple(
        name
        for name in route_modules(arch, hidden, kind, residual, w_dtype)
        if _selects(MODULES[name]["select_rule"], rows)
    )
    if not names:
        raise ValueError(
            f"no exported module of the {route_key(hidden, kind, residual, w_dtype)!r} "
            f"route on {arch!r} admits {rows} rows"
        )
    return names


def program_windows(
    arch: str, hidden: int, kind: str, residual: bool, w_dtype: str = "bf16"
) -> tuple[tuple[int, Optional[int], str], ...]:
    """The row-count windows of one route and the program each launches.

    One ``(above, at_most, library)`` per main program of the route (the dw
    reduce companions share their owner's window), from the smallest row
    counts up: the call with ``above < rows <= at_most`` (``at_most`` ``None``
    = unbounded) launches ``library``.  The windows tile every positive row
    count exactly once; a route whose records do not is refused.  Within one
    window a row's outputs do not depend on the token count (see the package
    docstrings); the edges are where they may change.
    """

    windows: list[tuple[int, Optional[int], str]] = []
    for name in route_modules(arch, hidden, kind, residual, w_dtype):
        record = MODULES[name]
        if record["role"] == "bwd_reduce":
            continue
        rule = record["select_rule"]
        if rule is None:
            windows.append((0, None, record["library"]))
            continue
        if rule["kind"] != "rows_between":
            raise ValueError(
                f"unknown Cake RMSNorm training selection rule: {rule['kind']!r}"
            )
        at_most = rule["at_most"]
        windows.append(
            (
                int(rule["above"]),
                None if at_most is None else int(at_most),
                record["library"],
            )
        )
    windows.sort(key=lambda window: window[0])
    key = route_key(hidden, kind, residual, w_dtype)
    above = 0
    for index, (low, high, _library) in enumerate(windows):
        last = index == len(windows) - 1
        if low != above or (high is None) != last or (high is not None and high <= low):
            raise ValueError(
                f"the windows of route {key!r} on {arch!r} do not tile the row counts: {windows}"
            )
        above = high if high is not None else above
    return tuple(windows)


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // int(multiple)) * int(multiple)


# The backward workspace starts with its ``uint32`` counter table (192 words at
# most, padded to four 256-byte lines); the FP32 partial rows follow.  The
# counters sit at a fixed offset of every workspace, so one buffer serves calls
# of different token counts (whose partial regions differ in size) without a
# host write between them: the main kernel resets the counter words it uses.
# A column-task kernel (workspace rule ``claim_sum_f32`` with ``sum_bytes`` 0)
# uses the table's first word for its tickets and resets it too; it keeps no
# sums and binds no partial rows.
COUNTERS_BYTES = 1024
# The workspace rule kinds of the backward: the main kernel of a pair (its FP32
# partial rows, folded by the dw reduce launched after it), and the column-task
# kernel, which runs alone and stores ``dw`` itself.
WORKSPACE_KINDS = ("chunk_partials", "claim_sum_f32")


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


# The chunk table of the main backward kernel.  TAIL_DIV and MAX_HALVING are
# the kernel helpers' own constants (a change there is a kernel text change and
# a change here): the halving stops once the next chunk would be at most
# ceil(share / tail_div) pipeline stages, and after at most max_halving halving
# rounds.  They are the default taper; a kernel built with a taper of its own
# records it under its grid rule ("tail_div" / "max_halving", each within the
# constant it is named after), and the host cuts that record's table with it
# (chunk_taper): the chunk count it passes must be the count the kernel wrote
# partial rows for.
TAIL_DIV = 12
MAX_HALVING = 5
# Rounds a table can have (the workspace bound): halving round k (k = 0, 1, ...) has chunks of
# ceil(share / 2 ** (k + 1)) rows and runs only while that exceeds
# ceil(share / TAIL_DIV), which no k with 2 ** (k + 1) >= TAIL_DIV does; the
# final equalised round adds at most ``slots`` chunks in its two parts.  At TAIL_DIV = 12
# that is k = 0, 1, 2 (2 ** 3 = 8 is below the divisor, 2 ** 4 = 16 is not): at most three
# halving rounds are ever cut, and with the final round a table holds at most 4 * slots chunks
# (MAX_HALVING = 5 does not bind), each of at least one stage of >= 1 row: n_chunks <= min(T, 4 * slots).
# A record's taper within the constants cuts no more halving rounds (a smaller
# divisor stops the halving sooner, a smaller cap caps it lower), so the bound
# holds the table of every record.
MAX_ROUNDS = min(MAX_HALVING, max(0, (TAIL_DIV - 1).bit_length() - 1)) + 1


def chunk_rounds(
    T: int,
    slots: int,
    rpi: int,
    tail_div: int = TAIL_DIV,
    max_halving: int = MAX_HALVING,
) -> list[tuple[int, int]]:
    """``[(count, rows)]`` of the backward chunk table of ``T`` tokens, in claim order.

    Cut in pipeline stages of ``rpi`` rows: with ``S = ceil(T / rpi)`` stages,
    ``sl = clamp(slots, 1, S)`` and ``share = ceil(S / sl)`` stages per slot,
    halving rounds of ``sl`` chunks of ``ceil(share / 2)``, ``ceil(share / 4)``,
    ... stages while the size is above ``max(1, ceil(share / tail_div))``, the
    round fits the stages left and fewer than ``max_halving`` ran, then one
    final round spreading the stages left evenly over ``sl`` chunks (the first
    ``m`` one stage longer).  ``rows`` is ``stages x rpi``; :func:`chunk_table`
    ends the last chunk at ``T``.  Rounds of zero stages are not emitted; with
    ``rpi = 1`` the stages are rows.  ``tail_div`` / ``max_halving`` are the
    kernel's taper: the module constants unless its record spells its own
    (:func:`chunk_taper`).  The main backward kernel's CTAs claim the chunks
    from a counter in this order and the kernels compute the same table in
    integer arithmetic from ``T``, ``slots``, their rows per step and their
    taper, so the FP32 partial row of every chunk, and the deterministic ``dw``
    the reduce kernel folds from them in chunk order, are pure functions of
    ``(T, slots, rpi, tail_div, max_halving)`` -- never of scheduling.
    """

    if tail_div < 1 or max_halving < 0:
        raise ValueError(
            f"tail_div must be >= 1 and max_halving >= 0, got {tail_div} / {max_halving}"
        )
    S = -(-T // rpi)
    sl = max(1, min(slots, S))
    share = -(-S // sl)
    tail = max(1, -(-share // tail_div))
    rounds: list[tuple[int, int]] = []
    rem, s = S, -(-share // 2)
    while len(rounds) < max_halving and s > tail:
        if sl * s > rem:
            break
        rounds.append((sl, s * rpi))
        rem -= sl * s
        s = -(-s // 2)
    q, m = divmod(rem, sl)
    if m:
        rounds.append((m, (q + 1) * rpi))
    if q:
        rounds.append((sl - m, q * rpi))
    return rounds


def chunk_table(
    T: int,
    slots: int,
    rpi: int,
    tail_div: int = TAIL_DIV,
    max_halving: int = MAX_HALVING,
) -> list[tuple[int, int]]:
    """``[(begin, end)]`` of every chunk in claim order: covers ``[0, T)`` exactly, no overlap."""

    table, b = [], 0
    for count, rows in chunk_rounds(T, slots, rpi, tail_div, max_halving):
        for _ in range(count):
            e = min(b + rows, T)
            table.append((b, e))
            b = e
    return table


@functools.lru_cache(maxsize=4096)
def n_chunks(
    T: int,
    slots: int,
    rpi: int,
    tail_div: int = TAIL_DIV,
    max_halving: int = MAX_HALVING,
) -> int:
    """Chunks of the backward chunk table of ``T`` tokens (see :func:`chunk_rounds`).

    Cached per ``(T, slots, rpi, tail_div, max_halving)``: a training step
    repeats its token counts, and the per-call host path is this lookup, the
    main launch (``slots`` its twelfth argument) and the reduce launch
    (``n_chunks``).
    """

    return len(chunk_table(T, slots, rpi, tail_div, max_halving))


def chunk_taper(rule: dict[str, Any]) -> tuple[int, int]:
    """``(tail_div, max_halving)`` of a chunk-claiming grid rule: the taper its
    kernel was built with, which the host cuts the record's table with.

    Every ``claimed_chunks`` record spells both (the module constants for a
    kernel at the default taper), each an integer within the constant it is
    named after -- ``1 <= tail_div <= TAIL_DIV`` and ``0 <= max_halving <=
    MAX_HALVING`` -- so :func:`chunk_capacity` bounds its table.  A rule that
    lacks one or exceeds a bound is refused rather than cut at the defaults:
    that would pass the reduce another chunk count than the main kernel wrote
    partial rows for.
    """

    taper = []
    for name, bound, floor in (
        ("tail_div", TAIL_DIV, 1),
        ("max_halving", MAX_HALVING, 0),
    ):
        value = rule.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"grid rule {rule.get('kind')!r} spells no integer {name} (found "
                f"{value!r}); every chunk-claiming record carries its kernel's taper"
            )
        if not floor <= value <= bound:
            raise ValueError(
                f"grid rule {name} {value} is outside [{floor}, {bound}] (the module "
                f"constant {name.upper()} bounds it)"
            )
        taper.append(value)
    return taper[0], taper[1]


def chunk_capacity(rows: int, slots: int) -> int:
    """Partial rows a workspace holds for every token count up to ``rows`` on ``slots`` slots.

    ``min(rows, MAX_ROUNDS * slots)``: a table has at most :data:`MAX_ROUNDS`
    rounds of at most ``slots`` chunks whatever the rows per step and the
    taper (a record's taper stays within the module constants, which cut the
    most rounds), and every chunk holds a row; the bound grows with ``rows``,
    so a buffer sized for ``rows`` serves every smaller token count too.
    """

    rows, slots = int(rows), int(slots)
    if rows <= 0 or slots <= 0:
        raise ValueError("rows and slots must be positive")
    return min(rows, MAX_ROUNDS * slots)


def slot_count(rule: dict[str, Any], *, device_index: int) -> int:
    """Slots of a chunk-claiming grid rule: ``ctas_per_sm * (SM count - free_sms)``.

    A record whose kernel runs beside a side stream that keeps SMs spells
    their count under its grid rule (``free_sms``); every other record counts
    every SM.  The main backward kernel's chunk table, its grid and its
    workspace bound are built on this count.
    """

    sms = sm_count(device_index)
    free = int(rule.get("free_sms", 0))
    if not 0 <= free < sms:
        raise ValueError(f"grid rule free_sms {free} is outside [0, {sms})")
    return int(rule["ctas_per_sm"]) * (sms - free)


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``rows_per_cta`` modules launch ``ceil(rows / rows_per_cta)`` CTAs, capped
    at ``grid_limit`` when the rule records a positive limit, else at
    ``grid_per_sm * SM count`` of the device when that is positive (persistent
    grid-stride row loop in the kernel).  ``fixed`` modules launch exactly
    ``ctas`` CTAs (the dw reduce kernel: one CTA per column slab).
    ``claimed_chunks`` modules are the main backward kernel: ``slots``
    (:func:`slot_count`) persistent CTAs claim the chunks of the table of
    :func:`chunk_rounds` (``rows_per_issue`` rows per pipeline step of the
    kernel, under the taper its record spells: :func:`chunk_taper`), so the
    grid is ``min(slots, n_chunks)``; a column-task kernel's rule spells its
    ``col_tasks`` (``H / col`` column tasks, tickets beside the chunks), so its
    grid is ``min(slots, n_chunks + col_tasks)`` and a small ``T`` runs every
    task at once.  The launch is a plain
    one -- no co-residency is required -- and ``ctas_per_sm`` is the production
    grid factor capped at the launch bound the kernel declares.  ``rows`` must
    be positive; the public API returns empty outputs for zero rows without
    launching.
    """

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "rows_per_cta":
        grid_x = _ceil_div(rows, int(rule["rows_per_cta"]))
        limit = int(rule.get("grid_limit", 0))
        per_sm = int(rule.get("grid_per_sm", 0))
        if limit <= 0 and per_sm > 0:
            limit = per_sm * sm_count(device_index)
        if limit > 0:
            grid_x = min(grid_x, limit)
    elif kind == "fixed":
        grid_x = int(rule["ctas"])
    elif kind == "claimed_chunks":
        slots = slot_count(rule, device_index=device_index)
        grid_x = min(
            slots,
            n_chunks(rows, slots, int(rule["rows_per_issue"]), *chunk_taper(rule))
            + int(rule.get("col_tasks", 0)),
        )
    else:
        raise ValueError(f"unknown Cake RMSNorm training grid rule: {kind!r}")
    return (max(grid_x, 1), 1, 1)


def chunking(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int]:
    """``(n_chunks, slots)`` of the backward partition for ``rows`` tokens.

    The main backward module's CTAs claim the ``n_chunks`` chunks of the table
    built on ``slots`` (:func:`slot_count`) under the module's taper
    (:func:`chunk_taper`); both are host-known and never depend on scheduling.
    The main kernel takes both, the dw reduce kernel launched after it takes
    ``n_chunks``; a column-task kernel takes both and runs alone.
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] not in WORKSPACE_KINDS:
        raise ValueError(f"module kernel_{record['kernel']} owns no backward workspace")
    grid = record["grid_rule"]
    if grid["kind"] != "claimed_chunks":
        raise ValueError(
            f"module kernel_{record['kernel']} launches under {grid['kind']!r}, not a "
            "chunk-claiming grid"
        )
    slots = slot_count(grid, device_index=device_index)
    return n_chunks(rows, slots, int(grid["rows_per_issue"]), *chunk_taper(grid)), slots


def partial_capacity(record: dict[str, Any], rows: int, *, device_index: int) -> int:
    """Partial rows the workspace of one main backward module holds for every token count up to ``rows``."""

    rule = record["grid_rule"]
    if rule["kind"] != "claimed_chunks":
        raise ValueError(f"grid rule {rule['kind']!r} has no chunk table")
    return chunk_capacity(rows, slot_count(rule, device_index=device_index))


def workspace_layout(
    record: dict[str, Any],
    rows: int,
    *,
    device_index: int,
    capacity_record: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module.

    The workspace is one CUDA ``uint8`` buffer: the ``uint32`` counters in its
    first :data:`COUNTERS_BYTES` bytes, then the FP32 ``[partial_rows x
    capacity, hidden]`` per-chunk dw partials, ``capacity`` the
    :func:`chunk_capacity` of ``rows`` and ``partial_rows`` the record's
    partial rows per chunk (its workspace rule's ``partial_rows``: 1 by
    default; 2 for a main kernel that stores each chunk's rounded high part in
    row ``c`` and its low part in row ``n_chunks + c``, whose dw reduce sums
    the ``2 n_chunks`` rows), so a buffer sized for ``rows`` serves every
    smaller token count; the counters never move.  The counters must be zero when the buffer is first
    used and the kernels leave them zero, so no host write is needed between
    launches or CUDA Graph replays.  A route whose backward modules switch on
    the row count shares one buffer: ``capacity_record`` (the module with the
    largest capacity) fixes the partial capacity for every module of the
    route, while ``n_chunks`` and ``slots`` follow ``record``, the module
    launched.  A column-task module (``claim_sum_f32`` with ``sum_bytes`` 0)
    binds the counters alone: no partial rows (``partial_bytes`` 0), whatever
    ``rows`` and ``capacity_record``.
    """

    rule = record["workspace_rule"]
    chunks, slots = chunking(record, rows, device_index=device_index)
    hidden = int(record["hidden"])
    counters = int(rule["counters"])
    partial_rows = int(rule.get("partial_rows", 1))
    if 4 * max(counters, 1) > COUNTERS_BYTES:
        raise ValueError(
            f"module kernel_{record['kernel']} declares {counters} counters; the "
            f"workspace holds {COUNTERS_BYTES // 4}"
        )
    if rule["kind"] == "claim_sum_f32":
        if int(rule["sum_bytes"]) != 0:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {rule['sum_bytes']} bytes of "
                "sums; this package's column-task kernels keep none"
            )
        return {
            "n_chunks": chunks,
            "slots": slots,
            "capacity_chunks": 0,
            "partial_rows": 0,
            "partial_bytes": 0,
            "partial_offset": COUNTERS_BYTES,
            "counters": counters,
            "counters_offset": 0,
            "total_bytes": COUNTERS_BYTES,
        }
    capacity = partial_capacity(
        record if capacity_record is None else capacity_record,
        rows,
        device_index=device_index,
    )
    if capacity < chunks:
        raise ValueError(
            f"the route's partial capacity {capacity} is below the {chunks} chunks of "
            f"module kernel_{record['kernel']} for {rows} rows"
        )
    partial_bytes = 4 * partial_rows * capacity * hidden
    total = COUNTERS_BYTES + _round_up(partial_bytes, 256)
    return {
        "n_chunks": chunks,
        "slots": slots,
        "capacity_chunks": capacity,
        "partial_rows": partial_rows,
        "partial_bytes": partial_bytes,
        "partial_offset": COUNTERS_BYTES,
        "counters": counters,
        "counters_offset": 0,
        "total_bytes": total,
    }


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake RMSNorm training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake RMSNorm training module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else the
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up)."""

    return cake_build.jit_specs(arch)


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid follows
    the module's recorded grid rule for ``rows`` tokens and the kernel runs on
    the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, rows, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        module.run(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "BACKWARD_ALIGNMENT_BYTES",
    "COUNTERS_BYTES",
    "FORWARD_ALIGNMENT_BYTES",
    "FP32_WEIGHT_ALIGNMENT_BYTES",
    "HIDDEN_SIZES",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_HALVING",
    "MAX_ROUNDS",
    "MODULES",
    "ROUTES",
    "ROW_ALIGNMENT_ELEMENTS",
    "TAIL_DIV",
    "WEIGHT_DTYPES",
    "WORKSPACE_KINDS",
    "arch_for_capability",
    "chunk_capacity",
    "chunk_rounds",
    "chunk_table",
    "chunking",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "n_chunks",
    "partial_capacity",
    "program_windows",
    "route_applies",
    "route_key",
    "route_modules",
    "run_module",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "supported_hidden_sizes",
    "weight_dtypes",
    "workspace_layout",
]
