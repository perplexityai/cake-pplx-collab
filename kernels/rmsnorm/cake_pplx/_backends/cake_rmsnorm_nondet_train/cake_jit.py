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

Host plan and loader glue of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (SM100 / SM103 / SM107).

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

A route is the ordered list of module records one public call may launch.
``fwd`` / ``fwd_residual`` launch one forward program of the deterministic
package (the forward has no cross-row reduction).  ``bwd`` / ``bwd_residual``
select one of two backward forms per token-count window.  The single kernel
(workspace rule ``claim_sum_f32``) is one launch: its persistent CTAs claim row
chunks from a counter in the workspace, write ``dx`` and add their per-column
``dw`` sums into the workspace's FP32 sums (one word per column) with bulk
reductions in the order they finish; the CTA that finishes last stores ``dw``,
the FP32 sum of the replicas, so ``dw`` is not guaranteed bitwise reproducible.
The part-row single kernel (workspace rule ``claim_part_f32``) is one launch
too: its CTAs write their FP32 ``dw`` sums as one FP32 row each into
scratch in the workspace, and once every row is in,
helper CTAs fold the rows and store ``dw`` (no column sums and no atomic add of
a sum; the row order follows scheduling, so ``dw`` is
not guaranteed bitwise reproducible either).  The deterministic
package's pair (``chunk_partials``; its programs renamed into this package)
is two launches: the main kernel writes one FP32 partial row of ``dw`` per
chunk, then the dw reduce kernel (role ``bwd_reduce``) folds the rows into
``dw`` in chunk order.  A pair whose grid rule spells ``free_sms`` (a
one-row-per-stage pair: one slot per resident CTA, the SMs it leaves to a
side stream) builds its chunk table, its grid and its workspace bound on
the other SMs (:func:`launch_sms`), as the deterministic package does.  No
launch is cooperative, so a backward never waits for the whole device.
Token counts are runtime scalars, so a changed number of rows never
recompiles anything.

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
    "cake_rmsnorm_train_03bdf5667404141e1eaf": {
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
        "kernel": "cake_rmsnorm_train_03bdf5667404141e1eaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03bdf5667404141e1eaf",
        "role": "fwd_mid_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_03bdf5667404141e1eaf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_e5a3aa2548fc41ee0485": {
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
        "kernel": "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
        "role": "fwd_mid_w32_residual",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0ad4d53c4a0df501a99d": {
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
        "kernel": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "role": "fwd_large",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_0ad4d53c4a0df501a99d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_595f506004038a65cc63": {
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
        "kernel": "cake_rmsnorm_train_595f506004038a65cc63",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_595f506004038a65cc63",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0ad4d53c4a0df501a99d__wide": {
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
        "kernel": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
        "role": "fwd_wide",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 12288 and 16384 rows: 40210950, the FP32-weight twin of bde56bd5.
    "cake_rmsnorm_train_40210950f2e7ee48bc3e__ov107_12288_16384": {
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
        "kernel": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ea88523077fc340e55f": {
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
        "kernel": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1ea88523077fc340e55f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_21d3416baa877808371e": {
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
        "kernel": "cake_rmsnorm_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ea88523077fc340e55f__107": {
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
        "kernel": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ea88523077fc340e55f",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1ea88523077fc340e55f__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_21d3416baa877808371e__107": {
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
        "kernel": "cake_rmsnorm_train_21d3416baa877808371e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_21d3416baa877808371e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_288d7b1d51c340236cb7": {
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
        "kernel": "cake_rmsnorm_train_288d7b1d51c340236cb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_288d7b1d51c340236cb7",
        "role": "fwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 1024 and 6144 rows: 3d619cd1, the FP32-weight twin of cf1d0554.
    "cake_rmsnorm_train_3d619cd1e3ae638cce3d__ov100_1024_6144": {
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
        "kernel": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, above 6144 rows: 5f89e5fd, the FP32-weight twin of 288d7b1d.
    "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov100_6144_inf": {
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
        "kernel": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 1024 and 6144 rows: 2763cd5e, the FP32-weight twin of bad4db91.
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ov103_1024_6144": {
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
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 6144 rows: 5f89e5fd, the FP32-weight twin of 288d7b1d.
    "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov103_6144_inf": {
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
        "kernel": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5f89e5fd2f8a6057997e",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2a42e5c02b240f8709b1": {
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
        "kernel": "cake_rmsnorm_train_2a42e5c02b240f8709b1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2a42e5c02b240f8709b1",
        "role": "fwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_2a42e5c02b240f8709b1 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_79fecc7016e44e940733": {
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
        "kernel": "cake_rmsnorm_train_79fecc7016e44e940733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_79fecc7016e44e940733",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2fdbab2af48db42cd2a9": {
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
        "kernel": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_2fdbab2af48db42cd2a9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_29d0ae48de6b86aa4a41": {
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
        "kernel": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low": {
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
        "kernel": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        "role": "fwd_main",
        "select_rule": {
            "above": 1536,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_2fdbab2af48db42cd2a9__low (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_29d0ae48de6b86aa4a41__low": {
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
        "kernel": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_308da18230017dadc02d": {
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
        "kernel": "cake_rmsnorm_train_308da18230017dadc02d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_308da18230017dadc02d",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_308da18230017dadc02d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_8e3407e059cfc005f421": {
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
        "kernel": "cake_rmsnorm_train_8e3407e059cfc005f421",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8e3407e059cfc005f421",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3430e5d9c053309b0323": {
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
        "kernel": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "role": "fwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3430e5d9c053309b0323 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_3c1e1f571f15d42e8d57": {
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
        "kernel": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3430e5d9c053309b0323__100": {
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
        "kernel": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3430e5d9c053309b0323",
        "role": "fwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3430e5d9c053309b0323__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_3c1e1f571f15d42e8d57__100": {
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
        "kernel": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3d20ab4a432bddf7349a": {
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
        "kernel": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3d20ab4a432bddf7349a (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_7da36ff557bb2bac8d89": {
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
        "kernel": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3d20ab4a432bddf7349a__100": {
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
        "kernel": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d20ab4a432bddf7349a",
        "role": "fwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_3d20ab4a432bddf7349a__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_7da36ff557bb2bac8d89__100": {
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
        "kernel": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7da36ff557bb2bac8d89",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4cdc4c495b3f59ff5069": {
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
        "kernel": "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
        "role": "fwd_large",
        "select_rule": {
            "above": 2048,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_4cdc4c495b3f59ff5069 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_92cf7b87f1ebffb1b12b": {
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
        "kernel": "cake_rmsnorm_train_92cf7b87f1ebffb1b12b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_92cf7b87f1ebffb1b12b",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_60f47c5d00570ecf7740": {
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
        "kernel": "cake_rmsnorm_train_60f47c5d00570ecf7740",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_60f47c5d00570ecf7740",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_60f47c5d00570ecf7740 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_61accbcc02891615f286": {
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
        "kernel": "cake_rmsnorm_train_61accbcc02891615f286",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_61accbcc02891615f286",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_682675bda578f5b7a312": {
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
        "kernel": "cake_rmsnorm_train_682675bda578f5b7a312",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_682675bda578f5b7a312",
        "role": "fwd_large",
        "select_rule": {
            "above": 12288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_682675bda578f5b7a312 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a30e5a3b69b7c183992f": {
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
        "kernel": "cake_rmsnorm_train_a30e5a3b69b7c183992f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a30e5a3b69b7c183992f",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 12288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bde56bd5a68c493f07a2": {
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
        "kernel": "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        "role": "fwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bde56bd5a68c493f07a2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_40210950f2e7ee48bc3e": {
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
        "kernel": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e7f4221a7c26b9dfdd23": {
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
        "kernel": "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_e7f4221a7c26b9dfdd23 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_fdfa500c83d95df0f5c1": {
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
        "kernel": "cake_rmsnorm_train_fdfa500c83d95df0f5c1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fdfa500c83d95df0f5c1",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7673978dc698790054d2": {
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
        "kernel": "cake_rmsnorm_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_7673978dc698790054d2 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_e2eabcb3b0915d916a13": {
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
        "kernel": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7673978dc698790054d2__quad": {
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
        "kernel": "cake_rmsnorm_train_7673978dc698790054d2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7673978dc698790054d2",
        "role": "fwd_quad",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_7673978dc698790054d2__quad (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_e2eabcb3b0915d916a13__quad": {
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
        "kernel": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e2eabcb3b0915d916a13",
        "role": "fwd_quad_w32",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_780b9833fd0a236ef0e4": {
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
        "kernel": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "role": "fwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_780b9833fd0a236ef0e4 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a4d3f82b3e776f4fc49b": {
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
        "kernel": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_780b9833fd0a236ef0e4__cut": {
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
        "kernel": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        "role": "fwd_cut",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_780b9833fd0a236ef0e4__cut (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a4d3f82b3e776f4fc49b__cut": {
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
        "kernel": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        "role": "fwd_cut_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7de1e51089d432dbc26b": {
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
        "kernel": "cake_rmsnorm_train_7de1e51089d432dbc26b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7de1e51089d432dbc26b",
        "role": "fwd_mid",
        "select_rule": {
            "above": 3584,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_7de1e51089d432dbc26b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_1974c5c86cee63f33370": {
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
        "kernel": "cake_rmsnorm_train_1974c5c86cee63f33370",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1974c5c86cee63f33370",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8392d1a008d58a006bba": {
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
        "kernel": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_8392d1a008d58a006bba (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_4371270e4db4b985998f": {
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
        "kernel": "cake_rmsnorm_train_4371270e4db4b985998f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4371270e4db4b985998f",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8392d1a008d58a006bba__107": {
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
        "kernel": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8392d1a008d58a006bba",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_8392d1a008d58a006bba__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_4371270e4db4b985998f__107": {
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
        "kernel": "cake_rmsnorm_train_4371270e4db4b985998f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4371270e4db4b985998f",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8757f6da86abf54363d5": {
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
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8757f6da86abf54363d5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8757f6da86abf54363d5",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_8757f6da86abf54363d5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_6769e393999bc2ba1926": {
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
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_6769e393999bc2ba1926",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6769e393999bc2ba1926",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_a4da85136fba4588239f": {
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
        "kernel": "cake_rmsnorm_train_a4da85136fba4588239f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a4da85136fba4588239f",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_a4da85136fba4588239f (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a72c6ece1900f748eaa2": {
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
        "kernel": "cake_rmsnorm_train_a72c6ece1900f748eaa2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a72c6ece1900f748eaa2",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187": {
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
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a": {
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
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187__band": {
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
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_band",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187__band (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a__band": {
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
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_band_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187__mid": {
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
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a__mid": {
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
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b44107e05796c2922187__small": {
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
        "kernel": "cake_rmsnorm_train_b44107e05796c2922187",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b44107e05796c2922187",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b44107e05796c2922187__small (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_0da8e18d4af432c6927a__small": {
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
        "kernel": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0da8e18d4af432c6927a",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bad4db9153cc903b7ace": {
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
        "kernel": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bad4db9153cc903b7ace (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6": {
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
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bad4db9153cc903b7ace__100": {
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
        "kernel": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bad4db9153cc903b7ace__100 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__100": {
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
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bad4db9153cc903b7ace__ps5": {
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
        "kernel": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bad4db9153cc903b7ace",
        "role": "fwd_small",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bad4db9153cc903b7ace__ps5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ps5": {
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
        "kernel": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 768,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bf933e1660cab8e5787b": {
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
        "kernel": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "role": "fwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 4096 and 20480 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_4096_20480": {
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
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bf933e1660cab8e5787b__wide": {
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
        "kernel": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bf933e1660cab8e5787b",
        "role": "fwd_wide",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 20480 and 49152 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_20480_49152": {
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
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5a8099e177e463daab9": {
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
        "kernel": "cake_rmsnorm_train_c5a8099e177e463daab9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5a8099e177e463daab9",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_c5a8099e177e463daab9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_01f0ebf3db9fa2c0884e": {
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
        "kernel": "cake_rmsnorm_train_01f0ebf3db9fa2c0884e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_01f0ebf3db9fa2c0884e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_cf1d05546951bdbd9f5d": {
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
        "kernel": "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_cf1d05546951bdbd9f5d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_3d619cd1e3ae638cce3d": {
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
        "kernel": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid": {
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
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_d447b7f6398b60fe5daa__mid (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid": {
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
        "kernel": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103": {
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
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid103": {
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
        "kernel": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid107": {
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
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, between 1024 and 4096 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_1024_4096": {
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
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103": {
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
        "kernel": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d447b7f6398b60fe5daa",
        "role": "fwd_wide",
        "select_rule": {
            "above": 3072,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_bdc524a233d2bc6e9571__wide103": {
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
        "kernel": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdc524a233d2bc6e9571",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 3072,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e646bbdbd852884953da": {
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
        "kernel": "cake_rmsnorm_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e646bbdbd852884953da",
        "role": "fwd_main_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_e646bbdbd852884953da (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_6325c474ad9955a319e9": {
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
        "kernel": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "role": "fwd_main_w32_residual",
        "select_rule": {
            "above": 1024,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e646bbdbd852884953da__small": {
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
        "kernel": "cake_rmsnorm_train_e646bbdbd852884953da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e646bbdbd852884953da",
        "role": "fwd_small_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_e646bbdbd852884953da__small (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_6325c474ad9955a319e9__small": {
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
        "kernel": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6325c474ad9955a319e9",
        "role": "fwd_small_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18": {
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
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd913ecd254a6bba4b18 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f0db3068d0d05442a59d": {
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
        "kernel": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18__107": {
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
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 49152 rows: f9f3423b, the FP32-weight twin of bf933e16.
    "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_49152_inf": {
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
        "kernel": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9f3423be0bd8ff8c794",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18__low": {
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
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 640,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd913ecd254a6bba4b18__low (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f0db3068d0d05442a59d__low": {
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
        "kernel": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 640,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd913ecd254a6bba4b18__low107": {
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
        "kernel": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        "role": "fwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd913ecd254a6bba4b18__low107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f0db3068d0d05442a59d__low107": {
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
        "kernel": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0db3068d0d05442a59d",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0489c784c13bb25499f6": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_0489c784c13bb25499f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0489c784c13bb25499f6",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_0489c784c13bb25499f6 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_f2e721db4c30900f12fe": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_f2e721db4c30900f12fe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f2e721db4c30900f12fe",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1b284baec9663d6de1b9": {
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
        "kernel": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1b284baec9663d6de1b9 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_77456435c9fbd1d18fcf": {
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
        "kernel": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_86d51eff64fc9fe57cf0": {
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
        "kernel": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "role": "fwd_rows",
        "select_rule": {
            "above": 40960,
            "at_most": 81920,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_86d51eff64fc9fe57cf0 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_42d82d26e97c36651dae": {
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
        "kernel": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 40960,
            "at_most": 81920,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1388ab779a35c4e41efb": {
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
        "kernel": "cake_rmsnorm_train_1388ab779a35c4e41efb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1388ab779a35c4e41efb",
        "role": "fwd_rows",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1388ab779a35c4e41efb (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_b0299f6e0881fb35f27b": {
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
        "kernel": "cake_rmsnorm_train_b0299f6e0881fb35f27b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b0299f6e0881fb35f27b",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_466331ff3cd2d6c62738": {
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
        "kernel": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_466331ff3cd2d6c62738 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_32bebe5a7782311d60a0": {
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
        "kernel": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_ed9483d8c356affee55d": {
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
        "kernel": "cake_rmsnorm_train_ed9483d8c356affee55d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ed9483d8c356affee55d",
        "role": "fwd_rows",
        "select_rule": {
            "above": 196608,
            "at_most": 327680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_ed9483d8c356affee55d (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_d0a656f053396a58c49a": {
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
        "kernel": "cake_rmsnorm_train_d0a656f053396a58c49a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d0a656f053396a58c49a",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 196608,
            "at_most": 327680,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5": {
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
        "kernel": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 327680,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_670bfb3eed0ddf6ba461": {
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
        "kernel": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 327680,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bbb05314930d192dfe7b": {
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
        "kernel": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bbb05314930d192dfe7b (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a8fffd93682c5e382dc5": {
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
        "kernel": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1b284baec9663d6de1b9__20480_40960": {
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
        "kernel": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1b284baec9663d6de1b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 20480,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_1b284baec9663d6de1b9__20480_40960 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_77456435c9fbd1d18fcf__20480_40960": {
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
        "kernel": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_77456435c9fbd1d18fcf",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 40960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_466331ff3cd2d6c62738__81920_114688": {
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
        "kernel": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_466331ff3cd2d6c62738__81920_114688 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_32bebe5a7782311d60a0__81920_114688": {
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
        "kernel": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 81920,
            "at_most": 114688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_86d51eff64fc9fe57cf0__114688_196608": {
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
        "kernel": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
        "role": "fwd_rows",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_86d51eff64fc9fe57cf0__114688_196608 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_42d82d26e97c36651dae__114688_196608": {
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
        "kernel": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_42d82d26e97c36651dae",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 114688,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_37f663183cbff23c2893": {
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
        "kernel": "cake_rmsnorm_train_37f663183cbff23c2893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_37f663183cbff23c2893",
        "role": "fwd_rows",
        "select_rule": {
            "above": 327680,
            "at_most": 368640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_37f663183cbff23c2893 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_18e3c0a5204006b3a56f": {
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
        "kernel": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 327680,
            "at_most": 368640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5__368640_inf": {
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
        "kernel": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 368640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5__368640_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_670bfb3eed0ddf6ba461__368640_inf": {
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
        "kernel": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 368640,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5ca409f44e169148301": {
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
        "kernel": "cake_rmsnorm_train_c5ca409f44e169148301",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5ca409f44e169148301",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_c5ca409f44e169148301 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_8f2756937eaac6b02502": {
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
        "kernel": "cake_rmsnorm_train_8f2756937eaac6b02502",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8f2756937eaac6b02502",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_bbb05314930d192dfe7b__6144_32768": {
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
        "kernel": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bbb05314930d192dfe7b",
        "role": "fwd_rows",
        "select_rule": {
            "above": 6144,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_bbb05314930d192dfe7b__6144_32768 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_a8fffd93682c5e382dc5__6144_32768": {
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
        "kernel": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a8fffd93682c5e382dc5",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_466331ff3cd2d6c62738__32768_98304": {
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
        "kernel": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_466331ff3cd2d6c62738",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": 98304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_466331ff3cd2d6c62738__32768_98304 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_32bebe5a7782311d60a0__32768_98304": {
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
        "kernel": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_32bebe5a7782311d60a0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 98304,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_37f663183cbff23c2893__98304_inf": {
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
        "kernel": "cake_rmsnorm_train_37f663183cbff23c2893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_37f663183cbff23c2893",
        "role": "fwd_rows",
        "select_rule": {
            "above": 98304,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_37f663183cbff23c2893__98304_inf (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_18e3c0a5204006b3a56f__98304_inf": {
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
        "kernel": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18e3c0a5204006b3a56f",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 98304,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b5e173f5ecc736d91109__mid": {
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
        "kernel": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "role": "fwd_mid",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 640 and 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_640_6144": {
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
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 640 and 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_640_6144": {
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
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 640,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5__small": {
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
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 0 and 640 rows: ba28d2c0, the FP32-weight twin of fd1097c4.
    "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__ov100_0_640": {
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
        "kernel": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, between 0 and 640 rows: 1defffdf, the FP32-weight twin of 5eef28cd.
    "cake_rmsnorm_train_1defffdf2712cae36784__ov103_0_640": {
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
        "kernel": "cake_rmsnorm_train_1defffdf2712cae36784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1defffdf2712cae36784",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5": {
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
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, between 6144 and 16384 rows: b49de9b4, the FP32-weight twin of 044d4a2a.
    "cake_rmsnorm_train_b49de9b47372fa4e1051__ov100_6144_16384": {
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
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_100a, above 16384 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_16384_inf": {
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
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_103a, above 6144 rows: 9ecb62c2, the FP32-weight twin of b5e173f5.
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_6144_inf": {
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
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 6144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fd1097c45c1e328871fd__107": {
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
        "kernel": "cake_rmsnorm_train_fd1097c45c1e328871fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd1097c45c1e328871fd",
        "role": "fwd_small",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_fd1097c45c1e328871fd__107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__107": {
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
        "kernel": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba28d2c0c2c81a2c714e",
        "role": "fwd_small_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5__low107": {
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
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_5eef28cd35498e9abfa5__low107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_1defffdf2712cae36784__low107": {
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
        "kernel": "cake_rmsnorm_train_1defffdf2712cae36784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1defffdf2712cae36784",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b5e173f5ecc736d91109__mid107": {
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
        "kernel": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5e173f5ecc736d91109",
        "role": "fwd_mid",
        "select_rule": {
            "above": 1152,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_b5e173f5ecc736d91109__mid107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_9ecb62c2640d571e50a8__mid107": {
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
        "kernel": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ecb62c2640d571e50a8",
        "role": "fwd_mid_w32",
        "select_rule": {
            "above": 1152,
            "at_most": 6144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_044d4a2aa0553c16db06__large107": {
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
        "kernel": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "role": "fwd_large",
        "select_rule": {
            "above": 6144,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_044d4a2aa0553c16db06__large107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_b49de9b47372fa4e1051__large107": {
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
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_large_w32",
        "select_rule": {
            "above": 6144,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_044d4a2aa0553c16db06__wide107": {
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
        "kernel": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_044d4a2aa0553c16db06",
        "role": "fwd_wide",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # The FP32-weight twin of cake_rmsnorm_train_044d4a2aa0553c16db06__wide107 (the effective weight read at FP32; every other operand and the host plan the owner's).
    "cake_rmsnorm_train_b49de9b47372fa4e1051__wide107": {
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
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_wide_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5eef28cd35498e9abfa5__107": {
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
        "kernel": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        "role": "fwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # A window of the re-cut FP32-weight route of sm_107a, above 49152 rows: b49de9b4, the FP32-weight twin of 044d4a2a.
    "cake_rmsnorm_train_b49de9b47372fa4e1051__ov107_49152_inf": {
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
        "kernel": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b49de9b47372fa4e1051",
        "role": "fwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 bf16 weight: (64, 20480] the single kernel 195c4317 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_195c4317b51bb4ea1137": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_195c4317b51bb4ea1137",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 2048,
        },
    },
    # sm_100a H256 f32 weight: (64, 20480] the single kernel 8de01314 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_8de0131445eed6eeabb7": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8de0131445eed6eeabb7",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 2048,
        },
    },
    # sm_100a H512 bf16 weight: (1280, 4096] the single kernel d0bde38a (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_d0bde38aa443d42a5f04": {
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
        "kernel": "cake_rmsnorm_train_d0bde38aa443d42a5f04",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d0bde38aa443d42a5f04",
        "role": "bwd_main",
        "select_rule": {
            "above": 1280,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    # sm_100a H512 bf16 weight: (4096, 32768] the deterministic package's det pair of these rows (main kernel ad1a5a3c and fixed-order dw reduce 7097069b with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_ad1a5a3c5f88e3d10036__4096_32768": {
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
        "kernel": "cake_rmsnorm_train_ad1a5a3c5f88e3d10036",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ad1a5a3c5f88e3d10036",
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
    # sm_100a H512 bf16 weight: (32768, inf] the deterministic package's det pair of these rows (main kernel 8877322a and fixed-order dw reduce 122721bd with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_8877322a3002e3a4fefd__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_8877322a3002e3a4fefd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8877322a3002e3a4fefd",
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
    # sm_100a H512 f32 weight: (960, inf] the single kernel fd08c6f4 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_fd08c6f436eb6071fa22": {
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
        "kernel": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 960,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    # sm_100a H1536 bf16 weight: (78784, 131072] the single kernel 0d5dc239 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_0d5dc239536f113f4926": {
        "arches": [
            "sm_100a",
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_0d5dc239536f113f4926",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0d5dc239536f113f4926",
        "role": "bwd_main",
        "select_rule": {
            "above": 78784,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 24576,
        },
    },
    # sm_100a H1536 bf16 weight: (131072, 214016] the deterministic package's one-row-per-stage det pair main kernel 38e32d90 with its fixed-order dw reduce.
    "cake_rmsnorm_train_38e32d90f62f7cfc7b51__131072_214016": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": 214016,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__131072_214016": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 131072,
            "at_most": 214016,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H1536 bf16 weight: (214016, inf] the single kernel ae610afd (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_ae610afdf77ff6714a5b__214016_inf": {
        "arches": [
            "sm_100a",
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_ae610afdf77ff6714a5b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ae610afdf77ff6714a5b",
        "role": "bwd_main",
        "select_rule": {
            "above": 214016,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 24576,
        },
    },
    # sm_100a H1536 f32 weight: (77824, 131072] the single kernel 363eabb3 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_363eabb3661dab78f3d3__77824_131072": {
        "arches": [
            "sm_100a",
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_363eabb3661dab78f3d3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_363eabb3661dab78f3d3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 77824,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 24576,
        },
    },
    # sm_100a H1536 f32 weight: (131072, 214016] the deterministic package's one-row-per-stage det pair main kernel e30c1899 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e30c1899e859da4f4c68__131072_214016": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": 214016,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H1536 f32 weight: (214016, inf] the single kernel 4b6981c1 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_4b6981c10fd40a9d8aec__214016_inf": {
        "arches": [
            "sm_100a",
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 214016,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 24576,
        },
    },
    # sm_100a H1536 f32 weight: (1024, 1184] the single kernel 4b6981c1 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_4b6981c10fd40a9d8aec__1024_1184": {
        "arches": [
            "sm_100a",
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4b6981c10fd40a9d8aec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 1184,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 24576,
        },
    },
    # sm_100a H5120 f32 weight: (767, 36864] the deterministic package's one-row-per-stage det pair main kernel f9b2d42c with its fixed-order dw reduce.
    "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__767_36864": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 767,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H2048 f32 weight: (336, 1024] the column-task kernel 17493054 (64 column tasks).
    "cake_rmsnorm_train_17493054fed5409b6df9__336_1024": {
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
        "kernel": "cake_rmsnorm_train_17493054fed5409b6df9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_17493054fed5409b6df9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 336,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (42496, 65536] the single kernel 9433061d (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_9433061da18b2db4299d__42496_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_9433061da18b2db4299d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9433061da18b2db4299d",
        "role": "bwd_main",
        "select_rule": {
            "above": 42496,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 bf16 weight: (65536, 131072] the single kernel 301133b6 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_301133b646c4197fe462": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_301133b646c4197fe462",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_301133b646c4197fe462",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 bf16 weight: (131072, inf] the single kernel 9433061d (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_9433061da18b2db4299d__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_9433061da18b2db4299d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9433061da18b2db4299d",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 f32 weight: (3584, 65536] the single kernel a562656c (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_a562656cef5bf5dda160__3584_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a562656cef5bf5dda160",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a562656cef5bf5dda160",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 f32 weight: (65536, 131072] the single kernel c61519fe (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_c61519fe43c45818b9b4": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c61519fe43c45818b9b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c61519fe43c45818b9b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 f32 weight: (131072, inf] the single kernel a562656c (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_a562656cef5bf5dda160__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a562656cef5bf5dda160",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a562656cef5bf5dda160",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 residual bf16 weight: (1408, 65536] the single kernel f6537c86 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_f6537c86c7ffae183a50__1408_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_f6537c86c7ffae183a50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f6537c86c7ffae183a50",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1408,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 residual bf16 weight: (65536, 131072] the single kernel 6feab384 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_6feab384e4c85e893ed5": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_6feab384e4c85e893ed5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6feab384e4c85e893ed5",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 residual bf16 weight: (131072, inf] the single kernel 3fbfdd75 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_3fbfdd750cbeb47032b5__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_3fbfdd750cbeb47032b5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3fbfdd750cbeb47032b5",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 residual f32 weight: (1408, 65536] the single kernel a640d0b7 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_a640d0b799889f785ad5__1408_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a640d0b799889f785ad5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a640d0b799889f785ad5",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1408,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 residual f32 weight: (65536, 131072] the single kernel f5453c09 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_f5453c09151f9d441744": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_f5453c09151f9d441744",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f5453c09151f9d441744",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H6144 residual f32 weight: (131072, inf] the single kernel a640d0b7 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_a640d0b799889f785ad5__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a640d0b799889f785ad5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a640d0b799889f785ad5",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 98304,
        },
    },
    # sm_100a H2048 bf16 weight: (12, 64] the column-task kernel 518cf16a (64 column tasks).
    "cake_rmsnorm_train_518cf16a7bf8e88042ee__12_64": {
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
        "kernel": "cake_rmsnorm_train_518cf16a7bf8e88042ee",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_518cf16a7bf8e88042ee",
        "role": "bwd_main",
        "select_rule": {
            "above": 12,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 bf16 weight: (64, 336] the column-task kernel 64e6aa82 (64 column tasks).
    "cake_rmsnorm_train_64e6aa82bb2c6d4ea978__64_336": {
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
        "kernel": "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64e6aa82bb2c6d4ea978",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 336,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (14, 64] the column-task kernel c2f08c95 (64 column tasks).
    "cake_rmsnorm_train_c2f08c957d4b0f17dddd__14_64": {
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
        "kernel": "cake_rmsnorm_train_c2f08c957d4b0f17dddd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c2f08c957d4b0f17dddd",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 14,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (64, 336] the column-task program 9030dbeb of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_9030dbeb948761ec2258__64_336": {
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
        "kernel": "cake_rmsnorm_train_9030dbeb948761ec2258",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9030dbeb948761ec2258",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 336,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (12, 64] the column-task kernel 82b84544 (128 column tasks).
    "cake_rmsnorm_train_82b84544dba8e2b70bc1__12_64": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_82b84544dba8e2b70bc1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_82b84544dba8e2b70bc1",
        "role": "bwd_main",
        "select_rule": {
            "above": 12,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (64, 1152] the column-task kernel 4822793e (128 column tasks).
    "cake_rmsnorm_train_4822793ea0c954c11af6__64_1152": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_4822793ea0c954c11af6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4822793ea0c954c11af6",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (14, 64] the column-task kernel cdef0b34 (128 column tasks).
    "cake_rmsnorm_train_cdef0b34f5080dbc61d1__14_64": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_cdef0b34f5080dbc61d1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cdef0b34f5080dbc61d1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 14,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (64, 1792] the column-task kernel 86160705 (128 column tasks).
    "cake_rmsnorm_train_8616070567821eaa3e35__64_1792": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_8616070567821eaa3e35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8616070567821eaa3e35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1792,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 bf16 weight: (4, 64] the column-task kernel 29bcbde1 (80 column tasks).
    "cake_rmsnorm_train_29bcbde1d271ea989fa8__4_64": {
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
        "kernel": "cake_rmsnorm_train_29bcbde1d271ea989fa8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29bcbde1d271ea989fa8",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 bf16 weight: (64, 767] the column-task kernel 080e0a6c (80 column tasks).
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__64_767": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (4, 64] the column-task kernel e6475d8f (96 column tasks).
    "cake_rmsnorm_train_e6475d8fa504ef4d7d2d__4_64": {
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
        "kernel": "cake_rmsnorm_train_e6475d8fa504ef4d7d2d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e6475d8fa504ef4d7d2d",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 bf16 weight: (64, 1152] the column-task kernel 44508df7 (96 column tasks).
    "cake_rmsnorm_train_44508df7dab2ab688f85__64_1152": {
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
        "kernel": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 f32 weight: (4, 64] the column-task kernel f4d41a09 (96 column tasks).
    "cake_rmsnorm_train_f4d41a0991647b82ac21__4_64": {
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
        "kernel": "cake_rmsnorm_train_f4d41a0991647b82ac21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f4d41a0991647b82ac21",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 f32 weight: (64, 1168] the column-task kernel b3e6531e (96 column tasks).
    "cake_rmsnorm_train_b3e6531e46667901ab91__64_1168": {
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
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (2, 64] the column-task kernel 3a55646d (96 column tasks).
    "cake_rmsnorm_train_3a55646d574035bfa271__2_64": {
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
        "kernel": "cake_rmsnorm_train_3a55646d574035bfa271",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a55646d574035bfa271",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 2,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (64, 1408] the column-task kernel 864e84f3 (96 column tasks).
    "cake_rmsnorm_train_864e84f3d874f850c091__64_1408": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 64,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual f32 weight: (4, 64] the column-task kernel 6e0d06ab (96 column tasks).
    "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c__4_64": {
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
        "kernel": "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual f32 weight: (64, 1408] the column-task kernel 71d36617 (96 column tasks).
    "cake_rmsnorm_train_71d36617003ab7742a77__64_1408": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 64,
            "at_most": 1408,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (4608, 16384] the single kernel cf4c213e (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_cf4c213e50482afdacdb__4608_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_cf4c213e50482afdacdb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cf4c213e50482afdacdb",
        "role": "bwd_main",
        "select_rule": {
            "above": 4608,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 bf16 weight: (16384, 36864] the single kernel 681b33d9 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_681b33d9c9d0ac4d29a4__16384_36864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_681b33d9c9d0ac4d29a4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_681b33d9c9d0ac4d29a4",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 f32 weight: (3584, 36864] the single kernel f7573bcb (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_f7573bcb748c0784f3ec": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 3584,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 f32 weight: (1792, 3584] the single kernel f7573bcb (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_f7573bcb748c0784f3ec__1792_3584": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1792,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H2048 bf16 weight: (7168, 11264] the single kernel 0fa4f255 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_0fa4f255caf709d09f66": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_0fa4f255caf709d09f66",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0fa4f255caf709d09f66",
        "role": "bwd_main",
        "select_rule": {
            "above": 7168,
            "at_most": 11264,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 bf16 weight: (16384, 58880] the deterministic package's det pair of these rows (main kernel ceeef043 and fixed-order dw reduce d0960be9 with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_ceeef043ac64f37fc529__16384_58880": {
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
        "kernel": "cake_rmsnorm_train_ceeef043ac64f37fc529",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ceeef043ac64f37fc529",
        "role": "bwd_main",
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
    # sm_100a H2048 f32 weight: (15403, 45056] the single kernel ba7dbf6a (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_ba7dbf6a07744268705e": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 15403,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 f32 weight: (45056, 58880] the single kernel ba7dbf6a (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_ba7dbf6a07744268705e__45056_58880": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 45056,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 f32 weight: (1024, 4096] the single kernel ba7dbf6a (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_ba7dbf6a07744268705e__1024_4096": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H5120 bf16 weight: (61952, 393216] the single kernel e21acade (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_e21acade55e05f342053__61952_393216": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_e21acade55e05f342053",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e21acade55e05f342053",
        "role": "bwd_main",
        "select_rule": {
            "above": 61952,
            "at_most": 393216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 40960,
        },
    },
    # sm_103a H5120 bf16 weight: (393216, inf] the single kernel 826e6a14 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_826e6a14a93bd988be4d__393216_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_826e6a14a93bd988be4d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_826e6a14a93bd988be4d",
        "role": "bwd_main",
        "select_rule": {
            "above": 393216,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 40960,
        },
    },
    # sm_103a H5120 f32 weight: (61952, 196608] the single kernel 05078a17 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_05078a174a09119cfe3c__61952_196608": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_05078a174a09119cfe3c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_05078a174a09119cfe3c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 61952,
            "at_most": 196608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 40960,
        },
    },
    # sm_103a H5120 f32 weight: (196608, inf] the single kernel 91a40f30 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_91a40f3094182a35c784__196608_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_91a40f3094182a35c784",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_91a40f3094182a35c784",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 196608,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 40960,
        },
    },
    # sm_103a H6144 residual bf16 weight: (6, 31] the column-task kernel 3a55646d (96 column tasks).
    "cake_rmsnorm_train_3a55646d574035bfa271__6_31": {
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
        "kernel": "cake_rmsnorm_train_3a55646d574035bfa271",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a55646d574035bfa271",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 6,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (31, 32] the deterministic package's column-task program of these token counts (96 column tasks).
    "cake_rmsnorm_train_864e84f3d874f850c091__31_32": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 31,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (32, 48] the column-task kernel 3a55646d (96 column tasks).
    "cake_rmsnorm_train_3a55646d574035bfa271__32_48": {
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
        "kernel": "cake_rmsnorm_train_3a55646d574035bfa271",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a55646d574035bfa271",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 32,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (48, 56] the column-task kernel 864e84f3 (96 column tasks).
    "cake_rmsnorm_train_864e84f3d874f850c091__48_56": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 48,
            "at_most": 56,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (304, 1920] the column-task kernel 864e84f3 (96 column tasks).
    "cake_rmsnorm_train_864e84f3d874f850c091__304_1920": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 304,
            "at_most": 1920,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (320, 1824] the column-task kernel 71d36617 (96 column tasks).
    "cake_rmsnorm_train_71d36617003ab7742a77__320_1824": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 320,
            "at_most": 1824,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (7168, 131072] the part-row single kernel 853d4447 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_853d4447e18f114f5da5__7168_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_853d4447e18f114f5da5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_853d4447e18f114f5da5",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 7168,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_107a H6144 residual bf16 weight: (131072, inf] the single kernel 87d2866c (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_87d2866c1ea3bcc08b13__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_87d2866c1ea3bcc08b13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_87d2866c1ea3bcc08b13",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_107a H6144 residual f32 weight: (7168, 131072] the part-row single kernel b394bc75 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_b394bc75f9df18f002f2__7168_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_b394bc75f9df18f002f2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b394bc75f9df18f002f2",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 7168,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_107a H6144 residual f32 weight: (131072, inf] the single kernel 64758944 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_647589443d4915ca8fde__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_647589443d4915ca8fde",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_647589443d4915ca8fde",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_107a H6144 residual bf16 weight: (576, 3584] the column-task kernel 864e84f3 (96 column tasks).
    "cake_rmsnorm_train_864e84f3d874f850c091__576_3584": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 576,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (768, 3584] the column-task kernel 71d36617 (96 column tasks).
    "cake_rmsnorm_train_71d36617003ab7742a77__768_3584": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 768,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H256 bf16 weight: (524288, inf] the single kernel 44ef3141 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_44ef314181f3f2e93293": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_44ef314181f3f2e93293",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44ef314181f3f2e93293",
        "role": "bwd_main",
        "select_rule": {
            "above": 524288,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 2048,
        },
    },
    # sm_100a / sm_103a H256 bf16 weight: (2048, 262144] the deterministic package's one-row-per-stage det pair main kernel 6bdbfe61 with its fixed-order dw reduce.
    "cake_rmsnorm_train_6bdbfe619b8270ff9a21": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "free_sms": 8,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 8,
            "tail_div": 1,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_6bdbfe619b8270ff9a21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6bdbfe619b8270ff9a21",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__2048_262144": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 bf16 weight: (20480, 92160] the single kernel 44ef3141 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_44ef314181f3f2e93293__20480_92160": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_44ef314181f3f2e93293",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44ef314181f3f2e93293",
        "role": "bwd_main",
        "select_rule": {
            "above": 20480,
            "at_most": 92160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 2048,
        },
    },
    "cake_rmsnorm_train_a385ae1681a2c078344e__92160_524288": {
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
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 92160,
            "at_most": 524288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__92160_524288": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 92160,
            "at_most": 524288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 f32 weight: (92160, 864240] the dw reduce of the det pair 420788e6 of that window (the BF16-weight route keeps affa1b95__92160_524288)
    "cake_rmsnorm_train_affa1b95f68bab8c411b__92160_864240": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 92160,
            "at_most": 864240,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H256 f32 weight: (864240, inf] the single kernel b71bbe98 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_b71bbe9852c16fe03114__864240_inf": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b71bbe9852c16fe03114",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b71bbe9852c16fe03114",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 864240,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 2048,
        },
    },
    # sm_100a / sm_103a H256 f32 weight: (2048, 262144] the deterministic package's one-row-per-stage det pair main kernel 6554d4eb with its fixed-order dw reduce.
    "cake_rmsnorm_train_6554d4eb4362141cb058": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "free_sms": 8,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 8,
            "tail_div": 1,
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_6554d4eb4362141cb058",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6554d4eb4362141cb058",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H256 f32 weight: (20480, 92160] the single kernel b71bbe98 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_b71bbe9852c16fe03114__20480_92160": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b71bbe9852c16fe03114",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b71bbe9852c16fe03114",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 92160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 2048,
        },
    },
    # sm_100a H256 f32 weight: (92160, 864240] the deterministic package's det pair main kernel 420788e6 with its fixed-order dw reduce.
    "cake_rmsnorm_train_420788e663a2c3a810f6__92160_864240": {
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
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 92160,
            "at_most": 864240,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H6144 residual bf16 weight: (1920, 131072] the part-row single kernel aac3ebbf (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_aac3ebbf0e11c453c472__1920_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_aac3ebbf0e11c453c472",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_aac3ebbf0e11c453c472",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 1920,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_103a H6144 residual bf16 weight: (131072, inf] the single kernel 115dcc71 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_115dcc719a6de4b079f1": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_115dcc719a6de4b079f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_115dcc719a6de4b079f1",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H6144 residual f32 weight: (1824, 131072] the part-row single kernel 5fbcc18d (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_5fbcc18d24f37f0ba607__1824_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5fbcc18d24f37f0ba607",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5fbcc18d24f37f0ba607",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 1824,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_103a H6144 residual f32 weight: (131072, inf] the single kernel 3c390339 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_3c390339fa8c82311a0a": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_3c390339fa8c82311a0a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3c390339fa8c82311a0a",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H1536 bf16 weight: (46592, 53184] the single kernel 20d7a230 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_20d7a230ed8dc161a93a__46592_53184": {
        "arches": [
            "sm_103a",
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_20d7a230ed8dc161a93a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_20d7a230ed8dc161a93a",
        "role": "bwd_main",
        "select_rule": {
            "above": 46592,
            "at_most": 53184,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 24576,
        },
    },
    # sm_103a H1536 bf16 weight: (64, 214] the deterministic package's column-task program of these token counts (32 column tasks).
    "cake_rmsnorm_train_4aef8d1ff2c1c2af7b80__64_214": {
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
        "kernel": "cake_rmsnorm_train_4aef8d1ff2c1c2af7b80",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4aef8d1ff2c1c2af7b80",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (53184, inf] the deterministic package's one-row-per-stage det pair main kernel 38e32d90 with its fixed-order dw reduce.
    "cake_rmsnorm_train_38e32d90f62f7cfc7b51__53184_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "role": "bwd_main",
        "select_rule": {
            "above": 53184,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__53184_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 53184,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 f32 weight: (46592, 53184] the deterministic package's det pair main kernel 51f46ff1 with its fixed-order dw reduce.
    "cake_rmsnorm_train_51f46ff164b142ec8888__46592_53184": {
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
        "kernel": "cake_rmsnorm_train_51f46ff164b142ec8888",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_51f46ff164b142ec8888",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 46592,
            "at_most": 53184,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__46592_53184": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 46592,
            "at_most": 53184,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 f32 weight: (53184, inf] the deterministic package's one-row-per-stage det pair main kernel e30c1899 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e30c1899e859da4f4c68__53184_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 53184,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 bf16 weight: (32, 152] the deterministic package's column-task program of these token counts (128 column tasks).
    "cake_rmsnorm_train_12d53fe5b85b212fc800__32_152": {
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
        "kernel": "cake_rmsnorm_train_12d53fe5b85b212fc800",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_12d53fe5b85b212fc800",
        "role": "bwd_main",
        "select_rule": {
            "above": 32,
            "at_most": 152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 bf16 weight: (20480, 65536] the part-row single kernel ef9f8d46 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_ef9f8d46c22e4af7aef1__20480_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 8,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_ef9f8d46c22e4af7aef1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ef9f8d46c22e4af7aef1",
        "role": "bwd_main",
        "select_rule": {
            "above": 20480,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 16384,
        },
    },
    # sm_103a H4096 bf16 weight: (65536, 262144] the part-row single kernel f0aebb74 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_f0aebb74650c8dc3f128__65536_262144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f0aebb74650c8dc3f128",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f0aebb74650c8dc3f128",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 16384,
        },
    },
    # sm_103a H4096 bf16 weight: (262144, inf] the single kernel 35888fb3 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_35888fb3b729e1f8e239__262144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_35888fb3b729e1f8e239",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35888fb3b729e1f8e239",
        "role": "bwd_main",
        "select_rule": {
            "above": 262144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_107a H256 bf16 weight: (786432, inf] the single kernel 44ef3141 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_44ef314181f3f2e93293__786432_inf": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_44ef314181f3f2e93293",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44ef314181f3f2e93293",
        "role": "bwd_main",
        "select_rule": {
            "above": 786432,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_107a H256 f32 weight: (786432, inf] the single kernel b71bbe98 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_b71bbe9852c16fe03114__786432_inf": {
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
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_b71bbe9852c16fe03114",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b71bbe9852c16fe03114",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 786432,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_107a H4096 bf16 weight: (65536, inf] the single kernel f11a8616 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_f11a86167db4f8e9a5da__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f11a86167db4f8e9a5da",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f11a86167db4f8e9a5da",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 131072,
        },
    },
    # sm_107a H4096 f32 weight: (24064, 65536] the deterministic package's det pair main kernel 0ff440da with its fixed-order dw reduce.
    "cake_rmsnorm_train_0ff440daa6f9f304e544__24064_65536": {
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
        "kernel": "cake_rmsnorm_train_0ff440daa6f9f304e544",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0ff440daa6f9f304e544",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 24064,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H4096 f32 weight: (65536, inf] the single kernel a1f537fc (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_a1f537fc94bde175bb3c__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_a1f537fc94bde175bb3c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a1f537fc94bde175bb3c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 131072,
        },
    },
    # sm_107a H6144 bf16 weight: (30720, 131072] the part-row single kernel ffbcfda2 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_ffbcfda26d306410e01f__30720_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_ffbcfda26d306410e01f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ffbcfda26d306410e01f",
        "role": "bwd_main",
        "select_rule": {
            "above": 30720,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_107a H6144 bf16 weight: (131072, inf] the single kernel ddedaa3b (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_ddedaa3be7466a4faa97__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_ddedaa3be7466a4faa97",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ddedaa3be7466a4faa97",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_107a H6144 f32 weight: (2304, 19456] the single kernel e77387f1 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_e77387f14390ba486055__2304_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e77387f14390ba486055",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e77387f14390ba486055",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2304,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_107a H6144 f32 weight: (28672, inf] the single kernel e77387f1 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_e77387f14390ba486055__28672_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e77387f14390ba486055",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e77387f14390ba486055",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 28672,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_103a H4096 f32 weight: (65536, 262144] the single kernel bd22da7e (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_bd22da7e2d4914167050__65536_262144": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_bd22da7e2d4914167050",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bd22da7e2d4914167050",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 262144,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H4096 f32 weight: (32, 152] the deterministic package's column-task program of these token counts (128 column tasks).
    "cake_rmsnorm_train_27615d0e36c9f2fe0856__32_152": {
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
        "kernel": "cake_rmsnorm_train_27615d0e36c9f2fe0856",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_27615d0e36c9f2fe0856",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32,
            "at_most": 152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 f32 weight: (262144, inf] the single kernel 36790f79 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_36790f79b8845172d188__262144_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_36790f79b8845172d188",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_36790f79b8845172d188",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 262144,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_103a H6144 bf16 weight: (24575, 65536] the part-row single kernel 455174e1 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_455174e1a3be71c79452__24575_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 6,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_455174e1a3be71c79452",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_455174e1a3be71c79452",
        "role": "bwd_main",
        "select_rule": {
            "above": 24575,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_103a H6144 bf16 weight: (65536, 131072] the part-row single kernel 850fba46 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_850fba46397a84b8cd7b__65536_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_850fba46397a84b8cd7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_850fba46397a84b8cd7b",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 24576,
        },
    },
    # sm_103a H6144 bf16 weight: (131072, inf] the single kernel 38337af1 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_38337af146209d6a60ea__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_38337af146209d6a60ea",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38337af146209d6a60ea",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H6144 f32 weight: (49152, 83392] the deterministic package's det pair main kernel 9a8c0396 with its fixed-order dw reduce.
    "cake_rmsnorm_train_9a8c0396edd85d4d545e__49152_83392": {
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
        "kernel": "cake_rmsnorm_train_9a8c0396edd85d4d545e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a8c0396edd85d4d545e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 49152,
            "at_most": 83392,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__49152_83392": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 49152,
            "at_most": 83392,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H6144 f32 weight: (83392, 131072] the single kernel cb1aee12 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_cb1aee125561ad8d4e29__83392_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_cb1aee125561ad8d4e29",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cb1aee125561ad8d4e29",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 83392,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 49152,
        },
    },
    # sm_103a H6144 f32 weight: (131072, inf] the single kernel 5a635713 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_5a635713b08645d57f48__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5a635713b08645d57f48",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5a635713b08645d57f48",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 49152,
        },
    },
    # sm_107a H2048 bf16 weight: (50688, inf] the deterministic package's det pair main kernel 21ee6fc9 with its fixed-order dw reduce.
    "cake_rmsnorm_train_21ee6fc93139c5e0cd41__50688_inf": {
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
        "kernel": "cake_rmsnorm_train_21ee6fc93139c5e0cd41",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_21ee6fc93139c5e0cd41",
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
    "cake_rmsnorm_train_c4d8be677fa70b841877__50688_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 50688,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H2048 f32 weight: (50688, inf] the deterministic package's det pair main kernel 3f270477 with its fixed-order dw reduce.
    "cake_rmsnorm_train_3f270477a248188afe2a__50688_inf": {
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
        "kernel": "cake_rmsnorm_train_3f270477a248188afe2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3f270477a248188afe2a",
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
    # sm_107a H2048 f32 weight: (1024, 16384] the deterministic package's one-row-per-stage det pair main kernel c83c314c with its fixed-order dw reduce.
    "cake_rmsnorm_train_c83c314c6dcf28ad7369__1024_16384": {
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
        "kernel": "cake_rmsnorm_train_c83c314c6dcf28ad7369",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c83c314c6dcf28ad7369",
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
    # sm_107a H1536 bf16 weight: (65536, inf] the single kernel c93d0266 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_c93d0266065acad326fb__65536_inf": {
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c93d0266065acad326fb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c93d0266065acad326fb",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 129024,
        },
    },
    # sm_107a H1536 f32 weight: (1024, 16384] the single kernel 7c0dd63f (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_7c0dd63f0b011049603e__1024_16384": {
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_7c0dd63f0b011049603e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7c0dd63f0b011049603e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 129024,
        },
    },
    # sm_107a H1536 f32 weight: (65536, inf] the single kernel c9dc9e43 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_c9dc9e43af546ff815d3__65536_inf": {
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
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c9dc9e43af546ff815d3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c9dc9e43af546ff815d3",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 129024,
        },
    },
    # sm_107a H512 f32 weight: (1024, 4096] the single kernel fd08c6f4 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_fd08c6f436eb6071fa22__1024_4096": {
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
        "kernel": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fd08c6f436eb6071fa22",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 bf16 weight: (58880, 131072] the single kernel 8404aa43 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_8404aa4380f6db7259cc": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_8404aa4380f6db7259cc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8404aa4380f6db7259cc",
        "role": "bwd_main",
        "select_rule": {
            "above": 58880,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 bf16 weight: (131072, inf] the single kernel 0fa4f255 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_0fa4f255caf709d09f66__131072_inf": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_0fa4f255caf709d09f66",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0fa4f255caf709d09f66",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 f32 weight: (58880, 65536] the FP32-weight det pair the deterministic package runs here (main kernel 7e6a0b63, the FP32-weight twin of ceeef043, and dw reduce d0960be9 with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_7e6a0b631aea3edf51f9__58880_65536": {
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
        "kernel": "cake_rmsnorm_train_7e6a0b631aea3edf51f9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7e6a0b631aea3edf51f9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 58880,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__58880_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 58880,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H2048 f32 weight: (65536, 131072] the single kernel 54f6e4b8 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_54f6e4b8d937f7fb5619": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_54f6e4b8d937f7fb5619",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_54f6e4b8d937f7fb5619",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_100a H2048 f32 weight: (131072, inf] the single kernel ba7dbf6a (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_ba7dbf6a07744268705e__131072_inf": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba7dbf6a07744268705e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__44032_65536": {
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
        "kernel": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H4096 bf16 weight: (65536, 131072] the single kernel cda582bb (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_cda582bb216e6776b449": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_cda582bb216e6776b449",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cda582bb216e6776b449",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 bf16 weight: (131072, inf] the single kernel 15e157d2 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_15e157d2ee9080d206a9__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_15e157d2ee9080d206a9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_15e157d2ee9080d206a9",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 f32 weight: (44032, 65536] the FP32-weight det pair the deterministic package runs here (main kernel 626aed74, the FP32-weight twin of 35a0b0ca, and dw reduce fcbfa14d with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_626aed743358c27513f1__44032_65536": {
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
        "kernel": "cake_rmsnorm_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H4096 f32 weight: (65536, 131072] the single kernel 3a10d379 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_3a10d379e6a7ae778964": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_3a10d379e6a7ae778964",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3a10d379e6a7ae778964",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_100a H4096 f32 weight: (131072, inf] the single kernel f7573bcb (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_f7573bcb748c0784f3ec__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f7573bcb748c0784f3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 3,
            "kind": "claim_sum_f32",
            "sum_bytes": 65536,
        },
    },
    # sm_103a H2048 bf16 weight: (61952, 68160] the deterministic package's det pair main kernel 302f64db with its fixed-order dw reduce.
    "cake_rmsnorm_train_302f64db6abe79e5259e__61952_68160": {
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
        "kernel": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "role": "bwd_main",
        "select_rule": {
            "above": 61952,
            "at_most": 68160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__61952_68160": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 61952,
            "at_most": 68160,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H2048 bf16 weight: (68160, 131072] the single kernel 87532fb0 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_87532fb038ad204517e1__68160_131072": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_87532fb038ad204517e1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_87532fb038ad204517e1",
        "role": "bwd_main",
        "select_rule": {
            "above": 68160,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 16384,
        },
    },
    # sm_103a H2048 bf16 weight: (131072, inf] the single kernel 33e5c382 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_33e5c382e06419d65390__131072_inf": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_33e5c382e06419d65390",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_33e5c382e06419d65390",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 16384,
        },
    },
    # sm_103a H2048 f32 weight: (61952, 68160] the deterministic package's det pair main kernel e98cb156 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e98cb156abfe1666c76f__61952_68160": {
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
        "kernel": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 61952,
            "at_most": 68160,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 f32 weight: (68160, 131072] the single kernel 149b129c (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_149b129cdd1468145b13__68160_131072": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_149b129cdd1468145b13",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_149b129cdd1468145b13",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 68160,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 16384,
        },
    },
    # sm_103a H2048 f32 weight: (64, 171] the deterministic package's column-task program of these token counts (32 column tasks).
    "cake_rmsnorm_train_978741b43372951cb5b9__64_171": {
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
        "kernel": "cake_rmsnorm_train_978741b43372951cb5b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_978741b43372951cb5b9",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 f32 weight: (131072, inf] the single kernel 1166d499 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_1166d499d4e846fcb2d6__131072_inf": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_1166d499d4e846fcb2d6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1166d499d4e846fcb2d6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 16384,
        },
    },
    "cake_rmsnorm_train_a385ae1681a2c078344e__262144_720896": {
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
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 262144,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_420788e663a2c3a810f6__262144_720896": {
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
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 262144,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H1536 bf16 weight: (64, 384] the column-task kernel dc478ac1 (96 column tasks).
    "cake_rmsnorm_train_dc478ac1195882aedee6__64_384": {
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
        "kernel": "cake_rmsnorm_train_dc478ac1195882aedee6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dc478ac1195882aedee6",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H512 bf16 weight: (16, 227] the column-task kernel 931757c8 (32 column tasks).
    "cake_rmsnorm_train_931757c844a9d257c08b__16_227": {
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
        "kernel": "cake_rmsnorm_train_931757c844a9d257c08b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_931757c844a9d257c08b",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (64, 384] the column-task kernel e83ee45f (96 column tasks).
    "cake_rmsnorm_train_e83ee45f66352eaef755__64_384": {
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
        "kernel": "cake_rmsnorm_train_e83ee45f66352eaef755",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e83ee45f66352eaef755",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 f32 weight: (64, 767] the column-task kernel 139d162a (80 column tasks).
    "cake_rmsnorm_train_139d162a8bb29d1b2d35__64_767": {
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
        "kernel": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H4096 f32 weight: (16, 256] the column-task program 30ec8bb6 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_30ec8bb68cf4fe0c817b__16_256": {
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
        "kernel": "cake_rmsnorm_train_30ec8bb68cf4fe0c817b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_30ec8bb68cf4fe0c817b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 bf16 weight: (2, 64] the column-task kernel 02e761bc (96 column tasks).
    "cake_rmsnorm_train_02e761bcf94053af6fc3__2_64": {
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
        "kernel": "cake_rmsnorm_train_02e761bcf94053af6fc3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_02e761bcf94053af6fc3",
        "role": "bwd_main",
        "select_rule": {
            "above": 2,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (2, 64] the column-task kernel 166e288c (96 column tasks).
    "cake_rmsnorm_train_166e288c9192be6b994c__2_64": {
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
        "kernel": "cake_rmsnorm_train_166e288c9192be6b994c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_166e288c9192be6b994c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H5120 f32 weight: (4, 64] the column-task kernel cb653957 (80 column tasks).
    "cake_rmsnorm_train_cb6539573725ac861f94__4_64": {
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
        "kernel": "cake_rmsnorm_train_cb6539573725ac861f94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cb6539573725ac861f94",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (5, 64] the column-task kernel 02e761bc (96 column tasks).
    "cake_rmsnorm_train_02e761bcf94053af6fc3__0_64_103": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_02e761bcf94053af6fc3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_02e761bcf94053af6fc3",
        "role": "bwd_main",
        "select_rule": {
            "above": 5,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 bf16 weight: (16, 63] the column-task kernel 518cf16a (64 column tasks).
    "cake_rmsnorm_train_518cf16a7bf8e88042ee__16_63": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_518cf16a7bf8e88042ee",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_518cf16a7bf8e88042ee",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 63,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 bf16 weight: (63, 171] the deterministic package's column-task program of these token counts (32 column tasks).
    "cake_rmsnorm_train_ed36a8d998abba41f227__63_171": {
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
        "kernel": "cake_rmsnorm_train_ed36a8d998abba41f227",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ed36a8d998abba41f227",
        "role": "bwd_main",
        "select_rule": {
            "above": 63,
            "at_most": 171,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H256 bf16 weight: (64, 1080] the deterministic package's det pair of these rows (main kernel a385ae16 and fixed-order dw reduce affa1b95 with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_a385ae1681a2c078344e__64_1080": {
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
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H5120 bf16 weight: (4, 16] the column-task kernel 29bcbde1 (80 column tasks).
    "cake_rmsnorm_train_29bcbde1d271ea989fa8__4_16": {
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
        "kernel": "cake_rmsnorm_train_29bcbde1d271ea989fa8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29bcbde1d271ea989fa8",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (6, 64] the column-task kernel 166e288c (96 column tasks).
    "cake_rmsnorm_train_166e288c9192be6b994c__0_64_103": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_166e288c9192be6b994c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_166e288c9192be6b994c",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 6,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (64, 214] the deterministic package's column-task program of these token counts (32 column tasks).
    "cake_rmsnorm_train_698c3d36d67408394ccf__64_214": {
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
        "kernel": "cake_rmsnorm_train_698c3d36d67408394ccf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_698c3d36d67408394ccf",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 214,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H2048 f32 weight: (16, 64] the column-task kernel c2f08c95 (64 column tasks).
    "cake_rmsnorm_train_c2f08c957d4b0f17dddd__0_64_103": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_c2f08c957d4b0f17dddd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c2f08c957d4b0f17dddd",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H256 f32 weight: (64, 1080] the deterministic package's det pair of these rows (main kernel 420788e6 and fixed-order dw reduce affa1b95 with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_420788e663a2c3a810f6__64_1080": {
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
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 f32 weight: (16, 32] the column-task kernel 8274f36f (128 column tasks).
    "cake_rmsnorm_train_8274f36f6fd2f50dd508__16_32": {
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
        "kernel": "cake_rmsnorm_train_8274f36f6fd2f50dd508",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8274f36f6fd2f50dd508",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H5120 f32 weight: (5, 16] the column-task kernel cb653957 (80 column tasks).
    "cake_rmsnorm_train_cb6539573725ac861f94__5_16": {
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
        "kernel": "cake_rmsnorm_train_cb6539573725ac861f94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cb6539573725ac861f94",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 5,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 f32 weight: (12, 16] the column-task kernel f4d41a09 (96 column tasks).
    "cake_rmsnorm_train_f4d41a0991647b82ac21__12_16": {
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
        "kernel": "cake_rmsnorm_train_f4d41a0991647b82ac21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f4d41a0991647b82ac21",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (10, 31] the column-task kernel 6e0d06ab (96 column tasks).
    "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c__10_31": {
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
        "kernel": "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 10,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (31, 32] the deterministic package's column-task program of these token counts (96 column tasks).
    "cake_rmsnorm_train_71d36617003ab7742a77__31_32": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 31,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (32, 48] the column-task kernel 6e0d06ab (96 column tasks).
    "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c__32_48": {
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
        "kernel": "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 32,
            "at_most": 48,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 bf16 weight: (16, 320] the column-task program 1207b056 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_1207b05663ed2b4de14b__16_320": {
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
        "kernel": "cake_rmsnorm_train_1207b05663ed2b4de14b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1207b05663ed2b4de14b",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 bf16 weight: (16, 31] the column-task kernel 518cf16a (64 column tasks).
    "cake_rmsnorm_train_518cf16a7bf8e88042ee__16_31": {
        "arches": [
            "sm_107a",
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
        "kernel": "cake_rmsnorm_train_518cf16a7bf8e88042ee",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_518cf16a7bf8e88042ee",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 bf16 weight: (31, 256] the column-task program 6720b4a1 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_6720b4a159be2b84b325__31_256": {
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
        "kernel": "cake_rmsnorm_train_6720b4a159be2b84b325",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6720b4a159be2b84b325",
        "role": "bwd_main",
        "select_rule": {
            "above": 31,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H256 bf16 weight: (64, 786432] the deterministic package's det pair of these rows (main kernel a385ae16 and fixed-order dw reduce 312bbe08), renamed into this package.
    "cake_rmsnorm_train_a385ae1681a2c078344e__64_786432": {
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
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 786432,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_312bbe084eac69d858c4__64_786432": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_312bbe084eac69d858c4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_312bbe084eac69d858c4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 786432,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H256 f32 weight: (64, 786432] the deterministic package's det pair of these rows (main kernel 420788e6 and fixed-order dw reduce 312bbe08), renamed into this package.
    "cake_rmsnorm_train_420788e663a2c3a810f6__64_786432": {
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
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 64,
            "at_most": 786432,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H512 f32 weight: (16, 384] the column-task program 8f29656c of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_8f29656c0018a749a588__16_384": {
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
        "kernel": "cake_rmsnorm_train_8f29656c0018a749a588",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8f29656c0018a749a588",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (1664, 2304] the column-task kernel b3e6531e (96 column tasks).
    "cake_rmsnorm_train_b3e6531e46667901ab91__1664_2304": {
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
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1664,
            "at_most": 2304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H4096 bf16 weight: (16, 256] the column-task program c1de7e41 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_c1de7e41da4550cca0bc__16_256": {
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
        "kernel": "cake_rmsnorm_train_c1de7e41da4550cca0bc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c1de7e41da4550cca0bc",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 f32 weight: (16, 320] the column-task program 137ba532 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_137ba5328176fb07b37d__16_320": {
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
        "kernel": "cake_rmsnorm_train_137ba5328176fb07b37d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_137ba5328176fb07b37d",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H2048 f32 weight: (16, 256] the column-task program 23c90013 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_23c900134dc889302136__16_256": {
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
        "kernel": "cake_rmsnorm_train_23c900134dc889302136",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_23c900134dc889302136",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (16, 1664] the column-task program 5cca2d0f of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_5cca2d0fa4553f992406__16_1664": {
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
        "kernel": "cake_rmsnorm_train_5cca2d0fa4553f992406",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5cca2d0fa4553f992406",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 1664,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (16, 768] the column-task program 3bbd4f84 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_3bbd4f848b4847bff893__16_768": {
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
        "kernel": "cake_rmsnorm_train_3bbd4f848b4847bff893",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3bbd4f848b4847bff893",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 16,
            "at_most": 768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H4096 bf16 weight: (16, 32] the column-task kernel dc7ed206 (128 column tasks).
    "cake_rmsnorm_train_dc7ed206fba49ab22d1d__16_32": {
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
        "kernel": "cake_rmsnorm_train_dc7ed206fba49ab22d1d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dc7ed206fba49ab22d1d",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (6, 16] the column-task kernel e6475d8f (96 column tasks).
    "cake_rmsnorm_train_e6475d8fa504ef4d7d2d__6_16": {
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
        "kernel": "cake_rmsnorm_train_e6475d8fa504ef4d7d2d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e6475d8fa504ef4d7d2d",
        "role": "bwd_main",
        "select_rule": {
            "above": 6,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (16, 960] the column-task program 0dafdf09 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_0dafdf094615d080fcc9__16_960": {
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
        "kernel": "cake_rmsnorm_train_0dafdf094615d080fcc9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dafdf094615d080fcc9",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (1151, 1152] the column-task program 0dafdf09 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_0dafdf094615d080fcc9__1151_1152": {
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
        "kernel": "cake_rmsnorm_train_0dafdf094615d080fcc9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dafdf094615d080fcc9",
        "role": "bwd_main",
        "select_rule": {
            "above": 1151,
            "at_most": 1152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (15, 320] the column-task program ee1be07a of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_ee1be07a89a0482de45e__15_320": {
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
        "kernel": "cake_rmsnorm_train_ee1be07a89a0482de45e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ee1be07a89a0482de45e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 15,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (447, 576] the column-task program ee1be07a of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_ee1be07a89a0482de45e__447_576": {
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
        "kernel": "cake_rmsnorm_train_ee1be07a89a0482de45e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ee1be07a89a0482de45e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 447,
            "at_most": 576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H512 bf16 weight: (16, 64] the deterministic package's column-task program of these token counts (32 column tasks).
    "cake_rmsnorm_train_bdfd77893d8f9db3c9bb__16_64": {
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
        "kernel": "cake_rmsnorm_train_bdfd77893d8f9db3c9bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdfd77893d8f9db3c9bb",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H512 bf16 weight: (16, 576] the column-task program 49af1e75 of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_49af1e75185ddd637b76__16_576": {
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
        "kernel": "cake_rmsnorm_train_49af1e75185ddd637b76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_49af1e75185ddd637b76",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (49152, inf] the single kernel 670f4caf (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_670f4cafe8c34a429d30": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_670f4cafe8c34a429d30",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_670f4cafe8c34a429d30",
        "role": "bwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_107a H5120 f32 weight: (32767, 131072] the part-row single kernel e7cb6972 (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_e7cb6972099b7e10c035__32767_131072": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 4,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_e7cb6972099b7e10c035",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e7cb6972099b7e10c035",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32767,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 20480,
        },
    },
    # sm_107a H5120 f32 weight: (131072, inf] the single kernel 13bd0c5f (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_13bd0c5f56e37867eb26__131072_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_13bd0c5f56e37867eb26",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_13bd0c5f56e37867eb26",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 122880,
        },
    },
    # sm_100a H512 f32 weight: (16, 227] the column-task kernel 404c8f54 (32 column tasks).
    "cake_rmsnorm_train_404c8f54ab3129877350__16_227": {
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
        "kernel": "cake_rmsnorm_train_404c8f54ab3129877350",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_404c8f54ab3129877350",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H512 f32 weight: (227, 292] the single kernel 1a322bb1 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_1a322bb1c550a7d70638__227_292": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_1a322bb1c550a7d70638",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a322bb1c550a7d70638",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 227,
            "at_most": 292,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    # sm_107a H5120 bf16 weight: (12, 31] the column-task kernel 29bcbde1 (80 column tasks).
    "cake_rmsnorm_train_29bcbde1d271ea989fa8__12_31": {
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
        "kernel": "cake_rmsnorm_train_29bcbde1d271ea989fa8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_29bcbde1d271ea989fa8",
        "role": "bwd_main",
        "select_rule": {
            "above": 12,
            "at_most": 31,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (31, 256] the column-task program 080e0a6c of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__31_256": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 31,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (511, 640] the column-task program 080e0a6c of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__511_640": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 511,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (895, 896] the column-task program 080e0a6c of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__895_896": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 895,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (15, 256] the column-task program 139d162a of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_139d162a8bb29d1b2d35__15_256": {
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
        "kernel": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 15,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (511, 640] the column-task program 139d162a of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_139d162a8bb29d1b2d35__511_640": {
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
        "kernel": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 511,
            "at_most": 640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (895, 1280] the column-task program 139d162a of the deterministic package, renamed into this package.
    "cake_rmsnorm_train_139d162a8bb29d1b2d35__895_1280": {
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
        "kernel": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 895,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H512 f32 weight: (16, 227] the deterministic package's column-task program of these token counts (32 column tasks).
    "cake_rmsnorm_train_2c60470d16e0a7fb6089__16_227": {
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
        "kernel": "cake_rmsnorm_train_2c60470d16e0a7fb6089",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2c60470d16e0a7fb6089",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 227,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_7dc29629ff21625607f8__384_1024": {
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
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # The deterministic package's backward pairs (main kernel, then its dw reduce), renamed
    # into this package, for the windows the routes run them in.
    "cake_rmsnorm_train_7dc29629ff21625607f8__214_1024": {
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
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
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
    "cake_rmsnorm_train_c1a9efdf06c3087b4efe__1024_1184": {
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
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 1184,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c1a9efdf06c3087b4efe__1024_1216": {
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
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c1a9efdf06c3087b4efe",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 1216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H1536 bf16 weight: (1184, 77824] the deterministic package's one-row-per-stage det pair main kernel 38e32d90 with its fixed-order dw reduce.
    "cake_rmsnorm_train_38e32d90f62f7cfc7b51__1184_77824": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "role": "bwd_main",
        "select_rule": {
            "above": 1184,
            "at_most": 77824,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__1184_77824": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1184,
            "at_most": 77824,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b457c4d134e44476dc6f__77824_78784": {
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
        "kernel": "cake_rmsnorm_train_b457c4d134e44476dc6f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b457c4d134e44476dc6f",
        "role": "bwd_large",
        "select_rule": {
            "above": 77824,
            "at_most": 78784,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__214_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 214,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__1024_1184": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 1184,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1863320f3158d35eead6__1024_1216": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_1863320f3158d35eead6",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_1863320f3158d35eead6",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 1216,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__77824_78784": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 77824,
            "at_most": 78784,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__384_1024": {
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
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_17d9cdfd157ba4b86c2b__1024_1216": {
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
            "tail_div": 6,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_17d9cdfd157ba4b86c2b",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 1216,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H1536 f32 weight: (1184, 77824] the deterministic package's one-row-per-stage det pair main kernel e30c1899 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e30c1899e859da4f4c68__1184_77824": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1184,
            "at_most": 77824,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__336_3072": {
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
        "kernel": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "role": "bwd_small",
        "select_rule": {
            "above": 336,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9ac8d21e46a658e7913f": {
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
        "kernel": "cake_rmsnorm_train_9ac8d21e46a658e7913f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9ac8d21e46a658e7913f",
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
    "cake_rmsnorm_train_c73074cf44f07fcf3bb6__4096_7168": {
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
        "kernel": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "role": "bwd_large",
        "select_rule": {
            "above": 4096,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H2048 bf16 weight: (11264, 16384] the deterministic package's det pair main kernel c73074cf with its fixed-order dw reduce.
    "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_16384": {
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
        "kernel": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c73074cf44f07fcf3bb6",
        "role": "bwd_large",
        "select_rule": {
            "above": 11264,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 336,
            "at_most": 3072,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 3072,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__4096_7168": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__11264_16384": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 11264,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__16384_58880": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 58880,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H2048 f32 weight: (171, 16384] the deterministic package's one-row-per-stage det pair main kernel c83c314c with its fixed-order dw reduce.
    "cake_rmsnorm_train_c83c314c6dcf28ad7369__171_16384": {
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
        "kernel": "cake_rmsnorm_train_c83c314c6dcf28ad7369",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c83c314c6dcf28ad7369",
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
    "cake_rmsnorm_train_b187e5989302def22b2d__4096_15403": {
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
        "kernel": "cake_rmsnorm_train_b187e5989302def22b2d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b187e5989302def22b2d",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 15403,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__4096_15403": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 15403,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9a830956447d4e304914__1152_4096": {
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
        "kernel": "cake_rmsnorm_train_9a830956447d4e304914",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a830956447d4e304914",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_4608": {
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
        "kernel": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_44032": {
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
        "kernel": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35a0b0ca6b2060313255",
        "role": "bwd_main",
        "select_rule": {
            "above": 36864,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__1152_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_4608": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 4608,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 36864,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__44032_65536": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 44032,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_626aed743358c27513f1__36864_44032": {
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
        "kernel": "cake_rmsnorm_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 36864,
            "at_most": 44032,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_09601d42628de24f6f87__64_1024": {
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
        "kernel": "cake_rmsnorm_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_09601d42628de24f6f87",
        "role": "bwd_small",
        "select_rule": {
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_09601d42628de24f6f87__227_1024": {
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
        "kernel": "cake_rmsnorm_train_09601d42628de24f6f87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_09601d42628de24f6f87",
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
    "cake_rmsnorm_train_123db0b2f7fb38cfab03__1024_1280": {
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
        "kernel": "cake_rmsnorm_train_123db0b2f7fb38cfab03",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_123db0b2f7fb38cfab03",
        "role": "bwd_low",
        "select_rule": {
            "above": 1024,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__64_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_1024": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 227,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__4096_32768": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_122721bd6b4650866553__32768_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fef03ce50a792c8704bd__1024_1280": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_fef03ce50a792c8704bd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fef03ce50a792c8704bd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 1280,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9f07cea1d13b923ed91e__292_960": {
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
        "kernel": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 292,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H512 f32 weight: (227, 1024] the deterministic package's det pair main kernel 9f07cea1 with its fixed-order dw reduce.
    "cake_rmsnorm_train_9f07cea1d13b923ed91e__227_1024": {
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
        "kernel": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9f07cea1d13b923ed91e",
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
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__292_960": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 292,
            "at_most": 960,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_100a H5120 bf16 weight: (767, 36864] the deterministic package's one-row-per-stage det pair main kernel bfde9b88 with its fixed-order dw reduce.
    "cake_rmsnorm_train_bfde9b88e800fc34722c__767_36864": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "role": "bwd_main",
        "select_rule": {
            "above": 767,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b482c84dbffcd75a4a4c__36864_56256": {
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
        "kernel": "cake_rmsnorm_train_b482c84dbffcd75a4a4c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b482c84dbffcd75a4a4c",
        "role": "bwd_main",
        "select_rule": {
            "above": 36864,
            "at_most": 56256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H5120 bf16 weight: (56256, 131072] the single kernel 670f4caf (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_670f4cafe8c34a429d30__56256_131072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_670f4cafe8c34a429d30",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_670f4cafe8c34a429d30",
        "role": "bwd_main",
        "select_rule": {
            "above": 56256,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 81920,
        },
    },
    # sm_100a H5120 bf16 weight: (131072, inf] the deterministic package's one-row-per-stage det pair main kernel bfde9b88 with its fixed-order dw reduce.
    "cake_rmsnorm_train_bfde9b88e800fc34722c__131072_inf": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "role": "bwd_main",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__131072_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__767_36864": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 767,
            "at_most": 36864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__36864_56256": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 36864,
            "at_most": 56256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9479b1298a1b19d4d996__36864_56256": {
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
        "kernel": "cake_rmsnorm_train_9479b1298a1b19d4d996",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9479b1298a1b19d4d996",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 36864,
            "at_most": 56256,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H5120 f32 weight: (56256, 131072] the single kernel 13bd0c5f (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_13bd0c5f56e37867eb26__56256_131072": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 5,
            "rows_per_issue": 1,
            "tail_div": 12,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_13bd0c5f56e37867eb26",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_13bd0c5f56e37867eb26",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 56256,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 81920,
        },
    },
    # sm_100a H5120 f32 weight: (131072, inf] the deterministic package's one-row-per-stage det pair main kernel f9b2d42c with its fixed-order dw reduce.
    "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__131072_inf": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096": {
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
        "kernel": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5c804aed7ad14b99e1d0",
        "role": "bwd_low",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # sm_100a H6144 bf16 weight: (4096, 42496] the deterministic package's det pair main kernel b93bf060 with its fixed-order dw reduce.
    "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_42496": {
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
        "kernel": "cake_rmsnorm_train_b93bf060d4d3defc8150",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b93bf060d4d3defc8150",
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
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_42496": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 42496,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_41b80b6af97974feb181__1168_3584": {
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
        "kernel": "cake_rmsnorm_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41b80b6af97974feb181",
        "role": "bwd_low_w32",
        "select_rule": {
            "above": 1168,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1168_3584": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1168,
            "at_most": 3584,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 bf16 weight: (1216, 4096] the deterministic package's one-row-per-stage det pair main kernel 38e32d90 with its fixed-order dw reduce, launched on every SM.
    "cake_rmsnorm_train_38e32d90f62f7cfc7b51__1216_4096": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "role": "bwd_main",
        "select_rule": {
            "above": 1216,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H1536 bf16 weight: (4096, 46592] the deterministic package's one-row-per-stage det pair main kernel 38e32d90 with its fixed-order dw reduce.
    "cake_rmsnorm_train_38e32d90f62f7cfc7b51__4096_46592": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_38e32d90f62f7cfc7b51",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 46592,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__1216_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1216,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_46592": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_9b727c2c9933b2ca6724",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 46592,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H1536 f32 weight: (214, 1024] the deterministic package's det pair main kernel baff9ecb with its fixed-order dw reduce.
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__214_1024": {
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
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
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
    # sm_103a H1536 f32 weight: (1216, 4096] the deterministic package's one-row-per-stage det pair main kernel e30c1899 with its fixed-order dw reduce, launched on every SM.
    "cake_rmsnorm_train_e30c1899e859da4f4c68__1216_4096": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1216,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H1536 f32 weight: (4096, 46592] the deterministic package's one-row-per-stage det pair main kernel e30c1899 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e30c1899e859da4f4c68__4096_46592": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 6,
            "free_sms": 8,
            "grid_factor": 6,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 6,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e30c1899e859da4f4c68",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 46592,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 bf16 weight: (171, 16384] the deterministic package's one-row-per-stage det pair main kernel cc26e316 with its fixed-order dw reduce.
    "cake_rmsnorm_train_cc26e316ae40d683a118__171_16384": {
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
        "kernel": "cake_rmsnorm_train_cc26e316ae40d683a118",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cc26e316ae40d683a118",
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
    "cake_rmsnorm_train_880768a0a264e04dbfb5__171_16384": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_880768a0a264e04dbfb5",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 171,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768": {
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
        "kernel": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_caf6978ed0a17fbf65b9",
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
    # sm_103a H2048 bf16 weight: (32768, 58496] the deterministic package's det pair main kernel 302f64db with its fixed-order dw reduce.
    "cake_rmsnorm_train_302f64db6abe79e5259e__32768_58496": {
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
        "kernel": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": 58496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": 58496,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H2048 bf16 weight: (58496, 61952] the deterministic package's det pair main kernel 302f64db with its fixed-order dw reduce.
    "cake_rmsnorm_train_302f64db6abe79e5259e__58496_61952": {
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
        "kernel": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_302f64db6abe79e5259e",
        "role": "bwd_main",
        "select_rule": {
            "above": 58496,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_d0960be986e58f449f50",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d0960be986e58f449f50",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 58496,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_596b457aaba073f8d69f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_596b457aaba073f8d69f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_596b457aaba073f8d69f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_07f9f2fbea7163639c27__16384_32768": {
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
        "kernel": "cake_rmsnorm_train_07f9f2fbea7163639c27",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_07f9f2fbea7163639c27",
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
    # sm_103a H2048 f32 weight: (32768, 58496] the deterministic package's det pair main kernel e98cb156 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_58496": {
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
        "kernel": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 58496,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H2048 f32 weight: (58496, 61952] the deterministic package's det pair main kernel e98cb156 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e98cb156abfe1666c76f__58496_61952": {
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
        "kernel": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e98cb156abfe1666c76f",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 58496,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 64,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_af005462cd6a301f31c4__1080_2048": {
        "arches": [
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
        "kernel": "cake_rmsnorm_train_af005462cd6a301f31c4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_af005462cd6a301f31c4",
        "role": "bwd_main",
        "select_rule": {
            "above": 1080,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1080,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_a385ae1681a2c078344e__720896_1572864": {
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
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
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
    "cake_rmsnorm_train_a385ae1681a2c078344e": {
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
        "kernel": "cake_rmsnorm_train_a385ae1681a2c078344e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a385ae1681a2c078344e",
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
    "cake_rmsnorm_train_affa1b95f68bab8c411b__64_1080": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 1080,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_affa1b95f68bab8c411b__262144_720896": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_affa1b95f68bab8c411b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 262144,
            "at_most": 720896,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_72138dce40d43da44aaf__720896_1572864": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 720896,
            "at_most": 1572864,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_72138dce40d43da44aaf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_72138dce40d43da44aaf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_72138dce40d43da44aaf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1572864,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b7d9577ff884de98ea06__1080_2048": {
        "arches": [
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
        "kernel": "cake_rmsnorm_train_b7d9577ff884de98ea06",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b7d9577ff884de98ea06",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1080,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_420788e663a2c3a810f6__720896_1572864": {
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
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
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
    "cake_rmsnorm_train_420788e663a2c3a810f6": {
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
        "kernel": "cake_rmsnorm_train_420788e663a2c3a810f6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_420788e663a2c3a810f6",
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
    "cake_rmsnorm_train_50044ee3bd1ca54bd821__152_256": {
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
        "kernel": "cake_rmsnorm_train_50044ee3bd1ca54bd821",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_50044ee3bd1ca54bd821",
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
    "cake_rmsnorm_train_ad2bb26ea5916baef963": {
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
        "kernel": "cake_rmsnorm_train_ad2bb26ea5916baef963",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ad2bb26ea5916baef963",
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
    # sm_103a H4096 bf16 weight: (1024, 20480] the deterministic package's one-row-per-stage det pair main kernel 5572b267 with its fixed-order dw reduce.
    "cake_rmsnorm_train_5572b2671862fe670ef4__1024_20480": {
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
        "kernel": "cake_rmsnorm_train_5572b2671862fe670ef4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5572b2671862fe670ef4",
        "role": "bwd_mid",
        "select_rule": {
            "above": 1024,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_20480": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__152_256": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 152,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3fd77c8ce97e8a523abb__152_256": {
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
        "kernel": "cake_rmsnorm_train_3fd77c8ce97e8a523abb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3fd77c8ce97e8a523abb",
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
    "cake_rmsnorm_train_16545056022785f2a7d8": {
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
        "kernel": "cake_rmsnorm_train_16545056022785f2a7d8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_16545056022785f2a7d8",
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
    # sm_103a H4096 f32 weight: (1024, 20480] the deterministic package's one-row-per-stage det pair main kernel 47a01388 with its fixed-order dw reduce.
    "cake_rmsnorm_train_47a01388444960da7aae__1024_20480": {
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
        "kernel": "cake_rmsnorm_train_47a01388444960da7aae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_47a01388444960da7aae",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 20480,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 f32 weight: (20480, 48640] the deterministic package's det pair main kernel 626aed74 with its fixed-order dw reduce.
    "cake_rmsnorm_train_626aed743358c27513f1__20480_48640": {
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
        "kernel": "cake_rmsnorm_train_626aed743358c27513f1",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_626aed743358c27513f1",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 20480,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H4096 f32 weight: (48640, 65536] the deterministic package's det pair main kernel a8eb7547 with its fixed-order dw reduce (A/B on GB300).
    "cake_rmsnorm_train_a8eb754739a1589de3ec__48640_65536": {
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
        "kernel": "cake_rmsnorm_train_a8eb754739a1589de3ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a8eb754739a1589de3ec",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 48640,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fcbfa14d591a9babfd26__20480_48640": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fcbfa14d591a9babfd26",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 20480,
            "at_most": 48640,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5d75ea5f3c25579af496__48640_65536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5d75ea5f3c25579af496",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 48640,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8e7f4d811e55a351a22e": {
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
        "kernel": "cake_rmsnorm_train_8e7f4d811e55a351a22e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8e7f4d811e55a351a22e",
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
    "cake_rmsnorm_train_274b0055aace183f2367": {
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
        "kernel": "cake_rmsnorm_train_274b0055aace183f2367",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_274b0055aace183f2367",
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
    "cake_rmsnorm_train_cfaae3cec98f8501832b": {
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
        "kernel": "cake_rmsnorm_train_cfaae3cec98f8501832b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cfaae3cec98f8501832b",
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
    "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_7097069b8eea6ba44d5a",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_122721bd6b4650866553__4096_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_122721bd6b4650866553": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56ee8e66479625fd078a__1024_32768": {
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
        "kernel": "cake_rmsnorm_train_56ee8e66479625fd078a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56ee8e66479625fd078a",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_a546f410ec8cf03ef20d": {
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
        "kernel": "cake_rmsnorm_train_a546f410ec8cf03ef20d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a546f410ec8cf03ef20d",
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
    "cake_rmsnorm_train_122721bd6b4650866553__1024_32768": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_122721bd6b4650866553",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_122721bd6b4650866553",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H5120 bf16 weight: (16, 64] the column-task kernel 080e0a6c (80 column tasks).
    "cake_rmsnorm_train_080e0a6c3b18e9bad790__16_64": {
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
        "kernel": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_080e0a6c3b18e9bad790",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__64_192": {
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
        "kernel": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 64,
            "at_most": 192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__64_192": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 64,
            "at_most": 192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H5120 bf16 weight: (192, 1024] the deterministic package's one-row-per-stage det pair main kernel bfde9b88 with its fixed-order dw reduce, launched on every SM.
    "cake_rmsnorm_train_bfde9b88e800fc34722c__192_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "role": "bwd_main",
        "select_rule": {
            "above": 192,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H5120 bf16 weight: (1024, 36301] the deterministic package's one-row-per-stage det pair main kernel bfde9b88 with its fixed-order dw reduce.
    "cake_rmsnorm_train_bfde9b88e800fc34722c__1024_36301": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bfde9b88e800fc34722c",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 36301,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2fc4932fbde0bc848b32__36301_61952": {
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
        "kernel": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2fc4932fbde0bc848b32",
        "role": "bwd_main",
        "select_rule": {
            "above": 36301,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__36301_61952": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 36301,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H5120 f32 weight: (16, 192] the column-task kernel 139d162a (80 column tasks).
    "cake_rmsnorm_train_139d162a8bb29d1b2d35__16_192": {
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
        "kernel": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_139d162a8bb29d1b2d35",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H5120 f32 weight: (192, 1024] the deterministic package's one-row-per-stage det pair main kernel f9b2d42c with its fixed-order dw reduce, launched on every SM.
    "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__192_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 0,
            "rows_per_issue": 1,
            "tail_div": 1,
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 192,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H5120 f32 weight: (1024, 36301] the deterministic package's one-row-per-stage det pair main kernel f9b2d42c with its fixed-order dw reduce.
    "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__1024_36301": {
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
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 36301,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 160,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__192_1024": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 192,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__1024_36301": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 36301,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_79811a76205d7d7dd197__36301_61952": {
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
        "kernel": "cake_rmsnorm_train_79811a76205d7d7dd197",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_79811a76205d7d7dd197",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 36301,
            "at_most": 61952,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H6144 bf16 weight: (16, 896] the column-task kernel 44508df7 (96 column tasks).
    "cake_rmsnorm_train_44508df7dab2ab688f85__16_896": {
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
        "kernel": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_44508df7dab2ab688f85",
        "role": "bwd_main",
        "select_rule": {
            "above": 16,
            "at_most": 896,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (896, 4096] the deterministic package's one-row-per-stage det pair main kernel 61421bc7 with its fixed-order dw reduce.
    "cake_rmsnorm_train_61421bc72e57d14d8a32__896_4096": {
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
        "kernel": "cake_rmsnorm_train_61421bc72e57d14d8a32",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_61421bc72e57d14d8a32",
        "role": "bwd_low",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__896_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_322574af704b51aef0bb__4096_24575": {
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
        "kernel": "cake_rmsnorm_train_322574af704b51aef0bb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_322574af704b51aef0bb",
        "role": "bwd_mid",
        "select_rule": {
            "above": 4096,
            "at_most": 24575,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1024_4096": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_24575": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_5f5cdc5151e3c0620bcd",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 24575,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_103a H6144 f32 weight: (16, 1024] the column-task kernel b3e6531e (96 column tasks).
    "cake_rmsnorm_train_b3e6531e46667901ab91__16_1024": {
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
        "kernel": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b3e6531e46667901ab91",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_41b80b6af97974feb181__1024_4096": {
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
        "kernel": "cake_rmsnorm_train_41b80b6af97974feb181",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41b80b6af97974feb181",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_9a8c0396edd85d4d545e__4096_49152": {
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
        "kernel": "cake_rmsnorm_train_9a8c0396edd85d4d545e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9a8c0396edd85d4d545e",
        "role": "bwd_mid_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # sm_103a H6144 residual bf16 weight: (56, 304] the column-task kernel 864e84f3 (96 column tasks).
    "cake_rmsnorm_train_864e84f3d874f850c091__56_304": {
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
        "kernel": "cake_rmsnorm_train_864e84f3d874f850c091",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_864e84f3d874f850c091",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 56,
            "at_most": 304,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (48, 320] the FP32-weight twin 71d36617 of the deterministic package's column-task program 864e84f3 of these token counts.
    "cake_rmsnorm_train_71d36617003ab7742a77__48_320": {
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
        "kernel": "cake_rmsnorm_train_71d36617003ab7742a77",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_71d36617003ab7742a77",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 48,
            "at_most": 320,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_7dc29629ff21625607f8__320_1024": {
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
        "kernel": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7dc29629ff21625607f8",
        "role": "bwd_small",
        "select_rule": {
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1536 bf16 weight: (1024, 12288] the deterministic package's one-row-per-stage det pair main kernel 88510c60 with its fixed-order dw reduce.
    "cake_rmsnorm_train_88510c609fd688aab969__1024_12288": {
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
        "kernel": "cake_rmsnorm_train_88510c609fd688aab969",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_88510c609fd688aab969",
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
    "cake_rmsnorm_train_db5c0b673c516bd0295d__1024_12288": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6a428da2227972ae18ae__12288_16384": {
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
        "kernel": "cake_rmsnorm_train_6a428da2227972ae18ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6a428da2227972ae18ae",
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
    "cake_rmsnorm_train_db5c0b673c516bd0295d__12288_16384": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56d26996239835327501__16384_45056": {
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
        "kernel": "cake_rmsnorm_train_56d26996239835327501",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56d26996239835327501",
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
    # sm_107a H1536 bf16 weight: (45056, 65536] the deterministic package's det pair main kernel 411e270c with its fixed-order dw reduce.
    "cake_rmsnorm_train_411e270c0575d26a77f4__45056_65536": {
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
        "kernel": "cake_rmsnorm_train_411e270c0575d26a77f4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_411e270c0575d26a77f4",
        "role": "bwd_main",
        "select_rule": {
            "above": 45056,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_45056": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 45056,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_db5c0b673c516bd0295d__45056_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_db5c0b673c516bd0295d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 45056,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_baff9ecbafe7de1e6030__320_1024_107": {
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
        "kernel": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_baff9ecbafe7de1e6030",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_45056": {
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
        "kernel": "cake_rmsnorm_train_de8e07d4cb312b06dbc4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_de8e07d4cb312b06dbc4",
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
    # sm_107a H1536 f32 weight: (45056, 65536] the deterministic package's det pair main kernel 64ffe19b with its fixed-order dw reduce.
    "cake_rmsnorm_train_64ffe19b041f5bfa34fb__45056_65536": {
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
        "kernel": "cake_rmsnorm_train_64ffe19b041f5bfa34fb",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64ffe19b041f5bfa34fb",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 45056,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 48,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_628248febbdfe17c3325__320_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 48,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_628248febbdfe17c3325",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_628248febbdfe17c3325",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 320,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_845cdcd8145ada631461__256_1024": {
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
        "kernel": "cake_rmsnorm_train_845cdcd8145ada631461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_845cdcd8145ada631461",
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
    # sm_107a H2048 bf16 weight: (1024, 16384] the deterministic package's one-row-per-stage det pair main kernel cc26e316 with its fixed-order dw reduce.
    "cake_rmsnorm_train_cc26e316ae40d683a118__1024_16384": {
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
        "kernel": "cake_rmsnorm_train_cc26e316ae40d683a118",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cc26e316ae40d683a118",
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
    "cake_rmsnorm_train_845cdcd8145ada631461__16384_50688": {
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
        "kernel": "cake_rmsnorm_train_845cdcd8145ada631461",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_845cdcd8145ada631461",
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
    "cake_rmsnorm_train_271082c56c222bf10d77": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_271082c56c222bf10d77",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_271082c56c222bf10d77",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c4d8be677fa70b841877": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50688": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c4d8be677fa70b841877",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 50688,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_838a9a4054731f7942af__256_1024": {
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
        "kernel": "cake_rmsnorm_train_838a9a4054731f7942af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838a9a4054731f7942af",
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
    "cake_rmsnorm_train_838a9a4054731f7942af__16384_50688": {
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
        "kernel": "cake_rmsnorm_train_838a9a4054731f7942af",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838a9a4054731f7942af",
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
    "cake_rmsnorm_train_a647fccc3d9fa9134733": {
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
        "kernel": "cake_rmsnorm_train_a647fccc3d9fa9134733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a647fccc3d9fa9134733",
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
    "cake_rmsnorm_train_d41a514f5210e9adc39f": {
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
        "kernel": "cake_rmsnorm_train_d41a514f5210e9adc39f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d41a514f5210e9adc39f",
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
    # sm_107a H4096 bf16 weight: (4096, 24064] the deterministic package's det pair main kernel ee4a9053 with its fixed-order dw reduce.
    "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_24064": {
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
        "kernel": "cake_rmsnorm_train_ee4a9053bb25b8d49fcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ee4a9053bb25b8d49fcf",
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
    # sm_107a H4096 bf16 weight: (24064, 24576] the deterministic package's det pair main kernel a647fccc with its fixed-order dw reduce (A/B on VR200, 212 SMs).
    "cake_rmsnorm_train_a647fccc3d9fa9134733__24064_24576": {
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
        "kernel": "cake_rmsnorm_train_a647fccc3d9fa9134733",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a647fccc3d9fa9134733",
        "role": "bwd_main",
        "select_rule": {
            "above": 24064,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H4096 bf16 weight: (24576, 65536] the part-row single kernel 23573e0c (FP32 dw sums, one FP32 row per CTA, folded by its helper CTAs).
    "cake_rmsnorm_train_23573e0c42a194a04876__24576_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 1,
            "table_slots_per_sm": 8,
            "tail_div": 12,
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_23573e0c42a194a04876",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_23573e0c42a194a04876",
        "role": "bwd_main",
        "select_rule": {
            "above": 24576,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "claim_part_f32",
            "part_row_bytes": 16384,
        },
    },
    "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2030d80884deb8abcf7b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_2030d80884deb8abcf7b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2030d80884deb8abcf7b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_dc439f8335327e482025": {
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
        "kernel": "cake_rmsnorm_train_dc439f8335327e482025",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dc439f8335327e482025",
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
    "cake_rmsnorm_train_b08633208e60b8ce94fd": {
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
        "kernel": "cake_rmsnorm_train_b08633208e60b8ce94fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b08633208e60b8ce94fd",
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
    "cake_rmsnorm_train_0f1859b9fe9e47668a1c__4096_24064": {
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
        "kernel": "cake_rmsnorm_train_0f1859b9fe9e47668a1c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0f1859b9fe9e47668a1c",
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
    "cake_rmsnorm_train_10e24516314c244771f4__4096_24064": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 24064,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_69bf6e104bb48b6b74bf__24064_24576": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_69bf6e104bb48b6b74bf",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24064,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_10e24516314c244771f4__24064_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_10e24516314c244771f4",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_10e24516314c244771f4",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 24064,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3feed5026e1156d2fba2__576_1024": {
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
        "kernel": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "role": "bwd_main",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096": {
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
        "kernel": "cake_rmsnorm_train_3feed5026e1156d2fba2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3feed5026e1156d2fba2",
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
    "cake_rmsnorm_train_dfaf378aa8ad93e68bbd": {
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
        "kernel": "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
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
    "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3__16384_49152": {
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
        "kernel": "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3",
        "role": "bwd_large",
        "select_rule": {
            "above": 16384,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H512 bf16 weight: (49152, inf] the single kernel 7764c8a3 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_7764c8a34009f3728312__49152_inf": {
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
        "kernel": "cake_rmsnorm_train_7764c8a34009f3728312",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7764c8a34009f3728312",
        "role": "bwd_main",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    "cake_rmsnorm_train_f94861c36a14eb75e39c__576_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 576,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d75459d901441384398b": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_d75459d901441384398b",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_d75459d901441384398b",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8f032213ba6eeb5d959f": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_8f032213ba6eeb5d959f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 16384,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b841d64bbcaefff5ce7b__384_1024": {
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
        "kernel": "cake_rmsnorm_train_b841d64bbcaefff5ce7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b841d64bbcaefff5ce7b",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_383f709cd5c4fadd9436": {
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
        "kernel": "cake_rmsnorm_train_383f709cd5c4fadd9436",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_383f709cd5c4fadd9436",
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
    "cake_rmsnorm_train_03efa681eb38f59b00c2__16384_49152": {
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
        "kernel": "cake_rmsnorm_train_03efa681eb38f59b00c2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03efa681eb38f59b00c2",
        "role": "bwd_large_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 16,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H512 f32 weight: (49152, inf] the single kernel 5117a0f6 (FP32 dw sums added in FP32 in no fixed order).
    "cake_rmsnorm_train_5117a0f693a160485d76__49152_inf": {
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
        "kernel": "cake_rmsnorm_train_5117a0f693a160485d76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5117a0f693a160485d76",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 49152,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    "cake_rmsnorm_train_f94861c36a14eb75e39c__384_1024": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 16,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_f94861c36a14eb75e39c",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 384,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__256_511": {
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
        "kernel": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 256,
            "at_most": 511,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__640_895": {
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
        "kernel": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 640,
            "at_most": 895,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_838c077df3a6d35d55b7__896_4096": {
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
        "kernel": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_838c077df3a6d35d55b7",
        "role": "bwd_small",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_6816fb80e97380f1bcbc": {
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
        "kernel": "cake_rmsnorm_train_6816fb80e97380f1bcbc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6816fb80e97380f1bcbc",
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
    "cake_rmsnorm_train_b8b0913b40f035f43b76__19968_49152": {
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
        "kernel": "cake_rmsnorm_train_b8b0913b40f035f43b76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b8b0913b40f035f43b76",
        "role": "bwd_main",
        "select_rule": {
            "above": 19968,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__256_511": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 256,
            "at_most": 511,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__640_895": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 640,
            "at_most": 895,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__896_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 896,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 19968,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19968,
            "at_most": 49152,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c5831b29d17867842e5f__19968_32767": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_c5831b29d17867842e5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19968,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H5120 f32 weight: (4096, 19968] the deterministic package's det pair of these rows (main kernel 468cc4d1, the FP32-weight twin of the plain program 6816fb80 the BF16-weight route runs here, and fixed-order dw reduce 2af4cf2f with that package's host plan), renamed into this package.
    "cake_rmsnorm_train_468cc4d1b8128187b4b3__4096_19968": {
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
        "kernel": "cake_rmsnorm_train_468cc4d1b8128187b4b3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_468cc4d1b8128187b4b3",
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
    "cake_rmsnorm_train_cd612de56c0985bb4caa__256_511": {
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
        "kernel": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 256,
            "at_most": 511,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_cd612de56c0985bb4caa__640_895": {
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
        "kernel": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "role": "bwd_small_w32",
        "select_rule": {
            "above": 640,
            "at_most": 895,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_cd612de56c0985bb4caa__1280_4096": {
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
        "kernel": "cake_rmsnorm_train_cd612de56c0985bb4caa",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cd612de56c0985bb4caa",
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
    "cake_rmsnorm_train_2af4cf2f58ba58f78099__1280_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_2af4cf2f58ba58f78099",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1280,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H5120 f32 weight: (19968, 32767] the deterministic package's det pair main kernel de0cf457 with its fixed-order dw reduce.
    "cake_rmsnorm_train_de0cf457cd240f9c6422__19968_32767": {
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
        "kernel": "cake_rmsnorm_train_de0cf457cd240f9c6422",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_de0cf457cd240f9c6422",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19968,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 128,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d292172df772adb17ae__960_1151": {
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
        "kernel": "cake_rmsnorm_train_5d292172df772adb17ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5d292172df772adb17ae",
        "role": "bwd_main",
        "select_rule": {
            "above": 960,
            "at_most": 1151,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_5d292172df772adb17ae__1152_19456": {
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
        "kernel": "cake_rmsnorm_train_5d292172df772adb17ae",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5d292172df772adb17ae",
        "role": "bwd_main",
        "select_rule": {
            "above": 1152,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H6144 bf16 weight: (19456, 30720] the deterministic package's det pair main kernel 9cc51adf with its fixed-order dw reduce.
    "cake_rmsnorm_train_9cc51adfe9826056658b__19456_30720": {
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
        "kernel": "cake_rmsnorm_train_9cc51adfe9826056658b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9cc51adfe9826056658b",
        "role": "bwd_main",
        "select_rule": {
            "above": 19456,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__960_1151": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 960,
            "at_most": 1151,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__1152_19456": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1152,
            "at_most": 19456,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_30720": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19456,
            "at_most": 30720,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H6144 f32 weight: (19456, 28672] the deterministic package's det pair main kernel bdfba721 with its fixed-order dw reduce.
    "cake_rmsnorm_train_bdfba721a43e0d093668__19456_28672": {
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
        "kernel": "cake_rmsnorm_train_bdfba721a43e0d093668",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_bdfba721a43e0d093668",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 19456,
            "at_most": 28672,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 192,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_28672": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 192,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_fb3f2a96e954f0e241bc",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 19456,
            "at_most": 28672,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__320_447": {
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
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 320,
            "at_most": 447,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_0dcdc563be377247044e__3584_7168": {
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
        "kernel": "cake_rmsnorm_train_0dcdc563be377247044e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0dcdc563be377247044e",
        "role": "bwd_main_residual",
        "select_rule": {
            "above": 3584,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_60bfcb24560bbae612c3__320_447": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 320,
            "at_most": 447,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 96,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_60bfcb24560bbae612c3",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 3584,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_56f988dde54db37f09d9__3584_7168": {
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
        "kernel": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_56f988dde54db37f09d9",
        "role": "bwd_main_w32_residual",
        "select_rule": {
            "above": 3584,
            "at_most": 7168,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 96,
            "kind": "chunk_partials",
        },
    },
    # sm_100a / sm_103a / sm_107a H256 bf16 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_780c64c6f2c4615bfbcf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 bf16 weight: (4, 24] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 24): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_f2ccf7197a641c39acbf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_f2ccf7197a641c39acbf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f2ccf7197a641c39acbf",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 24,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 bf16 weight: (24, 64] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 64): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_9fb6b410e91f9d92becf": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_9fb6b410e91f9d92becf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9fb6b410e91f9d92becf",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 24,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 f32 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_67cb56412bd46304b711": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_67cb56412bd46304b711",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_67cb56412bd46304b711",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 f32 weight: (4, 24] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 24): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_aa745395d7542d8a8e3a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_aa745395d7542d8a8e3a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_aa745395d7542d8a8e3a",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 24,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H256 f32 weight: (24, 64] the one-cluster kernel (1 CTA in one thread-block cluster, 8 row groups, max_rows 64): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_6a0114cb0b27a780a4c3": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 256,
        "kernel": "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 24,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 bf16 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_64a299691c17079fe510": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_64a299691c17079fe510",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64a299691c17079fe510",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 bf16 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_c167e0d26b363cf19e7b": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_c167e0d26b363cf19e7b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c167e0d26b363cf19e7b",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 f32 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_8c261dfc6e34351d7a11": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_8c261dfc6e34351d7a11",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8c261dfc6e34351d7a11",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H512 f32 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_88e6252beaf7d056d555": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 512,
        "kernel": "cake_rmsnorm_train_88e6252beaf7d056d555",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_88e6252beaf7d056d555",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 bf16 weight: (0, 2] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_db218c92c4c86b9da85a__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H1536 bf16 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_db218c92c4c86b9da85a__0_4": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_db218c92c4c86b9da85a",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 bf16 weight: (4, 5] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_c2a523de1ff578621a88__4_5": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 5,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 bf16 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_c2a523de1ff578621a88__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c2a523de1ff578621a88",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H1536 f32 weight: (0, 2] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H1536 f32 weight: (0, 4] the one-cluster kernel (1 CTA in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_4": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6d3986ee0dd962cafd21",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H1536 f32 weight: (4, 6] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_744aa77460c886f4f429__4_6": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_744aa77460c886f4f429",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_744aa77460c886f4f429",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 6,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H1536 f32 weight: (4, 16] the one-cluster kernel (2 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_744aa77460c886f4f429__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 2,
            "kind": "fixed",
        },
        "hidden": 1536,
        "kernel": "cake_rmsnorm_train_744aa77460c886f4f429",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_744aa77460c886f4f429",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H2048 bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_c5f62407b828a85f82ec": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_c5f62407b828a85f82ec",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5f62407b828a85f82ec",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 bf16 weight: (4, 12] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_a951623c157d070e85ab__4_12": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a951623c157d070e85ab",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a951623c157d070e85ab",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H2048 bf16 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_a951623c157d070e85ab__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_a951623c157d070e85ab",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a951623c157d070e85ab",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H2048 f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_0017ab41f68d8689ebc2": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_0017ab41f68d8689ebc2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0017ab41f68d8689ebc2",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H2048 f32 weight: (4, 14] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_14": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 14,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H2048 f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 2048,
        "kernel": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_65e5f2024d26a36f2e18",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H4096 bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_945a0afdb82ab47eb643": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_945a0afdb82ab47eb643",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_945a0afdb82ab47eb643",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 bf16 weight: (4, 12] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_41f44b4bc0b765350a50__4_12": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H4096 bf16 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_41f44b4bc0b765350a50__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_41f44b4bc0b765350a50",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H4096 f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_f6bd1634a115bd281055": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_f6bd1634a115bd281055",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f6bd1634a115bd281055",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H4096 f32 weight: (4, 14] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_b378ba007691c582171e__4_14": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_b378ba007691c582171e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b378ba007691c582171e",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 14,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H4096 f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_b378ba007691c582171e__4_16": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 4096,
        "kernel": "cake_rmsnorm_train_b378ba007691c582171e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b378ba007691c582171e",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H5120 bf16 weight: (0, 4] the one-cluster kernel (4 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_dbc680f356be4cd643de": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_dbc680f356be4cd643de",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_dbc680f356be4cd643de",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 bf16 weight: (4, 12] the one-cluster kernel (4 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_904b1ad45d6219b4f43c__4_12": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_904b1ad45d6219b4f43c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_904b1ad45d6219b4f43c",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H5120 f32 weight: (0, 4] the one-cluster kernel (4 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_d47138228b123e55fe5d": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_d47138228b123e55fe5d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d47138228b123e55fe5d",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H5120 f32 weight: (4, 5] the one-cluster kernel (4 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_05396cd3a077d3772fdc__4_5": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 5,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H5120 f32 weight: (4, 15] the one-cluster kernel (4 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_05396cd3a077d3772fdc__4_15": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 5120,
        "kernel": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_05396cd3a077d3772fdc",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 15,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H6144 bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_e37752b056dd4e4a0bee": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_e37752b056dd4e4a0bee",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e37752b056dd4e4a0bee",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 bf16 weight: (4, 6] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_fdfb73350c1a926a4929__4_6": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 6,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 bf16 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_fdfb73350c1a926a4929__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_fdfb73350c1a926a4929",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H6144 f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 f32 weight: (4, 12] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_18b8b8054c7053942bcd__4_12": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 12,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 8 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_18b8b8054c7053942bcd__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_18b8b8054c7053942bcd",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a H6144 residual bf16 weight: (0, 2] the one-cluster kernel (8 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_deeb187630473818923d__0_2": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_deeb187630473818923d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_deeb187630473818923d",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a / sm_107a H6144 residual bf16 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_deeb187630473818923d__0_4": {
        "arches": [
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_deeb187630473818923d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_deeb187630473818923d",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual bf16 weight: (4, 6] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_6": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 4,
            "at_most": 6,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual bf16 weight: (4, 15] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_15": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_94c547eccfe2ca37ee0f",
        "role": "bwd_tiny_residual",
        "select_rule": {
            "above": 4,
            "at_most": 15,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a / sm_107a H6144 residual f32 weight: (0, 4] the one-cluster kernel (8 CTAs in one thread-block cluster, 2 row groups, max_rows 4): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_a6b1a6e565dc98b49e1a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
        "role": "bwd_tiny_w32_residual",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_103a H6144 residual f32 weight: (4, 10] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_c29a8f856283578103dc__4_10": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c29a8f856283578103dc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c29a8f856283578103dc",
        "role": "bwd_tiny_w32_residual",
        "select_rule": {
            "above": 4,
            "at_most": 10,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_107a H6144 residual f32 weight: (4, 16] the one-cluster kernel (8 CTAs in one thread-block cluster, 4 row groups, max_rows 16): one launch, no workspace and no atomic; dx as the single kernel forms it and a fixed-order FP32 dw, the same bits on every launch.
    "cake_rmsnorm_train_c29a8f856283578103dc__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 8,
            "kind": "fixed",
        },
        "hidden": 6144,
        "kernel": "cake_rmsnorm_train_c29a8f856283578103dc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c29a8f856283578103dc",
        "role": "bwd_tiny_w32_residual",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__0_2": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__2_4": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__4_512": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__512_1024": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__1024_2048": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__4_32": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__32_64": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__64_512": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__512_1024": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__1024_32768": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3808022dd402387eed16": {
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
        "kernel": "cake_rmsnorm_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_628267fc724c2c8f0e07__2_4": {
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
        "kernel": "cake_rmsnorm_train_628267fc724c2c8f0e07",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_628267fc724c2c8f0e07",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3808022dd402387eed16__4_16384": {
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
        "kernel": "cake_rmsnorm_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_628267fc724c2c8f0e07": {
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
        "kernel": "cake_rmsnorm_train_628267fc724c2c8f0e07",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_628267fc724c2c8f0e07",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 2,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ecf9c260e418bc70046__2_4": {
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
        "kernel": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863__4_64": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ecf9c260e418bc70046__64_128": {
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
        "kernel": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863__128_16384": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 128,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ecf9c260e418bc70046": {
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
        "kernel": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752__16_32": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__32_256": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752__256_512": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 256,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__512_8192": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 512,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_39b9b41b16769252153d": {
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
        "kernel": "cake_rmsnorm_train_39b9b41b16769252153d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_39b9b41b16769252153d",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_007e82b24390c88156ac__24576_32768": {
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
        "kernel": "cake_rmsnorm_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03b5516039c533b453b9__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_03b5516039c533b453b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03b5516039c533b453b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__0_1": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c43709cadf31ed04033c": {
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
        "kernel": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__4_8": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c43709cadf31ed04033c__8_16": {
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
        "kernel": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__16_256": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c43709cadf31ed04033c__256_1024": {
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
        "kernel": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58": {
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
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_58c2fce1ea0b40758e91__2048_32768": {
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
        "kernel": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_99b209a37f0203af57ce__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9346600882e53a589509": {
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
        "kernel": "cake_rmsnorm_train_9346600882e53a589509",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9346600882e53a589509",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_401a018c68bc082078b2": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_401a018c68bc082078b2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_401a018c68bc082078b2",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_401a018c68bc082078b2__1024_2047": {
        "arches": [
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
        "kernel": "cake_rmsnorm_train_401a018c68bc082078b2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_401a018c68bc082078b2",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 2047,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    # sm_100a / sm_103a H128 bf16 weight: (2048, 8192] the deterministic package's one-row-per-stage det pair main kernel d197bcc3 with its fixed-order dw reduce.
    "cake_rmsnorm_train_d197bcc34403e94ee2bd__2048_8192": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "free_sms": 8,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 16,
            "tail_div": 1,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_d197bcc34403e94ee2bd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d197bcc34403e94ee2bd",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_096e5efbf0a00905ee5f__2048_8192": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_096e5efbf0a00905ee5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_096e5efbf0a00905ee5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8b0b86c827709e73ee26__8192_12288": {
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
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_8b0b86c827709e73ee26",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8b0b86c827709e73ee26",
        "role": "bwd_main",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    "cake_rmsnorm_train_c50d3bbd794af91a2538": {
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
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_c50d3bbd794af91a2538",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c50d3bbd794af91a2538",
        "role": "bwd_main",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    "cake_rmsnorm_train_8b0b86c827709e73ee26__16384_inf": {
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
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_8b0b86c827709e73ee26",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8b0b86c827709e73ee26",
        "role": "bwd_main",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    "cake_rmsnorm_train_ab8a320c36b909e7bf1e": {
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
        "kernel": "cake_rmsnorm_train_ab8a320c36b909e7bf1e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ab8a320c36b909e7bf1e",
        "role": "bwd_main_w32",
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
    "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea__1024_1536": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_a45d09aebf1bdca80e95__1536_2048": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_a45d09aebf1bdca80e95",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a45d09aebf1bdca80e95",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    # sm_100a / sm_103a H128 f32 weight: (2048, 8192] the deterministic package's one-row-per-stage det pair main kernel e23083c7 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e23083c7af388f9e2f50__2048_8192": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "free_sms": 8,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 16,
            "tail_div": 1,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_e23083c7af388f9e2f50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e23083c7af388f9e2f50",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_3e751b6f0ebbd6be9481": {
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
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    "cake_rmsnorm_train_d0fe504e1d8611c0b2dc__12288_16384": {
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
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_d0fe504e1d8611c0b2dc",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d0fe504e1d8611c0b2dc",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    "cake_rmsnorm_train_3e751b6f0ebbd6be9481__16384_inf": {
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
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 1024,
        },
    },
    "cake_rmsnorm_train_330930e9329d1e53aa5c": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_330930e9329d1e53aa5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_330930e9329d1e53aa5c",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_28647b7ceb414d4fd02c__4_8": {
        "arches": [
            "sm_100a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_28647b7ceb414d4fd02c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_28647b7ceb414d4fd02c",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_89fa3e964f6ac26dc6a2__8_64": {
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
        "kernel": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_main",
        "select_rule": {
            "above": 8,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac": {
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
        "kernel": "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_89fa3e964f6ac26dc6a2": {
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
        "kernel": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_0c581a133624ff396ab4": {
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
        "kernel": "cake_rmsnorm_train_0c581a133624ff396ab4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0c581a133624ff396ab4",
        "role": "bwd_main",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_af798d1034849cc692e9": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_af798d1034849cc692e9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_af798d1034849cc692e9",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    # sm_100a / sm_103a H1024 bf16 weight: (32768, inf] the deterministic package's one-row-per-stage det pair main kernel acd331db with its fixed-order dw reduce.
    "cake_rmsnorm_train_acd331db1201bf2fb44e__32768_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_acd331db1201bf2fb44e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_acd331db1201bf2fb44e",
        "role": "bwd_main",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_176f79f9faa86f2fa441__32768_inf": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_176f79f9faa86f2fa441",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_176f79f9faa86f2fa441",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3b5dc80e2313fd64010e": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_81e742d5d2cf1e0f790a": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_81e742d5d2cf1e0f790a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_81e742d5d2cf1e0f790a",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_8501e95ea685c476cb50__4_8": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_8501e95ea685c476cb50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8501e95ea685c476cb50",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_744d74299c2d45867c94": {
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
        "kernel": "cake_rmsnorm_train_744d74299c2d45867c94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_744d74299c2d45867c94",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 8,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_561bea88a25c2f245c29": {
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
        "kernel": "cake_rmsnorm_train_561bea88a25c2f245c29",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_561bea88a25c2f245c29",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_1a45ade5011422a0d019": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    "cake_rmsnorm_train_1a45ade5011422a0d019__4096_12288": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    "cake_rmsnorm_train_1a45ade5011422a0d019__12288_32768": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    # sm_100a H1024 f32 weight: (32768, inf] the deterministic package's one-row-per-stage det pair main kernel b201ba67 with its fixed-order dw reduce.
    "cake_rmsnorm_train_b201ba67ac366efbd7be__32768_inf": {
        "arches": [
            "sm_100a",
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
        "kernel": "cake_rmsnorm_train_b201ba67ac366efbd7be",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b201ba67ac366efbd7be",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_8895b0224bb3d596a3a6": {
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
        "kernel": "cake_rmsnorm_train_8895b0224bb3d596a3a6",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8895b0224bb3d596a3a6",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_a5f2942d61e591f57a76": {
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
        "kernel": "cake_rmsnorm_train_a5f2942d61e591f57a76",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a5f2942d61e591f57a76",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_809235a3965d337a6877": {
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
        "kernel": "cake_rmsnorm_train_809235a3965d337a6877",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_809235a3965d337a6877",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_12bbedff93fcaafa475d": {
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
        "kernel": "cake_rmsnorm_train_12bbedff93fcaafa475d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_12bbedff93fcaafa475d",
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
    "cake_rmsnorm_train_c646e0976e08c6fe8a91": {
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
        "kernel": "cake_rmsnorm_train_c646e0976e08c6fe8a91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c646e0976e08c6fe8a91",
        "role": "bwd_main",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_b2cf87e64c9399e38407__8192_inf": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_b2cf87e64c9399e38407",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b2cf87e64c9399e38407",
        "role": "bwd_main",
        "select_rule": {
            "above": 8192,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 10240,
        },
    },
    "cake_rmsnorm_train_59474336746fb32d4e51": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_59474336746fb32d4e51__4096_8192": {
        "arches": [
            "sm_100a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_103805d93149c8668264": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_103805d93149c8668264",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_103805d93149c8668264",
        "role": "bwd_main_w32",
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
    "cake_rmsnorm_train_8d8ef930b7d650c5538a": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_8d8ef930b7d650c5538a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8d8ef930b7d650c5538a",
        "role": "bwd_main_w32",
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
    "cake_rmsnorm_train_e476d0b2c0bd4285f16d": {
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
        "kernel": "cake_rmsnorm_train_e476d0b2c0bd4285f16d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e476d0b2c0bd4285f16d",
        "role": "bwd_main_w32",
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
    "cake_rmsnorm_train_80956d258172dfa76c2a": {
        "arches": [
            "sm_100a",
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_80956d258172dfa76c2a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_80956d258172dfa76c2a",
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
    "cake_rmsnorm_train_f6956a41e69eb126197e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_f6956a41e69eb126197e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f6956a41e69eb126197e",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 10240,
        },
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__0_2048": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3808022dd402387eed16__0_16384": {
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
        "kernel": "cake_rmsnorm_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863__0_16384": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__0_256": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752__256_1024": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__1024_8192": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_007e82b24390c88156ac__24576_inf": {
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
        "kernel": "cake_rmsnorm_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 24576,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__0_512": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c43709cadf31ed04033c__512_1024": {
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
        "kernel": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_99b209a37f0203af57ce": {
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
        "kernel": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_62e8e2756e015c39bf5c__8192_24576": {
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
        "kernel": "cake_rmsnorm_train_62e8e2756e015c39bf5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_62e8e2756e015c39bf5c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_58c2fce1ea0b40758e91__24576_32768": {
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
        "kernel": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea": {
        "arches": [
            "sm_103a",
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
        "kernel": "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea__1024_2047": {
        "arches": [
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
        "kernel": "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 2047,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_89fa3e964f6ac26dc6a2__4_64": {
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
        "kernel": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_main",
        "select_rule": {
            "above": 4,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_5e32dd889d72a75e348b": {
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
        "kernel": "cake_rmsnorm_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5e32dd889d72a75e348b",
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
    "cake_rmsnorm_train_33d9025920659f515c9f": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 5,
            "rows_per_issue": 2,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_33d9025920659f515c9f",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_33d9025920659f515c9f",
        "role": "bwd_main",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    "cake_rmsnorm_train_3b5dc80e2313fd64010e__1024_1536": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_ba8feeec37d2bdde1fca": {
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
        "kernel": "cake_rmsnorm_train_ba8feeec37d2bdde1fca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_ba8feeec37d2bdde1fca",
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
    "cake_rmsnorm_train_1a45ade5011422a0d019__1536_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    "cake_rmsnorm_train_1a45ade5011422a0d019__32768_131072": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1a45ade5011422a0d019",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32768,
            "at_most": 131072,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    "cake_rmsnorm_train_2980f513dd6581ec98e7__131072_inf": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_2980f513dd6581ec98e7",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_2980f513dd6581ec98e7",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 131072,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 8192,
        },
    },
    "cake_rmsnorm_train_12bbedff93fcaafa475d__512_2048": {
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
        "kernel": "cake_rmsnorm_train_12bbedff93fcaafa475d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_12bbedff93fcaafa475d",
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
    "cake_rmsnorm_train_b2cf87e64c9399e38407": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 5,
            "rows_per_issue": 4,
            "tail_div": 12,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_b2cf87e64c9399e38407",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b2cf87e64c9399e38407",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 10240,
        },
    },
    "cake_rmsnorm_train_59474336746fb32d4e51__512_2048": {
        "arches": [
            "sm_103a",
        ],
        "grid_rule": {
            "ctas": 80,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_59474336746fb32d4e51",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_59474336746fb32d4e51",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__0_8": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__8_16": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__16_256": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16,
            "at_most": 256,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__256_1024": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 256,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__1024_4096": {
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
        "kernel": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__4096_8192": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_cffbc5c8de9c874b207c__8192_8447": {
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
        "kernel": "cake_rmsnorm_train_cffbc5c8de9c874b207c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cffbc5c8de9c874b207c",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 8447,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_547372db70c0ca0be362__8447_13312": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 4,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_547372db70c0ca0be362",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_547372db70c0ca0be362",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8447,
            "at_most": 13312,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_9e8a70075f28be558bba__13312_32768": {
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
        "kernel": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_9e8a70075f28be558bba",
        "role": "fwd_rows",
        "select_rule": {
            "above": 13312,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_cffbc5c8de9c874b207c__32768_inf": {
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
        "kernel": "cake_rmsnorm_train_cffbc5c8de9c874b207c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_cffbc5c8de9c874b207c",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_141ce1cb9005dddeee33": {
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
        "kernel": "cake_rmsnorm_train_141ce1cb9005dddeee33",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_141ce1cb9005dddeee33",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_91c090c06a5add02f1fd": {
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
        "kernel": "cake_rmsnorm_train_91c090c06a5add02f1fd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_91c090c06a5add02f1fd",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16384,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__0_8": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__8_16": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__16_512": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__512_1536": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__1536_4096": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__4096_8192": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_f5110367098beea13130": {
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
        "kernel": "cake_rmsnorm_train_f5110367098beea13130",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f5110367098beea13130",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_364ed18ca31e5b754d4e__12288_16384": {
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
        "kernel": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_364ed18ca31e5b754d4e",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8bd2d83654da4780c48a__16384_16895": {
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
        "kernel": "cake_rmsnorm_train_8bd2d83654da4780c48a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8bd2d83654da4780c48a",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 16895,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__16895_24576": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16895,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_d99009bfce4043a5dffe__24576_32768": {
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
        "kernel": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d99009bfce4043a5dffe",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_008fb179743126fb2d35": {
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
        "kernel": "cake_rmsnorm_train_008fb179743126fb2d35",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_008fb179743126fb2d35",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3808022dd402387eed16__0_32": {
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
        "kernel": "cake_rmsnorm_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_628267fc724c2c8f0e07__32_64": {
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
        "kernel": "cake_rmsnorm_train_628267fc724c2c8f0e07",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_628267fc724c2c8f0e07",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3808022dd402387eed16__64_2048": {
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
        "kernel": "cake_rmsnorm_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 64,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7221de56ded18727adaf": {
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
        "kernel": "cake_rmsnorm_train_7221de56ded18727adaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7221de56ded18727adaf",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3808022dd402387eed16__4096_24576": {
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
        "kernel": "cake_rmsnorm_train_3808022dd402387eed16",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3808022dd402387eed16",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 24576,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_7221de56ded18727adaf__24576_32768": {
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
        "kernel": "cake_rmsnorm_train_7221de56ded18727adaf",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7221de56ded18727adaf",
        "role": "fwd_rows",
        "select_rule": {
            "above": 24576,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_a05a2262fd5fa6912442": {
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
        "kernel": "cake_rmsnorm_train_a05a2262fd5fa6912442",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_a05a2262fd5fa6912442",
        "role": "fwd_rows",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863__0_32": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 32,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ecf9c260e418bc70046__32_64": {
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
        "kernel": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32,
            "at_most": 64,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863__64_2048": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 64,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_04ea6484fdf4a9b14ee0": {
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
        "kernel": "cake_rmsnorm_train_04ea6484fdf4a9b14ee0",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_04ea6484fdf4a9b14ee0",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_513f4ffbfefd9d019863__4096_16384": {
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
        "kernel": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_513f4ffbfefd9d019863",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_1ecf9c260e418bc70046__16384_32768": {
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
        "kernel": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_1ecf9c260e418bc70046",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": 32768,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_67e0dfc12fbae0354e51": {
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
        "kernel": "cake_rmsnorm_train_67e0dfc12fbae0354e51",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_67e0dfc12fbae0354e51",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 32768,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 0,
            "at_most": 1,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__1_4": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752__4_8": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_007e82b24390c88156ac": {
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
        "kernel": "cake_rmsnorm_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 8,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__16_128": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752__128_512": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 128,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_007e82b24390c88156ac__512_1024": {
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
        "kernel": "cake_rmsnorm_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03b5516039c533b453b9": {
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
        "kernel": "cake_rmsnorm_train_03b5516039c533b453b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03b5516039c533b453b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4d9988d4680c7c3c02a5__1536_2048": {
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
        "kernel": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
        "role": "fwd_rows",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03a74dfe62df95c2a752__2048_4096": {
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
        "kernel": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03a74dfe62df95c2a752",
        "role": "fwd_rows",
        "select_rule": {
            "above": 2048,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_007e82b24390c88156ac__4096_12288": {
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
        "kernel": "cake_rmsnorm_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 4096,
            "at_most": 12288,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_03b5516039c533b453b9__12288_16384": {
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
        "kernel": "cake_rmsnorm_train_03b5516039c533b453b9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_03b5516039c533b453b9",
        "role": "fwd_rows",
        "select_rule": {
            "above": 12288,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_007e82b24390c88156ac__16384_inf": {
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
        "kernel": "cake_rmsnorm_train_007e82b24390c88156ac",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_007e82b24390c88156ac",
        "role": "fwd_rows",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6603555b17f6d901ea7a": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 3,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_6603555b17f6d901ea7a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6603555b17f6d901ea7a",
        "role": "fwd_rows",
        "select_rule": {
            "above": 6143,
            "at_most": 9984,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__0_4": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c43709cadf31ed04033c__4_8": {
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
        "kernel": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 4,
            "at_most": 8,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__8_128": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_c43709cadf31ed04033c__128_512": {
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
        "kernel": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c43709cadf31ed04033c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 128,
            "at_most": 512,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_58c2fce1ea0b40758e91": {
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
        "kernel": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 512,
            "at_most": 1024,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_62e8e2756e015c39bf5c": {
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
        "kernel": "cake_rmsnorm_train_62e8e2756e015c39bf5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_62e8e2756e015c39bf5c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1024,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_6258b6dac6840cc6bd58__1536_2048": {
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
        "kernel": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_6258b6dac6840cc6bd58",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_62e8e2756e015c39bf5c__2048_8192": {
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
        "kernel": "cake_rmsnorm_train_62e8e2756e015c39bf5c",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_62e8e2756e015c39bf5c",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 2048,
            "at_most": 8192,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_99b209a37f0203af57ce__8192_8703": {
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
        "kernel": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8192,
            "at_most": 8703,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_51ca3c875efd4b1a5746__8703_9984": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "grid_limit": 0,
            "grid_per_sm": 3,
            "kind": "rows_per_cta",
            "rows_per_cta": 8,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_51ca3c875efd4b1a5746",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_51ca3c875efd4b1a5746",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 8703,
            "at_most": 9984,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_99b209a37f0203af57ce__9984_16384": {
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
        "kernel": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_99b209a37f0203af57ce",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 9984,
            "at_most": 16384,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_58c2fce1ea0b40758e91__16384_inf": {
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
        "kernel": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_58c2fce1ea0b40758e91",
        "role": "fwd_rows_w32",
        "select_rule": {
            "above": 16384,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    # sm_107a H128 bf16 weight: (2047, 65536] the deterministic package's one-row-per-stage det pair main kernel d197bcc3 with its fixed-order dw reduce.
    "cake_rmsnorm_train_d197bcc34403e94ee2bd__2047_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 16,
            "tail_div": 1,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_d197bcc34403e94ee2bd",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_d197bcc34403e94ee2bd",
        "role": "bwd_main",
        "select_rule": {
            "above": 2047,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_096e5efbf0a00905ee5f__2047_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 32,
            "kind": "fixed",
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_096e5efbf0a00905ee5f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_096e5efbf0a00905ee5f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2047,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8b0b86c827709e73ee26__65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_8b0b86c827709e73ee26",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8b0b86c827709e73ee26",
        "role": "bwd_main",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    # sm_107a H128 f32 weight: (2047, 65536] the deterministic package's one-row-per-stage det pair main kernel e23083c7 with its fixed-order dw reduce.
    "cake_rmsnorm_train_e23083c7af388f9e2f50__2047_65536": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 5,
            "grid_factor": 5,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 5,
            "max_halving": 0,
            "rows_per_issue": 16,
            "tail_div": 1,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_e23083c7af388f9e2f50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_e23083c7af388f9e2f50",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 2047,
            "at_most": 65536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 4,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_3e751b6f0ebbd6be9481__107_65536_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 4,
            "grid_factor": 4,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 4,
            "max_halving": 5,
            "rows_per_issue": 8,
            "tail_div": 12,
        },
        "hidden": 128,
        "kernel": "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 65536,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 2,
            "kind": "claim_sum_f32",
            "sum_bytes": 32768,
        },
    },
    "cake_rmsnorm_train_89fa3e964f6ac26dc6a2__8_1536": {
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
        "kernel": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
        "role": "bwd_main",
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
    "cake_rmsnorm_train_0c581a133624ff396ab4__1536_2048": {
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
        "kernel": "cake_rmsnorm_train_0c581a133624ff396ab4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_0c581a133624ff396ab4",
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
    "cake_rmsnorm_train_5e32dd889d72a75e348b__2048_4095": {
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
        "kernel": "cake_rmsnorm_train_5e32dd889d72a75e348b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_5e32dd889d72a75e348b",
        "role": "bwd_main",
        "select_rule": {
            "above": 2048,
            "at_most": 4095,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1024 bf16 weight: (4095, inf] the deterministic package's grouped det pair main kernel 4572a519 with its fixed-order dw reduce.
    "cake_rmsnorm_train_4572a519afb83afc3743__4095_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "free_sms": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 0,
            "rows_per_issue": 5,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_4572a519afb83afc3743",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_4572a519afb83afc3743",
        "role": "bwd_main",
        "select_rule": {
            "above": 4095,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_4830a91c2471462a29c8": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_4830a91c2471462a29c8",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_4830a91c2471462a29c8",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1536,
            "at_most": 2048,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_3b5dc80e2313fd64010e__2048_4095": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 64,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_3b5dc80e2313fd64010e",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_3b5dc80e2313fd64010e",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 2048,
            "at_most": 4095,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_4c829a19ba34d598a48d__4095_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_4c829a19ba34d598a48d",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_4c829a19ba34d598a48d",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4095,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_8501e95ea685c476cb50__4_16": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 4,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_8501e95ea685c476cb50",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8501e95ea685c476cb50",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 4,
            "at_most": 16,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_744d74299c2d45867c94__16_1536": {
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
        "kernel": "cake_rmsnorm_train_744d74299c2d45867c94",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_744d74299c2d45867c94",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 16,
            "at_most": 1536,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_561bea88a25c2f245c29__1536_4095": {
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
        "kernel": "cake_rmsnorm_train_561bea88a25c2f245c29",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_561bea88a25c2f245c29",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 1536,
            "at_most": 4095,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1024 f32 weight: (4095, inf] the deterministic package's grouped det pair main kernel 64423f2d with its fixed-order dw reduce.
    "cake_rmsnorm_train_64423f2da36401d9de39__4095_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 2,
            "free_sms": 1,
            "grid_factor": 2,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 2,
            "max_halving": 0,
            "rows_per_issue": 5,
            "tail_div": 1,
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_64423f2da36401d9de39",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_64423f2da36401d9de39",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4095,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 32,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_4830a91c2471462a29c8__1536_4095": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 128,
            "kind": "fixed",
        },
        "hidden": 1024,
        "kernel": "cake_rmsnorm_train_4830a91c2471462a29c8",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_4830a91c2471462a29c8",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 1536,
            "at_most": 4095,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_12bbedff93fcaafa475d__512_6143": {
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
        "kernel": "cake_rmsnorm_train_12bbedff93fcaafa475d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_12bbedff93fcaafa475d",
        "role": "bwd_main",
        "select_rule": {
            "above": 512,
            "at_most": 6143,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1280 bf16 weight: (6143, 32767] the deterministic package's grouped det pair main kernel b1339aac with its fixed-order dw reduce.
    "cake_rmsnorm_train_b1339aac23ed13dedce9__6143_32767": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "free_sms": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 0,
            "rows_per_issue": 5,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_b1339aac23ed13dedce9",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_b1339aac23ed13dedce9",
        "role": "bwd_main",
        "select_rule": {
            "above": 6143,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1280 bf16 weight: (32767, inf] the deterministic package's one-warp grouped det pair main kernel f8cb729b with its fixed-order dw reduce.
    "cake_rmsnorm_train_f8cb729b7ec5f34002ca__32767_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 1,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 3,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_f8cb729b7ec5f34002ca",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_f8cb729b7ec5f34002ca",
        "role": "bwd_main",
        "select_rule": {
            "above": 32767,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_849ee5d846ddc145e3a9__512_6143": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_849ee5d846ddc145e3a9",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_849ee5d846ddc145e3a9",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 6143,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_849ee5d846ddc145e3a9__6143_32767": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_849ee5d846ddc145e3a9",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_849ee5d846ddc145e3a9",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 6143,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_e35c9369f7e555241f97__32767_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_e35c9369f7e555241f97",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_e35c9369f7e555241f97",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 32767,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_62559c1037dd2c61140b__0_4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_62559c1037dd2c61140b",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_62559c1037dd2c61140b",
        "role": "bwd_tiny",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_7905241de01123c84d2d__0_4": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 1,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_7905241de01123c84d2d",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_7905241de01123c84d2d",
        "role": "bwd_tiny_w32",
        "select_rule": {
            "above": 0,
            "at_most": 4,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 0,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_8d8ef930b7d650c5538a__4_128": {
        "arches": [
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
        "kernel": "cake_rmsnorm_train_8d8ef930b7d650c5538a",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_8d8ef930b7d650c5538a",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4,
            "at_most": 128,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 1,
            "kind": "claim_sum_f32",
            "sum_bytes": 0,
        },
    },
    "cake_rmsnorm_train_c66b8cbd5626e2c1f2b4__4096_6143": {
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
        "kernel": "cake_rmsnorm_train_c66b8cbd5626e2c1f2b4",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c66b8cbd5626e2c1f2b4",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 4096,
            "at_most": 6143,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1280 f32 weight: (6143, 32767] the deterministic package's grouped det pair main kernel de2f75d4 with its fixed-order dw reduce.
    "cake_rmsnorm_train_de2f75d431899520db87__6143_32767": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 1,
            "free_sms": 1,
            "grid_factor": 1,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 1,
            "max_halving": 0,
            "rows_per_issue": 5,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_de2f75d431899520db87",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_de2f75d431899520db87",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 6143,
            "at_most": 32767,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    # sm_107a H1280 f32 weight: (32767, inf] the deterministic package's one-warp grouped det pair main kernel c5386a35 with its fixed-order dw reduce.
    "cake_rmsnorm_train_c5386a35eeb7cfbd1658__32767_inf": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas_per_sm": 3,
            "free_sms": 1,
            "grid_factor": 3,
            "kind": "claimed_chunks",
            "launch_bound_ctas_per_sm": 3,
            "max_halving": 0,
            "rows_per_issue": 3,
            "tail_div": 1,
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_c5386a35eeb7cfbd1658",
        "launch": {
            "cooperative": False,
            "use_pdl": False,
        },
        "library": "cake_rmsnorm_train_c5386a35eeb7cfbd1658",
        "role": "bwd_main_w32",
        "select_rule": {
            "above": 32767,
            "at_most": None,
            "kind": "rows_between",
        },
        "workspace_rule": {
            "counters": 40,
            "kind": "chunk_partials",
        },
    },
    "cake_rmsnorm_train_849ee5d846ddc145e3a9__512_4096": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 160,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_849ee5d846ddc145e3a9",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_849ee5d846ddc145e3a9",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 512,
            "at_most": 4096,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
    "cake_rmsnorm_train_382c8727d8b150884d7f__4096_6143": {
        "arches": [
            "sm_107a",
        ],
        "grid_rule": {
            "ctas": 320,
            "kind": "fixed",
        },
        "hidden": 1280,
        "kernel": "cake_rmsnorm_train_382c8727d8b150884d7f",
        "launch": {
            "cooperative": False,
            "use_pdl": True,
        },
        "library": "cake_rmsnorm_train_382c8727d8b150884d7f",
        "role": "bwd_reduce",
        "select_rule": {
            "above": 4096,
            "at_most": 6143,
            "kind": "rows_between",
        },
        "workspace_rule": None,
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "fwd_h1024": [
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__0_2",
            "cake_rmsnorm_train_9e8a70075f28be558bba__2_4",
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__4_512",
            "cake_rmsnorm_train_9e8a70075f28be558bba__512_1024",
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__1024_2048",
            "cake_rmsnorm_train_9e8a70075f28be558bba",
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
            "cake_rmsnorm_train_9e8a70075f28be558bba__32768_inf",
        ],
        "fwd_w32_h1024": [
            "cake_rmsnorm_train_364ed18ca31e5b754d4e",
            "cake_rmsnorm_train_d99009bfce4043a5dffe",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__4_32",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__32_64",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__64_512",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__512_1024",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__1024_32768",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__32768_inf",
        ],
        "fwd_h128": [
            "cake_rmsnorm_train_3808022dd402387eed16",
            "cake_rmsnorm_train_628267fc724c2c8f0e07__2_4",
            "cake_rmsnorm_train_3808022dd402387eed16__4_16384",
            "cake_rmsnorm_train_628267fc724c2c8f0e07",
        ],
        "fwd_w32_h128": [
            "cake_rmsnorm_train_513f4ffbfefd9d019863",
            "cake_rmsnorm_train_1ecf9c260e418bc70046__2_4",
            "cake_rmsnorm_train_513f4ffbfefd9d019863__4_64",
            "cake_rmsnorm_train_1ecf9c260e418bc70046__64_128",
            "cake_rmsnorm_train_513f4ffbfefd9d019863__128_16384",
            "cake_rmsnorm_train_1ecf9c260e418bc70046",
        ],
        "fwd_h1280": [
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5",
            "cake_rmsnorm_train_03a74dfe62df95c2a752__16_32",
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__32_256",
            "cake_rmsnorm_train_03a74dfe62df95c2a752__256_512",
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__512_8192",
            "cake_rmsnorm_train_39b9b41b16769252153d",
            "cake_rmsnorm_train_007e82b24390c88156ac__24576_32768",
            "cake_rmsnorm_train_03b5516039c533b453b9__32768_inf",
        ],
        "fwd_w32_h1280": [
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__0_1",
            "cake_rmsnorm_train_c43709cadf31ed04033c",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__4_8",
            "cake_rmsnorm_train_c43709cadf31ed04033c__8_16",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__16_256",
            "cake_rmsnorm_train_c43709cadf31ed04033c__256_1024",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58",
            "cake_rmsnorm_train_58c2fce1ea0b40758e91__2048_32768",
            "cake_rmsnorm_train_99b209a37f0203af57ce__32768_inf",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_train_21d3416baa877808371e",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41__low",
            "cake_rmsnorm_train_1974c5c86cee63f33370",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b44107e05796c2922187",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4__cut",
            "cake_rmsnorm_train_b44107e05796c2922187__band",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_train_0da8e18d4af432c6927a",
            "cake_rmsnorm_train_a4d3f82b3e776f4fc49b__cut",
            "cake_rmsnorm_train_0da8e18d4af432c6927a__band",
            "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        ],
        "fwd_h256": [
            "cake_rmsnorm_train_0489c784c13bb25499f6",
            "cake_rmsnorm_train_1b284baec9663d6de1b9",
            "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
            "cake_rmsnorm_train_1388ab779a35c4e41efb",
            "cake_rmsnorm_train_466331ff3cd2d6c62738",
            "cake_rmsnorm_train_ed9483d8c356affee55d",
            "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_train_f2e721db4c30900f12fe",
            "cake_rmsnorm_train_77456435c9fbd1d18fcf",
            "cake_rmsnorm_train_42d82d26e97c36651dae",
            "cake_rmsnorm_train_b0299f6e0881fb35f27b",
            "cake_rmsnorm_train_32bebe5a7782311d60a0",
            "cake_rmsnorm_train_d0a656f053396a58c49a",
            "cake_rmsnorm_train_670bfb3eed0ddf6ba461",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_train_4371270e4db4b985998f",
            "cake_rmsnorm_train_f0db3068d0d05442a59d__low",
            "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid",
            "cake_rmsnorm_train_f0db3068d0d05442a59d",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_8757f6da86abf54363d5",
            "cake_rmsnorm_train_7673978dc698790054d2",
            "cake_rmsnorm_train_3d20ab4a432bddf7349a__100",
            "cake_rmsnorm_train_3430e5d9c053309b0323__100",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_train_6769e393999bc2ba1926",
            "cake_rmsnorm_train_e2eabcb3b0915d916a13",
            "cake_rmsnorm_train_7da36ff557bb2bac8d89__100",
            "cake_rmsnorm_train_3c1e1f571f15d42e8d57__100",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__small",
            "cake_rmsnorm_train_b5e173f5ecc736d91109__mid",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__ov100_0_640",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_640_6144",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__ov100_6144_16384",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov100_16384_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace__100",
            "cake_rmsnorm_train_bad4db9153cc903b7ace__ps5",
            "cake_rmsnorm_train_288d7b1d51c340236cb7",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__100",
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ps5",
            "cake_rmsnorm_train_3d619cd1e3ae638cce3d__ov100_1024_6144",
            "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov100_6144_inf",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_308da18230017dadc02d",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_train_8e3407e059cfc005f421",
            "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_train_6325c474ad9955a319e9",
        ],
        "bwd_h128": [
            "cake_rmsnorm_train_9346600882e53a589509",
            "cake_rmsnorm_train_401a018c68bc082078b2",
            "cake_rmsnorm_train_d197bcc34403e94ee2bd__2048_8192",
            "cake_rmsnorm_train_8b0b86c827709e73ee26__8192_12288",
            "cake_rmsnorm_train_c50d3bbd794af91a2538",
            "cake_rmsnorm_train_8b0b86c827709e73ee26__16384_inf",
            "cake_rmsnorm_train_096e5efbf0a00905ee5f__2048_8192",
        ],
        "bwd_w32_h128": [
            "cake_rmsnorm_train_ab8a320c36b909e7bf1e",
            "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea__1024_1536",
            "cake_rmsnorm_train_a45d09aebf1bdca80e95__1536_2048",
            "cake_rmsnorm_train_e23083c7af388f9e2f50__2048_8192",
            "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
            "cake_rmsnorm_train_d0fe504e1d8611c0b2dc__12288_16384",
            "cake_rmsnorm_train_3e751b6f0ebbd6be9481__16384_inf",
            "cake_rmsnorm_train_096e5efbf0a00905ee5f__2048_8192",
        ],
        "bwd_h256": [
            "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
            "cake_rmsnorm_train_f2ccf7197a641c39acbf",
            "cake_rmsnorm_train_9fb6b410e91f9d92becf",
            "cake_rmsnorm_train_195c4317b51bb4ea1137",
            "cake_rmsnorm_train_44ef314181f3f2e93293__20480_92160",
            "cake_rmsnorm_train_a385ae1681a2c078344e__92160_524288",
            "cake_rmsnorm_train_44ef314181f3f2e93293",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__92160_524288",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_67cb56412bd46304b711",
            "cake_rmsnorm_train_aa745395d7542d8a8e3a",
            "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
            "cake_rmsnorm_train_8de0131445eed6eeabb7",
            "cake_rmsnorm_train_b71bbe9852c16fe03114__20480_92160",
            "cake_rmsnorm_train_420788e663a2c3a810f6__92160_864240",
            "cake_rmsnorm_train_b71bbe9852c16fe03114__864240_inf",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__92160_864240",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_64a299691c17079fe510",
            "cake_rmsnorm_train_c167e0d26b363cf19e7b",
            "cake_rmsnorm_train_931757c844a9d257c08b__16_227",
            "cake_rmsnorm_train_09601d42628de24f6f87__227_1024",
            "cake_rmsnorm_train_123db0b2f7fb38cfab03__1024_1280",
            "cake_rmsnorm_train_d0bde38aa443d42a5f04",
            "cake_rmsnorm_train_ad1a5a3c5f88e3d10036__4096_32768",
            "cake_rmsnorm_train_8877322a3002e3a4fefd__32768_inf",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_train_fef03ce50a792c8704bd__1024_1280",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553__32768_inf",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_8c261dfc6e34351d7a11",
            "cake_rmsnorm_train_88e6252beaf7d056d555",
            "cake_rmsnorm_train_404c8f54ab3129877350__16_227",
            "cake_rmsnorm_train_1a322bb1c550a7d70638__227_292",
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__292_960",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__292_960",
        ],
        "bwd_h1024": [
            "cake_rmsnorm_train_330930e9329d1e53aa5c",
            "cake_rmsnorm_train_28647b7ceb414d4fd02c__4_8",
            "cake_rmsnorm_train_89fa3e964f6ac26dc6a2__8_64",
            "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac",
            "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
            "cake_rmsnorm_train_0c581a133624ff396ab4",
            "cake_rmsnorm_train_af798d1034849cc692e9",
            "cake_rmsnorm_train_acd331db1201bf2fb44e__32768_inf",
            "cake_rmsnorm_train_3b5dc80e2313fd64010e",
            "cake_rmsnorm_train_176f79f9faa86f2fa441__32768_inf",
        ],
        "bwd_w32_h1024": [
            "cake_rmsnorm_train_81e742d5d2cf1e0f790a",
            "cake_rmsnorm_train_8501e95ea685c476cb50__4_8",
            "cake_rmsnorm_train_744d74299c2d45867c94",
            "cake_rmsnorm_train_561bea88a25c2f245c29",
            "cake_rmsnorm_train_1a45ade5011422a0d019",
            "cake_rmsnorm_train_1a45ade5011422a0d019__4096_12288",
            "cake_rmsnorm_train_1a45ade5011422a0d019__12288_32768",
            "cake_rmsnorm_train_b201ba67ac366efbd7be__32768_inf",
            "cake_rmsnorm_train_3b5dc80e2313fd64010e",
            "cake_rmsnorm_train_176f79f9faa86f2fa441__32768_inf",
        ],
        "bwd_h1280": [
            "cake_rmsnorm_train_8895b0224bb3d596a3a6",
            "cake_rmsnorm_train_a5f2942d61e591f57a76",
            "cake_rmsnorm_train_809235a3965d337a6877",
            "cake_rmsnorm_train_12bbedff93fcaafa475d",
            "cake_rmsnorm_train_c646e0976e08c6fe8a91",
            "cake_rmsnorm_train_b2cf87e64c9399e38407__8192_inf",
            "cake_rmsnorm_train_59474336746fb32d4e51",
            "cake_rmsnorm_train_59474336746fb32d4e51__4096_8192",
        ],
        "bwd_w32_h1280": [
            "cake_rmsnorm_train_103805d93149c8668264",
            "cake_rmsnorm_train_8d8ef930b7d650c5538a",
            "cake_rmsnorm_train_e476d0b2c0bd4285f16d",
            "cake_rmsnorm_train_80956d258172dfa76c2a",
            "cake_rmsnorm_train_f6956a41e69eb126197e",
            "cake_rmsnorm_train_59474336746fb32d4e51",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_db218c92c4c86b9da85a__0_2",
            "cake_rmsnorm_train_02e761bcf94053af6fc3__2_64",
            "cake_rmsnorm_train_dc478ac1195882aedee6__64_384",
            "cake_rmsnorm_train_7dc29629ff21625607f8__384_1024",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe__1024_1184",
            "cake_rmsnorm_train_38e32d90f62f7cfc7b51__1184_77824",
            "cake_rmsnorm_train_b457c4d134e44476dc6f__77824_78784",
            "cake_rmsnorm_train_0d5dc239536f113f4926",
            "cake_rmsnorm_train_38e32d90f62f7cfc7b51__131072_214016",
            "cake_rmsnorm_train_ae610afdf77ff6714a5b__214016_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6__1024_1184",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__1184_77824",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__77824_78784",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__131072_214016",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_2",
            "cake_rmsnorm_train_166e288c9192be6b994c__2_64",
            "cake_rmsnorm_train_e83ee45f66352eaef755__64_384",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__384_1024",
            "cake_rmsnorm_train_4b6981c10fd40a9d8aec__1024_1184",
            "cake_rmsnorm_train_e30c1899e859da4f4c68__1184_77824",
            "cake_rmsnorm_train_363eabb3661dab78f3d3__77824_131072",
            "cake_rmsnorm_train_e30c1899e859da4f4c68__131072_214016",
            "cake_rmsnorm_train_4b6981c10fd40a9d8aec__214016_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__384_1024",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__1184_77824",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__131072_214016",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_c5f62407b828a85f82ec",
            "cake_rmsnorm_train_a951623c157d070e85ab__4_12",
            "cake_rmsnorm_train_518cf16a7bf8e88042ee__12_64",
            "cake_rmsnorm_train_64e6aa82bb2c6d4ea978__64_336",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__336_3072",
            "cake_rmsnorm_train_9ac8d21e46a658e7913f",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__4096_7168",
            "cake_rmsnorm_train_0fa4f255caf709d09f66",
            "cake_rmsnorm_train_c73074cf44f07fcf3bb6__11264_16384",
            "cake_rmsnorm_train_ceeef043ac64f37fc529__16384_58880",
            "cake_rmsnorm_train_8404aa4380f6db7259cc",
            "cake_rmsnorm_train_0fa4f255caf709d09f66__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__336_3072",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__3072_4096",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_7168",
            "cake_rmsnorm_train_d0960be986e58f449f50__11264_16384",
            "cake_rmsnorm_train_d0960be986e58f449f50__16384_58880",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_0017ab41f68d8689ebc2",
            "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_14",
            "cake_rmsnorm_train_c2f08c957d4b0f17dddd__14_64",
            "cake_rmsnorm_train_9030dbeb948761ec2258__64_336",
            "cake_rmsnorm_train_17493054fed5409b6df9__336_1024",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__1024_4096",
            "cake_rmsnorm_train_b187e5989302def22b2d__4096_15403",
            "cake_rmsnorm_train_ba7dbf6a07744268705e",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__45056_58880",
            "cake_rmsnorm_train_7e6a0b631aea3edf51f9__58880_65536",
            "cake_rmsnorm_train_54f6e4b8d937f7fb5619",
            "cake_rmsnorm_train_ba7dbf6a07744268705e__131072_inf",
            "cake_rmsnorm_train_d0960be986e58f449f50__4096_15403",
            "cake_rmsnorm_train_d0960be986e58f449f50__58880_65536",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_945a0afdb82ab47eb643",
            "cake_rmsnorm_train_41f44b4bc0b765350a50__4_12",
            "cake_rmsnorm_train_82b84544dba8e2b70bc1__12_64",
            "cake_rmsnorm_train_4822793ea0c954c11af6__64_1152",
            "cake_rmsnorm_train_9a830956447d4e304914__1152_4096",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__4096_4608",
            "cake_rmsnorm_train_cf4c213e50482afdacdb__4608_16384",
            "cake_rmsnorm_train_681b33d9c9d0ac4d29a4__16384_36864",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__36864_44032",
            "cake_rmsnorm_train_35a0b0ca6b2060313255__44032_65536",
            "cake_rmsnorm_train_cda582bb216e6776b449",
            "cake_rmsnorm_train_15e157d2ee9080d206a9__131072_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1152_4096",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__4096_4608",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__44032_65536",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_f6bd1634a115bd281055",
            "cake_rmsnorm_train_b378ba007691c582171e__4_14",
            "cake_rmsnorm_train_cdef0b34f5080dbc61d1__14_64",
            "cake_rmsnorm_train_8616070567821eaa3e35__64_1792",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec__1792_3584",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec",
            "cake_rmsnorm_train_626aed743358c27513f1__36864_44032",
            "cake_rmsnorm_train_626aed743358c27513f1__44032_65536",
            "cake_rmsnorm_train_3a10d379e6a7ae778964",
            "cake_rmsnorm_train_f7573bcb748c0784f3ec__131072_inf",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__36864_44032",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__44032_65536",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_dbc680f356be4cd643de",
            "cake_rmsnorm_train_29bcbde1d271ea989fa8__4_64",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__64_767",
            "cake_rmsnorm_train_bfde9b88e800fc34722c__767_36864",
            "cake_rmsnorm_train_b482c84dbffcd75a4a4c__36864_56256",
            "cake_rmsnorm_train_670f4cafe8c34a429d30__56256_131072",
            "cake_rmsnorm_train_bfde9b88e800fc34722c__131072_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__767_36864",
            "cake_rmsnorm_train_c5831b29d17867842e5f__36864_56256",
            "cake_rmsnorm_train_c5831b29d17867842e5f__131072_inf",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_d47138228b123e55fe5d",
            "cake_rmsnorm_train_cb6539573725ac861f94__4_64",
            "cake_rmsnorm_train_139d162a8bb29d1b2d35__64_767",
            "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__767_36864",
            "cake_rmsnorm_train_9479b1298a1b19d4d996__36864_56256",
            "cake_rmsnorm_train_13bd0c5f56e37867eb26__56256_131072",
            "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__131072_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__767_36864",
            "cake_rmsnorm_train_c5831b29d17867842e5f__36864_56256",
            "cake_rmsnorm_train_c5831b29d17867842e5f__131072_inf",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_e37752b056dd4e4a0bee",
            "cake_rmsnorm_train_e6475d8fa504ef4d7d2d__4_64",
            "cake_rmsnorm_train_44508df7dab2ab688f85__64_1152",
            "cake_rmsnorm_train_5c804aed7ad14b99e1d0__1152_4096",
            "cake_rmsnorm_train_b93bf060d4d3defc8150__4096_42496",
            "cake_rmsnorm_train_9433061da18b2db4299d__42496_65536",
            "cake_rmsnorm_train_301133b646c4197fe462",
            "cake_rmsnorm_train_9433061da18b2db4299d__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1152_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_42496",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_deeb187630473818923d__0_2",
            "cake_rmsnorm_train_3a55646d574035bfa271__2_64",
            "cake_rmsnorm_train_864e84f3d874f850c091__64_1408",
            "cake_rmsnorm_train_f6537c86c7ffae183a50__1408_65536",
            "cake_rmsnorm_train_6feab384e4c85e893ed5",
            "cake_rmsnorm_train_3fbfdd750cbeb47032b5__131072_inf",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
            "cake_rmsnorm_train_f4d41a0991647b82ac21__4_64",
            "cake_rmsnorm_train_b3e6531e46667901ab91__64_1168",
            "cake_rmsnorm_train_41b80b6af97974feb181__1168_3584",
            "cake_rmsnorm_train_a562656cef5bf5dda160__3584_65536",
            "cake_rmsnorm_train_c61519fe43c45818b9b4",
            "cake_rmsnorm_train_a562656cef5bf5dda160__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1168_3584",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
            "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c__4_64",
            "cake_rmsnorm_train_71d36617003ab7742a77__64_1408",
            "cake_rmsnorm_train_a640d0b799889f785ad5__1408_65536",
            "cake_rmsnorm_train_f5453c09151f9d441744",
            "cake_rmsnorm_train_a640d0b799889f785ad5__131072_inf",
        ],
    },
    "sm_103a": {
        "fwd_h1024": [
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__0_2048",
            "cake_rmsnorm_train_9e8a70075f28be558bba",
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e",
            "cake_rmsnorm_train_9e8a70075f28be558bba__32768_inf",
        ],
        "fwd_w32_h1024": [
            "cake_rmsnorm_train_364ed18ca31e5b754d4e",
            "cake_rmsnorm_train_d99009bfce4043a5dffe",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__4_32",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__32_64",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__64_512",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__512_1024",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__1024_32768",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__32768_inf",
        ],
        "fwd_h128": [
            "cake_rmsnorm_train_3808022dd402387eed16__0_16384",
            "cake_rmsnorm_train_628267fc724c2c8f0e07",
        ],
        "fwd_w32_h128": [
            "cake_rmsnorm_train_513f4ffbfefd9d019863__0_16384",
            "cake_rmsnorm_train_1ecf9c260e418bc70046",
        ],
        "fwd_h1280": [
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__0_256",
            "cake_rmsnorm_train_03a74dfe62df95c2a752__256_1024",
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__1024_8192",
            "cake_rmsnorm_train_39b9b41b16769252153d",
            "cake_rmsnorm_train_007e82b24390c88156ac__24576_inf",
        ],
        "fwd_w32_h1280": [
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__0_512",
            "cake_rmsnorm_train_c43709cadf31ed04033c__512_1024",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58",
            "cake_rmsnorm_train_99b209a37f0203af57ce",
            "cake_rmsnorm_train_62e8e2756e015c39bf5c__8192_24576",
            "cake_rmsnorm_train_58c2fce1ea0b40758e91__24576_32768",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__32768_inf",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9__low",
            "cake_rmsnorm_train_7de1e51089d432dbc26b",
            "cake_rmsnorm_train_2fdbab2af48db42cd2a9",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_train_21d3416baa877808371e",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41__low",
            "cake_rmsnorm_train_1974c5c86cee63f33370",
            "cake_rmsnorm_train_29d0ae48de6b86aa4a41",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_c5a8099e177e463daab9",
            "cake_rmsnorm_train_b44107e05796c2922187__mid",
            "cake_rmsnorm_train_780b9833fd0a236ef0e4",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_train_01f0ebf3db9fa2c0884e",
            "cake_rmsnorm_train_0da8e18d4af432c6927a__mid",
            "cake_rmsnorm_train_a4d3f82b3e776f4fc49b",
        ],
        "fwd_h256": [
            "cake_rmsnorm_train_0489c784c13bb25499f6",
            "cake_rmsnorm_train_bbb05314930d192dfe7b",
            "cake_rmsnorm_train_1b284baec9663d6de1b9__20480_40960",
            "cake_rmsnorm_train_86d51eff64fc9fe57cf0",
            "cake_rmsnorm_train_466331ff3cd2d6c62738__81920_114688",
            "cake_rmsnorm_train_86d51eff64fc9fe57cf0__114688_196608",
            "cake_rmsnorm_train_ed9483d8c356affee55d",
            "cake_rmsnorm_train_37f663183cbff23c2893",
            "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5__368640_inf",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_train_f2e721db4c30900f12fe",
            "cake_rmsnorm_train_a8fffd93682c5e382dc5",
            "cake_rmsnorm_train_77456435c9fbd1d18fcf__20480_40960",
            "cake_rmsnorm_train_42d82d26e97c36651dae",
            "cake_rmsnorm_train_32bebe5a7782311d60a0__81920_114688",
            "cake_rmsnorm_train_42d82d26e97c36651dae__114688_196608",
            "cake_rmsnorm_train_d0a656f053396a58c49a",
            "cake_rmsnorm_train_18e3c0a5204006b3a56f",
            "cake_rmsnorm_train_670bfb3eed0ddf6ba461__368640_inf",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid103",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__wide103",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_train_4371270e4db4b985998f",
            "cake_rmsnorm_train_f0db3068d0d05442a59d__low",
            "cake_rmsnorm_train_bdc524a233d2bc6e9571__mid103",
            "cake_rmsnorm_train_bdc524a233d2bc6e9571__wide103",
            "cake_rmsnorm_train_f0db3068d0d05442a59d",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_8757f6da86abf54363d5",
            "cake_rmsnorm_train_7673978dc698790054d2",
            "cake_rmsnorm_train_3d20ab4a432bddf7349a",
            "cake_rmsnorm_train_3430e5d9c053309b0323",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_train_6769e393999bc2ba1926",
            "cake_rmsnorm_train_e2eabcb3b0915d916a13",
            "cake_rmsnorm_train_7da36ff557bb2bac8d89",
            "cake_rmsnorm_train_3c1e1f571f15d42e8d57",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__small",
            "cake_rmsnorm_train_b5e173f5ecc736d91109__mid",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_train_1defffdf2712cae36784__ov103_0_640",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_640_6144",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__ov103_6144_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_train_288d7b1d51c340236cb7",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6__ov103_1024_6144",
            "cake_rmsnorm_train_5f89e5fd2f8a6057997e__ov103_6144_inf",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_308da18230017dadc02d",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_train_8e3407e059cfc005f421",
            "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_train_6325c474ad9955a319e9",
        ],
        "bwd_h128": [
            "cake_rmsnorm_train_9346600882e53a589509",
            "cake_rmsnorm_train_401a018c68bc082078b2",
            "cake_rmsnorm_train_d197bcc34403e94ee2bd__2048_8192",
            "cake_rmsnorm_train_8b0b86c827709e73ee26__8192_12288",
            "cake_rmsnorm_train_c50d3bbd794af91a2538",
            "cake_rmsnorm_train_8b0b86c827709e73ee26__16384_inf",
            "cake_rmsnorm_train_096e5efbf0a00905ee5f__2048_8192",
        ],
        "bwd_w32_h128": [
            "cake_rmsnorm_train_ab8a320c36b909e7bf1e",
            "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea",
            "cake_rmsnorm_train_e23083c7af388f9e2f50__2048_8192",
            "cake_rmsnorm_train_3e751b6f0ebbd6be9481",
            "cake_rmsnorm_train_d0fe504e1d8611c0b2dc__12288_16384",
            "cake_rmsnorm_train_3e751b6f0ebbd6be9481__16384_inf",
            "cake_rmsnorm_train_096e5efbf0a00905ee5f__2048_8192",
        ],
        "bwd_h256": [
            "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
            "cake_rmsnorm_train_f2ccf7197a641c39acbf",
            "cake_rmsnorm_train_9fb6b410e91f9d92becf",
            "cake_rmsnorm_train_a385ae1681a2c078344e__64_1080",
            "cake_rmsnorm_train_af005462cd6a301f31c4__1080_2048",
            "cake_rmsnorm_train_6bdbfe619b8270ff9a21",
            "cake_rmsnorm_train_a385ae1681a2c078344e__262144_720896",
            "cake_rmsnorm_train_a385ae1681a2c078344e__720896_1572864",
            "cake_rmsnorm_train_a385ae1681a2c078344e",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__64_1080",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_2048",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__2048_262144",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__262144_720896",
            "cake_rmsnorm_train_72138dce40d43da44aaf__720896_1572864",
            "cake_rmsnorm_train_72138dce40d43da44aaf",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_67cb56412bd46304b711",
            "cake_rmsnorm_train_aa745395d7542d8a8e3a",
            "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
            "cake_rmsnorm_train_420788e663a2c3a810f6__64_1080",
            "cake_rmsnorm_train_b7d9577ff884de98ea06__1080_2048",
            "cake_rmsnorm_train_6554d4eb4362141cb058",
            "cake_rmsnorm_train_420788e663a2c3a810f6__262144_720896",
            "cake_rmsnorm_train_420788e663a2c3a810f6__720896_1572864",
            "cake_rmsnorm_train_420788e663a2c3a810f6",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__64_1080",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__1080_2048",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__2048_262144",
            "cake_rmsnorm_train_affa1b95f68bab8c411b__262144_720896",
            "cake_rmsnorm_train_72138dce40d43da44aaf__720896_1572864",
            "cake_rmsnorm_train_72138dce40d43da44aaf",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_64a299691c17079fe510",
            "cake_rmsnorm_train_c167e0d26b363cf19e7b",
            "cake_rmsnorm_train_bdfd77893d8f9db3c9bb__16_64",
            "cake_rmsnorm_train_09601d42628de24f6f87__64_1024",
            "cake_rmsnorm_train_8e7f4d811e55a351a22e",
            "cake_rmsnorm_train_274b0055aace183f2367",
            "cake_rmsnorm_train_cfaae3cec98f8501832b",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__64_1024",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__1024_4096",
            "cake_rmsnorm_train_122721bd6b4650866553__4096_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_8c261dfc6e34351d7a11",
            "cake_rmsnorm_train_88e6252beaf7d056d555",
            "cake_rmsnorm_train_2c60470d16e0a7fb6089__16_227",
            "cake_rmsnorm_train_9f07cea1d13b923ed91e__227_1024",
            "cake_rmsnorm_train_56ee8e66479625fd078a__1024_32768",
            "cake_rmsnorm_train_a546f410ec8cf03ef20d",
            "cake_rmsnorm_train_7097069b8eea6ba44d5a__227_1024",
            "cake_rmsnorm_train_122721bd6b4650866553__1024_32768",
            "cake_rmsnorm_train_122721bd6b4650866553",
        ],
        "bwd_h1024": [
            "cake_rmsnorm_train_330930e9329d1e53aa5c",
            "cake_rmsnorm_train_89fa3e964f6ac26dc6a2__4_64",
            "cake_rmsnorm_train_5ad5f0962d5c5e8f63ac",
            "cake_rmsnorm_train_89fa3e964f6ac26dc6a2",
            "cake_rmsnorm_train_5e32dd889d72a75e348b",
            "cake_rmsnorm_train_33d9025920659f515c9f",
            "cake_rmsnorm_train_af798d1034849cc692e9",
            "cake_rmsnorm_train_acd331db1201bf2fb44e__32768_inf",
            "cake_rmsnorm_train_3b5dc80e2313fd64010e__1024_1536",
            "cake_rmsnorm_train_176f79f9faa86f2fa441__32768_inf",
        ],
        "bwd_w32_h1024": [
            "cake_rmsnorm_train_81e742d5d2cf1e0f790a",
            "cake_rmsnorm_train_8501e95ea685c476cb50__4_8",
            "cake_rmsnorm_train_744d74299c2d45867c94",
            "cake_rmsnorm_train_ba8feeec37d2bdde1fca",
            "cake_rmsnorm_train_1a45ade5011422a0d019__1536_2048",
            "cake_rmsnorm_train_1a45ade5011422a0d019",
            "cake_rmsnorm_train_1a45ade5011422a0d019__4096_12288",
            "cake_rmsnorm_train_1a45ade5011422a0d019__12288_32768",
            "cake_rmsnorm_train_1a45ade5011422a0d019__32768_131072",
            "cake_rmsnorm_train_2980f513dd6581ec98e7__131072_inf",
            "cake_rmsnorm_train_3b5dc80e2313fd64010e__1024_1536",
        ],
        "bwd_h1280": [
            "cake_rmsnorm_train_8895b0224bb3d596a3a6",
            "cake_rmsnorm_train_a5f2942d61e591f57a76",
            "cake_rmsnorm_train_809235a3965d337a6877",
            "cake_rmsnorm_train_12bbedff93fcaafa475d__512_2048",
            "cake_rmsnorm_train_b2cf87e64c9399e38407",
            "cake_rmsnorm_train_59474336746fb32d4e51__512_2048",
        ],
        "bwd_w32_h1280": [
            "cake_rmsnorm_train_103805d93149c8668264",
            "cake_rmsnorm_train_8d8ef930b7d650c5538a",
            "cake_rmsnorm_train_e476d0b2c0bd4285f16d",
            "cake_rmsnorm_train_80956d258172dfa76c2a",
            "cake_rmsnorm_train_f6956a41e69eb126197e",
            "cake_rmsnorm_train_59474336746fb32d4e51",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_db218c92c4c86b9da85a__0_4",
            "cake_rmsnorm_train_c2a523de1ff578621a88__4_5",
            "cake_rmsnorm_train_02e761bcf94053af6fc3__0_64_103",
            "cake_rmsnorm_train_4aef8d1ff2c1c2af7b80__64_214",
            "cake_rmsnorm_train_7dc29629ff21625607f8__214_1024",
            "cake_rmsnorm_train_c1a9efdf06c3087b4efe__1024_1216",
            "cake_rmsnorm_train_38e32d90f62f7cfc7b51__1216_4096",
            "cake_rmsnorm_train_38e32d90f62f7cfc7b51__4096_46592",
            "cake_rmsnorm_train_20d7a230ed8dc161a93a__46592_53184",
            "cake_rmsnorm_train_38e32d90f62f7cfc7b51__53184_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__214_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6__1024_1216",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__1216_4096",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_46592",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__53184_inf",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_4",
            "cake_rmsnorm_train_744aa77460c886f4f429__4_6",
            "cake_rmsnorm_train_166e288c9192be6b994c__0_64_103",
            "cake_rmsnorm_train_698c3d36d67408394ccf__64_214",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__214_1024",
            "cake_rmsnorm_train_17d9cdfd157ba4b86c2b__1024_1216",
            "cake_rmsnorm_train_e30c1899e859da4f4c68__1216_4096",
            "cake_rmsnorm_train_e30c1899e859da4f4c68__4096_46592",
            "cake_rmsnorm_train_51f46ff164b142ec8888__46592_53184",
            "cake_rmsnorm_train_e30c1899e859da4f4c68__53184_inf",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__214_1024",
            "cake_rmsnorm_train_1863320f3158d35eead6__1024_1216",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__1216_4096",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__4096_46592",
            "cake_rmsnorm_train_1863320f3158d35eead6__46592_53184",
            "cake_rmsnorm_train_9b727c2c9933b2ca6724__53184_inf",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_c5f62407b828a85f82ec",
            "cake_rmsnorm_train_a951623c157d070e85ab__4_16",
            "cake_rmsnorm_train_518cf16a7bf8e88042ee__16_63",
            "cake_rmsnorm_train_ed36a8d998abba41f227__63_171",
            "cake_rmsnorm_train_cc26e316ae40d683a118__171_16384",
            "cake_rmsnorm_train_caf6978ed0a17fbf65b9__16384_32768",
            "cake_rmsnorm_train_302f64db6abe79e5259e__32768_58496",
            "cake_rmsnorm_train_302f64db6abe79e5259e__58496_61952",
            "cake_rmsnorm_train_302f64db6abe79e5259e__61952_68160",
            "cake_rmsnorm_train_87532fb038ad204517e1__68160_131072",
            "cake_rmsnorm_train_33e5c382e06419d65390__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__171_16384",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496",
            "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952",
            "cake_rmsnorm_train_d0960be986e58f449f50__61952_68160",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_0017ab41f68d8689ebc2",
            "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_16",
            "cake_rmsnorm_train_c2f08c957d4b0f17dddd__0_64_103",
            "cake_rmsnorm_train_978741b43372951cb5b9__64_171",
            "cake_rmsnorm_train_c83c314c6dcf28ad7369__171_16384",
            "cake_rmsnorm_train_07f9f2fbea7163639c27__16384_32768",
            "cake_rmsnorm_train_e98cb156abfe1666c76f__32768_58496",
            "cake_rmsnorm_train_e98cb156abfe1666c76f__58496_61952",
            "cake_rmsnorm_train_e98cb156abfe1666c76f__61952_68160",
            "cake_rmsnorm_train_149b129cdd1468145b13__68160_131072",
            "cake_rmsnorm_train_1166d499d4e846fcb2d6__131072_inf",
            "cake_rmsnorm_train_880768a0a264e04dbfb5__171_16384",
            "cake_rmsnorm_train_596b457aaba073f8d69f",
            "cake_rmsnorm_train_d0960be986e58f449f50__32768_58496",
            "cake_rmsnorm_train_d0960be986e58f449f50__58496_61952",
            "cake_rmsnorm_train_d0960be986e58f449f50__61952_68160",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_945a0afdb82ab47eb643",
            "cake_rmsnorm_train_41f44b4bc0b765350a50__4_16",
            "cake_rmsnorm_train_dc7ed206fba49ab22d1d__16_32",
            "cake_rmsnorm_train_12d53fe5b85b212fc800__32_152",
            "cake_rmsnorm_train_50044ee3bd1ca54bd821__152_256",
            "cake_rmsnorm_train_ad2bb26ea5916baef963",
            "cake_rmsnorm_train_5572b2671862fe670ef4__1024_20480",
            "cake_rmsnorm_train_ef9f8d46c22e4af7aef1__20480_65536",
            "cake_rmsnorm_train_f0aebb74650c8dc3f128__65536_262144",
            "cake_rmsnorm_train_35888fb3b729e1f8e239__262144_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__152_256",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_20480",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_f6bd1634a115bd281055",
            "cake_rmsnorm_train_b378ba007691c582171e__4_16",
            "cake_rmsnorm_train_8274f36f6fd2f50dd508__16_32",
            "cake_rmsnorm_train_27615d0e36c9f2fe0856__32_152",
            "cake_rmsnorm_train_3fd77c8ce97e8a523abb__152_256",
            "cake_rmsnorm_train_16545056022785f2a7d8",
            "cake_rmsnorm_train_47a01388444960da7aae__1024_20480",
            "cake_rmsnorm_train_626aed743358c27513f1__20480_48640",
            "cake_rmsnorm_train_a8eb754739a1589de3ec__48640_65536",
            "cake_rmsnorm_train_bd22da7e2d4914167050__65536_262144",
            "cake_rmsnorm_train_36790f79b8845172d188__262144_inf",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__152_256",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__256_1024",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__1024_20480",
            "cake_rmsnorm_train_fcbfa14d591a9babfd26__20480_48640",
            "cake_rmsnorm_train_5d75ea5f3c25579af496__48640_65536",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_dbc680f356be4cd643de",
            "cake_rmsnorm_train_29bcbde1d271ea989fa8__4_16",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__16_64",
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__64_192",
            "cake_rmsnorm_train_bfde9b88e800fc34722c__192_1024",
            "cake_rmsnorm_train_bfde9b88e800fc34722c__1024_36301",
            "cake_rmsnorm_train_2fc4932fbde0bc848b32__36301_61952",
            "cake_rmsnorm_train_e21acade55e05f342053__61952_393216",
            "cake_rmsnorm_train_826e6a14a93bd988be4d__393216_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__64_192",
            "cake_rmsnorm_train_c5831b29d17867842e5f__192_1024",
            "cake_rmsnorm_train_c5831b29d17867842e5f__1024_36301",
            "cake_rmsnorm_train_c5831b29d17867842e5f__36301_61952",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_d47138228b123e55fe5d",
            "cake_rmsnorm_train_05396cd3a077d3772fdc__4_5",
            "cake_rmsnorm_train_cb6539573725ac861f94__5_16",
            "cake_rmsnorm_train_139d162a8bb29d1b2d35__16_192",
            "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__192_1024",
            "cake_rmsnorm_train_f9b2d42c4fc7a5a1aa48__1024_36301",
            "cake_rmsnorm_train_79811a76205d7d7dd197__36301_61952",
            "cake_rmsnorm_train_05078a174a09119cfe3c__61952_196608",
            "cake_rmsnorm_train_91a40f3094182a35c784__196608_inf",
            "cake_rmsnorm_train_c5831b29d17867842e5f__192_1024",
            "cake_rmsnorm_train_c5831b29d17867842e5f__1024_36301",
            "cake_rmsnorm_train_c5831b29d17867842e5f__36301_61952",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_e37752b056dd4e4a0bee",
            "cake_rmsnorm_train_fdfb73350c1a926a4929__4_6",
            "cake_rmsnorm_train_e6475d8fa504ef4d7d2d__6_16",
            "cake_rmsnorm_train_44508df7dab2ab688f85__16_896",
            "cake_rmsnorm_train_61421bc72e57d14d8a32__896_4096",
            "cake_rmsnorm_train_322574af704b51aef0bb__4096_24575",
            "cake_rmsnorm_train_455174e1a3be71c79452__24575_65536",
            "cake_rmsnorm_train_850fba46397a84b8cd7b__65536_131072",
            "cake_rmsnorm_train_38337af146209d6a60ea__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__896_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_24575",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_deeb187630473818923d__0_4",
            "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_6",
            "cake_rmsnorm_train_3a55646d574035bfa271__6_31",
            "cake_rmsnorm_train_864e84f3d874f850c091__31_32",
            "cake_rmsnorm_train_3a55646d574035bfa271__32_48",
            "cake_rmsnorm_train_864e84f3d874f850c091__48_56",
            "cake_rmsnorm_train_864e84f3d874f850c091__56_304",
            "cake_rmsnorm_train_864e84f3d874f850c091__304_1920",
            "cake_rmsnorm_train_aac3ebbf0e11c453c472__1920_131072",
            "cake_rmsnorm_train_115dcc719a6de4b079f1",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
            "cake_rmsnorm_train_18b8b8054c7053942bcd__4_12",
            "cake_rmsnorm_train_f4d41a0991647b82ac21__12_16",
            "cake_rmsnorm_train_b3e6531e46667901ab91__16_1024",
            "cake_rmsnorm_train_41b80b6af97974feb181__1024_4096",
            "cake_rmsnorm_train_9a8c0396edd85d4d545e__4096_49152",
            "cake_rmsnorm_train_9a8c0396edd85d4d545e__49152_83392",
            "cake_rmsnorm_train_cb1aee125561ad8d4e29__83392_131072",
            "cake_rmsnorm_train_5a635713b08645d57f48__131072_inf",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__1024_4096",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__4096_49152",
            "cake_rmsnorm_train_5f5cdc5151e3c0620bcd__49152_83392",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
            "cake_rmsnorm_train_c29a8f856283578103dc__4_10",
            "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c__10_31",
            "cake_rmsnorm_train_71d36617003ab7742a77__31_32",
            "cake_rmsnorm_train_6e0d06ab3a746b6a7a4c__32_48",
            "cake_rmsnorm_train_71d36617003ab7742a77__48_320",
            "cake_rmsnorm_train_71d36617003ab7742a77__320_1824",
            "cake_rmsnorm_train_5fbcc18d24f37f0ba607__1824_131072",
            "cake_rmsnorm_train_3c390339fa8c82311a0a",
        ],
    },
    "sm_107a": {
        "fwd_h1024": [
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__0_8",
            "cake_rmsnorm_train_9e8a70075f28be558bba__8_16",
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__16_256",
            "cake_rmsnorm_train_9e8a70075f28be558bba__256_1024",
            "cake_rmsnorm_train_6e5ca1f45eefb79bee7e__1024_4096",
            "cake_rmsnorm_train_9e8a70075f28be558bba__4096_8192",
            "cake_rmsnorm_train_cffbc5c8de9c874b207c__8192_8447",
            "cake_rmsnorm_train_547372db70c0ca0be362__8447_13312",
            "cake_rmsnorm_train_9e8a70075f28be558bba__13312_32768",
            "cake_rmsnorm_train_cffbc5c8de9c874b207c__32768_inf",
        ],
        "fwd_w32_h1024": [
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__0_8",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__8_16",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__16_512",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__512_1536",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__1536_4096",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__4096_8192",
            "cake_rmsnorm_train_f5110367098beea13130",
            "cake_rmsnorm_train_364ed18ca31e5b754d4e__12288_16384",
            "cake_rmsnorm_train_8bd2d83654da4780c48a__16384_16895",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__16895_24576",
            "cake_rmsnorm_train_d99009bfce4043a5dffe__24576_32768",
            "cake_rmsnorm_train_008fb179743126fb2d35",
        ],
        "fwd_h128": [
            "cake_rmsnorm_train_3808022dd402387eed16__0_32",
            "cake_rmsnorm_train_628267fc724c2c8f0e07__32_64",
            "cake_rmsnorm_train_3808022dd402387eed16__64_2048",
            "cake_rmsnorm_train_7221de56ded18727adaf",
            "cake_rmsnorm_train_3808022dd402387eed16__4096_24576",
            "cake_rmsnorm_train_7221de56ded18727adaf__24576_32768",
            "cake_rmsnorm_train_a05a2262fd5fa6912442",
        ],
        "fwd_w32_h128": [
            "cake_rmsnorm_train_513f4ffbfefd9d019863__0_32",
            "cake_rmsnorm_train_1ecf9c260e418bc70046__32_64",
            "cake_rmsnorm_train_513f4ffbfefd9d019863__64_2048",
            "cake_rmsnorm_train_04ea6484fdf4a9b14ee0",
            "cake_rmsnorm_train_513f4ffbfefd9d019863__4096_16384",
            "cake_rmsnorm_train_1ecf9c260e418bc70046__16384_32768",
            "cake_rmsnorm_train_67e0dfc12fbae0354e51",
        ],
        "fwd_h1280": [
            "cake_rmsnorm_train_03a74dfe62df95c2a752",
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__1_4",
            "cake_rmsnorm_train_03a74dfe62df95c2a752__4_8",
            "cake_rmsnorm_train_007e82b24390c88156ac",
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__16_128",
            "cake_rmsnorm_train_03a74dfe62df95c2a752__128_512",
            "cake_rmsnorm_train_007e82b24390c88156ac__512_1024",
            "cake_rmsnorm_train_03b5516039c533b453b9",
            "cake_rmsnorm_train_4d9988d4680c7c3c02a5__1536_2048",
            "cake_rmsnorm_train_03a74dfe62df95c2a752__2048_4096",
            "cake_rmsnorm_train_007e82b24390c88156ac__4096_12288",
            "cake_rmsnorm_train_03b5516039c533b453b9__12288_16384",
            "cake_rmsnorm_train_007e82b24390c88156ac__16384_inf",
        ],
        "fwd_w32_h1280": [
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__0_4",
            "cake_rmsnorm_train_c43709cadf31ed04033c__4_8",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__8_128",
            "cake_rmsnorm_train_c43709cadf31ed04033c__128_512",
            "cake_rmsnorm_train_58c2fce1ea0b40758e91",
            "cake_rmsnorm_train_62e8e2756e015c39bf5c",
            "cake_rmsnorm_train_6258b6dac6840cc6bd58__1536_2048",
            "cake_rmsnorm_train_62e8e2756e015c39bf5c__2048_8192",
            "cake_rmsnorm_train_99b209a37f0203af57ce__8192_8703",
            "cake_rmsnorm_train_51ca3c875efd4b1a5746__8703_9984",
            "cake_rmsnorm_train_99b209a37f0203af57ce__9984_16384",
            "cake_rmsnorm_train_58c2fce1ea0b40758e91__16384_inf",
        ],
        "fwd_h1536": [
            "cake_rmsnorm_train_1ea88523077fc340e55f__107",
            "cake_rmsnorm_train_4cdc4c495b3f59ff5069",
            "cake_rmsnorm_train_682675bda578f5b7a312",
        ],
        "fwd_w32_h1536": [
            "cake_rmsnorm_train_21d3416baa877808371e__107",
            "cake_rmsnorm_train_92cf7b87f1ebffb1b12b",
            "cake_rmsnorm_train_a30e5a3b69b7c183992f",
        ],
        "fwd_h2048": [
            "cake_rmsnorm_train_b44107e05796c2922187__small",
            "cake_rmsnorm_train_a4da85136fba4588239f",
            "cake_rmsnorm_train_0ad4d53c4a0df501a99d",
            "cake_rmsnorm_train_0ad4d53c4a0df501a99d__wide",
            "cake_rmsnorm_train_bde56bd5a68c493f07a2",
        ],
        "fwd_w32_h2048": [
            "cake_rmsnorm_train_0da8e18d4af432c6927a__small",
            "cake_rmsnorm_train_a72c6ece1900f748eaa2",
            "cake_rmsnorm_train_595f506004038a65cc63",
            "cake_rmsnorm_train_40210950f2e7ee48bc3e__ov107_12288_16384",
            "cake_rmsnorm_train_40210950f2e7ee48bc3e",
        ],
        "fwd_h256": [
            "cake_rmsnorm_train_c5ca409f44e169148301",
            "cake_rmsnorm_train_bbb05314930d192dfe7b__6144_32768",
            "cake_rmsnorm_train_466331ff3cd2d6c62738__32768_98304",
            "cake_rmsnorm_train_37f663183cbff23c2893__98304_inf",
        ],
        "fwd_w32_h256": [
            "cake_rmsnorm_train_8f2756937eaac6b02502",
            "cake_rmsnorm_train_a8fffd93682c5e382dc5__6144_32768",
            "cake_rmsnorm_train_32bebe5a7782311d60a0__32768_98304",
            "cake_rmsnorm_train_18e3c0a5204006b3a56f__98304_inf",
        ],
        "fwd_h4096": [
            "cake_rmsnorm_train_8392d1a008d58a006bba__107",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__low107",
            "cake_rmsnorm_train_d447b7f6398b60fe5daa__mid107",
            "cake_rmsnorm_train_bf933e1660cab8e5787b",
            "cake_rmsnorm_train_bf933e1660cab8e5787b__wide",
            "cake_rmsnorm_train_fd913ecd254a6bba4b18__107",
        ],
        "fwd_w32_h4096": [
            "cake_rmsnorm_train_4371270e4db4b985998f__107",
            "cake_rmsnorm_train_f0db3068d0d05442a59d__low107",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_1024_4096",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_4096_20480",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_20480_49152",
            "cake_rmsnorm_train_f9f3423be0bd8ff8c794__ov107_49152_inf",
        ],
        "fwd_h512": [
            "cake_rmsnorm_train_7673978dc698790054d2__quad",
            "cake_rmsnorm_train_2a42e5c02b240f8709b1",
            "cake_rmsnorm_train_60f47c5d00570ecf7740",
        ],
        "fwd_w32_h512": [
            "cake_rmsnorm_train_e2eabcb3b0915d916a13__quad",
            "cake_rmsnorm_train_79fecc7016e44e940733",
            "cake_rmsnorm_train_61accbcc02891615f286",
        ],
        "fwd_h5120": [
            "cake_rmsnorm_train_fd1097c45c1e328871fd__107",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__low107",
            "cake_rmsnorm_train_b5e173f5ecc736d91109__mid107",
            "cake_rmsnorm_train_044d4a2aa0553c16db06__large107",
            "cake_rmsnorm_train_044d4a2aa0553c16db06__wide107",
            "cake_rmsnorm_train_5eef28cd35498e9abfa5__107",
        ],
        "fwd_w32_h5120": [
            "cake_rmsnorm_train_ba28d2c0c2c81a2c714e__107",
            "cake_rmsnorm_train_1defffdf2712cae36784__low107",
            "cake_rmsnorm_train_9ecb62c2640d571e50a8__mid107",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__large107",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__wide107",
            "cake_rmsnorm_train_b49de9b47372fa4e1051__ov107_49152_inf",
        ],
        "fwd_h6144": [
            "cake_rmsnorm_train_bad4db9153cc903b7ace",
            "cake_rmsnorm_train_cf1d05546951bdbd9f5d",
            "cake_rmsnorm_train_e7f4221a7c26b9dfdd23",
        ],
        "fwd_w32_h6144": [
            "cake_rmsnorm_train_2763cd5ebcc501c9ddb6",
            "cake_rmsnorm_train_3d619cd1e3ae638cce3d",
            "cake_rmsnorm_train_fdfa500c83d95df0f5c1",
        ],
        "fwd_residual_h6144": [
            "cake_rmsnorm_train_e646bbdbd852884953da__small",
            "cake_rmsnorm_train_03bdf5667404141e1eaf",
            "cake_rmsnorm_train_e646bbdbd852884953da",
        ],
        "fwd_w32_residual_h6144": [
            "cake_rmsnorm_train_6325c474ad9955a319e9__small",
            "cake_rmsnorm_train_e5a3aa2548fc41ee0485",
            "cake_rmsnorm_train_6325c474ad9955a319e9",
        ],
        "bwd_h128": [
            "cake_rmsnorm_train_9346600882e53a589509",
            "cake_rmsnorm_train_401a018c68bc082078b2__1024_2047",
            "cake_rmsnorm_train_d197bcc34403e94ee2bd__2047_65536",
            "cake_rmsnorm_train_8b0b86c827709e73ee26__65536_inf",
            "cake_rmsnorm_train_096e5efbf0a00905ee5f__2047_65536",
        ],
        "bwd_w32_h128": [
            "cake_rmsnorm_train_ab8a320c36b909e7bf1e",
            "cake_rmsnorm_train_b5f8c6c9e6ae65a17cea__1024_2047",
            "cake_rmsnorm_train_e23083c7af388f9e2f50__2047_65536",
            "cake_rmsnorm_train_3e751b6f0ebbd6be9481__107_65536_inf",
            "cake_rmsnorm_train_096e5efbf0a00905ee5f__2047_65536",
        ],
        "bwd_h256": [
            "cake_rmsnorm_train_780c64c6f2c4615bfbcf",
            "cake_rmsnorm_train_f2ccf7197a641c39acbf",
            "cake_rmsnorm_train_9fb6b410e91f9d92becf",
            "cake_rmsnorm_train_a385ae1681a2c078344e__64_786432",
            "cake_rmsnorm_train_44ef314181f3f2e93293__786432_inf",
            "cake_rmsnorm_train_312bbe084eac69d858c4__64_786432",
        ],
        "bwd_w32_h256": [
            "cake_rmsnorm_train_67cb56412bd46304b711",
            "cake_rmsnorm_train_aa745395d7542d8a8e3a",
            "cake_rmsnorm_train_6a0114cb0b27a780a4c3",
            "cake_rmsnorm_train_420788e663a2c3a810f6__64_786432",
            "cake_rmsnorm_train_b71bbe9852c16fe03114__786432_inf",
            "cake_rmsnorm_train_312bbe084eac69d858c4__64_786432",
        ],
        "bwd_h512": [
            "cake_rmsnorm_train_64a299691c17079fe510",
            "cake_rmsnorm_train_c167e0d26b363cf19e7b",
            "cake_rmsnorm_train_49af1e75185ddd637b76__16_576",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__576_1024",
            "cake_rmsnorm_train_3feed5026e1156d2fba2__1024_4096",
            "cake_rmsnorm_train_dfaf378aa8ad93e68bbd",
            "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3__16384_49152",
            "cake_rmsnorm_train_7764c8a34009f3728312__49152_inf",
            "cake_rmsnorm_train_f94861c36a14eb75e39c__576_1024",
            "cake_rmsnorm_train_d75459d901441384398b",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152",
        ],
        "bwd_w32_h512": [
            "cake_rmsnorm_train_8c261dfc6e34351d7a11",
            "cake_rmsnorm_train_88e6252beaf7d056d555",
            "cake_rmsnorm_train_8f29656c0018a749a588__16_384",
            "cake_rmsnorm_train_b841d64bbcaefff5ce7b__384_1024",
            "cake_rmsnorm_train_fd08c6f436eb6071fa22__1024_4096",
            "cake_rmsnorm_train_383f709cd5c4fadd9436",
            "cake_rmsnorm_train_03efa681eb38f59b00c2__16384_49152",
            "cake_rmsnorm_train_5117a0f693a160485d76__49152_inf",
            "cake_rmsnorm_train_f94861c36a14eb75e39c__384_1024",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f",
            "cake_rmsnorm_train_8f032213ba6eeb5d959f__16384_49152",
        ],
        "bwd_h1024": [
            "cake_rmsnorm_train_330930e9329d1e53aa5c",
            "cake_rmsnorm_train_28647b7ceb414d4fd02c__4_8",
            "cake_rmsnorm_train_89fa3e964f6ac26dc6a2__8_1536",
            "cake_rmsnorm_train_0c581a133624ff396ab4__1536_2048",
            "cake_rmsnorm_train_5e32dd889d72a75e348b__2048_4095",
            "cake_rmsnorm_train_4572a519afb83afc3743__4095_inf",
            "cake_rmsnorm_train_4830a91c2471462a29c8",
            "cake_rmsnorm_train_3b5dc80e2313fd64010e__2048_4095",
            "cake_rmsnorm_train_4c829a19ba34d598a48d__4095_inf",
        ],
        "bwd_w32_h1024": [
            "cake_rmsnorm_train_81e742d5d2cf1e0f790a",
            "cake_rmsnorm_train_8501e95ea685c476cb50__4_16",
            "cake_rmsnorm_train_744d74299c2d45867c94__16_1536",
            "cake_rmsnorm_train_561bea88a25c2f245c29__1536_4095",
            "cake_rmsnorm_train_64423f2da36401d9de39__4095_inf",
            "cake_rmsnorm_train_4830a91c2471462a29c8__1536_4095",
            "cake_rmsnorm_train_4c829a19ba34d598a48d__4095_inf",
        ],
        "bwd_h1280": [
            "cake_rmsnorm_train_8895b0224bb3d596a3a6",
            "cake_rmsnorm_train_a5f2942d61e591f57a76",
            "cake_rmsnorm_train_809235a3965d337a6877",
            "cake_rmsnorm_train_12bbedff93fcaafa475d__512_6143",
            "cake_rmsnorm_train_b1339aac23ed13dedce9__6143_32767",
            "cake_rmsnorm_train_f8cb729b7ec5f34002ca__32767_inf",
            "cake_rmsnorm_train_849ee5d846ddc145e3a9__512_6143",
            "cake_rmsnorm_train_849ee5d846ddc145e3a9__6143_32767",
            "cake_rmsnorm_train_e35c9369f7e555241f97__32767_inf",
        ],
        "bwd_w32_h1280": [
            "cake_rmsnorm_train_7905241de01123c84d2d__0_4",
            "cake_rmsnorm_train_8d8ef930b7d650c5538a__4_128",
            "cake_rmsnorm_train_e476d0b2c0bd4285f16d",
            "cake_rmsnorm_train_80956d258172dfa76c2a",
            "cake_rmsnorm_train_c66b8cbd5626e2c1f2b4__4096_6143",
            "cake_rmsnorm_train_de2f75d431899520db87__6143_32767",
            "cake_rmsnorm_train_c5386a35eeb7cfbd1658__32767_inf",
            "cake_rmsnorm_train_849ee5d846ddc145e3a9__512_4096",
            "cake_rmsnorm_train_382c8727d8b150884d7f__4096_6143",
            "cake_rmsnorm_train_849ee5d846ddc145e3a9__6143_32767",
            "cake_rmsnorm_train_e35c9369f7e555241f97__32767_inf",
        ],
        "bwd_h1536": [
            "cake_rmsnorm_train_db218c92c4c86b9da85a__0_4",
            "cake_rmsnorm_train_c2a523de1ff578621a88__4_16",
            "cake_rmsnorm_train_1207b05663ed2b4de14b__16_320",
            "cake_rmsnorm_train_7dc29629ff21625607f8__320_1024",
            "cake_rmsnorm_train_88510c609fd688aab969__1024_12288",
            "cake_rmsnorm_train_6a428da2227972ae18ae__12288_16384",
            "cake_rmsnorm_train_56d26996239835327501__16384_45056",
            "cake_rmsnorm_train_411e270c0575d26a77f4__45056_65536",
            "cake_rmsnorm_train_c93d0266065acad326fb__65536_inf",
            "cake_rmsnorm_train_628248febbdfe17c3325__320_1024",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__1024_12288",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__12288_16384",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_45056",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__45056_65536",
        ],
        "bwd_w32_h1536": [
            "cake_rmsnorm_train_6d3986ee0dd962cafd21__0_4",
            "cake_rmsnorm_train_744aa77460c886f4f429__4_16",
            "cake_rmsnorm_train_137ba5328176fb07b37d__16_320",
            "cake_rmsnorm_train_baff9ecbafe7de1e6030__320_1024_107",
            "cake_rmsnorm_train_7c0dd63f0b011049603e__1024_16384",
            "cake_rmsnorm_train_de8e07d4cb312b06dbc4__16384_45056",
            "cake_rmsnorm_train_64ffe19b041f5bfa34fb__45056_65536",
            "cake_rmsnorm_train_c9dc9e43af546ff815d3__65536_inf",
            "cake_rmsnorm_train_628248febbdfe17c3325__320_1024",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__16384_45056",
            "cake_rmsnorm_train_db5c0b673c516bd0295d__45056_65536",
        ],
        "bwd_h2048": [
            "cake_rmsnorm_train_c5f62407b828a85f82ec",
            "cake_rmsnorm_train_a951623c157d070e85ab__4_16",
            "cake_rmsnorm_train_518cf16a7bf8e88042ee__16_31",
            "cake_rmsnorm_train_6720b4a159be2b84b325__31_256",
            "cake_rmsnorm_train_845cdcd8145ada631461__256_1024",
            "cake_rmsnorm_train_cc26e316ae40d683a118__1024_16384",
            "cake_rmsnorm_train_845cdcd8145ada631461__16384_50688",
            "cake_rmsnorm_train_21ee6fc93139c5e0cd41__50688_inf",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50688",
            "cake_rmsnorm_train_c4d8be677fa70b841877__50688_inf",
        ],
        "bwd_w32_h2048": [
            "cake_rmsnorm_train_0017ab41f68d8689ebc2",
            "cake_rmsnorm_train_65e5f2024d26a36f2e18__4_16",
            "cake_rmsnorm_train_23c900134dc889302136__16_256",
            "cake_rmsnorm_train_838a9a4054731f7942af__256_1024",
            "cake_rmsnorm_train_c83c314c6dcf28ad7369__1024_16384",
            "cake_rmsnorm_train_838a9a4054731f7942af__16384_50688",
            "cake_rmsnorm_train_3f270477a248188afe2a__50688_inf",
            "cake_rmsnorm_train_271082c56c222bf10d77",
            "cake_rmsnorm_train_c4d8be677fa70b841877",
            "cake_rmsnorm_train_c4d8be677fa70b841877__16384_50688",
            "cake_rmsnorm_train_c4d8be677fa70b841877__50688_inf",
        ],
        "bwd_h4096": [
            "cake_rmsnorm_train_945a0afdb82ab47eb643",
            "cake_rmsnorm_train_41f44b4bc0b765350a50__4_16",
            "cake_rmsnorm_train_c1de7e41da4550cca0bc__16_256",
            "cake_rmsnorm_train_a647fccc3d9fa9134733",
            "cake_rmsnorm_train_d41a514f5210e9adc39f",
            "cake_rmsnorm_train_ee4a9053bb25b8d49fcf__4096_24064",
            "cake_rmsnorm_train_a647fccc3d9fa9134733__24064_24576",
            "cake_rmsnorm_train_23573e0c42a194a04876__24576_65536",
            "cake_rmsnorm_train_f11a86167db4f8e9a5da__65536_inf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_24064",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__24064_24576",
        ],
        "bwd_w32_h4096": [
            "cake_rmsnorm_train_f6bd1634a115bd281055",
            "cake_rmsnorm_train_b378ba007691c582171e__4_16",
            "cake_rmsnorm_train_30ec8bb68cf4fe0c817b__16_256",
            "cake_rmsnorm_train_dc439f8335327e482025",
            "cake_rmsnorm_train_b08633208e60b8ce94fd",
            "cake_rmsnorm_train_0f1859b9fe9e47668a1c__4096_24064",
            "cake_rmsnorm_train_0ff440daa6f9f304e544__24064_65536",
            "cake_rmsnorm_train_a1f537fc94bde175bb3c__65536_inf",
            "cake_rmsnorm_train_69bf6e104bb48b6b74bf__256_1024",
            "cake_rmsnorm_train_2030d80884deb8abcf7b",
            "cake_rmsnorm_train_10e24516314c244771f4__4096_24064",
            "cake_rmsnorm_train_10e24516314c244771f4__24064_65536",
        ],
        "bwd_h5120": [
            "cake_rmsnorm_train_dbc680f356be4cd643de",
            "cake_rmsnorm_train_904b1ad45d6219b4f43c__4_12",
            "cake_rmsnorm_train_29bcbde1d271ea989fa8__12_31",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__31_256",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__256_511",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__511_640",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__640_895",
            "cake_rmsnorm_train_080e0a6c3b18e9bad790__895_896",
            "cake_rmsnorm_train_838c077df3a6d35d55b7__896_4096",
            "cake_rmsnorm_train_6816fb80e97380f1bcbc",
            "cake_rmsnorm_train_b8b0913b40f035f43b76__19968_49152",
            "cake_rmsnorm_train_670f4cafe8c34a429d30",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__256_511",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__640_895",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__896_4096",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__19968_49152",
        ],
        "bwd_w32_h5120": [
            "cake_rmsnorm_train_d47138228b123e55fe5d",
            "cake_rmsnorm_train_05396cd3a077d3772fdc__4_15",
            "cake_rmsnorm_train_139d162a8bb29d1b2d35__15_256",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__256_511",
            "cake_rmsnorm_train_139d162a8bb29d1b2d35__511_640",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__640_895",
            "cake_rmsnorm_train_139d162a8bb29d1b2d35__895_1280",
            "cake_rmsnorm_train_cd612de56c0985bb4caa__1280_4096",
            "cake_rmsnorm_train_468cc4d1b8128187b4b3__4096_19968",
            "cake_rmsnorm_train_de0cf457cd240f9c6422__19968_32767",
            "cake_rmsnorm_train_e7cb6972099b7e10c035__32767_131072",
            "cake_rmsnorm_train_13bd0c5f56e37867eb26__131072_inf",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__256_511",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__640_895",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__1280_4096",
            "cake_rmsnorm_train_2af4cf2f58ba58f78099__4096_19968",
            "cake_rmsnorm_train_c5831b29d17867842e5f__19968_32767",
        ],
        "bwd_h6144": [
            "cake_rmsnorm_train_e37752b056dd4e4a0bee",
            "cake_rmsnorm_train_fdfb73350c1a926a4929__4_16",
            "cake_rmsnorm_train_0dafdf094615d080fcc9__16_960",
            "cake_rmsnorm_train_5d292172df772adb17ae__960_1151",
            "cake_rmsnorm_train_0dafdf094615d080fcc9__1151_1152",
            "cake_rmsnorm_train_5d292172df772adb17ae__1152_19456",
            "cake_rmsnorm_train_9cc51adfe9826056658b__19456_30720",
            "cake_rmsnorm_train_ffbcfda26d306410e01f__30720_131072",
            "cake_rmsnorm_train_ddedaa3be7466a4faa97__131072_inf",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__960_1151",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__1152_19456",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_30720",
        ],
        "bwd_residual_h6144": [
            "cake_rmsnorm_train_deeb187630473818923d__0_4",
            "cake_rmsnorm_train_94c547eccfe2ca37ee0f__4_15",
            "cake_rmsnorm_train_ee1be07a89a0482de45e__15_320",
            "cake_rmsnorm_train_0dcdc563be377247044e__320_447",
            "cake_rmsnorm_train_ee1be07a89a0482de45e__447_576",
            "cake_rmsnorm_train_864e84f3d874f850c091__576_3584",
            "cake_rmsnorm_train_0dcdc563be377247044e__3584_7168",
            "cake_rmsnorm_train_853d4447e18f114f5da5__7168_131072",
            "cake_rmsnorm_train_87d2866c1ea3bcc08b13__131072_inf",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__320_447",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168",
        ],
        "bwd_w32_h6144": [
            "cake_rmsnorm_train_35b97e6e2bfa3e9f2ff9",
            "cake_rmsnorm_train_18b8b8054c7053942bcd__4_16",
            "cake_rmsnorm_train_5cca2d0fa4553f992406__16_1664",
            "cake_rmsnorm_train_b3e6531e46667901ab91__1664_2304",
            "cake_rmsnorm_train_e77387f14390ba486055__2304_19456",
            "cake_rmsnorm_train_bdfba721a43e0d093668__19456_28672",
            "cake_rmsnorm_train_e77387f14390ba486055__28672_inf",
            "cake_rmsnorm_train_fb3f2a96e954f0e241bc__19456_28672",
        ],
        "bwd_w32_residual_h6144": [
            "cake_rmsnorm_train_a6b1a6e565dc98b49e1a",
            "cake_rmsnorm_train_c29a8f856283578103dc__4_16",
            "cake_rmsnorm_train_3bbd4f848b4847bff893__16_768",
            "cake_rmsnorm_train_71d36617003ab7742a77__768_3584",
            "cake_rmsnorm_train_56f988dde54db37f09d9__3584_7168",
            "cake_rmsnorm_train_b394bc75f9df18f002f2__7168_131072",
            "cake_rmsnorm_train_647589443d4915ca8fde__131072_inf",
            "cake_rmsnorm_train_60bfcb24560bbae612c3__3584_7168",
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


# The backward workspace starts with the deterministic pairs' ``uint32``
# counter table (192 words at most, padded to four 256-byte lines); every pair
# kernel leaves the words it uses at zero.  The single kernel's counters
# (``SINGLE_COUNTERS_BYTES`` at ``SINGLE_COUNTERS_OFFSET``) and FP32 sums
# (``SUMS_BYTES`` at ``SUMS_OFFSET``: the widest sums of the package's single
# records, their workspace rules' ``sum_bytes``) follow; the single kernel leaves
# its sums at zero and its counters in their rest state after every launch.
# The FP32 partial rows of the pairs' main kernels come last
# (``PAIR_ROWS_OFFSET``); a part-row single kernel (``claim_part_f32``) binds the
# single counters and, at ``PAIR_ROWS_OFFSET``, scratch of one FP32 row
# per CTA (its rule's ``part_row_bytes``), which needs no rest state, like the
# pairs' rows.  Every offset is the package's, not the call's, so one
# buffer, zero when first used, serves calls of any token count, window, hidden
# size, residual flag and weight dtype with no host write between them, and the
# two forms never share a counter word.
COUNTERS_BYTES = 1024
SINGLE_COUNTERS_OFFSET = COUNTERS_BYTES
SINGLE_COUNTERS_BYTES = 256
SUMS_OFFSET = SINGLE_COUNTERS_OFFSET + SINGLE_COUNTERS_BYTES
# The workspace rule kinds of the backward forms: the main kernel of a
# deterministic pair, the single kernel, the part-row single kernel.
WORKSPACE_KINDS = ("chunk_partials", "claim_sum_f32", "claim_part_f32")


def _sum_bytes(record: dict[str, Any]) -> int:
    """Bytes of the FP32 column sums a single-kernel record binds to
    ``partial`` (its workspace rule's ``sum_bytes``, the kernel's
    ``workspace_bytes`` of the build); 0 for another record."""

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "claim_sum_f32":
        return 0
    return int(rule["sum_bytes"])


SUMS_BYTES = _round_up(
    max((_sum_bytes(record) for record in MODULES.values()), default=0), 256
)
PAIR_ROWS_OFFSET = SUMS_OFFSET + SUMS_BYTES


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
    taper.  Under a deterministic pair the FP32 partial row of every chunk, and
    the ``dw`` the reduce kernel folds from them in chunk order, are pure
    functions of ``(T, slots, rpi, tail_div, max_halving)``; the single kernel
    adds the chunks' sums into FP32 column sums in the order its CTAs finish,
    so its ``dw`` depends on scheduling (its ``dx`` does not).
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
    main launch (``slots`` its twelfth argument) and, under a deterministic
    pair, the reduce launch (``n_chunks``).
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


def launch_sms(rule: dict[str, Any], *, device_index: int) -> int:
    """SMs a ``claimed_chunks`` grid rule launches on: ``SM count - free_sms``.

    A record whose kernel runs beside a side stream that keeps SMs spells
    their count under its grid rule (``free_sms``: the deterministic
    package's one-row-per-stage pairs, which this package hosts with the
    deterministic package's host plan); every other record counts every SM.
    The chunk table's slots, the launch grid's bound and the workspace bound
    are built on this count, as the deterministic package's ``slot_count``
    builds them.
    """

    sms = sm_count(device_index)
    free = int(rule.get("free_sms", 0))
    if not 0 <= free < sms:
        raise ValueError(f"grid rule free_sms {free} is outside [0, {sms})")
    return sms - free


def table_slots(rule: dict[str, Any], *, device_index: int) -> int:
    """Chunk-table slots of a ``claimed_chunks`` grid rule on the device.

    ``table_slots_per_sm * SMs`` when the rule spells ``table_slots_per_sm``
    (a build whose table is cut for more, shorter chunks than its launch has
    CTAs: each CTA claims several), else the launch's own ``ctas_per_sm *
    SMs``, ``SMs`` the :func:`launch_sms` of the rule (the SM count less the
    ``free_sms`` it spells).  The main kernel takes it as its ``slots``
    argument; the grid stays bounded by ``ctas_per_sm * SMs``.
    """

    per_sm = int(rule.get("table_slots_per_sm", 0)) or int(rule["ctas_per_sm"])
    return per_sm * launch_sms(rule, device_index=device_index)


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one module for ``rows`` tokens under its recorded grid rule.

    ``rows_per_cta`` modules launch ``ceil(rows / rows_per_cta)`` CTAs, capped
    at ``grid_limit`` when the rule records a positive limit, else at
    ``grid_per_sm * SM count`` of the device when that is positive (persistent
    grid-stride row loop in the kernel).  ``fixed`` modules launch exactly
    ``ctas`` CTAs (the dw reduce kernel: one CTA per column slab; the
    one-cluster backward kernel: the CTAs of its one thread-block cluster, the
    cluster dimension its binding launches with).
    ``claimed_chunks`` modules are the main backward kernel: up to
    ``ctas_per_sm * SMs`` persistent CTAs (``SMs`` the :func:`launch_sms` of
    the rule: the SM count less the ``free_sms`` it spells) claim the chunks of
    the table of :func:`chunk_rounds` built on :func:`table_slots` slots
    (``rows_per_issue`` rows per pipeline step of the kernel, under the taper
    its record spells: :func:`chunk_taper`), so the grid is ``min(ctas_per_sm
    * SMs, n_chunks)``; a column-task kernel's rule spells its ``col_tasks`` (``H /
    col`` column tasks, tickets beside the chunks), so its grid is
    ``min(ctas_per_sm * SMs, n_chunks + col_tasks)`` and a small ``T``
    runs every task at once.  The launch is a plain
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
        slots = table_slots(rule, device_index=device_index)
        grid_x = min(
            int(rule["ctas_per_sm"]) * launch_sms(rule, device_index=device_index),
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
    built on ``slots`` = :func:`table_slots` under the module's taper
    (:func:`chunk_taper`); both are host-known and never depend on scheduling.
    The main kernel takes both; the dw reduce kernel a deterministic pair
    launches after it takes ``n_chunks``.  A one-cluster module (grid rule
    ``fixed``: its CTAs are one thread-block cluster over every row) reads
    neither: ``(1, 1)``.
    """

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] not in WORKSPACE_KINDS:
        raise ValueError(f"module kernel_{record['kernel']} owns no backward workspace")
    grid = record["grid_rule"]
    if grid["kind"] == "fixed":
        # The one-cluster kernel: its ``ctas`` CTAs (one thread-block cluster) cover
        # every row and read neither count; one chunk of one slot.
        return 1, 1
    if grid["kind"] != "claimed_chunks":
        raise ValueError(
            f"module kernel_{record['kernel']} launches under {grid['kind']!r}, not a "
            "chunk-claiming grid"
        )
    slots = table_slots(grid, device_index=device_index)
    return n_chunks(rows, slots, int(grid["rows_per_issue"]), *chunk_taper(grid)), slots


def partial_capacity(record: dict[str, Any], rows: int, *, device_index: int) -> int:
    """Partial rows the workspace of one main backward module holds for every token count up to ``rows``."""

    rule = record["grid_rule"]
    if rule["kind"] != "claimed_chunks":
        raise ValueError(f"grid rule {rule['kind']!r} has no chunk table")
    return chunk_capacity(rows, table_slots(rule, device_index=device_index))


def workspace_layout(
    record: dict[str, Any],
    rows: int,
    *,
    device_index: int,
    capacity_record: Optional[dict[str, Any]] = None,
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module.

    The workspace is one CUDA ``uint8`` buffer: the deterministic pairs'
    ``uint32`` counters in its first :data:`COUNTERS_BYTES` bytes, the single
    kernel's counters at :data:`SINGLE_COUNTERS_OFFSET`, its FP32 sums at
    :data:`SUMS_OFFSET` and the pairs' FP32 partial rows at
    :data:`PAIR_ROWS_OFFSET`.  A single-kernel module (workspace rule
    ``claim_sum_f32``) binds the single counters and its rule's ``sum_bytes``
    of sums, whatever ``rows``.  A part-row single-kernel module (workspace
    rule ``claim_part_f32``) binds the single counters and, at
    :data:`PAIR_ROWS_OFFSET`, ``part_row_bytes`` of FP32 scratch per CTA of its
    launch grid (:func:`launch_grid`, returned as ``grid``; the chunk table's
    ``slots`` may differ from it); a buffer sized for ``rows`` holds a row for
    every CTA of any launch up to ``rows`` (``min(rows, ctas_per_sm * SMs)``
    rows, ``SMs`` the :func:`launch_sms` of its grid rule: the bound of the
    launch grid).  The main kernel of a
    deterministic pair
    (``chunk_partials``) binds the pair counters and the FP32
    ``[capacity, hidden]`` per-chunk partial rows, ``capacity`` the
    :func:`chunk_capacity` of ``rows``, so a buffer sized for ``rows`` serves
    every smaller token count.  The buffer is zero when first used and every
    kernel leaves its counters in their rest state and the sums at zero, so no
    host write is needed between launches, layouts or CUDA Graph replays.  A
    route whose pair modules switch on the row count shares one buffer:
    ``capacity_record`` (the pair main module with the largest capacity) fixes
    the partial capacity for every pair module of the route, while
    ``n_chunks`` and ``slots`` follow ``record``, the module launched.
    """

    rule = record["workspace_rule"]
    chunks, slots = chunking(record, rows, device_index=device_index)
    hidden = int(record["hidden"])
    counters = int(rule["counters"])
    if rule["kind"] == "claim_sum_f32":
        if 4 * max(counters, 1) > SINGLE_COUNTERS_BYTES:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {counters} counters; the "
                f"single kernel's table holds {SINGLE_COUNTERS_BYTES // 4}"
            )
        partial_bytes = _sum_bytes(record)
        if partial_bytes % 4 or partial_bytes > SUMS_BYTES:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {partial_bytes} bytes of "
                f"FP32 sums; the workspace holds {SUMS_BYTES}"
            )
        return {
            "n_chunks": chunks,
            "slots": slots,
            "capacity_chunks": 0,
            "partial_bytes": partial_bytes,
            "partial_offset": SUMS_OFFSET,
            "counters": counters,
            "counters_offset": SINGLE_COUNTERS_OFFSET,
            "total_bytes": SUMS_OFFSET + _round_up(partial_bytes, 256),
        }
    if rule["kind"] == "claim_part_f32":
        if 4 * max(counters, 1) > SINGLE_COUNTERS_BYTES:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {counters} counters; the "
                f"single kernel's table holds {SINGLE_COUNTERS_BYTES // 4}"
            )
        row_bytes = int(rule["part_row_bytes"])
        if row_bytes <= 0 or row_bytes % 4:
            raise ValueError(
                f"module kernel_{record['kernel']} declares {row_bytes} part_row_bytes; "
                "one FP32 row is a positive multiple of 4 bytes"
            )
        grid = launch_grid(record, rows, device_index=device_index)[0]
        ctas = int(record["grid_rule"]["ctas_per_sm"]) * launch_sms(
            record["grid_rule"], device_index=device_index
        )
        return {
            "n_chunks": chunks,
            "slots": slots,
            "grid": grid,
            "capacity_chunks": 0,
            "partial_bytes": row_bytes * grid,
            "partial_offset": PAIR_ROWS_OFFSET,
            "counters": counters,
            "counters_offset": SINGLE_COUNTERS_OFFSET,
            "total_bytes": PAIR_ROWS_OFFSET
            + _round_up(row_bytes * min(int(rows), ctas), 256),
        }
    if 4 * max(counters, 1) > COUNTERS_BYTES:
        raise ValueError(
            f"module kernel_{record['kernel']} declares {counters} counters; the "
            f"workspace holds {COUNTERS_BYTES // 4}"
        )
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
    partial_bytes = 4 * capacity * hidden
    return {
        "n_chunks": chunks,
        "slots": slots,
        "capacity_chunks": capacity,
        "partial_bytes": partial_bytes,
        "partial_offset": PAIR_ROWS_OFFSET,
        "counters": counters,
        "counters_offset": 0,
        "total_bytes": PAIR_ROWS_OFFSET + _round_up(partial_bytes, 256),
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
    "PAIR_ROWS_OFFSET",
    "ROW_ALIGNMENT_ELEMENTS",
    "SINGLE_COUNTERS_BYTES",
    "SINGLE_COUNTERS_OFFSET",
    "SUMS_BYTES",
    "SUMS_OFFSET",
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
    "launch_sms",
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
