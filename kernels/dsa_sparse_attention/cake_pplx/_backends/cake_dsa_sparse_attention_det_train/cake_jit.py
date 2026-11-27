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
"""

from __future__ import annotations

import functools
from typing import Any

from . import cake_build

# Explicit target-owned registration of the generated training programs whose
# every output repeats bit for bit: one record per program (``PROGRAM``: the
# RoPE program, ``NOPE_PROGRAM``: the native zero-RoPE program) shared by every
# architecture it compiles for (``arches``), the kernel ``stages`` it registers
# in launch order, and one physical entry per stage: the ``library`` stem of its
# ``build.json`` entries (``<library>_<arch>``, see ``cake_build``), the FFI
# entry and the launch geometry the binding bakes in (block, cluster,
# cooperative launch, dynamic shared memory, programmatic dependent launch).
# A stage entry may narrow the record's architectures with its own ``arches``:
# its library ships for those only.  On ``sm_107a`` each forward runs its
# scanner-warp build, the stage ``SCAN_STAGES`` names (``arches``
# ``["sm_107a"]``), in place of the forward (whose ``arches`` are the other
# two): ``library_name`` resolves the forward there to the scanner stage's
# library.
# The forward stages are the forward kernels of the non-deterministic package
# (fixed order: bitwise reproducible), compiled under program names of this
# package.  The backward stages run in the record's order, every one of them on
# every backward (``BACKWARD_STAGES``): no stage is selected by a tensor value
# or by the device's SM count.  A backward of P > 1 key-range passes (the
# record's ``key_pass_policy``, or the caller's ``key_passes``) runs the
# program's ``KEY_PASS_STAGES_OF`` P times in place of its main stage.  Two
# records may name one library (the compaction of the key-range passes serves
# both programs); their launch dicts then agree.  What each library compiles -- translation
# units, nvcc flags, digests -- is recorded in ``build.json`` only; the
# positional argument order of every stage lives in the generated
# ``cake_launch`` module.  The records are populated by the generated-program
# export and only their stage names and policies are maintained by hand; the
# export rewrites the ``launch`` leaves.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_det_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_norm",
            "bwd_bound",
            "bwd_grid",
            "bwd_main",
            "bwd_redo_prep",
            "bwd_redo",
            "bwd_fold",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_bound_pack27",
            "bwd_grid_pack27",
            "bwd_redo_prep_fold",
            "bwd_fold_list",
            "fwd_scan",
            "fwd_h128",
            "bwd_delta_heads",
            "bwd_main_hw",
            "bwd_redo_hw",
            "bwd_main_pass_hw",
        ],
        # The key-range-pass policy: the non-deterministic package's record values (``cake_backend.KeyPassPolicy``).
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
            "max_passes": 4,
            "rules": {
                "sm_100a": {
                    "tail_rows": 0,
                    "fixed_passes": 0,
                    "min_kv": 0,
                    "tail_min_kv": 0,
                    "tail_cap": 0,
                },
                "sm_103a": {
                    "tail_rows": 1,
                    "fixed_passes": 0,
                    "min_kv": 0,
                    "tail_min_kv": 131072,
                    "tail_cap": 3,
                },
                "sm_107a": {
                    "tail_rows": 1,
                    "fixed_passes": 2,
                    "min_kv": 131072,
                    "tail_min_kv": 131072,
                    "tail_cap": 0,
                },
            },
        },
        "fwd": {
            "library": "cake_dsa_h64_det_train_c949a885132a772bdb53",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [384, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": True,
            },
        },
        "bwd_delta": {
            "library": "cake_dsa_h64_det_train_93ad63698c318aab988f",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 128,
                "use_pdl": False,
            },
        },
        "bwd_norm": {
            "library": "cake_dsa_h64_det_train_84376fdd68bed4f727cf",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 256,
                "use_pdl": False,
            },
        },
        "bwd_bound": {
            "library": "cake_dsa_h64_det_train_e515ef20935db22e2103",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_grid": {
            "library": "cake_dsa_h64_det_train_3dd5b7143dd684e308ea",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_main": {
            "library": "cake_dsa_h64_det_train_4487388f2950ac4f97a5",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_redo_prep": {
            "library": "cake_dsa_h64_det_train_f4ebce39fc25de804572",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_redo": {
            "library": "cake_dsa_h64_det_train_f947d21a4e7b4d35d0a5",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_fold": {
            "library": "cake_dsa_h64_det_train_9a9e7e6b8ea3cd322457",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_compact": {
            "library": "cake_dsa_h64_det_train_19c560118a407f5473fe",
            "ffi_entry": "run",
            "launch": {
                "block": [128, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_main_pass": {
            "library": "cake_dsa_h64_det_train_1893ca02d59bc5e3383a",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_bound_pack27": {
            "library": "cake_dsa_h64_det_train_3076bd8795bd9dcbdd22",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_grid_pack27": {
            "library": "cake_dsa_h64_det_train_44c3a785582753936b7f",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_redo_prep_fold": {
            "library": "cake_dsa_h64_det_train_e48ca0b7936f82162868",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_fold_list": {
            "library": "cake_dsa_h64_det_train_d3c5c3176a48d8e07ed4",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "fwd_scan": {
            "library": "cake_dsa_h64_det_train_491a3b1d55552281da1a",
            "ffi_entry": "run",
            "arches": ["sm_107a"],
            "launch": {
                "block": [416, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": True,
            },
        },
        "fwd_h128": {
            "library": "cake_dsa_h64_det_train_adfe34021dbfb2cffe71",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [512, 1, 1],
                "cluster": [2, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215808,
                "use_pdl": True,
            },
        },
        "bwd_delta_heads": {
            "library": "cake_dsa_h64_det_train_aac0a985a678f748245c",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [512, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 1536,
                "use_pdl": False,
            },
        },
        "bwd_main_hw": {
            "library": "cake_dsa_h64_det_train_f2912bada9840c99c4da",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_redo_hw": {
            "library": "cake_dsa_h64_det_train_9f1f44a507ca9dc8ce87",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_main_pass_hw": {
            "library": "cake_dsa_h64_det_train_91a238d5123ae4182d1c",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
    },
    # The native zero-RoPE (NoPE) training program: shared latent K = V of 512
    # dimensions, no rope operands, one combined dK + dV gradient per key row.
    # ``index_prep`` (the fused index preparation: per-document rows to global
    # key rows, optionally compacted) runs ahead of ``nope_fwd`` on the packed
    # routes and serves the RoPE program's packed entry too.  The key-range
    # passes run the RoPE program's compaction library (``nope_bwd_compact``)
    # and the NoPE pass build of the main kernel (``nope_bwd_main_pass``).  The
    # key-range-pass policy is the non-deterministic package's NoPE record
    # values (``key_bytes`` 2048; ``dq_partial_bytes_per_token`` = 64 heads x
    # 512 x 4 B of FP32 dQ partial per token; ``max_passes`` 1: the policy
    # never plans a second pass, and ``key_passes`` forces the pass form).  A
    # call with a trainable per-head sink runs the forward's sink build
    # (``nope_fwd_sink``: the forward's arguments plus the FP32 ``sink [64]``,
    # the forward's launch geometry; ``sm_100a`` / ``sm_103a``) in place of
    # ``nope_fwd`` -- on ``sm_107a`` its scanner-warp build
    # (``nope_fwd_scan_sink``), as ``nope_fwd_scan`` serves ``nope_fwd`` there
    # -- and, in its backward, the sink-gradient stage (``nope_bwd_dsink``, 256
    # threads, every architecture of the record) once right after the delta
    # stage.  These are the non-deterministic package's sink kernels (fixed
    # order: bitwise reproducible) under this package's program names.
    "cake_dsa_h64_det_nope_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "index_prep",
            "nope_fwd",
            "nope_bwd_delta",
            "nope_bwd_norm",
            "nope_bwd_bound",
            "nope_bwd_grid",
            "nope_bwd_main",
            "nope_bwd_redo_prep",
            "nope_bwd_redo",
            "nope_bwd_fold",
            "nope_bwd_compact",
            "nope_bwd_main_pass",
            "nope_bwd_bound_pack27",
            "nope_bwd_grid_pack27",
            "nope_bwd_redo_prep_fold",
            "nope_bwd_fold_list",
            "nope_bwd_bound_pipe",
            "nope_bwd_bound_pack27_pipe",
            "nope_fwd_scan",
            "nope_fwd_sink",
            "nope_bwd_dsink",
            "nope_fwd_scan_sink",
        ],
        # The key-range-pass policy: the non-deterministic package's NoPE record
        # values (``cake_backend.KeyPassPolicy``).
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2048,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
            "max_passes": 1,
            "dq_partial_bytes_per_token": 131072,
        },
        "index_prep": {
            "library": "cake_dsa_h64_det_nope_train_db791110b21a91be1388",
            "ffi_entry": "run",
            "launch": {
                "block": [128, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_fwd": {
            "library": "cake_dsa_h64_det_nope_train_46cdc056c891876ed545",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [320, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 223488,
                "use_pdl": True,
            },
        },
        "nope_bwd_delta": {
            "library": "cake_dsa_h64_det_nope_train_2575bc84fa1e92ec0b69",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 128,
                "use_pdl": False,
            },
        },
        "nope_bwd_norm": {
            "library": "cake_dsa_h64_det_nope_train_da63029a854ea164bdbd",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 256,
                "use_pdl": False,
            },
        },
        "nope_bwd_bound": {
            "library": "cake_dsa_h64_det_nope_train_bbf314a9c088b14d6317",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_grid": {
            "library": "cake_dsa_h64_det_nope_train_983a1842d246e99cf5f0",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_main": {
            "library": "cake_dsa_h64_det_nope_train_a8f757c468aff1aee239",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 217344,
                "use_pdl": False,
            },
        },
        "nope_bwd_redo_prep": {
            "library": "cake_dsa_h64_det_nope_train_9c5305b6ec6841752260",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_redo": {
            "library": "cake_dsa_h64_det_nope_train_bfbe0a61e0a693c742d7",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 217344,
                "use_pdl": False,
            },
        },
        "nope_bwd_fold": {
            "library": "cake_dsa_h64_det_nope_train_02223183ecc7ab5cc5c9",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_compact": {
            "library": "cake_dsa_h64_det_train_19c560118a407f5473fe",
            "ffi_entry": "run",
            "launch": {
                "block": [128, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_main_pass": {
            "library": "cake_dsa_h64_det_nope_train_987774aae8c0c668e8ea",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 217344,
                "use_pdl": False,
            },
        },
        "nope_bwd_bound_pack27": {
            "library": "cake_dsa_h64_det_nope_train_a75be993b6f056087d05",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_grid_pack27": {
            "library": "cake_dsa_h64_det_nope_train_bd40e409c46688d665e1",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_redo_prep_fold": {
            "library": "cake_dsa_h64_det_nope_train_335494a34f38973c492c",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_fold_list": {
            "library": "cake_dsa_h64_det_nope_train_9f59bd3e4363f97c2c56",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_bound_pipe": {
            "library": "cake_dsa_h64_det_nope_train_cd3686becf1223e8ca1b",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_bound_pack27_pipe": {
            "library": "cake_dsa_h64_det_nope_train_54d8d3485d31391417d3",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_fwd_scan": {
            "library": "cake_dsa_h64_det_nope_train_d13be3d78a70edab8c99",
            "ffi_entry": "run",
            "arches": ["sm_107a"],
            "launch": {
                "block": [352, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 223488,
                "use_pdl": True,
            },
        },
        "nope_fwd_sink": {
            "library": "cake_dsa_h64_det_nope_train_f41a0f9f0e10448aaf94",
            "ffi_entry": "run",
            "arches": ["sm_100a", "sm_103a"],
            "launch": {
                "block": [320, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 223488,
                "use_pdl": True,
            },
        },
        "nope_bwd_dsink": {
            "library": "cake_dsa_h64_det_nope_train_36d862698eea1ea770ef",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 1152,
                "use_pdl": False,
            },
        },
        "nope_fwd_scan_sink": {
            "library": "cake_dsa_h64_det_nope_train_e1e7a7620a259632759f",
            "ffi_entry": "run",
            "arches": ["sm_107a"],
            "launch": {
                "block": [352, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 223488,
                "use_pdl": True,
            },
        },
    },
}

PROGRAM = "cake_dsa_h64_det_train"
NOPE_PROGRAM = "cake_dsa_h64_det_nope_train"
PROGRAMS = (PROGRAM, NOPE_PROGRAM)
FORWARD_STAGES = ("fwd",)
# The backward of the RoPE program, in launch order: the delta and the per-query
# bound terms, the key norms (which zero the row-bound words), the per-row bounds
# and pair counts, the per-row grids (which zero the touched accumulator rows and
# counters), the main stage (dQ and the integer dK / dV accumulation with its
# overflow counters), the preparation and the launch of the recomputation of
# flagged rows (always launched; it exits when no row is flagged), and the fold
# of the integers into the floating-point results.
BACKWARD_STAGES = (
    "bwd_delta",
    "bwd_norm",
    "bwd_bound",
    "bwd_grid",
    "bwd_main",
    "bwd_redo_prep",
    "bwd_redo",
    "bwd_fold",
)
STAGES = FORWARD_STAGES + BACKWARD_STAGES
# The key-range passes of the RoPE program: a backward that splits every token's keys into ``P > 1`` key ranges
# runs, per pass and in place of ``bwd_main``, the compaction of the token's keys in the range (``bwd_compact``)
# and the pass build of the main stage (``bwd_main_pass``; its dQ goes through an FP32 partial between passes).
# The redo stays one launch.  The NoPE program's pass stages are their ``nope_`` forms (``KEY_PASS_STAGES_OF``).
KEY_PASS_STAGES = ("bwd_compact", "bwd_main_pass")
# The pack27 builds of the row bound and its grid decode (a 27-bit pair count with 10-bit packed terms): a backward of
# ``2 ** 26 <= num_queries * topk < 2 ** 27`` pair slots runs them in place of ``bwd_bound`` / ``bwd_grid`` (whose
# pack builds fall back to three reductions per pair there) when the record registers them; the choice is the
# shapes' only. Same arguments and grids as the stages they replace.
PACK27_STAGES = {"bwd_bound": "bwd_bound_pack27", "bwd_grid": "bwd_grid_pack27"}
# The accumulate-mode builds of the redo preparation and the fold (a backward that adds dK / dV into the caller's
# ``dkv_acc``): the redo preparation's fold build adds every unflagged row into ``dkv_acc`` as it scans the rows (the
# redo touches only flagged rows, so an unflagged row's integers are final there), and after the redo the listed fold
# adds the flagged rows alone (the redo list; every marked row when the list overflowed), on a persistent grid.  They
# replace ``bwd_redo_prep`` / ``bwd_fold`` (the NoPE program: ``nope_``) in such a backward; the results are the same
# bits.
ACCUMULATE_STAGES = {"bwd_redo_prep": "bwd_redo_prep_fold", "bwd_fold": "bwd_fold_list"}
# The NoPE program names its stages ``nope_<stage>`` (globally unique: the
# generated launch tables are keyed by stage name across both programs).
STAGE_PREFIX = {PROGRAM: "", NOPE_PROGRAM: "nope_"}
NOPE_STAGES = tuple(f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}" for stage in STAGES)
NOPE_FORWARD_STAGES = NOPE_STAGES[:1]
NOPE_BACKWARD_STAGES = NOPE_STAGES[1:]
NOPE_KEY_PASS_STAGES = tuple(
    f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}" for stage in KEY_PASS_STAGES
)
KEY_PASS_STAGES_OF = {PROGRAM: KEY_PASS_STAGES, NOPE_PROGRAM: NOPE_KEY_PASS_STAGES}
NOPE_PACK27_STAGES = tuple(
    f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}" for stage in PACK27_STAGES.values()
)
NOPE_ACCUMULATE_STAGES = {
    f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}": f"{STAGE_PREFIX[NOPE_PROGRAM]}{alt}"
    for stage, alt in ACCUMULATE_STAGES.items()
}
ACCUMULATE_STAGES_OF = {
    PROGRAM: ACCUMULATE_STAGES,
    NOPE_PROGRAM: NOPE_ACCUMULATE_STAGES,
}
# The pipelined NoPE row-bound builds (two 128-slot blocks in flight, launch_bounds(256, 5)): on the architectures of
# ``NOPE_PIPE_ARCHES`` (their stage entries' ``arches``, the only architectures they ship for) a NoPE backward runs
# them in place of ``nope_bwd_bound`` / ``nope_bwd_bound_pack27``; the device's architecture alone chooses.  Same
# arguments, grid and words (bit for bit) as the stages they replace.
NOPE_PIPE_STAGES = {
    "nope_bwd_bound": "nope_bwd_bound_pipe",
    "nope_bwd_bound_pack27": "nope_bwd_bound_pack27_pipe",
}
NOPE_PIPE_ARCHES = ("sm_100a", "sm_103a")
_ROPE_ALTERNATES = tuple(PACK27_STAGES.values()) + tuple(ACCUMULATE_STAGES.values())
_NOPE_ALTERNATES = (
    NOPE_PACK27_STAGES
    + tuple(NOPE_ACCUMULATE_STAGES.values())
    + tuple(NOPE_PIPE_STAGES.values())
)
# The scanner-warp builds of the two forwards and of the NoPE forward's sink
# build: on the architectures of their stage entries' ``arches``
# (``sm_107a``) the host runs them in place of the forward stage; the
# device's architecture alone chooses.  Same arguments, grid and outputs as
# the forward they replace.
SCAN_STAGES = {
    "fwd": "fwd_scan",
    "nope_fwd": "nope_fwd_scan",
    "nope_fwd_sink": "nope_fwd_scan_sink",
}
# The NoPE program's trainable-sink stages: the forward's sink build, which runs in place of ``nope_fwd`` when a call
# passes a sink (same arguments plus ``sink``, same grid and outputs; its ``lse`` includes the sink; its scanner-warp
# build ``SCAN_STAGES["nope_fwd_sink"]`` where that ships), and the sink-gradient stage the backward of such a call
# launches once after the delta stage.  Their entries' ``arches`` name the architectures that ship them.
NOPE_SINK_FORWARD_STAGE = "nope_fwd_sink"
NOPE_DSINK_STAGE = "nope_bwd_dsink"
NOPE_SINK_STAGES = (NOPE_SINK_FORWARD_STAGE, NOPE_DSINK_STAGE)
# The RoPE program's stages over 96 and 128 query heads (``cake_backend.SUPPORTED_HEADS``), registered for
# ``sm_100a`` and ``sm_103a`` (their entries' ``arches``): stage -> the stage a call over ``64 < H <= 128`` heads runs
# in its place.  The forward is the non-deterministic package's single-pass forward (one (2, 1, 1) cluster per
# token, every head of the token: ``out`` / ``lse`` / ``o_lo`` at ``t * H + h``); the delta stage's heads build
# writes ``delta [T, H]`` and the bound terms over the token's H heads; the main kernel's head-window builds (main,
# redo, key-range pass) run both 64-head windows of every token in one launch, ``2 T`` work items with the window
# selector -1.  The other backward stages (key norms, row bound and grid over the T tokens, redo preparation, fold,
# compaction) serve both head counts unchanged.  The record lists them after its other stages.
HEAD_COUNT_STAGES = {
    "fwd": "fwd_h128",
    "bwd_delta": "bwd_delta_heads",
    "bwd_main": "bwd_main_hw",
    "bwd_redo": "bwd_redo_hw",
    "bwd_main_pass": "bwd_main_pass_hw",
}
STAGES_OF = {
    PROGRAM: STAGES
    + KEY_PASS_STAGES
    + _ROPE_ALTERNATES
    + (SCAN_STAGES["fwd"],)
    + tuple(HEAD_COUNT_STAGES.values()),
    NOPE_PROGRAM: NOPE_STAGES
    + NOPE_KEY_PASS_STAGES
    + _NOPE_ALTERNATES
    + (SCAN_STAGES["nope_fwd"],)
    + NOPE_SINK_STAGES
    + (SCAN_STAGES[NOPE_SINK_FORWARD_STAGE],),
}
BACKWARD_STAGES_OF = {PROGRAM: BACKWARD_STAGES, NOPE_PROGRAM: NOPE_BACKWARD_STAGES}


def stage_name(stage: str, program: str = PROGRAM) -> str:
    """The name of the RoPE program's ``stage`` (``fwd``, ``bwd_main``, ...) in ``program``."""
    try:
        prefix = STAGE_PREFIX[program]
    except KeyError:
        raise ValueError(
            f"unknown program {program!r}; registered: {PROGRAMS}"
        ) from None
    return f"{prefix}{stage}"


def record(program: str = PROGRAM) -> dict[str, Any]:
    """The registered record of ``program`` (the RoPE program by default)."""
    if program not in PROGRAMS:
        raise ValueError(f"unknown program {program!r}; registered: {PROGRAMS}")
    return MODULES[program]


def registered_stages(program: str = PROGRAM) -> tuple[str, ...]:
    """Stages the record of ``program`` registers, in launch order."""
    rec = record(program)
    order = STAGES_OF[program]
    present = tuple(stage for stage in order if stage in rec)
    declared = tuple(rec.get("stages", present))
    if tuple(s for s in order if s in declared) != present:
        raise ValueError(
            f"registry record {program!r} declares stages {declared} but carries {present}"
        )
    return present


def stage_arches(stage: str, program: str = PROGRAM) -> tuple[str, ...]:
    """The architectures ``stage`` of ``program`` ships for: its entry's own ``arches``, else the record's."""
    rec = record(program)
    return tuple(rec[stage].get("arches", rec["arches"]))


def stage_for(stage: str, arch: str, program: str = PROGRAM) -> str:
    """The stage that runs ``stage`` of ``program`` on ``arch``: its scanner-warp build
    (``SCAN_STAGES``) where that ships for ``arch``, else ``stage`` itself."""
    scan = SCAN_STAGES.get(stage)
    if (
        scan is not None
        and scan in record(program)
        and arch in stage_arches(scan, program)
    ):
        return scan
    return stage


def library_name(stage: str, arch: str, program: str = PROGRAM) -> str:
    """The ``build.json`` entry that runs ``stage`` of ``program`` on ``arch``: ``<library>_<arch>``
    of the stage, or of its scanner-warp build where that ships for ``arch`` (``stage_for``)."""
    rec = record(program)
    if arch not in rec["arches"]:
        raise RuntimeError(
            f"the generated DSA training program {program!r} is registered for {rec['arches']}; "
            f"{arch!r} is not served by this checkout"
        )
    if stage not in rec:
        raise NotImplementedError(
            f"stage {stage!r} of the generated DSA training program {program!r} is not "
            f"registered in this checkout (registered: {registered_stages(program)})"
        )
    stage = stage_for(stage, arch, program)
    if arch not in stage_arches(stage, program):
        raise RuntimeError(
            f"stage {stage!r} of the DSA training program {program!r} ships for "
            f"{list(stage_arches(stage, program))} only, not {arch!r}"
        )
    return f"{rec[stage]['library']}_{arch}"


@functools.cache
def load_cake_dsa_train_module(stage: str, arch: str, program: str = PROGRAM):
    """The loaded library of ``stage`` of ``program`` on ``arch``: prebuilt from ``aot/``
    when the package ships it, otherwise JIT-built through the package's JIT shim from ``build.json``."""
    return cake_build.load_library(library_name(stage, arch, program))
