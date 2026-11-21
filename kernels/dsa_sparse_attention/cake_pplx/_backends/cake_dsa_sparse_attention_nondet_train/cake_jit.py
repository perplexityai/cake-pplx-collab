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

# Explicit target-owned registration of the generated training programs: one
# record per program (``PROGRAM``: the RoPE family, ``NOPE_PROGRAM``: the
# native zero-RoPE family) shared by every architecture it compiles for
# (``arches``), the kernel ``stages`` it registers in launch order, the host
# policies that select the key-range-pass form of the backward
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``) and the direct
# accumulation into the caller's packed rows through the natural-layout main
# stages (``dkv_direct``, see ``cake_backend.DkvDirectPolicy``: a base
# ``min_keys_per_query`` and, like ``key_pass_policy``, per-architecture
# ``rules`` that replace it on their target with a rule of three fields --
# ``min_keys_per_query``, ``cast_slots_per_key`` and ``cast_min_slots``, which
# weigh the cast path's extra cost, proportional to the key count, against
# the natural drain's, modelled as proportional to the main stage's query x
# selected-slot work -- and an optional key cap ``cast_max_keys``, above which the rule
# keeps the natural drain whatever the work; the base test still applies),
# and one physical entry per stage: the
# ``library`` stem of its ``build.json`` entries (``<library>_<arch>``, see
# ``cake_build``), the FFI entry and the launch
# geometry the binding bakes in (block, cluster, cooperative launch, dynamic
# shared memory, programmatic dependent launch).  A stage entry may narrow the
# record's architectures with its own ``arches``: its library ships for those
# only.  On ``sm_107a`` each forward runs its scanner-warp build, the stage
# ``SCAN_STAGES`` names (``arches`` ``["sm_107a"]``), in place of the forward
# (whose ``arches`` are the other two): ``library_name`` resolves the forward
# there to the scanner stage's library.  Two records may name one
# library (the rope-independent delta, compaction and cast stages serve both
# programs); their launch dicts then agree.  What each library compiles --
# translation units, nvcc flags, digests -- is recorded in ``build.json`` only;
# the positional argument order of every stage lives in the generated
# ``cake_launch`` module.  The records are populated by the generated-program
# export and only their stage names, policies and the pending stages of a
# record are maintained by hand; the export rewrites the ``launch`` leaves.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_main",
            "bwd_main_natural",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_main_pass_natural",
            "bwd_cast",
            "fwd_scan",
            "fwd_no_olo",
            "bwd_delta_no_olo",
            "fwd_scan_no_olo",
        ],
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
        # direct accumulation from 4 keys per query token.  On sm_107a the rule
        # models the natural-layout drain's extra cost as proportional to the main
        # stage's query x selected-slot work, the cast path's as a zero fill and a cast over
        # the S accumulator rows plus two launches: the cast path also serves
        # rows with T * K_eff >= 112 * S + 2^20, K_eff = min(K_slots,
        # ceil(S / num_segments)), without a key cap: the single-segment
        # T = 16,384, S = 278,528 row measured at parity on VR200 (see
        # cake_backend.DkvDirectPolicy and the backend README)
        "dkv_direct": {
            "min_keys_per_query": 4,
            "rules": {
                "sm_107a": {
                    "min_keys_per_query": 4,
                    "cast_slots_per_key": 112,
                    "cast_min_slots": 1048576,
                },
            },
        },
        "fwd": {
            "library": "cake_dsa_h64_train_c949a885132a772bdb53",
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
            "library": "cake_dsa_h64_nope_train_ae9db52dff326ae8409b",
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
            "library": "cake_dsa_h64_train_e29d165e3abca62c430b",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_main_natural": {
            "library": "cake_dsa_h64_train_e29d165e3abca62c430b_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_compact": {
            "library": "cake_dsa_h64_train_19c560118a407f5473fe",
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
            "library": "cake_dsa_h64_train_d6a7bc0b3aa5b7084667",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_main_pass_natural": {
            "library": "cake_dsa_h64_train_d6a7bc0b3aa5b7084667_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_cast": {
            "library": "cake_dsa_h64_train_c96be96662a6d97e0b53",
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
            "library": "cake_dsa_h64_train_491a3b1d55552281da1a",
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
        "fwd_no_olo": {
            "library": "cake_dsa_h64_train_b17222dad27ed54be27a",
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
        "bwd_delta_no_olo": {
            "library": "cake_dsa_h64_nope_train_ca5f47db0a3e8065ce81",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "fwd_scan_no_olo": {
            "library": "cake_dsa_h64_train_7e29ce0f6edc6dae144f",
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
    },
    # The native zero-RoPE (NoPE) training program: shared latent K = V of 512
    # dimensions, no rope operands, one combined FP32 dK + dV gradient of 2048 B
    # per key.  Its key-range-pass policy is re-calibrated for that row size
    # (``key_bytes`` 2048; ``dq_partial_bytes_per_token`` = 64 heads x 512 x 4 B
    # per token of FP32 dQ partial).  Measured on the acceptance GPUs, one key
    # pass is the fastest plan on every row and build (``max_passes`` 1: the
    # policy never plans a second pass; ``key_passes`` still forces the pass
    # form, whose stages and scratch stay registered) and, on ``sm_100a`` /
    # ``sm_103a``, the natural-layout main stage beats the permuted accumulator
    # plus cast whenever the row has at least one key per query
    # (``min_keys_per_query`` 1).  On ``sm_107a`` the rule models the natural-layout
    # drain's extra cost as proportional to the main stage's work (T x K_eff query x
    # selected-slot pairs, K_eff = min(K_slots, ceil(S / num_segments))), the
    # cast path a zero fill and a cast proportional to S plus two launches, so
    # the ``sm_107a`` rule takes the permuted accumulator plus cast where
    # S < T (the base test) or where T * K_eff >= 112 * S + 2^20 and
    # S <= 131,072 (``cast_max_keys``), and the natural stage otherwise.  On
    # long single-segment rows the cast path's advantage is smaller at the
    # largest key counts and turns into a loss (one segment, K_slots 2176, natural / cast
    # backward: T = 32,768, 19.660 / 19.156 ms at S = 131,072 and
    # 19.520 / 19.910 ms at 163,840; T = 16,384, 9.940 / 9.791 ms at 163,840
    # and 10.781 / 11.209 ms at 262,144); the cap counts the total S, since
    # the planners do not know the longest segment of a packed row.  On VR200
    # the rule picks the cast path on no measured row where it was slower, and
    # keeps the natural stage on small rows, small top-k and short-document
    # packs, where the cast path measured up to 8 % slower in backward, and on
    # rows above the cap.  The
    # compaction and cast stages are rope-independent kernels shared with the
    # RoPE program (same libraries, same launch geometry; the cast runs over the
    # one 512-wide accumulator, 16 groups per key); the delta stage is the strided delta
    # kernel, shared with the RoPE record like compact and cast (both records'
    # ``bwd_delta`` / ``nope_bwd_delta`` name one library; 256 threads: it
    # reads ``dout`` through its token and head strides, so a head-major
    # gradient view is consumed in place by both programs);
    # the forward (320 threads, 222,464 B dynamic shared memory) and the four
    # main backward forms (640 threads, 214,208 B; permuted / natural-layout
    # accumulation, single-pass / key-range-pass) are the NoPE kernels.  A call
    # with a trainable per-head sink runs the forward's sink build
    # (``nope_fwd_sink``: the forward's arguments plus the FP32 ``sink [64]``,
    # the forward's launch geometry; ``sm_100a`` / ``sm_103a``) in place of
    # ``nope_fwd`` -- on ``sm_107a`` its scanner-warp build
    # (``nope_fwd_scan_sink``, 352 threads), as ``nope_fwd_scan`` serves
    # ``nope_fwd`` there -- and, in its backward, the sink-gradient stage
    # (``nope_bwd_dsink``, 256 threads, every architecture of the record) once
    # right after the delta stage.  Stage order = launch order, the sink
    # stages last among the default stages, then the ``with_o_lo=False``
    # stages (``NO_OLO_STAGES``).  ``build.json`` carries every library named here
    # once the export delivers it; ``scripts/check_cake_build_manifests.py``
    # reports a registry library the manifest lacks.
    "cake_dsa_h64_nope_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "index_prep",
            "nope_fwd",
            "nope_bwd_delta",
            "nope_bwd_main",
            "nope_bwd_main_natural",
            "nope_bwd_compact",
            "nope_bwd_main_pass",
            "nope_bwd_main_pass_natural",
            "nope_bwd_cast",
            "nope_fwd_scan",
            "nope_fwd_sink",
            "nope_bwd_dsink",
            "nope_fwd_scan_sink",
            "nope_fwd_no_olo",
            "nope_bwd_delta_no_olo",
            "nope_fwd_scan_no_olo",
            "nope_fwd_sink_no_olo",
            "nope_fwd_scan_sink_no_olo",
        ],
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2048,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
            "max_passes": 1,
            "dq_partial_bytes_per_token": 131072,
        },
        "dkv_direct": {
            "min_keys_per_query": 1,
            "rules": {
                "sm_107a": {
                    "min_keys_per_query": 1,
                    "cast_slots_per_key": 112,
                    "cast_min_slots": 1048576,
                    "cast_max_keys": 131072,
                },
            },
        },
        "index_prep": {
            "library": "cake_dsa_h64_nope_train_db791110b21a91be1388",
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
            "library": "cake_dsa_h64_nope_train_46cdc056c891876ed545",
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
            "library": "cake_dsa_h64_nope_train_ae9db52dff326ae8409b",
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
            "library": "cake_dsa_h64_nope_train_d7b52eccaf718555e88f",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_main_natural": {
            "library": "cake_dsa_h64_nope_train_d7b52eccaf718555e88f_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_compact": {
            "library": "cake_dsa_h64_train_19c560118a407f5473fe",
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
            "library": "cake_dsa_h64_nope_train_90ab03044736858291f5",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_main_pass_natural": {
            "library": "cake_dsa_h64_nope_train_90ab03044736858291f5_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_cast": {
            "library": "cake_dsa_h64_train_c96be96662a6d97e0b53",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_fwd_scan": {
            "library": "cake_dsa_h64_nope_train_d13be3d78a70edab8c99",
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
            "library": "cake_dsa_h64_nope_train_f41a0f9f0e10448aaf94",
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
            "library": "cake_dsa_h64_nope_train_36d862698eea1ea770ef",
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
            "library": "cake_dsa_h64_nope_train_e1e7a7620a259632759f",
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
        "nope_fwd_no_olo": {
            "library": "cake_dsa_h64_nope_train_3571d6ea545eb531e395",
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
        "nope_bwd_delta_no_olo": {
            "library": "cake_dsa_h64_nope_train_ca5f47db0a3e8065ce81",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_fwd_scan_no_olo": {
            "library": "cake_dsa_h64_nope_train_0f925280bc0271d524dd",
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
        "nope_fwd_sink_no_olo": {
            "library": "cake_dsa_h64_nope_train_4ae850190859722841c2",
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
        "nope_fwd_scan_sink_no_olo": {
            "library": "cake_dsa_h64_nope_train_e120e5ca8bd3ae1fa358",
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

PROGRAM = "cake_dsa_h64_train"
NOPE_PROGRAM = "cake_dsa_h64_nope_train"
PROGRAMS = (PROGRAM, NOPE_PROGRAM)
STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_main_natural",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_main_pass_natural",
    "bwd_cast",
)
FORWARD_STAGES = ("fwd",)
# The NoPE program names its stages ``nope_<stage>`` (globally unique: the
# generated launch tables are keyed by stage name across both programs).
STAGE_PREFIX = {PROGRAM: "", NOPE_PROGRAM: "nope_"}
NOPE_STAGES = tuple(f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}" for stage in STAGES)
NOPE_FORWARD_STAGES = NOPE_STAGES[:1]
# The scanner-warp builds of the two forwards and of the NoPE forward's sink
# build, then those of their ``with_o_lo=False`` builds (``NO_OLO_STAGES``):
# on the architectures of their stage entries' ``arches``
# (``sm_107a``) the host runs them in place of the forward stage; the
# device's architecture alone chooses.  Same arguments, grid and outputs as
# the forward they replace.
SCAN_STAGES = {
    "fwd": "fwd_scan",
    "nope_fwd": "nope_fwd_scan",
    "nope_fwd_sink": "nope_fwd_scan_sink",
    "fwd_no_olo": "fwd_scan_no_olo",
    "nope_fwd_no_olo": "nope_fwd_scan_no_olo",
    "nope_fwd_sink_no_olo": "nope_fwd_scan_sink_no_olo",
}
# The NoPE program's trainable-sink stages: the forward's sink build, which runs in place of ``nope_fwd`` when a call
# passes a sink (same arguments plus ``sink``, same grid and outputs; its ``lse`` includes the sink; its scanner-warp
# build ``SCAN_STAGES["nope_fwd_sink"]`` where that ships), and the sink-gradient stage the backward of such a call
# launches once after the delta stage.  Their entries' ``arches`` name the architectures that ship them.
NOPE_SINK_FORWARD_STAGE = "nope_fwd_sink"
NOPE_DSINK_STAGE = "nope_bwd_dsink"
NOPE_SINK_STAGES = (NOPE_SINK_FORWARD_STAGE, NOPE_DSINK_STAGE)
# The stages a call with ``with_o_lo=False`` runs in place of the default ones: the forward builds that leave the
# ``o_lo`` operand unwritten and the delta that reads ``out`` alone (same arguments and grid as the default stage).
# Each record lists them after its default stages, in their order.
NO_OLO_STAGES = {
    "fwd": "fwd_no_olo",
    "bwd_delta": "bwd_delta_no_olo",
    "fwd_scan": "fwd_scan_no_olo",
    "nope_fwd": "nope_fwd_no_olo",
    "nope_bwd_delta": "nope_bwd_delta_no_olo",
    "nope_fwd_scan": "nope_fwd_scan_no_olo",
    "nope_fwd_sink": "nope_fwd_sink_no_olo",
    "nope_fwd_scan_sink": "nope_fwd_scan_sink_no_olo",
}
_DEFAULT_STAGES_OF = {
    PROGRAM: STAGES + (SCAN_STAGES["fwd"],),
    NOPE_PROGRAM: NOPE_STAGES
    + (SCAN_STAGES["nope_fwd"],)
    + NOPE_SINK_STAGES
    + (SCAN_STAGES[NOPE_SINK_FORWARD_STAGE],),
}
STAGES_OF = {
    program: order + tuple(NO_OLO_STAGES[s] for s in order if s in NO_OLO_STAGES)
    for program, order in _DEFAULT_STAGES_OF.items()
}


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
