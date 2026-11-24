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

Backend suite of the Cake non-deterministic packed KDA chunked training package
(``cake_pplx._backends.cake_kda_nondet_train``).

The suite mirrors the deterministic package's suite (``test_cake_kda_det_train.py``): the same shapes, ragged packs,
short documents, tails, document isolation, workspace reuse, poisoned arenas, lean and batched arenas and CUDA graphs.
Accuracy: ``out`` and every gradient of every path are judged against the exact FP64 per-document recurrence of the
same inputs beside the naive BF16 tier (``judge_arms`` of ``test_helpers/cake_kda_naive.py``: over the judge's
seeded draws, the median and the largest L2 error and the paired max-norm error).  Each path is an arm judged under
its own name; repeated launches of an arm are judged each and the worst is kept.  The package is non-deterministic: no
result is held to a bit pattern, from run to run or against the deterministic package.  The bit checks that remain
compare operands, integer tables, NaN write patterns and records.  The judged rows are those up to about 4096 tokens at
any head count and one production-size row per route (up to about 16k tokens at up to four heads); a larger row keeps
its exact checks and finite results, and its test names the judged row of the same route.  The states-saving route
of the 16-head rows and of the batched groups is judged on short packs under the size rule lowered for the test
(``seg_save_rule``: the rule chooses where the states are kept, not what they are).

The package's backward runs the S1 schedule on the S1 cells: the ``q * dq`` term of ``dg`` moves from the fused
gradient (``dqkg``) into the GLM intra program.  The suite adds the S1 contract: the dispatch of the S1 pairs (the
fused gradient and the GLM intra program always agree on S1), the saved sets and the replay judged on the cells of
both pairs, the worst case over repeated runs, the launches of the S1 records on exactly the S1 cells, and gradient
accumulation into a nonzero ``.grad``.  The checkpoint hand-off's intra twins (``cake_jit.HANDOFF_TWINS``): their
records against their intra records, no rule selecting them, and on a GPU the rows they write against the light
prep's.  The tape path's w-only WY twin (``cake_jit.TAPE_TWINS``): its record against its WY record, no rule selecting
it, the rule that launches it, and on a GPU the elements of ``w`` it writes and the tape backward's gradients judged at
every head count.  The in-kernel Q/K L2 norm (``use_qk_l2norm_in_kernel``): its records are the deterministic
package's, record for record (its output backward ``dav_l2n*`` and its fused gradient ``dqkg_*l2n`` among them), and
the raw class takes twins of the l2n fused gradient records (``RAW_L2N_DQKG``), so an l2n mode runs the deterministic
pair on every row; its host and device rows are in
``test_cake_kda_nondet_train_l2n.py``.  The deterministic package runs here only in mode contracts: the records and
launch lists this package copies, and each package's backward reading its own saved sets only (a set of the other
package is refused with ``ValueError`` naming the mode).  The facade, the refusal of the deterministic package's objects
and torch's deterministic mode are covered by ``test_cake_kda_nondet_train_api.py``, the names by
``test_cake_kda_nondet_train_names.py``; the hand-off itself by ``test_cake_kda_nondet_train_handoff.py``.
"""

import dataclasses
import hashlib
import itertools
import json
import os
import re
import types
import warnings
import zlib
from pathlib import Path

import pytest
import torch

from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
    make_raw_inputs,
)
from tests.experimental.test_helpers.cake_kda_naive import draw_case, judge_arms
from tests.cake.test_helpers.cake_package import cake_package

# The backend suite names its package once, through the shared alias helper, and imports nothing else from
# cake_pplx: a build that vendors the package under another import path runs it unchanged after pointing the
# ``PACKAGES`` entry at its own package.  The public entry points of ``cake_pplx.cake_kda_nondet_train`` are covered by
# ``test_cake_kda_nondet_train_api.py``.
cake_backend = cake_package("cake.cake_kda_nondet_train", "cake_backend")
cake_build = cake_package("cake.cake_kda_nondet_train", "cake_build")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")
# The deterministic package: the source of the records this package copies and the other package of the mode
# contracts (its records, launch lists and refusals); none of its results is compared with this package's.
det_backend = cake_package("cake.cake_kda_det_train", "cake_backend")
det_jit = cake_package("cake.cake_kda_det_train", "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent


# the package's entries (they take no mode argument)
cake_kda_nondet_train_forward = cake_backend.cake_kda_nondet_train_forward
cake_kda_nondet_train_backward = cake_backend.cake_kda_nondet_train_backward
cake_kda_nondet = cake_backend.cake_kda_nondet


ARCHES = tuple(cake_jit.ARCHES)
PATHS = ("fwd", "fwd_nosave", "light", "light_states", "tape", "replay")
BT, D, SCALE = cake_backend.BT, cake_backend.D, cake_backend.SCALE
CakeKDANondetChunkSaved = cake_backend.CakeKDANondetChunkSaved
CakeKDANondetWorkspace = cake_backend.CakeKDANondetWorkspace
# Each package's entries refuse the other package's saved set, workspace and arena option with ``ValueError`` naming
# the mode: this package's message, then the deterministic package's.
ND_REFUSES = "is not a non-deterministic-package"
DET_REFUSES = "is not a deterministic-package"


# Dispatch constants the registry rules must reproduce (the kernels' validated regimes).
TABLE_V2_MAX_DOCS = 256  # the chunk-parallel layout table serves up to 256 documents
RECUR_CTAS_PER_SM = (
    4  # the 32-column serial forms fit when num_docs * H * 4 <= 4 * sm_count
)
RECUR_SLICES = {"v1": 2, "v2": 4}
FUSED_FWD_MIN_H = 32  # the serial forward forms the WY representation itself from 32 heads (sm_107a: 32 only)
FUSED_BWD_MIN_H = 64  # the 64-column serial recompute forms it from 64 heads
# The forward's carry || pass-2 overlap pair on the segmented route: the rows whose pass-2 grid is at least
# OV1_MIN_WAVES_BY_ARCH[arch] waves of the device and at most OV1_MAX_SEGMENT_HEADS segment-heads (the flag words of the
# eye buffer); the pass-2 CTA order is the carry's release order up to OV1_ORDER_MAX_H heads, the chain-major order
# above.  The saving forward launches the pair's token-row twins (the saved Aqk rows) and the saved-Akk WY program.
OV1_MIN_WAVES_BY_ARCH = {"sm_100a": 3.0, "sm_103a": 3.0, "sm_107a": 2.3}
OV1_ORDER_MAX_H = 2
OV1_MAX_SEGMENT_HEADS = 8192
PDL_RECORDS = (
    "fwdh_p3_o_ov",
    "fwdh_p3_o_ov_id",
    "fwdh_p3_o_ov_rows",
    "fwdh_p3_o_ov_id_rows",
)
# the saving forward's token-row records: where one runs, no program of the path reads the internal Aqk / Akk copies;
# the forward without a saved set launches their internal-row twins instead (the paired serial rows -- the 32-column
# form below the fused threshold, and on sm_107a the 64-column form too -- launch wy_fwd_rows and fwdh_o_v2_rows /
# fwdh_o_v1_rows; the 64-column form keeps the internal rows on the other architectures)
TOKEN_ROW_RECORDS = (
    "fwdh_wy_rows",
    "wy_fwd_rows",
    "fwdh_p3_o_ov_rows",
    "fwdh_p3_o_ov_id_rows",
    "fwdh_o_v2_rows",
    "fwdh_o_v1_rows",
)
INTERNAL_ROW_TWIN = {
    "fwdh_wy_rows": "fwdh_wy",
    "wy_fwd_rows": "wy_fwd",
    "fwdh_p3_o_ov_rows": "fwdh_p3_o_ov",
    "fwdh_p3_o_ov_id_rows": "fwdh_p3_o_ov_id",
    "fwdh_o_v2_rows": "fwdh_o_v2",
    "fwdh_o_v1_rows": "fwdh_o_v1",
}
# the architectures whose saving forward reads the saved token rows on the 64-column serial rows too (fwdh_o_v1_rows)
PAIRED_V1_ARCHES = ("sm_107a",)
DAVQ_MIN_H = 16  # the fused gradient hosts dAqk on the serial route from 16 heads
# from this many heads the fused gradient runs with its L2 prefetch / evict-first hints off (the `_nohint` records:
# the same kernel built with the hint thresholds beyond any head count) on either route (the 16-head rows of
# sm_107a are segmented ones); the other architectures keep the hints
DQKG_NOHINT_MIN_H_BY_ARCH = {"sm_107a": 16}
FEW_CHAINS = 4  # at most four (document, head) chains: the few-chain segmented rows
# the saved-state pass 2 forms the WY representation itself (no WY launch) up to 32 (document, head) chains; on
# every segmented row the adjoint recurrence accumulates dv and the fused gradient hosts dAqk (no dav launch)
FUSED_P3_MAX_CHAINS = 32
DHU_TMA_WAVES = (
    4  # the segmented adjoint recurrence stores its tapes by TMA below four CTAs per SM
)
# on sm_103a the segmented adjoint recurrence of the rows below two waves of CTA pairs (two per SM: four of one CTA)
# prefetches the next chunk's tiles (dhu_v3p_f*); the plain TMA-tape form keeps the boundary rows there
DHU_PREFETCH_WAVES_BY_ARCH = {"sm_103a": 4.0}
# The segmented recurrences: at most SEG_MAX_H heads, or a head count of SEG_PLAN_HEADS_BY_ARCH on its architecture
# (the 16-head rows of sm_103a / sm_107a: their serial walk is the slower form there; the registry's rules read the
# route, not the head bound), clamp(ceil(SEG_SLOT_FACTOR * sm / (N * H)), 1, SEG_P_MAX) segments per document, at
# most SEG_P_SMALL of them with N * H <= SEG_P_SMALL_MAX_NH (document, head) chains.
SEG_MAX_H = 8
SEG_PLAN_HEADS_BY_ARCH = {"sm_103a": (16,), "sm_107a": (16,)}
SEG_SLOT_FACTOR = 4
SEG_P_MAX = 64
SEG_P_SMALL = 38
SEG_P_SMALL_BY_ARCH = {
    "sm_100a": 37
}  # the few-chain count of the architectures with one of their own
SEG_P_SMALL_MAX_NH = 4
# The two-level segment carry (seg_carry_v8): the segments of each document's chain in groups of g, the groups'
# products formed in parallel and chained, the CTAs handing over through CARRY_FLAG_WORDS_PER_CTA release-flag
# words each under a fresh epoch.  Dispatched on every architecture (CARRY_V8_ARCHES) at most CARRY_V8_MAX_H heads,
# on the segmented rows that are not the forward's overlap rows (those keep seg_carry_ov), where the host rule
# accepts: the shortest critical path that fits one CTA per SM gains more than CARRY_S_HOPS hops over the chain
# (_carry_choice).
CARRY_V8_ARCHES = ARCHES
CARRY_V8_MAX_H = 2
CARRY_S_HOPS = 14
CARRY_FLAG_WORDS_PER_CTA = 3
# the architectures whose GLM intra program (glm_intra_v14 on sm_103a, glm_intra_v18 on sm_107a: the same program
# with every named-barrier slot an immediate, two CTAs per SM there) writes the gate gradients itself: no gate_epilogue
# launch, no dg_int / db_int buffers in their layouts; sm_100a keeps glm_intra + gate_epilogue.  A byte query
# without an architecture takes the first exported one's answer.
GATE_FUSED_ARCHES = ("sm_103a", "sm_107a")
GATE_FUSED_RECORD = {"sm_103a": "glm_intra_v14", "sm_107a": "glm_intra_v18"}
# The S1 schedule of the backward (this package's non-deterministic mode): the fused gradient's dAqk-hosting forms
# without the ``q * dq`` term of ``dg`` (dqkg v24: the same operands but ``q``) and the GLM intra programs that add it
# (glm_intra_v21 on sm_100a, unfused like glm_intra with the gate epilogue after it; glm_intra_v22 / glm_intra_v23
# on sm_103a / sm_107a, gate-fused like glm_intra_v14 / glm_intra_v18).  The deterministic-form record -> its S1 twin:
S1_DQKG = {
    "dqkg_davq": "dqkg_davq_v24",
    "dqkg_int_davq": "dqkg_int_davq_v24",
    "dqkg_davq_nohint": "dqkg_davq_nohint_v24",
    "dqkg_int_davq_nohint": "dqkg_int_davq_nohint_v24",
}
S1_GLM = {
    "sm_100a": "glm_intra_v21",
    "sm_103a": "glm_intra_v22",
    "sm_107a": "glm_intra_v23",
}
# the GLM intra program of the deterministic pair on each architecture (byte copies of the deterministic package's)
DET_GLM = {"sm_100a": "glm_intra", **GATE_FUSED_RECORD}
S1_RECORDS = frozenset(S1_DQKG.values()) | frozenset(S1_GLM.values())
# The S1 cells, a pure function of (architecture, path, heads); every other backward cell runs the deterministic
# pair.  The saved-set paths (light, light_states, tape) run S1 from S1_SAVED_MIN_H heads, the replay (no saved set)
# up to S1_REPLAY_MAX_H heads (None: every head count); the serial rows below DAVQ_MIN_H heads (9..15) run the
# deterministic pair on every architecture.  The sm_107a cells are provisional (their record rows are pending).
SAVED_PATHS = ("light", "light_states", "tape")
S1_SAVED_MIN_H = {"sm_100a": 16, "sm_103a": 1, "sm_107a": 1}
S1_REPLAY_MAX_H = {"sm_100a": None, "sm_103a": 8, "sm_107a": None}
# The light saved set (version 4) carries the fold's qg on every row (the light backward's prep, bwd_prep_g8_noqg,
# writes none) and the WY representation w / u on the paired serial WU_SAVED_H-head rows (the 32-column form) of
# WU_SAVED_ARCHES and on the segmented WU_SAVED_H-head rows of WU_SEG_SAVED_ARCHES above WU_SEG_MIN_CHAINS (document,
# head) chains (the rows above the fused saved-state pass 2's bound): their prep is bwd_prep_g8_noqg_nokb on both
# light paths and their light backward launches no wy program.
WU_SAVED_ARCHES = ("sm_100a",)
WU_SAVED_H = 16
WU_SEG_SAVED_ARCHES = ("sm_103a", "sm_107a")
WU_SEG_MIN_CHAINS = 32
# the table stage is optional too: on the forward and the replay of the segmented route up to 256 documents the
# intra program writes the tables and no table record is selected; the gate epilogue launches on sm_100a alone (and
# never in raw mode or an in-kernel Q/K L2 norm mode), the parameter finalize in raw mode alone
OPTIONAL_STAGES = (
    "table",
    "seg_table",
    "seg_carry",
    "fwdh_p3",
    "wy_fwd",
    "wy",
    "fwdh",
    "dav",
    "gate_epilogue",
    "param_finalize",
    # the in-kernel Q/K L2 norm's token-local stages (#7817): dispatched in the fallback mode alone
    "qk_l2norm",
    "qk_apply",
    "qk_pullback",
)
# The stages of each path, in the registry's order (the forward's; the backward's with the tables, the intra
# program on the replay path and seg_carry on the segmented route).
_BWD = (
    "table",
    "seg_table",
    "wy",
    "fwdh",
    "seg_carry",
    "fwdh_p3",
    "bwd_prep",
    "dav",
    "dhu",
    "dqkg",
    "glm_intra",
    "gate_epilogue",
    "param_finalize",
)
PATH_STAGES = {
    "fwd": (
        "table",
        "seg_table",
        "fwd_intra",
        "wy_fwd",
        "fwdh",
        "seg_carry",
        "fwdh_p3",
    ),
    "light": _BWD,
    "light_states": _BWD,
    "tape": _BWD,
    "replay": tuple(s for s in cake_jit.STAGES if s in _BWD or s == "fwd_intra"),
}
# the forward without a saved set: the same stages, the internal-row records
PATH_STAGES["fwd_nosave"] = PATH_STAGES["fwd"]
# the backward's chunk-parallel tail (once per range of a lean workspace's tail passes); the gate epilogue where
# the architecture dispatches it
TAIL_STAGES = ("dqkg", "glm_intra", "gate_epilogue")

# Small packings: single documents around the chunk length, packs with unaligned tails, a zero-length document
# (a no-op that still counts in the chunk bound) and pad chunks.
SHAPES = [
    ((1,), 1),
    ((63,), 2),
    ((64,), 1),
    ((65,), 4),
    ((127,), 1),
    ((128,), 2),
    ((129,), 1),
    ((1000,), 1),
    ((63, 1, 65), 4),
    ((64, 64), 1),
    ((100, 0, 300), 2),
    ((100, 640, 77, 1000, 129), 2),
    # the overlap pair's rows (148 SMs): four documents at 2 heads (256 segments: the release order), seven at 8
    # (77 segments: the chain-major order)
    ((63, 1, 65, 30), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
]


def _shape_id(shape) -> str:
    lengths, heads = shape
    return "s" + "_".join(str(n) for n in lengths) + f"-h{heads}"


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {ARCHES}"
        )
    return torch.device("cuda", index)


def _current_arch() -> str:
    """The architecture :func:`cake_backend.plan` takes without ``arch``: the current CUDA device's where it is an
    exported target, else the first exported architecture."""
    if torch.cuda.is_available():
        arch = cake_jit.arch_for_capability(
            torch.cuda.get_device_capability(torch.cuda.current_device())
        )
        if arch is not None:
            return arch
    return ARCHES[0]


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _manifest_or_skip()


# --------------------------------------------------------------------------- registry and plan (no device)


def _rule_kinds(rule: dict) -> list[str]:
    kinds = [rule["kind"]]
    if rule["kind"] in ("all_of", "any_of"):
        for sub_rule in rule["rules"]:
            kinds.extend(_rule_kinds(sub_rule))
    return kinds


# the replay prep's saved-rstd twin (the deterministic package's copy, launched by name by the KCP backward in place of
# its record when it reads the saved rstd): no plan's form
FUSED_L2N_TWINS = {"bwd_prep_g8_replay_saved_l2n": "bwd_prep_g8_replay_l2n"}

SELECT_KINDS = (
    "always",
    "otherwise",
    "all_of",
    "any_of",
    "ctas_at_least",
    "ctas_below",
    "ctas_between",
    "docs_at_most",
    "docs_above",
    "heads_at_most",
    "heads_above",
    "heads_at_least",
    "heads_below",
    "chains_at_most",
    "chains_above",
    "slots_at_most",
    "slots_above",
    "walk_ctas_below",
    "walk_ctas_at_most",
    "walk_ctas_at_least",
    "walk_ctas_above",
    "segments_heads_at_most",
    "segments_heads_above",
    "route_in",
    "carry_two_level",
    "carry_chain",
    "arch_in",
    "path_in",
    "l2n_in",
    "saved_rstd_twin",  # a record launched by name in place of another, never selected (#7817)
)
GRID_KINDS = (
    "fixed",
    "chunks",
    "chunk_blocks",
    "pairs_bounded",
    "pair_items",
    "per_doc_head",
    "sm_bounded",
    "chunk_items",
    "docs_heads",
    "carry_groups",
    "segments_heads",
    "head_chunk_blocks",
    "tokens",  # the in-kernel Q/K L2 norm's token-local fallback programs (#7817)
)
# the records delivered for a subset of the architectures (every other record ships on all three, the two-level
# carry and the no-kb prep among them): the gate epilogue and its unfused GLM intra programs on sm_100a, the fused
# ones on sm_103a (glm_intra_v14, glm_intra_v22) and sm_107a (glm_intra_v18, glm_intra_v23), and the S1 fused
# gradient's builds with the L2 hints off on sm_107a (its rows from 16 heads).  The deterministic copies keep the
# deterministic package's architectures (the same builds; dqkg_davq / dqkg_int_davq share their programs with the
# raw class, so they ship on the three though their g rules launch them on sm_100a / sm_103a alone; the in-kernel
# Q/K L2 norm's sm_103a forms fwd_intra_v37_l2n and glm_intra_v30_l2n on sm_103a alone, and its replay prep's
# saved-rstd twin on sm_100a / sm_103a), and the raw class's two fused gradient builds with the L2 hints off and the
# two l2n fused gradient builds with the L2 hints off (with their raw twins) ship on sm_107a alone, as the
# deterministic package's do.
RECORD_ARCHES = {
    "glm_intra": ("sm_100a",),
    "glm_intra_v14": ("sm_103a",),
    "glm_intra_v18": ("sm_107a",),
    "gate_epilogue": ("sm_100a",),
    "glm_intra_v21": ("sm_100a",),
    "glm_intra_v22": ("sm_103a",),
    "glm_intra_v23": ("sm_107a",),
    "dqkg_davq_nohint_v24": ("sm_107a",),
    "dqkg_int_davq_nohint_v24": ("sm_107a",),
    "dqkg_davq_nohint_raw": ("sm_107a",),
    "dqkg_int_davq_nohint_raw": ("sm_107a",),
    "fwd_intra_v37_l2n": ("sm_103a",),
    "glm_intra_v30_l2n": ("sm_103a",),
    "bwd_prep_g8_replay_saved_l2n": ("sm_100a", "sm_103a"),
    "dqkg_nohint_l2n": ("sm_107a",),
    "dqkg_int_nohint_l2n": ("sm_107a",),
    "dqkg_nohint_raw_l2n": ("sm_107a",),
    "dqkg_int_nohint_raw_l2n": ("sm_107a",),
}
# The deterministic package's records no cell of this package launches: the sm_107a rows from 16 heads (the only
# rows of the builds with the L2 hints off) run the S1 pair on every backward path of the g mode with the in-kernel
# Q/K L2 norm off (their programs ship as the raw class's, RAW_DQKG).  The in-kernel Q/K L2 norm's records (#7817:
# its fused class, 7 g records with their 7 raw twins, the sm_103a forward intra and GLM intra forms and the replay
# prep's saved-rstd twin, the 2 GLM intra capture forms and the 3 programs of its fallback route; its output
# backward, the 5 dav_l2n records, and its 4 fused gradient records) are copied, record for record.
DET_RECORDS_NOT_LAUNCHED = frozenset(("dqkg_davq_nohint", "dqkg_int_davq_nohint"))
# The in-kernel Q/K L2 norm (``use_qk_l2norm_in_kernel``): the modes, and the stages that select within the class of
# the call's mode (a record's ``l2n``, ``["off"]`` when absent) -- the deterministic package's (the output backward
# ``dav`` and the fused gradient among them; the fused gradient's S1 records serve ``"off"`` alone: no GLM intra
# program of an l2n mode adds the ``q * dq`` term), and the parameter finalize, whose ``param_finalize`` serves
# ``"off"`` alone (an l2n mode's raw GLM intra program writes the deterministic per-chunk partials, which
# ``param_finalize_l2n``, the same program, reduces).
L2N_MODES = ("off", "fused", "fallback")
L2N_ON = ("fused", "fallback")
L2N_STAGES = frozenset(
    {
        "fwd_intra",
        "bwd_prep",
        "dav",
        "glm_intra",
        "qk_l2norm",
        "qk_apply",
        "qk_pullback",
        "dqkg",
        "param_finalize",
    }
)
# The token-local stages of the fallback route: dispatched in the "fallback" mode alone.
L2N_FALLBACK_STAGES = ("qk_l2norm", "qk_apply", "qk_pullback")
# The raw forget-gate mode (``use_gate_in_kernel=True``): the stages whose records come in a ``g`` and a ``raw`` class
# (a record's ``gate``, ``"g"`` when absent) -- the deterministic package's and the fused gradient, whose S1 records
# serve the activated gate alone -- and the fused gradient's raw class with the in-kernel Q/K L2 norm off, record ->
# the deterministic record it copies (rule, plan and program: raw mode runs the deterministic pair on every row,
# glm_intra_raw adds no q * dq term).
GATE_STAGES = frozenset(
    {"fwd_intra", "bwd_prep", "dqkg", "glm_intra", "gate_epilogue", "param_finalize"}
)
GATES = ("g", "raw")
RAW_DQKG = {
    "dqkg_raw": "dqkg",
    "dqkg_int_raw": "dqkg_int",
    "dqkg_davq_raw": "dqkg_davq",
    "dqkg_int_davq_raw": "dqkg_int_davq",
    "dqkg_davq_nohint_raw": "dqkg_davq_nohint",
    "dqkg_int_davq_nohint_raw": "dqkg_int_davq_nohint",
}
# the raw-class fused gradient records that share their program with a g-class record of the same build
RAW_DQKG_SHARED = ("dqkg_raw", "dqkg_int_raw", "dqkg_davq_raw", "dqkg_int_davq_raw")
# The fused gradient's l2n records (``l2n`` ``["fused", "fallback"]``, built with ``davq`` 0: no dAqk) are the
# deterministic package's records of the same names in the g class; their raw-class twins, record -> the
# deterministic record it copies (rule, plan, program and modes; the class and its own variant): an l2n mode runs
# the deterministic pair on every row in both gate modes.  Each twin shares its program with the g-class copy.
RAW_L2N_DQKG = {
    "dqkg_raw_l2n": "dqkg_l2n",
    "dqkg_int_raw_l2n": "dqkg_int_l2n",
    "dqkg_nohint_raw_l2n": "dqkg_nohint_l2n",
    "dqkg_int_nohint_raw_l2n": "dqkg_int_nohint_l2n",
}
# the deterministic package's l2n fused gradient records, copied under their own names
DET_L2N_DQKG = tuple(RAW_L2N_DQKG.values())
# The parameter finalize of the raw class's l2n modes, record -> the deterministic record it copies (rule, plan,
# architectures and program, shared with ``param_finalize``, which serves the flag-off mode): every mode's raw
# backward takes the deterministic finalize on every architecture.
L2N_PARAM = {"param_finalize_l2n": "param_finalize"}
# The deterministic record of every copied record (the fused gradient's raw class and its l2n twins and the l2n
# parameter finalize: their RAW_DQKG / RAW_L2N_DQKG / L2N_PARAM record; every other copy: its own name).
DET_SOURCE = {**RAW_DQKG, **RAW_L2N_DQKG, **L2N_PARAM}
# The checkpoint hand-off's intra twins, intra record -> twin: new programs that also write every gk row and the
# internal beta (``beta_int``); no path selects them, the hand-off's recompute launches one in place of its record.
HANDOFF_TWINS = {"fwd_intra": "fwd_intra_v20", "fwd_intra_v14": "fwd_intra_v19"}
TWIN_RECORDS = frozenset(HANDOFF_TWINS.values())
# The tape path's w-only WY twin, WY record -> twin: a new program, the record's without the u half (no v / beta
# operands, no u output); no path selects it, the tape backward launches it in place of its record.
TAPE_TWINS = {"wy_bwd": "wy_bwd_v9"}
TAPE_WY_RECORDS = frozenset(TAPE_TWINS.values())
# The logical names of the WY record's parameters the twin does not take: u and the operands only u reads.
TAPE_WY_DROPPED = frozenset({"v", "beta", "u"})
# The raw gate's parameter-gradient pair with the in-kernel Q/K L2 norm off, on every architecture, path and head
# count: the deterministic package's GLM intra program, writing the per-chunk partials dA_part / db_part, and its
# finalize, reducing them in a fixed order (the deterministic records' rules: always, and the backward paths).
RAW_PARAM_PAIR = {"glm_intra": "glm_intra_raw", "param_finalize": "param_finalize"}
RAW_BWD_RULE = {"kind": "path_in", "paths": ["light", "light_states", "tape", "replay"]}
# The registry's size: the deterministic package's 107 records (the raw class's kb-once replay prep
# bwd_prep_g8_replay_kbonce_raw_l2n among them), less those, plus the seven S1 records, the hand-off twins, the tape
# WY twin, the fused gradient's raw class and its l2n twins and the l2n parameter finalize.
MODULE_COUNT = (
    107
    - len(DET_RECORDS_NOT_LAUNCHED)
    + 7
    + len(TWIN_RECORDS)
    + len(TAPE_WY_RECORDS)
    + len(RAW_DQKG)
    + len(RAW_L2N_DQKG)
    + len(L2N_PARAM)
)
# The records of the segmented adjoint recurrence launch twice per stage (the carry between the passes); the
# forward recurrence's passes are two stages (fwdh, fwdh_p3).
TWO_PASS_RECORDS = (
    "dhu_v3_f",
    "dhu_v3_f_wt",
    "dhu_v3_f_replay",
    "dhu_v3p_f",
    "dhu_v3p_f_wt",
    "dhu_v3p_f_replay",
    "dhu_v3r_f",
    "dhu_v3r_f_wt",
    "dhu_v3r_f_replay",
    "dhu_v3s_f",
    "dhu_v3s_f_wt",
    "dhu_v3s_f_replay",
)


def _gate_fused(arch: str | None) -> bool:
    """The backend's rule, spelled independently: whether the GLM intra program of ``arch`` fuses the gate
    epilogue (``arch=None``: the first exported architecture's answer, as the byte queries take it)."""
    return (ARCHES[0] if arch is None else arch) in GATE_FUSED_ARCHES


def _s1(arch: str, path: str, h: int) -> bool:
    """Whether the backward of ``path`` runs the S1 pair on ``arch`` at ``h`` heads, spelled independently of the
    registry: a pure function of the three (the route, the dAqk-hosting form and the hints read the heads alone)."""
    if path not in (*SAVED_PATHS, "replay"):
        return False
    seg = h <= SEG_MAX_H or h in SEG_PLAN_HEADS_BY_ARCH.get(arch, ())
    if (
        not seg and h < DAVQ_MIN_H
    ):  # the serial rows of 9..15 heads: the det pair everywhere
        return False
    if path == "replay":
        bound = S1_REPLAY_MAX_H[arch]
        return bound is None or h <= bound
    return h >= S1_SAVED_MIN_H[arch]


def _wu_saved(n: int, h: int, sm: int, arch: str | None) -> bool:
    """Whether the light set of the row carries ``w`` / ``u``: the serial route at ``WU_SAVED_H`` heads in the
    32-column form on ``WU_SAVED_ARCHES``, the segmented route at ``WU_SAVED_H`` heads above ``WU_SEG_MIN_CHAINS``
    (document, head) chains on ``WU_SEG_SAVED_ARCHES`` (``arch=None``: the first exported architecture)."""
    arch = ARCHES[0] if arch is None else arch
    route = _expected_segments(n, h, sm, arch)[0]
    if h != WU_SAVED_H:
        return False
    if route == "serial":
        return (
            n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
            and arch in WU_SAVED_ARCHES
        )
    return n * h > WU_SEG_MIN_CHAINS and arch in WU_SEG_SAVED_ARCHES


def _carry_ctas(p: int, g: int) -> int:
    """CTAs per (document, head) chain of the two-level carry over ``p`` segments in groups of ``g``: two for a
    single group, else four per group less the four of the chain's ends."""
    groups = -(-p // g)
    return 2 if groups == 1 else 4 * groups - 4


def _carry_hops(p: int, g: int) -> int:
    """The critical path of the two-level carry in hop units: the chain's ``p - 1`` for a single group, else the
    longer of the group products' path and the last (shorter) group's."""
    groups = -(-p // g)
    if groups == 1:
        return p - 1
    last = p - (groups - 1) * g
    return max(2 * g + groups - 4, g + groups - 3 + last)


def _carry_choice(
    n: int, h: int, sm: int, nseg: int, s_hops: int = CARRY_S_HOPS
) -> tuple[int, int] | None:
    """The two-level carry's host rule, spelled independently of the registry: ``(g, grid)`` or ``None``.  Over
    the group sizes ``2..P`` the shortest critical path whose grid fits one CTA per SM (ties: the smaller grid);
    the grid is ``n * h * ctas`` CTAs, or on a proportional block table (at most ``FEW_CHAINS`` chains) the bound
    ``h * (4 * (nseg // g) + 2 n)``; taken only when it gains more than ``s_hops`` over the chain's ``P - 1``."""
    if n < 1 or nseg <= 0 or nseg % n:
        return None
    p = nseg // n
    if p < 2:
        return None
    proportional = n * h <= FEW_CHAINS
    best: tuple[int, int, int] | None = None
    for g in range(2, p + 1):
        if proportional:
            grid = h * (4 * (nseg // g) + 2 * n)
        else:
            grid = n * h * _carry_ctas(p, g)
        if grid > sm:
            continue
        hops = _carry_hops(p, g)
        if best is None or (hops, grid) < best[:2]:
            best = (hops, grid, g)
    if best is None or best[0] + s_hops >= p - 1:
        return None
    return best[2], best[1]


def _route_rule(route: str) -> dict:
    return {"kind": "route_in", "routes": [route]}


# the in-kernel Q/K L2 norm leaf of the copied deterministic rules (#7817): the unfused modes keep the record
L2N_UNFUSED = {"kind": "l2n_in", "modes": ["off", "fallback"]}


def test_registry_records_are_well_formed() -> None:
    # the deterministic package's records (their own copies, the in-kernel Q/K L2 norm's among them) less the
    # deterministic forms no g-mode cell of this package launches, plus the S1 records, the hand-off twins, the tape
    # WY twin, the fused gradient's raw class and its l2n twins and the l2n parameter finalize; eleven libraries are
    # shared (see the twins below): the two pairs of the deterministic package, the four raw-class fused gradient
    # records with their g-class records, the four raw-class l2n twins with their g-class records and the l2n
    # parameter finalize with param_finalize
    assert cake_jit.HANDOFF_TWINS == HANDOFF_TWINS
    assert cake_jit.TAPE_TWINS == TAPE_TWINS
    assert cake_jit.GATE_STAGES == GATE_STAGES and tuple(cake_jit.GATES) == GATES
    assert tuple(cake_jit.L2N_MODES) == L2N_MODES
    assert frozenset(cake_jit.L2N_STAGES) == L2N_STAGES
    assert set(cake_jit.MODULES) == (
        set(det_jit.MODULES) - DET_RECORDS_NOT_LAUNCHED
    ) | set(S1_RECORDS) | TWIN_RECORDS | TAPE_WY_RECORDS | set(RAW_DQKG) | set(
        RAW_L2N_DQKG
    ) | set(L2N_PARAM)
    assert len(det_jit.MODULES) == 107
    assert len(cake_jit.MODULES) == MODULE_COUNT == 126
    n_shared = 2 + len(RAW_DQKG_SHARED) + len(RAW_L2N_DQKG) + len(L2N_PARAM)
    libraries = {r["library"] for r in cake_jit.MODULES.values()}
    assert len(libraries) == MODULE_COUNT - n_shared == 115
    assert set(cake_jit.S1_RECORDS) == S1_RECORDS and len(cake_jit.S1_RECORDS) == 7
    assert tuple(cake_jit.PATHS) == PATHS
    assert tuple(cake_jit.OPTIONAL_STAGES) == OPTIONAL_STAGES
    assert {
        "max_heads": SEG_MAX_H,
        "slot_factor": SEG_SLOT_FACTOR,
        "max_segments": SEG_P_MAX,
        "small_segments": SEG_P_SMALL,
        "small_segments_by_arch": SEG_P_SMALL_BY_ARCH,
        "small_max_chains": SEG_P_SMALL_MAX_NH,
        "prop_max_chains": FEW_CHAINS,
        "plan_heads_by_arch": {
            arch: list(heads) for arch, heads in SEG_PLAN_HEADS_BY_ARCH.items()
        },
    } == cake_jit.SEGMENTS
    assert cake_jit.CARRY_FLAG_WORDS_PER_CTA == CARRY_FLAG_WORDS_PER_CTA
    # the two-level carry's operands and group size extend the logical vocabulary next to the chain carry's
    tensor_names, scalar_names = (
        tuple(cake_jit.LOGICAL_TENSORS),
        tuple(cake_jit.LOGICAL_SCALARS),
    )
    assert tensor_names.index("carry_out_tiles") == tensor_names.index("carry_out") + 1
    assert scalar_names.index("gsize") == scalar_names.index("epoch") + 1
    stages = set()
    tensors, scalars = set(tensor_names), set(scalar_names)
    for name, record in cake_jit.MODULES.items():
        assert record["stage"] in cake_jit.STAGES, name
        stages.add(record["stage"])
        expected_name = (
            record["stage"]
            if record["variant"] is None
            else f"{record['stage']}_{record['variant']}"
        )
        # the one name outside stage[_variant]: the Q/K L2 norm fallback's apply program (#7817), stage qk_apply
        if name == "qk_l2norm_apply":
            expected_name = name if record["stage"] == "qk_apply" else expected_name
        assert name == expected_name, name
        # every record ships on all three architectures but the few of RECORD_ARCHES, each on an ordered subset
        assert list(record["arches"]) == list(RECORD_ARCHES.get(name, ARCHES)), name
        assert [a for a in ARCHES if a in record["arches"]] == list(record["arches"])
        assert record["ffi_entry"] == "run", name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        assert re.fullmatch(r"cake_kda_train_[0-9a-f]{20}", record["library"]), name
        launch = record["launch"]
        assert set(launch) >= {
            "block",
            "cluster",
            "cooperative",
            "dynamic_smem_bytes",
            "use_pdl",
        }, name
        assert len(tuple(launch["block"])) == 3 and tuple(launch["block"])[1:] == (
            1,
            1,
        ), name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["cooperative"] is False, name
        # the overlap pair's pass 2 launches with the programmatic-stream-serialization attribute, nothing else does
        assert launch["use_pdl"] is (name in PDL_RECORDS), name
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        ), name
        plan = record["arg_plan"]
        assert [list(pair) for pair in plan[-3:]] == [
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ], name
        for kind, key in plan[:-3]:
            assert kind in ("buffer", "parameter"), (name, kind, key)
            assert key in record["logical"], (name, key)
            logical = record["logical"][key]
            assert logical in (tensors if kind == "buffer" else scalars), (
                name,
                key,
                logical,
            )
        for kind in _rule_kinds(record["select_rule"]):
            assert kind in SELECT_KINDS, (name, kind)
        assert record["grid_rule"]["kind"] in GRID_KINDS, name
        # the records sharing one library: the segmented forward recurrence's two passes, and the WY program on
        # the saved Akk rows (the backward's, launched by the saving forward on its overlap rows too)
        twins = [
            other
            for other, rec in cake_jit.MODULES.items()
            if rec["library"] == record["library"] and other != name
        ]
        assert twins == [] or {name, *twins} in (
            {"fwdh_v3", "fwdh_p3"},
            {"wy_bwd", "wy_fwd_rows"},
            *({raw, RAW_DQKG[raw]} for raw in RAW_DQKG_SHARED),
            *({raw, det} for raw, det in RAW_L2N_DQKG.items()),
            *({l2n, det} for l2n, det in L2N_PARAM.items()),
        ), name
        # only a gate stage's records name a class
        assert record.get("gate", "g") in (
            GATES if record["stage"] in GATE_STAGES else ("g",)
        ), name
        assert record["stage"] in GATE_STAGES or "gate" not in record, name
        # only an l2n stage's records name the modes they serve, a nonempty ordered subset of L2N_MODES
        if "l2n" in record:
            assert record["stage"] in L2N_STAGES, name
            modes = list(record["l2n"])
            assert modes and len(set(modes)) == len(modes), name
            assert set(modes) <= set(L2N_MODES), name
    assert stages == set(cake_jit.STAGES)
    for stage in cake_jit.STAGES:
        for gate in GATES if stage in GATE_STAGES else ("g",):
            names = cake_jit.records_of_stage(stage, gate)
            if not names:
                # only an optional stage leaves a gate class empty (the gate epilogue's raw class, the finalize's g)
                assert stage in OPTIONAL_STAGES and stage in GATE_STAGES, (stage, gate)
            # the single-record rules hold per in-kernel Q/K L2 norm mode on the l2n stages (a record serves the
            # modes its ``l2n`` lists, ``["off"]`` when absent), per class on every other stage (whose records serve
            # every mode)
            for mode in L2N_MODES if stage in L2N_STAGES else ("off",):
                mnames = [
                    n
                    for n in names
                    if stage not in L2N_STAGES
                    or mode in cake_jit.MODULES[n].get("l2n", ["off"])
                ]
                kinds = [cake_jit.MODULES[n]["select_rule"]["kind"] for n in mnames]
                where = (stage, gate, mode)
                if not mnames:  # the fallback route's stages serve no other mode
                    assert stage in OPTIONAL_STAGES, where
                elif len(mnames) == 1 and stage in OPTIONAL_STAGES:
                    assert kinds[0] not in ("always", "otherwise"), where
                elif len(mnames) == 1:
                    assert kinds == ["always"], where
                else:
                    assert "always" not in kinds and kinds.count("otherwise") <= 1, (
                        where
                    )
            # every non-optional l2n stage has a record of every mode in every gate class it has
            if stage in L2N_STAGES and stage not in OPTIONAL_STAGES and names:
                for mode in L2N_MODES:
                    assert any(
                        mode in cake_jit.MODULES[n].get("l2n", ["off"]) for n in names
                    ), (stage, gate, mode)
    for name in cake_jit.MODULES:
        assert cake_jit.launches(name) == (2 if name in TWO_PASS_RECORDS else 1), name
    # the segmented records read the route (segment_plan's: the head bound, or a plan head of the architecture);
    # the carry's records split the forward's overlap rows (the pair: the two device families, at least the wave
    # threshold, at most the flag words) from their complement (every backward path, the third family, below the
    # threshold, above the bound), which the two-level carry and the chain carry share by their own rule
    seg_rule, serial_rule = _route_rule("seg"), _route_rule("serial")
    # the table stage: on the segmented rows up to the document bound the intra program writes the tables on the
    # forward and the replay (no table record) and the fused table program rebuilds them for a saved-set path; the
    # chunk-parallel table on the serial rows up to the bound, the one-CTA table above; the block table is a launch
    # of its own above only
    docs_rule = {"kind": "docs_at_most", "max": TABLE_V2_MAX_DOCS}
    above_rule = {"kind": "docs_above", "min": TABLE_V2_MAX_DOCS}
    saved_paths = {"kind": "path_in", "paths": ["light", "light_states", "tape"]}
    assert cake_jit.MODULES["table_v3"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, docs_rule, saved_paths],
    }
    assert cake_jit.MODULES["table_v2"]["select_rule"] == {
        "kind": "all_of",
        "rules": [serial_rule, docs_rule],
    }
    assert cake_jit.MODULES["table"]["select_rule"] == above_rule
    assert cake_jit.MODULES["fwd_intra_v14"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, docs_rule],
    }
    # the deterministic package's fused Q/K L2 norm programs take its forward rows (#7817, copied here): the leaf
    # keeps the plain record in the unfused modes and on the replay
    assert cake_jit.MODULES["fwd_intra"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            {"kind": "any_of", "rules": [serial_rule, above_rule]},
            {
                "kind": "any_of",
                "rules": [L2N_UNFUSED, {"kind": "path_in", "paths": ["replay"]}],
            },
        ],
    }
    assert cake_jit.MODULES["seg_table"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, {"kind": "docs_above", "min": TABLE_V2_MAX_DOCS}],
    }

    # the pair's wave threshold is a leaf per device family (3.0 on sm_100a / sm_103a, 2.3 on sm_107a)
    def waves(kind: str) -> dict:
        return {
            "kind": "any_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                        {"kind": kind, "waves": OV1_MIN_WAVES_BY_ARCH["sm_100a"]},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_107a"]},
                        {"kind": kind, "waves": OV1_MIN_WAVES_BY_ARCH["sm_107a"]},
                    ],
                },
            ],
        }

    assert OV1_MIN_WAVES_BY_ARCH["sm_103a"] == OV1_MIN_WAVES_BY_ARCH["sm_100a"]
    assert cake_jit.MODULES["seg_carry_ov"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            {"kind": "path_in", "paths": ["fwd", "fwd_nosave"]},
            waves("walk_ctas_at_least"),
            {"kind": "segments_heads_at_most", "max": OV1_MAX_SEGMENT_HEADS},
        ],
    }
    # the backward paths and the forward's non-overlap rows: the two-level carry on every architecture at most
    # CARRY_V8_MAX_H heads where its host rule accepts, the chain carry everywhere else (the two rules partition)
    not_overlap = {
        "kind": "any_of",
        "rules": [
            {
                "kind": "path_in",
                "paths": ["light", "light_states", "tape", "replay"],
            },
            waves("walk_ctas_below"),
            {"kind": "segments_heads_above", "min": OV1_MAX_SEGMENT_HEADS},
        ],
    }
    assert cake_jit.MODULES["seg_carry"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            not_overlap,
            {
                "kind": "any_of",
                "rules": [
                    {"kind": "heads_above", "min": CARRY_V8_MAX_H},
                    {"kind": "carry_chain", "s_hops": CARRY_S_HOPS},
                ],
            },
        ],
    }
    assert cake_jit.MODULES["seg_carry_v8"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            not_overlap,
            {
                "kind": "all_of",
                "rules": [
                    {"kind": "heads_at_most", "max": CARRY_V8_MAX_H},
                    {"kind": "carry_two_level", "s_hops": CARRY_S_HOPS},
                ],
            },
        ],
    }
    assert cake_jit.MODULES["seg_carry_v8"]["grid_rule"] == {
        "kind": "carry_groups",
        "s_hops": CARRY_S_HOPS,
    }
    # the backward prep: the tape set's form (it writes qg), the replay's, and the light sets' no-qg forms (the
    # version-4 set carries the forward's qg): the no-kb one, on both light paths, on the rows whose set carries w / u
    # (the paired serial rows of WU_SAVED_ARCHES and the segmented rows of WU_SEG_SAVED_ARCHES above WU_SEG_MIN_CHAINS
    # chains), the other elsewhere
    light = {"kind": "path_in", "paths": ["light"]}
    light_states = {"kind": "path_in", "paths": ["light_states"]}
    slots_fit = {
        "kind": "slots_at_most",
        "slices": RECUR_SLICES["v2"],
        "ctas_per_sm": RECUR_CTAS_PER_SM,
    }
    slots_over = {
        "kind": "slots_above",
        "slices": RECUR_SLICES["v2"],
        "ctas_per_sm": RECUR_CTAS_PER_SM,
    }
    wu_rows = {
        "kind": "all_of",
        "rules": [
            serial_rule,
            {
                "kind": "all_of",
                "rules": [
                    {"kind": "heads_at_least", "min": WU_SAVED_H},
                    {"kind": "heads_at_most", "max": WU_SAVED_H},
                ],
            },
            slots_fit,
            {"kind": "arch_in", "arches": list(WU_SAVED_ARCHES)},
        ],
    }
    not_wu_rows = {
        "kind": "any_of",
        "rules": [
            seg_rule,
            {"kind": "heads_below", "max": WU_SAVED_H},
            {"kind": "heads_above", "min": WU_SAVED_H},
            slots_over,
            {
                "kind": "arch_in",
                "arches": [a for a in ARCHES if a not in WU_SAVED_ARCHES],
            },
        ],
    }
    seg_wu_rows = {
        "kind": "all_of",
        "rules": [
            seg_rule,
            {
                "kind": "all_of",
                "rules": [
                    {"kind": "heads_at_least", "min": WU_SAVED_H},
                    {"kind": "heads_at_most", "max": WU_SAVED_H},
                ],
            },
            {"kind": "chains_above", "min": WU_SEG_MIN_CHAINS},
            {"kind": "arch_in", "arches": list(WU_SEG_SAVED_ARCHES)},
        ],
    }
    not_seg_wu_rows = {
        "kind": "any_of",
        "rules": [
            serial_rule,
            {"kind": "heads_below", "max": WU_SAVED_H},
            {"kind": "heads_above", "min": WU_SAVED_H},
            {"kind": "chains_at_most", "max": WU_SEG_MIN_CHAINS},
            {
                "kind": "arch_in",
                "arches": [a for a in ARCHES if a not in WU_SEG_SAVED_ARCHES],
            },
        ],
    }
    light_paths = {"kind": "path_in", "paths": ["light", "light_states"]}
    # the serial rows' form is gated to the light path (the serial route never saves the states); its complement
    # in the kb-writing form admits the states-saving set
    wu_rows_light = {"kind": "all_of", "rules": [light, *wu_rows["rules"]]}
    not_wu_rows_or_states = {
        "kind": "any_of",
        "rules": [light_states, *not_wu_rows["rules"]],
    }
    assert cake_jit.MODULES["bwd_prep_g8"]["select_rule"] == {
        "kind": "path_in",
        "paths": ["tape"],
    }
    assert cake_jit.MODULES["bwd_prep_g8_replay"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            {"kind": "path_in", "paths": ["replay"]},
            {
                "kind": "any_of",
                "rules": [
                    L2N_UNFUSED,
                    {"kind": "all_of", "rules": [seg_rule, docs_rule]},
                ],
            },
        ],
    }
    assert cake_jit.MODULES["bwd_prep_g8_noqg"]["select_rule"] == {
        "kind": "all_of",
        "rules": [light_paths, not_wu_rows_or_states, not_seg_wu_rows],
    }
    assert cake_jit.MODULES["bwd_prep_g8_noqg_nokb"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            light_paths,
            {"kind": "any_of", "rules": [wu_rows_light, seg_wu_rows]},
        ],
    }
    # the backward's WY program: every tape set; the light sets but the fused 64-column recompute's rows and the
    # rows whose set carries w / u; the saved-state sets above 32 chains (the fused pass 2 forms it below) but,
    # again, the segmented rows whose set carries w / u
    assert cake_jit.MODULES["wy_bwd"]["select_rule"] == {
        "kind": "any_of",
        "rules": [
            {"kind": "path_in", "paths": ["tape"]},
            {
                "kind": "all_of",
                "rules": [
                    light,
                    {
                        "kind": "any_of",
                        "rules": [
                            {"kind": "heads_below", "max": FUSED_BWD_MIN_H},
                            slots_fit,
                        ],
                    },
                    not_wu_rows,
                    not_seg_wu_rows,
                ],
            },
            {
                "kind": "all_of",
                "rules": [
                    light_states,
                    {"kind": "chains_above", "min": FUSED_P3_MAX_CHAINS},
                    not_seg_wu_rows,
                ],
            },
        ],
    }
    # the gate epilogue launches on the unfused architecture (sm_100a) after either of its GLM intra programs
    # (glm_intra, glm_intra_v21); the fused ones (glm_intra_v14 / glm_intra_v22 on sm_103a, glm_intra_v18 /
    # glm_intra_v23 on sm_107a) write the gate gradients themselves
    unfused = [a for a in ARCHES if a not in GATE_FUSED_ARCHES]
    assert cake_jit.MODULES["gate_epilogue"]["select_rule"] == {
        "kind": "arch_in",
        "arches": unfused,
    }
    glm_records = set(cake_jit.records_of_stage("glm_intra"))
    # and the deterministic package's GLM intra programs of the in-kernel Q/K L2 norm (fused, its sm_103a form,
    # fallback capture)
    assert glm_records == set(DET_GLM.values()) | set(S1_GLM.values()) | {
        "glm_intra_l2n",
        "glm_intra_v30_l2n",
        "glm_intra_l2n_capture",
    }
    for name in (*sorted(glm_records), "gate_epilogue"):
        assert cake_jit.MODULES[name]["grid_rule"] == {"kind": "chunks", "x_factor": 1}
    assert set(cake_jit.records_of_stage("dqkg")) >= set(S1_DQKG.values()) | {
        "dqkg",
        "dqkg_int",
    }
    # the records of the two pair stages select on the path, the architecture, the route and the head count alone
    # (the route reads the heads and the architecture), so the pair of a backward is a pure function of
    # (architecture, path, heads), which test_cake_kda_nondet_train_api.py::test_the_pairs_agree_on_every_head_count sweeps
    pair_kinds = {
        "always",
        "otherwise",
        "all_of",
        "any_of",
        "path_in",
        "arch_in",
        "route_in",
        "heads_at_most",
        "heads_above",
        "heads_at_least",
        "heads_below",
    }
    for stage in ("dqkg", "glm_intra"):
        for name in cake_jit.records_of_stage(stage):
            kinds = set(_rule_kinds(cake_jit.MODULES[name]["select_rule"]))
            assert kinds <= pair_kinds, (name, kinds - pair_kinds)
    # The S1 records against the deterministic package's records they derive from: the GLM intra programs bind the
    # same operands (the S1 drain reads the q and dq_f the deterministic one loads already) at the same launch and
    # grid; the fused gradient binds the same operands but q (no q box, no q load), at the same launch and grid.
    for arch, name in S1_GLM.items():
        record, twin = cake_jit.MODULES[name], det_jit.MODULES[DET_GLM[arch]]
        assert record["stage"] == twin["stage"] == "glm_intra", name
        for key in ("logical", "arg_plan", "launch", "grid_rule"):
            assert record[key] == twin[key], (name, key)
    for det_name, name in S1_DQKG.items():
        record, twin = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        assert record["stage"] == twin["stage"] == "dqkg", name
        assert record["grid_rule"] == twin["grid_rule"] == {"kind": "chunk_items"}
        assert record["launch"] == twin["launch"], name
        assert record["logical"] == {
            key: value for key, value in twin["logical"].items() if key != "q_tma"
        }, name
        assert "q" not in record["logical"].values(), name
        assert [list(pair) for pair in record["arg_plan"]] == [
            list(pair) for pair in twin["arg_plan"] if pair[1] != "q_tma"
        ], name
    # the S1 programs, the hand-off twins and the tape WY twin are new code: libraries of their own, none shared with
    # a deterministic program's digest
    det_digests = {r["library"].rsplit("_", 1)[1] for r in det_jit.MODULES.values()}
    for name in (*S1_RECORDS, *TWIN_RECORDS, *TAPE_WY_RECORDS):
        assert cake_jit.MODULES[name]["library"].rsplit("_", 1)[1] not in det_digests
    # the saving forward's token-row twins: the same stage, grid and rule leaves as the internal-row forms but the path
    for internal, rows in (
        ("fwdh_wy", "fwdh_wy_rows"),
        ("fwdh_p3_o_ov", "fwdh_p3_o_ov_rows"),
        ("fwdh_p3_o_ov_id", "fwdh_p3_o_ov_id_rows"),
        ("fwdh_o_v2", "fwdh_o_v2_rows"),
    ):
        a, b = cake_jit.MODULES[internal], cake_jit.MODULES[rows]
        assert (a["stage"], a["grid_rule"]) == (b["stage"], b["grid_rule"]), rows
        assert a["library"] != b["library"]
        assert str(a["select_rule"]).replace("'fwd_nosave'", "'fwd'") == str(
            b["select_rule"]
        ), rows
    assert (
        cake_jit.MODULES["wy_fwd_rows"]["library"]
        == cake_jit.MODULES["wy_bwd"]["library"]
    )


def _expected_segments(
    n: int, h: int, sm: int, arch: str | None = None
) -> tuple[str, int, int]:
    plan_heads = SEG_PLAN_HEADS_BY_ARCH.get(arch, ()) if arch is not None else ()
    if h <= SEG_MAX_H or h in plan_heads:
        p = min(SEG_P_MAX, max(1, -(-SEG_SLOT_FACTOR * sm // (n * h))))
        p = max(1, min(p, sm // h))  # one walk CTA per SM per document at most
        if n * h <= SEG_P_SMALL_MAX_NH:
            p = min(p, SEG_P_SMALL_BY_ARCH.get(arch, SEG_P_SMALL))
        return "seg", p, n * p
    return "serial", 1, 0


def _query_arch() -> str | None:
    """The architecture the byte queries take without an explicit one: the current CUDA device's when it is an
    exported target, else None (the architecture-independent few-chain segment count)."""
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    return cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))


def _expected_records(
    nc_pad: int, n: int, h: int, sm: int, arch: str, path: str
) -> dict[str, str | None]:
    """The record of every stage of ``path`` on (``nc_pad``, ``n``, ``h``, ``sm``, ``arch``): the dispatch of the
    kernels' composition, spelled independently of the registry's rules."""
    route, _p, nseg = _expected_segments(n, h, sm, arch)
    seg = route == "seg"
    v2 = n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
    few = n * h <= FEW_CHAINS
    fused_p3 = seg and n * h <= FUSED_P3_MAX_CHAINS
    fused_fwd = (
        not seg and h >= FUSED_FWD_MIN_H and (arch != "sm_107a" or h <= FUSED_FWD_MIN_H)
    )
    fused_bwd = not seg and not v2 and h >= FUSED_BWD_MIN_H
    # the rows whose light set carries w / u: no wy launch on their light paths, a prep that writes no kb
    wu = _wu_saved(n, h, sm, arch)
    # the fused gradient hosts dAqk on the serial rows from 16 heads and on every segmented row (whose adjoint
    # recurrence accumulates dv): no dav launch there
    davq = (h >= DAVQ_MIN_H) if not seg else True
    replay = path == "replay"
    fwd = path in ("fwd", "fwd_nosave")
    # the saving forward reads the saved token rows where a program for them ships
    save = path == "fwd"
    # the paired serial rows: the 32-column form below the fused threshold reads the saved token rows on the saving
    # forward (wy_fwd_rows + fwdh_o_v2_rows); the 64-column form does so on sm_107a alone (wy_fwd_rows +
    # fwdh_o_v1_rows) and keeps the internal-row pair on both paths elsewhere
    paired = save and not seg and v2 and not fused_fwd
    paired_v1 = (
        save and not seg and not v2 and not fused_fwd and arch in PAIRED_V1_ARCHES
    )
    walk = nseg * h
    prefetch = walk < DHU_PREFETCH_WAVES_BY_ARCH.get(arch, 0.0) * sm
    tma = walk < DHU_TMA_WAVES * sm or (h <= 4 and walk <= DHU_TMA_WAVES * sm)
    ov1 = (
        seg
        and fwd
        and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
        and walk <= OV1_MAX_SEGMENT_HEADS
    )
    # the two-level carry on every architecture at most two heads where its host rule accepts (the forward's
    # overlap rows keep the pair's carry); the chain carry everywhere else
    two_level = (
        seg
        and arch in CARRY_V8_ARCHES
        and h <= CARRY_V8_MAX_H
        and _carry_choice(n, h, sm, nseg) is not None
    )
    gate_fused = _gate_fused(arch)
    if not seg:
        carry = None
    elif ov1:
        carry = "seg_carry_ov"
    else:
        carry = "seg_carry_v8" if two_level else "seg_carry"
    # the chunk-parallel tables up to the document bound (the segmented route's fused form writes the block table
    # too), the one-CTA table above, where the segmented route launches the block table on its own; on the segmented
    # route up to the bound the intra program writes the tables itself on the forward and the replay (no table
    # launch; the fused table program serves the saved-set paths whose set carries none)
    small = n <= TABLE_V2_MAX_DOCS
    intra_tables = seg and small
    if intra_tables:
        table = "table_v3" if path in ("light", "light_states", "tape") else None
    else:
        table = "table_v2" if small else "table"
    out: dict[str, str | None] = {
        "table": table,
        "seg_table": "seg_table" if seg and not small else None,
        "fwd_intra": "fwd_intra_v14" if intra_tables else "fwd_intra",
        "wy_fwd": None
        if fused_fwd
        else ("wy_fwd_rows" if (ov1 and save) or paired or paired_v1 else "wy_fwd"),
        "seg_carry": carry,
        "glm_intra": GATE_FUSED_RECORD[arch] if gate_fused else "glm_intra",
        "gate_epilogue": None if gate_fused else "gate_epilogue",
        "param_finalize": None,
    }
    if path == "tape":
        out["wy"] = "wy_bwd"
    elif path == "light":
        out["wy"] = None if (fused_bwd or wu) else "wy_bwd"
    elif path == "light_states":
        out["wy"] = None if (fused_p3 or wu) else "wy_bwd"
    elif replay:
        out["wy"] = None if fused_bwd else "wy_bwd_replay"
    else:
        out["wy"] = None
    if fwd:
        if seg:
            out["fwdh"] = "fwdh_v3"
        elif fused_fwd:
            out["fwdh"] = "fwdh_wy_rows" if save else "fwdh_wy"
        elif v2:
            out["fwdh"] = "fwdh_o_v2_rows" if paired else "fwdh_o_v2"
        else:
            out["fwdh"] = "fwdh_o_v1_rows" if paired_v1 else "fwdh_o_v1"
    elif path in ("light", "replay"):
        if seg:
            out["fwdh"] = "fwdh_v3"
        elif fused_bwd:
            out["fwdh"] = "fwdh_wy_bwd_replay" if replay else "fwdh_wy_bwd"
        elif v2:
            # the vh 32 build of the 32-column recompute from 16 heads; on sm_107a at 16 heads only
            narrow_wide = h >= 16 and (arch != "sm_107a" or h <= 16)
            out["fwdh"] = "fwdh_v2h" if narrow_wide else "fwdh_v2"
        else:
            out["fwdh"] = "fwdh_v1_sm107" if arch == "sm_107a" else "fwdh_v1"
    else:
        out["fwdh"] = None
    if not seg or path == "tape":
        out["fwdh_p3"] = None
    elif fwd:
        if ov1:
            out["fwdh_p3"] = (
                "fwdh_p3_o_ov" if h <= OV1_ORDER_MAX_H else "fwdh_p3_o_ov_id"
            ) + ("_rows" if save else "")
        else:
            out["fwdh_p3"] = "fwdh_p3_o"
    elif path == "light_states":
        out["fwdh_p3"] = "fwdh_p3_wy" if fused_p3 else "fwdh_p3"
    else:
        out["fwdh_p3"] = "fwdh_p3"
    # the backward prep: the tape set's form writes qg, the replay's the internal dout copy too; the light sets
    # carry the forward's qg (the no-qg forms) and, on the w / u rows, its WY representation (no kb either)
    if path == "tape":
        out["bwd_prep"] = "bwd_prep_g8"
    elif replay:
        out["bwd_prep"] = "bwd_prep_g8_replay"
    elif path in ("light", "light_states") and wu:
        out["bwd_prep"] = "bwd_prep_g8_noqg_nokb"
    elif path in ("light", "light_states"):
        out["bwd_prep"] = "bwd_prep_g8_noqg"
    else:
        out["bwd_prep"] = None
    if davq:
        out["dav"] = None
    else:  # the serial route below 16 heads and the few-chain segmented rows: dAqk alone
        out["dav"] = "dav_serial_replay" if replay else "dav_serial"
    if seg and few:
        if replay:
            out["dhu"] = "dhu_v3s_f_replay"
        else:
            out["dhu"] = "dhu_v3s_f_wt" if path == "light_states" else "dhu_v3s_f"
    elif seg:
        # the dv-accumulating adjoint in the row's tape build (the prefetching form below the sm_103a bound); its
        # w^T form behind the fused saved-state pass 2
        base = "dhu_v3p_f" if prefetch else ("dhu_v3_f" if tma else "dhu_v3r_f")
        if replay:
            out["dhu"] = base + "_replay"
        else:
            out["dhu"] = base + ("_wt" if path == "light_states" and fused_p3 else "")
    elif v2:
        out["dhu"] = "dhu_v2t_replay" if replay else "dhu_v2t"
    elif fused_bwd and path in ("light", "replay"):
        out["dhu"] = "dhu_v1_wt_replay" if replay else "dhu_v1_wt"
    else:
        out["dhu"] = "dhu_v1_replay" if replay else "dhu_v1"
    nohint = davq and h >= DQKG_NOHINT_MIN_H_BY_ARCH.get(arch, h + 1)
    out["dqkg"] = (
        ("dqkg_int" if replay else "dqkg")
        + ("_davq" if davq else "")
        + ("_nohint" if nohint else "")
    )
    # the S1 cells: the S1 twin of the dAqk-hosting fused gradient and the S1 GLM intra program, always together
    if _s1(arch, path, h):
        assert davq, (arch, path, h)
        out["dqkg"] = S1_DQKG[out["dqkg"]]
        out["glm_intra"] = S1_GLM[arch]
    return {stage: out[stage] for stage in PATH_STAGES[path]}


def _expected_grid(
    name: str, nc_pad: int, n: int, h: int, sm: int, nseg: int = 0, launch: int = 0
) -> tuple[int, int, int]:
    ctas = nc_pad * h
    if name in ("table", "seg_table"):
        return (1, 1, 1)
    if name in ("table_v2", "table_v3"):
        return (max(1, -(-nc_pad // 256)), 1, 1)
    if name in (
        "fwd_intra",
        "fwd_intra_v14",
        "bwd_prep_g8",
        "bwd_prep_g8_replay",
        "bwd_prep_g8_noqg",
        "bwd_prep_g8_noqg_nokb",
        "glm_intra",
        "glm_intra_v14",
        "glm_intra_v18",
        "glm_intra_v21",
        "glm_intra_v22",
        "glm_intra_v23",
        "gate_epilogue",
    ):
        return (nc_pad, h, 1)
    if name.startswith(("wy_", "dav")):  # one CTA per (pair, head) item
        return (max(1, nc_pad // 2 * h), 1, 1)
    if name == "fwdh_v3":
        return (2 * nseg * h, 1, 1)
    if name.startswith("fwdh_p3"):
        return (nseg * h, 1, 1)
    if name.startswith("dhu_v3s_f"):
        return (2 * nseg * h, 1, 1)
    if name.startswith(("dhu_v3_f", "dhu_v3p_f", "dhu_v3r_f")):
        return (nseg * h, 1, 1)
    if name == "seg_carry_v8":  # the two-level carry's grid: its host rule's CTA count
        choice = _carry_choice(n, h, sm, nseg)
        assert choice is not None, (name, n, h, sm, nseg)
        return (choice[1], 1, 1)
    if name.startswith("seg_carry"):  # two CTAs per (document, head) chain
        return (2 * n * h, 1, 1)
    if name.startswith("fwdh_") or name.startswith("dhu_"):
        narrow = name.startswith(("fwdh_o_v2", "fwdh_v2", "dhu_v2t"))
        return (n * h * (4 if narrow else 2), 1, 1)
    if name.startswith("dqkg"):  # one CTA per (chunk, head) item
        return (max(1, ctas), 1, 1)
    raise AssertionError(name)


PLAN_GRID = [
    (nc_pad, n, h, sm)
    for nc_pad in (2, 4, 6, 32, 64, 128, 256, 4096)
    for n in (1, 3, 4, 7, 300)
    for h in (1, 2, 4, 8, 16, 32, 64)
    for sm in (132, 148, 160, 212)
]


def test_the_raw_class_is_the_deterministic_raw_class() -> None:
    """Raw mode is the deterministic package's, record for record: every raw-class record of the registry is the
    deterministic record of its name (a fused gradient record: of its ``RAW_DQKG`` record with the in-kernel Q/K L2
    norm off, of its ``RAW_L2N_DQKG`` record in the l2n modes, with the class and its own variant) under the copied
    library, every deterministic raw-class record has its copy (the in-kernel Q/K L2 norm's among them, with the
    deterministic record's ``l2n``), the parameter-gradient
    pair (``glm_intra_raw``, ``param_finalize``: :data:`RAW_PARAM_PAIR`) is the deterministic records with their
    architectures and rules (all three architectures; ``always`` and the backward paths), ``param_finalize_l2n``
    (:data:`L2N_PARAM`) is the deterministic ``param_finalize`` with its architectures and rule, serving the l2n
    modes, and the gate epilogue keeps the deterministic record's g class."""
    renamed = {"library", "kernel_symbol"}
    raw = {name for name, rec in cake_jit.MODULES.items() if rec.get("gate") == "raw"}
    det_raw = {
        name for name, rec in det_jit.MODULES.items() if rec.get("gate") == "raw"
    }
    assert not det_raw & DET_RECORDS_NOT_LAUNCHED
    assert raw == det_raw | set(RAW_DQKG) | set(RAW_L2N_DQKG) | set(L2N_PARAM)
    assert set(RAW_PARAM_PAIR.values()) <= det_raw
    assert not det_raw & set(L2N_PARAM) and set(L2N_PARAM.values()) <= det_raw
    assert set(RAW_DQKG.values()) <= {
        name for name, rec in det_jit.MODULES.items() if rec["stage"] == "dqkg"
    }
    # the deterministic raw class's in-kernel Q/K L2 norm records are copied too (#7817)
    assert {
        name
        for name in det_raw
        if "off" not in det_jit.MODULES[name].get("l2n", ["off"])
    } <= raw
    checked = 0
    for name in sorted(raw | {"gate_epilogue"}):
        record, twin = (
            cake_jit.MODULES[name],
            det_jit.MODULES[DET_SOURCE.get(name, name)],
        )
        assert record["library"] == twin["library"].replace(
            "cake_kda_det_train_", "cake_kda_train_"
        ), name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        expected = {key: value for key, value in twin.items() if key not in renamed}
        if name in RAW_DQKG or name in RAW_L2N_DQKG:
            # the fused gradient's raw class: the flag-off records and the l2n twins, each with the deterministic
            # record's modes
            assert "gate" not in twin, name
            twin_modes = ["off"] if name in RAW_DQKG else list(L2N_ON)
            assert twin.get("l2n", ["off"]) == twin_modes, name
            expected.update(variant=name.removeprefix("dqkg_"), gate="raw")
        if name in L2N_PARAM:
            # the deterministic record: its architectures and rule, the l2n modes alone
            assert "l2n" not in twin and twin["variant"] is None, name
            expected.update(variant="l2n", l2n=list(L2N_ON))
        assert {
            key: value for key, value in record.items() if key not in renamed
        } == expected, name
        checked += 1
    assert checked == len(raw) + 1 > len(RAW_PARAM_PAIR) + len(L2N_PARAM)
    # the parameter-gradient pair, spelled out: the deterministic records on every architecture
    for stage, name in RAW_PARAM_PAIR.items():
        record = cake_jit.MODULES[name]
        assert (record["stage"], record["gate"]) == (stage, "raw"), name
        assert list(record["arches"]) == list(ARCHES), name
        assert "l2n" not in record, name
    assert cake_jit.MODULES["glm_intra_raw"]["select_rule"] == {"kind": "always"}
    assert cake_jit.MODULES["param_finalize"]["select_rule"] == RAW_BWD_RULE
    # no S1 record is of the raw class
    assert not raw & S1_RECORDS
    assert cake_jit.MODULES["gate_epilogue"]["gate"] == "g"


def test_the_l2n_class_is_the_deterministic_l2n_class() -> None:
    """The in-kernel Q/K L2 norm is the deterministic package's, record for record: every deterministic record
    that serves an l2n mode (its fused class, the sm_103a forward intra form, the GLM intra capture forms, the
    fallback route's programs and the plain records the l2n route keeps) is copied with its ``l2n`` under the copied
    library (the output backward ``dav_l2n*`` and the fused gradient ``dqkg_*l2n`` among them); the raw class's l2n
    twins of the fused gradient (``RAW_L2N_DQKG``) are those records' rules, plans, programs and modes in the raw
    class, each sharing its program with the g-class copy.  The S1 records, the hand-off twins and the tape WY
    twin carry no ``l2n`` key, and the narrowed g-class copies of the pair stages serve the flag-off mode alone: no
    ND-only record is selected in an l2n mode.  The tape WY twin is launched outside the selection
    (its stage ``wy`` is no l2n stage): a tape backward of an l2n mode may run it, writing its WY record's ``w`` bit
    for bit."""
    renamed = {"library", "kernel_symbol"}

    def serves(record: dict, mode: str) -> bool:
        return mode in record.get("l2n", ["off"])

    det_l2n = {
        name
        for name, rec in det_jit.MODULES.items()
        if rec["stage"] in L2N_STAGES
        and (serves(rec, "fused") or serves(rec, "fallback"))
    }
    # the deterministic l2n records: the 7 g + 7 raw fused ones and the raw class's kb-once replay prep, the sm_103a
    # forward intra and GLM intra forms, the replay prep's saved-rstd twin, the 2 capture forms, the 3 fallback
    # programs, the 5 dav_l2n records and the 4 l2n fused gradient records, and the 3 plain records the fused route
    # keeps (fwd_intra, g and raw, and the g class's bwd_prep_g8_replay; the raw class's replay prep serves the
    # flag-off and fallback modes alone)
    det_l2n_only = {n for n in det_l2n if not serves(det_jit.MODULES[n], "off")}
    assert len(det_l2n_only) == 32, sorted(det_l2n_only)
    assert set(DET_L2N_DQKG) <= det_l2n_only
    assert {n for n in det_l2n_only if det_jit.MODULES[n]["stage"] == "dav"} == {
        "dav_l2n",
        "dav_l2n_wt",
        "dav_l2n_wtk",
        "dav_l2n_replay",
        "dav_l2n_replay_wt",
    }
    assert "bwd_prep_g8_replay_kbonce_raw_l2n" in det_l2n_only
    assert {
        n
        for n in det_l2n
        if serves(det_jit.MODULES[n], "off") and serves(det_jit.MODULES[n], "fused")
    } == {"fwd_intra", "fwd_intra_raw", "bwd_prep_g8_replay"}
    assert det_l2n <= set(cake_jit.MODULES)
    compared = 0
    for name in sorted(det_l2n):
        if det_jit.MODULES[name]["stage"] in ("dqkg", "glm_intra") and serves(
            det_jit.MODULES[name], "off"
        ):
            continue  # a pair-stage record serving the flag-off mode too: none in this registry
        record, twin = cake_jit.MODULES[name], det_jit.MODULES[name]
        assert record["library"] == twin["library"].replace(
            "cake_kda_det_train_", "cake_kda_train_"
        ), name
        assert {k: v for k, v in record.items() if k not in renamed} == {
            k: v for k, v in twin.items() if k not in renamed
        }, name
        compared += 1
    assert compared == len(det_l2n) > len(det_l2n_only)
    # the raw class's l2n twins of the fused gradient
    assert len(RAW_L2N_DQKG) == 4
    for name, det_name in RAW_L2N_DQKG.items():
        record, twin = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        g_copy = cake_jit.MODULES[det_name]
        assert record["stage"] == "dqkg" and record["gate"] == "raw", name
        assert record["l2n"] == list(L2N_ON) == twin["l2n"], name
        assert g_copy.get("gate", "g") == "g" and "gate" not in twin, name
        assert (
            record["library"]
            == g_copy["library"]
            == twin["library"].replace("cake_kda_det_train_", "cake_kda_train_")
        ), name
        expected = {k: v for k, v in twin.items() if k not in renamed}
        expected.update(variant=name.removeprefix("dqkg_"), gate="raw")
        assert {k: v for k, v in record.items() if k not in renamed} == expected, name
    # no ND-only record carries an l2n mode
    nd_only = set(S1_RECORDS) | TWIN_RECORDS | TAPE_WY_RECORDS
    for name in sorted(nd_only):
        assert "l2n" not in cake_jit.MODULES[name], name
    for stage in ("dqkg", "glm_intra"):
        for name in cake_jit.records_of_stage(stage):
            record = cake_jit.MODULES[name]
            if serves(record, "off"):
                assert record.get("l2n", ["off"]) == ["off"], name
            else:
                assert name in det_l2n, name
    # the parameter finalize of the l2n modes: the deterministic param_finalize (L2N_PARAM); param_finalize serves
    # the flag-off mode alone
    for name in cake_jit.records_of_stage("param_finalize", "raw"):
        record = cake_jit.MODULES[name]
        if name in L2N_PARAM:
            assert record["l2n"] == list(L2N_ON), name
        else:
            assert "l2n" not in record and serves(record, "off"), name
    for mode in L2N_ON:
        assert [
            name
            for name in cake_jit.records_of_stage("param_finalize", "raw")
            if serves(cake_jit.MODULES[name], mode)
        ] == list(L2N_PARAM), mode


# the memory reports' in-kernel Q/K L2 norm fields (#7817): the mode, then its byte counts
DET_L2N_REPORT_KEYS = (
    "l2n",
    "saved_qk_rstd",
    "bwd_qn_kn",
    "replay_qn_kn",
    "replay_rstd",
)

RAW_PLAN_GRID = [
    (nc_pad, n, h, sm)
    for nc_pad in (2, 6, 128, 4096)
    for n in (1, 3, 300)
    for h in (1, 2, 4, 8, 16, 32, 64, 128)
    for sm in (148, 160, 212)
]


def test_raw_mode_selects_the_deterministic_records() -> None:
    """On every (architecture, path, head count 1..128) cell raw mode selects one record per (stage, gate) class --
    the deterministic package's raw-mode record of the stage (a fused gradient record: its ``RAW_DQKG`` copy) at the
    same grid, never an S1 record -- the parameter-gradient pair included: on every architecture each backward path
    selects ``glm_intra_raw`` and ``param_finalize`` (:data:`RAW_PARAM_PAIR`), the forward paths neither; and g mode
    selects the record it selects without a gate."""
    pair_cells = {arch: 0 for arch in ARCHES}
    for nc_pad, n, h, sm in RAW_PLAN_GRID:
        for arch in ARCHES:
            route, _segments, nseg = _expected_segments(n, h, sm, arch)
            for path in PATHS:
                if path == "light_states" and route == "serial":
                    continue
                for stage in PATH_STAGES[path]:
                    scalars = {
                        "num_chunks_pad": nc_pad,
                        "num_seqs": n,
                        "num_heads": h,
                        "sm_count": sm,
                        "arch": arch,
                        "path": path,
                    }
                    where = (stage, nc_pad, n, h, sm, arch, path)
                    g = cake_jit.select_record(stage, **scalars)
                    assert cake_jit.select_record(stage, **scalars, gate="g") == g, (
                        where
                    )
                    name = cake_jit.select_record(stage, **scalars, gate="raw")
                    det_name = det_jit.select_record(stage, **scalars, gate="raw")
                    assert RAW_DQKG.get(name, name) == det_name, where
                    if stage in RAW_PARAM_PAIR:
                        # the backward paths' pair (the forward paths launch neither stage)
                        assert name == RAW_PARAM_PAIR[stage], where
                        pair_cells[arch] += 1
                    if name is None:
                        assert stage in OPTIONAL_STAGES, where
                        continue
                    assert name not in S1_RECORDS, where
                    gate = cake_jit.MODULES[name].get("gate", "g")
                    assert gate == ("raw" if stage in GATE_STAGES else "g"), where
                    if stage not in GATE_STAGES:
                        assert name == g, where
                    for launch in range(cake_jit.launches(name)):
                        grid = {
                            "num_chunks_pad": nc_pad,
                            "num_seqs": n,
                            "num_heads": h,
                            "sm_count": sm,
                            "num_segments": nseg,
                            "launch": launch,
                        }
                        assert cake_jit.launch_grid(name, **grid) == (
                            det_jit.launch_grid(det_name, **grid)
                        ), where
    assert all(pair_cells.values()), pair_cells


def test_records_of_stage_filters_the_gate_stages_by_class() -> None:
    for stage in cake_jit.STAGES:
        g = cake_jit.records_of_stage(stage)
        raw = cake_jit.records_of_stage(stage, "raw")
        assert cake_jit.records_of_stage(stage, "g") == g
        if stage in GATE_STAGES:
            assert not set(g) & set(raw), stage
            assert all(cake_jit.MODULES[n].get("gate", "g") == "g" for n in g), stage
            assert all(cake_jit.MODULES[n]["gate"] == "raw" for n in raw), stage
        else:
            assert g == raw, stage
    raw_dqkg = set(RAW_DQKG) | set(RAW_L2N_DQKG)
    assert set(cake_jit.records_of_stage("dqkg", "raw")) == raw_dqkg
    # the deterministic l2n fused gradient records are of the g class, their twins of the raw class
    assert set(DET_L2N_DQKG) <= set(cake_jit.records_of_stage("dqkg"))
    # the raw GLM intra program of each in-kernel Q/K L2 norm mode (off, the fallback's capture form, fused)
    assert cake_jit.records_of_stage("glm_intra", "raw") == (
        "glm_intra_raw",
        "glm_intra_raw_l2n_capture",
        "glm_intra_raw_l2n",
    )
    assert cake_jit.records_of_stage("param_finalize") == ()
    # the deterministic finalize (off) and the l2n modes' copy of it
    assert cake_jit.records_of_stage("param_finalize", "raw") == (
        "param_finalize",
        "param_finalize_l2n",
    )
    assert cake_jit.records_of_stage("gate_epilogue", "raw") == ()
    with pytest.raises(ValueError, match="gate 'activated' is not one of"):
        cake_jit.records_of_stage("dqkg", "activated")


def test_raw_plans_are_the_deterministic_raw_plans() -> None:
    """``plan(..., gate="raw")`` is the deterministic package's raw plan of the problem: the same geometry and route,
    on every path the same record per stage (the fused gradient's: the ``RAW_DQKG`` copy of the deterministic
    record, never an S1 record) with the same launches and grids, no gate epilogue and the parameter finalize once
    per backward path (one CTA per head and segment of 128 chunks), on every architecture the deterministic
    parameter-gradient pair (``glm_intra_raw``, ``param_finalize``) on every backward path; a separate cache entry
    from the g-mode plan of the problem, whose records it shares on every stage outside ``GATE_STAGES``; the
    checkpoint hand-off never engages on it, in any mode; the raw mode's byte queries (the workspace, the outputs,
    the memory report outside the hand-off's fields) are the deterministic package's on every architecture: the
    per-call partials ``dA_part`` / ``db_part`` in the backward totals, no accumulator; an unknown gate mode is a
    ``ValueError``."""
    rows = list(SHAPES) + [
        ((2000,) * 3, 64),
        ((129, 300), 16),
        ((63, 1, 65), 16),
        ((129,), 128),
    ]
    for lengths, heads in rows:
        S, N = sum(lengths), len(lengths)
        for sm in (148, 160, 212):
            for arch in ARCHES:
                where = (lengths, heads, sm, arch)
                pg = cake_backend.plan(S, N, heads, sm, arch)
                pr = cake_backend.plan(S, N, heads, sm, arch, gate="raw")
                dr = det_backend.plan(S, N, heads, sm, arch, gate="raw")
                assert (pg.gate, pr.gate, dr.gate) == ("g", "raw", "raw"), where
                assert pr is not pg, where
                assert cake_backend.plan(S, N, heads, sm, arch, gate="raw") is pr
                assert cake_backend.plan(S, N, heads, sm, arch, gate="g") is pg
                assert pr.gate_fused is True and dr.gate_fused is True, where
                for field in (
                    "S",
                    "N",
                    "H",
                    "sm_count",
                    "arch",
                    "NC_max",
                    "NC_pad",
                    "R",
                    "recur",
                    "P",
                    "NSEG",
                    "proportional",
                    "seg_arg",
                    "gsize",
                ):
                    assert getattr(pr, field) == getattr(dr, field), (where, field)
                    assert getattr(pr, field) == getattr(pg, field), (where, field)
                assert set(pr.records) == set(dr.records) == set(pg.records), where
                for path, by_stage in pr.records.items():
                    at = (*where, path)
                    mapped = {
                        stage: RAW_DQKG.get(name, name)
                        for stage, name in by_stage.items()
                    }
                    backward = path not in ("fwd", "fwd_nosave")
                    if backward:
                        # the deterministic parameter-gradient pair, on every architecture
                        assert {
                            stage: by_stage[stage] for stage in RAW_PARAM_PAIR
                        } == RAW_PARAM_PAIR, at
                    assert mapped == dr.records[path], at
                    assert not set(by_stage.values()) & S1_RECORDS, at
                    for stage, name in by_stage.items():
                        if name is None:
                            continue
                        if stage == "dqkg":
                            assert name in RAW_DQKG, at
                        if stage not in GATE_STAGES:
                            assert name == pg.records[path][stage], (*at, stage)
                        assert pr.launches(stage, path) == dr.launches(stage, path)
                        for i in range(pr.launches(stage, path)):
                            assert pr.grid(stage, path, i) == dr.grid(stage, path, i), (
                                *at,
                                stage,
                                i,
                            )
                    assert pr.dispatched("param_finalize", path) is backward, at
                    assert not pr.dispatched("gate_epilogue", path), at
                    assert not pg.dispatched("param_finalize", path), at
                    if backward:
                        assert pr.grid("param_finalize", path) == (
                            heads,
                            max(1, -(-pr.NC_pad // 128)),
                            1,
                        ), at
                        assert pr.launches("param_finalize", path) == 1, at
                    for tables in (True, False):
                        assert pr.launch_count(path, tables=tables) == (
                            dr.launch_count(path, tables=tables)
                        ), (*at, tables)
                for mode in ("on", "auto", "off"):
                    assert not cake_backend._handoff_engages(pr, mode, 0), (where, mode)
                for mode in cake_backend.MODES:
                    assert pr.workspace_bytes(mode) == dr.workspace_bytes(mode), (
                        where,
                        mode,
                    )
        for arch in ARCHES:
            q = {"sm_count": 148, "arch": arch}
            for backward, save_mode in itertools.product(
                (True, False), ("light", "tape")
            ):
                tag = (lengths, heads, arch, backward, save_mode)
                nd_bytes = cake_backend.workspace_bytes(
                    S, N, heads, backward=backward, save_mode=save_mode, gate="raw", **q
                )
                det_bytes = det_backend.workspace_bytes(
                    S, N, heads, backward=backward, save_mode=save_mode, gate="raw", **q
                )
                assert nd_bytes == det_bytes, tag
            for save_mode in ("light", "tape"):
                tag = (lengths, heads, arch, save_mode)
                nd = cake_backend.memory_report(S, N, heads, save_mode, gate="raw", **q)
                det = det_backend.memory_report(S, N, heads, save_mode, gate="raw", **q)
                # the in-kernel Q/K L2 norm fields (#7817) at their off values in both reports (the flag off)
                assert det["l2n"] == "off" and not any(
                    det[key] for key in DET_L2N_REPORT_KEYS[1:]
                ), tag
                # the per-call partials in both backward totals, as the deterministic package's; no accumulator
                assert {key: nd[key] for key in det} == det, tag
                assert det["bwd_partials"] > 0 and nd["gate"] == "raw", tag
                assert "param_acc" not in nd, tag
                assert nd["handoff_bytes"] == nd["handoff_budget_bytes"] == 0
                assert nd["handoff_bytes_rows"] is None
                g_report = cake_backend.memory_report(S, N, heads, save_mode, **q)
                assert g_report["bwd_partials"] == 0, tag
                assert "param_acc" not in g_report, tag
        for backward in (True, False):
            assert cake_backend.output_bytes(
                S, heads, backward=backward, gate="raw"
            ) == det_backend.output_bytes(S, heads, backward=backward, gate="raw")
    for gate in ("G", None, "activated"):
        with pytest.raises(ValueError, match="gate must be one of"):
            cake_backend.plan(129, 1, 4, 148, "sm_100a", gate=gate)
        with pytest.raises(ValueError, match="gate must be one of"):
            cake_backend.memory_report(
                129, 1, 4, sm_count=148, arch="sm_100a", gate=gate
            )


def test_plan_refuses_a_mixed_parameter_gradient_pair(monkeypatch) -> None:
    """``plan`` refuses a raw plan (the in-kernel Q/K L2 norm off) whose GLM intra and finalize records do not pair
    on a backward path (``glm_intra_raw`` without a finalize, or before the l2n modes' ``param_finalize_l2n``):
    ``glm_intra_raw`` writes the per-chunk partials that ``param_finalize`` reduces.  The registry's own pair plans on
    every architecture."""
    jit = cake_backend.cake_jit
    real = jit.select_record
    backward = ("light", "light_states", "tape", "replay")

    def forcing(gate_of: str, glm: str, fin: str | None):
        def select(stage, *, gate="g", **scalars):
            if gate == gate_of and scalars["path"] in backward:
                if stage == "glm_intra":
                    return glm
                if stage == "param_finalize":
                    return fin
            return real(stage, gate=gate, **scalars)

        return select

    cases = (
        ("raw", "glm_intra_raw", None, "param_finalize record"),
        ("raw", "glm_intra_raw", "param_finalize_l2n", "param_finalize record"),
    )
    try:
        for gate, glm, fin, match in cases:
            monkeypatch.setattr(jit, "select_record", forcing(gate, glm, fin))
            cake_backend._plan.cache_clear()
            with pytest.raises(ValueError, match=match):
                cake_backend.plan(129, 1, 4, 148, "sm_100a", gate=gate)
        # the registry's own pairs plan
        monkeypatch.setattr(jit, "select_record", real)
        cake_backend._plan.cache_clear()
        for arch in ARCHES:
            for h in (4, 64):
                p = cake_backend.plan(129, 1, h, 148, arch, gate="raw")
                for path in backward[:1] + backward[2:]:
                    assert (
                        p.record("glm_intra", path),
                        p.record("param_finalize", path),
                    ) == ("glm_intra_raw", "param_finalize"), (arch, h, path)
    finally:
        monkeypatch.setattr(jit, "select_record", real)
        cake_backend._plan.cache_clear()


def test_every_stage_selects_the_expected_record_and_grid() -> None:
    for nc_pad, n, h, sm in PLAN_GRID:
        assert cake_jit.segment_plan(n, h, sm) == _expected_segments(n, h, sm)
        for arch in ARCHES:
            route, segments, nseg = _expected_segments(n, h, sm, arch)
            assert cake_jit.segment_plan(n, h, sm, arch) == (route, segments, nseg)
            for path in PATHS:
                if path == "light_states" and route == "serial":
                    continue
                expected = _expected_records(nc_pad, n, h, sm, arch, path)
                for stage, want in expected.items():
                    name = cake_jit.select_record(
                        stage,
                        num_chunks_pad=nc_pad,
                        num_seqs=n,
                        num_heads=h,
                        sm_count=sm,
                        arch=arch,
                        path=path,
                    )
                    assert name == want, (stage, nc_pad, n, h, sm, arch, path)
                    if name is None:
                        assert stage in OPTIONAL_STAGES
                        continue
                    for launch in range(cake_jit.launches(name)):
                        grid = cake_jit.launch_grid(
                            name,
                            num_chunks_pad=nc_pad,
                            num_seqs=n,
                            num_heads=h,
                            sm_count=sm,
                            num_segments=nseg,
                            launch=launch,
                        )
                        assert grid == _expected_grid(
                            name, nc_pad, n, h, sm, nseg, launch
                        ), (name, nc_pad, n, h, sm, launch)


def test_the_saving_forward_reads_the_token_rows_where_a_program_ships() -> None:
    # The intra program writes the internal Aqk / Akk copies (write_int) iff a later program of the path reads them:
    # never on the saving forward's fused serial rows, overlap rows and paired serial rows (the 32-column form below
    # the fused threshold: their token-row records), always on the forward without a saved set and on the replay.
    for n in (1, 3, 4, 7, 300):
        for h in (1, 2, 4, 8, 16, 32, 64):
            for sm in (132, 148, 160, 212):
                for arch in ARCHES:
                    p = cake_backend.plan(4096, n, h, sm, arch)
                    records = _expected_records(p.NC_pad, n, h, sm, arch, "fwd")
                    token = {
                        name for name in records.values() if name in TOKEN_ROW_RECORDS
                    }
                    seg = _expected_segments(n, h, sm, arch)[0] == "seg"
                    walk = p.NSEG * h
                    fused = (
                        not seg
                        and h >= FUSED_FWD_MIN_H
                        and (arch != "sm_107a" or h <= FUSED_FWD_MIN_H)
                    )
                    ov1 = (
                        seg
                        and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
                        and walk <= OV1_MAX_SEGMENT_HEADS
                    )
                    paired = (
                        not seg
                        and not fused
                        and n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
                    )
                    paired_v1 = (
                        not seg
                        and not fused
                        and n * h * RECUR_SLICES["v2"] > sm * RECUR_CTAS_PER_SM
                        and arch in PAIRED_V1_ARCHES
                    )
                    assert bool(token) == (fused or ov1 or paired or paired_v1), (
                        n,
                        h,
                        sm,
                        arch,
                        token,
                    )
                    if fused:
                        assert token == {"fwdh_wy_rows"}
                    if ov1:
                        assert "wy_fwd_rows" in token and len(token) == 2
                    if paired:
                        assert token == {"wy_fwd_rows", "fwdh_o_v2_rows"}
                    if paired_v1:
                        assert token == {"wy_fwd_rows", "fwdh_o_v1_rows"}
                    assert p.internal_rows("fwd") is (not token), (n, h, sm, arch)
                    assert p.internal_rows("fwd_nosave") is True
                    assert p.internal_rows("replay") is True
                    assert not set(p.records["fwd_nosave"].values()) & set(
                        TOKEN_ROW_RECORDS
                    )
                    assert p.launch_count("fwd") == p.launch_count("fwd_nosave")
                    for stage, name in p.records["fwd"].items():
                        other = p.records["fwd_nosave"][stage]
                        if name in TOKEN_ROW_RECORDS:
                            assert other == INTERNAL_ROW_TWIN[name], (stage, name)
                        else:
                            assert other == name, (stage, name)


def test_select_record_checks_its_inputs() -> None:
    kwargs = dict(num_chunks_pad=4, num_seqs=1, num_heads=4, sm_count=148)
    with pytest.raises(ValueError, match="arch"):
        cake_jit.select_record("table", **kwargs, arch="sm_90a", path="fwd")
    with pytest.raises(ValueError, match="path"):
        cake_jit.select_record("table", **kwargs, arch="sm_100a", path="bwd")
    with pytest.raises(ValueError, match="unknown stage"):
        cake_jit.select_record("fwd_o", **kwargs, arch="sm_100a", path="fwd")
    # a stage no record of the path admits is not an optional stage: the registry refuses the question
    with pytest.raises(ValueError, match="no record admits"):
        cake_jit.select_record("dqkg", **kwargs, arch="sm_100a", path="fwd")


def test_the_l2n_leaf_admits_exactly_when_the_calls_mode_is_listed() -> None:
    """An ``l2n_in`` leaf of a copied rule reads the call's in-kernel Q/K L2 norm mode (``l2n``, ``"off"`` when the
    caller passes none, as the flag-off selection always did): it admits exactly when that mode is one of its modes,
    alone or under ``all_of`` / ``any_of``, whatever the other scalars; an unknown kind still raises."""
    scalars = dict(
        ctas=64,
        num_seqs=2,
        num_heads=8,
        sm_count=148,
        num_segments=0,
        arch="sm_100a",
        path="fwd",
    )
    checked = 0
    for n in range(len(L2N_MODES) + 1):
        for listed in itertools.combinations(L2N_MODES, n):
            leaf = {"kind": "l2n_in", "modes": list(listed)}
            # the default is the flag-off mode
            assert cake_jit._admits(leaf, **scalars) is ("off" in listed), listed
            for mode in L2N_MODES:
                expected = mode in listed
                assert cake_jit._admits(leaf, **scalars, l2n=mode) is expected, (
                    listed,
                    mode,
                )
                for path in PATHS:
                    assert (
                        cake_jit._admits(leaf, **{**scalars, "path": path}, l2n=mode)
                        is expected
                    ), (listed, path, mode)
                always = {"kind": "always"}
                assert (
                    cake_jit._admits(
                        {"kind": "all_of", "rules": [always, leaf]}, **scalars, l2n=mode
                    )
                    is expected
                )
                never = {"kind": "path_in", "paths": []}
                assert (
                    cake_jit._admits(
                        {"kind": "any_of", "rules": [never, leaf]}, **scalars, l2n=mode
                    )
                    is expected
                )
                checked += 1
    assert checked == 2 ** len(L2N_MODES) * len(L2N_MODES)
    with pytest.raises(ValueError, match="unknown Cake KDA training selection rule"):
        cake_jit._admits({"kind": "l2n"}, **scalars)


# The rules of the copied records before the deterministic package's fused Q/K L2 norm class (the rules this
# package copied until the in-kernel Q/K L2 norm; the raw class's records had the same rules as their g records).
# Every record whose rule now reads the l2n leaf is listed.
_PRE_L2N_FWD_INTRA = {
    "kind": "any_of",
    "rules": [
        {"kind": "route_in", "routes": ["serial"]},
        {"kind": "docs_above", "min": 256},
    ],
}
PRE_L2N_RULES = {
    "fwd_intra": _PRE_L2N_FWD_INTRA,
    "bwd_prep_g8_replay": {"kind": "path_in", "paths": ["replay"]},
    "fwd_intra_raw": _PRE_L2N_FWD_INTRA,
}
# The raw class's replay prep serves the flag-off and fallback modes alone: its rule is its pre-l2n rule again (the
# fused mode's rows of the segmented route up to 256 documents are the kb-once replay prep's,
# bwd_prep_g8_replay_kbonce_raw_l2n), so it reads no l2n leaf.
PRE_L2N_RULES_KEPT = {
    "bwd_prep_g8_replay_raw": {"kind": "path_in", "paths": ["replay"]},
}


def test_the_l2n_leaf_keeps_every_pick_of_the_pre_l2n_rules(monkeypatch) -> None:
    """With l2n ``"off"`` (the flag-off selection, no ``l2n`` passed) the re-copied rules select the same record as
    the rules this package copied before the fused Q/K L2 norm class (``PRE_L2N_RULES``) on every stage that holds
    one of them, in both gate modes: every path and architecture, device sizes, head counts and (tokens, documents)
    rows with one-token, sub-chunk, chunk-plus-one and long documents around the 256-document bound.  A later change
    of a copied rule that moves a pick fails here."""
    with_l2n = {
        name
        for name, record in cake_jit.MODULES.items()
        if "l2n_in" in _rule_kinds(record["select_rule"])
    }
    assert with_l2n == set(PRE_L2N_RULES)
    for name, rule in PRE_L2N_RULES_KEPT.items():
        assert cake_jit.MODULES[name]["select_rule"] == rule, name
        assert cake_jit.MODULES[name]["l2n"] == ["off", "fallback"], name
    stages = sorted({cake_jit.MODULES[name]["stage"] for name in PRE_L2N_RULES})
    sm_counts = {
        "sm_100a": (132, 148, 152),
        "sm_103a": (148, 152, 160),
        "sm_107a": (148, 212, 216),
    }
    heads = (1, 2, 4, 8, 15, 16, 17, 32, 64, 96, 128)
    rows = set()
    for docs in (1, 2, 3, 4, 8, 16, 33, 64, 128, 255, 256, 257, 512, 1024):
        for length in (1, 63, 65, 4096):
            if docs * length <= 1 << 20:
                rows.add((docs * length, docs))
        rows.add((docs * BT + 1, docs))
    for tokens in (8192, 65536, 262144):
        rows.update({(tokens, 1), (tokens, 4)})

    def picks() -> dict:
        out = {}
        for arch in ARCHES:
            for sm_count in sm_counts[arch]:
                for num_heads in heads:
                    for tokens, docs in sorted(rows):
                        chunks = -(-tokens // BT) + docs
                        for path in PATHS:
                            for stage in stages:
                                for gate in GATES:
                                    key = (
                                        arch,
                                        sm_count,
                                        num_heads,
                                        tokens,
                                        docs,
                                        path,
                                        stage,
                                        gate,
                                    )
                                    try:
                                        out[key] = cake_jit.select_record(
                                            stage,
                                            num_chunks_pad=chunks + chunks % 2,
                                            num_seqs=docs,
                                            num_heads=num_heads,
                                            sm_count=sm_count,
                                            arch=arch,
                                            path=path,
                                            gate=gate,
                                        )
                                    except ValueError as error:
                                        out[key] = f"ValueError: {error}"
        return out

    now = picks()
    for name, rule in PRE_L2N_RULES.items():
        monkeypatch.setitem(cake_jit.MODULES[name], "select_rule", rule)
    before = picks()
    assert len(now) == len(before) > 0
    moved = [(key, before[key], now[key]) for key in now if now[key] != before[key]]
    assert moved == [], moved[:10]
    # the grid reaches every re-copied record and the raw replay prep
    assert set(PRE_L2N_RULES) | set(PRE_L2N_RULES_KEPT) <= set(now.values())


def test_segment_args_follow_the_proportional_rule() -> None:
    # at most four chains: a proportional block table and seg_arg = -num_docs; otherwise the uniform count
    for sm in (132, 148, 152, 160):
        assert cake_jit.segment_args(1, 4, sm) == (1, -1)
        assert cake_jit.segment_args(4, 1, sm) == (1, -4)
        assert cake_jit.segment_args(2, 2, sm) == (1, -2)
        _route, segments, _nseg = cake_jit.segment_plan(3, 2, sm)
        assert cake_jit.segment_args(3, 2, sm) == (0, segments)
        assert cake_jit.segment_args(1, 16, sm) == (0, 1)  # the serial route
    p = cake_backend.plan(1024, 4, 1, 148, ARCHES[0])  # the sm_100a rows of record
    assert (p.proportional, p.seg_arg) == (1, -4)
    p = cake_backend.plan(1024, 7, 8, 148, ARCHES[0])
    assert (p.proportional, p.seg_arg) == (0, p.P)


def test_the_16_head_rows_of_two_architectures_take_the_segmented_route() -> None:
    # SEGMENTS["plan_heads_by_arch"]: on sm_103a / sm_107a the 16-head rows take the segmented route (their serial
    # walk is the slower form there); sm_100a and a query without an architecture keep the serial walk from nine
    # heads on, and every other head count above the bound stays serial everywhere.
    for sm in (132, 148, 160, 212):
        for n in (1, 2, 7, 300):
            assert cake_jit.segment_plan(n, 16, sm) == ("serial", 1, 0)
            assert cake_jit.segment_plan(n, 16, sm, "sm_100a") == ("serial", 1, 0)
            assert cake_jit.segment_args(n, 16, sm, "sm_100a") == (0, 1)
            for arch in ("sm_103a", "sm_107a"):
                p = min(SEG_P_MAX, max(1, -(-SEG_SLOT_FACTOR * sm // (n * 16))))
                p = min(p, sm // 16)  # one walk CTA per SM per document
                assert cake_jit.segment_plan(n, 16, sm, arch) == ("seg", p, n * p)
                assert cake_jit.segment_args(n, 16, sm, arch) == (0, p)
            for arch in ARCHES:
                for h in (9, 12, 24, 32, 64):
                    assert cake_jit.segment_plan(n, h, sm, arch) == ("serial", 1, 0)
    assert cake_jit.segment_plan(1, 16, 148, "sm_103a") == ("seg", 9, 9)
    assert cake_jit.segment_plan(2, 16, 148, "sm_107a") == ("seg", 9, 18)
    # the route_in rules follow: the same 16-head row runs the segmented programs on sm_103a and the serial
    # 32-column ones on sm_100a (the plan heads bind route_in, not the heads_* leaves); its light backward runs the
    # S1 fused gradient on every architecture (with the L2 hints off on sm_107a)
    kw = dict(num_chunks_pad=64, num_seqs=1, num_heads=16, sm_count=148)
    want = {
        "sm_100a": (
            "table_v2",
            "fwdh_v2h",
            "dhu_v2t",
            "fwd_intra",
            S1_DQKG["dqkg_davq"],
        ),
        "sm_103a": (
            "table_v3",
            "fwdh_v3",
            "dhu_v3p_f",
            "fwd_intra_v14",
            S1_DQKG["dqkg_davq"],
        ),
        "sm_107a": (
            "table_v3",
            "fwdh_v3",
            "dhu_v3_f",
            "fwd_intra_v14",
            S1_DQKG["dqkg_davq_nohint"],
        ),
    }
    for arch, (table, fwdh, dhu, intra, dqkg) in want.items():
        assert cake_jit.select_record("table", **kw, arch=arch, path="light") == table
        assert cake_jit.select_record("fwdh", **kw, arch=arch, path="light") == fwdh
        assert cake_jit.select_record("dhu", **kw, arch=arch, path="light") == dhu
        assert cake_jit.select_record("fwd_intra", **kw, arch=arch, path="fwd") == intra
        assert cake_jit.select_record("dqkg", **kw, arch=arch, path="light") == dqkg
        p = cake_backend.plan(4096, 1, 16, 148, arch)
        assert (p.recur, p.P, p.NSEG) == (
            ("serial", 1, 0) if arch == "sm_100a" else ("seg", 9, 9)
        )
        assert ("light_states" in p.records) is (arch != "sm_100a")
    # arch=None is the current device's plan (the first exported architecture's without one)
    assert (
        cake_backend.plan(4096, 1, 16, 148).recur
        == cake_backend.plan(4096, 1, 16, 148, _current_arch()).recur
    )


def test_carry_choice_picks_the_shortest_critical_path_that_fits() -> None:
    choice = cake_jit.carry_choice
    # eight chains at 64 segments per document on 148 SMs (the uniform grid n * h * (4 G - 4)): groups of 13
    # (G = 5, 16 CTAs per chain, 128 CTAs) give the shortest critical path, 27 hops against the chain's 63
    assert choice(4, 2, 148, 256, 14) == (13, 128)
    assert (_carry_hops(64, 13), _carry_ctas(64, 13)) == (27, 16)
    # six chains at 64: groups of 10 (G = 7, 24 CTAs per chain: 144 CTAs fit 148 SMs, 23 hops); where they do not
    # fit (132 SMs) groups of 11 (G = 6, 120 CTAs, 24 hops); with 212 SMs groups of 8 (G = 8, 168 CTAs)
    assert choice(3, 2, 148, 192, 14) == (10, 144)
    assert choice(3, 2, 132, 192, 14) == (11, 120)
    assert choice(3, 2, 212, 192, 14) == (8, 168)
    # the proportional rows (at most four chains): the grid bound h * (4 * (NSEG // g) + 2 n); a tie on hops goes
    # to the smaller grid (one chain at 37 segments: g = 5 and g = 4 both take 14 hops, 30 against 38 CTAs)
    assert _carry_hops(37, 5) == _carry_hops(37, 4) == 14
    assert choice(1, 1, 148, 37, 14) == (5, 30)
    assert choice(1, 1, 148, 38, 14) == (5, 30)
    assert choice(1, 2, 148, 37, 14) == (5, 60)
    assert choice(2, 2, 148, 76, 14) == (5, 128)  # g = 4 would need 160 CTAs
    assert choice(2, 2, 148, 74, 14) == (5, 120)
    assert choice(4, 1, 148, 148, 14) == (5, 124)
    assert choice(4, 1, 148, 152, 14) == (5, 128)
    # the gain rule: 16 chains at 37 segments: the best two-level path (g = 13: 25 hops, 128 CTAs) gains 11 hops
    # over the chain's 36, less than the 14 level handoffs; with a cheaper handoff it is taken
    assert choice(8, 2, 148, 296, 14) is None
    assert choice(8, 2, 148, 296, 10) == (13, 128)
    # the fit guard: 80 chains at 8 segments: even the single group (two CTAs per chain) needs 160 CTAs; where it
    # fits, a single group never gains (its critical path is the chain's)
    assert choice(40, 2, 148, 320, 14) is None
    assert choice(40, 2, 160, 320, 0) is None
    # the rule accepts above two heads too (one document at 4 heads, 37 segments: g = 5, 120 CTAs); the records
    # bound the head count, not the rule
    assert choice(1, 4, 148, 37, 14) == (5, 120)
    # the serial route (no segments), fewer than two segments per document, a count that is no multiple
    assert choice(4, 2, 148, 0, 14) is None
    assert choice(4, 2, 148, 4, 14) is None
    assert choice(4, 2, 148, 6, 14) is None
    # the independent spelling agrees over a grid of problems and handoff costs
    for n, h, sm, p in itertools.product(
        (1, 2, 3, 4, 7, 8, 40),
        (1, 2, 4),
        (132, 148, 160, 212),
        (1, 2, 5, 9, 37, 38, 64),
    ):
        for s_hops in (0, 10, 14):
            assert choice(n, h, sm, n * p, s_hops) == _carry_choice(
                n, h, sm, n * p, s_hops
            ), (n, h, sm, p, s_hops)


def test_flag_words_count_three_per_cta() -> None:
    assert cake_jit.CARRY_FLAG_WORDS_PER_CTA == 3
    for grid in (1, 30, 128, 148, 216):
        assert cake_jit.flag_words(grid) == 3 * grid
    # every grid the rule can pick (one CTA per SM at most) fits the identity buffer's words a workspace views as
    # the flags
    for sm in (132, 148, 160, 212, 216):
        assert cake_jit.flag_words(sm) <= D * D


def test_the_two_level_carry_dispatches_on_every_architecture_at_most_two_heads() -> (
    None
):
    # The two-level carry takes the backward paths and the forward's non-overlap rows on every architecture at one
    # or two heads where its rule accepts; never above two heads, never on the forward's overlap rows
    # (seg_carry_ov); the chain carry where the rule rejects.
    sm = 148
    accepted = ((4, 2), (3, 2), (1, 1), (2, 2), (4, 1), (1, 2))
    for n, h in accepted:
        for arch in ARCHES:
            _route, _p, nseg = _expected_segments(n, h, sm, arch)
            assert _carry_choice(n, h, sm, nseg) is not None, (n, h, arch)
            for path in PATHS:
                walk = nseg * h
                ov1 = (
                    path in ("fwd", "fwd_nosave")
                    and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
                    and walk <= OV1_MAX_SEGMENT_HEADS
                )
                if ov1:
                    want = "seg_carry_ov"
                elif arch in CARRY_V8_ARCHES:
                    want = "seg_carry_v8"
                else:
                    want = "seg_carry"
                got = cake_jit.select_record(
                    "seg_carry",
                    num_chunks_pad=64,
                    num_seqs=n,
                    num_heads=h,
                    sm_count=sm,
                    arch=arch,
                    path=path,
                )
                assert got == want, (n, h, arch, path)
    # above two heads the chain carry although the rule accepts; where the rule rejects (16 chains at 37
    # segments) the chain carry on every architecture; the overlap pair on the forward of the latter row
    assert _carry_choice(1, 4, sm, 37) is not None
    assert _carry_choice(8, 2, sm, 296) is None
    for n, h in ((1, 4), (8, 2)):
        for arch in ARCHES:
            for path in PATHS:
                nseg = _expected_segments(n, h, sm, arch)[2]
                ov1 = path in ("fwd", "fwd_nosave") and (
                    nseg * h >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
                )
                got = cake_jit.select_record(
                    "seg_carry",
                    num_chunks_pad=64,
                    num_seqs=n,
                    num_heads=h,
                    sm_count=sm,
                    arch=arch,
                    path=path,
                )
                want = "seg_carry_ov" if ov1 else "seg_carry"
                assert got == want, (n, h, arch, path)
    # the plans: the group size where a path dispatches the two-level carry, zero otherwise
    p = cake_backend.plan(4096, 4, 2, sm, "sm_100a")
    assert p.gsize == 13 and p.grid("seg_carry", "light") == (128, 1, 1)
    assert p.records["fwd"]["seg_carry"] == "seg_carry_ov"
    for path in ("light", "light_states", "tape", "replay"):
        assert p.records[path]["seg_carry"] == "seg_carry_v8"
        assert p.launches("seg_carry", path) == 1
    # sm_103a: the same 64 segments per document, the same choice
    p = cake_backend.plan(4096, 4, 2, sm, "sm_103a")
    assert p.gsize == 13 and p.grid("seg_carry", "light") == (128, 1, 1)
    assert p.records["fwd"]["seg_carry"] == "seg_carry_ov"
    for path in ("light", "light_states", "tape", "replay"):
        assert p.records[path]["seg_carry"] == "seg_carry_v8"
    p = cake_backend.plan(4096, 1, 1, sm, "sm_107a")  # 38 segments: g = 5, 30 CTAs
    assert p.gsize == 5 and p.grid("seg_carry", "fwd") == (30, 1, 1)
    assert p.records["fwd"]["seg_carry"] == "seg_carry_v8"
    p = cake_backend.plan(
        4096, 1, 1, sm, "sm_100a"
    )  # 37 segments on sm_100a: the same choice
    assert p.gsize == 5 and p.grid("seg_carry", "light") == (30, 1, 1)
    assert cake_backend.plan(4096, 8, 2, sm, "sm_100a").gsize == 0
    assert cake_backend.plan(4096, 1, 4, sm, "sm_107a").gsize == 0
    assert cake_backend.plan(4096, 1, 16, sm, "sm_107a").gsize == 0


def test_carry_groups_grid_follows_the_rule_or_a_pinned_group_size() -> None:
    def grid(**kw):
        return cake_jit.launch_grid("seg_carry_v8", num_chunks_pad=64, **kw)

    whole = dict(num_seqs=4, num_heads=2, sm_count=148, num_segments=256)
    assert grid(**whole) == (128, 1, 1)
    assert grid(**whole, gsize=0) == (128, 1, 1)
    assert grid(**whole, gsize=13) == (128, 1, 1)
    # a pinned group size (a document group of a batched problem keeps the whole problem's): the uniform grid
    # num_seqs * num_heads * ctas(P, gsize) at the group's chain count, P segments per document
    for n in (1, 2, 3):
        assert grid(
            num_seqs=n, num_heads=2, sm_count=148, num_segments=64 * n, gsize=13
        ) == (n * 2 * _carry_ctas(64, 13), 1, 1)
    # the rule's rejection (16 chains at 37 segments) is the grid's
    with pytest.raises(ValueError, match="rejects"):
        grid(num_seqs=8, num_heads=2, sm_count=148, num_segments=296)
    # the other grid kinds ignore the group size
    assert cake_jit.launch_grid(
        "seg_carry",
        num_chunks_pad=4,
        num_seqs=4,
        num_heads=2,
        sm_count=148,
        num_segments=256,
        gsize=13,
    ) == (16, 1, 1)


def test_segmented_grids_need_segments_and_a_known_launch() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        cake_jit.launch_grid(
            "fwdh_v3", num_chunks_pad=4, num_seqs=1, num_heads=4, sm_count=148
        )
    with pytest.raises(ValueError, match="launch 1 of 1"):
        cake_jit.launch_grid(
            "fwdh_v3",
            num_chunks_pad=4,
            num_seqs=1,
            num_heads=4,
            sm_count=148,
            num_segments=64,
            launch=1,
        )
    with pytest.raises(ValueError, match="launch 2 of 2"):
        cake_jit.launch_grid(
            "dhu_v3_f",
            num_chunks_pad=4,
            num_seqs=1,
            num_heads=4,
            sm_count=148,
            num_segments=64,
            launch=2,
        )
    with pytest.raises(ValueError, match="must all be >= 1"):
        cake_jit.segment_plan(0, 4, 148)


def test_few_chains_cap_the_segment_count() -> None:
    # Four documents at H = 1 (four chains): 38 segments per document (NSEG = 152)
    # instead of the clamp's 64, a 37-step carry chain instead of 63; eight chains
    # keep the clamp.  A host-plan constant: no record changes with it.  sm_100a
    # has a few-chain count of its own, 37 (SEGMENTS["small_segments_by_arch"]);
    # arch=None is the architecture-independent 38.
    for sm in (132, 148, 152, 160):
        assert cake_jit.segment_plan(4, 1, sm) == ("seg", 38, 152)
        assert cake_jit.segment_plan(4, 1, sm, "sm_103a") == ("seg", 38, 152)
        assert cake_jit.segment_plan(4, 1, sm, "sm_107a") == ("seg", 38, 152)
        assert cake_jit.segment_plan(4, 1, sm, "sm_100a") == ("seg", 37, 148)
        # at H = 4 the per-SM cap sm // 4 (33, 37, 38, 40) applies before the few-chains cap of 38 (37 on sm_100a)
        cap = sm // 4
        assert cake_jit.segment_plan(1, 4, sm) == ("seg", min(38, cap), min(38, cap))
        assert cake_jit.segment_plan(1, 4, sm, "sm_100a") == (
            "seg",
            min(37, cap),
            min(37, cap),
        )
        assert cake_jit.segment_plan(8, 1, sm) == ("seg", 64, 512)
        assert cake_jit.segment_plan(8, 1, sm, "sm_100a") == ("seg", 64, 512)
        assert cake_jit.segment_plan(2, 4, sm) == ("seg", cap, 2 * cap)
        for arch, small in (("sm_100a", 37), ("sm_103a", 38), ("sm_107a", 38)):
            p = cake_backend.plan(1024, 4, 1, sm, arch)
            assert (p.recur, p.P, p.NSEG) == ("seg", small, 4 * small)
            assert cake_backend.plan(1024, 8, 1, sm, arch).P == 64
        # plan(arch=None) is the current device's plan, segment count included
        assert (
            cake_backend.plan(1024, 4, 1, sm).P
            == cake_backend.plan(1024, 4, 1, sm, _current_arch()).P
        )
    # below the clamp the cap is inert: four chains on a 32-SM device give 32
    assert cake_jit.segment_plan(4, 1, 32) == ("seg", 32, 128)
    assert cake_jit.segment_plan(4, 1, 32, "sm_100a") == ("seg", 32, 128)


def test_segment_count_is_capped_at_one_walk_cta_per_sm() -> None:
    # P * H <= sm_count: one document's pass-2 (and adjoint) walk CTAs fit one wave of the device and its first
    # pass one round at two CTAs per SM.  152 SMs: 38 segments at H = 4 (four documents: the clamp gives the
    # same), 19 at H = 8; at H = 2 the clamp of 64 binds first.
    for n in (1, 2, 3, 4):
        for h in (1, 2, 4, 8):
            for sm in (132, 148, 152, 160):
                route, p, nseg = cake_jit.segment_plan(n, h, sm)
                assert route == "seg" and p >= 1 and p * h <= sm and nseg == n * p
    assert cake_jit.segment_plan(3, 4, 152) == ("seg", 38, 114)
    assert cake_jit.segment_plan(1, 8, 152) == ("seg", 19, 19)
    assert cake_jit.segment_plan(3, 2, 152) == ("seg", 64, 192)
    assert cake_jit.segment_plan(4, 8, 148) == ("seg", 18, 72)


def test_pairs_grid_needs_an_even_chunk_bound() -> None:
    with pytest.raises(ValueError, match="not even"):
        cake_jit.launch_grid(
            "wy_bwd", num_chunks_pad=3, num_seqs=1, num_heads=1, sm_count=148
        )


def test_pair_kernels_launch_one_cta_per_pair_item() -> None:
    # The WY and dA programs: one CTA per (chunk pair, head) item at any SM count; the CTAs that start take the
    # items of the CTAs that have not started (Cluster Launch Control), so an SM another stream holds costs only
    # its share of the items.  The pair count is a plan scalar.
    for name in (
        "wy_fwd",
        "wy_fwd_rows",
        "wy_bwd",
        "wy_bwd_replay",
        "wy_bwd_v9",
        "dav_serial",
        "dav_serial_replay",
    ):
        record = cake_jit.MODULES[name]
        assert record["grid_rule"] == {"kind": "pair_items"}
        assert ["parameter", "num_pairs"] in [list(pair) for pair in record["arg_plan"]]
        assert record["logical"]["num_pairs"] == "num_pairs"
        for nc_pad, h, sm in (
            (2, 1, 148),
            (64, 4, 148),
            (4096, 64, 132),
            (4096, 1, 160),
        ):
            grid = cake_jit.launch_grid(
                name, num_chunks_pad=nc_pad, num_seqs=1, num_heads=h, sm_count=sm
            )
            assert grid == (max(1, nc_pad // 2 * h), 1, 1)


def test_fused_gradient_launches_one_cta_per_chunk_item() -> None:
    # The fused gradient: one CTA per (chunk, head) item at any SM count, its items handed out the same way (the
    # S1 forms included).
    for name in cake_jit.records_of_stage("dqkg"):
        assert cake_jit.MODULES[name]["grid_rule"] == {"kind": "chunk_items"}
        for nc_pad, h, sm in ((1, 1, 148), (64, 4, 148), (4096, 64, 132)):
            grid = cake_jit.launch_grid(
                name, num_chunks_pad=nc_pad, num_seqs=1, num_heads=h, sm_count=sm
            )
            assert grid == (max(1, nc_pad * h), 1, 1)


def test_the_backward_reads_the_token_rows_without_chunk_layout_copies() -> None:
    # The fused gradient and the GLM intra kernel read q / k / v where they are (TMA boxes on the token rows), so
    # the backward prep ships its copy-free forms alone and the backward layouts hold no slot for such copies.
    # The S1 fused gradient reads no q at all (its dg leaves the q * dq term to the GLM intra program).
    for name in cake_jit.records_of_stage("dqkg"):
        logical = cake_jit.MODULES[name]["logical"]
        assert (logical["v_tma"], logical["k_tma"]) == ("v", "k"), name
        if name in S1_RECORDS:
            assert "q_tma" not in logical and "q" not in logical.values(), name
        else:
            assert logical["q_tma"] == "q", name
    for name in ("glm_intra", S1_GLM["sm_100a"]):
        logical = cake_jit.MODULES[name]["logical"]
        assert (logical["k_e"], logical["q_e"]) == ("k", "q"), name
    for name in (
        "glm_intra_v14",
        "glm_intra_v18",
        S1_GLM["sm_103a"],
        S1_GLM["sm_107a"],
    ):
        assert {"k", "q", "dq_f"} <= set(cake_jit.MODULES[name]["logical"].values())
    preps = (
        "bwd_prep_g8",
        "bwd_prep_g8_replay",
        "bwd_prep_g8_noqg",
        "bwd_prep_g8_noqg_nokb",
    )
    # the in-kernel Q/K L2 norm's fused class (l2n ["fused"] alone, the deterministic package's) has one form per
    # flag-off one
    fused_preps = tuple(f"{name}_l2n" for name in preps)

    def l2n_class(fused: bool) -> tuple[str, ...]:
        # the saved-rstd twin (FUSED_L2N_TWINS) is no plan's form; the l2n class test compares it with its source
        return tuple(
            n
            for n in cake_jit.records_of_stage("bwd_prep")
            if (cake_jit.MODULES[n].get("l2n", ["off"]) == ["fused"]) == fused
            and n not in FUSED_L2N_TWINS
        )

    assert l2n_class(False) == preps
    assert l2n_class(True) == fused_preps
    for name in preps + fused_preps:
        prep = cake_jit.MODULES[name]
        assert prep["grid_rule"] == {"kind": "chunks", "x_factor": 1}
        assert not {"ke_out", "qe_out", "v_int", "vb_out"} & set(prep["logical"])
    # the saved-set paths' prep copies nothing of dout either; the replay's keeps the internal copy
    for name in preps + fused_preps:
        values = set(cake_jit.MODULES[name]["logical"].values())
        replay = name.startswith("bwd_prep_g8_replay")
        assert ("do_int" in values) == replay, name
        assert ("dout" in values) == replay, name
    # the fused forms read the raw rows (the backward's first q / k reader) in place of q / k
    for name in fused_preps:
        values = set(cake_jit.MODULES[name]["logical"].values())
        off = set(cake_jit.MODULES[name.removesuffix("_l2n")]["logical"].values())
        assert {"q_raw", "k_raw", "qn", "kn", "qk_rstd"} <= values, name
        assert not {"q", "k"} & values, name
        assert off - {"q", "k"} <= values, name
    # the tape set's and the replay's prep fold q into qg; the light sets' forms read no q and write no qg (the
    # version-4 set carries the forward's), and the w / u rows' form writes no kb either (no wy launch there)
    for name in ("bwd_prep_g8", "bwd_prep_g8_replay"):
        values = set(cake_jit.MODULES[name]["logical"].values())
        assert {"q", "qg", "kb", "kg", "gk"} <= values, name
    for name in ("bwd_prep_g8_noqg", "bwd_prep_g8_noqg_nokb"):
        values = set(cake_jit.MODULES[name]["logical"].values())
        assert not {"q", "qg"} & values, name
        assert {"kg", "gk", "beta_int", "v_new", "dv2"} <= values, name
        assert ("kb" in values) == (name == "bwd_prep_g8_noqg"), name
    # the fused GLM intra program writes the token-row gate gradients itself and binds no epilogue buffer; the
    # unfused one hands them to the gate epilogue through dg_int / db_int
    v14 = set(cake_jit.MODULES["glm_intra_v14"]["logical"].values())
    v18 = set(cake_jit.MODULES["glm_intra_v18"]["logical"].values())
    unfused = set(cake_jit.MODULES["glm_intra"]["logical"].values())
    epilogue = set(cake_jit.MODULES["gate_epilogue"]["logical"].values())
    assert v18 == v14  # the same program: v18 differs in its named-barrier slots alone
    assert {"dg", "dbeta"} <= v14 and not {"dg_int", "db_int"} & v14
    assert {"dg_int", "db_int"} <= unfused and not {"dg", "dbeta"} & unfused
    assert {"dg_int", "db_int", "dg", "dbeta"} <= epilogue
    assert v14 - {"dg", "dbeta"} == unfused - {"dg_int", "db_int"}
    # the S1 GLM intra programs bind what their deterministic twins bind: gate-fused on sm_103a / sm_107a, the gate
    # epilogue's buffers on sm_100a
    for arch, name in S1_GLM.items():
        values = set(cake_jit.MODULES[name]["logical"].values())
        assert values == (v14 if _gate_fused(arch) else unfused), name
    assert not {"ke", "qe", "v_int", "vb"} & set(cake_jit.LOGICAL_TENSORS)
    for mode, slots in cake_backend.SLOTS.items():
        names = {name for occupants in slots for name, *_ in occupants}
        assert not names & {"ke", "qe", "v_int", "vb"}, mode
        assert ("akk_int" in names) == (mode.startswith(("fwd", "replay"))), mode


def test_the_fused_gradient_turns_its_l2_hints_off_on_the_large_sm107a_rows() -> None:
    # The fused gradient's L2 prefetch and evict-first hints cost about 6 % on the sm_107a rows from 16 heads, so
    # its dAqk-hosting forms ship a second build there with the hints off (the `_nohint` records: the same kernel,
    # launch and operands; the hint thresholds beyond any head count).  The registry dispatches them on sm_107a
    # from 16 heads alone, on either route (its 16-head rows are segmented ones); the other architectures keep the
    # hinted builds.  In this package every backward path of those rows runs the S1 pair, so the builds with the
    # hints off are the S1 fused gradient's (shipped for sm_107a alone) and the deterministic package's are not shipped.
    for det_name in ("dqkg_davq", "dqkg_int_davq"):
        record = cake_jit.MODULES[S1_DQKG[det_name + "_nohint"]]
        hinted = cake_jit.MODULES[S1_DQKG[det_name]]
        assert record["stage"] == hinted["stage"] == "dqkg"
        assert record["library"] != hinted["library"]
        assert list(record["arches"]) == ["sm_107a"]
        assert list(hinted["arches"]) == list(ARCHES)
        assert record["logical"] == hinted["logical"]
        assert record["arg_plan"] == hinted["arg_plan"]
        assert record["launch"] == hinted["launch"]
        assert record["grid_rule"] == hinted["grid_rule"]
        assert det_name + "_nohint" not in cake_jit.MODULES
    for path, base in (
        ("light", "dqkg"),
        ("light_states", "dqkg"),
        ("tape", "dqkg"),
        ("replay", "dqkg_int"),
    ):
        for arch in ARCHES:
            for h in (1, 4, 8, 12, 16, 24, 32, 64, 128):
                seg = _expected_segments(1, h, 148, arch)[0] == "seg"
                davq = seg or h >= DAVQ_MIN_H
                nohint = davq and h >= DQKG_NOHINT_MIN_H_BY_ARCH.get(arch, h + 1)
                want = base + ("_davq" if davq else "") + ("_nohint" if nohint else "")
                if _s1(arch, path, h):
                    want = S1_DQKG[want]
                got = cake_jit.select_record(
                    "dqkg",
                    num_chunks_pad=64,
                    num_seqs=1,
                    num_heads=h,
                    sm_count=148,
                    arch=arch,
                    path=path,
                )
                assert got == want, (path, arch, h)


def test_the_saved_set_paths_read_the_token_rows_and_the_replay_the_internal_rows() -> (
    None
):
    # Operand sources: the records of the saved-set paths bind the saved Aqk / Akk token rows and the user's
    # dout; their replay twins bind the arena's internal rows (aqk_int / akk_int / do_int).
    pairs = {
        "wy_bwd": "wy_bwd_replay",
        "fwdh_wy_bwd": "fwdh_wy_bwd_replay",
        "dav_serial": "dav_serial_replay",
        "dhu_v3_f": "dhu_v3_f_replay",
        "dhu_v3p_f": "dhu_v3p_f_replay",
        "dhu_v3r_f": "dhu_v3r_f_replay",
        "dhu_v2t": "dhu_v2t_replay",
        "dhu_v1": "dhu_v1_replay",
        "dhu_v1_wt": "dhu_v1_wt_replay",
        "dhu_v3s_f": "dhu_v3s_f_replay",
        "dqkg": "dqkg_int",
        "dqkg_davq": "dqkg_int_davq",
        S1_DQKG["dqkg_davq"]: S1_DQKG["dqkg_int_davq"],
        S1_DQKG["dqkg_davq_nohint"]: S1_DQKG["dqkg_int_davq_nohint"],
    }
    token = {"aqk_src", "akk_src", "dout"}
    internal = {"aqk_int", "akk_int", "do_int"}
    for saved_name, replay_name in pairs.items():
        saved_values = set(cake_jit.MODULES[saved_name]["logical"].values())
        replay_values = set(cake_jit.MODULES[replay_name]["logical"].values())
        assert saved_values & token and not saved_values & internal, saved_name
        assert replay_values & internal and not replay_values & token, replay_name
        assert (
            cake_jit.MODULES[saved_name]["library"]
            != cake_jit.MODULES[replay_name]["library"]
        )
    # the segmented adjoint recurrences of the many-chain rows read dout on the saved-set paths (their w^T forms too)
    for name in (
        "dhu_v3_f",
        "dhu_v3_f_wt",
        "dhu_v3p_f",
        "dhu_v3p_f_wt",
        "dhu_v3r_f",
        "dhu_v3r_f_wt",
    ):
        assert "dout" in cake_jit.MODULES[name]["logical"].values(), name
    # the saving forward's token-row twins bind the saved Aqk / Akk where the internal-row forms bind the arena's copies
    saved_rows = {"aqk_ext", "akk_ext"}
    for rows_name, internal_name in INTERNAL_ROW_TWIN.items():
        internal_values = set(cake_jit.MODULES[internal_name]["logical"].values())
        rows_values = set(cake_jit.MODULES[rows_name]["logical"].values())
        assert internal_values & internal and not internal_values & saved_rows, (
            internal_name
        )
        assert rows_values & saved_rows and not rows_values & internal, rows_name
        assert (
            cake_jit.MODULES[internal_name]["library"]
            != cake_jit.MODULES[rows_name]["library"]
        )
        assert set(cake_jit.MODULES[internal_name]["logical"]) == set(
            cake_jit.MODULES[rows_name]["logical"]
        ), rows_name  # the same kernel parameters
    # the transposed w tile: the serial fused recompute writes w_t and its adjoint reads it, the segmented
    # saved-state pass 2 writes w_tk and its adjoint reads it
    assert cake_jit.MODULES["fwdh_wy_bwd"]["logical"]["w_out"] == "w_t"
    assert cake_jit.MODULES["dhu_v1_wt"]["logical"]["w_tma"] == "w_t"
    assert cake_jit.MODULES["fwdh_p3_wy"]["logical"]["w_t_out"] == "w_tk"
    assert cake_jit.MODULES["dhu_v3s_f_wt"]["logical"]["w_tma"] == "w_tk"
    for name in ("fwdh_v1", "dhu_v1", "dhu_v3s_f", "fwdh_p3"):
        assert cake_jit.MODULES[name]["logical"]["w_tma"] == "w", name


def test_the_forward_wy_forms_the_gated_values_from_the_token_rows() -> None:
    # The WY programs read the user's v and beta through the chunk table and form the gated values in shared
    # memory: no prep writes them and no layout holds a slot for them.
    fwd, bwd = cake_jit.MODULES["wy_fwd"], cake_jit.MODULES["wy_bwd"]
    assert fwd["library"] != bwd["library"]
    assert (fwd["stage"], bwd["stage"]) == ("wy_fwd", "wy")
    assert fwd["grid_rule"] == bwd["grid_rule"]
    for record in (fwd, bwd):
        assert (record["logical"]["v_tma"], record["logical"]["beta"]) == ("v", "beta")
        assert {"chunk_bos", "chunk_len"} <= set(record["logical"].values())
    assert (
        fwd["logical"]["akk_tma"] == "akk_int"
        and bwd["logical"]["akk_tma"] == "akk_src"
    )
    # the forward intra programs write the prep's outputs themselves (no prep launch) and read no v either; the
    # hand-off twins also write the internal beta
    for name in ("fwd_intra", "fwd_intra_v14", *sorted(TWIN_RECORDS)):
        intra = set(cake_jit.MODULES[name]["logical"].values())
        assert not intra & {"v", "vb"}, name
        assert {"gk", "kb", "kg", "qg", "v_new", "write_prep"} <= intra, name
        assert ("beta_int" in intra) is (name in TWIN_RECORDS), name
    # the segmented route's intra program computes the chunk layout from cu_seqlens and writes the five tables
    v14 = set(cake_jit.MODULES["fwd_intra_v14"]["logical"].values())
    tables = set(cake_backend.TABLE_NAMES)
    assert {
        "cu_seqlens",
        "num_seqs",
        "segments",
        "proportional",
    } <= v14 and tables <= v14
    assert "cu_seqlens" not in cake_jit.MODULES["fwd_intra"]["logical"].values()
    assert "fwd_prep" not in cake_jit.STAGES
    assert "wy_fwd" in cake_jit.STAGES_FWD and "wy_fwd" not in cake_jit.STAGES_BWD
    assert "wy" in cake_jit.STAGES_BWD and "wy" not in cake_jit.STAGES_FWD
    for mode, slots in cake_backend.SLOTS.items():
        names = {name for occupants in slots for name, *_ in occupants}
        assert "vb" not in names and "kb" in names, mode
    # the fused forward recurrence takes the WY operands itself: no wy_fwd record on its rows (the saving forward's
    # form from the saved Akk / Aqk token rows)
    fused = cake_jit.MODULES["fwdh_wy"]["logical"]
    assert {"akk_int", "kb", "v", "beta", "out"} <= set(fused.values())
    rows = cake_jit.MODULES["fwdh_wy_rows"]["logical"]
    assert {"akk_ext", "aqk_ext", "kb", "v", "beta", "out"} <= set(rows.values())
    assert not {"akk_int", "aqk_int"} & set(rows.values())
    # the saving forward's overlap rows launch the backward's saved-Akk WY program under the forward stage
    rows_wy = cake_jit.MODULES["wy_fwd_rows"]
    assert rows_wy["stage"] == "wy_fwd" and rows_wy["library"] == bwd["library"]
    assert (
        rows_wy["logical"][next(k for k, v in bwd["logical"].items() if v == "akk_src")]
        == "akk_ext"
    )


# rows of the three routes' intra records (the segmented route up to 256 documents: fwd_intra_v14; the serial route
# and more than 256 documents: fwd_intra), at sizes the plans accept
TWIN_PLAN_ROWS = [
    ((100, 640, 77, 1000, 129), 2),
    ((63, 1, 65), 4),
    ((65, 200), 32),
    ((63, 1, 65), 64),
    ((65,) * 257, 2),
    ((65,) * 257, 16),
    ((129, 300, 70), 16),
]


def test_the_handoff_twins_are_their_intra_records_plus_beta_int() -> None:
    """The checkpoint hand-off's intra twins (``cake_jit.HANDOFF_TWINS``): each twin is its intra record's program
    that also writes the internal ``beta_int``: the same stage, grid, launch, entry and architectures, the record's
    operands plus ``beta_int`` right before the grid arguments, a library of its own; no rule admits it (``_admits``
    is false on every ``PLAN_GRID`` cell, architecture and path), ``select_record`` never returns it, and
    ``Plan.handoff_twin`` names the twin of the plan's intra record."""
    assert cake_jit.HANDOFF_TWINS == HANDOFF_TWINS
    assert "beta_int" in cake_jit.LOGICAL_TENSORS
    for parent, twin in HANDOFF_TWINS.items():
        a, b = cake_jit.MODULES[parent], cake_jit.MODULES[twin]
        assert a["stage"] == b["stage"] == "fwd_intra", twin
        for key in ("grid_rule", "launch", "ffi_entry", "arches"):
            assert b[key] == a[key], (twin, key)
        assert b["logical"] == {**a["logical"], "beta_int": "beta_int"}, twin
        plan_a = [list(pair) for pair in a["arg_plan"]]
        assert [list(pair) for pair in b["arg_plan"]] == [
            *plan_a[:-3],
            ["buffer", "beta_int"],
            *plan_a[-3:],
        ], twin
        assert b["select_rule"] == {"kind": "path_in", "paths": []}, twin
        assert b["library"] != a["library"], twin
    for nc_pad, n, h, sm in PLAN_GRID:
        for arch in ARCHES:
            nseg = cake_jit.segment_plan(n, h, sm, arch)[2]
            for path in PATHS:
                for twin in TWIN_RECORDS:
                    assert not cake_jit._admits(
                        cake_jit.MODULES[twin]["select_rule"],
                        ctas=nc_pad * h,
                        num_seqs=n,
                        num_heads=h,
                        sm_count=sm,
                        num_segments=nseg,
                        arch=arch,
                        path=path,
                    ), (twin, nc_pad, n, h, sm, arch, path)
                name = cake_jit.select_record(
                    "fwd_intra",
                    num_chunks_pad=nc_pad,
                    num_seqs=n,
                    num_heads=h,
                    sm_count=sm,
                    arch=arch,
                    path=path,
                )
                assert name in HANDOFF_TWINS, (name, nc_pad, n, h, sm, arch, path)
    for (lengths, heads), arch, sm in itertools.product(
        TWIN_PLAN_ROWS, ARCHES, (148, 152)
    ):
        p = cake_backend.plan(sum(lengths), len(lengths), heads, sm, arch)
        assert p.handoff_twin == HANDOFF_TWINS[p.record("fwd_intra", "fwd")], (
            lengths,
            heads,
            arch,
            sm,
        )


def test_the_tape_wy_twin_is_its_wy_record_without_u() -> None:
    """The tape path's w-only WY twin (``cake_jit.TAPE_TWINS``): its record is its WY record's without the
    parameters of ``u`` and of the operands only ``u`` reads (``v``, ``beta``), in the record's order, with the
    same stage, grid rule, cluster, cooperative and PDL attributes, entry and architectures, a library of its own
    and the record's grid; no rule admits it (``_admits`` is false on every ``PLAN_GRID`` cell, architecture and
    path), ``select_record`` never returns it, and ``Plan.tape_wy_twin`` names it exactly where the tape path
    launches its WY record (every path that launches ``wy`` on the tape path launches that record)."""
    assert cake_jit.TAPE_TWINS == TAPE_TWINS
    for parent, twin in TAPE_TWINS.items():
        a, b = cake_jit.MODULES[parent], cake_jit.MODULES[twin]
        assert a["stage"] == b["stage"] == "wy", twin
        for key in ("grid_rule", "ffi_entry", "arches"):
            assert b[key] == a[key], (twin, key)
        for key in ("cluster", "cooperative", "use_pdl"):
            assert b["launch"][key] == a["launch"][key], (twin, key)
        assert b["logical"] == {
            key: value
            for key, value in a["logical"].items()
            if value not in TAPE_WY_DROPPED
        }, twin
        assert "w" in b["logical"].values() and "u" in a["logical"].values(), twin
        assert [list(pair) for pair in b["arg_plan"]] == [
            list(pair)
            for pair in a["arg_plan"]
            if a["logical"].get(pair[1]) not in TAPE_WY_DROPPED
        ], twin
        assert b["select_rule"] == {"kind": "path_in", "paths": []}, twin
        assert b["library"] != a["library"], twin
    for nc_pad, n, h, sm in PLAN_GRID:
        for arch in ARCHES:
            nseg = cake_jit.segment_plan(n, h, sm, arch)[2]
            for path in PATHS:
                for twin in TAPE_WY_RECORDS:
                    assert not cake_jit._admits(
                        cake_jit.MODULES[twin]["select_rule"],
                        ctas=nc_pad * h,
                        num_seqs=n,
                        num_heads=h,
                        sm_count=sm,
                        num_segments=nseg,
                        arch=arch,
                        path=path,
                    ), (twin, nc_pad, n, h, sm, arch, path)
                name = cake_jit.select_record(
                    "wy",
                    num_chunks_pad=nc_pad,
                    num_seqs=n,
                    num_heads=h,
                    sm_count=sm,
                    arch=arch,
                    path=path,
                )
                assert name not in TAPE_WY_RECORDS, (name, nc_pad, n, h, sm, arch, path)
                if path == "tape":
                    assert name in TAPE_TWINS, (name, nc_pad, n, h, sm, arch)
    for (lengths, heads), arch, sm in itertools.product(
        TWIN_PLAN_ROWS, ARCHES, (148, 152)
    ):
        p = cake_backend.plan(sum(lengths), len(lengths), heads, sm, arch)
        assert p.dispatched("wy", "tape"), (lengths, heads, arch, sm)
        assert p.tape_wy_twin == TAPE_TWINS[p.record("wy", "tape")], (
            lengths,
            heads,
            arch,
            sm,
        )
        assert p.grid("wy", "tape") == cake_jit.launch_grid(
            p.tape_wy_twin,
            num_chunks_pad=p.NC_pad,
            num_seqs=p.N,
            num_heads=heads,
            sm_count=sm,
        )


def test_the_serial_walk_orders_the_documents_and_the_carry_splits_the_chain() -> None:
    # The serial recurrences take the layout table's longest-first document order (seq_order, a table of every
    # call); seg_carry runs two CTAs per (document, head) chain and reads the block table's chain geometry.
    for name in (
        "fwdh_v1",
        "fwdh_v1_sm107",
        "fwdh_v2",
        "fwdh_v2h",
        "fwdh_o_v1",
        "fwdh_o_v1_rows",
        "fwdh_o_v2",
        "fwdh_o_v2_rows",
        "fwdh_wy",
        "fwdh_wy_rows",
        "fwdh_wy_bwd",
        "dhu_v2t",
    ):
        record = cake_jit.MODULES[name]
        assert record["logical"]["seq_order"] == "seq_order", name
        assert ["buffer", "seq_order"] in [list(pair) for pair in record["arg_plan"]]
    assert "seq_order" in cake_backend.TABLE_NAMES
    carry = cake_jit.MODULES["seg_carry"]
    assert carry["grid_rule"] == {"kind": "docs_heads", "ctas_per_chain": 2}
    assert carry["logical"]["chain_seg_start"] == "seg_table"
    for n, h, sm in ((1, 1, 148), (4, 4, 152), (7, 8, 132)):
        grid = cake_jit.launch_grid(
            "seg_carry", num_chunks_pad=4, num_seqs=n, num_heads=h, sm_count=sm
        )
        assert grid == (2 * n * h, 1, 1)
    # the two-level carry: the same stage and chain operands, plus the carried states viewed as tiles (the
    # groups' products), the group size and the release flags under an epoch
    v8 = cake_jit.MODULES["seg_carry_v8"]
    assert v8["stage"] == "seg_carry" and v8["library"] != carry["library"]
    chain_values = set(carry["logical"].values())
    v8_values = set(v8["logical"].values())
    assert {"carry_b", "carry_out", "seg_m", "seg_table", "reverse"} <= chain_values
    assert chain_values - {"num_seqs"} <= v8_values
    assert {"carry_out_tiles", "gsize", "flags", "epoch"} <= v8_values
    assert not {"carry_out_tiles", "gsize", "flags", "epoch"} & chain_values
    assert cake_jit.launch_grid(
        "seg_carry_v8",
        num_chunks_pad=4,
        num_seqs=4,
        num_heads=2,
        sm_count=148,
        num_segments=256,
    ) == (128, 1, 1)
    # the segmented programs read the block table and the walk programs' segment scalar
    for name in (
        "fwdh_v3",
        "fwdh_p3",
        "fwdh_p3_o",
        "fwdh_p3_wy",
        "fwdh_p3_o_ov_rows",
        "dhu_v3_f",
        "dhu_v3p_f",
        "dhu_v3r_f_wt",
        "dhu_v3s_f",
    ):
        logical = cake_jit.MODULES[name]["logical"]
        assert logical["seq_chunk_start"] == "seg_table", name
        assert logical["segments"] == "seg_arg", name
    assert cake_jit.MODULES["seg_table"]["logical"]["proportional"] == "proportional"


def _expected_launch_count(
    records: dict[str, str | None], seg: bool, path: str, tables: bool
) -> int:
    count = 0
    for stage, name in records.items():
        if name is None:
            continue
        if stage in ("table", "seg_table"):
            count += int((not tables) or path in ("fwd", "fwd_nosave", "replay"))
        elif stage == "seg_carry":
            count += int(records.get("fwdh") is not None) + int(
                path not in ("fwd", "fwd_nosave")
            )
        elif name in TWO_PASS_RECORDS:
            count += 2
        else:
            count += 1
    return count


def test_plan_is_cached_on_its_scalars(monkeypatch) -> None:
    """A repeated problem returns the same plan without selecting a record again (the default entry plans a
    repeated shape once); every scalar and the architecture are part of the key, and a refused problem raises
    on every call."""
    first = cake_backend.plan(261393, 4, 4, 152, "sm_100a")
    calls: list = []
    original = cake_jit.select_record

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(cake_jit, "select_record", counting)
    assert cake_backend.plan(261393, 4, 4, 152, "sm_100a") is first
    assert cake_backend.plan(261393.0, 4, 4, 152, "sm_100a") is first
    assert calls == []
    for scalars in ((261392, 4, 4, 152), (261393, 3, 4, 152), (261393, 4, 2, 152)):
        assert cake_backend.plan(*scalars, "sm_100a") is not first
    assert cake_backend.plan(261393, 4, 4, 148, "sm_100a").sm_count == 148
    for arch in ARCHES[1:]:
        assert cake_backend.plan(261393, 4, 4, 152, arch).arch == arch
    for _ in range(2):
        with pytest.raises(ValueError, match="must all be >= 1"):
            cake_backend.plan(0, 1, 1, 148, "sm_100a")


def test_plan_without_arch_takes_the_current_device(monkeypatch) -> None:
    """``plan(..., arch=None)`` selects the records of the current CUDA device's architecture, so its plan serves a
    workspace there on every exported target; without a CUDA device, or on a device no record targets, it takes
    the first exported architecture (whose records the byte queries also take without an ``arch``)."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for arch in ARCHES:
        monkeypatch.setattr(cake_backend, "_capability_arch", lambda index, a=arch: a)
        p = cake_backend.plan(4096, 1, 16, 148)
        assert p.arch == arch
        assert p.records == cake_backend.plan(4096, 1, 16, 148, arch).records
    monkeypatch.setattr(cake_backend, "_capability_arch", lambda index: None)
    assert cake_backend.plan(4096, 1, 16, 148).arch == ARCHES[0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cake_backend.plan(4096, 1, 16, 148).arch == ARCHES[0]


def test_plan_follows_the_chunk_bound_formulas() -> None:
    for lengths, heads in SHAPES:
        S, N = sum(lengths), len(lengths)
        for sm in (132, 148):
            for arch in ARCHES:
                p = cake_backend.plan(S, N, heads, sm, arch)
                nc_max = -(-S // BT) + N
                assert (p.NC_max, p.NC_pad, p.R) == (
                    nc_max,
                    nc_max + nc_max % 2,
                    64 * (nc_max + nc_max % 2),
                )
                assert (p.recur, p.P, p.NSEG) == _expected_segments(N, heads, sm, arch)
                seg = p.recur == "seg"
                assert set(p.records) == {
                    path for path in PATHS if seg or path != "light_states"
                }
                # the plan's round-3 fields: the gate fusion of the architecture, the light set's extra tensors
                # and the two-level carry's group size where a path dispatches it
                assert p.gate_fused is _gate_fused(arch)
                assert p.light_saves_qg is True
                assert p.light_saves_wu is _wu_saved(N, heads, sm, arch)
                two_level = any(
                    records.get("seg_carry") == "seg_carry_v8"
                    for records in p.records.values()
                )
                choice = _carry_choice(N, heads, sm, p.NSEG)
                assert p.gsize == (choice[0] if two_level else 0), (
                    lengths,
                    heads,
                    arch,
                )
                for path in p.records:
                    expected = _expected_records(p.NC_pad, N, heads, sm, arch, path)
                    assert p.records[path] == expected, (path, lengths, heads, sm, arch)
                    for stage in cake_jit.STAGES:
                        if stage not in expected:
                            with pytest.raises(ValueError, match="not a stage of"):
                                p.record(stage, path)
                            continue
                        if not p.dispatched(stage, path):
                            assert stage in OPTIONAL_STAGES
                            with pytest.raises(ValueError, match="not launched"):
                                p.record(stage, path)
                            continue
                        for launch in range(p.launches(stage, path)):
                            assert p.grid(stage, path, launch) == _expected_grid(
                                p.record(stage, path),
                                p.NC_pad,
                                N,
                                heads,
                                sm,
                                p.NSEG,
                                launch,
                            )
                    assert set(p.stages(path)) == {s for s, r in expected.items() if r}
                    for tables in (True, False):
                        assert p.launch_count(path, tables=tables) == (
                            _expected_launch_count(expected, seg, path, tables)
                        ), (path, tables)
                # These packs are far below the size rule (S < 384 * NSEG): the light forward saves no segment
                # states and the backward recomputes them.
                assert p.seg_save is False and p.light_path == "light"
                assert p.launch_count("bwd") == p.launch_count("light")
                if not seg:
                    with pytest.raises(ValueError, match="segmented route only"):
                        p.record("dhu", "light_states")
    assert cake_backend.plan(1, 1, 1, 1).NC_pad == 2
    for bad in ((0, 1, 1, 148), (1, 0, 1, 148), (1, 1, 0, 148), (1, 1, 1, 0)):
        with pytest.raises(ValueError):
            cake_backend.plan(*bad)
    with pytest.raises(ValueError, match="int32"):
        cake_backend.plan(64, 1, 2**31 // (D * BT), 148)
    with pytest.raises(ValueError, match="arch"):
        cake_backend.plan(64, 1, 1, 148, "sm_90a")


def test_launch_counts_of_the_paths() -> None:
    # 148 SMs, per architecture.  One document at 16 heads: on sm_100a the 32-column serial form (no dav: the fused
    # gradient hosts dAqk; on the saving forward the token-row pair wy_fwd_rows + fwdh_o_v2_rows, the same count)
    # whose light set carries w / u (no wy launch on the light path), on sm_103a / sm_107a a segmented row of 16
    # chains (nine segments per document); four documents at 16 heads: on sm_100a the same paired serial form, on
    # sm_103a / sm_107a a segmented row of 64 chains whose light set carries w / u (no wy launch on either light
    # path: one launch fewer there).  Four documents at 64 heads: the 64-column form (the fused recompute: no
    # WY launch on the light and replay paths; on sm_107a the saving forward's token-row pair wy_fwd_rows +
    # fwdh_o_v1_rows instead of the fused forward).  Segmented rows: four documents at 2 heads (eight chains: the
    # dv-accumulating adjoint, the fused saved-state pass 2; the overlap pair on the forward, the two-level carry
    # on the backward paths) and one document at 4 heads (four chains: the two-CTA adjoint);
    # no dav launch on either (the fused gradient hosts dAqk), no table launch on the forward and the replay (the
    # intra program writes the chunk tables and the block table itself).  The forward has no prep launch; the
    # backward of sm_103a has no gate epilogue launch (the fused GLM intra program); sm_100a / sm_107a launch it on
    # every backward path (the counts below are without it).
    serial_16 = {"fwd": 4, "fwd_nosave": 4, "light": 5, "tape": 5, "replay": 8}
    seg_16 = {
        "fwd": 5,
        "fwd_nosave": 5,
        "light": 10,
        "light_states": 7,
        "tape": 7,
        "replay": 11,
    }
    # four documents at 16 heads (64 chains) on sm_103a / sm_107a: the set carries w / u, no wy launch on the light
    # paths (one launch fewer than the one-document row there); on sm_100a the paired serial row again
    seg_16_wu = {
        "fwd": 5,
        "fwd_nosave": 5,
        "light": 9,
        "light_states": 7,
        "tape": 7,
        "replay": 11,
    }
    seg_small = dict(seg_16)
    counts_by_arch = {
        (1, 16): {"sm_100a": serial_16, "sm_103a": seg_16, "sm_107a": seg_16},
        (4, 16): {"sm_100a": serial_16, "sm_103a": seg_16_wu, "sm_107a": seg_16_wu},
        (4, 64): {
            "sm_100a": {"fwd": 3, "fwd_nosave": 3, "light": 5, "tape": 5, "replay": 7},
            "sm_103a": {"fwd": 3, "fwd_nosave": 3, "light": 5, "tape": 5, "replay": 7},
            "sm_107a": {"fwd": 4, "fwd_nosave": 4, "light": 5, "tape": 5, "replay": 7},
        },
        (4, 2): {arch: seg_small for arch in ARCHES},
        (1, 4): {arch: seg_small for arch in ARCHES},
    }
    for (n, h), by_arch in counts_by_arch.items():
        for arch, want in by_arch.items():
            p = cake_backend.plan(4096, n, h, 148, arch)
            # the gate epilogue is one more launch on every backward path of sm_100a / sm_107a
            gate = 0 if _gate_fused(arch) else 1
            expected = {
                path: count + (gate if path not in ("fwd", "fwd_nosave") else 0)
                for path, count in want.items()
            }
            # the saving forward and the forward without a saved set launch the same number of programs (the
            # token-row records replace the internal-row ones)
            assert {path: p.launch_count(path) for path in p.records} == expected, (
                n,
                h,
                arch,
            )
    # plan(arch=None) is the current device's (the first exported architecture's without one)
    for n, h in counts_by_arch:
        assert cake_backend.plan(4096, n, h, 148).records == (
            cake_backend.plan(4096, n, h, 148, _current_arch()).records
        )
    # the 16-head row of sm_100a: the light set carries w / u, so the light path launches no wy program (six launches:
    # the prep, the recompute, the adjoint, the fused gradient, GLM intra, the gate epilogue)
    p16 = cake_backend.plan(4096, 1, 16, 148, "sm_100a")
    assert p16.light_saves_wu and p16.records["light"]["wy"] is None
    assert p16.records["light"]["bwd_prep"] == "bwd_prep_g8_noqg_nokb"
    assert p16.launch_count("light") == 6 and p16.launch_count("tape") == 6
    assert p16.records["tape"]["wy"] == "wy_bwd"
    # a saved set whose tables the caller dropped adds the table launch to the saved-set paths (the fused table
    # program); the forward and the replay select no table record and the table-writing intra program
    for arch in ARCHES:
        gate = 0 if _gate_fused(arch) else 1
        p = cake_backend.plan(4096, 4, 2, 148, arch)
        assert p.launch_count("light", tables=False) == 11 + gate
        assert p.launch_count("tape", tables=False) == 8 + gate
        assert (
            p.launch_count("replay")
            == p.launch_count("replay", tables=False)
            == 11 + gate
        )
        for path in ("fwd", "fwd_nosave", "replay"):
            assert p.records[path]["table"] is None and not p.dispatched("table", path)
            assert p.records[path]["fwd_intra"] == "fwd_intra_v14"
        for path in ("light", "light_states", "tape"):
            assert p.records[path]["table"] == "table_v3"
        # the two-level carry replaces the chain carry launch for launch, on every architecture
        assert p.records["light"]["seg_carry"] == "seg_carry_v8"
        assert cake_backend.plan(4096, 1, 16, 148, arch).records["fwd"][
            "fwd_intra"
        ] == ("fwd_intra" if arch == "sm_100a" else "fwd_intra_v14")
        # above the layout-table document bound the segmented route launches the block table on its own
        wide = cake_backend.plan(4096, 300, 2, 148, arch)
        assert wide.records["fwd"]["table"] == "table" and wide.dispatched(
            "seg_table", "fwd"
        )
        assert wide.launch_count("fwd") == wide.launch_count("fwd_nosave") == 7
        assert wide.launch_count("replay") == 13 + gate
        # the backward from this plan's own light set: the states-saved path above the size rule
        assert p.launch_count("bwd") == p.launch_count("light") == 10 + gate
        tall = cake_backend.plan(384 * p.NSEG, 4, 2, 148, arch)
        assert tall.light_path == "light_states" and tall.NSEG == p.NSEG
        assert tall.launch_count("bwd") == 7 + gate
        assert tall.launch_count("bwd", tables=False) == 8 + gate
    with pytest.raises(ValueError, match="path must be one of"):
        p.launch_count("sideways")


def test_saved_set_version_4_fields_and_the_autograd_extras() -> None:
    # The light set is version 4: the fold's qg always, hseg / seg_state under the size rule, w / u on their
    # rows; each optional on its own, so the autograd node carries the names of the present extras (the absent
    # ones are no longer a suffix of a fixed order).  The tape set stays version 2 and carries none of them.
    assert cake_backend.SAVED_VERSION == 4 and cake_backend.SAVED_VERSION_TAPE == 2
    assert [f.name for f in dataclasses.fields(CakeKDANondetChunkSaved)] == [
        "version",
        "Aqk",
        "Akk",
        "tables",
        "S",
        "H",
        "num_docs",
        "scale",
        "h",
        "v_new",
        "seg_m",
        "hseg",
        "seg_state",
        "qg",
        "w",
        "u",
        "gate",
        "lower_bound",
        "qk_rstd",
        "qk_l2norm",
    ]
    for name in ("h", "v_new", "seg_m", "hseg", "seg_state", "qg", "w", "u", "qk_rstd"):
        assert CakeKDANondetChunkSaved.__dataclass_fields__[name].default is None, name
    # the deterministic package's gate-mode fields (#7816), at its defaults: this package's forward writes the
    # explicit gate and its backward refuses a raw-mode set
    assert CakeKDANondetChunkSaved.__dataclass_fields__["gate"].default == "g"
    assert CakeKDANondetChunkSaved.__dataclass_fields__["lower_bound"].default is None
    # the deterministic package's in-kernel Q/K L2 norm fields (#7817), at its defaults: a forward with the flag
    # writes the reciprocal norms (qk_rstd, FP32 [2, S, H]) and the mode on, as the deterministic package's does
    assert CakeKDANondetChunkSaved.__dataclass_fields__["qk_l2norm"].default is False
    # the deterministic package's fields, no more: the set records no mode (its type is the package's; each
    # package's backward reads its own type only and refuses the other's)
    assert [f.name for f in dataclasses.fields(CakeKDANondetChunkSaved)] == [
        f.name for f in dataclasses.fields(det_backend.CakeKDADetChunkSaved)
    ]
    # the in-kernel Q/K L2 norm's reciprocal norms ride last in both versions (the deterministic package's order)
    assert cake_backend._EXTRA_FIELDS == {
        4: ("qg", "hseg", "seg_state", "w", "u", "qk_rstd"),
        2: ("h", "v_new", "seg_m", "qk_rstd"),
    }
    assert cake_backend._EXTRA_FIELDS == det_backend._EXTRA_FIELDS
    t = torch.empty(0)
    light = CakeKDANondetChunkSaved(4, t, t, None, 1, 1, 1, SCALE, qg=t)
    tensors, names = cake_backend._saved_extra(light)
    assert names == ("qg",) and len(tensors) == 1 and tensors[0] is t
    full = CakeKDANondetChunkSaved(
        4, t, t, None, 1, 1, 1, SCALE, hseg=t, seg_state=t, qg=t, w=t, u=t
    )
    tensors, names = cake_backend._saved_extra(full)
    assert names == ("qg", "hseg", "seg_state", "w", "u") and len(tensors) == 5
    wu_only = CakeKDANondetChunkSaved(4, t, t, None, 1, 1, 1, SCALE, qg=t, w=t, u=t)
    assert cake_backend._saved_extra(wu_only)[1] == ("qg", "w", "u")
    tape = CakeKDANondetChunkSaved(2, t, t, None, 1, 1, 1, SCALE, h=t, v_new=t)
    assert cake_backend._saved_extra(tape)[1] == ("h", "v_new")
    tape_seg = CakeKDANondetChunkSaved(
        2, t, t, None, 1, 1, 1, SCALE, h=t, v_new=t, seg_m=t
    )
    assert cake_backend._saved_extra(tape_seg)[1] == ("h", "v_new", "seg_m")
    r = torch.empty(2, 1, 1)
    l2n = CakeKDANondetChunkSaved(
        4, t, t, None, 1, 1, 1, SCALE, qg=t, qk_rstd=r, qk_l2norm=True
    )
    tensors, names = cake_backend._saved_extra(l2n)
    assert names == ("qg", "qk_rstd") and tensors[-1] is r
    l2n_tape = dataclasses.replace(tape, qk_rstd=r, qk_l2norm=True)
    assert cake_backend._saved_extra(l2n_tape)[1] == ("h", "v_new", "qk_rstd")


def test_the_light_set_saves_qg_everywhere_and_w_u_where_the_prep_writes_no_kb() -> (
    None
):
    # Plan.light_saves_qg / light_saves_wu read the light path's prep record: no-qg forms on every row (the set
    # carries the forward's qg), the no-kb form (w / u in the set, no wy launch on either light path) on the serial
    # 16-head rows of sm_100a whose 32-column form fits one wave and on the segmented 16-head rows of sm_103a /
    # sm_107a above 32 chains; the byte query adds the tensors accordingly.
    S = 4096
    seen_wu = seen_seg_wu = 0
    for arch, n, h, sm in itertools.product(
        ARCHES, (1, 2, 7, 300), (1, 2, 4, 8, 12, 16, 24, 32, 64), (132, 148, 212)
    ):
        p = cake_backend.plan(S, n, h, sm, arch)
        wu = _wu_saved(n, h, sm, arch)
        seen_wu += wu
        seen_seg_wu += wu and p.recur == "seg"
        assert p.light_saves_qg is True
        assert p.light_saves_wu is wu, (arch, n, h, sm)
        prep = p.records["light"]["bwd_prep"]
        assert prep == ("bwd_prep_g8_noqg_nokb" if wu else "bwd_prep_g8_noqg"), (
            arch,
            n,
            h,
            sm,
        )
        if "light_states" in p.records:
            assert p.records["light_states"]["bwd_prep"] == prep
        assert p.records["tape"]["bwd_prep"] == "bwd_prep_g8"
        assert p.records["replay"]["bwd_prep"] == "bwd_prep_g8_replay"
        if wu:
            assert h == WU_SAVED_H
            if p.recur == "serial":
                assert arch in WU_SAVED_ARCHES and n * h * 4 <= 4 * sm
            else:
                assert arch in WU_SEG_SAVED_ARCHES and n * h > WU_SEG_MIN_CHAINS
                assert p.records["light_states"]["wy"] is None
            assert p.records["light"]["wy"] is None
            assert p.records["tape"]["wy"] == "wy_bwd"
            assert p.records["replay"]["wy"] == "wy_bwd_replay"
        tables = cake_backend.tables_bytes(S, n, h, sm_count=sm, arch=arch)
        states = 0
        if cake_backend.seg_states_saved(S, p.NSEG):
            states = 3 * p.NSEG * h * D * D * 4
        want = 256 * S * h + tables + p.R * h * D * 2 + states
        if wu:
            want += 2 * p.R * h * D * 2
        assert cake_backend.saved_bytes(S, h, n, sm_count=sm, arch=arch) == want, (
            arch,
            n,
            h,
            sm,
        )
    assert seen_wu > seen_seg_wu > 0
    # 300 documents at 16 heads on sm_100a need 19200 slots, above 4 * 148: the 64-column form keeps the wy launch
    p = cake_backend.plan(S, 300, 16, 148, "sm_100a")
    assert p.light_saves_wu is False and p.records["light"]["wy"] == "wy_bwd"
    assert p.records["light"]["bwd_prep"] == "bwd_prep_g8_noqg"
    for arch in WU_SEG_SAVED_ARCHES:
        # two documents at 16 heads are 32 chains, not above: the fused saved-state pass 2 keeps forming the WY
        # representation (its w^T adjoint behind it) and the light path keeps the wy launch
        p = cake_backend.plan(S, 2, 16, 148, arch)
        assert p.recur == "seg" and p.light_saves_wu is False
        assert p.records["light"]["wy"] == "wy_bwd"
        assert p.records["light"]["bwd_prep"] == "bwd_prep_g8_noqg"
        assert p.records["light_states"]["wy"] is None
        assert p.records["light_states"]["fwdh_p3"] == "fwdh_p3_wy"
        assert p.records["light_states"]["dhu"].endswith("_wt")
        # three documents are 48 chains: the set carries w / u, neither light path launches wy, pass 2 and the
        # adjoint take their plain forms (no w^T tile: the saved w is read as it is)
        p = cake_backend.plan(S, 3, 16, 148, arch)
        assert p.recur == "seg" and p.light_saves_wu is True
        for path in ("light", "light_states"):
            assert p.records[path]["bwd_prep"] == "bwd_prep_g8_noqg_nokb"
            assert p.records[path]["wy"] is None
            assert p.records[path]["fwdh_p3"] == "fwdh_p3"
            assert not p.records[path]["dhu"].endswith("_wt")
        assert p.records["light"]["fwdh"] == "fwdh_v3"
        assert p.records["light_states"]["fwdh"] is None


def test_gate_fused_architectures_lay_the_arena_out_without_the_epilogue_buffers() -> (
    None
):
    # glm_intra_v14 / glm_intra_v22 (sm_103a) and glm_intra_v18 / glm_intra_v23 (sm_107a) write the gate gradients
    # themselves: no gate_epilogue record, no dg_int / db_int in the layouts.  The default layouts keep their bytes (the two shared slots with h / do_int); the lean single-pass
    # layout loses the 4-byte db_int slot; the multi-pass per-pass set loses both buffers.  A query without an
    # architecture takes sm_100a's answer: the epilogue's buffers stay.
    assert cake_backend._gate_fused(None) is _gate_fused(None) is False
    lean = cake_backend.CakeKDANondetArena(lean=True)
    for arch in ARCHES:
        fused = _gate_fused(arch)
        assert cake_backend._gate_fused(arch) is fused
        for lengths, heads in (((63, 1, 65), 2), ((129,), 16), ((2000,) * 3, 64)):
            S, N = sum(lengths), len(lengths)
            p = cake_backend.plan(S, N, heads, 148, arch)
            assert p.gate_fused is fused
            assert p.dispatched("gate_epilogue", "light") is not fused
            # the GLM intra program of the light path: the S1 one on the S1 cells, the deterministic one elsewhere,
            # gate-fused alike on sm_103a / sm_107a
            assert p.records["light"]["glm_intra"] == (
                S1_GLM[arch] if _s1(arch, "light", heads) else DET_GLM[arch]
            )
            R, H, NC, NSEG = p.R, p.H, p.NC_pad, p.NSEG
            for mode in cake_backend.MODES:
                assert (
                    p.workspace_bytes(mode)
                    == cake_backend._mode_bytes(R, H, NC, NSEG, mode)
                    == cake_backend._mode_bytes(R, H, NC, NSEG, mode, 1, True)
                ), (arch, mode)
                names = {r.name for r in p.layout(mode)}
                assert bool(set(GATE_BUFFERS) & names) is (
                    not fused and not mode.startswith("fwd")
                ), (arch, mode)
            for mode in cake_backend.LEAN_MODES:
                plain = cake_backend._mode_bytes(R, H, NC, NSEG, mode, 1, False)
                dropped = cake_backend._mode_bytes(R, H, NC, NSEG, mode, 1, True)
                assert plain - dropped == 4 * R * H
                assert p.workspace_bytes(mode) == (dropped if fused else plain)
                for t in (2, 3, 4):
                    names = {r.name for r in p.layout(mode, t)}
                    assert names & PASS_NAMES == {
                        name for name, *_ in _pass_occupants(fused)
                    }, (arch, mode, t)
            q = {"sm_count": 148, "arch": arch}
            assert cake_backend.workspace_bytes(
                S, N, heads, arena=lean, **q
            ) == p.workspace_bytes("bwd_lean")
            assert cake_backend.memory_report(S, N, heads, **q)["gate_fused"] is fused


def test_seg_state_rule_is_a_pure_function_of_the_scalars() -> None:
    rule, k = cake_backend.seg_states_saved, cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG
    assert k == 384
    assert cake_backend.SAVED_VERSION == 4 and cake_backend.SAVED_VERSION_TAPE == 2
    for nseg in (1, 64, 153, 256):
        assert rule(k * nseg, nseg) is True and rule(k * nseg - 1, nseg) is False
    assert rule(10**9, 0) is False  # the serial route has no segment states
    # the sm_100a rule (a plan without an arch takes the current device's)
    a0 = cake_jit.ARCHES[0]
    for n, h, sm in ((1, 1, 148), (4, 4, 152), (3, 8, 148), (4, 1, 152)):
        nseg = cake_backend.plan(1, n, h, sm, a0).NSEG  # independent of S
        assert (
            nseg
            == cake_backend.plan(k * nseg, n, h, sm, a0).NSEG
            == _expected_segments(n, h, sm, a0)[2]
        )
        assert cake_backend.plan(k * nseg, n, h, sm, a0).seg_save is True
        assert cake_backend.plan(k * nseg - 1, n, h, sm, a0).seg_save is False
    # the serial route
    assert cake_backend.plan(10**6, 1, 16, 148, a0).seg_save is False


def test_byte_queries_follow_the_arena_layout() -> None:
    GiB = 2**30
    # three 16-head rows: the paired serial rows of sm_100a (their light set carries w / u), segmented elsewhere,
    # where the three-document row (48 chains) carries w / u too
    rows = SHAPES + [
        ((239718 // 3,) * 3, 4),
        ((261393 // 4,) * 4, 64),
        ((129,), 16),
        ((2000, 2000), 16),
        ((2000, 2000, 2000), 16),
    ]
    for (lengths, heads), arch in itertools.product(rows, ARCHES):
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, 148, arch)
        q = {"sm_count": 148, "arch": arch}  # the queries of this architecture
        assert tuple(cake_backend.MODES) == (
            "fwd",
            "fwd_tape",
            "bwd",
            "bwd_tape",
            "replay",
            "replay_tape",
        )
        for mode in cake_backend.MODES:
            regions = p.layout(mode)
            assert all(r.offset % cake_backend.ALIGN == 0 for r in regions), mode
            assert p.workspace_bytes(mode) == max(r.offset + r.nbytes for r in regions)
            names = {r.name for r in regions}
            seg_names = {"seg_b", "seg_m", "hseg"} | (
                {"dseg", "dhseg"} if not mode.startswith("fwd") else set()
            )
            assert (seg_names <= names) if p.recur == "seg" else not (seg_names & names)
            assert ("akk_int" in names) == (not mode.startswith("bwd")), mode
        # the saved-set paths' layouts drop the akk_int buffer: never larger than the replay's
        assert p.workspace_bytes("bwd") <= p.workspace_bytes("replay")
        assert p.workspace_bytes("bwd_tape") <= p.workspace_bytes("replay_tape")
        for backward in (False, True):
            for save_mode in cake_backend.SAVE_MODES:
                tape = save_mode == "tape"
                if backward:
                    want = max(
                        p.workspace_bytes("bwd_tape" if tape else "bwd"),
                        p.workspace_bytes("replay_tape" if tape else "replay"),
                    )
                else:
                    want = p.workspace_bytes("fwd_tape" if tape else "fwd")
                assert (
                    cake_backend.workspace_bytes(
                        S,
                        N,
                        heads,
                        backward=backward,
                        save_mode=save_mode,
                        **q,
                    )
                    == want
                    == p.arena_bytes(backward=backward, save_mode=save_mode)
                )
        assert cake_backend.workspace_bytes(
            S, N, heads, backward=False, **q
        ) <= cake_backend.workspace_bytes(S, N, heads, **q)
        assert (
            cake_backend.saved_bytes(S, heads)
            == 2 * S * heads * BT * 2
            == 256 * S * heads
        )
        tables = cake_backend.tables_bytes(S, N, heads, **q)
        assert tables == p.table_words() * 4 and tables % cake_backend.ALIGN == 0
        layout = p.table_layout()
        assert [name for name, _off, _words in layout] == list(
            cake_backend.TABLE_NAMES[: 5 if p.recur == "seg" else 4]
        )
        if p.recur == "seg":
            assert layout[-1][2] == p.NSEG + 2 * N + 1
        # light (version 4): the fold's qg BF16 [R, H, 128] on every row; the WY representation w / u (two more
        # such tensors) on the paired serial 16-head rows of sm_100a and on the segmented 16-head rows of sm_103a /
        # sm_107a above 32 chains; the segment-boundary states (three FP32
        # state blocks) when the size rule holds: of these rows only the large segmented one (H = 4), not the
        # small packs or the serial H = 64
        states = (
            3 * p.NSEG * heads * D * D * 4
            if cake_backend.seg_states_saved(S, p.NSEG)
            else 0
        )
        assert p.seg_save == (S >= 384 * p.NSEG > 0)
        assert (states > 0) == (S > 200_000 and heads == 4)
        qg = p.R * heads * D * 2
        wu = 2 * p.R * heads * D * 2 if _wu_saved(N, heads, 148, arch) else 0
        wu_row = heads == 16 and (
            arch in WU_SAVED_ARCHES or N * heads > WU_SEG_MIN_CHAINS
        )
        assert (wu > 0) == wu_row, (lengths, arch)
        assert p.light_saves_qg is True and p.light_saves_wu is (wu > 0)
        assert (
            cake_backend.saved_bytes(S, heads, N, **q)
            == 256 * S * heads + tables + qg + wu + states
        )
        tape_bytes = cake_backend.saved_bytes(S, heads, N, save_mode="tape", **q)
        assert tape_bytes == (
            256 * S * heads
            + tables
            + p.NC_pad * heads * D * D * 2
            + p.R * heads * D * 2
            + 2 * p.NSEG * heads * D * D * 4
        )
        assert cake_backend.output_bytes(S, heads, backward=False) == S * heads * D * 2
        assert cake_backend.output_bytes(S, heads, backward=True) == S * heads * (
            3 * D * 2 + 2 + D * 4
        )
        report = cake_backend.memory_report(S, N, heads, **q)
        assert (report["recur"], report["segments"], report["num_segments"]) == (
            p.recur,
            p.P,
            p.NSEG,
        )
        assert report["arena_bwd"] == max(
            p.workspace_bytes("bwd"), p.workspace_bytes("replay")
        )
        assert report["arena_bwd_layout_saved_path"] == p.workspace_bytes("bwd")
        assert report["arena_bwd_layout_replay_path"] == p.workspace_bytes("replay")
        assert report["saved"] == 256 * S * heads + tables + qg + wu + states
        assert report["seg_states_saved"] == (states > 0)
        assert report["saved_seg_state_bytes"] == states
        assert report["gate_fused"] is _gate_fused(arch) is p.gate_fused
        tape_report = cake_backend.memory_report(S, N, heads, "tape", **q)
        assert tape_report["saved"] == tape_bytes
        assert tape_report["saved_seg_state_bytes"] == 2 * p.NSEG * heads * D * D * 4
        assert tape_report["gate_fused"] is p.gate_fused
        assert report["bwd_total"] == report["arena_bwd"] + report["out_bwd"]
        assert report["arch"] == arch
        # without an architecture the queries take the current device's (None without a device): the bytes of
        # the explicit query of that architecture
        host = _query_arch()
        assert cake_backend.workspace_bytes(S, N, heads, sm_count=148) == (
            cake_backend.workspace_bytes(S, N, heads, sm_count=148, arch=host)
        )
        assert cake_backend.saved_bytes(S, heads, N, sm_count=148) == (
            cake_backend.saved_bytes(S, heads, N, sm_count=148, arch=host)
        )
        host_report = cake_backend.memory_report(S, N, heads, sm_count=148)
        assert host_report["arch"] == host
        # a query without a device's architecture takes the first exported one's gate fusion and w / u rows
        assert host_report["gate_fused"] is _gate_fused(host)
        if (
            S > 200_000
        ):  # the largest contract rows fit one 283 GiB device with room for two arenas (parity)
            assert report["arena_bwd"] < 85 * GiB
    with pytest.raises(ValueError, match="needs num_docs"):
        cake_backend.saved_bytes(64, 1, save_mode="tape")
    with pytest.raises(ValueError, match="save_mode"):
        cake_backend.saved_bytes(64, 1, save_mode="full")
    with pytest.raises(ValueError, match="save_mode"):
        cake_backend.workspace_bytes(64, 1, 1, save_mode="full", sm_count=148)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._layout(128, 1, 2, 0, "sideways")
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend.lifetime_table("sideways")


def _gate_fused_slots(slots):
    """The layout's slots without the gate epilogue's buffers (an emptied slot leaves too), as the fused GLM
    intra program's architectures lay the arena out."""
    kept = (
        tuple(o for o in occupants if o[0] not in GATE_BUFFERS) for occupants in slots
    )
    return tuple(occupants for occupants in kept if occupants)


GATE_BUFFERS = ("dg_int", "db_int")


def test_lifetime_table_lists_every_occupant() -> None:
    assert cake_backend._GATE_BUFFERS == GATE_BUFFERS
    for mode, slots in cake_backend.SLOTS.items():
        # without a plan the table keeps the gate epilogue's buffers (gate_fused=None -> False)
        for rows, expected in (
            (cake_backend.lifetime_table(mode), slots),
            (cake_backend.lifetime_table(mode, gate_fused=False), slots),
            (
                cake_backend.lifetime_table(mode, gate_fused=True),
                _gate_fused_slots(slots),
            ),
        ):
            assert len(rows) == sum(len(occupants) for occupants in expected)
            assert {(row["slot"], row["buffer"]) for row in rows} == {
                (slot, name)
                for slot, occupants in enumerate(expected)
                for name, *_ in occupants
            }
            for row in rows:
                assert row["first_write"] <= row["last_read"]
                assert row["unit"] in ("R*H", "seg_block")
        names = {name for occupants in slots for name, *_ in occupants}
        assert (set(GATE_BUFFERS) <= names) is not mode.startswith("fwd"), mode
    # a plan's table takes its architecture's fusion
    for arch in ARCHES:
        p = cake_backend.plan(4096, 3, 2, 148, arch)
        for mode in ("bwd", "replay", "bwd_tape", "replay_tape"):
            names = {row["buffer"] for row in cake_backend.lifetime_table(mode, plan=p)}
            assert (set(GATE_BUFFERS) & names) == (
                set() if _gate_fused(arch) else set(GATE_BUFFERS)
            ), (arch, mode)
            assert names == {r.name for r in p.layout(mode)}


# --------------------------------------------------------------------------- the lower-memory arena option (no device)

# bytes per internal token-head of the token kinds (a tape entry of 32768 B per chunk-head over the chunk's 64 rows)
KIND_BYTES_PER_TH = {
    "F32": 512,
    "B16": 256,
    "A16": 128,
    "A32": 256,
    "V32": 4,
    "TAPE": 512,
}
PASS_NAMES = frozenset(name for name, *_ in cake_backend._PASS_OCCUPANTS)


def _pass_occupants(gate_fused: bool):
    """The per-pass set of a lean layout: the fused GLM intra program's architectures drop the gate epilogue's
    ``dg_int`` (F32p) / ``db_int`` (V32p)."""
    return tuple(
        o
        for o in cake_backend._PASS_OCCUPANTS
        if not (gate_fused and o[0] in GATE_BUFFERS)
    )


def _pass_bytes_per_th(gate_fused: bool) -> int:
    """Bytes of the per-pass set per internal token-head of one tail range: 2568, fused 2052."""
    return sum(
        KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
        for _name, kind, _sub, _life in _pass_occupants(gate_fused)
    )


# the lean layouts' bytes per internal token-head: one pass (the saved-set layout without the replay's dout copy;
# the fused architectures drop the 4-byte db_int slot and the dg_int under the h tape), from two passes on the
# whole-sequence set
LEAN_SINGLE_BYTES = {False: {False: 4108, True: 3340}, True: {False: 4104, True: 3336}}
LEAN_WHOLE_BYTES = {False: 2820, True: 2052}
# what an in-kernel Q/K L2 norm mode adds to the whole-sequence set from two passes on: the dAqk the dav_l2n*
# records write on every row (A32), whole instead of per pass (3076 B per internal token-head, tape 2308)
L2N_LEAN_DAQK_BYTES = 256
# rows of H 9-15 (the serial rows whose flag-off backward launches the separate dav program, which a lean arena
# refuses; in an l2n mode a lean arena serves them)
L2N_LEAN_ROWS = [((2000, 2000, 2000), 12), ((129, 300), 9), ((63, 1, 65), 15)]
SEG_NAMES = frozenset(("seg_b", "seg_m", "hseg", "dseg", "dhseg"))
# the tail stages' operands the tail split hands over per range: offset views of the whole-sequence buffers, the
# range's table slices and count, the per-pass buffers; everything else they bind is addressed through the
# absolute chunk_bos (the token tensors) or is a scalar of the plan
TAIL_OFFSET_VIEWS = frozenset(("gk", "v_new", "dv2", "beta_int", "h2d", "dh2d"))
TAIL_TABLES = frozenset(
    ("chunk_bos", "chunk_len", "seq_chunk_start", "num_seqs", "num_chunks_pad")
)
TAIL_TOKEN_ROWS = frozenset(
    (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "dout",
        "aqk_src",
        "akk_src",
        "dq",
        "dk",
        "dv",
        "dg",
        "dbeta",
        "cu_seqlens",
    )
)
TAIL_SCALARS = frozenset(
    (
        "scale",
        "num_heads",
        "group",
        "num_segments",
        "segments",
        "seg_arg",
        "proportional",
        "num_pairs",
    )
)
# buffers a tail kernel must not bind whole (their rows would be the whole sequence's, not the range's)
TAIL_FORBIDDEN = frozenset(
    (
        "h",
        "dh",
        "seq_order",
        "seg_table",
        "w",
        "w_t",
        "w_tk",
        "u",
        "kb",
        "qg",
        "kg",
        "akk_int",
        "aqk_int",
        "do_int",
        "dv1",
    )
)


# every arena buffer any layout table declares, plus the derived views the host hands over
ARENA_BUFFER_NAMES = frozenset(
    name for slots in cake_backend.SLOTS.values() for slot in slots for name, *_ in slot
) | frozenset(("seg_state", "w_t", "w_tk"))


def _slot_bytes(slots, R: int, H: int, NSEG: int, R_b: int) -> int:
    """An independent spelling of the layout's packing: per slot the farthest byte any occupant reaches (its
    sub-offset in the unit of its kind plus its extent), rounded up to the alignment, summed over the slots."""
    block = NSEG * H * D * D * 4
    total = 0
    for occupants in slots:
        end = 0
        for _name, kind, sub, _life in occupants:
            if kind in cake_backend.SEG_KINDS:
                if NSEG == 0:
                    continue
                unit = nbytes = block
            elif kind in cake_backend.PASS_KINDS:
                unit = R_b * H
                nbytes = unit * KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
            else:
                unit = R * H
                nbytes = unit * KIND_BYTES_PER_TH[kind]
            end = max(end, sub * unit + nbytes)
        total += -(-end // cake_backend.ALIGN) * cake_backend.ALIGN
    return total


def _expected_lean_bytes(
    tape: bool,
    t: int,
    R: int,
    H: int,
    NC: int,
    NSEG: int,
    gate_fused: bool = False,
    l2n: bool = False,
) -> int:
    """The lean layout's bytes from the documented rule: one pass is the saved-set layout without the replay's
    dout copy (4108 B per internal token-head, tape 3340; the fused architectures 4104 / 3336: no db_int slot);
    from two passes on the whole-sequence set (2820 B per internal token-head, tape 2052) plus the per-pass set
    (without dg_int / db_int where the gate epilogue is fused) packed first-fit, in order, into the dead ``qg`` /
    ``kg`` (512) and ``w`` (256) slots, the rest into an overflow slot shared with the segmented route's three
    state blocks.  In an in-kernel Q/K L2 norm mode (``l2n``) the whole-sequence set holds ``dAqk`` too
    (:data:`L2N_LEAN_DAQK_BYTES` more: 3076, tape 2308) and the per-pass set packs none; one pass is unchanged."""
    block = NSEG * H * D * D * 4
    if t == 1:
        single = LEAN_SINGLE_BYTES[gate_fused][tape]
        return single * R * H + max(0, 3 * block - 256 * R * H)
    R_b = BT * -(-NC // t)
    capacity = {"qg": 512 * R * H, "w": 256 * R * H}
    used = {"qg": 0, "w": 0}
    overflow = 0
    for name, kind, _sub, _life in _pass_occupants(gate_fused):
        if l2n and name == "dAqk":
            continue  # whole-sequence in the l2n modes
        nbytes = R_b * H * KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
        for host in ("qg", "w"):
            if used[host] + nbytes <= capacity[host]:
                used[host] += nbytes
                break
        else:
            overflow += nbytes
    extra = max(overflow, 3 * block) if (overflow or NSEG) else 0
    whole = LEAN_WHOLE_BYTES[tape] + (L2N_LEAN_DAQK_BYTES if l2n else 0)
    return whole * R * H + extra


def _lean_rows():
    return SHAPES + [
        ((239718 // 3,) * 3, 4),
        ((261393 // 4,) * 4, 64),
        ((2000,) * 300, 1),
        ((129,), 16),  # the paired serial row of sm_100a whose light set carries w / u
    ]


def test_arena_option_validation() -> None:
    CakeKDANondetArena = cake_backend.CakeKDANondetArena
    assert (
        CakeKDANondetArena() == CakeKDANondetArena(lean=False, tail_passes=1)
        and not CakeKDANondetArena().lean
    )
    assert CakeKDANondetArena(
        tail_passes=2
    ).lean  # the tail split lives in the lean layouts
    assert CakeKDANondetArena(lean=True).tail_passes == 1
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="tail_passes"):
            CakeKDANondetArena(tail_passes=bad)
    off, lean = CakeKDANondetArena(), CakeKDANondetArena(lean=True, tail_passes=4)
    for base in cake_backend.MODES:
        assert off.mode(base) == base
    assert lean.mode("bwd") == "bwd_lean" and lean.mode("bwd_tape") == "bwd_tape_lean"
    assert lean.mode("fwd") == "fwd" and lean.mode("fwd_tape") == "fwd_tape"
    assert lean.mode("bwd_lean") == "bwd_lean"
    for base in ("replay", "replay_tape"):
        with pytest.raises(ValueError, match="lean=False"):
            lean.mode(base)
    assert tuple(cake_backend.LEAN_MODES) == ("bwd_lean", "bwd_tape_lean")
    assert not (set(cake_backend.LEAN_MODES) & set(cake_backend.MODES))
    assert cake_backend.PASS_KINDS == {"F32p": "F32", "A32p": "A32", "V32p": "V32"}
    with pytest.raises(ValueError, match="tail_passes"):
        cake_backend._slots_of("bwd", 2, 128, 1, 2, 0)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._slots_of("sideways", 1, 128, 1, 2, 0)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._layout(128, 1, 2, 0, "bwd_sideways", 2)
    for bad in ("full", "lean"):
        with pytest.raises(ValueError, match="save_mode"):
            cake_backend.workspace_bytes(
                64, 1, 1, save_mode=bad, sm_count=148, arena=lean
            )


def test_tail_ranges_cover_the_chunk_bound() -> None:
    ranges = cake_backend._tail_ranges
    assert ranges(10, 1) == ((0, 10),)
    assert ranges(10, 2) == ((0, 5), (5, 5))
    assert ranges(10, 3) == ((0, 4), (4, 4), (8, 2))
    assert ranges(10, 4) == ((0, 3), (3, 3), (6, 3), (9, 1))
    assert ranges(4, 3) == (
        (0, 2),
        (2, 2),
    )  # a clipped tail that is empty: fewer ranges than passes
    assert ranges(2, 8) == ((0, 1), (1, 1))
    for nc in (2, 4, 6, 8, 10, 64, 4086, 4088):
        for passes in (1, 2, 3, 4, 5, 8, 16):
            rs = ranges(nc, passes)
            length = -(-nc // passes)
            assert 1 <= len(rs) <= passes and all(0 < ln <= length for _c0, ln in rs)
            assert [c0 for c0, _ln in rs] == [i * length for i in range(len(rs))]
            assert sum(ln for _c0, ln in rs) == nc and rs[-1][0] + rs[-1][1] == nc


def test_lean_arena_refuses_the_rows_with_a_separate_dav_program() -> None:
    lean = cake_backend.CakeKDANondetArena(lean=True)
    for arch in ARCHES:
        for H in range(1, 65):
            p = cake_backend.plan(2000, 3, H, 148, arch)
            dav = any(
                p.dispatched("dav", path)
                for path in p.records
                if path not in ("fwd", "fwd_nosave")
            )
            assert dav == (9 <= H <= 15), (arch, H)  # the serial rows below 16 heads
            if dav:
                with pytest.raises(ValueError, match="dav"):
                    p.check_arena(lean)
                with pytest.raises(ValueError, match="dav"):
                    p.arena_bytes(arena=lean)
                with pytest.raises(ValueError, match="dav"):
                    cake_backend.workspace_bytes(
                        2000, 3, H, sm_count=148, arch=arch, arena=lean
                    )
                with pytest.raises(ValueError, match="dav"):
                    cake_backend.memory_report(
                        2000, 3, H, sm_count=148, arch=arch, arena=lean
                    )
                # the forward layout is the same under every option: no refusal there
                assert cake_backend.workspace_bytes(
                    2000, 3, H, backward=False, sm_count=148, arch=arch, arena=lean
                ) == p.workspace_bytes("fwd")
            else:
                p.check_arena(lean)
                assert p.arena_bytes(arena=lean) == p.workspace_bytes("bwd_lean")
            p.check_arena(
                cake_backend.CakeKDANondetArena()
            )  # the default serves every row


def test_lean_layouts_follow_the_byte_formulas() -> None:
    for (lengths, heads), arch in itertools.product(_lean_rows(), ARCHES):
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, 148, arch)
        q = {"sm_count": 148, "arch": arch}
        R, H, NC, NSEG = p.R, p.H, p.NC_pad, p.NSEG
        block = NSEG * H * D * D * 4
        blocks_over = max(0, 3 * block - 256 * R * H)
        fused = _gate_fused(arch)
        assert p.gate_fused is fused
        pass_names = frozenset(name for name, *_ in _pass_occupants(fused))
        # the default layouts' constants: the saved-set layouts and the replay layouts (the arena of a default
        # backward workspace is the replay's); the gate epilogue's buffers share their slots with h / do_int, so
        # the fused architectures' default layouts are no smaller
        assert p.workspace_bytes("bwd") == 4360 * R * H + blocks_over
        assert p.workspace_bytes("replay") == 4488 * R * H + blocks_over
        assert p.workspace_bytes("bwd_tape") == 3592 * R * H + blocks_over
        assert p.workspace_bytes("replay_tape") == 3720 * R * H + blocks_over
        # the forward layouts (the README's constants); a lean backward workspace serves the forward: its layout
        # is the smaller one under every option
        assert p.workspace_bytes("fwd") == 2560 * R * H + max(
            0, 3 * block - 128 * R * H
        )
        assert p.workspace_bytes("fwd_tape") == 2048 * R * H + max(
            0, 3 * block - 128 * R * H
        )
        assert p.workspace_bytes("fwd") <= min(
            p.workspace_bytes("bwd_lean", t) for t in (1, 2, 3, 4, 8)
        )
        assert p.workspace_bytes("fwd_tape") <= min(
            p.workspace_bytes("bwd_tape_lean", t) for t in (1, 2, 3, 4, 8)
        )
        for tape in (False, True):
            mode = "bwd_tape_lean" if tape else "bwd_lean"
            save_mode = "tape" if tape else "light"
            whole = LEAN_WHOLE_BYTES[tape]
            default = cake_backend.memory_report(S, N, heads, save_mode, **q)
            assert default["gate_fused"] is fused
            assert default["lean"] is False and default["tail_passes"] == 1
            assert (
                default["arena_bwd_configured"]
                == default["arena_bwd_default"]
                == default["arena_bwd"]
                == p.workspace_bytes("replay_tape" if tape else "replay")
            )
            assert default["arena_fwd_configured"] == default["arena_fwd"]
            assert default["arena_floor"] == whole * R * H + 3 * block
            for t in (1, 2, 3, 4, 8):
                arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
                regions = p.layout(mode, t)
                R_b = BT * -(-NC // t)
                assert all(r.offset % cake_backend.ALIGN == 0 for r in regions), (
                    mode,
                    t,
                )
                want = p.workspace_bytes(mode, t)
                assert want == max(r.offset + r.nbytes for r in regions)
                slots = cake_backend._slots_of(mode, t, R, H, NC, NSEG)
                if fused:
                    slots = _gate_fused_slots(slots)
                assert want == _slot_bytes(slots, R, H, NSEG, R_b), (mode, t)
                names = {r.name for r in regions}
                # no replay-only buffers (one pass keeps the saved-set layout's own aqk_int rows); no gate
                # epilogue buffers on the architectures that fuse it
                assert not ({"do_int", "akk_int"} & names), (mode, t)
                assert (t == 1) or "aqk_int" not in names, (mode, t)
                assert bool(set(GATE_BUFFERS) & names) is not fused, (mode, t)
                for r in regions:
                    if (
                        r.name in PASS_NAMES and t > 1
                    ):  # one pass keeps the whole-sequence forms
                        assert r.shape[0] == R_b and r.lifetime in ((6, 7), (7, 8)), r
                    elif r.name in SEG_NAMES:
                        assert r.shape[0] == NSEG and r.nbytes == block, r
                    elif r.name in ("h", "dh"):
                        assert r.shape[0] == NC, r
                    else:
                        assert r.shape[0] == R, r
                seg = SEG_NAMES if NSEG else set()
                if t == 1:
                    # the saved-set layout without the replay's dout copy; kb under the dh tape
                    base = (
                        {
                            name
                            for slot in cake_backend.SLOTS[
                                "bwd_tape" if tape else "bwd"
                            ]
                            for name, *_ in slot
                        }
                        - {"do_int"}
                        - (set() if NSEG else SEG_NAMES)
                        - (set(GATE_BUFFERS) if fused else set())
                    )
                    assert names == base, (mode, t)
                    by_name = {r.name: r for r in regions}
                    assert by_name["kb"].slot == by_name["dh"].slot
                    assert by_name["kb"].offset == by_name["dh"].offset + 256 * R * H
                    assert by_name["kb"].lifetime == (1, 3)
                    # the fused architectures drop the 4-byte db_int slot (4104 / 3336 B per token-head)
                    assert want == LEAN_SINGLE_BYTES[fused][tape] * R * H + blocks_over
                else:
                    whole_names = {
                        "gk",
                        "dv2",
                        "dh",
                        "u",
                        "kb",
                        "qg",
                        "kg",
                        "w",
                        "beta_int",
                    } | ({"v_new_scratch"} if tape else {"v_new", "h"})
                    assert names == whole_names | pass_names | seg, (mode, t)
                    assert "dv1" not in names and "dAqk" in PASS_NAMES
                    # the documented packing rule, spelled independently: the per-pass set in order, first-fit
                    # into the dead qg / kg and w slots, the rest into the overflow slot shared with the blocks
                    assert want == _expected_lean_bytes(
                        tape, t, R, H, NC, NSEG, fused
                    ), (mode, t)
                    if t == 2:
                        # two passes: dq_f / dk_f fill the qg slot, dg_f the w slot; dAkk, dAqk and db_f
                        # overflow (258 B per token-head), with dg_int / db_int 516
                        over = 258 if fused else 516
                        assert want == whole * R * H + max(over * R * H, 3 * block)
                    elif t >= 4 and NC % t == 0:  # from four passes on everything fits
                        assert want == whole * R * H + 3 * block
                    floor = whole * R * H + 3 * block
                    assert floor <= want <= floor + _pass_bytes_per_th(fused) * R_b * H
                # the per-pass buffers never share bytes with one another or with a live whole-sequence buffer
                # at the tail (the layout raises on a lifetime clash; here: the packing keeps them in dead slots)
                for r in regions:
                    if r.name in PASS_NAMES:
                        for o in regions:
                            if (
                                o is not r
                                and o.slot == r.slot
                                and o.name not in PASS_NAMES
                            ):
                                assert (
                                    o.lifetime[1] < r.lifetime[0] or o.name in SEG_NAMES
                                ), (r, o)
                # the queries
                assert (
                    cake_backend.workspace_bytes(
                        S, N, heads, save_mode=save_mode, arena=arena, **q
                    )
                    == want
                    == p.arena_bytes(save_mode=save_mode, arena=arena)
                )
                assert cake_backend.workspace_bytes(
                    S, N, heads, backward=False, save_mode=save_mode, arena=arena, **q
                ) == p.workspace_bytes("fwd_tape" if tape else "fwd")
                report = cake_backend.memory_report(
                    S, N, heads, save_mode, arena=arena, **q
                )
                assert report["lean"] is True and report["tail_passes"] == t
                assert report["arena_bwd_configured"] == want
                assert (
                    report["arena_bwd_default"]
                    == report["arena_bwd"]
                    == default["arena_bwd"]
                )
                assert (
                    report["arena_fwd_configured"]
                    == report["arena_fwd"]
                    == default["arena_fwd"]
                )
                assert (
                    report["arena_floor"]
                    == whole * R * H + 3 * block
                    <= want
                    <= report["arena_bwd"]
                )
                assert report["bwd_total"] == want + report["out_bwd"]
                assert (
                    report["fwd_total_bwd_arena"]
                    == want + report["out_fwd"] + report["saved"]
                )
                assert report["replay_total"] == default["replay_total"]
                assert report["saved"] == default["saved"]
            assert p.workspace_bytes(mode, 4) <= p.workspace_bytes(mode, 1)
        # with the option off the queries are the default's
        off = cake_backend.CakeKDANondetArena()
        assert cake_backend.workspace_bytes(
            S, N, heads, arena=off, **q
        ) == cake_backend.workspace_bytes(S, N, heads, **q)
        assert cake_backend.memory_report(
            S, N, heads, arena=off, **q
        ) == cake_backend.memory_report(S, N, heads, **q)


def test_lean_arena_serves_the_l2n_rows_with_their_dav_program() -> None:
    """In an in-kernel Q/K L2 norm mode every backward path launches a ``dav`` record of
    that mode (``dav_l2n*``: ``dAqk`` alone, which every lean layout holds whole), so a
    lean arena serves every head count there, H 9-15 included (the rows the flag-off
    test above refuses), in both gate modes, and ``arena_for_budget`` takes its lean
    steps on those rows."""
    modes = itertools.product(ARCHES, ("g", "raw"), ("fused", "fallback"))
    for arch, gate, l2n in modes:
        q = {"sm_count": 148, "arch": arch, "gate": gate, "l2n": l2n}
        for H in range(1, 65):
            case = (arch, gate, l2n, H)
            p = cake_backend.plan(2000, 3, H, 148, arch, gate=gate, l2n=l2n)
            for path in p.records:
                if path in ("fwd", "fwd_nosave"):
                    continue
                assert p.dispatched("dav", path), (case, path)
                served = cake_jit.MODULES[p.record("dav", path)].get("l2n", [])
                assert l2n in served, (case, path)
            for t in (1, 4):
                lean = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
                p.check_arena(lean)
                want = p.workspace_bytes("bwd_lean", t)
                assert p.arena_bytes(arena=lean) == want, (case, t)
                got = cake_backend.workspace_bytes(2000, 3, H, arena=lean, **q)
                assert got == want, (case, t)
                report = cake_backend.memory_report(2000, 3, H, arena=lean, **q)
                assert report["arena_bwd_configured"] == want, (case, t)
            if 9 <= H <= 15:
                got = cake_backend.arena_for_budget(p, p.arena_bytes() - 1)
                assert got.lean, (case, got)


def test_lean_layouts_follow_the_byte_formulas_in_the_l2n_modes() -> None:
    """The lean layouts of the in-kernel Q/K L2 norm modes (both modes, both gate modes,
    every architecture: the gate epilogue is fused on all of them), H 9-15 among the
    rows: one pass is the fused saved-set layout without the replay's dout copy (4104 B
    per internal token-head, tape 3336; its ``dAqk`` is whole in every mode); from two
    passes on ``dAqk`` (written whole by the ``dav_l2n*`` records at 4, read per range by
    GLM intra at 7) is a whole-sequence buffer, 256 B per internal token-head more (3076,
    tape 2308), and leaves the per-pass set; ``memory_report``'s floors take 3076 / 2308
    in place of 2820 / 2052."""
    assert cake_backend._L2N_LEAN_DAQK_BYTES == L2N_LEAN_DAQK_BYTES
    pass_names = frozenset(n for n, *_ in _pass_occupants(True)) - {"dAqk"}
    rows = _lean_rows() + L2N_LEAN_ROWS
    cases = itertools.product(rows, ARCHES, ("g", "raw"), ("fused", "fallback"))
    for (lengths, heads), arch, gate, l2n in cases:
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, 148, arch, gate=gate, l2n=l2n)
        q = {"sm_count": 148, "arch": arch, "gate": gate, "l2n": l2n}
        R, H, NC, NSEG = p.R, p.H, p.NC_pad, p.NSEG
        block = NSEG * H * D * D * 4
        blocks_over = max(0, 3 * block - 256 * R * H)
        assert p.gate_fused is True and p.l2n == l2n
        nc_min = -(-max(lengths) // BT) + 1
        rows_min = BT * (nc_min + nc_min % 2)
        block_min = p.P * H * D * D * 4 if p.recur == "seg" else 0
        for tape in (False, True):
            mode = "bwd_tape_lean" if tape else "bwd_lean"
            save_mode = "tape" if tape else "light"
            whole = LEAN_WHOLE_BYTES[tape] + L2N_LEAN_DAQK_BYTES
            assert whole == (2308 if tape else 3076)
            default = cake_backend.memory_report(
                S,
                N,
                heads,
                save_mode,
                doc_lengths=lengths,
                **q,
            )
            assert default["arena_floor"] == whole * R * H + 3 * block
            floor_batched = whole * rows_min * H + 3 * block_min
            assert default["arena_floor_batched"] == floor_batched
            for t in (1, 2, 3, 4, 8):
                arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
                p.check_arena(arena)
                regions = p.layout(mode, t)
                R_b = BT * -(-NC // t)
                want = p.workspace_bytes(mode, t)
                assert want == max(r.offset + r.nbytes for r in regions)
                slots = cake_backend._slots_of(mode, t, R, H, NC, NSEG, l2n)
                slots = _gate_fused_slots(slots)
                assert want == _slot_bytes(slots, R, H, NSEG, R_b), (mode, t)
                by_name = {r.name: r for r in regions}
                # dAqk is whole on every layout of the mode
                assert by_name["dAqk"].shape[0] == R, (mode, t)
                assert not set(GATE_BUFFERS) & set(by_name), (mode, t)
                if t == 1:
                    single = LEAN_SINGLE_BYTES[True][tape]
                    assert want == single * R * H + blocks_over, (mode, t)
                else:
                    assert by_name["dAqk"].lifetime == (4, 7), (mode, t)
                    assert by_name["dAqk"].nbytes == 256 * R * H, (mode, t)
                    assert pass_names <= set(by_name), (mode, t)
                    for name in pass_names:
                        assert by_name[name].shape[0] == R_b, (mode, t, name)
                    expect = _expected_lean_bytes(
                        tape,
                        t,
                        R,
                        H,
                        NC,
                        NSEG,
                        gate_fused=True,
                        l2n=True,
                    )
                    assert want == expect, (mode, t)
                    # the lifetime table of the plan's multi-pass layout: dAqk whole
                    table = cake_backend.lifetime_table(mode, t, plan=p)
                    rows_by = {row["buffer"]: row for row in table}
                    assert set(rows_by) == set(by_name), (mode, t)
                    daqk = rows_by["dAqk"]
                    assert daqk["unit"] == "R*H", (mode, t)
                    assert (daqk["first_write"], daqk["last_read"]) == (4, 7), (mode, t)
                    if t == 2:
                        # dq_f / dk_f fill the qg slot, dg_f the w slot; dAkk and db_f
                        # overflow: 130 B per token-head (258 with the per-pass dAqk)
                        over = max(130 * R * H, 3 * block)
                        assert want == whole * R * H + over, (mode, t)
                    elif t >= 4 and NC % t == 0:  # from four passes on everything fits
                        assert want == whole * R * H + 3 * block, (mode, t)
                floor = whole * R * H + 3 * block
                report = cake_backend.memory_report(
                    S,
                    N,
                    heads,
                    save_mode,
                    arena=arena,
                    **q,
                )
                assert report["arena_bwd_configured"] == want, (mode, t)
                assert report["arena_floor"] == floor <= want, (mode, t)
                assert want <= report["arena_bwd"], (mode, t)
                got = cake_backend.workspace_bytes(
                    S,
                    N,
                    heads,
                    save_mode=save_mode,
                    arena=arena,
                    **q,
                )
                assert got == want == p.arena_bytes(save_mode=save_mode, arena=arena)


def test_lifetime_table_renders_the_lean_layouts() -> None:
    for mode in cake_backend.LEAN_MODES:
        tape = mode == "bwd_tape_lean"
        rows = cake_backend.lifetime_table(mode)
        single = cake_backend._lean_slots_single(tape)
        assert len(rows) == sum(len(occupants) for occupants in single)
        assert {(row["slot"], row["buffer"]) for row in rows} == {
            (slot, name)
            for slot, occupants in enumerate(single)
            for name, *_ in occupants
        }
        by_name = {row["buffer"]: row for row in rows}
        assert "do_int" not in by_name and by_name["db_int"]["unit"] == "R*H"
        assert by_name["kb"]["slot"] == by_name["dh"]["slot"]
        assert by_name["kb"]["sub_offset_per_th"] == 256
        assert (by_name["kb"]["first_write"], by_name["kb"]["last_read"]) == (1, 3)
        for row in rows:
            assert row["first_write"] <= row["last_read"]
            assert row["unit"] in ("R*H", "seg_block")
        # the fused architectures' single-pass table: no db_int slot, the slots after it renumbered
        fused_rows = cake_backend.lifetime_table(mode, gate_fused=True)
        fused_single = _gate_fused_slots(single)
        assert {(row["slot"], row["buffer"]) for row in fused_rows} == {
            (slot, name)
            for slot, occupants in enumerate(fused_single)
            for name, *_ in occupants
        }
        assert len(fused_rows) == len(rows) - 2 and len(fused_single) == len(single) - 1
        with pytest.raises(ValueError, match="plan="):
            cake_backend.lifetime_table(mode, 2)
        for (lengths, heads), t, arch in itertools.product(
            SHAPES[::4], (2, 3, 4), ARCHES
        ):
            p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, arch)
            rows = cake_backend.lifetime_table(mode, t, plan=p)
            regions = {r.name: r for r in p.layout(mode, t)}
            assert {row["buffer"] for row in rows} == set(regions)
            # gate_fused=None takes the plan's; an explicit value overrides it
            assert bool(set(GATE_BUFFERS) & set(regions)) is not p.gate_fused
            unfused = cake_backend.lifetime_table(mode, t, plan=p, gate_fused=False)
            names = {row["buffer"] for row in unfused}
            assert names == set(regions) | set(GATE_BUFFERS)
            for row in rows:
                r = regions[row["buffer"]]
                assert row["slot"] == r.slot
                assert (row["first_write"], row["last_read"]) == r.lifetime
                if row["buffer"] in PASS_NAMES:
                    assert row["unit"] == "R_b*H"
                    assert row["bytes_per_th"] * r.shape[0] * p.H == r.nbytes
                elif row["buffer"] in SEG_NAMES:
                    assert (
                        row["unit"] == "seg_block"
                        and row["bytes_per_seg_head"] * p.NSEG * p.H == r.nbytes
                    )
                else:
                    assert (
                        row["unit"] == "R*H"
                        and row["bytes_per_th"] * p.R * p.H == r.nbytes
                    )
    with pytest.raises(ValueError, match="tail_passes"):
        cake_backend.lifetime_table("bwd", 2)


def test_saved_set_path_records_bind_only_lean_layout_buffers() -> None:
    """Every record a plan selects on a saved-set path (``light``, ``light_states``, ``tape``) binds, of the arena
    buffers, only ones the lean layouts of two or more passes keep (``h`` / ``v_new`` come from the saved set or
    per-call buffers on the tape layouts; ``seg_state`` is the pair of state blocks): a lean workspace can hand
    every launch its operands."""
    for (lengths, heads), arch in itertools.product(_lean_rows(), ARCHES):
        p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, arch)
        if any(
            p.dispatched("dav", path)
            for path in p.records
            if path not in ("fwd", "fwd_nosave")
        ):
            continue  # the rows a lean arena refuses
        for path in p.records:
            if path in ("fwd", "fwd_nosave", "replay"):
                continue
            mode = "bwd_tape_lean" if path == "tape" else "bwd_lean"
            kept = {r.name for r in p.layout(mode, 2)} | {"h", "v_new", "seg_state"}
            if "w" in kept:
                kept |= {"w_t", "w_tk"}
            for stage, name in p.records[path].items():
                if name is None:
                    continue
                logical = set(cake_jit.MODULES[name]["logical"].values())
                extra = (logical & ARENA_BUFFER_NAMES) - kept
                assert not extra, (
                    arch,
                    lengths,
                    heads,
                    path,
                    stage,
                    name,
                    sorted(extra),
                )


def test_tail_stages_bind_only_range_safe_operands() -> None:
    """The tail split hands every tail kernel the range's offset views, table slices and per-pass buffers; a
    kernel binding one of the whole-sequence buffers by its chunk index directly (``h`` instead of ``h2d``, the
    tables' ``seq_order``, the dead ``w`` / ``u`` / ``kb`` / ``qg`` / ``kg``) would read the wrong rows on a later
    range, so none of the records the plans select may bind one."""
    allowed = (
        TAIL_OFFSET_VIEWS | TAIL_TABLES | TAIL_TOKEN_ROWS | TAIL_SCALARS | PASS_NAMES
    )
    seen = set()
    # (a serial 12-head row too: the deterministic pair of every architecture's saved-set paths)
    rows = [*_lean_rows(), ((300, 77), 12)]
    for (lengths, heads), arch in itertools.product(rows, ARCHES):
        p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, arch)
        for path in p.records:
            if path in ("fwd", "fwd_nosave", "replay"):
                continue
            assert p.dispatched("gate_epilogue", path) is not _gate_fused(arch)
            for stage in TAIL_STAGES:
                if not p.dispatched(stage, path):
                    continue  # the fused GLM intra program's architectures launch no gate epilogue
                name = p.record(stage, path)
                logical = set(cake_jit.MODULES[name]["logical"].values())
                assert not (logical & TAIL_FORBIDDEN), (
                    name,
                    sorted(logical & TAIL_FORBIDDEN),
                )
                assert logical <= allowed, (name, sorted(logical - allowed))
                seen.add(name)
    assert {*DET_GLM.values(), *S1_GLM.values(), "gate_epilogue"} <= seen
    assert any(n.startswith("dqkg") and n not in S1_RECORDS for n in seen)
    assert {S1_DQKG["dqkg_davq"], S1_DQKG["dqkg_davq_nohint"]} <= seen


def test_manifest_is_well_formed(manifest: dict) -> None:
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "cake_pplx._backends.cake_kda_nondet_train"
    arch_flags = manifest["arch_flags"]
    assert set(ARCHES) <= set(arch_flags)
    libraries = manifest["libraries"]
    names = [entry["name"] for entry in libraries]
    # The programs of cake_jit.MODULES, each built for the architectures its record delivers (all three but the
    # records of RECORD_ARCHES; the library twins, fwdh_v3 / fwdh_p3, wy_bwd / wy_fwd_rows, the four raw-class
    # fused gradient records with their g-class records, the four raw-class l2n twins with their g-class records and
    # the l2n parameter finalize with param_finalize, are built once per architecture); the KCP programs have their
    # own package.
    train_libraries = {
        (record["library"], arch)
        for record in cake_jit.MODULES.values()
        for arch in record["arches"]
    }
    count = (
        sum(len(RECORD_ARCHES.get(name, ARCHES)) for name in cake_jit.MODULES)
        - (2 + len(RAW_DQKG_SHARED)) * len(ARCHES)
        - sum(len(RECORD_ARCHES.get(name, ARCHES)) for name in RAW_L2N_DQKG)
        - sum(
            len(
                set(RECORD_ARCHES.get(name, ARCHES))
                & set(RECORD_ARCHES.get(det, ARCHES))
            )
            for name, det in L2N_PARAM.items()
        )
    )
    # 227 entries before the in-kernel Q/K L2 norm's copies, plus its 87 (27 programs x 3 architectures, the raw
    # class's kb-once replay prep, the five dav_l2n programs and two l2n fused gradient programs among them, the
    # sm_103a forward intra and GLM intra forms, the replay prep's saved-rstd twin on sm_100a / sm_103a and the two
    # l2n fused gradient programs with the L2 hints off on sm_107a); the raw class's l2n twins of the fused gradient
    # and the l2n parameter finalize add no build (their programs ship as the g-class copies' and param_finalize's)
    assert count == 227 + 27 * 3 + 2 + 2 + 2 == 314
    assert len(train_libraries) == count
    assert len(names) == len(set(names)) == count
    assert {
        arch: sum(entry["arch"] == arch for entry in libraries) for arch in ARCHES
    } == {"sm_100a": 103, "sm_103a": 104, "sm_107a": 107}
    for entry in libraries:
        assert entry["arch"] in ARCHES
        assert entry["entry"] == "run"
        assert "--use_fast_math" in entry["nvcc_flags"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
        assert len(entry["sources"]) == 2
        for source in entry["sources"]:
            path = PACKAGE_DIR / source
            assert path.exists(), source
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"][source]
            ), source
            assert "Copyright (c) 2026, NVIDIA CORPORATION." in "\n".join(
                path.read_text().splitlines()[:3]
            ), source
    expected = {
        cake_jit.library_name(name, arch)
        for name, record in cake_jit.MODULES.items()
        for arch in record["arches"]
    }
    assert set(names) == expected


def _det_manifest_or_skip() -> dict:
    path = Path(det_jit.__file__).resolve().parent / "build.json"
    if not path.exists():
        pytest.skip("the deterministic package has no build.json (not exported)")
    return json.loads(path.read_text())


def test_copied_programs_are_the_deterministic_packages_sources(
    manifest: dict,
) -> None:
    """Every record but the S1 ones, the hand-off twins and the tape WY twin is a copy of the deterministic package's
    record of the same name (a raw-class fused gradient record: of its ``RAW_DQKG`` / ``RAW_L2N_DQKG`` record;
    ``param_finalize_l2n``: of ``param_finalize``): the same digest in its library name (the package prefix aside),
    stage, launch, operands and grid, and (outside the g class of the two pair stages with the in-kernel Q/K L2 norm
    off, whose rules route the S1 cells) the same rule and architectures; each of its libraries has the deterministic
    library's flags and sources, byte for byte once the package name in them is replaced.  The S1 programs and the
    twins are the only new code: the raw parameter-gradient pair (``glm_intra_raw``, ``param_finalize``) is the
    deterministic package's programs on every architecture."""
    det_manifest = _det_manifest_or_skip()
    det_dir = Path(det_jit.__file__).resolve().parent
    ours = {entry["name"]: entry for entry in manifest["libraries"]}
    theirs = {entry["name"]: entry for entry in det_manifest["libraries"]}
    copied = set()
    for name, record in cake_jit.MODULES.items():
        if name in S1_RECORDS or name in TWIN_RECORDS or name in TAPE_WY_RECORDS:
            continue
        det_name = DET_SOURCE.get(name, name)
        twin = det_jit.MODULES[det_name]
        assert record["library"] == twin["library"].replace(
            "cake_kda_det_train_", "cake_kda_train_"
        ), name
        for key in ("stage", "ffi_entry", "launch", "arg_plan", "logical"):
            assert record[key] == twin[key], (name, key)
        if name in RAW_DQKG or name in RAW_L2N_DQKG:
            assert record["variant"] == name.removeprefix("dqkg_"), name
            assert record["gate"] == "raw" and "gate" not in twin, name
            assert record.get("l2n") == twin.get("l2n"), name
        elif name in L2N_PARAM:
            assert record["variant"] == "l2n" and twin["variant"] is None, name
            assert record["gate"] == twin["gate"] == "raw", name
            assert record["l2n"] == list(L2N_ON) and "l2n" not in twin, name
        else:
            assert record["variant"] == twin["variant"], name
            assert record.get("gate") == twin.get("gate"), name
            # the in-kernel Q/K L2 norm modes it serves: the deterministic record's
            assert record.get("l2n") == twin.get("l2n"), name
        assert record["grid_rule"] == twin["grid_rule"], name
        pair_off = record["stage"] in ("dqkg", "glm_intra") and "off" in record.get(
            "l2n", ["off"]
        )
        # the raw class (the fused gradient's copies and glm_intra_raw) keeps the deterministic rule too
        if not pair_off or record.get("gate") == "raw":
            assert record["select_rule"] == twin["select_rule"], name
            assert list(record["arches"]) == list(twin["arches"]), name
        else:
            # a pair-stage copy ships where its narrowed rule launches it: some of the twin's architectures
            assert set(record["arches"]) <= set(twin["arches"]), name
        for arch in record["arches"]:
            mine = ours[cake_jit.library_name(name, arch)]
            source = theirs[det_jit.library_name(det_name, arch)]
            for key in ("nvcc_flags", "nvcc_flags_if", "min_cuda", "gencode", "entry"):
                assert mine[key] == source[key], (name, arch, key)
            assert len(mine["sources"]) == len(source["sources"]) == 2
            for path, det_path in zip(
                sorted(mine["sources"]), sorted(source["sources"]), strict=True
            ):
                text = (PACKAGE_DIR / path).read_text()
                det_text = (det_dir / det_path).read_text()
                assert "cake_kda_det_train" not in text, (name, path)
                assert text == det_text.replace(
                    "cake_kda_det_train", "cake_kda_train"
                ), (
                    name,
                    path,
                )
            copied.add(mine["name"])
    new = {
        cake_jit.library_name(name, arch)
        for name in (*S1_RECORDS, *TWIN_RECORDS, *TAPE_WY_RECORDS)
        for arch in cake_jit.MODULES[name]["arches"]
    }
    assert copied == set(ours) - new and new <= set(ours)
    # the raw parameter-gradient pair: 6 copied entries (two programs on the three architectures), no new one
    pair = {
        cake_jit.library_name(name, arch)
        for name in RAW_PARAM_PAIR.values()
        for arch in ARCHES
    }
    assert len(pair) == 6 and pair <= copied and not pair & new


def test_library_name_refuses_an_undelivered_architecture() -> None:
    assert (
        cake_jit.library_name("table", "sm_100a")
        == f"{cake_jit.MODULES['table']['library']}_sm_100a"
    )
    with pytest.raises(RuntimeError, match="not delivered"):
        cake_jit.library_name("table", "sm_90a")
    # a record delivered for a subset of the architectures refuses the others
    for name, arches in RECORD_ARCHES.items():
        for arch in ARCHES:
            if arch in arches:
                assert cake_jit.library_name(name, arch).endswith(f"_{arch}")
            else:
                with pytest.raises(RuntimeError, match="not delivered"):
                    cake_jit.library_name(name, arch)


def test_run_module_names_a_missing_logical_argument() -> None:
    # The table kernel's first parameter is the packing's ``cu_seqlens``; an empty
    # value set is refused before any library is loaded.
    with pytest.raises(ValueError, match="needs logical argument 'cu_seqlens'"):
        cake_jit.run_module("table", {}, grid=(1, 1, 1), device_index=0)


# --------------------------------------------------------------------------- kernels (device)


def _expected_launches(
    p,
    path: str,
    *,
    saved_tables: bool = True,
    tail_passes: int = 1,
    tape_wy: str | None = None,
    finalize: bool = True,
) -> list[str]:
    """The launch order of ``path`` on plan ``p`` (the records by name), spelled from the composition's rules:
    the tables (on the forward and the replay where the plan dispatches a table program; on a saved-set path only
    when the set carries none, ``saved_tables=False``), the intra program (the forward: the prep folded in; the
    segmented route's writes the tables itself up to 256 documents), the backward prep (and the intra program on
    the replay: before the prep where it writes the tables), WY where
    a fused program does not form the representation, the recurrence (the segmented passes around the carry),
    dA / dv where the fused gradient does not host it, the adjoint recurrence, the fused gradient, GLM intra and,
    on sm_107a, the gate epilogue (the tail stages once per chunk range of a lean workspace's ``tail_passes``),
    then on a raw-gate plan the parameter finalize once (``finalize=False``: a document group of a batched call,
    whose finalize runs once after the last group).  The carry record is the row's: the overlap pair's on the
    forward's overlap rows, the two-level carry where its rule accepts, else the chain carry.  ``tape_wy``: the
    record the tape backward launches for WY in place of the plan's (the tape WY twin,
    :func:`cake_backend._tape_wy_record`)."""
    seg = p.recur == "seg"
    seq: list[str] = []
    # the table programs where the plan dispatches them (the segmented route's intra program writes the tables
    # itself up to 256 documents on the forward and the replay: no table launch there)
    tables_here = (not saved_tables) or path in ("fwd", "fwd_nosave", "replay")
    if tables_here and p.dispatched("table", path):
        seq.append(p.record("table", path))
        if seg and p.dispatched(
            "seg_table", path
        ):  # above 256 documents: the block table on its own
            seq.append("seg_table")
    if path in ("fwd", "fwd_nosave"):
        seq.append(p.record("fwd_intra", path))
        if p.dispatched("wy_fwd", path):
            seq.append(p.record("wy_fwd", path))
        seq.append(p.record("fwdh", path))
        if (
            seg
        ):  # the carry record of the row (the overlap pair's on its rows), then pass 2
            seq += [p.record("seg_carry", path), p.record("fwdh_p3", path)]
        return seq
    # the replay's intra program: before the prep where it writes the tables the prep reads, after it otherwise
    intra_first = path == "replay" and not p.dispatched("table", path)
    if intra_first:
        seq.append(p.record("fwd_intra", path))
    seq.append(p.record("bwd_prep", path))
    if path == "replay" and not intra_first:
        seq.append(p.record("fwd_intra", path))
    if p.dispatched("wy", path):
        seq.append(tape_wy if path == "tape" and tape_wy else p.record("wy", path))
    if p.dispatched("fwdh", path):
        seq.append(p.record("fwdh", path))
        if seg:
            seq.append(p.record("seg_carry", path))
    if p.dispatched("fwdh_p3", path):
        seq.append(p.record("fwdh_p3", path))
    if p.dispatched("dav", path):
        seq.append(p.record("dav", path))
    dhu = p.record("dhu", path)
    seq += [dhu, p.record("seg_carry", path), dhu] if seg else [dhu]
    ranges = cake_backend._tail_ranges(p.NC_pad, tail_passes)
    seq += _tail_records(p, path) * len(ranges)
    if finalize and p.dispatched("param_finalize", path):
        seq.append(p.record("param_finalize", path))
    return seq


def _tail_records(p, path: str) -> list[str]:
    """The tail's records on ``path`` in launch order: the fused gradient, GLM intra and, where the architecture
    dispatches it (sm_107a), the gate epilogue."""
    return [p.record(stage, path) for stage in TAIL_STAGES if p.dispatched(stage, path)]


def _device_route(device: torch.device, n: int, h: int) -> str:
    """The route of the row on the current device (its SM count and architecture)."""
    return _expected_segments(
        n, h, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )[0]


def _run_fwd_bwd(inputs, *, scale=SCALE, workspace=None, save=True, save_mode=None):
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, saved = cake_kda_nondet_train_forward(
        q,
        k,
        v,
        g,
        beta,
        cu,
        scale=scale,
        save_for_backward=save,
        workspace=workspace,
        save_mode=save_mode,
    )
    grads = cake_kda_nondet_train_backward(
        q, k, v, g, beta, cu, dout, saved=saved, scale=scale, workspace=workspace
    )
    return out, dict(zip(("dq", "dk", "dv", "dbeta", "dg"), grads, strict=True)), saved


def _spy_call_workspaces(monkeypatch) -> list:
    """Record the workspace every call without one runs in (``_entry_acquire``): its option, direction, bytes, save
    mode and ``doc_lengths``, not the workspace itself (a strong reference would be "retained")."""
    created: list = []
    original = cake_backend._entry_acquire

    def spy(*args, **kwargs):
        ws_, key = original(*args, **kwargs)
        created.append(
            types.SimpleNamespace(
                arena_option=ws_.arena_option,
                backward=ws_.backward,
                bytes=ws_.bytes,
                save_mode=ws_.save_mode,
                doc_lengths=ws_.doc_lengths,
            )
        )
        return ws_, key

    monkeypatch.setattr(cake_backend, "_entry_acquire", spy)
    return created


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_forward_and_backward_match_the_exact_reference(
    device, manifest, shape, record_property
) -> None:
    """The forward with its light saved set and the backward from it: ``out`` and the five gradients judged
    against the FP64 reference beside the naive BF16 tier over the judge's draws (``s1-h1``: a batch of one
    one-token document)."""
    lengths, heads = shape

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        out, grads, _ = _run_fwd_bwd(inputs)
        return {"": {"out": out, **grads}}, inputs

    _judge(record_property, ("forward_and_backward", shape), draw)


GRAD_ORDER = ("dq", "dk", "dv", "dbeta", "dg")  # the order the backward returns
LEAVES = ("q", "k", "v", "g", "beta")
LEAF_GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # the gradients of LEAVES, in their order


def _seed(*key) -> int:
    """The base seed of one judged case, a CRC of its key (the test and the row): the judge draws every input of
    the case from it."""
    return zlib.crc32(repr(key).encode())


def _draw(lengths, heads, seed: int, device, *, gate: str = "g", batched: bool = False):
    """The operands of one draw of a judged case (``draw_case``): ``gate="g"`` the activated FP32 gate, ``"raw"``
    the BF16 raw gate with FP32 ``A_log`` / ``dt_bias``; every tensor a function of ``seed``."""
    return draw_case(lengths, heads, gate, False, seed, device, batched=batched)


def _ops(inputs) -> tuple:
    """``q, k, v, g, beta, cu_seqlens, dout`` of ``inputs``."""
    return tuple(inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout"))


def _named(results, names=GRAD_ORDER) -> dict:
    """A backward's results (in ``names`` order) by name, as the judge reads them."""
    return dict(zip(names, results, strict=True))


def _leaves(inputs, names=LEAVES) -> list[torch.Tensor]:
    return [inputs[n].detach().clone().requires_grad_(True) for n in names]


def _leaf_grads(leaves, names=LEAF_GRADS) -> dict:
    """The leaves' gradients by name."""
    return {name: leaf.grad for name, leaf in zip(names, leaves, strict=True)}


def _judge(record_property, key, draw):
    """Judge one case over the judge's draws (``judge_arms``): ``draw(j, s)`` makes draw ``j``'s operands from the
    seed ``s``, launches every arm of the case on them, runs the draw's exact checks and returns ``(arms,
    inputs)``; the case's seed is :func:`_seed` of ``key``.  The verdicts are recorded and settled against the
    owner-accepted gaps."""
    return judge_arms(record_property, draw, seed=_seed(*key), scale=SCALE)


def _finite(results: dict, tag=None) -> None:
    """Every result of a row too large to judge is finite, the judge's contract where the reference is finite: a
    NaN-filled arena or buffer that reached a result shows here."""
    bad = [name for name, t in results.items() if not bool(torch.isfinite(t).all())]
    assert not bad, (tag, bad)


def _finite_arms(arms: dict, tag=None) -> None:
    """:func:`_finite` on every result of every arm (``{arm: results}``, a dict or a list of dicts)."""
    for arm, results in arms.items():
        for res in results if isinstance(results, (list, tuple)) else (results,):
            _finite(res, (tag, arm))


# the replay rows: the deterministic suite's four, a serial 12-head row (the deterministic pair on both paths) and
# a 16- and a 32-head row (sm_103a: S1 from the saved set, the deterministic pair on the replay)
REPLAY_SHAPES = [*SHAPES[6:10], ((300, 77), 12), ((129, 63), 16), ((65, 200), 32)]


@pytest.mark.parametrize("shape", REPLAY_SHAPES, ids=_shape_id)
def test_replay_and_repeats_are_judged(
    device, manifest, shape, record_property
) -> None:
    """Every path of one call's inputs, each judged under its own name: the forward with its light set and the
    backward from it, twice (arm ``""``, the worse of the two), the checkpoint replay (``replay``; on some rows it
    runs the other pair than the saved set, :func:`_s1`), the light set without its tables and the light set on
    another workspace (``bare``, ``other_workspace``), the tape set of the default entry (``tape``), of a tape
    workspace (``tape_workspace``) and that workspace's replay (``tape_replay``).  The tape sets are version 2 with
    ``h`` (``seg_m`` on the segmented route) and no version-4 tensor; a tape workspace has the byte query's
    bytes."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    seg = _device_route(device, N, heads) == "seg"
    tape_ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, save_mode="tape")
    assert tape_ws.bytes == cake_backend.workspace_bytes(
        S, N, heads, save_mode="tape", sm_count=tape_ws.plan.sm_count
    )

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        runs = [_run_fwd_bwd(inputs) for _ in range(2)]
        saved = runs[0][2]
        arms = {"": [{"out": out, **grads} for out, grads, _ in runs]}
        arms["replay"] = _named(
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=None, scale=SCALE
            )
        )
        for name, other in (
            ("bare", dataclasses.replace(saved, tables=None)),
            ("other_workspace", saved),
        ):
            ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device)
            arms[name] = _named(
                cake_kda_nondet_train_backward(
                    q, k, v, g, beta, cu, dout, saved=other, scale=SCALE, workspace=ws
                )
            )
        out_t, grads_t, saved_t = _run_fwd_bwd(inputs, save_mode="tape")
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
        assert saved_t.h is not None and (saved_t.seg_m is not None) == seg
        assert saved_t.qg is None and saved_t.w is None and saved_t.u is None
        arms["tape"] = {"out": out_t, **grads_t}
        out_w, grads_w, saved_w = _run_fwd_bwd(inputs, workspace=tape_ws)
        assert saved_w.version == cake_backend.SAVED_VERSION_TAPE
        arms["tape_workspace"] = {"out": out_w, **grads_w}
        arms["tape_replay"] = _named(
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=None, scale=SCALE, workspace=tape_ws
            )
        )
        return arms, inputs

    _judge(record_property, ("replay_and_repeats", shape), draw)


def test_documents_do_not_leak_into_one_another(
    device, manifest, record_property
) -> None:
    """Each document of a pack run alone: the per-document results concatenated (arm ``alone``) and the pack's
    (arm ``""``) are each judged against the pack's FP64 reference, which is per document (a leak across
    documents moves the pack's results off it)."""
    lengths, heads = (100, 640, 77), 2

    def draw(j, s):
        packed = _draw(lengths, heads, s, device)
        out, grads, _ = _run_fwd_bwd(packed)
        pieces = []
        start = 0
        for length in lengths:
            stop = start + length
            single = {
                name: packed[name][start:stop].contiguous()
                for name in ("q", "k", "v", "g", "beta", "dout")
            }
            single["cu_seqlens"] = torch.tensor(
                [0, length], dtype=torch.int32, device=device
            )
            s_out, s_grads, _ = _run_fwd_bwd(single)
            pieces.append({"out": s_out, **s_grads})
            start = stop
        alone = {name: torch.cat([p[name] for p in pieces]) for name in pieces[0]}
        return {"": {"out": out, **grads}, "alone": alone}, packed

    _judge(record_property, ("documents_alone", lengths, heads), draw)


def test_autograd_and_checkpoint_give_the_explicit_gradients(
    device, manifest, record_property
) -> None:
    """The explicit forward and backward (arm ``""``), the autograd entry (``autograd``), the same under
    ``torch.utils.checkpoint``, which re-runs the forward in the backward with nothing special-cased (``ckpt``),
    and the autograd node with a non-contiguous upstream gradient (``dout_nc``; two heads: the transposed view of a
    one-head tensor would still be contiguous), each judged."""
    from torch.utils.checkpoint import checkpoint

    lengths, heads = (63, 1, 65), 2

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        out, grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}
        leaves = _leaves(inputs)
        o = cake_kda_nondet(*leaves, cu)
        o.backward(dout)
        arms["autograd"] = {"out": o.detach(), **_leaf_grads(leaves)}
        leaves = _leaves(inputs)
        o = checkpoint(lambda *a: cake_kda_nondet(*a, cu), *leaves, use_reentrant=False)
        o.backward(dout)
        arms["ckpt"] = {"out": o.detach(), **_leaf_grads(leaves)}
        leaves = _leaves(inputs)
        o = cake_kda_nondet(*leaves, cu)
        dout_nc = dout.transpose(0, 1).contiguous().transpose(0, 1)
        assert not dout_nc.is_contiguous()
        o.backward(dout_nc)
        arms["dout_nc"] = _leaf_grads(leaves)
        return arms, inputs

    _judge(record_property, ("autograd_and_checkpoint", lengths, heads), draw)


def test_batch_axis_is_optional(device, manifest, record_property) -> None:
    """Operands with the leading batch axis of 1 give results with it (judged as ``batched``, the flat call's as
    ``""``); the autograd entry takes them too (``autograd``: its output, of the batched shape)."""
    lengths, heads = (64, 64), 1

    def draw(j, s):
        flat = _draw(lengths, heads, s, device)
        batched = _draw(lengths, heads, s, device, batched=True)
        out, grads, _ = _run_fwd_bwd(flat)
        b_out, b_grads, _ = _run_fwd_bwd(batched)
        assert b_out.shape == (1, sum(lengths), heads, D)
        for name, t in b_grads.items():
            assert t.dim() == grads[name].dim() + 1 and t.shape[0] == 1, name
        o = cake_kda_nondet(*(batched[n] for n in LEAVES), batched["cu_seqlens"])
        assert o.shape == b_out.shape
        arms = {
            "": {"out": out, **grads},
            "batched": {"out": b_out, **b_grads},
            "autograd": {"out": o},
        }
        return arms, flat

    _judge(record_property, ("batch_axis", lengths, heads), draw)


def test_workspace_reuse_and_byte_counts(device, manifest, record_property) -> None:
    """A caller workspace: its bytes are the byte queries' (with or without the device's SM count and
    architecture), its outputs and saved set are fresh per-call tensors outside the arena, it refuses another
    problem, a backward on a forward-only workspace and an unknown save mode, and lists its paths and records; a
    light workspace serves the tape forward.  Judged: its first and second call (arm ``""``, the worse of the two),
    a call without a workspace (``default``) and the tape forward on the light workspace (``tape_forward``)."""
    lengths, heads = (129,), 2
    S, N = sum(lengths), len(lengths)
    ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device)
    sm = ws.plan.sm_count
    # the byte queries without an architecture take the device's: with its SM count given or taken from it, the
    # same bytes as the workspace (a few-chain row: 37 segments per document on sm_100a, 38 elsewhere)
    expected_segments = 37 if ws.plan.arch == "sm_100a" else 38
    assert expected_segments == ws.plan.P
    assert (
        ws.bytes
        == cake_backend.workspace_bytes(S, N, heads, sm_count=sm)
        == cake_backend.workspace_bytes(S, N, heads)
        == cake_backend.workspace_bytes(S, N, heads, sm_count=sm, arch=ws.plan.arch)
        == ws.arena.numel()
    )
    assert cake_backend.saved_bytes(
        S, heads, N, sm_count=sm
    ) == cake_backend.saved_bytes(S, heads, N)
    assert cake_backend.memory_report(S, N, heads)["arch"] == ws.plan.arch
    fwd_only = CakeKDANondetWorkspace.for_problem(S, N, heads, device, backward=False)
    assert fwd_only.bytes == cake_backend.workspace_bytes(
        S, N, heads, backward=False, sm_count=sm
    )
    other = CakeKDANondetWorkspace.for_problem(S + 1, N, heads, device)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, grads, saved = _run_fwd_bwd(inputs, workspace=ws)
        if j == 0:
            # Outputs and the saved set are fresh per-call tensors; the arena holds the intermediates only.
            arena_lo, arena_hi = ws.arena.data_ptr(), ws.arena.data_ptr() + ws.bytes
            assert saved.qg is not None
            for t in (
                out,
                *grads.values(),
                saved.Aqk,
                saved.Akk,
                saved.qg,
                *saved.tables,
            ):
                assert not (arena_lo <= t.data_ptr() < arena_hi)
            assert len(saved.tables) == len(ws.plan.table_layout())
            assert saved.Aqk.numel() * saved.Aqk.element_size() * 2 == (
                cake_backend.saved_bytes(S, heads)
            )
        out2, grads2, _ = _run_fwd_bwd(inputs, workspace=ws)
        out3, grads3, _ = _run_fwd_bwd(inputs)
        if j == 0:
            with pytest.raises(ValueError, match="backward=False"):
                cake_kda_nondet_train_backward(
                    q, k, v, g, beta, cu, dout, saved=saved, workspace=fwd_only
                )
            with pytest.raises(ValueError, match="allocated for"):
                cake_kda_nondet_train_forward(q, k, v, g, beta, cu, workspace=other)
            with pytest.raises(ValueError, match="save_mode"):
                cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=ws, save_mode="full"
                )
            assert set(ws.paths()) == {
                "fwd",
                "fwd_nosave",
                "light",
                "tape",
                "replay",
            } | ({"light_states"} if ws.plan.recur == "seg" else set())
            assert ws.records() == {
                path: {
                    stage: ws.plan.record(stage, path) for stage in ws.plan.stages(path)
                }
                for path in ws.paths()
            }
            assert fwd_only.paths() == ("fwd", "fwd_nosave")
            assert fwd_only.records() == {
                path: ws.records()[path] for path in ("fwd", "fwd_nosave")
            }
            ws.load()
        # A light workspace serves the tape forward (its layout is the smaller one); the tape backward needs a tape
        # workspace or a tape saved set.
        out_t, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
        )
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
        arms = {
            "": [{"out": out, **grads}, {"out": out2, **grads2}],
            "default": {"out": out3, **grads3},
            "tape_forward": {"out": out_t},
        }
        return arms, inputs

    _judge(record_property, ("workspace_reuse", lengths, heads), draw)
    with pytest.raises(ValueError, match="with_plan"):
        ws.with_plan(other.plan)
    twin = ws.with_plan(ws.plan)
    assert (
        twin.arena.data_ptr() == ws.arena.data_ptr() and twin.records() == ws.records()
    )


def test_launch_sequences_follow_the_plan(device, manifest, record_property) -> None:
    """Every path's launch list is the one this suite spells from the plan, and the records of every path the
    workspace serves name the programs these lists launched.  The results of every path are judged per row:
    ``<row>`` the forward and the light set's backward, ``<row>.nosave`` the forward without a saved set,
    ``<row>.bare`` the backward from the set without its tables, ``<row>.replay`` the replay and ``<row>.tape`` the
    tape forward and the tape backward on a tape workspace."""

    # Seven rows: the segmented route with six chains (the two-level carry), 16 heads (on
    # sm_100a the 32-column serial form whose light set carries w / u: no wy launch on the light path; segmented
    # on sm_103a / sm_107a), the 64-column serial form at 64 heads (the fused forward and the fused backward
    # recompute), the few-chain segmented rows (the two-CTA adjoint; the fused gradient hosts dAqk) and, on a
    # 148-SM device, the overlap pair's rows (the release-order pass 2 at 2 heads, the chain-major pass 2 at 8
    # heads, and three documents at 4 heads: one walk CTA per SM per document, exactly the 3.0-wave threshold).
    # The gate epilogue launches on sm_100a / sm_107a (sm_103a's GLM intra program fuses it).
    def check_row(lengths, heads):
        S, N = sum(lengths), len(lengths)
        row = _shape_id((lengths, heads))
        launched: list[str] = []

        def recording(name, values, *, grid, device_index, _log=launched):
            _log.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, launcher=recording)
        tape_ws = CakeKDANondetWorkspace.for_problem(
            S,
            N,
            heads,
            device,
            save_mode="tape",
            launcher=recording,
        )
        p = ws.plan
        assert p.recur == _device_route(device, N, heads)
        assert p.seg_save is False and p.light_path == "light"
        assert p.dispatched("gate_epilogue", "light") is not p.gate_fused

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            launched.clear()
            out, saved = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws
            )
            if j == 0:
                assert launched == _expected_launches(p, "fwd")
                assert len(launched) == p.launch_count("fwd")
            launched.clear()
            # the forward without a saved set: the internal-row programs, the same launch count
            out_nosave, none = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws, save_for_backward=False
            )
            assert none is None
            if j == 0:
                assert launched == _expected_launches(p, "fwd_nosave")
                assert (
                    len(launched)
                    == p.launch_count("fwd_nosave")
                    == p.launch_count("fwd")
                )
                assert (launched != _expected_launches(p, "fwd")) is (
                    not p.internal_rows("fwd")
                )
            launched.clear()
            grads = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
            )
            if j == 0:
                assert launched == _expected_launches(p, "light")
                assert len(launched) == p.launch_count("light") == p.launch_count("bwd")
                assert not (p.light_saves_wu and "wy_bwd" in launched)
            launched.clear()
            bare = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=dataclasses.replace(saved, tables=None),
                workspace=ws,
            )
            if j == 0:
                assert launched == _expected_launches(p, "light", saved_tables=False)
                assert len(launched) == p.launch_count("light", tables=False)
            launched.clear()
            replay = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=None, workspace=ws
            )
            if j == 0:
                assert launched == _expected_launches(p, "replay")
                assert len(launched) == p.launch_count("replay")
            launched.clear()
            out_t, saved_t = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
            )
            if j == 0:
                assert launched == _expected_launches(p, "fwd")
            launched.clear()
            grads_t = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved_t, workspace=tape_ws
            )
            if j == 0:
                # the WY launch: the tape WY twin where the device's rule admits it, the plan's record elsewhere
                tape_wy = cake_backend._tape_wy_record(p, device.index)
                assert launched == _expected_launches(p, "tape", tape_wy=tape_wy)
                assert len(launched) == p.launch_count("tape")
            arms = {
                row: {"out": out, **_named(grads)},
                f"{row}.nosave": {"out": out_nosave},
                f"{row}.bare": _named(bare),
                f"{row}.replay": _named(replay),
                f"{row}.tape": {"out": out_t, **_named(grads_t)},
            }
            return arms, inputs

        _judge(record_property, ("launch_sequences", lengths, heads), draw)
        # the records of every path the workspace serves name the programs these sequences launched
        records = ws.records()
        assert set(records) == set(ws.paths())
        for path in ("fwd", "fwd_nosave", "light", "tape", "replay"):
            # the tables are the path's records too: launched when the saved set carries none
            expected = set(_expected_launches(p, path, saved_tables=False))
            assert expected == set(records[path].values()), path

    for lengths, heads in (
        ((63, 1, 65), 2),
        ((129,), 16),
        ((63, 1, 65), 64),
        ((129,), 2),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((63, 1, 65), 4),
    ):
        check_row(lengths, heads)


def test_operand_checks(device, manifest) -> None:
    # Two heads: with one head the transposed view below would still be contiguous
    # (a size-1 axis has no stride of its own).
    lengths, heads = (65,), 2
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    with pytest.raises(ValueError, match="contiguous"):
        cake_kda_nondet_train_forward(
            q.transpose(0, 1).contiguous().transpose(0, 1), k, v, g, beta, cu
        )
    with pytest.raises(ValueError, match="expected torch.float32"):
        cake_kda_nondet_train_forward(q, k, v, g.to(torch.bfloat16), beta, cu)
    with pytest.raises(ValueError, match="expected shape"):
        cake_kda_nondet_train_forward(q, k, v, g, beta[:-1], cu)
    with pytest.raises(ValueError, match="batch axis"):
        cake_kda_nondet_train_forward(
            q.unsqueeze(0).expand(2, *q.shape), k, v, g, beta, cu
        )
    with pytest.raises(ValueError, match="cu_seqlens"):
        cake_kda_nondet_train_forward(q, k, v, g, beta, cu.to(torch.int64))
    _, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu)
    assert saved.version == cake_backend.SAVED_VERSION == 4
    assert saved.qg is not None and saved.w is None and saved.u is None

    def refused(match: str, other: CakeKDANondetChunkSaved) -> None:
        with pytest.raises(ValueError, match=match):
            cake_kda_nondet_train_backward(q, k, v, g, beta, cu, dout, saved=other)

    # versions 1 and 3 (the light sets without qg) are no longer accepted
    for version in (1, 3):
        refused(
            "version",
            CakeKDANondetChunkSaved(
                version, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE, qg=saved.qg
            ),
        )
    # a version-4 light set carries the forward's qg (the light backward's prep writes none), with the plan's shape
    refused("qg", dataclasses.replace(saved, qg=None))
    refused("saved.qg", dataclasses.replace(saved, qg=saved.qg[:-BT]))
    # w and u come together, and only on the rows whose light backward reads them from the set (none at two heads)
    refused("w and u together", dataclasses.replace(saved, w=saved.qg))
    refused("w and u together", dataclasses.replace(saved, u=saved.qg))
    refused("carries w / u", dataclasses.replace(saved, w=saved.qg, u=saved.qg))
    # a version-4 set carries hseg and seg_state together, with the plan's shapes
    nseg = cake_backend.plan(
        65,
        1,
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
    ).NSEG
    block = torch.zeros((nseg, heads, D, D), dtype=torch.float32, device=device)
    refused(
        "hseg and seg_state together",
        dataclasses.replace(saved, tables=None, hseg=block),
    )
    refused(
        "saved.seg_state",
        dataclasses.replace(saved, tables=None, hseg=block, seg_state=block),
    )
    refused(
        "needs h and v_new",
        CakeKDANondetChunkSaved(2, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE),
    )
    refused("saved.tables", dataclasses.replace(saved, tables=saved.tables[:2]))
    refused("saved: for", dataclasses.replace(saved, tables=None, S=64))
    out, _ = cake_kda_nondet_train_forward(
        q, k, v, g, beta, cu, save_for_backward=False
    )
    assert out.shape == (65, heads, D)
    assert cake_backend.is_supported(device)


def test_the_light_saved_set_carries_qg_and_w_u_where_the_plan_says(
    device, manifest, record_property
) -> None:
    """Version 4 on the device: every light set carries the fold's qg (BF16 [R, H, 128], a per-call tensor
    outside the arena); w / u exactly on the rows whose plan reads them from the set (the paired serial 16-head
    rows of sm_100a and the segmented 16-head rows of sm_103a / sm_107a above 32 chains: no wy launch there); the
    bytes are the query's; the backward refuses a set without qg, an unpaired w / u and w / u on the wrong row.
    Judged per row: the forward and the backward from the set (``<row>``) and from the set without its tables
    (``<row>.bare``)."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    seen_wu = 0
    # the 16-head rows: one, two and three documents (16, 32 and 48 chains: on sm_103a / sm_107a the third alone
    # carries w / u; on sm_100a all three are paired serial rows)
    cases = (
        ((129,), 16),
        ((65,), 2),
        ((63, 1, 65), 64),
        ((129, 300), 16),
        ((129, 300, 70), 16),
    )

    def check_row(lengths, heads):
        nonlocal seen_wu
        S, N = sum(lengths), len(lengths)
        row = _shape_id((lengths, heads))
        launched: list[str] = []

        def recording(name, values, *, grid, device_index, _log=launched):
            _log.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, launcher=recording)
        p = ws.plan
        wu = _wu_saved(N, heads, sm, arch)
        seen_wu += wu
        assert p.light_saves_qg is True and p.light_saves_wu is wu

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            out, saved = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws
            )
            if j == 0:
                assert saved.version == cake_backend.SAVED_VERSION
                rows = [saved.qg] + ([saved.w, saved.u] if wu else [])
                assert (saved.w is None) == (saved.u is None) == (not wu), (
                    lengths,
                    heads,
                )
                arena_lo, arena_hi = (
                    ws.arena.data_ptr(),
                    ws.arena.data_ptr() + ws.bytes,
                )
                for t in rows:
                    assert t.shape == (p.R, heads, D) and t.dtype == torch.bfloat16
                    assert t.device == ws.device
                    assert not (arena_lo <= t.data_ptr() < arena_hi)
                tables = p.table_words() * 4
                states = [saved.hseg, saved.seg_state] if saved.hseg is not None else []
                tensors = (saved.Aqk, saved.Akk, *rows, *states)
                owned = sum(t.numel() * t.element_size() for t in tensors) + tables
                assert owned == cake_backend.saved_bytes(
                    S, heads, N, sm_count=sm, arch=arch
                )
                assert owned == cake_backend.saved_bytes(S, heads, N, sm_count=sm)
            launched.clear()
            grads = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
            )
            if j == 0:
                assert launched == _expected_launches(p, p.light_path)
                assert ("wy_bwd" in launched) is (
                    p.records[p.light_path]["wy"] == "wy_bwd"
                )
                assert not (wu and "wy_bwd" in launched)
                assert (launched[0] == "bwd_prep_g8_noqg_nokb") is wu
            # the set without its tables (the backward rebuilds them)
            bare = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=dataclasses.replace(saved, tables=None),
                workspace=ws,
            )
            if j == 0:
                # the tape set of the same problem: no version-4 tensors
                _, saved_t = cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
                )
                assert saved_t.qg is None and saved_t.w is None and saved_t.u is None

                def refused(match: str, other: CakeKDANondetChunkSaved) -> None:
                    with pytest.raises(ValueError, match=match):
                        cake_kda_nondet_train_backward(
                            q, k, v, g, beta, cu, dout, saved=other, workspace=ws
                        )

                refused("qg", dataclasses.replace(saved, qg=None))
                if wu:
                    refused("carries none", dataclasses.replace(saved, w=None, u=None))
                    refused("w and u together", dataclasses.replace(saved, u=None))
                    refused("saved.w", dataclasses.replace(saved, w=saved.w[:-BT]))
                else:
                    refused("w and u together", dataclasses.replace(saved, w=saved.qg))
                    refused(
                        "carries w / u",
                        dataclasses.replace(saved, w=saved.qg, u=saved.qg),
                    )
            arms = {row: {"out": out, **_named(grads)}, f"{row}.bare": _named(bare)}
            return arms, inputs

        _judge(record_property, ("light_saved_set", lengths, heads), draw)

    for lengths, heads in cases:
        check_row(lengths, heads)
    assert seen_wu == (3 if arch in WU_SAVED_ARCHES else 1)


def test_the_states_saving_set_of_a_segmented_16_head_row_feeds_the_plain_passes(
    device, manifest, seg_save_rule, record_property
) -> None:
    """On the segmented 16-head rows whose light set carries ``w`` / ``u`` (above 32 (document, head) chains on
    sm_103a / sm_107a) a pack at the size rule's threshold saves the segment-boundary states too: the backward
    from that set launches the kb-free prep, no WY program, the plain pass 3 and the plain adjoint recurrence
    (the saved representation feeds both; three documents at 16 heads are 48 chains, above the fused pass 3's
    bound) and the set stripped of its states launches the light path.  At the production rule's threshold
    (384 tokens per segment: 10k to 15k tokens at three documents, by the SM count) the launches are checked and
    every result (the set's, the replay, the stripped set's and the set on a tape-mode workspace) is finite.  The
    same route is judged on the threshold pack of the rule lowered to whole chunks per segment within
    :data:`JUDGED_TOKENS` (:func:`seg_save_rule`): the same launches on the first draw, and the set's forward and
    backward (arm ``""``), the replay, the stripped set and the set on a tape-mode workspace (``replay``,
    ``stripped``, ``tape_workspace``)."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    num_docs, heads = 3, 16
    route, _, nseg = cake_jit.segment_plan(num_docs, heads, sm, arch)
    if route != "seg" or not _wu_saved(num_docs, heads, sm, arch):
        pytest.skip(f"{arch}: three documents at 16 heads are a {route} row")
    launched: list[str] = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    def workspaces(S: int) -> tuple:
        """The light workspace of the threshold pack (its plan saves the states and reads the plain passes) and
        a tape-mode workspace of the same problem."""
        ws = CakeKDANondetWorkspace.for_problem(
            S, num_docs, heads, device, launcher=recording
        )
        p = ws.plan
        assert p.recur == "seg" and nseg == p.NSEG and p.seg_save is True
        assert p.light_path == "light_states"
        assert p.light_saves_qg is True and p.light_saves_wu is True
        records = p.records["light_states"]
        assert records["bwd_prep"] == "bwd_prep_g8_noqg_nokb" and records["wy"] is None
        assert records["fwdh_p3"] == "fwdh_p3" and not records["dhu"].endswith("_wt")
        tape_ws = CakeKDANondetWorkspace.for_problem(
            S, num_docs, heads, device, save_mode="tape"
        )
        return ws, tape_ws

    def run(inputs, ws, tape_ws, first: bool) -> dict:
        p = ws.plan
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu, workspace=ws)
        assert saved.hseg is not None and saved.seg_state is not None
        assert saved.qg is not None and saved.w is not None and saved.u is not None
        launched.clear()
        grads = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        if first:
            assert launched == _expected_launches(p, "light_states")
            assert launched[0] == "bwd_prep_g8_noqg_nokb" and "wy_bwd" not in launched
            assert "fwdh_p3" in launched
            assert p.records["light_states"]["dhu"] in launched
        replay = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=None, workspace=ws
        )
        stripped = dataclasses.replace(saved, hseg=None, seg_state=None)
        launched.clear()
        from_stripped = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=stripped, workspace=ws
        )
        if first:
            assert launched == _expected_launches(p, "light")
            assert launched[0] == "bwd_prep_g8_noqg_nokb" and "wy_bwd" not in launched
        from_tape_ws = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=tape_ws
        )
        return {
            "": {"out": out, **_named(grads)},
            "replay": _named(replay),
            "stripped": _named(from_stripped),
            "tape_workspace": _named(from_tape_ws),
        }

    # the production rule: the shortest pack that saves the states (too large to judge)
    S = cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    lengths = _split(S, num_docs)
    ws, tape_ws = workspaces(S)
    inputs = make_inputs(lengths, heads, device)
    _finite_arms(run(inputs, ws, tape_ws, True), (lengths, heads))
    del ws, tape_ws, inputs
    # the lowered rule: its shortest pack that saves the states, judged
    S = nseg * seg_save_rule(_judged_seg_save_tokens(nseg))
    lengths = _split(S, num_docs)
    ws, tape_ws = workspaces(S)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        return run(inputs, ws, tape_ws, j == 0), inputs

    _judge(record_property, ("states_saving_16_heads", lengths, heads), draw)


def test_the_two_level_carry_binds_its_group_operands(
    device, manifest, record_property
) -> None:
    """Where a path dispatches the two-level carry (every architecture, at most two heads, the rule's rows) its
    launch binds the carried states viewed as [NSEG * H, 128, 128] tiles, the plan's group size, the flag buffer
    (enough words for its grid) and a fresh epoch per launch, at the rule's grid; the chain carry binds none of
    them.  The forward and the backward of every two-level row are judged (``<row>``)."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    seen = 0

    def check_row(lengths, heads):
        nonlocal seen
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, sm, arch)
        choice = _carry_choice(N, heads, sm, p.NSEG)
        two_level = arch in CARRY_V8_ARCHES and choice is not None
        assert (p.records["light"]["seg_carry"] == "seg_carry_v8") is two_level
        assert (p.gsize > 0) is two_level
        if not two_level:
            assert p.records["light"]["seg_carry"] == "seg_carry"
            return
        seen += 1
        assert p.gsize == choice[0]
        row = _shape_id((lengths, heads))
        captured: list = []

        def recording(name, values, *, grid, device_index, _seen=captured):
            if name == "seg_carry_v8":
                _seen.append((dict(values), tuple(grid)))
            elif name.startswith("seg_carry"):
                assert not {"gsize", "carry_out_tiles"} & set(values), name
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, launcher=recording)
        fwd_carry = p.record("seg_carry", "fwd")
        expected = int(fwd_carry == "seg_carry_v8") + 2 * int(
            p.record("seg_carry", p.light_path) == "seg_carry_v8"
        )

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            captured.clear()
            epoch0 = ws._epoch[0]
            out, saved = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws
            )
            grads = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
            )
            assert len(captured) == expected >= 2, (lengths, heads)
            # one fresh epoch per stamped carry launch (the overlap pair's forward carry stamps one too)
            epochs = [values["epoch"] for values, _grid in captured]
            assert epochs == sorted(epochs) and len(set(epochs)) == len(epochs)
            assert ws._epoch[0] - epoch0 == len(captured) + int(
                fwd_carry == "seg_carry_ov"
            )
            if j == 0:
                for values, grid in captured:
                    assert (
                        grid == (choice[1], 1, 1) == p.grid("seg_carry", p.light_path)
                    )
                    assert values["gsize"] == p.gsize
                    assert values["flags"].data_ptr() == ws.flags.data_ptr()
                    assert values["flags"].numel() >= cake_jit.flag_words(grid[0])
                    tiles = values["carry_out_tiles"]
                    assert tiles.shape == (p.NSEG * heads, D, D)
                    assert tiles.data_ptr() == values["carry_out"].data_ptr()
            captured.clear()
            return {row: {"out": out, **_named(grads)}}, inputs

        _judge(record_property, ("two_level_carry", lengths, heads), draw)

    for lengths, heads in (
        ((129,), 1),
        ((63, 1, 65), 2),
        ((100, 0, 300), 2),
        ((63, 1, 65, 30), 2),
    ):
        check_row(lengths, heads)
    assert (seen > 0) is (arch in CARRY_V8_ARCHES)


# The size rule's smallest instances: one document at H = 1 and two documents at H = 2 have at most four chains,
# so P = SEG_P_SMALL (37 on sm_100a) on any device with at least 38 SMs, NSEG = P * num_docs and the threshold is
# 384 * NSEG tokens (14592 and 29184; 14208 and 28416 on sm_100a).
STATE_SHAPES = [(1, 1), (2, 2)]


def _split(S: int, num_docs: int) -> tuple[int, ...]:
    base = S // num_docs
    return tuple(base + (1 if i < S % num_docs else 0) for i in range(num_docs))


# The judged size of a pack: the tokens of one draw (the FP64 reference walks every token).
JUDGED_TOKENS = 4096


@pytest.fixture
def seg_save_rule():
    """``seg_save_rule(tokens)`` lowers the size rule to ``tokens`` per segment for one test and returns it, so a
    pack within the judged size takes the states-saving route (``light_states``): ``seg_states_saved`` reads
    ``SEG_SAVE_MIN_TOKENS_PER_SEG`` at each call.  The rule is a size choice only (below it the backward recomputes
    the same states).  The cached plans (a plan holds its rule) and the default entry's pooled workspaces (each
    holds a plan) are dropped when the rule changes and again when the test ends, with the rule restored."""
    original = cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG

    def drop_cached() -> None:
        cake_backend._plan.cache_clear()
        cake_backend._entry_pool.clear()

    def lower(tokens: int) -> int:
        assert 1 <= int(tokens) <= original, tokens
        cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG = int(tokens)
        drop_cached()
        return int(tokens)

    try:
        yield lower
    finally:
        cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG = original
        drop_cached()


def _judged_seg_save_tokens(nseg: int) -> int:
    """The lowered size rule of a judged pack of ``nseg`` segments: the most whole chunks per segment, at most the
    production rule's, whose threshold pack (``nseg`` segments at that many tokens) stays within
    :data:`JUDGED_TOKENS`; one chunk at least.  Read it before lowering the rule."""
    chunks = min(
        cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG // BT, JUDGED_TOKENS // (BT * nseg)
    )
    return BT * max(1, chunks)


def _retained_after_checkpointed_forward(fn, *leaves) -> int:
    """Bytes a forward of ``fn`` under non-reentrant ``torch.utils.checkpoint`` keeps allocated beyond its output:
    the measure of the activation-checkpointing report (0 expected: only the saved tensors go through the saved-
    tensor hooks, and anything else the node holds would be kept for every layer at once)."""
    from torch.utils.checkpoint import checkpoint

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = checkpoint(fn, *leaves, use_reentrant=False)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
    del out
    return kept


def test_a_checkpointed_forward_retains_no_workspace(
    device, manifest, record_property
) -> None:
    """Three ways to run the autograd entry under activation checkpointing keep nothing beyond the output: the
    implicit workspace, a per-call workspace created inside the checkpointed function (held by the node only
    weakly) and a caller-owned workspace kept outside it.  Judged: the explicit call (arm ``""``) and per way its
    eager output (``<way>.warm``) and the forward and gradients through the checkpoint (``<way>.ckpt``)."""
    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    assert plan.arena_bytes(backward=True) > 0
    shared = CakeKDANondetWorkspace(plan, device, backward=True)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        out, grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}
        leaves = _leaves(inputs)

        def implicit(q, k, v, g, beta):
            return cake_kda_nondet(q, k, v, g, beta, cu)

        def per_call(q, k, v, g, beta):
            ws = CakeKDANondetWorkspace(plan, device, backward=True)
            return cake_kda_nondet(q, k, v, g, beta, cu, workspace=ws)

        def on_shared(q, k, v, g, beta):
            return cake_kda_nondet(q, k, v, g, beta, cu, workspace=shared)

        for fn in (implicit, per_call, on_shared):
            # the programs are loaded and the allocator primed before the measurement
            arms[f"{fn.__name__}.warm"] = {"out": fn(*leaves).detach()}
            if j == 0:
                assert _retained_after_checkpointed_forward(fn, *leaves) == 0, (
                    fn.__name__
                )
            for leaf in leaves:
                leaf.grad = None
            o = checkpoint(fn, *leaves, use_reentrant=False)
            o.backward(dout)
            arms[f"{fn.__name__}.ckpt"] = {
                "out": o.detach(),
                **{n: t.clone() for n, t in _leaf_grads(leaves).items()},
            }
        return arms, inputs

    _judge(record_property, ("checkpointed_forward", lengths, heads), draw)


# The issue's pack: four documents of S = 261393 tokens, H = 4 (a segmented row whose light set saves the states).
ISSUE_LENGTHS, ISSUE_HEADS = (120001, 70707, 65537, 5148), 4


def _autograd_step(inputs, **kwargs):
    """One forward and backward through the autograd entry on fresh leaves: ``out`` (detached) and the five
    gradients in ``(dq, dk, dv, dbeta, dg)`` order.  The backward accumulates into the leaves: the graph task of
    ``torch.autograd.grad`` holds the gradients it returns until the autograd engine's device thread drops it,
    which can be after the call returned, so a memory check right after it would count them."""
    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    out = cake_kda_nondet(*leaves, inputs["cu_seqlens"], **kwargs)
    out.backward(inputs["dout"])
    dq, dk, dv, dg, dbeta = (leaf.grad for leaf in leaves)
    return out.detach(), (dq, dk, dv, dbeta, dg)


def test_the_default_entry_plans_a_repeated_problem_once_and_keeps_no_device_memory(
    device, manifest, monkeypatch, record_property
) -> None:
    """The default entry (no workspace) plans a repeated problem once and reuses its workspaces: no record is
    selected and no workspace is built again on a repeat, under every arena option.  Its arena, eye and
    ``doc_lengths`` copy live for the call alone, so the kept workspaces hold no device memory; a call on another
    stream runs in a workspace of its own.  Judged per arena option: a caller-owned workspace's step
    (``<option>.workspace``) and the default entry's first and repeated steps (``<option>.entry``, the worse of
    the two)."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    plan = cake_backend.plan(S, N, heads, cake_jit.sm_count(device.index))
    assert plan.arch == cake_jit.device_arch(device.index)
    options = (
        ("default", None, None),
        ("lean", cake_backend.CakeKDANondetArena(lean=True, tail_passes=4), None),
        (
            "batched",
            cake_backend.CakeKDANondetArena(
                lean=True, tail_passes=2, batch_tokens=1000
            ),
            lengths,
        ),
    )

    def step(inputs, **kwargs) -> dict:
        out, grads = _autograd_step(inputs, **kwargs)
        return {"out": out, **_named(grads)}

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        arms = {}
        for label, arena, doc in options:
            ws = CakeKDANondetWorkspace(
                plan,
                device,
                backward=True,
                arena=arena,
                doc_lengths=doc,
            )
            arms[f"{label}.workspace"] = step(inputs, workspace=ws, doc_lengths=doc)
            del ws
            # the first call builds the workspaces (on the first draw)
            first = step(inputs, arena=arena, doc_lengths=doc)
            selected: list = []
            built: list = []
            if j == 0:
                original = cake_jit.select_record

                def counting(*args, _sel=selected, _orig=original, **kwargs):
                    _sel.append(args)
                    return _orig(*args, **kwargs)

                monkeypatch.setattr(cake_jit, "select_record", counting)
                original_init = CakeKDANondetWorkspace.__init__

                def counting_init(
                    self, *args, _built=built, _orig=original_init, **kwargs
                ):
                    _built.append(args)
                    _orig(self, *args, **kwargs)

                monkeypatch.setattr(CakeKDANondetWorkspace, "__init__", counting_init)
            arms[f"{label}.entry"] = [first, step(inputs, arena=arena, doc_lengths=doc)]
            if j == 0:
                assert selected == [] and built == [], label
                monkeypatch.setattr(cake_jit, "select_record", original)
                monkeypatch.setattr(CakeKDANondetWorkspace, "__init__", original_init)
                # the baseline holds no kept workspace: building them again and reusing them leaves the device
                # memory as it was, so a kept workspace that held its arena, eye or doc_lengths copy would show here
                cake_backend._entry_pool.clear()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                for _ in range(2):
                    _autograd_step(inputs, arena=arena, doc_lengths=doc)
                torch.cuda.synchronize()
                assert cake_backend._entry_pool, label
                assert torch.cuda.memory_allocated() == before, label
        if j == 0:
            # a call on another stream takes a workspace of its own (no two queued calls share an eye)
            original_entry = cake_backend._entry_acquire
            created = _spy_call_workspaces(monkeypatch)
            keys = []
            spied = cake_backend._entry_acquire

            def keyed(*args, **kwargs):
                ws_, key = spied(*args, **kwargs)
                keys.append(key)
                return ws_, key

            monkeypatch.setattr(cake_backend, "_entry_acquire", keyed)
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side):
                _autograd_step(inputs)
            torch.cuda.current_stream(device).wait_stream(side)
            _autograd_step(inputs)
            monkeypatch.setattr(cake_backend, "_entry_acquire", original_entry)
            assert len(created) == 4 and len(set(keys)) == 4
            assert {key[1] for key in keys} == {
                side.cuda_stream,
                torch.cuda.current_stream(device).cuda_stream,
            }
        return arms, inputs

    _judge(record_property, ("default_entry", lengths, heads), draw)


def test_the_arena_options_never_synchronize_the_host(device, manifest) -> None:
    """The issue's pack through the default entry and through a caller-owned workspace: the lean arena with four
    tail passes and the batched option over two document groups (``doc_lengths``) run the forward and the backward
    without one host synchronization (PyTorch's sync debug mode raises on any it routes), on the first call of a
    problem as on a repeat, every result finite.  Both groups of the issue's pack are one run of the saved state
    blocks (views and contiguous copies); a second pack above the size rule has a group that is not (its block
    indices ride with the ``doc_lengths`` copy), with the same guarantees.  These packs (above 80k tokens) are too
    large to judge: the lean arena on a states-saving row is judged in
    :func:`test_lean_arena_serves_the_saved_states_and_the_tape_paths` (``n1-h1``), the batched groups in
    :func:`test_batched_arena_results_are_judged` and the batched groups of a states-saving row (one run and by
    index) in :func:`test_batched_arena_on_a_row_that_saves_the_states`."""
    split = _saving_pack(SPLIT_LENGTHS, 4, device)
    for lengths, heads, budget, runs in (
        (ISSUE_LENGTHS, ISSUE_HEADS, 200000, (True, True)),
        (split, 4, split[0] + split[1], (False, True)),
    ):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        # the default arena's call first: the programs are loaded before the measured calls
        out, grads = _autograd_step(inputs)
        _finite({"out": out, **_named(grads)}, (lengths, "default"))
        del out, grads
        plan = cake_backend.plan(S, N, heads, cake_jit.sm_count(device.index))
        assert plan.light_path == "light_states"
        lean = cake_backend.CakeKDANondetArena(lean=True, tail_passes=4)
        batched = cake_backend.CakeKDANondetArena(
            lean=True, tail_passes=4, batch_tokens=budget
        )
        assert len(plan.batches(batched, lengths)) == 2
        packing = cake_backend._host_packing(
            plan,
            batched,
            tuple(lengths),
            nbytes=plan.arena_bytes(arena=batched, doc_lengths=lengths),
            save_mode="light",
        )
        assert tuple(b is not None for b in packing.blocks) == runs
        options = (
            ((lean, None), (batched, lengths)) if runs[0] else ((batched, lengths),)
        )
        for arena, doc in options:
            ws = CakeKDANondetWorkspace(
                plan,
                device,
                backward=True,
                arena=arena,
                doc_lengths=doc,
            )
            for kwargs in (
                {"arena": arena},
                {"arena": arena},
                {"workspace": ws},
                {"workspace": ws},
            ):
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    out, grads = _autograd_step(inputs, doc_lengths=doc, **kwargs)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                tag = (lengths, arena, tuple(kwargs))
                _finite({"out": out, **_named(grads)}, tag)
                del out, grads
            del ws


def test_plan_without_arch_serves_a_workspace_on_the_device(
    device, manifest, record_property
) -> None:
    """A plan made without ``arch`` selects the device's records, so a workspace accepts it on every exported
    target (the plan of another architecture is refused, naming both).  Judged: the call on that workspace (arm
    ``""``) and the default entry's (``default``)."""
    lengths, heads = (100, 640, 77), 4
    S, N = sum(lengths), len(lengths)
    p = cake_backend.plan(S, N, heads, cake_jit.sm_count(device.index))
    assert p.arch == cake_jit.device_arch(device.index)
    ws = CakeKDANondetWorkspace(p, device)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        out, grads, _ = _run_fwd_bwd(inputs, workspace=ws)
        ref_out, ref_grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}, "default": {"out": ref_out, **ref_grads}}
        return arms, inputs

    _judge(record_property, ("plan_without_arch", lengths, heads), draw)
    for arch in ARCHES:
        if arch != p.arch:
            other = cake_backend.plan(S, N, heads, p.sm_count, arch)
            with pytest.raises(ValueError, match=f"records of {arch}, the device is"):
                CakeKDANondetWorkspace(other, device)


def test_a_batched_default_entry_under_capture_asks_for_a_warmed_workspace(
    device, manifest
) -> None:
    """Under CUDA-graph capture the default entry's workspace is the captured call's alone, so a batched arena
    would capture a copy of ``doc_lengths`` from a host buffer the graph does not own: it raises instead,
    naming the caller-owned workspace that ran these ``doc_lengths`` before the capture."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    args = [inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")]
    batched = cake_backend.CakeKDANondetArena(
        lean=True, tail_passes=2, batch_tokens=1000
    )
    # eager first: the programs are loaded and the default entry's own copy is made outside the capture
    cake_kda_nondet_train_forward(*args, arena=batched, doc_lengths=lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    refused = pytest.raises(RuntimeError, match="under CUDA-graph capture needs")
    with refused, torch.cuda.graph(graph):
        cake_kda_nondet_train_forward(*args, arena=batched, doc_lengths=lengths)


def test_the_node_holds_the_workspace_weakly(device, manifest, record_property) -> None:
    """A caller-owned workspace serves the backward (its launcher records the backward's launches); a per-call
    workspace dropped after the forward is freed at once (the node holds no strong reference), and the backward
    then runs in a workspace of its own.  Judged: the explicit call (arm ``""``), the autograd step on the kept
    workspace (``kept``) and on the dropped one (``dropped``)."""
    import gc
    import weakref

    lengths, heads = (63, 1, 65), 2
    S, N = sum(lengths), len(lengths)
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    launched: list[str] = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        out, grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}
        launched.clear()
        kept = CakeKDANondetWorkspace(plan, device, backward=True, launcher=recording)
        held = _leaves(inputs)
        o = cake_kda_nondet(*held, cu, workspace=kept)
        assert launched == _expected_launches(plan, "fwd")
        launched.clear()
        o.backward(dout)
        assert launched == _expected_launches(plan, plan.light_path)
        arms["kept"] = {"out": o.detach(), **_leaf_grads(held)}
        launched.clear()
        dropped = CakeKDANondetWorkspace(
            plan, device, backward=True, launcher=recording
        )
        alive = weakref.ref(dropped)
        held = _leaves(inputs)
        o = cake_kda_nondet(*held, cu, workspace=dropped)
        del dropped
        gc.collect()
        assert alive() is None  # the autograd node did not keep it
        launched.clear()
        o.backward(dout)
        # the backward ran in a workspace of its own, not through the dropped one
        assert launched == []
        arms["dropped"] = {"out": o.detach(), **_leaf_grads(held)}
        return arms, inputs

    _judge(record_property, ("node_holds_weakly", lengths, heads), draw)


def test_a_set_planned_for_other_sms_is_refused_by_a_backward_without_its_workspace(
    device, manifest
) -> None:
    """The node's saved set is checked against the plan of the backward that reads it.  A forward on a caller's
    workspace planned for half the device's SMs (fewer segments: a shorter ``seg_table`` and segment states of
    another ``NSEG``) whose workspace is gone at the backward: the backward plans for the device's SMs and refuses
    the set with the shape ``ValueError`` before any launch, as for any set of another geometry."""
    import gc
    import weakref

    lengths, heads = (32768,), 8
    S, N = sum(lengths), len(lengths)
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    own = cake_backend.plan(S, N, heads, sm, arch)
    half = cake_backend.plan(S, N, heads, sm // 2, arch)
    if own.recur != "seg" or half.recur != "seg" or own.NSEG == half.NSEG:
        pytest.skip(
            f"{arch} with {sm} SMs: no segmented plan of another NSEG at half the SMs"
        )
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta = (
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    )
    workspace = CakeKDANondetWorkspace(half, device, backward=True)
    alive = weakref.ref(workspace)
    out = cake_kda_nondet(q, k, v, g, beta, inputs["cu_seqlens"], workspace=workspace)
    del workspace
    gc.collect()
    assert alive() is None  # the backward plans its own workspace
    words = (own.NSEG + 2 * N + 1,)
    with pytest.raises(
        ValueError,
        match=re.escape(f"saved.tables.seg_table: expected shape {words}"),
    ):
        out.backward(inputs["dout"])


def test_a_saved_tensor_hook_after_the_forward_keeps_the_saved_set_checks(
    device, manifest
) -> None:
    """The node's backward checks every saved tensor it reads back, also one a per-tensor saved-tensor hook
    registered after the forward (``grad_fn._raw_saved_tensors[i]``) hands back: ``Aqk`` with another shape, dtype
    or layout is refused with the same ``ValueError`` as a caller's saved set of that kind."""
    lengths, heads = (63, 1, 65), 2
    S = sum(lengths)
    inputs = make_inputs(lengths, heads, device)
    aqk = 6  # the saved tensors: q, k, v, g, beta, cu_seqlens, then the set's Aqk
    cases = (
        (lambda t: t[:-1], f"saved.Aqk: expected shape {(S, heads, BT)}"),
        (lambda t: t.float(), "saved.Aqk: expected torch.bfloat16"),
        (
            lambda t: t.transpose(0, 1).contiguous().transpose(0, 1),
            "saved.Aqk: must be contiguous",
        ),
    )
    for unpack, message in cases:
        q, k, v, g, beta = (
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        )
        out = cake_kda_nondet(q, k, v, g, beta, inputs["cu_seqlens"])
        out.grad_fn._raw_saved_tensors[aqk].register_hooks(lambda t: t, unpack)
        with pytest.raises(ValueError, match=re.escape(message)):
            out.backward(inputs["dout"])


def test_arena_content_never_reaches_a_result(
    device, manifest, record_property
) -> None:
    """Write-before-read: every element a launch reads is written earlier in the same call, so with an arena
    filled with NaN bytes before every call every path gives judged results (the forward with and without a saved
    set, the backward from the light set, the replay, and the tape layouts) and the saved set it writes is finite;
    the arenas need no fill (``zero_init`` defaults to ``False``).  Judged per case: the zero-filled arena's calls
    (``<case>.zeroed``, ``.zeroed_replay``, ``.zeroed_nosave``) and the poisoned arena's (``<case>.poisoned``,
    ``.poisoned_replay``, ``.poisoned_nosave``)."""
    cases = (
        ((129,), 2, "light"),
        ((100, 640, 77, 1000, 129), 2, "light"),
        ((63, 1, 65, 30), 2, "light"),
        ((20, 20, 20, 20, 20, 20, 20), 8, "light"),
        ((129, 300), 16, "light"),  # the paired serial row of sm_100a: w / u saved
        ((100, 640, 77, 1000, 129), 2, "tape"),
    )

    def check_row(lengths, heads, save_mode):
        S, N = sum(lengths), len(lengths)
        case = f"{_shape_id((lengths, heads))}-{save_mode}"
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
        )
        zeroed = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            zero_init=True,
            save_mode=save_mode,
        )
        assert zeroed.zero_init is True and not bool(zeroed.arena.any())
        poisoned = CakeKDANondetWorkspace(
            plan, device, backward=True, save_mode=save_mode
        )
        assert poisoned.zero_init is False
        assert (
            CakeKDANondetWorkspace.for_problem(S, N, heads, device).zero_init is False
        )
        # the tape forward saves no version-4 rows (its backward recomputes the fold from the tape); the light
        # forward saves qg, and w / u on the rows whose light backward reads the WY representation from the set
        want_rows = (
            set()
            if save_mode == "tape"
            else ({"qg", "w", "u"} if plan.light_saves_wu else {"qg"})
        )

        def poison(_ws=poisoned):
            _ws.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
            torch.cuda.synchronize()

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            out, grads, saved = _run_fwd_bwd(inputs, workspace=zeroed)
            rows = {
                name for name in ("qg", "w", "u") if getattr(saved, name) is not None
            }
            assert rows == want_rows, case
            arms = {f"{case}.zeroed": {"out": out, **grads}}
            arms[f"{case}.zeroed_replay"] = _named(
                cake_kda_nondet_train_backward(
                    q, k, v, g, beta, cu, dout, saved=None, workspace=zeroed
                )
            )
            nosave, _ = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, save_for_backward=False, workspace=zeroed
            )
            arms[f"{case}.zeroed_nosave"] = {"out": nosave}
            poison()
            out, saved = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=poisoned
            )
            # the saved set the poisoned arena's forward writes: no NaN byte of the arena reached it
            written = {"Aqk": saved.Aqk, "Akk": saved.Akk}
            # the version-4 rows of the real chunks (the rows above are the chunk padding)
            real = slice(0, BT * _real_chunks(lengths))
            written.update((name, getattr(saved, name)[real]) for name in rows)
            _finite(written, (case, "saved set"))
            poison()
            grads = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=poisoned
            )
            arms[f"{case}.poisoned"] = {"out": out, **_named(grads)}
            poison()
            arms[f"{case}.poisoned_replay"] = _named(
                cake_kda_nondet_train_backward(
                    q, k, v, g, beta, cu, dout, saved=None, workspace=poisoned
                )
            )
            poison()
            nosave, _ = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, save_for_backward=False, workspace=poisoned
            )
            arms[f"{case}.poisoned_nosave"] = {"out": nosave}
            return arms, inputs

        _judge(record_property, ("arena_content", lengths, heads, save_mode), draw)

    for lengths, heads, save_mode in cases:
        check_row(lengths, heads, save_mode)


@pytest.mark.parametrize(("num_docs", "heads"), STATE_SHAPES, ids=["n1-h1", "n2-h2"])
def test_light_saved_set_carries_the_segment_states_above_the_size_rule(
    device, manifest, num_docs, heads, record_property
) -> None:
    """The size rule's smallest instances, at its threshold (the shortest pack that saves the segment-boundary
    states) and one token below it: (a) the light set carries the states exactly at and above the threshold, with
    the query's bytes, and saving them adds no launch; (b) the backward from the set runs pass 2 alone with the
    states (``light_states``) and the full recompute without them, and writes nothing into the set; the tape set is
    unchanged by the rule; (c) the autograd node packs and unpacks the states (or their absence), directly and
    under ``torch.utils.checkpoint``.  At the threshold of ``n1-h1`` (one document at one head, about 14.6k tokens)
    every result is judged: the set's backward (arm ``""``), the replay, the set stripped of its states, the set on
    a tape-mode workspace, the autograd node and the checkpoint (``replay``, ``stripped``, ``tape_workspace``,
    ``autograd``, ``ckpt``).  One token below it and on ``n2-h2`` (about 29k tokens) every result is finite; the
    recompute route below the threshold is judged on the shorter few-chain rows of
    :func:`test_forward_and_backward_match_the_exact_reference`."""
    from torch.utils.checkpoint import checkpoint

    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    nseg = cake_backend.plan(1, num_docs, heads, sm, arch).NSEG  # independent of S
    assert nseg == SEG_P_SMALL_BY_ARCH.get(arch, SEG_P_SMALL) * num_docs
    threshold = cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    block = nseg * heads * D * D * 4

    def run(inputs, S: int, saves: bool, ws, launched: list, first: bool) -> dict:
        q, k, v, g, beta, cu, dout = _ops(inputs)
        p = ws.plan
        launched.clear()
        # (a) the saved set: the states present exactly at and above the threshold, the bytes as the query says
        out, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu, workspace=ws)
        if first:
            # saving the states adds no launch
            assert launched == _expected_launches(p, "fwd")
            assert len(launched) == p.launch_count("fwd") == 5
            assert saved.version == cake_backend.SAVED_VERSION
            tables = p.table_words() * 4
            # the fold's qg on every light set (no w / u: these rows are no paired 16-head rows)
            assert (
                saved.qg.shape == (p.R, heads, D) and saved.qg.dtype == torch.bfloat16
            )
            assert saved.w is None and saved.u is None and not p.light_saves_wu
            arena_lo, arena_hi = ws.arena.data_ptr(), ws.arena.data_ptr() + ws.bytes
            assert not (arena_lo <= saved.qg.data_ptr() < arena_hi)
            qg_bytes = p.R * heads * D * 2
            expected_bytes = 256 * S * heads + tables + qg_bytes
            if saves:
                expected_bytes += 3 * block
            assert (
                cake_backend.saved_bytes(S, heads, num_docs, sm_count=sm)
                == expected_bytes
            )
            report = cake_backend.memory_report(S, num_docs, heads, sm_count=sm)
            assert report["seg_states_saved"] is saves
            assert report["saved_seg_state_bytes"] == (3 * block if saves else 0)
            assert report["saved"] == expected_bytes
            if not saves:
                assert saved.hseg is None and saved.seg_state is None
            else:
                assert saved.hseg.shape == (nseg, heads, D, D)
                assert saved.seg_state.shape == (2, nseg, heads, D, D)
                assert saved.hseg.dtype == saved.seg_state.dtype == torch.float32
                for t in (saved.hseg, saved.seg_state):
                    assert not (arena_lo <= t.data_ptr() < arena_hi)
                owned = (saved.Aqk, saved.Akk, saved.qg, saved.hseg, saved.seg_state)
                assert (
                    sum(t.numel() * t.element_size() for t in owned) + tables
                    == expected_bytes
                )
        if saves:
            states_before = (saved.hseg.clone(), saved.seg_state.clone())
        # (b) the backward from the set: the recurrence as pass 2 alone with the states, the full recompute without
        launched.clear()
        grads = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        if first:
            # the few-chain rows: with the states the fused pass 2 forms the WY representation (no WY launch) and
            # the adjoint recurrence reads its transposed w tile; without them the full recompute
            path = "light_states" if saves else "light"
            assert p.light_path == path
            assert launched == _expected_launches(p, path)
            # (the gate epilogue is one more launch on sm_107a)
            gate = 0 if p.gate_fused else 1
            assert len(launched) == p.launch_count("bwd") == (7 if saves else 10) + gate
            assert launched[1:3] == (
                ["fwdh_p3_wy", "dhu_v3s_f_wt"] if saves else ["wy_bwd", "fwdh_v3"]
            )
            assert ("dhu_v3s_f_wt" if saves else "dhu_v3s_f") in launched
        arms = {"": {"out": out, **_named(grads)}}
        arms["replay"] = _named(
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=None, workspace=ws
            )
        )
        stripped = dataclasses.replace(saved, hseg=None, seg_state=None)
        arms["stripped"] = _named(
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=stripped, workspace=ws
            )
        )
        tape_ws = CakeKDANondetWorkspace.for_problem(
            S, num_docs, heads, device, save_mode="tape"
        )
        arms["tape_workspace"] = _named(
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=tape_ws
            )
        )
        del tape_ws
        if saves:  # the backward writes nothing into the saved set (the same bytes before and after)
            assert torch.equal(saved.hseg, states_before[0])
            assert torch.equal(saved.seg_state, states_before[1])
        if first:
            # the tape set is unchanged by the rule: version 2 with seg_m, without the version-4 fields
            _, saved_t = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
            )
            assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
            assert (
                saved_t.seg_m is not None
                and saved_t.hseg is None
                and saved_t.seg_state is None
            )
            assert saved_t.qg is None and saved_t.w is None and saved_t.u is None
            del saved_t
        # (c) the autograd node packs and unpacks the states (or their absence); the checkpoint path re-runs the
        # forward
        for name, wrap in (("autograd", False), ("ckpt", True)):
            leaves = _leaves(inputs)
            if wrap:
                o = checkpoint(
                    lambda *a: cake_kda_nondet(*a, cu, workspace=ws),
                    *leaves,
                    use_reentrant=False,
                )
            else:
                o = cake_kda_nondet(*leaves, cu, workspace=ws)
            o.backward(dout)
            arms[name] = {"out": o.detach(), **_leaf_grads(leaves)}
        return arms

    for S, saves in ((threshold - 1, False), (threshold, True)):
        lengths = _split(S, num_docs)
        launched: list[str] = []

        def recording(name, values, *, grid, device_index, _log=launched):
            _log.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = CakeKDANondetWorkspace.for_problem(
            S, num_docs, heads, device, launcher=recording
        )
        p = ws.plan
        assert p.recur == "seg" and nseg == p.NSEG and p.seg_save is saves
        assert cake_backend.seg_states_saved(S, nseg) is saves
        if saves and num_docs == 1:

            def draw(j, s, _lengths=lengths, _S=S, _ws=ws, _log=launched):
                inputs = _draw(_lengths, heads, s, device)
                return run(inputs, _S, True, _ws, _log, j == 0), inputs

            _judge(record_property, ("segment_states", lengths, heads), draw)
        else:
            inputs = make_inputs(lengths, heads, device)
            _finite_arms(run(inputs, S, saves, ws, launched, True), (S, heads))
        del ws


def test_plan_limits_cover_the_kernels_int32_offsets() -> None:
    # The [R, H, 64] intra-chunk buffers are addressed with 32-bit row-head offsets: R * H * 64 < 2**31 over the
    # padded rows R = 64 * NC_pad.
    assert cake_backend.plan(2**25 - 192, 1, 1, 148).R * 1 * BT < 2**31
    with pytest.raises(ValueError, match="row-head"):
        cake_backend.plan(2**25 - 128, 1, 1, 148)
    assert cake_backend.plan(2**19 - 192, 1, 64, 148).R * 64 * BT < 2**31
    with pytest.raises(ValueError, match="row-head"):
        cake_backend.plan(2**19 - 128, 1, 64, 148)
    # The layout table finds a chunk's document with a 16-step search: at most MAX_DOCS documents.
    assert (
        cake_backend.plan(70000, cake_backend.MAX_DOCS, 1, 148).N
        == cake_backend.MAX_DOCS
    )
    with pytest.raises(ValueError, match="document search"):
        cake_backend.plan(70000, cake_backend.MAX_DOCS + 1, 1, 148)


def test_tape_workspace_serves_light_calls(device, manifest, record_property) -> None:
    """A saved set can go to a backward on any workspace of the same problem: a tape-mode arena (sized for the tape
    layouts) recomputes h / v_new into per-call buffers for a version-4 saved set, for the replay and for a forward
    that saves nothing, as its own forward does.  Judged: the light workspace's call (arm ``""``), the light forward
    and backward on the tape workspace (``tape_workspace``), its replay (``tape_replay``) and the forward without a
    saved set on a forward-only tape workspace (``forward_only``)."""
    lengths, heads = (63, 1, 65), 4
    S, N = sum(lengths), len(lengths)
    ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, save_mode="tape")
    fwd_only = CakeKDANondetWorkspace.for_problem(
        S, N, heads, device, backward=False, save_mode="tape"
    )

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}
        o, saved = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, save_mode="light"
        )
        assert saved.version == cake_backend.SAVED_VERSION
        got = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        arms["tape_workspace"] = {"out": o, **_named(got)}
        arms["tape_replay"] = _named(
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=None, workspace=ws
            )
        )
        o, none = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=fwd_only, save_for_backward=False
        )
        assert none is None
        arms["forward_only"] = {"out": o}
        return arms, inputs

    _judge(record_property, ("tape_workspace", lengths, heads), draw)


def test_workspace_accepts_an_unindexed_cuda_device(device, manifest) -> None:
    lengths, heads = (65,), 2
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
    )
    with torch.cuda.device(q.device):
        ws = CakeKDANondetWorkspace.for_problem(sum(lengths), 1, heads, "cuda")
    assert ws.device == q.device
    o, _ = cake_kda_nondet_train_forward(q, k, v, g, beta, cu, workspace=ws)
    assert o.shape == (sum(lengths), heads, D)


# --------------------------------------------------------------------------- the lower-memory arena option (device)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality: ``torch.equal`` on the integer views, so signed zeros and NaN payloads count."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _check_tail_grids(launched: list, p, path: str, tail_passes: int) -> None:
    """The tail's launches: the dispatched tail stages (two where the gate epilogue is fused, three on sm_107a)
    once per range, each with the grid of the plan's rule at the range's chunk count (one pass: the plan's
    grids)."""
    stages = [stage for stage in TAIL_STAGES if p.dispatched(stage, path)]
    tail_names = tuple(p.record(stage, path) for stage in stages)
    width = len(tail_names)
    assert width == 2 + int(not p.gate_fused)
    tail = [(name, grid) for name, grid in launched if name in tail_names]
    ranges = cake_backend._tail_ranges(p.NC_pad, tail_passes)
    assert len(tail) == width * len(ranges), (path, tail_passes)
    for i, (_c0, length) in enumerate(ranges):
        for j, (stage, name) in enumerate(zip(stages, tail_names, strict=True)):
            got_name, grid = tail[width * i + j]
            assert got_name == name
            want = cake_jit.launch_grid(
                name,
                num_chunks_pad=length,
                num_seqs=p.N,
                num_heads=p.H,
                sm_count=p.sm_count,
                num_segments=p.NSEG,
            )
            assert tuple(grid) == tuple(want), (name, i, grid, want)
            if tail_passes == 1:
                assert tuple(grid) == tuple(p.grid(stage, path))


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_lean_arena_gradients_are_judged(
    device, manifest, shape, record_property
) -> None:
    """The lower-memory option changes where the backward's intermediates live and how many launches the tail
    takes: for one, two and three tail passes a lean workspace has the byte queries' bytes (fewer than the
    default's) and serves no replay; it refuses the replay and a differing option.  Judged: the default arena's
    call (arm ``""``) and per tail-pass count ``t`` the forward and backward on a lean workspace (``lean<t>``), the
    default's saved set on a lean workspace, on a per-call lean arena and with both given (``lean<t>.default_set``,
    the worst of the three) and the autograd node carrying the option (``lean<t>.autograd``)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    default = CakeKDANondetWorkspace.for_problem(S, N, heads, device)
    assert default.arena_option == cake_backend.CakeKDANondetArena()
    p = default.plan
    lean = {}
    for t in (1, 2, 3):
        arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
        ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, arena=arena)
        assert ws.arena_option == arena and ws.arena.dtype == torch.uint8
        assert (
            ws.bytes
            == ws.arena.numel()
            == p.arena_bytes(arena=arena)
            == p.workspace_bytes("bwd_lean", t)
            == cake_backend.workspace_bytes(
                S, N, heads, sm_count=p.sm_count, arena=arena
            )
        )
        assert ws.bytes < default.bytes
        assert "replay" not in ws.paths() and set(ws.records()) == set(ws.paths())
        assert set(ws.paths()) == {"fwd", "fwd_nosave", "light", "tape"} | (
            {"light_states"} if p.recur == "seg" else set()
        )
        lean[t] = (arena, ws)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, grads, saved = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}
        for t, (arena, ws) in lean.items():
            o, grads_l, saved_l = _run_fwd_bwd(inputs, workspace=ws)
            assert saved_l.version == saved.version
            arms[f"lean{t}"] = {"out": o, **grads_l}
            # the default's saved set goes to a lean backward too; a per-call lean arena; both given
            arms[f"lean{t}.default_set"] = [
                _named(
                    cake_kda_nondet_train_backward(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        cu,
                        dout,
                        saved=saved,
                        workspace=ws_arg,
                        arena=arena_arg,
                    )
                )
                for ws_arg, arena_arg in ((ws, None), (None, arena), (ws, arena))
            ]
            if j == 0:
                # a lean arena serves no replay; a differing option on a given workspace is refused
                with pytest.raises(ValueError, match="lean=False"):
                    cake_kda_nondet_train_backward(
                        q, k, v, g, beta, cu, dout, saved=None, workspace=ws
                    )
                with pytest.raises(ValueError, match="lean=False"):
                    cake_kda_nondet_train_backward(
                        q, k, v, g, beta, cu, dout, saved=None, arena=arena
                    )
                with pytest.raises(ValueError, match="lean=False"):
                    ws.views("replay")
                with pytest.raises(ValueError, match="arena="):
                    cake_kda_nondet_train_backward(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        cu,
                        dout,
                        saved=saved,
                        workspace=ws,
                        arena=cake_backend.CakeKDANondetArena(),
                    )
                with pytest.raises(ValueError, match="arena="):
                    cake_kda_nondet_train_forward(
                        q, k, v, g, beta, cu, workspace=default, arena=arena
                    )
            # the autograd node carries the option: the backward's own workspace is lean
            leaves = _leaves(inputs)
            o = cake_kda_nondet(*leaves, cu, arena=arena)
            o.backward(dout)
            arms[f"lean{t}.autograd"] = {"out": o.detach(), **_leaf_grads(leaves)}
        return arms, inputs

    _judge(record_property, ("lean_arena", shape), draw)


def test_lean_tail_passes_launch_the_tail_once_per_range(
    device, manifest, record_property
) -> None:
    """The lean workspace's launch list is the default's with the tail stages once per chunk range, each at the
    grid of the plan's rule for the range's chunk count; one pass is the default list with the default grids.
    Judged per row: the default arena's call (``<row>``) and per tail-pass count ``t`` the forward and the backward
    from its set on a lean workspace (``<row>.lean<t>``) and the backward from the default's set without its tables
    (``<row>.lean<t>.bare``)."""

    def check_row(lengths, heads):
        S, N = sum(lengths), len(lengths)
        row = _shape_id((lengths, heads))
        launched: list = []

        def recording(name, values, *, grid, device_index, _log=launched):
            _log.append((name, tuple(grid)))
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        lean = {
            t: CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                arena=cake_backend.CakeKDANondetArena(lean=True, tail_passes=t),
                launcher=recording,
            )
            for t in (1, 2, 3, 5)
        }

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            out, grads, saved = _run_fwd_bwd(inputs)
            arms = {row: {"out": out, **grads}}
            bare = dataclasses.replace(saved, tables=None)
            for t, ws in lean.items():
                p = ws.plan
                launched.clear()
                o, saved_l = cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=ws
                )
                if j == 0:
                    assert [n for n, _ in launched] == _expected_launches(p, "fwd")
                launched.clear()
                got = cake_kda_nondet_train_backward(
                    q, k, v, g, beta, cu, dout, saved=saved_l, workspace=ws
                )
                if j == 0:
                    names = [n for n, _ in launched]
                    assert names == _expected_launches(p, "light", tail_passes=t), (
                        lengths,
                        heads,
                        t,
                    )
                    assert len(names) == p.launch_count("light") + len(
                        _tail_records(p, "light")
                    ) * (len(cake_backend._tail_ranges(p.NC_pad, t)) - 1)
                    _check_tail_grids(launched, p, "light", t)
                arms[f"{row}.lean{t}"] = {"out": o, **_named(got)}
                # the tail without the saved tables rebuilds them first, as the default does
                launched.clear()
                got = cake_kda_nondet_train_backward(
                    q, k, v, g, beta, cu, dout, saved=bare, workspace=ws
                )
                if j == 0:
                    assert [n for n, _ in launched] == _expected_launches(
                        p, "light", saved_tables=False, tail_passes=t
                    )
                arms[f"{row}.lean{t}.bare"] = _named(got)
            return arms, inputs

        _judge(record_property, ("lean_tail_passes", lengths, heads), draw)

    for lengths, heads in (
        ((1,), 1),  # NC_pad 2: three passes give two ranges of one chunk
        ((63, 1, 65), 2),
        ((129,), 16),
        ((63, 1, 65), 64),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((100, 640, 77, 1000, 129), 2),
    ):
        check_row(lengths, heads)


@pytest.mark.parametrize(("num_docs", "heads"), STATE_SHAPES, ids=["n1-h1", "n2-h2"])
def test_lean_arena_serves_the_saved_states_and_the_tape_paths(
    device, manifest, monkeypatch, num_docs, heads, record_property
) -> None:
    """On the segmented rows that save the segment-boundary states (``light_states``) and on the tape paths (a
    version-2 set on a lean tape workspace; a version-4 set on a lean tape workspace, which recomputes ``h`` /
    ``v_new`` into per-call buffers) the lean layouts launch the tail once per range; a per-call lean arena is
    sized by the saved set's version.  On ``n1-h1`` (one document at one head at the threshold, about 14.6k tokens)
    every result is judged: the default arena's call (arm ``""``) and per tail-pass count ``t`` the backward from
    the states set (``lean<t>``), the lean tape workspace's forward and backward (``lean<t>.tape``), the states set
    on it (``lean<t>.light_states_on_tape``) and the per-call lean arenas of both versions (``lean<t>.per_call_tape``,
    ``lean<t>.per_call_light``).  On ``n2-h2`` (about 29k tokens) every result is finite."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    nseg = cake_backend.plan(1, num_docs, heads, sm, arch).NSEG
    S = (
        cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    )  # the shortest pack that saves the states
    lengths = _split(S, num_docs)
    created = _spy_call_workspaces(monkeypatch)
    launched: list = []

    def recording(name, values, *, grid, device_index):
        launched.append((name, tuple(grid)))
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    def run(inputs, first: bool) -> dict:
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, grads, saved = _run_fwd_bwd(inputs)
        assert saved.version == cake_backend.SAVED_VERSION and saved.hseg is not None
        assert saved.qg is not None
        arms = {"": {"out": out, **grads}}
        for t in (1, 3):
            arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
            ws = CakeKDANondetWorkspace.for_problem(
                S,
                num_docs,
                heads,
                device,
                arena=arena,
                launcher=recording,
            )
            p = ws.plan
            assert p.recur == "seg" and p.seg_save and p.light_path == "light_states"
            launched.clear()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
            )
            if first:
                assert [n for n, _ in launched] == _expected_launches(
                    p, "light_states", tail_passes=t
                )
                _check_tail_grids(launched, p, "light_states", t)
            arms[f"lean{t}"] = _named(got)
            # the lean tape workspace: its own version-2 set (the tape path) and the version-4 set (per-call h /
            # v_new)
            tape_ws = CakeKDANondetWorkspace.for_problem(
                S,
                num_docs,
                heads,
                device,
                save_mode="tape",
                arena=arena,
                launcher=recording,
            )
            assert tape_ws.bytes == p.workspace_bytes("bwd_tape_lean", t) < ws.bytes
            launched.clear()
            o, saved_t = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=tape_ws
            )
            assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
            if first:
                assert [n for n, _ in launched] == _expected_launches(p, "fwd")
            launched.clear()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved_t, workspace=tape_ws
            )
            if first:
                tape_wy = cake_backend._tape_wy_record(p, device.index)
                assert [n for n, _ in launched] == _expected_launches(
                    p, "tape", tail_passes=t, tape_wy=tape_wy
                )
                _check_tail_grids(launched, p, "tape", t)
            arms[f"lean{t}.tape"] = {"out": o, **_named(got)}
            launched.clear()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, workspace=tape_ws
            )
            if first:
                assert [n for n, _ in launched] == _expected_launches(
                    p, "light_states", tail_passes=t
                )
            arms[f"lean{t}.light_states_on_tape"] = _named(got)
            # per-call lean arenas: sized by the saved set's version
            created.clear()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved_t, arena=arena
            )
            arms[f"lean{t}.per_call_tape"] = _named(got)
            assert len(created) == 1 and created[0].save_mode == "tape"
            assert (
                created[0].arena_option == arena and created[0].bytes == tape_ws.bytes
            )
            created.clear()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved, arena=arena
            )
            arms[f"lean{t}.per_call_light"] = _named(got)
            assert (
                len(created) == 1
                and created[0].save_mode == "light"
                and created[0].bytes == ws.bytes
            )
            created.clear()
            del ws, tape_ws
        return arms

    if num_docs == 1:

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            return run(inputs, j == 0), inputs

        _judge(record_property, ("lean_saved_states", lengths, heads), draw)
    else:
        _finite_arms(run(make_inputs(lengths, heads, device), True), (S, heads))


def test_lean_arena_under_checkpointing_and_a_dropped_workspace(
    device, manifest, monkeypatch, record_property
) -> None:
    """A checkpointed forward with the option retains nothing beyond its output, and the backward's own
    workspace (the forward's per-call one is gone, or the caller dropped theirs) takes the recorded option.
    Judged: the default arena's call (arm ``""``), the option's eager forward (``warm``), its checkpointed step
    (``ckpt``) and the step whose caller workspace was dropped (``dropped``)."""
    import gc

    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=2)
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    created = _spy_call_workspaces(monkeypatch)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        out, grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}

        def implicit(q, k, v, g, beta):
            return cake_kda_nondet(q, k, v, g, beta, cu, arena=arena)

        held = _leaves(inputs)
        arms["warm"] = {"out": implicit(*held).detach()}
        if j == 0:
            gc.collect()  # garbage of earlier tests must not be collected inside the measurement
            assert _retained_after_checkpointed_forward(implicit, *held) == 0
        created.clear()
        o = checkpoint(implicit, *held, use_reentrant=False)
        o.backward(dout)
        arms["ckpt"] = {"out": o.detach(), **_leaf_grads(held)}
        # every workspace of the checkpointed step took the option; the backward's is the lean backward layout
        assert created and all(w.arena_option == arena for w in created)
        backward_ws = [w for w in created if w.backward]
        assert backward_ws and all(
            w.bytes == plan.workspace_bytes("bwd_lean", 2) for w in backward_ws
        )
        # a caller's workspace dropped after the forward: the node recorded its option
        dropped = CakeKDANondetWorkspace(plan, device, backward=True, arena=arena)
        created.clear()
        held = _leaves(inputs)
        o = cake_kda_nondet(*held, cu, workspace=dropped)
        del dropped
        gc.collect()
        o.backward(dout)
        assert (
            len(created) == 1
            and created[0].arena_option == arena
            and created[0].backward
        )
        arms["dropped"] = {"out": o.detach(), **_leaf_grads(held)}
        if j == 0:
            # the option given with a kept workspace of another option is refused by the node's forward
            kept = CakeKDANondetWorkspace(plan, device, backward=True)
            with pytest.raises(ValueError, match="arena="):
                cake_kda_nondet(*_leaves(inputs), cu, workspace=kept, arena=arena)
        return arms, inputs

    _judge(record_property, ("lean_checkpointing", lengths, heads), draw)


def test_lean_arena_content_never_reaches_a_result(
    device, manifest, record_property
) -> None:
    """Write-before-read on the lean layouts too: the per-pass buffers and the offset views are written in the
    same call before the tail reads them, so with a lean arena filled with NaN bytes before every call the results
    are judged and the saved ``Aqk`` / ``Akk`` are finite (the forward, the backward from the light and the tape
    sets, one to three passes).  Judged per case: the zero-filled default arena's call (``<case>.zeroed``), the
    poisoned lean arena's forward and backward (``<case>.poisoned``), a version-4 set on the poisoned tape
    workspace (``<case>.light_set``) and its forward without a saved set (``<case>.nosave``)."""
    cases = (
        ((129,), 2, "light", 2),
        ((100, 640, 77, 1000, 129), 2, "light", 3),
        ((63, 1, 65, 30), 2, "light", 1),
        ((20, 20, 20, 20, 20, 20, 20), 8, "light", 2),
        ((63, 1, 65), 64, "light", 3),
        ((100, 640, 77, 1000, 129), 2, "tape", 2),
    )

    def check_row(lengths, heads, save_mode, t):
        S, N = sum(lengths), len(lengths)
        case = f"{_shape_id((lengths, heads))}-{save_mode}-t{t}"
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
        )
        zeroed = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            zero_init=True,
            save_mode=save_mode,
        )
        arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
        poisoned = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            save_mode=save_mode,
            arena=arena,
        )
        assert poisoned.zero_init is False and poisoned.bytes < zeroed.bytes

        def poison(_ws=poisoned):
            _ws.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
            torch.cuda.synchronize()

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            out, grads, _ = _run_fwd_bwd(inputs, workspace=zeroed)
            arms = {f"{case}.zeroed": {"out": out, **grads}}
            poison()
            o, saved_p = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=poisoned
            )
            _finite({"Aqk": saved_p.Aqk, "Akk": saved_p.Akk}, (case, "saved set"))
            poison()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved_p, workspace=poisoned
            )
            arms[f"{case}.poisoned"] = {"out": o, **_named(got)}
            if save_mode == "tape":
                # a version-4 set on the poisoned tape workspace: per-call h / v_new
                _, saved_light = cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=poisoned, save_mode="light"
                )
                poison()
                arms[f"{case}.light_set"] = _named(
                    cake_kda_nondet_train_backward(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        cu,
                        dout,
                        saved=saved_light,
                        workspace=poisoned,
                    )
                )
            poison()
            nosave, _ = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, save_for_backward=False, workspace=poisoned
            )
            arms[f"{case}.nosave"] = {"out": nosave}
            return arms, inputs

        _judge(
            record_property, ("lean_arena_content", lengths, heads, save_mode, t), draw
        )

    for lengths, heads, save_mode, t in cases:
        check_row(lengths, heads, save_mode, t)


# ------------------------------------------------------------------------------------------------------------------
# The batched arena option (document groups, CakeKDANondetArena.batch_tokens): host checks


def _partitions(lengths):
    """``(label, CakeKDANondetArena keywords)`` of the document groupings a test runs on a pack: two groups and three (the
    smallest token budget under which the greedy partition gives exactly that many groups, when one exists; the
    group count never grows with the budget, so a binary search finds it) and one document per group."""
    lengths = tuple(int(n) for n in lengths)
    S, longest = sum(lengths), max(lengths)
    out = []
    for k in (2, 3):
        lo, hi = longest, S
        while lo < hi:  # the smallest budget with at most k groups
            mid = (lo + hi) // 2
            if len(cake_backend.doc_batches(lengths, mid)) <= k:
                hi = mid
            else:
                lo = mid + 1
        if len(cake_backend.doc_batches(lengths, lo)) == k:
            out.append((f"g{k}", {"batch_tokens": lo}))
    out.append(("gdoc", {"batch_tokens": S, "max_docs_per_batch": 1}))
    return out


def test_doc_batches_partitions() -> None:
    db = cake_backend.doc_batches
    assert db((10, 20, 30), 60) == ((0, 3),)
    assert db((10, 20, 30), 30) == ((0, 2), (2, 3))
    assert db((10, 20, 30), 59) == ((0, 2), (2, 3))
    assert db((10, 20, 30), 60, 1) == ((0, 1), (1, 2), (2, 3))
    assert db((10, 20, 30), 60, 2) == ((0, 2), (2, 3))
    assert db((30, 30, 30, 30), 60) == ((0, 2), (2, 4))
    # a zero-length document joins the group before it (never a group of its own); leading ones open the first
    assert db((100, 0, 300), 300) == ((0, 2), (2, 3))
    assert db((100, 0, 300), 400, 1) == ((0, 2), (2, 3))
    assert db((0, 0, 100), 100, 1) == ((0, 3),)
    assert db((100, 0), 100, 1) == ((0, 2),)
    assert db((1,), 1) == ((0, 1),)
    assert db([10, 20], 25) == ((0, 1), (1, 2))
    with pytest.raises(ValueError, match="document 2 has 30 tokens"):
        db((10, 20, 30), 29)
    with pytest.raises(ValueError, match="no token"):
        db((0, 0), 5)
    with pytest.raises(ValueError, match="non-negative"):
        db((), 5)
    with pytest.raises(ValueError, match="non-negative"):
        db((1, -1), 5)
    # the search helper of these tests: exact group counts, one document per group
    parts = dict(_partitions((23496, 88510, 114974, 34413)))
    assert (
        len(cake_backend.doc_batches((23496, 88510, 114974, 34413), **parts["g2"])) == 2
    )
    assert (
        len(cake_backend.doc_batches((23496, 88510, 114974, 34413), **parts["g3"])) == 3
    )
    assert parts["gdoc"]["max_docs_per_batch"] == 1


def test_arena_option_validation_batched() -> None:
    CakeKDANondetArena = cake_backend.CakeKDANondetArena
    option = CakeKDANondetArena(batch_tokens=1000)
    assert option.batched and not option.lean and option.max_docs_per_batch is None
    assert CakeKDANondetArena(
        lean=True, tail_passes=4, batch_tokens=10, max_docs_per_batch=1
    ).batched
    assert (
        not CakeKDANondetArena().batched
        and not CakeKDANondetArena(lean=True, tail_passes=3).batched
    )
    for bad in (0, -1, True, 1.5, "3"):
        with pytest.raises(ValueError, match="batch_tokens"):
            CakeKDANondetArena(batch_tokens=bad)
    with pytest.raises(ValueError, match="max_docs_per_batch"):
        CakeKDANondetArena(batch_tokens=10, max_docs_per_batch=0)
    with pytest.raises(ValueError, match="batch_tokens too"):
        CakeKDANondetArena(max_docs_per_batch=1)


def test_batch_plans_pin_the_whole_problems_records_and_segments() -> None:
    """The groups of a pack: contiguous, covering, each with its own geometry (at or above its real chunk count; the
    tape bound even and inside the whole problem's tape), a plan with the whole problem's records, ``P``, route,
    gate fusion and two-level carry group size, and the grids of the records' rules at the group's scalars (the
    two-level carry's at the pinned group size: ``N_b * H * ctas(P, gsize)`` CTAs); the arena of the exact groups
    never exceeds the bound of every group, which the byte query gives without ``doc_lengths``."""
    sm = 148
    pinned = 0
    for arch in ARCHES:
        for lengths, heads in (
            ((23496, 88510, 114974, 34413), 4),
            ((100, 0, 300), 2),
            ((63, 1, 65), 64),
            ((2000,) * 300, 1),
            ((20,) * 7, 8),
        ):
            S, N = sum(lengths), len(lengths)
            p = cake_backend.plan(S, N, heads, sm, arch)
            for _label, kw in _partitions(lengths):
                arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=4, **kw)
                batches = p.batches(arena, lengths)
                assert [(b.d0, b.d1) for b in batches] == list(
                    cake_backend.doc_batches(
                        lengths, arena.batch_tokens, arena.max_docs_per_batch
                    )
                )
                assert batches[0].d0 == 0 and batches[-1].d1 == N
                assert all(
                    a.d1 == b.d0 for a, b in zip(batches, batches[1:], strict=False)
                )
                c0 = 0
                for b in batches:
                    assert (b.S, b.N, b.c0) == (
                        sum(lengths[b.d0 : b.d1]),
                        b.d1 - b.d0,
                        c0,
                    )
                    real = sum(-(-n // BT) for n in lengths[b.d0 : b.d1])
                    _nc_max, nc_pad, _rows = cake_backend._geometry(b.S, b.N)
                    assert b.NC_pad == nc_pad >= real
                    assert b.NC_pad_tape % 2 == 0 and real <= b.NC_pad_tape <= b.NC_pad
                    assert c0 + b.NC_pad_tape <= p.NC_pad
                    c0 += real
                    for tape in (False, True):
                        pb = p.batch_plan(b, tape=tape)
                        assert pb.records is p.records and (pb.P, pb.recur) == (
                            p.P,
                            p.recur,
                        )
                        assert pb.proportional == 0
                        assert pb.NC_pad == (b.NC_pad_tape if tape else b.NC_pad)
                        assert BT * pb.NC_pad == pb.R and pb.NC_max <= pb.NC_pad
                        assert (b.N * p.P if p.recur == "seg" else 0) == pb.NSEG
                        assert pb.seg_arg == (p.P if p.recur == "seg" else p.seg_arg)
                        assert (pb.gsize, pb.gate_fused) == (p.gsize, p.gate_fused)
                        assert set(pb.grids) == set(p.grids)
                        for name, grids in pb.grids.items():
                            assert grids == tuple(
                                cake_jit.launch_grid(
                                    name,
                                    num_chunks_pad=pb.NC_pad,
                                    num_seqs=pb.N,
                                    num_heads=heads,
                                    sm_count=sm,
                                    num_segments=pb.NSEG,
                                    launch=launch,
                                    gsize=p.gsize,
                                )
                                for launch in range(cake_jit.launches(name))
                            )
                        if "seg_carry_v8" in pb.grids:
                            pinned += 1
                            assert p.gsize == _carry_choice(N, heads, sm, p.NSEG)[0]
                            ctas = _carry_ctas(p.P, p.gsize)
                            group_grid = pb.grids["seg_carry_v8"]
                            assert group_grid == ((b.N * heads * ctas, 1, 1),)
                            assert group_grid[0][0] <= p.grids["seg_carry_v8"][0][0]
                exact = p.arena_bytes(arena=arena, doc_lengths=lengths)
                bound = p.arena_bytes(arena=arena)
                nc_bar, rows_bar, nseg_bar = p.batch_bound(arena)
                assert rows_bar == BT * nc_bar and nseg_bar == p.NSEG
                assert all(b.NC_pad <= nc_bar for b in batches)
                assert (
                    exact
                    <= bound
                    <= p.arena_bytes(
                        arena=cake_backend.CakeKDANondetArena(lean=True, tail_passes=4)
                    )
                )
                assert exact == max(
                    cake_backend._arena_bytes(
                        b.NC_pad * BT,
                        heads,
                        b.NC_pad,
                        p.batch_segments(b),
                        backward=True,
                        save_mode="light",
                        arena=arena,
                    )
                    for b in batches
                )
                assert (
                    cake_backend.workspace_bytes(
                        S,
                        N,
                        heads,
                        sm_count=sm,
                        arch=arch,
                        arena=arena,
                        doc_lengths=lengths,
                    )
                    == exact
                )
                assert (
                    cake_backend.workspace_bytes(
                        S, N, heads, sm_count=sm, arch=arch, arena=arena
                    )
                    == bound
                )
                assert (
                    p.arena_bytes(backward=False, arena=arena, doc_lengths=lengths)
                    <= p.arena_bytes(backward=False, arena=arena)
                    <= p.arena_bytes(backward=False)
                )
                tape_exact = p.arena_bytes(
                    save_mode="tape", arena=arena, doc_lengths=lengths
                )
                assert tape_exact <= p.arena_bytes(save_mode="tape", arena=arena)
            with pytest.raises(ValueError, match="doc_lengths"):
                p.batches(cake_backend.CakeKDANondetArena(batch_tokens=S), None)
            with pytest.raises(ValueError, match="doc_lengths"):
                p.batches(cake_backend.CakeKDANondetArena(batch_tokens=S), lengths[:-1])
            with pytest.raises(ValueError, match="doc_lengths"):
                p.check_doc_lengths(tuple(lengths[:-1]) + (lengths[-1] + 1,))
            with pytest.raises(ValueError, match="above batch_tokens"):
                p.batches(
                    cake_backend.CakeKDANondetArena(batch_tokens=max(lengths) - 1),
                    lengths,
                )
    # the six-chain row at 2 heads pins the two-level carry's group size (on every architecture)
    assert pinned > 0


def test_batched_arena_refuses_the_proportional_rows() -> None:
    for arch in ARCHES:
        p = cake_backend.plan(
            128, 2, 1, 148, arch
        )  # two chains: proportional segment counts
        assert p.recur == "seg" and p.proportional == 1
        arena = cake_backend.CakeKDANondetArena(batch_tokens=64)
        with pytest.raises(ValueError, match="proportional"):
            p.check_arena(arena)
        with pytest.raises(ValueError, match="proportional"):
            p.check_arena(arena, backward=False)
        with pytest.raises(ValueError, match="proportional"):
            p.arena_bytes(arena=arena, doc_lengths=(64, 64))
        with pytest.raises(ValueError, match="proportional"):
            cake_backend.workspace_bytes(
                128, 2, 1, sm_count=148, arch=arch, arena=arena
            )
        with pytest.raises(ValueError, match="proportional"):
            cake_backend.memory_report(128, 2, 1, sm_count=148, arch=arch, arena=arena)
        p5 = cake_backend.plan(320, 5, 1, 148, arch)  # five chains are not
        assert p5.recur == "seg" and p5.proportional == 0
        p5.check_arena(arena)
        p64 = cake_backend.plan(128, 2, 64, 148, arch)  # the serial route never is
        assert p64.recur == "serial" and p64.proportional == 0
        p64.check_arena(arena)


def test_arena_for_budget_search_order() -> None:
    lengths, heads = (23496, 88510, 114974, 34413), 4
    S, N = sum(lengths), len(lengths)
    A = cake_backend.CakeKDANondetArena
    for arch in ARCHES:
        p = cake_backend.plan(S, N, heads, 148, arch)
        steps = [
            A(),
            A(lean=True),
            A(lean=True, tail_passes=2),
            A(lean=True, tail_passes=4),
        ]
        sizes = [p.arena_bytes(arena=a) for a in steps]
        assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == 4
        for a, nbytes in zip(steps, sizes, strict=True):
            assert cake_backend.arena_for_budget(p, nbytes) == a
            assert cake_backend.arena_for_budget(p, nbytes, doc_lengths=lengths) == a
        below = sizes[-1] - 1
        with pytest.raises(ValueError, match="doc_lengths"):
            cake_backend.arena_for_budget(p, below)
        got = cake_backend.arena_for_budget(p, below, doc_lengths=lengths)
        assert got == A(
            lean=True, tail_passes=4, batch_tokens=max(max(lengths), S // 2)
        )
        assert p.arena_bytes(arena=got, doc_lengths=lengths) <= below
        single = A(
            lean=True, tail_passes=4, batch_tokens=max(lengths), max_docs_per_batch=1
        )
        floor = p.arena_bytes(arena=single, doc_lengths=lengths)
        got = cake_backend.arena_for_budget(p, floor, doc_lengths=lengths)
        assert got.batched and p.arena_bytes(arena=got, doc_lengths=lengths) <= floor
        with pytest.raises(ValueError, match="floor"):
            cake_backend.arena_for_budget(p, floor - 1, doc_lengths=lengths)
        # the dav rows skip the lean steps and batch the default layouts; the proportional rows refuse batching
        p12 = cake_backend.plan(S, N, 12, 148, arch)
        assert cake_backend.arena_for_budget(p12, p12.arena_bytes()) == A()
        got = cake_backend.arena_for_budget(
            p12, p12.arena_bytes() - 1, doc_lengths=lengths
        )
        assert got.batched and not got.lean and got.tail_passes == 1
        p2 = cake_backend.plan(128, 2, 1, 148, arch)
        with pytest.raises(ValueError, match="refuses the batched option"):
            cake_backend.arena_for_budget(p2, 1, doc_lengths=(64, 64))


def test_memory_report_batched_fields() -> None:
    lengths, heads = (23496, 88510, 114974, 34413), 4
    S, N = sum(lengths), len(lengths)
    for sm in (148, 152):
        for arch in ARCHES:
            base = cake_backend.memory_report(S, N, heads, sm_count=sm, arch=arch)
            assert base["batch_tokens"] is None and base["batches"] is None
            assert (
                base["arena_bwd_bound_batched"] is None
                and base["arena_floor_batched"] is None
            )
            arena = cake_backend.CakeKDANondetArena(
                lean=True, tail_passes=4, batch_tokens=120000
            )
            p = cake_backend.plan(S, N, heads, sm, arch)
            r = cake_backend.memory_report(
                S, N, heads, sm_count=sm, arch=arch, arena=arena
            )
            assert r["batch_tokens"] == 120000 and r["max_docs_per_batch"] is None
            assert r["batches"] is None and r["batch_scratch"] is None
            assert (
                r["arena_bwd_configured"]
                == r["arena_bwd_bound_batched"]
                == p.arena_bytes(arena=arena)
            )
            assert r["largest_batch_rows"] == p.batch_bound(arena)[1]
            assert r["bwd_total"] == r["arena_bwd_configured"] + r["out_bwd"]
            r2 = cake_backend.memory_report(
                S, N, heads, sm_count=sm, arch=arch, arena=arena, doc_lengths=lengths
            )
            groups = p.batches(arena, lengths)
            assert r2["batches"] == [(b.d0, b.d1, b.S, b.N) for b in groups]
            assert r2["batches"] == [
                (0, 2, 112006, 2),
                (2, 3, 114974, 1),
                (3, 4, 34413, 1),
            ]
            assert r2["arena_bwd_configured"] == p.arena_bytes(
                arena=arena, doc_lengths=lengths
            )
            assert r2["arena_bwd_configured"] <= r2["arena_bwd_bound_batched"]
            assert r2["largest_batch_rows"] == max(b.NC_pad for b in groups) * BT
            assert r2["arena_fwd_configured"] == p.arena_bytes(
                backward=False, arena=arena, doc_lengths=lengths
            )
            # the largest table buffer of a group and three state blocks per segment of the group with the most
            # segments (here the two-document group, not the one with the most chunks)
            segments_b = max(p.batch_segments(b) for b in groups)
            if p.recur == "seg":
                assert (
                    segments_b
                    == p.batch_segments(groups[0])
                    > max(p.batch_segments(b) for b in groups[1:])
                )
            blocks = 3 * segments_b * heads * D * D * 4
            words = max(
                2 * b.NC_pad + 2 * b.N + 2 + p.batch_segments(b) + 2 * b.N + 1 + 5 * 64
                for b in groups
            )
            assert blocks <= r2["batch_scratch"] <= blocks + 4 * words
            nc_min = -(-max(lengths) // BT) + 1
            rows_min = BT * (nc_min + nc_min % 2)
            block_min = p.P * heads * D * D * 4 if p.recur == "seg" else 0
            assert r2["arena_floor_batched"] == 2820 * rows_min * heads + 3 * block_min
            single = cake_backend.CakeKDANondetArena(
                lean=True,
                tail_passes=4,
                batch_tokens=max(lengths),
                max_docs_per_batch=1,
            )
            exact_floor = p.arena_bytes(arena=single, doc_lengths=lengths)
            # the exact one-document arena: the floor plus the ceil slack of the last tail range
            assert 0 <= exact_floor - r2["arena_floor_batched"] <= 2568 * 3 * BT * heads
            tape = cake_backend.memory_report(
                S,
                N,
                heads,
                "tape",
                sm_count=sm,
                arch=arch,
                arena=arena,
                doc_lengths=lengths,
            )
            assert (
                tape["arena_floor_batched"] == 2052 * rows_min * heads + 3 * block_min
            )
            assert tape["arena_bwd_configured"] == p.arena_bytes(
                save_mode="tape", arena=arena, doc_lengths=lengths
            )


# ------------------------------------------------------------------------------------------------------------------
# The batched arena option: device checks

# the rows with at least two documents and more than four (document, head) chains (the proportional rows refuse the
# option), plus serial rows (16, 32 and 64 heads)
D_ROWS = [s for s in SHAPES if len(s[0]) >= 2 and len(s[0]) * s[1] > 4] + [
    ((63, 1, 65), 64),
    ((129, 300), 16),
    ((20, 20, 20, 20, 20, 20, 20), 32),
]


def _group_rows_sliced(path: str) -> bool:
    """Whether a batched call hands every group slices of whole-problem row tensors (the saving forward's qg / w /
    u or tape, the saved-set backward's saved rows): the groups then take the tape geometry
    (``batch_view(tape=True)``, the group's chunk bound inside the whole problem's rows).  The forward without a
    saved set and the replay slice nothing."""
    return path not in ("fwd_nosave", "replay")


def _fewest_tail_passes(pb, arena, nbytes, save_mode="light") -> int:
    """The tail passes of a group's backward (group plan ``pb``) on an arena of ``nbytes``, the package's rule
    restated: the fewest up to the option's ``tail_passes`` whose lean layouts fit (``bwd_tape_lean`` on a
    tape-mode workspace, also ``bwd_lean`` on a light one), else the option's."""
    modes = ("bwd_tape_lean",) if save_mode == "tape" else ("bwd_lean", "bwd_tape_lean")
    return next(
        (
            t
            for t in range(1, arena.tail_passes)
            if all(pb.workspace_bytes(mode, t) <= nbytes for mode in modes)
        ),
        arena.tail_passes,
    )


def _group_passes(p, arena, lengths, path, *, nbytes, save_mode="light") -> list[int]:
    """Per group of a batched call on ``path``, the tail passes of its backward (:func:`_fewest_tail_passes`)."""
    return [
        _fewest_tail_passes(
            p.batch_plan(b, tape=_group_rows_sliced(path)), arena, nbytes, save_mode
        )
        for b in p.batches(arena, lengths)
    ]


def _expected_batched_launches(p, arena, lengths, path, *, passes=None):
    """The launch order of a batched call: on the saving forward the whole problem's tables by the saved-set
    paths' table program, then every group's forward list (its own tables where the row dispatches a table program
    on the forward); on the backward every group's list with the group's tables first (the saved set's describe the
    whole problem) and the tail once per range of the group's chunk bound at the group's tail passes (``passes``,
    one per group: :func:`_group_passes`; ``None``: one pass each)."""
    seq = []
    batches = p.batches(arena, lengths)
    passes = [1] * len(batches) if passes is None else passes
    if path in ("fwd", "fwd_nosave"):
        if path == "fwd" and p.dispatched("table", "light"):
            seq.append(p.record("table", "light"))
            if p.recur == "seg" and p.dispatched("seg_table", "light"):
                seq.append("seg_table")
        for b in batches:
            pb = p.batch_plan(b, tape=_group_rows_sliced(path))
            seq += _expected_launches(pb, path)
        return seq
    for b, t in zip(batches, passes, strict=True):
        seq += _expected_launches(
            p.batch_plan(b, tape=_group_rows_sliced(path)),
            path,
            saved_tables=False,
            tail_passes=t,
            finalize=False,
        )
    # the raw mode's parameter finalize: once, after the last group, over the whole problem's partials
    if p.dispatched("param_finalize", path):
        seq.append(p.record("param_finalize", path))
    return seq


def _check_group_launches(launched, p, arena, lengths, path, *, passes=None) -> None:
    """Every launch of a group carries the group's scalars (``num_seqs``, ``num_chunks_pad``) and one of the
    group plan's grids; the tail passes (``passes``, one per group, ``None``: one pass each) carry
    ``num_seqs = 0`` and their range's chunk count."""
    batches = p.batches(arena, lengths)
    passes = [1] * len(batches) if passes is None else passes
    forward = path in ("fwd", "fwd_nosave")
    pos = 0
    if forward and path == "fwd" and p.dispatched("table", "light"):
        head = 2 if (p.recur == "seg" and p.dispatched("seg_table", "light")) else 1
        for name, grid, num_seqs, nc in launched[:head]:
            assert (num_seqs, nc) == (p.N, p.NC_pad) and grid in p.grids[name], name
        pos = head
    for b, t in zip(batches, passes, strict=True):
        pb = p.batch_plan(b, tape=_group_rows_sliced(path))
        expected = (
            _expected_launches(pb, path)
            if forward
            else _expected_launches(
                pb, path, saved_tables=False, tail_passes=t, finalize=False
            )
        )
        group = launched[pos : pos + len(expected)]
        pos += len(expected)
        assert [n for n, *_ in group] == expected
        tail_names = set() if forward else set(_tail_records(pb, path))
        width = len(tail_names)
        ranges = cake_backend._tail_ranges(pb.NC_pad, t)
        tails = []
        for name, grid, num_seqs, nc in group:
            if name in tail_names and t > 1:
                tails.append((name, grid, num_seqs, nc))
                continue
            assert (num_seqs, nc) == (pb.N, pb.NC_pad), (name, b)
            assert grid in pb.grids[name], (name, b, grid, pb.grids[name])
        if tails:
            assert len(tails) == width * len(ranges)
            for i, (_c0, length) in enumerate(ranges):
                for j in range(width):
                    name, grid, num_seqs, nc = tails[width * i + j]
                    assert (num_seqs, nc) == (0, length), (name, b)
                    want = cake_jit.launch_grid(
                        name,
                        num_chunks_pad=length,
                        num_seqs=pb.N,
                        num_heads=pb.H,
                        sm_count=pb.sm_count,
                        num_segments=pb.NSEG,
                    )
                    assert tuple(grid) == tuple(want), (name, b, i)
    # the raw mode's parameter finalize closes the call: once, one CTA per (head, segment of 128 chunks)
    if not forward and p.dispatched("param_finalize", path):
        name, grid, *_ = launched[pos]
        assert name == p.record("param_finalize", path)
        assert tuple(grid) == tuple(p.grid("param_finalize", path))
        assert tuple(grid) == (p.H, max(1, -(-p.NC_pad // 128)), 1)
        pos += 1
    assert pos == len(launched)


def _check_tail_counts(counts, p, arena, lengths, path, *, passes=None) -> None:
    """The one-entry chunk counts of a batched backward's tail launches (``seq_chunk_start`` at ``num_seqs = 0``, in
    launch order): every launch of a group's range ``c0`` reads ``clamp(real_b - c0, 0, ceil(NC_pad_b / b))`` at
    the group's tail passes ``b`` (``passes``, one per group, ``None``: one pass each), the device rule the host
    reproduces from ``doc_lengths``; none for a group in one pass."""
    batches = p.batches(arena, lengths)
    passes = [1] * len(batches) if passes is None else passes
    torch.cuda.synchronize()
    pos = 0
    for b, t in zip(batches, passes, strict=True):
        if t == 1:
            continue
        pb = p.batch_plan(b, tape=_group_rows_sliced(path))
        width = len(_tail_records(pb, path))
        real = _real_chunks(lengths[b.d0 : b.d1])
        ranges = cake_backend._tail_ranges(pb.NC_pad, t)
        for c0, _length in ranges:
            want = min(max(real - c0, 0), ranges[0][1])
            for _ in range(width):
                assert counts[pos].tolist() == [want], (b, c0)
                pos += 1
    assert pos == len(counts)
    assert (pos > 0) == any(t > 1 for t in passes)


def _real_chunks(lengths) -> int:
    return sum(-(-int(n) // BT) for n in lengths)


@pytest.mark.parametrize("shape", D_ROWS, ids=_shape_id)
def test_batched_arena_results_are_judged(
    device, manifest, shape, monkeypatch, record_property
) -> None:
    """Two groups, three and one document per group, with and without the lean layouts: the batched workspace has
    the exact groups' bytes; the batched forward writes the whole problem's tables and the default's saved fields
    (the states, ``qg`` and ``w`` / ``u`` where the default saves them, finite on the real token rows), the
    batched tape forward writes every row of the saved ``v_new`` (its allocation starts NaN-filled; the pad rows
    too) and the real rows of ``h``; a lean workspace refuses the replay; ``doc_lengths`` are required and checked.
    Judged: the default call (arm ``""``) and per grouping and layout (``<grouping>`` or ``<grouping>-lean``) the
    batched forward (``.forward``), the batched backward from the default's set and from its own
    (``.default_set``, ``.batched_set``), the default backward from the batched set (``.default_backward``), a
    per-call arena (``.per_call``), the replay group by group (``.replay``, not lean), the tape forward
    (``.tape_forward``), the tape workspace's backward from its own and the default's tape set and from the batched
    light set (``.own_tape_set``, ``.default_tape_set``, ``.light_set_on_tape``), the default backward from the
    batched tape set (``.default_from_tape``), the autograd node (``.autograd``) and the forward without a saved set
    (``.nosave``)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    default = CakeKDANondetWorkspace.for_problem(S, N, heads, device)
    tape_default = CakeKDANondetWorkspace.for_problem(
        S, N, heads, device, save_mode="tape"
    )
    p = default.plan
    real = _real_chunks(lengths)
    rows = slice(0, BT * real)
    options = []
    for label, kw in _partitions(lengths):
        for lean in (False, True):
            arena = cake_backend.CakeKDANondetArena(
                lean=lean, tail_passes=3 if lean else 1, **kw
            )
            ws = CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                arena=arena,
                doc_lengths=lengths,
            )
            assert ws.arena_option == arena and ws.doc_lengths == tuple(lengths)
            assert ws.bytes == p.arena_bytes(arena=arena, doc_lengths=lengths)
            assert ws.bytes <= p.arena_bytes(arena=arena) and ws.bytes <= default.bytes
            assert len(p.batches(arena, lengths)) >= 2, (label, lean)
            tws = CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                save_mode="tape",
                arena=arena,
                doc_lengths=lengths,
            )
            assert tws.bytes == p.arena_bytes(
                save_mode="tape", arena=arena, doc_lengths=lengths
            )
            options.append((f"{label}-lean" if lean else label, lean, arena, ws, tws))

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, grads, saved = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}
        _, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=tape_default
        )
        for tag, lean, arena, ws, tws in options:

            def bwd(source, **kwargs):
                return _named(
                    cake_kda_nondet_train_backward(
                        q, k, v, g, beta, cu, dout, saved=source, **kwargs
                    )
                )

            # the batched forward: the whole problem's tables, the default's saved fields
            o, saved_b = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
            )
            arms[f"{tag}.forward"] = {"out": o}
            assert len(saved_b.tables) == len(saved.tables), tag
            assert all(
                torch.equal(a, b)
                for a, b in zip(saved_b.tables, saved.tables, strict=True)
            ), tag
            assert (saved_b.hseg is None) == (saved.hseg is None), tag
            # the version-4 rows the groups write as slices: the fold's qg, and w / u on the rows that save them
            # (the real token rows; the rows above BT * real are the chunk padding, which the whole-problem and the
            # per-group forwards fill differently and no backward reads)
            assert (saved_b.w is None) == (saved.w is None), tag
            assert (saved.w is None) is not p.light_saves_wu, tag
            written = {"Aqk": saved_b.Aqk, "Akk": saved_b.Akk, "qg": saved_b.qg[rows]}
            if saved_b.hseg is not None:
                written.update(hseg=saved_b.hseg, seg_state=saved_b.seg_state)
            if saved_b.w is not None:
                written.update(w=saved_b.w[rows], u=saved_b.u[rows])
            _finite(written, (tag, "saved set"))
            # the batched backward from the default's set and from its own; the default backward from the batched
            # set; a per-call batched arena (sized for the exact groups)
            arms[f"{tag}.default_set"] = bwd(saved, workspace=ws, doc_lengths=lengths)
            arms[f"{tag}.batched_set"] = bwd(saved_b, workspace=ws, doc_lengths=lengths)
            arms[f"{tag}.default_backward"] = bwd(saved_b, workspace=default)
            arms[f"{tag}.per_call"] = bwd(saved, arena=arena, doc_lengths=lengths)
            # the replay group by group on a workspace that is not lean; a lean one refuses it naming both options
            if not lean:
                arms[f"{tag}.replay"] = bwd(None, workspace=ws, doc_lengths=lengths)
            elif j == 0:
                with pytest.raises(ValueError, match="batch_tokens"):
                    bwd(None, workspace=ws, doc_lengths=lengths)
            # the tape set: the saved v_new comes from its allocation filled with NaN, so a row no group writes
            # (the pad rows included, zero in the default's) would show
            original_alloc, poisoned = cake_backend._alloc, []
            ref_v_new = saved_t.v_new

            def nan_v_new(
                shape_,
                dtype,
                device_,
                _orig=original_alloc,
                _hit=poisoned,
                _ref=ref_v_new,
            ):
                t = _orig(shape_, dtype, device_)
                if dtype == _ref.dtype and tuple(shape_) == tuple(_ref.shape):
                    t.view(torch.uint8).fill_(0xFF)
                    _hit.append(t.data_ptr())
                return t

            monkeypatch.setattr(cake_backend, "_alloc", nan_v_new)
            try:
                o, saved_tb = cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=tws, doc_lengths=lengths
                )
            finally:
                monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
            assert poisoned == [saved_tb.v_new.data_ptr()], tag
            assert saved_tb.version == cake_backend.SAVED_VERSION_TAPE, tag
            assert (saved_tb.seg_m is None) == (saved_t.seg_m is None), tag
            written = {"h": saved_tb.h[:real], "v_new": saved_tb.v_new}
            if saved_tb.seg_m is not None:
                written["seg_m"] = saved_tb.seg_m
            _finite(written, (tag, "tape set"))
            arms[f"{tag}.tape_forward"] = {"out": o}
            arms[f"{tag}.own_tape_set"] = bwd(
                saved_tb, workspace=tws, doc_lengths=lengths
            )
            arms[f"{tag}.default_tape_set"] = bwd(
                saved_t, workspace=tws, doc_lengths=lengths
            )
            arms[f"{tag}.light_set_on_tape"] = bwd(
                saved_b, workspace=tws, doc_lengths=lengths
            )
            arms[f"{tag}.default_from_tape"] = bwd(saved_tb, workspace=default)
            # the autograd node carries the option and doc_lengths; a forward without a saved set
            leaves = _leaves(inputs)
            o = cake_kda_nondet(*leaves, cu, arena=arena, doc_lengths=lengths)
            o.backward(dout)
            arms[f"{tag}.autograd"] = {"out": o.detach(), **_leaf_grads(leaves)}
            nosave, none = cake_kda_nondet_train_forward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                save_for_backward=False,
                workspace=ws,
                doc_lengths=lengths,
            )
            assert none is None, tag
            arms[f"{tag}.nosave"] = {"out": nosave}
            if j == 0:
                # doc_lengths are required and checked on the host
                with pytest.raises(ValueError, match="doc_lengths"):
                    cake_kda_nondet_train_forward(q, k, v, g, beta, cu, workspace=ws)
                with pytest.raises(ValueError, match="doc_lengths"):
                    cake_kda_nondet_train_backward(
                        q, k, v, g, beta, cu, dout, saved=saved, arena=arena
                    )
                with pytest.raises(ValueError, match="doc_lengths"):
                    cake_kda_nondet_train_forward(
                        q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths[:-1]
                    )
                with pytest.raises(ValueError, match="doc_lengths"):
                    cake_kda_nondet(*[inputs[n] for n in LEAVES], cu, arena=arena)
        return arms, inputs

    _judge(record_property, ("batched_arena", shape), draw)


def test_batched_arena_launches_per_group(device, manifest, record_property) -> None:
    """The batched call's launch list is every group's list in turn (the whole problem's tables first on the saving
    forward; the group's tables first on the backward), each launch at the group's scalars and grids (the saving
    forward's groups in the tape geometry: they write slices of the saved qg / w / u); the overlap pair's and the
    two-level carry's epoch advances once per group through the shared counter; every group runs its tail in the
    fewest passes that fit the arena (groups in several passes and in one, within a call too), whose launches read
    the host's chunk counts of their ranges; the group tables are slices of the whole problem's.  Judged per row:
    the default call (``<row>``) and per grouping and tail-pass count the batched forward (``.forward``), the
    forward without a saved set (``.nosave``), the backward from the batched set (``.backward``) and the replay
    group by group (``.replay``, one pass)."""
    seen = {"multi": 0, "one_pass": 0, "mixed": 0}

    def check_row(lengths, heads):
        S, N = sum(lengths), len(lengths)
        row = _shape_id((lengths, heads))
        launched: list = []
        counts: list = []  # the tail passes' one-entry chunk counts (num_seqs = 0), in launch order

        def recording(
            name, values, *, grid, device_index, _log=launched, _counts=counts
        ):
            _log.append(
                (
                    name,
                    tuple(grid),
                    int(values["num_seqs"]),
                    int(values["num_chunks_pad"]),
                )
            )
            if int(values["num_seqs"]) == 0:
                _counts.append(values["seq_chunk_start"])
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        options = []
        for label, kw in _partitions(lengths):
            for t in (1, 3):
                arena = cake_backend.CakeKDANondetArena(lean=t > 1, tail_passes=t, **kw)
                ws = CakeKDANondetWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    arena=arena,
                    launcher=recording,
                    doc_lengths=lengths,
                )
                options.append((label, t, arena, ws))

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            q, k, v, g, beta, cu, dout = _ops(inputs)
            out, grads, saved = _run_fwd_bwd(inputs)
            arms = {row: {"out": out, **grads}}
            for label, t, arena, ws in options:
                p = ws.plan
                batches = p.batches(arena, lengths)
                case = (lengths, heads, label, t)
                prefix = f"{row}.{label}.t{t}"
                epoch0 = ws._epoch[0]
                launched.clear()
                o, saved_b = cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
                )
                arms[f"{prefix}.forward"] = {"out": o}
                stamped = p.recur == "seg" and p.record("seg_carry", "fwd") in (
                    "seg_carry_ov",
                    "seg_carry_v8",
                )
                assert ws._epoch[0] - epoch0 == (len(batches) if stamped else 0), case
                if j == 0:
                    names = [n for n, *_ in launched]
                    assert names == _expected_batched_launches(
                        p, arena, lengths, "fwd"
                    ), case
                    _check_group_launches(launched, p, arena, lengths, "fwd")
                    for view in ws._batch_views.values():
                        assert view._epoch is ws._epoch
                        assert view.eye.data_ptr() == ws.eye.data_ptr()
                        assert view.arena.data_ptr() == ws.arena.data_ptr()
                launched.clear()
                nosave, _none = cake_kda_nondet_train_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    save_for_backward=False,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                arms[f"{prefix}.nosave"] = {"out": nosave}
                if j == 0:
                    assert [n for n, *_ in launched] == _expected_batched_launches(
                        p, arena, lengths, "fwd_nosave"
                    ), case
                launched.clear()
                counts.clear()
                got = cake_kda_nondet_train_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=saved_b,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                arms[f"{prefix}.backward"] = _named(got)
                path = p.light_path
                if j == 0:
                    # every group's tail in the fewest passes up to t that fit the arena, as its view has it
                    passes = _group_passes(p, arena, lengths, path, nbytes=ws.bytes)
                    assert [
                        ws.batch_view(b, tape=_group_rows_sliced(path)).tail_passes
                        for b in batches
                    ] == passes, case
                    if t > 1:
                        seen["multi"] += sum(x > 1 for x in passes)
                        seen["one_pass"] += passes.count(1)
                        seen["mixed"] += len(set(passes)) > 1
                    names = [n for n, *_ in launched]
                    assert names == _expected_batched_launches(
                        p, arena, lengths, path, passes=passes
                    ), case
                    _check_group_launches(
                        launched, p, arena, lengths, path, passes=passes
                    )
                    _check_tail_counts(counts, p, arena, lengths, path, passes=passes)
                if t == 1:  # the replay group by group
                    launched.clear()
                    got = cake_kda_nondet_train_backward(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        cu,
                        dout,
                        saved=None,
                        workspace=ws,
                        doc_lengths=lengths,
                    )
                    arms[f"{prefix}.replay"] = _named(got)
                    if j == 0:
                        assert [n for n, *_ in launched] == _expected_batched_launches(
                            p, arena, lengths, "replay"
                        ), case
                        _check_group_launches(launched, p, arena, lengths, "replay")
                if j == 0:
                    # the group tables: absolute chunk offsets, the group's chunk index base, a permutation of its
                    # documents
                    full = saved.tables
                    for b in batches:
                        ws_b = ws.batch_view(b)
                        tb = cake_backend._tables(ws_b, cu[b.d0 : b.d1 + 1], "light")
                        real_b = _real_chunks(lengths[b.d0 : b.d1])
                        assert torch.equal(
                            tb[0][:real_b], full[0][b.c0 : b.c0 + real_b]
                        ), (case, b)
                        assert torch.equal(
                            tb[1][:real_b], full[1][b.c0 : b.c0 + real_b]
                        ), (case, b)
                        assert torch.equal(
                            tb[2][: b.N + 1] + b.c0, full[2][b.d0 : b.d1 + 1]
                        ), (case, b)
                        assert sorted(tb[3][: b.N].tolist()) == list(range(b.N)), (
                            case,
                            b,
                        )
                        assert (
                            not tb[0][real_b : b.NC_pad].any()
                            and not tb[1][real_b : b.NC_pad].any()
                        )
            return arms, inputs

        _judge(record_property, ("batched_launches", lengths, heads), draw)
        options.clear()  # the row's workspaces go before the next row's

    for lengths, heads in (
        ((63, 1, 65), 4),
        ((100, 0, 300), 2),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((100, 640, 77, 1000, 129), 2),
        ((63, 1, 65), 64),
        ((129, 300), 16),
    ):
        check_row(lengths, heads)
    assert min(seen.values()) > 0, seen


def test_batched_arena_on_a_row_that_saves_the_states(
    device, manifest, monkeypatch, seg_save_rule, record_property
) -> None:
    """A pack above the size rule (its light set carries the states) whose last document alone would be below it:
    the groups follow the whole problem's rule, the forward writes their blocks into the saved states by chain
    rank (the whole problem's tables; every state finite), the backward reads them there (``light_states`` as pass
    2 alone; the documents rank in pack order, so every group's blocks are one run of the saved ones: the carry
    writes the saved ``hseg``, the backward reads views), every result finite: the batched backward from the
    default's and the batched set, the default backward from the batched set and the checkpointed autograd node.
    That pack (above 80k tokens at 4 heads) is too large to judge.  The same route is judged on a small pack under
    the size rule lowered to its tokens per segment (:func:`seg_save_rule`), ``(640, 64, 320)`` at 2 heads: under
    ``batch_tokens = 704`` its group (0, 2) holds chain ranks 0 and 2 (the group's blocks gathered and scattered by
    index) and the group (2, 3) rank 1 (one run); both kinds are checked against the packing on the first draw.
    The saved state blocks of every call on its groupings start as NaN (each must be written: finite after the
    batched forward).  Judged: the default call (arm ``""``) and per grouping and tail-pass count
    (``<grouping>.t<t>``) the batched forward (``.forward``), the batched backward from the default's set and from
    its own (``.default_set``, ``.own_set``), the default backward from the batched set (``.default_backward``) and
    the checkpointed autograd step (``.checkpoint``)."""
    lengths, heads = (40000, 20000, 20000, 1000), 4
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    # The size rule is S >= SEG_SAVE_MIN_TOKENS_PER_SEG * NSEG with NSEG = the SM count here, so on a device
    # with more SMs than this pack covers (216 on some sm_107a boards) the first document grows until the
    # plan saves the states; the last document alone stays below the rule.
    for _ in range(64):
        if cake_backend.plan(sum(lengths), len(lengths), heads, sm, arch).seg_save:
            break
        lengths = (lengths[0] + 4096, *lengths[1:])
    S, N = sum(lengths), len(lengths)
    p = cake_backend.plan(S, N, heads, sm, arch)
    assert p.recur == "seg" and p.seg_save and p.light_path == "light_states"
    assert not cake_backend.plan(1000, 1, heads, sm, arch).seg_save
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = _ops(inputs)
    out, grads, saved = _run_fwd_bwd(inputs)
    assert saved.hseg is not None and saved.seg_state is not None
    _finite({"out": out, **grads}, "default")
    del out, grads
    from torch.utils.checkpoint import checkpoint

    for label, kw in _partitions(lengths):
        arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=4, **kw)
        ws = CakeKDANondetWorkspace.for_problem(
            S, N, heads, device, arena=arena, doc_lengths=lengths
        )
        tag = (label,)
        o, saved_b = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
        )
        _finite({"out": o, "hseg": saved_b.hseg, "seg_state": saved_b.seg_state}, tag)
        assert all(
            torch.equal(a, b) for a, b in zip(saved_b.tables, saved.tables, strict=True)
        ), tag
        del o
        for name, source, space in (
            ("batched backward, default set", saved, ws),
            ("batched backward, batched set", saved_b, ws),
            ("default backward, batched set", saved_b, None),
        ):
            got = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=source,
                workspace=space,
                doc_lengths=lengths if space is not None else None,
            )
            _finite(_named(got), (*tag, name))
            del got
        leaves = _leaves(inputs)
        o = checkpoint(
            lambda *a: cake_kda_nondet(*a, cu, arena=arena, doc_lengths=lengths),
            *leaves,
            use_reentrant=False,
        )
        o.backward(dout)
        _finite({"out": o.detach(), **_leaf_grads(leaves)}, (*tag, "checkpoint"))
        del ws, o, leaves
    del inputs, q, k, v, g, beta, dout, saved

    # the judged pack: the rule lowered to the tokens per segment of the whole pack (most segments hold no chunk,
    # as on the short documents of the pack above)
    lengths, heads = (640, 64, 320), 2
    S, N = sum(lengths), len(lengths)
    nseg = cake_backend.plan(S, N, heads, sm, arch).NSEG
    seg_save_rule(max(1, S // nseg))
    p = cake_backend.plan(S, N, heads, sm, arch)
    assert p.recur == "seg" and p.seg_save and p.light_path == "light_states"
    ranks = _kernel_ranks(lengths)
    groupings = list(RANK_GROUPINGS[lengths])
    groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
    hits = {"run": 0, "index": 0}
    original_alloc = cake_backend._alloc

    def poisoned_alloc(shape, dtype, device_):
        t = original_alloc(shape, dtype, device_)
        # the saved state blocks (hseg, seg_state): NaN until a kernel or a copy writes them
        if dtype == torch.float32 and tuple(shape[-2:]) == (D, D):
            t.view(torch.uint8).fill_(0xFF)
        return t

    def grouped(inputs, saved, kw, t: int, first: bool) -> dict:
        """The arms of one grouping and tail-pass count."""
        q, k, v, g, beta, cu, dout = _ops(inputs)
        prefix = f"{_kw_label(kw)}.t{t}"
        arena = cake_backend.CakeKDANondetArena(lean=t > 1, tail_passes=t, **kw)
        ws = CakeKDANondetWorkspace.for_problem(
            S, N, heads, device, arena=arena, doc_lengths=lengths
        )
        if first:
            packing = ws._packing(lengths)
            for i, b in enumerate(packing.batches):
                one_run = _one_run(_expected_group_blocks(ranks, lengths, b, p.P))
                assert (packing.blocks[i] is not None) == one_run, (prefix, b)
                hits["run" if one_run else "index"] += 1
        o, saved_b = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
        )
        _finite({"hseg": saved_b.hseg, "seg_state": saved_b.seg_state}, prefix)
        assert all(
            torch.equal(a, b) for a, b in zip(saved_b.tables, saved.tables, strict=True)
        ), prefix
        arms = {f"{prefix}.forward": {"out": o}}
        for name, source, space in (
            ("default_set", saved, ws),
            ("own_set", saved_b, ws),
            ("default_backward", saved_b, None),
        ):
            got = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=source,
                workspace=space,
                doc_lengths=lengths if space is not None else None,
            )
            arms[f"{prefix}.{name}"] = _named(got)
        leaves = _leaves(inputs)
        o = checkpoint(
            lambda *a: cake_kda_nondet(*a, cu, arena=arena, doc_lengths=lengths),
            *leaves,
            use_reentrant=False,
        )
        o.backward(dout)
        arms[f"{prefix}.checkpoint"] = {"out": o.detach(), **_leaf_grads(leaves)}
        return arms

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        out, grads, saved = _run_fwd_bwd(inputs)
        assert saved.hseg is not None and saved.seg_state is not None
        arms = {"": {"out": out, **grads}}
        monkeypatch.setattr(cake_backend, "_alloc", poisoned_alloc)
        try:
            for kw in groupings:
                for t in (1, 4):
                    arms.update(grouped(inputs, saved, kw, t, j == 0))
        finally:
            monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
        return arms, inputs

    _judge(record_property, ("batched_saved_states", lengths, heads), draw)
    assert hits["run"] > 0 and hits["index"] > 0, hits


def test_batched_arena_with_exact_zero_values(
    device, manifest, monkeypatch, record_property
) -> None:
    """Rows with exact zeros in ``v`` and ``dout`` (a zero token row, a zero chunk at a group's first chunk, a whole
    zero document at an odd chunk): the paired WY programs accumulate ``0 * partner`` for the other chunk of a
    pair, over every grouping.  The batched tape forward writes every row of the saved ``v_new`` (its allocation
    starts NaN-filled) and the real rows of ``h``.  Judged per row: the default call (``<row>``) and per grouping
    the batched forward (``.forward``), the backward from the batched set (``.backward``), the batched tape forward
    (``.tape_forward``) and the backward from the batched tape set (``.tape``)."""

    def check_row(lengths, heads):
        S, N = sum(lengths), len(lengths)
        row = _shape_id((lengths, heads))
        real = _real_chunks(lengths)
        tape_default = CakeKDANondetWorkspace.for_problem(
            S, N, heads, device, save_mode="tape"
        )
        options = []
        for label, kw in _partitions(lengths):
            arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=2, **kw)
            ws = CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                arena=arena,
                doc_lengths=lengths,
            )
            tws = CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                save_mode="tape",
                arena=arena,
                doc_lengths=lengths,
            )
            options.append((label, ws, tws))

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            v, dout = inputs["v"], inputs["dout"]
            cu = [0]
            for n in lengths:
                cu.append(cu[-1] + n)
            v[cu[1] : cu[2]] = (
                0  # the second document (one token at an odd chunk, or a whole document)
            )
            v[cu[2] : cu[2] + 6] = (
                0  # the first rows of the third document (a group's first chunk)
            )
            dout[cu[2] : cu[2] + 6] = 0
            v[cu[-1] - 7 : cu[-1]] = (
                0  # the pack's last rows (a pad partner's neighbour)
            )
            dout[:5] = 0
            q, k, g, beta, cu_t = (
                inputs[n] for n in ("q", "k", "g", "beta", "cu_seqlens")
            )
            out, grads, _ = _run_fwd_bwd(inputs)
            arms = {row: {"out": out, **grads}}
            _, saved_t = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu_t, workspace=tape_default
            )
            ref_v_new = saved_t.v_new
            for label, ws, tws in options:
                tag = (lengths, heads, label)
                o, saved_b = cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu_t, workspace=ws, doc_lengths=lengths
                )
                arms[f"{row}.{label}.forward"] = {"out": o}
                got = cake_kda_nondet_train_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu_t,
                    dout,
                    saved=saved_b,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                arms[f"{row}.{label}.backward"] = _named(got)
                # the saved v_new comes from its allocation filled with NaN: a row no group writes would show
                original_alloc, poisoned = cake_backend._alloc, []

                def nan_v_new(
                    shape_,
                    dtype,
                    device_,
                    _orig=original_alloc,
                    _hit=poisoned,
                    _ref=ref_v_new,
                ):
                    t = _orig(shape_, dtype, device_)
                    if dtype == _ref.dtype and tuple(shape_) == tuple(_ref.shape):
                        t.view(torch.uint8).fill_(0xFF)
                        _hit.append(t.data_ptr())
                    return t

                monkeypatch.setattr(cake_backend, "_alloc", nan_v_new)
                try:
                    o, saved_tb = cake_kda_nondet_train_forward(
                        q, k, v, g, beta, cu_t, workspace=tws, doc_lengths=lengths
                    )
                finally:
                    monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
                assert poisoned == [saved_tb.v_new.data_ptr()], tag
                # v_new whole, its pad rows included (zero in the default's: the rows after the last group's slice
                # too)
                _finite({"h": saved_tb.h[:real], "v_new": saved_tb.v_new}, tag)
                arms[f"{row}.{label}.tape_forward"] = {"out": o}
                got = cake_kda_nondet_train_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu_t,
                    dout,
                    saved=saved_tb,
                    workspace=tws,
                    doc_lengths=lengths,
                )
                arms[f"{row}.{label}.tape"] = _named(got)
            return arms, inputs

        _judge(record_property, ("exact_zero_values", lengths, heads), draw)
        options.clear()  # the row's workspaces go before the next row's
        tape_default = None

    for lengths, heads in (
        ((63, 1, 65, 30), 2),
        ((63, 1, 65), 4),
        ((100, 640, 77, 1000, 129), 2),
    ):
        check_row(lengths, heads)


def test_batched_arena_under_checkpointing_and_a_dropped_workspace(
    device, manifest, monkeypatch, record_property
) -> None:
    """A checkpointed forward with the batched option retains nothing beyond its output; the backward's own
    workspace takes the recorded option and ``doc_lengths`` (sized for the exact groups); a dropped workspace is
    replaced the same way; a kept workspace of another option is refused; ``doc_lengths`` are required.  Judged:
    the default call (arm ``""``), the option's eager forward (``warm``), its checkpointed step (``ckpt``) and the
    step whose caller workspace was dropped (``dropped``)."""
    import gc

    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=2, batch_tokens=1000)
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    assert [(b.d0, b.d1) for b in plan.batches(arena, lengths)] == [
        (0, 3),
        (3, 4),
        (4, 5),
    ]
    exact = plan.arena_bytes(arena=arena, doc_lengths=lengths)
    assert exact < plan.arena_bytes(arena=arena) < plan.arena_bytes()
    created = _spy_call_workspaces(monkeypatch)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        out, grads, _ = _run_fwd_bwd(inputs)
        arms = {"": {"out": out, **grads}}

        def implicit(q, k, v, g, beta):
            return cake_kda_nondet(
                q, k, v, g, beta, cu, arena=arena, doc_lengths=lengths
            )

        held = _leaves(inputs)
        arms["warm"] = {"out": implicit(*held).detach()}
        if j == 0:
            gc.collect()
            assert _retained_after_checkpointed_forward(implicit, *held) == 0
        created.clear()
        o = checkpoint(implicit, *held, use_reentrant=False)
        o.backward(dout)
        arms["ckpt"] = {"out": o.detach(), **_leaf_grads(held)}
        assert created and all(w.arena_option == arena for w in created)
        assert all(
            w.doc_lengths == tuple(lengths) and w.bytes == exact
            for w in created
            if w.backward
        )
        assert any(w.backward for w in created)
        dropped = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            arena=arena,
            doc_lengths=lengths,
        )
        created.clear()
        held = _leaves(inputs)
        o = cake_kda_nondet(*held, cu, workspace=dropped, doc_lengths=lengths)
        del dropped
        gc.collect()
        o.backward(dout)
        assert (
            len(created) == 1
            and created[0].arena_option == arena
            and created[0].backward
        )
        assert created[0].bytes == exact
        arms["dropped"] = {"out": o.detach(), **_leaf_grads(held)}
        if j == 0:
            kept = CakeKDANondetWorkspace(plan, device, backward=True)
            with pytest.raises(ValueError, match="arena="):
                cake_kda_nondet(
                    *_leaves(inputs),
                    cu,
                    workspace=kept,
                    arena=arena,
                    doc_lengths=lengths,
                )
            with pytest.raises(ValueError, match="doc_lengths"):
                cake_kda_nondet(*_leaves(inputs), cu, arena=arena)
        return arms, inputs

    _judge(record_property, ("batched_checkpointing", lengths, heads), draw)


def test_batched_views_share_the_arena_eye_and_epoch(device, manifest) -> None:
    """The views of one arena (its groups, ``with_plan``) share the bytes, the identity / flag buffer and the epoch
    counter; every group's layout fits the exact arena, and a packing whose group does not fit is refused before
    any launch."""
    lengths, heads = (63, 1, 65, 30), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
    )
    arena = cake_backend.CakeKDANondetArena(batch_tokens=95)
    ws = CakeKDANondetWorkspace.for_problem(
        S, N, heads, device, arena=arena, doc_lengths=lengths
    )
    p = ws.plan
    batches = p.batches(arena, lengths)
    assert [(b.d0, b.d1) for b in batches] == [(0, 2), (2, 4)]
    views = [ws.batch_view(b) for b in batches]
    assert views[0] is ws.batch_view(batches[0]) and views[1] is not views[0]
    for view in views:
        assert view.arena.data_ptr() == ws.arena.data_ptr() and view.bytes == ws.bytes
        assert view.eye.data_ptr() == ws.eye.data_ptr()
        assert view.flags.data_ptr() == ws.flags.data_ptr()
        assert view._epoch is ws._epoch and view.arena_option == arena
        for mode in ("fwd", "bwd", "replay"):
            view.views(mode)
    e = ws.next_epoch()
    assert views[0].next_epoch() == e + 1 and views[1].next_epoch() == e + 2
    assert ws.next_epoch() == e + 3
    twin = ws.with_plan(p)
    assert twin._epoch is ws._epoch and twin.next_epoch() == e + 4
    assert twin.eye.data_ptr() == ws.eye.data_ptr()
    # a workspace sized for one document per group refuses a packing whose group is larger, before any launch
    single = cake_backend.CakeKDANondetArena(batch_tokens=S, max_docs_per_batch=1)
    small = CakeKDANondetWorkspace.for_problem(
        S, N, heads, device, arena=single, doc_lengths=lengths
    )
    assert small.bytes < ws.bytes
    other = (S, 0, 0, 0)  # the same S and num_docs, one group of every token
    with pytest.raises(ValueError, match="needs"):
        cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=small, doc_lengths=other
        )
    # the bound sizes a workspace without doc_lengths: every grouping of the budget fits it
    bound = CakeKDANondetWorkspace.for_problem(S, N, heads, device, arena=arena)
    assert bound.doc_lengths is None and bound.bytes >= ws.bytes
    for b in batches:
        bound.batch_view(b).views("bwd")


def test_batched_arena_content_never_reaches_a_result(
    device, manifest, record_property
) -> None:
    """Write-before-read on the batched option too: with a batched arena filled with NaN bytes before every call
    every result is judged (the forward, the backward from the light and the tape sets, a light set on the tape
    workspace, the forward without a saved set) and the saved ``Aqk`` / ``Akk`` (and the states) are finite,
    including a row whose light set carries the states (the groups read the saved tensors, not the arena; pass 2
    never touches the arena's ``seg_state`` it is bound to at ``write_ht = 0``).  Judged per case: the zero-filled
    default arena's call (``<case>.zeroed``), the poisoned arena's forward and backward (``<case>.poisoned``), its
    backward from the default's set (``<case>.default_set``), a version-4 set on the poisoned tape workspace
    (``<case>.light_set``) and its forward without a saved set (``<case>.nosave``).  The states-saving row (above
    80k tokens at 4 heads, its first document alone in the first group) is too large to judge: every result of it is
    finite, and the states-saving route is judged in
    :func:`test_lean_arena_serves_the_saved_states_and_the_tape_paths` (``n1-h1``).  That row saves the states only
    from ``S >= 384 * NSEG`` on, and the segment count grows with the device's SMs (at 212 SMs NSEG = 212 needs
    81408 tokens), so its first document grows as :func:`_saving_pack` gives it on this device."""
    big = _saving_pack((40000, 20000, 20000, 1000), 4, device)
    cases = (
        ((63, 1, 65, 30), 2, "light", 1, {"batch_tokens": 95}),
        ((100, 640, 77, 1000, 129), 2, "light", 3, {"batch_tokens": 1000}),
        (
            (20, 20, 20, 20, 20, 20, 20),
            8,
            "light",
            2,
            {"batch_tokens": 140, "max_docs_per_batch": 3},
        ),
        ((63, 1, 65), 64, "light", 3, {"batch_tokens": 129, "max_docs_per_batch": 1}),
        ((100, 640, 77, 1000, 129), 2, "tape", 2, {"batch_tokens": 1000}),
        (big, 4, "light", 4, {"batch_tokens": big[0]}),
    )

    def check_row(lengths, heads, save_mode, t, kw):
        S, N = sum(lengths), len(lengths)
        case = f"{_shape_id((lengths, heads))}-{save_mode}-t{t}"
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
        )
        zeroed = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            zero_init=True,
            save_mode=save_mode,
        )
        arena = cake_backend.CakeKDANondetArena(lean=t > 1, tail_passes=t, **kw)
        poisoned = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            save_mode=save_mode,
            arena=arena,
            doc_lengths=lengths,
        )
        assert poisoned.zero_init is False and poisoned.bytes < zeroed.bytes
        assert len(plan.batches(arena, lengths)) >= 2

        def poison(_ws=poisoned):
            _ws.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
            torch.cuda.synchronize()

        def run(
            inputs, _case=case, _zeroed=zeroed, _poisoned=poisoned, _poison=poison
        ) -> dict:
            q, k, v, g, beta, cu, dout = _ops(inputs)
            out, grads, saved = _run_fwd_bwd(inputs, workspace=_zeroed)
            arms = {f"{_case}.zeroed": {"out": out, **grads}}
            _poison()
            o, saved_p = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=_poisoned, doc_lengths=lengths
            )
            written = {"Aqk": saved_p.Aqk, "Akk": saved_p.Akk}
            if saved.hseg is not None:
                written.update(hseg=saved_p.hseg, seg_state=saved_p.seg_state)
            _finite(written, (_case, "saved set"))
            _poison()
            got = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved_p,
                workspace=_poisoned,
                doc_lengths=lengths,
            )
            arms[f"{_case}.poisoned"] = {"out": o, **_named(got)}
            _poison()
            got = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved,
                workspace=_poisoned,
                doc_lengths=lengths,
            )
            arms[f"{_case}.default_set"] = _named(got)
            if save_mode == "tape":
                # a version-4 set on the poisoned tape workspace: per-call h / v_new per group
                _, saved_light = cake_kda_nondet_train_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    workspace=_poisoned,
                    save_mode="light",
                    doc_lengths=lengths,
                )
                _poison()
                got = cake_kda_nondet_train_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=saved_light,
                    workspace=_poisoned,
                    doc_lengths=lengths,
                )
                arms[f"{_case}.light_set"] = _named(got)
            _poison()
            nosave, _ = cake_kda_nondet_train_forward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                save_for_backward=False,
                workspace=_poisoned,
                doc_lengths=lengths,
            )
            arms[f"{_case}.nosave"] = {"out": nosave}
            return arms

        if S > 16384:  # the states-saving row: too large to judge
            assert plan.seg_save, case
            _finite_arms(run(make_inputs(lengths, heads, device)), case)
        else:

            def draw(j, s, _run=run):
                inputs = _draw(lengths, heads, s, device)
                return _run(inputs), inputs

            _judge(
                record_property,
                ("batched_arena_content", lengths, heads, save_mode, t),
                draw,
            )
        zeroed = poisoned = None  # (no del: the poison closure refers to the name)

    for lengths, heads, save_mode, t, kw in cases:
        check_row(lengths, heads, save_mode, t, kw)


# ------------------------------------------------------------------------------------------------------------------
# The batched arena option: the groups' state blocks by the host's chain ranks

# A pack above the size rule whose first group's state blocks are not one run of the whole problem's: document 1
# ranks last (16 chunks) and documents 2 and 3 between, so under ``batch_tokens = L0 + 1000`` the group (0, 2) holds
# ranks 0 and 3 (gathered and scattered by index) and the group (2, 4) ranks 1 and 2 (one run).  The first document
# grows with the device's segment count (_saving_pack).
SPLIT_LENGTHS = (40000, 1000, 20000, 20000)
# Packs of the host chain ranks: the issue's pack, tied chunk counts, zero-length documents, interleaved ties (no
# group of two or more documents is one run), one document, a small pack whose first group is not one run under
# ``batch_tokens = 704`` and 300 documents (the one-CTA ``table`` program; ties and two zero-length documents).
RANK_PACKS = (
    (ISSUE_LENGTHS, ISSUE_HEADS),
    ((128, 100, 128, 65, 1, 64), 2),
    ((100, 0, 300, 0, 64), 2),
    ((64, 640, 1, 600, 63, 577), 2),
    ((1000,), 8),
    ((640, 64, 320), 2),
    (tuple((37 * i + 11) % 130 for i in range(300)), 1),
)
# the groupings beyond _partitions a pack runs under (the issue's two groups of the lean four-pass budget, the
# groups that are not one run)
RANK_GROUPINGS = {
    ISSUE_LENGTHS: ({"batch_tokens": 200000},),
    (640, 64, 320): ({"batch_tokens": 704},),
    (1000,): ({"batch_tokens": 1000},),
}


def _saving_pack(lengths, heads, device) -> tuple[int, ...]:
    """``lengths`` with the first document grown (4096 tokens a step) until the plan on this device saves the
    segment states (``S >= 384 * NSEG``: on a device with more SMs the segment count grows)."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    lengths = tuple(lengths)
    for _ in range(64):
        if cake_backend.plan(sum(lengths), len(lengths), heads, sm, arch).seg_save:
            return lengths
        lengths = (lengths[0] + 4096, *lengths[1:])
    raise AssertionError(f"no pack from {lengths} saves the states on this device")


def _kernel_ranks(lengths) -> list[int]:
    """The chain rank every ``seq_order`` writer gives each document (``table``, ``table_v2``, ``table_v3`` and the
    segmented intra program ``fwd_intra_v14``, their loop spelled out): one per document with more chunks, one per
    earlier document with as many."""
    nc = [(int(n) + BT - 1) // BT for n in lengths]
    return [
        sum(1 for j, m in enumerate(nc) if m > nc[d] or (m == nc[d] and j < d))
        for d in range(len(nc))
    ]


def _expected_group_blocks(whole, lengths, batch, P) -> list[int]:
    """The whole problem's state blocks of one group in the group's own block order, by the kernels' layout: the
    group's tables rank its documents among themselves, the whole problem's among all (``whole``, the
    :func:`_kernel_ranks` of ``lengths``), ``P`` blocks per chain."""
    local = _kernel_ranks(lengths[batch.d0 : batch.d1])
    order = sorted(range(batch.N), key=lambda d: local[d])
    return [P * whole[batch.d0 + d] + s for d in order for s in range(P)]


def _one_run(blocks) -> bool:
    return blocks == list(range(blocks[0], blocks[0] + len(blocks)))


def test_host_chain_ranks_and_block_runs_follow_the_tables_rule() -> None:
    """The host's chain ranks (``_seq_order`` / ``_chain_ranks``) are the rank every ``seq_order`` writer computes,
    and a batched packing (``_Packing``) holds the kernels' block layout: per group the whole problem's first block
    where the group's blocks are one run of the whole problem's, else their indices, and the tail passes' chunk
    counts of the device rule at both of the group's chunk bounds.  On every architecture, over the rank packs and
    their groupings, with the counts at each group's own tail passes (:func:`_fewest_tail_passes`; none for a
    group in one pass); the issue's two groups of the lean four-pass budget are the runs [0, 76) and [76, 152) on
    a 152-SM ``sm_100a``, and the serial route has no blocks."""
    seen = {"run": 0, "index": 0, "tail": 0, "one_pass": 0}
    for lengths, heads in RANK_PACKS:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        ranks = _kernel_ranks(lengths)
        order = cake_backend._seq_order(lengths)
        assert cake_backend._chain_ranks(lengths) == tuple(ranks), lengths
        assert sorted(order) == list(range(N)), lengths
        assert all(ranks[d] == r for r, d in enumerate(order)), lengths
        groupings = list(RANK_GROUPINGS.get(lengths, ()))
        groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
        for arch in ARCHES:
            p = cake_backend.plan(S, N, heads, 148, arch)
            assert p.recur == "seg" and not p.proportional, (lengths, arch)
            for kw in groupings:
                for t in (1, 4):
                    arena = cake_backend.CakeKDANondetArena(
                        lean=t > 1, tail_passes=t, **kw
                    )
                    nbytes = p.arena_bytes(arena=arena, doc_lengths=lengths)
                    packing = cake_backend._host_packing(
                        p, arena, lengths, nbytes=nbytes, save_mode="light"
                    )
                    case = (lengths, arch, kw, t)
                    assert packing.batches == p.batches(arena, lengths), case
                    assert packing.words[:N] == lengths and packing.P == p.P, case
                    assert packing.words_dev is None and packing.index_dev is None
                    for i, b in enumerate(packing.batches):
                        blocks = _expected_group_blocks(ranks, lengths, b, p.P)
                        assert len(blocks) == b.N * p.P, (case, b)
                        if _one_run(blocks):
                            assert packing.blocks[i] == blocks[0], (case, b)
                            assert i not in packing.index_at, (case, b)
                            seen["run"] += 1
                        else:
                            assert packing.blocks[i] is None, (case, b)
                            first, n = packing.index_at[i]
                            assert list(packing.index[first : first + n]) == blocks
                            seen["index"] += 1
                        real = _real_chunks(lengths[b.d0 : b.d1])
                        for nc_pad in (b.NC_pad, b.NC_pad_tape):
                            pb = p.batch_plan(b, tape=nc_pad != b.NC_pad)
                            passes = _fewest_tail_passes(pb, arena, nbytes)
                            if passes == 1:
                                assert (i, nc_pad) not in packing.tail_at, (case, b)
                                seen["one_pass"] += t > 1
                                continue
                            ranges = cake_backend._tail_ranges(nc_pad, passes)
                            want = [
                                min(max(real - c0, 0), ranges[0][1])
                                for c0, _n in ranges
                            ]
                            first, n = packing.tail_at[(i, nc_pad)]
                            got = list(packing.words[first : first + n])
                            assert got == want, (case, b, nc_pad)
                            seen["tail"] += 1
    assert min(seen.values()) > 0, seen
    # the issue's pack under the lean four-pass budget of two groups: the second starts at an odd chunk, both are
    # one run
    S = sum(ISSUE_LENGTHS)
    p = cake_backend.plan(S, len(ISSUE_LENGTHS), ISSUE_HEADS, 152, "sm_100a")
    arena = cake_backend.CakeKDANondetArena(
        lean=True, tail_passes=4, batch_tokens=200000
    )
    packing = cake_backend._host_packing(
        p,
        arena,
        ISSUE_LENGTHS,
        nbytes=p.arena_bytes(arena=arena, doc_lengths=ISSUE_LENGTHS),
        save_mode="light",
    )
    assert [(b.d0, b.d1, b.c0) for b in packing.batches] == [(0, 2, 0), (2, 4, 2981)]
    assert (p.P, p.NSEG, packing.blocks, packing.index) == (38, 152, (0, 76), ())
    # the split pack: its first group by index, its second one run from the second chain
    for arch in ARCHES:
        p = cake_backend.plan(sum(SPLIT_LENGTHS), 4, 4, 148, arch)
        arena = cake_backend.CakeKDANondetArena(
            batch_tokens=SPLIT_LENGTHS[0] + SPLIT_LENGTHS[1]
        )
        packing = cake_backend._host_packing(
            p,
            arena,
            SPLIT_LENGTHS,
            nbytes=p.arena_bytes(arena=arena, doc_lengths=SPLIT_LENGTHS),
            save_mode="light",
        )
        assert [(b.d0, b.d1) for b in packing.batches] == [(0, 2), (2, 4)]
        assert packing.blocks == (None, p.P), arch
        assert packing.index == tuple(range(p.P)) + tuple(range(3 * p.P, 4 * p.P))
    # the serial route: no state blocks, no indices
    p = cake_backend.plan(129, 3, 64, 148, ARCHES[0])
    assert p.recur == "serial"
    arena = cake_backend.CakeKDANondetArena(batch_tokens=65)
    packing = cake_backend._host_packing(
        p,
        arena,
        (63, 1, 65),
        nbytes=p.arena_bytes(arena=arena, doc_lengths=(63, 1, 65)),
        save_mode="light",
    )
    assert packing.blocks == (None, None)
    assert packing.index == () and not packing.index_at


def test_batched_groups_take_the_fewest_tail_passes_that_fit() -> None:
    """Under a batched option with ``tail_passes = B`` every group runs its tail in the fewest passes up to ``B``
    whose lean layouts at the group's geometry fit the arena (``_group_tail_passes``, which the group views and the
    host's chunk counts take): the restated rule's count, between one and ``B``, fitting the arena, no fewer
    fitting; the bound of every group (a larger arena) never takes more than the exact groups.  Over the rank packs
    and their groupings on every architecture, light and tape, groups in one pass and in several, within one call
    too.  The issue's pack on a 152-SM ``sm_100a`` under the lean four-pass budget of two groups: the first group,
    which sizes the arena, keeps four passes and the second fits one (three and one on the bound's arena), so the
    host's counts are the first group's alone and the tail launches five times per stage instead of eight."""
    seen = {"multi": 0, "one_pass": 0, "mixed": 0}
    for lengths, heads in RANK_PACKS:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        groupings = list(RANK_GROUPINGS.get(lengths, ()))
        groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
        for arch in ARCHES:
            p = cake_backend.plan(S, N, heads, 148, arch)
            for kw in groupings:
                for t in (1, 2, 3, 4):
                    arena = cake_backend.CakeKDANondetArena(
                        lean=t > 1, tail_passes=t, **kw
                    )
                    batches = p.batches(arena, lengths)
                    for save_mode in ("light", "tape"):
                        modes = ("bwd_tape_lean",)
                        if save_mode == "light":
                            modes += ("bwd_lean",)
                        sized = {
                            sizing: p.arena_bytes(
                                save_mode=save_mode, arena=arena, doc_lengths=doc
                            )
                            for sizing, doc in (("exact", lengths), ("bound", None))
                        }
                        got = {}
                        for (sizing, nbytes), b in itertools.product(
                            sized.items(), batches
                        ):
                            for nc_pad in sorted({b.NC_pad, b.NC_pad_tape}):
                                pb = p.batch_plan(b, tape=nc_pad != b.NC_pad)
                                passes = cake_backend._group_tail_passes(
                                    p,
                                    b,
                                    nc_pad,
                                    arena,
                                    nbytes=nbytes,
                                    save_mode=save_mode,
                                )
                                case = (lengths, arch, kw, t, save_mode, sizing, b)
                                fits = [
                                    all(
                                        pb.workspace_bytes(m, x) <= nbytes
                                        for m in modes
                                    )
                                    for x in range(1, t + 1)
                                ]
                                assert passes == _fewest_tail_passes(
                                    pb, arena, nbytes, save_mode
                                ), case
                                assert 1 <= passes <= t and fits[passes - 1], case
                                assert not any(fits[: passes - 1]), case
                                got[(sizing, b, nc_pad)] = passes
                        for (sizing, b, nc_pad), passes in got.items():
                            assert passes <= got[("exact", b, nc_pad)], (lengths, t)
                            if t > 1 and sizing == "exact":
                                seen["multi" if passes > 1 else "one_pass"] += 1
                        if t > 1:
                            seen["mixed"] += (
                                len({got[("exact", b, b.NC_pad)] for b in batches}) > 1
                            )
    assert min(seen.values()) > 0, seen
    # the issue's pack: four passes for the group that sizes the arena, one for the other
    S = sum(ISSUE_LENGTHS)
    p = cake_backend.plan(S, len(ISSUE_LENGTHS), ISSUE_HEADS, 152, "sm_100a")
    arena = cake_backend.CakeKDANondetArena(
        lean=True, tail_passes=4, batch_tokens=200000
    )
    batches = p.batches(arena, ISSUE_LENGTHS)
    assert [(b.NC_pad, b.NC_pad_tape) for b in batches] == [(2982, 2982), (1108, 1108)]
    light = p.arena_bytes(arena=arena, doc_lengths=ISSUE_LENGTHS)
    assert light == 2212534272
    assert p.batch_plan(batches[0]).workspace_bytes("bwd_lean", 3) > light
    for save_mode in ("light", "tape"):
        for doc, want in ((ISSUE_LENGTHS, [4, 1]), (None, [3, 1])):
            nbytes = p.arena_bytes(save_mode=save_mode, arena=arena, doc_lengths=doc)
            got = [
                cake_backend._group_tail_passes(
                    p, b, b.NC_pad, arena, nbytes=nbytes, save_mode=save_mode
                )
                for b in batches
            ]
            assert got == want, (save_mode, doc, got)
    packing = cake_backend._host_packing(
        p, arena, ISSUE_LENGTHS, nbytes=light, save_mode="light"
    )
    assert packing.tail_at == {(0, 2982): (4, 4)}
    assert packing.words[4:8] == (746, 746, 746, 743)


# the device rows of the host ranks: every seq_order writer (table_v3 and fwd_intra_v14 on the segmented route up to
# 256 documents, table_v2 on the serial route there, table above 256 documents)
RANK_ROWS = (*RANK_PACKS, ((128, 100, 128, 65, 1, 64), 32))


def test_host_chain_ranks_match_every_seq_order_writer(
    device, manifest, monkeypatch
) -> None:
    """The host's ``seq_order`` is the table every writer leaves on the device, and its chain ranks the ``argsort``
    of that table (the rule of the device glue it replaces): the saved-set paths' table program (``table_v3`` /
    ``table_v2`` / ``table``) and the forward's own tables (the segmented intra program ``fwd_intra_v14`` up to
    256 documents, ``table_v2`` on the serial route), over tied, zero-length, interleaved, one-document and
    300-document packs."""
    original = cake_backend._tables
    writers: set = set()
    compared = 0
    for lengths, heads in RANK_ROWS:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
        )
        ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, backward=False)
        p = ws.plan
        written = [(p.record("table", "light"), original(ws, cu, "light"))]
        # the forward's own tables (above 256 documents the same table program writes them)
        if N <= 256:
            seen: list = []

            def spy(*args, _seen=seen, **kwargs):
                tables = original(*args, **kwargs)
                _seen.append(tables)
                return tables

            monkeypatch.setattr(cake_backend, "_tables", spy)
            cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, save_for_backward=False, workspace=ws
            )
            monkeypatch.setattr(cake_backend, "_tables", original)
            assert len(seen) == 1, lengths
            fwd = "fwd_nosave"
            writer = (
                p.record("table", fwd)
                if p.dispatched("table", fwd)
                else p.record("fwd_intra", fwd)
            )
            written.append((writer, seen[0]))
        torch.cuda.synchronize()
        want = list(cake_backend._seq_order(lengths))
        ranks = list(cake_backend._chain_ranks(lengths))
        assert ranks == _kernel_ranks(lengths), lengths
        for writer, tables in written:
            assert tables[3][:N].tolist() == want, (lengths, heads, writer)
            assert torch.argsort(tables[3][:N]).tolist() == ranks, (lengths, writer)
            writers.add(writer)
            compared += 1
        del ws
    assert writers == {"table", "table_v2", "table_v3", "fwd_intra_v14"}, writers
    assert compared == len(RANK_ROWS) + sum(
        1 for lengths, _heads in RANK_ROWS if len(lengths) <= 256
    )


def _kw_label(kw: dict) -> str:
    """A grouping's label: ``b<batch_tokens>``, with ``-d<max_docs_per_batch>`` where it is set."""
    label = f"b{kw['batch_tokens']}"
    if kw.get("max_docs_per_batch") is not None:
        label += f"-d{kw['max_docs_per_batch']}"
    return label


def test_batched_block_runs_results_are_judged(
    device, manifest, monkeypatch, record_property
) -> None:
    """The groups' state blocks by the host's chain ranks on both sides of the run test: the saving forward's carry
    writes a one-run group's carried states straight into the saved ``hseg`` (its offsets / transfers two
    contiguous copies; the tape's transfers one) and scatters the other groups' by index; the backward reads views
    of a one-run group's saved ``hseg`` and transfers (``light_states``: its pass 2 bound to the arena's
    ``seg_state``, poisoned here with the rest of the arena) or ``seg_m`` (``tape``) and gathers the others'; the
    lean tail passes take the host's chunk counts, a group whose one-pass lean layout fits the arena running its
    tail in one pass.  Over the issue's pack (two groups, the second at an odd chunk; one document per group), a
    pack above the size rule with a group that is not one run, tied chunk counts, zero-length documents,
    interleaved ties, one document and a small pack with a group that is not one run, under their groupings.  The
    saved state blocks start as NaN, so every block must be written (each is finite after the forward); the
    batched forward writes the whole problem's tables; every check is counted.  Judged on the smaller packs: the
    default call (``<pack>``) and per grouping, tail-pass count and save mode
    (``<pack>.<grouping>.t<t>.<mode>``) the batched forward (``.forward``) and the batched backward from its own
    set and from the default's (``.own_set``, ``.default_set``).  The issue's pack and the pack above the size rule
    (above 80k tokens at 4 heads) are too large to judge: every result of them is finite.  The judged packs here
    save no states: the states-saving groups of both kinds (one run and by index) are judged on a small pack under a
    lowered size rule in :func:`test_batched_arena_on_a_row_that_saves_the_states`."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    split = _saving_pack(SPLIT_LENGTHS, 4, device)
    cases = [(ISSUE_LENGTHS, ISSUE_HEADS, (1, 4), False), (split, 4, (1, 4), False)] + [
        (lengths, heads, (1, 3), True)
        for lengths, heads in RANK_PACKS[1:6]  # the smaller packs (no 300-document one)
    ]
    groupings_of = {**RANK_GROUPINGS, split: ({"batch_tokens": split[0] + split[1]},)}
    original_alloc = cake_backend._alloc

    def poisoned_alloc(shape, dtype, device_):
        t = original_alloc(shape, dtype, device_)
        # the saved state blocks (hseg, seg_state, seg_m): NaN until a kernel or a copy writes them
        if dtype == torch.float32 and tuple(shape[-2:]) == (D, D):
            t.view(torch.uint8).fill_(0xFF)
        return t

    hits: dict = {}
    total = [0]

    def run(lengths, heads, passes, inputs, first: bool) -> dict:
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, sm, arch)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, grads, saved = _run_fwd_bwd(inputs)
        pack = _shape_id((lengths, heads))
        arms = {pack: {"out": out, **grads}}
        tape_default = CakeKDANondetWorkspace.for_problem(
            S, N, heads, device, save_mode="tape"
        )
        _, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=tape_default
        )
        del tape_default
        real = _real_chunks(lengths)
        rows = slice(0, BT * real)
        ranks = _kernel_ranks(lengths)
        groupings = list(groupings_of.get(lengths, ()))
        groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
        monkeypatch.setattr(cake_backend, "_alloc", poisoned_alloc)
        try:
            for kw in groupings:
                for t in passes:
                    arena = cake_backend.CakeKDANondetArena(
                        lean=t > 1, tail_passes=t, **kw
                    )
                    for save_mode, reference in (("light", saved), ("tape", saved_t)):
                        path = "tape" if save_mode == "tape" else p.light_path
                        case = (lengths, heads, kw, t, save_mode)
                        prefix = f"{pack}.{_kw_label(kw)}.t{t}.{save_mode}"
                        ws = CakeKDANondetWorkspace.for_problem(
                            S,
                            N,
                            heads,
                            device,
                            save_mode=save_mode,
                            arena=arena,
                            doc_lengths=lengths,
                        )
                        if first:
                            packing = ws._packing(lengths)
                            for i, b in enumerate(packing.batches):
                                blocks = _expected_group_blocks(ranks, lengths, b, p.P)
                                one_run = _one_run(blocks)
                                assert (packing.blocks[i] is not None) == one_run, (
                                    case,
                                    b,
                                )
                                key = (path, "run" if one_run else "index")
                                hits[key] = hits.get(key, 0) + 1
                                # the group's tail in its own passes, with the host's counts where it takes several
                                group_passes = ws.batch_view(b, tape=True).tail_passes
                                assert group_passes == _fewest_tail_passes(
                                    ws.batch_view(b, tape=True).plan,
                                    arena,
                                    ws.bytes,
                                    save_mode,
                                ), (case, b)
                                counted = (
                                    packing.tail_counts(i, b.NC_pad_tape) is not None
                                )
                                assert counted == (group_passes > 1), (case, b)
                                if counted:
                                    hits["tail"] = hits.get("tail", 0) + 1
                                elif t > 1:
                                    hits["one_pass"] = hits.get("one_pass", 0) + 1
                        count = [0]

                        def same(a, b_, what, _case=case, _count=count):
                            # an integer table of the batched forward: the whole problem's
                            assert torch.equal(a, b_), (_case, what)
                            _count[0] += 1

                        def written(t_, what, _case=case, _count=count):
                            # a saved block or row the batched forward writes: no NaN of its allocation is left
                            _finite({what: t_}, _case)
                            _count[0] += 1

                        ws.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
                        o, saved_b = cake_kda_nondet_train_forward(
                            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
                        )
                        arms[f"{prefix}.forward"] = {"out": o}
                        written(saved_b.Aqk, "Aqk")
                        written(saved_b.Akk, "Akk")
                        for j, (a, b_) in enumerate(
                            zip(saved_b.tables, reference.tables, strict=True)
                        ):
                            same(a, b_, f"tables[{j}]")
                        want = 2 + len(reference.tables)
                        if save_mode == "light":
                            assert saved_b.version == cake_backend.SAVED_VERSION
                            for name in ("hseg", "seg_state"):
                                assert (getattr(saved_b, name) is None) == (
                                    not p.seg_save
                                )
                                if p.seg_save:
                                    written(getattr(saved_b, name), name)
                                    want += 1
                            for name in ("qg", "w", "u"):  # the real token rows
                                a, b_ = getattr(saved_b, name), getattr(reference, name)
                                assert (a is None) == (b_ is None), (case, name)
                                if b_ is not None:
                                    written(a[rows], name)
                                    want += 1
                        else:
                            assert saved_b.version == cake_backend.SAVED_VERSION_TAPE
                            written(saved_b.h[:real], "h")
                            written(saved_b.v_new[rows], "v_new")
                            written(saved_b.seg_m, "seg_m")
                            want += 3
                        for name, source in (
                            ("own_set", saved_b),
                            ("default_set", reference),
                        ):
                            ws.arena.fill_(0xFF)
                            got = cake_kda_nondet_train_backward(
                                q,
                                k,
                                v,
                                g,
                                beta,
                                cu,
                                dout,
                                saved=source,
                                workspace=ws,
                                doc_lengths=lengths,
                            )
                            arms[f"{prefix}.{name}"] = _named(got)
                        assert count[0] == want, case
                        total[0] += count[0]
                        del ws, o, saved_b, got
        finally:
            monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
        return arms

    for lengths, heads, passes, judged in cases:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, sm, arch)
        assert p.recur == "seg" and not p.proportional, lengths
        if judged:

            def draw(j, s, _lengths=lengths, _heads=heads, _passes=passes):
                inputs = _draw(_lengths, _heads, s, device)
                return run(_lengths, _heads, _passes, inputs, j == 0), inputs

            _judge(record_property, ("block_runs", lengths, heads), draw)
        else:
            inputs = make_inputs(lengths, heads, device)
            _finite_arms(run(lengths, heads, passes, inputs, True), (S, heads))
            del inputs
    for key in (
        ("light_states", "run"),
        ("light_states", "index"),
        ("tape", "run"),
        ("tape", "index"),
        "tail",
        "one_pass",
    ):
        assert hits.get(key, 0) > 0, (key, hits)
    assert total[0] > 0


def test_a_warmed_batched_workspace_captures_its_calls(
    device, manifest, record_property
) -> None:
    """A caller-owned batched workspace that ran its ``doc_lengths`` once holds their device copy (the lengths, the
    tail counts of its group in several passes and the block indices of its group that is not one run), so a
    forward and a backward captured in one CUDA graph make no copy, and the replay writes a finite ``seg_m``; a
    workspace without the copy still refuses the capture.  Judged: the eager forward and backward (``eager``) and
    the replay of the graph captured after them on the same draw (``graph``)."""
    lengths, heads = (640, 64, 320), 2
    S, N = sum(lengths), len(lengths)
    arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=3, batch_tokens=704)
    ws = CakeKDANondetWorkspace.for_problem(
        S,
        N,
        heads,
        device,
        save_mode="tape",
        arena=arena,
        doc_lengths=lengths,
    )
    assert ws.plan.recur == "seg"

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        o, saved = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
        )
        got = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, doc_lengths=lengths
        )
        _finite({"seg_m": saved.seg_m}, "eager")
        arms = {"eager": {"out": o, **_named(got)}}
        del o, saved, got
        packing = ws._doc_cache[lengths]
        if j == 0:
            assert packing.blocks[0] is None and packing.blocks[1] is not None
            assert packing.words_dev is not None and packing.index_dev is not None
            counted = [
                packing.tail_counts(i, b.NC_pad_tape) is not None
                for i, b in enumerate(packing.batches)
            ]
            assert counted == [
                ws.batch_view(b, tape=True).tail_passes > 1 for b in packing.batches
            ]
            assert any(counted)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            o, saved = cake_kda_nondet_train_forward(
                q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
            )
            got = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved,
                workspace=ws,
                doc_lengths=lengths,
            )
        after = ws._doc_cache[lengths]
        assert (
            after.words_dev is packing.words_dev
            and after.index_dev is packing.index_dev
        )
        graph.replay()
        torch.cuda.synchronize()
        assert len(got) == len(GRAD_ORDER)
        _finite({"seg_m": saved.seg_m}, "graph")
        arms["graph"] = {
            "out": o.clone(),
            **{n: t.clone() for n, t in _named(got).items()},
        }
        del graph, o, saved, got
        if j == 0:
            fresh = CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                save_mode="tape",
                arena=arena,
                doc_lengths=lengths,
            )
            refused = pytest.raises(
                RuntimeError, match="under CUDA-graph capture needs"
            )
            with refused, torch.cuda.graph(torch.cuda.CUDAGraph()):
                cake_kda_nondet_train_forward(
                    q, k, v, g, beta, cu, workspace=fresh, doc_lengths=lengths
                )
        return arms, inputs

    _judge(record_property, ("warmed_capture", lengths, heads), draw)


def test_a_batched_workspace_keeps_the_device_copy_its_graph_reads(
    device, manifest, record_property
) -> None:
    """A captured graph reads the device copy of its ``doc_lengths`` (the lengths, the tail counts of the groups in
    several passes and the block indices of its group that is not one run) at each replay, so the workspace keeps
    that packing for its lifetime: nine other packs run eagerly after the capture (the cache's eight and one more)
    evict the oldest of them, never the captured one, and the replay still writes a finite ``seg_m``.  Judged: the
    eager forward and backward (``eager``) and the replay after the nine other packs (``graph``), a graph captured
    on every draw."""
    lengths, heads = (640, 64, 320), 2
    others = [
        (640, 320, 64),
        (320, 640, 64),
        (64, 640, 320),
        (320, 64, 640),
        (64, 320, 640),
        (600, 360, 64),
        (600, 64, 360),
        (360, 600, 64),
        (64, 600, 360),
    ]
    S, N = sum(lengths), len(lengths)
    assert all(sum(o) == S and len(o) == N for o in others)
    names = ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=3, batch_tokens=704)
    # sized for the groups of the captured pack, whose first group then runs two tail passes; the other packs' groups
    # (at most two documents of at most 704 tokens) fit that arena too
    ws = CakeKDANondetWorkspace.for_problem(
        S,
        N,
        heads,
        device,
        save_mode="tape",
        arena=arena,
        doc_lengths=lengths,
    )
    assert ws.plan.recur == "seg"

    def fwd_bwd(pack, inputs):
        q, k, v, g, beta, cu, dout = (inputs[n] for n in names)
        o, saved = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=pack
        )
        got = cake_kda_nondet_train_backward(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, doc_lengths=pack
        )
        return [o, saved.seg_m, *got]

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        eager = fwd_bwd(lengths, inputs)
        _finite({"seg_m": eager[1]}, "eager")
        arms = {"eager": {"out": eager[0], **_named(eager[2:])}}
        del eager
        packing = ws._doc_cache[lengths]
        if j == 0:
            assert packing.blocks[0] is None and packing.index_dev is not None
            counted = [
                packing.tail_counts(i, b.NC_pad_tape) is not None
                for i, b in enumerate(packing.batches)
            ]
            assert counted == [
                ws.batch_view(b, tape=True).tail_passes > 1 for b in packing.batches
            ]
            assert any(counted)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = fwd_bwd(lengths, inputs)
        assert ws._doc_graphs == {lengths}
        for pack in others:
            fwd_bwd(pack, _draw(pack, heads, s, device))
        if j == 0:
            assert list(ws._doc_cache) == [lengths, *others[1:]]
        kept = ws._doc_cache[lengths]
        assert (
            kept.words_dev is packing.words_dev and kept.index_dev is packing.index_dev
        )
        graph.replay()
        torch.cuda.synchronize()
        assert len(results) == 7
        _finite({"seg_m": results[1]}, "graph")
        arms["graph"] = {
            "out": results[0].clone(),
            **{n: t.clone() for n, t in _named(results[2:]).items()},
        }
        del graph, results
        return arms, inputs

    _judge(record_property, ("device_copy_graph", lengths, heads), draw)


# --------------------------------------------------------------------------- the non-deterministic contract (device)


def _nd_fwd_bwd(inputs, *, save_mode=None, replay=False):
    """This package on ``inputs``: ``out`` and the gradients by name from the saved set of ``save_mode`` (``None``:
    the light set) or, with ``replay``, from the checkpoint replay."""
    q, k, v, g, beta, cu, dout = _ops(inputs)
    out, saved = cake_kda_nondet_train_forward(
        q, k, v, g, beta, cu, scale=SCALE, save_for_backward=True, save_mode=save_mode
    )
    grads = cake_kda_nondet_train_backward(
        q, k, v, g, beta, cu, dout, saved=None if replay else saved, scale=SCALE
    )
    return out.clone(), {n: t.clone() for n, t in zip(GRAD_ORDER, grads, strict=True)}


# The path rows: the segmented rows of the deterministic suite (up to 8 heads: S1 on the saved-set paths of sm_103a /
# sm_107a and on the replay of every architecture), a serial 12-head row (the deterministic pair on every path and
# architecture), the 16- / 32- / 64-head rows (S1 on the saved-set paths everywhere and on the replay of sm_100a /
# sm_107a) and a long pack of three documents.
PATH_ROWS = [
    ((129,), 1),
    ((63, 1, 65), 4),
    ((100, 640, 77, 1000, 129), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
    ((300, 77), 12),
    ((129, 63), 16),
    ((65, 200), 32),
    ((63, 1, 65), 64),
    ((2048, 100, 513), 4),
]
JUDGED_PATHS = (("light", None, False), ("tape", "tape", False), ("replay", None, True))


@pytest.mark.parametrize("shape", PATH_ROWS, ids=_shape_id)
def test_saved_set_and_replay_paths_are_judged(
    device, manifest, shape, record_property
) -> None:
    """The light set, the tape set and the replay on the S1 cells and on the cells of the deterministic pair
    (:func:`_s1`): ``out`` and the five gradients of each path judged under its name (``light``, ``tape``,
    ``replay``)."""
    lengths, heads = shape

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        arms = {}
        for path, save_mode, replay in JUDGED_PATHS:
            out, grads = _nd_fwd_bwd(inputs, save_mode=save_mode, replay=replay)
            arms[path] = {"out": out, **grads}
        return arms, inputs

    _judge(record_property, ("paths", shape), draw)


REPEATS = 8
REPEAT_ROWS = [
    ((2048, 100, 513), 4),
    ((63, 1, 65), 16),
    ((65, 200), 32),
    ((300, 77), 12),
]


@pytest.mark.parametrize("shape", REPEAT_ROWS, ids=_shape_id)
def test_worst_case_over_repeated_runs(
    device, manifest, shape, record_property
) -> None:
    """``REPEATS`` runs of the light set, the tape set and the replay on every draw, judged per path by the worst
    run of each output (``light``, ``tape``, ``replay``): the package promises no bits from run to run."""
    lengths, heads = shape

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        arms = {}
        for path, save_mode, replay in JUDGED_PATHS:
            runs = [
                _nd_fwd_bwd(inputs, save_mode=save_mode, replay=replay)
                for _ in range(REPEATS)
            ]
            arms[path] = [{"out": out, **grads} for out, grads in runs]
        return arms, inputs

    _judge(record_property, ("repeats", shape), draw)


# Rows of the launch check: segmented (S1 on the saved-set paths of sm_103a / sm_107a, on the replay of every
# architecture), serial 12 heads (the deterministic pair everywhere), 16 heads (segmented on sm_103a / sm_107a,
# serial on sm_100a) and 64 heads (serial, the builds with the L2 hints off on sm_107a).
LAUNCH_ROWS = [((63, 1, 65), 2), ((300, 77), 12), ((129, 63), 16), ((63, 1, 65), 64)]


def test_launches_are_this_packages_records_and_s1_on_the_s1_cells(
    device, manifest, monkeypatch
) -> None:
    """Every launch of the forward, the light set's, the tape set's and the replay's backward is a record of this
    package with a library of its own prefix, the deterministic package launches nothing, and the tail launches the
    S1 fused gradient and the S1 GLM intra program together exactly on the S1 cells (the deterministic pair's copies
    elsewhere), each the record the independent dispatch spelling (``_expected_records``) names."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("the deterministic package launched a kernel")

    monkeypatch.setattr(det_jit, "run_module", refuse)
    arch, sm = cake_jit.device_arch(device.index), cake_jit.sm_count(device.index)
    seen_s1 = seen_det = 0
    for lengths, heads in LAUNCH_ROWS:
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        launched: list[str] = []

        def recording(name, values, *, grid, device_index, _log=launched):
            _log.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, launcher=recording)
        tape_ws = CakeKDANondetWorkspace.for_problem(
            S,
            N,
            heads,
            device,
            save_mode="tape",
            launcher=recording,
        )
        p = ws.plan
        _, saved = cake_kda_nondet_train_forward(q, k, v, g, beta, cu, workspace=ws)
        _, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=tape_ws
        )
        assert not S1_RECORDS & set(launched), (lengths, heads)  # the forwards
        for path, workspace, source in (
            (p.light_path, ws, saved),
            ("tape", tape_ws, saved_t),
            ("replay", ws, None),
        ):
            launched.clear()
            cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=source, workspace=workspace
            )
            tag = (lengths, heads, arch, path)
            for name in launched:
                assert name in cake_jit.MODULES, (tag, name)
                assert cake_jit.MODULES[name]["library"].startswith("cake_kda_train_")
            want = _expected_records(p.NC_pad, N, heads, sm, arch, path)
            tail = [n for n in launched if cake_jit.MODULES[n]["stage"] in TAIL_STAGES]
            dqkg = [n for n in tail if cake_jit.MODULES[n]["stage"] == "dqkg"]
            glm = [n for n in tail if cake_jit.MODULES[n]["stage"] == "glm_intra"]
            assert dqkg == [want["dqkg"]] and glm == [want["glm_intra"]], (tag, tail)
            s1 = _s1(arch, path, heads)
            assert (dqkg[0] in S1_RECORDS) is (glm[0] in S1_RECORDS) is s1, tag
            seen_s1 += s1
            seen_det += not s1
    assert seen_s1 and seen_det


def test_gradients_accumulate_into_nonzero_grad(
    device, manifest, record_property
) -> None:
    """Two backward passes of the autograd entry into the same leaves accumulate both passes' gradients: the
    accumulated gradients halved (a power-of-two scale) are judged as ``<row>.twice`` beside the one-pass gradients
    (``<row>.once``), so a lost or duplicated update misses the tier by far; on a row whose saved set runs the S1
    pair on every architecture and one that runs it on sm_103a / sm_107a."""
    for lengths, heads in (((129, 63), 16), ((100, 640, 77), 2)):
        row = _shape_id((lengths, heads))

        def draw(j, s, _lengths=lengths, _heads=heads, _row=row):
            inputs = _draw(_lengths, _heads, s, device)
            cu, dout = inputs["cu_seqlens"], inputs["dout"]
            once = _leaves(inputs)
            cake_kda_nondet(*once, cu).backward(dout)
            twice = _leaves(inputs)
            for _ in range(2):
                cake_kda_nondet(*twice, cu).backward(dout)
            arms = {
                f"{_row}.once": _leaf_grads(once),
                f"{_row}.twice": {n: t / 2 for n, t in _leaf_grads(twice).items()},
            }
            return arms, inputs

        _judge(record_property, ("accumulate", lengths, heads), draw)


# --------------------------------------------------------------------------- the checkpoint hand-off's intra twins (device)

# The twin rows: the segmented route up to 256 documents (fwd_intra_v14's twin), the serial route and more than 256
# documents (fwd_intra's twin), and the 16-head row whose light prep writes no kb (sm_100a's paired serial rows;
# sm_103a / sm_107a segmented above 32 chains).
TWIN_ROWS = [
    ((100, 640, 77, 1000, 129), 2),
    ((65, 200), 32),
    ((65,) * 257, 2),
    ((129, 300, 70), 16),
]
# the intra programs' outputs (filled with NaN before the launch) and the light prep's rows a twin also writes
INTRA_OUTPUTS = (
    "aqk_ext",
    "akk_ext",
    "aqk_int",
    "akk_int",
    "gk",
    "kb",
    "kg",
    "qg",
    "v_new",
    "beta_int",
)
PREP_ROWS = ("gk", "kg", "beta_int", "kb")


def _nan_alloc(shape_, dtype, dev) -> torch.Tensor:
    """The backend's ``_alloc`` with floating-point tensors NaN-filled (integer tensors zero-filled)."""
    if dtype.is_floating_point:
        return torch.full(shape_, float("nan"), dtype=dtype, device=dev)
    return torch.zeros(shape_, dtype=dtype, device=dev)


def _twin_id(shape) -> str:
    lengths, heads = shape
    if len(lengths) > 8:
        return f"n{len(lengths)}-S{sum(lengths)}-h{heads}"
    return _shape_id(shape)


@pytest.mark.parametrize("shape", TWIN_ROWS, ids=_twin_id)
def test_the_twin_writes_the_light_prep_rows(
    device, manifest, monkeypatch, shape, record_property
) -> None:
    """The intra twin of the device's plan, launched by a hand-off recompute that writes a row record, against its
    intra record (launched by one that writes a tape record) and the light prep, with every floating-point output
    NaN-filled before each launch: the twin writes exactly the elements its record writes (``gk``: every element
    the record writes, on the rows the record leaves too) and ``beta_int``; the integer tables both write are
    equal; the twin writes exactly the elements of ``gk``, ``kg``, ``beta_int`` and (where the light prep writes
    it) ``kb`` the light prep writes; both forwards' saved sets have the same fields and tables.  Integration
    evidence for the hand-off: its backward reads these rows in place of the prep's.  Judged: the forward of each
    record and the light backward from its saved set (``record``, ``twin``)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    p = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    parent, twin = p.record("fwd_intra", "fwd"), p.handoff_twin
    if twin is None:
        pytest.skip(f"no intra twin of {parent} on {p.arch}")
    assert twin == HANDOFF_TWINS[parent]
    prep = p.record("bwd_prep", p.light_path)
    tag = (_twin_id(shape), p.arch, p.sm_count, parent, twin, prep)
    captured: dict = {}

    def launcher(name, values, *, grid, device_index):
        stage = cake_jit.MODULES[name]["stage"]
        watched = {"fwd_intra": INTRA_OUTPUTS, "bwd_prep": PREP_ROWS}.get(stage, ())
        bound = set(cake_jit.MODULES[name]["logical"].values())
        outs = {
            n: values[n]
            for n in watched
            if n in bound and isinstance(values.get(n), torch.Tensor)
        }
        for t in outs.values():
            if t.is_floating_point():
                t.fill_(float("nan"))
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)
        if stage == "fwd_intra":  # the tables the segmented intra program writes, too
            outs.update(
                (n, values[n])
                for n in bound
                if isinstance(values.get(n), torch.Tensor)
                and not values[n].is_floating_point()
            )
        if outs:
            captured[name] = {n: t.clone() for n, t in outs.items()}

    def same_writes(got, want, what) -> None:
        # the elements a NaN-filled output holds a value in: the write pattern, not the values
        if not want.is_floating_point():  # an integer table
            assert torch.equal(got, want), (tag, what)
            return
        assert got.shape == want.shape, (tag, what)
        differ = int((torch.isnan(got) != torch.isnan(want)).sum())
        assert differ == 0, (tag, what, f"{differ} elements written differently")

    monkeypatch.setattr(cake_backend, "_alloc", _nan_alloc)
    ws = CakeKDANondetWorkspace(p, device, backward=True, launcher=launcher)
    common = dict(
        scale=SCALE,
        save_for_backward=True,
        workspace=ws,
        save_mode=None,
        arena=None,
        doc_lengths=None,
    )
    # private hand-off state: this test's resets leave the session's counters (the
    # suite's evidence that its checkpointed tests took the hand-off) intact
    monkeypatch.setattr(cake_backend, "_handoff_devices", {})
    monkeypatch.setattr(
        cake_backend,
        "_handoff_rows_load_failures",
        type(cake_backend._handoff_rows_load_failures)(),
    )

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        ops = [inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")]
        results = {}
        for with_rows in (False, True):
            monkeypatch.setattr(cake_backend, "_HANDOFF_ROWS", with_rows)
            cake_backend._handoff_reset()
            request = {"task": 0, "index": device.index, "mode": "on"}
            out, saved = cake_backend._fwd_call(*ops, handoff=request, **common)
            assert "tensors" in request, (tag, with_rows)
            assert ("gk" in request["tensors"]) is with_rows, (tag, with_rows)
            results[with_rows] = (
                out,
                saved,
                captured.pop(twin if with_rows else parent),
            )
        cake_backend._handoff_reset()
        (out0, saved0, rec0), (out1, saved1, rec1) = results[False], results[True]
        arms = {}
        rows = None
        for name, out, saved in (("record", out0, saved0), ("twin", out1, saved1)):
            grads = cake_backend._bwd_call(
                *ops,
                inputs["dout"],
                saved=saved,
                scale=SCALE,
                workspace=ws,
                arena=None,
                doc_lengths=None,
            )
            arms[name] = {"out": out, **_named(grads)}
            # the light prep's rows of the tape-record forward's saved set
            prep_rows = captured.pop(prep)
            rows = prep_rows if rows is None else rows
        if j == 0:
            for field in dataclasses.fields(saved0):
                a, b = getattr(saved1, field.name), getattr(saved0, field.name)
                if isinstance(b, torch.Tensor):
                    assert a.shape == b.shape and a.dtype == b.dtype, (tag, field.name)
                elif isinstance(b, tuple):  # the tables
                    assert len(a) == len(b) and all(
                        torch.equal(x, y) for x, y in zip(a, b, strict=True)
                    ), (tag, field.name)
                else:
                    assert a == b, (tag, field.name)
            # the twin's outputs against its intra record's
            assert set(rec1) == set(rec0) | {"beta_int"}, (
                tag,
                sorted(rec0),
                sorted(rec1),
            )
            for name, want in rec0.items():
                got = rec1[name]
                # gk: the record writes some rows of each chunk, the twin all of them
                if name == "gk":
                    written = ~torch.isnan(want)
                    assert written.any(), tag
                    assert not bool((torch.isnan(got) & written).any()), (tag, name)
                else:
                    same_writes(got, want, name)
            # the twin's rows against the light prep's
            assert {"gk", "kg", "beta_int"} <= set(rows), (tag, sorted(rows))
            for name, want in rows.items():
                same_writes(rec1[name], want, ("prep", name))
        return arms, inputs

    _judge(record_property, ("handoff_twin", lengths, heads), draw)


# --------------------------------------------------------------------------- the tape path's w-only WY twin

# User tape mode (a version-2 saved set) at one head and every head count the checkpoint hand-off runs, on one pack of
# short documents with unaligned tails (both routes, by head count), pad chunks at 64 heads, more than 256 documents
# and the four-document 261k-token pack P8 at 32 heads; P8 at 64 heads is opt-in (``CAKE_KDA_TAPE_WY_TEST_P8_H64=1``,
# for memory and time).
TAPE_WY_P8 = (23496, 88510, 114974, 34413)
TAPE_WY_P8_H64_ENV = "CAKE_KDA_TAPE_WY_TEST_P8_H64"
TAPE_WY_ROWS = [
    *(((100, 640, 77, 1000, 129), heads) for heads in (1, 2, 4, 8, 12, 16, 32, 64)),
    ((63, 1, 65), 64),
    ((65,) * 257, 2),
    ((65,) * 257, 32),
    (TAPE_WY_P8, 32),
    (TAPE_WY_P8, 64),
]


@pytest.fixture
def private_tape_wy_state(monkeypatch):
    """Fresh tape WY counters and failed loads for one test (restored after it), so the session's counters stay
    intact."""
    monkeypatch.setattr(
        cake_backend, "_tape_wy_counts", type(cake_backend._tape_wy_counts)()
    )
    monkeypatch.setattr(
        cake_backend,
        "_tape_wy_load_failures",
        type(cake_backend._tape_wy_load_failures)(),
    )


def _tape_plan(lengths, heads, device):
    return cake_backend.plan(
        sum(lengths),
        len(lengths),
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
    )


def _wy_runs(monkeypatch) -> list:
    """Record ``(path, record)`` of every ``wy`` launch through ``CakeKDANondetWorkspace.run`` (``record``: the name that
    replaces the plan's record, ``None`` for the plan's own)."""
    runs: list = []
    real_run = CakeKDANondetWorkspace.run

    def run(self, stage, values, *, path, **kwargs):
        if stage == "wy":
            runs.append((path, kwargs.get("record")))
        return real_run(self, stage, values, path=path, **kwargs)

    monkeypatch.setattr(CakeKDANondetWorkspace, "run", run)
    return runs


def _judged_row(lengths, heads) -> bool:
    """Whether a row is judged: up to 4096 tokens at any head count, or a production-size row of up to about 16k
    tokens at up to four heads (the reference's FP64 token recurrence runs per token, twice with the naive tier)."""
    S = sum(lengths)
    return S <= 4096 or (S <= 17000 and heads <= 4)


@pytest.mark.parametrize("shape", TAPE_WY_ROWS, ids=_twin_id)
def test_tape_mode_with_the_tape_wy_twin_is_judged(
    device, manifest, monkeypatch, shape, record_property
) -> None:
    """User tape mode: the tape backward with the twin forced (``_TAPE_WY = True``) launches the twin in place of
    its WY record, on the record's grid, in the same launch sequence and count; the twin writes ``w`` on exactly the
    elements the record writes (pad rows included; both start NaN-filled) and leaves the arena's ``u`` unwritten
    (it takes no ``u``).  Judged: the tape forward (arm ``""``) and the gradients of the backward that launches the
    record (``record``) and of the one that launches the twin (``twin``).  More than 256 documents at 32 heads and
    the 261k-token pack P8 are too large to judge (:func:`_judged_row`): every result of them is finite; the twin is
    judged at every head count on the five-document pack and at 2 heads on more than 256 documents."""
    lengths, heads = shape
    if tuple(lengths) == TAPE_WY_P8 and heads == 64:
        if os.environ.get(TAPE_WY_P8_H64_ENV) != "1":
            pytest.skip(f"opt-in: {TAPE_WY_P8_H64_ENV}=1")
    p = _tape_plan(lengths, heads, device)
    parent, twin = p.record("wy", "tape"), p.tape_wy_twin
    assert twin == TAPE_TWINS[parent], (p.arch, parent)
    tag = (_twin_id(shape), p.arch, p.sm_count, parent, twin)
    launched: list[str] = []
    captured: dict = {}

    def launcher(name, values, *, grid, device_index):
        launched.append(name)
        wy = cake_jit.MODULES[name]["stage"] == "wy"
        if wy:
            values["w"].fill_(float("nan"))
            values["u"].fill_(float("nan"))
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)
        if wy:
            captured[name] = (
                values["w"].clone(),
                bool(torch.isnan(values["u"]).all()),
                tuple(grid),
            )

    ws = CakeKDANondetWorkspace(
        p,
        device,
        backward=True,
        save_mode="tape",
        launcher=launcher,
    )

    def run(inputs, first: bool) -> dict:
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, save_mode="tape"
        )
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE, tag
        arms = {"": {"out": out}}
        for forced in (False, True):
            monkeypatch.setattr(cake_backend, "_TAPE_WY", forced)
            launched.clear()
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved_t, workspace=ws
            )
            if first:
                want = _expected_launches(p, "tape", tape_wy=twin if forced else None)
                assert launched == want, (tag, forced, launched)
                assert len(launched) == p.launch_count("tape"), (tag, forced)
            arms["twin" if forced else "record"] = _named(got)
            del got
        (w0, u0_unwritten, grid0), (w1, u1_unwritten, grid1) = (
            captured.pop(parent),
            captured.pop(twin),
        )
        assert grid1 == grid0 == tuple(p.grid("wy", "tape")), tag
        assert w1.shape == w0.shape == (p.R, heads, D), (tag, tuple(w1.shape))
        # the elements of w each launch writes (both NaN-filled before it): the write pattern, not the values
        differ = int((torch.isnan(w1) != torch.isnan(w0)).any(dim=2).sum())
        assert differ == 0, (
            tag,
            "w",
            f"{differ} (row, head) pairs written differently",
        )
        assert u1_unwritten and not u0_unwritten, (tag, "u")
        return arms

    if _judged_row(lengths, heads):

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device)
            return run(inputs, j == 0), inputs

        _judge(record_property, ("tape_wy_twin", lengths, heads), draw)
    else:
        _finite_arms(run(make_inputs(lengths, heads, device), True), tag)


def test_batched_tape_mode_launches_the_tape_wy_twin_per_group(
    device, manifest, monkeypatch, record_property
) -> None:
    """A batched arena (document groups): the tape backward of every group launches the twin in place of its WY
    record when forced.  Judged: the batched tape forward (arm ``""``) and the gradients of the backward that
    launches the record (``record``) and of the one that launches the twin (``twin``)."""
    lengths, heads = (100, 640, 77, 1000, 129), 4
    S, N = sum(lengths), len(lengths)
    arena = cake_backend.CakeKDANondetArena(batch_tokens=1000)
    ws = CakeKDANondetWorkspace.for_problem(
        S,
        N,
        heads,
        device,
        save_mode="tape",
        arena=arena,
        doc_lengths=lengths,
    )
    p = ws.plan
    groups = len(p.batches(arena, lengths))
    assert groups > 1, groups
    twin = p.tape_wy_twin
    assert twin == TAPE_TWINS[p.record("wy", "tape")], p.arch
    runs = _wy_runs(monkeypatch)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
        )
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
        arms = {"": {"out": out}}
        for forced in (False, True):
            monkeypatch.setattr(cake_backend, "_TAPE_WY", forced)
            runs.clear()
            got = cake_kda_nondet_train_backward(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved_t,
                workspace=ws,
                doc_lengths=lengths,
            )
            assert runs == [("tape", twin if forced else None)] * groups, (forced, runs)
            arms["twin" if forced else "record"] = _named(got)
        return arms, inputs

    _judge(record_property, ("batched_tape_wy_twin", lengths, heads), draw)


def test_the_tape_wy_rule(monkeypatch, private_tape_wy_state) -> None:
    """Host only (a stand-in plan): the tape backward launches the twin in place of its WY record when the plan has
    one, the device (architecture, SM count) has an entry in ``TAPE_WY_VERIFIED`` (else ``unverified``, the build
    not read), the toolkit of the one build of the twin and the plan's WY record completes the key (another
    toolkit: ``unverified``; a mixed build or an unreadable toolkit: ``unready``) and the twin loads (else
    ``unready``); ``_TAPE_WY`` overrides all of it but the twin.  The rule counts no launch (the backward does).
    The shipped entries name exported architectures and ``major.minor.patch`` toolkits."""
    for key in cake_backend.TAPE_WY_VERIFIED:
        arch, sms, toolkit = key
        assert arch in ARCHES and sms > 0, key
        assert re.fullmatch(r"\d+\.\d+\.\d+", toolkit), key
    arch = ARCHES[0]
    monkeypatch.setattr(
        cake_backend, "TAPE_WY_VERIFIED", frozenset({(arch, 152, "9.9.1")})
    )
    asked: list[tuple] = []
    state = {"toolkit": "9.9.1", "loads": True}

    def toolkit_(twin, parent, arch_):
        asked.append(("toolkit", twin, parent, arch_))
        return state["toolkit"]

    def loads_(twin, arch_):
        asked.append(("load", twin, arch_))
        return state["loads"]

    monkeypatch.setattr(cake_backend, "_tape_wy_toolkit", toolkit_)
    monkeypatch.setattr(cake_backend, "_tape_wy_loads", loads_)

    def plan_(twin="twin", sms=152, arch_=arch):
        return types.SimpleNamespace(
            tape_wy_twin=twin,
            arch=arch_,
            sm_count=sms,
            record=lambda stage, path: f"{stage}@{path}",
        )

    # (override, plan, toolkit, loads, record, counter, checks asked)
    cases = (
        (None, plan_(), "9.9.1", True, "twin", None, 2),
        (None, plan_(sms=148), "9.9.1", True, None, "unverified", 0),
        (None, plan_(arch_=ARCHES[1]), "9.9.1", True, None, "unverified", 0),
        (None, plan_(), "9.9.2", True, None, "unverified", 1),
        (None, plan_(), None, True, None, "unready", 1),
        (None, plan_(), "9.9.1", False, None, "unready", 2),
        (None, plan_(twin=None), "9.9.1", True, None, None, 0),
        (False, plan_(), "9.9.1", True, None, None, 0),
        (True, plan_(sms=148), None, False, "twin", None, 0),
        (True, plan_(twin=None), "9.9.1", True, None, None, 0),
    )
    for override, p, toolkit, loads, want, counter, checks in cases:
        tag = (override, p.arch, p.sm_count, p.tape_wy_twin, toolkit, loads)
        cake_backend._tape_wy_reset()
        asked.clear()
        monkeypatch.setattr(cake_backend, "_TAPE_WY", override)
        state.update(toolkit=toolkit, loads=loads)
        assert cake_backend._tape_wy_record(p, 0) == want, tag
        s = cake_backend._tape_wy_stats(0)
        assert set(s) == set(cake_backend._TAPE_WY_COUNTERS), (tag, s)
        assert s["launched"] == 0, (tag, s)
        for name in ("unverified", "unready"):
            assert s[name] == int(name == counter), (tag, s)
        # the build is read last but one, on the twin and the plan's tape WY record; the load last
        expected = [("toolkit", "twin", "wy@tape", p.arch), ("load", "twin", p.arch)]
        assert asked == expected[:checks], (tag, asked)


def test_the_tape_wy_toolkit_reads_one_build_of_the_twin_and_its_record(
    monkeypatch,
) -> None:
    """Host only: the toolkit of the tape WY twin is the one build's of the twin and the tape path's WY record
    (``_one_build_toolkit`` of the two, the rule of the hand-off's row records), cached per arguments."""
    seen: list = []

    def one_build(records, arch):
        seen.append((tuple(records), arch))
        return "13.4.92"

    monkeypatch.setattr(cake_backend, "_one_build_toolkit", one_build)
    cake_backend._tape_wy_toolkit.cache_clear()
    try:
        for _ in range(2):
            got = cake_backend._tape_wy_toolkit("twin", "parent", "sm_100a")
            assert got == "13.4.92"
        assert seen == [(("twin", "parent"), "sm_100a")], seen
    finally:
        cake_backend._tape_wy_toolkit.cache_clear()


def test_a_failed_tape_wy_load_is_retried_a_bounded_number_of_times(
    monkeypatch, private_tape_wy_state
) -> None:
    """Host only: a failed load of the twin is counted, not kept (the next tape backward loads again) up to
    ``_TAPE_WY_LOAD_TRIES`` failures per twin and architecture; after those the process launches the WY record
    until ``_tape_wy_reset``.  The hand-off's failed twin loads count apart."""
    loads: list[tuple] = []
    state = {"fail": True}

    def load(name, arch):
        loads.append((name, arch))
        if state["fail"]:
            raise RuntimeError("JIT lock timeout")
        return types.SimpleNamespace()

    monkeypatch.setattr(cake_jit, "load", load)
    monkeypatch.setattr(
        cake_backend,
        "_handoff_rows_load_failures",
        type(cake_backend._handoff_rows_load_failures)(),
    )
    tries = cake_backend._TAPE_WY_LOAD_TRIES
    assert tries >= 2
    for _ in range(tries + 2):
        assert cake_backend._tape_wy_loads("twin", "sm_100a") is False
    assert loads == [("twin", "sm_100a")] * tries, loads
    assert not cake_backend._handoff_rows_load_failures
    # another architecture counts apart; a success is not counted
    state["fail"] = False
    assert cake_backend._tape_wy_loads("twin", "sm_103a") is True
    assert cake_backend._tape_wy_loads("twin", "sm_100a") is False
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True
    cake_backend._tape_wy_reset()
    assert cake_backend._tape_wy_loads("twin", "sm_100a") is True
    assert loads[tries:] == [("twin", "sm_103a")] + [("twin", "sm_100a")] * 2, loads


class _Built:
    """A stand-in for ``cake_build.prebuilt_path(name)``: ``is_file()`` is whether the library counts as prebuilt."""

    def __init__(self, built: bool) -> None:
        self.built = built

    def is_file(self) -> bool:
        return self.built


@pytest.mark.parametrize(
    "case", ("unverified", "mixed_build", "load_fails", "verified")
)
def test_the_tape_wy_twin_falls_back_to_its_record_where_the_rule_refuses_it(
    device, manifest, monkeypatch, private_tape_wy_state, case, record_property
) -> None:
    """The rule (``_TAPE_WY`` unset) on a user tape-mode backward, this device listed in ``TAPE_WY_VERIFIED`` with
    this build's toolkit (another patch version in the ``unverified`` case): the twin launches only with the one
    build and a twin that loads (``verified``: counted ``launched``); otherwise the WY record launches and the
    refusal is counted: another toolkit listed (``unverified``), the twin and the record from different builds
    (``mixed_build``: ``unready``), the twin's load failing (``load_fails``: ``unready``, the fail-closed fallback).
    Judged: the tape forward (arm ``""``), the backward that launches the record (``record``) and the rule's
    backward (``rule``)."""
    lengths, heads = (100, 640, 77, 1000, 129), 4
    p = _tape_plan(lengths, heads, device)
    parent, twin = p.record("wy", "tape"), p.tape_wy_twin
    assert twin == TAPE_TWINS[parent], (p.arch, parent)
    verified = case == "verified"
    state: dict = {}

    def patch_the_build_state() -> list:
        cake_backend._tape_wy_toolkit.cache_clear()
        toolkit = cake_backend._tape_wy_toolkit(twin, parent, p.arch)
        assert toolkit is not None, "this build's twin and WY record are not one build"
        cake_backend._tape_wy_toolkit.cache_clear()
        major, minor, patch = toolkit.split(".")
        listed = (
            toolkit if case != "unverified" else f"{major}.{minor}.{int(patch) + 1}"
        )
        monkeypatch.setattr(
            cake_backend, "TAPE_WY_VERIFIED", frozenset({(p.arch, p.sm_count, listed)})
        )
        if case == "mixed_build":
            # one of the two libraries flips its build state: the twin's where nothing is prebuilt here, else the
            # WY record's (loaded by the reference backward, so its launch does not read the flipped state)
            twin_library = cake_jit.library_name(twin, p.arch)
            parent_library = cake_jit.library_name(parent, p.arch)
            real_prebuilt = cake_build.prebuilt_path
            flipped = (
                parent_library
                if real_prebuilt(twin_library).is_file()
                else twin_library
            )

            def prebuilt(name):
                if name == flipped:
                    return _Built(not real_prebuilt(name).is_file())
                return real_prebuilt(name)

            monkeypatch.setattr(cake_build, "prebuilt_path", prebuilt)
        elif case == "load_fails":
            real_load = cake_jit.load

            def load(name, arch):
                if name == twin:
                    raise RuntimeError("the twin does not load")
                return real_load(name, arch)

            monkeypatch.setattr(cake_jit, "load", load)
        return _wy_runs(monkeypatch)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, save_mode="tape"
        )
        # the record's backward first: on the first draw it loads the WY record before any build state is patched
        monkeypatch.setattr(cake_backend, "_TAPE_WY", False)
        want = cake_kda_nondet_train_backward(q, k, v, g, beta, cu, dout, saved=saved_t)
        monkeypatch.setattr(cake_backend, "_TAPE_WY", None)
        if j == 0:
            state["runs"] = patch_the_build_state()
        runs = state["runs"]
        runs.clear()
        cake_backend._tape_wy_toolkit.cache_clear()
        try:
            got = cake_kda_nondet_train_backward(
                q, k, v, g, beta, cu, dout, saved=saved_t
            )
        finally:
            cake_backend._tape_wy_toolkit.cache_clear()
        assert runs == [("tape", twin if verified else None)], (case, runs)
        if j == 0:
            stats = cake_backend._tape_wy_stats(device.index)
            assert stats["launched"] == int(verified), (case, stats)
            assert stats["unverified"] == int(case == "unverified"), (case, stats)
            assert stats["unready"] == int(case in ("mixed_build", "load_fails")), (
                case,
                stats,
            )
        arms = {"": {"out": out}, "record": _named(want), "rule": _named(got)}
        return arms, inputs

    _judge(record_property, ("tape_wy_fallback", case), draw)


def test_the_tape_wy_rule_on_this_device(
    device, manifest, monkeypatch, private_tape_wy_state, record_property
) -> None:
    """As shipped (``_TAPE_WY`` unset, nothing patched): the toolkit of the one build of the twin and the tape path's
    WY record reads as ``major.minor.patch``; a user tape-mode backward launches the twin exactly when this device
    with that toolkit is in ``TAPE_WY_VERIFIED`` (else ``unverified``).  Judged: the tape forward (arm ``""``), the
    backward that launches the record (``record``) and the rule's backward (``rule``)."""
    lengths, heads = (63, 1, 65), 2
    p = _tape_plan(lengths, heads, device)
    parent, twin = p.record("wy", "tape"), p.tape_wy_twin
    assert twin == TAPE_TWINS[parent], (p.arch, parent)
    toolkit = cake_backend._tape_wy_toolkit(twin, parent, p.arch)
    assert re.fullmatch(r"\d+\.\d+\.\d+", toolkit or ""), toolkit
    listed = (p.arch, p.sm_count, toolkit) in cake_backend.TAPE_WY_VERIFIED
    key = (p.arch, p.sm_count, toolkit)
    runs = _wy_runs(monkeypatch)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device)
        q, k, v, g, beta, cu, dout = _ops(inputs)
        out, saved_t = cake_kda_nondet_train_forward(
            q, k, v, g, beta, cu, save_mode="tape"
        )
        monkeypatch.setattr(cake_backend, "_TAPE_WY", False)
        want = cake_kda_nondet_train_backward(q, k, v, g, beta, cu, dout, saved=saved_t)
        monkeypatch.setattr(cake_backend, "_TAPE_WY", None)
        runs.clear()
        got = cake_kda_nondet_train_backward(q, k, v, g, beta, cu, dout, saved=saved_t)
        assert runs == [("tape", twin if listed else None)], (key, runs)
        if j == 0:
            stats = cake_backend._tape_wy_stats(device.index)
            assert stats == {
                "launched": int(listed),
                "unverified": int(not listed),
                "unready": 0,
            }, (key, stats)
        arms = {"": {"out": out}, "record": _named(want), "rule": _named(got)}
        return arms, inputs

    _judge(record_property, ("tape_wy_rule", lengths, heads), draw)


# ---------------------------------------------------------------------------------------------------------------------
# The raw gate mode (``use_gate_in_kernel=True``) on this package: the BF16 raw gate (any view with a unit innermost
# stride and 16-byte token / head strides), ``A_log`` FP32 [H], ``dt_bias`` FP32 [H * 128] or [H, 128] and the lower
# bound in place of the activated FP32 ``g``; the backward returns the BF16 gradient of the raw gate and the two
# parameter gradients too.  The raw mode runs the deterministic package's raw-mode programs in the same order (the
# fused gradient's raw class copies the deterministic records, no S1 pair, no checkpoint hand-off): its launch lists
# are the deterministic package's (a mode contract).  Every path is judged against the FP64 recurrence whose gate is
# activated in the same graph, beside the naive BF16 tier: repeats, the replay, the light, light-states and tape sets
# (each package's backward reads its own sets and refuses the other package's), caller workspaces, the lean tail
# passes, the batched groups, strided views, autograd and ``torch.utils.checkpoint``.
#
# The parameter gradients included: on every architecture the raw backward runs the deterministic package's GLM intra
# program (``glm_intra_raw``, writing the per-chunk partials) and ``param_finalize`` (reducing them in a fixed order),
# and no call raises PyTorch's non-determinism alert under ``torch.use_deterministic_algorithms(True)``.

RAW_LEAVES = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")
RAW_LEAF_GRADS = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
RAW_RETURN_NAMES = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")
# the raw mode's parameter-gradient finalize record (the in-kernel Q/K L2 norm off), on every architecture
RAW_FINALIZE = "param_finalize"


def _rg_kw(inputs) -> dict:
    return {
        "use_gate_in_kernel": True,
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lower_bound": LOWER_BOUND,
    }


def _rg_ops(inputs):
    return tuple(inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout"))


def _rg_fwd(inputs, *, g=None, **kwargs):
    """The raw-mode forward of this package on ``inputs`` (``g``: another view of the same raw gate)."""
    q, k, v, raw, beta, cu, _dout = _rg_ops(inputs)
    return cake_kda_nondet_train_forward(
        q,
        k,
        v,
        raw if g is None else g,
        beta,
        cu,
        scale=SCALE,
        **_rg_kw(inputs),
        **kwargs,
    )


def _rg_bwd(inputs, saved, *, g=None, **kwargs):
    """The raw-mode backward of this package from ``saved`` (``None``: the replay)."""
    q, k, v, raw, beta, cu, dout = _rg_ops(inputs)
    return cake_kda_nondet_train_backward(
        q,
        k,
        v,
        raw if g is None else g,
        beta,
        cu,
        dout,
        saved=saved,
        scale=SCALE,
        **_rg_kw(inputs),
        **kwargs,
    )


def _rg_det_fwd(inputs, **kwargs):
    """The deterministic package's raw-mode forward on ``inputs``: the other package's saved sets of the refusals
    and its launch lists (mode contracts); none of its values is compared with this package's."""
    q, k, v, raw, beta, cu, _dout = _rg_ops(inputs)
    # mode contract
    return det_backend.cake_kda_det_train_forward(
        q, k, v, raw, beta, cu, scale=SCALE, **_rg_kw(inputs), **kwargs
    )


def _rg_det_bwd(inputs, saved, **kwargs):
    """The deterministic package's raw-mode backward from ``saved``: its refusal of this package's sets and its
    launch lists (mode contracts)."""
    q, k, v, raw, beta, cu, dout = _rg_ops(inputs)
    # mode contract
    return det_backend.cake_kda_det_train_backward(
        q,
        k,
        v,
        raw,
        beta,
        cu,
        dout,
        saved=saved,
        scale=SCALE,
        **_rg_kw(inputs),
        **kwargs,
    )


def _rg_run(inputs, *, g=None, save=True, **kwargs):
    """The raw-mode forward with its saved set and the backward from it: ``out``, the seven results by name
    (:data:`RAW_RETURN_NAMES`) and the saved set."""
    bwd_kwargs = {
        n: kwargs[n] for n in ("workspace", "arena", "doc_lengths") if n in kwargs
    }
    out, saved = _rg_fwd(inputs, g=g, save_for_backward=save, **kwargs)
    grads = _rg_bwd(inputs, saved, g=g, **bwd_kwargs)
    assert len(grads) == len(RAW_RETURN_NAMES)
    return out, dict(zip(RAW_RETURN_NAMES, grads, strict=True)), saved


def _rg_named(results) -> dict:
    """A raw-mode backward's seven results by name (:data:`RAW_RETURN_NAMES`)."""
    results = list(results)
    assert len(results) == len(RAW_RETURN_NAMES)
    return _named(results, RAW_RETURN_NAMES)


def _rg_leaves(inputs) -> list[torch.Tensor]:
    return [inputs[n].detach().clone().requires_grad_(True) for n in RAW_LEAVES]


def _rg_autograd(leaves, cu, **kwargs) -> torch.Tensor:
    q, k, v, raw, beta, a_log, dt_bias = leaves
    return cake_kda_nondet(
        q,
        k,
        v,
        raw,
        beta,
        cu,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        **kwargs,
    )


def _rg_leaf_grads(leaves) -> dict:
    """The raw-mode leaves' gradients by name (:data:`RAW_LEAF_GRADS`)."""
    return _leaf_grads(leaves, RAW_LEAF_GRADS)


def _one_finalize_last(names: list, p, path: str, tag=None) -> None:
    """The raw backward's launch list ends with its one parameter-gradient finalize, the plan's record of the
    stage on ``path``: the deterministic ``param_finalize`` on every architecture."""
    finals = [n for n in names if cake_jit.MODULES[n]["stage"] == "param_finalize"]
    assert finals == [RAW_FINALIZE] == [p.record("param_finalize", path)], (
        tag,
        finals,
    )
    assert names[-1] == RAW_FINALIZE, (tag, names[-1])


def _rg_recorder(launched: list, jit=None):
    """A workspace launcher that records ``(record, grid, num_seqs, num_chunks_pad)`` and runs the record (``jit``:
    the package's ``cake_jit``, this package's by default)."""
    jit = cake_jit if jit is None else jit

    def recording(name, values, *, grid, device_index):
        launched.append(
            (
                name,
                tuple(grid),
                int(values.get("num_seqs", -1)),
                int(values.get("num_chunks_pad", -1)),
            )
        )
        jit.run_module(name, values, grid=grid, device_index=device_index)

    return recording


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_raw_gate_matches_the_exact_reference(
    device, manifest, shape, record_property
) -> None:
    """The raw mode judged against the exact FP64 recurrence whose gate is activated from the same BF16 raw gate and
    FP32 parameters in FP64, beside the naive BF16 tier: ``out`` and the seven results, the parameter gradients
    included; the result dtypes and shapes (``dg`` the BF16 contiguous gradient of the raw gate, ``dA_log`` FP32
    ``[H]``, ``ddt_bias`` FP32 in ``dt_bias``'s shape); the saved set is this package's type and records the gate
    mode and the lower bound."""
    lengths, heads = shape
    S = sum(lengths)

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device, gate="raw")
        out, grads, saved = _rg_run(inputs)
        assert type(saved) is CakeKDANondetChunkSaved
        assert (saved.gate, saved.lower_bound) == ("raw", LOWER_BOUND)
        assert out.shape == (S, heads, D) and out.dtype == torch.bfloat16
        assert grads["dg"].shape == (S, heads, D)
        assert grads["dg"].dtype == torch.bfloat16 and grads["dg"].is_contiguous()
        assert grads["dA_log"].shape == (heads,)
        assert grads["dA_log"].dtype == torch.float32
        assert grads["ddt_bias"].shape == (heads * D,)
        assert grads["ddt_bias"].dtype == torch.float32
        return {"": {"out": out, **grads}}, inputs

    _judge(record_property, ("raw_gate", shape), draw)


@pytest.mark.parametrize("shape", SHAPES[6:10] + SHAPES[11:13], ids=_shape_id)
def test_raw_gate_paths_are_judged(device, manifest, shape, record_property) -> None:
    """Every raw-mode path on one call's inputs, each judged under its own name: the saved set's backward and two
    repeats (arm ``""``, the worst of the three), the replay (``replay``), a saved set without its tables and the
    set on caller workspaces of the raw mode (``bare``, ``saved_workspace``), a caller workspace's forward and
    backward and its replay (``workspace``, ``workspace_replay``), the tape set from the default entry and from a
    tape workspace and that workspace's replay (``tape``, ``tape_workspace``, ``tape_workspace_replay``) and a light
    set on the tape workspace (``light_set_on_tape``).  A raw workspace holds the arena alone, the byte query's
    bytes and the deterministic package's (a mode contract); each package's backward refuses the other package's
    light and tape sets (``ValueError`` naming the mode)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, gate="raw")
    assert ws.gate == ws.plan.gate == "raw"
    # the arena alone, the deterministic package's bytes (the parameter partials are per call): a byte query of
    # the other package, a mode contract
    assert (
        ws.bytes
        == cake_backend.workspace_bytes(
            S, N, heads, sm_count=ws.plan.sm_count, gate="raw"
        )
        == det_backend.workspace_bytes(
            S, N, heads, sm_count=ws.plan.sm_count, gate="raw"
        )
    )
    tape_ws = CakeKDANondetWorkspace.for_problem(
        S, N, heads, device, save_mode="tape", gate="raw"
    )

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device, gate="raw")
        runs = [_rg_run(inputs) for _ in range(3)]
        saved = runs[0][2]
        arms = {"": [{"out": out, **grads} for out, grads, _ in runs]}
        arms["replay"] = _rg_named(_rg_bwd(inputs, None))
        bare = dataclasses.replace(saved, tables=None)
        assert (bare.gate, bare.lower_bound) == ("raw", LOWER_BOUND)
        for name, other in (("bare", bare), ("saved_workspace", saved)):
            ws_ = CakeKDANondetWorkspace.for_problem(S, N, heads, device, gate="raw")
            assert ws_.gate == ws_.plan.gate == "raw"
            arms[name] = _rg_named(_rg_bwd(inputs, other, workspace=ws_))
        out_w, grads_w, _ = _rg_run(inputs, workspace=ws)
        arms["workspace"] = {"out": out_w, **grads_w}
        arms["workspace_replay"] = _rg_named(_rg_bwd(inputs, None, workspace=ws))
        out_t, grads_t, saved_t = _rg_run(inputs, save_mode="tape")
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
        assert saved_t.gate == "raw"
        arms["tape"] = {"out": out_t, **grads_t}
        out_tw, grads_tw, saved_tw = _rg_run(inputs, workspace=tape_ws)
        assert saved_tw.version == cake_backend.SAVED_VERSION_TAPE
        arms["tape_workspace"] = {"out": out_tw, **grads_tw}
        arms["tape_workspace_replay"] = _rg_named(
            _rg_bwd(inputs, None, workspace=tape_ws)
        )
        arms["light_set_on_tape"] = _rg_named(_rg_bwd(inputs, saved, workspace=tape_ws))
        if j == 0:
            # each package's backward refuses the other's raw sets (light and tape): mode contracts
            _, det_saved = _rg_det_fwd(inputs)  # mode contract
            _, det_saved_t = _rg_det_fwd(inputs, save_mode="tape")  # mode contract
            for nd_set, det_set in ((saved, det_saved), (saved_t, det_saved_t)):
                with pytest.raises(ValueError, match=ND_REFUSES):
                    _rg_bwd(inputs, det_set)
                with pytest.raises(ValueError, match=DET_REFUSES):
                    _rg_det_bwd(inputs, nd_set)  # mode contract
        return arms, inputs

    _judge(record_property, ("raw_gate_paths", shape), draw)


@pytest.mark.parametrize(("num_docs", "heads"), STATE_SHAPES, ids=["n1-h1", "n2-h2"])
def test_raw_gate_saved_states_are_judged(
    device, manifest, num_docs, heads, record_property
) -> None:
    """The segmented rows that save the segment-boundary states (``light_states``) in raw mode: the backward from
    that set on lean workspaces launches the tail once per range and the finalize once, last.  On ``n1-h1`` (one
    document at one head at the threshold, about 14.6k tokens) every result is judged: the set's backward (arm
    ``""``), the replay (``replay``), the tape set (``tape``) and the lean workspaces (``lean1``, ``lean3``).  On
    ``n2-h2`` (about 29k tokens) every result is finite."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    nseg = cake_backend.plan(1, num_docs, heads, sm, arch).NSEG
    S = cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    lengths = _split(S, num_docs)
    p = cake_backend.plan(S, num_docs, heads, sm, arch, gate="raw")
    assert p.recur == "seg" and p.seg_save and p.light_path == "light_states"
    launched: list = []
    lean = {
        t: CakeKDANondetWorkspace.for_problem(
            S,
            num_docs,
            heads,
            device,
            arena=cake_backend.CakeKDANondetArena(lean=True, tail_passes=t),
            gate="raw",
            launcher=_rg_recorder(launched),
        )
        for t in (1, 3)
    }

    def run(inputs, first: bool) -> dict:
        out, grads, saved = _rg_run(inputs)
        assert saved.hseg is not None and saved.gate == "raw"
        arms = {"": {"out": out, **grads}}
        arms["replay"] = _rg_named(_rg_bwd(inputs, None))
        out_t, grads_t, _ = _rg_run(inputs, save_mode="tape")
        arms["tape"] = {"out": out_t, **grads_t}
        for t, ws in lean.items():
            launched.clear()
            got = _rg_bwd(inputs, saved, workspace=ws)
            if first:
                names = [n for n, *_ in launched]
                assert names == _expected_launches(
                    ws.plan, "light_states", tail_passes=t
                )
                _one_finalize_last(names, ws.plan, "light_states", tag=t)
                _check_tail_grids(
                    [(n, g) for n, g, *_ in launched], ws.plan, "light_states", t
                )
            arms[f"lean{t}"] = _rg_named(got)
        return arms

    if num_docs == 1:

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device, gate="raw")
            return run(inputs, j == 0), inputs

        _judge(record_property, ("raw_gate_saved_states", lengths, heads), draw)
    else:
        _finite_arms(run(make_raw_inputs(lengths, heads, device), True), (S, heads))


@pytest.mark.parametrize("shape", [((63, 1, 65), 2), ((129, 63), 16)], ids=_shape_id)
def test_raw_gate_autograd_and_checkpoint_are_judged(
    device, manifest, monkeypatch, shape, record_property
) -> None:
    """The autograd entry gives the leaves the raw mode's gradients (``dg`` to the raw gate, ``dA_log`` /
    ``ddt_bias`` to the parameters, the latter in ``dt_bias``'s shape), directly and under
    ``torch.utils.checkpoint`` (non-reentrant and reentrant) with every ``checkpoint_handoff`` value and the
    environment override: the raw mode never makes or takes a hand-off record.  The parameters may be frozen; a
    non-contiguous upstream gradient is accepted; under torch's deterministic algorithms (an alert as an error) the
    package raises nothing on any architecture.  The node rebuilds the saved set with its gate mode and FP32 lower
    bound, and the backward refuses a call whose lower bound or gate mode differs from the set's.  Judged: per
    environment value and mode (``<env>.<mode>``) the autograd step and its checkpointed steps (``.ckpt``,
    ``.ckpt_reentrant``); the replay with ``dt_bias`` ``[H, 128]`` and its autograd step (``dt_bias_2d``,
    ``dt_bias_2d.autograd``); frozen parameters (``frozen``); the non-contiguous upstream gradient (``dout_nc``);
    the explicit call and the autograd step under torch's deterministic algorithms (``torch_deterministic``,
    ``torch_deterministic.autograd``); the node's backward with its call captured (``captured``)."""
    from torch.utils.checkpoint import checkpoint

    lengths, heads = shape

    def no_handoff(*args, **kwargs):
        raise AssertionError(
            "the raw gate mode made or took a checkpoint hand-off record"
        )

    monkeypatch.setattr(cake_backend, "_handoff_request", no_handoff)
    monkeypatch.setattr(cake_backend, "_handoff_take", no_handoff)
    monkeypatch.setattr(cake_backend, "_handoff_publish", no_handoff)
    original = cake_backend._bwd_call

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device, gate="raw")
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        arms = {}
        for env in (None, "on", "off"):
            if env is None:
                monkeypatch.delenv("CAKE_KDA_CKPT_HANDOFF", raising=False)
            else:
                monkeypatch.setenv("CAKE_KDA_CKPT_HANDOFF", env)
            for mode in ("auto", "on", "off"):
                tag = f"{env or 'unset'}.{mode}"
                leaves = _rg_leaves(inputs)
                o = _rg_autograd(leaves, cu, checkpoint_handoff=mode)
                o.backward(dout)
                arms[tag] = {"out": o.detach(), **_rg_leaf_grads(leaves)}
                for reentrant in (False, True):
                    leaves = _rg_leaves(inputs)
                    o = checkpoint(
                        lambda *a, _m=mode: _rg_autograd(
                            list(a), cu, checkpoint_handoff=_m
                        ),
                        *leaves,
                        use_reentrant=reentrant,
                    )
                    o.backward(dout)
                    arm = f"{tag}.ckpt_reentrant" if reentrant else f"{tag}.ckpt"
                    arms[arm] = {"out": o.detach(), **_rg_leaf_grads(leaves)}
        monkeypatch.delenv("CAKE_KDA_CKPT_HANDOFF", raising=False)
        # dt_bias as [H, 128]: the same values, its gradient in that shape (explicit and autograd)
        inputs_2d = dict(inputs, dt_bias=inputs["dt_bias"].view(heads, D).clone())
        got = _rg_bwd(inputs_2d, None)
        assert got[6].shape == (heads, D)
        arms["dt_bias_2d"] = _rg_named(got)
        leaves = _rg_leaves(inputs_2d)
        _rg_autograd(leaves, cu).backward(dout)
        assert leaves[6].grad.shape == (heads, D)
        arms["dt_bias_2d.autograd"] = _rg_leaf_grads(leaves)
        # frozen parameters: the other gradients
        leaves = _rg_leaves(inputs)
        leaves[5].requires_grad_(False)
        leaves[6].requires_grad_(False)
        _rg_autograd(leaves, cu).backward(dout)
        assert leaves[5].grad is None and leaves[6].grad is None
        arms["frozen"] = _leaf_grads(leaves[:5], RAW_LEAF_GRADS[:5])
        # a non-contiguous upstream gradient
        leaves = _rg_leaves(inputs)
        o = _rg_autograd(leaves, cu)
        dout_nc = dout.transpose(0, 1).contiguous().transpose(0, 1)
        assert not dout_nc.is_contiguous()
        o.backward(dout_nc)
        arms["dout_nc"] = _rg_leaf_grads(leaves)
        # torch's deterministic algorithms: no alert on any architecture
        saved_mode = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        torch.use_deterministic_algorithms(True)
        try:
            out, grads, _ = _rg_run(inputs)
            leaves = _rg_leaves(inputs)
            _rg_autograd(leaves, cu).backward(dout)
        finally:
            torch.use_deterministic_algorithms(saved_mode[0], warn_only=saved_mode[1])
        arms["torch_deterministic"] = {"out": out, **grads}
        arms["torch_deterministic.autograd"] = _rg_leaf_grads(leaves)
        # the node's backward: the rebuilt saved set carries the gate mode and the FP32 lower bound
        seen: list = []

        def capture(*args, **kwargs):
            saved_ = kwargs["saved"]
            seen.append(
                (
                    saved_.gate,
                    saved_.lower_bound,
                    kwargs.get("use_gate_in_kernel"),
                    kwargs.get("lower_bound"),
                    kwargs.get("handoff"),
                )
            )
            return original(*args, **kwargs)

        monkeypatch.setattr(cake_backend, "_bwd_call", capture)
        try:
            leaves = _rg_leaves(inputs)
            _rg_autograd(leaves, cu).backward(dout)
        finally:
            monkeypatch.setattr(cake_backend, "_bwd_call", original)
        assert seen == [("raw", LOWER_BOUND, True, LOWER_BOUND, None)]
        arms["captured"] = _rg_leaf_grads(leaves)
        if j == 0:
            # the round trip refuses a backward whose lower bound or gate mode differs from the forward's set
            def other_bound(*args, **kwargs):
                return original(*args, **dict(kwargs, lower_bound=-4.0))

            monkeypatch.setattr(cake_backend, "_bwd_call", other_bound)
            o = _rg_autograd(_rg_leaves(inputs), cu)
            with pytest.raises(ValueError, match="lower_bound"):
                o.backward(dout)
            g32 = torch.zeros(
                sum(lengths), heads, D, dtype=torch.float32, device=device
            )

            def g_mode(*args, **kwargs):
                args = list(args)
                args[3] = g32
                kwargs = dict(
                    kwargs,
                    use_gate_in_kernel=False,
                    A_log=None,
                    dt_bias=None,
                    lower_bound=None,
                )
                return original(*args, **kwargs)

            monkeypatch.setattr(cake_backend, "_bwd_call", g_mode)
            o = _rg_autograd(_rg_leaves(inputs), cu)
            with pytest.raises(ValueError, match="written by a forward in gate mode"):
                o.backward(dout)
            monkeypatch.setattr(cake_backend, "_bwd_call", original)
        return arms, inputs

    _judge(record_property, ("raw_gate_autograd", shape), draw)


def test_raw_gate_strided_views_are_judged(device, manifest, record_property) -> None:
    """The raw gate as a column slice of a fused projection (strides ``(F, 128, 1)``) and as a head-interleaved
    split (strides ``(H w, w, 1)``), the columns outside the view filled with NaN: the view holds the raw gate's
    bytes, its gradient is contiguous, a set saved from the contiguous gate serves the strided call and the other
    way round, and through autograd the raw gate's gradient lands in the slice of the projection's gradient with
    zeros elsewhere.  Judged per row: the contiguous gate's call (``<row>``) and per view (``<row>.<view>``) the
    forward and backward, the replay (``.replay``), the contiguous set's backward on the strided call
    (``.strided_call``), the strided set's backward on the contiguous call (``.contiguous_call``) and, for the
    projection slices, the autograd step (``.autograd``).  The batch axis of 1 is optional: the batched call's
    results keep it (``s64_64-h1.batched``; the parameters carry no batch axis)."""
    for lengths, heads in (((63, 1, 65), 2), ((129,), 4)):
        S = sum(lengths)
        row = _shape_id((lengths, heads))
        width = heads * D

        def draw(j, s, _lengths=lengths, _heads=heads, _row=row, _S=S, _width=width):
            inputs = _draw(_lengths, _heads, s, device, gate="raw")
            raw = inputs["g"]
            cu, dout = inputs["cu_seqlens"], inputs["dout"]
            out, grads, saved = _rg_run(inputs)
            arms = {_row: {"out": out, **grads}}
            views = []
            for c0, extra in ((8, 0), (24, 40)):
                proj = torch.full(
                    (_S, c0 + _width + extra),
                    float("nan"),
                    dtype=torch.bfloat16,
                    device=device,
                )
                proj[:, c0 : c0 + _width] = raw.reshape(_S, _width)
                view = proj[:, c0 : c0 + _width].view(_S, _heads, D)
                assert view.stride() == (c0 + _width + extra, D, 1)
                views.append((f"projection{c0}_{extra}", proj, c0, view))
            split = torch.full(
                (_S, _heads, 136), float("nan"), dtype=torch.bfloat16, device=device
            )
            split[..., :D] = raw
            views.append(("interleaved136", None, 0, split[..., :D]))
            for name, proj, c0, view in views:
                tag = f"{_row}.{name}"
                # the view holds the raw gate's bytes (an operand, not a result)
                assert not view.is_contiguous() and _bits_equal(view.contiguous(), raw)
                out_v, grads_v, saved_v = _rg_run(inputs, g=view)
                assert grads_v["dg"].is_contiguous()
                arms[tag] = {"out": out_v, **grads_v}
                arms[f"{tag}.replay"] = _rg_named(_rg_bwd(inputs, None, g=view))
                arms[f"{tag}.strided_call"] = _rg_named(_rg_bwd(inputs, saved, g=view))
                arms[f"{tag}.contiguous_call"] = _rg_named(_rg_bwd(inputs, saved_v))
                if proj is None:
                    continue
                leaf = proj.detach().clone().requires_grad_(True)
                leaves = _rg_leaves(inputs)
                leaves[3] = leaf[:, c0 : c0 + _width].view(_S, _heads, D)
                o = _rg_autograd(leaves, cu)
                o.backward(dout)
                outside = torch.cat(
                    [leaf.grad[:, :c0], leaf.grad[:, c0 + _width :]], dim=1
                )
                assert not bool(outside.any()), tag
                result = {
                    "out": o.detach(),
                    "dg": leaf.grad[:, c0 : c0 + _width]
                    .reshape(_S, _heads, D)
                    .contiguous(),
                }
                result.update(
                    (n, x.grad)
                    for i, (n, x) in enumerate(zip(RAW_LEAF_GRADS, leaves, strict=True))
                    if i != 3
                )
                arms[f"{tag}.autograd"] = result
            return arms, inputs

        _judge(record_property, ("raw_strided_views", lengths, heads), draw)
    # the batch axis of 1: the outputs ranked like the inputs
    lengths, heads = (64, 64), 1
    row = _shape_id((lengths, heads))

    def draw(j, s):
        flat = _draw(lengths, heads, s, device, gate="raw")
        batched = _draw(lengths, heads, s, device, gate="raw", batched=True)
        out, grads, _ = _rg_run(flat)
        b_out, b_grads, _ = _rg_run(batched)
        assert b_out.shape == (1, sum(lengths), heads, D)
        for name in ("dq", "dk", "dv", "dbeta", "dg"):
            assert b_grads[name].dim() == grads[name].dim() + 1, name
        for name in ("dA_log", "ddt_bias"):  # the parameters carry no batch axis
            assert b_grads[name].shape == grads[name].shape, name
        arms = {row: {"out": out, **grads}, f"{row}.batched": {"out": b_out, **b_grads}}
        return arms, flat

    _judge(record_property, ("raw_batch_axis", lengths, heads), draw)


def test_raw_gate_launch_sequences_are_the_deterministic_raw_mode(
    device, manifest, monkeypatch, record_property
) -> None:
    """The raw mode's launch lists on every path are the deterministic package's raw-mode lists (the fused gradient's
    records by their ``RAW_DQKG`` names; ``glm_intra_raw`` and ``param_finalize`` on every architecture) with the
    same grids and scalars (a mode contract), and the lists this suite spells from the plan: the raw records on the
    gate stages, no gate epilogue, no S1 record and the parameter finalize once, last, on every backward path.  No
    call of the raw mode allocates an FP32 token-row tensor: neither the activated gate nor its gradient exists.
    Judged per row (``<row>``, :func:`_shape_id`): the forward and the backward from its light set on a caller
    workspace (arm ``<row>``), the set without its tables (``<row>.bare``), the replay (``<row>.replay``), the tape
    set with the plan's WY record (``<row>.tape``) and with the tape WY twin's rule (``<row>.tape_wy_rule``, the twin
    where it admits it)."""
    allocs: list = []
    original_alloc = cake_backend._alloc

    def spy(shape, dtype, device_):
        allocs.append((tuple(int(x) for x in shape), dtype))
        return original_alloc(shape, dtype, device_)

    monkeypatch.setattr(cake_backend, "_alloc", spy)

    def check_row(lengths, heads):
        S, N = sum(lengths), len(lengths)
        row = _shape_id((lengths, heads))
        nd_log: list = []
        det_log: list = []
        workspaces = {}
        for save_mode in ("light", "tape"):
            workspaces[save_mode] = (
                CakeKDANondetWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    save_mode=save_mode,
                    gate="raw",
                    launcher=_rg_recorder(nd_log),
                ),
                # the deterministic package's workspace: its launch lists (a mode contract)
                det_backend.CakeKDADetWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    save_mode=save_mode,
                    gate="raw",
                    launcher=_rg_recorder(det_log, det_jit),
                ),
            )
        ws, det_ws = workspaces["light"]
        tape_ws, det_tape_ws = workspaces["tape"]
        p = ws.plan
        assert p.gate == "raw" and p.gate_fused and ws.gate == "raw"

        def taken():
            nd = [(RAW_DQKG.get(n, n), *rest) for n, *rest in nd_log]
            names = [n for n, *_ in nd_log]
            det = list(det_log)
            nd_log.clear()
            det_log.clear()
            assert nd == det, (lengths, heads)
            assert not set(names) & S1_RECORDS
            return names

        def no_fp32_token_rows(tag):
            for shape, dtype in allocs:
                assert not (
                    dtype == torch.float32
                    and len(shape) >= 3
                    and shape[-2:] == (heads, D)
                    and shape[-3] in (S, p.R)
                ), (tag, shape)
            allocs.clear()

        def check_backward(seq, path, **kw):
            assert seq == _expected_launches(p, path, **kw), (lengths, heads, path)
            _one_finalize_last(seq, p, path, tag=(lengths, heads, path))
            assert "gate_epilogue" not in seq
            assert p.record("glm_intra", path) == "glm_intra_raw" in seq
            assert p.record("dqkg", path) in seq and p.record("dqkg", path) in RAW_DQKG

        def draw(j, s):
            first = j == 0
            inputs = _draw(lengths, heads, s, device, gate="raw")
            nd_log.clear()
            det_log.clear()
            allocs.clear()
            out, saved = _rg_fwd(inputs, workspace=ws)
            if first:
                _, det_saved = _rg_det_fwd(inputs, workspace=det_ws)  # mode contract
                seq = taken()
                assert seq == _expected_launches(p, "fwd")
                assert len(seq) == p.launch_count("fwd")
                no_fp32_token_rows("fwd")
                _rg_fwd(inputs, workspace=ws, save_for_backward=False)
                _rg_det_fwd(  # mode contract
                    inputs, workspace=det_ws, save_for_backward=False
                )
                assert taken() == _expected_launches(p, "fwd_nosave")
            arms = {
                row: {"out": out, **_rg_named(_rg_bwd(inputs, saved, workspace=ws))}
            }
            if first:
                _rg_det_bwd(inputs, det_saved, workspace=det_ws)  # mode contract
                seq = taken()
                check_backward(seq, "light")
                assert len(seq) == p.launch_count("light")
                no_fp32_token_rows("light")
            bare = dataclasses.replace(saved, tables=None)
            arms[f"{row}.bare"] = _rg_named(_rg_bwd(inputs, bare, workspace=ws))
            if first:
                det_bare = dataclasses.replace(det_saved, tables=None)
                _rg_det_bwd(inputs, det_bare, workspace=det_ws)  # mode contract
                seq = taken()
                check_backward(seq, "light", saved_tables=False)
                assert len(seq) == p.launch_count("light", tables=False)
            arms[f"{row}.replay"] = _rg_named(_rg_bwd(inputs, None, workspace=ws))
            if first:
                _rg_det_bwd(inputs, None, workspace=det_ws)  # mode contract
                seq = taken()
                check_backward(seq, "replay")
                assert len(seq) == p.launch_count("replay")
                no_fp32_token_rows("replay")
            monkeypatch.setattr(cake_backend, "_TAPE_WY", False)  # the plan's WY record
            _, saved_t = _rg_fwd(inputs, workspace=tape_ws)
            if first:
                _, det_saved_t = _rg_det_fwd(  # mode contract
                    inputs, workspace=det_tape_ws
                )
                assert taken() == _expected_launches(p, "fwd")
            arms[f"{row}.tape"] = _rg_named(_rg_bwd(inputs, saved_t, workspace=tape_ws))
            if first:
                _rg_det_bwd(inputs, det_saved_t, workspace=det_tape_ws)  # mode contract
                seq = taken()
                check_backward(seq, "tape")
                assert len(seq) == p.launch_count("tape")
                no_fp32_token_rows("tape")
            monkeypatch.setattr(cake_backend, "_TAPE_WY", None)  # the twin's rule
            nd_log.clear()
            got = _rg_bwd(inputs, saved_t, workspace=tape_ws)
            arms[f"{row}.tape_wy_rule"] = _rg_named(got)
            if first:
                twin = cake_backend._tape_wy_record(tape_ws.plan, device.index)
                check_backward([n for n, *_ in nd_log], "tape", tape_wy=twin)
            nd_log.clear()
            det_log.clear()
            return arms, inputs

        _judge(record_property, ("raw_launches", lengths, heads), draw)
        records = ws.records()
        for path in ("fwd", "fwd_nosave", "light", "tape", "replay"):
            expected = set(_expected_launches(p, path, saved_tables=False))
            assert expected == set(records[path].values()), path

    for lengths, heads in (
        ((63, 1, 65), 2),
        ((129,), 16),
        ((63, 1, 65), 64),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((63, 1, 65), 4),
    ):
        check_row(lengths, heads)


RAW_BATCHED_ROWS = [
    ((63, 1, 65), 4),
    ((100, 640, 77, 1000, 129), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
    ((63, 1, 65), 64),
    ((129, 300), 16),
]


@pytest.mark.parametrize("shape", RAW_BATCHED_ROWS, ids=_shape_id)
def test_raw_gate_lean_and_batched_arenas_are_judged(
    device, manifest, shape, record_property
) -> None:
    """Lean workspaces of one to four tail passes (each range's tail writes its slice of the per-chunk partials, the
    finalize runs once after the last range) and document groups (two, three, one document each; default and lean
    layouts: the finalize once after the last group over the whole pack).  Judged: per tail-pass count
    (``lean<t>``) the forward with the backward from its own set, the backward from this package's default set on
    the workspace, on a per-call arena and on both (``lean<t>.ws``, ``.arena``, ``.ws_arena``) and the autograd
    node carrying the option (``lean<t>.autograd``); per grouping and layout (``<g>.default`` / ``<g>.lean``) the
    forward with the backward from its grouped set (``.batched_set``), the backward from the default set
    (``.default_set``), the default backward of the grouped set (``.default_backward``), a per-call arena
    (``.per_call``), the replay group by group (``.replay``, not lean) and the autograd node (``.autograd``).  The
    launch lists are the plan's, every group's in turn and the finalize once, last.  A lean workspace refuses the
    replay; the backward refuses the deterministic package's default set before any launch, and the deterministic
    package's backward refuses a grouped set of this package (``ValueError`` naming the mode; mode contracts)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    launched: list = []

    def draw(j, s):
        first = j == 0
        inputs = _draw(lengths, heads, s, device, gate="raw")
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        _, saved = _rg_fwd(inputs)
        det_saved = _rg_det_fwd(inputs)[1] if first else None  # mode contract
        arms = {}
        for t in (1, 2, 3, 4):
            tag = f"lean{t}"
            arena = cake_backend.CakeKDANondetArena(lean=True, tail_passes=t)
            ws = CakeKDANondetWorkspace.for_problem(
                S,
                N,
                heads,
                device,
                arena=arena,
                gate="raw",
                launcher=_rg_recorder(launched),
            )
            p = ws.plan
            assert ws.bytes == cake_backend.workspace_bytes(
                S, N, heads, sm_count=p.sm_count, arena=arena, gate="raw"
            )
            o, saved_l = _rg_fwd(inputs, workspace=ws)
            launched.clear()
            got = _rg_bwd(inputs, saved_l, workspace=ws)
            if first:
                names = [n for n, *_ in launched]
                assert names == _expected_launches(p, "light", tail_passes=t), (
                    shape,
                    t,
                )
                _one_finalize_last(names, p, "light", tag=(shape, t))
                _check_tail_grids([(n, g) for n, g, *_ in launched], p, "light", t)
            arms[tag] = {"out": o, **_rg_named(got)}
            for arm, ws_arg, arena_arg in (
                ("ws", ws, None),
                ("arena", None, arena),
                ("ws_arena", ws, arena),
            ):
                got = _rg_bwd(inputs, saved, workspace=ws_arg, arena=arena_arg)
                arms[f"{tag}.{arm}"] = _rg_named(got)
                if first:
                    launched.clear()
                    with pytest.raises(ValueError, match=ND_REFUSES):
                        _rg_bwd(inputs, det_saved, workspace=ws_arg, arena=arena_arg)
                    assert launched == [], (t, arm, "det set")
            if first:
                with pytest.raises(ValueError, match="lean=False"):
                    _rg_bwd(inputs, None, workspace=ws)
            leaves = _rg_leaves(inputs)
            o = _rg_autograd(leaves, cu, arena=arena)
            o.backward(dout)
            arms[f"{tag}.autograd"] = {"out": o.detach(), **_rg_leaf_grads(leaves)}
            del ws
        for label, kw in _partitions(lengths):
            for lean in (False, True):
                arena = cake_backend.CakeKDANondetArena(
                    lean=lean, tail_passes=3 if lean else 1, **kw
                )
                tag = f"{label}.{'lean' if lean else 'default'}"
                ws = CakeKDANondetWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    arena=arena,
                    doc_lengths=lengths,
                    gate="raw",
                    launcher=_rg_recorder(launched),
                )
                p = ws.plan
                assert p.gate == "raw" and len(p.batches(arena, lengths)) >= 2, tag
                launched.clear()
                o, saved_b = _rg_fwd(inputs, workspace=ws, doc_lengths=lengths)
                if first:
                    assert [n for n, *_ in launched] == _expected_batched_launches(
                        p, arena, lengths, "fwd"
                    ), tag
                    launched.clear()
                    with pytest.raises(ValueError, match=ND_REFUSES):
                        _rg_bwd(inputs, det_saved, workspace=ws, doc_lengths=lengths)
                    assert launched == [], (tag, "det default set")
                for name, source in (("default_set", saved), ("batched_set", saved_b)):
                    launched.clear()
                    got = _rg_bwd(inputs, source, workspace=ws, doc_lengths=lengths)
                    arms[f"{tag}.{name}"] = _rg_named(got)
                    if first:
                        path = p.light_path
                        passes = _group_passes(p, arena, lengths, path, nbytes=ws.bytes)
                        names = [n for n, *_ in launched]
                        assert names == _expected_batched_launches(
                            p, arena, lengths, path, passes=passes
                        ), (tag, name)
                        _one_finalize_last(names, p, path, tag=(tag, name))
                        _check_group_launches(
                            launched, p, arena, lengths, path, passes=passes
                        )
                arms[f"{tag}.batched_set"]["out"] = o
                arms[f"{tag}.default_backward"] = _rg_named(_rg_bwd(inputs, saved_b))
                if first:
                    with pytest.raises(ValueError, match=DET_REFUSES):
                        _rg_det_bwd(inputs, saved_b)  # mode contract
                got = _rg_bwd(inputs, saved, arena=arena, doc_lengths=lengths)
                arms[f"{tag}.per_call"] = _rg_named(got)
                if not lean:
                    launched.clear()
                    got = _rg_bwd(inputs, None, workspace=ws, doc_lengths=lengths)
                    arms[f"{tag}.replay"] = _rg_named(got)
                    if first:
                        names = [n for n, *_ in launched]
                        _one_finalize_last(names, p, "replay", tag=(tag, "replay"))
                leaves = _rg_leaves(inputs)
                o = _rg_autograd(leaves, cu, arena=arena, doc_lengths=lengths)
                o.backward(dout)
                arms[f"{tag}.autograd"] = {"out": o.detach(), **_rg_leaf_grads(leaves)}
                del ws
        return arms, inputs

    _judge(record_property, ("raw_lean_batched", shape), draw)


def test_raw_gate_checkpointed_forward_retains_no_workspace(
    device, manifest, record_property
) -> None:
    """Under activation checkpointing the raw mode's forward keeps nothing beyond its output (the implicit
    workspace, a per-call one, a caller's).  Judged per way: the warm-up forward (``<way>.warm``) and the output
    and the gradients through the checkpoint (``implicit``, ``per_call``, ``on_shared``)."""
    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    plan = cake_backend.plan(
        S,
        N,
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
        gate="raw",
    )
    shared = CakeKDANondetWorkspace(plan, device, backward=True)
    assert shared.gate == "raw"

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device, gate="raw")
        cu, dout = inputs["cu_seqlens"], inputs["dout"]
        leaves = _rg_leaves(inputs)

        def implicit(*a):
            return _rg_autograd(list(a), cu)

        def per_call(*a):
            ws = CakeKDANondetWorkspace(plan, device, backward=True)
            return _rg_autograd(list(a), cu, workspace=ws)

        def on_shared(*a):
            return _rg_autograd(list(a), cu, workspace=shared)

        arms = {}
        for fn in (implicit, per_call, on_shared):
            warm = fn(*leaves)
            arms[f"{fn.__name__}.warm"] = {"out": warm.detach()}
            del warm
            if j == 0:
                assert _retained_after_checkpointed_forward(fn, *leaves) == 0, (
                    fn.__name__
                )
            for leaf in leaves:
                leaf.grad = None
            o = checkpoint(fn, *leaves, use_reentrant=False)
            o.backward(dout)
            arms[fn.__name__] = {"out": o.detach(), **_rg_leaf_grads(leaves)}
        return arms, inputs

    _judge(record_property, ("raw_checkpointed_forward", lengths, heads), draw)


def test_raw_gate_workspace_content_never_reaches_a_result(
    device, manifest, monkeypatch, record_property
) -> None:
    """Write-before-read in raw mode: an arena filled with NaN bytes and per-call backward buffers (the outputs and
    the per-chunk partials, only the real chunks of which are written and read) filled with NaN bytes, on the
    light, replay, tape and lean paths.  Judged per case (``<row>-<save mode>``, ``-t<passes>`` on a lean arena):
    the forward on the poisoned workspace and the backward from its set under the poisoned buffers (arm ``<case>``)
    and, without a lean arena, the replay (``<case>.replay``)."""
    cases = (
        ((129,), 2, "light", None),
        ((100, 640, 77, 1000, 129), 2, "light", None),
        ((63, 1, 65, 30), 2, "light", 3),
        ((20, 20, 20, 20, 20, 20, 20), 8, "light", 2),
        ((100, 0, 300), 2, "tape", None),
    )
    original_alloc = cake_backend._alloc

    def nan_alloc(shape, dtype, device_):
        t = original_alloc(shape, dtype, device_)
        if t.is_floating_point():
            t.view(torch.uint8).fill_(0xFF)
        return t

    def check_row(lengths, heads, save_mode, passes):
        S, N = sum(lengths), len(lengths)
        case = f"{_shape_id((lengths, heads))}-{save_mode}" + (
            "" if passes is None else f"-t{passes}"
        )
        arena = (
            None
            if passes is None
            else cake_backend.CakeKDANondetArena(lean=True, tail_passes=passes)
        )
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
            gate="raw",
        )
        poisoned = CakeKDANondetWorkspace(
            plan,
            device,
            backward=True,
            save_mode=save_mode,
            arena=arena,
        )

        def draw(j, s):
            inputs = _draw(lengths, heads, s, device, gate="raw")
            poisoned.arena.fill_(0xFF)
            torch.cuda.synchronize()
            o, saved_p = _rg_fwd(inputs, workspace=poisoned)
            poisoned.arena.fill_(0xFF)
            monkeypatch.setattr(cake_backend, "_alloc", nan_alloc)
            try:
                got = _rg_bwd(inputs, saved_p, workspace=poisoned)
            finally:
                monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
            arms = {case: {"out": o, **_rg_named(got)}}
            if arena is None:
                poisoned.arena.fill_(0xFF)
                monkeypatch.setattr(cake_backend, "_alloc", nan_alloc)
                try:
                    got = _rg_bwd(inputs, None, workspace=poisoned)
                finally:
                    monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
                arms[f"{case}.replay"] = _rg_named(got)
            return arms, inputs

        key = ("raw_workspace_content", lengths, heads, save_mode, passes)
        _judge(record_property, key, draw)

    for lengths, heads, save_mode, passes in cases:
        check_row(lengths, heads, save_mode, passes)


# The raw backward repeated on a device: heads 1..64 cover the serial and segmented routes on every architecture (the
# raw rows of 16, 32 and 64 heads among them), on a pack of four documents (3206 tokens).  Each repeat's seven results
# are held for the judge, so a draw runs a few.
RAW_REPEAT_HEADS = (1, 2, 4, 16, 32, 64)
RAW_REPEAT_LENGTHS = (2000, 77, 1000, 129)
RAW_REPEATS = 4


@pytest.mark.parametrize("heads", RAW_REPEAT_HEADS, ids=lambda h: f"h{h}")
def test_raw_gate_repeats_are_judged(device, manifest, heads, record_property) -> None:
    """``RAW_REPEATS`` raw backwards from one saved set on one caller workspace and as many without one (a
    per-call workspace each): every repeat's seven results, ``dA_log`` / ``ddt_bias`` included, judged, the worst
    kept (``workspace``, ``per_call``; the package promises no bits from run to run), with the forward that saved
    the set (arm ``""``).  The caller workspace holds the arena alone: its bytes are ``workspace_bytes`` of the
    problem, the deterministic package's (a byte query of the other package: a mode contract)."""
    lengths = RAW_REPEAT_LENGTHS
    S, N = sum(lengths), len(lengths)
    ws = CakeKDANondetWorkspace.for_problem(S, N, heads, device, gate="raw")
    sm = ws.plan.sm_count
    # mode contract: the byte query of the deterministic package
    assert (
        ws.bytes
        == cake_backend.workspace_bytes(S, N, heads, sm_count=sm, gate="raw")
        == det_backend.workspace_bytes(S, N, heads, sm_count=sm, gate="raw")
    )

    def draw(j, s):
        inputs = _draw(lengths, heads, s, device, gate="raw")
        out, saved = _rg_fwd(inputs, workspace=ws)
        arms = {"": {"out": out}}
        for name, kw in (("workspace", {"workspace": ws}), ("per_call", {})):
            arms[name] = [
                _rg_named(_rg_bwd(inputs, saved, **kw)) for _ in range(RAW_REPEATS)
            ]
        return arms, inputs

    _judge(record_property, ("raw_gate_repeats", lengths, heads), draw)


@pytest.mark.parametrize("l2n", ("off", "fused"))
def test_raw_gate_raises_no_alert_under_torch_deterministic_mode(
    device, manifest, l2n, record_property
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)``, strict and with ``warn_only=True``, a raw backward raises
    no PyTorch non-determinism alert on any architecture: no ``RuntimeError`` and no warning naming a missing
    deterministic implementation, from the saved set and from the replay, with and without a caller workspace, with
    the in-kernel Q/K L2 norm off and fused (``use_qk_l2norm_in_kernel=True``: ``q`` / ``k`` are the raw rows the
    kernels normalize).  The caller workspace's backward finishes the parameter gradients with the one
    deterministic finalize (``param_finalize``; ``param_finalize_l2n`` in the l2n mode).  The g mode raises no
    alert either.  (The saving forwards run with torch's deterministic algorithms off; the backward and the
    replay's recompute run under them.)  Judged per head count (``h<heads>``): the forward (arm ``h<heads>``) and
    every backward, ``h<heads>.<strict|warn_only>.<workspace|per_call>.<saved|replay>``."""
    alert = "deterministic implementation"
    lengths = (63, 1, 65)
    S, N = sum(lengths), len(lengths)
    fused = l2n != "off"
    L2 = {"use_qk_l2norm_in_kernel": True} if fused else {}
    finalize = RAW_FINALIZE if l2n == "off" else "param_finalize_l2n"
    saved_mode = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    def check_row(heads):
        launched: list = []
        ws = CakeKDANondetWorkspace.for_problem(
            S,
            N,
            heads,
            device,
            gate="raw",
            l2n=l2n,
            launcher=_rg_recorder(launched),
        )

        def draw(j, s):
            inputs = draw_case(lengths, heads, "raw", fused, s, device)
            arms = {}
            try:
                torch.use_deterministic_algorithms(False)
                launched.clear()
                out, saved = _rg_fwd(inputs, workspace=ws, **L2)
                assert launched
                arms[f"h{heads}"] = {"out": out}
                if j == 0:
                    g_inputs = make_inputs(lengths, heads, device)
                    gq = [
                        g_inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
                    ]
                    _, g_saved = cake_kda_nondet_train_forward(*gq)
                torch.cuda.synchronize(device)
                for warn_only in (False, True):
                    torch.use_deterministic_algorithms(True, warn_only=warn_only)
                    for ws_name, kw in (
                        ("workspace", {"workspace": ws}),
                        ("per_call", {}),
                    ):
                        for s_name, s_in in (("saved", saved), ("replay", None)):
                            mode = "warn_only" if warn_only else "strict"
                            arm = f"h{heads}.{mode}.{ws_name}.{s_name}"
                            launched.clear()
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                got = _rg_bwd(inputs, s_in, **kw, **L2)
                                torch.cuda.synchronize(device)
                            assert not [w for w in caught if alert in str(w.message)], (
                                heads,
                                arm,
                                [str(w.message) for w in caught],
                            )
                            arms[arm] = _rg_named(got)
                            if kw:
                                # the caller's workspace ran the backward (its results are the ones judged)
                                names = [n for n, *_ in launched]
                                finals = [
                                    n
                                    for n in names
                                    if cake_jit.MODULES[n]["stage"] == "param_finalize"
                                ]
                                assert finals == [finalize], (heads, arm, names)
                    if j == 0:
                        # the g mode: no alert either
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            cake_kda_nondet_train_backward(
                                *gq, g_inputs["dout"], saved=g_saved
                            )
                            torch.cuda.synchronize(device)
                        assert not [w for w in caught if alert in str(w.message)], (
                            warn_only
                        )
            finally:
                torch.use_deterministic_algorithms(
                    saved_mode[0], warn_only=saved_mode[1]
                )
            return arms, inputs

        judge_arms(
            record_property,
            draw,
            seed=_seed("raw_torch_deterministic", l2n, heads),
            scale=SCALE,
            l2n=fused,
        )

    for heads in (4, 64):
        check_row(heads)
