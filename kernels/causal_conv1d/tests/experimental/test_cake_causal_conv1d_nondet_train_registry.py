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

The single-rank registry (``cake_jit``) of the non-deterministic causal convolution training package against the
deterministic package's: the forward records are the deterministic forward programs (only their names and the
per-call programmatic dependent launch scalar differ), every backward route keeps the deterministic route's programs
and selection (but where a channel-band program takes the lowest channel counts of its twin, the program of the route
it shares a body with, at the rows of ``FALLBACK_ROWS``, where the route's fallback runs, and at the rows of
``SM_WIDE_ROWS``, where a program of the ``rows`` selection kind runs: a build of a deterministic SM-wide program, one
CTA per SM), every program but the opt-in pair routes' takes the per-call PDL scalar (its kernels wait for the preceding
grid before touching global memory; the host's policy tables, and the backward rows of ``PDL_OFF_ROWS`` that launch
without it, which the backward entry finds by passing its route, ``C`` and ``S``), each backward program is ONE
ordinary launch (no reduce program) that owns an ``accumulators`` workspace (one FP32 word per sum and accumulator
copy on every route, :data:`ACC_SLOTS` copies; ``[C, 4]`` taps then ``[C]`` bias words; uint32 ticket counters per
channel tile), and the workspace byte
layout and size depend on ``C`` and the route only.  The direct backward programs (the backward keys of
``DIRECT_ROWS``, which have no deterministic twin): their records, keys and tile forms, their selection ahead of every
other backward table, their plans and capped grids at and around their keys at every ``bwd_grid_mult``, and every
backward plan, the public workspace and the KCP routes as without them outside those keys.  The small-S grid cap
(``GRID_CAP_ROWS``): its table and rules, the capped grid of a single-wave backward inside its spans, the same
programs, plans, workspace and KCP routes as without it, and no cap at a backward ``DIRECT_ROWS`` range.  The public
names (each carries the mode word ``nondet``; no entry point takes a mode keyword), and the package's wiring
(test-helper key, package data, no ``__init__.py`` at the ``cake_pplx`` / ``cake_pplx._backends`` namespace levels).
No device is needed; nothing is built or launched.  The ``build.json`` checks skip until the exporter renders the
package.
"""

from __future__ import annotations

import filecmp
import hashlib
import importlib.util
import inspect
import itertools
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
cake_jit = cake_package(KEY, "cake_jit")
kcp_jit = cake_package(KEY, "kcp_jit")
det_build = cake_package("cake_causal_conv1d_det_train", "cake_build")
det_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).parent
REPO = Path(__file__).resolve().parents[2]
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
NAME_FIELDS = ("kernel", "kernel_symbol", "library")
# A placeholder name until the exporter renders the package, then the identity rule's prefix + 20 hex digits.
PROGRAM_NAME = re.compile(
    r"cake_causal_conv1d_train_(?:TODOEXPORT_[a-z0-9_]+|[0-9a-f]{20})"
)
CHANNELS = (
    8,
    128,
    136,
    256,
    264,
    384,
    504,
    512,
    520,
    576,
    608,
    616,
    640,
    680,
    688,
    704,
    760,
    768,
    776,
    1016,
    1024,
    1032,
    1160,
    1280,
    1344,
    1536,
    1600,
    1920,
    2048,
    2560,
    4096,
    8192,
)
# 262143 / 262144: both sides of the sm_107a token-floored backward program's floor; S_max and S_max + 1 of the
# default backward route's ``FALLBACK_ROWS`` keys (:data:`FALLBACK_ROWS_OF_RECORD`; :func:`_backward_edges` adds those of
# every backward key); both sides of every bound of the ``SM_WIDE_ROWS`` segments (:data:`SM_WIDE_ROWS_OF_RECORD`).
TOKENS = (
    1,
    1000,
    20479,
    20480,
    24575,
    24576,
    31291,
    31292,
    31293,
    32767,
    32768,
    62585,
    62586,
    65536,
    98303,
    98304,
    125170,
    125171,
    131071,
    131072,
    131073,
    147456,
    250340,
    250341,
    262143,
    262144,
    262145,
)
# The rows where a route runs its fallback program in place of the program its selection rules pick: arch -> (route,
# C_lo, C_hi) -> (S_min, S_max), every multiple of 8 from C_lo to C_hi.  On sm_100a and sm_103a the ``bwd`` route's
# C 520 to 1016 run the route's fallback (the stage_pair body) up to the last measured token count at which no
# backward, training-step or checkpointed-step row was slower than the program the thresholds pick; at C 520 to 608
# up to the token count before the route's first ``SM_WIDE_ROWS`` segment (:data:`SM_WIDE_ROWS_OF_RECORD`).  The
# ``bwd_bias`` and ``bwd_dt`` routes run their fallback at small token counts by the same rule (FP32 and BF16 taps
# alike: each route's programs evaluate SiLU' at the unrounded FP32 preactivation with either taps dtype).  The
# forward routes' keys (C 512 to 1024) run the route's static program, the deterministic package's at the same rows,
# over the measured token counts where it was not slower than the dynamic program; a direct range of
# :data:`DIRECT_ROWS_OF_RECORD` starts where such a static range ends, and a pair direct range of
# :data:`PAIR_ROWS_OF_RECORD` runs ahead of both where it covers them.
FALLBACK_ROWS_OF_RECORD = {
    "sm_100a": {
        ("bwd", 520, 608): (1, 24575),
        ("bwd", 616, 760): (1, 131072),
        ("bwd", 768, 768): (1, 262144),
        ("bwd", 776, 1016): (1, 62585),
        ("bwd_bias", 520, 568): (1, 16384),
        ("bwd_bias", 576, 640): (1, 24576),
        ("bwd_bias", 648, 1016): (1, 16384),
        ("bwd_dt", 520, 568): (1, 16384),
        ("bwd_dt", 576, 640): (1, 24576),
        ("bwd_dt", 648, 1016): (1, 16384),
        ("fwd", 512, 512): (1, 24576),
        ("fwd", 520, 544): (1, 12288),
        ("fwd", 552, 568): (1, 8192),
        ("fwd", 576, 576): (1, 4096),
        ("fwd", 584, 640): (1, 8192),
        ("fwd", 648, 704): (1, 12288),
        ("fwd", 712, 712): (1, 12288),
        ("fwd", 720, 768): (1, 8192),
        ("fwd", 776, 888): (1, 8192),
        ("fwd", 896, 896): (1, 12288),
        ("fwd", 904, 952): (1, 8192),
        ("fwd", 960, 1024): (1, 24576),
        ("fwd_bias", 512, 512): (1, 24576),
        ("fwd_bias", 520, 544): (1, 12288),
        ("fwd_bias", 552, 568): (1, 8192),
        ("fwd_bias", 576, 576): (1, 4096),
        ("fwd_bias", 584, 640): (1, 4096),
        ("fwd_bias", 648, 696): (1, 8192),
        ("fwd_bias", 704, 704): (1, 17408),
        ("fwd_bias", 712, 768): (1, 16384),
        ("fwd_bias", 776, 776): (1, 8192),
        ("fwd_bias", 784, 896): (1, 8192),
        ("fwd_bias", 904, 1024): (1, 12288),
        ("fwd_dt", 512, 512): (1, 24576),
        ("fwd_dt", 520, 568): (1, 8192),
        ("fwd_dt", 576, 576): (1, 4096),
        ("fwd_dt", 584, 640): (1, 8192),
        ("fwd_dt", 648, 696): (1, 8192),
        ("fwd_dt", 704, 704): (1, 12288),
        ("fwd_dt", 712, 768): (1, 16384),
        ("fwd_dt", 776, 896): (1, 12288),
        ("fwd_dt", 904, 1024): (1, 12288),
    },
    "sm_103a": {
        ("bwd", 520, 608): (1, 20479),
        ("bwd", 616, 760): (1, 31292),
        ("bwd", 768, 768): (1, 250340),
        ("bwd", 776, 1016): (1, 125170),
        ("bwd_bias", 520, 568): (1, 16384),
        ("bwd_bias", 576, 640): (1, 24576),
        ("bwd_bias", 648, 1016): (1, 16384),
        ("bwd_dt", 520, 568): (1, 16384),
        ("bwd_dt", 576, 640): (1, 24576),
        ("bwd_dt", 648, 1016): (1, 16384),
        ("fwd", 512, 512): (1, 2048),
        ("fwd", 520, 520): (1, 8192),
        ("fwd", 528, 568): (1, 4096),
        ("fwd", 576, 632): (1, 2048),
        ("fwd", 640, 688): (1, 4096),
        ("fwd", 696, 696): (1, 4096),
        ("fwd", 704, 704): (1, 8192),
        ("fwd", 712, 712): (1, 8192),
        ("fwd", 720, 768): (1, 8192),
        ("fwd", 776, 824): (1, 4096),
        ("fwd", 832, 832): (1, 8192),
        ("fwd", 840, 896): (1, 4096),
        ("fwd", 904, 952): (1, 2048),
        ("fwd", 960, 1024): (1, 24576),
        ("fwd_bias", 512, 512): (1, 24576),
        ("fwd_bias", 520, 520): (1, 12288),
        ("fwd_bias", 528, 544): (1, 24576),
        ("fwd_bias", 552, 568): (1, 8192),
        ("fwd_bias", 576, 576): (1, 2048),
        ("fwd_bias", 584, 640): (1, 4104),
        ("fwd_bias", 648, 704): (1, 16384),
        ("fwd_bias", 712, 712): (1, 12800),
        ("fwd_bias", 720, 728): (13824, 16383),
        ("fwd_bias", 736, 736): (14336, 16383),
        ("fwd_bias", 744, 760): (13824, 16383),
        ("fwd_bias", 768, 768): (13824, 16384),
        ("fwd_bias", 776, 896): (1, 8192),
        ("fwd_bias", 904, 952): (1, 12288),
        ("fwd_bias", 960, 1024): (1, 20480),
        ("fwd_dt", 512, 512): (1, 20480),
        ("fwd_dt", 520, 568): (1, 8192),
        ("fwd_dt", 576, 640): (1, 8192),
        ("fwd_dt", 648, 704): (1, 8192),
        ("fwd_dt", 712, 752): (1, 12800),
        ("fwd_dt", 760, 760): (1, 12288),
        ("fwd_dt", 768, 768): (1, 12288),
        ("fwd_dt", 776, 776): (1, 12288),
        ("fwd_dt", 784, 832): (1, 12288),
        ("fwd_dt", 840, 888): (1, 8192),
        ("fwd_dt", 896, 896): (1, 8192),
        ("fwd_dt", 904, 952): (1, 8192),
        ("fwd_dt", 960, 1016): (1, 20480),
        ("fwd_dt", 1024, 1024): (1, 24576),
    },
    "sm_107a": {
        ("fwd", 512, 512): (1, 8192),
        ("fwd", 520, 520): (1, 24575),
        ("fwd", 528, 528): (1, 16384),
        ("fwd", 536, 536): (1, 16384),
        ("fwd", 544, 568): (1, 16384),
        ("fwd", 576, 704): (1, 4096),
        ("fwd", 712, 760): (1, 4096),
        ("fwd", 768, 768): (1, 2048),
        ("fwd", 776, 824): (1, 32768),
        ("fwd", 832, 832): (1, 4096),
        ("fwd", 840, 888): (1, 8192),
        ("fwd", 896, 896): (1, 16384),
        ("fwd", 904, 1016): (1, 12288),
        ("fwd", 1024, 1024): (1, 32768),
        ("fwd_bias", 512, 512): (1, 32768),
        ("fwd_bias", 520, 528): (1, 24575),
        ("fwd_bias", 536, 568): (1, 32768),
        ("fwd_bias", 576, 576): (1, 12296),
        ("fwd_bias", 584, 584): (1, 24576),
        ("fwd_bias", 592, 592): (1, 16384),
        ("fwd_bias", 600, 600): (1, 24576),
        ("fwd_bias", 608, 608): (1, 16384),
        ("fwd_bias", 616, 616): (1, 24576),
        ("fwd_bias", 624, 624): (1, 16384),
        ("fwd_bias", 632, 632): (1, 24576),
        ("fwd_bias", 640, 696): (1, 24576),
        ("fwd_bias", 704, 768): (1, 32768),
        ("fwd_bias", 776, 1024): (1, 32768),
        ("fwd_dt", 512, 512): (1, 32768),
        ("fwd_dt", 520, 528): (1, 16384),
        ("fwd_dt", 536, 544): (1, 16384),
        ("fwd_dt", 552, 568): (1, 16384),
        ("fwd_dt", 576, 640): (1, 12288),
        ("fwd_dt", 648, 704): (1, 12288),
        ("fwd_dt", 712, 712): (1, 32768),
        ("fwd_dt", 720, 768): (1, 12288),
        ("fwd_dt", 776, 824): (1, 24576),
        ("fwd_dt", 832, 832): (1, 20480),
        ("fwd_dt", 840, 888): (1, 20480),
        ("fwd_dt", 896, 896): (1, 20480),
        ("fwd_dt", 904, 1024): (1, 32768),
    },
}
# The programs whose rows the fallback takes inside :data:`FALLBACK_ROWS_OF_RECORD` (named by program, not by
# route position): the ``bwd`` program the channel thresholds pick at C 520 to 1016 on each architecture, and the
# ``bwd_bias`` and ``bwd_dt`` programs the thresholds pick from C 512.
FALLBACK_REPLACES_OF_RECORD = {
    "sm_100a": frozenset(
        {
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_92aeee41bc2e2d394937",
            "cake_causal_conv1d_train_092af77575f0fdbf3615",
        }
    ),
    "sm_103a": frozenset(
        {
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
            "cake_causal_conv1d_train_92aeee41bc2e2d394937",
            "cake_causal_conv1d_train_092af77575f0fdbf3615",
        }
    ),
}
# The rows where a forward route runs its direct program (a build of the deterministic direct program; each CTA takes
# the next token blocks of its channel tile, no ``work``): arch -> (route, C_lo, C_hi) -> (S_lo, S_hi, program), every
# multiple of 8 from C_lo to C_hi.  Each range starts above the key's static range of :data:`FALLBACK_ROWS_OF_RECORD`
# (or at the token count from which the direct program beat the dynamic one) and ends at the last measured token
# count where it was not slower than the dynamic program, eager and CUDA Graph replay; the deterministic package
# keeps the same rows.  A :data:`PAIR_ROWS_OF_RECORD` range runs ahead of the part of a range it covers.
DIRECT_ROWS_OF_RECORD = {
    "sm_100a": {
        ("fwd", 520, 544): (
            12289,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 552, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 576, 576): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 584, 640): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 648, 704): (
            12289,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 712, 712): (
            12289,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 720, 768): (
            8193,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 776, 888): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 896, 896): (
            12289,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 904, 952): (
            8193,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd_bias", 520, 544): (
            12289,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 552, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 576, 576): (
            4097,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 584, 640): (
            4097,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 648, 696): (
            8193,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 704, 704): (
            17409,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 776, 776): (
            8193,
            20480,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 784, 896): (
            8193,
            14336,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_dt", 520, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 576, 576): (
            4097,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 584, 640): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 648, 696): (
            8193,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 704, 704): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 712, 768): (
            16385,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 776, 896): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 904, 1024): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
    },
    "sm_103a": {
        ("fwd", 512, 512): (
            2049,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 520, 520): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 528, 568): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 576, 632): (
            2049,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 640, 688): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 696, 696): (
            4097,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 704, 704): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 712, 712): (
            8193,
            20479,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 720, 768): (
            8193,
            19456,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 776, 824): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 832, 832): (
            8193,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 840, 896): (
            4097,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 904, 952): (
            2049,
            20480,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd_bias", 520, 520): (
            12289,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 552, 568): (
            8193,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 576, 576): (
            2049,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 584, 640): (
            4105,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 712, 712): (
            12801,
            14592,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 776, 896): (
            8193,
            16384,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_dt", 512, 512): (
            20481,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 520, 568): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 576, 640): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 648, 704): (
            8193,
            18432,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 712, 752): (
            12801,
            19456,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 760, 760): (
            12289,
            19456,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 768, 768): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 776, 776): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 784, 832): (
            12289,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 840, 888): (
            8193,
            20480,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 896, 896): (
            8193,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 904, 952): (
            8193,
            16384,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
    },
    "sm_107a": {
        ("fwd", 512, 512): (
            8193,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 528, 528): (
            16385,
            24575,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 536, 536): (
            16385,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 544, 568): (
            16385,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 576, 704): (
            4097,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 712, 760): (
            4097,
            16384,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 768, 768): (
            2049,
            24576,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 832, 832): (
            4097,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 840, 888): (
            8193,
            16384,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 896, 896): (
            16385,
            32768,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd", 904, 1016): (
            12289,
            16384,
            "cake_causal_conv1d_train_a81632301ba92a887c46",
        ),
        ("fwd_bias", 576, 576): (
            12297,
            32768,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 592, 592): (
            16385,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 608, 608): (
            16385,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_bias", 624, 624): (
            16385,
            24576,
            "cake_causal_conv1d_train_8ed855b04178bc5d90f9",
        ),
        ("fwd_dt", 520, 528): (
            16385,
            24575,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 536, 544): (
            16385,
            32768,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 552, 568): (
            16385,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 576, 640): (
            12289,
            32768,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 648, 704): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 720, 768): (
            12289,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 832, 832): (
            20481,
            24576,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
        ("fwd_dt", 896, 896): (
            20481,
            32768,
            "cake_causal_conv1d_train_21b9be3d1093abe38406",
        ),
    },
}
# The channel tiles of each route's direct program on each architecture :data:`DIRECT_ROWS_OF_RECORD` names: arch ->
# route -> (ct_vec, n_sub), ``ct_vec`` channel vectors per channel tile and ``n_sub`` token sub-tiles of 8 tokens.
DIRECT_PLANS_OF_RECORD = {
    "sm_100a": {"fwd": (8, 16), "fwd_dt": (8, 16), "fwd_bias": (8, 16)},
    "sm_103a": {"fwd": (8, 16), "fwd_dt": (16, 8), "fwd_bias": (8, 16)},
    "sm_107a": {"fwd": (8, 16), "fwd_dt": (16, 8), "fwd_bias": (8, 16)},
}
# The CTAs per SM of each forward route's direct program (its co-resident grid, capped at its token blocks): the bias
# route's program, built with ``min_blocks=3``, holds three.
DIRECT_CTAS_PER_SM = {"fwd": 4, "fwd_dt": 4, "fwd_bias": 3}
# The rows where a forward route runs its pair direct program (a build of the deterministic pair direct program: the
# direct program with each thread on a pair of channels, geometry ``vec`` 2; no ``work``): arch -> (route, C_lo, C_hi)
# -> (S_lo, S_hi, program), the keys of one architecture and forward route disjoint in C and together C 512 to 1024
# (one key, two on sm_100a ``fwd_dt``).  A pair range runs ahead of the static and the direct ranges it covers; the
# deterministic package keeps the same rows.
PAIR_ROWS_OF_RECORD = {
    "sm_100a": {
        ("fwd", 512, 1024): (
            64,
            3072,
            "cake_causal_conv1d_train_d0b237e92b28348315d0",
        ),
        ("fwd_dt", 512, 640): (
            64,
            1536,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_dt", 648, 1024): (
            64,
            1024,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_bias", 512, 1024): (
            128,
            1536,
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ),
    },
    "sm_103a": {
        ("fwd", 512, 1024): (
            64,
            3072,
            "cake_causal_conv1d_train_d0b237e92b28348315d0",
        ),
        ("fwd_dt", 512, 1024): (
            32,
            1792,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_bias", 512, 1024): (
            16,
            1536,
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ),
    },
    "sm_107a": {
        ("fwd", 512, 1024): (
            64,
            2560,
            "cake_causal_conv1d_train_d0b237e92b28348315d0",
        ),
        ("fwd_dt", 512, 1024): (
            16,
            1792,
            "cake_causal_conv1d_train_96bf2b8aa1132598f130",
        ),
        ("fwd_bias", 512, 1024): (
            16,
            1792,
            "cake_causal_conv1d_train_6cc04dd65e493836b918",
        ),
    },
}
# The channel tiles of each route's pair direct program: arch -> route -> (ct_vec, n_sub), ``ct_vec`` channel pairs per
# channel tile, so a pair row of C channels runs ceil(C / 2 / ct_vec) tiles.
PAIR_PLANS_OF_RECORD = {
    "sm_100a": {"fwd": (16, 8), "fwd_dt": (16, 8), "fwd_bias": (16, 8)},
    "sm_103a": {"fwd": (16, 8), "fwd_dt": (16, 8), "fwd_bias": (16, 8)},
    "sm_107a": {"fwd": (16, 8), "fwd_dt": (16, 8), "fwd_bias": (16, 8)},
}
# The CTAs per SM of each forward route's pair direct program (its co-resident grid, capped at its token blocks).
PAIR_CTAS_PER_SM = {"fwd": 7, "fwd_dt": 7, "fwd_bias": 6}
# The geometry ``vec`` (channels per thread) of the direct and the pair direct programs.
FORM_VEC = {"static": 8, "direct": 8, "pair": 2}
# The direct backward program of each backward route that has one (one record for every architecture): one ordinary
# launch without TMA or a staged pipeline, 128 threads, 2 CTAs per SM, each CTA the next token blocks of ``n_sub x 4``
# tokens of its channel tile.  It has no deterministic twin; it runs only at its route's keys of
# :data:`BACKWARD_DIRECT_ROWS_OF_RECORD`.
BACKWARD_DIRECT_PROGRAMS = {
    "bwd": "cake_causal_conv1d_train_a9deef7249fb422bc28a",
    "bwd_dt": "cake_causal_conv1d_train_6a1e78e37349abd841ff",
    "bwd_bias": "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
}
# The backward keys of ``DIRECT_ROWS`` (the registry holds them after the forward keys of
# :data:`DIRECT_ROWS_OF_RECORD`): arch -> (route, C_lo, C_hi) -> (S_lo, S_hi, program), every multiple of 8 from C_lo
# to C_hi at S_lo to S_hi.  Measured at C 512, 520, 576, 600, 640, 704, 768, 832, 896, 960, 1016 and 1024 and S 1,
# 128, 1000, 2048 and 4096 (every storage of the route; the backward, the training step and the checkpointed step,
# eager and under CUDA Graph), a key holds the measured points where no cell of the direct program ran slower than the
# programs the other tables and the thresholds pick, beyond the spread of a copy of those programs: per channel count
# the longest run of such token counts, one key over consecutive measured channel counts with the same run, no key
# across a change of the run and none for a single token count.
# Each architecture's keys were then timed again at those channel counts and at C 544, 616, 672, 736, 800, 864, 928
# and 992 (B200 and GB200 for sm_100a, GB300 for sm_103a, VR200 for sm_107a): a point slower there in two runs on one
# GPU leaves the table by the same rules over all 20 channel counts, inside the first keys.
BACKWARD_DIRECT_ROWS_OF_RECORD = {
    "sm_100a": {
        ("bwd", 512, 832): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 896, 1016): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 1024, 1024): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd_dt", 512, 616): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 640, 736): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 768, 768): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 800, 800): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 832, 832): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 864, 864): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 896, 896): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 960, 1016): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_bias", 512, 512): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 520, 576): (
            1000,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
    },
    "sm_103a": {
        ("bwd", 512, 896): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 960, 1016): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 1024, 1024): (1, 4096, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd_dt", 512, 640): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 672, 672): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 704, 768): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 800, 832): (
            128,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 864, 896): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 960, 1016): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_bias", 512, 512): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 520, 576): (
            1000,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 768, 768): (
            1000,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 960, 960): (
            128,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
    },
    "sm_107a": {
        ("bwd", 512, 512): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 520, 576): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 600, 616): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 640, 640): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 704, 704): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 768, 768): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 832, 832): (
            1000,
            2048,
            "cake_causal_conv1d_train_a9deef7249fb422bc28a",
        ),
        ("bwd", 992, 992): (1, 1000, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd", 1024, 1024): (1, 2048, "cake_causal_conv1d_train_a9deef7249fb422bc28a"),
        ("bwd_dt", 512, 512): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 520, 520): (
            1000,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 576, 576): (
            1000,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 600, 640): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 704, 704): (
            1000,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 768, 832): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 896, 896): (
            1,
            2048,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 928, 960): (
            1,
            1000,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1016, 1016): (
            1,
            1000,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_dt", 1024, 1024): (
            1,
            4096,
            "cake_causal_conv1d_train_6a1e78e37349abd841ff",
        ),
        ("bwd_bias", 512, 512): (
            1,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 520, 832): (
            1000,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
        ("bwd_bias", 1024, 1024): (
            1,
            2048,
            "cake_causal_conv1d_train_2a2c48dec0406daa9e81",
        ),
    },
}
# The tile forms of the direct backward programs, arch -> route -> ((ct_vec, n_sub), ...): 128-channel tiles, then
# 64-channel tiles; a call takes the first form whose tiles pad C to the fewest channel vectors.
BACKWARD_DIRECT_FORMS = {
    arch: {route: ((16, 8), (8, 16)) for route in BACKWARD_DIRECT_PROGRAMS}
    for arch in ARCHES
}
# The direct split ``(ct_vec, n_sub, n_ch_tiles)`` of a backward direct row by C (every route and architecture): C 576
# runs nine 64-channel tiles (five 128-channel tiles would leave 64 channels idle), a C that 128 divides 128-channel
# tiles, and a tie (C 600, 640, 1016) the 128-channel tiles; at most 15 tiles (C 960).
BACKWARD_DIRECT_SPLITS = {
    512: (16, 8, 4),
    520: (8, 16, 9),
    576: (8, 16, 9),
    600: (16, 8, 5),
    640: (16, 8, 5),
    704: (8, 16, 11),
    768: (16, 8, 6),
    832: (8, 16, 13),
    896: (16, 8, 7),
    960: (8, 16, 15),
    1016: (16, 8, 8),
    1024: (16, 8, 8),
}
# The program and split of record of a backward call just outside a backward direct key (the plan the package runs
# there with or without the direct rows): (arch, route, C, S) -> (program, (ct_vec, n_sub, n_ch_tiles)).
BACKWARD_OUTSIDE_PLANS_OF_RECORD = {
    ("sm_100a", "bwd", 1024, 4097): (
        "cake_causal_conv1d_train_461d358d7090642c65de",
        (128, 1, 1),
    ),
    ("sm_100a", "bwd", 504, 1): (
        "cake_causal_conv1d_train_e8e950fb254d3376db52",
        (63, 2, 1),
    ),
    ("sm_100a", "bwd", 1016, 4097): (
        "cake_causal_conv1d_train_ea326ebe0f84d8cf2adb",
        (32, 4, 4),
    ),
    ("sm_103a", "bwd_dt", 768, 4097): (
        "cake_causal_conv1d_train_7035e23df8c9bf3c11e8",
        (32, 4, 3),
    ),
    ("sm_107a", "bwd_dt", 1024, 4097): (
        "cake_causal_conv1d_train_092af77575f0fdbf3615",
        (128, 1, 1),
    ),
    ("sm_107a", "bwd_bias", 1024, 2049): (
        "cake_causal_conv1d_train_4c8b21939791ace6e830",
        (128, 1, 1),
    ),
}


def _direct_rows_of_record() -> dict:
    """The literal ``DIRECT_ROWS``: per architecture the forward keys of :data:`DIRECT_ROWS_OF_RECORD`, then the
    backward keys of :data:`BACKWARD_DIRECT_ROWS_OF_RECORD`."""

    return {
        arch: {
            **DIRECT_ROWS_OF_RECORD.get(arch, {}),
            **BACKWARD_DIRECT_ROWS_OF_RECORD[arch],
        }
        for arch in ARCHES
    }


def _backward_direct_split(C: int) -> tuple[int, int, int]:
    """The direct split of a backward direct row of ``C`` channels by the rule: of :data:`BACKWARD_DIRECT_FORMS` the
    first form whose tiles pad ``C / 8`` channel vectors to the fewest, ``ceil(C / 8 / ct_vec)`` tiles."""

    cv = C // 8
    best = None
    for ct_vec, n_sub in BACKWARD_DIRECT_FORMS["sm_100a"]["bwd"]:
        padded = -(-cv // ct_vec) * ct_vec
        if best is None or padded < best[0]:
            best = (padded, ct_vec, n_sub)
    _, ct_vec, n_sub = best
    return ct_vec, n_sub, -(-cv // ct_vec)


def _backward_direct_cell(arch: str, route: str, C: int, S: int) -> str | None:
    """The direct backward program a call of ``route`` with ``(C, S)`` on ``arch`` runs by
    :data:`BACKWARD_DIRECT_ROWS_OF_RECORD` (``C`` a multiple of 8 inside a key's channels, ``S`` inside its tokens),
    ``None`` elsewhere."""

    if C % 8:
        return None
    for (name, c_lo, c_hi), (s_lo, s_hi, program) in BACKWARD_DIRECT_ROWS_OF_RECORD.get(
        arch, {}
    ).items():
        if name == route and c_lo <= C <= c_hi and s_lo <= S <= s_hi:
            return program
    return None


@contextmanager
def _without_backward_direct_rows():
    """Inside the block the registry is the package's without its direct backward programs: ``DIRECT_ROWS`` keeps its
    forward keys only and no route lists a program of :data:`BACKWARD_DIRECT_PROGRAMS`.  The host's row plans and its
    route lookup cache are dropped on entry and on exit."""

    directs = set(BACKWARD_DIRECT_PROGRAMS.values())
    rows = {
        arch: {
            key: value
            for key, value in table.items()
            if cake_jit.route_role(key[0]) == "fwd"
        }
        for arch, table in cake_jit.DIRECT_ROWS.items()
    }
    routes = {
        arch: {
            kind: [name for name in names if name not in directs]
            for kind, names in by_kind.items()
        }
        for arch, by_kind in cake_jit.ROUTES.items()
    }
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    cake_jit._library_route.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_jit, "DIRECT_ROWS", rows)
            m.setattr(cake_jit, "ROUTES", routes)
            yield
    finally:
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
        cake_jit._library_route.cache_clear()


# Measured sm_103a forward cells at C 704 to 768 that lie inside a ``DIRECT_ROWS`` range: (route, taps, bias, C, S).
# At each cell a call selects the route's direct program and plans its direct tiles.
SM103_DIRECT_PLAN_CELLS = (
    ("fwd", "f32", "none", 704, 12288),
    ("fwd", "f32", "none", 704, 14336),
    ("fwd", "f32", "none", 704, 16384),
    ("fwd", "f32", "none", 704, 18432),
    ("fwd", "f32", "none", 704, 20479),
    ("fwd", "f32", "none", 712, 12288),
    ("fwd", "f32", "none", 712, 12800),
    ("fwd", "f32", "none", 712, 12960),
    ("fwd", "f32", "none", 712, 13312),
    ("fwd", "f32", "none", 712, 13824),
    ("fwd", "f32", "none", 712, 14336),
    ("fwd", "f32", "none", 712, 14592),
    ("fwd", "f32", "none", 712, 14848),
    ("fwd", "f32", "none", 712, 15360),
    ("fwd", "f32", "none", 712, 16384),
    ("fwd", "f32", "none", 712, 17408),
    ("fwd", "f32", "none", 712, 18432),
    ("fwd", "f32", "none", 712, 19456),
    ("fwd", "f32", "none", 712, 20479),
    ("fwd", "f32", "none", 736, 12288),
    ("fwd", "f32", "none", 736, 14336),
    ("fwd", "f32", "none", 736, 16384),
    ("fwd", "f32", "none", 736, 18432),
    ("fwd", "f32", "none", 760, 12288),
    ("fwd", "f32", "none", 760, 14336),
    ("fwd", "f32", "none", 760, 16384),
    ("fwd", "f32", "none", 760, 18432),
    ("fwd_dt", "bf16", "none", 704, 12288),
    ("fwd_dt", "bf16", "none", 704, 14336),
    ("fwd_dt", "bf16", "none", 704, 16384),
    ("fwd_dt", "bf16", "none", 704, 18432),
    ("fwd_dt", "bf16", "none", 712, 12960),
    ("fwd_dt", "bf16", "none", 712, 13312),
    ("fwd_dt", "bf16", "none", 712, 13824),
    ("fwd_dt", "bf16", "none", 712, 14336),
    ("fwd_dt", "bf16", "none", 712, 14592),
    ("fwd_dt", "bf16", "none", 712, 14848),
    ("fwd_dt", "bf16", "none", 712, 15360),
    ("fwd_dt", "bf16", "none", 712, 16384),
    ("fwd_dt", "bf16", "none", 712, 17408),
    ("fwd_dt", "bf16", "none", 712, 18432),
    ("fwd_dt", "bf16", "none", 712, 19456),
    ("fwd_dt", "bf16", "none", 736, 14336),
    ("fwd_dt", "bf16", "none", 736, 16384),
    ("fwd_dt", "bf16", "none", 736, 18432),
    ("fwd_dt", "bf16", "none", 760, 14336),
    ("fwd_dt", "bf16", "none", 760, 16384),
    ("fwd_dt", "bf16", "none", 760, 18432),
    ("fwd_dt", "bf16", "none", 768, 12800),
    ("fwd_dt", "bf16", "none", 768, 12960),
    ("fwd_dt", "bf16", "none", 768, 13312),
    ("fwd_dt", "bf16", "none", 768, 13824),
    ("fwd_dt", "bf16", "none", 768, 14336),
    ("fwd_dt", "bf16", "none", 768, 14592),
    ("fwd_dt", "bf16", "none", 768, 14848),
    ("fwd_dt", "bf16", "none", 768, 15360),
    ("fwd_dt", "bf16", "none", 768, 16384),
    ("fwd_dt", "bf16", "none", 768, 17408),
    ("fwd_dt", "bf16", "none", 768, 18432),
    ("fwd_dt", "bf16", "none", 768, 19456),
    ("fwd_dt", "bf16", "none", 768, 20479),
    ("fwd_dt", "bf16", "none", 768, 20480),
)
# The rows where route ``bwd`` runs an SM-wide program (one CTA per SM for the whole launch): arch -> (route, C_lo,
# C_hi) -> ((S_lo, S_hi, program), ...), every multiple of 8 from C_lo to C_hi, the segments contiguous up to
# ``INT32_MAX``.  On each architecture the first segment's program runs from the measured token count from which it
# beat the program the rules or ``FALLBACK_ROWS`` pick there, and the second from the token count at which the other
# variant ran faster.
SM_WIDE_ROWS_OF_RECORD = {
    "sm_100a": {
        ("bwd", 520, 608): (
            (24576, 98303, "cake_causal_conv1d_train_82e874f30406be2334b6"),
            (98304, 2147483647, "cake_causal_conv1d_train_1ec222b2d908a7d6ea21"),
        ),
    },
    "sm_103a": {
        ("bwd", 520, 608): (
            (20480, 31291, "cake_causal_conv1d_train_77692909592ff14be82f"),
            (31292, 2147483647, "cake_causal_conv1d_train_ccd45ec8667831b67344"),
        ),
    },
    "sm_107a": {
        ("bwd", 520, 680): (
            (32768, 131071, "cake_causal_conv1d_train_175b82ccd2309f434dfb"),
            (131072, 2147483647, "cake_causal_conv1d_train_3293c0e34e70413fb4af"),
        ),
    },
}
# The SM-wide programs (the records of selection-rule kind ``rows``) and the deterministic SM-wide program each is a
# build of (the same body and build keywords, one CTA per SM; the reduction of this package): program -> det twin.
SM_WIDE_COPIES = {
    "cake_causal_conv1d_train_82e874f30406be2334b6": "cake_causal_conv1d_det_train_e057357c655024d4ff1d",
    "cake_causal_conv1d_train_1ec222b2d908a7d6ea21": "cake_causal_conv1d_det_train_e057357c655024d4ff1d",
    "cake_causal_conv1d_train_77692909592ff14be82f": "cake_causal_conv1d_det_train_baca4c969c783d07d81f",
    "cake_causal_conv1d_train_ccd45ec8667831b67344": "cake_causal_conv1d_det_train_baca4c969c783d07d81f",
    "cake_causal_conv1d_train_175b82ccd2309f434dfb": "cake_causal_conv1d_det_train_5ca7515dd2118dd96241",
    "cake_causal_conv1d_train_3293c0e34e70413fb4af": "cake_causal_conv1d_det_train_5ca7515dd2118dd96241",
}
# The FP32 accumulator copies of each SM-wide program: two for sm_107a's program from 131072 tokens, one otherwise.
SM_WIDE_SLOTS = {name: 2 if name == "cake_causal_conv1d_train_3293c0e34e70413fb4af" else 1 for name in SM_WIDE_COPIES}
# The programs whose rows the SM-wide programs take inside :data:`SM_WIDE_ROWS_OF_RECORD`: the ``bwd`` program the
# channel thresholds pick at C 520 to 608 (sm_107a: 520 to 680), named by program; on sm_107a this package's own copy
# of the deterministic channel-band program (:data:`OWN_COPIES`).
SM_WIDE_REPLACES_OF_RECORD = {
    "sm_100a": frozenset({"cake_causal_conv1d_train_461d358d7090642c65de"}),
    "sm_103a": frozenset({"cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab"}),
    "sm_107a": frozenset({"cake_causal_conv1d_train_e5d6d187103ab6e0cbbb"}),
}
# The deterministic package's programs this package does not have: the row-group backward program of route ``bwd``
# on sm_107a (264 <= C < 512) and the SM-wide row-group programs of route ``bwd`` on sm_100a, sm_103a and sm_107a
# (channel bands of 264 <= C < 512, from a token floor), and the copy of route ``bwd``'s fallback program on sm_107a
# (the channel band of 520 <= C < 1024).  Every other record pairs with this package's at the same route position.
# This package's SM-wide programs (:data:`SM_WIDE_COPIES`) are builds of the three SM-wide programs under another
# selection (the rows of ``SM_WIDE_ROWS`` only), listed after their route's other programs.
# This package carries its own channel-band copy of its route's fallback for the last of them (:data:`OWN_COPIES`).
DET_ONLY = frozenset(
    {
        "cake_causal_conv1d_det_train_50aa818e2b2338fb34a4",
        "cake_causal_conv1d_det_train_e057357c655024d4ff1d",
        "cake_causal_conv1d_det_train_baca4c969c783d07d81f",
        "cake_causal_conv1d_det_train_5ca7515dd2118dd96241",
        "cake_causal_conv1d_det_train_0d99ef0e0ed10b11dfd5",
    }
)
# The deterministic channel-band programs of :data:`DET_ONLY` whose rows this package runs on its own channel-band copy
# of the same route's fallback (the fallback's build keywords with the same ``c_band``, under this package's name):
# deterministic program -> this package's copy.  The copy is listed after the deterministic route's programs, as a
# channel-band program, but its twin (the fallback) keeps no channel threshold to give it.
OWN_COPIES = {
    "cake_causal_conv1d_det_train_0d99ef0e0ed10b11dfd5": "cake_causal_conv1d_train_e5d6d187103ab6e0cbbb",
}


@contextmanager
def warnings_recorded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _strip(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _args(record: dict) -> set[str]:
    return {name for kind, name in record["arg_plan"] if kind != "grid"}


# The launch fields of a program with the per-call programmatic dependent launch scalar (``launch.use_pdl`` of
# ``"runtime"``): its binding takes ``enable_pdl`` (0 or 1) as its last host parameter, before the grid.
PDL_FIELDS = ("launch", "arg_plan", "logical")
PDL_ARG = ["launch", "enable_pdl"]
# The single-rank programs that take the scalar: every record but the opt-in pair routes' (18 forward: the static, direct
# and pair direct program of each forward route among them; 32 backward: the three direct backward programs among them).
RUNTIME_PDL_PROGRAMS = 50


def _pdl_program_of(det: dict) -> dict:
    """The launch fields a program with the per-call PDL scalar takes from the deterministic record ``det`` it
    mirrors (launched without PDL): ``use_pdl`` ``"runtime"``, ``("launch", "enable_pdl")`` before the three grid
    entries of the arguments, and the logical name ``enable_pdl``."""

    assert det["launch"]["use_pdl"] is False and PDL_ARG not in det["arg_plan"]
    assert [kind for kind, _ in det["arg_plan"][-3:]] == ["grid", "grid", "grid"]
    return {
        "launch": {**det["launch"], "use_pdl": "runtime"},
        "arg_plan": [*det["arg_plan"][:-3], PDL_ARG, *det["arg_plan"][-3:]],
        "logical": {**det["logical"], "enable_pdl": "enable_pdl"},
    }


def _det_route(arch: str, kind: str) -> list[str]:
    """The deterministic route of ``(arch, kind)`` without its :data:`DET_ONLY` programs."""

    return [name for name in det_jit.ROUTES[arch][kind] if name not in DET_ONLY]


def _pairs(role: str):
    """``(arch, kind, new record name, det record name)`` of every route of ``role``, in route order: the
    deterministic route's programs, position for position (the route's channel-band programs follow them, see
    :func:`_bands`)."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            if cake_jit.route_role(kind) != role:
                continue
            new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
            assert len(new) >= len(det), (arch, kind)
            for name, det_name in zip(new[: len(det)], det, strict=True):
                yield arch, kind, name, det_name


def _channels(registry, name: str, arch: str) -> int | None:
    """The lowest channel threshold of a program's rule on ``arch``: a deterministic program the det-only programs
    split into several channel bands runs here from the first of them."""

    rule = registry.select_rule(registry.MODULES[name])
    value = None if rule is None or rule["kind"] == "rows" else rule["channels"].get(arch)
    return value[0] if isinstance(value, list) else value


def _bands():
    """``(arch, kind, band name, twin name, det twin name)`` of every channel-band program: a program a route lists
    after the deterministic route's programs (but this package's copies of deterministic programs,
    :data:`OWN_COPIES`, its SM-wide programs, :data:`SM_WIDE_COPIES`, and its direct backward programs,
    :data:`BACKWARD_DIRECT_PROGRAMS`).  Its twin is the program of the route whose
    channel threshold on ``arch`` is no longer the deterministic one: the band program owns the twin's lowest channel
    counts there, from the deterministic threshold up to the twin's new one."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
            moved = [
                (name, det_name)
                for name, det_name in zip(new[: len(det)], det, strict=True)
                if _channels(cake_jit, name, arch) != _channels(det_jit, det_name, arch)
            ]
            for band in new[len(det) :]:
                if (
                    band in OWN_COPIES.values()
                    or band in SM_WIDE_COPIES
                    or band in BACKWARD_DIRECT_PROGRAMS.values()
                ):
                    continue
                (twin,) = [
                    pair
                    for pair in moved
                    if _channels(det_jit, pair[1], arch)
                    == _channels(cake_jit, band, arch)
                ]
                moved.remove(twin)
                yield arch, kind, band, *twin
            assert not moved, (arch, kind, moved)


@pytest.mark.cpu_only
def test_loader_is_the_det_loader() -> None:
    """``cake_build.py`` is byte for byte the deterministic package's (one manifest-driven loader)."""

    assert filecmp.cmp(
        Path(cake_build.__file__), Path(det_build.__file__), shallow=False
    )


@pytest.mark.cpu_only
def test_routes_mirror_the_deterministic_package() -> None:
    assert cake_jit.ARCHES == det_jit.ARCHES == ARCHES
    assert cake_jit.KINDS == det_jit.KINDS
    assert cake_jit.REQUIRED_KINDS == det_jit.REQUIRED_KINDS
    assert cake_jit.ROLES == ("fwd", "bwd")
    assert cake_jit.GRID_KINDS == ("co_resident", "occupancy")
    for arch in ARCHES:
        assert set(cake_jit.ROUTES[arch]) == set(det_jit.ROUTES[arch]), arch
    # The host's argument -> route map is the deterministic package's, over every combination.
    for bias, fp32, taps, dweight in itertools.product((False, True), repeat=4):
        args = dict(bias=bias, fp32_grad=fp32, bf16_taps=taps, bf16_dweight=dweight)
        assert cake_jit.backward_route(**args) == det_jit.backward_route(**args), args
    for bias, taps in itertools.product((False, True), repeat=2):
        args = dict(bias=bias, bf16_taps=taps)
        assert cake_jit.forward_route(**args) == det_jit.forward_route(**args), args
    routed = {
        name
        for by_kind in cake_jit.ROUTES.values()
        for names in by_kind.values()
        for name in names
    }
    # No reduce program: every record is a routed forward or backward program.
    assert set(cake_jit.MODULES) == routed
    assert not set(cake_jit.MODULES) & set(det_jit.MODULES)
    for name, record in cake_jit.MODULES.items():
        assert PROGRAM_NAME.fullmatch(name), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["role"] in cake_jit.ROLES and "reduce" not in record, name
        assert record["grid_rule"]["kind"] in cake_jit.GRID_KINDS, name
        assert record["grid_rule"]["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name


@pytest.mark.cpu_only
def test_forward_records_are_the_deterministic_forward_records() -> None:
    """The forward programs are the deterministic ones (same module, build keywords, plan): ``y`` keeps its bits.
    Only the three name fields and the launch's programmatic dependent launch differ: the deterministic record
    launches without it, this package's takes the per-call ``enable_pdl`` scalar (:func:`_pdl_program_of`).  The
    static, direct and pair direct programs of every forward route are among them, at the same route positions."""

    paired = 0
    for arch, kind, name, det_name in _pairs("fwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        label = (arch, kind, name)
        assert _strip(record, *NAME_FIELDS, *PDL_FIELDS) == _strip(det, *NAME_FIELDS, *PDL_FIELDS), label
        assert {field: record[field] for field in PDL_FIELDS} == _pdl_program_of(det), label
        paired += cake_jit.MODULES[name]["workspace_rule"] is None
    assert paired == len(ARCHES) * 3 + sum(
        len(routes)
        for plans in (DIRECT_PLANS_OF_RECORD, PAIR_PLANS_OF_RECORD)
        for routes in plans.values()
    )


def _forward_program(arch: str, kind: str, form: str) -> str | None:
    """The static (``form`` ``"static"``: the record without a selection rule), direct (``"direct"``: the record of
    selection-rule kind ``rows`` with eight channels per thread) or pair direct (``"pair"``: the record of
    selection-rule kind ``rows`` with two channels per thread) program of forward route ``kind`` on ``arch``, ``None``
    when the route lists none."""

    names = [
        name
        for name in cake_jit.route_modules(arch, kind)
        if (cake_jit.select_rule(cake_jit.MODULES[name]) or {"kind": None})["kind"]
        == {"static": None, "direct": "rows", "pair": "rows"}[form]
        and cake_jit.MODULES[name]["grid_rule"]["geometry"]["vec"] == FORM_VEC[form]
    ]
    assert len(names) <= 1, (arch, kind, form, names)
    return names[0] if names else None


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_static_and_direct_forward_records(arch) -> None:
    """Every forward route lists its static program first (the route's fallback: no selection rule), its direct
    program next to last and its pair direct program last, each where its table (:data:`DIRECT_ROWS_OF_RECORD`,
    :data:`PAIR_ROWS_OF_RECORD`) names a key of the route on ``arch`` and nowhere else; the dynamic programs between
    them keep their order, the one-stage program now from 8 channels (``channels_at_least``).  Neither the static, the
    direct nor the pair direct program takes ``work``: no ``work`` argument or logical name and ``workspace_rule``
    None.  Each runs a co-resident grid of 128 threads, of 8-vectors (the pair direct program: of channel pairs); the
    direct and the pair direct program's grids are capped at their token blocks (``cap`` ``token_blocks``) with
    :data:`DIRECT_CTAS_PER_SM` and :data:`PAIR_CTAS_PER_SM` CTAs per SM, and each ships for exactly the architectures
    whose table names its route."""

    tables = {"direct": DIRECT_ROWS_OF_RECORD, "pair": PAIR_ROWS_OF_RECORD}
    plans = {"direct": DIRECT_PLANS_OF_RECORD, "pair": PAIR_PLANS_OF_RECORD}
    ctas = {"direct": DIRECT_CTAS_PER_SM, "pair": PAIR_CTAS_PER_SM}
    for kind in cake_jit.forward_kinds(arch):
        names = cake_jit.route_modules(arch, kind)
        forms = {
            form: _forward_program(arch, kind, form)
            for form in ("static", "direct", "pair")
        }
        label = (arch, kind)
        assert forms["static"] is not None and names[0] == forms["static"], label
        tail = []
        for form in ("direct", "pair"):
            keyed = any(route == kind for route, _, _ in tables[form].get(arch, {}))
            assert (
                (forms[form] is not None)
                is keyed
                is (kind in plans[form].get(arch, {}))
            ), (label, form)
            if forms[form] is not None:
                tail.append(forms[form])
        assert list(names[len(names) - len(tail) :]) == tail, label
        dynamic = [name for name in names if name not in forms.values()]
        twins = [
            name.replace("cake_causal_conv1d_train_", "cake_causal_conv1d_det_train_")
            for name in dynamic
        ]
        assert dynamic and twins == _det_route(arch, kind)[1 : 1 + len(dynamic)], label
        assert cake_jit.select_rule(cake_jit.MODULES[dynamic[0]]) == {
            "channels": {a: 8 for a in ARCHES},
            "kind": "channels_at_least",
        }, label
        for form, name in forms.items():
            if name is None:
                continue
            record = cake_jit.MODULES[name]
            grid = record["grid_rule"]
            assert record["role"] == "fwd" and record["workspace_rule"] is None, (
                label,
                name,
            )
            assert "work" not in _args(record) and "work" not in record["logical"], (
                label,
                name,
            )
            assert grid["kind"] == "co_resident" and grid["cooperative"] is False, (
                label,
                name,
            )
            assert grid["geometry"] == {"threads": 128, "vec": FORM_VEC[form]}, (
                label,
                name,
            )
            assert record["launch"]["use_pdl"] == "runtime", (label, name)
            if form == "static":
                assert "cap" not in grid and cake_jit.select_rule(record) is None, (
                    label,
                    name,
                )
                assert record["arches"] == list(ARCHES), (label, name)
                continue
            assert (
                grid["cap"] == "token_blocks"
                and grid["ctas_per_sm"] == ctas[form][kind]
            ), (label, name)
            assert cake_jit.select_rule(record) == {"kind": "rows"}, (label, name)
            assert record["arches"] == [
                a
                for a in ARCHES
                if any(route == kind for route, _, _ in tables[form].get(a, {}))
            ], (label, name)


@pytest.mark.cpu_only
def test_backward_records_are_one_launch_with_accumulators() -> None:
    """Every backward program replaces the deterministic program + reduce pair of the same route position: the
    deterministic arguments without ``partial``, plus ``dw_acc`` / ``work`` and the gradient outs it now stores
    (``dw``, and ``db`` on the bias routes); the same build geometry, block, cluster, architectures and selection
    rule; an ordinary occupancy-sized launch with the per-call programmatic dependent launch scalar ``enable_pdl``
    (last before the grid; the opt-in pair routes' programs launch without PDL); an ``accumulators`` workspace of the
    route's dtype and tap count."""

    vocabulary = set(cake_jit.LOGICAL_TENSORS) | set(cake_jit.LOGICAL_SCALARS)
    twins = {twin: band_arch for band_arch, _, _, twin, _ in _bands()}
    for arch, kind, name, det_name in _pairs("bwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        bias = "bias" in kind
        label = (arch, kind, name)
        assert record["role"] == "bwd" and record["arches"] == det["arches"], label
        assert record["ffi_entry"] == det["ffi_entry"], label
        runtime = kind not in cake_jit.PAIR_ROUTES
        expected = (
            (_args(det) - {"partial"})
            | {"dw_acc", "work", "dw"}
            | ({"db"} if bias else set())
            | ({"enable_pdl"} if runtime else set())
        )
        assert _args(record) == expected, label
        assert [k for k, _ in record["arg_plan"][-3:]] == ["grid", "grid", "grid"], (
            label
        )
        assert (record["arg_plan"][-4] == PDL_ARG) is runtime, label
        assert set(record["logical"]) == _args(record), label
        assert set(record["logical"].values()) <= vocabulary, label
        assert (
            record["logical"].get("x_map", "x") == "x"
            and record["logical"].get("dy_map", "dy") == "dy"
        )
        # The global accumulators: kernel parameter ``dw_acc`` (the det fragments' ``acc`` is a register), logical ``acc``.
        assert (
            record["logical"]["dw_acc"] == "acc" and record["logical"]["work"] == "work"
        ), label
        assert ("flags" in _args(record)) is (kind != "bwd"), label
        grid, det_grid = record["grid_rule"], det["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det_grid["geometry"], label
        assert set(grid["grid_mult"]) == set(record["arches"]), label
        # ``ctas_per_sm`` is the rendered program's occupancy (the exporter's leaf).
        assert isinstance(grid["ctas_per_sm"], int) and grid["ctas_per_sm"] >= 1, label
        launch = record["launch"]
        assert launch["cooperative"] is False, label
        assert launch["use_pdl"] == ("runtime" if runtime else False), label
        assert launch["block"] == det["launch"]["block"], label
        assert launch["cluster"] == det["launch"]["cluster"], label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        # A twin keeps the deterministic rule but on the architecture of its channel-band program.
        det_rule = det_jit.select_rule(det)
        if det_rule is not None:
            channels = {
                a: _channels(det_jit, det_name, a) for a in det_rule["channels"]
            }
            det_rule = {**det_rule, "channels": channels}
        if name in twins:
            det_rule["channels"][twins[name]] = _channels(cake_jit, name, twins[name])
        assert cake_jit.select_rule(record) == det_rule, label
        rule = dict(record["workspace_rule"])
        assert "slots" in rule, label  # required: the package never defaults it
        assert rule == {
            "acc": "float32",
            "slots": ACC_SLOTS[kind],
            "kind": "accumulators",
            "taps": 5 if bias else 4,
            "work": "uint32",
        }, label


@pytest.mark.cpu_only
def test_channel_band_programs_take_their_twins_lowest_channels() -> None:
    """A channel-band program ships for one architecture: its record is its twin's under its own name, architecture
    and selection rule (the rendered occupancy and shared memory aside).  It owns the calls from the deterministic
    program's channel threshold, and the twin those from a higher one, so the route still runs one program per
    ``(S, C)``."""

    bands = list(_bands())
    assert {(arch, kind) for arch, kind, *_ in bands} == {
        ("sm_107a", "bwd"),
        ("sm_107a", "bwd_bias"),
        ("sm_107a", "bwd_fp32g"),
    }
    for arch, kind, band, twin, det_twin in bands:
        record, twin_record = cake_jit.MODULES[band], cake_jit.MODULES[twin]
        label = (arch, kind, band)
        assert band not in det_jit.MODULES and record["arches"] == [arch], label
        assert record["library"] == record["kernel"] == band, label
        assert record["kernel_symbol"] == f"kernel_{band}", label
        threshold = _channels(cake_jit, band, arch)
        assert threshold == _channels(det_jit, det_twin, arch), label
        assert threshold < _channels(cake_jit, twin, arch), label
        assert cake_jit.select_rule(record) == {
            "channels": {arch: threshold},
            "kind": "channels_at_least",
        }, label
        grid, twin_grid = record["grid_rule"], twin_record["grid_rule"]
        assert grid["grid_mult"] == {arch: twin_grid["grid_mult"][arch]}, label
        derived = ("grid_mult", "ctas_per_sm")
        assert _strip(grid, *derived) == _strip(twin_grid, *derived), label
        launch = _strip(record["launch"], "dynamic_smem_bytes")
        assert launch == _strip(twin_record["launch"], "dynamic_smem_bytes"), label
        own = ("arches", "select_rule", "grid_rule", "launch", *NAME_FIELDS)
        assert _strip(record, *own) == _strip(twin_record, *own), label


@pytest.mark.cpu_only
def test_own_copies_take_the_deterministic_band_rows() -> None:
    """Each copy of :data:`OWN_COPIES` is a channel-band program of the deterministic program's route and
    architecture: listed after the deterministic route's programs, its record the route's fallback's under its own
    name and selection rule (the rendered occupancy and shared memory aside), which owns the calls from the
    deterministic program's lowest channel threshold."""

    for det_name, copy in OWN_COPIES.items():
        det_record = det_jit.MODULES[det_name]
        (arch,) = det_record["arches"]
        (kind,) = [k for k, names in det_jit.ROUTES[arch].items() if det_name in names]
        new = cake_jit.ROUTES[arch][kind]
        label = (arch, kind, copy)
        assert copy in new[len(_det_route(arch, kind)) :], label
        (fallback,) = [n for n in new if cake_jit.select_rule(cake_jit.MODULES[n]) is None]
        record, twin_record = cake_jit.MODULES[copy], cake_jit.MODULES[fallback]
        assert copy not in det_jit.MODULES and record["arches"] == [arch], label
        assert record["library"] == record["kernel"] == copy, label
        assert record["kernel_symbol"] == f"kernel_{copy}", label
        assert cake_jit.select_rule(twin_record) is None, label
        assert cake_jit.select_rule(record) == {
            "channels": {arch: _channels(det_jit, det_name, arch)},
            "kind": "channels_at_least",
        }, label
        grid, twin_grid = record["grid_rule"], twin_record["grid_rule"]
        derived = ("grid_mult", "ctas_per_sm")
        assert grid["grid_mult"] == {arch: twin_grid["grid_mult"][arch]}, label
        assert _strip(grid, *derived) == _strip(twin_grid, *derived), label
        launch = _strip(record["launch"], "dynamic_smem_bytes")
        assert launch == _strip(twin_record["launch"], "dynamic_smem_bytes"), label
        own = ("arches", "select_rule", "grid_rule", "launch", *NAME_FIELDS)
        assert _strip(record, *own) == _strip(twin_record, *own), label


# The host's programmatic dependent launch policy per architecture: on for every call, outside and inside a CUDA Graph
# capture.
PDL_EAGER_OF_RECORD = {"sm_100a": True, "sm_103a": True, "sm_107a": True}
PDL_GRAPH_OF_RECORD = {"sm_100a": True, "sm_103a": True, "sm_107a": True}
# The backward rows that launch without programmatic dependent launch in both modes, whatever the two tables say
# (``PDL_OFF_ROWS``): arch -> (route, C_lo, C_hi) -> (S_min, S_max), every multiple of 8 from C_lo to C_hi.  On sm_100a
# and sm_103a the ``bwd`` route at its largest channel counts below 1024, each key from the first measured token count
# from which the backward without the attribute is not slower than with it in the training step and the checkpointed
# step, eager and graph, up to every token count the package takes.
PDL_OFF_ROWS_OF_RECORD = {
    "sm_100a": {
        ("bwd", 1000, 1008): (187755, 2147483647),
        ("bwd", 1016, 1016): (155648, 2147483647),
    },
    "sm_103a": {
        ("bwd", 1000, 1008): (131073, 2147483647),
        ("bwd", 1016, 1016): (125171, 2147483647),
    },
}
# The program the calls of the :data:`PDL_OFF_ROWS_OF_RECORD` rows launch (the one the rows were measured with): the
# ``bwd`` program the channel thresholds pick there on both architectures.
PDL_OFF_PROGRAMS_OF_RECORD = {
    "sm_100a": frozenset({"cake_causal_conv1d_train_461d358d7090642c65de"}),
    "sm_103a": frozenset({"cake_causal_conv1d_train_461d358d7090642c65de"}),
}


def _pair_programs() -> set[str]:
    """The programs of the opt-in pair routes on any architecture."""

    return {
        name
        for by_kind in cake_jit.ROUTES.values()
        for kind, names in by_kind.items()
        if kind in cake_jit.PAIR_ROUTES
        for name in names
    }


@pytest.mark.cpu_only
def test_single_rank_programs_take_the_runtime_pdl_scalar() -> None:
    """Every program but the opt-in pair routes' launches with ``launch.use_pdl`` ``"runtime"``: one
    ``("launch", "enable_pdl")`` argument right before the grid, bound to the logical scalar ``enable_pdl`` of
    ``LOGICAL_SCALARS`` (not a plan scalar); the pair routes' programs launch without PDL and take no such
    argument."""

    assert "enable_pdl" in cake_jit.LOGICAL_SCALARS and "enable_pdl" not in cake_jit.PLAN_SCALARS
    pair = _pair_programs()
    assert pair and pair <= set(cake_jit.MODULES)
    runtime = 0
    for name, record in cake_jit.MODULES.items():
        launch_args = [entry for entry in record["arg_plan"] if entry[0] == "launch"]
        if name in pair:
            assert record["launch"]["use_pdl"] is False, name
            assert not launch_args and "enable_pdl" not in record["logical"], name
            continue
        assert record["launch"]["use_pdl"] == "runtime", name
        assert launch_args == [PDL_ARG] and record["arg_plan"][-4] == PDL_ARG, name
        assert record["logical"]["enable_pdl"] == "enable_pdl", name
        runtime += 1
    assert runtime == RUNTIME_PDL_PROGRAMS == len(cake_jit.MODULES) - len(pair)


@pytest.mark.cpu_only
def test_pdl_policy_tables(monkeypatch) -> None:
    """``PDL_EAGER`` and ``PDL_GRAPH`` are the literal policy (:data:`PDL_EAGER_OF_RECORD`,
    :data:`PDL_GRAPH_OF_RECORD`), one bool per architecture of ``ARCHES``; ``pdl_enabled`` reads ``PDL_GRAPH``
    inside a capture and ``PDL_EAGER`` outside one, as the int 0 or 1, and refuses an architecture outside
    ``ARCHES``."""

    assert cake_jit.PDL_EAGER == PDL_EAGER_OF_RECORD and cake_jit.PDL_GRAPH == PDL_GRAPH_OF_RECORD
    for table in (cake_jit.PDL_EAGER, cake_jit.PDL_GRAPH):
        assert tuple(table) == ARCHES and all(type(value) is bool for value in table.values()), table
    for arch in ARCHES:
        for capturing, table in ((False, PDL_EAGER_OF_RECORD), (True, PDL_GRAPH_OF_RECORD)):
            got = cake_jit.pdl_enabled(arch, capturing=capturing)
            assert type(got) is int and got == int(table[arch]), (arch, capturing)
    for arch in ("sm_90a", None):
        with pytest.raises(ValueError):
            cake_jit.pdl_enabled(arch, capturing=False)
    for value in (False, True):
        monkeypatch.setitem(cake_jit.PDL_EAGER, "sm_103a", value)
        monkeypatch.setitem(cake_jit.PDL_GRAPH, "sm_103a", not value)
        assert cake_jit.pdl_enabled("sm_103a", capturing=False) == int(value)
        assert cake_jit.pdl_enabled("sm_103a", capturing=True) == int(not value)


@pytest.mark.cpu_only
def test_pdl_off_rows_are_the_literal_rows() -> None:
    """``PDL_OFF_ROWS`` is the literal :data:`PDL_OFF_ROWS_OF_RECORD`.  Every key names an exported architecture, one
    of its backward routes other than a pair route (whose programs launch without PDL), a closed range of channel
    counts (multiples of 8) and a closed range of token counts from one token on; the keys of one architecture and
    route are disjoint; and at each corner of a key the call selects a program of
    :data:`PDL_OFF_PROGRAMS_OF_RECORD` that takes the per-call PDL scalar."""

    assert cake_jit.PDL_OFF_ROWS == PDL_OFF_ROWS_OF_RECORD
    assert set(cake_jit.PDL_OFF_ROWS) == set(PDL_OFF_PROGRAMS_OF_RECORD)
    for arch, rows in cake_jit.PDL_OFF_ROWS.items():
        assert arch in ARCHES and rows, arch
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.backward_kinds(arch) and route not in cake_jit.PAIR_ROUTES, label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert type(s_min) is int and type(s_max) is int, label
            assert 1 <= s_min <= s_max <= cake_jit.INT32_MAX, label
            for C in (c_lo, c_hi):
                for S in (s_min, s_max):
                    (name,) = cake_jit.selected_modules(arch, route, C=C, S=S)
                    assert name in PDL_OFF_PROGRAMS_OF_RECORD[arch], (label, C, S, name)
                    assert cake_jit.MODULES[name]["launch"]["use_pdl"] == "runtime", (label, name)
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in itertools.pairwise(ranges):
                assert hi < lo, (arch, route, ranges)


def _pdl_off_expected(arch: str, kind: str, C: int, S: int, capturing: bool) -> int:
    """The ``enable_pdl`` value of a call of route ``kind`` with ``C`` channels and ``S`` tokens on ``arch``, from
    :data:`PDL_OFF_ROWS_OF_RECORD` and the live ``PDL_EAGER`` / ``PDL_GRAPH`` (written apart from ``pdl_enabled``)."""

    for (route, c_lo, c_hi), (s_min, s_max) in PDL_OFF_ROWS_OF_RECORD.get(arch, {}).items():
        if route == kind and c_lo <= C <= c_hi and C % 8 == 0 and s_min <= S <= s_max:
            return 0
    return int((cake_jit.PDL_GRAPH if capturing else cake_jit.PDL_EAGER)[arch])


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_pdl_enabled_is_off_inside_the_pdl_off_rows(arch, monkeypatch) -> None:
    """With a call's row (``kind``, ``C`` and ``S``) ``pdl_enabled`` is 0 inside a :data:`PDL_OFF_ROWS_OF_RECORD` key
    (at both channel and both token bounds), eagerly and inside a capture, whatever ``PDL_EAGER`` / ``PDL_GRAPH``
    say; outside every key it is the tables' value, as without a row: one token below a key, a channel count off the
    8-channel grid, every other route (``bwd_bias``, ``bwd_dt`` and the forward routes among them) at a key's row,
    and every row on an architecture without keys (sm_107a); one channel step below and above a key and one token
    above it unless another key covers them.
    ``pdl_off_row_range`` holds the key's token range at its channel counts only.  The row is ``kind``, ``C`` and
    ``S`` together: a ``kind`` outside ``KINDS``, or a part of the row alone, is a ``ValueError``."""

    rows = PDL_OFF_ROWS_OF_RECORD.get(arch, {})
    assert cake_jit.PDL_EAGER[arch] and cake_jit.PDL_GRAPH[arch], arch
    for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
        for capturing in (False, True):
            for C in (c_lo, c_hi):
                for S in (s_min, s_max):
                    assert cake_jit.pdl_enabled(arch, capturing=capturing, kind=route, C=C, S=S) == 0, (C, S)
            # one token below the key, the key's row on another route (forward included), the key's row on sm_107a
            outside = [(route, c_lo, s_min - 1), (route, c_hi, s_min - 1)]
            outside += [
                (kind, C, s_min) for kind in ("bwd_bias", "bwd_dt", "fwd") if kind != route for C in (c_lo, c_hi)
            ]
            for kind, C, S in outside:
                got = cake_jit.pdl_enabled(arch, capturing=capturing, kind=kind, C=C, S=S)
                assert got == 1, (kind, C, S, capturing)
            assert cake_jit.pdl_enabled("sm_107a", capturing=capturing, kind=route, C=c_hi, S=s_min) == 1
            # next to the key (another key of the route may cover these): the tables' value unless a key covers it
            near = [(c_lo - 8, s_min), (c_hi + 8, s_min), (c_lo + 4, s_min), (c_hi, s_max + 1)]
            for C, S in near:
                got = cake_jit.pdl_enabled(arch, capturing=capturing, kind=route, C=C, S=S)
                assert got == _pdl_off_expected(arch, route, C, S, capturing), (route, C, S, capturing)
            assert cake_jit.pdl_enabled(arch, capturing=capturing, kind=route, C=c_lo + 4, S=s_min) == 1
    channels = {512, 1016, 1024}
    tokens = {1, 1000, 125170, 131072, 131073, 187754, 187755, 250340, cake_jit.INT32_MAX}
    for (_, c_lo, c_hi), (s_min, s_max) in rows.items():
        channels |= {c_lo - 8, c_lo, c_lo + 4, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
        tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    turned_off = 0
    for eager_on, graph_on in ((True, True), (False, True), (True, False), (False, False)):
        monkeypatch.setitem(cake_jit.PDL_EAGER, arch, eager_on)
        monkeypatch.setitem(cake_jit.PDL_GRAPH, arch, graph_on)
        for capturing in (False, True):
            table = int(graph_on if capturing else eager_on)
            assert cake_jit.pdl_enabled(arch, capturing=capturing) == table
            for kind in cake_jit.KINDS:
                for C in sorted(c for c in channels if c > 0):
                    keys = [b for (r, lo, hi), b in rows.items() if r == kind and lo <= C <= hi and C % 8 == 0]
                    assert cake_jit.pdl_off_row_range(arch, kind, C) == (keys[0] if keys else None), (kind, C)
                    for S in sorted(s for s in tokens if s > 0):
                        want = _pdl_off_expected(arch, kind, C, S, capturing)
                        got = cake_jit.pdl_enabled(arch, capturing=capturing, kind=kind, C=C, S=S)
                        assert type(got) is int and got == want, (kind, C, S, capturing, eager_on, graph_on)
                        turned_off += want == 0 and table == 1
    assert bool(turned_off) is bool(rows), (arch, turned_off)
    for row in (
        dict(kind="bwd"),
        dict(C=1016),
        dict(S=187755),
        dict(kind="bwd", C=1016),
        dict(kind="bwd", S=187755),
        dict(C=1016, S=187755),
        dict(kind="bwd_unknown", C=1016, S=187755),
    ):
        with pytest.raises(ValueError):
            cake_jit.pdl_enabled(arch, capturing=False, **row)
    with pytest.raises(ValueError):
        cake_jit.pdl_enabled("sm_90a", capturing=False, kind="bwd", C=1016, S=187755)


_KERNEL_SIGNATURE = re.compile(r"__global__[^;{]*?\b(kernel_\w+)\s*\(")


def _kernel_waits_first(text: str, library: str) -> int:
    """Check every ``__global__`` function of a kernel unit: its body executes ``griddepcontrol.wait`` and, before the
    first one, names none of its pointer or tensor-map parameters (the text before it, comments dropped; a member
    access such as ``threadIdx.x`` is not the parameter ``x``).  Returns the number of functions checked."""

    checked = 0
    for match in _KERNEL_SIGNATURE.finditer(text):
        depth, i = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        params = text[match.end() : i - 1].split(",")
        guarded = [
            re.findall(r"\w+", param)[-1]
            for param in params
            if "*" in param or "CUtensorMap" in param
        ]
        body_start = text.index("{", i)
        wait = text.find("griddepcontrol.wait", body_start)
        label = (library, match.group(1))
        assert guarded and wait > body_start, label
        before = re.sub(r"//[^\n]*", "", text[body_start:wait])
        named = [name for name in guarded if re.search(rf"(?<!\.)(?<!->)\b{re.escape(name)}\b", before)]
        assert not named, (label, named)
        checked += 1
    return checked


@pytest.mark.cpu_only
def test_runtime_pdl_bindings_and_kernels() -> None:
    """The rendered binding of every library whose records take the per-call PDL scalar takes ``enable_pdl`` as its
    last host parameter, refuses a value other than 0 and 1 and sets the programmatic stream serialization attribute
    only when it is 1; a pair-route library's binding has no PDL attribute.  Every kernel of a PDL library waits for
    the preceding grid (``griddepcontrol.wait``) before it names any pointer or tensor-map parameter, so the
    attribute only lets its launch start early, never read or write before the preceding kernel finishes."""

    pair = _pair_programs()
    libraries = {record["library"]: name in pair for name, record in cake_jit.MODULES.items()}
    csrc = PACKAGE_DIR / "csrc"
    kernels = 0
    for library, is_pair in sorted(libraries.items()):
        binding = (csrc / f"{library}_binding.cu").read_text()
        if is_pair:
            assert "enable_pdl" not in binding and "ProgrammaticStreamSerialization" not in binding, library
            continue
        (run,) = re.findall(r"void Run\(([^)]*)\)", binding)
        params = [param.strip() for param in run.split(",")]
        assert params[-4:] == [
            "int64_t a_enable_pdl",
            "int64_t grid_x",
            "int64_t grid_y",
            "int64_t grid_z",
        ], library
        assert "a_enable_pdl == 0LL || a_enable_pdl == 1LL" in binding, library
        assert binding.count("cudaLaunchAttributeProgrammaticStreamSerialization") == 1, library
        attribute = binding.index("cudaLaunchAttributeProgrammaticStreamSerialization")
        assert binding.rfind("if (a_enable_pdl) {", 0, attribute) > binding.index("void Run("), library
        kernels += _kernel_waits_first((csrc / f"{library}_kernel.cu").read_text(), library)
    assert kernels >= RUNTIME_PDL_PROGRAMS


FORWARD_ROUTES = ("fwd", "fwd_dt", "fwd_bias")


def _forward_rows(arch: str) -> list[tuple[str, int, int, int, int, str]]:
    """``(route, C_lo, C_hi, S_lo, S_hi, form)`` of every forward key of ``arch`` in :data:`FALLBACK_ROWS_OF_RECORD`
    (form ``"static"``), :data:`DIRECT_ROWS_OF_RECORD` (form ``"direct"``) and :data:`PAIR_ROWS_OF_RECORD` (form
    ``"pair"``)."""

    rows = [
        (route, c_lo, c_hi, s_lo, s_hi, "static")
        for (route, c_lo, c_hi), (s_lo, s_hi) in FALLBACK_ROWS_OF_RECORD.get(
            arch, {}
        ).items()
        if route in FORWARD_ROUTES
    ]
    for form, table in (
        ("direct", DIRECT_ROWS_OF_RECORD),
        ("pair", PAIR_ROWS_OF_RECORD),
    ):
        rows += [
            (route, c_lo, c_hi, s_lo, s_hi, form)
            for (route, c_lo, c_hi), (s_lo, s_hi, _) in table.get(arch, {}).items()
        ]
    return rows


def _in_pair_of_record(arch: str, kind: str, C: int, S: int) -> bool:
    """Whether a call of route ``kind`` on ``arch`` with ``C`` channels (a multiple of 8) and ``S`` tokens lies in a
    range of :data:`PAIR_ROWS_OF_RECORD`, where the route's pair direct program runs ahead of every other table."""

    return C % 8 == 0 and any(
        route == kind and c_lo <= C <= c_hi and s_lo <= S <= s_hi
        for (route, c_lo, c_hi), (s_lo, s_hi, _) in PAIR_ROWS_OF_RECORD.get(
            arch, {}
        ).items()
    )


def _forward_edges(arch: str) -> tuple[set[int], set[int]]:
    """The channel counts (both sides of every channel bound, one off the 8-channel grid) and token counts (1000 and
    both sides of every token bound) of the forward keys of ``arch`` (:func:`_forward_rows`)."""

    channels, tokens = set(), {1000}
    for _, c_lo, c_hi, s_lo, s_hi, _ in _forward_rows(arch):
        channels |= {c_lo - 8, c_lo, c_lo + 4, c_hi, c_hi + 8}
        tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    return {C for C in channels if C > 0}, {S for S in tokens if S > 0}


def _backward_edges(arch: str) -> tuple[set[int], set[int]]:
    """The channel counts (both sides of every channel bound) and token counts (both sides of every token bound) of
    the backward keys of ``arch`` in :data:`FALLBACK_ROWS_OF_RECORD` and :data:`BACKWARD_DIRECT_ROWS_OF_RECORD`."""

    channels, tokens = set(), set()
    for (route, c_lo, c_hi), (s_lo, s_hi) in FALLBACK_ROWS_OF_RECORD.get(
        arch, {}
    ).items():
        if route not in FORWARD_ROUTES:
            channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
            tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    for (_, c_lo, c_hi), (s_lo, s_hi, _) in BACKWARD_DIRECT_ROWS_OF_RECORD.get(
        arch, {}
    ).items():
        channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
        tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    return {C for C in channels if C > 0}, {S for S in tokens if S > 0}


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_selection_matches_the_deterministic_package(arch, monkeypatch) -> None:
    """A call selects the program at the same route position as the deterministic package for every ``(S, C)``,
    and plans the same scalars (``y`` / ``dx`` keep their per-element arithmetic).  Where the deterministic package
    selects a det-only program, this package selects what the deterministic package selects without its det-only
    records but those this package copies (:data:`OWN_COPIES`), and its copy where that is one of them.  At a row
    of :data:`FALLBACK_ROWS_OF_RECORD` of a backward route this package selects its route's fallback (the program at
    the deterministic route's first position), where the deterministic package selects the program at the position
    of one of :data:`FALLBACK_REPLACES_OF_RECORD`.  At a segment of :data:`SM_WIDE_ROWS_OF_RECORD` this package
    selects the segment's SM-wide program, where the deterministic package (without its det-only records) selects
    the program at the position of one of :data:`SM_WIDE_REPLACES_OF_RECORD` or its copy, and plans the scalars of
    the SM-wide program's deterministic twin (:data:`SM_WIDE_COPIES`).  The forward keys of ``FALLBACK_ROWS``,
    ``DIRECT_ROWS`` and ``PAIR_ROWS`` exist in both packages: inside and outside every range (both sides of each
    bound, :func:`_forward_edges`) both select the program at the same route position, the static program inside a
    static range, the direct program inside a direct range and the pair direct program inside a pair range, which
    runs ahead of the other two.  Inside a backward key of :data:`BACKWARD_DIRECT_ROWS_OF_RECORD` this package selects
    the route's direct backward program, which has no deterministic twin; on both sides of each of its bounds
    (:func:`_backward_edges`) the rules above hold."""

    twin_of = {
        band: twin for band_arch, _, band, twin, _ in _bands() if band_arch == arch
    }
    det_only_hits = set()
    replaced = set()
    sm_wide_replaced = set()
    forward_hits = set()
    direct_hits = set()
    edge_channels, edge_tokens = _forward_edges(arch)
    backward_channels, backward_tokens = _backward_edges(arch)
    for kind in cake_jit.KINDS:
        new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
        forward = cake_jit.route_role(kind) == "fwd"
        channels = (
            sorted(set(CHANNELS) | {C for C in edge_channels if C % 8 == 0})
            if forward
            else sorted(set(CHANNELS) | backward_channels)
        )
        tokens = sorted(set(TOKENS) | (edge_tokens if forward else backward_tokens))
        for C in channels:
            for S in tokens:
                (name,) = cake_jit.selected_modules(arch, kind, C=C, S=S)
                (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                if det_name in DET_ONLY:
                    det_only_hits.add(det_name)
                    copied = [
                        d
                        for d in det_jit.ROUTES[arch][kind]
                        if d not in DET_ONLY or d in OWN_COPIES
                    ]
                    with monkeypatch.context() as m:
                        m.setitem(det_jit.ROUTES[arch], kind, copied)
                        (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                direct = _backward_direct_cell(arch, kind, C, S)
                if direct is not None:
                    # The route's direct backward program, ahead of every other table; it has no deterministic twin.
                    assert name == direct, (arch, kind, C, S, name)
                    direct_hits.add(kind)
                    continue
                bounds = cake_jit.fallback_row_range(arch, kind, C)
                segment = [
                    program
                    for (route, c_lo, c_hi), segments in SM_WIDE_ROWS_OF_RECORD.get(arch, {}).items()
                    if route == kind and c_lo <= C <= c_hi and C % 8 == 0
                    for s_lo, s_hi, program in segments
                    if s_lo <= S <= s_hi
                ]
                if segment:
                    # The segment's SM-wide program, in place of the program at the deterministic pick's position.
                    assert [name] == segment, (arch, kind, C, S, name)
                    taken = OWN_COPIES.get(det_name) or new[det.index(det_name)]
                    assert taken in SM_WIDE_REPLACES_OF_RECORD[arch], (arch, kind, C, S, taken)
                    sm_wide_replaced.add(taken)
                    det_name = SM_WIDE_COPIES[name]
                elif not forward and bounds is not None and bounds[0] <= S <= bounds[1]:
                    # The route's fallback, in place of the program at the deterministic pick's position.
                    assert name == new[0] and cake_jit.select_rule(cake_jit.MODULES[name]) is None, (
                        arch, kind, C, S, name
                    )
                    taken = new[det.index(det_name)]
                    assert taken in FALLBACK_REPLACES_OF_RECORD[arch], (arch, kind, C, S, taken)
                    replaced.add(taken)
                elif det_name in OWN_COPIES:
                    # This package's copy of the deterministic channel-band program.
                    assert name == OWN_COPIES[det_name], (arch, kind, C, S, name)
                else:
                    # A channel-band program serves its twin's route position.
                    position = new.index(twin_of.get(name, name))
                    assert position == det.index(det_name), (arch, kind, C, S)
                if forward:
                    for form in ("static", "direct", "pair"):
                        if name == _forward_program(arch, kind, form):
                            forward_hits.add((kind, form))
                record, det_record = cake_jit.MODULES[name], det_jit.MODULES[det_name]
                geometry = record["grid_rule"]["geometry"]
                assert cake_jit.plan(
                    S, C, 3, geometry=geometry, role=record["role"]
                ) == det_jit.plan(
                    S,
                    C,
                    3,
                    geometry=det_record["grid_rule"]["geometry"],
                    role=det_record["role"],
                ), (arch, kind, C, S)
    det_only_routed = {
        name for name in DET_ONLY if arch in det_jit.MODULES[name]["arches"]
    }
    assert det_only_hits == det_only_routed, arch
    assert replaced == FALLBACK_REPLACES_OF_RECORD.get(arch, frozenset()), arch
    assert sm_wide_replaced == SM_WIDE_REPLACES_OF_RECORD[arch], arch
    assert forward_hits == {(route, form) for route, *_, form in _forward_rows(arch)}, (
        arch
    )
    assert direct_hits == {
        route for route, _, _ in BACKWARD_DIRECT_ROWS_OF_RECORD[arch]
    }, arch


# The FP32 accumulator slots each backward route keeps (one FP32 word per sum and slot): the fp32g routes spread a
# channel tile's CTAs over 16 copies, the others add into one; the opt-in pair routes keep 4 copies of an FP32
# (hi, lo) pair, 8 slots.
ACC_SLOTS = {
    "bwd": 1,
    "bwd_dt": 1,
    "bwd_bias": 1,
    "bwd_fp32g": 16,
    "bwd_bias_fp32g": 16,
    "bwd_fp32g_pair": 8,
    "bwd_bias_fp32g_pair": 8,
}


def _align16(n: int) -> int:
    return -(-n // 16) * 16


@pytest.mark.cpu_only
def test_accumulator_layout() -> None:
    assert cake_jit.WORKSPACE_ALIGN == 16
    assert cake_jit.ACCUMULATOR_DTYPES == {"float32": 4, "float64": 8}
    assert cake_jit.COUNTER_DTYPES == {"uint32": 4}
    for arch, kind, name, _ in _pairs("bwd"):
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        size = cake_jit.ACCUMULATOR_DTYPES[rule["acc"]]
        for C in CHANNELS:
            _, _, n_ch_tiles = cake_jit.channel_split(
                C, geometry=record["grid_rule"]["geometry"], role="bwd"
            )
            layout = cake_jit.accumulator_layout(record, C, n_ch_tiles)
            words = rule["slots"] * rule["taps"]
            assert cake_jit.accumulator_words_per_channel(record) == words
            acc_bytes = words * C * size
            assert layout == {
                "acc_dtype": rule["acc"],
                "acc_elems": words * C,
                "acc_bytes": acc_bytes,
                "work_dtype": "uint32",
                "work_offset": _align16(acc_bytes),
                "work_words": n_ch_tiles,
                "total_bytes": _align16(_align16(acc_bytes) + 4 * n_ch_tiles),
            }, (arch, kind, C)
            # The views the host takes stay inside the buffer and do not overlap.
            assert layout["acc_bytes"] <= layout["work_offset"]
            assert layout["work_offset"] + 4 * n_ch_tiles <= layout["total_bytes"]
            assert layout["work_offset"] % size == 0
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 0, 1)
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 256, 0)
    forward = cake_jit.MODULES[cake_jit.ROUTES["sm_100a"]["fwd"][0]]
    with pytest.raises(ValueError, match="no backward accumulator workspace"):
        cake_jit.accumulator_layout(forward, 256, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_accumulators_are_sized_from_the_rule(kind, C) -> None:
    """The accumulator size comes from the rendered rule, never a fixed ``C x taps``: an fp32g program whose rule
    names ``slots`` FP32 copies gets ``slots x taps x C`` words; a rule without ``slots``, or with a malformed one,
    is refused."""

    import copy

    for arch in ARCHES:
        if kind not in cake_jit.ROUTES[arch]:
            continue
        name = cake_jit.route_modules(arch, kind)[0]
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        assert rule["acc"] == "float32" and rule["slots"] == ACC_SLOTS[kind]
        _, _, n_ch_tiles = cake_jit.channel_split(
            C, geometry=record["grid_rule"]["geometry"], role="bwd"
        )
        rendered = cake_jit.accumulator_layout(record, C, n_ch_tiles)
        assert rendered["acc_elems"] == rule["slots"] * rule["taps"] * C
        for slots in (1, 4):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            layout = cake_jit.accumulator_layout(variant, C, n_ch_tiles)
            assert layout["acc_elems"] == slots * rule["taps"] * C, (arch, slots)
            assert layout["acc_bytes"] == 4 * slots * rule["taps"] * C
            assert layout["work_offset"] == _align16(layout["acc_bytes"])
            assert layout["total_bytes"] >= layout["work_offset"] + 4 * n_ch_tiles
        for slots in (0, -1, 1.5, True, "4", None):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            with pytest.raises(ValueError, match="slots"):
                cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        # A rule without the field is refused (no default could size dw_acc for every program).
        variant = copy.deepcopy(record)
        del variant["workspace_rule"]["slots"]
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_words_per_channel(variant)
        with pytest.raises(ValueError, match="slots"):
            pkg._accumulator_layout(variant, C, n_ch_tiles)


def _owner_tiles(record: dict, arch: str, C: int) -> int:
    """The largest channel tile count a backward ``record`` launches with at ``C`` on ``arch``, whatever ``S``:
    the split's, or its route's :data:`cake_jit.TILE_OVERRIDES`, :data:`cake_jit.NARROW_TILES` or backward
    ``DIRECT_ROWS`` row's (:data:`BACKWARD_DIRECT_SPLITS`, by the rule) when that has more."""

    geometry = record["grid_rule"]["geometry"]
    route = cake_jit.record_route(record, arch)
    _, _, n_ch_tiles = cake_jit.channel_split(C, geometry=geometry, role="bwd")
    entry = cake_jit.TILE_OVERRIDES.get(arch, {}).get((route, C))
    if entry is not None:
        n_ch_tiles = max(n_ch_tiles, -(-(C // geometry["vec"]) // entry[0]))
    if (
        cake_jit.narrow_tile_range(arch, route, C) is not None
        and geometry["threads"] == 128
    ):
        n_ch_tiles = max(n_ch_tiles, -(-(C // geometry["vec"]) // 32))
    if (
        any(
            name == route and c_lo <= C <= c_hi and C % 8 == 0
            for name, c_lo, c_hi in BACKWARD_DIRECT_ROWS_OF_RECORD.get(arch, {})
        )
        and geometry["threads"] == 128
    ):
        n_ch_tiles = max(n_ch_tiles, _backward_direct_split(C)[2])
    return n_ch_tiles


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_workspace_serves_every_backward_program(arch) -> None:
    """The public backward workspace is the largest layout over every backward program of the architecture, each
    at the largest tile count it may launch with (its split, or its route's ``TILE_OVERRIDES`` or ``NARROW_TILES``
    row), so one zeroed buffer serves whichever program, plan and route a call selects; the capacity owner is an
    fp32g bias program (4 bytes x 16 copies x 5 taps per channel, 320).  A narrow row's three tickets fit the
    16-byte alignment of the single tile's, so its size is the split's.  The SM-wide programs (another build
    geometry: one channel tile at 384 or 256 threads) own the workspace too but never bound its size (16 or 32 bytes
    per channel and one ticket), and the capacity owner stays a program of the fallbacks' geometry."""

    owner = cake_jit.workspace_record(arch)
    assert (
        owner["workspace_rule"]["acc"] == "float32"
        and owner["workspace_rule"]["slots"] == ACC_SLOTS["bwd_bias_fp32g"]
        and owner["workspace_rule"]["taps"] == 5
    )
    owners = cake_jit.workspace_owners(arch)
    assert len(owners) == sum(
        len(cake_jit.route_modules(arch, kind))
        for kind in cake_jit.backward_kinds(arch)
    )
    assert owner["grid_rule"]["geometry"] == cake_jit.workspace_geometry(arch) == {"threads": 128, "vec": 8}
    sm_wide = [record for record in owners if record["kernel"] in SM_WIDE_COPIES]
    assert {record["kernel"] for record in sm_wide} == {
        name for name, det in SM_WIDE_COPIES.items() if arch in det_jit.MODULES[det]["arches"]
    }
    overridden = {C for _, C in cake_jit.TILE_OVERRIDES.get(arch, {})}
    narrow = {
        C for _, c_lo, c_hi in cake_jit.NARROW_TILES.get(arch, {}) for C in (c_lo, c_hi)
    }
    for C in sorted(set(CHANNELS) | overridden | narrow):
        nbytes = cake_jit.backward_workspace_bytes(arch, C)
        layouts = [
            cake_jit.accumulator_layout(record, C, _owner_tiles(record, arch, C))
            for record in owners
        ]
        assert nbytes == max(layout["total_bytes"] for layout in layouts), (arch, C)
        for record in sm_wide:
            if C // 8 <= record["grid_rule"]["geometry"]["threads"] and C not in overridden:
                assert _owner_tiles(record, arch, C) == 1, (arch, C, record["kernel"])
            layout = cake_jit.accumulator_layout(record, C, _owner_tiles(record, arch, C))
            assert layout["acc_bytes"] == 16 * SM_WIDE_SLOTS[record["kernel"]] * C, (arch, C)
            assert layout["total_bytes"] < nbytes, (arch, C, record["kernel"])
        if C not in overridden:
            per_channel = 4 * cake_jit.accumulator_words_per_channel(owner)
            assert nbytes == _align16(_align16(per_channel * C) + 4 * cake_jit.channel_split(
                C, geometry=owner["grid_rule"]["geometry"], role="bwd")[2])  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# The narrower channel tiles of the backward (``TILE_OVERRIDES``).


@pytest.mark.cpu_only
def test_every_route_plans_by_its_own_tile_rows() -> None:
    """No route plans by another route's tile rows: the package keeps no map from one route to another's rows, and
    every key of ``TILE_OVERRIDES`` and ``NARROW_TILES`` names a route of ``KINDS``, whose calls plan by it."""

    assert not hasattr(cake_jit, "TILE_ROUTE")
    for table in (cake_jit.TILE_OVERRIDES, cake_jit.NARROW_TILES):
        for rows in table.values():
            for key in rows:
                assert key[0] in cake_jit.KINDS, key


@pytest.mark.cpu_only
def test_tile_overrides_are_backward_rows() -> None:
    """Every row of the rendered table names an exported architecture, one of its backward routes, a channel count
    of the package and a ``ct_vec`` that is a whole number of warps no wider than a CTA, over a closed ``S`` range."""

    assert set(cake_jit.TILE_OVERRIDES) <= set(ARCHES)
    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (ct_vec, s_min, s_max) in rows.items():
            label = (arch, route, C)
            assert route in cake_jit.backward_kinds(arch), label
            assert C > 0 and C % cake_jit.CHANNEL_MULTIPLE == 0, label
            assert 0 < s_min <= s_max, label
            for name in cake_jit.route_modules(arch, route):
                threads = cake_jit.MODULES[name]["grid_rule"]["geometry"]["threads"]
                assert ct_vec % 32 == 0 and 0 < ct_vec <= threads, label


@pytest.mark.cpu_only
def test_tile_override_bounds_and_plan() -> None:
    """A row's ``ct_vec`` applies from ``S_min`` through ``S_max`` (both included) and to no other token count,
    route or architecture; ``plan`` of a backward record given its architecture and route then plans ``(ct_vec,
    threads // ct_vec, ceil(C / vec / ct_vec))`` (but the direct tiles of :data:`BACKWARD_DIRECT_SPLITS` at 128
    threads inside a backward direct key), and :func:`cake_jit.channel_split`'s split outside the range, for
    the forward role without a route, under ``legacy`` and without the architecture or the route; the forward role
    refuses a backward route (``ValueError``, as the deterministic plan does)."""

    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (ct_vec, s_min, s_max) in rows.items():
            label = (arch, route, C)
            assert cake_jit.tile_override(arch, route, C, s_min - 1) is None, label
            assert cake_jit.tile_override(arch, route, C, s_min) == ct_vec, label
            assert cake_jit.tile_override(arch, route, C, s_max) == ct_vec, label
            assert cake_jit.tile_override(arch, route, C, s_max + 1) is None, label
            for other in ARCHES:
                if (route, C) not in cake_jit.TILE_OVERRIDES.get(other, {}):
                    assert cake_jit.tile_override(other, route, C, s_min) is None
            for name in cake_jit.route_modules(arch, route):
                geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
                threads, vec = geometry["threads"], geometry["vec"]
                split = dict(
                    zip(
                        ("ct_vec", "n_sub", "n_ch_tiles"),
                        cake_jit.channel_split(C, geometry=geometry, role="bwd"),
                        strict=True,
                    )
                )
                wanted = {
                    "ct_vec": ct_vec,
                    "n_sub": threads // ct_vec,
                    "n_ch_tiles": -(-(C // vec) // ct_vec),
                }
                for S in (s_min, s_max):
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                    )
                    want = wanted
                    if (
                        _backward_direct_cell(arch, route, C, S) is not None
                        and threads == 128
                    ):
                        # Inside a backward direct key the direct tiles.
                        want = dict(
                            zip(
                                ("ct_vec", "n_sub", "n_ch_tiles"),
                                BACKWARD_DIRECT_SPLITS[C],
                                strict=True,
                            )
                        )
                    assert {k: got[k] for k in want} == want, (label, S)
                for S in (s_min - 1, s_max + 1):
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                    )
                    assert {k: got[k] for k in split} == split, (label, S)
                for kwargs in (
                    {"role": "bwd"},
                    {"role": "bwd", "arch": arch},
                    {"role": "bwd", "route": route},
                    {"role": "fwd", "arch": arch},
                ):
                    got = cake_jit.plan(s_min, C, 3, geometry=geometry, **kwargs)
                    want = cake_jit.channel_split(
                        C, geometry=geometry, role=kwargs["role"]
                    )
                    assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, (
                        label,
                        kwargs,
                    )
                for registry in (cake_jit, det_jit):
                    with pytest.raises(ValueError, match="is not a route of role"):
                        registry.plan(
                            s_min,
                            C,
                            3,
                            geometry=geometry,
                            role="fwd",
                            arch=arch,
                            route=route,
                        )
                got = cake_jit.plan(
                    s_min, C, 3, geometry=geometry, legacy=True, arch=arch, route=route
                )
                want = cake_jit.channel_split(C, geometry=geometry, legacy=True)
                assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, label


@pytest.mark.cpu_only
def test_backward_workspace_covers_every_override_row() -> None:
    """The public workspace of a row's ``C`` holds the layout of every program of the row's route (every route plans
    by its own rows) at the row's tile count (one ticket per tile), whatever ``S``: planned at the row's ``S_max``,
    above every backward direct key."""

    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (_, _, s_max) in rows.items():
            nbytes = cake_jit.backward_workspace_bytes(arch, C)
            for name in cake_jit.route_modules(arch, route):
                record = cake_jit.MODULES[name]
                geometry = record["grid_rule"]["geometry"]
                tiles = cake_jit.plan(
                    s_max,
                    C,
                    3,
                    geometry=geometry,
                    role="bwd",
                    arch=arch,
                    route=route,
                )["n_ch_tiles"]
                assert tiles == cake_jit.override_tiles(arch, route, C, geometry)
                layout = cake_jit.accumulator_layout(record, C, tiles)
                assert layout["total_bytes"] <= nbytes, (arch, route, C, name)


# The C 768 rows of ``TILE_OVERRIDES`` per architecture, route -> (ct_vec, S_min, S_max): on sm_100a and sm_103a the
# measured rows of the package's 0.2.0 release on every base route but ``bwd`` (whose C 768 runs the 32-vector tiles of
# its ``NARROW_TILES`` key at every S, :data:`C768_NARROW_RANGES`), on sm_107a the ``bwd_dt`` row of every S (the
# route's calls with FP32 and with BF16 taps plan by it).
C768_TILE_OVERRIDES = {
    "sm_100a": {
        "bwd_dt": (32, 62585, 500680),
        "bwd_bias": (32, 62585, 500680),
        "bwd_fp32g": (32, 31292, 500680),
        "bwd_bias_fp32g": (32, 31292, 500680),
    },
    "sm_103a": {
        "bwd_dt": (32, 62585, 500680),
        "bwd_bias": (32, 62585, 500680),
        "bwd_fp32g": (32, 62585, 500680),
        "bwd_bias_fp32g": (32, 31292, 500680),
    },
    "sm_107a": {"bwd_dt": (32, 1, 2147483647)},
}
# The ``NARROW_TILES`` token ranges at C 768 per architecture whose C 768 backward rows are measured, route ->
# (S_min, S_max): the ``bwd`` route's C 520 to 1016 key over every S; the ``bwd_dt`` and ``bwd_bias`` routes' own
# C 768 keys from S 1 up to the last S at which no measured backward, step or checkpointed-step row of the route's
# fallback ran slower with the 32-vector tiles than with the one tile (S 16384, S 8192 for sm_100a's ``bwd_dt``),
# below the ``S_min`` of their C 768 ``TILE_OVERRIDES`` rows, so between the two those calls keep the one tile; the
# ``bwd_fp32g`` and ``bwd_bias_fp32g`` routes have none.
C768_NARROW_RANGES = {
    "sm_100a": {"bwd": (1, 2147483647), "bwd_dt": (1, 8192), "bwd_bias": (1, 16384)},
    "sm_103a": {"bwd": (1, 2147483647), "bwd_dt": (1, 16384), "bwd_bias": (1, 16384)},
}


@pytest.mark.cpu_only
def test_c768_backward_rows_are_the_literal_rows() -> None:
    """The C 768 rows of the rendered ``TILE_OVERRIDES`` are the literal :data:`C768_TILE_OVERRIDES` on every
    architecture.  On the architectures of :data:`C768_NARROW_RANGES` the ``NARROW_TILES`` keys covering C 768 are
    exactly one per route of the literal, with its token range; no other backward route has a narrow range at C 768;
    a range at a ``(route, 768)`` of ``TILE_OVERRIDES`` ends below the row's ``S_min``; and ``plan`` gives the direct
    tiles inside the route's backward direct key covering C 768 (:data:`BACKWARD_DIRECT_SPLITS`), elsewhere the
    32-vector tiles up to the range's end, the one tile from the next token to the row's ``S_min - 1``, and the row's
    tiles from its ``S_min``."""

    assert set(C768_TILE_OVERRIDES) == set(ARCHES)
    for arch in ARCHES:
        rows = {
            route: tuple(int(value) for value in entry)
            for (route, C), entry in cake_jit.TILE_OVERRIDES.get(arch, {}).items()
            if C == 768
        }
        assert rows == C768_TILE_OVERRIDES[arch], arch
    geometry = {"threads": 128, "vec": 8}
    narrow, one_tile = (32, 4, 3), (96, 1, 1)
    for arch, ranges in C768_NARROW_RANGES.items():
        keys = cake_jit.NARROW_TILES.get(arch, {})
        covering = [key for key in keys if key[1] <= 768 <= key[2]]
        assert sorted(key[0] for key in covering) == sorted(ranges), (arch, covering)
        for route in cake_jit.backward_kinds(arch):
            want = ranges.get(route)
            assert cake_jit.narrow_tile_range(arch, route, 768) == want, (arch, route)
            row = C768_TILE_OVERRIDES[arch].get(route)
            if want is None or row is None:
                continue
            assert want[1] < row[1], (arch, route, want, row)
            override = (row[0], 128 // row[0], 96 // row[0])
            tokens = {
                want[0]: narrow,
                want[1]: narrow,
                want[1] + 1: one_tile,
                row[1] - 1: one_tile,
                row[1]: override,
            }
            # Inside the route's backward direct key covering C 768 the direct tiles; one token outside it the tiles
            # the other tables give there.
            for (name, c_lo, c_hi), (s_lo, s_hi, _) in BACKWARD_DIRECT_ROWS_OF_RECORD[
                arch
            ].items():
                if name != route or not c_lo <= 768 <= c_hi:
                    continue
                tokens.update({S: BACKWARD_DIRECT_SPLITS[768] for S in (s_lo, s_hi)})
                for S in (s_lo - 1, s_hi + 1):
                    if S >= 1:
                        tokens[S] = (
                            narrow
                            if want[0] <= S <= want[1]
                            else one_tile
                            if S < row[1]
                            else override
                        )
            for S, tiles in tokens.items():
                got = cake_jit.plan(
                    S, 768, 1, geometry=geometry, role="bwd", arch=arch, route=route
                )
                split = (got["ct_vec"], got["n_sub"], got["n_ch_tiles"])
                assert split == tiles, (arch, route, S, split)
                inside = S <= want[1]
                assert cake_jit.narrow_tile_row(arch, route, 768, S) == inside, (arch, S)


# ----------------------------------------------------------------------------------------------------------------
# The rows where a route runs its fallback program (``FALLBACK_ROWS``) and a forward route its direct program
# (``DIRECT_ROWS``).


def _route_fallback(arch: str, kind: str) -> str:
    """The fallback program of route ``kind`` on ``arch``: its one record without a selection rule."""

    (name,) = [
        name for name in cake_jit.route_modules(arch, kind) if cake_jit.select_rule(cake_jit.MODULES[name]) is None
    ]
    return name


def _rules_pick(arch: str, kind: str, C: int, S: int) -> str:
    """The program the selection rules pick for ``(S, C)`` of route ``kind`` on ``arch`` (no fallback rows, no SM-wide
    rows, no pair rows, no direct rows)."""

    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FALLBACK_ROWS", {})
        m.setattr(cake_jit, "SM_WIDE_ROWS", {})
        m.setattr(cake_jit, "PAIR_ROWS", {})
        m.setattr(cake_jit, "DIRECT_ROWS", {})
        (name,) = cake_jit.selected_modules(arch, kind, C=C, S=S)
    return name


def _table_pick(arch: str, kind: str, C: int, S: int) -> str | None:
    """The program the registry's ``PAIR_ROWS`` range, else its ``DIRECT_ROWS`` range, else its ``SM_WIDE_ROWS``
    segment, else its ``FALLBACK_ROWS`` range, holding ``(C, S)`` of route ``kind`` on ``arch`` names (read from the
    four tables, not through the selection), ``None`` outside all four."""

    if C % 8:
        return None
    for table in (cake_jit.PAIR_ROWS, cake_jit.DIRECT_ROWS):
        for (route, c_lo, c_hi), (s_lo, s_hi, name) in table.get(arch, {}).items():
            if route == kind and c_lo <= C <= c_hi and s_lo <= S <= s_hi:
                return name
    for (route, c_lo, c_hi), segments in cake_jit.SM_WIDE_ROWS.get(arch, {}).items():
        if route == kind and c_lo <= C <= c_hi:
            for s_lo, s_hi, name in segments:
                if s_lo <= S <= s_hi:
                    return name
    for (route, c_lo, c_hi), (s_min, s_max) in cake_jit.FALLBACK_ROWS.get(arch, {}).items():
        if route == kind and c_lo <= C <= c_hi and s_min <= S <= s_max:
            return _route_fallback(arch, kind)
    return None


@pytest.mark.cpu_only
def test_fallback_rows_are_the_literal_rows() -> None:
    """``FALLBACK_ROWS`` is the literal :data:`FALLBACK_ROWS_OF_RECORD`.  Every key names an exported architecture,
    one of its forward routes or of its backward routes other than a pair route (an opt-in route selects by its own
    rules only), a closed range of channel counts (multiples of 8) and a closed range of token counts inside
    ``S >= 1``; the keys of one architecture and route are disjoint, and the route's fallback program ships for the
    architecture (a forward route's static program)."""

    assert cake_jit.FALLBACK_ROWS == FALLBACK_ROWS_OF_RECORD
    for arch, rows in cake_jit.FALLBACK_ROWS.items():
        assert arch in ARCHES and rows, arch
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert (
                route in cake_jit.forward_kinds(arch) and route in FORWARD_ROUTES
            ) or (
                route in cake_jit.backward_kinds(arch)
                and route not in cake_jit.PAIR_ROUTES
            ), label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert type(s_min) is int and type(s_max) is int, label
            assert 1 <= s_min <= s_max <= cake_jit.INT32_MAX, label
            assert arch in cake_jit.MODULES[_route_fallback(arch, route)]["arches"], label
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in itertools.pairwise(ranges):
                assert hi < lo, (arch, route, ranges)


@pytest.mark.cpu_only
def test_direct_rows_are_the_literal_rows() -> None:
    """``DIRECT_ROWS`` is the literal :data:`DIRECT_ROWS_OF_RECORD` with the backward keys of
    :data:`BACKWARD_DIRECT_ROWS_OF_RECORD` (:func:`_direct_rows_of_record`) and ``DIRECT_PLANS`` the literal
    :data:`DIRECT_PLANS_OF_RECORD`.  Every forward key names an exported architecture, one of its forward routes, a
    closed range of channel counts (multiples of 8), a closed range of token counts inside ``S >= 1`` and the route's
    direct program (selection-rule kind ``rows``), which ships for the architecture; the keys of one architecture and
    route are disjoint; no ``(C, S)`` of a direct range lies in a ``FALLBACK_ROWS`` range, and where a
    ``FALLBACK_ROWS`` key covers the same channel counts its static range lies below the direct range.
    ``DIRECT_PLANS`` names a form
    ``(ct_vec, n_sub)`` for exactly the forward routes the architecture's keys name, within 128 threads (the backward
    keys: :func:`test_backward_direct_rows_are_the_literal_rows`)."""

    assert cake_jit.DIRECT_ROWS == _direct_rows_of_record()
    assert cake_jit.DIRECT_PLANS == DIRECT_PLANS_OF_RECORD
    assert set(cake_jit.DIRECT_ROWS) <= set(ARCHES) and set(
        cake_jit.DIRECT_PLANS
    ) <= set(ARCHES)
    for arch in ARCHES:
        rows = DIRECT_ROWS_OF_RECORD.get(arch, {})
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_lo, s_hi, name) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.forward_kinds(arch) and route in FORWARD_ROUTES, (
                label
            )
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert (
                type(s_lo) is int
                and type(s_hi) is int
                and 1 <= s_lo <= s_hi <= cake_jit.INT32_MAX
            ), label
            assert name == _forward_program(arch, route, "direct"), label
            assert arch in cake_jit.MODULES[name]["arches"], label
            for C in range(c_lo, c_hi + 1, 8):
                assert cake_jit.direct_row_range(arch, route, C) == (s_lo, s_hi), (
                    label,
                    C,
                )
                bounds = cake_jit.fallback_row_range(arch, route, C)
                assert bounds is None or bounds[1] < s_lo, (label, C, bounds)
            for C in (c_lo - 8, c_hi + 8, c_lo + 4):
                inside = [
                    k
                    for k in rows
                    if k[0] == route and k[1] <= C <= k[2] and C % 8 == 0
                ]
                assert inside or cake_jit.direct_row_range(arch, route, C) is None, (
                    label,
                    C,
                )
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in itertools.pairwise(ranges):
                assert hi < lo, (arch, route, ranges)
        plans = cake_jit.DIRECT_PLANS.get(arch, {})
        assert set(plans) == set(spans), arch
        for route, (ct_vec, n_sub) in plans.items():
            assert type(ct_vec) is int and type(n_sub) is int, (arch, route)
            assert ct_vec >= 1 and n_sub >= 1 and ct_vec * n_sub <= 128, (arch, route)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_direct_module_refuses_registry_errors(arch, monkeypatch) -> None:
    """``direct_module`` names the key's direct program inside its range and ``None`` outside it or without a key;
    ``S`` of ``None`` under a key is a ``ValueError``; a key naming a program of another route, or a record whose rule
    is not of kind ``rows``, is a ``RuntimeError`` (also through the selection).  ``direct_split`` refuses an
    architecture or route without a form and a form wider than the geometry (``ValueError``)."""

    geometry = {"threads": 128, "vec": 8}
    rows = DIRECT_ROWS_OF_RECORD.get(arch, {})
    for (route, c_lo, c_hi), (s_lo, s_hi, name) in rows.items():
        assert cake_jit.direct_module(arch, route, c_lo, s_lo) == name
        assert cake_jit.direct_module(arch, route, c_hi, s_hi) == name
        assert cake_jit.direct_module(arch, route, c_lo, s_hi + 1) is None
        assert cake_jit.direct_module(arch, route, c_lo, s_lo - 1) is None
        assert cake_jit.direct_module(arch, route, c_lo + 4, s_lo) is None
        assert cake_jit.direct_row(arch, route, c_lo, s_lo) and not cake_jit.direct_row(
            arch, route, c_lo, None
        )
        with pytest.raises(ValueError, match="needs S"):
            cake_jit.direct_module(arch, route, c_lo, None)
        ct_vec, n_sub = DIRECT_PLANS_OF_RECORD[arch][route]
        for C in (c_lo, c_hi):
            assert cake_jit.direct_split(
                C, geometry=geometry, route=route, arch=arch
            ) == (
                ct_vec,
                n_sub,
                -(-(C // 8) // ct_vec),
            ), (route, C)
    for route in FORWARD_ROUTES:
        if route not in DIRECT_PLANS_OF_RECORD.get(arch, {}):
            assert not any(key[0] == route for key in rows), (arch, route)
            with pytest.raises(ValueError, match="no direct plan"):
                cake_jit.direct_split(640, geometry=geometry, route=route, arch=arch)
    with pytest.raises(ValueError):
        cake_jit.direct_split(640, geometry=geometry, route="fwd", arch="sm_90a")
    if not rows:
        return
    (route, c_lo, c_hi), (s_lo, s_hi, name) = next(iter(rows.items()))
    with pytest.raises(ValueError, match="does not fit"):
        cake_jit.direct_split(
            c_lo, geometry={"threads": 32, "vec": 8}, route=route, arch=arch
        )
    other = _forward_program(
        arch, next(r for r in FORWARD_ROUTES if r != route), "static"
    )
    # A pair range would run ahead of the key's first token count.
    monkeypatch.setitem(cake_jit.PAIR_ROWS, arch, {})
    for wrong in (other, _forward_program(arch, route, "static")):
        monkeypatch.setitem(
            cake_jit.DIRECT_ROWS, arch, {(route, c_lo, c_hi): (s_lo, s_hi, wrong)}
        )
        with pytest.raises(RuntimeError, match="rows"):
            cake_jit.direct_module(arch, route, c_lo, s_lo)
        with pytest.raises(RuntimeError, match="rows"):
            cake_jit.selected_modules(arch, route, C=c_lo, S=s_lo)


@pytest.mark.cpu_only
def test_pair_rows_are_the_literal_rows() -> None:
    """``PAIR_ROWS`` is the literal :data:`PAIR_ROWS_OF_RECORD` and ``PAIR_PLANS`` the literal
    :data:`PAIR_PLANS_OF_RECORD`.  Every architecture has keys on every forward route, disjoint in C and together C
    512 to 1024 (multiples of 8): each a closed range of token counts inside ``S >= 1`` and the route's pair direct
    program (selection-rule kind ``rows``, two channels per thread), which ships for the architecture;
    ``pair_row_range`` gives the key's range at each of its channel counts, the neighbouring key's range at a channel
    count of the next key and ``None`` off every key.  ``PAIR_PLANS`` names a form ``(ct_vec, n_sub)`` for exactly
    the routes the architecture's keys name, filling 128 threads."""

    assert cake_jit.PAIR_ROWS == PAIR_ROWS_OF_RECORD
    assert cake_jit.PAIR_PLANS == PAIR_PLANS_OF_RECORD
    assert set(cake_jit.PAIR_ROWS) == set(ARCHES) == set(cake_jit.PAIR_PLANS)
    for arch in ARCHES:
        rows = cake_jit.PAIR_ROWS[arch]
        assert {route for route, _, _ in rows} == set(FORWARD_ROUTES), arch
        for route in FORWARD_ROUTES:
            spans = sorted((c_lo, c_hi) for key, c_lo, c_hi in rows if key == route)
            assert spans[0][0] == 512 and spans[-1][1] == 1024, (arch, route, spans)
            for (_, hi), (lo, _) in zip(spans, spans[1:]):
                assert lo == hi + 8, (arch, route, spans)
        for (route, c_lo, c_hi), (s_lo, s_hi, name) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.forward_kinds(arch), label
            assert 512 <= c_lo <= c_hi <= 1024, label
            assert (
                type(s_lo) is int
                and type(s_hi) is int
                and 1 <= s_lo <= s_hi <= cake_jit.INT32_MAX
            ), label
            assert name == _forward_program(arch, route, "pair"), label
            assert arch in cake_jit.MODULES[name]["arches"], label
            for C in range(c_lo, c_hi + 1, 8):
                assert cake_jit.pair_row_range(arch, route, C) == (s_lo, s_hi), (
                    label,
                    C,
                )
            for C in (c_lo - 8, c_hi + 8, c_lo + 4):
                near = [
                    (lo, hi)
                    for (key, k_lo, k_hi), (lo, hi, _) in rows.items()
                    if key == route and k_lo <= C <= k_hi and C % 8 == 0
                ]
                want = near[0] if near else None
                assert cake_jit.pair_row_range(arch, route, C) == want, (label, C)
        plans = cake_jit.PAIR_PLANS[arch]
        assert set(plans) == {route for route, _, _ in rows}, arch
        for route, (ct_vec, n_sub) in plans.items():
            assert type(ct_vec) is int and type(n_sub) is int, (arch, route)
            assert ct_vec >= 1 and n_sub >= 1 and ct_vec * n_sub == 128, (
                arch,
                route,
            )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_pair_module_refuses_registry_errors(arch, monkeypatch) -> None:
    """``pair_module`` names the key's pair direct program inside its range and ``None`` outside it or without a key
    (the neighbouring key's program at a channel count of the next key of the route, inside that key's range);
    ``S`` of ``None`` under a key is a ``ValueError``; a key naming a program of another route, or a record whose rule
    is not of kind ``rows``, is a ``RuntimeError`` (also through the selection).  ``direct_split`` with ``pair``
    plans the pair form at two channels per thread and refuses an architecture or route without a form and a form
    wider than the geometry (``ValueError``)."""

    geometry = {"threads": 128, "vec": FORM_VEC["pair"]}
    rows = PAIR_ROWS_OF_RECORD[arch]
    for (route, c_lo, c_hi), (s_lo, s_hi, name) in rows.items():
        assert cake_jit.pair_module(arch, route, c_lo, s_lo) == name
        assert cake_jit.pair_module(arch, route, c_hi, s_hi) == name
        assert cake_jit.pair_module(arch, route, c_lo, s_hi + 1) is None
        assert cake_jit.pair_module(arch, route, c_lo, s_lo - 1) is None
        assert cake_jit.pair_module(arch, route, c_lo + 4, s_lo) is None
        want = name if _in_pair_of_record(arch, route, c_lo - 8, s_lo) else None
        assert cake_jit.pair_module(arch, route, c_lo - 8, s_lo) == want
        assert cake_jit.pair_row(arch, route, c_lo, s_lo)
        assert not cake_jit.pair_row(arch, route, c_lo, None)
        assert not cake_jit.pair_row(None, route, c_lo, s_lo)
        with pytest.raises(ValueError, match="needs S"):
            cake_jit.pair_module(arch, route, c_lo, None)
        ct_vec, n_sub = PAIR_PLANS_OF_RECORD[arch][route]
        for C in (c_lo, 776, c_hi):
            assert cake_jit.direct_split(
                C, geometry=geometry, route=route, arch=arch, pair=True
            ) == (ct_vec, n_sub, -(-(C // 2) // ct_vec)), (route, C)
    assert cake_jit.pair_module(arch, "bwd", 768, 1000) is None
    with pytest.raises(ValueError, match="no pair plan"):
        cake_jit.direct_split(640, geometry=geometry, route="bwd", arch=arch, pair=True)
    with pytest.raises(ValueError):
        cake_jit.direct_split(
            640, geometry=geometry, route="fwd", arch="sm_90a", pair=True
        )
    (route, c_lo, c_hi), (s_lo, s_hi, name) = next(iter(rows.items()))
    with pytest.raises(ValueError, match="does not fit"):
        cake_jit.direct_split(
            c_lo, geometry={"threads": 32, "vec": 2}, route=route, arch=arch, pair=True
        )
    other = _forward_program(
        arch, next(r for r in FORWARD_ROUTES if r != route), "pair"
    )
    for wrong in (other, _forward_program(arch, route, "static")):
        monkeypatch.setitem(
            cake_jit.PAIR_ROWS, arch, {(route, c_lo, c_hi): (s_lo, s_hi, wrong)}
        )
        with pytest.raises(RuntimeError, match="PAIR_ROWS"):
            cake_jit.pair_module(arch, route, c_lo, s_lo)
        with pytest.raises(RuntimeError, match="PAIR_ROWS"):
            cake_jit.selected_modules(arch, route, C=c_lo, S=s_lo)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_fallback_rows_select_the_route_fallback(arch, fake_device) -> None:
    """Inside a key (``C_lo <= C <= C_hi``, ``C`` a multiple of 8, ``S_min <= S <= S_max``) a call of the key's route
    selects the route's fallback program (a forward route's static program), inside a ``DIRECT_ROWS`` range the
    route's direct program (a backward one also where a ``FALLBACK_ROWS`` range holds the same ``(C, S)``), inside a
    ``PAIR_ROWS`` range (ahead of the other two) the route's pair direct program, inside an ``SM_WIDE_ROWS`` segment
    the segment's program, and the program the selection
    rules pick everywhere else: both sides of every channel and token bound, on every forward and backward route of
    the architecture.  ``fallback_row_range`` holds the key's token range at its channel counts only; ``S`` of
    ``None`` is a ``ValueError`` where a key covers ``C``.  ``plan_row`` needs ``at`` there (``ValueError`` without
    it) and plans, at the first token count of every side of ``row_sides``, the program of that side (``None`` on a
    rules side where a token floor decides); at a row no key covers it plans the same row whatever ``at``."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    rows = cake_jit.FALLBACK_ROWS.get(arch, {})
    assert bool(rows) is (arch in FALLBACK_ROWS_OF_RECORD), arch
    channels = {768, 1024}
    tokens = {1, 1000, 1 << 20, cake_jit.INT32_MAX}
    for (_, c_lo, c_hi), (s_min, s_max) in rows.items():
        channels |= {c_lo - 8, c_lo, c_lo + 4, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
        tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    for (_, c_lo, c_hi), segments in cake_jit.SM_WIDE_ROWS.get(arch, {}).items():
        channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
        tokens |= {bound for s_lo, s_hi, _ in segments for bound in (s_lo - 1, s_lo, s_hi, s_hi + 1)}
    for table in (cake_jit.DIRECT_ROWS, cake_jit.PAIR_ROWS):
        for (_, c_lo, c_hi), (s_lo, s_hi, _) in table.get(arch, {}).items():
            channels |= {c_lo - 8, c_lo, c_lo + 4, c_hi, c_hi + 8}
            tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    inside_calls = direct_calls = pair_calls = 0
    for kind in (*cake_jit.forward_kinds(arch), *cake_jit.backward_kinds(arch)):
        fallback = _route_fallback(arch, kind)
        direct, pair = (
            (
                _forward_program(arch, kind, "direct"),
                _forward_program(arch, kind, "pair"),
            )
            if kind in FORWARD_ROUTES
            else (BACKWARD_DIRECT_PROGRAMS.get(kind), None)
        )
        for C in sorted(C for C in channels if C > 0):
            keys = [
                bounds
                for (route, c_lo, c_hi), bounds in rows.items()
                if route == kind and c_lo <= C <= c_hi and C % 8 == 0
            ]
            assert len(keys) <= 1, (kind, C)
            bounds = keys[0] if keys else None
            assert cake_jit.fallback_row_range(arch, kind, C) == bounds, (kind, C)
            for S in sorted(S for S in tokens if 0 < S <= cake_jit.INT32_MAX):
                # A backward direct range runs ahead of the fallback range it shares ``(C, S)`` with.
                inside = (
                    bounds is not None
                    and bounds[0] <= S <= bounds[1]
                    and not cake_jit.direct_row(arch, kind, C, S)
                )
                want = _table_pick(arch, kind, C, S) or _rules_pick(arch, kind, C, S)
                paired = _in_pair_of_record(arch, kind, C, S)
                # A pair range runs ahead of the fallback and the direct range it covers.
                assert (inside and not paired) is (
                    want == fallback and _table_pick(arch, kind, C, S) is not None
                ), (kind, C, S)
                assert cake_jit.selected_modules(arch, kind, C=C, S=S) == (want,), (kind, C, S)
                assert (want == direct) is (
                    cake_jit.direct_row(arch, kind, C, S) and not paired
                ), (
                    kind,
                    C,
                    S,
                )
                assert (
                    (want == pair) is paired is cake_jit.pair_row(arch, kind, C, S)
                ), (kind, C, S)
                inside_calls += inside
                direct_calls += want == direct
                pair_calls += want == pair
            if C % 8:
                continue

            def row(at, kind=kind, C=C):
                return cake_jit.plan_row(arch, kind, C, device_index=FAKE_DEVICE, at=at)

            sides = cake_jit.row_sides(arch, kind, C)
            if len(sides) == 1:
                assert bounds is None and sides == ((0, sys.maxsize),), (kind, C)
                assert row(None) == row(1) == row(cake_jit.INT32_MAX), (kind, C)
                continue
            with pytest.raises(ValueError, match="needs S"):
                cake_jit.selected_modules(arch, kind, C=C)
            with pytest.raises(ValueError, match="at="):
                row(None)
            for s_lo, _ in sides:
                planned, picked = row(s_lo), _table_pick(arch, kind, C, s_lo)
                if picked is None and cake_jit.tokens_select(arch, kind, C):
                    assert planned is None, (kind, C, s_lo)
                    continue
                assert planned.names == (picked or _rules_pick(arch, kind, C, s_lo),), (kind, C, s_lo)
                # Above a backward direct range with no FALLBACK_ROWS key at ``C`` the rules may pick the
                # route's fallback program itself (sm_107a ``bwd_dt`` at C 512).
                if (
                    picked is None
                    and s_lo > 0
                    and (kind in FORWARD_ROUTES or bounds is not None)
                ):
                    assert planned.names != (fallback,), (kind, C, s_lo)
    assert bool(inside_calls) is bool(rows), arch
    assert bool(direct_calls) is bool(DIRECT_ROWS_OF_RECORD.get(arch)), arch
    assert bool(pair_calls) is bool(PAIR_ROWS_OF_RECORD.get(arch)), arch


# ----------------------------------------------------------------------------------------------------------------
# The rows where route ``bwd`` runs an SM-wide program (``SM_WIDE_ROWS``).


@pytest.mark.cpu_only
def test_sm_wide_rows_are_the_literal_rows() -> None:
    """``SM_WIDE_ROWS`` is the literal :data:`SM_WIDE_ROWS_OF_RECORD`.  Every key names an exported architecture,
    its ``bwd`` route and a closed range of channel counts (multiples of 8) on which the route's rules plan one
    channel tile at 128 threads (no ``TILE_OVERRIDES`` row); its segments ascend, are contiguous and end at
    ``INT32_MAX``; each names a program of the route of selection-rule kind ``rows`` that ships for the architecture
    only, every such program of the route but its direct backward program is named by exactly one segment, and no
    ``(C, S)`` of a segment lies in a ``FALLBACK_ROWS`` range."""

    assert cake_jit.SM_WIDE_ROWS == SM_WIDE_ROWS_OF_RECORD
    for arch in ARCHES:
        rows = cake_jit.SM_WIDE_ROWS.get(arch, {})
        assert rows, arch
        named = []
        for (route, c_lo, c_hi), segments in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route == "bwd" and route in cake_jit.backward_kinds(arch), label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert segments and segments[-1][1] == cake_jit.INT32_MAX, label
            assert all(type(bound) is int for s_lo, s_hi, _ in segments for bound in (s_lo, s_hi)), label
            assert 1 <= segments[0][0], label
            for (_, hi, _), (lo, _, _) in itertools.pairwise(segments):
                assert lo == hi + 1, label
            for s_lo, s_hi, name in segments:
                assert s_lo <= s_hi and name in cake_jit.route_modules(arch, route), label
                assert cake_jit.MODULES[name]["arches"] == [arch], (label, name)
                assert cake_jit.select_rule(cake_jit.MODULES[name]) == {"kind": "rows"}, (label, name)
                named.append(name)
            for C in range(c_lo, c_hi + 1, 8):
                assert (route, C) not in cake_jit.TILE_OVERRIDES.get(arch, {}), (label, C)
                assert cake_jit.channel_split(C, geometry={"threads": 128, "vec": 8}, role="bwd")[2] == 1, (label, C)
                bounds = cake_jit.fallback_row_range(arch, route, C)
                assert bounds is None or bounds[1] < segments[0][0], (label, C, bounds)
                assert cake_jit.sm_wide_segments(arch, route, C) == segments, (label, C)
            for C in (c_lo - 8, c_hi + 8, c_lo + 4):
                assert cake_jit.sm_wide_segments(arch, route, C) is None, (label, C)
        rows_programs = [
            name
            for kind in cake_jit.backward_kinds(arch)
            for name in cake_jit.route_modules(arch, kind)
            if (cake_jit.select_rule(cake_jit.MODULES[name]) or {}).get("kind") == "rows"
            and name not in BACKWARD_DIRECT_PROGRAMS.values()
        ]
        assert sorted(named) == sorted(rows_programs) == sorted(set(named)), arch


@pytest.mark.cpu_only
def test_sm_wide_programs_are_rows_records_of_their_det_twins() -> None:
    """Each SM-wide program (:data:`SM_WIDE_COPIES`) is a ``bwd`` record of one architecture with the selection rule
    ``{"kind": "rows"}``, listed after its route's other programs but the direct backward program, which comes last:
    the deterministic SM-wide program's build geometry,
    one CTA per SM, block, cluster and architecture; an ordinary occupancy launch with the per-call PDL scalar; the
    arguments and logical names of its route's fallback (the one-launch backward with ``dw_acc`` / ``work``); an
    ``accumulators`` workspace of :data:`SM_WIDE_SLOTS` copies of four FP32 taps.  No channel threshold or token
    floor admits it (``_ranked_candidates`` never returns it and it is no fallback), and the KCP registry has
    none.  The comment above each record names its deterministic twin."""

    assert len(SM_WIDE_COPIES) == 6 and set(SM_WIDE_COPIES.values()) <= DET_ONLY
    source = Path(cake_jit.__file__).read_text()
    by_arch: dict[str, list[str]] = {}
    for name, det_name in SM_WIDE_COPIES.items():
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        (arch,) = det["arches"]
        by_arch.setdefault(arch, []).append(name)
        label = (arch, name)
        at = source.index(f'    "{name}": {{\n')
        assert f"``{det_name}``" in source[source.rindex("\n    },\n", 0, at) : at], label
        assert record["arches"] == [arch] and record["role"] == "bwd", label
        assert record["library"] == record["kernel"] == name and record["kernel_symbol"] == f"kernel_{name}", label
        assert cake_jit.select_rule(record) == {"kind": "rows"}, label
        assert det_jit.select_rule(det)["kind"] == "channels_tokens_at_least", label
        grid, det_grid = record["grid_rule"], det["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det_grid["geometry"], label
        assert grid["geometry"] == {"threads": 384 if arch != "sm_107a" else 256, "vec": 8}, label
        assert grid["ctas_per_sm"] == det_grid["ctas_per_sm"] == 1, label
        assert grid["grid_mult"] == {arch: 1}, label
        launch = record["launch"]
        assert launch["block"] == det["launch"]["block"] == (grid["geometry"]["threads"], 1, 1), label
        assert launch["cluster"] == det["launch"]["cluster"] and launch["cooperative"] is False, label
        assert launch["use_pdl"] == "runtime", label
        fallback = cake_jit.MODULES[_route_fallback(arch, "bwd")]
        assert record["arg_plan"] == fallback["arg_plan"] and record["logical"] == fallback["logical"], label
        assert record["ffi_entry"] == fallback["ffi_entry"], label
        assert record["workspace_rule"] == {
            "acc": "float32",
            "slots": SM_WIDE_SLOTS[name],
            "kind": "accumulators",
            "taps": 4,
            "work": "uint32",
        }, label
        assert cake_jit.record_route(record, arch) == "bwd", label
        for C in (*CHANNELS, *range(520, 1024, 8)):
            candidates, chosen_fallback = cake_jit._ranked_candidates(arch, "bwd", C)
            assert name not in candidates and name != chosen_fallback, (label, C)
            for S in (None, 1, cake_jit.INT32_MAX):
                assert cake_jit._admits(cake_jit.select_rule(record), arch=arch, C=C, S=S) is False, (label, C, S)
    for arch, names in by_arch.items():
        # The route's direct backward program comes last, after the SM-wide programs.
        route = cake_jit.route_modules(arch, "bwd")
        assert route[-1] == BACKWARD_DIRECT_PROGRAMS["bwd"], arch
        assert list(route[-len(names) - 1 : -1]) == names, arch
        assert not set(names) & {n for k in cake_jit.KINDS if k != "bwd" for n in cake_jit.ROUTES[arch].get(k, ())}
    assert set(by_arch) == set(ARCHES)
    for name, record in kcp_jit.MODULES.items():
        assert (kcp_jit.select_rule(record) or {}).get("kind") != "rows", name
        assert name not in SM_WIDE_COPIES, name


@pytest.mark.cpu_only
def test_rows_rule_kind() -> None:
    """A selection rule of kind ``rows`` has no channel threshold or token floor (``_threshold`` and ``_tokens`` are
    ``None``, ``_admits`` is ``False`` for every call); an unknown kind is still a ``ValueError`` that lists it."""

    assert cake_jit.ROWS_KIND == "rows" and cake_jit.ROWS_KIND not in cake_jit.SELECT_KINDS
    rule = {"kind": "rows"}
    for arch in ARCHES:
        assert cake_jit._threshold(rule, arch=arch) is None
        assert cake_jit._tokens(rule, arch=arch) is None
        assert cake_jit._admits(rule, arch=arch, C=576, S=None) is False
    with pytest.raises(ValueError, match="rows"):
        cake_jit._threshold({"kind": "rowz"}, arch="sm_100a")


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_sm_wide_module_refuses_registry_errors(arch, monkeypatch) -> None:
    """``sm_wide_module`` names the segment's program inside a segment and ``None`` outside every segment; ``S`` of
    ``None`` under a key is a ``ValueError``; a segment naming a program of another route, or a record whose rule is
    not of kind ``rows``, is a ``RuntimeError``."""

    ((route, c_lo, c_hi), segments), *_ = SM_WIDE_ROWS_OF_RECORD[arch].items()
    for s_lo, s_hi, name in segments:
        assert cake_jit.sm_wide_module(arch, route, c_lo, s_lo) == name
        assert cake_jit.sm_wide_module(arch, route, c_hi, s_hi) == name
    assert cake_jit.sm_wide_module(arch, route, c_lo, segments[0][0] - 1) is None
    assert cake_jit.sm_wide_module(arch, route, c_hi + 8, segments[0][0]) is None
    assert cake_jit.sm_wide_module(arch, "bwd_dt", c_lo, segments[0][0]) is None
    with pytest.raises(ValueError, match="needs S"):
        cake_jit.sm_wide_module(arch, route, c_lo, None)
    other = _route_fallback(arch, "bwd_dt")
    monkeypatch.setitem(
        cake_jit.SM_WIDE_ROWS, arch, {(route, c_lo, c_hi): ((segments[0][0], cake_jit.INT32_MAX, other),)}
    )
    with pytest.raises(RuntimeError, match="rows"):
        cake_jit.sm_wide_module(arch, route, c_lo, segments[0][0])
    fallback = _route_fallback(arch, route)
    monkeypatch.setitem(
        cake_jit.SM_WIDE_ROWS, arch, {(route, c_lo, c_hi): ((segments[0][0], cake_jit.INT32_MAX, fallback),)}
    )
    with pytest.raises(RuntimeError, match="rows"):
        cake_jit.selected_modules(arch, route, C=c_lo, S=segments[0][0])


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_sides_are_the_selection_edges(arch) -> None:
    """``row_sides`` cuts the token axis of every forward and backward route and ``C`` at the bounds of the route's
    ``SM_WIDE_ROWS`` segments, ``PAIR_ROWS`` range, ``DIRECT_ROWS`` range (forward or backward) and ``FALLBACK_ROWS``
    range covering ``C`` (from 0, the last side to ``sys.maxsize``), and the selection holds one program on every
    side outside the rules' token floors: the program at both ends of a side is the one at its first token count."""

    edge_channels, _ = _forward_edges(arch)
    for kind in (*cake_jit.forward_kinds(arch), *cake_jit.backward_kinds(arch)):
        for C in sorted(
            {*CHANNELS, 520, 528, 600, 608, 616, 672, 680, 688, *edge_channels}
        ):
            cuts = {0}
            for (route, c_lo, c_hi), segments in SM_WIDE_ROWS_OF_RECORD.get(arch, {}).items():
                if route == kind and c_lo <= C <= c_hi and C % 8 == 0:
                    cuts |= {bound for s_lo, s_hi, _ in segments for bound in (s_lo, s_hi + 1)}
            for table in (PAIR_ROWS_OF_RECORD, _direct_rows_of_record()):
                for (route, c_lo, c_hi), (s_lo, s_hi, _) in table.get(arch, {}).items():
                    if route == kind and c_lo <= C <= c_hi and C % 8 == 0:
                        cuts |= {s_lo, s_hi + 1}
            for (route, c_lo, c_hi), (s_min, s_max) in FALLBACK_ROWS_OF_RECORD.get(arch, {}).items():
                if route == kind and c_lo <= C <= c_hi and C % 8 == 0:
                    cuts |= {s_min, s_max + 1}
            ordered = sorted(cuts)
            expected = tuple(zip(ordered, [*(cut - 1 for cut in ordered[1:]), sys.maxsize]))
            sides = cake_jit.row_sides(arch, kind, C)
            assert sides == expected, (kind, C, sides)
            if cake_jit.tokens_select(arch, kind, C):
                continue
            for s_lo, s_hi in sides:
                ends = {s_lo, min(s_hi, cake_jit.INT32_MAX + 1)}
                picks = {cake_jit.selected_modules(arch, kind, C=C, S=S) for S in ends}
                assert len(picks) == 1, (kind, C, s_lo, s_hi, picks)


# ----------------------------------------------------------------------------------------------------------------
# The 32-vector channel tiles of the narrow single-tile backward rows (``NARROW_TILES``).

# A table that exercises the narrow-row plan on every architecture whatever the rendered table holds: the sm_100a
# rows of the package's 0.2.0 release (route ``bwd`` only, from 31292 to 500680 tokens), plus rows on the other
# architectures (one of them spans a ``TILE_OVERRIDES`` row of its route, C = 768 of route ``bwd_dt`` on sm_103a,
# which keeps its override).
SYNTHETIC_NARROW_TILES = {
    "sm_100a": {
        ("bwd", 224, 224): (31292, 500680),
        ("bwd", 520, 624): (31292, 500680),
        ("bwd", 632, 760): (31292, 500680),
    },
    "sm_103a": {
        ("bwd_dt", 632, 768): (62585, 250340),
        ("bwd_bias", 224, 224): (31292, 125170),
    },
    "sm_107a": {("bwd_bias", 520, 760): (31292, 62585)},
}
NARROW_TABLES = ("rendered", "synthetic")


@pytest.fixture
def narrow_table(request, monkeypatch):
    """The ``NARROW_TILES`` table of a test: the rendered one, or :data:`SYNTHETIC_NARROW_TILES` in its place."""

    if request.param == "synthetic":
        monkeypatch.setattr(cake_jit, "NARROW_TILES", SYNTHETIC_NARROW_TILES)
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    yield cake_jit.NARROW_TILES
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _covering(table: dict, arch: str, route: str, C: int):
    """The ``(S_min, S_max)`` of the first key of ``table[arch]`` on ``route`` covering ``C`` (``None``: none, or
    a ``TILE_OVERRIDES`` row of ``(route, C)`` whose token range meets the key's)."""

    if C % 8:
        return None
    entry = cake_jit.TILE_OVERRIDES.get(arch, {}).get((route, C))
    for (name, c_lo, c_hi), bounds in table.get(arch, {}).items():
        if name == route and c_lo <= C <= c_hi:
            if entry is not None and bounds[0] <= entry[2] and entry[1] <= bounds[1]:
                return None
            return bounds
    return None


def _split3(plan: dict) -> tuple[int, int, int]:
    return plan["ct_vec"], plan["n_sub"], plan["n_ch_tiles"]


@pytest.mark.cpu_only
def test_narrow_tiles_are_backward_rows() -> None:
    """Every key of the rendered table names an exported architecture, one of its backward routes, a closed range of
    channel counts (multiples of 8) and a closed ``S`` range; no key of an architecture and route overlaps another,
    and no ``(route, C)`` of the architecture's ``TILE_OVERRIDES`` lies inside a key but a key of that one channel
    count whose tokens end below the row's ``S_min``."""

    assert set(cake_jit.NARROW_TILES) <= set(ARCHES)
    for arch, rows in cake_jit.NARROW_TILES.items():
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.backward_kinds(arch), label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert 0 < s_min <= s_max, label
            for name in cake_jit.route_modules(arch, route):
                geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
                assert geometry["threads"] % 32 == 0, (label, name)
            overrides = cake_jit.TILE_OVERRIDES.get(arch, {})
            for (override_route, C), (_, row_min, _) in overrides.items():
                if override_route == route and c_lo <= C <= c_hi:
                    assert c_lo == c_hi and s_max < row_min, label
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in zip(ranges, ranges[1:], strict=False):
                assert hi < lo, (arch, route, ranges)


def _check_narrow_plan(table: dict, arch: str, route: str, C: int, S: int) -> int:
    """One ``(route, C, S)`` call of ``arch`` against ``table``: the narrow-row reading on every architecture, and
    ``plan`` of every program of the route with and without the architecture / route, under ``legacy`` and at
    another thread count (inside a backward direct key the direct tiles at 128 threads only).  Returns the number of
    programs checked."""

    for other in ARCHES:
        bounds = _covering(table, other, route, C)
        inside = bounds is not None and bounds[0] <= S <= bounds[1]
        assert cake_jit.narrow_tile_range(other, route, C) == bounds, (other, route, C)
        assert cake_jit.narrow_tile_row(other, route, C, S) == inside, (other, route, C)
    bounds = _covering(table, arch, route, C)
    inside = bounds is not None and bounds[0] <= S <= bounds[1]
    override = cake_jit.tile_override(arch, route, C, S)
    on_row = {"arch": arch, "route": route}
    for name in cake_jit.route_modules(arch, route):
        geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
        threads, vec = geometry["threads"], geometry["vec"]
        label = (arch, route, C, S, name)
        split = cake_jit.channel_split(C, geometry=geometry, role="bwd")
        if _backward_direct_cell(arch, route, C, S) is not None and threads == 128:
            # Inside a backward direct key the direct tiles, ahead of the override and the narrow tiles.
            want = _backward_direct_split(C)
        elif override is not None:
            want = (override, threads // override, -(-(C // vec) // override))
        elif inside and threads == 128:
            want = (32, threads // 32, -(-(C // vec) // 32))
        else:
            want = split
        got = cake_jit.plan(S, C, 3, geometry=geometry, role="bwd", **on_row)
        assert _split3(got) == want, label
        for kwargs in ({}, {"arch": arch}, {"route": route}):
            got = cake_jit.plan(S, C, 3, geometry=geometry, role="bwd", **kwargs)
            assert _split3(got) == split, (label, kwargs)
        got = cake_jit.plan(S, C, 3, geometry=geometry, legacy=True, **on_row)
        legacy = cake_jit.channel_split(C, geometry=geometry, legacy=True)
        assert _split3(got) == legacy, label
        wide = {**geometry, "threads": 256}
        got = cake_jit.plan(S, C, 3, geometry=wide, role="bwd", **on_row)
        if override is None:
            wide_split = cake_jit.channel_split(C, geometry=wide, role="bwd")
            assert _split3(got) == wide_split, label
        tiles = cake_jit.narrow_tiles(arch, route, C, geometry)
        narrow = bounds is not None and threads == 128
        assert tiles == (-(-(C // vec) // 32) if narrow else 0), label
    return len(cake_jit.route_modules(arch, route))


@pytest.mark.cpu_only
@pytest.mark.parametrize("narrow_table", NARROW_TABLES, indirect=True)
def test_narrow_tile_bounds_and_plan(narrow_table) -> None:
    """A key covers its channel counts from ``C_lo`` through ``C_hi`` and its token counts from ``S_min`` through
    ``S_max`` (both ends included; never a ``TILE_OVERRIDES`` row) on its architecture and route only.  ``plan`` of a
    backward record given its architecture and route then plans 32-vector tiles at 128 threads, ``(32, threads //
    32, ceil(C / vec / 32))``, but the direct tiles inside a backward direct key; :func:`cake_jit.channel_split`'s
    split (or the override's) everywhere else, at
    another thread count, under ``legacy`` and without the architecture or the route.  ``narrow_tiles`` (the
    workspace bound) holds the key's tile count whatever ``S``."""

    checked = 0
    for arch, rows in narrow_table.items():
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            channels = {c_lo - 8, c_lo, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
            for C in sorted(C for C in channels if C > 0):
                for S in (s_min - 1, s_min, s_max, s_max + 1):
                    checked += _check_narrow_plan(narrow_table, arch, route, C, S)
    assert checked or not narrow_table


@pytest.mark.cpu_only
@pytest.mark.parametrize("narrow_table", NARROW_TABLES, indirect=True)
def test_backward_workspace_covers_every_narrow_row(narrow_table) -> None:
    """The public workspace of a key's channel counts holds the layout of every program of the key's route (every
    route plans by its own rows) at the key's tile count (one ticket per tile), whatever ``S``, and is the largest
    layout over every backward program at the largest tile count it may launch with."""

    for arch, rows in narrow_table.items():
        owners = cake_jit.workspace_owners(arch)
        for (route, c_lo, c_hi), (s_min, _) in rows.items():
            for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                nbytes = cake_jit.backward_workspace_bytes(arch, C)
                layouts = [
                    cake_jit.accumulator_layout(
                        record, C, _owner_tiles(record, arch, C)
                    )
                    for record in owners
                ]
                assert nbytes == max(layout["total_bytes"] for layout in layouts), (
                    arch,
                    route,
                    C,
                )
                for name in cake_jit.route_modules(arch, route):
                    record = cake_jit.MODULES[name]
                    geometry = record["grid_rule"]["geometry"]
                    tiles = cake_jit.plan(
                        s_min,
                        C,
                        3,
                        geometry=geometry,
                        role="bwd",
                        arch=arch,
                        route=route,
                    )["n_ch_tiles"]
                    layout = cake_jit.accumulator_layout(record, C, tiles)
                    assert layout["total_bytes"] <= nbytes, (arch, route, C, name)


# ----------------------------------------------------------------------------------------------------------------
# The forward plan: the deterministic package's forward rule (route, ``S`` and architecture) over the same programs
# from the row's token floor (``forward_narrow_min_s``: ``FORWARD_NARROW_MIN_S`` or ``FORWARD_NARROW_ROW_MIN_S``);
# below it the single tile of the package's 0.2.0 release.

# The forward's narrow rows at 128 threads x 8 channels: C -> (ct_vec, n_sub, n_ch_tiles) on every forward route
# below the row's token floor (:func:`_fwd_row_floor`; in a plan without ``S`` or the architecture, and on an
# architecture without that floor), from that floor up to the width floor of ``FWD_FLOORS``, and from both floors on
# the call's architecture; a call without a route and the legacy rule keep one tile of C / 8 vectors at every ``S``.
# The rows below 72 vectors (C 520 to 568) run their width (or the 32-vector tiles, C 536 and 568) from their row
# floor (``FWD_ROW_FLOORS``), which is their width floor: they never run the 32-vector tiles below a width.
FWD_ONE_TILE_SPLITS = {
    520: (65, 1, 1),
    528: (66, 1, 1),
    536: (67, 1, 1),
    544: (68, 1, 1),
    552: (69, 1, 1),
    560: (70, 1, 1),
    568: (71, 1, 1),
    576: (72, 1, 1),
    640: (80, 1, 1),
    704: (88, 1, 1),
    712: (89, 1, 1),
    768: (96, 1, 1),
}
FWD_NARROW_SPLITS = {C: (32, 4, 3) for C in FWD_ONE_TILE_SPLITS}
FWD_WIDE_SPLITS = {
    520: (13, 9, 5),
    528: (11, 11, 6),
    536: (32, 4, 3),
    544: (17, 7, 4),
    552: (23, 5, 3),
    560: (14, 9, 5),
    568: (32, 4, 3),
    576: (18, 7, 4),
    640: (40, 3, 2),
    704: (32, 4, 3),
    712: (32, 4, 3),
    768: (32, 4, 3),
}
# The rows below this many vectors take ``FWD_ROW_FLOORS`` (``FORWARD_ROW_FLOOR_VECTORS``).
FWD_ROW_FLOOR_VECTORS = 72
# The token count from which the narrow rows from 72 vectors run narrower tiles than one tile, per architecture.
FWD_MIN_S = {"sm_100a": 20480, "sm_103a": 12288, "sm_107a": 16384}
# The token count from which each narrow row below 72 vectors (C 520 to 568) runs narrower tiles, per architecture:
# arch -> {C / 8: S}, the smallest measured token count from which no measured cell ran slower with them.
FWD_ROW_FLOORS = {
    "sm_100a": {65: 20480, 66: 24576, 67: 16384, 68: 20480, 69: 16384, 70: 24576, 71: 12288},
    "sm_103a": {65: 20480, 66: 24576, 67: 16384, 68: 20480, 69: 16384, 70: 24576, 71: 12288},
    "sm_107a": {65: 40960, 66: 62585, 67: 16384, 68: 24576, 69: 24576, 70: 24576, 71: 16384},
}
# The tap dtypes of the forward row tables.
FWD_TAPS = ("f32", "bf16")
# The rows from 72 vectors that take another token floor: arch -> ((C / 8 from which, S, forward routes, taps), ...).
# On sm_103a the ``fwd_bias`` rows of 89 to 96 vectors (C 712 to 768) take 20480 with FP32 taps and 12288 with BF16
# taps; the ``fwd`` and ``fwd_dt`` rows there, and C 704 (88 vectors) on every route, keep the architecture's 12288.
FWD_ROW_MIN_S = {"sm_103a": ((89, 20480, ("fwd_bias",), ("f32",)), (89, 12288, ("fwd_bias",), ("bf16",)))}
# The rows that keep the single tile in a token range at or above their floor: arch -> ((C / 8 from, C / 8 to,
# forward routes, taps, S_lo, S_hi), ...).  On sm_103a the BF16-taps ``fwd_bias`` rows of 89 to 96 vectors.
FWD_SINGLE_TILE_ROWS = {
    "sm_103a": (
        (89, 89, ("fwd_bias",), ("bf16",), 12961, 16383),
        (90, 95, ("fwd_bias",), ("bf16",), 12289, 16383),
        (96, 96, ("fwd_bias",), ("bf16",), 13313, 16383),
    )
}
# The token count from which each width row runs its width, per architecture: arch -> {C / 8: S}; a row below 72
# vectors at its row floor.
FWD_FLOORS = {
    "sm_100a": {65: 20480, 66: 24576, 68: 20480, 69: 16384, 70: 24576, 72: 32768, 80: 24576},
    "sm_103a": {65: 20480, 66: 24576, 68: 20480, 69: 16384, 70: 24576, 72: 32768, 80: 24576},
    "sm_107a": {65: 40960, 66: 62585, 68: 24576, 69: 24576, 70: 24576, 72: 40960, 80: 36864},
}
# The forward workspace's channel tiles of the narrow rows: the largest tile count of the three plans (C 576: 1, 3,
# then 4; C 528: 1, then 6), whatever the token floors.  C 520 to 568 hold more than one tile from 0.2.4 on.
FWD_NARROW_WORKSPACE_TILES = {
    520: 5,
    528: 6,
    536: 3,
    544: 4,
    552: 3,
    560: 5,
    568: 3,
    576: 4,
    640: 3,
    704: 3,
    712: 3,
    768: 3,
}


def _fwd_taps(route: str, bf16_taps: bool = False) -> str:
    """The taps of a forward call on ``route``: FP32 on ``fwd``, BF16 on ``fwd_dt``, the call's on ``fwd_bias``."""

    if route == "fwd_bias":
        return "bf16" if bf16_taps else "f32"
    return "bf16" if route == "fwd_dt" else "f32"


def _fwd_row_floor(arch: str, route: str, channel_vectors: int, bf16_taps: bool = False) -> int | None:
    """The token floor of the forward row of ``channel_vectors`` vectors on forward route ``route`` (and taps) on
    ``arch`` from the literal tables: below 72 vectors its ``FWD_ROW_FLOORS`` floor (``None`` without one); else the
    last ``FWD_ROW_MIN_S`` entry listing the route and taps that the row reaches, else ``FWD_MIN_S``; ``None`` on an
    architecture without a floor."""

    if channel_vectors < FWD_ROW_FLOOR_VECTORS:
        return FWD_ROW_FLOORS.get(arch, {}).get(channel_vectors)
    floor = FWD_MIN_S.get(arch)
    if floor is None:
        return None
    for cv, S, routes, taps in FWD_ROW_MIN_S.get(arch, ()):
        if channel_vectors >= cv and route in routes and _fwd_taps(route, bf16_taps) in taps:
            floor = S
    return floor


def _fwd_gap(arch: str, route: str, channel_vectors: int, bf16_taps: bool = False) -> tuple[int, int] | None:
    """The ``(S_lo, S_hi)`` single-tile range of the forward row from the literal ``FWD_SINGLE_TILE_ROWS``, else
    ``None``."""

    for cv_lo, cv_hi, routes, taps, s_lo, s_hi in FWD_SINGLE_TILE_ROWS.get(arch, ()):
        if cv_lo <= channel_vectors <= cv_hi and route in routes and _fwd_taps(route, bf16_taps) in taps:
            return s_lo, s_hi
    return None


def _fwd_taps_of(route: str) -> tuple[bool, ...]:
    """The ``bf16_taps`` values a forward call of ``route`` may pass: both on ``fwd_bias`` (which reads them), the
    route's own on ``fwd`` (FP32) and ``fwd_dt`` (BF16)."""

    return (False, True) if route == "fwd_bias" else (route == "fwd_dt",)


def _forward_tokens() -> tuple[int, ...]:
    """Token counts on both sides of every forward token floor and single-tile range of either package, of every
    token floor of the forward selection rules and of every bound of the forward keys of ``FALLBACK_ROWS`` and
    ``DIRECT_ROWS`` (:func:`_forward_edges`), plus a few others."""

    tokens = {1, 2, 1000, 2048, 4096, 8192, 16384, 250340, 262144, 500680, 1 << 20}
    for arch in ARCHES:
        tokens |= _forward_edges(arch)[1]
    for table in (
        cake_jit.FORWARD_WIDTH_FLOORS,
        det_jit.NARROW_WIDTH_FLOORS["fwd"],
        cake_jit.FORWARD_NARROW_ROW_FLOORS,
        det_jit.NARROW_ROW_FLOORS["fwd"],
    ):
        for floors in table.values():
            for S in floors.values():
                tokens |= {S - 1, S, S + 1}
    for S in cake_jit.FORWARD_NARROW_MIN_S.values():
        tokens |= {S - 1, S, S + 1}
    for rows in cake_jit.FORWARD_NARROW_ROW_MIN_S.values():
        for _, S, _, _ in rows:
            tokens |= {S - 1, S, S + 1}
    for rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.values():
        for *_, s_lo, s_hi in rows:
            tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    for registry in (cake_jit, det_jit):
        for record in registry.MODULES.values():
            if record["role"] != "fwd":
                continue
            rule = registry.select_rule(record)
            for S in (rule or {}).get("tokens", {}).values():
                tokens |= {int(S) - 1, int(S)}
    return tuple(sorted(S for S in tokens if S > 0))


def _direct_literal(
    arch: str, kind: str, C: int, S: int, vec: int = 8
) -> tuple[int, int, int] | None:
    """The literal split of a forward call of route ``kind`` on ``arch`` inside a range of
    :data:`PAIR_ROWS_OF_RECORD` (the route's :data:`PAIR_PLANS_OF_RECORD` form; a pair range runs ahead of the direct
    range it covers) or else of :data:`DIRECT_ROWS_OF_RECORD` (the route's :data:`DIRECT_PLANS_OF_RECORD` form):
    ``(ct_vec, n_sub)`` and ``ceil(C / vec / ct_vec)`` tiles at 128 threads x ``vec`` channels, the planned record's
    geometry (8 but for the pair direct program's 2); ``None`` outside every range."""

    for rows, plans in (
        (PAIR_ROWS_OF_RECORD, PAIR_PLANS_OF_RECORD),
        (DIRECT_ROWS_OF_RECORD, DIRECT_PLANS_OF_RECORD),
    ):
        for (route, c_lo, c_hi), (s_lo, s_hi, _) in rows.get(arch, {}).items():
            if route == kind and c_lo <= C <= c_hi and C % 8 == 0 and s_lo <= S <= s_hi:
                ct_vec, n_sub = plans[arch][kind]
                return ct_vec, n_sub, -(-(C // vec) // ct_vec)
    return None


@pytest.mark.cpu_only
def test_forward_narrow_rows_are_the_deterministic_forward_rows() -> None:
    """The forward's narrow tables are the deterministic package's forward rows: every forward route, the thread
    count, the widths, the width floors and the row floors of the rows below 72 vectors on every architecture.  The
    rows below 72 vectors are C 520 to 568, the same on every architecture; a width row among them takes its width
    from its row floor (its width floor is its row floor)."""

    forward = tuple(kind for kind in cake_jit.KINDS if cake_jit.route_role(kind) == "fwd")
    assert cake_jit.FORWARD_NARROW_ROUTES == det_jit.NARROW_TILE_ROUTES["fwd"] == forward
    assert cake_jit.FORWARD_NARROW_THREADS == det_jit.NARROW_TILE_THREADS == 128
    widths = {65: 13, 66: 11, 68: 17, 69: 23, 70: 14, 72: 18, 80: 40}
    assert cake_jit.FORWARD_NARROW_WIDTHS == det_jit.NARROW_TILE_WIDTHS["fwd"] == widths
    assert cake_jit.FORWARD_WIDTH_FLOORS == det_jit.NARROW_WIDTH_FLOORS["fwd"] == FWD_FLOORS
    assert cake_jit.FORWARD_NARROW_ROW_FLOORS == det_jit.NARROW_ROW_FLOORS["fwd"] == FWD_ROW_FLOORS
    assert cake_jit.FORWARD_ROW_FLOOR_VECTORS == FWD_ROW_FLOOR_VECTORS
    assert set(FWD_FLOORS) == set(FWD_ROW_FLOORS) == set(ARCHES)
    for cv, width in widths.items():
        assert cv % width == 0 and 0 < width <= cake_jit.FORWARD_NARROW_THREADS, (cv, width)
    for arch in ARCHES:
        rows = FWD_ROW_FLOORS[arch]
        assert sorted(rows) == list(range(65, FWD_ROW_FLOOR_VECTORS)), arch
        for cv, S in rows.items():
            assert type(S) is int and S > 0, (arch, cv, S)
            if cv in widths:
                assert FWD_FLOORS[arch][cv] == S, (arch, cv)
        assert {cv for cv in FWD_FLOORS[arch] if cv < FWD_ROW_FLOOR_VECTORS} == set(rows) & set(widths), arch


@pytest.mark.cpu_only
def test_forward_narrow_token_floors() -> None:
    """``FORWARD_NARROW_MIN_S``, ``FORWARD_NARROW_ROW_MIN_S`` and ``FORWARD_SINGLE_TILE_ROWS`` are the literal tables:
    one positive token floor per architecture of ``ARCHES``; the ``fwd_bias`` rows of sm_103a from 89 vectors (C 712
    to 768) at 20480 with FP32 taps and 12288 with BF16 taps (each row entry ``(cv, S, routes, taps)``: ``routes`` a
    non-empty tuple of forward routes in ``FORWARD_NARROW_ROUTES`` order, ``taps`` of ``FORWARD_TAPS`` in that
    order, no ``(cv, route, taps)`` twice), and the BF16-taps ``fwd_bias`` single-tile ranges there (each ``(cv_lo,
    cv_hi, routes, taps, S_lo, S_hi)``, no ``(cv, route, taps)`` in two).  ``forward_narrow_min_s`` gives every row
    of every forward route and taps its floor (:func:`_fwd_row_floor`: below 72 vectors ``FORWARD_NARROW_ROW_FLOORS``)
    and ``forward_single_tile_range`` its range (:func:`_fwd_gap`); both refuse a route outside
    ``FORWARD_NARROW_ROUTES``; below that floor and inside the range the narrow rows keep one tile and elsewhere from
    the floor they plan as the deterministic forward; ``channel_split`` reads only the row's floor and range on the
    call's route, taps and architecture, and an architecture without a floor keeps the single tile at every ``S``."""

    routes_all = cake_jit.FORWARD_NARROW_ROUTES
    assert cake_jit.FORWARD_TAPS == FWD_TAPS
    assert cake_jit.FORWARD_NARROW_MIN_S == FWD_MIN_S
    for arch, S in cake_jit.FORWARD_NARROW_MIN_S.items():
        assert arch in ARCHES and type(S) is int and S > 0, (arch, S)
    assert cake_jit.FORWARD_NARROW_ROW_MIN_S == FWD_ROW_MIN_S
    for arch, rows in cake_jit.FORWARD_NARROW_ROW_MIN_S.items():
        assert arch in cake_jit.FORWARD_NARROW_MIN_S and type(rows) is tuple and rows, (arch, rows)
        assert [cv for cv, *_ in rows] == sorted(cv for cv, *_ in rows), (arch, rows)
        keys = [(cv, route, taps) for cv, _, routes, row_taps in rows for route in routes for taps in row_taps]
        assert len(keys) == len(set(keys)), (arch, rows)
        for cv, S, routes, taps in rows:
            assert type(cv) is int and 72 <= cv <= 96 and type(S) is int and S > 0, (arch, cv, S)
            assert type(routes) is tuple and routes, (arch, cv, routes)
            assert routes == tuple(route for route in routes_all if route in routes), (arch, cv, routes)
            assert type(taps) is tuple and taps == tuple(t for t in FWD_TAPS if t in taps), (arch, cv, taps)
    assert cake_jit.FORWARD_SINGLE_TILE_ROWS == FWD_SINGLE_TILE_ROWS
    for arch, rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.items():
        assert arch in cake_jit.FORWARD_NARROW_MIN_S and type(rows) is tuple and rows, (arch, rows)
        keys = [
            (cv, route, taps)
            for cv_lo, cv_hi, routes, row_taps, _, _ in rows
            for cv in range(cv_lo, cv_hi + 1)
            for route in routes
            for taps in row_taps
        ]
        assert len(keys) == len(set(keys)), (arch, rows)
        for cv_lo, cv_hi, routes, taps, s_lo, s_hi in rows:
            label = (arch, cv_lo, cv_hi)
            assert type(cv_lo) is int and type(cv_hi) is int and 72 <= cv_lo <= cv_hi <= 96, label
            assert type(routes) is tuple and routes == tuple(r for r in routes_all if r in routes) and routes, label
            assert type(taps) is tuple and taps == tuple(t for t in FWD_TAPS if t in taps) and taps, label
            assert type(s_lo) is int and type(s_hi) is int and 0 < s_lo <= s_hi < cake_jit.INT32_MAX, label
            for cv in range(cv_lo, cv_hi + 1):
                for route in routes:
                    for bf16 in _fwd_taps_of(route):
                        if _fwd_taps(route, bf16) in taps:
                            # A range lies above the row's floor (else part of it would plan as below the floor).
                            assert _fwd_row_floor(arch, route, cv, bf16) < s_lo, (label, cv, route, bf16)
    for arch in ARCHES:
        for route in routes_all:
            for bf16 in (False, True):
                for cv in range(1, 1025):
                    label = (arch, route, bf16, cv)
                    got = cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16)
                    assert got == _fwd_row_floor(arch, route, cv, bf16), label
                    got = cake_jit.forward_single_tile_range(arch, cv, route=route, bf16_taps=bf16)
                    assert got == _fwd_gap(arch, route, cv, bf16), label
            assert cake_jit.forward_narrow_min_s(arch, 96, route=route) == _fwd_row_floor(arch, route, 96), arch
    for route in routes_all:
        assert cake_jit.forward_narrow_min_s(None, 96, route=route) is None
        assert cake_jit.forward_narrow_min_s("sm_90a", 96, route=route) is None
        assert cake_jit.forward_narrow_min_s("sm_90a", 65, route=route) is None
        assert cake_jit.forward_single_tile_range("sm_90a", 96, route=route, bf16_taps=True) is None
    for route in ("bwd", None, "fwd_pair"):
        with pytest.raises(ValueError, match="route must be one of"):
            cake_jit.forward_narrow_min_s("sm_103a", 96, route=route)
        with pytest.raises(ValueError, match="route must be one of"):
            cake_jit.forward_single_tile_range("sm_103a", 96, route=route)
    # The literal floors of the rows the plans below cross: on sm_103a C 704 at 12288 on every route, C 712 and 768
    # at 20480 on ``fwd_bias`` with FP32 taps and at 12288 with BF16 taps and on ``fwd`` / ``fwd_dt``; the rows of C
    # 520 to 568 at their row floors on every route and taps.
    floors = {
        (route, bf16): [
            cake_jit.forward_narrow_min_s("sm_103a", C // 8, route=route, bf16_taps=bf16)
            for C in (520, 568, 576, 640, 704, 712, 768)
        ]
        for route in routes_all
        for bf16 in (False, True)
    }
    plain = [20480, 12288, 12288, 12288, 12288, 12288, 12288]
    assert floors == {
        ("fwd", False): plain,
        ("fwd", True): plain,
        ("fwd_dt", False): plain,
        ("fwd_dt", True): plain,
        ("fwd_bias", False): [20480, 12288, 12288, 12288, 12288, 20480, 20480],
        ("fwd_bias", True): plain,
    }
    gaps = {
        (route, bf16): [
            cake_jit.forward_single_tile_range("sm_103a", C // 8, route=route, bf16_taps=bf16)
            for C in (704, 712, 720, 760, 768)
        ]
        for route in routes_all
        for bf16 in (False, True)
    }
    none5 = [None] * 5
    assert gaps == {
        ("fwd", False): none5,
        ("fwd", True): none5,
        ("fwd_dt", False): none5,
        ("fwd_dt", True): none5,
        ("fwd_bias", False): none5,
        ("fwd_bias", True): [None, (12961, 16383), (12289, 16383), (12289, 16383), (13313, 16383)],
    }
    for arch in ("sm_100a", "sm_107a"):
        for route in routes_all:
            for bf16 in (False, True):
                got = {cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16) for cv in range(72, 97)}
                assert got == {FWD_MIN_S[arch]}, (arch, route, bf16)
                got = {cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16) for cv in range(65, 72)}
                assert got == set(FWD_ROW_FLOORS[arch].values()), (arch, route, bf16)
    geometry = {"threads": 128, "vec": 8}
    for arch in ARCHES:
        for route in routes_all:
            for bf16 in (False, True):  # ``fwd`` and ``fwd_dt`` ignore it (their taps are the route's)
                for C, one_tile in FWD_ONE_TILE_SPLITS.items():
                    S_min = _fwd_row_floor(arch, route, C // 8, bf16)
                    gap = _fwd_gap(arch, route, C // 8, bf16)
                    steps = (12287, 12288, 12960, 12961, 13312, 13313, 14336, 16383, 16384, 20479, 20480, 24575)
                    edges = (
                        31291,
                        31292,
                        32766,
                        32767,
                        32768,
                        40959,
                        40960,
                        62584,
                        62585,
                    )
                    for S in (0, 1, 4096, 8192, *steps, 24576, *edges, 1 << 20):
                        got = cake_jit.channel_split(
                            C, geometry=geometry, role="fwd", route=route, S=S, arch=arch, bf16_taps=bf16
                        )
                        label = (arch, route, bf16, C, S)
                        if S_min is None or S < S_min or (gap is not None and gap[0] <= S <= gap[1]):
                            assert got == one_tile, label
                        else:
                            assert got == det_jit.channel_split(
                                C, geometry=geometry, role="fwd", route=route, S=S, arch=arch
                            ), label
                            assert got in (FWD_NARROW_SPLITS[C], FWD_WIDE_SPLITS[C]), label
                            if C // 8 < FWD_ROW_FLOOR_VECTORS:
                                assert got == FWD_WIDE_SPLITS[C], label
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FORWARD_NARROW_MIN_S", {})
        m.setattr(cake_jit, "FORWARD_NARROW_ROW_FLOORS", {})
        for arch in ARCHES:
            for C, one_tile in FWD_ONE_TILE_SPLITS.items():
                for S in (1, 1 << 20):
                    got = cake_jit.channel_split(C, geometry=geometry, role="fwd", route="fwd", S=S, arch=arch)
                    assert got == one_tile, (arch, C, S)
    # Without the row table every row from 72 vectors takes its architecture's floor: sm_103a's ``fwd_bias`` C 712 and
    # 768 from 12288 with either taps; without the single-tile table the BF16-taps rows narrow inside the ranges too.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FORWARD_NARROW_ROW_MIN_S", {})
        m.setattr(cake_jit, "FORWARD_SINGLE_TILE_ROWS", {})
        for C in (712, 768):
            for bf16 in (False, True):
                for S, want in (
                    (12287, FWD_ONE_TILE_SPLITS[C]),
                    (12288, FWD_NARROW_SPLITS[C]),
                    (14336, FWD_NARROW_SPLITS[C]),
                ):
                    got = cake_jit.channel_split(
                        C, geometry=geometry, role="fwd", route="fwd_bias", S=S, arch="sm_103a", bf16_taps=bf16
                    )
                    assert got == want, (C, bf16, S)
    # The single-tile table alone decides the BF16-taps ``fwd_bias`` rows inside the ranges (the floors are 12288).
    for C, inside, outside in ((712, (12961, 16383), (12960, 16384)), (768, (13313, 16383), (13312, 16384))):
        for S in inside:
            got = cake_jit.channel_split(
                C, geometry=geometry, role="fwd", route="fwd_bias", S=S, arch="sm_103a", bf16_taps=True
            )
            assert got == FWD_ONE_TILE_SPLITS[C], (C, S)
        for S in outside:
            got = cake_jit.channel_split(
                C, geometry=geometry, role="fwd", route="fwd_bias", S=S, arch="sm_103a", bf16_taps=True
            )
            assert got == FWD_NARROW_SPLITS[C], (C, S)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_forward_plan_is_the_deterministic_forward_plan(arch) -> None:
    """Every forward record plans what the deterministic record it copies plans from the row's floor on ``arch``
    (``forward_narrow_min_s``) outside the row's single-tile range (``forward_single_tile_range``), and below the floor
    and inside the range (and on an architecture without a floor) the route-less plan of the 0.2.0 release, for every
    multiple of 8 up to 8192 channels, every token count of :func:`_forward_tokens` and both tap dtypes on
    ``fwd_bias``: ``plan`` given the record's architecture, route and taps (as ``_launch`` and the KCP forward's
    single-rank main program call it), ``channel_split`` with and without the route, ``S`` and the architecture, and
    the legacy rule.  At 128 x 8 the narrow rows run one tile of C / 8 vectors below the floor, the 32-vector tiles
    from it and their width from the width floor of ``arch`` (the literal tables; the rows below 72 vectors their
    width from the row floor); without a route, without ``S`` or the architecture and under ``legacy`` one tile of C /
    8 vectors stays; both packages refuse a route of the other role and an unknown architecture.  Inside a range of
    ``PAIR_ROWS`` (ahead of the direct range it covers) or of ``DIRECT_ROWS`` every forward record of the route plans
    the deterministic forward's pair or direct tiles at its geometry's ``vec`` (the literal :func:`_direct_literal`),
    whatever the floors and the taps; the rows of ``FALLBACK_ROWS`` keep the plan above."""

    tokens = _forward_tokens()
    literal = direct = pair = 0
    for pair_arch, kind, name, det_name in _pairs("fwd"):
        if pair_arch != arch:
            continue
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        geometry, det_geometry = record["grid_rule"]["geometry"], det["grid_rule"]["geometry"]
        route = cake_jit.record_route(record, arch)
        assert route == kind == det_jit.record_route(det, arch), (name, det_name)
        for C in range(8, 8193, 8):
            label = (kind, name, C)
            S_min = cake_jit.forward_narrow_min_s(arch, C // int(geometry["vec"]), route=route)
            # The route-less plan, the 0.2.0 plan of every row: that of each call below the floor.
            route_less = det_jit.channel_split(C, geometry=det_geometry, role="fwd")
            for kwargs in (
                {},
                {"route": route},
                {"route": route, "arch": arch},
                {"route": route, "S": tokens[-1]},
                {"arch": arch, "S": tokens[-1]},
            ):
                got = cake_jit.channel_split(C, geometry=geometry, role="fwd", **kwargs)
                assert got == route_less, (label, kwargs)
                if "route" not in kwargs:
                    want = det_jit.channel_split(C, geometry=det_geometry, role="fwd", **kwargs)
                    assert got == want, (label, kwargs)
            kwargs = {"route": route, "arch": arch, "S": tokens[-1]}
            got = cake_jit.channel_split(C, geometry=geometry, role="fwd", **kwargs)
            want = det_jit.channel_split(C, geometry=det_geometry, role="fwd", **kwargs)
            assert got == (route_less if S_min is None else want), (label, kwargs)
            legacy = det_jit.channel_split(C, geometry=det_geometry, legacy=True)
            assert cake_jit.channel_split(C, geometry=geometry, legacy=True) == legacy, label
            got = cake_jit.plan(
                tokens[-1], C, 3, geometry=geometry, role="fwd", legacy=True, arch=arch, route=route
            )
            assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == legacy, label
            for bf16 in _fwd_taps_of(kind):
                cv = C // int(geometry["vec"])
                S_min = cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16)
                gap = cake_jit.forward_single_tile_range(arch, cv, route=route, bf16_taps=bf16)
                for S in tokens:
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role=record["role"], arch=arch, route=route, bf16_taps=bf16
                    )
                    in_pair = cake_jit.pair_row(arch, route, C, S)
                    if in_pair or cake_jit.direct_row(arch, route, C, S):
                        want = det_jit.plan(
                            S,
                            C,
                            3,
                            geometry=det_geometry,
                            role=det["role"],
                            route=route,
                            arch=arch,
                        )
                        vec = int(geometry["vec"])
                        assert _split3(want) == _direct_literal(
                            arch, kind, C, S, vec
                        ), (
                            label,
                            bf16,
                            S,
                        )
                        pair += bool(in_pair)
                        direct += not in_pair
                    elif (
                        S_min is not None
                        and S >= S_min
                        and not (gap is not None and gap[0] <= S <= gap[1])
                    ):
                        want = det_jit.plan(
                            S, C, 3, geometry=det_geometry, role=det["role"], route=route, arch=arch
                        )
                    else:
                        want = det_jit.plan(S, C, 3, geometry=det_geometry, role=det["role"])
                        assert (want["ct_vec"], want["n_sub"], want["n_ch_tiles"]) == route_less
                    assert got == want, (label, bf16, S)
        if (geometry["threads"], geometry["vec"]) != (128, 8):
            continue
        for C, narrow in FWD_NARROW_SPLITS.items():
            one_tile = FWD_ONE_TILE_SPLITS[C]
            assert one_tile == (C // 8, 128 // (C // 8), 1)
            assert cake_jit.channel_split(C, geometry=geometry, role="fwd") == one_tile
            assert cake_jit.channel_split(C, geometry=geometry, legacy=True) == one_tile
            assert cake_jit.channel_split(C, geometry=geometry, role="fwd", route=route) == one_tile
            floor = FWD_FLOORS[arch].get(C // 8)
            for bf16 in _fwd_taps_of(kind):
                S_lo = _fwd_row_floor(arch, kind, C // 8, bf16)
                gap = _fwd_gap(arch, kind, C // 8, bf16)
                steps = [(1, one_tile), (S_lo - 1, one_tile), (1 << 20, FWD_WIDE_SPLITS[C])]
                steps += [(S_lo, narrow if floor is None or S_lo < floor else FWD_WIDE_SPLITS[C])]
                if floor is not None:
                    steps += [(floor - 1, narrow if floor - 1 >= S_lo else one_tile)]
                    steps += [(floor, FWD_WIDE_SPLITS[C])]
                # Both sides of the architecture's floor and of every row floor (sm_103a: C 704 from 12288; C 712 and
                # 768 from 20480 on ``fwd_bias`` with FP32 taps, the single tile at 12288 to 20479, and from 12288
                # with BF16 taps and on ``fwd`` / ``fwd_dt``; C 520 to 568 from their row floors).
                for S in sorted({FWD_MIN_S[arch] - 1, FWD_MIN_S[arch], 16384, 20479, 20480, 24575, 24576}):
                    if S < S_lo:
                        steps += [(S, one_tile)]
                    elif floor is None or S < floor:
                        steps += [(S, narrow)]
                # Both ends of a single-tile range and the token counts around them (sm_103a BF16-taps ``fwd_bias``).
                if gap is not None:
                    steps = [(S, want) for S, want in steps if not gap[0] <= S <= gap[1]]
                    steps += [(gap[0], one_tile), (gap[1], one_tile), (gap[0] - 1, narrow), (gap[1] + 1, narrow)]
                # Inside a range of ``PAIR_ROWS`` or ``DIRECT_ROWS`` (both of its bounds among them) the pair or the
                # direct tiles at the record's 8-vectors.
                for rows in (PAIR_ROWS_OF_RECORD, DIRECT_ROWS_OF_RECORD):
                    for (key_route, c_lo, c_hi), (s_lo, s_hi, _) in rows.get(
                        arch, {}
                    ).items():
                        if key_route == kind and c_lo <= C <= c_hi:
                            steps += [(s_lo, None), (s_hi, None)]
                for S, want in steps:
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role="fwd", arch=arch, route=route, bf16_taps=bf16
                    )
                    want = _direct_literal(arch, kind, C, S) or want
                    assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, (kind, C, bf16, S)
                    literal += 1
    assert literal, arch
    assert bool(direct) is bool(DIRECT_ROWS_OF_RECORD.get(arch)), arch
    assert bool(pair) is bool(PAIR_ROWS_OF_RECORD.get(arch)), arch
    geometry = {"threads": 128, "vec": 8}
    for registry in (cake_jit, det_jit):
        for kwargs in (
            {"role": "fwd", "route": "bwd"},
            {"role": "bwd", "route": "fwd_dt"},
            {"legacy": True, "route": "fwd"},
            {"role": "fwd", "route": "fwd", "S": 1 << 20, "arch": "sm_90a"},
        ):
            with pytest.raises(ValueError):
                registry.channel_split(576, geometry=geometry, **kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_forward_workspace_is_the_deterministic_forward_workspace(arch) -> None:
    """``forward_workspace_words`` is the deterministic package's for every multiple of 8 up to 8192 channels (C 520
    to 568 included, whose rows hold 3 to 6 tiles from their row floors): two int32 counters per channel tile, the
    most over every forward program and every plan of a row whose tiles change at a token floor
    (``FWD_NARROW_WORKSPACE_TILES``: C 576 holds 4 tiles, C 528 6), whatever ``FORWARD_NARROW_MIN_S``,
    ``FORWARD_NARROW_ROW_MIN_S``, ``FORWARD_NARROW_ROW_FLOORS`` and ``FORWARD_SINGLE_TILE_ROWS`` hold (the sizing
    takes no ``S`` and no taps); every forward program's plan at every token count of :func:`_forward_tokens` and
    both tap dtypes fits it, so a buffer of the public size never trips the ``work`` size check (a dynamic program's
    outside the ``PAIR_ROWS`` and ``DIRECT_ROWS`` ranges: inside one, only the pair or the direct program runs, on
    its own channel tiles).  The static, direct and pair direct programs take no counters (``workspace_rule`` None, 0
    words at every plan) and add nothing to the size: it is the size over the dynamic programs alone."""

    tokens = _forward_tokens()
    records = [
        (kind, cake_jit.MODULES[name])
        for kind in cake_jit.forward_kinds(arch)
        for name in cake_jit.route_modules(arch, kind)
    ]
    assert records
    no_work = {
        name for kind in cake_jit.forward_kinds(arch) for form in ("static", "direct", "pair")
        if (name := _forward_program(arch, kind, form)) is not None
    }  # fmt: skip
    for _, record in records:
        if record["kernel"] in no_work:
            assert record["workspace_rule"] is None, record["kernel"]
            assert (
                cake_jit.counter_words(record, 1)
                == cake_jit.counter_words(record, 96)
                == 0
            ), record["kernel"]
        else:
            assert record["workspace_rule"] == {
                "kind": "tile_counters",
                "words_per_tile": 2,
            }, record["kernel"]
    dynamic_only = {
        kind: [
            name for name in cake_jit.route_modules(arch, kind) if name not in no_work
        ]
        for kind in cake_jit.forward_kinds(arch)
    }
    with pytest.MonkeyPatch.context() as m:
        m.setitem(cake_jit.ROUTES, arch, {**cake_jit.ROUTES[arch], **dynamic_only})
        sized = {
            C: cake_jit.forward_workspace_words(arch, C) for C in range(8, 8193, 8)
        }
    assert sized == {C: cake_jit.forward_workspace_words(arch, C) for C in sized}, arch
    for C in range(8, 8193, 8):
        words = cake_jit.forward_workspace_words(arch, C)
        assert words == det_jit.forward_workspace_words(arch, C), C
        for kind, record in records:
            for bf16 in _fwd_taps_of(kind):
                for S in tokens:
                    if record["kernel"] not in no_work and (
                        cake_jit.pair_row(arch, kind, C, S)
                        or cake_jit.direct_row(arch, kind, C, S)
                    ):
                        continue
                    tiles = cake_jit.plan(
                        S,
                        C,
                        1,
                        geometry=record["grid_rule"]["geometry"],
                        role="fwd",
                        arch=arch,
                        route=kind,
                        bf16_taps=bf16,
                    )["n_ch_tiles"]
                    assert cake_jit.counter_words(record, tiles) <= words, (kind, C, bf16, S)
    for C, tiles in FWD_NARROW_WORKSPACE_TILES.items():
        assert cake_jit.forward_workspace_words(arch, C) == 2 * tiles, C
    # The size does not read the token floors or ranges: none, or every narrow row's from one token, sizes the same
    # buffer.
    every = {cv: 1 for cv in range(65, FWD_ROW_FLOOR_VECTORS)}
    gaps = {arch: ((72, 96, cake_jit.FORWARD_NARROW_ROUTES, FWD_TAPS, 1, 1 << 20),)}
    for table, row_floors in (({}, {}), ({arch: 1}, {arch: every})):
        for rows in ({}, {arch: ((72, 1 << 20, cake_jit.FORWARD_NARROW_ROUTES, FWD_TAPS),)}):
            for single in ({}, gaps):
                with pytest.MonkeyPatch.context() as m:
                    m.setattr(cake_jit, "FORWARD_NARROW_MIN_S", table)
                    m.setattr(cake_jit, "FORWARD_NARROW_ROW_FLOORS", row_floors)
                    m.setattr(cake_jit, "FORWARD_NARROW_ROW_MIN_S", rows)
                    m.setattr(cake_jit, "FORWARD_SINGLE_TILE_ROWS", single)
                    for C, tiles in FWD_NARROW_WORKSPACE_TILES.items():
                        assert cake_jit.forward_workspace_words(arch, C) == 2 * tiles, (table, rows, single, C)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_forward_main_program_plans_the_single_rank_forward(arch, fake_device) -> None:
    """The KCP two-launch forward's single-rank main program (``kcp._sr_main``) runs the single-rank ``fwd`` route
    record ``selected_modules`` picks with that record's forward plan (its role, route and architecture: the narrow
    rows' single tile below the row's ``forward_narrow_min_s`` floor, their 32-vector tiles from it, the width from
    the width floor) and its counter words, which the forward workspace rule covers; the KCP forward program keeps the
    legacy plan, one tile of C / 8 vectors at C 576 to 768.  At the rank's local token count inside a forward key of
    ``FALLBACK_ROWS`` / ``DIRECT_ROWS`` / ``PAIR_ROWS`` it runs the static / direct / pair direct program with 0
    counter words, the direct one on its direct tiles and the pair direct one on its pair tiles of two channels per
    thread (a pair range runs ahead of the others)."""

    kcp = cake_package(KEY, "kcp")
    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    keyed = {
        C
        for route, c_lo, c_hi, *_ in _forward_rows(arch)
        if route == "fwd"
        for C in (c_lo, c_hi)
    }
    forms = set()
    for C in sorted(
        {
            512,
            520,
            528,
            536,
            544,
            552,
            560,
            568,
            576,
            640,
            704,
            712,
            768,
            776,
            1024,
            *keyed,
        }
    ):
        S_min = _fwd_row_floor(arch, "fwd", C // 8)
        for S in _forward_tokens():
            name, scalars, grid, words = kcp._sr_main(
                arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE
            )
            assert (name,) == cake_jit.selected_modules(arch, "fwd", C=C, S=S), (C, S)
            record = cake_jit.MODULES[name]
            want = cake_jit.plan(
                S, C, 3, geometry=record["grid_rule"]["geometry"], role="fwd", arch=arch, route="fwd"
            )
            assert scalars == want, (C, S)
            assert grid[0] >= scalars["n_ch_tiles"], (C, S)
            assert words == cake_jit.counter_words(record, want["n_ch_tiles"]), (C, S)
            assert words <= cake_jit.forward_workspace_words(arch, C), (C, S)
            for form in ("static", "direct", "pair"):
                if name == _forward_program(arch, "fwd", form):
                    assert words == 0, (C, S, form)
                    forms.add(form)
            split = (scalars["ct_vec"], scalars["n_sub"], scalars["n_ch_tiles"])
            if _in_pair_of_record(arch, "fwd", C, S):
                assert name == _forward_program(arch, "fwd", "pair"), (C, S)
                assert split == _direct_literal(arch, "fwd", C, S, FORM_VEC["pair"]), (
                    C,
                    S,
                )
            elif C in FWD_NARROW_SPLITS and (
                record["grid_rule"]["geometry"] == {"threads": 128, "vec": 8}
            ):
                floor = FWD_FLOORS[arch].get(C // 8)
                if _direct_literal(arch, "fwd", C, S) is not None:
                    assert name == _forward_program(arch, "fwd", "direct"), (C, S)
                    assert split == _direct_literal(arch, "fwd", C, S), (C, S)
                elif S_min is None or S < S_min:
                    assert split == FWD_ONE_TILE_SPLITS[C], (C, S)
                else:
                    wide = floor is not None and S >= floor
                    assert split == (FWD_WIDE_SPLITS[C] if wide else FWD_NARROW_SPLITS[C]), (C, S)
    assert forms == {form for route, *_, form in _forward_rows(arch) if route == "fwd"}, arch
    (name,) = kcp_jit.route_modules(arch, kcp.KCP_FWD)
    geometry = kcp_jit.MODULES[name]["grid_rule"]["geometry"]
    for C in FWD_NARROW_SPLITS:
        cv = C // geometry["vec"]
        got = cake_jit.plan(1 << 20, C, 3, geometry=geometry, legacy=True)
        threads = geometry["threads"]
        assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == (
            min(cv, threads),
            threads // min(cv, threads),
            -(-cv // min(cv, threads)),
        ), C


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_forward_plan_ignores_taps(arch, fake_device) -> None:
    """The KCP forward's single-rank main program plans on the ``fwd`` route (FP32 taps by the KCP contract), so the
    tap-keyed rows of ``FORWARD_NARROW_ROW_MIN_S`` and ``FORWARD_SINGLE_TILE_ROWS`` (``fwd_bias`` only) never change
    its plan: with both tables emptied every ``(C, S)`` plans the same."""

    kcp = cake_package(KEY, "kcp")
    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    channels = (520, 568, 576, 704, 712, 736, 760, 768)
    planned = {
        (C, S): kcp._sr_main(arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE)
        for C in channels
        for S in _forward_tokens()
    }
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FORWARD_NARROW_ROW_MIN_S", {})
        m.setattr(cake_jit, "FORWARD_SINGLE_TILE_ROWS", {})
        for (C, S), want in planned.items():
            assert kcp._sr_main(arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE) == want, (C, S)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_direct_rows_plan_as_the_deterministic_forward(arch, fake_device) -> None:
    """At both channel ends and both token ends of every range of :data:`DIRECT_ROWS_OF_RECORD` (an end inside a
    :data:`PAIR_ROWS_OF_RECORD` range aside: the pair range runs ahead of it) and of :data:`PAIR_ROWS_OF_RECORD`, of
    both tap dtypes, every forward record of the route at 128 threads plans the deterministic package's plan of the
    same row (its ``plan`` given the route and the architecture), which is the literal direct or pair tiles at the
    record's ``vec`` (:func:`_direct_literal`) whatever the token floors; so do ``_plan_split``, the row plan
    (``plan_row``) and a per-call plan (``_route`` + ``_launch``) of the call, which launches the route's direct or
    pair direct program on its own tiles.  One token outside the range the plan is the one without the range's table
    (``DIRECT_ROWS`` or ``PAIR_ROWS``)."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    checked = {"direct": 0, "pair": 0}
    geometries = ({"threads": 128, "vec": 8}, {"threads": 128, "vec": 2})
    for form, table, rows in (
        ("direct", "DIRECT_ROWS", DIRECT_ROWS_OF_RECORD),
        ("pair", "PAIR_ROWS", PAIR_ROWS_OF_RECORD),
    ):
        for (route, c_lo, c_hi), (s_lo, s_hi, name) in rows.get(arch, {}).items():
            for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                for S in (s_lo, s_hi):
                    if form == "direct" and _in_pair_of_record(arch, route, C, S):
                        continue
                    want = _direct_literal(arch, route, C, S, FORM_VEC[form])
                    assert want is not None, (route, C, S)
                    for bf16 in _fwd_taps_of(route):
                        label = (form, route, C, S, bf16)
                        for program in cake_jit.route_modules(arch, route):
                            geometry = cake_jit.MODULES[program]["grid_rule"][
                                "geometry"
                            ]
                            if geometry not in geometries:
                                continue
                            vec = int(geometry["vec"])
                            got = cake_jit.plan(
                                S,
                                C,
                                3,
                                geometry=geometry,
                                role="fwd",
                                arch=arch,
                                route=route,
                                bf16_taps=bf16,
                            )
                            det = det_jit.plan(
                                S,
                                C,
                                3,
                                geometry=geometry,
                                role="fwd",
                                route=route,
                                arch=arch,
                            )
                            literal = _direct_literal(arch, route, C, S, vec)
                            assert got == det and _split3(got) == literal, (
                                label,
                                program,
                            )
                            split = cake_jit._plan_split(
                                S,
                                C,
                                geometry=geometry,
                                role="fwd",
                                arch=arch,
                                route=route,
                                bf16_taps=bf16,
                            )
                            assert split == literal, (label, program)
                        per_call = _per_call(arch, route, S, C, 3, 1, None, bf16)
                        assert per_call == _row_planned(
                            arch, route, S, C, 3, 1, None, bf16
                        ), label
                        assert per_call[0] == (name,), label
                        assert _split3(per_call[1]) == want, label
                        checked[form] += 1
                for S in (s_lo - 1, s_hi + 1):
                    if S < 1:
                        continue
                    for geometry in geometries:
                        got = cake_jit.plan(
                            S,
                            C,
                            3,
                            geometry=geometry,
                            role="fwd",
                            arch=arch,
                            route=route,
                        )
                        with pytest.MonkeyPatch.context() as m:
                            m.setattr(cake_jit, table, {})
                            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                            assert got == cake_jit.plan(
                                S,
                                C,
                                3,
                                geometry=geometry,
                                role="fwd",
                                arch=arch,
                                route=route,
                            ), (form, route, C, S, geometry)
                        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    assert bool(checked["direct"]) is bool(DIRECT_ROWS_OF_RECORD.get(arch)), arch
    assert bool(checked["pair"]) is bool(PAIR_ROWS_OF_RECORD.get(arch)), arch


@pytest.mark.cpu_only
def test_conformance_cells_inside_direct_ranges(fake_device) -> None:
    """Every cell of :data:`SM103_DIRECT_PLAN_CELLS` (distinct, in route order) lies inside a ``DIRECT_ROWS`` range
    of its route on sm_103a: the call selects the route's direct program and plans its direct tiles for the cell's
    taps, per call and from the row cache.  The cells are measured forward cells at C 704 to 768 (multiples of 8)."""

    arch = "sm_103a"
    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    cells = SM103_DIRECT_PLAN_CELLS
    assert len(set(cells)) == len(cells)
    assert list(cells) == sorted(
        cells, key=lambda c: (FORWARD_ROUTES.index(c[0]), *c[1:])
    )
    for route, taps, bias, C, S in cells:
        label = (route, taps, bias, C, S)
        assert (
            route in FORWARD_ROUTES
            and taps in FWD_TAPS
            and bias in ("none", "bf16", "f32")
        ), label
        assert (
            (bias == "none") is (route != "fwd_bias") and 704 <= C <= 768 and C % 8 == 0
        ), label
        assert cake_jit.direct_row(arch, route, C, S), label
        bf16 = route == "fwd_bias" and taps == "bf16"
        want = _per_call(arch, route, S, C, 1, 1, None, bf16)
        assert want == _row_planned(arch, route, S, C, 1, 1, None, bf16), label
        assert want[0] == (_forward_program(arch, route, "direct"),), label
        assert _split3(want[1]) == _direct_literal(arch, route, C, S), label


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_static_forward_keeps_the_measured_plan(arch, fake_device) -> None:
    """Inside every forward range of :data:`FALLBACK_ROWS_OF_RECORD` (both channel ends, both token ends, both tap
    dtypes; an end inside a :data:`PAIR_ROWS_OF_RECORD` range aside, where the pair direct program runs ahead of it)
    the route's static program plans the forward plan of the package without ``PAIR_ROWS`` and ``DIRECT_ROWS`` (the
    route's floors and single-tile ranges of :func:`channel_split`), the plan its dynamic program had there; the row
    plan and a per-call plan launch the static program on it."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    checked = 0
    for route, c_lo, c_hi, s_lo, s_hi, form in _forward_rows(arch):
        if form != "static":
            continue
        static = _forward_program(arch, route, "static")
        geometry = cake_jit.MODULES[static]["grid_rule"]["geometry"]
        for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
            for S in (s_lo, s_hi):
                if _in_pair_of_record(arch, route, C, S):
                    continue
                for bf16 in _fwd_taps_of(route):
                    label = (route, C, S, bf16)
                    want = cake_jit.channel_split(
                        C,
                        geometry=geometry,
                        role="fwd",
                        route=route,
                        S=S,
                        arch=arch,
                        bf16_taps=bf16,
                    )
                    got = cake_jit.plan(
                        S,
                        C,
                        3,
                        geometry=geometry,
                        role="fwd",
                        arch=arch,
                        route=route,
                        bf16_taps=bf16,
                    )
                    assert _split3(got) == want, label
                    with pytest.MonkeyPatch.context() as m:
                        m.setattr(cake_jit, "PAIR_ROWS", {})
                        m.setattr(cake_jit, "DIRECT_ROWS", {})
                        assert cake_jit.plan(
                            S, C, 3, geometry=geometry, role="fwd", arch=arch, route=route, bf16_taps=bf16
                        ) == got, label  # fmt: skip
                    per_call = _per_call(arch, route, S, C, 3, 1, None, bf16)
                    assert per_call == _row_planned(
                        arch, route, S, C, 3, 1, None, bf16
                    ), label
                    assert per_call[0] == (static,) and _split3(per_call[1]) == want, (
                        label
                    )
                    checked += 1
    assert checked, arch


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_direct_grid_is_capped_at_its_token_blocks(
    arch, fake_device, monkeypatch
) -> None:
    """A direct or pair direct program launches ``min(ctas_per_sm x SMs, n_ch_tiles x ceil(S / (8 x n_sub)))`` CTAs
    (``cap`` ``token_blocks``; ``ctas_per_sm`` the route's :data:`DIRECT_CTAS_PER_SM` / :data:`PAIR_CTAS_PER_SM`) on
    its own tiles (:func:`_direct_literal` at its ``vec``), as the deterministic package's launch of the same record
    on the same device does, at both channel and token ends of every direct and pair range and every SM count of
    :data:`PLAN_SWEEP_SMS`; without ``C`` and ``S`` it refuses to size the grid (``ValueError``), and a cap on an
    ``occupancy`` rule or an unknown cap is a ``ValueError``.  A static program launches ``ctas_per_sm x SMs`` CTAs
    whatever ``C`` and ``S``."""

    assert cake_jit.GRID_CAPS == ("token_blocks",) and cake_jit.DIRECT_BLOCK_TOKENS == 8
    for sms in PLAN_SWEEP_SMS[arch]:
        fake_device(arch, sms)
        monkeypatch.setattr(det_jit, "device_arch", lambda device_index: arch)
        monkeypatch.setattr(det_jit, "sm_count", lambda device_index, sms=sms: sms)
        for form, rows, ctas_per_sm in (
            ("direct", DIRECT_ROWS_OF_RECORD, DIRECT_CTAS_PER_SM),
            ("pair", PAIR_ROWS_OF_RECORD, PAIR_CTAS_PER_SM),
        ):
            for (route, c_lo, c_hi), (s_lo, s_hi, name) in rows.get(arch, {}).items():
                record = cake_jit.MODULES[name]
                assert record["grid_rule"]["ctas_per_sm"] == ctas_per_sm[route], name
                det = det_jit.MODULES[
                    name.replace(
                        "cake_causal_conv1d_train_", "cake_causal_conv1d_det_train_"
                    )
                ]
                for C in (c_lo, c_hi):
                    for S in sorted({s_lo, min(s_lo + 7, s_hi), s_hi}):
                        _, n_sub, tiles = _direct_literal(
                            arch, route, C, S, FORM_VEC[form]
                        )
                        want = min(
                            ctas_per_sm[route] * sms, tiles * -(-S // (8 * n_sub))
                        )
                        got = cake_jit.launch_grid(
                            record, device_index=FAKE_DEVICE, C=C, S=S
                        )
                        assert got == (want, 1, 1), (sms, form, route, C, S)
                        assert (
                            det_jit.launch_grid(det, device_index=FAKE_DEVICE, C=C, S=S)
                            == got
                        ), (sms, form, route, C, S)
                for kwargs in ({}, {"C": c_lo}, {"S": s_lo}):
                    with pytest.raises(ValueError, match="needs C and S"):
                        cake_jit.launch_grid(record, device_index=FAKE_DEVICE, **kwargs)
                for grid_rule in (
                    {**record["grid_rule"], "cap": "tokens"},
                    {
                        **record["grid_rule"],
                        "kind": "occupancy",
                        "grid_mult": {arch: 1},
                    },
                ):
                    with pytest.raises(ValueError, match="cap"):
                        cake_jit.launch_grid(
                            {**record, "grid_rule": grid_rule},
                            device_index=FAKE_DEVICE,
                            C=c_lo,
                            S=s_lo,
                        )
        for kind in cake_jit.forward_kinds(arch):
            record = cake_jit.MODULES[_forward_program(arch, kind, "static")]
            for C, S in ((576, 1), (768, 1 << 20), (None, None)):
                got = cake_jit.launch_grid(record, device_index=FAKE_DEVICE, C=C, S=S)
                assert got == (record["grid_rule"]["ctas_per_sm"] * sms, 1, 1), (
                    sms,
                    kind,
                    C,
                    S,
                )


# ----------------------------------------------------------------------------------------------------------------
# The host plan of a call row (``_ROW_PLANS``): planned by the row's first call, the same values as a per-call plan.

# The SM counts of the fake devices of the sweep, per architecture (B200 / GB300 / R200 boards first: the counts the
# grid-trim rules were measured at).  The default sweep takes the first; the full sweep every one.
PLAN_SWEEP_SMS = {"sm_100a": (148, 160), "sm_103a": (152, 148), "sm_107a": (212, 216)}
# A device index no real device has: the rows the sweep plans never serve a launch.
FAKE_DEVICE = 1 << 20
# CAKE_CONV1D_FULL_PLAN_SWEEP=1 runs the full sweep: every SM count, five document counts and three token orders.
FULL_PLAN_SWEEP = os.environ.get("CAKE_CONV1D_FULL_PLAN_SWEEP") == "1"


def _plan_sweep_channels() -> tuple[int, ...]:
    """``CHANNELS``, rows off the 32-channel grid, and every selection threshold and ``TILE_OVERRIDES`` channel count
    of the registry with the multiple of 8 below it, both ends of every ``NARROW_TILES`` key with their neighbours
    outside it, both ends of every ``FALLBACK_ROWS``, ``PAIR_ROWS``, ``DIRECT_ROWS``, ``SM_WIDE_ROWS`` and
    ``GRID_CAP_ROWS`` key with their neighbours outside it, and
    the forward's narrow rows (C 520 to 768, two of them off the 64-channel grid, C 712, the first row of sm_103a's
    ``FORWARD_NARROW_ROW_MIN_S`` floors of the ``fwd_bias`` route, and every row of ``FORWARD_NARROW_ROW_FLOORS`` and
    ``FORWARD_SINGLE_TILE_ROWS``) with their neighbours."""

    channels = {*CHANNELS, 16, 24, 248, 520, 640, 1048, 1152, 3072}
    channels |= {512, 568, 600, *FWD_NARROW_SPLITS, 720, 736, 760, 776}
    for rows in cake_jit.FORWARD_NARROW_ROW_FLOORS.values():
        channels |= {8 * cv for cv in rows}
    for rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.values():
        for cv_lo, cv_hi, *_ in rows:
            channels |= {8 * cv_lo - 8, 8 * cv_lo, 8 * cv_hi, 8 * cv_hi + 8}
    for record in cake_jit.MODULES.values():
        rule = cake_jit.select_rule(record)
        for C in (rule or {}).get("channels", {}).values():
            channels |= {int(C) - 8, int(C)}
    for rows in cake_jit.TILE_OVERRIDES.values():
        for _, C in rows:
            channels |= {C - 8, C, C + 8}
    for table in (
        cake_jit.NARROW_TILES,
        cake_jit.FALLBACK_ROWS,
        cake_jit.PAIR_ROWS,
        cake_jit.DIRECT_ROWS,
        cake_jit.SM_WIDE_ROWS,
        cake_jit.GRID_CAP_ROWS,
    ):
        for rows in table.values():
            for _, c_lo, c_hi in rows:
                channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
    return tuple(sorted(C for C in channels if C > 0))


def _plan_sweep_tokens() -> tuple[int, ...]:
    """Token counts on both sides of every bound where ``S`` enters a plan: the ``TILE_OVERRIDES``, ``NARROW_TILES``,
    ``FALLBACK_ROWS``, ``PAIR_ROWS`` and ``DIRECT_ROWS`` ranges, the ``SM_WIDE_ROWS`` segments and the
    ``GRID_CAP_ROWS`` spans, the forward's
    ``FORWARD_NARROW_MIN_S``, ``FORWARD_NARROW_ROW_MIN_S``, ``FORWARD_NARROW_ROW_FLOORS`` and ``FORWARD_WIDTH_FLOORS``
    floors and ``FORWARD_SINGLE_TILE_ROWS`` ranges, the grid-trim minimums and the selection rules' token floors, plus
    a few others."""

    tokens = {1, 2, 63, 64, 1000, 250340, 1 << 20}
    for table in (cake_jit.FORWARD_WIDTH_FLOORS, cake_jit.FORWARD_NARROW_ROW_FLOORS):
        for floors in table.values():
            for S in floors.values():
                tokens |= {S - 1, S}
    for S in cake_jit.FORWARD_NARROW_MIN_S.values():
        tokens |= {S - 1, S}
    for rows in cake_jit.FORWARD_NARROW_ROW_MIN_S.values():
        for _, S, _, _ in rows:
            tokens |= {S - 1, S}
    for rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.values():
        for *_, s_lo, s_hi in rows:
            tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    for table in (cake_jit.TILE_OVERRIDES, cake_jit.NARROW_TILES, cake_jit.FALLBACK_ROWS):
        for rows in table.values():
            for value in rows.values():
                s_min, s_max = value[-2:]
                tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    for table in (cake_jit.SM_WIDE_ROWS, cake_jit.GRID_CAP_ROWS):
        for rows in table.values():
            for segments in rows.values():
                for s_lo, s_hi, _ in segments:
                    tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    for table in (cake_jit.PAIR_ROWS, cake_jit.DIRECT_ROWS):
        for rows in table.values():
            for s_lo, s_hi, _ in rows.values():
                tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    tokens |= {cake_jit.TILE_ALIGNED_MIN_S - 1, cake_jit.TILE_ALIGNED_MIN_S}
    for classes in cake_jit.TILE_ALIGNED_CLASSES.values():
        for s_min, _ in classes.values():
            tokens |= {s_min - 1, s_min}
    for record in cake_jit.MODULES.values():
        rule = cake_jit.select_rule(record)
        for S in (rule or {}).get("tokens", {}).values():
            tokens |= {int(S) - 1, int(S)}
    return tuple(sorted(S for S in tokens if S > 0))


@pytest.fixture
def fake_device(monkeypatch):
    """Plan on :data:`FAKE_DEVICE` of a given architecture and SM count: ``use(arch, sms)`` points ``cake_jit``'s
    device queries at it and empties ``_ROW_PLANS``; the rows planned are dropped again after the test."""

    def use(arch: str, sms: int) -> None:
        monkeypatch.setattr(cake_jit, "device_arch", lambda device_index: arch)
        monkeypatch.setattr(cake_jit, "sm_count", lambda device_index: sms)
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()

    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    yield use
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _per_call(arch: str, kind: str, S: int, C: int, num_docs: int, gm: int, policy, bf16_taps: bool = False):
    """What a call plans without ``_ROW_PLANS`` (``_route`` then ``_launch``), or the error it raises."""

    try:
        names = pkg._route(arch, kind, needs=pkg._route_needs(policy), C=C, S=S)
        scalars, grid = pkg._launch(
            cake_jit.MODULES[names[0]],
            S,
            C,
            num_docs,
            device_index=FAKE_DEVICE,
            arch=arch,
            bwd_grid_mult=gm,
            bf16_taps=bf16_taps,
        )
        return tuple(names), pkg._scalars(scalars), tuple(grid)
    except (ValueError, RuntimeError) as exc:
        return type(exc).__name__, str(exc)


def _row_planned(arch: str, kind: str, S: int, C: int, num_docs: int, gm: int, policy, bf16_taps: bool = False):
    """What a call plans from its ``_ROW_PLANS`` row (``_call_plan``), or the error it raises."""

    try:
        names, record, scalars, grid, _ = pkg._call_plan(
            arch,
            kind,
            S,
            C,
            num_docs,
            device_index=FAKE_DEVICE,
            policy=policy,
            bwd_grid_mult=gm,
            bf16_taps=bf16_taps,
        )
        assert record is cake_jit.MODULES[names[0]]
        assert tuple(scalars) == cake_jit.PLAN_SCALARS
        assert all(type(value) is int for value in scalars.values())
        return tuple(names), dict(scalars), tuple(grid)
    except (ValueError, RuntimeError) as exc:
        return type(exc).__name__, str(exc)


def _sweep_row_plans(arch: str) -> int:
    """Every route of ``arch`` x the sweep's channel counts, token counts (in each order of the sweep, the row
    planned by the order's first call), document counts, backward grid multiples and (on ``fwd_bias``) both tap
    dtypes: the row plan equals the per-call plan, errors included; an SM-wide program plans its one channel tile of
    every channel vector, ``threads // cv`` token sub-tiles (5 at 384 threads, 3 at 256).  Returns the number of
    calls checked."""

    channels, tokens = _plan_sweep_channels(), _plan_sweep_tokens()
    orders = [tokens]
    if FULL_PLAN_SWEEP:
        half = len(tokens) // 2
        orders += [tokens[::-1], tokens[half:] + tokens[:half]]
    docs = (1, 2, 4, 5, 1000) if FULL_PLAN_SWEEP else (1, 1000)
    checked = 0
    for order in orders:
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
        for kind in cake_jit.ROUTES[arch]:
            backward = cake_jit.route_role(kind) == "bwd"
            policy = "fp32" if kind.endswith("fp32g") else "bf16" if backward else None
            for gm in pkg.BWD_GRID_MULTS if backward else (1,):
                for bf16 in (False,) if backward else _fwd_taps_of(kind):
                    for C in channels:
                        for S in order:
                            for num_docs in docs:
                                want = _per_call(arch, kind, S, C, num_docs, gm, policy, bf16)
                                got = _row_planned(arch, kind, S, C, num_docs, gm, policy, bf16)
                                assert got == want, (arch, kind, C, S, num_docs, gm, bf16)
                                if isinstance(want[0], tuple) and want[0][0] in SM_WIDE_COPIES:
                                    threads = cake_jit.MODULES[want[0][0]]["grid_rule"]["geometry"]["threads"]
                                    split = (want[1]["ct_vec"], want[1]["n_sub"], want[1]["n_ch_tiles"])
                                    assert split == (C // 8, threads // (C // 8), 1), (arch, kind, C, S, split)
                                    assert split[1] == (5 if threads == 384 else 3), (arch, kind, C, S, split)
                                checked += 1
        assert len(pkg._ROW_PLANS) <= pkg._ROW_PLANS_MAX
    return checked


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_match_the_per_call_plan(arch, fake_device, record_property) -> None:
    """A call planned from its ``_ROW_PLANS`` row launches the programs, plan scalars and grid a per-call plan
    (``_route`` + ``_launch``) gives, and raises the same errors, on fake devices of the architecture: every route,
    channel counts at every selection threshold, the forward's narrow rows and ``C % 32`` rows (whose grid
    ``launch_grid`` may trim by ``S``), token counts at every ``TILE_OVERRIDES`` / ``NARROW_TILES`` bound, forward
    token floor and width floor, grid-trim minimum and token floor, document counts and every ``bwd_grid_mult``.  ``CAKE_CONV1D_FULL_PLAN_SWEEP=1`` runs the full sweep."""

    checked = 0
    for sms in PLAN_SWEEP_SMS[arch][: None if FULL_PLAN_SWEEP else 1]:
        fake_device(arch, sms)
        checked += _sweep_row_plans(arch)
    record_property("row_plan_calls_checked", checked)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_override_table(
    arch, fake_device, monkeypatch
) -> None:
    """A row planned before ``TILE_OVERRIDES`` changes keeps its plan until
    :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties the cache; after it, with the rows of every architecture on ``arch``, every row's calls on both
    sides of its token range plan as a per-call plan does, the override's tiles inside the range (the direct tiles
    inside a backward direct key), on the row's route (every route plans by its own rows)."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    shipped = dict(cake_jit.TILE_OVERRIDES.get(arch, {}))
    rows = {
        key: value
        for table in cake_jit.TILE_OVERRIDES.values()
        for key, value in table.items()
    }

    def policy_of(route: str) -> str:
        return "fp32" if route.endswith("fp32g") else "bf16"

    planned = {
        (route, C): _row_planned(arch, route, value[1], C, 3, 1, policy_of(route))
        for (route, C), value in rows.items()
        if route in cake_jit.ROUTES[arch]
    }
    monkeypatch.setitem(cake_jit.TILE_OVERRIDES, arch, rows)
    changed = 0
    for (route, C), before in planned.items():
        _, s_min, _ = rows[(route, C)]
        now = _per_call(arch, route, s_min, C, 3, 1, policy_of(route))
        assert _row_planned(arch, route, s_min, C, 3, 1, policy_of(route)) == before
        changed += now != before
    assert changed or rows.items() <= shipped.items(), arch
    assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
    checked = 0
    for (route, C), (ct_vec, s_min, s_max) in rows.items():
        if route not in cake_jit.ROUTES[arch]:
            continue
        for S in (s_min - 1, s_min, s_max, s_max + 1):
            for gm in pkg.BWD_GRID_MULTS:
                want = _per_call(arch, route, S, C, 3, gm, policy_of(route))
                got = _row_planned(arch, route, S, C, 3, gm, policy_of(route))
                assert got == want, (route, C, S, gm)
                if _backward_direct_cell(arch, route, C, S) is not None and isinstance(
                    got[1], dict
                ):
                    # A backward direct key runs its direct tiles ahead of the override.
                    assert _split3(got[1]) == BACKWARD_DIRECT_SPLITS[C], (
                        route,
                        C,
                        S,
                        gm,
                    )
                elif s_min <= S <= s_max and isinstance(got[1], dict):
                    assert got[1]["ct_vec"] == ct_vec, (route, C, S, gm)
                checked += 1
    assert checked


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_narrow_table(
    arch, fake_device, monkeypatch
) -> None:
    """A row planned before ``NARROW_TILES`` changes keeps its plan until
    :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties the cache; after it, with :data:`SYNTHETIC_NARROW_TILES` in place, every key's calls at its channel ends and an
    inner channel count, on both sides of its token range and at every ``bwd_grid_mult``, plan as a per-call plan
    does (the 32-vector tiles inside the range, the side-stream and whole-rank grid trims included), on fake
    devices of the architecture's SM counts, on the key's route (every route plans by its own rows).  At a
    row of ``SM_WIDE_ROWS`` the SM-wide program runs its own split (one channel tile, ``threads // cv`` token
    sub-tiles): the narrow tiles are a split of 128-thread programs."""

    rows = SYNTHETIC_NARROW_TILES[arch]

    def policy_of(route: str) -> str:
        return "fp32" if route.endswith("fp32g") else "bf16"

    def planned(route: str, S: int, C: int, gm: int = 1):
        return _row_planned(arch, route, S, C, 3, gm, policy_of(route))

    checked = 0
    for sms in PLAN_SWEEP_SMS[arch]:
        fake_device(arch, sms)
        before = {
            (route, s_min, c_lo): planned(route, s_min, c_lo)
            for (route, c_lo, _), (s_min, _) in rows.items()
        }
        with monkeypatch.context() as m:
            m.setattr(cake_jit, "NARROW_TILES", SYNTHETIC_NARROW_TILES)
            for (route, S, C), plan_before in before.items():
                assert planned(route, S, C) == plan_before, (sms, route, S, C)
            assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
            for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
                for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                    for S in (s_min - 1, s_min, s_max, s_max + 1):
                        ct_vec = cake_jit.tile_override(arch, route, C, S) or 32
                        for gm in pkg.BWD_GRID_MULTS:
                            want = _per_call(
                                arch, route, S, C, 3, gm, policy_of(route)
                            )
                            got = planned(route, S, C, gm)
                            assert got == want, (arch, sms, route, C, S, gm)
                            if (
                                isinstance(got[0], tuple)
                                and got[0][0] in SM_WIDE_COPIES
                            ):
                                # an SM_WIDE_ROWS row: its SM-wide program plans its own one-tile split
                                record = cake_jit.MODULES[got[0][0]]
                                threads = record["grid_rule"]["geometry"]["threads"]
                                split = (
                                    got[1]["ct_vec"],
                                    got[1]["n_sub"],
                                    got[1]["n_ch_tiles"],
                                )
                                one_tile = (C // 8, threads // (C // 8), 1)
                                assert split == one_tile, (route, C, S, gm, split)
                            elif s_min <= S <= s_max and isinstance(got[1], dict):
                                assert got[1]["ct_vec"] == ct_vec, (route, C, S, gm)
                            checked += 1
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    assert checked


# Forward token floors the replaced-floor test plans with on every architecture: one token, below, at and between the
# width floors of ``FWD_FLOORS`` and above all of them.
SYNTHETIC_FORWARD_MIN_S = (1, 8192, 24576, 32768, 36864, 65536)
# A row table the replaced-floor test plans with besides the architecture's own: on ``fwd_dt`` and ``fwd_bias`` the
# rows from 80 vectors (C 640 to 768) of either taps from a floor at the lower width floor, and on ``fwd`` and
# ``fwd_bias`` the FP32-taps rows from 89 vectors (C 712 to 768) from a floor between the width floors (the last
# entry of a route and taps wins: FP32-taps ``fwd_bias`` C 712 to 768 from 28672).
SYNTHETIC_FORWARD_ROW_MIN_S = (
    (80, 24576, ("fwd_dt", "fwd_bias"), ("f32", "bf16")),
    (89, 28672, ("fwd", "fwd_bias"), ("f32",)),
)
# A single-tile table the replaced-floor test plans with besides the architecture's own: BF16-taps ``fwd_bias`` and
# ``fwd_dt`` rows of 72 to 89 vectors (C 576 to 712) keep the single tile from 30000 to 40000 tokens.
SYNTHETIC_FORWARD_SINGLE_TILE_ROWS = ((72, 89, ("fwd_dt", "fwd_bias"), ("bf16",), 30000, 40000),)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_forward_floor(arch, fake_device, monkeypatch) -> None:
    """A forward row planned before ``FORWARD_NARROW_MIN_S``, ``FORWARD_NARROW_ROW_MIN_S`` or
    ``FORWARD_SINGLE_TILE_ROWS`` changes keeps its plan until :func:`cake_causal_conv1d_nondet_train_plan_cache_clear`
    empties the cache (a row whose program selection reads ``S`` holds no plan and plans every call); after it, with
    each floor of :data:`SYNTHETIC_FORWARD_MIN_S` on ``arch`` and either ``arch``'s row and single-tile tables or the
    synthetic ones, every forward route's calls of the narrow rows and their neighbours, of both tap dtypes, on both
    sides of the row's floor (``forward_narrow_min_s``: the floor, the row table's, or below 72 vectors the row
    floor), of the width floors, of the single-tile ranges and of the forward keys of ``FALLBACK_ROWS``,
    ``PAIR_ROWS`` and ``DIRECT_ROWS`` plan as a per-call plan does: inside a pair or direct range the pair or direct
    tiles of the selected program's ``vec`` whatever the floors, else
    the deterministic forward's tiles from the row's floor outside its range, the route-less single tile below it and
    inside it.  The row cache keeps the two tap dtypes apart."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    channels = (512, 520, 528, 568, 576, 600, 640, 704, 712, 768, 776)
    tables = (
        (cake_jit.FORWARD_NARROW_ROW_MIN_S.get(arch, ()), cake_jit.FORWARD_SINGLE_TILE_ROWS.get(arch, ())),
        (SYNTHETIC_FORWARD_ROW_MIN_S, SYNTHETIC_FORWARD_SINGLE_TILE_ROWS),
    )
    checked = 0
    for S_min in SYNTHETIC_FORWARD_MIN_S:
        for rows, gaps in tables:
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
            before = {
                (kind, C, bf16): _row_planned(arch, kind, S_min, C, 3, 1, None, bf16)
                for kind in cake_jit.forward_kinds(arch)
                for C in channels
                for bf16 in _fwd_taps_of(kind)
            }
            with monkeypatch.context() as m:
                m.setitem(cake_jit.FORWARD_NARROW_MIN_S, arch, S_min)
                m.setitem(cake_jit.FORWARD_NARROW_ROW_MIN_S, arch, rows)
                m.setitem(cake_jit.FORWARD_SINGLE_TILE_ROWS, arch, gaps)
                for (kind, C, bf16), plan_before in before.items():
                    sides = pkg._ROW_PLANS.get((arch, kind, C, FAKE_DEVICE, 1, bf16))
                    if sides is None:  # the call raised before planning a row
                        assert isinstance(plan_before[1], str), (kind, C, plan_before)
                        continue
                    (row,) = [row for s_lo, s_hi, row, _ in sides if s_lo <= S_min <= s_hi]
                    if row is None:  # the selection reads ``S``: every call is planned
                        assert cake_jit.tokens_select(arch, kind, C), (kind, C)
                        continue
                    assert _row_planned(arch, kind, S_min, C, 3, 1, None, bf16) == plan_before, (kind, C, bf16)
                assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
                bounds = {S_min, *FWD_FLOORS[arch].values(), *FWD_ROW_FLOORS[arch].values(), *(S for _, S, *_ in rows)}
                bounds |= {S for *_, s_lo, s_hi in gaps for S in (s_lo, s_hi + 1)}
                bounds |= {S for _, _, _, s_lo, s_hi, _ in _forward_rows(arch) for S in (s_lo, s_hi + 1)}
                tokens = sorted({1, 1 << 20, *(S + d for S in bounds for d in (-1, 0, 1))} - {0})
                for kind in cake_jit.forward_kinds(arch):
                    for bf16 in _fwd_taps_of(kind):
                        taps = _fwd_taps(kind, bf16)
                        for C in channels:
                            for S in tokens:
                                want = _per_call(arch, kind, S, C, 3, 1, None, bf16)
                                got = _row_planned(arch, kind, S, C, 3, 1, None, bf16)
                                label = (S_min, rows, gaps, kind, bf16, C, S)
                                assert got == want, label
                                assert isinstance(got[1], dict), (label, got)
                                geometry = cake_jit.MODULES[got[0][0]]["grid_rule"]["geometry"]
                                cv = C // int(geometry["vec"])
                                if cv < FWD_ROW_FLOOR_VECTORS:
                                    row_floor = FWD_ROW_FLOORS[arch].get(cv)
                                else:
                                    row_floor = S_min
                                    for cv_from, S_row, row_routes, row_taps in rows:
                                        if cv >= cv_from and kind in row_routes and taps in row_taps:
                                            row_floor = S_row
                                assert cake_jit.forward_narrow_min_s(arch, cv, route=kind, bf16_taps=bf16) == (
                                    row_floor
                                ), label
                                gap = next(
                                    (
                                        (s_lo, s_hi)
                                        for cv_lo, cv_hi, row_routes, row_taps, s_lo, s_hi in gaps
                                        if cv_lo <= cv <= cv_hi and kind in row_routes and taps in row_taps
                                    ),
                                    None,
                                )
                                inside = gap is not None and gap[0] <= S <= gap[1]
                                vec = int(geometry["vec"])
                                if _direct_literal(arch, kind, C, S, vec) is not None:
                                    split = _direct_literal(arch, kind, C, S, vec)
                                elif (
                                    row_floor is not None
                                    and S >= row_floor
                                    and not inside
                                ):
                                    split = det_jit.channel_split(
                                        C, geometry=geometry, role="fwd", route=kind, S=S, arch=arch
                                    )
                                else:
                                    split = det_jit.channel_split(C, geometry=geometry, role="fwd")
                                assert _split3(got[1]) == split, label
                                checked += 1
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    assert checked


# ----------------------------------------------------------------------------------------------------------------
# The direct backward rows (the backward keys of ``DIRECT_ROWS``): one ordinary launch per call without TMA or a staged
# pipeline, ahead of every other backward table, on the tiles of ``BACKWARD_DIRECT_PLANS``.

# The grid rule of every direct backward record: a co-resident grid of 2 CTAs per SM of 128 threads x 8-vectors,
# capped at one CTA per token block of ``n_sub x 4`` tokens; no ``grid_mult`` (the call's ``bwd_grid_mult`` multiplies
# the capped grid).
BACKWARD_DIRECT_GRID_RULE = {
    "block_tokens": 4,
    "cap": "token_blocks",
    "cooperative": False,
    "ctas_per_sm": 2,
    "geometry": {"threads": 128, "vec": 8},
    "kind": "co_resident",
}
# The channel counts the edge tests plan at besides both sides of every channel bound of the backward keys: around C 512
# and 1024, 64-channel tiles (C 520 and 576), a tie of the two forms (C 600) and 128-channel tiles (C 768 and 1016, the
# last tile partial).
BACKWARD_DIRECT_EDGE_CHANNELS = (504, 512, 520, 576, 600, 768, 1016, 1024, 1032)


def _backward_direct_edges(
    arch: str, route: str | None = None
) -> tuple[list[int], list[int]]:
    """``(channel counts, token counts)``, ascending: both sides of every channel bound and the token counts below,
    at, one above the first and at and above the last token count of every backward direct key of ``arch`` (of
    ``route`` only when given), the channel counts with :data:`BACKWARD_DIRECT_EDGE_CHANNELS`."""

    channels, tokens = set(BACKWARD_DIRECT_EDGE_CHANNELS), set()
    for (name, c_lo, c_hi), (s_lo, s_hi, _) in BACKWARD_DIRECT_ROWS_OF_RECORD.get(
        arch, {}
    ).items():
        if route is None or name == route:
            channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
            tokens |= {s_lo - 1, s_lo, s_lo + 1, s_hi, s_hi + 1}
    return sorted(C for C in channels if C > 0), sorted(S for S in tokens if S > 0)


def _direct_grid(sms: int, C: int, S: int) -> int:
    """The grid of a direct backward call at ``bwd_grid_mult`` 1: at least one CTA per channel tile, at most one per
    token block of ``n_sub x 4`` tokens, and at most the co-resident 2 CTAs per SM."""

    _, n_sub, tiles = _backward_direct_split(C)
    return max(tiles, min(2 * sms, tiles * max(1, -(-S // (4 * n_sub)))))


@pytest.mark.cpu_only
def test_backward_direct_records() -> None:
    """Each direct backward program is one record for every architecture: ``ffi_entry`` ``run``, its name as kernel
    and library, ``kernel_`` + name as symbol, role ``bwd``, selection rule ``{"kind": "rows"}``, the grid rule
    :data:`BACKWARD_DIRECT_GRID_RULE`, a 128-thread ordinary launch with shared memory and the per-call PDL scalar,
    one plain FP32 accumulator copy of 4 taps (5 on ``bwd_bias``) with uint32 tickets, and the arguments and logical
    names of its route's fallback without the TMA maps (``x_map`` / ``dy_map``).  Every architecture's route lists
    it last, no other route lists it, and no channel threshold or token floor admits it."""

    assert cake_jit.BACKWARD_DIRECT_THREADS == 128
    for route, name in BACKWARD_DIRECT_PROGRAMS.items():
        record = cake_jit.MODULES[name]
        label = (route, name)
        assert PROGRAM_NAME.fullmatch(name) and name not in det_jit.MODULES, label
        assert record["arches"] == list(ARCHES) and record["ffi_entry"] == "run", label
        assert (
            record["kernel"] == record["library"] == name
            and record["kernel_symbol"] == f"kernel_{name}"
        ), label
        assert record["role"] == "bwd" and cake_jit.select_rule(record) == {
            "kind": "rows"
        }, label
        assert record["grid_rule"] == BACKWARD_DIRECT_GRID_RULE, label
        launch = record["launch"]
        assert {
            key: launch[key] for key in ("block", "cluster", "cooperative", "use_pdl")
        } == {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "use_pdl": "runtime",
        }, label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] > 0
        ), label
        assert record["workspace_rule"] == {
            "acc": "float32",
            "slots": 1,
            "kind": "accumulators",
            "taps": 5 if route == "bwd_bias" else 4,
            "work": "uint32",
        }, label
        maps = ("x_map", "dy_map")
        for arch in ARCHES:
            fallback = cake_jit.MODULES[_route_fallback(arch, route)]
            assert record["arg_plan"] == [
                entry for entry in fallback["arg_plan"] if entry[1] not in maps
            ], (
                label,
                arch,
            )
            assert record["logical"] == {
                k: v for k, v in fallback["logical"].items() if k not in maps
            }, (label, arch)
            names = cake_jit.route_modules(arch, route)
            assert names[-1] == name and names.count(name) == 1, (label, arch)
            others = {
                n
                for kind, listed in cake_jit.ROUTES[arch].items()
                if kind != route
                for n in listed
            }
            assert name not in others, (label, arch)
            assert cake_jit.record_route(record, arch) == route, (label, arch)
            for C in (*CHANNELS, *BACKWARD_DIRECT_SPLITS):
                candidates, fallback_name = cake_jit._ranked_candidates(arch, route, C)
                assert name not in candidates and name != fallback_name, (
                    label,
                    arch,
                    C,
                )
                for S in (None, 1, cake_jit.INT32_MAX):
                    assert (
                        cake_jit._admits(
                            cake_jit.select_rule(record), arch=arch, C=C, S=S
                        )
                        is False
                    ), (label, C, S)
    assert not set(BACKWARD_DIRECT_PROGRAMS.values()) & set(kcp_jit.MODULES)


@pytest.mark.cpu_only
def test_backward_direct_rows_are_the_literal_rows() -> None:
    """The backward keys of ``DIRECT_ROWS`` are :data:`BACKWARD_DIRECT_ROWS_OF_RECORD` (each naming its route's
    direct backward program), ``BACKWARD_DIRECT_PLANS`` is :data:`BACKWARD_DIRECT_FORMS` (forms for exactly the routes
    of the keys, each within 128 threads).  The keys of a route are disjoint, in channel order, within C 512 to 1024
    and S 1 to 4096, each over at least two token counts.  Every key covers each multiple of 8 from its ``C_lo`` to
    its ``C_hi`` (``direct_row_range``), and next to it (``C_lo - 8``, ``C_hi + 8``) only a neighbouring key of the
    route does; ``direct_module`` names the program at both channel and token ends and ``None`` one token below and
    above; no ``(C, S)`` of a key lies in an ``SM_WIDE_ROWS`` segment."""

    assert cake_jit.BACKWARD_DIRECT_PLANS == BACKWARD_DIRECT_FORMS
    for arch in ARCHES:
        backward = {
            key: value
            for key, value in cake_jit.DIRECT_ROWS[arch].items()
            if cake_jit.route_role(key[0]) == "bwd"
        }
        assert backward == BACKWARD_DIRECT_ROWS_OF_RECORD[arch], arch
        assert set(cake_jit.BACKWARD_DIRECT_PLANS[arch]) == {
            key[0] for key in backward
        }, arch
        for route in {key[0] for key in backward}:
            ends = [(c_lo, c_hi) for other, c_lo, c_hi in backward if other == route]
            assert all(a[1] < b[0] for a, b in zip(ends, ends[1:])), (arch, route)
        for (route, c_lo, c_hi), (s_lo, s_hi, name) in backward.items():
            label = (arch, route)
            assert name == BACKWARD_DIRECT_PROGRAMS[route], label
            assert 512 <= c_lo <= c_hi <= 1024, (label, c_lo, c_hi)
            assert 1 <= s_lo < s_hi <= 4096, (label, c_lo, s_lo, s_hi)
            for ct_vec, n_sub in cake_jit.BACKWARD_DIRECT_PLANS[arch][route]:
                assert ct_vec * n_sub == 128, label
            for C in range(c_lo, c_hi + 1, 8):
                assert cake_jit.direct_row_range(arch, route, C) == (s_lo, s_hi), (
                    label,
                    C,
                )
                for s_seg_lo, s_seg_hi, _ in (
                    cake_jit.sm_wide_segments(arch, route, C) or ()
                ):
                    assert s_hi < s_seg_lo or s_seg_hi < s_lo, (label, C)
            assert cake_jit.direct_row_range(arch, route, c_lo + 4) is None, label
            for C in (c_lo - 8, c_hi + 8):
                near = [
                    (lo, hi)
                    for (other, c_near_lo, c_near_hi), (lo, hi, _) in backward.items()
                    if other == route and c_near_lo <= C <= c_near_hi
                ]
                want = near[0] if near else None
                assert cake_jit.direct_row_range(arch, route, C) == want, (label, C)
            for C in (c_lo, c_hi):
                for S in (s_lo, s_hi):
                    assert cake_jit.direct_module(arch, route, C, S) == name, (
                        label,
                        C,
                        S,
                    )
                for S in (s_lo - 1, s_hi + 1):
                    assert cake_jit.direct_module(arch, route, C, S) is None, (
                        label,
                        C,
                        S,
                    )
                assert cake_jit.direct_module(arch, route, C, 0) is None, (label, C)
        for route in cake_jit.backward_kinds(arch):
            if route not in BACKWARD_DIRECT_PROGRAMS:
                assert cake_jit.direct_row_range(arch, route, 768) is None, (
                    arch,
                    route,
                )


@pytest.mark.cpu_only
def test_backward_direct_split_by_channel_count() -> None:
    """A backward direct row of ``C`` channels runs the first form of ``BACKWARD_DIRECT_PLANS`` whose tiles pad ``C``
    to the fewest channel vectors (:data:`BACKWARD_DIRECT_SPLITS`, by the rule at every multiple of 8 from 512 to
    1024: at most 15 tiles), whatever ``S`` and the architecture; ``direct_tiles`` (the workspace bound) is that tile
    count at a channel count a key covers, 0 outside every key, at another thread count and on a route without a
    key.  The forward routes keep their one form of ``DIRECT_PLANS``.  A form beyond the geometry's threads, a route
    without a form and ``C`` off the 8-channel grid are ``ValueError``."""

    geometry = {"threads": 128, "vec": 8}
    for C, split in BACKWARD_DIRECT_SPLITS.items():
        assert _backward_direct_split(C) == split, C
    for arch in ARCHES:
        for route in BACKWARD_DIRECT_PROGRAMS:
            for C in range(512, 1025, 8):
                split = cake_jit.direct_split(
                    C, geometry=geometry, route=route, arch=arch
                )
                assert split == _backward_direct_split(C), (arch, route, C)
                assert split[0] * split[2] >= C // 8 and split[2] <= 15, (
                    arch,
                    route,
                    C,
                )
                covered = any(
                    name == route and c_lo <= C <= c_hi
                    for name, c_lo, c_hi in BACKWARD_DIRECT_ROWS_OF_RECORD[arch]
                )
                assert cake_jit.direct_tiles(arch, route, C, geometry) == (
                    split[2] if covered else 0
                ), (arch, route, C)
                assert (
                    cake_jit.direct_tiles(arch, route, C, {**geometry, "threads": 256})
                    == 0
                ), (arch, route, C)
            for C in (504, 1032, 2048):
                assert cake_jit.direct_tiles(arch, route, C, geometry) == 0, (
                    arch,
                    route,
                    C,
                )
            with pytest.raises(ValueError, match="does not fit"):
                cake_jit.direct_split(
                    576, geometry={"threads": 64, "vec": 8}, route=route, arch=arch
                )
            with pytest.raises(ValueError):
                cake_jit.direct_split(580, geometry=geometry, route=route, arch=arch)
        for route in ("bwd_fp32g", "bwd_bias_fp32g"):
            assert cake_jit.direct_tiles(arch, route, 768, geometry) == 0, (arch, route)
            with pytest.raises(ValueError, match="no direct plan"):
                cake_jit.direct_split(768, geometry=geometry, route=route, arch=arch)
        for route, (ct_vec, n_sub) in DIRECT_PLANS_OF_RECORD[arch].items():
            want = (ct_vec, n_sub, -(-96 // ct_vec))
            assert (
                cake_jit.direct_split(768, geometry=geometry, route=route, arch=arch)
                == want
            ), (arch, route)
            assert cake_jit.direct_tiles(arch, route, 768, geometry) == 0, (arch, route)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_direct_rows_run_ahead_of_every_other_table(arch, monkeypatch) -> None:
    """Inside a backward direct key the route's direct program runs on its direct tiles even where ``SM_WIDE_ROWS``,
    ``FALLBACK_ROWS``, ``TILE_OVERRIDES`` and ``NARROW_TILES`` keys cover the same ``(C, S)`` (each replaced here by a
    key over every token count), at C 768 of the ``bwd`` route; one token above the key those tables decide again, in
    that order.  ``plan`` of a record of another thread count keeps its own plan there."""

    route, C = "bwd", 768
    keys = BACKWARD_DIRECT_ROWS_OF_RECORD[arch]
    ((s_lo, s_hi),) = [
        (lo, hi)
        for (name, c_lo, c_hi), (lo, hi, _) in keys.items()
        if name == route and c_lo <= C <= c_hi
    ]
    sm_wide = next(
        name
        for name, det in SM_WIDE_COPIES.items()
        if arch in det_jit.MODULES[det]["arches"]
    )
    geometry = {"threads": 128, "vec": 8}
    direct = BACKWARD_DIRECT_PROGRAMS[route]
    tables = (
        ("SM_WIDE_ROWS", {(route, 512, 1024): ((1, cake_jit.INT32_MAX, sm_wide),)}),
        ("FALLBACK_ROWS", {(route, 512, 1024): (1, cake_jit.INT32_MAX)}),
        ("TILE_OVERRIDES", {(route, C): (32, 1, cake_jit.INT32_MAX)}),
        ("NARROW_TILES", {(route, 512, 1024): (1, cake_jit.INT32_MAX)}),
    )
    for table, rows in tables:
        with monkeypatch.context() as m:
            m.setitem(getattr(cake_jit, table), arch, rows)
            for S in sorted({s_lo, s_hi, *(S for S in (1000,) if s_lo <= S <= s_hi)}):
                assert cake_jit.selected_modules(arch, route, C=C, S=S) == (direct,), (
                    table,
                    S,
                )
                got = cake_jit.plan(
                    S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                )
                assert _split3(got) == BACKWARD_DIRECT_SPLITS[C], (table, S)
                wide = cake_jit.plan(
                    S,
                    C,
                    3,
                    geometry={**geometry, "threads": 256},
                    role="bwd",
                    arch=arch,
                    route=route,
                )
                assert _split3(wide) != BACKWARD_DIRECT_SPLITS[C], (table, S)
            (above,) = cake_jit.selected_modules(arch, route, C=C, S=s_hi + 1)
            got = cake_jit.plan(
                s_hi + 1, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
            )
            if table == "SM_WIDE_ROWS":
                assert above == sm_wide, table
            elif table == "FALLBACK_ROWS":
                assert above == _route_fallback(arch, route), table
            else:
                assert above != direct and _split3(got) == (32, 4, 3), (
                    table,
                    above,
                    _split3(got),
                )


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_direct_rows_select_and_plan_at_their_edges(arch, fake_device) -> None:
    """At the channel and token counts around the bounds of each backward direct key of the route
    (:func:`_backward_direct_edges`), at every ``bwd_grid_mult``, a call inside a key launches the route's direct
    program on the direct split with the capped grid times ``bwd_grid_mult`` (:func:`_direct_grid`); a call outside
    every key launches what the package without the direct rows launches (:func:`_without_backward_direct_rows`), with
    the same plan and grid.  The row plan equals the per-call plan everywhere."""

    sms = PLAN_SWEEP_SMS[arch][0]
    fake_device(arch, sms)
    calls = {}
    for route, name in BACKWARD_DIRECT_PROGRAMS.items():
        channels, tokens = _backward_direct_edges(arch, route)
        for C in channels:
            for S in tokens:
                for gm in pkg.BWD_GRID_MULTS:
                    got = _row_planned(arch, route, S, C, 3, gm, "bf16")
                    assert got == _per_call(arch, route, S, C, 3, gm, "bf16"), (
                        route,
                        C,
                        S,
                        gm,
                    )
                    if _backward_direct_cell(arch, route, C, S) is not None:
                        assert got[0] == (name,), (route, C, S, gm)
                        assert _split3(got[1]) == _backward_direct_split(C), (
                            route,
                            C,
                            S,
                            gm,
                        )
                        assert got[2] == (_direct_grid(sms, C, S) * gm, 1, 1), (
                            route,
                            C,
                            S,
                            gm,
                        )
                    else:
                        assert got[0] != (name,), (route, C, S, gm)
                        calls[(route, C, S, gm)] = got
    assert calls
    with _without_backward_direct_rows():
        for (route, C, S, gm), got in calls.items():
            assert _row_planned(arch, route, S, C, 3, gm, "bf16") == got, (
                route,
                C,
                S,
                gm,
            )
            assert _per_call(arch, route, S, C, 3, gm, "bf16") == got, (route, C, S, gm)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_direct_grid_and_bwd_grid_mult(arch, fake_device) -> None:
    """At the cells a backward direct key holds (its channel and token bounds among them), a direct backward record's
    grid is ``max(n_ch_tiles, min(2 x SMs, n_ch_tiles x max(1, ceil(S / (4 x n_sub)))))`` (``block_tokens`` 4 of its
    grid rule), on every SM count of :data:`PLAN_SWEEP_SMS` and on a device too small for one CTA per tile at 2 per SM;
    ``bwd_grid_mult`` 1 to 4 multiplies it, in the row plan and the per-call plan alike.  Without ``C`` and ``S``
    ``launch_grid`` refuses to size it."""

    planned_cells = 0
    for sms in (*PLAN_SWEEP_SMS[arch], 2):
        fake_device(arch, sms)
        for route, name in BACKWARD_DIRECT_PROGRAMS.items():
            record = cake_jit.MODULES[name]
            rows = BACKWARD_DIRECT_ROWS_OF_RECORD[arch]
            keys = [
                (c_lo, c_hi, s_lo, s_hi)
                for (key_route, c_lo, c_hi), (s_lo, s_hi, _) in rows.items()
                if key_route == route
            ]
            channels = {512, 520, 576, 768, 960, 1024}
            channels.update(c for c_lo, c_hi, _, _ in keys for c in (c_lo, c_hi))
            tokens = {1, 2, 5, 33, 64, 65, 1000}
            tokens.update(s for _, _, s_lo, s_hi in keys for s in (s_lo, s_hi))
            for C in sorted(channels):
                for S in sorted(tokens):
                    if _backward_direct_cell(arch, route, C, S) is None:
                        continue
                    planned_cells += 1
                    want = _direct_grid(sms, C, S)
                    assert want >= _backward_direct_split(C)[2], (sms, route, C, S)
                    got = cake_jit.launch_grid(
                        record, device_index=FAKE_DEVICE, C=C, S=S
                    )
                    assert got == (want, 1, 1), (sms, route, C, S)
                    for gm in pkg.BWD_GRID_MULTS:
                        planned = _row_planned(arch, route, S, C, 3, gm, "bf16")
                        assert planned[2] == (want * gm, 1, 1), (sms, route, C, S, gm)
                        assert _per_call(arch, route, S, C, 3, gm, "bf16") == planned, (
                            sms,
                            route,
                            C,
                            S,
                            gm,
                        )
            for kwargs in ({}, {"C": 512}, {"S": 1}):
                with pytest.raises(ValueError, match="needs C and S"):
                    cake_jit.launch_grid(record, device_index=FAKE_DEVICE, **kwargs)
    assert planned_cells, arch


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_plans_outside_the_direct_rows_are_unchanged(
    arch, fake_device
) -> None:
    """Outside the backward direct keys every backward call of every route (the fp32g and pair routes included) plans
    the program, scalars and grid of the package without its direct backward programs
    (:func:`_without_backward_direct_rows`), in the row plan and the per-call plan, at every ``bwd_grid_mult``: channel
    counts at and around the keys' channel bounds and token counts around their token bounds.  The cells of
    :data:`BACKWARD_OUTSIDE_PLANS_OF_RECORD` keep their program and split."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    edge_channels, edge_tokens = _backward_direct_edges(arch)
    channels = sorted({*CHANNELS, *edge_channels, *BACKWARD_DIRECT_SPLITS})
    tokens = sorted(
        {1, 2, 1000, 2048, 2049, 4096, 4097, 16384, 16385, 62585, 250340, *edge_tokens}
    )

    def policy_of(route: str) -> str:
        return "fp32" if "fp32g" in route else "bf16"

    def sweep() -> dict:
        planned = {}
        for route in cake_jit.backward_kinds(arch):
            for C in channels:
                for S in tokens:
                    if _backward_direct_cell(arch, route, C, S) is not None:
                        continue
                    for gm in pkg.BWD_GRID_MULTS:
                        got = _row_planned(arch, route, S, C, 3, gm, policy_of(route))
                        assert got == _per_call(
                            arch, route, S, C, 3, gm, policy_of(route)
                        ), (route, C, S, gm)
                        planned[(route, C, S, gm)] = got
        return planned

    shipped = sweep()
    with _without_backward_direct_rows():
        base = sweep()
    assert shipped == base, arch
    checked = 0
    for (cell_arch, route, C, S), (
        program,
        split,
    ) in BACKWARD_OUTSIDE_PLANS_OF_RECORD.items():
        if cell_arch != arch:
            continue
        assert _backward_direct_cell(arch, route, C, S) is None, (route, C, S)
        names, scalars, _ = _row_planned(arch, route, S, C, 3, 1, "bf16")
        assert names == (program,) and _split3(scalars) == split, (
            route,
            C,
            S,
            names,
            _split3(scalars),
        )
        checked += 1
    assert checked, arch


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_public_backward_workspace_is_unchanged_by_the_direct_rows(arch) -> None:
    """The public backward workspace (``backward_workspace_bytes``) is the same with and without the direct backward
    programs at every channel count (the fp32g programs' sixteen accumulator copies set it), and holds every direct
    backward program's layout at its direct tiles: ``acc`` of 4 (5) FP32 words per channel and one ticket per tile."""

    channels = sorted({*CHANNELS, *range(8, 2049, 8)})
    shipped = {C: cake_jit.backward_workspace_bytes(arch, C) for C in channels}
    with _without_backward_direct_rows():
        base = {C: cake_jit.backward_workspace_bytes(arch, C) for C in channels}
    assert shipped == base, arch
    for route, name in BACKWARD_DIRECT_PROGRAMS.items():
        record = cake_jit.MODULES[name]
        taps = 5 if route == "bwd_bias" else 4
        for C in range(512, 1025, 8):
            tiles = _backward_direct_split(C)[2]
            layout = cake_jit.accumulator_layout(record, C, tiles)
            assert (
                layout["acc_bytes"] == 4 * taps * C and layout["work_words"] == tiles
            ), (route, C)
            assert layout["total_bytes"] < shipped[C], (route, C)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_is_unchanged_by_the_backward_direct_rows(arch, fake_device) -> None:
    """The KCP routes select, plan and size their workspace as without the direct backward programs: every KCP route's
    selection, the KCP workspace and the KCP forward's single-rank main program (``kcp._sr_main``) at channel and
    token counts inside and around the backward direct keys."""

    kcp = cake_package(KEY, "kcp")
    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    channels = (504, 512, 576, 768, 1024, 1032, 2048)
    tokens = (1, 2, 1000, 2048, 2049, 4096, 4097, 65536)

    def read() -> dict:
        out = {}
        for C in channels:
            out[("workspace", C)] = kcp_jit.workspace_bytes(arch, C)
            for S in tokens:
                for kind in kcp_jit.ROUTES.get(arch, {}):
                    out[(kind, C, S)] = kcp_jit.selected_modules(arch, kind, C=C, S=S)
                out[("main", C, S)] = kcp._sr_main(
                    arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE
                )
        return out

    shipped = read()
    with _without_backward_direct_rows():
        base = read()
    assert shipped == base, arch
    assert not {
        name for names in kcp_jit.ROUTES.get(arch, {}).values() for name in names
    } & set(BACKWARD_DIRECT_PROGRAMS.values())


# ----------------------------------------------------------------------------------------------------------------
# The public names: the mode word in every name, no mode keyword.

PUBLIC = (
    "cake_causal_conv1d_nondet_train_forward",
    "cake_causal_conv1d_nondet_train_backward",
    "cake_causal_conv1d_nondet",
    "cake_causal_conv1d_nondet_train_forward_workspace",
    "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_backward_workspace",
    "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
)
# The names before the mode word joined them: removed, with no alias.
OLD_NAMES = (
    "plan_cache_clear",
    "cake_causal_conv1d",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "is_cake_causal_conv1d_train_supported",
)


@pytest.mark.cpu_only
def test_public_names_carry_the_mode_and_take_no_mode_keyword() -> None:
    """Every public name of ``__all__`` but the constants carries the mode word ``nondet``; no
    entry point and no parameter of the autograd function's forward is a ``deterministic`` keyword; the names before
    the rename are gone (no alias)."""

    for name in PUBLIC:
        assert name in pkg.__all__, name
        assert (
            "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
        ), name
    assert "CakeCausalConv1dNondetFunction" in pkg.__all__
    assert (
        "deterministic"
        not in inspect.signature(pkg.CakeCausalConv1dNondetFunction.forward).parameters
    )
    for name in pkg.__all__:
        assert name.isupper() or "nondet" in name.lower(), name
    for name in OLD_NAMES:
        assert name not in pkg.__all__ and not hasattr(pkg, name), name


@pytest.mark.cpu_only
def test_a_mode_keyword_is_an_ordinary_type_error() -> None:
    """``deterministic=`` (either value) is Python's ordinary ``TypeError`` for an unexpected keyword at every entry
    point, raised before any argument, device or rendering check, so it needs neither a device nor ``build.json``;
    the autograd function takes no positional mode argument after ``dweight_sum`` (and a mode value in the
    ``dweight_sum`` position is a ``TypeError`` too)."""

    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    cpu = torch.device("cpu")
    calls = {
        "forward": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, **d),
        "backward": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x, **d),
        "autograd": lambda **d: pkg.cake_causal_conv1d_nondet(x, w, cu, **d),
        "fwd_ws": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward_workspace(8, 16, cpu, **d),
        "fwd_ws_bytes": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward_workspace_bytes(8, 16, cpu, **d),
        "bwd_ws": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward_workspace(8, 16, cpu, **d),
        "bwd_ws_bytes": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(8, 16, cpu, **d),
    }  # fmt: skip
    for name, call in calls.items():
        for value in (True, False):
            with pytest.raises(TypeError) as info:
                call(deterministic=value)
            assert "unexpected keyword argument 'deterministic'" in str(info.value), (
                name
            )
    with pytest.raises(TypeError):
        pkg.CakeCausalConv1dNondetFunction.apply(x, w, cu, None, "bf16", 1, False)
    with pytest.raises(TypeError):
        pkg.CakeCausalConv1dNondetFunction.apply(x, w, cu, None, "bf16", 1, "fp32", False)


@pytest.mark.cpu_only
def test_alert_helper_in_both_torch_modes(monkeypatch) -> None:
    """The backend's own non-determinism alert (no private PyTorch hook: ``torch._C._alert_not_deterministic`` is
    absent from some PyTorch builds, so it is removed here): silent with torch's deterministic algorithms off, a
    ``RuntimeError`` naming the entry and the deterministic package when on, a ``UserWarning`` with the same naming
    under ``warn_only=True``.  The KCP module uses the same helper."""

    conv_kcp = cake_package(KEY, "kcp")

    assert conv_kcp._alert_nondeterministic is pkg._alert_nondeterministic
    monkeypatch.delattr(torch._C, "_alert_not_deterministic", raising=False)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(False)
        with warnings_recorded() as caught:
            pkg._alert_nondeterministic("probe_entry")
        assert caught == []
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError,
            match="probe_entry does not have a deterministic implementation",
        ) as info:
            pkg._alert_nondeterministic("probe_entry")
        assert "cake_pplx.cake_causal_conv1d_det_train" in str(info.value)
        assert "warn_only=True" in str(info.value)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning,
            match="probe_entry does not have a deterministic implementation",
        ) as record:
            pkg._alert_nondeterministic("probe_entry")
        assert any(
            "cake_pplx.cake_causal_conv1d_det_train" in str(w.message) for w in record
        )
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


@pytest.mark.cpu_only
def test_backward_entries_alert_before_any_device_work(monkeypatch) -> None:
    """Under torch's deterministic mode the single-rank backward alerts after argument validation and before any
    allocation or launch, and the forward does not alert.  Checked without a device: zero-token CPU tensors with the
    device / placement checks stubbed (the zero-token path returns before any device query)."""

    monkeypatch.setattr(pkg, "_device_arch", lambda device: "sm_100a")
    monkeypatch.setattr(pkg, "_check_placement", lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(pkg, "run_module", lambda *a, **k: launched.append(a))
    x = torch.empty(0, 16, dtype=torch.bfloat16)
    w = torch.zeros(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 0], dtype=torch.int32)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward does not have"
        ):
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
        # The forward runs the deterministic forward programs: no alert.
        y = pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu)
        assert tuple(y.shape) == (0, 16)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning, match="cake_causal_conv1d_nondet_train_backward does not have"
        ):
            dx, dw = pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
        assert tuple(dx.shape) == (0, 16) and torch.equal(dw, torch.zeros_like(dw))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    assert launched == []


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_entry_passes_its_row_to_pdl_enabled(arch, fake_device, monkeypatch) -> None:
    """The backward entry passes its route, ``C`` and ``S`` to ``pdl_enabled``: its launch takes ``enable_pdl`` 0 at
    the rows of :data:`PDL_OFF_ROWS_OF_RECORD`, eagerly and inside a capture, and the tables' value elsewhere (one
    token below a key, every row of an architecture without keys); a call of the bias route at a key's row and every
    forward launch at those rows take the tables' value.  Checked without a device: meta tensors, a fake device of
    the architecture, the capture query, the device index, the dweight alignment check and the workspace helpers
    stubbed, and each launch's ``enable_pdl`` recorded at ``run_module`` (nothing launches)."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    monkeypatch.setattr(pkg, "_device_arch", lambda device: arch)
    monkeypatch.setattr(pkg, "_device_index", lambda device: FAKE_DEVICE)
    monkeypatch.setattr(pkg, "_dweight_aligned", lambda dw: dw)
    monkeypatch.setattr(pkg, "_forward_work", lambda *args, **kwargs: None)
    monkeypatch.setattr(pkg, "_backward_accumulators", lambda *args, **kwargs: (None, None))
    launched = []
    monkeypatch.setattr(
        pkg, "run_module", lambda name, values, **kwargs: launched.append((name, values["enable_pdl"]))
    )
    capture = {"on": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture["on"])
    meta = torch.device("meta")
    rows = PDL_OFF_ROWS_OF_RECORD.get(arch, {})
    cases = {(250340, 1016)}
    for (route, c_lo, c_hi), (s_min, _s_max) in rows.items():
        assert route == "bwd", route
        cases |= {(S, C) for C in (c_lo, c_hi) for S in (s_min - 1, s_min, max(s_min, 250340))}
    turned_off = 0
    for S, C in sorted(cases):
        x = torch.empty((S, C), dtype=torch.bfloat16, device=meta)
        w = torch.empty((C, 4), dtype=torch.float32, device=meta)
        b = torch.empty((C,), dtype=torch.float32, device=meta)
        cu = torch.empty((2,), dtype=torch.int32, device=meta)
        for capturing in (False, True):
            capture["on"] = capturing
            table = int((cake_jit.PDL_GRAPH if capturing else cake_jit.PDL_EAGER)[arch])
            want = _pdl_off_expected(arch, "bwd", C, S, capturing)
            assert want == cake_jit.pdl_enabled(arch, capturing=capturing, kind="bwd", C=C, S=S)
            label = (S, C, capturing)
            launched.clear()
            pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu)
            forward = list(launched)
            assert forward and all(name in cake_jit.route_modules(arch, "fwd") for name, _ in forward), label
            assert all(value == table for _, value in forward), (label, forward)
            launched.clear()
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
            ((name, value),) = launched
            assert name in cake_jit.route_modules(arch, "bwd"), (label, name)
            assert value == want, (label, name, value, want)
            launched.clear()
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x, bias=b)
            ((name, value),) = launched
            assert name in cake_jit.route_modules(arch, "bwd_bias"), (label, name)
            assert value == table, (label, name, value, table)
            turned_off += want != table
    assert bool(turned_off) is bool(rows), (arch, turned_off)


@pytest.mark.cpu_only
def test_unrendered_package_reports_unsupported() -> None:
    assert cake_jit.is_rendered() is (PACKAGE_DIR / "build.json").is_file()
    if not cake_jit.is_rendered():
        assert (
            pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cpu"))
            is False
        )
        with pytest.raises(RuntimeError):
            cake_build.manifest()


# ----------------------------------------------------------------------------------------------------------------
# Wiring.


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    assert (
        pkg.__name__ == "cake_pplx._backends.cake_causal_conv1d_nondet_train"
    )
    # ``cake_pplx`` and ``cake_pplx._backends`` are PEP 420 namespace levels shared by every family: neither carries
    # an ``__init__.py``.
    assert not (REPO / "cake_pplx" / "__init__.py").exists()
    assert not (REPO / "cake_pplx" / "_backends" / "__init__.py").exists()
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    # One family-wide rule ships every backend package's ``build.json`` and ``csrc/`` tree.
    assert {"build.json", "csrc/**/*"} <= set(package_data["*"]), package_data


@pytest.mark.cpu_only
def test_build_manifest_lists_every_program() -> None:
    """Once rendered: ``build.json`` names this package, carries exactly one entry per (program, architecture) of
    the single-rank and KCP registries, each with its kernel and binding sources under ``csrc/`` and their digests,
    IEEE division / square root and no fast math; no record is a placeholder any more."""

    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    referenced = set()
    for registry in (cake_jit, kcp_jit):
        for name, record in registry.MODULES.items():
            assert "TODOEXPORT" not in name, name
            assert set(record["arches"]) <= set(manifest["arch_flags"]), name
            for arch in ARCHES:
                library = f"{name}_{arch}"
                if arch not in record["arches"]:
                    assert library not in libraries, library
                    continue
                entry = libraries[registry.library_name(name, arch)]
                assert entry["arch"] == arch and entry["entry"] == "run", library
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                assert "--use_fast_math" not in entry["nvcc_flags"], library
                assert {"--prec-div=true", "--prec-sqrt=true", "--ftz=false"} <= set(
                    entry["nvcc_flags"]
                )
                for source in entry["sources"]:
                    path = PACKAGE_DIR / source
                    assert path.is_file(), source
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        == entry["sha256"][source]
                    ), source
                referenced.add(library)
    assert referenced == set(libraries)


def _manifest_checker():
    """``scripts/check_cake_build_manifests.py`` of this checkout, loaded as a module."""

    path = REPO / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
def test_manifest_checker_lists_the_package() -> None:
    """The build-manifest checker lists this package (its ``cake_jit.py`` and the KCP registry ``kcp_jit.py``) and
    its vendored gate finds the package's suite key; once rendered, the in-tree checks report nothing."""

    checker = _manifest_checker()
    try:
        path = str(PACKAGE_DIR.relative_to(REPO))
    except ValueError:
        pytest.skip(
            f"the package is imported from {PACKAGE_DIR}, outside this checkout"
        )
    entries = [p for p in checker.PACKAGES if p.name == pkg.__name__]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert (entry.path, entry.loader) == (path, "cake_jit.py")
    assert entry.refs is checker.refs_with_kcp
    assert checker.suite_key(REPO, entry) == KEY
    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet")
    assert checker.run_in_tree(REPO, [entry]) == []


# ----------------------------------------------------------------------------------------------------------------
# The small-S grid cap of the single-wave backward (``GRID_CAP_ROWS``).

GRID_CAP_ROWS_OF_RECORD = {
    "sm_107a": {
        ("bwd", 584, 592): ((512, 1000, 3),),
        ("bwd", 600, 616): ((512, 999, 3),),
        ("bwd", 624, 632): ((512, 1000, 3),),
        ("bwd", 648, 696): ((512, 1000, 3),),
        ("bwd", 704, 704): ((512, 999, 3),),
        ("bwd", 712, 760): ((512, 1000, 3),),
        ("bwd", 768, 768): ((512, 512, 3),),
        ("bwd_bias", 512, 512): ((2049, 3072, 5),),
    },
}
# Device SM counts the cap is checked at: VR200's, one more and one of a smaller part.
GRID_CAP_SMS = (212, 216, 132)


@contextmanager
def _without_grid_cap_rows():
    """Inside the block ``GRID_CAP_ROWS`` is empty (every launch keeps its uncapped grid).  The host's row plans are
    dropped on entry and on exit."""

    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    try:
        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_jit, "GRID_CAP_ROWS", {})
            yield
    finally:
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


@pytest.mark.cpu_only
def test_grid_cap_rows_are_the_literal_rows() -> None:
    """``GRID_CAP_ROWS`` is the table of record: sm_107a alone, seven ``bwd`` keys of 3 tokens per CTA and one
    ``bwd_bias`` key of 5.  Each key is on a backward route of the architecture, with multiples of 8 at both ends and
    ascending, disjoint spans of at least one token; at every channel count of a key and both ends of each span
    ``grid_cap_tokens`` gives the span's tokens, and no backward ``DIRECT_ROWS`` range, ``SM_WIDE_ROWS`` segment or
    ``FALLBACK_ROWS`` range holds the call; one token outside a span, off the 8-channel grid and on another route
    there is no cap."""

    assert cake_jit.GRID_CAP_ROWS == GRID_CAP_ROWS_OF_RECORD
    for arch, rows in cake_jit.GRID_CAP_ROWS.items():
        for (route, c_lo, c_hi), spans in rows.items():
            assert route in cake_jit.backward_kinds(arch), (arch, route)
            assert c_lo % 8 == 0 and c_hi % 8 == 0 and 0 < c_lo <= c_hi, (route, c_lo)
            assert all(1 <= lo <= hi and tokens >= 1 for lo, hi, tokens in spans)
            assert all(a[1] < b[0] for a, b in zip(spans, spans[1:]))
            for C in range(c_lo, c_hi + 1, 8):
                assert cake_jit.grid_cap_spans(arch, route, C) == spans, (route, C)
                assert cake_jit.grid_cap_tokens(arch, route, C + 4, spans[0][0]) is None
                for s_lo, s_hi, tokens in spans:
                    for S in (s_lo, s_hi):
                        cell = (arch, route, C, S)
                        assert cake_jit.grid_cap_tokens(*cell) == tokens, cell
                        assert not cake_jit.direct_row(*cell), cell
                        assert cake_jit.sm_wide_module(*cell) is None, cell
                        bounds = cake_jit.fallback_row_range(arch, route, C)
                        assert bounds is None or not bounds[0] <= S <= bounds[1], cell
                        assert cake_jit.grid_cap_tokens(arch, "bwd_dt", C, S) is None
                    for S in (s_lo - 1, s_hi + 1):
                        assert cake_jit.grid_cap_tokens(arch, route, C, S) is None
    for arch in ("sm_100a", "sm_103a"):
        assert cake_jit.grid_cap_spans(arch, "bwd", 600) is None


@pytest.mark.cpu_only
def test_small_s_grid_is_the_rule() -> None:
    """``small_s_grid`` inside a span at ``grid_mult`` 1 is ``max(n_ch_tiles, min(grid, n_ch_tiles x ceil(S /
    (n_sub x tokens))))``; outside every span, at ``grid_mult`` 2 to 4 and at ``S`` 0 it is ``grid``."""

    for (route, c_lo, _), spans in GRID_CAP_ROWS_OF_RECORD["sm_107a"].items():
        for s_lo, s_hi, tokens in spans:
            for S in (s_lo, (s_lo + s_hi) // 2, s_hi):
                for grid, n_ch_tiles, n_sub in ((424, 3, 4), (432, 1, 2), (64, 3, 4)):
                    plan = {"n_ch_tiles": n_ch_tiles, "n_sub": n_sub}
                    cell = {"arch": "sm_107a", "route": route, "C": c_lo, "S": S}
                    want = n_ch_tiles * -(-S // (n_sub * tokens))
                    want = max(n_ch_tiles, min(grid, want))
                    assert cake_jit.small_s_grid(grid, **cell, **plan) == want
                    for gm in (2, 3, 4):
                        got = cake_jit.small_s_grid(grid, **cell, **plan, grid_mult=gm)
                        assert got == grid
            for S in (s_lo - 1, s_hi + 1, 0):
                cell = {"arch": "sm_107a", "route": route, "C": c_lo, "S": S}
                assert cake_jit.small_s_grid(424, **cell, n_ch_tiles=3, n_sub=4) == 424


@pytest.mark.cpu_only
@pytest.mark.parametrize("sms", GRID_CAP_SMS)
def test_grid_cap_caps_the_single_wave_backward(sms, fake_device) -> None:
    """At the channel ends and an inner channel count of every key, at each span's ends, its middle and one token
    outside it: a backward call of the key's route (row-planned and per-call alike) launches the program and plan
    scalars of the uncapped table; at ``bwd_grid_mult`` 1 inside a span its grid is ``max(n_ch_tiles, min(grid,
    n_ch_tiles x ceil(S / (n_sub x tokens))))`` of the uncapped ``grid``, below it at some cells, and elsewhere (one
    token outside, ``bwd_grid_mult`` 2 to 4, the ``bwd_dt`` and ``bwd_fp32g`` routes) the uncapped grid."""

    fake_device("sm_107a", sms)
    capped = 0
    for (route, c_lo, c_hi), spans in GRID_CAP_ROWS_OF_RECORD["sm_107a"].items():
        for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
            for s_lo, s_hi, tokens in spans:
                for S in (s_lo - 1, s_lo, (s_lo + s_hi) // 2, s_hi, s_hi + 1):
                    for kind in (route, "bwd_dt", "bwd_fp32g"):
                        policy = "fp32" if kind.endswith("fp32g") else "bf16"
                        for gm in pkg.BWD_GRID_MULTS:
                            call = ("sm_107a", kind, S, C, 4, gm, policy)
                            with _without_grid_cap_rows():
                                base = _per_call(*call)
                            got = _per_call(*call)
                            assert _row_planned(*call) == got, call
                            assert got[:2] == base[:2], (call, got, base)
                            assert isinstance(got[1], dict), (call, got)
                            n_sub, n_ch_tiles = got[1]["n_sub"], got[1]["n_ch_tiles"]
                            want = base[2]
                            if kind == route and gm == 1 and s_lo <= S <= s_hi:
                                cap = n_ch_tiles * -(-S // (n_sub * tokens))
                                want = (max(n_ch_tiles, min(base[2][0], cap)), 1, 1)
                            assert got[2] == want, (call, got[2], base[2])
                            capped += got[2] != base[2]
    assert capped, sms


@pytest.mark.cpu_only
def test_grid_cap_leaves_the_direct_rows(fake_device, monkeypatch) -> None:
    """No call at a backward ``DIRECT_ROWS`` range takes the cap: the cells of the direct ranges next to the keys hold
    no cap, and under a cap key laid over every backward key of sm_107a (C 512 to 1024, S 1 to 4096 on ``bwd`` and
    ``bwd_bias``) each direct range still launches its direct program on the direct grid of the uncapped table, and
    an ``occupancy`` record's ``launch_grid`` there is its uncapped grid, while one channel count off the direct
    rows (C 584, S 100) the laid key caps the static program's grid by the rule."""

    fake_device("sm_107a", 212)
    for route, C, S in (
        ("bwd", 600, 1000),
        ("bwd", 640, 512),
        ("bwd", 704, 1000),
        ("bwd", 768, 1000),
        ("bwd_bias", 512, 2048),
    ):
        assert cake_jit.direct_row("sm_107a", route, C, S), (route, C, S)
        assert cake_jit.grid_cap_tokens("sm_107a", route, C, S) is None, (route, C, S)
    cells = [
        (route, C, S)
        for (route, c_lo, c_hi), (s_lo, s_hi, _) in BACKWARD_DIRECT_ROWS_OF_RECORD[
            "sm_107a"
        ].items()
        if route in ("bwd", "bwd_bias")
        for C in sorted({c_lo, c_hi})
        for S in (s_lo, s_hi)
    ]
    assert cells
    static = {}
    for route, C, S in cells:
        with _without_backward_direct_rows():
            names = cake_jit.selected_modules("sm_107a", route, C=C, S=S)
        static[route, C, S] = cake_jit.MODULES[names[0]]
    shipped = {}
    for route, C, S in cells:
        call = _per_call("sm_107a", route, S, C, 4, 1, "bf16")
        record = static[route, C, S]
        grid = None
        if record["grid_rule"]["kind"] == "occupancy":
            grid = cake_jit.launch_grid(record, device_index=FAKE_DEVICE, C=C, S=S)
        shipped[route, C, S] = call, grid
    assert any(grid is not None for _, grid in shipped.values())
    off = ("sm_107a", "bwd", 100, 584, 4, 1, "bf16")
    base = _per_call(*off)
    laid = {
        "sm_107a": {
            ("bwd", 512, 1024): ((1, 4096, 3),),
            ("bwd_bias", 512, 1024): ((1, 4096, 5),),
        }
    }
    monkeypatch.setattr(cake_jit, "GRID_CAP_ROWS", laid)
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    for route, C, S in cells:
        call, grid = shipped[route, C, S]
        assert call[0] == (BACKWARD_DIRECT_PROGRAMS[route],), (route, C, S, call)
        assert _per_call("sm_107a", route, S, C, 4, 1, "bf16") == call, (route, C, S)
        assert _row_planned("sm_107a", route, S, C, 4, 1, "bf16") == call, (route, C, S)
        if grid is not None:
            record = static[route, C, S]
            got = cake_jit.launch_grid(record, device_index=FAKE_DEVICE, C=C, S=S)
            assert got == grid, (route, C, S)
    got = _per_call(*off)
    assert _row_planned(*off) == got
    assert got[:2] == base[:2], (got, base)
    n_sub, n_ch_tiles = got[1]["n_sub"], got[1]["n_ch_tiles"]
    want = max(n_ch_tiles, min(base[2][0], n_ch_tiles * -(-100 // (n_sub * 3))))
    assert got[2] == (want, 1, 1) and want < base[2][0], (got, base)
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


@pytest.mark.cpu_only
def test_grid_cap_leaves_kcp_and_the_workspace(fake_device) -> None:
    """The KCP routes' selection, the KCP workspace, the KCP forward's single-rank main program (``kcp._sr_main``),
    the public backward workspace and the token ranges of the host's row plans (``row_sides``: the spans cut no row)
    are the same with and without ``GRID_CAP_ROWS``, at channel and token counts inside and around its keys."""

    kcp = cake_package(KEY, "kcp")
    arch = "sm_107a"
    fake_device(arch, 212)
    channels = (504, 512, 584, 600, 640, 704, 712, 760, 768, 776)
    tokens = (1, 511, 512, 513, 999, 1000, 1001, 2048, 2049, 3072, 3073, 65536)

    def attempt(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, RuntimeError) as exc:
            return type(exc).__name__, str(exc)

    def read() -> dict:
        out = {}
        for C in channels:
            out[("workspace", C)] = attempt(kcp_jit.workspace_bytes, arch, C)
            out[("backward_workspace", C)] = attempt(
                cake_jit.backward_workspace_bytes, arch, C
            )
            for kind in cake_jit.backward_kinds(arch):
                out[("row_sides", kind, C)] = cake_jit.row_sides(arch, kind, C)
            for S in tokens:
                for kind in kcp_jit.ROUTES.get(arch, {}):
                    out[(kind, C, S)] = attempt(
                        kcp_jit.selected_modules, arch, kind, C=C, S=S
                    )
                out[("main", C, S)] = attempt(
                    kcp._sr_main, arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE
                )
        return out

    shipped = read()
    with _without_grid_cap_rows():
        base = read()
    assert shipped == base
