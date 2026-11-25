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

The token-aware segment rule of the KDA training packages on the host (no CUDA
device): ``cake_jit.segment_route`` / ``segment_plan`` / ``segment_args`` given
the pack's token count, the plans of ``cake_backend.plan`` (its
``segment_rule``), the byte queries and the document groups of the batched
arena option, in the deterministic and the non-deterministic package.

- :data:`MEASURED_ROWS` are the rows measured on GB300 (``sm_103a``, 152 SMs)
  with the plan of each: the occupancy rule's segments per document and the
  token-aware rule's choice (``P<n>`` a uniform table of ``n`` segments per
  document, ``=`` the occupancy plan).  :data:`MEASURED_BY_ENVELOPE` gives the
  same rows with their plans on every entry of the rule's envelope: B200
  (``sm_100a``, 148 SMs), GB300 and VR200 (``sm_107a``, 212 SMs).  Every row
  takes its plan in both packages on each entry, with every record selected
  at that plan's segment count.
- One row per branch of the rule and the values at each bound.
- Over a grid of scalars: the route and the block table form never change, the
  segment count never grows, the light set saves the segment states exactly
  where the occupancy plan does, and the rule changes nothing outside its
  envelope nor in its bands that keep the occupancy plan (GB300's and VR200's
  4-head packs of at most 4096 tokens, VR200's 16-head packs of 8193 to 16383
  tokens).
- The memory of the token-aware plan (saved set, tables, arenas, the totals of
  the memory report) is at or below the occupancy plan's on every measured row
  of every entry of the envelope.
- The document groups of the batched arena option keep the whole pack's plan
  (its segments per document and its records), and the option's arena bound
  holds for every group.
- The context-parallel packages plan their local problems with the occupancy
  rule.
"""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pytestmark = pytest.mark.cpu_only

PACKAGES = ("cake.cake_kda_det_train", "cake.cake_kda_nondet_train")
PACKAGE_IDS = ("det", "nondet")
JIT = {name: cake_package(name, "cake_jit") for name in PACKAGES}
BACKEND = {name: cake_package(name, "cake_backend") for name in PACKAGES}
ARENA = {
    PACKAGES[0]: BACKEND[PACKAGES[0]].CakeKDADetArena,
    PACKAGES[1]: BACKEND[PACKAGES[1]].CakeKDANondetArena,
}
KCP_PACKAGE = {
    PACKAGES[0]: "cake.cake_kda_det_train_kcp",
    PACKAGES[1]: "cake.cake_kda_nondet_train_kcp",
}
KCP_PLAN = {name: cake_package(KCP_PACKAGE[name], "cake_kcp_plan") for name in PACKAGES}
KCP_WORKSPACE = {
    PACKAGES[0]: "CakeKDADetKcpWorkspace",
    PACKAGES[1]: "CakeKDANondetKcpWorkspace",
}

# GB300: the architecture and SM count of the rule's envelope
ARCH, SM = "sm_103a", 152
# a pack of 16384 tokens in three documents at 4 heads (16384x3:4): a short
# pack whose token-aware plan is the uniform table of 15 segments per document
# (the occupancy plan: 38)
DOCS3_LENGTHS, DOCS3_HEADS = (4096, 8192, 4096), 4
# the issue's pack (four documents at 4 heads): a pack whose occupancy plan
# saves the segment states, so both rules give that plan
ISSUE_LENGTHS, ISSUE_HEADS = (120001, 70707, 65537, 5148), 4

# The measured rows on GB300: (row, the occupancy plan's segments per document
# (``None``: the serial route), the token-aware plan).  A row is ``S x docs : H
# : gate : l2n`` (``skew``: one document holds most of the tokens, which the
# host rule cannot see); ``P<n>`` is the uniform table of ``n`` segments per
# document (proportional only at most four (document, head) chains, as the
# occupancy plan's table there), ``=`` the occupancy plan.  The 4-head rows of
# at most 4096 tokens keep the occupancy plan, a band of
# ``occupancy_bands_by_arch``.
MEASURED_ROWS = (
    ("2048x1:4:g:0", 38, "="),
    ("2048x3:16:g:0", 9, "="),
    ("4096x16:4:g:0", 10, "="),
    ("4096x1:4:g:0", 38, "="),
    ("4096x2:4:g:0", 38, "="),
    ("4096x2:8:g:0", 19, "P12"),
    ("4096x3:4:g:0", 38, "="),
    ("4096x4:16:g:0", 9, "="),
    ("4096x8:4:g:0", 19, "="),
    ("4096x8:8:g:0", 10, "="),
    ("4096xskew16:4:g:0", 10, "="),
    ("4096xskew3:4:g:0", 38, "="),
    ("4096xskew8:4:g:0", 19, "="),
    ("8192x12:8:g:0", 7, "="),
    ("8192x16:4:g:0", 10, "="),
    ("8192x1:4:g:0", 38, "P22"),
    ("8192x2:16:g:0", 9, "="),
    ("8192x2:4:g:0", 38, "P12"),
    ("8192x3:4:g:0", 38, "P12"),
    ("8192x3:8:g:0", 19, "P12"),
    ("8192x4:8:g:0", 19, "P12"),
    ("8192x8:4:g:0", 19, "P12"),
    ("8192x8:8:g:0", 10, "="),
    ("8192xskew12:8:g:0", 7, "="),
    ("8192xskew16:4:g:0", 10, "="),
    ("8192xskew3:4:g:0", 38, "P12"),
    ("8192xskew3:8:g:0", 19, "P12"),
    ("8192xskew8:4:g:0", 19, "P12"),
    ("8192xskew8:8:g:0", 10, "="),
    ("12288x16:4:g:0", 10, "="),
    ("12288x1:4:g:0", 38, "P33"),
    ("12288x2:4:g:0", 38, "P17"),
    ("12288x3:16:g:0", 9, "="),
    ("12288x6:8:g:0", 13, "P12"),
    ("12288x8:4:g:0", 19, "P12"),
    ("12288xskew16:4:g:0", 10, "="),
    ("12288xskew2:4:g:0", 38, "P17"),
    ("12288xskew6:8:g:0", 13, "P12"),
    ("12288xskew8:4:g:0", 19, "P12"),
    ("14336x16:4:g:0", 10, "="),
    ("14336xskew16:4:g:0", 10, "="),
    ("16384x10:4:g:0", 16, "P12"),
    ("16384x12:4:g:0", 13, "P12"),
    ("16384x16:16:g:0", 3, "="),
    ("16384x16:4:g:0", 10, "="),
    ("16384x16:4:g:1", 10, "="),
    ("16384x16:4:raw:0", 10, "="),
    ("16384x16:8:g:0", 5, "="),
    ("16384x1:4:g:0", 38, "="),
    ("16384x24:4:g:0", 7, "="),
    ("16384x2:16:g:0", 9, "="),
    ("16384x2:4:g:0", 38, "P22"),
    ("16384x2:8:g:0", 19, "="),
    ("16384x32:4:g:0", 5, "="),
    ("16384x32:8:g:0", 3, "="),
    ("16384x3:16:g:0", 9, "="),
    ("16384x3:4:g:0", 38, "P15"),
    ("16384x3:4:g:1", 38, "P15"),
    ("16384x3:4:raw:0", 38, "P15"),
    ("16384x3:64:g:0", None, "="),
    ("16384x3:8:g:0", 19, "P15"),
    ("16384x4:16:g:0", 9, "="),
    ("16384x4:4:g:0", 38, "P12"),
    ("16384x5:4:g:0", 31, "P12"),
    ("16384x64:4:g:0", 3, "="),
    ("16384x6:4:g:0", 26, "P12"),
    ("16384x7:4:g:0", 22, "P12"),
    ("16384x8:4:g:0", 19, "P12"),
    ("16384x8:4:g:1", 19, "P12"),
    ("16384x8:4:raw:0", 19, "P12"),
    ("16384xskew10:4:g:0", 16, "P12"),
    ("16384xskew12:4:g:0", 13, "P12"),
    ("16384xskew16:4:g:0", 10, "="),
    ("16384xskew16:4:g:1", 10, "="),
    ("16384xskew16:8:g:0", 5, "="),
    ("16384xskew24:4:g:0", 7, "="),
    ("16384xskew2:16:g:0", 9, "="),
    ("16384xskew2:4:g:0", 38, "P22"),
    ("16384xskew2:8:g:0", 19, "="),
    ("16384xskew32:4:g:0", 5, "="),
    ("16384xskew32:8:g:0", 3, "="),
    ("16384xskew3:4:g:0", 38, "P15"),
    ("16384xskew3:8:g:0", 19, "P15"),
    ("16384xskew4:16:g:0", 9, "="),
    ("16384xskew4:4:g:0", 38, "P12"),
    ("16384xskew5:4:g:0", 31, "P12"),
    ("16384xskew64:4:g:0", 3, "="),
    ("16384xskew6:4:g:0", 26, "P12"),
    ("16384xskew7:4:g:0", 22, "P12"),
    ("16384xskew8:4:g:0", 19, "P12"),
    ("20480x16:4:g:0", 10, "="),
    ("20480x32:4:g:0", 5, "="),
    ("20480xskew16:4:g:0", 10, "="),
    ("20480xskew32:4:g:0", 5, "="),
    ("24576x16:4:g:0", 10, "="),
    ("24576x24:4:g:0", 7, "="),
    ("24576x3:4:g:0", 38, "P22"),
    ("24576x8:4:g:0", 19, "P18"),
    ("24576xskew16:4:g:0", 10, "="),
    ("24576xskew24:4:g:0", 7, "="),
    ("24576xskew3:4:g:0", 38, "P22"),
    ("24576xskew8:4:g:0", 19, "P18"),
    ("32768x16:4:g:0", 10, "="),
    ("32768x2:4:g:0", 38, "="),
    ("32768x32:4:g:0", 5, "="),
    ("32768x3:4:g:0", 38, "P29"),
    ("32768x8:4:g:0", 19, "="),
    ("32768xskew2:4:g:0", 38, "="),
    ("32768xskew32:4:g:0", 5, "="),
    ("49152x16:4:g:0", 10, "="),
    ("49152x8:4:g:0", 19, "="),
    ("57344x16:4:g:0", 10, "="),
    ("57344x6:4:g:0", 26, "="),
    ("57344xskew16:4:g:0", 10, "="),
    ("65536x16:16:g:0", 3, "="),
    ("65536x16:4:g:0", 10, "="),
    ("65536x2:4:g:0", 38, "="),
    ("65536x3:16:g:0", 9, "="),
    ("65536x3:4:g:0", 38, "="),
    ("65536x3:8:g:0", 19, "="),
    ("65536x64:4:g:0", 3, "="),
    ("65536x8:4:g:0", 19, "="),
    ("65536xskew16:4:g:0", 10, "="),
    ("65536xskew4:4:g:0", 38, "="),
    ("131072x16:4:g:0", 10, "="),
    ("131072x3:4:g:0", 38, "="),
    ("131072x8:4:g:0", 19, "="),
    ("239718x3:4:g:0", 38, "="),
    ("261393x4:4:g:0", 38, "="),
    ("261393xskew8:4:g:0", 19, "="),
    ("262144x16:4:g:0", 10, "="),
)
MEASURED_KINDS = {"P": 46, "=": 85}

# The same rows on B200 (``sm_100a``, 148 SMs; 16 heads take the serial route
# there, and no band: its 4-head rows of at most 4096 tokens take the
# token-aware plan) and on VR200 (``sm_107a``, 212 SMs; the 4-head rows of at
# most 4096 tokens and the 16-head pack of 12288 tokens keep the occupancy plan
# there, bands of ``occupancy_bands_by_arch``), in the columns of
# :data:`MEASURED_ROWS`.
MEASURED_ROWS_B200 = (
    ("2048x1:4:g:0", 37, "P12"),
    ("2048x3:16:g:0", None, "="),
    ("4096x16:4:g:0", 10, "="),
    ("4096x1:4:g:0", 37, "P12"),
    ("4096x2:4:g:0", 37, "P12"),
    ("4096x2:8:g:0", 18, "P12"),
    ("4096x3:4:g:0", 37, "P12"),
    ("4096x4:16:g:0", None, "="),
    ("4096x8:4:g:0", 19, "P12"),
    ("4096x8:8:g:0", 10, "="),
    ("4096xskew16:4:g:0", 10, "="),
    ("4096xskew3:4:g:0", 37, "P12"),
    ("4096xskew8:4:g:0", 19, "P12"),
    ("8192x12:8:g:0", 7, "="),
    ("8192x16:4:g:0", 10, "="),
    ("8192x1:4:g:0", 37, "P22"),
    ("8192x2:16:g:0", None, "="),
    ("8192x2:4:g:0", 37, "P12"),
    ("8192x3:4:g:0", 37, "P12"),
    ("8192x3:8:g:0", 18, "P12"),
    ("8192x4:8:g:0", 18, "P12"),
    ("8192x8:4:g:0", 19, "P12"),
    ("8192x8:8:g:0", 10, "="),
    ("8192xskew12:8:g:0", 7, "="),
    ("8192xskew16:4:g:0", 10, "="),
    ("8192xskew3:4:g:0", 37, "P12"),
    ("8192xskew3:8:g:0", 18, "P12"),
    ("8192xskew8:4:g:0", 19, "P12"),
    ("8192xskew8:8:g:0", 10, "="),
    ("12288x16:4:g:0", 10, "="),
    ("12288x1:4:g:0", 37, "P33"),
    ("12288x2:4:g:0", 37, "P17"),
    ("12288x3:16:g:0", None, "="),
    ("12288x6:8:g:0", 13, "P12"),
    ("12288x8:4:g:0", 19, "P12"),
    ("12288xskew16:4:g:0", 10, "="),
    ("12288xskew2:4:g:0", 37, "P17"),
    ("12288xskew6:8:g:0", 13, "P12"),
    ("12288xskew8:4:g:0", 19, "P12"),
    ("14336x16:4:g:0", 10, "="),
    ("14336xskew16:4:g:0", 10, "="),
    ("16384x10:4:g:0", 15, "P12"),
    ("16384x12:4:g:0", 13, "P12"),
    ("16384x16:16:g:0", None, "="),
    ("16384x16:4:g:0", 10, "="),
    ("16384x16:4:g:1", 10, "="),
    ("16384x16:4:raw:0", 10, "="),
    ("16384x16:8:g:0", 5, "="),
    ("16384x1:4:g:0", 37, "="),
    ("16384x24:4:g:0", 7, "="),
    ("16384x2:16:g:0", None, "="),
    ("16384x2:4:g:0", 37, "P22"),
    ("16384x2:8:g:0", 18, "="),
    ("16384x32:4:g:0", 5, "="),
    ("16384x32:8:g:0", 3, "="),
    ("16384x3:16:g:0", None, "="),
    ("16384x3:4:g:0", 37, "P15"),
    ("16384x3:4:g:1", 37, "P15"),
    ("16384x3:4:raw:0", 37, "P15"),
    ("16384x3:64:g:0", None, "="),
    ("16384x3:8:g:0", 18, "P15"),
    ("16384x4:16:g:0", None, "="),
    ("16384x4:4:g:0", 37, "P12"),
    ("16384x5:4:g:0", 30, "P12"),
    ("16384x64:4:g:0", 3, "="),
    ("16384x6:4:g:0", 25, "P12"),
    ("16384x7:4:g:0", 22, "P12"),
    ("16384x8:4:g:0", 19, "P12"),
    ("16384x8:4:g:1", 19, "P12"),
    ("16384x8:4:raw:0", 19, "P12"),
    ("16384xskew10:4:g:0", 15, "P12"),
    ("16384xskew12:4:g:0", 13, "P12"),
    ("16384xskew16:4:g:0", 10, "="),
    ("16384xskew16:4:g:1", 10, "="),
    ("16384xskew16:8:g:0", 5, "="),
    ("16384xskew24:4:g:0", 7, "="),
    ("16384xskew2:16:g:0", None, "="),
    ("16384xskew2:4:g:0", 37, "P22"),
    ("16384xskew2:8:g:0", 18, "="),
    ("16384xskew32:4:g:0", 5, "="),
    ("16384xskew32:8:g:0", 3, "="),
    ("16384xskew3:4:g:0", 37, "P15"),
    ("16384xskew3:8:g:0", 18, "P15"),
    ("16384xskew4:16:g:0", None, "="),
    ("16384xskew4:4:g:0", 37, "P12"),
    ("16384xskew5:4:g:0", 30, "P12"),
    ("16384xskew64:4:g:0", 3, "="),
    ("16384xskew6:4:g:0", 25, "P12"),
    ("16384xskew7:4:g:0", 22, "P12"),
    ("16384xskew8:4:g:0", 19, "P12"),
    ("20480x16:4:g:0", 10, "="),
    ("20480x32:4:g:0", 5, "="),
    ("20480xskew16:4:g:0", 10, "="),
    ("20480xskew32:4:g:0", 5, "="),
    ("24576x16:4:g:0", 10, "="),
    ("24576x24:4:g:0", 7, "="),
    ("24576x3:4:g:0", 37, "P22"),
    ("24576x8:4:g:0", 19, "P18"),
    ("24576xskew16:4:g:0", 10, "="),
    ("24576xskew24:4:g:0", 7, "="),
    ("24576xskew3:4:g:0", 37, "P22"),
    ("24576xskew8:4:g:0", 19, "P18"),
    ("32768x16:4:g:0", 10, "="),
    ("32768x2:4:g:0", 37, "="),
    ("32768x32:4:g:0", 5, "="),
    ("32768x3:4:g:0", 37, "P29"),
    ("32768x8:4:g:0", 19, "="),
    ("32768xskew2:4:g:0", 37, "="),
    ("32768xskew32:4:g:0", 5, "="),
    ("49152x16:4:g:0", 10, "="),
    ("49152x8:4:g:0", 19, "="),
    ("57344x16:4:g:0", 10, "="),
    ("57344x6:4:g:0", 25, "="),
    ("57344xskew16:4:g:0", 10, "="),
    ("65536x16:16:g:0", None, "="),
    ("65536x16:4:g:0", 10, "="),
    ("65536x2:4:g:0", 37, "="),
    ("65536x3:16:g:0", None, "="),
    ("65536x3:4:g:0", 37, "="),
    ("65536x3:8:g:0", 18, "="),
    ("65536x64:4:g:0", 3, "="),
    ("65536x8:4:g:0", 19, "="),
    ("65536xskew16:4:g:0", 10, "="),
    ("65536xskew4:4:g:0", 37, "="),
    ("131072x16:4:g:0", 10, "="),
    ("131072x3:4:g:0", 37, "="),
    ("131072x8:4:g:0", 19, "="),
    ("239718x3:4:g:0", 37, "="),
    ("261393x4:4:g:0", 37, "="),
    ("261393xskew8:4:g:0", 19, "="),
    ("262144x16:4:g:0", 10, "="),
)
MEASURED_ROWS_VR200 = (
    ("2048x1:4:g:0", 38, "="),
    ("2048x3:16:g:0", 13, "P12"),
    ("4096x16:4:g:0", 14, "="),
    ("4096x1:4:g:0", 38, "="),
    ("4096x2:4:g:0", 53, "="),
    ("4096x2:8:g:0", 26, "P12"),
    ("4096x3:4:g:0", 53, "="),
    ("4096x4:16:g:0", 13, "P12"),
    ("4096x8:4:g:0", 27, "="),
    ("4096x8:8:g:0", 14, "P12"),
    ("4096xskew16:4:g:0", 14, "="),
    ("4096xskew3:4:g:0", 53, "="),
    ("4096xskew8:4:g:0", 27, "="),
    ("8192x12:8:g:0", 9, "="),
    ("8192x16:4:g:0", 14, "P12"),
    ("8192x1:4:g:0", 38, "P22"),
    ("8192x2:16:g:0", 13, "P12"),
    ("8192x2:4:g:0", 53, "P12"),
    ("8192x3:4:g:0", 53, "P12"),
    ("8192x3:8:g:0", 26, "P12"),
    ("8192x4:8:g:0", 26, "P12"),
    ("8192x8:4:g:0", 27, "P12"),
    ("8192x8:8:g:0", 14, "P12"),
    ("8192xskew12:8:g:0", 9, "="),
    ("8192xskew16:4:g:0", 14, "P12"),
    ("8192xskew3:4:g:0", 53, "P12"),
    ("8192xskew3:8:g:0", 26, "P12"),
    ("8192xskew8:4:g:0", 27, "P12"),
    ("8192xskew8:8:g:0", 14, "P12"),
    ("12288x16:4:g:0", 14, "P12"),
    ("12288x1:4:g:0", 38, "P33"),
    ("12288x2:4:g:0", 53, "P17"),
    ("12288x3:16:g:0", 13, "="),
    ("12288x6:8:g:0", 18, "P12"),
    ("12288x8:4:g:0", 27, "P12"),
    ("12288xskew16:4:g:0", 14, "P12"),
    ("12288xskew2:4:g:0", 53, "P17"),
    ("12288xskew6:8:g:0", 18, "P12"),
    ("12288xskew8:4:g:0", 27, "P12"),
    ("14336x16:4:g:0", 14, "P12"),
    ("14336xskew16:4:g:0", 14, "P12"),
    ("16384x10:4:g:0", 22, "P12"),
    ("16384x12:4:g:0", 18, "P12"),
    ("16384x16:16:g:0", 4, "="),
    ("16384x16:4:g:0", 14, "P12"),
    ("16384x16:4:g:1", 14, "P12"),
    ("16384x16:4:raw:0", 14, "P12"),
    ("16384x16:8:g:0", 7, "="),
    ("16384x1:4:g:0", 38, "="),
    ("16384x24:4:g:0", 9, "="),
    ("16384x2:16:g:0", 13, "="),
    ("16384x2:4:g:0", 53, "P22"),
    ("16384x2:8:g:0", 26, "P22"),
    ("16384x32:4:g:0", 7, "="),
    ("16384x32:8:g:0", 4, "="),
    ("16384x3:16:g:0", 13, "="),
    ("16384x3:4:g:0", 53, "P15"),
    ("16384x3:4:g:1", 53, "P15"),
    ("16384x3:4:raw:0", 53, "P15"),
    ("16384x3:64:g:0", None, "="),
    ("16384x3:8:g:0", 26, "P15"),
    ("16384x4:16:g:0", 13, "P12"),
    ("16384x4:4:g:0", 53, "P12"),
    ("16384x5:4:g:0", 43, "P12"),
    ("16384x64:4:g:0", 4, "="),
    ("16384x6:4:g:0", 36, "P12"),
    ("16384x7:4:g:0", 31, "P12"),
    ("16384x8:4:g:0", 27, "P12"),
    ("16384x8:4:g:1", 27, "P12"),
    ("16384x8:4:raw:0", 27, "P12"),
    ("16384xskew10:4:g:0", 22, "P12"),
    ("16384xskew12:4:g:0", 18, "P12"),
    ("16384xskew16:4:g:0", 14, "P12"),
    ("16384xskew16:4:g:1", 14, "P12"),
    ("16384xskew16:8:g:0", 7, "="),
    ("16384xskew24:4:g:0", 9, "="),
    ("16384xskew2:16:g:0", 13, "="),
    ("16384xskew2:4:g:0", 53, "P22"),
    ("16384xskew2:8:g:0", 26, "P22"),
    ("16384xskew32:4:g:0", 7, "="),
    ("16384xskew32:8:g:0", 4, "="),
    ("16384xskew3:4:g:0", 53, "P15"),
    ("16384xskew3:8:g:0", 26, "P15"),
    ("16384xskew4:16:g:0", 13, "P12"),
    ("16384xskew4:4:g:0", 53, "P12"),
    ("16384xskew5:4:g:0", 43, "P12"),
    ("16384xskew64:4:g:0", 4, "="),
    ("16384xskew6:4:g:0", 36, "P12"),
    ("16384xskew7:4:g:0", 31, "P12"),
    ("16384xskew8:4:g:0", 27, "P12"),
    ("20480x16:4:g:0", 14, "="),
    ("20480x32:4:g:0", 7, "="),
    ("20480xskew16:4:g:0", 14, "="),
    ("20480xskew32:4:g:0", 7, "="),
    ("24576x16:4:g:0", 14, "="),
    ("24576x24:4:g:0", 9, "="),
    ("24576x3:4:g:0", 53, "P22"),
    ("24576x8:4:g:0", 27, "P18"),
    ("24576xskew16:4:g:0", 14, "="),
    ("24576xskew24:4:g:0", 9, "="),
    ("24576xskew3:4:g:0", 53, "P22"),
    ("24576xskew8:4:g:0", 27, "P18"),
    ("32768x16:4:g:0", 14, "="),
    ("32768x2:4:g:0", 53, "P43"),
    ("32768x32:4:g:0", 7, "="),
    ("32768x3:4:g:0", 53, "P29"),
    ("32768x8:4:g:0", 27, "P24"),
    ("32768xskew2:4:g:0", 53, "P43"),
    ("32768xskew32:4:g:0", 7, "="),
    ("49152x16:4:g:0", 14, "="),
    ("49152x8:4:g:0", 27, "="),
    ("57344x16:4:g:0", 14, "="),
    ("57344x6:4:g:0", 36, "="),
    ("57344xskew16:4:g:0", 14, "="),
    ("65536x16:16:g:0", 4, "="),
    ("65536x16:4:g:0", 14, "="),
    ("65536x2:4:g:0", 53, "="),
    ("65536x3:16:g:0", 13, "="),
    ("65536x3:4:g:0", 53, "="),
    ("65536x3:8:g:0", 26, "="),
    ("65536x64:4:g:0", 4, "="),
    ("65536x8:4:g:0", 27, "="),
    ("65536xskew16:4:g:0", 14, "="),
    ("65536xskew4:4:g:0", 53, "P47"),
    ("131072x16:4:g:0", 14, "="),
    ("131072x3:4:g:0", 53, "="),
    ("131072x8:4:g:0", 27, "="),
    ("239718x3:4:g:0", 53, "="),
    ("261393x4:4:g:0", 53, "="),
    ("261393xskew8:4:g:0", 27, "="),
    ("262144x16:4:g:0", 14, "="),
)
# Every entry of the envelope, ``(architecture, SM count)``: its rows and the
# count of each plan kind among them.
MEASURED_BY_ENVELOPE = {
    (ARCH, SM): (MEASURED_ROWS, MEASURED_KINDS),
    ("sm_100a", 148): (MEASURED_ROWS_B200, {"P": 53, "=": 78}),
    ("sm_107a", 212): (MEASURED_ROWS_VR200, {"P": 71, "=": 60}),
}
ENVELOPE_IDS = tuple(f"{arch}-{sm}" for arch, sm in MEASURED_BY_ENVELOPE)


def _row(spec: str) -> tuple[int, int, int, str, str]:
    """``(S, num_docs, H, gate, l2n)`` of a :data:`MEASURED_ROWS` row."""
    tokens, rest = spec.split("x")
    docs, heads, gate, l2n = rest.split(":")
    return (
        int(tokens),
        int(docs.removeprefix("skew")),
        int(heads),
        gate,
        "fused" if l2n == "1" else "off",
    )


def _check_records(jit, p, gate: str, l2n: str, num_tokens: int | None, tag) -> None:
    """Every record of plan ``p`` is the one ``cake_jit.select_record`` selects
    at the plan's own segment count: given the pack's token count
    ``num_tokens`` for the token-aware rule, ``None`` for the occupancy rule
    (the stages whose rules read the walk CTA count, ``num_segments * H``,
    change with the count).  With an in-kernel Q/K L2 norm mode no gate
    epilogue runs."""
    for path, by_stage in p.records.items():
        for stage, name in by_stage.items():
            if l2n != "off" and stage == "gate_epilogue":
                assert name is None, (tag, path, stage)
                continue
            want = jit.select_record(
                stage,
                num_chunks_pad=p.NC_pad,
                num_seqs=p.N,
                num_heads=p.H,
                sm_count=p.sm_count,
                arch=p.arch,
                path=path,
                gate=gate,
                l2n=l2n,
                num_tokens=num_tokens,
            )
            assert name == want, (tag, path, stage, name, want)


def _expected_route(N: int, H: int, p_old: int | None, arm: str):
    """The token-aware route of a measured row from its plan columns."""
    if p_old is None:
        return ("serial", 1, 0, 0)
    count = p_old if arm == "=" else int(arm.removeprefix("P"))
    return ("seg", count, N * count, int(N * H <= 4))


def test_the_rule_constants() -> None:
    """The constants of the token-aware rule, the same in both packages; its
    saving threshold is the backend's size rule of the light set's segment
    states, and the plans take the rule by default."""
    for name in PACKAGES:
        jit, backend = JIT[name], BACKEND[name]
        assert jit.TOKEN_SEGMENTS == {
            "sm_counts_by_arch": {
                "sm_100a": [148],
                "sm_103a": [152],
                "sm_107a": [212],
            },
            "heads": [4, 8, 16],
            "save_min_tokens_per_seg": 384,
            "min_uniform_segments": 12,
            "max_segment_tokens": 1408,
            "occupancy_bands_by_arch": {
                "sm_103a": [{"heads": 4, "tokens_above": -1, "tokens_below": 4097}],
                "sm_107a": [
                    {"heads": 16, "tokens_above": 8192, "tokens_below": 16384},
                    {"heads": 4, "tokens_above": -1, "tokens_below": 4097},
                ],
            },
        }
        assert set(MEASURED_BY_ENVELOPE) == {
            (arch, sm)
            for arch, sms in jit.TOKEN_SEGMENTS["sm_counts_by_arch"].items()
            for sm in sms
        }
        assert (
            jit.TOKEN_SEGMENTS["save_min_tokens_per_seg"]
            == backend.SEG_SAVE_MIN_TOKENS_PER_SEG
        )
        assert backend.SEGMENT_RULES == ("tokens", "occupancy")
        assert "TOKEN_SEGMENTS" in jit.__all__ and "segment_route" in jit.__all__
        assert "SEGMENT_RULES" in backend.__all__
        S, N, H = sum(DOCS3_LENGTHS), len(DOCS3_LENGTHS), DOCS3_HEADS
        assert backend.plan(S, N, H, SM, ARCH) is backend.plan(
            S, N, H, SM, ARCH, segment_rule="tokens"
        )
        q = {"sm_count": SM, "arch": ARCH, "segment_rule": "none"}
        with pytest.raises(ValueError, match="segment_rule"):
            backend.plan(S, N, H, SM, ARCH, segment_rule="tokens_and_more")
        with pytest.raises(ValueError, match="segment_rule"):
            backend.saved_bytes(S, H, N, **q)
        with pytest.raises(ValueError, match="segment_rule"):
            backend.workspace_bytes(S, N, H, **q)
        with pytest.raises(ValueError, match="segment_rule"):
            backend.memory_report(S, N, H, **q)
        with pytest.raises(ValueError, match="num_tokens"):
            jit.segment_route(N, H, SM, ARCH, num_tokens=-1)
        # outside the envelope (here ``sm_100a`` with 152 SMs) the token count
        # is not read: the calls are the occupancy rule's, a negative count
        # included
        assert jit.segment_route(
            N, H, 152, "sm_100a", num_tokens=-1
        ) == jit.segment_route(N, H, 152, "sm_100a")
        assert backend.tables_bytes(
            -64, 1, H, sm_count=152, arch="sm_100a"
        ) == backend.tables_bytes(
            -64, 1, H, sm_count=152, arch="sm_100a", segment_rule="occupancy"
        )
        with pytest.raises(ValueError, match="must all be >= 1"):
            jit.segment_route(0, H, SM, ARCH, num_tokens=S)
    assert JIT[PACKAGES[0]].TOKEN_SEGMENTS == JIT[PACKAGES[1]].TOKEN_SEGMENTS
    assert JIT[PACKAGES[0]].SEGMENTS == JIT[PACKAGES[1]].SEGMENTS


@pytest.mark.parametrize(("arch", "sm"), tuple(MEASURED_BY_ENVELOPE), ids=ENVELOPE_IDS)
@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_the_measured_rows_take_their_plans(name, arch, sm) -> None:
    """Every measured row on every entry of the envelope (``arch`` with ``sm``
    SMs): the occupancy rule gives the row's segments per
    document (the shipped plan), the token-aware rule the row's plan, in
    ``segment_route`` / ``segment_plan`` / ``segment_args`` and in the backend's
    plan of the row's gate and l2n modes (the occupancy rule's plan is the
    shipped one).  Every record of both plans is the one selected at the plan's
    own segment count (:func:`_check_records`), and on some rows the two plans
    select different records."""
    jit, backend = JIT[name], BACKEND[name]
    rows, want_kinds = MEASURED_BY_ENVELOPE[(arch, sm)]
    kinds = Counter(arm[0] for _spec, _p_old, arm in rows)
    assert len(rows) == len({spec for spec, *_ in rows}) == 131
    assert [spec for spec, *_ in rows] == [spec for spec, *_ in MEASURED_ROWS]
    assert kinds == want_kinds
    flips = 0
    for spec, p_old, arm in rows:
        S, N, H, gate, l2n = _row(spec)
        occupancy = jit.segment_route(N, H, sm, arch)
        if p_old is None:
            assert occupancy == ("serial", 1, 0, 0), spec
        else:
            assert occupancy == ("seg", p_old, N * p_old, int(N * H <= 4)), spec
        assert jit.segment_plan(N, H, sm, arch) == occupancy[:3], spec
        route = jit.segment_route(N, H, sm, arch, num_tokens=S)
        assert route == _expected_route(N, H, p_old, arm), spec
        assert jit.segment_plan(N, H, sm, arch, num_tokens=S) == route[:3], spec
        proportional = route[3]
        assert jit.segment_args(N, H, sm, arch, num_tokens=S) == (
            proportional,
            -N if proportional else route[1],
        ), spec
        records = {}
        for rule, want in (("tokens", route), ("occupancy", occupancy)):
            p = backend.plan(S, N, H, sm, arch, gate=gate, l2n=l2n, segment_rule=rule)
            assert (p.recur, p.P, p.NSEG, p.proportional) == want, (spec, rule)
            assert p.seg_arg == (-N if want[3] else want[1]), (spec, rule)
            tokens = S if rule == "tokens" else None
            _check_records(jit, p, gate, l2n, tokens, (spec, rule))
            records[rule] = p.records
        if arm == "=":
            assert records["tokens"] == records["occupancy"], spec
        flips += records["tokens"] != records["occupancy"]
    assert flips > 0


@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_the_records_follow_the_plan_count(name) -> None:
    """A pack of 16384 tokens in three documents at 4 heads on GB300: the
    forward's segment carry is the record its rule selects at the plan's walk
    CTA count, as on the measured arms.  The token-aware plan's 45 segments are
    180 walk CTAs, below three waves of 152 SMs: ``seg_carry``; the occupancy
    plan's 114 segments are 456, three waves: the overlap carry
    ``seg_carry_ov``."""
    backend = BACKEND[name]
    S, N, H = sum(DOCS3_LENGTHS), len(DOCS3_LENGTHS), DOCS3_HEADS
    tokens = backend.plan(S, N, H, SM, ARCH)
    occupancy = backend.plan(S, N, H, SM, ARCH, segment_rule="occupancy")
    assert (tokens.P, tokens.NSEG, occupancy.P, occupancy.NSEG) == (15, 45, 38, 114)
    assert tokens.NSEG * H < 3 * SM <= occupancy.NSEG * H
    assert tokens.record("seg_carry", "fwd") == "seg_carry"
    assert occupancy.record("seg_carry", "fwd") == "seg_carry_ov"


def test_the_bounds_of_each_branch() -> None:
    """The rule at the bounds of its branches (GB300, both packages): a pack
    whose occupancy plan saves the segment states keeps that plan; a short pack
    takes the uniform table of ``P_u = max(P_ns, 12, ceil(S / 1408))`` segments
    per document below ``P_old`` (proportional only at most four chains, as the
    occupancy plan's table), else the occupancy plan; the packs of a band keep
    the occupancy plan (on GB300 the 4-head packs of at most 4096 tokens); the
    occupancy plan outside the envelope (other architectures, SM counts and head
    counts, the serial route, no token count); the rule and its bands on the
    B200 and VR200 entries of the envelope."""
    cases = (
        # (S, N, H, (route, segments, num_segments, proportional))
        # packs whose occupancy plan saves the states: the occupancy plan
        (29184, 2, 4, ("seg", 38, 76, 0)),  # the threshold 384 * 2 * 38
        (65536, 4, 4, ("seg", 38, 152, 0)),
        (65536, 5, 4, ("seg", 31, 155, 0)),
        (16384, 1, 4, ("seg", 38, 38, 1)),  # four chains: proportional
        (7296, 1, 8, ("seg", 19, 19, 0)),
        (8192, 2, 16, ("seg", 9, 18, 0)),
        (12288, 3, 16, ("seg", 9, 27, 0)),
        (sum(ISSUE_LENGTHS), len(ISSUE_LENGTHS), ISSUE_HEADS, ("seg", 38, 152, 0)),
        # short packs: P_ns = S // (384 N) + 1 binds below P_old
        (28415, 2, 4, ("seg", 37, 74, 0)),  # P_ns = 37
        (28416, 2, 4, ("seg", 38, 76, 0)),  # P_ns = 38 = P_old
        (29183, 2, 4, ("seg", 38, 76, 0)),
        (14591, 1, 4, ("seg", 38, 38, 1)),  # P_ns = 38 = P_old
        (4607, 1, 8, ("seg", 12, 12, 0)),  # the floor of 12 segments
        (4608, 1, 8, ("seg", 13, 13, 0)),  # P_ns = 13
        (12288, 6, 8, ("seg", 12, 72, 0)),
        (12288, 2, 4, ("seg", 17, 34, 0)),
        # segments of at most 1408 tokens: ceil(S / 1408) binds
        (16896, 8, 4, ("seg", 12, 96, 0)),
        (16897, 8, 4, ("seg", 13, 104, 0)),
        (25344, 8, 4, ("seg", 18, 144, 0)),
        (25345, 8, 4, ("seg", 19, 152, 0)),  # 19 = P_old
        # GB300's band: 4 heads and at most 4096 tokens keep the occupancy plan
        # (proportional at four chains); above it and at 8 heads the rule decides
        (0, 3, 4, ("seg", 38, 114, 0)),  # an empty pack
        (1, 1, 4, ("seg", 38, 38, 1)),
        (64, 3, 4, ("seg", 38, 114, 0)),
        (4096, 1, 4, ("seg", 38, 38, 1)),
        (4096, 2, 4, ("seg", 38, 76, 0)),  # case E of the raw-gate accuracy test
        (4096, 12, 4, ("seg", 13, 156, 0)),  # P_old 13, the band's last N
        (4097, 1, 4, ("seg", 12, 12, 1)),  # four chains: the proportional table
        (4097, 2, 4, ("seg", 12, 24, 0)),
        (4096, 2, 8, ("seg", 12, 24, 0)),  # P_old 19
        (16384, 3, 4, ("seg", 15, 45, 0)),
        # P_u at or above P_old: the occupancy plan (no proportional count)
        (16384, 13, 4, ("seg", 12, 156, 0)),
        (4096, 4, 16, ("seg", 9, 36, 0)),
        (2048, 3, 16, ("seg", 9, 27, 0)),
        (16384, 16, 4, ("seg", 10, 160, 0)),
        (4096, 16, 4, ("seg", 10, 160, 0)),
        (8192, 1, 16, ("seg", 9, 9, 0)),
    )
    for name in PACKAGES:
        jit = JIT[name]
        for S, N, H, want in cases:
            assert jit.segment_route(N, H, SM, ARCH, num_tokens=S) == want, (S, N, H)
            assert jit.segment_args(N, H, SM, ARCH, num_tokens=S) == (
                want[3],
                -N if want[3] else want[1],
            ), (S, N, H)
        # outside the envelope the occupancy rule decides
        assert jit.segment_route(3, 64, SM, ARCH, num_tokens=16384) == (
            "serial",
            1,
            0,
            0,
        )
        assert jit.segment_route(3, 32, SM, ARCH, num_tokens=16384)[0] == "serial"
        assert jit.segment_route(3, 2, SM, ARCH, num_tokens=16384) == (
            "seg",
            64,
            192,
            0,
        )
        # the other entries of the envelope: B200 (148 SMs; 16 heads take the
        # serial route there) and VR200 (212 SMs)
        for arch, sm, S, N, H, want in (
            ("sm_100a", 148, 16384, 3, 4, ("seg", 15, 45, 0)),  # P_old 37
            ("sm_100a", 148, 4096, 1, 4, ("seg", 12, 12, 1)),  # P_old 37
            # no band on B200: 4 heads and at most 4096 tokens take the rule
            ("sm_100a", 148, 0, 3, 4, ("seg", 12, 36, 0)),  # P_old 37
            ("sm_100a", 148, 1, 1, 4, ("seg", 12, 12, 1)),
            ("sm_100a", 148, 64, 3, 4, ("seg", 12, 36, 0)),
            ("sm_100a", 148, 4096, 2, 4, ("seg", 12, 24, 0)),  # P_old 37
            ("sm_100a", 148, 16384, 16, 4, ("seg", 10, 160, 0)),  # P_u 12 >= 10
            ("sm_100a", 148, 16384, 3, 16, ("serial", 1, 0, 0)),
            ("sm_107a", 212, 16384, 3, 4, ("seg", 15, 45, 0)),  # P_old 53
            ("sm_107a", 212, 16384, 16, 4, ("seg", 12, 192, 0)),  # P_old 14
            ("sm_107a", 212, 65536, 4, 4, ("seg", 47, 188, 0)),  # P_old 53
            # VR200's 4-head band: at most 4096 tokens keep the occupancy plan;
            # above it and at 8 heads the rule decides
            ("sm_107a", 212, 0, 3, 4, ("seg", 53, 159, 0)),
            ("sm_107a", 212, 4096, 1, 4, ("seg", 38, 38, 1)),
            ("sm_107a", 212, 4096, 2, 4, ("seg", 53, 106, 0)),  # case E
            ("sm_107a", 212, 4096, 17, 4, ("seg", 13, 221, 0)),  # the last N
            ("sm_107a", 212, 4097, 2, 4, ("seg", 12, 24, 0)),  # P_old 53
            ("sm_107a", 212, 4096, 2, 8, ("seg", 12, 24, 0)),  # P_old 26
            # VR200's 16-head band: 16 heads and 8192 < S < 16384 keep the
            # occupancy plan (P_old 13); at its bounds and at 8 heads the rule
            # decides
            ("sm_107a", 212, 8192, 3, 16, ("seg", 12, 36, 0)),
            ("sm_107a", 212, 8193, 3, 16, ("seg", 13, 39, 0)),
            ("sm_107a", 212, 12288, 3, 16, ("seg", 13, 39, 0)),
            ("sm_107a", 212, 16383, 4, 16, ("seg", 13, 52, 0)),
            ("sm_107a", 212, 16384, 4, 16, ("seg", 12, 48, 0)),
            ("sm_107a", 212, 12288, 3, 8, ("seg", 12, 36, 0)),  # P_old 26
        ):
            assert jit.segment_route(N, H, sm, arch, num_tokens=S) == want, (
                arch,
                sm,
                S,
                N,
                H,
            )
        for arch, sm in (
            ("sm_100a", SM),
            ("sm_107a", SM),
            ("sm_107a", 216),
            ("sm_100a", 212),
            (None, SM),
            (ARCH, 148),
            (ARCH, 160),
        ):
            for S, N, H in ((16384, 3, 4), (16896, 8, 4), (4096, 1, 4)):
                assert jit.segment_route(
                    N, H, sm, arch, num_tokens=S
                ) == jit.segment_route(N, H, sm, arch), (arch, sm, S, N, H)
        assert jit.segment_route(3, 4, SM, ARCH, num_tokens=None) == (
            "seg",
            38,
            114,
            0,
        )


def test_the_rule_never_grows_the_plan() -> None:
    """Over a grid of scalars (every architecture and none, six SM counts, ten
    head counts, document counts up to 300, packs at and around the size rule's
    and the segment bound's thresholds): the token-aware route and block table
    form are the occupancy rule's, its segment count is at most the occupancy
    count, the light set saves the segment states exactly where the occupancy
    plan's does, and outside the envelope and in a band of
    ``occupancy_bands_by_arch`` the occupancy plan itself; the two packages
    agree everywhere."""
    det, nondet = JIT[PACKAGES[0]], JIT[PACKAGES[1]]
    rule = BACKEND[PACKAGES[0]].seg_states_saved
    per_seg = det.TOKEN_SEGMENTS["save_min_tokens_per_seg"]
    changed = banded = 0
    envelope = det.TOKEN_SEGMENTS["sm_counts_by_arch"]
    bands = det.TOKEN_SEGMENTS["occupancy_bands_by_arch"]
    sm_counts = (132, 148, 152, 160, 212, 216)
    for arch, sm in itertools.product((*det.ARCHES, None), sm_counts):
        served = arch is not None and sm in envelope.get(arch, ())
        for H in (1, 2, 3, 4, 5, 8, 12, 16, 32, 64):
            for N in (*range(1, 18), 31, 64, 100, 257, 300):
                occupancy = det.segment_route(N, H, sm, arch)
                packs = {0, 1, 63, 1000, 4096, 16384, 65536, 262144}
                if occupancy[0] == "seg":
                    for k in (1, 2, 3, occupancy[1] - 1, occupancy[1]):
                        packs |= {per_seg * N * k + d for d in (-1, 0, 1)}
                for j in (12, 13, 40):
                    packs |= {1408 * j + d for d in (-1, 0, 1)}
                for S in sorted(x for x in packs if x >= 0):
                    route = det.segment_route(N, H, sm, arch, num_tokens=S)
                    key = (arch, sm, H, N, S)
                    assert nondet.segment_route(N, H, sm, arch, num_tokens=S) == route
                    assert route[0] == occupancy[0], key
                    assert route[3] == occupancy[3], key
                    assert route[2] <= occupancy[2], key
                    assert route[0] == "serial" or route[2] == N * route[1], key
                    assert rule(S, route[2]) == rule(S, occupancy[2]), key
                    if route != occupancy:
                        assert served and H in det.TOKEN_SEGMENTS["heads"], key
                        assert route[1] >= det.TOKEN_SEGMENTS["min_uniform_segments"]
                        changed += 1
                    if served and any(
                        H == b["heads"] and b["tokens_above"] < S < b["tokens_below"]
                        for b in bands.get(arch, ())
                    ):
                        assert route == occupancy, key
                        banded += occupancy[0] == "seg"
    assert changed > 0 and banded > 0


def _saved_state_bytes(backend, S: int, H: int, p, save_mode: str) -> int:
    """Bytes of the segment states in the saved set of plan ``p``: two FP32
    ``[128, 128]`` blocks per segment and head in the tape set, three in the
    light set where it saves them."""
    block = H * backend.D * backend.D * 4
    if save_mode == "tape":
        return 2 * p.NSEG * block
    return 3 * p.NSEG * block if backend.seg_states_saved(S, p.NSEG) else 0


@pytest.mark.parametrize(("arch", "sm"), tuple(MEASURED_BY_ENVELOPE), ids=ENVELOPE_IDS)
@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_the_memory_of_the_measured_rows_never_grows(name, arch, sm) -> None:
    """On every measured row (its gate and l2n modes) of every entry of the
    envelope (``arch`` with ``sm`` SMs) the token-aware plan needs
    at most the occupancy plan's bytes: the saved set and its segment states
    (light and tape), the tables, the forward-only and the backward arenas, and
    every total of the memory report, whose segment count is the plan's; the
    rows that change the plan have fewer segment blocks.  Each rule's tables are
    its own plan's, and its saved set differs from the other rule's by exactly
    their tables and segment states."""
    backend = BACKEND[name]
    rows, _kinds = MEASURED_BY_ENVELOPE[(arch, sm)]
    q = {"sm_count": sm, "arch": arch}
    occ = {**q, "segment_rule": "occupancy"}
    totals = (
        "saved",
        "saved_seg_state_bytes",
        "seg_block_bytes",
        "arena_fwd",
        "arena_bwd",
        "fwd_total_fwd_arena",
        "fwd_total_bwd_arena",
        "bwd_total",
        "replay_total",
    )
    smaller = fewer_tables = 0
    for spec, _p_old, arm in rows:
        S, N, H, gate, l2n = _row(spec)
        tokens = backend.plan(S, N, H, sm, arch, gate=gate, l2n=l2n)
        occupancy = backend.plan(
            S, N, H, sm, arch, gate=gate, l2n=l2n, segment_rule="occupancy"
        )
        mode = {"gate": gate, "l2n": l2n}
        # each rule's tables are its own plan's (the block table of its NSEG)
        tables = backend.tables_bytes(S, N, H, **q)
        tables_occ = backend.tables_bytes(S, N, H, **occ)
        assert tables == tokens.table_words() * 4, spec
        assert tables_occ == occupancy.table_words() * 4, spec
        assert tables <= tables_occ, spec
        fewer_tables += tables < tables_occ
        for save_mode in backend.SAVE_MODES:
            kw = {"save_mode": save_mode, "l2n": l2n}
            mine = backend.saved_bytes(S, H, N, **kw, **q)
            theirs = backend.saved_bytes(S, H, N, **kw, **occ)
            assert mine <= theirs, (spec, kw)
            # the two sets differ by their own plans' tables and segment states
            states = _saved_state_bytes(backend, S, H, tokens, save_mode)
            states_occ = _saved_state_bytes(backend, S, H, occupancy, save_mode)
            assert mine - theirs == tables - tables_occ + states - states_occ, (
                spec,
                kw,
            )
            for backward in (False, True):
                kw = {"backward": backward, "save_mode": save_mode, **mode}
                mine = backend.workspace_bytes(S, N, H, **kw, **q)
                theirs = backend.workspace_bytes(S, N, H, **kw, **occ)
                assert mine <= theirs, (spec, kw)
                smaller += mine < theirs
            mine = backend.memory_report(S, N, H, save_mode, **mode, **q)
            theirs = backend.memory_report(S, N, H, save_mode, **mode, **occ)
            assert mine["num_segments"] == tokens.NSEG, spec
            assert theirs["num_segments"] == occupancy.NSEG, spec
            assert mine["seg_states_saved"] == theirs["seg_states_saved"], spec
            for key in totals:
                assert mine[key] <= theirs[key], (spec, save_mode, key)
            if arm != "=":
                assert mine["num_segments"] < theirs["num_segments"], spec
                assert mine["seg_block_bytes"] < theirs["seg_block_bytes"], spec
    assert smaller > 0 and fewer_tables > 0


@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_document_groups_keep_the_whole_pack_plan(name) -> None:
    """A pack of 16384 tokens in three documents at 4 heads on GB300: the
    token-aware plan is the uniform table of 15 segments per document, which the
    batched arena option serves.  Every document group plans its documents with
    the whole pack's 15 segments and records (``Plan.batch_plan``), not with the
    rule on the group's own tokens, so the groups run the whole pack's plan; the
    option's arena bound (``Plan.batch_bound``) holds for every group, and the
    byte queries and the memory report size the same plan."""
    backend, Arena = BACKEND[name], ARENA[name]
    lengths, H = DOCS3_LENGTHS, DOCS3_HEADS
    S, N = sum(lengths), len(lengths)
    q = {"sm_count": SM, "arch": ARCH}
    p = backend.plan(S, N, H, SM, ARCH)
    assert (p.P, p.NSEG, p.proportional, p.seg_arg) == (15, 45, 0, 15)
    options = (
        Arena(batch_tokens=8192),
        Arena(batch_tokens=12288),
        Arena(lean=True, tail_passes=4, batch_tokens=12288),
        Arena(batch_tokens=S, max_docs_per_batch=1),
    )
    replanned = 0
    for option in options:
        p.check_arena(option)
        groups = p.batches(option, lengths)
        assert len(groups) >= 2, option
        assert sum(b.N for b in groups) == N and sum(b.S for b in groups) == S
        nc_pad, _rows, nseg = p.batch_bound(option)
        assert nseg == p.NSEG
        for b in groups:
            g = p.batch_plan(b)
            assert (g.P, g.NSEG, g.proportional, g.seg_arg) == (15, 15 * b.N, 0, 15)
            assert g.records is p.records and g.gsize == p.gsize
            assert b.NC_pad <= nc_pad and g.NSEG <= nseg, (option, b)
            replanned += backend.plan(b.S, b.N, H, SM, ARCH).P != p.P
        exact = p.arena_bytes(arena=option, doc_lengths=lengths)
        assert exact <= p.arena_bytes(arena=option), option
        assert (
            backend.workspace_bytes(S, N, H, arena=option, doc_lengths=lengths, **q)
            == exact
        ), option
        report = backend.memory_report(S, N, H, arena=option, doc_lengths=lengths, **q)
        assert (report["segments"], report["num_segments"]) == (15, 45), option
        assert report["arena_bwd_configured"] == exact, option
    # a group planned on its own tokens would take another count
    assert replanned > 0
    # the issue's pack saves the segment states: both rules give the occupancy
    # plan, whose uniform table the batched option serves as before
    lengths, H = ISSUE_LENGTHS, ISSUE_HEADS
    S, N = sum(lengths), len(lengths)
    issue = backend.plan(S, N, H, SM, ARCH)
    for p in (issue, backend.plan(S, N, H, SM, ARCH, segment_rule="occupancy")):
        assert (p.P, p.NSEG, p.proportional, p.seg_save) == (38, 152, 0, True)
    issue.check_arena(Arena(lean=True, tail_passes=4, batch_tokens=200000))
    # four chains: the proportional table under both rules, the batched option
    # refused as before
    for rule in backend.SEGMENT_RULES:
        four = backend.plan(65, 1, 4, SM, ARCH, segment_rule=rule)
        assert four.proportional == 1
        with pytest.raises(ValueError, match="batch_tokens"):
            four.check_arena(Arena(batch_tokens=65))


def test_det_and_nondet_choose_the_same_plan() -> None:
    """The deterministic and the non-deterministic package plan every measured
    row, the three-document pack and the issue's pack with the same route,
    segment counts and block table form under both rules, on every entry of
    the envelope."""
    det, nondet = BACKEND[PACKAGES[0]], BACKEND[PACKAGES[1]]
    rows = [_row(spec) for spec, *_ in MEASURED_ROWS]
    for lengths, H in ((DOCS3_LENGTHS, DOCS3_HEADS), (ISSUE_LENGTHS, ISSUE_HEADS)):
        rows.append((sum(lengths), len(lengths), H, "g", "off"))
    for (S, N, H, gate, l2n), rule, (arch, sm) in itertools.product(
        rows, det.SEGMENT_RULES, MEASURED_BY_ENVELOPE
    ):
        a = det.plan(S, N, H, sm, arch, gate=gate, l2n=l2n, segment_rule=rule)
        b = nondet.plan(S, N, H, sm, arch, gate=gate, l2n=l2n, segment_rule=rule)
        assert (a.recur, a.P, a.NSEG, a.proportional, a.seg_arg, a.seg_save) == (
            b.recur,
            b.P,
            b.NSEG,
            b.proportional,
            b.seg_arg,
            b.seg_save,
        ), (S, N, H, gate, l2n, rule, arch, sm)


def _context_on_cuda(name: str, sm: int):
    """Rank 0's context of a two-rank contiguous pack of two documents with
    ``sm`` SMs, built on the host and seen as a context on ``cuda:0``: the byte
    query reads the context's host fields and its ``device``."""
    kcp_plan, prefix = KCP_PLAN[name], KCP_PACKAGE[name].removeprefix("cake.")
    part = getattr(kcp_plan, f"{prefix}_partition")([0, 3000, 8192], 2)
    ctx = getattr(kcp_plan, f"{prefix}_build_context")(
        part, None, 0, "cpu", 4, sm_count=sm
    )
    assert ctx.T_r > 0

    class OnDevice:
        def __getattr__(self, attr: str):
            return getattr(ctx, attr)

    view = OnDevice()
    view.device = torch.device("cuda", 0)
    return view


LOCAL_BYTE_QUERY_CALLS = ("plan", "workspace_bytes", "tables_bytes")


@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_the_context_parallel_local_plans_keep_the_occupancy_rule(
    name, monkeypatch
) -> None:
    """The context-parallel package plans each rank's local problem with the
    occupancy rule (``LOCAL_SEGMENT_RULE``) where the token-aware rule may
    decide (its SM and head counts), and passes no rule elsewhere, where both
    rules give the same plan.  The keyword follows the SM counts of the
    envelope (148, 152 and 212).  On a CUDA device of GB300's architecture
    faked on the host: the workspace's local plan, and the plan, arena and
    table calls of the byte query, at 152 SMs with 4 heads (the keyword), at
    160 SMs and at 64 heads (none)."""
    kcp_plan, backend, jit = KCP_PLAN[name], BACKEND[name], JIT[name]
    kcp = cake_package(KCP_PACKAGE[name], "cake_kcp")
    assert kcp_plan.LOCAL_SEGMENT_RULE == "occupancy"
    assert kcp_plan.LOCAL_SEGMENT_RULE in backend.SEGMENT_RULES
    rule = {"segment_rule": "occupancy"}
    for sm, H in itertools.product((148, SM, 212), (4, 8, 16)):
        assert kcp_plan.local_segment_rule_kwargs(sm, H) == rule, (sm, H)
    for sm, H in ((160, 4), (216, 8), (SM, 2), (SM, 12), (SM, 64)):
        assert kcp_plan.local_segment_rule_kwargs(sm, H) == {}, (sm, H)
    S, N = sum(DOCS3_LENGTHS), len(DOCS3_LENGTHS)
    tokens = backend.plan(S, N, DOCS3_HEADS, SM, ARCH)
    local = backend.plan(S, N, DOCS3_HEADS, SM, ARCH, segment_rule="occupancy")
    assert (tokens.P, local.P, local.NSEG) == (15, 38, 114)
    cases = ((SM, DOCS3_HEADS, rule), (160, DOCS3_HEADS, {}), (SM, 64, {}))
    ws = getattr(kcp, KCP_WORKSPACE[name])()
    # a CUDA device of GB300's architecture (``ARCH``) on the host
    capability = {arch: cap for cap, arch in jit.ARCH_BY_CAPABILITY.items()}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda index=None: capability[ARCH]
    )
    monkeypatch.setattr(kcp_plan, "_cuda_arch", lambda index: ARCH)
    monkeypatch.setattr(jit, "device_arch", lambda index: ARCH)
    if hasattr(kcp, "_ARCH_BY_INDEX"):  # a per-device cache of the architecture
        monkeypatch.setattr(kcp, "_ARCH_BY_INDEX", {})
    # the workspace's local plan: the forward's and the arena's
    for sm, H, kw in cases:
        monkeypatch.setattr(jit, "sm_count", lambda index, _sm=sm: _sm)
        got = ws._local_plan((S, N, H), torch.device("cuda", 0))
        assert got is backend.plan(S, N, H, sm, ARCH, **kw), (sm, H)
    # the byte query's calls of the single-rank backend (the real functions run);
    # every context's intra record is taken to write its scratch table set
    seen: list[tuple[str, str]] = []
    for call in LOCAL_BYTE_QUERY_CALLS:
        real = getattr(backend, call)

        def spy(*args, _real=real, _call=call, **kwargs):
            seen.append((_call, kwargs.get("segment_rule", "not passed")))
            return _real(*args, **kwargs)

        monkeypatch.setattr(backend, call, spy)
    monkeypatch.setattr(kcp_plan, "intra_writes_tables", lambda *a, **k: True)
    workspace_bytes = getattr(
        kcp_plan, f"{KCP_PACKAGE[name].removeprefix('cake.')}_workspace_bytes"
    )
    for sm, H, kw in cases:
        seen.clear()
        assert workspace_bytes(_context_on_cuda(name, sm), H=H) > 0
        want = kw.get("segment_rule", "not passed")
        assert seen == [(call, want) for call in LOCAL_BYTE_QUERY_CALLS], (sm, H)
