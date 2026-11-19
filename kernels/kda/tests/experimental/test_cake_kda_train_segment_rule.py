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
  document, ``=`` the occupancy plan).  Every row takes its plan in both
  packages, with every record selected at that plan's segment count.
- One row per branch of the rule and the values at each bound.
- Over a grid of scalars: the route and the block table form never change, the
  segment count never grows, the light set saves the segment states exactly
  where the occupancy plan does, and the rule changes nothing outside its
  envelope.
- The memory of the token-aware plan (saved set, tables, arenas, the totals of
  the memory report) is at or below the occupancy plan's on every measured row.
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
KCP_PLAN = {
    PACKAGES[0]: cake_package("cake.cake_kda_det_train_kcp", "cake_kcp_plan"),
    PACKAGES[1]: cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_plan"),
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
# occupancy plan's table there), ``=`` the occupancy plan.
MEASURED_ROWS = (
    ("2048x1:4:g:0", 38, "P12"),
    ("2048x3:16:g:0", 9, "="),
    ("4096x16:4:g:0", 10, "="),
    ("4096x1:4:g:0", 38, "P12"),
    ("4096x2:4:g:0", 38, "P12"),
    ("4096x2:8:g:0", 19, "P12"),
    ("4096x3:4:g:0", 38, "P12"),
    ("4096x4:16:g:0", 9, "="),
    ("4096x8:4:g:0", 19, "P12"),
    ("4096x8:8:g:0", 10, "="),
    ("4096xskew16:4:g:0", 10, "="),
    ("4096xskew3:4:g:0", 38, "P12"),
    ("4096xskew8:4:g:0", 19, "P12"),
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
MEASURED_KINDS = {"P": 53, "=": 78}


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
            "sm_counts_by_arch": {"sm_103a": [152]},
            "heads": [4, 8, 16],
            "save_min_tokens_per_seg": 384,
            "min_uniform_segments": 12,
            "max_segment_tokens": 1408,
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
        # outside the envelope the token count is not read: the calls are the
        # occupancy rule's, a negative count included
        assert jit.segment_route(
            N, H, 148, "sm_100a", num_tokens=-1
        ) == jit.segment_route(N, H, 148, "sm_100a")
        assert backend.tables_bytes(
            -64, 1, H, sm_count=148, arch="sm_100a"
        ) == backend.tables_bytes(
            -64, 1, H, sm_count=148, arch="sm_100a", segment_rule="occupancy"
        )
        with pytest.raises(ValueError, match="must all be >= 1"):
            jit.segment_route(0, H, SM, ARCH, num_tokens=S)
    assert JIT[PACKAGES[0]].TOKEN_SEGMENTS == JIT[PACKAGES[1]].TOKEN_SEGMENTS
    assert JIT[PACKAGES[0]].SEGMENTS == JIT[PACKAGES[1]].SEGMENTS


@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_the_measured_rows_take_their_plans(name) -> None:
    """Every measured row: the occupancy rule gives the row's segments per
    document (the shipped plan), the token-aware rule the row's plan, in
    ``segment_route`` / ``segment_plan`` / ``segment_args`` and in the backend's
    plan of the row's gate and l2n modes (the occupancy rule's plan is the
    shipped one).  Every record of both plans is the one selected at the plan's
    own segment count (:func:`_check_records`), and on some rows the two plans
    select different records."""
    jit, backend = JIT[name], BACKEND[name]
    kinds = Counter(arm[0] for _spec, _p_old, arm in MEASURED_ROWS)
    assert len(MEASURED_ROWS) == len({spec for spec, *_ in MEASURED_ROWS}) == 131
    assert kinds == MEASURED_KINDS
    flips = 0
    for spec, p_old, arm in MEASURED_ROWS:
        S, N, H, gate, l2n = _row(spec)
        occupancy = jit.segment_route(N, H, SM, ARCH)
        if p_old is None:
            assert occupancy == ("serial", 1, 0, 0), spec
        else:
            assert occupancy == ("seg", p_old, N * p_old, int(N * H <= 4)), spec
        assert jit.segment_plan(N, H, SM, ARCH) == occupancy[:3], spec
        route = jit.segment_route(N, H, SM, ARCH, num_tokens=S)
        assert route == _expected_route(N, H, p_old, arm), spec
        assert jit.segment_plan(N, H, SM, ARCH, num_tokens=S) == route[:3], spec
        proportional = route[3]
        assert jit.segment_args(N, H, SM, ARCH, num_tokens=S) == (
            proportional,
            -N if proportional else route[1],
        ), spec
        records = {}
        for rule, want in (("tokens", route), ("occupancy", occupancy)):
            p = backend.plan(S, N, H, SM, ARCH, gate=gate, l2n=l2n, segment_rule=rule)
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
    occupancy plan's table), else the occupancy plan; the occupancy plan outside
    the envelope (other architectures, SM counts and head counts, the serial
    route, no token count)."""
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
        # four chains: the proportional table, as the occupancy plan's
        (1, 1, 4, ("seg", 12, 12, 1)),
        (4096, 1, 4, ("seg", 12, 12, 1)),
        (64, 3, 4, ("seg", 12, 36, 0)),
        (0, 3, 4, ("seg", 12, 36, 0)),  # an empty pack is a short pack
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
        for arch, sm in (
            ("sm_100a", SM),
            ("sm_107a", SM),
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
    """Over a grid of scalars (every architecture and none, five SM counts, ten
    head counts, document counts up to 300, packs at and around the size rule's
    and the segment bound's thresholds): the token-aware route and block table
    form are the occupancy rule's, its segment count is at most the occupancy
    count, the light set saves the segment states exactly where the occupancy
    plan's does, and outside the envelope the occupancy plan itself; the two
    packages agree everywhere."""
    det, nondet = JIT[PACKAGES[0]], JIT[PACKAGES[1]]
    rule = BACKEND[PACKAGES[0]].seg_states_saved
    per_seg = det.TOKEN_SEGMENTS["save_min_tokens_per_seg"]
    changed = 0
    for arch, sm in itertools.product((*det.ARCHES, None), (132, 148, 152, 160, 216)):
        served = arch == ARCH and sm == SM
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
    assert changed > 0


def _saved_state_bytes(backend, S: int, H: int, p, save_mode: str) -> int:
    """Bytes of the segment states in the saved set of plan ``p``: two FP32
    ``[128, 128]`` blocks per segment and head in the tape set, three in the
    light set where it saves them."""
    block = H * backend.D * backend.D * 4
    if save_mode == "tape":
        return 2 * p.NSEG * block
    return 3 * p.NSEG * block if backend.seg_states_saved(S, p.NSEG) else 0


@pytest.mark.parametrize("name", PACKAGES, ids=PACKAGE_IDS)
def test_the_memory_of_the_measured_rows_never_grows(name) -> None:
    """On every measured row (its gate and l2n modes) the token-aware plan needs
    at most the occupancy plan's bytes: the saved set and its segment states
    (light and tape), the tables, the forward-only and the backward arenas, and
    every total of the memory report, whose segment count is the plan's; the
    rows that change the plan have fewer segment blocks.  Each rule's tables are
    its own plan's, and its saved set differs from the other rule's by exactly
    their tables and segment states."""
    backend = BACKEND[name]
    q = {"sm_count": SM, "arch": ARCH}
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
    for spec, _p_old, arm in MEASURED_ROWS:
        S, N, H, gate, l2n = _row(spec)
        tokens = backend.plan(S, N, H, SM, ARCH, gate=gate, l2n=l2n)
        occupancy = backend.plan(
            S, N, H, SM, ARCH, gate=gate, l2n=l2n, segment_rule="occupancy"
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
    segment counts and block table form under both rules."""
    det, nondet = BACKEND[PACKAGES[0]], BACKEND[PACKAGES[1]]
    rows = [_row(spec) for spec, *_ in MEASURED_ROWS]
    for lengths, H in ((DOCS3_LENGTHS, DOCS3_HEADS), (ISSUE_LENGTHS, ISSUE_HEADS)):
        rows.append((sum(lengths), len(lengths), H, "g", "off"))
    for (S, N, H, gate, l2n), rule in itertools.product(rows, det.SEGMENT_RULES):
        a = det.plan(S, N, H, SM, ARCH, gate=gate, l2n=l2n, segment_rule=rule)
        b = nondet.plan(S, N, H, SM, ARCH, gate=gate, l2n=l2n, segment_rule=rule)
        assert (a.recur, a.P, a.NSEG, a.proportional, a.seg_arg, a.seg_save) == (
            b.recur,
            b.P,
            b.NSEG,
            b.proportional,
            b.seg_arg,
            b.seg_save,
        ), (S, N, H, gate, l2n, rule)


def test_the_context_parallel_local_plans_keep_the_occupancy_rule() -> None:
    """The context-parallel packages plan each rank's local problem, its
    workspace and its byte predictions with the occupancy rule
    (``LOCAL_SEGMENT_RULE``): the keyword is passed where the token-aware rule
    may decide (its SM and head counts), and nowhere else, where both rules
    give the same plan."""
    for name in PACKAGES:
        kcp_plan, backend = KCP_PLAN[name], BACKEND[name]
        assert kcp_plan.LOCAL_SEGMENT_RULE == "occupancy"
        assert kcp_plan.LOCAL_SEGMENT_RULE in backend.SEGMENT_RULES
        rule = {"segment_rule": "occupancy"}
        for H in (4, 8, 16):
            assert kcp_plan.local_segment_rule_kwargs(SM, H) == rule
        for sm, H in ((148, 4), (216, 8), (SM, 2), (SM, 12), (SM, 64)):
            assert kcp_plan.local_segment_rule_kwargs(sm, H) == {}, (sm, H)
        S, N, H = sum(DOCS3_LENGTHS), len(DOCS3_LENGTHS), DOCS3_HEADS
        local = backend.plan(
            S, N, H, SM, ARCH, **kcp_plan.local_segment_rule_kwargs(SM, H)
        )
        assert local is backend.plan(S, N, H, SM, ARCH, segment_rule="occupancy")
        assert (local.P, local.NSEG) == (38, 114)
