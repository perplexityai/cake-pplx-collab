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

import collections
import contextlib
import functools
import math

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_moe_judge import judge_draws
from tests.cake.test_helpers.cake_naive import DRAWS
from tests.cake.test_helpers.cake_package import cake_package

# Backend suite: the package is named once, through ``cake_package``; nothing else
# of ``cake_pplx`` is imported here.  The tests of the stable entry points and of
# the ``grouped_mm_bf16`` handoff live in
# ``test_cake_moe_grouped_gemm_det_train_api.py``.
pkg = cake_package("cake_moe_grouped_gemm_det_train")
cb = cake_package("cake_moe_grouped_gemm_det_train", "cake_backend")
cake_jit = cake_package("cake_moe_grouped_gemm_det_train", "cake_jit")
cake_autograd = cake_package("cake_moe_grouped_gemm_det_train", "cake_autograd")
cake_moe_grouped_gemm_det_train_dgrad = pkg.cake_moe_grouped_gemm_det_train_dgrad
cake_moe_grouped_gemm_det_train_forward = pkg.cake_moe_grouped_gemm_det_train_forward
cake_moe_grouped_gemm_det_train_wgrad = pkg.cake_moe_grouped_gemm_det_train_wgrad
SUPPORTED_COMPUTE_CAPABILITIES = cb.SUPPORTED_COMPUTE_CAPABILITIES
cake_moe_grouped_gemm_det_train_prepare_dgrad = (
    cb.cake_moe_grouped_gemm_det_train_prepare_dgrad
)
cake_moe_grouped_gemm_det_train_prepare_forward = (
    cb.cake_moe_grouped_gemm_det_train_prepare_forward
)
cake_moe_grouped_gemm_det_train_prepare_wgrad = (
    cb.cake_moe_grouped_gemm_det_train_prepare_wgrad
)
HOST_PLAN_CONSTANTS = cake_jit.HOST_PLAN_CONSTANTS
PROGRAMS = cake_jit.PROGRAMS
ROUTES = cake_jit.ROUTES

# Group size lists: an empty group, a one-row group, sizes that are not tile
# multiples, and a long-reduction case that can select the 512 k tile.
RAGGED_E4 = [300, 0, 1, 700]
RAGGED_E6 = [1, 257, 511, 4099, 0, 33]
TINY_E5 = [1, 2, 3, 0, 4]
LONG_E3 = [6145, 0, 12289]
# Groups shorter than one 64-row k-step: every output tile is a single k-step, and
# a CTA switches group at almost every tile (the race repro shape).
SMALL_E512 = [32] * 512
SMALL_GROUP_LAUNCHES = 200
# Two ragged groups of more than every architecture's ``k512_min_rows_per_group``
# rows on average: the weight gradient takes the 512-wide k tile on the
# architectures of ``k512_archs``.
K512_E2 = [8200, 8333]
# Two ragged groups of at least ``dgrad_evict_first_min_rows`` rows on average: with
# ``K >= dgrad_evict_first_min_k`` the data gradient takes the evict-first twin of its
# wide tile on the architectures of ``dgrad_evict_first_archs`` (and, with
# ``K >= dgrad_wide_ts33_min_k``, its 3 / 3 head / tail twin on those of
# ``dgrad_wide_ts33_archs``).
EVICT_FIRST_E2 = [16390, 16500]
# The MoE expert rows the evict-first and k-tile thresholds were measured on:
# ``EXPERT_ROWS`` routed rows split over each group count of ``EXPERT_GROUPS``.
EXPERT_ROWS = 131072
EXPERT_GROUPS = (4, 8, 16, 32)
# The MoE benchmark rows the 3 / 3 head / tail rule was measured on: (label, sum_m, E,
# row variants); the balanced, skewed and one-empty variants of an E4 row share sum_m
# and E, so the host plan gives them one stage.
BENCH_ROWS = (
    ("E4_m112576", 112576, 4, 3),
    ("E4_m136064", 136064, 4, 3),
    ("E8", EXPERT_ROWS, 8, 1),
    ("E16", EXPERT_ROWS, 16, 1),
    ("E32", EXPERT_ROWS, 32, 1),
)
# The benchmark projections, (N, K) of w ``[E, N, K]``.
BENCH_PROJ = (("gate_up", 4096, 6144), ("down", 6144, 2048))
# Weight-gradient cases of the queue walk (``cake_backend.wgrad_queue_walk``): (label,
# sizes, N, K), each over more 512-wide k tiles than one wave of clusters, so the queue
# claims tiles over several waves before a split tail.  Eight ragged groups with an empty
# and a one-row group under ``wave_sync_min_k`` (the ``main_halves`` stand-in), and the
# skewed and one-empty E4 rows at a quarter of the benchmark rows' sums (the
# ``main_wave_halves`` stand-in).
QUEUE_CASES = (
    ("e8_ragged", [4100, 0, 4170, 1, 8000, 3001, 513, 12289], 1024, 2048),
    ("e4_skewed", [13608, 10208, 6800, 3400], 2048, 4608),
    ("e4_one_empty", [14072, 8440, 5632, 0], 2048, 4608),
)
QUEUE_CASE_BY_LABEL = {case[0]: case for case in QUEUE_CASES}
DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _wide_ts33(op, k, arch, sum_m, groups, c):
    """The 3 / 3 head / tail rule of ``op``: ``dgrad_wide_ts33`` / ``fwd_wide_ts33``."""
    rule = cb.dgrad_wide_ts33 if op == "dgrad" else cb.fwd_wide_ts33
    return rule(k, arch, sum_m, groups, c)


def _evict_first(op, k, arch, sum_m, groups, c):
    """The evict-first rule of ``op``: ``dgrad_evict_first``; ``fwd_lwh_ef`` or
    ``fwd_evict_first``."""
    if op == "dgrad":
        return cb.dgrad_evict_first(k, arch, sum_m, groups, c)
    return cb.fwd_lwh_ef(k, arch, c) or cb.fwd_evict_first(k, arch, c)


def _require_program(op, out_dtype=None):
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    if not cb.generated_program_available(device, op, out_dtype):
        pytest.skip(
            f"no generated ragged grouped GEMM program for {op} registered for {arch}"
        )
    return device


def _offs(sizes, device):
    return torch.tensor(sizes, dtype=torch.int32, device=device).cumsum(
        0, dtype=torch.int32
    )


def _inputs(sizes, n, k, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    sum_m = sum(sizes)
    x = torch.randn(sum_m, k, generator=gen, device=device, dtype=torch.float32)
    w = torch.randn(len(sizes), n, k, generator=gen, device=device, dtype=torch.float32)
    g = torch.randn(sum_m, n, generator=gen, device=device, dtype=torch.float32)
    x = (x / math.sqrt(k)).to(torch.bfloat16)
    w = (w / math.sqrt(k)).to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    return x, w, g, _offs(sizes, device)


def _bounds(sizes):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return list(zip([0] + ends[:-1], ends, strict=True))


# Accuracy: every output a GPU test checks is judged over ``cake_naive.DRAWS`` independent
# inputs against the naive BF16 tier (``tests/cake/test_helpers/cake_moe_judge.py``): its
# error against the per-group float64 product, in the L2 and the max norm, median and
# largest over the draws, at most ``NEAR_TIE`` times torch's BF16 matmul's, with no
# tolerance.  Draw ``j`` of a case of seed ``seed`` takes ``_inputs(..., seed=draw_seed(seed,
# j))``; draw 0, the case's own inputs, is judged last.

# The output each op's launch writes, as the TIER lines name it.
OUT_NAMES = {"fwd": "Y", "dgrad": "dX", "wgrad": "dW"}


def _operands(op, x, w, g):
    """The operands of ``op``: fwd (x, w), dgrad (g, w), wgrad (g, x)."""
    return {"fwd": (x, w), "dgrad": (g, w), "wgrad": (g, x)}[op]


class _Case:
    """The inputs of a prepared launch, ``_inputs(sizes, n, k, device, seed)``; a draw writes
    its own inputs into the same tensors (:meth:`refill`), which the launch reads at every
    launch."""

    def __init__(self, sizes, n, k, device, seed=0):
        self.sizes, self.n, self.k, self.device, self.seed = sizes, n, k, device, seed
        self.x, self.w, self.g, self.offs = _inputs(sizes, n, k, device, seed=seed)

    def refill(self, seed):
        x, w, g, _ = _inputs(self.sizes, self.n, self.k, self.device, seed=seed)
        self.x.copy_(x)
        self.w.copy_(w)
        self.g.copy_(g)

    def operands(self, op):
        return _operands(op, self.x, self.w, self.g)


def _judge_launch(record_property, op, launch, case, *, launches=1, name=None, then=None):
    """Judge the prepared ``launch`` of ``op`` on ``case`` over the draws of ``case.seed``:
    each draw refills the inputs, then ``launches`` launches into an output refilled with
    NaN each judge it and give the bits of the draw's first launch (run to run).
    ``then``: the test's later checks (``judge_draws``)."""

    def draw(d):
        case.refill(d.seed)
        first = None
        for call in range(launches):
            launch.out.fill_(float("nan"))
            out = launch.launch()
            d.op(name or OUT_NAMES[op], out, op, *case.operands(op), case.sizes)
            if first is None:
                first = out.clone()
            else:
                assert torch.equal(out, first), f"draw {d.index} launch {call}"

    judge_draws(record_property, draw, seed=case.seed, then=then)


def _fresh_stream_workspaces(monkeypatch):
    """Start the test with no kept stream workspace and no queue probe (both restored
    after it): torch reuses its pooled stream handles, so a new ``torch.cuda.Stream``
    could otherwise find the workspace an earlier test kept for the same handle."""
    monkeypatch.setattr(cb, "_stream_workspaces", {})
    monkeypatch.setattr(cb, "_queue_splits", {})


def _probes_queue_split(device, op):
    """Whether ``op`` is listed in ``domain_split_ops`` and ``device``'s architecture in
    ``domain_split_archs``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    c = HOST_PLAN_CONSTANTS
    return op in c["domain_split_ops"] and arch in c["domain_split_archs"]


# ---------------------------------------------------------------------------
# Host plan (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sm_count", [148, 160, 212])
@pytest.mark.parametrize(
    "sizes", [RAGGED_E4, RAGGED_E6, LONG_E3, [131072] * 1, [4096] * 32]
)
def test_plan_invariants(sm_count, sizes):
    c = HOST_PLAN_CONSTANTS
    sum_m, e = sum(sizes), len(sizes)
    cta_group = int(c["cta_group"])
    for op, n, k in (("fwd", 4096, 6144), ("dgrad", 4096, 6144)):
        plan = cb.plan_fwd_dgrad(
            op, sum_m=sum_m, num_groups=e, n=n, k=k, sm_count=sm_count, constants=c
        )
        grid_x = plan["grid"][0]
        assert (
            grid_x % cta_group == 0
            and cta_group <= grid_x <= (sm_count // cta_group) * cta_group
        )
        assert plan["launches"] == 1 and plan["tile_k"] is None
        # The two claim queues: c0 = clusters // 2, c1 = the rest.
        assert plan["clusters"] == grid_x // cta_group
        assert plan["c0"] == plan["clusters"] // 2
        assert plan["c0"] + plan["c1"] == plan["clusters"] and plan["c1"] >= 1
        assert plan["tile_n"] == int(c["block_n"])
        # The wide tile bounds the clusters by its 512-column tiles.
        wide = cb.plan_fwd_dgrad(
            op,
            sum_m=sum_m,
            num_groups=e,
            n=n,
            k=k,
            sm_count=sm_count,
            tile_n=int(c["wide_block_n"]),
            constants=c,
        )
        tiles = (sum_m // int(c["cluster_m"]) + e) * (
            (n if op == "fwd" else k) // int(c["wide_block_n"])
        )
        assert wide["tile_n"] == int(c["wide_block_n"])
        assert wide["clusters"] == min(sm_count // cta_group, max(1, tiles))
        assert wide["grid"][0] == wide["clusters"] * cta_group
        assert wide["c0"] == wide["clusters"] // 2
        assert wide["c0"] + wide["c1"] == wide["clusters"]
        assert plan["queue_split"] == wide["queue_split"] == "even"
    # The probed queue split: the ops of ``domain_split_ops`` on the architectures of
    # ``domain_split_archs``, weights of at most ``domain_split_max_weight_elems``
    # elements; c0 / c1 stay the even split.
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for op, n, k in (
            ("fwd", 4096, 6144),
            ("fwd", 6144, 2048),
            ("dgrad", 4096, 6144),
            ("dgrad", 6144, 2048),
        ):
            plan = cb.plan_fwd_dgrad(
                op,
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                sm_count=sm_count,
                tile_n=int(c["wide_block_n"]),
                constants=c,
                arch=arch,
            )
            probed = (
                op in c["domain_split_ops"]
                and arch in c["domain_split_archs"]
                and n * k <= int(c["domain_split_max_weight_elems"])
            )
            assert plan["queue_split"] == ("probed" if probed else "even")
            assert plan["c0"] == plan["clusters"] // 2
            assert plan["c0"] + plan["c1"] == plan["clusters"]
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for n, k in ((4096, 6144), (6144, 2048), (256, 256)):
            tile_k = cb.select_wgrad_tile(k, arch, sum_m, e, c)
            assert tile_k in (256, 512) and k % tile_k == 0
            if tile_k == 512:
                assert (
                    arch in c["k512_archs"]
                    and sum_m / e >= c["k512_min_rows_per_group"][arch]
                )
            plan = cb.plan_wgrad(
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                tile_k=tile_k,
                sm_count=sm_count,
                reduce_threads=256,
                constants=c,
            )
            assert plan["grid"][0] == plan["clusters"] * cta_group
            assert 0 <= plan["num_tail"] < plan["clusters"]
            assert 1 <= plan["tail_splits"] <= int(c["tail_plan"]["max_splits"])
            assert plan["raster_rows"] in (0, 1)
            assert plan["launches"] == 1 + int(plan["tail_splits"] > 1)
            assert (plan["tail_reduce_grid"] is None) == (plan["tail_splits"] == 1)
            if plan["tail_reduce_grid"] is not None:
                assert plan["tail_reduce_grid"][0] > 0
            if plan["num_tail"] == 0:
                assert plan["tail_splits"] == 1
            assert plan["tail_tile_k"] == tile_k
            if tile_k != 512:
                continue
            # The k256-tail stage: the 512-wide tile's grid, n-major, and the tail of
            # the 256-wide tiling of the same clusters, split as that tile's own.
            half = cb.K256_TAIL_TILE_K
            k256t = cb.plan_wgrad(
                sum_m=sum_m,
                num_groups=e,
                n=n,
                k=k,
                tile_k=tile_k,
                sm_count=sm_count,
                reduce_threads=256,
                k256_tail=True,
                constants=c,
            )
            assert k256t["grid"] == plan["grid"] and k256t["tile_k"] == 512
            assert k256t["tail_tile_k"] == half and k256t["raster_rows"] == 0
            clusters = k256t["clusters"]
            assert k256t["num_tail"] == (
                (e * (n // int(c["block_n"])) * (k // half)) % clusters
                if int(c["wgrad_tail_split"])
                else 0
            )
            assert k256t["tail_splits"] == (
                cb.wgrad_tail_plan(
                    k256t["num_tail"],
                    clusters,
                    sum_m,
                    e,
                    half,
                    block_k=int(c["block_k"]),
                    cluster_m=int(c["cluster_m"]),
                    max_splits=int(c["tail_plan"]["max_splits"]),
                    sm_clock_ghz=float(c["tail_plan"]["sm_clock_ghz"]),
                    dram_tbps=float(c["tail_plan"]["dram_tbps"]),
                    launch_us=float(c["tail_plan"]["launch_us"]),
                    cycles_per_step=c["tail_plan"]["cycles_per_step"],
                )
            )
            assert k256t["launches"] == 1 + int(k256t["tail_splits"] > 1)
            # One reduce CTA per tail tile and row block; a thread sums epi_chunk columns.
            rows_per_cta = 256 // (half // int(c["epi_chunk"]))
            assert k256t["tail_reduce_grid"] == (
                [k256t["num_tail"] * (int(c["cluster_m"]) // rows_per_cta), 1, 1]
                if k256t["tail_splits"] > 1
                else None
            )


def test_stage_bindings_are_unique_per_architecture():
    """A route may spell two stages of one name; they serve disjoint architectures,
    and ``stage_binding`` returns the one built for the asked architecture."""
    for name, route in ROUTES.items():
        for arch in route["arches"]:
            served = {}
            for stage in route["stages"]:
                if arch in PROGRAMS[stage["program"]]["arches"]:
                    assert stage["name"] not in served, (name, arch, stage["name"])
                    served[stage["name"]] = stage
            assert "main" in served or "main_wave" in served, (name, arch)
            for stage_name, stage in served.items():
                assert cake_jit.stage_binding(name, stage_name, arch) is stage
                assert cake_jit.stage_served(name, stage_name, arch)
            assert not cake_jit.stage_served(name, "no_such_stage", arch)
            with pytest.raises(KeyError):
                cake_jit.stage_binding(name, "no_such_stage", arch)


def test_program_choice_follows_the_host_plan_constants():
    """``wide_fwd_archs`` / ``wide_dgrad_archs`` select the 512-column tile of
    ``fwd`` / ``dgrad`` when the output columns are a multiple of
    ``wide_block_n``; ``dgrad_evict_first_archs`` the evict-first twin of the
    ``dgrad`` wide tile (``main_wide_ef``) where ``dgrad_evict_first`` holds, and
    ``fwd_lwh_ef_archs`` / ``fwd_evict_first_archs`` that of the ``fwd`` wide tile
    where ``fwd_lwh_ef`` / ``fwd_evict_first`` holds;
    ``dgrad_wide_ts33_archs`` / ``fwd_wide_ts33_archs`` the 3 / 3 head / tail twin of
    the ``dgrad`` / ``fwd`` wide tile (``main_wide_ts33``) where ``dgrad_wide_ts33`` /
    ``fwd_wide_ts33`` holds and the evict-first twin does not; ``wave_sync_archs``
    the wave build of each k tile from ``wave_sync_min_k``, launched with the
    architecture's ``wave_sync_mode``; ``k512_release_archs`` the twin releasing the
    512-wide accumulator in halves; ``k256_tail_archs`` its twin running the
    leftover tiles as 256-column half tiles
    (``main_halves_k256t``) where ``wgrad_k256_tail`` holds.  Each listed
    architecture's routes serve the stage the constant selects."""
    c = HOST_PLAN_CONSTANTS
    block_n, wide_n = int(c["block_n"]), int(c["wide_block_n"])
    assert wide_n == 2 * block_n
    for key in ("wide_fwd_archs", "wide_dgrad_archs", "k512_archs"):
        assert set(c[key]) <= set(cake_jit.ARCHES), key
    assert set(c["dgrad_evict_first_archs"]) <= set(c["wide_dgrad_archs"])
    assert set(c["dgrad_evict_first_archs"]) <= set(c["dgrad_evict_first_min_rows"])
    assert all(int(v) > 0 for v in c["dgrad_evict_first_min_rows"].values())
    assert int(c["dgrad_evict_first_min_k"]) % wide_n == 0
    assert set(c["dgrad_wide_ts33_archs"]) <= set(c["wide_dgrad_archs"])
    assert set(c["dgrad_wide_ts33_min_rows"]) == set(c["dgrad_wide_ts33_archs"])
    assert all(int(v) > 0 for v in c["dgrad_wide_ts33_min_rows"].values())
    assert int(c["dgrad_wide_ts33_min_k"]) % wide_n == 0
    assert set(c["fwd_wide_ts33_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_wide_ts33_max_k"]) > 0
    assert int(c["fwd_wide_ts33_max_rows"]) > 0
    assert set(c["fwd_lwh_ef_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_lwh_ef_max_k"]) > 0
    assert set(c["fwd_evict_first_archs"]) <= set(c["wide_fwd_archs"])
    assert int(c["fwd_evict_first_min_k"]) > 0
    # The probed queue split serves the domain-scheduled programs only (sm_107a's are
    # static and ignore c0 / c1), and the data gradient only.
    assert set(c["domain_split_archs"]) <= {"sm_100a", "sm_103a"}
    assert list(c["domain_split_ops"]) == ["dgrad"]
    assert int(c["domain_split_max_weight_elems"]) > 0
    assert set(c["k512_min_rows_per_group"]) == set(c["k512_archs"])
    assert all(int(v) > 0 for v in c["k512_min_rows_per_group"].values())
    assert set(c["k512_release_archs"]) <= set(c["k512_archs"])
    assert set(c["k256_tail_archs"]) <= set(c["k512_release_archs"])
    assert int(c["k256_tail_max_k"]) % 512 == 0 and int(c["k256_tail_max_k"]) > 0
    assert int(c["k256_tail_min_rows"]) > 0
    assert set(c["wave_sync_archs"]) == {256, 512}
    for archs in c["wave_sync_archs"].values():
        assert set(archs) <= set(cake_jit.ARCHES)
        assert all(int(c["wave_sync_mode"][arch]) > 0 for arch in archs)
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        wide_archs = c[f"wide_{op}_archs"]
        for arch in ROUTES[name]["arches"]:
            served = cake_jit.stage_served(name, "main_wide", arch)
            if arch in wide_archs:
                assert served, (name, arch)
            served_ef = cake_jit.stage_served(name, "main_wide_ef", arch)
            ef_keys = (
                ("dgrad_evict_first_archs",)
                if op == "dgrad"
                else ("fwd_lwh_ef_archs", "fwd_evict_first_archs")
            )
            if any(arch in c[key] for key in ef_keys):
                assert served_ef, (name, arch)
            served_ts33 = cake_jit.stage_served(name, "main_wide_ts33", arch)
            if arch in c[f"{op}_wide_ts33_archs"]:
                assert served_ts33, (name, arch)
            for cols, red in (
                (4096, 512),
                (4096 - block_n, 512),
                (block_n, 512),
                (4096, 2048),
                (4096, 6144),
            ):
                n, k = (cols, red) if op == "fwd" else (red, cols)
                for sum_m, groups in ((4096, 4), (EXPERT_ROWS, 4)):
                    stage, tile_n = cb.select_row_stage(
                        name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                    )
                    wide = cols % wide_n == 0 and arch in wide_archs and served
                    ef = (
                        wide
                        and served_ef
                        and _evict_first(op, k, arch, sum_m, groups, c)
                    )
                    ts33 = (
                        wide
                        and not ef
                        and served_ts33
                        and _wide_ts33(op, k, arch, sum_m, groups, c)
                    )
                    expected = (
                        "main_wide_ef"
                        if ef
                        else "main_wide_ts33"
                        if ts33
                        else "main_wide"
                    )
                    assert (stage, tile_n) == (
                        (expected, wide_n) if wide else ("main", block_n)
                    ), (op, arch, cols, sum_m)
                assert cb.fwd_dgrad_tile_n(op, n, k, arch, c) == (
                    wide_n if cols % wide_n == 0 and arch in wide_archs else block_n
                )
    min_k = int(c["wave_sync_min_k"])
    for out_dtype in ("bfloat16", "float32"):
        for tile_k in (256, 512):
            name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=tile_k)
            assert name in ROUTES, name
            for arch in ROUTES[name]["arches"]:
                if tile_k == 512 and arch in c["k512_archs"]:
                    assert cake_jit.stage_served(name, "main", arch), (name, arch)
                if arch in c["wave_sync_archs"][tile_k]:
                    assert cake_jit.stage_served(name, "main_wave", arch), (name, arch)
                if tile_k == 512 and arch in c["k256_tail_archs"]:
                    assert cake_jit.stage_served(name, "main_halves_k256t", arch)
                groups = 4
                for k, sum_m in (
                    (min_k, 4096),
                    (min_k, EXPERT_ROWS),
                    (min_k // 2, 4096),
                    (min_k // 2, EXPERT_ROWS),
                ):
                    stage, wave = cb.select_wgrad_stage(
                        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=groups
                    )
                    expect_wave = (
                        k >= min_k
                        and arch in c["wave_sync_archs"][tile_k]
                        and cake_jit.stage_served(name, "main_wave", arch)
                    )
                    base = "main_wave" if expect_wave else "main"
                    halves = (
                        tile_k == 512
                        and arch in c["k512_release_archs"]
                        and cake_jit.stage_served(name, f"{base}_halves", arch)
                    )
                    if tile_k == 512 and arch in c["k512_release_archs"]:
                        assert halves, (name, arch, base)
                    k256t = (
                        halves
                        and base == "main"
                        and cb.wgrad_k256_tail(k, arch, sum_m, groups, c)
                        and cake_jit.stage_served(name, "main_halves_k256t", arch)
                    )
                    expected = f"{base}_halves" if halves else base
                    assert wave == expect_wave, (name, arch, k)
                    assert stage == ("main_halves_k256t" if k256t else expected), (
                        name,
                        arch,
                        k,
                        sum_m,
                    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_expert_rows_take_the_measured_dgrad_stage_and_k_tile(arch):
    """Over ``EXPERT_ROWS`` routed rows: the data gradient of the gate/up projection
    (``K`` 6144) takes the evict-first wide tile on sm_100a at 4 to 32 groups and on
    sm_107a at 4 and 8 groups, and the 3 / 3 head / tail wide tile on sm_103a at 4, 8
    and 16 groups, else the plain wide tile; the down projection (``K`` 2048) takes
    the plain wide tile.
    The weight gradient's 512-wide k tile starts at the architecture's
    ``k512_min_rows_per_group`` (3072 rows per group on sm_100a and sm_103a, 4096 on
    sm_107a): at 32 groups on every architecture, and at the measured E32 row of
    112576 rows (3518 per group) on sm_100a and sm_103a only."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("dgrad")
    for groups in EXPERT_GROUPS:
        for n, k in ((4096, 6144), (6144, 2048)):
            ef = k == 6144 and (
                arch == "sm_100a" or (arch == "sm_107a" and groups in (4, 8))
            )
            ts33 = k == 6144 and arch == "sm_103a" and groups in (4, 8, 16)
            assert cb.dgrad_evict_first(k, arch, EXPERT_ROWS, groups, c) == ef
            assert cb.dgrad_wide_ts33(k, arch, EXPERT_ROWS, groups, c) == ts33
            stage, tile_n = cb.select_row_stage(
                name, "dgrad", n, k, arch, c, sum_m=EXPERT_ROWS, num_groups=groups
            )
            assert tile_n == int(c["wide_block_n"]), (arch, groups, k)
            expected = (
                "main_wide_ef" if ef else "main_wide_ts33" if ts33 else "main_wide"
            )
            assert stage == expected, (arch, groups, k)
        k512 = EXPERT_ROWS // groups >= c["k512_min_rows_per_group"][arch]
        for k in (2048, 6144):
            tile_k = cb.select_wgrad_tile(k, arch, EXPERT_ROWS, groups, c)
            assert tile_k == (512 if k512 else 256), (arch, groups, k)
    for k in (2048, 6144):
        tile_k = cb.select_wgrad_tile(k, arch, 112576, 32, c)
        assert tile_k == (256 if arch == "sm_107a" else 512), (arch, k)


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_rows_take_the_measured_wide_ts33_stage(arch):
    """Over the MoE benchmark rows (``BENCH_ROWS`` x ``BENCH_PROJ``, both ops): sm_103a
    takes ``main_wide_ts33`` for the down forward pass at E32 and the gate/up data
    gradient at E4, E8 and E16 (9 rows with the E4 variants); sm_100a and sm_107a
    for none.  Every other row keeps its stage without the rule."""
    c = HOST_PLAN_CONSTANTS
    taken = set()
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        for proj, n, k in BENCH_PROJ:
            for label, sum_m, groups, _variants in BENCH_ROWS:
                stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                )
                assert tile_n == int(c["wide_block_n"]), (op, proj, label)
                if stage == "main_wide_ts33":
                    taken.add((op, proj, label))
                else:
                    assert stage in ("main_wide", "main_wide_ef"), (op, proj, label)
    expected = {
        "sm_100a": set(),
        "sm_103a": {
            ("fwd", "down", "E32"),
            ("dgrad", "gate_up", "E4_m112576"),
            ("dgrad", "gate_up", "E4_m136064"),
            ("dgrad", "gate_up", "E8"),
            ("dgrad", "gate_up", "E16"),
        },
        "sm_107a": set(),
    }[arch]
    assert taken == expected, (arch, sorted(taken ^ expected))
    variants = {label: v for label, _sum_m, _groups, v in BENCH_ROWS}
    rows = sum(variants[label] for _op, _proj, label in taken)
    assert rows == {"sm_100a": 0, "sm_103a": 9, "sm_107a": 0}[arch]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_rows_take_the_measured_evict_first_stage(arch):
    """Over the MoE benchmark rows (``BENCH_ROWS`` x ``BENCH_PROJ``, both ops): the
    wide tile's L2 evict-first twin (``main_wide_ef``) serves the gate/up forward pass
    (``K`` 6144) on sm_100a and sm_103a at every group count, the gate/up data
    gradient on sm_100a at every group count (``sum_m // E >= 4096``) and on sm_107a
    at E4 and E8 (``sum_m // E >= 16384``), and the down forward pass (``K`` 2048) on
    sm_107a; no down data gradient, and no sm_103a data gradient (the 3 / 3 head /
    tail rule keeps its rows).  Every row takes the stage its rules select."""
    c = HOST_PLAN_CONSTANTS
    taken = set()
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        for proj, n, k in BENCH_PROJ:
            for label, sum_m, groups, _variants in BENCH_ROWS:
                stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                )
                assert tile_n == int(c["wide_block_n"]), (op, proj, label)
                ef = _evict_first(op, k, arch, sum_m, groups, c)
                ts33 = not ef and _wide_ts33(op, k, arch, sum_m, groups, c)
                assert stage == (
                    "main_wide_ef" if ef else "main_wide_ts33" if ts33 else "main_wide"
                ), (op, proj, label)
                if ef:
                    taken.add((op, proj, label))
    labels = [label for label, _sum_m, _groups, _variants in BENCH_ROWS]
    expected = {
        "sm_100a": {("fwd", "gate_up", label) for label in labels}
        | {("dgrad", "gate_up", label) for label in labels},
        "sm_103a": {("fwd", "gate_up", label) for label in labels},
        "sm_107a": {("fwd", "down", label) for label in labels}
        | {("dgrad", "gate_up", label) for label in ("E4_m112576", "E4_m136064", "E8")},
    }[arch]
    assert taken == expected, (arch, sorted(taken ^ expected))
    variants = {label: v for label, _sum_m, _groups, v in BENCH_ROWS}
    rows = sum(variants[label] for _op, _proj, label in taken)
    assert rows == {"sm_100a": 18, "sm_103a": 9, "sm_107a": 16}[arch]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_bench_rows_take_the_probed_queue_split(arch):
    """Over the MoE benchmark rows (``BENCH_ROWS`` x ``BENCH_PROJ``, both ops): sm_100a
    and sm_103a pass the probed queue split for every data-gradient down-projection row
    (9 rows with the E4 variants) and the even split for every forward row and every
    gate/up row; sm_107a passes the even split everywhere."""
    c = HOST_PLAN_CONSTANTS
    taken = set()
    for op in ("fwd", "dgrad"):
        name = cake_jit.record_name(op)
        for proj, n, k in BENCH_PROJ:
            for label, sum_m, groups, _variants in BENCH_ROWS:
                _stage, tile_n = cb.select_row_stage(
                    name, op, n, k, arch, c, sum_m=sum_m, num_groups=groups
                )
                plan = cb.plan_fwd_dgrad(
                    op,
                    sum_m=sum_m,
                    num_groups=groups,
                    n=n,
                    k=k,
                    sm_count=152,
                    tile_n=tile_n,
                    constants=c,
                    arch=arch,
                )
                if plan["queue_split"] == "probed":
                    taken.add((op, proj, label))
                else:
                    assert plan["queue_split"] == "even", (op, proj, label)
    expected = (
        {("dgrad", "down", label) for label, _sum_m, _groups, _variants in BENCH_ROWS}
        if arch in ("sm_100a", "sm_103a")
        else set()
    )
    assert taken == expected, (arch, sorted(taken ^ expected))
    variants = {label: v for label, _sum_m, _groups, v in BENCH_ROWS}
    rows = sum(variants[label] for _op, _proj, label in taken)
    assert rows == {"sm_100a": 9, "sm_103a": 9, "sm_107a": 0}[arch]


@pytest.mark.parametrize("out_dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_expert_rows_take_the_measured_wgrad_stage(arch, out_dtype):
    """Over ``EXPERT_ROWS`` routed rows the weight gradient ``dW[e]`` (``N`` x ``K``)
    takes the 512-wide k tile released in halves.  On sm_107a the down projection
    (``N`` 6144, ``K`` 2048) takes its k256-tail twin (``main_halves_k256t``) at 4, 8
    and 16 groups; at 32 groups, and for the gate/up projection (``N`` 4096, ``K``
    6144) at every group count, it keeps ``main_halves``.  The other architectures
    never take the twin."""
    c = HOST_PLAN_CONSTANTS
    name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=512)
    for groups in EXPERT_GROUPS:
        for k in (2048, 6144):
            assert cb.select_wgrad_tile(k, arch, EXPERT_ROWS, groups, c) == 512
            k256t = arch == "sm_107a" and k == 2048 and groups in (4, 8, 16)
            assert cb.wgrad_k256_tail(k, arch, EXPERT_ROWS, groups, c) == k256t
            stage, wave = cb.select_wgrad_stage(
                name, k, 512, arch, c, sum_m=EXPERT_ROWS, num_groups=groups
            )
            base = "main_wave" if wave else "main"
            assert stage == ("main_halves_k256t" if k256t else f"{base}_halves"), (
                arch,
                groups,
                k,
            )


def test_workspace_arguments_of_the_argument_plans():
    """``sched`` is followed by the queue split ``c0``, ``c1`` (a queue-walk program's
    ``sched``, its tile counters, is its last argument before the grid); ``wave_sync`` by
    ``wave_mode``; the ``main_wave`` stages bind programs taking ``wave_sync``."""
    queue_programs = {
        stage["program"]
        for route in ROUTES.values()
        for stage in route["stages"]
        if stage["name"].removesuffix(cb.ACC_SUFFIX) == cb.QUEUE_STAGE
    }
    for program_name, program in PROGRAMS.items():
        plan = [tuple(item) for item in program["arg_plan"]]
        if program_name in queue_programs:
            at = plan.index(("buffer", "sched"))
            assert all(kind == "grid" for kind, _ in plan[at + 1 :]), program_name
            assert ("parameter", "c0") not in plan, program_name
            plan = plan[:at] + plan[at + 1 :]
        for buffer, parameters in (
            ("sched", [("parameter", "c0"), ("parameter", "c1")]),
            ("wave_sync", [("parameter", "wave_mode")]),
        ):
            if ("buffer", buffer) in plan:
                at = plan.index(("buffer", buffer))
                assert plan[at + 1 : at + 1 + len(parameters)] == parameters
    for route in ROUTES.values():
        for stage in route["stages"]:
            plan = [tuple(item) for item in PROGRAMS[stage["program"]]["arg_plan"]]
            if stage["name"] in ("main_wave", "main_wave_halves"):
                assert ("buffer", "wave_sync") in plan
            if route["op"] != "wgrad":
                assert ("buffer", "wave_sync") not in plan
    rest = HOST_PLAN_CONSTANTS["sched_rest"]
    assert len(rest) == 3 and all(0 <= int(v) < 2**32 for v in rest)
    assert int(HOST_PLAN_CONSTANTS["sched_words"]) >= len(rest)
    assert int(HOST_PLAN_CONSTANTS["wave_sync_words"]) >= 1
    assert int(HOST_PLAN_CONSTANTS["queue_sched_words"]) == 2


def test_queue_stages_take_the_half_release_arguments():
    """The queue walk's stages: served by both 512-wide k tile routes (the accumulate
    stage by the fp32 route alone) on the architectures of the ``queue_walk`` rules and
    nowhere else, each program with the launch shape and FFI entry of the route's
    ``main_halves`` program on that architecture and its argument plan plus the
    ``sched`` buffer, which the weight-gradient binding binds to the ``queue_sched``
    stream workspace (the row programs' ``sched`` keeps its own kind).  Every rule names
    a stage the queue walk may stand in for, an output kind and a group range, and the
    routes serve the stage it binds.  (A CPU check.)"""
    assert cb.BINDING_WORKSPACE_KINDS == {"wgrad": {"sched": cb.QUEUE_SCHED}}
    assert cb.WORKSPACE_KINDS == cb.STREAM_WORKSPACES + (cb.QUEUE_SCHED,)
    assert cb.QUEUE_SELECTED_STAGES == ("main_halves", "main_wave_halves")
    rules = HOST_PLAN_CONSTANTS["queue_walk"]
    assert rules, "no queue_walk rule"
    rule_arches = {rule["arch"] for rule in rules}
    for rule in rules:
        assert rule["stage"] in cb.QUEUE_SELECTED_STAGES, rule
        assert set(rule["outputs"]) <= {"bfloat16", "float32", "accumulate"}, rule
        low, high = rule["groups"]
        assert 1 <= low <= high, rule
        assert rule["sm_counts"] and all(int(v) > 0 for v in rule["sm_counts"]), rule
    for out_dtype in ("bfloat16", "float32"):
        name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=512)
        for arch in ROUTES[name]["arches"]:
            served = cake_jit.stage_served(name, cb.QUEUE_STAGE, arch)
            assert served == (arch in rule_arches), (name, arch)
            acc_served = cake_jit.stage_served(
                name, cb.QUEUE_STAGE + cb.ACC_SUFFIX, arch
            )
            assert acc_served == (served and out_dtype == "float32"), (name, arch)
            if not served:
                continue
            for stage, twin in (
                (cb.QUEUE_STAGE, "main_halves"),
                (cb.QUEUE_STAGE + cb.ACC_SUFFIX, "main_halves" + cb.ACC_SUFFIX),
            ):
                if not cake_jit.stage_served(name, stage, arch):
                    continue
                binding = cake_jit.stage_binding(name, stage, arch)
                twin_binding = cake_jit.stage_binding(name, twin, arch)
                assert binding["specializations"] == twin_binding["specializations"]
                program = PROGRAMS[binding["program"]]
                halves = PROGRAMS[twin_binding["program"]]
                assert program["launch"] == halves["launch"], (name, stage, arch)
                assert program["ffi_entry"] == halves["ffi_entry"], (name, stage, arch)
                plan = [tuple(i) for i in program["arg_plan"]]
                assert plan.count(("buffer", "sched")) == 1, (name, stage, arch)
                assert [i for i in plan if i != ("buffer", "sched")] == [
                    tuple(i) for i in halves["arg_plan"]
                ], (name, stage, arch)
            if out_dtype == "float32":
                for rule in rules:
                    if rule["arch"] == arch and "accumulate" in rule["outputs"]:
                        assert acc_served, (name, arch, rule)
    for name, route in ROUTES.items():
        if route["op"] == "wgrad":
            continue
        for stage in route["stages"]:
            assert stage["name"].removesuffix(cb.ACC_SUFFIX) != cb.QUEUE_STAGE, name


def test_queue_stages_have_lead_twins():
    """The lead route of each 512-wide k tile route serves the queue walk's stages exactly
    where the route does, each on the lead twin of the route's queue program, whose
    argument plan takes ``m_begin`` and the ``sched`` buffer: an ``m_indptr`` row table
    binds the queue stage of the lead route, as every other stage.  (A CPU check.)"""
    seen = 0
    for out_dtype in ("bfloat16", "float32"):
        name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=512)
        lead = cake_jit.lead_route(name)
        assert lead in ROUTES, lead
        for arch in ROUTES[name]["arches"]:
            for stage in (cb.QUEUE_STAGE, cb.QUEUE_STAGE + cb.ACC_SUFFIX):
                served = cake_jit.stage_served(name, stage, arch)
                assert cake_jit.stage_served(lead, stage, arch) == served, (
                    name,
                    stage,
                    arch,
                )
                if not served:
                    continue
                program = cake_jit.stage_binding(name, stage, arch)["program"]
                twin = cake_jit.stage_binding(lead, stage, arch)["program"]
                assert twin == cake_jit.lead_twin(program), (name, stage, arch, twin)
                plan = [tuple(item) for item in PROGRAMS[twin]["arg_plan"]]
                assert ("buffer", "m_begin") in plan, (twin, plan)
                assert plan.count(("buffer", "sched")) == 1, (twin, plan)
                seen += 1
    assert seen, "no queue-walk stage on any 512-wide k tile route"


def test_wgrad_queue_walk_follows_its_rules():
    """``wgrad_queue_walk`` holds exactly where one rule of ``queue_walk`` names the
    architecture, the SM count, the selected stage (one of ``QUEUE_SELECTED_STAGES``),
    the output and a group range holding the group count.  (A CPU check.)"""
    c = HOST_PLAN_CONSTANTS
    stages = ("main", "main_wave", "main_halves_k256t", *cb.QUEUE_SELECTED_STAGES)
    outputs = ("bfloat16", "float32", "accumulate")
    for arch in cake_jit.ARCHES:
        for sms in (148, 152, 160, 212):
            for stage in stages:
                for output in outputs:
                    for groups in (1, 2, 4, 8, 16, 17, 32, 33, 64):
                        want = any(
                            rule["arch"] == arch
                            and sms in rule["sm_counts"]
                            and rule["stage"] == stage
                            and output in rule["outputs"]
                            and rule["groups"][0] <= groups <= rule["groups"][1]
                            for rule in c["queue_walk"]
                        )
                        got = cb.wgrad_queue_walk(stage, output, arch, sms, groups, c)
                        assert got == want, (arch, sms, stage, output, groups)
    assert not cb.wgrad_queue_walk(
        "main_halves", "bfloat16", "sm_100a", 148, 8, {"queue_walk": []}
    )
    # A bfloat16 row of the 148-SM sm_100a main_halves stage was slower on the queue walk.
    assert not cb.wgrad_queue_walk("main_halves", "bfloat16", "sm_100a", 148, 16, c)
    assert cb.wgrad_queue_walk("main_halves", "accumulate", "sm_100a", 148, 16, c)


# The keywords the host binds a stage's argument plan from, by the stage it binds: the
# forward and data-gradient stages, the weight-gradient main stages and the tail reduce.
# ``m_begin`` (the first entry of an ``m_indptr`` row table) goes to the lead twins only.
ROW_ARGUMENTS = {
    "A",
    "B",
    "C",
    "offs",
    "m_begin",
    "num_groups",
    "sum_m",
    "N",
    "K",
    "ldc",
    "c0",
    "c1",
}
WGRAD_MAIN_ARGUMENTS = {
    "A",
    "B",
    "C",
    "offs",
    "m_begin",
    "tensormap_workspace",
    "partials",
    "tail_splits",
    "raster_rows",
    "num_groups",
    "sum_m",
    "N",
    "K",
    "ldc",
    "stride_e",
    "lda",
    "ldb",
    "wave_mode",
}
TAIL_ARGUMENTS = {
    "partials",
    "C",
    "offs",
    "m_begin",
    "num_groups",
    "N",
    "K",
    "ldc",
    "stride_e",
    "num_clusters",
    "tail_splits",
    "raster_rows",
}
GRID_ARGUMENTS = {"grid_x", "grid_y", "grid_z"}


def test_argument_plans_name_only_bound_arguments():
    """Every argument the program of a route stage names, on every architecture, is one
    the host binds for that stage: a grid dimension, a stream workspace, a keyword of the
    stage's preparation, or an operand alias (``OPERAND_ALIASES``: a row-addressed base
    of ``A`` / ``B`` that a build takes beside its TMA descriptors).  The sets are the
    backend's ``HOST_ARGUMENTS``, which ``_bind`` holds the keywords to; ``m_begin`` is
    named by the lead twins' plans only (after ``offs``)."""
    assert cb.HOST_ARGUMENTS == {
        "row_op": frozenset(ROW_ARGUMENTS),
        "wgrad": frozenset(WGRAD_MAIN_ARGUMENTS),
        "tail_reduce": frozenset(TAIL_ARGUMENTS),
    }
    for name, route in ROUTES.items():
        for stage in route["stages"]:
            assert cb.host_binding(route["op"], stage["name"]) == (
                "row_op"
                if route["op"] != "wgrad"
                else "tail_reduce"
                if stage["name"].startswith(cb.TAIL_REDUCE)
                else "wgrad"
            ), (name, stage["name"])
            plan = [tuple(item) for item in PROGRAMS[stage["program"]]["arg_plan"]]
            if cake_jit.is_lead_route(name):
                assert plan[plan.index(("buffer", "offs")) + 1] == ("buffer", "m_begin")
            else:
                assert ("buffer", "m_begin") not in plan, (name, stage["name"])
    for name, route in ROUTES.items():
        for stage in route["stages"]:
            if route["op"] != "wgrad":
                bound = ROW_ARGUMENTS
            elif stage["name"].startswith(cb.TAIL_REDUCE):
                bound = TAIL_ARGUMENTS
            else:
                bound = WGRAD_MAIN_ARGUMENTS
            for kind, arg in PROGRAMS[stage["program"]]["arg_plan"]:
                if kind == "grid":
                    assert arg in GRID_ARGUMENTS, (name, stage["name"], arg)
                elif kind == "buffer" and arg in cb.STREAM_WORKSPACES:
                    continue
                else:
                    assert arg in bound or cb.OPERAND_ALIASES.get(arg) in bound, (
                        name,
                        stage["name"],
                        stage["program"],
                        arg,
                    )


def test_bind_passes_each_operand_alias_its_operand(monkeypatch):
    """A program whose argument plan names ``A_rows`` / ``B_rows`` gets the row bases of
    the tensors the host binds as ``A`` / ``B``, in the plan's order: on an architecture
    of ``ROW_BASE_READ_ARCHS`` the tensor itself when its rows are dense, else the flat
    span of its rows (no copy; the build reads it at the row pitch ``lda`` / ``ldb``);
    elsewhere a one-element view at its first element, whatever its row stride (the
    build does not read it).  An argument the host does not bind is refused with a
    ``KeyError`` naming it."""
    program = "cake_moe_grouped_det_gemm_alias_probe"
    arg_plan = [
        ["buffer", "A"],
        ["buffer", "A_rows"],
        ["buffer", "B_rows"],
        ["buffer", "B"],
        ["parameter", "K"],
        ["grid", "grid_x"],
        ["grid", "grid_y"],
        ["grid", "grid_z"],
    ]
    monkeypatch.setitem(
        PROGRAMS,
        program,
        {"arg_plan": arg_plan, "ffi_entry": "run", "arches": ["sm_100a", "sm_107a"]},
    )
    monkeypatch.setitem(
        ROUTES, "alias_probe", {"op": "fwd", "stages": [], "arches": ["sm_100a"]}
    )
    monkeypatch.setattr(
        cb,
        "stage_binding",
        lambda name, stage, arch: {"program": program, "specializations": {}},
    )
    entry = object()
    monkeypatch.setattr(
        cb, "load_module", lambda *args: type("Module", (), {"run": entry})
    )
    assert cb.ROW_BASE_READ_ARCHS == ("sm_107a",)
    a, b = torch.zeros(4, 8), torch.zeros(2, 6)
    padded = torch.zeros(4, 16)[:, :8]
    kwargs = dict(A=a, B=b, K=7, grid=(4, 2, 1))
    for arch in ("sm_100a", "sm_107a"):
        launch = cb._bind("alias_probe", "main", kwargs, arch)
        assert launch.entry is entry
        args = launch.arguments
        assert args[0] is a and args[3] is b and args[4:] == (7, 4, 2, 1)
        assert launch.workspace_slots == ()
        for row_base, t in ((args[1], a), (args[2], b)):
            if arch in cb.ROW_BASE_READ_ARCHS:
                assert row_base is t
            else:
                assert row_base.shape == (1,) and row_base.data_ptr() == t.data_ptr()
    launch = cb._bind("alias_probe", "main", dict(kwargs, A=padded), "sm_100a")
    assert launch.arguments[0] is padded
    assert launch.arguments[1].shape == (1,)
    assert launch.arguments[1].data_ptr() == padded.data_ptr()
    launch = cb._bind("alias_probe", "main", dict(kwargs, A=padded), "sm_107a")
    assert launch.arguments[0] is padded
    span = launch.arguments[1]
    assert span.shape == (3 * 16 + 8,) and span.stride() == (1,)
    assert span.is_contiguous() and span.data_ptr() == padded.data_ptr()
    assert span[-1].data_ptr() == padded[-1, -1].data_ptr()
    assert launch.arguments[2] is b
    arg_plan.insert(1, ["buffer", "C_rows"])
    with pytest.raises(KeyError, match="C_rows"):
        cb._bind("alias_probe", "main", kwargs, "sm_100a")


def test_k256_wave_build_launches_only_on_its_wave_archs():
    """The 256-wide k tile's wave-synchronised build (``main_wave``) launches only on
    the architectures ``wave_sync_archs[256]`` lists, never on sm_107a, whatever ``K``
    and whether or not the route also builds it there; every architecture of the route
    keeps its weight gradient (the ``main`` stage, ``program_registered``)."""
    c = HOST_PLAN_CONSTANTS
    assert "sm_107a" not in c["wave_sync_archs"][256]
    min_k = int(c["wave_sync_min_k"])
    for out_dtype in ("bfloat16", "float32"):
        name = cake_jit.record_name("wgrad", out_dtype=out_dtype, tile_k=256)
        for arch in ROUTES[name]["arches"]:
            assert cake_jit.stage_served(name, "main", arch), (name, arch)
            assert cake_jit.program_registered(
                "wgrad", arch, out_dtype=out_dtype, tile_k=256
            )
            launched = arch in c["wave_sync_archs"][256]
            if launched:
                assert cake_jit.stage_served(name, "main_wave", arch), (name, arch)
            for k in (min_k, 4 * min_k):
                stage, wave = cb.select_wgrad_stage(
                    name, k, 256, arch, c, sum_m=EXPERT_ROWS, num_groups=4
                )
                assert wave == launched, (name, arch, k)
                assert stage == ("main_wave" if launched else "main"), (name, arch, k)


# ---------------------------------------------------------------------------
# GPU correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 1024, 512), (TINY_E5, 256, 64)]
)
def test_fwd_matches_reference(sizes, n, k, record_property):
    device = _require_program("fwd")

    def draw(d):
        x, w, _g, offs = _inputs(sizes, n, k, device, seed=d.seed)
        out = cake_moe_grouped_gemm_det_train_forward(x, w, offs)
        assert out.shape == (sum(sizes), n)
        d.op("Y", out, "fwd", x, w, sizes)

    judge_draws(record_property, draw, seed=0)


@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 256, 512), (RAGGED_E6, 1024, 512), (TINY_E5, 64, 256)]
)
def test_dgrad_matches_reference(sizes, n, k, record_property):
    device = _require_program("dgrad")

    def draw(d):
        _x, w, g, offs = _inputs(sizes, n, k, device, seed=d.seed)
        out = cake_moe_grouped_gemm_det_train_dgrad(g, w, offs)
        assert out.shape == (sum(sizes), k)
        d.op("dX", out, "dgrad", g, w, sizes)

    judge_draws(record_property, draw, seed=0)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "sizes,n,k", [(RAGGED_E4, 512, 256), (RAGGED_E6, 512, 256), (TINY_E5, 256, 512)]
)
def test_wgrad_matches_reference(sizes, n, k, out_dtype, record_property):
    device = _require_program("wgrad", out_dtype)
    case = _Case(sizes, n, k, device)
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )

    def draw(d):
        case.refill(d.seed)
        out = launch.launch()
        assert out.shape == (len(sizes), n, k) and out.dtype == out_dtype
        d.op("dW", out, "wgrad", case.g, case.x, sizes)
        for e, (s, t) in enumerate(_bounds(sizes)):
            if t == s:
                d.exact(f"dW[{e}] (empty group)", out[e], torch.zeros_like(out[e]))
        # Deterministic: a second launch into the same output is bitwise identical.
        first = out.clone()
        out.fill_(float("nan"))
        launch.launch()
        torch.cuda.synchronize()
        assert torch.equal(out, first)

    judge_draws(record_property, draw, seed=0)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_long_reduction_selects_a_valid_tile(out_dtype, record_property):
    device = _require_program("wgrad", out_dtype)
    n, k = 512 if out_dtype == torch.bfloat16 else 256, 1024
    case = _Case(LONG_E3, n, k, device, seed=3)
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )
    assert launch.plan["tile_k"] in (256, 512) and k % launch.plan["tile_k"] == 0

    def draw(d):
        case.refill(d.seed)
        out = launch.launch()
        d.op("dW", out, "wgrad", case.g, case.x, LONG_E3)
        d.exact("dW[1] (empty group)", out[1], torch.zeros_like(out[1]))

    judge_draws(record_property, draw, seed=case.seed)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_short_groups_never_include_the_next_group(out_dtype, record_property):
    """A tile of a group shorter than one k-step must not add the rows of the group
    the same CTA reduces next (#6570).  At least SMALL_GROUP_LAUNCHES launches of one
    prepared call, spread over the draws: every launch is judged (the draw keeps its
    largest error) and gives the bits of its draw's first launch."""
    device = _require_program("wgrad", out_dtype)
    n, k = 512, 3072
    case = _Case(SMALL_E512, n, k, device, seed=6)
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )
    per_draw = -(-SMALL_GROUP_LAUNCHES // DRAWS)
    differ = []

    def draw(d):
        case.refill(d.seed)
        first = None
        for call in range(per_draw):
            out = launch.launch()
            d.op("dW", out, "wgrad", case.g, case.x, SMALL_E512)
            if first is None:
                first = out.clone()
            elif not torch.equal(out, first):
                differ.append((d.index, call, int((out != first).sum())))

    def after():
        assert not differ, (
            f"{len(differ)} of {DRAWS * (per_draw - 1)} relaunches differ from their draw's "
            f"first launch (draw, launch, elements): {differ[:3]}"
        )

    judge_draws(record_property, draw, seed=case.seed, then=after)


def test_prepared_launch_reuses_buffers_without_allocation(record_property):
    device = _require_program("fwd")
    case = _Case(RAGGED_E6, 1024, 512, device)
    out = torch.full(
        (sum(RAGGED_E6), 1024), float("nan"), dtype=torch.bfloat16, device=device
    )
    launch = cake_moe_grouped_gemm_det_train_prepare_forward(
        case.x, case.w, case.offs, out=out
    )
    assert launch.launch() is out
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    launch.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    _judge_launch(record_property, "fwd", launch, case)


def _u32_words(tensor):
    """The words of a u32 workspace as Python ints."""
    return [v & 0xFFFFFFFF for v in tensor.view(torch.int32).tolist()]


def _workspace_kinds(launch):
    return {kind for item in launch.launches for _, kind in item.workspace_slots}


def _assert_queue_at_rest(device, stream=None):
    """The queue walk's counters kept for ``stream`` (the current stream by default; if a
    queue launch kept them) are ``queue_sched_words`` zero words."""
    words = cb.stream_workspace(cb.QUEUE_SCHED, device, stream)
    if words is None:
        return
    assert words.numel() == int(HOST_PLAN_CONSTANTS["queue_sched_words"])
    assert _u32_words(words) == [0] * words.numel(), _u32_words(words)


def _prepare_row_op(op, sizes, n, k, device, out=None):
    """A prepared ``fwd`` / ``dgrad`` launch on the inputs of seed 0, and its :class:`_Case`."""
    case = _Case(sizes, n, k, device)
    if op == "fwd":
        return cake_moe_grouped_gemm_det_train_prepare_forward(
            case.x, case.w, case.offs, out=out
        ), case
    return cake_moe_grouped_gemm_det_train_prepare_dgrad(
        case.g, case.w, case.offs, out=out
    ), case


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_scheduler_workspace_rests_after_every_launch(op, monkeypatch, record_property):
    """The kernels re-arm ``sched`` at the end of every launch: repeated launches of
    one prepared call on one stream each compute every tile (the output is refilled
    with NaN before each), and the stream's workspace is at its rest words.  Where the
    launch probes the queue split, the probe's add-zero atomics leave it at rest too
    and the later launches pass the probed split."""
    device = _require_program(op)
    _fresh_stream_workspaces(monkeypatch)
    launch, case = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(launch):
        pytest.skip(
            f"the {op} program bound on this device takes no scheduler workspace"
        )
    c = HOST_PLAN_CONSTANTS

    def after():
        torch.cuda.synchronize()
        sched = cb.stream_workspace("sched", device)
        assert sched is not None and sched.dtype == torch.uint32
        assert sched.numel() == int(c["sched_words"]) and sched.is_contiguous()
        assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
        if _probes_queue_split(device, op):
            assert launch.plan["queue_split"] == "probed"
            assert cb.stream_queue_split(device) is not None

    _judge_launch(record_property, op, launch, case, launches=4, then=after)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_wave_sync_route_and_rest_state(out_dtype, monkeypatch, record_property):
    """From ``wave_sync_min_k`` on, an architecture ``wave_sync_archs`` lists for the
    k tile, whose route has a ``main_wave`` stage, launches the wave build with
    its ``wave_sync_mode`` as ``wave_mode``; it leaves its counters at zero and
    gives the plain build's output bits."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 256, int(c["wave_sync_min_k"])
    case = _Case(RAGGED_E4, n, k, device, seed=7)
    x, g, offs = case.x, case.g, case.offs
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g, x, offs, out_dtype=out_dtype
    )
    tile_k = launch.plan["tile_k"]
    wave = arch in c["wave_sync_archs"][tile_k] and cake_jit.stage_served(
        launch.record_name, "main_wave", arch
    )
    assert launch.stages[0] == ("main_wave" if wave else "main")
    assert launch.plan["wave_mode"] == (int(c["wave_sync_mode"][arch]) if wave else 0)
    assert ("wave_sync" in _workspace_kinds(launch)) == wave

    def after():
        out = launch.out
        torch.cuda.synchronize()
        if wave:
            counters = cb.stream_workspace("wave_sync", device)
            assert counters is not None and counters.dtype == torch.uint32
            assert counters.numel() == int(c["wave_sync_words"])
            assert _u32_words(counters) == [0] * int(c["wave_sync_words"])
        # Below the threshold (or with the wave build switched off) the plain build runs.
        short = cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x[:, : k // 2].contiguous(), offs, out_dtype=out_dtype
        )
        assert short.stages[0] == "main" and short.plan["wave_mode"] == 0
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wave_sync_min_k", 1 << 30)
        plain = cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out_dtype=out_dtype
        )
        assert plain.stages[0] == "main" and plain.plan["wave_mode"] == 0
        assert plain.plan["tile_k"] == launch.plan["tile_k"]
        torch.cuda.synchronize()
        assert torch.equal(plain.launch(), out)

    _judge_launch(record_property, "wgrad", launch, case, launches=3, then=after)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_wave_sync_takes_row_padded_operands(out_dtype):
    """The 256-wide k tile's weight gradient takes ``g`` and ``x`` with padded rows (the
    leading columns of wider NaN-filled tensors) and gives the bits of contiguous
    copies.  Its ``main_wave`` build on sm_100a / sm_103a names the operands' row bases
    and never reads them; the host binds a one-element view there
    (``cake_backend.ROW_BASE_READ_ARCHS``)."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    sizes, n, k = [2048] * 8, 4096, 4096
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=11)

    def padded(t):
        wide = torch.full(
            (t.size(0), t.size(1) + 256), float("nan"), dtype=t.dtype, device=t.device
        )
        wide[:, : t.size(1)] = t
        return wide[:, : t.size(1)]

    gp, xp = padded(g), padded(x)
    assert not gp.is_contiguous() and not xp.is_contiguous()
    dense = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g, x, offs, out_dtype=out_dtype
    )
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        gp, xp, offs, out_dtype=out_dtype
    )
    wave = arch in c["wave_sync_archs"][256] and cake_jit.stage_served(
        launch.record_name, "main_wave", arch
    )
    assert launch.plan["tile_k"] == 256, launch.plan
    assert launch.stages[0] == ("main_wave" if wave else "main"), launch.stages
    assert launch.stages == dense.stages
    ref = dense.launch().clone()
    launch.out.fill_(float("nan"))
    out = launch.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("operand", ["g", "x"])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_refuses_row_pitches_off_16_bytes(operand, out_dtype):
    """The weight gradient of a ``g`` or an ``x`` whose row stride is not a multiple of
    16 bytes (the leading columns of a tensor 4 columns wider), on the shape of
    ``test_wgrad_wave_sync_takes_row_padded_operands``: its launch raises ``ValueError``
    before any kernel runs (the binding's TMA descriptor of the operand refuses the
    pitch; on sm_107a the ``main`` build also reads the rows at that pitch, ``lda`` /
    ``ldb``), and ``out``, NaN-filled, keeps its bytes."""
    device = _require_program("wgrad", out_dtype)
    sizes, n, k = [2048] * 8, 4096, 4096
    x, _w, g, offs = _inputs(sizes, n, k, device, seed=11)
    operands = {"g": g, "x": x}
    dense = operands[operand]
    cols = dense.size(1)
    wide = torch.zeros(dense.size(0), cols + 4, dtype=dense.dtype, device=device)
    wide[:, :cols] = dense
    operands[operand] = wide[:, :cols]
    pitch = operands[operand].stride(0) * operands[operand].element_size()
    assert pitch % 16, pitch
    out = torch.full((len(sizes), n, k), float("nan"), dtype=out_dtype, device=device)
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        operands["g"], operands["x"], offs, out=out
    )
    assert launch.plan["tile_k"] == 256, launch.plan
    torch.cuda.synchronize()
    before = out.view(torch.uint8).clone()
    with pytest.raises(ValueError, match="multiples of 16 bytes"):
        launch.launch()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), before), f"{operand} pitch {pitch}"


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_wide_tile_gives_the_256_column_tile_bits(op, monkeypatch, record_property):
    """On an architecture of ``wide_fwd_archs`` / ``wide_dgrad_archs`` an output
    width that is a multiple of ``wide_block_n`` launches the 512-column tile; it
    leaves ``sched`` at rest and gives the 256-column tile's output bits (both
    accumulate each element's k steps in order and round once)."""
    device = _require_program(op)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = (1536, 512) if op == "fwd" else (512, 1536)
    launch, case = _prepare_row_op(op, RAGGED_E6, n, k, device)
    wide = arch in c[f"wide_{op}_archs"] and cake_jit.stage_served(
        launch.record_name, "main_wide", arch
    )
    # Where an evict-first rule holds the wide tile is its twin (``main_wide_ef``).
    ef = (
        wide
        and _evict_first(op, k, arch, sum(RAGGED_E6), len(RAGGED_E6), c)
        and cake_jit.stage_served(launch.record_name, "main_wide_ef", arch)
    )
    ts33 = (
        wide
        and not ef
        and _wide_ts33(op, k, arch, sum(RAGGED_E6), len(RAGGED_E6), c)
        and cake_jit.stage_served(launch.record_name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )
    assert launch.plan["tile_n"] == int(c["wide_block_n" if wide else "block_n"])

    def after():
        torch.cuda.synchronize()
        if not wide:
            pytest.skip(f"no 512-column {op} tile on {arch}")
        sched = cb.stream_workspace("sched", device)
        assert sched is not None
        assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, f"wide_{op}_archs", [])
        narrow, _ = _prepare_row_op(op, RAGGED_E6, n, k, device)
        assert narrow.stages[0] == "main"
        assert narrow.plan["tile_n"] == int(c["block_n"])
        assert narrow.plan["clusters"] >= launch.plan["clusters"]
        torch.cuda.synchronize()
        assert torch.equal(narrow.launch(), launch.out)

    _judge_launch(record_property, op, launch, case, launches=3, then=after)


def test_dgrad_evict_first_gives_the_wide_tile_bits(monkeypatch, record_property):
    """Where ``dgrad_evict_first`` holds, the data gradient launches the evict-first
    twin of its wide tile (``main_wide_ef``).  The cache policy changes no arithmetic:
    the output bits are those of the wide tile and of the 256-column tile."""
    device = _require_program("dgrad")
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 64, int(c["dgrad_evict_first_min_k"])
    sizes = EVICT_FIRST_E2
    assert sum(sizes) // len(sizes) >= max(
        int(v) for v in c["dgrad_evict_first_min_rows"].values()
    )
    launch, case = _prepare_row_op("dgrad", sizes, n, k, device)
    name = launch.record_name
    ef = arch in c["dgrad_evict_first_archs"] and cake_jit.stage_served(
        name, "main_wide_ef", arch
    )
    wide = arch in c["wide_dgrad_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ts33 = (
        wide
        and not ef
        and cb.dgrad_wide_ts33(k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )

    def after():
        torch.cuda.synchronize()
        if not ef:
            pytest.skip(f"no evict-first data-gradient tile on {arch}")
        monkeypatch.setitem(
            HOST_PLAN_CONSTANTS,
            "dgrad_evict_first_min_rows",
            dict.fromkeys(c["dgrad_evict_first_min_rows"], 1 << 30),
        )
        plain, _ = _prepare_row_op("dgrad", sizes, n, k, device)
        assert plain.stages[0] == "main_wide"
        assert plain.plan["tile_n"] == launch.plan["tile_n"]
        torch.cuda.synchronize()
        assert torch.equal(plain.launch(), launch.out)
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_dgrad_archs", [])
        narrow, _ = _prepare_row_op("dgrad", sizes, n, k, device)
        assert narrow.stages[0] == "main"
        torch.cuda.synchronize()
        assert torch.equal(narrow.launch(), launch.out)

    _judge_launch(record_property, "dgrad", launch, case, launches=2, then=after)


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_wide_ts33_gives_the_wide_tile_bits(op, monkeypatch, record_property):
    """Where ``fwd_wide_ts33`` / ``dgrad_wide_ts33`` holds, the forward pass / data
    gradient launches the twin of its wide tile with three head and three tail k steps
    around the release of the accumulator's halves (``main_wide_ts33``).  Every element
    still accumulates its k steps in order: the output bits are those of the wide tile
    and of the 256-column tile, a CUDA-graph replay gives them too, and the launch
    leaves ``sched`` at rest."""
    device = _require_program(op)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    if op == "dgrad":  # the gate/up data gradient over few groups
        n, k, sizes = 64, int(c["dgrad_wide_ts33_min_k"]), EVICT_FIRST_E2
    else:  # the down projection over many small groups
        n, k = 2 * int(c["wide_block_n"]), int(c["fwd_wide_ts33_max_k"])
        sizes = RAGGED_E6
    launch, case = _prepare_row_op(op, sizes, n, k, device)
    name = launch.record_name
    ef = _evict_first(op, k, arch, sum(sizes), len(sizes), c) and cake_jit.stage_served(
        name, "main_wide_ef", arch
    )
    wide = arch in c[f"wide_{op}_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ts33 = (
        wide
        and not ef
        and _wide_ts33(op, k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )

    def after():
        torch.cuda.synchronize()
        if not ts33:
            pytest.skip(f"no 3 / 3 head / tail {op} tile on {arch}")
        sched = cb.stream_workspace("sched", device)
        assert sched is not None
        assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
        captured = launch.out.clone()
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            launch.launch()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize()
        assert torch.equal(launch.out, captured)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch.launch()
        for _ in range(2):
            launch.out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(launch.out, captured)
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, f"{op}_wide_ts33_archs", [])
        plain, _ = _prepare_row_op(op, sizes, n, k, device)
        assert plain.stages[0] == "main_wide"
        assert plain.plan["tile_n"] == launch.plan["tile_n"]
        torch.cuda.synchronize()
        assert torch.equal(plain.launch(), launch.out)
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, f"wide_{op}_archs", [])
        narrow, _ = _prepare_row_op(op, sizes, n, k, device)
        assert narrow.stages[0] == "main"
        torch.cuda.synchronize()
        assert torch.equal(narrow.launch(), launch.out)

    _judge_launch(record_property, op, launch, case, launches=3, then=after)


@pytest.mark.parametrize("op", ["fwd", "dgrad"])
def test_probed_queue_split_gives_the_even_split_bits(op, monkeypatch, record_property):
    """On the architectures of ``domain_split_archs`` a launch of an op in
    ``domain_split_ops`` whose weight has at most ``domain_split_max_weight_elems``
    elements probes its stream's scheduler workspace at its first launch there and
    passes the measured SM pairs of each die as ``c0`` / ``c1`` once the probe has
    completed; the other ops pass the even split and start no probe.  The split only
    divides the tiles between the claim queues: every launch gives the bits of the
    even split, the workspace stays at rest, and a second stream probes a workspace
    of its own."""
    device = _require_program(op)
    _fresh_stream_workspaces(monkeypatch)
    c = HOST_PLAN_CONSTANTS
    launch, case = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(launch):
        pytest.skip(
            f"the {op} program bound on this device takes no scheduler workspace"
        )
    probed = _probes_queue_split(device, op)
    assert launch.plan["queue_split"] == ("probed" if probed else "even")
    main = torch.cuda.current_stream(device)
    outs, splits = [], []
    for stream in (torch.cuda.Stream(device), torch.cuda.Stream(device)):
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch.out.fill_(float("nan"))
                launch.launch()
                outs.append(launch.out.clone())
                stream.synchronize()
                splits.append(cb.stream_queue_split(device, stream))
            sched = cb.stream_workspace("sched", device, stream)
            assert sched is not None
            assert _u32_words(sched[:3]) == [int(v) for v in c["sched_rest"]]
        main.wait_stream(stream)
    torch.cuda.synchronize()
    assert all(torch.equal(out, outs[0]) for out in outs)

    # The launches of both streams now pass the probed split (where the launch probes):
    # judge it over the draws, then (``after``) the even split's bits on the case's own
    # inputs.
    def after():
        assert torch.equal(launch.out, outs[0])
        if not probed:
            assert splits == [None] * len(splits)
            pytest.skip(f"no probed queue split for {op} on this device")
        # The first launch on each stream enqueues the probe and passes the even split;
        # the stream is synchronized after it, so the next launches pass the probe's split.
        for split in splits:
            assert split is not None
            c0, c1 = split
            assert c0 >= 1 and c1 >= 1 and c0 + c1 == cb.sm_count(device) // 2
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "domain_split_archs", [])
        even, _ = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
        assert even.plan["queue_split"] == "even"
        assert (even.plan["c0"], even.plan["c1"]) == (launch.plan["c0"], launch.plan["c1"])
        torch.cuda.synchronize()
        assert torch.equal(even.launch(), outs[0])

    _judge_launch(record_property, op, launch, case, then=after)


@pytest.mark.parametrize("probe", ["no_compiler", "no_sm_seen"])
def test_queue_split_stays_even_without_a_probe(probe, monkeypatch, record_property):
    """Where the queue probe cannot be built (no CUDA compiler) or does not see every
    SM (here: a probe that records nothing), a probed launch keeps passing the even
    split, with the same bits."""
    device = _require_program("dgrad")
    if not _probes_queue_split(device, "dgrad"):
        pytest.skip("no probed queue split on this device")
    _fresh_stream_workspaces(monkeypatch)

    class _NoRecord:
        @staticmethod
        def cake_queue_probe(sched, lat, ctas, smem):
            return None

    module = None if probe == "no_compiler" else _NoRecord()
    monkeypatch.setattr(cb, "_queue_probe_module", lambda capability: module)
    launch, case = _prepare_row_op("dgrad", RAGGED_E6, 1024, 512, device)
    assert launch.plan["queue_split"] == "probed"

    def after():
        torch.cuda.synchronize()
        assert cb.stream_queue_split(device) is None

    _judge_launch(record_property, "dgrad", launch, case, launches=3, then=after)


def test_fwd_lwh_ef_gives_the_wide_tile_bits(monkeypatch, record_property):
    """Where ``fwd_lwh_ef`` holds (a short reduction, the down projection), the forward
    pass launches the twin of its wide tile that releases the last wave in halves and
    stores Y evict-first (``main_wide_ef``).  Neither changes the arithmetic: the
    output bits are those of the wide tile and of the 256-column tile."""
    device = _require_program("fwd")
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 2 * int(c["wide_block_n"]), int(c["fwd_lwh_ef_max_k"])
    sizes = EVICT_FIRST_E2
    launch, case = _prepare_row_op("fwd", sizes, n, k, device)
    name = launch.record_name
    ef = arch in c["fwd_lwh_ef_archs"] and cake_jit.stage_served(
        name, "main_wide_ef", arch
    )
    wide = arch in c["wide_fwd_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ts33 = (
        wide
        and not ef
        and cb.fwd_wide_ts33(k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )

    def after():
        torch.cuda.synchronize()
        if not ef:
            pytest.skip(f"no last-wave-halves forward tile on {arch}")
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "fwd_lwh_ef_max_k", k - 1)
        plain, _ = _prepare_row_op("fwd", sizes, n, k, device)
        assert plain.stages[0] == "main_wide"
        assert plain.plan["tile_n"] == launch.plan["tile_n"]
        torch.cuda.synchronize()
        assert torch.equal(plain.launch(), launch.out)
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_fwd_archs", [])
        narrow, _ = _prepare_row_op("fwd", sizes, n, k, device)
        assert narrow.stages[0] == "main"
        torch.cuda.synchronize()
        assert torch.equal(narrow.launch(), launch.out)

    _judge_launch(record_property, "fwd", launch, case, launches=2, then=after)


@pytest.mark.parametrize("sizes", [RAGGED_E6, EVICT_FIRST_E2], ids=["ragged", "long"])
def test_fwd_evict_first_gives_the_wide_tile_bits(sizes, monkeypatch, record_property):
    """Where ``fwd_evict_first`` holds (a long reduction, the gate/up projection), the
    forward pass launches the twin of its wide tile that loads X and stores Y with the
    L2 evict-first policy (``main_wide_ef``), whatever the group sizes (an empty group
    and a one-row group among them).  The cache policy changes no arithmetic: five
    launches and a CUDA-graph replay give one set of output bits, those of the wide
    tile and of the 256-column tile."""
    device = _require_program("fwd")
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 2 * int(c["wide_block_n"]), int(c["fwd_evict_first_min_k"])
    launch, case = _prepare_row_op("fwd", sizes, n, k, device)
    name = launch.record_name
    wide = arch in c["wide_fwd_archs"] and cake_jit.stage_served(
        name, "main_wide", arch
    )
    ef = (
        wide
        and _evict_first("fwd", k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ef", arch)
    )
    if arch in c["fwd_evict_first_archs"]:
        assert ef, (name, arch)
    ts33 = (
        wide
        and not ef
        and cb.fwd_wide_ts33(k, arch, sum(sizes), len(sizes), c)
        and cake_jit.stage_served(name, "main_wide_ts33", arch)
    )
    assert launch.stages[0] == (
        "main_wide_ef"
        if ef
        else "main_wide_ts33"
        if ts33
        else "main_wide"
        if wide
        else "main"
    )

    def after():
        first = launch.out.view(torch.int16).clone()
        torch.cuda.synchronize()
        if not ef:
            pytest.skip(f"no X / Y evict-first forward tile on {arch}")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch.launch()
        launch.out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(launch.out.view(torch.int16), first)
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "fwd_evict_first_archs", [])
        plain, _ = _prepare_row_op("fwd", sizes, n, k, device)
        assert plain.stages[0] == "main_wide"
        assert plain.plan["tile_n"] == launch.plan["tile_n"]
        torch.cuda.synchronize()
        assert torch.equal(plain.launch().view(torch.int16), first)
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "wide_fwd_archs", [])
        narrow, _ = _prepare_row_op("fwd", sizes, n, k, device)
        assert narrow.stages[0] == "main"
        torch.cuda.synchronize()
        assert torch.equal(narrow.launch().view(torch.int16), first)

    _judge_launch(record_property, "fwd", launch, case, launches=5, then=after)


@pytest.mark.parametrize("wave", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_k512_halves_gives_the_drain_build_bits(
    out_dtype, wave, monkeypatch, record_property
):
    """With the 512-wide k tile on an architecture of ``k512_release_archs`` the
    weight gradient launches the twin of its build that releases the accumulator in
    two halves (``main_halves``; ``main_wave_halves`` from ``wave_sync_min_k``).  The
    output bits are those of the build that drains the whole accumulator."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    # The k256-tail twin of main_halves has its own test.
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k256_tail_archs", [])
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    n, k = 256, (int(c["wave_sync_min_k"]) if wave else 1024)
    case = _Case(K512_E2, n, k, device, seed=10)
    x, g, offs = case.x, case.g, case.offs
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g, x, offs, out_dtype=out_dtype
    )
    if launch.plan["tile_k"] != 512:
        assert arch not in c["k512_archs"]
        pytest.skip(f"the weight gradient takes no 512-wide k tile on {arch}")
    name = launch.record_name
    base = "main_wave" if launch.plan["wave_mode"] else "main"
    assert (base == "main_wave") == (
        wave
        and arch in c["wave_sync_archs"][512]
        and cake_jit.stage_served(name, "main_wave", arch)
    )
    halves = arch in c["k512_release_archs"]
    assert launch.stages[0] == (f"{base}_halves" if halves else base)

    def after():
        out = launch.out
        torch.cuda.synchronize()
        if base == "main_wave":
            counters = cb.stream_workspace("wave_sync", device)
            assert _u32_words(counters) == [0] * int(c["wave_sync_words"])
        if not halves:
            pytest.skip(f"no half-release twin on {arch}")
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k512_release_archs", [])
        drain = cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out_dtype=out_dtype
        )
        assert drain.stages[0] == base and drain.plan["tile_k"] == 512
        assert drain.plan["wave_mode"] == launch.plan["wave_mode"]
        torch.cuda.synchronize()
        assert torch.equal(drain.launch(), out)

    _judge_launch(record_property, "wgrad", launch, case, launches=2, then=after)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("label", list(QUEUE_CASE_BY_LABEL))
def test_wgrad_queue_walk_gives_the_selected_stage_bits(
    label, out_dtype, monkeypatch, record_property
):
    """Where a ``queue_walk`` rule names the call (forced here for the device, the
    selected stage and the output, at any group count), the weight gradient binds the
    queue walk (``cb.QUEUE_STAGE``, then the ordered tail reduce of a split tail) in place
    of the stage the host selects, with that stage's plan.  Its counters
    (``cb.QUEUE_SCHED``) are back at zero after the launches, and its output, judged
    against the naive BF16 tier and run to run, has the bits of the selected stage's
    launch (the same call with no rule)."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    _label, sizes, n, k = QUEUE_CASE_BY_LABEL[label]
    case = _Case(sizes, n, k, device, seed=13)
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", [])
    selected = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )
    name = selected.record_name
    if not cake_jit.stage_served(name, cb.QUEUE_STAGE, arch):
        pytest.skip(f"no queue-walk weight-gradient build on {arch}")
    stage = selected.stages[0]
    assert selected.plan["tile_k"] == 512, selected.plan
    assert stage == (
        "main_wave_halves" if k >= int(c["wave_sync_min_k"]) else "main_halves"
    )
    rule = {
        "arch": arch,
        "sm_counts": [cb.sm_count(device)],
        "stage": stage,
        "outputs": [DTYPE_NAMES[out_dtype]],
        "groups": [1, 1 << 30],
    }
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", [rule])
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )
    assert launch.stages == (cb.QUEUE_STAGE, *selected.stages[1:]), launch.stages
    assert launch.plan == selected.plan
    assert _workspace_kinds(launch) == {cb.QUEUE_SCHED}, _workspace_kinds(launch)

    def after():
        out = launch.out.clone()
        torch.cuda.synchronize()
        words = cb.stream_workspace(cb.QUEUE_SCHED, device)
        assert words is not None
        assert _u32_words(words) == [0] * int(c["queue_sched_words"])
        bits = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[out.dtype]
        assert torch.equal(selected.launch().view(bits), out.view(bits))

    _judge_launch(record_property, "wgrad", launch, case, launches=2, then=after)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_k256_tail_gives_the_k256_tile_bits(out_dtype, monkeypatch, record_property):
    """Where ``wgrad_k256_tail`` holds, the half-release 512-wide k tile launches its
    twin that runs the tiles left after its full waves as 256-column half tiles
    (``main_halves_k256t``): n-major, the last partial wave of them split as the
    256-wide k tile splits its tail and summed by that tile's tail reduce.  Each
    output element accumulates its rows in the order of the 256-wide k tile, so where
    the two plans split the same tiles the same way the output bits are that tile's."""
    device = _require_program("wgrad", out_dtype)
    c = HOST_PLAN_CONSTANTS
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    # A down-projection-like output (K at most k256_tail_max_k) over two groups of more
    # than k256_tail_min_rows rows on average, with 128 512-wide tiles, more than the
    # 106 clusters of a 212-SM device: the 256-column tail is split and reduced.
    sizes = K512_E2
    n, k = 4096, int(c["k256_tail_max_k"])
    assert sum(sizes) // len(sizes) >= int(c["k256_tail_min_rows"])
    case = _Case(sizes, n, k, device, seed=11)
    x, g, offs = case.x, case.g, case.offs
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        g, x, offs, out_dtype=out_dtype
    )
    if arch not in c["k256_tail_archs"]:
        assert launch.stages[0] != "main_halves_k256t"
        pytest.skip(f"no k256-tail weight-gradient build on {arch}")
    assert cake_jit.stage_served(launch.record_name, "main_halves_k256t", arch)
    plan = launch.plan
    assert plan["tile_k"] == 512 and plan["tail_tile_k"] == cb.K256_TAIL_TILE_K
    assert plan["raster_rows"] == 0
    tail = ("tail_reduce",) if plan["tail_splits"] > 1 else ()
    assert launch.stages == ("main_halves_k256t", *tail)

    def after():
        outs = [launch.out.clone()]
        torch.cuda.synchronize()
        monkeypatch.setitem(HOST_PLAN_CONSTANTS, "k512_archs", [])
        k256 = cake_moe_grouped_gemm_det_train_prepare_wgrad(
            g, x, offs, out_dtype=out_dtype
        )
        assert k256.plan["tile_k"] == cb.K256_TAIL_TILE_K and k256.stages[0] == "main"
        keys = ("clusters", "num_tail", "tail_splits", "raster_rows")
        if any(k256.plan[key] != plan[key] for key in keys):
            pytest.skip(f"the 256-wide k tile splits other tiles on this {arch} device")
        torch.cuda.synchronize()
        assert torch.equal(k256.launch(), outs[0])

    _judge_launch(record_property, "wgrad", launch, case, launches=2, then=after)


def test_two_streams_take_separate_workspaces(monkeypatch, record_property):
    """Launches on two streams in flight together never share a workspace; each
    stream keeps its own buffer at rest, and both results are exact.  Where a launch
    probes the queue split (the data gradient), its stream's workspace has a probe of
    its own; the forward's stream has none."""
    device = _require_program("fwd")
    _fresh_stream_workspaces(monkeypatch)
    first, case = _prepare_row_op("fwd", RAGGED_E6, 1024, 512, device)
    if "sched" not in _workspace_kinds(first):
        pytest.skip("the fwd program bound on this device takes no scheduler workspace")
    second, case2 = _prepare_row_op("dgrad", RAGGED_E4, 256, 512, device)
    streams = [torch.cuda.Stream(device), torch.cuda.Stream(device)]
    main = torch.cuda.current_stream(device)
    for stream in streams:
        stream.wait_stream(main)
    # One launch per stream alone first: a probe sees every SM only when no other
    # kernel holds one.
    for stream, launch in zip(streams, (first, second), strict=True):
        with torch.cuda.stream(stream):
            launch.launch()
        stream.synchronize()

    def draw(d):
        # each draw: both cases refilled, 8 rounds of a launch on each stream in flight
        # together; the last round's outputs are judged
        case.refill(d.seed)
        case2.refill(d.seed)
        for stream in streams:
            stream.wait_stream(main)
        for _ in range(8):
            for stream, launch in zip(streams, (first, second), strict=True):
                with torch.cuda.stream(stream):
                    launch.launch()
        for stream in streams:
            main.wait_stream(stream)
        torch.cuda.synchronize()
        d.op("Y", first.out, "fwd", *case.operands("fwd"), case.sizes)
        d.op("dX", second.out, "dgrad", *case2.operands("dgrad"), case2.sizes)

    def after():
        buffers = [cb.stream_workspace("sched", device, stream) for stream in streams]
        assert all(buffer is not None for buffer in buffers)
        assert buffers[0].data_ptr() != buffers[1].data_ptr()
        rest = [int(v) for v in HOST_PLAN_CONSTANTS["sched_rest"]]
        for buffer in buffers:
            assert _u32_words(buffer[:3]) == rest
        current = cb.stream_workspace("sched", device)
        assert current is None or current.data_ptr() not in {
            buffer.data_ptr() for buffer in buffers
        }
        for stream, launch, op in zip(
            streams, (first, second), ("fwd", "dgrad"), strict=True
        ):
            probed = _probes_queue_split(device, op)
            assert launch.plan["queue_split"] == ("probed" if probed else "even")
            assert (cb.stream_queue_split(device, stream) is not None) == probed

    judge_draws(record_property, draw, seed=0, then=after)


@pytest.mark.parametrize("op", ["fwd", "dgrad", "wgrad"])
def test_cuda_graph_replay(op, monkeypatch, record_property):
    """A captured launch takes workspaces of its own, set to their rest state in the
    graph; every replay computes every tile.  A captured forward or data-gradient
    launch passes the even queue split and starts no probe; its replays give the bits
    of the eager launches, which pass the probed split where the launch probes it.
    The replays are judged over the draws (each draw writes its inputs into the
    captured operands)."""
    _fresh_stream_workspaces(monkeypatch)
    if op in ("fwd", "dgrad"):
        device = _require_program(op)
        launch, case = _prepare_row_op(op, RAGGED_E6, 1024, 512, device)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        k = int(HOST_PLAN_CONSTANTS["wave_sync_min_k"])
        case = _Case(RAGGED_E4, 256, k, device, seed=8)
        launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
            case.g, case.x, case.offs
        )
    side = torch.cuda.Stream(device)
    graph = torch.cuda.CUDAGraph()
    probes = None

    def draw(d):
        nonlocal probes
        case.refill(d.seed)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            launch.launch()
            side.synchronize()
            launch.launch()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize()
        eager = launch.out.clone()
        if probes is None:  # the first draw captures the graph
            probes = len(cb._queue_splits)
            with torch.cuda.graph(graph):
                launch.launch()
            assert len(cb._queue_splits) == probes
        for _ in range(3):
            launch.out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            d.op(OUT_NAMES[op], launch.out, op, *case.operands(op), case.sizes)
            if op != "wgrad":
                assert torch.equal(launch.out, eager)

    def after():
        if _probes_queue_split(device, op):
            assert launch.plan["queue_split"] == "probed" and probes == 1
            assert cb.stream_queue_split(device, side) is not None

    judge_draws(record_property, draw, seed=case.seed, then=after)


# The queue-walk case of the forced-rule stream and graph tests: the ``main_halves``
# stand-in of ``QUEUE_CASES`` on its seed there, so every output these tests judge has
# the bits of an output ``test_wgrad_queue_walk_gives_the_selected_stage_bits`` judges.
QUEUE_STREAM_CASE = "e8_ragged"
INT_VIEWS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


def _forced_queue_walk(case, out_dtype, monkeypatch):
    """Force a ``queue_walk`` rule for the device, the stage the call selects with no
    rule and the output (any group count), as
    ``test_wgrad_queue_walk_gives_the_selected_stage_bits`` does, and return the launch of
    the selected stage (prepared with no rule); skips where the route serves no
    queue-walk stage on this architecture."""
    device = case.device
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", [])
    selected = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )
    if not cake_jit.stage_served(selected.record_name, cb.QUEUE_STAGE, arch):
        pytest.skip(f"no queue-walk weight-gradient build on {arch}")
    assert selected.plan["tile_k"] == 512, selected.plan
    assert selected.stages[0] in cb.QUEUE_SELECTED_STAGES, selected.stages
    rule = {
        "arch": arch,
        "sm_counts": [cb.sm_count(device)],
        "stage": selected.stages[0],
        "outputs": [DTYPE_NAMES[out_dtype]],
        "groups": [1, 1 << 30],
    }
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", [rule])
    return selected


def _queue_launch(case, out_dtype, selected):
    """A weight gradient of ``case`` prepared under the forced rule: it binds the queue
    walk (``cb.QUEUE_STAGE``) with the plan and later stages of ``selected`` and takes
    the queue counters (``cb.QUEUE_SCHED``) as its only stream workspace."""
    launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
        case.g, case.x, case.offs, out_dtype=out_dtype
    )
    assert launch.stages == (cb.QUEUE_STAGE, *selected.stages[1:]), launch.stages
    assert launch.plan == selected.plan
    assert _workspace_kinds(launch) == {cb.QUEUE_SCHED}, _workspace_kinds(launch)
    return launch


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_wgrad_queue_walk_on_two_concurrent_streams(out_dtype, monkeypatch, record_property):
    """Under a forced ``queue_walk`` rule, two prepared weight gradients bound to the
    queue walk, launched interleaved on two streams in flight together, give the bits of
    an isolated launch on the quiet current stream (judged against the naive BF16 tier
    over the draws).  Each stream keeps queue counters (``cb.QUEUE_SCHED``) of its own,
    which neither the other stream nor the current one holds, and every stream's
    counters are back at zero after the launches."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    _label, sizes, n, k = QUEUE_CASE_BY_LABEL[QUEUE_STREAM_CASE]
    case = _Case(sizes, n, k, device, seed=13)
    selected = _forced_queue_walk(case, out_dtype, monkeypatch)
    launches = [_queue_launch(case, out_dtype, selected) for _ in range(2)]
    bits = INT_VIEWS[out_dtype]

    def draw(d):
        # the isolated launch: alone on the current stream
        case.refill(d.seed)
        launches[0].out.fill_(float("nan"))
        out = launches[0].launch()
        torch.cuda.synchronize()
        d.op("dW", out, "wgrad", case.g, case.x, sizes)

    def after():
        # the case's own inputs (draw 0, the last)
        want = launches[0].out.clone().view(bits)
        main = torch.cuda.current_stream(device)
        streams = [torch.cuda.Stream(device) for _ in range(2)]
        for stream in streams:
            stream.wait_stream(main)
        # each launch's words compared on its own stream before its next launch
        differ = [[], []]
        for _ in range(8):
            for index, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    launches[index].out.fill_(float("nan"))
                    got = launches[index].launch().view(bits)
                    differ[index].append(torch.ne(got, want).sum())
        for stream in streams:
            main.wait_stream(stream)
        torch.cuda.synchronize()
        for index in range(2):
            for call, count in enumerate(differ[index]):
                assert int(count) == 0, (
                    f"stream {index} launch {call}: {int(count)} elements differ"
                )
        words = [cb.stream_workspace(cb.QUEUE_SCHED, device, s) for s in streams]
        assert all(w is not None for w in words), words
        assert words[0].data_ptr() != words[1].data_ptr()
        current = cb.stream_workspace(cb.QUEUE_SCHED, device)
        assert current is not None
        assert current.data_ptr() not in {w.data_ptr() for w in words}
        for stream in (None, *streams):
            _assert_queue_at_rest(device, stream)

    judge_draws(record_property, draw, seed=case.seed, then=after)


def test_wgrad_queue_walk_cuda_graph_replay(monkeypatch, record_property):
    """Under a forced ``queue_walk`` rule, a captured weight gradient binds the queue
    walk and takes queue counters of its own (``cb.QUEUE_SCHED``, set to zero in the
    graph), never those kept for the stream of its eager launches; every replay gives
    the bits of the eager launch (judged against the naive BF16 tier over the draws), and
    after the replays both the graph's counters and the stream's are back at zero."""
    device = _require_program("wgrad", torch.bfloat16)
    _fresh_stream_workspaces(monkeypatch)
    _label, sizes, n, k = QUEUE_CASE_BY_LABEL[QUEUE_STREAM_CASE]
    case = _Case(sizes, n, k, device, seed=13)
    selected = _forced_queue_walk(case, torch.bfloat16, monkeypatch)
    launch = _queue_launch(case, torch.bfloat16, selected)
    side = torch.cuda.Stream(device)
    graph = torch.cuda.CUDAGraph()
    captured = []

    def draw(d):
        case.refill(d.seed)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            launch.launch()
            side.synchronize()
            launch.launch()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize()
        eager = launch.out.view(torch.int16).clone()
        if not captured:  # the first draw captures the graph
            kept = cb.stream_workspace(cb.QUEUE_SCHED, device, side)
            assert kept is not None
            launch_workspace = cb._launch_workspace

            def recording(kind, device_, device_index, keep):
                buffer = launch_workspace(kind, device_, device_index, keep)
                captured.append((kind, keep, buffer))
                return buffer

            monkeypatch.setattr(cb, "_launch_workspace", recording)
            with torch.cuda.graph(graph):
                launch.launch()
            monkeypatch.setattr(cb, "_launch_workspace", launch_workspace)
            assert [(kind, keep) for kind, keep, _ in captured] == [
                (cb.QUEUE_SCHED, False)
            ], captured
            assert captured[0][2].data_ptr() != kept.data_ptr()
        for _ in range(3):
            launch.out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            d.op("dW", launch.out, "wgrad", case.g, case.x, sizes)
            assert torch.equal(launch.out.view(torch.int16), eager)

    def after():
        for _kind, _keep, buffer in captured:
            assert _u32_words(buffer) == [0] * buffer.numel(), _u32_words(buffer)
        assert cb.stream_workspace(cb.QUEUE_SCHED, device, side) is not None
        _assert_queue_at_rest(device, side)

    judge_draws(record_property, draw, seed=case.seed, then=after)


@pytest.mark.parametrize("op", ["dgrad", "wgrad"])
def test_prepared_launches_allocate_nothing_after_the_first(op, monkeypatch, record_property):
    """After the first launch on a stream (which creates the stream workspaces and
    starts the queue probe where the launch probes one), a prepared launch allocates
    no device memory, also at the launch that takes up the probed split: dgrad, and
    the weight gradient on its wave build where the device has one."""
    _fresh_stream_workspaces(monkeypatch)
    if op == "dgrad":
        device = _require_program("dgrad")
        launch, case = _prepare_row_op("dgrad", RAGGED_E6, 1024, 512, device)
    else:
        device = _require_program("wgrad", torch.bfloat16)
        k = int(HOST_PLAN_CONSTANTS["wave_sync_min_k"])
        case = _Case(RAGGED_E4, 256, k, device, seed=9)
        launch = cake_moe_grouped_gemm_det_train_prepare_wgrad(
            case.g, case.x, case.offs
        )
    launch.launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats(device)
    for _ in range(3):
        launch.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats(device)
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]

    def split_checks():
        if _probes_queue_split(device, op):
            assert launch.plan["queue_split"] == "probed"
            assert cb.stream_queue_split(device) is not None

    _judge_launch(record_property, op, launch, case, then=split_checks)


def test_prepared_launch_follows_new_offsets(record_property):
    """The same prepared launch serves new group boundaries written into offs."""
    device = _require_program("dgrad")
    first, second = [300, 0, 1, 700], [1, 500, 250, 250]
    case = _Case(first, 256, 512, device)
    launch = cake_moe_grouped_gemm_det_train_prepare_dgrad(case.g, case.w, case.offs)

    def draw(d):
        case.refill(d.seed)
        for label, sizes in (("first", first), ("second", second)):
            case.offs.copy_(_offs(sizes, device))
            d.op(f"dX({label})", launch.launch(), "dgrad", case.g, case.w, sizes)

    judge_draws(record_property, draw, seed=case.seed)


def test_padded_outputs_leave_the_padding_untouched(record_property):
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.bfloat16)
    sum_m, n, k = sum(RAGGED_E4), 512, 256

    def draw(d):
        x, w, g, offs = _inputs(RAGGED_E4, n, k, device, seed=d.seed)
        padded = torch.full(
            (sum_m, n + 64), float("nan"), dtype=torch.bfloat16, device=device
        )
        out = cake_moe_grouped_gemm_det_train_forward(x, w, offs, out=padded[:, :n])
        d.op("Y", out, "fwd", x, w, RAGGED_E4)
        assert torch.isnan(padded[:, n:]).all()
        padded = torch.full(
            (len(RAGGED_E4), n, k + 32), float("nan"), dtype=torch.bfloat16, device=device
        )
        out = cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out=padded[:, :, :k])
        d.op("dW", out, "wgrad", g, x, RAGGED_E4)
        assert torch.isnan(padded[:, :, k:]).all()

    judge_draws(record_property, draw, seed=0)


def test_empty_batch():
    device = _require_program("fwd")
    device = _require_program("wgrad", torch.float32)
    x, w, g, offs = _inputs([0, 0], 256, 256, device)
    y = torch.full((0, 256), 0.0, dtype=torch.bfloat16, device=device)
    assert cake_moe_grouped_gemm_det_train_forward(x, w, offs, out=y).shape == (0, 256)
    dw = torch.full((2, 256, 256), float("nan"), dtype=torch.float32, device=device)
    assert torch.equal(
        cake_moe_grouped_gemm_det_train_wgrad(g, x, offs, out=dw), torch.zeros_like(dw)
    )


def test_rejects_misaligned_shapes():
    device = _require_program("fwd")
    x, w, _g, offs = _inputs(RAGGED_E4, 384, 256, device)  # N % 256 != 0
    with pytest.raises(ValueError):
        cake_moe_grouped_gemm_det_train_prepare_forward(x, w, offs)
    with pytest.raises(ValueError):
        cake_moe_grouped_gemm_det_train_prepare_forward(x, w, offs.to(torch.int64))


# ---------------------------------------------------------------------------
# The m_indptr form and the autograd wrapper
# ---------------------------------------------------------------------------


def _m_indptr(sizes, device):
    ends = [sum(sizes[: i + 1]) for i in range(len(sizes))]
    return torch.tensor([0] + ends, dtype=torch.int32, device=device)


def _bits(t):
    """The integer view of ``t``'s elements (a bitwise comparison: ``-0.0`` differs from
    ``+0.0``, and a NaN equals only its own payload)."""
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _assert_lead_binding(lead, base):
    """``lead`` (an ``m_indptr`` call) binds the lead twins of what ``base`` (the ``[E]``
    call over the same shapes) binds: the same plan and stages, every launch on the lead
    twin of its route, every program the twin of the base program."""
    assert lead.record_name == base.record_name
    assert lead.plan == base.plan and lead.stages == base.stages
    assert base.bound_routes and not any(
        cake_jit.is_lead_route(r) for r in base.bound_routes
    )
    assert lead.bound_routes == tuple(cake_jit.lead_route(r) for r in base.bound_routes)
    assert lead.bound_programs == tuple(
        cake_jit.lead_twin(p) for p in base.bound_programs
    )


def test_m_indptr_form_matches_offs_form():
    """An ``m_indptr`` from 0 binds the lead twins (``_assert_lead_binding``) and gives the
    bits of the ``[E]`` form, which binds the programs themselves; a table of neither
    ``E`` nor ``E + 1`` entries is refused."""
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device)
    m_indptr = _m_indptr(RAGGED_E6, device)
    for prepare, a, b, kwargs in (
        (cake_moe_grouped_gemm_det_train_prepare_forward, x, w, {}),
        (cake_moe_grouped_gemm_det_train_prepare_dgrad, g, w, {}),
        (
            cake_moe_grouped_gemm_det_train_prepare_wgrad,
            g,
            x,
            {"num_groups": len(RAGGED_E6)},
        ),
    ):
        lead = prepare(a, b, m_indptr, **kwargs)
        base = prepare(a, b, offs)
        _assert_lead_binding(lead, base)
        lead.out.fill_(float("nan"))
        base.out.fill_(float("nan"))
        assert torch.equal(_bits(lead.launch()), _bits(base.launch())), prepare.__name__
    with pytest.raises(ValueError):  # neither E nor E + 1 entries
        cake_moe_grouped_gemm_det_train_wgrad(
            g, x, m_indptr[:-2], num_groups=len(RAGGED_E6)
        )


# The rows of the queue walk's lead-row tests: group 0 starts at row 37 and five rows
# follow the last group, all NaN (a launch that read one would not give finite bits).
QUEUE_LEAD_ROWS = (37, 5)


def _queue_lead_operands(case):
    """``(g, x, m_indptr, g0, x0)`` over ``case``'s rows: ``g`` / ``x`` with the NaN rows of
    ``QUEUE_LEAD_ROWS`` before group 0 and after the last group, and the ``[E + 1]`` table
    ``m_indptr`` from row 37; ``g0`` / ``x0`` of the same shapes with the case's rows from
    row 0 and zero rows after them, for the ``[E]`` call over the same ``sum_m`` (so the
    same plan)."""
    before, after = QUEUE_LEAD_ROWS
    rows = sum(case.sizes)

    def framed(t, start, fill):
        out = torch.full(
            (before + rows + after, t.shape[1]), fill, dtype=t.dtype, device=t.device
        )
        out[start : start + rows] = t
        return out

    table = torch.cat((case.offs.new_zeros(1), case.offs)) + before
    assert table.dtype == torch.int32
    return (
        framed(case.g, before, float("nan")),
        framed(case.x, before, float("nan")),
        table,
        framed(case.g, 0, 0.0),
        framed(case.x, 0, 0.0),
    )


def _queue_lead_launches(case, g, x, table, g0, x0, output, monkeypatch, **kwargs):
    """Prepare the weight gradient of ``case`` four ways: the ``m_indptr`` call with no
    ``queue_walk`` rule (the selected stage of the lead route), then under a rule forced for
    the device, the stage it selects and ``output`` (any group count): the ``m_indptr`` call,
    the ``[E]`` call over the same ``sum_m`` (``g0`` / ``x0``) and the ``[E]`` call over the
    case's rows alone.  ``kwargs`` (output dtype, ``out`` factory, accumulate) go to every
    call; ``out`` is a factory so each call writes a tensor of its own.  Returns
    ``(plain, lead, base, alone)``; skips where the route serves no queue-walk stage."""
    device = case.device
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    groups = len(case.sizes)
    make_out = kwargs.pop("out", None)
    prepare = cake_moe_grouped_gemm_det_train_prepare_wgrad

    def call(a, b, offs, **extra):
        out = None if make_out is None else make_out()
        return prepare(a, b, offs, out=out, **extra, **kwargs)

    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", [])
    plain = call(g, x, table, num_groups=groups)
    name = plain.record_name
    if not cake_jit.stage_served(name, cb.QUEUE_STAGE, arch):
        pytest.skip(f"no queue-walk weight-gradient build on {arch}")
    assert plain.plan["tile_k"] == 512, plain.plan
    assert plain.stages[0].removesuffix(cb.ACC_SUFFIX) in cb.QUEUE_SELECTED_STAGES
    rule = {
        "arch": arch,
        "sm_counts": [cb.sm_count(device)],
        "stage": plain.stages[0].removesuffix(cb.ACC_SUFFIX),
        "outputs": [output],
        "groups": [1, 1 << 30],
    }
    monkeypatch.setitem(HOST_PLAN_CONSTANTS, "queue_walk", [rule])
    lead = call(g, x, table, num_groups=groups)
    base = call(g0, x0, case.offs)
    alone = call(case.g, case.x, case.offs)
    return plain, lead, base, alone


def _check_queue_lead_launches(plain, lead, base, alone, stage, record_property):
    """``lead`` binds ``stage`` (the queue walk, or its accumulate stage) on the lead route,
    on the lead twin of the program the ``[E]`` call ``base`` binds, with ``plain``'s plan
    and later stages; one launch of each gives finite bits equal to ``base``'s, to
    ``plain``'s (the selected stage) and, where it plans the same, to ``alone``'s, with the
    queue counters back at zero."""
    device = lead.out.device
    arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
    assert lead.stages == (stage, *plain.stages[1:]), (lead.stages, plain.stages)
    assert lead.plan == plain.plan, (lead.plan, plain.plan)
    _assert_lead_binding(lead, base)
    name = base.record_name
    program = cake_jit.stage_binding(name, stage, arch)["program"]
    assert lead.bound_routes[0] == cake_jit.lead_route(name), lead.bound_routes
    assert lead.bound_programs[0] == cake_jit.lead_twin(program), lead.bound_programs
    assert _workspace_kinds(lead) == {cb.QUEUE_SCHED}, _workspace_kinds(lead)
    same_plan = alone.plan == lead.plan and alone.stages == lead.stages
    record_property("bound", list(lead.bound_programs))
    record_property("alone_plans_the_same", same_plan)
    outs = [launch.launch() for launch in (lead, base, plain, alone)]
    torch.cuda.synchronize()
    got = _bits(outs[0])
    assert bool(torch.isfinite(outs[0]).all()), "the lead launch read a NaN row"
    assert torch.equal(got, _bits(outs[1])), "m_indptr vs [E] over the same sum_m"
    assert torch.equal(got, _bits(outs[2])), "queue walk vs the selected stage"
    if same_plan:
        assert torch.equal(got, _bits(outs[3])), "m_indptr vs [E] over its rows alone"
    _assert_queue_at_rest(device)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("label", list(QUEUE_CASE_BY_LABEL))
def test_wgrad_queue_walk_lead_rows_give_the_E_bits(
    label, out_dtype, monkeypatch, record_property
):
    """Under a forced ``queue_walk`` rule, an ``m_indptr`` row table whose group 0 starts at
    row 37, with NaN rows before it and five after the last group (``QUEUE_LEAD_ROWS``),
    binds the queue stage of the lead route (``cb.QUEUE_STAGE`` on the lead twin of the
    ``[E]`` call's program, with ``m_begin``) with the plan of the selected stage.  Its
    output has the bits of the ``[E]`` queue launch over the same ``sum_m``, of the
    selected stage's lead launch (no rule) and, where it plans the same, of the ``[E]``
    queue launch over the case's rows alone; the counters (``cb.QUEUE_SCHED``) are back at
    zero."""
    device = _require_program("wgrad", out_dtype)
    _fresh_stream_workspaces(monkeypatch)
    _label, sizes, n, k = QUEUE_CASE_BY_LABEL[label]
    case = _Case(sizes, n, k, device, seed=13)
    g, x, table, g0, x0 = _queue_lead_operands(case)
    launches = _queue_lead_launches(
        case,
        g,
        x,
        table,
        g0,
        x0,
        DTYPE_NAMES[out_dtype],
        monkeypatch,
        out_dtype=out_dtype,
    )
    for launch in launches:
        launch.out.fill_(float("nan"))
    _check_queue_lead_launches(*launches, cb.QUEUE_STAGE, record_property)


@pytest.mark.parametrize("label", list(QUEUE_CASE_BY_LABEL))
def test_wgrad_queue_walk_acc_lead_rows(label, monkeypatch, record_property):
    """The fused FP32 accumulate of the queue walk (``cb.QUEUE_STAGE`` with
    ``cb.ACC_SUFFIX``) on the rows of ``test_wgrad_queue_walk_lead_rows_give_the_E_bits``:
    every call adds into a copy of one finite ``out``, and the ``m_indptr`` call binds the
    lead twin of the ``[E]`` call's accumulate program and gives the bits of the ``[E]``
    calls and of the selected stage's fused accumulate."""
    device = _require_program("wgrad", torch.float32)
    if not cb.generated_program_available(
        device, "wgrad", torch.float32, accumulate=True
    ):
        pytest.skip("no fused accumulate build of the float32 weight gradient here")
    _fresh_stream_workspaces(monkeypatch)
    _label, sizes, n, k = QUEUE_CASE_BY_LABEL[label]
    case = _Case(sizes, n, k, device, seed=13)
    g, x, table, g0, x0 = _queue_lead_operands(case)
    gen = torch.Generator(device=device).manual_seed(29)
    dst0 = torch.randn(len(sizes), n, k, generator=gen, device=device)
    plain, lead, base, alone = _queue_lead_launches(
        case,
        g,
        x,
        table,
        g0,
        x0,
        "accumulate",
        monkeypatch,
        out=dst0.clone,
        accumulate=True,
    )
    for launch in (plain, lead, base, alone):
        assert launch.plan["accumulate"] == "fused", launch.plan
    _check_queue_lead_launches(
        plain, lead, base, alone, cb.QUEUE_STAGE + cb.ACC_SUFFIX, record_property
    )


def test_autograd_wrapper_matches_reference_and_is_deterministic(record_property):
    device = _require_program("fwd")
    device = _require_program("dgrad")
    device = _require_program("wgrad", torch.bfloat16)
    cake_moe_grouped_gemm_det = pkg.cake_moe_grouped_gemm_det
    kept = {}

    def draw(d):
        x, w, g, offs = _inputs(RAGGED_E6, 512, 256, device, seed=d.seed)
        x = x.clone().requires_grad_(True)
        w = w.clone().requires_grad_(True)
        y = cake_moe_grouped_gemm_det(x, w, offs)
        assert y.shape == (sum(RAGGED_E6), 512) and y.dtype == torch.bfloat16
        d.op("Y", y.detach(), "fwd", x.detach(), w.detach(), RAGGED_E6)
        y.backward(g)
        assert x.grad.dtype == torch.bfloat16 and w.grad.dtype == torch.bfloat16
        d.op("dX", x.grad, "dgrad", g, w.detach(), RAGGED_E6)
        d.op("dW", w.grad, "wgrad", g, x.detach(), RAGGED_E6)
        for e, (s, t) in enumerate(_bounds(RAGGED_E6)):
            if t == s:
                d.exact(f"dW[{e}] (empty group)", w.grad[e], torch.zeros_like(w.grad[e]))
        kept.update(x=x, w=w, g=g, offs=offs)

    def after():
        x, w, g, offs = kept["x"], kept["w"], kept["g"], kept["offs"]
        dx, dw = x.grad.clone(), w.grad.clone()
        x.grad = None
        w.grad = None
        # Same gradients from the m_indptr form; bitwise reproducible.
        cake_moe_grouped_gemm_det(x, w, _m_indptr(RAGGED_E6, device)).backward(g)
        torch.cuda.synchronize()
        assert torch.equal(x.grad, dx) and torch.equal(w.grad, dw)
        with pytest.raises(ValueError):
            cake_moe_grouped_gemm_det(x.float(), w, offs)

    judge_draws(record_property, draw, seed=0, then=after)


# ---------------------------------------------------------------------------
# Autograd wrapper: the forward launch after the saves (checkpoint early stop)
# ---------------------------------------------------------------------------

# ``cake_moe_grouped_gemm_det`` launches the forward after autograd saved its inputs, so
# the early stop of a non-reentrant checkpoint skips the recomputed GEMM when nothing
# after it saves a tensor.  These tests count the forward launches (a pass-through on
# ``cake_autograd._fwd_op``) and, where the torch profiler records kernels, the CUDA
# kernels of checkpointed steps; the control arm is a direct
# ``CakeMoEGroupedGemmDetFunction.apply``, which launches inside ``forward``.  Both reductions are
# deterministic, so every output and gradient is compared bitwise.

# The small shape of the autograd tests: ragged groups with an empty and a one-row
# group.
SMALL_SIZES = RAGGED_E4
SMALL_SHAPE = (512, 256)  # (N, K)
# (sizes, (N, K)) of the early-stop tests: the small shape, and ``K512_E2``, whose
# weight gradient takes the 512-wide k tile on the architectures of ``k512_archs``.
AUTOGRAD_CASES = {"small": (SMALL_SIZES, SMALL_SHAPE), "k512": (K512_E2, (256, 1024))}
# Forward launches of a checkpointed step of each ``_region`` (early stop on): the
# recomputation stops after the GEMM's saves unless a later op saves a tensor.
REGION_FORWARDS = {"silu": 2, "scale": 1, "pair": 2}
# The MoE expert block of ``_moe_block`` and the (gate_up, down) forward launches of
# its checkpointed step: only the down projection's recomputation is skipped, and not
# when the routing multiply after it saves tensors.
MOE_HIDDEN, MOE_INTER = 256, 256
MOE_BLOCK_FORWARDS = {"block": (2, 1), "routed": (2, 2), "selective": (2, 1)}
# Forward launches of a checkpointed step: reentrant (always recomputes); nested
# non-reentrant (the outer stop fires at the inner checkpoint's input saves, the inner
# one before the GEMM); nested with an outer op that saves the inner output.
NESTED_FORWARDS = {"reentrant": 2, "nested": 1, "nested_then_save": 2}


def _require_autograd_programs():
    """The device, when the forward, data-gradient and bf16 weight-gradient programs are
    all registered for it."""
    _require_program("fwd")
    _require_program("dgrad")
    return _require_program("wgrad", torch.bfloat16)


def _wrapper(offs):
    """``gemm(a, b)`` through ``cake_moe_grouped_gemm_det``: the forward launch after ``apply``."""
    return lambda a, b: pkg.cake_moe_grouped_gemm_det(a, b, offs)


def _in_forward(offs):
    """The control arm: ``gemm(a, b)`` through a direct ``CakeMoEGroupedGemmDetFunction.apply``, which
    launches inside ``forward``, before autograd saves the inputs."""
    return lambda a, b: cake_autograd.CakeMoEGroupedGemmDetFunction.apply(a, b, offs)


def _leaves(*tensors):
    """Fresh leaves requiring grad, cloned from ``tensors``."""
    return [t.clone().requires_grad_(True) for t in tensors]


def _count_launches(monkeypatch):
    """Replace ``_fwd_op``, ``_dgrad_op`` and ``_wgrad_op`` of ``cake_autograd`` by
    pass-throughs that record the projection ``(N, K)`` of every call (of ``w``, or of the
    ``[E, N, K]`` weight gradient); returns ``{"fwd": [...], "dgrad": [...], "wgrad": [...]}``."""
    calls = {"fwd": [], "dgrad": [], "wgrad": []}

    def counting(op, launch):
        def call(*args):
            shape = args[3].shape if op == "wgrad" else args[1].shape
            calls[op].append((int(shape[1]), int(shape[2])))
            launch(*args)

        return call

    for op in calls:
        name = f"_{op}_op"
        monkeypatch.setattr(
            cake_autograd, name, counting(op, getattr(cake_autograd, name))
        )
    return calls


def _step(fn, leaves, grads, ckpt=None, early_stop=True, context_fn=None):
    """One training step: ``fn(*leaves)`` plain (``ckpt`` None) or under
    ``torch.utils.checkpoint`` (``"non_reentrant"`` / ``"reentrant"``; ``context_fn`` for
    a selective checkpoint), the early stop on (torch's default) or off around the forward
    and the backward, then the backward of the output(s) with ``grads``.  Returns
    ``(outputs, leaf gradients)``, both tuples, the outputs detached."""
    kwargs = {} if context_fn is None else {"context_fn": context_fn}
    stop = torch.utils.checkpoint.set_checkpoint_early_stop(bool(early_stop))
    with stop:
        if ckpt is None:
            out = fn(*leaves)
        else:
            out = torch.utils.checkpoint.checkpoint(
                fn, *leaves, use_reentrant=ckpt == "reentrant", **kwargs
            )
        outs = out if isinstance(out, tuple) else (out,)
        torch.autograd.backward(outs, grads)
    torch.cuda.synchronize()
    return tuple(o.detach() for o in outs), tuple(t.grad for t in leaves)


def _assert_same_step(got, want):
    """The outputs and gradients of two ``_step`` results are bitwise equal (None where
    None)."""
    pairs = zip((*got[0], *got[1]), (*want[0], *want[1]), strict=True)
    for index, (a, b) in enumerate(pairs):
        assert (a is None) == (b is None), index
        assert a is None or torch.equal(a, b), index


def _region(gemm, region):
    """``fn(x, w)`` of a checkpointed region around ``gemm(x, w)``: ``"silu"`` (saves the
    GEMM's output), ``"scale"`` (``* 2.0``, saves nothing) or ``"pair"`` (also returns
    ``x.sin()``, which saves ``x`` after the GEMM)."""
    if region == "silu":
        return lambda a, b: torch.nn.functional.silu(gemm(a, b))
    if region == "scale":
        return lambda a, b: gemm(a, b) * 2.0
    return lambda a, b: (gemm(a, b), a.sin())


def _region_grads(region, g, x):
    """The output gradients of a ``_region``: ``g``, and for ``"pair"`` one for ``x.sin()``."""
    if region != "pair":
        return (g,)
    gen = torch.Generator(device=x.device).manual_seed(7)
    return (g, torch.randn(x.shape, generator=gen, device=x.device).to(x.dtype))


def _moe_block(gemm, routed):
    """``fn(x, w1, w2[, probs])``: the MoE expert block ``down(silu(gate) * up)`` with
    ``gate, up = gate_up(x).chunk(2)``, both projections ``gemm``; with ``routed`` its
    output is weighted by routing probabilities."""

    def block(x, w1, w2, *probs):
        gate, up = gemm(x, w1).chunk(2, dim=-1)
        out = gemm(torch.nn.functional.silu(gate) * up, w2)
        return out * probs[0] if routed else out

    return block


def _cuda_kernel_names(fn):
    """The multiset of the names of the CUDA kernels one call of ``fn`` launches, the
    backward's (launched from the autograd thread) included (torch profiler; memcpy /
    memset activities excluded)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return collections.Counter(
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
        and not e.name.startswith(("Memcpy", "Memset"))
    )


def _require_profiled_kernels(device):
    if not _cuda_kernel_names(lambda: torch.ones(8, device=device).add_(1)):
        pytest.skip("the torch profiler records no CUDA kernels in this environment")


@contextlib.contextmanager
def _torch_deterministic():
    """``torch.use_deterministic_algorithms(True)`` inside the block, the previous setting
    restored after it."""
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])


@pytest.mark.parametrize("sizes", [RAGGED_E4, [0, 0]], ids=["ragged", "empty"])
def test_wrapper_saves_inputs_before_the_forward_launch(sizes, monkeypatch):
    """Through ``cake_moe_grouped_gemm_det`` autograd saves ``x``, ``w`` and ``offs`` (the
    inputs themselves, in this order) and then the forward launches into ``Y``, saving
    nothing; a direct ``CakeMoEGroupedGemmDetFunction.apply`` launches first and saves after.  The
    node is ``CakeMoEGroupedGemmDetFunctionBackward`` and ``Y`` has the forward's bits either way,
    also with no rows (``sum_m == 0``)."""
    device = _require_program("fwd")
    x0, w0, _g, offs = _inputs(sizes, *SMALL_SHAPE, device, seed=40)
    want = cake_moe_grouped_gemm_det_train_forward(x0, w0, offs)
    events = []
    launch = cake_autograd._fwd_op

    def spy(x, w, offs_, out):
        events.append(("fwd", out.data_ptr()))
        launch(x, w, offs_, out)

    def pack(t):
        events.append(("pack", t))
        return t

    monkeypatch.setattr(cake_autograd, "_fwd_op", spy)
    for gemm, order in (
        (_wrapper(offs), ["pack", "pack", "pack", "fwd"]),
        (_in_forward(offs), ["fwd", "pack", "pack", "pack"]),
    ):
        x, w = _leaves(x0, w0)
        events.clear()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = gemm(x, w)
        torch.cuda.synchronize()
        assert [kind for kind, _ in events] == order
        packed = [t for kind, t in events if kind == "pack"]
        for t, src in zip(packed, (x, w, offs), strict=True):
            assert t.data_ptr() == src.data_ptr(), (t.shape, src.shape)
            assert t.shape == src.shape and t.dtype == src.dtype
        assert [p for kind, p in events if kind == "fwd"] == [y.data_ptr()]
        assert type(y.grad_fn).__name__ == "CakeMoEGroupedGemmDetFunctionBackward"
        assert torch.equal(y.detach(), want)


@pytest.mark.parametrize("case", list(AUTOGRAD_CASES))
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_checkpoint_early_stop_skips_the_forward_launch(
    early_stop, case, monkeypatch, record_property
):
    """Under ``torch.utils.checkpoint`` (non-reentrant) the wrapper launches the forward
    once with the early stop on (torch's default: the recomputation stops at the third
    saved input, before the launch) and twice with it off; the control arm (direct
    ``apply``) twice either way.  ``Y``, ``dX`` and ``dW`` are bitwise the plain step's
    in both arms.  The plain step's outputs are judged over the draws."""
    device = _require_autograd_programs()
    sizes, (n, k) = AUTOGRAD_CASES[case]
    calls = _count_launches(monkeypatch)
    kept = {}

    def draw(d):
        x0, w0, g, offs = _inputs(sizes, n, k, device, seed=d.seed)
        calls["fwd"].clear()
        plain = _step(_wrapper(offs), _leaves(x0, w0), (g,))
        assert calls["fwd"] == [(n, k)]
        (y,), (dx, dw) = plain
        d.op("Y", y, "fwd", x0, w0, sizes)
        d.op("dX", dx, "dgrad", g, w0, sizes)
        d.op("dW", dw, "wgrad", g, x0, sizes)
        kept.update(x0=x0, w0=w0, g=g, offs=offs, plain=plain)

    def after():
        x0, w0, g, offs, plain = (kept[key] for key in ("x0", "w0", "g", "offs", "plain"))
        for arm, gemm, forwards in (
            ("wrapper", _wrapper(offs), 1 if early_stop else 2),
            ("control", _in_forward(offs), 2),
        ):
            calls["fwd"].clear()
            got = _step(
                gemm, _leaves(x0, w0), (g,), ckpt="non_reentrant", early_stop=early_stop
            )
            assert calls["fwd"] == [(n, k)] * forwards, arm
            _assert_same_step(got, plain)

    judge_draws(record_property, draw, seed=13, then=after)


@pytest.mark.parametrize("case", list(AUTOGRAD_CASES))
@pytest.mark.parametrize(
    "early_stop", [True, False], ids=["early_stop", "no_early_stop"]
)
def test_checkpoint_early_stop_skips_the_forward_kernels(early_stop, case):
    """The device side of ``test_checkpoint_early_stop_skips_the_forward_launch``: after a
    warm-up (programs loaded, stream workspaces and queue probe done) the CUDA kernels of
    a checkpointed step through the wrapper are those of the plain step with the early
    stop on, and those plus the kernels of one bare forward launch with it off."""
    device = _require_autograd_programs()
    _require_profiled_kernels(device)
    sizes, (n, k) = AUTOGRAD_CASES[case]
    x0, w0, g, offs = _inputs(sizes, n, k, device, seed=13)
    wrapper = _wrapper(offs)
    out = torch.empty(x0.shape[0], n, dtype=torch.bfloat16, device=device)
    bare = lambda: cake_autograd._fwd_op(x0, w0, offs, out)
    plain = lambda: _step(wrapper, _leaves(x0, w0), (g,))
    checkpointed = lambda: _step(
        wrapper, _leaves(x0, w0), (g,), ckpt="non_reentrant", early_stop=early_stop
    )
    for warm in (bare, plain, checkpointed):
        warm()
    fwd = _cuda_kernel_names(bare)
    assert fwd, "the forward launch records no kernel"
    want = _cuda_kernel_names(plain)
    if not early_stop:
        want += fwd
    assert _cuda_kernel_names(checkpointed) == want


@pytest.mark.parametrize("region", list(REGION_FORWARDS))
def test_checkpoint_recomputes_the_forward_when_a_later_op_saves(region, monkeypatch):
    """The early stop skips the recomputed GEMM only when no op after it in the region
    saves a tensor, as for a built-in op: ``silu`` of its output saves that output (2
    forward launches), ``* 2.0`` saves nothing (1), ``x.sin()`` after it saves ``x`` (2).
    The outputs and gradients are bitwise the plain step's."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=42)
    grads = _region_grads(region, g, x0)
    fn = _region(_wrapper(offs), region)
    calls = _count_launches(monkeypatch)
    plain = _step(fn, _leaves(x0, w0), grads)
    assert calls["fwd"] == [SMALL_SHAPE]
    calls["fwd"].clear()
    checkpointed = _step(fn, _leaves(x0, w0), grads, ckpt="non_reentrant")
    assert calls["fwd"] == [SMALL_SHAPE] * REGION_FORWARDS[region]
    _assert_same_step(checkpointed, plain)


@pytest.mark.parametrize("region", list(REGION_FORWARDS))
@pytest.mark.parametrize("impl", ["cake_det", "torch_grouped_mm"])
def test_checkpoint_recompute_kernels_match_torch_grouped_mm(impl, region):
    """The regions of ``test_checkpoint_recomputes_the_forward_when_a_later_op_saves``
    through this package and through ``torch._grouped_mm`` (whose autograd saves its
    inputs before its kernel): after a warm-up the CUDA kernels of a checkpointed step are
    those of the plain step plus ``REGION_FORWARDS[region] - 1`` times those of one bare
    forward GEMM, the same pattern for both."""
    device = _require_autograd_programs()
    _require_profiled_kernels(device)
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=43)
    grads = _region_grads(region, g, x0)
    if impl == "cake_det":
        gemm = _wrapper(offs)
        out = torch.empty(
            x0.shape[0], SMALL_SHAPE[0], dtype=torch.bfloat16, device=device
        )
        bare = lambda: cake_autograd._fwd_op(x0, w0, offs, out)
    else:
        if not hasattr(torch, "_grouped_mm"):
            pytest.skip("this torch has no torch._grouped_mm")
        gemm = lambda a, b: torch._grouped_mm(a, b.transpose(1, 2), offs=offs)
        bare = lambda: gemm(x0, w0)
    fn = _region(gemm, region)
    plain = lambda: _step(fn, _leaves(x0, w0), grads)
    checkpointed = lambda: _step(fn, _leaves(x0, w0), grads, ckpt="non_reentrant")
    try:
        for warm in (bare, plain, checkpointed):
            warm()
    except RuntimeError as exc:
        if impl == "cake_det":
            raise
        arch = SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]
        pytest.skip(f"torch._grouped_mm does not run this step on {arch}: {exc}")
    fwd = _cuda_kernel_names(bare)
    assert fwd, f"a bare {impl} forward records no kernel"
    want = _cuda_kernel_names(plain)
    for _ in range(REGION_FORWARDS[region] - 1):
        want += fwd
    assert _cuda_kernel_names(checkpointed) == want


@pytest.mark.parametrize("variant", list(MOE_BLOCK_FORWARDS))
def test_checkpoint_moe_block_skips_only_the_last_gemm(variant, monkeypatch):
    """A checkpointed MoE expert block (``_moe_block``, both projections through the
    wrapper): with the early stop on the gate / up projection is recomputed (SiLU saves
    its output) and the down projection is not; a routing multiply whose probabilities
    require grad saves the down projection's output, which is then recomputed too (the
    scope of the skip, as for built-in ops); a selective checkpoint that saves the
    ``aten.mul.Tensor`` outputs launches as the plain one.  The output and the gradients
    of ``x``, ``w1``, ``w2`` (and of the probabilities) are bitwise the plain step's."""
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    device = _require_autograd_programs()
    gate_up, down = (2 * MOE_INTER, MOE_HIDDEN), (MOE_HIDDEN, MOE_INTER)
    x0, w1, _g, offs = _inputs(SMALL_SIZES, *gate_up, device, seed=44)
    _x, w2, g, _offs = _inputs(SMALL_SIZES, *down, device, seed=45)
    inputs = [x0, w1, w2]
    if variant == "routed":
        gen = torch.Generator(device=device).manual_seed(46)
        probs = torch.rand(x0.shape[0], 1, generator=gen, device=device)
        inputs.append(probs.to(torch.bfloat16))

    def policy(ctx, op, *args, **kwargs):
        if op == torch.ops.aten.mul.Tensor:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    context_fn = None
    if variant == "selective":
        context_fn = functools.partial(create_selective_checkpoint_contexts, policy)
    block = _moe_block(_wrapper(offs), variant == "routed")
    calls = _count_launches(monkeypatch)
    plain = _step(block, _leaves(*inputs), (g,))
    assert collections.Counter(calls["fwd"]) == {gate_up: 1, down: 1}
    calls["fwd"].clear()
    checkpointed = _step(
        block, _leaves(*inputs), (g,), ckpt="non_reentrant", context_fn=context_fn
    )
    want = dict(zip((gate_up, down), MOE_BLOCK_FORWARDS[variant], strict=True))
    assert collections.Counter(calls["fwd"]) == want
    _assert_same_step(checkpointed, plain)


@pytest.mark.parametrize("structure", list(NESTED_FORWARDS))
def test_reentrant_and_nested_checkpoints(structure, monkeypatch):
    """A reentrant checkpoint recomputes the forward (2 launches); a non-reentrant one
    nested in another launches it once (the outer recomputation stops at the inner
    checkpoint's input saves, the inner one before the launch), twice when an outer op
    saves the inner output (``.sin()``).  The outputs and gradients are bitwise the plain
    step's."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=46)
    wrapper = _wrapper(offs)

    def inner(a, b):
        return torch.utils.checkpoint.checkpoint(wrapper, a, b, use_reentrant=False)

    plain_fn, fn, ckpt = wrapper, inner, "non_reentrant"
    if structure == "reentrant":
        fn, ckpt = wrapper, "reentrant"
    elif structure == "nested_then_save":
        plain_fn = lambda a, b: wrapper(a, b).sin()
        fn = lambda a, b: inner(a, b).sin()
    calls = _count_launches(monkeypatch)
    plain = _step(plain_fn, _leaves(x0, w0), (g,))
    calls["fwd"].clear()
    checkpointed = _step(fn, _leaves(x0, w0), (g,), ckpt=ckpt)
    assert calls["fwd"] == [SMALL_SHAPE] * NESTED_FORWARDS[structure]
    _assert_same_step(checkpointed, plain)


def test_checkpointed_gradient_accumulation_and_retain_graph(monkeypatch):
    """Two backward passes, each step under a non-reentrant checkpoint: one forward
    launch per step, ``x.grad`` the bf16 sum of the two data gradients and ``w.grad``
    that of the two weight gradients of the package functions (bitwise), the empty
    group's exact zeros.  One checkpointed graph backpropagated twice
    (``retain_graph=True`` first): one forward launch in all through the wrapper (both
    recomputations stop before it), three through the control arm, and ``x.grad`` twice
    the data gradient (bitwise); a third backward raises."""
    device = _require_autograd_programs()
    _fresh_stream_workspaces(monkeypatch)
    sizes, (n, k) = AUTOGRAD_CASES["small"]
    x0, w0, g1, offs = _inputs(sizes, n, k, device, seed=14)
    g2 = _inputs(sizes, n, k, device, seed=15)[2]
    calls = _count_launches(monkeypatch)
    wrapper = _wrapper(offs)
    x, w = _leaves(x0, w0)
    for g in (g1, g2):
        y = torch.utils.checkpoint.checkpoint(wrapper, x, w, use_reentrant=False)
        y.backward(g)
    torch.cuda.synchronize()
    assert calls["fwd"] == [(n, k)] * 2
    dx1 = cake_moe_grouped_gemm_det_train_dgrad(g1, w0, offs)
    dx2 = cake_moe_grouped_gemm_det_train_dgrad(g2, w0, offs)
    dw1 = cake_moe_grouped_gemm_det_train_wgrad(g1, x0, offs)
    dw2 = cake_moe_grouped_gemm_det_train_wgrad(g2, x0, offs)
    assert torch.equal(x.grad, dx1 + dx2)
    assert torch.equal(w.grad, dw1 + dw2)
    assert not w.grad[sizes.index(0)].any()
    for gemm, forwards in ((wrapper, 1), (_in_forward(offs), 3)):
        calls["fwd"].clear()
        x, w = _leaves(x0, w0)
        y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
        y.backward(g1, retain_graph=True)
        y.backward(g1)
        torch.cuda.synchronize()
        assert calls["fwd"] == [(n, k)] * forwards
        assert torch.equal(x.grad, dx1 + dx1)
        with pytest.raises(RuntimeError, match="a second time"):
            y.backward(g1)


@pytest.mark.parametrize("frozen", ["w", "x"])
def test_checkpoint_with_a_frozen_operand(frozen, monkeypatch):
    """A checkpointed step with a frozen operand, under torch's deterministic mode (no
    alert: both reductions are deterministic), launches the forward once and only the
    gradient it needs: a frozen ``w`` the data gradient alone, a frozen ``x`` the weight
    gradient alone.  The gradient has the bits of the package function."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=47)
    x = x0.clone().requires_grad_(frozen != "x")
    w = w0.clone().requires_grad_(frozen != "w")
    calls = _count_launches(monkeypatch)
    with _torch_deterministic():
        _step(_wrapper(offs), [x, w], (g,), ckpt="non_reentrant")
    assert calls["fwd"] == [SMALL_SHAPE]
    if frozen == "w":
        assert calls["dgrad"] == [SMALL_SHAPE] and calls["wgrad"] == []
        assert w.grad is None
        assert torch.equal(x.grad, cake_moe_grouped_gemm_det_train_dgrad(g, w0, offs))
    else:
        assert calls["wgrad"] == [SMALL_SHAPE] and calls["dgrad"] == []
        assert x.grad is None
        assert torch.equal(w.grad, cake_moe_grouped_gemm_det_train_wgrad(g, x0, offs))


@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
def test_wrapper_forward_cuda_graph_replay(grad, record_property):
    """A CUDA graph captured around ``cake_moe_grouped_gemm_det`` (with leaves that require
    grad and grad mode on, or under ``no_grad``) replays the forward launch the wrapper
    makes after ``apply``: after new group boundaries (same ``sum_m``) are written into
    ``offs``, every replay gives the bits of the eager wrapper on those boundaries."""
    device = _require_program("fwd")
    first, second = list(SMALL_SIZES), [1, 500, 250, 250]
    x0, w0, _g, offs = _inputs(first, *SMALL_SHAPE, device, seed=48)
    x = x0.clone().requires_grad_(grad)
    w = w0.clone().requires_grad_(grad)
    wrapper = _wrapper(offs)
    grad_mode = contextlib.nullcontext if grad else torch.no_grad
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with grad_mode(), torch.cuda.stream(side):
        wrapper(x, w)  # program load before the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with grad_mode(), torch.cuda.graph(graph):
        y = wrapper(x, w)
    assert (y.grad_fn is not None) == grad

    def draw(d):
        # the draw's inputs, written into the captured leaves (and their sources)
        x1, w1, _g1, _ = _inputs(first, *SMALL_SHAPE, device, seed=d.seed)
        with torch.no_grad():
            for dst, src in ((x0, x1), (w0, w1), (x, x1), (w, w1)):
                dst.copy_(src)
        for label, sizes in (("first", first), ("second", second), ("first", first)):
            offs.copy_(_offs(sizes, device))
            y.detach().fill_(float("nan"))
            graph.replay()
            eager = pkg.cake_moe_grouped_gemm_det(x0, w0, offs)
            torch.cuda.synchronize()
            assert torch.equal(y.detach(), eager), sizes
            d.op(f"Y({label})", y.detach(), "fwd", x0, w0, sizes)

    judge_draws(record_property, draw, seed=48)


@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint"])
def test_wrapper_retains_the_memory_of_the_in_forward_path(checkpointed):
    """Against the control arm (direct ``apply``), plain and under a non-reentrant
    checkpoint, after a warm-up: the same allocated memory after the forward, a backward
    peak no higher, and after the backward and ``del y`` the two gradients alone (dropping
    them returns to the memory before the forward)."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=49)
    ckpt = "non_reentrant" if checkpointed else None

    def measure(gemm):
        x, w = _leaves(x0, w0)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(device)
        if checkpointed:
            y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
        else:
            y = gemm(x, w)
        torch.cuda.synchronize()
        forward = torch.cuda.memory_allocated(device) - base
        torch.cuda.reset_peak_memory_stats(device)
        y.backward(g)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) - base
        del y
        kept = torch.cuda.memory_allocated(device) - base
        grads = x.grad.untyped_storage().nbytes() + w.grad.untyped_storage().nbytes()
        x.grad = w.grad = None
        assert torch.cuda.memory_allocated(device) == base
        return forward, peak, kept, grads

    arms = (_wrapper(offs), _in_forward(offs))
    for gemm in arms:  # stream workspaces and the queue probe
        _step(gemm, _leaves(x0, w0), (g,), ckpt=ckpt)
    new, old = (measure(gemm) for gemm in arms)
    assert new[0] == old[0], (new, old)
    assert new[1] <= old[1], (new, old)
    assert new[2] == new[3] and old[2] == old[3], (new, old)


def test_wrapper_does_not_synchronize():
    """After a warm-up, a forward and a checkpointed step (the forward, then the backward
    with its recomputation) through the wrapper and through the control arm run under
    ``torch.cuda.set_sync_debug_mode("error")``, so without a host synchronisation, and
    give the same gradients."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=50)
    arms = (_wrapper(offs), _in_forward(offs))
    for gemm in arms:
        _step(gemm, _leaves(x0, w0), (g,), ckpt="non_reentrant")
    leaves = [_leaves(x0, w0) for _ in arms]
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for gemm, (x, w) in zip(arms, leaves, strict=True):
            gemm(x, w)
            y = torch.utils.checkpoint.checkpoint(gemm, x, w, use_reentrant=False)
            y.backward(g)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    (x_new, w_new), (x_old, w_old) = leaves
    assert torch.equal(x_new.grad, x_old.grad)
    assert torch.equal(w_new.grad, w_old.grad)


def test_wrapper_under_jit_trace(monkeypatch):
    """Under ``torch.jit.trace`` the wrapper keeps the launch inside ``forward`` (a trace
    records ``apply`` only): the traced function replayed on a new ``x`` launches the
    forward once and gives the eager wrapper's bits."""
    device = _require_program("fwd")
    x0, w0, _g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=51)
    x1 = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=52)[0]
    traced = torch.jit.trace(
        lambda a, b, o: pkg.cake_moe_grouped_gemm_det(a, b, o),
        (x0, w0, offs),
        check_trace=False,
    )
    calls = _count_launches(monkeypatch)
    out = traced(x1, w0, offs)
    torch.cuda.synchronize()
    assert calls["fwd"] == [SMALL_SHAPE]
    eager = pkg.cake_moe_grouped_gemm_det(x1, w0, offs)
    assert torch.equal(out, eager)
    assert not torch.equal(eager, cake_moe_grouped_gemm_det_train_forward(x0, w0, offs))


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_wrapper_under_torch_compile(backend):
    """``torch.compile`` (``backend``) of a non-reentrant checkpointed region through the
    wrapper gives the bits of the eager checkpointed step (``Y``, ``dX``, ``dW``); under
    Dynamo the wrapper keeps the launch inside ``forward`` (Dynamo cannot lift the
    deferred launch out of the checkpointed region).  Where the compiled step raises, the
    control arm (direct ``apply``) must raise the same exception type, and the case is
    skipped with that failure as the reason (a limit the wrapper did not introduce); a
    failure of the wrapper alone fails."""
    import torch._dynamo

    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=53)
    eager = _step(_wrapper(offs), _leaves(x0, w0), (g,), ckpt="non_reentrant")

    def compiled_step(gemm):
        def region(a, b):
            return torch.utils.checkpoint.checkpoint(gemm, a, b, use_reentrant=False)

        torch._dynamo.reset()
        try:
            step = torch.compile(region, backend=backend)
            return _step(step, _leaves(x0, w0), (g,)), None
        except Exception as exc:  # noqa: BLE001 - compared with the control arm
            return None, exc
        finally:
            torch._dynamo.reset()

    got, failure = compiled_step(_wrapper(offs))
    if failure is None:
        _assert_same_step(got, eager)
        return
    _, control_failure = compiled_step(_in_forward(offs))
    assert type(control_failure) is type(failure), (
        f"torch.compile ({backend}) fails through the wrapper only: {failure!r}; "
        f"control arm: {control_failure!r}"
    )
    pytest.skip(
        f"torch.compile ({backend}) fails the same way through CakeMoEGroupedGemmDetFunction.apply: "
        f"{type(failure).__name__}: {str(failure)[:400]}"
    )


def test_direct_apply_keeps_the_in_forward_launch(monkeypatch):
    """A direct ``CakeMoEGroupedGemmDetFunction.apply(x, w, offs)`` launches the forward inside
    ``forward`` and gives the wrapper's bits (``Y``, ``dX``, ``dW``); a stale fourth
    argument (a ``True`` / ``False`` numerical mode) raises ``TypeError`` before any
    launch; shape and offset errors, which the wrapper now raises after autograd saved
    the inputs, are still ``ValueError``."""
    device = _require_autograd_programs()
    x0, w0, g, offs = _inputs(SMALL_SIZES, *SMALL_SHAPE, device, seed=54)
    calls = _count_launches(monkeypatch)
    wrapped = _step(_wrapper(offs), _leaves(x0, w0), (g,))
    direct = _step(_in_forward(offs), _leaves(x0, w0), (g,))
    _assert_same_step(direct, wrapped)
    assert calls["fwd"] == [SMALL_SHAPE] * 2
    for stale in (True, False):
        x, w = _leaves(x0, w0)
        with pytest.raises(TypeError, match="takes no fourth argument"):
            cake_autograd.CakeMoEGroupedGemmDetFunction.apply(x, w, offs, stale)
    assert calls["fwd"] == [SMALL_SHAPE] * 2
    x, _w = _leaves(x0, w0)
    n, k = SMALL_SHAPE
    for bad_w, bad_offs in (
        (w0[:, : n - 128], offs),  # N % 256 != 0
        (w0[:, :, : k // 2], offs),  # K of w differs from x's
        (w0, offs.to(torch.int64)),
    ):
        with pytest.raises(ValueError):
            pkg.cake_moe_grouped_gemm_det(
                x, bad_w.clone().requires_grad_(True), bad_offs
            )
