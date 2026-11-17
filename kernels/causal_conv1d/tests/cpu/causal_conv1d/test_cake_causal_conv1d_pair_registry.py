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

CPU tests of the registry rows and the host rules of the opt-in FP32 (hi, lo) pair ``dweight`` / ``dbias`` sums
(``dweight_sum="fp32_pair"``).  No kernel runs here.

- Both packages: the pair routes ``bwd_fp32g_pair`` / ``bwd_bias_fp32g_pair`` ship on every architecture as the
  twin route's fallback and channel-selected program; each pair program has its twin's launch shape (block, grid
  rule, selection rule, architectures) and fits its recorded co-residency in shared memory; the host plans a pair
  record by its twin route's rule; only ``dweight_sum="fp32_pair"`` with the FP32 SiLU' policy selects a pair route.
- The deterministic package: each pair route plans its twin route's channel tiles (``channel_split``) and its
  programs share one pair reduce program (``reduce`` role).
- The non-deterministic package: the pair programs keep an even number of FP32 accumulator words per sum and copy
  (``(hi, lo)``), no more than the twin's.
- The non-deterministic KCP registry: ``kcp_bwd_dyn_pair`` ships on every architecture with the launch shape and
  workspace rule of ``kcp_bwd_dyn``; ``"fp32_pair"`` is refused with a reserve or beyond the overlapped backward's
  channel limit.
"""

from __future__ import annotations

import pytest

from cake_pplx._backends import cake_causal_conv1d_det_train as det
from cake_pplx._backends import cake_causal_conv1d_nondet_train as nondet
from cake_pplx._backends.cake_causal_conv1d_det_train import cake_jit as det_jit
from cake_pplx._backends.cake_causal_conv1d_nondet_train import cake_jit as nondet_jit
from cake_pplx._backends.cake_causal_conv1d_nondet_train import kcp as nondet_kcp
from cake_pplx._backends.cake_causal_conv1d_nondet_train import kcp_jit

PAIR_KINDS = {"bwd_fp32g_pair": "bwd_fp32g", "bwd_bias_fp32g_pair": "bwd_bias_fp32g"}
PACKAGES = {"det": (det, det_jit), "nondet": (nondet, nondet_jit)}
# Shared memory per SM of SM100 / SM103 / SM107 (228 KiB) and the 1 KiB the runtime reserves per CTA.
SMEM_PER_SM = 228 * 1024
SMEM_RESERVED_PER_CTA = 1024


# A token count past every token floor.
LARGE_S = 1 << 20


def _channels_of(loader, arch: str, name: str) -> int:
    """The smallest channel count that selects ``name`` on ``arch``: its rule's threshold, 128 for a fallback."""
    rule = loader.select_rule(loader.MODULES[name])
    if rule is None:
        return 128
    value = rule["channels"][arch]
    return int(value[0] if isinstance(value, list) else value)


def _twin_of(loader, arch: str, kind: str, name: str) -> str:
    """The program the twin route selects at the smallest channel count that selects pair program ``name`` (which
    the pair route selects there)."""
    C = _channels_of(loader, arch, name)
    assert loader.selected_modules(arch, kind, C=C, S=LARGE_S)[0] == name, (arch, kind, name, C)
    return loader.selected_modules(arch, PAIR_KINDS[kind], C=C, S=LARGE_S)[0]


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_pair_routes_mirror_their_twins(package: str) -> None:
    backend, loader = PACKAGES[package]
    assert dict(loader.PAIR_ROUTES) == PAIR_KINDS
    assert tuple(loader.DWEIGHT_SUMS) == ("fp32", "fp32_pair") == tuple(backend.DWEIGHT_SUMS)
    assert set(PAIR_KINDS) <= set(loader.KINDS)
    for arch in loader.ARCHES:
        for kind in PAIR_KINDS:
            names = loader.route_modules(arch, kind)
            assert len(names) == 2, (arch, kind, names)
            assert sum(loader.select_rule(loader.MODULES[name]) is None for name in names) == 1
            for name in names:
                record = loader.MODULES[name]
                twin = loader.MODULES[_twin_of(loader, arch, kind, name)]
                assert arch in record["arches"] and record["role"] == twin["role"] == "bwd"
                assert record["launch"]["block"] == twin["launch"]["block"], (arch, name)
                assert record["launch"]["cooperative"] is False
                for key in ("geometry", "ctas_per_sm", "kind"):
                    assert record["grid_rule"][key] == twin["grid_rule"][key], (arch, name, key)
                assert loader.record_route(record, arch) == PAIR_KINDS[kind]
                smem = int(record["launch"]["dynamic_smem_bytes"])
                assert smem > 0, (arch, name, "the render fills the shared memory")
                ctas = int(record["grid_rule"]["ctas_per_sm"])
                assert ctas * (smem + SMEM_RESERVED_PER_CTA) <= SMEM_PER_SM, (arch, name, smem, ctas)


@pytest.mark.cpu_only
def test_det_pair_routes_plan_their_twins_channel_tiles() -> None:
    """A deterministic pair route plans the channel tiles of its twin route (the narrow 32-vector tiles of C = 224
    and 520 to 1016 included), with and without a token count and architecture."""

    geometry = {"threads": 128, "vec": 8}
    for kind, twin in PAIR_KINDS.items():
        assert kind in det_jit.NARROW_TILE_ROUTES["bwd"] and twin in det_jit.NARROW_TILE_ROUTES["bwd"]
        for C in range(8, 8193, 8):
            for S, arch in ((None, None), *((LARGE_S, arch) for arch in det_jit.ARCHES)):
                assert det_jit.channel_split(C, geometry=geometry, role="bwd", route=kind, S=S, arch=arch) == (
                    det_jit.channel_split(C, geometry=geometry, role="bwd", route=twin, S=S, arch=arch)
                ), (kind, C, arch)
        assert det_jit.channel_split(1016, geometry=geometry, role="bwd", route=kind) == (32, 4, 4)


@pytest.mark.cpu_only
def test_det_pair_routes_share_one_pair_reduce_per_route() -> None:
    for arch in det_jit.ARCHES:
        for kind, twin in PAIR_KINDS.items():
            reduces = {det_jit.MODULES[name]["reduce"] for name in det_jit.route_modules(arch, kind)}
            twin_reduces = {det_jit.MODULES[name]["reduce"] for name in det_jit.route_modules(arch, twin)}
            assert len(reduces) == 1 and reduces.isdisjoint(twin_reduces), (arch, kind, reduces)
            (reduce_name,) = reduces
            assert det_jit.MODULES[reduce_name]["role"] == "reduce" and arch in det_jit.MODULES[reduce_name]["arches"]


@pytest.mark.cpu_only
def test_nondet_pair_programs_keep_hi_lo_accumulator_words() -> None:
    for arch in nondet_jit.ARCHES:
        for kind in PAIR_KINDS:
            for name in nondet_jit.route_modules(arch, kind):
                record = nondet_jit.MODULES[name]
                twin = nondet_jit.MODULES[_twin_of(nondet_jit, arch, kind, name)]
                slots, twin_slots = record["workspace_rule"]["slots"], twin["workspace_rule"]["slots"]
                assert record["workspace_rule"]["acc"] == "float32"
                assert slots % 2 == 0 and 0 < slots <= twin_slots, (arch, name, slots, twin_slots)
                taps = 5 if "bias" in kind else 4
                assert nondet_jit.accumulator_words_per_channel(record) == slots * taps


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_only_the_pair_opt_in_selects_a_pair_route(package: str) -> None:
    backend, loader = PACKAGES[package]
    for bias in (False, True):
        for fp32_grad in (False, True):
            for bf16 in (False, True):
                route = loader.backward_route(bias=bias, fp32_grad=fp32_grad, bf16_taps_or_dweight=bf16)
                assert route not in PAIR_KINDS
                if fp32_grad:
                    paired = loader.backward_route(bias=bias, fp32_grad=True, bf16_taps_or_dweight=bf16, pair=True)
                    assert paired == f"{route}_pair" and loader.PAIR_ROUTES[paired] == route
                else:
                    with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
                        loader.backward_route(bias=bias, fp32_grad=False, bf16_taps_or_dweight=bf16, pair=True)
    assert backend._dweight_sum("fp32", "bf16") == "fp32"
    assert backend._dweight_sum("fp32_pair", "fp32") == "fp32_pair"
    with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
        backend._dweight_sum("fp32_pair", "bf16")
    for bad in ("fp64", "pair", ""):
        with pytest.raises(ValueError, match="dweight_sum must be one of"):
            backend._dweight_sum(bad, "fp32")
    for bad in (None, 1, True, b"fp32"):
        with pytest.raises(TypeError, match="dweight_sum must be a str"):
            backend._dweight_sum(bad, "fp32")


@pytest.mark.cpu_only
def test_kcp_pair_route() -> None:
    for arch in kcp_jit.ARCHES:
        (name,) = kcp_jit.route_modules(arch, "kcp_bwd_dyn_pair")
        (twin_name,) = kcp_jit.route_modules(arch, "kcp_bwd_dyn")
        record, twin = kcp_jit.MODULES[name], kcp_jit.MODULES[twin_name]
        assert record["role"] == "kcp_bwd_dyn_pair" and twin["role"] == "kcp_bwd_dyn"
        assert record["launch"]["block"] == twin["launch"]["block"]
        assert record["workspace_rule"] == twin["workspace_rule"]
        assert record["grid_rule"] == twin["grid_rule"]
        smem = int(record["launch"]["dynamic_smem_bytes"])
        assert 0 < smem and smem + SMEM_RESERVED_PER_CTA <= SMEM_PER_SM, (arch, name, smem)
    assert nondet_kcp.KCP_BWD_DYN_PAIR == "kcp_bwd_dyn_pair" and "kcp_bwd_dyn_pair" in kcp_jit.ROUTE_KINDS
    assert tuple(nondet_kcp.DWEIGHT_SUMS) == ("fp32", "fp32_pair")
    assert nondet_kcp._dweight_sum(None, 0, "e") is False and nondet_kcp._dweight_sum("fp32", 0, "e") is False
    assert nondet_kcp._dweight_sum("fp32_pair", 0, "e") is True
    with pytest.raises(ValueError, match="runs on the overlapped backward"):
        nondet_kcp._dweight_sum("fp32_pair", 1, "e")
    for bad in ("fp64", ""):
        with pytest.raises(ValueError, match="dweight_sum must be None or one of"):
            nondet_kcp._dweight_sum(bad, 0, "e")
    for bad in (1, True, b"fp32_pair"):
        with pytest.raises(TypeError, match="dweight_sum must be None or a str"):
            nondet_kcp._dweight_sum(bad, 0, "e")
    limit = kcp_jit.DYN_MAX_CHANNELS
    nondet_kcp._pair_fits(True, limit, 100, 0, "e")
    nondet_kcp._pair_fits(True, limit + 8, 0, 0, "e")  # no tokens: no launch
    with pytest.raises(ValueError, match="runs on the overlapped backward"):
        nondet_kcp._pair_fits(True, limit + 8, 100, 0, "e")
