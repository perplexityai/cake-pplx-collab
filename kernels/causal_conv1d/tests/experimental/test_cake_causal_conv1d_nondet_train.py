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

Tests of the backend of the non-deterministic Cake packed width-4 depthwise causal convolution + SiLU training
package (``cake_pplx.cake_causal_conv1d_nondet_train``, backend
``cake_pplx._backends.cake_causal_conv1d_nondet_train``; SM100 / SM103 / SM107).

Every floating output (``y``, ``dx``, ``dweight``, ``dbias``) has no tolerance: it is judged against the naive
BF16 tier (``tests.cake.test_helpers.cake_naive``; PyTorch in BF16 on the same inputs) over 32 independent random
inputs of each case, by its error against the float64 oracle of the function (SiLU' at the unrounded preactivation
on every route and policy), in the L2 and the max norm: the median and the largest of its errors over the draws at
most 1.005 times the tier's, and 0 wherever the tier's is 0 (``tests.cake.test_helpers.cake_causal_conv1d_oracle``).
The package is not deterministic: no bit rule holds its outputs, and the suite never compares it with the
deterministic package.  ``REPEATS`` launches on each draw's inputs are all judged (the larger error of the launches
of a draw counts), and so are the fp32 policy's ``PRECISION_REPEATS`` launches at the perf size.

The workspace contract: a caller-owned zeroed workspace and the package's per-(device, stream) workspace read
back all zero after every launch, over changed shapes and routes; NaN-poisoned outputs are fully overwritten;
CUDA Graph capture / replay with and without a caller-owned workspace (each replay on a draw's inputs, judged),
one pre-zeroed package buffer per capture and stream shared by the capture's backward calls; two streams with two
workspaces at once; the backward is ONE ordinary launch (no reduce program) and keeps its speed beside a kernel
resident on another stream.  The pre-zeroed capture buffers: captures record no fill node (with the pool forced empty,
one fill per captured call, as ``_CAPTURE_FILL_FALLBACKS`` counts, and the package keeps no reference to those
buffers), two graphs replayed at once, eager calls interleaved with replays, buffers returned after graph deletion
(also with a replay in flight and after an aborted capture), a capture without synchronization, more captures than the
pool holds, a capture larger than the pool, pool growth (on a stream held on the device until the checks end; a
graph's released buffer smaller than the grown size dropped), one thread at a time inside the package's memory pool
(a refill and two threads' first eager calls), no ``cuda.bindings``, no ``torch.cuda.MemPool`` (eager calls judged
too), and CUDA Graph trees (``torch.compile``, backend ``"cudagraphs"``; one with more recorded backward nodes
than the pool holds).  The default backward route's ``FALLBACK_ROWS`` programs under CUDA Graph replay (``bwd_grid_mult``
1 to 4, the program checked per row), programmatic dependent launch, repeated launches and two streams.  Programmatic dependent
launch: ``enable_pdl`` follows ``PDL_EAGER`` / ``PDL_GRAPH``, except the backward's 0 at the rows of ``PDL_OFF_ROWS``
(eager and captured; one token below a row and the forward keep the tables' value); with the policy forced on, and
with the shipped graph policy, a launch reads what the kernel before it wrote and does not overwrite what that kernel
reads, and back-to-back launches share the package's workspaces; both packages in one process.  Autograd,
non-reentrant activation checkpointing (2 x forward + backward) and gradient accumulation; torch's deterministic mode.

The opt-in ``bwd_grid_mult`` (every value of ``BWD_GRID_MULTS``) on every route, at single-tile, multi-sub-tile and
partial-last-tile channel counts and at small and large ``S``: the backward launches that many times its untrimmed
grid, its outputs are judged against the naive tier, the workspace (the same size at every multiple) reads back zero
after every launch, under CUDA Graph capture / replay and through the autograd entry (plain and checkpointed).

The narrower channel tiles of ``TILE_OVERRIDES``: each row of ``OVERRIDE_CASES`` (one per architecture, on sm_107a on
both routes that plan by it) launches the row's
tiles, on a caller-owned workspace of the public size and on the package's, with every output judged against the
naive tier and the workspace zero after the call.  The 32-vector tiles of ``NARROW_TILES`` the same way at every key of the architecture's table
(``NARROW_CASES``: the key's channel ends and an inner channel count, at token counts inside the key's range), and
the side-stream one-SM trim of a three-tile narrow row's grid.  The forward's narrow rows (``FORWARD_NARROW_CASES``,
the dynamic program's rows with the forward keys removed: C 520 to 768 at the row's token floors on the device's
architecture, the row's route and the taps' dtype,
``forward_narrow_min_s`` and ``FORWARD_WIDTH_FLOORS``, and one token below a floor or inside a single-tile range of
``FORWARD_SINGLE_TILE_ROWS``) launch their 32-vector, width or single tiles on a caller-owned forward workspace of the
public size and on the package's, with ``y`` judged against the naive tier and the workspace zero after the call.  The
backward rows of ``FALLBACK_ROWS`` (each key's channel ends and C 640 / 896 at the key's last token count and the next)
launch the route's fallback program inside the key's token range and the rules' program above it.  The rows of
``SM_WIDE_ROWS`` (route ``bwd``: each key's lowest, an inner and its highest channel count on both sides of every
segment's first token count) launch the segment's SM-wide program (one CTA per SM), judged against the naive tier, on a
caller-owned workspace and the package's (sm_107a's program from 131072 tokens keeps two accumulator copies in the
package-sized workspace), repeated, at every ``bwd_grid_mult``, under CUDA Graph replay, on two streams, with
programmatic dependent launch, through autograd and checkpointing, beside a kernel that holds one SM's shared memory on
another stream, and beside the deterministic package's SM-wide program in one process.  The forward's narrow rows of
C 520 to 568 (every one, at the row's token floor and one token below it), the ``fwd_bias`` rows with BF16 taps (their
own floor and single-tile ranges on sm_103a), and a CUDA Graph capture at a forward row's floor after an eager call one
token below it.  The forward keys of ``FALLBACK_ROWS`` and ``DIRECT_ROWS`` (C 576 to 768 below the token ceilings): at S
1000 and on both sides of every split and ceiling the static, direct or dynamic program the tables name, the static and
direct programs with no ``work`` counters, every ``y`` judged (the route's static program on the same inputs too), with
``enable_pdl`` 0 and 1, on two streams and under CUDA Graph replay (no fill node at a static or direct row); a capture
above a direct range after an eager call inside it (that call refills no pre-zeroed buffer, so the capture zeroes its
own); and both packages in one process at a direct row.

This suite names its package once, through ``tests.cake.test_helpers.cake_package`` (key
``cake_causal_conv1d_nondet_train``).  Every test skips until the exporter has rendered the package (``build.json``
and ``csrc/``).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import os
import re
import statistics
import sys
import tempfile
import threading
import time
import weakref

import pytest
import torch

from tests.cake.test_helpers.cake_causal_conv1d_oracle import (
    assert_judged,
    naive_bf16,
    oracle,
    problem,
)
from tests.cake.test_helpers.cake_naive import (
    DRAWS,
    TierJudge,
    contract,
    draw_seed,
    seed_all,
)
from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTH = 4
F32, BF16 = torch.float32, torch.bfloat16
PERF_FRACTIONS = (0.37, 0.29, 0.21)
SEED = 8065
# Launches per draw of the repeated-launch rows (every launch is judged).
REPEATS = 20

# (S, C, documents): one-token and short documents, empty documents before / between / after others, boundaries
# inside a sub-tile and at 64 / 65, the perf packing, 38 documents, the channel thresholds of the routed programs,
# the multi-sub-tile tiles (1280, 1536, 2560) and the partial last channel tiles (1160, 1344, 1600).
CASES = (
    (1, 128, None),
    (5, 136, (2, 0, 3)),
    (63, 384, (1, 2, 3, 57)),
    (64, 128, None),
    (65, 256, (64, 1)),
    (1000, 512, "perf"),
    (1000, 256, (27,) * 37 + (1,)),
    (4097, 1024, (0, 4097, 0)),
    (4097, 2048, (1000, 0, 3, 3094)),
    (4097, 768, (1000, 0, 3, 3094)),
    (5, 512, (2, 0, 3)),
    (5, 1024, (2, 0, 3)),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1280, (2, 0, 3)),
    (5, 1536, (2, 0, 3)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (4097, 1160, (1000, 0, 3, 3094)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (1000, 1600, "perf"),
)
ROUTE_CASES = (
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (5, 1536, (2, 0, 3)),
    (65, 256, (64, 1)),
    (4097, 1344, (1000, 0, 3, 3094)),
)
# A large row per architecture class for the repeated launches (many CTAs per channel tile race on the accumulators).
REPEAT_CASES = (
    (65536, 512, "perf"),
    (65536, 1280, "perf"),
    (4097, 1160, (1000, 0, 3, 3094)),
)
PRECISION_CASES = ((250340, 512, "perf"), (250340, 1280, "perf"))
# Rows of ``TILE_OVERRIDES`` per architecture (route, S, C): the backward runs its narrower channel tiles there.  The
# sm_107a ``bwd_dt`` row serves both routes that plan by it: ``bwd_dt_bf16_taps`` (BF16 taps, through
# ``TILE_ROUTE``) and ``bwd_dt`` (FP32 taps with a BF16 ``dweight`` out).
OVERRIDE_CASES = {
    "sm_100a": (("bwd_bias_fp32g", 62585, 2048),),
    "sm_103a": (("bwd", 62585, 1536),),
    "sm_107a": (("bwd_dt_bf16_taps", 62585, 768), ("bwd_dt", 62585, 768)),
}
# The token counts a ``NARROW_TILES`` key over every S (one token to ``INT32_MAX``) launches its rows at, in the perf
# packing: the largest S of ``CASES`` and a longer stream.
EVERY_S_TOKENS = (4097, 31292)


def _key_tokens(s_min: int, s_max: int) -> tuple[int, ...]:
    """The token counts a ``NARROW_TILES`` key's rows launch at: its smallest token count and one token above it
    (inside its range), or :data:`EVERY_S_TOKENS` for a key over every S."""

    if (s_min, s_max) == (1, loader.INT32_MAX):
        return EVERY_S_TOKENS
    return tuple(sorted({s_min, min(s_min + 1, s_max)}))


def _narrow_tokens(arch: str, route: str, C: int, s_min: int, s_max: int) -> tuple[int, ...]:
    """The token counts of :func:`_key_tokens` at which the narrow tiles run for ``(arch, route, C)``: a token count
    inside a segment of ``SM_WIDE_ROWS`` (where the SM-wide program runs its own one-tile split, not the narrow
    tiles) is replaced by one token below the first segment of the key covering ``C``, when the narrow key's range
    holds it."""

    tokens = _key_tokens(s_min, s_max)
    segments = loader.sm_wide_segments(arch, route, C)
    if segments is None:
        return tokens
    kept = {S for S in tokens if loader.sm_wide_module(arch, route, C, S) is None}
    below = segments[0][0] - 1
    if len(kept) < len(tokens) and s_min <= below <= s_max:
        kept.add(below)
    return tuple(sorted(kept))


# The rows of ``NARROW_TILES`` (rendered with the table; empty for an architecture without rows) the narrow-tile
# test launches, per architecture (route, S, C): each key's lowest, an inner and its highest channel count at the
# token counts of :func:`_narrow_tokens` (inside the delivered range; ``CASES`` stop at S = 4097; at a channel count
# of ``SM_WIDE_ROWS`` the token count below its first segment), on the key's route and on every route that plans by
# its rows (``TILE_ROUTE``).
NARROW_CASES = {
    arch: tuple(
        (route, S, C)
        for (base, c_lo, c_hi), (s_min, s_max) in sorted(rows.items())
        for route in (base, *sorted(t for t, b in loader.TILE_ROUTE.items() if b == base))
        for C in sorted({int(c_lo), (int(c_lo) + int(c_hi)) // 16 * 8, int(c_hi)})
        for S in _narrow_tokens(arch, route, C, int(s_min), int(s_max))
    )
    for arch, rows in loader.NARROW_TILES.items()
}
# The forward's narrow rows (C 520 to 768) in each region of the device's architecture, (route, C, tiles, taps):
# ``"narrow"`` at the row's ``forward_narrow_min_s`` floor (the 32-vector tiles), ``"wide"`` at the row's
# ``FORWARD_WIDTH_FLOORS`` floor (its ``FORWARD_NARROW_WIDTHS`` tiles: C 576 as (18, 7, 4), C 640 as (40, 3, 2), C 520
# as (13, 9, 5)), ``"below"`` one token below the row's floor (the single tile of C / 8 vectors), ``"gap"`` inside the
# row's ``FORWARD_SINGLE_TILE_ROWS`` range (the single tile; on an architecture without the range, at sm_103a's range
# start, the plan of the device's own tables there); ``taps`` the call's tap dtype (``"f32"`` / ``"bf16"``, read on
# ``fwd_bias``: :data:`FORWARD_SETTINGS`).  The token counts come from the package's tables at run time
# (:func:`_forward_narrow_tokens`, per route and taps).  Every row on the default route ``fwd``; the rows of C 520 to
# 568 (every one of them, whose row floors are their own, ``FORWARD_NARROW_ROW_FLOORS``) at their floor and one token
# below it; rows on each other forward route, C 768 among them (the ``fwd_bias`` route's floors differ on sm_103a by
# the taps' dtype).
FORWARD_NARROW_CASES = (
    ("fwd", 576, "narrow", "f32"),
    ("fwd", 576, "wide", "f32"),
    ("fwd", 640, "narrow", "f32"),
    ("fwd", 640, "wide", "f32"),
    ("fwd", 704, "narrow", "f32"),
    ("fwd", 768, "narrow", "f32"),
    ("fwd_dt", 704, "narrow", "bf16"),
    ("fwd_dt", 768, "narrow", "bf16"),
    ("fwd_bias", 576, "wide", "f32"),
    ("fwd_bias", 768, "narrow", "f32"),
    ("fwd_bias", 768, "narrow", "bf16"),
    ("fwd_bias", 768, "below", "bf16"),
    ("fwd_bias", 768, "gap", "bf16"),
    ("fwd_bias", 712, "gap", "bf16"),
    ("fwd", 520, "wide", "f32"),
    ("fwd", 528, "wide", "f32"),
    ("fwd", 536, "narrow", "f32"),
    ("fwd", 544, "wide", "f32"),
    ("fwd", 552, "wide", "f32"),
    ("fwd", 560, "wide", "f32"),
    ("fwd", 568, "narrow", "f32"),
    ("fwd", 520, "below", "f32"),
    ("fwd", 528, "below", "f32"),
    ("fwd", 536, "below", "f32"),
    ("fwd", 544, "below", "f32"),
    ("fwd", 552, "below", "f32"),
    ("fwd", 560, "below", "f32"),
    ("fwd", 568, "below", "f32"),
    ("fwd_dt", 528, "wide", "bf16"),
    ("fwd_dt", 568, "narrow", "bf16"),
    ("fwd_bias", 552, "wide", "f32"),
    ("fwd_bias", 536, "narrow", "bf16"),
)
# The taps' and the bias's dtypes (``None``: no bias) of a forward call of each route and tap dtype.
FORWARD_SETTINGS = {
    ("fwd", "f32"): (F32, None),
    ("fwd_dt", "bf16"): (BF16, None),
    ("fwd_bias", "f32"): (F32, F32),
    ("fwd_bias", "bf16"): (BF16, BF16),
}
# Wide channel counts on every route: C = 4096 (four documents, one empty) and C = 10240 (a packed q / k / v
# projection width), at small S.
WIDE_CASES = ((4097, 4096, (1000, 0, 3, 3094)), (2048, 10240, "perf"))
# Launches per draw of the fp32 policy's perf-size rows (every launch is judged).
PRECISION_REPEATS = 5
# S x C > 2^31 elements (64-bit offsets) at a token count past the sm_107a token floor.
LARGE_CASE = (262401, 8192, "perf")
# The large case's inputs and outputs (x, dy, y, dx: 4 x 4.3 GB), its float64 oracle of y and dx (2 x 17.2 GB) and
# the naive tier's BF16 y and dx with their blocks (about 4 x 4.3 GB), with headroom.
LARGE_CASE_FREE_BYTES = 96 * 2**30
GRAPH_CASES = (
    (65, 256, (64, 1)),
    (5, 136, (2, 0, 3)),
    (1000, 512, "perf"),
    (4097, 1280, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
)
# The ``bwd_grid_mult`` rows: fewer tokens than CTAs (S = 5, 65), a single-tile C, the multi-sub-tile tiles (1280,
# 1536, 2560) and the partial last channel tiles (136, 1160, 1344), at small S and at large S (65536, the perf size).
GRID_MULT_CASES = (
    (5, 136, (2, 0, 3)),
    (65, 256, (64, 1)),
    (5, 1536, (2, 0, 3)),
    (4097, 1344, (1000, 0, 3, 3094)),
    (4097, 2560, (1000, 0, 3, 3094)),
    (65536, 1160, "perf"),
    (65536, 1280, "perf"),
    (250340, 512, "perf"),
)
# The call settings that resolve to each backward route.  ``dweight_dtype`` is the dtype of the preallocated
# ``dweight`` out a call of the route passes (``None``: none, the taps' dtype): ``bwd_dt`` is FP32 taps with a BF16
# ``dweight`` out (:func:`_backward` with ``kind`` allocates a fresh one per call).
ROUTE_SETTINGS = {
    "bwd": dict(weight_dtype=F32, bias_dtype=None, policy="bf16", dweight_dtype=None),
    "bwd_dt": dict(weight_dtype=F32, bias_dtype=None, policy="bf16", dweight_dtype=BF16),
    "bwd_bias": dict(weight_dtype=F32, bias_dtype=F32, policy="bf16", dweight_dtype=None),
    "bwd_fp32g": dict(weight_dtype=F32, bias_dtype=None, policy="fp32", dweight_dtype=None),
    "bwd_bias_fp32g": dict(weight_dtype=F32, bias_dtype=F32, policy="fp32", dweight_dtype=None),
    "bwd_dt_bf16_taps": dict(weight_dtype=BF16, bias_dtype=None, policy="bf16", dweight_dtype=None),
    "bwd_bias_bf16_taps": dict(weight_dtype=BF16, bias_dtype=BF16, policy="bf16", dweight_dtype=None),
}
# The backward routes the autograd entry reaches: it returns ``dweight`` in the taps' dtype, so ``bwd_dt`` (FP32 taps
# with a BF16 ``dweight`` out) is a route of direct backward calls only.
AUTOGRAD_KINDS = tuple(
    kind for kind, settings in ROUTE_SETTINGS.items() if settings["dweight_dtype"] is None
)
FORWARD_OF = {
    "bwd": "fwd",
    "bwd_dt": "fwd",
    "bwd_bias": "fwd_bias",
    "bwd_fp32g": "fwd",
    "bwd_bias_fp32g": "fwd_bias",
    "bwd_dt_bf16_taps": "fwd_dt",
    "bwd_bias_bf16_taps": "fwd_bias",
}
# Every dtype combination a backward call takes: taps x bias (none, FP32, BF16) x SiLU' policy x the dweight /
# dbias out dtypes (each FP32 or BF16, independently of its parameter).  Each resolves to one route: BF16 taps on
# the fp32g routes, BF16 taps under the "bf16" policy (bwd_dt_bf16_taps, bwd_bias_bf16_taps, with either dweight
# dtype) and FP32 taps with a BF16 dweight (bwd_dt, and bwd_bias with a bias) are among them.
DTYPE_COMBOS = tuple(
    (weight_dtype, bias_dtype, policy, dweight_dtype, dbias_dtype)
    for policy in ("bf16", "fp32")
    for weight_dtype in (F32, BF16)
    for bias_dtype in (None, F32, BF16)
    for dweight_dtype in (F32, BF16)
    for dbias_dtype in ((None,) if bias_dtype is None else (F32, BF16))
)
# The autograd entry returns the gradients in the parameters' dtypes: taps x bias x policy.
AUTOGRAD_COMBOS = tuple(
    (weight_dtype, bias_dtype, policy)
    for policy in ("bf16", "fp32")
    for weight_dtype in (F32, BF16)
    for bias_dtype in (None, F32, BF16)
)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_causal_conv1d_nondet_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs an SM100 / SM103 / SM107 device and "
        "its rendered build.json / csrc"
    ),
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _case_id(case) -> str:
    S, C, docs = case
    tag = "one" if docs is None else ("perf" if docs == "perf" else f"{len(docs)}docs")
    return f"S{S}-C{C}-{tag}"


def _dtype_id(dtype) -> str:
    return {None: "none", F32: "f32", BF16: "bf16"}[dtype]


def _combo_id(combo) -> str:
    weight_dtype, bias_dtype, policy, *outs = combo
    label = f"w{_dtype_id(weight_dtype)}-b{_dtype_id(bias_dtype)}-{policy}"
    if outs:
        dweight_dtype, dbias_dtype = outs
        label += f"-dw{_dtype_id(dweight_dtype)}"
        if bias_dtype is not None:
            label += f"-db{_dtype_id(dbias_dtype)}"
    return label


def _combo_route(weight_dtype, bias_dtype, policy, dweight_dtype=None) -> str:
    """The backward route a call of these dtypes runs (``dweight_dtype`` ``None``: the taps' dtype)."""

    return loader.backward_route(
        bias=bias_dtype is not None,
        fp32_grad=policy == "fp32",
        bf16_taps=weight_dtype == BF16,
        bf16_dweight=(dweight_dtype or weight_dtype) == BF16,
    )


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _route_pending(arch: str, kind: str) -> str | None:
    if kind not in loader.ROUTES.get(arch, {}):
        return f"route {kind!r} is not exported for {arch}"
    libraries = cake_build.libraries_by_name()
    for name in loader.route_modules(arch, kind):
        library = loader.library_name(name, arch)
        if library not in libraries:
            return f"route {kind!r}: {library} is not in build.json yet (awaiting the exporter's render)"
    return None


def _require_route(*kinds: str) -> None:
    for kind in kinds:
        for needed in (kind, FORWARD_OF.get(kind, kind)):
            pending = _route_pending(_arch(), needed)
            if pending:
                pytest.skip(pending)


def _backward_thresholds(arch: str, kind: str) -> tuple[int, ...]:
    """The channel thresholds of the programs route ``kind`` selects by channel count on ``arch`` (empty when the
    route ships its fallback program only there)."""

    thresholds = set()
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and rule["kind"] != loader.ROWS_KIND and arch in rule["channels"]:
            thresholds.add(int(rule["channels"][arch]))
    return tuple(sorted(thresholds))


def _token_floors(arch: str, kind: str) -> dict[str, tuple[int, int]]:
    """The programs of route ``kind`` on ``arch`` whose rule carries a token floor: name -> (channel threshold,
    token floor)."""

    floors = {}
    for name in loader.route_modules(arch, kind):
        rule = loader.select_rule(loader.MODULES[name])
        if rule is not None and rule["kind"] != loader.ROWS_KIND and arch in rule.get("tokens", {}):
            floors[name] = (int(rule["channels"][arch]), int(rule["tokens"][arch]))
    return floors


# The channel counts inside the ``FALLBACK_ROWS`` keys the channel-switch test adds besides each key's ends.
FALLBACK_INNER_CHANNELS = (640, 896)


def _fallback_rows(arch: str, kind: str) -> list[tuple[int, int]]:
    """``(C, S_max)`` of the ``FALLBACK_ROWS`` keys of route ``kind`` on ``arch``: each key's channel ends and the
    channel counts of :data:`FALLBACK_INNER_CHANNELS` the key covers, with the key's last token count."""

    rows = set()
    for (route, c_lo, c_hi), (_s_min, s_max) in loader.FALLBACK_ROWS.get(arch, {}).items():
        if route == kind:
            for C in (int(c_lo), int(c_hi), *(c for c in FALLBACK_INNER_CHANNELS if c_lo <= c <= c_hi)):
                rows.add((C, int(s_max)))
    return sorted(rows)


def _fallback_program(arch: str, kind: str) -> str | None:
    """The route's fallback program on ``arch`` (its record without a selection rule); ``None`` when the route is not
    exported there."""

    names = [
        name for name in loader.route_modules(arch, kind) if loader.select_rule(loader.MODULES[name]) is None
    ] if kind in loader.ROUTES.get(arch, {}) else []
    return names[0] if len(names) == 1 else None


# The default backward route's channel counts at its ``FALLBACK_ROWS`` keys that the graph, PDL, grid-multiple,
# repeated-launch and two-stream tests add (C 768: the 768 key; C 1016: the top of the 776-1016 key), with the
# document layout of their S 4097 rows.
FALLBACK_ROW_CHANNELS = (768, 1016)
FALLBACK_ROW_DOCS = (1000, 0, 3, 3094)
# The architectures that ship ``FALLBACK_ROWS`` keys at those channel counts (asserted by the tests that use them).
FALLBACK_ROW_ARCHES = ("sm_100a", "sm_103a")


def _fallback_row_bounds(C: int) -> tuple[int, int]:
    """The token range of the default backward route's ``FALLBACK_ROWS`` key covering ``C`` on this architecture, or
    on an architecture without one (sm_107a, which runs the same rows on its own plan) sm_100a's."""

    arch = _arch()
    bounds = loader.fallback_row_range(arch, "bwd", C)
    if arch in FALLBACK_ROW_ARCHES:
        assert bounds is not None, f"{arch} has no FALLBACK_ROWS key at C={C}"
    return bounds or loader.fallback_row_range("sm_100a", "bwd", C)


def _fallback_row_cases() -> tuple:
    """``(S, C, docs)``: C 768 and C 1016 at S 4097 (inside their keys' token ranges: the route's fallback program on
    sm_100a / sm_103a) and C 1016 one token above its key's range (the rules' program)."""

    above = _fallback_row_bounds(1016)[1] + 1
    return (
        *((4097, C, FALLBACK_ROW_DOCS) for C in FALLBACK_ROW_CHANNELS),
        (above, 1016, "perf"),
    )


def _fallback_row_large_cases() -> tuple:
    """``(S, C, "perf")``: C 768 and C 1016 with many CTAs per channel tile, at the smaller of S 65536 and the key's
    last token count (inside the key's range)."""

    return tuple((min(65536, _fallback_row_bounds(C)[1]), C, "perf") for C in FALLBACK_ROW_CHANNELS)


def _expected_program(arch: str, kind: str, S: int, C: int) -> str:
    """The program route ``kind`` launches at ``(S, C)`` on ``arch``, read from the tables (asserted against
    ``selected_modules``): inside an ``SM_WIDE_ROWS`` segment the segment's program, else inside a ``DIRECT_ROWS``
    range the route's direct program, else inside a ``FALLBACK_ROWS`` key's token range the route's fallback program
    (a forward route's static program), else the rules' program (the three tables emptied)."""

    (selected,) = loader.selected_modules(arch, kind, C=C, S=S)
    segment = [name for s_lo, s_hi, name in loader.sm_wide_segments(arch, kind, C) or () if s_lo <= S <= s_hi]
    bounds = loader.fallback_row_range(arch, kind, C)
    with pytest.MonkeyPatch.context() as rules_only:
        rules_only.setitem(loader.FALLBACK_ROWS, arch, {})
        rules_only.setitem(loader.SM_WIDE_ROWS, arch, {})
        rules_only.setitem(loader.DIRECT_ROWS, arch, {})
        (by_rules,) = loader.selected_modules(arch, kind, C=C, S=S)
    if segment:
        expected = segment[0]
    elif loader.direct_row(arch, kind, C, S):
        expected = _forward_program(arch, kind, "direct")
    elif bounds is not None and bounds[0] <= S <= bounds[1]:
        expected = _fallback_program(arch, kind)
    else:
        expected = by_rules
    assert selected == expected, (kind, S, C, selected, expected)
    return selected


def _forward_program(arch: str, kind: str, form: str) -> str | None:
    """The static (``form`` ``"static"``: the record without a selection rule) or direct (``"direct"``: the record of
    selection-rule kind ``rows``) program of forward route ``kind`` on ``arch``; ``None`` when the route lists none."""

    want = {"static": None, "direct": loader.ROWS_KIND}[form]
    names = [
        name
        for name in loader.route_modules(arch, kind)
        if (loader.select_rule(loader.MODULES[name]) or {"kind": None})["kind"] == want
    ]
    assert len(names) <= 1, (arch, kind, form, names)
    return names[0] if names else None


def _forward_keys(arch: str) -> list[tuple[str, int, int, int, int, str]]:
    """``(route, C_lo, C_hi, S_lo, S_hi, form)`` of the forward keys of ``arch``: each ``FALLBACK_ROWS`` key of a
    forward route (form ``"static"``) and each ``DIRECT_ROWS`` key (form ``"direct"``)."""

    keys = [
        (route, int(c_lo), int(c_hi), int(s_lo), int(s_hi), "static")
        for (route, c_lo, c_hi), (s_lo, s_hi) in loader.FALLBACK_ROWS.get(
            arch, {}
        ).items()
        if loader.route_role(route) == "fwd"
    ]
    keys += [
        (route, int(c_lo), int(c_hi), int(s_lo), int(s_hi), "direct")
        for (route, c_lo, c_hi), (s_lo, s_hi, _) in loader.DIRECT_ROWS.get(
            arch, {}
        ).items()
    ]
    return sorted(keys)


@contextlib.contextmanager
def _forward_rules_only(
    arch: str, route: str | None = None, static_from: tuple[int, int] | None = None
):
    """Inside the block the forward keys of ``arch`` are gone (``DIRECT_ROWS`` emptied, ``FALLBACK_ROWS`` without its
    forward keys), so every forward call selects by the rules (the dynamic programs, which run every call above the
    keys); with ``route`` and ``static_from`` (``(C_lo, C_hi)``) one key over every token count puts the route's
    static program on those channel counts instead.  The host's row plans are dropped on entry and on exit."""

    rows = {
        key: value
        for key, value in loader.FALLBACK_ROWS.get(arch, {}).items()
        if loader.route_role(key[0]) != "fwd"
    }
    if route is not None:
        rows[(route, *static_from)] = (1, loader.INT32_MAX)
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    try:
        with pytest.MonkeyPatch.context() as m:
            m.setitem(loader.DIRECT_ROWS, arch, {})
            m.setitem(loader.FALLBACK_ROWS, arch, rows)
            yield
    finally:
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _expected_backward_program(S: int, C: int) -> str:
    """The program the default backward route launches at ``(S, C)`` here (:func:`_expected_program`)."""

    return _expected_program(_arch(), "bwd", S, C)


def _sm_wide_key(arch: str) -> tuple[int, int, tuple[tuple[int, int, str], ...]]:
    """``(C_lo, C_hi, segments)`` of the default backward route's ``SM_WIDE_ROWS`` key on ``arch`` (one per
    architecture, asserted)."""

    keys = [
        (int(c_lo), int(c_hi), segments)
        for (route, c_lo, c_hi), segments in loader.SM_WIDE_ROWS.get(arch, {}).items()
        if route == "bwd"
    ]
    assert len(keys) == 1, f"{arch}: expected one SM_WIDE_ROWS key of route bwd, got {keys}"
    return keys[0]


def _sm_wide_mid(c_lo: int, c_hi: int) -> int:
    """An inner channel count of an ``SM_WIDE_ROWS`` key (a multiple of 8)."""

    return (c_lo + c_hi) // 16 * 8


def _sm_wide_cases() -> tuple:
    """``(S, C, "perf")``: per segment of the device's ``SM_WIDE_ROWS`` key its first token count at an inner channel
    count of the key (sm_107a's two-copy program among them)."""

    c_lo, c_hi, segments = _sm_wide_key(_arch())
    return tuple((int(s_lo), _sm_wide_mid(c_lo, c_hi), "perf") for s_lo, _, _ in segments)


# The segment indices of the default backward route's ``SM_WIDE_ROWS`` keys, over every architecture (a test of one
# segment index runs that segment of the device's key).
SM_WIDE_SEGMENTS = tuple(
    range(
        max(
            (
                len(segments)
                for rows in loader.SM_WIDE_ROWS.values()
                for (route, _, _), segments in rows.items()
                if route == "bwd"
            ),
            default=0,
        )
    )
)


def _record_programs(monkeypatch) -> list:
    """From now on, every launch of the package appends ``(module name, S, C)`` (the launch's plan scalars) to the
    returned list."""

    seen = []
    launch = pkg.run_module

    def record(name, values, *, device_index, grid=None):
        seen.append((name, values.get("S"), values.get("C")))
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record)
    return seen


def _backward_programs(seen: list) -> set:
    """The default backward route's programs among the launches ``seen``."""

    route = set(loader.route_modules(_arch(), "bwd"))
    return {name for name, _, _ in seen if name in route}


def _backward_programs_by_shape(seen: list) -> dict:
    """The default backward route's programs among the launches ``seen``, per launched ``(S, C)``."""

    route = set(loader.route_modules(_arch(), "bwd"))
    found: dict = {}
    for name, S, C in seen:
        if name in route:
            found.setdefault((int(S), int(C)), set()).add(name)
    return found


def _generator(*keys: int, seed: int = SEED) -> torch.Generator:
    """A generator of the device seeded from ``seed`` (a draw's, or :data:`SEED`) and ``keys``."""

    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def doc_lengths(S: int, docs) -> tuple[int, ...]:
    if docs is None:
        return (S,)
    if docs == "perf":
        lens = []
        for fraction in PERF_FRACTIONS:
            n = int(fraction * S)
            if n % 64 == 0:
                n -= 1
            lens.append(n)
        lens.append(S - sum(lens))
        assert all(n > 0 for n in lens), f"S={S} is too short for the perf packing"
        return tuple(lens)
    lens = tuple(int(n) for n in docs)
    assert sum(lens) == S and all(n >= 0 for n in lens)
    return lens


def _cu_seqlens(S: int, docs) -> torch.Tensor:
    lens = doc_lengths(S, docs)
    return torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=_device()
    )


def _inputs(S: int, C: int, docs, *, key: int = 0, weight_dtype=F32, seed: int = SEED):
    """``(x, w, cu, dy)``: BF16 ``x`` / ``dy`` of N(0, 1), taps of N(0, 0.25) in ``weight_dtype``, drawn from
    ``seed`` (a draw's seed; :data:`SEED` is draw 0's)."""

    generator = _generator(S, C, key, seed=seed)
    x = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(BF16)
    w = 0.5 * torch.randn(
        C, WIDTH, dtype=torch.float32, device=_device(), generator=generator
    )
    dy = torch.randn(
        S, C, dtype=torch.float32, device=_device(), generator=generator
    ).to(BF16)
    return x, w.to(weight_dtype), _cu_seqlens(S, docs), dy


def _bias_input(C: int, dtype, *, key: int = 0, seed: int = SEED) -> torch.Tensor:
    generator = _generator(C, 97, key, seed=seed)
    return (
        0.5 * torch.randn(C, dtype=torch.float32, device=_device(), generator=generator)
    ).to(dtype)


def _route_inputs(S: int, C: int, docs, kind: str, *, key: int = 0):
    """``(x, w, cu, dy, b, policy)`` resolving to backward route ``kind``."""

    settings = ROUTE_SETTINGS[kind]
    x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=settings["weight_dtype"])
    b = (
        None
        if settings["bias_dtype"] is None
        else _bias_input(C, settings["bias_dtype"], key=key)
    )
    return x, w, cu, dy, b, settings["policy"]


def _forward(x, w, cu, **kwargs) -> torch.Tensor:
    return pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, **kwargs)


def _backward(x, w, cu, dy, b=None, policy="bf16", *, kind=None, **kwargs):
    """``(dx, dweight, dbias_or_None)`` of one non-deterministic backward call.  With ``kind`` (a backward route of
    :data:`ROUTE_SETTINGS`) and no ``dweight`` out passed, the call writes a fresh NaN-filled ``dweight`` out of the
    route's ``dweight_dtype`` when it names one (``bwd_dt``), and the call's arguments must resolve to ``kind``."""

    if kind is not None:
        dweight_dtype = ROUTE_SETTINGS[kind]["dweight_dtype"]
        if kwargs.get("dweight") is None and dweight_dtype is not None:
            kwargs["dweight"] = torch.full(
                (w.shape[0], WIDTH), float("nan"), dtype=dweight_dtype, device=w.device
            )
        dweight = kwargs.get("dweight")
        route = _combo_route(
            w.dtype, None if b is None else b.dtype, policy, None if dweight is None else dweight.dtype
        )
        assert route == kind, f"the call resolves to route {route}, not {kind}"
    grads = pkg.cake_causal_conv1d_nondet_train_backward(
        x,
        w,
        cu,
        dy,
        bias=b,
        silu_grad_preactivation=policy,
        **kwargs,
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _grads(dx, dw, db) -> dict:
    """The gradients of one backward call by output name (``dbias`` only with a bias)."""

    out = {"dx": dx, "dweight": dw}
    if db is not None:
        out["dbias"] = db
    return out


def _forward_backward(x, w, cu, dy, b, policy, **kwargs) -> dict:
    """``y`` of one forward call and the gradients of one backward call, by output name."""

    y = _forward(x, w, cu, bias=b)
    return {"y": y, **_grads(*_backward(x, w, cu, dy, b, policy, **kwargs))}


def _judge(
    S: int,
    C: int,
    docs,
    call,
    *,
    weight_dtype=F32,
    bias_dtype=None,
    policy: str = "bf16",
    key: int = 0,
    label: str = "",
    perturb=None,
) -> None:
    """One problem judged over the draws: each draw's inputs (``weight_dtype`` taps, a ``bias_dtype`` bias or none;
    ``perturb(x, dy)`` returns the draw's ``x`` and ``dy`` changed) and ``call(x, w, cu, dy, b, policy)``, which
    launches and returns the outputs by name (a value, or a list of the values of several launches on the draw)."""

    def launch(_draw: int, seed: int):
        x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=weight_dtype, seed=seed)
        b = None if bias_dtype is None else _bias_input(C, bias_dtype, key=key, seed=seed)
        if perturb is not None:
            x, dy = perturb(x, dy)
        return call(x, w, cu, dy, b, policy), problem(x, w, cu, b, dy)

    assert_judged(
        launch,
        seed=SEED,
        label=label
        or f"S={S} C={C} docs={docs} w={_dtype_id(weight_dtype)} b={_dtype_id(bias_dtype)} {policy}",
    )


def _route_judge(S: int, C: int, docs, kind: str, call, **kwargs) -> None:
    """:func:`_judge` with the taps / bias dtypes and the policy that resolve to backward route ``kind`` (``call``
    passes ``kind=kind`` to :func:`_backward`, which writes the route's ``dweight`` out)."""

    settings = ROUTE_SETTINGS[kind]
    _judge(
        S,
        C,
        docs,
        call,
        weight_dtype=settings["weight_dtype"],
        bias_dtype=settings["bias_dtype"],
        policy=settings["policy"],
        **kwargs,
    )


def _stream_workspace(device: torch.device) -> torch.Tensor | None:
    """The package's backward workspace of the device's current stream (``None`` before its first call)."""

    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


def _forward_stream_workspace(device: torch.device) -> torch.Tensor | None:
    """The package's forward workspace of the device's current stream (``None`` before its first call)."""

    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._FORWARD_WORKSPACES.get(key)


def _forward_narrow_tokens(arch: str, kind: str, channel_vectors: int, tiles: str, bf16_taps: bool = False) -> int:
    """The token count at which the forward row of ``channel_vectors`` vectors on route ``kind`` (with BF16 taps on
    ``fwd_bias`` when ``bf16_taps``) runs ``tiles`` on ``arch``, from the package's tables: the row's
    ``forward_narrow_min_s`` floor on its route and taps (``FORWARD_NARROW_ROW_FLOORS`` below 72 vectors, else
    ``FORWARD_NARROW_MIN_S`` or ``FORWARD_NARROW_ROW_MIN_S``: on sm_103a the ``fwd_bias`` route's C 768 from 20480 with
    FP32 taps and from 12288 with BF16 taps, its C 704 and the other routes' C 768 from 12288) for ``"narrow"`` (below
    the row's width floor), the row's ``FORWARD_WIDTH_FLOORS`` floor (at least the former) for ``"wide"``, one token
    below the floor for ``"below"``, the first token count of the row's ``FORWARD_SINGLE_TILE_ROWS`` range (sm_103a's
    on an architecture without one) for ``"gap"``.  A table that leaves the region empty on ``arch`` fails the
    test."""

    S_min = loader.forward_narrow_min_s(arch, channel_vectors, route=kind, bf16_taps=bf16_taps)
    assert S_min is not None, f"{arch} has no forward token floor for the row of {channel_vectors} vectors"
    floor = loader.FORWARD_WIDTH_FLOORS.get(arch, {}).get(channel_vectors)
    if tiles == "below":
        return int(S_min) - 1
    if tiles == "gap":
        gap = loader.forward_single_tile_range(arch, channel_vectors, route=kind, bf16_taps=bf16_taps)
        if gap is None:
            gap = loader.forward_single_tile_range("sm_103a", channel_vectors, route=kind, bf16_taps=bf16_taps)
        assert gap is not None, f"no FORWARD_SINGLE_TILE_ROWS range for the row of {channel_vectors} vectors"
        return int(gap[0])
    if tiles == "wide":
        assert channel_vectors in loader.FORWARD_NARROW_WIDTHS and floor is not None, (
            f"{arch}: the forward row of {channel_vectors} vectors has no width floor"
        )
        return max(int(floor), int(S_min))
    assert floor is None or int(S_min) < int(floor), (
        f"{arch}: the forward row of {channel_vectors} vectors runs no 32-vector tiles below its width floor"
    )
    return int(S_min)


# ----------------------------------------------------------------------------------------------------------------
# Accuracy against the naive tier.


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_forward_backward_match_reference(case) -> None:
    S, C, docs = case

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu)
        assert y.dtype == BF16 and tuple(y.shape) == tuple(x.shape) and y.is_contiguous()
        dx, dw, _ = _backward(x, w, cu, dy)
        assert dx.dtype == BF16 and dw.dtype == F32 and tuple(dw.shape) == (C, WIDTH)
        return {"y": y, **_grads(dx, dw, None)}

    _judge(S, C, docs, call, label=str(case))


@pytest.mark.parametrize("combo", DTYPE_COMBOS, ids=_combo_id)
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_routes_match_reference(case, combo) -> None:
    """Every backward route in every dtype combination (FP32 / BF16 taps, no / FP32 / BF16 bias, both SiLU'
    policies, FP32 / BF16 ``dweight`` and ``dbias`` outs): ``y``, ``dx``, ``dweight`` and ``dbias`` judged against
    the naive tier, the gradients written into the caller's outs."""

    weight_dtype, bias_dtype, policy, dweight_dtype, dbias_dtype = combo
    kind = _combo_route(weight_dtype, bias_dtype, policy, dweight_dtype)
    _require_route(kind)
    S, C, docs = case
    device = _device()
    label = f"{kind} {_combo_id(combo)} {case}"

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu, bias=b)
        out = {
            "dweight": torch.empty(C, WIDTH, dtype=dweight_dtype, device=device),
            "dbias": None
            if b is None
            else torch.empty(C, dtype=dbias_dtype, device=device),
        }
        dx, dw, db = _backward(x, w, cu, dy, b, policy, **out)
        assert dw is out["dweight"] and db is out["dbias"], label
        return {"y": y, **_grads(dx, dw, db)}

    _judge(
        S,
        C,
        docs,
        call,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        policy=policy,
        key=1,
        label=label,
    )


# BF16 taps without a bias (bwd_dt_bf16_taps) and with a BF16 bias (bwd_bias_bf16_taps) under the "bf16" policy, at
# keys whose draws put dx past the naive tier when SiLU' is evaluated at the BF16-rounded preactivation (C 512 runs
# other programs on sm_107a than on sm_100a / sm_103a; each architecture misses on some of these keys).
BF16_TAPS_KEYS = (
    ("bwd_dt_bf16_taps", None, 33),
    ("bwd_bias_bf16_taps", BF16, 11),
    ("bwd_bias_bf16_taps", BF16, 56),
)


@pytest.mark.parametrize(
    "kind, bias_dtype, key",
    BF16_TAPS_KEYS,
    ids=[f"{kind}-b{_dtype_id(bias)}-key{key}" for kind, bias, key in BF16_TAPS_KEYS],
)
def test_bf16_taps_routes_match_reference(kind, bias_dtype, key) -> None:
    """BF16 taps under the "bf16" policy at S = 1000, C = 512 ("perf" documents): ``y``, ``dx``, ``dweight`` and
    ``dbias`` judged against the naive tier."""

    _require_route(kind)

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu, bias=b)
        dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind)
        return {"y": y, **_grads(dx, dw, db)}

    _judge(1000, 512, "perf", call, weight_dtype=BF16, bias_dtype=bias_dtype, policy="bf16", key=key,
           label=f"{kind} BF16 taps key {key}")


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", WIDE_CASES, ids=_case_id)
def test_wide_channel_rows(case, kind) -> None:
    """C = 4096 and C = 10240 on every backward route (the fp32g routes included): ``y``, ``dx``, ``dweight`` and
    ``dbias`` judged against the naive tier."""

    _require_route(kind)
    S, C, docs = case

    def call(x, w, cu, dy, b, policy):
        return _forward_backward(x, w, cu, dy, b, policy, kind=kind)

    _route_judge(S, C, docs, kind, call, key=3, label=f"{kind} {case}")


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_backward_programs_switch_on_channel_count(kind, monkeypatch) -> None:
    """Exactly one program of the route runs per call and every program the route ships on this architecture runs
    for some ``(S, C)``: each is judged against the naive tier on both sides of its channel threshold (and of its
    token floor, where its rule has one) on a caller-owned workspace that reads back zero after every launch.  At a
    ``FALLBACK_ROWS`` key the route's fallback program runs at the key's last token count and the rules' program (or
    an ``SM_WIDE_ROWS`` segment's program) at the next; at an ``SM_WIDE_ROWS`` key each segment's program runs from
    the segment's first token count at the key's channel ends, and the program below the key's first segment one
    token before it."""

    _require_route(kind)
    arch, device = _arch(), _device()
    names = loader.route_modules(arch, kind)
    thresholds = _backward_thresholds(arch, kind)
    channels = sorted({C for T in thresholds for C in (T - 8, T, T + 8)} | {8192})
    rows = [
        (S, docs, C) for S, docs in ((5, (2, 0, 3)), (1000, "perf")) for C in channels
    ]
    # A token-floored program runs from its floor only: both sides of the floor at its channel threshold.
    for T, floor in _token_floors(arch, kind).values():
        rows += [(S, "perf", C) for S in (floor - 8, floor) for C in (T - 8, T, T + 8)]
    # The rows of ``FALLBACK_ROWS``: each key's channel ends (and C 640 / 896 inside a key) at the key's last token
    # count (the route's fallback program) and the next (the rules' program); C 640 / 896 also at S 1000.
    fallback = _fallback_rows(arch, kind)
    rows += [(S, "perf", C) for C, S_max in fallback for S in (S_max, S_max + 1)]
    rows += [(1000, "perf", C) for C, _ in fallback if C in FALLBACK_INNER_CHANNELS]
    # The rows of ``SM_WIDE_ROWS``: each segment's first token count (and one token below the first segment) at the
    # key's channel ends.
    for (route, c_lo, c_hi), segments in loader.SM_WIDE_ROWS.get(arch, {}).items():
        if route == kind:
            starts = [int(segments[0][0]) - 1, *(int(s_lo) for s_lo, _, _ in segments)]
            rows += [(S, "perf", C) for S in starts for C in (int(c_lo), int(c_hi))]
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(name)
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    launched = set()
    fallback_program = _fallback_program(arch, kind)
    for S, docs, C in rows:
        # inside an SM-wide segment its program, inside a fallback key's token range the route's fallback program,
        # elsewhere the rules' program
        selected = _expected_program(arch, kind, S, C)
        bounds = loader.fallback_row_range(arch, kind, C)
        if bounds is not None and bounds[0] <= S <= bounds[1]:
            assert selected == fallback_program, (kind, S, C, selected)
        label = f"{kind} S={S} C={C} docs={docs} program={selected}"
        workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)

        def call(x, w, cu, dy, b, policy, workspace=workspace, selected=selected, label=label):
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            assert seen == [selected], f"{label}: launched {seen}"
            assert int(torch.count_nonzero(workspace)) == 0, label
            return _grads(dx, dw, db)

        _route_judge(S, C, docs, kind, call, key=6, label=label)
        launched.add(selected)
    assert launched == set(names), f"never launched {set(names) - launched}"


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_large_case_over_two_to_the_31_elements(kind) -> None:
    """``S x C > 2^31`` elements (64-bit offsets) at ``C`` 8192: ``y``, ``dx``, ``dweight`` and ``dbias`` judged
    against the naive tier, the workspace left zero."""

    _require_route(kind)
    free, _total = torch.cuda.mem_get_info(_device())
    if free < LARGE_CASE_FREE_BYTES:
        pytest.skip(
            f"needs {LARGE_CASE_FREE_BYTES >> 30} GiB free, {free >> 30} GiB available"
        )
    S, C, docs = LARGE_CASE
    assert S * C > 2**31
    device = _device()
    label = f"{kind} {_case_id(LARGE_CASE)}"
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)

    def call(x, w, cu, dy, b, policy):
        y = _forward(x, w, cu, bias=b)
        dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
        assert int(torch.count_nonzero(workspace)) == 0, label
        return {"y": y, **_grads(dx, dw, db)}

    _route_judge(S, C, docs, kind, call, key=5, label=label)
    torch.cuda.empty_cache()


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", REPEAT_CASES, ids=_case_id)
def test_repeated_launches_worst_case(case, kind, record_property) -> None:
    """``REPEATS`` launches on each draw's inputs into one workspace: every launch's ``dx``, ``dweight`` and
    ``dbias`` judged against the naive tier (the worst launch of a draw counts), the workspace zero after every
    launch; the number of distinct ``dweight`` bit patterns of the first draw is recorded (the order of the
    cross-CTA sums may vary; no bound is placed on that count)."""

    _repeated_launches(case, kind, record_property)


def test_fallback_row_repeated_launches(monkeypatch, record_property) -> None:
    """The repeated-launch rows at the default backward route's ``FALLBACK_ROWS`` channel counts (C 768, C 1016 at
    many CTAs per channel tile, inside the keys' token ranges): each row launches the expected program (the route's
    fallback program on sm_100a / sm_103a, the architecture's own plan elsewhere), every launch judged against the
    naive tier and the workspace zero after each."""

    _require_route("bwd")
    seen = _record_programs(monkeypatch)
    for case in _fallback_row_large_cases():
        seen.clear()
        _repeated_launches(case, "bwd", record_property)
        assert _backward_programs(seen) == {_expected_backward_program(case[0], case[1])}, (case, set(seen))


def _repeated_launches(case, kind: str, record_property) -> None:
    """The body of :func:`test_repeated_launches_worst_case` at ``case`` ``(S, C, docs)``."""

    _require_route(kind)
    S, C, docs = case
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, _device())
    patterns = set()
    draws = []

    def call(x, w, cu, dy, b, policy):
        found = {"dx": [], "dweight": [], "dbias": []}
        for repeat in range(REPEATS):
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            assert int(torch.count_nonzero(workspace)) == 0, (
                f"the workspace is not zero after repeat {repeat}"
            )
            for name, value in _grads(dx, dw, db).items():
                found[name].append(value)
            if not draws:
                patterns.add(bytes(dw.contiguous().view(torch.uint8).cpu().numpy()))
        draws.append(len(draws))
        return {name: values for name, values in found.items() if values}

    _route_judge(S, C, docs, kind, call, key=3, label=f"{kind} {case}")
    print(f"{kind} S={S} C={C}: {REPEATS} launches per draw, {len(patterns)} distinct dweight bit patterns")
    record_property("repeats", REPEATS)
    record_property("distinct_dweight_patterns", len(patterns))


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
def test_tile_override_row(own_workspace, monkeypatch, record_property) -> None:
    """Every ``OVERRIDE_CASES`` row of the architecture's ``TILE_OVERRIDES`` launches the row's narrower channel tiles
    (``ct_vec``, ``threads // ct_vec`` sub-tiles, ``ceil(C / vec / ct_vec)`` tiles) on a caller-owned workspace of
    the public size, or the package's: ``dx``, ``dweight`` and ``dbias`` judged against the naive tier, the
    workspace zero after every call."""

    arch, device = _arch(), _device()
    rows = OVERRIDE_CASES[arch]
    for kind, _, _ in rows:
        _require_route(kind)
    seen = []
    launch = pkg.run_module
    split_keys = ("ct_vec", "n_sub", "n_ch_tiles")

    def record_launch(name, values, *, device_index, grid=None):
        seen.append((name, {key: int(values[key]) for key in split_keys}))
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for kind, S, C in rows:
        ct_vec = loader.tile_override(arch, kind, C, S)
        assert ct_vec is not None, f"{arch} {kind} C={C} S={S} is no TILE_OVERRIDES row"
        (name,) = loader.selected_modules(arch, kind, C=C, S=S)
        geometry = loader.MODULES[name]["grid_rule"]["geometry"]
        wanted = {
            "ct_vec": ct_vec,
            "n_sub": geometry["threads"] // ct_vec,
            "n_ch_tiles": -(-(C // geometry["vec"]) // ct_vec),
        }
        assert wanted != dict(
            zip(
                split_keys,
                loader.channel_split(C, geometry=geometry, role="bwd"),
                strict=True,
            )
        ), "the row must differ from the default split"
        label = f"{arch} {kind} S={S} C={C} ct_vec={ct_vec}"
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own_workspace
            else None
        )

        def call(x, w, cu, dy, b, policy, kind=kind, name=name, wanted=wanted, label=label, workspace=workspace):
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            assert seen == [(name, wanted)], f"{label}: launched {seen}"
            held = workspace if own_workspace else _stream_workspace(device)
            assert held is not None and int(torch.count_nonzero(held)) == 0, label
            return _grads(dx, dw, db)

        _route_judge(S, C, "perf", kind, call, key=9, label=label)
        record_property("override_ct_vec", ct_vec)


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
def test_narrow_tile_row(own_workspace, monkeypatch, record_property) -> None:
    """Every ``NARROW_CASES`` row of the architecture's ``NARROW_TILES`` launches 32-vector channel tiles (``32``,
    ``threads // 32`` sub-tiles, ``ceil(C / vec / 32)`` tiles) instead of :func:`channel_split`'s single tile, on
    the grid the host plans for the call (``_call_plan``: ``launch_grid`` with the side-stream trim), on a
    caller-owned workspace of the public size or the package's: ``dx``, ``dweight`` and ``dbias`` judged against the
    naive tier, the workspace zero after every call."""

    arch, device = _arch(), _device()
    cases = NARROW_CASES.get(arch, ())
    if not cases:
        pytest.skip(f"{arch} has no NARROW_TILES row")
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(
            (
                name,
                {key: int(values[key]) for key in ("ct_vec", "n_sub", "n_ch_tiles")},
                grid,
            )
        )
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for kind, S, C in cases:
        _require_route(kind)
        label = f"{arch} {kind} S={S} C={C}"
        assert loader.narrow_tile_row(arch, kind, C, S), label
        assert loader.sm_wide_module(arch, kind, C, S) is None, label
        (name,) = loader.selected_modules(arch, kind, C=C, S=S)
        geometry = loader.MODULES[name]["grid_rule"]["geometry"]
        wanted = {
            "ct_vec": 32,
            "n_sub": geometry["threads"] // 32,
            "n_ch_tiles": -(-(C // geometry["vec"]) // 32),
        }
        assert wanted != dict(
            zip(
                ("ct_vec", "n_sub", "n_ch_tiles"),
                loader.channel_split(C, geometry=geometry, role="bwd"),
                strict=True,
            )
        ), f"{label}: the row must differ from the default split"
        _, _, _, grid, _ = pkg._call_plan(
            arch,
            kind,
            S,
            C,
            len(doc_lengths(S, "perf")),
            device_index=device.index,
            policy=ROUTE_SETTINGS[kind]["policy"],
        )
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own_workspace
            else None
        )

        def call(x, w, cu, dy, b, policy, name=name, wanted=wanted, grid=grid, workspace=workspace, label=label):
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
            torch.cuda.synchronize(device)
            assert seen == [(name, wanted, grid)], f"{label}: launched {seen}"
            held = workspace if own_workspace else _stream_workspace(device)
            assert held is not None and int(torch.count_nonzero(held)) == 0, label
            return _grads(dx, dw, db)

        _route_judge(S, C, "perf", kind, call, key=11, label=label)
    record_property("narrow_rows", len(cases))


def test_narrow_tile_row_gives_up_one_sm_at_three_tiles(monkeypatch) -> None:
    """A three-tile ``NARROW_TILES`` row: on the ``bwd`` and ``bwd_fp32g`` routes a
    single-wave launch of ``ctas_per_sm x SM count`` CTAs whose last channel-tile rank holds at least
    ``ctas_per_sm`` CTAs (``full % 3 >= ctas_per_sm``; B200: 296 % 3 = 2) launches ``ctas_per_sm`` fewer, as at a
    route-free plan of three tiles; a grid ``launch_grid`` already trimmed to whole ranks is kept.  The trimmed
    launch's outputs and those of the whole grid are judged against the naive tier, and the workspace reads back
    zero."""

    arch, device = _arch(), _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rows = [
        (kind, S, C)
        for kind, S, C in NARROW_CASES.get(arch, ())
        if kind in ("bwd", "bwd_fp32g")
    ]
    if not rows:
        pytest.skip(f"{arch} has no NARROW_TILES row on the bwd / bwd_fp32g routes")
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((int(values["n_ch_tiles"]), grid))
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    checked = 0
    for kind, S, C in rows:
        _require_route(kind)
        (name,) = loader.selected_modules(arch, kind, C=C, S=S)
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        geometry = rule["geometry"]
        tiles = -(-(C // geometry["vec"]) // 32)
        if (
            tiles != 3
            or rule["kind"] != "occupancy"
            or loader.grid_mult(record, arch) != 1
        ):
            continue
        k = int(rule["ctas_per_sm"])
        full = k * sms
        untrimmed = loader.launch_grid(record, device_index=device.index, C=C, S=S)
        cut = untrimmed[0] == full and full % tiles >= k
        expected = (full - k, 1, 1) if cut else tuple(untrimmed)
        label = f"{arch} {kind} S={S} C={C} full={full} k={k}"

        def call(x, w, cu, dy, b, policy, tiles=tiles, expected=expected, cut=cut, untrimmed=untrimmed, label=label):
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
            launched.clear()
            runs = [_backward(x, w, cu, dy, b, policy, kind=kind)]
            torch.cuda.synchronize(device)
            assert launched == [(tiles, expected)], f"{label}: launched {launched}"
            held = _stream_workspace(device)
            assert held is None or int(torch.count_nonzero(held)) == 0, label
            if cut:
                with monkeypatch.context() as m:
                    m.setattr(pkg, "_one_sm_trim", lambda record, n, grid, **kw: grid)
                    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                    launched.clear()
                    runs.append(_backward(x, w, cu, dy, b, policy, kind=kind))
                    torch.cuda.synchronize(device)
                pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                assert launched == [(tiles, tuple(untrimmed))], f"{label}: {launched}"
            found = {}
            for run in runs:
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, "perf", kind, call, key=43, label=label)
        checked += 1
    assert checked, (
        f"{arch}: no three-tile NARROW_TILES row on an occupancy single-wave record"
    )


def _forward_region(arch: str, kind: str, channel_vectors: int, S: int, bf16_taps: bool) -> str:
    """The tiles (``"one"``, ``"narrow"`` or ``"wide"``) the forward row of ``channel_vectors`` vectors on route
    ``kind`` runs at ``S`` tokens on ``arch``, read from the tables: below the row's floor and inside its single-tile
    range the single tile, from its width floor its width, else the 32-vector tiles."""

    S_min = loader.forward_narrow_min_s(arch, channel_vectors, route=kind, bf16_taps=bf16_taps)
    gap = loader.forward_single_tile_range(arch, channel_vectors, route=kind, bf16_taps=bf16_taps)
    if S_min is None or S < S_min or (gap is not None and gap[0] <= S <= gap[1]):
        return "one"
    floor = loader.FORWARD_WIDTH_FLOORS.get(arch, {}).get(channel_vectors)
    if channel_vectors in loader.FORWARD_NARROW_WIDTHS and floor is not None and S >= floor:
        return "wide"
    return "narrow"


@pytest.mark.parametrize(
    "kind, C, tiles, taps",
    FORWARD_NARROW_CASES,
    ids=[f"{kind}-C{C}-{tiles}-{taps}" for kind, C, tiles, taps in FORWARD_NARROW_CASES],
)
def test_forward_narrow_row(kind, C, tiles, taps, monkeypatch, record_property) -> None:
    """Every ``FORWARD_NARROW_CASES`` row at the token count of :func:`_forward_narrow_tokens` on the device's
    architecture: the public forward launches the route's dynamic program with the region's channel tiles -- the
    32-vector tiles (``32``, ``threads // 32`` sub-tiles, ``ceil(C / vec / 32)`` tiles) at the row's
    ``forward_narrow_min_s`` floor, the row's ``FORWARD_NARROW_WIDTHS`` tiles at its width floor, one tile of ``C /
    vec`` vectors one token below the floor and inside a single-tile range -- once on a caller-owned forward workspace
    of the public size and once on the package's, each holding the launch's counters and zero after the call; ``y``
    of both launches judged against the naive tier.  The forward keys of ``FALLBACK_ROWS`` and ``DIRECT_ROWS`` are
    removed for the test (:func:`_forward_rules_only`): these are the dynamic program's rows, which it runs above the
    keys' token ceilings (the keys' own programs: :func:`test_forward_small_s_row`)."""

    _require_route(kind)
    with _forward_rules_only(_arch()):
        _forward_narrow_row(kind, C, tiles, taps, monkeypatch, record_property)


def _forward_narrow_row(kind, C, tiles, taps, monkeypatch, record_property) -> None:
    """The body of :func:`test_forward_narrow_row`."""

    arch, device = _arch(), _device()
    weight_dtype, bias_dtype = FORWARD_SETTINGS[(kind, taps)]
    assert loader.forward_route(bias=bias_dtype is not None, bf16_taps=weight_dtype == BF16) == kind
    bf16_taps = weight_dtype == BF16
    # The forward programs run 8 channels per thread (checked below on the selected program's geometry).
    S = _forward_narrow_tokens(arch, kind, C // 8, tiles, bf16_taps)
    (name,) = loader.selected_modules(arch, kind, C=C, S=S)
    record = loader.MODULES[name]
    geometry = record["grid_rule"]["geometry"]
    threads, vec = int(geometry["threads"]), int(geometry["vec"])
    assert (threads, vec) == (loader.FORWARD_NARROW_THREADS, 8), f"{name}: geometry {geometry}"
    region = _forward_region(arch, kind, C // vec, S, bf16_taps)
    if tiles in ("narrow", "wide"):
        assert region == tiles, f"{arch} {kind} C={C} S={S} taps={taps}: the tables plan {region}"
    elif tiles == "below" or loader.forward_single_tile_range(arch, C // vec, route=kind, bf16_taps=bf16_taps):
        assert region == "one", f"{arch} {kind} C={C} S={S} taps={taps}: the tables plan {region}"
    single = loader.channel_split(C, geometry=geometry, role="fwd")
    if region == "one":
        ct_vec = single[0]
    else:
        ct_vec = loader.FORWARD_NARROW_WIDTHS[C // vec] if region == "wide" else 32
    wanted = {"ct_vec": ct_vec, "n_sub": threads // ct_vec, "n_ch_tiles": -(-(C // vec) // ct_vec)}
    split_keys = tuple(wanted)
    assert (tuple(wanted.values()) == single) is (region == "one"), (
        f"{arch} {kind} C={C}: a narrow row must differ from the single tile"
    )
    words = loader.counter_words(record, wanted["n_ch_tiles"])
    label = f"{arch} {kind} S={S} C={C} {tiles} taps={taps} split={tuple(wanted.values())}"
    workspace = pkg.cake_causal_conv1d_nondet_train_forward_workspace(S, C, device)
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append((name, {key: int(values[key]) for key in split_keys}, values.get("work")))
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)

    def call(x, w, cu, dy, b, policy):
        ys = []
        for own in (True, False):
            seen.clear()
            y = _forward(x, w, cu, bias=b, workspace=workspace if own else None)
            assert [(launched, split) for launched, split, _ in seen] == [(name, wanted)], (
                f"{label} own={own}: launched {seen}"
            )
            held = workspace if own else _forward_stream_workspace(device)
            assert held is not None and seen[0][2] is held and held.numel() >= words, f"{label} own={own}"
            assert int(torch.count_nonzero(held)) == 0, f"{label} own={own}: the workspace is not zero"
            ys.append(y)
        return {"y": ys}

    _judge(S, C, "perf", call, weight_dtype=weight_dtype, bias_dtype=bias_dtype, key=47, label=label)
    record_property("forward_split", tuple(wanted.values()))
    record_property("forward_tokens", S)


def _small_s_rows(arch: str, kind: str) -> list[tuple[int, int, int, int]]:
    """``(S, C, C_lo, C_hi)`` of the forward keys of route ``kind`` on ``arch`` (:func:`_forward_keys`): at the key's
    lowest channel count S 1000 (inside the key's range), the range's first token count and the one below it; at its
    highest channel count the range's last token count and the one above it (so ``S_x`` and ``S_x + 1`` of a key
    with both ranges, and ``S_hi`` and ``S_hi + 1``)."""

    rows = set()
    for route, c_lo, c_hi, s_lo, s_hi, _ in _forward_keys(arch):
        if route != kind:
            continue
        lower = (s_lo - 1, s_lo, *((1000,) if s_lo <= 1000 <= s_hi else ()))
        rows |= {(S, c_lo, c_lo, c_hi) for S in lower if S >= 1}
        rows |= {(S, c_hi, c_lo, c_hi) for S in (s_hi, s_hi + 1)}
    return sorted(rows)


@pytest.mark.parametrize(
    "kind, taps", tuple(FORWARD_SETTINGS), ids=[f"{k}-{t}" for k, t in FORWARD_SETTINGS]
)
def test_forward_small_s_row(kind, taps, monkeypatch, record_property) -> None:
    """Every forward key of ``FALLBACK_ROWS`` and ``DIRECT_ROWS`` of route ``kind`` on the device's architecture
    (:func:`_small_s_rows`: S 1000, both sides of the static / direct split and of the key's ceiling, at the key's
    channel ends): the public forward launches the program the tables name (the static program inside a static
    range, the direct program inside a direct range, the dynamic program above), the static and direct programs with
    no ``work`` counters (a passed forward workspace is accepted, not read, and stays zero).  On each of the 32 draws
    every ``y`` is judged against the naive tier: 3 launches with and without a caller workspace, the route's static
    program on the same inputs (one key over every token count of the key's channel counts), a launch with
    ``enable_pdl`` 0 and one with 1, two launches on two streams at once, and the replays of a CUDA Graph captured
    after one eager warm-up, whose capture at a static or direct row has no fill node and counts no capture
    fallback."""

    _require_route(kind)
    arch, device = _arch(), _device()
    weight_dtype, bias_dtype = FORWARD_SETTINGS[(kind, taps)]
    assert (
        loader.forward_route(
            bias=bias_dtype is not None, bf16_taps=weight_dtype == BF16
        )
        == kind
    )
    static, direct = (
        _forward_program(arch, kind, "static"),
        _forward_program(arch, kind, "direct"),
    )
    rows = _small_s_rows(arch, kind)
    assert static is not None and rows, f"{arch} {kind}: no forward key"
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append((name, values.get("work"), values.get("enable_pdl")))
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    forms = set()
    for S, C, c_lo, c_hi in rows:
        name = _expected_program(arch, kind, S, C)
        form = "direct" if name == direct else "static" if name == static else "dynamic"
        forms.add(form)
        no_work = form != "dynamic"
        label = f"{arch} {kind} taps={taps} S={S} C={C} {form} {name}"
        workspace = pkg.cake_causal_conv1d_nondet_train_forward_workspace(S, C, device)

        def call(
            x,
            w,
            cu,
            dy,
            b,
            policy,
            name=name,
            no_work=no_work,
            label=label,
            workspace=workspace,
            c=(c_lo, c_hi),
        ):
            ys = []
            for repeat in range(3):
                for own in (False, True):
                    seen.clear()
                    ys.append(
                        _forward(x, w, cu, bias=b, workspace=workspace if own else None)
                    )
                    assert [n for n, _, _ in seen] == [name], (
                        f"{label} own={own}: launched {seen}"
                    )
                    assert (seen[0][1] is None) is no_work, (
                        f"{label} own={own}: work {seen[0][1] is not None}"
                    )
                    assert int(torch.count_nonzero(workspace)) == 0, (
                        f"{label}: the forward workspace is not zero"
                    )
            with _forward_rules_only(arch, kind, c):
                seen.clear()
                ys.append(_forward(x, w, cu, bias=b))
                assert [n for n, _, _ in seen] == [static], (
                    f"{label}: the static run launched {seen}"
                )
            shipped = loader.PDL_EAGER[arch]
            for pdl in (False, True):
                monkeypatch.setitem(loader.PDL_EAGER, arch, pdl)
                seen.clear()
                ys.append(_forward(x, w, cu, bias=b))
                assert seen[0][2] == int(pdl), (
                    f"{label}: enable_pdl {seen[0][2]} with PDL_EAGER {pdl}"
                )
            monkeypatch.setitem(loader.PDL_EAGER, arch, shipped)
            pair = [None, None]
            torch.cuda.synchronize(device)
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    pair[i] = _forward(x, w, cu, bias=b)
            torch.cuda.synchronize(device)
            return {"y": ys + pair}

        _judge(
            S,
            C,
            "perf",
            call,
            weight_dtype=weight_dtype,
            bias_dtype=bias_dtype,
            key=59,
            label=label,
        )
        # CUDA Graph capture after one eager warm-up, its replays on each draw's inputs judged
        x, w, cu, _ = _inputs(S, C, "perf", key=59, weight_dtype=weight_dtype)
        b = None if bias_dtype is None else _bias_input(C, bias_dtype, key=59)
        out = torch.empty_like(x)
        cap = _capture(
            lambda x=x, w=w, cu=cu, b=b, out=out: _forward(x, w, cu, bias=b, out=out),
            streams[0],
        )
        nodes = _graph_nodes(cap["graph"])
        record_property(f"fill_nodes S={S} C={C} {form}", nodes["fill"])
        if no_work:
            assert nodes["fill"] == 0 and cap["fallbacks"]["fwd"] == 0, (
                label,
                nodes,
                cap["fallbacks"],
            )
            assert "fwd" not in cap["buffers"], (label, cap["buffers"])

        def replay(_draw, seed, x=x, w=w, b=b, out=out, graph=cap["graph"], S=S, C=C):
            fx, fw, fcu, fdy = _inputs(
                S, C, "perf", key=59, weight_dtype=weight_dtype, seed=seed
            )
            fb = (
                None
                if bias_dtype is None
                else _bias_input(C, bias_dtype, key=59, seed=seed)
            )
            x.copy_(fx)
            w.copy_(fw)
            if b is not None:
                b.copy_(fb)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            return {"y": out}, problem(fx, fw, fcu, fb, fdy)

        assert_judged(replay, seed=SEED, label=f"{label} graph")
        del cap
    assert {"static", "dynamic"} <= forms and ("direct" in forms) is (
        direct is not None
    ), (arch, kind, forms)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
@pytest.mark.parametrize("case", GRID_MULT_CASES, ids=_case_id)
def test_bwd_grid_mult_outputs_and_workspace(case, kind) -> None:
    """Every ``bwd_grid_mult`` on every route, on a caller-owned workspace (allocated at that multiple) and on the
    package's stream workspace: every launch's ``dx``, ``dweight`` and ``dbias`` judged against the naive tier (the
    worst launch of a draw counts), and the workspace read back all zero after every launch."""

    _require_route(kind)
    S, C, docs = case
    device = _device()
    workspaces = {
        mult: pkg.cake_causal_conv1d_nondet_train_backward_workspace(
            S, C, device, bwd_grid_mult=mult
        )
        for mult in pkg.BWD_GRID_MULTS
    }

    def call(x, w, cu, dy, b, policy):
        found = {}
        for mult in pkg.BWD_GRID_MULTS:
            for own in (True, False):
                kwargs = {"workspace": workspaces[mult]} if own else {}
                run = _backward(x, w, cu, dy, b, policy, kind=kind, bwd_grid_mult=mult, **kwargs)
                label = f"{kind} S={S} C={C} bwd_grid_mult={mult} own={own}"
                held = workspaces[mult] if own else _stream_workspace(device)
                assert held is not None and int(torch.count_nonzero(held)) == 0, (
                    f"workspace not zero ({label})"
                )
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
        return found

    _route_judge(S, C, docs, kind, call, key=31, label=f"{kind} {case}")


@pytest.mark.parametrize("case", PRECISION_CASES, ids=_case_id)
def test_fp32_grad_perf_size_precision(case) -> None:
    """The fp32 policy at the perf size (S = 250340): ``PRECISION_REPEATS`` launches per draw, every launch's ``dx``
    and ``dweight`` judged against the naive tier (the worst launch of a draw counts)."""

    _require_route("bwd_fp32g")
    S, C, docs = case

    def call(x, w, cu, dy, b, policy):
        found = {}
        for _ in range(PRECISION_REPEATS):
            for out, value in _grads(*_backward(x, w, cu, dy, None, "fp32")).items():
                found.setdefault(out, []).append(value)
        return found

    _judge(S, C, docs, call, policy="fp32", label=f"fp32 {case}")
    torch.cuda.empty_cache()


def test_document_isolation() -> None:
    """Perturbing one document (the 3-token document, rows 1000..1002, drawn again) leaves every document's outputs
    judged against the naive tier of the changed problem."""

    S, C, docs = 4097, 512, (1000, 0, 3, 3094)

    def perturb(x, dy):
        x = x.clone()
        x[1000:1003] = torch.randn(3, C, device=_device()).to(BF16)
        return x, dy

    _judge(S, C, docs, _forward_backward, key=5, perturb=perturb, label="perturbed document")


def test_batched_layout() -> None:
    """``[1, S, C]`` inputs give ``[1, S, C]`` outputs, judged against the naive tier."""

    S, C, docs = 1000, 512, "perf"

    def call(x, w, cu, dy, b, policy):
        y3 = _forward(x.unsqueeze(0), w, cu)
        dx3, dw3, _ = _backward(x.unsqueeze(0), w, cu, dy.unsqueeze(0))
        assert tuple(y3.shape) == (1, S, C) and tuple(dx3.shape) == (1, S, C)
        return {"y": y3, **_grads(dx3, dw3, None)}

    _judge(S, C, docs, call, label="[1, S, C]")


# ----------------------------------------------------------------------------------------------------------------
# The workspace contract.


def test_workspace_bytes_depend_on_channels_only() -> None:
    device = _device()
    arch = _arch()
    for C in (128, 512, 1160, 1280, 1344, 4096, 8192):
        nbytes = pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
            1, C, device
        )
        assert nbytes == pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
            10**6, C, device
        )
        assert nbytes == loader.backward_workspace_bytes(arch, C)
        assert nbytes % loader.WORKSPACE_ALIGN == 0
        # One accumulator set and one counter per channel tile, whatever the grid: every bwd_grid_mult, one size.
        for mult in pkg.BWD_GRID_MULTS:
            assert (
                nbytes
                == pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
                    1, C, device, bwd_grid_mult=mult
                )
            )
        # At least one FP32 word per tap and channel (the bias routes' owners hold five per channel).
        assert nbytes >= 4 * WIDTH * C
        ws = pkg.cake_causal_conv1d_nondet_train_backward_workspace(1, C, device)
        assert ws.dtype == torch.uint8 and ws.numel() == nbytes
        assert int(torch.count_nonzero(ws)) == 0


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_changed_shapes_reuse_one_zero_workspace(kind) -> None:
    """One caller-owned workspace, sized for the largest ``C``, serves a sequence of shapes of the route (and the
    package's own stream workspace serves the same sequence): every result is judged against the naive tier and the
    buffer reads back all zero after every launch."""

    _require_route(kind)
    device = _device()
    shapes = ((4097, 1280, (1000, 0, 3, 3094)), (65, 256, (64, 1)), (1000, 1344, "perf"),
              (5, 136, (2, 0, 3)), (4097, 1160, (1000, 0, 3, 3094)), (1000, 512, "perf"))  # fmt: skip
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(
        1, max(C for _, C, _ in shapes), device
    )
    for own in (True, False):
        for S, C, docs in shapes:
            label = f"{kind} S={S} C={C} own={own}"

            def call(x, w, cu, dy, b, policy, own=own, label=label):
                kwargs = {"workspace": workspace} if own else {}
                run = _backward(x, w, cu, dy, b, policy, kind=kind, **kwargs)
                held = workspace if own else _stream_workspace(device)
                assert held is not None
                assert int(torch.count_nonzero(held)) == 0, (
                    f"workspace not zero after {label}"
                )
                return _grads(*run)

            _route_judge(S, C, docs, kind, call, key=6, label=label)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_nan_poisoned_outputs_are_overwritten(kind) -> None:
    _require_route(kind)
    device = _device()
    for S, C, docs in ROUTE_CASES:

        def call(x, w, cu, dy, b, policy, C=C):
            dx = torch.full_like(x, float("nan"))
            dweight_dtype = ROUTE_SETTINGS[kind]["dweight_dtype"] or w.dtype
            dw = torch.full((C, WIDTH), float("nan"), dtype=dweight_dtype, device=device)
            db = (
                None
                if b is None
                else torch.full((C,), float("nan"), dtype=b.dtype, device=device)
            )
            y = torch.full_like(x, float("nan"))
            _forward(x, w, cu, bias=b, out=y)
            got = _backward(x, w, cu, dy, b, policy, kind=kind, dx=dx, dweight=dw, dbias=db)
            assert got[0] is dx and got[1] is dw and got[2] is db
            return {"y": y, **_grads(dx, dw, db)}

        _route_judge(S, C, docs, kind, call, key=7, label=f"NaN-poisoned {kind} S={S} C={C}")


def test_workspace_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    nbytes = pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(
        S, C, _device()
    )
    with pytest.raises(ValueError, match="workspace holds"):
        _backward(
            x, w, cu, dy, workspace=torch.zeros(16, dtype=torch.uint8, device=_device())
        )
    with pytest.raises(ValueError, match="uint8"):
        _backward(
            x,
            w,
            cu,
            dy,
            workspace=torch.zeros(nbytes // 4, dtype=torch.int32, device=_device()),
        )
    with pytest.raises(ValueError, match="aligned"):
        big = torch.zeros(nbytes + 16, dtype=torch.uint8, device=_device())
        _backward(x, w, cu, dy, workspace=big[1 : nbytes + 1])
    with pytest.raises(ValueError, match="1-D"):
        _backward(
            x,
            w,
            cu,
            dy,
            workspace=torch.zeros(1, nbytes, dtype=torch.uint8, device=_device()),
        )


def _held_views(workspace: torch.Tensor) -> dict | None:
    """The package's cached ``{layout key: (data_ptr, acc, work)}`` of ``workspace`` (``None`` without an entry)."""

    hit = pkg._ACCUMULATOR_VIEWS.get(id(workspace))
    return hit[1] if hit is not None and hit[0]() is workspace else None


def test_workspace_views_are_built_once_per_workspace() -> None:
    """The ``acc`` / ``work`` tensors of a workspace are built once per layout and reused by later calls; they
    alias its storage but not the tensor, follow the storage when it moves, and are dropped with the
    workspace; an entry left under the workspace's ``id`` by a dead workspace is never served.  The calls' outputs
    are judged against the naive tier."""

    S, C, docs = 1000, 1280, "perf"
    device = _device()
    nbytes = pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(S, C, device)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    # Entries of earlier tests' workspaces that are already garbage (a failed test's traceback holds its workspace
    # in a reference cycle) are dropped here, not by the collection at the end of this test.
    gc.collect()
    entries = len(pkg._ACCUMULATOR_VIEWS)

    def calls(count, ws):
        def call(x, w, cu, dy, b, policy):
            found = {}
            for _ in range(count):
                for out, value in _grads(*_backward(x, w, cu, dy, workspace=ws)).items():
                    found.setdefault(out, []).append(value)
            return found

        return call

    _judge(S, C, docs, calls(2, workspace), label="two calls")
    held = _held_views(workspace)
    assert held is not None and len(held) == 1
    ((ptr, acc, work),) = held.values()
    assert ptr == workspace.data_ptr()
    assert acc._base is None and work._base is None
    storage = workspace.untyped_storage().data_ptr()
    assert acc.untyped_storage().data_ptr() == storage
    assert work.untyped_storage().data_ptr() == storage
    moved = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    workspace.set_(moved.untyped_storage(), 0, (nbytes,), (1,))
    del moved
    _judge(S, C, docs, calls(1, workspace), label="moved storage")
    ((ptr, acc, work),) = _held_views(workspace).values()
    assert ptr == workspace.data_ptr()
    assert acc.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert int(torch.count_nonzero(workspace)) == 0
    # The entry of a dead workspace under this workspace's id (an id reused before the entry was dropped), with
    # views into other buffers: the call must not serve it but rebuild the views on this workspace.
    dead = torch.empty(0)
    ((key, _),) = held.items()
    pkg._ACCUMULATOR_VIEWS[id(workspace)] = (
        weakref.ref(dead),
        {key: (workspace.data_ptr(), torch.zeros_like(acc), torch.zeros_like(work))},
    )
    del dead
    assert pkg._ACCUMULATOR_VIEWS[id(workspace)][0]() is None
    _judge(S, C, docs, calls(1, workspace), label="dead entry under the id")
    ((ptr, acc, work),) = _held_views(workspace).values()
    assert ptr == workspace.data_ptr()
    assert acc.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert work.untyped_storage().data_ptr() == workspace.untyped_storage().data_ptr()
    assert int(torch.count_nonzero(workspace)) == 0
    del workspace, held, acc, work
    gc.collect()
    assert len(pkg._ACCUMULATOR_VIEWS) == entries


# ----------------------------------------------------------------------------------------------------------------
# Graphs, streams, launches.


def _capture_and_replay(
    cases, kind: str, *, key: int, own_workspace: bool, bwd_grid_mult: int = 1
) -> int:
    """One forward + backward graph per shape (the backward at ``bwd_grid_mult``), replayed once per draw on that
    draw's inputs (copied into the captured input tensors) into NaN-filled outputs: every replay's ``y``, ``dx``,
    ``dweight`` and ``dbias`` judged against the naive tier, and a caller-owned workspace and the package's capture
    buffers of the capture read back all zero after every replay.  Returns the replays."""

    device = _device()
    captured = []
    for S, C, docs in cases:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=key)
        y = torch.empty_like(x)
        dx = torch.empty_like(x)
        dw = torch.empty(
            C, WIDTH, dtype=ROUTE_SETTINGS[kind]["dweight_dtype"] or w.dtype, device=device
        )
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        workspace = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(
                S, C, device, bwd_grid_mult=bwd_grid_mult
            )
            if own_workspace
            else None
        )

        def step(
            x=x,
            w=w,
            cu=cu,
            dy=dy,
            b=b,
            policy=policy,
            y=y,
            dx=dx,
            dw=dw,
            db=db,
            ws=workspace,
        ):
            _forward(x, w, cu, bias=b, out=y)
            _backward(
                x,
                w,
                cu,
                dy,
                b,
                policy,
                kind=kind,
                workspace=ws,
                dx=dx,
                dweight=dw,
                dbias=db,
                bwd_grid_mult=bwd_grid_mult,
            )

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        entries = _capture_entries(device, side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            step()
        torch.cuda.synchronize(device)
        # the package's capture buffers of this capture (the stream's new entries)
        package = [
            entry[1]
            for kind_of, entry in _capture_entries(device, side).items()
            if entry is not None and entry is not entries[kind_of]
        ]
        captured.append(((S, C, docs), graph, (x, w, cu, dy, b), (y, dx, dw, db), workspace, package))
    replays = 0
    for (S, C, docs), graph, inputs, (y, dx, dw, db), workspace, package in captured:

        def replay(
            xn, wn, cu, dyn, bn, policy, graph=graph, inputs=inputs, outs=(y, dx, dw, db), workspace=workspace,
            package=package,
        ):  # fmt: skip
            nonlocal replays
            for held, value in zip(inputs, (xn, wn, cu, dyn, bn), strict=True):
                if held is not None:
                    held.copy_(value)
            for t in outs:
                if t is not None:
                    t.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            replays += 1
            if workspace is not None:
                assert int(torch.count_nonzero(workspace)) == 0
            for t in package:
                assert int(torch.count_nonzero(t)) == 0, "a package capture buffer is not zero after a replay"
            return {"y": outs[0], **_grads(*outs[1:])}

        _route_judge(
            S,
            C,
            docs,
            kind,
            replay,
            key=key,
            label=f"replay {kind} S={S} C={C} bwd_grid_mult={bwd_grid_mult} own={own_workspace}",
        )
    return replays


@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_cuda_graph_replays(kind, own_workspace) -> None:
    _require_route(kind)
    cases = GRAPH_CASES if kind == "bwd" else ROUTE_CASES[:3]
    replays = _capture_and_replay(cases, kind, key=11, own_workspace=own_workspace)
    assert replays == DRAWS * len(cases)


@pytest.mark.parametrize("bwd_grid_mult", (2, 3, 4))
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_cuda_graph_replays_with_bwd_grid_mult(kind, bwd_grid_mult) -> None:
    """CUDA Graph capture / replay of a ``bwd_grid_mult`` 2-4 backward, with a caller-owned workspace and with the
    package's: every replay judged against the naive tier, the caller-owned workspace zero after every replay."""

    _require_route(kind)
    cases = ROUTE_CASES[:3]
    for own_workspace in (True, False):
        replays = _capture_and_replay(
            cases,
            kind,
            key=33,
            own_workspace=own_workspace,
            bwd_grid_mult=bwd_grid_mult,
        )
        assert replays == DRAWS * len(cases)


@pytest.mark.parametrize("bwd_grid_mult", (1, 2, 3, 4))
def test_fallback_row_graph_replays(bwd_grid_mult, fresh_pool, monkeypatch) -> None:
    """CUDA Graph capture / replay of forward + backward at the default backward route's ``FALLBACK_ROWS`` rows
    (:func:`_fallback_row_cases`: inside the keys' token ranges and above one) at ``bwd_grid_mult`` 1 to 4, with a
    caller-owned workspace and with the package's pre-zeroed buffers (no capture zeroes a buffer of its own): each row
    launches the expected program (checked per ``(S, C)`` of the launches), every replay judged against the naive tier,
    the workspaces zero after every replay."""

    _require_route("bwd")
    device = _device()
    cases = _fallback_row_cases()
    seen = _record_programs(monkeypatch)
    for own_workspace in (True, False):
        seen.clear()
        before = _fallbacks(device)
        replays = _capture_and_replay(
            cases, "bwd", key=191, own_workspace=own_workspace, bwd_grid_mult=bwd_grid_mult
        )
        assert replays == DRAWS * len(cases)
        assert _fallbacks(device) == before, (own_workspace, before, _fallbacks(device))
        expected = {(S, C): {_expected_backward_program(S, C)} for S, C, _ in cases}
        assert _backward_programs_by_shape(seen) == expected, (own_workspace, seen, expected)


def test_capture_takes_one_prezeroed_backward_workspace_per_capture_and_stream(
    fresh_pool, monkeypatch
) -> None:
    """Without a caller workspace, the backward calls that one CUDA Graph capture records on one stream share one
    pre-zeroed package buffer (no fill node in the graph, no fallback counted), not the stream's eager workspace,
    and a second capture gets a buffer of its own; replays of both graphs on each draw's inputs into NaN-filled
    outputs are judged against the naive tier, and the buffer reads back zero after the replays."""

    device = _device()
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append(values["acc"].untyped_storage().data_ptr())
        return run_module(name, values, device_index=device_index, grid=grid)

    # the largest workspace first, so the later calls of a capture fit in the first call's buffer
    shapes = ((4097, 1280, (1000, 0, 3, 3094)), (65, 256, (64, 1)), (1000, 512, "perf"))
    problems = [_inputs(S, C, docs, key=41) for S, C, docs in shapes]
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for x, w, cu, dy in problems:
            _backward(x, w, cu, dy)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    monkeypatch.setattr(pkg, "run_module", record_launch)
    with torch.cuda.stream(side):
        eager = _stream_workspace(device)
    assert eager is not None
    captured = []
    for index in range(2):
        outs = [
            (
                torch.empty_like(x),
                torch.empty(x.shape[1], WIDTH, dtype=w.dtype, device=device),
            )
            for x, w, _, _ in problems
        ]
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        launched.clear()
        before = _fallbacks(device)
        with torch.cuda.graph(graph, stream=side):
            for (x, w, cu, dy), (dx, dw) in zip(problems, outs, strict=True):
                _backward(x, w, cu, dy, dx=dx, dweight=dw)
        torch.cuda.synchronize(device)
        assert len(launched) == len(shapes) and len(set(launched)) == 1, launched
        assert _fallbacks(device) == before, (index, before, _fallbacks(device))
        nodes = _graph_nodes(graph)
        assert nodes["fill"] == 0 and len(nodes["kernels"]) == len(shapes), (index, nodes)
        assert launched[0] != eager.untyped_storage().data_ptr(), "a capture took the stream's eager workspace"
        captured.append((graph, outs, launched[0]))
    assert captured[0][2] != captured[1][2], "two captures share a package workspace"
    for index, (graph, outs, _) in enumerate(captured):
        for i in range(len(shapes)):

            def launch(_draw, seed, graph=graph, outs=outs, i=i):
                fresh = [_inputs(S, C, docs, key=41, seed=seed) for S, C, docs in shapes]
                for held, value in zip(problems, fresh, strict=True):
                    for h, v in zip(held, value, strict=True):
                        h.copy_(v)
                for dx, dw in outs:
                    dx.fill_(float("nan"))
                    dw.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                x, w, cu, dy = fresh[i]
                return _grads(*outs[i], None), problem(x, w, cu, None, dy)

            assert_judged(launch, seed=SEED, label=f"capture {index} replay {shapes[i]}")
    held = pkg._CAPTURE_WORKSPACES.get((device.index, side.cuda_stream))
    assert held is not None and held[1].untyped_storage().data_ptr() == captured[1][2]
    assert int(torch.count_nonzero(held[1])) == 0


def test_two_streams_with_two_workspaces() -> None:
    """Two backward calls running at once on two streams, each with its own workspace (caller-owned, then the
    package's per-stream buffers): both judged against the naive tier on every draw, and every workspace is zero
    afterwards."""

    _two_streams(65536, 1280, "perf")


def test_fallback_row_two_streams(monkeypatch) -> None:
    """The two-stream test at the default backward route's ``FALLBACK_ROWS`` channel counts (C 768, C 1016, inside
    the keys' token ranges): the expected program on both streams, both calls judged against the naive tier on every
    draw, every workspace zero afterwards."""

    seen = _record_programs(monkeypatch)
    for S, C, docs in _fallback_row_large_cases():
        seen.clear()
        _two_streams(S, C, docs)
        assert _backward_programs(seen) == {_expected_backward_program(S, C)}, (S, C, set(seen))


def _two_streams(S: int, C: int, docs) -> None:
    """The body of :func:`test_two_streams_with_two_workspaces` at ``(S, C, docs)``."""

    device = _device()
    keys = (21, 22)
    streams = [torch.cuda.Stream(device=device) for _ in keys]
    for own in (True, False):
        workspaces = [
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
            if own
            else None
            for _ in keys
        ]
        for i in range(len(keys)):

            def launch(_draw, seed, i=i, workspaces=workspaces, own=own):
                problems = [_inputs(S, C, docs, key=k, seed=seed) for k in keys]
                results = [None, None]
                torch.cuda.synchronize(device)
                for j, ((x, w, cu, dy), stream) in enumerate(
                    zip(problems, streams, strict=True)
                ):
                    with torch.cuda.stream(stream):
                        kwargs = {"workspace": workspaces[j]} if own else {}
                        results[j] = _backward(x, w, cu, dy, **kwargs)
                torch.cuda.synchronize(device)
                for j, stream in enumerate(streams):
                    if own:
                        assert int(torch.count_nonzero(workspaces[j])) == 0
                    else:
                        with torch.cuda.stream(stream):
                            held = _stream_workspace(device)
                        assert held is not None and int(torch.count_nonzero(held)) == 0
                x, w, cu, dy = problems[i]
                return _grads(*results[i]), problem(x, w, cu, None, dy)

            assert_judged(launch, seed=SEED, label=f"stream {i} own={own}")


# The shape of the pre-zeroed capture buffer tests: the forward takes token-chunk counters, the backward accumulators.
CAPTURE_CASE = (4097, 1280, (1000, 0, 3, 3094))
# The calls one capture records, ``(forward calls, backward calls)``, forwards first: ``ckpt`` is a non-reentrant
# checkpointed step (two forwards and a backward), ``bwd2`` two backward calls of one capture.
CAPTURE_PHASES = {"fwd": (1, 0), "bwd": (0, 1), "bwd2": (0, 2), "step": (1, 1), "ckpt": (2, 1)}
# The capture / delete cycles of the buffer-return test.
GRAPH_CYCLES = 100
# GPU clock cycles of a ``torch.cuda._sleep`` that keeps a stream busy while the host goes on (about 0.2 s).
BUSY_CYCLES = 400_000_000
# The node types of a CUDA graph DOT print (``cudaGraphDebugDotPrint``).
NODE_TYPES = (
    "KERNEL", "MEMSET", "MEMCPY", "HOST", "EMPTY", "WAIT_EVENT", "EVENT_RECORD", "MEM_ALLOC", "MEM_FREE",
    "EXT_SEMAS_SIGNAL", "EXT_SEMAS_WAIT", "BATCH_MEM_OP", "CONDITIONAL", "GRAPH",
)  # fmt: skip
_LIBC = ctypes.CDLL(None)
_LIBC.sem_getvalue.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]


def _graph_nodes(graph: torch.cuda.CUDAGraph) -> dict:
    """The nodes of a graph captured with ``enable_debug_mode``, from its DOT print (``debug_dump``): ``kernels``,
    the kernel node names in node order; ``fill``, the memset nodes and the kernel nodes of torch's fill kernel
    (``FillFunctor``, a captured ``torch.zeros``); ``edges``, the count of each edge attribute text."""

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "graph.dot")
        graph.debug_dump(path)
        with open(path) as fh:
            text = fh.read()
    kinds, kernels, edges = [], [], {}
    for stmt in re.split(r";\s*\n|\n(?=\s*\")", text):
        s = stmt.strip()
        if "->" in s:
            attrs = re.search(r"\[(.*)\]", s, re.S)
            label = attrs.group(1).strip() if attrs else ""
            edges[label] = edges.get(label, 0) + 1
            continue
        if not re.match(r'^"?graph_\d+_node_\d+"?\s*\[', s):
            continue
        found = [(m.start(), t) for t in NODE_TYPES for m in (re.search(r"\b" + t + r"\b", s),) if m]
        kind = min(found)[1] if found else "UNKNOWN"
        kinds.append(kind)
        if kind == "KERNEL":
            name = re.search(r"(_Z[\w$]+|\w*cake_causal_conv1d\w+|\w*FillFunctor\w*)", s)
            kernels.append(name.group(1) if name else s[:160])
    fill = kinds.count("MEMSET") + sum("FillFunctor" in k for k in kernels)
    return {"kernels": kernels, "fill": fill, "kinds": kinds, "edges": edges}


@pytest.fixture
def fresh_pool(monkeypatch):
    """The package's pre-zeroed capture buffers, their sizes, the buffers graphs hold and the capture-buffer tables
    emptied for one test and restored after it; the idle buffers the test leaves give their user objects back."""

    torch.cuda.synchronize(_device())
    for name, value in (
        ("_ZEROED_POOL", {}),
        ("_POOL_SIZE", {}),
        ("_POOL_READY", {"fwd": {}, "bwd": {}}),
        ("_GRAPH_HELD", []),
        ("_CAPTURE_WORKSPACES", {}),
        ("_CAPTURE_FORWARD_WORKSPACES", {}),
    ):
        monkeypatch.setattr(pkg, name, value)
    yield
    gc.collect()
    torch.cuda.synchronize(_device())
    uo = pkg._USER_OBJECTS
    for pool in pkg._ZEROED_POOL.values():
        for buf in pool:
            if uo:
                uo.release(buf.obj, buf.sem)
        pool.clear()


def _fallbacks(device: torch.device) -> dict:
    """The package's count of captured calls that zeroed a buffer of their own, per kind."""

    return {kind: pkg._CAPTURE_FILL_FALLBACKS.get((device.index, kind), 0) for kind in ("fwd", "bwd")}


def _capture_entries(device: torch.device, stream: torch.cuda.Stream) -> dict:
    """The package's capture-buffer entries of ``stream`` per kind (``(capture id, buffer)`` or ``None``)."""

    key = (device.index, stream.cuda_stream)
    return {"fwd": pkg._CAPTURE_FORWARD_WORKSPACES.get(key), "bwd": pkg._CAPTURE_WORKSPACES.get(key)}


def _draw_inputs(S: int, C: int, docs, kind: str, *, key: int, seed: int = SEED) -> tuple:
    """``(x, w, cu, dy, b)`` of a draw (``seed``) resolving to backward route ``kind`` (``_route_inputs``'s values at
    :data:`SEED`)."""

    settings = ROUTE_SETTINGS[kind]
    x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=settings["weight_dtype"], seed=seed)
    bias = settings["bias_dtype"]
    return x, w, cu, dy, None if bias is None else _bias_input(C, bias, key=key, seed=seed)


def _phase_outs(phase: str, x: torch.Tensor, w: torch.Tensor, b, kind: str) -> dict:
    """The output tensors of the calls of capture phase ``phase``, by output name (one per call)."""

    n_fwd, n_bwd = CAPTURE_PHASES[phase]
    C = x.shape[1]
    dweight = ROUTE_SETTINGS[kind]["dweight_dtype"] or w.dtype
    return {
        "y": [torch.empty_like(x) for _ in range(n_fwd)],
        "dx": [torch.empty_like(x) for _ in range(n_bwd)],
        "dweight": [torch.empty(C, WIDTH, dtype=dweight, device=x.device) for _ in range(n_bwd)],
        "dbias": [] if b is None else [torch.empty(C, dtype=b.dtype, device=x.device) for _ in range(n_bwd)],
    }


def _phase_step(phase: str, inputs: tuple, outs: dict, kind: str) -> None:
    """The calls of capture phase ``phase`` on ``inputs`` (``(x, w, cu, dy, b)``) into ``outs``."""

    x, w, cu, dy, b = inputs
    for y in outs["y"]:
        _forward(x, w, cu, bias=b, out=y)
    policy = ROUTE_SETTINGS[kind]["policy"]
    for i in range(len(outs["dx"])):
        _backward(
            x,
            w,
            cu,
            dy,
            b,
            policy,
            kind=kind,
            dx=outs["dx"][i],
            dweight=outs["dweight"][i],
            dbias=outs["dbias"][i] if outs["dbias"] else None,
        )


def _load(inputs: tuple, fresh: tuple, outs: dict) -> None:
    """Copy a draw's ``fresh`` inputs into the captured ``inputs`` and NaN-fill ``outs``."""

    for held, value in zip(inputs, fresh, strict=True):
        if held is not None:
            held.copy_(value)
    for values in outs.values():
        for t in values:
            t.fill_(float("nan"))


def _outputs(outs: dict, index: int | None = None) -> dict:
    """``outs`` as judged outputs (each name with the values of its calls); ``index``: names of one tensor of a
    multi-pair launch (``y[0]``)."""

    suffix = "" if index is None else f"[{index}]"
    return {name + suffix: list(values) for name, values in outs.items() if values}


def _capture(step, stream: torch.cuda.Stream, *, debug: bool = True) -> dict:
    """``step`` run once on ``stream`` (warm-up), then captured on it by ``torch.cuda.graph``: the ``graph``, the
    capture's ``fallbacks`` per kind (calls that zeroed a buffer of their own), its package capture ``buffers`` per
    kind (the stream's new ``_CAPTURE_*WORKSPACES`` entries) and the pre-zeroed buffers it ``taken`` from the pool."""

    device = _device()
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    before, entries, held = _fallbacks(device), _capture_entries(device, stream), len(pkg._GRAPH_HELD)
    graph = torch.cuda.CUDAGraph()
    if debug:
        graph.enable_debug_mode()
    with torch.cuda.graph(graph, stream=stream):
        step()
    torch.cuda.synchronize(device)
    after = _fallbacks(device)
    buffers = {
        kind: entry[1]
        for kind, entry in _capture_entries(device, stream).items()
        if entry is not None and entry is not entries[kind]
    }
    return {
        "graph": graph,
        "fallbacks": {kind: after[kind] - before[kind] for kind in after},
        "buffers": buffers,
        "taken": [buf.tensor for buf in pkg._GRAPH_HELD[held:]],
    }


def _judge_replays(graph, inputs: tuple, outs: dict, case, kind: str, *, key: int, label: str, buffers=()) -> None:
    """Replays of ``graph`` on each draw's inputs (copied into the captured ``inputs``) into NaN-filled ``outs``,
    judged against the naive tier; every tensor of ``buffers`` reads back zero after every replay."""

    device = _device()
    S, C, docs = case

    def launch(_draw: int, seed: int):
        fresh = _draw_inputs(S, C, docs, kind, key=key, seed=seed)
        _load(inputs, fresh, outs)
        graph.replay()
        torch.cuda.synchronize(device)
        for t in buffers:
            assert int(torch.count_nonzero(t)) == 0, f"{label}: a package capture buffer is not zero after a replay"
        x, w, cu, dy, b = fresh
        return _outputs(outs), problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label=label)


def _posted(sem: int) -> bool:
    """Whether the POSIX semaphore at ``sem`` was posted (``sem_getvalue``; the post is not consumed)."""

    value = ctypes.c_int(0)
    assert _LIBC.sem_getvalue(sem, ctypes.byref(value)) == 0
    return value.value > 0


# The seconds after which :func:`_stream_held` lets its stream go on by itself (the block then fails).
HOLD_TIMEOUT = 60.0


@contextlib.contextmanager
def _stream_held(stream: torch.cuda.Stream):
    """The work queued on ``stream`` inside the block starts on the device only when the block ends: the stream first
    waits (``cuStreamWaitValue32``) for a mapped pinned host word that the block's end sets to 1, so checks inside the
    block see that work not started, whatever the host's speed.  If the host waits for the stream inside the block (a
    call that synchronizes the device, such as a program's first launch in the process under CUDA's lazy module
    loading), a watchdog sets the word after ``HOLD_TIMEOUT`` seconds and sets the yielded event; the block fails."""

    from cuda.bindings import driver, runtime

    err, word = runtime.cudaHostAlloc(4, getattr(runtime, "cudaHostAllocMapped", 2))  # cudaHostAllocMapped = 2
    assert err == runtime.cudaError_t.cudaSuccess, f"cudaHostAlloc: {err}"
    value = ctypes.c_int32.from_address(int(word))
    value.value = 0
    try:
        err, address = runtime.cudaHostGetDevicePointer(word, 0)
        assert err == runtime.cudaError_t.cudaSuccess, f"cudaHostGetDevicePointer: {err}"
        geq = int(driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ)
        (err,) = driver.cuStreamWaitValue32(stream.cuda_stream, int(address), 1, geq)
        assert err == driver.CUresult.CUDA_SUCCESS, f"cuStreamWaitValue32: {err}"
        fired = threading.Event()

        def release() -> None:
            fired.set()
            value.value = 1

        watchdog = threading.Timer(HOLD_TIMEOUT, release)
        watchdog.start()
        try:
            yield fired
        finally:
            watchdog.cancel()
            value.value = 1
            stream.synchronize()
        assert not fired.is_set(), f"the host waited for the held stream (released after {HOLD_TIMEOUT} s)"
    finally:
        value.value = 1
        stream.synchronize()
        runtime.cudaFreeHost(word)


def _record_capture_workspaces(monkeypatch) -> list:
    """From now on, the workspace views (``acc`` / ``work``) of every launch the package makes inside a CUDA Graph
    capture are appended to the returned list (the test's references keep their memory for the checks)."""

    found = []
    launch = pkg.run_module

    def record(name, values, *, device_index, grid=None):
        if torch.cuda.is_current_stream_capturing():
            found.extend(t for t in (values.get("acc"), values.get("work")) if t is not None)
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record)
    return found


def _table_storages() -> set:
    """The storage addresses of the tensors of the package's capture-buffer tables."""

    return {
        t.untyped_storage().data_ptr()
        for table in (pkg._CAPTURE_WORKSPACES, pkg._CAPTURE_FORWARD_WORKSPACES)
        for _, t in table.values()
    }


def _assert_tables_hold_only_pool_buffers() -> None:
    """Every tensor of the package's capture-buffer tables is one of its pre-zeroed buffers (idle, held by a graph
    or released by one), never memory a capture allocated."""

    uo = pkg._USER_OBJECTS
    pool = [buf.tensor for p in pkg._ZEROED_POOL.values() for buf in p] + [buf.tensor for buf in pkg._GRAPH_HELD]
    pool += [buf.tensor for buf in uo.released] if uo else []
    for table in (pkg._CAPTURE_WORKSPACES, pkg._CAPTURE_FORWARD_WORKSPACES):
        for key, (_, t) in table.items():
            assert any(t is p for p in pool), f"capture-table entry {key} holds a tensor that is no pre-zeroed buffer"


@pytest.mark.parametrize("pool", ("prezeroed", "empty"))
def test_capture_records_no_fill_node(pool, fresh_pool, monkeypatch) -> None:
    """A forward, a backward, two backwards, a step (forward + backward) and a checkpointed step (two forwards and a
    backward) captured after one eager warm-up without a caller workspace: the captured calls take the package's
    pre-zeroed buffers (one per kind, the later calls of the capture share it), so the graph's kernel nodes are only
    the package's programs and it has no fill node, and the package counts no fallback.  With the pool forced empty
    (``_POOL_DEPTH`` 0) every captured call zeroes a buffer of its own (fill nodes 1 / 1 / 2 / 2 / 3), each counted in
    ``_CAPTURE_FILL_FALLBACKS``, and the package's tables keep none of them.  Every replay is judged against the naive
    tier and the capture's buffers (the package's, or those the captured calls zeroed) read back zero after it."""

    if pool == "empty":
        monkeypatch.setattr(pkg, "_POOL_DEPTH", 0)
    captured = _record_capture_workspaces(monkeypatch)
    kind, case, device = "bwd", CAPTURE_CASE, _device()
    for phase, (n_fwd, n_bwd) in CAPTURE_PHASES.items():
        inputs = _draw_inputs(*case, kind, key=51)
        outs = _phase_outs(phase, inputs[0], inputs[1], inputs[4], kind)
        captured.clear()
        cap = _capture(lambda inputs=inputs, outs=outs, phase=phase: _phase_step(phase, inputs, outs, kind),
                       torch.cuda.Stream(device=device))  # fmt: skip
        nodes = _graph_nodes(cap["graph"])
        label = f"{phase} pool={pool}"
        expected = {"fwd": 0, "bwd": 0} if pool == "prezeroed" else {"fwd": n_fwd, "bwd": n_bwd}
        assert cap["fallbacks"] == expected, (label, cap["fallbacks"])
        assert nodes["fill"] == sum(expected.values()), (label, nodes)
        programs = [k for k in nodes["kernels"] if "cake_causal_conv1d" in k]
        others = [k for k in nodes["kernels"] if "cake_causal_conv1d" not in k and "FillFunctor" not in k]
        assert len(programs) == n_fwd + n_bwd and not others, (label, nodes)
        if pool == "prezeroed":
            assert len(cap["taken"]) == (n_fwd > 0) + (n_bwd > 0), (label, len(cap["taken"]))
            assert {id(t) for t in cap["taken"]} == {id(t) for t in cap["buffers"].values()}, label
            buffers = list(cap["buffers"].values())
        else:
            assert not cap["taken"] and not cap["buffers"], label
            own = {t.untyped_storage().data_ptr() for t in captured}
            assert len(own) == n_fwd + n_bwd, (label, len(own))  # one buffer per captured call
            assert not own & _table_storages(), f"{label}: the package keeps a buffer a captured call zeroed"
            buffers = list(captured)
        _judge_replays(cap["graph"], inputs, outs, case, kind, key=51, label=label, buffers=buffers)
        del cap


# The forward row of the capture-after-eager test: C 528 (66 vectors) on the default route, whose row floor
# (``FORWARD_NARROW_ROW_FLOORS``) turns its one tile into six tiles of 11 vectors (two counter words per tile).
FLOOR_CAPTURE_C = 528


def test_forward_capture_at_a_floor_after_an_eager_call_below_it(fresh_pool) -> None:
    """An eager forward one token below a narrow row's floor (C 528: its row plan is cached, the package's pre-zeroed
    forward buffers sized for its one tile), then a CUDA Graph capture of the forward at the floor (six tiles, six
    times the counter words) with no eager call at the floor: the captured call takes the plan of the floor from the
    cached row, finds no pre-zeroed buffer large enough and zeroes its own (one fill node, counted in
    ``_CAPTURE_FILL_FALLBACKS``); after an eager call at the floor, a second capture takes a pre-zeroed buffer (no
    fill node, no fallback).  Every replay of both graphs is judged against the naive tier."""

    _require_route("bwd")
    arch, device = _arch(), _device()
    C = FLOOR_CAPTURE_C
    floor = loader.forward_narrow_min_s(arch, C // 8, route="fwd")
    assert floor is not None, f"{arch}: no forward floor at C={C}"
    single = loader.channel_split(C, geometry={"threads": 128, "vec": 8}, role="fwd")
    narrow = loader.channel_split(C, geometry={"threads": 128, "vec": 8}, role="fwd", route="fwd", S=floor, arch=arch)
    assert single[2] == 1 and narrow[2] > 1, (single, narrow)
    below = _draw_inputs(floor - 1, C, "perf", "bwd", key=223)
    _forward(below[0], below[1], below[2])
    torch.cuda.synchronize(device)
    assert pkg._POOL_SIZE.get((device.index, "fwd"), 0) < 2 * narrow[2], pkg._POOL_SIZE
    case = (floor, C, "perf")
    for eager_at_floor, fills in ((False, 1), (True, 0)):
        inputs = _draw_inputs(*case, "bwd", key=223)
        outs = _phase_outs("fwd", inputs[0], inputs[1], inputs[4], "bwd")
        if eager_at_floor:
            _forward(inputs[0], inputs[1], inputs[2])
            torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        before = _fallbacks(device)
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=stream):
            _phase_step("fwd", inputs, outs, "bwd")
        torch.cuda.synchronize(device)
        after = _fallbacks(device)
        nodes = _graph_nodes(graph)
        label = f"{arch} C={C} S={floor} eager_at_floor={eager_at_floor}"
        assert {kind: after[kind] - before[kind] for kind in after} == {"fwd": fills, "bwd": 0}, (label, before, after)
        assert nodes["fill"] == fills, (label, nodes)
        programs = [k for k in nodes["kernels"] if "cake_causal_conv1d" in k]
        assert len(programs) == 1, (label, nodes)
        _judge_replays(graph, inputs, outs, case, "bwd", key=223, label=label)
        del graph


def test_forward_capture_after_an_eager_call_inside_a_forward_key(
    fresh_pool, record_property
) -> None:
    """An eager forward at the last token count of the default route's first ``DIRECT_ROWS`` range (the direct
    program: no counters) refills no pre-zeroed forward buffer.  A CUDA Graph capture of the forward one token above
    the range (the dynamic program, with counters) then finds none and zeroes its own (one fill node, counted in
    ``_CAPTURE_FILL_FALLBACKS``); a capture inside the range has no fill node; a capture one token below the range
    has none when the static program runs there, else (the dynamic program, the pool still empty) one; after an eager
    call one token above the range a capture there takes a pre-zeroed buffer (no fill node, no fallback).  Every
    replay is judged against the naive tier."""

    _require_route("bwd")
    arch, device = _arch(), _device()
    keys = [
        key for key in _forward_keys(arch) if key[0] == "fwd" and key[5] == "direct"
    ]
    assert keys, f"{arch}: no DIRECT_ROWS key on route fwd"
    _, C, _, s_lo, s_hi, _ = keys[0]
    no_work = {
        _forward_program(arch, "fwd", "static"),
        _forward_program(arch, "fwd", "direct"),
    }
    inside = _draw_inputs(s_hi, C, "perf", "bwd", key=227)
    _forward(inside[0], inside[1], inside[2])
    torch.cuda.synchronize(device)
    assert pkg._POOL_SIZE.get((device.index, "fwd"), 0) == 0, pkg._POOL_SIZE
    steps = [
        (s_hi + 1, False),
        (s_hi, False),
        *(((s_lo - 1, False),) if s_lo > 1 else ()),
        (s_hi + 1, True),
    ]
    refilled = False
    for S, eager_first in steps:
        name = _expected_program(arch, "fwd", S, C)
        inputs = _draw_inputs(S, C, "perf", "bwd", key=227)
        outs = _phase_outs("fwd", inputs[0], inputs[1], inputs[4], "bwd")
        if eager_first:
            _forward(inputs[0], inputs[1], inputs[2])
            torch.cuda.synchronize(device)
            refilled |= name not in no_work
        fills = 0 if name in no_work or refilled else 1
        stream = torch.cuda.Stream(device=device)
        before = _fallbacks(device)
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=stream):
            _phase_step("fwd", inputs, outs, "bwd")
        torch.cuda.synchronize(device)
        after = _fallbacks(device)
        nodes = _graph_nodes(graph)
        label = f"{arch} C={C} S={S} {name} eager_first={eager_first}"
        record_property(f"fill_nodes S={S} eager_first={eager_first}", nodes["fill"])
        assert {kind: after[kind] - before[kind] for kind in after} == {
            "fwd": fills,
            "bwd": 0,
        }, (label, before, after)
        assert nodes["fill"] == fills, (label, nodes)
        programs = [k for k in nodes["kernels"] if "cake_causal_conv1d" in k]
        assert len(programs) == 1, (label, nodes)
        _judge_replays(graph, inputs, outs, (S, C, "perf"), "bwd", key=227, label=label)
        del graph


@pytest.mark.parametrize("shapes", ("same", "different"))
def test_two_graphs_replayed_at_once_on_two_streams(shapes, fresh_pool) -> None:
    """Two step graphs (forward + backward, no caller workspace) of the same or of different shapes, captured on two
    streams, take distinct pre-zeroed buffers; on every draw both replay at once on their two streams on the draw's
    inputs: both judged against the naive tier, every buffer zero after the replays."""

    kind, device = "bwd", _device()
    cases = (CAPTURE_CASE, CAPTURE_CASE if shapes == "same" else GRAPH_CASES[2])
    streams = [torch.cuda.Stream(device=device) for _ in cases]
    graphs = []
    for i, (case, stream) in enumerate(zip(cases, streams, strict=True)):
        inputs = _draw_inputs(*case, kind, key=61 + i)
        outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
        cap = _capture(lambda inputs=inputs, outs=outs: _phase_step("step", inputs, outs, kind), stream, debug=False)
        assert cap["fallbacks"] == {"fwd": 0, "bwd": 0}, (i, cap["fallbacks"])
        graphs.append((cap, inputs, outs))
    buffers = [t for cap, _, _ in graphs for t in cap["buffers"].values()]
    assert len(buffers) == 4 and len({t.data_ptr() for t in buffers}) == 4, "two graphs share a capture buffer"

    def launch(_draw: int, seed: int):
        fresh = [_draw_inputs(*case, kind, key=61 + i, seed=seed) for i, case in enumerate(cases)]
        for (_, inputs, outs), values in zip(graphs, fresh, strict=True):
            _load(inputs, values, outs)
        for (cap, _, _), stream in zip(graphs, streams, strict=True):
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                cap["graph"].replay()
        torch.cuda.synchronize(device)
        assert all(int(torch.count_nonzero(t)) == 0 for t in buffers), "a capture buffer is not zero"
        return [
            (_outputs(outs, i), problem(x, w, cu, b, dy))
            for i, ((_, _, outs), (x, w, cu, dy, b)) in enumerate(zip(graphs, fresh, strict=True))
        ]

    assert_judged(launch, seed=SEED, label=f"two graphs at once shapes={shapes}")


def test_eager_calls_interleaved_with_replays(fresh_pool) -> None:
    """Eager steps (no caller workspace) and replays of two step graphs interleaved on two streams with no host
    synchronization between them (eager on A, replay on B, eager on B, replay on A): every result judged against
    the naive tier; the graphs' buffers and the eager calls' per-stream workspaces read back zero afterwards."""

    kind, case, device = "bwd", CAPTURE_CASE, _device()
    stream_a, stream_b = (torch.cuda.Stream(device=device) for _ in range(2))
    graphs = []
    for _ in range(2):
        inputs = _draw_inputs(*case, kind, key=71)
        outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
        cap = _capture(lambda inputs=inputs, outs=outs: _phase_step("step", inputs, outs, kind),
                       torch.cuda.Stream(device=device), debug=False)  # fmt: skip
        assert cap["fallbacks"] == {"fwd": 0, "bwd": 0}, cap["fallbacks"]
        graphs.append((cap, inputs, outs))
    buffers = [t for cap, _, _ in graphs for t in cap["buffers"].values()]
    eager = [_phase_outs("step", graphs[0][1][0], graphs[0][1][1], graphs[0][1][4], kind) for _ in range(2)]

    def launch(_draw: int, seed: int):
        fresh = _draw_inputs(*case, kind, key=71, seed=seed)
        for _, inputs, outs in graphs:
            _load(inputs, fresh, outs)
        for outs in eager:
            _load((None,) * 5, fresh, outs)
        for stream in (stream_a, stream_b):
            stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream_a):
            _phase_step("step", fresh, eager[0], kind)
        with torch.cuda.stream(stream_b):
            graphs[0][0]["graph"].replay()
            _phase_step("step", fresh, eager[1], kind)
        with torch.cuda.stream(stream_a):
            graphs[1][0]["graph"].replay()
        torch.cuda.synchronize(device)
        assert all(int(torch.count_nonzero(t)) == 0 for t in buffers), "a capture buffer is not zero"
        for stream in (stream_a, stream_b):
            with torch.cuda.stream(stream):
                for held in (_stream_workspace(device), _forward_stream_workspace(device)):
                    assert held is not None and int(torch.count_nonzero(held)) == 0, "an eager workspace is not zero"
        found = {name: [] for name in eager[0]}
        for outs in (eager[0], graphs[0][2], eager[1], graphs[1][2]):
            for name, values in outs.items():
                found[name] += values
        x, w, cu, dy, b = fresh
        return _outputs(found), problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label="eager interleaved with replays")


def test_graph_buffers_return_after_graph_deletion(fresh_pool) -> None:
    """``GRAPH_CYCLES`` cycles of an eager step, a step capture, a replay and ``del graph``: no capture zeroes a buffer
    of its own; the buffers graphs hold and the idle pool stay bounded (the deleted graph's two until the next eager
    refill returns them, at most ``_POOL_DEPTH`` idle per kind) and ``torch.cuda.memory_allocated`` does not grow;
    a buffer a deleted graph released is taken again by a later capture, whose replays are judged against the naive
    tier with the buffer zero after every replay."""

    kind, case, device = "bwd", GRAPH_CASES[2], _device()
    stream = torch.cuda.Stream(device=device)
    inputs = _draw_inputs(*case, kind, key=81)
    outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
    seen: list[weakref.ref] = []
    allocated, judged = [], 0
    for cycle in range(GRAPH_CYCLES):
        cap = _capture(lambda: _phase_step("step", inputs, outs, kind), stream, debug=False)
        assert cap["fallbacks"] == {"fwd": 0, "bwd": 0}, (cycle, cap["fallbacks"])
        assert len(cap["taken"]) == 2, (cycle, len(cap["taken"]))
        recycled = any(ref() is t for t in cap["taken"] for ref in seen)
        seen += [weakref.ref(t) for t in cap["taken"]]
        with torch.cuda.stream(stream):
            cap["graph"].replay()
        if recycled and not judged:
            _judge_replays(cap["graph"], inputs, outs, case, kind, key=81, label=f"recycled buffer cycle {cycle}",
                           buffers=cap["taken"])  # fmt: skip
            judged += 1
        del cap
        torch.cuda.synchronize(device)
        assert len(pkg._GRAPH_HELD) <= 4, (cycle, len(pkg._GRAPH_HELD))
        assert all(len(p) <= pkg._POOL_DEPTH for p in pkg._ZEROED_POOL.values()), cycle
        allocated.append(torch.cuda.memory_allocated(device))
    assert judged, "no capture took a buffer that a deleted graph released"
    # one graph's two buffers as the caching allocator counts them (512-byte blocks); two graphs' buffers may wait
    # for their release at once (the held bound above)
    one_graph = sum(
        -(-pkg._POOL_SIZE[(device.index, k)] * pkg._POOL_DTYPES[k].itemsize // 512) * 512 for k in ("fwd", "bwd")
    )
    half = GRAPH_CYCLES // 2
    early, late = max(allocated[10:half]), max(allocated[half:])
    assert late <= early + 2 * one_graph, (early, late, one_graph, sorted(set(allocated)))


def test_graph_deleted_while_replay_in_flight(fresh_pool, monkeypatch) -> None:
    """With one idle buffer per kind (``_POOL_DEPTH`` 1): a step graph replays behind a ``torch.cuda._sleep`` on
    stream B and is deleted at once (no synchronization); an eager step on stream A then refills the pool while the
    replay is in flight and does not return the deleted graph's buffers (they stay held until its launches complete),
    and a new capture takes other buffers.  The in-flight replay's results and the new graph's replay are judged
    against the naive tier on every draw."""

    monkeypatch.setattr(pkg, "_POOL_DEPTH", 1)
    kind, case, device = "bwd", GRAPH_CASES[2], _device()
    stream_a, stream_b = (torch.cuda.Stream(device=device) for _ in range(2))

    def launch(_draw: int, seed: int):
        fresh = _draw_inputs(*case, kind, key=91, seed=seed)
        x, w, _, _, b = fresh
        first, second, eager = (_phase_outs("step", x, w, b, kind) for _ in range(3))
        cap = _capture(lambda: _phase_step("step", fresh, first, kind), stream_b, debug=False)
        assert cap["fallbacks"] == {"fwd": 0, "bwd": 0} and len(cap["taken"]) == 2, cap["fallbacks"]
        held = cap.pop("taken")
        graph = cap.pop("graph")
        with torch.cuda.stream(stream_b):
            torch.cuda._sleep(BUSY_CYCLES)
            graph.replay()
        del graph, cap
        stream_a.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream_a):
            _phase_step("step", fresh, eager, kind)
        in_flight = not stream_b.query()
        idle = [buf.tensor for p in pkg._ZEROED_POOL.values() for buf in p]
        assert in_flight, "the replay completed before the check (BUSY_CYCLES too short for this host)"
        assert not any(t is h for t in idle for h in held), "a buffer of an in-flight graph went back to the pool"
        assert all(any(buf.tensor is h for buf in pkg._GRAPH_HELD) for h in held)
        later = _capture(lambda: _phase_step("step", fresh, second, kind), stream_a, debug=False)
        assert later["fallbacks"] == {"fwd": 0, "bwd": 0}, later["fallbacks"]
        assert not {t.data_ptr() for t in later["taken"]} & {h.data_ptr() for h in held}
        with torch.cuda.stream(stream_a):
            later["graph"].replay()
        torch.cuda.synchronize(device)
        found = {name: first[name] + second[name] for name in first if first[name]}
        x, w, cu, dy, b = fresh
        return found, problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label="graph deleted in flight")


def test_raw_capture_without_device_sync(fresh_pool) -> None:
    """An eager step on stream A behind a ``torch.cuda._sleep`` refills the empty pool (the new buffers' zero fills
    and readiness copies wait behind the sleep); ``CUDAGraph.capture_begin`` / ``capture_end`` on stream B at once,
    with no synchronization, takes none of them (each captured call zeroes a buffer of its own: two fill nodes, two
    fallbacks) and the graph replays on B at once; that replay and the later ones are judged against the naive tier."""

    kind, case, device = "bwd", CAPTURE_CASE, _device()
    stream_a, stream_b = (torch.cuda.Stream(device=device) for _ in range(2))
    inputs = _draw_inputs(*case, kind, key=101)
    outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
    eager = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
    torch.cuda.synchronize(device)
    before = _fallbacks(device)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.stream(stream_a):
        torch.cuda._sleep(BUSY_CYCLES)
        _phase_step("step", inputs, eager, kind)
    with torch.cuda.stream(stream_b):
        graph.capture_begin()
        _phase_step("step", inputs, outs, kind)
        graph.capture_end()
        graph.replay()
    in_flight = not stream_a.query()
    torch.cuda.synchronize(device)
    after = _fallbacks(device)
    assert in_flight, "the refill completed before the capture (BUSY_CYCLES too short for this host)"
    assert {k: after[k] - before[k] for k in after} == {"fwd": 1, "bwd": 1}, (before, after)
    nodes = _graph_nodes(graph)
    assert nodes["fill"] == 2, nodes
    first = _outputs({name: [t.clone() for t in values] for name, values in outs.items()})

    def launch(draw: int, seed: int):
        fresh = _draw_inputs(*case, kind, key=101, seed=seed)
        if draw == 0:
            found = first  # the replay at once on the captured inputs (draw 0's)
        else:
            _load(inputs, fresh, outs)
            graph.replay()
            torch.cuda.synchronize(device)
            found = _outputs(outs)
        x, w, cu, dy, b = fresh
        return found, problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label="raw capture without sync")


def test_aborted_capture_returns_its_buffer(fresh_pool) -> None:
    """An exception raised inside ``torch.cuda.graph`` after a step that took two pre-zeroed buffers: once the graph
    object is freed, CUDA releases them (their semaphores are posted) and the next eager step returns them, zero, to
    the pool; a later capture's replays are judged against the naive tier."""

    kind, case, device = "bwd", GRAPH_CASES[2], _device()
    stream = torch.cuda.Stream(device=device)
    inputs = _draw_inputs(*case, kind, key=111)
    outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)

    def step():
        _phase_step("step", inputs, outs, kind)

    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        step()
    torch.cuda.synchronize(device)
    held = len(pkg._GRAPH_HELD)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="aborted capture"):
        with torch.cuda.graph(graph, stream=stream):
            step()
            raise RuntimeError("aborted capture")
    taken = list(pkg._GRAPH_HELD[held:])
    assert len(taken) == 2, len(taken)
    del graph
    gc.collect()
    torch.cuda.synchronize(device)
    deadline = time.monotonic() + 10.0
    while not all(_posted(buf.sem) for buf in taken):
        assert time.monotonic() < deadline, "CUDA did not release the aborted capture's buffers"
        time.sleep(0.001)
    with torch.cuda.stream(stream):
        step()
    torch.cuda.synchronize(device)
    for buf in taken:
        assert any(b is buf for b in pkg._ZEROED_POOL[(device.index, buf.kind)]), buf.kind
        assert not any(b is buf for b in pkg._GRAPH_HELD), buf.kind
        assert int(torch.count_nonzero(buf.tensor)) == 0, buf.kind
    cap = _capture(step, stream, debug=False)
    assert cap["fallbacks"] == {"fwd": 0, "bwd": 0}, cap["fallbacks"]
    _judge_replays(cap["graph"], inputs, outs, case, kind, key=111, label="capture after an aborted one",
                   buffers=cap["taken"])  # fmt: skip


def test_capture_without_cuda_bindings(fresh_pool, monkeypatch) -> None:
    """With ``cuda.bindings`` not importable the package keeps no pre-zeroed buffers and reads no capture id: every
    captured call zeroes a buffer of its own (fill nodes 1 / 1 / 2 / 2 / 3 for the phases of
    :data:`CAPTURE_PHASES`, each counted in ``_CAPTURE_FILL_FALLBACKS``), keeps no capture-buffer entry, and every
    replay is judged against the naive tier."""

    monkeypatch.setitem(sys.modules, "cuda.bindings", None)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", None)
    monkeypatch.setattr(pkg, "_USER_OBJECTS", None)
    kind, case, device = "bwd", CAPTURE_CASE, _device()
    for phase, (n_fwd, n_bwd) in CAPTURE_PHASES.items():
        inputs = _draw_inputs(*case, kind, key=121)
        outs = _phase_outs(phase, inputs[0], inputs[1], inputs[4], kind)
        cap = _capture(lambda inputs=inputs, outs=outs, phase=phase: _phase_step(phase, inputs, outs, kind),
                       torch.cuda.Stream(device=device))  # fmt: skip
        assert pkg._USER_OBJECTS is False, phase
        assert cap["fallbacks"] == {"fwd": n_fwd, "bwd": n_bwd}, (phase, cap["fallbacks"])
        assert not cap["buffers"] and not cap["taken"], phase
        assert _graph_nodes(cap["graph"])["fill"] == n_fwd + n_bwd, phase
        _judge_replays(cap["graph"], inputs, outs, case, kind, key=121, label=f"no cuda.bindings {phase}")
        del cap


@pytest.mark.parametrize("failure", ("missing", "raising"))
def test_capture_without_mem_pool(failure, fresh_pool, monkeypatch) -> None:
    """Without ``torch.cuda.MemPool`` (``missing``), or with ``torch.cuda.use_mem_pool`` raising, the package keeps
    no pre-zeroed buffers (the pool of each kind is off) and no memory pool: the eager steps run on new per-stream
    workspaces of the caching allocator (each draw's outputs judged against the naive tier, both workspaces zero after
    every step) and every captured step zeroes its two buffers itself (two fill nodes, one fallback per kind, no
    table entry); replays are judged against the naive tier, with those buffers zero after each."""

    monkeypatch.setattr(pkg, "_MEM_POOLS", {})
    monkeypatch.setattr(pkg, "_FORWARD_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_BACKWARD_WORKSPACES", {})
    if failure == "missing":
        monkeypatch.setattr(torch.cuda, "MemPool", None)
    else:

        def refuse(*args, **kwargs):
            raise RuntimeError("use_mem_pool refused")

        monkeypatch.setattr(torch.cuda, "use_mem_pool", refuse)
    kind, case, device = "bwd", CAPTURE_CASE, _device()
    inputs = _draw_inputs(*case, kind, key=131)
    outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
    stream = torch.cuda.Stream(device=device)
    captured = _record_capture_workspaces(monkeypatch)

    def eager(_draw: int, seed: int):
        fresh = _draw_inputs(*case, kind, key=131, seed=seed)
        _load(inputs, fresh, outs)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            _phase_step("step", inputs, outs, kind)
        torch.cuda.synchronize(device)
        with torch.cuda.stream(stream):
            held = (_stream_workspace(device), _forward_stream_workspace(device))
        assert all(t is not None and int(torch.count_nonzero(t)) == 0 for t in held), "an eager workspace is not zero"
        x, w, cu, dy, b = fresh
        return _outputs(outs), problem(x, w, cu, b, dy)

    assert_judged(eager, seed=SEED, label=f"no MemPool ({failure}) eager")
    assert pkg._MEM_POOLS.get(device.index) is None
    cap = _capture(lambda: _phase_step("step", inputs, outs, kind), stream)
    assert not pkg._ZEROED_POOL.get((device.index, "bwd")) and not pkg._ZEROED_POOL.get((device.index, "fwd"))
    assert all(pkg._POOL_READY[k].get(device.index) == sys.maxsize for k in ("fwd", "bwd")), pkg._POOL_READY
    assert pkg._MEM_POOLS.get(device.index) is None
    assert cap["fallbacks"] == {"fwd": 1, "bwd": 1} and not cap["taken"] and not cap["buffers"], cap["fallbacks"]
    assert _graph_nodes(cap["graph"])["fill"] == 2
    assert len({t.untyped_storage().data_ptr() for t in captured}) == 2, len(captured)
    _judge_replays(cap["graph"], inputs, outs, case, kind, key=131, label=f"no MemPool ({failure})",
                   buffers=list(captured))  # fmt: skip


def test_out_of_memory_keeps_the_package_pool(fresh_pool) -> None:
    """An out-of-memory error while the package allocates does not take the device off its memory pool or its
    pre-zeroed buffers: a refill for a size that does not fit leaves the pool's buffers and size as they were, a
    per-stream workspace that does not fit raises the error and the device keeps its pool, and a later capture still
    takes the pre-zeroed buffers (no fill node, no fallback); its replays are judged against the naive tier."""

    kind, case, device = "bwd", GRAPH_CASES[2], _device()
    d = device.index
    stream = torch.cuda.Stream(device=device)
    inputs = _draw_inputs(*case, kind, key=135)
    outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)

    def step():
        _phase_step("step", inputs, outs, kind)

    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        step()
    torch.cuda.synchronize(device)
    mem_pool = pkg._MEM_POOLS.get(d)
    assert mem_pool is not None
    huge = 1 << 50
    for k in ("fwd", "bwd"):
        idle, size = list(pkg._ZEROED_POOL[(d, k)]), pkg._POOL_SIZE[(d, k)]
        assert len(idle) == pkg._POOL_DEPTH, (k, len(idle))
        with torch.cuda.stream(stream):
            pkg._pool_refill(d, k, huge, device)
        assert [id(b) for b in pkg._ZEROED_POOL[(d, k)]] == [id(b) for b in idle], k
        assert pkg._POOL_SIZE[(d, k)] == size and pkg._POOL_READY[k][d] == huge, k
    with pytest.raises(pkg._OUT_OF_MEMORY):
        pkg._package_zeros(huge, torch.uint8, device, d)
    assert pkg._MEM_POOLS.get(d) is mem_pool
    cap = _capture(step, stream)
    assert cap["fallbacks"] == {"fwd": 0, "bwd": 0} and len(cap["taken"]) == 2, cap["fallbacks"]
    assert _graph_nodes(cap["graph"])["fill"] == 0
    _judge_replays(cap["graph"], inputs, outs, case, kind, key=135, label="capture after an out-of-memory error",
                   buffers=cap["taken"])  # fmt: skip


class _WatchedLock:
    """``pkg._POOL_LOCK`` with the package's ``with`` statements unchanged, recording under ``cond`` (notified on every
    change) its ``owner`` (the name of the thread that holds it, or ``None``) and the names of the threads ``waiting``
    for it: a thread enters ``waiting`` only after a non-blocking attempt to take it failed."""

    def __init__(self, lock, cond: threading.Condition, waiting: set) -> None:
        self.lock, self.cond, self.waiting, self.owner = lock, cond, waiting, None

    def __enter__(self):
        me = threading.current_thread().name
        if not self.lock.acquire(blocking=False):
            with self.cond:
                self.waiting.add(me)
                self.cond.notify_all()
            self.lock.acquire()
        with self.cond:
            self.waiting.discard(me)
            self.owner = me
            self.cond.notify_all()
        return self

    def __exit__(self, *exc) -> None:
        with self.cond:
            self.owner = None
            self.cond.notify_all()
        self.lock.release()


# The threads of the pool-lock test: a refill and two first eager steps on new streams.
LOCK_THREADS = ("refill", "worker-0", "worker-1")
# The seconds any thread of the pool-lock test waits for the others before it fails.
LOCK_TIMEOUT = 60.0


def test_pool_lock_keeps_one_thread_in_the_package_pool(fresh_pool, monkeypatch) -> None:
    """``_POOL_LOCK`` lets one thread at a time into the package's memory pool.  ``torch.cuda.use_mem_pool`` is
    wrapped to count the threads inside each pool, and a thread that enters stays at the entry until another thread
    is inside too, or each other thread has finished or is blocked on ``_POOL_LOCK`` while the entering thread holds
    it (:class:`_WatchedLock`).  A refill that grows the backward pool enters first; only then two threads make their
    first eager steps on two new streams (each zeroes its per-stream workspaces in the pool).  With the lock around
    every pool entry the steps block on it while the refill is inside (forced contention, no timing).  An entry
    without the lock (``_package_zeros`` or the refill) does not hold it, so it stays at the entry until another
    thread is inside too, and a step cannot finish without entering the pool: two threads are inside at once whatever
    the schedule.  At most one thread is ever inside, the refill entered once (holding the lock) and each step twice,
    all into the device's pool, which stays on (no ``_pool_off``, the grown pre-zeroed buffers in place) and the
    steps' workspaces read back zero."""

    kind, case, device = "bwd", GRAPH_CASES[2], _device()
    d = device.index
    monkeypatch.setattr(pkg, "_FORWARD_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_BACKWARD_WORKSPACES", {})
    inputs = _draw_inputs(*case, kind, key=149)
    warm = torch.cuda.Stream(device=device)
    warm.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warm):
        _phase_step("step", inputs, _phase_outs("step", inputs[0], inputs[1], inputs[4], kind), kind)
    torch.cuda.synchronize(device)
    mem_pool = pkg._MEM_POOLS.get(d)
    assert mem_pool is not None
    ready = {k: pkg._POOL_READY[k][d] for k in ("fwd", "bwd")}
    grown = 2 * pkg._POOL_SIZE[(d, "bwd")]
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    outs = [_phase_outs("step", inputs[0], inputs[1], inputs[4], kind) for _ in streams]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream(device))

    cond = threading.Condition()
    waiting: set = set()
    done: set = set()
    inside: dict = {}
    entries: list = []
    errors: list = []
    entered = threading.Event()  # the refill is inside the pool
    real = torch.cuda.use_mem_pool
    watched = _WatchedLock(pkg._POOL_LOCK, cond, waiting)

    def blocked(name: str, me: str) -> bool:  # ``name`` waits for the lock that ``me`` holds
        return name in waiting and watched.owner == me

    @contextlib.contextmanager
    def counted(pool, *args, **kwargs):
        me = threading.current_thread().name
        others = [name for name in LOCK_THREADS if name != me]
        deadline = time.monotonic() + LOCK_TIMEOUT
        with cond:
            inside[id(pool)] = inside.get(id(pool), 0) + 1
            entry = [me, id(pool), inside[id(pool)]]  # + the lock's waiting threads and owner when it went in
            entries.append(entry)
            cond.notify_all()
            if me == "refill":
                entered.set()
            while inside[id(pool)] < 2 and not all(name in done or blocked(name, me) for name in others):
                left = deadline - time.monotonic()
                if left <= 0:
                    errors.append(AssertionError(f"{me}: the other threads neither waited for the lock nor ended"))
                    break
                cond.wait(left)
            entry += [frozenset(waiting), watched.owner]
        try:
            with real(pool, *args, **kwargs):
                yield
        finally:
            with cond:
                inside[id(pool)] -= 1
                cond.notify_all()

    def run(name: str, body) -> None:
        try:
            with torch.cuda.device(d):
                body()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the test's thread
            errors.append(exc)
        finally:
            with cond:
                done.add(name)
                cond.notify_all()

    def refill() -> None:
        with torch.cuda.stream(warm):
            pkg._pool_refill(d, "bwd", grown, device)

    def step(i: int) -> None:
        if not entered.wait(LOCK_TIMEOUT):
            raise AssertionError("the refill never entered the package's memory pool")
        with torch.cuda.stream(streams[i]):
            _phase_step("step", inputs, outs[i], kind)

    bodies = (refill, lambda: step(0), lambda: step(1))
    threads = [
        threading.Thread(target=run, args=(name, body), name=name, daemon=True)
        for name, body in zip(LOCK_THREADS, bodies, strict=True)
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_POOL_LOCK", watched)
        patch.setattr(torch.cuda, "use_mem_pool", counted)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(3 * LOCK_TIMEOUT)
        assert not any(thread.is_alive() for thread in threads), "a thread of the pool-lock test did not end"
    torch.cuda.synchronize(device)
    if errors:
        raise errors[0]
    assert max(entry[2] for entry in entries) == 1, f"two threads were inside the package's pool: {entries}"
    assert sorted(name for name, *_ in entries) == ["refill", "worker-0", "worker-0", "worker-1", "worker-1"], entries
    assert {entry[1] for entry in entries} == {id(mem_pool)}, entries
    (refill_entry,) = [entry for entry in entries if entry[0] == "refill"]
    assert refill_entry[4] == "refill", f"the refill entered the pool without the lock: {entries}"
    assert {"worker-0", "worker-1"} <= refill_entry[3], f"the steps did not wait for the lock: {entries}"
    assert pkg._MEM_POOLS.get(d) is mem_pool, "the device left its memory pool"
    assert pkg._POOL_READY["fwd"][d] == ready["fwd"] and pkg._POOL_READY["bwd"][d] == grown, pkg._POOL_READY
    idle = pkg._ZEROED_POOL[(d, "bwd")]
    assert len(idle) == pkg._POOL_DEPTH and all(buf.tensor.numel() == grown for buf in idle), "the refill"
    for stream in streams:
        key = (d, stream.cuda_stream)
        for table in (pkg._FORWARD_WORKSPACES, pkg._BACKWARD_WORKSPACES):
            assert key in table and int(torch.count_nonzero(table[key])) == 0, "a step's workspace is not zero"


def test_more_captures_than_the_pool_keep_no_capture_memory(fresh_pool, monkeypatch) -> None:
    """``_POOL_DEPTH`` + 2 step graphs (forward + backward, no caller workspace) captured back to back by
    ``torch.cuda.graph`` after one eager step, with no eager call between the captures: the first ``_POOL_DEPTH`` take
    one pre-zeroed buffer per kind and their launches use exactly those buffers, each call of the later ones zeroes a
    buffer of its own (a fill node and a fallback per kind and capture).  After the captures the package's tables hold
    only pre-zeroed buffers and none of the buffers the captured calls zeroed; every graph's replays are judged
    against the naive tier, with its buffers zero after each."""

    kind, case, device = "bwd", GRAPH_CASES[2], _device()
    stream = torch.cuda.Stream(device=device)
    inputs = _draw_inputs(*case, kind, key=137)
    captured = _record_capture_workspaces(monkeypatch)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        _phase_step("step", inputs, _phase_outs("step", inputs[0], inputs[1], inputs[4], kind), kind)
    torch.cuda.synchronize(device)
    depth = pkg._POOL_DEPTH
    graphs = []
    for index in range(depth + 2):
        outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
        before, held = _fallbacks(device), len(pkg._GRAPH_HELD)
        captured.clear()
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
        with torch.cuda.graph(graph, stream=stream):
            _phase_step("step", inputs, outs, kind)
        torch.cuda.synchronize(device)
        after = _fallbacks(device)
        taken = [buf.tensor for buf in pkg._GRAPH_HELD[held:]]
        expected = {"fwd": 0, "bwd": 0} if index < depth else {"fwd": 1, "bwd": 1}
        assert {k: after[k] - before[k] for k in after} == expected, (index, before, after)
        assert len(taken) == (2 if index < depth else 0), (index, len(taken))
        assert _graph_nodes(graph)["fill"] == sum(expected.values()), index
        launched = {t.untyped_storage().data_ptr() for t in captured}
        if index < depth:  # the capture's launches use the buffers it took
            assert launched == {t.untyped_storage().data_ptr() for t in taken}, index
            own = []
        else:
            own = list(captured)
            assert len(launched) == sum(expected.values()), index
        graphs.append((graph, outs, taken + own))
    _assert_tables_hold_only_pool_buffers()
    own = {t.untyped_storage().data_ptr() for _, _, buffers in graphs[depth:] for t in buffers}
    assert own and not own & _table_storages(), "the package keeps a buffer a captured call zeroed"
    for index, (graph, outs, buffers) in enumerate(graphs):
        _judge_replays(graph, inputs, outs, case, kind, key=137, label=f"capture {index} of {depth + 2}",
                       buffers=buffers)  # fmt: skip


def test_capture_larger_than_the_pool_zeroes_its_own_buffer(fresh_pool, monkeypatch) -> None:
    """The pre-zeroed backward buffers hold the largest need of the eager calls that omitted ``workspace=``: an eager
    backward at C 1280 with a caller-owned workspace fills no pool, one at C 512 without one fills it at C 512's
    size.  A backward captured at C 1280 needs more: it takes no buffer (the idle ones stay as they were), zeroes one
    of its own (one fill node, one fallback) and the package keeps no reference to it; the replays are judged against
    the naive tier, with that buffer zero after each."""

    kind, device = "bwd", _device()
    d = device.index
    small, large = GRAPH_CASES[2], CAPTURE_CASE
    stream = torch.cuda.Stream(device=device)
    big = _draw_inputs(*large, kind, key=139)
    little = _draw_inputs(*small, kind, key=139)
    outs = _phase_outs("bwd", big[0], big[1], big[4], kind)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(large[0], large[1], device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        x, w, cu, dy, b = big
        _backward(x, w, cu, dy, b, ROUTE_SETTINGS[kind]["policy"], kind=kind, workspace=workspace)
    torch.cuda.synchronize(device)
    assert not pkg._ZEROED_POOL.get((d, "bwd")), "an eager call with a caller-owned workspace filled the pool"
    with torch.cuda.stream(stream):
        _phase_step("bwd", little, _phase_outs("bwd", little[0], little[1], little[4], kind), kind)
    torch.cuda.synchronize(device)
    idle = list(pkg._ZEROED_POOL[(d, "bwd")])
    assert len(idle) == pkg._POOL_DEPTH, len(idle)
    captured = _record_capture_workspaces(monkeypatch)
    before, held = _fallbacks(device), len(pkg._GRAPH_HELD)
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph, stream=stream):
        _phase_step("bwd", big, outs, kind)
    torch.cuda.synchronize(device)
    after = _fallbacks(device)
    assert {k: after[k] - before[k] for k in after} == {"fwd": 0, "bwd": 1}, (before, after)
    assert len(pkg._GRAPH_HELD) == held and [id(b) for b in pkg._ZEROED_POOL[(d, "bwd")]] == [id(b) for b in idle]
    assert _graph_nodes(graph)["fill"] == 1
    need = max(t.untyped_storage().nbytes() for t in captured)
    assert all(buf.tensor.numel() < need for buf in idle), (need, [buf.tensor.numel() for buf in idle])
    assert not {t.untyped_storage().data_ptr() for t in captured} & _table_storages()
    _judge_replays(graph, big, outs, large, kind, key=139, label="capture larger than the pool",
                   buffers=list(captured))  # fmt: skip


def test_pool_growth_releases_smaller_buffers_and_keeps_graphs_apart(fresh_pool) -> None:
    """The pool grows with the eager need.  Graph G1, captured at C 512, takes one buffer per kind; eager steps at
    C 1280 and then C 2560 on another stream (their programs launched once before on caller-owned workspaces; the
    steps in the hold launch no other program), held on the device until the checks end (:func:`_stream_held`: their
    zero fills and readiness copies wait; the host never waits for the stream), each raise the pool size: the idle
    buffers below the new size leave the pool and ``_POOL_DEPTH`` buffers of the new size replace them, a released
    buffer whose readiness copy has not run gives its flag back as pending, and no semaphore or flag serves two buffers.
    CUDA gets the first growth's released user objects only after the checks, so their semaphores are not posted at
    the second growth's drains: they stay retired and no new buffer takes one.  The second growth drops each idle
    buffer through the package's own release, watched: the call puts the semaphore in ``uo.retired`` at once, and the
    backward refill's semaphores are still there when the hold ends (no drain runs after that refill).  After the hold
    each released semaphore is posted by CUDA, or a drain in the hold took its post and moved it from ``uo.retired``
    to ``uo.free_sems``.  A capture at C 2560 (G2) then takes new buffers (no fallback) and the next eager step's drain
    gives the posted semaphores back (free, or taken by a new buffer; none retired) and the pending flags.  On every
    draw G1 and G2 replay at once on two streams: both judged against the naive tier, their buffers zero after the
    replays and held by one graph each.  Then G1 is deleted: after a capture at C 2560 (G3), an eager step's drain
    drops each of G1's released buffers that is smaller than the grown size (the package keeps no reference to it) and
    gives its semaphore back."""

    kind, device = "bwd", _device()
    d = device.index
    kinds = ("fwd", "bwd")
    cases = (GRAPH_CASES[2], CAPTURE_CASE, GRAPH_CASES[4])
    stream_a, stream_b = (torch.cuda.Stream(device=device) for _ in range(2))
    inputs = [_draw_inputs(*case, kind, key=145 + i) for i, case in enumerate(cases)]
    outs = [_phase_outs("step", x, w, b, kind) for x, w, _, _, b in inputs]
    first = _capture(lambda: _phase_step("step", inputs[0], outs[0], kind), stream_a, debug=False)
    assert first["fallbacks"] == {"fwd": 0, "bwd": 0} and len(first["taken"]) == 2, first["fallbacks"]
    uo = pkg._USER_OBJECTS

    def state() -> tuple[dict, dict]:
        return (
            {k: list(pkg._ZEROED_POOL.get((d, k), ())) for k in kinds},
            {k: pkg._POOL_SIZE[(d, k)] for k in kinds},
        )

    def live_sems() -> set:
        return {buf.sem for p in pkg._ZEROED_POOL.values() for buf in p} | {buf.sem for buf in pkg._GRAPH_HELD}

    def check_unique(label: str) -> None:
        live = [buf for p in pkg._ZEROED_POOL.values() for buf in p] + list(pkg._GRAPH_HELD)
        sems = [buf.sem for buf in live]
        assert len(set(sems)) == len(sems), f"{label}: two buffers share a semaphore"
        assert not set(sems) & (set(uo.retired) | set(uo.free_sems)), f"{label}: a buffer's semaphore is free"
        flags = [buf.flag[2] for buf in live if buf.flag]
        assert len(set(flags)) == len(flags), f"{label}: two buffers share a readiness flag"
        spare = {f[2] for f in uo.free_flags} | {f[2] for f in uo.pending_flags}
        assert not set(flags) & spare, f"{label}: a buffer's flag is free"

    stages = [state()]
    released: list = [[], []]  # the idle buffers each growth step released
    deferred: list = []  # (user object, semaphore) of each idle buffer the FIRST growth released; CUDA gets it later

    def release_later(obj, sem: int) -> None:
        deferred.append((obj, sem))
        uo.retired.append(sem)

    calls: list = []  # (semaphore, whether it was in uo.retired right after the call): the SECOND growth's releases
    drained: set = set()  # the semaphores a drain moved from uo.retired to uo.free_sems (it took their post)

    def release_watched(obj, sem: int, release=uo.release) -> None:
        release(obj, sem)  # the package's own release
        # recorded here and asserted after the step: the refill's ``except Exception`` would catch an assert here
        calls.append((sem, sem in uo.retired))

    def drain_watched(objects, device_index: int, drain=pkg._pool_drain):
        before = list(objects.retired)
        mine = drain(objects, device_index)
        kept, free = set(objects.retired), set(objects.free_sems)
        drained.update(sem for sem in before if sem not in kept and sem in free)
        return mine

    with pytest.MonkeyPatch.context() as watch:
        seen = _record_programs(watch)
        watch.setattr(pkg, "_pool_drain", drain_watched)
        # The eager steps' programs launched once first, on this thread and device and on caller-owned workspaces (no
        # pool refill): a program's first launch in the process may synchronize the device (CUDA's lazy module loading,
        # a binding's first call per thread and device), which would wait for the held stream.  The steps in the hold
        # then launch only programs launched here (asserted), and any other host wait for the held stream fails the
        # ``waited`` check right after them.
        for (S, C, _), (x, w, cu, dy, b), out in zip(cases[1:], inputs[1:], outs[1:], strict=True):
            _forward(x, w, cu, bias=b, out=out["y"][0],
                     workspace=pkg.cake_causal_conv1d_nondet_train_forward_workspace(S, C, device))
            _backward(x, w, cu, dy, b, ROUTE_SETTINGS[kind]["policy"], kind=kind,
                      workspace=pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device),
                      dx=out["dx"][0], dweight=out["dweight"][0], dbias=out["dbias"][0] if out["dbias"] else None)
        torch.cuda.synchronize(device)
        warm = {name for name, _, _ in seen}
        seen.clear()
        assert state() == stages[0], "an eager call with caller-owned workspaces changed the pool"
        stream_b.wait_stream(torch.cuda.current_stream(device))
        try:
            with _stream_held(stream_b) as waited:
                with torch.cuda.stream(stream_b):
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(uo, "release", release_later)  # the first growth's releases: deferred
                        _phase_step("step", inputs[1], outs[1], kind)
                    stages.append(state())
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(uo, "release", release_watched)  # the second growth's: the package's release
                        _phase_step("step", inputs[2], outs[2], kind)
                    stages.append(state())
                assert not waited.is_set(), "the host waited for the held stream (a call that synchronizes the device)"
                assert not stream_b.query(), "the held stream ran"
                in_hold = {name for name, _, _ in seen}
                assert in_hold and in_hold <= warm, f"a program first launched inside the hold: {in_hold - warm}"
                pending = {f[2] for f in uo.pending_flags}
                free = {f[2] for f in uo.free_flags}
                check_unique("after the growth")
                for step, ((old, old_size), (new, new_size)) in enumerate(zip(stages, stages[1:])):
                    assert new_size["bwd"] > old_size["bwd"], (old_size, new_size)
                    for k in kinds:
                        assert len(new[k]) == pkg._POOL_DEPTH and all(b.tensor.numel() >= new_size[k] for b in new[k])
                        if new_size[k] > old_size[k]:
                            stayed = {id(b) for b in old[k]} & {id(b) for b in new[k]}
                            assert not stayed, f"{k}: a smaller buffer stayed idle"
                            released[step] += old[k]
                for k in kinds:
                    assert pkg._POOL_READY[k][d] == stages[-1][1][k], (k, pkg._POOL_READY[k][d])
                # the second growth released the first growth's buffers before their readiness copies ran
                for buf in stages[1][0]["bwd"]:
                    assert buf.flag[2] in pending and buf.flag[2] not in free, "a flag whose copy had not run was freed"
                # the first growth's semaphores are not posted yet: the later drains kept them retired, no new buffer
                # took one
                deferred_sems = {sem for _, sem in deferred}
                assert deferred_sems == {buf.sem for buf in released[0]}, "a first-growth release was not deferred"
                assert deferred_sems <= set(uo.retired), "a drain recycled a semaphore that was not posted"
                assert not deferred_sems & live_sems(), "a new buffer took a semaphore that was not posted"
                # the second growth dropped each idle buffer through the package's release; each call put the
                # semaphore in uo.retired at once, and no drain ran after the backward refill's releases
                real_sems = {buf.sem for buf in released[1]}
                assert real_sems and not real_sems & deferred_sems, (real_sems, deferred_sems)
                assert sorted(sem for sem, _ in calls) == sorted(real_sems), "an idle buffer left without a release"
                accounted = set(uo.retired) | set(uo.free_sems) | live_sems()
                assert real_sems <= accounted, "a released idle buffer's semaphore was lost (no release)"
                assert all(retired for _, retired in calls), "a release did not retire its semaphore"
                bwd_sems = {buf.sem for buf in released[1] if buf.kind == "bwd"}
                assert bwd_sems and bwd_sems <= set(uo.retired), "a backward release's semaphore left uo.retired"
        finally:
            for obj, _ in deferred:
                uo.cudart.cudaUserObjectRelease(obj, 1)
    torch.cuda.synchronize(device)
    released_sems = deferred_sems | real_sems
    deadline = time.monotonic() + 10.0
    # CUDA ran the destructor of every released idle object: its semaphore is posted, or a drain in the hold took the
    # post (it moved the semaphore from uo.retired to uo.free_sems)
    while not all(_posted(sem) or sem in drained for sem in released_sems):
        assert time.monotonic() < deadline, "CUDA did not release the released buffers' user objects"
        time.sleep(0.001)
    second = _capture(lambda: _phase_step("step", inputs[2], outs[2], kind), stream_b, debug=False)
    assert second["fallbacks"] == {"fwd": 0, "bwd": 0} and len(second["taken"]) == 2, second["fallbacks"]
    with torch.cuda.stream(stream_b):
        _phase_step("step", inputs[2], _phase_outs("step", inputs[2][0], inputs[2][1], inputs[2][4], kind), kind)
    torch.cuda.synchronize(device)
    assert not released_sems & set(uo.retired), "a posted semaphore stayed retired"
    assert released_sems <= set(uo.free_sems) | live_sems(), "a drain lost a posted semaphore"
    assert not uo.pending_flags, "a flag whose copy completed stayed pending"
    check_unique("after the refill")
    taken = first["taken"] + second["taken"]
    assert len({t.data_ptr() for t in taken}) == 4, "two graphs hold one buffer"
    idle = [buf.tensor for p in pkg._ZEROED_POOL.values() for buf in p]
    assert not any(t is h for t in idle for h in taken), "a buffer a graph holds is idle"
    graphs = ((first, inputs[0], outs[0], cases[0], 0), (second, inputs[2], outs[2], cases[2], 2))

    def launch(_draw: int, seed: int, graphs=graphs, taken=taken):  # bound here: G1 is deleted below
        fresh = [_draw_inputs(*case, kind, key=145 + i, seed=seed) for _, _, _, case, i in graphs]
        for (_, held, out, _, _), values in zip(graphs, fresh, strict=True):
            _load(held, values, out)
        for (cap, *_), stream in zip(graphs, (stream_a, stream_b), strict=True):
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                cap["graph"].replay()
        torch.cuda.synchronize(device)
        assert all(int(torch.count_nonzero(t)) == 0 for t in taken), "a capture buffer is not zero"
        return [
            (_outputs(out, j), problem(x, w, cu, b, dy))
            for j, ((_, _, out, _, _), (x, w, cu, dy, b)) in enumerate(zip(graphs, fresh, strict=True))
        ]

    assert_judged(launch, seed=SEED, label="graphs before and after the pool growth at once")
    # G1 deleted after the growth: the next drain drops its buffers that are smaller than the grown size
    g1 = {buf.kind: buf for buf in pkg._GRAPH_HELD if any(buf.tensor is t for t in first["taken"])}
    assert sorted(g1) == sorted(kinds), sorted(g1)
    smaller = {k: buf.tensor.numel() < pkg._POOL_SIZE[(d, k)] for k, buf in g1.items()}
    assert smaller["bwd"], "G1's backward buffer is not smaller than the grown pool size"
    old_sems = {buf.sem for buf in g1.values()}
    old_refs = {k: weakref.ref(buf.tensor) for k, buf in g1.items()}
    del g1, launch, graphs, taken, idle, first
    gc.collect()
    torch.cuda.synchronize(device)
    deadline = time.monotonic() + 10.0
    while not all(_posted(sem) for sem in old_sems):
        assert time.monotonic() < deadline, "CUDA did not release G1's buffers"
        time.sleep(0.001)
    third = _capture(lambda: _phase_step("step", inputs[2], outs[2], kind), stream_b, debug=False)
    assert third["fallbacks"] == {"fwd": 0, "bwd": 0} and len(third["taken"]) == 2, third["fallbacks"]
    with torch.cuda.stream(stream_b):
        _phase_step("step", inputs[2], _phase_outs("step", inputs[2][0], inputs[2][1], inputs[2][4], kind), kind)
    torch.cuda.synchronize(device)
    gc.collect()
    held = [buf.tensor for buf in pkg._GRAPH_HELD] + [buf.tensor for buf in uo.released]
    for k, ref in old_refs.items():
        if smaller[k]:
            assert ref() is None, f"{k}: the package kept a released buffer smaller than the pool size"
        else:  # a released buffer of the pool size returns to the idle pool
            assert any(buf.tensor is ref() for buf in pkg._ZEROED_POOL[(d, k)]), k
        assert not any(t is ref() for t in held), f"{k}: G1's released buffer was not drained"
    assert not old_sems & set(uo.retired) and old_sems <= set(uo.free_sems) | live_sems(), "G1's semaphores"
    check_unique("after G1's buffers were drained")


_TREE_OPS: dict = {}


def _tree_op():
    """A custom op (opaque to ``torch.compile``) running a forward and a backward of the package without a caller
    workspace: ``(y, dx, dweight)``."""

    if "op" not in _TREE_OPS:

        @torch.library.custom_op("cake_conv1d_nondet_gt::fwd_bwd", mutates_args=())
        def fwd_bwd(
            x: torch.Tensor, w: torch.Tensor, cu: torch.Tensor, dy: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _TREE_OPS["capturing"].append(torch.cuda.is_current_stream_capturing())
            y = _forward(x, w, cu)
            dx, dw, _ = _backward(x, w, cu, dy)
            return y, dx, dw

        @fwd_bwd.register_fake
        def _(x, w, cu, dy):
            return torch.empty_like(x), torch.empty_like(x), torch.empty_like(w)

        _TREE_OPS["op"] = fwd_bwd
        _TREE_OPS["capturing"] = []
    return _TREE_OPS["op"]


def test_cudagraph_tree_warm_up_and_replays(fresh_pool, monkeypatch) -> None:
    """``torch.compile(backend="cudagraphs")`` (a CUDA Graph tree) of a forward and a backward without a caller
    workspace, with no per-stream workspace made yet (as in a fresh process): the tree's warm-up runs the eager
    calls, which make the per-stream workspaces and the pre-zeroed buffers in the package's own memory pool, so the
    tree finds no tensor of its private pool other than its outputs (a per-stream workspace of the caching allocator
    would be one: the tree refuses it, and its replays would reuse that memory); the recording takes pre-zeroed
    buffers (no fallback); every call (warm-up, recording, replays) is judged against the naive tier."""

    import torch._dynamo

    monkeypatch.setattr(pkg, "_FORWARD_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_BACKWARD_WORKSPACES", {})
    op = _tree_op()
    _TREE_OPS["capturing"].clear()
    kind, (S, C, docs), device = "bwd", GRAPH_CASES[2], _device()
    torch._dynamo.reset()
    compiled = torch.compile(lambda x, w, cu, dy: op(x, w, cu, dy), backend="cudagraphs")
    before, held = _fallbacks(device), len(pkg._GRAPH_HELD)

    def launch(_draw: int, seed: int):
        x, w, cu, dy, b = _draw_inputs(S, C, docs, kind, key=141, seed=seed)
        y, dx, dw = compiled(x, w, cu, dy)
        torch.cuda.synchronize(device)
        return {"y": y.clone(), "dx": dx.clone(), "dweight": dw.clone()}, problem(x, w, cu, b, dy)

    try:
        assert_judged(launch, seed=SEED, label="cudagraph tree")
    finally:
        torch._dynamo.reset()
    assert True in _TREE_OPS["capturing"] and False in _TREE_OPS["capturing"], _TREE_OPS["capturing"]
    assert _fallbacks(device) == before
    assert len(pkg._GRAPH_HELD) > held


# The graph partitions of the tree test below: more than the pool holds (``_POOL_DEPTH`` + 2, asserted there).
TREE_NODES = 6


def _tree_nodes(op):
    """A function of :data:`TREE_NODES` calls of ``op``, each its own graph partition (a graph break after each)."""

    def nodes(x, w, cu, dy) -> tuple:
        r0 = op(x, w, cu, dy)
        torch._dynamo.graph_break()
        r1 = op(x, w, cu, dy)
        torch._dynamo.graph_break()
        r2 = op(x, w, cu, dy)
        torch._dynamo.graph_break()
        r3 = op(x, w, cu, dy)
        torch._dynamo.graph_break()
        r4 = op(x, w, cu, dy)
        torch._dynamo.graph_break()
        r5 = op(x, w, cu, dy)
        return r0, r1, r2, r3, r4, r5

    return nodes


def test_cudagraph_tree_with_more_backward_nodes_than_the_pool(fresh_pool, monkeypatch) -> None:
    """A CUDA Graph tree (``torch.compile``, backend ``"cudagraphs"``) of :data:`TREE_NODES` graph partitions, each a
    forward and a backward of the package without a caller workspace, more than the pool holds: the tree records the
    partitions back to back with no eager call between them, so the later ones find no pre-zeroed buffer and zero
    their own (fallbacks of both kinds).  The package keeps no reference to those, so the tree's pool holds only the
    tensors it tracks and every later recording and replay goes through; every partition's outputs on every draw are
    judged against the naive tier, and afterwards the package's tables hold only pre-zeroed buffers."""

    import torch._dynamo

    assert TREE_NODES >= pkg._POOL_DEPTH + 2, (TREE_NODES, pkg._POOL_DEPTH)
    monkeypatch.setattr(pkg, "_FORWARD_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_BACKWARD_WORKSPACES", {})
    op = _tree_op()
    _TREE_OPS["capturing"].clear()
    kind, (S, C, docs), device = "bwd", GRAPH_CASES[2], _device()
    torch._dynamo.reset()
    compiled = torch.compile(_tree_nodes(op), backend="cudagraphs")
    before = _fallbacks(device)

    def launch(_draw: int, seed: int):
        x, w, cu, dy, b = _draw_inputs(S, C, docs, kind, key=143, seed=seed)
        results = compiled(x, w, cu, dy)
        torch.cuda.synchronize(device)
        assert len(results) == TREE_NODES, len(results)
        found = {
            name: [r[i].clone() for r in results] for i, name in enumerate(("y", "dx", "dweight"))
        }
        return found, problem(x, w, cu, b, dy)

    try:
        assert_judged(launch, seed=SEED, label="cudagraph tree with more backward nodes than the pool")
    finally:
        torch._dynamo.reset()
    after = _fallbacks(device)
    assert True in _TREE_OPS["capturing"], _TREE_OPS["capturing"]
    assert after["fwd"] > before["fwd"] and after["bwd"] > before["bwd"], (before, after)
    _assert_tables_hold_only_pool_buffers()


# The shape of the PDL tests (each route's forward and backward take package workspaces there).
PDL_CASE = (1000, 512, "perf")
# GPU clock cycles of the ``torch.cuda._sleep`` in front of a PDL test's producer or reader kernels, so the package's
# launch is queued while they wait to run.
PDL_SLEEP_CYCLES = 20_000_000
# Calls of each back-to-back chain of :func:`test_pdl_back_to_back_launches_share_the_stream_workspace`.
PDL_CHAIN = 64
# The bit pattern the outputs hold before a PDL reader test's launch (exact in BF16 and FP32).
SENTINEL = -7.5
# ``eager``: eager calls with the policy forced on; ``graph``: captured calls with the policy forced on in capture;
# ``graph_shipped``: captured calls with the shipped policy (``PDL_GRAPH``).
PDL_MODES = ("eager", "graph", "graph_shipped")


def _pdl_mode(mode: str, monkeypatch) -> int:
    """Set the package's PDL policy of this architecture for ``mode`` and return the ``enable_pdl`` value the
    package's launches of that mode pass."""

    arch = _arch()
    if mode != "graph_shipped":
        monkeypatch.setitem(loader.PDL_EAGER, arch, True)
        monkeypatch.setitem(loader.PDL_GRAPH, arch, True)
    return loader.pdl_enabled(arch, capturing=mode != "eager")


def _record_pdl(monkeypatch) -> list:
    """Every launch of the package from now on appends ``(module name, enable_pdl value)``."""

    seen = []
    launch = pkg.run_module

    def record(name, values, *, device_index, grid=None):
        seen.append((name, values.get("enable_pdl")))
        return launch(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record)
    return seen


def _assert_pdl(seen: list, expected: int, label: str) -> None:
    """Every recorded launch is of a record with ``launch.use_pdl`` ``"runtime"`` and passed ``expected``."""

    assert seen, label
    for name, value in seen:
        assert loader.MODULES[name]["launch"]["use_pdl"] == "runtime", (label, name)
        assert value == expected, (label, name, value, expected)


def _pdl_run(mode: str, bodies: list, load, outputs, check, case, kind: str, *, key: int, seen: list, label: str):
    """Judge ``bodies`` (callables queuing the test's kernels on the current stream; draw ``j`` runs
    ``bodies[j % len(bodies)]``) in ``mode`` over the draws: ``load(fresh)`` copies a draw's inputs in, then the
    body runs eagerly (``eager``) or replays its graph, captured once on a side stream after a warm-up; after the
    synchronization ``check()`` runs and ``outputs()`` is judged against the naive tier on the draw's problem.  The
    package's launches of the mode pass the mode's ``enable_pdl`` value; returns the graphs' edge attribute counts."""

    device = _device()
    S, C, docs = case
    graphs, edges = [], {}
    load(_draw_inputs(S, C, docs, kind, key=key))
    for body in bodies if mode != "eager" else ():
        seen.clear()
        cap = _capture(body, torch.cuda.Stream(device=device))
        n = len(seen) // 2
        _assert_pdl(seen[n:], loader.pdl_enabled(_arch(), capturing=True), f"{label} capture")
        for text, count in _graph_nodes(cap["graph"])["edges"].items():
            edges[text] = edges.get(text, 0) + count
        graphs.append(cap)
    seen.clear()

    def launch(draw: int, seed: int):
        fresh = _draw_inputs(S, C, docs, kind, key=key, seed=seed)
        load(fresh)
        if mode == "eager":
            bodies[draw % len(bodies)]()
        else:
            graphs[draw % len(graphs)]["graph"].replay()
        torch.cuda.synchronize(device)
        check()
        if mode != "eager":
            for cap in graphs:
                for t in cap["buffers"].values():
                    assert int(torch.count_nonzero(t)) == 0, f"{label}: a capture buffer is not zero"
        x, w, cu, dy, b = fresh
        return outputs(), problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label=label)
    if mode == "eager":
        _assert_pdl(seen, loader.pdl_enabled(_arch(), capturing=False), label)
    return edges


def _rotations(items: list) -> list[list]:
    """Each rotation of ``items`` (every item last once)."""

    return [items[i:] + items[:i] for i in range(1, len(items) + 1)]


@pytest.mark.parametrize("eager_on,graph_on", ((True, False), (False, True), (True, True), (False, False)))
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_enable_pdl_follows_the_policy(kind, eager_on, graph_on, monkeypatch) -> None:
    """Every forward and backward launch (all of them records with ``launch.use_pdl`` ``"runtime"``) passes
    ``enable_pdl`` = ``cake_jit.pdl_enabled(arch, capturing=)``: ``PDL_EAGER`` of the architecture when eager,
    ``PDL_GRAPH`` inside a CUDA Graph capture, for each setting of the two tables; the captured graph replays."""

    _require_route(kind)
    arch, device = _arch(), _device()
    monkeypatch.setitem(loader.PDL_EAGER, arch, eager_on)
    monkeypatch.setitem(loader.PDL_GRAPH, arch, graph_on)
    seen = _record_pdl(monkeypatch)
    inputs = _draw_inputs(*PDL_CASE, kind, key=151)
    outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], kind)
    cap = _capture(lambda: _phase_step("step", inputs, outs, kind), torch.cuda.Stream(device=device), debug=False)
    assert len(seen) == 4, seen
    _assert_pdl(seen[:2], int(eager_on), f"{kind} eager")
    _assert_pdl(seen[2:], int(graph_on), f"{kind} capture")
    cap["graph"].replay()
    torch.cuda.synchronize(device)


def _pdl_off_cases() -> tuple:
    """``(S, C)`` of the default backward route around its ``PDL_OFF_ROWS`` keys on this architecture: each key's last
    channel count at the key's first token count and one token below it; on an architecture without keys, C 1016 at
    S 250340."""

    cases = set()
    for (route, _c_lo, c_hi), (s_min, _s_max) in loader.PDL_OFF_ROWS.get(_arch(), {}).items():
        if route == "bwd":
            cases |= {(s_min - 1, c_hi), (s_min, c_hi)}
    return tuple(sorted(cases)) or ((250340, 1016),)


@pytest.mark.parametrize("eager_on,graph_on", ((True, True), (False, False)))
def test_enable_pdl_is_off_at_the_pdl_off_rows(eager_on, graph_on, monkeypatch) -> None:
    """At a row of ``PDL_OFF_ROWS`` (:func:`_pdl_off_cases`) the backward launch passes ``enable_pdl`` 0, eagerly and
    inside a CUDA Graph capture, whatever ``PDL_EAGER`` / ``PDL_GRAPH`` say; one token below the row, and at every
    forward launch, it passes the tables' value (``cake_jit.pdl_enabled`` of the call's row); the captured graph
    replays."""

    _require_route("bwd")
    arch, device = _arch(), _device()
    monkeypatch.setitem(loader.PDL_EAGER, arch, eager_on)
    monkeypatch.setitem(loader.PDL_GRAPH, arch, graph_on)
    seen = _record_pdl(monkeypatch)
    backward = set(loader.route_modules(arch, "bwd"))
    for S, C in _pdl_off_cases():
        bounds = loader.pdl_off_row_range(arch, "bwd", C)
        inside = bounds is not None and bounds[0] <= S <= bounds[1]
        inputs = _draw_inputs(S, C, "perf", "bwd", key=153)
        outs = _phase_outs("step", inputs[0], inputs[1], inputs[4], "bwd")
        seen.clear()
        cap = _capture(
            lambda: _phase_step("step", inputs, outs, "bwd"), torch.cuda.Stream(device=device), debug=False
        )
        assert [name in backward for name, _ in seen] == [False, True, False, True], seen
        for (name, value), capturing in zip(seen, (False, False, True, True), strict=True):
            table = int(graph_on if capturing else eager_on)
            want = 0 if name in backward and inside else table
            if name in backward:
                assert loader.pdl_enabled(arch, capturing=capturing, kind="bwd", C=C, S=S) == want
            assert value == want, (S, C, name, capturing, value, want)
        cap["graph"].replay()
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("mode", PDL_MODES)
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_pdl_launch_reads_what_the_producer_wrote(kind, mode, monkeypatch, record_property) -> None:
    """Producer kernels (``copy_`` of the draw's ``x``, ``w``, ``cu``, ``dy`` and bias into the call's inputs, which
    held NaN / other documents) queued behind a ``torch.cuda._sleep`` right before each forward and backward launch
    on the same stream, each producer the launch's immediate predecessor on some draw: the outputs judged against
    the naive tier on the draw's problem, eagerly and replayed (policy forced on, and the shipped graph policy)."""

    _pdl_producer(kind, mode, PDL_CASE, monkeypatch, record_property)


def _pdl_producer(kind: str, mode: str, case, monkeypatch, record_property) -> None:
    """The body of :func:`test_pdl_launch_reads_what_the_producer_wrote` at ``case`` ``(S, C, docs)``."""

    _require_route(kind)
    _pdl_mode(mode, monkeypatch)
    seen = _record_pdl(monkeypatch)
    S, C, docs = case
    src = list(_draw_inputs(S, C, docs, kind, key=161))
    names = ["x", "w", "cu", "dy", "b"]
    present = [n for n, t in zip(names, src, strict=True) if t is not None]
    fwd_held = {n: torch.empty_like(t) for n, t in zip(names, src, strict=True) if t is not None and n != "dy"}
    bwd_held = {n: torch.empty_like(t) for n, t in zip(names, src, strict=True) if t is not None}
    source = dict(zip(names, src, strict=True))
    other_docs = torch.zeros_like(source["cu"])
    other_docs[-1] = S  # a valid cu of other documents
    outs = _phase_outs("step", src[0], src[1], src[4], kind)

    def stale(held: dict) -> None:
        for n, t in held.items():
            if n == "cu":
                t.copy_(other_docs)
            else:
                t.fill_(float("nan"))

    def body_for(order: list):
        def body():
            stale(fwd_held)
            stale(bwd_held)
            torch.cuda._sleep(PDL_SLEEP_CYCLES)
            for n in order:
                if n in fwd_held:
                    fwd_held[n].copy_(source[n])
            _forward(fwd_held["x"], fwd_held["w"], fwd_held["cu"], bias=fwd_held.get("b"), out=outs["y"][0])
            torch.cuda._sleep(PDL_SLEEP_CYCLES)
            for n in order:
                bwd_held[n].copy_(source[n])
            held = tuple(bwd_held.get(n) for n in names)
            _phase_step("bwd", held, {**outs, "y": []}, kind)

        return body

    def load(fresh):
        for t, v in zip(src, fresh, strict=True):
            if t is not None:
                t.copy_(v)
        _load((None,) * 5, fresh, outs)

    edges = _pdl_run(
        mode,
        [body_for(order) for order in _rotations(present)],
        load,
        lambda: _outputs(outs),
        lambda: None,
        case,
        kind,
        key=161,
        seen=seen,
        label=f"pdl producer {kind} {mode}",
    )
    record_property("graph_edges", edges)


@pytest.mark.parametrize("mode", PDL_MODES)
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_pdl_launch_does_not_overwrite_what_the_producer_reads(kind, mode, monkeypatch, record_property) -> None:
    """The outputs hold ``SENTINEL`` before each launch; reader kernels (``copy_`` of ``y`` / ``dx`` / ``dweight`` /
    ``dbias`` into snapshots) queued behind a ``torch.cuda._sleep`` right before the launch that writes them, each
    reader the launch's immediate predecessor on some draw: every snapshot equals the sentinel bitwise and the
    outputs are judged against the naive tier, eagerly and replayed (policy forced on, and the shipped graph
    policy)."""

    _pdl_reader(kind, mode, PDL_CASE, monkeypatch, record_property)


def _pdl_reader(kind: str, mode: str, case, monkeypatch, record_property) -> None:
    """The body of :func:`test_pdl_launch_does_not_overwrite_what_the_producer_reads` at ``case``."""

    _require_route(kind)
    _pdl_mode(mode, monkeypatch)
    seen = _record_pdl(monkeypatch)
    S, C, docs = case
    src = list(_draw_inputs(S, C, docs, kind, key=171))
    outs = _phase_outs("step", src[0], src[1], src[4], kind)
    grads = [n for n in ("dx", "dweight", "dbias") if outs[n]]
    snaps = {n: torch.empty_like(outs[n][0]) for n in ("y", *grads)}

    def body_for(order: list):
        def body():
            for n in ("y", *grads):
                outs[n][0].fill_(SENTINEL)
            torch.cuda._sleep(PDL_SLEEP_CYCLES)
            snaps["y"].copy_(outs["y"][0])
            _phase_step("fwd", tuple(src), {**outs, "dx": [], "dweight": [], "dbias": []}, kind)
            torch.cuda._sleep(PDL_SLEEP_CYCLES)
            for n in order:
                snaps[n].copy_(outs[n][0])
            _phase_step("bwd", tuple(src), {**outs, "y": []}, kind)

        return body

    def load(fresh):
        for t, v in zip(src, fresh, strict=True):
            if t is not None:
                t.copy_(v)

    def check():
        for n, snap in snaps.items():
            assert torch.equal(snap, torch.full_like(snap, SENTINEL)), f"{kind} {mode}: {n} written before its reader"

    edges = _pdl_run(
        mode,
        [body_for(order) for order in _rotations(grads)],
        load,
        lambda: _outputs(outs),
        check,
        case,
        kind,
        key=171,
        seen=seen,
        label=f"pdl reader {kind} {mode}",
    )
    record_property("graph_edges", edges)


@pytest.mark.parametrize("mode", PDL_MODES)
@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_pdl_back_to_back_launches_share_the_stream_workspace(kind, mode, monkeypatch) -> None:
    """Chains of ``PDL_CHAIN`` calls with no host synchronization and no caller workspace: forward after forward,
    backward after backward, and forward / backward alternating, each call into outputs of its own: every output
    judged against the naive tier, and the package's workspaces (per-stream when eager, the capture's buffers when
    replayed) read back zero after the chain (policy forced on, and the shipped graph policy)."""

    _pdl_chain(kind, mode, PDL_CASE, monkeypatch)


def _pdl_chain(kind: str, mode: str, case, monkeypatch) -> None:
    """The body of :func:`test_pdl_back_to_back_launches_share_the_stream_workspace` at ``case``."""

    _require_route(kind)
    _pdl_mode(mode, monkeypatch)
    seen = _record_pdl(monkeypatch)
    device = _device()
    S, C, docs = case
    src = tuple(_draw_inputs(S, C, docs, kind, key=181))
    x, w, b = src[0], src[1], src[4]
    chains = [_phase_outs("fwd", x, w, b, kind) for _ in range(PDL_CHAIN)]
    chains += [_phase_outs("bwd", x, w, b, kind) for _ in range(PDL_CHAIN)]
    chains += [_phase_outs(p, x, w, b, kind) for _ in range(PDL_CHAIN // 2) for p in ("fwd", "bwd")]
    phases = ["fwd"] * PDL_CHAIN + ["bwd"] * PDL_CHAIN + ["fwd", "bwd"] * (PDL_CHAIN // 2)

    def body():
        for phase, outs in zip(phases, chains, strict=True):
            _phase_step(phase, src, outs, kind)

    def load(fresh):
        for t, v in zip(src, fresh, strict=True):
            if t is not None:
                t.copy_(v)
        for outs in chains:
            _load((None,) * 5, fresh, outs)

    def outputs():
        found: dict = {}
        for outs in chains:
            for name, values in outs.items():
                found.setdefault(name, []).extend(values)
        return _outputs(found)

    def check():
        if mode == "eager":
            for held in (_stream_workspace(device), _forward_stream_workspace(device)):
                assert held is not None and int(torch.count_nonzero(held)) == 0, f"{kind}: a workspace is not zero"

    _pdl_run(mode, [body], load, outputs, check, case, kind, key=181, seen=seen, label=f"pdl chain {kind} {mode}")


# The PDL tests' row at a ``FALLBACK_ROWS`` key of the default backward route (inside its token range).
PDL_FALLBACK_CASE = (4097, 768, FALLBACK_ROW_DOCS)


@pytest.mark.parametrize("mode", PDL_MODES)
@pytest.mark.parametrize("check", ("producer", "reader", "chain"))
def test_fallback_row_pdl(check, mode, monkeypatch, record_property) -> None:
    """The three PDL tests above (``producer``: a launch reads what the kernel before it wrote; ``reader``: it does
    not overwrite what that kernel reads; ``chain``: back-to-back launches share the package's workspaces) on the
    default backward route at :data:`PDL_FALLBACK_CASE`, inside its ``FALLBACK_ROWS`` key: the backward launches the
    expected program (the route's fallback program on sm_100a / sm_103a), eagerly and inside captures with the policy
    forced on and with the shipped graph policy, every output judged against the naive tier."""

    S, C, _ = PDL_FALLBACK_CASE
    _fallback_row_bounds(C)
    seen = _record_programs(monkeypatch)
    if check == "producer":
        _pdl_producer("bwd", mode, PDL_FALLBACK_CASE, monkeypatch, record_property)
    elif check == "reader":
        _pdl_reader("bwd", mode, PDL_FALLBACK_CASE, monkeypatch, record_property)
    else:
        _pdl_chain("bwd", mode, PDL_FALLBACK_CASE, monkeypatch)
    assert _backward_programs(seen) == {_expected_backward_program(S, C)}, set(seen)


# ----------------------------------------------------------------------------------------------------------------
# The rows of ``SM_WIDE_ROWS``: route ``bwd`` runs an SM-wide program (one CTA per SM for the whole launch).


def _sm_wide_rows() -> tuple:
    """``(S, C)`` of the device's ``SM_WIDE_ROWS`` key: its lowest, an inner and its highest channel count, each at
    one token below the first segment, at every segment's first token count and at 250340 tokens."""

    c_lo, c_hi, segments = _sm_wide_key(_arch())
    tokens = sorted({int(segments[0][0]) - 1, *(int(s_lo) for s_lo, _, _ in segments), 250340})
    return tuple((S, C) for C in (c_lo, _sm_wide_mid(c_lo, c_hi), c_hi) for S in tokens)


def test_sm_wide_rows_match_reference(monkeypatch) -> None:
    """At every row of :func:`_sm_wide_rows` the default backward route launches the expected program (the segment's
    SM-wide program inside a segment, the route's fallback or the rules' program one token below the key's first
    segment), on a caller-owned workspace of the public size and on the package's, each zero after the launch;
    ``dx`` / ``dweight`` of both launches judged against the naive tier."""

    _require_route("bwd")
    device = _device()
    seen = _record_programs(monkeypatch)
    sm_wide = set()
    for S, C in _sm_wide_rows():
        expected = _expected_backward_program(S, C)
        if loader.sm_wide_module(_arch(), "bwd", C, S) is not None:
            sm_wide.add(expected)
        workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
        label = f"bwd S={S} C={C} program={expected}"

        def call(x, w, cu, dy, b, policy, workspace=workspace, expected=expected, label=label):
            found = {}
            for own in (True, False):
                seen.clear()
                run = _backward(x, w, cu, dy, b, policy, kind="bwd", **({"workspace": workspace} if own else {}))
                assert _backward_programs(seen) == {expected}, f"{label} own={own}: {seen}"
                held = workspace if own else _stream_workspace(device)
                assert held is not None and int(torch.count_nonzero(held)) == 0, f"{label} own={own}"
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, "perf", "bwd", call, key=211, label=label)
    _, _, segments = _sm_wide_key(_arch())
    assert sm_wide == {name for _, _, name in segments}, sm_wide


def test_sm_wide_row_repeated_launches(monkeypatch, record_property) -> None:
    """The repeated-launch rows at the first token count of every ``SM_WIDE_ROWS`` segment (an inner channel count
    of the key; on sm_107a the two-copy program from 131072 tokens on the package-sized workspace): the segment's
    program, every launch judged against the naive tier and the workspace zero after each."""

    _require_route("bwd")
    seen = _record_programs(monkeypatch)
    for case in _sm_wide_cases():
        seen.clear()
        _repeated_launches(case, "bwd", record_property)
        assert _backward_programs(seen) == {loader.sm_wide_module(_arch(), "bwd", case[1], case[0])}, (case, seen)


def test_sm_wide_row_bwd_grid_mult(monkeypatch) -> None:
    """Every ``bwd_grid_mult`` (1 to 4) at the first token count of every ``SM_WIDE_ROWS`` segment, on a caller-owned
    workspace (allocated at that multiple) and on the package's: the segment's program launches ``bwd_grid_mult``
    times its one-CTA-per-SM grid (the CTAs past the first wave run as the earlier ones finish), every launch's
    ``dx`` / ``dweight`` judged against the naive tier and the workspace zero after every launch."""

    _require_route("bwd")
    device = _device()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, grid))
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for S, C, docs in _sm_wide_cases():
        name = loader.sm_wide_module(_arch(), "bwd", C, S)
        workspaces = {
            mult: pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device, bwd_grid_mult=mult)
            for mult in pkg.BWD_GRID_MULTS
        }

        def call(x, w, cu, dy, b, policy, S=S, C=C, name=name, workspaces=workspaces):
            found = {}
            for mult in pkg.BWD_GRID_MULTS:
                for own in (True, False):
                    launched.clear()
                    kwargs = {"workspace": workspaces[mult]} if own else {}
                    run = _backward(x, w, cu, dy, b, policy, kind="bwd", bwd_grid_mult=mult, **kwargs)
                    label = f"bwd S={S} C={C} bwd_grid_mult={mult} own={own}"
                    ((launched_name, grid),) = launched
                    assert launched_name == name and grid[0] == mult * sms, (label, launched)
                    held = workspaces[mult] if own else _stream_workspace(device)
                    assert held is not None and int(torch.count_nonzero(held)) == 0, label
                    for out, value in _grads(*run).items():
                        found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, docs, "bwd", call, key=213, label=f"bwd grid mults S={S} C={C}")


@pytest.mark.parametrize("bwd_grid_mult", (1, 2, 3, 4))
def test_sm_wide_row_graph_replays(bwd_grid_mult, fresh_pool, monkeypatch) -> None:
    """CUDA Graph capture / replay of forward + backward at the first token count of every ``SM_WIDE_ROWS`` segment
    at ``bwd_grid_mult`` 1 to 4, with a caller-owned workspace and with the package's pre-zeroed buffers (on sm_107a
    the two-copy program in the package-sized buffer): the segment's program per row, every replay judged against the
    naive tier, the workspaces zero after every replay and no buffer zeroed by a captured call."""

    _require_route("bwd")
    device = _device()
    cases = _sm_wide_cases()
    seen = _record_programs(monkeypatch)
    for own_workspace in (True, False):
        seen.clear()
        before = _fallbacks(device)
        replays = _capture_and_replay(cases, "bwd", key=215, own_workspace=own_workspace, bwd_grid_mult=bwd_grid_mult)
        assert replays == DRAWS * len(cases)
        assert _fallbacks(device) == before, (own_workspace, before, _fallbacks(device))
        expected = {(S, C): {_expected_backward_program(S, C)} for S, C, _ in cases}
        assert _backward_programs_by_shape(seen) == expected, (own_workspace, seen, expected)


def test_sm_wide_row_two_streams(monkeypatch) -> None:
    """The two-stream test at the first token count of every ``SM_WIDE_ROWS`` segment: the segment's program on both
    streams, both calls judged against the naive tier on every draw, every workspace zero afterwards."""

    _require_route("bwd")
    seen = _record_programs(monkeypatch)
    for S, C, docs in _sm_wide_cases():
        seen.clear()
        _two_streams(S, C, docs)
        assert _backward_programs(seen) == {_expected_backward_program(S, C)}, (S, C, set(seen))


@pytest.mark.parametrize("segment", SM_WIDE_SEGMENTS, ids=[f"seg{index}" for index in SM_WIDE_SEGMENTS])
@pytest.mark.parametrize("mode", PDL_MODES)
@pytest.mark.parametrize("check", ("producer", "reader", "chain"))
def test_sm_wide_row_pdl(check, mode, segment, monkeypatch, record_property) -> None:
    """The three PDL tests (``producer``, ``reader``, ``chain``) on the default backward route at the first token
    count of every segment of the device's ``SM_WIDE_ROWS`` key (one test per segment index): the backward launches
    the segment's program, eagerly and inside captures with the policy forced on and with the shipped graph policy,
    every output judged against the naive tier."""

    arch = _arch()
    cases = _sm_wide_cases()
    if segment >= len(cases):
        pytest.skip(f"{arch}: its SM_WIDE_ROWS key of route bwd has {len(cases)} segments")
    case = cases[segment]
    S, C, _ = case
    _, _, segments = _sm_wide_key(arch)
    assert loader.sm_wide_module(arch, "bwd", C, S) == segments[segment][2], (arch, segment, S, C)
    seen = _record_programs(monkeypatch)
    if check == "producer":
        _pdl_producer("bwd", mode, case, monkeypatch, record_property)
    elif check == "reader":
        _pdl_reader("bwd", mode, case, monkeypatch, record_property)
    else:
        _pdl_chain("bwd", mode, case, monkeypatch)
    assert _backward_programs(seen) == {_expected_backward_program(S, C)}, set(seen)


def test_sm_wide_row_autograd_and_checkpoint(monkeypatch) -> None:
    """The autograd entry at the first token count of every ``SM_WIDE_ROWS`` segment, plain and under non-reentrant
    checkpointing (2 x forward + backward), at ``bwd_grid_mult`` 1 and 2: the segment's program runs every backward,
    ``y`` and the gradients judged against the naive tier."""

    from torch.utils.checkpoint import checkpoint

    _require_route("bwd")
    seen = _record_programs(monkeypatch)
    for S, C, docs in _sm_wide_cases():
        name = loader.sm_wide_module(_arch(), "bwd", C, S)

        def call(x0, w0, cu, dy, b0, policy, name=name, S=S, C=C):
            found = {}
            for mult in (1, 2):
                for ckpt in (False, True):
                    seen.clear()
                    x = x0.detach().clone().requires_grad_(True)
                    w = w0.detach().clone().requires_grad_(True)

                    def layer(x, w, mult=mult):
                        return pkg.cake_causal_conv1d_nondet(x, w, cu, None, silu_grad_preactivation=policy,
                                                             bwd_grid_mult=mult)  # fmt: skip

                    y = checkpoint(layer, x, w, use_reentrant=False) if ckpt else layer(x, w)
                    y.backward(dy)
                    assert _backward_programs(seen) == {name}, (S, C, mult, ckpt, seen)
                    for out, value in {"y": y.detach(), **_grads(x.grad, w.grad, None)}.items():
                        found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, docs, "bwd", call, key=217, label=f"autograd sm-wide S={S} C={C}")


# A kernel of one CTA that holds as much of its SM's shared memory as a block may take (opt-in) and spins for
# ``cycles`` clock cycles: no other CTA that takes shared memory fits on that SM while it runs.  PTX, loaded by the
# driver (no compiler needed).
_SMEM_HOLD_PTX = r"""
.version 8.0
.target sm_90
.address_size 64
.extern .shared .align 16 .b8 cake_test_hold_smem[];
.visible .entry cake_test_hold_sm(.param .u64 cycles)
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [cycles];
    mov.u64 %rd2, %clock64;
    mov.u32 %r1, 1;
    st.shared.u32 [cake_test_hold_smem], %r1;
$L_spin:
    mov.u64 %rd3, %clock64;
    sub.s64 %rd4, %rd3, %rd2;
    setp.lt.s64 %p1, %rd4, %rd1;
    @%p1 bra $L_spin;
    ret;
}
"""
# The clock cycles the shared-memory side kernel spins (about 10 ms): longer than any backward it runs beside.
HOLD_CYCLES = 20_000_000
_SMEM_HOLD: dict = {}


def _smem_hold_kernel():
    """``(module, function, shared bytes)`` of the shared-memory side kernel on the current device (loaded once)."""

    from cuda.bindings import driver

    device = _device()
    if device.index in _SMEM_HOLD:
        return _SMEM_HOLD[device.index]
    torch.cuda.synchronize(device)
    err, module = driver.cuModuleLoadData(_SMEM_HOLD_PTX.encode() + b"\0")
    assert err == driver.CUresult.CUDA_SUCCESS, f"cuModuleLoadData: {err}"
    err, function = driver.cuModuleGetFunction(module, b"cake_test_hold_sm")
    assert err == driver.CUresult.CUDA_SUCCESS, f"cuModuleGetFunction: {err}"
    err, cu_device = driver.cuDeviceGet(device.index)
    assert err == driver.CUresult.CUDA_SUCCESS, f"cuDeviceGet: {err}"
    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    err, nbytes = driver.cuDeviceGetAttribute(attribute, cu_device)
    assert err == driver.CUresult.CUDA_SUCCESS and nbytes >= 40 * 1024, (err, nbytes)
    (err,) = driver.cuFuncSetAttribute(
        function, driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(nbytes)
    )
    assert err == driver.CUresult.CUDA_SUCCESS, f"cuFuncSetAttribute: {err}"
    _SMEM_HOLD[device.index] = (module, function, int(nbytes))
    return _SMEM_HOLD[device.index]


def _hold_one_sm(stream: torch.cuda.Stream) -> None:
    """Launch the shared-memory side kernel (one CTA of 32 threads, :data:`HOLD_CYCLES`) on ``stream``."""

    from cuda.bindings import driver

    _, function, nbytes = _smem_hold_kernel()
    (err,) = driver.cuLaunchKernel(
        function, 1, 1, 1, 32, 1, 1, nbytes, stream.cuda_stream, ((HOLD_CYCLES,), (ctypes.c_uint64,)), 0
    )
    assert err == driver.CUresult.CUDA_SUCCESS, f"cuLaunchKernel: {err}"


# The row of the shared-memory side-kernel test: an SM-wide row of every architecture (its last segment).
SIDE_KERNEL_CASE = (261393, 576, "perf")
# The SM-wide backward beside the side kernel takes at most this many times its time alone (the CTA left without a
# free SM makes at most a second wave).
SIDE_KERNEL_BUSY_BOUND = 2.0


def test_backward_finishes_beside_a_kernel_holding_one_sms_shared_memory(monkeypatch, record_property) -> None:
    """A kernel that holds one SM's shared memory on another stream (one CTA, the block opt-in maximum, about 10 ms)
    leaves one CTA of the one-CTA-per-SM program at :data:`SIDE_KERNEL_CASE` without a free SM until another CTA ends:
    the backward still finishes while the side kernel runs (it waits for no SM the side kernel holds), on the SM-wide
    program and on the program the rules pick there (``SM_WIDE_ROWS`` emptied).  The time beside the side kernel over
    the time alone (medians of 7 after 3 warm-ups, CUDA events on the current stream) of both is recorded, and the
    SM-wide program's is at most :data:`SIDE_KERNEL_BUSY_BOUND`; the outputs of backward calls launched beside it are
    judged against the naive tier."""

    _require_route("bwd")
    device = _device()
    S, C, docs = SIDE_KERNEL_CASE
    arch = _arch()
    sm_wide = loader.sm_wide_module(arch, "bwd", C, S)
    assert sm_wide is not None, f"{arch}: {SIDE_KERNEL_CASE} is no SM_WIDE_ROWS row"
    x, w, cu, dy = _inputs(S, C, docs)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    dx = torch.empty_like(x)
    dw = torch.empty(C, WIDTH, dtype=torch.float32, device=device)
    side = torch.cuda.Stream(device=device)
    start, end, side_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    seen = _record_programs(monkeypatch)
    _smem_hold_kernel()

    def timed(beside: bool) -> tuple[float, float]:
        torch.cuda.synchronize(device)
        if beside:
            with torch.cuda.stream(side):
                _hold_one_sm(side)
                side_end.record()
        torch.cuda._sleep(1_000_000)
        start.record()
        _backward(x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end), (start.elapsed_time(side_end) if beside else 0.0)

    for program, tables in ((sm_wide, None), (None, {})):
        with monkeypatch.context() as m:
            if tables is not None:
                m.setitem(loader.SM_WIDE_ROWS, arch, tables)
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
            seen.clear()
            for _ in range(3):
                timed(False)
            alone = statistics.median(timed(False)[0] for _ in range(7))
            runs = [timed(True) for _ in range(7)]
            busy = statistics.median(run[0] for run in runs)
            launched = _backward_programs(seen)
            assert len(launched) == 1 and (program is None or launched == {program}), launched
            (name,) = launched
            tag = "sm_wide" if program is not None else "rules"
            print(f"{tag} {name}: alone {alone:.3f} ms, beside the shared-memory kernel {busy:.3f} ms "
                  f"({busy / alone:.2f}x)")  # fmt: skip
            record_property(f"{tag}_program", name)
            record_property(f"{tag}_alone_ms", f"{alone:.4f}")
            record_property(f"{tag}_beside_smem_kernel_ms", f"{busy:.4f}")
            record_property(f"{tag}_busy_over_alone", f"{busy / alone:.3f}")
            for backward_ms, side_ms in runs:
                assert backward_ms < side_ms, (
                    f"{tag} {name}: the backward ({backward_ms:.3f} ms) waited for the kernel holding one SM's shared "
                    f"memory ({side_ms:.3f} ms)"
                )
            if program is not None:
                assert busy <= SIDE_KERNEL_BUSY_BOUND * alone, (
                    f"{arch} {name}: beside the kernel holding one SM's shared memory the SM-wide backward took "
                    f"{busy / alone:.3f} times its time alone (bound {SIDE_KERNEL_BUSY_BOUND})"
                )
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()

    def beside(x, w, cu, dy, b, policy):
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            _hold_one_sm(side)
        torch.cuda._sleep(1_000_000)
        run = _backward(x, w, cu, dy, workspace=workspace)
        torch.cuda.synchronize(device)
        return _grads(*run)

    _judge(S, C, docs, beside, label="beside the shared-memory side kernel")


def test_det_and_nondet_sm_wide_programs_in_one_process(monkeypatch) -> None:
    """Both packages loaded in one process, their backward calls interleaved on one stream at :data:`SIDE_KERNEL_CASE`
    (an SM-wide row of this package and, past its token floor, of the deterministic package): this package launches
    its SM-wide program, the deterministic package its own program of that row; every output of both is judged
    against the naive tier (no comparison between them)."""

    _require_route("bwd")
    det = cake_package("cake_causal_conv1d_det_train")
    det_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")
    seen = _record_programs(monkeypatch)
    det_seen = []
    det_one, det_many = det.run_module, det.run_modules

    def det_record_one(name, values, *, device_index, grid=None):
        det_seen.append(name)
        return det_one(name, values, device_index=device_index, grid=grid)

    def det_record_many(launches, values, *, device_index):
        det_seen.extend(name for name, _ in launches)
        return det_many(launches, values, device_index=device_index)

    monkeypatch.setattr(det, "run_module", det_record_one)
    monkeypatch.setattr(det, "run_modules", det_record_many)
    S, C, docs = SIDE_KERNEL_CASE
    arch = _arch()
    name = loader.sm_wide_module(arch, "bwd", C, S)
    det_names = det_jit.selected_modules(arch, "bwd", C=C, S=S)

    def call(x, w, cu, dy, b, policy):
        found = {"dx": [], "dweight": []}
        for _ in range(2):
            seen.clear()
            det_seen.clear()
            dx, dw, _ = _backward(x, w, cu, dy)
            det_dx, det_dw = det.cake_causal_conv1d_det_train_backward(x, w, cu, dy)[:2]
            assert _backward_programs(seen) == {name}, seen
            assert det_seen[: len(det_names)] == list(det_names), (det_seen, det_names)
            found["dx"] += [dx, det_dx]
            found["dweight"] += [dw, det_dw]
        return found

    _route_judge(S, C, docs, "bwd", call, key=219, label="det and nondet SM-wide programs in one process")


def test_det_and_nondet_packages_in_one_process(monkeypatch) -> None:
    """The deterministic and the non-deterministic packages loaded in one process, their calls interleaved on one
    stream (policy forced on), at :data:`PDL_CASE` and at the first token count of the default route's first
    ``DIRECT_ROWS`` range (where both packages launch the direct program): the non-deterministic launches are of
    records with ``launch.use_pdl`` ``"runtime"`` and pass ``enable_pdl`` 1; each forward program both packages launch
    has two names, and the deterministic record of it launches without PDL.  Every output of both packages is judged
    against the naive tier (no comparison between them)."""

    _pdl_mode("eager", monkeypatch)
    seen = _record_pdl(monkeypatch)
    det = cake_package("cake_causal_conv1d_det_train")
    det_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")
    det_seen = []
    det_one, det_many = det.run_module, det.run_modules

    def det_record_one(name, values, *, device_index, grid=None):
        det_seen.append(name)
        return det_one(name, values, device_index=device_index, grid=grid)

    def det_record_many(launches, values, *, device_index):
        det_seen.extend(name for name, _ in launches)
        return det_many(launches, values, device_index=device_index)

    monkeypatch.setattr(det, "run_module", det_record_one)
    monkeypatch.setattr(det, "run_modules", det_record_many)
    arch = _arch()
    direct = [
        key for key in _forward_keys(arch) if key[0] == "fwd" and key[5] == "direct"
    ]
    assert direct, f"{arch}: no DIRECT_ROWS key on route fwd"
    _, c_lo, _, s_lo, _, _ = direct[0]

    def call(x, w, cu, dy, b, policy):
        found = {"y": [], "dx": [], "dweight": []}
        for _ in range(2):
            found["y"].append(_forward(x, w, cu))
            found["y"].append(det.cake_causal_conv1d_det_train_forward(x, w, cu))
            dx, dw, _ = _backward(x, w, cu, dy)
            det_dx, det_dw = det.cake_causal_conv1d_det_train_backward(x, w, cu, dy)[:2]
            found["dx"] += [dx, det_dx]
            found["dweight"] += [dw, det_dw]
        return found

    for S, C, docs in (PDL_CASE, (s_lo, c_lo, "perf")):
        _route_judge(
            S,
            C,
            docs,
            "bwd",
            call,
            key=191,
            label=f"det and nondet in one process S={S} C={C}",
        )
    _assert_pdl(seen, 1, "nondet")
    assert _forward_program(arch, "fwd", "direct") in {name for name, _ in seen}, seen
    assert det_jit.direct_module(arch, "fwd", c_lo, s_lo) in det_seen, det_seen
    # the programs both packages launch (the forward: one program, two names and bindings)
    shared = [
        (name, det_name)
        for det_name in sorted(set(det_seen))
        for name in sorted({n for n, _ in seen})
        if name.rsplit("_", 1)[-1] == det_name.rsplit("_", 1)[-1]
    ]
    assert shared, (det_seen, seen)
    for name, det_name in shared:
        assert name != det_name and det_jit.MODULES[det_name]["launch"]["use_pdl"] is False, det_name


def test_backward_is_one_launch_with_zeroed_accumulators(monkeypatch) -> None:
    """A backward call is ONE ordinary launch of the program ``C`` selects (no reduce program): its arguments
    carry the ``acc`` / ``work`` views of the record's accumulator layout (FP32 ``slots x taps x C`` words as the
    rule names them, one FP32 word per sum and accumulator copy on every route; uint32 ``n_ch_tiles`` counters), zero
    at launch, and the grid of the call's plan, at least one CTA per channel tile."""

    S, C, docs = 1000, 1280, "perf"
    device = _device()
    launched = []

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, values, grid))
        acc, work = values["acc"], values["work"]
        launched[-1] += (int(torch.count_nonzero(acc)), int(torch.count_nonzero(work)))

    monkeypatch.setattr(pkg, "run_module", record_launch)
    arch = _arch()
    for kind in ROUTE_SETTINGS:
        if _route_pending(arch, kind):
            continue
        launched.clear()
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        _backward(x, w, cu, dy, b, policy, kind=kind)
        assert len(launched) == 1, f"{kind}: {len(launched)} launches"
        name, values, grid, acc_nonzero, work_nonzero = launched[0]
        assert name == loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        assert "reduce" not in record and record["launch"]["cooperative"] is False
        assert grid == pkg._one_sm_trim(
            record,
            values["n_ch_tiles"],
            loader.launch_grid(record, device_index=device.index, C=C, S=S),
            device_index=device.index,
            arch=arch,
            bwd_grid_mult=1,
        )
        assert grid[0] >= values["n_ch_tiles"]
        layout = loader.accumulator_layout(record, C, values["n_ch_tiles"])
        acc_dtype = torch.float32
        taps = 5 if "bias" in kind else 4
        words = record["workspace_rule"]["slots"] * taps
        assert loader.accumulator_words_per_channel(record) == words
        assert values["acc"].dtype == acc_dtype and values["acc"].numel() == words * C
        assert (
            values["work"].dtype == torch.uint32
            and values["work"].numel() == values["n_ch_tiles"]
        )
        assert layout["acc_elems"] == words * C
        assert acc_nonzero == 0 and work_nonzero == 0
        assert {"x", "w", "cu", "dy", "dx", "dw", "acc", "work"} <= set(values)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_bwd_grid_mult_launches_that_many_times_the_untrimmed_grid(
    kind, monkeypatch
) -> None:
    """A backward at ``bwd_grid_mult`` k launches the program ``C`` selects once, with k times the record's
    untrimmed grid for k > 1 and the call's plan (possibly trimmed) for k = 1: the option is never a silent no-op."""

    _require_route(kind)
    device = _device()
    arch = _arch()
    launched = []
    tiles = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append((name, grid))
        tiles.append(values["n_ch_tiles"])
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    for S, C, docs in (
        (1000, 1344, "perf"),
        (65536, 1280, "perf"),
        (5, 136, (2, 0, 3)),
    ):
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=32)
        name = loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        untrimmed = loader.launch_grid(record, device_index=device.index)
        for mult in pkg.BWD_GRID_MULTS:
            launched.clear()
            tiles.clear()
            _backward(x, w, cu, dy, b, policy, kind=kind, bwd_grid_mult=mult)
            expected = (
                pkg._one_sm_trim(
                    record,
                    tiles[0],
                    loader.launch_grid(record, device_index=device.index, C=C, S=S),
                    device_index=device.index,
                    arch=arch,
                    bwd_grid_mult=1,
                )
                if mult == 1
                else (untrimmed[0] * mult, 1, 1)
            )
            assert launched == [(name, expected)], (kind, S, C, mult, launched)


@pytest.mark.parametrize("kind", tuple(ROUTE_SETTINGS))
def test_backward_grid_gives_up_one_sm_when_the_last_rank_is_partial(
    kind, monkeypatch
) -> None:
    """On the ``bwd`` and ``bwd_fp32g`` routes a default-grid backward launches ``ctas_per_sm
    x SM count`` CTAs less ``ctas_per_sm`` when the last channel-tile rank of that grid holds at least
    ``ctas_per_sm`` CTAs (so a kernel on another stream that keeps one SM's shared memory leaves the launch one
    wave), else the whole grid; the other backward routes launch ``launch_grid``'s grid.  The trimmed launch's
    outputs and those of the whole grid are judged against the naive tier and the workspace reads back zero."""

    _require_route(kind)
    device = _device()
    arch = _arch()
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    S, docs = 4097, (1000, 0, 3, 3094)
    launched = []
    run_module = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        launched.append(grid)
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(pkg, "run_module", record_launch)
    rows = {}
    for C in range(1024, 16385, 1024):
        name = loader.selected_modules(arch, kind, C=C, S=S)[0]
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        k = int(rule["ctas_per_sm"])
        full = loader.grid_mult(record, arch) * k * sms
        n = loader.plan(
            S,
            C,
            len(docs),
            geometry=rule["geometry"],
            role=record["role"],
            arch=arch,
            route=loader.record_route(record, arch),
        )["n_ch_tiles"]
        trims = (
            rule["kind"] == "occupancy"
            and loader.grid_mult(record, arch) == 1
            and full % n >= k
        )
        rows.setdefault(trims, (C, full, k))
    assert set(rows) == {True, False}, rows
    for trims, (C, full, k) in sorted(rows.items()):
        cut = trims and kind in ("bwd", "bwd_fp32g")

        def call(x, w, cu, dy, b, policy, C=C, full=full, k=k, cut=cut):
            launched.clear()
            runs = [_backward(x, w, cu, dy, b, policy, kind=kind)]
            torch.cuda.synchronize(device)
            assert launched == [(full - k if cut else full, 1, 1)], (kind, C, launched)
            held = _stream_workspace(device)
            assert held is None or int(torch.count_nonzero(held)) == 0, (kind, C)
            if cut:
                with monkeypatch.context() as m:
                    m.setattr(pkg, "_one_sm_trim", lambda record, n, grid, **kw: grid)
                    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                    launched.clear()
                    runs.append(_backward(x, w, cu, dy, b, policy, kind=kind))
                    torch.cuda.synchronize(device)
                pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
                assert launched == [(full, 1, 1)], (kind, C, launched)
            found = {}
            for run in runs:
                for out, value in _grads(*run).items():
                    found.setdefault(out, []).append(value)
            return found

        _route_judge(S, C, docs, kind, call, key=41, label=f"{kind} C={C}")


@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_fp32g_accumulators_follow_the_rendered_rule(kind, C, monkeypatch) -> None:
    """An fp32g backward at ``C`` 512 and 8192 hands its program an FP32 ``acc`` (one FP32 word per sum and
    accumulator copy) of the size the rendered rule names (``slots x taps x C``, however many accumulator copies the
    program keeps) inside the public workspace; the outputs are judged against the naive tier, and the workspace
    reads back zero."""

    _require_route(kind)
    S, docs = 4097, (1000, 0, 3, 3094)
    device = _device()
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    seen = []
    launch = pkg.run_module

    def record_launch(name, values, *, device_index, grid=None):
        seen.append(
            (name, values["acc"].dtype, values["acc"].numel(), values["n_ch_tiles"])
        )
        return launch(name, values, device_index=device_index, grid=grid)

    def call(x, w, cu, dy, b, policy):
        seen.clear()
        with monkeypatch.context() as m:
            m.setattr(pkg, "run_module", record_launch)
            dx, dw, db = _backward(x, w, cu, dy, b, policy, kind=kind, workspace=workspace)
        ((name, acc_dtype, acc_numel, n_ch_tiles),) = seen
        record = loader.MODULES[name]
        layout = loader.accumulator_layout(record, C, n_ch_tiles)
        assert acc_dtype == torch.float32
        assert (
            acc_numel
            == layout["acc_elems"]
            == loader.accumulator_words_per_channel(record) * C
        )
        assert workspace.numel() >= layout["total_bytes"]
        assert int(torch.count_nonzero(workspace)) == 0
        return _grads(dx, dw, db)

    _route_judge(S, C, docs, kind, call, key=11, label=f"{kind} C={C}")


def test_backward_keeps_its_speed_beside_a_resident_side_kernel(
    record_property,
) -> None:
    """One 1-thread kernel resident on another stream must not stall the
    one-launch backward (``busy <= 2 x alone``, medians of 7 after 3 warm-ups, CUDA events on the current stream);
    the outputs of backward calls launched beside it are judged against the naive tier."""

    device = _device()
    S, C, docs = 261393, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    dx = torch.empty_like(x)
    dw = torch.empty(C, WIDTH, dtype=torch.float32, device=device)
    side = torch.cuda.Stream(device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def timed(with_side_kernel: bool) -> float:
        torch.cuda.synchronize(device)
        if with_side_kernel:
            with torch.cuda.stream(side):
                torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        start.record()
        _backward(x, w, cu, dy, workspace=workspace, dx=dx, dweight=dw)
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(3):
        timed(False)
    alone = statistics.median(timed(False) for _ in range(7))
    busy = statistics.median(timed(True) for _ in range(7))
    print(
        f"backward alone {alone:.3f} ms, beside a resident kernel {busy:.3f} ms ({busy / alone:.2f}x)"
    )
    record_property("backward_alone_ms", f"{alone:.4f}")
    record_property("backward_beside_side_kernel_ms", f"{busy:.4f}")
    record_property("busy_over_alone", f"{busy / alone:.3f}")
    assert busy <= 2.0 * alone, (
        f"the backward stalled behind a kernel resident on another stream: {busy:.3f} ms vs {alone:.3f} ms"
    )

    def beside(x, w, cu, dy, b, policy):
        torch.cuda.synchronize(device)
        with torch.cuda.stream(side):
            torch.cuda._sleep(10_000_000)
        torch.cuda._sleep(1_000_000)
        run = _backward(x, w, cu, dy, workspace=workspace)
        torch.cuda.synchronize(device)
        return _grads(*run)

    _judge(S, C, docs, beside, label="beside the side kernel")


def test_zero_tokens_return_empty_outputs() -> None:
    C = 256
    x = torch.empty(0, C, dtype=BF16, device=_device())
    w = torch.zeros(C, WIDTH, dtype=F32, device=_device())
    b = torch.zeros(C, dtype=BF16, device=_device())
    cu = torch.zeros(2, dtype=torch.int32, device=_device())
    assert tuple(_forward(x, w, cu).shape) == (0, C)
    dx, dw, _ = _backward(x, w, cu, x)
    assert tuple(dx.shape) == (0, C) and torch.equal(dw, torch.zeros_like(dw))
    dw_out = torch.full((C, WIDTH), float("nan"), dtype=F32, device=_device())
    db_out = torch.full((C,), float("nan"), dtype=BF16, device=_device())
    dx, dw, db = _backward(x, w, cu, x, b, "fp32", dweight=dw_out, dbias=db_out)
    assert dw is dw_out and db is db_out
    assert torch.equal(dw, torch.zeros_like(dw)) and torch.equal(
        db, torch.zeros_like(db)
    )


# ----------------------------------------------------------------------------------------------------------------
# Autograd, checkpointing, accumulation.


@pytest.mark.parametrize("combo", AUTOGRAD_COMBOS, ids=_combo_id)
def test_autograd_end_to_end(combo) -> None:
    """``cake_causal_conv1d_nondet`` with FP32 / BF16 taps, no / FP32 / BF16 bias, under both SiLU' policies: ``y``
    and the gradients, in the parameters' dtypes, judged against the naive tier, with a contiguous and a
    non-contiguous (materialized) incoming gradient."""

    weight_dtype, bias_dtype, policy = combo
    kind = _combo_route(weight_dtype, bias_dtype, policy)
    _require_route(kind)
    S, C, docs = 1000, 512, "perf"

    def call(x0, w0, cu, dy, b0, policy):
        found = {}
        for incoming in (dy, dy.t().contiguous().t()):
            x = x0.detach().clone().requires_grad_(True)
            w = w0.detach().clone().requires_grad_(True)
            b = None if b0 is None else b0.detach().clone().requires_grad_(True)
            y = pkg.cake_causal_conv1d_nondet(x, w, cu, b, silu_grad_preactivation=policy)
            y.backward(incoming)
            assert x.grad.dtype == BF16
            assert w.grad.dtype == w0.dtype and (b is None or b.grad.dtype == b0.dtype)
            outs = {"y": y.detach(), **_grads(x.grad, w.grad, None if b is None else b.grad)}
            for out, value in outs.items():
                found.setdefault(out, []).append(value)
        return found

    _judge(
        S,
        C,
        docs,
        call,
        weight_dtype=weight_dtype,
        bias_dtype=bias_dtype,
        policy=policy,
        key=3,
        label=f"autograd {kind} {_combo_id(combo)}",
    )


def test_non_reentrant_checkpoint_is_two_forwards_and_a_backward() -> None:
    """``torch.utils.checkpoint(..., use_reentrant=False)`` around the autograd entry (2 x forward + backward) and
    the unchecked call: ``y`` and the gradients of both judged against the naive tier."""

    from torch.utils.checkpoint import checkpoint

    _require_route("bwd_bias")
    S, C, docs = 4097, 1280, (1000, 0, 3, 3094)

    def call(x0, w0, cu, dy, b0, policy):
        found = {}
        for ckpt in (False, True):
            x = x0.detach().clone().requires_grad_(True)
            w = w0.detach().clone().requires_grad_(True)
            b = b0.detach().clone().requires_grad_(True)

            def layer(x, w, b):
                return pkg.cake_causal_conv1d_nondet(
                    x, w, cu, b, silu_grad_preactivation=policy
                )

            y = checkpoint(layer, x, w, b, use_reentrant=False) if ckpt else layer(x, w, b)
            y.backward(dy)
            for out, value in {"y": y.detach(), **_grads(x.grad, w.grad, b.grad)}.items():
                found.setdefault(out, []).append(value)
        return found

    _route_judge(S, C, docs, "bwd_bias", call, key=13, label="checkpoint")


def test_gradient_accumulation() -> None:
    """Two backward passes accumulate into the parameters' ``.grad`` (autograd's sum of the two gradients): the sum
    judged against the naive tier's sum of its two BF16 ``dweight`` (the float64 oracle's sum of the two)."""

    S, C, docs = 4097, 512, (1000, 0, 3, 3094)
    judge = TierJudge(DRAWS)
    failures = []
    for j in range(DRAWS):
        seed = draw_seed(SEED, j)
        seed_all(seed)
        x0, w0, cu, dy = _inputs(S, C, docs, key=17, seed=seed)
        dy2 = torch.randn_like(dy.float()).to(BF16)
        w = w0.detach().clone().requires_grad_(True)
        for g in (dy, dy2):
            pkg.cake_causal_conv1d_nondet(x0, w, cu).backward(g)
        want = sum(oracle(x0, w0, cu, None, g, outputs=("dweight",))["dweight"] for g in (dy, dy2))
        naive = sum(naive_bf16(x0, w0, cu, None, g, outputs=("dweight",))["dweight"] for g in (dy, dy2))
        if message := contract("dweight", w.grad, want):
            failures.append(f"draw {j}: {message}")
            continue
        judge.add("dweight", j, w.grad, want, {"bf16": naive})
    failures += judge.failures()
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("kind", AUTOGRAD_KINDS)
def test_bwd_grid_mult_through_autograd(kind) -> None:
    """The autograd entry carries ``bwd_grid_mult`` to its backward: at every multiple, plain and under
    non-reentrant checkpointing, ``y`` and the gradients judged against the naive tier, on every route the autograd
    entry reaches."""

    from torch.utils.checkpoint import checkpoint

    _require_route(kind)
    S, C, docs = 4097, 1344, (1000, 0, 3, 3094)

    def call(x0, w0, cu, dy, b0, policy):
        found = {}
        for mult in pkg.BWD_GRID_MULTS:
            for ckpt in (False, True):
                x = x0.detach().clone().requires_grad_(True)
                w = w0.detach().clone().requires_grad_(True)
                b = None if b0 is None else b0.detach().clone().requires_grad_(True)

                def layer(x, w, b, mult=mult):
                    return pkg.cake_causal_conv1d_nondet(
                        x,
                        w,
                        cu,
                        b,
                        silu_grad_preactivation=policy,
                        bwd_grid_mult=mult,
                    )

                y = (
                    checkpoint(layer, x, w, b, use_reentrant=False)
                    if ckpt
                    else layer(x, w, b)
                )
                y.backward(dy)
                outs = {"y": y.detach(), **_grads(x.grad, w.grad, None if b is None else b.grad)}
                for out, value in outs.items():
                    found.setdefault(out, []).append(value)
        return found

    _route_judge(S, C, docs, kind, call, key=34, label=f"autograd {kind}")


# ----------------------------------------------------------------------------------------------------------------
# Torch's deterministic mode.


def test_torch_deterministic_mode_alerts_on_the_backward() -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` the backward raises (``warn_only=True``: warns and runs the
    non-deterministic backward, whose outputs are judged against the naive tier); the forward runs without an alert.
    The mode is read when the backward runs: an autograd graph recorded with the mode off raises once it is on."""

    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    recorded = pkg.cake_causal_conv1d_nondet(x.clone().requires_grad_(True), w, cu)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        y = _forward(x, w, cu)
        assert y.dtype == BF16 and tuple(y.shape) == (S, C)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward"
        ):
            _backward(x, w, cu, dy)
        xg = x.clone().requires_grad_(True)
        y = pkg.cake_causal_conv1d_nondet(xg, w, cu)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward"
        ):
            y.backward(dy)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward"
        ):
            recorded.backward(dy)
        torch.use_deterministic_algorithms(True, warn_only=True)

        def call(x, w, cu, dy, b, policy):
            with pytest.warns(
                UserWarning, match="cake_causal_conv1d_nondet_train_backward"
            ):
                return _grads(*_backward(x, w, cu, dy))

        _judge(S, C, docs, call, label="warn_only")
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


def test_argument_validation() -> None:
    S, C = 64, 256
    x, w, cu, dy = _inputs(S, C, None)
    b = _bias_input(C, F32)
    with pytest.raises(ValueError, match="multiple of 8"):
        _forward(x[:, :130].contiguous(), w[:130].contiguous(), cu)
    with pytest.raises(ValueError, match="width-4"):
        _forward(x, w[:, :3].contiguous(), cu)
    with pytest.raises(ValueError, match="torch.int32"):
        _forward(x, w, cu.long())
    with pytest.raises(ValueError, match="shape"):
        _backward(x, w, cu, dy[: S - 1])
    with pytest.raises(
        ValueError, match="weight must be torch.float32 or torch.bfloat16"
    ):
        _backward(x, w.half(), cu, dy)
    with pytest.raises(ValueError, match="dbias"):
        _backward(x, w, cu, dy, dbias=torch.empty_like(b))
    with pytest.raises(ValueError, match="silu_grad_preactivation"):
        _backward(x, w, cu, dy, policy="fp16")
    for bad, error in (
        (0, ValueError),
        (5, ValueError),
        (2.0, TypeError),
        (True, TypeError),
        ("2", TypeError),
    ):
        with pytest.raises(error, match="bwd_grid_mult must be"):
            _backward(x, w, cu, dy, bwd_grid_mult=bad)
        with pytest.raises(error, match="bwd_grid_mult must be"):
            pkg.cake_causal_conv1d_nondet(x, w, cu, bwd_grid_mult=bad)
        with pytest.raises(error, match="bwd_grid_mult must be"):
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(
                S, C, _device(), bwd_grid_mult=bad
            )
    with pytest.raises(ValueError, match="CUDA"):
        _forward(x.cpu(), w.cpu(), cu.cpu())


def test_fp32_dweight_must_be_16_byte_aligned() -> None:
    """The backward programs store one channel's four FP32 ``dweight`` taps with one 16-byte store: an FP32
    ``dweight`` out off the 16-byte grid is refused (``ValueError``) before anything launches; a 16-byte aligned
    view inside a larger buffer is written, and a BF16 ``dweight`` or an FP32 / BF16 ``dbias`` (stored per
    element) is accepted at any element offset; the written views are judged against the naive tier."""

    S, C, docs = 1000, 512, "perf"
    device = _device()
    x, w, cu, dy = _inputs(S, C, docs, key=8)
    b = _bias_input(C, F32, key=8)
    n = C * WIDTH
    buf = torch.zeros(n + 4, dtype=F32, device=device)
    workspace = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device)
    for bias in (None, b):
        for offset in (1, 2, 3):
            misaligned = buf[offset : offset + n].view(C, WIDTH)
            assert misaligned.data_ptr() % 16 != 0
            with pytest.raises(ValueError, match="16-byte aligned"):
                _backward(x, w, cu, dy, bias, dweight=misaligned, workspace=workspace)
    torch.cuda.synchronize(device)
    assert (
        int(torch.count_nonzero(buf)) == 0 and int(torch.count_nonzero(workspace)) == 0
    )

    def aligned_view(x, w, cu, dy, b, policy):
        aligned = torch.zeros(n + 4, dtype=F32, device=device)[4:].view(C, WIDTH)
        db_view = torch.zeros(C + 1, dtype=F32, device=device)[1:]
        dx, dw, db = _backward(
            x, w, cu, dy, b, dweight=aligned, dbias=db_view, workspace=workspace
        )
        assert dw is aligned and db is db_view
        return _grads(dx, dw, db)

    _judge(S, C, docs, aligned_view, bias_dtype=F32, key=8, label="16-byte aligned dweight view")
    for dweight_offset, dbias_offset in ((4, 1), (2, 3)):

        def bf16_outs(x, w, cu, dy, b, policy, dweight_offset=dweight_offset, dbias_offset=dbias_offset):
            dw16 = torch.zeros(n + 4, dtype=BF16, device=device)[dweight_offset:][:n].view(
                C, WIDTH
            )
            db16 = torch.zeros(C + 4, dtype=BF16, device=device)[dbias_offset:][:C]
            assert dw16.data_ptr() % 16 != 0
            dx, dw, db = _backward(
                x, w, cu, dy, b, dweight=dw16, dbias=db16, workspace=workspace
            )
            assert dw is dw16 and db is db16
            return _grads(dx, dw, db)

        _judge(
            S,
            C,
            docs,
            bf16_outs,
            bias_dtype=F32,
            key=8,
            label=f"bf16 outs at element offsets {dweight_offset} / {dbias_offset}",
        )
    assert int(torch.count_nonzero(workspace)) == 0


# ----------------------------------------------------------------------------------------------------------------
# The registry against the rendered manifest.


def test_every_route_ships_on_every_architecture() -> None:
    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        assert set(loader.ROUTES[arch]) == set(loader.KINDS), arch
        for kind in loader.KINDS:
            for name in loader.route_modules(arch, kind):
                assert loader.library_name(name, arch) in libraries, (arch, kind, name)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every record names its ``build.json`` library (kernel + binding sources under ``csrc/``, digests) and no
    record is left a TODO-EXPORT placeholder; the manifest names this package's import path."""

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "cake_pplx._backends.cake_causal_conv1d_nondet_train"
    )
    libraries = cake_build.libraries_by_name()
    kcp_loader = cake_package(KEY, "kcp_jit")
    referenced = set()
    for registry in (loader.MODULES, kcp_loader.MODULES):
        for name, record in registry.items():
            assert "TODOEXPORT" not in name, f"{name} is still a placeholder"
            assert record["library"] == record["kernel"] == name
            assert record["kernel_symbol"] == f"kernel_{name}"
            for arch in record["arches"]:
                entry = libraries[f"{name}_{arch}"]
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                referenced.add(entry["name"])
    assert referenced == set(libraries)
