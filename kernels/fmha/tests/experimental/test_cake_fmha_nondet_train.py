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

Backend suite of the Cake non-deterministic (``nondet``) dense multi-head attention backward package
(``cake_pplx._backends.cake_fmha_nondet_train``, the backend of ``cake_pplx.cake_fmha_nondet_train``): head dimension
64 or 128, any head count, query and key lengths multiples of 128, causal (top-left, ``Sq == Skv``) or not, on SM100,
SM103 and SM107.  The package promises no bits: on the three-kernel routes the main stage adds the FP32 dQ tiles of
every key block of a head in arrival order, so no test of those routes compares two runs bit for bit.  The one-launch
stage writes dq, dk and dv in one fixed order, and its own test compares three runs bit for bit.

Without a device: the registry (the three stages of the three-kernel routes in launch order, then the one-launch
stage, one program per record, every record delivered for the three architectures and for no other, the launch facts:
a cluster of two CTAs for the 2-CTA main kernels), the routing table ``RECORD_OF`` (the preprocess, main-stage and
postprocess record of each head dimension, causal flag and route, a 2-CTA main stage with the pair-row postprocess and
no other), the one-launch record of each head dimension without the mask (``FUSED_OF``: every record serves a variant
or ``FUSED_OF``, and no variant or host sequence names a one-launch record) and the host sequence of each variant
(``SEQUENCES``: its three records), the route of a call (``variant``: the pair route for head dimension 128, a key
length that is a multiple of 256, at least as many queries as keys and ``B * H`` and ``Skv`` in a region of
``PAIR_REGIONS``; every measured crossover call on its decided route, ``DECIDED_LEAST``; the long-sequence causal region
of ``sm_103a`` on both sides of each bound, ``CAUSAL_PAIR_CALLS``; the pair route without the mask for fewer queries
than keys within the bounds ``PAIR_FEWER_QUERIES`` of the architecture, the shipped bounds
``SHIPPED_PAIR_FEWER_QUERIES``, on both sides of each bound and refused with the mask, and every measured call with
fewer queries than keys on its decided route, ``DECIDED_FEWER_QUERIES``; ``fused_record``: the one-launch route inside
any piece of the region of ``FUSED_REGION``, on both sides of each bound of a region of one piece and of two pieces;
the shipped region of each architecture and head dimension, ``SHIPPED_FUSED_REGION``, on both sides of each bound of
each of its pieces and, for each piece, a call that no other piece of its region holds, every measured long call on
its decided route, ``DECIDED_LONG_CALLS``, and the long calls of the benchmark's ``T1`` to ``T4`` rows on the
three-kernel routes on every architecture), the launch grids, the stage launchers' keywords, their positional launch of
every routed record (the FFI arguments and grid of the registry-ordered launch), the two layouts of the dQ accumulator
in the judge's helper, the build manifest (every record's and host sequence's library, the recorded source digests), the
standalone loader, the prebuilt libraries served first, and no environment read outside the JIT shim.

On SM100 / SM103 / SM107, each stage judged on its own against the naive BF16 tier (``tests/cake_naive.py``: the error
against an FP64 oracle of the same inputs over 32 independent draws, in the L2 and the max norm, median and largest,
beside the error of the same math in BF16), every output buffer filled with NaN before the launch so every element
must be written: the preprocess (``dPsum``, ``LSE_log2``, the cleared ``dQ_accum``) at both head dimensions; the main
stage on its own inputs (the FP32 ``LSE_log2`` and ``dPsum``, a zeroed ``dQ_accum``) for every variant, judged on
``dQ_accum`` (in the pair-row layout for the 2-CTA main kernels), ``dK`` and ``dV``; the postprocess (``dq`` in the
input layout) at both head dimensions and in the pair-row build; the one-launch stage (``dq``, ``dk`` and ``dv`` from
the forward's ``out`` and ``lse``) at both head dimensions on the kernel's own cases, and run to run bit for bit.  The
public backward judged on the cases of every variant at four heads (the source tests' cases), one head and five
heads, with the one-launch route turned off (``fused_record`` patched to give None), and unpatched on the one-launch
stage's cases; a call launches the kernels of the route ``fused_record`` and ``variant`` give it, in order and no
other: the one-launch record alone, without a host sequence, or the three records ``RECORD_OF`` names for its variant
(observed on the device, on both routes of head dimension 128), its inputs left unchanged; the same records, and
``dq``, ``dk`` and ``dv`` within the tier, when the host sequence's ``run`` returns ``False`` (the workspace from
Python) and without a host sequence (the stage launchers).  Where ``torch.profiler`` records no kernel of a plain torch
operation on the device (a CUPTI that does not support it), the tests that observe kernels run every other check, then
skip the observation.  The public backward is also judged on the shipped tables on the least call with fewer queries
than keys within the bounds of the pair route (the pair route on SM103 and SM107, the 1-CTA route on SM100), and on
the least causal call of the long-sequence causal region of SM103 (``B * H`` 32, 8192 queries and keys: the pair route
on SM103 and SM107, the 1-CTA route on SM100).

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_fmha_naive import (
    Case,
    accumulator,
    assert_backward_contract,
    from_accumulator,
    fused_outputs,
    judge_draws,
    judged_backward,
    judged_stage,
    main_buffers,
    main_outputs,
    make_inputs,
    post_inputs,
    post_outputs,
    pre_inputs,
    pre_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake.cake_fmha_nondet_train")
cake_backend = cake_package("cake.cake_fmha_nondet_train", "cake_backend")
cake_build = cake_package("cake.cake_fmha_nondet_train", "cake_build")
cake_jit = cake_package("cake.cake_fmha_nondet_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
STAGES = ("bwd_pre", "bwd_main", "bwd_post", "bwd_fused")
# The stages of the three-kernel routes, in launch order.
CLOSURE_STAGES = ("bwd_pre", "bwd_main", "bwd_post")
HEAD_DIMS = (128, 64)
# (head dimension, causal, pair) of every variant; each is delivered for the three architectures.  ``pair``: the
# pair route of head dimension 128 (the 2-CTA main kernel and the pair-row postprocess).
VARIANTS = (
    (128, False, False),
    (128, True, False),
    (64, False, False),
    (64, True, False),
    (128, False, True),
    (128, True, True),
)
VARIANT_IDS = ["d128", "d128_causal", "d64", "d64_causal", "d128_pair", "d128_pair_causal"]
# The 2-CTA main-stage records (a cluster of two CTAs along x) and the postprocess that reads their pair-row order.
PAIR_MAINS = ("bwd_main_2cta", "bwd_main_2cta_causal")
PAIR_POST = "bwd_post_pair"
# The one-launch record of each (head dimension, causal): no mask.
FUSED_OF = {(128, False): "bwd_fused", (64, False): "bwd_fused_hd64"}
# Threads per CTA of each stage's kernel (8, 16, 4 and 16 warps).
BLOCK_THREADS = {"bwd_pre": 256, "bwd_main": 512, "bwd_post": 128, "bwd_fused": 512}
ENTRY = "cake_fmha_nondet_train_backward"
# The keywords of each stage launcher, after the record name.
LAUNCHER_KEYWORDS = {
    "launch_bwd_pre": ("out", "dout", "lse", "lse_log2", "dpsum", "dq_accum"),
    "launch_bwd_main": ("q", "k", "v", "dout", "lse_log2", "dpsum", "dq_accum", "dk", "dv", "softmax_scale"),
    "launch_bwd_post": ("dq_accum", "dq", "softmax_scale"),
    "launch_bwd_fused": ("q", "k", "v", "out", "lse", "dout", "dq", "dk", "dv", "softmax_scale"),
}
# The decided route of every measured crossover call (head dimension 128, ``Sq == Skv``), by architecture and mask: for
# each measured ``B * H``, the least length on the pair route (every longer measured length on it too), or None (no
# length on it).  This is the decision taken from the crossover run, not a measurement: a call is on the pair route
# where it was FASTER than 0.1.4 in every counted round, beyond the identical-kernel controls' single-round noise, and
# so was every longer call of its ``B * H``.  On ``sm_107a`` one-head causal calls (``B * H`` 1) with 1024 keys stay on
# the 1-CTA route, because the final measurement found the pair route slower there than 0.1.4.  On ``sm_107a`` four-head
# non-causal calls (``B * H`` 4) under 2048 keys stay on the 1-CTA route, because the final measurement found the pair
# route slower than 0.1.4 there in eager calls.  On ``sm_103a`` the causal calls of ``B * H`` 32 and 128 take the pair
# route from 8192 keys since a later measurement against the 1-CTA route: every measured causal call of ``B * H`` 32,
# 64, 128 and 256 with 8192 and 16384 keys was faster in every counted round of both replicate runs, eager and in
# CUDA-graph replays; ``B * H`` 16 stays on the 1-CTA route (its eager calls with 8192 keys were not faster beyond the
# identical-kernel controls' noise), and so do the calls of ``B * H`` 16 or more with 4096 keys.  ``PAIR_REGIONS``
# must reproduce it.
MEASURED_COUNTS = (1, 4, 16, 32, 128)
MEASURED_LENGTHS = (256, 512, 1024, 2048, 4096, 8192)
DECIDED_LEAST = {
    "sm_100a": {
        False: {1: 4096, 4: 8192, 16: None, 32: None, 128: 8192},
        True: {1: None, 4: None, 16: None, 32: None, 128: None},
    },
    "sm_103a": {
        False: {1: 2048, 4: 2048, 16: 8192, 32: 8192, 128: 8192},
        True: {1: 2048, 4: 2048, 16: None, 32: 8192, 128: 8192},
    },
    "sm_107a": {
        False: {1: 2048, 4: 2048, 16: 4096, 32: 4096, 128: 4096},
        True: {1: 2048, 4: 2048, 16: 4096, 32: 8192, 128: 8192},
    },
}
# Causal calls (``Sq == Skv``) on both sides of each bound of the long-sequence causal region of ``sm_103a`` (``B * H``
# 32 or more, 8192 keys or more), as ((B * H, S), routed on ``sm_103a``): the least call inside; ``B * H`` 31, and 16
# (the measured neighbour); 7936 keys; 8320 keys (not a multiple of 256) and 8448 keys there; then more heads and more
# keys inside, 4096 keys outside, and both sides of the region of four heads or fewer beside it.
CAUSAL_PAIR_CALLS = (
    ((32, 8192), True),
    ((31, 8192), False),
    ((16, 8192), False),
    ((32, 7936), False),
    ((32, 8320), False),
    ((32, 8448), True),
    ((256, 8192), True),
    ((4096, 16384), True),
    ((128, 4096), False),
    ((16, 16384), False),
    ((4, 2048), True),
    ((5, 2048), False),
)
# The shipped bounds of the pair route for a call without the mask with fewer queries than keys, per architecture:
# ``(least B * H, least Sq, least Skv, most Skv / Sq)``, or None (no such call on the pair route).
# ``PAIR_FEWER_QUERIES`` must be this table.
SHIPPED_PAIR_FEWER_QUERIES = {"sm_100a": None, "sm_103a": (8, 4096, 8192, 2), "sm_107a": (8, 4096, 8192, 2)}
# Calls with fewer queries than keys on both sides of each shipped bound, as ((B * H, Sq, Skv), routed where the
# architecture has the bounds (8, 4096, 8192, 2)): the least call inside; B * H at 7; 3968 queries (then 8192 keys are
# also more than twice the queries); 7936 keys; keys at twice the queries and 256 past it (4096 and 6144 queries); 8320
# keys, not a multiple of 256 (against 4224 queries), and 8448 keys there; then more heads, more keys and fewer keys per
# query inside, fewer heads and fewer queries outside.
FEWER_QUERIES_CALLS = (
    ((8, 4096, 8192), True),
    ((7, 4096, 8192), False),
    ((8, 3968, 8192), False),
    ((8, 4096, 7936), False),
    ((8, 4096, 8448), False),
    ((8, 6144, 12288), True),
    ((8, 6144, 12544), False),
    ((8, 4224, 8320), False),
    ((8, 4224, 8448), True),
    ((256, 4096, 8192), True),
    ((8, 8192, 16384), True),
    ((16, 6144, 8192), True),
    ((4, 4096, 8192), False),
    ((32, 2048, 8192), False),
)
# The decided route of the measured calls with fewer queries than keys (head dimension 128, no mask), as ((B, H, Sq,
# Skv), routed where the architecture has the bounds: ``sm_103a`` and ``sm_107a``).  This is the decision taken from the
# measurement, not a measurement: the routed calls were faster than 0.1.8 (the 1-CTA route) in both replicate runs on
# SM103 and SM107; 32 heads of 1024 queries against 4096 keys and four heads of 2048 queries against 8192 keys were
# not, and stay on the 1-CTA route.  ``PAIR_FEWER_QUERIES`` must reproduce it.
DECIDED_FEWER_QUERIES = (
    ((1, 32, 4096, 8192), True),
    ((2, 32, 4096, 8192), True),
    ((4, 32, 4096, 8192), True),
    ((1, 32, 1024, 4096), False),
    ((1, 4, 2048, 8192), False),
)


def _inside_fewer_queries(bounds: tuple[int, int, int, int] | None, count: int, Sq: int, Skv: int) -> bool:
    """A call with fewer queries than keys inside the bounds ``(least B * H, least Sq, least Skv, most Skv / Sq)`` of
    the pair route (None: no call), at ``B * H`` ``count``; the key length a multiple of 256 is checked apart."""
    if bounds is None:
        return False
    least_count, least_q, least_kv, most_per_q = bounds
    return count >= least_count and Sq >= least_q and least_kv <= Skv <= most_per_q * Sq


def _pin_route(monkeypatch, pin: bool | None) -> None:
    """The pair regions a test runs under: ``pin`` True, one region holding every call on every architecture and mask
    (the pair route wherever the head dimension, the key length and ``Sq >= Skv`` allow it); False, no region (the
    1-CTA route); None, the shipped ``PAIR_REGIONS``.  ``PAIR_LEAST`` is pinned with them, and with True or False
    ``PAIR_FEWER_QUERIES`` holds no bounds (no call with fewer queries than keys on the pair route)."""
    if pin is None:
        return
    regions = ((1, None, 2 * cake_jit.BLOCK),) if pin else ()
    least = 2 * cake_jit.BLOCK if pin else None
    for arch in ARCHES:
        monkeypatch.setitem(cake_jit.PAIR_REGIONS, arch, {False: regions, True: regions})
        monkeypatch.setitem(cake_jit.PAIR_LEAST, arch, {False: least, True: least})
        monkeypatch.setitem(cake_jit.PAIR_FEWER_QUERIES, arch, None)


def main_cases(causal: bool, pair: bool = False) -> list[Case]:
    """The main stage's and the backward's cases of one causal flag and route.  The 1-CTA route: the source tests'
    four cases at four heads (the uneven one has more keys than queries without the causal mask, a longer sequence
    with it), then one head and an odd head count.  The pair route (key lengths that are multiples of 256): the 2-CTA
    source tests' cases, at four heads (without the mask also one query tile, more queries than keys and fewer), then
    one head and an odd head count."""
    if pair:
        if causal:
            return [
                Case(1, 4, 256),
                Case(2, 4, 512),
                Case(1, 4, 768),
                Case(3, 4, 1024),
                Case(2, 1, 512),
                Case(2, 5, 512),
            ]
        return [
            Case(1, 4, 256),
            Case(2, 4, 128, 256),
            Case(2, 4, 512),
            Case(1, 4, 384, 512),
            Case(2, 4, 768, 256),
            Case(3, 4, 1024),
            Case(2, 1, 512),
            Case(2, 5, 512),
        ]
    uneven = Case(1, 4, 640) if causal else Case(1, 4, 384, 640)
    odd = Case(2, 5, 640) if causal else Case(2, 5, 384, 640)
    return [Case(1, 4, 128), Case(2, 4, 256), uneven, Case(3, 4, 1024), Case(3, 1, 256), odd]


MAIN_PARAMS = [
    pytest.param(D, causal, pair, case, id=f"{variant}-{case.id}")
    for (D, causal, pair), variant in zip(VARIANTS, VARIANT_IDS, strict=True)
    for case in main_cases(causal, pair)
]
# The one-launch kernel's own test cases (one and four heads; more keys than queries, fewer; a long sequence), at each
# head dimension.
FUSED_CASES = (
    Case(1, 1, 128),
    Case(1, 1, 256),
    Case(1, 4, 128),
    Case(2, 4, 256),
    Case(1, 4, 384, 640),
    Case(2, 5, 384, 256),
    Case(3, 4, 1024),
)
FUSED_PARAMS = [pytest.param(D, case, id=f"d{D}-{case.id}") for D in HEAD_DIMS for case in FUSED_CASES]
# The backward's cases: the main stage's, each call on its variant's route (``_pin_route``), with the one-launch route
# turned off (``fused`` False: ``fused_record`` patched to give None).  A pair case with more keys than queries takes
# the 1-CTA route through the backward (under ``_pin_route`` no such call takes the pair route), so its id names the
# variant it runs (``d128``).  Then the one-launch stage's cases at each head dimension without the mask, unpatched
# (``fused`` True: the route ``fused_record`` gives).
BACKWARD_PARAMS = [
    pytest.param(
        D,
        causal,
        pair,
        case,
        False,
        id=f"{VARIANT_IDS[VARIANTS.index((D, causal, pair and case.S >= case.kv))]}-{case.id}",
    )
    for D, causal, pair in VARIANTS
    for case in main_cases(causal, pair)
] + [pytest.param(D, False, False, case, True, id=f"d{D}-fused-{case.id}") for D in HEAD_DIMS for case in FUSED_CASES]
# The postprocess records, by (head dimension, pair).
POST_VARIANTS = ((128, False), (64, False), (128, True))
POST_IDS = ["d128", "d64", "d128_pair"]
# The calls whose launched kernels are observed, each with the regions it runs under (``_pin_route``) and with the
# one-launch route turned off (``fused`` False: ``fused_record`` patched to give None): two (head, batch) pairs of three
# heads and 256 queries and keys for every head dimension and mask, at head dimension 128 both with every call in a pair
# region (the pair route) and with none (the 1-CTA route); 384 keys (an odd key-block count: the 1-CTA route in any
# region) at head dimension 128.  Then one call of two heads and 256 queries and keys per head dimension without the
# mask, unpatched (``fused`` True: the route ``fused_record`` gives), with no pair region (``pin`` False: the 1-CTA
# route when ``fused_record`` gives None).
ROUTE_PARAMS = [
    pytest.param(128, False, Case(2, 3, 256), True, False, id="d128"),
    pytest.param(128, True, Case(2, 3, 256), True, False, id="d128_causal"),
    pytest.param(128, False, Case(2, 3, 256), False, False, id="d128-1cta"),
    pytest.param(128, True, Case(2, 3, 256), False, False, id="d128_causal-1cta"),
    pytest.param(64, False, Case(2, 3, 256), True, False, id="d64"),
    pytest.param(64, True, Case(2, 3, 256), True, False, id="d64_causal"),
    pytest.param(128, False, Case(2, 3, 384), True, False, id="d128-S384"),
    pytest.param(128, True, Case(2, 3, 384), True, False, id="d128_causal-S384"),
    pytest.param(128, False, Case(1, 2, 256), False, True, id="d128-fused"),
    pytest.param(64, False, Case(1, 2, 256), False, True, id="d64-fused"),
]
# The shipped regions on the device (``pin`` None), with the one-launch route turned off (``fused`` False): 8192
# queries and keys at four (head, batch) pairs, on the route ``DECIDED_LEAST`` names for the architecture (the pair
# route without the mask everywhere, with it on sm_103a and sm_107a); 1024 queries and keys at sixteen (head, batch)
# pairs, on the 1-CTA route on every architecture; with the mask at 32 (head, batch) pairs, 8192 queries and keys and
# 7936 (both sides of the length bound of the long-sequence causal region of sm_103a), on the route ``DECIDED_LEAST``
# names for the architecture.
SHIPPED_ROUTE_PARAMS = [
    pytest.param(128, False, Case(1, 4, 8192), None, False, id="d128-S8192"),
    pytest.param(128, True, Case(1, 4, 8192), None, False, id="d128_causal-S8192"),
    pytest.param(128, False, Case(1, 16, 1024), None, False, id="d128-S1024"),
    pytest.param(128, True, Case(1, 16, 1024), None, False, id="d128_causal-S1024"),
    pytest.param(128, True, Case(1, 32, 8192), None, False, id="d128_causal-H32-S8192"),
    pytest.param(128, True, Case(1, 32, 7936), None, False, id="d128_causal-H32-S7936"),
]
# The source tests' preprocess cases at four heads, then one head and an odd head count.
PRE_CASES = (Case(1, 4, 128), Case(3, 4, 384), Case(2, 4, 1024), Case(2, 1, 256), Case(3, 5, 384))
# The source tests' postprocess cases (one, five and four heads).
POST_CASES = (Case(1, 1, 128), Case(3, 5, 256), Case(2, 4, 1024))


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the package ships {ARCHES}; this device is {capability}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert sorted(cake_jit.HEAD_DIMS) == sorted(HEAD_DIMS)
    assert cake_jit.BLOCK == 128
    assert cake_jit.LIBRARY_PREFIX == "cake_fmha_train"
    prefix = cake_jit.LIBRARY_PREFIX + "_"
    libraries = set()
    for name, rec in cake_jit.MODULES.items():
        assert rec["stage"] in STAGES, name
        assert name == rec["stage"] or name.startswith(rec["stage"] + "_"), name
        # Every variant is delivered for the three architectures.
        assert tuple(rec["arches"]) == ARCHES, (name, rec["arches"])
        assert rec["ffi_entry"] == "run", name
        assert re.fullmatch(re.escape(prefix) + "[0-9a-f]{20}", rec["library"]), (name, rec["library"])
        assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        launch = rec["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        # The 2-CTA main kernels run in clusters of two CTAs along x; every other kernel in single-CTA clusters.
        assert tuple(launch["cluster"]) == ((2, 1, 1) if name in PAIR_MAINS else (1, 1, 1)), name
        assert tuple(launch["block"]) == (BLOCK_THREADS[rec["stage"]], 1, 1), name
        libraries.add(rec["library"])
    assert len(libraries) == len(cake_jit.MODULES)
    assert set(PAIR_MAINS) | {PAIR_POST} <= set(cake_jit.MODULES)
    # The one-launch records serve the bwd_fused stage, and no other record does.
    assert dict(cake_jit.FUSED_OF) == FUSED_OF
    fused = {name for name, rec in cake_jit.MODULES.items() if rec["stage"] == "bwd_fused"}
    assert fused == set(FUSED_OF.values())


def test_routing_table_names_three_records_per_variant() -> None:
    """``RECORD_OF[(head_dim, causal, pair)]`` is the preprocess, main-stage and postprocess record of each variant:
    the preprocess depends on the head dimension only, the postprocess on the head dimension and the route, each
    variant has its own main stage, a 2-CTA main stage goes with the pair-row postprocess and no other (the pair
    route, head dimension 128 only), and every record of the registry serves some variant or is the one-launch record
    of ``FUSED_OF``, which no variant and no host sequence names."""
    table = cake_jit.RECORD_OF
    assert set(table) == set(VARIANTS)
    used = set()
    for variant, names in table.items():
        assert tuple(cake_jit.MODULES[name]["stage"] for name in names) == CLOSURE_STAGES, (variant, names)
        used.update(names)
    for D, causal, pair in VARIANTS:
        assert table[(D, causal, pair)][0] == table[(D, False, False)][0], (D, causal, pair)
        assert table[(D, causal, pair)][2] == table[(D, False, pair)][2], (D, causal, pair)
        _, main, post = table[(D, causal, pair)]
        assert (main in PAIR_MAINS) == (post == PAIR_POST) == pair, (D, causal, pair)
        assert D == 128 or not pair, (D, causal, pair)
    assert table[(64, False, False)][0] != table[(128, False, False)][0]
    assert len({table[variant][2] for variant in VARIANTS}) == 3
    assert len({table[variant][1] for variant in VARIANTS}) == len(VARIANTS)
    assert used.isdisjoint(cake_jit.FUSED_OF.values())
    assert used | set(cake_jit.FUSED_OF.values()) == set(cake_jit.MODULES)
    # The host sequence of each variant runs its three records, in order, on the three architectures.
    assert set(cake_jit.SEQUENCES) == set(table)
    for variant, sequence in cake_jit.SEQUENCES.items():
        assert tuple(sequence["records"]) == tuple(table[variant]), variant
        assert tuple(sequence["arches"]) == ARCHES, variant
        assert set(sequence["records"]).isdisjoint(cake_jit.FUSED_OF.values()), variant
    assert len({sequence["library"] for sequence in cake_jit.SEQUENCES.values()}) == len(VARIANTS)


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name, rec in cake_jit.MODULES.items():
        for arch in ARCHES:
            assert cake_backend.library_name(name, arch) == f"{rec['library']}_{arch}"
        for arch in ("sm_110a", "sm_90a", "sm_120a"):
            with pytest.raises(RuntimeError):
                cake_backend.library_name(name, arch)


def test_the_route_of_a_call() -> None:
    """``variant(D, causal, *, batch, heads, seqlen_q, seqlen_kv, arch)`` is the ``RECORD_OF`` key of a call: ``(D,
    causal, pair)`` with ``pair`` True for head dimension 128, a key length that is a multiple of 256 (an even key-block
    count: the 2-CTA kernel's CTA pairs), and either at least as many queries as keys and ``B * H`` and ``Skv`` in a
    region of ``PAIR_REGIONS[arch][causal]`` (each call of ``CAUSAL_PAIR_CALLS`` on its side of each bound of the
    long-sequence causal region of ``sm_103a``), or no mask and fewer queries than keys within the bounds
    ``PAIR_FEWER_QUERIES[arch]`` (the shipped table ``SHIPPED_PAIR_FEWER_QUERIES``; each call of
    ``FEWER_QUERIES_CALLS`` on its side of each bound), False otherwise (every head-dimension 64 call, every odd
    key-block count, every other call with more keys than queries).  Every bound is a measured value and ``PAIR_LEAST``
    the least length of each table; every measured crossover call is on its decided route (``DECIDED_LEAST``), and so
    is every measured call with fewer queries than keys (``DECIDED_FEWER_QUERIES``); a longer call stays on the pair
    route; an unmeasured ``B * H`` or length is on it exactly where its measured neighbours are; a call with at least
    as many queries as keys takes the route of ``PAIR_REGIONS`` alone."""
    assert set(cake_jit.PAIR_REGIONS) == set(ARCHES) and set(cake_jit.PAIR_LEAST) == set(ARCHES)
    assert dict(cake_jit.PAIR_FEWER_QUERIES) == SHIPPED_PAIR_FEWER_QUERIES
    for arch, regions in cake_jit.PAIR_REGIONS.items():
        assert set(regions) == {False, True}, arch
        for causal, boxes in regions.items():
            assert type(boxes) is tuple, (arch, causal)
            for least, most, length in boxes:
                assert least in MEASURED_COUNTS and (most is None or most in MEASURED_COUNTS), (arch, causal)
                assert most is None or most >= least, (arch, causal, least, most)
                assert length in MEASURED_LENGTHS, (arch, causal, length)
            assert cake_jit.PAIR_LEAST[arch][causal] == min((box[2] for box in boxes), default=None), (arch, causal)
    with pytest.raises(TypeError):  # every argument after ``causal`` is keyword-only
        cake_jit.variant(128, False, 1, 1, 256, 256, ARCHES[0])

    def pair(arch: str, causal: bool, count: int, Sq: int, Skv: int) -> bool:
        found = cake_jit.variant(128, causal, batch=1, heads=count, seqlen_q=Sq, seqlen_kv=Skv, arch=arch)
        swapped = cake_jit.variant(128, causal, batch=count, heads=1, seqlen_q=Sq, seqlen_kv=Skv, arch=arch)
        assert found == swapped, (arch, causal, count, Sq, Skv)
        return found[2]

    step = 2 * cake_jit.BLOCK
    # The causal calls on both sides of each bound of the long-sequence causal region of sm_103a, each on its side, and
    # the same calls at head dimension 64 on the 1-CTA route.
    for (count, S), routed in CAUSAL_PAIR_CALLS:
        assert pair("sm_103a", True, count, S, S) == routed, (count, S)
        found = cake_jit.variant(64, True, batch=1, heads=count, seqlen_q=S, seqlen_kv=S, arch="sm_103a")
        assert found == (64, True, False), (count, S, found)
    for arch in ARCHES:
        bounds = SHIPPED_PAIR_FEWER_QUERIES[arch]
        # Fewer queries than keys: each call on its side of each bound where the architecture has the bounds, and no
        # call on the pair route with the mask, at head dimension 64 or where the architecture has no bounds.
        for (count, Sq, Skv), routed in FEWER_QUERIES_CALLS:
            inside = Skv % step == 0 and _inside_fewer_queries((8, 4096, 8192, 2), count, Sq, Skv)
            assert Sq < Skv and routed == inside, (count, Sq, Skv)
            assert pair(arch, False, count, Sq, Skv) == (bounds is not None and routed), (arch, count, Sq, Skv)
            assert not pair(arch, True, count, Sq, Skv), (arch, count, Sq, Skv)
            found = cake_jit.variant(64, False, batch=1, heads=count, seqlen_q=Sq, seqlen_kv=Skv, arch=arch)
            assert found == (64, False, False), (arch, count, Sq, Skv, found)
        # Every measured call with fewer queries than keys on its decided route.
        for (B, H, Sq, Skv), routed in DECIDED_FEWER_QUERIES:
            found = cake_jit.variant(128, False, batch=B, heads=H, seqlen_q=Sq, seqlen_kv=Skv, arch=arch)
            assert Sq < Skv and found == (128, False, bounds is not None and routed), (arch, B, H, Sq, Skv, found)
        for causal in (False, True):
            for count, least in DECIDED_LEAST[arch][causal].items():
                for S in MEASURED_LENGTHS:
                    expected = least is not None and S >= least
                    assert pair(arch, causal, count, S, S) == expected, (arch, causal, count, S)
            for count in (1, 32, 128):
                assert not pair(arch, False, count, 128, 8192), (arch, count)
                assert not pair(arch, False, count, 8192, 256), (arch, count)
            # Every call: the head dimension, the key-block count, Sq >= Skv and the regions decide the route, and with
            # fewer queries than keys the mask and the bounds.
            for D in HEAD_DIMS:
                for count in (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 128, 129, 256):
                    for blocks in range(1, 70):
                        Skv = blocks * cake_jit.BLOCK
                        for Sq in (Skv,) if causal else (Skv, 128, 2048, 3968, 4096, 6144, 8192, 16384):
                            found = cake_jit.variant(
                                D, causal, batch=1, heads=count, seqlen_q=Sq, seqlen_kv=Skv, arch=arch
                            )
                            if Sq >= Skv:
                                inside = any(
                                    least <= count and (most is None or count <= most) and Skv >= length
                                    for least, most, length in cake_jit.PAIR_REGIONS[arch][causal]
                                )
                            else:
                                inside = not causal and _inside_fewer_queries(bounds, count, Sq, Skv)
                            expected = (D, causal, D == 128 and blocks % 2 == 0 and inside)
                            assert found == expected, (D, causal, count, Sq, Skv, arch, found)
                            assert type(found[1]) is bool and type(found[2]) is bool, found
                            assert found in cake_jit.RECORD_OF, found
            # A longer call stays on the pair route; a call with more queries than keys takes the route of its key
            # length and one with twice as many keys as queries the pair route exactly within the bounds; an unmeasured
            # B * H or length is on the pair route exactly where its measured neighbours are (the largest measured value
            # stands for every larger one).
            for count in range(1, 300):
                below = max(c for c in MEASURED_COUNTS if c <= count)
                above = min((c for c in MEASURED_COUNTS if c >= count), default=below)
                for S in range(step, 16385, step):
                    on = pair(arch, causal, count, S, S)
                    shorter = max(s for s in MEASURED_LENGTHS if s <= S)
                    longer = min((s for s in MEASURED_LENGTHS if s >= S), default=shorter)
                    around = all(pair(arch, causal, c, s, s) for c in (below, above) for s in (shorter, longer))
                    assert on == around, (arch, causal, count, S)
                    if on:
                        assert pair(arch, causal, count, S + step, S + step), (arch, causal, count, S)
                    if not causal:
                        assert pair(arch, causal, count, 2 * S, S) == on, (arch, count, S)
                        fewer = _inside_fewer_queries(bounds, count, S, 2 * S)
                        assert pair(arch, causal, count, S, 2 * S) == fewer, (arch, count, S)


# Calls on both sides of each bound of the region of one piece (max_ctas, s_max, skv_any, skv_per_sq) =
# (64, 1024, 512, 2), as ((B, H, Sq, Skv), routed): the CTAs of the one-launch grid B * H * (Sq / 128 + Skv / 128) at
# 63, 64 and 66; the longer sequence at 1024 and 1152; the key length at 512 and 640 against 128 queries; the key length
# at twice and past twice 384 queries.
FUSED_REGION_PROBE = ((64, 1024, 512, 2),)
FUSED_REGION_CALLS = (
    ((21, 1, 128, 256), True),
    ((22, 1, 128, 256), False),
    ((32, 1, 128, 128), True),
    ((33, 1, 128, 128), False),
    ((1, 32, 128, 128), True),
    ((1, 33, 128, 128), False),
    ((1, 1, 1024, 128), True),
    ((1, 1, 1152, 128), False),
    ((1, 1, 512, 1024), True),
    ((1, 1, 640, 1152), False),
    ((1, 1, 128, 512), True),
    ((1, 1, 128, 640), False),
    ((1, 1, 384, 768), True),
    ((1, 1, 384, 896), False),
)
# A region of two pieces: the piece above and (192, 512, 128, 0).  Calls as ((B, H, Sq, Skv), routed): inside both
# pieces; inside the first piece only; inside the second piece only, outside the first one by its CTAs, on the inner
# side of each bound of the second piece (the CTAs at 192 with 256 queries and 128 keys and with 128 queries and keys,
# the longer sequence at 512, the key length at 128); outside both pieces, one step past each bound of the second piece
# (the CTAs at 195 and 194, the longer sequence at 640, 256 keys against 128 queries and against 512 queries).
FUSED_REGION_PROBE_TWO = ((64, 1024, 512, 2), (192, 512, 128, 0))
FUSED_REGION_TWO_CALLS = (
    ((1, 1, 128, 128), True),
    ((1, 1, 128, 512), True),
    ((1, 33, 128, 128), True),
    ((1, 64, 256, 128), True),
    ((1, 65, 256, 128), False),
    ((2, 48, 128, 128), True),
    ((97, 1, 128, 128), False),
    ((1, 32, 512, 128), True),
    ((1, 32, 640, 128), False),
    ((1, 40, 256, 128), True),
    ((1, 40, 128, 256), False),
    ((1, 16, 512, 256), False),
)


def test_the_one_launch_route_of_a_call(monkeypatch) -> None:
    """``fused_record(arch, B, H, Sq, Skv, D, causal)`` is ``FUSED_OF[(D, causal)]`` for a call without the mask
    inside the region ``FUSED_REGION[(arch, D)]``, a tuple of one or more pieces ``(max_ctas, s_max, skv_any,
    skv_per_sq)``: inside a piece when ``B * H * (Sq / 128 + Skv / 128) <= max_ctas``, ``max(Sq, Skv) <= s_max`` and
    ``Skv <= skv_any`` or ``Skv <= skv_per_sq * Sq``, and inside the region when inside any of its pieces; None
    otherwise, for every causal call and wherever the region is None.  Checked on region tables with one entry set to
    one piece of distinct bounds, every call on both sides of each bound, and to two pieces, a call inside only the
    second piece routed and a call outside both not; and on the shipped table."""
    assert dict(cake_jit.FUSED_OF) == FUSED_OF
    shipped = dict(cake_jit.FUSED_REGION)
    assert set(shipped) == {(arch, D) for arch in ARCHES for D in HEAD_DIMS}
    calls = [shape for shape, _ in FUSED_REGION_CALLS]
    for (arch, D), region in shipped.items():
        assert region is None or (type(region) is tuple and len(region) >= 1), (arch, D)
        for piece in region or ():
            assert type(piece) is tuple and len(piece) == 4, (arch, D, piece)
            assert all(type(n) is int and n >= 0 for n in piece), (arch, D, piece)
        for B, H, Sq, Skv in calls:
            assert cake_jit.fused_record(arch, B, H, Sq, Skv, D, True) is None, (arch, D, B, H, Sq, Skv)
            if region is None:
                assert cake_jit.fused_record(arch, B, H, Sq, Skv, D, False) is None, (arch, D, B, H, Sq, Skv)
    probes = ((FUSED_REGION_PROBE, FUSED_REGION_CALLS), (FUSED_REGION_PROBE_TWO, FUSED_REGION_TWO_CALLS))
    for probe, probe_calls in probes:
        for arch in ARCHES:
            for D in HEAD_DIMS:
                table = dict.fromkeys(shipped)
                table[(arch, D)] = probe
                monkeypatch.setattr(cake_jit, "FUSED_REGION", table)
                for (B, H, Sq, Skv), routed in probe_calls:
                    found = cake_jit.fused_record(arch, B, H, Sq, Skv, D, False)
                    assert found == (FUSED_OF[(D, False)] if routed else None), (probe, arch, D, B, H, Sq, Skv, found)
                    assert cake_jit.fused_record(arch, B, H, Sq, Skv, D, True) is None, (arch, D, B, H, Sq, Skv)
                    for other in table:
                        if other != (arch, D):
                            assert cake_jit.fused_record(other[0], B, H, Sq, Skv, other[1], False) is None, other


# The shipped regions of the one-launch route, per (architecture, head dimension): the pieces ``(max_ctas, s_max,
# skv_any, skv_per_sq)`` of each region.  ``FUSED_REGION`` must be this table.
SHIPPED_FUSED_REGION = {
    ("sm_100a", 128): (
        (152, 2048, 512, 1),
        (190, 1024, 256, 0),
        (152, 1024, 0, 3),
        (640, 512, 256, 0),
        (301, 1024, 128, 0),
        (128, 8192, 0, 1),
    ),
    ("sm_100a", 64): (
        (152, 2048, 512, 2),
        (190, 1024, 256, 0),
        (152, 1024, 0, 6),
        (606, 256, 128, 0),
        (212, 256, 0, 1),
    ),
    ("sm_103a", 128): (
        (152, 2048, 512, 1),
        (190, 1024, 256, 0),
        (152, 1024, 0, 3),
        (303, 512, 256, 0),
        (301, 1024, 128, 0),
        (128, 8192, 0, 1),
    ),
    ("sm_103a", 64): (
        (204, 2048, 256, 0),
        (156, 512, 0, 2),
        (256, 1024, 128, 0),
        (152, 1024, 512, 3),
        (152, 2048, 0, 1),
    ),
    ("sm_107a", 128): (
        (256, 2048, 512, 0),
        (320, 1024, 256, 0),
        (423, 512, 256, 0),
        (210, 2048, 768, 2),
        (423, 1024, 128, 0),
        (128, 8192, 0, 1),
    ),
    ("sm_107a", 64): (
        (212, 2048, 512, 2),
        (210, 1024, 0, 6),
        (423, 512, 256, 0),
        (320, 1024, 256, 0),
        (423, 1024, 128, 0),
    ),
}
# Calls on both sides of each bound of each shipped piece, as ((B, H, Sq, Skv), inside the piece), every other bound of
# the piece held: the CTAs of the one-launch grid at the most CTAs within max_ctas and one step past it, at 128 and at
# 256 queries and keys (256 queries and 128 keys where the piece holds no 256 keys); the longer sequence at s_max and
# 128 past it (the queries, and with skv_per_sq > 0 the keys too); with skv_any > 0 the key length at skv_any and 128
# past it against 128 queries; the key length at skv_per_sq times the queries and 128 past it, above skv_any, or with
# skv_per_sq 0 at skv_any and 128 past it against s_max queries (a call listed once per piece).  A call outside one
# piece may lie inside another piece of the same region; the call is routed when it is inside any piece of the region.
SHIPPED_FUSED_CALLS = {
    (152, 2048, 512, 1): (
        ((1, 76, 128, 128), True),
        ((77, 1, 128, 128), False),
        ((2, 19, 256, 256), True),
        ((3, 13, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 2048, 2048), True),
        ((1, 1, 2176, 2176), False),
        ((1, 1, 128, 512), True),
        ((1, 1, 128, 640), False),
        ((1, 1, 640, 640), True),
        ((1, 1, 640, 768), False),
    ),
    (152, 2048, 512, 2): (
        ((1, 76, 128, 128), True),
        ((77, 1, 128, 128), False),
        ((2, 19, 256, 256), True),
        ((3, 13, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 2048, 2048), True),
        ((1, 1, 2176, 2176), False),
        ((1, 1, 128, 512), True),
        ((1, 1, 128, 640), False),
        ((1, 1, 384, 768), True),
        ((1, 1, 384, 896), False),
    ),
    (204, 2048, 256, 0): (
        ((1, 102, 128, 128), True),
        ((103, 1, 128, 128), False),
        ((3, 17, 256, 256), True),
        ((4, 13, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
        ((1, 1, 2048, 256), True),
        ((1, 1, 2048, 384), False),
    ),
    (256, 2048, 512, 0): (
        ((1, 128, 128, 128), True),
        ((129, 1, 128, 128), False),
        ((4, 16, 256, 256), True),
        ((5, 13, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 128, 512), True),
        ((1, 1, 128, 640), False),
        ((1, 1, 2048, 512), True),
        ((1, 1, 2048, 640), False),
    ),
    (212, 2048, 512, 2): (
        ((1, 106, 128, 128), True),
        ((107, 1, 128, 128), False),
        ((1, 53, 256, 256), True),
        ((2, 27, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 2048, 2048), True),
        ((1, 1, 2176, 2176), False),
        ((1, 1, 128, 512), True),
        ((1, 1, 128, 640), False),
        ((1, 1, 384, 768), True),
        ((1, 1, 384, 896), False),
    ),
    (190, 1024, 256, 0): (
        ((1, 95, 128, 128), True),
        ((96, 1, 128, 128), False),
        ((1, 47, 256, 256), True),
        ((2, 24, 256, 256), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
        ((1, 1, 1024, 256), True),
        ((1, 1, 1024, 384), False),
    ),
    (152, 1024, 0, 6): (
        ((1, 76, 128, 128), True),
        ((77, 1, 128, 128), False),
        ((2, 19, 256, 256), True),
        ((3, 13, 256, 256), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 1024, 1024), True),
        ((1, 1, 1152, 1152), False),
        ((1, 1, 128, 768), True),
        ((1, 1, 128, 896), False),
    ),
    (152, 1024, 0, 3): (
        ((1, 76, 128, 128), True),
        ((77, 1, 128, 128), False),
        ((2, 19, 256, 256), True),
        ((3, 13, 256, 256), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 1024, 1024), True),
        ((1, 1, 1152, 1152), False),
        ((1, 1, 128, 384), True),
        ((1, 1, 128, 512), False),
    ),
    (156, 512, 0, 2): (
        ((1, 78, 128, 128), True),
        ((79, 1, 128, 128), False),
        ((3, 13, 256, 256), True),
        ((2, 20, 256, 256), False),
        ((1, 1, 512, 128), True),
        ((1, 1, 640, 128), False),
        ((1, 1, 512, 512), True),
        ((1, 1, 640, 640), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
    ),
    (256, 1024, 128, 0): (
        ((1, 128, 128, 128), True),
        ((129, 1, 128, 128), False),
        ((1, 85, 256, 128), True),
        ((2, 43, 256, 128), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
        ((1, 1, 1024, 256), False),
    ),
    (320, 1024, 256, 0): (
        ((1, 160, 128, 128), True),
        ((161, 1, 128, 128), False),
        ((4, 20, 256, 256), True),
        ((3, 27, 256, 256), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
        ((1, 1, 1024, 256), True),
        ((1, 1, 1024, 384), False),
    ),
    (423, 512, 256, 0): (
        ((1, 211, 128, 128), True),
        ((212, 1, 128, 128), False),
        ((3, 35, 256, 256), True),
        ((2, 53, 256, 256), False),
        ((1, 1, 512, 128), True),
        ((1, 1, 640, 128), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
        ((1, 1, 512, 256), True),
        ((1, 1, 512, 384), False),
    ),
    (210, 1024, 0, 6): (
        ((1, 105, 128, 128), True),
        ((106, 1, 128, 128), False),
        ((4, 13, 256, 256), True),
        ((1, 53, 256, 256), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 1024, 1024), True),
        ((1, 1, 1152, 1152), False),
        ((1, 1, 128, 768), True),
        ((1, 1, 128, 896), False),
    ),
    (640, 512, 256, 0): (
        ((1, 320, 128, 128), True),
        ((321, 1, 128, 128), False),
        ((1, 160, 256, 256), True),
        ((161, 1, 256, 256), False),
        ((1, 1, 512, 128), True),
        ((1, 1, 640, 128), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
        ((1, 1, 512, 256), True),
        ((1, 1, 512, 384), False),
    ),
    (301, 1024, 128, 0): (
        ((1, 150, 128, 128), True),
        ((151, 1, 128, 128), False),
        ((1, 100, 256, 128), True),
        ((101, 1, 256, 128), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
        ((1, 1, 1024, 256), False),
    ),
    (606, 256, 128, 0): (
        ((1, 303, 128, 128), True),
        ((304, 1, 128, 128), False),
        ((1, 202, 256, 128), True),
        ((203, 1, 256, 128), False),
        ((1, 1, 256, 128), True),
        ((1, 1, 384, 128), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
        ((1, 1, 256, 256), False),
    ),
    (212, 256, 0, 1): (
        ((1, 106, 128, 128), True),
        ((107, 1, 128, 128), False),
        ((1, 53, 256, 256), True),
        ((2, 27, 256, 256), False),
        ((1, 1, 256, 128), True),
        ((1, 1, 384, 128), False),
        ((1, 1, 256, 256), True),
        ((1, 1, 384, 384), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
    ),
    (303, 512, 256, 0): (
        ((1, 151, 128, 128), True),
        ((152, 1, 128, 128), False),
        ((1, 75, 256, 256), True),
        ((76, 1, 256, 256), False),
        ((1, 1, 512, 128), True),
        ((1, 1, 640, 128), False),
        ((1, 1, 128, 256), True),
        ((1, 1, 128, 384), False),
        ((1, 1, 512, 256), True),
        ((1, 1, 512, 384), False),
    ),
    (152, 1024, 512, 3): (
        ((1, 76, 128, 128), True),
        ((77, 1, 128, 128), False),
        ((2, 19, 256, 256), True),
        ((3, 13, 256, 256), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 1024, 1024), True),
        ((1, 1, 1152, 1152), False),
        ((1, 1, 128, 512), True),
        ((1, 1, 128, 640), False),
        ((1, 1, 256, 768), True),
        ((1, 1, 256, 896), False),
    ),
    (152, 2048, 0, 1): (
        ((1, 76, 128, 128), True),
        ((77, 1, 128, 128), False),
        ((2, 19, 256, 256), True),
        ((3, 13, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 2048, 2048), True),
        ((1, 1, 2176, 2176), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
    ),
    (210, 2048, 768, 2): (
        ((1, 105, 128, 128), True),
        ((106, 1, 128, 128), False),
        ((4, 13, 256, 256), True),
        ((1, 53, 256, 256), False),
        ((1, 1, 2048, 128), True),
        ((1, 1, 2176, 128), False),
        ((1, 1, 2048, 2048), True),
        ((1, 1, 2176, 2176), False),
        ((1, 1, 128, 768), True),
        ((1, 1, 128, 896), False),
        ((1, 1, 512, 1024), True),
        ((1, 1, 512, 1152), False),
    ),
    (423, 1024, 128, 0): (
        ((1, 211, 128, 128), True),
        ((212, 1, 128, 128), False),
        ((1, 141, 256, 128), True),
        ((142, 1, 256, 128), False),
        ((1, 1, 1024, 128), True),
        ((1, 1, 1152, 128), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
        ((1, 1, 1024, 256), False),
    ),
    (128, 8192, 0, 1): (
        ((1, 64, 128, 128), True),
        ((65, 1, 128, 128), False),
        ((1, 32, 256, 256), True),
        ((33, 1, 256, 256), False),
        ((1, 1, 8192, 128), True),
        ((1, 1, 8320, 128), False),
        ((1, 1, 8192, 8192), True),
        ((1, 1, 8320, 8320), False),
        ((1, 1, 128, 128), True),
        ((1, 1, 128, 256), False),
    ),
}
# The benchmark's long rows ``T1`` to ``T4``, 8192 queries and keys, as (B, H, D, causal).
LONG_ROWS = ((4, 32, 128, False), (4, 32, 128, True), (8, 32, 64, False), (8, 32, 64, True))
# The decided route of the measured long calls (head dimension 128, no mask, 4096 or 8192 queries and keys, up to four
# heads), as ((B, H, Sq, Skv), routed) on every architecture.  This is the decision taken from the measurement, not a
# measurement: the routed calls (128 CTAs or fewer) were faster on the one-launch route than 0.1.8 (the three-kernel
# routes) in both replicate runs on SM100, SM103 and SM107; the calls on 256 CTAs or more were slower there and stay on
# the three-kernel routes.  ``FUSED_REGION`` must reproduce it.
DECIDED_LONG_CALLS = (
    ((1, 1, 4096, 4096), True),
    ((1, 1, 8192, 4096), True),
    ((1, 2, 4096, 4096), True),
    ((1, 2, 8192, 8192), False),
    ((1, 4, 8192, 4096), False),
    ((1, 4, 8192, 8192), False),
)


def _inside_piece(piece: tuple[int, int, int, int], B: int, H: int, Sq: int, Skv: int) -> bool:
    """A call without the mask inside one piece ``(max_ctas, s_max, skv_any, skv_per_sq)`` of a region."""
    max_ctas, s_max, skv_any, skv_per_sq = piece
    fits = B * H * (Sq // 128 + Skv // 128) <= max_ctas and max(Sq, Skv) <= s_max
    return fits and (Skv <= skv_any or Skv <= skv_per_sq * Sq)


# The query and key lengths searched for a call that only one piece of a region holds (up to the longest ``s_max`` of
# a shipped piece).
PIECE_ONLY_LENGTHS = (128, 256, 512, 768, 1024, 2048, 4096, 8192)


def _piece_only_call(region: tuple[tuple[int, int, int, int], ...], index: int) -> tuple[int, int, int, int] | None:
    """A call ``(1, H, Sq, Skv)`` inside the piece ``region[index]`` and inside no other piece of ``region``, with
    ``Sq`` and ``Skv`` in ``PIECE_ONLY_LENGTHS`` (the first such lengths, the most heads), or None."""
    piece = region[index]
    others = [other for at, other in enumerate(region) if at != index]
    for Sq in PIECE_ONLY_LENGTHS:
        for Skv in PIECE_ONLY_LENGTHS:
            for H in range(piece[0] // (Sq // 128 + Skv // 128), 0, -1):
                if _inside_piece(piece, 1, H, Sq, Skv) and not any(_inside_piece(o, 1, H, Sq, Skv) for o in others):
                    return 1, H, Sq, Skv
    return None


def test_the_shipped_one_launch_regions() -> None:
    """``FUSED_REGION`` is the shipped table ``SHIPPED_FUSED_REGION``, and ``fused_record`` keeps each region on both
    sides of each bound of each of its pieces: the one-launch record for a call inside any piece of the region (a call
    one step past a bound of one piece may lie inside another piece), None for a call inside no piece and for every
    causal call.  Every piece of a region routes a call that no other piece of the region holds.  The benchmark's long
    rows take the three-kernel routes on every architecture, and every measured long call its decided route
    (``DECIDED_LONG_CALLS``)."""
    assert dict(cake_jit.FUSED_REGION) == SHIPPED_FUSED_REGION
    pieces = {piece for region in SHIPPED_FUSED_REGION.values() for piece in region}
    assert set(SHIPPED_FUSED_CALLS) == pieces, sorted(set(SHIPPED_FUSED_CALLS) ^ pieces)
    for (arch, D), region in SHIPPED_FUSED_REGION.items():
        for piece in region:
            for (B, H, Sq, Skv), inside in SHIPPED_FUSED_CALLS[piece]:
                assert _inside_piece(piece, B, H, Sq, Skv) == inside, (piece, B, H, Sq, Skv)
                routed = any(_inside_piece(other, B, H, Sq, Skv) for other in region)
                found = cake_jit.fused_record(arch, B, H, Sq, Skv, D, False)
                assert found == (FUSED_OF[(D, False)] if routed else None), (arch, D, piece, B, H, Sq, Skv, found)
                assert cake_jit.fused_record(arch, B, H, Sq, Skv, D, True) is None, (arch, D, B, H, Sq, Skv)
        for index, piece in enumerate(region):
            call = _piece_only_call(region, index)
            assert call is not None, (arch, D, piece)
            B, H, Sq, Skv = call
            found = cake_jit.fused_record(arch, B, H, Sq, Skv, D, False)
            assert found == FUSED_OF[(D, False)], (arch, D, piece, call, found)
    for arch in ARCHES:
        for B, H, D, causal in LONG_ROWS:
            assert cake_jit.fused_record(arch, B, H, 8192, 8192, D, causal) is None, (arch, B, H, D, causal)
        for (B, H, Sq, Skv), routed in DECIDED_LONG_CALLS:
            found = cake_jit.fused_record(arch, B, H, Sq, Skv, 128, False)
            assert found == (FUSED_OF[(128, False)] if routed else None), (arch, B, H, Sq, Skv, found)


def test_the_accumulator_layouts() -> None:
    """The judge's helper writes and reads the FP32 dQ accumulator in the layout of each main kernel: per 128-row tile
    of a head, element ``(r, d)`` at ``(d // 4) * 512 + r * 4 + d % 4`` (the 1-CTA kernels) or at ``(r // 64) * 8192 +
    (d // 4) * 256 + (r % 64) * 4 + d % 4`` (the 2-CTA kernel, head dimension 128); each layout reads back."""
    B, H, S, D = 2, 3, 384, 128
    values = torch.arange(B * H * S * D, dtype=torch.float64).view(B * H, S, D)
    for pair in (False, True):
        acc = accumulator(values, pair=pair)
        assert acc.shape == (B * H * S * D,)
        for bh, row, d in ((0, 0, 0), (1, 5, 7), (2, 63, 127), (3, 64, 1), (4, 130, 66), (5, 383, 125)):
            tile, r = divmod(row, 128)
            if pair:
                at = (r // 64) * 8192 + (d // 4) * 256 + (r % 64) * 4 + d % 4
            else:
                at = (d // 4) * 512 + r * 4 + d % 4
            assert acc[(bh * (S // 128) + tile) * 128 * D + at] == values[bh, row, d], (pair, bh, row, d)
        back = from_accumulator(acc, B, H, S, D, pair=pair)
        assert torch.equal(back, values.view(B, H, S, D)), pair
    assert not torch.equal(accumulator(values, pair=True), accumulator(values, pair=False))


def test_grids_cover_every_tile() -> None:
    """A CTA per 128-row query tile of each (head, batch) for the preprocess and the postprocess, per 128-key block of
    each (head, batch) for the main stage, and for the one-launch stage one per query tile (the Q-owners) then one per
    key block (the KV-owners): the grid is ``(tiles, H, B)`` (the kernels read the head from ``blockIdx.y`` and the
    batch from ``blockIdx.z``).  A call on the pair route has an even key-block count, so the 2-CTA main kernel's
    clusters of two CTAs along x tile its grid."""
    rng = random.Random(7091)
    for _ in range(200):
        B, H = rng.randint(1, 64), rng.randint(1, 64)
        Sq, Skv = 128 * rng.randint(1, 96), 128 * rng.randint(1, 96)
        dims = dict(batch=B, heads=H, seqlen_q=Sq, seqlen_kv=Skv)
        assert tuple(cake_jit.grid("bwd_pre", **dims)) == (Sq // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_main", **dims)) == (Skv // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_post", **dims)) == (Sq // 128, H, B), dims
        assert tuple(cake_jit.grid("bwd_fused", **dims)) == (Sq // 128 + Skv // 128, H, B), dims
        for arch in ARCHES:
            if cake_jit.variant(128, False, batch=B, heads=H, seqlen_q=Sq, seqlen_kv=Skv, arch=arch)[2]:
                assert cake_jit.grid("bwd_main", **dims)[0] % 2 == 0, (dims, arch)


def test_stage_launchers_take_named_buffers() -> None:
    """Each stage launcher takes the record name, then its buffers and scalars by keyword only."""
    for launcher, keywords in LAUNCHER_KEYWORDS.items():
        params = list(inspect.signature(getattr(cake_backend, launcher)).parameters.values())
        assert params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD), launcher
        rest = params[1:]
        assert all(p.kind is p.KEYWORD_ONLY for p in rest), launcher
        assert sorted(p.name for p in rest) == sorted(keywords), launcher


@pytest.mark.parametrize(("D", "causal", "pair"), VARIANTS, ids=VARIANT_IDS)
def test_stage_launchers_pass_the_registry_arguments_positionally(D, causal, pair, monkeypatch) -> None:
    """Every record ``RECORD_OF`` routes to is launched positionally by its stage launcher, and its FFI entry gets the
    same arguments and grid, of the same types, as through the registry-ordered launch of ``_launch`` (no device: the
    FFI entry is a recorder)."""
    calls = []
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: ARCHES[0])
    monkeypatch.setattr(cake_backend, "_entry", lambda name, arch: lambda *args: calls.append((name, args)))
    # Sizes and tile counts all distinct, so a swapped integer argument or grid axis changes the recorded call (an even
    # key-block count on the pair route).
    B, H, Sq, Skv = 2, 3, 5 * cake_jit.BLOCK, (8 if pair else 7) * cake_jit.BLOCK
    assert len({B, H, Sq, Skv, Sq // cake_jit.BLOCK, Skv // cake_jit.BLOCK}) == 6
    bf16, fp32 = {"dtype": torch.bfloat16}, {"dtype": torch.float32}
    q, out, dout, dq = (torch.empty(B, Sq, H, D, **bf16) for _ in range(4))
    k, v, dk, dv = (torch.empty(B, Skv, H, D, **bf16) for _ in range(4))
    lse = torch.empty(B, H, Sq, **fp32)
    lse_log2, dpsum = torch.empty(B * H, Sq, **fp32), torch.empty(B * H, Sq, **fp32)
    dq_accum = torch.empty(B * H * Sq * D, **fp32)
    pre, main, post = cake_jit.RECORD_OF[(D, causal, pair)]

    def launch_all() -> list:
        calls.clear()
        cake_backend.launch_bwd_pre(pre, out=out, dout=dout, lse=lse, lse_log2=lse_log2, dpsum=dpsum, dq_accum=dq_accum)
        cake_backend.launch_bwd_main(
            main,
            q=q,
            k=k,
            v=v,
            dout=dout,
            lse_log2=lse_log2,
            dpsum=dpsum,
            dq_accum=dq_accum,
            dk=dk,
            dv=dv,
            softmax_scale=0.0883883,
        )
        cake_backend.launch_bwd_post(post, dq_accum=dq_accum, dq=dq, softmax_scale=0.0883883)
        return list(calls)

    for stage, name in zip(CLOSURE_STAGES, (pre, main, post), strict=True):
        assert cake_backend._positional(name, stage), (stage, name)
    positional = launch_all()
    monkeypatch.setattr(cake_backend, "_positional", lambda name, stage: False)
    registry = launch_all()
    assert [name for name, _ in positional] == [name for name, _ in registry] == [pre, main, post]
    for (name, got), (_, want) in zip(positional, registry, strict=True):
        assert len(got) == len(want), name
        for i, (a, b) in enumerate(zip(got, want, strict=True)):
            assert type(a) is type(b), (name, i, type(a), type(b))
            assert a is b if isinstance(a, torch.Tensor) else a == b, (name, i, a, b)


@pytest.mark.parametrize("D", HEAD_DIMS, ids=[f"d{D}" for D in HEAD_DIMS])
def test_the_fused_launcher_passes_the_registry_arguments_positionally(D, monkeypatch) -> None:
    """Every ``FUSED_OF`` record is launched positionally by ``launch_bwd_fused``, and its FFI entry gets the same
    arguments and grid, of the same types, as through the registry-ordered launch of ``_launch`` (no device: the FFI
    entry is a recorder)."""
    name = cake_jit.FUSED_OF[(D, False)]
    calls = []
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: ARCHES[0])
    monkeypatch.setattr(cake_backend, "_entry", lambda name, arch: lambda *args: calls.append((name, args)))
    # Sizes and tile counts all distinct, so a swapped integer argument or grid axis changes the recorded call.
    B, H, Sq, Skv = 2, 3, 5 * cake_jit.BLOCK, 7 * cake_jit.BLOCK
    tiles = (Sq // cake_jit.BLOCK, Skv // cake_jit.BLOCK, (Sq + Skv) // cake_jit.BLOCK)
    assert len({B, H, Sq, Skv, *tiles}) == 7
    bf16 = {"dtype": torch.bfloat16}
    q, out, dout, dq = (torch.empty(B, Sq, H, D, **bf16) for _ in range(4))
    k, v, dk, dv = (torch.empty(B, Skv, H, D, **bf16) for _ in range(4))
    lse = torch.empty(B, H, Sq, dtype=torch.float32)

    def launch() -> list:
        calls.clear()
        cake_backend.launch_bwd_fused(
            name, q=q, k=k, v=v, out=out, lse=lse, dout=dout, dq=dq, dk=dk, dv=dv, softmax_scale=0.0883883
        )
        return list(calls)

    assert cake_backend._positional(name, "bwd_fused"), name
    positional = launch()
    monkeypatch.setattr(cake_backend, "_positional", lambda name, stage: False)
    registry = launch()
    assert [found for found, _ in positional] == [found for found, _ in registry] == [name]
    (_, got), (_, want) = positional[0], registry[0]
    assert len(got) == len(want), name
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert type(a) is type(b), (name, i, type(a), type(b))
        assert a is b if isinstance(a, torch.Tensor) else a == b, (name, i, a, b)
    assert tuple(got[-3:]) == (tiles[2], H, B), got[-3:]


def test_the_package_exports_the_backend_entry() -> None:
    assert pkg.cake_fmha_nondet_train_backward is cake_backend.cake_fmha_nondet_train_backward
    assert ENTRY in pkg.__all__ and ENTRY in cake_backend.__all__


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    libraries = (*cake_jit.MODULES.values(), *cake_jit.SEQUENCES.values())
    wanted = {f"{rec['library']}_{arch}" for rec in libraries for arch in rec["arches"]}
    assert set(entries) == wanted
    assert len(entries) == len(libraries) * len(ARCHES)
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert name.endswith("_" + entry["arch"]), name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_the_loader_imports_standalone_by_path() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a fresh process reads the manifest."""
    _manifest_or_skip()
    loader = str(PACKAGE_DIR / "cake_build.py")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {loader!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    records = sum(len(rec["arches"]) for rec in (*cake_jit.MODULES.values(), *cake_jit.SEQUENCES.values()))
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path; without one the JIT spec names the entry."""
    _manifest_or_skip()
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_backend.library_name(name, arch)
            assert cake_build.prebuilt_path(library) == PACKAGE_DIR / "aot" / f"{library}.so"
            assert cake_backend.is_prebuilt(name, arch) == cake_build.prebuilt_path(library).is_file()


# The variables the JIT shim (``_cake_pplx_jit.py``) reads: the toolchain, cache and JIT switches.
JIT_SHIM_ENVIRONMENT_VARIABLES = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CAKE_PPLX_CACHE_DIR",
        "CAKE_PPLX_NVCC",
        "CXX",
        "CAKE_PPLX_JIT_WORKERS",
        "CAKE_PPLX_DISABLE_JIT",
    }
)


def _environment_reads(tree: ast.AST) -> tuple[set[str], list[int]]:
    """``(names, unresolved)``: the variable names ``tree`` reads through ``os.environ.get(...)``,
    ``os.environ[...]``, ``... in os.environ`` or ``os.getenv(...)`` -- a string literal, or a loop variable over a
    tuple / list of string literals -- and the lines of every other ``environ`` / ``getenv`` use."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def keys(expr: ast.AST, at: ast.AST) -> set[str] | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.Name):
            node = parents.get(at)
            while node is not None:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == expr.id
                    and isinstance(node.iter, (ast.Tuple, ast.List))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)
                ):
                    return {e.value for e in node.iter.elts}
                node = parents.get(node)
        return None

    names: set[str] = set()
    unresolved: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")):
            continue
        parent = parents.get(node)
        found = None
        if node.attr == "getenv":
            if isinstance(parent, ast.Call) and parent.func is node and parent.args:
                found = keys(parent.args[0], parent)
        elif isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and call.args:
                found = keys(call.args[0], call)
        elif isinstance(parent, ast.Subscript) and parent.value is node:
            found = keys(parent.slice, parent)
        elif (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is node
        ):
            found = keys(parent.left, parent)
        if found is None:
            unresolved.append(node.lineno)
        else:
            names |= found
    return names, unresolved


def test_the_package_reads_no_environment_variable() -> None:
    """No module of the package reads an environment variable, except the JIT shim (``_cake_pplx_jit.py``), which
    reads exactly :data:`JIT_SHIM_ENVIRONMENT_VARIABLES`, each by a literal name."""
    offenders = []
    shim = None
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        names, unresolved = _environment_reads(ast.parse(path.read_text()))
        if path.name == "_cake_pplx_jit.py":
            shim = names
            offenders += [f"{path.name}:{line} (variable not resolved)" for line in unresolved]
        else:
            offenders += [f"{path.name}:{line}" for line in unresolved]
            offenders += [f"{path.name}: {name}" for name in sorted(names)]
    assert not offenders, offenders
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES, (
        sorted(shim - JIT_SHIM_ENVIRONMENT_VARIABLES) if shim is not None else "no _cake_pplx_jit.py",
        sorted(JIT_SHIM_ENVIRONMENT_VARIABLES - shim) if shim is not None else None,
    )


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: every record loads


def test_the_records_of_every_variant_load(device) -> None:
    arch = cake_backend.device_arch(device.index)
    assert arch in ARCHES
    for variant, names in cake_jit.RECORD_OF.items():
        for name in names:
            assert callable(getattr(cake_backend.load(name, arch), cake_jit.MODULES[name]["ffi_entry"])), (
                variant,
                name,
            )
    for key, name in cake_jit.FUSED_OF.items():
        assert callable(getattr(cake_backend.load(name, arch), cake_jit.MODULES[name]["ffi_entry"])), (key, name)


# ----------------------------------------------------------------------------------------------------------------
# Each stage on its own, judged against the naive BF16 tier


@pytest.mark.parametrize("case", PRE_CASES, ids=[case.id for case in PRE_CASES])
@pytest.mark.parametrize("D", HEAD_DIMS, ids=[f"d{D}" for D in HEAD_DIMS])
def test_the_preprocess_matches_the_reference(D, case, device, record_property) -> None:
    """``dPsum`` and ``LSE_log2`` (0 on the rows whose ``lse`` is ``-inf``) meet the tier, and ``dQ_accum`` is
    cleared to exactly 0 (the tier's error there is 0)."""
    name = cake_jit.RECORD_OF[(D, False, False)][0]

    def draw(seed: int):
        t = pre_inputs(case, D, seed=seed, device=device)
        cake_backend.launch_bwd_pre(
            name, out=t.out, dout=t.dout, lse=t.lse, lse_log2=t.lse_log2, dpsum=t.dpsum, dq_accum=t.dq_accum
        )
        return judged_stage(pre_outputs, t, {"dPsum": t.dpsum, "LSE_log2": t.lse_log2, "dQ_accum": t.dq_accum})

    judge_draws(record_property, draw)


@pytest.mark.parametrize(("D", "causal", "pair", "case"), MAIN_PARAMS)
def test_the_main_stage_matches_the_reference(D, causal, pair, case, device, record_property) -> None:
    """The main stage alone on its own inputs (the FP32 ``LSE_log2`` and ``dPsum`` as stored, ``dQ_accum`` zeroed,
    ``dK`` and ``dV`` filled with NaN): ``dQ_accum`` (the unscaled ``dS K`` in the reduction layout, or in the pair-row
    layout for a 2-CTA main kernel), ``dK`` and ``dV`` meet the tier of the same formula on the same inputs."""
    name = cake_jit.RECORD_OF[(D, causal, pair)][1]

    def draw(seed: int):
        x = make_inputs(case, D, causal=causal, seed=seed, device=device)
        m = main_buffers(x)
        cake_backend.launch_bwd_main(
            name,
            q=x.q,
            k=x.k,
            v=x.v,
            dout=x.dout,
            lse_log2=m.lse_log2,
            dpsum=m.dpsum,
            dq_accum=m.dq_accum,
            dk=m.dk,
            dv=m.dv,
            softmax_scale=x.softmax_scale,
        )
        return judged_stage(
            lambda _, dtype: main_outputs(x, m, dtype, pair=pair),
            x,
            {"dQ_accum": m.dq_accum, "dK": m.dk, "dV": m.dv},
        )

    judge_draws(record_property, draw)


@pytest.mark.parametrize("case", POST_CASES, ids=[case.id for case in POST_CASES])
@pytest.mark.parametrize(("D", "pair"), POST_VARIANTS, ids=POST_IDS)
def test_the_postprocess_matches_the_reference(D, pair, case, device, record_property) -> None:
    """``dq = bf16(dQ_accum * softmax_scale)``, written in the input layout ``[B, Sq, H, D]`` over a NaN-filled
    buffer, meets the tier; the pair-row build reads ``dQ_accum`` in the 2-CTA main kernel's layout."""
    name = cake_jit.RECORD_OF[(D, False, pair)][2]

    def draw(seed: int):
        t = post_inputs(case, D, seed=seed, device=device)
        cake_backend.launch_bwd_post(name, dq_accum=t.dq_accum, dq=t.dq, softmax_scale=t.softmax_scale)
        return judged_stage(lambda u, dtype: post_outputs(u, dtype, pair=pair), t, {"dq": t.dq})

    judge_draws(record_property, draw)


def _launch_fused(name: str, x) -> dict[str, torch.Tensor]:
    """``launch_bwd_fused`` of the record ``name`` on the operands ``x``, over ``dq`` ``[B, Sq, H, D]``, ``dk`` and
    ``dv`` ``[B, Skv, H, D]`` (BF16 on the operands' device) filled with NaN, which it returns by name."""
    nan = {"device": x.q.device, "dtype": torch.bfloat16}
    dq = torch.full(tuple(x.q.shape), math.nan, **nan)
    dk = torch.full(tuple(x.k.shape), math.nan, **nan)
    dv = torch.full(tuple(x.k.shape), math.nan, **nan)
    cake_backend.launch_bwd_fused(
        name, q=x.q, k=x.k, v=x.v, out=x.out, lse=x.lse, dout=x.dout, dq=dq, dk=dk, dv=dv, softmax_scale=x.softmax_scale
    )
    return {"dq": dq, "dk": dk, "dv": dv}


@pytest.mark.parametrize(("D", "case"), FUSED_PARAMS)
def test_the_fused_stage_matches_the_reference(D, case, device, record_property) -> None:
    """The one-launch stage alone (``launch_bwd_fused`` on the record ``FUSED_OF`` names for the head dimension, no
    mask), from the forward's ``out`` and ``lse``, ``dq``, ``dk`` and ``dv`` filled with NaN: ``dq``, ``dk`` and ``dv``
    meet the tier of the same formula on the same inputs with the row statistics the kernel makes itself
    (``fused_outputs``), on the kernel's own cases."""
    name = cake_jit.FUSED_OF[(D, False)]

    def draw(seed: int):
        x = make_inputs(case, D, causal=False, seed=seed, device=device)
        return judged_stage(lambda _, dtype: fused_outputs(x, dtype), x, _launch_fused(name, x))

    judge_draws(record_property, draw)


@pytest.mark.parametrize(("D", "case"), FUSED_PARAMS)
def test_the_fused_stage_is_run_to_run_bitwise(D, case, device) -> None:
    """The one-launch stage writes ``dq``, ``dk`` and ``dv`` each in one fixed order: three launches on the same
    inputs, each over NaN-filled outputs, give the same bits, every element finite."""
    name = cake_jit.FUSED_OF[(D, False)]
    x = make_inputs(case, D, causal=False, seed=5, device=device)
    runs = [_launch_fused(name, x) for _ in range(3)]
    torch.cuda.synchronize(device)
    assert_backward_contract("fused", x, runs[0])
    for i, run in enumerate(runs[1:], start=2):
        for label, value in run.items():
            assert torch.equal(value.view(torch.int16), runs[0][label].view(torch.int16)), f"run {i}: {label} bits"


# ----------------------------------------------------------------------------------------------------------------
# The backward: one launch, or the three stages composed


def _count_fused_launches(monkeypatch) -> list:
    """Wrap ``launch_bwd_fused`` so that every call appends its record name to the returned list, then launches."""
    launches = []
    launch = cake_backend.launch_bwd_fused

    def counted(name, **kwargs):
        launches.append(name)
        launch(name, **kwargs)

    monkeypatch.setattr(cake_backend, "launch_bwd_fused", counted)
    return launches


@pytest.mark.parametrize(("D", "causal", "pair", "case", "fused"), BACKWARD_PARAMS)
def test_the_backward_matches_the_reference(D, causal, pair, case, fused, device, monkeypatch, record_property) -> None:
    """``dq``, ``dk`` and ``dv`` of the backward hold the contract (BF16, contiguous, the inputs' layouts, finite) and
    meet the tier of the backward formula on the same inputs (``dPsum`` from ``out`` and ``dout``), on the cases of
    each variant (``pair``: the pair route's cases, whose key lengths are multiples of 256), each call on that
    variant's route (``_pin_route``: the pair route's cases with every call in a pair region, the 1-CTA cases with
    none) with the one-launch route turned off, and on the one-launch stage's cases unpatched; a pair case with more
    keys than queries takes the 1-CTA route.  The call takes the route ``fused_record`` and ``variant`` pick for it: it
    launches the one-launch record once exactly when ``fused_record`` gives it."""
    _pin_route(monkeypatch, pair)
    arch = cake_backend.device_arch(device.index)
    found = cake_jit.variant(D, causal, batch=case.B, heads=case.H, seqlen_q=case.S, seqlen_kv=case.kv, arch=arch)
    assert found == (D, causal, pair and case.S >= case.kv), (arch, found)
    if not fused:
        monkeypatch.setattr(cake_jit, "fused_record", lambda *args: None)
    record = cake_jit.fused_record(arch, case.B, case.H, case.S, case.kv, D, causal)
    launches = _count_fused_launches(monkeypatch)

    def draw(seed: int):
        x = make_inputs(case, D, causal=causal, seed=seed, device=device)
        launches.clear()
        dq, dk, dv = cake_backend.cake_fmha_nondet_train_backward(
            x.q, x.k, x.v, x.out, x.lse, x.dout, softmax_scale=x.softmax_scale, causal=causal
        )
        assert launches == ([] if record is None else [record]), (record, launches)
        return judged_backward(x, {"backward": {"dq": dq, "dk": dk, "dv": dv}})

    judge_draws(record_property, draw)


# The backward's calls with fewer queries than keys on the shipped tables (no pin): the least call within the bounds of
# ``PAIR_FEWER_QUERIES`` (``B * H`` 8, 4096 queries, 8192 keys), on the pair route where the architecture has the bounds
# and on the 1-CTA route where it has none; outside every one-launch piece.
FEWER_QUERIES_CASES = (Case(1, 8, 4096, 8192),)


@pytest.mark.parametrize("case", FEWER_QUERIES_CASES, ids=[f"d128-{case.id}" for case in FEWER_QUERIES_CASES])
def test_the_backward_with_fewer_queries_matches_the_reference(case, device, monkeypatch, record_property) -> None:
    """A head-dimension 128 call without the mask with fewer queries than keys, on the shipped tables: ``variant``
    gives it the pair route exactly when the call is within the bounds ``SHIPPED_PAIR_FEWER_QUERIES`` of the device's
    architecture, ``fused_record`` gives it no one-launch record, and ``dq``, ``dk`` and ``dv`` of the backward hold the
    contract and meet the tier of the backward formula on the same inputs."""
    arch = cake_backend.device_arch(device.index)
    bounds = SHIPPED_PAIR_FEWER_QUERIES[arch]
    on = case.kv % (2 * cake_jit.BLOCK) == 0 and _inside_fewer_queries(bounds, case.B * case.H, case.S, case.kv)
    assert case.S < case.kv and on == (bounds is not None), (arch, case)
    found = cake_jit.variant(128, False, batch=case.B, heads=case.H, seqlen_q=case.S, seqlen_kv=case.kv, arch=arch)
    assert found == (128, False, on), (arch, found)
    assert cake_jit.fused_record(arch, case.B, case.H, case.S, case.kv, 128, False) is None, arch
    launches = _count_fused_launches(monkeypatch)

    def draw(seed: int):
        x = make_inputs(case, 128, causal=False, seed=seed, device=device)
        launches.clear()
        dq, dk, dv = cake_backend.cake_fmha_nondet_train_backward(
            x.q, x.k, x.v, x.out, x.lse, x.dout, softmax_scale=x.softmax_scale, causal=False
        )
        assert launches == [], launches
        return judged_backward(x, {"backward": {"dq": dq, "dk": dk, "dv": dv}})

    judge_draws(record_property, draw)


# The backward's causal calls on the shipped tables (no pin): the least call of the long-sequence causal region of
# ``sm_103a`` (``B * H`` 32, 8192 queries and keys), on the route ``DECIDED_LEAST`` names for the architecture (the pair
# route on SM103 and SM107, the 1-CTA route on SM100); outside every one-launch piece.
LONG_CAUSAL_CASES = (Case(1, 32, 8192),)


@pytest.mark.parametrize("case", LONG_CAUSAL_CASES, ids=[f"d128_causal-{case.id}" for case in LONG_CAUSAL_CASES])
def test_the_long_causal_backward_matches_the_reference(case, device, monkeypatch, record_property) -> None:
    """A head-dimension 128 call with the mask on the shipped tables: ``variant`` gives it the route ``DECIDED_LEAST``
    names for the device's architecture, ``fused_record`` gives it no one-launch record, and ``dq``, ``dk`` and ``dv``
    of the backward hold the contract and meet the tier of the backward formula on the same inputs."""
    arch = cake_backend.device_arch(device.index)
    least = DECIDED_LEAST[arch][True][case.B * case.H]
    on = least is not None and case.kv >= least
    assert case.S == case.kv and on == (arch != "sm_100a"), (arch, case)
    found = cake_jit.variant(128, True, batch=case.B, heads=case.H, seqlen_q=case.S, seqlen_kv=case.kv, arch=arch)
    assert found == (128, True, on), (arch, found)
    assert cake_jit.fused_record(arch, case.B, case.H, case.S, case.kv, 128, True) is None, arch
    launches = _count_fused_launches(monkeypatch)

    def draw(seed: int):
        x = make_inputs(case, 128, causal=True, seed=seed, device=device)
        launches.clear()
        dq, dk, dv = cake_backend.cake_fmha_nondet_train_backward(
            x.q, x.k, x.v, x.out, x.lse, x.dout, softmax_scale=x.softmax_scale, causal=True
        )
        assert launches == [], launches
        return judged_backward(x, {"backward": {"dq": dq, "dk": dk, "dv": dv}})

    judge_draws(record_property, draw)


def _kernels(call, trace: Path) -> tuple[object, list[str]]:
    """``(result, names)``: ``call()``'s result and the names of the CUDA kernels it ran, in launch order, observed on
    the device (``torch.profiler``; the trace is written to ``trace``)."""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=activities) as prof:
        result = call()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    kernels = sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"])
    return result, [event["name"] for event in kernels]


def _launched(call, device: torch.device, trace: Path) -> tuple[object, list[tuple[str, str]] | None]:
    """``(result, launched)``: ``call()``'s result and the CUDA kernels it ran, in launch order, observed on the device
    (``torch.profiler``; the trace is written to ``trace``), each as ``(stage, record)`` of the registry record whose
    kernel it is, or ``("no record", name)``.

    Positive control first: a plain torch operation on ``device``, profiled the same way.  When the profiler records
    no kernel of it (a CUPTI that does not support the device), ``call()`` runs without the profiler and ``launched``
    is ``None``: the test runs its other checks, then :func:`_assert_launched` skips the observation."""
    _, control = _kernels(lambda: torch.ones(1024, device=device).mul_(2), trace.with_name("control-" + trace.name))
    if not control:
        return call(), None
    record_of = {rec["kernel_symbol"]: (rec["stage"], name) for name, rec in cake_jit.MODULES.items()}
    result, names = _kernels(call, trace)
    return result, [record_of.get(name, ("no record", name)) for name in names]


def _assert_launched(
    launched: list[tuple[str, str]] | None, variant: tuple[int, bool, bool], record: str | None = None
) -> None:
    """The kernels observed are those of the route of the call, in order, and no other: the one-launch ``record``
    alone when ``fused_record`` gave one, else the three records of ``variant`` (a ``RECORD_OF`` key).  Called after
    every other check of the test: when the profiler observed nothing (``None``), only this observation skips."""
    if launched is None:
        pytest.skip("torch.profiler records no CUDA kernel on this device (CUPTI); install a CUPTI that supports it")
    if record is not None:
        assert launched == [("bwd_fused", record)]
        return
    assert launched == list(zip(CLOSURE_STAGES, cake_jit.RECORD_OF[variant], strict=True))


def _route(D: int, causal: bool, case: Case, fused: bool, device: torch.device, monkeypatch):
    """``(variant, record)`` of a call: its ``RECORD_OF`` key and the one-launch record ``fused_record`` gives it.  With
    ``fused`` False the one-launch route is turned off first (``fused_record`` patched to give None)."""
    if not fused:
        monkeypatch.setattr(cake_jit, "fused_record", lambda *args: None)
    arch = cake_backend.device_arch(device.index)
    record = cake_jit.fused_record(arch, case.B, case.H, case.S, case.kv, D, causal)
    variant = cake_jit.variant(D, causal, batch=case.B, heads=case.H, seqlen_q=case.S, seqlen_kv=case.kv, arch=arch)
    return variant, record


@pytest.mark.parametrize(("D", "causal", "case", "pin", "fused"), ROUTE_PARAMS + SHIPPED_ROUTE_PARAMS)
def test_the_backward_launches_the_records_of_its_variant(
    D, causal, case, pin, fused, device, monkeypatch, tmp_path
) -> None:
    """A backward call launches the records of its route and no other kernel: the one-launch record ``fused_record``
    gives it, alone and without a host sequence, or else, in order, the preprocess, main-stage and postprocess records
    that ``RECORD_OF`` names for its variant (``variant(D, causal, *, batch, heads, seqlen_q, seqlen_kv, arch)``: at
    head dimension 128 the pair route for 256 keys in a pinned region and the 1-CTA route without one, the 1-CTA route
    for 384 keys, the shipped regions' route for 8192 queries and keys at ``B * H`` 4 and, with the mask, for 8192 and
    7936 queries and keys at ``B * H`` 32, and the 1-CTA route for 1024 queries and keys at ``B * H`` 16 on every
    architecture); the kernels one call runs, observed on the device, are those records' kernels; it writes none of
    its inputs."""
    _pin_route(monkeypatch, pin)
    variant, record = _route(D, causal, case, fused, device, monkeypatch)
    arch = cake_backend.device_arch(device.index)
    if pin is None:
        least = DECIDED_LEAST[arch][causal][case.B * case.H]
        assert variant[2] == (least is not None and case.kv >= least), (arch, variant)
        if case.kv == 1024 and case.B * case.H == 16:  # d128-S1024: the 1-CTA route on every architecture
            assert not variant[2], (arch, variant)
    else:
        on = pin and D == 128 and case.kv % (2 * cake_jit.BLOCK) == 0 and case.S >= case.kv
        assert variant[2] == on, (arch, variant)
    sequences = []
    if fused:
        sequence = cake_backend._sequence
        monkeypatch.setattr(cake_backend, "_sequence", lambda *key: sequences.append(key) or sequence(*key))
    x = make_inputs(case, D, causal=causal, seed=11, device=device)
    inputs = (x.q, x.k, x.v, x.out, x.lse, x.dout)
    before = [t.clone() for t in inputs]

    def call():
        return cake_backend.cake_fmha_nondet_train_backward(*inputs, softmax_scale=x.softmax_scale, causal=causal)

    (dq, dk, dv), launched = _launched(call, device, tmp_path / "trace.json")
    assert_backward_contract("routed", x, {"dq": dq, "dk": dk, "dv": dv})
    for name, old, new in zip(("q", "k", "v", "out", "lse", "dout"), before, inputs, strict=True):
        assert torch.equal(old, new), f"the backward wrote its input {name}"
    if record is not None:
        assert sequences == [], "the one-launch route looked up a host sequence"
    _assert_launched(launched, variant, record)


# The backward's paths besides the host sequence's ``run``: ``run`` returns False (no tvm-ffi environment allocator, or
# a failed allocation: the workspace from Python, then ``run_with_workspace``), and no host sequence for the variant
# (the stage launchers).
FALLBACKS = ("run_with_workspace", "stage_launchers")


@pytest.mark.parametrize("path", FALLBACKS)
@pytest.mark.parametrize(("D", "causal", "case", "pin", "fused"), ROUTE_PARAMS)
def test_the_backward_without_run_launches_the_same_records(
    D, causal, case, pin, fused, path, device, monkeypatch, tmp_path, record_property
) -> None:
    """When ``run`` returns ``False`` and when the variant has no host sequence, the backward launches the records of
    its route in order, and its ``dq``, ``dk`` and ``dv`` meet the tier (on both routes of head dimension 128): the
    variant's three records, or the one-launch record ``fused_record`` gives, which looks up no host sequence."""
    _pin_route(monkeypatch, pin)
    variant, record = _route(D, causal, case, fused, device, monkeypatch)
    arch = cake_backend.device_arch(device.index)
    on = pin and D == 128 and case.kv % (2 * cake_jit.BLOCK) == 0 and case.S >= case.kv
    assert variant[2] == on, (arch, variant)
    found = cake_backend._sequence(variant, arch)
    assert found is not None, "every variant ships a host sequence"
    library, run, run_with_workspace = found
    x = make_inputs(case, D, causal=causal, seed=11, device=device)
    inputs = (x.q, x.k, x.v, x.out, x.lse, x.dout)
    calls = []
    sequences = []

    def without_allocator(*args):
        calls.append(len(args))
        return False

    replacement = None if path == "stage_launchers" else (library, without_allocator, run_with_workspace)
    monkeypatch.setattr(cake_backend, "_sequence", lambda *key: sequences.append(key) or replacement)

    def call():
        return cake_backend.cake_fmha_nondet_train_backward(*inputs, softmax_scale=x.softmax_scale, causal=causal)

    (dq_path, dk_path, dv_path), launched = _launched(call, device, tmp_path / "trace.json")
    if record is None:
        assert calls == ([] if replacement is None else [11])
    else:
        assert calls == [] and sequences == [], "the one-launch route looked up a host sequence"
    assert_backward_contract(path, x, {"dq": dq_path, "dk": dk_path, "dv": dv_path})

    def draw(seed: int):
        y = make_inputs(case, D, causal=causal, seed=seed, device=device)
        got = cake_backend.cake_fmha_nondet_train_backward(
            y.q, y.k, y.v, y.out, y.lse, y.dout, softmax_scale=y.softmax_scale, causal=causal
        )
        return judged_backward(y, {path: dict(zip(("dq", "dk", "dv"), got, strict=True))})

    judge_draws(record_property, draw)
    _assert_launched(launched, variant, record)
