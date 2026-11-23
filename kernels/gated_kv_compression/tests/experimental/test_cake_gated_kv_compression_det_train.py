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

Backend suite of the Cake deterministic (``det``) ratio-2 gated KV compression package
(``cake_pplx._backends.cake_gated_kv_compression_det_train``, the backend of
``cake_pplx.cake_gated_kv_compression_det_train``): width 512, packed documents bounded by a device ``cu_seqlens``, on
SM100, SM103 and SM107.

Without a device: the registry (the two pair-table roles, ``pair_table_small`` for at most 512 documents and
``pair_table`` for more, the two forward roles, ``fwd`` for at most 256 documents and ``fwd_unfused`` for more, and the
two backward roles, each served on each of the three architectures by exactly one record and delivered for no other:
the forward by ``fwd`` on sm_100a / sm_103a and ``fwd_sm_107a`` on sm_107a, each building the pair table of up to 256
documents, and by ``fwd_unfused`` / ``fwd_unfused_sm_103a`` / ``fwd_unfused_sm_107a`` (the warp row search, a record an
architecture), the backward by ``bwd`` /
``bwd_sm_107a`` (two-level search), ``bwd_small`` / ``bwd_small_sm_107a`` (one-level search) and ``bwd_warp`` /
``bwd_warp_sm_103a`` / ``bwd_warp_sm_107a`` (the warp row search, one program); the launch facts,
the kernel parameters each record takes), the record each role launches on each architecture, the pair-table role of
512 and 513 documents, the forward role of 256, 257, 512, 513 and more documents, the backward role on either side of
each architecture's bounds of documents, the launch grids (every row covered, the pair tables on one CTA, the unfused
forward's grid capped at its CTAs a multiprocessor in its range of documents, as the binding entry receives it), the
launchers' keywords and the kernel arguments they pass, the exact overlap rule of two row-strided views
(the two column halves of one buffer share no element), the operand checks' fast path (for every well-formed and
malformed operand variant it gives the full checks' result or exception type and message, or defers to them; an output
a call allocates takes no overlap check), the memory report, the build manifest (every record's
library, the recorded source digests), the standalone loader, the prebuilt libraries served first, and no environment
read outside the JIT shim.

On SM100 / SM103 / SM107: the pair table equal to the exclusive prefix of the pair counts (an integer table: exactly;
both pair-table programs, on either side of 512 documents);
each kernel judged on its own against the naive BF16 tier (``tests/cake_naive.py``: the error against an FP64 oracle
of the same inputs over 32 independent draws, in the L2 and the max norm, median and largest, beside the error of the
same math in BF16), every output buffer filled with NaN before the launch so every element must be written: the
forward (``out``) and the backward (``dC``, ``dZ``, the tail rows exactly +0.0) on contiguous ``C`` / ``Z`` and on the
column halves of one ``[T, 1024]`` tensor (the gradients written into the halves of one ``[T, 1024]`` buffer), on
small packs with empty and one-token documents, close pairs, gate extremes, many small odd documents and the two
performance packs; the forward rows past the pair count stored as +0.0, and a forward ``G`` below the pair count
giving the first ``G`` rows.  Then the documented invariances, bit for bit: repeated calls, the two layouts, one call
with four different row strides (judged too), a ``G`` past the pair count in the explicit and autograd backward, each
document alone against the pack, the tail tokens' values, and the records each call launches (no pair table before a
forward of at most 256 documents, which runs ``fwd``; ``fwd_unfused`` after the pair table for more; the small pair
table for at most 512 and the default one for 513).

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import random
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_gated_kv_compression_naive import (
    PERF_PACKS,
    WIDTH,
    Pack,
    assert_backward_contract,
    assert_bitwise,
    assert_forward_contract,
    assert_zero_rows,
    bounds,
    judge_draws,
    judged,
    make_inputs,
    nan_halves,
    nan_like_rows,
    pair_table_of,
    tail_rows,
)
from tests.cake.test_helpers.cake_package import cake_package

PACKAGE_KEY = "cake.cake_gated_kv_compression_det_train"
pkg = cake_package(PACKAGE_KEY)
cake_backend = cake_package(PACKAGE_KEY, "cake_backend")
cake_jit = cake_package(PACKAGE_KEY, "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ROLES = ("pair_table", "pair_table_small", "fwd", "fwd_unfused", "bwd", "bwd_small", "bwd_warp")
# The architectures of a role not served on all three: none, every role is served on the three.
ROLE_ARCHES: dict[str, tuple[str, ...]] = {}
# Record -> (its role, the architectures it serves, the lanes of a 512-wide row): both forwards ship 128-bit lanes
# (4 FP32 channels) on sm_100a / sm_103a and 256-bit lanes (8 channels) on sm_107a; the backward 256-bit lanes, with
# the two-level search in ``bwd``, the one-level search in ``bwd_small`` and the warp search in ``bwd_warp``.
RECORDS = {
    "pair_table": ("pair_table", ARCHES, None),
    "pair_table_small": ("pair_table_small", ARCHES, None),
    "fwd": ("fwd", ("sm_100a", "sm_103a"), 128),
    "fwd_sm_107a": ("fwd", ("sm_107a",), 64),
    "fwd_unfused": ("fwd_unfused", ("sm_100a",), 128),
    "fwd_unfused_sm_103a": ("fwd_unfused", ("sm_103a",), 128),
    "fwd_unfused_sm_107a": ("fwd_unfused", ("sm_107a",), 64),
    "bwd": ("bwd", ("sm_100a", "sm_103a"), 64),
    "bwd_sm_107a": ("bwd", ("sm_107a",), 64),
    "bwd_small": ("bwd_small", ("sm_100a", "sm_103a"), 64),
    "bwd_small_sm_107a": ("bwd_small", ("sm_107a",), 64),
    "bwd_warp": ("bwd_warp", ("sm_100a",), 64),
    "bwd_warp_sm_103a": ("bwd_warp", ("sm_103a",), 64),
    "bwd_warp_sm_107a": ("bwd_warp", ("sm_107a",), 64),
}
# The most documents a pair-table launch runs ``pair_table_small`` for (the rendered ``small_docs``).
SMALL_DOCS = 512
# The most documents a forward launch runs ``fwd`` for, which builds the pair table itself (the rendered ``fuse_docs``);
# a forward of more runs ``fwd_unfused``.
FUSE_DOCS = 256
# The most documents a backward launch runs ``bwd_small`` for below ``bwd`` (the rendered ``bwd_small_docs`` of the
# record serving the role); a backward of more runs ``bwd``, up to ``BWD_MAX_DOCS``.
BWD_SMALL_DOCS = {"sm_100a": 26, "sm_103a": 26, "sm_107a": 125}
# The most documents a backward launch runs ``bwd`` for (the rendered ``bwd_max_docs`` of the record serving the role;
# ``None``: the record has no such leaf, no bound); a backward of more runs ``bwd_small``.
BWD_MAX_DOCS = {"sm_100a": None, "sm_103a": None, "sm_107a": 16384}
# The most documents a backward launch runs ``bwd`` or ``bwd_small`` for outside ``BWD_WARP_RANGES`` (the rendered
# ``bwd_warp_docs`` of the record serving ``bwd_warp``); a backward of more runs ``bwd_warp``.
BWD_WARP_DOCS = dict.fromkeys(ARCHES, 16384)
# The documents below that a backward launch also runs ``bwd_warp`` for (the rendered ``bwd_warp_min_docs`` and
# ``bwd_warp_max_docs`` of the record serving ``bwd_warp``; ``None``: the record has no such leaves).
BWD_WARP_RANGES = {"sm_100a": None, "sm_103a": (15, 32), "sm_107a": None}
# What :func:`cake_jit.bwd_warp_region` reads on each architecture: ``(lo, hi, above)``.
BWD_WARP_REGIONS = {arch: (*(BWD_WARP_RANGES[arch] or (None, None)), BWD_WARP_DOCS[arch]) for arch in ARCHES}
# Record -> its capped grid (the rendered ``capped_min_docs``, ``capped_max_docs`` and ``ctas_per_sm``, what
# :func:`cake_jit.capped_leaves` reads): a forward of 2048 to 1048577 documents on sm_100a runs the unfused forward on
# at most 6 CTAs a multiprocessor; no other record has one (sm_103a and sm_107a keep the rows grid).
CAPPED = {"fwd_unfused": (2048, 1048577, 6)}
# Multiprocessors of a B200 (sm_100a), a GB300 (sm_103a) and a VR200 (sm_107a), as the capped-grid tests stub them.
SMS = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}
# Backward packs on either side of the documents bounds and of the forward's fused bound (``FUSE_DOCS``, 256).
BWD_BOUND_DOCS = (1, 14, 15, 16, 26, 27, 32, 33, 40, 41, 124, 125, 126, 256, 257, 1024, 1025, 16384, 16385)
# The backward's role at each of these documents, the same on every architecture, a letter a document count: ``s``
# ``bwd_small``, ``b`` ``bwd``, ``w`` ``bwd_warp``.  The role does not depend on the tokens: each count is checked at
# every token count of ``ROUTING_TOKENS``.
ROUTING_TOKENS = (16384, 32768, 65536)
ROUTING_DOCS = (14, 15, 26, 27, 32, 33, 40, 41, 124, 125, 126, 1024, 1025, 16383, 16384, 16385, 32769)
ROUTING = {
    "sm_100a": "sssbbbbbbbbbbbbww",
    "sm_103a": "swwwwbbbbbbbbbbww",
    "sm_107a": "ssssssssssbbbbbww",
}
ROUTING_ROLES = {"s": "bwd_small", "b": "bwd", "w": "bwd_warp"}
# The record serving each backward role on each architecture.
BACKWARD_RECORDS = {
    "sm_100a": {"bwd": "bwd", "bwd_small": "bwd_small", "bwd_warp": "bwd_warp"},
    "sm_103a": {"bwd": "bwd", "bwd_small": "bwd_small", "bwd_warp": "bwd_warp_sm_103a"},
    "sm_107a": {"bwd": "bwd_sm_107a", "bwd_small": "bwd_small_sm_107a", "bwd_warp": "bwd_warp_sm_107a"},
}
# The backward programs: the two-level search (``bwd``, ``bwd_sm_107a``) and the one-level search of 0.1.0
# (``bwd_small``, ``bwd_small_sm_107a``), each one program on disjoint architectures, and the warp search (``bwd_warp``,
# ``bwd_warp_sm_103a``, ``bwd_warp_sm_107a``), one program on the three.
TWO_LEVEL_LIBRARY = "cake_gated_kv_compression_det_train_857e5499b5edd675ffd3"
ONE_LEVEL_LIBRARY = "cake_gated_kv_compression_det_train_0a0e2df45d6eb8316daf"
WARP_LIBRARY = "cake_gated_kv_compression_det_train_aa24263114554decff8e"
BACKWARD_LIBRARIES = {
    "bwd": TWO_LEVEL_LIBRARY,
    "bwd_sm_107a": TWO_LEVEL_LIBRARY,
    "bwd_small": ONE_LEVEL_LIBRARY,
    "bwd_small_sm_107a": ONE_LEVEL_LIBRARY,
    "bwd_warp": WARP_LIBRARY,
    "bwd_warp_sm_103a": WARP_LIBRARY,
    "bwd_warp_sm_107a": WARP_LIBRARY,
}


def _bwd_role(arch: str, docs: int) -> str:
    """The backward role of ``docs`` documents on ``arch``: ``bwd_warp`` for ``docs > BWD_WARP_DOCS`` or ``docs`` in
    ``BWD_WARP_RANGES``, else ``bwd`` for ``BWD_SMALL_DOCS < docs <= BWD_MAX_DOCS``, ``bwd_small`` for the others."""
    span = BWD_WARP_RANGES[arch]
    if docs > BWD_WARP_DOCS[arch] or (span is not None and span[0] <= docs <= span[1]):
        return "bwd_warp"
    high = BWD_MAX_DOCS[arch]
    return "bwd" if BWD_SMALL_DOCS[arch] < docs and (high is None or docs <= high) else "bwd_small"


# The forward programs with the warp row search, which ``fwd_unfused`` ships: one program in the records of sm_100a and
# sm_103a, another on sm_107a.
UNFUSED_LIBRARIES = {
    "fwd_unfused": "cake_gated_kv_compression_det_train_2ac79e6766ca09bf2903",
    "fwd_unfused_sm_103a": "cake_gated_kv_compression_det_train_2ac79e6766ca09bf2903",
    "fwd_unfused_sm_107a": "cake_gated_kv_compression_det_train_64aae35cfb0e308d103a",
}
# The record serving ``fwd`` and ``fwd_unfused`` on each architecture.
FORWARD_RECORDS = {"sm_100a": "fwd", "sm_103a": "fwd", "sm_107a": "fwd_sm_107a"}
UNFUSED_RECORDS = {"sm_100a": "fwd_unfused", "sm_103a": "fwd_unfused_sm_103a", "sm_107a": "fwd_unfused_sm_107a"}
# The records of each grid kind.
ROW_RECORDS = tuple(name for name in RECORDS if not name.startswith("pair_table"))
# The kernel parameters of each role (the argument plan without the grid), in any order: the host passes them by name.
PARAMETERS = {
    "pair_table": {"cu_seqlens", "cu_pairs", "num_docs"},
    "pair_table_small": {"cu_seqlens", "cu_pairs", "num_docs"},
    "fwd": {"c", "z", "cu_seqlens", "cu_pairs", "out", "num_docs", "G", "c_ld", "z_ld"},
    "fwd_unfused": {"c", "z", "cu_seqlens", "cu_pairs", "out", "num_docs", "G", "c_ld", "z_ld"},
    "bwd": {
        "dout",
        "c",
        "z",
        "cu_seqlens",
        "cu_pairs",
        "dc",
        "dz",
        "num_docs",
        "G",
        "c_ld",
        "z_ld",
        "dc_ld",
        "dz_ld",
    },
}
PARAMETERS["bwd_small"] = PARAMETERS["bwd_warp"] = PARAMETERS["bwd"]
LAUNCHER_KEYWORDS = {
    "launch_pair_table": ("cu_seqlens", "cu_pairs"),
    "launch_fwd": ("c", "z", "cu_seqlens", "cu_pairs", "out", "c_ld", "z_ld"),
    "launch_bwd": ("dout", "c", "z", "cu_seqlens", "cu_pairs", "dc", "dz", "c_ld", "z_ld", "dc_ld", "dz_ld"),
}
# What a public call has already looked up and passes to its launches (each defaults to None: the launcher looks it up).
LAUNCHER_OPTIONS = {
    "launch_pair_table": ("device_index", "arch"),
    "launch_fwd": ("role", "device_index", "arch"),
    "launch_bwd": ("role", "device_index", "arch"),
}
LAYOUTS = ("contiguous", "halves")

# Packs with at least one pair: one document, empty and one-token documents, odd tails, the reference's mixed pack, many
# small odd documents (lengths 1, 3, 5, 7 repeated), and the two performance packs.
PAIR_PACKS = (
    Pack((2,)),
    Pack((0, 1, 2, 3)),
    Pack((0, 1, 3, 2, 7, 4)),
    Pack((127, 256, 1, 641)),
    Pack((1, 3, 5, 7) * 75, "small-odd-S300"),
    *PERF_PACKS,
)
# Input variants judged on the reference's mixed pack: close pairs and gate extremes.
VARIANTS = (
    pytest.param({"close": 12}, id="close-2^-12"),
    pytest.param({"close": 16}, id="close-2^-16"),
    pytest.param({"gates": True}, id="gate-extremes"),
)
MIXED = Pack((127, 256, 1, 641))
# Pair-table packs: one document, empty documents only, odd and even lengths, and more documents than one tile; the
# packs of 512 and 513 documents run the two pair-table programs on either side of ``small_docs``.
TABLE_PACKS = (
    Pack((0,)),
    Pack((2,)),
    Pack((0, 0, 0)),
    Pack((0, 1, 3, 2, 7, 4)),
    Pack(tuple((i * 7919) % 13 for i in range(SMALL_DOCS)), "S512"),
    Pack(tuple((i * 7919) % 13 for i in range(SMALL_DOCS + 1)), "S513"),
    Pack(tuple((i * 7919) % 13 for i in range(5000)), "S5000"),
    *PERF_PACKS,
)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the package ships {ARCHES}; this device is {capability}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in cake_jit.specs_for_arch(cake_jit.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return json.loads(path.read_text())


def _lds(x) -> dict:
    return {"c_ld": x.c.stride(0), "z_ld": x.z.stride(0)}


def _grad_buffers(x, layout: str) -> tuple[torch.Tensor, torch.Tensor]:
    """NaN ``dc`` and ``dz``: two tensors, or the two halves of one ``[T, 1024]`` buffer."""
    if layout == "halves":
        _, dc, dz = nan_halves(x.T, x.c.device)
        return dc, dz
    return nan_like_rows(x.T, torch.float32, x.c.device), nan_like_rows(x.T, torch.float32, x.c.device)


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.ROLES) == ROLES
    assert dict(cake_jit.ROLE_ARCHES) == ROLE_ARCHES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert cake_jit.WIDTH == WIDTH and cake_jit.CZ_WIDTH == 2 * WIDTH
    assert cake_jit.LIBRARY_PREFIX == "cake_gated_kv_compression_det_train"
    assert list(cake_jit.MODULES) == list(RECORDS)
    rendered = (PACKAGE_DIR / "build.json").is_file()
    libraries = set()
    for name, rec in cake_jit.MODULES.items():
        role, arches, lanes = RECORDS[name]
        assert rec["role"] == role, name
        assert tuple(rec["arches"]) == arches, (name, rec["arches"])
        if lanes is not None:
            assert rec["grid_rule"]["lanes_per_row"] == lanes, name
        assert rec["ffi_entry"] == "run", name
        assert rec["grid_rule"]["kind"] in cake_jit.GRID_KINDS, name
        launch = rec["launch"]
        assert launch["cooperative"] is False, name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert int(launch["block"][1]) == int(launch["block"][2]) == 1, name
        if rendered:
            assert re.fullmatch(r"cake_gated_kv_compression_det_train_[0-9a-f]{20}", rec["library"]), rec["library"]
            assert rec["kernel_symbol"] == "kernel_" + rec["library"], name
        libraries.add(rec["library"])
    for name in ("pair_table", "pair_table_small"):
        assert cake_jit.MODULES[name]["grid_rule"]["kind"] == "one_cta", name
    for name in ROW_RECORDS:
        assert cake_jit.MODULES[name]["grid_rule"]["kind"] == ("rows_capped" if name in CAPPED else "rows"), name
    if rendered:
        # one program a record, but the three backward programs, two each the program of two records and the warp
        # search the program of three, and the sm_100a / sm_103a unfused forward of two, on disjoint architectures
        assert len(libraries) == len(cake_jit.MODULES) - 5
        for name, library in BACKWARD_LIBRARIES.items():
            assert cake_jit.MODULES[name]["library"] == library, name


def test_the_backward_records_carry_their_bounds() -> None:
    """The ``bwd_small`` records carry ``bwd_small_docs`` (the rendered bounds: 26 on sm_100a / sm_103a, 125 on
    sm_107a), and :func:`cake_jit.bwd_small_docs` reads the one of the record serving each architecture; the
    ``bwd_sm_107a`` record carries ``bwd_max_docs`` (16384), and :func:`cake_jit.bwd_max_docs` reads it, ``None`` (no
    bound) on sm_100a / sm_103a, whose ``bwd`` record has no such leaf; the ``bwd_warp`` records carry
    ``bwd_warp_docs`` (16384), ``bwd_warp_sm_103a`` also ``bwd_warp_min_docs`` and
    ``bwd_warp_max_docs`` (15 and 32), and :func:`cake_jit.bwd_warp_region` reads them on the three
    architectures.  No other record carries these leaves.  The records of each backward program have its library,
    kernel symbol and launch, and every backward record has the arguments of ``bwd`` and rows of 64 lanes."""
    for name, rec in cake_jit.MODULES.items():
        docs = BWD_SMALL_DOCS[rec["arches"][0]] if rec["role"] == "bwd_small" else None
        assert rec.get("bwd_small_docs") == docs, name
        high = BWD_MAX_DOCS[rec["arches"][0]] if rec["role"] == "bwd" else None
        assert rec.get("bwd_max_docs") == high, name
        warp = BWD_WARP_DOCS[rec["arches"][0]] if rec["role"] == "bwd_warp" else None
        assert rec.get("bwd_warp_docs") == warp, name
        span = BWD_WARP_RANGES[rec["arches"][0]] if rec["role"] == "bwd_warp" else None
        assert (rec.get("bwd_warp_min_docs"), rec.get("bwd_warp_max_docs")) == (span or (None, None)), name
    assert {arch: cake_jit.bwd_small_docs(arch) for arch in ARCHES} == BWD_SMALL_DOCS
    assert {arch: cake_jit.bwd_max_docs(arch) for arch in ARCHES} == BWD_MAX_DOCS
    assert {arch: cake_jit.bwd_warp_region(arch) for arch in ARCHES} == BWD_WARP_REGIONS
    assert "bwd_max_docs" not in cake_jit.MODULES["bwd"]
    # the warp search takes the packs past the other backwards' bounds, and a range of packs at most that bound
    for arch in ARCHES:
        assert BWD_WARP_DOCS[arch] > BWD_SMALL_DOCS[arch] and BWD_WARP_DOCS[arch] >= (BWD_MAX_DOCS[arch] or 0), arch
        lo, hi = BWD_WARP_RANGES[arch] or (1, 1)
        assert 1 <= lo <= hi <= BWD_WARP_DOCS[arch], arch
    two_level = cake_jit.MODULES["bwd"]
    for name, library in BACKWARD_LIBRARIES.items():
        rec = cake_jit.MODULES[name]
        assert rec["arg_plan"] == two_level["arg_plan"], name
        assert rec["grid_rule"] == {"kind": "rows", "lanes_per_row": 64, "rows_per_lane_group": 1}, name
        assert rec["library"] == library, name
    for name in ("bwd_warp", "bwd_warp_sm_103a", "bwd_warp_sm_107a"):
        assert cake_jit.MODULES[name]["launch"]["dynamic_smem_bytes"] == 0, name
    if (PACKAGE_DIR / "build.json").is_file():
        pairs = (
            ("bwd", "bwd_sm_107a"),
            ("bwd_small", "bwd_small_sm_107a"),
            ("bwd_warp", "bwd_warp_sm_103a"),
            ("bwd_warp", "bwd_warp_sm_107a"),
        )
        for first, second in pairs:
            one, other = cake_jit.MODULES[first], cake_jit.MODULES[second]
            assert one["library"] == other["library"] == BACKWARD_LIBRARIES[first], first
            assert one["kernel_symbol"] == other["kernel_symbol"] == f"kernel_{one['library']}", first
            assert one["launch"] == other["launch"], first


def test_the_forward_records_build_the_table_of_up_to_256_documents() -> None:
    """Each ``fwd`` record carries ``fuse_docs`` (the rendered bound: 256), and :func:`cake_jit.fuse_docs` reads the
    one of the ``fwd`` record serving each architecture; the pair-table, ``fwd_unfused`` and backward records carry
    none."""
    for name, rec in cake_jit.MODULES.items():
        assert rec.get("fuse_docs") == (FUSE_DOCS if rec["role"] == "fwd" else None), name
    assert {arch: cake_jit.fuse_docs(arch) for arch in ARCHES} == dict.fromkeys(ARCHES, FUSE_DOCS)


def test_the_unfused_forward_records_are_the_warp_search_programs() -> None:
    """``fwd_unfused`` (sm_100a), ``fwd_unfused_sm_103a`` (sm_103a, the same program) and ``fwd_unfused_sm_107a``
    (sm_107a) are the forward programs with the warp row search, with the lanes and arguments of the ``fwd`` record of
    the same architecture, and no dynamic shared memory (they build no table); their grid rule is the forward's, capped
    (``rows_capped``) for a record of ``CAPPED`` (``fwd_unfused`` on sm_100a)."""
    for name, library in UNFUSED_LIBRARIES.items():
        rec = cake_jit.MODULES[name]
        (arch,) = rec["arches"]
        fused = cake_jit.MODULES[FORWARD_RECORDS[arch]]
        kind = "rows_capped" if name in CAPPED else "rows"
        assert rec["library"] == library and rec["kernel_symbol"] == f"kernel_{library}", name
        assert arch in fused["arches"] and rec["grid_rule"] == {**fused["grid_rule"], "kind": kind}, name
        assert rec["arg_plan"] == fused["arg_plan"], name
        assert rec["launch"]["dynamic_smem_bytes"] == 0, name
        assert rec["library"] != fused["library"], name


def test_the_small_pair_table_serves_at_most_512_documents() -> None:
    """``pair_table_small`` carries ``small_docs`` (the rendered bound: 512), and :func:`cake_jit.small_docs` reads it
    on each architecture; no other record carries one.  It is one CTA of 256 threads, the default pair table one of
    1024 once rendered."""
    for name, rec in cake_jit.MODULES.items():
        assert rec.get("small_docs") == (SMALL_DOCS if name == "pair_table_small" else None), name
    assert {arch: cake_jit.small_docs(arch) for arch in ARCHES} == dict.fromkeys(ARCHES, SMALL_DOCS)
    assert SMALL_DOCS > cake_jit.fuse_docs("sm_100a")
    if (PACKAGE_DIR / "build.json").is_file():
        assert tuple(cake_jit.MODULES["pair_table_small"]["launch"]["block"]) == (256, 1, 1)
        assert tuple(cake_jit.MODULES["pair_table"]["launch"]["block"]) == (1024, 1, 1)


def test_each_role_has_one_record_per_architecture() -> None:
    """Every role is served on every architecture of its ``ROLE_ARCHES`` (all three for the others) by exactly one
    record: the forward by ``fwd`` (128-bit lanes) on sm_100a and sm_103a and by ``fwd_sm_107a`` (256-bit lanes) on
    sm_107a, ``fwd_unfused`` by ``fwd_unfused`` on sm_100a, ``fwd_unfused_sm_103a`` on sm_103a and
    ``fwd_unfused_sm_107a`` on sm_107a, the backward by ``bwd`` (two-level search)
    on sm_100a and sm_103a and by ``bwd_sm_107a`` on sm_107a, the small backward by ``bwd_small`` (one-level search)
    on sm_100a and sm_103a and by ``bwd_small_sm_107a`` on sm_107a, the warp backward by ``bwd_warp`` on sm_100a,
    ``bwd_warp_sm_103a`` on sm_103a and ``bwd_warp_sm_107a`` on sm_107a; no record serves another architecture."""
    for role in ROLES:
        for arch in ARCHES:
            serving = [name for name, rec in cake_jit.MODULES.items() if rec["role"] == role and arch in rec["arches"]]
            if arch in ROLE_ARCHES.get(role, ARCHES):
                assert serving == [cake_jit.role_record(role, arch)], (role, arch, serving)
            else:
                assert serving == [], (role, arch, serving)
                with pytest.raises(KeyError):
                    cake_jit.role_record(role, arch)
    assert [cake_jit.role_record("fwd", arch) for arch in ARCHES] == ["fwd", "fwd", "fwd_sm_107a"]
    assert [cake_jit.role_record("fwd_unfused", arch) for arch in ARCHES] == [
        "fwd_unfused",
        "fwd_unfused_sm_103a",
        "fwd_unfused_sm_107a",
    ]
    assert [cake_jit.role_record("bwd", arch) for arch in ARCHES] == ["bwd", "bwd", "bwd_sm_107a"]
    assert [cake_jit.role_record("bwd_small", arch) for arch in ARCHES] == [
        "bwd_small",
        "bwd_small",
        "bwd_small_sm_107a",
    ]
    assert [cake_jit.role_record("bwd_warp", arch) for arch in ARCHES] == [
        "bwd_warp",
        "bwd_warp_sm_103a",
        "bwd_warp_sm_107a",
    ]
    for arch in ARCHES:
        for role, name in BACKWARD_RECORDS[arch].items():
            assert cake_jit.role_record(role, arch) == name, (role, arch)
    assert {cake_jit.role_record(role, arch) for role in ("pair_table", "pair_table_small") for arch in ARCHES} == {
        "pair_table",
        "pair_table_small",
    }
    for arch in ("sm_110a", "sm_90a", "sm_120a"):
        with pytest.raises(KeyError):
            cake_jit.role_record("fwd", arch)
    with pytest.raises(KeyError):
        cake_jit.role_record("pool", "sm_100a")
    # 256 threads a CTA: two forward rows on sm_100a / sm_103a, four on sm_107a
    assert cake_jit.rows_per_cta("fwd") * 2 == cake_jit.rows_per_cta("fwd_sm_107a")


@pytest.mark.parametrize("name", list(RECORDS))
def test_each_record_takes_its_kernel_parameters(name) -> None:
    """The argument plan of each record names the kernel parameters of its role (the host passes them by name), then
    the grid."""
    plan = cake_jit.MODULES[name]["arg_plan"]
    if not plan:
        pytest.skip("the record's argument plan is not rendered yet")
    found = set(cake_jit.parameters(name)) - {"enable_pdl"}
    logical = cake_jit.MODULES[name].get("logical", {})
    assert {logical.get(key, key) for key in found} == PARAMETERS[RECORDS[name][0]]


def test_no_record_is_delivered_outside_its_architectures() -> None:
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            assert cake_jit.library_name(name, arch) == f"{rec['library']}_{arch}"
        for arch in ("sm_110a", "sm_90a", "sm_120a", *(arch for arch in ARCHES if arch not in rec["arches"])):
            with pytest.raises(RuntimeError):
                cake_jit.library_name(name, arch)


def test_grids_cover_every_row() -> None:
    """The pair table runs on one CTA; the forward covers ``G`` rows and the backward ``G + S`` virtual rows with the
    fewest CTAs of :func:`rows_per_cta` rows each."""
    for name in ("pair_table", "pair_table_small"):
        assert cake_jit.launch_grid(name, 1) == (1, 1, 1)
        assert cake_jit.launch_grid(name, 10**6) == (1, 1, 1)
    rng = random.Random(7091)
    for name in ROW_RECORDS:
        per = cake_jit.rows_per_cta(name)
        assert per >= 1
        for rows in [1, 2, per - 1, per, per + 1, *[rng.randint(1, 1 << 22) for _ in range(200)]]:
            if rows < 1:
                continue
            grid = cake_jit.launch_grid(name, rows)
            assert grid[1:] == (1, 1), (name, rows, grid)
            assert grid[0] * per >= rows > (grid[0] - 1) * per, (name, rows, grid, per)


def test_the_unfused_forward_grid_is_capped_in_its_range() -> None:
    """A ``rows_capped`` record's grid is its ``rows`` grid but for ``lo <= num_docs <= hi`` documents
    (:func:`cake_jit.capped_leaves`), where it is at most ``ctas_per_sm`` CTAs a multiprocessor: at each bound and
    one past it, for a grid above, at and below the cap; the multiprocessor count is needed only when the cap can bind.
    Every other record has no capped grid."""
    assert cake_jit.GRID_KINDS == ("one_cta", "rows", "rows_capped")
    assert {name: cake_jit.capped_leaves(name) for name in RECORDS} == {name: CAPPED.get(name) for name in RECORDS}
    for name, (lo, hi, ctas) in CAPPED.items():
        per = cake_jit.rows_per_cta(name)
        for arch in cake_jit.MODULES[name]["arches"]:
            cap = ctas * SMS[arch]
            for rows in (1, per * ctas, per * ctas + 1, per * cap - 1, per * cap, per * cap + 1, per * cap * 3 + 5):
                full = cake_jit.launch_grid(name, rows)
                assert full == (max(1, -(-rows // per)), 1, 1), (name, rows)
                for docs in (1, lo - 1, lo, hi, hi + 1, cake_jit.MAX_DOCS):
                    inside = lo <= docs <= hi
                    grid = cake_jit.launch_grid(name, rows, docs, SMS[arch])
                    assert grid == ((min(full[0], cap), 1, 1) if inside else full), (name, arch, rows, docs, grid)
                    if not inside or full[0] <= ctas:
                        assert cake_jit.launch_grid(name, rows, docs) == full, (name, rows, docs)
                    else:
                        with pytest.raises(ValueError, match=rf"record {name}: the capped grid of {docs} documents"):
                            cake_jit.launch_grid(name, rows, docs)
    for name in RECORDS:
        if name not in CAPPED:
            for rows in (1, 10**6):
                assert cake_jit.launch_grid(name, rows, 8193, 148) == cake_jit.launch_grid(name, rows), name


def test_the_pair_table_role_follows_small_docs() -> None:
    """:func:`cake_jit.pair_table_role` gives ``pair_table_small`` for at most ``small_docs`` documents and
    ``pair_table`` for more, on each architecture."""
    for arch in ARCHES:
        for docs in (1, SMALL_DOCS - 1, SMALL_DOCS, SMALL_DOCS + 1, 8193, cake_jit.MAX_DOCS):
            role = "pair_table_small" if docs <= SMALL_DOCS else "pair_table"
            assert cake_jit.pair_table_role(docs, arch) == role, (arch, docs)


def test_the_multiprocessor_count_is_read_once_a_device(monkeypatch) -> None:
    """:func:`cake_jit.sm_count` reads the device's multiprocessor count from torch once per device index (no
    device: the properties are a recorder)."""
    asked = []

    def properties(index):
        asked.append(index)
        return types.SimpleNamespace(multi_processor_count=SMS["sm_100a"] + index)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    cake_jit.sm_count.cache_clear()
    try:
        assert [cake_jit.sm_count(index) for index in (0, 1, 0, 1, 0)] == [148, 149, 148, 149, 148]
        assert asked == [0, 1]
    finally:
        cake_jit.sm_count.cache_clear()


def test_launchers_take_named_buffers() -> None:
    """Each launcher takes its buffers and strides by keyword only, with no default, and the device index, the
    architecture (and the forward's or the backward's role) a public call passes down by keyword only, each defaulting
    to None."""
    for launcher, keywords in LAUNCHER_KEYWORDS.items():
        params = list(inspect.signature(getattr(cake_backend, launcher)).parameters.values())
        assert all(p.kind is p.KEYWORD_ONLY for p in params), launcher
        assert sorted(p.name for p in params) == sorted(keywords + LAUNCHER_OPTIONS[launcher]), launcher
        assert all(p.default is p.empty for p in params if p.name in keywords), launcher
        assert all(p.default is None for p in params if p.name in LAUNCHER_OPTIONS[launcher]), launcher


def test_launchers_pass_every_kernel_argument(monkeypatch) -> None:
    """Each launcher hands ``run_module`` exactly the kernel parameters of its record, with ``num_docs = S``, the
    rows of ``out`` / ``dout`` as ``G``, the strides it was given and the rows of its grid (``S`` for the pair table,
    ``G`` for the forward, ``G + S`` for the backward); the pair table of at most
    512 documents is the ``pair_table_small`` role, of more the ``pair_table`` role; the forward of at most 256
    documents is the ``fwd`` role, of more (257, 512, 513, 8193 and 2^20) the ``fwd_unfused`` role; the backward of 23
    tokens and at most ``bwd_small_docs`` documents (26 on sm_100a / sm_103a, 125 on sm_107a) is the ``bwd_small``
    role, of more the ``bwd`` role, of more than ``bwd_warp_docs`` (16384) and of ``bwd_warp_min_docs`` to
    ``bwd_warp_max_docs`` (15 to 32 on sm_103a) the ``bwd_warp`` role, and of the documents of
    ``ROUTING``, at each token count of ``ROUTING_TOKENS``, its role there; every launcher passes the architecture it
    looked up to ``run_module``, and a device index, an architecture (and a forward or backward role) a caller gives as
    given, with no lookup (no device: the architecture is given and the launch is a recorder)."""
    calls = []
    options = []

    def run_module(name, values, *, rows, device_index, arch=None):
        calls.append((name, dict(values), rows))
        options.append((device_index, arch))

    monkeypatch.setattr(cake_jit, "run_module", run_module)
    monkeypatch.setattr(cake_jit, "device_arch", lambda index: "sm_100a")
    S, G, T = 5, 9, 23
    cu = torch.zeros(S + 1, dtype=torch.int32)
    table = torch.zeros(S + 1, dtype=torch.int32)
    cz = torch.zeros(T, 2 * WIDTH)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    out = torch.zeros(G, WIDTH, dtype=torch.bfloat16)
    dc, dz = torch.zeros(T, WIDTH), torch.zeros(T, WIDTH)
    cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table)
    cake_backend.launch_fwd(c=c, z=z, cu_seqlens=cu, cu_pairs=table, out=out, c_ld=1024, z_ld=1024)
    cake_backend.launch_bwd(
        dout=out, c=c, z=z, cu_seqlens=cu, cu_pairs=table, dc=dc, dz=dz, c_ld=1024, z_ld=1024, dc_ld=512, dz_ld=512
    )
    assert [name for name, _, _ in calls] == ["pair_table_small", "fwd", "bwd_small"]
    # CPU tensors: no device index; each launcher looks the architecture up to choose its role and passes it on
    assert options == [(None, "sm_100a")] * 3
    for name, values, rows in calls:
        assert set(values) == PARAMETERS[name], name
        assert values["num_docs"] == S and values["cu_seqlens"] is cu and values["cu_pairs"] is table, name
    (_, _, rows_table), (_, fwd, rows_fwd), (_, bwd, rows_bwd) = calls
    assert rows_table == S and rows_fwd == G and rows_bwd == G + S
    assert fwd["G"] == bwd["G"] == G
    assert (fwd["c_ld"], fwd["z_ld"], bwd["dc_ld"], bwd["dz_ld"]) == (1024, 1024, 512, 512)
    assert fwd["out"] is out and bwd["dout"] is out and bwd["dc"] is dc and bwd["dz"] is dz
    for docs, role in ((SMALL_DOCS, "pair_table_small"), (SMALL_DOCS + 1, "pair_table")):
        calls.clear()
        bound = torch.zeros(docs + 1, dtype=torch.int32)
        cake_backend.launch_pair_table(cu_seqlens=bound, cu_pairs=torch.zeros_like(bound))
        assert [(name, values["num_docs"], rows) for name, values, rows in calls] == [(role, docs, docs)], docs
        assert set(calls[0][1]) == PARAMETERS[role]
    for arch in ARCHES:
        monkeypatch.setattr(cake_jit, "device_arch", lambda index, arch=arch: arch)
        for docs in (1, FUSE_DOCS - 1, FUSE_DOCS, FUSE_DOCS + 1, SMALL_DOCS, SMALL_DOCS + 1, 8193, 1 << 20):
            role = "fwd" if docs <= FUSE_DOCS else "fwd_unfused"
            calls.clear()
            bound = torch.zeros(docs + 1, dtype=torch.int32)
            cake_backend.launch_fwd(c=c, z=z, cu_seqlens=bound, cu_pairs=bound, out=out, c_ld=1024, z_ld=1024)
            assert [(name, values["num_docs"], rows) for name, values, rows in calls] == [(role, docs, G)], (arch, docs)
            assert set(calls[0][1]) == PARAMETERS[role]
        for docs in (*BWD_BOUND_DOCS, 8193):
            role = _bwd_role(arch, docs)
            calls.clear()
            bound = torch.zeros(docs + 1, dtype=torch.int32)
            cake_backend.launch_bwd(
                dout=out,
                c=c,
                z=z,
                cu_seqlens=bound,
                cu_pairs=bound,
                dc=dc,
                dz=dz,
                c_ld=1024,
                z_ld=1024,
                dc_ld=512,
                dz_ld=512,
            )
            found = [(name, values["num_docs"], rows) for name, values, rows in calls]
            assert found == [(role, docs, G + docs)], (arch, docs)
            assert set(calls[0][1]) == PARAMETERS[role]
        for tokens in ROUTING_TOKENS:
            # T tokens: the rows of c, an expanded view (the launcher reads its shape only)
            rows_c = torch.zeros(1, 2 * WIDTH).expand(tokens, 2 * WIDTH)
            for docs, letter in zip(ROUTING_DOCS, ROUTING[arch], strict=True):
                assert ROUTING_ROLES[letter] == _bwd_role(arch, docs), (arch, docs)
                calls.clear()
                bound = torch.zeros(docs + 1, dtype=torch.int32)
                cake_backend.launch_bwd(
                    dout=out,
                    c=rows_c[:, :WIDTH],
                    z=rows_c[:, WIDTH:],
                    cu_seqlens=bound,
                    cu_pairs=bound,
                    dc=dc,
                    dz=dz,
                    c_ld=1024,
                    z_ld=1024,
                    dc_ld=512,
                    dz_ld=512,
                )
                found = [(name, values["num_docs"], rows) for name, values, rows in calls]
                assert found == [(ROUTING_ROLES[letter], docs, G + docs)], (arch, tokens, docs)
    # A device index, an architecture and a forward or backward role given by the caller reach run_module as given,
    # and no architecture is looked up.
    monkeypatch.setattr(cake_jit, "device_arch", lambda index: pytest.fail("looked up the architecture again"))
    for arch in ARCHES:
        calls.clear()
        options.clear()
        bound = torch.zeros(SMALL_DOCS + 2, dtype=torch.int32)
        cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table, device_index=3, arch=arch)
        cake_backend.launch_pair_table(cu_seqlens=bound, cu_pairs=bound, device_index=3, arch=arch)
        for role in ("fwd", "fwd_unfused"):
            cake_backend.launch_fwd(
                c=c,
                z=z,
                cu_seqlens=cu,
                cu_pairs=table,
                out=out,
                c_ld=1024,
                z_ld=1024,
                role=role,
                device_index=3,
                arch=arch,
            )
        served = [role for role in ("bwd", "bwd_small", "bwd_warp") if arch in ROLE_ARCHES.get(role, ARCHES)]
        for role in (None, *served):
            cake_backend.launch_bwd(
                dout=out,
                c=c,
                z=z,
                cu_seqlens=cu,
                cu_pairs=table,
                dc=dc,
                dz=dz,
                c_ld=1024,
                z_ld=1024,
                dc_ld=512,
                dz_ld=512,
                role=role,
                device_index=3,
                arch=arch,
            )
        # five documents of 23 tokens: the backward's own choice is bwd_small
        backward = ["bwd_small", *served]
        assert [name for name, _, _ in calls] == ["pair_table_small", "pair_table", "fwd", "fwd_unfused", *backward]
        assert options == [(3, arch)] * (4 + len(backward)), arch


def test_the_backward_of_the_calls_takes_the_role_of_its_documents(monkeypatch) -> None:
    """The backward of the explicit call and of the autograd nodes (``_backward``, on the ``S`` of their checks or of
    the forward's plan) launches the role of ``ROUTING`` at each of its documents, at every token count of
    ``ROUTING_TOKENS``, on each architecture, the role ``launch_bwd`` chooses there: ``bwd_warp`` for the documents
    of ``bwd_warp_region``, else ``bwd`` for more than ``bwd_small_docs`` and at most ``bwd_max_docs`` documents, else
    ``bwd_small``; each role resolves to the record of
    ``BACKWARD_RECORDS`` (no device: the table is given, so no pair table is launched, and the launch is a
    recorder)."""
    calls = []

    def run_module(name, values, *, rows, device_index, arch=None):
        calls.append((name, values["num_docs"], rows, arch))

    monkeypatch.setattr(cake_jit, "run_module", run_module)
    G = 9
    out = torch.zeros(G, WIDTH, dtype=torch.bfloat16)
    cpu = torch.device("cpu")
    for arch in ARCHES:
        for tokens in ROUTING_TOKENS:
            rows_c = torch.zeros(1, 2 * WIDTH).expand(tokens, 2 * WIDTH)
            c, z = rows_c[:, :WIDTH], rows_c[:, WIDTH:]
            for docs, letter in zip(ROUTING_DOCS, ROUTING[arch], strict=True):
                calls.clear()
                table = torch.zeros(docs + 1, dtype=torch.int32)
                lds = (1024, 1024, 1024, 1024)
                cake_backend._backward(out, c, z, table, table, c, z, tokens, docs, G, *lds, cpu, None, arch)
                role = ROUTING_ROLES[letter]
                assert calls == [(role, docs, G + docs, arch)], (arch, tokens, docs)
                assert cake_jit.role_record(role, arch) == BACKWARD_RECORDS[arch][role], (arch, role)


def test_each_role_launches_the_record_of_the_device_architecture(monkeypatch) -> None:
    """``run_module`` takes a role and launches the record that serves it on the device's architecture, on that
    record's grid: the forward of ``G`` rows runs ``fwd`` (two rows a CTA) on sm_100a / sm_103a and ``fwd_sm_107a``
    (four rows a CTA) on sm_107a; the pair tables run their one record everywhere, ``pair_table_small`` for 5 and 512
    documents and ``pair_table`` for 513; the forward of 256 documents runs ``fwd`` / ``fwd_sm_107a``, of 257, 512,
    513 and 8193 ``fwd_unfused`` / ``fwd_unfused_sm_103a`` / ``fwd_unfused_sm_107a``; the backward of 23 tokens in 5,
    26, 27, 124, 125, 126,
    16384, 16385 and 32769 documents runs the record of its role on the architecture (``_bwd_role``): on sm_100a
    ``bwd_small`` for 5 and 26, ``bwd`` for 27 to 16384, ``bwd_warp`` for more; on sm_103a ``bwd_small`` for 5,
    ``bwd_warp_sm_103a`` for 26 and 27, ``bwd`` for 124 to 16384, ``bwd_warp_sm_103a`` for more; on sm_107a
    ``bwd_small_sm_107a`` for 5 to 27, ``bwd_warp_sm_107a`` for 124 to 126, ``bwd_sm_107a`` for 16384,
    ``bwd_warp_sm_107a`` for more (no device: the architecture is given and the binding entry is a recorder)."""
    if not all(rec["arg_plan"] for rec in cake_jit.MODULES.values()):
        pytest.skip("the records' argument plans are not rendered yet")
    launched = []

    def entry(name, arch):
        return lambda *args: launched.append((name, arch, len(args), tuple(args[-3:])))

    streams = []

    def use_raw_stream(device, stream):
        streams.append((device, stream))
        return contextlib.nullcontext()

    monkeypatch.setattr(cake_jit, "_entry", entry)
    monkeypatch.setattr(cake_jit, "_tvm_ffi", lambda: types.SimpleNamespace(use_raw_stream=use_raw_stream))
    monkeypatch.setattr(cake_jit, "_tvm_device", lambda index: ("device", index))
    monkeypatch.setattr(cake_jit, "_raw_stream", lambda index: ("stream", index))
    S, G, T = 5, 9, 23
    cu = torch.zeros(S + 1, dtype=torch.int32)
    table = torch.zeros(S + 1, dtype=torch.int32)
    cz = torch.zeros(T, 2 * WIDTH)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    out = torch.zeros(G, WIDTH, dtype=torch.bfloat16)
    dc, dz = torch.zeros(T, WIDTH), torch.zeros(T, WIDTH)
    edge, past = torch.zeros(SMALL_DOCS + 1, dtype=torch.int32), torch.zeros(SMALL_DOCS + 2, dtype=torch.int32)
    fused_edge = torch.zeros(FUSE_DOCS + 1, dtype=torch.int32)
    fused_past = torch.zeros(FUSE_DOCS + 2, dtype=torch.int32)
    tiny = torch.zeros(8193 + 1, dtype=torch.int32)
    backward_docs = [torch.zeros(docs + 1, dtype=torch.int32) for docs in (26, 27, 124, 125, 126, 16384, 16385, 32769)]
    for arch in ARCHES:
        monkeypatch.setattr(cake_jit, "device_arch", lambda index, arch=arch: arch)
        cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table)
        cake_backend.launch_pair_table(cu_seqlens=edge, cu_pairs=torch.zeros_like(edge))
        cake_backend.launch_pair_table(cu_seqlens=past, cu_pairs=torch.zeros_like(past))
        cake_backend.launch_fwd(c=c, z=z, cu_seqlens=cu, cu_pairs=table, out=out, c_ld=1024, z_ld=1024)
        for bound in (fused_edge, fused_past, edge, past, tiny):
            cake_backend.launch_fwd(c=c, z=z, cu_seqlens=bound, cu_pairs=bound, out=out, c_ld=1024, z_ld=1024)
        for bound in (cu, *backward_docs):
            cake_backend.launch_bwd(
                dout=out,
                c=c,
                z=z,
                cu_seqlens=bound,
                cu_pairs=bound,
                dc=dc,
                dz=dz,
                c_ld=1024,
                z_ld=1024,
                dc_ld=512,
                dz_ld=512,
            )
    expected = []
    for arch in ARCHES:
        forward = FORWARD_RECORDS[arch]
        unfused = UNFUSED_RECORDS[arch]
        tables = (("pair_table_small", 1), ("pair_table_small", 1), ("pair_table", 1))
        forwards = ((forward, G), (forward, G), (unfused, G), (unfused, G), (unfused, G), (unfused, G))
        backwards = tuple(
            (BACKWARD_RECORDS[arch][_bwd_role(arch, docs)], G + docs)
            for docs in (S, 26, 27, 124, 125, 126, 16384, 16385, 32769)
        )
        for name, rows in (*tables, *forwards, *backwards):
            grid = cake_jit.launch_grid(name, rows)
            expected.append((name, arch, len(cake_jit.MODULES[name]["arg_plan"]), grid))
    assert launched == expected
    # every launch on the stream handle and tvm-ffi device of its operands' device (CPU tensors here: index None)
    assert streams == [(("device", None), ("stream", None))] * len(expected)
    # nine forward rows: five CTAs of two rows on sm_100a / sm_103a, three CTAs of four rows on sm_107a
    assert [grid for name, _, _, grid in launched if name.startswith("fwd")] == [(5, 1, 1)] * 12 + [(3, 1, 1)] * 6


def test_the_unfused_forward_launch_takes_the_capped_grid(monkeypatch) -> None:
    """The forward launcher's ``fwd_unfused`` launch passes the binding entry the grid :func:`cake_jit.launch_grid`
    gives its rows, documents and the device's multiprocessor count: on sm_100a at most ``ctas_per_sm`` CTAs a
    multiprocessor for ``capped_min_docs <= S <= capped_max_docs`` (at each bound and one past it, for a grid at, just
    above and past the cap), one CTA a block otherwise, and on sm_103a and sm_107a (a ``rows`` record) one CTA a block
    for every pack; :func:`cake_jit.sm_count` is read once a launch, of the launch's device,
    only when the cap can bind, and never for the backward (no device: the binding entry and the multiprocessor count
    are recorders)."""
    if not all(rec["arg_plan"] for rec in cake_jit.MODULES.values()):
        pytest.skip("the records' argument plans are not rendered yet")
    launched = []
    monkeypatch.setattr(cake_jit, "_entry", lambda name, arch: lambda *args: launched.append((name, tuple(args[-3:]))))
    streams = types.SimpleNamespace(use_raw_stream=lambda device, stream: contextlib.nullcontext())
    monkeypatch.setattr(cake_jit, "_tvm_ffi", lambda: streams)
    monkeypatch.setattr(cake_jit, "_tvm_device", lambda index: ("device", index))
    monkeypatch.setattr(cake_jit, "_raw_stream", lambda index: ("stream", index))
    T = 23
    cz = torch.zeros(T, 2 * WIDTH)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    dc, dz = torch.zeros(T, WIDTH), torch.zeros(T, WIDTH)
    for arch in ARCHES:
        name = cake_jit.role_record("fwd_unfused", arch)
        per = cake_jit.rows_per_cta(name)
        if name not in CAPPED:
            # a rows record (sm_103a, sm_107a): one CTA a block for every pack, the SM count never read
            monkeypatch.setattr(cake_jit, "sm_count", lambda index: pytest.fail("read the SM count for a rows record"))
            G = per * 6 * SMS[arch] * 3 + 5
            out = torch.zeros(1, WIDTH, dtype=torch.bfloat16).expand(G, WIDTH)
            for docs in (2047, 2048, 8193, 16384, 16385, 1048577):
                launched.clear()
                bound = torch.zeros(docs + 1, dtype=torch.int32)
                cake_backend.launch_fwd(
                    c=c, z=z, cu_seqlens=bound, cu_pairs=bound, out=out, c_ld=1024, z_ld=1024, device_index=7, arch=arch
                )
                assert launched == [(name, (-(-G // per), 1, 1))], (arch, docs, launched)
            continue
        lo, hi, ctas = CAPPED[name]
        cap = ctas * SMS[arch]
        asked = []

        def sm_count(index, arch=arch, asked=asked):
            asked.append(index)
            return SMS[arch]

        monkeypatch.setattr(cake_jit, "sm_count", sm_count)
        for G in (per * ctas, per * ctas + 1, per * cap, per * cap + 1, per * cap * 3 + 5):
            # G output rows: an expanded view (the launcher reads its shape only)
            out = torch.zeros(1, WIDTH, dtype=torch.bfloat16).expand(G, WIDTH)
            full = -(-G // per)
            for docs in (lo - 1, lo, hi, hi + 1):
                launched.clear()
                asked.clear()
                bound = torch.zeros(docs + 1, dtype=torch.int32)
                cake_backend.launch_fwd(
                    c=c, z=z, cu_seqlens=bound, cu_pairs=bound, out=out, c_ld=1024, z_ld=1024, device_index=7, arch=arch
                )
                inside = lo <= docs <= hi
                want = (min(full, cap) if inside else full, 1, 1)
                assert launched == [(name, want)], (arch, G, docs, launched)
                assert want == cake_jit.launch_grid(name, G, docs, SMS[arch]), (arch, G, docs)
                assert asked == ([7] if inside and full > ctas else []), (arch, G, docs, asked)
        monkeypatch.setattr(cake_jit, "sm_count", lambda index: pytest.fail("read the SM count for a rows record"))
        G = per * cap * 3 + 5
        out = torch.zeros(1, WIDTH, dtype=torch.bfloat16).expand(G, WIDTH)
        for docs in (lo, hi):
            launched.clear()
            bound = torch.zeros(docs + 1, dtype=torch.int32)
            cake_backend.launch_bwd(
                dout=out,
                c=c,
                z=z,
                cu_seqlens=bound,
                cu_pairs=bound,
                dc=dc,
                dz=dz,
                c_ld=1024,
                z_ld=1024,
                dc_ld=512,
                dz_ld=512,
                device_index=7,
                arch=arch,
            )
            role = cake_jit.role_record(_bwd_role(arch, docs), arch)
            assert launched == [(role, cake_jit.launch_grid(role, G + docs))], (arch, docs)


def test_a_launch_without_a_kernel_argument_is_refused(monkeypatch) -> None:
    """``run_module`` refuses ``values`` that lack a kernel parameter of the record with a ``ValueError`` naming the
    record and the parameter, before it looks up the binding entry (no device: the entry lookup fails the test)."""
    monkeypatch.setattr(cake_jit, "_entry", lambda name, arch: pytest.fail("looked up the binding entry"))
    missing = r"record pair_table_small needs 'num_docs' \(kernel parameter 'num_docs'\)"
    for arch in ARCHES:
        with pytest.raises(ValueError, match=missing):
            cake_jit.run_module("pair_table_small", {"cu_seqlens": 0, "cu_pairs": 0}, rows=1, device_index=0, arch=arch)
        bwd = cake_jit.role_record("bwd", arch)
        values = dict.fromkeys(PARAMETERS["bwd"] - {"dz_ld"}, 0)
        with pytest.raises(ValueError, match=rf"record {bwd} needs 'dz_ld' \(kernel parameter 'dz_ld'\)"):
            cake_jit.run_module("bwd", values, rows=9, device_index=0, arch=arch)


def test_the_first_lookup_on_an_architecture_loads_every_role(monkeypatch) -> None:
    """The first FFI lookup on an architecture loads the library of every role served there, in role order, and later
    lookups of any record load nothing new: a pack of more than ``fuse_docs``, ``small_docs`` or ``bwd_small_docs``
    documents, or of the documents of ``bwd_warp_region``, after smaller ones finds ``fwd_unfused``, ``pair_table``,
    ``bwd`` and ``bwd_warp`` loaded (no device: ``load`` is a recorder)."""
    loaded = []

    def load(name, arch):
        loaded.append((name, arch))
        return types.SimpleNamespace(run=lambda *args: None)

    monkeypatch.setattr(cake_jit, "load", load)
    cake_jit._entry.cache_clear()
    cake_jit._libraries.cache_clear()
    try:
        for arch in ARCHES:
            loaded.clear()
            cake_jit._entry(cake_jit.role_record("pair_table_small", arch), arch)
            served = [role for role in ROLES if arch in ROLE_ARCHES.get(role, ARCHES)]
            assert len(served) == 7, arch
            every = [(cake_jit.role_record(role, arch), arch) for role in served]
            assert loaded[: len(every)] == every, arch
            seen = set(loaded)
            for role in served:
                cake_jit._entry(cake_jit.role_record(role, arch), arch)
            assert set(loaded) == seen, arch
    finally:
        cake_jit._entry.cache_clear()
        cake_jit._libraries.cache_clear()


def test_the_overlap_rule_is_exact_for_column_halves() -> None:
    """The two column halves of one ``[T, 1024]`` buffer share no element; a view overlaps itself, a shifted window,
    a row of the other half's buffer region, and a dense alias; separate allocations never overlap."""
    overlaps = cake_backend._overlaps
    buffer = torch.zeros(7, 2 * WIDTH)
    left, right = buffer[:, :WIDTH], buffer[:, WIDTH:]
    assert not overlaps(left, right) and not overlaps(right, left)
    assert overlaps(left, left) and overlaps(right, right)
    assert overlaps(left, buffer[:, 256:768]) and overlaps(buffer[:, 256:768], right)
    assert overlaps(left, buffer[3:4, :WIDTH]) and not overlaps(left[:3], buffer[3:, :WIDTH])
    assert overlaps(left, buffer.view(-1)[5 * 2 * WIDTH : 5 * 2 * WIDTH + 8])
    assert not overlaps(left, buffer.view(-1)[WIDTH : WIDTH + 8])  # inside row 0 of the right half
    assert not overlaps(left, torch.zeros(7, WIDTH)) and not overlaps(left, torch.zeros(0, WIDTH))


def test_the_disjoint_check_refuses_exactly_the_overlapping_pairs() -> None:
    """``_disjoint`` (which reads the written tensor's byte range once and runs the exact test only where two byte
    ranges meet) refuses a written tensor exactly when ``_overlaps`` finds it shares a byte with one of the others,
    naming the first such other; ``None`` and empty others are skipped."""
    buffer = torch.zeros(7, 2 * WIDTH)
    flat = buffer.view(-1)
    views = {
        "left": buffer[:, :WIDTH],
        "right": buffer[:, WIDTH:],
        "middle": buffer[:, 256:768],
        "row3": buffer[3:4, :WIDTH],
        "rows3-": buffer[3:, :WIDTH],
        "left3": buffer[:3, :WIDTH],
        "alias": flat[5 * 2 * WIDTH : 5 * 2 * WIDTH + 8],
        "inside-right": flat[WIDTH : WIDTH + 8],
        "separate": torch.zeros(7, WIDTH),
        "empty": torch.zeros(0, WIDTH),
    }
    for written_name, written in views.items():
        others = tuple((name, view) for name, view in views.items() if name != written_name) + (("none", None),)
        hits = [name for name, view in others if view is not None and cake_backend._overlaps(written, view)]
        if hits:
            with pytest.raises(ValueError, match=f"^{written_name} must not overlap {hits[0]}$"):
                cake_backend._disjoint((written_name, written), others)
        else:
            cake_backend._disjoint((written_name, written), others)
        for name, view in others:
            pair = ((name, view),)
            if view is not None and cake_backend._overlaps(written, view):
                with pytest.raises(ValueError, match=f"^{written_name} must not overlap {name}$"):
                    cake_backend._disjoint((written_name, written), pair)
            else:
                cake_backend._disjoint((written_name, written), pair)


def _outcome(check, *args):
    """``("ok", result)`` of ``check(*args)``, or ``("error", exception type, message)``."""
    try:
        return ("ok", check(*args))
    except (TypeError, ValueError) as error:
        return ("error", type(error), str(error))


def _offset(dtype: torch.dtype, shape: tuple[int, ...], offset_bytes: int) -> torch.Tensor:
    """A contiguous CPU tensor of ``shape`` whose base is ``offset_bytes`` past a fresh 64-byte-aligned allocation."""
    size = torch.tensor([], dtype=dtype).element_size()
    count = 1
    for n in shape:
        count *= n
    storage = torch.zeros(count + 64, dtype=dtype)
    skip = (-storage.data_ptr() % 64 + offset_bytes) // size
    return storage[skip : skip + count].view(shape)


def _token_variants() -> list[tuple[str, tuple]]:
    """``(label, (c, z, cu_seqlens))`` on the CPU: well-formed operands (column halves, separate tensors, padded rows,
    one and no token, a subclass) and one malformed operand each (every check of the token operands)."""
    cz = torch.zeros(7, 2 * WIDTH)
    c, z = cz[:, :WIDTH], cz[:, WIDTH:]
    cu = torch.tensor([0, 2, 7], dtype=torch.int32)
    padded = torch.zeros(7, 520)[:, :WIDTH]
    return [
        ("halves", (c, z, cu)),
        ("separate", (torch.zeros(7, WIDTH), torch.zeros(7, WIDTH), cu)),
        ("padded rows", (padded, z, cu)),
        ("one token", (torch.zeros(1, WIDTH), torch.zeros(1, WIDTH), torch.tensor([0, 1], dtype=torch.int32))),
        ("no token", (torch.zeros(0, WIDTH), torch.zeros(0, WIDTH), torch.zeros(1, dtype=torch.int32))),
        ("parameter c", (torch.nn.Parameter(torch.zeros(7, WIDTH)), z, cu)),
        ("c not a tensor", ([0.0] * WIDTH, z, cu)),
        ("z not a tensor", (c, None, cu)),
        ("cu_seqlens not a tensor", (c, z, [0, 2, 7])),
        ("z on meta", (c, z.to("meta"), cu)),
        ("cu_seqlens on meta", (c, z, cu.to("meta"))),
        ("c dtype", (c.to(torch.bfloat16), z, cu)),
        ("z dtype", (c, z.double(), cu)),
        ("cu_seqlens dtype", (c, z, cu.long())),
        ("c rank", (c[None], z, cu)),
        ("c width", (cz[:, :256], z, cu)),
        ("z rows", (c, cz[:6, WIDTH:], cu)),
        ("c feature stride", (torch.zeros(WIDTH, 7).t(), z, cu)),
        ("z feature stride", (c, torch.zeros(WIDTH, 7).t(), cu)),
        ("c row stride off 8", (torch.zeros(7, 1028)[:, :WIDTH], z, cu)),
        ("z rows overlap", (c, torch.zeros(7 * 256 + 256).as_strided((7, WIDTH), (256, 1)), cu)),
        ("c misaligned", (cz[:, 4 : 4 + WIDTH], z, cu)),
        ("z misaligned", (c, cz[:, WIDTH - 4 : 2 * WIDTH - 4], cu)),
        ("cu_seqlens rank", (c, z, cu[None])),
        ("cu_seqlens empty", (c, z, cu[:0])),
        ("cu_seqlens strided", (c, z, torch.zeros(6, dtype=torch.int32)[::2])),
        ("too many documents", (c, z, torch.zeros(cake_jit.MAX_DOCS + 2, dtype=torch.int32))),
    ]


def test_the_fast_operand_checks_give_the_checks_result_or_defer() -> None:
    """For every token-operand variant: the fast path returns the full checks' result or defers (``None``), and
    ``_token_operands`` gives exactly the full checks' outcome: the same result, or the same exception type and
    message (on the CPU device, given; without a device, every CPU operand is refused alike)."""
    cpu = torch.device("cpu")
    deferred = []
    for label, (c, z, cu) in _token_variants():
        want = _outcome(cake_backend._check_token_operands, c, z, cu, cpu)
        fast = cake_backend._token_operands_fast(c, z, cu, cpu)
        if fast is None:
            deferred.append(label)
        else:
            assert want == ("ok", fast), label
        assert _outcome(cake_backend._token_operands, c, z, cu, cpu) == want, label
        assert _outcome(cake_backend._token_operands, c, z, cu, None) == _outcome(
            cake_backend._check_token_operands, c, z, cu, None
        ), label
    # the plain operands of two tokens or more take the fast path; the others and every malformed variant defer
    assert sorted({label for label, _ in _token_variants()} - set(deferred)) == [
        "halves",
        "padded rows",
        "separate",
    ]


def _backward_variants() -> list[tuple[str, dict]]:
    """``(label, operands)`` of the explicit backward on the CPU: well-formed operands with and without caller
    ``dc`` / ``dz`` and ``pair_offsets``, and one malformed or overlapping operand each."""
    cz = torch.zeros(7, 2 * WIDTH)
    base = {
        "dout": torch.zeros(3, WIDTH, dtype=torch.bfloat16),
        "c": cz[:, :WIDTH],
        "z": cz[:, WIDTH:],
        "cu_seqlens": torch.tensor([0, 2, 7], dtype=torch.int32),
        "dc": None,
        "dz": None,
        "pair_offsets": None,
    }
    grads = torch.zeros(7, 2 * WIDTH)
    overlapping = torch.zeros(8, WIDTH)
    shared = torch.zeros(7, WIDTH)
    variants = [
        ("allocated", {}),
        ("given halves", {"dc": grads[:, :WIDTH], "dz": grads[:, WIDTH:]}),
        ("given dc", {"dc": torch.zeros(7, WIDTH)}),
        ("given dz", {"dz": torch.zeros(7, WIDTH)}),
        ("pair_offsets", {"pair_offsets": torch.tensor([0, 1, 3], dtype=torch.int32)}),
        ("no pair", {"dout": torch.zeros(0, WIDTH, dtype=torch.bfloat16)}),
        ("dout dtype", {"dout": torch.zeros(3, WIDTH)}),
        ("dout rank", {"dout": torch.zeros(3, WIDTH, 1, dtype=torch.bfloat16)}),
        ("dout width", {"dout": torch.zeros(3, 511, dtype=torch.bfloat16)}),
        ("dout strided", {"dout": torch.zeros(WIDTH, 3, dtype=torch.bfloat16).t()}),
        ("dout misaligned", {"dout": _offset(torch.bfloat16, (3, WIDTH), 8)}),
        ("dc not a tensor", {"dc": [0.0]}),
        ("dc on meta", {"dc": torch.zeros(7, WIDTH, device="meta")}),
        ("dc dtype", {"dc": torch.zeros(7, WIDTH, dtype=torch.bfloat16)}),
        ("dc rows", {"dc": torch.zeros(6, WIDTH)}),
        ("dz width", {"dz": torch.zeros(7, 256)}),
        ("dz feature stride", {"dz": torch.zeros(WIDTH, 7).t()}),
        ("dz row stride off 8", {"dz": torch.zeros(7, 1028)[:, :WIDTH]}),
        ("dz misaligned", {"dz": _offset(torch.float32, (7, WIDTH), 16)}),
        ("dc is dz", {"dc": shared, "dz": shared}),
        ("dc over dz", {"dc": overlapping[:7], "dz": overlapping[1:]}),
        ("dc over c", {"dc": cz[:, :WIDTH]}),
        ("dz over z", {"dz": cz[:, WIDTH:]}),
        ("dc over cu_seqlens", {"dc": torch.zeros(7, WIDTH)}),
        ("pair_offsets not a tensor", {"pair_offsets": [0, 1, 3]}),
        ("pair_offsets on meta", {"pair_offsets": torch.zeros(3, dtype=torch.int32, device="meta")}),
        ("pair_offsets dtype", {"pair_offsets": torch.tensor([0, 1, 3])}),
        ("pair_offsets length", {"pair_offsets": torch.tensor([0, 1], dtype=torch.int32)}),
        ("pair_offsets strided", {"pair_offsets": torch.zeros(6, dtype=torch.int32)[::2]}),
    ]
    out = []
    for label, edit in variants:
        o = dict(base, **edit)
        if label == "dc over cu_seqlens":
            o["cu_seqlens"] = o["dc"].view(-1)[:3].view(torch.int32)
            o["cu_seqlens"][:] = torch.tensor([0, 2, 7], dtype=torch.int32)
        out.append((label, o))
    buffer = torch.zeros(7, WIDTH)
    dout = buffer.view(-1).view(torch.bfloat16)[: 3 * WIDTH].view(3, WIDTH)
    out.append(("dz over dout", dict(base, dz=buffer, dout=dout)))
    out += [(f"token: {label}", dict(base, c=c, z=z, cu_seqlens=cu)) for label, (c, z, cu) in _token_variants()]
    return out


def _same_grads(fast, want, o) -> None:
    """The fast path's backward operands equal the checks': every host value, a caller ``dc`` / ``dz`` itself and an
    allocated one of the same dtype, device, shape and strides."""
    assert fast[:6] == want[:6] and (fast[7], fast[9]) == (want[7], want[9])
    for name, got, ref in (("dc", fast[6], want[6]), ("dz", fast[8], want[8])):
        if o[name] is not None:
            assert got is o[name] and ref is o[name], name
        else:
            assert (got.dtype, got.device, got.shape, got.stride()) == (ref.dtype, ref.device, ref.shape, ref.stride())


def test_the_fast_backward_checks_give_the_checks_result_or_defer() -> None:
    """For every explicit-backward variant on the CPU device: the fast path gives the full checks' outcome (the same
    host values and caller outputs, or the same exception type and message) or defers (``None``); an output the call
    allocates takes no overlap check.  The plain well-formed operands of at least two tokens and one pair take the fast
    path."""
    cpu = torch.device("cpu")
    names = ("dout", "c", "z", "cu_seqlens", "dc", "dz", "pair_offsets")
    taken = []
    for label, o in _backward_variants():
        args = [o[name] for name in names]
        want = _outcome(cake_backend._check_backward_operands, *args, cpu)
        fast = _outcome(cake_backend._backward_operands_fast, *args, cpu)
        if fast == ("ok", None):
            continue
        taken.append(label)
        assert fast[0] == want[0], (label, fast, want)
        if want[0] == "ok":
            _same_grads(fast[1], want[1], o)
        else:
            assert fast == want, label
    assert sorted(taken) == sorted(
        [
            "allocated",
            "given halves",
            "given dc",
            "given dz",
            "pair_offsets",
            "dc is dz",
            "dc over dz",
            "dc over c",
            "dz over z",
            "dc over cu_seqlens",
            "dz over dout",
            "token: halves",
            "token: separate",
            "token: padded rows",
        ]
    )


def test_the_memory_report_separates_workspace_and_retained_state() -> None:
    for T, S in ((0, 0), (16384, 27), (65536, 68), (7, 1)):
        report = pkg.cake_gated_kv_compression_det_train_memory_bytes(T, S)
        table = 4 * (S + 1)
        assert report == {
            "forward_workspace": table,
            "backward_workspace": table,
            "retained_for_backward": table,
            "retained_inputs": 4096 * T + table,
        }
    with pytest.raises(ValueError):
        pkg.cake_gated_kv_compression_det_train_memory_bytes(-1, 2)
    with pytest.raises(TypeError):
        pkg.cake_gated_kv_compression_det_train_memory_bytes(torch.tensor(3), 2)


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]}
    assert set(entries) == wanted
    assert len(entries) == sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert sorted(entry["arch"] for entry in entries.values()) == sorted(
        arch for _, arches, _ in RECORDS.values() for arch in arches
    )
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
    records = sum(len(rec["arches"]) for rec in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path."""
    _manifest_or_skip()
    cake_build = cake_package(PACKAGE_KEY, "cake_build")
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_jit.library_name(name, arch)
            assert cake_build.prebuilt_path(library).resolve() == PACKAGE_DIR / "aot" / f"{library}.so"


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
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES


def test_the_host_never_reads_device_values() -> None:
    """No module of the package (the JIT shim aside) calls a tensor-to-host conversion: ``.item()``, ``.tolist()``,
    ``.cpu()``, ``.numpy()``, ``.to("cpu")`` or ``.synchronize()``."""
    banned = {"item", "tolist", "cpu", "numpy", "synchronize"}
    offenders = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name in ("_cake_pplx_jit.py", "cake_build.py"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in banned:
                    offenders.append(f"{path.name}:{node.lineno} .{node.func.attr}()")
                if node.func.attr == "to" and any(isinstance(a, ast.Constant) and a.value == "cpu" for a in node.args):
                    offenders.append(f"{path.name}:{node.lineno} .to('cpu')")
    assert not offenders, offenders


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: every record loads; the pair table


def test_every_record_loads(device) -> None:
    """Every record serving the device's architecture loads: one per role served there."""
    arch = cake_jit.device_arch(device.index)
    assert arch in ARCHES
    names = [cake_jit.role_record(role, arch) for role in ROLES if arch in ROLE_ARCHES.get(role, ARCHES)]
    for name in names:
        assert callable(getattr(cake_jit.load(name, arch), cake_jit.MODULES[name]["ffi_entry"])), name


@pytest.mark.parametrize("pack", TABLE_PACKS, ids=[p.id for p in TABLE_PACKS])
def test_the_pair_table_is_the_prefix_of_the_pair_counts(pack, device) -> None:
    """``cu_pairs`` (filled with -7 first) equals the exclusive prefix of ``floor(length / 2)`` exactly, also with
    empty documents and more documents than one tile, from ``pair_table_small`` up to 512 documents and from
    ``pair_table`` above."""
    cu = torch.tensor(bounds(pack.lengths), dtype=torch.int32, device=device)
    table = torch.full((pack.S + 1,), -7, dtype=torch.int32, device=device)
    cake_backend.launch_pair_table(cu_seqlens=cu, cu_pairs=table)
    torch.cuda.synchronize(device)
    assert torch.equal(table, pair_table_of(pack.lengths, device))


# ----------------------------------------------------------------------------------------------------------------
# Each kernel on its own, judged against the naive BF16 tier


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("pack", PAIR_PACKS, ids=[p.id for p in PAIR_PACKS])
def test_the_forward_kernel_matches_the_reference(pack, layout, device, record_property) -> None:
    """``out`` (NaN first) of the forward kernel alone, on the pair table of the pack, meets the tier."""

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout=layout)
        out = nan_like_rows(x.G, torch.bfloat16, device)
        table = pair_table_of(pack.lengths, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=out, **_lds(x))
        assert_forward_contract("fwd", x, out)
        return judged(x, {"fwd": {"out": out}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_forward_kernel_matches_the_reference_on_extreme_inputs(variant, device, record_property) -> None:
    def draw(seed: int):
        x = make_inputs(MIXED, seed=seed, device=device, layout="halves", **variant)
        out = nan_like_rows(x.G, torch.bfloat16, device)
        table = pair_table_of(MIXED.lengths, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=out, **_lds(x))
        assert_forward_contract("fwd", x, out)
        return judged(x, {"fwd": {"out": out}})

    judge_draws(record_property, draw)


def test_forward_rows_past_the_pair_count_are_zero(device) -> None:
    """A caller ``G`` above the pair count: the rows past it are +0.0 (NaN first), the rows below are the bits of
    the call with the exact ``G``."""
    for pack in (MIXED, Pack((0, 1, 3, 2, 7, 4))):
        x = make_inputs(pack, seed=5, device=device)
        table = pair_table_of(pack.lengths, device)
        exact = nan_like_rows(pack.G, torch.bfloat16, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=exact, **_lds(x))
        wide = nan_like_rows(pack.G + 37, torch.bfloat16, device)
        cake_backend.launch_fwd(c=x.c, z=x.z, cu_seqlens=x.cu, cu_pairs=table, out=wide, **_lds(x))
        assert_bitwise(wide[: pack.G], exact, f"{pack.id}: the rows below the pair count")
        assert_zero_rows(wide, range(pack.G, pack.G + 37), f"{pack.id}: the rows past the pair count")


def test_forward_with_fewer_rows_than_pairs_gives_the_first_rows(device) -> None:
    """A caller ``G`` below the pair count: ``out`` (NaN first) has the bits of the first ``G`` rows of the call with
    the exact ``G``."""
    for pack in (MIXED, Pack((0, 1, 3, 2, 7, 4))):
        x = make_inputs(pack, seed=7, device=device, layout="halves")
        exact = pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, pack.G)
        for rows in sorted({1, pack.G // 2, pack.G - 1}):
            short = nan_like_rows(rows, torch.bfloat16, device)
            pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, out=short)
            assert_bitwise(short, exact[:rows], f"{pack.id}: G = {rows} of {pack.G} pairs")


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("pack", PAIR_PACKS, ids=[p.id for p in PAIR_PACKS])
def test_the_backward_kernel_matches_the_reference(pack, layout, device, record_property) -> None:
    """``dC`` and ``dZ`` (NaN first; with ``halves``, the two halves of one NaN ``[T, 1024]`` buffer) of the backward
    kernel alone meet the tier, the tail rows exactly +0.0."""

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout=layout)
        dc, dz = _grad_buffers(x, layout)
        table = pair_table_of(pack.lengths, device)
        cake_backend.launch_bwd(
            dout=x.dout,
            c=x.c,
            z=x.z,
            cu_seqlens=x.cu,
            cu_pairs=table,
            dc=dc,
            dz=dz,
            dc_ld=dc.stride(0),
            dz_ld=dz.stride(0),
            **_lds(x),
        )
        assert_backward_contract("bwd", x, dc, dz)
        return judged(x, {"bwd": {"dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_backward_kernel_matches_the_reference_on_extreme_inputs(variant, device, record_property) -> None:
    def draw(seed: int):
        x = make_inputs(MIXED, seed=seed, device=device, layout="halves", **variant)
        dc, dz = _grad_buffers(x, "halves")
        table = pair_table_of(MIXED.lengths, device)
        cake_backend.launch_bwd(
            dout=x.dout,
            c=x.c,
            z=x.z,
            cu_seqlens=x.cu,
            cu_pairs=table,
            dc=dc,
            dz=dz,
            dc_ld=dc.stride(0),
            dz_ld=dz.stride(0),
            **_lds(x),
        )
        assert_backward_contract("bwd", x, dc, dz)
        return judged(x, {"bwd": {"dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


# ----------------------------------------------------------------------------------------------------------------
# The composed calls and their invariances, bit for bit


@pytest.mark.parametrize("pack", PAIR_PACKS, ids=[p.id for p in PAIR_PACKS])
def test_the_composed_calls_match_the_reference(pack, device, record_property) -> None:
    """The backend's forward and backward entries (pair table, then the kernel), on the column halves, meet the
    tier."""

    def draw(seed: int):
        x = make_inputs(pack, seed=seed, device=device, layout="halves")
        out = pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
        dc, dz = pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
        assert_forward_contract("forward", x, out)
        assert_backward_contract("backward", x, dc, dz)
        return judged(x, {"composed": {"out": out, "dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


def _run(x, layout: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = nan_like_rows(x.G, torch.bfloat16, x.c.device)
    pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, out=out)
    dc, dz = _grad_buffers(x, layout)
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    return out, dc, dz


@pytest.mark.parametrize("pack", (MIXED, *PERF_PACKS), ids=[p.id for p in (MIXED, *PERF_PACKS)])
def test_repeated_calls_give_the_same_bits(pack, device) -> None:
    """Three calls on the same inputs (outputs NaN first each time) give the same bits: the deterministic contract."""
    for layout in LAYOUTS:
        x = make_inputs(pack, seed=17, device=device, layout=layout, gates=True)
        first = _run(x, layout)
        for _ in range(2):
            for name, got, want in zip(("out", "dc", "dz"), _run(x, layout), first, strict=True):
                assert_bitwise(got, want, f"{pack.id} {layout}: {name} run to run")


@pytest.mark.parametrize("pack", (MIXED, *PERF_PACKS), ids=[p.id for p in (MIXED, *PERF_PACKS)])
def test_column_halves_give_the_bits_of_contiguous_operands(pack, device) -> None:
    """``C`` / ``Z`` as the halves of one ``[T, 1024]`` tensor and the gradients written into the halves of one
    buffer give the bits of contiguous operands and gradients."""
    halves = make_inputs(pack, seed=23, device=device, layout="halves")
    dense = make_inputs(pack, seed=23, device=device, layout="contiguous")
    assert torch.equal(halves.c, dense.c) and torch.equal(halves.z, dense.z)
    for name, got, want in zip(("out", "dc", "dz"), _run(halves, "halves"), _run(dense, "contiguous"), strict=True):
        assert_bitwise(got, want, f"{pack.id}: {name} of the halves against contiguous")


def _mixed_strides(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(c, z, dc, dz)`` with four row strides: ``C`` contiguous (512), ``Z`` the right half of a ``[T, 1024]`` tensor
    whose left half is NaN (1024), ``dC`` the left half of a NaN ``[T, 1024]`` buffer (1024) and ``dZ`` the first 512
    columns of a NaN ``[T, 520]`` buffer (520)."""
    device = x.c.device
    z_buffer = torch.full((x.T, 2 * WIDTH), float("nan"), dtype=torch.float32, device=device)
    z_buffer[:, WIDTH:] = x.z
    _, dc, _ = nan_halves(x.T, device)
    dz = torch.full((x.T, WIDTH + 8), float("nan"), dtype=torch.float32, device=device)[:, :WIDTH]
    return x.c.contiguous(), z_buffer[:, WIDTH:], dc, dz


def test_mixed_row_strides_give_the_bits_of_contiguous_operands(device, record_property) -> None:
    """``C`` contiguous beside ``Z`` the right half of a ``[T, 1024]`` tensor, and ``dC`` the left half of a NaN
    ``[T, 1024]`` buffer beside ``dZ`` the first 512 columns of a NaN ``[T, 520]`` buffer: every operand is read or
    written at its own row stride, so the outputs meet the tier and have the bits of the all-contiguous call."""

    def draw(seed: int):
        x = make_inputs(MIXED, seed=seed, device=device)
        c, z, dc, dz = _mixed_strides(x)
        assert (c.stride(0), z.stride(0), dc.stride(0), dz.stride(0)) == (WIDTH, 2 * WIDTH, 2 * WIDTH, WIDTH + 8)
        out = pkg.cake_gated_kv_compression_det_train_forward(c, z, x.cu, x.G)
        pkg.cake_gated_kv_compression_det_train_backward(x.dout, c, z, x.cu, dc=dc, dz=dz)
        for name, got, want in zip(("out", "dc", "dz"), (out, dc, dz), _run(x, "contiguous"), strict=True):
            assert_bitwise(got, want, f"{name} of mixed row strides against contiguous")
        assert_forward_contract("forward", x, out)
        assert_backward_contract("backward", x, dc, dz)
        return judged(x, {"mixed": {"out": out, "dc": dc, "dz": dz}})

    judge_draws(record_property, draw)


@pytest.mark.parametrize("extra", (1, 5, 37), ids=lambda k: f"G+{k}")
@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED, PERF_PACKS[0]), ids=lambda p: p.id)
def test_rows_past_the_pair_count_change_no_gradient(pack, extra, device) -> None:
    """``G`` = the pair count + ``extra``, with NaN in the rows of ``dout`` past the pair count: the explicit backward
    (``dC`` / ``dZ`` NaN first) and the autograd backward of ``cake_gated_kv_compression_det_cz`` give the bits of the
    explicit backward with the exact ``G`` (the rows past the pair count are not read, every token row is written),
    and the autograd forward's rows past the pair count are +0.0."""
    x = make_inputs(pack, seed=11, device=device, layout="halves")
    wide = torch.cat([x.dout, nan_like_rows(extra, torch.bfloat16, device)])
    _, dc, dz = nan_halves(x.T, device)
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    _, dc_wide, dz_wide = nan_halves(x.T, device)
    pkg.cake_gated_kv_compression_det_train_backward(wide, x.c, x.z, x.cu, dc=dc_wide, dz=dz_wide)
    assert_bitwise(dc_wide, dc, f"{pack.id}: explicit dC with G + {extra}")
    assert_bitwise(dz_wide, dz, f"{pack.id}: explicit dZ with G + {extra}")
    out = pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, pack.G)
    cz = x.cz.detach().clone().requires_grad_(True)
    out_wide = pkg.cake_gated_kv_compression_det_cz(cz, x.cu, pack.G + extra)
    # A freed NaN block of the gradient's size: the backward's uninitialized gradient most likely reuses it.
    poison = torch.full((x.T, 2 * WIDTH), float("nan"), dtype=torch.float32, device=device)
    del poison
    out_wide.backward(wide)
    assert_bitwise(out_wide.detach()[: pack.G], out, f"{pack.id}: autograd out with G + {extra}")
    assert_zero_rows(out_wide.detach(), range(pack.G, pack.G + extra), f"{pack.id}: autograd out past the pairs")
    assert_bitwise(cz.grad[:, :WIDTH], dc, f"{pack.id}: autograd dC with G + {extra}")
    assert_bitwise(cz.grad[:, WIDTH:], dz, f"{pack.id}: autograd dZ with G + {extra}")


@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED), ids=["L0-1-3-2-7-4", MIXED.id])
def test_each_document_alone_gives_the_bits_of_the_pack(pack, device) -> None:
    """Each document run alone (its own rows of ``C`` / ``Z``, ``cu_seqlens = [0, L]``, its rows of ``dout``) gives
    the bits of its rows in the pack's ``out``, ``dC`` and ``dZ``."""
    x = make_inputs(pack, seed=29, device=device)
    out, dc, dz = _run(x, "contiguous")
    b = bounds(pack.lengths)
    row = 0
    for s, n in enumerate(pack.lengths):
        pairs = n // 2
        doc = Pack((n,))
        y = make_inputs(doc, seed=0, device=device)
        y.c, y.z = x.c[b[s] : b[s + 1]].contiguous(), x.z[b[s] : b[s + 1]].contiguous()
        y.dout = x.dout[row : row + pairs].contiguous()
        got_out, got_dc, got_dz = _run(y, "contiguous")
        assert_bitwise(got_out, out[row : row + pairs], f"document {s} (length {n}): out")
        assert_bitwise(got_dc, dc[b[s] : b[s + 1]], f"document {s} (length {n}): dc")
        assert_bitwise(got_dz, dz[b[s] : b[s + 1]], f"document {s} (length {n}): dz")
        row += pairs


@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED, *PERF_PACKS), ids=lambda p: p.id)
def test_the_tail_tokens_change_no_output(pack, device) -> None:
    """New values of ``C`` and ``Z`` on every odd-length document's last token (``+1e6`` / ``-1e6``, then NaN) change
    no byte of ``out``, ``dC`` or ``dZ``; the tail rows of ``dC`` and ``dZ`` stay +0.0."""
    tails = tail_rows(pack.lengths)
    if not tails:
        pytest.skip("the pack has no odd-length document")
    x = make_inputs(pack, seed=31, device=device, layout="halves")
    want = _run(x, "halves")
    index = torch.tensor(tails, dtype=torch.long, device=device)
    for c_value, z_value in ((1e6, -1e6), (float("nan"), float("nan"))):
        x.cz[index, :WIDTH] = c_value
        x.cz[index, WIDTH:] = z_value
        for name, got, ref in zip(("out", "dc", "dz"), _run(x, "halves"), want, strict=True):
            assert_bitwise(got, ref, f"{pack.id}: {name} after the tail tokens changed to {c_value}")


def test_a_launch_takes_the_current_stream_of_its_device(device) -> None:
    """The stream handle a launch takes (``cake_jit._raw_stream``) is ``torch.cuda.current_stream(index).cuda_stream``
    of the operands' device: its current stream, and a side stream made current."""
    assert cake_jit._raw_stream(device.index) == torch.cuda.current_stream(device).cuda_stream
    side = torch.cuda.Stream(device)
    with torch.cuda.stream(side):
        assert cake_jit._raw_stream(device.index) == side.cuda_stream != 0
    assert cake_jit._raw_stream(device.index) == torch.cuda.current_stream(device).cuda_stream


def test_the_calls_launch_their_records_in_order(device, monkeypatch) -> None:
    """The forward of a pack of at most ``fuse_docs`` (256) documents launches ``fwd`` alone (the record of the
    device's architecture: ``fwd`` on sm_100a / sm_103a, ``fwd_sm_107a`` on sm_107a), which builds the pair table
    itself; the forward of more documents (257, 300, 512, 513, 8193) launches a pair table then ``fwd_unfused``
    (``fwd_unfused`` on sm_100a, ``fwd_unfused_sm_103a`` on sm_103a, ``fwd_unfused_sm_107a`` on sm_107a); the backward
    of those packs a pair
    table then the record of their backward role (``bwd`` / ``bwd_sm_107a``), and of a pack of at most
    ``bwd_small_docs`` documents a pair table then
    ``bwd_small`` (``bwd_small`` on sm_100a / sm_103a, ``bwd_small_sm_107a`` on sm_107a), or the backward alone with
    ``pair_offsets``; the backward of 16384 tokens in 27 documents a pair table then ``bwd`` on sm_100a (more than 26
    documents), ``bwd_warp_sm_103a`` on sm_103a (15 to 32) and ``bwd_small_sm_107a`` on sm_107a (at most 125); the
    backward of 16385 documents (more than ``bwd_warp_docs``) a pair table then ``bwd_warp`` (``bwd_warp``,
    ``bwd_warp_sm_103a``, ``bwd_warp_sm_107a``); a call of the ``bwd_warp`` role looks up no other backward bound; the
    autograd entries launch no pair table in the backward and the same backward role.  The pair table of at most
    ``small_docs`` (512) documents is ``pair_table_small``, of 513 ``pair_table``, in the forward and in the backward.
    A call looks up the device's architecture, ``fuse_docs``, ``small_docs``, ``bwd_warp_region``, ``bwd_max_docs`` and
    ``bwd_small_docs`` at most once each and passes the device index and the architecture to its launches.  No call
    writes its inputs."""
    launched = []
    entries = []
    passed = []
    lookups = []
    original = cake_jit.run_module
    entry_of = cake_jit._entry

    def spy(role, values, *, rows, device_index, **options):
        launched.append(role)
        passed.append((device_index, options.get("arch")))
        return original(role, values, rows=rows, device_index=device_index, **options)

    def entry_spy(name, arch):
        entries.append((name, arch))
        return entry_of(name, arch)

    def counted(name, function):
        def lookup(*args):
            lookups.append(name)
            return function(*args)

        return lookup

    arch = cake_jit.device_arch(device.index)
    monkeypatch.setattr(cake_jit, "run_module", spy)
    monkeypatch.setattr(cake_jit, "_entry", entry_spy)
    lookup_arch = counted("device_arch", cake_jit.device_arch)
    for module in (cake_jit, cake_backend):
        monkeypatch.setattr(module, "device_arch", lookup_arch)
    for name in ("fuse_docs", "small_docs", "bwd_small_docs", "bwd_max_docs", "bwd_warp_region"):
        monkeypatch.setattr(cake_jit, name, counted(name, getattr(cake_jit, name)))
    forward = FORWARD_RECORDS[arch]
    unfused = UNFUSED_RECORDS[arch]
    # the backward of at most bwd_small_docs documents: the bwd_small role
    small_role = "bwd_small"
    small = "bwd_small_sm_107a" if arch == "sm_107a" else "bwd_small"
    assert cake_jit.fuse_docs(arch) == FUSE_DOCS and cake_jit.small_docs(arch) == SMALL_DOCS
    assert cake_jit.bwd_small_docs(arch) == BWD_SMALL_DOCS[arch]
    assert cake_jit.bwd_max_docs(arch) == BWD_MAX_DOCS[arch]
    assert cake_jit.bwd_warp_region(arch) == BWD_WARP_REGIONS[arch]
    roles = ["device_arch", "bwd_warp_region", "bwd_max_docs", "bwd_small_docs"]

    def lookups_of(role: str) -> list[str]:
        """The lookups of a backward call of ``role``: a ``bwd_warp`` call needs the architecture and the warp
        region only."""
        return roles[:2] if role == "bwd_warp" else roles

    # 256 documents: the forward's fused bound (no pair table before it)
    many = make_inputs(Pack((1, 3, 5, 7) * 64), seed=41, device=device, layout="halves")
    launched.clear()
    entries.clear()
    passed.clear()
    lookups.clear()
    pkg.cake_gated_kv_compression_det_train_forward(many.c, many.z, many.cu, many.G)
    assert (many.S, launched, entries) == (FUSE_DOCS, ["fwd"], [(forward, arch)])
    # one call looks up the architecture and the forward's bound once and passes the device and the architecture down
    assert lookups == ["device_arch", "fuse_docs"] and passed == [(device.index, arch)]
    # 257, 300, 512, 513 and 8193 documents: past the forward's fused bound, on either side of small_docs, and far past
    for lengths, table in (
        ((1, 3, 5, 7) * 64 + (2,), "pair_table_small"),
        ((1, 3, 5, 7) * 75, "pair_table_small"),
        ((1, 3, 5, 7) * 128, "pair_table_small"),
        ((1, 3, 5, 7) * 128 + (2,), "pair_table"),
        ((1, 3) * 4096 + (2,), "pair_table"),
    ):
        many = make_inputs(Pack(lengths), seed=41, device=device, layout="halves")
        launched.clear()
        entries.clear()
        passed.clear()
        lookups.clear()
        pkg.cake_gated_kv_compression_det_train_forward(many.c, many.z, many.cu, many.G)
        assert launched == [table, "fwd_unfused"], many.S
        assert entries == [(table, arch), (unfused, arch)], many.S
        assert lookups == ["device_arch", "fuse_docs", "small_docs"], many.S
        assert passed == [(device.index, arch)] * 2, many.S
        launched.clear()
        entries.clear()
        passed.clear()
        lookups.clear()
        pkg.cake_gated_kv_compression_det_train_backward(many.dout, many.c, many.z, many.cu)
        many_role = _bwd_role(arch, many.S)
        assert many_role == "bwd", many.S
        assert launched == [table, many_role], many.S
        assert entries == [(table, arch), (BACKWARD_RECORDS[arch][many_role], arch)], many.S
        assert lookups == [*lookups_of(many_role), "small_docs"], many.S
        assert passed == [(device.index, arch)] * 2, many.S
    launched.clear()
    entries.clear()
    x = make_inputs(MIXED, seed=37, device=device, layout="halves")
    before = [t.clone() for t in (x.cz, x.cu, x.dout)]
    pkg.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
    assert launched == ["fwd"]
    assert entries == [(forward, arch)]
    # four documents: at most bwd_small_docs
    launched.clear()
    entries.clear()
    lookups.clear()
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
    assert launched == ["pair_table_small", small_role]
    assert entries == [("pair_table_small", arch), (small, arch)]
    assert lookups == [*roles, "small_docs"]
    table = pair_table_of(MIXED.lengths, device)
    launched.clear()
    entries.clear()
    passed.clear()
    lookups.clear()
    pkg.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, pair_offsets=table)
    assert launched == [small_role] and entries == [(small, arch)]
    assert lookups == roles and passed == [(device.index, arch)]
    for entry, args in (
        (pkg.cake_gated_kv_compression_det_cz, lambda cz: (cz, x.cu, x.G)),
        (pkg.cake_gated_kv_compression_det, lambda cz: (cz[:, :WIDTH], cz[:, WIDTH:], x.cu, x.G)),
    ):
        launched.clear()
        cz = x.cz.detach().clone().requires_grad_(True)
        entry(*args(cz)).backward(x.dout)
        assert launched == ["fwd", small_role], entry.__name__
    for old, new in zip(before, (x.cz, x.cu, x.dout), strict=True):
        assert torch.equal(old, new), "a call wrote its input"
    # 16384 tokens in 27 documents: bwd on sm_100a, bwd_warp on sm_103a, bwd_small on sm_107a
    issue = make_inputs(PERF_PACKS[0], seed=43, device=device, layout="halves")
    assert (issue.T, issue.S) == (16384, 27)
    issue_role = {"sm_100a": "bwd", "sm_103a": "bwd_warp", "sm_107a": small_role}[arch]
    assert issue_role == _bwd_role(arch, issue.S)
    issue_record = BACKWARD_RECORDS[arch][issue_role]
    launched.clear()
    entries.clear()
    passed.clear()
    lookups.clear()
    pkg.cake_gated_kv_compression_det_train_backward(issue.dout, issue.c, issue.z, issue.cu)
    assert launched == ["pair_table_small", issue_role]
    assert entries == [("pair_table_small", arch), (issue_record, arch)]
    assert lookups == [*lookups_of(issue_role), "small_docs"]
    assert passed == [(device.index, arch)] * 2
    for entry, args in (
        (pkg.cake_gated_kv_compression_det_cz, lambda cz: (cz, issue.cu, issue.G)),
        (pkg.cake_gated_kv_compression_det, lambda cz: (cz[:, :WIDTH], cz[:, WIDTH:], issue.cu, issue.G)),
    ):
        launched.clear()
        entries.clear()
        cz = issue.cz.detach().clone().requires_grad_(True)
        entry(*args(cz)).backward(issue.dout)
        assert launched == ["fwd", issue_role], entry.__name__
        assert entries == [(forward, arch), (issue_record, arch)], entry.__name__
    # 16385 documents of 32770 tokens: past bwd_warp_docs, bwd_warp on the three architectures; the backward's role
    # needs the architecture and bwd_warp_region only
    warp = BACKWARD_RECORDS[arch]["bwd_warp"]
    past_pack = Pack((1, 3) * 8192 + (2,))
    past = make_inputs(past_pack, seed=47, device=device, layout="halves")
    assert (past.T, past.S) == (32770, BWD_WARP_DOCS[arch] + 1)
    past_table = pair_table_of(past_pack.lengths, device)
    launched.clear()
    entries.clear()
    passed.clear()
    lookups.clear()
    pkg.cake_gated_kv_compression_det_train_backward(past.dout, past.c, past.z, past.cu, pair_offsets=past_table)
    assert launched == ["bwd_warp"] and entries == [(warp, arch)]
    assert lookups == ["device_arch", "bwd_warp_region"] and passed == [(device.index, arch)]
    launched.clear()
    entries.clear()
    passed.clear()
    lookups.clear()
    pkg.cake_gated_kv_compression_det_train_backward(past.dout, past.c, past.z, past.cu)
    assert launched == ["pair_table", "bwd_warp"]
    assert entries == [("pair_table", arch), (warp, arch)]
    assert lookups == ["device_arch", "bwd_warp_region", "small_docs"]
    assert passed == [(device.index, arch)] * 2
    for entry, args in (
        (pkg.cake_gated_kv_compression_det_cz, lambda cz: (cz, past.cu, past.G)),
        (pkg.cake_gated_kv_compression_det, lambda cz: (cz[:, :WIDTH], cz[:, WIDTH:], past.cu, past.G)),
    ):
        launched.clear()
        entries.clear()
        cz = past.cz.detach().clone().requires_grad_(True)
        entry(*args(cz)).backward(past.dout)
        assert launched == ["pair_table", "fwd_unfused", "bwd_warp"], entry.__name__
        assert entries == [("pair_table", arch), (unfused, arch), (warp, arch)], entry.__name__
