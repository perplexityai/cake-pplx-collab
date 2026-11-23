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

The pre-handoff records and routes of the two mHC training registries, on the host rules alone (no device).

The handoff routes (``HANDOFF_KINDS``) are the standard routes with one swap each, so a handoff call launches the
programs a standard call launches, with the same grids (on sm_107a the collapse's own grid, below), PDL arguments
and workspaces:

* ``fwd_prepare_handoff`` is ``fwd_prepare`` with the collapse ``fwd_collapse__pi_pn`` (it reads ``pre_in`` in place
  of the coefficient pass's ``pre`` and writes ``pre_next``) in place of ``fwd_collapse``;
* ``bwd_finish_handoff`` is ``bwd_finish`` with each token-owner form ``bwd_token__<form>`` replaced by its
  ``bwd_token__<form>_ho`` twin (the base record with ``pre_in``, ``dpre_next`` and ``dpre_in`` after the parameter
  record, before ``T``; its launch, dynamic shared memory included, is the base record's: ``dpre_in`` is one FP32
  word per stream, reduced in the kernel's own order);
* ``collapse_forward`` / ``collapse_backward`` run the records ``collapse_fwd`` / ``collapse_bwd__vjpwa_pf0`` on
  sm_100a / sm_103a, with their own library stems and no workspace (on sm_100a / sm_103a ``collapse_backward`` is a
  token-count band of six records on sm_100a and of eight on sm_103a, ``collapse_bwd__vjpwa_pf0`` from 1024 tokens to
  the stage-free one-CTA ``collapse_bwd__vjpwd`` from 6144 / 8192 tokens; on sm_107a
  ``collapse_forward`` is a token-count band of three records and ``collapse_backward`` one of four, below); the forward takes the per-call PDL attribute on its own
  edge ``K4->C`` (no band: on), the VJP none; the VJP is the coalesced form: its dynamic shared memory stages each
  warp's slab of the ``dU`` row and of the four ``X`` rows (6400 bytes a warp, 8 warps) beside its 256 bytes of partial
  sums.

The routes exist at hidden 5120 on sm_100a, sm_103a and sm_107a (each token-owner twin on the architectures of its
base form: sm_107a's backward runs the ``_iu`` forms of ``ao22_rae7_pd1_ntr`` and ``dot_ao8_l2l`` (the idle-lane
unit quotient, the same bits) where the others run ``ao22_rae7_pd1_ntr``, ``dot_ao8`` and its group-row form);
the nondeterministic records are the deterministic ones under the nondeterministic prefix.

On sm_107a the three collapse records are that architecture's own forms (:data:`COLLAPSE_RECORDS`), each the
sm_100a / sm_103a record under its own name, on sm_107a only, with the same launch facts but the grid:
``fwd_collapse__m2_pi_pn`` and ``collapse_fwd__m2`` take two sub-tokens per thread group and trip
(``rows_per_cta`` 4: ``ceil(5 T / 4)`` CTAs in place of ``ceil(5 T / 2)``), and ``collapse_bwd__vjpwa_pfq2`` (the
VJP staged by ``cp.async`` with an L2 prefetch of a later token) keeps the grid ``T`` and the dynamic shared memory.
The base collapse records ship on sm_100a and sm_103a only.  On sm_107a ``collapse_forward`` holds three records chosen
by the token count alone (``rows_between``, :data:`COLLAPSE_FWD_BAND`): ``collapse_fwd__m2`` up to 2047 tokens,
``collapse_fwd__m2_xhnone_rv`` from 2048 to 6655 tokens (:data:`RV_BAND`) and ``collapse_fwd__m2_xhnone`` from
:data:`BAND_T` = 6656 tokens.  ``collapse_fwd__m2_xhnone`` is the ``collapse_fwd__m2`` record under its own name (the
same form reading ``X`` without the L2 evict-first policy) and ``collapse_fwd__m2_xhnone_rv`` the
``collapse_fwd__m2_xhnone`` record under its own name (its CTAs take the trips in reverse launch order: the same
outputs bit for bit), each with every other fact the same.  On sm_107a ``collapse_backward`` holds four records chosen
by the token count alone (``rows_between``, :data:`COLLAPSE_BWD_BAND`, edges :data:`BWD_EDGES`):
``collapse_bwd__vjpwa_sp4_mb4`` up to 215 tokens, ``collapse_bwd__vjpwa_sp2_mb4`` from 216 to 383 tokens,
``collapse_bwd__vjpwa`` from 384 to 640 tokens and ``collapse_bwd__vjpwa_pfq2`` from 641 tokens.
``collapse_bwd__vjpwa`` is the ``collapse_bwd__vjpwa_pfq2`` record under its own name (the same VJP without the L2
prefetch) with every other fact the same; ``collapse_bwd__vjpwa_sp4_mb4`` and ``collapse_bwd__vjpwa_sp2_mb4`` are the
``collapse_bwd__vjpwa`` record under their own names split over four / two CTAs a token (:data:`COLLAPSE_BWD_SPLIT`: a
grid of 4 T / 2 T CTAs, each on a quarter / half of the streams, with :func:`vjp_smem` bytes of dynamic shared memory;
the same outputs bit for bit), every other fact but the architectures the same.  On sm_100a ``collapse_backward``
holds six records chosen by the token count alone (``rows_between``, :data:`COLLAPSE_BWD_BAND`, edges
:data:`BWD_EDGES`): the ``collapse_bwd__vjpwa_sp4_mb4`` library from 1 to 30 and from 38 to 215 tokens under the record
names ``collapse_bwd__vjpwa_sp4_mb4__h5120__le30`` / ``collapse_bwd__vjpwa_sp4_mb4__h5120__gt37_le215``, the
``collapse_bwd__vjpwd_sp4_mb4`` library from 31 to 37 tokens as ``collapse_bwd__vjpwd_sp4_mb4`` itself, the
``collapse_bwd__vjpwa_sp2_mb4`` library from 216 to 1023 tokens under the record name
``collapse_bwd__vjpwa_sp2_mb4__h5120__le1023`` (:data:`COLLAPSE_BWD_ALIAS`: the same library and kernel, every fact
but the architectures and the select rule the same), ``collapse_bwd__vjpwa_pf0`` from 1024 to 6143 tokens and
``collapse_bwd__vjpwd`` from 6144 tokens, each as itself.  On sm_103a it holds eight: the
``collapse_bwd__vjpwa_sp4_mb4`` library from 1 to 30 and from 39 to 215 tokens under the record names
``collapse_bwd__vjpwa_sp4_mb4__h5120__le30`` / ``collapse_bwd__vjpwa_sp4_mb4__h5120__gt38_le215``, the
``collapse_bwd__vjpwd_sp4_mb4`` library from 31 to 38 tokens under the record name
``collapse_bwd__vjpwd_sp4_mb4__h5120__gt30_le38``, the ``collapse_bwd__vjpwa_sp2_mb4`` library from 216 to 247 and from
320 to 1023 tokens under the record names ``collapse_bwd__vjpwa_sp2_mb4__h5120__le247`` /
``collapse_bwd__vjpwa_sp2_mb4__h5120__gt319_le1023``, the ``collapse_bwd__vjpwa_pf0`` library from 248 to 319 and from
1024 to 8191 tokens under the record names ``collapse_bwd__vjpwa_pf0__h5120__gt247_le319`` /
``collapse_bwd__vjpwa_pf0__h5120__gt1023_le8191``, and the ``collapse_bwd__vjpwd`` library from 8192 tokens under the
record name ``collapse_bwd__vjpwd__h5120__gt8191``.  ``collapse_bwd__vjpwa_pf0`` (the architecture's VJP of
:data:`COLLAPSE_RECORDS`: the VJP staged by ``cp.async`` after an L2 prefetch of its own token's rows) is sm_107a's
``collapse_bwd__vjpwa_pfq2`` under its own name, every other fact but the architectures and the select rule the same.
The split records are the band's first one-CTA record split the same way (the same outputs bit for bit); the
stage-free records (:data:`COLLAPSE_BWD_DIRECT`: each warp reads its ``dU`` and ``X`` words from global memory into
registers; :data:`VJP_PART_SMEM` bytes of dynamic shared memory hold the partial sums alone) are the four-CTA split
``collapse_bwd__vjpwd_sp4_mb4`` and the one-CTA ``collapse_bwd__vjpwd``, with the same outputs bit for bit.
"""

from __future__ import annotations

import copy
import itertools
import math

import pytest
from tests.cake.test_helpers.cake_package import cake_package

DET = cake_package("cake.cake_mhc_det_train", "cake_jit")
ND = cake_package("cake.cake_mhc_nondet_train", "cake_jit")
REGISTRIES = {"det": DET, "nondet": ND}
PREFIX = {"det": "cake_mhc_det_train_", "nondet": "cake_mhc_train_"}
HIDDEN = 5120
HANDOFF_KINDS = (
    "fwd_prepare_handoff",
    "bwd_finish_handoff",
    "collapse_forward",
    "collapse_backward",
)
HANDOFF_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# The collapse records of the handoff routes per architecture: (sublayer collapse of fwd_prepare_handoff, standalone
# collapse of collapse_forward, its VJP of collapse_backward), record names without the prefix and the hidden suffix.
COLLAPSE_RECORDS = {
    "sm_100a": ("fwd_collapse__pi_pn", "collapse_fwd", "collapse_bwd__vjpwa_pf0"),
    "sm_103a": ("fwd_collapse__pi_pn", "collapse_fwd", "collapse_bwd__vjpwa_pf0"),
    "sm_107a": (
        "fwd_collapse__m2_pi_pn",
        "collapse_fwd__m2",
        "collapse_bwd__vjpwa_pfq2",
    ),
}
# The architectures of each collapse record.
COLLAPSE_RECORD_ARCHES = {
    "fwd_collapse__pi_pn": ["sm_100a", "sm_103a"],
    "collapse_fwd": ["sm_100a", "sm_103a"],
    "collapse_bwd__vjpwa_pf0": ["sm_100a"],
    "fwd_collapse__m2_pi_pn": ["sm_107a"],
    "collapse_fwd__m2": ["sm_107a"],
    "collapse_fwd__m2_xhnone": ["sm_107a"],
    "collapse_bwd__vjpwa_pfq2": ["sm_107a"],
    "collapse_bwd__vjpwa": ["sm_107a"],
    "collapse_bwd__vjpwa_sp4_mb4": ["sm_107a"],
    "collapse_bwd__vjpwa_sp2_mb4": ["sm_107a"],
    "collapse_fwd__m2_xhnone_rv": ["sm_107a"],
    "collapse_bwd__vjpwd_sp4_mb4": ["sm_100a"],
    "collapse_bwd__vjpwd": ["sm_100a"],
}
# The token-count band of the standalone collapse forward per architecture, in route order: (record without the prefix
# and the hidden suffix, above, at_most) of its ``rows_between`` select rule.  An architecture without an entry runs
# its one record of COLLAPSE_RECORDS at every token count.
COLLAPSE_FWD_BAND = {
    "sm_107a": (
        ("collapse_fwd__m2", 0, 2047),
        ("collapse_fwd__m2_xhnone_rv", 2047, 6655),
        ("collapse_fwd__m2_xhnone", 6655, None),
    ),
}
# The first token count of sm_107a's ``collapse_fwd__m2_xhnone``.
BAND_T = 6656
# The first and the last token count of sm_107a's ``collapse_fwd__m2_xhnone_rv``.
RV_BAND = (2048, 6655)
# The token-count band of the standalone collapse VJP per architecture, in route order: (record without the prefix,
# with the hidden suffix, which an alias's tag follows; above, at_most) of its ``rows_between`` select rule.  Every
# handoff architecture has one; the architecture's VJP of COLLAPSE_RECORDS (or an alias of it) sits at BWD_VJP_AT.
COLLAPSE_BWD_BAND = {
    "sm_100a": (
        ("collapse_bwd__vjpwa_sp4_mb4__h5120__le30", 0, 30),
        ("collapse_bwd__vjpwd_sp4_mb4__h5120", 30, 37),
        ("collapse_bwd__vjpwa_sp4_mb4__h5120__gt37_le215", 37, 215),
        ("collapse_bwd__vjpwa_sp2_mb4__h5120__le1023", 215, 1023),
        ("collapse_bwd__vjpwa_pf0__h5120", 1023, 6143),
        ("collapse_bwd__vjpwd__h5120", 6143, None),
    ),
    "sm_103a": (
        ("collapse_bwd__vjpwa_sp4_mb4__h5120__le30", 0, 30),
        ("collapse_bwd__vjpwd_sp4_mb4__h5120__gt30_le38", 30, 38),
        ("collapse_bwd__vjpwa_sp4_mb4__h5120__gt38_le215", 38, 215),
        ("collapse_bwd__vjpwa_sp2_mb4__h5120__le247", 215, 247),
        ("collapse_bwd__vjpwa_pf0__h5120__gt247_le319", 247, 319),
        ("collapse_bwd__vjpwa_sp2_mb4__h5120__gt319_le1023", 319, 1023),
        ("collapse_bwd__vjpwa_pf0__h5120__gt1023_le8191", 1023, 8191),
        ("collapse_bwd__vjpwd__h5120__gt8191", 8191, None),
    ),
    "sm_107a": (
        ("collapse_bwd__vjpwa_sp4_mb4__h5120", 0, 215),
        ("collapse_bwd__vjpwa_sp2_mb4__h5120", 215, 383),
        ("collapse_bwd__vjpwa__h5120", 383, 640),
        ("collapse_bwd__vjpwa_pfq2__h5120", 640, None),
    ),
}
# The last token count of each record of ``collapse_backward`` but the last, per architecture.
BWD_EDGES = {
    "sm_100a": (30, 37, 215, 1023, 6143),
    "sm_103a": (30, 38, 215, 247, 319, 1023, 8191),
    "sm_107a": (215, 383, 640),
}
# The CTAs a token of each record of each architecture's band, in route order: 4 / 2 for the split records (grid
# k T), 1 for the one-CTA-a-token records.
BWD_SPLITS = {
    "sm_100a": (4, 4, 4, 2, 1, 1),
    "sm_103a": (4, 4, 4, 2, 1, 2, 1, 1),
    "sm_107a": (4, 2, 1, 1),
}
# The one-CTA-a-token record of each architecture's band that its split records split over k CTAs a token with the
# same outputs bit for bit: the first one-CTA record of the band (on sm_103a the ``collapse_bwd__vjpwa_pf0`` alias of
# 248 to 319 tokens).
BWD_ONE_CTA = {
    "sm_100a": "collapse_bwd__vjpwa_pf0__h5120",
    "sm_103a": "collapse_bwd__vjpwa_pf0__h5120__gt247_le319",
    "sm_107a": "collapse_bwd__vjpwa__h5120",
}
# The position in each architecture's band of the architecture's VJP of COLLAPSE_RECORDS (itself or an alias of it):
# the first one-CTA record on sm_100a / sm_103a, the last record on sm_107a.
BWD_VJP_AT = {"sm_100a": 4, "sm_103a": 4, "sm_107a": 3}
# The aliases of the VJP's bands: record -> the record whose library, kernel and every other fact but the
# architectures and the select rule it carries (sm_100a runs the ``_vjpwa_sp2_mb4`` library from 216 to 1023 tokens
# under one alias, sm_103a from 216 to 247 and from 320 to 1023 tokens under two; the ``_vjpwa_sp2_mb4`` record keeps
# sm_107a's band from 216 to 383 tokens.  Up to 215 tokens sm_100a and sm_103a run the ``_vjpwa_sp4_mb4`` library
# under three aliases, its record keeping sm_107a's band up to 215 tokens, and the ``_vjpwd_sp4_mb4`` library under
# its own record (sm_100a, 31 to 37 tokens) and one alias (sm_103a, 31 to 38 tokens).  sm_100a runs the
# ``_vjpwa_pf0`` and ``_vjpwd`` libraries under their own records, sm_103a under three aliases (``_vjpwa_pf0`` from
# 248 to 319 and from 1024 to 8191 tokens, ``_vjpwd`` from 8192 tokens)).
COLLAPSE_BWD_ALIAS = {
    "collapse_bwd__vjpwa_sp2_mb4__h5120__le1023": "collapse_bwd__vjpwa_sp2_mb4__h5120",
    "collapse_bwd__vjpwa_sp2_mb4__h5120__le247": "collapse_bwd__vjpwa_sp2_mb4__h5120",
    "collapse_bwd__vjpwa_sp2_mb4__h5120__gt319_le1023": "collapse_bwd__vjpwa_sp2_mb4__h5120",
    "collapse_bwd__vjpwa_sp4_mb4__h5120__le30": "collapse_bwd__vjpwa_sp4_mb4__h5120",
    "collapse_bwd__vjpwa_sp4_mb4__h5120__gt37_le215": "collapse_bwd__vjpwa_sp4_mb4__h5120",
    "collapse_bwd__vjpwa_sp4_mb4__h5120__gt38_le215": "collapse_bwd__vjpwa_sp4_mb4__h5120",
    "collapse_bwd__vjpwd_sp4_mb4__h5120__gt30_le38": "collapse_bwd__vjpwd_sp4_mb4__h5120",
    "collapse_bwd__vjpwa_pf0__h5120__gt247_le319": "collapse_bwd__vjpwa_pf0__h5120",
    "collapse_bwd__vjpwa_pf0__h5120__gt1023_le8191": "collapse_bwd__vjpwa_pf0__h5120",
    "collapse_bwd__vjpwd__h5120__gt8191": "collapse_bwd__vjpwd__h5120",
}
# The architectures of each alias record.
COLLAPSE_BWD_ALIAS_ARCHES = {
    "collapse_bwd__vjpwa_sp2_mb4__h5120__le1023": ["sm_100a"],
    "collapse_bwd__vjpwa_sp2_mb4__h5120__le247": ["sm_103a"],
    "collapse_bwd__vjpwa_sp2_mb4__h5120__gt319_le1023": ["sm_103a"],
    "collapse_bwd__vjpwa_sp4_mb4__h5120__le30": ["sm_100a", "sm_103a"],
    "collapse_bwd__vjpwa_sp4_mb4__h5120__gt37_le215": ["sm_100a"],
    "collapse_bwd__vjpwa_sp4_mb4__h5120__gt38_le215": ["sm_103a"],
    "collapse_bwd__vjpwd_sp4_mb4__h5120__gt30_le38": ["sm_103a"],
    "collapse_bwd__vjpwa_pf0__h5120__gt247_le319": ["sm_103a"],
    "collapse_bwd__vjpwa_pf0__h5120__gt1023_le8191": ["sm_103a"],
    "collapse_bwd__vjpwd__h5120__gt8191": ["sm_103a"],
}
# The split VJP records of the bands: record -> k, the CTAs a token (grid k T, each CTA on 4 / k streams).
COLLAPSE_BWD_SPLIT = {
    "collapse_bwd__vjpwa_sp4_mb4__h5120": 4,
    "collapse_bwd__vjpwa_sp2_mb4__h5120": 2,
    "collapse_bwd__vjpwa_sp2_mb4__h5120__le1023": 2,
    "collapse_bwd__vjpwa_sp2_mb4__h5120__le247": 2,
    "collapse_bwd__vjpwa_sp2_mb4__h5120__gt319_le1023": 2,
    "collapse_bwd__vjpwa_sp4_mb4__h5120__le30": 4,
    "collapse_bwd__vjpwa_sp4_mb4__h5120__gt37_le215": 4,
    "collapse_bwd__vjpwa_sp4_mb4__h5120__gt38_le215": 4,
    "collapse_bwd__vjpwd_sp4_mb4__h5120": 4,
    "collapse_bwd__vjpwd_sp4_mb4__h5120__gt30_le38": 4,
}
# The VJP records without the shared-memory stage (``_vjpwd``: each warp reads its ``dU`` and ``X`` words from global
# memory into registers), split over four CTAs a token or one CTA a token: VJP_PART_SMEM bytes of dynamic shared
# memory, the partial sums alone.
COLLAPSE_BWD_DIRECT = frozenset(
    {
        "collapse_bwd__vjpwd_sp4_mb4__h5120",
        "collapse_bwd__vjpwd_sp4_mb4__h5120__gt30_le38",
        "collapse_bwd__vjpwd__h5120",
        "collapse_bwd__vjpwd__h5120__gt8191",
    }
)
# The token counts of the stage-free split VJP per architecture (every other token count up to 215 runs the
# ``_vjpwa_sp4_mb4`` library).
BWD_DIRECT_ROWS = {
    "sm_100a": tuple(range(31, 38)),
    "sm_103a": tuple(range(31, 39)),
}
# Sub-tokens (token column parts) per CTA of the two collapse forwards' grids: ceil(5 T / rows_per_cta) CTAs.
COLLAPSE_ROWS_PER_CTA = {"sm_100a": 2, "sm_103a": 2, "sm_107a": 4}
BOARDS = (("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 216))
TOKEN_FORMS = (
    "ao22_rae7_pd1_ntr",
    "dot_ao8",
    "dot_ao8_trows_dyn",
    "rt64",
    "ao22_rae7_pd1_ntr_iu",
    "dot_ao8_l2l_iu",
)
# The standalone VJP's dynamic shared memory: 8 warps x 6400 bytes of staged dU / X words + 256 bytes of partial sums.
VJP_SMEM = 8 * 6400 + 256


def vjp_smem(k: int) -> int:
    """The dynamic shared memory of the VJP split over ``k`` CTAs a token: 8 warps x (4 / k + 1) staged rows of 1280
    bytes (the ``dU`` row and the CTA's ``X`` rows) + 256 bytes of partial sums (``k`` 1: :data:`VJP_SMEM`)."""
    return 8 * (4 // k + 1) * 1280 + 256


# The dynamic shared memory of the VJPs without the stage (COLLAPSE_BWD_DIRECT): the 256 bytes of partial sums.
VJP_PART_SMEM = 256
HANDOFF_OPERANDS = (
    ["buffer", "pre_in"],
    ["buffer", "dpre_next"],
    ["buffer", "dpre_in"],
)
ROWS = (
    1,
    2,
    3,
    7,
    30,
    31,
    37,
    38,
    39,
    53,
    54,
    55,
    64,
    65,
    74,
    75,
    77,
    78,
    215,
    216,
    247,
    248,
    255,
    256,
    319,
    320,
    383,
    384,
    460,
    461,
    640,
    641,
    656,
    657,
    768,
    1000,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4096,
    4097,
    6143,
    6144,
    6145,
    6655,
    6656,
    6657,
    8191,
    8192,
    8193,
    12000,
    16384,
    20001,
    32768,
    65536,
    131072,
)
NAME_KEYS = ("kernel", "kernel_symbol", "library")


def _rec(mode: str, suffix: str) -> dict:
    return REGISTRIES[mode].MODULES[f"{PREFIX[mode]}{suffix}"]


def _strip(record: dict, *keys: str) -> dict:
    return {k: v for k, v in record.items() if k not in (*NAME_KEYS, *keys)}


def _collapse(mode: str, arch: str) -> tuple[str, str, str]:
    return tuple(f"{PREFIX[mode]}{s}__h{HIDDEN}" for s in COLLAPSE_RECORDS[arch])


def _collapse_forward_route(mode: str, arch: str) -> list[str]:
    """The records of ``collapse_forward`` on ``arch``, in route order."""
    if arch in COLLAPSE_FWD_BAND:
        return [f"{PREFIX[mode]}{s}__h{HIDDEN}" for s, _, _ in COLLAPSE_FWD_BAND[arch]]
    return [_collapse(mode, arch)[1]]


def _collapse_forward(mode: str, arch: str, rows: int) -> str:
    """The record ``collapse_forward`` launches at ``rows`` tokens on ``arch``."""
    if arch in COLLAPSE_FWD_BAND:
        names = _collapse_forward_route(mode, arch)
        band = COLLAPSE_FWD_BAND[arch]
        hits = [
            n
            for n, (_, above, at_most) in zip(names, band)
            if above < rows and (at_most is None or rows <= at_most)
        ]
        assert len(hits) == 1, (rows, hits)
        return hits[0]
    return _collapse(mode, arch)[1]


def _collapse_backward_route(mode: str, arch: str) -> list[str]:
    """The records of ``collapse_backward`` on ``arch``, in route order."""
    if arch in COLLAPSE_BWD_BAND:
        return [f"{PREFIX[mode]}{s}" for s, _, _ in COLLAPSE_BWD_BAND[arch]]
    return [_collapse(mode, arch)[2]]


def _collapse_backward(mode: str, arch: str, rows: int) -> str:
    """The record ``collapse_backward`` launches at ``rows`` tokens on ``arch``."""
    if arch in COLLAPSE_BWD_BAND:
        names = _collapse_backward_route(mode, arch)
        band = COLLAPSE_BWD_BAND[arch]
        hits = [
            n
            for n, (_, above, at_most) in zip(names, band)
            if above < rows and (at_most is None or rows <= at_most)
        ]
        assert len(hits) == 1, (rows, hits)
        return hits[0]
    return _collapse(mode, arch)[2]


def _swaps(mode: str, arch: str) -> dict[str, str]:
    p = PREFIX[mode]
    out = {f"{p}fwd_collapse__h{HIDDEN}": _collapse(mode, arch)[0]}
    out.update(
        {
            f"{p}bwd_token__{f}__h{HIDDEN}": f"{p}bwd_token__{f}_ho__h{HIDDEN}"
            for f in TOKEN_FORMS
        }
    )
    return out


def _selected(
    mode: str, arch: str, kind: str, rows: int, sm: int, iters: int
) -> tuple[str, ...]:
    return REGISTRIES[mode].selected_modules(
        arch, HIDDEN, kind, rows=rows, sm_count=sm, sinkhorn_iters=iters
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
def test_handoff_kinds_keys_and_sizes(mode: str) -> None:
    reg = REGISTRIES[mode]
    assert tuple(reg.HANDOFF_KINDS) == HANDOFF_KINDS
    assert not set(reg.HANDOFF_KINDS) & set(reg.KINDS)
    for kind in HANDOFF_KINDS:
        assert reg.route_key(HIDDEN, kind) == f"{kind}_h{HIDDEN}"
    assert reg.workspace_kind("fwd_prepare_handoff") == reg.workspace_kind(
        "fwd_prepare"
    )
    assert reg.workspace_kind("bwd_finish_handoff") == reg.workspace_kind("bwd_finish")
    assert reg.workspace_kind("collapse_forward") is None
    assert reg.workspace_kind("collapse_backward") is None
    for arch in HANDOFF_ARCHES:
        assert tuple(reg.handoff_hidden_sizes(arch)) == (HIDDEN,)
        assert HIDDEN in reg.supported_hidden_sizes(arch)
    for routes in reg.ROUTES.values():
        for key in routes:
            kind, hidden = key.rsplit("_h", 1)
            assert kind in (*reg.KINDS, *HANDOFF_KINDS), key
            if kind in HANDOFF_KINDS:
                assert int(hidden) == HIDDEN, key


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("form", TOKEN_FORMS)
def test_token_owner_twins_are_the_base_records_with_the_handoff_operands(
    mode: str, form: str
) -> None:
    base = _rec(mode, f"bwd_token__{form}__h{HIDDEN}")
    twin = _rec(mode, f"bwd_token__{form}_ho__h{HIDDEN}")
    name = f"{PREFIX[mode]}bwd_token__{form}_ho__h{HIDDEN}"
    assert (twin["kernel"], twin["library"], twin["kernel_symbol"]) == (
        name,
        name,
        f"kernel_{name}",
    )
    assert _strip(twin, "arches", "arg_plan", "logical", "launch") == _strip(
        base, "arches", "arg_plan", "logical", "launch"
    )
    assert twin["launch"] == base["launch"]
    assert twin["arches"] == [a for a in base["arches"] if a in HANDOFF_ARCHES]
    plan = copy.deepcopy(base["arg_plan"])
    at = plan.index(["buffer", "param_partial"]) + 1
    assert plan[at] == ["parameter", "T"]
    plan[at:at] = [list(op) for op in HANDOFF_OPERANDS]
    assert twin["arg_plan"] == plan
    assert twin["logical"] == {
        **base["logical"],
        "pre_in": "pre_in",
        "dpre_next": "dpre_next",
        "dpre_in": "dpre_in",
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("arch", HANDOFF_ARCHES)
def test_collapse_records(mode: str, arch: str) -> None:
    base = _rec(mode, f"fwd_collapse__h{HIDDEN}")
    pi_pn, fwd, bwd = (_rec(mode, f"{s}__h{HIDDEN}") for s in COLLAPSE_RECORDS[arch])
    grid = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
    for suffix, rec in zip(COLLAPSE_RECORDS[arch], (pi_pn, fwd, bwd)):
        name = f"{PREFIX[mode]}{suffix}__h{HIDDEN}"
        assert (rec["kernel"], rec["library"], rec["kernel_symbol"]) == (
            name,
            name,
            f"kernel_{name}",
        )
        assert rec["arches"] == COLLAPSE_RECORD_ARCHES[suffix]
    collapse_grid = {
        "col_split": 5,
        "grid_limit": 0,
        "grid_per_sm": 0,
        "kind": "rows_per_cta",
        "rows_per_cta": COLLAPSE_ROWS_PER_CTA[arch],
    }
    assert base["grid_rule"] == {**collapse_grid, "rows_per_cta": 2}
    # the sublayer collapse: the base record (same launch facts, PDL edge) with pre_in in and pre_next out, on the
    # architecture's grid
    assert _strip(pi_pn, "arches", "arg_plan", "logical", "grid_rule") == _strip(
        base, "arches", "arg_plan", "logical", "grid_rule"
    )
    assert pi_pn["grid_rule"] == collapse_grid
    assert pi_pn["arg_plan"] == [
        ["buffer", "x"],
        ["buffer", "pre_in"],
        ["buffer", "stats"],
        ["buffer", "u"],
        ["buffer", "pre_next"],
        ["parameter", "T"],
        ["launch", "enable_pdl"],
        *grid,
    ]
    assert (
        REGISTRIES[mode].pdl_edge(pi_pn["library"])
        == REGISTRIES[mode].pdl_edge(base["library"])
        is not None
    )
    # the standalone collapse: own library stems, no workspace; the forward takes the per-call PDL attribute on its own
    # edge K4->C (after the last sublayer's residual mixing, no band: on), the VJP none
    assert fwd["arg_plan"] == [
        ["buffer", "x"],
        ["buffer", "pre_in"],
        ["buffer", "u"],
        ["parameter", "T"],
        ["launch", "enable_pdl"],
        *grid,
    ]
    assert bwd["arg_plan"] == [
        ["buffer", "x"],
        ["buffer", "pre_in"],
        ["buffer", "du"],
        ["buffer", "dx"],
        ["buffer", "dpre_in"],
        ["parameter", "T"],
        *grid,
    ]
    assert (fwd["role"], bwd["role"]) == ("collapse_fwd", "collapse_bwd")
    assert fwd["launch"]["use_pdl"] == "runtime" and bwd["launch"]["use_pdl"] is False
    assert REGISTRIES[mode].pdl_edge(fwd["library"]) == "K4->C"
    assert REGISTRIES[mode].pdl_edge(bwd["library"]) is None
    for rec, smem in ((fwd, 0), (bwd, VJP_SMEM)):
        assert rec["workspace_rule"] is None
        assert rec["launch"]["dynamic_smem_bytes"] == smem
        assert tuple(rec["launch"]["block"]) == (256, 1, 1)
    one_token_per_cta = {
        "col_split": 1,
        "grid_limit": 0,
        "grid_per_sm": 0,
        "kind": "rows_per_cta",
        "rows_per_cta": 1,
    }
    # the VJP: on every architecture a token-count band of records (rows_between, tiling every token count at the
    # edges BWD_EDGES), each under its own record name, the architecture's VJP of COLLAPSE_RECORDS (itself or under an
    # alias) at BWD_VJP_AT.  On sm_107a (T <= 215 / 216 <= T <= 383 / 384 <= T <= 640 / T >= 641) the third is the
    # record of the token counts from 641 under its own name and library, with every other fact the same (the same
    # VJP without the L2 prefetch), and the last the VJP; on sm_100a (T <= 30 / 31 <= T <= 37 / 38 <= T <= 215 /
    # 216 <= T <= 1023 / 1024 <= T <= 6143 / T >= 6144) the first three are the sm_107a band's first library under two
    # alias records and the stage-free split VJP (COLLAPSE_BWD_DIRECT) as its own record, the fourth the sm_107a
    # band's second library under an alias record (COLLAPSE_BWD_ALIAS), the fifth the VJP itself and the sixth the
    # stage-free one-CTA VJP (COLLAPSE_BWD_DIRECT) as its own record; on sm_103a (T <= 30 / 31 <= T <= 38 /
    # 39 <= T <= 215 / 216 <= T <= 247 / 248 <= T <= 319 / 320 <= T <= 1023 / 1024 <= T <= 8191 / T >= 8192) the first
    # three are the same two libraries under three alias records, the fourth and the sixth the second library under
    # two alias records, the fifth and the seventh the VJP under two alias records and the eighth the stage-free
    # one-CTA VJP under an alias record.  The split records (BWD_SPLITS) and the stage-free records are the band's
    # first one-CTA record (BWD_ONE_CTA) split over k CTAs a token (k 4 / 2 / 1: grid k T, col_split k) with
    # vjp_smem(k) bytes of dynamic shared memory (VJP_PART_SMEM without the stage), every other fact but the
    # architectures the same
    assert (
        set(COLLAPSE_BWD_BAND) == set(BWD_EDGES) == set(BWD_ONE_CTA) == set(BWD_SPLITS)
    )
    assert set(COLLAPSE_BWD_BAND) == set(HANDOFF_ARCHES) == set(BWD_VJP_AT)
    band, edges = COLLAPSE_BWD_BAND[arch], BWD_EDGES[arch]
    vjp_at = BWD_VJP_AT[arch]
    vjp = COLLAPSE_BWD_ALIAS.get(band[vjp_at][0], band[vjp_at][0])
    assert vjp == f"{COLLAPSE_RECORDS[arch][2]}__h{HIDDEN}"
    one_at = BWD_SPLITS[arch].index(1)
    assert band[one_at][0] == BWD_ONE_CTA[arch]
    assert vjp_at == (len(band) - 1 if arch == "sm_107a" else one_at)
    lowers, uppers = (0, *edges), (*edges, None)
    pairs = [(above, at_most) for _, above, at_most in band]
    assert pairs == list(zip(lowers, uppers))
    for suffix, above, at_most in band:
        rec = _rec(mode, suffix)
        assert rec["select_rule"] == {
            "above": above,
            "at_most": at_most,
            "kind": "rows_between",
        }, suffix
        name = f"{PREFIX[mode]}{COLLAPSE_BWD_ALIAS.get(suffix, suffix)}"
        assert (rec["kernel"], rec["library"], rec["kernel_symbol"]) == (
            name,
            name,
            f"kernel_{name}",
        ), suffix
        if suffix in COLLAPSE_BWD_ALIAS:
            arches = COLLAPSE_BWD_ALIAS_ARCHES[suffix]
        else:
            arches = COLLAPSE_RECORD_ARCHES[suffix.removesuffix(f"__h{HIDDEN}")]
        assert arch in arches and rec["arches"] == arches, suffix
        assert REGISTRIES[mode].pdl_edge(rec["library"]) is None, suffix
    # every record of the band runs its own library but an alias, which runs its record's; neighbours differ
    libs = [_rec(mode, s)["library"] for s, _, _ in band]
    assert len(set(libs)) == len({COLLAPSE_BWD_ALIAS.get(s, s) for s, _, _ in band})
    assert all(a != b for a, b in itertools.pairwise(libs))
    # an alias is its record under its own name, architectures and select rule: every other fact the same; every alias
    # runs on architectures its record does not list
    assert set(COLLAPSE_BWD_ALIAS) == set(COLLAPSE_BWD_ALIAS_ARCHES)
    for alias, own in COLLAPSE_BWD_ALIAS.items():
        a, o = _rec(mode, alias), _rec(mode, own)
        assert {k: a[k] for k in NAME_KEYS} == {k: o[k] for k in NAME_KEYS}, alias
        keys = ("arches", "select_rule")
        assert _strip(a, *keys) == _strip(o, *keys), alias
        assert a["arches"] == COLLAPSE_BWD_ALIAS_ARCHES[alias], alias
        assert not set(a["arches"]) & set(o["arches"]), alias
    one_cta = _rec(mode, band[one_at][0])
    assert one_cta["grid_rule"] == one_token_per_cta
    # the one-CTA record is the architecture's VJP (on sm_103a under an alias: its architectures checked above)
    own_keys = ("arches",) if band[one_at][0] in COLLAPSE_BWD_ALIAS else ()
    assert _strip(one_cta, "select_rule", *own_keys) == _strip(
        bwd, "select_rule", *own_keys
    )
    assert vjp_smem(1) == VJP_SMEM
    assert VJP_SMEM - 8 * 6400 == VJP_PART_SMEM
    splits = BWD_SPLITS[arch]
    # in route order the band's CTAs a token fall from 4 to 2 to 1 (runs of equal counts merged) before anything else
    assert len(splits) == len(band)
    assert [k for k, _ in itertools.groupby(splits)][:3] == [4, 2, 1]
    assert [COLLAPSE_BWD_SPLIT.get(s, 1) for s, _, _ in band] == list(splits)
    assert set(COLLAPSE_BWD_SPLIT) == {
        s
        for a, b in COLLAPSE_BWD_BAND.items()
        for (s, _, _), k in zip(b, BWD_SPLITS[a])
        if k > 1
    }
    # the stage-free VJPs: the four-CTA split and the one-CTA form, each its own library, on sm_100a / sm_103a alone
    direct_splits = {
        "collapse_bwd__vjpwd_sp4_mb4__h5120": 4,
        "collapse_bwd__vjpwd__h5120": 1,
    }
    direct_libraries = {COLLAPSE_BWD_ALIAS.get(s, s) for s in COLLAPSE_BWD_DIRECT}
    assert direct_libraries == set(direct_splits)
    for s in COLLAPSE_BWD_DIRECT:
        k = direct_splits[COLLAPSE_BWD_ALIAS.get(s, s)]
        assert COLLAPSE_BWD_SPLIT.get(s, 1) == k, s
    for (suffix, _, _), k in zip(band, splits):
        rec = _rec(mode, suffix)
        direct = suffix in COLLAPSE_BWD_DIRECT
        assert ("_vjpwd_" in rec["library"]) == direct, suffix
        assert not direct or arch in ("sm_100a", "sm_103a"), suffix
        if k == 1 and not direct:
            continue
        assert COLLAPSE_BWD_SPLIT.get(suffix, 1) == k
        assert rec["grid_rule"] == {**one_token_per_cta, "col_split": k}, suffix
        assert rec["launch"] == {
            **one_cta["launch"],
            "dynamic_smem_bytes": VJP_PART_SMEM if direct else vjp_smem(k),
        }, suffix
        keys = ("arches", "select_rule", "grid_rule", "launch")
        assert _strip(rec, *keys) == _strip(one_cta, *keys), suffix
    # the forward: one record at every token count, or on sm_107a a token-count band of three records (rows_between,
    # T <= 2047 / 2048 <= T <= 6655 / T >= 6656); the record from 6656 is the first under its own name (the same form
    # without the L2 evict-first policy on X) and the record from 2048 to 6655 the one from 6656 under its own name (its
    # CTAs take the trips in reverse launch order), each with every other fact the same
    if arch in COLLAPSE_FWD_BAND:
        band = COLLAPSE_FWD_BAND[arch]
        assert band[0][0] == COLLAPSE_RECORDS[arch][1]
        first, last = RV_BAND
        assert last == BAND_T - 1
        assert [(above, at_most) for _, above, at_most in band] == [
            (0, first - 1),
            (first - 1, last),
            (last, None),
        ]
        for suffix, above, at_most in band:
            assert _rec(mode, f"{suffix}__h{HIDDEN}")["select_rule"] == {
                "above": above,
                "at_most": at_most,
                "kind": "rows_between",
            }, suffix
        assert [s for s, _, _ in band[1:]] == [
            "collapse_fwd__m2_xhnone_rv",
            "collapse_fwd__m2_xhnone",
        ]
        source = fwd
        for twin_suffix in (band[2][0], band[1][0]):
            twin = _rec(mode, f"{twin_suffix}__h{HIDDEN}")
            name = f"{PREFIX[mode]}{twin_suffix}__h{HIDDEN}"
            assert (twin["kernel"], twin["library"], twin["kernel_symbol"]) == (
                name,
                name,
                f"kernel_{name}",
            ), twin_suffix
            assert twin["arches"] == COLLAPSE_RECORD_ARCHES[twin_suffix] == [arch]
            assert _strip(twin, "select_rule") == _strip(source, "select_rule")
            assert REGISTRIES[mode].pdl_edge(twin["library"]) == "K4->C"
            source = twin
        assert len({_rec(mode, f"{s}__h{HIDDEN}")["library"] for s, _, _ in band}) == 3
    else:
        assert fwd["select_rule"] is None
    assert fwd["grid_rule"] == collapse_grid
    assert bwd["grid_rule"] == one_token_per_cta
    for rec in (pi_pn, fwd, bwd):
        assert rec["hidden"] == HIDDEN
        assert set(rec["logical"]) == {
            k for kind, k in rec["arg_plan"] if kind in ("buffer", "parameter")
        }
    # an architecture's own form is the sm_100a / sm_103a record under its own name, on its architectures, with the
    # architecture's grid and select rule (both checked above): every other fact the same
    for own, shared in zip(COLLAPSE_RECORDS[arch], COLLAPSE_RECORDS["sm_100a"]):
        if own != shared:
            a, b = _rec(mode, f"{own}__h{HIDDEN}"), _rec(mode, f"{shared}__h{HIDDEN}")
            assert _strip(a, "arches", "grid_rule", "select_rule") == _strip(
                b, "arches", "grid_rule", "select_rule"
            ), own


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("arch", HANDOFF_ARCHES)
def test_handoff_routes_are_the_standard_routes_with_the_swaps(
    mode: str, arch: str
) -> None:
    reg, p, swaps = REGISTRIES[mode], PREFIX[mode], _swaps(mode, arch)
    routes = reg.ROUTES[arch]
    fwd = routes[f"fwd_prepare_h{HIDDEN}"]
    bwd = routes[f"bwd_finish_h{HIDDEN}"]
    assert routes[f"fwd_prepare_handoff_h{HIDDEN}"] == [swaps.get(n, n) for n in fwd]
    assert routes[f"bwd_finish_handoff_h{HIDDEN}"] == [swaps.get(n, n) for n in bwd]
    # every collapse / token-owner record of the standard routes has its twin (no base form left in a handoff route)
    assert [
        n for n in fwd if REGISTRIES[mode].MODULES[n]["role"] == "fwd_collapse"
    ] == [f"{p}fwd_collapse__h{HIDDEN}"]
    tokens = {n for n in bwd if REGISTRIES[mode].MODULES[n]["role"] == "bwd_token"}
    assert tokens and tokens == {
        f"{p}bwd_token__{f}__h{HIDDEN}"
        for f in TOKEN_FORMS
        if arch in REGISTRIES[mode].MODULES[f"{p}bwd_token__{f}__h{HIDDEN}"]["arches"]
    }
    assert routes[f"collapse_forward_h{HIDDEN}"] == _collapse_forward_route(mode, arch)
    bwd_route = _collapse_backward_route(mode, arch)
    assert routes[f"collapse_backward_h{HIDDEN}"] == bwd_route
    for kind in HANDOFF_KINDS:
        for name in reg.route_modules(arch, HIDDEN, kind):
            assert arch in reg.MODULES[name]["arches"], (kind, name)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
@pytest.mark.parametrize("iters", (20, 7))
def test_a_handoff_call_selects_the_standard_programs_with_the_swaps(
    mode: str, arch: str, sm: int, iters: int
) -> None:
    swaps = _swaps(mode, arch)
    for rows in ROWS:
        for kind, standard in (
            ("fwd_prepare_handoff", "fwd_prepare"),
            ("bwd_finish_handoff", "bwd_finish"),
        ):
            want = tuple(
                swaps.get(n, n)
                for n in _selected(mode, arch, standard, rows, sm, iters)
            )
            assert _selected(mode, arch, kind, rows, sm, iters) == want, (kind, rows)
        for kind, name in (
            ("collapse_forward", _collapse_forward(mode, arch, rows)),
            ("collapse_backward", _collapse_backward(mode, arch, rows)),
        ):
            assert _selected(mode, arch, kind, rows, sm, iters) == (name,), (kind, rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("sm", (208, 212, 216))
def test_the_sm107a_collapse_forward_band_reads_the_token_count_only(
    mode: str, sm: int
) -> None:
    """sm_107a's ``collapse_forward`` takes ``collapse_fwd__m2`` up to 2047 tokens, ``collapse_fwd__m2_xhnone_rv``
    from 2048 to 6655 tokens (:data:`RV_BAND`) and ``collapse_fwd__m2_xhnone`` from :data:`BAND_T` tokens, on every
    sm_107a board and Sinkhorn count (and, in the deterministic registry, along every call path)."""
    low, mid, high = _collapse_forward_route(mode, "sm_107a")
    reg = REGISTRIES[mode]
    paths = [{}]
    if mode == "det":
        paths += [{"path": path} for path in reg.PDL_PATHS]
    first, _ = RV_BAND
    edge = (first - 2, first - 1, first, first + 1)
    edge += (BAND_T - 2, BAND_T - 1, BAND_T, BAND_T + 1)
    for rows in (
        1,
        2,
        1024,
        1920,
        4096,
        6144,
        *edge,
        7168,
        8191,
        8192,
        8193,
        16384,
        65536,
        131072,
    ):
        want = (low if rows < first else mid if rows < BAND_T else high,)
        for path in paths:
            for iters in (20, 7):
                got = reg.selected_modules(
                    "sm_107a",
                    HIDDEN,
                    "collapse_forward",
                    rows=rows,
                    sm_count=sm,
                    sinkhorn_iters=iters,
                    **path,
                )
                assert got == want, (rows, path, iters)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("sm", (208, 212, 216))
def test_the_sm107a_collapse_backward_band_reads_the_token_count_only(
    mode: str, sm: int
) -> None:
    """sm_107a's ``collapse_backward`` takes ``collapse_bwd__vjpwa_sp4_mb4`` up to 215 tokens,
    ``collapse_bwd__vjpwa_sp2_mb4`` from 216 to 383 tokens, ``collapse_bwd__vjpwa`` from 384 to 640 tokens and
    ``collapse_bwd__vjpwa_pfq2`` from 641 tokens (:data:`BWD_EDGES`), on every sm_107a board and Sinkhorn count (and,
    in the deterministic registry, along every call path); both sides of each edge of sm_100a's and sm_103a's band up
    to 215 tokens, and 53, 54, 55, 74, 75, 77 and 78, take ``collapse_bwd__vjpwa_sp4_mb4`` here, and both sides of
    their edges from 1023 tokens (1023 / 1024, 6143 / 6144, 8191 / 8192) ``collapse_bwd__vjpwa_pfq2``."""
    route = _collapse_backward_route(mode, "sm_107a")
    assert BWD_EDGES["sm_107a"] == (215, 383, 640)
    assert len(route) == len(BWD_EDGES["sm_107a"]) + 1
    reg = REGISTRIES[mode]
    paths = [{}]
    if mode == "det":
        paths += [{"path": path} for path in reg.PDL_PATHS]
    edges = tuple(e + d for e in BWD_EDGES["sm_107a"] for d in (-1, 0, 1, 2))
    others = tuple(
        e + d
        for arch in ("sm_100a", "sm_103a")
        for e in BWD_EDGES[arch]
        if e < 215
        for d in (0, 1)
    )
    assert set(others) == {30, 31, 37, 38, 39}
    for rows in (
        1,
        2,
        *others,
        53,
        54,
        55,
        74,
        75,
        77,
        78,
        128,
        256,
        383,
        384,
        512,
        640,
        641,
        *edges,
        1023,
        1024,
        2049,
        6143,
        6144,
        8191,
        8192,
        65536,
        131072,
    ):
        want = (route[sum(rows > e for e in BWD_EDGES["sm_107a"])],)
        for path in paths:
            for iters in (20, 7):
                got = reg.selected_modules(
                    "sm_107a",
                    HIDDEN,
                    "collapse_backward",
                    rows=rows,
                    sm_count=sm,
                    sinkhorn_iters=iters,
                    **path,
                )
                assert got == want, (rows, path, iters)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("sm", (132, 148, 152))
def test_the_sm100a_sm103a_collapse_backward_band_reads_the_token_count_only(
    mode: str, arch: str, sm: int
) -> None:
    """sm_100a's and sm_103a's ``collapse_backward`` takes the ``collapse_bwd__vjpwa_sp4_mb4`` library up to 215 tokens
    but the stage-free ``collapse_bwd__vjpwd_sp4_mb4`` library from 31 to 37 tokens on sm_100a and from 31 to 38
    tokens on sm_103a (:data:`BWD_DIRECT_ROWS`), each band under its own record, then the
    ``collapse_bwd__vjpwa_sp2_mb4`` library (sm_100a from 216 to 1023 tokens as
    ``collapse_bwd__vjpwa_sp2_mb4__h5120__le1023``; sm_103a from 216 to 247 tokens as ``__le247`` and from 320 to 1023
    tokens as ``__gt319_le1023``), the ``collapse_bwd__vjpwa_pf0`` library (sm_100a from 1024 to 6143 tokens as itself;
    sm_103a from 248 to 319 tokens as ``__gt247_le319`` and from 1024 to 8191 tokens as ``__gt1023_le8191``) and the
    stage-free one-CTA ``collapse_bwd__vjpwd`` library (sm_100a from 6144 tokens as itself; sm_103a from 8192 tokens
    as ``__gt8191``) (:data:`BWD_EDGES`), on every board and Sinkhorn count (and, in the deterministic registry, along
    every call path)."""
    p = PREFIX[mode]
    route = _collapse_backward_route(mode, arch)
    sp4, vjpwd, sp2, pf0, vd = (
        f"{p}collapse_bwd__vjpwa_sp4_mb4__h{HIDDEN}",
        f"{p}collapse_bwd__vjpwd_sp4_mb4__h{HIDDEN}",
        f"{p}collapse_bwd__vjpwa_sp2_mb4__h{HIDDEN}",
        f"{p}collapse_bwd__vjpwa_pf0__h{HIDDEN}",
        f"{p}collapse_bwd__vjpwd__h{HIDDEN}",
    )
    if arch == "sm_100a":
        assert BWD_EDGES[arch] == (30, 37, 215, 1023, 6143)
        assert route == [
            f"{sp4}__le30",
            vjpwd,
            f"{sp4}__gt37_le215",
            f"{sp2}__le1023",
            pf0,
            vd,
        ]
        libraries = [sp4, vjpwd, sp4, sp2, pf0, vd]
        tail = ((1023, sp2), (6143, pf0), (None, vd))
    else:
        assert BWD_EDGES[arch] == (30, 38, 215, 247, 319, 1023, 8191)
        assert route == [
            f"{sp4}__le30",
            f"{vjpwd}__gt30_le38",
            f"{sp4}__gt38_le215",
            f"{sp2}__le247",
            f"{pf0}__gt247_le319",
            f"{sp2}__gt319_le1023",
            f"{pf0}__gt1023_le8191",
            f"{vd}__gt8191",
        ]
        libraries = [sp4, vjpwd, sp4, sp2, pf0, sp2, pf0, vd]
        tail = ((247, sp2), (319, pf0), (1023, sp2), (8191, pf0), (None, vd))
    reg = REGISTRIES[mode]
    assert [reg.MODULES[n]["library"] for n in route] == libraries
    # up to 215 tokens: the stage-free library exactly at BWD_DIRECT_ROWS, the staged four-CTA split elsewhere
    direct = set(BWD_DIRECT_ROWS[arch])
    assert len(direct) == {"sm_100a": 7, "sm_103a": 8}[arch]
    for rows in range(1, 216):
        (name,) = reg.selected_modules(
            arch, HIDDEN, "collapse_backward", rows=rows, sm_count=sm
        )
        assert reg.MODULES[name]["library"] == (vjpwd if rows in direct else sp4), rows
    # from 216 to 16384 tokens: the library of every token count (the bands above 215 written out)
    for rows in range(216, 16385):
        (name,) = reg.selected_modules(
            arch, HIDDEN, "collapse_backward", rows=rows, sm_count=sm
        )
        want = next(lib for last, lib in tail if last is None or rows <= last)
        assert reg.MODULES[name]["library"] == want, rows
    paths = [{}]
    if mode == "det":
        paths += [{"path": path} for path in reg.PDL_PATHS]
    edges = tuple(e + d for e in BWD_EDGES[arch] for d in (-1, 0, 1, 2))
    for rows in (
        1,
        2,
        30,
        31,
        37,
        38,
        39,
        53,
        54,
        55,
        74,
        75,
        77,
        78,
        128,
        *edges,
        247,
        248,
        319,
        320,
        383,
        384,
        512,
        640,
        641,
        768,
        2049,
        4096,
        6143,
        6144,
        8191,
        8192,
        16384,
        65536,
        131072,
    ):
        want = (route[sum(rows > e for e in BWD_EDGES[arch])],)
        for path in paths:
            for iters in (20, 7):
                got = reg.selected_modules(
                    arch,
                    HIDDEN,
                    "collapse_backward",
                    rows=rows,
                    sm_count=sm,
                    sinkhorn_iters=iters,
                    **path,
                )
                assert got == want, (rows, path, iters)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
def test_handoff_launch_grids_and_pdl_arguments(
    monkeypatch: pytest.MonkeyPatch, mode: str, arch: str, sm: int
) -> None:
    reg, swaps = REGISTRIES[mode], _swaps(mode, arch)
    monkeypatch.setattr(reg, "sm_count", lambda device_index: sm)
    monkeypatch.setattr(reg, "device_arch", lambda device_index: arch)
    pi_pn = reg.MODULES[_collapse(mode, arch)[0]]
    fwd_names = _collapse_forward_route(mode, arch)
    bwd_names = _collapse_backward_route(mode, arch)
    fwds = [reg.MODULES[n] for n in fwd_names]
    base_collapse = f"{PREFIX[mode]}fwd_collapse__h{HIDDEN}"
    collapse_ctas = COLLAPSE_ROWS_PER_CTA[arch]
    for rows in ROWS:
        assert reg.launch_grid(reg.MODULES[base_collapse], rows, device_index=0) == (
            math.ceil(5 * rows / 2),
            1,
            1,
        )
        assert reg.launch_grid(pi_pn, rows, device_index=0) == (
            math.ceil(5 * rows / collapse_ctas),
            1,
            1,
        )
        for base, twin in swaps.items():
            if base != base_collapse:
                assert reg.launch_grid(
                    reg.MODULES[twin], rows, device_index=0
                ) == reg.launch_grid(reg.MODULES[base], rows, device_index=0), (
                    twin,
                    rows,
                )
            for path in reg.PDL_PATHS:
                args = (rows, arch, path) if mode == "det" else (rows, arch, path, sm)
                assert reg.enable_pdl(twin, *args) == reg.enable_pdl(base, *args), (
                    twin,
                    rows,
                    path,
                )
        for fwd in fwds:
            assert reg.launch_grid(fwd, rows, device_index=0) == (
                math.ceil(5 * rows / collapse_ctas),
                1,
                1,
            ), (fwd["kernel"], rows)
        for name in bwd_names:
            grid = reg.launch_grid(reg.MODULES[name], rows, device_index=0)
            k = COLLAPSE_BWD_SPLIT.get(name.removeprefix(PREFIX[mode]), 1)
            assert grid == (k * rows, 1, 1), (name, rows)
        for path in reg.PDL_PATHS:
            for name in (*fwd_names, *bwd_names):
                args = (rows, arch, path) if mode == "det" else (rows, arch, path, sm)
                assert reg.enable_pdl(name, *args) == 1, (name, rows, path)


@pytest.mark.cpu_only
def test_the_nondet_handoff_records_are_the_det_records_renamed() -> None:
    def renamed(value):
        if isinstance(value, str):
            return value.replace(PREFIX["det"], PREFIX["nondet"])
        if isinstance(value, dict):
            return {renamed(k): renamed(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(renamed(v) for v in value)
        return value

    det_names = {
        name
        for routes in DET.ROUTES.values()
        for key, names in routes.items()
        if key.rsplit("_h", 1)[0] in HANDOFF_KINDS
        for name in names
    }
    new = {
        n
        for n in det_names
        if any(
            s in n for s in ("_pi_pn__", "collapse_fwd__", "collapse_bwd__", "_ho__")
        )
    }
    # the fused collapse, the standalone collapse pair (each in its sm_100a / sm_103a and its sm_107a form, the
    # forward's sm_107a form in its three token-count records, the VJP's in its four, the sm_100a / sm_103a VJP bands'
    # aliases and the stage-free VJPs) and one twin per token-owner form
    want = {f"{PREFIX['det']}{s}__h{HIDDEN}" for s in COLLAPSE_RECORD_ARCHES}
    want |= {f"{PREFIX['det']}{s}" for s in COLLAPSE_BWD_ALIAS}
    want |= {f"{PREFIX['det']}bwd_token__{f}_ho__h{HIDDEN}" for f in TOKEN_FORMS}
    assert new == want, sorted(new)
    for name in new:
        assert ND.MODULES[renamed(name)] == renamed(DET.MODULES[name]), name
    for arch in HANDOFF_ARCHES:
        for kind in ("collapse_forward", "collapse_backward"):
            key = f"{kind}_h{HIDDEN}"
            assert ND.ROUTES[arch][key] == renamed(DET.ROUTES[arch][key])
