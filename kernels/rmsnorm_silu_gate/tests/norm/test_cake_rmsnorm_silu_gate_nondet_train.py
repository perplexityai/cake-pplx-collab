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

Backend suite of the Cake SiLU-gated RMSNorm training package of the
non-deterministic family
(``cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train``, the backend of
:mod:`cake_pplx.cake_rmsnorm_silu_gate_nondet_train`):
``RMSNorm(x) * w * SiLU(z)``.

Every training output (``y``, ``dx``, ``dz`` and the weight gradient of
either dtype, accumulated or not) is judged against the PyTorch BF16 naive
implementation of the operation, each against a float64 oracle of the BF16
inputs, over 32 random inputs of a case
(:mod:`tests.cake.test_helpers.silu_reference`, the ``TIER`` lines of the
terminal summary); no output has an error bound of its own, and no test
compares this package with the deterministic one.  ``y``, ``dx``, ``dz`` and an
FP32 weight gradient (accumulated or not) must be the same bits on every
launch and on every path: streams, CUDA Graph replays, autograd,
``compute_grad_weight`` (``False`` the bits of the BF16 bands' module, but
on the packed rows of the build of packed rows only, the repeatable
backward's), grids
smaller than the work (the static backward of an FP32 weight gradient on
sm_100a and sm_103a, the fused backward and the repeatable backwards of
both weight-gradient dtypes and their long roles at 1, 3 and 7
CTAs; the libraries no band selects, ``UNBANDED_LIBRARIES``, only at the
host's grid on a forced route; a repeatable backward's
weight gradient the bits of its grid, whose chunks follow it), a workspace
reused across shapes; so must a BF16 weight gradient of every backward but
the fused backward (``bwd_bf16_fused``, which runs a BF16 weight gradient's
rows in a band of ``BACKWARD_BANDS`` past the small-row bands on every
architecture, on sm_107a between two repeatable bands) and the row-claims backwards (``_rc``
and ``_rc_mid``, :func:`_varies`), whose CTAs claim 16-row blocks at run time
and add FP32 runs of rows into FP32 replicas with FP32 atomic adds,
so that their BF16 weight gradient may vary in its last bits from run to run:
there every launch's BF16 weight gradient, alone and accumulated, is judged
against the naive implementation instead.  The matrix covers dynamic token and head
counts (``H = 48`` and small or irregular row counts), ``[T, H, 128]`` strided
and flattened layouts (also in the bands of the static backward of an FP32
weight gradient and of the fused and the repeatable backwards of both
weight-gradient dtypes; the BF16 claimed and the
row-claims programs, which no band runs, only on the forced route, on
contiguous and projection inputs),
near-zero variance, saturated gates, weight-gradient terms that cancel per
column (each into a BF16 and an FP32 weight gradient), FP32 and BF16 weight
gradients,
accumulation, ``compute_grad_weight=False``, the small-row backwards (one
cluster each, in the rows of their ``BACKWARD_BANDS`` bands: the short
backward of a BF16 weight gradient and the exact tiny backwards; no workspace made
or touched, a caller workspace checked and left as it is, the first row count
past them on the backward), empty work, a workspace reused across shapes (zero after every
call), a caller workspace's checked buffers serving its later calls, made
again once its storage was freed in place and the tensor rebound, and going
with the tensor, a workspace with nonzero scratch returned to zero by
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace``, two streams with their own
workspaces, CUDA Graph capture and replay (captured calls without a
workspace: one workspace per capture and capture stream, zero-filled outside
the graph, no fill kernel in it; two such graphs replayed at the same time on
two streams; the fill inside the graph past the graph-workspace limit), the
autograd wrapper (non-reentrant checkpointing included), the state a
checkpointed step keeps, the registry's routes (a
module for every call, which the support query requires; none of the programs
the package no longer ships, the claimed backwards of an FP32 weight
gradient and its static backward of sm_107a among them), bands, chunks and
grids (the first and the last
rows of every band of ``BACKWARD_BANDS`` run the band's program; packed rows
of the last repeatable band of either weight gradient from
``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on run its long role,
``bwd_bf16_rep_long`` or ``bwd_f32_rep_long``, the bits of the repeatable
backward; the grid ``run_module`` launches is ``launch_grid``'s for
every record, also with a forward's rule changed at run time), the programs
without the weight gradient (``NODW_ROLES``, which
``nodw_module`` names in place of a BF16 band's module for
``compute_grad_weight=False``: one cluster in the small-row bands, in the
fused and the repeatable bands the strided program up to
``NODW_STRIDED_ROWS_PER_SM`` rows per SM (on packed rows its 128-thread
build where routed: the repeatable band's on sm_107a) and the claimed one
past it, or the one of them the architecture routes, and the long role's;
the ``dx`` and ``dz`` bits of the BF16 call around the cut
on packed and strided rows, in the small-row bands and at grids of 1, 3 and
7 CTAs; the strided and the one-cluster ones with no stream or reserve
workspace, also in CUDA Graph capture, and a caller workspace left as it is;
the claimed ones leaving the workspace zero; on packed rows, and one token
of head stride 128, up to ``PACKED_SMALL_ROWS_PER_SM`` rows per SM the
repeatable band's build of packed rows only in every band's place, around
its cut, at grids of 1, 3 and 7 CTAs, never on other layouts, with the
``dx`` and ``dz`` bits of the BF16 call on the repeatable backward
(``bwd_bf16_rep`` forced) and judged against the naive implementation; the
plans ``nodw_module`` reads, read once at import, those of the tables), and the mode
policy: the mode is in the name, so no entry point takes a ``deterministic``
keyword (passing one, of any value, is a ``TypeError`` before any launch); under
``torch.use_deterministic_algorithms(True)`` a backward that computes the
weight gradient on the fused or a row-claims backward (grid kind
``row_claims``) raises PyTorch's alert immediately
before its launch (a ``RuntimeError``; with ``warn_only=True`` a
``UserWarning``, after which it runs; a CUDA Graph warns when it is captured
and not when it is replayed), and every other call (the other bands,
``compute_grad_weight=False``, no rows, the forward, the workspace entry
points, the reset and the support query) runs without an alert or a warning
(strict and ``warn_only``, CUDA Graph capture included) and returns the bits
it returns with the setting off; the deterministic package's FP32 workspace is
refused naming both modes' workspaces.

This suite exercises the package alone, named once through
``tests.cake.test_helpers.cake_package`` (so a vendored copy runs it unchanged), and
imports nothing else from ``cake_pplx``.  The facade is
``test_cake_rmsnorm_silu_gate_nondet_train_api.py``'s.
"""

from __future__ import annotations

import contextlib
import gc
import inspect
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_naive import DRAWS
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.silu_reference import Draw, judge, silu_draw

KEY = "cake_rmsnorm_silu_gate_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
loader = cake_package(KEY, "cake_jit")
forward = pkg.cake_rmsnorm_silu_gate_nondet_train_forward
backward = pkg.cake_rmsnorm_silu_gate_nondet_train_backward
workspace_for = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace
workspace_bytes = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes
reset_workspace = pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace
gated_rmsnorm = pkg.cake_rmsnorm_silu_gate_nondet

D = 128
EPS = 1e-6
REPEATS = 10
DET = "cake_pplx.cake_rmsnorm_silu_gate_det_train"
# PyTorch's alert of a backward under its deterministic algorithms.
ALERT = "does not have a deterministic implementation"
# The workspace of a backward module by its slots, ``(buffer, byte offset,
# elements)``: the u64 windows of the exact weight-gradient sum (slots x 7
# windows x 128 columns; on the sparse programs of sm_107a the last slot holds
# the occupancy words), the FP32 class words (128) and the u32 arrival ticket
# and claim counter, each buffer 16-byte aligned; and its bytes.
WORKSPACE_LAYOUTS = {
    4: (("dw_win", 0, 4 * 7 * 128), ("dw_spec", 28672, 128), ("dw_done", 29184, 2)),
    5: (("dw_win", 0, 5 * 7 * 128), ("dw_spec", 35840, 128), ("dw_done", 36352, 2)),
}
MODULE_WORKSPACE_BYTES = {4: 29192, 5: 36360}
# The slots of each static and claimed backward module, by architecture and
# role: on sm_100a and sm_103a 4 for the static backward of an FP32 weight
# gradient and the claimed backward (``_mid``) of a BF16 one; on sm_107a 5 for
# the static backward and 4 for the claimed backward of a BF16 weight
# gradient (no claimed backward of an FP32 weight gradient ships, nor a
# static one on sm_107a).
BACKWARD_SLOTS = {
    "sm_100a": {"bwd_f32": 4, "bwd_bf16_mid": 4},
    "sm_103a": {"bwd_f32": 4, "bwd_bf16_mid": 4},
    "sm_107a": {"bwd_bf16": 5, "bwd_bf16_mid": 4},
}
# The torch dtype of each workspace buffer's view.
WORKSPACE_DTYPES = {
    "dw_win": torch.uint64,
    "dw_sums": torch.float32,
    "dw_spec": torch.float32,
    "dw_done": torch.uint32,
    "acc": torch.uint64,
    "flags": torch.uint32,
    "done": torch.uint32,
}
# The row-claims backwards of a BF16 weight gradient (the base and the mid
# backward whose CTAs claim 16-row blocks at run time) and their grid rules:
# at most 4 (``_rc``) and 5 (``_rc_mid``) CTAs per SM; and their workspace: 4
# FP32 replicas of the 128 column sums and the u32 arrival ticket and claim
# counter, 2056 bytes.
ROW_CLAIMS_GRID_RULES = {
    "bwd_bf16_rc": {"ctas_per_sm": 4, "kind": "row_claims", "rows_per_cta": 16},
    "bwd_bf16_rc_mid": {"ctas_per_sm": 5, "kind": "row_claims", "rows_per_cta": 16},
}
ROW_CLAIMS_REPLICAS = 4
ROW_CLAIMS_LAYOUT = (("dw_sums", 0, 4 * D), ("dw_done", 2048, 2))
ROW_CLAIMS_WORKSPACE_BYTES = 2056
# The fused backward of a BF16 weight gradient on every architecture
# (``bwd_bf16_fused``: the row arithmetic of the deterministic package's BF16
# backward program, the claimed 16-row block schedule, the weight gradient
# added into FP32 replicas with FP32 atomic adds; an FP32 weight gradient runs
# no fused backward) and its grid rule: at most 4 CTAs per SM claiming 16-row
# blocks, the grid kind and the rule of ``bwd_bf16_rc``; and its workspace: 8
# FP32 replicas of the 128 column sums and the two words, 4104 bytes.
FUSED_ROLES = {
    "sm_100a": ("bwd_bf16_fused",),
    "sm_103a": ("bwd_bf16_fused",),
    "sm_107a": ("bwd_bf16_fused",),
}
FUSED_GRID_RULE = {"ctas_per_sm": 4, "kind": "row_claims", "rows_per_cta": 16}
FUSED_REPLICAS = 8
FUSED_LAYOUT = (("dw_sums", 0, 8 * D), ("dw_done", 4096, 2))
FUSED_WORKSPACE_BYTES = 4104
# The repeatable backwards on every architecture, of a BF16 weight gradient
# (``bwd_bf16_rep``) and of an FP32 one (``bwd_f32_rep``, past the FP32
# small-row band): the deterministic package's fused backward program's BF16
# and FP32 builds under this package's names, the fused backward's schedule
# and row arithmetic with the weight gradient summed exactly in fixed point,
# so every output repeats its bits; and their long roles
# (``bwd_bf16_rep_long``, ``bwd_f32_rep_long``: the deterministic package's
# look-ahead programs of the same builds, the same bits, which run packed
# rows from ``LONG_ROWS_PER_SM`` rows per SM on) and their grid rule: at most
# 4 CTAs per SM claiming 16-row blocks; and their workspace: 2304 u64
# fixed-point digits (``acc``), a u32 flag word a column (``flags``) and the
# u32 arrival ticket and claim counter (``done``), 18952 bytes.
REP_ROLES = {
    arch: ("bwd_bf16_rep", "bwd_bf16_rep_long", "bwd_f32_rep", "bwd_f32_rep_long")
    for arch in ("sm_100a", "sm_103a", "sm_107a")
}
# The rows per SM from which a repeatable backward's long role runs packed
# rows of its band (``LONG_BACKWARD_ROWS_PER_SM``, the same for both
# weight-gradient dtypes): 2560 on sm_100a and sm_103a, 1536 on sm_107a,
# where of a BF16 weight gradient only the last repeatable band reaches
# them.
LONG_ROWS_PER_SM = {"sm_100a": 2560, "sm_103a": 2560, "sm_107a": 1536}
REP_GRID_RULE = {"ctas_per_sm": 4, "kind": "claimed_row_blocks", "rows_per_block": 16}
REP_LAYOUT = (("acc", 0, 18 * D), ("flags", 18432, D), ("done", 18944, 2))
REP_WORKSPACE_BYTES = 18952
# The bytes of the backward workspace of each architecture, the largest of its
# backward modules' (one zero-filled workspace serves them all, each module
# laying its buffers over the first bytes): 29192 on sm_100a and sm_103a, 36360
# on sm_107a; and the stride of the device's reserve of capture workspaces
# (256-byte aligned).
WORKSPACE_BYTES = {"sm_100a": 29192, "sm_103a": 29192, "sm_107a": 36360}
RESERVE_STRIDE = {"sm_100a": 29440, "sm_103a": 29440, "sm_107a": 36608}
# The programs without the weight gradient (``NODW_ROLES``) and the band
# role of the program each runs in place of for ``compute_grad_weight=False``
# (its twin, whose ``dx`` and ``dz`` bits it gives), whose arguments it
# takes: the one-cluster and the claimed programs launch as their twin and
# keep its grid rule and workspace; the strided programs take no workspace
# and launch by their own rule (``NODW_STRIDED_GRID_RULES``) CTAs of
# ``NODW_STRIDED_THREADS`` threads: their twin's CTA count of 256-thread CTAs
# over 16-row blocks, and the repeatable band's strided program of packed
# rows 128-thread CTAs over 8-row blocks, at most 8 per SM.
NODW_TWINS = {
    "bwd_bf16_short_nodw": "bwd_bf16_short",
    "bwd_bf16_t512_nodw": "bwd_bf16_t512",
    "bwd_bf16_fused_nodw": "bwd_bf16_fused",
    "bwd_bf16_fused_nodw_strided": "bwd_bf16_fused",
    "bwd_bf16_rep_nodw": "bwd_bf16_rep",
    "bwd_bf16_rep_nodw_strided": "bwd_bf16_rep",
    "bwd_bf16_rep_nodw_strided_packed": "bwd_bf16_rep",
    "bwd_bf16_rep_nodw_strided_packed_small": "bwd_bf16_rep",
    "bwd_bf16_rep_nodw_long": "bwd_bf16_rep_long",
}
NODW_STRIDED_GRID_RULES = {
    "bwd_bf16_fused_nodw_strided": {
        "ctas_per_sm": 4,
        "kind": "strided_row_blocks",
        "rows_per_block": 16,
    },
    "bwd_bf16_rep_nodw_strided": {
        "ctas_per_sm": 4,
        "kind": "strided_row_blocks",
        "rows_per_block": 16,
    },
    "bwd_bf16_rep_nodw_strided_packed": {
        "ctas_per_sm": 8,
        "kind": "strided_row_blocks",
        "rows_per_block": 8,
    },
    "bwd_bf16_rep_nodw_strided_packed_small": {
        "ctas_per_sm": 8,
        "kind": "strided_row_blocks",
        "rows_per_block": 8,
    },
}
NODW_STRIDED_THREADS = {
    "bwd_bf16_fused_nodw_strided": 256,
    "bwd_bf16_rep_nodw_strided": 256,
    "bwd_bf16_rep_nodw_strided_packed": 128,
    "bwd_bf16_rep_nodw_strided_packed_small": 128,
}
# The repeatable band's build of packed rows only (``PACKED_SMALL_ROLE``),
# which a call with ``compute_grad_weight=False`` runs on packed rows (and
# one token of head stride 128) up to ``PACKED_SMALL_ROWS_PER_SM`` rows per
# SM in the place of the program without the weight gradient of every band
# that routes one: the ``dx`` and ``dz`` bits of the repeatable backward,
# not of the band's module; and its shipped cuts, rows per SM by
# architecture (the BF16 bands' role), none on sm_107a, where no
# measurement backs one.
NODW_SMALL = "bwd_bf16_rep_nodw_strided_packed_small"
SMALL_ROWS_PER_SM = {
    "sm_100a": {"bwd_bf16": 128},
    "sm_103a": {"bwd_bf16": 128},
    "sm_107a": {},
}
# The bands with a strided program, which has a cut
# (``NODW_STRIDED_ROWS_PER_SM``) where the band's architecture routes both
# its strided and its claimed program.
NODW_CUT_BANDS = ("bwd_bf16_fused", "bwd_bf16_rep")

# The accuracy matrix (shapes, layouts, edge data and epsilons judged against
# the PyTorch BF16 naive tier).
SHAPES = [
    (1, 1),
    (1, 48),
    (3, 48),
    (5, 48),
    (13, 48),
    (17, 7),
    (255, 2),
    (1031, 48),
    (4097, 16),
    (2, 64),
    (33, 1),
]
LAYOUTS = ["flat", "projection", "head_major", "misaligned"]
EDGE_DATA = ["near_zero", "saturated", "dw_cancel"]
EPSILONS = [1e-5, 1e-3]
ISSUE_TOKENS = [30193, 32737, 32388]
FP32_SHAPES = [(7, 48), (1031, 48), (32737, 48)]


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture(autouse=True)
def deterministic_algorithms_off():
    """Run with ``torch.use_deterministic_algorithms`` off (tests that need it
    on turn it on); restore the caller's setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The module names this package's registry launches, in order."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def _launched_grid(name, args):
    """The grid of a launch of module ``name`` with the kernel arguments
    ``args``: the arguments at its grid slots (``_arguments``), by axis."""
    grid = [None, None, None]
    for arg, (axis, slot, _) in zip(args, loader._arguments(name), strict=True):
        if axis:
            grid[slot] = arg
    return tuple(grid)


def _record_launches(patch):
    """Patch the registry's ``_entry`` so that each launch appends ``(name,
    grid)`` (``_launched_grid``: the grid the kernel is launched with) to the
    returned list, then runs the kernel."""
    launched = []
    entry = loader._entry

    def entry_spy(name, device_index):
        run = entry(name, device_index)

        def launch(*args):
            launched.append((name, _launched_grid(name, args)))
            return run(*args)

        return launch

    patch.setattr(loader, "_entry", entry_spy)
    return launched


def _routes() -> dict[str, str]:
    return loader.ROUTES[loader.device_arch(torch.cuda.current_device())]


def _module(role: str, rows: int, *, packed: bool = True) -> str:
    """The module ``module_for_rows`` runs for ``role`` and ``rows`` rows on
    this device (``packed``: the forward's inputs hold packed rows)."""
    device_index = torch.cuda.current_device()
    return loader.module_for_rows(
        loader.device_arch(device_index),
        role,
        rows,
        loader.device_sm_count(device_index),
        packed=packed,
    )


def _packed(T: int, H: int, layout: str = "contiguous") -> bool:
    """Whether ``make(T, H, layout=layout)``'s ``x`` and ``z`` hold packed
    rows (row ``n`` at element ``128 n``) as the forward reads them: the
    stride of a single token or a single head is never read, and the host
    copies a misaligned input to packed rows."""
    if layout in ("contiguous", "flat", "misaligned"):
        return True
    if layout == "projection":
        return T == 1
    if layout == "head_major":
        return T == 1 or H == 1
    raise ValueError(layout)


def _backward_packed(T: int, H: int, layout: str = "contiguous") -> bool:
    """Whether the backward kernels read ``make(T, H, layout=layout)``'s
    ``dy``, ``x`` and ``z`` as packed rows (head stride 128, token stride
    ``H * 128``; a single head's stride passed as 128), the test that picks a
    long role (``LONG_BACKWARD_ROWS_PER_SM``): a single token's stride is
    passed as 0, which is not packed, and the host copies a misaligned input
    to packed rows."""
    tokens = T * H if layout == "flat" else T
    if tokens == 1:
        return False
    if layout in ("contiguous", "flat", "misaligned"):
        return True
    if layout == "projection":
        return False
    if layout == "head_major":
        return H == 1
    raise ValueError(layout)


def _forward_module(rows: int, packed: bool = True) -> str:
    """The forward module ``module_for_rows`` runs for ``rows`` rows on this
    device (the program of the architecture's ``FORWARD_BANDS`` band, of its
    ``STRIDED_FORWARD_BANDS`` band where the rows are not ``packed``)."""
    return _module("fwd", rows, packed=packed)


def _backward_module(
    rows: int, dw_dtype: torch.dtype = torch.bfloat16, *, packed: bool = True
) -> str:
    """The backward module ``module_for_rows`` runs for ``rows`` rows and a
    ``dw_dtype`` weight gradient on this device: the program of the role's
    first ``BACKWARD_BANDS`` band that holds the rows (a small-row backward,
    a static, a row-claims, a fused, the repeatable or the claimed backward),
    that band's long role where the rows are ``packed`` (the backward's test,
    :func:`_backward_packed`) and reach its ``LONG_BACKWARD_ROWS_PER_SM``."""
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    return _module(role, rows, packed=packed)


def _nodw_module(name: str, rows: int, *, packed: bool = True) -> str:
    """The program a call with ``compute_grad_weight=False`` runs for
    ``rows`` rows on this device where ``module_for_rows`` names ``name``
    for its BF16 bands (``nodw_module``), the rows ``packed`` (the
    backward's test, :func:`_backward_packed`) or not."""
    device_index = torch.cuda.current_device()
    return loader.nodw_module(
        loader.device_arch(device_index),
        name,
        rows,
        loader.device_sm_count(device_index),
        packed=packed,
    )


def _one_token_packed(arch: str) -> bool:
    """Whether a call without the weight gradient on ``arch`` passes one
    token of head stride 128 as packed rows (token stride ``H * 128``): where
    the build of packed rows only has a cut (``PACKED_SMALL_ROWS_PER_SM``);
    elsewhere (sm_107a) its token stride stays 0 and it runs as other layouts
    do."""
    return loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0) > 0


def _nodw_packed(T: int, H: int, layout: str = "contiguous") -> bool:
    """Whether a call with ``compute_grad_weight=False`` on ``make(T, H,
    layout=layout)`` tells ``nodw_module`` that its rows are packed: the
    backward's test (:func:`_backward_packed`), and also one token (of the
    flat layout, one row) whose heads have stride 128 on an architecture
    where :func:`_one_token_packed` holds (``make``'s contiguous, flat,
    misaligned and projection inputs; head-major ``x`` and ``z`` of one token
    too, whose heads lie 128 apart)."""
    tokens = T * H if layout == "flat" else T
    arch = loader.device_arch(torch.cuda.current_device())
    if tokens == 1 and _one_token_packed(arch):
        return True
    return _backward_packed(T, H, layout)


def _small_cut_rows() -> int:
    """The most packed rows the build of packed rows only runs on this
    device (``PACKED_SMALL_ROWS_PER_SM * SMs``; 0 where it has no cut or is
    not routed)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    if NODW_SMALL not in loader.ROUTES[arch]:
        return 0
    per_sm = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
    return per_sm * loader.device_sm_count(device_index)


def _past_small_cut(T: int, H: int) -> int:
    """``T``, or the fewest tokens of ``H`` heads whose packed rows lie past
    this device's cut of the build of packed rows only where that is more
    (:func:`_small_cut_rows`)."""
    return max(T, _small_cut_rows() // H + 1)


def _is_small(name: str) -> bool:
    """Whether module ``name`` is the build of packed rows only."""
    return loader.MODULES[name]["role"] == NODW_SMALL


def _rep_bits(dy, x, z, w, *, workspace=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``dx`` and ``dz`` of the BF16 call on the repeatable backward
    (``bwd_bf16_rep`` forced through ``module_for_rows``, :func:`_forcing`)
    on the same inputs: the bits the build of packed rows only gives at any
    grid (the repeatable backward's ``dx`` and ``dz`` do not depend on the
    grid)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "module_for_rows", _forcing("bwd_bf16_rep"))
        dx, dz, _ = backward(dy, x, z, w, EPS, workspace=workspace)
    return dx, dz


def _nodw_bits(nodw: str, dy, x, z, w, *, workspace=None):
    """``dx`` and ``dz`` whose bits a call with ``compute_grad_weight=False``
    that runs ``nodw`` gives: the repeatable backward's for the build of
    packed rows only (:func:`_rep_bits`), else the BF16 call's on the same
    inputs (the band's module)."""
    if _is_small(nodw):
        return _rep_bits(dy, x, z, w, workspace=workspace)
    dx, dz, _ = backward(dy, x, z, w, EPS, workspace=workspace)
    return dx, dz


def _patch_routes(patch, arch: str, table: dict[str, str]) -> None:
    """``ROUTES[arch]`` set to ``table`` for the ``patch`` context, with the
    plans :func:`nodw_module` reads for ``arch`` (``_NODW_PLANS``, read once
    at import) read again from it."""
    patch.setitem(loader.ROUTES, arch, table)
    plans = {name: loader._nodw_plan(arch, name) for name in loader.MODULES}
    patch.setitem(loader._NODW_PLANS, arch, plans)


def _chunks(rows: int) -> int:
    """The chunks of the backward with a workspace for ``rows`` rows: 8-row
    blocks, at most 8192 chunks of the fewest consecutive blocks each."""
    blocks = -(-rows // 8)
    return -(-blocks // max(1, -(-blocks // 8192)))


def _band_end(last, sm_count: int) -> int | None:
    """The most rows of a ``BACKWARD_BANDS`` band whose ``last`` is ``last``
    on ``sm_count`` SMs: ``n`` for ``(ROWS, n)`` (a count of rows), the most
    rows with ``rows <= last * sm_count`` for a number (rows per SM), ``None``
    for the last band, which has no end."""
    if last is None:
        return None
    if isinstance(last, tuple):
        assert len(last) == 2 and last[0] == loader.ROWS, last
        return int(last[1])
    return math.floor(last * sm_count)


def _backward_bands(
    arch: str, role: str, sm_count: int
) -> list[tuple[str, int, int | None]]:
    """``(band role, first rows, last rows or None)`` of each
    ``BACKWARD_BANDS`` band of the backward ``role`` (``bwd_bf16`` or
    ``bwd_f32``) on ``arch`` with ``sm_count`` SMs that runs at least one row
    count, in order: a band runs the rows past the ends of the bands before it
    up to its own end (:func:`_band_end`); the last band has no end."""
    bands = []
    covered = 0
    for last, band in loader.BACKWARD_BANDS[arch][role]:
        end = _band_end(last, sm_count)
        if end is None or end > covered:
            bands.append((band, covered + 1, end))
        if end is None:
            break
        covered = max(covered, end)
    return bands


def _long_bands(
    arch: str, role: str, sm_count: int
) -> list[tuple[str, int, int | None]]:
    """``(long role, first rows, last rows or None)`` of each band of
    :func:`_backward_bands` whose role names a cut in
    ``LONG_BACKWARD_ROWS_PER_SM[arch]`` that its rows reach: the long role
    (the band's role and ``LONG``) runs the band's packed rows from ``cut *
    sm_count`` rows (rows per SM) on, up to the band's end; other rows of the
    band run its role."""
    cuts = loader.LONG_BACKWARD_ROWS_PER_SM.get(arch, {})
    found = []
    for band, first, last in _backward_bands(arch, role, sm_count):
        if band in cuts:
            start = max(first, cuts[band] * sm_count)
            if last is None or start <= last:
                found.append((band + loader.LONG, start, last))
    return found


def _device_bands(role: str) -> list[tuple[str, int, int | None]]:
    """:func:`_backward_bands` of the backward ``role`` on this device."""
    device_index = torch.cuda.current_device()
    return _backward_bands(
        loader.device_arch(device_index), role, loader.device_sm_count(device_index)
    )


def _device_long_bands(role: str) -> list[tuple[str, int, int | None]]:
    """:func:`_long_bands` of the backward ``role`` on this device."""
    device_index = torch.cuda.current_device()
    return _long_bands(
        loader.device_arch(device_index), role, loader.device_sm_count(device_index)
    )


def _device_band(band: str) -> tuple[int, int | None] | None:
    """``(first rows, last rows or None)`` of the ``BACKWARD_BANDS`` band of
    the backward role ``band`` (``bwd_bf16_rc``, for one) on this device
    (:func:`_device_bands`), or of the long role ``band``
    (``bwd_bf16_rep_long``: its packed rows, :func:`_device_long_bands`);
    ``None`` where this device routes no such band."""
    role = "bwd_f32" if band.startswith("bwd_f32") else "bwd_bf16"
    for name, first, last in _device_bands(role) + _device_long_bands(role):
        if name == band:
            return first, last
    return None


def _band_tokens(band: str, heads: int = 48) -> int | None:
    """The fewest tokens of ``heads`` heads whose rows lie in the
    ``BACKWARD_BANDS`` band of the backward role ``band`` (``bwd_bf16_rc``,
    for one) on this device; ``None`` where this device routes no such band
    or no such count of rows lies in it."""
    found = _device_band(band)
    if found is None:
        return None
    first, last = found
    tokens = -(-first // heads)
    return tokens if last is None or tokens * heads <= last else None


def _band_shape(band: str) -> tuple[int, int] | None:
    """``(T, H)`` of the fewest rows in the ``BACKWARD_BANDS`` band of the
    backward role ``band`` on this device as tokens of 48 heads, else (a
    band narrower than that) the band's first rows as tokens of one head;
    ``None`` where this device routes no such band.  The first band of the
    role where it has more than one (on sm_107a the BF16 repeatable role
    runs on both sides of the fused band, on sm_100a and sm_103a the FP32 one
    on both sides of the static band; :func:`_range_shape` takes any); a long
    role's band (:func:`_device_band`) holds packed rows, which these shapes
    are."""
    tokens = _band_tokens(band)
    if tokens is not None:
        return tokens, 48
    tokens = _band_tokens(band, heads=1)
    return None if tokens is None else (tokens, 1)


def _range_shape(first: int, last: int | None) -> tuple[int, int]:
    """``(T, H)`` of the fewest rows from ``first`` to ``last`` (``None``: no
    end) as tokens of 48 heads, else (a range narrower than that) ``first``
    as tokens of one head."""
    tokens = -(-first // 48)
    if last is None or tokens * 48 <= last:
        return tokens, 48
    return first, 1


def _device_band_first(rows: int, dw_dtype: torch.dtype) -> tuple[str, int]:
    """``(role, first rows)`` of the band that runs ``rows`` packed rows of a
    ``dw_dtype`` weight gradient on this device: a long role's
    (:func:`_device_long_bands`) where it holds them, else the band of
    :func:`_device_bands` that holds them."""
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    for _band, first, last in _device_long_bands(role) + _device_bands(role):
        if first <= rows and (last is None or rows <= last):
            return role, first
    raise AssertionError(f"no {role} band of this device holds {rows} rows")


def _varies(name: str) -> bool:
    """Whether the BF16 weight gradient of backward module ``name`` may vary
    in its last bits from run to run: the fused backward's and a row-claims
    backward's (``_fused``, ``_rc`` and ``_rc_mid``, grid kind
    ``row_claims``), whose CTAs claim 16-row blocks at run time and add the
    FP32 sums of their rows into FP32 replicas with FP32 atomic adds.  Every
    other backward repeats its bits: the small-row backwards sum on chip in a
    fixed order, the static
    and the claimed backwards add their partials exactly, as integers, and so
    do the repeatable backwards their chunks' FP32 words, as fixed-point
    digits."""
    return loader.MODULES[name]["grid_rule"]["kind"] == "row_claims"


def _rep(name: str) -> bool:
    """Whether backward module ``name`` is a repeatable backward of either
    weight-gradient dtype (the deterministic package's fused backward
    program, ``_rep``, or its look-ahead program, ``_rep_long``): its weight
    gradient's bits are those of the rows and the grid (its chunks' rows
    follow the grid), so a forced grid may give other bits than the host's."""
    return loader.MODULES[name]["grid_rule"]["kind"] == "claimed_row_blocks"


def _dw_varies(rows: int, dw_dtype: torch.dtype = torch.bfloat16) -> bool:
    """Whether the ``dw_dtype`` weight gradient of ``rows`` rows on this
    device may vary in its last bits from run to run (:func:`_varies` of the
    module ``module_for_rows`` runs; never for an FP32 weight gradient)."""
    return _varies(_backward_module(rows, dw_dtype))


def _expected_layout(arch: str, role: str) -> tuple:
    """The workspace layout ``(buffer, byte offset, elements)`` of the
    backward module that ``role`` routes on ``arch``: ``ROW_CLAIMS_LAYOUT``
    for a row-claims backward, ``FUSED_LAYOUT`` for a fused backward,
    ``REP_LAYOUT`` for the repeatable backward, the layout of its slots
    (``BACKWARD_SLOTS``, ``WORKSPACE_LAYOUTS``) for a static or a claimed
    backward, none for a small-row backward (one cluster); a program without
    the weight gradient (``NODW_TWINS``) its twin's layout, a strided one
    none."""
    if role in NODW_TWINS:
        if role in NODW_STRIDED_GRID_RULES:
            return ()
        return _expected_layout(arch, NODW_TWINS[role])
    if role in ROW_CLAIMS_GRID_RULES:
        return ROW_CLAIMS_LAYOUT
    if role in FUSED_ROLES[arch]:
        return FUSED_LAYOUT
    if role in REP_ROLES[arch]:
        return REP_LAYOUT
    if role in BACKWARD_SLOTS[arch]:
        return WORKSPACE_LAYOUTS[BACKWARD_SLOTS[arch][role]]
    return ()


def assert_dw_repeats(dw, first, *, varies, what):
    """A weight gradient ``dw`` of a call that ran before as ``first``: the
    same bits where its band repeats them; where it may vary in its last bits
    (``varies``: a fused or a row-claims band, :func:`_varies`) the dtype and
    shape of ``first`` and finite, as the oracle of finite inputs is (such a
    band's accuracy is judged by :func:`check_all` and the band tests)."""
    if varies:
        assert dw.dtype == first.dtype and dw.shape == first.shape, what
        assert bool(dw.isfinite().all()), f"{what}: not finite"
    else:
        assert torch.equal(dw, first), f"{what}: not the bits of the first call"


def make(T, H, *, layout="contiguous", data="normal", seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=g).bfloat16()

    if layout == "contiguous":
        x, z, dy = randn(T, H, D), randn(T, H, D), randn(T, H, D)
    elif layout == "flat":
        x, z, dy = randn(T * H, D), randn(T * H, D), randn(T * H, D)
    elif layout == "projection":
        # z (and dy) as the gate columns of wider projections, read in place.
        x = randn(T, H, D)
        z = randn(T, 2 * H * D + 16)[:, H * D : 2 * H * D].view(T, H, D)
        dy = randn(T, 2 * H * D + 16)[:, : H * D].view(T, H, D)
    elif layout == "head_major":
        x = randn(H, T, D).permute(1, 0, 2)
        z = randn(H, T, D).permute(1, 0, 2)
        dy = randn(T, H, D)
    elif layout == "misaligned":
        # Storage offsets of 8 elements: materialized once by the host.
        x = randn(T * H * D + 8)[8:].view(T, H, D)
        z = randn(T * H * D + 8)[8:].view(T, H, D)
        dy = randn(T * H * D + 8)[8:].view(T, H, D)
    else:
        raise ValueError(layout)
    if data == "near_zero":
        flat = torch.arange(T * H, device=device).view(T, H)
        xv = x.view(T, H, D)
        xv[flat % 3 == 0] = 0
        xv[flat % 3 == 1] = 2.0**-60
    elif data == "dw_cancel":
        # Rows in pairs with the same x and z and opposite dy (but for a small
        # perturbation): each column's weight-gradient terms cancel to a small
        # remainder of their magnitudes.
        xv, zv, gv = x.reshape(-1, D), z.reshape(-1, D), dy.reshape(-1, D)
        pairs = xv.shape[0] // 2
        xv[1 : 2 * pairs : 2] = xv[0 : 2 * pairs : 2]
        zv[1 : 2 * pairs : 2] = zv[0 : 2 * pairs : 2]
        small = (2.0**-6 * torch.randn(pairs, D, device=device, generator=g)).bfloat16()
        gv[1 : 2 * pairs : 2] = -gv[0 : 2 * pairs : 2] + small
    elif data == "saturated":
        values = torch.tensor(
            [-1e4, -100.0, -90.0, -30.0, -10.0, 10.0, 30.0, 90.0, 100.0, 1e4],
            device=device,
        ).bfloat16()
        pick = torch.randint(0, 10, z.shape, device=device, generator=g)
        z.copy_(
            torch.where(
                torch.rand(z.shape, device=device, generator=g) < 0.5, values[pick], z
            )
        )
    w = (1.0 + 0.5 * torch.randn(D, device=device, generator=g)).bfloat16()
    return x, z, w, dy


# The judged outputs of ``check_all`` and the quantity of each.
JUDGED = {"y": "y", "dx": "dx", "dz": "dz", "dw": "dw", "accumulated dw": "dw"}


def check_all(
    record_property,
    inputs,
    eps=EPS,
    *,
    dw_dtype=torch.bfloat16,
    repeats=REPEATS,
    seed=0,
    label="",
):
    """Judge ``y``, ``dx``, ``dz``, the ``dw_dtype`` weight gradient and the
    weight gradient accumulated onto 0.25 against the PyTorch BF16 naive
    implementation over the draws of ``judge`` (``inputs(s)`` gives the
    draw's ``(x, z, w, dy)`` from its seed ``s``; ``label`` prefixes the
    output names).  The first draw launches the forward and the backward
    ``repeats`` times, the others once: ``y``, ``dx`` and ``dz`` the same
    bits on every launch, and so the weight gradient, alone and accumulated,
    where its band repeats it (:func:`_dw_varies`); where it may vary every
    launch's weight gradient is judged (the judge keeps the larger error).
    Returns the first draw's ``(x, z, w, dy)`` and its first launch's
    ``(y, dx, dz, dw)``."""
    first_draw = {}

    def run(j, s):
        x, z, w, dy = inputs(s)
        varies = _dw_varies(x.numel() // D, dw_dtype)
        repeated = ("y", "dx", "dz") if varies else tuple(JUDGED)

        def out(fill=None):
            if fill is not None:
                return torch.full((D,), fill, dtype=dw_dtype, device=x.device)
            return (
                None
                if dw_dtype == torch.bfloat16
                else torch.empty(D, dtype=dw_dtype, device=x.device)
            )

        launched = []
        first = None
        for i in range(repeats if j == 0 else 1):
            y = forward(x, z, w, eps)
            assert y.dtype == torch.bfloat16 and y.shape == x.shape
            dx, dz, dw = backward(dy, x, z, w, eps, grad_weight=out())
            assert dx.dtype == dz.dtype == torch.bfloat16 and dw.dtype == dw_dtype
            assert dx.shape == dz.shape == x.shape and dw.shape == (D,)
            main = out(0.25)
            backward(dy, x, z, w, eps, grad_weight=main, accumulate_grad_weight=True)
            got = dict(zip(JUDGED, (y, dx, dz, dw, main), strict=True))
            if first is None:
                first = got
                launched.append(got)
                continue
            for what in repeated:
                assert torch.equal(got[what], first[what]), (
                    f"{what}: launch {i} vs the first launch"
                )
            if varies:
                launched.append({"dw": dw, "accumulated dw": main})
        if j == 0:
            first_draw["inputs"] = (x, z, w, dy)
            first_draw["outputs"] = tuple(first[k] for k in ("y", "dx", "dz", "dw"))
        return silu_draw(
            x,
            z,
            w,
            dy,
            eps,
            [{label + k: v for k, v in got.items()} for got in launched],
            {label + k: v for k, v in JUDGED.items()},
            starts={label + "accumulated dw": torch.full((D,), 0.25, dtype=dw_dtype)},
        )

    judge(record_property, run, seed=seed)
    return first_draw["inputs"], first_draw["outputs"]


def merge_draws(draws):
    """One :class:`Draw` of several (outputs of distinct names): their
    launches, oracles and naive values together."""

    def naive_values():
        found = {}
        for draw in draws:
            found.update(draw.naive())
        return found

    expected, launched = {}, []
    for draw in draws:
        expected.update(draw.expected)
        launched += list(draw.launches)
    return Draw(launched, expected, naive_values)


def _launch_count(repeats=REPEATS):
    """The launches of each module ``check_all`` makes: ``repeats`` on its
    first draw, one on each other draw."""
    return repeats + DRAWS - 1


@pytest.mark.parametrize("T,H", SHAPES)
def test_forward_backward_shapes(T, H, record_property):
    check_all(record_property, lambda s: make(T, H, seed=s))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layouts(layout, record_property):
    check_all(record_property, lambda s: make(65, 48, layout=layout, seed=s), seed=1)


# Inputs in the bands of the backwards past the small-row bands
# (band, layout), at the fewest tokens of 64 heads in the band on this device
# (``_bulk_band_shape``; ``one_token``: one token of as many heads as the
# band's first rows).  On sm_100a and sm_103a the claimed backward
# (``bwd_bf16_mid``) and the row-claims mid backward (``bwd_bf16_rc_mid``)
# warm a block's rows in L2 with one bulk prefetch per tensor only where dy,
# x and z share one packed row stride (head stride 128, token stride
# H * 128); the other layouts here (``z`` and ``dy`` the gate columns of
# wider projections, ``x`` and ``z`` head-major) fall back to a prefetch per
# thread.  No band runs the row-claims and the claimed backwards
# (``bwd_bf16_rc``, ``bwd_bf16_rc_mid``, ``bwd_bf16_mid``): their cases skip.
# One token of many heads (its token stride passed as 0) and contiguous tokens
# are packed rows.  The static backward of an FP32 weight gradient
# (``bwd_f32``, sm_100a and sm_103a; no band of sm_107a: its cases skip) and
# the row-claims backward (``bwd_bf16_rc``) warm no rows;
# the fused and the repeatable backwards (``bwd_bf16_fused``,
# ``bwd_bf16_rep``, ``bwd_f32_rep``) warm the next block of a claim on every
# architecture (one bulk prefetch per tensor for packed rows, else a prefetch
# per thread); all run the same checks.  The repeatable backwards' long roles
# (``bwd_bf16_rep_long``, ``bwd_f32_rep_long``, which look ahead along a
# claim) run packed rows only: their bands take the contiguous input alone.
BULK_BANDS = (
    "bwd_f32",
    "bwd_bf16_rc",
    "bwd_bf16_rc_mid",
    "bwd_bf16_mid",
    "bwd_bf16_fused",
    "bwd_bf16_rep",
    "bwd_bf16_rep_long",
    "bwd_f32_rep",
    "bwd_f32_rep_long",
)
BULK_BAND_CASES = [
    pytest.param(band, layout, id=f"{band}-{layout}")
    for band in BULK_BANDS
    for layout in ("projection", "head_major", "one_token", "contiguous")
]


def _bulk_band_shape(band: str, layout: str) -> tuple[int, int, str] | None:
    """``(T, H, make's layout)`` of the ``BULK_BAND_CASES`` case ``(band,
    layout)`` on this device: the fewest tokens (at least 2) of 64 heads whose
    rows lie in the ``BACKWARD_BANDS`` band of the backward role ``band`` (or
    of its long role, :func:`_device_band`), or for ``one_token`` one
    contiguous token of the band's first rows of heads; ``None`` where this
    device routes no such band or no such shape lies in it (a long role's
    band holds packed rows only, :func:`_backward_packed`)."""
    found = _device_band(band)
    if found is None:
        return None
    first, last = found
    if layout == "one_token":
        T, H, layout = 1, first, "contiguous"
    else:
        T, H = max(2, -(-first // 64)), 64
        if last is not None and last < T * H:
            return None
    if band.endswith(loader.LONG) and not _backward_packed(T, H, layout):
        return None
    return T, H, layout


@pytest.mark.parametrize("band,layout", BULK_BAND_CASES)
def test_layouts_in_the_row_claims_and_claimed_backward_bands(
    band, layout, launches, record_property
):
    """``BULK_BAND_CASES``: the band's program runs the rows
    (``module_for_rows``), every output judged against the naive
    implementation, ``y``, ``dx`` and ``dz`` (and ``dw`` where the band repeats
    it) the same bits on every launch (``check_all``), and ``dx`` and ``dz``
    the bits of the same rows packed as ``[T * H, 1, 128]`` (which the bulk
    prefetch warms on sm_100a and sm_103a; sm_107a's programs take no bulk
    prefetch and run the same checks), so is ``dw`` where the band repeats
    it."""
    shape = _bulk_band_shape(band, layout)
    if shape is None:
        pytest.skip(f"this device routes no {band} band with a {layout} input")
    T, H, layout = shape
    rows = T * H
    dw_dtype = torch.float32 if band.startswith("bwd_f32") else torch.bfloat16
    name = _backward_module(rows, dw_dtype, packed=_backward_packed(T, H, layout))
    assert name == _routes()[band], (rows, name)
    (x, z, w, dy), (_, dx, dz, dw) = check_all(
        record_property,
        lambda s: make(T, H, layout=layout, seed=s),
        dw_dtype=dw_dtype,
        seed=90,
    )
    fwd = _forward_module(rows, _packed(T, H, layout))
    assert launches == [fwd, name, name] * _launch_count(), launches
    gw = None
    if dw_dtype != torch.bfloat16:
        gw = torch.empty(D, dtype=dw_dtype, device=x.device)
    packed = [t.contiguous().view(rows, 1, D) for t in (dy, x, z)]
    again = backward(*packed, w, EPS, grad_weight=gw)
    for what, a, b in zip(("dx", "dz"), again[:2], (dx, dz), strict=True):
        assert torch.equal(a.reshape(-1, D), b.reshape(-1, D)), (what, layout)
    assert_dw_repeats(again[2], dw, varies=_varies(name), what=f"packed dw ({layout})")


@pytest.mark.parametrize("data", EDGE_DATA)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_edge_values(data, dw_dtype, record_property):
    """``EDGE_DATA`` at 37 tokens of 48 heads (1776 rows: past the small-row
    bands, in a band of each weight-gradient dtype that runs on a workspace,
    for an FP32 weight gradient its first repeatable band), judged by
    ``check_all`` into a ``dw_dtype`` weight gradient."""
    check_all(
        record_property,
        lambda s: make(37, 48, data=data, seed=s),
        dw_dtype=dw_dtype,
        seed=2,
    )


@pytest.mark.parametrize("eps", EPSILONS)
def test_epsilon(eps, record_property):
    check_all(record_property, lambda s: make(19, 48, seed=s), eps, seed=3)


@pytest.mark.parametrize("T", ISSUE_TOKENS)
def test_issue_shapes(T, record_property):
    check_all(record_property, lambda s: make(T, 48, seed=s), seed=4)


@pytest.mark.parametrize("T,H", FP32_SHAPES)
def test_fp32_weight_gradient(T, H, record_property):
    check_all(
        record_property,
        lambda s: make(T, H, seed=s),
        dw_dtype=torch.float32,
        seed=5,
    )


def test_weight_gradient_accumulation(record_property):
    """Two backwards accumulating into one FP32 or BF16 ``grad_weight`` that
    starts at 0.25, three times: every repeat judged against the naive
    implementation's BF16 sum of the starting value and the two gradients,
    and the same bits on every repeat but for a BF16 accumulator in a fused
    or a row-claims band (:func:`_dw_varies`); autograd's ``.grad`` over two
    microbatches the sums of the functional backward's gradients (``dw``
    where its band repeats it), its weight gradient judged against the naive
    implementation's BF16 sum of the two gradients."""

    def run(j, s):
        x, z, w, dy = make(1031, 48, seed=s)
        x2, z2, _, dy2 = make(1031, 48, seed=s + 1)
        varies = _dw_varies(1031 * 48)
        launched = []
        for dtype, name in (
            (torch.float32, "accumulated FP32 dw"),
            (torch.bfloat16, "accumulated BF16 dw"),
        ):
            results = []
            for _ in range(3):
                main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
                for args in ((dy, x, z), (dy2, x2, z2)):
                    backward(
                        *args, w, EPS, grad_weight=main, accumulate_grad_weight=True
                    )
                results.append(main)
            if dtype == torch.float32 or not varies:
                for other in results[1:]:
                    assert torch.equal(other, results[0]), dtype
            launched += [{name: main} for main in results]
        # autograd accumulation into .grad over two microbatches: the sums of
        # the functional backward's gradients
        xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
        gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        dx1, dz1, dw1 = backward(dy, x, z, w, EPS)
        assert torch.equal(xa.grad, dx1 + dx1)
        assert torch.equal(za.grad, dz1 + dz1)
        if not varies:
            assert torch.equal(wa.grad, dw1 + dw1)
        launched.append({"autograd dw": wa.grad})
        first = silu_draw(x, z, w, dy, EPS, [], {"dw": "dw"})
        second = silu_draw(x2, z2, w, dy2, EPS, [], {"dw": "dw"})
        both = 0.25 + first.expected["dw"] + second.expected["dw"]
        expected = {
            "accumulated FP32 dw": both,
            "accumulated BF16 dw": both,
            "autograd dw": 2 * first.expected["dw"],
        }

        def naive_values():
            one, two = first.naive()["dw"], second.naive()["dw"]
            accumulated = (torch.full_like(one, 0.25) + one) + two
            return {
                "accumulated FP32 dw": accumulated,
                "accumulated BF16 dw": accumulated,
                "autograd dw": one + one,
            }

        return Draw(launched, expected, naive_values)

    judge(record_property, run, seed=6)


def _nodw_claimed_rows() -> int | None:
    """Rows of one head a row past the largest strided cut of this device
    (``NODW_STRIDED_ROWS_PER_SM``), where a call without the weight gradient
    runs a claimed program (on a workspace); ``None`` where it does not or
    the device's architecture has no cut."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sms = loader.device_sm_count(device_index)
    cuts = loader.NODW_STRIDED_ROWS_PER_SM[arch]
    if not cuts:
        return None
    rows = max(cuts.values()) * sms + 1
    nodw = _nodw_module(_backward_module(rows), rows)
    if loader.MODULES[nodw]["role"] not in loader.NODW_ROLES:
        return None
    return rows if loader.workspace_bytes(nodw) else None


@pytest.mark.parametrize("program", ["small", "strided", "claimed"])
def test_compute_grad_weight_false_touches_no_weight_state(
    program, monkeypatch, launches
):
    """``compute_grad_weight=False`` returns ``None`` for the weight gradient,
    leaves ``grad_weight`` as it is and runs the program ``nodw_module``
    names, whose ``dx`` and ``dz`` are the bits of the BF16 call that
    computes it (of the BF16 call on the repeatable backward for the build of
    packed rows only, :func:`_nodw_bits`).  At 257 x 48 packed rows the build
    of packed rows only where its cut holds them (sm_100a and sm_103a), which
    makes no stream workspace and leaves a caller workspace zero; at the
    fewest tokens of 48 heads past that cut, at least 257 (up to the strided
    cut of the band on every product that has one), the strided program,
    which does the same; a row past the largest cut
    (:func:`_nodw_claimed_rows`) a claimed program, which takes the arrival
    ticket and claims row blocks on the stream's workspace of its own (its
    buffers, left zero) and leaves a caller workspace zero.  Autograd with a
    weight that needs no gradient gives the same ``dx`` and ``dz``."""
    if program == "small":
        T, H = 257, 48
        if T * H > _small_cut_rows():
            pytest.skip("no build of packed rows only runs 257 x 48 rows here")
    elif program == "strided":
        T, H = _past_small_cut(257, 48), 48
    else:
        rows = _nodw_claimed_rows()
        if rows is None:
            pytest.skip("no claimed program without the weight gradient runs here")
        T, H = rows, 1
    rows = T * H
    nodw = _nodw_module(_backward_module(rows), rows)
    assert loader.MODULES[nodw]["role"] in loader.NODW_ROLES, nodw
    assert _is_small(nodw) == (program == "small"), nodw
    assert bool(loader.workspace_bytes(nodw)) == (program == "claimed"), nodw
    x, z, w, dy = make(T, H, seed=12)
    dx_d, dz_d = _nodw_bits(nodw, dy, x, z, w)
    ws = workspace_for(rows, x.device)
    main = torch.full((D,), 3.0, dtype=torch.float32, device="cuda")
    launches.clear()
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert launches == [nodw], launches
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 3.0))
    assert not bool(ws.any())
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    torch.cuda.synchronize()
    if program != "claimed":
        assert pkg._STREAM_WORKSPACES == {}
    else:
        key = (
            torch.cuda.current_device(),
            torch.cuda.current_stream().cuda_stream,
            nodw,
        )
        assert list(pkg._STREAM_WORKSPACES) == [key]
        views = pkg._STREAM_WORKSPACES[key]
        layout = loader.workspace_layout(nodw)
        assert layout and sorted(views) == sorted(b for b, _, _ in layout), key
        # (through bytes: uint64 tensors take few operations)
        assert not any(bool(view.view(torch.uint8).any()) for view in views.values())
    # autograd with a weight that needs no gradient
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    gated_rmsnorm(xa, za, w, EPS).backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert w.grad is None


# Shapes and ``make`` layouts whose BF16 bands run a program without the
# weight gradient that takes no workspace: at 257 x 48 rows on packed rows
# the build of packed rows only (sm_100a and sm_103a; on sm_107a the
# 128-thread strided program) and on a gate and ``grad_out`` sliced from
# projections a strided program (on sm_107a the 256-thread one); at 5 x 48
# rows (a small-row band on every product) on packed rows the build of
# packed rows only (sm_100a and sm_103a; on sm_107a one cluster) and on
# projections one cluster.
NODW_WORKSPACE_FREE_SHAPES = [
    (257, 48, "contiguous"),
    (257, 48, "projection"),
    (5, 48, "contiguous"),
    (5, 48, "projection"),
]


@pytest.mark.parametrize("T,H,layout", NODW_WORKSPACE_FREE_SHAPES)
def test_programs_without_a_workspace_make_none(T, H, layout, monkeypatch, launches):
    """A call with ``compute_grad_weight=False`` on ``make(T, H,
    layout=layout)`` whose program takes no workspace (a strided or a
    one-cluster program without the weight gradient) makes no stream
    workspace and takes no reserve workspace, eagerly and in CUDA Graph
    capture, whose graph holds its one kernel (no fill kernel, memset or
    event wait); a caller workspace filled with 7 is checked and left as it
    is.  Every call and every replay gives the ``dx`` and ``dz`` bits of the
    BF16 call that computes the weight gradient (of the BF16 call on the
    repeatable backward for the build of packed rows only, :func:`_nodw_bits`)."""
    pytest.importorskip("cuda.bindings.runtime")
    rows = T * H
    packed = _backward_packed(T, H, layout)
    nodw = _nodw_module(
        _backward_module(rows, packed=packed), rows, packed=_nodw_packed(T, H, layout)
    )
    assert loader.MODULES[nodw]["role"] in loader.NODW_ROLES, nodw
    assert loader.workspace_bytes(nodw) == 0, nodw
    x, z, w, dy = make(T, H, layout=layout, seed=13)
    dx_d, dz_d = _nodw_bits(nodw, dy, x, z, w)
    ws = workspace_for(rows, x.device)
    ws.fill_(7)
    torch.cuda.synchronize()
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    launches.clear()
    for workspace in (None, ws, None):
        dx, dz, dw = backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=workspace
        )
        assert dw is None
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), workspace is None
    assert launches == [nodw] * 3, launches
    torch.cuda.synchronize()
    assert bool((ws == 7).all()), "a program without a workspace wrote the workspace"
    assert pkg._STREAM_WORKSPACES == {} and pkg._RESERVES == {}
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = [
            backward(dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws_)
            for ws_ in (None, ws)
        ]
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 2, types
    assert pkg._STREAM_WORKSPACES == {} and pkg._RESERVES == {}
    assert pkg._CAPTURE_WORKSPACES == {}
    for k in range(3):
        for dx, dz, _ in outs:
            dx.zero_()
            dz.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for i, (dx, dz, dw) in enumerate(outs):
            assert dw is None
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), (k, i)
    assert bool((ws == 7).all()), "a program without a workspace wrote the workspace"


# The SM counts at which the programs without the weight gradient are
# compared with the BF16 call around their strided cut (``device_sm_count``
# patched; ``0`` is the device's own).
NODW_CUT_SM_COUNTS = (0, 16, 66)


@pytest.mark.parametrize("layout", ["flat", "projection"])
@pytest.mark.parametrize("sm_count", NODW_CUT_SM_COUNTS)
@pytest.mark.parametrize("band", NODW_CUT_BANDS)
def test_programs_without_the_weight_gradient_give_the_bits_at_the_cut(
    band, sm_count, layout, launches
):
    """For each band with a strided program (``NODW_CUT_BANDS``, its module
    forced through ``module_for_rows``, :func:`_forcing`), at ``cut - 1``
    and ``cut`` rows per SM, a row past the cut and ``cut + 1`` rows per SM
    (``NODW_STRIDED_ROWS_PER_SM``; ``device_sm_count`` patched to
    ``sm_count``), on packed rows and on a gate and ``grad_out`` sliced from
    projections: a call with ``compute_grad_weight=False`` runs the band's
    strided program up to the cut (on packed rows its strided program of
    packed rows where routed) and its claimed program past it (where
    routed, :func:`nodw_module`), gives the ``dx`` and ``dz`` bits of the BF16
    call on the band's module and leaves a caller workspace zero.  A band
    without a cut (no claimed program routed) runs its strided program
    instead at a row per SM fewer than the repeatable band's long cut
    (``LONG_BACKWARD_ROWS_PER_SM``) and at a row fewer than it."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    routes = loader.ROUTES[arch]
    if band + loader.NODW + loader.STRIDED not in routes:
        pytest.skip(f"{arch} routes no strided program of {band}")
    sm_count = sm_count or loader.device_sm_count(device_index)
    cut = loader.NODW_STRIDED_ROWS_PER_SM[arch].get(band)
    if cut is None:
        last = loader.LONG_BACKWARD_ROWS_PER_SM[arch]["bwd_bf16_rep"]
        row_counts = ((last - 1) * sm_count, last * sm_count - 1)
    else:
        row_counts = (
            (cut - 1) * sm_count,
            cut * sm_count,
            cut * sm_count + 1,
            (cut + 1) * sm_count,
        )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "device_sm_count", lambda index: sm_count)
        patch.setattr(pkg, "module_for_rows", _forcing(band))
        for rows in row_counts:
            what = f"{band}: {rows} rows on {sm_count} SMs ({layout})"
            x, z, w, dy = make(rows, 1, layout=layout, seed=14)
            ws = workspace_for(rows, x.device)
            launches.clear()
            dx_d, dz_d, _ = backward(dy, x, z, w, EPS, workspace=ws)
            dx, dz, dw = backward(
                dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
            )
            packed = layout == "flat"
            nodw = loader.nodw_module(
                arch, routes[band], rows, sm_count, packed=packed
            )
            if cut is None or rows <= cut * sm_count:
                strided = routes[band + loader.NODW + loader.STRIDED]
                if packed:
                    strided = routes.get(
                        band + loader.NODW + loader.STRIDED + loader.PACKED, strided
                    )
                assert nodw == strided, what
            assert launches == [routes[band], nodw], what
            assert dw is None
            assert torch.equal(dx, dx_d), f"{what}: dx"
            assert torch.equal(dz, dz_d), f"{what}: dz"
            torch.cuda.synchronize()
            assert not bool(ws.any()), f"the workspace after {what}"


# The programs without the weight gradient that run at a forced grid:
# ``(band, program, layout)``, the build of packed rows only (packed rows,
# contiguous, up to its cut), the strided one (on packed rows past that cut,
# contiguous, its strided program of packed rows where routed; on a gate and
# ``grad_out`` sliced from projections the strided one) or (the strided ones
# and the build of packed rows only unrouted) the claimed one of the band.
NODW_GRID_CASES = [
    pytest.param(band, program, layout, id=f"{band}-{program}-{layout}")
    for band in NODW_CUT_BANDS
    for program, layout in (
        ("small", "contiguous"),
        ("strided", "contiguous"),
        ("strided", "projection"),
        ("claimed", "contiguous"),
    )
]


@pytest.mark.parametrize("ctas", [1, 3, 7])
@pytest.mark.parametrize("band,program,layout", NODW_GRID_CASES)
def test_programs_without_the_weight_gradient_at_forced_grids(
    band, program, layout, ctas, launches, monkeypatch
):
    """On grids of ``ctas`` CTAs (``backward_ctas`` patched), fewer than the
    row blocks of 257 x 48 rows (of the fewest tokens of 48 heads past the
    cut of the build of packed rows only, at least 257, for the strided
    program on packed rows), the build of packed rows only (packed rows up to
    its cut, where routed), the band's strided program (on packed rows its
    strided program of packed rows where routed, on a gate and ``grad_out``
    sliced from projections the strided one) and its claimed program (the
    strided ones and the build of packed rows only unrouted, ``ROUTES``
    patched), the band's module forced through ``module_for_rows``
    (:func:`_forcing`), give the ``dx`` and ``dz`` bits of the BF16 call on
    the band's module at the host's grid (on the repeatable backward for the
    build of packed rows only, :func:`_rep_bits`); the claimed program leaves
    a caller workspace zero after back-to-back calls, and the strided ones do
    not touch it."""
    arch = loader.device_arch(torch.cuda.current_device())
    routes = loader.ROUTES[arch]
    strided = band + loader.NODW + loader.STRIDED
    strided_packed = strided + loader.PACKED
    claimed = band + loader.NODW
    T, H = 257, 48
    if program == "small":
        own = strided in routes or claimed in routes
        want = routes.get(NODW_SMALL) if own and T * H <= _small_cut_rows() else None
    elif program == "claimed":
        want = routes.get(claimed)
    elif layout == "contiguous" and strided in routes:
        T = _past_small_cut(T, H)
        want = routes.get(strided_packed, routes[strided])
    else:
        want = routes.get(strided)
    if want is None:
        pytest.skip(f"{arch} routes no {program} program of {band} at these rows")
    x, z, w, dy = make(T, H, layout=layout, seed=15)
    ws = workspace_for(T * H, x.device)
    if program == "small":
        dx_d, dz_d = _rep_bits(dy, x, z, w)
    monkeypatch.setattr(pkg, "module_for_rows", _forcing(band))
    if program != "small":
        dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    if program == "claimed":
        drop = (strided, strided_packed, NODW_SMALL)
        table = {r: name for r, name in routes.items() if r not in drop}
        _patch_routes(monkeypatch, arch, table)
    monkeypatch.setattr(pkg, "backward_ctas", lambda name, rows, sm_count: ctas)
    launches.clear()
    for i in range(3):
        dx, dz, dw = backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
        )
        assert dw is None
        assert torch.equal(dx, dx_d), f"{want} on {ctas} CTAs, call {i}: dx"
        assert torch.equal(dz, dz_d), f"{want} on {ctas} CTAs, call {i}: dz"
    assert launches == [want] * 3, launches
    torch.cuda.synchronize()
    assert not bool(ws.any()), f"the workspace after {want} on {ctas} CTAs"


@pytest.mark.parametrize("band", ["bwd_bf16_short", "bwd_bf16_t512"])
def test_one_cluster_programs_without_the_weight_gradient(band, launches):
    """In each small-row band of this device (the first, a middle and the
    last rows of the band, as tokens of one head, at least 2: one token runs
    as packed rows; skipped where this device routes no such band or no
    program without the weight gradient for it), on a gate and ``grad_out``
    sliced from projections (rows that are not packed, which the build of
    packed rows only never runs), a call with ``compute_grad_weight=False``
    runs the band's one-cluster program without the weight gradient and
    gives the ``dx`` and ``dz`` bits of the BF16 call on the band's
    one-cluster module."""
    found = _device_band(band)
    nodw = _routes().get(band + loader.NODW)
    if found is None or nodw is None:
        pytest.skip(f"this device routes no {band} band without the weight gradient")
    first, last = found
    name = _routes()[band]
    for rows in sorted({max(2, first), (first + last) // 2, last}):
        x, z, w, dy = make(rows, 1, layout="projection", seed=17)
        launches.clear()
        dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
        assert launches == [name, nodw], (rows, launches)
        assert dw is None
        assert torch.equal(dx, dx_d), f"{band} at {rows} rows: dx"
        assert torch.equal(dz, dz_d), f"{band} at {rows} rows: dz"


# The layouts of ``BULK_BAND_CASES`` (64 heads: a gate and ``grad_out``
# sliced from projections, head-major ``x`` and ``z``, contiguous tokens; one
# token of many heads).
NODW_LAYOUTS = ("projection", "head_major", "one_token", "contiguous")
# The BF16 bands of a call with ``compute_grad_weight=False`` and the layouts
# of ``NODW_LAYOUTS`` (:func:`_nodw_layout_shape`; a long role's band packed
# rows only).
NODW_LAYOUT_CASES = [
    pytest.param(band, layout, id=f"{band}-{layout}")
    for band in (
        "bwd_bf16_short",
        "bwd_bf16_t512",
        "bwd_bf16_fused",
        "bwd_bf16_rep",
        "bwd_bf16_rep_long",
    )
    for layout in NODW_LAYOUTS
]


def _nodw_layout_shape(band: str, layout: str) -> tuple[int, int, str] | None:
    """``(T, H, make's layout)`` of the ``NODW_LAYOUT_CASES`` case ``(band,
    layout)`` on this device: :func:`_bulk_band_shape`, but ``one_token``
    takes at least 64 heads (the small-row band's first rows are one row);
    ``None`` where this device routes no such band or no such shape lies in
    it."""
    shape = _bulk_band_shape(band, layout)
    if shape is None or layout != "one_token":
        return shape
    T, H, layout = shape
    _, last = _device_band(band)
    H = max(H, 64)
    return None if last is not None and last < H else (T, H, layout)


def _check_nodw_layout(band: str, T: int, H: int, layout: str, nodw: str, launches):
    """A call with ``compute_grad_weight=False`` on ``make(T, H,
    layout=layout)`` runs ``nodw`` after the BF16 call on the band's module
    (``launches``), gives that call's ``dx`` and ``dz`` bits (the BF16 call's
    on the repeatable backward for the build of packed rows only,
    :func:`_rep_bits`) and leaves a caller workspace zero."""
    rows = T * H
    name = _routes()[band]
    x, z, w, dy = make(T, H, layout=layout, seed=91)
    ws = workspace_for(rows, x.device)
    if _is_small(nodw):
        # the build of packed rows only: the repeatable backward's bits
        dx_r, dz_r = _rep_bits(dy, x, z, w, workspace=ws)
    launches.clear()
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS, workspace=ws)
    if _is_small(nodw):
        dx_d, dz_d = dx_r, dz_r
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws)
    what = f"{band} ({layout}, {T} x {H})"
    assert launches == [name, nodw], (what, launches)
    assert dw is None
    assert torch.equal(dx, dx_d), f"{what}: dx"
    assert torch.equal(dz, dz_d), f"{what}: dz"
    torch.cuda.synchronize()
    assert not bool(ws.any()), f"the workspace after {what}"


@pytest.mark.parametrize("band,layout", NODW_LAYOUT_CASES)
def test_programs_without_the_weight_gradient_on_strided_layouts(
    band, layout, launches
):
    """``NODW_LAYOUT_CASES`` at the band's first rows of 64 heads (one token
    of at least 64 heads) on this device: a call with
    ``compute_grad_weight=False`` runs the program ``nodw_module`` names for
    the band's module (one cluster, strided, claimed or look-ahead; the
    strided program of packed rows on the contiguous ones where routed; the
    build of packed rows only on packed rows and one token up to its cut;
    skipped where this device routes no such band or shape), gives the
    ``dx`` and ``dz`` bits of the BF16 call on the band's module on the same
    inputs (on the repeatable backward for the build of packed rows only)
    and leaves a caller workspace zero."""
    shape = _nodw_layout_shape(band, layout)
    if shape is None:
        pytest.skip(f"this device routes no {band} band with a {layout} input")
    T, H, layout = shape
    rows = T * H
    packed = _backward_packed(T, H, layout)
    name = _backward_module(rows, packed=packed)
    assert name == _routes()[band], (rows, name)
    nodw = _nodw_module(name, rows, packed=_nodw_packed(T, H, layout))
    assert loader.MODULES[nodw]["role"] in loader.NODW_ROLES, (band, nodw)
    _check_nodw_layout(band, T, H, layout, nodw, launches)


@pytest.mark.parametrize("layout", NODW_LAYOUTS)
def test_claimed_programs_without_the_weight_gradient_on_strided_layouts(
    layout, launches
):
    """For each band with a cut on this device (``NODW_STRIDED_ROWS_PER_SM``:
    its strided program up to the cut, its claimed program past it), the
    ``NODW_LAYOUTS`` layout just past the cut (the fewest tokens of 64 heads
    past it; one token of a head past it, a row past the cut): a call with
    ``compute_grad_weight=False`` runs the band's
    claimed program without the weight gradient, gives the ``dx`` and ``dz``
    bits of the BF16 call on the band's module on the same inputs and leaves
    a caller workspace zero (skipped where this device has no cut)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    cuts = loader.NODW_STRIDED_ROWS_PER_SM[arch]
    if not cuts:
        pytest.skip(f"{arch} routes no claimed program without the weight gradient")
    sms = loader.device_sm_count(device_index)
    for band, cut in sorted(cuts.items()):
        if layout == "one_token":
            T, H, make_layout = 1, cut * sms + 1, "contiguous"
        else:
            T, H, make_layout = cut * sms // 64 + 1, 64, layout
        rows = T * H
        packed = _backward_packed(T, H, make_layout)
        name = _backward_module(rows, packed=packed)
        assert name == _routes()[band], (band, rows, name)
        nodw = _nodw_module(name, rows, packed=_nodw_packed(T, H, make_layout))
        assert nodw == _routes()[band + loader.NODW], (band, rows, nodw)
        assert loader.workspace_bytes(nodw), (band, nodw)
        _check_nodw_layout(band, T, H, make_layout, nodw, launches)


# The band roles of the backwards that run on a workspace: the static
# backwards (the call roles: ``bwd_f32`` on sm_100a and sm_103a; ``bwd_bf16``,
# which no band runs: skipped), the row-claims backwards and the claimed
# backward of a BF16 weight gradient (no band runs them: skipped), the fused
# backward, the repeatable backwards of both weight-gradient dtypes and their
# long roles (on their packed rows).
GRID_BANDS = (
    "bwd_bf16",
    "bwd_f32",
    "bwd_bf16_rc",
    "bwd_bf16_rc_mid",
    "bwd_bf16_mid",
    "bwd_bf16_fused",
    "bwd_bf16_rep",
    "bwd_bf16_rep_long",
    "bwd_f32_rep",
    "bwd_f32_rep_long",
)


@pytest.mark.parametrize("ctas", [1, 3, 7])
@pytest.mark.parametrize("band", GRID_BANDS)
def test_backward_at_grids_smaller_than_the_chunks(
    band, ctas, launches, monkeypatch, record_property
):
    """The host's grid forced to ``ctas`` CTAs for the fewest rows of the
    ``BACKWARD_BANDS`` band of ``band`` (or of its long role, packed rows) on
    this device (:func:`_band_shape`; skipped where this device routes no
    such band), so every CTA runs many
    chunks or row blocks: ``dx`` and ``dz`` the bits of the host's grid; the
    weight gradient of the band's dtype, alone and accumulated, the bits of
    the host's grid where the band repeats them (:func:`assert_dw_repeats`;
    the repeatable backward's, whose chunks follow the grid, the bits of a
    second launch at the same grid instead), and judged against the naive
    implementation at the forced grid over the draws of ``judge``;
    ``compute_grad_weight=False`` (the program without the weight gradient
    ``nodw_module`` names for the BF16 bands' module, at the forced grid)
    leaves ``grad_weight`` as it is, ``dx`` and ``dz`` the bits of the BF16
    bands' module at the host's grid; the workspace is zero after every
    launch."""
    shape = _band_shape(band)
    if shape is None:
        pytest.skip(f"this device routes no {band} band")
    T, H = shape
    rows = T * H
    dtype = torch.float32 if band.startswith("bwd_f32") else torch.bfloat16
    name = _backward_module(rows, dtype)
    assert name == _routes()[band], (rows, name)
    varies = _varies(name)
    x, z, w, dy = make(T, H, seed=12)
    dw_d = torch.empty(D, dtype=dtype, device="cuda")
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS, grad_weight=dw_d)
    main_d = torch.full((D,), 0.25, dtype=dtype, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main_d, accumulate_grad_weight=True)
    # compute_grad_weight=False runs the program without the weight gradient
    # of the BF16 bands' module, whose dx and dz are another program's where
    # that is not the band's module (an FP32 band's rows in the fused band)
    # or where it is the build of packed rows only (the repeatable
    # backward's): the bits of its call at the host's grid
    nodw = _nodw_module(_backward_module(rows), rows)
    dx_n, dz_n = dx_d, dz_d
    if _backward_module(rows) != name or _is_small(nodw):
        dx_n, dz_n, _ = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    ws = workspace_for(rows, x.device)
    monkeypatch.setattr(pkg, "backward_ctas", lambda name, rows, sm_count: ctas)
    launches.clear()
    what = f"{band} at {ctas} CTAs"
    dw = torch.empty(D, dtype=dtype, device="cuda")
    dx, dz, _ = backward(dy, x, z, w, EPS, grad_weight=dw, workspace=ws)
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), what
    if _rep(name):
        # the bits of the forced grid: a second launch at it
        dw_d = torch.empty(D, dtype=dtype, device="cuda")
        backward(dy, x, z, w, EPS, grad_weight=dw_d, workspace=ws)
        main_d = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main_d,
            accumulate_grad_weight=True,
            workspace=ws,
        )
    assert_dw_repeats(dw, dw_d, varies=varies, what=f"dw {what}")
    assert not bool(ws.any()), what
    main = torch.full((D,), 0.25, dtype=dtype, device="cuda")
    backward(
        dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True, workspace=ws
    )
    assert_dw_repeats(main, main_d, varies=varies, what=f"accumulated dw {what}")
    assert not bool(ws.any()), what
    main = torch.full((D,), 3.0, dtype=dtype, device="cuda")
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert dw is None
    assert torch.equal(dx, dx_n) and torch.equal(dz, dz_n), what
    assert torch.equal(main, torch.full_like(main, 3.0)), what
    assert not bool(ws.any()), what
    # a call with compute_grad_weight=False runs the program without the
    # weight gradient of the BF16 bands' module whatever the dtype of
    # grad_weight (it writes no weight gradient)
    repeats = 4 if _rep(name) else 2
    assert launches == [name] * repeats + [nodw], launches

    # the weight gradient at the forced grid, alone and accumulated, on the
    # inputs of each draw
    def run(j, s):
        xj, zj, wj, dyj = make(T, H, seed=s)
        dwj = torch.empty(D, dtype=dtype, device="cuda")
        backward(dyj, xj, zj, wj, EPS, grad_weight=dwj, workspace=ws)
        mainj = torch.full((D,), 0.25, dtype=dtype, device="cuda")
        backward(
            dyj,
            xj,
            zj,
            wj,
            EPS,
            grad_weight=mainj,
            accumulate_grad_weight=True,
            workspace=ws,
        )
        assert not bool(ws.any()), (what, j)
        return silu_draw(
            xj,
            zj,
            wj,
            dyj,
            EPS,
            [{"dw": dwj, "accumulated dw": mainj}],
            {"dw": "dw", "accumulated dw": "dw"},
            starts={"accumulated dw": torch.full((D,), 0.25, dtype=dtype)},
        )

    judge(record_property, run, seed=12)


@pytest.mark.parametrize("shape", [(0, 48, D), (5, 0, D), (0, D)])
def test_empty_work_keeps_the_accumulator(shape, launches):
    x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    ws = workspace_for(0, x.device)
    assert forward(x, x, w).shape == x.shape
    for dtype in (torch.bfloat16, torch.float32):
        main = torch.full((D,), 1.5, dtype=dtype, device="cuda")
        dx, dz, dw = backward(
            x, x, x, w, grad_weight=main, accumulate_grad_weight=True, workspace=ws
        )
        assert dw is main and torch.equal(main, torch.full_like(main, 1.5)), dtype
        assert dx.shape == dz.shape == x.shape
        backward(x, x, x, w, grad_weight=main)
        assert torch.equal(main, torch.zeros_like(main)), dtype
    _, _, dw = backward(x, x, x, w)
    assert dw.dtype == torch.bfloat16 and torch.equal(dw, torch.zeros_like(dw))
    assert launches == []
    assert not bool(ws.any())


def test_workspace_reused_across_shapes_without_reset(record_property):
    """One zero-filled byte workspace of the architecture's
    ``WORKSPACE_BYTES`` (the largest of its backward modules' workspaces, each
    module's layout over the first bytes: a static or a claimed backward's by
    its slots, a row-claims backward's ``ROW_CLAIMS_LAYOUT``, a fused
    backward's ``FUSED_LAYOUT``, the repeatable backward's and its long
    role's ``REP_LAYOUT``) serves every
    backward module and row count: each backward on it leaves it zero, gives
    ``dx`` and ``dz`` the bits of the same call on the stream's own
    workspace, and so the weight gradient where its band repeats it
    (:func:`assert_dw_repeats`); the sequence of calls on the one workspace,
    never reset, runs on the inputs of every draw of ``judge``, each call's
    ``dx``, ``dz`` and weight gradient judged against the naive
    implementation.  The shapes run every
    ``BACKWARD_BANDS`` band of both weight-gradient dtypes on this device
    and every long role's packed rows (the fewest rows of each band no
    listed shape runs, :func:`_range_shape`), the small-row backwards among
    them."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    assert ws.dtype == torch.uint8 and ws.is_contiguous() and not bool(ws.any())
    assert ws.data_ptr() % 16 == 0
    arch = loader.device_arch(torch.cuda.current_device())
    nbytes = workspace_bytes(1, device)
    assert nbytes == WORKSPACE_BYTES[arch], (arch, nbytes)
    assert ws.numel() == nbytes == workspace_bytes(10**7, device)
    largest = 0
    for role, name in loader.ROUTES[arch].items():
        if not role.startswith("bwd"):
            continue
        layout = _expected_layout(arch, role)
        assert loader.workspace_layout(name) == layout, role
        module_bytes = loader.workspace_bytes(name)
        if role in loader.BAND_ROLES:
            assert layout == () and module_bytes == 0, role
        elif role in NODW_TWINS:
            twin = loader.ROUTES[arch][NODW_TWINS[role]]
            assert module_bytes == (loader.workspace_bytes(twin) if layout else 0), role
        elif role in ROW_CLAIMS_GRID_RULES:
            assert module_bytes == ROW_CLAIMS_WORKSPACE_BYTES, role
        elif role in FUSED_ROLES[arch]:
            assert module_bytes == FUSED_WORKSPACE_BYTES, role
        elif role in REP_ROLES[arch]:
            assert module_bytes == REP_WORKSPACE_BYTES, role
        else:
            slots = BACKWARD_SLOTS[arch][role]
            assert module_bytes == MODULE_WORKSPACE_BYTES[slots], role
        largest = max(largest, module_bytes)
    assert largest == nbytes, (arch, largest, nbytes)
    cases = [
        (1031, 48, torch.bfloat16, False),
        (7, 48, torch.float32, False),
        (4097, 16, torch.bfloat16, False),
        (1, 1, torch.float32, True),
        (30193, 48, torch.bfloat16, False),
        (33, 7, torch.float32, True),
        (129, 48, torch.float32, False),
    ]
    bands = {
        band
        for role in ("bwd_bf16", "bwd_f32")
        for band, _, _ in _device_bands(role) + _device_long_bands(role)
    }
    listed = {_device_band_first(T * H, dtype) for T, H, dtype, _ in cases}
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        for _band, first, last in _device_bands(role) + _device_long_bands(role):
            if (role, first) not in listed:
                T, H = _range_shape(first, last)
                cases.append((T, H, dtype, False))
    ran = set()

    def run(j, s):
        draws = []
        for i, (T, H, dtype, accumulate) in enumerate(cases):
            name = _backward_module(T * H, dtype)
            ran.add(loader.MODULES[name]["role"])
            x, z, w, dy = make(T, H, seed=s + i)
            start = 0.5 if accumulate else 0.0
            gw = (
                torch.full((D,), start, dtype=dtype, device=device)
                if accumulate or dtype != torch.bfloat16
                else None
            )
            dx, dz, dw = backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=gw,
                accumulate_grad_weight=accumulate,
                workspace=ws,
            )
            torch.cuda.synchronize()
            assert not bool(ws.any()), (T, H, dtype)
            assert dw.dtype == dtype
            # the same call on the stream's workspace: the same bits (the
            # weight gradient where its band repeats it)
            gw2 = None if gw is None else torch.full_like(gw, start)
            again = backward(
                dy, x, z, w, EPS, grad_weight=gw2, accumulate_grad_weight=accumulate
            )
            for a, b in zip(again[:2], (dx, dz), strict=True):
                assert torch.equal(a, b), (T, H, dtype)
            assert_dw_repeats(
                again[2],
                dw,
                varies=_varies(name),
                what=f"dw on the stream's workspace T={T} H={H} {dtype}",
            )
            label = f"{i} {T}x{H} {str(dtype).removeprefix('torch.')} "
            draws.append(
                silu_draw(
                    x,
                    z,
                    w,
                    dy,
                    EPS,
                    [{label + "dx": dx, label + "dz": dz, label + "dw": dw}],
                    {label + "dx": "dx", label + "dz": "dz", label + "dw": "dw"},
                    starts={label + "dw": torch.full((D,), start, dtype=dtype)},
                )
            )
        return merge_draws(draws)

    judge(record_property, run, seed=20)
    assert ran == bands, (arch, ran, bands)


def _unfit_workspaces(nbytes: int) -> list[torch.Tensor]:
    """Buffers that are not a backward workspace of ``nbytes`` bytes: too
    small, strided, misaligned, of another dtype."""
    big = torch.zeros(nbytes + 64, dtype=torch.uint8, device="cuda")
    return [
        torch.zeros(nbytes - 1, dtype=torch.uint8, device="cuda"),
        torch.zeros(2 * nbytes, dtype=torch.uint8, device="cuda")[::2],
        big[1 : 1 + nbytes],
        big[8 : 8 + nbytes],
        torch.zeros(nbytes, dtype=torch.float64, device="cuda"),
        torch.zeros(nbytes, dtype=torch.int8, device="cuda"),
    ]


def test_caller_workspace_buffers_follow_the_tensor():
    """A caller workspace is checked and viewed by its first call and its
    buffers serve its later calls; once its storage is freed in place and the
    tensor rebound to a new zero-filled workspace (the allocator may give it
    the old address), the next call checks it and makes new buffers, gives
    the ``dx`` and ``dz`` bits of a call on a fresh workspace (and its weight
    gradient where the band repeats it) and leaves the new storage zero.
    Rebound to a buffer one byte short, it is refused and keeps no entry; an
    entry goes with its tensor."""
    T, H = 1031, 48
    name = _backward_module(T * H)
    x, z, w, dy = make(T, H, seed=31)
    want = backward(dy, x, z, w, EPS, workspace=workspace_for(1, x.device))

    def check(got, what):
        for a, b in zip(got[:2], want[:2], strict=True):
            assert torch.equal(a, b), what
        assert_dw_repeats(got[2], want[2], varies=_varies(name), what=what)

    ws = workspace_for(1, x.device)
    for i in range(2):
        check(backward(dy, x, z, w, EPS, workspace=ws), f"call {i}")
    held = pkg._CALLER_WORKSPACES[id(ws)]
    assert held[0]() is ws and held[4].data_ptr() == ws.data_ptr()
    backward(dy, x, z, w, EPS, workspace=ws)
    assert pkg._CALLER_WORKSPACES[id(ws)] is held
    torch.cuda.synchronize()
    ws.untyped_storage().resize_(0)
    ws.set_(workspace_for(1, x.device))
    got = backward(dy, x, z, w, EPS, workspace=ws)
    torch.cuda.synchronize()
    again = pkg._CALLER_WORKSPACES[id(ws)]
    assert again is not held and again[4].data_ptr() == ws.data_ptr()
    check(got, "rebound")
    assert not bool(ws.any())
    del held, again
    nbytes = ws.numel()
    ws.set_(torch.ones(nbytes - 1, dtype=torch.uint8, device=x.device))
    with pytest.raises(ValueError, match="uint8"):
        backward(dy, x, z, w, EPS, workspace=ws)
    assert id(ws) not in pkg._CALLER_WORKSPACES
    assert bool(ws.eq(1).all())
    ws = workspace_for(1, x.device)
    backward(dy, x, z, w, EPS, workspace=ws)
    key = id(ws)
    assert pkg._CALLER_WORKSPACES[key][0]() is ws
    del ws
    gc.collect()
    assert key not in pkg._CALLER_WORKSPACES

def test_workspace_errors():
    x, z, w, dy = make(65, 48, seed=13)
    nbytes = workspace_for(1, x.device).numel()
    # an FP32 buffer (the deterministic package's kind)
    fp32 = torch.zeros(nbytes, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="deterministic mode"):
        backward(dy, x, z, w, EPS, workspace=fp32)
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError, match="uint8"):
            backward(dy, x, z, w, EPS, workspace=bad)
        assert not bool(bad.any())


@pytest.mark.parametrize("role", ["bwd_bf16", "bwd_f32"])
def test_workspace_errors_in_every_band(role, launches):
    """In every ``BACKWARD_BANDS`` band of ``role`` on this device (the
    small-row, the static, the fused and the repeatable backwards, whatever
    part of the workspace the band's module uses) and on the packed rows of
    its long roles (:func:`_device_long_bands`), a caller workspace that is
    not a contiguous ``torch.uint8`` buffer of at least ``workspace_bytes``
    bytes (one byte short, strided, misaligned, of another dtype:
    :func:`_unfit_workspaces`) raises ``ValueError`` naming ``uint8``,
    launches nothing and is left as it is."""
    dw_dtype = torch.float32 if role == "bwd_f32" else torch.bfloat16
    nbytes = workspace_for(1, torch.device("cuda")).numel()
    bands = _device_bands(role)
    assert bands, role
    for band, first, last in bands + _device_long_bands(role):
        shape = _range_shape(first, last)
        x, z, w, dy = make(*shape, seed=17)
        assert _backward_module(shape[0] * shape[1], dw_dtype) == _routes()[band]
        for bad in _unfit_workspaces(nbytes):
            gw = torch.empty(D, dtype=dw_dtype, device="cuda")
            with pytest.raises(ValueError, match="uint8"):
                backward(dy, x, z, w, EPS, grad_weight=gw, workspace=bad)
            torch.cuda.synchronize()
            assert not bool(bad.any()), (band, bad.dtype, bad.numel())
    assert launches == [], launches


def test_reset_workspace_clears_nonzero_scratch(record_property):
    """A caller workspace whose integer windows, class words, arrival ticket
    and claim counter hold garbage (written by something other than the
    backward) is zeroed by ``reset_workspace``; backward launches at two
    different shapes and weight-gradient dtypes then run on it in sequence,
    with the bits of the same calls on a fresh workspace (the weight gradient
    where its band repeats it, :func:`assert_dw_repeats`), and leave it zero.
    The garbage, the reset and the two calls run on the inputs of every draw
    of ``judge``, each call's ``dx``, ``dz`` and weight gradient judged
    against the naive implementation."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    # a stale arrival ticket and a stale claim count, no byte of them zero, in
    # the layout of each backward module of the architecture
    arch = loader.device_arch(torch.cuda.current_device())
    offsets = {
        offset
        for role in loader.ROUTES[arch]
        if role.startswith("bwd")
        for buffer, offset, _ in _expected_layout(arch, role)
        if buffer == "dw_done"
    }
    assert offsets, arch

    def run(j, s):
        g = torch.Generator(device="cuda").manual_seed(s)
        ws.copy_(
            torch.randint(1, 256, ws.shape, device=device, generator=g).to(
                torch.uint8
            )
        )
        for offset in sorted(offsets):
            ws[offset : offset + 8].view(torch.int32).fill_(0x03030303)
        assert bool(ws.ne(0).all())
        reset_workspace(ws)
        torch.cuda.synchronize()
        assert not bool(ws.any())
        # 149 x 7 = 1043 rows: past every small-row backward's band, so both
        # calls use the workspace
        draws = []
        for i, (T, H, dtype) in enumerate(
            ((1031, 48, torch.bfloat16), (149, 7, torch.float32))
        ):
            x, z, w, dy = make(T, H, seed=s + 1 + i)
            gw = (
                None
                if dtype == torch.bfloat16
                else torch.empty(D, dtype=dtype, device=device)
            )
            dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=gw, workspace=ws)
            assert dw.dtype == dtype
            gw2 = None if gw is None else torch.empty_like(gw)
            fresh = backward(
                dy, x, z, w, EPS, grad_weight=gw2, workspace=workspace_for(1, device)
            )
            for a, b in zip(fresh[:2], (dx, dz), strict=True):
                assert torch.equal(a, b), (T, H)
            assert_dw_repeats(
                fresh[2],
                dw,
                varies=_dw_varies(T * H, dtype),
                what=f"dw on a fresh workspace T={T} H={H} {dtype}",
            )
            torch.cuda.synchronize()
            assert not bool(ws.any()), (T, H)
            label = f"{T}x{H} "
            draws.append(
                silu_draw(
                    x,
                    z,
                    w,
                    dy,
                    EPS,
                    [
                        {label + "dx": dx, label + "dz": dz, label + "dw": dw},
                        {label + "dw": fresh[2]},
                    ],
                    {label + "dx": "dx", label + "dz": "dz", label + "dw": "dw"},
                )
            )
        return merge_draws(draws)

    judge(record_property, run, seed=60)


def test_reset_workspace_of_a_zero_workspace_is_a_no_op(launches, record_property):
    """Resetting a zero workspace keeps it zero and in place, launches no module
    and does not synchronize the host; the next backward is unaffected: its
    weight gradient on the workspace, on the inputs of every draw of
    ``judge``, judged against the naive implementation, the workspace zero
    after each."""
    ws = workspace_for(129 * 48, torch.device("cuda"))
    ptr = ws.data_ptr()
    torch.cuda.set_sync_debug_mode("error")
    try:
        reset_workspace(ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert ws.data_ptr() == ptr and not bool(ws.any())
    assert launches == []

    def run(j, s):
        x, z, w, dy = make(129, 48, seed=s)
        _, _, dw = backward(dy, x, z, w, EPS, workspace=ws)
        torch.cuda.synchronize()
        assert not bool(ws.any()), j
        return silu_draw(x, z, w, dy, EPS, [{"dw": dw}], {"dw": "dw"})

    judge(record_property, run, seed=62)


def test_reset_workspace_errors():
    nbytes = workspace_for(1, torch.device("cuda")).numel()
    with pytest.raises(TypeError):
        reset_workspace(None)
    fp32 = torch.zeros(nbytes, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="deterministic mode"):
        reset_workspace(fp32)
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError, match="uint8"):
            reset_workspace(bad)
        assert not bool(bad.any())
    with pytest.raises(ValueError, match="CUDA"):
        reset_workspace(torch.zeros(nbytes, dtype=torch.uint8))
    ws = workspace_for(1, torch.device("cuda"))
    # the mode is in the name: a mode keyword is a TypeError
    for value in (False, True, None):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(
                ws, deterministic=value
            )


def test_two_streams_with_their_own_workspaces(record_property):
    """Two side streams, each with its own workspace, run five backwards of
    their own shape: ``dx`` and ``dz`` the bits of one launch on the current
    stream, so the weight gradient where its band repeats it
    (:func:`assert_dw_repeats`), the workspaces zero after; on the inputs of
    every draw of ``judge``, every launch's weight gradient judged against the
    naive implementation."""
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [workspace_for(1, torch.device("cuda")) for _ in streams]

    def run(j, s):
        cases = [make(4097, 48, seed=s), make(1031, 16, seed=s + 1)]
        # dx and dz of one launch on the current stream
        refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
        outputs: list[list] = [[], []]
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for _ in range(5):
            for i, stream in enumerate(streams):
                x, z, w, dy = cases[i]
                with torch.cuda.stream(stream):
                    outputs[i].append(
                        backward(dy, x, z, w, EPS, workspace=workspaces[i])
                    )
        torch.cuda.synchronize()
        draws = []
        for i in range(2):
            dx_d, dz_d, dw_d = refs[i]
            varies = _dw_varies(cases[i][0].numel() // D)
            for dx, dz, dw in outputs[i]:
                assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
                assert_dw_repeats(dw, dw_d, varies=varies, what=f"stream {i} dw")
            assert not bool(workspaces[i].any()), i
            name = f"stream {i} dw"
            draws.append(
                silu_draw(
                    *cases[i],
                    EPS,
                    [{name: dw_d}] + [{name: out[2]} for out in outputs[i]],
                    {name: "dw"},
                )
            )
        return merge_draws(draws)

    judge(record_property, run, seed=30)


def test_side_stream_and_no_host_sync(record_property):
    """A forward and two backwards (on a caller workspace and on the side
    stream's own) on a side stream with the host's synchronization an error:
    ``y``, ``dx`` and ``dz`` the bits of the calls on the current stream, so
    the weight gradient where its band repeats it
    (:func:`assert_dw_repeats`), the workspace zero after; on the inputs of
    every draw of ``judge``, every call's weight gradient judged against the
    naive implementation."""

    def run(j, s):
        x, z, w, dy = make(2049, 48, seed=s)
        ws = workspace_for(x.shape[0] * x.shape[1], x.device)
        # Load both modules before the sync-checked region; the bits of y, dx
        # and dz on the current stream.
        want_y = forward(x, z, w, EPS)
        dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS, workspace=ws)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.cuda.stream(stream):
                y = forward(x, z, w, EPS)
                got = backward(dy, x, z, w, EPS, workspace=ws)
                fresh = backward(dy, x, z, w, EPS)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert torch.equal(y, want_y)
        varies = _dw_varies(x.numel() // D)
        for dx, dz, dw in (got, fresh):
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
            assert_dw_repeats(dw, dw_d, varies=varies, what="side stream dw")
        assert not bool(ws.any())
        return silu_draw(
            x,
            z,
            w,
            dy,
            EPS,
            [{"dw": dw_d}, {"dw": got[2]}, {"dw": fresh[2]}],
            {"dw": "dw"},
        )

    judge(record_property, run, seed=9)


def test_cuda_graph_capture_and_replay(record_property):
    """A forward, a backward on a caller workspace and an FP32 accumulating
    backward without one, captured in a graph, replayed once on the inputs of
    each draw of ``judge`` (copied into the captured inputs): ``y``, ``dx``,
    ``dz`` and the accumulated FP32 weight gradient the bits of eager calls
    on the same inputs (the accumulator through every replay so far), the
    BF16 weight gradient too where its band repeats it
    (:func:`assert_dw_repeats`), the workspace zero after each; ``y``,
    ``dx``, ``dz``, the BF16 weight gradient and the accumulated one (from
    its value before the replay) judged against the naive
    implementation."""
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    ws = workspace_for(T * H, x.device)

    def step():
        forward(xs, zs, w, EPS, out=y)
        backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )
        # FP32 main-grad accumulation with the workspace allocated in the graph
        backward(dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    main.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    main_e = torch.zeros(D, dtype=torch.float32, device="cuda")
    varies = _dw_varies(T * H)

    def run(k, s):
        xn, zn, _, dyn = make(T, H, seed=s)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        before = main.clone()
        graph.replay()
        torch.cuda.synchronize()
        # the replay against eager launches on the same inputs: the same bits
        # (the BF16 weight gradient where its band repeats it)
        assert torch.equal(y, forward(xn, zn, w, EPS)), k
        dx_e, dz_e, dw_e = backward(dyn, xn, zn, w, EPS)
        assert torch.equal(dx, dx_e) and torch.equal(dz, dz_e), k
        assert_dw_repeats(dw, dw_e, varies=varies, what=f"replay {k} dw")
        backward(dyn, xn, zn, w, EPS, grad_weight=main_e, accumulate_grad_weight=True)
        assert torch.equal(main, main_e), k
        assert not bool(ws.any()), k
        return silu_draw(
            xn,
            zn,
            w,
            dyn,
            EPS,
            [
                {
                    "y": y.clone(),
                    "dx": dx.clone(),
                    "dz": dz.clone(),
                    "dw": dw.clone(),
                    "accumulated FP32 dw": main.clone(),
                }
            ],
            {"y": "y", "dx": "dx", "dz": "dz", "dw": "dw", "accumulated FP32 dw": "dw"},
            starts={"accumulated FP32 dw": before},
        )

    judge(record_property, run, seed=40)


def _kept_graph() -> torch.cuda.CUDAGraph:
    try:
        return torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        pytest.skip("listing graph nodes needs torch.cuda.CUDAGraph(keep_graph=True)")


def _graph_node_types(graph: torch.cuda.CUDAGraph) -> list[str]:
    """The node types of a graph captured with ``keep_graph=True``."""
    driver = pytest.importorskip("cuda.bindings.driver")
    handle = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(handle)
    _, nodes, count = driver.cuGraphGetNodes(handle, count)
    return [
        driver.cuGraphNodeGetType(node)[1].name.replace("CU_GRAPH_NODE_TYPE_", "")
        for node in nodes[:count]
    ]


@pytest.fixture
def reserve(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fresh reserve and capture tables; the reserve of the current device,
    filled and seen ready by eager calls without a workspace."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    # 1056 rows: past every small-row backward band, a backward with a workspace
    x, z, w, dy = make(22, 48, seed=69)
    backward(dy, x, z, w, EPS)  # loads the module, fills the reserve
    torch.cuda.synchronize()
    backward(dy, x, z, w, EPS)  # sees the fill complete
    torch.cuda.synchronize()
    state = pkg._RESERVES[torch.cuda.current_device()]
    assert len(state["ready"]) == pkg._RESERVE_SIZE and not state["pending"]
    return state


def test_capture_without_workspace_takes_a_reserve_workspace(reserve, record_property):
    """Backwards without a caller workspace captured in one graph on one
    stream share one zero-filled workspace taken from the reserve: the graph
    holds the backward kernels only (no fill kernel, memset or event wait), for
    both weight-gradient dtypes; every replay matches eager launches and leaves
    the workspace zero; eager calls do not touch it.  The graph replays once
    on the inputs of each draw of ``judge`` (copied into the captured
    inputs), its weight gradients and the FP32 one accumulated from zero
    judged against the naive implementation."""
    pytest.importorskip("cuda.bindings.runtime")
    T, H = 129, 48
    x, z, w, dy = make(T, H, seed=70)
    backward(dy, x, z, w, EPS)
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    torch.cuda.synchronize()
    main.zero_()
    ready = list(reserve["ready"])

    def step():
        outs = [backward(dy, x, z, w, EPS) for _ in range(2)]
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
        return outs

    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    ws = taken[0]
    varies = _dw_varies(T * H)

    def run(k, s):
        for captured, new in zip((x, z, w, dy), make(T, H, seed=s), strict=True):
            captured.copy_(new)
        # the eager bits of the draw: an eager call between replays
        dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
        main_e = torch.zeros(D, dtype=torch.float32, device="cuda")
        backward(dy, x, z, w, EPS, grad_weight=main_e, accumulate_grad_weight=True)
        main.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for dx, dz, dw in outs:
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), k
            assert_dw_repeats(dw, dw_d, varies=varies, what=f"replay {k}")
        assert torch.equal(main, main_e), k
        assert not bool(ws.any()), k
        return silu_draw(
            x,
            z,
            w,
            dy,
            EPS,
            [{"dw": dw.clone()} for _, _, dw in outs]
            + [{"dw": dw_d, "accumulated FP32 dw": main.clone()}],
            {"dw": "dw", "accumulated FP32 dw": "dw"},
        )

    judge(record_property, run, seed=70)


def test_graphs_without_workspace_replay_together_on_two_streams(
    reserve, record_property
):
    """Two graphs, each capturing a backward without a caller workspace, own
    two reserve workspaces and replay at the same time on two streams, the
    first replays right after the captures with no host synchronization:
    five replays of each on the inputs of each draw of ``judge`` (the first
    draw's the captured ones, the others' copied into them), ``dx`` and
    ``dz`` the bits of an eager call, so the weight gradient where its band
    repeats it (:func:`assert_dw_repeats`), every replay's weight gradient
    judged against the naive implementation."""
    shapes = [(4097, 48), (1031, 16)]
    cases = [make(T, H, seed=71 + i) for i, (T, H) in enumerate(shapes)]
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    torch.cuda.synchronize()
    graphs, outs, taken = [], [], []
    for x, z, w, dy in cases:
        ready = list(reserve["ready"])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs.append(backward(dy, x, z, w, EPS))
        graphs.append(graph)
        taken += [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 2 and taken[0].data_ptr() != taken[1].data_ptr()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    def run(j, s):
        if j:
            torch.cuda.synchronize()
            for i, (T, H) in enumerate(shapes):
                for captured, new in zip(
                    cases[i], make(T, H, seed=s + i), strict=True
                ):
                    captured.copy_(new)
                refs[i] = backward(*_args(cases[i]), EPS)
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        results: list[list] = [[], []]
        for _ in range(5):
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    graphs[i].replay()
                    results[i].append(tuple(t.clone() for t in outs[i]))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        draws = []
        for i in range(2):
            dx_d, dz_d, dw_d = refs[i]
            varies = _dw_varies(cases[i][0].numel() // D)
            for dx, dz, dw in results[i]:
                assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), i
                assert_dw_repeats(dw, dw_d, varies=varies, what=f"graph {i} dw")
            assert not bool(taken[i].any()), i
            name = f"graph {i} dw"
            draws.append(
                silu_draw(
                    *cases[i],
                    EPS,
                    [{name: dw_d}] + [{name: out[2]} for out in results[i]],
                    {name: "dw"},
                )
            )
        return merge_draws(draws)

    judge(record_property, run, seed=71)


def _args(case):
    """The leading arguments of a backward, ``(dy, x, z, w)``, of ``make``'s
    ``(x, z, w, dy)``."""
    x, z, w, dy = case
    return dy, x, z, w


def test_captured_calls_without_capture_ids_take_one_workspace_each(
    reserve, monkeypatch, record_property
):
    """Without cuda-python's capture ids, each captured call takes its own
    reserve workspace; replays stay correct and leave them zero: on the
    inputs of each draw of ``judge`` (the first draw's the captured ones, the
    others' copied into them), two replays, ``dx`` and ``dz`` the bits of an
    eager call, so the weight gradient where its band repeats it
    (:func:`assert_dw_repeats`), every replay's weight gradient judged
    against the naive implementation."""
    monkeypatch.setattr(pkg, "_cudart", lambda: None)
    x, z, w, dy = make(129, 48, seed=74)
    backward(dy, x, z, w, EPS)
    torch.cuda.synchronize()
    ready = list(reserve["ready"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [backward(dy, x, z, w, EPS) for _ in range(3)]
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 3
    varies = _dw_varies(129 * 48)

    def run(j, s):
        if j:
            for captured, new in zip((x, z, w, dy), make(129, 48, seed=s), strict=True):
                captured.copy_(new)
        dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
        launched = [{"dw": dw_d}]
        for k in range(2):
            graph.replay()
            torch.cuda.synchronize()
            for dx, dz, dw in outs:
                assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), (j, k)
                assert_dw_repeats(dw, dw_d, varies=varies, what=f"replay {k}")
                launched.append({"dw": dw.clone()})
            assert not any(bool(ws.any()) for ws in taken), (j, k)
        return silu_draw(x, z, w, dy, EPS, launched, {"dw": "dw"})

    judge(record_property, run, seed=74)


def test_capture_with_no_ready_workspace_fills_in_the_graph(
    monkeypatch, record_property
):
    """With no ready reserve workspace a captured backward without a caller
    workspace zero-fills one from the graph's pool inside the graph (a memset
    or fill kernel before the backward kernel) and stays correct: on the
    inputs of each draw of ``judge`` (the first draw's the captured ones, the
    others' copied into them), three replays, ``dx`` and ``dz`` the bits of
    an eager call, so the weight gradient where its band repeats it
    (:func:`assert_dw_repeats`), every replay's weight gradient judged
    against the naive implementation."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVE_SIZE", 0)
    x, z, w, dy = make(129, 48, seed=73)
    backward(dy, x, z, w, EPS)
    torch.cuda.synchronize()
    assert pkg._RESERVES[torch.cuda.current_device()]["blocks"] == []
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        dx, dz, dw = backward(dy, x, z, w, EPS)
    types = _graph_node_types(graph)
    assert types in (["MEMSET", "KERNEL"], ["KERNEL", "KERNEL"]), types
    varies = _dw_varies(129 * 48)

    def run(j, s):
        if j:
            for captured, new in zip((x, z, w, dy), make(129, 48, seed=s), strict=True):
                captured.copy_(new)
        dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
        launched = [{"dw": dw_d}]
        for k in range(3):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), (j, k)
            assert_dw_repeats(dw, dw_d, varies=varies, what=f"replay {k}")
            launched.append({"dw": dw.clone()})
        return silu_draw(x, z, w, dy, EPS, launched, {"dw": "dw"})

    judge(record_property, run, seed=73)


def test_autograd_matches_functional_and_checkpoint(launches):
    """Autograd and non-reentrant checkpointing give the functional entry
    points' bits (the weight gradient where its band repeats it,
    :func:`assert_dw_repeats`) and save only the inputs; the functional
    entry points' accuracy is ``check_all``'s."""
    x, z, w, dy = make(513, 48, seed=8)
    # the functional entry points' bits
    y_ref = forward(x, z, w, EPS)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    varies = _dw_varies(x.numel() // D)

    def check(y, xg, zg, wg, what):
        assert torch.equal(y, y_ref), what
        assert torch.equal(xg, dx_d) and torch.equal(zg, dz_d), what
        assert wg.dtype == torch.bfloat16, what
        assert_dw_repeats(wg, dw_d, varies=varies, what=what)

    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = gated_rmsnorm(xa, za, wa, EPS)
    y.backward(dy)
    check(y, xa.grad, za.grad, wa.grad, "autograd")

    # Saved state: only the inputs, no FP32 temporary.
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel(), t.data_ptr()))
        return t

    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        gated_rmsnorm(xb, zb, wb, EPS)
    assert sorted(saved) == sorted(
        [
            (torch.bfloat16, x.numel(), xb.data_ptr()),
            (torch.bfloat16, z.numel(), zb.data_ptr()),
            (torch.bfloat16, D, wb.data_ptr()),
        ]
    )

    # Non-reentrant activation checkpointing replays the forward, in the same
    # mode, in the backward.
    launches.clear()
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        gated_rmsnorm, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    check(yc, xc.grad, zc.grad, wc.grad, "checkpoint")
    fwd = _forward_module(x.numel() // D)
    assert launches == [fwd, fwd, _backward_module(x.numel() // D)], launches


def test_nondeterministic_mode_launches_this_package(launches):
    x, z, w, dy = make(129, 48, seed=14)
    forward(x, z, w, EPS)
    backward(dy, x, z, w, EPS)
    backward(
        dy, x, z, w, EPS, grad_weight=torch.empty(D, dtype=torch.float32, device="cuda")
    )
    fwd = _forward_module(x.numel() // D)
    rows = 129 * 48
    bf16, f32 = _backward_module(rows), _backward_module(rows, torch.float32)
    assert launches == [fwd, bf16, f32], launches
    for name in launches:
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
    arch = loader.device_arch(torch.cuda.current_device())
    for name, role in ((bf16, "bwd_bf16"), (f32, "bwd_f32")):
        bands = {band for _, band in loader.BACKWARD_BANDS[arch][role]}
        assert loader.MODULES[name]["role"] in bands, (name, role)
    # A mode keyword is a TypeError before any launch.
    launches.clear()
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_rmsnorm_silu_gate_nondet_train_forward(
            x, z, w, EPS, deterministic=False
        )
    with pytest.raises(TypeError, match="deterministic"):
        pkg.cake_rmsnorm_silu_gate_nondet_train_backward(
            dy, x, z, w, EPS, deterministic=False
        )
    assert launches == []


def test_empty_and_errors():
    x = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    assert forward(x, x, w).shape == x.shape
    dx, dz, dw = backward(x, x, x, w)
    assert dx.shape == x.shape and torch.equal(dw, torch.zeros_like(dw))
    good = torch.randn(4, 2, D, device="cuda").bfloat16()
    with pytest.raises(ValueError):
        forward(good.float(), good, w)
    with pytest.raises(ValueError):
        forward(good, good[:, :1], w)
    with pytest.raises(ValueError):
        forward(
            torch.randn(4, 64, device="cuda").bfloat16(),
            torch.randn(4, 64, device="cuda").bfloat16(),
            w[:64],
        )
    with pytest.raises(ValueError):
        backward(good, good, good, w, accumulate_grad_weight=True)
    with pytest.raises(ValueError):
        backward(
            good,
            good,
            good,
            w,
            grad_weight=torch.empty(D, dtype=torch.float16, device="cuda"),
        )


# Programs this package no longer ships: no record, ``build.json`` library or
# source file names them.  The second static backwards of sm_100a and sm_103a
# (9e8bf507c9563bb24193, 348592b55aa04975cc33) and of sm_107a (3 slots:
# f5a44b8ba8bccb590488 of a BF16, de4ac875a4e07bb547fd of an FP32 weight
# gradient), the static backward of a BF16 weight gradient on sm_100a and
# sm_103a (7e9f1c0cadbc18c390d2), the forwards without the split gate
# (c7919260b39b69e4e646, 83fe9e25ce0d01cc3ab1, c6151f64f22ae65a48eb,
# cf7a87977aaad1a9822f, 3cb6aa31e140d9b13052, 04bb67629537ef270bd4), and the
# static backward of an FP32 weight gradient on sm_100a and sm_103a whose
# last CTA reads its windows by fetching adds (f2915438ea54b487d93f; the
# build that reads them by exchanges ships, with the same bits); the
# row-claims backwards and the fused backward that added their rows'
# FP32 sums into FP64 column sums (615c6fae07a6a89f3a45, f14663d91454a0e833f7,
# 1d4b7311041b9046e379; the programs that add them into FP32 replicas ship);
# and the static backward of an FP32 weight gradient on sm_107a
# (a8cfbb391df971a9cb59) and the claimed backwards of an FP32 weight gradient
# (f2f608aa8d7a81906c9b, 75a7f19a91e6ec05dffc): past its small-row band an
# FP32 weight gradient runs the repeatable backward (``bwd_f32_rep``, the
# deterministic package's FP32 backward program), and on sm_100a and sm_103a
# the static backward 421e73578399336342e3 in a band of rows per SM.
REMOVED_PROGRAMS = tuple(
    f"cake_rmsnorm_silu_gate_train_{digest}"
    for digest in (
        "9e8bf507c9563bb24193",
        "348592b55aa04975cc33",
        "7e9f1c0cadbc18c390d2",
        "f5a44b8ba8bccb590488",
        "de4ac875a4e07bb547fd",
        "c7919260b39b69e4e646",
        "83fe9e25ce0d01cc3ab1",
        "c6151f64f22ae65a48eb",
        "cf7a87977aaad1a9822f",
        "3cb6aa31e140d9b13052",
        "04bb67629537ef270bd4",
        "f2915438ea54b487d93f",
        "615c6fae07a6a89f3a45",
        "f14663d91454a0e833f7",
        "1d4b7311041b9046e379",
        "a8cfbb391df971a9cb59",
        "f2f608aa8d7a81906c9b",
        "75a7f19a91e6ec05dffc",
    )
)


def test_registry_records_reference_the_build_manifest():
    """Every architecture routes the forward, the roles of its forward bands
    and the roles of its ``BACKWARD_BANDS`` bands, one module a role, and
    besides them only the long role of each band role
    ``LONG_BACKWARD_ROWS_PER_SM`` names (of either weight gradient, which
    runs in its place on long packed rows) and backward roles of a BF16
    weight gradient that no band runs (programs kept routed for
    measurement; the call role ``bwd_bf16`` on sm_107a only: no static
    backward of a BF16 weight gradient ships on sm_100a and sm_103a); the
    call role ``bwd_f32`` (the static backward of an FP32 weight gradient) is
    routed on sm_100a and sm_103a only, where a band runs it, and no claimed
    backward of an FP32 weight gradient ships; every
    module record names its ``build.json`` library (schema ``cake-build/1``),
    whose entries carry the package's kernel and binding sources and the
    precise-arithmetic flags; no record, library or source file names a
    program the package no longer ships (``REMOVED_PROGRAMS``).  The static
    and the claimed backward records cut their rows into chunks by one grid
    rule (``CHUNK_GRID_RULE``), launch as the claimed backward of a BF16
    weight gradient of their architecture, ship on the architectures that
    route them and carry the workspace buffers of their slots
    (``BACKWARD_SLOTS``); the row-claims backward records claim 16-row blocks
    by their grid rule (``ROW_CLAIMS_GRID_RULES``; the fused backward's
    ``FUSED_GRID_RULE``), launch 256-thread CTAs, take FP32 replicas of the
    column sums (``ROW_CLAIMS_REPLICAS``, the fused backward's
    ``FUSED_REPLICAS``) and the two words (``ROW_CLAIMS_LAYOUT``,
    ``FUSED_LAYOUT``) and no chunks, and ship on the
    architectures that route them (the fused backward on every architecture,
    ``FUSED_ROLES``); the repeatable backward records claim 16-row blocks by
    their grid rule (``REP_GRID_RULE``), launch 256-thread CTAs, take the
    fixed-point digits, the flag words and the two words (``REP_LAYOUT``) and
    no chunks, and ship on every architecture (``REP_ROLES``; the long role's
    record launches as its band role's, with the same arguments); the
    small-row backward records launch one cluster and take no workspace.
    Each program without the weight gradient (``NODW_ROLES``) ships on the
    architectures that route it, each of which routes its twin
    (``NODW_TWINS``), launches as its twin with the same arguments and keeps
    its twin's grid rule and workspace, but a strided program, whose rule is
    its ``NODW_STRIDED_GRID_RULES`` entry, whose CTAs have its
    ``NODW_STRIDED_THREADS`` threads and which takes no workspace; the strided
    cuts (``NODW_STRIDED_ROWS_PER_SM``) of an architecture name exactly the
    bands of ``NODW_CUT_BANDS`` whose strided and claimed programs it both
    routes, and an architecture routes a band's strided program of packed
    rows only where it routes the band's strided program.  The repeatable
    band's build of packed rows only (``PACKED_SMALL_ROLE``, ``NODW_SMALL``)
    is routed exactly where ``PACKED_SMALL_ROWS_PER_SM`` (the pinned
    ``SMALL_ROWS_PER_SM``) has a cut."""
    assert loader.NODW == "_nodw" and loader.STRIDED == "_strided"
    assert loader.PACKED == "_packed" and loader.SMALL == "_small"
    assert set(loader.NODW_ROLES) == set(NODW_TWINS), loader.NODW_ROLES
    assert loader.PACKED_SMALL_ROLE == NODW_SMALL
    small = "bwd_bf16_rep" + loader.NODW + loader.STRIDED + loader.PACKED
    assert NODW_SMALL == small + loader.SMALL
    assert loader.PACKED_SMALL_ROWS_PER_SM == SMALL_ROWS_PER_SM
    assert set(loader.PACKED_SMALL_ROWS_PER_SM) == set(loader.ARCHES)
    for arch, cuts in loader.PACKED_SMALL_ROWS_PER_SM.items():
        assert set(cuts) <= {"bwd_bf16"}, (arch, cuts)
        assert all(isinstance(v, int) and v >= 1 for v in cuts.values()), arch
        routed = NODW_SMALL in loader.ROUTES[arch]
        assert routed == bool(cuts.get("bwd_bf16")), (arch, cuts)
    assert "strided_row_blocks" in loader.GRID_KINDS
    assert set(loader.NODW_STRIDED_ROWS_PER_SM) == set(loader.ARCHES)
    for arch, cuts in loader.NODW_STRIDED_ROWS_PER_SM.items():
        both = {
            band
            for band in NODW_CUT_BANDS
            if {band + loader.NODW, band + loader.NODW + loader.STRIDED}
            <= set(loader.ROUTES[arch])
        }
        assert set(cuts) == both, (arch, cuts)
        assert all(isinstance(v, int) and v >= 1 for v in cuts.values()), arch
        for band in NODW_CUT_BANDS:
            strided = band + loader.NODW + loader.STRIDED
            if strided + loader.PACKED in loader.ROUTES[arch]:
                assert strided in loader.ROUTES[arch], (arch, band)
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train"
    )
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        routed = set(loader.ROUTES[arch])
        forward_bands = {
            band
            for table in (loader.FORWARD_BANDS, loader.STRIDED_FORWARD_BANDS)
            for _, band in table[arch]
        }
        assert forward_bands <= set(FORWARD_ROLES), (arch, forward_bands)
        assert set(loader.BACKWARD_BANDS[arch]) == {"bwd_bf16", "bwd_f32"}, arch
        backward_bands = {
            band for bands in loader.BACKWARD_BANDS[arch].values() for _, band in bands
        }
        banded = {"fwd"} | forward_bands | backward_bands
        assert banded <= routed, (arch, banded - routed)
        unbanded = routed - banded
        cuts = loader.LONG_BACKWARD_ROWS_PER_SM[arch]
        assert cuts == {
            "bwd_bf16_rep": LONG_ROWS_PER_SM[arch],
            "bwd_f32_rep": LONG_ROWS_PER_SM[arch],
        }, (arch, cuts)
        assert set(cuts) <= backward_bands, (arch, cuts)
        longs = {band + loader.LONG for band in cuts}
        assert longs <= unbanded, (arch, longs - unbanded)
        # besides the long roles, only roles of a BF16 weight gradient run in
        # no band
        assert all(role.startswith("bwd_bf16") for role in unbanded - longs), (
            arch,
            unbanded,
        )
        assert not unbanded & set(loader.BAND_ROLES), (arch, unbanded)
        # the static backward of an FP32 weight gradient on sm_100a and
        # sm_103a only, in a band; no claimed backward of an FP32 weight
        # gradient ships
        assert ("bwd_f32" in routed) == (arch != "sm_107a"), arch
        assert ("bwd_f32" in backward_bands) == (arch != "sm_107a"), arch
        assert "bwd_f32_mid" not in routed, arch
        assert ("bwd_bf16" in routed) == (arch == "sm_107a"), arch
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            assert record["role"] == role and arch in record["arches"], (arch, role)
            assert loader.library_name(name, arch) in libraries, (arch, role, name)
    routed = {name for by_role in loader.ROUTES.values() for name in by_role.values()}
    assert set(loader.MODULES) == routed
    sources = [path.name for path in (package / "csrc").iterdir()]
    for program in REMOVED_PROGRAMS:
        assert program not in loader.MODULES, program
        assert not [lib for lib in libraries if lib.startswith(program)], program
        assert not [s for s in sources if s.startswith(program)], program
    for name, record in loader.MODULES.items():
        assert name.startswith("cake_rmsnorm_silu_gate_train_"), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["kind"] in loader.GRID_KINDS
        assert record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False
        clustered = record["grid_rule"]["kind"] == "cluster"
        assert (record["launch"]["cluster"] != (1, 1, 1)) == clustered, name
        plan = [key for _kind, key in record["arg_plan"]]
        if record["role"] in loader.NODW_ROLES:
            role = record["role"]
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            strided = role in NODW_STRIDED_GRID_RULES
            for arch in record["arches"]:
                twin = loader.MODULES[loader.ROUTES[arch][NODW_TWINS[role]]]
                assert arch in twin["arches"], (name, arch)
                for key in ("logical", "arg_plan", "ffi_entry"):
                    assert record[key] == twin[key], (name, arch, key)
                for key in ("cluster", "cooperative", "use_pdl"):
                    assert record["launch"][key] == twin["launch"][key], (name, key)
                if not strided:
                    assert record["launch"]["block"] == twin["launch"]["block"], name
                    assert record["grid_rule"] == twin["grid_rule"], (name, arch)
                    assert record.get("workspace") == twin.get("workspace"), name
            if strided:
                assert record["grid_rule"] == NODW_STRIDED_GRID_RULES[role], name
                threads = NODW_STRIDED_THREADS[role]
                assert record["launch"]["block"] == (threads, 1, 1), name
                assert record.get("workspace") is None, name
                assert loader.workspace_layout(name) == (), name
                assert loader.workspace_bytes(name) == 0, name
        elif record["role"] in ("fwd", "fwd_tiny", "fwd_l32_tiny", "fwd_short"):
            assert record["grid_rule"]["kind"] == "row_blocks"
            assert record.get("workspace") is None
        elif record["role"] in (
            "fwd_xlong",
            "fwd_long",
            "fwd_mid",
            "fwd_pipe",
            "fwd_pipe_packed",
        ):
            assert record["grid_rule"]["kind"] == "persistent_row_blocks"
            assert record.get("workspace") is None
        elif record["role"] in loader.BAND_ROLES:
            assert clustered
            assert record["launch"]["cluster"] == (record["grid_rule"]["cluster"], 1, 1)
            assert {"dw", "weight_grad"} <= set(plan), name
            assert not {"dw_win", "dw_sums", "dw_spec", "dw_done"} & set(plan), name
            assert record.get("workspace") is None
            assert loader.workspace_bytes(name) == 0
        elif record["grid_rule"]["kind"] == "row_claims":
            role = record["role"]
            fused = role in FUSED_ROLES[record["arches"][0]]
            rule = FUSED_GRID_RULE if fused else ROW_CLAIMS_GRID_RULES.get(role)
            assert record["grid_rule"] == rule, name
            assert record["launch"]["block"] == (256, 1, 1)
            assert {"dw_sums", "dw_done", "dw", "weight_grad"} <= set(plan), name
            assert not {"dw_win", "dw_spec", "blocks_per_chunk"} & set(plan), name
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            if fused:
                assert routed_on == [a for a in loader.ARCHES if role in FUSED_ROLES[a]]
            replicas = FUSED_REPLICAS if fused else ROW_CLAIMS_REPLICAS
            layout = FUSED_LAYOUT if fused else ROW_CLAIMS_LAYOUT
            nbytes = FUSED_WORKSPACE_BYTES if fused else ROW_CLAIMS_WORKSPACE_BYTES
            assert record["workspace"] == {
                "dw_done": 2,
                "dw_sums": replicas * D,
            }, name
            assert loader.workspace_layout(name) == layout, name
            assert loader.workspace_bytes(name) == nbytes, name
            for arch in record["arches"]:
                assert loader.workspace_bytes(name) <= WORKSPACE_BYTES[arch], name
        elif record["grid_rule"]["kind"] == "claimed_row_blocks":
            role = record["role"]
            assert record["grid_rule"] == REP_GRID_RULE, name
            assert record["launch"]["block"] == (256, 1, 1)
            assert {"acc", "flags", "done", "dw", "weight_grad"} <= set(plan), name
            assert not {
                "dw_win",
                "dw_sums",
                "dw_spec",
                "dw_done",
                "blocks_per_chunk",
            } & set(plan), name
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            assert routed_on == [a for a in loader.ARCHES if role in REP_ROLES[a]]
            if role.endswith(loader.LONG):
                band = role[: -len(loader.LONG)]
                twin = loader.MODULES[loader.ROUTES[routed_on[0]][band]]
                for key in ("arches", "grid_rule", "launch", "arg_plan", "workspace"):
                    assert record[key] == twin[key], (name, key)
            assert record["workspace"] == {"acc": 18 * D, "done": 2, "flags": D}, name
            assert loader.workspace_layout(name) == REP_LAYOUT, name
            assert loader.workspace_bytes(name) == REP_WORKSPACE_BYTES, name
            for arch in record["arches"]:
                assert loader.workspace_bytes(name) <= WORKSPACE_BYTES[arch], name
        else:
            role = record["role"]
            assert record["grid_rule"] == CHUNK_GRID_RULE, name
            assert record["launch"]["block"] == (128, 1, 1)
            assert {
                "dw_win",
                "dw_spec",
                "dw_done",
                "dw",
                "weight_grad",
                "blocks_per_chunk",
            } <= set(plan), name
            assert "dw_sums" not in plan, name
            routed_on = [a for a in loader.ARCHES if loader.ROUTES[a].get(role) == name]
            assert record["arches"] == routed_on, (name, record["arches"])
            for arch in record["arches"]:
                slots = BACKWARD_SLOTS[arch][role]
                assert record["workspace"] == {
                    "dw_done": 2,
                    "dw_spec": D,
                    "dw_win": slots * 7 * D,
                }, (name, arch)
                assert loader.workspace_layout(name) == WORKSPACE_LAYOUTS[slots]
                assert loader.workspace_bytes(name) == MODULE_WORKSPACE_BYTES[slots]
                assert loader.workspace_bytes(name) <= WORKSPACE_BYTES[arch], name
                claimed = loader.MODULES[loader.ROUTES[arch]["bwd_bf16_mid"]]
                assert record["launch"] == claimed["launch"], (name, arch)
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for flag in ("--prec-div=true", "--prec-sqrt=true", "--ftz=false"):
                assert flag in entry["nvcc_flags"], (name, arch, flag)
            assert "--use_fast_math" not in entry["nvcc_flags"], (name, arch)
            for source in entry["sources"]:
                assert (package / source).is_file(), (name, source)


# (rows, SMs, blocks per chunk, chunks, CTAs) of the static and the claimed
# backwards: 8-row blocks, at most 8192 chunks of the fewest consecutive
# blocks each (the chunks of the deterministic backward at 2e989556),
# at most 8 CTAs per SM.
BACKWARD_GRIDS = [
    (1, 148, 1, 1, 1),
    (48, 148, 1, 6, 6),
    (6191, 148, 1, 774, 774),
    (64 * 148, 148, 1, 1184, 1184),  # every chunk has a CTA of one wave
    (64 * 148 + 1, 148, 1, 1185, 1184),  # a chunk more than one wave
    (49488, 148, 1, 6186, 1184),
    (49489, 148, 1, 6187, 1184),
    (8 * 8192, 148, 1, 8192, 1184),
    (8 * 8192 + 1, 148, 2, 4097, 1184),  # two blocks a chunk
    (245760, 148, 4, 7680, 1184),
    (245761, 148, 4, 7681, 1184),
    (32737 * 48, 152, 24, 8185, 1216),
    (1440000, 160, 22, 8182, 1280),
    (64 * 212, 212, 1, 1696, 1696),
    (64 * 212 + 1, 212, 1, 1697, 1696),
    (98352, 212, 2, 6147, 1696),
    (98353, 212, 2, 6148, 1696),
    (64 * 216, 216, 1, 1728, 1728),
    (64 * 216 + 1, 216, 1, 1729, 1728),
    (2**31 - 1, 216, 32768, 8192, 1728),
]


@pytest.mark.parametrize(
    ("rows", "sm_count", "per_chunk", "chunks", "ctas"), BACKWARD_GRIDS
)
def test_backward_cta_rule(rows, sm_count, per_chunk, chunks, ctas):
    """The static and the claimed backwards (every routed module whose grid
    rule is ``chunks``): ``blocks_per_chunk = max(1, ceil(blocks / 8192))``
    blocks a chunk, ``ceil(blocks / blocks_per_chunk)`` chunks, ``min(chunks,
    8 * SMs)`` CTAs (each running chunks ``b``, ``b + grid``, ... in the static
    backwards; claiming runs of chunks in the claimed ones), so a smaller grid
    still covers every row; each backward role runs the module of the
    ``BACKWARD_BANDS`` band that holds the rows (:func:`_band_module`)."""
    assert _chunks(rows) == chunks, rows
    for arch in loader.ARCHES:
        chunked = {
            name
            for name in loader.ROUTES[arch].values()
            if loader.MODULES[name]["grid_rule"]["kind"] == "chunks"
        }
        assert chunked, arch
        for name in sorted(chunked):
            record = loader.MODULES[name]
            assert loader.backward_chunks(name, rows) == (per_chunk, chunks)
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == ctas == min(chunks, 8 * sm_count), (name, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (ctas, 1, 1)
        for role in ("bwd_bf16", "bwd_f32"):
            want = _band_module(arch, role, rows, sm_count)
            got = loader.module_for_rows(arch, role, rows, sm_count)
            assert got == want, (arch, role, rows)


# (rows, SMs, 16-row blocks, CTAs at 4 CTAs per SM, CTAs at 5 CTAs per SM) of
# the row-claims backwards.
ROW_CLAIMS_GRIDS = [
    (1, 148, 1, 1, 1),
    (16, 148, 1, 1, 1),
    (17, 148, 2, 2, 2),
    (11 * 48, 148, 33, 33, 33),
    (16 * 592, 148, 592, 592, 592),  # a block a CTA of one wave of 4 per SM
    (16 * 592 + 1, 148, 593, 592, 593),
    (16 * 740, 148, 740, 592, 740),  # a block a CTA of one wave of 5 per SM
    (16 * 740 + 1, 148, 741, 592, 740),
    (1031 * 48, 148, 3093, 592, 740),
    (4097 * 48, 152, 12291, 608, 760),
    (300 * 48, 212, 900, 848, 900),
    (2**31 - 1, 216, 2**27, 864, 1080),
]


@pytest.mark.parametrize(
    ("rows", "sm_count", "blocks", "ctas4", "ctas5"), ROW_CLAIMS_GRIDS
)
def test_row_claims_backward_cta_rule(rows, sm_count, blocks, ctas4, ctas5):
    """The row-claims backwards (every routed module whose grid rule is
    ``row_claims`` but the fused backward's, :func:`test_fused_backward_cta_rule`):
    ``min(ceil(rows / 16), ctas_per_sm * SMs)`` CTAs, at most
    one CTA a 16-row block (past one wave the CTAs claim the blocks from the
    claim counter at run time), a grid of that many CTAs, and no chunks."""
    assert -(-rows // 16) == blocks, rows
    found = set()
    for arch in loader.ARCHES:
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            rule = record["grid_rule"]
            if rule["kind"] != "row_claims" or role in FUSED_ROLES[arch]:
                continue
            if NODW_TWINS.get(role) in FUSED_ROLES[arch]:
                continue
            found.add(role)
            assert rule == ROW_CLAIMS_GRID_RULES[role], (arch, role)
            want = {4: ctas4, 5: ctas5}[rule["ctas_per_sm"]]
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == want == min(blocks, rule["ctas_per_sm"] * sm_count), (
                name,
                rows,
            )
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (want, 1, 1)
            with pytest.raises(ValueError, match="chunks"):
                loader.backward_chunks(name, rows)
    assert found == set(ROW_CLAIMS_GRID_RULES), found


@pytest.mark.parametrize(
    ("rows", "sm_count", "blocks", "ctas4", "_ctas5"), ROW_CLAIMS_GRIDS
)
def test_fused_backward_cta_rule(rows, sm_count, blocks, ctas4, _ctas5):
    """The fused backward (``FUSED_ROLES``: ``bwd_bf16_fused`` on every
    architecture, grid kind ``row_claims``) and its claimed program without
    the weight gradient (``bwd_bf16_fused_nodw``): ``min(ceil(rows / 16), 4 * SMs)``
    CTAs (``ROW_CLAIMS_GRIDS`` at 4 CTAs per SM; past one wave the CTAs
    claim the blocks at run time), a grid of that many CTAs, and no chunks."""
    assert -(-rows // 16) == blocks, rows
    found = set()
    for arch in loader.ARCHES:
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            if record["grid_rule"]["kind"] != "row_claims":
                continue
            if NODW_TWINS.get(role, role) not in FUSED_ROLES[arch]:
                continue
            found.add((arch, role))
            assert record["grid_rule"] == FUSED_GRID_RULE, (arch, role)
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == ctas4 == min(blocks, 4 * sm_count), (name, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (ctas4, 1, 1)
            with pytest.raises(ValueError, match="chunks"):
                loader.backward_chunks(name, rows)
    want = {(a, r) for a, roles in FUSED_ROLES.items() for r in roles}
    want |= {
        (a, "bwd_bf16_fused_nodw")
        for a in loader.ARCHES
        if "bwd_bf16_fused_nodw" in loader.ROUTES[a]
    }
    assert found == want, found


@pytest.mark.parametrize(
    ("rows", "sm_count", "blocks", "ctas4", "_ctas5"), ROW_CLAIMS_GRIDS
)
def test_repeatable_backward_cta_rule(rows, sm_count, blocks, ctas4, _ctas5):
    """The repeatable backwards (``REP_ROLES``: ``bwd_bf16_rep``,
    ``bwd_f32_rep`` and their long roles on every architecture, grid kind
    ``claimed_row_blocks``) and the claimed programs without the weight
    gradient of the BF16 ones (``bwd_bf16_rep_nodw``,
    ``bwd_bf16_rep_nodw_long``):
    ``min(ceil(rows / 16), 4 * SMs)`` CTAs (``ROW_CLAIMS_GRIDS`` at 4 CTAs
    per SM; past one wave the CTAs claim the blocks at run time), a grid of
    that many CTAs, and no chunks."""
    assert -(-rows // 16) == blocks, rows
    found = set()
    for arch in loader.ARCHES:
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            if record["grid_rule"]["kind"] != "claimed_row_blocks":
                continue
            found.add((arch, role))
            assert record["grid_rule"] == REP_GRID_RULE, (arch, role)
            got = loader.backward_ctas(name, rows, sm_count)
            assert got == ctas4 == min(blocks, 4 * sm_count), (name, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
            assert grid == (ctas4, 1, 1)
            with pytest.raises(ValueError, match="chunks"):
                loader.backward_chunks(name, rows)
    want = {(a, r) for a, roles in REP_ROLES.items() for r in roles}
    want |= {
        (a, r)
        for a in loader.ARCHES
        for r in ("bwd_bf16_rep_nodw", "bwd_bf16_rep_nodw_long")
        if r in loader.ROUTES[a]
    }
    assert found == want, found


# The forward bands of each architecture for packed rows: (last rows per SM,
# role) in increasing rows per SM (``None``: no end), a call of ``rows`` rows on
# ``sm_count`` SMs running the first band's role with ``rows <= last *
# sm_count`` (cuts that follow the checkpointed training step at each rows per
# SM; the tiny forwards end where beside a kernel holding 16 SMs they still
# end no later than the deterministic package's forward as it routes the rows:
# the 32-lane tiny forward up to 4 rows per SM on sm_100a and 12 on sm_103a,
# past which it is slower than the tiny forward, then the tiny forward up to
# 36).
# The forward's band roles, the call role ``fwd`` among them (the plain
# forward, the last band of other layouts); the backward's are
# ``BACKWARD_BANDS``.
FORWARD_BANDS = {
    "sm_100a": (
        (4, "fwd_l32_tiny"),
        (36, "fwd_tiny"),
        (85, "fwd_pipe"),
        (160, "fwd_mid"),
        (400, "fwd_long"),
        (520, "fwd_xlong"),
        (6500, "fwd_pipe"),
        (None, "fwd_mid"),
    ),
    "sm_103a": (
        (12, "fwd_l32_tiny"),
        (36, "fwd_tiny"),
        (240, "fwd_mid"),
        (600, "fwd_xlong"),
        (6000, "fwd_pipe"),
        (8192, "fwd_mid"),
        (None, "fwd_pipe"),
    ),
    "sm_107a": (
        (65, "fwd_tiny"),
        (175, "fwd_mid"),
        (280, "fwd_xlong"),
        (None, "fwd_pipe_packed"),
    ),
}
# The forward bands of inputs whose rows are not packed (the tiny forwards
# read packed rows only), and of packed rows from PACKED_ROWS on, as
# FORWARD_BANDS.
STRIDED_FORWARD_BANDS = {
    "sm_100a": (
        (12, "fwd_short"),
        (140, "fwd_mid"),
        (400, "fwd_long"),
        (700, "fwd_xlong"),
        (None, "fwd"),
    ),
    "sm_103a": (
        (12, "fwd_short"),
        (250, "fwd_mid"),
        (700, "fwd_xlong"),
        (None, "fwd"),
    ),
    "sm_107a": (
        (59, "fwd_short"),
        (480, "fwd_mid"),
        (None, "fwd"),
    ),
}
FORWARD_ROLES = (
    "fwd",
    "fwd_tiny",
    "fwd_l32_tiny",
    "fwd_short",
    "fwd_mid",
    "fwd_long",
    "fwd_xlong",
    "fwd_pipe",
    "fwd_pipe_packed",
)
# The first ``FORWARD_BANDS`` band of each architecture (packed rows): the
# 32-lane tiny forward on sm_100a and sm_103a, the tiny forward on sm_107a.
FIRST_FORWARD_BAND = {
    "sm_100a": "fwd_l32_tiny",
    "sm_103a": "fwd_l32_tiny",
    "sm_107a": "fwd_tiny",
}
# Packed rows from this many on run by STRIDED_FORWARD_BANDS (a packed row's
# element offset no longer fits 32 bits).
PACKED_ROWS = 16777216
# The one grid rule of the static and the claimed backwards (no cut of their
# own: ``BACKWARD_BANDS`` gives each its rows).
CHUNK_GRID_RULE = {
    "chunk_limit": 8192,
    "ctas_per_sm": 8,
    "kind": "chunks",
    "rows_per_block": 8,
}


def _short_end(arch, sm_count):
    """The most rows the short forward runs on ``arch`` by
    ``STRIDED_FORWARD_BANDS``."""
    (last, role) = STRIDED_FORWARD_BANDS[arch][0]
    assert role == "fwd_short"
    return last * sm_count


def _tiny_band(arch, sm_count, role="fwd_tiny"):
    """``(first, last)``: the fewest and the most rows the tiny forward
    ``role`` (``fwd_tiny``, or ``fwd_l32_tiny`` on sm_100a and sm_103a) runs
    on ``arch`` by ``FORWARD_BANDS``, whose first bands are the tiny
    forwards'."""
    first = 1
    for last, band in FORWARD_BANDS[arch]:
        assert band in ("fwd_tiny", "fwd_l32_tiny"), (arch, role, band)
        if band == role:
            return first, last * sm_count
        first = last * sm_count + 1
    raise AssertionError(f"{arch} has no {role} band")


def _band_module(arch, role, rows, sm_count, packed=True):
    """The module of the backward ``role`` (``bwd_bf16`` or ``bwd_f32``) for
    ``rows`` rows on ``arch`` with ``sm_count`` SMs: the routed module of the
    band of :func:`_backward_bands` whose first and last rows hold ``rows``
    (the last band has no end), or where the rows are ``packed`` the routed
    module of the long role of :func:`_long_bands` that holds them."""
    longs = _long_bands(arch, role, sm_count) if packed else []
    for band, first, last in longs + _backward_bands(arch, role, sm_count):
        if first <= rows and (last is None or rows <= last):
            return loader.ROUTES[arch][band]
    raise AssertionError(f"no {role} band of {arch} holds {rows} rows")


def _tier_module(arch, role, rows, sm_count, packed=True):
    """The module of ``role`` for ``rows`` rows: the forward's by
    ``FORWARD_BANDS`` (by ``STRIDED_FORWARD_BANDS`` where the rows are not
    ``packed`` or are ``PACKED_ROWS`` or more), the module of the first forward
    band whose rows per SM hold ``rows``; a backward role's by
    :func:`_band_module` (its long roles on ``packed`` rows)."""
    if role == "fwd":
        packed = packed and rows < PACKED_ROWS
        table = FORWARD_BANDS if packed else STRIDED_FORWARD_BANDS
        for last, band in table[arch]:
            if last is None or rows <= last * sm_count:
                return loader.ROUTES[arch][band]
        return loader.route_module(arch, role)
    return _band_module(arch, role, rows, sm_count, packed)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_forward_and_backward_modules_run_in_their_bands(sm_count):
    """``module_for_rows``: the forward's module of the first
    ``FORWARD_BANDS`` band (``STRIDED_FORWARD_BANDS`` band where the rows are
    not packed) with ``rows <= last * sm_count`` (rows per SM; the tiny
    forwards of 32 lanes a row, sm_100a and sm_103a, and of 16, the short,
    pipelined, mid, long, extra-long and plain forwards, each where the
    architecture's bands name it), a backward role's module of the first
    ``BACKWARD_BANDS`` band whose ``last`` holds (:func:`_band_module`).  The
    backward bands of each architecture and role: the small-row bands (a
    ``BAND_ROLES`` role each, whose module is one cluster), each ending at a
    count of rows ``(ROWS, n)`` with ``n`` at most its module's ``max_rows``,
    and the bands of rows per SM, each running a static, a row-claims (a BF16
    weight gradient on sm_100a and sm_103a only), a fused (``FUSED_ROLES``), a
    repeatable (``REP_ROLES``) or a claimed backward by its grid rule, the
    ends of each kind in increasing order; last, with no end, the repeatable
    backward of either weight gradient (of a BF16 one after the fused band,
    of an FP32 one after its one small-row band, ``bwd_f32_t512`` up to 512
    rows, on sm_100a and sm_103a after a first repeatable band up to 64 rows
    per SM and the static backward up to 128, on sm_107a right after the
    small-row band: no claimed backward of an FP32 weight gradient ships);
    packed rows of the last repeatable band from
    ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on (``LONG_ROWS_PER_SM``, both
    weight-gradient dtypes) run its long role (:func:`_long_bands`), other
    rows the band's role.  No grid rule carries a band's
    rows per SM or a chunks-per-SM cut (the pipelined forwards'
    ``persistent_from_rows_per_sm`` shapes their grid only): the bands and
    the long roles' cuts are host plan only.  Packed rows of
    ``PACKED_ROWS`` or more run by ``STRIDED_FORWARD_BANDS``."""
    assert loader.PACKED_ROWS == PACKED_ROWS == 2**31 // D
    assert loader.FORWARD_BANDS == FORWARD_BANDS
    assert loader.STRIDED_FORWARD_BANDS == STRIDED_FORWARD_BANDS
    clustered = {
        loader.MODULES[name]["role"]
        for routes in loader.ROUTES.values()
        for name in routes.values()
        if loader.MODULES[name]["grid_rule"]["kind"] == "cluster"
        and loader.MODULES[name]["role"] not in loader.NODW_ROLES
    }
    assert set(loader.BAND_ROLES) == clustered, (loader.BAND_ROLES, clustered)
    assert loader.ROWS == "rows"
    for gone in ("ONE_WAVE", "MAX_ROWS", "_claims"):
        assert not hasattr(loader, gone), gone
    assert set(loader.BACKWARD_BANDS) == set(loader.ARCHES)
    for arch in loader.ARCHES:
        forward = {"fwd"}
        for table, first in (
            (FORWARD_BANDS, FIRST_FORWARD_BAND[arch]),
            (STRIDED_FORWARD_BANDS, "fwd_short"),
        ):
            bands = table[arch]
            lasts = [last for last, _ in bands[:-1]]
            assert lasts == sorted(set(lasts)) and bands[-1][0] is None, arch
            assert bands[0][1] == first, arch
            forward |= {band for _, band in bands}
            for _, band in bands:
                record = loader.MODULES[loader.ROUTES[arch][band]]
                assert record["role"] == band
                row_blocks = band in ("fwd", "fwd_tiny", "fwd_l32_tiny", "fwd_short")
                kind = "row_blocks" if row_blocks else None
                assert record["grid_rule"]["kind"] == (kind or "persistent_row_blocks")
        routed = {role for role in loader.ROUTES[arch] if role.startswith("fwd")}
        assert routed == forward, arch
        assert set(loader.BACKWARD_BANDS[arch]) == {"bwd_bf16", "bwd_f32"}, arch
        for role, bands in loader.BACKWARD_BANDS[arch].items():
            # the small-row bands (counts of rows) and the bands of rows per SM,
            # each kind's ends increasing; the last with no end the repeatable
            # backward of either weight gradient, of a BF16 one after the
            # fused band, of an FP32 one after its small-row band and, on
            # sm_100a and sm_103a, a first repeatable band up to 64 rows per
            # SM and the static backward up to 128 (the two cuts on the
            # repeatable backward's waves of 64 rows per SM)
            final = {"bwd_bf16": "bwd_bf16_rep", "bwd_f32": "bwd_f32_rep"}[role]
            assert bands[-1] == (None, final), (arch, role, bands[-1])
            if role == "bwd_bf16":
                assert bands[-2][1] == "bwd_bf16_fused", (arch, bands)
                assert type(bands[-2][0]) in (int, float), (arch, bands)
            elif arch == "sm_107a":
                assert bands == (
                    ((loader.ROWS, 512), "bwd_f32_t512"),
                    (None, "bwd_f32_rep"),
                ), (arch, bands)
            else:
                assert (loader.F32_REP_LAST, loader.F32_STATIC_LAST) == (64, 128)
                assert bands == (
                    ((loader.ROWS, 512), "bwd_f32_t512"),
                    (64, "bwd_f32_rep"),
                    (128, "bwd_f32"),
                    (None, "bwd_f32_rep"),
                ), (arch, bands)
            counts, per_sm = [], []
            for last, band in bands:
                assert band == role or band.startswith(role + "_"), (arch, band)
                record = loader.MODULES[loader.ROUTES[arch][band]]
                assert record["role"] == band
                rule = record["grid_rule"]
                if band in loader.BAND_ROLES:
                    assert rule["kind"] == "cluster", (arch, band)
                    assert isinstance(last, tuple) and len(last) == 2, (arch, band)
                    assert last[0] == loader.ROWS, (arch, band, last)
                    assert last[1] <= int(rule["max_rows"]), (arch, band, last)
                    counts.append(last[1])
                    continue
                if last is not None:
                    assert type(last) in (int, float), (arch, band, last)
                    per_sm.append(last)
                if band in ROW_CLAIMS_GRID_RULES:
                    assert arch != "sm_107a", (arch, band)
                    assert rule == ROW_CLAIMS_GRID_RULES[band], (arch, band)
                elif band in FUSED_ROLES[arch]:
                    assert rule == FUSED_GRID_RULE, (arch, band)
                elif band in REP_ROLES[arch]:
                    assert rule == REP_GRID_RULE, (arch, band)
                else:
                    assert rule == CHUNK_GRID_RULE, (arch, band)
            assert counts == sorted(set(counts)), (arch, role, counts)
            assert per_sm == sorted(set(per_sm)), (arch, role, per_sm)
            ends = [end for _, _, end in _backward_bands(arch, role, sm_count)]
            assert ends[:-1] == sorted(set(ends[:-1])), (arch, role, ends)
            # the long role of the last repeatable band of either weight
            # gradient, from its cut on
            longs = _long_bands(arch, role, sm_count)
            cut = LONG_ROWS_PER_SM[arch] * sm_count
            assert longs == [(final + loader.LONG, cut, None)], (arch, role, longs)
            assert _backward_bands(arch, role, sm_count)[-1][1] < cut, (arch, role)
        for name in loader.ROUTES[arch].values():
            rule = loader.MODULES[name]["grid_rule"]
            assert not {"min_rows_per_sm", "max_rows_per_sm"} & set(rule), name
            assert "more_chunks_than_per_sm" not in rule, name
        for role, name in loader.ROUTES[arch].items():
            assert loader.MODULES[name]["role"] == role
        edges = {1, 48, 255, 256, 257, 8 * 8192, 8 * 8192 + 1, 2**31 - 1}
        for role in loader.BACKWARD_BANDS[arch]:
            for _, _, end in _backward_bands(arch, role, sm_count)[:-1]:
                edges |= {end - 1, end, end + 1}
        for table in (FORWARD_BANDS, STRIDED_FORWARD_BANDS):
            for bands in table.values():
                for last, _ in bands[:-1]:
                    edges |= {last * sm_count + d for d in (-1, 0, 1)}
        for start in (10, 12, 59, 64, LONG_ROWS_PER_SM[arch]):
            edges |= {start * sm_count - 1, start * sm_count, start * sm_count + 1}
        edges |= {PACKED_ROWS - 1, PACKED_ROWS, PACKED_ROWS + 1}
        for role in loader.ROLES:
            for rows in sorted(edges):
                want = _tier_module(arch, role, rows, sm_count)
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == want, (arch, role, rows)
                want = _tier_module(arch, role, rows, sm_count, packed=False)
                got = loader.module_for_rows(arch, role, rows, sm_count, packed=False)
                assert got == want, (arch, role, rows, "strided")
    # the forward on sm_100a: packed rows the 32-lane tiny forward up to 4
    # rows per SM, the tiny forward up to 36, the pipelined forward up to 85,
    # the mid forward up to 160, the long forward up to 400, the extra-long
    # forward up to 520, the pipelined forward up to 6500, the mid forward
    # past it; other layouts the short forward up to 12 rows per SM, the mid
    # forward up to 140, the long forward up to 400, the extra-long forward up
    # to 700, the plain forward past it
    routes = loader.ROUTES["sm_100a"]
    for rows, packed, strided in (
        (1, "fwd_l32_tiny", "fwd_short"),
        (4 * sm_count, "fwd_l32_tiny", "fwd_short"),
        (4 * sm_count + 1, "fwd_tiny", "fwd_short"),
        (12 * sm_count, "fwd_tiny", "fwd_short"),
        (12 * sm_count + 1, "fwd_tiny", "fwd_mid"),
        (24 * sm_count, "fwd_tiny", "fwd_mid"),
        (36 * sm_count, "fwd_tiny", "fwd_mid"),
        (36 * sm_count + 1, "fwd_pipe", "fwd_mid"),
        (85 * sm_count, "fwd_pipe", "fwd_mid"),
        (85 * sm_count + 1, "fwd_mid", "fwd_mid"),
        (140 * sm_count + 1, "fwd_mid", "fwd_long"),
        (160 * sm_count, "fwd_mid", "fwd_long"),
        (160 * sm_count + 1, "fwd_long", "fwd_long"),
        (400 * sm_count, "fwd_long", "fwd_long"),
        (400 * sm_count + 1, "fwd_xlong", "fwd_xlong"),
        (520 * sm_count, "fwd_xlong", "fwd_xlong"),
        (520 * sm_count + 1, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count + 1, "fwd_pipe", "fwd"),
        (6500 * sm_count, "fwd_pipe", "fwd"),
        (6500 * sm_count + 1, "fwd_mid", "fwd"),
        (PACKED_ROWS - 1, "fwd_mid", "fwd"),
        (PACKED_ROWS, "fwd", "fwd"),
        (2**31 - 1, "fwd", "fwd"),
    ):
        got = loader.module_for_rows("sm_100a", "fwd", rows, sm_count)
        assert got == routes[packed], (rows, packed)
        got = loader.module_for_rows("sm_100a", "fwd", rows, sm_count, packed=False)
        assert got == routes[strided], (rows, strided)
    # the forward on sm_103a: packed rows the 32-lane tiny forward up to 12
    # rows per SM, the tiny forward up to 36, the mid forward up to 240, the
    # extra-long forward up to 600, the pipelined forward up to 6000, the mid
    # forward up to 8192, the pipelined forward past it; other layouts the
    # short forward up to 12, the mid forward up to 250, the extra-long
    # forward up to 700, the plain forward past it (sm_103a routes no long
    # forward)
    routes = loader.ROUTES["sm_103a"]
    for rows, packed, strided in (
        (1, "fwd_l32_tiny", "fwd_short"),
        (12 * sm_count, "fwd_l32_tiny", "fwd_short"),
        (12 * sm_count + 1, "fwd_tiny", "fwd_mid"),
        (32 * sm_count, "fwd_tiny", "fwd_mid"),
        (36 * sm_count, "fwd_tiny", "fwd_mid"),
        (36 * sm_count + 1, "fwd_mid", "fwd_mid"),
        (240 * sm_count, "fwd_mid", "fwd_mid"),
        (240 * sm_count + 1, "fwd_xlong", "fwd_mid"),
        (250 * sm_count, "fwd_xlong", "fwd_mid"),
        (250 * sm_count + 1, "fwd_xlong", "fwd_xlong"),
        (600 * sm_count, "fwd_xlong", "fwd_xlong"),
        (600 * sm_count + 1, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count, "fwd_pipe", "fwd_xlong"),
        (700 * sm_count + 1, "fwd_pipe", "fwd"),
        (6000 * sm_count, "fwd_pipe", "fwd"),
        (6000 * sm_count + 1, "fwd_mid", "fwd"),
        (8192 * sm_count, "fwd_mid", "fwd"),
        (8192 * sm_count + 1, "fwd_pipe", "fwd"),
        (PACKED_ROWS - 1, "fwd_pipe", "fwd"),
        (PACKED_ROWS, "fwd", "fwd"),
        (2**31 - 1, "fwd", "fwd"),
    ):
        got = loader.module_for_rows("sm_103a", "fwd", rows, sm_count)
        assert got == routes[packed], (rows, packed)
        got = loader.module_for_rows("sm_103a", "fwd", rows, sm_count, packed=False)
        assert got == routes[strided], (rows, strided)
    assert "fwd_long" not in routes
    # the forward on sm_107a: packed rows the tiny forward up to 65 rows per
    # SM, the mid forward up to 175, the extra-long forward up to 280, the
    # pipelined forward of packed rows past it; other layouts the short
    # forward up to 59, the mid forward up to 480, the plain forward past it
    # (sm_107a routes no long forward, no pipelined forward of other layouts
    # and no 32-lane tiny forward)
    routes = loader.ROUTES["sm_107a"]
    for rows, packed, strided in (
        (1, "fwd_tiny", "fwd_short"),
        (10 * sm_count + 1, "fwd_tiny", "fwd_short"),
        (59 * sm_count, "fwd_tiny", "fwd_short"),
        (59 * sm_count + 1, "fwd_tiny", "fwd_mid"),
        (65 * sm_count, "fwd_tiny", "fwd_mid"),
        (65 * sm_count + 1, "fwd_mid", "fwd_mid"),
        (175 * sm_count, "fwd_mid", "fwd_mid"),
        (175 * sm_count + 1, "fwd_xlong", "fwd_mid"),
        (280 * sm_count, "fwd_xlong", "fwd_mid"),
        (280 * sm_count + 1, "fwd_pipe_packed", "fwd_mid"),
        (480 * sm_count, "fwd_pipe_packed", "fwd_mid"),
        (480 * sm_count + 1, "fwd_pipe_packed", "fwd"),
        (65536 * 48, "fwd_pipe_packed", "fwd"),
        (PACKED_ROWS - 1, "fwd_pipe_packed", "fwd"),
        (PACKED_ROWS, "fwd", "fwd"),
        (2**31 - 1, "fwd", "fwd"),
    ):
        got = loader.module_for_rows("sm_107a", "fwd", rows, sm_count)
        assert got == routes[packed], (rows, packed)
        got = loader.module_for_rows("sm_107a", "fwd", rows, sm_count, packed=False)
        assert got == routes[strided], (rows, strided)
    assert "fwd_long" not in routes and "fwd_pipe" not in routes
    assert "fwd_l32_tiny" not in routes


def _check_backward_bands(arch: str, sm_count: int) -> None:
    """Every ``BACKWARD_BANDS`` band of ``arch`` runs rows on ``sm_count`` SMs
    (:func:`_backward_bands`), and ``module_for_rows`` runs its routed module
    at its first and at its last rows (the last band, which has no end, at
    ``INT32_MAX`` rows) that are not packed, and so on packed rows below its
    long role's first rows (:func:`_band_module`); neighbouring bands run
    different modules, and the row after a band's end runs the next band's
    module, so the bands cover every row count without a gap; a band of a
    count of rows, ``(ROWS, n)``, ends at most at its cluster module's
    ``max_rows``.  A long role (:func:`_long_bands`) runs packed rows at its
    first and at its last rows, not the row before them, and its band's
    role runs the same rows when they are not packed."""
    for role in ("bwd_bf16", "bwd_f32"):
        table = loader.BACKWARD_BANDS[arch][role]
        bands = _backward_bands(arch, role, sm_count)
        assert [band for band, _, _ in bands] == [band for _, band in table], (
            arch,
            role,
            sm_count,
            bands,
        )
        assert bands[0][1] == 1 and bands[-1][2] is None, (arch, role, bands)
        for i, (band, first, last) in enumerate(bands):
            name = loader.ROUTES[arch][band]
            assert loader.MODULES[name]["role"] == band, (arch, band)
            for rows in (first, loader.INT32_MAX if last is None else last):
                got = loader.module_for_rows(arch, role, rows, sm_count, packed=False)
                assert got == name, (arch, role, band, rows)
                got = loader.module_for_rows(arch, role, rows, sm_count)
                want = _band_module(arch, role, rows, sm_count)
                assert got == want, (arch, role, band, rows, "packed")
            if last is None:
                continue
            after = loader.ROUTES[arch][bands[i + 1][0]]
            assert after != name, (arch, role, band, bands[i + 1][0])
            got = loader.module_for_rows(arch, role, last + 1, sm_count, packed=False)
            assert got == after, (arch, role, band, last + 1)
        for band, first, last in _long_bands(arch, role, sm_count):
            name = loader.ROUTES[arch][band]
            assert loader.MODULES[name]["role"] == band, (arch, band)
            short = loader.ROUTES[arch][band[: -len(loader.LONG)]]
            for rows in (first, loader.INT32_MAX if last is None else last):
                got = loader.module_for_rows(arch, role, rows, sm_count)
                assert got == name, (arch, role, band, rows)
                got = loader.module_for_rows(arch, role, rows, sm_count, packed=False)
                assert got == short, (arch, role, band, rows, "strided")
            got = loader.module_for_rows(arch, role, first - 1, sm_count)
            assert got != name, (arch, role, band, first - 1)
        for last, band in table:
            if isinstance(last, tuple):
                rule = loader.MODULES[loader.ROUTES[arch][band]]["grid_rule"]
                assert rule["kind"] == "cluster", (arch, band)
                assert last[1] <= int(rule["max_rows"]), (arch, band, last)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_backward_bands_run_their_modules_from_their_first_to_their_last_rows(
    sm_count,
):
    """``BACKWARD_BANDS`` of every architecture on ``sm_count`` SMs
    (:func:`_check_backward_bands`); the call role ``bwd_bf16`` has no module
    on sm_100a and sm_103a and the call role ``bwd_f32`` none on sm_107a
    (``route_module`` raises ``ValueError``): there the bands of the role
    cover every row count without it.  On sm_100a and sm_103a the call role
    ``bwd_f32`` (the static backward) is the module of a band."""
    for arch in loader.ARCHES:
        _check_backward_bands(arch, sm_count)
    for arch, role in (
        ("sm_100a", "bwd_bf16"),
        ("sm_103a", "bwd_bf16"),
        ("sm_107a", "bwd_f32"),
    ):
        assert role not in loader.ROUTES[arch], (arch, role)
        with pytest.raises(ValueError, match="no exported"):
            loader.route_module(arch, role)
    for arch in ("sm_100a", "sm_103a"):
        banded = [band for _, band in loader.BACKWARD_BANDS[arch]["bwd_f32"]]
        assert banded.count("bwd_f32") == 1, (arch, banded)
        assert "bwd_f32" in loader.ROUTES[arch], arch


def test_backward_bands_run_their_modules_on_this_device(launches, record_property):
    """``BACKWARD_BANDS`` on this device's architecture and SM count
    (:func:`_check_backward_bands`, :func:`_device_bands`), and a backward of
    the first and of the last rows of every band (``T`` tokens of one head,
    packed rows; the last band at its first rows) and of the first rows of
    every long role (:func:`_device_long_bands`) launches the band's module;
    on the inputs of every draw of ``judge``, ``dx``, ``dz`` and the weight
    gradient of the band's dtype of each judged against the naive
    implementation."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    _check_backward_bands(arch, sm_count)
    cases = []
    for role, dtype in (("bwd_bf16", torch.bfloat16), ("bwd_f32", torch.float32)):
        assert _device_bands(role) == _backward_bands(arch, role, sm_count)
        assert _device_long_bands(role) == _long_bands(arch, role, sm_count)
        for band, first, last in _device_bands(role) + _device_long_bands(role):
            name = loader.ROUTES[arch][band]
            for rows in sorted({first} | ({last} if last is not None else set())):
                cases.append((band, name, rows, dtype))

    def run(j, s):
        draws = []
        for band, name, rows, dtype in cases:
            x, z, w, dy = make(rows, 1, seed=s)
            launches.clear()
            gw = torch.empty(D, dtype=dtype, device="cuda")
            dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=gw)
            assert launches == [name], (band, rows, launches)
            label = f"{band} {rows} rows "
            draws.append(
                silu_draw(
                    x,
                    z,
                    w,
                    dy,
                    EPS,
                    [{label + "dx": dx, label + "dz": dz, label + "dw": dw}],
                    {label + "dx": "dx", label + "dz": "dz", label + "dw": "dw"},
                )
            )
        return merge_draws(draws)

    judge(record_property, run, seed=91)


# SM counts on which the host routes the repeatable backward's long role:
# small MIG-like instances, the products' counts and larger ones.
LONG_SM_COUNTS = (1, 2, 7, 14, 16, 24, 33, 66, 148, 152, 160, 212, 216)


@pytest.mark.parametrize("sm_count", LONG_SM_COUNTS)
def test_backward_runs_the_long_role_on_packed_rows_only(sm_count):
    """The host (no launch): a backward of either weight gradient runs the
    repeatable backward's long role (``bwd_bf16_rep_long``,
    ``bwd_f32_rep_long``) from ``LONG_ROWS_PER_SM * SMs`` rows on when
    ``dy``, ``x`` and ``z`` all have head stride 128 and token stride ``H *
    128`` as ``_kernel_view`` passes them (``_backward_packed``; also with
    one head, whose head stride is passed as 128, and at the kernels' row
    limit), and the repeatable backward a row fewer, for a single token
    (token stride 0), a token stride of 0 in any one tensor, a gate or a
    ``dy`` sliced from a projection, head-major rows and a head stride other
    than 128 (every row count here is in the last repeatable band of either
    weight gradient)."""
    assert loader.LONG == "_long"
    cuts = {
        arch: {
            "bwd_bf16_rep": LONG_ROWS_PER_SM[arch],
            "bwd_f32_rep": LONG_ROWS_PER_SM[arch],
        }
        for arch in loader.ARCHES
    }
    assert cuts == loader.LONG_BACKWARD_ROWS_PER_SM
    H = 48
    row = H * D
    packed = (row, D, row, D, row, D)
    one_head = (D, D, D, D, D, D)
    for arch in loader.ARCHES:
        rep = loader.ROUTES[arch]["bwd_bf16_rep"]
        long = loader.ROUTES[arch]["bwd_bf16_rep_long"]
        f32_rep = loader.ROUTES[arch]["bwd_f32_rep"]
        f32_long = loader.ROUTES[arch]["bwd_f32_rep_long"]
        start = LONG_ROWS_PER_SM[arch] * sm_count
        assert start - 1 > 512, (arch, sm_count)
        for rows, heads, strides, is_packed, want in (
            (start - 1, H, packed, True, rep),
            (start, H, packed, True, long),
            (start + 1, H, packed, True, long),
            (loader.INT32_MAX, H, packed, True, long),
            (start - 1, 1, one_head, True, rep),
            (start, 1, one_head, True, long),
            (start, start, (0, D, 0, D, 0, D), False, rep),
            (start, H, (0, D, row, D, row, D), False, rep),
            (start, H, (row, D, 0, D, row, D), False, rep),
            (start, H, (row, D, row, D, 0, D), False, rep),
            (start, H, (row, D, row, D, 2 * row + 16, D), False, rep),
            (start, H, (2 * row + 16, D, row, D, row, D), False, rep),
            (start, H, (row, D, D, 65 * D, D, 65 * D), False, rep),
            (start, H, (row, D, row, D, row, 2 * D), False, rep),
        ):
            what = (arch, rows, heads, strides)
            assert pkg._backward_packed(heads, *strides) == is_packed, what
            got = loader.module_for_rows(
                arch, "bwd_bf16", rows, sm_count, packed=is_packed
            )
            assert got == want, what
            f32 = loader.module_for_rows(
                arch, "bwd_f32", rows, sm_count, packed=is_packed
            )
            assert f32 == (f32_long if want == long else f32_rep), what
            want_f32 = _band_module(arch, "bwd_f32", rows, sm_count, is_packed)
            assert f32 == want_f32, what


def _forcing(band: str, role: str = "bwd_bf16"):
    """A ``module_for_rows`` that names the routed module of the backward band
    role ``band`` for every call of the backward ``role`` (``bwd_bf16``, a
    BF16 weight gradient's, ``compute_grad_weight=False`` among them, or
    ``bwd_f32``) and the host's module for every other call."""
    plan = loader.module_for_rows

    def module(arch, call_role, rows, sm_count, *, packed=True):
        if call_role == role:
            return loader.ROUTES[arch][band]
        return plan(arch, call_role, rows, sm_count, packed=packed)

    return module


def _rep_calls(
    dy, x, z, w, ws, dw_dtype: torch.dtype = torch.bfloat16
) -> list[torch.Tensor]:
    """The backward calls of the long-role tests, all of the role of a
    ``dw_dtype`` weight gradient: a weight gradient, one accumulated onto
    0.5 (on ``ws``) and, for a BF16 weight gradient (whose bands
    ``compute_grad_weight=False`` takes), ``compute_grad_weight=False``.
    Their ``dx``, ``dz`` and weight gradients."""
    acc = torch.full((D,), 0.5, dtype=dw_dtype, device=x.device)
    if dw_dtype == torch.bfloat16:
        first = {}
    else:
        first = {"grad_weight": torch.empty(D, dtype=dw_dtype, device=x.device)}
    calls = [
        first,
        {"grad_weight": acc, "accumulate_grad_weight": True, "workspace": ws},
    ]
    if dw_dtype == torch.bfloat16:
        calls.append({"compute_grad_weight": False})
    got = []
    for kwargs in calls:
        dx, dz, dw = backward(dy, x, z, w, EPS, **kwargs)
        got += [dx, dz] if dw is None else [dx, dz, dw]
    return got


def _rep_launches(
    arch: str, name: str, rows: int, sm_count: int, calls: int
) -> list[str]:
    """The modules the ``calls`` calls of :func:`_rep_calls` launch for
    ``rows`` packed rows on ``sm_count`` SMs where ``module_for_rows`` names
    ``name``: ``name``, and for ``compute_grad_weight=False`` (the third
    call) the program ``nodw_module`` names for it."""
    names = [name] * calls
    if calls == 3:
        names[2] = loader.nodw_module(arch, name, rows, sm_count, packed=True)
    return names


# Packed shapes on which the repeatable backward's long role is compared with
# the repeatable backward: flat rows a row below, at and a row above the long
# role's first rows, and the issue's 32737 x 48 rows (claims far past the
# look-ahead), each on this device's grid and on the grids of
# ``LONG_BIT_SM_COUNTS`` SMs (``device_sm_count`` patched: the first rows of
# the long role follow the SMs).
LONG_BIT_SHAPES = ("long-1", "long", "long+1", "32737x48")
LONG_BIT_SM_COUNTS = (1, 16, 66)


@pytest.mark.parametrize("shape", LONG_BIT_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_long_role_gives_the_bits_of_the_repeatable_backward(
    dw_dtype, shape, launches
):
    """On this device's grid and on the grids of ``LONG_BIT_SM_COUNTS`` SMs,
    for ``LONG_BIT_SHAPES`` and a ``dw_dtype`` weight gradient: the host runs
    the long role (``bwd_bf16_rep_long``, ``bwd_f32_rep_long``) on packed
    rows from ``LONG_ROWS_PER_SM * SMs`` rows on and the repeatable backward
    of the dtype a row fewer; forced through ``module_for_rows``
    (:func:`_forcing`), the long role gives the bits of the repeatable
    backward on the same packed inputs and grid, and so do the host's calls:
    ``dx``, ``dz``, the weight gradient, accumulated or not, and for a BF16
    weight gradient ``compute_grad_weight=False`` (:func:`_rep_calls`; the
    programs without the weight gradient of the two, :func:`_rep_launches`);
    the workspace is zero after the calls."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    routes = loader.ROUTES[arch]
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    band = role + "_rep"
    rep, long = routes[band], routes[band + loader.LONG]
    calls = 3 if dw_dtype == torch.bfloat16 else 2
    for sm_count in (loader.device_sm_count(device_index), *LONG_BIT_SM_COUNTS):
        start = LONG_ROWS_PER_SM[arch] * sm_count
        if shape.startswith("long"):
            rows = start + {"long-1": -1, "long": 0, "long+1": 1}[shape]
            x, z, w, dy = make(rows, 1, layout="flat", seed=46)
        else:
            T, H = (int(v) for v in shape.split("x"))
            rows = T * H
            x, z, w, dy = make(T, H, seed=46)
        assert (rows >= start) == (shape != "long-1"), (shape, sm_count, rows)
        ws = workspace_for(rows, x.device)
        runs = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "device_sm_count", lambda index, count=sm_count: count)
            launches.clear()
            host = _rep_calls(dy, x, z, w, ws, dw_dtype)
            want = long if rows >= start else rep
            want = _rep_launches(arch, want, rows, sm_count, calls)
            assert launches == want, (shape, sm_count, launches)
            for forced in (band, band + loader.LONG):
                patch.setattr(pkg, "module_for_rows", _forcing(forced, role))
                launches.clear()
                runs.append(_rep_calls(dy, x, z, w, ws, dw_dtype))
                want = _rep_launches(arch, routes[forced], rows, sm_count, calls)
                assert launches == want, (forced, launches)
        torch.cuda.synchronize()
        assert not bool(ws.any()), (shape, sm_count)
        what = f"{shape} ({rows} rows) on the grids of {sm_count} SMs"
        for i, (a, b, c) in enumerate(zip(runs[1], runs[0], host, strict=True)):
            assert torch.equal(a, b), f"{what}: output {i} of the long role"
            assert torch.equal(c, b), f"{what}: output {i} of the host's module"


def test_capture_runs_the_long_role_and_the_repeatable_backward(reserve, launches):
    """Backwards of this device's first rows of the long role
    (``LONG_ROWS_PER_SM * SMs`` flat packed rows) captured in one graph on one
    stream: a BF16 weight gradient on a caller workspace runs the long role
    (``bwd_bf16_rep_long``), an FP32 one accumulated without a workspace the
    module of its band (on these packed rows its long role,
    ``bwd_f32_rep_long``), and a BF16 one whose ``grad_out`` is strided (read in
    place, so not packed) the repeatable backward on the reserve workspace
    the FP32 call took.  The graph holds the three backward kernels only (no
    fill kernel, memset or event wait).  Every replay on new inputs gives the
    bits of eager calls with the BF16 calls forced to the repeatable backward
    (:func:`_forcing`; the accumulated weight gradient those of the same
    accumulation run eagerly) and leaves the caller workspace and the reserve
    workspace zero."""
    pytest.importorskip("cuda.bindings.runtime")
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    routes = loader.ROUTES[arch]
    rows = LONG_ROWS_PER_SM[arch] * loader.device_sm_count(device_index)
    xs, zs, w, dys = make(rows, 1, layout="flat", seed=76)
    # grad_out read in place with token stride 2 * 128 + 16, so not packed
    wide = torch.zeros(rows, 2 * D + 16, dtype=torch.bfloat16, device=xs.device)
    dy_strided = wide[:, :D]
    dy_strided.copy_(dys)
    main = torch.zeros(D, dtype=torch.float32, device=xs.device)
    ws = workspace_for(rows, xs.device)

    def step():
        return (
            backward(dys, xs, zs, w, EPS, workspace=ws),
            backward(
                dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True
            ),
            backward(dy_strided, xs, zs, w, EPS),
        )

    f32 = _module("bwd_f32", rows)
    want = [routes["bwd_bf16_rep_long"], f32, routes["bwd_bf16_rep"]]
    launches.clear()
    step()  # loads the modules before the capture
    torch.cuda.synchronize()
    assert launches == want, launches
    assert not bool(ws.any())
    main.zero_()
    ready = list(reserve["ready"])
    launches.clear()
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    assert launches == want, launches
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [t for t in ready if all(t is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    eager_main = torch.zeros_like(main)
    forced = [routes["bwd_bf16_rep"], f32, routes["bwd_bf16_rep"]]
    for k in range(3):
        xn, zn, _, dyn = make(rows, 1, layout="flat", seed=77 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        dy_strided.copy_(dyn)
        graph.replay()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "module_for_rows", _forcing("bwd_bf16_rep"))
            launches.clear()
            refs = (
                backward(dyn, xn, zn, w, EPS),
                backward(
                    dyn,
                    xn,
                    zn,
                    w,
                    EPS,
                    grad_weight=eager_main,
                    accumulate_grad_weight=True,
                ),
                backward(dy_strided, xn, zn, w, EPS),
            )
            assert launches == forced, launches
        torch.cuda.synchronize()
        for i, (got, ref) in enumerate(zip(outs, refs, strict=True)):
            for what, a, b in zip(("dx", "dz", "dw"), got, ref, strict=True):
                assert torch.equal(a, b), f"replay {k}: {what} of call {i}"
        assert torch.equal(main, eager_main), k
        assert not bool(ws.any()), k
        assert not bool(taken[0].any()), k


def test_support_query_needs_a_route_for_every_call(monkeypatch):
    """``routes_every_call``: every exported architecture routes a module for
    every call (the role of every band of its ``FORWARD_BANDS``,
    ``STRIDED_FORWARD_BANDS`` and ``BACKWARD_BANDS``, and the long role of
    every band role of its ``LONG_BACKWARD_ROWS_PER_SM``; a role whose bands
    end without a ``None`` band, its own module), so the support query is
    true on this device; an architecture that routes no module for one band's
    role or one long role, or whose bands of a role without a module of its
    own end (the call role ``bwd_bf16`` has none on sm_100a and sm_103a,
    ``bwd_f32`` none on sm_107a), or that is not exported, does not."""
    assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda"))
    assert not loader.routes_every_call("sm_90a")
    routes = loader.ROUTES
    for arch in loader.ARCHES:
        assert loader.routes_every_call(arch), arch
        band_roles = {
            band
            for table in (loader.FORWARD_BANDS, loader.STRIDED_FORWARD_BANDS)
            for _, band in table[arch]
        }
        band_roles |= {
            band for bands in loader.BACKWARD_BANDS[arch].values() for _, band in bands
        }
        longs = {band + loader.LONG for band in loader.LONG_BACKWARD_ROWS_PER_SM[arch]}
        assert longs == {"bwd_bf16_rep_long", "bwd_f32_rep_long"}, (arch, longs)
        for role in sorted(band_roles | longs):
            fewer = {a: dict(by_role) for a, by_role in routes.items()}
            del fewer[arch][role]
            monkeypatch.setattr(loader, "ROUTES", fewer)
            assert not loader.routes_every_call(arch), (arch, role)
            monkeypatch.setattr(loader, "ROUTES", routes)
    # the bands of a BF16 weight gradient without their last (``None``) band:
    # sm_100a routes no module of the call role past them, sm_107a its static
    # backward
    full = loader.BACKWARD_BANDS
    bands = {a: dict(by_role) for a, by_role in full.items()}
    for arch in loader.ARCHES:
        bands[arch]["bwd_bf16"] = bands[arch]["bwd_bf16"][:-1]
    monkeypatch.setattr(loader, "BACKWARD_BANDS", bands)
    for arch in loader.ARCHES:
        own = "bwd_bf16" in loader.ROUTES[arch]
        assert loader.routes_every_call(arch) == own, arch
    assert not loader.routes_every_call("sm_100a")
    assert loader.routes_every_call("sm_107a")
    # the bands of an FP32 weight gradient without their last (``None``)
    # band (the BF16 bands whole): sm_107a routes no module of the call role
    # ``bwd_f32`` past them, sm_100a and sm_103a its static backward
    bands = {a: dict(by_role) for a, by_role in full.items()}
    for arch in loader.ARCHES:
        bands[arch]["bwd_f32"] = bands[arch]["bwd_f32"][:-1]
    monkeypatch.setattr(loader, "BACKWARD_BANDS", bands)
    for arch in loader.ARCHES:
        own = "bwd_f32" in loader.ROUTES[arch]
        assert own == (arch != "sm_107a"), arch
        assert loader.routes_every_call(arch) == own, arch


# (rows, SMs, CTAs) of the long forward: 32-row blocks, the fewest whole waves
# of 3 CTAs per SM whose CTAs walk at most 6 blocks, never more CTAs than blocks.
LONG_FORWARD_GRIDS = [
    (1, 148, 1),
    (32 * 444, 148, 444),  # a block per CTA of one wave
    (32 * 444 * 6, 148, 444),  # one wave, 6 blocks a CTA
    (32 * 444 * 6 + 1, 148, 888),  # a block more: two waves
    (140 * 148 + 1, 148, 444),  # B200 strided long band's first row: 648 blocks
    (160 * 148 + 1, 148, 444),  # B200 packed long band's first row: 741 blocks
    (400 * 148, 148, 444),  # its last row: 1850 blocks, 1 wave
    (2400 * 148, 148, 2220),  # 11100 blocks, 5 waves
    (32737 * 48, 148, 8436),  # 49106 blocks, 19 waves
    (600 * 152, 152, 912),  # GB300 packed extra-long band's last row: 2 waves
    (700 * 152, 152, 912),  # GB300 strided extra-long band's last row: 2 waves
    (2400 * 152, 152, 2280),  # 11400 blocks, 5 waves
    (32737 * 48, 152, 8208),  # 18 waves of 456
    (1740 * 212, 212, 2544),  # 11528 blocks, 4 waves
    (8192 * 48, 212, 2544),  # 12288 blocks, 4 waves of 636
    (2**31 - 1, 148, 25192 * 444),  # 2**26 blocks, 25192 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), LONG_FORWARD_GRIDS)
def test_long_forward_grid_is_whole_waves_of_at_most_six_blocks_a_cta(
    rows, sm_count, ctas
):
    name = loader.ROUTES["sm_100a"]["fwd_long"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {
        "ctas_per_sm": 3,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 6,
        "rows_per_block": 32,
    }
    # shipped on every architecture, routed on sm_100a only
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert "fwd_long" not in loader.ROUTES["sm_103a"]
    assert "fwd_long" not in loader.ROUTES["sm_107a"]
    # the extra-long forward launches the same grid
    xlong = loader.MODULES[loader.ROUTES["sm_107a"]["fwd_xlong"]]
    got = loader.launch_grid(xlong, rows=rows, sm_count=sm_count, arch="sm_107a")
    assert got == (ctas, 1, 1)
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch="sm_100a")
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 32), 3 * sm_count
    assert -(-blocks // ctas) <= 6
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 6 < blocks
    # the grid is the long forward's whatever ``ctas`` the call passes
    got = loader.launch_grid(
        record, rows=rows, sm_count=sm_count, ctas=7, arch="sm_100a"
    )
    assert got == grid
    # the forward: one 64-thread CTA per 8-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert fwd["launch"]["block"] == (64, 1, 1)
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count, arch="sm_100a") == (
        -(-rows // 8),
        1,
        1,
    )


def test_extra_long_forward_is_the_long_forwards_grid():
    # the extra-long forward: the long forward's grid rule (32-row blocks,
    # whole waves of 3 CTAs per SM walking at most 6 blocks), 256 threads, no
    # PDL, routed on every architecture (FORWARD_BANDS gives it its rows)
    name = loader.ROUTES["sm_103a"]["fwd_xlong"]
    assert loader.ROUTES["sm_107a"]["fwd_xlong"] == name
    assert loader.ROUTES["sm_100a"]["fwd_xlong"] == name
    record = loader.MODULES[name]
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert record["role"] == "fwd_xlong"
    long_record = loader.MODULES[loader.ROUTES["sm_100a"]["fwd_long"]]
    assert record["grid_rule"] == long_record["grid_rule"]
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["use_pdl"] is False
    for rows, sm_count in (
        (400 * 148 + 1, 148),
        (520 * 148, 148),
        (700 * 148, 148),
        (240 * 152 + 1, 152),
        (600 * 152, 152),
        (700 * 152, 152),
        (3000 * 152, 152),
        (32737 * 48, 152),
        (2**31 - 1, 152),
        (175 * 212 + 1, 212),
        (280 * 212, 212),
        (1740 * 212, 212),
        (32737 * 48, 212),
        (2**31 - 1, 216),
    ):
        assert loader.launch_grid(
            record, rows=rows, sm_count=sm_count, arch="sm_103a"
        ) == loader.launch_grid(
            long_record, rows=rows, sm_count=sm_count, arch="sm_100a"
        )


# (rows, SMs, CTAs) of the mid forward: 16-row blocks, the fewest whole waves
# of 6 CTAs per SM whose CTAs walk at most 2 blocks, never more CTAs than blocks.
MID_FORWARD_GRIDS = [
    (1, 148, 1),
    (16 * 888, 148, 888),  # a block per CTA of one wave
    (16 * 888 * 2, 148, 888),  # one wave, 2 blocks a CTA
    (16 * 888 * 2 + 1, 148, 1776),  # a block more: two waves
    (12 * 148 + 1, 148, 112),  # B200 strided mid band's first row: 112 blocks
    (85 * 148 + 1, 148, 787),  # B200 packed mid band's first row: 787 blocks
    (100 * 148 + 1, 148, 888),  # 926 blocks, one wave
    (140 * 148, 148, 888),  # B200 strided mid band's last row: 1295 blocks
    (160 * 148, 148, 888),  # B200 packed mid band's last row: 1480 blocks
    (976 * 148, 148, 5328),  # 9028 blocks, 6 waves
    (2400 * 148 - 1, 148, 11544),  # 22200 blocks, 13 waves
    (4097 * 48, 148, 6216),  # 12291 blocks, 7 waves of 888
    (6500 * 148 + 1, 148, 30192),  # B200 last packed mid band: 34 waves
    (36 * 152 + 1, 152, 343),  # GB300 first packed mid band's first row
    (240 * 152, 152, 1824),  # GB300 first packed mid band's last row: 2 waves
    (250 * 152, 152, 1824),  # GB300 strided mid band's last row: 2 waves
    (976 * 152, 152, 5472),  # 9272 blocks, 6 waves
    (4097 * 48, 152, 6384),  # 12291 blocks, 7 waves of 912
    (6000 * 152 + 1, 152, 29184),  # GB300 second packed mid band's first row
    (8192 * 152, 152, 39216),  # its last row: 77824 blocks, 43 waves
    (59 * 212 + 1, 212, 782),  # R200 strided mid band's first row: 782 blocks
    (65 * 212 + 1, 212, 862),  # R200 packed mid band's first row: 862 blocks
    (100 * 212 + 1, 212, 1272),  # 1326 blocks, one wave
    (175 * 212, 212, 1272),  # R200 packed mid band's last row: 2319 blocks
    (200 * 212, 212, 2544),  # 2650 blocks, 2 waves
    (480 * 212, 212, 3816),  # R200 strided mid band's last row: 3 waves
    (800 * 212 + 1, 212, 6360),  # 10601 blocks, 5 waves
    (1740 * 212, 212, 12720),  # 23055 blocks, 10 waves
    (65536 * 48, 212, 99216),  # 196608 blocks, 78 waves
    (2**31 - 1, 148, 67109712),  # 2**27 blocks, 75574 waves
]


@pytest.mark.parametrize(("rows", "sm_count", "ctas"), MID_FORWARD_GRIDS)
def test_mid_forward_grid_is_whole_waves_of_at_most_two_blocks_a_cta(
    rows, sm_count, ctas
):
    # each architecture's mid forward module (warming later blocks in L2) has
    # this grid rule
    records = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        record = loader.MODULES[loader.ROUTES[arch]["fwd_mid"]]
        assert arch in record["arches"]
        assert record["grid_rule"] == {
            "ctas_per_sm": 6,
            "kind": "persistent_row_blocks",
            "max_blocks_per_cta": 2,
            "rows_per_block": 16,
        }
        assert record["launch"]["block"] == (128, 1, 1)
        records.append((arch, record))
    arch, record = records[0]
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
    for each, other in records[1:]:
        got = loader.launch_grid(other, rows=rows, sm_count=sm_count, arch=each)
        assert got == grid, each
    assert grid == (ctas, 1, 1)
    blocks, wave = -(-rows // 16), 6 * sm_count
    assert -(-blocks // ctas) <= 2
    assert ctas == blocks or ctas % wave == 0
    assert ctas == blocks or (ctas - wave) * 2 < blocks
    got = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=7, arch=arch)
    assert got == grid


# The pipelined forwards' cut of rows per SM of rows the kernel reads as
# packed, and that of other rows where the rule has one (``fwd_pipe``, which
# reads rows of any layout; ``fwd_pipe_packed`` reads packed rows only).
CUT = "persistent_from_rows_per_sm"
STRIDED_CUT = "persistent_from_rows_per_sm_strided"
# (rows, SMs, architecture, CTAs) of the pipelined forward of the architecture
# (``fwd_pipe`` on sm_100a and sm_103a, ``fwd_pipe_packed`` on sm_107a) on
# rows the kernel reads as packed: one CTA per 16-row block below the
# architecture's cut of rows per SM times the SMs (8192 rows per SM on
# sm_100a and sm_107a, 5600 on sm_103a), from there the fewest whole waves of
# 6 CTAs per SM whose CTAs walk at most 2 blocks, never more CTAs than
# blocks.  A row less than the cut times the SMs and the cut times the SMs on
# 148, 152 and 212 SMs on every architecture, and the rows of the bands.  On
# sm_100a the cut lies past the bands of packed rows of the pipelined forward
# (up to 6500 rows per SM; the mid forward runs packed rows past it), so the
# pipelined forward runs every packed row it is routed one CTA a block there.
PIPE_FORWARD_GRIDS = [
    (1, 148, "sm_100a", 1),
    (36 * 148 + 1, 148, "sm_100a", 334),  # B200 first pipelined band's first row
    (85 * 148, 148, "sm_100a", 787),  # its last row: 787 blocks
    (520 * 148 + 1, 148, "sm_100a", 4811),  # B200 second pipelined band's first row
    (6500 * 148, 148, "sm_100a", 60125),  # its last row: a CTA a block
    (32737 * 48, 212, "sm_100a", 98211),  # 7412 rows per SM: a CTA a block
    (8192 * 148 - 1, 148, "sm_100a", 75776),  # a row below the cut: a CTA a block
    (8192 * 148, 148, "sm_100a", 38184),  # the cut: 75776 blocks, 43 waves of 888
    (8192 * 152 - 1, 152, "sm_100a", 77824),
    (8192 * 152, 152, "sm_100a", 39216),  # 77824 blocks, 43 waves of 912
    (8192 * 212 - 1, 212, "sm_100a", 108544),
    (8192 * 212, 212, "sm_100a", 54696),  # 108544 blocks, 43 waves of 1272
    (32737 * 48, 148, "sm_100a", 49728),  # 98211 blocks, 56 waves of 888
    (600 * 152 + 1, 152, "sm_103a", 5701),  # GB300 first pipelined band's first row
    (5600 * 148 - 1, 148, "sm_103a", 51800),
    (5600 * 148, 148, "sm_103a", 26640),  # 51800 blocks, 30 waves of 888
    (5600 * 152 - 1, 152, "sm_103a", 53200),  # a row below the cut: a CTA a block
    (5600 * 152, 152, "sm_103a", 27360),  # the cut: 53200 blocks, 30 waves of 912
    (5600 * 212 - 1, 212, "sm_103a", 74200),
    (5600 * 212, 212, "sm_103a", 38160),  # 74200 blocks, 30 waves of 1272
    (6000 * 152, 152, "sm_103a", 29184),  # first band's last row: 32 waves
    (8192 * 152 + 1, 152, "sm_103a", 39216),  # GB300 last band's first row: 43 waves
    (32737 * 48, 212, "sm_103a", 49608),  # 7412 rows per SM: 39 waves of 1272
    (280 * 212 + 1, 212, "sm_107a", 3711),  # R200 pipelined band's first row
    (32737 * 48, 212, "sm_107a", 98211),  # 7412 rows per SM: a CTA a block
    (8192 * 148 - 1, 148, "sm_107a", 75776),
    (8192 * 148, 148, "sm_107a", 38184),  # 75776 blocks, 43 waves of 888
    (8192 * 152 - 1, 152, "sm_107a", 77824),
    (8192 * 152, 152, "sm_107a", 39216),  # 77824 blocks, 43 waves of 912
    (8192 * 212 - 1, 212, "sm_107a", 108544),  # a row below the cut: a CTA a block
    (8192 * 212, 212, "sm_107a", 54696),  # the cut: 108544 blocks, 43 waves
    (65536 * 48, 212, "sm_107a", 99216),  # 196608 blocks, 78 waves of 1272
    (2**24 - 1, 212, "sm_107a", 525336),  # R200 pipelined band's last row: 413 waves
]
# (rows, SMs, architecture, CTAs) of ``fwd_pipe`` (sm_100a and sm_103a) on rows
# the kernel does not read as packed (``x`` or ``z`` broadcast over tokens, or
# one token): its cut of such rows, 500 rows per SM on sm_100a and 400 on
# sm_103a, a row less than it times the SMs and it times the SMs on 148, 152
# and 212 SMs, and rows of its bands (the second band of sm_100a, from 521
# rows per SM, and the first of sm_103a, from 601, lie past the cut: whole
# waves from their first rows on).
STRIDED_PIPE_FORWARD_GRIDS = [
    (85 * 148, 148, "sm_100a", 787),  # B200 first pipelined band's last row
    (520 * 148 + 1, 148, "sm_100a", 2664),  # B200 second band's first row: 3 waves
    (500 * 148 - 1, 148, "sm_100a", 4625),  # a row below the cut: a CTA a block
    (500 * 148, 148, "sm_100a", 2664),  # the cut: 4625 blocks, 3 waves of 888
    (500 * 152 - 1, 152, "sm_100a", 4750),
    (500 * 152, 152, "sm_100a", 2736),  # 4750 blocks, 3 waves of 912
    (500 * 212 - 1, 212, "sm_100a", 6625),
    (500 * 212, 212, "sm_100a", 3816),  # 6625 blocks, 3 waves of 1272
    (2000 * 148, 148, "sm_100a", 9768),  # 18500 blocks, 11 waves of 888
    (6500 * 148, 148, "sm_100a", 30192),  # second band's last row: 34 waves
    (600 * 152 + 1, 152, "sm_103a", 3648),  # GB300 first band's first row: 4 waves
    (400 * 148 - 1, 148, "sm_103a", 3700),
    (400 * 148, 148, "sm_103a", 2664),  # 3700 blocks, 3 waves of 888
    (400 * 152 - 1, 152, "sm_103a", 3800),  # a row below the cut: a CTA a block
    (400 * 152, 152, "sm_103a", 2736),  # the cut: 3800 blocks, 3 waves of 912
    (400 * 212 - 1, 212, "sm_103a", 5300),
    (400 * 212, 212, "sm_103a", 3816),  # 5300 blocks, 3 waves of 1272
    (2000 * 152, 152, "sm_103a", 10032),  # 19000 blocks, 11 waves of 912
    (5600 * 152 - 1, 152, "sm_103a", 27360),  # 53200 blocks, 30 waves of 912
    (6000 * 152, 152, "sm_103a", 29184),  # first band's last row: 32 waves
]


def _pipelined_records():
    """``{arch: record}`` of the pipelined forward each architecture routes
    (``fwd_pipe`` or ``fwd_pipe_packed``), each checked: the deterministic
    package's grid rule (16-row blocks, whole waves of 6 CTAs per SM of at
    most 2 blocks a CTA) with its positive cuts keyed by exactly the record's
    architectures, ``CUT`` and, for ``fwd_pipe`` only, ``STRIDED_CUT``, and
    128 threads."""
    records = {}
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for role in ("fwd_pipe", "fwd_pipe_packed"):
            name = loader.ROUTES[arch].get(role)
            if name is None:
                continue
            record = loader.MODULES[name]
            assert arch in record["arches"]
            rule = dict(record["grid_rule"])
            cuts = (CUT, STRIDED_CUT) if role == "fwd_pipe" else (CUT,)
            for key in cuts:
                cut = rule.pop(key)
                assert isinstance(cut, dict), (name, key)
                assert list(cut) == sorted(record["arches"]), (name, key)
                assert all(isinstance(v, int) and v >= 1 for v in cut.values())
            assert rule == {
                "ctas_per_sm": 6,
                "kind": "persistent_row_blocks",
                "max_blocks_per_cta": 2,
                "rows_per_block": 16,
            }, name
            assert record["launch"]["block"] == (128, 1, 1)
            assert arch not in records, arch
            records[arch] = record
    assert sorted(records) == ["sm_100a", "sm_103a", "sm_107a"]
    return records


def _check_pipelined_grid(record, arch, rows, sm_count, packed):
    """The grid of pipelined forward ``record`` on ``arch`` for ``rows`` rows
    (``packed``: read as packed rows) on ``sm_count`` SMs: one CTA a block
    below its cut of such rows (this test's own reading of the rule) times
    the SMs, from there whole waves of at most 2 blocks a CTA, never more
    CTAs than blocks; the same whatever ``ctas`` the call passes.  Returns
    the CTAs."""
    rule = record["grid_rule"]
    key = STRIDED_CUT if not packed and STRIDED_CUT in rule else CUT
    grid = loader.launch_grid(
        record, rows=rows, sm_count=sm_count, arch=arch, packed=packed
    )
    again = loader.launch_grid(
        record, rows=rows, sm_count=sm_count, ctas=7, arch=arch, packed=packed
    )
    assert again == grid, (arch, rows, packed)
    ctas = grid[0]
    assert grid[1:] == (1, 1), grid
    blocks, wave = -(-rows // 16), 6 * sm_count
    if rows < rule[key][arch] * sm_count:
        assert ctas == blocks, (arch, rows, packed, ctas)
    else:
        assert -(-blocks // ctas) <= 2, (arch, rows, packed, ctas)
        assert ctas == blocks or ctas % wave == 0, (arch, rows, packed, ctas)
        assert ctas == blocks or (ctas - wave) * 2 < blocks, (arch, rows, ctas)
    return ctas


@pytest.mark.parametrize(("rows", "sm_count", "arch", "ctas"), PIPE_FORWARD_GRIDS)
def test_pipelined_forward_grid_is_a_block_a_cta_then_whole_waves(
    rows, sm_count, arch, ctas
):
    # the pipelined forward of ``arch`` launches ``ctas`` CTAs on packed rows
    # (``packed`` True, the default); every architecture's pipelined forward
    # launches one CTA a block below its cut, whole waves from it
    records = _pipelined_records()
    record = records[arch]
    assert loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch) == (
        ctas,
        1,
        1,
    )
    assert _check_pipelined_grid(record, arch, rows, sm_count, True) == ctas
    for each, other in records.items():
        _check_pipelined_grid(other, each, rows, sm_count, True)


@pytest.mark.parametrize(
    ("rows", "sm_count", "arch", "ctas"), STRIDED_PIPE_FORWARD_GRIDS
)
def test_pipelined_forward_grid_of_rows_not_read_as_packed(rows, sm_count, arch, ctas):
    # fwd_pipe launches ``ctas`` CTAs on rows the kernel does not read as
    # packed; the pipelined forward of packed rows only (sm_107a) launches the
    # grid of packed rows on any rows
    records = _pipelined_records()
    assert _check_pipelined_grid(records[arch], arch, rows, sm_count, False) == ctas
    packed_only = records["sm_107a"]
    assert STRIDED_CUT not in packed_only["grid_rule"]
    assert loader.launch_grid(
        packed_only, rows=rows, sm_count=sm_count, arch="sm_107a", packed=False
    ) == loader.launch_grid(packed_only, rows=rows, sm_count=sm_count, arch="sm_107a")


def test_pipelined_forward_grid_needs_an_architecture_its_cut_keys():
    """A pipelined forward record's cuts are keyed by architecture:
    ``launch_grid`` without ``arch``, with ``arch`` None, or with an
    architecture a cut does not key (another product's, or an exported one the
    record does not ship on) raises ``ValueError`` naming the record's
    kernel, the cut, the architecture and the keys, at any rows (below the cut
    too) and for rows read as packed or not; with each of its architectures
    it gives a grid.  A forward whose rule has no cut (the mid forward)
    takes whole waves whatever ``arch``, and a rule whose cut is a single
    number serves every architecture and none."""
    for record in {id(r): r for r in _pipelined_records().values()}.values():
        name = record["kernel"]
        rule = record["grid_rule"]
        others = [arch for arch in loader.ARCHES if arch not in record["arches"]]
        for packed in (True, False):
            key = STRIDED_CUT if not packed and STRIDED_CUT in rule else CUT
            keys = str(sorted(rule[key]))
            for rows in (1, 16 * 1024 * 148, 2**31 - 1):
                for arch in (None, "sm_90a", *others):
                    with pytest.raises(ValueError) as raised:
                        loader.launch_grid(
                            record, rows=rows, sm_count=148, arch=arch, packed=packed
                        )
                    message = str(raised.value)
                    for part in (name, key, repr(arch), keys):
                        assert part in message, (name, arch, packed, message)
                with pytest.raises(ValueError, match=name):
                    loader.launch_grid(record, rows=rows, sm_count=148, packed=packed)
                for arch in record["arches"]:
                    got = loader.launch_grid(
                        record, rows=rows, sm_count=148, arch=arch, packed=packed
                    )
                    assert got[1:] == (1, 1) and got[0] >= 1, (name, arch, got)
    mid = loader.MODULES[loader.ROUTES["sm_100a"]["fwd_mid"]]
    for arch in (None, "sm_90a", *loader.ARCHES):
        for packed in (True, False):
            got = loader.launch_grid(
                mid, rows=16 * 888 * 2 + 1, sm_count=148, arch=arch, packed=packed
            )
            assert got == (1776, 1, 1), (arch, packed, got)
    # one number: 100 rows per SM on 7 SMs, a CTA a block up to 699 rows
    single = {
        "ctas_per_sm": 6,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 2,
        CUT: 100,
        "rows_per_block": 16,
    }
    for arch in (None, "sm_90a", *loader.ARCHES):
        for packed in (True, False):
            for rows, ctas in ((699, 44), (700, 42)):
                got = loader.launch_grid(
                    {"grid_rule": single},
                    rows=rows,
                    sm_count=7,
                    arch=arch,
                    packed=packed,
                )
                assert got == (ctas, 1, 1), (arch, packed, rows, got)


def test_forward_packed_is_the_kernels_packed_rows_test():
    """``forward_packed`` (host only): the persistent forward kernels' test of
    packed rows, both head strides 128 and both token strides ``heads *
    128``; a single token's stride (passed as 0), a broadcast of ``x`` or
    ``z`` over tokens, a gate or an ``x`` sliced from a projection,
    head-major rows and heads of another stride fail it, and so does any
    one stride off."""
    for strides, want in (
        ((48, 48 * D, D, 48 * D, D), True),
        ((1, D, D, D, D), True),
        ((48, 0, D, 0, D), False),
        ((48, 0, D, 48 * D, D), False),
        ((48, 48 * D, D, 0, D), False),
        ((48, 48 * D, D, 2 * 48 * D + 16, D), False),
        ((48, 2 * 48 * D + 16, D, 48 * D, D), False),
        ((48, D, 257 * D, D, 257 * D), False),
        ((48, 2 * 48 * D, 2 * D, 2 * 48 * D, 2 * D), False),
    ):
        assert loader.forward_packed(*strides) is want, strides
    for heads in (1, 2, 48, 3120):
        row = heads * D
        assert loader.forward_packed(heads, row, D, row, D) is True, heads
        for slot in range(1, 5):
            for delta in (-16, 16):
                strides = [heads, row, D, row, D]
                strides[slot] += delta
                assert loader.forward_packed(*strides) is False, (heads, strides)


# SM counts of the host's comparison of the grids ``run_module`` launches
# with ``launch_grid``'s: the products' (148, 152, 212) and one more.
HOST_GRID_SM_COUNTS = (148, 152, 160, 212)
# The architectures of that comparison: the exported ones and one that no
# record's cut keys.
HOST_GRID_ARCHES = (*loader.ARCHES, "sm_90a")
# Strides (in elements) of ``x`` and ``z`` the host passes a kernel in that
# comparison (``(heads, x_tok, x_head, z_tok, z_head)``): packed rows (48
# heads, one head) and rows the kernels do not read as packed (one token,
# whose stride is passed as 0; a gate or an ``x`` sliced from a projection;
# head-major rows; ``x`` and ``z`` broadcast over tokens, or one of them;
# heads of stride 256; mixed strides).
HOST_STRIDES = {
    "contiguous": (48, 48 * D, D, 48 * D, D),
    "one_head": (1, D, D, D, D),
    "one_token": (48, 0, D, 0, D),
    "projection": (48, 48 * D, D, 2 * 48 * D + 16, D),
    "x_projection": (48, 2 * 48 * D + 16, D, 48 * D, D),
    "head_major": (48, D, 257 * D, D, 257 * D),
    "broadcast": (48, 0, D, 0, D),
    "x_broadcast": (48, 0, D, 48 * D, D),
    "z_broadcast": (48, 48 * D, D, 0, D),
    "wide_heads": (48, 2 * 48 * D, 2 * D, 2 * 48 * D, 2 * D),
    "mixed": (48, 48 * D, D, D, 257 * D),
}


def _forward_names():
    """Every record of the forwards of whole waves (grid kind
    ``persistent_row_blocks``: the extra-long, long, mid and pipelined
    forwards)."""
    return sorted(
        name
        for name, record in loader.MODULES.items()
        if record["grid_rule"]["kind"] == "persistent_row_blocks"
    )


def _keyed(rule):
    """Whether a cut of ``rule`` is keyed by architecture: then a call on an
    architecture it does not key (``sm_90a``, which no cut keys) raises
    ``launch_grid``'s ``ValueError``, for rows of that cut's reading."""
    return any(isinstance(rule.get(key), dict) for key in (CUT, STRIDED_CUT))


def _computes_grid(rule, arch):
    """Whether ``run_module`` computes the grid of a forward of ``rule`` on
    ``arch`` itself, with no call of ``launch_grid``: each cut of the rule (it
    may have none: whole waves at any rows) is keyed by architecture, with
    ``arch`` among the keys.  A cut of one number or None, or one keyed
    without ``arch``, takes ``launch_grid``."""
    return all(
        isinstance(rule[key], dict) and arch in rule[key]
        for key in (CUT, STRIDED_CUT)
        if key in rule
    )


def _host_launches(patch, device):
    """Patch the registry so that ``run_module`` runs no kernel: each launch
    appends ``(name, grid)`` (``_launched_grid``) to the returned list, on a
    device of ``device[0]`` SMs and architecture ``device[1]`` (a list the
    caller sets before each call), with no stream."""
    launched = []

    def entry(name, device_index):
        return lambda *args: launched.append((name, _launched_grid(name, args)))

    patch.setattr(loader, "_sm_count_and_arch", lambda index: tuple(device))
    patch.setattr(loader, "_entry", entry)
    patch.setattr(loader, "_ffi_device", lambda index: None)
    patch.setattr(loader, "current_raw_stream", lambda: lambda index: None)
    patch.setattr(
        loader,
        "tvm_ffi",
        SimpleNamespace(use_raw_stream=lambda *_: contextlib.nullcontext()),
    )
    return launched


def _launch_grid_calls(patch):
    """Patch the registry's ``launch_grid`` so that each call adds one to the
    returned one-item list, then gives its grid (or raises its
    ``ValueError``)."""
    calls = [0]
    launch_grid = loader.launch_grid

    def spy(*args, **kwargs):
        calls[0] += 1
        return launch_grid(*args, **kwargs)

    patch.setattr(loader, "launch_grid", spy)
    return calls


def _host_grid_rows(rule, arch, sm_count):
    """Row counts of ``rule``'s grid on ``arch`` with ``sm_count`` SMs: a few
    rows, a block and multiples of it, about one and two waves (of
    ``max_blocks_per_cta`` blocks a CTA), the kernels' row limit, and each
    cut of the rule (a single number, or keyed by ``arch``) times the SMs and
    the fewest rows of whole waves at or above it: each of them, a row less
    and a row more."""
    per_block = rule.get("rows_per_block", rule.get("rows_per_cta", 16))
    wave = rule.get("ctas_per_sm", 1) * sm_count
    step = wave * rule.get("max_blocks_per_cta", 1) * per_block
    edges = [per_block, 2 * per_block, 3 * per_block, 7 * per_block]
    edges += [wave * per_block, step, 2 * step, 7 * step + 3 * per_block, 2**31 - 1]
    for key in (CUT, STRIDED_CUT):
        cut = rule.get(key)
        if isinstance(cut, dict):
            cut = cut.get(arch)
        if cut is not None:
            edges += [cut * sm_count, -(-cut * sm_count // step) * step]
    rows = {1, 2, 3}
    for edge in edges:
        rows |= {edge - 1, edge, edge + 1}
    return sorted(r for r in rows if 1 <= r < 2**31)


def _host_values(record, strides, ctas):
    """Logical arguments of a launch of ``record`` without a kernel: the
    ``heads``, ``x_tok``, ``x_head``, ``z_tok`` and ``z_head`` of ``strides``
    where the record takes them, the backward's ``ctas``, 0 for every other
    argument."""
    logical = record["logical"]
    values = dict.fromkeys(logical.values(), 0)
    keys = ("heads", "x_tok", "x_head", "z_tok", "z_head")
    for key, value in zip(keys, strides, strict=True):
        if key in logical:
            values[logical[key]] = value
    values["ctas"] = ctas
    return values


def _launched_or_raised(launched, name, values, rows):
    """The grid ``run_module`` launches for module ``name`` (``launched``:
    ``_host_launches``'s list), or the type and text of the ``ValueError`` it
    raises."""
    launched.clear()
    try:
        loader.run_module(name, values, rows=rows, device_index=0)
    except ValueError as error:
        assert not launched, (name, launched)
        return ValueError, str(error)
    ((kernel, grid),) = launched
    assert kernel == name, (kernel, name)
    return grid


def _grid_or_raised(record, rows, sm_count, ctas, arch, packed):
    """``launch_grid``'s grid, or the type and text of its ``ValueError``."""
    try:
        return loader.launch_grid(
            record, rows=rows, sm_count=sm_count, ctas=ctas, arch=arch, packed=packed
        )
    except ValueError as error:
        return ValueError, str(error)


def _check_host_grids(launched, calls, device, name, layouts):
    """``run_module``'s grid (or ``ValueError``) of module ``name`` is
    ``launch_grid``'s, with ``packed`` ``forward_packed`` of the strides and
    ``ctas`` the call's, for each of ``layouts`` (``HOST_STRIDES``), on each
    of ``HOST_GRID_ARCHES`` and ``HOST_GRID_SM_COUNTS`` at
    ``_host_grid_rows`` of its rule as it is now; each ``run_module`` call
    calls ``launch_grid`` (``calls``: ``_launch_grid_calls``'s list) never
    where it computes a forward's grid itself (``_computes_grid``), else
    once.  Returns ``(raised, differ)``: whether some call raised
    ``ValueError``, and the architectures on which the grids of rows read as
    packed and of other rows differ at some row count."""
    record = loader.MODULES[name]
    rule = record["grid_rule"]
    forward = rule["kind"] == "persistent_row_blocks"
    raised, differ = False, set()
    for arch in HOST_GRID_ARCHES:
        want_calls = 0 if forward and _computes_grid(rule, arch) else 1
        for sm_count in HOST_GRID_SM_COUNTS:
            device[:] = (sm_count, arch)
            for rows in _host_grid_rows(rule, arch, sm_count):
                ctas = rows % 4093 + 1
                grids = {}
                for layout in layouts:
                    strides = HOST_STRIDES[layout]
                    packed = loader.forward_packed(*strides)
                    values = _host_values(record, strides, ctas)
                    before = calls[0]
                    got = _launched_or_raised(launched, name, values, rows)
                    made = calls[0] - before
                    want = _grid_or_raised(record, rows, sm_count, ctas, arch, packed)
                    what = (name, arch, sm_count, rows, layout)
                    assert got == want, (*what, got, want)
                    assert made == want_calls, (*what, made, want_calls)
                    if want[0] is ValueError:
                        raised = True
                    else:
                        grids[packed] = want
                if len(set(grids.values())) > 1:
                    differ.add(arch)
    return raised, differ


def test_run_module_launches_the_grids_of_launch_grid():
    """Without a device (``_host_launches``): every record of every grid kind,
    on each of ``HOST_GRID_ARCHES`` (its own and others) and
    ``HOST_GRID_SM_COUNTS`` SMs, at ``_host_grid_rows`` of its rule (its
    cuts times the SMs and a row either side, multiples of its blocks, few
    rows), launches ``launch_grid``'s grid for the call's rows, SM count,
    ``ctas`` and architecture and ``forward_packed`` of its strides (every
    layout of ``HOST_STRIDES`` for a forward of whole waves: rows read as
    packed and not; a packed and a strided one for the others), and raises
    its ``ValueError`` where it does.  A record whose cuts are keyed by
    architecture raises it on one they do not key, and on each of its
    architectures the grids of the two readings differ at some row count
    where its rule has a cut of rows not read as packed.  ``run_module``
    calls ``launch_grid`` never for a forward on its own architectures
    (``_computes_grid``), and once for a forward on an architecture its cuts
    do not key and for a record of every other grid kind."""
    forwards = _forward_names()
    assert any(_keyed(loader.MODULES[name]["grid_rule"]) for name in forwards)
    device = [0, ""]
    with pytest.MonkeyPatch.context() as patch:
        launched = _host_launches(patch, device)
        calls = _launch_grid_calls(patch)
        for name, record in loader.MODULES.items():
            rule = record["grid_rule"]
            if name in forwards:
                layouts = tuple(HOST_STRIDES)
                for arch in record["arches"]:
                    assert _computes_grid(rule, arch), (name, arch)
            else:
                layouts = ("contiguous", "projection")
            raised, differ = _check_host_grids(launched, calls, device, name, layouts)
            assert raised is _keyed(rule), (name, raised)
            if name in forwards and STRIDED_CUT in rule:
                assert differ == set(record["arches"]), (name, differ)
            else:
                assert not differ, (name, differ)


# A cut that a variant of ``FORWARD_RULE_VARIANTS`` deletes from the rule.
DELETE = object()
# Forward rules changed at run time, as functions of the record's
# architectures giving the cuts to set (``DELETE``: delete the cut): the cut
# of rows not read as packed above the other; cuts of one number (on any
# architecture), also beside a cut keyed by architecture; cuts of 0 (whole
# waves at any rows); one cut keyed by the first architecture only; no cut of
# rows not read as packed; no cut (whole waves at any rows, as the mid
# forward); a cut of rows not read as packed only; cuts of None (no cut, as
# ``launch_grid`` reads them).  Where a variant sets a cut a forward's rule
# has not, the rule gains it.
FORWARD_RULE_VARIANTS = {
    "strided_above": lambda arches: {
        CUT: dict.fromkeys(arches, 400),
        STRIDED_CUT: dict.fromkeys(arches, 5600),
    },
    "numbers": lambda arches: {CUT: 100, STRIDED_CUT: 50},
    "number_and_keyed": lambda arches: {
        CUT: 6000,
        STRIDED_CUT: dict.fromkeys(arches, 500),
    },
    "zero": lambda arches: {
        CUT: dict.fromkeys(arches, 0),
        STRIDED_CUT: dict.fromkeys(arches, 0),
    },
    "packed_first_only": lambda arches: {
        CUT: {arches[0]: 8192},
        STRIDED_CUT: dict.fromkeys(arches, 500),
    },
    "strided_first_only": lambda arches: {
        CUT: dict.fromkeys(arches, 5600),
        STRIDED_CUT: {arches[0]: 400},
    },
    "no_strided": lambda arches: {STRIDED_CUT: DELETE},
    "no_cut": lambda arches: {CUT: DELETE, STRIDED_CUT: DELETE},
    "strided_only": lambda arches: {
        CUT: DELETE,
        STRIDED_CUT: dict.fromkeys(arches, 400),
    },
    "none": lambda arches: {CUT: None, STRIDED_CUT: None},
}
# The layouts of those rules' host comparison: rows read as packed (of 48
# heads and of one) and not (one token, a projection, head-major rows and a
# broadcast over tokens).
VARIANT_LAYOUTS = (
    "contiguous",
    "one_head",
    "one_token",
    "projection",
    "head_major",
    "broadcast",
)


def test_run_module_reads_the_forward_rule_at_each_call():
    """Without a device (``_host_launches``), each forward record's rule
    (every forward of whole waves) changed at run time to each of
    ``FORWARD_RULE_VARIANTS`` in turn (and then restored): ``run_module``'s
    grid at the next call is ``launch_grid``'s for the changed rule, or its
    ``ValueError``, on every architecture of ``HOST_GRID_ARCHES`` and
    ``HOST_GRID_SM_COUNTS`` SMs, at ``_host_grid_rows`` of the changed rule,
    for ``VARIANT_LAYOUTS``; it calls ``launch_grid`` once where
    ``_computes_grid`` does not hold for the changed rule (a cut of one
    number or None, or one keyed without the architecture), else never."""
    device = [0, ""]
    with pytest.MonkeyPatch.context() as patch:
        launched = _host_launches(patch, device)
        calls = _launch_grid_calls(patch)
        for name in _forward_names():
            record = loader.MODULES[name]
            rule = record["grid_rule"]
            for variant, cuts in FORWARD_RULE_VARIANTS.items():
                with pytest.MonkeyPatch.context() as change:
                    for key, cut in cuts(record["arches"]).items():
                        if cut is DELETE:
                            change.delitem(rule, key, raising=False)
                        else:
                            change.setitem(rule, key, cut)
                    raised, _ = _check_host_grids(
                        launched, calls, device, name, VARIANT_LAYOUTS
                    )
                    assert raised is _keyed(rule), (name, variant, raised)
            raised, _ = _check_host_grids(
                launched, calls, device, name, VARIANT_LAYOUTS
            )
            assert raised is _keyed(rule), (name, raised)


# The pipelined forward's grids on this device, forced: its record's cuts
# patched to 1 row per SM (whole waves at every row count of its bands) and to
# 2**31 rows per SM (one CTA a block), and as shipped.  Its rows: packed rows
# (contiguous ``[T, H, 128]`` and flat ``[T * H, 128]``); ``x`` and ``z``, or
# one of them, broadcast over tokens (token stride 0, rows the kernel does not
# read as packed: on sm_100a and sm_103a the pipelined forward reads them at
# their token stride; on sm_107a its program of packed rows does not run
# them); and the same rows as one token ``[1, T * H, 128]`` (its token stride
# passed as 0: on sm_100a and sm_103a rows the kernel does not read as packed;
# on sm_107a the program of packed rows runs it, passed the token stride
# ``H * 128``).  Tokens of 47 heads: partial last row blocks.
PIPE_GRID_LAYOUTS = (
    "contiguous",
    "flat",
    "broadcast",
    "x_broadcast",
    "z_broadcast",
    "one_token",
)
PIPE_GRID_HEADS = 47


def _pipe_grid_tokens(arch: str, sm_count: int) -> list[int]:
    """Token counts of ``PIPE_GRID_HEADS`` heads in each band of packed rows
    of ``arch``'s pipelined forward (``FORWARD_BANDS``) on ``sm_count`` SMs:
    the fewest past the band's first row per SM, the most up to its last
    (four times its first where it has no end) and their middle."""
    tokens, first = [], 1
    for last, role in loader.FORWARD_BANDS[arch]:
        if role in ("fwd_pipe", "fwd_pipe_packed"):
            end = 4 * first if last is None else int(last)
            low = -(-first * sm_count // PIPE_GRID_HEADS)
            high = end * sm_count // PIPE_GRID_HEADS
            tokens += [low, (low + high) // 2, high]
        if last is not None:
            first = int(last) + 1
    return sorted(set(tokens))


@pytest.mark.parametrize("layout", PIPE_GRID_LAYOUTS)
def test_pipelined_forward_gives_the_same_bits_on_both_grids(layout, launches):
    """The pipelined forward of this device (``fwd_pipe`` or
    ``fwd_pipe_packed``), at ``_pipe_grid_tokens`` of ``PIPE_GRID_HEADS``
    heads, in each of
    ``PIPE_GRID_LAYOUTS``: with its cuts forced to whole waves and to one CTA
    a block (the rule changed at run time) and as shipped, the call launches
    the pipelined forward on ``launch_grid``'s grid (``_record_launches``:
    the grid the kernel is launched with) for the device's SM count and
    architecture and whether the kernel reads the rows as packed
    (``forward_packed``; for the program of packed rows only the same grid
    either way), and gives the same ``y`` bits on every grid; at some row
    count the two forced grids differ, and where the rule has a cut of rows
    not read as packed the shipped grid at some row count is not that of the
    other reading of the rows."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sms = loader.device_sm_count(device_index)
    routes = loader.ROUTES[arch]
    role = "fwd_pipe" if "fwd_pipe" in routes else "fwd_pipe_packed"
    name = routes[role]
    record = loader.MODULES[name]
    broadcast = layout.endswith("broadcast")
    if broadcast and role == "fwd_pipe_packed":
        pytest.skip(f"{arch}: rows broadcast over tokens run no pipelined forward")
    rule = record["grid_rule"]
    cuts = [key for key in (CUT, STRIDED_CUT) if key in rule]
    # rows broadcast over tokens and one token: not packed rows to the
    # pipelined forward of a rule with a cut of such rows
    packed = not (broadcast or layout == "one_token") or STRIDED_CUT not in rule
    differ = other_reading = 0
    for T in _pipe_grid_tokens(arch, sms):
        rows = T * PIPE_GRID_HEADS
        assert loader.module_for_rows(arch, "fwd", rows, sms) == name, rows
        if layout == "one_token":
            tokens, heads = 1, rows
        else:
            tokens, heads = T, PIPE_GRID_HEADS
        base = "flat" if layout == "flat" else "contiguous"
        x, z, w, _ = make(tokens, heads, layout=base, seed=41)
        if layout in ("broadcast", "x_broadcast"):
            x = x[:1].expand(x.shape)
        if layout in ("broadcast", "z_broadcast"):
            z = z[:1].expand(z.shape)
        ys, grids = [], []
        for forced in (None, 1, 2**31):
            with pytest.MonkeyPatch.context() as patch:
                if forced is not None:
                    for key in cuts:
                        patch.setitem(rule, key, {each: forced for each in rule[key]})
                launched = _record_launches(patch)
                launches.clear()
                ys.append(forward(x, z, w, EPS))
                assert launches == [name], (rows, forced, launches)
                ((kernel, grid),) = launched
                assert kernel == name, (kernel, name)
                want = loader.launch_grid(
                    record, rows=rows, sm_count=sms, arch=arch, packed=packed
                )
                assert grid == want, (layout, rows, forced, grid, want)
                if forced is None:
                    other = loader.launch_grid(
                        record, rows=rows, sm_count=sms, arch=arch, packed=not packed
                    )
                    other_reading += other != grid
                grids.append(grid)
        blocks = -(-rows // 16)
        assert grids[2] == (blocks, 1, 1), (rows, grids)
        assert grids[1][0] <= blocks and grids[1][1:] == (1, 1), (rows, grids)
        differ += grids[1] != grids[2]
        for y, forced in zip(ys[1:], (1, 2**31), strict=True):
            assert torch.equal(y.view(torch.int16), ys[0].view(torch.int16)), (
                f"{layout} {tokens}x{heads} on {arch}: y with the cut at "
                f"{forced} rows per SM"
            )
    assert differ, (arch, layout, "the forced grids never differ")
    assert other_reading or STRIDED_CUT not in rule, (
        arch,
        layout,
        "the other reading of the rows never gives another grid",
    )


# (rows, SMs) of the short forward: one CTA per 16-row block, up to 12 rows per
# SM on sm_100a and sm_103a and 59 on sm_107a (each row count is checked on
# every architecture: past an architecture's band the next band's forward runs
# it), so at most the SMs less 16 CTAs on sm_100a and sm_103a, and on sm_107a,
# at the 212 and 216 SMs listed (the bound needs 205 or more,
# SHORT_FORWARD_MIN_SMS), at most one wave of the 4 CTAs per SM its launch
# bounds give on the SMs less 16.
SHORT_FORWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48: 3 blocks
    (13 * 7, 148),
    (21 * 48, 148),  # 6.8 rows per SM on B200
    (10 * 148, 148),  # B200 at 10 rows per SM: 93 blocks
    (10 * 152, 152),  # GB300 at 10 rows per SM: 95 blocks
    (12 * 148, 148),  # B200 at 12 rows per SM: 111 blocks
    (12 * 152, 152),  # GB300 at 12 rows per SM: 114 blocks
    (10 * 212, 212),  # R200 at 10 rows per SM: 133 blocks
    (30 * 212, 212),  # R200 at 30 rows per SM: 398 blocks
    (1786 * 7, 212),  # T 1786, H 7: 782 blocks, 58.97 rows per SM
    (59 * 212, 212),  # R200 at 59 rows per SM: 782 blocks
    (59 * 216, 216),  # 216-SM R200 at 59 rows per SM: 797 blocks
]
# The most CTAs of the short forward's grid on sm_count SMs, from
# SHORT_FORWARD_MIN_SMS SMs (64 on sm_100a and sm_103a, 205 on sm_107a; on
# fewer SMs, as on a small MIG instance, the grid at the end of the
# architecture's band has more): at most the SMs less 16 on sm_100a and
# sm_103a, one wave of 4 CTAs per SM on the SMs less 16 on sm_107a.
SHORT_FORWARD_MAX_CTAS = {
    "sm_100a": lambda sm_count: sm_count - 16,
    "sm_103a": lambda sm_count: sm_count - 16,
    "sm_107a": lambda sm_count: 4 * (sm_count - 16),
}
# The fewest SMs on which the short forward's grid at the end of the
# architecture's band keeps SHORT_FORWARD_MAX_CTAS: ceil(12 * SMs / 16) <= SMs -
# 16 from 64 SMs on sm_100a and sm_103a, ceil(59 * SMs / 16) <= 4 * (SMs - 16)
# from 205 SMs on sm_107a (212 and 216 keep it; a small MIG instance does not).
SHORT_FORWARD_MIN_SMS = {"sm_100a": 64, "sm_103a": 64, "sm_107a": 205}


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_FORWARD_GRIDS)
def test_short_forward_grid_is_one_cta_a_block_beside_16_held_sms(rows, sm_count):
    name = loader.ROUTES["sm_100a"]["fwd_short"]
    routed = []
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert loader.ROUTES[arch]["fwd_short"] == name
        end = _short_end(arch, sm_count)
        want = loader.ROUTES[arch][STRIDED_FORWARD_BANDS[arch][1][1]]
        if rows <= end:
            routed.append(arch)
            want = name
        got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
        assert got == want
    assert routed, (rows, sm_count)
    record = loader.MODULES[name]
    assert record["grid_rule"] == {"kind": "row_blocks", "rows_per_cta": 16}
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    for arch in routed:
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
        assert grid == (-(-rows // 16), 1, 1), arch
        assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, grid)
        got = loader.launch_grid(
            record, rows=rows, sm_count=sm_count, ctas=7, arch=arch
        )
        assert got == grid, arch
    # the plain forward: one 64-thread CTA per 8-row block
    fwd = loader.MODULES[loader.route_module("sm_100a", "fwd")]
    assert loader.launch_grid(fwd, rows=rows, sm_count=sm_count, arch="sm_100a") == (
        -(-rows // 8),
        1,
        1,
    )


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a", "sm_107a"])
def test_short_forward_grid_bound_holds_from_its_fewest_sms(arch):
    name = loader.ROUTES[arch]["fwd_short"]
    record = loader.MODULES[name]
    for sm_count in range(1, 1025):
        rows = _short_end(arch, sm_count)
        got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
        assert got == name
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
        fits = grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count)
        assert fits == (sm_count >= SHORT_FORWARD_MIN_SMS[arch]), (
            arch,
            sm_count,
            grid,
        )


def test_short_forward_grid_at_its_bands_end_keeps_the_bound_on_this_device():
    """On the running device, the short forward's grid at the end of the
    architecture's band (the most rows it runs) keeps SHORT_FORWARD_MAX_CTAS; skipped
    where the device has fewer SMs than the bound needs."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    name = loader.ROUTES[arch].get("fwd_short")
    if name is None:
        pytest.skip(f"{arch} routes no fwd_short module")
    if sm_count < SHORT_FORWARD_MIN_SMS[arch]:
        pytest.skip(
            f"{sm_count} SMs: the short forward's grid bound on {arch} needs "
            f"{SHORT_FORWARD_MIN_SMS[arch]} SMs or more"
        )
    rows = _short_end(arch, sm_count)
    assert loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False) == name
    assert loader.module_for_rows(arch, "fwd", rows + 1, sm_count, packed=False) != name
    grid = loader.launch_grid(
        loader.MODULES[name], rows=rows, sm_count=sm_count, arch=arch
    )
    assert grid == (-(-rows // 16), 1, 1)
    assert grid[0] <= SHORT_FORWARD_MAX_CTAS[arch](sm_count), (arch, sm_count, grid)


# The arguments of the tiny forwards: no stride.
TINY_ARGUMENTS = ["x", "z", "w", "y", "rows", "eps", "grid_x", "grid_y", "grid_z"]


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_l32_tiny_forward_runs_packed_rows_one_cta_a_4_row_block(sm_count):
    """The 32-lane tiny forward, delivered and routed on sm_100a and sm_103a
    only, in the first band of packed rows there, for packed rows only: one
    128-thread CTA per 4-row block whatever the SMs, reading no stride (the
    tiny forward's arguments: ``x``, ``z``, ``w``, ``y``, the rows and
    ``eps``); the tiny forward runs the rows past its band."""
    name = loader.ROUTES["sm_100a"]["fwd_l32_tiny"]
    record = loader.MODULES[name]
    assert record["role"] == "fwd_l32_tiny"
    assert record["grid_rule"] == {"kind": "row_blocks", "rows_per_cta": 4}
    assert record["launch"]["block"] == (128, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record.get("workspace") is None
    assert record["arches"] == ["sm_100a", "sm_103a"]
    assert [key for _kind, key in record["arg_plan"]] == TINY_ARGUMENTS
    tiny = loader.ROUTES["sm_100a"]["fwd_tiny"]
    assert record["arg_plan"] == loader.MODULES[tiny]["arg_plan"]
    assert "fwd_l32_tiny" not in loader.ROUTES["sm_107a"]
    for arch in ("sm_100a", "sm_103a"):
        assert loader.ROUTES[arch]["fwd_l32_tiny"] == name
        first, end = _tiny_band(arch, sm_count, "fwd_l32_tiny")
        assert first == 1, (arch, first)
        for rows in (1, 3, 4, 5, 7, 8, 9, end - 1, end):
            assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
            got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
            assert got != name, (arch, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
            assert grid == (-(-rows // 4), 1, 1)
            got = loader.launch_grid(
                record, rows=rows, sm_count=sm_count, ctas=7, arch=arch
            )
            assert got == grid
        got = loader.module_for_rows(arch, "fwd", end + 1, sm_count)
        assert got == loader.ROUTES[arch]["fwd_tiny"], (arch, got)


@pytest.mark.parametrize("sm_count", [148, 152, 212, 216])
def test_tiny_forward_runs_packed_rows_one_cta_an_8_row_block(sm_count):
    """The tiny forward, delivered and routed on every architecture for packed
    rows only (on sm_100a and sm_103a in the band past the 32-lane tiny
    forward's, on sm_107a in the first band): one 128-thread CTA per 8-row
    block whatever the SMs, reading no stride (its arguments are ``x``,
    ``z``, ``w``, ``y``, the rows and ``eps``)."""
    name = loader.ROUTES["sm_100a"]["fwd_tiny"]
    record = loader.MODULES[name]
    assert record["grid_rule"] == {"kind": "row_blocks", "rows_per_cta": 8}
    assert record["launch"]["block"] == (128, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 0
    assert record["arches"] == ["sm_100a", "sm_103a", "sm_107a"]
    assert [key for _kind, key in record["arg_plan"]] == TINY_ARGUMENTS
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert loader.ROUTES[arch]["fwd_tiny"] == name
        first, end = _tiny_band(arch, sm_count)
        assert (first == 1) == (arch == "sm_107a"), (arch, first)
        if first > 1:
            got = loader.module_for_rows(arch, "fwd", first - 1, sm_count)
            assert got != name, (arch, first)
        for rows in sorted({first, first + 6, first + 7, first + 8, end}):
            assert loader.module_for_rows(arch, "fwd", rows, sm_count) == name
            got = loader.module_for_rows(arch, "fwd", rows, sm_count, packed=False)
            assert got != name, (arch, rows)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
            assert grid == (-(-rows // 8), 1, 1)
            got = loader.launch_grid(
                record, rows=rows, sm_count=sm_count, ctas=7, arch=arch
            )
            assert got == grid
        assert loader.module_for_rows(arch, "fwd", end + 1, sm_count) != name


# (T, H) whose rows both a tiny forward (packed rows) and the short forward
# (other layouts) run on 148 SMs or more: on sm_100a and sm_103a the 32-lane
# tiny forward up to its cut and the tiny forward past it, on sm_107a the tiny
# forward.
TINY_AND_SHORT_SHAPES = [(1, 1), (2, 48), (8, 1), (13, 7), (255, 2), (37, 48)]


@pytest.mark.parametrize(("T", "H"), TINY_AND_SHORT_SHAPES)
def test_tiny_forward_gives_the_short_forwards_bits(
    T, H, launches, monkeypatch: pytest.MonkeyPatch
):
    """The tiny forwards have the short forward's arithmetic (the 32-lane
    one its sum of squares on the lane pairs of the 16-lane row): on the same
    rows, ``z`` once packed (the tiny forward of the rows'
    ``FORWARD_BANDS`` band) and once the gate columns of a wider projection
    (the short forward), ``y`` is the same bits.  On sm_100a and sm_103a,
    which route both tiny forwards, the packed rows run once more with the
    other tiny forward as the one band of packed rows (the bands of other
    layouts unchanged): the same bits."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    rows, end = T * H, _short_end(arch, loader.device_sm_count(device_index))
    if rows > end:
        pytest.skip(f"{rows} rows are past the short forward's band on {arch}")
    tiny = _module("fwd", rows)
    role = loader.MODULES[tiny]["role"]
    assert role in ("fwd_l32_tiny", "fwd_tiny"), (arch, tiny)
    x, z, w, _ = make(max(T, 2), H, layout="projection", seed=90)
    x, z = x[:T], z[:T]
    y_short = forward(x, z, w, EPS)
    y_tiny = forward(x, z.contiguous(), w, EPS)
    routes = _routes()
    if T == 1:
        # a single token's stride is never read: both calls read packed rows
        assert launches == [tiny] * 2, launches
    else:
        assert launches == [routes["fwd_short"], tiny], launches
    assert torch.equal(y_short, y_tiny)
    other = "fwd_tiny" if role == "fwd_l32_tiny" else "fwd_l32_tiny"
    if other in routes:
        bands = {**loader.FORWARD_BANDS, arch: ((None, other),)}
        monkeypatch.setattr(loader, "FORWARD_BANDS", bands)
        launches.clear()
        y_other = forward(x, z.contiguous(), w, EPS)
        assert launches == [routes[other]], launches
        assert torch.equal(y_short, y_other)


# (rows, SMs) of the small-row backwards: one cluster whatever the rows and the
# SMs, up to each program's ``max_rows``.
SHORT_BACKWARD_GRIDS = [
    (1, 148),
    (48, 148),  # T 1, H 48
    (13 * 7, 152),
    (5 * 48, 148),
    (256, 148),
    (257, 160),
    (512, 152),
    (1024, 160),
    (1024, 212),
]
# The threads of one row in every backward program (16 lanes a row).
ROW_LANES = 16
# role -> (threads a CTA, teams a CTA, CTAs of the cluster) of each small-row
# backward (``BAND_ROLES``), one cluster each, whose CTAs each run their
# teams' row blocks: the short backward of a BF16 weight gradient (on every
# architecture; one 256-thread team a CTA, CTA ``r`` running the 16-row blocks
# ``r``, ``r + 16``, ...) and the exact tiny backwards (``_t512``: 4 teams of
# the 128-thread CTA of the deterministic backward at 2e989556 a CTA, team ``q``
# of CTA ``r`` running the 8-row block ``r + 16 q``).  A row block holds
# :func:`_cluster_block_rows` rows.
SMALL_ROW_BACKWARDS = {
    "bwd_bf16_short": (256, 1, 16),
    "bwd_bf16_t512": (512, 4, 16),
    "bwd_f32_t512": (512, 4, 16),
}


def _cluster_block_rows(name: str) -> int:
    """The rows of one row block of the small-row backward module ``name``
    (grid kind ``cluster``), derived from its record: the threads of its
    launch block over the teams of its role's CTA (``SMALL_ROW_BACKWARDS``),
    over ``ROW_LANES`` threads a row: 256 / 1 / 16 = 16 rows for the short
    backward, 512 / 4 / 16 = 8 rows (a team's block) for the exact tiny
    backwards."""
    record = loader.MODULES[name]
    assert record["grid_rule"]["kind"] == "cluster", (name, record["grid_rule"])
    threads = int(record["launch"]["block"][0])
    teams = SMALL_ROW_BACKWARDS[record["role"]][1]
    assert threads % (teams * ROW_LANES) == 0, (name, threads, teams)
    return threads // teams // ROW_LANES


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_BACKWARD_GRIDS)
def test_small_row_backwards_are_one_cluster(rows, sm_count):
    """Every small-row backward (the module each ``BAND_ROLES`` role routes,
    ``SMALL_ROW_BACKWARDS``): one module a role, shipped for exactly the
    architectures that route it; one cluster of its grid rule's ``cluster``
    CTAs of its role's threads whatever the rows and the SMs, up to its
    rule's ``max_rows``, which is one row block (:func:`_cluster_block_rows`)
    for each team of the cluster, so up to it no team runs more than one
    block; no workspace and no chunks."""
    assert set(loader.BAND_ROLES) == set(SMALL_ROW_BACKWARDS), loader.BAND_ROLES
    for role, (threads, teams, ctas) in SMALL_ROW_BACKWARDS.items():
        routed_on = [arch for arch in loader.ARCHES if role in loader.ROUTES[arch]]
        names = {loader.ROUTES[arch][role] for arch in routed_on}
        assert len(names) == 1, (role, routed_on, names)
        (name,) = names
        record = loader.MODULES[name]
        assert record["role"] == role
        assert record["arches"] == routed_on, (name, record["arches"])
        rule = record["grid_rule"]
        assert rule["kind"] == "cluster", name
        assert set(rule) == {"cluster", "kind", "max_rows"}, name
        assert int(rule["cluster"]) == ctas, name
        assert record["launch"]["block"] == (threads, 1, 1), name
        assert record["launch"]["cluster"] == (ctas, 1, 1), name
        block = _cluster_block_rows(name)
        assert int(rule["max_rows"]) == ctas * teams * block, (name, rule, block)
        assert loader.workspace_layout(name) == (), name
        assert loader.workspace_bytes(name) == 0, name
        if rows > int(rule["max_rows"]):
            continue
        assert loader.backward_ctas(name, rows, sm_count) == ctas
        grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=ctas)
        assert grid == (ctas, 1, 1)
        assert -(-rows // block) <= ctas * teams, (name, rows)
        with pytest.raises(ValueError, match="chunks"):
            loader.backward_chunks(name, rows)


@pytest.mark.parametrize(("rows", "sm_count"), SHORT_BACKWARD_GRIDS)
def test_short_backward_is_one_cluster_up_to_its_max_rows(rows, sm_count):
    """The short backward (``bwd_bf16_short``): one module, routed on the
    architectures whose BF16 ``BACKWARD_BANDS`` start with its band, a count
    of rows ``(ROWS, n)`` with ``n`` at most its rule's ``max_rows`` (no FP32
    band runs it); ``module_for_rows`` runs it for 1 to ``n`` rows whatever
    the SMs and the next band's module (:func:`_band_module`) from ``n + 1``
    rows; one cluster of 16 CTAs of 256 threads, 6144 bytes of dynamic shared
    memory each, so that up to ``max_rows`` no CTA runs more than one 16-row
    block (:func:`_cluster_block_rows`); no workspace."""
    role = "bwd_bf16_short"
    routed_on = [arch for arch in loader.ARCHES if role in loader.ROUTES[arch]]
    assert routed_on, role
    (name,) = {loader.ROUTES[arch][role] for arch in routed_on}
    record = loader.MODULES[name]
    rule = record["grid_rule"]
    most = int(rule["max_rows"])
    assert rule == {"cluster": 16, "kind": "cluster", "max_rows": most}, rule
    assert record["launch"]["block"] == (256, 1, 1)
    assert record["launch"]["cluster"] == (16, 1, 1)
    assert record["launch"]["dynamic_smem_bytes"] == 6144
    assert record["arches"] == routed_on, record["arches"]
    block = _cluster_block_rows(name)
    assert block == 16, block
    for arch in loader.ARCHES:
        banded = [band for _, band in loader.BACKWARD_BANDS[arch]["bwd_bf16"]]
        assert (role in banded) == (arch in routed_on), (arch, banded)
        assert role not in [band for _, band in loader.BACKWARD_BANDS[arch]["bwd_f32"]]
    for arch in routed_on:
        last, band = loader.BACKWARD_BANDS[arch]["bwd_bf16"][0]
        assert band == role and isinstance(last, tuple), (arch, last, band)
        assert last[0] == loader.ROWS and 0 < int(last[1]) <= most, (arch, last)
        end = int(last[1])
        got = loader.module_for_rows(arch, "bwd_bf16", rows, sm_count)
        if rows <= end:
            assert got == name, (arch, rows)
        else:
            assert got != name, (arch, rows)
        after = loader.module_for_rows(arch, "bwd_bf16", end + 1, sm_count)
        assert after != name, arch
        assert after == _band_module(arch, "bwd_bf16", end + 1, sm_count), arch
    if rows > most:
        return
    assert loader.backward_ctas(name, rows, sm_count) == 16
    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=16)
    assert grid == (16, 1, 1)
    assert -(-rows // block) <= grid[0]
    assert loader.workspace_layout(name) == () and loader.workspace_bytes(name) == 0


# Small-row backward shapes (T, H, layout): T * H rows in the bands in every
# layout, and row counts past them.
SHORT_BACKWARD_SHAPES = [
    (1, 1, "contiguous"),
    (1, 48, "contiguous"),
    (2, 48, "contiguous"),
    (5, 48, "contiguous"),
    (13, 7, "contiguous"),
    (8, 32, "contiguous"),
    (16, 16, "contiguous"),
    (256, 1, "flat"),
    (5, 48, "projection"),
    (13, 7, "head_major"),
    (2, 48, "misaligned"),
    (257, 1, "flat"),
    (37, 7, "contiguous"),
    (513, 1, "flat"),
    (621, 1, "flat"),
    (1025, 1, "flat"),
]


def _short_backward(rows: int, dw_dtype: torch.dtype) -> tuple[str, str | None]:
    """``(the module module_for_rows runs, that module where it is a small-row
    backward or None)`` for ``rows`` rows and a ``dw_dtype`` weight gradient on
    this device."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    role = "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"
    name = loader.module_for_rows(
        arch, role, rows, loader.device_sm_count(device_index)
    )
    small = loader.MODULES[name]["role"] in loader.BAND_ROLES
    return name, (name if small else None)


@pytest.mark.parametrize("T,H,layout", SHORT_BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_short_backward_dw_is_the_same_bits_and_takes_no_workspace(
    T, H, layout, dw_dtype, launches, monkeypatch, record_property
):
    """Judged against the naive implementation and the same bits on every
    launch (``check_all``), the module ``module_for_rows`` names launched; on a
    small-row backward (the rows of its ``BACKWARD_BANDS`` band) no stream
    workspace is made, a caller workspace holding anything is left as it is (the results
    the same bits) and one that is not a workspace is refused."""
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    name, short = _short_backward(T * H, dw_dtype)
    (x, z, w, dy), _ = check_all(
        record_property,
        lambda s: make(T, H, layout=layout, seed=s),
        dw_dtype=dw_dtype,
        repeats=5,
        seed=80,
    )
    fwd = _forward_module(T * H, _packed(T, H, layout))
    assert launches == [fwd, name, name] * _launch_count(5), launches
    if short is None:
        return
    assert pkg._STREAM_WORKSPACES == {}

    def out():
        return torch.empty(D, dtype=dw_dtype, device=x.device)

    want = backward(dy, x, z, w, EPS, grad_weight=out())
    ws = workspace_for(T * H, x.device)
    ws.fill_(7)
    got = backward(dy, x, z, w, EPS, grad_weight=out(), workspace=ws)
    torch.cuda.synchronize()
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)
    assert bool((ws == 7).all()), "the small-row backward wrote the workspace"
    with pytest.raises(ValueError, match="deterministic mode"):
        backward(dy, x, z, w, EPS, workspace=ws.float())
    with pytest.raises(ValueError, match="uint8"):
        backward(dy, x, z, w, EPS, workspace=ws[1:])
    assert pkg._STREAM_WORKSPACES == {}


@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_short_backward_accumulates_and_skips_the_weight_gradient(
    dw_dtype, record_property
):
    """On the small-row backward of 240 rows of a ``dw_dtype`` weight
    gradient (the short backward of a BF16 weight gradient, else the exact
    tiny backward): ``accumulate_grad_weight`` adds
    to the accumulator before the one rounding (the same bits on every
    launch; on the inputs of every draw of ``judge`` judged against the naive
    implementation's BF16 sum), ``compute_grad_weight=False`` (on a gate and
    ``grad_out`` sliced from projections, rows that are not packed: on packed
    rows of sm_100a and sm_103a the build of packed rows only runs instead)
    leaves ``grad_weight`` as it is, and ``dx`` and ``dz`` are the same bits
    either way."""
    _name, short = _short_backward(5 * 48, dw_dtype)
    if short is None:
        pytest.skip("this architecture routes no small-row backward at 240 rows")

    def run(j, s):
        x, z, w, dy = make(5, 48, seed=s)
        dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
        first = None
        for _ in range(3):
            main = torch.full((D,), 0.25, dtype=dw_dtype, device="cuda")
            dx, dz, dw = backward(
                dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
            )
            assert dw is main
            assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
            first = main if first is None else first
            assert torch.equal(main, first)
        return silu_draw(
            x,
            z,
            w,
            dy,
            EPS,
            [{"accumulated dw": first}],
            {"accumulated dw": "dw"},
            starts={"accumulated dw": torch.full((D,), 0.25, dtype=dw_dtype)},
        )

    judge(record_property, run, seed=81)
    x, z, w, dy = make(5, 48, layout="projection", seed=81)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    main = torch.full((D,), 3.0, dtype=dw_dtype, device="cuda")
    dx, dz, dw = backward(dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False)
    torch.cuda.synchronize()
    assert dw is None
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    assert torch.equal(main, torch.full_like(main, 3.0))


def test_calls_without_workspace_share_one_buffer_per_stream():
    """Calls without a workspace keep one per (device, stream, backward
    module): the module's buffers (``_expected_layout``) as views of their
    dtypes (``WORKSPACE_DTYPES``), zero after the calls."""
    # 1043 rows: past every small-row backward band, a backward with a workspace
    x, z, w, dy = make(149, 7)
    for _ in range(2):
        backward(dy, x, z, w, 1e-6)
    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    arch = loader.device_arch(device)
    stream = torch.cuda.current_stream().cuda_stream
    held = {k: v for k, v in pkg._STREAM_WORKSPACES.items() if k[1] == stream}
    assert (device, stream, _backward_module(149 * 7)) in held, list(held)
    for key, views in held.items():
        layout = _expected_layout(arch, loader.MODULES[key[2]]["role"])
        assert sorted(views) == sorted(b for b, _, _ in layout), key
        for buffer, _, elements in layout:
            assert views[buffer].dtype == WORKSPACE_DTYPES[buffer], (key, buffer)
            assert views[buffer].numel() == elements, (key, buffer)
        for view in views.values():
            assert not bool(view.view(torch.uint8).any())
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        backward(dy, x, z, w, 1e-6)
    side.synchronize()
    assert any(k[1] == side.cuda_stream for k in pkg._STREAM_WORKSPACES)


def test_checkpointed_steps_keep_only_the_stream_workspace(monkeypatch):
    """A non-reentrant checkpointed forward keeps nothing
    alive but its output, and once the first step has run, every checkpointed
    forward and backward leaves ``torch.cuda.memory_allocated()`` unchanged.  The
    buffers kept across steps are the workspace of the backward without a
    caller workspace, one per (device, stream, backward module) of
    ``workspace_bytes`` bytes, and the device's reserve of zero-filled
    workspaces for CUDA Graph captures (one allocation).  The first step's
    kept bytes are requested bytes, because the sizes of the cached blocks
    that serve a request depend on earlier tests."""
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVES", {})
    # 1056 rows: past every small-row backward band (a backward with a
    # workspace), and small enough for the caching allocator's small pool, whose
    # blocks are the requested sizes rounded up to 512 bytes.
    x, z, w, dy = make(22, 48, seed=63)
    leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
    for leaf in leaves:
        leaf.grad = torch.zeros_like(leaf)

    def checkpointed():
        return torch.utils.checkpoint.checkpoint(
            gated_rmsnorm, *leaves, EPS, use_reentrant=False
        )

    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["requested_bytes.all.current"]
    checkpointed().backward(dy)  # loads the modules, allocates the workspace
    torch.cuda.synchronize()
    kept = torch.cuda.memory_stats()["requested_bytes.all.current"] - before
    nbytes = workspace_bytes(1, x.device)
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        _backward_module(22 * 48),
    )
    assert list(pkg._STREAM_WORKSPACES) == [key]
    for view in pkg._STREAM_WORKSPACES[key].values():
        assert view.untyped_storage().nbytes() == nbytes
    assert nbytes == WORKSPACE_BYTES[loader.device_arch(x.device.index)], nbytes
    (block,) = pkg._RESERVES[torch.cuda.current_device()]["blocks"]
    reserved = block.untyped_storage().nbytes()
    # 256-byte aligned workspaces: 29440 bytes each (about 1.8 MiB in all) on
    # sm_100a and sm_103a, 36608 bytes each (about 2.2 MiB) on sm_107a
    stride = RESERVE_STRIDE[loader.device_arch(x.device.index)]
    assert stride == -(-nbytes // 256) * 256, (stride, nbytes)
    assert reserved == pkg._RESERVE_SIZE * stride, reserved
    assert nbytes + reserved <= kept < nbytes + reserved + 512, (kept, nbytes)
    base = torch.cuda.memory_allocated()
    for step in range(3):
        y = checkpointed()
        torch.cuda.synchronize()
        retained = torch.cuda.memory_allocated() - base - y.untyped_storage().nbytes()
        assert retained == 0, (step, retained)
        y.backward(dy)
        del y
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == base, step
    assert list(pkg._STREAM_WORKSPACES) == [key]


# Values a caller might pass as ``deterministic``.
MODES = (True, False, None, 0, 1, "False", numpy.bool_(False))


def test_no_entry_point_takes_a_mode_keyword(launches):
    """The mode is in the name (``nondet``): no entry point of the package has a
    ``deterministic`` parameter, and passing one is a ``TypeError`` before any
    launch, whatever its value and PyTorch's deterministic-algorithms setting;
    the autograd function takes no fifth input."""
    x, z, w, dy = make(5, 48, seed=64)
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    calls = {
        "cake_rmsnorm_silu_gate_nondet_train_forward": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_forward(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet(x, z, w, EPS, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace(1, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(1, device, **m),
        "cake_rmsnorm_silu_gate_nondet_train_reset_workspace": lambda **m: pkg.cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws, **m),
        "is_cake_rmsnorm_silu_gate_nondet_train_supported": lambda **m: pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(device, **m),
    }  # fmt: skip
    assert set(calls) | {"CakeRMSNormSiLUGateNondetFunction"} == set(pkg.__all__) - {
        "ARCHES",
        "SUPPORTED_COMPUTE_CAPABILITIES",
    }
    for name in calls:
        assert "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
    assert list(
        inspect.signature(pkg.CakeRMSNormSiLUGateNondetFunction.forward).parameters
    ) == ["ctx", "x", "z", "weight", "eps"]
    for enabled, warn_only in ((False, False), (True, False), (True, True)):
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        for call in calls.values():
            for value in MODES:
                with pytest.raises(TypeError, match="deterministic"):
                    call(deterministic=value)
        for value in MODES:
            with pytest.raises(TypeError):
                pkg.CakeRMSNormSiLUGateNondetFunction.apply(x, z, w, EPS, value)
    assert launches == []
    torch.cuda.synchronize()
    assert not bool(ws.any())


def _forward_rows() -> dict[str, tuple[int, bool]]:
    """One row count of each forward module this device routes, the first row
    of the module's first ``FORWARD_BANDS`` band, else (a module only other
    layouts run) of its first ``STRIDED_FORWARD_BANDS`` band, at least 2 rows
    so that a projection layout's rows are not packed: ``{module: (rows,
    packed)}``."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sm_count = loader.device_sm_count(device_index)
    found: dict[str, tuple[int, bool]] = {}
    for packed, table in (
        (True, loader.FORWARD_BANDS),
        (False, loader.STRIDED_FORWARD_BANDS),
    ):
        start = 1 if packed else 2
        for last, _role in table[arch]:
            found.setdefault(_forward_module(start, packed), (start, packed))
            if last is not None:
                start = last * sm_count + 1
    return found


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_no_entry_point_alerts_under_deterministic_algorithms(warn_only, launches):
    """Every launch but the weight gradient of the fused or a row-claims
    backward repeats bit for bit, so under
    ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with every warning an error, those calls run without an
    alert and return the bits they return with the setting off, launching the
    same modules: the forward on every forward module this device routes; at
    the fewest rows of every ``BACKWARD_BANDS`` band of this device and of
    every long role's packed rows (:func:`_range_shape`) an FP32 weight
    gradient (accumulation included),
    ``compute_grad_weight=False`` (the program without the weight gradient
    of the BF16 bands' module, the fused backward's included, or the build
    of packed rows only on its packed rows; where that build runs and the
    band's BF16 weight gradient may vary, also on ``dy`` and ``z`` as the
    gate columns of wider projections, rows that are not packed, which run
    the band's own program without the weight gradient) and the
    autograd backward of a weight that needs no gradient, and where the band
    repeats a BF16 weight gradient (:func:`_dw_varies`) the BF16 weight
    gradient (caller workspace and none)
    and the autograd backward with it, also of a node built before the
    setting was enabled, and non-reentrant checkpointing; no rows; the
    workspace entry points, the reset and the support query.  Every band's
    module of this device, or its program without the weight gradient, is
    among the launches."""
    forward_rows = _forward_rows()
    assert len(forward_rows) >= 3, forward_rows
    ranges = [
        (band, first, last)
        for role in ("bwd_bf16", "bwd_f32")
        for band, first, last in _device_bands(role) + _device_long_bands(role)
    ]
    bands = [band for band, _, _ in ranges]
    shapes = list(dict.fromkeys(_range_shape(first, last) for _, first, last in ranges))
    cases = [make(T, H, seed=69 + i) for i, (T, H) in enumerate(shapes)]
    repeats = [not _dw_varies(T * H) for T, H in shapes]
    assert any(repeats), shapes
    # where the build of packed rows only runs these packed rows and no BF16
    # call launches the band's module (its weight gradient may vary), the
    # same rows as gate columns (not packed) launch the band's own program
    # without the weight gradient (several tokens: one token of head stride
    # 128 is passed as packed rows)
    gates = []
    for (T, H), bf16 in zip(shapes, repeats, strict=True):
        nodw = _nodw_module(_backward_module(T * H), T * H, packed=_nodw_packed(T, H))
        gates.append(not bf16 and _is_small(nodw))
    assert all(T > 1 for (T, _), gate in zip(shapes, gates, strict=True) if gate)
    empty = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    inputs = [
        make(rows, 1, layout="contiguous" if packed else "projection", seed=72)
        for rows, packed in forward_rows.values()
    ]
    ws = workspace_for(1, empty.device)

    def leaves(x, z, w, weight_grad=True):
        return [
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
        ]

    def run(nodes):
        out = [forward(x, z, w, EPS) for x, z, w, _ in inputs]
        for (x, z, w, dy), bf16, gate in zip(cases, repeats, gates, strict=True):
            if bf16:
                out += backward(dy, x, z, w, EPS)
            out += backward(dy, x, z, w, EPS, grad_weight=torch.empty(D, device="cuda"))
            main = torch.full((D,), 0.25, dtype=torch.float32, device="cuda")
            out += backward(
                dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True
            )
            out += backward(dy, x, z, w, EPS, compute_grad_weight=False)[:2]
            if gate:
                out += backward(
                    _gate_columns(dy),
                    x,
                    _gate_columns(z),
                    w,
                    EPS,
                    compute_grad_weight=False,
                )[:2]
            if bf16:
                out += backward(dy, x, z, w, EPS, workspace=ws)
        out += backward(empty, empty, empty, cases[0][2], EPS)
        nbytes = WORKSPACE_BYTES[loader.device_arch(empty.device.index)]
        assert workspace_bytes(1, empty.device) == nbytes
        assert workspace_for(1, empty.device).numel() == nbytes
        reset_workspace(ws)
        assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(empty.device)
        for (x, z, w, dy), node in zip(cases, nodes, strict=True):
            if node is not None:
                y, grads = node
                y.backward(dy)
                out += [y, *(t.grad for t in grads)]
                ts = leaves(x, z, w)
                yc = torch.utils.checkpoint.checkpoint(
                    gated_rmsnorm, *ts, EPS, use_reentrant=False
                )
                yc.backward(dy)
                out += [yc, *(t.grad for t in ts)]
            ts = leaves(x, z, w, weight_grad=False)
            gated_rmsnorm(*ts, EPS).backward(dy)
            out += [ts[0].grad, ts[1].grad]
            assert ts[2].grad is None
        return out

    def nodes():
        built = []
        for (x, z, w, _), bf16 in zip(cases, repeats, strict=True):
            grads = leaves(x, z, w) if bf16 else None
            built.append((gated_rmsnorm(*grads, EPS), grads) if bf16 else None)
        return built

    want = run(nodes())
    torch.cuda.synchronize()
    want_launches = list(launches)
    launches.clear()
    built = nodes()  # with the setting off
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = run(built)
        torch.cuda.synchronize()
    assert len(got) == len(want)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert torch.equal(a, b), i
    assert launches == want_launches, launches
    modules = set(launches)
    assert set(forward_rows) <= modules, forward_rows
    for (T, H), gate in zip(shapes, gates, strict=True):
        nodw = _nodw_module(_backward_module(T * H), T * H, packed=_nodw_packed(T, H))
        assert {nodw, _backward_module(T * H, torch.float32)} <= modules, (T, H)
        if gate:
            own = _nodw_module(
                _backward_module(T * H, packed=False), T * H, packed=False
            )
            assert own in modules and not _is_small(own), (T, H, own)
    roles = {loader.MODULES[name]["role"] for name in modules}
    roles = {NODW_TWINS.get(role, role) for role in roles}
    assert set(bands) <= roles, (bands, roles)
    assert not bool(ws.any())


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_graph_capture_under_deterministic_algorithms(warn_only, record_property):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with every warning an error, a CUDA Graph captures the
    forward and the backward (caller workspace and none, BF16 and FP32 weight
    gradients) without an alert at the fewest rows of the short backward's
    band (:func:`_band_shape`), whose BF16 weight gradient repeats its bits;
    its replays return the bits of eager calls, the workspace zero after each.
    The eager calls, on the inputs of every draw of ``judge`` (copied into the
    captured inputs), are judged against the naive implementation first."""
    shape = _band_shape("bwd_bf16_short")
    assert shape is not None
    T, H = shape
    assert not _dw_varies(T * H), (T, H)
    x, z, w, dy = make(T, H, seed=75)
    ws = workspace_for(T * H, x.device)
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.empty(D, dtype=torch.float32, device="cuda")

    def step():
        forward(x, z, w, EPS, out=y)
        backward(dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws)
        backward(dy, x, z, w, EPS, grad_weight=main)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    names = ("y", "dx", "dz", "dw", "FP32 dw")

    def run(j, s):
        for captured, new in zip((x, z, w, dy), make(T, H, seed=s), strict=True):
            captured.copy_(new)
        step()
        torch.cuda.synchronize()
        assert not bool(ws.any()), j
        return silu_draw(
            x.clone(),
            z.clone(),
            w.clone(),
            dy.clone(),
            EPS,
            [{k: t.clone() for k, t in zip(names, (y, dx, dz, dw, main), strict=True)}],
            {"y": "y", "dx": "dx", "dz": "dz", "dw": "dw", "FP32 dw": "dw"},
        )

    judge(record_property, run, seed=75)
    for captured, new in zip((x, z, w, dy), make(T, H, seed=75), strict=True):
        captured.copy_(new)
    step()
    torch.cuda.synchronize()
    want = [t.clone() for t in (y, dx, dz, dw, main)]
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        for k in range(3):
            for t in (y, dx, dz, dw, main):
                t.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            for i, (a, b) in enumerate(zip((y, dx, dz, dw, main), want, strict=True)):
                assert torch.equal(a, b), (k, i)
            assert not bool(ws.any()), k


# The bands whose backward of a weight gradient alerts under PyTorch's
# deterministic algorithms (the fused and the row-claims backwards; the
# row-claims backwards where a band runs them), and the rows of each band a
# test runs: its first, its last (a band with an end) and the first rows past
# one wave of claims (more 16-row blocks than 4 CTAs per SM, a band without
# an end).
ALERT_BAND_CASES = [
    pytest.param(band, edge, id=f"{band}-{edge}")
    for band in ("bwd_bf16_fused", "bwd_bf16_rc", "bwd_bf16_rc_mid")
    for edge in ("first", "last", "claims")
]


@pytest.mark.parametrize("band,edge", ALERT_BAND_CASES)
def test_backward_alerts_under_deterministic_algorithms(band, edge, launches):
    """Under ``torch.use_deterministic_algorithms(True)`` the paths that launch
    the order-varying weight-gradient sum of the fused or a row-claims
    backward (the first rows of the band ``band`` on this device and its last
    or, in a band without an end, the first rows past one wave of claims, as
    tokens of one head; skipped where this device routes no such band or the
    band has no such rows) -- the backward
    with ``compute_grad_weight=True`` (also accumulating) and the autograd
    backward when ``weight`` needs a gradient, also for a node built before
    the mode was enabled -- raise PyTorch's alert immediately before the
    launch, which never runs; under ``warn_only=True`` they warn and run the
    non-deterministic backward (never a fallback): ``dx`` and ``dz`` the bits
    of the call with the mode off, ``dw`` a finite BF16 weight gradient (the
    accuracy of these bands' weight gradient is judged by the band tests).  The
    forward, the workspace entry points, the reset, the support query, the
    backward without the weight gradient (the band's program built without
    it, ``nodw_module``) and with an FP32 one, and the autograd backward of a weight that needs no
    gradient run without a warning."""
    found = _device_band(band)
    if found is None:
        pytest.skip(f"this device routes no {band} band")
    first, last = found
    if edge == "first":
        rows = first
    elif edge == "last":
        if last is None:
            pytest.skip(f"the {band} band has no end")
        rows = last
    else:
        rows = 64 * loader.device_sm_count(torch.cuda.current_device()) + 1
        if rows < first or (last is not None and rows > last):
            pytest.skip(f"the {band} band holds no rows past one wave of claims")
    name = _backward_module(rows)
    assert name == _routes()[band] and _varies(name), (rows, name)
    x, z, w, dy = make(rows, 1, seed=65)
    dx_d, dz_d, dw_d = backward(dy, x, z, w, EPS)
    # the bits of compute_grad_weight=False (the repeatable backward's where
    # the build of packed rows only runs these packed rows)
    dx_n, dz_n = _nodw_bits(_nodw_module(name, rows), dy, x, z, w)
    ws = workspace_for(1, x.device)

    def leaves(weight_grad=True):
        return (
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(weight_grad),
        )

    node = gated_rmsnorm(*leaves(), EPS)  # built with the mode off
    torch.use_deterministic_algorithms(True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forward(x, z, w, EPS)
        assert workspace_bytes(1, x.device) == workspace_for(1, x.device).numel()
        reset_workspace(ws)
        assert pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(x.device)
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
        assert dw is None
        assert torch.equal(dx, dx_n) and torch.equal(dz, dz_n)
        f32 = torch.empty(D, dtype=torch.float32, device="cuda")
        backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
        xa, za, wa = leaves(weight_grad=False)
        gated_rmsnorm(xa, za, wa, EPS).backward(dy)
        assert xa.grad is not None and wa.grad is None
    launches.clear()
    with pytest.raises(RuntimeError, match=ALERT):
        backward(dy, x, z, w, EPS)
    main = torch.full((D,), 0.25, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match=ALERT):
        backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=main,
            accumulate_grad_weight=True,
            workspace=ws,
        )
    with pytest.raises(RuntimeError, match=ALERT):
        node.backward(dy)
    assert launches == []
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 0.25))
    assert not bool(ws.any())
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning, match=ALERT):
        dx, dz, dw = backward(dy, x, z, w, EPS)
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d)
    assert_dw_repeats(dw, dw_d, varies=True, what="warn_only dw")
    xa, za, wa = leaves()
    y = gated_rmsnorm(xa, za, wa, EPS)
    with pytest.warns(UserWarning, match=ALERT):
        y.backward(dy)
    assert torch.equal(xa.grad, dx_d) and torch.equal(za.grad, dz_d)
    assert_dw_repeats(wa.grad, dw_d, varies=True, what="warn_only autograd dw")
    fwd = _forward_module(rows)
    assert launches == [name, fwd, name], launches


def _alert(entry: str, deterministic_entry: str, *, warn_only: bool) -> str:
    """The exact text of the alert of ``entry`` of this package."""
    q = f"{pkg.__name__}.{entry}"
    d = f"{DET}.{deterministic_entry}"
    if warn_only:
        return (
            f"{q} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; the "
            f"non-deterministic kernels run. {d} is the deterministic one."
        )
    return (
        f"{q} does not have a deterministic implementation, but you set "
        f"'torch.use_deterministic_algorithms(True)'. You can call {d} instead, "
        "or use the 'warn_only=True' option, if that's acceptable for your "
        "application."
    )


def test_alert_texts_name_the_entry_point_and_the_deterministic_one(launches):
    """At the fewest rows of the fused band (:func:`_band_shape`), the
    alert of the functional backward names it and the deterministic
    backward; the autograd backward's names the autograd backward and the
    deterministic autograd function (strict: ``RuntimeError`` before the
    launch; ``warn_only``: one ``UserWarning`` a call, then the launch)."""
    shape = _band_shape("bwd_bf16_fused")
    assert shape is not None
    T, H = shape
    assert _dw_varies(T * H), (T, H)
    x, z, w, dy = make(T, H, seed=66)

    def node():
        torch.use_deterministic_algorithms(False)
        return gated_rmsnorm(
            x.clone().requires_grad_(True),
            z.clone().requires_grad_(True),
            w.clone().requires_grad_(True),
            EPS,
        )

    functional = (
        "cake_rmsnorm_silu_gate_nondet_train_backward",
        "cake_rmsnorm_silu_gate_det_train_backward",
    )
    autograd = (
        "CakeRMSNormSiLUGateNondetFunction.backward",
        "CakeRMSNormSiLUGateDetFunction",
    )
    torch.use_deterministic_algorithms(True)
    launches.clear()
    with pytest.raises(RuntimeError) as info:
        backward(dy, x, z, w, EPS)
    assert str(info.value) == _alert(*functional, warn_only=False)
    y = node()
    torch.use_deterministic_algorithms(True)
    launches.clear()
    with pytest.raises(RuntimeError) as info:
        y.backward(dy)
    assert _alert(*autograd, warn_only=False) in str(info.value), str(info.value)
    assert launches == []
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        backward(dy, x, z, w, EPS)
    got = [str(r.message) for r in record if ALERT in str(r.message)]
    assert got == [_alert(*functional, warn_only=True)], got
    y = node()
    torch.use_deterministic_algorithms(True, warn_only=True)
    with pytest.warns(UserWarning) as record:
        y.backward(dy)
    got = [str(r.message) for r in record if ALERT in str(r.message)]
    assert got == [_alert(*autograd, warn_only=True)], got
    bwd = _routes()["bwd_bf16_fused"]
    assert launches == [bwd, _forward_module(T * H), bwd], launches


def test_graph_capture_alerts_at_capture_and_replays_without_a_check(
    record_property,
):
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)``
    capturing a backward that computes the weight gradient at the fewest rows
    of the fused band (:func:`_band_shape`) warns once (the check runs on
    the host, at capture); the replays run the non-deterministic kernel
    without a warning, three on the inputs of each draw of ``judge`` (the
    first draw's the captured ones, the others' copied into them): ``dx`` and
    ``dz`` the bits of the eager call (the eager call without the weight
    gradient on ``dy`` and ``z`` as the gate columns of wider projections,
    which runs the band's program without the weight gradient on rows that
    are not packed, the same bits, and alerts not; on packed rows the build
    of packed rows only would run), the workspace zero
    after each, every replay's ``dx``, ``dz`` and ``dw`` judged against the
    naive implementation."""
    shape = _band_shape("bwd_bf16_fused")
    assert shape is not None
    T, H = shape
    assert _dw_varies(T * H), (T, H)
    x, z, w, dy = make(T, H, seed=72)
    ws = workspace_for(T * H, x.device)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")

    def step():
        backward(dy, x, z, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the module before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dx_e, dz_e = dx.clone(), dz.clone()
    torch.use_deterministic_algorithms(True, warn_only=True)
    graph = torch.cuda.CUDAGraph()
    with pytest.warns(UserWarning) as record, torch.cuda.graph(graph):
        step()
    got = [str(r.message) for r in record if ALERT in str(r.message)]
    assert got == [
        _alert(
            "cake_rmsnorm_silu_gate_nondet_train_backward",
            "cake_rmsnorm_silu_gate_det_train_backward",
            warn_only=True,
        )
    ], got

    def run(j, s):
        if j:
            for captured, new in zip((x, z, w, dy), make(T, H, seed=s), strict=True):
                captured.copy_(new)
        want = backward(
            _gate_columns(dy), x, _gate_columns(z), w, EPS, compute_grad_weight=False
        )[:2]
        if not j:
            assert torch.equal(want[0], dx_e) and torch.equal(want[1], dz_e)
        launched = []
        for k in range(3):
            dw.fill_(7.0)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(dx, want[0]) and torch.equal(dz, want[1]), (j, k)
            assert not bool(ws.any()), (j, k)
            launched.append({"dx": dx.clone(), "dz": dz.clone(), "dw": dw.clone()})
        return silu_draw(
            x.clone(),
            z.clone(),
            w.clone(),
            dy.clone(),
            EPS,
            launched,
            {"dx": "dx", "dz": "dz", "dw": "dw"},
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        judge(record_property, run, seed=72)


def test_deterministic_workspace_is_refused_naming_both_modes():
    """The backward (past the short backward and on it, which checks a caller
    workspace too) and the reset refuse an FP32 tensor, the deterministic
    package's workspace, naming both modes' workspaces, and leave it as it is;
    any other unfit buffer gets the byte-workspace requirement."""
    device = torch.device("cuda")
    nbytes = workspace_for(1, device).numel()
    fp32 = torch.zeros(8192, dtype=torch.float32, device="cuda")
    want = (
        "workspace is FP32, a workspace of the deterministic mode "
        f"({DET}.cake_rmsnorm_silu_gate_det_train_backward_workspace); the "
        "non-deterministic backward needs its own zero-filled uint8 workspace "
        f"({pkg.__name__}.cake_rmsnorm_silu_gate_nondet_train_backward_workspace)"
    )
    for T in (65, 5):
        x, z, w, dy = make(T, 48, seed=73)
        for dtype in (torch.bfloat16, torch.float32):
            with pytest.raises(ValueError) as info:
                backward(
                    dy,
                    x,
                    z,
                    w,
                    EPS,
                    grad_weight=torch.empty(D, dtype=dtype, device="cuda"),
                    workspace=fp32,
                )
            assert str(info.value) == want, (T, dtype)
    with pytest.raises(ValueError) as info:
        reset_workspace(fp32)
    assert str(info.value) == want
    torch.cuda.synchronize()
    assert not bool(fp32.any())
    x, z, w, dy = make(65, 48, seed=74)
    device = x.device
    for bad in _unfit_workspaces(nbytes):
        with pytest.raises(ValueError) as info:
            backward(dy, x, z, w, EPS, workspace=bad)
        assert str(info.value) == (
            f"workspace must be a contiguous torch.uint8 tensor on {device} with at "
            f"least {nbytes} bytes starting at a 16-byte aligned address, zero "
            "before its first use "
            "(cake_rmsnorm_silu_gate_nondet_train_backward_workspace(rows, device), "
            "or cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace))"
        )


# The libraries (a module record built for one architecture, ``build.json``'s
# ``<library>_<arch>``) that no band of their architecture selects: no
# ``FORWARD_BANDS``, ``STRIDED_FORWARD_BANDS`` or ``BACKWARD_BANDS`` band names
# their role, they are no long role of ``LONG_BACKWARD_ROWS_PER_SM`` and no
# role whose bands end without a ``None`` band falls back to them, so no call of
# the public API runs them and the band tests skip them.  ``(module role,
# architecture)``: the long forward on sm_103a and sm_107a (routed on sm_100a
# only), the static backward of a BF16 weight gradient on sm_107a, the
# row-claims backwards on sm_100a and sm_103a and the claimed backward of a
# BF16 weight gradient on every architecture.
UNBANDED_LIBRARIES = {
    ("fwd_long", "sm_103a"),
    ("fwd_long", "sm_107a"),
    ("bwd_bf16", "sm_107a"),
    ("bwd_bf16_rc", "sm_100a"),
    ("bwd_bf16_rc", "sm_103a"),
    ("bwd_bf16_rc_mid", "sm_100a"),
    ("bwd_bf16_rc_mid", "sm_103a"),
    ("bwd_bf16_mid", "sm_100a"),
    ("bwd_bf16_mid", "sm_103a"),
    ("bwd_bf16_mid", "sm_107a"),
}
# The rows per SM of a mid and of a large call of each module role of
# ``UNBANDED_LIBRARIES``, inside the module's intended range: the long
# forward's sm_100a band (past 160 rows per SM up to 400); the sm_107a static
# backward up to 64 rows per SM (one row block a CTA of its grid of 8 CTAs per
# SM); the row-claims backwards past one wave of their CTAs (64 and 80 rows per
# SM) up to the end of the band of a BF16 weight gradient that may vary on
# sm_100a and sm_103a (1000 rows per SM); the claimed backward past every
# static backward band (at most 128 rows per SM: 2 row blocks a CTA of the
# static backward's grid of 8 CTAs per SM), with no end.  A forced
# route runs every call of the role's kind on the module, so these cases run
# each library whatever the bands; each case runs on every architecture the
# module is built for (also one where a band selects it).
FORCED_ROWS_PER_SM = {
    "fwd_long": {"mid": 280, "large": 400},
    "bwd_bf16": {"mid": 32, "large": 64},
    "bwd_bf16_rc": {"mid": 128, "large": 1000},
    "bwd_bf16_rc_mid": {"mid": 128, "large": 1000},
    "bwd_bf16_mid": {"mid": 512, "large": 4096},
}
FORCED_SIZES = ("mid", "large")
FORCED_BACKWARD_ROLES = ("bwd_bf16", "bwd_bf16_rc", "bwd_bf16_rc_mid", "bwd_bf16_mid")


def _unbanded_libraries() -> set[tuple[str, str]]:
    """``(module role, architecture)`` of every module record built for an
    architecture whose bands never select it: not the route of a role of its
    ``FORWARD_BANDS``, ``STRIDED_FORWARD_BANDS`` or ``BACKWARD_BANDS`` bands,
    of a long role of ``LONG_BACKWARD_ROWS_PER_SM``, of a backward role
    whose bands end without a ``None`` band (:func:`module_for_rows` falls
    back to its own module), or of a program without the weight gradient
    (``NODW_ROLES``, which :func:`nodw_module` names in its band's place)."""
    found = set()
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        roles = {
            band
            for table in (loader.FORWARD_BANDS, loader.STRIDED_FORWARD_BANDS)
            for _, band in table[arch]
        }
        for role, bands in loader.BACKWARD_BANDS[arch].items():
            roles |= {band for _, band in bands}
            if all(last is not None for last, _ in bands):
                roles.add(role)
        roles |= {
            band + loader.LONG
            for band in loader.LONG_BACKWARD_ROWS_PER_SM.get(arch, {})
        }
        roles |= set(loader.NODW_ROLES) & set(routes)
        selected = {routes[role] for role in roles}
        for name, record in loader.MODULES.items():
            if arch in record["arches"] and name not in selected:
                found.add((record["role"], arch))
    return found


def _forced_module(role: str) -> str:
    """The module record of ``role`` built for this device's architecture;
    the test skips where none is."""
    arch = loader.device_arch(torch.cuda.current_device())
    names = [
        name
        for name, record in loader.MODULES.items()
        if record["role"] == role and arch in record["arches"]
    ]
    if not names:
        built = sorted(
            {
                a
                for record in loader.MODULES.values()
                if record["role"] == role
                for a in record["arches"]
            }
        )
        pytest.skip(
            f"the {role} module has no build for {arch} (built for {', '.join(built)})"
        )
    assert len(names) == 1, (role, arch, names)
    return names[0]


def _force_route(monkeypatch: pytest.MonkeyPatch, role: str, name: str) -> None:
    """Route every call of ``role``'s kind on this device's architecture to
    module ``name``: ``ROUTES[arch][role]`` names it, and the forward's
    ``FORWARD_BANDS`` and ``STRIDED_FORWARD_BANDS`` (a forward role) or the
    BF16 weight gradient's ``BACKWARD_BANDS`` (a backward role; also the
    bands of ``compute_grad_weight=False``) are the one band ``(None,
    role)``.  Copies of the tables, restored by ``monkeypatch``."""
    arch = loader.device_arch(torch.cuda.current_device())
    routes = {a: dict(by_role) for a, by_role in loader.ROUTES.items()}
    assert routes[arch].get(role, name) == name, (arch, role, routes[arch][role])
    routes[arch][role] = name
    monkeypatch.setattr(loader, "ROUTES", routes)
    # the plans nodw_module reads (read once at import), read again
    plans = {a: {n: loader._nodw_plan(a, n) for n in loader.MODULES} for a in routes}
    monkeypatch.setattr(loader, "_NODW_PLANS", plans)
    if role.startswith("fwd"):
        for table in ("FORWARD_BANDS", "STRIDED_FORWARD_BANDS"):
            bands = dict(getattr(loader, table))
            bands[arch] = ((None, role),)
            monkeypatch.setattr(loader, table, bands)
    else:
        bands = {a: dict(by_role) for a, by_role in loader.BACKWARD_BANDS.items()}
        bands[arch]["bwd_bf16"] = ((None, role),)
        monkeypatch.setattr(loader, "BACKWARD_BANDS", bands)


def _forced_shape(role: str, size: str) -> tuple[int, int]:
    """``(T, 48)``: the most tokens of 48 heads (at least 2) within
    ``FORCED_ROWS_PER_SM[role][size]`` rows per SM on this device."""
    sm_count = loader.device_sm_count(torch.cuda.current_device())
    return max(2, FORCED_ROWS_PER_SM[role][size] * sm_count // 48), 48


def _gate_columns(t: torch.Tensor) -> torch.Tensor:
    """``t`` (``[T, H, 128]``) as the gate columns of a wider projection, read
    in place: the same values with token stride ``2 * H * 128 + 16``."""
    T, H, _ = t.shape
    wide = torch.empty(T, 2 * H * D + 16, dtype=t.dtype, device=t.device)
    view = wide[:, H * D : 2 * H * D].view(T, H, D)
    view.copy_(t)
    return view


def test_forced_routes_cover_every_library_no_band_selects():
    """The libraries no band selects are ``UNBANDED_LIBRARIES``, each one a
    ``build.json`` library, and the forced-route tests cover each one's role
    (``FORCED_ROWS_PER_SM``); every unbanded backward writes a BF16 weight
    gradient (a ``bwd_bf16`` role), so an FP32 weight gradient, which takes
    the ``bwd_f32`` bands, never runs it."""
    found = _unbanded_libraries()
    assert found == UNBANDED_LIBRARIES, sorted(found ^ UNBANDED_LIBRARIES)
    assert {role for role, _ in found} == set(FORCED_ROWS_PER_SM)
    assert set(FORCED_BACKWARD_ROLES) == set(FORCED_ROWS_PER_SM) - {"fwd_long"}
    libraries = cake_build.libraries_by_name()
    for role, arch in sorted(found):
        names = [
            name
            for name, record in loader.MODULES.items()
            if record["role"] == role and arch in record["arches"]
        ]
        assert len(names) == 1, (role, arch, names)
        assert loader.library_name(names[0], arch) in libraries, (role, arch)
        assert role == "fwd_long" or role.startswith("bwd_bf16"), role


@pytest.mark.parametrize("size", FORCED_SIZES)
def test_long_forward_runs_through_a_forced_route(
    size, launches, monkeypatch, record_property
):
    """The long forward (``fwd_long``, which no band selects on sm_103a and
    sm_107a) on a forced route (:func:`_force_route`; skipped where it has no
    build) for ``FORCED_ROWS_PER_SM`` rows: ``y`` of packed rows and of ``z``
    as the gate columns of a wider projection (other layouts' bands) the
    same bits, the same bits on every launch (the first draw's ``REPEATS``),
    and judged against the naive implementation over the draws of
    ``judge``."""
    role = "fwd_long"
    name = _forced_module(role)
    T, H = _forced_shape(role, size)
    rows = T * H
    _force_route(monkeypatch, role, name)
    assert _forward_module(rows) == name
    assert _forward_module(rows, packed=False) == name

    def run(j, s):
        x, z, w, _ = make(T, H, seed=s)
        zp = _gate_columns(z)
        first = None
        for i in range(REPEATS if j == 0 else 1):
            y = forward(x, z, w, EPS)
            assert y.dtype == torch.bfloat16 and y.shape == x.shape
            assert torch.equal(forward(x, zp, w, EPS), y), f"projection y, launch {i}"
            if first is None:
                first = y
            else:
                assert torch.equal(y, first), f"y: launch {i} vs the first launch"
        return silu_draw(x, z, w, None, EPS, [{"y": first}], {"y": "y"})

    judge(record_property, run, seed=101)
    assert launches == [name] * (2 * _launch_count()), launches


@pytest.mark.parametrize("size", FORCED_SIZES)
@pytest.mark.parametrize("role", FORCED_BACKWARD_ROLES)
def test_backward_libraries_no_band_selects_run_through_a_forced_route(
    role, size, launches, monkeypatch, record_property
):
    """A backward module that no band selects (``UNBANDED_LIBRARIES``) on a
    forced route (:func:`_force_route`; skipped where it has no build) for
    ``FORCED_ROWS_PER_SM`` rows.  Each is a build of a BF16 weight gradient
    (an FP32 one takes the ``bwd_f32`` bands, another program): ``dx``,
    ``dz``, the BF16 weight gradient and the weight gradient accumulated onto
    0.25 judged against the naive implementation over the draws of
    ``judge``, ``dx`` and ``dz`` (and the weight gradient where the module
    repeats it, :func:`_varies`) the same bits on every launch
    (``check_all``), the stream's workspace zero after them.  On a caller
    workspace, over the same draws of ``judge`` (two launches on the first
    draw, one on the others) and on a launch with ``dy`` and ``z`` as the
    gate columns of a wider projection: the BF16 weight gradient, the one
    accumulated onto 0.25 and the projection's judged against the naive
    implementation (every launch's: the judge keeps the larger error),
    ``dx`` and ``dz`` the bits of the draw's first launch (on the first draw
    ``check_all``'s judged ones), and so is the weight gradient, alone and
    accumulated, where the module repeats it (:func:`assert_dw_repeats`).
    ``compute_grad_weight=False`` returns ``None``, leaves ``grad_weight`` as
    it is and gives the same ``dx`` and ``dz``; the workspace is zero after
    every call."""
    name = _forced_module(role)
    T, H = _forced_shape(role, size)
    rows = T * H
    _force_route(monkeypatch, role, name)
    assert _backward_module(rows) == name
    assert _backward_module(rows, packed=False) == name
    assert _backward_module(rows, torch.float32) != name
    varies = _varies(name)
    what = f"{role} ({name}) at {rows} rows"
    (x, z, w, dy), (_, dx, dz, dw) = check_all(
        record_property, lambda s: make(T, H, seed=s), seed=100
    )
    fwd = _forward_module(rows)
    assert launches == [fwd, name, name] * _launch_count(), launches
    key = (
        torch.cuda.current_device(),
        torch.cuda.current_stream().cuda_stream,
        name,
    )
    views = pkg._STREAM_WORKSPACES[key]
    # (through bytes: uint64 tensors take few operations)
    assert not any(bool(view.view(torch.uint8).any()) for view in views.values()), what
    launches.clear()
    ws = workspace_for(rows, x.device)

    def bf16(fill=None):
        if fill is None:
            return torch.empty(D, dtype=torch.bfloat16, device=x.device)
        return torch.full((D,), fill, dtype=torch.bfloat16, device=x.device)

    # The weight gradients of the caller-workspace and projection launches,
    # judged like check_all's: a row-claims module's may vary from launch to
    # launch, and a repeated wrong sum has the same bits every time, so bits
    # alone do not check them.
    kinds = {
        "caller-workspace dw": "dw",
        "caller-workspace accumulated dw": "dw",
        "projection dw": "dw",
    }
    starts = {
        "caller-workspace accumulated dw": torch.full((D,), 0.25, dtype=torch.bfloat16)
    }

    def run(j, s):
        x, z, w, dy = make(T, H, seed=s)
        # Draw 0's inputs are check_all's first draw's (the same seed).
        want = (dx, dz, dw) if j == 0 else None
        accumulated = None
        launched = []
        for i in range(2 if j == 0 else 1):
            at = f"{what}: draw {j}, launch {i}"
            gx, gz, gw = backward(dy, x, z, w, EPS, grad_weight=bf16(), workspace=ws)
            assert not bool(ws.any()), at
            if want is None:
                want = (gx, gz, gw)
            assert torch.equal(gx, want[0]) and torch.equal(gz, want[1]), at
            assert_dw_repeats(gw, want[2], varies=varies, what=f"dw {at}")
            main = bf16(0.25)
            backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=main,
                accumulate_grad_weight=True,
                workspace=ws,
            )
            assert not bool(ws.any()), f"accumulating {at}"
            if accumulated is None:
                accumulated = main
            else:
                assert_dw_repeats(
                    main, accumulated, varies=varies, what=f"accumulated dw {at}"
                )
            launched.append(
                {"caller-workspace dw": gw, "caller-workspace accumulated dw": main}
            )
        at = f"{what}: draw {j}, projection"
        dyp, zp = _gate_columns(dy), _gate_columns(z)
        gx, gz, gw = backward(dyp, x, zp, w, EPS, grad_weight=bf16(), workspace=ws)
        assert not bool(ws.any()), at
        assert torch.equal(gx, want[0]) and torch.equal(gz, want[1]), at
        assert_dw_repeats(gw, want[2], varies=varies, what=f"dw {at}")
        launched.append({"projection dw": gw})
        return silu_draw(x, z, w, dy, EPS, launched, kinds, starts=starts)

    judge(record_property, run, seed=100)
    main = bf16(3.0)
    gx, gz, gw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert gw is None, what
    assert torch.equal(gx, dx) and torch.equal(gz, dz), what
    assert torch.equal(main, torch.full_like(main, 3.0)), what
    assert not bool(ws.any()), f"{what}: compute_grad_weight=False"
    # Per caller-workspace launch a backward and an accumulating one, a
    # projection backward a draw, and the compute_grad_weight=False call.
    assert launches == [name] * (2 * _launch_count(2) + DRAWS + 1), launches


# Without a device: SM counts of small MIG-like instances and the products'.
NODW_SM_COUNTS = (1, 2, 7, 14, 16, 24, 33, 66, 148, 152, 160, 212, 216)


@pytest.mark.parametrize("sm_count", NODW_SM_COUNTS)
def test_nodw_module_and_the_strided_ctas(sm_count):
    """``nodw_module`` and the strided programs' CTAs (host only), the
    programs without the weight gradient of a band unrouted in turn
    (``ROUTES`` patched, the plans read again, :func:`_patch_routes`): for
    packed rows (``packed``; omitted, not packed) up to
    ``PACKED_SMALL_ROWS_PER_SM * SMs`` the build of packed rows only where
    routed and the band routes a program without the weight gradient (a
    strided or a claimed one, a small-row band its one-cluster one); else for
    the module of a band with a strided program (``NODW_CUT_BANDS``) the
    strided program up to ``NODW_STRIDED_ROWS_PER_SM * SMs`` rows where
    routed (at every row count where the band has no cut), else the claimed
    program where routed, else the strided one, else the module itself; for
    a small-row band's module its one-cluster program where routed, else the
    module; for the long role's module the long program where routed, else
    the long role's module (run with ``weight_grad`` 0), packed or not; for
    any other module the module itself, packed or not; for packed rows past
    the build of packed rows only the band's strided program of packed rows
    in the strided program's place where both are routed.  A strided program
    launches ``min(ceil(rows / rows_per_block), ctas_per_sm * SMs)`` CTAs of
    its ``NODW_STRIDED_GRID_RULES`` rule at every row count: the 256-thread
    ones ``min(ceil(rows / 16), 4 * SMs)``, their twin's count, the
    repeatable band's of packed rows and its build of packed rows only
    ``min(ceil(rows / 8), 8 * SMs)``."""
    wave = 4 * sm_count
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        small_per_sm = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        small_cut = small_per_sm * sm_count
        small_edges = {small_cut - 1, small_cut, small_cut + 1} if small_per_sm else set()
        for band in NODW_CUT_BANDS:
            name = routes[band]
            per_sm = loader.NODW_STRIDED_ROWS_PER_SM[arch].get(band)
            cut = None if per_sm is None else per_sm * sm_count
            strided_role = band + loader.NODW + loader.STRIDED
            packed_role = strided_role + loader.PACKED
            claimed_role = band + loader.NODW
            edges = {1, 7, 8, 9, 15, 16, 17, 16 * wave - 1, 16 * wave, 16 * wave + 1}
            edges |= small_edges
            if cut is not None:
                edges |= {cut - 1, cut, cut + 1}
            rows_list = sorted(r for r in edges | {2**31 - 1} if r >= 1)
            for drop in (
                (),
                (NODW_SMALL,),
                (packed_role,),
                (strided_role,),
                (claimed_role,),
                (strided_role, claimed_role),
                (NODW_SMALL, strided_role, claimed_role),
            ):
                table = {r: n for r, n in routes.items() if r not in drop}
                claimed = table.get(claimed_role)
                small = None
                if table.get(strided_role) or claimed:
                    small = table.get(NODW_SMALL)
                with pytest.MonkeyPatch.context() as patch:
                    _patch_routes(patch, arch, table)
                    for packed in (False, True):
                        strided = table.get(strided_role)
                        if packed and strided is not None:
                            strided = table.get(packed_role, strided)
                        for rows in rows_list:
                            if packed and small is not None and rows <= small_cut:
                                want = small
                            elif strided is not None and cut is not None and rows <= cut:
                                want = strided
                            else:
                                want = claimed or strided or name
                            what = (arch, band, drop, packed, rows)
                            got = loader.nodw_module(
                                arch, name, rows, sm_count, packed=packed
                            )
                            assert got == want, (*what, got)
                            if not packed:
                                got = loader.nodw_module(arch, name, rows, sm_count)
                                assert got == want, (*what, got)
            for role in (strided_role, packed_role):
                strided = routes.get(role)
                if strided is None:
                    continue
                record = loader.MODULES[strided]
                rule = NODW_STRIDED_GRID_RULES[role]
                for rows in rows_list:
                    blocks = -(-rows // rule["rows_per_block"])
                    got = loader.backward_ctas(strided, rows, sm_count)
                    want = min(blocks, rule["ctas_per_sm"] * sm_count)
                    assert got == want, (arch, strided, rows, got)
                    if role == strided_role:
                        twin = loader.backward_ctas(name, rows, sm_count)
                        assert got == twin, (arch, strided, rows, got, twin)
                    grid = loader.launch_grid(
                        record, rows=rows, sm_count=sm_count, ctas=got
                    )
                    assert grid == (want, 1, 1), (arch, strided, rows, grid)
        if NODW_SMALL in routes:
            small = routes[NODW_SMALL]
            record = loader.MODULES[small]
            rule = NODW_STRIDED_GRID_RULES[NODW_SMALL]
            rows_list = sorted(r for r in small_edges | {1, 7, 8, 9, 16, 17} if r >= 1)
            for rows in rows_list:
                blocks = -(-rows // rule["rows_per_block"])
                got = loader.backward_ctas(small, rows, sm_count)
                want = min(blocks, rule["ctas_per_sm"] * sm_count)
                assert got == want, (arch, small, rows, got)
                grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, ctas=got)
                assert grid == (want, 1, 1), (arch, small, rows, grid)
        long_role = "bwd_bf16_rep" + loader.LONG
        nodw_long_role = "bwd_bf16_rep" + loader.NODW + loader.LONG
        long = routes[long_role]
        without = {r: n for r, n in routes.items() if r != nodw_long_role}
        for table, want in (
            (dict(routes), routes.get(nodw_long_role, long)),
            (without, long),
        ):
            with pytest.MonkeyPatch.context() as patch:
                _patch_routes(patch, arch, table)
                for rows in (1, 2560 * sm_count, 2**31 - 1):
                    # the long role runs packed rows only (and so is never
                    # the build of packed rows only)
                    got = loader.nodw_module(arch, long, rows, sm_count, packed=True)
                    assert got == want, (arch, rows, got)
        for band in ("bwd_bf16_short", "bwd_bf16_t512"):
            if band not in routes:
                continue
            name = routes[band]
            own = band + loader.NODW
            without = {r: n for r, n in routes.items() if r != own}
            for table, want in ((dict(routes), routes.get(own, name)), (without, name)):
                small = table.get(NODW_SMALL) if own in table else None
                with pytest.MonkeyPatch.context() as patch:
                    _patch_routes(patch, arch, table)
                    for rows in sorted({1, 256, 512} | small_edges):
                        if rows < 1:
                            continue
                        got = loader.nodw_module(arch, name, rows, sm_count)
                        assert got == want, (arch, band, rows, got)
                        got = loader.nodw_module(arch, name, rows, sm_count, True)
                        packed_want = want
                        if small is not None and rows <= small_cut:
                            packed_want = small
                        assert got == packed_want, (arch, band, rows, got)
        others = ("bwd_bf16", "bwd_bf16_rc", "bwd_bf16_rc_mid", "bwd_bf16_mid")
        others += ("bwd_f32", "bwd_f32_rep", "bwd_f32_rep_long", "bwd_f32_t512")
        for role in others:
            if role in routes:
                for rows in (1, 4096):
                    for packed in (False, True):
                        got = loader.nodw_module(
                            arch, routes[role], rows, sm_count, packed=packed
                        )
                        assert got == routes[role], (arch, role, rows, packed, got)


# The strides (in elements) of ``dy``, ``x`` and ``z`` of 48 heads as
# ``_kernel_view`` passes them that the backward reads as packed rows (head
# stride 128, token stride ``48 * 128``), and of layouts it reads otherwise:
# a gate and ``dy`` sliced from ``[T, 2 * 48 * 128 + 16]`` projections,
# ``dy`` alone sliced so, head-major ``x`` and ``z``, and a token-stride-0
# broadcast of ``x`` and ``z``.
ROUTE_H = 48
PACKED_STRIDES = (ROUTE_H * D, D, ROUTE_H * D, D, ROUTE_H * D, D)
OTHER_STRIDES = {
    "projection": (
        2 * ROUTE_H * D + 16,
        D,
        ROUTE_H * D,
        D,
        2 * ROUTE_H * D + 16,
        D,
    ),
    "dy_projection": (2 * ROUTE_H * D + 16, D, ROUTE_H * D, D, ROUTE_H * D, D),
    "head_major": (ROUTE_H * D, D, D, 65 * D, D, 65 * D),
    "broadcast": (ROUTE_H * D, D, 0, D, 0, D),
}


@pytest.mark.parametrize("sm_count", NODW_SM_COUNTS)
def test_strided_programs_follow_the_layout(sm_count):
    """On every architecture (host only; ``_backward_packed`` and
    ``nodw_module`` as the backward calls them), a call without the weight
    gradient in a band with a strided program (``NODW_CUT_BANDS``, the
    band's module) below the band's strided cut (where the band has none, up
    to a row short of the long role's first rows) runs on packed rows up to
    ``PACKED_SMALL_ROWS_PER_SM * SMs`` the build of packed rows only where
    routed (sm_100a and sm_103a, 128-thread CTAs), past it the band's
    strided program of packed rows where routed (sm_107a routes the
    repeatable band's, 128-thread CTAs, and no other architecture or band
    routes one), and the band's strided program (256-thread CTAs) on each of
    ``OTHER_STRIDES`` at every row count below the cut (which without a cut
    runs them at every row count), never the build of packed rows only; past
    a cut both run the band's claimed program."""
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        packed_roles = {
            role
            for role in routes
            if role.endswith(loader.NODW + loader.STRIDED + loader.PACKED)
        }
        want_roles = set()
        if arch == "sm_107a":
            want_roles = {"bwd_bf16_rep_nodw_strided_packed"}
        assert packed_roles == want_roles, (arch, packed_roles)
        small = routes.get(NODW_SMALL)
        small_cut = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0) * sm_count
        assert (small is not None) == (arch != "sm_107a"), arch
        if small is not None:
            assert loader.MODULES[small]["launch"]["block"] == (128, 1, 1)
        start = LONG_ROWS_PER_SM[arch] * sm_count
        for band in NODW_CUT_BANDS:
            strided_role = band + loader.NODW + loader.STRIDED
            if strided_role not in routes:
                continue
            name = routes[band]
            per_sm = loader.NODW_STRIDED_ROWS_PER_SM[arch].get(band)
            last = start - 1 if per_sm is None else per_sm * sm_count
            below = {1, 7, 8, 9, 16, 17, ROUTE_H, 65 * ROUTE_H, last // 2, last}
            below |= {small_cut - 1, small_cut, small_cut + 1}
            below = {rows for rows in below if 1 <= rows <= last}
            packed_role = strided_role + loader.PACKED
            if packed_role not in routes:
                packed_role = strided_role
            cases = [(PACKED_STRIDES, True, packed_role)]
            cases += [(s, False, strided_role) for s in OTHER_STRIDES.values()]
            for strides, packed, role in cases:
                assert pkg._backward_packed(ROUTE_H, *strides) is packed, strides
                want = routes[role]
                threads = NODW_STRIDED_THREADS[role]
                assert loader.MODULES[want]["launch"]["block"] == (threads, 1, 1)
                rows_list = set(below)
                if not packed and per_sm is None:
                    rows_list |= {start, start + 1, 2**31 - 1}
                for rows in sorted(rows_list):
                    got = loader.nodw_module(arch, name, rows, sm_count, packed=packed)
                    if packed and small is not None and rows <= small_cut:
                        assert got == small, (arch, band, rows, strides, got)
                    else:
                        assert got == want, (arch, band, rows, strides, got)
                if per_sm is not None:
                    rows = per_sm * sm_count + 1
                    got = loader.nodw_module(arch, name, rows, sm_count, packed=packed)
                    claimed = routes[band + loader.NODW]
                    if packed and small is not None and rows <= small_cut:
                        claimed = small
                    assert got == claimed, (arch, band, rows, got)


def test_nodw_plans_read_at_import_are_the_tables():
    """The plans ``nodw_module`` reads (``_NODW_PLANS``, read once at import
    for every routed architecture and module) are the ones ``_nodw_plan``
    reads from ``ROUTES`` and the cut tables now: on sm_100a and sm_103a the
    fused and the repeatable band's carry the build of packed rows only and
    its cut (``PACKED_SMALL_ROWS_PER_SM``), on sm_107a none (cut -1, which no
    row count is at most); an architecture without plans (none routed) gets
    one read on the call, the module itself."""
    assert set(loader._NODW_PLANS) == set(loader.ROUTES)
    for arch, plans in loader._NODW_PLANS.items():
        assert set(plans) == set(loader.MODULES), arch
        for name, plan in plans.items():
            assert plan == loader._nodw_plan(arch, name), (arch, name, plan)
        routes = loader.ROUTES[arch]
        per_sm = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        for band in NODW_CUT_BANDS:
            _, small, small_cut, *_ = plans[routes[band]]
            if per_sm:
                assert (small, small_cut) == (routes[NODW_SMALL], per_sm), arch
            else:
                assert (small, small_cut) == (None, -1), arch
    name = loader.ROUTES["sm_100a"]["bwd_bf16_fused"]
    assert loader.nodw_module("sm_90a", name, 1, 132, packed=True) == name


# SM counts of grids (``device_sm_count`` patched) on which the cut of the
# build of packed rows only is crossed, besides this device's: on one SM its
# cut of 128 rows lies in the small-row band, on 16 and 66 SMs in the fused
# band.
SMALL_CUT_SM_COUNTS = (1, 16, 66)


@pytest.mark.parametrize("edge", [-1, 0, 1], ids=["cut-1", "cut", "cut+1"])
def test_small_program_runs_packed_rows_up_to_its_cut(edge, launches):
    """On this device's grid and on the grids of ``SMALL_CUT_SM_COUNTS`` SMs
    (``device_sm_count`` patched: the cut follows the SMs), a call with
    ``compute_grad_weight=False`` on packed rows a row below, at and a row
    above the cut of the build of packed rows only
    (``PACKED_SMALL_ROWS_PER_SM * SMs`` flat rows) runs that build up to the
    cut and a row past it the program the band of the rows runs without the
    weight gradient (``nodw_module``), and gives up to the cut the ``dx``
    and ``dz`` bits of the repeatable backward (:func:`_rep_bits`), past it
    those of the BF16 call on the band's module; skipped where this device
    routes no such build (sm_107a)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    per_sm = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
    if NODW_SMALL not in loader.ROUTES[arch] or not per_sm:
        pytest.skip(f"{arch} routes no build of packed rows only")
    small = loader.ROUTES[arch][NODW_SMALL]
    for sm_count in (loader.device_sm_count(device_index), *SMALL_CUT_SM_COUNTS):
        rows = per_sm * sm_count + edge
        x, z, w, dy = make(rows, 1, layout="flat", seed=47)
        what = f"{rows} rows on the grids of {sm_count} SMs"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "device_sm_count", lambda index, count=sm_count: count)
            name = loader.module_for_rows(arch, "bwd_bf16", rows, sm_count, packed=True)
            nodw = loader.nodw_module(arch, name, rows, sm_count, packed=True)
            assert (nodw == small) == (edge <= 0), (what, name, nodw)
            if edge <= 0:
                dx_d, dz_d = _rep_bits(dy, x, z, w)
            else:
                dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
            launches.clear()
            dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
            assert launches == [nodw], (what, launches)
        assert dw is None, what
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), what


# ``make`` layouts of one token whose heads lie 128 elements apart in ``dy``,
# ``x`` and ``z``: a misaligned one the host copies to packed rows.
ONE_TOKEN_LAYOUTS = ("contiguous", "projection", "head_major", "misaligned")
STRIDE_KEYS = ("dy_tok", "dy_head", "x_tok", "x_head", "z_tok", "z_head")


@pytest.mark.parametrize("layout", ONE_TOKEN_LAYOUTS)
@pytest.mark.parametrize("H", [1, 48])
def test_one_token_runs_as_packed_rows_without_the_weight_gradient(
    H, layout, monkeypatch
):
    """A call with ``compute_grad_weight=False`` on one token of ``H`` heads
    (``make``'s ``layout``) passes the token as packed rows where the build
    of packed rows only has a cut (sm_100a and sm_103a,
    :func:`_one_token_packed`): token strides ``H * 128`` of ``dy``, ``x``
    and ``z`` and head strides 128 (the same addresses), that build (the
    program ``nodw_module`` names for packed rows) and the repeatable
    backward's ``dx`` and ``dz`` bits; elsewhere (sm_107a) token strides 0
    and the program for rows that are not packed, with the bits of the BF16
    call on the band's module.  The call that computes the weight gradient
    keeps token stride 0 and the band's module."""
    arch = loader.device_arch(torch.cuda.current_device())
    seen = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        seen.append((name, tuple(int(values[k]) for k in STRIDE_KEYS)))
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    x, z, w, dy = make(1, H, layout=layout, seed=48)
    one = _one_token_packed(arch)
    assert _nodw_packed(1, H, layout) == one
    name = _backward_module(H, packed=False)
    nodw = _nodw_module(name, H, packed=one)
    assert _is_small(nodw) == one, (arch, nodw)
    dx_d, dz_d = _nodw_bits(nodw, dy, x, z, w)
    seen.clear()
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    assert dw is None
    tok = H * D if one else 0
    assert seen == [(nodw, (tok, D, tok, D, tok, D))], (layout, seen)
    assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), layout
    seen.clear()
    backward(dy, x, z, w, EPS)
    assert seen == [(name, (0, D, 0, D, 0, D))], (layout, seen)


@pytest.mark.parametrize("layout", ["projection", "head_major"])
def test_small_program_never_runs_rows_that_are_not_packed(layout, launches):
    """At 2 x 48 and 257 x 48 rows on a gate and ``grad_out`` sliced from
    projections and on head-major ``x`` and ``z`` (rows that are not packed,
    :func:`_backward_packed`), a call with ``compute_grad_weight=False``
    runs the program ``nodw_module`` names for rows that are not packed,
    never the build of packed rows only, and gives the ``dx`` and ``dz`` bits
    of the BF16 call on the band's module."""
    for T, H in ((2, 48), (257, 48)):
        rows = T * H
        assert not _nodw_packed(T, H, layout), (T, H, layout)
        name = _backward_module(rows, packed=False)
        nodw = _nodw_module(name, rows, packed=False)
        assert not _is_small(nodw), (T, H, layout, nodw)
        x, z, w, dy = make(T, H, layout=layout, seed=49)
        dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
        launches.clear()
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
        assert dw is None
        assert launches == [nodw], (T, H, layout, launches)
        assert torch.equal(dx, dx_d) and torch.equal(dz, dz_d), (T, H, layout)


# Packed shapes ``(T, H)`` the build of packed rows only runs on sm_100a and
# sm_103a (at most ``PACKED_SMALL_ROWS_PER_SM * SMs`` rows), judged against
# the naive implementation: one token, few rows of 7, 16 and 48 heads, one
# head of 256 and 512 rows, and 257 x 48.
SMALL_ACCURACY_SHAPES = (
    (1, 16),
    (1, 48),
    (3, 48),
    (10, 48),
    (21, 48),
    (64, 7),
    (256, 1),
    (512, 1),
    (257, 48),
)


@pytest.mark.parametrize("T,H", SMALL_ACCURACY_SHAPES)
def test_small_program_is_judged_against_the_naive_implementation(
    T, H, record_property, launches
):
    """Over the draws of ``judge``, a call with ``compute_grad_weight=False``
    on ``make(T, H)`` runs the build of packed rows only (skipped where it
    does not run these rows), twice with the same ``dx`` and ``dz`` bits, and
    both launches' ``dx`` and ``dz`` are judged against the naive
    implementation (the error versus the FP64 oracle within the naive
    budget)."""
    rows = T * H
    name = _backward_module(rows, packed=_backward_packed(T, H))
    nodw = _nodw_module(name, rows, packed=_nodw_packed(T, H))
    if not _is_small(nodw):
        pytest.skip(f"the build of packed rows only does not run {T} x {H} here")

    def run(j, s):
        x, z, w, dy = make(T, H, seed=s)
        launches.clear()
        launched = []
        for _ in range(2):
            dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
            assert dw is None
            launched.append({"dx": dx, "dz": dz})
        assert launches == [nodw, nodw], (j, launches)
        first, second = launched
        assert torch.equal(first["dx"], second["dx"]), (T, H, j)
        assert torch.equal(first["dz"], second["dz"]), (T, H, j)
        return silu_draw(x, z, w, dy, EPS, launched, {"dx": "dx", "dz": "dz"})

    judge(record_property, run, seed=50)
