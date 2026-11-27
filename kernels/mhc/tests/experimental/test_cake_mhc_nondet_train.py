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

Backend suite of the Cake four-stream mHC training package
``cake_pplx._backends.cake_mhc_nondet_train`` (hidden 4096, 5120 and
7168 on SM100 / SM103 / SM107; 4 streams).

The package is standalone: its forward, statistics replay, residual mixing
and backward prepare run its own copies of the deterministic programs, and
``cake_mhc_nondet_train_backward_finish(...)`` launches its copy of the
deterministic token-owner pass (its group-row form inside ``BWD_ROW_BANDS``),
its atomic fn split and its column-slab pass, which stores the FP32 ``dfn``
sum of every (column slab, token chunk) unit in its chunk's slots of the
workspace and adds the slots of each element into ``dfn`` once every unit is
done (on SM107 with the default ``accumulation="fp32"``, outside
``torch.use_deterministic_algorithms(True)``, its reduction form adds every
unit's FP32 ``dfn`` sums into ``dfn`` with relaxed reductions instead:
``cake_jit.BWD_SLAB_DFN``); there is no reduction kernel.  The mode is part of the names: no entry
point takes a ``deterministic`` keyword (passing one is a ``TypeError``) and
nothing falls back to the deterministic package (checked without a device by
``test_cake_mhc_nondet_train_rules.py``).

Accuracy is judged with ``tests/cake_naive.py``: every floating output (BF16
``U``, ``Y``, ``dB``, ``dX``; the FP32 statistics and gradient statistics; FP32
``dfn``, ``dscale``, ``dbase``) is compared with the float64 oracle of the
function (``tests/cake/test_helpers/cake_mhc_naive.py``), and its error, in the
L2 and max norms over 32 independent inputs of the case, must be at most 1.005
times the error of the naive BF16 tier (the same function in PyTorch BF16 on
the same inputs), in the median and the maximum over the draws, and 0 where the
tier's error is 0; every launch of a draw is judged (the largest error counts).
The non-deterministic package promises no bits: no test compares it with the
deterministic package or one of its launches with another bit for bit.  This
module checks:

* every output against the oracle over 32 inputs at the token counts 1 ...
  32768 (hidden 5120 and 7168 on a representative subset: small, a middle one,
  large ones where the token-owner grid reaches its per-SM bound and token
  counts at the R200 board's order cells, on that board also both sides of
  its cell's last band edge at 7168, 131072 / 131073), each
  draw's backward launched REPEATS times, alternating the default workspace
  with fresh outputs and a caller workspace (bytes past the zero prefix 0xFF)
  with NaN-filled outputs, one launch with ``B=None``; the saturated-sigmoid
  tails (tail gradients finite and non-zero), the single-cast cancellation
  inputs, a zero row of ``X`` and ``fn x 1e3`` logits the same way;
* the column slab in each unit order (``cake_jit.BWD_SLAB_ORDER`` forced) and
  the two-size schedule at forced geometries (``cake_jit.lpt_chunks`` replaced
  for one token count: one size, one small chunk, several, small chunks past
  ``T``, empty small chunks, 32 chunks, one round of units, a short odd row,
  with and without ``accumulate``, and the screen's best geometry), in each
  unit order whose record the device's route holds: the selection, the launch
  grid ``min(SMs, H / 64 x P)``, the loaded modules, the zero prefix and the
  judged outputs of two launches;
* the forward entry points on the default and a caller workspace, and at
  hidden 5120 and 7168 the 19 benchmark token counts; every launch one of this
  package's records (the deterministic package launches nothing; the backward
  loads exactly the token-owner record, the atomic fn split and the column
  slab's form of the call, in that order); the column slab's form per call by
  launch trace: the reduction form on SM107 with PyTorch's deterministic
  algorithms off, the fold with them on (strict and ``warn_only``; direct
  calls and the prepare node's backward);
* PyTorch's deterministic algorithms (strict and ``warn_only``): every entry
  point, ``backward_finish`` and the backward of the
  ``cake_mhc_nondet_train_prepare`` node included (also when the mode is
  enabled between the forward and the backward), runs without an alert or a
  warning, its outputs judged;
* four ``accumulate=True`` passes (the oracle: the carried-in values plus the
  four FP64 gradients); the autograd functions plain and under a
  non-reentrant checkpoint, around a residual and a linear branch;
* ``test_cake_mhc_det_train.py``'s checkpoint and lifetime cases with this
  package's autograd functions (the saved-tensor and autograd-edge contract:
  nothing of a connection alive once its backward has run, nothing beyond the
  outputs under a checkpoint; their gradients finite), the statistics saved
  and replayed (both judged), a changed token count without a rebuild and the
  argument validation;
* concurrent launches on several streams (each judged); ``T == 0``; the build
  manifest; the hidden sizes mixed in one process (the host caches keyed by
  the hidden size; each size judged).

The siblings hold the rest of the deterministic suite's organization:
``test_cake_mhc_nondet_train_graphs.py`` (CUDA Graph capture and replay),
``test_cake_mhc_nondet_train_iters.py`` (the run-time Sinkhorn count and the
epsilons), ``test_cake_mhc_nondet_train_workspace.py`` (caller and default
workspaces across token-count sequences, their sizes and which sizing serves
which backward), ``test_cake_mhc_nondet_train_rules.py`` (the registry against the
deterministic one, the standalone package and the mode policy without a
device) and ``test_cake_mhc_nondet_train_api.py`` (the
``cake_pplx.cake_mhc_nondet_train`` facade).

The largest cases hold the FP64 reference in row blocks of 8192 tokens.  Like
``test_cake_mhc_det_train.py`` this suite exercises the backend packages alone
(both named through ``tests.cake.test_helpers.cake_package``) and imports nothing
else from ``cake_pplx``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.cake.test_helpers.cake_mhc_naive import (
    OUTPUTS,
    STATS_COLUMNS,
    expected_and_naive,
    judge_draws,
    package_outputs,
    stats_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import DRAWS

KEY = "cake.cake_mhc_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: its loader and records (host rules, workspace sizes, the launches it must not make).
det = cake_package("cake.cake_mhc_det_train")
det_loader = cake_package("cake.cake_mhc_det_train", "cake_jit")
# The package's entry points.
backward_finish = pkg.cake_mhc_nondet_train_backward_finish
forward_prepare = pkg.cake_mhc_nondet_train_forward_prepare
forward_stats = pkg.cake_mhc_nondet_train_forward_stats
forward_finish = pkg.cake_mhc_nondet_train_forward_finish
backward_prepare = pkg.cake_mhc_nondet_train_backward_prepare

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# HIDDEN / N: the hidden size and flattened token of the cases that take no ``hidden`` (the argument checks run at
# H 4096 only).
STREAMS, HIDDEN, N, MAPPING_ROWS = 4, 4096, 16384, 24
# The package's hidden sizes; a case of a size runs where the device's architecture routes it (:func:`_require_hidden`).
HIDDEN_SIZES = (4096, 5120, 7168)
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
# The cancellation band of the cancellation inputs: |dX| <= CANCEL_FACTOR x the largest of its three contributions.
CANCEL_FACTOR = 2.0**-6
BLOCK_TOKENS = 8192
SEED = 2032
# Backward launches of this package per judged draw: every launch's outputs are judged (the largest error counts).
REPEATS = 5
# The backward workspace's zero prefix: the ticket region (256 bytes) and the column slab's arrival counts (512 bytes),
# first for every token count and hidden size; every call finds it zero and leaves it zero.
ZERO_PREFIX_BYTES = 768
# The judged token counts: hidden 4096 every one; hidden 5120 and 7168 small ones, a middle one (2049: 257 token
# groups, below the token owner's 2 x SMs at 5120 and above its SM count at 7168) and large ones where the token
# owner's grid G6 = min(K6_CTAS_PER_SM x SMs, ceil(T / 8)) reaches its bound at both sizes (4096, 16263, 32768).
T_ACCURACY = (
    1,
    2,
    3,
    7,
    63,
    64,
    65,
    129,
    255,
    257,
    1000,
    1024,
    2049,
    4096,
    5000,
    8192,
    14928,
    16263,
    32768,
)
T_ACCURACY_NEW = (1, 7, 65, 257, 1000, 2049, 4096, 16263, 32768)
# Judged after T_ACCURACY_NEW at its size, on every device: token counts of the R200 board's (212 SMs) order cells, at
# 5120 the tile 8001..8064 (at 8064), which keeps the chunk order, both sides of that tile (8000 / 8065) and both
# sides of the cell's first chunk / slab edge (9280 / 9281), at 7168 both sides of the chunk / slab edge.
T_ACCURACY_CELL = {5120: (8000, 8064, 8065, 9280, 9281), 7168: (13568, 13569)}
# Both sides of the slab / chunk edge of that board's cell at 7168, judged on that board alone (the board key
# R200_CELL_CAP_BOARD; skipped on every other device): the float64 oracle of the judge grows with the token count (at
# 7168 and 147456 tokens it runs out of the 278.5 GiB of one R200 device).
T_ACCURACY_CELL_CAP = (131072, 131073)
R200_CELL_CAP_BOARD = ("sm_107a", 212, 7168)
# Both sides of band edges of the draw cells (BWD_SLAB_DRAW_OVERRIDES) and the chain cells (FN_SPLIT_CHAIN_OVERRIDES),
# judged on every device (where a board has no cell its records of record run): at 4096 the B200 (148 SMs) draw edges
# 1664 / 1665, 2048 / 2049 and 5248 / 5249 and the GB300 (152 SMs) edge 1920 / 1921, at 7168 the B200 edge 1024 / 1025
# and the GB300 edge 576 / 577, and the B200 chain edges 1664 / 1665 and 1728 / 1729 at 4096 and 4096 / 4097 at both
# sizes (the token counts T_ACCURACY / T_ACCURACY_NEW already judge at the size are not repeated).
T_ACCURACY_DRAW = {
    4096: (1664, 1665, 1728, 1729, 1920, 1921, 2048, 4097, 5248, 5249),
    7168: (576, 577, 1024, 1025, 4097),
}
# The 19 benchmark token counts of the performance tables.
T_BENCH = (1024, 1536, 2048, 2304, 2560, 3072, 3584, 4096, 8192, 14928, 15288, 15566, 15590, 16114, 16132, 16164,
           16263, 32768, 65536)  # fmt: skip
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
_PARAMETER = ("dfn", "dscale", "dbase")
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")
# PyTorch's wording of the alert of a non-deterministic operation (no call of this package raises it).
_ALERT = "does not have a deterministic implementation"
SLAB = "cake_mhc_train_bwd_slab_fold__dynr"
LPT = "cake_mhc_train_bwd_slab_fold__lpt"
# The slab-fastest copies of the column slab's two forms (program keys ``dynr_os`` / ``lpt_os``).
SLAB_OS = "cake_mhc_train_bwd_slab_fold__dynr_os"
LPT_OS = "cake_mhc_train_bwd_slab_fold__lpt_os"
# The reduction forms of dynr / dynr_os (program keys ``dynr_red`` / ``dynr_os_red``; cake_jit.BWD_SLAB_DFN "red":
# SM107 outside torch.use_deterministic_algorithms(True)).
RED = "cake_mhc_train_bwd_slab_red__dynr"
RED_OS = "cake_mhc_train_bwd_slab_red__dynr_os"
# The late-draw copies of dynr / dynr_os (program keys ``dynr_late`` / ``dynr_os_late``;
# cake_jit.BWD_SLAB_DRAW_OVERRIDES).
SLAB_LATE = "cake_mhc_train_bwd_slab_fold__dynr_late"
SLAB_OS_LATE = "cake_mhc_train_bwd_slab_fold__dynr_os_late"
# The column-slab record of each program key (the size suffix appended at 5120 and 7168).
SLAB_RECORDS = {
    "dynr": SLAB,
    "lpt": LPT,
    "dynr_os": SLAB_OS,
    "lpt_os": LPT_OS,
    "dynr_red": RED,
    "dynr_os_red": RED_OS,
    "dynr_late": SLAB_LATE,
    "dynr_os_late": SLAB_OS_LATE,
}
# The lpt record's chunk scalars (cake_jit.slab_plan), in kernel-parameter order (P, rpc, p_big, rpc_small).
LPT_HOST_KEYS = (
    "lpt_parts",
    "lpt_rows_per_chunk",
    "lpt_big_parts",
    "lpt_small_rows_per_chunk",
)
# Forced two-size geometries (P, rpc, p_big, rpc_small) at T = 2049 (33 tiles, the last one token).
LPT_FORCED = (
    (7, 320, 7, 320),  # p_big = P: one size, 7 chunks of 5 tiles
    (5, 448, 4, 320),  # one small chunk: 4 x 7 tiles, then 5 tiles
    (6, 512, 3, 192),  # several small chunks: 3 x 8 tiles, then 3 x 3 tiles
    (8, 512, 3, 192),  # small chunks past T: 3 x 8 tiles, then 5 x 3 tiles of which the last 2 hold no token
    (6, 512, 4, 256),  # a small chunk longer than the tokens left: 4 x 8 tiles leave 1 token, the second small is empty
    (5, 768, 3, 128),  # big chunks that reach T (12, 12 and 9 tiles): both small chunks are empty
    (32, 128, 12, 64),  # 32 chunks: 12 x 2 tiles, then 20 x 1 tile of which the last 11 are empty
    (2, 1280, 1, 832),  # 128 units, one round on every board: 20 tiles, then 13
)  # fmt: skip


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_nondet_train_supported(_device())
        and loader.route_applies(
            device_capability=_capability(), hidden=HIDDEN, kind="bwd_finish"
        )
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_nondet_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries of both packages that have no prebuilt file (this package's, and the
    # deterministic package's, whose handle the argument checks refuse) through the spec objects the loaders build on
    # first use; a prebuilt or cached library loads without compiling.
    if _supported():
        arch = _arch()
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
            spec.build_and_load()
    yield


@pytest.fixture
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms``; the process-wide flags are restored however the test ends."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield torch.use_deterministic_algorithms
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture routes every kind of."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


def _require_hidden(hidden: int) -> int:
    if hidden not in _hidden_sizes():
        pytest.skip(
            f"hidden {hidden} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return hidden


def _require_lpt(hidden: int, order: str = "chunk") -> int:
    """Skip a case of the two-size column slab at ``hidden`` in the unit order ``order`` where the device's
    ``bwd_finish`` route of the size holds no such record (``lpt``, or ``lpt_os`` for ``"slab"``): where neither the
    size's table nor a board cell of ``BWD_SLAB_LPT_OVERRIDES`` at the size has a two-size calibration of the
    architecture, or for ``"slab"`` where neither its ``BWD_SLAB_ORDER`` of the size nor a board cell of
    ``BWD_SLAB_ORDER_OVERRIDES`` at the size has a ``"slab"`` band; a calibrated size without the record fails.  The
    registry calibrates the two-size schedule nowhere today (``BWD_SLAB_LPT`` and ``BWD_SLAB_LPT_HIDDEN`` are None on
    every architecture and hidden size, and ``BWD_SLAB_LPT_OVERRIDES`` is empty), so no route holds ``lpt`` or
    ``lpt_os`` and every case that calls this skips on every architecture."""
    _require_hidden(hidden)
    arch = _arch()
    name = (LPT_OS if order == "slab" else LPT) + _suffix(hidden)
    if name not in loader.route_modules(arch, hidden, "bwd_finish"):
        table = (
            loader.BWD_SLAB_LPT
            if hidden == HIDDEN
            else loader.BWD_SLAB_LPT_HIDDEN[hidden]
        )
        entries = [table[arch]] + [
            cell
            for (a, _sm, h), cell in loader.BWD_SLAB_LPT_OVERRIDES.items()
            if (a, h) == (arch, hidden)
        ]
        orders = [loader.BWD_SLAB_ORDER[hidden][arch]] + [
            cell
            for (a, _sm, h), cell in loader.BWD_SLAB_ORDER_OVERRIDES.items()
            if (a, h) == (arch, hidden)
        ]
        calibrated = any(entry is not None for entry in entries)
        slab_band = any(o == "slab" for bands in orders for _, o in bands)
        assert not (calibrated and (order == "chunk" or slab_band)), (
            f"the {arch} route of hidden {hidden} holds no {name}"
        )
        pytest.skip(f"hidden {hidden} has no {name} on {arch}")
    return hidden


def _hidden_cases(Ts: tuple, new_Ts: tuple, cell_Ts: dict | None = None) -> list:
    """``(hidden, T)`` parameters: the token counts ``Ts`` at hidden 4096, ``new_Ts`` at 5120 and 7168 and after them
    the size's ``cell_Ts`` (where given; each case skipped where the architecture has no route of its size)."""
    cell_Ts = cell_Ts or {}
    return [
        pytest.param(hidden, T, id=f"H{hidden}-{T}")
        for hidden in HIDDEN_SIZES
        for T in (Ts if hidden == HIDDEN else new_Ts + cell_Ts.get(hidden, ()))
    ]


def _suffix(hidden: int) -> str:
    """The library suffix of this package's records of ``hidden``: none at 4096, ``__h<H>`` otherwise."""
    return "" if hidden == HIDDEN else f"__h{hidden}"


def _hidden_of(i: dict) -> int:
    return int(i["X"].shape[2])


# ----------------------------------------------------------------------------------------------------------------
# Input arithmetic (the FP64 statistics the cancellation inputs are solved from)


def _round_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """One round-to-nearest-even step to BF16, also from FP64 (``Tensor.to`` rounds FP64 twice, through FP32)."""
    if x.dtype != F64:
        return x.to(BF16)
    x32 = x.float()
    b = x32.to(BF16)
    bits = x32.contiguous().view(torch.int32)
    tie = (bits & 0xFFFF) == 0x8000
    exact = x32.double() == x
    fix = tie & ~exact
    up = x.abs() > x32.abs().double()
    hi = (bits + 0x8000) & -65536
    lo = bits & -65536
    fixed = torch.where(up, hi, lo).view(F32).to(BF16)
    return torch.where(fix, fixed, b)


def _sinkhorn(logits: torch.Tensor, hc_eps: float, iters: int) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _coefficients(z, scale, base, hc_eps: float = HC_EPS, iters: int = SINKHORN_ITERS):
    T = z.shape[0]
    pre = torch.sigmoid(z[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = _sinkhorn(
        (z[:, 8:] * scale[2] + base[8:]).reshape(T, STREAMS, STREAMS), hc_eps, iters
    )
    return pre, post, C


def _constants(i: dict) -> dict:
    """The launch constants of a case: the epsilons, the Sinkhorn count and the accumulation mode it carries (default:
    the model's, one FP32 word)."""
    return {
        k: i[k]
        for k in ("rms_eps", "hc_eps", "sinkhorn_iters", "accumulation")
        if k in i
    }


# ----------------------------------------------------------------------------------------------------------------
# Inputs and launches


def _make_inputs(
    T: int,
    *,
    hidden: int = HIDDEN,
    mode: str = "unit",
    seed: int = SEED,
    fn_scale: float = 1.0,
    zero_rows=(),
) -> dict:
    """The inputs of one case (``test_cake_mhc_det_train.py``'s generator): ``X, B, dY, dU ~ N(0, 1)`` BF16;
    ``fn ~ 1e-2 N(0, 1)`` (``1e-4`` in the ``tail`` and ``cancel`` regimes, times ``fn_scale``); ``scale ~ 1 +
    0.1 N``; ``base ~ 0.1 N`` (``base[:8] = 20`` in ``tail``); the tokens of ``zero_rows`` zeroed in ``X``; in
    ``cancel`` ``dU := BF16(-(sum_j C_0j dY_j) / pre_0)`` from the FP64 statistics, so the collapse and
    residual-mixing contributions of stream 0 to ``dX`` cancel.  ``X`` / ``dY`` are ``[T, 4, hidden]``, ``B`` /
    ``dU`` ``[T, hidden]`` and ``fn`` ``[24, 4 hidden]``."""
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    fs = {"unit": 1e-2, "tail": 1e-4, "cancel": 1e-4}[mode]
    X = randn(T, STREAMS, hidden, dtype=BF16)
    fn = (randn(MAPPING_ROWS, STREAMS * hidden) * fs).contiguous()
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    if mode == "tail":
        base[:8] = 20.0
    B = randn(T, hidden, dtype=BF16)
    dY = randn(T, STREAMS, hidden, dtype=BF16)
    dU = randn(T, hidden, dtype=BF16)
    if fn_scale != 1.0:
        fn = (fn * float(fn_scale)).contiguous()
    if zero_rows:
        X[list(zero_rows)] = 0
    if mode == "cancel" and T > 0:
        with torch.no_grad():
            dU_new = torch.empty_like(dU)
            for lo in range(0, T, BLOCK_TOKENS):
                hi = min(T, lo + BLOCK_TOKENS)
                u = X[lo:hi].double().reshape(hi - lo, -1)
                r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
                z = (u * r) @ fn.double().T
                pre, _post, C = _coefficients(z, scale.double(), base.double())
                mix_0 = torch.einsum("tj,tjd->td", C[:, 0, :], dY[lo:hi].double())
                dU_new[lo:hi] = _round_to_bf16(-mix_0 / pre[:, 0:1])
            dU = dU_new
    return {
        "X": X,
        "fn": fn,
        "scale": scale,
        "base": base,
        "B": B,
        "dY": dY,
        "dU": dU,
        "mode": mode,
    }


def _forward(i: dict) -> dict:
    """The package's forward prepare and backward prepare of a case: the ``stats`` / ``grad_stats`` records
    ``backward_finish`` takes (copies of the deterministic programs, so both backwards take the same records)."""
    consts = _constants(i)
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"], **consts)
    _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    return {"stats": stats, "gstats": gstats}


def _backward(entry, i: dict, prep: dict, **kwargs) -> dict:
    out = entry(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"],
        **_constants(i), **kwargs,
    )  # fmt: skip
    return dict(zip(_OUTPUTS, out, strict=True))


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    return _backward(backward_finish, i, prep, **kwargs)


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _poisoned_outputs(T: int, hidden: int = HIDDEN) -> dict:
    """Caller-provided outputs holding NaN in every element (what the kernels must overwrite)."""
    nan = float("nan")
    return {
        "dX": torch.full((T, STREAMS, hidden), nan, dtype=BF16, device=_device()),
        "dfn": torch.full(
            (MAPPING_ROWS, STREAMS * hidden), nan, dtype=F32, device=_device()
        ),
        "dscale": torch.full((3,), nan, dtype=F32, device=_device()),
        "dbase": torch.full((MAPPING_ROWS,), nan, dtype=F32, device=_device()),
    }


def _nd_workspace_exact(T: int, hidden: int = HIDDEN) -> torch.Tensor:
    """A fresh zeroed backward workspace of this package of exactly ``T`` tokens' layout at ``hidden``."""
    return torch.zeros(
        loader.workspace_bytes("bwd", T, _sm_count(), _arch(), hidden),
        dtype=torch.uint8,
        device=_device(),
    )


def _poison_workspace(ws: torch.Tensor) -> None:
    """Every byte past the zero prefix (the ticket region and the arrival counts, first for every token count) set
    to 0xFF."""
    ws[ZERO_PREFIX_BYTES:].fill_(0xFF)


def _assert_zero_prefix(ws: torch.Tensor, what: str) -> None:
    torch.cuda.synchronize()
    nonzero = int(torch.count_nonzero(ws[:ZERO_PREFIX_BYTES]))
    assert nonzero == 0, (
        f"{what}: {nonzero} non-zero bytes in the ticket region and the arrival counts"
    )


# ----------------------------------------------------------------------------------------------------------------
# The judge: every floating output against the naive BF16 tier over DRAWS inputs (tests/cake_naive.py)


def _forward_all(i: dict, **workspace) -> dict:
    """This package's forward prepare, residual mixing and backward prepare on ``i`` (with its launch constants):
    ``U``, ``stats``, ``Y``, ``dB``, ``gstats``."""
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], **_constants(i), **workspace
    )
    Y = forward_finish(i["X"], i["B"], stats)
    dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    return {"U": U, "stats": stats, "Y": Y, "dB": dB, "gstats": gstats}


def _random_carried(seed: int, hidden: int) -> dict:
    """Non-zero carried-in values of the parameter gradients (``accumulate``), ``~ 1e-3 N(0, 1)``."""
    g = torch.Generator(device=_device()).manual_seed(seed)
    return {
        "dfn": torch.randn(
            (MAPPING_ROWS, STREAMS * hidden), generator=g, device=_device()
        )
        * 1e-3,
        "dscale": torch.randn((3,), generator=g, device=_device()) * 1e-3,
        "dbase": torch.randn((MAPPING_ROWS,), generator=g, device=_device()) * 1e-3,
    }


def _judge_nd(
    make,
    record_property,
    *,
    seed: int,
    launches: int = REPEATS,
    caller_workspace: bool = True,
    forward: bool = True,
    accumulate: bool = False,
    without_b: bool = True,
    prefix: str = "",
    draws: int = DRAWS,
    check=None,
) -> None:
    """Judge one case of this package over ``draws`` inputs (``make(s)``: the inputs of the draw of seed ``s``)
    against the naive BF16 tier: the forward outputs (``U``, ``Y``, ``dB``, the statistics ``r, z, pre, post, C``,
    the gradient statistics ``dpost, dC``; with ``forward``) and the backward outputs (``dX``, ``dfn``, ``dscale``,
    ``dbase``) of ``launches`` backward launches, each launch's error recorded (a draw keeps the largest).  The
    launches alternate: even launches on the default workspace with fresh outputs (launch 2 passes ``B=None``, whose
    contribution is in the gradient statistics, with ``without_b``), odd launches on one caller workspace whose bytes
    past the zero prefix are 0xFF before the launch, into NaN-filled outputs, the zero prefix checked zero after it
    (``caller_workspace``).  With ``accumulate`` every launch adds into a fresh copy of the draw's carried-in values
    (the oracle: those values plus the FP64 gradients; the tier: the BF16 values plus the BF16 gradients).
    ``check(i, outs, expected)`` runs on every draw (``outs`` the backward launches' outputs)."""

    def case(j: int, s: int):
        i = make(s)
        T, hidden = int(i["X"].shape[0]), _hidden_of(i)
        fwd = _forward_all(i)
        prep = {"stats": fwd["stats"], "gstats": fwd["gstats"]}
        carried = _random_carried(s + 1, hidden) if accumulate else None
        ws = None
        if caller_workspace and launches > 1:
            ws = pkg.cake_mhc_nondet_train_backward_workspace(
                T, _device(), hidden=hidden
            )
        outs = []
        for launch in range(launches):
            kwargs = {}
            if carried is not None:
                kwargs.update(
                    accumulate=True, **{k: v.clone() for k, v in carried.items()}
                )
            if launch % 2 and ws is not None:
                _poison_workspace(ws)
                kwargs["workspace"] = ws
                if carried is None:
                    kwargs.update(_poisoned_outputs(T, hidden))
            inputs = {**i, "B": None} if (launch == 2 and without_b) else i
            outs.append(_snapshot(_nd_backward(inputs, prep, **kwargs)))
            if "workspace" in kwargs:
                _assert_zero_prefix(ws, f"{prefix}T={T} H={hidden} launch {launch}")
        expected, naive = expected_and_naive(i, OUTPUTS)
        if carried is not None:
            for k in _PARAMETER:
                expected[k] = carried[k].double() + expected[k]
                naive[k] = carried[k].to(BF16) + naive[k]
        if check is not None:
            check(i, outs, expected)
        judged = ([package_outputs(fwd)] if forward else []) + outs
        names = {k for launch in judged for k in launch}
        return judged, {k: expected[k] for k in names}, {k: naive[k] for k in names}

    judge_draws(case, record_property, seed=seed, draws=draws, prefix=prefix)


def _assert_tail_gradients(res: dict, expected: dict, label: str) -> None:
    """Saturated sigmoid tails: the pre / post lanes (rows 0..7) of dbase, the pre / post entries of dscale and the
    pre / post rows of dfn are finite and non-zero where the FP64 tail gradient is non-zero (their accuracy is the
    judge's; the naive tier's BF16 sigmoid rounds to 1 there and loses the tail)."""
    lanes = {
        "dbase[0:8]": (res["dbase"][:8], expected["dbase"][:8]),
        "dscale[0:2]": (res["dscale"][:2], expected["dscale"][:2]),
        "dfn[0:8]": (res["dfn"][:8], expected["dfn"][:8]),
    }
    for name, (out, want) in lanes.items():
        assert torch.isfinite(out).all(), f"{label} {name}: non-finite"
        assert (want != 0).all(), (
            f"{label} {name}: the FP64 tail gradient is zero; the case does not test the tail"
        )
        assert (out != 0).all() if out.ndim == 1 else (out != 0).any(dim=1).all(), (
            f"{label} {name}: a tail gradient is zero"
        )


def _assert_cancellation_inputs(i: dict, expected: dict, label: str) -> None:
    """The cancellation inputs exercise the band: some element of ``dX`` lies within ``CANCEL_FACTOR`` of the largest
    of its three FP64 contributions (the mapping, collapse and residual-mixing terms)."""
    total = expected["dX"]
    collapse = expected["pre"].unsqueeze(-1) * i["dU"].double().unsqueeze(1)
    mix = torch.einsum("tij,tjd->tid", expected["C"], i["dY"].double())
    mapping = total - collapse - mix
    biggest = torch.maximum(torch.maximum(collapse.abs(), mix.abs()), mapping.abs())
    inside = int((total.abs() <= CANCEL_FACTOR * biggest).sum())
    assert inside > 0, (
        f"{label}: the cancellation inputs produced no element inside the band"
    )


# ----------------------------------------------------------------------------------------------------------------
# Accuracy against the FP64 oracle, judged against the naive tier


@pytest.mark.parametrize(
    ("hidden", "T"), _hidden_cases(T_ACCURACY, T_ACCURACY_NEW, T_ACCURACY_CELL)
)
def test_nondeterministic_backward_matches_fp64(
    hidden: int, T: int, record_property
) -> None:
    """Every output of this package (the forward entry points and REPEATS backward launches, alternating the default
    workspace with fresh outputs and a poisoned caller workspace with NaN outputs, one launch with ``B=None``) judged
    against the naive BF16 tier over 32 inputs; the caller workspace's zero prefix is zero after every launch."""
    _require_hidden(hidden)
    _judge_nd(
        lambda s: _make_inputs(T, hidden=hidden, seed=s), record_property, seed=SEED
    )


@pytest.mark.parametrize(
    ("hidden", "T"),
    [pytest.param(h, T, id=f"H{h}-{T}") for h, Ts in T_ACCURACY_DRAW.items() for T in Ts],
)
def test_draw_and_chain_cell_edges_match_fp64(
    hidden: int, T: int, record_property
) -> None:
    """Both sides of the band edges of the draw and chain cells (:data:`T_ACCURACY_DRAW`), judged as
    :func:`test_nondeterministic_backward_matches_fp64` judges each case: on B200 (148 SMs) and GB300 (152 SMs) the
    late-draw column slab on one side of a draw edge and the fold's draw on the other, and on B200 the ``"dep1"`` fn
    split to 4096 tokens (at 4096 but from 1665 to 1728) and the ``"dep"`` one elsewhere; every other device runs its
    records of record."""
    _require_hidden(hidden)
    _judge_nd(
        lambda s: _make_inputs(T, hidden=hidden, seed=s), record_property, seed=SEED
    )


@pytest.mark.parametrize("T", T_ACCURACY_CELL_CAP)
def test_r200_cell_cap_matches_fp64(T: int, record_property) -> None:
    """Both sides of the last band edge of the R200 board's (212 SMs) order cell at 7168 (the slab order to 131072
    tokens, the chunk order above), judged as :func:`test_nondeterministic_backward_matches_fp64` judges each case, on
    a device whose board key ``(arch, SM count, hidden)`` is exactly :data:`R200_CELL_CAP_BOARD`; skipped on every
    other device.  Elsewhere both token counts run one order (the slab order on sm_100a and sm_103a, GB300's cell
    included; the chunk order on the other sm_107a boards), so they are no band edge there, and the judge's float64
    oracle grows with the token count."""
    hidden = _require_hidden(7168)
    board = (_arch(), _sm_count(), hidden)
    if board != R200_CELL_CAP_BOARD:
        pytest.skip(f"{board} is not the board {R200_CELL_CAP_BOARD} of the cell")
    _judge_nd(
        lambda s: _make_inputs(T, hidden=hidden, seed=s), record_property, seed=SEED
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_saturated_sigmoid_tails(T: int, record_property) -> None:
    """Saturated sigmoid tails (``scale z + base >= 17``): every output judged against the naive BF16 tier over 32
    inputs; on every launch the tail gradients are finite and non-zero."""

    def check(i: dict, outs: list, expected: dict) -> None:
        for launch, out in enumerate(outs):
            _assert_tail_gradients(out, expected, f"T={T} mode=tail launch {launch}")

    _judge_nd(
        lambda s: _make_inputs(T, mode="tail", seed=s),
        record_property,
        seed=SEED,
        check=check,
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_single_cast_cancellation(T: int, record_property) -> None:
    """Inputs whose collapse and residual-mixing contributions to ``dX`` cancel on stream 0: every output, ``dX``
    included, judged against the naive BF16 tier over 32 inputs; every draw holds elements inside the band."""

    def check(i: dict, outs: list, expected: dict) -> None:
        _assert_cancellation_inputs(i, expected, f"T={T} mode=cancel")

    _judge_nd(
        lambda s: _make_inputs(T, mode="cancel", seed=s),
        record_property,
        seed=SEED,
        check=check,
    )


@pytest.fixture
def forced_order():
    """``force(hidden, order)``: the board cells of ``cake_jit.BWD_SLAB_ORDER_OVERRIDES`` and
    ``cake_jit.BWD_SLAB_LPT_OVERRIDES`` emptied (a board cell would replace the forced order, as on sm_107a with 212
    SMs at hidden 4096 and on sm_103a with 152 SMs at 7168, and its calibration the architecture's) and
    ``cake_jit.BWD_SLAB_ORDER`` of the device's architecture at ``hidden`` set to ``order`` (``"chunk"`` or
    ``"slab"``) at every token count, with the package's route-plan caches cleared on entry, after each edit and after
    the restore (a plan binds the selection of its token count)."""
    with pytest.MonkeyPatch.context() as patch:

        def force(hidden: int, order: str) -> None:
            patch.setattr(loader, "BWD_SLAB_ORDER_OVERRIDES", {})
            patch.setattr(loader, "BWD_SLAB_LPT_OVERRIDES", {})
            patch.setitem(loader.BWD_SLAB_ORDER[hidden], _arch(), ((None, order),))
            pkg._PLANS.clear()
            pkg._FAST_ROUTES.clear()

        pkg._PLANS.clear()
        pkg._FAST_ROUTES.clear()
        try:
            yield force
        finally:
            patch.undo()
            pkg._PLANS.clear()
            pkg._FAST_ROUTES.clear()


@pytest.mark.parametrize("order", ("chunk", "slab"))
@pytest.mark.parametrize(("hidden", "T"), _hidden_cases((2049, 16263), (2049, 16263)))
def test_backward_in_each_unit_order(
    forced_order, hidden: int, T: int, order: str, record_property
) -> None:
    """The size's unit order forced to ``order`` at every token count (the board cells emptied: :func:`forced_order`):
    the chunk-order records (``dynr`` / ``lpt``) and the slab-fastest ones (``dynr_os`` / ``lpt_os``) both run where
    the device's route holds them, whichever order the tables of record pick on this device (skipped where the route
    holds no record of the order), and the backward outputs of REPEATS launches (the workspaces alternating as in
    :func:`test_nondeterministic_backward_matches_fp64`) are judged against the naive BF16 tier over 32 inputs (on
    SM107 the reduction forms ``dynr_red`` / ``dynr_os_red`` of those records)."""
    _require_hidden(hidden)
    forced_order(hidden, order)
    program = loader.bwd_slab_atomic_program(T, _sm_count(), _arch(), hidden)
    assert program.removesuffix("_red").endswith("_os") == (order == "slab"), program
    assert program.endswith("_red") == (loader.BWD_SLAB_DFN[_arch()] == "red"), program
    name = SLAB_RECORDS[program] + _suffix(hidden)
    if name not in loader.route_modules(_arch(), hidden, "bwd_finish"):
        pytest.skip(f"hidden {hidden} has no {name} on {_arch()}")
    _judge_nd(
        lambda s: _make_inputs(T, hidden=hidden, seed=s),
        record_property,
        seed=SEED + 1,
        forward=False,
    )


@pytest.fixture
def forced_lpt():
    """``forced[T] = (P, rpc, p_big, rpc_small)``: the geometry ``cake_jit.lpt_chunks`` returns at ``T`` instead of
    the host rule's at every hidden size (every other token count keeps the rule), with the package's route-plan
    caches cleared on entry and exit (a plan binds the selection and the chunk scalars of its token count)."""
    forced: dict[int, tuple[int, int, int, int]] = {}
    rule = loader.lpt_chunks
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            loader,
            "lpt_chunks",
            lambda T, sm, arch, hidden=HIDDEN: (
                forced[T] if T in forced else rule(T, sm, arch, hidden=hidden)
            ),
        )
        pkg._PLANS.clear()
        pkg._FAST_ROUTES.clear()
        try:
            yield forced
        finally:
            pkg._PLANS.clear()
            pkg._FAST_ROUTES.clear()


# (hidden, T, forced geometry or None for the screen's best geometry on this device, accumulate)
_FORCED_CASES = (
    *((HIDDEN, 2049, geometry, False) for geometry in LPT_FORCED),
    (HIDDEN, 257, (3, 128, 2, 64), False),  # 5 tiles: 2 x 2 tiles, then one token
    (HIDDEN, 2049, LPT_FORCED[2], True),
    (HIDDEN, 2049, LPT_FORCED[6], True),
    (HIDDEN, 16263, None, False),
    # hidden 5120 and 7168 (80 / 112 column slabs): one small chunk, an empty small chunk, the short odd row, 32
    # chunks under accumulate, and the screen's best geometry at a token count the size's kernel-level pairs measured
    *(
        (hidden, T, geometry, accumulate)
        for hidden in HIDDEN_SIZES[1:]
        for T, geometry, accumulate in (
            (2049, LPT_FORCED[1], False),
            (2049, LPT_FORCED[4], False),
            (257, (3, 128, 2, 64), False),
            (2049, LPT_FORCED[6], True),
        )
    ),
    (5120, 15566, None, False),
    (7168, 15288, None, False),
)


def _forced_id(case: tuple) -> str:
    hidden, T, geometry, accumulate = case
    shape = "screen" if geometry is None else "x".join(str(v) for v in geometry)
    return f"H{hidden}-T{T}-{shape}" + ("-accumulate" if accumulate else "")


@pytest.mark.parametrize("order", ("chunk", "slab"))
@pytest.mark.parametrize(
    ("hidden", "T", "geometry", "accumulate"),
    _FORCED_CASES,
    ids=[_forced_id(case) for case in _FORCED_CASES],
)
def test_two_size_schedule_at_forced_geometries(
    forced_lpt,
    forced_order,
    monkeypatch,
    hidden: int,
    T: int,
    geometry,
    accumulate: bool,
    order: str,
    record_property,
) -> None:
    """The column slab's ``lpt`` record of ``hidden`` (under ``order="slab"`` its slab-fastest copy ``lpt_os``; where
    the device's route holds the record, see :func:`_require_lpt`), with the size's unit order forced to ``order`` at
    every token count, at a forced geometry (``geometry`` None: the screen's best two-size geometry of the size at
    ``T`` on this device, ``_lpt_plan`` at ``o_tiles = 1``, also where the table of record is ``None``): the
    selection's column slab is that record, the only column slab ``backward_finish`` loads, launched over
    ``min(SMs, H / 64 x P)`` CTAs with ``slab_plan`` = the geometry.  Over 32 inputs, two launches a draw -- fresh
    outputs on the default workspace, then NaN outputs (the carried-in values under ``accumulate``) on a caller
    workspace whose bytes past the zero prefix are 0xFF, its zero prefix zero after the launch -- have their
    backward outputs judged against the naive BF16 tier."""
    _require_lpt(hidden, order)
    arch, sm, index = _arch(), _sm_count(), _device().index
    forced_order(hidden, order)
    suffix = "_os" if order == "slab" else ""
    slab = SLAB_RECORDS["dynr" + suffix] + _suffix(hidden)
    lpt = SLAB_RECORDS["lpt" + suffix] + _suffix(hidden)
    if geometry is None:
        plan = loader._lpt_plan(T, sm, 1.0, hidden=hidden)
        assert plan is not None, (T, sm, hidden)
        geometry = plan[2]
    forced_lpt[T] = geometry
    P = geometry[0]
    assert loader.slab_plan(T, sm, arch, hidden) == dict(
        zip(LPT_HOST_KEYS, geometry, strict=True)
    )
    selection = loader.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
    assert selection[-1] == lpt and slab not in selection, selection
    assert loader.launch_grid(loader.MODULES[lpt], T, device_index=index) == (
        min(sm, hidden // loader.SLAB_D * P),
        1,
        1,
    )
    # The forward programs load before the recording starts (only the backward's loads are counted).
    _forward_all(_make_inputs(T, hidden=hidden, seed=SEED + 70))
    loaded: list[str] = []
    load = loader.load

    def recording_load(name: str, record_arch: str):
        loaded.append(name)
        return load(name, record_arch)

    monkeypatch.setattr(loader, "load", recording_load)
    _judge_nd(
        lambda s: _make_inputs(T, hidden=hidden, seed=s), record_property, seed=SEED + 71, launches=2,
        forward=False, accumulate=accumulate, without_b=False,
    )  # fmt: skip
    assert loaded == list(selection), loaded


# ----------------------------------------------------------------------------------------------------------------
# The forward entry points and the launches


def _leaves(i: dict) -> list:
    return [
        i[k].detach().clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")
    ]


def _grads(leaves: list) -> list:
    return [t.grad for t in leaves]


def _forward_outputs(i: dict, **workspace) -> dict:
    """The forward entry points and the backward prepare of this package on ``i`` (``workspace``: the forward
    workspace keyword of the forward prepare and the statistics replay)."""
    X, fn, scale, base = (i[k] for k in ("X", "fn", "scale", "base"))
    U, stats = forward_prepare(X, fn, scale, base, **workspace)
    dB, gstats = backward_prepare(i["dY"], i["B"], X, stats)
    return {
        "U": U,
        "stats": stats,
        "replay": forward_stats(X, fn, scale, base, **workspace),
        "Y": forward_finish(X, i["B"], stats),
        "dB": dB,
        "gstats": gstats,
    }


@pytest.mark.parametrize(
    ("hidden", "T"), _hidden_cases((1, 7, 129, 2049, 16263), (1, 7, 129, 2049, 16263))
)
def test_forward_entry_points_on_both_workspaces(
    hidden: int, T: int, record_property
) -> None:
    """The package's forward prepare, statistics replay, residual mixing and backward prepare (its copies of the
    deterministic programs), on the default workspaces and on a caller forward workspace of this package (the
    deterministic forward layout: the same size, zero-initialized): every output (``U``, ``Y``, ``dB``, the statistics
    of the record and of its replay, the gradient statistics) judged against the naive BF16 tier over 32 inputs."""
    _require_hidden(hidden)
    nbytes = pkg.cake_mhc_nondet_train_forward_workspace_bytes(
        T, _device(), hidden=hidden
    )
    assert nbytes == det.cake_mhc_det_train_forward_workspace_bytes(
        T, _device(), hidden=hidden
    )
    ws = pkg.cake_mhc_nondet_train_forward_workspace(T, _device(), hidden=hidden)
    assert ws.numel() == nbytes and int(torch.count_nonzero(ws)) == 0

    def case(j: int, s: int):
        i = _make_inputs(T, hidden=hidden, seed=s)
        judged = []
        for workspace in (None, ws):
            got = _forward_outputs(i, workspace=workspace)
            judged.append(package_outputs(got))
            judged.append(stats_outputs(got["replay"]))
        expected, naive = expected_and_naive(i, OUTPUTS)
        names = {k for launch in judged for k in launch}
        return judged, {k: expected[k] for k in names}, {k: naive[k] for k in names}

    judge_draws(case, record_property, seed=SEED + 12)


@pytest.mark.parametrize(
    ("hidden", "T"),
    [pytest.param(h, T, id=f"H{h}-{T}") for h in HIDDEN_SIZES[1:] for T in T_BENCH],
)
def test_benchmark_token_counts_at_the_new_hidden_sizes(
    hidden: int, T: int, record_property
) -> None:
    """At hidden 5120 and 7168 on the 19 benchmark token counts (their copies of the deterministic programs are
    renders of their own; H 4096 is the cases above): the forward entry points, the backward prepare and two
    ``backward_finish`` launches (the column slab's form of the call: ``lpt`` where the size's table fires on this
    device, else ``dynr``; each as its ``_os`` copy where the size's ``BWD_SLAB_ORDER`` band is ``"slab"``; on SM107
    its reduction form ``dynr_red``), the second on a poisoned caller workspace whose zero prefix is zero after it,
    judged against the naive BF16 tier over 32 inputs."""
    _require_hidden(hidden)
    _judge_nd(
        lambda s: _make_inputs(T, hidden=hidden, seed=s), record_property, seed=SEED + 14, launches=2,
        without_b=False,
    )  # fmt: skip


@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_launches_are_this_packages_records(monkeypatch, hidden: int) -> None:
    """Every module the package's entry points load (both forwards, the residual mixing, both backward passes, the
    autograd connection with the statistics saved and replayed, a run-time Sinkhorn count) is one of this package's
    records for the local architecture and the call's hidden size, and the deterministic package launches nothing
    (its launch paths refuse); ``backward_finish`` loads exactly its ``bwd_finish`` selection, in launch order: one
    token-owner record, the atomic fn split of the architecture and size (its ``"dep1"`` copy where
    ``fn_split_bwd_program`` names it) and the column slab's form of the call (``bwd_slab_atomic_program``: the
    size's ``dynr``, ``lpt``, ``dynr_os`` or ``lpt_os`` record, the late-draw copy of a fold, on SM107 the reduction
    form ``dynr_red`` / ``dynr_os_red``)."""
    _require_hidden(hidden)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the deterministic package launched a kernel")

    monkeypatch.setattr(det, "_launch_route", refuse)
    monkeypatch.setattr(det, "_run_route", refuse)
    # Fresh plan caches: every route below resolves (and loads) its records again.
    monkeypatch.setattr(pkg, "_PLANS", {})
    monkeypatch.setattr(pkg, "_FAST_ROUTES", {})
    loaded: list[tuple[str, str]] = []
    load = loader.load

    def recording_load(name: str, arch: str):
        loaded.append((name, arch))
        return load(name, arch)

    monkeypatch.setattr(loader, "load", recording_load)
    T, arch, sm = 2049, _arch(), _sm_count()
    i = _make_inputs(T, hidden=hidden, seed=SEED + 13)
    prep = _forward(i)
    forward_stats(i["X"], i["fn"], i["scale"], i["base"])
    forward_finish(i["X"], i["B"], prep["stats"])
    start = len(loaded)
    out = _nd_backward(i, prep)
    selection = loader.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm)
    assert [name for name, _ in loaded[start:]] == list(selection), loaded[start:]
    assert loader.MODULES[selection[0]]["role"] == "bwd_token", selection
    chain = (
        "dep1"
        if (arch, hidden) == ("sm_107a", 4096)
        else loader.fn_split_bwd_program(T, sm, arch, hidden)
    )
    assert selection[1] == (
        f"cake_mhc_train_fn_split_atomic__{loader.fn_split_program(arch, 1, hidden)}_{chain}"
        + _suffix(hidden)
    )
    program = loader.bwd_slab_atomic_program(T, sm, arch, hidden)
    assert selection[-1] == SLAB_RECORDS[program] + _suffix(hidden), (
        program,
        selection,
    )
    assert len(selection) == 3, selection
    assert all(torch.isfinite(v).all() for v in out.values())
    for save_stats in (True, False):
        leaves = _leaves(i)
        U, handle = _nd_prepare(*leaves, save_stats=save_stats)
        Y = _nd_finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
        (Y.float() * i["dY"].float()).sum().backward()
        assert all(torch.isfinite(t.grad).all() for t in leaves), save_stats
    counted = {**i, "sinkhorn_iters": 7}
    out = _nd_backward(counted, _forward(counted))
    assert all(torch.isfinite(v).all() for v in out.values())
    assert loaded
    for name, record_arch in loaded:
        assert name in loader.MODULES and name.startswith("cake_mhc_train_"), name
        assert record_arch == arch and arch in loader.MODULES[name]["arches"], name
        assert loader.MODULES[name]["hidden"] == hidden, name
    assert selection[-1] in {name for name, _ in loaded}


@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_deterministic_algorithms_launch_the_fold(
    deterministic_algorithms, monkeypatch, hidden: int
) -> None:
    """The column slab's form follows PyTorch's deterministic algorithms call by call, by launch trace (every FFI
    entry the route plans call is recorded under its record's name; fresh plan caches).  At 576, 1152, 2049, 16263
    and 40960 tokens (40960: a "slab" band of the R200 board's cell at every hidden size; 576 and 1152: "late" bands of
    the B200 and GB300 draw cells at 4096, 576 at 7168 too),
    ``backward_finish`` with the mode off launches its ``bwd_finish`` selection of record -- the token-owner record,
    the atomic fn split and the column slab's record (``bwd_slab_atomic_program``; on an architecture whose
    ``BWD_SLAB_DFN`` form is ``"red"``, SM107, the reduction form ``cake_mhc_train_bwd_slab_red__*``) -- and under
    the mode (strict with warnings as errors, and ``warn_only``) the same records with the fold in place of the
    reduction form (``deterministic_algorithms``), on the default workspace and on a caller workspace, the mode
    switched on and off between calls in one process (the plans are keyed on it, so each call launches its own).
    The prepare node's backward (``save_stats`` True and False) launches the form of the mode at ``backward()``.
    Every output is finite; the outputs of both forms are judged by the other cases of this module (the policy
    cases run the fold on SM107)."""
    _require_hidden(hidden)
    arch, sm = _arch(), _sm_count()
    red = loader.BWD_SLAB_DFN[arch] == "red"
    monkeypatch.setattr(pkg, "_PLANS", {})
    monkeypatch.setattr(pkg, "_FAST_ROUTES", {})
    launched: list[str] = []
    load = loader.load

    class _Traced:
        """A loaded module whose FFI entries record their record's name when called."""

        def __init__(self, name: str, module) -> None:
            self._name, self._module = name, module

        def __getattr__(self, entry: str):
            run = getattr(self._module, entry)

            def traced(*args):
                launched.append(self._name)
                return run(*args)

            return traced

    monkeypatch.setattr(
        loader, "load", lambda name, record_arch: _Traced(name, load(name, record_arch))
    )
    for T in (576, 1152, 2049, 16263, 40960):
        i = _make_inputs(T, hidden=hidden, seed=SEED + 90)
        prep = _forward(i)
        ws = pkg.cake_mhc_nondet_train_backward_workspace(T, _device(), hidden=hidden)
        want = {
            flag: list(
                loader.selected_modules(
                    arch,
                    hidden,
                    "bwd_finish",
                    rows=T,
                    sm_count=sm,
                    deterministic_algorithms=flag,
                )
            )
            for flag in (False, True)
        }
        fold = loader.bwd_slab_atomic_program(
            T, sm, arch, hidden, deterministic_algorithms=True
        )
        assert want[True][-1] == SLAB_RECORDS[fold] + _suffix(hidden), want
        assert want[False][:-1] == want[True][:-1] and len(want[True]) == 3, want
        assert want[False][-1] == (
            want[True][-1].replace(SLAB, RED, 1) if red else want[True][-1]
        ), want
        for on, warn_only in (
            (False, False),
            (True, False),
            (True, True),
            (False, False),
            (True, False),
        ):
            for workspace in (None, ws):
                launched.clear()
                deterministic_algorithms(on, warn_only=warn_only)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        out = _nd_backward(i, prep, workspace=workspace)
                        torch.cuda.synchronize()
                finally:
                    deterministic_algorithms(False)
                what = (T, on, warn_only, workspace is None)
                assert launched == want[on], (what, launched)
                assert all(torch.isfinite(v).all() for v in out.values()), what
        for save_stats in (True, False):
            for on in (False, True):
                leaves = _leaves(i)
                U, handle = _nd_prepare(*leaves, save_stats=save_stats)
                Y = _nd_finish(leaves[0], (U.float() + i["B"].float()).to(BF16), handle)
                loss = (Y.float() * i["dY"].float()).sum()
                launched.clear()
                deterministic_algorithms(on)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        loss.backward()
                        torch.cuda.synchronize()
                finally:
                    deterministic_algorithms(False)
                slabs = [n for n in launched if loader.MODULES[n]["role"] == "bwd_slab"]
                assert slabs == [want[on][-1]], (T, save_stats, on, launched)
                assert all(torch.isfinite(t.grad).all() for t in leaves), (
                    T,
                    save_stats,
                    on,
                )


# ----------------------------------------------------------------------------------------------------------------
# PyTorch's deterministic algorithms (no call of this package alerts; the facade, which forwards to this package, is
# test_cake_mhc_nondet_train_api.py's).  The oracle and the naive tier are evaluated with the mode off.


def _determinism_alerts(caught) -> list[str]:
    """The recorded warnings that carry PyTorch's alert of a non-deterministic operation (from this package or from
    any operation it runs)."""
    return [str(w.message) for w in caught if _ALERT in str(w.message)]


def test_policy_strict_runs_every_entry_point(
    deterministic_algorithms, record_property
) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (warnings are errors): the forward prepare, the statistics
    replay, the residual mixing, the backward prepare, the workspace entry points and ``backward_finish`` (on the
    default workspace with fresh outputs and on a caller workspace whose bytes past the zero prefix are 0xFF, with
    NaN-filled outputs) run without an alert or a warning, the zero prefix is zero after the call, and every output
    is judged against the naive BF16 tier over 32 inputs."""
    T = 257
    ws = pkg.cake_mhc_nondet_train_backward_workspace(T, _device())

    def case(j: int, s: int):
        i = _make_inputs(T, seed=s)
        deterministic_algorithms(True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                fwd = _snapshot(_forward_outputs(i))
                assert pkg.cake_mhc_nondet_train_backward_workspace_bytes(
                    T, _device()
                ) == loader.workspace_bytes_upto("bwd", T, _sm_count(), _arch())
                pkg.cake_mhc_nondet_train_forward_workspace(T, _device())
                prep = {"stats": fwd["stats"], "gstats": fwd["gstats"]}
                default = _snapshot(_nd_backward(i, prep))
                _poison_workspace(ws)
                caller = _snapshot(
                    _nd_backward(i, prep, workspace=ws, **_poisoned_outputs(T))
                )
        finally:
            deterministic_algorithms(False)
        _assert_zero_prefix(ws, "the call under the strict mode")
        judged = [package_outputs(fwd), stats_outputs(fwd["replay"]), default, caller]
        expected, naive = expected_and_naive(i, OUTPUTS)
        names = {k for launch in judged for k in launch}
        return judged, {k: expected[k] for k in names}, {k: naive[k] for k in names}

    judge_draws(case, record_property, seed=SEED + 6)


def test_policy_warn_only_runs_the_backward_without_a_warning(
    deterministic_algorithms, record_property
) -> None:
    """Under ``torch.use_deterministic_algorithms(True, warn_only=True)`` no ``backward_finish`` call warns (no
    recorded warning carries PyTorch's alert) and each runs this package's backward -- the package never falls
    back: on the default workspace and on one of this package's sizing the backward outputs are judged against the
    naive BF16 tier over 32 inputs, and the zero prefix is zero."""
    T = 2049
    nd_sized = _nd_workspace_exact(T)

    def case(j: int, s: int):
        i = _make_inputs(T, seed=s)
        prep = _forward(i)
        judged = []
        deterministic_algorithms(True, warn_only=True)
        try:
            for name, workspace in (("default", None), ("nd-sized", nd_sized)):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    judged.append(_snapshot(_nd_backward(i, prep, workspace=workspace)))
                assert _determinism_alerts(caught) == [], (
                    name,
                    [str(w.message) for w in caught],
                )
        finally:
            deterministic_algorithms(False)
        _assert_zero_prefix(nd_sized, "warn_only workspace=nd-sized")
        expected, naive = expected_and_naive(i, _OUTPUTS)
        return judged, expected, naive

    judge_draws(case, record_property, seed=SEED + 7)


@pytest.mark.parametrize("save_stats", (True, False))
def test_policy_prepare_node_runs_its_backward(
    deterministic_algorithms, save_stats: bool, record_property
) -> None:
    """The prepare node's forward, the finish node, (``save_stats=False``) the statistics replay and the prepare
    node's backward run without an alert or a warning whenever the deterministic algorithms are on -- recorded under
    the strict mode (warnings are errors), recorded with the mode off and the strict mode enabled before
    ``backward()``, and under ``warn_only`` (no recorded warning carries PyTorch's alert), also on a caller backward
    workspace of this package's sizing.  Over 32 inputs, ``U``, ``Y`` and the gradients of every one of these graphs
    are judged against the naive BF16 tier (each graph's outputs under its own name prefix)."""
    T = 257
    nd_sized = _nd_workspace_exact(T)

    def graph(i: dict, **kwargs) -> tuple:
        leaves = _leaves(i)
        U, handle = _nd_prepare(*leaves, save_stats=save_stats, **kwargs)
        U.retain_grad()
        B = (U.float() + i["B"].float()).to(BF16)
        Y = _nd_finish(leaves[0], B, handle)
        return {"leaves": leaves, "U": U, "B": B, "Y": Y}, (
            Y.float() * i["dY"].float()
        ).sum()

    def case(j: int, s: int):
        i = _make_inputs(T, seed=s)
        runs = {}
        # recorded under the strict mode: the forward and the backward run without an alert
        deterministic_algorithms(True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                c, loss = graph(i)
                loss.backward()
        finally:
            deterministic_algorithms(False)
        runs["strict "] = c
        # recorded with the mode off, the strict mode enabled before backward()
        c, loss = graph(i)
        deterministic_algorithms(True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                loss.backward()
        finally:
            deterministic_algorithms(False)
        runs["strict at backward "] = c
        for name, workspace in (("default", None), ("nd-sized", nd_sized)):
            c, loss = graph(i, backward_workspace=workspace)
            deterministic_algorithms(True, warn_only=True)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loss.backward()
            finally:
                deterministic_algorithms(False)
            assert _determinism_alerts(caught) == [], (
                name,
                [str(w.message) for w in caught],
            )
            runs[f"warn_only {name} "] = c
        outputs, expected, naive = {}, {}, {}
        for prefix, c in runs.items():
            for d, part in zip(
                (outputs, expected, naive), _connection_case(i, c), strict=True
            ):
                d.update({prefix + k: v for k, v in part.items()})
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 8)


# ----------------------------------------------------------------------------------------------------------------
# accumulate


@pytest.mark.parametrize(
    ("hidden", "T"), _hidden_cases((1000, 1152, 4096), (576, 4096, 40960))
)
def test_accumulate_four_times(hidden: int, T: int, record_property) -> None:
    """Four consecutive ``accumulate=True`` passes on four input sets into caller buffers that start from non-zero
    values (at 5120 and 7168 also at 40960 tokens, a "slab" band of the R200 board's cell at both sizes; at 4096 also
    at 1152 and at 5120 / 7168 at 576 tokens, a "late" band of the B200 and GB300 draw cells and a "dep1" band of the
    B200 chain cells at 4096 / 7168): the returned
    tensors are the caller's buffers; ``dfn``, ``dscale``, ``dbase`` after the four passes are
    judged against the naive BF16 tier over 32 draws of the four-fold chain (the oracle: the carried-in values plus
    the four FP64 gradients; the tier: the carried-in values in BF16 plus the four BF16 gradients, added in BF16)."""
    _require_hidden(hidden)

    def case(j: int, s: int):
        carried = _random_carried(s, hidden)
        acc = {k: v.clone() for k, v in carried.items()}
        want = {k: v.double() for k, v in carried.items()}
        tier = {k: v.to(BF16) for k, v in carried.items()}
        for step in range(4):
            i = _make_inputs(T, hidden=hidden, seed=s + 100 + step)
            prep = _forward(i)
            got = _nd_backward(i, prep, accumulate=True, **acc)
            for k in _PARAMETER:
                assert got[k] is acc[k], (k, step)
            expected, naive = expected_and_naive(i, _PARAMETER)
            for k in _PARAMETER:
                want[k] = want[k] + expected[k]
                tier[k] = tier[k] + naive[k]
            del i, prep, expected, naive
        return [acc], want, tier

    judge_draws(case, record_property, seed=SEED + 7919)


def test_accumulate_needs_the_buffers() -> None:
    i = _make_inputs(8, seed=SEED + 11)
    prep = _forward(i)
    zeros = {
        "dfn": torch.zeros((MAPPING_ROWS, N), dtype=F32, device=_device()),
        "dscale": torch.zeros((3,), dtype=F32, device=_device()),
        "dbase": torch.zeros((MAPPING_ROWS,), dtype=F32, device=_device()),
    }
    with pytest.raises(ValueError, match="accumulate=True"):
        _nd_backward(i, prep, accumulate=True)
    for missing in _PARAMETER:
        buffers = {k: v for k, v in zeros.items() if k != missing}
        with pytest.raises(ValueError, match="accumulate=True"):
            _nd_backward(i, prep, accumulate=True, **buffers)


# ----------------------------------------------------------------------------------------------------------------
# Autograd and activation checkpointing


def _branch(name: str, i: dict):
    if name == "residual":  # dB/dU = 1: the branch backward hands dB on as dU
        return lambda U: (U.float() + i["B"].float()).to(BF16)
    W = i["W"]
    return lambda U: U @ W  # a BF16 linear layer


def _connection(prepare, finish, branch, *, save_stats: bool, record=None, **kwargs):
    def connection(X, fn, scale, base):
        U, handle = prepare(X, fn, scale, base, save_stats=save_stats, **kwargs)
        B = branch(U)
        if (
            record is not None and "U" not in record
        ):  # the first forward (not a checkpoint's recompute)
            U.retain_grad()
            record["U"], record["B"] = U, B
        return finish(X, B, handle)

    return connection


_CONNECTION_OUTPUTS = ("U", "Y", "dB", "dX", "dfn", "dscale", "dbase")


def _connection_case(
    i: dict, c: dict, *, residual: bool = True
) -> tuple[dict, dict, dict]:
    """The judged outputs of one autograd connection ``c`` after its backward (``leaves``, ``U`` with its gradient
    retained, the branch output ``B``, ``Y``): ``U``, ``Y``, the gradients of the leaves (``dX``, ``dfn``, ``dscale``,
    ``dbase``) and, for the residual branch (``dB/dU = 1``), the gradient of ``U`` as ``dB``; with the oracle's and the
    naive tier's values on the branch output and the branch's ``dU`` the autograd graph produced."""
    U, leaves = c["U"], c["leaves"]
    assert U.grad is not None and U.grad.dtype == BF16
    assert leaves[0].grad.dtype == BF16
    j = {**i, "B": c["B"].detach(), "dU": U.grad.detach()}
    names = (
        _CONNECTION_OUTPUTS
        if residual
        else tuple(k for k in _CONNECTION_OUTPUTS if k != "dB")
    )
    expected, naive = expected_and_naive(j, names)
    outputs = {
        "U": U.detach(), "Y": c["Y"].detach(), "dB": U.grad, "dX": leaves[0].grad, "dfn": leaves[1].grad,
        "dscale": leaves[2].grad, "dbase": leaves[3].grad,
    }  # fmt: skip
    return {k: outputs[k] for k in names}, expected, naive


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
@pytest.mark.parametrize("branch", ("residual", "linear"))
def test_autograd_matches_fp64(
    T: int, save_stats: bool, checkpointed: bool, branch: str, record_property
) -> None:
    """``cake_mhc_nondet_train_prepare()`` / ``cake_mhc_nondet_train_finish`` around an external branch, plain or
    inside a non-reentrant ``torch.utils.checkpoint``: ``U``, ``Y`` and the gradients (and ``U``'s gradient as ``dB``
    for the residual branch) judged against the naive BF16 tier over 32 inputs, the oracle and the tier evaluated on
    the branch output and the branch's ``dU`` the autograd graph produced."""

    def case(j: int, s: int):
        i = _make_inputs(T, seed=s)
        g = torch.Generator(device=_device()).manual_seed(s + 1)
        i["W"] = (
            torch.randn((HIDDEN, HIDDEN), generator=g, device=_device()) / 64.0
        ).to(BF16)
        record: dict = {}
        connection = _connection(
            _nd_prepare,
            _nd_finish,
            _branch(branch, i),
            save_stats=save_stats,
            record=record,
        )
        leaves = _leaves(i)
        if checkpointed:
            Y = checkpoint(connection, *leaves, use_reentrant=False)
        else:
            Y = connection(*leaves)
        (Y.float() * i["dY"].float()).sum().backward()
        c = {"leaves": leaves, "U": record["U"], "B": record["B"], "Y": Y}
        outputs, expected, naive = _connection_case(i, c, residual=branch == "residual")
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 31)


# ----------------------------------------------------------------------------------------------------------------
# The autograd connection of this package: test_cake_mhc_det_train.py's checkpoint and lifetime cases (the
# saved-tensor and autograd-edge contract: nothing of a connection stays alive
# once its backward has run, nothing beyond the outputs under a checkpoint)


def _nd_prepare(*args, **kwargs):
    return pkg.cake_mhc_nondet_train_prepare(*args, **kwargs)


def _nd_finish(*args, **kwargs):
    return pkg.cake_mhc_nondet_train_finish(*args, **kwargs)


# (prepare, finish) of this package: a handle goes to the finish of the package that made it.
ND_API = (_nd_prepare, _nd_finish)


def _assert_like_grads(got, want, label: str) -> None:
    """The leaf gradients ``[X, *(fn, scale, base) per connection]`` of two runs of this package: the same dtypes and
    shapes, every one finite (this package promises no bits; the accuracy of its autograd path is judged by
    :func:`test_autograd_matches_fp64` and :func:`test_checkpoint_replay`)."""
    assert len(got) == len(want) and len(got) % 3 == 1, (label, len(got), len(want))
    for index, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g is not None and g.dtype == w.dtype and g.shape == w.shape, (
            label,
            index,
        )
        assert torch.isfinite(g).all(), f"grad {index} {label}: non-finite values"


def _plain_grads(i: dict, api, *, save_stats: bool = True) -> list:
    """The leaf gradients of one plain connection of ``api`` (branch ``B = U + B_0``, loss ``sum(Y * dY)``)."""
    prepare, finish = api
    leaves = _leaves(i)
    U, handle = prepare(*leaves, save_stats=save_stats)
    B = (U.float() + i["B"].float()).to(BF16)
    Y = finish(leaves[0], B, handle)
    (Y.float() * i["dY"].float()).sum().backward()
    return _grads(leaves)


def _live_bytes() -> int | None:
    """Bytes the live CUDA tensors asked the caching allocator for (``requested_bytes``; see
    ``test_cake_mhc_det_train.py``); ``None`` under another allocator backend, where the retention checks are
    skipped and the gradient checks still run."""
    torch.cuda.synchronize()
    if torch.cuda.get_allocator_backend() != "native":
        return None
    return int(torch.cuda.memory_stats()["requested_bytes.all.current"])


def _assert_nothing_kept(before: int | None, Y: torch.Tensor, what: str) -> None:
    if before is None:
        return
    kept = _live_bytes() - before - Y.untyped_storage().nbytes()
    assert kept == 0, f"{kept} bytes kept alive after a checkpointed {what}"


def _assert_kept(
    before: int | None, what: str, *alive: torch.Tensor, expected: int = 0
) -> None:
    """The live bytes beyond ``before`` are those of ``alive`` plus ``expected``."""
    if before is None:
        return
    kept = _live_bytes() - before - sum(t.untyped_storage().nbytes() for t in alive)
    assert kept == expected, f"{kept - expected} bytes kept alive after {what}"


@pytest.mark.parametrize(("hidden", "T"), _hidden_cases((129, 2049), (129, 2049)))
def test_checkpoint_replay(hidden: int, T: int, record_property) -> None:
    """This package's connection with the statistics saved (``save_stats=True``) and replayed in the backward
    (``False``, the checkpoint path), and the statistics replay ``forward_stats``: over 32 inputs, ``U``, ``Y``,
    ``dB`` and the gradients of each connection (each under its own name prefix) and the replayed statistics are
    judged against the naive BF16 tier."""
    _require_hidden(hidden)

    def case(j: int, s: int):
        i = _make_inputs(T, hidden=hidden, seed=s)
        outputs, expected, naive = {}, {}, {}
        for save_stats in (True, False):
            leaves = _leaves(i)
            U, handle = _nd_prepare(*leaves, save_stats=save_stats)
            U.retain_grad()
            B = (U.float() + i["B"].float()).to(BF16)
            Y = _nd_finish(leaves[0], B, handle)
            (Y.float() * i["dY"].float()).sum().backward()
            c = {"leaves": leaves, "U": U, "B": B, "Y": Y}
            for d, part in zip(
                (outputs, expected, naive), _connection_case(i, c), strict=True
            ):
                d.update({f"save_stats={save_stats} {k}": v for k, v in part.items()})
        replay = stats_outputs(forward_stats(i["X"], i["fn"], i["scale"], i["base"]))
        e, t = expected_and_naive(i, replay)
        for d, part in zip((outputs, expected, naive), (replay, e, t), strict=True):
            d.update({f"replay {k}": v for k, v in part.items()})
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 11)


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_keeps_nothing_alive(T: int, save_stats: bool) -> None:
    """A non-reentrant ``torch.utils.checkpoint`` around this package's prepare / finish keeps no tensor of
    the layer alive after the forward, and the checkpointed backward gives finite gradients of the plain ones'
    formats (their accuracy: :func:`test_autograd_matches_fp64`)."""
    i = _make_inputs(T, seed=SEED + 41)

    def connection(X, fn, scale, base):
        U, handle = _nd_prepare(X, fn, scale, base, save_stats=save_stats)
        B = (U.float() + i["B"].float()).to(BF16)
        return _nd_finish(X, B, handle)

    plain = _leaves(i)
    (connection(*plain).float() * i["dY"].float()).sum().backward()

    checkpointed = _leaves(i)
    before = _live_bytes()
    Y = checkpoint(connection, *checkpointed, use_reentrant=False)
    _assert_nothing_kept(before, Y, "forward")
    (Y.float() * i["dY"].float()).sum().backward()
    label = f"T={T} save_stats={save_stats} checkpointed"
    _assert_like_grads(_grads(checkpointed), _grads(plain), f"{label} vs plain")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_after_prepare_replays_statistics(T: int, save_stats: bool) -> None:
    """A non-reentrant checkpoint region that starts after this package's prepare (the branch and the finish
    inside it) keeps nothing beyond its output after the forward, the finish forward takes the edges off the handle
    and its rerun replays the released statistics; the gradients are finite, in the plain run's formats."""
    i = _make_inputs(T, seed=SEED + 42)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return _nd_finish(X, B, handle)

    plain = _leaves(i)
    U, handle = _nd_prepare(*plain, save_stats=save_stats)
    (finish(plain[0], U, handle).float() * i["dY"].float()).sum().backward()

    checkpointed = _leaves(i)
    U, handle = _nd_prepare(*checkpointed, save_stats=save_stats)
    # pinned: save_stats=False releases it in the finish forward
    stats = handle.state.stats
    before = _live_bytes()
    Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=False)
    _assert_nothing_kept(before, Y, "finish")
    assert handle.edges == () and handle.state.stats is None, (
        "the handle kept the edges or the statistics after the forward"
    )
    del stats
    (Y.float() * i["dY"].float()).sum().backward()
    _assert_like_grads(
        _grads(checkpointed),
        _grads(plain),
        f"T={T} save_stats={save_stats} checkpointed finish",
    )


def _two_layers(
    X, params: list, api, *, save_stats: bool, checkpointed: bool
) -> torch.Tensor:
    """Two layers of two connections each of ``api`` (the branch hands ``U`` on as ``B``), each layer a non-reentrant
    checkpoint region or not."""
    prepare, finish = api

    def connection(x, fn, scale, base):
        U, handle = prepare(x, fn, scale, base, save_stats=save_stats)
        return finish(x, U, handle)

    def layer(x, k):
        return connection(connection(x, *params[2 * k]), *params[2 * k + 1])

    if not checkpointed:
        return layer(layer(X, 0), 1)
    return checkpoint(
        layer, checkpoint(layer, X, 0, use_reentrant=False), 1, use_reentrant=False
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
def test_backward_keeps_nothing_alive(
    T: int, save_stats: bool, checkpointed: bool
) -> None:
    """Once a connection's backward has run, nothing of it stays alive: after
    ``torch.autograd.grad`` over two layers of two connections the only live tensor of the step is its output ``Y``.
    A second step gives finite gradients of the first step's formats."""
    i = _make_inputs(T, seed=SEED + 51)
    params = [
        [p.clone().requires_grad_(True) for p in (q["fn"], q["scale"], q["base"])]
        for q in (_make_inputs(1, seed=SEED + 52 + k) for k in range(4))
    ]
    X = i["X"].clone().requires_grad_(True)
    leaves = [X, *[p for ps in params for p in ps]]

    def step(api) -> tuple:
        Y = _two_layers(
            X, params, api, save_stats=save_stats, checkpointed=checkpointed
        )
        return Y, torch.autograd.grad(Y, leaves, i["dY"])

    reference = step(ND_API)[1]  # also the warm-up
    before = _live_bytes()
    Y, grads = step(ND_API)
    label = f"T={T} save_stats={save_stats} checkpointed={checkpointed}"
    _assert_like_grads(grads, reference, f"second step {label}")
    del grads
    _assert_kept(before, "the backward", Y)


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_repeated_backward_with_retain_graph(T: int, save_stats: bool) -> None:
    """Backward passes over one retained graph of this package are independent: two full passes equal a fresh
    graph's formats, and a pass in between that reaches the prepare node but not the finish node (the loss is ``U``)
    gives finite gradients of the same pass on a fresh graph's formats."""
    i = _make_inputs(T, seed=SEED + 56)

    def graph(leaves: list) -> tuple:
        U, handle = _nd_prepare(*leaves, save_stats=save_stats)
        B = (U.float() + i["B"].float()).to(BF16)
        return U, _nd_finish(leaves[0], B, handle)

    leaves = _leaves(i)
    U, Y = graph(leaves)
    first = torch.autograd.grad(Y, leaves, i["dY"], retain_graph=True)
    u_only = torch.autograd.grad(U, leaves, i["dU"], retain_graph=True)
    second = torch.autograd.grad(Y, leaves, i["dY"])

    fresh = _leaves(i)
    full = torch.autograd.grad(graph(fresh)[1], fresh, i["dY"])
    fresh = _leaves(i)
    u_fresh = torch.autograd.grad(graph(fresh)[0], fresh, i["dU"])
    label = f"T={T} save_stats={save_stats} retain_graph"
    _assert_like_grads(first, full, f"{label} first pass")
    _assert_like_grads(second, full, f"{label} second pass")
    _assert_like_grads(u_only, u_fresh, f"{label} U-only pass")


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("region", ("connection", "after_prepare"))
def test_reentrant_checkpoint(T: int, save_stats: bool, region: str) -> None:
    """A reentrant ``torch.utils.checkpoint`` around a connection of this package, or around the branch and the
    finish with the prepare outside, gives finite gradients of the plain ones' formats and keeps nothing alive after
    the backward beyond ``Y`` and the leaves' gradients."""
    i = _make_inputs(T, seed=SEED + 57)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return _nd_finish(X, B, handle)

    def connection(X, fn, scale, base):
        U, handle = _nd_prepare(X, fn, scale, base, save_stats=save_stats)
        return finish(X, U, handle)

    plain = _leaves(i)
    connection(*plain).backward(i["dY"])

    checkpointed = _leaves(i)
    before = _live_bytes()
    if region == "connection":
        Y = checkpoint(connection, *checkpointed, use_reentrant=True)
    else:
        U, handle = _nd_prepare(*checkpointed, save_stats=save_stats)
        Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=True)
        del U
    Y.backward(i["dY"])
    _assert_like_grads(
        _grads(checkpointed),
        _grads(plain),
        f"T={T} save_stats={save_stats} reentrant checkpoint ({region})",
    )
    if region == "after_prepare":
        assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    _assert_kept(
        before, "a reentrant checkpointed backward", Y, *(c.grad for c in checkpointed)
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("prepare", ("skipped", "not_recorded"))
def test_backward_without_the_prepare_node(T: int, prepare: str) -> None:
    """A backward that runs the finish node but not this package's prepare node leaves the finish node's
    ``dY`` and gradient statistics on no handle; what stays alive is the prepare node's own saved statistics while
    its backward has not run."""
    i = _make_inputs(T, seed=SEED + 58)
    leaves = (
        _leaves(i)
        if prepare == "skipped"
        else [i[k] for k in ("X", "fn", "scale", "base")]
    )

    def step() -> tuple:
        U, handle = _nd_prepare(*leaves)
        B = (U.float() + i["B"].float()).to(BF16)
        if prepare == "not_recorded":
            B.requires_grad_(True)
        Y = _nd_finish(leaves[0], B, handle)
        return Y, B, handle, torch.autograd.grad(Y, [B], i["dY"])

    step()
    before = _live_bytes()
    Y, B, handle, (dB,) = step()
    assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    saved_stats = (
        T * 256 if prepare == "skipped" else 0
    )  # the 256-byte record per token
    _assert_kept(
        before, "a backward without the prepare node", Y, B, dB, expected=saved_stats
    )


@pytest.mark.parametrize("save_stats", (True, False))
def test_interleaved_connections(save_stats: bool) -> None:
    """Two connections of this package of one shape whose prepares both run before either finish give finite
    gradients of the formats of the two connections run one after the other."""
    T = 129
    a, b = _make_inputs(T, seed=SEED + 61), _make_inputs(T, seed=SEED + 62)

    def branch(i: dict, U: torch.Tensor) -> torch.Tensor:
        return (U.float() + i["B"].float()).to(BF16)

    def run(interleaved: bool) -> list:
        la, lb = _leaves(a), _leaves(b)
        Ua, ha = _nd_prepare(*la, save_stats=save_stats)
        if interleaved:
            Ub, hb = _nd_prepare(*lb, save_stats=save_stats)
            Yb = _nd_finish(lb[0], branch(b, Ub), hb)
            Ya = _nd_finish(la[0], branch(a, Ua), ha)
        else:
            Ya = _nd_finish(la[0], branch(a, Ua), ha)
            Ub, hb = _nd_prepare(*lb, save_stats=save_stats)
            Yb = _nd_finish(lb[0], branch(b, Ub), hb)
        torch.autograd.backward([Ya, Yb], [a["dY"], b["dY"]])
        return _grads(la), _grads(lb)

    serial, interleaved = run(False), run(True)
    for n in range(2):
        _assert_like_grads(
            interleaved[n], serial[n], f"connection {n} save_stats={save_stats}"
        )


def test_zero_row_of_x_takes_the_eps_path(record_property) -> None:
    """A zero row of ``X`` (the ``rms_eps`` path): every output judged against the naive BF16 tier over 32 inputs."""
    _judge_nd(
        lambda s: _make_inputs(7, seed=s, zero_rows=(3,)), record_property, seed=SEED
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_large_logits(T: int, record_property) -> None:
    """``fn x 1e3`` (``|z| ~ 1e3``: saturated sigmoids and softmax rows): every output judged against the naive BF16
    tier over 32 inputs."""
    _judge_nd(
        lambda s: _make_inputs(T, fn_scale=1e3, seed=s), record_property, seed=SEED
    )


def test_changing_rows_in_one_process_does_not_rebuild(
    record_property, monkeypatch
) -> None:
    """A changed token count rebuilds no module: it loads no module of the deterministic package, a module of this
    package only where its route selects a record of another token-count band than the earlier calls' (a mapping-pass
    ``xn`` form at 65 or 255 tokens, for example), every module loaded during the changed counts is a record of
    their selection, and calls back at the earlier counts load nothing.  Every output at 64 tokens, and ``dX`` and
    ``dfn`` of one launch at each other count, are judged against the naive BF16 tier over 32 inputs."""
    _judge_nd(
        lambda s: _make_inputs(64, seed=s), record_property, seed=SEED, prefix="T=64 "
    )
    loaded: list[str] = []
    load = loader.load

    def recording_load(name: str, arch: str):
        misses = load.cache_info().misses
        module = load(name, arch)
        if load.cache_info().misses != misses:
            loaded.append(name)
        return module

    monkeypatch.setattr(loader, "load", recording_load)
    det_loaded = det_loader.load.cache_info().misses
    counts = (1, 65, 2049, 7, 255)
    for T in counts:

        def case(j: int, s: int, T: int = T):
            i = _make_inputs(T, seed=s)
            out = _nd_backward(i, _forward(i))
            launch = {k: out[k] for k in ("dX", "dfn")}
            expected, naive = expected_and_naive(i, launch)
            return [launch], expected, naive

        judge_draws(case, record_property, seed=SEED + T, prefix=f"T={T} ")
    arch, sm = _arch(), _sm_count()
    selected = {
        name
        for T in counts
        for kind in ("fwd_prepare", "bwd_prepare", "bwd_finish")
        for name in loader.selected_modules(arch, HIDDEN, kind, rows=T, sm_count=sm)
    }
    assert set(loaded) <= selected, (
        f"a changed token count loaded modules outside its routes' records: {sorted(set(loaded) - selected)}"
    )
    loaded_before = (load.cache_info().misses, det_loader.load.cache_info().misses)
    assert loaded_before[1] == det_loaded, (
        "a changed token count loaded a module of the deterministic package"
    )
    for T in (64, *reversed(counts)):
        i = _make_inputs(T, seed=SEED + 5 + T)
        _nd_backward(i, _forward(i))
    assert (
        load.cache_info().misses,
        det_loader.load.cache_info().misses,
    ) == loaded_before, (
        "a call back at an earlier token count rebuilt or reloaded a module"
    )


def test_argument_validation() -> None:
    """This package's backward validates like the deterministic backward (``test_cake_mhc_det_train.py``'s checks,
    on ``backward_finish`` and the autograd prepare / finish): a wrong shape, dtype, device or layout of an operand or
    an output is a ``ValueError``, as are out-of-range epsilons and Sinkhorn counts (a count above
    ``SINKHORN_ITERS_MAX`` a ``NotImplementedError``) and ``accumulate=True`` without its buffers; the workspace must
    be a 1-D, 256-byte aligned ``uint8`` CUDA tensor of this package's size (a non-tensor a ``TypeError``).
    An operand or output of another hidden size than ``X``, and a call whose hidden size the package does not list
    (6144) or the architecture does not route (none of the three sizes on the architectures of record), is a
    ``ValueError``.  ``B`` is optional: ``None`` gives every output (``dfn`` included) in the same dtype and shape as
    with ``B``, all finite; the test makes no bit comparison.  ``finish`` takes
    only this package's handles (any other object, a deterministic package's handle included, is a ``TypeError``
    naming the deterministic finish); a ``deterministic`` keyword is a ``TypeError`` too."""
    T = 8
    i = _make_inputs(T, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    prep = _forward(i)
    dev = _device()
    args = {
        "X": X,
        "dY": i["dY"],
        "dU": i["dU"],
        "B": i["B"],
        "fn": fn,
        "scale": scale,
        "base": base,
        "stats": prep["stats"],
        "grad_stats": prep["gstats"],
    }

    def call(**changes):
        operands = {**args, **{k: v for k, v in changes.items() if k in args}}
        keywords = {k: v for k, v in changes.items() if k not in args}
        return backward_finish(*operands.values(), **keywords)

    # Padded rows (a [T, 4, 4096] view of a wider buffer) are not materialized silently: the caller decides.
    padded = torch.zeros(T, STREAMS, HIDDEN + 64, dtype=BF16, device=dev)[:, :, :HIDDEN]
    nbytes = pkg.cake_mhc_nondet_train_backward_workspace_bytes(T, dev)
    shifted = torch.zeros(nbytes + 512, dtype=torch.uint8, device=dev)[1 : 1 + nbytes]
    for change in (
        {"X": X.float()},
        {"X": padded},
        {"fn": fn.to(BF16)},
        {"fn": fn[:, 1:]},
        {"scale": scale[:2]},
        {"dY": i["dY"][:4]},
        {"dU": i["dU"].float()},
        {"B": i["B"][:4]},
        {"stats": prep["stats"].double()},
        {"grad_stats": prep["gstats"][:, :16].contiguous()},
        {"accumulate": True},
        {"dX": torch.empty(T, STREAMS, HIDDEN, dtype=F32, device=dev)},
        {"dfn": torch.empty(MAPPING_ROWS, N - 1, dtype=F32, device=dev)},
        {"rms_eps": 0.0},
        {"rms_eps": -1e-5},
        {"hc_eps": float("nan")},
        {"hc_eps": float("inf")},
        {"sinkhorn_iters": 0},
        {"sinkhorn_iters": 20.5},
        {"workspace": torch.zeros(nbytes // 4, dtype=F32, device=dev)},
        {"workspace": torch.zeros(2, nbytes, dtype=torch.uint8, device=dev)},
        {"workspace": shifted},
        {"workspace": torch.zeros(256, dtype=torch.uint8, device=dev)},
        # operands and outputs of another hidden size than X's
        {"fn": torch.zeros(MAPPING_ROWS, STREAMS * 5120, dtype=F32, device=dev)},
        {"dY": torch.zeros(T, STREAMS, 7168, dtype=BF16, device=dev)},
        {"dU": torch.zeros(T, 5120, dtype=BF16, device=dev)},
        {"B": torch.zeros(T, 7168, dtype=BF16, device=dev)},
        {"dX": torch.empty(T, STREAMS, 5120, dtype=BF16, device=dev)},
        {"dfn": torch.empty(MAPPING_ROWS, STREAMS * 7168, dtype=F32, device=dev)},
    ):
        with pytest.raises(ValueError):
            call(**change)
    # every operand of a hidden size the package does not list, or that the architecture does not route
    for hidden in (6144, *(h for h in HIDDEN_SIZES if h not in _hidden_sizes())):
        j = _make_inputs(T, hidden=hidden, seed=SEED + 10)
        with pytest.raises(ValueError):
            backward_finish(
                j["X"], j["dY"], j["dU"], j["B"], j["fn"], j["scale"], j["base"], prep["stats"], prep["gstats"],
            )  # fmt: skip
        with pytest.raises(ValueError):
            forward_prepare(j["X"], j["fn"], j["scale"], j["base"])
        with pytest.raises(ValueError):
            _nd_prepare(j["X"], j["fn"], j["scale"], j["base"])
    with pytest.raises(NotImplementedError):
        call(sinkhorn_iters=det.SINKHORN_ITERS_MAX + 1)
    with pytest.raises(TypeError):
        call(workspace=[0] * 256)
    with pytest.raises(ValueError):
        backward_finish(*(t.cpu() for t in args.values()))
    # B is optional in the backward finish (its contribution is in grad_stats; the judged cases of
    # test_nondeterministic_backward_matches_fp64 launch with B=None too).
    with_b = _snapshot(dict(zip(_OUTPUTS, call(), strict=True)))
    without_b = _snapshot(dict(zip(_OUTPUTS, call(B=None), strict=True)))
    for k in _OUTPUTS:
        assert (
            without_b[k].dtype == with_b[k].dtype
            and without_b[k].shape == with_b[k].shape
        ), k
        assert torch.isfinite(without_b[k]).all(), f"{k} B=None: non-finite values"
    with pytest.raises(ValueError):
        _nd_prepare(X.float(), fn, scale, base)
    with pytest.raises(ValueError):
        _nd_prepare(padded, fn, scale, base)
    foreign = (
        "handle must be the handle returned by cake_mhc_nondet_train_prepare (a cake_mhc_det_train_prepare "
        "handle goes to cake_pplx.cake_mhc_det_train.cake_mhc_det_train_finish)"
    )
    det_handle = det.cake_mhc_det_train_prepare(X, fn, scale, base)[1]
    for handle in (object(), det_handle):
        with pytest.raises(TypeError, match=re.escape(foreign)):
            _nd_finish(X, i["B"], handle)
    for value in (False, True):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_mhc_nondet_train_finish(X, i["B"], det_handle, deterministic=value)


# ----------------------------------------------------------------------------------------------------------------
# Concurrent streams


@pytest.mark.parametrize("Ts", ((2049, 2049), (4096, 1024, 257)))
def test_concurrent_streams(Ts: tuple, record_property) -> None:
    """Launches on several ``torch.cuda.Stream`` objects at once, each with its own workspace and outputs (workspace
    bytes past the zero prefix 0xFF and NaN outputs before every round): over 32 inputs of every stream, three rounds
    a draw, each stream's backward outputs are judged against the naive BF16 tier of its inputs (each stream under its
    own name prefix), and each workspace's zero prefix is zero after every round."""
    workspaces = [
        pkg.cake_mhc_nondet_train_backward_workspace(T, _device()) for T in Ts
    ]
    outs = [_poisoned_outputs(T) for T in Ts]
    streams = [torch.cuda.Stream() for _ in Ts]
    main = torch.cuda.current_stream()

    def case(j: int, s: int):
        cases = []
        for n, T in enumerate(Ts):
            i = _make_inputs(T, seed=s + 500 + n)
            cases.append({"T": T, "i": i, "prep": _forward(i)})
        torch.cuda.synchronize()
        judged = []
        for rnd in range(3):
            for ws, out in zip(workspaces, outs, strict=True):
                _poison_workspace(ws)
                for v in out.values():
                    v.fill_(float("nan"))
            for stream, c, ws, out in zip(
                streams, cases, workspaces, outs, strict=True
            ):
                stream.wait_stream(main)
                with torch.cuda.stream(stream):
                    _nd_backward(c["i"], c["prep"], workspace=ws, **out)
            for stream in streams:
                main.wait_stream(stream)
            torch.cuda.synchronize()
            launch = {}
            for n, (c, ws, out) in enumerate(zip(cases, workspaces, outs, strict=True)):
                _assert_zero_prefix(ws, f"round {rnd} stream {n} (T={c['T']}) of {Ts}")
                launch.update({f"stream {n} {k}": v.clone() for k, v in out.items()})
            judged.append(launch)
        expected, naive = {}, {}
        for n, c in enumerate(cases):
            e, t = expected_and_naive(c["i"], _OUTPUTS)
            expected.update({f"stream {n} {k}": v for k, v in e.items()})
            naive.update({f"stream {n} {k}": v for k, v in t.items()})
        return judged, expected, naive

    judge_draws(case, record_property, seed=SEED + 500)


# ----------------------------------------------------------------------------------------------------------------
# Hidden sizes mixed in one process (test_cake_mhc_det_train.py's cases: the host caches -- layouts, default
# workspaces and their views, caller-workspace views, launch plans, fast routes, route checks -- are keyed by the
# hidden size; one keyed by the token count alone would serve a size's views or plan to another size's call)

_MIXED_T = (129, 2049)
_WORKSPACE_MODES = ("default", "per_hidden", "shared")


def _mixed_hidden_sizes() -> tuple[int, ...]:
    """Every routed hidden size, for the cases that mix sizes in one process (4096 and 7168 at least)."""
    sizes = _hidden_sizes()
    if not {4096, 7168} <= set(sizes):
        pytest.skip(
            f"mixing hidden sizes needs 4096 and 7168 on one device; {_arch()} exports {sizes}"
        )
    return sizes


def _mixed_workspaces(sizes: tuple[int, ...], T: int, mode: str) -> dict:
    """``{H: (forward workspace, backward workspace)}`` of a mixed-size case: ``default`` (``None``: the per-device
    default buffers), ``per_hidden`` (one caller buffer per size, ``hidden=H``) or ``shared`` (one zeroed caller
    buffer per kind serving every size, as large as the largest size's request)."""
    dev = _device()
    if mode == "default":
        return {H: (None, None) for H in sizes}
    nbytes = {
        "fwd": pkg.cake_mhc_nondet_train_forward_workspace_bytes,
        "bwd": pkg.cake_mhc_nondet_train_backward_workspace_bytes,
    }
    if mode == "per_hidden":
        return {
            H: tuple(
                torch.zeros(
                    size(T, dev, hidden=H),
                    dtype=torch.uint8,
                    device=dev,
                )
                for size in nbytes.values()
            )
            for H in sizes
        }
    assert mode == "shared", mode
    shared = tuple(
        torch.zeros(
            max(size(T, dev, hidden=H) for H in sizes),
            dtype=torch.uint8,
            device=dev,
        )
        for size in nbytes.values()
    )
    return {H: shared for H in sizes}


def _staged_run(cases: dict, order: tuple[int, ...], ws: dict) -> dict:
    """This package's entry points of every case stage by stage: each stage is issued for every hidden size (in
    ``order``, the next stage in the reverse order) before the next stage, so consecutive calls switch the hidden size
    at one token count.  ``replay`` is the ``forward_stats`` record of each size."""
    out: dict[int, dict] = {H: {} for H in order}
    back = tuple(reversed(order))
    for H in order:
        i = cases[H]
        out[H]["U"], out[H]["stats"] = forward_prepare(
            i["X"], i["fn"], i["scale"], i["base"], workspace=ws[H][0]
        )
    for H in back:
        i = cases[H]
        out[H]["replay"] = forward_stats(
            i["X"], i["fn"], i["scale"], i["base"], workspace=ws[H][0]
        )
    for H in order:
        i = cases[H]
        out[H]["Y"] = forward_finish(i["X"], i["B"], out[H]["stats"])
    for H in back:
        i, o = cases[H], out[H]
        o["dB"], o["gstats"] = backward_prepare(i["dY"], i["B"], i["X"], o["stats"])
    for H in order:
        i, o = cases[H], out[H]
        o["dX"], o["dfn"], o["dscale"], o["dbase"] = backward_finish(
            i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], o["stats"], o["gstats"],
            workspace=ws[H][1],
        )  # fmt: skip
    return {H: _snapshot(o) for H, o in out.items()}


@pytest.mark.parametrize("T", _MIXED_T)
@pytest.mark.parametrize("workspaces", _WORKSPACE_MODES)
def test_mixed_hidden_sizes_same_token_count(
    T: int, workspaces: str, record_property
) -> None:
    """Calls of every routed hidden size at the same token count, interleaved stage by stage through the direct
    entry points (default workspaces, one caller workspace per size, one caller workspace shared by the sizes), in
    two rounds of opposite size order: over 32 inputs of every size, every output of both rounds is judged against
    the naive BF16 tier of its size, and a caller backward workspace keeps its zero prefix zero."""
    sizes = _mixed_hidden_sizes()
    ws = _mixed_workspaces(sizes, T, workspaces)

    def case(j: int, s: int):
        cases = {H: _make_inputs(T, hidden=H, seed=s + H) for H in sizes}
        judged = []
        for rnd, order in enumerate((sizes, tuple(reversed(sizes)))):
            got = _staged_run(cases, order, ws)
            launch = {}
            for H in sizes:
                what = f"T={T} H={H} mixed with {sizes} ({workspaces} workspaces, round {rnd})"
                if ws[H][1] is not None:
                    _assert_zero_prefix(ws[H][1], what)
                outputs = package_outputs(got[H])
                outputs.update(
                    {
                        "replay " + k: v
                        for k, v in stats_outputs(got[H]["replay"]).items()
                    }
                )
                launch.update({f"H{H} {k}": v for k, v in outputs.items()})
            judged.append(launch)
        expected, naive = {}, {}
        for H in sizes:
            e, t = expected_and_naive(cases[H], OUTPUTS)
            for k in OUTPUTS:
                expected[f"H{H} {k}"], naive[f"H{H} {k}"] = e[k], t[k]
            for k in STATS_COLUMNS:
                expected[f"H{H} replay {k}"], naive[f"H{H} replay {k}"] = e[k], t[k]
        return judged, expected, naive

    judge_draws(case, record_property, seed=SEED + 71)


@pytest.mark.parametrize("T", _MIXED_T)
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("workspaces", _WORKSPACE_MODES)
def test_mixed_hidden_sizes_autograd(
    T: int, save_stats: bool, workspaces: str, record_property
) -> None:
    """This package's autograd connections with every routed hidden size at the same token count: the
    prepares of every size, then the finishes in the reverse order, then one backward over the sum of the losses.
    Over 32 inputs of every size, each size's ``U``, ``Y``, ``dB`` and gradients are judged against the naive BF16
    tier of its size, the oracle and the tier on the branch output and the ``dU`` the graph produced."""
    sizes = _mixed_hidden_sizes()
    ws = _mixed_workspaces(sizes, T, workspaces)

    def case(j: int, s: int):
        cases = {H: _make_inputs(T, hidden=H, seed=s + H) for H in sizes}
        connections = {}
        for H in sizes:
            leaves = _leaves(cases[H])
            U, handle = _nd_prepare(
                *leaves,
                save_stats=save_stats,
                workspace=ws[H][0],
                backward_workspace=ws[H][1],
            )
            U.retain_grad()
            connections[H] = {"leaves": leaves, "U": U, "handle": handle}
        for H in reversed(sizes):
            c = connections[H]
            c["B"] = (c["U"].float() + cases[H]["B"].float()).to(BF16)
            c["Y"] = _nd_finish(c["leaves"][0], c["B"], c.pop("handle"))
        torch.stack(
            [
                (connections[H]["Y"].float() * cases[H]["dY"].float()).sum()
                for H in sizes
            ]
        ).sum().backward()
        outputs, expected, naive = {}, {}, {}
        for H in sizes:
            for d, part in zip(
                (outputs, expected, naive),
                _connection_case(cases[H], connections[H]),
                strict=True,
            ):
                d.update({f"H{H} {k}": v for k, v in part.items()})
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 81)


# ----------------------------------------------------------------------------------------------------------------
# Zero tokens, the build manifest


@pytest.mark.parametrize("hidden", HIDDEN_SIZES, ids=lambda h: f"H{h}")
def test_zero_tokens_launch_nothing(hidden: int) -> None:
    """``T == 0``: ``dX`` is empty; the parameter gradients are zeroed without ``accumulate`` (also caller buffers
    holding other values) and left untouched with it; the workspace needs no bytes and an empty one is accepted;
    PyTorch's deterministic mode raises no alert."""
    _require_hidden(hidden)
    i = _make_inputs(0, hidden=hidden)
    prep = _forward(i)
    empty_ws = pkg.cake_mhc_nondet_train_backward_workspace(0, _device(), hidden=hidden)
    assert empty_ws.numel() == 0
    for workspace in (None, empty_ws):
        got = _nd_backward(i, prep, workspace=workspace)
        assert (
            tuple(got["dX"].shape) == (0, STREAMS, hidden) and got["dX"].dtype == BF16
        )
        assert tuple(got["dfn"].shape) == (MAPPING_ROWS, STREAMS * hidden)
        for k in _PARAMETER:
            assert torch.equal(got[k], torch.zeros_like(got[k])), k
    for accumulate in (False, True):
        buffers = {
            "dfn": torch.full(
                (MAPPING_ROWS, STREAMS * hidden), 3.0, dtype=F32, device=_device()
            ),
            "dscale": torch.full((3,), 5.0, dtype=F32, device=_device()),
            "dbase": torch.full((MAPPING_ROWS,), 7.0, dtype=F32, device=_device()),
        }
        got = _nd_backward(i, prep, accumulate=accumulate, **buffers)
        for k, value in (("dfn", 3.0), ("dscale", 5.0), ("dbase", 7.0)):
            assert got[k] is buffers[k], (k, accumulate)
            expected = value if accumulate else 0.0
            assert bool((got[k] == expected).all()), (k, accumulate)
    # PyTorch's deterministic mode raises no alert at ``T == 0`` either.
    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = _nd_backward(i, prep)
        assert tuple(got["dX"].shape) == (0, STREAMS, hidden)
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])


# The token-owner programs whose sm_107a binding ships as that architecture's own unit
# (csrc/<library>_sm_107a_binding.cu): at hidden 7168 sm_107a launches them in its oversized shared-memory mode (two
# CTAs per SM), unlike the program's sm_100a / sm_103a binding.
OWN_BINDING_UNITS = {("cake_mhc_train_bwd_token__ao22_rae7_pd1_ntr__h7168", "sm_107a")}


def _binding_source(library: str, arch: str) -> str:
    """The binding unit of ``library`` on ``arch``: the architecture's own unit where it ships one
    (:data:`OWN_BINDING_UNITS`), else the program's ``csrc/<library>_binding.cu``."""
    if (library, arch) in OWN_BINDING_UNITS:
        return f"csrc/{library}_{arch}_binding.cu"
    return f"csrc/{library}_binding.cu"


def _own_binding_units(package) -> set:
    """``(library, arch)`` of every ``csrc/<library>_<arch>_binding.cu`` of the package."""
    units = set()
    for path in (package / "csrc").glob("*_binding.cu"):
        stem = path.name[: -len("_binding.cu")]
        for arch in ("sm_100a", "sm_103a", "sm_107a"):
            if stem.endswith("_" + arch):
                units.add((stem[: -len(arch) - 1], arch))
    return units


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record of the package names its ``build.json`` library (schema ``cake-build/1``): one entry
    per (library, architecture) with the kernel and binding sources under ``csrc/`` of this package."""
    cake_build = cake_package(KEY, "cake_build")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
        assert record["library"].startswith("cake_mhc_train_"), name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                _binding_source(record["library"], arch),
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    # an architecture's own binding unit exists exactly where the manifest lists it
    assert _own_binding_units(package) == OWN_BINDING_UNITS, sorted(_own_binding_units(package))


# ----------------------------------------------------------------------------------------------------------------
# The opt-in two-word accumulation (``accumulation="fp32x2"``): hidden 4096 and the Sinkhorn count of record only.
# The mapping pass, the token-owner pass and the column slab with its folds run their two-word programs
# (``loader.ACC_WORDS_PROGRAMS``); the rest of the route and every workspace are those of the default mode.  No bit
# is promised: every output is judged against the naive BF16 tier.

TWO_WORD = "fp32x2"


def _two_word_hidden() -> int:
    return _require_hidden(loader.ACC_WORDS_HIDDEN)


@pytest.mark.parametrize("T", (1, 257, 2049, 4097, 16263))
def test_two_word_backward_matches_fp64(T: int, record_property) -> None:
    """``accumulation="fp32x2"``: every output (the forward entry points and REPEATS backward launches on the default
    and on a poisoned caller workspace) judged against the naive BF16 tier over 32 inputs; the caller workspace's zero
    prefix is zero after every launch."""
    hidden = _two_word_hidden()
    _judge_nd(
        lambda s: {**_make_inputs(T, hidden=hidden, seed=s), "accumulation": TWO_WORD},
        record_property, seed=SEED + 701, prefix="fp32x2 ",
    )  # fmt: skip


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
def test_two_word_autograd_matches_fp64(
    T: int, save_stats: bool, checkpointed: bool, record_property
) -> None:
    """``cake_mhc_nondet_train_prepare(..., accumulation="fp32x2")`` / ``cake_mhc_nondet_train_finish`` around the
    residual branch, plain or inside a non-reentrant checkpoint: ``U``, ``Y``, ``dB`` and the gradients judged against
    the naive BF16 tier over 32 inputs."""
    hidden = _two_word_hidden()

    def case(j: int, s: int):
        i = _make_inputs(T, hidden=hidden, seed=s)
        record: dict = {}
        connection = _connection(
            _nd_prepare, _nd_finish, _branch("residual", i), save_stats=save_stats, record=record,
            accumulation=TWO_WORD,
        )  # fmt: skip
        leaves = _leaves(i)
        if checkpointed:
            Y = checkpoint(connection, *leaves, use_reentrant=False)
        else:
            Y = connection(*leaves)
        (Y.float() * i["dY"].float()).sum().backward()
        c = {"leaves": leaves, "U": record["U"], "B": record["B"], "Y": Y}
        outputs, expected, naive = _connection_case(i, c)
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 731, prefix="fp32x2 ")


@pytest.mark.parametrize("T", (129, 2049, 4097))
def test_two_word_checkpoint_replay(T: int, record_property) -> None:
    """The two-word connection with the statistics saved and replayed in the backward (``save_stats=False``), and
    the two-word statistics replay ``forward_stats``: over 32 inputs, ``U``, ``Y``, ``dB`` and the gradients of each
    connection and the replayed statistics judged against the naive BF16 tier."""
    hidden = _two_word_hidden()

    def case(j: int, s: int):
        i = _make_inputs(T, hidden=hidden, seed=s)
        outputs, expected, naive = {}, {}, {}
        for save_stats in (True, False):
            leaves = _leaves(i)
            U, handle = _nd_prepare(
                *leaves, save_stats=save_stats, accumulation=TWO_WORD
            )
            U.retain_grad()
            B = (U.float() + i["B"].float()).to(BF16)
            Y = _nd_finish(leaves[0], B, handle)
            (Y.float() * i["dY"].float()).sum().backward()
            c = {"leaves": leaves, "U": U, "B": B, "Y": Y}
            for d, part in zip(
                (outputs, expected, naive), _connection_case(i, c), strict=True
            ):
                d.update({f"save_stats={save_stats} {k}": v for k, v in part.items()})
        replay = stats_outputs(
            forward_stats(i["X"], i["fn"], i["scale"], i["base"], accumulation=TWO_WORD)
        )
        e, n = expected_and_naive(i, replay)
        for d, part in zip((outputs, expected, naive), (replay, e, n), strict=True):
            d.update({f"replay {k}": v for k, v in part.items()})
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 711, prefix="fp32x2 ")


@pytest.mark.parametrize("T", (129, 16263))
def test_two_word_workspaces_are_the_default_ones(T: int) -> None:
    """The two-word mode runs in workspaces of exactly ``*_workspace_bytes(T)`` (the default mode's sizes): nothing
    is written past them (guard bytes after each buffer keep their value), the backward workspace's zero prefix is
    zero after the launch, and every output is finite."""
    hidden = _two_word_hidden()
    guard = 4096
    i = {**_make_inputs(T, hidden=hidden, seed=SEED + 717), "accumulation": TWO_WORD}
    sizes = {
        "fwd": pkg.cake_mhc_nondet_train_forward_workspace_bytes(
            T, _device(), hidden=hidden
        ),
        "bwd": pkg.cake_mhc_nondet_train_backward_workspace_bytes(
            T, _device(), hidden=hidden
        ),
    }
    buffers = {
        k: torch.zeros(n + guard, dtype=torch.uint8, device=_device())
        for k, n in sizes.items()
    }
    for k, n in sizes.items():
        buffers[k][n:].fill_(0xA5)
    fwd = _forward_all(i, workspace=buffers["fwd"][: sizes["fwd"]])
    out = _nd_backward(i, fwd, workspace=buffers["bwd"][: sizes["bwd"]])
    torch.cuda.synchronize()
    for k, n in sizes.items():
        tail = buffers[k][n:]
        assert torch.equal(tail, torch.full_like(tail, 0xA5)), (
            f"the two-word mode wrote past the {k} workspace"
        )
    _assert_zero_prefix(buffers["bwd"][: sizes["bwd"]], f"two-word T={T}")
    for k, v in {**fwd, **out}.items():
        assert torch.isfinite(v.float()).all(), k


def test_two_word_launches_the_two_word_programs(monkeypatch) -> None:
    """With fresh plan caches, a default backward and then a two-word backward at the same token count: the two-word
    call loads exactly its selection (``selected_modules(..., accumulation="fp32x2")``, in launch order; the plan cache
    is keyed by the mode), whose records of the two-word roles are the programs of ``ACC_WORDS_PROGRAMS`` (the
    group-row form where ``bwd_group_rows`` holds) and whose other records are the default selection's; the two-word
    forward loads the two-word mapping pass."""
    hidden = _two_word_hidden()
    monkeypatch.setattr(pkg, "_PLANS", {})
    monkeypatch.setattr(pkg, "_FAST_ROUTES", {})
    loaded: list[str] = []
    load = loader.load

    def recording_load(name: str, arch: str):
        loaded.append(name)
        return load(name, arch)

    monkeypatch.setattr(loader, "load", recording_load)
    arch, sm = _arch(), _sm_count()
    for T in (2049, 16263):
        i = _make_inputs(T, hidden=hidden, seed=SEED + 719)
        prep = _forward(i)
        _nd_backward(i, prep)
        default = loader.selected_modules(
            arch, hidden, "bwd_finish", rows=T, sm_count=sm
        )
        two = loader.selected_modules(
            arch, hidden, "bwd_finish", rows=T, sm_count=sm, accumulation=TWO_WORD
        )
        rows = loader.bwd_group_rows(T, sm, arch, hidden)
        assert len(two) == len(default), (default, two)
        for one, pair in zip(default, two, strict=True):
            role = loader.MODULES[one]["role"]
            if role in loader.ACC_WORDS_ROLES:
                key = (
                    f"{role}/rows"
                    if rows and f"{role}/rows" in loader.ACC_WORDS_PROGRAMS
                    else role
                )
                assert pair == loader.ACC_WORDS_PROGRAMS[key][arch], (T, role, pair)
                assert pair not in default and arch in loader.MODULES[pair]["arches"], (
                    T,
                    pair,
                )
            else:
                assert pair == one, (T, one, pair)
        pi = {**i, "accumulation": TWO_WORD}
        start = len(loaded)
        pprep = _forward(pi)
        assert loader.ACC_WORDS_PROGRAMS["fwd_map"][arch] in loaded[start:], loaded[
            start:
        ]
        start = len(loaded)
        out = _nd_backward(pi, pprep)
        assert loaded[start:] == list(two), (T, loaded[start:], two)
        assert all(torch.isfinite(v).all() for v in out.values())


def test_two_word_mode_is_refused_where_it_has_no_program() -> None:
    """``accumulation="fp32x2"`` at another hidden size or Sinkhorn count is a ``NotImplementedError`` on every entry
    point that takes the mode; a value other than ``"fp32"`` / ``"fp32x2"`` is a ``ValueError``."""
    for hidden in _hidden_sizes():
        i = _make_inputs(8, hidden=hidden, seed=SEED + 723)
        args = (i["X"], i["fn"], i["scale"], i["base"])
        prep = _forward(i)
        bad = [{"accumulation": TWO_WORD}] if hidden != loader.ACC_WORDS_HIDDEN else []
        bad.append({"accumulation": TWO_WORD, "sinkhorn_iters": 7})
        for kwargs in bad:
            with pytest.raises(NotImplementedError, match="fp32x2"):
                forward_prepare(*args, **kwargs)
            with pytest.raises(NotImplementedError, match="fp32x2"):
                forward_stats(*args, **kwargs)
            with pytest.raises(NotImplementedError, match="fp32x2"):
                _nd_backward(i, prep, **kwargs)
            with pytest.raises(NotImplementedError, match="fp32x2"):
                _nd_prepare(*args, **kwargs)
        for value in ("fp64", "FP32X2", "fp32x4", None, 2):
            with pytest.raises(ValueError, match="accumulation"):
                forward_prepare(*args, accumulation=value)
            with pytest.raises(ValueError, match="accumulation"):
                _nd_backward(i, prep, accumulation=value)
            with pytest.raises(ValueError, match="accumulation"):
                _nd_prepare(*args, accumulation=value)
