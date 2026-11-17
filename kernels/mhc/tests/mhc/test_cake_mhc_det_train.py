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

Tests for the Cake four-stream mHC training forward / backward (hidden 4096,
5120 and 7168, 4 streams; SM100 / SM103 / SM107 at the three hidden sizes).

Every case below runs at each hidden size ``H`` the device's architecture
exports (``cake_jit.supported_hidden_sizes``); a size it does not export is
skipped with the reason.  Accuracy is judged with ``tests/cake_naive.py``:
every floating output of the package (BF16 ``U``, ``Y``, ``dB``, ``dX``; the
FP32 statistics ``r``, ``z``, ``pre``, ``post``, ``C`` of ``stats``; the FP32
gradient statistics ``dpost``, ``dC`` of ``gstats``; FP32 ``dfn``, ``dscale``,
``dbase``) is compared with a pure-torch float64 oracle of the function
(``tests/cake/test_helpers/cake_mhc_naive.py``: autograd through the FP64
forward, the BF16 outputs taken before their cast), and its error, in the L2
and max norms over 32 independent inputs of the case, must be at most 1.005
times the error of the naive BF16 tier (the same function in PyTorch BF16 on
the same inputs), in the median and the maximum over the draws, and 0 on every
draw where the tier's error is 0.  Every output is also held to its contract
on every draw (shape; finite where the oracle is finite).  The judged cases:

* token counts 1, 2, 3, 7, 63, 64, 65, 127, 128, 129, 255, 1000, 2049, 16263,
  20001 and the 13 performance rows (1024 ... 65536); ``T = 0`` and a zero row
  of ``X`` (the ``rms_eps`` path);
* saturated sigmoid tails (``scale z + base >= 17`` on every token), where the
  tail gradients of ``dbase`` / ``dscale`` / ``dfn`` are also finite and
  non-zero on every draw;
* single-cast cancellation (inputs whose collapse and residual-mixing
  contributions to ``dX`` cancel on one stream; host-drawn, numpy PCG64, the
  same inputs on every device), every draw holding elements inside the band;
* ``fn x 1e3`` (``|z| ~ 1e3``; host-drawn), ``dX`` also over 64 draws;
* non-default ``rms_eps`` / ``hc_eps`` (the oracle at the same values; the
  statistics actually move), four consecutive ``accumulate=True`` backward
  passes (the oracle: the carried-in values plus the four FP64 gradients; the
  tier: the same sums in BF16), the autograd wrappers, and the hidden sizes
  mixed in one process.

The exact checks stay bitwise: three launches on the same inputs (outputs and
workspaces re-poisoned) give the same bytes, the run-to-run contract of a
deterministic package; the checkpoint replay (``forward_stats``,
``cake_mhc_det_train_prepare(save_stats=False)``) gives the statistics and
gradients of the saved path; repeated backward passes over a retained graph,
interleaved connections, checkpoints and the hidden sizes mixed in one process
give the bits of the plain run; the autograd wrappers keep nothing of a
connection alive once its backward has run; the zero pad of ``gstats``; the
workspace helpers, the argument checks and the module cache (no rebuild or
reload when the token count, the epsilons or the hidden size change).

The oracle and the naive tier evaluate the largest cases in row blocks of
8192 tokens (``BLOCK_TOKENS`` of ``tests/cake/test_helpers/cake_mhc_naive.py``,
as the reference they port does): the per-token values are concatenated and
the parameter gradients are added in block order.
This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
``cake_pplx``, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``cake_pplx.cake_mhc_det_train`` facade is covered by ``test_cake_mhc_det_train_api.py``; the host rules by
``test_cake_mhc_det_train_rules.py``.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.cake.test_helpers.cake_mhc_naive import (
    PARAMETERS,
    expected_and_naive,
    judge_draws,
    judge_package,
    package_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import DRAWS

pkg = cake_package("cake.cake_mhc_det_train")
cake_build = cake_package("cake.cake_mhc_det_train", "cake_build")
loader = cake_package("cake.cake_mhc_det_train", "cake_jit")
forward_prepare = pkg.cake_mhc_det_train_forward_prepare
forward_stats = pkg.cake_mhc_det_train_forward_stats
forward_finish = pkg.cake_mhc_det_train_forward_finish
backward_prepare = pkg.cake_mhc_det_train_backward_prepare
backward_finish = pkg.cake_mhc_det_train_backward_finish
forward_workspace = pkg.cake_mhc_det_train_forward_workspace
backward_workspace = pkg.cake_mhc_det_train_backward_workspace
forward_workspace_bytes = pkg.cake_mhc_det_train_forward_workspace_bytes
backward_workspace_bytes = pkg.cake_mhc_det_train_backward_workspace_bytes
is_supported = pkg.is_cake_mhc_det_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, MAPPING_ROWS = 4, 24
# Every hidden size of the package; each test that builds inputs runs at the sizes the device's architecture exports.
HIDDEN_SIZES = tuple(loader.HIDDEN_SIZES)
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
T_CORRECTNESS = (
    0,
    1,
    2,
    3,
    7,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    1000,
    2049,
    16263,
    20001,
)
T_PERFORMANCE = (
    16263,
    14928,
    15288,
    15566,
    15590,
    16114,
    16132,
    16164,
    1024,
    4096,
    8192,
    32768,
    65536,
)
T_CHECKED = tuple(sorted((set(T_CORRECTNESS) | set(T_PERFORMANCE)) - {0}))
# The cancellation band of the cancellation inputs: |dX| <= CANCEL_FACTOR x the largest of its three contributions.
CANCEL_FACTOR = 2.0**-6
BLOCK_TOKENS = 8192
SEED = 2032
# The draws of the fn x 1e3 dX case (test_large_logits_dx_over_draws); every other judged case takes DRAWS (32).
LARGE_LOGITS_DRAWS = 64
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before the first test, through the spec
    # objects the package's loader itself builds on first use; a prebuilt or cached library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _arch() -> str | None:
    return loader.arch_for_capability(_capability())


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture exports every route of."""
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


def _require_hidden(hidden: int) -> int:
    if hidden not in _hidden_sizes():
        pytest.skip(
            f"hidden {hidden} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return hidden


def _mixed_hidden_sizes() -> tuple[int, ...]:
    """Every exported hidden size, for the tests that mix sizes in one process (4096 and 7168 at least)."""
    sizes = _hidden_sizes()
    if not {4096, 7168} <= set(sizes):
        pytest.skip(
            f"mixing hidden sizes needs 4096 and 7168 on one device; {_arch()} exports {sizes}"
        )
    return sizes


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a test's inputs (every size of the package; skipped where the architecture has no route)."""
    return _require_hidden(request.param)


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


def _sinkhorn(logits: torch.Tensor, hc_eps: float = HC_EPS) -> torch.Tensor:
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(SINKHORN_ITERS - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _coefficients(z, scale, base, hc_eps: float = HC_EPS):
    T = z.shape[0]
    pre = torch.sigmoid(z[:, 0:4] * scale[0] + base[0:4]) + hc_eps
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = _sinkhorn((z[:, 8:] * scale[2] + base[8:]).reshape(T, STREAMS, STREAMS), hc_eps)
    return pre, post, C


def _make_inputs(
    T: int,
    *,
    hidden: int,
    mode: str = "unit",
    seed: int = SEED,
    fn_scale: float = 1.0,
    zero_rows=(),
    host_draw: bool = False,
) -> dict:
    """The inputs of one case: ``X, B, dY, dU ~ N(0, 1)`` BF16; ``fn ~ 1e-2 N(0, 1)`` (``1e-4`` in the ``tail`` and
    ``cancel`` regimes); ``scale ~ 1 + 0.1 N``; ``base ~ 0.1 N`` (``base[:8] = 20`` in ``tail``, so every pre / post
    logit is >= 17); in ``cancel`` ``dU := BF16(-(sum_j C_0j dY_j) / pre_0)`` from the FP64 statistics, so the
    collapse and residual-mixing contributions of stream 0 to ``dX`` cancel.  ``X`` / ``dY`` are ``[T, 4, hidden]``,
    ``B`` / ``dU`` ``[T, hidden]`` and ``fn`` ``[24, 4 hidden]``.  ``host_draw`` draws the normals on the host from
    numpy's PCG64 (FP64, rounded to FP32), builds the draw on the CPU and then moves it to the device: the same inputs on
    every device and host architecture (the device generator's stream depends on the device's SM count, and torch's
    CPU generator differs between x86_64 and aarch64)."""
    n = STREAMS * hidden
    if host_draw:
        rng = np.random.Generator(np.random.PCG64(seed))
    else:
        g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        if host_draw:
            return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(
                dtype
            )
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    fs = {"unit": 1e-2, "tail": 1e-4, "cancel": 1e-4}[mode]
    X = randn(T, STREAMS, hidden, dtype=BF16)
    fn = randn(MAPPING_ROWS, n) * fs
    scale = 1.0 + 0.1 * randn(3)
    base = 0.1 * randn(MAPPING_ROWS)
    if mode == "tail":
        base[:8] = 20.0
    B = randn(T, hidden, dtype=BF16)
    dY = randn(T, STREAMS, hidden, dtype=BF16)
    dU = randn(T, hidden, dtype=BF16)
    if fn_scale != 1.0:
        fn = fn * float(fn_scale)
    if zero_rows:
        X[list(zero_rows)] = 0
    fn = fn.contiguous()
    if host_draw:
        X, fn, scale, base, B, dY, dU = (
            t.to(_device()) for t in (X, fn, scale, base, B, dY, dU)
        )
    if mode == "cancel" and T > 0:
        with torch.no_grad():
            dU_new = torch.empty_like(dU)
            for lo in range(0, T, BLOCK_TOKENS):
                hi = min(T, lo + BLOCK_TOKENS)
                u = X[lo:hi].double().reshape(hi - lo, n)
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
        "T": T,
        "hidden": hidden,
    }


def _run(i: dict, *, fwd_ws=None, bwd_ws=None) -> dict:
    """The package end to end: forward prepare / finish, backward prepare / finish on the inputs of a case (with the
    case's epsilons and accumulation mode when it carries non-default ones)."""
    eps = {k: i[k] for k in ("rms_eps", "hc_eps", "accumulation") if k in i}
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws, **eps
    )
    Y = forward_finish(i["X"], i["B"], stats)
    dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    dX, dfn, dscale, dbase = backward_finish(
        i["X"],
        i["dY"],
        i["dU"],
        i["B"],
        i["fn"],
        i["scale"],
        i["base"],
        stats,
        gstats,
        workspace=bwd_ws,
        **eps,
    )
    return {
        "U": U,
        "stats": stats,
        "Y": Y,
        "dB": dB,
        "gstats": gstats,
        "dX": dX,
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
    }


# ----------------------------------------------------------------------------------------------------------------
# Gates: every floating output judged against the naive BF16 tier over DRAWS inputs (tests/cake_naive.py), beside the
# exact checks of a case


def _assert_formats(i: dict, res: dict, label: str) -> None:
    """The formats of a run's outputs (``res`` of ``_run``): BF16 ``U`` / ``Y`` / ``dB`` / ``dX`` and FP32 ``stats``
    ``[T, 64]``, ``gstats`` ``[T, 32]``, ``dfn`` / ``dscale`` / ``dbase``, all contiguous; the pad columns of
    ``gstats`` exact zeros."""
    T = i["X"].shape[0]
    for k in ("U", "Y", "dB", "dX"):
        assert res[k].dtype == BF16 and res[k].is_contiguous(), f"{k} {label}"
    for k in ("stats", "gstats", *PARAMETERS):
        assert res[k].dtype == F32 and res[k].is_contiguous(), f"{k} {label}"
    assert tuple(res["stats"].shape) == (T, 64) and tuple(res["gstats"].shape) == (
        T,
        32,
    ), label
    assert torch.equal(
        res["gstats"][:, 20:], torch.zeros_like(res["gstats"][:, 20:])
    ), f"gstats pad {label}"


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


def _label(T: int, hidden: int, **extra) -> str:
    return f"T={T} H={hidden}" + "".join(
        f" {k}={v}" for k, v in extra.items() if v is not None
    )


def _judge_case(
    T: int,
    *,
    hidden: int,
    record_property,
    mode: str = "unit",
    fn_scale: float = 1.0,
    zero_rows=(),
    seed: int = SEED,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
    host_draw: bool = False,
    names=None,
    draws: int = DRAWS,
    prefix: str = "",
    accumulation: str | None = None,
) -> None:
    """Every output of the package run end to end (``U``, ``Y``, ``dB``, ``dX``, the statistics ``r, z, pre, post,
    C``, the gradient statistics ``dpost, dC``, ``dfn``, ``dscale``, ``dbase``), or the outputs ``names``, judged
    against the naive BF16 tier over ``draws`` inputs of the case (draw ``j`` of seed ``draw_seed(seed, j)``); every
    draw also holds the case's exact checks (formats, the zero pad of ``gstats``, the tail gradients non-zero in the
    ``tail`` regime, the band exercised in the ``cancel`` regime)."""
    label = _label(
        T, hidden, mode=mode if mode != "unit" else None, fn_scale=fn_scale if fn_scale != 1.0 else None,
        zero_rows=tuple(zero_rows) or None, rms_eps=rms_eps if rms_eps != RMS_EPS else None,
        hc_eps=hc_eps if hc_eps != HC_EPS else None, accumulation=accumulation,
    )  # fmt: skip

    def make(s: int) -> dict:
        i = _make_inputs(
            T,
            hidden=hidden,
            mode=mode,
            seed=s,
            fn_scale=fn_scale,
            zero_rows=zero_rows,
            host_draw=host_draw,
        )
        i["rms_eps"], i["hc_eps"] = rms_eps, hc_eps
        if accumulation is not None:
            i["accumulation"] = accumulation
        return i

    def check(i: dict, results: list, expected: dict) -> None:
        for res in results:
            _assert_formats(i, res, label)
            if mode == "tail":
                _assert_tail_gradients(res, expected, label)
        if mode == "cancel":
            _assert_cancellation_inputs(i, expected, label)

    judge_package(
        make,
        _run,
        record_property,
        seed=seed,
        names=names,
        draws=draws,
        prefix=prefix,
        check=check,
    )


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("T", T_CHECKED)
def test_forward_backward_match_fp64(T: int, hidden: int, record_property) -> None:
    """The 15 non-zero correctness token counts and the 13 performance rows (T = 0 is ``test_zero_rows``): every
    output judged against the naive BF16 tier over 32 inputs."""
    _judge_case(T, hidden=hidden, record_property=record_property)


def test_zero_row_of_x_takes_the_eps_path(hidden: int, record_property) -> None:
    _judge_case(7, hidden=hidden, zero_rows=(3,), record_property=record_property)


@pytest.mark.parametrize("T", (257, 2049))
def test_saturated_sigmoid_tails(T: int, hidden: int, record_property) -> None:
    """Saturated sigmoid tails (``base[:8] = 20``): every output judged against the naive BF16 tier over 32 inputs;
    on every draw the tail gradients are finite and non-zero."""
    _judge_case(T, hidden=hidden, mode="tail", record_property=record_property)


@pytest.mark.parametrize("T", (257, 2049))
def test_single_cast_cancellation(T: int, hidden: int, record_property) -> None:
    """Inputs whose collapse and residual-mixing contributions to ``dX`` cancel on stream 0 (host-drawn, the same on
    every device): every output, ``dX`` included, judged against the naive BF16 tier over 32 inputs; every draw holds
    elements inside the cancellation band."""
    _judge_case(
        T, hidden=hidden, mode="cancel", host_draw=True, record_property=record_property
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_large_logits(T: int, hidden: int, record_property) -> None:
    """``fn x 1e3`` (|z| ~ 1e3: saturated sigmoids and softmax rows; host-drawn inputs): every output judged against
    the naive BF16 tier over 32 inputs."""
    _judge_case(
        T, hidden=hidden, fn_scale=1e3, host_draw=True, record_property=record_property
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_large_logits_dx_over_draws(T: int, hidden: int, record_property) -> None:
    """BF16 ``dX`` at ``fn x 1e3`` over LARGE_LOGITS_DRAWS (64) host-drawn inputs, judged against the naive BF16
    tier."""
    _judge_case(
        T, hidden=hidden, fn_scale=1e3, host_draw=True, names=("dX",), draws=LARGE_LOGITS_DRAWS,
        record_property=record_property,
    )  # fmt: skip


def test_zero_rows(hidden: int) -> None:
    i = _make_inputs(0, hidden=hidden)
    res = _run(i)
    assert tuple(res["U"].shape) == (0, hidden) and tuple(res["stats"].shape) == (0, 64)
    assert tuple(res["Y"].shape) == (0, STREAMS, hidden) and tuple(res["dB"].shape) == (
        0,
        hidden,
    )
    assert tuple(res["gstats"].shape) == (0, 32) and tuple(res["dX"].shape) == (
        0,
        STREAMS,
        hidden,
    )
    assert tuple(res["dfn"].shape) == (MAPPING_ROWS, STREAMS * hidden)
    assert torch.equal(res["dfn"], torch.zeros_like(res["dfn"]))
    assert torch.equal(res["dscale"], torch.zeros_like(res["dscale"]))
    assert torch.equal(res["dbase"], torch.zeros_like(res["dbase"]))
    assert forward_stats(i["X"], i["fn"], i["scale"], i["base"]).shape == (0, 64)
    # accumulate=True leaves the caller's gradients untouched.
    dfn = torch.full((MAPPING_ROWS, STREAMS * hidden), 3.0, dtype=F32, device=_device())
    dscale = torch.full((3,), 5.0, dtype=F32, device=_device())
    dbase = torch.full((MAPPING_ROWS,), 7.0, dtype=F32, device=_device())
    out = backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], res["stats"], res["gstats"],
        accumulate=True, dfn=dfn, dscale=dscale, dbase=dbase,
    )  # fmt: skip
    assert out[1] is dfn and out[2] is dscale and out[3] is dbase
    assert (
        float(dfn.min()) == 3.0
        and float(dscale.min()) == 5.0
        and float(dbase.min()) == 7.0
    )
    assert (
        forward_workspace_bytes(0, _device(), hidden=hidden) == 0
        and backward_workspace_bytes(0, _device(), hidden=hidden) == 0
    )


@pytest.mark.parametrize("T", (129, 2049))
def test_checkpoint_replay_is_bitwise(T: int, hidden: int) -> None:
    i = _make_inputs(T, hidden=hidden, seed=SEED + 11)
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
    replay = forward_stats(i["X"], i["fn"], i["scale"], i["base"])
    assert torch.equal(stats, replay), (
        "forward_stats is not bitwise the forward_prepare record"
    )
    grads = []
    for save_stats in (True, False):
        X = i["X"].clone().requires_grad_(True)
        fn = i["fn"].clone().requires_grad_(True)
        scale = i["scale"].clone().requires_grad_(True)
        base = i["base"].clone().requires_grad_(True)
        U, handle = pkg.cake_mhc_det_train_prepare(
            X, fn, scale, base, save_stats=save_stats
        )
        B = (U.float() + i["B"].float()).to(BF16)
        Y = pkg.cake_mhc_det_train_finish(X, B, handle)
        (Y.float() * i["dY"].float()).sum().backward()
        grads.append(
            (X.grad.clone(), fn.grad.clone(), scale.grad.clone(), base.grad.clone())
        )
    for saved, recomputed in zip(*grads, strict=False):
        assert torch.equal(saved, recomputed), (
            "checkpoint replay changed a gradient bit"
        )


@pytest.mark.parametrize("T", (129, 1024, 2049, 4097))
@pytest.mark.parametrize("bump", (False, True))
def test_checkpoint_carry(T: int, bump: bool, hidden: int, monkeypatch) -> None:
    """The checkpoint carry (``save_stats=False`` inside a band of ``cake_jit.FN_SPLIT_CARRY``): the replay's dual-layout
    split hands the transposed split to the prepare node's backward on a fourth edge, whose ``bwd_finish`` then runs
    without its split; the gradients are bitwise those of the saved-statistics backward.  An in-place update of ``fn``
    between the replay and the prepare node's backward (``bump``: a version bump that keeps the values) makes the
    prepare node's backward split ``fn`` again."""
    i = _make_inputs(T, hidden=hidden, seed=SEED + 13)
    carry = loader.fn_split_carry(T, _arch(), hidden) is not None
    carried: list[bool] = []
    finish = pkg._backward_finish

    def spy(*args, **kwargs):
        carried.append(kwargs.get("fnTp") is not None)
        return finish(*args, **kwargs)

    monkeypatch.setattr(pkg, "_backward_finish", spy)
    grads, edges = [], []
    for save_stats in (True, False):
        leaves = [
            i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")
        ]
        U, handle = pkg.cake_mhc_det_train_prepare(*leaves, save_stats=save_stats)
        edges.append(len(handle.edges))
        B = (U.float() + i["B"].float()).to(BF16)
        if bump:

            def bump_fn(grad, fn=leaves[1]):
                with torch.no_grad():
                    fn.mul_(1.0)
                return grad

            B.register_hook(bump_fn)
        Y = pkg.cake_mhc_det_train_finish(leaves[0], B, handle)
        (Y.float() * i["dY"].float()).sum().backward()
        grads.append(tuple(leaf.grad.clone() for leaf in leaves))
    assert edges == [2, 4 if carry else 3], edges
    assert carried == [False, carry and not bump], carried
    for saved, replayed in zip(*grads, strict=True):
        assert torch.equal(saved, replayed), (
            "the checkpoint carry changed a gradient bit"
        )


def _live_bytes(*, baseline: bool = False) -> int | None:
    """Bytes the live CUDA tensors asked the caching allocator for (its ``requested_bytes``), not the bytes of the
    blocks they occupy (``torch.cuda.memory_allocated``): a tensor placed in a reused free block up to 1 MiB larger
    than itself (the allocator does not split a large block for a remainder of at most 1 MiB) would otherwise count
    that slack as memory kept alive (on torch 2.15, 186880 B on B200 and 1047040 B on R200 for the checkpointed
    output below).  ``None`` under another allocator backend (``cudaMallocAsync`` keeps
    no ``requested_bytes``), where the retention check is skipped and the gradient checks still run.  The baseline
    (``baseline=True``, before the call) first collects unreachable reference cycles, so tensors held only by earlier
    garbage (for example the frames of a failed test kept by pytest's exception record) neither count in it nor vanish
    from it during the call; the measurement after the call never collects, so cyclic garbage the call creates counts
    as kept."""
    if baseline:
        gc.collect()
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


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_keeps_nothing_alive(T: int, save_stats: bool, hidden: int) -> None:
    """A non-reentrant ``torch.utils.checkpoint`` around prepare / finish keeps no tensor of the layer alive after
    the forward (``stats`` and the inputs travel through the saved tensors; the handle holds no branch output), and
    the checkpointed backward is bitwise the plain one."""
    from torch.utils.checkpoint import checkpoint

    i = _make_inputs(T, hidden=hidden, seed=SEED + 41)

    def connection(X, fn, scale, base):
        U, handle = pkg.cake_mhc_det_train_prepare(
            X, fn, scale, base, save_stats=save_stats
        )
        B = (U.float() + i["B"].float()).to(BF16)
        return pkg.cake_mhc_det_train_finish(X, B, handle)

    def leaves() -> list:
        return [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]

    plain = leaves()
    (connection(*plain).float() * i["dY"].float()).sum().backward()

    checkpointed = leaves()
    before = _live_bytes(baseline=True)
    Y = checkpoint(connection, *checkpointed, use_reentrant=False)
    _assert_nothing_kept(before, Y, "forward")
    (Y.float() * i["dY"].float()).sum().backward()
    for p, c in zip(plain, checkpointed, strict=True):
        assert torch.equal(p.grad, c.grad), "checkpointing changed a gradient bit"


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_after_prepare_replays_statistics(
    T: int, save_stats: bool, hidden: int
) -> None:
    """A non-reentrant ``torch.utils.checkpoint`` region that starts after ``cake_mhc_det_train_prepare`` (the branch and
    the finish inside it, the prepare outside) keeps nothing beyond its output after the forward; its backward reruns
    the finish on a handle that already released the statistics, which the finish forward replays bitwise, so the
    gradients equal the plain run bit for bit."""
    from torch.utils.checkpoint import checkpoint

    i = _make_inputs(T, hidden=hidden, seed=SEED + 42)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return pkg.cake_mhc_det_train_finish(X, B, handle)

    def leaves() -> list:
        return [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]

    plain = leaves()
    U, handle = pkg.cake_mhc_det_train_prepare(*plain, save_stats=save_stats)
    (finish(plain[0], U, handle).float() * i["dY"].float()).sum().backward()

    checkpointed = leaves()
    U, handle = pkg.cake_mhc_det_train_prepare(*checkpointed, save_stats=save_stats)
    # pinned: save_stats=False releases it in the finish forward
    stats = handle.state.stats
    before = _live_bytes(baseline=True)
    Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=False)
    _assert_nothing_kept(before, Y, "finish")
    # The finish forward takes the prepare node's edge outputs off the handle.
    assert handle.edges == () and handle.state.stats is None, (
        "the handle kept the edges or the statistics after the forward"
    )
    del stats
    (Y.float() * i["dY"].float()).sum().backward()
    for p, c in zip(plain, checkpointed, strict=True):
        assert torch.equal(p.grad, c.grad), "checkpointed finish: gradient bit changed"


def _connection_leaves(i: dict) -> list:
    return [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]


def _two_layers(
    X, params: list, *, save_stats: bool, checkpointed: bool
) -> torch.Tensor:
    """Two layers of two connections each (the branch hands ``U`` on as ``B``), each layer a non-reentrant checkpoint
    region or not."""
    from torch.utils.checkpoint import checkpoint

    def connection(x, fn, scale, base):
        U, handle = pkg.cake_mhc_det_train_prepare(
            x, fn, scale, base, save_stats=save_stats
        )
        return pkg.cake_mhc_det_train_finish(x, U, handle)

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
    T: int, save_stats: bool, checkpointed: bool, hidden: int
) -> None:
    """Once a connection's backward has run, nothing of it stays alive: after ``torch.autograd.grad`` over two layers
    of two connections the only live tensor of the step is its output ``Y`` (which keeps the graph alive), since the
    finish node's backward hands ``dY`` and the gradient statistics to the prepare node's backward on autograd edges.
    A second step gives bitwise the gradients of the first."""
    i = _make_inputs(T, hidden=hidden, seed=SEED + 51)
    params = [
        [p.clone().requires_grad_(True) for p in (q["fn"], q["scale"], q["base"])]
        for q in (_make_inputs(1, hidden=hidden, seed=SEED + 52 + k) for k in range(4))
    ]
    X = i["X"].clone().requires_grad_(True)
    leaves = [X, *[p for ps in params for p in ps]]

    def step() -> tuple:
        Y = _two_layers(X, params, save_stats=save_stats, checkpointed=checkpointed)
        return Y, torch.autograd.grad(Y, leaves, i["dY"])

    reference = step()[1]  # also the warm-up: default workspaces and loaded modules
    before = _live_bytes(baseline=True)
    Y, grads = step()
    assert all(torch.equal(g, r) for g, r in zip(grads, reference, strict=True)), (
        "a second step changed a gradient bit"
    )
    del grads
    _assert_kept(before, "the backward", Y)


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_repeated_backward_with_retain_graph(
    T: int, save_stats: bool, hidden: int
) -> None:
    """Backward passes over one retained graph are independent: two full passes give bitwise the gradients of a fresh
    graph, and a pass in between that reaches the prepare node but not the finish node (the loss is ``U``) takes no
    ``dY`` of the earlier pass: it equals the same pass on a fresh graph."""
    i = _make_inputs(T, hidden=hidden, seed=SEED + 56)

    def graph(leaves: list) -> tuple:
        U, handle = pkg.cake_mhc_det_train_prepare(*leaves, save_stats=save_stats)
        B = (U.float() + i["B"].float()).to(BF16)
        return U, pkg.cake_mhc_det_train_finish(leaves[0], B, handle)

    leaves = _connection_leaves(i)
    U, Y = graph(leaves)
    first = torch.autograd.grad(Y, leaves, i["dY"], retain_graph=True)
    u_only = torch.autograd.grad(U, leaves, i["dU"], retain_graph=True)
    second = torch.autograd.grad(Y, leaves, i["dY"])

    fresh = _connection_leaves(i)
    full = torch.autograd.grad(graph(fresh)[1], fresh, i["dY"])
    fresh = _connection_leaves(i)
    u_fresh = torch.autograd.grad(graph(fresh)[0], fresh, i["dU"])
    for a, b, f in zip(first, second, full, strict=True):
        assert torch.equal(a, f) and torch.equal(b, f), (
            "retain_graph: gradient bit changed"
        )
    for a, f in zip(u_only, u_fresh, strict=True):
        assert torch.equal(a, f), (
            "a backward that skipped the finish node took a stale dY"
        )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("region", ("connection", "after_prepare"))
def test_reentrant_checkpoint(
    T: int, save_stats: bool, region: str, hidden: int
) -> None:
    """A reentrant ``torch.utils.checkpoint`` around the connection, or around the branch and the finish with the
    prepare outside (the rerun finish node has no edge to the prepare node, so its backward leaves ``dY`` and the
    gradient statistics on the shared state and the prepare node's backward empties it), gives bitwise the plain
    gradients and keeps nothing alive after the backward beyond ``Y`` and the leaves' gradients."""
    from torch.utils.checkpoint import checkpoint

    i = _make_inputs(T, hidden=hidden, seed=SEED + 57)

    def finish(X, U, handle):
        B = (U.float() + i["B"].float()).to(BF16)
        return pkg.cake_mhc_det_train_finish(X, B, handle)

    def connection(X, fn, scale, base):
        U, handle = pkg.cake_mhc_det_train_prepare(
            X, fn, scale, base, save_stats=save_stats
        )
        return finish(X, U, handle)

    plain = _connection_leaves(i)
    connection(*plain).backward(i["dY"])

    checkpointed = _connection_leaves(i)
    before = _live_bytes(baseline=True)
    if region == "connection":
        Y = checkpoint(connection, *checkpointed, use_reentrant=True)
    else:
        U, handle = pkg.cake_mhc_det_train_prepare(*checkpointed, save_stats=save_stats)
        Y = checkpoint(finish, checkpointed[0], U, handle, use_reentrant=True)
        del U
    Y.backward(i["dY"])
    for p, c in zip(plain, checkpointed, strict=True):
        assert torch.equal(p.grad, c.grad), (
            f"reentrant checkpoint ({region}): gradient bit changed"
        )
    if region == "after_prepare":
        assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    _assert_kept(
        before, "a reentrant checkpointed backward", Y, *(c.grad for c in checkpointed)
    )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("prepare", ("skipped", "not_recorded"))
def test_backward_without_the_prepare_node(T: int, prepare: str, hidden: int) -> None:
    """A backward that runs the finish node but not the prepare node (``torch.autograd.grad`` for the branch output
    alone, or a prepare whose inputs need no gradient) leaves the finish node's ``dY`` and gradient statistics on no
    handle: the engine drops the edge gradients.  What stays alive is the prepare node's own saved statistics while
    its backward has not run."""
    i = _make_inputs(T, hidden=hidden, seed=SEED + 58)
    leaves = (
        _connection_leaves(i)
        if prepare == "skipped"
        else [i[k] for k in ("X", "fn", "scale", "base")]
    )

    def step() -> tuple:
        U, handle = pkg.cake_mhc_det_train_prepare(*leaves)
        B = (U.float() + i["B"].float()).to(BF16)
        if prepare == "not_recorded":
            B.requires_grad_(True)
        Y = pkg.cake_mhc_det_train_finish(leaves[0], B, handle)
        return Y, B, handle, torch.autograd.grad(Y, [B], i["dY"])

    step()
    before = _live_bytes(baseline=True)
    Y, B, handle, (dB,) = step()
    assert all(t is None for t in handle.state.take()), "the state kept a gradient"
    saved_stats = (
        T * 256 if prepare == "skipped" else 0
    )  # the 256-byte record per token
    _assert_kept(
        before, "a backward without the prepare node", Y, B, dB, expected=saved_stats
    )


@pytest.mark.parametrize("save_stats", (True, False))
def test_interleaved_connections(save_stats: bool, hidden: int) -> None:
    """Two connections of one shape whose prepares both run before either finish (their edge outputs alias the same
    stand-in buffers but are distinct autograd outputs) give bitwise the gradients of the two connections run one
    after the other."""
    T = 129
    a = _make_inputs(T, hidden=hidden, seed=SEED + 61)
    b = _make_inputs(T, hidden=hidden, seed=SEED + 62)

    def branch(i: dict, U: torch.Tensor) -> torch.Tensor:
        return (U.float() + i["B"].float()).to(BF16)

    def run(interleaved: bool) -> list:
        la, lb = _connection_leaves(a), _connection_leaves(b)
        Ua, ha = pkg.cake_mhc_det_train_prepare(*la, save_stats=save_stats)
        if interleaved:
            Ub, hb = pkg.cake_mhc_det_train_prepare(*lb, save_stats=save_stats)
            Yb = pkg.cake_mhc_det_train_finish(lb[0], branch(b, Ub), hb)
            Ya = pkg.cake_mhc_det_train_finish(la[0], branch(a, Ua), ha)
        else:
            Ya = pkg.cake_mhc_det_train_finish(la[0], branch(a, Ua), ha)
            Ub, hb = pkg.cake_mhc_det_train_prepare(*lb, save_stats=save_stats)
            Yb = pkg.cake_mhc_det_train_finish(lb[0], branch(b, Ub), hb)
        torch.autograd.backward([Ya, Yb], [a["dY"], b["dY"]])
        return [p.grad for p in la + lb]

    for s, t in zip(run(False), run(True), strict=True):
        assert torch.equal(s, t), "interleaved connections: gradient bit changed"


@pytest.mark.parametrize("T", (1000,))
def test_accumulate_four_times(T: int, hidden: int, record_property) -> None:
    """Four consecutive ``accumulate=True`` passes on four input sets: ``dfn``, ``dscale``, ``dbase`` judged against
    the naive BF16 tier over 32 draws of the four-fold chain (the oracle: the carried-in values plus the four FP64
    gradients; the tier: the carried-in values in BF16 plus the four BF16 gradients, added in BF16)."""

    def case(j: int, s: int):
        g = torch.Generator(device=_device()).manual_seed(s)
        carried = {
            "dfn": torch.randn(
                (MAPPING_ROWS, STREAMS * hidden), generator=g, device=_device()
            )
            * 1e-3,
            "dscale": torch.randn((3,), generator=g, device=_device()) * 1e-3,
            "dbase": torch.randn((MAPPING_ROWS,), generator=g, device=_device()) * 1e-3,
        }
        acc = {k: v.clone() for k, v in carried.items()}
        want = {k: v.double() for k, v in carried.items()}
        tier = {k: v.to(BF16) for k, v in carried.items()}
        for step in range(4):
            i = _make_inputs(T, hidden=hidden, seed=s + 100 + step)
            _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
            _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
            out = backward_finish(
                i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
                accumulate=True, dfn=acc["dfn"], dscale=acc["dscale"], dbase=acc["dbase"],
            )  # fmt: skip
            assert (
                out[1] is acc["dfn"]
                and out[2] is acc["dscale"]
                and out[3] is acc["dbase"]
            )
            expected, naive = expected_and_naive(i, PARAMETERS)
            for k in PARAMETERS:
                want[k] = want[k] + expected[k]
                tier[k] = tier[k] + naive[k]
            del i, expected, naive
        return [acc], want, tier

    judge_draws(case, record_property, seed=SEED + 7919)


@pytest.mark.parametrize("T", (129, 16263))
def test_outputs_are_deterministic(T: int, hidden: int) -> None:
    """Three launches of the whole forward / backward on the same inputs, outputs and the non-ticket workspace
    regions re-poisoned with NaN between them, give bitwise identical outputs."""
    i = _make_inputs(T, hidden=hidden, seed=SEED + 3)
    fwd_ws = forward_workspace(T, _device(), hidden=hidden)
    bwd_ws = backward_workspace(T, _device(), hidden=hidden)
    device_index = _device().index
    layouts = {
        kind: loader.workspace_layout(kind, T, device_index=device_index, hidden=hidden)
        for kind in ("fwd", "bwd")
    }

    def poison(ws: torch.Tensor, kind: str) -> None:
        for name, region in layouts[kind]["regions"].items():
            if name != "ticket":
                ws[region["offset"] : region["offset"] + region["bytes"]].fill_(0xFF)

    results = []
    for _repeat in range(3):
        poison(fwd_ws, "fwd")
        poison(bwd_ws, "bwd")
        U = torch.full((T, hidden), float("nan"), dtype=BF16, device=_device())
        stats = torch.full((T, 64), float("nan"), dtype=F32, device=_device())
        Y = torch.full((T, STREAMS, hidden), float("nan"), dtype=BF16, device=_device())
        dB = torch.full((T, hidden), float("nan"), dtype=BF16, device=_device())
        gstats = torch.full((T, 32), float("nan"), dtype=F32, device=_device())
        dX = torch.full(
            (T, STREAMS, hidden), float("nan"), dtype=BF16, device=_device()
        )
        dfn = torch.full(
            (MAPPING_ROWS, STREAMS * hidden), float("nan"), dtype=F32, device=_device()
        )
        dscale = torch.full((3,), float("nan"), dtype=F32, device=_device())
        dbase = torch.full((MAPPING_ROWS,), float("nan"), dtype=F32, device=_device())
        forward_prepare(
            i["X"],
            i["fn"],
            i["scale"],
            i["base"],
            out=U,
            stats_out=stats,
            workspace=fwd_ws,
        )
        forward_finish(i["X"], i["B"], stats, out=Y)
        backward_prepare(i["dY"], i["B"], i["X"], stats, dB=dB, grad_stats_out=gstats)
        backward_finish(
            i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], stats, gstats,
            workspace=bwd_ws, dX=dX, dfn=dfn, dscale=dscale, dbase=dbase,
        )  # fmt: skip
        torch.cuda.synchronize()
        results.append(
            [t.clone() for t in (U, stats, Y, dB, gstats, dX, dfn, dscale, dbase)]
        )
        for t in results[-1]:
            assert torch.isfinite(t.float()).all()
    for later in results[1:]:
        for first, again in zip(results[0], later, strict=False):
            assert torch.equal(first, again), "a repeated launch changed an output bit"
    fresh = _run(i)
    assert torch.equal(fresh["dfn"], results[0][6]) and torch.equal(
        fresh["dX"], results[0][5]
    )
    # The ticket words self-reset: the caller-held workspaces are reusable without a reset.
    for ws, kind in ((fwd_ws, "fwd"), (bwd_ws, "bwd")):
        region = layouts[kind]["regions"]["ticket"]
        assert (
            torch.count_nonzero(
                ws[region["offset"] : region["offset"] + region["bytes"]]
            )
            == 0
        )


def test_changing_rows_in_one_process_does_not_rebuild(
    hidden: int, record_property
) -> None:
    """Calls at other token counts in one process rebuild no module; ``dX`` and ``dfn`` at each count are judged
    against the naive BF16 tier over 32 inputs (and every output at 64 tokens first)."""
    _judge_case(64, hidden=hidden, record_property=record_property, prefix="T=64 ")
    loaded_before = loader.load.cache_info().misses
    for T in (1, 65, 2049, 7, 255):
        _judge_case(
            T,
            hidden=hidden,
            seed=SEED + T,
            names=("dX", "dfn"),
            prefix=f"T={T} ",
            record_property=record_property,
        )
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed token count rebuilt a module"
    )


_EPS_CASES = ((2e-5, 2e-6), (1e-6, 1e-5), (1e-5, 4e-6))


@pytest.mark.parametrize("T", (64, 2049))
@pytest.mark.parametrize("rms_eps,hc_eps", _EPS_CASES)
def test_non_default_eps_match_fp64(
    T: int, rms_eps: float, hc_eps: float, hidden: int, record_property
) -> None:
    """``rms_eps`` / ``hc_eps`` are launch arguments: every output at other values is judged against the naive BF16
    tier over 32 inputs (the oracle and the tier at the same values), and the statistics actually move."""
    _judge_case(
        T,
        hidden=hidden,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        record_property=record_property,
    )
    i = _make_inputs(T, hidden=hidden)
    default = _run(i)
    moved = _run({**i, "rms_eps": rms_eps, "hc_eps": hc_eps})
    assert not torch.equal(moved["stats"], default["stats"]), (
        f"stats unchanged by rms_eps={rms_eps:g} hc_eps={hc_eps:g} at H={hidden}"
    )


def test_changing_eps_in_one_process_does_not_rebuild(
    hidden: int, record_property
) -> None:
    """A different epsilon is a launch argument of the already compiled programs: no module is rebuilt or reloaded
    when the epsilons change between calls, and ``dX`` and ``dfn`` at each value are judged against the naive BF16
    tier over 32 inputs (the oracle and the tier at that value)."""
    _judge_case(64, hidden=hidden, record_property=record_property, prefix="T=64 ")
    loaded_before = loader.load.cache_info().misses
    for rms_eps, hc_eps in _EPS_CASES + ((RMS_EPS, HC_EPS),):
        _judge_case(
            255, hidden=hidden, seed=SEED + 3, rms_eps=rms_eps, hc_eps=hc_eps, names=("dX", "dfn"),
            prefix=f"rms_eps={rms_eps:g} hc_eps={hc_eps:g} ", record_property=record_property,
        )  # fmt: skip
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed epsilon rebuilt or reloaded a module"
    )


def _autograd_prepare(i: dict, *, save_stats: bool, **kwargs) -> dict:
    """``cake_mhc_det_train_prepare`` on fresh leaves of the inputs ``i``: the leaves, ``U`` (gradient retained) and the
    handle."""
    X = i["X"].clone().requires_grad_(True)
    fn = i["fn"].clone().requires_grad_(True)
    scale = i["scale"].clone().requires_grad_(True)
    base = i["base"].clone().requires_grad_(True)
    U, handle = pkg.cake_mhc_det_train_prepare(
        X, fn, scale, base, save_stats=save_stats, **kwargs
    )
    U.retain_grad()
    return {"X": X, "fn": fn, "scale": scale, "base": base, "U": U, "handle": handle}


def _autograd_finish(i: dict, c: dict) -> dict:
    """``cake_mhc_det_train_finish`` of a prepared connection ``c`` around the branch ``B = BF16(U + i["B"])`` (dB/dU = 1,
    so the branch backward hands dB on as dU)."""
    c["B"] = (c["U"].float() + i["B"].float()).to(BF16)
    c["Y"] = pkg.cake_mhc_det_train_finish(c["X"], c["B"], c.pop("handle"))
    return c


def _autograd_connection(i: dict, *, save_stats: bool, **kwargs) -> dict:
    """Prepare and finish of one connection, before the backward: the leaves, ``U``, ``B`` and ``Y``."""
    return _autograd_finish(i, _autograd_prepare(i, save_stats=save_stats, **kwargs))


_AUTOGRAD_OUTPUTS = ("U", "Y", "dB", "dX", "dfn", "dscale", "dbase")


def _autograd_case(i: dict, c: dict) -> tuple[dict, dict, dict]:
    """The judged outputs of one autograd connection ``c`` (``_autograd_connection`` after its backward) --
    ``U``, ``Y``, ``dB`` (the gradient of ``U``, which the branch hands on as ``dU``), ``dX``, ``dfn``, ``dscale``,
    ``dbase`` -- with the oracle's and the naive tier's values on the branch output and the branch's ``dU`` the
    autograd graph produced."""
    U, X = c["U"], c["X"]
    assert U.grad is not None and U.grad.dtype == BF16
    assert X.grad.dtype == BF16
    j = {**i, "B": c["B"].detach(), "dU": U.grad.detach()}
    expected, naive = expected_and_naive(j, _AUTOGRAD_OUTPUTS)
    outputs = {
        "U": U.detach(), "Y": c["Y"].detach(), "dB": U.grad, "dX": X.grad, "dfn": c["fn"].grad,
        "dscale": c["scale"].grad, "dbase": c["base"].grad,
    }  # fmt: skip
    return outputs, expected, naive


def _prefixed(prefix: str, *dicts: dict) -> list[dict]:
    return [{prefix + k: v for k, v in d.items()} for d in dicts]


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_autograd_end_to_end(
    T: int, save_stats: bool, hidden: int, record_property
) -> None:
    """``cake_mhc_det_train_prepare`` / ``cake_mhc_det_train_finish`` around a branch: ``U``, ``Y``, ``dB``, ``dX``,
    ``dfn``, ``dscale``, ``dbase`` judged against the naive BF16 tier over 32 inputs, the oracle and the tier on the
    branch output and the branch's ``dU`` the autograd graph produced."""

    def case(j: int, s: int):
        i = _make_inputs(T, hidden=hidden, seed=s)
        c = _autograd_connection(i, save_stats=save_stats)
        (c["Y"].float() * i["dY"].float()).sum().backward()
        outputs, expected, naive = _autograd_case(i, c)
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 31)


def test_workspace_bytes_and_layout(hidden: int) -> None:
    dev, h = _device(), {"hidden": hidden}
    small_f, small_b = (
        forward_workspace_bytes(1, dev, **h),
        backward_workspace_bytes(1, dev, **h),
    )
    large_f, large_b = (
        forward_workspace_bytes(65536, dev, **h),
        backward_workspace_bytes(65536, dev, **h),
    )
    assert (
        small_f > 0
        and small_b > 0
        and all(n % 256 == 0 for n in (small_f, small_b, large_f, large_b))
    )
    assert large_f >= small_f and large_b >= small_b
    previous = (0, 0)
    for T in (1, 7, 128, 129, 255, 1000, 2049, 16263, 20001, 65536):
        current = (
            forward_workspace_bytes(T, dev, **h),
            backward_workspace_bytes(T, dev, **h),
        )
        assert current[0] >= previous[0] and current[1] >= previous[1], (T, hidden)
        device_index = dev.index
        assert (
            current[0]
            >= loader.workspace_layout("fwd", T, device_index=device_index, **h)[
                "total_bytes"
            ]
        )
        assert (
            current[1]
            >= loader.workspace_layout("bwd", T, device_index=device_index, **h)[
                "total_bytes"
            ]
        )
        previous = current
    ws = backward_workspace(2049, dev, **h)
    assert ws.dtype == torch.uint8 and ws.numel() == backward_workspace_bytes(
        2049, dev, **h
    )
    assert torch.count_nonzero(ws) == 0
    ws.fill_(1)
    pkg.cake_mhc_det_train_reset_workspace(ws)
    assert torch.count_nonzero(ws) == 0
    with pytest.raises(ValueError):
        forward_workspace_bytes(-1, dev, **h)
    # A workspace sized for a larger token count serves a smaller one; a smaller one is refused.
    i = _make_inputs(129, hidden=hidden, seed=SEED + 5)
    big = forward_workspace(4096, dev, **h)
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"], workspace=big)
    assert torch.equal(
        stats, forward_prepare(i["X"], i["fn"], i["scale"], i["base"])[1]
    )
    with pytest.raises(ValueError):
        forward_prepare(
            i["X"],
            i["fn"],
            i["scale"],
            i["base"],
            workspace=torch.zeros(256, dtype=torch.uint8, device=_device()),
        )


def test_argument_validation(hidden: int) -> None:
    i = _make_inputs(8, hidden=hidden, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    with pytest.raises(ValueError):
        forward_prepare(X.float(), fn, scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X, fn.to(BF16), scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X, fn[:, 1:], scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X, fn, scale[:2], base)
    with pytest.raises(ValueError):
        forward_prepare(X.view(8, hidden * STREAMS), fn, scale, base)
    # Padded rows (a [T, 4, H] view of a wider buffer) are not materialized silently: the caller decides.
    padded = torch.zeros(8, STREAMS, hidden + 64, dtype=BF16, device=_device())[
        :, :, :hidden
    ]
    with pytest.raises(ValueError):
        forward_prepare(padded, fn, scale, base)
    with pytest.raises(ValueError):
        forward_prepare(X.cpu(), fn.cpu(), scale.cpu(), base.cpu())
    # The epsilons are launch arguments of the compiled programs (any finite positive value runs); the Sinkhorn
    # iteration count is an integer from 1 to SINKHORN_ITERS_MAX (test_cake_mhc_det_train_iters.py runs the counts).
    for kwargs in (
        {"rms_eps": 1e-6},
        {"hc_eps": 1e-5},
        {"rms_eps": 2e-5, "hc_eps": 2e-6},
    ):
        forward_prepare(X, fn, scale, base, **kwargs)
    for kwargs in (
        {"rms_eps": 0.0},
        {"rms_eps": -1e-5},
        {"hc_eps": float("nan")},
        {"hc_eps": float("inf")},
    ):
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, **kwargs)
    for kwargs in (
        {"sinkhorn_iters": 0},
        {"sinkhorn_iters": 20.5},
    ):
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, **kwargs)
    with pytest.raises(NotImplementedError):
        forward_prepare(X, fn, scale, base, sinkhorn_iters=pkg.SINKHORN_ITERS_MAX + 1)
    assert pkg.SHIPPED_SINKHORN_ITERS == (SINKHORN_ITERS,)
    U, stats = forward_prepare(X, fn, scale, base)
    with pytest.raises(ValueError):
        forward_finish(X, i["B"][:, : hidden // 2], stats)
    with pytest.raises(ValueError):
        forward_finish(X, i["B"], stats.double())
    dB, gstats = backward_prepare(i["dY"], i["B"], X, stats)
    with pytest.raises(ValueError):
        backward_finish(
            X, i["dY"], i["dU"], i["B"], fn, scale, base, stats, gstats, accumulate=True
        )
    with pytest.raises(ValueError):
        backward_finish(X, i["dY"], i["dU"], i["B"][:4], fn, scale, base, stats, gstats)
    with pytest.raises(ValueError):
        backward_finish(
            X,
            i["dY"],
            i["dU"],
            i["B"],
            fn,
            scale,
            base,
            stats,
            gstats[:, :16].contiguous(),
        )
    # B is optional in the backward finish (its contribution is in grad_stats): None gives the same bits.
    with_b = backward_finish(
        X, i["dY"], i["dU"], i["B"], fn, scale, base, stats, gstats
    )
    without_b = backward_finish(
        X, i["dY"], i["dU"], None, fn, scale, base, stats, gstats
    )
    assert all(torch.equal(a, b) for a, b in zip(with_b, without_b, strict=False))
    with pytest.raises(TypeError):
        pkg.cake_mhc_det_train_finish(X, i["B"], object())


# ----------------------------------------------------------------------------------------------------------------
# Hidden sizes mixed in one process (the host caches -- layouts, default workspaces and their views, caller-workspace
# views, launch plans, fast routes, route checks -- are keyed by the hidden size)

_MIXED_T = (129, 2049)
_WORKSPACE_MODES = ("default", "per_hidden", "shared")


def _bits(res: dict) -> dict:
    """A bitwise snapshot of the tensors of ``res``."""
    return {k: v.detach().clone() for k, v in res.items()}


def _assert_same_bits(ref: dict, got: dict, what: str) -> None:
    for k, v in ref.items():
        assert torch.equal(
            v.reshape(-1).view(torch.uint8), got[k].reshape(-1).view(torch.uint8)
        ), f"{what}: {k} differs bitwise"


def _mixed_workspaces(sizes: tuple[int, ...], T: int, mode: str) -> dict:
    """``{H: (forward workspace, backward workspace)}`` of a mixed-size test: ``default`` (``None``: the per-device
    default buffers), ``per_hidden`` (one caller buffer per size, ``hidden=H``) or ``shared`` (one zeroed caller
    buffer per kind serving every size, as large as the largest size's request)."""
    dev = _device()
    if mode == "default":
        return {H: (None, None) for H in sizes}
    if mode == "per_hidden":
        return {
            H: (
                forward_workspace(T, dev, hidden=H),
                backward_workspace(T, dev, hidden=H),
            )
            for H in sizes
        }
    assert mode == "shared", mode
    fwd = torch.zeros(
        max(forward_workspace_bytes(T, dev, hidden=H) for H in sizes),
        dtype=torch.uint8,
        device=dev,
    )
    bwd = torch.zeros(
        max(backward_workspace_bytes(T, dev, hidden=H) for H in sizes),
        dtype=torch.uint8,
        device=dev,
    )
    return {H: (fwd, bwd) for H in sizes}


def _interleaved_run(cases: dict, order: tuple[int, ...], ws: dict) -> dict:
    """The entry points of every case stage by stage: each stage is issued for every hidden size (in ``order``, the
    next stage in the reverse order) before the next stage, so consecutive calls switch the hidden size at one token
    count.  ``replay`` is the ``forward_stats`` record of each size."""
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
    return out


@pytest.mark.parametrize("T", _MIXED_T)
@pytest.mark.parametrize("workspaces", _WORKSPACE_MODES)
def test_mixed_hidden_sizes_same_token_count(
    T: int, workspaces: str, record_property
) -> None:
    """Calls of every exported hidden size at the same token count, interleaved stage by stage through the direct
    entry points (default workspaces, one caller workspace per size, one caller workspace shared by the sizes), in
    two rounds of opposite size order: ``forward_stats`` is bitwise the ``forward_prepare`` record, and both rounds
    are bitwise the sizes run one after the other; over 32 inputs of every size, every output of the interleaved run
    is judged against the naive BF16 tier of its size."""
    sizes = _mixed_hidden_sizes()
    cases = {H: _make_inputs(T, hidden=H, seed=SEED + 71 + H) for H in sizes}
    ws = _mixed_workspaces(sizes, T, workspaces)
    alone = {H: _bits(_run(cases[H], fwd_ws=ws[H][0], bwd_ws=ws[H][1])) for H in sizes}
    for rnd, order in enumerate((sizes, tuple(reversed(sizes)))):
        got = _interleaved_run(cases, order, ws)
        for H in sizes:
            what = f"T={T} H={H} mixed with {sizes} ({workspaces} workspaces, {rnd})"
            assert torch.equal(got[H]["replay"], got[H]["stats"]), (
                f"{what}: forward_stats is not bitwise the forward_prepare record"
            )
            _assert_same_bits(alone[H], got[H], what)
        del got

    def case(j: int, s: int):
        draw = {H: _make_inputs(T, hidden=H, seed=s + H) for H in sizes}
        got = _interleaved_run(draw, sizes, ws)
        outputs, expected, naive = {}, {}, {}
        for H in sizes:
            _assert_formats(
                draw[H],
                got[H],
                f"T={T} H={H} mixed with {sizes} ({workspaces} workspaces)",
            )
            o = package_outputs(got[H])
            e, n = expected_and_naive(draw[H], o)
            for d, part in zip(
                (outputs, expected, naive), _prefixed(f"H{H} ", o, e, n), strict=True
            ):
                d.update(part)
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 71)


@pytest.mark.parametrize("T", _MIXED_T)
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("workspaces", _WORKSPACE_MODES)
def test_mixed_hidden_sizes_autograd(
    T: int, save_stats: bool, workspaces: str, record_property
) -> None:
    """The autograd wrappers with every exported hidden size at the same token count: the prepares of every size,
    then the finishes in the reverse order, then one backward over the sum of the losses.  Each size's outputs and
    gradients are bitwise those of the sizes run one after the other, and over 32 inputs of every size they are
    judged as in ``test_autograd_end_to_end``."""
    sizes = _mixed_hidden_sizes()
    cases = {H: _make_inputs(T, hidden=H, seed=SEED + 81 + H) for H in sizes}
    ws = _mixed_workspaces(sizes, T, workspaces)

    def prepare(cs: dict, H: int) -> dict:
        wsf, wsb = ws[H]
        return _autograd_prepare(
            cs[H], save_stats=save_stats, workspace=wsf, backward_workspace=wsb
        )

    def loss(cs: dict, H: int, c: dict) -> torch.Tensor:
        return (c["Y"].float() * cs[H]["dY"].float()).sum()

    def mixed(cs: dict) -> dict:
        connections = {H: prepare(cs, H) for H in sizes}
        for H in reversed(sizes):
            _autograd_finish(cs[H], connections[H])
        torch.stack([loss(cs, H, c) for H, c in connections.items()]).sum().backward()
        return connections

    def bits(c: dict) -> dict:
        return _bits(
            {
                "U": c["U"], "Y": c["Y"], "dU": c["U"].grad, "dX": c["X"].grad, "dfn": c["fn"].grad,
                "dscale": c["scale"].grad, "dbase": c["base"].grad,
            }
        )  # fmt: skip

    alone = {}
    for H in sizes:
        c = _autograd_finish(cases[H], prepare(cases, H))
        loss(cases, H, c).backward()
        alone[H] = bits(c)
        del c
    for H, c in mixed(cases).items():
        what = f"T={T} H={H} mixed with {sizes} ({save_stats=}, {workspaces=})"
        _assert_same_bits(alone[H], bits(c), what)

    def case(j: int, s: int):
        draw = {H: _make_inputs(T, hidden=H, seed=s + H) for H in sizes}
        outputs, expected, naive = {}, {}, {}
        for H, c in mixed(draw).items():
            for d, part in zip(
                (outputs, expected, naive),
                _prefixed(f"H{H} ", *_autograd_case(draw[H], c)),
                strict=True,
            ):
                d.update(part)
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 81)


def test_workspace_bytes_per_hidden_size() -> None:
    """The workspace helpers size per hidden size: at each exported size the request is a multiple of 256 bytes,
    non-decreasing in the token count and at least the exact layout of the call; the forward request grows with the
    hidden size at a token count (its N-sized split of ``fn``; the backward requests are printed per size); an
    unexported or unknown hidden size is a ``ValueError``; the default is 4096."""
    dev = _device()
    sizes = _hidden_sizes()
    unexported = tuple(H for H in HIDDEN_SIZES if H not in sizes)
    helpers = {"fwd": forward_workspace_bytes, "bwd": backward_workspace_bytes}
    for kind, nbytes in helpers.items():
        for H in sizes:
            previous = 0
            for T in (1, 7, 128, 129, 1000, 2049, 8192, 16263, 65536):
                current = nbytes(T, dev, hidden=H)
                exact = loader.workspace_layout(
                    kind, T, device_index=dev.index, hidden=H
                )["total_bytes"]
                assert current % 256 == 0 and current >= max(previous, exact), (
                    kind,
                    H,
                    T,
                    current,
                    previous,
                    exact,
                )
                previous = current
        assert nbytes(129, dev) == nbytes(129, dev, hidden=4096), kind
        for bad in (6144, 0, -4096, 4097, 2 * 7168, True) + unexported:
            with pytest.raises(ValueError):
                nbytes(129, dev, hidden=bad)
    for allocate in (forward_workspace, backward_workspace):
        for bad in (6144,) + unexported:
            with pytest.raises(ValueError):
                allocate(129, dev, hidden=bad)
    # Below the P_K band of 8192+ tokens on sm_103a, P_K is non-decreasing in the hidden size and the split of fn
    # grows with it, so the forward request strictly grows with the hidden size.
    for T in (1, 129, 2049, 8192):
        forward = [forward_workspace_bytes(T, dev, hidden=H) for H in sizes]
        assert all(a < b for a, b in zip(forward, forward[1:], strict=False)), (
            T,
            dict(zip(sizes, forward, strict=True)),
        )
        backward = {H: backward_workspace_bytes(T, dev, hidden=H) for H in sizes}
        print(f"workspace bytes T={T}: forward {forward}, backward {backward}")


@pytest.mark.parametrize("T", _MIXED_T)
def test_workspace_sized_for_another_hidden_size(T: int) -> None:
    """A caller workspace serves a call when it holds the call's ``(T, H)`` layout and is refused (``ValueError``)
    when it is too small: buffers sized for ``(T, 4096)`` used by an H 7168 call at the same ``T`` are refused where
    smaller than its layout and otherwise give bitwise the default-workspace outputs; buffers one 256-byte block
    short of the H 7168 layout are refused and exact-size buffers serve it.  At 129 tokens the forward buffer of 4096
    is always too small (the N-sized split of ``fn`` alone grows by 2359296 bytes, more than the token-dependent
    regions of 129 tokens hold)."""
    sizes = _mixed_hidden_sizes()
    assert 7168 in sizes
    dev = _device()
    i = _make_inputs(T, hidden=7168, seed=SEED + 91)
    ref = _bits(_run(i))
    need = {
        kind: loader.workspace_layout(kind, T, device_index=dev.index, hidden=7168)[
            "total_bytes"
        ]
        for kind in ("fwd", "bwd")
    }
    sized_4096 = {
        "fwd": forward_workspace(T, dev, hidden=4096),
        "bwd": backward_workspace(T, dev, hidden=4096),
    }
    if T == 129:
        held = sized_4096["fwd"].numel()
        assert held < need["fwd"], (held, need)
    print(
        f"T={T}: H 4096 buffers {sized_4096['fwd'].numel()} / {sized_4096['bwd'].numel()} bytes, H 7168 layouts "
        f"{need['fwd']} / {need['bwd']} bytes"
    )

    def forward_calls(ws: torch.Tensor) -> list:
        return [
            lambda: forward_prepare(i["X"], i["fn"], i["scale"], i["base"], workspace=ws)[1],
            lambda: forward_stats(i["X"], i["fn"], i["scale"], i["base"], workspace=ws),
            lambda: pkg.cake_mhc_det_train_prepare(i["X"], i["fn"], i["scale"], i["base"], workspace=ws)[0],
        ]  # fmt: skip

    def backward_call(ws: torch.Tensor) -> tuple:
        return backward_finish(
            i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], ref["stats"], ref["gstats"],
            workspace=ws,
        )  # fmt: skip

    short = {
        kind: torch.zeros(need[kind] - 256, dtype=torch.uint8, device=dev)
        for kind in need
    }
    exact = {
        kind: torch.zeros(need[kind], dtype=torch.uint8, device=dev) for kind in need
    }
    for label, buffers in (("H 4096", sized_4096), ("short", short), ("exact", exact)):
        fwd_ws, bwd_ws = buffers["fwd"], buffers["bwd"]
        what = f"T={T} H=7168 on a {label} buffer"
        if fwd_ws.numel() < need["fwd"]:
            for call in forward_calls(fwd_ws):
                with pytest.raises(ValueError):
                    call()
        else:
            _U, stats = forward_prepare(
                i["X"], i["fn"], i["scale"], i["base"], workspace=fwd_ws
            )
            assert torch.equal(stats, ref["stats"]), f"{what}: forward stats"
        if bwd_ws.numel() < need["bwd"]:
            with pytest.raises(ValueError):
                backward_call(bwd_ws)
        else:
            names = ("dX", "dfn", "dscale", "dbase")
            got = dict(zip(names, backward_call(bwd_ws), strict=True))
            _assert_same_bits({k: ref[k] for k in names}, got, f"{what}: backward")


def test_hidden_size_validation() -> None:
    """``X`` of a hidden size outside the package's sizes, ``fn`` / ``B`` / ``dU`` of another size than ``X`` and an
    unexported size of the architecture are ``ValueError``s at every entry point; at every size a Sinkhorn count above
    ``SINKHORN_ITERS_MAX`` is a ``NotImplementedError`` at every entry point that takes one, a count within it runs
    (``test_cake_mhc_det_train_iters.py`` checks its numerics), and the size's default count runs unchanged afterwards."""
    sizes = _hidden_sizes()
    T = 8
    zeros_stats = torch.zeros(T, 64, dtype=F32, device=_device())
    zeros_gstats = torch.zeros(T, 32, dtype=F32, device=_device())

    def every_entry_point(i: dict, stats, gstats, **kwargs) -> list:
        args = (i["X"], i["fn"], i["scale"], i["base"])
        return [
            lambda: forward_prepare(*args, **kwargs),
            lambda: forward_stats(*args, **kwargs),
            lambda: forward_finish(i["X"], i["B"], stats),
            lambda: backward_prepare(i["dY"], i["B"], i["X"], stats),
            lambda: backward_finish(
                i["X"], i["dY"], i["dU"], i["B"], *args[1:], stats, gstats, **kwargs
            ),
            lambda: pkg.cake_mhc_det_train_prepare(*args, **kwargs),
        ]

    # A hidden size outside (4096, 5120, 7168), with every operand of that size.
    bad = _make_inputs(T, hidden=6144, seed=SEED + 93)
    for call in every_entry_point(bad, zeros_stats, zeros_gstats):
        with pytest.raises(ValueError):
            call()
    # A hidden size the architecture does not export (5120 / 7168 on sm_107a): the route and the supported sizes.
    for H in (H for H in HIDDEN_SIZES if H not in sizes):
        unexported = _make_inputs(T, hidden=H, seed=SEED + 94)
        for call in every_entry_point(unexported, zeros_stats, zeros_gstats):
            with pytest.raises(ValueError, match="supported hidden sizes"):
                call()
    for H in sizes:
        i = _make_inputs(T, hidden=H, seed=SEED + 95)
        _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])
        _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
        # Operands of another hidden size than X.
        for other in (o for o in HIDDEN_SIZES if o != H):
            j = _make_inputs(T, hidden=other, seed=SEED + 96)
            for wrong in ("fn", "B", "dU"):
                mixed = {**i, wrong: j[wrong]}
                calls = every_entry_point(mixed, stats, gstats)
                for k, call in enumerate(calls):
                    if wrong == "B" and k in (0, 1, 5):
                        continue  # the prepares take no B
                    if wrong == "dU" and k != 4:
                        continue  # dU enters the backward finish only
                    if wrong == "fn" and k in (2, 3):
                        continue  # the finish and the backward prepare take no fn
                    with pytest.raises(ValueError):
                        call()
        forward_prepare(i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=5)
        calls = every_entry_point(
            i, stats, gstats, sinkhorn_iters=pkg.SINKHORN_ITERS_MAX + 1
        )
        for k, call in enumerate(calls):
            if k in (2, 3):
                continue  # the finish and the backward prepare take no count
            with pytest.raises(NotImplementedError):
                call()
        again = forward_prepare(i["X"], i["fn"], i["scale"], i["base"])[1]
        assert torch.equal(again, stats), (
            f"H={H}: the default count changed after a refused count"
        )


def test_changing_hidden_size_in_one_process_does_not_reload(record_property) -> None:
    """Once each exported hidden size ran at a token count, switching the size between calls loads no module, and
    every call is bitwise its first run; ``dX`` and ``dfn`` of every (token count, size) are judged against the naive
    BF16 tier of the size over 32 inputs, the sizes switching between the calls."""
    sizes = _mixed_hidden_sizes()
    rows = (1, 65, 255, 2049)
    cases = {
        (T, H): _make_inputs(T, hidden=H, seed=SEED + 97 + T)
        for T in rows
        for H in sizes
    }
    first = {key: _bits(_run(i)) for key, i in cases.items()}
    loaded = loader.load.cache_info().misses
    for T in rows:
        for H in sizes + tuple(reversed(sizes)):
            res = _run(cases[(T, H)])
            _assert_same_bits(first[(T, H)], res, f"T={T} H={H} after a size switch")

    def case(j: int, s: int):
        outputs, expected, naive = {}, {}, {}
        for T in rows:
            for H in sizes:
                i = _make_inputs(T, hidden=H, seed=s + T)
                res = _run(i)
                o = {k: res[k] for k in ("dX", "dfn")}
                e, n = expected_and_naive(i, o)
                for d, part in zip(
                    (outputs, expected, naive),
                    _prefixed(f"T={T} H={H} ", o, e, n),
                    strict=True,
                ):
                    d.update(part)
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 97)
    assert loader.load.cache_info().misses == loaded, (
        "a changed hidden size loaded a module again"
    )


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library, architecture) the registry
    references, each with the kernel and binding sources under ``csrc/`` of this package, their digests and a
    closure digest; the registry itself records the host plan only.
    """
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
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
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()


# ----------------------------------------------------------------------------------------------------------------
# The opt-in two-word accumulation (``accumulation="fp32x2"``): hidden 4096 and the Sinkhorn count of record only.
# The mapping pass, the token-owner pass, the column slab and the final reduction run their two-word programs
# (``loader.ACC_WORDS_PROGRAMS``); the rest of the route and every workspace are those of the default mode.

TWO_WORD = "fp32x2"
# 1 token; the small bands; 2049; 4097 (above the checkpoint carry); 16263 (the group-row forms of sm_100a / sm_103a)
T_TWO_WORD = (1, 64, 255, 2049, 4097, 16263)


def _two_word_hidden() -> int:
    """The hidden size of the two-word programs, skipped where the device does not export it."""
    return _require_hidden(loader.ACC_WORDS_HIDDEN)


@pytest.mark.parametrize("T", T_TWO_WORD)
def test_two_word_accumulation_matches_fp64(T: int, record_property) -> None:
    """``accumulation="fp32x2"``: every output (``U``, ``Y``, ``dB``, ``dX``, the statistics, the gradient statistics,
    ``dfn``, ``dscale``, ``dbase``) judged against the naive BF16 tier over 32 inputs."""
    _judge_case(
        T, hidden=_two_word_hidden(), accumulation=TWO_WORD, record_property=record_property, prefix="fp32x2 "
    )  # fmt: skip


@pytest.mark.parametrize("T", (257, 2049))
def test_two_word_cancellation_and_tails(T: int, record_property) -> None:
    """The two-word mode on the cancellation inputs and on saturated sigmoid tails: every output judged against the
    naive BF16 tier over 32 inputs, with the exact checks of those regimes on every draw."""
    hidden = _two_word_hidden()
    _judge_case(
        T, hidden=hidden, mode="cancel", host_draw=True, accumulation=TWO_WORD, record_property=record_property,
        prefix="fp32x2 cancel ",
    )  # fmt: skip
    _judge_case(
        T, hidden=hidden, mode="tail", accumulation=TWO_WORD, record_property=record_property, prefix="fp32x2 tail "
    )  # fmt: skip


@pytest.mark.parametrize("T", (7, 2049, 4097))
@pytest.mark.parametrize("save_stats", (True, False))
def test_two_word_autograd_matches_fp64(
    T: int, save_stats: bool, record_property
) -> None:
    """``cake_mhc_det_train_prepare(..., accumulation="fp32x2")`` / ``cake_mhc_det_train_finish`` around a branch (the
    statistics saved or replayed in the backward): ``U``, ``Y``, ``dB``, ``dX``, ``dfn``, ``dscale``, ``dbase`` judged
    against the naive BF16 tier over 32 inputs."""
    hidden = _two_word_hidden()

    def case(j: int, s: int):
        i = _make_inputs(T, hidden=hidden, seed=s)
        c = _autograd_connection(i, save_stats=save_stats, accumulation=TWO_WORD)
        (c["Y"].float() * i["dY"].float()).sum().backward()
        outputs, expected, naive = _autograd_case(i, c)
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 131, prefix="fp32x2 ")


@pytest.mark.parametrize("T", (129, 16263))
def test_two_word_outputs_are_deterministic(T: int) -> None:
    """Three launches of the whole two-word forward / backward on the same inputs, outputs and the non-ticket
    workspace regions re-poisoned with NaN between them, give bitwise identical outputs (run-to-run, one device),
    equal to a call on the default workspaces; the ticket words are zero after the launches."""
    hidden = _two_word_hidden()
    i = {**_make_inputs(T, hidden=hidden, seed=SEED + 3), "accumulation": TWO_WORD}
    fwd_ws = forward_workspace(T, _device(), hidden=hidden)
    bwd_ws = backward_workspace(T, _device(), hidden=hidden)
    layouts = {
        kind: loader.workspace_layout(
            kind, T, device_index=_device().index, hidden=hidden
        )
        for kind in ("fwd", "bwd")
    }

    def poison(ws: torch.Tensor, kind: str) -> None:
        for name, region in layouts[kind]["regions"].items():
            if name != "ticket":
                ws[region["offset"] : region["offset"] + region["bytes"]].fill_(0xFF)

    results = []
    for _repeat in range(3):
        poison(fwd_ws, "fwd")
        poison(bwd_ws, "bwd")
        res = _run(i, fwd_ws=fwd_ws, bwd_ws=bwd_ws)
        torch.cuda.synchronize()
        results.append({k: v.clone() for k, v in res.items()})
        for k, v in res.items():
            assert torch.isfinite(v.float()).all(), k
    fresh = _run(i)
    for later in results[1:] + [fresh]:
        for k, v in results[0].items():
            assert torch.equal(v, later[k]), (
                f"a repeated two-word launch changed a bit of {k}"
            )
    for ws, kind in ((fwd_ws, "fwd"), (bwd_ws, "bwd")):
        region = layouts[kind]["regions"]["ticket"]
        assert (
            torch.count_nonzero(
                ws[region["offset"] : region["offset"] + region["bytes"]]
            )
            == 0
        )


@pytest.mark.parametrize("T", (129, 2049, 4097))
def test_two_word_checkpoint_replay_is_bitwise(T: int) -> None:
    """In the two-word mode ``forward_stats`` is bitwise the ``forward_prepare`` record, and the autograd gradients
    with the statistics replayed in the backward (``save_stats=False``; the checkpoint carry where it applies) are
    bitwise those with the statistics saved."""
    hidden = _two_word_hidden()
    i = _make_inputs(T, hidden=hidden, seed=SEED + 11)
    _U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], accumulation=TWO_WORD
    )
    replay = forward_stats(
        i["X"], i["fn"], i["scale"], i["base"], accumulation=TWO_WORD
    )
    assert torch.equal(stats, replay), (
        "two-word forward_stats is not bitwise the forward_prepare record"
    )
    grads = []
    for save_stats in (True, False):
        c = _autograd_connection(i, save_stats=save_stats, accumulation=TWO_WORD)
        (c["Y"].float() * i["dY"].float()).sum().backward()
        grads.append(tuple(c[k].grad.clone() for k in ("X", "fn", "scale", "base")))
    for saved, recomputed in zip(*grads, strict=True):
        assert torch.equal(saved, recomputed), (
            "two-word checkpoint replay changed a gradient bit"
        )


@pytest.mark.parametrize("T", (129, 16263))
def test_two_word_workspaces_are_the_default_ones(T: int) -> None:
    """The two-word mode runs in workspaces of exactly ``*_workspace_bytes(T)`` (the default mode's sizes), writes
    nothing past them (guard bytes after each buffer keep their value) and gives the bits of a call on the default
    workspaces."""
    hidden = _two_word_hidden()
    guard = 4096
    i = {**_make_inputs(T, hidden=hidden, seed=SEED + 17), "accumulation": TWO_WORD}
    sizes = {
        "fwd": forward_workspace_bytes(T, _device(), hidden=hidden),
        "bwd": backward_workspace_bytes(T, _device(), hidden=hidden),
    }
    buffers = {
        k: torch.zeros(n + guard, dtype=torch.uint8, device=_device())
        for k, n in sizes.items()
    }
    for k, n in sizes.items():
        buffers[k][n:].fill_(0xA5)
    res = _run(
        i, fwd_ws=buffers["fwd"][: sizes["fwd"]], bwd_ws=buffers["bwd"][: sizes["bwd"]]
    )
    torch.cuda.synchronize()
    for k, n in sizes.items():
        tail = buffers[k][n:]
        assert torch.equal(tail, torch.full_like(tail, 0xA5)), (
            f"the two-word mode wrote past the {k} workspace"
        )
    fresh = _run(i)
    for k, v in res.items():
        assert torch.equal(v, fresh[k]), (
            f"{k}: caller workspaces of the default size changed a bit"
        )


def test_two_word_launches_the_two_word_programs(monkeypatch) -> None:
    """With fresh plan caches, a default call and then a two-word call at the same token count: the two-word call
    loads exactly its own selection (``selected_modules(..., accumulation="fp32x2")``, the plan cache keyed by the
    mode), whose records of the two-word roles are the programs of ``ACC_WORDS_PROGRAMS`` (the group-row forms where
    ``bwd_group_rows`` holds), every other record the default selection's."""
    hidden = _two_word_hidden()
    monkeypatch.setattr(pkg, "_PLANS", {})
    monkeypatch.setattr(pkg, "_FAST_ROUTES", {})
    loaded: list[str] = []
    load = loader.load

    def recording_load(name: str, arch: str):
        loaded.append(name)
        return load(name, arch)

    monkeypatch.setattr(loader, "load", recording_load)
    arch, sm = _arch(), loader.sm_count(_device().index)
    for T in (2049, 16263):
        i = _make_inputs(T, hidden=hidden, seed=SEED + 19)
        _run(i)
        for kind, path in (("fwd_prepare", "forward"), ("bwd_finish", "backward")):
            default = loader.selected_modules(
                arch, hidden, kind, rows=T, sm_count=sm, path=path
            )
            two = loader.selected_modules(
                arch,
                hidden,
                kind,
                rows=T,
                sm_count=sm,
                path=path,
                accumulation=TWO_WORD,
            )
            rows = loader.bwd_group_rows(T, sm, arch, hidden)
            assert len(two) == len(default), (kind, default, two)
            for one, pair in zip(default, two, strict=True):
                role = loader.MODULES[one]["role"]
                if role in loader.ACC_WORDS_ROLES:
                    key = (
                        f"{role}/rows"
                        if rows and f"{role}/rows" in loader.ACC_WORDS_PROGRAMS
                        else role
                    )
                    assert pair == loader.ACC_WORDS_PROGRAMS[key][arch], (T, role, pair)
                    assert (
                        pair not in default and arch in loader.MODULES[pair]["arches"]
                    ), (T, pair)
                else:
                    assert pair == one, (T, kind, one, pair)
        start = len(loaded)
        res = _run({**i, "accumulation": TWO_WORD})
        assert all(torch.isfinite(v.float()).all() for v in res.values())
        wanted = [
            *loader.selected_modules(arch, hidden, "fwd_prepare", rows=T, sm_count=sm, path="forward",
                                     accumulation=TWO_WORD),
            *loader.selected_modules(arch, hidden, "bwd_finish", rows=T, sm_count=sm, path="backward",
                                     accumulation=TWO_WORD),
        ]  # fmt: skip
        pairs = [
            n for n in wanted if loader.MODULES[n]["role"] in loader.ACC_WORDS_ROLES
        ]
        assert pairs and all(n in loaded[start:] for n in pairs), (
            T,
            pairs,
            loaded[start:],
        )


def test_two_word_mode_is_refused_where_it_has_no_program() -> None:
    """``accumulation="fp32x2"`` at another hidden size or Sinkhorn count is a ``NotImplementedError`` (no two-word
    program is built there) on every entry point that takes the mode; a value other than ``"fp32"`` / ``"fp32x2"`` is
    a ``ValueError``.  The default mode keeps serving those calls."""
    two = loader.ACC_WORDS_HIDDEN
    for hidden in _hidden_sizes():
        i = _make_inputs(8, hidden=hidden, seed=SEED + 23)
        args = (i["X"], i["fn"], i["scale"], i["base"])
        _U, stats = forward_prepare(*args)
        _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
        bwd = (
            i["X"],
            i["dY"],
            i["dU"],
            i["B"],
            i["fn"],
            i["scale"],
            i["base"],
            stats,
            gstats,
        )
        bad = [{"accumulation": TWO_WORD}] if hidden != two else []
        bad.append({"accumulation": TWO_WORD, "sinkhorn_iters": 7})
        for kwargs in bad:
            with pytest.raises(NotImplementedError, match="fp32x2"):
                forward_prepare(*args, **kwargs)
            with pytest.raises(NotImplementedError, match="fp32x2"):
                forward_stats(*args, **kwargs)
            with pytest.raises(NotImplementedError, match="fp32x2"):
                backward_finish(*bwd, **kwargs)
            with pytest.raises(NotImplementedError, match="fp32x2"):
                pkg.cake_mhc_det_train_prepare(*args, **kwargs)
        for value in ("fp64", "FP32X2", "fp32x4", None, 2):
            with pytest.raises(ValueError, match="accumulation"):
                forward_prepare(*args, accumulation=value)
            with pytest.raises(ValueError, match="accumulation"):
                backward_finish(*bwd, accumulation=value)
            with pytest.raises(ValueError, match="accumulation"):
                pkg.cake_mhc_det_train_prepare(*args, accumulation=value)
