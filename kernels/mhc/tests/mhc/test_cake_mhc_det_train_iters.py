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

Tests of the run-time Sinkhorn iteration count of the Cake four-stream mHC training package (SM100 / SM103 / SM107).

``sinkhorn_iters`` is an integer from 1 to ``SINKHORN_ITERS_MAX`` (64).  The default 20 runs the compiled programs,
whose chain is unrolled for it; every other count selects the run-time-count programs of the coefficient pass and the
token-owner pass, which take the count as a kernel argument.  This module checks:

* the selection (pure host rules: at 20 no run-time-count record launches, at any other count the coefficient
  and token-owner passes launch exactly their run-time-count record) is checked on CPU in
  ``tests/cpu/mhc/test_cake_mhc_det_train_rules.py``;
* the numerics at counts 1, 2, 5, 10, 19, 21, 30, 50 and 64 on T = 1, 7, 257, 1024 and 16263: every output judged
  with ``tests/cake_naive.py`` against the naive BF16 tier over 32 inputs, the float64 oracle and the tier evaluated
  at the same count (``tests/cake/test_helpers/cake_mhc_naive.py``); where the FP64 Sinkhorn matrices of the count and
  of 20 lie apart, the kernel's ``C`` is nearer the count's;
* the paths: at each count the checkpoint replay (``forward_stats``) is bitwise the ``forward_prepare`` record, and the
  autograd functions (``save_stats`` True and False) give bitwise the outputs and gradients of the direct entry points;
* that changing the count in one process loads no module after the first run-time-count call, and the argument checks.

The cases above run at ``H = 4096``; ``test_counts_match_fp64_at_every_hidden_size`` and
``test_paths_agree_bitwise_at_every_hidden_size`` repeat the numerics and the paths at the counts 1, 7, 20 and 64 at
every hidden size the device's architecture exports (4096, 5120 and 7168 on SM100 / SM103 / SM107).

The oracle and the naive tier are those of ``test_cake_mhc_det_train.py`` with the count as a parameter.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_mhc_naive import case_oracle, judge_package
from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake.cake_mhc_det_train")
loader = cake_package("cake.cake_mhc_det_train", "cake_jit")
forward_prepare = pkg.cake_mhc_det_train_forward_prepare
forward_stats = pkg.cake_mhc_det_train_forward_stats
forward_finish = pkg.cake_mhc_det_train_forward_finish
backward_prepare = pkg.cake_mhc_det_train_backward_prepare
backward_finish = pkg.cake_mhc_det_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, MAPPING_ROWS = 4, 4096, 24
HIDDEN_SIZES = (4096, 5120, 7168)
HIDDEN_COUNTS = (1, 7, 20, 64)
RMS_EPS, HC_EPS, SINKHORN_ITERS, SINKHORN_ITERS_MAX = 1e-5, 1e-6, 20, 64
COUNTS = (1, 2, 5, 10, 19, 21, 30, 50, 64)
T_COUNTS = (1, 7, 257, 1024, 16263)
# The relative L2 distance between the FP64 Sinkhorn matrices of a count and of 20 above which the kernel's C must
# lie nearer its own count's (below it a chain at its fixed point leaves nothing to tell apart).
COUNT_APART = 4e-5
SEED = 2032
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and pkg.is_cake_mhc_det_train_supported(_device())
    )


needs_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


def _require_hidden(hidden: int) -> int:
    """``hidden`` if the device's architecture exports every route of it, else the case is skipped."""
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(_device()))
    sizes = tuple(loader.supported_hidden_sizes(arch)) if arch else ()
    if hidden not in sizes:
        pytest.skip(
            f"hidden {hidden} has no exported route on {arch} (exported: {sizes})"
        )
    return hidden


# ----------------------------------------------------------------------------------------------------------------
# Inputs, launches and the judge


def _make_inputs(T: int, *, seed: int = SEED, hidden: int = HIDDEN) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, hidden, dtype=BF16), "fn": (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3), "base": 0.1 * randn(MAPPING_ROWS), "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, STREAMS, hidden, dtype=BF16), "dU": randn(T, hidden, dtype=BF16),
    }  # fmt: skip


def _run(i: dict, iters: int) -> dict:
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters
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
        sinkhorn_iters=iters,
    )
    return {"U": U, "stats": stats, "Y": Y, "dB": dB, "gstats": gstats, "dX": dX, "dfn": dfn, "dscale": dscale,
            "dbase": dbase}  # fmt: skip


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.double() - b.double()).norm().item()
    den = b.double().norm().item()
    return num / den if den > 0 else (0.0 if num == 0 else float("inf"))


def _check(
    T: int,
    iters: int,
    record_property,
    *,
    seed: int = SEED,
    hidden: int = HIDDEN,
    names=None,
    prefix: str = "",
) -> None:
    """Every output of the package at the count (or the outputs ``names``) judged against the naive BF16 tier over 32
    inputs, the oracle and the tier at the same count; every draw's formats and the zero pad of ``gstats``."""
    label = f"T={T} sinkhorn_iters={iters} H={hidden}"

    def check(i: dict, results: list, expected: dict) -> None:
        for res in results:
            for k in ("U", "Y", "dB", "dX"):
                assert res[k].dtype == BF16, f"{k} {label}"
            assert torch.equal(
                res["gstats"][:, 20:], torch.zeros_like(res["gstats"][:, 20:])
            ), f"gstats pad {label}"

    judge_package(
        lambda s: _make_inputs(T, seed=s, hidden=hidden), lambda i: _run(i, iters), record_property, seed=seed,
        names=names, prefix=prefix, check=check, iters=iters,
    )  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# Tests


@needs_device
@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", T_COUNTS)
def test_counts_match_fp64(T: int, iters: int, record_property) -> None:
    _check_counts(T, iters, HIDDEN, record_property)


@needs_device
@pytest.mark.parametrize("iters", HIDDEN_COUNTS)
@pytest.mark.parametrize("T", T_COUNTS)
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_counts_match_fp64_at_every_hidden_size(
    hidden: int, T: int, iters: int, record_property
) -> None:
    """The counts 1, 7, 20 and 64 at hidden 5120 and 7168, judged against the naive tier of the size (H 4096 is
    ``test_counts_match_fp64``)."""
    _check_counts(T, iters, _require_hidden(hidden), record_property)


def _check_counts(T: int, iters: int, hidden: int, record_property) -> None:
    _check(T, iters, record_property, hidden=hidden)
    # The count reaches the kernels: where the FP64 Sinkhorn matrices of this count and of 20 lie apart, the record's
    # matrix is nearer this count's. A chain at its fixed point (T=1 from about 19 steps on) leaves the two FP64
    # matrices closer than that and nothing to tell apart; the values alone are judged there.
    i = _make_inputs(T, hidden=hidden)
    C = _run(i, iters)["stats"][:, 33:49].reshape(T, STREAMS, STREAMS)
    ref_n = case_oracle(i, iters=iters)["C"]
    ref_20 = case_oracle(i, iters=SINKHORN_ITERS)["C"]
    if _rel_l2(ref_n, ref_20) > COUNT_APART:
        assert _rel_l2(C, ref_n) < _rel_l2(C, ref_20), (
            f"stats C at sinkhorn_iters={iters} H={hidden} is nearer the count of 20's"
        )


@needs_device
@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
def test_paths_agree_bitwise(T: int, iters: int) -> None:
    """At each count: the checkpoint replay is bitwise the forward record, and the autograd functions (statistics saved
    or replayed) give bitwise the outputs and gradients of the direct entry points on the same branch."""
    _check_paths(T, iters, HIDDEN)


@needs_device
@pytest.mark.parametrize("iters", HIDDEN_COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_paths_agree_bitwise_at_every_hidden_size(
    hidden: int, T: int, iters: int
) -> None:
    """``test_paths_agree_bitwise`` at hidden 5120 and 7168 and the counts 1, 7, 20 and 64."""
    _check_paths(T, iters, _require_hidden(hidden))


def _check_paths(T: int, iters: int, hidden: int) -> None:
    i = _make_inputs(T, seed=SEED + 41, hidden=hidden)
    U, stats = forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters
    )
    assert torch.equal(
        stats,
        forward_stats(i["X"], i["fn"], i["scale"], i["base"], sinkhorn_iters=iters),
    )
    B = (U.float() + i["B"].float()).to(
        BF16
    )  # the branch: dB/dU = 1, so its backward hands dB on as dU
    Y = forward_finish(i["X"], B, stats)
    dB, gstats = backward_prepare(i["dY"], B, i["X"], stats)
    direct = backward_finish(
        i["X"],
        i["dY"],
        dB,
        B,
        i["fn"],
        i["scale"],
        i["base"],
        stats,
        gstats,
        sinkhorn_iters=iters,
    )
    for save_stats in (True, False):
        X = i["X"].clone().requires_grad_(True)
        fn = i["fn"].clone().requires_grad_(True)
        scale = i["scale"].clone().requires_grad_(True)
        base = i["base"].clone().requires_grad_(True)
        Ua, handle = pkg.cake_mhc_det_train_prepare(
            X, fn, scale, base, sinkhorn_iters=iters, save_stats=save_stats
        )
        Ua.retain_grad()
        Ba = (Ua.float() + i["B"].float()).to(BF16)
        Ya = pkg.cake_mhc_det_train_finish(X, Ba, handle)
        (Ya.float() * i["dY"].float()).sum().backward()
        label = f"T={T} sinkhorn_iters={iters} save_stats={save_stats} H={hidden}"
        assert torch.equal(Ua.detach(), U) and torch.equal(Ya.detach(), Y), label
        assert torch.equal(Ua.grad, dB), label
        for got, want, name in zip(
            (X.grad, fn.grad, scale.grad, base.grad),
            direct,
            ("dX", "dfn", "dscale", "dbase"),
            strict=True,
        ):
            assert torch.equal(got, want), (
                f"autograd {name} differs from the direct entry point, {label}"
            )


@needs_device
def test_changing_count_does_not_rebuild(record_property) -> None:
    """The count is a launch argument of the run-time-count programs: after one call at a non-default count, calls at
    other counts (and back at the default) load no module, and ``dX`` and ``dfn`` at each count are judged against
    the naive BF16 tier at that count over 32 inputs."""
    _check(64, SINKHORN_ITERS, record_property, prefix="T=64 sinkhorn_iters=20 ")
    _check(64, 3, record_property, prefix="T=64 sinkhorn_iters=3 ")
    loaded = loader.load.cache_info().misses
    for iters in (1, 2, 7, 33, 64, SINKHORN_ITERS, 11):
        _check(
            255,
            iters,
            record_property,
            seed=SEED + iters,
            names=("dX", "dfn"),
            prefix=f"sinkhorn_iters={iters} ",
        )
    assert loader.load.cache_info().misses == loaded, (
        "a changed sinkhorn_iters loaded a module"
    )


@needs_device
def test_count_validation() -> None:
    assert pkg.SHIPPED_SINKHORN_ITERS == (SINKHORN_ITERS,)
    assert pkg.SINKHORN_ITERS_MAX == loader.SINKHORN_ITERS_MAX == SINKHORN_ITERS_MAX
    i = _make_inputs(8, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    for iters in (0, -1, 2.5, 20.5, True, float("inf"), float("nan"), "20"):
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(ValueError):
            forward_stats(X, fn, scale, base, sinkhorn_iters=iters)
    for iters in (SINKHORN_ITERS_MAX + 1, 1000):
        with pytest.raises(NotImplementedError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(NotImplementedError):
            pkg.cake_mhc_det_train_prepare(X, fn, scale, base, sinkhorn_iters=iters)
    for iters in (1, 20.0, SINKHORN_ITERS_MAX):
        forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
