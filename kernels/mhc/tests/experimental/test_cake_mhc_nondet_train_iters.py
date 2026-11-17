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

Tests of the run-time Sinkhorn iteration count and the epsilons of the Cake
four-stream mHC training package
``cake_pplx._backends.cake_mhc_nondet_train`` (SM100 / SM103 /
SM107), in
the form of ``test_cake_mhc_det_train_iters.py``.

``sinkhorn_iters`` is an integer from 1 to ``SINKHORN_ITERS_MAX`` (64).  The
package's backward launches its copy of the deterministic token-owner pass,
whose run-time-count record takes every count but the compiled 20 (the per-CTA
form: at another count the column slab folds the per-CTA parameter rows, also
inside ``BWD_ROW_BANDS``, as the deterministic reduction does); the atomic fn
split and column-slab pass do not depend on the count.  The package makes no
bit promise: no output is compared bit for bit with another launch, another
path or the deterministic package.  Every judged output is held with
``tests/cake_naive.py`` against the naive BF16 tier over 32 inputs: its error
against the float64 oracle at the same count and epsilons
(``tests/cake/test_helpers/cake_mhc_naive.py``), in the L2 and the max norm,
median and max over the 32 inputs, is at most 1.005 times the tier's, and zero
where the tier's is zero.  This module checks:

* the numerics at counts 1, 2, 5, 10, 19, 21, 30, 50 and 64 on T = 1, 7, 257,
  1024 and 16263: every output of each of 5 launches judged; where the FP64
  ``dfn`` of the count and of 20 lie apart, every launch's ``dfn`` is nearer
  the count's;
* the paths: at each count the outputs of the package's forward prepare,
  statistics replay, residual mixing, backward prepare and backward finish, and
  the outputs and gradients of the autograd functions (statistics saved: the
  backward path; replayed: the checkpoint path), each judged;
* that changing the count in one process loads no module after the first
  run-time-count call, the count checks, and non-default epsilons with a
  run-time count.

The cases above run at ``H = 4096``;
``test_counts_match_fp64_at_every_hidden_size`` and
``test_paths_match_fp64_at_every_hidden_size`` repeat the numerics and the
paths at the counts 1, 7, 20 and 64 at hidden 5120 and 7168 where the device's
architecture routes them (SM100 / SM103 / SM107).
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.cake.test_helpers.cake_mhc_naive import (
    OUTPUTS,
    case_oracle,
    expected_and_naive,
    judge_draws,
    package_outputs,
    stats_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake.cake_mhc_nondet_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
# The deterministic package: its count constants and its loader (a changed count loads no module of either package).
det = cake_package("cake.cake_mhc_det_train")
det_loader = cake_package("cake.cake_mhc_det_train", "cake_jit")
# The package's entry points.
forward_prepare = pkg.cake_mhc_nondet_train_forward_prepare
forward_stats = pkg.cake_mhc_nondet_train_forward_stats
forward_finish = pkg.cake_mhc_nondet_train_forward_finish
backward_prepare = pkg.cake_mhc_nondet_train_backward_prepare
backward_finish = pkg.cake_mhc_nondet_train_backward_finish

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
STREAMS, HIDDEN, MAPPING_ROWS = 4, 4096, 24
HIDDEN_SIZES = (4096, 5120, 7168)
RMS_EPS, HC_EPS, SINKHORN_ITERS, SINKHORN_ITERS_MAX = 1e-5, 1e-6, 20, 64
COUNTS = (1, 2, 5, 10, 19, 21, 30, 50, 64)
T_COUNTS = (1, 7, 257, 1024, 16263)
HIDDEN_COUNTS = (1, 7, 20, 64)  # the counts of the hidden sizes other than 4096
# The relative L2 distance between the FP64 dfn of a count and of 20 above which the kernel's dfn must lie nearer its
# own count's (below it a chain at its fixed point leaves nothing to tell apart).
COUNT_APART = 4e-5
SEED = 2032
# Launches of this package's backward per draw: each is judged (the judge keeps the larger error of a draw).
REPEATS = 5
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64
_OUTPUTS = ("dX", "dfn", "dscale", "dbase")


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
    # Both packages' libraries of the local architecture (a changed count must load no module of either); a prebuilt
    # or cached library loads without compiling.
    if _supported():
        arch = _arch()
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
            spec.build_and_load()
    yield


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _require_hidden(hidden: int) -> int:
    """``hidden`` if the device's architecture routes every kind of it, else the case is skipped."""
    arch = _arch()
    sizes = tuple(loader.supported_hidden_sizes(arch)) if arch else ()
    if hidden not in sizes:
        pytest.skip(
            f"hidden {hidden} has no exported route on {arch} (exported: {sizes})"
        )
    return hidden


# ----------------------------------------------------------------------------------------------------------------
# Inputs, launches and the judge


def _constants(i: dict) -> dict:
    """The launch constants of a case: the epsilons and the Sinkhorn count it carries (default: the model's)."""
    return {k: i[k] for k in ("rms_eps", "hc_eps", "sinkhorn_iters") if k in i}


def _make_inputs(T: int, *, seed: int = SEED, hidden: int = HIDDEN) -> dict:
    g = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = F32) -> torch.Tensor:
        return torch.randn(*shape, generator=g, device=_device(), dtype=F32).to(dtype)

    return {
        "X": randn(T, STREAMS, hidden, dtype=BF16), "fn": (randn(MAPPING_ROWS, STREAMS * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3), "base": 0.1 * randn(MAPPING_ROWS), "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, STREAMS, hidden, dtype=BF16), "dU": randn(T, hidden, dtype=BF16),
    }  # fmt: skip


def _forward(i: dict) -> dict:
    """The forward prepare at the case's constants and the backward prepare: the records ``backward_finish`` takes."""
    _U, stats = forward_prepare(i["X"], i["fn"], i["scale"], i["base"], **_constants(i))
    _dB, gstats = backward_prepare(i["dY"], i["B"], i["X"], stats)
    return {"stats": stats, "gstats": gstats}


def _nd_backward(i: dict, prep: dict, **kwargs) -> dict:
    out = backward_finish(
        i["X"], i["dY"], i["dU"], i["B"], i["fn"], i["scale"], i["base"], prep["stats"], prep["gstats"],
        **{**_constants(i), **kwargs},
    )  # fmt: skip
    return dict(zip(_OUTPUTS, out, strict=True))


def _snapshot(out: dict) -> dict:
    return {k: v.detach().clone() for k, v in out.items()}


def _nd_repeats(i: dict, prep: dict, n: int = REPEATS) -> list[dict]:
    """``n`` launches of this package's backward, each output of its dtype (``dX`` BF16, the parameter gradients
    FP32)."""
    outs = []
    for repeat in range(n):
        out = _snapshot(_nd_backward(i, prep))
        assert out["dX"].dtype == BF16, f"dX repeat {repeat}"
        for k in ("dfn", "dscale", "dbase"):
            assert out[k].dtype == F32, f"{k} repeat {repeat}"
        outs.append(out)
    return outs


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a.double() - b.double()).norm().item()
    den = b.double().norm().item()
    return num / den if den > 0 else (0.0 if num == 0 else math.inf)


def _check(
    T: int,
    iters: int,
    record_property,
    *,
    seed: int = SEED,
    hidden: int = HIDDEN,
    names=_OUTPUTS,
    prefix: str = "",
) -> None:
    """The backward outputs ``names`` of ``REPEATS`` launches at the count, judged against the naive BF16 tier at the
    count over 32 inputs."""

    def case(j: int, s: int):
        i = _make_inputs(T, seed=s, hidden=hidden)
        i["sinkhorn_iters"] = iters
        outs = [{k: o[k] for k in names} for o in _nd_repeats(i, _forward(i))]
        expected, naive = expected_and_naive(i, names)
        return outs, expected, naive

    judge_draws(case, record_property, seed=seed, prefix=prefix)


def _connection(i: dict, iters: int, save_stats: bool) -> tuple[dict, dict, dict]:
    """The autograd functions at the count around the residual branch ``B = U + B_ext`` (``dB/dU = 1``: the branch
    backward hands ``dB`` on as ``dU``) and the loss ``sum(Y dY)``: ``U``, ``Y``, ``U``'s gradient as ``dB`` and the
    gradients of the leaves, with the oracle's and the naive tier's values on the branch output and the ``dU`` the
    graph produced."""
    leaves = [i[k].clone().requires_grad_(True) for k in ("X", "fn", "scale", "base")]
    U, handle = pkg.cake_mhc_nondet_train_prepare(
        *leaves, sinkhorn_iters=iters, save_stats=save_stats
    )
    U.retain_grad()
    B = (U.float() + i["B"].float()).to(BF16)
    Y = pkg.cake_mhc_nondet_train_finish(leaves[0], B, handle)
    (Y.float() * i["dY"].float()).sum().backward()
    outputs = {
        "U": U.detach(), "Y": Y.detach(), "dB": U.grad, "dX": leaves[0].grad, "dfn": leaves[1].grad,
        "dscale": leaves[2].grad, "dbase": leaves[3].grad,
    }  # fmt: skip
    expected, naive = expected_and_naive(
        {**i, "B": B.detach(), "dU": U.grad.detach()}, outputs
    )
    return outputs, expected, naive


# ----------------------------------------------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", T_COUNTS)
def test_counts_match_fp64(T: int, iters: int, record_property) -> None:
    _check_counts(T, iters, HIDDEN, record_property)


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
    # The count reaches the kernels: where the FP64 dfn of this count and of 20 lie apart, every launch's dfn is
    # nearer this count's (test_cake_mhc_det_train_iters.py's check of the record's C, on the output of this
    # package's route). A chain at its fixed point leaves the two FP64 gradients closer than that and nothing to tell
    # apart; the values alone are judged there.
    i = _make_inputs(T, hidden=hidden)
    i["sinkhorn_iters"] = iters
    outs = _nd_repeats(i, _forward(i))
    ref_n = case_oracle(i)["dfn"]
    ref_20 = case_oracle({**i, "sinkhorn_iters": SINKHORN_ITERS})["dfn"]
    if _rel_l2(ref_n, ref_20) > COUNT_APART:
        for repeat, o in enumerate(outs):
            assert _rel_l2(o["dfn"], ref_n) < _rel_l2(o["dfn"], ref_20), (
                f"dfn at sinkhorn_iters={iters} H={hidden} launch {repeat} is nearer the count of 20's"
            )


@pytest.mark.parametrize("iters", COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
def test_paths_match_fp64(T: int, iters: int, record_property) -> None:
    """At each count, over 32 inputs: the outputs of the package's forward prepare, statistics replay (under the name
    prefix ``replay``), residual mixing, backward prepare and backward finish, and the outputs and gradients of the
    autograd functions with the statistics saved (the backward path) and replayed (the checkpoint path; each under the
    name prefix ``autograd save_stats=<bool>``), each judged against the naive BF16 tier at the count, the oracle and
    the tier on each path's branch output and ``dU``."""
    _check_paths(T, iters, HIDDEN, record_property)


@pytest.mark.parametrize("iters", HIDDEN_COUNTS)
@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("hidden", HIDDEN_SIZES[1:], ids=lambda h: f"H{h}")
def test_paths_match_fp64_at_every_hidden_size(
    hidden: int, T: int, iters: int, record_property
) -> None:
    """``test_paths_match_fp64`` at hidden 5120 and 7168 and the counts 1, 7, 20 and 64."""
    _check_paths(T, iters, _require_hidden(hidden), record_property)


def _check_paths(T: int, iters: int, hidden: int, record_property) -> None:
    def case(j: int, s: int):
        i = _make_inputs(T, seed=s, hidden=hidden)
        i["sinkhorn_iters"] = iters
        X, fn, scale, base = (i[k] for k in ("X", "fn", "scale", "base"))
        U, stats = forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        replay = forward_stats(X, fn, scale, base, sinkhorn_iters=iters)
        B = (U.float() + i["B"].float()).to(
            BF16
        )  # the branch: dB/dU = 1, so its backward hands dB on as dU
        Y = forward_finish(X, B, stats)
        dB, gstats = backward_prepare(i["dY"], B, X, stats)
        dX, dfn, dscale, dbase = backward_finish(
            X, i["dY"], dB, B, fn, scale, base, stats, gstats, sinkhorn_iters=iters
        )
        outputs = package_outputs(
            {
                "U": U,
                "stats": stats,
                "Y": Y,
                "dB": dB,
                "gstats": gstats,
                "dX": dX,
                "dfn": dfn,
                "dscale": dscale,
                "dbase": dbase,
            }  # fmt: skip
        )
        e, t = expected_and_naive({**i, "B": B, "dU": dB}, OUTPUTS)
        expected, naive = dict(e), dict(t)
        for k, v in stats_outputs(replay).items():
            outputs["replay " + k] = v
            expected["replay " + k], naive["replay " + k] = e[k], t[k]
        for save_stats in (True, False):
            prefix = f"autograd save_stats={save_stats} "
            for d, part in zip(
                (outputs, expected, naive),
                _connection(i, iters, save_stats),
                strict=True,
            ):
                d.update({prefix + k: v for k, v in part.items()})
        return [outputs], expected, naive

    judge_draws(case, record_property, seed=SEED + 41)


def test_changing_count_does_not_rebuild(record_property) -> None:
    """The count is a launch argument of the run-time-count programs: after one call at a non-default count, calls at
    other counts (and back at the default) load no module of either package, and ``dX`` and ``dfn`` at each count are
    judged against the naive BF16 tier at that count over 32 inputs.  The calls of the counts run at 255 tokens,
    whose route may select records of other token-count bands than 64 tokens' (a mapping-pass ``xn`` form):
    one call at 255 tokens at the default count and one at a non-default count load them first, so the count of
    loads compares calls that differ in their Sinkhorn count alone, whatever the earlier tests of the process
    loaded."""
    _check(64, SINKHORN_ITERS, record_property, prefix="T=64 sinkhorn_iters=20 ")
    _check(64, 3, record_property, prefix="T=64 sinkhorn_iters=3 ")
    for iters in (SINKHORN_ITERS, 3):
        i = {**_make_inputs(255, seed=SEED + 3), "sinkhorn_iters": iters}
        _nd_backward(i, _forward(i))
    loaded = (loader.load.cache_info().misses, det_loader.load.cache_info().misses)
    for iters in (1, 2, 7, 33, 64, SINKHORN_ITERS, 11):
        _check(
            255, iters, record_property, seed=SEED + iters, names=("dX", "dfn"), prefix=f"sinkhorn_iters={iters} "
        )  # fmt: skip
    assert (
        loader.load.cache_info().misses,
        det_loader.load.cache_info().misses,
    ) == loaded, "a changed sinkhorn_iters loaded a module"


def test_count_validation(record_property) -> None:
    """The package checks the count like the deterministic package: a non-integer, non-positive or bool count is a
    ``ValueError``, one above ``SINKHORN_ITERS_MAX`` a ``NotImplementedError``, on ``backward_finish``,
    ``forward_prepare`` and ``cake_mhc_nondet_train_prepare``; at the counts 1, 20.0 and ``SINKHORN_ITERS_MAX`` the
    backward outputs are judged against the naive BF16 tier at the count over 32 inputs."""
    assert det.SHIPPED_SINKHORN_ITERS == pkg.SHIPPED_SINKHORN_ITERS == (SINKHORN_ITERS,)
    assert det.SINKHORN_ITERS_MAX == loader.SINKHORN_ITERS_MAX == SINKHORN_ITERS_MAX
    i = _make_inputs(8, seed=SEED + 9)
    X, fn, scale, base = i["X"], i["fn"], i["scale"], i["base"]
    prep = _forward(i)
    for iters in (0, -1, 2.5, 20.5, True, float("inf"), float("nan"), "20"):
        with pytest.raises(ValueError):
            _nd_backward(i, prep, sinkhorn_iters=iters)
        with pytest.raises(ValueError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(ValueError):
            pkg.cake_mhc_nondet_train_prepare(X, fn, scale, base, sinkhorn_iters=iters)
    for iters in (SINKHORN_ITERS_MAX + 1, 1000):
        with pytest.raises(NotImplementedError):
            _nd_backward(i, prep, sinkhorn_iters=iters)
        with pytest.raises(NotImplementedError):
            forward_prepare(X, fn, scale, base, sinkhorn_iters=iters)
        with pytest.raises(NotImplementedError):
            pkg.cake_mhc_nondet_train_prepare(X, fn, scale, base, sinkhorn_iters=iters)
    for iters in (1, 20.0, SINKHORN_ITERS_MAX):

        def case(j: int, s: int, iters=iters):
            c = {**_make_inputs(8, seed=s), "sinkhorn_iters": iters}
            got = _snapshot(_nd_backward(c, _forward(c)))
            expected, naive = expected_and_naive(c, _OUTPUTS, iters=int(iters))
            return [got], expected, naive

        judge_draws(
            case, record_property, seed=SEED + 9, prefix=f"sinkhorn_iters={iters!r} "
        )


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize(
    ("iters", "rms_eps", "hc_eps"),
    (
        (7, RMS_EPS, HC_EPS),
        (64, RMS_EPS, HC_EPS),
        (SINKHORN_ITERS, 2e-5, 2e-6),
        (7, 1e-6, 1e-5),
    ),
)
def test_run_time_count_and_epsilons(
    T: int, iters: int, rms_eps: float, hc_eps: float, record_property
) -> None:
    """A run-time Sinkhorn count (the deterministic token-owner pass's run-time-count record) and non-default
    epsilons of this package's backward: every output of each of 5 launches judged against the naive BF16 tier at
    the same constants over 32 inputs."""

    def case(j: int, s: int):
        i = _make_inputs(T, seed=s)
        i.update(sinkhorn_iters=iters, rms_eps=rms_eps, hc_eps=hc_eps)
        outs = _nd_repeats(i, _forward(i))
        expected, naive = expected_and_naive(i, _OUTPUTS)
        return outs, expected, naive

    judge_draws(case, record_property, seed=SEED + 41)
