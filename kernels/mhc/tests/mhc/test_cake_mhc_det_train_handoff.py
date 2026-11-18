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

The pre handoff and the standalone collapse of ``cake_mhc_det_train`` (hidden 5120, ``rms_eps`` 1e-20, ``hc_eps``
1e-6, 20 Sinkhorn iterations in every case): the cases of ``tests/experimental/test_cake_mhc_det_train_handoff.py``
under this package's names, and its bit promise.

A sublayer collapses its streams with the incoming ``pre_in`` (``U = sum_i pre_in_i X_i``) and hands its own ``pre`` on
as ``pre_next``; its backward takes ``dpre_next`` (the gradient of ``pre_next`` from its consumer) for the coefficient
VJP and returns ``dpre_in = sum_d dU_d X_id``.  The final collapse ``(X, pre_in) -> U`` and its VJP run alone.

Every floating output is judged against the naive BF16 tier over 32 inputs (``tests/cake_naive.py``), and ``dpre_in``
also against the issue's criterion: its relative L2 error at most 2 x that of the IEEE-FP32 evaluation of the
reference, median and largest over the draws.  This package promises its bits for a given token count on a given
device: every repeated launch (fresh outputs and default workspaces, or NaN-filled caller outputs and poisoned caller
workspaces), the autograd entry points, a checkpointed run and a CUDA-graph replay give the same bits as the first
launch.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from tests.cake.test_helpers.cake_mhc_handoff import (
    CONSTANTS,
    HIDDEN,
    MUTATIONS,
    PRE_COLUMNS,
    SUBLAYER_OUTPUTS,
    Api,
    Fp32Gate,
    assert_equal,
    autograd_sublayer,
    chain_expected_and_naive,
    chain_inputs,
    chain_names,
    chain_package,
    collapse_expected_and_naive,
    make_inputs,
    nan_outputs,
    sublayer_forward_backward,
)
from tests.cake.test_helpers.cake_mhc_naive import (
    expected_and_naive,
    judge_draws,
    package_outputs,
)
from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import DRAWS
from torch.utils.checkpoint import checkpoint

KEY = "cake.cake_mhc_det_train"
pkg = cake_package(KEY)
loader = cake_package(KEY, "cake_jit")
nd_loader = cake_package("cake.cake_mhc_nondet_train", "cake_jit")
api = Api(pkg, loader, "det")

BF16, F32 = torch.bfloat16, torch.float32
STREAMS, MAPPING_ROWS = 4, 24
SEED = 8900
# A1 / A9: T = 1, small, irregular and the issue's benchmark points.
T_ACCURACY = (1, 7, 65, 257, 1000, 2049, 4096, 16263, 16384, 32768, 65536)
T_COLLAPSE = (1, 7, 257, 2049, 16384, 65536)
_PARAMETER = ("dfn", "dscale", "dbase")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return torch.cuda.is_available() and pkg.is_cake_mhc_det_train_handoff_supported(
        _device(), HIDDEN
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the pre handoff of cake_mhc_det_train needs an SM100 / SM103 / SM107 device (hidden 5120)",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


def _arch() -> str:
    return loader.arch_for_capability(
        tuple(torch.cuda.get_device_capability(_device()))
    )


def _sm_count() -> int:
    return loader.sm_count(_device().index)


def _poison(fwd_ws: torch.Tensor, bwd_ws: torch.Tensor, T: int) -> None:
    """Every region of both workspaces but the ticket words set to 0xFF."""
    for kind, ws in (("fwd", fwd_ws), ("bwd", bwd_ws)):
        layout = loader.workspace_layout(
            kind, T, device_index=_device().index, hidden=HIDDEN
        )
        for name, region in layout["regions"].items():
            if name != "ticket":
                ws[region["offset"] : region["offset"] + region["bytes"]].fill_(0xFF)


def _workspaces(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        pkg.cake_mhc_det_train_forward_workspace(T, _device(), hidden=HIDDEN),
        pkg.cake_mhc_det_train_backward_workspace(T, _device(), hidden=HIDDEN),
    )


class _Strict:
    """``torch.use_deterministic_algorithms(True)`` with warnings as errors; the process-wide flags restored after."""

    def __enter__(self):
        self.prev = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.warnings = warnings.catch_warnings()
        self.warnings.__enter__()
        warnings.simplefilter("error")
        torch.use_deterministic_algorithms(True)
        return self

    def __exit__(self, *exc):
        torch.use_deterministic_algorithms(self.prev[0], warn_only=self.prev[1])
        self.warnings.__exit__(*exc)
        return False


def _snapshot(out: dict) -> dict:
    return {
        k: v.detach().clone() for k, v in out.items() if isinstance(v, torch.Tensor)
    }


def _assert_pre_next(res: dict, label: str) -> None:
    """``pre_next`` is the statistics record's ``pre`` (the coefficient pass's FP32 value), bit for bit."""
    lo, hi = PRE_COLUMNS
    assert res["pre_next"].dtype == F32 and tuple(res["pre_next"].shape) == (
        res["stats"].shape[0],
        STREAMS,
    ), label
    assert torch.equal(res["pre_next"], res["stats"][:, lo:hi]), (
        f"{label}: pre_next is not the statistics' pre"
    )


def _judge_sublayer(
    make, record_property, *, seed: int, launches: int = 2, check=None, prefix: str = ""
) -> None:
    """One sublayer per draw through the low-level entry points (``make(s)``: the draw's inputs), ``launches``
    launches: even ones on the default workspaces with fresh outputs, odd ones on poisoned caller workspaces with
    NaN-filled caller outputs, every launch bit for bit the first.  Every output of :data:`SUBLAYER_OUTPUTS` is judged
    against the naive tier, ``pre_next`` equals the statistics' ``pre``, and ``dpre_in`` meets the FP32 gate.
    ``check(i, outs, expected)`` runs on every draw."""
    gate = Fp32Gate(prefix + "dpre_in")

    def case(j: int, s: int):
        i = make(s)
        T = int(i["X"].shape[0])
        fwd_ws, bwd_ws = _workspaces(T)
        outs = []
        for launch in range(launches):
            kwargs = {}
            if launch % 2:
                _poison(fwd_ws, bwd_ws, T)
                kwargs = {
                    "fwd_ws": fwd_ws,
                    "bwd_ws": bwd_ws,
                    "outputs": nan_outputs(T, _device()),
                }
            res = sublayer_forward_backward(api, i, **kwargs)
            _assert_pre_next(res, f"{prefix}T={T} launch {launch}")
            outs.append(_snapshot(package_outputs(res)))
            assert_equal(
                outs[-1],
                outs[0],
                tuple(outs[0]),
                f"{prefix}T={T} launch {launch} vs launch 0",
            )
        expected, naive = expected_and_naive(i, SUBLAYER_OUTPUTS)
        for out in outs:
            gate.add(j, out["dpre_in"], i["X"], i["dU"], expected["dpre_in"])
        if check is not None:
            check(i, outs, expected)
        return outs, expected, naive

    judge_draws(case, record_property, seed=seed, prefix=prefix)
    gate.check(record_property, DRAWS)


# ----------------------------------------------------------------------------------------------------------------
# A1 / A2 / A9: accuracy of one sublayer and of the standalone collapse


@pytest.mark.parametrize("T", T_ACCURACY)
def test_handoff_matches_fp64(T: int, record_property) -> None:
    """One handoff sublayer at ``T`` tokens: ``U``, ``Y``, ``dB``, ``dX``, the statistics, the gradient statistics,
    the parameter gradients, ``pre_next`` and ``dpre_in`` judged against the naive BF16 tier over 32 inputs (default
    and poisoned caller workspaces); ``dpre_in`` within 2 x the IEEE-FP32 evaluation's error."""
    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s),
        record_property,
        seed=SEED + T,
    )


@pytest.mark.parametrize("T", T_COLLAPSE)
def test_collapse_matches_fp64(T: int, record_property) -> None:
    """The standalone collapse (``collapse_forward`` / ``collapse_backward``, and the autograd ``collapse``): ``U``,
    ``dX`` and ``dpre_in`` judged against the naive BF16 tier over 32 inputs; ``dpre_in`` within 2 x the IEEE-FP32
    evaluation's error."""
    gate = Fp32Gate()

    def case(j: int, s: int):
        i = make_inputs(T, device=_device(), seed=s)
        U = pkg.cake_mhc_det_train_collapse_forward(i["X"], i["pre_in"])
        dX, dpre_in = pkg.cake_mhc_det_train_collapse_backward(
            i["X"], i["pre_in"], i["dU"]
        )
        low = {"U": U, "dX": dX, "dpre_in": dpre_in}
        X = i["X"].clone().requires_grad_(True)
        pre_in = i["pre_in"].clone().requires_grad_(True)
        Ua = pkg.cake_mhc_det_train_collapse(X, pre_in)
        Ua.backward(i["dU"])
        auto = {"U": Ua.detach(), "dX": X.grad, "dpre_in": pre_in.grad}
        for out in (low, auto):
            assert (
                out["U"].dtype == BF16
                and out["dX"].dtype == BF16
                and out["dpre_in"].dtype == F32
            )
        expected, naive = collapse_expected_and_naive(i)
        for out in (low, auto):
            gate.add(j, out["dpre_in"], i["X"], i["dU"], expected["dpre_in"])
        return [low, auto], expected, naive

    judge_draws(case, record_property, seed=SEED + 100 + T)
    gate.check(record_property, DRAWS)


# ----------------------------------------------------------------------------------------------------------------
# A3: two sublayers from the one-hot start through the final collapse


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
def test_two_sublayer_handoff_chain_matches_fp64(
    T: int, save_stats: bool, checkpointed: bool, record_property
) -> None:
    """Two sublayers with distinct parameters and branches ``tanh(U W)``, from the one-hot start through the
    standalone final collapse, an asymmetric upstream gradient ``G`` on the final ``U``: the final ``U``, both
    ``pre_next`` and the gradients of ``X0`` and of both sublayers' parameters judged against the FP64 chain (the BF16
    chain the tier) over 32 inputs, each sublayer plain or in its own non-reentrant checkpoint; every gradient is
    non-zero."""

    def case(j: int, s: int):
        i = chain_inputs(T, device=_device(), seed=s)
        out = chain_package(api, i, save_stats=save_stats, checkpointed=checkpointed)
        names = chain_names(2)
        for k in names[3:]:
            assert bool(out[k].abs().max() > 0), f"{k} is zero"
        expected, naive = chain_expected_and_naive(i)
        return [{k: out[k] for k in names}], expected, naive

    judge_draws(case, record_property, seed=SEED + 300 + T)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_two_sublayer_chain_rejects_mutations(mutation: str) -> None:
    """The chain test detects the issue's three mutations: a collapse with the sublayer's own ``pre`` (the final
    ``U`` moves), a detached handoff and a coefficient VJP that ignores ``dpre_next`` (the parameter gradients move):
    the judge of :func:`test_two_sublayer_handoff_chain_matches_fp64` fails on each."""

    def case(j: int, s: int):
        i = chain_inputs(7, device=_device(), seed=s)
        out = chain_package(api, i, mutation=mutation)
        names = chain_names(2)
        expected, naive = chain_expected_and_naive(i)
        return [{k: out[k] for k in names}], expected, naive

    with pytest.raises(AssertionError) as caught:
        judge_draws(case, None, seed=SEED + 400)
    moved = "U_final" if mutation == "own_pre" else "dfn1"
    assert f"{moved}: misses the naive tier" in str(caught.value), str(caught.value)[
        :2000
    ]


# ----------------------------------------------------------------------------------------------------------------
# A4 - A8: the exact cases


def _assert_pre_lanes_zero(out: dict, label: str) -> None:
    """``dpre_next = 0``: the ``pre`` lanes of the parameter gradients are exactly zero (``dU`` does not reach the
    coefficients)."""
    assert torch.count_nonzero(out["dbase"][0:4]) == 0, f"{label}: dbase[0:4] non-zero"
    assert torch.count_nonzero(out["dscale"][0]) == 0, f"{label}: dscale[0] non-zero"
    assert torch.count_nonzero(out["dfn"][0:4]) == 0, f"{label}: dfn[0:4] non-zero"


@pytest.mark.parametrize("T", (257, 2049))
def test_handoff_zero_dY_with_dpre_next(T: int, record_property) -> None:
    """``dY = 0`` with a non-zero ``dpre_next``: every output judged against the naive tier over 32 inputs (``dB``,
    ``dpost`` and ``dC`` are zero in the oracle: the tier's zero rule holds them to exact zeros)."""
    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s, dY="zero"),
        record_property,
        seed=SEED + 500,
    )


@pytest.mark.parametrize("T", (257, 2049))
def test_handoff_zero_dpre_next_with_dY(T: int, record_property) -> None:
    """``dpre_next = 0`` with a non-zero ``dY``: every output judged against the naive tier over 32 inputs, and the
    ``pre`` lanes of ``dbase``, ``dscale`` and ``dfn`` exactly zero through the low-level and the autograd entry points
    (``pre_next`` reaching no loss: autograd hands no ``dpre_next``)."""

    def check(i, outs, expected):
        for n, out in enumerate(outs):
            _assert_pre_lanes_zero(out, f"launch {n}")
        X = i["X"].clone().requires_grad_(True)
        params = [i[k].clone().requires_grad_(True) for k in ("fn", "scale", "base")]
        U, _pre_next, handle = pkg.cake_mhc_det_train_handoff_prepare(
            X, i["pre_in"], *params, **CONSTANTS
        )
        Y = pkg.cake_mhc_det_train_finish(X, i["B"], handle)
        torch.autograd.backward([U, Y], [i["dU"], i["dY"]])
        _assert_pre_lanes_zero(
            dict(zip(_PARAMETER, (p.grad for p in params), strict=True)), "autograd"
        )

    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s, dpre_next="zero"),
        record_property,
        seed=SEED + 600,
        check=check,
    )


@pytest.mark.parametrize("T", (7, 2049))
def test_one_hot_pre_in_collapses_to_stream_0(T: int, record_property) -> None:
    """One-hot ``pre_in = [1, 0, 0, 0]`` (the first sublayer): ``U`` equals stream 0 of ``X`` bit for bit from the
    sublayer and from the standalone collapse, and every output is judged against the naive tier over 32 inputs."""

    def check(i, outs, expected):
        for out in outs:
            assert torch.equal(out["U"], i["X"][:, 0, :]), "sublayer U is not stream 0"
        assert torch.equal(
            pkg.cake_mhc_det_train_collapse_forward(i["X"], i["pre_in"]),
            i["X"][:, 0, :],
        )
        X = i["X"].clone().requires_grad_(True)
        U, _pre_next, handle = pkg.cake_mhc_det_train_handoff_prepare(
            X, i["pre_in"], i["fn"], i["scale"], i["base"]
        )
        assert torch.equal(U, i["X"][:, 0, :]), "autograd U is not stream 0"
        del handle

    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s, pre_in="one_hot"),
        record_property,
        seed=SEED + 700,
        check=check,
    )


@pytest.mark.parametrize("T", (7, 2049))
def test_zero_dU_gives_zero_dpre_in(T: int, record_property) -> None:
    """``dU = 0`` with non-zero ``dY`` and ``dpre_next``: ``dpre_in`` is exactly zero and ``dX`` is not (every output
    judged against the naive tier over 32 inputs); the standalone collapse VJP at ``dU = 0`` gives exact zeros."""

    def check(i, outs, expected):
        for out in outs:
            assert torch.count_nonzero(out["dpre_in"]) == 0, (
                "dpre_in is not exactly zero"
            )
            assert torch.count_nonzero(out["dX"]) > 0, "dX is zero"
        dX, dpre_in = pkg.cake_mhc_det_train_collapse_backward(
            i["X"], i["pre_in"], i["dU"]
        )
        assert torch.count_nonzero(dX) == 0 and torch.count_nonzero(dpre_in) == 0, (
            "standalone VJP at dU = 0"
        )

    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s, dU="zero"),
        record_property,
        seed=SEED + 800,
        check=check,
    )


# The cancellation band: |dX| <= CANCEL_FACTOR x the largest of its three contributions.
CANCEL_FACTOR = 2.0**-6


@pytest.mark.parametrize("T", (257, 2049))
@pytest.mark.parametrize("dY", ("zero", "random"))
def test_single_cast_cancellation_driven_by_dpre_next(
    T: int, dY: str, record_property
) -> None:
    """``dU := BF16(-dX_0 / pre_in_0)`` (``dX_0``: the FP64 ``dX`` of stream 0 at ``dU = 0``, its coefficient term
    driven by ``dpre_next``): the collapse term of stream 0 cancels the others, the inputs put elements of ``dX`` inside
    the cancellation band, and every output is judged against the naive tier over 32 inputs (one rounding of ``dX``)."""

    def check(i, outs, expected):
        total = expected["dX"]
        collapse = i["pre_in"].double().unsqueeze(-1) * i["dU"].double().unsqueeze(1)
        mix = torch.einsum("tij,tjd->tid", expected["C"], i["dY"].double())
        mapping = total - collapse - mix
        assert bool(mapping.abs().max() > 0), "the coefficient term is zero"
        biggest = torch.maximum(torch.maximum(collapse.abs(), mix.abs()), mapping.abs())
        assert int((total.abs() <= CANCEL_FACTOR * biggest).sum()) > 0, (
            "no element inside the cancellation band"
        )

    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s, mode="cancel", dY=dY),
        record_property,
        seed=SEED + 900,
        check=check,
    )


def _assert_tail_lanes(out: dict, expected: dict, lanes: int, label: str) -> None:
    """The saturated lanes ``0 .. lanes - 1`` of ``dbase``, the matching entries of ``dscale`` and rows of ``dfn`` are
    finite and non-zero where the FP64 tail gradient is non-zero."""
    groups = {
        "dbase": out["dbase"][:lanes],
        "dscale": out["dscale"][: lanes // 4],
        "dfn": out["dfn"][:lanes],
    }
    want = {
        "dbase": expected["dbase"][:lanes],
        "dscale": expected["dscale"][: lanes // 4],
        "dfn": expected["dfn"][:lanes],
    }
    for name, value in groups.items():
        assert torch.isfinite(value).all(), f"{label} {name}: non-finite"
        assert (want[name] != 0).all(), (
            f"{label} {name}: the FP64 tail gradient is zero; the case does not test the tail"
        )
        nonzero = (
            (value != 0).all() if value.ndim == 1 else (value != 0).any(dim=1).all()
        )
        assert nonzero, f"{label} {name}: a tail gradient is zero"


@pytest.mark.parametrize("T", (257, 2049))
@pytest.mark.parametrize("dY", ("random", "zero"))
def test_saturated_pre_tails_through_dpre_next(
    T: int, dY: str, record_property
) -> None:
    """``base[:8] = 20`` (saturated ``pre`` / ``post`` sigmoids) with ``dpre_next ~ N``: the tail gradients of the
    ``pre`` lanes (and, with ``dY``, the ``post`` lanes) are finite and non-zero through the low-level and the autograd
    entry points, and every output is judged against the naive tier over 32 inputs."""
    lanes = 8 if dY == "random" else 4

    def check(i, outs, expected):
        for n, out in enumerate(outs):
            _assert_tail_lanes(out, expected, lanes, f"launch {n}")
        auto = autograd_sublayer(api, i)
        _assert_tail_lanes(auto, expected, lanes, "autograd")

    _judge_sublayer(
        lambda s: make_inputs(T, device=_device(), seed=s, mode="tail", dY=dY),
        record_property,
        seed=SEED + 1000,
        check=check,
    )


def test_zero_row_of_x_takes_the_eps_path(record_property) -> None:
    """A zero row of ``X`` at ``rms_eps`` 1e-20 (``r = 1e10``): every output judged against the naive tier over 32
    inputs."""
    _judge_sublayer(
        lambda s: make_inputs(7, device=_device(), seed=s, zero_rows=(3,)),
        record_property,
        seed=SEED + 1100,
    )


# ----------------------------------------------------------------------------------------------------------------
# A9: zero tokens; A10: changing token counts


def test_handoff_zero_tokens_launch_nothing() -> None:
    """``T == 0``: the handoff entry points return empty ``U``, ``pre_next`` ``[0, 4]``, ``stats``, ``dX`` and
    ``dpre_in`` ``[0, 4]`` and zero the parameter gradients (left untouched with ``accumulate``); the standalone
    collapse and its VJP return empty outputs; the autograd entry points run; no deterministic-mode alert."""
    i = make_inputs(0, device=_device(), seed=SEED)
    with _Strict():
        res = sublayer_forward_backward(api, i)
        U = pkg.cake_mhc_det_train_collapse_forward(i["X"], i["pre_in"])
        dX, dpre_in = pkg.cake_mhc_det_train_collapse_backward(
            i["X"], i["pre_in"], i["dU"]
        )
        auto = autograd_sublayer(api, i)
    shapes = {
        "U": (0, HIDDEN),
        "pre_next": (0, STREAMS),
        "stats": (0, 64),
        "dX": (0, STREAMS, HIDDEN),
        "dpre_in": (0, 4),
    }
    for k, shape in shapes.items():
        assert tuple(res[k].shape) == shape, k
    for k in _PARAMETER:
        assert torch.equal(res[k], torch.zeros_like(res[k])), k
        assert torch.equal(auto[k], torch.zeros_like(auto[k])), k
    assert (
        tuple(U.shape) == (0, HIDDEN)
        and tuple(dX.shape) == (0, STREAMS, HIDDEN)
        and tuple(dpre_in.shape) == (0, 4)
    )
    buffers = {
        "dfn": torch.full((MAPPING_ROWS, STREAMS * HIDDEN), 3.0, device=_device()),
        "dscale": torch.full((3,), 5.0, device=_device()),
        "dbase": torch.full((MAPPING_ROWS,), 7.0, device=_device()),
    }
    out = pkg.cake_mhc_det_train_handoff_backward_finish(
        i["X"], i["dY"], i["dU"], i["pre_in"], i["dpre_next"], i["B"], i["fn"], i["scale"], i["base"],
        res["stats"], res["gstats"], **CONSTANTS, accumulate=True, **buffers,
    )  # fmt: skip
    for k, got in zip(_PARAMETER, out[2:], strict=True):
        assert got is buffers[k] and bool(
            (got == {"dfn": 3.0, "dscale": 5.0, "dbase": 7.0}[k]).all()
        ), k


def test_handoff_changing_rows_does_not_rebuild(record_property, monkeypatch) -> None:
    """A changed token count loads only records of the handoff routes' selection, nothing of the deterministic
    package, and calls back at earlier counts load nothing; ``dX``, ``dpre_in`` and ``dfn`` of each count are judged
    against the naive tier over 32 inputs."""
    loaded: list[str] = []
    load = loader.load

    def recording_load(name: str, arch: str):
        misses = load.cache_info().misses
        module = load(name, arch)
        if load.cache_info().misses != misses:
            loaded.append(name)
        return module

    monkeypatch.setattr(loader, "load", recording_load)
    nd_loaded = nd_loader.load.cache_info().misses
    counts = (64, 1, 65, 2049, 7, 255)
    for T in counts:

        def case(j: int, s: int, T: int = T):
            i = make_inputs(T, device=_device(), seed=s)
            res = sublayer_forward_backward(api, i)
            launch = {k: res[k] for k in ("dX", "dpre_in", "dfn")}
            expected, naive = expected_and_naive(i, launch)
            return [launch], expected, naive

        judge_draws(case, record_property, seed=SEED + 1200 + T, prefix=f"T={T} ")
    arch, sm = _arch(), _sm_count()
    kinds = ("fwd_prepare_handoff", "fwd_finish", "bwd_prepare", "bwd_finish_handoff")
    selected = {
        name
        for T in counts
        for kind in kinds
        for name in loader.selected_modules(arch, HIDDEN, kind, rows=T, sm_count=sm)
    }
    assert set(loaded) <= selected, (
        f"modules outside the routes' records: {sorted(set(loaded) - selected)}"
    )
    before = (load.cache_info().misses, nd_loader.load.cache_info().misses)
    assert before[1] == nd_loaded, (
        "a handoff call loaded a module of the nondeterministic package"
    )
    for T in reversed(counts):
        sublayer_forward_backward(
            api, make_inputs(T, device=_device(), seed=SEED + 5 + T)
        )
    assert (load.cache_info().misses, nd_loader.load.cache_info().misses) == before, (
        "a call rebuilt a module"
    )


def test_handoff_policy_strict_runs_every_entry_point(record_property) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (warnings are errors) every handoff and collapse entry point
    runs without an alert; the outputs are judged against the naive tier over 32 inputs."""

    def case(j: int, s: int):
        i = make_inputs(129, device=_device(), seed=s)
        with _Strict():
            res = _snapshot(package_outputs(sublayer_forward_backward(api, i)))
            auto = {k: v for k, v in autograd_sublayer(api, i).items() if k != "dB"}
            U = pkg.cake_mhc_det_train_collapse_forward(i["X"], i["pre_in"])
            dX, dpre_in = pkg.cake_mhc_det_train_collapse_backward(
                i["X"], i["pre_in"], i["dU"]
            )
        assert U.dtype == BF16 and dX.dtype == BF16 and dpre_in.dtype == F32
        expected, naive = expected_and_naive(i, SUBLAYER_OUTPUTS)
        return [res, auto], expected, naive

    judge_draws(case, record_property, seed=SEED + 1300)


# ----------------------------------------------------------------------------------------------------------------
# The autograd entry points


@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
@pytest.mark.parametrize("checkpointed", (False, True))
def test_handoff_autograd_matches_fp64(
    T: int, save_stats: bool, checkpointed: bool, record_property
) -> None:
    """``handoff_prepare`` / ``finish`` with the branch outside (``U`` and ``B`` taken as given), plain or in a
    non-reentrant checkpoint returning ``(U, Y, pre_next)``: ``U``, ``Y``, ``pre_next`` and the gradients of ``X``,
    ``pre_in``, the parameters and ``B`` (as ``dB``) judged against the naive tier over 32 inputs."""

    def case(j: int, s: int):
        i = make_inputs(T, device=_device(), seed=s)
        if checkpointed:
            X = i["X"].clone().requires_grad_(True)
            pre_in = i["pre_in"].clone().requires_grad_(True)
            params = [
                i[k].clone().requires_grad_(True) for k in ("fn", "scale", "base")
            ]
            B = i["B"].clone().requires_grad_(True)

            def region(X, pre_in, fn, scale, base, B):
                U, pre_next, handle = pkg.cake_mhc_det_train_handoff_prepare(
                    X, pre_in, fn, scale, base, save_stats=save_stats, **CONSTANTS
                )
                return U, pkg.cake_mhc_det_train_finish(X, B, handle), pre_next

            U, Y, pre_next = checkpoint(
                region, X, pre_in, *params, B, use_reentrant=False
            )
            torch.autograd.backward(
                [U, Y, pre_next], [i["dU"], i["dY"], i["dpre_next"]]
            )
            out = {
                "U": U.detach(), "Y": Y.detach(), "pre_next": pre_next.detach(), "dX": X.grad, "dpre_in": pre_in.grad,
                "dfn": params[0].grad, "dscale": params[1].grad, "dbase": params[2].grad, "dB": B.grad,
            }  # fmt: skip
        else:
            out = autograd_sublayer(api, i, save_stats=save_stats)
        expected, naive = expected_and_naive(i, out)
        return [out], expected, naive

    judge_draws(case, record_property, seed=SEED + 1400 + T)


def test_autograd_normalizes_the_incoming_gradients(record_property) -> None:
    """Autograd hands ``dU`` and ``dpre_next`` as zero-stride expansions (a loss ``U.sum()`` / ``pre_next.sum()``):
    the node materializes them, and the gradients are judged against the naive tier with ``dU = 1``, ``dpre_next =
    1`` over 32 inputs; a ``pre_next`` that reaches no loss is a zero ``dpre_next``."""

    def case(j: int, s: int):
        i = make_inputs(129, device=_device(), seed=s)
        i["dU"] = torch.ones_like(i["dU"])
        i["dpre_next"] = torch.ones_like(i["dpre_next"])
        X = i["X"].clone().requires_grad_(True)
        pre_in = i["pre_in"].clone().requires_grad_(True)
        params = [i[k].clone().requires_grad_(True) for k in ("fn", "scale", "base")]
        U, pre_next, handle = pkg.cake_mhc_det_train_handoff_prepare(
            X, pre_in, *params, **CONSTANTS
        )
        Y = pkg.cake_mhc_det_train_finish(X, i["B"], handle)
        loss = U.float().sum() + pre_next.sum() + (Y.float() * i["dY"].float()).sum()
        loss.backward()
        out = {
            "dX": X.grad, "dpre_in": pre_in.grad, "dfn": params[0].grad, "dscale": params[1].grad,
            "dbase": params[2].grad,
        }  # fmt: skip
        expected, naive = expected_and_naive(i, out)
        return [out], expected, naive

    judge_draws(case, record_property, seed=SEED + 1500)


# ----------------------------------------------------------------------------------------------------------------
# A12: checkpoint regions a pre_next crosses; A13: workspace and retained bytes


def _live_bytes() -> int | None:
    torch.cuda.synchronize()
    if torch.cuda.get_allocator_backend() != "native":
        return None
    return int(torch.cuda.memory_stats()["requested_bytes.all.current"])


def _assert_same_bits(got: list, want: list, label: str) -> None:
    """Two runs of this package: the same bits."""
    assert len(got) == len(want), label
    for index, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g is not None and g.dtype == w.dtype and g.shape == w.shape, (
            label,
            index,
        )
        assert torch.equal(g.view(torch.uint8), w.view(torch.uint8)), (
            f"{label} grad {index}: bits differ"
        )


@pytest.mark.parametrize("T", (129, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_checkpoint_across_pre_next(T: int, save_stats: bool, record_property) -> None:
    """Two layers of two handoff sublayers each, every layer one non-reentrant checkpoint region (the first
    sublayer's ``pre_next`` stays inside the region, the layer's last ``pre_next`` crosses into the next region),
    then the final collapse: the forward outputs (the final ``U`` and the crossing ``pre_next``) and every gradient
    equal the plain run's bit for bit, and the final ``U`` and all gradients are judged against the FP64 chain over
    32 inputs."""
    names = tuple(n for n in chain_names(4) if not n.startswith("pre"))

    def case(j: int, s: int):
        i = chain_inputs(T, device=_device(), seed=s, sublayers=4)
        plain = chain_package(api, i, save_stats=save_stats, region=2)
        ckpt = chain_package(api, i, save_stats=save_stats, checkpointed=True, region=2)
        assert torch.equal(ckpt["U_final"], plain["U_final"]), (
            "the final U differs from the plain run"
        )
        for a, b in zip(ckpt["pres"], plain["pres"], strict=True):
            assert torch.equal(a, b), "a crossing pre_next differs from the plain run"
        _assert_same_bits(
            [ckpt[k] for k in names[1:]], [plain[k] for k in names[1:]], f"T={T}"
        )
        expected, naive = chain_expected_and_naive(i)
        return (
            [{k: plain[k] for k in names}, {k: ckpt[k] for k in names}],
            expected,
            naive,
        )

    judge_draws(case, record_property, seed=SEED + 1600 + T)


@pytest.mark.parametrize("T", (129, 2049))
def test_checkpointed_layer_keeps_only_its_outputs(T: int) -> None:
    """After the forward of a checkpointed layer of two handoff sublayers, only the layer's outputs ``Y`` and the
    crossing ``pre_next`` stay alive (the internal ``pre_next`` and every statistic are released)."""
    i = chain_inputs(T, device=_device(), seed=SEED + 41, sublayers=2)
    X0 = i["X"].clone().requires_grad_(True)
    params = [[p.clone().requires_grad_(True) for p in ps] for ps in i["params"]]
    pre0 = torch.zeros(T, STREAMS, device=_device())
    pre0[:, 0] = 1.0

    def layer(X, pre_in, *flat):
        for k in range(0, len(flat), 4):
            fn, scale, base, W = flat[k : k + 4]
            U, pre_in, handle = pkg.cake_mhc_det_train_handoff_prepare(
                X, pre_in, fn, scale, base, **CONSTANTS
            )
            X = pkg.cake_mhc_det_train_finish(X, torch.tanh(U @ W), handle)
        return X, pre_in

    flat = [t for k in range(2) for t in (*params[k], i["W"][k])]
    before = _live_bytes()
    Y, pre_next = checkpoint(layer, X0, pre0, *flat, use_reentrant=False)
    if before is not None:
        kept = (
            _live_bytes()
            - before
            - Y.untyped_storage().nbytes()
            - pre_next.untyped_storage().nbytes()
        )
        assert kept == 0, f"{kept} bytes kept alive after a checkpointed layer"
    pkg.cake_mhc_det_train_collapse(Y, pre_next).backward(i["G"])
    assert torch.isfinite(X0.grad).all()


@pytest.mark.parametrize("T", (16384, 65536))
def test_handoff_workspace_and_retained_bytes_per_token(
    T: int, record_property
) -> None:
    """Per token (``record_property("handoff_bytes_per_token", ...)``): the forward / backward workspace bytes of the
    handoff routes (equal to the standard routes'), and the bytes a sublayer keeps alive after its forward beyond its
    outputs under ``save_stats`` True / False (equal to the standard connection's: ``pre_in`` is saved as an alias of
    the previous ``pre_next``), the ``pre_next`` kept until its consumer runs (16 B), and what the standalone collapse
    keeps beyond ``U`` (its saved ``X`` and ``pre_in`` are aliases: nothing)."""
    if _live_bytes() is None:
        pytest.skip("the retained bytes need the native caching allocator")
    i = make_inputs(T, device=_device(), seed=SEED + 77)
    report: dict[str, float] = {}
    for kind, name in (("fwd", "forward"), ("bwd", "backward")):
        nbytes = getattr(pkg, f"cake_mhc_det_train_{name}_workspace_bytes")(
            T, _device(), hidden=HIDDEN
        )
        handoff = loader.workspace_kind(
            f"{'fwd_prepare' if kind == 'fwd' else 'bwd_finish'}_handoff"
        )
        assert handoff == kind, (kind, handoff)
        report[f"{kind}_workspace"] = nbytes / T
    for save_stats in (True, False):
        kept = {}
        for route in ("standard", "handoff"):
            X = i["X"].clone().requires_grad_(True)
            pre_in = i["pre_in"].clone().requires_grad_(True)
            params = [
                i[k].clone().requires_grad_(True) for k in ("fn", "scale", "base")
            ]
            B = i["B"].clone().requires_grad_(True)
            before = _live_bytes()
            if route == "standard":
                U, handle = pkg.cake_mhc_det_train_prepare(
                    X, *params, save_stats=save_stats, **CONSTANTS
                )
                outputs = [U]
            else:
                U, pre_next, handle = pkg.cake_mhc_det_train_handoff_prepare(
                    X, pre_in, *params, save_stats=save_stats, **CONSTANTS
                )
                outputs = [U, pre_next]
            Y = pkg.cake_mhc_det_train_finish(X, B, handle)
            kept[route] = (
                _live_bytes()
                - before
                - sum(t.untyped_storage().nbytes() for t in (*outputs, Y))
            )
            del U, Y, handle, outputs
        assert kept["handoff"] == kept["standard"], (save_stats, kept)
        report[f"retained_save_stats_{save_stats}"] = kept["handoff"] / T
    report["pre_next_until_consumed"] = STREAMS * 4
    before = _live_bytes()
    X = i["X"].clone().requires_grad_(True)
    pre_in = i["pre_in"].clone().requires_grad_(True)
    base_bytes = _live_bytes() - before
    U = pkg.cake_mhc_det_train_collapse(X, pre_in)
    collapse_kept = _live_bytes() - before - base_bytes - U.untyped_storage().nbytes()
    assert collapse_kept == 0, (
        f"the standalone collapse keeps {collapse_kept} bytes beyond U"
    )
    report["collapse_retained_beyond_inputs"] = 0.0
    report["collapse_saved_aliases_X_pre_in"] = STREAMS * HIDDEN * 2 + STREAMS * 4
    record_property(
        "handoff_bytes_per_token", {k: round(float(v), 3) for k, v in report.items()}
    )


# ----------------------------------------------------------------------------------------------------------------
# The bit promise


@pytest.mark.parametrize("T", (129, 16263))
def test_handoff_outputs_are_deterministic(T: int) -> None:
    """Three runs of the direct entry points (poisoned caller workspaces and NaN-filled caller outputs before each),
    of the autograd entry points (``save_stats`` both ways) and of the standalone collapse on the same inputs give the
    same bits; the autograd gradients equal the direct ones bit for bit."""
    i = make_inputs(T, device=_device(), seed=SEED + 3)
    fwd_ws, bwd_ws = _workspaces(T)
    direct, auto, collapse = [], [], []
    for _ in range(3):
        _poison(fwd_ws, bwd_ws, T)
        direct.append(
            _snapshot(
                sublayer_forward_backward(
                    api,
                    i,
                    fwd_ws=fwd_ws,
                    bwd_ws=bwd_ws,
                    outputs=nan_outputs(T, _device()),
                )
            )
        )
        for save_stats in (True, False):
            auto.append(_snapshot(autograd_sublayer(api, i, save_stats=save_stats)))
        U = pkg.cake_mhc_det_train_collapse_forward(i["X"], i["pre_in"])
        dX, dpre_in = pkg.cake_mhc_det_train_collapse_backward(
            i["X"], i["pre_in"], i["dU"]
        )
        collapse.append({"U": U, "dX": dX, "dpre_in": dpre_in})
    for n in range(1, 3):
        assert_equal(direct[n], direct[0], tuple(direct[0]), f"direct run {n}")
        assert_equal(
            collapse[n], collapse[0], ("U", "dX", "dpre_in"), f"collapse run {n}"
        )
    for n in range(1, len(auto)):
        assert_equal(auto[n], auto[0], tuple(auto[0]), f"autograd run {n}")
    shared = ("U", "Y", "pre_next", "dX", "dpre_in", "dfn", "dscale", "dbase")
    assert_equal(auto[0], direct[0], shared, "autograd vs direct")


def test_captured_handoff_replays_the_eager_bits() -> None:
    """One handoff sublayer through the direct entry points captured in a CUDA graph: a replay on new inputs (copied
    into the static inputs) gives the bits of an eager call on those inputs."""
    T = 2049
    statics = make_inputs(T, device=_device(), seed=SEED + 4)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            sublayer_forward_backward(api, statics)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = sublayer_forward_backward(api, statics)
    for seed in (SEED + 5, SEED + 6):
        new = make_inputs(T, device=_device(), seed=seed)
        with torch.no_grad():
            for k in (
                "X",
                "fn",
                "scale",
                "base",
                "B",
                "dY",
                "dU",
                "pre_in",
                "dpre_next",
            ):
                statics[k].copy_(new[k])
        graph.replay()
        torch.cuda.synchronize()
        eager = sublayer_forward_backward(api, new)
        assert_equal(
            _snapshot(out), _snapshot(eager), tuple(eager), f"replay seed {seed}"
        )
