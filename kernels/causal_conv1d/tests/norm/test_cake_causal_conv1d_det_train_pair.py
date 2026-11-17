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

Tests of the opt-in FP32 (hi, lo) pair ``dweight`` / ``dbias`` sums of the deterministic package
(``dweight_sum="fp32_pair"``, the pair routes ``bwd_fp32g_pair`` / ``bwd_bias_fp32g_pair``).

Every floating output is judged as in ``test_cake_causal_conv1d_det_train.py``: against the naive BF16 tier over 32
independent random inputs, by its error against the float64 oracle in the L2 and the max norm (median and largest
error at most 1.005 times the tier's).  The pair routes are deterministic: ``dx``, ``dweight`` and ``dbias`` are
bitwise equal over repeated launches, a NaN-poisoned workspace and a fresh one.  ``dx`` is the twin route's (the pair
programs change the ``dweight`` / ``dbias`` sums only).  The suite also covers the program a call selects (the pair
of the program the twin route selects, on the twin's launch plan), the opt-in ``bwd_grid_mult``, CUDA Graph capture
and replay, the autograd entry and the refusals: ``"fp32_pair"`` without ``silu_grad_preactivation="fp32"`` and an
unknown ``dweight_sum`` raise ``ValueError`` before any launch.  The default (``dweight_sum="fp32"``) never selects a
pair route.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_causal_conv1d_oracle import assert_judged, problem
from tests.norm.test_cake_causal_conv1d_det_train import (
    BF16,
    F32,
    PRECISION_CASES,
    ROUTE_CASES,
    ROUTE_SETTINGS,
    SEED,
    WIDTH,
    _arch,
    _bias_input,
    _capability,
    _case_id,
    _device,
    _dtype_id,
    _inputs,
    _require_route,
    _supported,
    cake_build,
    loader,
    pkg,
)

# pair route -> the route it pairs (its twin)
PAIR_KINDS = {"bwd_fp32g_pair": "bwd_fp32g", "bwd_bias_fp32g_pair": "bwd_bias_fp32g"}
PAIR = "fp32_pair"
GRAPH_REPLAYS = 3

pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_causal_conv1d_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        for spec in loader.specs_for_arch(loader.arch_for_capability(_capability())):
            spec.build_and_load()
    yield


def _require_pair(kind: str) -> None:
    _require_route(PAIR_KINDS[kind])
    _require_route(kind)


def _settings(kind: str) -> dict:
    return ROUTE_SETTINGS[PAIR_KINDS[kind]]


def _route_inputs(S: int, C: int, docs, kind: str, *, key: int = 0, seed: int = SEED):
    settings = _settings(kind)
    x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=settings["weight_dtype"], seed=seed)
    b = None if settings["bias_dtype"] is None else _bias_input(C, settings["bias_dtype"], key=key, seed=seed)
    return x, w, cu, dy, b, settings["policy"]


def _backward(x, w, cu, dy, b, policy, **kwargs):
    grads = pkg.cake_causal_conv1d_det_train_backward(
        x, w, cu, dy, bias=b, silu_grad_preactivation=policy, **kwargs
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _judge(S: int, C: int, docs, kind: str, *, weight_dtype=None, dweight_dtype=None, label: str = "", **kwargs):
    """``y`` and the pair route's gradients of one problem, judged over the draws."""

    settings = dict(_settings(kind))
    if weight_dtype is not None:
        settings["weight_dtype"] = weight_dtype

    def launch(_draw: int, seed: int):
        x, w, cu, dy = _inputs(S, C, docs, weight_dtype=settings["weight_dtype"], seed=seed)
        b = None if settings["bias_dtype"] is None else _bias_input(C, settings["bias_dtype"], seed=seed)
        y = pkg.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b)
        dw_out = None if dweight_dtype is None else torch.empty(C, WIDTH, dtype=dweight_dtype, device=_device())
        dx, dw, db = _backward(x, w, cu, dy, b, settings["policy"], dweight=dw_out, dweight_sum=PAIR, **kwargs)
        assert dx.dtype == BF16 and tuple(dx.shape) == tuple(x.shape)
        assert dw.dtype == (dweight_dtype or w.dtype) and tuple(dw.shape) == (C, WIDTH)
        outputs = {"y": y, "dx": dx, "dweight": dw}
        if db is not None:
            assert db.dtype == b.dtype and tuple(db.shape) == (C,)
            outputs["dbias"] = db
        return outputs, problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label=label or f"{kind} S={S} C={C} docs={docs} {kwargs}")


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_pair_routes_are_judged(case, kind) -> None:
    """``y``, ``dx``, ``dweight`` (and ``dbias``) of every pair route judged against the naive tier."""

    _require_pair(kind)
    _judge(*case, kind)


@pytest.mark.parametrize("weight_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("case", ROUTE_CASES[:3], ids=_case_id)
def test_pair_route_with_bf16_taps_and_bf16_dweight(case, weight_dtype) -> None:
    """The pair route reads the tap and gradient dtypes from ``flags`` like its twin: BF16 taps and a BF16
    ``dweight`` out judged against the naive tier."""

    _require_pair("bwd_fp32g_pair")
    _judge(*case, "bwd_fp32g_pair", weight_dtype=weight_dtype, dweight_dtype=BF16)


@pytest.mark.parametrize("case", PRECISION_CASES, ids=_case_id)
def test_pair_perf_size_precision(case) -> None:
    """The pair route at the perf size and on every narrow and multi-tile plan of its twin: judged against the naive
    tier."""

    _require_pair("bwd_fp32g_pair")
    _judge(*case, "bwd_fp32g_pair")
    torch.cuda.empty_cache()


@pytest.mark.parametrize("case", PRECISION_CASES[:2], ids=_case_id)
def test_pair_bias_perf_size_precision(case) -> None:
    _require_pair("bwd_bias_fp32g_pair")
    _judge(*case, "bwd_bias_fp32g_pair")
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bwd_grid_mult", (2, 3, 4))
@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_routes_at_bwd_grid_mult(kind, bwd_grid_mult) -> None:
    """Every opt-in ``bwd_grid_mult`` on the pair routes: judged against the naive tier, and three launches on a
    NaN-poisoned workspace of that multiple give bitwise equal gradients."""

    _require_pair(kind)
    for S, C, docs in ROUTE_CASES[:3]:
        _judge(S, C, docs, kind, bwd_grid_mult=bwd_grid_mult)
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=5)
        ws = pkg.cake_causal_conv1d_det_train_backward_workspace(S, C, _device(), bwd_grid_mult=bwd_grid_mult)
        runs = []
        for _ in range(3):
            ws.view(torch.float32).fill_(float("nan"))
            runs.append(_backward(x, w, cu, dy, b, policy, workspace=ws, dweight_sum=PAIR, bwd_grid_mult=bwd_grid_mult))
        for again in runs[1:]:
            for got, expected in zip(again, runs[0], strict=True):
                if expected is not None:
                    assert torch.equal(got, expected), (kind, S, C, bwd_grid_mult)


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_gradients_are_bitwise_reproducible(kind) -> None:
    """Three launches on a NaN-poisoned workspace and one on a fresh workspace give bitwise equal ``dx``,
    ``dweight`` and ``dbias``; ``dx`` is the twin route's bits."""

    _require_pair(kind)
    for S, C, docs in (*ROUTE_CASES, (65536, 576, "perf")):
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind)
        ws = pkg.cake_causal_conv1d_det_train_backward_workspace(S, C, _device())
        runs = []
        for _ in range(3):
            ws.view(torch.float32).fill_(float("nan"))
            runs.append(_backward(x, w, cu, dy, b, policy, workspace=ws, dweight_sum=PAIR))
        runs.append(_backward(x, w, cu, dy, b, policy, dweight_sum=PAIR))
        for again in runs[1:]:
            for got, expected, name in zip(again, runs[0], ("dx", "dweight", "dbias"), strict=True):
                if expected is not None:
                    assert torch.isfinite(got.float()).all(), (kind, S, C, name)
                    assert torch.equal(got, expected), (kind, S, C, name)
        twin_dx = _backward(x, w, cu, dy, b, policy)[0]
        assert torch.equal(runs[0][0], twin_dx), (kind, S, C, "dx is not the twin route's")


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_call_launches_the_pair_of_the_twin_program(kind, monkeypatch) -> None:
    """A pair call launches its route's program for ``C`` then that record's reduce program (the pair reduce); the
    program shares its twin route's selected program's launch plan (geometry, co-residency, grid and partial rows),
    and the default policy of the same call never launches a pair program."""

    _require_pair(kind)
    twin = PAIR_KINDS[kind]
    arch = _arch()
    pair_names = set(loader.route_modules(arch, kind))
    launched: list[tuple[str, dict, tuple | None]] = []

    def record(launches, values, *, device_index):
        for name, grid in launches:
            launched.append((name, dict(values), grid))

    monkeypatch.setattr(pkg, "run_modules", record)
    Cs = sorted({128, 256, 512, 640, 768, 1016, 1024, 1280, 1536, 2048, 2560})
    for C in Cs:
        S = 4097
        x, w, cu, dy, b, policy = _route_inputs(S, C, (1000, 0, 3, 3094), kind)
        launched.clear()
        _backward(x, w, cu, dy, b, policy, dweight_sum=PAIR)
        (main, values, grid), (reduce_name, _, _) = launched
        assert main == loader.selected_modules(arch, kind, C=C, S=S)[0] and main in pair_names
        pair_record = loader.MODULES[main]
        assert reduce_name == pair_record["reduce"] and loader.MODULES[reduce_name]["role"] == "reduce"
        launched.clear()
        _backward(x, w, cu, dy, b, policy)
        (twin_main, twin_values, twin_grid), _ = launched
        assert twin_main == loader.selected_modules(arch, twin, C=C, S=S)[0]
        assert twin_main not in pair_names
        twin_record = loader.MODULES[twin_main]
        for key in ("geometry", "ctas_per_sm"):
            assert pair_record["grid_rule"][key] == twin_record["grid_rule"][key], (C, main, twin_main, key)
        assert grid == twin_grid, (C, main, twin_main)
        assert tuple(values["partial"].shape) == tuple(twin_values["partial"].shape), (C, main)
        assert values["n_ch_tiles"] == twin_values["n_ch_tiles"] and values["ct_vec"] == twin_values["ct_vec"]


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_cuda_graph_replays_match_eager(kind) -> None:
    """One forward + pair backward graph per shape, replayed into NaN-filled outputs and workspace: every replay
    bitwise equals eager."""

    _require_pair(kind)
    device = _device()
    for S, C, docs in ROUTE_CASES:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=11)
        eager = (
            pkg.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b),
            *_backward(x, w, cu, dy, b, policy, dweight_sum=PAIR),
        )
        y, dx = torch.empty_like(x), torch.empty_like(x)
        dw = torch.empty(C, WIDTH, dtype=w.dtype, device=device)
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        ws = pkg.cake_causal_conv1d_det_train_backward_workspace(S, C, device)

        def step():
            pkg.cake_causal_conv1d_det_train_forward(x, w, cu, bias=b, out=y)
            pkg.cake_causal_conv1d_det_train_backward(
                x, w, cu, dy, bias=b, silu_grad_preactivation=policy, workspace=ws, dx=dx, dweight=dw, dbias=db,
                dweight_sum=PAIR,
            )

        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            step()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            step()
        torch.cuda.synchronize(device)
        outs = (y, dx, dw, db)
        for replay in range(GRAPH_REPLAYS):
            for t in outs:
                if t is not None:
                    t.fill_(float("nan"))
            ws.view(torch.float32).fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            for got, expected, name in zip(outs, eager, ("y", "dx", "dweight", "dbias"), strict=True):
                if got is not None:
                    assert torch.equal(got, expected), (kind, S, C, name, replay)
        del graph


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_autograd_end_to_end(kind) -> None:
    """``cake_causal_conv1d_det(..., dweight_sum="fp32_pair")`` returns the forward's bits and the pair backward's
    gradients, and is judged against the naive tier."""

    _require_pair(kind)
    S, C, docs = 1000, 512, "perf"
    x0, w0, cu, dy, b0, policy = _route_inputs(S, C, docs, kind, key=3)
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    b = None if b0 is None else b0.detach().clone().requires_grad_(True)
    y = pkg.cake_causal_conv1d_det(x, w, cu, b, silu_grad_preactivation=policy, dweight_sum=PAIR)
    assert torch.equal(y.detach(), pkg.cake_causal_conv1d_det_train_forward(x0, w0, cu, bias=b0))
    y.backward(dy)
    dx_ref, dw_ref, db_ref = _backward(x0, w0, cu, dy, b0, policy, dweight_sum=PAIR)
    assert torch.equal(x.grad, dx_ref) and torch.equal(w.grad, dw_ref)
    if b is not None:
        assert torch.equal(b.grad, db_ref)
    settings = _settings(kind)

    def launch(_draw: int, seed: int):
        xd, wd, cud, dyd = _inputs(S, C, docs, weight_dtype=settings["weight_dtype"], seed=seed)
        bd = None if settings["bias_dtype"] is None else _bias_input(C, settings["bias_dtype"], seed=seed)
        xg = xd.clone().requires_grad_(True)
        wg = wd.clone().requires_grad_(True)
        bg = None if bd is None else bd.clone().requires_grad_(True)
        yg = pkg.cake_causal_conv1d_det(xg, wg, cud, bg, silu_grad_preactivation=policy, dweight_sum=PAIR)
        yg.backward(dyd)
        out = {"y": yg.detach(), "dx": xg.grad, "dweight": wg.grad}
        if bg is not None:
            out["dbias"] = bg.grad
        return out, problem(xd, wd, cud, bd, dyd)

    assert_judged(launch, seed=SEED, label=f"autograd {kind}")


def test_pair_refusals_before_any_launch(monkeypatch) -> None:
    """``"fp32_pair"`` without the FP32 SiLU' policy and an unknown ``dweight_sum`` raise ``ValueError`` (the
    backward and the autograd entry, before any launch); the default policy and ``"fp32"`` never select a pair
    route."""

    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    b = _bias_input(C, F32)
    launches = []
    monkeypatch.setattr(pkg, "run_modules", lambda *a, **k: launches.append(a))
    monkeypatch.setattr(pkg, "run_module", lambda *a, **k: launches.append(a))
    for bias in (None, b):
        with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
            pkg.cake_causal_conv1d_det_train_backward(x, w, cu, dy, bias=bias, dweight_sum=PAIR)
        with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
            pkg.cake_causal_conv1d_det(x, w, cu, bias, dweight_sum=PAIR)
        for bad, error in (("fp64", ValueError), ("pair", ValueError), ("", ValueError), ("FP32_PAIR", ValueError),
                           (None, TypeError), (1, TypeError), (True, TypeError)):
            with pytest.raises(error, match="dweight_sum must be"):
                pkg.cake_causal_conv1d_det_train_backward(
                    x, w, cu, dy, bias=bias, silu_grad_preactivation="fp32", dweight_sum=bad
                )
            with pytest.raises(error, match="dweight_sum must be"):
                pkg.cake_causal_conv1d_det(x, w, cu, bias, silu_grad_preactivation="fp32", dweight_sum=bad)
    assert launches == []
    assert pkg.DWEIGHT_SUMS == ("fp32", "fp32_pair")
    for bias in (False, True):
        for fp32_grad in (False, True):
            for bf16 in (False, True):
                route = loader.backward_route(bias=bias, fp32_grad=fp32_grad, bf16_taps_or_dweight=bf16)
                assert route not in PAIR_KINDS
    assert loader.backward_route(bias=False, fp32_grad=True, bf16_taps_or_dweight=False, pair=True) == "bwd_fp32g_pair"
    assert loader.backward_route(bias=True, fp32_grad=True, bf16_taps_or_dweight=True, pair=True) == "bwd_bias_fp32g_pair"


def test_pair_routes_ship_on_every_architecture() -> None:
    """Both pair routes are in the registry of every exported architecture, with their libraries in ``build.json``
    (the suite runs on one architecture; the others' libraries are listed too)."""

    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        for kind in PAIR_KINDS:
            names = loader.route_modules(arch, kind)
            assert names, (arch, kind)
            for name in names:
                assert loader.library_name(name, arch) in libraries, (arch, kind, name)
                reduce_name = loader.MODULES[name]["reduce"]
                assert loader.library_name(reduce_name, arch) in libraries, (arch, kind, reduce_name)
