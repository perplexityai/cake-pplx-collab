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

Tests of the opt-in FP32 (hi, lo) pair ``dweight`` / ``dbias`` sums of the non-deterministic package
(``dweight_sum="fp32_pair"``, the pair routes ``bwd_fp32g_pair`` / ``bwd_bias_fp32g_pair``).

Every floating output is judged as in ``test_cake_causal_conv1d_nondet_train.py``: against the naive BF16 tier over
32 independent random inputs, by its error against the float64 oracle in the L2 and the max norm (median and largest
error at most 1.005 times the tier's); repeated launches on a draw's inputs are all judged.  No bit rule holds the
package, and the suite compares no bits between launches or packages.  The suite covers the program a call selects
(the pair of the program the twin route selects, on the twin's launch plan) and its accumulators (the rendered rule's
FP32 (hi, lo) words inside the public workspace, zero after every launch), the opt-in ``bwd_grid_mult``, CUDA Graph
capture / replay with and without a caller-owned workspace, the autograd entry with checkpointing, and the refusals:
``"fp32_pair"`` without ``silu_grad_preactivation="fp32"`` and an unknown ``dweight_sum`` raise ``ValueError`` before
any launch.  The default (``dweight_sum="fp32"``) never selects a pair route.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_causal_conv1d_oracle import assert_judged, problem
from tests.cake.test_helpers.cake_naive import DRAWS
from tests.experimental.test_cake_causal_conv1d_nondet_train import (
    BF16,
    F32,
    GRID_MULT_CASES,
    PRECISION_CASES,
    PRECISION_REPEATS,
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
    _grads,
    _inputs,
    _require_route,
    _stream_workspace,
    _supported,
    cake_build,
    loader,
    pkg,
)

# pair route -> the route it pairs (its twin)
PAIR_KINDS = {"bwd_fp32g_pair": "bwd_fp32g", "bwd_bias_fp32g_pair": "bwd_bias_fp32g"}
PAIR = "fp32_pair"
# Launches per draw of the repeated rows (every launch judged).
REPEATS = 5

pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_causal_conv1d_nondet_train needs an SM100 / SM103 / SM107 device and its rendered package",
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
    grads = pkg.cake_causal_conv1d_nondet_train_backward(
        x, w, cu, dy, bias=b, silu_grad_preactivation=policy, **kwargs
    )
    return grads[0], grads[1], (grads[2] if b is not None else None)


def _judge(S: int, C: int, docs, kind: str, call, *, weight_dtype=None, key: int = 0, label: str = "") -> None:
    """``call(x, w, cu, dy, b, policy)`` (the outputs by name: a value or a list of launches' values) on each draw's
    inputs of the pair route ``kind``, judged against the naive tier."""

    settings = dict(_settings(kind))
    if weight_dtype is not None:
        settings["weight_dtype"] = weight_dtype

    def launch(_draw: int, seed: int):
        x, w, cu, dy = _inputs(S, C, docs, key=key, weight_dtype=settings["weight_dtype"], seed=seed)
        b = None if settings["bias_dtype"] is None else _bias_input(C, settings["bias_dtype"], key=key, seed=seed)
        return call(x, w, cu, dy, b, settings["policy"]), problem(x, w, cu, b, dy)

    assert_judged(launch, seed=SEED, label=label or f"{kind} S={S} C={C} docs={docs}")


def _repeated(repeats: int, *, dweight_dtype=None, **kwargs):
    """A ``call`` of :func:`_judge`: ``y`` once and ``repeats`` pair backward launches, every launch's outputs
    returned (and the package's stream workspace read back zero after each)."""

    def call(x, w, cu, dy, b, policy):
        found: dict[str, list] = {"y": [pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, bias=b)]}
        for _ in range(repeats):
            dw_out = None
            if dweight_dtype is not None:
                dw_out = torch.empty(x.shape[-1], WIDTH, dtype=dweight_dtype, device=x.device)
            grads = _grads(*_backward(x, w, cu, dy, b, policy, dweight=dw_out, dweight_sum=PAIR, **kwargs))
            torch.cuda.synchronize()
            if "workspace" not in kwargs:
                held = _stream_workspace(x.device)
                assert held is not None and int(torch.count_nonzero(held)) == 0
            for name, value in grads.items():
                found.setdefault(name, []).append(value)
        return found

    return call


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
@pytest.mark.parametrize("case", ROUTE_CASES, ids=_case_id)
def test_pair_routes_are_judged(case, kind) -> None:
    """``y`` and ``REPEATS`` pair backward launches per draw, every launch judged against the naive tier; the
    accumulators are zero after every launch."""

    _require_pair(kind)
    _judge(*case, kind, _repeated(REPEATS))


@pytest.mark.parametrize("weight_dtype", (F32, BF16), ids=_dtype_id)
@pytest.mark.parametrize("case", ROUTE_CASES[:3], ids=_case_id)
def test_pair_route_with_bf16_taps_and_bf16_dweight(case, weight_dtype) -> None:
    _require_pair("bwd_fp32g_pair")
    _judge(*case, "bwd_fp32g_pair", _repeated(2, dweight_dtype=BF16), weight_dtype=weight_dtype)


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
@pytest.mark.parametrize("case", PRECISION_CASES, ids=_case_id)
def test_pair_perf_size_precision(case, kind) -> None:
    """The pair routes at the perf size: ``PRECISION_REPEATS`` launches per draw, every launch judged."""

    _require_pair(kind)
    _judge(*case, kind, _repeated(PRECISION_REPEATS))
    torch.cuda.empty_cache()


@pytest.mark.parametrize("bwd_grid_mult", (2, 3, 4))
@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_routes_at_bwd_grid_mult(kind, bwd_grid_mult) -> None:
    """Every opt-in ``bwd_grid_mult`` on the pair routes, on a caller-owned workspace of the public size: judged
    against the naive tier, the workspace zero after every launch."""

    _require_pair(kind)
    for S, C, docs in GRID_MULT_CASES:
        ws = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, _device(), bwd_grid_mult=bwd_grid_mult)

        def call(x, w, cu, dy, b, policy, ws=ws):
            found = _repeated(2, workspace=ws, bwd_grid_mult=bwd_grid_mult)(x, w, cu, dy, b, policy)
            assert int(torch.count_nonzero(ws)) == 0
            return found

        _judge(S, C, docs, kind, call, key=7, label=f"{kind} S={S} C={C} bwd_grid_mult={bwd_grid_mult}")
        torch.cuda.empty_cache()


@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
@pytest.mark.parametrize("C", (512, 1280, 8192))
def test_pair_call_launches_the_pair_of_the_twin_program(kind, C, monkeypatch) -> None:
    """A pair call is ONE launch of its route's program for ``C``: its launch plan (geometry, co-residency, grid,
    channel tiles) is the twin route's selected program's, and its ``acc`` is the rendered rule's FP32 (hi, lo) words
    (``slots x taps x C``, ``slots`` even) inside the public workspace, zero at launch; the default policy of the same
    call never launches a pair program.  The outputs are judged against the naive tier."""

    _require_pair(kind)
    twin = PAIR_KINDS[kind]
    arch = _arch()
    S, docs = 4097, (1000, 0, 3, 3094)
    ws = pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, _device())
    pair_names = set(loader.route_modules(arch, kind))
    seen: list[tuple] = []
    run = pkg.run_module

    def record(name, values, *, device_index, grid=None):
        seen.append((name, grid, values["n_ch_tiles"], values.get("ct_vec"), values["acc"].dtype, values["acc"].numel(),
                     int(torch.count_nonzero(values["acc"]))))
        return run(name, values, device_index=device_index, grid=grid)

    def call(x, w, cu, dy, b, policy):
        with monkeypatch.context() as m:
            m.setattr(pkg, "run_module", record)
            seen.clear()
            dx, dw, db = _backward(x, w, cu, dy, b, policy, workspace=ws, dweight_sum=PAIR)
            ((name, grid, tiles, ct_vec, acc_dtype, acc_numel, acc_nonzero),) = seen
            seen.clear()
            _backward(x, w, cu, dy, b, policy, workspace=ws)
            ((twin_name, twin_grid, twin_tiles, twin_ct_vec, _, _, _),) = seen
        assert name == loader.selected_modules(arch, kind, C=C, S=S)[0] and name in pair_names
        assert twin_name == loader.selected_modules(arch, twin, C=C, S=S)[0] and twin_name not in pair_names
        record_, twin_record = loader.MODULES[name], loader.MODULES[twin_name]
        for key in ("geometry", "ctas_per_sm"):
            assert record_["grid_rule"][key] == twin_record["grid_rule"][key], (C, name, twin_name, key)
        assert (grid, tiles, ct_vec) == (twin_grid, twin_tiles, twin_ct_vec), (C, name, twin_name)
        taps = 5 if "bias" in kind else 4
        slots = record_["workspace_rule"]["slots"]
        assert slots % 2 == 0 and loader.accumulator_words_per_channel(record_) == slots * taps
        assert acc_dtype == torch.float32 and acc_numel == slots * taps * C and acc_nonzero == 0
        assert ws.numel() >= loader.accumulator_layout(record_, C, tiles)["total_bytes"]
        assert int(torch.count_nonzero(ws)) == 0
        return _grads(dx, dw, db)

    _judge(S, C, docs, kind, call, key=11, label=f"{kind} C={C} launch")


def _capture_and_replay(kind: str, *, own_workspace: bool, bwd_grid_mult: int = 1) -> int:
    device = _device()
    replays = 0
    for S, C, docs in ROUTE_CASES[:3]:
        x, w, cu, dy, b, policy = _route_inputs(S, C, docs, kind, key=13)
        y, dx = torch.empty_like(x), torch.empty_like(x)
        dw = torch.empty(C, WIDTH, dtype=w.dtype, device=device)
        db = None if b is None else torch.empty(C, dtype=b.dtype, device=device)
        ws = (
            pkg.cake_causal_conv1d_nondet_train_backward_workspace(S, C, device, bwd_grid_mult=bwd_grid_mult)
            if own_workspace
            else None
        )

        def step(x=x, w=w, cu=cu, dy=dy, b=b, policy=policy, y=y, dx=dx, dw=dw, db=db, ws=ws):
            pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, bias=b, out=y)
            pkg.cake_causal_conv1d_nondet_train_backward(
                x, w, cu, dy, bias=b, silu_grad_preactivation=policy, workspace=ws, dx=dx, dweight=dw, dbias=db,
                bwd_grid_mult=bwd_grid_mult, dweight_sum=PAIR,
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

        def replay(xn, wn, cun, dyn, bn, policy, graph=graph, held=(x, w, cu, dy, b), outs=(y, dx, dw, db), ws=ws):
            nonlocal replays
            for t, value in zip(held, (xn, wn, cun, dyn, bn), strict=True):
                if t is not None:
                    t.copy_(value)
            for t in outs:
                if t is not None:
                    t.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            replays += 1
            if ws is not None:
                assert int(torch.count_nonzero(ws)) == 0
            return {"y": outs[0], **_grads(*outs[1:])}

        _judge(S, C, docs, kind, replay, key=13,
               label=f"replay {kind} S={S} C={C} own={own_workspace} bwd_grid_mult={bwd_grid_mult}")
    return replays


@pytest.mark.parametrize("bwd_grid_mult", (1, 2))
@pytest.mark.parametrize("own_workspace", (True, False), ids=("own-ws", "pkg-ws"))
@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_cuda_graph_replays(kind, own_workspace, bwd_grid_mult) -> None:
    """CUDA Graph capture / replay of the pair backward, every replay on a draw's inputs judged against the naive
    tier, a caller-owned workspace zero after every replay."""

    _require_pair(kind)
    assert _capture_and_replay(kind, own_workspace=own_workspace, bwd_grid_mult=bwd_grid_mult) == DRAWS * 3


@pytest.mark.parametrize("checkpointed", (False, True), ids=("plain", "checkpoint"))
@pytest.mark.parametrize("kind", tuple(PAIR_KINDS))
def test_pair_autograd_end_to_end(kind, checkpointed) -> None:
    """``cake_causal_conv1d_nondet(..., dweight_sum="fp32_pair")``, plain and under non-reentrant activation
    checkpointing (2 x forward + backward): ``y`` and the gradients in the parameters' dtypes, judged against the
    naive tier."""

    from torch.utils.checkpoint import checkpoint

    _require_pair(kind)
    S, C, docs = 1000, 512, "perf"

    def call(x, w, cu, dy, b, policy):
        xg = x.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        bg = None if b is None else b.clone().requires_grad_(True)

        def fn(x_, w_, b_):
            return pkg.cake_causal_conv1d_nondet(x_, w_, cu, b_, silu_grad_preactivation=policy, dweight_sum=PAIR)

        y = checkpoint(fn, xg, wg, bg, use_reentrant=False) if checkpointed else fn(xg, wg, bg)
        y.backward(dy)
        assert xg.grad.dtype == BF16 and wg.grad.dtype == w.dtype
        out = {"y": y.detach(), "dx": xg.grad, "dweight": wg.grad}
        if bg is not None:
            assert bg.grad.dtype == b.dtype
            out["dbias"] = bg.grad
        return out

    _judge(S, C, docs, kind, call, key=3, label=f"autograd {kind} checkpoint={checkpointed}")


def test_pair_refusals_before_any_launch(monkeypatch) -> None:
    """``"fp32_pair"`` without the FP32 SiLU' policy and an unknown ``dweight_sum`` raise ``ValueError`` (the
    backward and the autograd entry at its forward call, before any launch); the default never selects a pair
    route."""

    S, C, docs = 1000, 512, "perf"
    x, w, cu, dy = _inputs(S, C, docs)
    b = _bias_input(C, F32)
    launches = []
    monkeypatch.setattr(pkg, "run_module", lambda *a, **k: launches.append(a))
    for bias in (None, b):
        with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, dy, bias=bias, dweight_sum=PAIR)
        with pytest.raises(ValueError, match='needs silu_grad_preactivation="fp32"'):
            pkg.cake_causal_conv1d_nondet(x, w, cu, bias, dweight_sum=PAIR)
        for bad, error in (("fp64", ValueError), ("pair", ValueError), ("", ValueError), ("FP32_PAIR", ValueError),
                           (None, TypeError), (1, TypeError), (True, TypeError)):
            with pytest.raises(error, match="dweight_sum must be"):
                pkg.cake_causal_conv1d_nondet_train_backward(
                    x, w, cu, dy, bias=bias, silu_grad_preactivation="fp32", dweight_sum=bad
                )
            with pytest.raises(error, match="dweight_sum must be"):
                pkg.cake_causal_conv1d_nondet(x, w, cu, bias, silu_grad_preactivation="fp32", dweight_sum=bad)
    assert launches == []
    assert pkg.DWEIGHT_SUMS == ("fp32", "fp32_pair")
    for bias in (False, True):
        for fp32_grad in (False, True):
            for bf16 in (False, True):
                assert loader.backward_route(bias=bias, fp32_grad=fp32_grad, bf16_taps_or_dweight=bf16) not in PAIR_KINDS


def test_pair_routes_ship_on_every_architecture() -> None:
    libraries = cake_build.libraries_by_name()
    for arch in loader.ARCHES:
        for kind in PAIR_KINDS:
            names = loader.route_modules(arch, kind)
            assert names, (arch, kind)
            for name in names:
                assert loader.library_name(name, arch) in libraries, (arch, kind, name)
