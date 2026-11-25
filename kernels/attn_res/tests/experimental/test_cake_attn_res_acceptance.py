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

The acceptance criteria of issue #262 for both modules (``det`` and ``nondet``), through the autograd entry, with
``H = 7168`` and ``eps = 1e-5`` unless stated:

* accuracy at ``n`` = 2, 5 and 9 (32 draws each): against an FP64 evaluation of the issue's math on the same BF16 /
  FP32 inputs, ``out`` and every ``dv_j`` meet the naive BF16 tier (``tests/cake_naive.py``), and ``dw`` is no worse
  than an FP32 sum: the issue's own FP32 evaluation (``issue_backward`` in FP32 on the CPU), in the L2 and the max
  norm, median and largest over the draws (``Fp32SumJudge``).  The draws: values of different magnitudes (the running
  sum largest), rows where one value takes almost all the softmax weight, and all-zero rows in a closed block state.
  The relative L2 errors of the kernel and of the issue's FP32 evaluation (its yardstick: ``out`` 1.5e-3 to 1.6e-3,
  ``dv`` 1.7e-3, ``dw`` 3.4e-6 to 4.5e-6 at ``T = 2048``) are printed and recorded beside the verdicts;
* every ``n`` from 1 to 9 (``T = 1025``) and the token counts 1, 4097, 16380 and 32768 (``n`` = 2 and 9, and
  ``n = 5`` at 16380): every output meets the tier and ``dw`` is no worse than an FP32 sum (from 4097 tokens the
  issue's FP32 evaluation runs on the device);
* ``T = 0`` launches no Cake program: the forward runs nothing, the backward one ``torch.zeros`` fill of ``dw``;
* ``eps`` is a run-time scalar: at 1e-3 and 1e-6, on all-zero rows (``r = 1 / sqrt(eps)`` there), the autograd entry
  and the explicit forward and backward meet the tier against the oracle at that ``eps`` (``p``, ``r`` and ``d`` too),
  ``dw`` is no worse than the FP32 sum at that ``eps``, and no new library loads;
* token counts and ``n`` that change from call to call load no new library, and steady-state calls never synchronize
  the host (``torch.cuda.set_sync_debug_mode("error")``); a forward and a backward captured in one CUDA graph replay
  the eager bits;
* ``torch.utils.checkpoint(..., use_reentrant=False)`` gives plain autograd's output and value gradients bit for bit
  (``dw`` too in ``det``) and meets the tier;
* the workspace and the retained state are reported separately (``T = 16380``, ``n`` = 2, 5 and 9);
* ``det`` repeats every output bit for bit run to run (``T = 16380``, ``n = 9``).
"""

from __future__ import annotations

import functools
import importlib
import math
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_attn_res_naive import (
    EPS,
    HIDDEN,
    Fp32SumJudge,
    issue_fp32,
    make_inputs,
    naive,
    oracle,
    rel_l2,
    yardstick_rows,
)
from tests.cake_naive import judge_case

MODES = ("det", "nondet")
DATA = ("scaled", "peaked", "zero_rows")
# The token count of each accuracy draw: the issue's yardstick count for the scaled values, fewer for the others.
ACCURACY_T = {"scaled": 2048, "peaked": 512, "zero_rows": 512}
# From this token count up the issue's FP32 evaluation (the yardstick of dw) runs on the device, not on the CPU.
DEVICE_YARDSTICK_T = 4097


def _api(mode: str):
    return importlib.import_module(f"cake_pplx.cake_attn_res_{mode}_train")


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake AttnRes kernels ship for SM100 / SM103 / SM107")
    for mode in MODES:
        jit = importlib.import_module(f"cake_pplx._backends.cake_attn_res_{mode}_train.cake_jit")
        if not (Path(jit.__file__).parent / "build.json").is_file():
            pytest.skip("the package has no build.json yet (not rendered)")
    _build_libraries()
    return torch.device("cuda:0")


@functools.cache
def _build_libraries() -> None:
    for mode in MODES:
        backend = importlib.import_module(f"cake_pplx._backends.cake_attn_res_{mode}_train.cake_backend")
        for spec in backend.specs_for_arch(backend.device_arch(0)):
            spec.build_and_load()


def _autograd(mode: str, ns, *, checkpoint: bool = False, eps: float = EPS) -> dict[str, torch.Tensor]:
    """Output and gradients of the module's autograd entry, keyed as the judge's outputs."""
    entry = getattr(_api(mode), f"cake_attn_res_{mode}")
    values = [v.detach().clone().requires_grad_(True) for v in ns.values]
    w = ns.w.detach().clone().requires_grad_(True)
    fn = functools.partial(entry, values, w, eps=eps)
    out = torch.utils.checkpoint.checkpoint(fn, use_reentrant=False) if checkpoint else fn()
    out.backward(ns.dout)
    return {"out": out.detach(), **{f"dv{j}": v.grad for j, v in enumerate(values)}, "dw": w.grad}


def _explicit(mode: str, ns, *, eps: float = EPS) -> dict[str, torch.Tensor]:
    """Output, state and gradients of the module's explicit forward and backward, keyed as the judge's outputs."""
    api = _api(mode)
    out, p, r, d = getattr(api, f"cake_attn_res_{mode}_train_forward")(ns.values, ns.w, eps=eps)
    dv, dw = getattr(api, f"cake_attn_res_{mode}_train_backward")(ns.dout, ns.values, ns.w, p, r, d)
    return {"out": out, **{f"dv{j}": g for j, g in enumerate(dv)}, "dw": dw, "p": p, "r": r, "d": d}


def _no_launch(name: str, arch: str):
    raise AssertionError(f"Cake program {name} ({arch}) looked up")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _judge_modes(
    make, *, seed: int, checkpoint: bool = False, fp32_dw: bool = False, eps: float = EPS, explicit: bool = False
):
    """Judge both modes on the same draws (one oracle, one tier and, with ``fp32_dw``, one yardstick per draw, all at
    ``eps``): each mode's autograd entry, outputs keyed ``<mode>.<output>``, and with ``explicit`` its explicit forward
    and backward too, keyed ``<mode>.explicit.<output>`` (``p``, ``r`` and ``d`` included).  The yardstick runs on the
    CPU, or on the device from :data:`DEVICE_YARDSTICK_T` tokens.  Returns the tier judge and its failures, the
    ``Fp32SumJudge`` of each arm, and per arm the relative L2 errors of the kernel and of the yardstick."""
    arms = {mode: functools.partial(_autograd, mode, checkpoint=checkpoint, eps=eps) for mode in MODES}
    if explicit:
        arms.update({f"{mode}.explicit": functools.partial(_explicit, mode, eps=eps) for mode in MODES})
    cache = {}
    sums = {arm: Fp32SumJudge() for arm in arms}
    ours = {arm: {} for arm in arms}
    theirs = {arm: {} for arm in arms}

    def run(j, s):
        ns = make(s)
        want = oracle(ns, eps=eps)
        got = {arm: fn(ns) for arm, fn in arms.items()}
        torch.cuda.synchronize()
        keys = {f"{arm}.{name}": (arm, name) for arm in arms for name in got[arm]}
        cache[j] = (ns, keys)
        outputs = {key: got[arm][name] for key, (arm, name) in keys.items()}
        expected = {key: want[name] for key, (_, name) in keys.items()}
        if fp32_dw:
            device = ns.values[0].device if ns.T >= DEVICE_YARDSTICK_T else None
            evaluation = issue_fp32(ns, eps=eps, device=device)
            errors = {name: rel_l2(t, want[name]) for name, t in evaluation.items()}
            for arm in arms:
                sums[arm].add(got[arm]["dw"], evaluation["dw"], want["dw"])
                for name, error in errors.items():
                    ours[arm].setdefault(name, []).append(rel_l2(got[arm][name], want[name]))
                    theirs[arm].setdefault(name, []).append(error)
        return outputs, expected

    def tier(j, _tier):
        ns, keys = cache.pop(j)
        found = naive(ns, eps=eps)
        return {key: found[name] for key, (_, name) in keys.items()}

    judge, failures = judge_case(run, tier, seed=seed)
    return judge, failures, sums, ours, theirs


def _fp32_sum_failures(label: str, sums, ours, theirs, record_property) -> list[str]:
    """Print and record each arm's ``dw`` rows (``Fp32SumJudge``) and yardstick rows; returns the ``dw`` failures."""
    failures = []
    for arm, judge in sums.items():
        for row in judge.rows():
            line = (
                f"{arm} {label} dw {row['norm']}: ours med/max={row['ours'][0]:.4e}/{row['ours'][1]:.4e} "
                f"fp32 sum med/max={row['fp32'][0]:.4e}/{row['fp32'][1]:.4e} {'ok' if row['ok'] else 'FAIL'}"
            )
            print(line)
            record_property(f"{arm}-dw-{row['norm']}", line)
        for row in yardstick_rows(ours[arm], theirs[arm]):
            line = (
                f"{arm} {label} {row['output']} rel-L2: ours med/max={row['ours'][0]:.3e}/"
                f"{row['ours'][1]:.3e} issue-fp32 med/max={row['fp32'][0]:.3e}/{row['fp32'][1]:.3e}"
            )
            print(line)
            record_property(f"{arm}-yardstick-{row['output']}", line)
        failures += [f"{arm}: {message}" for message in judge.failures()]
    return failures


@pytest.mark.parametrize("data", DATA)
@pytest.mark.parametrize("n", (2, 5, 9))
def test_accuracy_meets_the_tier_and_dw_no_worse_than_an_fp32_sum(n, data, record_property) -> None:
    device = _device_or_skip()
    T = ACCURACY_T[data]

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data=data, weights="small")

    judge, failures, *sums = _judge_modes(make, seed=26200 + 10 * n, fp32_dw=True)
    judge.record(record_property)
    failures += _fp32_sum_failures(f"n={n} {data} T={T}", *sums, record_property)
    assert not failures, failures


@pytest.mark.parametrize("n", range(1, 10))
def test_every_value_count(n, record_property) -> None:
    """Every ``n`` from 1 to 9 (every value-bound program of both modes) meets the tier through the autograd entry,
    and ``dw`` is no worse than an FP32 sum."""
    device = _device_or_skip()
    T = 1025

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data="scaled", weights="normproj")

    judge, failures, *sums = _judge_modes(make, seed=26400 + n, fp32_dw=True)
    judge.record(record_property)
    failures += _fp32_sum_failures(f"n={n} scaled T={T}", *sums, record_property)
    assert not failures, failures


# (n, T): the issue's token counts at n = 2 and 9, and its production count at n = 5.
TOKEN_CASES = (*((n, T) for n in (2, 9) for T in (1, 4097, 16380, 32768)), (5, 16380))


@pytest.mark.parametrize(("n", "T"), TOKEN_CASES, ids=[f"n{n}-T{T}" for n, T in TOKEN_CASES])
def test_token_counts(n, T, record_property) -> None:
    """Every output meets the tier and ``dw`` is no worse than an FP32 sum (from :data:`DEVICE_YARDSTICK_T` tokens
    the yardstick runs on the device)."""
    device = _device_or_skip()

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data="scaled", weights="normproj")

    judge, failures, *sums = _judge_modes(make, seed=26300 + T + n, fp32_dw=True)
    judge.record(record_property)
    failures += _fp32_sum_failures(f"n={n} scaled T={T}", *sums, record_property)
    assert not failures, failures


@pytest.mark.parametrize("eps", (1e-3, 1e-6))
def test_eps_is_a_run_time_scalar(eps, record_property) -> None:
    """``eps`` other than the default, on all-zero rows of the first value (``r = 1 / sqrt(eps)`` there): the autograd
    entry and the explicit forward and backward of both modes meet the tier against the oracle at that ``eps``
    (``p``, ``r`` and ``d`` too), and ``dw`` is no worse than the FP32 sum at that ``eps``.  The same programs serve
    every ``eps``: after a warm-up at the default, no new library loads."""
    device = _device_or_skip()
    n, T = 5, ACCURACY_T["zero_rows"]
    backends = [importlib.import_module(f"cake_pplx._backends.cake_attn_res_{m}_train.cake_backend") for m in MODES]
    warm = make_inputs(T, n, seed=4, device=device, data="zero_rows")
    for mode in MODES:
        _autograd(mode, warm)
        _explicit(mode, warm)
    loads = [b.load.cache_info().misses for b in backends]

    def make(seed):
        return make_inputs(T, n, seed=seed, device=device, data="zero_rows", weights="small")

    seed = 26500 + round(-math.log10(eps))
    judge, failures, *sums = _judge_modes(make, seed=seed, fp32_dw=True, eps=eps, explicit=True)
    judge.record(record_property)
    failures += _fp32_sum_failures(f"n={n} zero_rows T={T} eps={eps:g}", *sums, record_property)
    assert [b.load.cache_info().misses for b in backends] == loads
    assert not failures, failures


@pytest.mark.parametrize("n", (2, 5, 9))
def test_no_token(n, monkeypatch) -> None:
    """``T = 0`` launches no Cake program (each backend's FFI entry is replaced by one that fails): an empty output,
    empty value gradients and a zero ``dw``, the backward's ``torch.zeros`` fill."""
    device = _device_or_skip()
    ns = make_inputs(0, n, seed=1, device=device)
    for mode in MODES:
        backend = importlib.import_module(f"cake_pplx._backends.cake_attn_res_{mode}_train.cake_backend")
        monkeypatch.setattr(backend, "_entry", _no_launch)
    for mode in MODES:
        for got in (_autograd(mode, ns), _explicit(mode, ns)):
            assert tuple(got["out"].shape) == (0, HIDDEN)
            assert all(tuple(got[f"dv{j}"].shape) == (0, HIDDEN) for j in range(n))
            assert got["dw"].dtype == torch.float32 and tuple(got["dw"].shape) == (HIDDEN,) and not got["dw"].any()


def test_dynamic_shapes_load_no_new_library_and_never_synchronize() -> None:
    """After one warm-up call per mode and value count (each program loaded once: the libraries are a fixed set, no
    shape compiles a new one), calls with changing ``n`` and token counts load no new library, and run under
    ``torch.cuda.set_sync_debug_mode("error")`` (any synchronizing CUDA call raises)."""
    device = _device_or_skip()
    backends = [importlib.import_module(f"cake_pplx._backends.cake_attn_res_{m}_train.cake_backend") for m in MODES]
    for n in range(1, 10):
        warm = make_inputs(5, n, seed=2 + n, device=device)
        for mode in MODES:
            _autograd(mode, warm)
    loads = [b.load.cache_info().misses for b in backends]
    shapes = ((1, 2), (4097, 9), (129, 5), (16380, 2), (7, 1), (2049, 8), (33, 4))
    inputs = [make_inputs(T, n, seed=T + n, device=device) for T, n in shapes]
    torch.cuda.synchronize(device)
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        results = [{mode: _autograd(mode, ns) for mode in MODES} for ns in inputs]
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize(device)
    assert [b.load.cache_info().misses for b in backends] == loads
    for (T, _), got in zip(shapes, results, strict=True):
        for mode in MODES:
            assert tuple(got[mode]["out"].shape) == (T, HIDDEN)
            assert all(torch.isfinite(t.float()).all() for t in got[mode].values()), mode


@pytest.mark.parametrize("mode", MODES)
def test_forward_and_backward_capture_in_one_cuda_graph(mode) -> None:
    device = _device_or_skip()
    api = _api(mode)
    fwd = getattr(api, f"cake_attn_res_{mode}_train_forward")
    bwd = getattr(api, f"cake_attn_res_{mode}_train_backward")
    ns = make_inputs(4097, 9, seed=5, device=device, data="peaked")
    eager = fwd(ns.values, ns.w)
    eager_dv, eager_dw = bwd(ns.dout, ns.values, ns.w, *eager[1:])
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        state = fwd(ns.values, ns.w)
        bwd(ns.dout, ns.values, ns.w, *state[1:])
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state = fwd(ns.values, ns.w)
        dv, dw = bwd(ns.dout, ns.values, ns.w, *state[1:])
    for t in (*state, *dv, dw):
        t.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize(device)
    for a, b in zip((*state, *dv), (*eager, *eager_dv), strict=True):
        assert _same_bits(a, b)
    if mode == "det":
        assert _same_bits(dw, eager_dw)
    else:
        assert torch.isfinite(dw).all()


@pytest.mark.parametrize("n", (2, 9))
def test_checkpoint(n, record_property) -> None:
    """Under ``torch.utils.checkpoint(use_reentrant=False)``: the output and the value gradients are plain autograd's
    bits (``dw`` too in ``det``), and the checkpointed calls meet the tier."""
    device = _device_or_skip()
    ns = make_inputs(1025, n, seed=7, device=device, data="zero_rows", weights="normproj")
    for mode in MODES:
        plain = _autograd(mode, ns)
        ckpt = _autograd(mode, ns, checkpoint=True)
        names = ["out", *(f"dv{j}" for j in range(n))] + (["dw"] if mode == "det" else [])
        for name in names:
            assert _same_bits(plain[name], ckpt[name]), (mode, name)

    def make(seed):
        return make_inputs(129, n, seed=seed, device=device, data="scaled")

    judge, failures, *_ = _judge_modes(make, seed=26400 + n, checkpoint=True)
    judge.record(record_property)
    assert not failures, failures


@pytest.mark.parametrize("n", (2, 5, 9))
def test_workspace_and_retained_state_are_reported(n, record_property) -> None:
    """``T = 16380``: the retained state (saved beside references to the inputs) and the backward workspace of each
    mode, from the API, checked against what the autograd node saves."""
    device = _device_or_skip()
    T = 16380
    ns = make_inputs(T, n, seed=9, device=device)
    for mode in MODES:
        api = _api(mode)
        retained = getattr(api, f"cake_attn_res_{mode}_train_retained_bytes")(T, n)
        workspace = getattr(api, f"cake_attn_res_{mode}_train_workspace_bytes")(T, n, device)
        values = [v.detach().clone().requires_grad_(True) for v in ns.values]
        w = ns.w.detach().clone().requires_grad_(True)
        out = getattr(api, f"cake_attn_res_{mode}")(values, w)
        inputs = {t.untyped_storage().data_ptr() for t in (*values, w)}
        new = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in out.grad_fn.saved_tensors}
        saved = sum(size for ptr, size in new.items() if ptr not in inputs)
        line = (
            f"{mode} T={T} n={n}: retained state {retained} bytes (p, r, d FP32 [T, n]; saved {saved}), "
            f"backward workspace {workspace} bytes, value bytes {n * T * HIDDEN * 2}"
        )
        print(line)
        record_property(f"{mode}-memory", line)
        assert saved == retained == 12 * T * n
        assert workspace == 0 if mode == "nondet" else workspace > 0


def test_det_repeats_bit_for_bit() -> None:
    device = _device_or_skip()
    ns = make_inputs(16380, 9, seed=11, device=device, data="peaked", weights="normproj")
    first = _autograd("det", ns)
    for _ in range(2):
        again = _autograd("det", ns)
        for name, t in first.items():
            assert _same_bits(t, again[name]), name
