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

The acceptance criteria of the feature request (perplexityai/cake-pplx-collab#76) as explicit tests of
``cake_pplx.cake_gated_kv_compression_det_train``.  The bounds below are the contract owner's own criteria, stated in
the request against an FP64 evaluation of the same math on the same FP32 ``C``, ``Z`` and BF16 ``dout``; they sit
beside the repository's naive-tier judge (``tests/cake_naive.py``), which every other suite of the family applies.

* ``out``, in the request's words: "Each `out` element is within 1 BF16 ulp of the FP64 result rounded once to BF16,
  or its absolute error is at most 1 BF16 ulp of `max(|C[t0,d]|, |C[t1,d]|)`. The second case covers near-cancelling
  pairs, where the released FP32 math has the same error. Report the count of second-case elements and the worst ulp
  distance."  :func:`classify_out` applies it per element: the first case is a distance of at most :data:`OUT_ULPS`
  BF16 values between ``out`` and the FP64 result rounded once (round to nearest even) to BF16; the second, for a
  larger distance, is ``|out - y| <= ulp_bf16(max(|C[t0, d]|, |C[t1, d]|))`` with ``y`` the unrounded FP64 result
  (:func:`bf16_ulp`).  Every element of every seed meets one of the two; the test reports, per case, the elements, the
  second-case count, the count outside both and the worst distance.
* ``dC`` and ``dZ`` have a relative L2 error of at most 1e-6 (:data:`GRAD_REL_L2`), also when every pair has
  ``C[t1] = C[t0] (1 + 2^-12)`` and ``C[t1] = C[t0] (1 + 2^-16)``.
* Document lengths 0, 1, 2 and 3, in the pack ``(0, 1, 3, 2, 7, 4)``: ``out`` equals the per-document result.
  Changing the last token of an odd-length document changes no output, and its gradient rows are exactly zero.
* Packs with no complete pair (no documents, ``(0,)``, ``(1, 0, 1)``) return an empty ``out`` and all-zero ``dC``
  and ``dZ``.
* Column-half inputs and gradient outputs give the same results as contiguous ones.
* Gate gaps of 100 and more, and exact ties, give finite ``out``, ``dC`` and ``dZ``.
* Any ``T``, ``S`` and ``G`` without recompiling; works under ``torch.utils.checkpoint(..., use_reentrant=False)``;
  no host synchronization in steady state.
* Workspace and state retained for the backward are reported separately; the measured peak of a backward is the
  reported workspace (and, for the autograd backward of an expanded incoming gradient, its documented copy).

Every test runs on SM100 / SM103 / SM107 and skips elsewhere, except the check of :func:`classify_out` on hand-made
values, which needs no GPU.  Each measured figure is recorded with pytest's ``record_property`` (``acceptance``).
"""

from __future__ import annotations

import functools
import random
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

import cake_pplx.cake_gated_kv_compression_det_train as api
from tests.cake.test_helpers.cake_gated_kv_compression_naive import (
    PERF_PACKS,
    WIDTH,
    Pack,
    assert_bitwise,
    assert_zero_rows,
    backward_values,
    bf16_ulp_distance,
    bounds,
    forward_values,
    make_inputs,
    nan_halves,
    nan_like_rows,
    rel_l2,
    tail_rows,
)

#: The first case of the contract owner's ``out`` criterion: BF16 values between ``out`` and the FP64 result rounded
#: once to BF16.
OUT_ULPS = 1
#: The contract owner's bound on ``dC`` and ``dZ``: relative L2 error against FP64.
GRAD_REL_L2 = 1e-6
#: Independent seeds of each measured case.
SEEDS = tuple(range(8))
MIXED = Pack((127, 256, 1, 641))
MEASURED = (
    pytest.param(MIXED, {}, id="random"),
    pytest.param(MIXED, {"close": 12}, id="close-2^-12"),
    pytest.param(MIXED, {"close": 16}, id="close-2^-16"),
    pytest.param(MIXED, {"gates": True}, id="gate-extremes"),
    pytest.param(Pack((1, 3, 5, 7) * 75, "small-odd-S300"), {}, id="small-odd"),
    pytest.param(PERF_PACKS[0], {}, id=PERF_PACKS[0].id),
    pytest.param(PERF_PACKS[1], {}, id=PERF_PACKS[1].id),
    pytest.param(PERF_PACKS[0], {"close": 16}, id=f"{PERF_PACKS[0].id}-close-2^-16"),
)


def _jit():
    from cake_pplx._backends.cake_gated_kv_compression_det_train import cake_jit

    return cake_jit


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake gated KV compression ships for SM100 / SM103 / SM107")
    if not (Path(_jit().__file__).parent / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    _build_libraries(0)
    return torch.device("cuda:0")


@functools.cache
def _build_libraries(index: int) -> None:
    jit = _jit()
    for spec in jit.specs_for_arch(jit.device_arch(index)):
        spec.build_and_load()


def _autograd_cz(x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cz = x.cz.detach().clone().requires_grad_(True)
    out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
    out.backward(x.dout)
    return out.detach(), cz.grad[:, :WIDTH], cz.grad[:, WIDTH:]


def _explicit(x, layout: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = nan_like_rows(x.G, torch.bfloat16, x.c.device)
    api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, out=out)
    if layout == "halves":
        _, dc, dz = nan_halves(x.T, x.c.device)
    else:
        dc, dz = nan_like_rows(x.T, torch.float32, x.c.device), nan_like_rows(x.T, torch.float32, x.c.device)
    api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    return out, dc, dz


def bf16_rounded_once(y: torch.Tensor) -> torch.Tensor:
    """FP64 ``y`` rounded once (to nearest, ties to even) to BF16: the quantum of ``y``'s binade at BF16's 8
    significant bits (the subnormal quantum ``2^-133`` below ``2^-126``), then an exact cast."""
    _, exponent = torch.frexp(y)
    two = torch.tensor(2.0, dtype=torch.float64, device=y.device)
    quantum = torch.pow(two, (exponent.clamp(min=-125) - 8).to(torch.float64))
    return (torch.round(y / quantum) * quantum).to(torch.bfloat16)


def bf16_ulp(m: torch.Tensor) -> torch.Tensor:
    """The BF16 ulp of FP64 ``m >= 0``: ``2^(clamp(e, min=-125) - 8)`` with ``(_, e) = frexp(m)``, the quantum rule of
    :func:`bf16_rounded_once` (``2^-133`` for every ``m`` below ``2^-125``), and ``2^-133`` for ``m == 0``."""
    _, exponent = torch.frexp(m)
    two = torch.tensor(2.0, dtype=torch.float64, device=m.device)
    quantum = torch.pow(two, (exponent.clamp(min=-125) - 8).to(torch.float64))
    return torch.where(m == 0, torch.pow(two, -133.0), quantum)


def classify_out(
    out: torch.Tensor, y: torch.Tensor, c0: torch.Tensor, c1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The feature request's ``out`` criterion per element (module docstring): ``(distance, second, outside)``.

    ``out`` is BF16, ``y`` the unrounded FP64 result, ``c0`` and ``c1`` the pair's stored ``C[t0, d]`` and
    ``C[t1, d]`` in FP64, all of one shape.  ``distance`` is the number of BF16 values from ``out`` to ``y`` rounded
    once to BF16 (the first case: at most :data:`OUT_ULPS`).  ``second`` marks the elements beyond that distance whose
    absolute error ``|out - y|`` is at most :func:`bf16_ulp` of ``max(|c0|, |c1|)``; ``outside`` the elements beyond
    that distance that exceed it, which fail the criterion.  The benchmark's gate
    (``benchmarks/cake_gated_kv_compression_baselines.py``) keeps an identical copy of this function and of
    :func:`bf16_ulp`: the tests do not import the benchmark."""
    distance = bf16_ulp_distance(out, bf16_rounded_once(y))
    within = (out.double() - y).abs() <= bf16_ulp(torch.maximum(c0.abs(), c1.abs()))
    beyond = distance > OUT_ULPS
    return distance, beyond & within, beyond & ~within


# ----------------------------------------------------------------------------------------------------------------
# Accuracy against FP64


def test_the_out_criterion_classifies_hand_made_elements() -> None:
    """:func:`classify_out` and :func:`bf16_ulp` on hand-made values (no GPU): an element 2 BF16 values from the
    rounded result whose error is within the ulp of the larger input is a second-case element; one whose error exceeds
    both bounds is outside; ``max(|c0|, |c1|) == 0`` and a subnormal one use the quantum ``2^-133``.  The bound is the
    ulp of the larger magnitude of the two inputs (not of the larger signed value, of ``c0`` alone or of the smaller
    magnitude), an error equal to it is inside, and the error is to the unrounded ``y``, not to BF16(``y``)."""
    q = 2.0**-15  # the BF16 ulp of 0.005
    big = 2.0**-7  # the BF16 ulp of 1.0
    small = 2.0**-18  # the BF16 ulp of 0.00095
    tiny = 2.0**-133  # the smallest BF16 subnormal
    rows = (
        # (out, y, c0, c1, distance, second, outside)
        (164 * q, 0.005, 1.0, -0.99, 0, False, False),  # BF16(0.005) itself
        (165 * q, 0.005, 1.0, -0.99, 1, False, False),  # the next value: first case
        (166 * q, 0.005, 1.0, -0.99, 2, True, False),  # error 6.6e-5 <= 2^-7: second case
        (-166 * q, -0.005, -0.99, 1.0, 2, True, False),  # the same, negated and with the pair swapped
        (252 * small, 0.00095, 0.001, 0.0009, 3, False, True),  # error 1.1e-5 > 2^-17 (ulp of 0.001): outside
        (2 * tiny, 0.0, 0.0, 0.0, 2, False, True),  # max |C| == 0: bound 2^-133, error 2^-132
        (tiny, 0.0, 0.0, 0.0, 1, False, False),  # max |C| == 0, one value off: first case
        (2 * tiny, 0.0, 3 * tiny, -tiny, 2, False, True),  # subnormal max |C|: bound 2^-133, error 2^-132
        (166 * q, 0.005, 0.001, -1.0, 2, True, False),  # bound 2^-7 from |-1.0|, not 2^-17 from 0.001
        (420 * q, 164 * q, 1.0, 0.5, 174, True, False),  # error exactly 2^-7: equal to the bound is inside
        (420 * q, 163.6 * q, 1.0, 0.5, 174, False, True),  # error 256.4 q > 2^-7 (256 q from BF16(y)): outside
    )
    f64 = torch.float64
    out = torch.tensor([r[0] for r in rows], dtype=f64).to(torch.bfloat16)
    assert torch.equal(out.double(), torch.tensor([r[0] for r in rows], dtype=f64)), "a hand-made out is not BF16"
    y, c0, c1 = (torch.tensor([r[i] for r in rows], dtype=f64) for i in (1, 2, 3))
    distance, second, outside = classify_out(out, y, c0, c1)
    assert distance.tolist() == [r[4] for r in rows]
    assert second.tolist() == [r[5] for r in rows]
    assert outside.tolist() == [r[6] for r in rows]
    m = torch.tensor([0.0, tiny, 3 * tiny, 2.0**-126, 2.0**-125, 0.00095, 0.001, 1.0, 1.5, 2.0], dtype=f64)
    want = [tiny, tiny, tiny, tiny, 2.0**-132, small, 2.0**-17, big, big, 2.0**-6]
    assert bf16_ulp(m).tolist() == want


@pytest.mark.parametrize(("pack", "variant"), MEASURED)
def test_out_is_within_one_bf16_ulp_or_one_ulp_of_the_larger_input(pack, variant, request, record_property) -> None:
    """The feature request's ``out`` criterion (:func:`classify_out`) on every element of every seed, and the report
    it asks for, per case: the elements, the second-case count, the count outside both cases (0 to pass) and the worst
    distance in BF16 values.  The report is also printed as one ``OUT_REPORT`` line (shown with ``pytest -s``)."""
    device = _device_or_skip()
    report = {"elements": 0, "second_case": 0, "outside": 0, "worst_ulps": 0}
    worst, worst_excess = None, -float("inf")
    for seed in SEEDS:
        x = make_inputs(pack, seed=seed, device=device, layout="halves", **variant)
        out, _, _ = _autograd_cz(x)
        y = forward_values(x, torch.float64)["out"]
        c = x.c.double()
        c0, c1 = c[x.t0], c[x.t0 + 1]
        distance, second, outside = classify_out(out, y, c0, c1)
        report["elements"] += distance.numel()
        report["second_case"] += int(second.sum())
        report["outside"] += int(outside.sum())
        report["worst_ulps"] = max(report["worst_ulps"], int(distance.max()) if distance.numel() else 0)
        if bool(outside.any()):
            # The outside element farthest beyond its second-case bound (a NaN error counts as the farthest).
            error = (out.double() - y).abs()
            bound = bf16_ulp(torch.maximum(c0.abs(), c1.abs()))
            excess = torch.where(outside, (error / bound).nan_to_num(nan=float("inf")), -float("inf"))
            row, col = divmod(int(excess.argmax()), WIDTH)
            if float(excess[row, col]) > worst_excess:
                worst_excess = float(excess[row, col])
                worst = {
                    "seed": seed,
                    "row": row,
                    "column": col,
                    "out": float(out[row, col]),
                    "fp64": float(y[row, col]),
                    "c_t0": float(c0[row, col]),
                    "c_t1": float(c1[row, col]),
                    "ulps": int(distance[row, col]),
                    "error": float(error[row, col]),
                    "bound": float(bound[row, col]),
                }
    case = request.node.callspec.id
    print(
        f"OUT_REPORT case={case} elements={report['elements']} second_case={report['second_case']} "
        f"outside={report['outside']} worst_ulps={report['worst_ulps']}"
    )
    record_property(
        "acceptance",
        {
            "criterion": "out within 1 BF16 ulp of BF16(FP64), or |error| <= 1 BF16 ulp of max(|C[t0]|, |C[t1]|)",
            "case": case,
            **report,
            "worst_outside": worst,
        },
    )
    assert report["outside"] == 0, (
        f"{report['outside']} out elements meet neither case of the criterion (second case {report['second_case']}, "
        f"worst distance {report['worst_ulps']} BF16 values); worst element {worst}"
    )


@pytest.mark.parametrize(("pack", "variant"), MEASURED)
def test_gradients_are_within_the_relative_l2_bound(pack, variant, record_property) -> None:
    device = _device_or_skip()
    worst = {"dc": 0.0, "dz": 0.0}
    for seed in SEEDS:
        x = make_inputs(pack, seed=seed, device=device, layout="halves", **variant)
        _, dc, dz = _autograd_cz(x)
        ref = backward_values(x, torch.float64)
        for name, got in (("dc", dc), ("dz", dz)):
            worst[name] = max(worst[name], rel_l2(got, ref[name]))
    record_property("acceptance", {"criterion": "grad rel-L2", **worst})
    for name, value in worst.items():
        assert value <= GRAD_REL_L2, f"{name}: relative L2 error {value:.3e} above {GRAD_REL_L2:.0e}"


# ----------------------------------------------------------------------------------------------------------------
# Documents, tails and empty packs


@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED), ids=lambda p: p.id)
def test_the_pack_equals_the_per_document_result(pack) -> None:
    """Each document run alone gives the bits of its rows of the pack's ``out``, ``dC`` and ``dZ``: the six documents
    of ``(0, 1, 3, 2, 7, 4)`` (lengths 0, 1, 2 and 3 among them) and the four of ``(127, 256, 1, 641)``.  A pack of
    one document is not a case: it would compare a call with an identical call."""
    device = _device_or_skip()
    x = make_inputs(pack, seed=101, device=device)
    out, dc, dz = _explicit(x, "contiguous")
    b = bounds(pack.lengths)
    row = 0
    for s, n in enumerate(pack.lengths):
        pairs = n // 2
        y = make_inputs(Pack((n,)), seed=0, device=device)
        y.c, y.z = x.c[b[s] : b[s + 1]].contiguous(), x.z[b[s] : b[s + 1]].contiguous()
        y.dout = x.dout[row : row + pairs].contiguous()
        got_out, got_dc, got_dz = _explicit(y, "contiguous")
        assert_bitwise(got_out, out[row : row + pairs], f"document {s} (length {n}): out")
        assert_bitwise(got_dc, dc[b[s] : b[s + 1]], f"document {s} (length {n}): dC")
        assert_bitwise(got_dz, dz[b[s] : b[s + 1]], f"document {s} (length {n}): dZ")
        row += pairs


@pytest.mark.parametrize("pack", (Pack((0, 1, 3, 2, 7, 4)), MIXED, *PERF_PACKS), ids=lambda p: p.id)
def test_the_tail_tokens_change_nothing_and_get_zero_gradients(pack) -> None:
    device = _device_or_skip()
    tails = tail_rows(pack.lengths)
    x = make_inputs(pack, seed=103, device=device, layout="halves")
    want = _autograd_cz(x)
    for name, value in (("dC", want[1]), ("dZ", want[2])):
        assert_zero_rows(value, tails, f"{pack.id}: {name}")
    index = torch.tensor(tails, dtype=torch.long, device=device)
    x.cz[index, :WIDTH] += 1e6
    x.cz[index, WIDTH:] -= 1e6
    for name, got, ref in zip(("out", "dC", "dZ"), _autograd_cz(x), want, strict=True):
        assert_bitwise(got, ref, f"{pack.id}: {name} after the tail tokens changed")


@pytest.mark.parametrize("lengths", ((), (0,), (1, 0, 1)), ids=("no-documents", "0", "1-0-1"))
def test_packs_without_a_complete_pair_give_an_empty_out_and_zero_gradients(lengths) -> None:
    device = _device_or_skip()
    x = make_inputs(Pack(lengths), seed=107, device=device, layout="halves")
    for out, dc, dz in (_autograd_cz(x), _explicit(x, "halves")):
        assert tuple(out.shape) == (0, WIDTH)
        for value in (dc, dz):
            assert tuple(value.shape) == (x.T, WIDTH)
            assert int((value.view(torch.int32) != 0).sum()) == 0


def test_column_halves_give_the_contiguous_results() -> None:
    device = _device_or_skip()
    for pack in (MIXED, *PERF_PACKS):
        halves = make_inputs(pack, seed=109, device=device, layout="halves")
        dense = make_inputs(pack, seed=109, device=device, layout="contiguous")
        for name, got, want in zip(
            ("out", "dC", "dZ"), _explicit(halves, "halves"), _explicit(dense, "contiguous"), strict=True
        ):
            assert_bitwise(got, want, f"{pack.id}: {name}")


def test_gate_gaps_of_100_and_more_and_ties_are_finite() -> None:
    """Logit gaps of 100, 1000, 2000, 1e30 and an FP32 overflow of the gap, and exact ties (also at 1e30): ``out``,
    ``dC`` and ``dZ`` are finite."""
    device = _device_or_skip()
    pack = MIXED
    x = make_inputs(pack, seed=113, device=device, layout="halves", gates=True)
    t0 = x.t0
    z = x.cz[:, WIDTH:]
    for start, (a, b) in enumerate(((0.0, 100.0), (0.0, -1000.0), (1e30, -1e30), (3e38, -3e38), (1e30, 1e30))):
        cols = slice(384 + 16 * start, 384 + 16 * (start + 1))
        z[t0, cols] = a
        z[t0 + 1, cols] = b
    out, dc, dz = _autograd_cz(x)
    for name, value in (("out", out), ("dC", dc), ("dZ", dz)):
        assert bool(torch.isfinite(value).all()), f"{name} is not finite"


# ----------------------------------------------------------------------------------------------------------------
# Dynamic shapes, checkpointing, synchronization, memory


def test_any_tokens_documents_and_pairs_run_without_recompiling(monkeypatch, record_property) -> None:
    """After one call, twenty packs of other ``T``, ``S`` and ``G`` (one to two thousand documents, empty and odd
    ones, ``G`` above the pair count) build and load no library: the first call (a pack of six documents) loads the
    library of every role served on the device's architecture, both pair tables, both forwards (``fwd``,
    ``fwd_unfused``) and the backwards (``bwd_small``, ``bwd``, ``bwd_warp``) among them, and these serve every shape,
    the packs of more than 26, 125, 256 and 512 documents included."""
    device = _device_or_skip()
    jit = _jit()
    _explicit(make_inputs(MIXED, seed=1, device=device), "contiguous")
    loaded = jit.load.cache_info().currsize
    from cake_pplx._backends.cake_gated_kv_compression_det_train import _cake_pplx_jit

    def refuse(self):
        raise AssertionError(f"a call built {self.name} again")

    monkeypatch.setattr(_cake_pplx_jit.JitSpec, "build", refuse)
    rng = random.Random(127)
    shapes = []
    for _ in range(20):
        S = rng.choice((1, 2, 3, 17, 300, 2000))
        lengths = tuple(rng.choice((0, 1, 2, 3, 8, 64, 255)) for _ in range(S))
        pack = Pack(lengths)
        extra = rng.choice((0, 0, 5))
        x = make_inputs(pack, seed=rng.randrange(1 << 30), device=device, rows=pack.G + extra)
        out = api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
        api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu)
        assert tuple(out.shape) == (x.G, WIDTH)
        shapes.append((pack.T, pack.S, x.G))
    torch.cuda.synchronize(device)
    record_property("acceptance", {"criterion": "shapes without a rebuild", "shapes": shapes})
    assert jit.load.cache_info().currsize == loaded


def test_checkpoint_without_reentrance_gives_the_plain_gradients() -> None:
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=131, device=device)

    def run(checkpointed: bool):
        cz = x.cz.detach().clone().requires_grad_(True)
        fn = api.cake_gated_kv_compression_det_cz
        out = (
            torch.utils.checkpoint.checkpoint(fn, cz, x.cu, x.G, use_reentrant=False)
            if checkpointed
            else fn(cz, x.cu, x.G)
        )
        out.backward(x.dout)
        return out.detach(), cz.grad

    for name, got, want in zip(("out", "grad"), run(True), run(False), strict=True):
        assert_bitwise(got, want, f"checkpointed {name}")


def test_no_host_synchronization_in_steady_state() -> None:
    """After a warm-up call, the explicit steps and the autograd forward and backward run under
    ``torch.cuda.set_sync_debug_mode("error")`` (any synchronizing call raises) and capture in a CUDA graph."""
    device = _device_or_skip()
    x = make_inputs(MIXED, seed=137, device=device, layout="halves")
    _explicit(x, "halves")
    _autograd_cz(x)
    torch.cuda.synchronize(device)
    torch.cuda.set_sync_debug_mode("error")
    try:
        _explicit(x, "halves")
        cz = x.cz.detach().clone().requires_grad_(True)
        api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G).backward(x.dout)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        _explicit(x, "halves")
    graph.replay()
    torch.cuda.synchronize(device)


def _rounded(size: int) -> int:
    """Bytes PyTorch's caching allocator assigns to a request of ``size`` bytes (multiples of 512)."""
    return max(512, -(-size // 512) * 512)


@pytest.mark.parametrize("pack", (MIXED, *PERF_PACKS), ids=lambda p: p.id)
def test_workspace_and_retained_state_are_reported_separately(pack, record_property) -> None:
    """The report's workspace is the pair table an explicit forward allocates and frees (the peak above the output),
    and its retained state is the pair table the autograd node keeps from its forward to its backward (the memory the
    node holds beyond its output)."""
    device = _device_or_skip()
    x = make_inputs(pack, seed=139, device=device, layout="halves")
    report = api.cake_gated_kv_compression_det_train_memory_bytes(x.T, x.S)
    _explicit(x, "halves")
    torch.cuda.synchronize(device)
    # The output's block as the caching allocator sizes it, from a fresh segment: a large request keeps the rest of
    # its segment when that rest is at most 1 MiB, which is not workspace.
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    probe = torch.empty(x.G, WIDTH, dtype=torch.bfloat16, device=device)
    out_bytes = torch.cuda.memory_allocated(device) - base
    del probe
    assert out_bytes >= _rounded(x.G * WIDTH * 2)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = api.cake_gated_kv_compression_det_train_forward(x.c, x.z, x.cu, x.G)
    torch.cuda.synchronize(device)
    workspace = torch.cuda.max_memory_allocated(device) - base - out_bytes
    after = torch.cuda.memory_allocated(device) - base - out_bytes
    del out
    cz = x.cz.detach().clone().requires_grad_(True)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
    torch.cuda.synchronize(device)
    retained = torch.cuda.memory_allocated(device) - base - out_bytes
    out.backward(x.dout)
    record_property(
        "acceptance",
        {"criterion": "memory", "report": report, "workspace": workspace, "retained": retained, "T": x.T, "S": x.S},
    )
    assert workspace == _rounded(report["forward_workspace"]) and after == 0
    assert retained == _rounded(report["retained_for_backward"])
    assert report["retained_inputs"] == 4096 * x.T + 4 * (x.S + 1)


def _accounted(device: torch.device, requests) -> int:
    """Bytes the caching allocator accounts for the ``(shape, dtype)`` requests, made in this order from the current
    state with its cache emptied (a large block keeps the rest of its segment when that rest is at most 1 MiB)."""
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    held = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in requests]
    found = torch.cuda.memory_allocated(device) - base
    del held
    torch.cuda.empty_cache()
    return found


@pytest.mark.parametrize("pack", (MIXED, *PERF_PACKS), ids=lambda p: p.id)
def test_the_backward_workspace_is_reported(pack, record_property) -> None:
    """The peak a backward allocates: the report's ``backward_workspace`` (the pair table) for an explicit backward
    without ``pair_offsets`` into caller buffers; for the autograd backward of ``cake_gated_kv_compression_det_cz``,
    only its ``[T, 1024]`` gradient (an output) with a contiguous incoming gradient, and that gradient plus the
    documented ``1024 G``-byte copy with an expanded one."""
    device = _device_or_skip()
    x = make_inputs(pack, seed=149, device=device, layout="halves")
    report = api.cake_gated_kv_compression_det_train_memory_bytes(x.T, x.S)
    _explicit(x, "halves")
    _, dc, dz = nan_halves(x.T, device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    api.cake_gated_kv_compression_det_train_backward(x.dout, x.c, x.z, x.cu, dc=dc, dz=dz)
    torch.cuda.synchronize(device)
    explicit = torch.cuda.max_memory_allocated(device) - base
    after = torch.cuda.memory_allocated(device) - base
    gradient = ((x.T, 2 * WIDTH), torch.float32)
    copy = ((x.G, WIDTH), torch.bfloat16)
    expanded = x.dout[:1].expand(x.G, WIDTH)
    found = {}
    for label, dout, requests in (("contiguous", x.dout, [gradient]), ("expanded", expanded, [gradient, copy])):
        cz = x.cz.detach().clone().requires_grad_(True)
        out = api.cake_gated_kv_compression_det_cz(cz, x.cu, x.G)
        torch.cuda.synchronize(device)
        # The backward allocates its gradient first, then the copy of a non-contiguous incoming gradient.
        want = _accounted(device, requests)
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        (grad,) = torch.autograd.grad(out, cz, dout)
        torch.cuda.synchronize(device)
        found[label] = (torch.cuda.max_memory_allocated(device) - base, want)
        del out, grad, cz
    copied = found["expanded"][1] - found["contiguous"][1]
    record_property(
        "acceptance",
        {"criterion": "backward workspace", "report": report, "explicit": explicit, "autograd": found, "G": x.G},
    )
    assert explicit == _rounded(report["backward_workspace"]) and after == 0
    for label, (peak, want) in found.items():
        assert peak == want, f"autograd backward, {label} incoming gradient: peak {peak} B, expected {want} B"
    assert copied >= _rounded(1024 * x.G), (copied, x.G)
