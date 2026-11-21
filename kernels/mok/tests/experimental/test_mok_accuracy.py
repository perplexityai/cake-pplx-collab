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

Accuracy of the single-GPU MoK kernels: every floating output against the float64 oracle, beside the naive PyTorch
BF16 implementation, over :data:`tests.cake_naive.DRAWS` random inputs per case, and over stage 2's 128 where an output's
max-norm wins leave it undecided (:func:`tests.cake_naive.judge_case`).

The oracle and the naive implementation are defined in :mod:`tests.experimental._mok_naive`.  The cases are the fused
BF16 training cases (three geometries, ring replay with several macrobatches, empty experts, clamped SwiGLU, FP32
weight-gradient accumulation onto nonzero starts, 36 and 9 local experts at top-8), the fused MXFP8 training cases
(plain and clamped, BF16 and FP32 weight gradients, 36 and 9 local experts at top-8), the forward epilogue, and the
router weight before the down projection on SM100a, SM103a and SM107a (clamped SwiGLU, FP32 weight gradients): top-2
over the three geometries, top-6 over 6 to 96 local experts, half of the weights exactly zero, the requester's input
distribution at hidden 5120 / intermediate 2304 over 6 and 48 local experts
(:func:`tests.experimental._mok_reference.production_inputs`, its gate and up exactly at the limit on one route), and
the adapter's top-6 step at each 384-expert layout as rank 0 of its EP group emulated on one GPU
(:mod:`tests.experimental._mok_one_gpu`; ``y``, ``dx`` and the router-weight gradients), and at the EP64 and EP8 layouts
over changing source counts on one workspace (padded and unpadded staging; also the shared weight gradients).

An FP32 weight-gradient accumulator is judged by its increment, ``accumulator - start`` in float64 (exact for these
values), against the oracle's weight gradient: the same error as the accumulator's against ``start`` plus the exact
gradient.  E4M3 values the MXFP8 kernels store (the saved gate, up and activation of the resident macrobatch) are judged
as their dequantized values against the oracle's value before quantization; ``x``'s E4M3 codes and UE8M0 scales are
exact against the recipe.  The kernel-shaped alternatives of :data:`tests.experimental._mok_naive.ALTERNATIVE` are
measured on the same draws and reported (``ALT`` lines), never judged.  The known accuracy gaps
(:mod:`tests.experimental._mok_gaps`) are judged in items of their own, under a strict expected-failure marker on the
architectures where they miss the tier: the MXFP8 score gradient of every MXFP8 case, and routed weight gradients of
the two BF16 cases whose routed experts cross a macrobatch boundary (:data:`STRADDLE_GAPS`).

Backend suite: the package is imported only through :func:`tests.cake.test_helpers.cake_package.cake_package`.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import TierJudge, judge_case
from tests.experimental import _mok_gaps as gaps
from tests.experimental import _mok_naive as naive
from tests.experimental._mok_one_gpu import (
    CHANGED_COUNTS,
    OneGpuGroup,
    bound_multiplier,
    one_gpu_source_workspace,
    one_gpu_workspace,
)
from tests.experimental._mok_reference import (
    mxfp8_quantize,
    production_inputs,
    require_gpu,
    single_rank_schedule,
)

kernels = cake_package("cake_mok_bf16", "_kernels")
backend = cake_package("cake_mok_bf16", "backend")
workspace_module = cake_package("cake_mok_bf16", "workspace")

GEOMETRIES = ((256, 256, 768), (512, 512, 768), (512, 256, 1536))
MXFP8_GEOMETRIES = ((512, 256, 768), (512, 512, 1536), (512, 256, 512))


@pytest.fixture(autouse=True)
def _exact_matmul(cuda_matmul_flags):
    cuda_matmul_flags(allow_tf32=False, allow_bf16_reduced_precision_reduction=False)


def _segments(s):
    """Per expert of a single-rank schedule: its padded rows in schedule order."""
    out, offset = [], 0
    for expert, count in enumerate(s["count_values"]):
        out.append(naive.Segment(expert, s["schedule"][offset : offset + count], offset))
        offset += count
    return out


def _weights(experts, hidden, intermediate, options):
    shared = [
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    return shared, routed


def _scores(tokens, topk):
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    return scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)


DW_NAMES = (
    "dw_shared_gate",
    "dw_routed_gate",
    "dw_shared_up",
    "dw_routed_up",
    "dw_shared_down",
    "dw_routed_down",
)


#: The index into the ``(*shared, *routed)`` FP32 accumulators of each of :data:`DW_NAMES`.
ACCUMULATOR = (0, 3, 1, 4, 2, 5)


def _accumulators(shared, routed):
    """Nonzero FP32 starts of the ``(*shared, *routed)`` weight-gradient accumulators, and the accumulators."""
    initial = [torch.randn(w.shape, device="cuda") * 0.01 for w in (*shared, *routed)]
    return initial, [value.clone() for value in initial]


def _weight_outputs(result, initial):
    """The six weight gradients of a backward result (indices 9 - 14), FP32 accumulators as their float64
    increments over their starts."""
    out = {}
    for i, name in enumerate(DW_NAMES):
        value = result[9 + i]
        if initial is not None:
            value = value.double() - initial[ACCUMULATOR[i]].double()
        out[name] = value
    return out


def _reference_values(r, sh, dtype_cast=None):
    """Oracle or naive values by output name from :func:`_mok_naive.routed` / :func:`_mok_naive.shared`."""
    tokens_topk = r["y_route"].shape[0]
    values = dict(
        y_route=r["y_route"],
        y_shared=sh["y"],
        dx_route=r["dx_route"],
        dx_shared=sh["dx"],
        dw_shared_gate=sh["dw"][0],
        dw_routed_gate=r["dw"][0],
        dw_shared_up=sh["dw"][1],
        dw_routed_up=r["dw"][1],
        dw_shared_down=sh["dw"][2],
        dw_routed_down=r["dw"][2],
        d_gate_shared=sh["d_gate"],
        d_up_shared=sh["d_up"],
        dh_shared=sh["dh"],
    )
    values["d_scores"] = r["d_scores"].view(values["y_shared"].shape[0], tokens_topk // values["y_shared"].shape[0])
    return values


def _check(record_property, part, judged, known=()):
    """Judge item ``part`` of a case (:func:`tests.experimental._mok_gaps.check`) from the case's shared run
    ``judged`` = ``(judge, alternatives, failures)``; the main item also records the alternatives."""
    judge, alt, failures = judged
    if part == gaps.OUTPUTS:
        for verdict in alt.verdicts().values():
            record_property("naive_alternative", verdict)
    gaps.check(record_property, part, list(judge.verdicts().values()), failures, known)


def _bf16_case(
    *,
    swiglu_limit=None,
    fp32_wgrad=False,
    experts=4,
    topk=2,
    tokens=512,
    geometry=GEOMETRIES[0],
    empty=False,
    mini=256,
    comm=4,
    seed=9173,
):
    hidden, intermediate, macro = geometry
    clamped = swiglu_limit is not None
    forward, backward = kernels.MoKForward(clamped), kernels.MoKBackward(clamped, fp32_wgrad)
    options = dict(device="cuda", dtype=torch.bfloat16)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row + slot) % (topk if empty else experts) for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    segments = _segments(s)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    alt = TierJudge()
    naive_values = {}

    def run(j, _seed):
        x = torch.randn(tokens, hidden, **options) * 0.125
        dy = torch.randn_like(x) * 0.125
        shared, routed = _weights(experts, hidden, intermediate, options)
        scores = _scores(tokens, topk)
        combine = torch.zeros(tokens * topk, hidden, **options)
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        initial = accumulators = None
        if fp32_wgrad:
            initial, accumulators = _accumulators(shared, routed)
        weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
        context = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()], *weights,
            *schedule, topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            *weights, *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        torch.cuda.synchronize()
        outputs = dict(
            y_route=combine,
            y_shared=context[7],
            dx_route=dx_peer,
            d_scores=ds_peer,
            dx_shared=result[0],
            d_gate_shared=result[2],
            d_up_shared=result[4],
            dh_shared=result[6],
            **_weight_outputs(result, initial),
        )
        x_routes, dy_routes = x.repeat_interleave(topk, 0), dy.repeat_interleave(topk, 0)
        flat = scores.flatten()
        values = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            r = naive.routed(
                x_routes, dy_routes, flat, segments, routed, swiglu_limit, dtype,
                macro=None if fp32_wgrad else macro,
            )  # fmt: skip
            sh = naive.shared(x, dy, shared, swiglu_limit, dtype)
            values[dtype] = (r, _reference_values(r, sh))
        oracle = values[naive.ORACLE][1]
        r_naive, naive_values[j] = values[naive.NAIVE]
        if not fp32_wgrad:
            for i, name in enumerate(("dw_routed_gate", "dw_routed_up", "dw_routed_down")):
                alt.add(name, j, outputs[name], oracle[name], {"bf16": r_naive["alt_dw"][i]})
        alt.add("d_scores", j, outputs["d_scores"], oracle["d_scores"], {"bf16": r_naive["alt_d_scores"].view(tokens, topk)})
        return outputs, oracle

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed + hidden, reported=(alt,))
    return judge, alt, failures


def _bf16(record_property, part="outputs", known=(), **kwargs):
    require_gpu()
    _check(record_property, part, gaps.judged(("bf16", *sorted(kwargs.items())), lambda: _bf16_case(**kwargs)), known)


#: The fused BF16 cases ``(geometry, empty)`` whose routed experts cross a macrobatch boundary (the empty-experts
#: layout puts all routed rows on few experts: with 768-row macrobatches their rows span two macrobatches), and their
#: outputs judged in items of their own: the routed weight gradients (BF16 weight gradients) that miss the tier on at
#: least one architecture (:data:`tests.experimental._mok_gaps.CAPS`), each marked where it does.
STRADDLE_GAPS = {
    ((256, 256, 768), True): ("dw_routed_gate", "dw_routed_up", "dw_routed_down"),
    ((512, 512, 768), True): ("dw_routed_gate", "dw_routed_up", "dw_routed_down"),
}


def _fused_cases():
    """``(geometry, empty, part)`` parameters: one item per case; a case of :data:`STRADDLE_GAPS` gets its main
    item, then one item per known-gap output under its marker."""
    cases = []
    for geometry in GEOMETRIES:
        for empty in (False, True):
            case = "h{}-i{}-m{}-{}".format(*geometry, "empty-experts" if empty else "routed")
            if (geometry, empty) not in STRADDLE_GAPS:
                cases.append(pytest.param(geometry, empty, gaps.OUTPUTS, id=case))
                continue
            known = [(name, gaps.bf16_straddle_dw) for name in STRADDLE_GAPS[(geometry, empty)]]
            cases += [
                pytest.param(geometry, empty, *p.values, marks=p.marks, id=p.id)
                for p in gaps.parts("test_fused_training_accuracy", known, case)
            ]
    return cases


@pytest.mark.parametrize("geometry,empty,part", _fused_cases())
def test_fused_training_accuracy(record_property, geometry, empty, part):
    _bf16(record_property, part, STRADDLE_GAPS.get((geometry, empty), ()), geometry=geometry, empty=empty)


def test_fused_training_clamped_swiglu_accuracy(record_property):
    """A limit near the activations' standard deviation clamps a large fraction of gate and up."""
    _bf16(record_property, swiglu_limit=0.125)


@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: "h{}-i{}-m{}".format(*g))
def test_fused_training_fp32_wgrad_accuracy(record_property, geometry):
    _bf16(record_property, fp32_wgrad=True, geometry=geometry)


@pytest.mark.parametrize("experts", [36, 9], ids=["local36", "local9"])
def test_fused_training_local_experts_accuracy(record_property, experts):
    """288 routed experts over 8 or 32 ranks: 36 or 9 local experts, top-8, clamped."""
    _bf16(record_property, swiglu_limit=0.125, experts=experts, topk=8, geometry=(512, 256, 2048))


def _mxfp8_case(
    *,
    swiglu_limit,
    geometry,
    fp32_wgrad=False,
    experts=4,
    topk=2,
    tokens=512,
    mini=256,
    comm=4,
    seed=7000,
):
    hidden, intermediate, macro = geometry
    clamped = swiglu_limit is not None
    forward = kernels.MoKForwardMXFP8(clamped)
    backward = kernels.MoKBackwardMXFP8(clamped, fp32_wgrad)
    options = dict(device="cuda", dtype=torch.bfloat16)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    s = single_rank_schedule(routes, experts)
    segments = _segments(s)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    actual = s["actual"]
    resident = min(actual, macro)
    slab_routes = s["schedule"][:actual]
    alt = TierJudge()
    naive_values = {}

    def run(j, _seed):
        x = torch.randn(tokens, hidden, **options)
        dy = torch.randn(tokens, hidden, **options) * 0.125
        shared, routed = _weights(experts, hidden, intermediate, options)
        scores = _scores(tokens, topk)
        q = [mxfp8_quantize(w) for w in routed]
        fwd_w = [(qq[0], qq[1]) for qq in q]
        bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
        combine = torch.zeros(tokens * topk, hidden, **options)
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        initial = accumulators = None
        if fp32_wgrad:
            initial, accumulators = _accumulators(shared, routed)
        values = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        torch.cuda.synchronize()
        (x_t, x_sc_t), _, (gate_q, gate_sc), _, (up_q, up_sc), _, (h_t, h_sc_t), y_shared, _ = values
        r = resident
        saved = dict(
            gate_saved=naive.dequantized((gate_q[:r], gate_sc[: r // 128])),
            up_saved=naive.dequantized((up_q[:r], up_sc[: r // 128])),
            hidden_saved=naive.dequantized((h_t[:, :r], h_sc_t[:, : r // 128]), transposed=True),
        )
        # Copies: the backward reuses the context ring for the later macrobatches, and a slice that spans the whole ring
        # is a view of it.
        x_codes = x_t[:, :r].clone().view(torch.uint8)
        x_scales = x_sc_t[:, : r // 128].clone().view(torch.uint8)
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *values[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        torch.cuda.synchronize()
        outputs = dict(
            y_route=combine,
            y_shared=y_shared,
            dx_route=dx_peer,
            d_scores=ds_peer,
            dx_shared=result[0],
            **_weight_outputs(result, initial),
            **saved,
            x_codes=x_codes,
            x_scales=x_scales,
        )
        # The recipe's x codes and scales of the resident macrobatch (schedule order, padding rows zero).
        valid = slab_routes >= 0
        source = torch.where(valid[:, None], x[slab_routes.clamp(min=0).long() // topk], torch.zeros_like(x[:1]))
        _, _, xq_t, xs_t = mxfp8_quantize(source)
        exact = dict(
            x_codes=xq_t.view(torch.uint8)[:, :r],
            x_scales=xs_t.view(hidden // 128, actual // 128, 32, 16)[:, : r // 128],
        )
        x_routes, dy_routes = x.repeat_interleave(topk, 0), dy.repeat_interleave(topk, 0)
        mx = dict(forward=fwd_w, backward=bwd_w)
        flat = scores.flatten()
        refs = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            rr = naive.routed(
                x_routes, dy_routes, flat, segments, routed, swiglu_limit, dtype,
                mx=mx, macro=None if fp32_wgrad else macro,
            )  # fmt: skip
            sh = naive.shared(x, dy, shared, swiglu_limit, dtype)
            v = _reference_values(rr, sh)
            for name in ("d_gate_shared", "d_up_shared", "dh_shared"):
                v.pop(name)
            gate, up, hid = (naive.slab(rr["mx_rows"], n, actual)[:r] for n in ("gate", "up", "hidden"))
            if dtype == naive.ORACLE:
                v.update(gate_saved=gate, up_saved=up, hidden_saved=hid, **exact)
            else:
                v.update(
                    gate_saved=naive.qdq(gate),
                    up_saved=naive.qdq(up),
                    hidden_saved=naive.qdq(hid, transposed=True),
                )
            refs[dtype] = (rr, v)
        oracle = refs[naive.ORACLE][1]
        r_naive, naive_values[j] = refs[naive.NAIVE]
        if not fp32_wgrad:
            for i, name in enumerate(("dw_routed_gate", "dw_routed_up", "dw_routed_down")):
                alt.add(name, j, outputs[name], oracle[name], {"bf16": r_naive["alt_dw"][i]})
        alt.add("d_scores", j, outputs["d_scores"], oracle["d_scores"], {"bf16": r_naive["alt_d_scores"].view(tokens, topk)})
        return outputs, oracle

    judge, failures = judge_case(
        run,
        lambda j, tier: naive_values.pop(j),
        seed=seed + hidden + intermediate + macro,
        exact=frozenset({"x_codes", "x_scales"}),
        reported=(alt,),
    )
    return judge, alt, failures


#: The known-gap outputs of every MXFP8 case.
MXFP8_GAPS = ("d_scores",)


def _mxfp8_cases(function, cases):
    """``(*values, part)`` parameters of MXFP8 test ``function`` for ``cases`` (``(id, values)``): per case its main
    item, then the score gradient in an item of its own under that item's marker (:func:`_mok_gaps.parts`)."""
    known = [(name, gaps.mxfp8_d_scores) for name in MXFP8_GAPS]
    return [
        pytest.param(*values, *p.values, marks=p.marks, id=p.id)
        for case, values in cases
        for p in gaps.parts(function, known, case)
    ]


def _mxfp8(record_property, part, **kwargs):
    require_gpu()
    run = gaps.judged(("mxfp8", *sorted(kwargs.items())), lambda: _mxfp8_case(**kwargs))
    _check(record_property, part, run, MXFP8_GAPS)


#: ``(id, (geometry, swiglu_limit))`` of the plain and clamped MXFP8 cases of each geometry.
MXFP8_CASES = [
    ("h{}-i{}-m{}-{}".format(*geometry, name), (geometry, limit))
    for geometry in MXFP8_GEOMETRIES
    for limit, name in ((None, "plain"), (0.75, "clamped"))
]


@pytest.mark.parametrize("geometry,swiglu_limit,part", _mxfp8_cases("test_mxfp8_training_accuracy", MXFP8_CASES))
def test_mxfp8_training_accuracy(record_property, swiglu_limit, geometry, part):
    _mxfp8(record_property, part, swiglu_limit=swiglu_limit, geometry=geometry)


@pytest.mark.parametrize(
    "geometry,swiglu_limit,part", _mxfp8_cases("test_mxfp8_training_fp32_wgrad_accuracy", MXFP8_CASES)
)
def test_mxfp8_training_fp32_wgrad_accuracy(record_property, swiglu_limit, geometry, part):
    _mxfp8(record_property, part, swiglu_limit=swiglu_limit, geometry=geometry, fp32_wgrad=True)


@pytest.mark.parametrize(
    "experts,part",
    _mxfp8_cases("test_mxfp8_training_local_experts_accuracy", [("local36", (36,)), ("local9", (9,))]),
)
def test_mxfp8_training_local_experts_accuracy(record_property, experts, part):
    """288 routed experts over 8 or 32 ranks: 36 or 9 local experts, top-8, clamped."""
    _mxfp8(record_property, part, swiglu_limit=0.75, geometry=(512, 256, 2048), experts=experts, topk=8)


@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("tokens,hidden", [(512, 256), (37, 6144), (1, 1032)])
def test_epilogue_forward_accuracy(record_property, topk, tokens, hidden):
    """``y = shared + sum_k s_k routed_k``: FP32 accumulation from the shared row, then ascending slots."""
    require_gpu()
    epilogues = kernels.MoKEpilogues(topk)
    naive_values = {}

    def run(j, _seed):
        shared = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
        routed = torch.randn(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
        scores = torch.rand(tokens, topk, device="cuda")
        y = epilogues.forward(shared, routed, scores)
        naive_values[j] = dict(y=naive.epilogue_forward(shared, routed, scores, naive.NAIVE))
        return dict(y=y), dict(y=naive.epilogue_forward(shared, routed, scores, naive.ORACLE))

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=6100 + tokens + topk)
    judge.record(record_property)
    assert not failures, "\n".join(failures)


def _require_before_down():
    """Skip on a target without the router-weight-before-down kernels (``BEFORE_DOWN_TARGETS``: SM100a, SM103a and
    SM107a have them)."""
    require_gpu()
    arch = gaps.TARGETS[torch.cuda.get_device_capability()]
    if arch not in kernels.BEFORE_DOWN_TARGETS:
        pytest.skip(f"The router weight before the down projection has no {arch} build")


def _before_down_weights(tokens, topk, zeros):
    """FP32 router weights in [0, 1.5): every 5th token's first weight and every weight of token 3 exactly 0; with
    ``zeros`` also every second slot's."""
    scores = torch.rand(tokens, topk, device="cuda") * 1.5
    scores[::5, 0] = 0.0
    scores[3] = 0.0
    if zeros:
        scores[:, 1::2] = 0.0
    return scores


def _before_down_case(
    *,
    experts=4,
    topk=2,
    tokens=512,
    geometry=GEOMETRIES[0],
    empty=False,
    swiglu_limit=0.125,
    zeros=False,
    production=False,
    mini=256,
    comm=4,
    seed=8311,
):
    """One router-weight-before-down case: forward + backward (clamped SwiGLU, FP32 accumulators from nonzero
    starts) per draw, every floating output against the oracle beside the naive implementation.  ``production``
    draws the requester's inputs (:func:`production_inputs`, its routes and a schedule per draw) and, right after the
    forward (the backward rewrites the context's ring rows while it replays the later macrobatches), checks that the
    saved gate and up of the resident macrobatch hold values exactly at the limit and beyond it, and that the pinned
    route's saved gate and up are exactly the limit in every draw that holds its row in the resident macrobatch;
    the case fails if no draw holds it."""
    hidden, intermediate, macro = geometry
    forward = kernels.MoKForward(True, weight_before_down=True)
    backward = kernels.MoKBackward(True, True, weight_before_down=True)
    options = dict(device="cuda", dtype=torch.bfloat16)
    row = torch.arange(tokens, device="cuda")
    fixed = torch.stack([(row + slot) % (topk if empty else experts) for slot in range(topk)], -1)
    naive_values = {}
    pinned_draws = []

    def run(j, draw):
        if production:
            p = production_inputs(tokens, hidden, intermediate, experts, topk, swiglu_limit, seed=draw)
            x, dy, scores, shared, routed, routes = p["x"], p["dy"], p["weights"], p["shared"], p["routed"], p["ids"]
        else:
            x = torch.randn(tokens, hidden, **options) * 0.125
            dy = torch.randn_like(x) * 0.125
            shared, routed = _weights(experts, hidden, intermediate, options)
            scores = _before_down_weights(tokens, topk, zeros)
            routes = fixed
        s = single_rank_schedule(routes, experts)
        schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
        combine = torch.zeros(tokens * topk, hidden, **options)
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        initial, accumulators = _accumulators(shared, routed)
        weights = (shared[0], routed[0], shared[1], routed[1], shared[2], routed[2])
        context = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()], *weights,
            *schedule, topk, swiglu_limit, comm, macro, mini, weight_ptrs=[scores.data_ptr()],
        )  # fmt: skip
        if production:
            # The forward's saved context, before the backward rewrites its ring rows.
            torch.cuda.synchronize()
            resident = s["schedule"][: min(s["actual"], macro)]
            defined = resident >= 0
            gate, up = context[2][: resident.numel()][defined], context[4][: resident.numel()][defined]
            assert (gate > swiglu_limit).any().item() and (up.abs() > swiglu_limit).any().item(), j
            assert ((gate == swiglu_limit).sum() + (up.abs() == swiglu_limit).sum()).item() > 0, j
            assert p["pinned"] is not None
            token, expert = p["pinned"]
            slot = token * topk + int((routes[token] == expert).nonzero()[0, 0])
            position = (resident[defined] == slot).nonzero()
            if position.numel():
                pinned_gate, pinned_up = gate[position[0, 0]], up[position[0, 0]]
                assert pinned_gate[0].item() == swiglu_limit and pinned_up[0].item() == swiglu_limit, j
                assert pinned_up[1].item() == -swiglu_limit, j
                pinned_draws.append(j)
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            *weights, *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        torch.cuda.synchronize()
        outputs = dict(
            y_route=combine,
            y_shared=context[7],
            dx_route=dx_peer,
            d_scores=ds_peer,
            dx_shared=result[0],
            d_gate_shared=result[2],
            d_up_shared=result[4],
            dh_shared=result[6],
            **_weight_outputs(result, initial),
        )
        x_routes, dy_routes = x.repeat_interleave(topk, 0), dy.repeat_interleave(topk, 0)
        flat = scores.flatten()
        segments = _segments(s)
        values = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            r = naive.routed(
                x_routes, dy_routes, flat, segments, routed, swiglu_limit, dtype, weight_before_down=True,
            )  # fmt: skip
            sh = naive.shared(x, dy, shared, swiglu_limit, dtype)
            values[dtype] = _reference_values(r, sh)
        naive_values[j] = values[naive.NAIVE]
        return outputs, values[naive.ORACLE]

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed + hidden + experts)
    if production and not pinned_draws:
        failures = [*failures, "pinned route: no draw held its row in the resident macrobatch"]
    return judge, TierJudge(), failures


def _before_down(record_property, **kwargs):
    _require_before_down()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(("before_down", *sorted(kwargs.items())), lambda: _before_down_case(**kwargs)),
    )


@pytest.mark.parametrize("empty", [False, True], ids=["routed", "empty-experts"])
@pytest.mark.parametrize("geometry", GEOMETRIES, ids=lambda g: "h{}-i{}-m{}".format(*g))
def test_before_down_training_accuracy(record_property, geometry, empty):
    """Top-2 over 4 local experts (the diagnostic layouts), the three geometries, ring replay and empty experts."""
    _before_down(record_property, geometry=geometry, empty=empty)


@pytest.mark.parametrize("experts", [6, 12, 24, 48, 96], ids=lambda e: f"local{e}")
def test_before_down_top6_accuracy(record_property, experts):
    """Top-6 over the local expert counts of the 384-expert layouts (EP 64, 32, 16, 8, 4), several macrobatches."""
    _before_down(record_property, experts=experts, topk=6, geometry=(512, 256, 1536))


def test_before_down_zero_weights_accuracy(record_property):
    """Half of the router weights exactly zero: their routes' outputs and input gradients are zero, their weight
    gradients are not."""
    _before_down(record_property, experts=6, topk=6, geometry=(512, 256, 1536), zeros=True)


@pytest.mark.parametrize(
    "experts,geometry,tokens,mini",
    [(6, (5120, 2304, 1536), 512, 256), (48, (5120, 2304, 4096), 512, 256), (6, (5120, 2304, 4096), 1024, 4096)],
    ids=["local6", "local48", "local6-mini4096"],
)
def test_before_down_production_shape_accuracy(record_property, experts, geometry, tokens, mini):
    """The requester's inputs (unit-RMS ``x``, ``sqrtsoftplus`` top-6 weights times 1.5 with exact zeros, gate and up
    past the limit 10 on a small fraction and exactly at it on one route) at hidden 5120 and intermediate 2304, the
    production layouts' 6 and 48 local experts, several macrobatches; the requester's minibatch of 4,096 rows on 6
    local experts."""
    _before_down(
        record_property, experts=experts, topk=6, geometry=geometry, swiglu_limit=10.0, production=True,
        tokens=tokens, mini=mini,
    )  # fmt: skip


def _before_down_adapter_case(
    *, ep, experts, tokens=512, hidden=512, intermediate=256, macro=1536, mini=256, swiglu_limit=0.125, seed=8417
):  # fmt: skip
    """The adapter's top-6 router-weight-before-down step (``build_schedule``, ``forward``, ``backward``; clamped
    SwiGLU, FP32 accumulators, several macrobatches) at a 384-expert layout, as rank 0 of its EP group emulated on one
    GPU (:mod:`tests.experimental._mok_one_gpu`): every token routes to this rank's experts and reaches them from every
    emulated peer, so ``y``, ``dx`` and the router-weight gradients are those of a single-rank layer over this rank's
    experts.  Those three are judged per draw against the oracle beside the naive implementation of that layer; the
    routed weight gradients sum EP copies of every route and are checked exactly in ``test_mok_bf16.py``."""
    topk = 6
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep, experts, topk, clamped_swiglu=True, fp32_wgrad=True, router_weight_position="before_down"
    )
    functional.communication = OneGpuGroup()
    multiplier = bound_multiplier(tokens, topk, ep, experts)
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, multiplier)
    workspace = one_gpu_workspace(ep, tokens, hidden, topk, capacity, device)
    config = workspace_module.MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    options = dict(device=device, dtype=torch.bfloat16)
    naive_values = {}

    def run(j, _draw):
        x = torch.randn(tokens, hidden, **options) * 0.125
        dy = torch.randn_like(x) * 0.125
        shared, routed = _weights(experts, hidden, intermediate, options)
        scores = _before_down_weights(tokens, topk, False)
        ids = torch.rand(tokens, experts, device=device).argsort(-1)[:, :topk].contiguous()
        weights = (*shared, *routed)
        accumulators = [torch.zeros(w.shape, device=device) for w in weights]
        schedule = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
        assert int(schedule.num_tokens.item()) > macro  # several macrobatches: backward replays the later ones
        y, context = functional.forward(
            config, workspace, schedule, x, scores, *weights, swiglu_limit=swiglu_limit
        )  # fmt: skip
        gradients = functional.backward(
            config, workspace, schedule, context, dy, x, scores, *weights, swiglu_limit=swiglu_limit,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        torch.cuda.synchronize()
        outputs = dict(y=y.clone(), dx=gradients[0].clone(), d_scores=gradients[1].clone())
        segments = _segments(single_rank_schedule(ids, experts))
        values = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            r = naive.layer(x, dy, scores, shared, routed, segments, swiglu_limit, dtype, weight_before_down=True)
            values[dtype] = {name: r[name] for name in outputs}
        naive_values[j] = values[naive.NAIVE]
        return outputs, values[naive.ORACLE]

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed + ep + experts)
    return judge, TierJudge(), failures


@pytest.mark.parametrize(
    "ep,experts", kernels.SCHEDULER_LAYOUTS_384, ids=[f"ep{e}-local{n}" for e, n in kernels.SCHEDULER_LAYOUTS_384]
)
def test_before_down_adapter_top6_accuracy(record_property, ep, experts):
    """The adapter's top-6 step at the 384-expert layouts (EP 4, 8, 16, 32, 64), emulated on one GPU as rank 0 of the
    group with every route on this rank: ``y``, ``dx`` and the router-weight gradients."""
    _require_before_down()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(("before_down_adapter", ep, experts), lambda: _before_down_adapter_case(ep=ep, experts=experts)),
    )


#: The shared weight gradients of :func:`tests.experimental._mok_naive.layer` (``dw`` entries 3-5, stacked over one
#: source rank) and their indexes in the adapter's ``backward`` result.
SHARED_DW = (("dw_shared_gate", 3, 5), ("dw_shared_up", 4, 6), ("dw_shared_down", 5, 7))


def _before_down_adapter_counts_case(
    *, ep, experts, hidden=512, intermediate=256, macro=1536, mini=256, swiglu_limit=0.125, seed=8423
):  # fmt: skip
    """The adapter's top-6 router-weight-before-down step over the source counts of
    :data:`tests.experimental._mok_one_gpu.CHANGED_COUNTS` in order, on one source workspace whose capacity covers the
    largest (padded staging of rows ``[count, shared_rows)`` at 1031, 257 and 513 rows, each smaller count after a
    larger one; unpadded at 768), as rank 0 of its EP group emulated on one GPU (as :func:`_before_down_adapter_case`).
    Per draw and count: new inputs, a new schedule, ``forward`` and ``backward`` with zeroed FP32 accumulators;
    ``y``, ``dx``, the router-weight gradients and the shared weight gradients (this rank's own, over the count's rows
    and the zero rows of the padding) are judged against the oracle beside the naive implementation of the
    single-rank layer, as ``rows<count>:<output>``."""
    topk = 6
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep, experts, topk, clamped_swiglu=True, fp32_wgrad=True, router_weight_position="before_down"
    )
    functional.communication = OneGpuGroup()
    workspace, multiplier = one_gpu_source_workspace(ep, experts, CHANGED_COUNTS, hidden, topk, device)
    config = workspace_module.MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    options = dict(device=device, dtype=torch.bfloat16)
    naive_values = {}

    def run(j, _draw):
        shared, routed = _weights(experts, hidden, intermediate, options)
        weights = (*shared, *routed)
        outputs, oracle, naive_values[j] = {}, {}, {}
        for count in CHANGED_COUNTS:
            x = torch.randn(count, hidden, **options) * 0.125
            dy = torch.randn_like(x) * 0.125
            scores = _before_down_weights(count, topk, False)
            ids = torch.rand(count, experts, device=device).argsort(-1)[:, :topk].contiguous()
            accumulators = [torch.zeros(w.shape, device=device) for w in weights]
            schedule = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
            assert schedule.num_source_tokens == count and int(schedule.num_tokens.item()) > macro
            y, context = functional.forward(
                config, workspace, schedule, x, scores, *weights, swiglu_limit=swiglu_limit
            )  # fmt: skip
            gradients = functional.backward(
                config, workspace, schedule, context, dy, x, scores, *weights, swiglu_limit=swiglu_limit,
                weight_grad_accumulators=accumulators,
            )  # fmt: skip
            torch.cuda.synchronize()
            values = dict(y=y, dx=gradients[0], d_scores=gradients[1])
            values.update((name, gradients[index]) for name, _, index in SHARED_DW)
            segments = _segments(single_rank_schedule(ids, experts))
            for dtype, target in ((naive.ORACLE, oracle), (naive.NAIVE, naive_values[j])):
                r = naive.layer(x, dy, scores, shared, routed, segments, swiglu_limit, dtype, weight_before_down=True)
                reference = dict(y=r["y"], dx=r["dx"], d_scores=r["d_scores"])
                reference.update((name, r["dw"][index][0]) for name, index, _ in SHARED_DW)
                target.update((f"rows{count}:{name}", reference[name]) for name in values)
            outputs.update((f"rows{count}:{name}", value.clone()) for name, value in values.items())
        return outputs, oracle

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed + ep + experts)
    return judge, TierJudge(), failures


@pytest.mark.parametrize("ep,experts", [(64, 6), (8, 48)], ids=["ep64-local6", "ep8-local48"])
def test_before_down_adapter_top6_changed_counts_accuracy(record_property, ep, experts):
    """The adapter's top-6 step over changing source counts on one source workspace (1031, 768, 257 and 513 rows; the
    padded staging at every count that is not a multiple of 256), at the requester's EP64 and EP8 layouts, emulated on
    one GPU as rank 0 of the group with every route on this rank: per count ``y``, ``dx``, the router-weight gradients
    and the shared weight gradients."""
    _require_before_down()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(
            ("before_down_adapter_counts", ep, experts),
            lambda: _before_down_adapter_counts_case(ep=ep, experts=experts),
        ),
    )
