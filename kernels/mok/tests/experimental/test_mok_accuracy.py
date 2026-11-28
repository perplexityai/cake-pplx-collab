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
over changing source counts on one workspace (padded and unpadded staging; also the shared weight gradients).  The
512-expert top-10 group (``router_weight_gradient="direct"``, ``shared_expert="scaled"``) on SM100a, SM103a and SM107a,
on inputs drawn on the CPU: the shared scale's dot helper, the scaled forward epilogue (``s`` in {0, 1, U}) and the
adapter's step at EP8 ``(8, 64)`` emulated on one GPU (``y``, ``dx``, the router-weight gradients and those at zero
weights alone, ``d_shared_scale`` and the shared weight gradients).

An FP32 weight-gradient accumulator is judged by its increment, ``accumulator - start`` in float64 (exact for these
values), against the oracle's weight gradient: the same error as the accumulator's against ``start`` plus the exact
gradient.  E4M3 values the MXFP8 kernels store (the saved gate, up and activation of the resident macrobatch) are judged
as their dequantized values against the oracle's value before quantization; ``x``'s E4M3 codes and UE8M0 scales are
exact against the recipe.  The kernel-shaped alternatives of :data:`tests.experimental._mok_naive.ALTERNATIVE` are
measured on the same draws and reported (``ALT`` lines), never judged.  The known accuracy gaps
(:mod:`tests.experimental._mok_gaps`) are judged in items of their own, under a strict expected-failure marker on the
architectures where they miss the tier: the MXFP8 score gradient of every MXFP8 case, and routed weight gradients of
the two BF16 cases whose routed experts cross a macrobatch boundary (:data:`STRADDLE_GAPS`).

The fused MXFP8 training cases also run with the requantized programs (``mxfp8_transposed="requantized"``, sm_100a and
sm_103a): the outputs they change are judged, the routed weight gradients and the saved activation against the oracle
and naive implementation with the same token-direction copies, and ``x``'s requantized codes and scales exactly;
every other output is the default programs' bytes (``test_mok_mxfp8.py``), so its judgment is the default case's.
The checkpointed MXFP8 step on ``blocks="32x32"`` routed weights (:data:`MXFP8_32X32_CASES`: hidden 6144 with 32 local
experts, plain; hidden 4096 with 36 local experts, clamped) is judged on every output but the score gradient, and with
the requantized programs on the outputs they change.

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
    one_rank_ids,
    one_rank_multiplier,
)
from tests.experimental._mok_reference import (
    mxfp8_dequantize,
    mxfp8_quantize,
    production_inputs,
    rand,
    randn,
    randn_like,
    require_gpu,
    single_rank_schedule,
)

kernels = cake_package("cake_mok_bf16", "_kernels")
api = cake_package("cake_mok_bf16", "api")
backend = cake_package("cake_mok_bf16", "backend")
jit = cake_package("cake_mok_bf16", "jit")
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
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, intermediate, hidden, **options) / hidden**0.5,
        randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    return shared, routed


def _scores(tokens, topk):
    scores = rand(tokens, topk, device="cuda") + 0.1
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
    initial = [randn(w.shape, device="cuda") * 0.01 for w in (*shared, *routed)]
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
        x = randn(tokens, hidden, **options) * 0.125
        dy = randn_like(x) * 0.125
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
    requantized=False,
    blocks=None,
    checkpointed=False,
):
    """The fused MXFP8 training case.  The routed weights are the test reference's recipe (``mxfp8_quantize``), or with
    ``blocks`` those of ``quantize_mok_mxfp8_weights(..., blocks=blocks)``; the oracle and the naive implementation
    read the stored pairs either way.  ``requantized`` runs the requantized programs and judges
    :data:`REQUANTIZED_OUTPUTS` only.  ``checkpointed``: the backward reads the context of the context-only recompute,
    which equals the forward's bit for bit (the resident macrobatch's rows), and the first draw's step runs twice with
    the same bytes."""
    hidden, intermediate, macro = geometry
    clamped = swiglu_limit is not None
    forward = kernels.MoKForwardMXFP8(clamped, requantized)
    backward = kernels.MoKBackwardMXFP8(clamped, fp32_wgrad, requantized)
    recompute = None
    if checkpointed:
        recompute = kernels.MoKRecomputeMXFP8(clamped, requantized)
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
        x = randn(tokens, hidden, **options)
        dy = randn(tokens, hidden, **options) * 0.125
        shared, routed = _weights(experts, hidden, intermediate, options)
        scores = _scores(tokens, topk)
        if blocks is None:
            q = [mxfp8_quantize(w) for w in routed]
            fwd_w = [(qq[0], qq[1]) for qq in q]
            bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
        else:
            fwd_w, bwd_w = api.quantize_mok_mxfp8_weights(*routed, blocks=blocks)
        combine = torch.zeros(tokens * topk, hidden, **options)
        dx_peer = torch.zeros(tokens * topk, hidden, **options)
        ds_peer = torch.zeros_like(scores)
        initial = accumulators = None
        if fp32_wgrad:
            initial, accumulators = _accumulators(shared, routed)
        r = resident

        def step():
            values = forward(
                x, [x.data_ptr()], combine, [combine.data_ptr()],
                shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
                *schedule, topk, swiglu_limit, comm, macro, mini,
            )  # fmt: skip
            torch.cuda.synchronize()
            (x_t, x_sc_t), _, (gate_q, gate_sc), _, (up_q, up_sc), _, (h_t, h_sc_t), y_shared, _ = values
            saved = dict(
                gate_saved=naive.dequantized((gate_q[:r], gate_sc[: r // 128])),
                up_saved=naive.dequantized((up_q[:r], up_sc[: r // 128])),
                hidden_saved=naive.dequantized((h_t[:, :r], h_sc_t[:, : r // 128]), transposed=True),
            )
            # Copies: the backward reuses the context ring for the later macrobatches, and a slice that spans the whole
            # ring is a view of it.
            x_codes = x_t[:, :r].clone().view(torch.uint8)
            x_scales = x_sc_t[:, : r // 128].clone().view(torch.uint8)
            context = values[:7]
            if checkpointed:
                kept = [piece.clone() for piece in _resident_context(values, r)]
                context = recompute(
                    x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
                    *schedule, topk, swiglu_limit, comm, macro, mini,
                )[:7]  # fmt: skip
                torch.cuda.synchronize()
                rebuilt = _resident_context(context, r)
                assert all(torch.equal(a, b) for a, b in zip(kept, rebuilt, strict=True))
            result = backward(
                dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
                scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
                shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
                *context, x, [x.data_ptr()], *schedule,
                topk, swiglu_limit, comm, macro, mini,
                weight_grad_accumulators=accumulators,
            )  # fmt: skip
            torch.cuda.synchronize()
            return dict(
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

        outputs = step()
        if checkpointed and j == 0:
            first = {name: value.clone() for name, value in outputs.items()}
            for buffer in (combine, dx_peer, ds_peer):
                buffer.zero_()
            for accumulator, start in zip(accumulators or (), initial or (), strict=True):
                accumulator.copy_(start)
            again = step()
            assert all(torch.equal(first[name], again[name]) for name in first)
            outputs = first
        if requantized:
            outputs = {name: outputs[name] for name in REQUANTIZED_OUTPUTS}
        # The recipe's x codes and scales of the resident macrobatch (schedule order, padding rows zero); requantized:
        # the token-direction copy of the forward's copy dequantized.
        valid = slab_routes >= 0
        source = torch.where(valid[:, None], x[slab_routes.clamp(min=0).long() // topk], torch.zeros_like(x[:1]))
        if requantized:
            xq, xs = mxfp8_quantize(source, transposed=False)[:2]
            _, _, xq_t, xs_t = mxfp8_quantize(mxfp8_dequantize(xq, xs), normal=False)
        else:
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
                mx=mx, macro=None if fp32_wgrad else macro, requantized=requantized,
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
                    hidden_saved=naive.qdq(
                        naive.qdq(hid) if requantized else hid, transposed=True
                    ),
                )
            refs[dtype] = (rr, v)
        oracle = refs[naive.ORACLE][1]
        r_naive, naive_values[j] = refs[naive.NAIVE]
        if not fp32_wgrad:
            for i, name in enumerate(("dw_routed_gate", "dw_routed_up", "dw_routed_down")):
                alt.add(name, j, outputs[name], oracle[name], {"bf16": r_naive["alt_dw"][i]})
        if not requantized:
            alt.add(
                "d_scores",
                j,
                outputs["d_scores"],
                oracle["d_scores"],
                {"bf16": r_naive["alt_d_scores"].view(tokens, topk)},
            )
        return outputs, oracle

    judge, failures = judge_case(
        run,
        lambda j, tier: naive_values.pop(j),
        seed=seed + hidden + intermediate + macro,
        exact=frozenset({"x_codes", "x_scales"}),
        reported=(alt,),
    )
    return judge, alt, failures


def _resident_context(context, rows):
    """The saved-context pieces of the resident macrobatch's ``rows`` (the forward's and the recompute's alike): the
    E4M3 codes and scales of ``x_t`` and ``h_t`` (``[features, tokens]``) and of the gate and up values, and the
    context's other three entries."""
    pieces = []
    for index, transposed in ((0, True), (2, False), (4, False), (6, True)):
        data, scales = context[index]
        if transposed:
            pieces += [data[:, :rows], scales[:, : rows // 128]]
        else:
            pieces += [data[:rows], scales[: rows // 128]]
    return pieces + [context[index] for index in (1, 3, 5)]


#: The known-gap outputs of every MXFP8 case.
MXFP8_GAPS = ("d_scores",)
#: The outputs of an MXFP8 case judged with the requantized programs: those they change.
REQUANTIZED_OUTPUTS = (
    "dw_routed_gate",
    "dw_routed_up",
    "dw_routed_down",
    "hidden_saved",
    "x_codes",
    "x_scales",
)


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


def _mxfp8_requantized(record_property, **kwargs):
    """Judge :data:`REQUANTIZED_OUTPUTS` of an MXFP8 case with the requantized programs; skip on a target without
    them."""
    require_gpu()
    arch = jit.target_arch()
    if arch not in kernels.MXFP8_REQUANTIZED_TARGETS:
        pytest.skip(f"The requantized MXFP8 programs have no {arch} build")
    run = gaps.judged(
        ("mxfp8-requantized", *sorted(kwargs.items())),
        lambda: _mxfp8_case(**kwargs, requantized=True),
    )
    _check(record_property, gaps.OUTPUTS, run)


@pytest.mark.parametrize(
    "geometry,swiglu_limit",
    [values for _, values in MXFP8_CASES],
    ids=[case for case, _ in MXFP8_CASES],
)
def test_mxfp8_requantized_training_accuracy(record_property, swiglu_limit, geometry):
    _mxfp8_requantized(record_property, swiglu_limit=swiglu_limit, geometry=geometry)


@pytest.mark.parametrize(
    "geometry,swiglu_limit",
    [values for _, values in MXFP8_CASES],
    ids=[case for case, _ in MXFP8_CASES],
)
def test_mxfp8_requantized_training_fp32_wgrad_accuracy(
    record_property, swiglu_limit, geometry
):
    _mxfp8_requantized(
        record_property, swiglu_limit=swiglu_limit, geometry=geometry, fp32_wgrad=True
    )


@pytest.mark.parametrize("experts", [36, 9], ids=["local36", "local9"])
def test_mxfp8_requantized_training_local_experts_accuracy(record_property, experts):
    """288 routed experts over 8 or 32 ranks: 36 or 9 local experts, top-8, clamped."""
    _mxfp8_requantized(
        record_property,
        swiglu_limit=0.75,
        geometry=(512, 256, 2048),
        experts=experts,
        topk=8,
    )


#: ``(id, keywords)`` of the MXFP8 cases on ``blocks="32x32"`` routed weights at the requested model shapes: hidden
#: 6144 with 256 routed experts (32 local experts: the (8, 32) layout) and plain SwiGLU; hidden 4096 with 288 routed
#: experts (36 local experts: the (8, 36) layout) and clamped SwiGLU.  Intermediate 2048, top-8, FP32 weight
#: gradients; 2048-row macrobatches (four and five per step, so the backward replays the later ones).
MXFP8_32X32_CASES = (
    ("h6144-i2048-local32-plain", dict(swiglu_limit=None, geometry=(6144, 2048, 2048), experts=32)),
    ("h4096-i2048-local36-clamped", dict(swiglu_limit=0.75, geometry=(4096, 2048, 2048), experts=36)),
)


@pytest.mark.parametrize("case", [c for _, c in MXFP8_32X32_CASES], ids=[i for i, _ in MXFP8_32X32_CASES])
def test_mxfp8_32x32_weights_accuracy(record_property, case):
    """The fused MXFP8 training step on ``quantize_mok_mxfp8_weights(..., blocks="32x32")`` routed weights, the BF16
    shared expert and FP32 weight-gradient accumulators, checkpointed: forward, the context-only recompute (equal to
    the forward's context bit for bit) and the backward from the recomputed context, which replays the later
    macrobatches; the first draw's step repeats bit for bit.  Every output is judged against the oracle on the same
    stored weights, beside the naive implementation, as in the main items of the other MXFP8 cases (their score
    gradient, a known gap of every single-GPU MXFP8 case, is judged in items of its own there and not here)."""
    require_gpu()
    kwargs = dict(case, topk=8, fp32_wgrad=True, blocks="32x32", checkpointed=True)
    run = gaps.judged(("mxfp8", *sorted(kwargs.items())), lambda: _mxfp8_case(**kwargs))
    _check(record_property, gaps.OUTPUTS, run, MXFP8_GAPS)


@pytest.mark.parametrize("case", [c for _, c in MXFP8_32X32_CASES], ids=[i for i, _ in MXFP8_32X32_CASES])
def test_mxfp8_requantized_32x32_weights_accuracy(record_property, case):
    """Both options together: the requantized programs on ``blocks="32x32"`` routed weights in the checkpointed
    training step of :func:`test_mxfp8_32x32_weights_accuracy` (same cases, FP32 weight gradients, the backward
    replaying the later macrobatches).  :data:`REQUANTIZED_OUTPUTS` are judged as in :func:`_mxfp8_requantized`; every
    other output is the default programs' bytes on the same weights (``test_mok_mxfp8.py``), judged by
    :func:`test_mxfp8_32x32_weights_accuracy`."""
    extra = {"topk": 8, "fp32_wgrad": True, "blocks": "32x32", "checkpointed": True}
    _mxfp8_requantized(record_property, **case, **extra)


@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("tokens,hidden", [(512, 256), (37, 6144), (1, 1032)])
def test_epilogue_forward_accuracy(record_property, topk, tokens, hidden):
    """``y = shared + sum_k s_k routed_k``: FP32 accumulation from the shared row, then ascending slots."""
    require_gpu()
    epilogues = kernels.MoKEpilogues(topk)
    naive_values = {}

    def run(j, _seed):
        shared = randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
        routed = randn(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16) * 0.125
        scores = rand(tokens, topk, device="cuda")
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
    scores = rand(tokens, topk, device="cuda") * 1.5
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
            x = randn(tokens, hidden, **options) * 0.125
            dy = randn_like(x) * 0.125
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
        x = randn(tokens, hidden, **options) * 0.125
        dy = randn_like(x) * 0.125
        shared, routed = _weights(experts, hidden, intermediate, options)
        scores = _before_down_weights(tokens, topk, False)
        ids = rand(tokens, experts, device=device).argsort(-1)[:, :topk].contiguous()
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
            x = randn(count, hidden, **options) * 0.125
            dy = randn_like(x) * 0.125
            scores = _before_down_weights(count, topk, False)
            ids = rand(count, experts, device=device).argsort(-1)[:, :topk].contiguous()
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


#: The 512-expert top-10 group of the explicit modes and the kernels its cases launch.
GROUP = kernels.MODE_GROUPS["scaled_top10"]
GROUP_ROLES = (
    "backward_fp32_direct",
    "epilogue_forward_scaled_10",
    "shared_scale_rows",
    "shared_scale_dot",
)


def _require_group():
    """Skip on a target without the 512-expert top-10 group's kernels (``MODE_GROUPS["scaled_top10"].targets``: SM100a,
    SM103a and SM107a) or while they are not in this package's registry (``test_mok_bf16.py`` fails for that)."""
    require_gpu()
    arch = gaps.TARGETS[torch.cuda.get_device_capability()]
    if arch not in GROUP.targets:
        pytest.skip(f"The 512-expert top-10 group has no {arch} build")
    registry = jit._registry()
    if any(role not in registry for role in GROUP_ROLES):
        pytest.skip(
            "The 512-expert top-10 group's kernels are not in this package's registry"
        )


def _cpu_scales(tokens):
    """FP32 shared-expert scales drawn on the CPU: ``U[0, 2)``, every 4th token's exactly 0 and the next one's exactly
    1."""
    scale = rand(tokens) * 2.0
    scale[::4] = 0.0
    scale[1::4] = 1.0
    return scale


@pytest.mark.parametrize(
    "tokens,width", [(512, 1024), (37, 4096)], ids=["t512-w1024", "t37-w4096"]
)
def test_shared_scale_dot_accuracy(record_property, tokens, width):
    """The shared scale's gradient helper ``dot(a, b)`` (FP32 row dots in one fixed order; ``dot(dh_shared,
    hidden_shared)`` at width I) against the float64 dot beside the naive BF16 ``(a * b).sum(-1)``; inputs drawn on
    the CPU."""
    _require_group()
    helpers = kernels.MoKSharedScale()
    naive_values = {}

    def run(j, _seed):
        a = (randn(tokens, width) * 0.125).bfloat16().cuda()
        b = (randn(tokens, width) * 0.125).bfloat16().cuda()
        with kernels._ffi_stream():
            dots = helpers.dot(a, b)
        naive_values[j] = dict(d_shared_scale=(a * b).sum(-1))
        return dict(d_shared_scale=dots), dict(
            d_shared_scale=(a.double() * b.double()).sum(-1)
        )

    judge, failures = judge_case(
        run, lambda j, tier: naive_values.pop(j), seed=6300 + tokens + width
    )
    judge.record(record_property)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("tokens,hidden", [(512, 256), (37, 4096), (1, 1032)])
def test_epilogue_forward_scaled_accuracy(record_property, tokens, hidden):
    """``y = s * shared + sum_k w_k routed_k`` at top-10 (the scaled forward epilogue: FP32 accumulation from
    ``s * shared``, then ascending slots), ``s`` in {0, 1, U[0, 2)} per token and every 5th token's first weight 0;
    inputs drawn on the CPU."""
    _require_group()
    topk = GROUP.topk
    epilogues = kernels.MoKEpilogues(topk, shared_row="scaled")
    naive_values = {}

    def run(j, _seed):
        shared = (randn(tokens, hidden) * 0.125).bfloat16().cuda()
        routed = (randn(tokens * topk, hidden) * 0.125).bfloat16().cuda()
        scores = rand(tokens, topk) * 1.5
        scores[::5, 0] = 0.0
        scores, scale = scores.cuda(), _cpu_scales(tokens).cuda()
        y = epilogues.forward(shared, routed, scores, scale)
        naive_values[j] = dict(
            y=naive.epilogue_forward(
                shared, routed, scores, naive.NAIVE, shared_scale=scale
            )
        )
        return dict(y=y), dict(
            y=naive.epilogue_forward(
                shared, routed, scores, naive.ORACLE, shared_scale=scale
            )
        )

    judge, failures = judge_case(
        run, lambda j, tier: naive_values.pop(j), seed=6200 + tokens + hidden
    )
    judge.record(record_property)
    assert not failures, "\n".join(failures)


def _scaled_group_adapter_case(
    *,
    tokens=512,
    hidden=512,
    intermediate=256,
    macro=1536,
    mini=256,
    seed=8431,
    layout=None,
):
    """The group's adapter step (``build_schedule``, ``forward``, ``backward``; ``router_weight_gradient="direct"``,
    ``shared_expert="scaled"``, plain SwiGLU, FP32 accumulators, several macrobatches) at EP8 ``(8, 64)`` (or at the
    group's ``layout``), top-10, as rank 0 of its EP group emulated on one GPU (:mod:`tests.experimental._mok_one_gpu`;
    every route on this rank), on inputs drawn on the CPU: ``s`` in {0, 1, U[0, 2)} per token, every 5th token's first
    router weight and every weight of token 3 exactly 0.  Judged per draw against the oracle beside the naive
    implementation of the single-rank layer (:func:`tests.experimental._mok_naive.layer` with ``shared_scale`` and
    ``direct=True``): ``y``, ``dx``, the router-weight gradients (also alone at the zero weights,
    ``d_scores_zero_weight``), ``d_shared_scale`` and the shared weight gradients (this rank's own).  The routed weight
    gradients sum EP copies of every route and are checked exactly in ``test_mok_bf16.py``."""
    ep, experts = GROUP.layouts[0] if layout is None else layout
    topk = GROUP.topk
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep,
        experts,
        topk,
        fp32_wgrad=True,
        router_weight_gradient="direct",
        shared_expert="scaled",
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
    naive_values = {}

    def run(j, _draw):
        x = (randn(tokens, hidden) * 0.125).bfloat16()
        dy = (randn(tokens, hidden) * 0.125).bfloat16()
        shared = [
            (randn(intermediate, hidden) / hidden**0.5).bfloat16(),
            (randn(intermediate, hidden) / hidden**0.5).bfloat16(),
            (randn(hidden, intermediate) / intermediate**0.5).bfloat16(),
        ]
        routed = [
            (randn(experts, intermediate, hidden) / hidden**0.5).bfloat16(),
            (randn(experts, intermediate, hidden) / hidden**0.5).bfloat16(),
            (randn(experts, hidden, intermediate) / intermediate**0.5).bfloat16(),
        ]
        scores = rand(tokens, topk) * 1.5
        scores[::5, 0] = 0.0
        scores[3] = 0.0
        scale = _cpu_scales(tokens)
        ids = rand(tokens, experts).argsort(-1)[:, :topk].contiguous()
        x, dy, scores, scale, ids = (
            value.to(device) for value in (x, dy, scores, scale, ids)
        )
        shared, routed = [w.to(device) for w in shared], [w.to(device) for w in routed]
        weights = (*shared, *routed)
        accumulators = [torch.zeros(w.shape, device=device) for w in weights]
        schedule = functional.build_schedule(
            workspace, config, ids, num_local_experts=experts
        )
        assert (
            int(schedule.num_tokens.item()) > macro
        )  # several macrobatches: backward replays the later ones
        y, context = functional.forward(
            config, workspace, schedule, x, scores, *weights, shared_scale=scale
        )
        gradients = functional.backward(
            config, workspace, schedule, context, dy, x, scores, *weights, weight_grad_accumulators=accumulators,
            shared_scale=scale,
        )  # fmt: skip
        torch.cuda.synchronize()
        zero = scores == 0
        outputs = dict(
            y=y.clone(), dx=gradients[0].clone(), d_scores=gradients[1].clone()
        )
        outputs["d_scores_zero_weight"] = gradients[1][zero].clone()
        outputs["d_shared_scale"] = gradients[8].clone()
        outputs.update((name, gradients[index].clone()) for name, _, index in SHARED_DW)
        segments = _segments(single_rank_schedule(ids, experts))
        values = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            r = naive.layer(
                x,
                dy,
                scores,
                shared,
                routed,
                segments,
                None,
                dtype,
                shared_scale=scale,
                direct=True,
            )
            reference = dict(
                y=r["y"],
                dx=r["dx"],
                d_scores=r["d_scores"],
                d_shared_scale=r["d_shared_scale"],
            )
            reference["d_scores_zero_weight"] = r["d_scores"][zero]
            reference.update((name, r["dw"][index][0]) for name, index, _ in SHARED_DW)
            values[dtype] = reference
        naive_values[j] = values[naive.NAIVE]
        return outputs, values[naive.ORACLE]

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed)
    return judge, TierJudge(), failures


def test_scaled_group_adapter_accuracy(record_property):
    """The adapter's step of the 512-expert top-10 group (``router_weight_gradient="direct"``,
    ``shared_expert="scaled"``) at EP8 ``(8, 64)``, emulated on one GPU as rank 0 of the group with every route on this
    rank: ``y``, ``dx``, the router-weight gradients (and those at zero weights alone), ``d_shared_scale`` and the
    shared weight gradients."""
    _require_group()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(("scaled_group_adapter",), _scaled_group_adapter_case),
    )


#: The 896-expert top-16 group (SiTU-GLU routed experts, no shared expert) and the kernels its cases launch.
SITU_GROUP = kernels.MODE_GROUPS["situ_top16"]
SITU_ROLES = (
    "forward_situ_noshared",
    "backward_fp32_direct_situ_noshared",
    "epilogue_forward_none_16",
    "epilogue_backward_none_16",
)
#: The SiTU-GLU scalars of the cases.
SITU_BETA, SITU_LB = 4.0, 25.0
#: The SiTU-GLU data kinds: per kind the routed gate and up weights' ``(spread, offset range)``.  ``x`` is ``randn *
#: 0.125`` with each token's first feature 1, the weights ``randn * spread / 0.125 / sqrt(hidden)`` with their first
#: column (the one that feature multiplies) 0 or uniform over the offset range: gate and up values of spread about
#: ``spread`` around the offset.  ``saturated`` reaches ``|gate|`` about 190 and ``|up|`` about 950;
#: ``deep_negative`` puts every gate in about [-88.5, -80.5], where ``sigmoid(gate)`` is a normal FP32 value or just
#: below the normal range: gates further below (the FP32 subnormal range of ``sigmoid``, ``gate < ln(FLT_MIN) =
#: -87.3``) are outside the accuracy requirements of these kernels.  ``zero_gate`` and ``zero_up`` set a random half
#: of the gate or up weight rows exactly 0, so those gate or up values are exactly 0 (and so are their hidden values).
SITU_DATA = {
    "normal": ((2.0, None), (2.0, None)),
    "wide": ((6.0, None), (40.0, None)),
    "saturated": ((1.0, (-190.0, 190.0)), (5.0, (-950.0, 950.0))),
    "deep_negative": ((0.5, (-87.0, -82.0)), (2.0, None)),
    "small": ((0.1, None), (0.1, None)),
    "zero_gate": ((2.0, None), (2.0, None)),
    "zero_up": ((2.0, None), (2.0, None)),
}
#: The tiny SiTU-GLU data kinds: per kind the fraction of tokens whose gate values are tiny and the fraction whose up
#: values are tiny.  ``x`` is ``randn * 0.125`` with a gate half and an up half of the features: the routed gate
#: weights read only the gate half and the up weights only the up half (``randn * spread / 0.125 / sqrt(hidden / 2)``),
#: so a factor on one half of a token's features scales that token's gate or up values alone.  A tiny gate is about
#: ``beta 2^U`` and a tiny up about ``lb 2^U``, with ``U`` uniform on [-34, -14) per token (independent for gates and
#: ups): ``|gate| / beta`` and ``|up| / lb`` from about ``2^-14`` down to far below ``2^-24``, the spacing of the FP32
#: values just below 1.  Every other gate and up value has spread 2.  ``tiny_gate``: every token's gates; ``tiny_up``:
#: every token's ups; ``tiny_both``: both; ``mixed``: a random quarter of the tokens' gates and an independent
#: quarter's ups.
SITU_TINY = {
    "tiny_gate": (1.0, 0.0),
    "tiny_up": (0.0, 1.0),
    "tiny_both": (1.0, 1.0),
    "mixed": (0.25, 0.25),
}
#: ``U`` of a tiny value: uniform on ``[SITU_TINY_LOG2[0], SITU_TINY_LOG2[1])``.
SITU_TINY_LOG2 = (-34.0, -14.0)


def _require_situ_group():
    """Skip on a target without the 896-expert top-16 group's kernels (``MODE_GROUPS["situ_top16"].targets``: SM100a,
    SM103a and SM107a) or while they are not in this package's registry (``test_mok_bf16.py`` fails for that)."""
    require_gpu()
    arch = gaps.TARGETS[torch.cuda.get_device_capability()]
    if arch not in SITU_GROUP.targets:
        pytest.skip(f"The 896-expert top-16 group has no {arch} build")
    registry = jit._registry()
    if any(role not in registry for role in SITU_ROLES):
        pytest.skip("The 896-expert top-16 group's kernels are not in this package's registry")


def _situ_weights(kind, experts, hidden, intermediate):
    """BF16 routed ``(gate, up, down)`` weights of the SiTU-GLU data kind ``kind`` (:data:`SITU_DATA`), drawn on the
    CPU."""

    def projection(spec):
        spread, offset = spec
        w = randn(experts, intermediate, hidden) * (spread / 0.125 / hidden**0.5)
        w[:, :, 0] = 0.0
        if offset is not None:
            w[:, :, 0] = offset[0] + (offset[1] - offset[0]) * rand(experts, intermediate)
        return w

    gate_spec, up_spec = SITU_DATA[kind]
    gate, up = projection(gate_spec), projection(up_spec)
    if kind in ("zero_gate", "zero_up"):
        (gate if kind == "zero_gate" else up)[rand(experts, intermediate) < 0.5] = 0.0
    down = randn(experts, hidden, intermediate) / intermediate**0.5
    return [w.bfloat16() for w in (gate, up, down)]


def _situ_tiny_inputs(kind, tokens, hidden, experts, intermediate, spread=2.0):
    """BF16 ``x`` and routed ``(gate, up, down)`` weights of the tiny SiTU-GLU data kind ``kind``
    (:data:`SITU_TINY`), drawn on the CPU."""
    low, high = SITU_TINY_LOG2

    def tiny(scale):
        return (scale * torch.exp2(low + (high - low) * rand(tokens, dtype=torch.float64))).float()

    def factor(fraction, scale):
        """Per token: the factor on its gate or up half, and the spread of the weights reading that half."""
        if fraction == 1.0:
            return tiny(scale), 1.0
        if fraction == 0.0:
            return torch.ones(tokens), spread
        return torch.where(rand(tokens) < fraction, tiny(scale) / spread, torch.ones(tokens)), spread

    gate_fraction, up_fraction = SITU_TINY[kind]
    half = hidden // 2
    gate_factor, gate_spread = factor(gate_fraction, SITU_BETA)
    up_factor, up_spread = factor(up_fraction, SITU_LB)
    x = randn(tokens, hidden) * 0.125
    x[:, :half] *= gate_factor[:, None]
    x[:, half:] *= up_factor[:, None]
    gate = torch.zeros(experts, intermediate, hidden)
    up = torch.zeros(experts, intermediate, hidden)
    gate[:, :, :half] = randn(experts, intermediate, half) * (gate_spread / 0.125 / half**0.5)
    up[:, :, half:] = randn(experts, intermediate, hidden - half) * (up_spread / 0.125 / (hidden - half) ** 0.5)
    down = randn(experts, hidden, intermediate) / intermediate**0.5
    return x.bfloat16(), [w.bfloat16() for w in (gate, up, down)]


@pytest.mark.parametrize("tokens,hidden", [(512, 256), (37, 4096), (1, 1032)])
def test_epilogue_forward_none_accuracy(record_property, tokens, hidden):
    """``y = sum_k w_k routed_k`` at top-16 (the forward epilogue without a shared row: FP32 accumulation from zero in
    ascending slots, one BF16 rounding), every 5th token's first weight 0; inputs drawn on the CPU."""
    _require_situ_group()
    topk = SITU_GROUP.topk
    epilogues = kernels.MoKEpilogues(topk, shared_row="none")
    naive_values = {}

    def run(j, _seed):
        routed = (randn(tokens * topk, hidden) * 0.125).bfloat16().cuda()
        scores = rand(tokens, topk) * 1.5
        scores[::5, 0] = 0.0
        scores = scores.cuda()
        zeros = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="cuda")
        y = epilogues.forward(None, routed, scores)
        naive_values[j] = dict(y=naive.epilogue_forward(zeros, routed, scores, naive.NAIVE))
        return dict(y=y), dict(y=naive.epilogue_forward(zeros, routed, scores, naive.ORACLE))

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=6400 + tokens + hidden)
    judge.record(record_property)
    assert not failures, "\n".join(failures)


def _situ_group_adapter_case(
    kind,
    *,
    tokens=256,
    hidden=512,
    intermediate=256,
    macro=8192,
    mini=256,
    seed=8531,
    layout=None,
):
    """The 896-expert top-16 group's adapter step (``build_schedule``, ``forward``, ``backward``;
    ``activation="situ_glu"`` with ``situ_beta`` 4 and ``situ_lb`` 25, ``shared_expert="none"``,
    ``router_weight_gradient="direct"``, FP32 accumulators, several macrobatches) at EP64 ``(64, 14)`` (or at the
    group's ``layout``) as rank 0 of its EP group emulated on one GPU (:mod:`tests.experimental._mok_one_gpu`; every
    route on this rank; at ``(64, 14)`` two of each token's 16 routes repeat an expert, :func:`one_rank_ids`), on the
    SiTU-GLU data kind ``kind`` (:data:`SITU_DATA` or :data:`SITU_TINY`) drawn on the CPU, every 5th token's first
    router weight and every weight of token 3 exactly 0.  Judged per draw against the oracle beside the naive
    implementation of the single-rank layer without a shared expert (:func:`tests.experimental._mok_naive.layer` with
    ``shared_w=None``, ``activation=("situ_glu", 4, 25)`` and ``direct=True``): ``y``, ``dx``, the router-weight
    gradients (also alone at the zero weights, ``d_scores_zero_weight``) and the three routed weight gradients (each the
    sum of EP equal copies of every route in the emulation, judged divided by EP, a power of two)."""
    (ep, experts), topk = (
        SITU_GROUP.layouts[0] if layout is None else layout,
        SITU_GROUP.topk,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    functional = backend.MoKFunctional(
        ep, experts, topk, fp32_wgrad=True, router_weight_gradient="direct", shared_expert="none",
        activation="situ_glu",
    )  # fmt: skip
    functional.communication = OneGpuGroup()
    multiplier = one_rank_multiplier(tokens, topk, ep, experts)
    capacity = workspace_module.schedule_capacity_for(tokens, topk, ep, multiplier)
    workspace = one_gpu_workspace(ep, tokens, hidden, topk, capacity, device)
    config = workspace_module.MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=mini,
        macrobatch_size=macro,
        schedule_capacity_multiplier=multiplier,
    )
    situ = dict(situ_beta=SITU_BETA, situ_lb=SITU_LB)
    names = ("dw_routed_gate", "dw_routed_up", "dw_routed_down")
    naive_values = {}

    def run(j, _draw):
        if kind in SITU_TINY:
            dy = (randn(tokens, hidden) * 0.125).bfloat16()
            x, routed = _situ_tiny_inputs(kind, tokens, hidden, experts, intermediate)
        else:
            x = randn(tokens, hidden) * 0.125
            x[:, 0] = 1.0
            x = x.bfloat16()
            dy = (randn(tokens, hidden) * 0.125).bfloat16()
            routed = _situ_weights(kind, experts, hidden, intermediate)
        scores = rand(tokens, topk) + 0.1
        scores[::5, 0] = 0.0
        scores[3] = 0.0
        ids = one_rank_ids(tokens, experts, topk)
        x, dy, scores, ids = (value.to(device) for value in (x, dy, scores, ids))
        routed = [w.to(device) for w in routed]
        accumulators = [torch.zeros(w.shape, device=device) for w in routed]
        schedule = functional.build_schedule(workspace, config, ids, num_local_experts=experts)
        assert int(schedule.num_tokens.item()) > macro  # several macrobatches: backward replays the later ones
        y, context = functional.forward(config, workspace, schedule, x, scores, None, None, None, *routed, **situ)
        gradients = functional.backward(
            config, workspace, schedule, context, dy, x, scores, None, None, None, *routed,
            weight_grad_accumulators=accumulators, **situ,
        )  # fmt: skip
        torch.cuda.synchronize()
        zero = scores == 0
        outputs = dict(y=y.clone(), dx=gradients[0].clone(), d_scores=gradients[1].clone())
        outputs["d_scores_zero_weight"] = gradients[1][zero].clone()
        outputs.update((name, gradients[2 + i].double() / ep) for i, name in enumerate(names))
        segments = _segments(single_rank_schedule(ids, experts))
        values = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            r = naive.layer(
                x, dy, scores, None, routed, segments, None, dtype, activation=("situ_glu", SITU_BETA, SITU_LB),
                direct=True,
            )  # fmt: skip
            reference = dict(y=r["y"], dx=r["dx"], d_scores=r["d_scores"])
            reference["d_scores_zero_weight"] = r["d_scores"][zero]
            reference.update((name, r["dw"][i]) for i, name in enumerate(names))
            values[dtype] = reference
        naive_values[j] = values[naive.NAIVE]
        return outputs, values[naive.ORACLE]

    judge, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed)
    return judge, TierJudge(), failures


@pytest.mark.parametrize("kind", list(SITU_DATA) + list(SITU_TINY))
def test_situ_group_adapter_accuracy(record_property, kind):
    """The adapter's step of the 896-expert top-16 group (SiTU-GLU, ``shared_expert="none"``,
    ``router_weight_gradient="direct"``) at EP64 ``(64, 14)``, emulated on one GPU as rank 0 of the group with every
    route on this rank, on each SiTU-GLU data kind, the tiny ones included: ``y``, ``dx``, the router-weight gradients
    (and those at zero weights alone) and the three routed weight gradients."""
    _require_situ_group()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(("situ_group_adapter", kind), lambda: _situ_group_adapter_case(kind)),
    )


#: The later layouts of the two groups (their first layout's cases above): the 512-expert group's, then the
#: 896-expert group's with each SiTU-GLU data kind.
SCALED_LAYOUTS_ADDED = list(GROUP.layouts[1:])
SITU_LAYOUT_CASES = [
    pytest.param(layout, kind, id=f"ep{layout[0]}-local{layout[1]}-{kind}")
    for layout in SITU_GROUP.layouts[1:]
    for kind in (*SITU_DATA, *SITU_TINY)
]


@pytest.mark.parametrize(
    "layout",
    SCALED_LAYOUTS_ADDED,
    ids=[f"ep{ep}-local{e}" for ep, e in SCALED_LAYOUTS_ADDED],
)
def test_scaled_group_layout_adapter_accuracy(record_property, layout):
    """:func:`test_scaled_group_adapter_accuracy` at a later layout of the 512-expert top-10 group, emulated on one GPU
    as rank 0 of its EP group with every route on this rank."""
    _require_group()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(
            ("scaled_group_adapter", layout),
            lambda: _scaled_group_adapter_case(layout=layout),
        ),
    )


@pytest.mark.parametrize("layout,kind", SITU_LAYOUT_CASES)
def test_situ_group_layout_adapter_accuracy(record_property, layout, kind):
    """:func:`test_situ_group_adapter_accuracy` at a later layout of the 896-expert top-16 group (every route on this
    rank's experts, distinct experts per token), emulated on one GPU as rank 0 of its EP group, on each SiTU-GLU data
    kind."""
    _require_situ_group()
    _check(
        record_property,
        gaps.OUTPUTS,
        gaps.judged(
            ("situ_group_adapter", layout, kind),
            lambda: _situ_group_adapter_case(kind, layout=layout),
        ),
    )
