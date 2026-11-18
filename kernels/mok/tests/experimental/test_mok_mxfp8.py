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

Independent single-GPU fixtures for the MoK kernel tests.
Single-GPU native MXFP8 MoK checks: the recipe's stored codes, contracts and determinism.

The MoK recipe in schedule order: x, the SwiGLU activation, the score-scaled dY and d_gate/d_up are MXFP8 (32-element
blocks along K), routed weights are caller-prequantized, GEMMs accumulate in FP32 on dequantized operands, and the
shared expert stays BF16.  Here: the weight quantizer writes the recipe's bytes (every chunking of its PyTorch path
the same bytes), the forward stores ``x``'s codes and scales exactly as the recipe does, every output is finite, and
reruns, recompute and the mixed forward programs repeat bit for bit.  The accuracy of every floating output (and of
the stored E4M3 gate, up and activation values) against the float64 oracle is judged in ``test_mok_accuracy.py``.

Backend suite: the package is imported only through
:func:`tests.cake.test_helpers.cake_package.cake_package`; the :mod:`cake_pplx.cake_mok_det_train`
facade is covered by ``test_cake_mok_det_train_api.py``.
"""

import pytest
import torch

from tests.experimental._mok_reference import (
    mxfp8_dequantize as dequantize,
    mxfp8_quantize as quantize,
    require_gpu,
    single_rank_schedule,
    swiglu,
)
from tests.cake.test_helpers.cake_package import cake_package

kernels = cake_package("cake_mok_bf16", "_kernels")
api = cake_package("cake_mok_bf16", "api")
MoKBackwardMXFP8 = kernels.MoKBackwardMXFP8
MoKForwardMXFP8 = kernels.MoKForwardMXFP8
MoKRecomputeMXFP8 = kernels.MoKRecomputeMXFP8
quantize_mok_mxfp8_weights = api.quantize_mok_mxfp8_weights
mxfp8 = cake_package("cake_mok_bf16", "mxfp8")

LIMITS = pytest.mark.parametrize("swiglu_limit", [None, 0.75], ids=["plain", "clamped"])


@pytest.fixture(autouse=True)
def _exact_matmul():
    # Save and restore by hand: monkeypatch's undo deletes this module attribute.
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_tf32
    matmul.allow_tf32 = False
    yield
    matmul.allow_tf32 = previous


def _setup(hidden, intermediate, seed, experts=4, tokens=512, topk=2):
    torch.manual_seed(seed)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, **options)
    dy = torch.randn(tokens, hidden, **options) * 0.125
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
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row * 3 + slot) % experts for slot in range(topk)], -1)
    return x, dy, shared, routed, scores, single_rank_schedule(routes, experts)


REPEATS = 5


def _same(a, b):
    return all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b, strict=True))


def test_weight_quantizer_matches_recipe():
    require_gpu()
    torch.manual_seed(17)
    gate = torch.randn(3, 256, 512, device="cuda", dtype=torch.bfloat16)
    gate[0, 5, 64:96] = 0  # an all-zero block takes the minimum scale
    gate[1, 7, :32] *= 1e4  # large blocks saturate to the finite range
    up = torch.randn_like(gate)
    down = torch.randn(3, 512, 256, device="cuda", dtype=torch.bfloat16)
    forward_w, backward_w = quantize_mok_mxfp8_weights(gate, up, down)
    for weight, fwd, bwd in zip((gate, up), forward_w[:2], backward_w[:2], strict=True):
        expected = quantize(weight)
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(bwd, expected, strict=True)
        )
        assert fwd[0] is bwd[0] and fwd[1] is bwd[1]
    expected_down = quantize(down)
    assert torch.equal(
        forward_w[2][0].view(torch.uint8), expected_down[0].view(torch.uint8)
    )
    assert torch.equal(forward_w[2][1], expected_down[1])
    assert torch.equal(
        backward_w[2][0].view(torch.uint8), expected_down[2].view(torch.uint8)
    )
    assert torch.equal(backward_w[2][1], expected_down[3])
    rel = (dequantize(*expected[:2]) - up.float()).norm() / up.float().norm()
    assert rel.item() < 0.05


def _flat(value):
    return [t for v in value for t in (_flat(v) if isinstance(v, tuple) else [v])]


@pytest.mark.parametrize("experts_per_chunk", [1, 2, 3])
def test_weight_quantizer_chunks_are_bitwise(monkeypatch, experts_per_chunk):
    """The quantizer's PyTorch path writes the per-matrix reference's bytes, NaN / infinity / zero blocks included,
    for one layout or both, and every chunking of a stack of matrices writes the same bytes."""
    require_gpu()
    torch.manual_seed(23)
    gate = torch.randn(5, 256, 512, device="cuda", dtype=torch.bfloat16)
    gate[2, 9, 32:64] = float("nan")
    gate[3, 1, :32] = float("inf")
    gate[3, 7, 100] = -float("inf")
    gate[4, 0, 64:96] = 0
    gate[1, 201, 160:192] = torch.tensor([448.0, 449.0, -464.0, 480.0] * 8)
    up = torch.randn_like(gate) * 1e-3
    down = torch.randn(5, 512, 256, device="cuda", dtype=torch.bfloat16) * 1e3
    for matrix in (gate[1], gate[2], down[3]):
        single = mxfp8.mxfp8_quantize(matrix)
        assert all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(single, quantize(matrix), strict=True)
        )
    both = mxfp8.mxfp8_quantize(gate)
    normal = mxfp8.mxfp8_quantize(gate, return_transposed=False)[:2]
    transposed = mxfp8.mxfp8_quantize(gate, return_normal=False)[2:]
    for a, b in zip(both, normal + transposed, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    whole = quantize_mok_mxfp8_weights(gate, up, down)
    monkeypatch.setattr(mxfp8, "_CHUNK_BYTES", experts_per_chunk * 256 * 512 * 4)
    for a, b in zip(both, mxfp8.mxfp8_quantize(gate), strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    chunked = quantize_mok_mxfp8_weights(gate, up, down)
    pairs = list(zip(_flat(whole), _flat(chunked), strict=True))
    assert len(pairs) == 16
    for a, b in pairs:
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("comm", [4, 16])
def test_mxfp8_forward_reruns_are_bitwise(comm):
    """Fused MXFP8 forward reruns are bitwise over several macrobatches, also with
    16 communication SMs, where the compute CTAs steal tiles and BF16 shared tiles
    interleave with the routed MXFP8 tiles that read the block-scale ring."""
    require_gpu()
    hidden, intermediate, macro, mini = 1024, 256, 1024, 256
    tokens, topk = 2048, 2
    forward = MoKForwardMXFP8(False)
    x, _, shared, routed, _, s = _setup(
        hidden, intermediate, 7821, experts=5, tokens=tokens, topk=topk
    )
    routed_q = [quantize(w, transposed=False)[:2] for w in routed]
    runs = []
    for _ in range(3):
        combine = torch.zeros(
            tokens * topk, hidden, device="cuda", dtype=torch.bfloat16
        )
        values = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], routed_q[0], shared[1], routed_q[1], shared[2], routed_q[2],
            s["peers"], s["schedule"], s["num_tokens"], s["counts"],
            topk, None, comm, macro, mini,
        )  # fmt: skip
        torch.cuda.synchronize()
        runs.append((combine.clone(), values[7].clone()))
    for combine, y_shared in runs[1:]:
        assert torch.equal(combine.view(torch.int16), runs[0][0].view(torch.int16))
        assert torch.equal(y_shared.view(torch.int16), runs[0][1].view(torch.int16))


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 768), (512, 256, 1536)]
)
def test_mxfp8_forward(swiglu_limit, hidden, intermediate, macro):
    """Fused MXFP8 forward: E4M3 saves and both activation quantizations."""
    require_gpu()
    forward = MoKForwardMXFP8(swiglu_limit is not None)
    tokens, topk, mini, comm = 512, 2, 256, 4
    x, _, shared, routed, _, s = _setup(
        hidden, intermediate, 5197 + hidden + intermediate
    )
    routed_q = [quantize(w, transposed=False)[:2] for w in routed]
    combine = torch.zeros(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    values = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()],
        shared[0], routed_q[0], shared[1], routed_q[1], shared[2], routed_q[2],
        s["peers"], s["schedule"], s["num_tokens"], s["counts"],
        topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    torch.cuda.synchronize()
    (
        (x_t, x_sc_t),
        _,
        (gate_q, gate_sc),
        _,
        (up_q, up_sc),
        _,
        (h_t, h_sc_t),
        y_shared,
        _,
    ) = values
    actual = s["actual"]
    rows = s["schedule"][:actual]
    valid = rows >= 0
    source = torch.zeros(actual, hidden, device="cuda", dtype=torch.bfloat16)
    source[valid] = x[rows[valid].long() // topk]
    _, _, xq_t, xs_t = quantize(source)
    for value in (combine, y_shared):
        assert torch.isfinite(value).all().item()
    # The ring keeps the first macrobatch as backward context. x quantization is exact against the recipe; the stored
    # gate, up and activation values are judged in test_mok_accuracy.py.
    r = min(actual, macro)
    assert torch.equal(x_t[:, :r], xq_t.view(torch.uint8)[:, :r])
    tiles = xs_t.view(hidden // 128, actual // 128, 32, 16)
    assert torch.equal(x_sc_t[:, : r // 128], tiles[:, : r // 128])
    first = (combine.clone(), y_shared.clone(), gate_q[:r].clone(), gate_sc[: r // 128].clone(),
             up_q[:r].clone(), up_sc[: r // 128].clone(), h_t[:, :r].clone(), h_sc_t[:, : r // 128].clone())
    for _ in range(REPEATS - 1):
        combine.zero_()
        values = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], routed_q[0], shared[1], routed_q[1], shared[2], routed_q[2],
            s["peers"], s["schedule"], s["num_tokens"], s["counts"],
            topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        torch.cuda.synchronize()
        (_, _, (gq, gs), _, (uq, us), _, (ht, hs), ys, _) = values
        again = (combine, ys, gq[:r], gs[: r // 128], uq[:r], us[: r // 128], ht[:, :r], hs[:, : r // 128])
        assert _same(first, again)


def _mxfp8_training(
    swiglu_limit,
    hidden,
    intermediate,
    macro,
    seed,
    repeat=REPEATS - 1,
    experts=4,
    topk=2,
    fp32_wgrad=False,
):
    """Contract and repeatability of the fused MXFP8 forward + backward: every returned gradient finite, and
    ``repeat`` more launches on the same inputs give the same bytes.  ``fp32_wgrad`` adds every weight gradient into
    nonzero FP32 starting accumulators.  Returns the gradients of the first launch."""
    forward = MoKForwardMXFP8(swiglu_limit is not None)
    backward = MoKBackwardMXFP8(swiglu_limit is not None, fp32_wgrad)
    tokens, mini, comm = 512, 256, 4
    x, dy, shared, routed, scores, s = _setup(
        hidden,
        intermediate,
        seed + hidden + intermediate + macro,
        experts=experts,
        topk=topk,
    )
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer = torch.zeros(tokens * topk, hidden, **options)
    ds_peer = torch.zeros_like(scores)
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    initial = accumulators = None
    if fp32_wgrad:
        initial = [
            torch.randn(w.shape, device="cuda") * 0.01 for w in (*shared, *routed)
        ]
        accumulators = [value.clone() for value in initial]

    def run():
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
        if fp32_wgrad:
            for accumulator, start in zip(accumulators, initial, strict=True):
                accumulator.copy_(start)
        values = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *values[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        torch.cuda.synchronize()
        # Returned gradients only; macrobatch scratch has uninitialized padding.
        return tuple(t.clone() for t in (combine, values[7], dx_peer, ds_peer, result[0], *result[9:]))

    first = run()
    assert all(torch.isfinite(value).all().item() for value in first)
    for _ in range(repeat):
        again = run()
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    return first


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 1536), (512, 256, 512)]
)
def test_mxfp8_training(swiglu_limit, hidden, intermediate, macro):
    """Fused MXFP8 forward+backward: finite, reruns bitwise (accuracy: ``test_mok_accuracy.py``)."""
    require_gpu()
    _mxfp8_training(swiglu_limit, hidden, intermediate, macro, seed=7000)


@LIMITS
@pytest.mark.parametrize(
    "hidden,intermediate,macro", [(512, 256, 768), (512, 512, 1536), (512, 256, 512)]
)
def test_mxfp8_training_fp32_wgrad(swiglu_limit, hidden, intermediate, macro):
    """MXFP8 training with FP32 weight-gradient accumulation onto nonzero starts.

    Shared (BF16) and routed (MXFP8) weight gradients are both added into the
    caller's FP32 accumulators; the shapes include multiple macrobatches.
    """
    require_gpu()
    _mxfp8_training(swiglu_limit, hidden, intermediate, macro, seed=7000, fp32_wgrad=True)


@pytest.mark.parametrize("experts", [36, 9], ids=["ep8-local36", "ep32-local9"])
def test_mxfp8_training_local_experts(experts):
    """288 routed experts at EP8 / EP32 (36 or 9 local experts), top-8, clamped: finite, reruns bitwise."""
    require_gpu()
    _mxfp8_training(0.75, 512, 256, 2048, seed=7000, experts=experts, topk=8)


class _LaunchRecorder:
    """Stands in for a prepared kernel: records the launch keywords, launches nothing."""

    def __init__(self):
        self.calls = []

    def launch(self, *, grid, **kwargs):
        self.calls.append((grid, kwargs))

    def last(self, name):
        return next(kwargs for _, kwargs in reversed(self.calls) if name in kwargs)


def _check_counters(kwargs, expected):
    """Counter keywords ``expected`` (name -> numel) of one recorded fused launch:
    CUDA, contiguous, 1-D, int32 (uint32 for ``combine_next``), the exact
    numel, zero, 128-byte aligned, pairwise disjoint on one storage."""
    views = {name: kwargs[name] for name in expected}
    assert len({v.untyped_storage().data_ptr() for v in views.values()}) == 1
    for name, numel in expected.items():
        view = views[name]
        assert view.is_cuda and view.is_contiguous() and view.dim() == 1, name
        assert view.dtype == (torch.uint32 if name == "combine_next" else torch.int32)
        assert view.numel() == numel, (name, view.numel(), numel)
        assert torch.count_nonzero(view.view(torch.int32)).item() == 0, name
        assert view.data_ptr() % 128 == 0, name
    ranges = sorted(
        (v.data_ptr(), v.data_ptr() + 4 * v.numel()) for v in views.values()
    )
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:], strict=False))


def test_mxfp8_launch_counters_are_one_slab_per_launch():
    """At the launch boundary the MXFP8 launchers hand the kernels the counter
    names, dtypes and extents they always did, as views of one zero slab per
    launch (``combine_next`` the uint32 view of its element); a warm call
    dispatches one ``zeros`` and one split for them and no fill, zeroing or
    copy."""
    require_gpu()
    from collections import Counter

    from torch.utils._python_dispatch import TorchDispatchMode

    hidden, intermediate, macro, tokens, topk, mini, comm = (
        512,
        256,
        768,
        512,
        2,
        256,
        4,
    )
    x, dy, shared, routed, scores, s = _setup(hidden, intermediate, 2027)
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    capacity = s["peers"].numel()
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.zeros_like(combine), torch.zeros_like(scores)
    forward, recompute, backward = (
        MoKForwardMXFP8(),
        MoKRecomputeMXFP8(),
        MoKBackwardMXFP8(),
    )
    for launcher in (forward, recompute, backward):
        launcher.module = launcher.layout_module = _LaunchRecorder()
    backward.zero_module = backward.module

    def run():
        saved = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, None, comm, macro, mini,
        )  # fmt: skip
        recompute(
            x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
            *schedule, topk, None, comm, macro, mini,
        )  # fmt: skip
        backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *saved[:7], x, [x.data_ptr()], *schedule,
            topk, None, comm, macro, mini,
        )  # fmt: skip

    run()  # cold: peer tables and geometry
    ops = []

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            ops.append(func.overloadpacket.__name__)
            return func(*args, **(kwargs or {}))

    with Mode():
        forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, None, comm, macro, mini,
        )  # fmt: skip
    counted = Counter(ops)
    assert counted["zeros"] == 1 and counted["split_with_sizes"] == 1, counted
    assert not (set(counted) & {"fill_", "zero_", "copy_"}), counted
    run()
    minis, macros = (capacity + mini - 1) // mini, (capacity + macro - 1) // macro
    shared_rows, routed_rows = tokens // 256, capacity // 256
    ib = intermediate // 256
    _check_counters(
        forward.module.last("x_ready"),
        dict(
            x_ready=minis,
            gate_ready=(shared_rows + routed_rows) * ib,
            hidden_ready=shared_rows + routed_rows,
            y_ready=minis,
            y_done=capacity // 128,
            combine_next=1,
        ),
    )
    routed_capacity = min(capacity, macro)
    _check_counters(
        recompute.module.last("x_ready"),
        dict(
            x_ready=(routed_capacity + mini - 1) // mini,
            gate_ready=(shared_rows + routed_capacity // 256) * ib,
            hidden_ready=shared_rows + routed_capacity // 256,
        ),
    )
    _check_counters(
        backward.module.last("dy_ready"),
        dict(
            dy_ready=minis,
            dx_ready=minis,
            replay_x=minis,
            dh_ready=(shared_rows + routed_rows) * ib,
            dg_ready=shared_rows + routed_rows,
            replay_gu=routed_rows * ib,
            replay_h=routed_rows,
            buffers_done=macros * (len(routed[0]) + 1),
            weight_ready=macros,
        ),
    )
    launch = forward.module.last("x_ready")
    for name, weight in (("wg_q", fwd_w[0][0]), ("wg_sc", fwd_w[0][1])):
        assert launch[name].dtype == torch.uint8
        assert launch[name].data_ptr() == weight.data_ptr()
    assert launch["wg_shared"].shape == (1, *shared[0].shape)
    assert launch["wg_shared"].data_ptr() == shared[0].data_ptr()
    assert (
        len(forward._geometry)
        == len(recompute._geometry)
        == len(backward._geometry)
        == 1
    )


@LIMITS
def test_mxfp8_launchers_accept_a_precomputed_layout(swiglu_limit):
    """The MXFP8 launchers derive the expert layout themselves (``layout=None``)
    or read the one passed to them; both paths are bitwise equal."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, recompute = MoKForwardMXFP8(clamped), MoKRecomputeMXFP8(clamped)
    backward = MoKBackwardMXFP8(clamped)
    tokens, hidden, intermediate, topk, mini, comm, macro = (
        512,
        512,
        256,
        2,
        256,
        4,
        768,
    )
    x, dy, shared, routed, scores, s = _setup(hidden, intermediate, 811)
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    capacity = s["peers"].numel()
    layout = kernels.expert_layout(
        forward.layout_module, s["counts"], capacity, s["peers"]
    )
    options = dict(device="cuda", dtype=torch.bfloat16)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer, ds_peer = torch.zeros_like(combine), torch.zeros_like(scores)
    rows = min(s["actual"], macro)

    def run(layout_arg):
        combine.zero_()
        dx_peer.zero_()
        ds_peer.zero_()
        saved = forward(
            x, [x.data_ptr()], combine, [combine.data_ptr()],
            shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
            *schedule, topk, swiglu_limit, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        rebuilt = recompute(
            x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
            *schedule, topk, swiglu_limit, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        result = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *saved[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini, layout=layout_arg,
        )  # fmt: skip
        torch.cuda.synchronize()
        pieces = [combine, dx_peer, ds_peer, result[0], *result[9:], saved[7]]
        for context in (saved, rebuilt):
            for index, transposed in ((0, True), (2, False), (4, False), (6, True)):
                data, scales = context[index]
                if transposed:  # [features, tokens] data, [feature tiles, token tiles]
                    pieces += [data[:, :rows], scales[:, : rows // 128]]
                else:
                    pieces += [data[:rows], scales[: rows // 128]]
            pieces += [context[index] for index in (1, 3, 5)]
        return [piece.clone() for piece in pieces]

    first = run(None)
    for layout_arg in (None, layout, layout):
        again = run(layout_arg)
        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))


@LIMITS
@pytest.mark.parametrize("macro", [768, 4096], ids=["replay", "resident"])
def test_mxfp8_recompute_context(swiglu_limit, macro):
    """MXFP8 context-only recompute equals the saved context; backward is bitwise."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward, recompute = MoKForwardMXFP8(clamped), MoKRecomputeMXFP8(clamped)
    backward = MoKBackwardMXFP8(clamped)
    tokens, hidden, intermediate, topk, mini, comm = 512, 512, 256, 2, 256, 4
    x, dy, shared, routed, scores, s = _setup(hidden, intermediate, 811)
    q = [quantize(w) for w in routed]
    fwd_w = [(qq[0], qq[1]) for qq in q]
    bwd_w = (q[0], q[1], (q[2][2], q[2][3]))
    schedule = (s["peers"], s["schedule"], s["num_tokens"], s["counts"])
    combine = torch.zeros(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
    saved = forward(
        x, [x.data_ptr()], combine, [combine.data_ptr()],
        shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rebuilt = recompute(
        x, [x.data_ptr()], shared[0], fwd_w[0], shared[1], fwd_w[1],
        *schedule, topk, swiglu_limit, comm, macro, mini,
    )  # fmt: skip
    rows = min(s["actual"], macro)
    for index, transposed in ((0, True), (2, False), (4, False), (6, True)):
        (a, a_sc), (b, b_sc) = saved[index], rebuilt[index]
        if transposed:  # [features, tokens] data, [feature tiles, token tiles] scales
            assert torch.equal(a[:, :rows], b[:, :rows]), index
            assert torch.equal(a_sc[:, : rows // 128], b_sc[:, : rows // 128]), index
        else:
            assert torch.equal(a[:rows], b[:rows]), index
            assert torch.equal(a_sc[: rows // 128], b_sc[: rows // 128]), index
    for index in (1, 3, 5):
        assert torch.equal(saved[index], rebuilt[index]), index

    def gradients(context):
        dx_peer = torch.zeros(
            tokens * topk, hidden, device="cuda", dtype=torch.bfloat16
        )
        ds_peer = torch.zeros_like(scores)
        values = backward(
            dy, [dy.data_ptr()], dx_peer, [dx_peer.data_ptr()],
            scores, [scores.data_ptr()], ds_peer, [ds_peer.data_ptr()],
            shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
            *context[:7], x, [x.data_ptr()], *schedule,
            topk, swiglu_limit, comm, macro, mini,
        )  # fmt: skip
        return (dx_peer, ds_peer, values[0], *values[9:])

    for a, b in zip(gradients(saved), gradients(rebuilt), strict=True):
        assert torch.equal(a, b)


backend = cake_package("cake_mok_bf16", "backend")
jit = cake_package("cake_mok_bf16", "jit")
MoKConfig = cake_package("cake_mok_bf16", "workspace").MoKConfig


def test_select_multimacro_decision_table():
    """The sm_107a multimacro forward runs exactly when the observed schedule
    rows fill two macro batches; no observation keeps the default program."""
    select = backend.select_multimacro
    assert select(None, 1024) is False
    assert not select(0, 1024)
    assert not select(1024, 1024)  # one macro batch
    assert select(1025, 1024)
    assert select(2048, 1024)
    assert not select(2049, 1024)  # three macro batches
    assert select(265472, 262144)  # u32768-class padded rows
    assert not select(262144, 262144)


def test_multimacro_choice_follows_the_lagged_schedule_rows(monkeypatch, capsys):
    """The forward program choice reads this rank's padded schedule rows from the
    asynchronous load observation: a capacity of at most one macro batch never
    observes and keeps the default program; otherwise the latest completed
    observation decides, the choice in force holds during graph capture, the
    fixed communication split is unchanged and ``comm_split_log`` prints each
    change. Off sm_107a the default program always runs."""
    require_gpu()
    from types import SimpleNamespace

    ep_size, experts = 4, 4
    functional = backend.MoKFunctional(ep_size, experts, 2)
    sm107 = jit.target_arch() == "sm_107a"
    assert functional._multimacro_target is sm107
    # Every expert receives 128 routes (rank 0: 4 x 256 padded rows), then
    # every route goes to rank 0's experts (4 x 512 rows).
    balanced = (
        torch.arange(ep_size * 256 * 2, device="cuda", dtype=torch.int32)
        % (ep_size * experts)
    ).view(ep_size, 256, 2)
    skewed = balanced % experts
    capacity = ep_size * 256 * 2 + experts * 256
    config = MoKConfig(
        fwd_num_comm_sms=4,
        bwd_num_comm_sms=4,
        minibatch_size=256,
        macrobatch_size=512,
        comm_split_log=True,
    )
    workspaces = []  # keep every stand-in alive: the adapter keys its state by id

    def observe(routes, config, capacity=capacity):
        workspace = workspaces[-1]
        count = functional.scheduler(routes, capacity, 0)[2]
        functional._observe_load(workspace, config, count)
        torch.cuda.synchronize()
        return int(count.item())

    # Static guard: one macro batch holds the whole capacity.
    single = MoKConfig(
        fwd_num_comm_sms=4, bwd_num_comm_sms=4, minibatch_size=256, macrobatch_size=4096
    )
    workspaces.append(
        SimpleNamespace(ep_rank=0, ep_size=ep_size, schedule_capacity=capacity)
    )
    assert observe(balanced, single) == 1024
    assert functional._load_entry(workspaces[-1], create=False) is None
    assert functional._multimacro(single, workspaces[-1]) is False

    workspaces.append(
        SimpleNamespace(ep_rank=0, ep_size=ep_size, schedule_capacity=capacity)
    )
    capsys.readouterr()
    assert functional._multimacro(config, workspaces[-1]) is False  # no observation yet
    assert observe(balanced, config) == 1024  # two macro batches of 512
    assert functional._multimacro(config, workspaces[-1]) is sm107
    assert functional._comm_split(config, workspaces[-1]) == (4, 4)
    # Captured launches keep the choice in force: no observation is queued.
    count = functional.scheduler(skewed, capacity, 0)[2]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    functional._observe_load(workspaces[-1], config, count)
    assert functional._multimacro(config, workspaces[-1]) is sm107
    monkeypatch.undo()
    assert observe(skewed, config) == 2048  # four macro batches of 512
    assert functional._multimacro(config, workspaces[-1]) is False
    prefix = "[cake_mok_bf16 rank 0] forward program -> "
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if "forward program" in line
    ]
    assert lines == (
        [
            prefix + "multimacro (schedule rows 1024, macro batch 512)",
            prefix + "default (schedule rows 2048, macro batch 512)",
        ]
        if sm107
        else []
    )


def _two_rank_routes(tokens):
    """Expert IDs of two source ranks over 2 x 3 experts (owner = expert // 3):
    owner 0 receives 768 + 768 + 512 rows, owner 1 512 + 299 + 213 (padded
    512 + 512 + 256): owner 0's 2048 rows are three macro batches of 768 and
    two of 1024 or 1280, owner 1's 1280 rows two of 768 or 1024 and one of
    1280."""
    t = torch.arange(tokens, device="cuda")
    block = t % 3
    pairs = (
        torch.tensor([[0, 3], [1, 3], [2, 4]], device="cuda"),
        torch.tensor([[0, 1], [1, 0], [2, 5]], device="cuda"),
    )
    routes = [pair[block] for pair in pairs]
    routes[1][(t >= tokens - 128) & (block == 2), 1] = 4
    return routes


def _owner_schedule(routes, owner, local_experts, capacity):
    """Owner ``owner``'s padded schedule: per local expert the route slots of
    every source rank in rank order, then ``-1`` padding; the peer of a row is
    its source rank."""
    rows, peers, count_values = [], [], []
    for expert in range(owner * local_experts, (owner + 1) * local_experts):
        hits = [torch.where(r.flatten() == expert)[0].int() for r in routes]
        slots = torch.cat(hits)
        source = torch.cat([torch.full_like(h, rank) for rank, h in enumerate(hits)])
        padded = (slots.numel() + 255) // 256 * 256
        pad = torch.full(
            (padded - slots.numel(),), -1, dtype=torch.int32, device="cuda"
        )
        rows += [slots, pad]
        peers += [source, pad]
        count_values.append(padded)
    actual = sum(count_values)
    tail = torch.full((capacity - actual,), -1, dtype=torch.int32, device="cuda")
    return dict(
        peers=torch.cat(peers + [tail]),
        schedule=torch.cat(rows + [tail]),
        num_tokens=torch.tensor([actual], device="cuda", dtype=torch.int32),
        counts=torch.tensor(count_values, device="cuda", dtype=torch.int32),
        actual=actual,
    )


@LIMITS
@pytest.mark.parametrize("macro", [768, 1024, 1280])
def test_mxfp8_mixed_forward_programs_are_bitwise(swiglu_limit, macro):
    """Ranks may launch different forward programs in one step (the backend
    chooses per rank from its own lagged observation). Two-rank emulation on one
    GPU: each rank's forward reads both ranks' x and writes its routed rows into
    both ranks' combine buffers through the peer tables, then each rank's
    backward runs from its context. The programs synchronize only on this
    GPU's own counters (no system-scope flag), and the caller's barriers order
    the ranks, so running the ranks one after the other is the protocol. Every
    mix of default and multimacro programs gives the all-default combine rows,
    shared outputs, saved context and gradients bit for bit."""
    require_gpu()
    clamped = swiglu_limit is not None
    forward = MoKForwardMXFP8(clamped)
    if forward.multimacro_module is None:
        pytest.skip("The multimacro forward is generated for sm_107a only")
    backward = MoKBackwardMXFP8(clamped)
    tokens, hidden, intermediate, topk, mini, comm, local = 768, 1024, 256, 2, 256, 4, 3
    capacity = 2048 + 512
    ranks = [
        _setup(hidden, intermediate, 5668 + rank, experts=local, tokens=tokens)
        for rank in range(2)
    ]
    routes = _two_rank_routes(tokens)
    schedules = [_owner_schedule(routes, owner, local, capacity) for owner in range(2)]
    assert [s["actual"] for s in schedules] == [2048, 1280]
    weights = []
    for _, _, _, routed, _, _ in ranks:
        q = [quantize(w) for w in routed]
        weights.append(([(qq[0], qq[1]) for qq in q], (q[0], q[1], (q[2][2], q[2][3]))))
    x = [r[0] for r in ranks]
    dy = [r[1] for r in ranks]
    scores = [r[4] for r in ranks]
    x_ptrs = [t.data_ptr() for t in x]
    dy_ptrs = [t.data_ptr() for t in dy]
    score_ptrs = [t.data_ptr() for t in scores]

    def step(programs):
        combine = [
            torch.zeros(tokens * topk, hidden, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        combine_ptrs = [t.data_ptr() for t in combine]
        contexts = []
        for rank, multimacro in enumerate(programs):
            s, shared = schedules[rank], ranks[rank][2]
            fwd_w = weights[rank][0]
            values = forward(
                x[rank], x_ptrs, combine[rank], combine_ptrs,
                shared[0], fwd_w[0], shared[1], fwd_w[1], shared[2], fwd_w[2],
                s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                topk, swiglu_limit, comm, macro, mini, multimacro=multimacro,
            )  # fmt: skip
            torch.cuda.synchronize()
            contexts.append(values)
        dx = [torch.zeros_like(c) for c in combine]
        ds = [torch.zeros_like(t) for t in scores]
        grads = []
        for rank in range(2):
            s, shared = schedules[rank], ranks[rank][2]
            bwd_w = weights[rank][1]
            values = backward(
                dy[rank], dy_ptrs, dx[rank], [t.data_ptr() for t in dx],
                scores[rank], score_ptrs, ds[rank], [t.data_ptr() for t in ds],
                shared[0], bwd_w[0], shared[1], bwd_w[1], shared[2], bwd_w[2],
                *contexts[rank][:7], x[rank], x_ptrs,
                s["peers"], s["schedule"], s["num_tokens"], s["counts"],
                topk, swiglu_limit, comm, macro, mini,
            )  # fmt: skip
            torch.cuda.synchronize()
            grads.append((values[0], *values[9:]))
        words = list(combine) + dx + ds
        for rank, values in enumerate(contexts):
            rows = min(schedules[rank]["actual"], macro)  # the resident ring rows
            words.append(values[7])
            for index in (1, 3, 5):
                words.append(values[index])
            for index, transposed in ((0, True), (2, False), (4, False), (6, True)):
                data, scale = values[index]
                if transposed:  # [features, tokens] data, [feature, token] tiles
                    words += [data[:, :rows], scale[:, : rows // 128]]
                else:
                    words += [data[:rows], scale[: rows // 128]]
            words += _flat(grads[rank])
        return [w.clone() for w in words]

    reference = step((False, False))
    for programs in ((True, False), (False, True), (True, True)):
        words = step(programs)
        assert len(words) == len(reference)
        for index, (a, b) in enumerate(zip(words, reference, strict=True)):
            assert a.shape == b.shape and a.dtype == b.dtype, (programs, index)
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (
                programs,
                index,
            )
