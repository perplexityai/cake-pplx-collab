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
Workers of the multi-GPU MoK tests (``test_mok_distributed.py``): each runs in every rank of a group that
:func:`tests.experimental._mok_spawn.run_ranks` starts, and returns rank-local records.

* :func:`training_graph`: one complete schedule + forward + backward, eager and under CUDA Graph replay, over
  changing routes (including empty experts), at two widths.
* :func:`unequal_sources`: unequal source counts per rank (empty ranks and an all-empty step included), several
  graphs over one workspace, earlier graphs replayed after later ones, same-shape input updates.
* :func:`training_features`: clamped SwiGLU, context-only recompute and the context-free forward, FP32
  weight-gradient accumulation and native MXFP8 routed experts over unequal ranks.
* :func:`source_capacity`, :func:`backward_scratch`: the adapter's source-length guards, the schedule's expert
  layout, context ownership and the backward's scratch lifetime.

Every case checks the contract (shapes, finite values, exact zeros of empty ranks) and repeatability: five eager
launches, five graph replays and the eager launch give the same bytes, recompute and the context-free forward give
the saved context's bytes.  Every case with inputs is also judged (:func:`judge`): ``y``, ``dx``, the score
gradients and the six weight gradients of the whole group (gathered on every rank) against the float64 oracle beside
the naive BF16 implementation of :mod:`tests.experimental._mok_naive`, over the template's draws.  The routed rows of
each expert are taken in the order of the kernel's own schedule (exactly checked to hold every route once), which the
MX recipe's row blocks follow.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch
import torch.distributed as dist

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import TierJudge, judge_case
from tests.experimental import _mok_naive as naive

api = cake_package("cake_mok_bf16", "api")
kernels = cake_package("cake_mok_bf16", "_kernels")
MoKConfig = cake_package("cake_mok_bf16", "workspace").MoKConfig
create_mok_bf16_workspace = api.create_mok_bf16_workspace
prepare_mok_bf16 = api.prepare_mok_bf16
quantize_mok_mxfp8_weights = api.quantize_mok_mxfp8_weights

RESULT_NAMES = (
    "y",
    "dx",
    "d_scores",
    "dw_routed_gate",
    "dw_routed_up",
    "dw_routed_down",
    "dw_shared_gate",
    "dw_shared_up",
    "dw_shared_down",
)
REPEATS = 5


def _exact_matmul():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def _snapshot(values):
    torch.cuda.synchronize()
    return tuple(v.detach().clone() for v in values)


def _contract(values, n, hidden, topk):
    """Shapes of ``(y, dx, d_scores, ...)`` for ``n`` local tokens, every value finite."""
    assert values[0].shape == values[1].shape == (n, hidden), (values[0].shape, n)
    assert values[2].shape == (n, topk)
    for name, value in zip(RESULT_NAMES, values, strict=True):
        assert torch.isfinite(value).all().item(), name


def _weights(ep, local_experts, hidden, intermediate, device, seed=32471):
    """BF16 shared weights and the group's routed weights (identical on every rank); this rank's slice."""
    options = dict(dtype=torch.bfloat16, device=device)
    torch.manual_seed(seed)
    shared = [
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    total = ep * local_experts
    global_routed = [
        torch.randn(total, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(total, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(total, hidden, intermediate, **options) / intermediate**0.5,
    ]
    rank = dist.get_rank()
    routed = [w[rank * local_experts : (rank + 1) * local_experts].contiguous() for w in global_routed]
    return shared, global_routed, routed


def _random_data(counts, hidden, topk, experts, device):
    """Global inputs of one step from the current random state (identical on every rank)."""
    total = sum(counts)
    options = dict(dtype=torch.bfloat16, device=device)
    x = torch.randn(total, hidden, **options) * 0.125
    dy = torch.randn(total, hidden, **options) * 0.125
    ids = torch.rand(total, experts, device=device).argsort(-1)[:, :topk].contiguous()
    scores = torch.rand(total, topk, device=device) + 0.1
    scores = scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
    return dict(x=x, d_output=dy, expert_ids=ids, scores=scores)


def _load(iteration, data, counts):
    """Copy this rank's section of the global ``data`` into the iteration's input tensors."""
    rank = dist.get_rank()
    start = sum(counts[:rank])
    section = slice(start, start + counts[rank])
    for destination, key in zip(
        (iteration.x, iteration.ids, iteration.scores, iteration.dy),
        ("x", "expert_ids", "scores", "d_output"),
        strict=True,
    ):
        destination.copy_(data[key][section])
    torch.cuda.synchronize()
    dist.barrier()


def _gather_rows(local, counts):
    """The group's rows of a per-rank ``[counts[rank], ...]`` tensor, in rank order."""
    cap = max(max(counts), 1)
    padded = torch.zeros((cap, *local.shape[1:]), dtype=local.dtype, device=local.device)
    padded[: local.shape[0]] = local
    parts = [torch.empty_like(padded) for _ in counts]
    dist.all_gather(parts, padded)
    return torch.cat([p[:n] for p, n in zip(parts, counts, strict=True)])


def _gather(local):
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, local.contiguous())
    return parts


def segments(schedule, counts, local_experts, topk):
    """Per routed expert of the group: its rows in the owner's schedule order (route ``t * topk + k`` of the group's
    token ``t``, ``-1`` for padding), from every rank's schedule.  Exact checks: each expert's row count is a
    multiple of 256, the counts fill ``num_tokens``, every route of the step is held exactly once."""
    peers = _gather(schedule.peer_rank)
    tokens = _gather(schedule.peer_token_idx)
    per_expert = _gather(schedule.tokens_per_expert)
    totals = _gather(schedule.num_tokens)
    starts = [sum(counts[:r]) for r in range(len(counts))]
    start_of = torch.tensor(starts, device=schedule.peer_rank.device, dtype=torch.int64)
    out = []
    for owner in range(len(counts)):
        offset = 0
        for e, n in enumerate(per_expert[owner].tolist()):
            assert n % 256 == 0, (owner, e, n)
            p = peers[owner][offset : offset + n].long()
            t = tokens[owner][offset : offset + n].long()
            valid = p >= 0
            routes = torch.where(valid, (start_of[p.clamp(min=0)] + t // topk) * topk + t % topk, -1)
            out.append(naive.Segment(owner * local_experts + e, routes, offset))
            offset += n
        assert offset == int(totals[owner].item()), (owner, offset)
    held = torch.cat([s.routes[s.routes >= 0] for s in out]).sort().values
    assert torch.equal(held, torch.arange(sum(counts) * topk, device=held.device)), "schedule routes"
    return out


def judge(
    iteration,
    counts,
    shared,
    global_routed,
    *,
    topk,
    local_experts,
    hidden,
    seed,
    limit=None,
    mxfp8_global=None,
    fp32_wgrad=False,
    macro=768,
):
    """Judge the iteration's outputs of the whole group over :data:`tests.cake_naive.DRAWS` draws of random inputs,
    and over stage 2's where an output asks for them (every rank runs the same judge on gathered values, so every rank
    takes the same draws).  Returns the verdicts, the alternatives' verdicts and the failures."""
    experts = len(counts) * local_experts
    device = iteration.x.device
    alt = TierJudge()
    naive_values = {}
    mx = None if mxfp8_global is None else dict(forward=mxfp8_global[0], backward=mxfp8_global[1])

    def run(j, _seed):
        data = _random_data(counts, hidden, topk, experts, device)
        _load(iteration, data, counts)
        values = _snapshot(iteration.run())
        local_routed_dw = values[3:6]
        outputs = dict(
            y=_gather_rows(values[0], counts),
            dx=_gather_rows(values[1], counts),
            d_scores=_gather_rows(values[2], counts),
        )
        for name, value in zip(RESULT_NAMES[3:6], local_routed_dw, strict=True):
            outputs[name] = torch.cat(_gather(value))
        for name, value in zip(RESULT_NAMES[6:], values[6:], strict=True):
            outputs[name] = torch.stack(_gather(value))
        segs = segments(iteration.schedule, counts, local_experts, topk)
        refs = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            refs[dtype] = naive.layer(
                data["x"], data["d_output"], data["scores"], shared, global_routed, segs, limit, dtype,
                sections=counts, mx=mx, macro=None if fp32_wgrad else macro,
            )  # fmt: skip
        oracle = {n: v for n, v in zip(("y", "dx", "d_scores"), (refs[naive.ORACLE][k] for k in ("y", "dx", "d_scores")))}
        oracle.update(zip(RESULT_NAMES[3:], refs[naive.ORACLE]["dw"]))
        r = refs[naive.NAIVE]
        naive_values[j] = dict(y=r["y"], dx=r["dx"], d_scores=r["d_scores"], **dict(zip(RESULT_NAMES[3:], r["dw"])))
        if "alt_dw" in r:
            for name, value in zip(RESULT_NAMES[3:6], r["alt_dw"], strict=True):
                alt.add(name, j, outputs[name], oracle[name], {"bf16": value})
        alt.add("d_scores", j, outputs["d_scores"], oracle["d_scores"], {"bf16": r["alt_d_scores"]})
        return outputs, oracle

    verdict, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed, reported=(alt,))
    return list(verdict.verdicts().values()), list(alt.verdicts().values()), failures


class TrainingIteration:
    """One complete schedule + forward + backward, optionally CUDA-Graph captured.

    ``weights`` are the BF16 shared and routed weights. ``mxfp8_weights`` (the
    pair returned by ``quantize_mok_mxfp8_weights``) runs the routed experts in
    native MXFP8. ``recompute`` discards the forward context and rebuilds it
    with ``recompute_forward_context`` before backward (activation
    checkpointing); with ``save_context=False`` (BF16 routed experts, needs
    ``recompute``) the forward stores no context at all. ``accumulators`` are
    six FP32 tensors zeroed and filled by an ``fp32_wgrad`` adapter.
    """

    def __init__(
        self,
        config,
        workspace,
        x,
        ids,
        scores,
        dy,
        weights,
        *,
        functional,
        swiglu_limit=None,
        mxfp8_weights=None,
        recompute=False,
        accumulators=None,
        save_context=True,
    ):
        if not save_context and not recompute:
            raise ValueError(
                "save_context=False needs recompute=True: backward reads the recomputed context"
            )
        if (
            x.dtype != torch.bfloat16
            or dy.dtype != torch.bfloat16
            or scores.dtype != torch.float32
            or ids.dtype != torch.int64
            or len(weights) != 6
            or any(
                not isinstance(v, torch.Tensor) or v.dtype != torch.bfloat16
                for v in weights
            )
        ):
            raise ValueError(
                "This workload requires BF16 x/dY/expert tensors, FP32 scores and int64 IDs"
            )
        if (
            x.shape != dy.shape
            or ids.shape != scores.shape
            or ids.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "Source input, routing and upstream-gradient shapes must agree"
            )
        self.config, self.workspace = config, workspace
        self.functional = functional
        self.x, self.ids, self.scores, self.dy = x, ids, scores, dy
        self.weights = weights
        self.swiglu_limit = swiglu_limit
        self.recompute = recompute
        self.save_context = save_context
        self.accumulators = accumulators
        if mxfp8_weights is None:
            self.forward_weights = self.backward_weights = weights
        else:
            self.forward_weights = (*weights[:3], *mxfp8_weights[0])
            self.backward_weights = (*weights[:3], *mxfp8_weights[1])
        self.graph = None
        self.outputs = None

    def _execute(self):
        limit = self.swiglu_limit
        self.schedule = self.functional.build_schedule(
            self.workspace,
            self.config,
            self.ids,
            num_local_experts=self.weights[3].shape[0],
        )
        y, self.context = self.functional.forward(
            self.config,
            self.workspace,
            self.schedule,
            self.x,
            self.scores,
            *self.forward_weights,
            swiglu_limit=limit,
            save_context=self.save_context,
        )
        if self.recompute:
            self.context = None
            w = self.forward_weights
            self.context = self.functional.recompute_forward_context(
                self.config,
                self.workspace,
                self.schedule,
                self.x,
                w[0],
                w[1],
                w[3],
                w[4],
                swiglu_limit=limit,
            )
        extra = {}
        if self.accumulators is not None:
            for accumulator in self.accumulators:
                accumulator.zero_()
            extra["weight_grad_accumulators"] = self.accumulators
        gradients = self.functional.backward(
            self.config,
            self.workspace,
            self.schedule,
            self.context,
            self.dy,
            self.x,
            self.scores,
            *self.backward_weights,
            swiglu_limit=limit,
            **extra,
        )
        self.outputs = (y, *gradients)
        weight_dtype = torch.bfloat16 if self.accumulators is None else torch.float32
        expected = (torch.bfloat16, torch.bfloat16, torch.float32) + (weight_dtype,) * 6
        if any(v.dtype != d for v, d in zip(self.outputs, expected, strict=True)):
            raise RuntimeError("Output/gradient formats changed")
        return self.outputs

    def capture(self):
        if self.graph is not None:
            raise RuntimeError("Iteration is already captured")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._execute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._execute()

    def run(self):
        if self.graph is None:
            return self._execute()
        self.graph.replay()
        return self.outputs


def training_graph():
    """A complete training step, eager then under CUDA Graph replay, at two widths; then judged."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    tokens = 512
    topk, local_experts = (2, 4) if ep in (1, 4) else (8, 256 // ep)
    total_experts = ep * local_experts
    functional = prepare_mok_bf16(ep_size=ep, local_experts=local_experts, topk=topk)
    counts = [tokens] * ep
    out = []
    for hidden, intermediate in ((256, 256), (512, 512)):
        config, workspace = create_mok_bf16_workspace(
            group=dist.group.WORLD,
            device=device,
            num_local_tokens=tokens,
            hidden_size=hidden,
            topk=topk,
            fwd_num_comm_sms=8,
            bwd_num_comm_sms=8,
            minibatch_size=256,
            macrobatch_size=768,
            schedule_capacity_multiplier=2.0,
        )
        options = dict(dtype=torch.bfloat16, device=device)
        shared, global_routed, routed = _weights(ep, local_experts, hidden, intermediate, device)
        weights = (*shared, *routed)
        x, dy = torch.empty(tokens, hidden, **options), torch.empty(tokens, hidden, **options)
        ids = torch.empty(tokens, topk, dtype=torch.int64, device=device)
        scores = torch.empty(tokens, topk, device=device)
        iteration = TrainingIteration(config, workspace, x, ids, scores, dy, weights, functional=functional)

        def fill(generation, empty):
            torch.manual_seed(8291 + generation)
            all_x = torch.randn(tokens * ep, hidden, **options) * 0.125
            all_dy = torch.randn_like(all_x) * 0.125
            all_scores = torch.rand(tokens * ep, topk, device=device) + 0.1
            all_scores.div_(all_scores.sum(-1, keepdim=True)).mul_(2.5)
            rows = torch.arange(tokens * ep, device=device) + generation
            all_ids = torch.stack(
                [(rows + slot) % (topk if empty else total_experts) for slot in range(topk)], -1
            )
            _load(iteration, dict(x=all_x, d_output=all_dy, expert_ids=all_ids, scores=all_scores), counts)

        eager = {}
        for generation, empty in ((0, False), (1, True), (2, False), (11, False)):
            fill(generation, empty)
            first = _snapshot(iteration.run())
            _contract(first, tokens, hidden, topk)
            again = [_snapshot(iteration.run()) for _ in range(REPEATS - 1)]
            assert all(_same(first, other) for other in again), (rank, hidden, generation)
            eager[generation] = first
        iteration.capture()
        replays = []
        for generation, empty in ((11, False),) * REPEATS + ((17, True), (23, False)):
            fill(generation, empty)
            values = _snapshot(iteration.run())
            _contract(values, tokens, hidden, topk)
            if generation == 11:
                replays.append(values)
        assert all(_same(eager[11], other) for other in replays), (rank, hidden)
        verdicts = judge(
            iteration, counts, shared, global_routed, topk=topk, local_experts=local_experts, hidden=hidden,
            seed=4100 + hidden,
        )  # fmt: skip
        out.append(dict(case=f"h{hidden}", judged=verdicts))
        del iteration
        torch.cuda.synchronize()
        dist.barrier()
    return out


def unequal_sources():
    """Unequal source counts (empty ranks included), several graphs over one workspace; judged."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    assert ep == 4
    hidden, intermediate, topk, local_experts = 256, 256, 2, 4
    functional = prepare_mok_bf16(ep_size=ep, local_experts=local_experts, topk=topk)
    shared, global_routed, routed = _weights(ep, local_experts, hidden, intermediate, device)
    weights = (*shared, *routed)
    options = dict(device=device, dtype=torch.bfloat16)
    out = []
    for counts in ([0, 1, 255, 513], [0, 513, 769, 8193], [0, 0, 0, 0]):
        config, workspace = create_mok_bf16_workspace(
            group=dist.group.WORLD,
            device=device,
            num_local_tokens=counts[rank],
            hidden_size=hidden,
            topk=topk,
            fwd_num_comm_sms=8,
            bwd_num_comm_sms=8,
            minibatch_size=256,
            macrobatch_size=768,
            schedule_capacity_multiplier=2.0,
        )
        assert workspace.initial_source_counts == tuple(counts)

        def data_for(lengths, seed, empty_experts):
            torch.manual_seed(seed)
            total = sum(lengths)
            x = torch.randn(total, hidden, **options) * 0.125
            dy = torch.randn_like(x) * 0.125
            scores = torch.rand(total, topk, device=device) + 0.1
            scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
            row = torch.arange(total, device=device) + seed
            expert_count = topk if empty_experts else ep * local_experts
            ids = torch.stack([(row + slot) % expert_count for slot in range(topk)], -1)
            return dict(x=x, d_output=dy, expert_ids=ids, scores=scores)

        def iteration_for(lengths, data):
            start = sum(lengths[:rank])
            section = slice(start, start + lengths[rank])
            values = [data[key][section].clone() for key in ("x", "expert_ids", "scores", "d_output")]
            return TrainingIteration(config, workspace, *values, weights, functional=functional)

        def check(iteration, lengths):
            actual = iteration.run()
            torch.cuda.synchronize()
            _contract(actual, lengths[rank], hidden, topk)
            if lengths[rank] == 0:
                assert all(torch.count_nonzero(a).item() == 0 for a in actual[6:])
            saved = tuple(a.detach().clone() for a in actual)
            # All peers finish reading before another graph mutates symmetric storage.
            dist.barrier()
            return saved

        data = data_for(counts, 8291, False)
        first = iteration_for(counts, data)
        eager = check(first, counts)
        first.capture()
        saved = [check(first, counts) for _ in range(REPEATS)]
        assert all(_same(eager, other) for other in saved)
        changed_counts = counts[1:] + counts[:1]
        second = iteration_for(changed_counts, data_for(changed_counts, 8299, True))
        second.capture()
        check(second, changed_counts)
        assert _same(saved[0], check(first, counts))  # the earlier graph after the later one
        _load(first, data_for(counts, 8311, True), counts)
        check(first, counts)
        record = dict(case="counts-" + "-".join(map(str, counts)))
        if sum(counts):
            record["judged"] = judge(
                first, counts, shared, global_routed, topk=topk, local_experts=local_experts, hidden=hidden,
                seed=4200 + sum(counts),
            )  # fmt: skip
        out.append(record)
        del first, second
        torch.cuda.synchronize()
        dist.barrier()
    return out


LAYOUTS = {1: (4, 2), 4: (64, 8), 8: (36, 8), 16: (16, 8), 32: (9, 8), 64: (4, 8)}
LIMIT = 0.125  # Clamps a substantial fraction of gate and up at this input scale.
MXFP8_LIMIT = 0.375
FEATURE_CASES = (
    dict(name="bf16-clamped", swiglu_limit=LIMIT),
    dict(name="bf16-recompute", recompute=True),
    dict(name="bf16-clamped-recompute", swiglu_limit=LIMIT, recompute=True),
    dict(name="bf16-fp32-wgrad", fp32_wgrad=True),
    dict(name="mxfp8", mxfp8=True),
    dict(name="mxfp8-fp32-wgrad", mxfp8=True, fp32_wgrad=True),
    dict(name="mxfp8-clamped-recompute", mxfp8=True, swiglu_limit=MXFP8_LIMIT, recompute=True),
)
#: The MXFP8 run's ``y`` differs from the BF16 run's by more than this relative L2 distance: the routed GEMMs ran in
#: MXFP8 (a property of the program, not an accuracy bound).
MXFP8_MIN_DISTANCE = 1e-3


def _rel_l2(actual, expected):
    stats = torch.zeros(2, dtype=torch.float64, device=actual.device)
    stats[0] = (actual.double() - expected.double()).square().sum()
    stats[1] = expected.double().square().sum()
    dist.all_reduce(stats)
    return (stats[0] / stats[1]).sqrt().item() if stats[1] > 0 else 0.0


def training_features():
    """Clamped SwiGLU, recompute, the context-free forward, FP32 accumulation and MXFP8 over unequal ranks."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    local_experts, topk = LAYOUTS[ep]
    total_experts = ep * local_experts
    hidden, intermediate = 512, 256
    counts = ([513, 0, 1031, 256] * ep)[:ep] if ep > 1 else [1031]
    shared, global_routed, routed = _weights(ep, local_experts, hidden, intermediate, device)
    weights = (*shared, *routed)
    mxfp8_weights = quantize_mok_mxfp8_weights(*routed)
    mxfp8_global = quantize_mok_mxfp8_weights(*global_routed)
    adapters = {
        False: prepare_mok_bf16(ep_size=ep, local_experts=local_experts, topk=topk),
        True: prepare_mok_bf16(ep_size=ep, local_experts=local_experts, topk=topk, fp32_wgrad=True),
    }
    config, workspace = create_mok_bf16_workspace(
        group=dist.group.WORLD,
        device=device,
        num_local_tokens=counts[rank],
        hidden_size=hidden,
        topk=topk,
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=1.0,
    )
    assert workspace.initial_source_counts == tuple(counts)
    torch.manual_seed(8291)
    data = _random_data(counts, hidden, topk, total_experts, device)
    start = sum(counts[:rank])
    section = slice(start, start + counts[rank])
    x, ids, scores, dy = (data[k][section].clone() for k in ("x", "expert_ids", "scores", "d_output"))
    bf16_outputs, records = None, []
    for case in FEATURE_CASES:
        limit = case.get("swiglu_limit")
        mxfp8 = case.get("mxfp8", False)
        fp32_wgrad = case.get("fp32_wgrad", False)
        functional = adapters[fp32_wgrad]
        functional.prepare(limit, mxfp8)
        functional.prepare_recompute(limit, mxfp8)
        accumulators = [torch.zeros(w.shape, device=device) for w in weights] if fp32_wgrad else None
        # Each case starts from the same inputs (the judge of the previous case loaded others).
        for destination, value in zip((x, ids, scores, dy), (data[k][section] for k in ("x", "expert_ids", "scores", "d_output")), strict=True):
            destination.copy_(value)

        def iteration(recompute, save_context=True):
            return TrainingIteration(
                config, workspace, x, ids, scores, dy, weights, functional=functional, swiglu_limit=limit,
                mxfp8_weights=mxfp8_weights if mxfp8 else None, recompute=recompute, accumulators=accumulators,
                save_context=save_context,
            )  # fmt: skip

        saved_context = iteration(False)
        outputs = _snapshot(saved_context.run())
        _contract(outputs, counts[rank], hidden, topk)
        record = dict(case=case["name"], counts=counts)
        if limit is not None:
            above = (x.float() @ shared[0].float().T > limit).float()
            stats = torch.tensor([above.sum(), above.numel()], device=device)
            dist.all_reduce(stats)
            record["shared_gate_clamped_fraction"] = (stats[0] / stats[1]).item()
        again = [_snapshot(saved_context.run()) for _ in range(REPEATS - 1)]
        assert all(_same(outputs, other) for other in again), (rank, case["name"], "eager repeat")
        dist.barrier()
        saved_context.capture()
        replays = [_snapshot(saved_context.run()) for _ in range(REPEATS)]
        assert all(_same(outputs, other) for other in replays), (rank, case["name"], "graph replay")
        dist.barrier()
        if case.get("recompute"):
            checkpointed = _snapshot(iteration(True).run())
            differs = [n for n, a, b in zip(RESULT_NAMES, outputs, checkpointed, strict=True) if not torch.equal(a, b)]
            assert not differs, (rank, case["name"], "recompute", differs)
            if not mxfp8:
                # Context-free forward: y and every gradient keep the saved-context bits, eagerly and under graph
                # replay.
                functional.prepare(limit, save_context=False)
                context_free = iteration(True, save_context=False)
                runs = [_snapshot(context_free.run())]
                dist.barrier()
                context_free.capture()
                runs.append(_snapshot(context_free.run()))
                differs = sorted(
                    {n for run in runs for n, a, b in zip(RESULT_NAMES, outputs, run, strict=True) if not torch.equal(a, b)}
                )
                assert not differs, (rank, case["name"], "context-free", differs)
                del context_free
                torch.cuda.synchronize()
        dist.barrier()
        if counts[rank] == 0:
            assert all(torch.count_nonzero(v).item() == 0 for v in outputs[6:])
        if not mxfp8 and limit is None and not fp32_wgrad and bf16_outputs is None:
            bf16_outputs = outputs
        if mxfp8 and limit is None and bf16_outputs is not None:
            record["rel_l2_vs_bf16_run"] = _rel_l2(outputs[0], bf16_outputs[0])
            assert record["rel_l2_vs_bf16_run"] > MXFP8_MIN_DISTANCE
        record["judged"] = judge(
            saved_context, counts, shared, global_routed, topk=topk, local_experts=local_experts, hidden=hidden,
            seed=4300 + len(records), limit=limit, mxfp8_global=mxfp8_global if mxfp8 else None,
            fp32_wgrad=fp32_wgrad,
        )  # fmt: skip
        records.append(record)
        del saved_context
        torch.cuda.synchronize()
        dist.barrier()
    return records


def source_capacity():
    """The adapter's source-length guards, the schedule's expert layout and context ownership (EP4 / 16 / 64)."""
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    rank, ep = dist.get_rank(), dist.get_world_size()
    counts = [0, 1, 255, 256] * (ep // 4)
    n, h = counts[rank], 256
    k, le = (2, 4) if ep == 4 else (8, 256 // ep)
    f = prepare_mok_bf16(ep_size=ep, local_experts=le, topk=k)
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=2.0,
    )
    args = dict(device=device, num_local_tokens=n, hidden_size=h, topk=k)
    with pytest.raises(ValueError, match="nonnegative"):
        f.create_workspace(
            config,
            dist.group.WORLD,
            **dict(args, num_local_tokens=-1 if rank == 0 else n),
        )
    with pytest.raises(ValueError, match="agree"):
        f.create_workspace(
            config,
            dist.group.WORLD,
            **args,
            source_capacity=512 if rank == 0 else 1024,
        )
    with pytest.raises(ValueError, match="covering"):
        f.create_workspace(config, dist.group.WORLD, **args, source_capacity=200)
    workspace = f.create_workspace(
        config, dist.group.WORLD, **args, source_capacity=513
    )
    assert (
        workspace.source_capacity == 768
        and workspace.initial_source_counts == tuple(counts)
    )
    with pytest.raises(ValueError, match="exceeds source capacity"):
        f.build_schedule(
            workspace,
            config,
            torch.empty(769, k, dtype=torch.int64),
            num_local_experts=le,
        )
    with pytest.raises(ValueError, match="Expert IDs"):
        f.build_schedule(
            workspace,
            config,
            torch.empty(n, k, device=device, dtype=torch.float32),
            num_local_experts=le,
        )
    row = torch.arange(n, device=device)
    ids = torch.stack([(row + slot) % (ep * le) for slot in range(k)], -1)
    with pytest.raises(TypeError, match="comm_split_log"):
        f.build_schedule(
            workspace,
            dataclasses.replace(config, comm_split_log=1),
            ids,
            num_local_experts=le,
        )
    x = torch.zeros(n, h, device=device, dtype=torch.bfloat16)
    dy, scores = torch.zeros_like(x), torch.ones(n, k, device=device)
    shared = [
        torch.ones(h, h, device=device, dtype=torch.bfloat16) * 0.01 for _ in range(3)
    ]
    routed = [
        torch.ones(le, h, h, device=device, dtype=torch.bfloat16) * 0.01
        for _ in range(3)
    ]
    schedule = f.build_schedule(workspace, config, ids, num_local_experts=le)
    # The schedule carries its expert layout, derived once; it equals a fresh
    # derivation on the written prefix (the rest is never read).
    capacity = workspace.storage.schedule_capacity
    assert schedule.layout is not None and schedule.layout.dtype == torch.int32
    assert schedule.layout.numel() == 3 * le + capacity // 256
    written = 3 * le + int(schedule.num_tokens.item()) // 256
    derived = kernels.expert_layout(
        f.scheduler.layout_module,
        schedule.tokens_per_expert,
        capacity,
        schedule.peer_rank,
    )
    assert torch.equal(schedule.layout[:written], derived[:written])
    with pytest.raises(ValueError, match="belong"):
        f.forward(
            config,
            dataclasses.replace(workspace),
            schedule,
            x,
            scores,
            *shared,
            *routed,
        )
    with pytest.raises(ValueError, match="shape"):
        f.forward(
            config,
            workspace,
            schedule,
            torch.empty(n + 1, h, device=device, dtype=torch.bfloat16),
            scores,
            *shared,
            *routed,
        )
    with pytest.raises(ValueError, match="router_weights shape"):
        narrow = scores[:, :1].contiguous()
        f.forward(config, workspace, schedule, x, narrow, *shared, *routed)
    y, context = f.forward(config, workspace, schedule, x, scores, *shared, *routed)
    with pytest.raises(ValueError, match="matching"):
        f.backward(
            config,
            workspace,
            dataclasses.replace(schedule),
            context,
            dy,
            x,
            scores,
            *shared,
            *routed,
        )
    gradients = f.backward(
        config, workspace, schedule, context, dy, x, scores, *shared, *routed
    )
    assert y.shape == gradients[0].shape == (n, h) and gradients[1].shape == (n, k)
    assert all(torch.count_nonzero(v).item() == 0 for v in (y, *gradients))
    torch.cuda.synchronize()
    dist.barrier()


def backward_scratch():
    """``backward`` keeps only ``dx_shared`` and the six weight gradients of the
    fused launch alive past the launch: the kernel's routed and shared scratch
    is released before the output epilogue runs, ``dx_shared`` before the
    score-gradient copy, and the returned gradients are the launcher's weight
    gradients in the documented order."""
    import weakref

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    ep = dist.get_world_size()
    n, h = 256, 256
    k, le = (2, 4) if ep == 4 else (8, 256 // ep)
    f = prepare_mok_bf16(ep_size=ep, local_experts=le, topk=k)
    config = MoKConfig(
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=2.0,
    )
    workspace = f.create_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=n,
        hidden_size=h,
        topk=k,
    )
    generator = torch.Generator(device=device).manual_seed(2026 + dist.get_rank())
    options = dict(device=device, dtype=torch.bfloat16)
    x = torch.randn(n, h, generator=generator, **options) * 0.125
    dy = torch.randn(n, h, generator=generator, **options) * 0.125
    scores = torch.rand(n, k, device=device, generator=generator) + 0.1
    shared = [
        torch.randn(h, h, generator=generator, **options) / h**0.5 for _ in range(3)
    ]
    routed = [
        torch.randn(le, h, h, generator=generator, **options) / h**0.5 for _ in range(3)
    ]
    row = torch.arange(n, device=device)
    ids = torch.stack([(row + slot) % (ep * le) for slot in range(k)], -1)
    schedule = f.build_schedule(workspace, config, ids, num_local_experts=le)
    _, context = f.forward(config, workspace, schedule, x, scores, *shared, *routed)

    key = (False, False)  # (MXFP8, clamped SwiGLU) of the BF16 plain variant
    launcher = f._backward_kernels[key]
    scratch, dx_shared_ref, weight_grads = [], [], []
    # The BF16 backward runs the launcher's prebuilt plan: record its launch
    # and the planned output epilogue (instance attributes shadow the static
    # methods the backend calls through the launcher and ``f.epilogues``).
    launch = launcher.launch_planned

    def recording(*args, **kwargs):
        values = launch(*args, **kwargs)
        dx_shared_ref.append(weakref.ref(values[0]))
        scratch.extend(weakref.ref(value) for value in values[1:9])
        weight_grads.extend(values[9:])
        return values

    epilogue = f.epilogues.launch_planned

    def checking(plan, shared_grad, *rest):
        # Entry of the epilogue: every scratch output of the launch is dead,
        # dx_shared is the tensor the launcher returned.
        assert len(scratch) == 8 and all(ref() is None for ref in scratch)
        assert dx_shared_ref[0]() is shared_grad
        return epilogue(plan, shared_grad, *rest)

    launcher.launch_planned = recording
    f.epilogues.launch_planned = checking
    gradients = f.backward(
        config, workspace, schedule, context, dy, x, scores, *shared, *routed
    )
    assert len(scratch) == 8 and dx_shared_ref[0]() is None
    assert gradients[0].shape == (n, h) and gradients[1].shape == (n, k)
    # (dx, dscores, routed gate / up / down, shared gate / up / down) = the
    # launcher's (shared gate, routed gate, shared up, routed up, shared down,
    # routed down) weight gradients, reordered.
    assert len(weight_grads) == 6
    order = (1, 3, 5, 0, 2, 4)
    assert all(gradients[2 + i] is weight_grads[j] for i, j in enumerate(order))
    torch.cuda.synchronize()
    dist.barrier()
