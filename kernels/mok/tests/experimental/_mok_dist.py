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
* :func:`training_features`: clamped SwiGLU, context-only recompute (also called with router weights, which the
  ``"after_down"`` recompute checks and does not read) and the context-free forward, FP32 weight-gradient
  accumulation and native MXFP8 routed experts over unequal ranks.
* :func:`requantized_training`: the MXFP8 token-direction activation copies requantized from the forward's copies
  (``mxfp8_transposed="requantized"``) over unequal ranks, plain and clamped SwiGLU with recompute, BF16 and FP32
  weight gradients: repeatability, graph replay, recompute and every output but the routed weight gradients equal to
  the default adapter's on the same inputs (not judged here: the outputs it changes are judged at one GPU, in
  ``test_mok_accuracy.py``).
* :func:`source_capacity`, :func:`backward_scratch`: the adapter's source-length guards, the schedule's expert
  layout, context ownership and the backward's scratch lifetime.
* :func:`before_down_training`: the router weight before the down projection (``router_weight_position=
  "before_down"``, clamped SwiGLU, FP32 weight gradients) through the adapter over changing source counts on one
  workspace (padded and unpadded staging, an empty source rank in groups of several ranks), recompute and the
  context-free forward included, another communication-SM split on each rank, the unplanned launches of a schedule
  without its expert layout, and a test-local autograd function around the checkpointed step that overwrites the
  workspace's router weights between its launches; at the diagnostic layouts and, on the requester's inputs, at the
  384-expert layouts (4, 96) and (8, 48) with top-6.
* :func:`before_down_placement`, :func:`before_down_full_skew`: the requester's inputs at a 384-expert layout over
  chosen placements (a token's experts on one rank, spread over the ranks, random; no route to one rank's experts and
  an empty source rank) and a full skew (every route of the group to one rank's experts, padded the most) at the
  requester's no-overflow multiplier, with the recompute's context against the saved one.
* :func:`before_down_overflow`: the full skew on a workspace at the package's default multiplier, whose schedule
  traps on the target rank (run by :func:`tests.experimental._mok_spawn.run_ranks_isolated`);
  :func:`before_down_small_step` then checks the GPUs serve a new group.
* :func:`scaled_group_training`: the 512-expert top-10 group of the explicit modes (``router_weight_gradient=
  "direct"``, ``shared_expert="scaled"``) through the adapter at EP8 ``(8, 64)`` over changing source counts, on inputs
  drawn on the CPU: contract, repeatability, graph replay, the checkpointed step and the whole group judged.
* :func:`before_down_digests`: SHA-256 of every output of a placement step and a full skew, compared across group
  launches.

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
import hashlib
import math
import os

import pytest
import torch
import torch.distributed as dist

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake_naive import TierJudge, judge_case
from tests.experimental import _mok_naive as naive
from tests.experimental._mok_one_gpu import bound_multiplier
from tests.experimental._mok_reference import rand, randn, randn_like

api = cake_package("cake_mok_bf16", "api")
jit = cake_package("cake_mok_bf16", "jit")
kernels = cake_package("cake_mok_bf16", "_kernels")
MoKConfig = cake_package("cake_mok_bf16", "workspace").MoKConfig
schedule_capacity_for = cake_package("cake_mok_bf16", "workspace").schedule_capacity_for
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
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(intermediate, hidden, **options) / hidden**0.5,
        randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    total = ep * local_experts
    global_routed = [
        randn(total, intermediate, hidden, **options) / hidden**0.5,
        randn(total, intermediate, hidden, **options) / hidden**0.5,
        randn(total, hidden, intermediate, **options) / intermediate**0.5,
    ]
    rank = dist.get_rank()
    routed = [w[rank * local_experts : (rank + 1) * local_experts].contiguous() for w in global_routed]
    return shared, global_routed, routed


def _random_data(counts, hidden, topk, experts, device, weight_before_down=False):
    """Global inputs of one step from the current random state (identical on every rank).  With
    ``weight_before_down`` the router weights are in [0, 1.5) and every 5th token's first one is exactly 0."""
    total = sum(counts)
    options = dict(dtype=torch.bfloat16, device=device)
    x = randn(total, hidden, **options) * 0.125
    dy = randn(total, hidden, **options) * 0.125
    ids = rand(total, experts, device=device).argsort(-1)[:, :topk].contiguous()
    scores = rand(total, topk, device=device) + 0.1
    scores = scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
    if weight_before_down:
        scores = rand(total, topk, device=device) * 1.5
        scores[::5, 0] = 0.0
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
    weight_before_down=False,
    inputs=None,
):
    """Judge the iteration's outputs of the whole group over :data:`tests.cake_naive.DRAWS` draws of random inputs,
    and over stage 2's where an output asks for them (every rank runs the same judge on gathered values, so every rank
    takes the same draws).  ``weight_before_down``: the router weight before the down projection (the naive layer's
    form, :func:`_random_data`'s weights).  ``inputs()`` makes one draw's global inputs from the current random state
    (default :func:`_random_data`'s).  Returns the verdicts, the alternatives' verdicts and the failures."""
    experts = len(counts) * local_experts
    device = iteration.x.device
    alt = TierJudge()
    naive_values = {}
    mx = None if mxfp8_global is None else dict(forward=mxfp8_global[0], backward=mxfp8_global[1])

    def run(j, _seed):
        if inputs is None:
            data = _random_data(counts, hidden, topk, experts, device, weight_before_down)
        else:
            data = inputs()
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
                sections=counts, mx=mx, macro=None if fp32_wgrad else macro, weight_before_down=weight_before_down,
            )  # fmt: skip
        oracle = {n: v for n, v in zip(("y", "dx", "d_scores"), (refs[naive.ORACLE][k] for k in ("y", "dx", "d_scores")))}
        oracle.update(zip(RESULT_NAMES[3:], refs[naive.ORACLE]["dw"]))
        r = refs[naive.NAIVE]
        naive_values[j] = dict(y=r["y"], dx=r["dx"], d_scores=r["d_scores"], **dict(zip(RESULT_NAMES[3:], r["dw"])))
        if "alt_dw" in r:
            for name, value in zip(RESULT_NAMES[3:6], r["alt_dw"], strict=True):
                alt.add(name, j, outputs[name], oracle[name], {"bf16": value})
        if "alt_d_scores" in r:
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
    ``recompute_router_weights`` is passed to the recompute as its
    ``router_weights``; by default the ``"before_down"`` recompute takes the
    step's scores and the ``"after_down"`` call has no such argument.
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
        recompute_router_weights=None,
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
        self.recompute_router_weights = recompute_router_weights
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
            # The "before_down" recompute takes the router weights; the "after_down" call keeps the form without them
            # unless the iteration passes some (they are checked and not read).
            router = self.recompute_router_weights
            if router is None and self.functional.router_weight_position == "before_down":
                router = self.scores
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
                **({} if router is None else dict(router_weights=router)),
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
            all_x = randn(tokens * ep, hidden, **options) * 0.125
            all_dy = randn_like(all_x) * 0.125
            all_scores = rand(tokens * ep, topk, device=device) + 0.1
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
            x = randn(total, hidden, **options) * 0.125
            dy = randn_like(x) * 0.125
            scores = rand(total, topk, device=device) + 0.1
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

        def iteration(recompute, save_context=True, recompute_router_weights=None):
            return TrainingIteration(
                config, workspace, x, ids, scores, dy, weights, functional=functional, swiglu_limit=limit,
                mxfp8_weights=mxfp8_weights if mxfp8 else None, recompute=recompute, accumulators=accumulators,
                save_context=save_context, recompute_router_weights=recompute_router_weights,
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
            # "after_down" checks the recompute's router weights and does not read them, so the call form of
            # "before_down" serves both positions: router weights of another layer give the same bytes.
            other = torch.full_like(scores, OTHER_LAYER_WEIGHT)
            weighted_call = _snapshot(iteration(True, recompute_router_weights=other).run())
            differs = [n for n, a, b in zip(RESULT_NAMES, outputs, weighted_call, strict=True) if not torch.equal(a, b)]
            assert not differs, (rank, case["name"], "recompute with router weights", differs)
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


#: The cases of :func:`requantized_training`.
REQUANTIZED_CASES = (
    dict(name="mxfp8-requantized"),
    dict(name="mxfp8-requantized-fp32-wgrad", fp32_wgrad=True),
    dict(
        name="mxfp8-requantized-clamped-recompute",
        swiglu_limit=MXFP8_LIMIT,
        recompute=True,
    ),
)
#: The outputs ``mxfp8_transposed="requantized"`` changes: the routed weight gradients.
REQUANTIZED_CHANGES = ("dw_routed_gate", "dw_routed_up", "dw_routed_down")


def requantized_training():
    """``mxfp8_transposed="requantized"`` over unequal ranks (several macrobatches per rank, so the backward replays):
    the adapter refuses BF16 routed weights before any launch; per case (:data:`REQUANTIZED_CASES`) five eager launches
    and five graph replays give the same bytes, the checkpointed step (recompute) the saved-context step's bytes, and
    every output but :data:`REQUANTIZED_CHANGES` the default adapter's bytes on the same inputs.  Returns per case the
    routed weight gradients that differ from the default adapter's on this rank."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    local_experts, topk = LAYOUTS[ep]
    hidden, intermediate = 512, 256
    counts = ([513, 0, 1031, 256] * ep)[:ep] if ep > 1 else [1031]
    shared, _, routed = _weights(ep, local_experts, hidden, intermediate, device)
    weights = (*shared, *routed)
    mxfp8_weights = quantize_mok_mxfp8_weights(*routed)
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
    torch.manual_seed(8291)
    data = _random_data(counts, hidden, topk, ep * local_experts, device)
    start = sum(counts[:rank])
    section = slice(start, start + counts[rank])
    x, ids, scores, dy = (
        data[k][section].clone() for k in ("x", "expert_ids", "scores", "d_output")
    )
    records = []
    for case in REQUANTIZED_CASES:
        limit = case.get("swiglu_limit")
        fp32_wgrad = case.get("fp32_wgrad", False)
        adapters = {
            mode: prepare_mok_bf16(
                ep_size=ep,
                local_experts=local_experts,
                topk=topk,
                clamped_swiglu=limit is not None,
                fp32_wgrad=fp32_wgrad,
                mxfp8_transposed=mode,
            )  # fmt: skip
            for mode in api.MXFP8_TRANSPOSED_MODES
        }
        for functional in adapters.values():
            functional.prepare(limit, True)
            functional.prepare_recompute(limit, True)
        requantized = adapters["requantized"]
        with pytest.raises(ValueError, match="runs MXFP8 routed weights only"):
            requantized.prepare(limit)

        def iteration(mode, recompute=False, mxfp8=True):
            accumulators = (
                [torch.zeros(w.shape, device=device) for w in weights]
                if fp32_wgrad
                else None
            )
            return TrainingIteration(
                config, workspace, x, ids, scores, dy, weights, functional=adapters[mode], swiglu_limit=limit,
                mxfp8_weights=mxfp8_weights if mxfp8 else None, recompute=recompute, accumulators=accumulators,
            )  # fmt: skip

        if case is REQUANTIZED_CASES[0]:
            with pytest.raises(ValueError, match="runs MXFP8 routed weights only"):
                iteration("requantized", mxfp8=False).run()
            dist.barrier()
        default = _snapshot(iteration("source").run())
        saved_context = iteration("requantized")
        outputs = _snapshot(saved_context.run())
        _contract(outputs, counts[rank], hidden, topk)
        again = [_snapshot(saved_context.run()) for _ in range(REPEATS - 1)]
        assert all(_same(outputs, other) for other in again), (
            rank,
            case["name"],
            "eager repeat",
        )
        dist.barrier()
        saved_context.capture()
        replays = [_snapshot(saved_context.run()) for _ in range(REPEATS)]
        assert all(_same(outputs, other) for other in replays), (
            rank,
            case["name"],
            "graph replay",
        )
        dist.barrier()
        if case.get("recompute"):
            checkpointed = _snapshot(iteration("requantized", recompute=True).run())
            differs = [
                n
                for n, a, b in zip(RESULT_NAMES, outputs, checkpointed, strict=True)
                if not torch.equal(a, b)
            ]
            assert not differs, (rank, case["name"], "recompute", differs)
        differs = [
            n
            for n, a, b in zip(RESULT_NAMES, outputs, default, strict=True)
            if not torch.equal(a, b)
        ]
        assert set(differs) <= set(REQUANTIZED_CHANGES), (
            rank,
            case["name"],
            "against the default adapter",
            differs,
        )
        records.append(dict(case=case["name"], counts=counts, routed_dw_differ=differs))
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
    x = randn(n, h, generator=generator, **options) * 0.125
    dy = randn(n, h, generator=generator, **options) * 0.125
    scores = rand(n, k, device=device, generator=generator) + 0.1
    shared = [randn(h, h, generator=generator, **options) / h**0.5 for _ in range(3)]
    routed = [
        randn(le, h, h, generator=generator, **options) / h**0.5 for _ in range(3)
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


#: (local experts, top-k) of the router-weight-before-down worker per EP size when the item names no layout: the
#: diagnostic layouts.
BEFORE_DOWN_DIST_LAYOUTS = {1: (4, 2), 4: (4, 2)}
#: Source rows per rank of the router-weight-before-down worker's steps, all on one workspace whose source capacity
#: covers the largest: padded staging (1031, 257 and 513 rows) and unpadded staging (768, a multiple of 256), several
#: macrobatches.  EP1 runs 1031, 768 and 257.
BEFORE_DOWN_COUNTS = ((1031, 768, 257, 513), (768, 257, 513, 1031), (257, 513, 1031, 768))
#: The last step of the router-weight-before-down worker in a group of several ranks (its first ranks' counts): rank 0
#: holds no token (and at EP8 rank 5 too).
BEFORE_DOWN_EMPTY_COUNTS = (0, 1031, 768, 257, 513, 0, 1024, 256)
#: Hidden and intermediate widths of the router-weight-before-down workers.
BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE = 512, 256
#: The requester's clamp limit: gate and up of the router weight before the down projection clamp at +-10.
R8_LIMIT = 10.0
#: The expert :func:`_r8_weights` pins for the steps with random routes (token 1's first expert).
R8_PINNED = 0
#: Seed of the signs of the pinned token's ``x`` row (:func:`_r8_signs`).
R8_SIGN_SEED = 8501
#: Source rows per rank of the placement, full-skew and run-to-run steps.
SKEW_TOKENS = 1024
#: Rank offsets of the routes of a token spread over the group (placement): six distinct ranks at EP8, all four ranks
#: as 2 + 2 + 1 + 1 at EP4.
SPREAD = {4: (0, 0, 1, 1, 2, 3), 8: (0, 1, 2, 3, 4, 5)}
#: The text of the error the target rank of an overflowing schedule raises (:func:`before_down_overflow`).
SCHEDULE_TRAPPED = "the overflowing schedule trapped"


def before_down_steps(ep):
    """Source rows per rank of each step of :func:`before_down_training` in a group of ``ep`` ranks: the rotations of
    :data:`BEFORE_DOWN_COUNTS`, then with several ranks :data:`BEFORE_DOWN_EMPTY_COUNTS`."""
    steps = [(list(counts) * ep)[:ep] for counts in BEFORE_DOWN_COUNTS]
    if ep > 1:
        steps.append(list(BEFORE_DOWN_EMPTY_COUNTS[:ep]))
    return steps


def _source_rows(count, topk):
    """The physical source rows a workspace reserves for at most ``count`` rows per rank (``create_mok_bf16_workspace``:
    at least 512, aligned to the 256-row tile and whole metadata chunks)."""
    alignment = math.lcm(256, 512 // math.gcd(topk, 512))
    return -(-max(512, count) // alignment) * alignment


def _r8_signs(hidden, device):
    """The ``+-1`` row of the pinned token's ``x`` (fixed seed)."""
    generator = torch.Generator(device=device).manual_seed(R8_SIGN_SEED)
    return torch.where(randn(hidden, device=device, generator=generator) < 0, -1.0, 1.0)


def _r8_weights(ep, local_experts, hidden, intermediate, device, pinned, limit=R8_LIMIT, seed=32479):
    """The requester's weights (the form of :func:`tests.experimental._mok_reference.production_inputs`; identical on
    every rank): gate and up ``randn * 4 / sqrt(hidden)`` (a small fraction of gate and up passes the limit), down
    ``randn / sqrt(intermediate)``, shared and routed alike.  The group's expert ``pinned`` has gate row 0 and up rows 0
    and 1 equal to ``+-signs * limit / hidden`` (:func:`_r8_signs`; ``limit / hidden`` is a BF16 value), so the pinned
    token of :func:`_r8_data` has gate exactly ``limit`` and up exactly ``+limit`` and ``-limit`` there.  Returns the
    shared weights, the group's routed weights and this rank's slice of them."""
    torch.manual_seed(seed)
    gate_up, down = 4.0 / hidden**0.5, 1.0 / intermediate**0.5
    total = ep * local_experts

    def draw(*shape, scale):
        return (randn(*shape, device=device) * scale).bfloat16()

    shared = [
        draw(intermediate, hidden, scale=gate_up),
        draw(intermediate, hidden, scale=gate_up),
        draw(hidden, intermediate, scale=down),
    ]
    global_routed = [
        draw(total, intermediate, hidden, scale=gate_up),
        draw(total, intermediate, hidden, scale=gate_up),
        draw(total, hidden, intermediate, scale=down),
    ]
    step = torch.tensor(limit / hidden).bfloat16().float()
    assert step.item() * hidden == limit, (limit, hidden)
    row = (_r8_signs(hidden, device) * step).bfloat16()
    global_routed[0][pinned, 0] = row
    global_routed[1][pinned, 0] = row
    global_routed[1][pinned, 1] = -row
    rank = dist.get_rank()
    routed = [w[rank * local_experts : (rank + 1) * local_experts].contiguous() for w in global_routed]
    return shared, global_routed, routed


def _r8_scores(tokens, experts, device):
    """The requester's router scores ``sqrt(softplus(randn))`` (current random state)."""
    return torch.nn.functional.softplus(randn(tokens, experts, device=device)).sqrt()


def _r8_data(counts, hidden, topk, experts, device, *, ids=None, pinned=R8_PINNED):
    """One draw of the requester's inputs over the group's tokens from the current random state (identical on every
    rank; the form of :func:`tests.experimental._mok_reference.production_inputs`): unit-RMS BF16 ``x``; the top
    ``topk`` router scores per token (:func:`_r8_scores`) normalized to sum 1.5 (FP32, each in [0, 1.5]), every 8th
    token's smallest weight exactly 0 (the others rescaled to sum 1.5) and token 2's weights ``(1.5, 0, ...)``; BF16
    ``randn`` ``dy``.  The expert ids are those top ``topk`` experts, or ``ids`` (a placement; the weights in the top
    scores' order).  Token 1 is pinned: its ``x`` row is :func:`_r8_signs` and its first expert ``pinned``, the expert
    :func:`_r8_weights` pinned (moved to the first slot, or replacing its first expert)."""
    total = sum(counts)
    x = randn(total, hidden, device=device)
    x = x / x.square().mean(-1, keepdim=True).sqrt()
    top, chosen = _r8_scores(total, experts, device).topk(topk, dim=-1)
    weights = top / top.sum(-1, keepdim=True) * 1.5
    zero = torch.arange(total, device=device) % 8 == 7
    weights[zero, topk - 1] = 0.0
    weights[zero] = weights[zero] / weights[zero].sum(-1, keepdim=True) * 1.5
    if total > 2:
        weights[2] = 0.0
        weights[2, 0] = 1.5
    ids = (chosen if ids is None else ids).clone()
    if total > 1:
        x[1] = _r8_signs(hidden, device)
        held = (ids[1] == pinned).nonzero()
        if held.numel():
            ids[1, int(held[0, 0])] = int(ids[1, 0])
        ids[1, 0] = pinned
    dy = randn(total, hidden, device=device).bfloat16()
    return dict(x=x.bfloat16(), d_output=dy, expert_ids=ids.contiguous(), scores=weights.float().contiguous())


#: A router weight no step uses: what another layer leaves in the shared workspace's router-weight buffer.
OTHER_LAYER_WEIGHT = 0.75


def _other_layer(step):
    """Overwrite the workspace's router-weight buffer, as another layer's forward on the shared workspace does; the
    launch that follows must stage its own router weights.  Every launch ends with a peer barrier, so no peer still
    reads this rank's buffer."""
    if step.get("other_layer"):
        step["workspace"].storage.router_weight_buffer.fill_(OTHER_LAYER_WEIGHT)


class _CheckpointedStep(torch.autograd.Function):
    """Test-local autograd function around one checkpointed router-weight-before-down step (the form a caller wraps
    the adapter in; the package ships none): forward without context, backward rebuilds the context with
    ``recompute_forward_context`` (the router weights included) and runs backward.  ``x`` and the router weights are
    differentiable; the weight gradients go to the caller's FP32 accumulators.  With ``other_layer`` the workspace's
    router-weight buffer holds another value before the recompute and before the backward (:func:`_other_layer`)."""

    @staticmethod
    def forward(ctx, x, scores, step):
        f = step["functional"]
        y, context = f.forward(
            step["config"], step["workspace"], step["schedule"], x, scores, *step["weights"],
            swiglu_limit=step["limit"], save_context=False,
        )  # fmt: skip
        assert context is None
        ctx.step = step
        ctx.save_for_backward(x, scores)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, scores = ctx.saved_tensors
        step = ctx.step
        f, w = step["functional"], step["weights"]
        _other_layer(step)
        context = f.recompute_forward_context(
            step["config"], step["workspace"], step["schedule"], x, w[0], w[1], w[3], w[4],
            swiglu_limit=step["limit"], router_weights=scores,
        )  # fmt: skip
        _other_layer(step)
        for accumulator in step["accumulators"]:
            accumulator.zero_()
        gradients = f.backward(
            step["config"], step["workspace"], step["schedule"], context, dy.contiguous(), x, scores, *w,
            swiglu_limit=step["limit"], weight_grad_accumulators=step["accumulators"],
        )  # fmt: skip
        return gradients[0], gradients[1], None


def _unplanned_steps(step, x, scores, dy, ids, local_experts):
    """The adapter's unplanned launches (``_forward_native``, ``_recompute_native``, ``_backward_native``): a copy of
    a fresh schedule without its expert layout, which every launcher then derives itself.  The context step, then the
    checkpointed step (context-free forward, recompute, backward), each ``(y, dx, d_scores, *weight gradients)``."""
    f, w, accumulators = step["functional"], step["weights"], step["accumulators"]
    config, workspace, limit = step["config"], step["workspace"], step["limit"]
    schedule = dataclasses.replace(
        f.build_schedule(workspace, config, ids, num_local_experts=local_experts), layout=None
    )
    out = []
    for save_context in (True, False):
        for accumulator in accumulators:
            accumulator.zero_()
        y, context = f.forward(
            config, workspace, schedule, x, scores, *w, swiglu_limit=limit, save_context=save_context
        )
        if not save_context:
            context = f.recompute_forward_context(
                config, workspace, schedule, x, w[0], w[1], w[3], w[4], swiglu_limit=limit, router_weights=scores,
            )  # fmt: skip
        gradients = f.backward(
            config, workspace, schedule, context, dy, x, scores, *w, swiglu_limit=limit,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        out.append(_snapshot((y, *gradients)))
    return out


def before_down_training(local_experts=None):
    """The router weight before the down projection through the adapter (clamped SwiGLU, FP32 weight gradients from
    zeroed accumulators) on one workspace over the source counts of :func:`before_down_steps` (padded and unpadded
    staging, several macrobatches, a new schedule per count, an empty source rank in groups of several ranks): per
    count the contract, repeatability eager and under CUDA Graph replay, recompute and the context-free forward with
    the saved context's bytes, the unplanned launches with the same bytes, a test-local autograd function around the
    checkpointed step with the same bytes while another router weight sits in the workspace between its launches; no
    count after the first loads a kernel; the first count's graph replays its bytes after the later counts ran;
    another communication-SM split per rank gives the same bytes; every count judged.  ``local_experts`` selects the
    layout (default :data:`BEFORE_DOWN_DIST_LAYOUTS`); at the 384-expert layouts (top-6) the steps take the
    requester's inputs (:func:`_r8_data`, clamp limit :data:`R8_LIMIT`).  The workspace has the requester's
    no-overflow multiplier at its physical source rows."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    if local_experts is None:
        local_experts = BEFORE_DOWN_DIST_LAYOUTS[ep][0]
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    total_experts = ep * local_experts
    requester = topk == 6
    hidden, intermediate = BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE
    limit = R8_LIMIT if requester else LIMIT
    steps = before_down_steps(ep)
    if requester:
        shared, global_routed, routed = _r8_weights(ep, local_experts, hidden, intermediate, device, R8_PINNED)
    else:
        shared, global_routed, routed = _weights(ep, local_experts, hidden, intermediate, device)
    weights = (*shared, *routed)

    def inputs(counts):
        if requester:
            return _r8_data(counts, hidden, topk, total_experts, device)
        return _random_data(counts, hidden, topk, total_experts, device, weight_before_down=True)

    source_capacity = max(max(counts) for counts in steps)
    multiplier = bound_multiplier(_source_rows(source_capacity, topk), topk, ep, local_experts)
    functional = prepare_mok_bf16(
        ep_size=ep, local_experts=local_experts, topk=topk, clamped_swiglu=True, fp32_wgrad=True,
        router_weight_position="before_down",
    )  # fmt: skip
    functional.prepare(limit, save_context=False)
    functional.prepare_recompute(limit)
    config, workspace = create_mok_bf16_workspace(
        group=dist.group.WORLD,
        device=device,
        num_local_tokens=steps[0][rank],
        source_capacity=source_capacity,
        hidden_size=hidden,
        topk=topk,
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=multiplier,
    )
    assert workspace.initial_source_counts == tuple(steps[0])
    assert workspace.source_capacity == _source_rows(source_capacity, topk)
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    records, first, loads = [], None, None
    for index, counts in enumerate(steps):
        torch.manual_seed(8293 + index)
        data = inputs(counts)
        start = sum(counts[:rank])
        section = slice(start, start + counts[rank])
        x, ids, scores, dy = (data[k][section].clone() for k in ("x", "expert_ids", "scores", "d_output"))
        case = "counts-" + "-".join(map(str, counts))

        def iteration(recompute, save_context=True, x=x, ids=ids, scores=scores, dy=dy, cfg=config):
            return TrainingIteration(
                cfg, workspace, x, ids, scores, dy, weights, functional=functional, swiglu_limit=limit,
                recompute=recompute, accumulators=accumulators, save_context=save_context,
            )  # fmt: skip

        saved_context = iteration(False)
        outputs = _snapshot(saved_context.run())
        _contract(outputs, counts[rank], hidden, topk)
        again = [_snapshot(saved_context.run()) for _ in range(REPEATS - 1)]
        assert all(_same(outputs, other) for other in again), (rank, case, "eager repeat")
        # No bit depends on the communication-SM split (the dispatch's communication CTAs gather the router weights
        # into the row ring), also with the ranks on different splits in one step.
        split = dataclasses.replace(
            config, fwd_num_comm_sms=4 + 2 * (rank % 2), bwd_num_comm_sms=6 - 2 * (rank % 2), imbalance_comm_sms=None
        )
        for recompute in (False, True):
            values = _snapshot(iteration(recompute, cfg=split).run())
            differs = [n for n, a, b in zip(RESULT_NAMES, outputs, values, strict=True) if not torch.equal(a, b)]
            assert not differs, (rank, case, "communication split", recompute, differs)
        dist.barrier()
        for recompute, save_context in ((True, True), (True, False)):
            checkpointed = _snapshot(iteration(recompute, save_context).run())
            differs = [n for n, a, b in zip(RESULT_NAMES, outputs, checkpointed, strict=True) if not torch.equal(a, b)]
            assert not differs, (rank, case, "recompute" if save_context else "context-free", differs)
        dist.barrier()
        step = dict(
            functional=functional, config=config, workspace=workspace, weights=weights, limit=limit,
            accumulators=accumulators, other_layer=True, schedule=functional.build_schedule(
                workspace, config, ids, num_local_experts=local_experts
            ),
        )  # fmt: skip
        for name, values in zip(
            ("unplanned", "unplanned context-free"), _unplanned_steps(step, x, scores, dy, ids, local_experts),
            strict=True,
        ):
            differs = [n for n, a, b in zip(RESULT_NAMES, outputs, values, strict=True) if not torch.equal(a, b)]
            assert not differs, (rank, case, name, differs)
        dist.barrier()
        leaf_x, leaf_scores = x.detach().clone().requires_grad_(), scores.detach().clone().requires_grad_()
        y = _CheckpointedStep.apply(leaf_x, leaf_scores, step)
        y.backward(dy)
        wrapped = _snapshot((y, leaf_x.grad, leaf_scores.grad, *(accumulators[i] for i in (3, 4, 5, 0, 1, 2))))
        differs = [n for n, a, b in zip(RESULT_NAMES, outputs, wrapped, strict=True) if not torch.equal(a, b)]
        assert not differs, (rank, case, "autograd function", differs)
        dist.barrier()
        saved_context.capture()
        replays = [_snapshot(saved_context.run()) for _ in range(REPEATS)]
        assert all(_same(outputs, other) for other in replays), (rank, case, "graph replay")
        dist.barrier()
        # A changed source count builds a new schedule and launch plan on the workspace; it loads no kernel.
        misses = jit._load.cache_info().misses
        if loads is None:
            loads = misses
        assert misses == loads, (rank, case, "kernel loads", loads, misses)
        record = dict(case=f"bf16-clamped-before-down-{case}", counts=counts)
        record["judged"] = judge(
            saved_context, counts, shared, global_routed, topk=topk, local_experts=local_experts, hidden=hidden,
            seed=4400 + index, limit=limit, fp32_wgrad=True, weight_before_down=True,
            inputs=(lambda counts=counts: inputs(counts)) if requester else None,
        )  # fmt: skip
        records.append(record)
        if first is None:
            first = (saved_context, data, counts, outputs)
        else:
            del saved_context
        torch.cuda.synchronize()
        dist.barrier()
    saved_context, data, counts, outputs = first
    _load(saved_context, data, counts)
    assert _same(outputs, _snapshot(saved_context.run())), (rank, "the first count's graph after the later counts")
    assert jit._load.cache_info().misses == loads
    del saved_context, first
    torch.cuda.synchronize()
    dist.barrier()
    return records


#: The fields of a forward context, shared and routed.
CONTEXT_FIELDS = ("x_routed", "gate_shared", "gate_routed", "up_shared", "up_routed", "hidden_shared", "hidden_routed")


def _digests(values):
    """SHA-256 of the bytes of each of ``(y, dx, d_scores, *weight gradients)``, by name."""
    found = {}
    for name, value in zip(RESULT_NAMES, values, strict=True):
        data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        found[name] = hashlib.sha256(data.tobytes()).hexdigest()
    return found


def _section(data, counts):
    """This rank's ``(x, expert ids, scores, dy)`` of the global ``data``."""
    rank = dist.get_rank()
    start = sum(counts[:rank])
    section = slice(start, start + counts[rank])
    return tuple(data[key][section].clone() for key in ("x", "expert_ids", "scores", "d_output"))


def _before_down_group(local_experts, counts, macro=768, bound=True):
    """The adapter of the router weight before the down projection (clamped SwiGLU, FP32 weight gradients) at the
    layout ``(EP, local_experts)`` and a workspace for the source rows ``counts`` (physical rows :func:`_source_rows` of
    the largest), minibatch 256, macrobatch ``macro`` (None: one resident macrobatch, the schedule capacity), at the
    requester's no-overflow multiplier for those rows (host check: the capacity holds the worst case
    ``min(topk, E) * T * EP + E * 255``) or, without ``bound``, the package's default multiplier.  Returns the
    adapter, the config and the workspace."""
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    rows = _source_rows(max(counts), topk)
    extra = {}
    if bound:
        extra["schedule_capacity_multiplier"] = bound_multiplier(rows, topk, ep, local_experts)
        capacity = schedule_capacity_for(rows, topk, ep, extra["schedule_capacity_multiplier"])
        assert capacity >= min(topk, local_experts) * rows * ep + local_experts * 255, (capacity, rows)
        if macro is None:
            macro = capacity
    functional = prepare_mok_bf16(
        ep_size=ep, local_experts=local_experts, topk=topk, clamped_swiglu=True, fp32_wgrad=True,
        router_weight_position="before_down",
    )  # fmt: skip
    functional.prepare(R8_LIMIT, save_context=False)
    functional.prepare_recompute(R8_LIMIT)
    config, workspace = create_mok_bf16_workspace(
        group=dist.group.WORLD, device=device, num_local_tokens=counts[rank], source_capacity=rows,
        hidden_size=BEFORE_DOWN_HIDDEN, topk=topk, fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=256,
        macrobatch_size=macro, **extra,
    )  # fmt: skip
    assert workspace.source_capacity == rows and workspace.initial_source_counts == tuple(counts)
    if bound:
        assert workspace.storage.schedule_capacity == capacity
    return functional, config, workspace


def _recompute_context(
    functional, config, workspace, x, ids, scores, dy, weights, accumulators, local_experts, outputs
):  # fmt: skip
    """A new schedule; the forward with its saved context, then ``recompute_forward_context`` (the router weights
    included): every shared field of the two contexts byte-equal whole, every routed field byte-equal on the
    scheduled rows (``peer_rank >= 0``) of the resident macrobatch (the schedule's first ``min(num_tokens,
    macrobatch_size)`` rows); backward from either context gives the bytes of ``outputs``.  Returns, per routed field,
    whether its padding rows there are equal too (recorded, not required)."""
    rank, limit = dist.get_rank(), R8_LIMIT
    schedule = functional.build_schedule(workspace, config, ids, num_local_experts=local_experts)
    y, context = functional.forward(config, workspace, schedule, x, scores, *weights, swiglu_limit=limit)
    saved = _snapshot(getattr(context, name) for name in CONTEXT_FIELDS)

    def gradients(ctx):
        for accumulator in accumulators:
            accumulator.zero_()
        values = functional.backward(
            config, workspace, schedule, ctx, dy, x, scores, *weights, swiglu_limit=limit,
            weight_grad_accumulators=accumulators,
        )  # fmt: skip
        return _snapshot((y, *values))

    from_saved = gradients(context)
    rebuilt = functional.recompute_forward_context(
        config, workspace, schedule, x, weights[0], weights[1], weights[3], weights[4], swiglu_limit=limit,
        router_weights=scores,
    )  # fmt: skip
    again = _snapshot(getattr(rebuilt, name) for name in CONTEXT_FIELDS)
    rows = min(int(schedule.num_tokens.item()), config.macrobatch_size)
    scheduled = schedule.peer_rank[:rows] >= 0
    padding = {}
    for name, a, b in zip(CONTEXT_FIELDS, saved, again, strict=True):
        if name.endswith("_shared"):
            assert torch.equal(a, b), (rank, name, "recomputed context")
        else:
            assert torch.equal(a[:rows][scheduled], b[:rows][scheduled]), (rank, name, "recomputed context")
            padding[name] = torch.equal(a[:rows][~scheduled], b[:rows][~scheduled])
    from_rebuilt = gradients(rebuilt)
    for label, values in (("saved context", from_saved), ("recomputed context", from_rebuilt)):
        differs = [n for n, a, b in zip(RESULT_NAMES, outputs, values, strict=True) if not torch.equal(a, b)]
        assert not differs, (rank, "backward from the " + label, differs)
    return padding


def _before_down_case(name, functional, config, workspace, counts, ids, local_experts, *, seed, checks=None):
    """One router-weight-before-down step of the requester's inputs over the global expert ids ``ids`` (weights
    :func:`_r8_weights` pinned at token 1's first expert, inputs :func:`_r8_data`): the contract, every route held
    once in the group's schedules (:func:`segments`), ``checks(outputs, schedule)`` of the case, five eager launches
    with the same bytes, the recompute's context against the saved one (:func:`_recompute_context`), five CUDA Graph
    replays with the eager bytes; judged over the draws (new inputs on the same ids).  Returns its record."""
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = ids.shape[1]
    hidden, intermediate = BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE
    pinned = int(ids[1, 0])
    shared, global_routed, routed = _r8_weights(ep, local_experts, hidden, intermediate, device, pinned)
    weights = (*shared, *routed)
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]

    def inputs():
        return _r8_data(counts, hidden, topk, ep * local_experts, device, ids=ids, pinned=pinned)

    torch.manual_seed(seed)
    x, local_ids, scores, dy = _section(inputs(), counts)
    iteration = TrainingIteration(
        config, workspace, x, local_ids, scores, dy, weights, functional=functional, swiglu_limit=R8_LIMIT,
        accumulators=accumulators,
    )  # fmt: skip
    outputs = _snapshot(iteration.run())
    _contract(outputs, counts[rank], hidden, topk)
    segments(iteration.schedule, counts, local_experts, topk)
    record = dict(
        case=f"bf16-clamped-before-down-{name}", counts=counts, num_tokens=int(iteration.schedule.num_tokens.item()),
        capacity=workspace.storage.schedule_capacity, macrobatch=config.macrobatch_size,
    )  # fmt: skip
    if checks is not None:
        checks(outputs, iteration.schedule)
    again = [_snapshot(iteration.run()) for _ in range(REPEATS - 1)]
    assert all(_same(outputs, other) for other in again), (rank, name, "eager repeat")
    dist.barrier()
    record["padding_rows_equal"] = _recompute_context(
        functional, config, workspace, x, local_ids, scores, dy, weights, accumulators, local_experts, outputs
    )
    dist.barrier()
    iteration.capture()
    replays = [_snapshot(iteration.run()) for _ in range(REPEATS)]
    assert all(_same(outputs, other) for other in replays), (rank, name, "graph replay")
    dist.barrier()
    record["judged"] = judge(
        iteration, counts, shared, global_routed, topk=topk, local_experts=local_experts, hidden=hidden, seed=seed,
        limit=R8_LIMIT, fp32_wgrad=True, weight_before_down=True, inputs=inputs,
    )  # fmt: skip
    del iteration
    torch.cuda.synchronize()
    dist.barrier()
    return record


def _placement_ids(counts, ep, local_experts, topk, device):
    """Global expert ids of the placement mix (identical on every rank; the random part from the current random
    state), by the group's token ``t``: ``t % 3 == 0`` all ``topk`` experts on rank ``t % ep``; ``1`` the experts on the
    ranks ``t + SPREAD[ep]`` (six distinct ranks at EP8, 2 + 2 + 1 + 1 over the four ranks at EP4); ``2`` the top
    ``topk`` of the requester's random scores.  Route ``k``'s local expert is ``t + k * (local_experts // topk)``, so a
    token's experts on one rank are distinct."""
    total = sum(counts)
    t = torch.arange(total, device=device)[:, None]
    k = torch.arange(topk, device=device)[None, :]
    local = (t + k * (local_experts // topk)) % local_experts
    one_rank = (t % ep).expand(total, topk)
    spread = (t + torch.tensor(SPREAD[ep], device=device)[None, :]) % ep
    random = _r8_scores(total, ep * local_experts, device).topk(topk, dim=-1).indices
    kind = (t % 3).expand(total, topk)
    placed = torch.where(kind == 0, one_rank, spread) * local_experts + local
    return torch.where(kind == 2, random, placed).contiguous()


def _skew_ids(counts, local_experts, topk, target, device):
    """Global expert ids of a full skew: every route of the group to rank ``target``'s experts, with the route counts
    per expert that pad the most (every expert but the last ``1 (mod 256)`` routes, the last the rest).  The sorted
    list of the routes' experts is strided over the group's tokens (route ``k`` of token ``t`` is entry ``k * T + t``
    of ``T`` tokens); every expert holds fewer than ``T`` routes, so a token's experts are distinct.  Returns the ids
    and the padded rows of the target rank's schedule (each expert's routes rounded up to 256 rows)."""
    total = sum(counts)
    routes = total * topk
    per = 256 * ((routes // local_experts - 1) // 256 + 1) + 1
    sizes = [per] * (local_experts - 1) + [routes - per * (local_experts - 1)]
    assert 0 < sizes[-1] and max(sizes) < total, sizes
    experts = torch.arange(local_experts, device=device) + target * local_experts
    slots = experts.repeat_interleave(torch.tensor(sizes, device=device))
    ids = slots.view(topk, total).T.contiguous()
    assert bool((ids.sort(-1).values.diff(dim=-1) != 0).all()), "a token routed twice to one expert"
    return ids, sum(-(-n // 256) * 256 for n in sizes)


def before_down_placement(local_experts, macro):
    """The requester's inputs at a 384-expert layout over chosen placements, at the no-overflow multiplier and
    macrobatch ``macro`` (None: one resident macrobatch): (1) :func:`_placement_ids` over :data:`SKEW_TOKENS` source
    rows per rank (a token's experts on one rank, spread over the ranks, random); (2) rank 0 without tokens and no route
    to the last rank's experts (the top-6 of random scores over the other ranks' experts): the last rank's routed
    weight gradients and rank 0's shared weight gradients are exactly zero, rank 0's ``y``, ``dx`` and score gradients
    empty.  Each step as :func:`_before_down_case`."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    mixed = [SKEW_TOKENS] * ep
    sparse = [0] + [SKEW_TOKENS] * (ep - 1)
    functional, config, workspace = _before_down_group(local_experts, mixed, macro)
    torch.manual_seed(8521)
    mixed_ids = _placement_ids(mixed, ep, local_experts, topk, device)
    torch.manual_seed(8523)
    sparse_ids = _r8_scores(sum(sparse), (ep - 1) * local_experts, device).topk(topk, dim=-1).indices.contiguous()

    def sparse_checks(outputs, schedule):
        if rank == ep - 1:
            assert int(schedule.num_tokens.item()) == 0, (rank, "rows of a rank without routes")
            assert all(torch.count_nonzero(v).item() == 0 for v in outputs[3:6]), (rank, "routed dW without routes")
        if sparse[rank] == 0:
            assert all(torch.count_nonzero(v).item() == 0 for v in outputs[6:]), (rank, "shared dW without tokens")

    return [
        _before_down_case(
            "placement-mixed", functional, config, workspace, mixed, mixed_ids, local_experts, seed=4500
        ),
        _before_down_case(
            "placement-no-route-to-last-rank", functional, config, workspace, sparse, sparse_ids, local_experts,
            seed=4501, checks=sparse_checks,
        ),
    ]  # fmt: skip


def before_down_full_skew(local_experts, target, macro=768):
    """The requester's inputs at a 384-expert layout with every route of the group on rank ``target``'s experts,
    padded the most (:func:`_skew_ids`; :data:`SKEW_TOKENS` source rows per rank), at the no-overflow multiplier and
    macrobatch ``macro`` (None: one resident macrobatch, so the recompute's context is checked over the whole
    schedule): the target rank's schedule holds exactly the padded rows, the other ranks' none; as
    :func:`_before_down_case`."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    counts = [SKEW_TOKENS] * ep
    functional, config, workspace = _before_down_group(local_experts, counts, macro)
    ids, padded = _skew_ids(counts, local_experts, topk, target, device)

    def checks(outputs, schedule):
        assert int(schedule.num_tokens.item()) == (padded if rank == target else 0), (rank, padded)

    record = _before_down_case(
        f"full-skew-rank{target}", functional, config, workspace, counts, ids, local_experts, seed=4600 + target,
        checks=checks,
    )  # fmt: skip
    record["padded_rows"] = padded
    return [record]


def before_down_overflow(local_experts, target):
    """The full skew of :func:`before_down_full_skew` on a workspace at the package's default multiplier, whose
    schedule capacity is below the target rank's padded rows (checked on the host): the target rank's rows kernel
    traps, and the next CUDA call raises: one of the later launches or copies of ``build_schedule``, or the
    synchronization right after it (which one depends on timing); that error is raised again as
    :data:`SCHEDULE_TRAPPED`.  The other ranks go on to the forward and wait there for the target rank until the
    caller stops them.  A rank that gets past the step returns the digests of its outputs."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    counts = [SKEW_TOKENS] * ep
    functional, config, workspace = _before_down_group(local_experts, counts, bound=False)
    ids, padded = _skew_ids(counts, local_experts, topk, target, device)
    assert workspace.storage.schedule_capacity < padded, (workspace.storage.schedule_capacity, padded)
    pinned = int(ids[1, 0])
    shared, _, routed = _r8_weights(ep, local_experts, BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE, device, pinned)
    weights = (*shared, *routed)
    torch.manual_seed(4700 + target)
    data = _r8_data(counts, BEFORE_DOWN_HIDDEN, topk, ep * local_experts, device, ids=ids, pinned=pinned)
    x, local_ids, scores, dy = _section(data, counts)
    try:
        schedule = functional.build_schedule(workspace, config, local_ids, num_local_experts=local_experts)
        torch.cuda.synchronize()
    except RuntimeError as error:
        raise RuntimeError(f"{SCHEDULE_TRAPPED} on rank {rank}: {error}") from error
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    y, context = functional.forward(config, workspace, schedule, x, scores, *weights, swiglu_limit=R8_LIMIT)
    gradients = functional.backward(
        config, workspace, schedule, context, dy, x, scores, *weights, swiglu_limit=R8_LIMIT,
        weight_grad_accumulators=accumulators,
    )  # fmt: skip
    return _digests(_snapshot((y, *gradients)))


def before_down_small_step(local_experts):
    """One router-weight-before-down step of the requester's inputs (512 source rows per rank, random routes) at the
    no-overflow multiplier: the contract and five eager launches with the same bytes."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    counts = [512] * ep
    functional, config, workspace = _before_down_group(local_experts, counts)
    shared, _, routed = _r8_weights(
        ep, local_experts, BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE, device, R8_PINNED
    )
    weights = (*shared, *routed)
    torch.manual_seed(8531)
    data = _r8_data(counts, BEFORE_DOWN_HIDDEN, topk, ep * local_experts, device)
    x, ids, scores, dy = _section(data, counts)
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    iteration = TrainingIteration(
        config, workspace, x, ids, scores, dy, weights, functional=functional, swiglu_limit=R8_LIMIT,
        accumulators=accumulators,
    )  # fmt: skip
    outputs = _snapshot(iteration.run())
    _contract(outputs, counts[rank], BEFORE_DOWN_HIDDEN, topk)
    again = [_snapshot(iteration.run()) for _ in range(REPEATS - 1)]
    assert all(_same(outputs, other) for other in again), (rank, "eager repeat")
    del iteration
    torch.cuda.synchronize()
    dist.barrier()
    return dict(case="small-step", num_tokens=int(outputs[0].shape[0]))


def before_down_digests(local_experts):
    """SHA-256 of every output of this rank (:func:`_digests`) of two steps of the requester's inputs at a 384-expert
    layout and the no-overflow multiplier, on a new workspace: the placement mix of :func:`before_down_placement` and
    the full skew to rank 0 of :func:`before_down_full_skew`; each step launched twice in this process (a new schedule
    each) with the same bytes."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    counts = [SKEW_TOKENS] * ep
    functional, config, workspace = _before_down_group(local_experts, counts)
    torch.manual_seed(8521)
    steps = (
        ("placement-mixed", _placement_ids(counts, ep, local_experts, topk, device)),
        ("full-skew-rank0", _skew_ids(counts, local_experts, topk, 0, device)[0]),
    )
    out = {}
    for index, (name, ids) in enumerate(steps):
        pinned = int(ids[1, 0])
        shared, _, routed = _r8_weights(
            ep, local_experts, BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE, device, pinned
        )
        weights = (*shared, *routed)
        torch.manual_seed(8541 + index)
        data = _r8_data(counts, BEFORE_DOWN_HIDDEN, topk, ep * local_experts, device, ids=ids, pinned=pinned)
        x, local_ids, scores, dy = _section(data, counts)
        accumulators = [torch.zeros(w.shape, device=device) for w in weights]
        launches = []
        for _ in range(2):
            iteration = TrainingIteration(
                config, workspace, x, local_ids, scores, dy, weights, functional=functional, swiglu_limit=R8_LIMIT,
                accumulators=accumulators,
            )  # fmt: skip
            launches.append(_digests(_snapshot(iteration.run())))
            dist.barrier()
        assert launches[0] == launches[1], (rank, name, "in-process repeat")
        out[name] = launches[0]
    torch.cuda.synchronize()
    dist.barrier()
    return out


#: Source rows per rank of :func:`before_down_source_split` by group size: unaligned counts, an empty rank, and a group
#: total whose full skew to the last rank's experts fits :func:`_skew_ids` (every expert's routes padded the most).
SOURCE_SPLIT_COUNTS = {4: (2277, 300, 0, 1517), 8: (2277, 300, 0, 1025, 64, 1, 426, 1)}
#: ``(name, rich_comm_sms(rank), source_split_threshold)`` of the forward source splits :func:`before_down_source_split`
#: runs beside the split off (all below the workspace's 8 forward communication SMs); ``held`` sets a threshold no rank
#: reaches.
SOURCE_SPLITS = (
    ("rich2", lambda rank: 2, 1.5),
    ("rich4-low", lambda rank: 4, 1.1),
    ("per-rank", lambda rank: 2 + 2 * (rank % 2), 1.25),
    ("held", lambda rank: 2, 64.0),
)


def source_rich(counts, threshold):
    """Per rank, whether the forward kernel's source split takes the rich count (host form of its rule): the rank's
    shared rows (its source rows rounded up to 256, at least 256) less 256 exceed ``threshold`` (in 1/256 steps, as
    the adapter rounds it) times the mean source rows of the group."""
    total, ep, units = sum(counts), len(counts), round(threshold * 256)
    return [total > 0 and (max(256, -(-n // 256) * 256) - 256) * ep * 256 > units * total for n in counts]


def before_down_source_split(local_experts):
    """The forward source split at a 384-expert layout on the requester's inputs (weights pinned at expert
    :data:`R8_PINNED`): source rows :data:`SOURCE_SPLIT_COUNTS` (one or two source-rich ranks, an empty rank), the
    random routes, the placement mix and the full skew to the last rank's experts.  Per routing, the step with the
    split off, then each split of :data:`SOURCE_SPLITS` (the context forward, and the context-free forward with the
    recompute): every output of every rank has the split-off bytes.  One step captured under CUDA Graph with the
    ``rich2`` split and replayed after loading each routing's inputs (the kernel decides per launch from the replayed
    schedule) gives the same bytes.  Returns per routing the ranks the host rule finds rich per split."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    topk = kernels.BEFORE_DOWN_LAYOUTS[(ep, local_experts)]
    counts = list(SOURCE_SPLIT_COUNTS[ep])
    functional, config, workspace = _before_down_group(local_experts, counts)
    off = dataclasses.replace(config, source_split_comm_sms=None, imbalance_comm_sms=None)
    assert off.fwd_num_comm_sms == 8
    splits = [
        (name, dataclasses.replace(off, source_split_comm_sms=rich(rank), source_split_threshold=threshold), threshold)
        for name, rich, threshold in SOURCE_SPLITS
    ]
    rich_ranks = {name: [r for r, hit in enumerate(source_rich(counts, t)) if hit] for name, _, t in splits}
    assert all(rich_ranks[name] for name, _, _ in SOURCE_SPLITS if name != "held"), rich_ranks
    assert not rich_ranks["held"], rich_ranks
    shared, _, routed = _r8_weights(
        ep, local_experts, BEFORE_DOWN_HIDDEN, BEFORE_DOWN_INTERMEDIATE, device, R8_PINNED
    )
    weights = (*shared, *routed)
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    torch.manual_seed(8571)
    routings = (
        ("random", None),
        ("placement-mixed", _placement_ids(counts, ep, local_experts, topk, device)),
        ("full-skew-last", _skew_ids(counts, local_experts, topk, ep - 1, device)[0]),
    )
    graph, records = None, []
    for index, (name, ids) in enumerate(routings):
        torch.manual_seed(8581 + index)
        data = _r8_data(counts, BEFORE_DOWN_HIDDEN, topk, ep * local_experts, device, ids=ids)
        x, local_ids, scores, dy = _section(data, counts)

        def iteration(cfg, recompute=False, save_context=True, x=x, ids=local_ids, scores=scores, dy=dy):
            return TrainingIteration(
                cfg, workspace, x, ids, scores, dy, weights, functional=functional, swiglu_limit=R8_LIMIT,
                accumulators=accumulators, recompute=recompute, save_context=save_context,
            )  # fmt: skip

        outputs = _snapshot(iteration(off).run())
        _contract(outputs, counts[rank], BEFORE_DOWN_HIDDEN, topk)
        dist.barrier()
        for split, cfg, _ in splits:
            for recompute, save_context in ((False, True), (True, False)):
                values = _snapshot(iteration(cfg, recompute, save_context).run())
                differs = [n for n, a, b in zip(RESULT_NAMES, outputs, values, strict=True) if not torch.equal(a, b)]
                assert not differs, (rank, name, split, "context-free" if recompute else "context", differs)
            dist.barrier()
        if graph is None:
            graph = iteration(splits[0][1])
            graph.capture()
        else:
            _load(graph, data, counts)
        replays = [_snapshot(graph.run()) for _ in range(2)]
        assert all(_same(outputs, other) for other in replays), (rank, name, "graph replay")
        dist.barrier()
        records.append(dict(case=f"bf16-clamped-before-down-source-split-{name}", counts=counts, rich=rich_ranks))
    del graph
    torch.cuda.synchronize()
    dist.barrier()
    return records


#: The 512-expert top-10 group of the explicit modes (``router_weight_gradient="direct"``, ``shared_expert="scaled"``)
#: and the names of its step's values: the package's nine and ``d_shared_scale``.
SCALED_GROUP = kernels.MODE_GROUPS["scaled_top10"]
SCALED_RESULT_NAMES = (*RESULT_NAMES, "d_shared_scale")
#: Hidden and intermediate widths of the scaled group's worker.
SCALED_HIDDEN, SCALED_INTERMEDIATE = 512, 256


def _cpu_group_weights(ep, local_experts, hidden, intermediate, device, seed=32483):
    """BF16 shared weights and the group's routed weights drawn on the CPU (identical on every rank) and copied to
    ``device``; this rank's slice."""
    torch.manual_seed(seed)
    total = ep * local_experts
    shapes = (
        ((intermediate, hidden), hidden), ((intermediate, hidden), hidden), ((hidden, intermediate), intermediate),
        ((total, intermediate, hidden), hidden), ((total, intermediate, hidden), hidden),
        ((total, hidden, intermediate), intermediate),
    )  # fmt: skip
    drawn = [
        (randn(*shape) / fan_in**0.5).bfloat16().to(device) for shape, fan_in in shapes
    ]
    shared, global_routed = drawn[:3], drawn[3:]
    rank = dist.get_rank()
    routed = [
        w[rank * local_experts : (rank + 1) * local_experts].contiguous()
        for w in global_routed
    ]
    return shared, global_routed, routed


def _cpu_group_data(counts, hidden, topk, experts, device):
    """Global inputs of one step drawn on the CPU from the current random state (identical on every rank), copied to
    ``device``: BF16 ``x`` and ``dY``, int64 expert ids (``topk`` distinct experts per token), FP32 router weights in
    [0, 1.5) with every 5th token's first one exactly 0, and FP32 shared scales in [0, 2) with every 4th token's
    exactly 0 and the next one's exactly 1."""
    total = sum(counts)
    x = (randn(total, hidden) * 0.125).bfloat16()
    dy = (randn(total, hidden) * 0.125).bfloat16()
    ids = rand(total, experts).argsort(-1)[:, :topk].contiguous()
    scores = rand(total, topk) * 1.5
    scores[::5, 0] = 0.0
    scale = rand(total) * 2.0
    scale[::4] = 0.0
    scale[1::4] = 1.0
    data = dict(x=x, d_output=dy, expert_ids=ids, scores=scores, shared_scale=scale)
    return {key: value.to(device) for key, value in data.items()}


class _ScaledIteration:
    """One step of the scaled group's adapter on fixed input tensors (this rank's section of ``x``, ``ids``,
    ``scores``, ``dy`` and ``shared_scale``): ``build_schedule``, the context forward (with ``recompute`` the
    context-free forward and ``recompute_forward_context``), ``backward`` into the zeroed FP32 accumulators.  ``run()``
    returns the values of :data:`SCALED_RESULT_NAMES`; after ``capture()`` it replays the step's CUDA Graph."""

    def __init__(self, functional, config, workspace, weights, accumulators, x, ids, scores, dy, shared_scale, *,
                 recompute=False):  # fmt: skip
        self.functional, self.config, self.workspace = functional, config, workspace
        self.weights, self.accumulators, self.recompute = (
            weights,
            accumulators,
            recompute,
        )
        self.x, self.ids, self.scores, self.dy, self.shared_scale = (
            x,
            ids,
            scores,
            dy,
            shared_scale,
        )
        self.graph = self.outputs = self.schedule = None

    def _execute(self):
        f, w = self.functional, self.weights
        self.schedule = f.build_schedule(
            self.workspace, self.config, self.ids, num_local_experts=w[3].shape[0]
        )
        y, context = f.forward(
            self.config, self.workspace, self.schedule, self.x, self.scores, *w, save_context=not self.recompute,
            shared_scale=self.shared_scale,
        )  # fmt: skip
        if self.recompute:
            context = f.recompute_forward_context(
                self.config, self.workspace, self.schedule, self.x, w[0], w[1], w[3], w[4]
            )  # fmt: skip
        for accumulator in self.accumulators:
            accumulator.zero_()
        gradients = f.backward(
            self.config, self.workspace, self.schedule, context, self.dy, self.x, self.scores, *w,
            weight_grad_accumulators=self.accumulators, shared_scale=self.shared_scale,
        )  # fmt: skip
        self.outputs = (y, *gradients)
        expected = (torch.bfloat16, torch.bfloat16) + (torch.float32,) * 8
        if any(v.dtype != d for v, d in zip(self.outputs, expected, strict=True)):
            raise RuntimeError("Output/gradient formats changed")
        return self.outputs

    def capture(self):
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

    def load(self, data, counts):
        """Copy this rank's section of the global ``data`` into the step's input tensors."""
        rank = dist.get_rank()
        start = sum(counts[:rank])
        section = slice(start, start + counts[rank])
        for destination, key in zip(
            (self.x, self.ids, self.scores, self.dy, self.shared_scale),
            ("x", "expert_ids", "scores", "d_output", "shared_scale"),
            strict=True,
        ):
            destination.copy_(data[key][section])
        torch.cuda.synchronize()
        dist.barrier()


def _scaled_contract(values, n, hidden, topk):
    """Shapes of the scaled group's step values for ``n`` local tokens, every value finite."""
    _contract(values[:9], n, hidden, topk)
    assert values[9].shape == (n,) and torch.isfinite(values[9]).all().item(), (
        "d_shared_scale"
    )


def _scaled_judge(
    iteration, counts, shared, global_routed, *, local_experts, hidden, seed
):
    """Judge the scaled group's whole-group outputs over the draws of :func:`tests.cake_naive.judge_case` (inputs of
    :func:`_cpu_group_data`) against the float64 oracle beside the naive layer with ``shared_scale`` and
    ``direct=True``: ``y``, ``dx``, the router-weight gradients, ``d_shared_scale`` and the weight gradients (routed
    over every expert, shared per rank).  Returns the verdicts, the alternatives' verdicts (none) and the failures."""
    topk = SCALED_GROUP.topk
    experts = len(counts) * local_experts
    device = iteration.x.device
    naive_values = {}

    def run(j, _seed):
        data = _cpu_group_data(counts, hidden, topk, experts, device)
        iteration.load(data, counts)
        values = _snapshot(iteration.run())
        outputs = {
            name: _gather_rows(values[i], counts)
            for i, name in ((0, "y"), (1, "dx"), (2, "d_scores"))
        }
        outputs["d_shared_scale"] = _gather_rows(values[9], counts)
        for name, value in zip(RESULT_NAMES[3:6], values[3:6], strict=True):
            outputs[name] = torch.cat(_gather(value))
        for name, value in zip(RESULT_NAMES[6:], values[6:9], strict=True):
            outputs[name] = torch.stack(_gather(value))
        segs = segments(iteration.schedule, counts, local_experts, topk)
        refs = {}
        for dtype in (naive.ORACLE, naive.NAIVE):
            r = naive.layer(
                data["x"], data["d_output"], data["scores"], shared, global_routed, segs, None, dtype,
                sections=counts, shared_scale=data["shared_scale"], direct=True,
            )  # fmt: skip
            refs[dtype] = dict(
                y=r["y"],
                dx=r["dx"],
                d_scores=r["d_scores"],
                d_shared_scale=r["d_shared_scale"],
            )
            refs[dtype].update(zip(RESULT_NAMES[3:], r["dw"]))
        naive_values[j] = refs[naive.NAIVE]
        return outputs, refs[naive.ORACLE]

    verdict, failures = judge_case(run, lambda j, tier: naive_values.pop(j), seed=seed)
    return list(verdict.verdicts().values()), [], failures


def scaled_group_training():
    """The 512-expert top-10 group (``router_weight_gradient="direct"``, ``shared_expert="scaled"``, plain SwiGLU, FP32
    weight gradients) through the adapter at EP8 ``(8, 64)`` on one workspace over the source counts of
    :func:`before_down_steps` (padded and unpadded staging, several macrobatches, a new schedule per count, two empty
    source ranks in the last step), on inputs drawn on the CPU: per count the contract (ten values, ``d_shared_scale``
    FP32 per source row), repeatability eager and under CUDA Graph replay, the checkpointed step (context-free forward,
    recompute, backward) with the context step's bytes, no kernel load after the first count, and the whole group's
    outputs judged.  The workspace has the no-overflow multiplier at its physical source rows."""
    _exact_matmul()
    device = torch.device("cuda", torch.cuda.current_device())
    rank, ep = dist.get_rank(), dist.get_world_size()
    (group_ep, local_experts), topk = SCALED_GROUP.layouts[0], SCALED_GROUP.topk
    assert ep == group_ep, (ep, group_ep)
    hidden, intermediate = SCALED_HIDDEN, SCALED_INTERMEDIATE
    steps = before_down_steps(ep)
    shared, global_routed, routed = _cpu_group_weights(
        ep, local_experts, hidden, intermediate, device
    )
    weights = (*shared, *routed)
    source_capacity = max(max(counts) for counts in steps)
    multiplier = bound_multiplier(
        _source_rows(source_capacity, topk), topk, ep, local_experts
    )
    functional = prepare_mok_bf16(
        ep_size=ep, local_experts=local_experts, topk=topk, fp32_wgrad=True, router_weight_gradient="direct",
        shared_expert="scaled",
    )  # fmt: skip
    functional.prepare(save_context=False)
    functional.prepare_recompute()
    config, workspace = create_mok_bf16_workspace(
        group=dist.group.WORLD,
        device=device,
        num_local_tokens=steps[0][rank],
        source_capacity=source_capacity,
        hidden_size=hidden,
        topk=topk,
        fwd_num_comm_sms=8,
        bwd_num_comm_sms=8,
        minibatch_size=256,
        macrobatch_size=768,
        schedule_capacity_multiplier=multiplier,
    )
    accumulators = [torch.zeros(w.shape, device=device) for w in weights]
    records, loads = [], None
    for index, counts in enumerate(steps):
        torch.manual_seed(8317 + index)
        data = _cpu_group_data(counts, hidden, topk, ep * local_experts, device)
        start = sum(counts[:rank])
        section = slice(start, start + counts[rank])
        inputs = [
            data[k][section].clone()
            for k in ("x", "expert_ids", "scores", "d_output", "shared_scale")
        ]
        case = "counts-" + "-".join(map(str, counts))

        def iteration(recompute, inputs=inputs):
            return _ScaledIteration(
                functional, config, workspace, weights, accumulators, *inputs, recompute=recompute
            )  # fmt: skip

        saved_context = iteration(False)
        outputs = _snapshot(saved_context.run())
        _scaled_contract(outputs, counts[rank], hidden, topk)
        again = [_snapshot(saved_context.run()) for _ in range(REPEATS - 1)]
        assert all(_same(outputs, other) for other in again), (
            rank,
            case,
            "eager repeat",
        )
        dist.barrier()
        checkpointed = _snapshot(iteration(True).run())
        differs = [
            n
            for n, a, b in zip(SCALED_RESULT_NAMES, outputs, checkpointed, strict=True)
            if not torch.equal(a, b)
        ]
        assert not differs, (rank, case, "checkpointed", differs)
        dist.barrier()
        saved_context.capture()
        replays = [_snapshot(saved_context.run()) for _ in range(REPEATS)]
        assert all(_same(outputs, other) for other in replays), (
            rank,
            case,
            "graph replay",
        )
        dist.barrier()
        misses = jit._load.cache_info().misses
        if loads is None:
            loads = misses
        assert misses == loads, (rank, case, "kernel loads", loads, misses)
        record = dict(case=f"bf16-scaled-direct-top10-{case}", counts=counts)
        record["judged"] = _scaled_judge(
            saved_context, counts, shared, global_routed, local_experts=local_experts, hidden=hidden, seed=4500 + index
        )  # fmt: skip
        records.append(record)
        del saved_context
        torch.cuda.synchronize()
        dist.barrier()
    return records
