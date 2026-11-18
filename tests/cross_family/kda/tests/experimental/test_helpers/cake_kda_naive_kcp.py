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

The multi-GPU judged cases of the context-parallel (KCP) KDA packages (``test_cake_kda_*_train_kcp_naive.py``): one
process per rank (``torch.multiprocessing`` spawn, one GPU each, an NCCL group through a ``file://`` store), all
of the judge's draws inside one process group.

On every draw each rank draws the whole batch from the draw's seed on the host (the same tensors on every rank),
runs one training step of its shard through the KCP autograd entry (``cake_kda_det_kcp`` or
``cake_kda_nondet_kcp``; in the raw gate mode the parameter gradients summed over the ranks by the package's
``..._kcp_reduce_gate_param_grads``) and sends its outputs to rank 0.  Rank 0 puts the shards back in the global
token order and judges ``out`` and every gradient of the whole batch against the FP64 reference of the unsplit
batch, beside the naive BF16 tier (``cake_kda_naive.py``), with the judge of ``tests/cake_naive.py``.  Rank 0
writes the verdicts and the failures to a JSON file that the test reads; :func:`judge_kcp_case` records them and
settles them against the owner-accepted gaps (``cake_kda_known_gaps.py``).  A case is either drawn by
``cake_kda_naive.draw_case`` (any gate form and the in-kernel norm) or a KCP harness row of ``cake_kda_kcp_shapes``
(:func:`cake_kda_naive.kcp_row_inputs`), split by a layout of the package or an explicit fragment table.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MODES = ("det", "nondet")


def _api(mode: str) -> dict:
    """The KCP facade entries of ``mode``."""
    if mode == "det":
        from cake_pplx import cake_kda_det_train_kcp as m

        return {
            "partition": m.cake_kda_det_train_kcp_partition,
            "context": m.cake_kda_det_train_kcp_build_context,
            "step": m.cake_kda_det_kcp,
            "reduce": m.cake_kda_det_train_kcp_reduce_gate_param_grads,
        }
    from cake_pplx import cake_kda_nondet_train_kcp as m

    return {
        "partition": m.cake_kda_nondet_train_kcp_partition,
        "context": m.cake_kda_nondet_train_kcp_build_context,
        "step": m.cake_kda_nondet_kcp,
        "reduce": m.cake_kda_nondet_train_kcp_reduce_gate_param_grads,
    }


def shard(x: torch.Tensor, cu: list[int], fragments) -> torch.Tensor:
    """The rows of ``x`` of one rank's fragments, in the partition's slot order."""
    pieces = [
        x[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
        for f in fragments
        if f.length > 0
    ]
    return torch.cat(pieces, 0) if pieces else x[:0].clone()


def to_global(
    fragments_by_rank, rank_tensors: list[torch.Tensor], cu: list[int]
) -> torch.Tensor:
    """The global ``[S, ...]`` tensor of the ranks' local buffers (inverse of :func:`shard`)."""
    ref = rank_tensors[0]
    out = torch.empty(
        (cu[-1],) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device
    )
    for fragments, t in zip(fragments_by_rank, rank_tensors, strict=True):
        pos = 0
        for f in fragments:
            if f.length:
                out[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] = t[
                    pos : pos + f.length
                ]
            pos += f.length
    return out


def _rank_step(
    api,
    ctx,
    inputs: dict,
    fragments,
    cu: list[int],
    device,
    lower_bound: float,
    l2n: bool,
) -> dict:
    """One KCP training step of this rank's shard: ``out``, ``dq``, ``dk``, ``dv``, ``dg``, ``dbeta`` of its tokens
    and, in the raw gate mode, the rank-summed ``dA_log`` / ``ddt_bias``, on the host."""
    from .cake_kda_naive import GRAD_NAMES

    raw = "A_log" in inputs
    local = [
        shard(inputs[n], cu, fragments).to(device) for n in ("q", "k", "v", "g", "beta")
    ]
    leaves = [x.detach().clone().requires_grad_(True) for x in local]
    params = []
    kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
    if raw:
        params = [
            inputs[n].to(device).detach().clone().requires_grad_(True)
            for n in ("A_log", "dt_bias")
        ]
        kw.update(
            use_gate_in_kernel=True,
            A_log=params[0],
            dt_bias=params[1],
            lower_bound=lower_bound,
        )
    o = api["step"](*leaves, ctx, **kw)
    dout = shard(inputs["dout"], cu, fragments).to(device)
    grads = torch.autograd.grad(o, leaves + params, dout)
    res = {"out": o.detach()}
    res.update(dict(zip(GRAD_NAMES, grads[:5], strict=True)))
    if raw:
        res["dA_log"], res["ddt_bias"] = api["reduce"](ctx, grads[5], grads[6])
    return {n: t.detach().cpu() for n, t in res.items()}


def _run_rank(
    rank: int,
    W: int,
    mode: str,
    case: tuple,
    seed: int,
    rdzv: str,
    result: str,
    row: str | None,
    fragments,
) -> None:
    from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed

    from .cake_kda_naive import (
        draw_case,
        kcp_row_inputs,
        naive_of,
        oracle_with_grads,
    )

    lengths, H, gate, l2n, layout, scale, lower_bound = case
    harness_row = None
    if row is not None:
        from .cake_kda_kcp_shapes import ROWS

        harness_row = ROWS[row]
        lengths, H = harness_row.lengths, harness_row.h
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl", init_method=rdzv, rank=rank, world_size=W, device_id=device
    )
    try:
        api = _api(mode)
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + int(n))
        if fragments is not None:
            part = api["partition"](cu, W, fragments=fragments)
        else:
            part = api["partition"](cu, W, layout)
        ctx = api["context"](part, dist.group.WORLD, rank, device)
        judge, failures = TierJudge(DRAWS), []
        for j in range(DRAWS):
            s = draw_seed(seed, j)
            if harness_row is not None:
                inputs = kcp_row_inputs(harness_row, s, "cpu")
            else:
                inputs = draw_case(lengths, H, gate, l2n, s, "cpu")
            mine = _rank_step(
                api, ctx, inputs, part.fragments[rank], cu, device, lower_bound, l2n
            )
            every = [None] * W
            dist.all_gather_object(every, mine)
            if rank != 0:
                continue
            got = {}
            for name in mine:
                if name in ("dA_log", "ddt_bias"):
                    got[name] = every[0][name]
                else:
                    got[name] = to_global(part.fragments, [e[name] for e in every], cu)
            on_dev = {n: t.to(device) for n, t in inputs.items()}
            ref = oracle_with_grads(on_dev, scale, lower_bound, l2n)
            nv = naive_of(on_dev, scale, lower_bound, l2n)
            for name, actual in got.items():
                actual = actual.to(device)
                if message := contract(name, actual, ref[name]):
                    failures.append(f"draw {j}: {message}")
                    if actual.shape != ref[name].shape:
                        continue
                judge.add(name, j, actual, ref[name], {"bf16": nv[name]})
        if rank == 0:
            failures += judge.failures()
            with open(result, "w") as fh:
                json.dump({"verdicts": judge.verdicts(), "failures": failures}, fh)
        dist.barrier()
    except BaseException:
        if rank == 0:
            with open(result, "w") as fh:
                json.dump({"verdicts": {}, "failures": [traceback.format_exc()]}, fh)
        raise
    finally:
        dist.destroy_process_group()


def judge_kcp(
    mode: str,
    case: tuple,
    seed: int,
    W: int,
    *,
    row: str | None = None,
    fragments=None,
) -> tuple[dict, list[str]]:
    """Spawn ``W`` ranks for one case ``(lengths, H, gate, l2n, layout, scale, lower_bound)`` of ``mode`` (``"det"``
    or ``"nondet"``) and return rank 0's ``(verdicts, failures)``.  ``row``: the name of a KCP harness row
    (``cake_kda_kcp_shapes.ROWS``) whose drawn operands (:func:`cake_kda_naive.kcp_row_inputs`) replace
    :func:`cake_kda_naive.draw_case`'s (its lengths and heads replace the case's; the activated gate, no in-kernel
    norm).  ``fragments``: an explicit table, per rank its ``(doc, offset, length)`` fragments, in place of
    ``layout``.  :func:`judge_kcp_case` records and settles the result."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    with tempfile.TemporaryDirectory(prefix="cake_kda_kcp_naive_") as tmp:
        rdzv = "file://" + os.path.join(tmp, "store")
        result = os.path.join(tmp, "result.json")
        try:
            mp.start_processes(
                _run_rank,
                args=(W, mode, case, seed, rdzv, result, row, fragments),
                nprocs=W,
                join=True,
                start_method="spawn",
            )
        except Exception as exc:
            if not os.path.exists(result):
                return {}, [f"{type(exc).__name__}: {exc}"]
        with open(result) as fh:
            report = json.load(fh)
    return report["verdicts"], report["failures"]


def judge_kcp_case(
    record_property,
    mode: str,
    case: tuple,
    seed: int,
    W: int,
    *,
    row: str | None = None,
    fragments=None,
) -> dict:
    """:func:`judge_kcp`, its verdicts recorded with ``record_property`` (``naive_tier``) and settled against the
    owner-accepted gaps (``cake_kda_known_gaps.settle``; the scope is the case's document lengths).  A run that judged
    nothing fails.  Returns the verdicts."""
    from .cake_kda_known_gaps import settle

    verdicts, failures = judge_kcp(mode, case, seed, W, row=row, fragments=fragments)
    for verdict in verdicts.values():
        record_property("naive_tier", verdict)
    assert verdicts, failures
    if row is not None:
        from .cake_kda_kcp_shapes import ROWS

        lengths = ROWS[row].lengths
    else:
        lengths = case[0]
    settle(verdicts, failures, lengths=lengths)
    return verdicts


__all__ = ["MODES", "judge_kcp", "judge_kcp_case", "shard", "to_global"]
