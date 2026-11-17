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

Benchmark of the context-parallel (KCP) Cake packed KDA chunked training kernels in both numerical modes against
the cuDNN frontend's context-parallel KDA, one process per rank (``torchrun --nproc_per_node W``), every arm on the
same packed batch split at the same cut points (the Cake partition's ``contiguous`` layout):

* ``cake_det_kcp``: ``cake_pplx.cake_kda_det_train_kcp``;
* ``cake_nondet_kcp``: ``cake_pplx.cake_kda_nondet_train_kcp``;
* ``cudnn_cp_bi1``: the cuDNN frontend's summary operations around ``kimi_delta_attention`` (plan ``kda_frost``,
  ``batch_invariant=True``, ``checkpoint_every_n_tokens=16``; ``cake_kda_baselines.py``), the baseline of
  ``cake_det_kcp``;
* ``cudnn_cp_bi0``: the same with ``batch_invariant=False``, the baseline of ``cake_nondet_kcp``.

The cuDNN frontend is optional: when it does not import, or refuses the row, its arms are reported as unsupported
with the reason and the Cake arms run on.

Timed boundaries, every one including its collectives (in the raw gate mode the parameter gradients are summed over
the ranks: ``..._kcp_reduce_gate_param_grads`` for the Cake arms, an NCCL all-reduce for the cuDNN arms):

* ``fwd``: the explicit forward with its saved set (Cake: ``..._kcp_forward``; cuDNN: summary, all-gather,
  forward);
* ``bwd``: the explicit backward from one saved forward (Cake: ``..._kcp_backward(saved=...)``; cuDNN: backward
  summary, all-gather, backward);
* ``step``: ``fwd`` + ``bwd``;
* ``ckpt``: ``torch.utils.checkpoint(..., use_reentrant=False)`` around the autograd step (Cake:
  ``cake_kda_*_kcp``; cuDNN: an autograd function over the same calls), then the backward: two forwards (each with
  its exchange) and the backward.

Timing: eager (no CUDA graph), CUDA events around one call on every rank, after an L2 flush, a device
synchronization and a barrier; the slowest rank's time counts for the call.  As the device is idle when a call
starts, its time includes the host time to issue the call: a call whose launches take longer to issue than to run
is timed by its host path (the explicit calls and the autograd step can differ by that alone).  Per round and arm:
``--warmup`` untimed calls, then ``--samples`` timed calls of every boundary, whose median is the round's figure;
the arms run interleaved for ``--rounds`` rounds, the order reversed on every other round.  A row reports the
median of the round figures with their ``[min, max]``, and the speedups ``cudnn_cp_bi1 / cake_det_kcp``,
``cudnn_cp_bi0 / cake_nondet_kcp`` and ``cake_det_kcp / cake_nondet_kcp`` (above 1: the second arm is faster).

Check (``--check``, on by default): every arm's ``out`` and gradients of the whole batch (gathered to rank 0)
against the same implementation's unsplit call on one GPU (the Cake single-GPU packages; the cuDNN single-GPU
call with the same settings): the relative L2 difference, and with ``--fp64`` (on by default) the relative L2
error of the split and of the unsplit result against the FP64 token recurrence.  The check confirms that every arm
computes the same function under the split (a wiring error shows as a difference of the order of the outputs);
an output whose split error exceeds 1.005 times its unsplit error is flagged.  It is not the accuracy gate: the
judged accuracy of the KCP packages is ``tests/experimental/test_cake_kda_*_train_kcp_naive.py``.  A non-finite
value or a missing Cake arm fails the run (exit status 1).

Rows: every ``--shapes`` entry (``SxN`` or ``L1,L2,...``) with every ``--heads`` value, ``--gate`` and ``--l2n``
mode.  Official numbers: run one benchmark per node (one process per GPU, all GPUs of the node in the group), with
no other work on the node.

Example::

    torchrun --nproc_per_node 4 benchmarks/bench_cake_kda_kcp_baselines.py --shapes 16384x3 --heads 4 \\
        --gate g raw --l2n 0 1 --json kda_kcp_baselines.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cake_kda_baselines as baselines  # noqa: E402
from bench_cake_kda_baselines import (  # noqa: E402
    GRAD_NAMES,
    RAW_GRAD_NAMES,
    env_info,
    make_row,
    rel_l2,
    shape_lengths,
)
from cake_pplx import cake_kda_det_train_kcp as det_kcp  # noqa: E402
from cake_pplx import cake_kda_nondet_train_kcp as nd_kcp  # noqa: E402
from cake_pplx.cake_kda_det_train import cake_kda_det  # noqa: E402
from cake_pplx.cake_kda_nondet_train import cake_kda_nondet  # noqa: E402
from tests.cake_naive import NEAR_TIE  # noqa: E402
from tests.experimental.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    LOWER_BOUND,
)
from tests.experimental.test_helpers.cake_kda_naive import oracle_with_grads  # noqa: E402

CAKE_ARMS = ("cake_det_kcp", "cake_nondet_kcp")
ARMS = CAKE_ARMS + baselines.CP_ARMS
BOUNDARIES = ("fwd", "bwd", "step", "ckpt")
SPEEDUPS = (
    ("cudnn_cp_bi1", "cake_det_kcp"),
    ("cudnn_cp_bi0", "cake_nondet_kcp"),
    ("cake_det_kcp", "cake_nondet_kcp"),
)
L2_FLUSH_BYTES = 256 << 20
CAKE_API = {
    "cake_det_kcp": {
        "partition": det_kcp.cake_kda_det_train_kcp_partition,
        "context": det_kcp.cake_kda_det_train_kcp_build_context,
        "forward": det_kcp.cake_kda_det_train_kcp_forward,
        "backward": det_kcp.cake_kda_det_train_kcp_backward,
        "step": det_kcp.cake_kda_det_kcp,
        "reduce": det_kcp.cake_kda_det_train_kcp_reduce_gate_param_grads,
        "workspace": lambda **kw: det_kcp.CakeKDADetKcpWorkspace(**kw),
        "unsplit": cake_kda_det,
    },
    "cake_nondet_kcp": {
        "partition": nd_kcp.cake_kda_nondet_train_kcp_partition,
        "context": nd_kcp.cake_kda_nondet_train_kcp_build_context,
        "forward": nd_kcp.cake_kda_nondet_train_kcp_forward,
        "backward": nd_kcp.cake_kda_nondet_train_kcp_backward,
        "step": nd_kcp.cake_kda_nondet_kcp,
        "reduce": nd_kcp.cake_kda_nondet_train_kcp_reduce_gate_param_grads,
        "workspace": lambda **kw: nd_kcp.CakeKDANondetKcpWorkspace(**kw),
        "unsplit": cake_kda_nondet,
    },
}


def _init() -> tuple[int, int, torch.device]:
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local % torch.cuda.device_count())
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(minutes=60)
        )
    return dist.get_rank(), dist.get_world_size(), device


def _shard(x: torch.Tensor, cu: list[int], fragments) -> torch.Tensor:
    pieces = [
        x[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
        for f in fragments
        if f.length > 0
    ]
    return torch.cat(pieces, 0) if pieces else x[:0].clone()


def _to_global(fragments_by_rank, rank_tensors, cu: list[int]) -> torch.Tensor:
    ref = rank_tensors[0]
    out = torch.empty((cu[-1],) + tuple(ref.shape[1:]), dtype=ref.dtype)
    for fragments, t in zip(fragments_by_rank, rank_tensors, strict=True):
        pos = 0
        for f in fragments:
            if f.length:
                out[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] = t[
                    pos : pos + f.length
                ]
            pos += f.length
    return out


def _reduce_sum(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    out = []
    for t in tensors:
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out.append(t)
    return tuple(out)


class Arm:
    """One arm of a row on this rank: its timed boundaries and :meth:`results` (one untimed step)."""

    def __init__(
        self,
        name: str,
        local: dict,
        raw: bool,
        l2n: bool,
        rank: int,
        part,
        cu,
        H: int,
        device,
    ) -> None:
        self.name, self.local, self.raw, self.l2n = name, local, raw, l2n
        names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw else ())
        self.tensors = [local[n] for n in names]
        self.dout = local["dout"]
        self.info: dict = {}
        if name in CAKE_ARMS:
            api = CAKE_API[name]
            self.api = api
            self.ctx = api["context"](part, dist.group.WORLD, rank, device)
            self.ws = api["workspace"](
                use_gate_in_kernel=raw, use_qk_l2norm_in_kernel=l2n
            )
            self.kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
            self.info = {
                "entry": api["step"].__module__ + "." + api["step"].__name__,
                "layout": "contiguous",
            }
        else:
            frags = [
                [(f.doc, f.off, f.length) for f in part.fragments[r]]
                for r in range(len(part.fragments))
            ]
            plan = baselines.CudnnCpPlan(frags, cu, rank, dist.group.WORLD, device)
            self.cp = baselines.CudnnCp(name, plan, raw, l2n, LOWER_BOUND, H)
            self.info = dict(self.cp.info)
        self.saved = None

    def _cake_kw(self, params) -> dict:
        kw = dict(self.kw)
        if self.raw:
            kw.update(
                use_gate_in_kernel=True,
                A_log=params[0],
                dt_bias=params[1],
                lower_bound=LOWER_BOUND,
            )
        return kw

    def fwd(self):
        t = self.tensors
        if hasattr(self, "cp"):
            return self.cp.forward(*t)
        return self.api["forward"](
            *t[:5], self.ctx, workspace=self.ws, **self._cake_kw(t[5:])
        )

    def bwd(self, saved=None):
        t = self.tensors
        saved = self.saved if saved is None else saved
        if hasattr(self, "cp"):
            grads = self.cp.backward(saved[1], self.dout, *t)
            if self.raw:
                grads = grads[:5] + _reduce_sum(*grads[5:])
            return grads
        grads = self.api["backward"](
            *t[:5],
            self.ctx,
            self.dout,
            saved=saved[1],
            workspace=self.ws,
            **self._cake_kw(t[5:]),
        )
        if self.raw:
            grads = tuple(grads[:5]) + tuple(
                self.api["reduce"](self.ctx, grads[5], grads[6])
            )
        return grads

    def step(self):
        return self.bwd(self.fwd())

    def ckpt(self):
        leaves = [x.detach().clone().requires_grad_(True) for x in self.tensors]
        if hasattr(self, "cp"):
            fn = self.cp.step
        else:

            def fn(*a):
                return self.api["step"](
                    *a[:5], self.ctx, workspace=self.ws, **self._cake_kw(a[5:])
                )

        o = checkpoint(fn, *leaves, use_reentrant=False)
        grads = torch.autograd.grad(o, leaves, self.dout)
        if self.raw:
            if hasattr(self, "cp"):
                grads = tuple(grads[:5]) + _reduce_sum(*grads[5:])
            else:
                grads = tuple(grads[:5]) + tuple(
                    self.api["reduce"](self.ctx, grads[5], grads[6])
                )
        return grads

    def boundaries(self) -> dict[str, Callable[[], object]]:
        self.saved = self.fwd()
        return {"fwd": self.fwd, "bwd": self.bwd, "step": self.step, "ckpt": self.ckpt}

    def results(self) -> dict[str, torch.Tensor]:
        """``out`` and the gradients of this rank's tokens (the parameter gradients summed over the ranks), on the
        host; the Cake arms' backward order ``dq, dk, dv, dbeta, dg`` and the cuDNN order ``dq, dk, dv, dg, dbeta``
        are both mapped to names."""
        saved = self.fwd()
        out = saved[0]
        grads = self.bwd(saved)
        if hasattr(self, "cp"):
            names = ("dq", "dk", "dv", "dg", "dbeta")
        else:
            names = ("dq", "dk", "dv", "dbeta", "dg")
        res = {"out": out, **dict(zip(names, grads[:5], strict=True))}
        if self.raw:
            res["dA_log"], res["ddt_bias"] = grads[5], grads[6]
        return {n: t.detach().cpu() for n, t in res.items()}


def _time_boundary(fn, samples: int, warmup: int, flush: torch.Tensor, device) -> float:
    """The median over ``samples`` calls of the slowest rank's CUDA-event milliseconds (after ``warmup`` calls)."""
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for i in range(warmup + samples):
        flush.fill_(i & 0xFF)
        torch.cuda.synchronize(device)
        dist.barrier()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        t = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if i >= warmup:
            times.append(float(t.item()))
    return statistics.median(times)


def _unsplit(arm: str, inputs: dict, l2n: bool) -> dict[str, torch.Tensor]:
    """The same implementation's ``out`` and gradients of the whole batch on this GPU, on the host."""
    raw = "A_log" in inputs
    names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw else ())
    leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]
    if arm in CAKE_ARMS:
        fn = CAKE_API[arm]["unsplit"]

        def call(*a):
            kw = {"use_qk_l2norm_in_kernel": True} if l2n else {}
            if raw:
                kw.update(
                    use_gate_in_kernel=True,
                    A_log=a[5],
                    dt_bias=a[6],
                    lower_bound=LOWER_BOUND,
                )
            return fn(*a[:5], inputs["cu_seqlens"], **kw)

    else:
        call, _ = baselines.single_gpu_arm(
            arm.replace("_cp", ""), inputs, LOWER_BOUND, l2n
        )
    o = call(*leaves)
    grads = torch.autograd.grad(o, leaves, inputs["dout"])
    names = RAW_GRAD_NAMES if raw else GRAD_NAMES
    res = {"out": o, **dict(zip(names, grads, strict=True))}
    return {n: t.detach().cpu() for n, t in res.items()}


def run_row(
    lengths, H: int, gate: str, l2n: bool, args, rank: int, world: int, device
) -> dict:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + int(n))
    inputs = make_row(lengths, H, gate, l2n, torch.device("cpu"))
    raw = gate == "raw"
    row = {
        "lengths": list(lengths),
        "tokens": cu[-1],
        "docs": len(lengths),
        "heads": H,
        "gate": gate,
        "l2n": l2n,
        "world_size": world,
        "layout": "contiguous",
        "arms": {},
        "unsupported": {},
    }
    part = det_kcp.cake_kda_det_train_kcp_partition(cu, world, "contiguous")
    nd_part = nd_kcp.cake_kda_nondet_train_kcp_partition(cu, world, "contiguous")
    same = [[(f.doc, f.off, f.length) for f in fr] for fr in part.fragments] == [
        [(f.doc, f.off, f.length) for f in fr] for fr in nd_part.fragments
    ]
    if not same:
        raise RuntimeError("the two Cake packages split the batch differently")
    row["tokens_per_rank"] = [sum(f.length for f in fr) for fr in part.fragments]
    local = {
        n: _shard(inputs[n], cu, part.fragments[rank]).to(device)
        for n in ("q", "k", "v", "g", "beta", "dout")
    }
    if raw:
        local["A_log"], local["dt_bias"] = (
            inputs["A_log"].to(device),
            inputs["dt_bias"].to(device),
        )
    arms, closures = {}, {}
    for name in args.arms:
        t0 = time.time()
        error = None
        try:
            arm = Arm(
                name,
                local,
                raw,
                l2n,
                rank,
                nd_part if name == "cake_nondet_kcp" else part,
                cu,
                H,
                device,
            )
            fns = arm.boundaries()
            for b in BOUNDARIES:
                fns[b]()
            torch.cuda.synchronize(device)
        except baselines.Unsupported as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 -- a baseline's own failure leaves its arm out
            if name in CAKE_ARMS:
                raise
            error = f"{type(exc).__name__}: {exc}"
            row.setdefault("errors", {})[name] = traceback.format_exc()[-3000:]
        # every rank must agree on the arm, or the collectives of the next arm would pair up wrongly
        flags = [None] * world
        dist.all_gather_object(flags, error)
        reasons = [f for f in flags if f is not None]
        if reasons:
            row["unsupported"][name] = reasons[0]
            torch.cuda.empty_cache()
            continue
        arms[name], closures[name] = arm, fns
        row["arms"][name] = {"info": arm.info, "setup_s": round(time.time() - t0, 1)}
    flush = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device=device)
    live = list(closures)
    per = {a: {b: [] for b in BOUNDARIES} for a in live}
    for r in range(args.rounds):
        order = live if r % 2 == 0 else list(reversed(live))
        for name in order:
            for b in BOUNDARIES:
                per[name][b].append(
                    _time_boundary(
                        closures[name][b], args.samples, args.warmup, flush, device
                    )
                )
    med = {}
    for name in live:
        row["arms"][name]["ms"] = {
            b: {
                "median": statistics.median(v),
                "min": min(v),
                "max": max(v),
                "rounds": v,
            }
            for b, v in per[name].items()
        }
        med[name] = {b: statistics.median(v) for b, v in per[name].items()}
    row["speedup"] = {
        f"{base}/{cake}": {b: med[base][b] / med[cake][b] for b in BOUNDARIES}
        for base, cake in SPEEDUPS
        if base in med and cake in med
    }
    if args.check:
        row["check"] = _check(arms, inputs, part, cu, l2n, args, rank, world, device)
    del closures, arms
    torch.cuda.empty_cache()
    return row


def _check(
    arms: dict, inputs: dict, part, cu, l2n: bool, args, rank: int, world: int, device
) -> dict:
    """Every arm's split results against its unsplit call (and the FP64 reference), on rank 0."""
    split = {}
    for name, arm in arms.items():
        mine = arm.results()
        every = [None] * world
        dist.all_gather_object(every, mine)
        if rank == 0:
            split[name] = {
                n: (
                    every[0][n]
                    if n in ("dA_log", "ddt_bias")
                    else _to_global(part.fragments, [e[n] for e in every], cu)
                )
                for n in mine
            }
    report: dict = {}
    if rank == 0:
        on_dev = {n: t.to(device) for n, t in inputs.items()}
        ref = None
        if args.fp64:
            t0 = time.time()
            ref = {
                n: t.cpu()
                for n, t in oracle_with_grads(
                    on_dev, baselines.SCALE, LOWER_BOUND, l2n
                ).items()
            }
            report["fp64_seconds"] = round(time.time() - t0, 1)
        for name, got in split.items():
            whole = _unsplit(name, on_dev, l2n)
            entry = {}
            for n, t in got.items():
                e = {
                    "finite": bool(torch.isfinite(t.double()).all()),
                    "split_vs_unsplit_rel_l2": rel_l2(t, whole[n]),
                }
                if ref is not None:
                    es, eu = rel_l2(t, ref[n]), rel_l2(whole[n], ref[n])
                    e.update(split_vs_fp64_rel_l2=es, unsplit_vs_fp64_rel_l2=eu)
                    e["flag"] = bool(
                        es > NEAR_TIE * eu and not torch.equal(t, whole[n])
                    )
                entry[n] = e
            report[name] = entry
    holder = [report]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def print_row(row: dict) -> None:
    head = f"W={row['world_size']} S={row['tokens']} N={row['docs']} H={row['heads']} gate={row['gate']} l2n={int(row['l2n'])}"
    for arm, reason in row["unsupported"].items():
        print(f"UNSUPPORTED {head} {arm}: {reason}", flush=True)
    cols = [a for a in ARMS if a in row["arms"] and "ms" in row["arms"][a]]
    print(
        f"| {head} | boundary | "
        + " | ".join(f"{a} ms" for a in cols)
        + " | "
        + " | ".join(row["speedup"])
        + " |",
        flush=True,
    )
    for b in BOUNDARIES:
        cells = [
            f"{row['arms'][a]['ms'][b]['median']:.3f} [{row['arms'][a]['ms'][b]['min']:.3f}, {row['arms'][a]['ms'][b]['max']:.3f}]"
            for a in cols
        ]
        ratios = [f"{v[b]:.3f}" for v in row["speedup"].values()]
        print(
            f"| {head} | {b} | "
            + " | ".join(cells)
            + " | "
            + " | ".join(ratios)
            + " |",
            flush=True,
        )
    for arm, entry in (row.get("check") or {}).items():
        if not isinstance(entry, dict) or arm == "fp64_seconds":
            continue
        parts = []
        for n, e in entry.items():
            s = f"{n}: split/unsplit {e['split_vs_unsplit_rel_l2']:.2e}"
            if "split_vs_fp64_rel_l2" in e:
                s += f" fp64 {e['split_vs_fp64_rel_l2']:.2e}/{e['unsplit_vs_fp64_rel_l2']:.2e}"
            if not e["finite"]:
                s += " NONFINITE"
            if e.get("flag"):
                s += " FLAG"
            parts.append(s)
        print(f"CHECK {head} {arm} " + "; ".join(parts), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("limitations under the License.")[-1].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Official numbers: one benchmark per node, one process per GPU, no other work on the node.",
    )
    parser.add_argument(
        "--shapes", nargs="+", default=["16384x3"], help="SxN or L1,L2,..."
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[4])
    parser.add_argument("--gate", nargs="+", default=["g", "raw"], choices=("g", "raw"))
    parser.add_argument("--l2n", type=int, nargs="+", default=[0, 1], choices=(0, 1))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--json", type=str, default=None, help="rank 0 writes the rows here"
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.samples < 1 or args.warmup < 1:
        parser.error("--rounds, --samples and --warmup must be at least 1")
    rank, world, device = _init()
    torch.backends.cuda.matmul.allow_tf32 = False
    env = env_info(device) if rank == 0 else None
    if rank == 0:
        print("ENV " + json.dumps(env, default=str), flush=True)
        for arm in baselines.CP_ARMS:
            if arm in args.arms:
                print(
                    f"BASELINE {arm}: {baselines.arm_label(arm, env['baselines'])}",
                    flush=True,
                )
    doc = {
        "env": env,
        "world_size": world,
        "hosts": None,
        "protocol": {
            "boundaries": list(BOUNDARIES),
            "execution": "eager",
            "timer": "CUDA events per call on every rank, slowest rank; L2 flushed, synchronized, barrier before each call "
            "(so the host time to issue the call is included)",
            "warmup_calls": args.warmup,
            "samples": args.samples,
            "rounds": args.rounds,
            "statistic": "median over rounds of the per-round median of the samples",
            "order": "arms interleaved, reversed every other round",
            "speedup": "baseline ms / Cake ms (above 1: Cake faster)",
            "official": "one benchmark per node, one process per GPU",
        },
        "rows": [],
    }
    hosts = [None] * world
    dist.all_gather_object(hosts, platform.node())
    doc["hosts"] = hosts
    status = 0
    rows = [
        (shape_lengths(shape), H, gate, bool(l2n))
        for shape in args.shapes
        for H in args.heads
        for gate in args.gate
        for l2n in args.l2n
    ]
    for lengths, H, gate, l2n in rows:
        try:
            row = run_row(lengths, H, gate, l2n, args, rank, world, device)
        except Exception as exc:  # noqa: BLE001
            row = {
                "lengths": list(lengths),
                "heads": H,
                "gate": gate,
                "l2n": l2n,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
            print(
                f"ERROR rank {rank} S={sum(lengths)} H={H} gate={gate} l2n={int(l2n)}: {row['error']}",
                flush=True,
            )
            doc["rows"].append(row)
            status = 1
            break  # the ranks may now disagree on the collectives: stop
        for entry in (row.get("check") or {}).values():
            if isinstance(entry, dict) and any(not e["finite"] for e in entry.values()):
                status = 1
        if any(a not in row["arms"] for a in CAKE_ARMS if a in args.arms):
            status = 1
        if rank == 0:
            print_row(row)
        doc["rows"].append(row)
        if rank == 0 and args.json:
            tmp = args.json + ".part"
            with open(tmp, "w") as fh:
                json.dump(doc, fh, indent=1, default=str)
            os.replace(tmp, args.json)
    if rank == 0 and args.json:
        tmp = args.json + ".part"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1, default=str)
        os.replace(tmp, args.json)
    dist.barrier()
    dist.destroy_process_group()
    return status


if __name__ == "__main__":
    sys.exit(main())
