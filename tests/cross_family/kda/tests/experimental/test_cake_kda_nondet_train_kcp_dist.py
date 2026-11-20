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

Multi-GPU rows of the non-deterministic context-parallel (KCP) KDA training route
(``cake_kda_nondet_train_kcp.cake_kcp``): ``W`` ranks over NCCL, each holding the fragments of one partition of a packed
batch.  Accuracy: over the judge's 32 draws (all in one process group) every form of a row (saved and replay mode, a
repeat, the autograd entry, a checkpoint, ``save_transfer=False``, the forward-only layouts, ``workspace=None``, the ring
exchange, graph replay; the upload, checkpoint and default-workspace rows' own forms) is gathered to rank 0 and judged
there against the FP64 reference of the unsplit batch beside the naive BF16 tier (``tests/cake_naive.py``,
``tests/experimental/test_helpers/cake_kda_naive.py``), each form under its own name, with no tolerance; the test records
the verdicts and settles them against the owner-accepted gaps.  The package has no bit contract: no form is compared
with another's bits, nor with the deterministic package's.  Bit checks remain for plumbing only (workspace bytes, carry
slots, collective counts, graph events, the deterministic conv halo and outputs -- KDA's operands in the checkpoint
rows' layer core --, ``dw`` read from its own forward).

Two launch forms: ``torch.multiprocessing.spawn`` (``pytest``: the ranks meet through a ``file://`` store under the
test's temporary directory; skipped when fewer than ``W`` CUDA devices are visible or the KCP programs are not
exported yet) and one process per Slurm task (``SLURM_PROCID`` / ``SLURM_NTASKS`` set: the test body runs in the
calling process, the rendezvous from ``MASTER_ADDR`` / ``MASTER_PORT``).

The rank-agreement check (``cake_kcp._rank_check``, one all-gather of a 16-byte digest at the first call of each call
key): CPU rows of the digest's inputs, the once-per-key cache and the drivers' issue points; two spawned ranks where
matching calls pass with every check counted and a mismatch trips the device-side assert instead of hanging.  The
checkpoint rows count the checks exactly beside the exchange's collectives.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
import types
import weakref
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.experimental.test_helpers.cake_kda_train_kcp_saved import (
    carry_slots,
    poison_unwritten_carry,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    GRAD_NAMES,
    LOWER_BOUND,
    make_inputs,
)
from tests.cake.test_helpers.cake_package import cake_package

cake_kcp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp")
cake_kcp_plan = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_plan")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")
cake_kcp_jit = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_jit")

PACKAGE_DIR = Path(cake_kcp_jit.__file__).resolve().parent
TENSORS = ("out",) + GRAD_NAMES
# graph="replay" over four forward + backward calls of one key: the backward's key sees the saved tensors, which
# change once (the eager forward's -> the captured forward's static ones), so it captures one call after the forward
GRAPH_EVENTS = [
    ["eager", "eager"],
    ["capture", "eager"],
    ["replay", "capture"],
    ["replay", "replay"],
]
SCALE = cake_kcp.SCALE
JUDGE_SEED = (
    8937  # the base seed of the judged rows (each draw's own seed follows from it)
)

# (lengths, H, W, layout): small packings with boundaries inside chunks, a document crossing >= 3 ranks (zigzag
# and per-document zigzag), empty / one-token fragments and an explicit ragged table with a zero-token rank.
ROWS = [
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((100, 640, 77, 1000, 129), 4, 4, "contiguous"),
    ((3000,), 4, 4, "zigzag"),
    ((3000,), 4, 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"),
    ((5000, 1003), 1, 4, "ragged:1,2,4997,1003"),
    # 16 heads: the S1 pair on sm_100a / sm_107a, the deterministic pair on sm_103a (the routing table's boundary)
    ((100, 640, 77, 1000, 129), 16, 2, "zigzag_per_document"),
]


def _eager_capture_replay(seq: list) -> bool:
    """One kind's graph events over the calls of one key: one or more eager calls, one capture, then replays only."""
    if seq.count("capture") != 1:
        return False
    i = seq.index("capture")
    return (
        i > 0
        and all(e == "eager" for e in seq[:i])
        and all(e == "replay" for e in seq[i + 1 :])
    )


def _bwd_sequence_first_main(rec: dict) -> list:
    """A backward ``LAST_CALL`` record's ``sequence`` in the order of ``bwd_tape_order="first"`` /
    ``bwd_k3_stream="main"``, by the decisions the record holds: where it ran ``"under_b1"``, the ``P = 1`` K1 tape
    replay (``kcp_fwdh_tape``, right after ``a2a:b1``) goes back right before ``kcp_dav`` (``dav``); where it ran
    ``"side"``, the dm-form K3 (the one ``kcp_transfer_walk`` before K2 ``_offset``, right before the replay, ``dav`` or
    ``kcp_dhu_offset``) goes back right after K2 ``_offset`` (``kcp_dhu_offset``).  A record with neither (``"first"`` /
    ``"main"``, a forward or the deterministic package's) comes back unchanged.  A record whose sequence does not hold
    a moved launch where its decision puts it, or holds the replay right after ``a2a:b1`` under ``"first"``, comes back
    with a first entry no launch has, so it never equals a launch sequence (a flag, not a raise: the multi-GPU rows
    gather their flags before any rank fails)."""
    seq = list(rec["sequence"])
    order, stream = rec.get("bwd_tape_order"), rec.get("bwd_k3_stream")
    bad = ["not the recorded backward order", *seq]
    b1 = seq.index("a2a:b1") if "a2a:b1" in seq else -1
    if stream == "side":
        k2 = (
            seq.index("kcp_dhu_offset") if "kcp_dhu_offset" in seq[: max(b1, 0)] else -1
        )
        k3 = [i for i in range(max(k2, 0)) if seq[i] == "kcp_transfer_walk"]
        if (
            k2 < 0
            or len(k3) != 1
            or seq[k3[0] + 1] not in ("kcp_fwdh_tape", "dav", "kcp_dhu_offset")
            or "kcp_transfer_walk" in seq[k2:b1]
        ):
            return bad
        del seq[k3[0]]
        seq.insert(seq.index("kcp_dhu_offset") + 1, "kcp_transfer_walk")
    elif stream not in (None, "main"):
        return bad
    if order == "under_b1":
        if (
            b1 < 0
            or seq[b1 + 1 : b1 + 2] != ["kcp_fwdh_tape"]
            or "kcp_fwdh_tape" in seq[:b1]
            or seq[:b1].count("dav") != 1
        ):
            return bad
        del seq[b1 + 1]
        seq.insert(seq.index("dav"), "kcp_fwdh_tape")
    elif order == "first":
        if b1 >= 0 and seq[b1 + 1 : b1 + 2] == ["kcp_fwdh_tape"]:
            return bad
    elif order is not None:
        return bad
    return seq


def _row_id(row) -> str:
    lengths, H, W, layout = row
    return "s" + "_".join(map(str, lengths)) + f"-h{H}-w{W}-{layout.split(':')[0]}"


def _programs_exported() -> bool:
    """Every shipped KCP record of ``cake_kcp_jit.MODULES`` has a library in the
    package's ``build.json`` -- else the kernel path cannot run."""
    manifest = PACKAGE_DIR / "build.json"
    if not manifest.exists():
        return False
    try:
        with open(manifest) as fh:
            libraries = {lib["name"] for lib in json.load(fh)["libraries"]}
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return all(
        any(
            cake_kcp_jit.library_name(name, arch) in libraries
            for arch in record["arches"]
        )
        for name, record in cake_kcp_jit.MODULES.items()
    )


def _partition(lengths, W: int, layout: str):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if layout.startswith("ragged:"):
        t_r = [int(x) for x in layout.split(":")[1].split(",")]
        assert len(t_r) == W and sum(t_r) == cu[-1]
        # contiguous ragged split of the global buffer, cut at document boundaries, in local order
        nested = [[] for _ in range(W)]
        pos = 0
        for r, n in enumerate(t_r):
            end = pos + n
            d = 0
            while d < len(cu) - 1 and cu[d + 1] <= pos:
                d += 1
            p = pos
            while p < end and d < len(cu) - 1:
                e = min(end, cu[d + 1])
                if e > p:
                    nested[r].append((d, p - cu[d], e - p))
                p = e
                d += 1
            pos = end
        return cake_kcp_plan.cake_kda_nondet_train_kcp_partition(
            cu, W, fragments=nested
        ), cu
    return cake_kcp_plan.cake_kda_nondet_train_kcp_partition(cu, W, layout=layout), cu


def _local_slices(part, rank: int, cu: list[int], tensors: dict) -> dict:
    """This rank's local buffer (fragments in slot order) of every global operand."""
    out = {}
    for name, t in tensors.items():
        if name == "cu_seqlens":
            continue
        parts = [
            t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
            for f in part.fragments[rank]
        ]
        out[name] = torch.cat(parts, 0) if parts else t[:0].clone()
    return out


def _to_global(
    part, rank_tensors: list[torch.Tensor], cu: list[int], total: int
) -> torch.Tensor:
    ref = rank_tensors[0]
    g = torch.empty((total,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=ref.device)
    for r, t in enumerate(rank_tensors):
        pos = 0
        for f in part.fragments[r]:
            if f.length:
                g[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length] = t[
                    pos : pos + f.length
                ]
            pos += f.length
    return g


def _gather_global(part, t: torch.Tensor, cu: list[int], total: int, W: int, rank: int):
    """The global ``[S, ...]`` tensor of every rank's local buffer ``t`` (fragments in slot order), on rank 0's device;
    ``None`` on the other ranks.  One NCCL gather of the buffers padded to the longest rank's length."""
    if W == 1:
        return _to_global(part, [t], cu, total)
    sizes = [sum(f.length for f in frs) for frs in part.fragments]
    pad = t.new_zeros((max(sizes),) + tuple(t.shape[1:]))
    pad[: t.shape[0]] = t
    bufs = [torch.empty_like(pad) for _ in range(W)] if rank == 0 else None
    dist.gather(pad, bufs, dst=0)
    if rank != 0:
        return None
    return _to_global(
        part, [b[:n] for b, n in zip(bufs, sizes, strict=True)], cu, total
    )


class _RankJudge:
    """Rank 0's judge of one multi-GPU row over the judge's draws (``tests/cake_naive.py``): every rank calls
    :meth:`draw` and :meth:`arm` in the same order (the gathers are collectives); rank 0 forms the draw's FP64
    reference, naive BF16 tier and FLA (one GPU) of the global operands and judges each arm's gathered outputs, an
    arm ``""`` as ``<output>``, another as ``<arm>.<output>``, a list of results of one arm keeping the worst."""

    def __init__(self, part, cu: list[int], W: int, rank: int, scale: float) -> None:
        from tests.cake_naive import DRAWS, TierJudge

        self.part, self.cu, self.W, self.rank, self.scale = part, cu, W, rank, scale
        self.total = cu[-1]
        self.judge = TierJudge(DRAWS) if rank == 0 else None
        self.failures: list[str] = []
        self.j = 0
        self.ref = self.tier = self.fla = None

    def draw(self, j: int, glob: dict) -> None:
        """Start draw ``j`` of the global operands ``glob`` (rank 0: its reference, tier and FLA)."""
        from tests.experimental.test_helpers.cake_kda_naive import (
            fla_of,
            naive_of,
            oracle_with_grads,
        )

        self.j = j
        if self.rank == 0:
            self.ref = oracle_with_grads(glob, self.scale, LOWER_BOUND, False)
            self.tier = naive_of(glob, self.scale, LOWER_BOUND, False)
            self.fla = fla_of(glob, self.scale, LOWER_BOUND, False)

    def arm(self, name: str, results) -> None:
        """Gather and judge (rank 0) one arm: a dict ``{output: local tensor}`` or a list of them."""
        from tests.cake_naive import contract

        for res in results if isinstance(results, (list, tuple)) else (results,):
            for out_name in TENSORS:
                if out_name not in res:
                    continue
                got = _gather_global(
                    self.part, res[out_name], self.cu, self.total, self.W, self.rank
                )
                if self.rank != 0:
                    continue
                key = f"{name}.{out_name}" if name else out_name
                want = self.ref[out_name]
                if message := contract(key, got, want):
                    self.failures.append(f"draw {self.j}: {message}")
                    if got.shape != want.shape:
                        continue
                self.judge.add(
                    key, self.j, got, want, {"bf16": self.tier[out_name]}, fla=self.fla.value(out_name, got)
                )

    def report(self) -> dict:
        """Rank 0's verdicts and failures (empty on the other ranks)."""
        if self.rank != 0:
            return {}
        return {
            "verdicts": self.judge.verdicts(),
            "failures": self.failures + self.judge.failures(),
        }


def _settle(record_property, report: dict, lengths) -> None:
    """Record rank 0's verdicts (``naive_tier``) and settle them against the owner-accepted gaps
    (``cake_kda_known_gaps.settle``; the scope is the row's document lengths).  A run that judged nothing fails."""
    from tests.experimental.test_helpers.cake_kda_known_gaps import settle

    verdicts = report.get("verdicts") or {}
    for verdict in verdicts.values():
        record_property("naive_tier", verdict)
    assert verdicts, report.get("failures")
    settle(verdicts, report.get("failures") or [], lengths=lengths)


def _run_rank(rank: int, W: int, row, rdzv: str | None, result_dir: str | None) -> dict:
    """The body of one rank over the judge's draws (all in one process group): on every draw the KCP forward /
    backward in every form, each form's ``out`` and gradients gathered to rank 0 and judged there against the FP64
    reference of the unsplit batch beside the naive BF16 tier, under its own name (:class:`_RankJudge`; the package
    has no bit contract between its forms).  The plumbing flags (the forward-only workspace bytes, the carry slots,
    the saved-set flags, every rank's graph events) hold on every draw, the SPEC-3 probe on the first."""
    from tests.cake_naive import DRAWS, draw_seed
    from tests.experimental.test_helpers.cake_kda_naive import draw_case

    lengths, H, _w, layout = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    ws = cake_kcp.CakeKDANondetKcpWorkspace()
    judge = _RankJudge(part, cu, W, rank, SCALE)

    def one_draw(ops: dict, first: bool):
        """Every form of one draw on this rank's slices ``ops``: its arms and its flags."""
        q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))

        def saved_pass(**opts):
            out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, **opts
            )
            # ``boundary_mode`` is a forward keyword: the backward replays the forward whenever ``saved`` is None
            bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
            dq, dk, dv, dbeta, dg = cake_kcp.cake_kda_nondet_train_kcp_backward(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                dout,
                saved=saved,
                workspace=ws,
                **bwd_opts,
            )
            return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}

        def autograd_pass(ckpt: bool, wsp=ws):
            leaves = [
                x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)
            ]

            def fn(*a):
                if wsp is None:
                    return cake_kcp.cake_kda_nondet_kcp(*a, ctx)
                return cake_kcp.cake_kda_nondet_kcp(
                    *a, ctx, None, "light", "saved", wsp
                )

            if ckpt:
                from torch.utils.checkpoint import checkpoint

                out = checkpoint(fn, *leaves, use_reentrant=False)
            else:
                out = fn(*leaves)
            out.backward(dout)
            return {
                "out": out.detach(),
                **{
                    f"d{n}": lf.grad
                    for n, lf in zip(("q", "k", "v", "g", "beta"), leaves, strict=True)
                },
            }

        # the probe runs once per backend, on request (spec3_probe="record"; "off" is the default), on the first
        # draw's first pass: a row after an earlier probing backward in this process (the Slurm per-task form runs
        # every row in one process) finds it spent
        pending = bool(getattr(cake_kcp.get_backend(), "_spec3_pending", False))
        arms = {"": saved_pass(spec3_probe="record") if first else saved_pass()}
        spec3 = (cake_kcp.LAST_CALL.get("bwd") or {}).get(
            "spec3"
        )  # SPEC-3 debug assertion of the first saved-mode backward
        backend = cake_kcp.get_backend()
        arms["replay"] = saved_pass(boundary_mode="replay")
        arms["repeat"] = saved_pass()
        arms["autograd"] = autograd_pass(False)
        arms["ckpt"] = autograd_pass(True)
        # the shipped default keeps the owner's c1 receive tensor for the backward (save_transfer=True); the
        # recompute form (save_transfer=False) re-sends it
        arms["save_transfer_false"] = saved_pass(save_transfer=False)
        # the forward-only workspace (the KCP forward layout, the lazy pools) beside this backward workspace: a saving
        # forward on an empty workspace holds the predicted bytes and none of the lazy pools, a forward without a
        # saved set adds their predicted bytes, and the backward upgrades the forward-only workspace; workspace=None
        # (a forward-only arena per call) through autograd too
        out_b, saved_b = cake_kcp.cake_kda_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=ws
        )
        ws_f, ws_n = (
            cake_kcp.CakeKDANondetKcpWorkspace(),
            cake_kcp.CakeKDANondetKcpWorkspace(),
        )
        out_f, saved_f = cake_kcp.cake_kda_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=ws_f
        )
        out_n, _ = cake_kcp.cake_kda_nondet_train_kcp_forward(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            workspace=ws_n,
            save_for_backward=False,
        )
        pred_f = cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
            ctx, H=H, backward=False
        )
        lazy_f = cake_kcp_plan.kcp_arena_layout(ctx, H=H, backward=False)["lazy"]
        fwd_only_bytes = (
            ws_f.bytes == pred_f
            and not any(ws_f.has(n) for n in cake_kcp_plan.FWD_LAZY_POOLS)
            and ws_n.bytes == pred_f + lazy_f
            and (ws_f.local is None or ws_f.local.fwd_layout == "fwd_kcp")
        )
        # without save_transfer a forward with a saved set reads the c1 receive and transfer staging pools (that
        # option's first state) and a forward without a saved set then adds the carry plane; the ring exchange
        # (below) never allocates the c1 receive pool
        ws_t = cake_kcp.CakeKDANondetKcpWorkspace()
        out_t, _ = cake_kcp.cake_kda_nondet_train_kcp_forward(
            q, k, v, g, beta, ctx, workspace=ws_t, save_transfer=False
        )
        pred_t = cake_kcp_plan.cake_kda_nondet_train_kcp_workspace_bytes(
            ctx, H=H, backward=False, save_transfer=False
        )
        lay_t = cake_kcp_plan.kcp_arena_layout(
            ctx, H=H, backward=False, save_transfer=False
        )
        fwd_only_bytes = (
            fwd_only_bytes and ws_t.bytes == pred_t and not ws_t.has("carry")
        )
        cake_kcp.cake_kda_nondet_train_kcp_forward(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            workspace=ws_t,
            save_for_backward=False,
        )
        fwd_only_bytes = (
            fwd_only_bytes and ws_t.bytes == pred_t + lay_t["lazy"] == pred_f + lazy_f
        )
        # the carry plane slots the forward never writes (the extra slots beyond n_in when n_out > n_in, the slots of
        # a zero-length fragment) hold allocator bytes: NaN in them must not reach a gradient (the judge's contract:
        # finite wherever the reference is).  Each rank's NaN bytes go to the report (0 where the plan leaves no slot
        # unwritten).  The slot invariants differ per rank: a rank that breaks one fails this flag with the reason,
        # instead of raising before the backward's collectives and leaving its peers waiting
        try:
            carry_slots(saved_b)
            carry_poisoned = poison_unwritten_carry(saved_f)
        except AssertionError as exc:
            carry_poisoned = f"carry_slots: {exc!r}"
        dq, dk, dv, dbeta, dg = cake_kcp.cake_kda_nondet_train_kcp_backward(
            q, k, v, g, beta, ctx, dout, saved=saved_f, workspace=ws_f
        )
        arms["forward"] = {"out": out_b}
        arms["fwd_only"] = {
            "out": out_f,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "dg": dg,
            "dbeta": dbeta,
        }
        arms["fwd_only.no_saved_set"] = {"out": out_n}
        arms["fwd_only.save_transfer_false"] = {"out": out_t}
        saved_flags_eq = saved_f.flags() == saved_b.flags()
        del ws_f, ws_n, ws_t, saved_b, saved_f
        arms["per_call"] = autograd_pass(False, None)
        if (
            W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
        ):  # gate 5: the ring path; the ring receives no c1 tensor, so recompute form only
            arms["ring"] = saved_pass(exchange="ring", save_transfer=False)
            # a forward-only ring workspace: no c1 receive pool, the hop pools on first use
            ws_r = cake_kcp.CakeKDANondetKcpWorkspace()
            out_r, _ = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws_r, exchange="ring"
            )
            arms["ring.fwd_only"] = {"out": out_r}
            fwd_only_bytes = (
                fwd_only_bytes
                and not ws_r.has("c1_recv")
                and ws_r.bytes == pred_t - lay_t["pool.c1_recv"] + lay_t["ring"]
            )
            del ws_r
        # every rank (a rank without tokens included) captures and replays the same calls: a rank that stayed eager
        # while the others captured would run its collectives outside the graphs of its peers
        cake_kcp.graph_cache_clear()
        events, graph = [], []
        try:
            for _ in range(len(GRAPH_EVENTS)):
                graph.append(
                    {n: t.clone() for n, t in saved_pass(graph="replay").items()}
                )
                events.append(
                    [
                        str(cake_kcp.LAST_CALL["fwd"].get("graph_event")),
                        str(cake_kcp.LAST_CALL["bwd"].get("graph_event")),
                    ]
                )
            torch.cuda.synchronize(device)
        except BaseException as exc:
            # the failed call's frames still reference its cache entry and graph: release them before the cleanup
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            # a cached graph holding a NCCL collective blocks the process group's destruction
            cake_kcp.graph_cache_clear()
        arms["graph"] = graph
        flags = {
            "carry_slots": isinstance(carry_poisoned, int),
            "saved_flags_eq": saved_flags_eq,
            "fwd_only_bytes": fwd_only_bytes,
            "graph_events": events == GRAPH_EVENTS,
        }
        if first:
            # a backend with local stages whose probe was pending must have run it (spec3_probe="record" on the
            # first pass); a rank without tokens launches no local stage, so it has nothing to probe.  The probe's
            # row comparison (the forward's intra prep against the backward's prep program) is reported in
            # spec3_by_rank, not gated
            flags["spec3"] = (
                (not backend.local_stages)
                or not pending
                or (spec3 is None and q.shape[0] == 0)
                or bool(spec3 and spec3.get("checked"))
            )
        return arms, flags, (spec3, events, carry_poisoned, backend)

    flags: dict = {}
    for j in range(DRAWS):
        # the same seeded operands on every rank; each takes its slices
        glob = draw_case(lengths, H, "g", False, draw_seed(JUDGE_SEED, j), device)
        arms, found, info = one_draw(_local_slices(part, rank, cu, glob), j == 0)
        if j == 0:
            spec3, events, carry_poisoned, backend = info
        for key, ok in found.items():
            flags[key] = flags.get(key, True) and bool(ok)
        judge.draw(j, glob)
        for name, results in arms.items():
            judge.arm(name, results)
        del arms, glob
    flag_list = [None] * W
    spec3_list = [None] * W
    events_list = [None] * W
    carry_list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(spec3_list, spec3)
        dist.all_gather_object(events_list, events)
        dist.all_gather_object(carry_list, carry_poisoned)
    else:
        flag_list = [flags]
        spec3_list = [spec3]
        events_list = [events]
        carry_list = [carry_poisoned]
    report = {
        "rank": rank,
        "world": W,
        "pass": True,
        "flags": {},
        "backend": backend.name,
        "backend_path": getattr(backend, "path", None),
        "spec3_by_rank": spec3_list,
        "graph_events_by_rank": events_list,
        "carry_poisoned_bytes_by_rank": carry_list,
    }
    for key in flags:
        report["flags"][key] = all(f[key] for f in flag_list)
        report["pass"] &= report["flags"][key]
    report.update(judge.report())
    if W > 1:
        verdict = [None] * W
        dist.all_gather_object(verdict, report["pass"])
        report["pass"] = all(verdict)
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _bound_rank_threads(W: int) -> None:
    """Give a spawned rank its share of the host's CPUs: each of the ``W`` ranks would
    otherwise start as many threads as the host has, and rank 0's FP64 reference then
    ran past the process group's ten-minute collective timeout while the other ranks
    waited (813 s on a four-rank row that takes 22 s with the share)."""
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // W))


def _spawn_entry(rank: int, W: int, row, rdzv: str | None, result_dir: str) -> None:
    """One spawned rank: the verdict travels through rank 0's report file (no synchronization primitive is
    pickled into the children); a failing rank raises so the parent sees its report."""
    try:
        _bound_rank_threads(W)
        report = _run_rank(rank, W, row, rdzv, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


def _rendezvous(tmp: str) -> str:
    """The rendezvous of the ranks one test spawns on this host: ``file://`` of a store in a new directory under the
    test's temporary directory ``tmp``.  A TCP port picked by binding port 0 and closing the socket can be taken by
    another process before rank 0's store listens on it (``EADDRINUSE``)."""
    return "file://" + os.path.join(
        tempfile.mkdtemp(prefix="rendezvous_", dir=tmp), "store"
    )


def _init_group(rank: int, W: int, device: torch.device, rdzv: str | None) -> None:
    """Join the ``W``-rank NCCL group on ``device``: through the spawning test's ``file://`` store
    (:func:`_rendezvous`), or for ``rdzv=None`` (one process per Slurm task) from ``MASTER_ADDR`` / ``MASTER_PORT``."""
    if rdzv is None:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    else:
        dist.init_process_group(
            "nccl", init_method=rdzv, rank=rank, world_size=W, device_id=device
        )


def _start_ranks(entry, W: int, row, tmp: str, result_dir: str) -> None:
    """``entry`` on ``W`` spawned ranks that meet through a fresh ``file://`` store under the test's temporary
    directory ``tmp`` (:func:`_rendezvous`)."""
    mp.start_processes(
        entry,
        args=(W, row, _rendezvous(tmp), result_dir),
        nprocs=W,
        join=True,
        start_method="spawn",
    )


def _skip_unless_runnable(W: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    capability = tuple(torch.cuda.get_device_capability(torch.cuda.current_device()))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip("the current device's architecture is not covered by the package")


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_multi_gpu_rows_match_the_exact_reference(record_property, row) -> None:
    """One row on ``W`` ranks (:func:`_run_rank`): every rank's plumbing flags, and rank 0's verdicts of every form's
    ``out`` and gradients of the whole pack against the FP64 reference beside the naive BF16 tier over 32 draws,
    recorded and settled here (in the one-process-per-Slurm-task form by the process of rank 0)."""
    lengths, H, W, layout = row
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1")) == W
    ):
        # one process per Slurm task: the body runs here (rendezvous from MASTER_ADDR / MASTER_PORT)
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        rank = int(os.environ["SLURM_PROCID"])
        report = _run_rank(rank, W, row, None, os.environ.get("KCP_TEST_RESULT_DIR"))
        assert report["pass"], report
        if rank == 0:
            _settle(record_property, report, lengths)
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_dist_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry, W, row, tmp, result_dir)
        with open(os.path.join(result_dir, _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
    _settle(record_property, report, lengths)


# (lengths, H, W, layout) of the host-synchronization rows: CP2 / CP4 / CP8, every layout, H 4 / 64, one-token
# documents and a document spanning >= 3 ranks; (2000, 1000) at CP4 is a ring row with a three-fragment chain (a hop-2
# chain launch)
UPLOAD_ROWS = [
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((2000, 1000), 4, 4, "contiguous"),
    ((3000,), 64, 4, "zigzag"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((700, 1, 1300, 999, 3000), 64, 8, "zigzag_per_document"),
    ((5000, 1003), 4, 8, "zigzag"),
    ((1200 + 900 + 900,), 4, 8, "contiguous"),
]
# the rows whose KDA outputs are not judged: 6,000 tokens at 64 heads is larger than the judged cases (about 4,096
# tokens per draw, one production-size case per layout up to about 16k tokens x 4 heads); they keep the exact checks,
# and the rows ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document") and ((3000,), 64, 4, "zigzag") judge the layout and the
# head count
UPLOAD_EXACT_ONLY = (((700, 1, 1300, 999, 3000), 64, 8, "zigzag_per_document"),)
# the previous copies (A/B, negative control)
OLD_PATH = {"upload": "blocking", "table_cache": "workspace"}


DIST_SLEEP_MS = 2000.0  # the queue probe of the dist census windows (the host must return within a quarter of it)


def _run_upload_rank(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None
) -> dict:
    """Host synchronization on every rank of a CP row, each window a census of ``tests/experimental/test_helpers/cake_host_sync.py``
    (upload counter, profiler copies where traced, sync-debug warnings, an event-based queue probe and a host-time
    bound), on the first draw: the first call of a fresh context (with the reused workspace and with
    ``workspace=None``) issues only pinned copies (at least one: its table sets) and does not wait; steady-state calls
    issue no copy and do not wait; the previous path's calls and preparation copy from pageable memory and wait; a
    pack prepared behind queued work (``heads=H``) issues one pinned copy and does not wait; a pack prepared from
    another thread on a held-back side stream does the same; repeated ``exchange="ring"`` calls record no further
    upload on the context (ring rows); the conv halo forward / backward (the deterministic conv package) is bitwise
    the same on every path; the plan's ``P`` of each rank is reported.  On every draw (over the judge's draws, all in
    one process group) the KDA ``out`` and gradients of each path are judged on rank 0 against the FP64 reference
    beside the naive BF16 tier, each under its own name (:class:`_RankJudge`): the previous path (``previous.*``) and
    the new one (``per_call``, ``ws``, ``replay``, ``ckpt``), the first calls of fresh contexts (``first_ws``,
    ``first_none``), the steady-state calls (``steady_ws``, ``steady_none``) and the prepared packs (``prepared``,
    ``thread_prepared``); a row of :data:`UPLOAD_EXACT_ONLY` runs the first draw only and judges nothing."""
    from cake_pplx._backends.cake_causal_conv1d_det_train import kcp as conv
    from tests.cake_naive import DRAWS, draw_seed
    from tests.experimental.test_helpers.cake_host_sync import (
        census,
        counts_agree,
        nonblocking,
        sleep_cycles,
    )
    from tests.experimental.test_helpers.cake_kda_naive import draw_case

    lengths, H, _w, layout = row
    judged = row not in UPLOAD_EXACT_ONLY
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    gen = torch.Generator(device="cpu").manual_seed(0x7815)
    C = 256
    xg = torch.randn(cu[-1], C, generator=gen).to(torch.bfloat16).to(device)
    dyg = torch.randn(cu[-1], C, generator=gen).to(torch.bfloat16).to(device)
    w = (0.5 * torch.randn(C, 4, generator=gen)).to(device).contiguous()
    loc = _local_slices(part, rank, cu, {"x": xg, "dy": dyg})
    x, dy = loc["x"].contiguous(), loc["dy"].contiguous()
    judge = _RankJudge(part, cu, W, rank, SCALE)

    def halo(c) -> list:
        y, hist = conv.cake_causal_conv1d_det_train_kcp_forward(x, w, c)
        return [y, *conv.cake_causal_conv1d_det_train_kcp_backward(x, w, c, dy, hist)]

    def eq(a, b) -> bool:
        return len(a) == len(b) and all(
            torch.equal(s, t) for s, t in zip(a, b, strict=True)
        )

    def build(**kw):
        return cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
            part, group, rank, device, **kw
        )

    def brief(c: dict) -> dict:
        keys = ("pageable", "pinned", "prof_pageable", "prof_pinned", "sync_warn")
        return {
            **{key: c[key] for key in keys},
            "busy": c["busy"],
            "host_s": round(c["host_s"], 4),
        }

    def fresh(c):
        return c, not c.ready.query()

    def one_draw(ops: dict, first: bool):
        """Every path of one draw on this rank's slices ``ops``: the KDA arms and, on the first draw, the census
        windows' flags and counts."""
        q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))

        def run(c, ws, **opts) -> dict:
            out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g, beta, c, workspace=ws, **opts
            )
            # ``boundary_mode`` is a forward keyword
            bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
            dq, dk, dv, dbeta, dg = cake_kcp.cake_kda_nondet_train_kcp_backward(
                q,
                k,
                v,
                g,
                beta,
                c,
                dout,
                saved=saved,
                workspace=ws,
                **bwd_opts,
            )
            return {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}

        def ckpt(c, ws) -> dict:
            from torch.utils.checkpoint import checkpoint

            leaves = [
                t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)
            ]
            out = checkpoint(
                lambda *a: cake_kcp.cake_kda_nondet_kcp(
                    *a, c, None, "light", "saved", ws
                ),
                *leaves,
                use_reentrant=False,
            )
            out.backward(dout)
            return {
                "out": out.detach(),
                **{
                    f"d{n}": lf.grad
                    for n, lf in zip(("q", "k", "v", "g", "beta"), leaves, strict=True)
                },
            }

        def window(fn, **kw) -> dict:
            if first:
                return census(fn, device, sleep_ms=DIST_SLEEP_MS, **kw)
            return {"out": fn()}

        arms: dict = {}
        old = build(_kcp_test_knobs=dict(OLD_PATH))
        ctx = build(heads=H)
        ws, ws_old = (
            cake_kcp.CakeKDANondetKcpWorkspace(),
            cake_kcp.CakeKDANondetKcpWorkspace(),
        )
        P = cake_kcp.CakeKDANondetKcpWorkspace(local=False).tables(ctx, H).P
        arms["previous.per_call"] = run(old, None)
        arms["previous.ws"] = run(old, ws_old)
        arms["previous.replay"] = run(old, None, boundary_mode="replay")
        arms["previous.ckpt"] = ckpt(old, None)
        halo_old = halo(old)
        arms["per_call"] = run(ctx, None)
        arms["ws"] = run(ctx, ws)
        arms["replay"] = run(ctx, ws, boundary_mode="replay")
        arms["ckpt"] = ckpt(ctx, ws)
        halo_new = halo(ctx)
        for _ in range(2):
            run(ctx, ws)
            run(ctx, None)
        steady_ws = window(lambda: run(ctx, ws))
        steady_none = window(lambda: (run(ctx, None), halo(ctx)))
        old_calls = window(lambda: run(old, None)) if first else None
        # the first call of a fresh context: its table sets are uploaded by the call
        ctx_f = build(heads=H)
        first_ws = window(lambda: (run(ctx_f, ws), halo(ctx_f)))
        ctx_g = build(heads=H)
        first_none = window(lambda: (run(ctx_g, None), halo(ctx_g)))
        # the next pack prepared behind queued work, and the previous path's preparation
        prep = window(lambda: fresh(build(heads=H)))
        ctx_b, prep_pending = prep["out"]
        prep_old = (
            window(lambda: build(heads=H, _kcp_test_knobs=dict(OLD_PATH)))
            if first
            else None
        )
        res_b, halo_b = run(ctx_b, ws), halo(ctx_b)
        # prepared from another thread on a side stream held back by ~0.5 s of its own work
        side = torch.cuda.Stream(device)

        def hold() -> None:
            with torch.cuda.stream(side):
                torch.cuda._sleep(sleep_cycles(device, 500.0))

        if first:
            thread_prep = window(
                lambda: fresh(build(heads=H, stream=side)), in_thread=True, setup=hold
            )
        else:
            thread_prep = window(lambda: fresh(build(heads=H, stream=side)))
        ctx_t, held = thread_prep["out"]
        res_t, halo_t = run(ctx_t, None), halo(ctx_t)
        arms["steady_ws"] = steady_ws["out"]
        arms["steady_none"] = steady_none["out"][0]
        arms["first_ws"] = first_ws["out"][0]
        arms["first_none"] = first_none["out"][0]
        arms["prepared"] = res_b
        arms["thread_prepared"] = res_t
        if not first:
            return arms, None
        # the ring exchange: its hop chains are uploaded once per table set, not per call
        ring_uploads, ring_chains = None, None
        if W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS:
            ws_r = cake_kcp.CakeKDANondetKcpWorkspace()
            run(ctx, ws_r, exchange="ring", save_transfer=False)
            n1 = len(ctx._uploads)
            for _ in range(3):
                run(ctx, ws_r, exchange="ring", save_transfer=False)
            ring_uploads = (n1, len(ctx._uploads))
            ring_chains = sorted(
                cake_kcp._ring_tables(
                    cake_kcp.CakeKDANondetKcpWorkspace(local=False).tables(ctx, H)
                ).chains
            )
        torch.cuda.synchronize(device)
        flags = {
            "halo_new_eq_previous_path": eq(halo_new, halo_old),
            "first_call_reused_ws_nonblocking": nonblocking(first_ws, DIST_SLEEP_MS)
            and first_ws["pinned"] >= 1
            and eq(first_ws["out"][1], halo_new),
            "first_call_no_ws_nonblocking": nonblocking(first_none, DIST_SLEEP_MS)
            and first_none["pinned"] >= 1
            and eq(first_none["out"][1], halo_new),
            "steady_reused_ws_nonblocking": nonblocking(steady_ws, DIST_SLEEP_MS)
            and steady_ws["pinned"] == 0,
            "steady_no_ws_nonblocking": nonblocking(steady_none, DIST_SLEEP_MS)
            and steady_none["pinned"] == 0
            and eq(steady_none["out"][1], halo_new),
            "previous_path_calls_block": old_calls["pageable"] > 0
            and counts_agree(old_calls)
            and old_calls["sync_warn"] > 0
            and not old_calls["busy"],
            "prepare_nonblocking": nonblocking(prep, DIST_SLEEP_MS)
            and prep["pinned"] == 1
            and prep_pending,
            "previous_path_prepare_blocks": prep_old["pageable"] > 0
            and counts_agree(prep_old)
            and prep_old["sync_warn"] > 0
            and not prep_old["busy"],
            "prepared_pack_halo_eq": eq(halo_b, halo_new),
            "thread_prepare_nonblocking": nonblocking(thread_prep, DIST_SLEEP_MS)
            and thread_prep["pinned"] == 1
            and held,
            "thread_prepared_pack_halo_eq": eq(halo_t, halo_new),
            "ring_uploads_bounded": ring_uploads is None
            or ring_uploads[0] == ring_uploads[1],
        }
        counts = {
            "P": P,
            "ring_uploads": ring_uploads,
            "ring_chains": ring_chains,
            "first_ws": brief(first_ws),
            "first_none": brief(first_none),
            "steady_ws": brief(steady_ws),
            "steady_none": brief(steady_none),
            "old_calls": brief(old_calls),
            "prep": brief(prep),
            "prep_old": brief(prep_old),
            "thread_prep": brief(thread_prep),
        }
        return arms, (flags, counts)

    prev = cake_kcp_plan.enable_test_knobs(True)
    try:
        sleep_cycles(device, DIST_SLEEP_MS)  # calibrated before any window
        for j in range(DRAWS if judged else 1):
            # the same seeded operands on every rank; each takes its slices
            glob = draw_case(lengths, H, "g", False, draw_seed(JUDGE_SEED, j), device)
            arms, found = one_draw(_local_slices(part, rank, cu, glob), j == 0)
            if j == 0:
                flags, counts = found
            if judged:
                judge.draw(j, glob)
                for name, results in arms.items():
                    judge.arm(name, results)
            del arms, glob
        torch.cuda.synchronize(device)
    finally:
        cake_kcp_plan.enable_test_knobs(prev)
    if W > 1:
        flag_list, count_list = [None] * W, [None] * W
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(count_list, counts)
    else:
        flag_list, count_list = [flags], [counts]
    report = {
        "rank": rank,
        "world": W,
        "row": _row_id(row),
        "flags": {},
        "counts_by_rank": count_list,
        "judged": judged,
    }
    for key in flags:
        report["flags"][key] = all(f[key] for f in flag_list)
    report["pass"] = all(report["flags"].values())
    if judged:
        report.update(judge.report())
    if W > 1:
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "upload-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_upload_entry(
    rank: int, W: int, row, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_upload_rank(rank, W, row, rdzv, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("row", UPLOAD_ROWS, ids=_row_id)
def test_multi_gpu_upload_rows_are_judged_and_nonblocking(record_property, row) -> None:
    """The rows of :func:`_run_upload_rank`: one process per GPU (``spawn``, rows up to the visible GPU count) or
    one process per Slurm task (a step with ``SLURM_NTASKS`` > 1 runs the rows of its own world size and skips the
    others).  Every rank's flags hold; rank 0's verdicts of each path (a row of :data:`UPLOAD_EXACT_ONLY` aside) are
    recorded and settled here (in the one-process-per-Slurm-task form by the process of rank 0)."""
    lengths, H, W, layout = row
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"this Slurm step runs the W = {ntasks} rows")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        rank = int(os.environ["SLURM_PROCID"])
        report = _run_upload_rank(
            rank, W, row, None, os.environ.get("KCP_TEST_RESULT_DIR")
        )
        assert report["pass"], report
        if rank == 0 and report["judged"]:
            _settle(record_property, report, lengths)
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_upload_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_upload_entry, W, row, tmp, result_dir)
        with open(os.path.join(result_dir, "upload-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
    if report["judged"]:
        _settle(record_property, report, lengths)


def test_public_api_cannot_select_a_precision_mode(monkeypatch) -> None:
    """The private test hook is refused without the environment variable; the public options never reach K3's
    non-shipped programs (SPEC-11)."""
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 2}})
    with pytest.raises(TypeError):
        cake_kcp._options({"precision_mode": 1})
    assert cake_kcp._options({}).precision_mode == 0
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", True)
    assert (
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 1}}).precision_mode
        == 1
    )
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 7}})


def test_public_api_cannot_select_a_transfer_walk(monkeypatch) -> None:
    """The ``transfer_walk`` knob (K3's ``tf32_hilo`` / ``bf16x3`` programs of ``build.test.json``) is reachable only through
    the gated test hook; the public options never select a non-shipped transfer-walk program (SPEC-11)."""
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(TypeError):
        cake_kcp._options({"transfer_walk": "bf16x3"})
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"transfer_walk": "bf16x3"}})
    assert cake_kcp._options({}).transfer_walk == "bf16x3_hw64"
    monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", True)
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"transfer_walk": "bf16x3"}}
        ).transfer_walk
        == "bf16x3"
    )
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"transfer_walk": "bf16x3_hw64"}}
        ).transfer_walk
        == "bf16x3_hw64"
    )
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"transfer_walk": "tf32_hilo"}}
        ).transfer_walk
        == "tf32_hilo"
    )
    assert (
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 1}}).transfer_walk
        == "tf32_hilo"
    )
    assert (
        cake_kcp._options(
            {"_kcp_test_knobs": {"precision_mode": 1, "transfer_walk": "tf32_hilo"}}
        ).transfer_walk
        == "tf32_hilo"
    )
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"transfer_walk": "fp64"}})
    with pytest.raises(
        ValueError
    ):  # a K3 precision_mode program belongs to the tf32_hilo family
        cake_kcp._options(
            {"_kcp_test_knobs": {"precision_mode": 1, "transfer_walk": "bf16x3"}}
        )
    with pytest.raises(
        ValueError
    ):  # the shipped BF16x3 walk named beside it is refused too
        cake_kcp._options(
            {"_kcp_test_knobs": {"precision_mode": 2, "transfer_walk": "bf16x3_hw64"}}
        )


def test_module_imports_only_torch_and_its_package() -> None:
    """The design note §11.3: the host module imports torch, torch.distributed, tvm_ffi and its own package
    (plus ``ctypes`` for the CUDA runtime's device cache-config preference around the tape launches,
    ``cake_kcp.walk_device_cache_config``; the same ``ctypes`` import also reaches the capture-graph entry points of the
    replay cap of record
    ``graph_cap="node_attr"`` -- ``cudaStreamGetCaptureInfo`` (``_v2`` / ``_v3``), ``cudaGraphNodeGetType``,
    ``cudaGraphKernelNodeSetAttribute``, ``cudaGraphKernelNodeGetAttribute``, ``cudaRuntimeGetVersion``,
    ``cudaGetLastError``; no new module; the trimmed host path:
    ``functools.cached_property`` on the host table set (``walk_all_p0`` / ``walk_all_rest``, host ints only) -- stdlib,
    no new module either; ``weakref`` for the trimmed
    path's begin_call template cache, owned by the local arena and weakly keyed by the table set -- stdlib, no new
    module; ``torch.utils`` for the checkpoint hand-off, which reads the pack-hook classes of
    ``torch.utils.checkpoint`` once (``cake_kcp._handoff_frames_readable``) -- torch, no new package;
    ``operator.itemgetter`` for the trimmed local launch plan's argument fetch -- stdlib, no new module;
    ``sys.modules`` for the backward's mode check, which recognizes the deterministic package's workspace type
    without importing that package -- stdlib, no new module); ``importlib`` in ``_single_rank``, through which it
    imports its single-rank sibling by name (README "Vendoring")."""
    import ast

    package = Path(cake_kcp.__file__).parent
    allowed = {
        "__future__",
        "dataclasses",
        "math",
        "os",
        "types",
        "typing",
        "collections",
        "collections.abc",
        "torch",
        "torch.distributed",
        "torch.utils",
        "tvm_ffi",
        "ctypes",
        "warnings",
        "functools",
        "inspect",  # inspect.signature: the entry points' signatures state their keyword options (stdlib)
        "weakref",
        "operator",
        "sys",  # sys.modules: the deterministic package's types are found without importing that package
        "importlib",
    }  # importlib: _single_rank resolves the single-rank sibling package by name (README "Vendoring"); operator: itemgetter of the trimmed local launch plan's arguments; weakref: the arena-owned template cache of the trimmed host path; functools: cached_property of the trimmed host path; ctypes: the cache-config bracket of the walk-launch geometry and the capture-graph entry points of the replay cap; warnings: the one RuntimeWarning without a loadable CUDA runtime or for a degraded node cap; torch.utils: the checkpoint hand-off reads the pack-hook classes of torch.utils.checkpoint once; no other use
    for name in ("cake_kcp.py", "_single_rank.py"):
        for node in ast.walk(ast.parse((package / name).read_text())):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name in allowed, (name, a.name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module in allowed, (name, node.module)


# ---------------------------------------------------------------------------------------------------------------
# Rank agreement: the one-time check of a call key's collective-relevant metadata (``cake_kcp._rank_check``).  CPU:
# the digest's inputs (``cake_kcp_plan.kcp_rank_digest``), the check's once-per-key cache, its skips and its device
# assert (a fake ``torch.distributed``), and where the real drivers issue it (before the first collective of every
# path).  Two GPUs: matching ranks pass with the checks counted exactly; a mismatch trips the device-side assert in
# the spawned ranks instead of hanging.

# one value of every digest field (cake_kcp_plan.RANK_CHECK_FIELDS, in order) and another value of each
RANK_FIELDS = {
    "H": 4,
    "exchange": "a2a",
    "save_transfer": True,
    "boundary": "saved",
    "debug_sequence": False,
    "recompute_handle": False,
    "gate": "g",
    "qk_l2norm": False,
}
RANK_FIELD_CHANGES = {
    "H": 64,
    "exchange": "ring",
    "save_transfer": False,
    "boundary": "replay",
    "debug_sequence": True,
    "recompute_handle": True,
    "gate": "raw",
    "qk_l2norm": True,
}


def _cpu_context(lengths, W: int, layout: str, rank: int = 0, *, sm: int = 148, **kw):
    part, _ = _partition(lengths, W, layout)
    return cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, None, rank, "cpu", sm_count=sm, **kw
    )


def _words(digest: bytes) -> list[int]:
    """The two int64 words the check gathers for a 16-byte digest."""
    return [
        int.from_bytes(digest[:8], "little", signed=True),
        int.from_bytes(digest[8:], "little", signed=True),
    ]


def test_rank_digest_covers_the_collective_inputs_only() -> None:
    """``kcp_partition_digest`` / ``kcp_rank_digest``: BLAKE2b-128 of canonical JSON (no per-process state); equal for
    equal partitions held by other objects, different for another layout, world size, document split or fragment
    table, for another ``conv_kernel_size`` and for another value of each field; the rank-local inputs (the rank, the
    SM count and the plan rule, hence ``P``) stay out: two ranks of one partition with ``P = 1`` and ``P > 1`` give
    the same digest; malformed inputs are refused."""
    kp = cake_kcp_plan
    assert tuple(RANK_FIELDS) == kp.RANK_CHECK_FIELDS == tuple(RANK_FIELD_CHANGES)
    part_a, _ = _partition((3000,), 2, "contiguous")
    part_b, _ = _partition((3000,), 2, "contiguous")
    assert part_a is not part_b
    pd = kp.kcp_partition_digest(part_a)
    assert isinstance(pd, bytes) and len(pd) == 16
    assert kp.kcp_partition_digest(part_b) == pd
    canonical = json.dumps(part_a.to_dict(), sort_keys=True, separators=(",", ":"))
    assert pd == hashlib.blake2b(canonical.encode(), digest_size=16).digest()
    others = {
        "layout": _partition((3000,), 2, "zigzag")[0],
        "world": _partition((3000,), 4, "contiguous")[0],
        "documents": _partition((1500, 1500), 2, "contiguous")[0],
        "fragments": _partition((3000,), 2, "ragged:1000,2000")[0],
    }
    digests = {name: kp.kcp_partition_digest(p) for name, p in others.items()}
    assert len({pd, *digests.values()}) == len(others) + 1, digests

    base = kp.kcp_rank_digest(pd, 4, RANK_FIELDS)
    assert isinstance(base, bytes) and len(base) == 16
    payload = {
        "version": kp.RANK_CHECK_VERSION,
        "partition": pd.hex(),
        "conv_kernel_size": 4,
        **RANK_FIELDS,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert base == hashlib.blake2b(canonical.encode(), digest_size=16).digest()
    assert (
        kp.kcp_rank_digest(bytearray(pd), 4, dict(reversed(RANK_FIELDS.items())))
        == base
    )
    changed = {
        name: kp.kcp_rank_digest(pd, 4, {**RANK_FIELDS, name: value})
        for name, value in RANK_FIELD_CHANGES.items()
    }
    changed["conv_kernel_size"] = kp.kcp_rank_digest(pd, 3, RANK_FIELDS)
    for name, p in others.items():
        changed[f"partition.{name}"] = kp.kcp_rank_digest(
            kp.kcp_partition_digest(p), 4, RANK_FIELDS
        )
    assert len({base, *changed.values()}) == len(changed) + 1, changed

    # rank-local inputs: rank 0 at 148 SMs under the "occ" plan rule, rank 1 at one SM under the shipped rule
    previous = kp.enable_test_knobs(True)
    try:
        c0 = kp.cake_kda_nondet_train_kcp_build_context(
            part_a, None, 0, "cpu", sm_count=148, _kcp_test_knobs={"p_rule": "occ"}
        )
    finally:
        kp.enable_test_knobs(previous)
    c1 = kp.cake_kda_nondet_train_kcp_build_context(part_b, None, 1, "cpu", sm_count=1)
    assert (c0.rank, c1.rank, c0.sm_count, c1.sm_count) == (0, 1, 148, 1)
    assert c0.p_rule == "occ" != c1.p_rule
    assert c0.heads(4).P > 1 == c1.heads(4).P, (c0.heads(4).P, c1.heads(4).P)
    r0, r1 = cake_kcp._RankChecks(c0), cake_kcp._RankChecks(c1)
    assert r0.partition == r1.partition == pd and r0.done == r1.done == {}
    assert int(c0.conv_kernel_size) == int(c1.conv_kernel_size) == 4

    def call(ctx, **local):
        # the check reads the fields only: the rank-local attributes beside them differ freely
        return types.SimpleNamespace(
            ctx=ctx,
            H=4,
            exchange="a2a",
            opts=types.SimpleNamespace(save_transfer=True, debug_sequence=False),
            with_handle=False,
            gate="g",
            l2n=False,
            **local,
        )

    f0 = cake_kcp._rank_check_fields(
        call(c0, rank=0, sm_count=148, schedule="overlap", workspace=object()), "saved"
    )
    f1 = cake_kcp._rank_check_fields(
        call(c1, rank=1, sm_count=1, schedule="wait_first", workspace=None), "saved"
    )
    assert f0 == f1 == RANK_FIELDS and tuple(f0) == kp.RANK_CHECK_FIELDS
    assert kp.kcp_rank_digest(r0.partition, int(c0.conv_kernel_size), f0) == base
    assert kp.kcp_rank_digest(r1.partition, int(c1.conv_kernel_size), f1) == base

    # refusals: a short or non-bytes partition digest, a missing or an extra (rank-local) field, a wrong host type
    with pytest.raises(ValueError, match="16 bytes"):
        kp.kcp_rank_digest(pd[:8], 4, RANK_FIELDS)
    with pytest.raises(ValueError, match="16 bytes"):
        kp.kcp_rank_digest(pd.hex(), 4, RANK_FIELDS)
    with pytest.raises(ValueError, match="exactly"):
        kp.kcp_rank_digest(pd, 4, {k: v for k, v in RANK_FIELDS.items() if k != "gate"})
    with pytest.raises(ValueError, match="exactly"):
        kp.kcp_rank_digest(pd, 4, {**RANK_FIELDS, "sm_count": 148})
    for name, bad in (("H", True), ("H", 4.0), ("save_transfer", 1), ("gate", None)):
        with pytest.raises(TypeError, match=name):
            kp.kcp_rank_digest(pd, 4, {**RANK_FIELDS, name: bad})
    with pytest.raises(TypeError, match="CakeKDANondetKcpPartition"):
        kp.kcp_partition_digest(part_a.to_dict())


class _FakeGather:
    """``torch.distributed`` as :func:`cake_kcp._rank_check` sees it: a group of ``world`` ranks (this process rank
    0), initialized or not, whose all-gather fills every peer row with ``peer`` (``None``: this rank's own words) and
    logs ``(output shape, output dtype, input copy, group, async_op)``."""

    def __init__(self, world: int = 2) -> None:
        self.world, self.initialized, self.available = world, True, True
        self.peer: list[int] | None = None
        self.log: list[tuple] = []

    def is_available(self) -> bool:
        return self.available

    def is_initialized(self) -> bool:
        return self.initialized

    def all_gather_into_tensor(self, out, inp, group=None, async_op=False):
        self.log.append((tuple(out.shape), out.dtype, inp.clone(), group, async_op))
        rows = out.view(self.world, -1)
        rows[0].copy_(inp)
        for r in range(1, self.world):
            rows[r].copy_(
                inp if self.peer is None else torch.tensor(self.peer, dtype=inp.dtype)
            )
        return None


def _check_call(ctx, *, world: int = 2, **fields) -> types.SimpleNamespace:
    """A call as :func:`cake_kcp._rank_check` reads it, with the digest fields ``RANK_FIELDS`` updated by ``fields``
    (``boundary`` is the check's argument: kept as ``boundary``)."""
    f = {**RANK_FIELDS, **fields}
    return types.SimpleNamespace(
        ctx=ctx,
        world=world,
        group="group",
        device=torch.device("cpu"),
        H=f["H"],
        exchange=f["exchange"],
        opts=types.SimpleNamespace(
            save_transfer=f["save_transfer"], debug_sequence=f["debug_sequence"]
        ),
        with_handle=f["recompute_handle"],
        gate=f["gate"],
        l2n=f["qk_l2norm"],
        boundary=f["boundary"],
        scratch={},
    )


def test_rank_check_is_once_per_key_and_device_side(monkeypatch) -> None:
    """:func:`cake_kcp._rank_check` on a fake two-rank group: the first call of a (context, key) all-gathers the two
    int64 words of the key's digest on the call's group (``async_op=False``: ordered on the stream, no work handle),
    records the digest and counts it; the same key again issues nothing, every field and another context of an equal partition issue a
    new check; one rank, an uninitialized group, no ``torch.distributed`` and a CUDA-graph capture issue nothing (the
    capture leaves the key unchecked: the next eager call checks it); the per-context state dies with the context;
    a peer with another digest trips the assert (raised on the CPU, a device-side assert on CUDA)."""
    kp = cake_kcp_plan
    fake = _FakeGather()
    monkeypatch.setattr(cake_kcp, "dist", fake)
    stats = cake_kcp.RANK_CHECK_STATS
    base = stats["issued"]

    def check(ctx, **kw):
        c = _check_call(ctx, **kw)
        cake_kcp._rank_check(c, c.boundary)
        return c

    part, _ = _partition((3000,), 2, "contiguous")
    ctx = kp.cake_kda_nondet_train_kcp_build_context(part, None, 0, "cpu", sm_count=148)
    pd = kp.kcp_partition_digest(part)
    want = kp.kcp_rank_digest(pd, int(ctx.conv_kernel_size), RANK_FIELDS)
    first = check(ctx)
    assert len(fake.log) == 1 and stats["issued"] == base + 1
    shape, dtype, sent, group, async_op = fake.log[0]
    assert (shape, dtype, group, async_op) == ((4,), torch.int64, "group", False)
    assert sent.dtype == torch.int64 and sent.tolist() == _words(want)
    assert first.scratch["rank_check"] == want.hex()
    assert cake_kcp._RANK_CHECKS[ctx].done == {tuple(RANK_FIELDS.values()): want}
    # the same key again, from another call object: nothing
    again = check(ctx)
    assert len(fake.log) == 1 and stats["issued"] == base + 1
    assert "rank_check" not in again.scratch
    # every field is part of the key: one check each, each with its own digest
    for name, value in RANK_FIELD_CHANGES.items():
        n = len(fake.log)
        c = check(ctx, **{name: value})
        d = kp.kcp_rank_digest(pd, 4, {**RANK_FIELDS, name: value})
        assert len(fake.log) == n + 1 and c.scratch["rank_check"] == d.hex(), name
        assert fake.log[-1][2].tolist() == _words(d), name
    assert stats["issued"] == base + 1 + len(RANK_FIELD_CHANGES)
    assert len(cake_kcp._RANK_CHECKS[ctx].done) == 1 + len(RANK_FIELD_CHANGES)
    # per context: another context of an equal partition checks the key again, with the same digest
    ctx2 = _cpu_context((3000,), 2, "contiguous")
    n = len(fake.log)
    assert check(ctx2).scratch["rank_check"] == want.hex() and len(fake.log) == n + 1

    # nothing on one rank, without an initialized group or without torch.distributed (a fresh context: its first
    # call would check)
    ctx3 = _cpu_context((3000,), 2, "contiguous")
    n, issued = len(fake.log), stats["issued"]
    assert "rank_check" not in check(ctx3, world=1).scratch
    fake.initialized = False
    assert "rank_check" not in check(ctx3).scratch
    fake.initialized, fake.available = True, False
    assert "rank_check" not in check(ctx3).scratch
    fake.available = True
    # inside a CUDA-graph capture: nothing, and the key stays unchecked
    capturing = cake_kcp._capturing
    monkeypatch.setattr(cake_kcp, "_capturing", lambda: True)
    assert "rank_check" not in check(ctx3).scratch
    monkeypatch.setattr(cake_kcp, "_capturing", capturing)
    assert len(fake.log) == n and stats["issued"] == issued
    assert ctx3 not in cake_kcp._RANK_CHECKS
    assert check(ctx3).scratch["rank_check"] == want.hex() and len(fake.log) == n + 1

    # the per-context state is weak: it dies with the context, and it holds no reference to it
    def checked_context() -> weakref.ref:
        c = _cpu_context((3000,), 2, "contiguous")
        check(c, boundary="replay")
        assert c in cake_kcp._RANK_CHECKS
        assert not any(r is c for r in gc.get_referents(cake_kcp._RANK_CHECKS[c]))
        return weakref.ref(c)

    ref = checked_context()
    gc.collect()
    assert ref() is None
    assert isinstance(cake_kcp._RANK_CHECKS, weakref.WeakKeyDictionary)

    # a peer with another digest: the assert trips (synchronously on the CPU; on CUDA a device-side assert)
    fake.peer = [w ^ 1 for w in _words(want)]
    n = len(fake.log)
    with pytest.raises(RuntimeError, match="disagree"):
        check(ctx3, boundary="replay")
    assert len(fake.log) == n + 1


class _IssueCall(types.SimpleNamespace):
    """A call for the real :func:`cake_kcp._forward` / :func:`cake_kcp._backward` whose launches, merges, buffer
    views and waits do nothing; its collectives (``a2a:<tag>``) are logged in ``log``, beside the patched
    rank-agreement check (``check:<boundary>``) and ring exchanges (``ring:fwd`` / ``ring:bwd``)."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None

    def a2a(self, tag: str, *a, **k) -> str:
        self.log.append(f"a2a:{tag}")
        return tag


# the order of the check and the collectives on every driver path (``fwd_bound``: a recompute that binds a handle's
# states; ``bwd_replay``: the replay-mode backward, whose forward replay checks the backward's key)
ISSUE_ORDER = {
    "a2a": {
        "fwd_save": ["check:saved", "a2a:c1", "a2a:c2"],
        "fwd_nosave": ["check:replay", "a2a:c1", "a2a:c2"],
        "fwd_bound": [],
        "bwd_saved": ["check:saved", "a2a:b1", "a2a:b2"],
        "bwd_saved_transfer": ["check:saved", "a2a:b1", "a2a:b2"],
        "bwd_replay": ["check:replay", "a2a:c1", "a2a:c2", "a2a:b1", "a2a:b2"],
    },
    "ring": {
        "fwd_save": ["check:saved", "ring:fwd"],
        "fwd_nosave": ["check:replay", "ring:fwd"],
        "fwd_bound": [],
        "bwd_saved": ["check:saved", "ring:bwd"],
        "bwd_replay": ["check:replay", "ring:fwd", "ring:bwd"],
    },
}


@pytest.mark.parametrize("schedule", ["wait_first", "overlap"])
@pytest.mark.parametrize("sm", [1, 148], ids=["P1", "P_gt1"])
def test_rank_check_issue_points(monkeypatch, schedule: str, sm: int) -> None:
    """The real drivers issue the check once per call right before its first collective (c1 / ring hop 1 of a
    forward, b1 / ring hop 1 of a saved-mode backward, the replay forward's c1 / hop 1 of a replay-mode backward) with
    the boundary of the call's key, and none in a recompute that binds a handle's states: both exchanges, both
    schedules, ``P = 1`` and ``P > 1`` (rank 1 of a two-rank contiguous split, crossing-in).  The launches, the
    buffers and the ring exchange are stubbed (CPU)."""
    H = 4
    ctx = _cpu_context((3000,), 2, "contiguous", rank=1, sm=sm)
    opts = cake_kcp._options({})
    tb = cake_kcp._Tables(ctx, H, fused_walks=opts.fused_walks)
    assert (tb.P > 1) == (sm > 1) and int(ctx.n_in) > 0, (tb.P, ctx.n_in)
    for name in (
        "_fwd_buffers",
        "_summaries",
        "_stage_walks",
        "_fwd_tapes",
        "_adj_tapes",
    ):
        monkeypatch.setattr(cake_kcp, name, lambda *a, **k: None)
    monkeypatch.setattr(
        cake_kcp,
        "_rank_check",
        lambda call, boundary: call.log.append(f"check:{boundary}"),
    )
    monkeypatch.setattr(
        cake_kcp, "_ring_forward", lambda call: call.log.append("ring:fwd")
    )
    monkeypatch.setattr(
        cake_kcp, "_ring_backward", lambda call: call.log.append("ring:bwd")
    )

    def saved(transfer: bool = False):
        return types.SimpleNamespace(
            S_in=torch.zeros(1),
            c1_recv=torch.zeros(1) if transfer else None,
            M_seg=None,
            carry=None,
            tape=None,
            rows=None,
        )

    paths = {
        "fwd_save": lambda c: cake_kcp._forward(c, save=True, replay=False),
        "fwd_nosave": lambda c: cake_kcp._forward(c, save=False, replay=False),
        "fwd_bound": lambda c: cake_kcp._forward(
            c, save=True, replay=False, bound={"S_in": torch.zeros(1)}
        ),
        "bwd_saved": lambda c: cake_kcp._backward(c, saved()),
        "bwd_saved_transfer": lambda c: cake_kcp._backward(c, saved(True)),
        "bwd_replay": lambda c: cake_kcp._backward(c, None),
    }
    seen = {}
    for exchange, want in ISSUE_ORDER.items():
        for path, want_log in want.items():
            call = _IssueCall(
                tb=tb,
                ctx=ctx,
                T_r=int(ctx.T_r),
                H=H,
                backward=path.startswith("bwd"),
                exchange=exchange,
                schedule=schedule,
                merge_fuse=str(opts.merge_fuse),
                host_trim=int(opts.host_trim),
                fwd_o_fuse=1,
                l2n=False,
                l2n_mode="off",
                gate="g",
                local=None,
                local_values=None,
                out=None,
                bufs={},
                scratch={},
                log=[],
            )
            paths[path](call)
            seen[(exchange, path)] = call.log
            assert call.log == want_log, (exchange, path, call.log)
    assert len(seen) == sum(len(w) for w in ISSUE_ORDER.values())


# --------------------------------------------------------------------------- two ranks (spawned)

RANK_CHECK_ROW = ((100, 640, 77, 1000, 129), 4, 2, "contiguous")
# the mismatch each rank-1 process introduces (rank 0 keeps the row's metadata); "match": none
RANK_CHECK_CASES = ("match", "partition", "heads", "save_transfer", "handle")
RANK_CHECK_DEADLINE_S = 900.0
ASSERT_MARKERS = ("device-side assert", "cudaErrorAssert", "CUDA_ERROR_ASSERT")


def _rank_check_body(rank: int, W: int, case: str, device, stage) -> dict:
    """The rank-agreement rows of one spawned rank (:func:`test_rank_check_two_ranks`).  ``stage(name)`` marks the
    progress in the rank's log.  A mismatch case returns only if its call raised nothing (the parent then fails)."""
    import contextlib

    from torch.utils.checkpoint import checkpoint

    lengths, H, _w, layout = RANK_CHECK_ROW
    group = dist.group.WORLD
    stats = cake_kcp.RANK_CHECK_STATS

    def operands(part, cu, heads):
        glob = make_inputs(lengths, heads, device)
        o = _local_slices(part, rank, cu, glob)
        return tuple(o[n] for n in ("q", "k", "v", "g", "beta", "dout"))

    def build(lay: str = layout):
        part, cu = _partition(lengths, W, lay)
        ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
            part, group, rank, device
        )
        return part, cu, ctx

    def fwd(ctx, o, ws, **opts):
        return cake_kcp.cake_kda_nondet_train_kcp_forward(
            *o[:5], ctx, workspace=ws, **opts
        )

    def bwd(ctx, o, ws, saved, **opts):
        return cake_kcp.cake_kda_nondet_train_kcp_backward(
            *o[:5], ctx, o[5], saved=saved, workspace=ws, **opts
        )

    def autograd(ctx, o, ws, handle, cc=None):
        leaves = [x.detach().clone().requires_grad_(True) for x in o[:5]]

        def fn(q_, k_, v_, g_, b_, h_):
            return cake_kcp.cake_kda_nondet_kcp(
                q_, k_, v_, g_, b_, ctx, workspace=ws, recompute_handle=h_
            )

        kw = {}
        if cc is not None:
            kw["context_fn"] = lambda: (contextlib.nullcontext(), cc.recompute_cm())
        return checkpoint(fn, *leaves, handle, use_reentrant=False, **kw)

    res: dict = {"flags": {}, "info": {}}
    flags, info = res["flags"], res["info"]
    # warm-up on a matching context (modules loaded, the communicator used): its first call checks one key
    part, cu, ctx_w = build()
    o = operands(part, cu, H)
    ws = cake_kcp.CakeKDANondetKcpWorkspace()  # one per context below
    n0 = stats["issued"]
    out, saved = fwd(ctx_w, o, ws)
    bwd(ctx_w, o, ws, saved)
    torch.cuda.synchronize(device)
    flags["warm_one_check"] = stats["issued"] - n0 == 1
    dist.barrier()
    stage("warm_done")

    if case != "match":
        # rank 1 departs from rank 0 in one input of the digest; every rank's first call of the fresh context checks
        mine = case if rank == 1 else None
        p_m, cu_m, ctx_m = build("zigzag" if mine == "partition" else layout)
        heads = 8 if mine == "heads" else H
        o_m = operands(p_m, cu_m, heads)
        ws_m = cake_kcp.CakeKDANondetKcpWorkspace()
        stage("mismatch_call")
        if case == "handle":
            hd = cake_kcp.CakeKDANondetKcpRecomputeHandle() if rank == 0 else None
            autograd(ctx_m, o_m, ws_m, hd)
        else:
            transfer = {} if case != "save_transfer" else {"save_transfer": rank == 0}
            fwd(ctx_m, o_m, ws_m, **transfer)
        torch.cuda.synchronize(device)
        stage("mismatch_returned")
        res["no_error"] = True
        return res

    # exact accounting on a fresh context: the first saved-mode step checks once (in its forward), the
    # replay-mode step its own key once; repeats check nothing; the exchange counts are the records' collectives
    part, cu, ctx_s = build()
    o = operands(part, cu, H)
    ws = cake_kcp.CakeKDANondetKcpWorkspace()
    steps = []
    for mode in ("saved", "saved", "replay", "replay", "saved"):
        with _Counter() as cc:
            out, sv = fwd(ctx_s, o, ws, boundary_mode=mode)
            rf = dict(cake_kcp.LAST_CALL["fwd"])
            cc.phase = "bwd"
            bwd(ctx_s, o, ws, sv)
            rb = dict(cake_kcp.LAST_CALL["bwd"])
        rec = cc.record()
        steps.append(
            {
                "mode": mode,
                "checks": rec["checks"],
                "gathers": rec["gathers"],
                "exchange": _exchange_counts(rec)["calls"],
                "n_collectives": [rf["n_collectives"], rb["n_collectives"]],
                "rank_check": [rf["rank_check"], rb["rank_check"]],
            }
        )
    first = {"orig": 1, "recompute": 0, "bwd": 0}
    want = [first, NO_CHECKS, first, NO_CHECKS, NO_CHECKS]
    flags["steps_exact"] = all(
        s["checks"] == s["gathers"] == w
        and s["exchange"]["orig"] == s["n_collectives"][0] > 0
        and s["exchange"]["bwd"] == s["n_collectives"][1] > 0
        and isinstance(s["rank_check"][0], str) == (w["orig"] == 1)
        and s["rank_check"][1] is None
        for s, w in zip(steps, want, strict=True)
    )
    info["steps"] = steps
    fields = {
        **RANK_FIELDS,
        "H": H,
        "save_transfer": bool(cake_kcp_plan.SAVE_TRANSFER_DEFAULT),
    }
    pd = cake_kcp_plan.kcp_partition_digest(part)
    digest_saved = cake_kcp_plan.kcp_rank_digest(pd, 4, fields).hex()
    digest_replay = cake_kcp_plan.kcp_rank_digest(
        pd, 4, {**fields, "boundary": "replay"}
    ).hex()
    got = [steps[0]["rank_check"][0], steps[2]["rank_check"][0]]
    flags["digests_are_the_keys"] = got == [digest_saved, digest_replay]

    # graph="replay" on a fresh context: the eager first call checks, the capture and the replays do not.  The fresh
    # workspace's pools grow in the first backward, which changes the forward's graph key (it holds the pool addresses),
    # so either kind may run eagerly more than once before its capture: two more rounds than GRAPH_EVENTS
    part, cu, ctx_g = build()
    o = operands(part, cu, H)
    ws_g = cake_kcp.CakeKDANondetKcpWorkspace()
    cake_kcp.graph_cache_clear()
    deltas, events = [], []
    try:
        for _ in range(len(GRAPH_EVENTS) + 2):
            n0 = stats["issued"]
            out, sv = fwd(ctx_g, o, ws_g, graph="replay")
            bwd(ctx_g, o, ws_g, sv, graph="replay")
            deltas.append(stats["issued"] - n0)
            events.append(
                [
                    str(cake_kcp.LAST_CALL["fwd"].get("graph_event")),
                    str(cake_kcp.LAST_CALL["bwd"].get("graph_event")),
                ]
            )
        torch.cuda.synchronize(device)
    finally:
        cake_kcp.graph_cache_clear()
    flags["graph_exact"] = (
        deltas == [1] + [0] * (len(deltas) - 1)
        and events[0] == ["eager", "eager"]
        and all(_eager_capture_replay([e[i] for e in events]) for i in (0, 1))
        and events[-1] == ["replay", "replay"]
    )
    info["graph"] = {"events": events, "deltas": deltas}

    # a recompute handle on a fresh context: the original forward checks (the handle is a field), the recompute
    # binds and issues nothing, the backward checks its own key (states tier: save_transfer off, no handle)
    part, cu, ctx_h = build()
    o = operands(part, cu, H)
    ws = cake_kcp.CakeKDANondetKcpWorkspace()
    with _Counter() as cc:
        out = autograd(ctx_h, o, ws, cake_kcp.CakeKDANondetKcpRecomputeHandle(), cc)
        ro = cake_kcp.LAST_CALL.pop("fwd")
        cc.phase = "bwd"
        out.backward(o[5])
        rr = cake_kcp.LAST_CALL.pop("fwd", None)
    rec = cc.record()
    flags["handle_exact"] = (
        rec["checks"] == rec["gathers"] == {"orig": 1, "recompute": 0, "bwd": 1}
        and rec["calls"]["recompute"] == 0
        and ro["recompute_handle"] == "capture"
        and ro["rank_check"]
        == cake_kcp_plan.kcp_rank_digest(
            pd, 4, {**fields, "save_transfer": False, "recompute_handle": True}
        ).hex()
        and rr is not None
        and rr["recompute_handle"] == "replay"
        and rr["rank_check"] is None
        and rr["n_collectives"] == 0
        and not any(s.startswith(("a2a:", "ring:")) for s in rr["sequence"])
    )
    info["handle"] = {k: rec[k] for k in ("calls", "checks", "gathers")}

    # every rank recorded the same digests; no assert tripped on the device
    mine = [steps[0]["rank_check"][0], steps[2]["rank_check"][0], ro["rank_check"]]
    every = [None] * W
    dist.all_gather_object(every, mine)
    flags["ranks_agree"] = all(e == mine for e in every)
    torch.cuda.synchronize(device)
    dist.barrier()
    stage("match_done")
    return res


def _rank_check_entry(rank: int, W: int, case: str, rdzv: str, out_dir: str) -> None:
    """One spawned rank of :func:`test_rank_check_two_ranks`: stdout / stderr (the device assert's report, NCCL's
    and the watchdog's) go to ``rank<r>.log``, the result to ``rank<r>.json``; a mismatch exits at once after its
    error (a group with a failed device stays undestroyed)."""
    import datetime

    fd = os.open(
        os.path.join(out_dir, f"rank{rank}.log"),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o644,
    )
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    result: dict = {"rank": rank, "case": case, "stages": []}

    def stage(name: str) -> None:
        result["stages"].append(name)
        print(f"RANK_CHECK_STAGE {rank} {name}", flush=True)

    def dump() -> None:
        # write then rename: after the trap the NCCL watchdog may abort this process at any point, and a truncated
        # file must not stand for this rank's result
        path = os.path.join(out_dir, f"rank{rank}.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(result, fh, default=str)
        os.replace(path + ".tmp", path)
        sys.stdout.flush()
        sys.stderr.flush()

    try:
        _bound_rank_threads(W)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl",
            init_method=rdzv,
            rank=rank,
            world_size=W,
            device_id=device,
            timeout=datetime.timedelta(seconds=300),
        )
        stage("init")
        result.update(_rank_check_body(rank, W, case, device, stage))
    except BaseException as exc:  # every failure is the rank's result
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        dump()
        os._exit(3)
    result["pass"] = case == "match" and all(result["flags"].values())
    if case != "match":
        dump()
        os._exit(0)  # no error: the parent fails the case
    cake_kcp.graph_cache_clear()
    dist.destroy_process_group()
    stage("destroyed")
    dump()
    os._exit(0 if result["pass"] else 4)


@pytest.mark.parametrize("case", RANK_CHECK_CASES)
def test_rank_check_two_ranks(case: str) -> None:
    """Two spawned ranks on two GPUs (``RANK_CHECK_ROW``).  ``match``: the first call of every key checks exactly
    once, in the forward (saved and replay mode, a graph="replay" key before its capture, a handle's original
    forward and its states-tier backward), never in a repeat, a capture, a replay or a recompute; the exchange counts
    are the records' collectives; both ranks record the same digests, which are the keys' ``kcp_rank_digest``; no
    device assert; a clean shutdown.  Mismatch (rank 1 departs in the partition, the head count, ``save_transfer`` or
    the recompute handle): after a matching warm-up both ranks fail with a device-side assert at the mismatching
    call, well within the deadline, instead of hanging in (or exchanging wrong bytes through) the exchange."""
    W = 2
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1") or 1) > 1
    ):
        pytest.skip(
            "the rank-agreement rows spawn their own two ranks (one Slurm task)"
        )
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_rank_check_") as tmp:
        keep = os.environ.get("KCP_TEST_RESULT_DIR")
        out_dir = os.path.join(keep, f"rank_check-{case}") if keep else tmp
        os.makedirs(out_dir, exist_ok=True)
        rdzv = _rendezvous(tmp)
        spawn = mp.get_context("spawn")
        procs = [
            spawn.Process(target=_rank_check_entry, args=(r, W, case, rdzv, out_dir))
            for r in range(W)
        ]
        start = time.monotonic()
        for p in procs:
            p.start()
        for p in procs:
            p.join(max(0.0, RANK_CHECK_DEADLINE_S - (time.monotonic() - start)))
        hung = [r for r, p in enumerate(procs) if p.is_alive()]
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(60)
        elapsed = time.monotonic() - start
        ranks = []
        for r, p in enumerate(procs):
            log = Path(out_dir, f"rank{r}.log")
            js = Path(out_dir, f"rank{r}.json")
            text = log.read_text(errors="replace") if log.exists() else ""
            try:
                result = json.loads(js.read_text()) if js.exists() else None
            except json.JSONDecodeError:
                result = None  # a rank aborted mid-write (files of an older writer)
            ranks.append(
                {
                    "exitcode": p.exitcode,
                    "result": result,
                    "text": text,
                    "tail": text[-3000:],
                }
            )
    brief = [
        {"exitcode": x["exitcode"], "result": x["result"], "tail": x["tail"]}
        for x in ranks
    ]
    assert not hung, (case, "hung ranks", hung, brief)
    if case == "match":
        assert all(
            x["exitcode"] == 0 and x["result"] is not None and x["result"].get("pass")
            for x in ranks
        ), (case, brief)
        return
    for x in ranks:
        text = x["text"] + json.dumps(x["result"] or {})
        assert x["exitcode"] != 0, (case, "no error", brief)
        assert "warm_done" in text and "mismatch_call" in text, (case, brief)
        assert "mismatch_returned" not in text, (case, brief)
        # the CUDA error of the trap (cudaErrorAssert), as torch, NCCL or the watchdog reports it
        assert any(m in text for m in ASSERT_MARKERS), (case, brief)
    assert elapsed < RANK_CHECK_DEADLINE_S, (case, elapsed)


# ---------------------------------------------------------------------------------------------------------------
# Checkpoint recompute (issue #7815): ``cake_kda_nondet_kcp`` under ``torch.utils.checkpoint``.  Rows: ``ROWS`` plus one
# H = 64 row (``P = 1`` ranks) and one row where no document crosses a rank; under a Slurm step of 8 or 16 tasks also the rows of that width (``CKPT_WIDE``; a
# one-node run has no such rows, so they never show up as skips there).

CKPT_WIDE = {
    8: [
        ((3000,), 4, 8, "zigzag"),
        ((2048, 5, 3000, 1), 4, 8, "zigzag_per_document"),
        ((6, 1, 2, 3, 4000), 4, 8, "ragged:1,1,1,3,0,2,4000,4"),
    ],
    16: [
        ((3000,), 4, 16, "zigzag"),
        ((2048, 5, 3000, 1), 4, 16, "zigzag_per_document"),
    ],
}
# no document crosses a rank (one document per rank): no rank binds a state, and the ring has no hop
CKPT_NONCROSSING = ((1200, 900, 900, 600), 4, 4, "ragged:1200,900,900,600")
CKPT_ROWS = (
    ROWS
    + [((2048, 5, 3000, 1), 64, 4, "zigzag_per_document"), CKPT_NONCROSSING]
    + CKPT_WIDE.get(int(os.environ.get("SLURM_NTASKS", "0") or 0), [])
)
# the rows whose outputs are not judged: 5,054 tokens at 64 heads is larger than the judged cases (about 4,096 tokens
# per draw, one production-size case per layout up to about 16k tokens x 4 heads); it keeps the exact checks, and the
# row ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document") judges the same packing at 4 heads
CKPT_EXACT_ONLY = (((2048, 5, 3000, 1), 64, 4, "zigzag_per_document"),)


def _requested(device) -> int:
    return int(torch.cuda.memory_stats(device).get("requested_bytes.all.current", 0))


def _poison(device) -> None:
    """Fill the caching allocator's free blocks (large and small pool) with all-ones bits (NaN in BF16 / FP32): rows
    of a fresh ``torch.empty`` tensor that no kernel writes then read as NaN, not as a stale earlier result."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    big = torch.full((1 << 27,), -1, dtype=torch.int32, device=device)  # 512 MiB
    small = [
        torch.full((1 << 17,), -1, dtype=torch.int32, device=device) for _ in range(256)
    ]  # 512 KiB each: the small pool's 2 MiB segments
    torch.cuda.synchronize(device)
    del big, small


# the launches a bound recompute may drop: the exchange (c1 / c2 or the ring hops) and its feeders (the owner merge,
# the P = 1 summaries, the P > 1 map merge and transpose)
DROPPABLE = {
    "a2a:c1",
    "a2a:c2",
    "kcp_affine_merge",
    "kcp_fwdh_offset",
    "kcp_transfer_walk",
    "kcp_transpose_tiles",
}


def _removed_launches(orig: list, replay: list) -> tuple[bool, list]:
    """``(ok, removed)``: the original forward's launches the replay drops; ``ok`` when the replay is the original
    sequence with launches deleted (same order, nothing added) and every dropped one an exchange launch or a feeder
    of it."""
    removed, i = [], 0
    for x in orig:
        if i < len(replay) and replay[i] == x:
            i += 1
        else:
            removed.append(x)
    ok = i == len(replay) and all(
        x in DROPPABLE or x.startswith("ring:") for x in removed
    )
    return ok, removed


class _Counter:
    """Counts the ``torch.distributed`` collectives issued, the waits on their work handles and the bytes of
    ``all_to_all_single`` (input / output tensors) by phase: ``orig``, ``recompute`` (set by the checkpoint's
    ``context_fn``, re-entered per recompute) and ``bwd``.  ``isend`` / ``irecv`` stay unshimmed: the ring exchange
    passes them to ``P2POp``, which checks their identity; its hops are counted as ``batch_isend_irecv`` calls.
    The one-time rank-agreement checks: ``gathers`` counts the ``all_gather_into_tensor`` calls (counted in ``calls``
    too) and ``checks`` the checks the package reports (``cake_kcp.RANK_CHECK_STATS["issued"]``, attributed to the
    phase they ran in); :func:`_exchange_counts` takes the checks out of ``calls``."""

    NAMES = (
        "all_to_all_single",
        "batch_isend_irecv",
        "all_gather_object",
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "broadcast",
        "reduce_scatter_tensor",
        "send",
        "recv",
    )

    def __init__(self) -> None:
        self._phase = "orig"
        self.calls: dict[str, int] = {}
        self.waits: dict[str, int] = {}
        self.send: dict[str, int] = {}
        self.recv: dict[str, int] = {}
        self.gathers: dict[str, int] = {}
        self.checks: dict[str, int] = {}
        # RANK_CHECK_STATS["issued"] at the last flush (None outside the with-block)
        self._issued: int | None = None
        self._saved: dict[str, object] = {}
        self._stack: list[str] = []

    @property
    def phase(self) -> str:
        return self._phase

    @phase.setter
    def phase(self, value: str) -> None:
        self._flush()  # the checks issued so far belong to the phase that ends here
        self._phase = value

    def _flush(self) -> None:
        if self._issued is None:
            return
        now = int(cake_kcp.RANK_CHECK_STATS["issued"])
        if now != self._issued:
            self._bump(self.checks, now - self._issued)
        self._issued = now

    def _bump(self, table: dict[str, int], n: int = 1) -> None:
        table[self.phase] = table.get(self.phase, 0) + n

    def _work(self, w):
        counter = self

        class _Work:
            def __init__(self, inner) -> None:
                self._inner = inner

            def wait(self, *a, **kw):
                counter._bump(counter.waits)
                return self._inner.wait(*a, **kw)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        return _Work(w) if hasattr(w, "wait") else w

    def __enter__(self) -> "_Counter":
        for name in self.NAMES:
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._saved[name] = fn

            def shim(*a, _fn=fn, _name=name, **kw):
                self._bump(self.calls)
                if _name == "all_gather_into_tensor":
                    self._bump(self.gathers)
                if _name == "all_to_all_single":
                    out_t, in_t = a[0], a[1]
                    self._bump(self.recv, out_t.numel() * out_t.element_size())
                    self._bump(self.send, in_t.numel() * in_t.element_size())
                r = _fn(*a, **kw)
                if isinstance(r, list):
                    return [self._work(x) for x in r]
                return self._work(r)

            setattr(dist, name, shim)
        self._issued = int(cake_kcp.RANK_CHECK_STATS["issued"])
        return self

    def __exit__(self, *exc) -> None:
        self._flush()
        self._issued = None
        for name, fn in self._saved.items():
            setattr(dist, name, fn)
        self._saved.clear()

    def recompute_cm(self):
        counter = self

        class _Phase:
            def __enter__(self):
                counter._stack.append(counter.phase)
                counter.phase = "recompute"
                return self

            def __exit__(self, *exc):
                counter.phase = counter._stack.pop()
                return False

        return _Phase()

    def record(self) -> dict:
        self._flush()
        return {
            t: {p: getattr(self, t).get(p, 0) for p in ("orig", "recompute", "bwd")}
            for t in ("calls", "waits", "send", "recv", "gathers", "checks")
        }


NO_CHECKS = {"orig": 0, "recompute": 0, "bwd": 0}


def _exchange_counts(rec: dict) -> dict:
    """``rec`` (:meth:`_Counter.record`) with the rank-agreement checks taken out of ``calls``: the collectives of
    the exchange (c1 / c2 / b1 / b2 / ring hops) and of the convolutions.  The checks themselves are compared
    exactly by :func:`_checks_exact`."""
    out = dict(rec)
    out["calls"] = {p: rec["calls"][p] - rec["checks"][p] for p in rec["calls"]}
    return out


def _checks_exact(rec: dict, want: dict) -> bool:
    """Per phase: the ``all_gather_into_tensor`` calls are the checks the package reports, and both are ``want``
    (``NO_CHECKS`` but for a run that is the first to use a call key of its context; never in a recompute)."""
    return rec["gathers"] == rec["checks"] == want and want["recompute"] == 0


def _handle_recompute_flags(handle_runs: dict, W: int) -> dict:
    """The handle runs of the raw-gate and in-kernel L2 norm rows (their ``autograd_pass``), per handle tier
    (``handle_runs[keep]``: the original forward's, the recompute's and the backward's LAST_CALL records and the
    :class:`_Counter` record): the recompute binds the stored states and exchanges nothing -- the call's own counter
    and sequence (no c1 / c2, no ring hop) and no torch.distributed call or wait in the recompute phase --, while the
    original forward and the backward exchange (positive control: c1 + c2 and b1 + b2, as torch.distributed calls on
    more than one rank; the one-time rank-agreement checks are counted apart, :func:`_exchange_counts`).  The bits
    are compared by the rows' ``handle*_eq_ckpt`` flags."""
    flags = {}
    for keep, (orig, rep, bwd, rec) in handle_runs.items():
        calls = _exchange_counts(rec)["calls"]
        flags[f"handle_{keep}_recompute_no_collective"] = (
            orig is not None
            and orig["recompute_handle"] == "capture"
            and orig["n_collectives"] == 2
            and rep is not None
            and rep["recompute_handle"] == "replay"
            and rep["n_collectives"] == 0
            and not any(s.startswith(("a2a:", "ring:")) for s in rep["sequence"])
            and bwd["n_collectives"] == 2
            and calls["recompute"] == rec["waits"]["recompute"] == 0
            and calls["orig"] == (orig["n_collectives"] if W > 1 else 0)
            and calls["bwd"] == (bwd["n_collectives"] if W > 1 else 0)
            and rec["gathers"] == rec["checks"]
            and rec["checks"]["recompute"] == 0
        )
    return flags


def _run_rank_ckpt(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None, local_id: int
) -> dict:
    """One rank of a checkpoint row (issue #7815), over the judge's draws (all in one process group):

    - accuracy, on every draw: ``out`` and the five gradients of the direct pass, of the plain checkpoint
      (``use_reentrant`` False and True) and of the checkpoint with a recompute handle (both tiers, ``use_reentrant``
      False and True, the ring exchange where the partition allows it, the overlap schedule with a2a and ring,
      ``retain_graph`` + a second backward -- each backward's own gradients --, early stop off, nested regions with
      the handle outside / inside the outer region), and every recompute's ``out``, gathered to rank 0 and judged
      there against the FP64 reference beside the naive BF16 tier, each under its own name (:class:`_RankJudge`;
      the package has no bit contract between the forms);
    - plumbing, exact, on every draw: ``dw`` (the checkpointed function returns ``out * w``) is bitwise ``dout``
      times the ``out`` of the forward each backward read; on the first draw: with ``retain_graph`` every leaf's
      gradient is bitwise the sum of the two backwards'; the recompute's launches are the original's in order minus
      the exchange and its feeders;
    - the rest on the first draw: #7039: without a handle a checkpointed forward retains nothing (per-call and
      caller-held workspace, after a warm-up step of the same form);
    - binding: a rank with crossing-in fragments stores a non-empty ``S_in``, and perturbing it before the backward
      changes the recompute's ``out`` and ``dw`` (positive control);
    - collectives, waits and all-to-all bytes per phase (``orig`` / ``recompute`` / ``bwd``): the plain checkpoint
      exchanges in every phase (positive control), the handle's recompute issues none, its original forward and
      backward match the byte queries of the tier (``cake_kda_nondet_train_kcp_comm_bytes(checkpoint_keep=...)``);
    - retained bytes after a checkpointed forward with a handle == ``handle.nbytes()`` ==
      ``cake_kda_nondet_train_kcp_saved_bytes(checkpoint_keep=..., with_conv=False)``;
    - a symmetric fingerprint mismatch raises on every rank (no collective is issued -- counted --, no hang);
    - the one-time rank-agreement checks: exactly one ``all_gather_into_tensor`` per call key at its first call
      (``cake_kcp.RANK_CHECK_STATS``), in the phase of that call, none in a recompute; the exchange counts above
      leave them out (:func:`_exchange_counts`);
    - the layer core, on every draw (inputs of the draw): three KCP convolutions (with ``"conv"`` handles, one of them
      shared with KDA) + q / k normalization + KDA, direct, under a plain checkpoint and under one with handles, and
      the same with the multi convolution and ONE handle: KDA's ``out``, its recompute's ``out`` and KDA's gradients
      at its operands (``dq`` / ``dk`` / ``dv`` kept on the normalized q / k and on v, ``dg`` / ``dbeta`` of the
      leaves), gathered to rank 0 and judged there, each form under its own name (``layer_<form>``,
      ``layer_multi_<form>``), against the FP64 reference of KDA's own operands (gathered) with the draw's ``g`` /
      ``beta`` / ``dout``; plumbing, exact: every call's KDA operands (original forward and recompute, every form) are
      the bits of the direct form's (the deterministic convolution package: run-to-run bits, the recompute binding
      bitwise the exchanging recompute, the multi call bitwise the single calls), one recompute per checkpoint; on
      the first draw the recompute issues no collective and the handles hold the queried bytes.

    A row of :data:`CKPT_EXACT_ONLY` runs the first draw only and judges nothing."""
    import contextlib

    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    from tests.cake_naive import DRAWS, draw_seed
    from tests.experimental.test_helpers.cake_kda_naive import draw_case

    conv_kcp = cake_package("cake_causal_conv1d_det_train", "kcp")
    lengths, H, _w, layout = row
    judged = tuple(row) not in CKPT_EXACT_ONLY
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    ws = cake_kcp.CakeKDANondetKcpWorkspace()
    judge = _RankJudge(part, cu, W, rank, SCALE)
    flags: dict[str, bool] = {}
    info: dict[str, object] = {
        "T_r": sum(int(f.length) for f in part.fragments[rank]),
        "n_frag": int(ctx.n_frag),
        "n_in": int(ctx.n_in),
        "n_out": int(ctx.n_out),
        "P": int(ctx.heads(H).P),
        "torch": torch.__version__,
    }

    def bits(t: torch.Tensor) -> torch.Tensor:
        # bit view: -0.0 vs 0.0 and NaN payloads count as differences
        return t.contiguous().view(
            torch.int16 if t.element_size() == 2 else torch.int32
        )

    def same(a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))
        )

    def one_draw(ops: dict, j: int) -> tuple:
        """Every form of draw ``j`` on this rank's slices ``ops``: the rows' arms to judge, the layer core's arms and
        its KDA operands (local); the flags of the first draw go to ``flags`` (later draws: the ``dw`` identity, the
        layer core's recompute counts and operand bits only)."""
        first = j == 0
        q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
        fl = flags if first else {}
        inf = info if first else {}
        counts: dict[str, dict] = {}
        arms: dict = {}
        dw_bad: list = []

        def leaves():
            return [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]

        if first:
            # #7039: retained bytes after a checkpointed forward without a handle (the output alone)
            retained = {}
            for form, wsp in (("per_call", None), ("caller_ws", ws)):

                def fn0(*a, _w=wsp):
                    return cake_kcp.cake_kda_nondet_kcp(
                        *a, ctx, None, "light", "saved", _w
                    )

                for step in ("warm", "measure"):
                    lv = leaves()
                    torch.cuda.synchronize(device)
                    gc.collect()
                    before = _requested(device)
                    out = checkpoint(fn0, *lv, use_reentrant=False)
                    torch.cuda.synchronize(device)
                    if step == "measure":
                        retained[form] = (
                            _requested(device)
                            - before
                            - out.numel() * out.element_size()
                        )
                    out.backward(dout)
                    del out, lv
            info["retained_bytes"] = retained
            flags["retained_zero"] = all(v_ == 0 for v_ in retained.values())

        def run(
            mode: str,
            handle=None,
            *,
            keep=None,
            twice=False,
            reentrant=False,
            early_stop=True,
            poison=False,
            tamper=None,
            **opts,
        ):
            """mode: direct | ckpt | nested_out (handle outside both regions) | nested_in (handle made in the outer).
            The checkpointed function returns ``out * w`` (``w`` a leaf of ones: no bit of ``out`` or of its gradient
            changes, but the product's backward reads the ``out`` of the forward it belongs to -- the recompute's under
            a checkpoint; ``dw`` = ``dout * out``).  Returns the results, the per-phase counts and ``aux``: the
            LAST_CALL record of the original forward and of each backward's recompute (popped before each backward,
            ``None`` when none ran), the recomputes' ``out`` (``"outs"``) and whether they hold the original forward's
            bits (``"outs_same"``, read by the binding control only: the package has no bit contract between calls).
            Plumbing, exact (``dw_bad``): in the direct and checkpoint modes ``dw`` is bitwise ``dout`` times the ``out``
            of the forward each backward read, summed over the backwards; with ``twice`` each backward's own gradients
            of the leaves are taken by tensor hooks (``"backwards"``) and every leaf's gradient is bitwise their sum.
            ``poison``: :func:`_poison` first; ``tamper(handle)`` runs between the forward and the backward."""
            if poison:
                _poison(device)
            lv = leaves()
            wl = torch.ones_like(dout).requires_grad_(True)
            parts: dict = {f"d{n}": [] for n in ("q", "k", "v", "g", "beta")}
            if twice:
                for n, lf in zip(("q", "k", "v", "g", "beta"), lv, strict=True):
                    lf.register_hook(
                        lambda gr, _n=f"d{n}": parts[_n].append(gr.detach().clone())
                    )
            kw = dict(opts)
            if keep is not None:
                kw["checkpoint_keep"] = keep
            outs = []

            def fn(q_, k_, v_, g_, b_, h_):
                y = cake_kcp.cake_kda_nondet_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctx,
                    workspace=ws,
                    recompute_handle=h_,
                    **kw,
                )
                outs.append(y.detach().clone())
                return y

            def top(q_, k_, v_, g_, b_, h_, w_):
                return fn(q_, k_, v_, g_, b_, h_) * w_

            recs = []
            with _Counter() as cc, set_checkpoint_early_stop(early_stop):
                ctx_fn = lambda: (contextlib.nullcontext(), cc.recompute_cm())  # noqa: E731
                if mode == "direct":
                    out = top(*lv, handle, wl)
                elif mode == "ckpt":
                    if reentrant:
                        out = checkpoint(top, *lv, handle, wl, use_reentrant=True)
                    else:
                        out = checkpoint(
                            top, *lv, handle, wl, use_reentrant=False, context_fn=ctx_fn
                        )
                else:

                    def outer(q_, k_, v_, g_, b_, h_, w_):
                        # the product saves the outer region's yi (the outer recompute runs and its yi is read)
                        hh = (
                            cake_kcp.CakeKDANondetKcpRecomputeHandle()
                            if mode == "nested_in"
                            else h_
                        )
                        yi = checkpoint(fn, q_, k_, v_, g_, b_, hh, use_reentrant=False)
                        return yi * w_

                    out = checkpoint(
                        outer, *lv, handle, wl, use_reentrant=False, context_fn=ctx_fn
                    )
                recs.append(cake_kcp.LAST_CALL.pop("fwd"))
                if tamper is not None:
                    tamper(handle)
                cc.phase = "bwd"
                for i in range(2 if twice else 1):
                    out.backward(dout, retain_graph=twice and i == 0)
                    recs.append(cake_kcp.LAST_CALL.pop("fwd", None))
            res = {"out": out.detach(), "dw": wl.grad}
            res.update(
                {
                    f"d{n}": lf.grad
                    for n, lf in zip(("q", "k", "v", "g", "beta"), lv, strict=True)
                }
            )
            if mode in ("direct", "ckpt") and tamper is None:
                # the forward each backward read: the recompute's under a checkpoint, else the one forward
                read = outs[1:] if mode == "ckpt" else outs * (2 if twice else 1)
                want_dw = dout * read[0]
                for o in read[1:]:
                    want_dw = want_dw + dout * o
                if not same(wl.grad, want_dw):
                    dw_bad.append((mode, keep, reentrant, twice, sorted(opts)))
            modes = [None if r is None else r.get("recompute_handle") for r in recs]
            aux = {
                "recs": recs,
                "n_outs": len(outs),
                "outs": outs[1:],
                "outs_same": all(same(o, outs[0]) for o in outs[1:]),
            }
            if twice:
                # torch's own accumulation of the two backwards' gradients (plumbing, exact)
                aux["accumulates"] = all(
                    len(parts[n]) == 2 and same(res[n], parts[n][0] + parts[n][1])
                    for n in parts
                )
                # each backward's own gradients (NaN where a hook did not fire twice: the judge's contract reports it, and
                # every rank gathers the same outputs)
                aux["backwards"] = [
                    {
                        n: parts[n][i]
                        if len(parts[n]) == 2
                        else torch.full_like(res[n], float("nan"))
                        for n in parts
                    }
                    for i in range(2)
                ]
            return res, {**_exchange_counts(cc.record()), "modes": modes}, aux

        def recomputed(aux) -> list:
            """The recomputes' ``out`` of a :func:`run`, as an arm's repeated results."""
            return [{"out": o} for o in aux["outs"]]

        def replay_ok(aux, exchange: str) -> bool:
            """The recompute ran, and its launches are the original's in order minus the exchange (every c1 / c2 or every
            ring hop) and its feeders (:func:`_removed_launches`)."""
            orig, rep = aux["recs"][0], aux["recs"][1]
            if rep is None or aux["n_outs"] < 2:
                return False
            ok, removed = _removed_launches(orig["sequence"], rep["sequence"])
            if exchange == "ring":
                want = [x for x in orig["sequence"] if x.startswith("ring:")]
                exch = [x for x in removed if x.startswith(("ring:", "a2a:"))]
            else:
                want, exch = (
                    ["a2a:c1", "a2a:c2"],
                    [x for x in removed if x.startswith("a2a:")],
                )
            return ok and exch == want and rep["n_collectives"] == 0

        def plain_ok(aux) -> bool:
            """The plain checkpoint's recompute: one, with the original's launches."""
            orig, rep = aux["recs"][0], aux["recs"][1]
            return (
                rep is not None
                and aux["n_outs"] == 2
                and rep["sequence"] == orig["sequence"]
            )

        direct, counts["direct"], _ = run("direct")
        arms["direct"] = direct
        plain, counts["plain"], aux = run("ckpt")
        arms["plain"], arms["plain.recompute"] = plain, recomputed(aux)
        fl["plain_recompute"] = plain_ok(aux)
        plain_re, counts["plain_reentrant"], aux = run("ckpt", reentrant=True)
        arms["plain_reentrant"] = plain_re
        arms["plain_reentrant.recompute"] = recomputed(aux)
        fl["plain_reentrant_recompute"] = plain_ok(aux)
        pc = counts["plain"]
        # positive control: the plain checkpoint exchanges (c1 + c2) in the original forward and in its recompute and
        # (b1 + b2) in the backward on every rank
        fl["positive_control"] = all(
            pc[t]["orig"] > 0
            and pc[t]["recompute"] == pc[t]["orig"]
            and pc[t]["bwd"] > 0
            for t in ("calls", "waits")
        )
        n_in = int(ctx.n_in)
        for keep in ("states", "transfer"):
            for reentrant in (False, True):
                tag = f"handle_{keep}" + ("_reentrant" if reentrant else "")
                hd = cake_kcp.CakeKDANondetKcpRecomputeHandle()
                res, rec, aux = run("ckpt", hd, keep=keep, reentrant=reentrant)
                counts[tag] = rec
                arms[tag], arms[f"{tag}.recompute"] = res, recomputed(aux)
                fl[f"{tag}_recompute"] = replay_ok(aux, "a2a")
                ref = counts["plain_reentrant" if reentrant else "plain"]
                if reentrant:  # the recompute runs inside the backward (no context_fn): plain = bwd + recompute
                    want = {
                        t: {
                            "orig": ref[t]["orig"],
                            "recompute": 0,
                            "bwd": ref[t]["bwd"] - ref[t]["orig"],
                        }
                        for t in ("calls", "waits")
                    }
                else:
                    want = {
                        t: {
                            "orig": ref[t]["orig"],
                            "recompute": 0,
                            "bwd": ref[t]["bwd"],
                        }
                        for t in ("calls", "waits")
                    }
                fl[f"{tag}_counts"] = all(rec[t] == want[t] for t in ("calls", "waits"))
                fl[f"{tag}_modes"] = rec["modes"] == ["capture", "replay"]
                # bytes the collectives moved == the byte query of the tier (KDA alone: c1 + c2, b1 + b2; none in the
                # recompute)
                cb = cake_kcp_plan.cake_kda_nondet_train_kcp_comm_bytes(
                    ctx, H=H, checkpoint_keep=keep
                )
                want_b = {
                    "send": {
                        "orig": cb["c1"]["send"] + cb["c2"]["send"],
                        "recompute": cb["recompute"]["send"],
                        "bwd": cb["b1"]["send"] + cb["b2"]["send"],
                    },
                    "recv": {
                        "orig": cb["c1"]["recv"] + cb["c2"]["recv"],
                        "recompute": cb["recompute"]["recv"],
                        "bwd": cb["b1"]["recv"] + cb["b2"]["recv"],
                    },
                }
                if not reentrant:
                    fl[f"{tag}_bytes_eq_query"] = (
                        rec["send"] == want_b["send"] and rec["recv"] == want_b["recv"]
                    )
                query = cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                    ctx, H=H, checkpoint_keep=keep, with_conv=False
                )
                fl[f"{tag}_nbytes_eq_query"] = hd.nbytes() == query
                # a rank with crossing-in fragments binds a non-empty S_in (the checks above are not vacuous there)
                stored = hd.entries["kda"]
                fl[f"{tag}_binding_nonempty"] = n_in == 0 or (
                    stored["S_in"].numel() > 0 and hd.nbytes() > 0
                )
                inf.setdefault("handle_nbytes", {})[tag] = hd.nbytes()

        if first:
            # binding positive control: perturbed stored states change the recompute's out (dw) on every rank with
            # crossing-in fragments; a recompute that exchanged instead of binding would leave them unchanged (a
            # difference, not a bit contract)
            def perturb(h) -> None:
                e = h.entries["kda"]
                e["S_in"] = e["S_in"] + 1.0

            pert, _, pert_aux = run(
                "ckpt", cake_kcp.CakeKDANondetKcpRecomputeHandle(), tamper=perturb
            )
            fl["binding_control"] = n_in == 0 or (
                not pert_aux["outs_same"] and not same(pert["dw"], plain["dw"])
            )
            del pert, pert_aux
        # the ring exchange (where the partition allows it): its recompute skips the hops as well
        ring_ok = int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS
        inf["ring_eligible"] = ring_ok
        inf["max_hops"] = max(int(ctx.n_d_max) - 1, 0)
        if ring_ok:
            plain_ring, counts["plain_ring"], aux_p = run("ckpt", exchange="ring")
            hd_ring, counts["handle_ring"], aux_h = run(
                "ckpt", cake_kcp.CakeKDANondetKcpRecomputeHandle(), exchange="ring"
            )
            arms["plain_ring"], arms["plain_ring.recompute"] = (
                plain_ring,
                recomputed(aux_p),
            )
            arms["handle_ring"], arms["handle_ring.recompute"] = (
                hd_ring,
                recomputed(aux_h),
            )
            pr, hr = counts["plain_ring"], counts["handle_ring"]
            fl["ring_counts"] = all(
                hr[t]["recompute"] == 0
                and hr[t]["orig"] == pr[t]["orig"]
                and hr[t]["bwd"] == pr[t]["bwd"]
                and pr[t]["recompute"] == pr[t]["orig"]
                for t in ("calls", "waits")
            ) and hr["modes"] == ["capture", "replay"]
            fl["ring_recompute"] = plain_ok(aux_p) and replay_ok(aux_h, "ring")
            fl["ring_exchange"] = (
                aux_h["recs"][0]["exchange"] == "ring"
                and aux_p["recs"][0]["exchange"] == "ring"
            )
        # the overlap schedule (test knob): the bound branch issues the exchange window's launches without the
        # exchange; with the ring also where no document crosses a rank (no hop: the document-start / -end walks run
        # alone).  Every pass starts from a poisoned allocator, so an unwritten row reads as NaN (the judge's contract:
        # finite wherever the reference is).
        previous_knobs = cake_kcp_plan.enable_test_knobs(True)
        try:
            ov = {"_kcp_test_knobs": {"schedule": "overlap"}}
            plain_ov, counts["plain_overlap"], aux = run("ckpt", poison=True, **ov)
            arms["plain_overlap"], arms["plain_overlap.recompute"] = (
                plain_ov,
                recomputed(aux),
            )
            fl["overlap_plain"] = plain_ok(aux)
            po = counts["plain_overlap"]
            for keep in ("states", "transfer"):
                tag = f"handle_overlap_{keep}"
                res, rec, aux = run(
                    "ckpt",
                    cake_kcp.CakeKDANondetKcpRecomputeHandle(),
                    keep=keep,
                    poison=True,
                    **ov,
                )
                counts[tag] = rec
                arms[tag], arms[f"{tag}.recompute"] = res, recomputed(aux)
                fl[f"{tag}_recompute"] = replay_ok(aux, "a2a")
                fl[f"{tag}_counts"] = all(
                    rec[t]["recompute"] == 0
                    and rec[t]["orig"] == po[t]["orig"]
                    and rec[t]["bwd"] == po[t]["bwd"]
                    for t in ("calls", "waits")
                ) and rec["modes"] == ["capture", "replay"]
            if ring_ok:
                plain_rov, counts["plain_ring_overlap"], aux_p = run(
                    "ckpt", poison=True, exchange="ring", **ov
                )
                res, counts["handle_ring_overlap"], aux_h = run(
                    "ckpt",
                    cake_kcp.CakeKDANondetKcpRecomputeHandle(),
                    poison=True,
                    exchange="ring",
                    **ov,
                )
                arms["plain_ring_overlap"] = plain_rov
                arms["plain_ring_overlap.recompute"] = recomputed(aux_p)
                arms["handle_ring_overlap"] = res
                arms["handle_ring_overlap.recompute"] = recomputed(aux_h)
                pro, hro = counts["plain_ring_overlap"], counts["handle_ring_overlap"]
                fl["ring_overlap_recompute"] = plain_ok(aux_p) and replay_ok(
                    aux_h, "ring"
                )
                fl["ring_overlap_counts"] = all(
                    hro[t]["recompute"] == 0
                    and hro[t]["orig"] == pro[t]["orig"]
                    and hro[t]["bwd"] == pro[t]["bwd"]
                    for t in ("calls", "waits")
                ) and hro["modes"] == ["capture", "replay"]
        finally:
            cake_kcp_plan.enable_test_knobs(previous_knobs)
        # retain_graph: each backward's recompute binds the entry (no collective); each backward's gradients judged, and
        # their sum is what the leaves accumulate
        twice, counts["handle_twice"], aux = run(
            "ckpt", cake_kcp.CakeKDANondetKcpRecomputeHandle(), twice=True
        )
        arms["twice"] = aux["backwards"]
        arms["twice.out"] = {"out": twice["out"]}
        arms["twice.recompute"] = recomputed(aux)
        tw = counts["handle_twice"]
        fl["twice_accumulates"] = (
            aux["accumulates"]
            and tw["calls"]["recompute"] == 0
            and tw["calls"]["bwd"] == 2 * pc["calls"]["bwd"]
            and tw["modes"] == ["capture", "replay", "replay"]
            and aux["n_outs"] == 3
            and all(r is not None and r["n_collectives"] == 0 for r in aux["recs"][1:])
        )
        # early stop off
        late, counts["handle_no_early_stop"], aux = run(
            "ckpt", cake_kcp.CakeKDANondetKcpRecomputeHandle(), early_stop=False
        )
        arms["no_early_stop"], arms["no_early_stop.recompute"] = late, recomputed(aux)
        fl["no_early_stop"] = counts["handle_no_early_stop"]["calls"][
            "recompute"
        ] == 0 and replay_ok(aux, "a2a")
        # nested checkpoints
        nest_out, counts["nested_out"], aux_o = run(
            "nested_out", cake_kcp.CakeKDANondetKcpRecomputeHandle()
        )
        nest_in, counts["nested_in"], aux_i = run(
            "nested_in", cake_kcp.CakeKDANondetKcpRecomputeHandle()
        )
        arms["nested_out"], arms["nested_out.recompute"] = nest_out, recomputed(aux_o)
        arms["nested_in"], arms["nested_in.recompute"] = nest_in, recomputed(aux_i)
        # handle made outside both regions: the outer recompute (recompute phase) and the inner recompute (counted in
        # the backward phase, outside the outer's context) bind the entry; only the backward exchanges
        fl["nested_out_counts"] = counts["nested_out"]["calls"] == {
            "orig": pc["calls"]["orig"],
            "recompute": 0,
            "bwd": pc["calls"]["bwd"],
        }
        # handle made inside the outer region: the outer recompute's inner forward is an original forward (exchanges)
        fl["nested_in_counts"] = counts["nested_in"]["calls"] == pc["calls"]
        flags["dw_reads_own_forward"] = (
            flags.get("dw_reads_own_forward", True) and not dw_bad
        )
        if dw_bad:
            info.setdefault("dw_bad", []).append(dw_bad)
        if first:
            # retained bytes after a checkpointed forward with a handle (after a warm-up of the same form): the handle's
            # entry, i.e. cake_kda_nondet_train_kcp_saved_bytes(checkpoint_keep=..., with_conv=False)
            retained_h = {}
            for keep in ("states", "transfer"):
                for form, wsp in (("per_call", None), ("caller_ws", ws)):

                    def fn1(q_, k_, v_, g_, b_, h_, _w=wsp, _k=keep):
                        return cake_kcp.cake_kda_nondet_kcp(
                            q_,
                            k_,
                            v_,
                            g_,
                            b_,
                            ctx,
                            workspace=_w,
                            recompute_handle=h_,
                            checkpoint_keep=_k,
                        )

                    for step in ("warm", "measure"):
                        lv = leaves()
                        hd = cake_kcp.CakeKDANondetKcpRecomputeHandle()
                        torch.cuda.synchronize(device)
                        gc.collect()
                        before = _requested(device)
                        out = checkpoint(fn1, *lv, hd, use_reentrant=False)
                        torch.cuda.synchronize(device)
                        if step == "measure":
                            retained_h[f"{keep}/{form}"] = (
                                _requested(device)
                                - before
                                - out.numel() * out.element_size()
                            )
                            query = cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                                ctx, H=H, checkpoint_keep=keep, with_conv=False
                            )
                            flags[f"retained_{keep}_{form}"] = (
                                retained_h[f"{keep}/{form}"] == hd.nbytes() == query
                            )
                        out.backward(dout)
                        del out, lv, hd
            info["retained_bytes_handle"] = retained_h
            info["saved_bytes_query"] = {
                keep: cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                    ctx, H=H, checkpoint_keep=keep, with_conv=False
                )
                for keep in ("states", "transfer")
            }
            # a symmetric mismatch (every rank's entry describes another mode) raises on every rank before any collective
            # (counted: none in the failing backward)
            hd = cake_kcp.CakeKDANondetKcpRecomputeHandle()
            lv = leaves()
            out = checkpoint(
                lambda *a: cake_kcp.cake_kda_nondet_kcp(
                    *a[:5], ctx, recompute_handle=a[5]
                ),
                *lv,
                hd,
                use_reentrant=False,
            )
            fp = dict(hd.entries["kda"]["fingerprint"])
            fp["modes"] = (("use_qk_l2norm_in_kernel", True),)
            hd.entries["kda"]["fingerprint"] = tuple(fp.items())
            with _Counter() as mc:
                mc.phase = "bwd"
                try:
                    out.backward(dout)
                    flags["mismatch_raises"] = False
                except ValueError as exc:
                    flags["mismatch_raises"] = "differs in ['modes']" in str(exc)
            counts["mismatch"] = mc.record()
            flags["mismatch_no_collective"] = all(
                v_ == 0
                for t in ("calls", "waits")
                for v_ in counts["mismatch"][t].values()
            )
            del out, lv, hd

        # the layer core, on every draw: three convolutions + q / k normalization + KDA in one region, on inputs of the
        # draw (seed draw_seed(78150 + W, j)).  KDA's operands are the deterministic convolutions' outputs (q / k
        # normalized by torch): the same bits in every form, original forward and recompute (plumbing, exact: the conv
        # package's run-to-run bits, its recompute binding bitwise the exchanging recompute, its multi call bitwise the
        # single calls).  Each form's KDA ``out``, its recompute's ``out`` and KDA's gradients at its operands are
        # judged on rank 0 against the FP64 reference of those operands (``layer_judge`` below)
        C = H * 128
        gen = torch.Generator(device="cpu").manual_seed(draw_seed(78150 + W, j))
        T_all = cu[-1]
        x_all = [
            torch.randn(T_all, C, generator=gen).to(torch.bfloat16) for _ in range(3)
        ]
        w_all = [
            (torch.randn(C, 4, generator=gen) * 0.5).contiguous() for _ in range(3)
        ]
        xs = [
            _local_slices(part, rank, cu, {"x": x})["x"].to(device).contiguous()
            for x in x_all
        ]
        wts = [w.to(device) for w in w_all]

        def l2(x):
            xf = x.float()
            return (xf * torch.rsqrt((xf * xf).sum(-1, keepdim=True) + 1e-6)).to(
                torch.bfloat16
            )

        l_outs: list[torch.Tensor] = []
        l_ins: list[tuple] = []

        def kda_operands(qq, kk, vv) -> tuple:
            """KDA's ``q, k, v`` of one call (the original forward's first in ``l_ins``, then each recompute's); each
            feeds KDA alone, so the gradient it keeps is KDA's ``dq`` / ``dk`` / ``dv``."""
            T = qq.shape[0]
            ops_ = (
                l2(qq.view(T, H, 128)),
                l2(kk.view(T, H, 128)),
                vv.view(T, H, 128),
            )
            for t in ops_:
                if t.requires_grad:
                    t.retain_grad()
            l_ins.append(ops_)
            return ops_

        def layer(xq, xk, xv, wq, wk, wv, g_, b_, w_, hq, hk, hv):
            qq = conv_kcp.cake_causal_conv1d_det_kcp(xq, wq, ctx, recompute_handle=hq)
            kk = conv_kcp.cake_causal_conv1d_det_kcp(xk, wk, ctx, recompute_handle=hk)
            vv = conv_kcp.cake_causal_conv1d_det_kcp(xv, wv, ctx, recompute_handle=hv)
            y = cake_kcp.cake_kda_nondet_kcp(
                *kda_operands(qq, kk, vv),
                g_,
                b_,
                ctx,
                workspace=ws,
                recompute_handle=hq,
            )
            l_outs.append(y.detach().clone())
            return y * w_  # the product's backward reads the (recomputed) KDA out

        def layer_leaves() -> list:
            return [
                x.detach().clone().requires_grad_(True)
                for x in (*xs, *wts, g, beta, torch.ones_like(dout))
            ]

        def layer_pass(mode: str, fn, lv, hs: tuple):
            """One pass of a layer form (``direct``, else under a non-reentrant checkpoint): its per-phase counts, and
            the structural flags of every draw: one KDA forward (and one recompute under a checkpoint), and every
            call's KDA operands the bits of the direct form's (the first pass)."""
            l_outs.clear()
            l_ins.clear()
            with _Counter() as cc:
                if mode.endswith("direct"):
                    out = fn(*lv, *hs)
                else:
                    out = checkpoint(
                        fn,
                        *lv,
                        *hs,
                        use_reentrant=False,
                        context_fn=lambda: (
                            contextlib.nullcontext(),
                            cc.recompute_cm(),
                        ),
                    )
                cc.phase = "bwd"
                out.backward(dout)
            inf.setdefault("layer_n_outs", {})[mode] = len(l_outs)
            n_want = 1 if mode.endswith("direct") else 2
            ok = len(l_outs) == n_want and len(l_ins) == n_want
            flags[f"{mode}_recomputes"] = flags.get(f"{mode}_recomputes", True) and ok
            if not layer_ref:
                layer_ref.extend(t.detach() for t in l_ins[0])
            ops_ok = all(
                same(a.detach(), b)
                for call_ops in l_ins
                for a, b in zip(call_ops, layer_ref, strict=True)
            )
            flags["layer_operands_bitwise"] = (
                flags.get("layer_operands_bitwise", True) and ops_ok
            )
            return out, _exchange_counts(cc.record())

        def layer_judged(name: str, out, lv) -> None:
            """The arms of one layer pass: ``out`` (``y * w``, ``w`` ones: ``y``'s bits) with KDA's gradients at its
            operands (the original forward's) and at ``g`` / ``beta`` (leaves that feed KDA alone), NaN where no gradient
            arrived (the judge's contract reports it); under a checkpoint the recompute's ``out`` (``.recompute``)."""

            def grad(t):
                return (
                    t.grad if t.grad is not None else torch.full_like(t, float("nan"))
                )

            q0, k0, v0 = l_ins[0]
            layer_arms[name] = {
                "out": out.detach(),
                "dq": grad(q0),
                "dk": grad(k0),
                "dv": grad(v0),
                "dg": grad(lv[6]),
                "dbeta": grad(lv[7]),
            }
            if l_outs[1:]:
                layer_arms[f"{name}.recompute"] = [{"out": o} for o in l_outs[1:]]

        layer_arms: dict = {}
        layer_ref: list[torch.Tensor] = []
        l_nbytes = 0
        l_hs: tuple = ()
        for mode in ("direct", "plain", "handle"):
            lv = layer_leaves()
            hs = (
                (
                    cake_kcp.CakeKDANondetKcpRecomputeHandle(),  # shared by the q convolution and KDA
                    conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(),
                    conv_kcp.CakeCausalConv1dDetKcpRecomputeHandle(),
                )
                if mode == "handle"
                else (None, None, None)
            )
            out, counts[f"layer_{mode}"] = layer_pass(f"layer_{mode}", layer, lv, hs)
            layer_judged(f"layer_{mode}", out, lv)
            if mode == "handle":
                l_nbytes, l_hs = sum(h.nbytes() for h in hs), hs
            del out, lv
        lp, lh = counts["layer_plain"], counts["layer_handle"]
        fl["layer_positive_control"] = all(
            lp[t]["orig"] > 0
            and lp[t]["recompute"] == lp[t]["orig"]
            and lp[t]["bwd"] > 0
            for t in ("calls", "waits")
        )
        fl["layer_recompute_no_comm"] = all(
            lh[t]["recompute"] == 0
            and lh[t]["orig"] == lp[t]["orig"]
            and lh[t]["bwd"] == lp[t]["bwd"]
            for t in ("calls", "waits")
        )
        # the shared handle holds "conv" + "kda"; the three handles together = KDA states + three hist pools
        fl["layer_nbytes"] = sorted(l_hs[0].entries) == [
            "conv",
            "kda",
        ] and l_nbytes == (
            cake_kcp_plan.cake_kda_nondet_train_kcp_saved_bytes(
                ctx, H=H, checkpoint_keep="states", C=C
            )
            + 2 * int(ctx.halo.n_hist_rows) * C * 2
        )

        # the same layer with cake_causal_conv1d_det_kcp_multi (the q / k / v convolutions in one call, one halo exchange
        # per direction) and ONE handle for the convolutions and KDA: two halo collectives (and waits) fewer in every
        # phase that exchanges halos, the same retained bytes
        def layer_multi(xq, xk, xv, wq, wk, wv, g_, b_, w_, h):
            qq, kk, vv = conv_kcp.cake_causal_conv1d_det_kcp_multi(
                (xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h
            )
            y = cake_kcp.cake_kda_nondet_kcp(
                *kda_operands(qq, kk, vv),
                g_,
                b_,
                ctx,
                workspace=ws,
                recompute_handle=h,
            )
            l_outs.append(y.detach().clone())
            return y * w_

        lm_nbytes = 0
        lm_h = None
        for mode in ("direct", "plain", "handle"):
            lv = layer_leaves()
            h = cake_kcp.CakeKDANondetKcpRecomputeHandle() if mode == "handle" else None
            out, counts[f"layer_multi_{mode}"] = layer_pass(
                f"layer_multi_{mode}", layer_multi, lv, (h,)
            )
            layer_judged(f"layer_multi_{mode}", out, lv)
            if mode == "handle":
                lm_nbytes, lm_h = h.nbytes(), h
            del out, lv
        lmd, lmp, lmh = (
            counts["layer_multi_direct"],
            counts["layer_multi_plain"],
            counts["layer_multi_handle"],
        )
        ld = counts["layer_direct"]
        fl["layer_multi_counts"] = all(
            lmd[t]["orig"] == ld[t]["orig"] - 2
            and lmp[t]["orig"] == lp[t]["orig"] - 2
            and lmp[t]["recompute"] == lp[t]["recompute"] - 2
            and lmp[t]["bwd"] == lp[t]["bwd"] - 2
            and lmh[t]["orig"] == lp[t]["orig"] - 2
            and lmh[t]["recompute"] == 0
            and lmh[t]["bwd"] == lp[t]["bwd"] - 2
            for t in ("calls", "waits")
        )
        fl["layer_multi_nbytes"] = (
            sorted(lm_h.entries) == ["conv", "kda"] and lm_nbytes == l_nbytes
        )
        layer_ops = dict(zip(("q", "k", "v"), layer_ref, strict=True))
        l_ins.clear()
        l_outs.clear()

        if first:
            # the one-time rank-agreement checks (cake_kcp._rank_check), exactly: the #7039 warm-up above checked the default
            # key (a2a, save_transfer, saved set, no handle) outside any count; a run checks each key it is the first to use
            # -- the handle's original forward (whether a call carries a handle is a field), the states tier's backward
            # (save_transfer off without a handle), the ring's first forward (its backward and the ring handle's backward
            # share that key) -- and nothing else: no recompute, no later run of a key (the overlap schedule and the layer
            # reuse checked keys: knobs and the convolutions are not fields); every all_gather_into_tensor is one of them
            one = int(W > 1)
            want_checks = {
                "handle_states": {"orig": one, "recompute": 0, "bwd": one},
                "handle_transfer": {"orig": one, "recompute": 0, "bwd": 0},
            }
            if ring_ok:
                want_checks["plain_ring"] = {"orig": one, "recompute": 0, "bwd": 0}
                want_checks["handle_ring"] = {"orig": one, "recompute": 0, "bwd": 0}
            flags["rank_checks_exact"] = all(
                _checks_exact(c, want_checks.get(tag, NO_CHECKS))
                for tag, c in counts.items()
            ) and set(want_checks) <= set(counts)
            info["rank_checks"] = {
                tag: c["checks"]
                for tag, c in counts.items()
                if any(c["checks"].values())
            }
            info["counts"] = counts

        return arms, layer_arms, layer_ops

    # the layer core's KDA: judged against the FP64 reference of its own operands (the deterministic convolutions'
    # outputs, the same bits in every form) with the draw's g / beta / dout
    layer_judge = _RankJudge(part, cu, W, rank, SCALE)
    for j in range(DRAWS if judged else 1):
        # the same seeded operands on every rank; each takes its slices
        glob = draw_case(lengths, H, "g", False, draw_seed(JUDGE_SEED, j), device)
        arms, layer_arms, layer_ops = one_draw(_local_slices(part, rank, cu, glob), j)
        if judged:
            judge.draw(j, glob)
            for name, results in arms.items():
                judge.arm(name, results)
            layer_glob = {
                n: _gather_global(part, t, cu, cu[-1], W, rank)
                for n, t in layer_ops.items()
            }
            if rank == 0:
                layer_glob.update(
                    {n: glob[n] for n in ("g", "beta", "dout", "cu_seqlens")}
                )
            layer_judge.draw(j, layer_glob)
            for name, results in layer_arms.items():
                layer_judge.arm(name, results)
        del arms, layer_arms, layer_ops, glob

    flag_list = [None] * W
    info_list = [None] * W
    dist.all_gather_object(flag_list, flags)
    dist.all_gather_object(info_list, info)
    report = {
        "world": W,
        "row": _row_id(row),
        "torch": torch.__version__,
        "flags": {key: all(f[key] for f in flag_list) for key in flags},
        "failing_ranks": {
            key: [r for r, f in enumerate(flag_list) if not f[key]] for key in flags
        },
        "info_by_rank": info_list,
        "judged": judged,
    }
    # flags present on every rank only (ring_* where eligible: the eligibility is a property of the partition)
    # every row but the non-crossing one binds a non-empty S_in on some rank (the binding checks are not vacuous)
    report["n_in_by_rank"] = [i["n_in"] for i in info_list]
    report["pass"] = (
        all(report["flags"].values())
        and all(set(f) == set(flag_list[0]) for f in flag_list)
        and (tuple(row) == CKPT_NONCROSSING) == (max(report["n_in_by_rank"]) == 0)
    )
    if judged and rank == 0:
        rows_rep, layer_rep = judge.report(), layer_judge.report()
        report["verdicts"] = {**rows_rep["verdicts"], **layer_rep["verdicts"]}
        report["failures"] = rows_rep["failures"] + layer_rep["failures"]
    dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "ckpt-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_ckpt(
    rank: int, W: int, row, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank_ckpt(rank, W, row, rdzv, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("row", CKPT_ROWS, ids=_row_id)
def test_checkpoint_recompute_rows(record_property, row) -> None:
    """The checkpoint rows of ``_run_rank_ckpt`` on every rank of the row: every rank's flags hold; rank 0's verdicts
    of each form (a row of :data:`CKPT_EXACT_ONLY` aside) are recorded and settled here (in the
    one-process-per-Slurm-task form by the process of rank 0)."""
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        rank = int(os.environ["SLURM_PROCID"])
        report = _run_rank_ckpt(
            rank,
            W,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        if rank == 0 and report["judged"]:
            _settle(record_property, report, row[0])
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_ckpt_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_ckpt, W, row, tmp, result_dir)
        with open(os.path.join(result_dir, "ckpt-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report
    if report["judged"]:
        _settle(record_property, report, row[0])


# --------------------------------------------------------------------------- the package-kept default workspace

# ((lengths, H, W, layout), form): a2a rows (a zero-token rank; P = 1 at 64 heads), a ring row (the hop pools, lazily
# allocated) and the zero-token row with TORCH_NCCL_AVOID_RECORD_STREAMS=1 (ProcessGroupNCCL keeps the collective
# tensors alive instead of recording them on its stream)
DEFAULT_WS_ROWS = [
    (((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"), "a2a"),
    (((100, 640, 77, 1000, 129), 64, 4, "contiguous"), "a2a"),
    (((2000, 1000), 4, 4, "contiguous"), "ring"),
    (((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"), "avoid_record_streams"),
]


def _default_ws_id(spec) -> str:
    row, form = spec
    return f"{_row_id(row)}-{form}"


def _run_default_ws_rank(
    rank: int, W: int, spec, rdzv: str | None, result_dir: str | None
) -> dict:
    """``workspace=None`` steps on every rank of a CP row, a rank without tokens included, over the judge's draws (all
    in one process group): rank 0 drops the package's kept workspaces before every step (it builds new ones, the other
    ranks reuse theirs), the forward / backward in saved and replay mode, with no hang; no call leaves a collective
    unwaited; a forward that does not read the lazily allocated pools a forward of its key grew in the kept workspace
    (the ring's hop pools on the ring row, else the forward-only c1 receive / stage-map / carry pools of a forward
    without a saved set) requests at most the previous form's (``default_ws=0``) high-water of bytes; these flags hold
    on every draw.  The ``out`` and gradients of the same calls on a caller-held workspace (``held``,
    ``held.replay``), of the four default steps (``default``, ``default.replay``; the worst kept) and the ``out`` of
    the lazy-pool forward and of its previous form (``lazy``, ``lazy.previous``) are judged on rank 0 against the FP64
    reference beside the naive BF16 tier (:class:`_RankJudge`)."""
    from tests.cake_naive import DRAWS, draw_seed
    from tests.experimental.test_helpers.cake_kda_naive import draw_case

    row, form = spec
    lengths, H, _w, layout = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        if form == "avoid_record_streams":  # read when the process group starts
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
        part, group, rank, device
    )
    ring = form == "ring"
    opts = {"exchange": "ring", "save_transfer": False} if ring else {}
    off = {"_kcp_test_knobs": {"default_ws": 0}}
    judge = _RankJudge(part, cu, W, rank, SCALE)

    def one_draw(ops: dict):
        """The arms, the flags and the counts of one draw on this rank's slices ``ops``."""
        q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))

        def fwd(ws, **kw):
            return cake_kcp.cake_kda_nondet_train_kcp_forward(
                q, k, v, g, beta, ctx, workspace=ws, **kw
            )

        def step(ws, mode: str):
            out, saved = fwd(ws, boundary_mode=mode, **opts)
            found = [cake_kcp.LAST_CALL["fwd"]["default_ws"]]
            dq, dk, dv, dbeta, dg = cake_kcp.cake_kda_nondet_train_kcp_backward(
                q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, **opts
            )
            found.append(cake_kcp.LAST_CALL["bwd"]["default_ws"])
            torch.cuda.synchronize(device)
            res = {"out": out, "dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta}
            return res, found

        def high_water(**kw):
            """``(requested, allocated)`` high-water of a forward above the bytes held before it, and its output:
            the caching allocator's ``requested_bytes`` peak and its block-size ``allocated_bytes`` peak, which also
            counts the rounding and the unsplit tail (up to 1 MiB) of each reused cached block and so depends on which
            blocks the allocator has cached, not on the form alone.  The forward runs once unmeasured first, so the
            measured one follows a forward of its own form: on GB300 the requested peak of the first
            ``default_ws=0`` ring forward after the kept-workspace ring forwards was 256 KiB - 1 MiB per rank below
            that of every later one.  A block freed while another stream still had a use recorded on it stays in
            ``requested_bytes`` until the allocator processes that stream's event, which it does at its next
            allocation: inside the measured forward that late release would be subtracted from the forward's own
            high-water.  On B200 the ring hop pool (256 KiB on rank 3) that the kept workspace of the ring forwards
            before the measurement parks at the end of its call was released that way inside the measured
            ``default_ws=0`` forward, whose value then fell 256 KiB below its own high-water on every draw after the
            first.  ``empty_cache`` after the synchronize completes every such release first, so ``before`` counts
            live tensors only and the value is the forward's own high-water."""
            fwd(None, **kw)
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            before = torch.cuda.memory_stats(device)
            torch.cuda.reset_peak_memory_stats(device)
            out, _saved = fwd(None, **kw)
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            peaks = tuple(
                int(after[f"{kind}_bytes.all.peak"])
                - int(before[f"{kind}_bytes.all.current"])
                for kind in ("requested", "allocated")
            )
            return peaks, out.clone()

        flags: dict = {
            "ring_runs": not ring or int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS,
            "avoid_record_streams_set": form != "avoid_record_streams"
            or os.environ.get("TORCH_NCCL_AVOID_RECORD_STREAMS") == "1",
        }
        held = cake_kcp.CakeKDANondetKcpWorkspace()
        modes = ("saved", "replay")
        arms = {
            "held": step(held, "saved")[0],
            "held.replay": step(held, "replay")[0],
            "default": [],
            "default.replay": [],
        }
        unwaited = cake_kcp.DEFAULT_WS_STATS["unwaited"]
        cake_kcp.default_workspace_cache_clear()
        found = []
        for _i in range(4):
            if rank == 0:
                cake_kcp.default_workspace_cache_clear()
            for mode in modes:
                got, f = step(None, mode)
                arms["default" if mode == "saved" else "default.replay"].append(got)
                found.append(f)
        # rank 0's first call of every step builds; every other rank builds in its first step only
        flags["rank0_builds"] = rank != 0 or all(
            found[i][0] == "miss" for i in range(0, len(found), 2)
        )
        flags["others_reuse"] = rank == 0 or all(f == ["hit", "hit"] for f in found[2:])
        flags["kept"] = bool(cake_kcp._DEFAULT_WS)
        flags["no_unwaited"] = cake_kcp.DEFAULT_WS_STATS["unwaited"] == unwaited
        # the lazily allocated pools a forward of the key grew stay unallocated in a later forward that does not
        # read them (every rank calls the same forwards: they are collectives)
        lazy = dict(opts) if ring else {"save_for_backward": False}
        cake_kcp.default_workspace_cache_clear()
        for _ in range(2):
            fwd(None, **lazy)
        peak_new, out_new = high_water(**off)
        peak, out = high_water()
        arms["lazy.previous"] = {"out": out_new}
        arms["lazy"] = {"out": out}
        # the bytes requested (the allocated block sizes are recorded only, see high_water)
        flags["lazy_peak"] = peak[0] <= peak_new[0]
        counts = {
            "T_r": int(ctx.T_r),
            "found": found,
            "peak": peak,
            "peak_new": peak_new,
            "stats": dict(cake_kcp.DEFAULT_WS_STATS),
        }
        del held
        return arms, flags, counts

    prev = cake_kcp_plan.enable_test_knobs(True)
    try:
        flags: dict = {}
        for j in range(DRAWS):
            # the same seeded operands on every rank; each takes its slices
            glob = draw_case(lengths, H, "g", False, draw_seed(JUDGE_SEED, j), device)
            arms, found, info = one_draw(_local_slices(part, rank, cu, glob))
            if j == 0:
                counts = info
            for key, ok in found.items():
                flags[key] = flags.get(key, True) and bool(ok)
            judge.draw(j, glob)
            for name, results in arms.items():
                judge.arm(name, results)
            del arms, glob
    finally:
        cake_kcp_plan.enable_test_knobs(prev)
    if W > 1:
        flag_list, count_list = [None] * W, [None] * W
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(count_list, counts)
    else:
        flag_list, count_list = [flags], [counts]
    report = {
        "rank": rank,
        "world": W,
        "row": _default_ws_id(spec),
        "flags": {key: all(f[key] for f in flag_list) for key in flags},
        "failing_ranks": {
            key: [r for r, f in enumerate(flag_list) if not f[key]] for key in flags
        },
        "counts_by_rank": count_list,
    }
    report["pass"] = all(report["flags"].values())
    report.update(judge.report())
    if W > 1:
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "default-ws-" + _default_ws_id(spec) + ".json"),
            "w",
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_default_ws_entry(
    rank: int, W: int, spec, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_default_ws_rank(rank, W, spec, rdzv, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            cake_kcp.default_workspace_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("spec", DEFAULT_WS_ROWS, ids=_default_ws_id)
def test_multi_gpu_default_workspace_rows(record_property, spec) -> None:
    """The rows of :func:`_run_default_ws_rank`: one process per GPU (``spawn``) or one process per Slurm task (a step
    with ``SLURM_NTASKS`` > 1 runs the rows of its own world size; the ``TORCH_NCCL_AVOID_RECORD_STREAMS`` row only
    in a process whose group it starts).  Every rank's flags hold; rank 0's verdicts are recorded and settled here (in
    the one-process-per-Slurm-task form by the process of rank 0)."""
    row, form = spec
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        if form == "avoid_record_streams" and dist.is_initialized():
            pytest.skip(
                "TORCH_NCCL_AVOID_RECORD_STREAMS is read when the process group starts"
            )
        rank = int(os.environ["SLURM_PROCID"])
        report = _run_default_ws_rank(
            rank, W, spec, None, os.environ.get("KCP_TEST_RESULT_DIR")
        )
        assert report["pass"], report
        if rank == 0:
            _settle(record_property, report, row[0])
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_default_ws_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_default_ws_entry, W, spec, tmp, result_dir)
        with open(
            os.path.join(result_dir, "default-ws-" + _default_ws_id(spec) + ".json")
        ) as fh:
            report = json.load(fh)
    assert report["pass"], report
    _settle(record_property, report, row[0])
