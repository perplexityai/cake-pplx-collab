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

One KCP workspace shared by several prepared packs (``cake_kda_nondet_train_kcp.cake_kcp.CakeKDANondetKcpWorkspace``, the non-deterministic package): ``reserve`` sizes it
for a list of contexts (each pool at its largest need, the local arena at the largest layout, a plan view of it per
other pack), ``cake_kda_nondet_train_kcp_workspace_bytes([ctx, ...], H=)`` predicts its bytes, and the calls of the reserved packs, in any
interleaving, allocate nothing and keep every pointer.  The
workspace holds no context, its host caches are bounded, a call on another stream is ordered after the last one on
the device and a storage it releases is reused only after that stream's queued work (``record_stream`` on a switch
only), and ``allocate=False`` sizes the first forward for the backward.  Checkpointed layers with one recompute
handle per pack and layer on a shared workspace are exchange-free in their recompute, and the tensors a handle stores
are never in the workspace and never change before their backward (the other packs' calls cannot overwrite them).
The multi-GPU rows are in ``test_cake_kda_nondet_train_kcp_dist.py``-style launch forms at the end of this file
(``spawn`` or one process per Slurm task).

Accuracy: this package has no bit contract, so no result of one workspace, stream, schedule or path is compared with
another's.  Every output and gradient of every arm (one workspace per pack, the shared workspace, the poisoned arena,
the other stream, the graph replay, the recompute handles, ...) is judged under its own name against the FP64
reference of its pack beside the naive BF16 tier of ``tests/cake_naive.py`` (``cake_kda_naive.py``): 32 draws of the
operands per case, L2 and max norm, the owner-accepted gaps of ``cake_kda_known_gaps.py`` only (:class:`NaiveJudge`).
The exact checks (bytes, pointers, allocations, host synchronization, events, records, stored-tensor copies) run on
the first draw unless they are cheap.  A written saved set is an internal of the package: it is not compared, its
backward's gradients are judged.  Every pack of these rows is small enough to judge (:func:`judged`).
"""

import gc
import json
import os
import tempfile
import time
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.cake_naive import DRAWS, TierJudge, contract, draw_seed
from tests.experimental.test_cake_kda_nondet_train_kcp_dist import (
    _init_group,
    _rendezvous,
)
from tests.experimental.test_helpers.cake_host_sync import (
    allocator_counts,
    census,
    nonblocking,
    sleep_cycles,
    summary,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
)
from tests.experimental.test_helpers.cake_kda_fla import FlaReference
from tests.experimental.test_helpers.cake_kda_known_gaps import settle
from tests.experimental.test_helpers.cake_kda_naive import (
    fla_of,
    naive_of,
    oracle_with_grads,
    stored,
)
from tests.experimental.test_helpers.cake_kda_naive_kcp import to_global
from tests.experimental.test_helpers.cake_kda_train_kcp_saved import written_tensors
from tests.cake.test_helpers.cake_package import cake_package

kp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp_plan")
cake_kcp = cake_package("cake.cake_kda_nondet_train_kcp", "cake_kcp")
cake_backend = cake_package("cake.cake_kda_nondet_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_nondet_train", "cake_jit")

SM_COUNT = 152
# the document lengths of four packs of one rank: different T_r and n_frag; at H = 4 the one-document and
# two-document packs split into P > 1 segments, at H = 64 every pack runs P = 1
PACKS = ((1000, 577), (50, 640, 77, 1000, 129), (1450,), (300, 64, 700))
HEADS = (4, 64)
# the judged operands: draw j of a case at draw_seed(JUDGE_SEED, j) (pack i of a draw at that seed plus i)
JUDGE_SEED = 0x7815
# the cases the FP64 reference judges (its token recurrence runs per token, twice with the naive tier): a pack of at
# most JUDGE_MAX_TOKENS tokens at any head count, or of at most JUDGE_MAX_TOKEN_HEADS token-heads (16k tokens x 4
# heads); a larger row keeps its exact checks only
JUDGE_MAX_TOKENS = 6144
JUDGE_MAX_TOKEN_HEADS = 16384 * 4
TOKEN_OPERANDS = ("q", "k", "v", "g", "beta", "dout")
# the gradients of cake_kda_nondet_train_kcp_backward in its order, and of the autograd leaves (q, k, v, g, beta)
BWD_GRADS = ("dq", "dk", "dv", "dbeta", "dg")
LEAF_GRADS = ("dq", "dk", "dv", "dg", "dbeta")


def _cu(lengths) -> list[int]:
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu


# --------------------------------------------------------------------------- the naive-tier judge


class NaiveJudge:
    """The naive-tier judge of every case of a test over the judge's draws (``tests/cake_naive.py``): each case's arms
    against the FP64 reference of the case's own operands (``cake_kda_naive.oracle_with_grads``) beside the naive BF16
    tier (``cake_kda_naive.naive_of``) and FLA (:meth:`fla`; for the stacks of :func:`stack_refs`, FLA's layers
    composed the same way, :func:`stack_fla`), all in one ``TierJudge``.  An output of case ``c`` and arm ``a`` is
    judged as ``"c.a.<output>"`` (empty parts left out); an arm holding a list of results is judged as repeats (the
    judge keeps the worst).  ``dw`` (the gradient of the ones leaf ``w`` of a checkpointed ``out * w``) is judged
    against ``dout * out`` of the reference, of the tier and of FLA (the tier's and FLA's product in BF16: the PyTorch
    graph of the test on their ``out``).
    Every output is held to its contract on every draw; :meth:`result` gives the verdicts and the failures,
    :meth:`settle` records them (``naive_tier``) and settles them against the owner-accepted gaps
    (``cake_kda_known_gaps.settle``)."""

    def __init__(self, scale: float | None = None) -> None:
        self.judge = TierJudge(DRAWS)
        self.failures: list[str] = []
        self.scale = cake_kcp.SCALE if scale is None else scale

    def refs(
        self, inputs: dict, *, l2n: bool = False, lower_bound: float = LOWER_BOUND
    ) -> tuple[dict, dict]:
        """The FP64 reference and the naive tier of one case's operands (``q, k, v, g, beta, dout, cu_seqlens`` in
        the global token order; ``A_log`` / ``dt_bias`` in the raw gate mode), ``dw`` among them."""
        ref = oracle_with_grads(inputs, self.scale, lower_bound, l2n)
        tier = naive_of(inputs, self.scale, lower_bound, l2n)
        ref["dw"] = inputs["dout"].to(torch.float64) * ref["out"]
        tier["dw"] = stored(inputs["dout"]) * tier["out"]
        return ref, tier

    def fla(
        self, inputs: dict, *, l2n: bool = False, lower_bound: float = LOWER_BOUND
    ) -> FlaReference:
        """FLA's values of one case's operands (``cake_kda_naive.fla_of``), ``dw`` among them: ``dout`` times FLA's
        ``out`` in FLA's ``out`` dtype (BF16), as :meth:`refs` forms the tier's; or the reason FLA does not run the
        case (the outputs are then judged against the tier only)."""
        fla = fla_of(inputs, self.scale, lower_bound, l2n)
        if fla.values is not None:
            out = fla.values["out"]
            fla.values["dw"] = inputs["dout"].to(out.dtype) * out
        return fla

    def add(
        self,
        j: int,
        case: str,
        arms: dict,
        inputs: dict,
        *,
        l2n: bool = False,
        lower_bound: float = LOWER_BOUND,
    ) -> None:
        """Draw ``j`` of every arm of ``case`` (``{arm: {output: tensor}}`` or ``{arm: [{output: tensor}, ...]}``)
        on the operands ``inputs``."""
        ref, tier = self.refs(inputs, l2n=l2n, lower_bound=lower_bound)
        self.add_refs(j, case, arms, ref, tier, self.fla(inputs, l2n=l2n, lower_bound=lower_bound))

    def add_refs(self, j: int, case: str, arms: dict, ref: dict, tier: dict, fla=None) -> None:
        """:meth:`add` with the reference and the tier given (``{output: tensor}`` each), and FLA's when given."""
        for arm, results in arms.items():
            for res in results if isinstance(results, (list, tuple)) else (results,):
                for name, actual in res.items():
                    key = ".".join(p for p in (case, arm, name) if p)
                    self.add_one(j, key, actual, ref[name], tier[name], fla.value(name, actual) if fla else None)

    def add_one(self, j: int, key: str, actual, want, naive, fla=None) -> None:
        """Draw ``j`` of the output ``key``: its contract against the reference ``want``, then the judge (the
        reference and the tier read in ``actual``'s shape when they hold as many elements)."""
        if want.shape != actual.shape and want.numel() == actual.numel():
            want, naive = want.reshape(actual.shape), naive.reshape(actual.shape)
        actual = actual.detach().to(want.device)
        if message := contract(key, actual, want):
            self.failures.append(f"draw {j}: {message}")
            if actual.shape != want.shape:
                return
        self.judge.add(key, j, actual, want, {"bf16": naive.to(want.device)}, fla=fla)

    def result(self) -> dict:
        """The verdicts and the failures (the contracts', then the judge's), in JSON form."""
        return {
            "verdicts": self.judge.verdicts(),
            "failures": self.failures + self.judge.failures(),
        }

    def settle(self, record_property, lengths=None) -> None:
        """Record and settle the verdicts (:func:`settle_naive`); nothing when no case was judged (every case of
        the test larger than the judged size, :func:`judged`)."""
        if self.judge.errors or self.failures:
            settle_naive(record_property, self.result(), lengths)


def settle_naive(record_property, result: dict, lengths=None) -> None:
    """Record the verdicts of a :meth:`NaiveJudge.result` (rank 0's, read back from a row's report) with
    ``record_property`` (``naive_tier``) and settle them against the owner-accepted gaps; a run that judged nothing
    fails.  ``lengths``: the documents of the case (the gaps' scope), ``None`` for several cases."""
    for verdict in result["verdicts"].values():
        record_property("naive_tier", verdict)
    assert result["verdicts"], result["failures"]
    settle(result["verdicts"], result["failures"], lengths=lengths)


class RankJudge:
    """The naive-tier judge of the arms of a multi-rank row (one draw at a time, :meth:`draw`): each call of the judge
    (``judge(arm, result)``, a collective: every rank calls it with the same arms in the same order) gathers every
    rank's local results of the arm (an NCCL ``all_gather`` of the local buffers padded to the longest rank), puts
    them in the global token order on rank 0 and judges them there against the FP64 reference of the row's whole
    operands beside the naive tier (``"<arm>.<output>"``, :class:`NaiveJudge`).  The parameter gradients ``dA_log``
    / ``ddt_bias`` of the raw gate mode are judged as the package's cross-rank sums (``dA_log_sum`` /
    ``ddt_bias_sum`` of a result; a rank's own partials are not a gradient of the row and are not judged); a twice-run
    backward's accumulated gradients are judged halved (``halve``: the mean of the two backwards).  Rank 0 holds the
    verdicts (:meth:`result`)."""

    PARAMS = ("dA_log", "ddt_bias")

    def __init__(self, part, cu, W: int, rank: int, scale: float | None = None):
        self.part, self.cu, self.W, self.rank = part, [int(c) for c in cu], W, rank
        self.naive = NaiveJudge(scale) if rank == 0 else None
        self.sizes = [sum(int(f.length) for f in frs) for frs in part.fragments]
        self.j = -1
        self.ref = self.tier = self.fla = None

    def draw(
        self,
        j: int,
        inputs: dict,
        *,
        l2n: bool = False,
        lower_bound: float = LOWER_BOUND,
    ) -> None:
        """Start draw ``j`` on the row's whole operands ``inputs`` (``make_inputs`` form, the same on every rank):
        rank 0 computes their reference, tier and FLA's values (:meth:`NaiveJudge.fla`)."""
        self.j = j
        self.ref = self.tier = self.fla = None
        if self.naive is not None:
            self.ref, self.tier = self.naive.refs(
                inputs, l2n=l2n, lower_bound=lower_bound
            )
            self.fla = self.naive.fla(inputs, l2n=l2n, lower_bound=lower_bound)

    def judged_outputs(self, res: dict, halve: bool = False) -> dict:
        """The judged outputs of one local result: the token outputs and gradients, the cross-rank parameter sums
        as ``dA_log`` / ``ddt_bias``; ``halve``: every gradient halved (``out`` as it is)."""
        out = {}
        for n, t in res.items():
            if n in self.PARAMS:
                continue
            name = n.removesuffix("_sum")
            t = t.detach()
            out[name] = t / 2 if halve and name != "out" else t
        return out

    def __call__(self, arm: str, res, halve: bool = False) -> None:
        """Gather and judge the arm's local result (``{output: tensor}``, or a list of them: repeats) of the current
        draw."""
        for one in res if isinstance(res, (list, tuple)) else (res,):
            local = self.judged_outputs(one, halve)
            glob = {n: self._global(n, local[n]) for n in sorted(local)}
            if self.naive is not None:
                self.naive.add_refs(self.j, "", {arm: glob}, self.ref, self.tier, self.fla)

    def add_global(self, arm: str, res: dict) -> None:
        """Rank 0, no collective: judge a result already in the global token order (a single-rank run on the row's
        whole operands) as ``arm``; its ``dA_log`` / ``ddt_bias`` are whole gradients."""
        if self.naive is not None:
            self.naive.add_refs(
                self.j,
                "",
                {arm: {n: t.detach() for n, t in res.items()}},
                self.ref,
                self.tier,
                self.fla,
            )

    def _global(self, name: str, t: torch.Tensor):
        """Rank 0: ``t`` of every rank in the global token order (a parameter sum: rank 0's); ``None`` elsewhere."""
        if name in self.PARAMS:
            return t if self.rank == 0 else None
        if self.W == 1:
            return to_global(self.part.fragments, [t], self.cu)
        T_max = max(self.sizes)
        buf = t.new_zeros((T_max, *t.shape[1:]))
        buf[: t.shape[0]] = t
        bufs = [torch.empty_like(buf) for _ in range(self.W)]
        dist.all_gather(bufs, buf)
        if self.rank != 0:
            return None
        return to_global(
            self.part.fragments,
            [b[:n] for b, n in zip(bufs, self.sizes, strict=True)],
            self.cu,
        )

    def result(self) -> dict | None:
        """Rank 0's verdicts and failures (:meth:`NaiveJudge.result`); ``None`` on the other ranks."""
        return None if self.naive is None else self.naive.result()


def judged(lengths, H: int) -> bool:
    """Whether a pack of ``lengths`` at ``H`` heads is judged (:data:`JUDGE_MAX_TOKENS`,
    :data:`JUDGE_MAX_TOKEN_HEADS`)."""
    T = sum(lengths)
    return T <= JUDGE_MAX_TOKENS or T * H <= JUDGE_MAX_TOKEN_HEADS


def stack_refs(judge: NaiveJudge, inputs: dict, layers: int) -> tuple[dict, dict]:
    """The FP64 reference and the naive tier of ``layers`` stacked KDA layers on one case's operands (layer ``l + 1``
    reads layer ``l``'s ``out`` as ``v``; the activated gate; ``dout`` on the last layer): ``out`` (the last layer's),
    ``dq, dk, dv, dg, dbeta`` of the leaves (the sums over the layers that read them) and ``out_l<l>`` of each layer.
    The reference composes the FP64 layers on the exact outputs; the tier the BF16 layers on its BF16 outputs, its
    gradient sums in BF16."""

    def oracle(x: dict) -> dict:
        return oracle_with_grads(x, judge.scale, LOWER_BOUND, False)

    def naive(x: dict) -> dict:
        return naive_of(x, judge.scale, LOWER_BOUND, False)

    ref_in, tier_in = [], []
    v_ref = v_tier = inputs["v"]
    for i in range(layers):
        ref_in.append({**inputs, "v": v_ref})
        tier_in.append({**inputs, "v": v_tier})
        if i < layers - 1:  # the next layer's v: this layer's out
            v_ref, v_tier = oracle(ref_in[-1])["out"], naive(tier_in[-1])["out"]
    ref: dict = {}
    tier: dict = {}
    d_ref = d_tier = inputs["dout"]
    for i in reversed(range(layers)):
        r = oracle({**ref_in[i], "dout": d_ref})
        t = naive({**tier_in[i], "dout": d_tier})
        ref[f"out_l{i}"], tier[f"out_l{i}"] = r["out"], t["out"]
        if i == layers - 1:
            ref["out"], tier["out"] = r["out"], t["out"]
        for n in ("dq", "dk", "dg", "dbeta"):
            ref[n] = r[n] if n not in ref else ref[n] + r[n]
            tier[n] = t[n] if n not in tier else tier[n] + t[n]
        d_ref, d_tier = r["dv"], t["dv"]
    ref["dv"], tier["dv"] = d_ref, d_tier
    return ref, tier


def stack_fla(judge: NaiveJudge, inputs: dict, layers: int) -> FlaReference:
    """FLA's values of the ``layers`` stacked KDA layers of :func:`stack_refs` on one case's operands, keyed as
    :func:`stack_refs`: FLA's layers composed as the test composes the kernels' (layer ``l + 1`` reads FLA's ``out``
    of layer ``l`` as ``v``; ``dout`` on the last layer, FLA's ``dv`` of layer ``l + 1`` as the ``dout`` of layer
    ``l``), the gradients of the leaves summed over the layers in FLA's dtypes, as autograd accumulates them; or the
    reason FLA does not run a layer (the stack's outputs are then judged against the tier only)."""

    def fla(x: dict) -> FlaReference:
        return fla_of(x, judge.scale, LOWER_BOUND, False)

    seconds = 0.0
    v = [inputs["v"]]
    for _ in range(layers - 1):  # the next layer's v: FLA's out of this layer
        step = fla({**inputs, "v": v[-1]})
        if step.values is None:
            return step
        seconds += step.seconds
        v.append(step.values["out"])
    values: dict = {}
    d = inputs["dout"]
    for i in reversed(range(layers)):
        step = fla({**inputs, "v": v[i], "dout": d})
        if step.values is None:
            return step
        seconds += step.seconds
        values[f"out_l{i}"] = step.values["out"] if i == layers - 1 else v[i + 1]
        if i == layers - 1:
            values["out"] = step.values["out"]
        for n in ("dq", "dk", "dg", "dbeta"):
            values[n] = step.values[n] if n not in values else values[n] + step.values[n]
        d = step.values["dv"]
    values["dv"] = d
    return FlaReference(values, None, seconds)


def pack_inputs(part, cu: list[int], ops: dict, device) -> dict:
    """The judge's operands of a one-rank pack: the rank's local operands (fragments in slot order) in the global
    token order, ``cu_seqlens`` int32 on ``device`` and, in the raw gate mode, ``A_log`` / ``dt_bias``."""
    inputs = {n: to_global([part.fragments[0]], [ops[n]], cu) for n in TOKEN_OPERANDS}
    inputs["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device=device)
    for n in ("A_log", "dt_bias"):
        if n in ops:
            inputs[n] = ops[n]
    return inputs


def local_ops(part, rank: int, cu: list[int], glob: dict) -> dict:
    """This rank's local buffer (fragments in slot order) of every token operand of ``glob``; ``A_log`` /
    ``dt_bias`` whole."""
    out = {}
    for n in TOKEN_OPERANDS:
        pieces = [
            glob[n][cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
            for f in part.fragments[rank]
        ]
        out[n] = torch.cat(pieces, 0) if pieces else glob[n][:0].clone()
    for n in ("A_log", "dt_bias"):
        if n in glob:
            out[n] = glob[n]
    return out


def global_results(part, cu: list[int], per_rank: list[dict]) -> dict:
    """Rank 0: the gathered local results of every rank (``{output: tensor}`` each) in the global token order; an
    output without a token axis (``dA_log`` / ``ddt_bias``, already summed over the ranks) from rank 0."""
    out = {}
    for n, t in per_rank[0].items():
        if n in ("dA_log", "ddt_bias"):
            out[n] = t
        else:
            out[n] = to_global(part.fragments, [r[n] for r in per_rank], cu)
    return out


def gather_to_rank0(obj, W: int, rank: int) -> list | None:
    """``obj`` of every rank in rank order on rank 0 (``None`` on the others); ``[obj]`` at ``W = 1``.  Tensors are
    sent from the host."""
    if W == 1:
        return [obj]
    out = [None] * W if rank == 0 else None
    dist.gather_object(obj, out, dst=0)
    return out


def host(res: dict) -> dict:
    """A result's tensors on the host (``{output: tensor}``)."""
    return {n: t.detach().cpu() for n, t in res.items()}


def schedule_arms(r: dict, arm: str = "") -> dict:
    """The judged arms of one pack's :func:`_schedule` results: the saving forward and the saved-mode backward
    (``saved``), the replay-mode forward and backward (``replay``), the forward without a saved set (``nosave``) and the
    checkpointed autograd step (``ckpt``), each prefixed by ``arm``.  The written saved set is not judged (an internal
    of the package; its backward's gradients are)."""
    p = f"{arm}." if arm else ""
    return {
        p + "saved": {"out": r["out"], **dict(zip(BWD_GRADS, r["grads"], strict=True))},
        p + "replay": {
            "out": r["out_replay"],
            **dict(zip(BWD_GRADS, r["grads_replay"], strict=True)),
        },
        p + "nosave": {"out": r["out_nosave"]},
        p + "ckpt": {
            "out": r["out_ckpt"],
            **dict(zip(LEAF_GRADS, r["grads_ckpt"], strict=True)),
        },
    }


def merge_repeats(*arm_sets: dict) -> dict:
    """The arms of several runs of the same arms as repeats (``{arm: [result, ...]}``)."""
    out: dict = {}
    for arms in arm_sets:
        for name, res in arms.items():
            out.setdefault(name, []).append(res)
    return out


def _part(lengths, W: int = 1, layout: str = "contiguous"):
    return kp.cake_kda_nondet_train_kcp_partition(
        torch.tensor(_cu(lengths), dtype=torch.int64), W, layout=layout
    )


def _new_ws(**kw):
    backend = cake_kcp.get_backend()
    return cake_kcp.CakeKDANondetKcpWorkspace(
        state_dtype=backend.dtypes["state"], local=backend.local_stages, **kw
    )


def _storages(ws) -> list[torch.Tensor]:
    """Every device storage the workspace holds, once: the pools, the local arena and the scratch table sets of the
    arena and its cached plan views."""
    out = {t.untyped_storage().data_ptr(): t for t in ws._pools.values() if t.numel()}
    if ws.local is not None:
        out[ws.local.arena.data_ptr()] = ws.local.arena
        for arena in (ws.local, *ws._views.values()):
            tables = arena.__dict__.get(cake_kcp.SCRATCH_TABLES_ATTR)
            if tables:
                first = next(iter(tables.values()))
                out[first.untyped_storage().data_ptr()] = first
    return list(out.values())


def _pointers(ws) -> dict:
    ptrs = {name: t.data_ptr() for name, t in ws._pools.items()}
    if ws.local is not None:
        ptrs["arena"] = ws.local.arena.data_ptr()
        ptrs["arena_id"] = id(ws.local)
        ptrs["views"] = {key: id(v) for key, v in ws._views.items()}
    return ptrs


# --------------------------------------------------------------------------- CPU


def test_byte_query_of_a_list_is_the_per_pool_max() -> None:
    """``cake_kda_nondet_train_kcp_workspace_bytes([ctx, ...])`` (host only): a list of one context is that context's bytes; a list gives
    each pool at its largest need over the contexts (a :data:`SHARED_POOLS` pair one storage of the larger), the
    largest local arena and each distinct pack's scratch table set -- computed here independently from the
    single-context queries; a ``reserve`` of the same list on a pools-only workspace allocates exactly that pool
    term, whichever order the contexts come in, and a later ``ensure`` of any of them grows nothing.  Rows: the
    C / Z / D partitions at W = 2 / 4 / 8, every rank's four packs, H = 4 / 64, backward and forward-only."""
    seen = 0
    for W in (2, 4, 8):
        for layout in ("contiguous", "zigzag", "zigzag_per_document"):
            parts = [_part(lengths, W, layout) for lengths in PACKS]
            for r in range(W):
                ctxs = [
                    kp.cake_kda_nondet_train_kcp_build_context(
                        p, None, r, "cpu", 4, sm_count=SM_COUNT
                    )
                    for p in parts
                ]
                for H in HEADS:
                    for backward in (True, False):
                        kw = {"H": H, "backward": backward}
                        single = [
                            kp.cake_kda_nondet_train_kcp_workspace_bytes(c, **kw)
                            for c in ctxs
                        ]
                        assert (
                            kp.cake_kda_nondet_train_kcp_workspace_bytes(
                                [ctxs[0]], **kw
                            )
                            == single[0]
                        )
                        lays = [kp.kcp_arena_layout(c, **kw) for c in ctxs]
                        need = {
                            key[5:]: max(lay[key] for lay in lays)
                            for key in lays[0]
                            if key.startswith("pool.")
                        }
                        lazy = kp.LAZY_POOLS + (() if backward else kp.FWD_LAZY_POOLS)
                        paired = {n for pair in kp.SHARED_POOLS for n in pair}
                        pools = sum(
                            v
                            for n, v in need.items()
                            if n not in lazy and n not in paired
                        ) + sum(
                            max(need.get(n, 0) for n in pair)
                            for pair in kp.SHARED_POOLS
                        )
                        local = [
                            s - lay["arena"]
                            for s, lay in zip(single, lays, strict=True)
                        ]
                        arena = [
                            cake_backend.workspace_bytes(
                                c.T_r,
                                c.n_frag,
                                H,
                                backward=backward,
                                sm_count=SM_COUNT,
                                fwd_layout=kp.local_fwd_layout(backward),
                            )
                            if c.T_r
                            else 0
                            for c in ctxs
                        ]
                        scratch = {
                            (c.T_r, c.n_frag): loc - a
                            for c, loc, a in zip(ctxs, local, arena, strict=True)
                        }
                        got = kp.cake_kda_nondet_train_kcp_workspace_bytes(ctxs, **kw)
                        assert got == max(arena) + sum(scratch.values()) + pools, (
                            W,
                            layout,
                            r,
                            H,
                            backward,
                        )
                        assert max(single) <= got < sum(single)
                        if backward:
                            for order in (ctxs, ctxs[::-1]):
                                ws = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
                                ws.reserve(order, H)
                                assert ws.bytes == pools, (W, layout, r, H)
                                ptrs = _pointers(ws)
                                for c in order:
                                    ws.ensure(c, H, device="cpu", backward=False)
                                assert _pointers(ws) == ptrs and ws.bytes == pools
                            ws_r = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
                            ws_r.reserve(ctxs, H, ring=True)  # the hop pools too
                            assert ws_r.bytes == pools + sum(
                                max(lay[f"pool.{n}"] for lay in lays)
                                for n in kp.LAZY_POOLS
                            )
                        seen += 1
    assert seen > 0


def test_reserve_floor_and_allocate_false_size_forwards_for_the_backward() -> None:
    """``reserve`` and ``CakeKDANondetKcpWorkspace(ctx, H, allocate=False)`` keep a backward sizing floor: a later forward ensure
    sizes the backward pools too (no forward-only state to upgrade); ``allocate=False`` allocates nothing until the
    first ensure; a workspace without a context keeps today's forward-only first state."""
    part = _part((1000, 577), 4, "zigzag_per_document")
    ctx = kp.cake_kda_nondet_train_kcp_build_context(
        part, None, 1, "cpu", 4, sm_count=SM_COUNT
    )
    full = kp.kcp_arena_layout(ctx, H=4)["arena"]
    fwd = kp.kcp_arena_layout(ctx, H=4, backward=False)["arena"]
    assert fwd < full
    ws = cake_kcp.CakeKDANondetKcpWorkspace(
        ctx, 4, device="cpu", local=False, allocate=False
    )
    assert ws.bytes == 0 and not ws._pools and ws.local is None
    ws.ensure(ctx, 4, device="cpu", backward=False)
    assert ws.bytes == full and ws.has("b1_send")
    plain = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
    plain.ensure(ctx, 4, device="cpu", backward=False)
    assert plain.bytes == fwd and not plain.has("b1_send")
    reserved = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
    reserved.reserve([ctx], 4)
    ptrs = _pointers(reserved)
    reserved.ensure(ctx, 4, device="cpu", backward=False)
    assert reserved.bytes == full and _pointers(reserved) == ptrs
    with pytest.raises(ValueError):
        reserved.reserve([], 4)


def test_shared_workspace_holds_no_context_and_bounds_its_caches() -> None:
    """A long-lived workspace keeps no context alive (the trimmed path's ensure records hold the context and its
    table set weakly; the table set lives on the context); its pool-view and ensure-record caches stay bounded,
    and a record per context lets interleaved packs keep the fast path."""
    part = _part((1000, 577))
    ws = cake_kcp.CakeKDANondetKcpWorkspace(local=False)
    refs = []
    for _ in range(3):
        ctx = kp.cake_kda_nondet_train_kcp_build_context(
            part, None, 0, "cpu", 4, sm_count=SM_COUNT
        )
        tb = ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True)
        assert ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is tb
        refs.append((weakref.ref(ctx), weakref.ref(tb)))
        del ctx, tb
    gc.collect()
    assert all(a() is None and b() is None for a, b in refs)
    assert ws._fast and all(rec[0]() is None for rec in ws._fast.values())
    # a dead ensure record misses: the next ensure takes the full path
    ctx = kp.cake_kda_nondet_train_kcp_build_context(
        part, None, 0, "cpu", 4, sm_count=SM_COUNT
    )
    assert ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is not None
    n = ws._pools["c1_send"].numel()
    for i in range(1, cake_kcp.SLICE_CACHE_MAX + 50):
        ws.pool("c1_send", min(i, n), cached=True)
        assert len(ws._slices) <= cake_kcp.SLICE_CACHE_MAX
    # interleaved packs: one record each, every one a hit while no storage changes
    other = kp.cake_kda_nondet_train_kcp_build_context(
        _part((50, 640, 77)), None, 0, "cpu", 4, sm_count=SM_COUNT
    )
    for _ in range(2):  # the first round may grow a pool, which drops the records
        tbs = [
            ws.ensure(c, 64, device="cpu", backward=True, host_trim=True)
            for c in (ctx, other)
        ]
    calls = []
    real = ws.tables
    ws.tables = lambda *a, **k: calls.append(1) or real(*a, **k)
    for _ in range(3):
        for c, tb in zip((ctx, other), tbs, strict=True):
            assert ws.ensure(c, 64, device="cpu", backward=True, host_trim=True) is tb
    assert not calls  # no full-path ensure
    ws.tables = real
    part = _part((50, 640, 77))
    for _ in range(cake_kcp.ENSURE_CACHE_MAX + 10):
        c = kp.cake_kda_nondet_train_kcp_build_context(
            part, None, 0, "cpu", 4, sm_count=SM_COUNT
        )
        ws.ensure(c, 64, device="cpu", backward=True, host_trim=True)
        assert len(ws._fast) <= cake_kcp.ENSURE_CACHE_MAX


def test_fla_values_of_dw_and_of_the_layer_stacks_compose_as_the_tier(monkeypatch) -> None:
    """FLA's ``dw`` (:meth:`NaiveJudge.fla`) and FLA's layer stack (:func:`stack_fla`) are composed as the naive
    tier's (:meth:`NaiveJudge.refs`, :func:`stack_refs`): with FLA's step replaced by the naive tier (on the CPU),
    every value is the tier's bit for bit, three steps for two layers; a step FLA does not run gives its reason for
    every output."""
    steps = []

    def tier_as_fla(inputs, scale, lower_bound, l2n):
        steps.append(inputs["v"])
        return FlaReference(naive_of(inputs, scale, lower_bound, l2n), None, 0.25)

    monkeypatch.setitem(globals(), "fla_of", tier_as_fla)
    judge = NaiveJudge()
    inputs = make_inputs((5, 3), 2, "cpu", seed=0x7815)
    _, tier = judge.refs(inputs)
    one = judge.fla(inputs)
    assert sorted(one.values) == sorted(tier) and one.reason is None
    for name, want in tier.items():
        assert torch.equal(one.values[name], want), name
    steps.clear()
    _, tier = stack_refs(judge, inputs, 2)
    stack = stack_fla(judge, inputs, 2)
    assert stack.reason is None and stack.seconds == 0.75 and len(steps) == 3
    # layer 1 reads FLA's out of layer 0 as v; layer 0 runs on the operands' v
    assert torch.equal(steps[1], tier["out_l0"]) and steps[2] is inputs["v"]
    assert sorted(stack.values) == sorted(tier)
    for name, want in tier.items():
        assert torch.equal(stack.values[name], want), name
    assert stack.value("out", tier["out"]) is stack.values["out"]

    calls = []

    def layer_one_fails(inputs, scale, lower_bound, l2n):
        calls.append(1)
        if len(calls) == 2:  # the step of layer 1
            return FlaReference(None, "fla unavailable: a test reason")
        return tier_as_fla(inputs, scale, lower_bound, l2n)

    monkeypatch.setitem(globals(), "fla_of", layer_one_fails)
    stack = stack_fla(judge, inputs, 2)
    assert stack.values is None and stack.reason == "fla unavailable: a test reason"
    assert stack.value("dv", tier["dv"]) == "fla unavailable: a test reason"
    monkeypatch.setitem(globals(), "fla_of", lambda *a: FlaReference(None, "fla unavailable: a test reason"))
    assert judge.fla(inputs).value("dw", tier["out"]) == "fla unavailable: a test reason"


# --------------------------------------------------------------------------- GPU


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {tuple(cake_jit.ARCHES)}"
        )
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


def _operands(device, T: int, H: int, seed: int) -> dict:
    """Seeded operands of the delta rule's contract (q / k L2-normalized per head, natural-log decay in (-5, 0)) and
    a ``dout``, in the backend's dtypes."""
    dt = cake_kcp.get_backend().dtypes
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)

    def unit(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    D = cake_kcp.D
    return {
        "q": unit(randn(T, H, D)).to(dt["q"]),
        "k": unit(randn(T, H, D)).to(dt["k"]),
        "v": (0.5 * randn(T, H, D)).to(dt["v"]),
        "g": (-5.0 * torch.sigmoid(randn(T, H, D))).to(dt["g"]),
        "beta": torch.sigmoid(randn(T, H)).to(dt["beta"]),
        "dout": (0.5 * randn(T, H, D)).to(dt["dout"]),
    }


def _packs(
    device,
    H: int,
    packs=PACKS,
    W: int = 1,
    rank: int = 0,
    layout="contiguous",
    seed: int = JUDGE_SEED,
):
    ctxs = [
        kp.cake_kda_nondet_train_kcp_build_context(
            _part(lengths, W, layout), None, rank, device
        )
        for lengths in packs
    ]
    return ctxs, _draw_ops(ctxs, H, device, seed)


def _draw_ops(ctxs, H: int, device, seed: int) -> list[dict]:
    """The operands of one draw of one-rank packs: pack ``i`` at ``seed + i``."""
    return [_operands(device, int(c.T_r), H, seed + i) for i, c in enumerate(ctxs)]


def _judge_packs(
    judge: NaiveJudge, j: int, case: str, packs, ops, arms: list[dict], device
) -> None:
    """Draw ``j`` of every judged one-rank pack of ``packs`` (case ``"<case>p<i>"``; :func:`judged`): its arms
    ``arms[i]`` on the pack's operands ``ops[i]``.  A larger pack keeps its exact checks only."""
    for i, lengths in enumerate(packs):
        if not judged(lengths, int(ops[i]["q"].shape[1])):
            continue
        inputs = pack_inputs(_part(lengths), _cu(lengths), ops[i], device)
        judge.add(j, f"{case}p{i}", arms[i], inputs)


def draws_of(packs, H: int) -> int:
    """The draws of a test over ``packs`` at ``H`` heads: the judge's when a pack is judged (:func:`judged`), else one
    (the exact checks only)."""
    return DRAWS if any(judged(lengths, H) for lengths in packs) else 1


def _fwd(ctx, o, ws, **kw):
    return cake_kcp.cake_kda_nondet_train_kcp_forward(
        o["q"],
        o["k"],
        o["v"],
        o["g"],
        o["beta"],
        ctx,
        workspace=ws,
        **kw,
    )


def _bwd(ctx, o, ws, saved, **kw):
    return cake_kcp.cake_kda_nondet_train_kcp_backward(
        o["q"],
        o["k"],
        o["v"],
        o["g"],
        o["beta"],
        ctx,
        o["dout"],
        saved=saved,
        workspace=ws,
        **kw,
    )


def _schedule(ctxs, ops, wss, *, poison=None, **kw) -> list[dict]:
    """One interleaved pass over the packs (``wss[i]``: the workspace of pack ``i``; one object for a shared one):
    the saving forwards in pack order, the replay-mode forwards in reverse order, the saved-mode backwards in pack
    order, the replay-mode backwards in reverse order, the forwards without a saved set, then the checkpointed
    autograd forwards of every pack before one backward through all of them.  ``poison(ws)`` runs before every
    forward (the arena filled with NaN).  Returns per pack its outputs, written saved sets and gradients."""
    n = len(ctxs)
    if not isinstance(wss, list):  # one shared workspace
        wss = [wss] * n
    res: list[dict] = [{} for _ in range(n)]
    saved = [None] * n

    def pre(i):
        if poison is not None:
            poison(wss[i])

    for i in range(n):
        pre(i)
        out, s = _fwd(ctxs[i], ops[i], wss[i], **kw)
        saved[i] = s
        res[i]["out"] = out
        res[i]["saved"] = written_tensors(s)
    for i in reversed(range(n)):
        pre(i)
        out, none = _fwd(ctxs[i], ops[i], wss[i], boundary_mode="replay", **kw)
        assert none is None
        res[i]["out_replay"] = out
    for i in range(n):
        res[i]["grads"] = _bwd(ctxs[i], ops[i], wss[i], saved[i], **kw)
    for i in reversed(range(n)):
        res[i]["grads_replay"] = _bwd(ctxs[i], ops[i], wss[i], None, **kw)
    for i in range(n):
        pre(i)
        out, none = _fwd(ctxs[i], ops[i], wss[i], save_for_backward=False, **kw)
        res[i]["out_nosave"] = out
    leaves, outs = [], []
    for i in range(n):
        pre(i)
        lf = [
            ops[i][name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        ]
        out = torch.utils.checkpoint.checkpoint(
            lambda *a, i=i: cake_kcp.cake_kda_nondet_kcp(
                *a, ctxs[i], None, "light", "saved", wss[i], **kw
            ),
            *lf,
            use_reentrant=False,
        )
        leaves.append(lf)
        outs.append(out)
    torch.autograd.backward(outs, [o["dout"] for o in ops])
    for i in range(n):
        res[i]["out_ckpt"] = outs[i].detach()
        res[i]["grads_ckpt"] = tuple(t.grad for t in leaves[i])
    return res


def _poison(ws) -> None:
    if ws.local is not None:
        ws.local.arena.fill_(0xFF)


def test_gpu_reserved_shared_workspace_per_pack_judged(device, record_property) -> None:
    """Four packs of one rank (different ``T_r`` / ``n_frag``; P > 1 and P = 1) at H = 4 and 64 on ONE reserved
    workspace, in an interleaved schedule (saving and replay-mode forwards, saved and replay-mode backwards, forwards
    without a saved set, checkpointed autograd through all packs): out and every gradient of one workspace per pack
    (``per``), of the shared workspace with the arena NaN-filled before every forward (``shared_poisoned``) and of a
    second schedule on it (``shared``), each judged per pack against the FP64 reference beside the naive tier (case
    ``h<H>.p<i>``).  ``reserve`` allocates exactly the workspace's storages (``memory_allocated`` delta = their
    allocator-rounded bytes) and its ``bytes`` equal ``cake_kda_nondet_train_kcp_workspace_bytes(ctxs, H=H)``, below
    the sum of the per-pack workspaces; the local arena is the largest per-pack arena; after every schedule of every
    draw no pointer, arena, view or byte count has changed."""
    judge = NaiveJudge()
    seen_p1 = seen_pn = 0
    for H in HEADS:
        ctxs, _ = _packs(device, H)
        Ps = [
            cake_kcp.CakeKDANondetKcpWorkspace(local=False).tables(c, H).P for c in ctxs
        ]
        seen_p1 += sum(P == 1 for P in Ps)
        seen_pn += sum(P > 1 for P in Ps)
        ws = ptrs = pred = None
        for j in range(DRAWS):
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
            per = [_new_ws() for _ in ctxs]
            for c, w in zip(ctxs, per, strict=True):
                w.ensure(c, H, device=device, backward=True)
            if ws is None:
                per_bytes = [w.bytes for w in per]
                assert per_bytes == [
                    kp.cake_kda_nondet_train_kcp_workspace_bytes(c, H=H) for c in ctxs
                ]
                per_arena = max(w.local.bytes for w in per)
            ref = _schedule(ctxs, ops, per)
            del per
            if ws is None:
                gc.collect()
                torch.cuda.synchronize(device)
                base = torch.cuda.memory_allocated(device)
                ws = _new_ws()
                got = ws.reserve(ctxs, H)
                torch.cuda.synchronize(device)
                delta = torch.cuda.memory_allocated(device) - base
                rounded = sum(
                    -(-t.untyped_storage().nbytes() // 512) * 512 for t in _storages(ws)
                )
                pred = kp.cake_kda_nondet_train_kcp_workspace_bytes(ctxs, H=H)
                print(
                    f"SHARED_WS H={H} P={Ps} per_pack={per_bytes} sum={sum(per_bytes)} max={max(per_bytes)} "
                    f"shared={ws.bytes} pred={pred} delta={delta} rounded={rounded}"
                )
                assert got == ws.bytes == pred, (H, got, pred)
                # the caching allocator may hand out a cached block up to 1 MiB larger than a request without
                # splitting it
                assert rounded <= delta < rounded + len(_storages(ws)) * (1 << 20), (
                    H,
                    delta,
                    rounded,
                )
                assert ws.local.bytes == per_arena and ws.local.backward
                assert ws.bytes < sum(per_bytes)
                ptrs = _pointers(ws)
            got = []
            for rnd in range(2):
                got.append(
                    _schedule(ctxs, ops, ws, poison=_poison if rnd == 0 else None)
                )
                assert _pointers(ws) == ptrs and ws.bytes == pred, (H, j, rnd)
            arms = [
                {
                    **schedule_arms(ref[i], "per"),
                    **schedule_arms(got[0][i], "shared_poisoned"),
                    **schedule_arms(got[1][i], "shared"),
                }
                for i in range(len(ctxs))
            ]
            _judge_packs(judge, j, f"h{H}.", PACKS, ops, arms, device)
            del ref, got, arms
        del ws
    assert seen_p1 > 0 and seen_pn > 0, (seen_p1, seen_pn)
    judge.settle(record_property)


def test_gpu_unreserved_shared_workspace_grows_to_the_max_only(
    device, record_property
) -> None:
    """Without ``reserve``: a workspace that first serves the smallest pack grows its arena only when a pack does not
    fit (a larger pack's arena replaces it, released first), never for a smaller one, and a forward never turns its
    backward arena into a forward-only one (the first draw); after one pass over the packs it holds the bytes of the
    reserved form (every draw).  The schedule's results on one workspace per pack (``per``) and on the grown
    workspace (``unreserved``) are judged per pack (case ``p<i>``, the packs by ``T_r``)."""
    H = 4
    ctxs, _ = _packs(device, H)
    order = sorted(range(len(ctxs)), key=lambda i: ctxs[i].T_r)
    ctxs = [ctxs[i] for i in order]
    packs = [PACKS[i] for i in order]
    judge = NaiveJudge()
    ws = _new_ws()
    for draw in range(DRAWS):
        ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, draw))
        ref = _schedule(ctxs, ops, [_new_ws() for _ in ctxs])
        if draw == 0:
            replaced = 0
            for j, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
                before = ws.local
                out, saved = _fwd(c, o, ws)
                if (
                    j > 0
                ):  # a forward never turns the backward arena into a forward-only one
                    assert ws.local.backward, j
                    replaced += ws.local is not before
                _bwd(c, o, ws, saved)
                assert ws.local.backward, j
                cur = ws.local
                for k in range(j):
                    # every pack seen before fits: it runs on a view, no replacement
                    out_k, _ = _fwd(ctxs[k], ops[k], ws)
                    assert ws.local is cur, (j, k)
                    del out_k
                del out, saved
            print(f"UNRESERVED replaced={replaced} bytes={ws.bytes}")
        res = _schedule(ctxs, ops, ws)
        assert ws.bytes == kp.cake_kda_nondet_train_kcp_workspace_bytes(ctxs, H=H)
        arms = [
            {**schedule_arms(ref[i], "per"), **schedule_arms(res[i], "unreserved")}
            for i in range(len(ctxs))
        ]
        _judge_packs(judge, draw, "", packs, ops, arms, device)
        del ref, res, arms
    judge.settle(record_property)


def test_gpu_allocate_false_sizes_the_first_forward_for_the_backward(
    device, record_property
) -> None:
    """B T3 with the sizing floor (compared after the first backward), on every draw: ``CakeKDANondetKcpWorkspace(ctx,
    H, allocate=False)`` allocates nothing; its first forward allocates the backward workspace
    (``cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=H)``, the backward arena) and the first backward allocates
    nothing more and moves no pointer.  Out and the five gradients of ``CakeKDANondetKcpWorkspace(ctx, H)`` (``ensured``)
    and of the ``allocate=False`` workspace (``lazy``) are judged (case ``h<H>``)."""
    judge = NaiveJudge()
    for H in HEADS:
        ctxs, _ = _packs(device, H, packs=PACKS[:2])
        ctx = ctxs[0]
        for j in range(DRAWS):
            o = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))[0]
            ref_ws = _new_ws()
            ref_ws.ensure(ctx, H, device=device)
            out_r, saved_r = _fwd(ctx, o, ref_ws)
            ensured = host(
                {
                    "out": out_r,
                    **dict(zip(BWD_GRADS, _bwd(ctx, o, ref_ws, saved_r), strict=True)),
                }
            )
            del ref_ws, out_r, saved_r
            gc.collect()
            torch.cuda.synchronize(device)
            base = torch.cuda.memory_allocated(device)
            ws = _new_ws(ctx=ctx, H=H, device=device, allocate=False)
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == base and ws.bytes == 0
            out, saved = _fwd(ctx, o, ws)
            assert (
                ws.local.backward
                and ws.bytes == kp.cake_kda_nondet_train_kcp_workspace_bytes(ctx, H=H)
            )
            ptrs = _pointers(ws)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_allocated(device)
            grads = _bwd(ctx, o, ws, saved)
            assert _pointers(
                ws
            ) == ptrs and ws.bytes == kp.cake_kda_nondet_train_kcp_workspace_bytes(
                ctx, H=H
            )
            lazy = host({"out": out, **dict(zip(BWD_GRADS, grads, strict=True))})
            del grads
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) <= before, (H, j)
            del ws, out, saved
            _judge_packs(
                judge,
                j,
                f"h{H}.",
                PACKS[:1],
                [o],
                [{"ensured": ensured, "lazy": lazy}],
                device,
            )
    judge.settle(record_property)


def test_gpu_four_pack_step_peak_drops_by_sum_minus_max(
    device, record_property
) -> None:
    """The in-repo memory proxy (four prepared packs, one checkpointed KDA layer per pack, forwards of all packs then
    one backward through all of them, at H = 64; the first draw): the step's peak with one workspace per pack minus
    the step's with one reserved shared workspace equals the per-pack workspaces' bytes minus the shared one's (within
    1 MiB).  The peaks count the bytes the callers requested (``torch.cuda.memory_stats`` ``requested_bytes``), as
    the workspaces' ``bytes`` do: ``max_memory_allocated`` counts the caching allocator's blocks, whose rounding and
    the slack of an unsplit cached block (up to 1 MiB per tensor) differ between the arms (four workspaces' pool
    tensors against one's, and the step's own tensors placed around the per-pack arm's gradients, which stay alive
    through the shared arm).  Out and the five gradients of the step on one workspace per pack (``per``) and on the
    shared workspace (``shared``) are judged per pack on every draw (case ``p<i>``)."""
    H = 64
    ctxs, ops = _packs(device, H)

    def requested(stat: str) -> int:
        return int(torch.cuda.memory_stats(device)[f"requested_bytes.all.{stat}"])

    def step(wss) -> tuple[int, list]:
        torch.cuda.synchronize(device)
        base = requested("current")
        torch.cuda.reset_peak_memory_stats(device)
        leaves, outs = [], []
        for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
            lf = [
                o[name].detach().clone().requires_grad_(True)
                for name in ("q", "k", "v", "g", "beta")
            ]
            outs.append(
                torch.utils.checkpoint.checkpoint(
                    lambda *a, c=c, i=i: cake_kcp.cake_kda_nondet_kcp(
                        *a, c, None, "light", "saved", wss[i]
                    ),
                    *lf,
                    use_reentrant=False,
                )
            )
            leaves.append(lf)
        torch.autograd.backward(outs, [o["dout"] for o in ops])
        torch.cuda.synchronize(device)
        peak = requested("peak") - base
        grads = [
            [outs[i].detach()] + [t.grad for t in leaves[i]] for i in range(len(ctxs))
        ]
        return peak, grads

    def arm(shared: bool) -> tuple[int, int, list]:
        gc.collect()
        torch.cuda.synchronize(device)
        base = requested("current")
        if shared:
            ws = _new_ws()
            ws.reserve(ctxs, H)
            wss = [ws] * len(ctxs)
            held = ws.bytes
        else:
            wss = [_new_ws() for _ in ctxs]
            for c, w in zip(ctxs, wss, strict=True):
                w.ensure(c, H, device=device)
            held = sum(w.bytes for w in wss)
        step(wss)  # warm-up: the caching allocator's blocks of the step's own tensors
        gc.collect()
        torch.cuda.synchronize(device)
        mid = requested("current")
        peak, grads = step(wss)
        peak_total = mid - base + peak
        del wss
        return held, peak_total, grads

    def named(grads: list) -> list[dict]:
        return [
            host({"out": g[0], **dict(zip(LEAF_GRADS, g[1:], strict=True))})
            for g in grads
        ]

    warm = [_new_ws() for _ in ctxs]
    step(
        warm
    )  # the contexts' tables are copied by their first calls: before either arm
    del warm
    judge = NaiveJudge()
    for j in range(DRAWS):
        if j:
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        held_p, peak_p, grads_p = arm(False)
        held_s, peak_s, grads_s = arm(True)
        if j == 0:
            print(
                f"PROXY4 H={H} per_pack_ws={held_p} shared_ws={held_s} peak_per_pack={peak_p} "
                f"peak_shared={peak_s} drop={peak_p - peak_s} expected={held_p - held_s}"
            )
            assert abs((peak_p - peak_s) - (held_p - held_s)) <= (1 << 20), (
                peak_p,
                peak_s,
                held_p,
                held_s,
            )
        arms = [
            {"per": p, "shared": s}
            for p, s in zip(named(grads_p), named(grads_s), strict=True)
        ]
        del grads_p, grads_s
        _judge_packs(judge, j, "", PACKS, ops, arms, device)
    judge.settle(record_property)


def test_gpu_shared_workspace_steady_calls_do_not_block_or_allocate(
    device, record_property
) -> None:
    """Census (``tests/experimental/test_helpers/cake_host_sync.py``) of a steady interleaved round of the four packs
    (saving forwards, then saved-mode backwards in reverse order) on one reserved workspace (the first draw): no
    pageable or pinned copy, no synchronizing operation, the queued probe still runs when the host returns, and no
    device allocation or free.  Out and the five gradients of the first round (``first``) and of the steady round
    (``steady``, the census's on the first draw) are judged per pack on every draw (case ``p<i>``)."""
    alloc_ok = allocator_counts(device)
    H = 4
    ctxs, ops = _packs(device, H)
    ws = _new_ws()
    ws.reserve(ctxs, H)

    def rnd():
        saved, res = [], [{} for _ in ctxs]
        for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
            out, s = _fwd(c, o, ws)
            saved.append(s)
            res[i]["out"] = out
        for i in reversed(range(len(ctxs))):
            grads = _bwd(ctxs[i], ops[i], ws, saved[i])
            res[i].update(zip(BWD_GRADS, grads, strict=True))
        return res

    judge = NaiveJudge()
    for j in range(DRAWS):
        if j:
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        first = rnd()
        rnd()
        if j == 0:
            c = census(rnd, device)
            print(f"CENSUS_SHARED {summary(c)}")
            assert nonblocking(c), summary(c)
            assert c["pinned"] == 0, summary(c)
            if alloc_ok:
                assert (c["device_alloc"], c["device_free"], c["alloc_retries"]) == (
                    0,
                    0,
                    0,
                ), summary(c)
                assert c["prof_malloc"] in (None, 0), summary(c)
            steady = c["out"]
            del c
        else:
            steady = rnd()
        arms = [{"first": a, "steady": b} for a, b in zip(first, steady, strict=True)]
        _judge_packs(judge, j, "", PACKS, ops, arms, device)
        del first, steady, arms
    judge.settle(record_property)
    if not alloc_ok:
        pytest.skip(
            "the caching allocator's statistics do not count device allocations here"
        )


def test_gpu_fresh_contexts_on_a_long_lived_shared_workspace_release_everything(
    device, record_property
) -> None:
    """A training loop that prepares NEW contexts every step and runs them on one long-lived reserved workspace keeps
    no context: with the cycle collector disabled every step's contexts and table sets are freed by reference
    counting alone and ``memory_allocated`` returns to its level after the first step (the first draw: three more
    steps).  Out and the five gradients of the first step (``first``) and of the later steps (``later``) are judged
    per pack on every draw (case ``p<i>``; one later step on the other draws)."""
    H = 4

    def step(ws, seed: int):
        ctxs, ops = _packs(device, H, seed=seed)
        ws.reserve(ctxs, H)
        res = []
        for c, o in zip(ctxs, ops, strict=True):
            out, s = _fwd(c, o, ws)
            grads = _bwd(c, o, ws, s)
            res.append(host({"out": out, **dict(zip(BWD_GRADS, grads, strict=True))}))
            del out, s, grads
        refs = [weakref.ref(c) for c in ctxs] + [
            weakref.ref(ws.tables(c, H)) for c in ctxs
        ]
        return res, refs, ops

    judge = NaiveJudge()
    ws = _new_ws()
    for j in range(DRAWS):
        seed = draw_seed(JUDGE_SEED, j)
        first, refs, ops = step(ws, seed)
        gc.collect()
        torch.cuda.synchronize(device)
        assert all(r() is None for r in refs)
        later = []
        if j == 0:
            base = torch.cuda.memory_allocated(device)
            enabled = gc.isenabled()
            gc.disable()
            try:
                for i in range(3):
                    res, refs = step(ws, seed)[:2]  # the step's operands freed here
                    later.append(res)
                    del res
                    torch.cuda.synchronize(device)
                    assert all(r() is None for r in refs), (
                        i,
                        "a context or table set is retained",
                    )
                    assert torch.cuda.memory_allocated(device) == base, i
            finally:
                if enabled:
                    gc.enable()
        else:
            later.append(step(ws, seed)[0])
        arms = [
            {"first": first[i], "later": [r[i] for r in later]}
            for i in range(len(first))
        ]
        _judge_packs(judge, j, "", PACKS, ops, arms, device)
        del first, later, arms, ops
    judge.settle(record_property)


def test_gpu_call_on_another_stream_waits_for_the_last_one(
    device, record_property
) -> None:
    """Stream rule: a call on another stream is ordered after the work the workspace's last stream queued (one
    ``wait_stream``, no host wait, no error).  The order is checked by events on the first draw, with no timing
    assumption: once the second stream's work is complete, an event recorded after the first stream's queued call
    (behind ~1.5 s of device sleep) is complete too.  Without the wait the second stream's work completes while the
    first stream still sleeps.  Out and the five gradients of the second pack's call on one stream (``one_stream``)
    and on the second stream (``other_stream``) are judged on every draw (the other draws without the sleep)."""
    H = 4
    ctxs, ops = _packs(device, H, packs=PACKS[:2])
    ws = _new_ws()
    ws.reserve(ctxs, H)
    judge = NaiveJudge()
    for j in range(DRAWS):
        if j:
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        ref_out, ref_saved = _fwd(ctxs[1], ops[1], ws)
        ref_grads = _bwd(ctxs[1], ops[1], ws, ref_saved)
        one_stream = {"out": ref_out, **dict(zip(BWD_GRADS, ref_grads, strict=True))}
        torch.cuda.synchronize(device)
        s1, s2 = torch.cuda.Stream(device), torch.cuda.Stream(device)
        assert s1.cuda_stream != s2.cuda_stream
        cycles = sleep_cycles(device, 1500.0) if j == 0 else 0
        for s in (s1, s2):
            s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s1):
            if cycles:
                torch.cuda._sleep(cycles)
            out0, _ = _fwd(ctxs[0], ops[0], ws)
            first = torch.cuda.Event()
            first.record()
        t0 = time.perf_counter()
        with torch.cuda.stream(s2):
            out, saved = _fwd(ctxs[1], ops[1], ws)
            grads = _bwd(ctxs[1], ops[1], ws, saved)
            done = torch.cuda.Event()
            done.record()
        host_s = time.perf_counter() - t0
        done.synchronize()
        # complete whenever the second stream's work is, if that work waited for it
        ordered = first.query()
        torch.cuda.synchronize(device)
        if j == 0:
            assert host_s < 1.0, host_s
            assert ordered, (
                "the call on the second stream did not wait for the first stream's queued work"
            )
        assert ws._stream is not None and ws._stream[0] == s2.cuda_stream
        other_stream = {"out": out, **dict(zip(BWD_GRADS, grads, strict=True))}
        _judge_packs(
            judge,
            j,
            "",
            PACKS[1:2],
            ops[1:2],
            [{"one_stream": one_stream, "other_stream": other_stream}],
            device,
        )
        del out0, out, saved, grads, ref_out, ref_saved, ref_grads
    judge.settle(record_property)


def _live_ranges(ws) -> dict[int, int]:
    """``data_ptr -> nbytes`` of every device storage the workspace holds (:func:`_storages`)."""
    return {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in _storages(ws)
    }


def _hits(tensors, ranges: dict[int, int]) -> list[tuple[int, int]]:
    """The ``(data_ptr, nbytes)`` of ``ranges`` that one of ``tensors`` overlaps."""
    out = []
    for t in tensors:
        a = t.untyped_storage().data_ptr()
        b = a + t.untyped_storage().nbytes()
        out += [(p, n) for p, n in ranges.items() if a < p + n and p < b]
    return out


SWITCH_PACKS = (PACKS[0], (4000, 1700))


def _switch_setup(device, H: int, seed: int = JUDGE_SEED):
    """A small pack and a pack whose arena and pools are larger (their tables already copied by a first call), and a
    workspace reserved for the small pack on a fresh stream ``sa``, with the large pack's tables built; plus a second
    fresh stream ``sb``.  The packs' operands at ``seed`` and ``seed + 1``."""
    small, small_ops = _packs(device, H, packs=SWITCH_PACKS[:1], seed=seed)
    big, big_ops = _packs(device, H, packs=SWITCH_PACKS[1:], seed=seed + 1)
    warm = _new_ws()
    _fwd(small[0], small_ops[0], warm)
    _fwd(big[0], big_ops[0], warm)
    del warm
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    sa, sb = torch.cuda.Stream(device), torch.cuda.Stream(device)
    with torch.cuda.stream(sa):
        ws = _new_ws()
        ws.reserve(small, H)
        ws.tables(big[0], H)
    torch.cuda.synchronize(device)
    return (small[0], small_ops[0]), (big[0], big_ops[0]), ws, sa, sb


# device sleep of the stream-switch rows, in ms: a later attempt runs only when stream B finished before the host's
# check (an inconclusive attempt, never a pass); each attempt's results are checked whatever its timing
SWITCH_SLEEPS_MS = (1000.0, 4000.0, 16000.0)


def test_gpu_storage_a_switched_stream_used_is_not_reused_while_it_runs(device) -> None:
    """Allocator rule of a stream switch (``record_stream``): a workspace allocated on stream A serves a call on
    stream B queued behind device sleep, and an ``ensure`` on B for a larger pack then replaces its pools and its
    arena.  While B is still busy (checked after the allocations; an attempt where B already finished is repeated
    with a longer sleep), allocations on A of every released storage's size (twice each) never overlap a released
    storage: the caching allocator returns a block to the stream that allocated it, so without the record it would
    hand A the block B's queued work still reads and writes (the negative control: the previous head)."""
    H = 4
    for ms in SWITCH_SLEEPS_MS:
        (sc, so), (bc, _bo), ws, sa, sb = _switch_setup(device, H)
        old = _live_ranges(ws)
        cycles = sleep_cycles(device, ms)
        with torch.cuda.stream(sb):
            torch.cuda._sleep(cycles)
            out, _ = _fwd(sc, so, ws, save_for_backward=False)
            ws.ensure(bc, H, device=device, backward=True)
        live = _live_ranges(ws)
        freed = {p: n for p, n in old.items() if p not in live}
        held = []
        with torch.cuda.stream(sa):
            for n in freed.values():
                for _ in range(2):
                    held.append(torch.empty((n,), dtype=torch.uint8, device=device))
        busy = not sb.query()  # B had not finished when the allocations were made
        hits = _hits(held, freed)
        torch.cuda.synchronize(device)
        print(
            f"STREAM_SWITCH_REUSE sleep_ms={ms} freed={sorted(freed.values())} held={len(held)} busy={busy} "
            f"hits={hits}"
        )
        assert ws.local is not None and ws.local.arena.data_ptr() not in old
        assert len(freed) >= 2, (old, live)  # the arena and at least one pool
        del out, held, ws
        if busy:
            assert not hits, (
                f"a storage stream B's queued work uses was handed to stream A: {hits}"
            )
            return
    pytest.fail(f"stream B finished before the check at every sleep {SWITCH_SLEEPS_MS}")


def test_gpu_stream_switch_with_a_replacement_and_poisoned_reuse_judged(
    device, record_property
) -> None:
    """A workspace allocated on stream A serves, on stream B behind device sleep, a forward and backward of a small
    pack and then of a larger pack that replaces its pools and arena; once B's sleep ends, stream A fills blocks of
    the released sizes allocated after the switch with NaN, over and over, while B's calls run.  Out and the five
    gradients of one workspace per pack on one stream (``one_stream``) and of B's calls (``switched``, every attempt a
    repeat) are judged per pack (case ``p<i>``).  On the first draw an attempt whose poison was queued only after B
    finished is repeated with a longer sleep; the other draws run one attempt at the shortest sleep."""
    H = 4
    judge = NaiveJudge()
    for j in range(DRAWS):
        _stream_switch_draw(judge, j, device, H)
    judge.settle(record_property)


def _stream_switch_draw(judge: NaiveJudge, j: int, device, H: int) -> None:
    """Draw ``j`` of :func:`test_gpu_stream_switch_with_a_replacement_and_poisoned_reuse_judged`."""
    seed = draw_seed(JUDGE_SEED, j)
    one_stream = switched = ops = None
    for ms in SWITCH_SLEEPS_MS if j == 0 else SWITCH_SLEEPS_MS[:1]:
        (sc, so), (bc, bo), ws, sa, sb = _switch_setup(device, H, seed)
        ops = [so, bo]
        refs = []
        for c, o in ((sc, so), (bc, bo)):
            w = _new_ws()
            out, s = _fwd(c, o, w)
            refs.append((out, _bwd(c, o, w, s)))
            del w
        torch.cuda.synchronize(device)
        old = _live_ranges(ws)
        cycles = sleep_cycles(device, ms)
        got = []
        with torch.cuda.stream(sb):
            torch.cuda._sleep(cycles)
            woke = torch.cuda.Event()
            woke.record()
            for c, o in ((sc, so), (bc, bo)):
                out, s = _fwd(c, o, ws)
                got.append((out, _bwd(c, o, ws, s)))
            done = torch.cuda.Event()
            done.record()
        freed = {p: n for p, n in old.items() if p not in _live_ranges(ws)}
        assert len(freed) >= 2, freed
        with torch.cuda.stream(sa):
            held = [
                torch.empty((n // 4,), dtype=torch.float32, device=device)
                for n in freed.values()
            ]
            sa.wait_event(woke)
            for _ in range(64):
                for t in held:
                    t.fill_(float("nan"))
                torch.cuda._sleep(max(cycles // 20000, 1))
        busy = not done.query()  # the poison was queued before B's calls ran
        torch.cuda.synchronize(device)
        print(f"STREAM_SWITCH_POISON draw={j} sleep_ms={ms} busy={busy}")

        def named(res) -> dict:
            out, grads = res
            return host({"out": out, **dict(zip(BWD_GRADS, grads, strict=True))})

        if one_stream is None:
            one_stream, switched = [[], []], [[], []]
        for i in range(2):
            one_stream[i].append(named(refs[i]))
            switched[i].append(named(got[i]))
        del held, got, refs, ws
        if busy or j > 0:
            break
    else:
        pytest.fail(
            f"stream B finished before the poison was queued at every sleep {SWITCH_SLEEPS_MS}"
        )
    arms = [{"one_stream": one_stream[i], "switched": switched[i]} for i in range(2)]
    _judge_packs(judge, j, "", SWITCH_PACKS, ops, arms, device)


def test_gpu_record_stream_only_on_a_stream_switch(
    device, monkeypatch, record_property
) -> None:
    """Census of ``record_stream`` (the first draw): steady rounds of two packs on one stream record nothing; the
    first call on another stream records each live storage of the workspace once (every pool, the arena and the
    scratch table sets among them), and the later calls on that stream record nothing again.  Out and the five
    gradients of the first round (``first``), the steady rounds (``steady``), the first round on the new stream
    (``switch``) and the next one (``after_switch``) are judged per pack on every draw (case ``p<i>``; a new stream
    per draw)."""
    H = 4
    ctxs, ops = _packs(device, H, packs=PACKS[:2])
    ws = _new_ws()
    ws.reserve(ctxs, H)

    def rnd():
        res = []
        for c, o in zip(ctxs, ops, strict=True):
            out, s = _fwd(c, o, ws)
            grads = _bwd(c, o, ws, s)
            res.append({"out": out, **dict(zip(BWD_GRADS, grads, strict=True))})
        return res

    judge = NaiveJudge()
    seen: list[tuple[int, int]] = []
    record = torch.Tensor.record_stream

    def counting(self, stream):
        seen.append((self.untyped_storage().data_ptr(), stream.cuda_stream))
        return record(self, stream)

    for j in range(DRAWS):
        if j:
            ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        first = rnd()
        rnd()
        if j == 0:
            monkeypatch.setattr(torch.Tensor, "record_stream", counting)
        steady_runs = [rnd() for _ in range(2)]
        steady = list(seen)
        s2 = torch.cuda.Stream(device)
        s2.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s2):
            res = rnd()
            switch = list(seen[len(steady) :])
            res2 = rnd()
            after = list(seen[len(steady) + len(switch) :])
        torch.cuda.current_stream(device).wait_stream(s2)
        if j == 0:
            recorded = {p for p, s in switch if s == s2.cuda_stream}
            print(
                f"RECORD_STREAM steady={len(steady)} switch={len(switch)} storages={len(_live_ranges(ws))} "
                f"recorded={len(recorded & set(_live_ranges(ws)))} after={len(after)}"
            )
            assert steady == [], steady
            assert set(_live_ranges(ws)) <= recorded, (
                "a storage of the workspace was not recorded on the new stream"
            )
            live = {t.untyped_storage().data_ptr() for t in ws._live_storages()}
            mine = sorted(p for p, s in switch if p in live and s == s2.cuda_stream)
            assert mine == sorted(live), (mine, sorted(live))
            assert after == [], after
        seen.clear()
        torch.cuda.synchronize(device)
        arms = [
            {
                "first": first[i],
                "steady": [r[i] for r in steady_runs],
                "switch": res[i],
                "after_switch": res2[i],
            }
            for i in range(len(ctxs))
        ]
        _judge_packs(judge, j, "", PACKS[:2], ops, arms, device)
        del first, steady_runs, res, res2, arms
    judge.settle(record_property)


def test_gpu_graph_replay_on_a_reserved_shared_workspace(
    device, record_property
) -> None:
    """``graph="replay"`` with two packs on one reserved workspace, interleaved over four steps: each pack's keys
    capture once and replay (the pointers never move; every draw, the graph cache cleared before it); out and the five
    gradients of the eager per-pack call (``eager``) and of the four graph steps (``graph``, repeats) are judged per
    pack (case ``p<i>``).  After the draws a pack that does not fit the arena replaces it and drops every cached graph
    that pins the old arena or one of its plan views."""
    H = 4
    packs = (PACKS[0], PACKS[2])
    ctxs, ops = _packs(device, H, packs=packs)
    per = [_new_ws() for _ in ctxs]
    cake_kcp.graph_cache_clear()
    ws = _new_ws()
    ws.reserve(ctxs, H)
    ptrs = _pointers(ws)
    judge = NaiveJudge()
    try:
        for j in range(DRAWS):
            if j:
                ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
                cake_kcp.graph_cache_clear()
            eager = []
            for c, o, w in zip(ctxs, ops, per, strict=True):
                out, s = _fwd(c, o, w)
                grads = _bwd(c, o, w, s)
                eager.append(
                    host({"out": out, **dict(zip(BWD_GRADS, grads, strict=True))})
                )
                del out, s, grads
            graph = [[] for _ in ctxs]
            events = []
            for _step in range(4):
                row = []
                for i, (c, o) in enumerate(zip(ctxs, ops, strict=True)):
                    out, s = _fwd(c, o, ws, graph="replay")
                    row.append(cake_kcp.LAST_CALL["fwd"]["graph_event"])
                    grads = _bwd(c, o, ws, s, graph="replay")
                    row.append(cake_kcp.LAST_CALL["bwd"]["graph_event"])
                    graph[i].append(
                        host({"out": out, **dict(zip(BWD_GRADS, grads, strict=True))})
                    )
                    del out, s, grads
                events.append(row)
            if j == 0:
                print(f"GRAPH_EVENTS {events}")
            assert events[-1] == ["replay"] * 4, (j, events)
            assert all(e.disabled is None for e in cake_kcp._GRAPHS.values())
            assert _pointers(ws) == ptrs, j
            arms = [{"eager": eager[i], "graph": graph[i]} for i in range(len(ctxs))]
            _judge_packs(judge, j, "", packs, ops, arms, device)
            del eager, graph, arms
        old = ws.local
        old_views = list(ws._views.values())
        assert old_views, "the reserved packs share no plan view"
        pinned = [
            e
            for e in cake_kcp._GRAPHS.values()
            if any(p is old or p in old_views for p in e.pins)
        ]
        assert pinned
        big, big_ops = _packs(device, H, packs=((4000, 1700),))
        _fwd(big[0], big_ops[0], ws)
        assert ws.local is not old
        for e in cake_kcp._GRAPHS.values():
            assert not any(
                isinstance(p, cake_backend.CakeKDANondetWorkspace)
                and p.arena is old.arena
                for p in e.pins
            )
    finally:
        cake_kcp.graph_cache_clear()
    judge.settle(record_property)


# --------------------------------------------------------------------------- recompute handles on a shared workspace

# checkpointed KDA layers per pack: y1 = KDA(q, k, v, g, beta), y2 = KDA(q, k, y1, g, beta) on the pack's context
HANDLE_LAYERS = 2
# "interleaved": every pack's forward, then the backwards in pack order (A fwd, B fwd, A bwd, B bwd);
# "sequential": pack by pack, forward then backward (A fwd, A bwd, B fwd, B bwd)
HANDLE_ORDERS = ("interleaved", "sequential")
RECOMPUTE_EXCHANGES = (
    "a2a:",
    "ring:",
)  # sequence entries of the state exchange (c1 / c2 or the ring hops)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Byte view: -0.0 vs 0.0 and NaN payloads count as differences."""
    return t.detach().contiguous().view(torch.uint8)


def _same_bits(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(_bits(a), _bits(b))
        )
    return len(a) == len(b) and all(_same_bits(x, y) for x, y in zip(a, b, strict=True))


def _poison_all(ws) -> None:
    """The arena and every pool but ``eye`` filled with all-ones bytes (NaN in BF16 / FP32): a stored tensor that
    lived in the workspace would read NaN afterwards.  ``eye`` is the identity the workspace writes once when it
    sizes that pool, a constant every call reads; the other pools carry no contents across calls."""
    _poison(ws)
    eye = ws._pools.get("eye")
    keep = None if eye is None else eye.untyped_storage().data_ptr()
    for t in ws._pools.values():
        if t.numel() and t.untyped_storage().data_ptr() != keep:
            t.view(torch.uint8).fill_(0xFF)


def _overlaps(t: torch.Tensor, storages) -> bool:
    """Whether ``t``'s storage shares a byte with one of ``storages``."""
    if t.numel() == 0:
        return False
    a0 = t.untyped_storage().data_ptr()
    a1 = a0 + t.untyped_storage().nbytes()
    for s in storages:
        b0 = s.untyped_storage().data_ptr()
        if b0 < a1 and a0 < b0 + s.untyped_storage().nbytes():
            return True
    return False


class _Records:
    """Every forward / backward record (``cake_kcp.LAST_CALL`` as ``_record`` writes it) while active."""

    def __enter__(self):
        self.recs: list[tuple[str, dict]] = []
        self._orig = cake_kcp._record

        def record(kind, call):
            self._orig(kind, call)
            self.recs.append((kind, dict(cake_kcp.LAST_CALL[kind])))

        cake_kcp._record = record
        return self

    def __exit__(self, *exc):
        cake_kcp._record = self._orig
        return False

    def phases(self, n_calls: int, exchange: str) -> list[str]:
        """The violations of the handle phases over ``n_calls`` checkpointed calls: one "capture" (the original
        forward, which exchanges) and one "replay" per call; a replay issues no collective and launches no exchange."""
        bad = []
        fwd = [r for kind, r in self.recs if kind == "fwd"]
        capture = [r for r in fwd if r["recompute_handle"] == "capture"]
        replay = [r for r in fwd if r["recompute_handle"] == "replay"]
        if len(capture) != n_calls or len(replay) != n_calls or len(fwd) != 2 * n_calls:
            bad.append(
                f"phases fwd {len(fwd)} capture {len(capture)} replay {len(replay)} != {n_calls}"
            )
        for r in replay:
            if r["n_collectives"] != 0:
                bad.append(f"replay n_collectives {r['n_collectives']}")
            if any(s.startswith(RECOMPUTE_EXCHANGES) for s in r["sequence"]):
                bad.append("replay launches an exchange")
        if exchange == "a2a" and any(r["n_collectives"] != 2 for r in capture):
            bad.append("an original forward without its c1 + c2")
        return bad


def _handle_pass(ctxs, ops, wss, *, keep=None, order="interleaved", poison=None, **kw):
    """Per pack :data:`HANDLE_LAYERS` KDA layers on its context (layer ``l + 1`` reads layer ``l``'s ``out`` as
    ``v``; leaves ``q, k, v, g, beta``; ``backward(dout)`` from the last layer).  ``keep=None``: direct autograd, no
    checkpoint, no handle (the reference).  Otherwise every layer runs under ``torch.utils.checkpoint`` (non-reentrant)
    with its own :class:`CakeKDANondetKcpRecomputeHandle` and ``checkpoint_keep=keep``.  ``wss[i]``: pack ``i``'s workspace (one
    object: shared).  ``poison(ws)`` before every forward and backward.  Right after each checkpointed forward the
    handle's stored tensors are copied; before every backward and at the end every stored tensor is compared bitwise
    with its copy (a byte copy: a stored tensor in the shared workspace would be overwritten by the other packs' calls
    and poisoned), and each must lie outside the workspace's storages.  Returns per pack ``out`` and the five gradients
    and, with handles, ``layer_outs`` (per layer the original forward's and the recompute's ``out``, for the judge),
    and a log of the violations (a layer whose recompute did not run exactly once among them) and of the stored
    element counts."""
    n = len(ctxs)
    if not isinstance(wss, list):
        wss = [wss] * n
    res: list[dict] = [{} for _ in range(n)]
    handles: list[list] = [[] for _ in range(n)]
    snaps: list[list] = [[] for _ in range(n)]
    seen: list[list] = [[[] for _ in range(HANDLE_LAYERS)] for _ in range(n)]
    log: dict = {
        "changed": [],
        "overlap": [],
        "recompute_out": [],
        "S_in": 0,
        "c1_recv": 0,
    }
    live: dict = {}

    def check(tag):
        for i in range(n):
            for layer, (h, snap) in enumerate(zip(handles[i], snaps[i], strict=True)):
                entry = h.entries.get(cake_kcp.HANDLE_KEY) or {}
                for name, copy in snap.items():
                    t = entry.get(name)
                    if not isinstance(t, torch.Tensor) or not _same_bits(t, copy):
                        log["changed"].append(f"{tag}: pack {i} layer {layer} {name}")

    def forward(i):
        ctx, ws = ctxs[i], wss[i]
        leaves = [
            ops[i][name].detach().clone().requires_grad_(True)
            for name in ("q", "k", "v", "g", "beta")
        ]
        q, k, y, g, b = leaves
        for layer in range(HANDLE_LAYERS):
            if poison is not None:
                poison(ws)
            if keep is None:
                y = cake_kcp.cake_kda_nondet_kcp(
                    q,
                    k,
                    y,
                    g,
                    b,
                    ctx,
                    None,
                    "light",
                    "saved",
                    ws,
                    **kw,
                )
                continue

            def fn(q_, k_, v_, g_, b_, h_, w_, _i=i, _layer=layer):
                out = cake_kcp.cake_kda_nondet_kcp(
                    q_,
                    k_,
                    v_,
                    g_,
                    b_,
                    ctxs[_i],
                    None,
                    "light",
                    "saved",
                    wss[_i],
                    recompute_handle=h_,
                    checkpoint_keep=keep,
                    **kw,
                )
                seen[_i][_layer].append(out.detach().clone())
                return out * w_

            h = cake_kcp.CakeKDANondetKcpRecomputeHandle()
            # ``out * w`` with ``w`` a leaf of ones changes no bit of ``out`` or of its gradient, but the product's
            # backward reads the recompute's ``out``: the recompute runs past the record above (torch stops a
            # recompute once the tensors it needs are back)
            w = torch.ones_like(y).requires_grad_(True)
            y = torch.utils.checkpoint.checkpoint(
                fn, q, k, y, g, b, h, w, use_reentrant=False
            )
            stored = {
                name: t
                for name, t in h.entries[cake_kcp.HANDLE_KEY].items()
                if isinstance(t, torch.Tensor)
            }
            handles[i].append(h)
            snaps[i].append({name: t.clone() for name, t in stored.items()})
            for name, t in stored.items():
                log[name] += t.numel()
                if _overlaps(t, _storages(ws)):
                    log["overlap"].append(f"pack {i} layer {layer} {name}")
        live[i] = (leaves, y)

    def backward(i):
        check(f"before the backward of pack {i}")
        if poison is not None:
            poison(wss[i])
        leaves, y = live.pop(i)
        y.backward(ops[i]["dout"])
        res[i]["out"] = y.detach()
        res[i]["grads"] = tuple(t.grad for t in leaves)

    if order == "interleaved":
        for i in range(n):
            forward(i)
        for i in range(n):
            backward(i)
    else:
        for i in range(n):
            forward(i)
            backward(i)
    check("after the last backward")
    if keep is not None:
        for i in range(n):
            res[i]["layer_outs"] = [list(outs) for outs in seen[i]]
            for layer in range(HANDLE_LAYERS):
                outs = seen[i][layer]
                if len(outs) != 2:
                    log["recompute_out"].append(
                        f"pack {i} layer {layer}: {len(outs)} outs"
                    )
    return res, log


def _handle_violations(log, recs, n_calls, exchange) -> list[str]:
    """The plumbing violations of a handle pass: a stored tensor changed before its backward or placed in the
    workspace, a recompute that did not run once per layer, a phase record off its rule."""
    bad = [f"stored tensor changed: {c}" for c in log["changed"]]
    bad += [f"stored tensor in the workspace: {c}" for c in log["overlap"]]
    bad += [f"recompute did not run once: {c}" for c in log["recompute_out"]]
    return bad + recs.phases(n_calls, exchange)


def handle_arms(r: dict, arm: str) -> dict:
    """The judged arms of one pack's :func:`_handle_pass` result: ``out`` and the five gradients of the leaves
    (``arm``) and, with handles, each layer's original and recomputed ``out`` (``<arm>.layer<l>``, repeats)."""
    arms = {arm: {"out": r["out"], **dict(zip(LEAF_GRADS, r["grads"], strict=True))}}
    for layer, outs in enumerate(r.get("layer_outs", ())):
        arms[f"{arm}.layer{layer}"] = [{f"out_l{layer}": t} for t in outs]
    return arms


@pytest.mark.parametrize("order", HANDLE_ORDERS)
@pytest.mark.parametrize("keep", ("states", "transfer"))
@pytest.mark.parametrize("H", HEADS)
def test_gpu_recompute_handles_on_a_reserved_shared_workspace(
    device, H, keep, order, record_property
) -> None:
    """Four packs on ONE reserved workspace, two checkpointed KDA layers per pack, one recompute handle per pack and
    layer (``checkpoint_keep`` ``"states"`` / ``"transfer"``), in the interleaved (A fwd, B fwd, A bwd, B bwd) and the
    sequential order, the arena and the pools poisoned before every forward and backward: each recompute binds its
    entry ("replay": no collective, no exchange launch) and runs once per layer; no stored tensor changes before its
    backward and none lies in the workspace; no pointer moves.  Out and the five gradients of the two-layer stack on
    one workspace per pack without handles (``direct``) and with the handles (``handle``), and each layer's original
    and recomputed ``out`` (``handle.layer<l>``), are judged per pack against the FP64 reference of the stack beside
    its naive tier (:func:`stack_refs`) and FLA's stack (:func:`stack_fla`; case ``p<i>``).  One rank: no fragment
    crosses a rank, so the stored tensors are empty here; the W = 4 rows at the end of this file store and check real
    ones."""
    ctxs, _ = _packs(device, H)
    ws = _new_ws()
    ws.reserve(ctxs, H)
    ptrs = _pointers(ws)
    judge = NaiveJudge()
    for j in range(DRAWS):
        ops = _draw_ops(ctxs, H, device, draw_seed(JUDGE_SEED, j))
        ref, _ = _handle_pass(ctxs, ops, [_new_ws() for _ in ctxs])
        with _Records() as recs:
            got, log = _handle_pass(
                ctxs, ops, ws, keep=keep, order=order, poison=_poison_all
            )
        bad = _handle_violations(log, recs, len(ctxs) * HANDLE_LAYERS, "a2a")
        assert not bad, (j, bad)
        assert _pointers(ws) == ptrs, j
        assert all(c.n_in == 0 for c in ctxs) and log["S_in"] == log["c1_recv"] == 0
        for i, lengths in enumerate(PACKS):
            inputs = pack_inputs(_part(lengths), _cu(lengths), ops[i], device)
            r, t = stack_refs(judge, inputs, HANDLE_LAYERS)
            arms = {**handle_arms(ref[i], "direct"), **handle_arms(got[i], "handle")}
            judge.add_refs(j, f"p{i}", arms, r, t, stack_fla(judge, inputs, HANDLE_LAYERS))
        del ref, got, log
    judge.settle(record_property)


# --------------------------------------------------------------------------- multi-GPU

# (packs, H, W, layout, exchange): four prepared packs per rank on one reserved workspace per rank; a2a and ring
# (every document of the ring rows spans at most three ranks), P = 1 and P > 1 (H = 4), forward-only and training
DIST_ROWS = [
    (
        ((3000,), (2048, 5, 3000, 1), (700, 1, 1300, 999), (1200, 900, 900)),
        4,
        4,
        "zigzag_per_document",
        "a2a",
    ),
    (
        ((100, 640, 77, 1000, 129), (2000, 1000), (3000,), (500, 2500)),
        64,
        4,
        "contiguous",
        "a2a",
    ),
    (
        ((100, 640, 77, 1000, 129), (2000, 1000), (1500, 1500), (1000, 2000)),
        4,
        4,
        "contiguous",
        "ring",
    ),
    (
        ((5000, 1003), (700, 1, 1300, 999, 3000), (3000,), (2500, 2500, 1000)),
        4,
        8,
        "zigzag_per_document",
        "a2a",
    ),
    (
        ((5000, 1003), (700, 1, 1300, 999, 3000), (6000,), (2500, 2500, 1000)),
        64,
        8,
        "zigzag",
        "a2a",
    ),
    (
        (
            (1000, 1000, 1000, 1000),
            (500,) * 8,
            (800, 700, 900, 600, 1000),
            (300, 900, 1000, 800, 1000),
        ),
        4,
        8,
        "contiguous",
        "ring",
    ),
]


def _dist_row_id(row) -> str:
    packs, H, W, layout, exchange = row
    return f"p{len(packs)}-h{H}-w{W}-{layout}-{exchange}"


def _rank_packs(rank: int, W: int, row, rdzv: str | None):
    """This rank's device (the process group initialized for ``W > 1``), the row's packs prepared on it and each pack's
    partition and ``cu`` (:func:`_rank_ops` draws the operands)."""
    packs, _H, _w, layout, exchange = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    group = dist.group.WORLD if W > 1 else None
    parts = [(_part(lengths, W, layout), _cu(lengths)) for lengths in packs]
    ctxs = [
        kp.cake_kda_nondet_train_kcp_build_context(part, group, rank, device)
        for part, _cu_ in parts
    ]
    if exchange == "ring":
        assert all(c.ring_eligible for c in ctxs), (
            "a ring row holds a document over more than three ranks"
        )
    return device, ctxs, parts


def _rank_ops(
    packs, parts, rank: int, H: int, device, seed: int
) -> tuple[list[dict], list[dict] | None]:
    """The operands of one draw of a row's packs on this rank: every rank draws each pack whole (pack ``i`` at
    ``seed + i``; ``make_inputs``' host generator, so the same tensors on every rank) and keeps its local buffer;
    rank 0 also keeps the whole packs on the device (the judge's operands)."""
    ops, globs = [], []
    for i, (lengths, (part, cu)) in enumerate(zip(packs, parts, strict=True)):
        glob = make_inputs(lengths, H, "cpu", seed=seed + i)
        ops.append(
            {n: t.to(device) for n, t in local_ops(part, rank, cu, glob).items()}
        )
        if rank == 0:
            globs.append({n: t.to(device) for n, t in glob.items()})
    return ops, (globs if rank == 0 else None)


def _judge_rank_arms(
    judge: NaiveJudge | None,
    j: int,
    case: str,
    arms: dict,
    part,
    cu: list[int],
    glob: dict | None,
    W: int,
    rank: int,
) -> None:
    """Draw ``j`` of one pack of a multi-rank row: every rank's local results of every arm (``{arm: {output:
    tensor}}``) gathered to rank 0, put in the global token order and judged there (``case``) on the pack's whole
    operands ``glob``."""
    mine = {name: host(res) for name, res in arms.items()}
    every = gather_to_rank0(mine, W, rank)
    if judge is not None:
        judge.add(
            j,
            case,
            {name: global_results(part, cu, [e[name] for e in every]) for name in mine},
            glob,
        )


def _run_dist_rank(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None
) -> dict:
    """One rank, over the judge's draws: the four packs on one workspace per pack and on one reserved shared workspace
    (the interleaved schedule twice, the arena NaN-filled before every forward of the first); the shared workspace's
    bytes against ``cake_kda_nondet_train_kcp_workspace_bytes(ctxs)`` and the per-pack sum, the reserve's allocation,
    pointer stability over every schedule.  Out and the five gradients of every arm (``per``, ``shared_poisoned``,
    ``shared``) of every pack are gathered to rank 0 and judged there (``report["naive"]``).  The flags are gathered
    to every rank."""
    packs, H, _w, layout, exchange = row
    device, ctxs, parts = _rank_packs(rank, W, row, rdzv)
    kw = {"exchange": exchange}
    ring = exchange == "ring"
    report: dict = {"rank": rank, "row": _dist_row_id(row), "pass": True, "fail": []}
    judge = NaiveJudge() if rank == 0 else None
    ws = ptrs = poison = None
    for j in range(DRAWS):
        ops, globs = _rank_ops(packs, parts, rank, H, device, draw_seed(JUDGE_SEED, j))
        per = [_new_ws() for _ in ctxs]
        ref = _schedule(ctxs, ops, per, **kw)
        if ws is None:
            report["per_pack_bytes"] = [w.bytes for w in per]
            report["P"] = [
                cake_kcp.CakeKDANondetKcpWorkspace(local=False).tables(c, H).P
                for c in ctxs
            ]
        del per
        if ws is None:
            gc.collect()
            torch.cuda.synchronize(device)
            base = torch.cuda.memory_allocated(device)
            ws = _new_ws()
            ws.reserve(ctxs, H, ring=ring)
            torch.cuda.synchronize(device)
            report["reserve_delta"] = torch.cuda.memory_allocated(device) - base
            report["rounded"] = sum(
                -(-t.untyped_storage().nbytes() // 512) * 512 for t in _storages(ws)
            )
            report["shared_bytes"] = ws.bytes
            # the ring rows reserve the hop pools too: each at its largest need over the packs
            report["pred_bytes"] = kp.cake_kda_nondet_train_kcp_workspace_bytes(
                ctxs, H=H
            ) + (
                sum(
                    max(kp.kcp_arena_layout(c, H=H)[f"pool.{name}"] for c in ctxs)
                    for name in kp.LAZY_POOLS
                )
                if ring
                else 0
            )
            report["storages"] = len(_storages(ws))
            ptrs = _pointers(ws)
            poison = _poison if ws.local is not None else None
        got = []
        for rnd in range(2):
            got.append(
                _schedule(ctxs, ops, ws, poison=poison if rnd == 0 else None, **kw)
            )
            if _pointers(ws) != ptrs:
                report["fail"].append(f"pointers moved d{j} r{rnd}")
        for i, (part, cu) in enumerate(parts):
            arms = {
                **schedule_arms(ref[i], "per"),
                **schedule_arms(got[0][i], "shared_poisoned"),
                **schedule_arms(got[1][i], "shared"),
            }
            glob = globs[i] if globs is not None else None
            _judge_rank_arms(judge, j, f"p{i}", arms, part, cu, glob, W, rank)
        del ref, got, ops, globs
    if report["shared_bytes"] != report["pred_bytes"]:
        report["fail"].append(
            "bytes != cake_kda_nondet_train_kcp_workspace_bytes(ctxs)"
        )
    # the caching allocator may hand out a cached block up to 1 MiB larger than a request without splitting it
    if not (
        report["rounded"]
        <= report["reserve_delta"]
        < report["rounded"] + report["storages"] * (1 << 20)
    ):
        report["fail"].append("reserve delta outside the rounded storages")
    if sum(1 for b in report["per_pack_bytes"] if b) > 1 and not (
        report["shared_bytes"] < sum(report["per_pack_bytes"])
    ):
        report["fail"].append("shared bytes not below the per-pack sum")
    return _finish_rank_report(
        report, judge, W, rank, result_dir, "shared-" + _dist_row_id(row)
    )


def _finish_rank_report(
    report: dict, judge, W: int, rank: int, result_dir: str | None, name: str
) -> dict:
    """The row's flags gathered to every rank (``report["pass"]``: every rank's), rank 0's judge result in
    ``report["naive"]`` and, on rank 0, the report written to ``<result_dir>/<name>.json``."""
    report["pass"] = not report["fail"]
    if W > 1:
        verdicts = [None] * W
        dist.all_gather_object(
            verdicts, {k: report[k] for k in report if k not in ("row", "naive")}
        )
        report["ranks"] = verdicts
        report["pass"] = all(v["pass"] for v in verdicts) and not report["fail"]
        dist.barrier()
    if judge is not None:
        report["naive"] = judge.result()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, name + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _bound_rank_threads(W: int) -> None:
    """Give a spawned rank its share of the host's CPUs: each of the ``W`` ranks would
    otherwise start as many threads as the host has, and rank 0's FP64 reference then
    ran past the process group's ten-minute collective timeout while the other ranks
    waited (813 s on a four-rank row that takes 22 s with the share)."""
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // W))


def _spawn_dist_entry(
    rank: int, W: int, row, rdzv: str | None, result_dir: str, kind: str = "shared"
) -> None:
    try:
        _bound_rank_threads(W)
        report = RANK_RUNNERS[kind](rank, W, row, rdzv, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("row", DIST_ROWS, ids=_dist_row_id)
def test_multi_gpu_shared_workspace_rows_judged_per_pack(row, record_property) -> None:
    """The rows of :func:`_run_dist_rank`: one process per GPU (``spawn``, rows up to the visible GPU count) or one
    process per Slurm task (a step with ``SLURM_NTASKS`` > 1 runs the rows of its own world size and skips the
    others); rank 0's verdicts recorded and settled here (case ``p<i>``)."""
    _launch(row, "shared", record_property)


def _launch(row, kind: str, record_property) -> None:
    """Run ``RANK_RUNNERS[kind]`` for ``row``: in this process as one Slurm task (a step with ``SLURM_NTASKS`` > 1
    runs the rows of its own world size and skips the others) or one spawned process per GPU (rows up to the
    visible GPU count); assert the gathered flags, then record and settle rank 0's verdicts."""
    _packs_, _H, W, _layout, _exchange = row
    ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"this Slurm step runs the W = {ntasks} rows")
        report = RANK_RUNNERS[kind](
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )
        assert report["pass"], report
        if "naive" in report:  # rank 0
            settle_naive(record_property, report["naive"])
        return
    _device_or_skip()
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    with tempfile.TemporaryDirectory(prefix="cake_kcp_shared_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_dist_entry,
            args=(W, row, _rendezvous(tmp), result_dir, kind),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, RESULT_NAMES[kind](row))) as fh:
            report = json.load(fh)
    assert report["pass"], {k: v for k, v in report.items() if k != "naive"}
    settle_naive(record_property, report["naive"])


# (packs, H, W, layout, exchange): the W = 4 rows above with recompute handles -- a2a P > 1 (H = 4, both tiers),
# a2a P = 1 (H = 64, both tiers), ring (H = 4, "states": the ring receives no c1 tensor)
HANDLE_DIST_ROWS = [row for row in DIST_ROWS if row[2] == 4]


def _handle_row_id(row) -> str:
    return "handles-" + _dist_row_id(row)


def _run_handle_rank(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None
) -> dict:
    """One rank, over the judge's draws: the row's packs on one workspace per pack without handles (``direct``), then
    on ONE reserved shared workspace with two checkpointed layers per pack and one recompute handle per pack and
    layer, for every tier (``"transfer"`` with a2a only) and both orders (:func:`_handle_pass`; the arena and the pools
    poisoned before every forward and backward): no stored ``S_in`` / c1 receive tensor changed before its backward or
    placed in the workspace, every recompute exchange-free and run once per layer, no pointer moved.  Out and the five
    gradients of every arm (``direct``, ``<tier>.<order>``) and each layer's original and recomputed ``out`` are
    gathered to rank 0 and judged there against the FP64 reference of the two-layer stack (:func:`stack_refs`, beside
    FLA's stack, :func:`stack_fla`).  The stored element counts are gathered: the row must store non-empty tensors on
    some rank (of each tier's members).  The flags are gathered to every rank."""
    packs, H, _w, _layout, exchange = row
    device, ctxs, parts = _rank_packs(rank, W, row, rdzv)
    kw = {"exchange": exchange}
    report: dict = {"rank": rank, "row": _handle_row_id(row), "pass": True, "fail": []}
    judge = NaiveJudge() if rank == 0 else None
    ws = _new_ws()
    ws.reserve(ctxs, H, ring=exchange == "ring")
    ptrs = _pointers(ws)
    keeps = ("states",) if exchange == "ring" else ("states", "transfer")
    stored: dict = {}
    for j in range(DRAWS):
        ops, globs = _rank_ops(packs, parts, rank, H, device, draw_seed(JUDGE_SEED, j))
        ref, _ = _handle_pass(ctxs, ops, [_new_ws() for _ in ctxs], **kw)
        arms = [handle_arms(ref[i], "direct") for i in range(len(ctxs))]
        del ref
        for keep in keeps:
            for order in HANDLE_ORDERS:
                with _Records() as recs:
                    got, log = _handle_pass(
                        ctxs, ops, ws, keep=keep, order=order, poison=_poison_all, **kw
                    )
                bad = _handle_violations(log, recs, len(ctxs) * HANDLE_LAYERS, exchange)
                report["fail"] += [f"d{j} {keep} {order}: {b}" for b in bad]
                if _pointers(ws) != ptrs:
                    report["fail"].append(f"d{j} {keep} {order}: pointers moved")
                if j == 0:
                    stored[f"{keep}/{order}"] = {
                        "S_in": log["S_in"],
                        "c1_recv": log["c1_recv"],
                    }
                for i in range(len(ctxs)):
                    arms[i].update(handle_arms(got[i], f"{keep}.{order}"))
                del got, log
        for i, (part, cu) in enumerate(parts):
            mine = {name: host(res) for name, res in _flat_repeats(arms[i]).items()}
            every = gather_to_rank0(mine, W, rank)
            if judge is not None:
                glob = {n: globs[i][n] for n in (*TOKEN_OPERANDS, "cu_seqlens")}
                r, t = stack_refs(judge, glob, HANDLE_LAYERS)
                judge.add_refs(
                    j,
                    f"p{i}",
                    _unflat_repeats(
                        {
                            name: global_results(part, cu, [e[name] for e in every])
                            for name in mine
                        }
                    ),
                    r,
                    t,
                    stack_fla(judge, glob, HANDLE_LAYERS),
                )
        del arms, ops, globs
    report["stored"] = stored
    if W > 1:
        every_stored = [None] * W
        dist.all_gather_object(every_stored, stored)
        for key in stored:
            members = ("S_in", "c1_recv") if key.startswith("transfer") else ("S_in",)
            for m in members:
                if not any(v[key][m] for v in every_stored):
                    report["fail"].append(f"{key}: no rank stored a non-empty {m}")
    return _finish_rank_report(
        report, judge, W, rank, result_dir, "shared-" + _handle_row_id(row)
    )


def _flat_repeats(arms: dict) -> dict:
    """``{arm: result or [result, ...]}`` as ``{"<arm>#<k>": result}`` (one name per repeat; ``#0`` alone for one
    result), for a gather of plain dicts."""
    out = {}
    for name, res in arms.items():
        for k, r in enumerate(res if isinstance(res, list) else [res]):
            out[f"{name}#{k}"] = r
    return out


def _unflat_repeats(flat: dict) -> dict:
    """The inverse of :func:`_flat_repeats` (every arm a list of its repeats)."""
    out: dict = {}
    for key, res in flat.items():
        out.setdefault(key.rsplit("#", 1)[0], []).append(res)
    return out


RANK_RUNNERS = {"shared": _run_dist_rank, "handles": _run_handle_rank}
RESULT_NAMES = {
    "shared": lambda row: "shared-" + _dist_row_id(row) + ".json",
    "handles": lambda row: "shared-" + _handle_row_id(row) + ".json",
}


@pytest.mark.parametrize("row", HANDLE_DIST_ROWS, ids=_dist_row_id)
def test_multi_gpu_recompute_handles_on_a_shared_workspace(
    row, record_property
) -> None:
    """The rows of :func:`_run_handle_rank`, launched like the shared-workspace rows (``spawn`` or one process per
    Slurm task); rank 0's verdicts recorded and settled here (case ``p<i>``)."""
    _launch(row, "handles", record_property)
