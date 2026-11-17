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

The accuracy judge of the family's DSA sparse-attention training op (64 query heads, top-k index rows): the FP64
oracle, the naive BF16 tier and the contract of one call, for the judge of ``tests/cake_naive.py``.

A :class:`Problem` holds one call's own inputs: BF16 ``q_latent [T, 64, 512]`` and ``kv_latent [S, 512]``, in the
RoPE form also ``q_rope [T, 64, 64]`` and ``k_rope [S, 64]`` (the NoPE form has none: the latent is key and value),
the int32 global key rows ``indices [T, K]``, the optional ``topk_length [T]``, ``dout [T, 64, 512]`` (any strided
view) and the softmax scale.  Slot ``j`` of row ``t`` is valid when ``0 <= indices[t, j] < S`` and, with
``topk_length``, ``j < topk_length[t]``.

:func:`outputs` evaluates the forward and the backward of the problem with every op in one dtype: float64 is the
oracle, bfloat16 the naive tier (torch's own ops on BF16 tensors, each rounding its result to BF16 after torch's
internal accumulation).  Per row, with ``q`` / ``k`` the latent and rope parts side by side (the latent alone in the
NoPE form) and ``v`` the latent: the gathered keys ``k_j = k[indices[t, j]]``, the scores ``s = q k_j^T * scale``
over the valid slots, ``lse = logsumexp(s)``, ``P = softmax(s)``, ``out = P v``, ``dP = dout v^T``, ``dS = P (dP -
rowsum(dout * out)) * scale``, ``dq = dS k``, and per slot the key gradient ``dS^T q`` plus the value gradient
``P^T dout`` (each summed over the 64 heads), added into key row ``indices[t, j]`` with ``index_add_``.  The RoPE
form returns ``dq_latent`` / ``dq_rope`` and ``dkv_latent`` / ``dk_rope`` (latent and rope columns of the key
gradient, the value gradient in the latent), the NoPE form ``dq`` and the combined ``dkv``.  A row without a valid
slot has ``out = 0``, ``lse = -inf`` and ``dq = 0``; a key row no valid slot names has zero key gradients.  The rows
go in blocks of at most :data:`BLOCK_SLOTS` (row, slot) pairs: each row's quantities stay inside its block and each
slot's key gradient is its own ``index_add_`` term, so the blocks bound the memory and do not change the values.

:func:`judged` holds the outputs of one or several calls on a problem to their contract (:func:`assert_contract`:
shapes, dtypes, finite values, the exact zeros) and pairs each with the oracle and the tier of the same draw.  The
forward's ``o_lo`` (the BF16 residual of the FP32 output) is judged as ``o_sum = out + o_lo`` in FP32.  An
accumulating call (``dkv_acc``) is judged by its increment: the buffer after the call minus the buffer before it,
exact in float64, against the key gradients added through the destination map (the error of that increment is the
error of the buffer against the oracle's ``before + gradient``).  :func:`judge_draws` runs the template's
:func:`judge_case` over the draws and records every verdict.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Optional

import torch

from tests.cake_naive import DRAWS, TIER_DTYPES, TIERS, judge_case

NUM_HEADS = 64
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE
ROPE_SCALE = D_QK**-0.5
NOPE_SCALE = 1.0 / 16
(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
#: (row, slot) pairs of one block of :func:`outputs`.
BLOCK_SLOTS = 1 << 16
_BITS = {2: torch.int16, 4: torch.int32, 8: torch.int64}

_UNSET = object()


@dataclass
class Problem:
    """One call's inputs (see the module docstring); ``q_rope is None`` is the NoPE form."""

    q_latent: torch.Tensor
    kv_latent: torch.Tensor
    indices: torch.Tensor
    q_rope: Optional[torch.Tensor] = None
    k_rope: Optional[torch.Tensor] = None
    topk_length: Optional[torch.Tensor] = None
    dout: Optional[torch.Tensor] = None
    scale: Optional[float] = None

    @property
    def nope(self) -> bool:
        return self.q_rope is None

    @property
    def softmax_scale(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        return NOPE_SCALE if self.nope else ROPE_SCALE

    @property
    def total_q(self) -> int:
        return int(self.q_latent.shape[0])

    @property
    def total_k(self) -> int:
        return int(self.kv_latent.shape[0])

    def valid(self) -> torch.Tensor:
        """``[T, K]`` bool: the valid slots."""
        ix = self.indices.long()
        valid = (ix >= 0) & (ix < self.total_k)
        if self.topk_length is not None:
            slot = torch.arange(ix.shape[1], device=ix.device)
            valid &= slot[None, :] < self.topk_length.to(ix.device).long()[:, None]
        return valid

    def with_(self, **changes) -> "Problem":
        return replace(self, **changes)


def rope_problem(inp, *, indices=None, topk_length=None, dout=_UNSET, scale=None) -> Problem:
    """The :class:`Problem` of a RoPE call on ``inp`` (the reference helper's ``Inputs``): its global index rows
    unless ``indices`` is given, the ``topk_length`` the call passed (none by default) and ``inp.dout``."""
    return Problem(
        q_latent=inp.q_latent,
        kv_latent=inp.kv_latent,
        indices=inp.idx_global if indices is None else indices,
        q_rope=inp.q_rope,
        k_rope=inp.k_rope,
        topk_length=topk_length,
        dout=inp.dout if dout is _UNSET else dout,
        scale=scale,
    )


def nope_problem(inp, *, indices=None, topk_length=None, dout=_UNSET, scale=None) -> Problem:
    """The NoPE :class:`Problem` of a call on ``inp``: a ``NopeInputs`` (``q``, ``kv``) or an ``Inputs``
    (``q_latent``, ``kv_latent``; its rope parts unused)."""
    q = inp.q if hasattr(inp, "q") else inp.q_latent
    kv = inp.kv if hasattr(inp, "kv") else inp.kv_latent
    return Problem(
        q_latent=q,
        kv_latent=kv,
        indices=inp.idx_global if indices is None else indices,
        topk_length=topk_length,
        dout=inp.dout if dout is _UNSET else dout,
        scale=scale,
    )


def _bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` batched; a contraction of length 1 as the broadcast product it equals (torch may hand a one-long
    contraction to a code generator that needs a ptxas which knows the GPU)."""
    return a * b if a.shape[-1] == 1 else torch.bmm(a, b)


@torch.no_grad()
def outputs(p: Problem, dtype: torch.dtype, *, backward: bool = True) -> dict[str, torch.Tensor]:
    """The forward (``out``, ``lse``) and, with ``backward`` and ``p.dout``, the gradients of ``p`` with every op in
    ``dtype`` on the inputs' device (see the module docstring)."""
    device = p.q_latent.device
    total_q, total_k = p.total_q, p.total_k
    width = int(p.indices.shape[1]) if p.indices.dim() == 2 else 0
    on = dict(device=device, dtype=dtype)
    scale = p.softmax_scale
    kv = p.kv_latent.reshape(total_k, D_LATENT).to(dtype)
    k_all = kv if p.nope else torch.cat([kv, p.k_rope.reshape(total_k, D_ROPE).to(dtype)], -1)
    d_k = int(k_all.shape[-1])
    want_grad = backward and p.dout is not None
    out = torch.zeros(total_q, NUM_HEADS, D_LATENT, **on)
    lse = torch.full((total_q, NUM_HEADS), -math.inf, **on)
    if want_grad:
        dq = torch.zeros(total_q, NUM_HEADS, d_k, **on)
        dk = torch.zeros(total_k, d_k, **on)
    if total_q and width and total_k:
        valid_all = p.valid()
        rows_per_block = max(1, BLOCK_SLOTS // width)
        for r0 in range(0, total_q, rows_per_block):
            r1 = min(total_q, r0 + rows_per_block)
            valid = valid_all[r0:r1]
            ix = p.indices[r0:r1].long().clamp(0, total_k - 1)
            q = p.q_latent[r0:r1].to(dtype)
            if not p.nope:
                q = torch.cat([q, p.q_rope[r0:r1].to(dtype)], -1)  # [rows, 64, 576]
            kg = k_all[ix]  # [rows, K, d_k]
            vg = kg[..., :D_LATENT]
            s = _bmm(q, kg.transpose(1, 2)) * scale  # [rows, 64, K]
            s = s.masked_fill(~valid[:, None, :], -math.inf)
            row_lse = torch.logsumexp(s, -1)
            prob = torch.softmax(s, -1).nan_to_num(0.0)  # a row without a valid slot: NaN -> 0
            o = _bmm(prob, vg)
            out[r0:r1], lse[r0:r1] = o, row_lse
            if not want_grad:
                continue
            g = p.dout[r0:r1].to(dtype)
            dp = _bmm(g, vg.transpose(1, 2))
            ds = prob * (dp - (g * o).sum(-1, keepdim=True)) * scale
            dq[r0:r1] = _bmm(ds, kg)
            dk_slot = _bmm(ds.transpose(1, 2), q)  # [rows, K, d_k]
            dv_slot = _bmm(prob.transpose(1, 2), g)  # [rows, K, 512]
            dk_slot = torch.cat([dk_slot[..., :D_LATENT] + dv_slot, dk_slot[..., D_LATENT:]], -1)
            dk.index_add_(0, ix[valid], dk_slot[valid])
    found = {"out": out, "lse": lse}
    if want_grad:
        if p.nope:
            found.update(dq=dq, dkv=dk)
        else:
            found.update(
                dq_latent=dq[..., :D_LATENT],
                dq_rope=dq[..., D_LATENT:],
                dkv_latent=dk[:, :D_LATENT],
                dk_rope=dk[:, D_LATENT:],
            )
    return found


FORWARD = ("out", "lse", "o_lo")
#: Every output name a call may report, by form (``o_lo`` is judged as ``o_sum``; ``dkv_acc`` by its increment).
OUTPUTS_ROPE = FORWARD + ("dq_latent", "dq_rope", "dkv_latent", "dk_rope", "dkv_acc")
OUTPUTS_NOPE = FORWARD + ("dq", "dkv", "dkv_acc")
_KEY_GRADS = ("dkv_latent", "dk_rope", "dkv")
_DTYPES = {
    "out": (torch.bfloat16,),
    "lse": (torch.float32,),
    "o_lo": (torch.bfloat16,),
    "dq_latent": (torch.bfloat16,),
    "dq_rope": (torch.bfloat16,),
    "dq": (torch.bfloat16,),
    "dkv_latent": (torch.bfloat16, torch.float32),
    "dk_rope": (torch.bfloat16, torch.float32),
    "dkv": (torch.float32, torch.bfloat16),
}


@dataclass
class Acc:
    """An accumulating call's buffer: ``before`` (a copy taken before the call), ``after`` (the buffer after it) and
    the optional int32 ``dst_map [S]`` (key row -> destination row; identity when None)."""

    before: torch.Tensor
    after: torch.Tensor
    dst_map: Optional[torch.Tensor] = None


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().view(_BITS[t.element_size()])


def _zero(t: torch.Tensor) -> bool:
    return bool((t == 0).all()) if t.numel() else True


def _acc_parts(p: Problem, acc: Acc) -> tuple[int, torch.Tensor]:
    """The gradient width of the buffer and the mask of its destination rows that some key row maps to."""
    width = D_LATENT if p.nope else D_QK
    rows = int(acc.after.shape[0])
    touched = torch.zeros(rows, dtype=torch.bool, device=acc.after.device)
    if acc.dst_map is None:
        touched[: min(rows, p.total_k)] = True
    else:
        touched[acc.dst_map.long()] = True
    return width, touched


def assert_contract(label: str, p: Problem, got: Mapping) -> None:
    """The contract of one call's outputs ``got`` on ``p``: each has its shape and dtype and is finite (``lse`` is
    exactly ``-inf`` on the rows without a valid slot); ``out``, ``o_lo`` and ``dq*`` are exactly 0 on those rows;
    the fresh key gradients are exactly 0 on the key rows no valid slot names; an accumulating call keeps the bits
    of every destination row no key row maps to and of every column past the gradient width."""
    names = OUTPUTS_NOPE if p.nope else OUTPUTS_ROPE
    unknown = set(got) - set(names)
    assert not unknown, f"{label}: outputs {sorted(unknown)} are not outputs of the {'NoPE' if p.nope else 'RoPE'} form"
    valid = p.valid()
    no_slot = ~valid.any(-1)
    referenced = torch.zeros(p.total_k, dtype=torch.bool, device=valid.device)
    referenced[p.indices.long()[valid]] = True
    t_q, t_k = p.total_q, p.total_k
    shapes = {
        "out": (t_q, NUM_HEADS, D_LATENT),
        "lse": (t_q, NUM_HEADS),
        "o_lo": (t_q, NUM_HEADS, D_LATENT),
        "dq_latent": (t_q, NUM_HEADS, D_LATENT),
        "dq_rope": (t_q, NUM_HEADS, D_ROPE),
        "dq": (t_q, NUM_HEADS, D_LATENT),
        "dkv_latent": (t_k, D_LATENT),
        "dk_rope": (t_k, D_ROPE),
        "dkv": (t_k, D_LATENT),
    }
    for name, value in got.items():
        if name == "dkv_acc":
            acc = value
            assert isinstance(acc, Acc), f"{label}: dkv_acc is reported as an Acc"
            assert acc.after.dtype == torch.float32 and acc.before.shape == acc.after.shape, (
                f"{label}: dkv_acc {acc.after.dtype} {tuple(acc.after.shape)} / before {tuple(acc.before.shape)}"
            )
            width, touched = _acc_parts(p, acc)
            assert bool(torch.isfinite(acc.after).all()), f"{label}: non-finite dkv_acc"
            assert torch.equal(_bits(acc.after[:, width:]), _bits(acc.before[:, width:])), (
                f"{label}: dkv_acc changed a column past {width}"
            )
            assert torch.equal(_bits(acc.after[~touched]), _bits(acc.before[~touched])), (
                f"{label}: dkv_acc changed a destination row no key row maps to"
            )
            continue
        want = shapes[name]
        shape = tuple(value.shape)
        if name in _KEY_GRADS and shape != want and value.numel() == math.prod(want):
            shape = (shape[0], value.numel() // max(1, shape[0]))  # the one-head [S, 1, 512] leaf
        assert shape == want, f"{label}: {name} has shape {tuple(value.shape)}, expected {want}"
        assert value.dtype in _DTYPES[name], f"{label}: {name} is {value.dtype}, expected one of {_DTYPES[name]}"
        if name == "lse":
            assert bool((value[no_slot] == -math.inf).all()), f"{label}: lse of a row without a valid slot is not -inf"
            assert bool(torch.isfinite(value[~no_slot]).all()), f"{label}: non-finite lse"
            continue
        assert bool(torch.isfinite(value).all()), f"{label}: non-finite {name}"
        if name in _KEY_GRADS:
            rows = value.reshape(t_k, -1)[~referenced]
            assert _zero(rows), f"{label}: {name} of a key row no valid slot names is not 0"
        else:
            assert _zero(value[no_slot]), f"{label}: {name} of a row without a valid slot is not 0"


def _mapped(grad: torch.Tensor, acc: Acc, width: int, dtype: torch.dtype) -> torch.Tensor:
    """``grad [S, width]`` added into zero destination rows through the map, in ``dtype``."""
    rows = int(acc.after.shape[0])
    dst = torch.zeros(rows, width, device=grad.device, dtype=dtype)
    if acc.dst_map is None:
        dst[: grad.shape[0]] = grad.to(dtype)
        return dst
    return dst.index_add_(0, acc.dst_map.long(), grad.to(dtype))


def judged(
    p: Problem, runs: Mapping[str, Mapping], *, prefix: str = ""
) -> tuple[dict, dict, dict]:
    """``(outputs, oracle, naive)`` of one draw for :func:`judge_draws`: the calls ``runs`` (label -> outputs by
    name, see :data:`OUTPUTS_ROPE` / :data:`OUTPUTS_NOPE`) on the problem ``p``, each held to
    :func:`assert_contract`, beside the FP64 oracle and the naive tier of ``p``.  With several calls each output is
    named ``"<label> <output>"``; ``prefix`` goes in front of every name (several problems in one draw)."""
    backward = any(name not in FORWARD for got in runs.values() for name in got)
    ref = outputs(p, torch.float64, backward=backward)
    low = outputs(p, NAIVE_DTYPE, backward=backward)
    found, oracle, naive = {}, {}, {}
    for label, got in runs.items():
        assert_contract(label or "call", p, got)
        head = prefix + (f"{label} " if len(runs) > 1 else "")
        for name, value in got.items():
            if name == "o_lo":
                assert "out" in got, f"{label}: o_lo is judged with its out (o_sum = out + o_lo)"
                found[head + "o_sum"] = got["out"].float() + value.float()
                oracle[head + "o_sum"], naive[head + "o_sum"] = ref["out"], low["out"]
                continue
            if name == "dkv_acc":
                width, _ = _acc_parts(p, value)
                if p.nope:
                    want, tier = ref["dkv"], low["dkv"]
                else:
                    want = torch.cat([ref["dkv_latent"], ref["dk_rope"]], -1)
                    tier = torch.cat([low["dkv_latent"], low["dk_rope"]], -1)
                found[head + "dkv_acc"] = value.after[:, :width].double() - value.before[:, :width].double()
                oracle[head + "dkv_acc"] = _mapped(want, value, width, torch.float64)
                naive[head + "dkv_acc"] = _mapped(tier, value, width, NAIVE_DTYPE)
                continue
            want, tier = ref[name], low[name]
            if want.shape != value.shape:
                want, tier = want.reshape(value.shape), tier.reshape(value.shape)
            found[head + name], oracle[head + name], naive[head + name] = value, want, tier
    return found, oracle, naive


def merge(*draws: tuple[dict, dict, dict]) -> tuple[dict, dict, dict]:
    """One draw's ``(outputs, oracle, naive)`` from several :func:`judged` results (distinct names)."""
    found, oracle, naive = {}, {}, {}
    for f, o, n in draws:
        assert not set(f) & set(found), f"outputs judged twice in one draw: {sorted(set(f) & set(found))}"
        found.update(f)
        oracle.update(o)
        naive.update(n)
    return found, oracle, naive


def judge_draws(
    record_property: Callable,
    case: Callable[[int, int], tuple[dict, dict, dict]],
    *,
    seed: int,
    draws: int = DRAWS,
) -> None:
    """Judge one case with :func:`judge_case` over its draws: ``case(s, j)`` makes draw ``j``'s inputs from the seed
    ``s``, launches the package and returns :func:`judged` (or :func:`merge`) of its calls.  Every verdict is
    recorded (``TIER`` lines), and any contract or tier failure fails the test with the judge's numbers."""
    held: dict[int, dict] = {}

    def run(j: int, s: int):
        found, oracle, naive = case(s, j)
        held.clear()
        held[j] = naive
        return found, oracle

    judge, failures = judge_case(run, lambda j, tier: held[j], seed=seed, draws=draws)
    judge.record(record_property)
    assert not failures, "\n".join(failures)
