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

The accuracy judge of the family's DSA sparse-attention training op (``H`` query heads, top-k index rows): the FP64
oracle, the naive BF16 tier and the contract of one call, for the judge of ``tests/cake_naive.py``.

A :class:`Problem` holds one call's own inputs: BF16 ``q_latent [T, H, 512]`` and ``kv_latent [S, 512]``, in the
RoPE form also ``q_rope [T, H, 64]`` and ``k_rope [S, 64]`` (the NoPE form has none: the latent is key and value),
the int32 global key rows ``indices [T, K]``, the optional ``topk_length [T]``, ``dout [T, H, 512]`` (any strided
view) and the softmax scale.  The head count ``H`` is the problem's own, ``q_latent.shape[1]``
(:attr:`Problem.heads`): every head is an independent attention over the same key rows, so the oracle, the tier
and the contract hold for any ``H`` (the package's RoPE form takes 64, 96 or 128 heads, its NoPE form 64).  Slot
``j`` of row ``t`` is valid when ``0 <= indices[t, j] < S`` and, with ``topk_length``, ``j < topk_length[t]``.

:func:`outputs` evaluates the forward and the backward of the problem with every op in one dtype: float64 is the
oracle, bfloat16 the naive tier (torch's own ops on BF16 tensors, each rounding its result to BF16 after torch's
internal accumulation).  Per row, with ``q`` / ``k`` the latent and rope parts side by side (the latent alone in the
NoPE form) and ``v`` the latent: the gathered keys ``k_j = k[indices[t, j]]``, the scores ``s = q k_j^T * scale``
over the valid slots, ``lse = logsumexp(s)``, ``P = softmax(s)``, ``out = P v``, ``dP = dout v^T``, ``dS = P (dP -
rowsum(dout * out)) * scale``, ``dq = dS k``, and per slot the key gradient ``dS^T q`` plus the value gradient
``P^T dout`` (each summed over the ``H`` heads), added into key row ``indices[t, j]`` with ``index_add_``.  The RoPE
form returns ``dq_latent`` / ``dq_rope`` and ``dkv_latent`` / ``dk_rope`` (latent and rope columns of the key
gradient, the value gradient in the latent), the NoPE form ``dq`` and the combined ``dkv``.  A row without a valid
slot has ``out = 0``, ``lse = -inf`` and ``dq = 0``; a key row no valid slot names has zero key gradients.  The rows
go in blocks of at most :data:`BLOCK_SLOTS` (row, slot) pairs: each row's quantities stay inside its block and each
slot's key gradient is its own ``index_add_`` term, so the blocks bound the memory and do not change the values.

A NoPE problem may carry a trainable per-head sink, FP32 ``sink [H]``: the sink column ``sink[h]`` (not scaled) is
appended to every row's scores of head ``h`` -- in the dtype of the scores, ``sink.to(dtype)`` -- before
``logsumexp`` and ``softmax``, and its probability is dropped before ``P v`` (the sink has no value).  ``lse`` then
includes the sink, a row without a valid slot has ``out = 0``, ``dq = 0`` and ``lse = sink[h]`` (``-inf`` for a
``-inf`` sink), and the gradients above use the key part of that softmax.  The sink gradient ``dsink [H]`` is, in
float64 (the oracle), ``-sum_t exp(sink - lse[t]) * delta[t]`` with ``delta = rowsum(dout * out)`` (0 where ``lse``
is ``-inf``); in the naive tier it is torch's autograd of the same BF16 forward with respect to the sink column, one
term per row (0 on a row whose scores and sink are all ``-inf``, as its ``P``), summed over all rows by one BF16
``sum`` (the reduction of the broadcast sink's gradient), so the blocks do not change it either.

:func:`judged` holds the outputs of one or several calls on a problem to their contract (:func:`assert_contract`:
shapes, dtypes, finite values, the exact zeros) and pairs each with the oracle and the tier of the same draw.  The
forward's ``o_lo`` (the BF16 residual of the FP32 output) is judged as ``o_sum = out + o_lo`` in FP32; a
``with_o_lo=False`` call returns ``None`` there, which is left out (:data:`WITH_O_LO`).  An
accumulating call (``dkv_acc``) is judged by its increment: the buffer after the call minus the buffer before it,
exact in float64, against the key gradients added through the destination map (the error of that increment is the
error of the buffer against the oracle's ``before + gradient``).  :func:`judge_draws` runs the template's
:func:`judge_case` over the draws and records every verdict.
"""

from __future__ import annotations

import inspect
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Optional

import pytest
import torch

from tests.cake_naive import DRAWS, TIER_DTYPES, TIERS, judge_case

#: The head count of the package's 64-head problems (the NoPE form, the default inputs); a problem's own count is
#: :attr:`Problem.heads`.
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
    #: NoPE only: the FP32 ``[H]`` trainable per-head sink (``None``: no sink).
    sink: Optional[torch.Tensor] = None

    @property
    def heads(self) -> int:
        """The query head count ``H`` of the problem: ``q_latent.shape[1]``."""
        return int(self.q_latent.shape[1])

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
    total_q, total_k, heads = p.total_q, p.total_k, p.heads
    width = int(p.indices.shape[1]) if p.indices.dim() == 2 else 0
    on = dict(device=device, dtype=dtype)
    scale = p.softmax_scale
    kv = p.kv_latent.reshape(total_k, D_LATENT).to(dtype)
    k_all = kv if p.nope else torch.cat([kv, p.k_rope.reshape(total_k, D_ROPE).to(dtype)], -1)
    d_k = int(k_all.shape[-1])
    want_grad = backward and p.dout is not None
    sink = None if p.sink is None else p.sink.to(device=device, dtype=dtype)
    out = torch.zeros(total_q, heads, D_LATENT, **on)
    # a row without a valid slot: lse = -inf, or the sink (logsumexp of the sink column alone)
    lse = torch.full((total_q, heads), -math.inf, **on)
    if sink is not None:
        lse = lse.copy_(sink.expand(total_q, heads))
    if want_grad:
        dq = torch.zeros(total_q, heads, d_k, **on)
        dk = torch.zeros(total_k, d_k, **on)
        if sink is not None:
            # each row's sink-gradient term
            dsink_rows = torch.zeros(total_q, heads, **on)
    if total_q and width and total_k:
        valid_all = p.valid()
        rows_per_block = max(1, BLOCK_SLOTS // width)
        for r0 in range(0, total_q, rows_per_block):
            r1 = min(total_q, r0 + rows_per_block)
            valid = valid_all[r0:r1]
            ix = p.indices[r0:r1].long().clamp(0, total_k - 1)
            q = p.q_latent[r0:r1].to(dtype)
            if not p.nope:
                q = torch.cat([q, p.q_rope[r0:r1].to(dtype)], -1)  # [rows, H, 576]
            kg = k_all[ix]  # [rows, K, d_k]
            vg = kg[..., :D_LATENT]
            s = _bmm(q, kg.transpose(1, 2)) * scale  # [rows, H, K]
            s = s.masked_fill(~valid[:, None, :], -math.inf)
            g = p.dout[r0:r1].to(dtype) if want_grad else None
            if sink is None:
                row_lse = torch.logsumexp(s, -1)
                # a row without a valid slot: NaN -> 0
                prob = torch.softmax(s, -1).nan_to_num(0.0)
                o = _bmm(prob, vg)
            else:
                row_lse, prob, o, term = _sink_rows(s, sink, vg, g, dtype)
                if want_grad:
                    dsink_rows[r0:r1] = term
            out[r0:r1], lse[r0:r1] = o, row_lse
            if not want_grad:
                continue
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
            if sink is not None:
                found["dsink"] = dsink_rows.sum(0)
        else:
            found.update(
                dq_latent=dq[..., :D_LATENT],
                dq_rope=dq[..., D_LATENT:],
                dkv_latent=dk[:, :D_LATENT],
                dk_rope=dk[:, D_LATENT:],
            )
    return found


def _sink_rows(
    s: torch.Tensor,
    sink: torch.Tensor,
    vg: torch.Tensor,
    g: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(lse, P, out, dsink term)`` of a block of rows with the sink column: the scores ``s [rows, H, K]`` (``-inf``
    on invalid slots) and ``sink[h]`` side by side, ``lse = logsumexp``, ``P`` the softmax's key part, ``out = P v``.
    The sink term of each row (``None`` without ``g``): in float64 ``-exp(sink - lse) * rowsum(g * out)`` (0 where
    ``lse`` is ``-inf``), otherwise the autograd of this forward in ``dtype`` with respect to the row's own copy of
    the sink column."""
    rows, heads = int(s.shape[0]), int(s.shape[1])
    column = sink[None, :, None].expand(rows, heads, 1)
    if g is not None and dtype != torch.float64:
        column = column.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = torch.cat([s, column], -1)
        row_lse = torch.logsumexp(logits, -1)
        full = torch.softmax(logits, -1)
        # a row whose scores and sink are all -inf: NaN -> 0 (its key part; the sink part has no value)
        prob = torch.where(torch.isnan(full[..., :-1]), 0.0, full[..., :-1])
        o = _bmm(prob, vg)
        term = None
        if g is not None:
            if dtype == torch.float64:
                live = torch.isfinite(row_lse)
                p_sink = torch.where(
                    live,
                    torch.exp(sink[None, :] - torch.where(live, row_lse, 0.0)),
                    0.0,
                )
                term = -p_sink * (g * o).sum(-1)
            else:
                # a row whose scores and sink are all -inf has no softmax (NaN -> 0, as P above)
                (grad,) = torch.autograd.grad(o, column, g)
                term = grad[..., 0].nan_to_num(0.0)
    return row_lse.detach(), prob.detach(), o.detach(), term


FORWARD = ("out", "lse", "o_lo")
#: Every output name a call may report, by form (``o_lo`` is judged as ``o_sum``; ``dkv_acc`` by its increment;
#: ``dsink`` only for a problem with a sink).
OUTPUTS_ROPE = FORWARD + ("dq_latent", "dq_rope", "dkv_latent", "dk_rope", "dkv_acc")
OUTPUTS_NOPE = FORWARD + ("dq", "dkv", "dkv_acc", "dsink")
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
    "dsink": (torch.float32,),
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
    exactly ``-inf`` on the rows without a valid slot -- with a sink, exactly the sink there, ``-inf`` for a ``-inf``
    head); ``out``, ``o_lo`` and ``dq*`` are exactly 0 on those rows; the fresh key gradients are exactly 0 on the
    key rows no valid slot names; ``dsink`` is finite and exactly 0 on every ``-inf`` head; an accumulating call
    keeps the bits of every destination row no key row maps to and of every column past the gradient width."""
    names = OUTPUTS_NOPE if p.nope else OUTPUTS_ROPE
    if p.sink is None:
        names = tuple(name for name in names if name != "dsink")
    unknown = set(got) - set(names)
    assert not unknown, f"{label}: outputs {sorted(unknown)} are not outputs of the {'NoPE' if p.nope else 'RoPE'} form"
    valid = p.valid()
    no_slot = ~valid.any(-1)
    referenced = torch.zeros(p.total_k, dtype=torch.bool, device=valid.device)
    referenced[p.indices.long()[valid]] = True
    t_q, t_k, heads = p.total_q, p.total_k, p.heads
    shapes = {
        "out": (t_q, heads, D_LATENT),
        "lse": (t_q, heads),
        "o_lo": (t_q, heads, D_LATENT),
        "dq_latent": (t_q, heads, D_LATENT),
        "dq_rope": (t_q, heads, D_ROPE),
        "dq": (t_q, heads, D_LATENT),
        "dkv_latent": (t_k, D_LATENT),
        "dk_rope": (t_k, D_ROPE),
        "dkv": (t_k, D_LATENT),
        "dsink": (heads,),
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
            if p.sink is None:
                assert bool((value[no_slot] == -math.inf).all()), (
                    f"{label}: lse of a row without a valid slot is not -inf"
                )
                assert bool(torch.isfinite(value[~no_slot]).all()), (
                    f"{label}: non-finite lse"
                )
                continue
            sink = p.sink.to(device=value.device, dtype=value.dtype)
            empty = value[no_slot]
            assert torch.equal(empty, sink.expand_as(empty)), (
                f"{label}: lse of a row without a valid slot is not the sink"
            )
            finite = ~no_slot[:, None] | torch.isfinite(sink)[None, :]
            assert bool(torch.isfinite(value[finite]).all()), f"{label}: non-finite lse"
            continue
        if name == "dsink":
            assert bool(torch.isfinite(value).all()), f"{label}: non-finite dsink"
            dead = ~torch.isfinite(p.sink.to(value.device))
            assert _zero(value[dead]), f"{label}: dsink of a -inf sink head is not 0"
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


#: Whether the calls a test judges write ``o_lo`` (``with_o_lo=True``, the default).  :func:`with_o_lo_case` sets it
#: to ``False`` for the ``with_o_lo=False`` cases: every ``o_lo`` a run reports must then be ``None`` and is left out,
#: so the judge builds no ``o_sum`` (``out``, ``lse`` and the gradients keep their judgement); under the default an
#: ``o_lo`` a run reports must be a tensor.
WITH_O_LO = True

#: ``pytest.fixture`` arguments of the training suites' ``with_o_lo`` fixture: the default, then ``with_o_lo=False``
#: under the ``no_olo`` marker.
WITH_O_LO_FIXTURE = dict(
    params=[True, pytest.param(False, marks=pytest.mark.no_olo)],
    ids=["with_o_lo", "no_olo"],
)


def with_o_lo_case(with_o_lo: bool, monkeypatch, backend, jit, *modules, caller=None):
    """Run one case of a training-suite test under ``with_o_lo`` (the body of the suites' ``with_o_lo`` fixture,
    a generator: ``yield from`` it).  ``False`` makes ``with_o_lo=False`` the default of every function of
    ``backend`` and ``modules`` that takes the keyword with a default -- the entries, the step entries, the prepared
    runners and the binding-key builders, not ``generated_program_available`` (a missing stage then fails the case
    instead of skipping it) -- so each call of the test, its helpers included, runs the mode as a caller passing
    ``with_o_lo=False`` does; :data:`WITH_O_LO` follows.  ``caller`` (the test module) has the same done to the
    functions of those modules it holds under its own names: a reload of a module replaces its functions there but
    not the ones the test module imported.  Every case gets an empty binding cache, and the stages its plans resolve
    (``backend._stage_entry``) are recorded: when the case ends none may belong to the other mode -- under ``False``
    no forward or delta stage that has a ``with_o_lo=False`` twin (``jit.NO_OLO_STAGES``, and the head-window
    forward ``fwd_hw`` beside its twin ``fwd_hw_no_olo``, ``backend.HEAD_WINDOW_STAGES``, and the single-pass forward
    ``fwd_h128`` beside ``fwd_h128_no_olo``, ``backend.SINGLE_PASS_STAGES``), so every plan of the case ran the twins
    (a plan resolves its forward stage first), under ``True`` no twin."""
    resolved: list = []
    stage_entry = backend._stage_entry

    def recorded(stage, *args, **kwargs):
        resolved.append(stage)
        return stage_entry(stage, *args, **kwargs)

    monkeypatch.setattr(backend, "_stage_entry", recorded)
    monkeypatch.setattr(backend.BINDING_CACHE, "_entries", OrderedDict())
    monkeypatch.setitem(globals(), "WITH_O_LO", with_o_lo)
    if not with_o_lo:
        defining = {module.__name__ for module in (backend, *modules)}
        holders = (backend, *modules) + (() if caller is None else (caller,))
        for holder in holders:
            for value in list(vars(holder).values()):
                function = inspect.unwrap(value) if inspect.isfunction(value) else None
                if (
                    function is None
                    or function.__module__ not in defining
                    or function.__name__ == "generated_program_available"
                ):
                    continue
                defaults = function.__kwdefaults__
                if defaults is not None and defaults.get("with_o_lo") is True:
                    monkeypatch.setitem(defaults, "with_o_lo", False)
    yield with_o_lo
    twins = dict(jit.NO_OLO_STAGES)
    head_window = getattr(backend, "HEAD_WINDOW_STAGES", {})
    twins.update(
        (head_window[stage], head_window[twin])
        for stage, twin in jit.NO_OLO_STAGES.items()
        if stage in head_window and twin in head_window
    )
    single_pass = getattr(backend, "SINGLE_PASS_STAGES", {})
    twins.update(
        (single_pass[stage], single_pass[twin])
        for stage, twin in list(twins.items())
        if stage in single_pass and twin in single_pass
    )
    other = set(twins.values()) if with_o_lo else set(twins)
    wrong = sorted(other.intersection(resolved))
    assert not wrong, (
        f"a with_o_lo={with_o_lo} case planned the stages {wrong} of the other mode"
    )


def _o_lo_slot(label: str, got: Mapping) -> Mapping:
    """``got`` as :func:`judged` reads it under :data:`WITH_O_LO`: a ``with_o_lo=False`` call's ``None`` ``o_lo``
    left out, a default call's ``o_lo`` a tensor."""
    if "o_lo" not in got:
        return got
    if WITH_O_LO:
        assert got["o_lo"] is not None, (
            f"{label}: a with_o_lo=True call returned no o_lo"
        )
        return got
    assert got["o_lo"] is None, f"{label}: a with_o_lo=False call returned an o_lo"
    return {name: value for name, value in got.items() if name != "o_lo"}


def judged(
    p: Problem, runs: Mapping[str, Mapping], *, prefix: str = ""
) -> tuple[dict, dict, dict]:
    """``(outputs, oracle, naive)`` of one draw for :func:`judge_draws`: the calls ``runs`` (label -> outputs by
    name, see :data:`OUTPUTS_ROPE` / :data:`OUTPUTS_NOPE`) on the problem ``p``, each held to
    :func:`assert_contract`, beside the FP64 oracle and the naive tier of ``p``.  With several calls each output is
    named ``"<label> <output>"``; ``prefix`` goes in front of every name (several problems in one draw).  An
    ``o_lo`` follows :data:`WITH_O_LO`."""
    runs = {label: _o_lo_slot(label or "call", got) for label, got in runs.items()}
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
