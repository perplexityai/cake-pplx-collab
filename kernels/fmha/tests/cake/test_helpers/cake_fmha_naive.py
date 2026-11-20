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

The accuracy judge of the family's dense multi-head attention backward (head dimension 64 or 128): the FP64 oracle,
the naive BF16 tier and the inputs of each judged form, for the judge of ``tests/cake_naive.py``.

Operands: ``q``, ``out`` and ``dout`` BF16 ``[B, Sq, H, D]``; ``k`` and ``v`` BF16 ``[B, Skv, H, D]``; ``lse`` FP32
``[B, H, Sq]``, the natural-log ``logsumexp`` of ``softmax_scale * q k^T`` over each query row, the causal mask applied
(top-left: query ``i`` sees the keys ``j <= i``; a causal call has ``Sq == Skv``).

:func:`make_inputs` draws ``q``, ``k`` and ``v`` unit ``randn`` and ``dout`` ``randn x 8``, all BF16, and takes ``out``
and ``lse`` from an FP32 forward of them with ``P`` rounded to BF16 before ``P v``, in 128-row chunks.  The logits
``q k^T * softmax_scale`` then have a standard deviation of 1, where the kernel's FP32 ``S`` and ``P`` part from the
tier's BF16 ones.

Each judged form is one function evaluated in one dtype: float64 is the oracle, bfloat16 the naive tier (PyTorch, the
same ops in BF16 on the kernel's own inputs; an FP32 input such as ``lse`` stays FP32 in the tier, as in BF16 user code
that reads it, :func:`stored`).  ``scale_log2`` is ``softmax_scale * log2(e)`` rounded to FP32, the kernel's argument.

* :func:`backward_outputs`: the public backward of ``(q, k, v, out, lse, dout)``.  With ``L2 = lse * log2(e)``,
  ``x = (q k^T) * scale_log2 - L2`` (masked), ``P = exp2(x)``, ``dPsum = rowsum(out * dout)`` and
  ``dS = P (dout v^T - dPsum)``: ``dq = dS k * softmax_scale`` (``[B, Sq, H, D]``), ``dk = dS^T q * softmax_scale``
  and ``dv = P^T dout`` (``[B, Skv, H, D]``).  ``dS`` and ``P`` are cast to the dtype as the operands of the products.
* :func:`main_outputs`: the main stage alone on its own inputs (:func:`main_buffers`): the same formula with the FP32
  ``LSE_log2 = lse * log2(e)`` and ``dPsum = rowsum(dout * out)`` as stored (the preprocess's outputs, computed in FP32
  as the preprocess does); it adds the unscaled ``dS k`` into ``dQ_accum``, FP32 ``[B*H*Sq*D]`` in the reduction layout
  the postprocess reads (per 128-row tile of a head, element ``(r, d)`` at ``(d // 4) * 512 + r * 4 + d % 4``), and
  writes ``dK`` and ``dV``.  ``dQ_accum`` is compared in that layout: the L2 and max norms do not depend on the order
  of the elements, and ``softmax_scale``, which the postprocess applies, would scale every error alike.  With
  ``pair=True`` (the 2-CTA main kernel of the pair route, head dimension 128) ``dQ_accum`` is in the pair-row layout:
  element ``(r, d)`` of a tile at ``(r // 64) * 8192 + (d // 4) * 256 + (r % 64) * 4 + d % 4`` (:func:`accumulator`).
* :func:`pre_outputs`: the preprocess (:func:`pre_inputs`: ``out`` and ``dout`` ``randn x 0.1``, ``lse`` ``log(Sq) +
  randn`` with every seventh value ``-inf``): ``dPsum = rowsum(out * dout)`` and ``LSE_log2 = lse * log2(e)`` (0 where
  ``lse`` is ``-inf``), FP32 ``[B*H, Sq]``, and ``dQ_accum`` cleared to 0.
* :func:`post_outputs`: the postprocess (:func:`post_inputs`: ``dQ_accum`` ``randn x 0.1`` in the reduction layout,
  or with ``pair=True`` in the pair-row layout the pair route's postprocess reads): ``dq = bf16(dQ_accum *
  softmax_scale)`` in the input layout ``[B, Sq, H, D]``, the FP32 accumulation and scale as stored.

:func:`judge_draws` runs the template's :func:`judge_case` over the draws of a case and records every verdict.
"""

from __future__ import annotations

import math
import types
from collections.abc import Callable, Mapping
from typing import NamedTuple, Optional

import torch

from tests.cake_naive import TIER_DTYPES, TIERS, judge_case

(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
ORACLE_DTYPE = torch.float64
BLOCK = 128
LOG2_E = math.log2(math.e)


class Case(NamedTuple):
    """One shape: ``B`` batches of ``H`` heads, ``S`` query rows and ``Skv`` keys (``Skv`` None: ``S``)."""

    B: int
    H: int
    S: int
    Skv: Optional[int] = None

    @property
    def kv(self) -> int:
        return self.S if self.Skv is None else self.Skv

    @property
    def id(self) -> str:
        tail = "" if self.Skv in (None, self.S) else f"-Skv{self.Skv}"
        return f"B{self.B}-H{self.H}-S{self.S}{tail}"


def stored(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x`` as a computation in ``dtype`` reads it: cast to ``dtype`` when ``x`` is stored no wider, else as is (an
    FP32 tensor stays FP32 in the BF16 tier and is widened in the float64 oracle)."""
    return x.to(dtype) if x.dtype.itemsize <= dtype.itemsize else x


def scale_log2(softmax_scale: float) -> float:
    """``softmax_scale * log2(e)`` rounded to FP32: the exponent scale the kernel receives."""
    return torch.tensor(softmax_scale * LOG2_E, dtype=torch.float32).item()


def mask(q_start: int, q_stop: int, Skv: int, device: torch.device) -> torch.Tensor:
    """The causal mask of queries ``q_start:q_stop`` over ``Skv`` keys: True where key ``j`` > query ``i``."""
    q = torch.arange(q_start, q_stop, device=device)[:, None]
    return torch.arange(Skv, device=device)[None, :] > q


# ----------------------------------------------------------------------------------------------------------------
# The public backward and the main stage


def make_inputs(
    case: Case,
    D: int,
    *,
    causal: bool,
    seed: int,
    device: torch.device,
    softmax_scale: Optional[float] = None,
) -> types.SimpleNamespace:
    """The operands of one backward call (module docstring): ``q``, ``k``, ``v``, ``dout`` drawn from a generator of
    ``device`` seeded with ``seed``; ``out`` and ``lse`` from the FP32 forward with ``P`` rounded to BF16."""
    B, H, Sq, Skv = case.B, case.H, case.S, case.kv
    if causal and Sq != Skv:
        raise ValueError(f"a causal case has Sq == Skv, got {Sq} and {Skv}")
    generator = torch.Generator(device=device).manual_seed(seed)
    scale = D**-0.5 if softmax_scale is None else float(softmax_scale)

    def tensor(seq: int, std: float) -> torch.Tensor:
        return (torch.randn(B, seq, H, D, device=device, generator=generator) * std).bfloat16()

    q, k = tensor(Sq, 1.0), tensor(Skv, 1.0)
    v, dout = tensor(Skv, 1.0), tensor(Sq, 8.0)
    qh, kh, vh = (x.float().permute(0, 2, 1, 3) for x in (q, k, v))
    out = torch.empty(B, H, Sq, D, device=device, dtype=torch.bfloat16)
    lse = torch.empty(B, H, Sq, device=device, dtype=torch.float32)
    for start in range(0, Sq, BLOCK):  # in row chunks: the scores of a long sequence do not fit at once
        scores = (qh[:, :, start : start + BLOCK] @ kh.transpose(-1, -2)) * scale
        if causal:
            scores = scores.masked_fill(mask(start, start + BLOCK, Skv, device), -math.inf)
        chunk_lse = scores.logsumexp(-1)
        p = (scores - chunk_lse[..., None]).exp().bfloat16().float()
        out[:, :, start : start + BLOCK] = (p @ vh).bfloat16()
        lse[:, :, start : start + BLOCK] = chunk_lse
    return types.SimpleNamespace(
        q=q,
        k=k,
        v=v,
        out=out.permute(0, 2, 1, 3).contiguous(),
        lse=lse,
        dout=dout,
        B=B,
        H=H,
        Sq=Sq,
        Skv=Skv,
        D=D,
        causal=causal,
        softmax_scale=scale,
    )


def _heads(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[B, S, H, D]`` as ``[B*H, S, D]`` in ``dtype``."""
    B, S, H, D = x.shape
    return x.to(dtype).permute(0, 2, 1, 3).reshape(B * H, S, D)


def _bshd(x: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """``[B*H, S, D]`` as ``[B, S, H, D]`` (a view)."""
    return x.reshape(B, H, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3)


def _probabilities(x: types.SimpleNamespace, q, k, lse_log2: torch.Tensor) -> torch.Tensor:
    """``P = exp2((q k^T) * scale_log2 - LSE_log2)``, masked, from the ``[B*H, S, D]`` operands."""
    logits = (q @ k.transpose(-1, -2)) * scale_log2(x.softmax_scale) - lse_log2[..., None]
    if x.causal:
        logits = logits.masked_fill(mask(0, x.Sq, x.Skv, logits.device), -math.inf)
    return torch.exp2(logits)


def backward_outputs(x: types.SimpleNamespace, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """``dq``, ``dk`` and ``dv`` of the public backward of ``x`` in ``dtype`` (float64: the oracle; bfloat16: the
    naive tier): ``lse`` as stored, ``dPsum`` the row sum of ``out * dout`` in ``dtype``."""
    q, k, v = _heads(x.q, dtype), _heads(x.k, dtype), _heads(x.v, dtype)
    out, dout = _heads(x.out, dtype), _heads(x.dout, dtype)
    lse = stored(x.lse, dtype).reshape(x.B * x.H, x.Sq)
    p = _probabilities(x, q, k, lse * LOG2_E)
    dpsum = (out * dout).sum(-1)
    ds = (p * (dout @ v.transpose(-1, -2) - dpsum[..., None])).to(dtype)
    p = p.to(dtype)
    scale = x.softmax_scale
    return {
        "dq": _bshd((ds @ k) * scale, x.B, x.H),
        "dk": _bshd((ds.transpose(-1, -2) @ q) * scale, x.B, x.H),
        "dv": _bshd(p.transpose(-1, -2) @ dout, x.B, x.H),
    }


def main_buffers(x: types.SimpleNamespace) -> types.SimpleNamespace:
    """The main stage's own inputs and output buffers for the operands ``x``: the FP32 ``lse_log2`` and ``dpsum``
    ``[B*H, Sq]`` the preprocess would write (computed in FP32 here), ``dq_accum`` zeroed (the main stage adds into
    it), ``dk`` and ``dv`` ``[B, Skv, H, D]`` filled with NaN (every element must be written)."""
    bh, device = x.B * x.H, x.q.device
    dpsum = (x.dout.float() * x.out.float()).sum(-1).permute(0, 2, 1).reshape(bh, x.Sq).contiguous()
    return types.SimpleNamespace(
        lse_log2=(x.lse * LOG2_E).reshape(bh, x.Sq).contiguous(),
        dpsum=dpsum,
        dq_accum=torch.zeros(bh * x.Sq * x.D, device=device, dtype=torch.float32),
        dk=torch.full((x.B, x.Skv, x.H, x.D), math.nan, device=device, dtype=torch.bfloat16),
        dv=torch.full((x.B, x.Skv, x.H, x.D), math.nan, device=device, dtype=torch.bfloat16),
    )


def accumulator(dq: torch.Tensor, *, pair: bool = False) -> torch.Tensor:
    """``dq`` ``[B*H, Sq, D]`` flattened in the layout of the FP32 dQ accumulator: per 128-row tile of a head, element
    ``(r, d)`` at ``(d // 4) * 512 + r * 4 + d % 4`` (the reduction layout of the 1-CTA main kernels), or with ``pair``
    at ``(r // 64) * (D // 4) * 256 + (d // 4) * 256 + (r % 64) * 4 + d % 4`` (the pair-row layout of the 2-CTA main
    kernel, ``D`` 128: each CTA of a pair adds its 64 rows of a tile as one block)."""
    bh, S, D = dq.shape
    if pair:
        # [bh, tile, r // 64, d // 4, r % 64, d % 4]
        return dq.view(bh, S // BLOCK, 2, BLOCK // 2, D // 4, 4).permute(0, 1, 2, 4, 3, 5).reshape(-1)
    # [bh, tile, d // 4, r, d % 4]
    return dq.view(bh, S // BLOCK, BLOCK, D // 4, 4).permute(0, 1, 3, 2, 4).reshape(-1)


def from_accumulator(acc: torch.Tensor, B: int, H: int, S: int, D: int, *, pair: bool = False) -> torch.Tensor:
    """The FP32 dQ accumulator ``acc`` (:func:`accumulator`'s layout) as ``[B, H, S, D]``."""
    if pair:
        # [bh * tiles, r // 64, d // 4, r % 64, d % 4] -> [bh * tiles, r // 64, r % 64, d // 4, d % 4]
        tiles = acc.view(-1, 2, D // 4, BLOCK // 2, 4).permute(0, 1, 3, 2, 4)
    else:
        # [bh * tiles, d // 4, r, d % 4] -> [bh * tiles, r, d // 4, d % 4]
        tiles = acc.view(-1, D // 4, BLOCK, 4).permute(0, 2, 1, 3)
    return tiles.reshape(B, H, S, D)


def main_outputs(
    x: types.SimpleNamespace, m: types.SimpleNamespace, dtype: torch.dtype, *, pair: bool = False
) -> dict[str, torch.Tensor]:
    """``dQ_accum`` (unscaled, in the reduction layout, or with ``pair`` in the pair-row layout of the 2-CTA main
    kernel: :func:`accumulator`), ``dK`` and ``dV`` of the main stage on ``x`` and its buffers ``m`` in ``dtype``: the
    FP32 ``lse_log2`` and ``dpsum`` as stored, ``P`` and ``dS`` in ``dtype``."""
    q, k, v, dout = (_heads(t, dtype) for t in (x.q, x.k, x.v, x.dout))
    p = _probabilities(x, q, k, stored(m.lse_log2, dtype))
    ds = (p * (dout @ v.transpose(-1, -2) - stored(m.dpsum, dtype)[..., None])).to(dtype)
    p = p.to(dtype)
    dq = accumulator(ds @ k, pair=pair)
    return {
        "dQ_accum": dq,
        "dK": _bshd((ds.transpose(-1, -2) @ q) * x.softmax_scale, x.B, x.H),
        "dV": _bshd(p.transpose(-1, -2) @ dout, x.B, x.H),
    }


# ----------------------------------------------------------------------------------------------------------------
# The preprocess and the postprocess


def pre_inputs(case: Case, D: int, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """The preprocess's inputs (module docstring) and its output buffers filled with NaN."""
    B, H, S = case.B, case.H, case.S
    generator = torch.Generator(device=device).manual_seed(seed)

    def tensor() -> torch.Tensor:
        return torch.randn(B, S, H, D, device=device, generator=generator).mul_(0.1).bfloat16()

    lse = torch.randn(B, H, S, device=device, generator=generator).add_(math.log(S))
    lse.view(-1)[::7] = -math.inf
    return types.SimpleNamespace(
        out=tensor(),
        dout=tensor(),
        lse=lse,
        lse_log2=torch.full((B * H, S), math.nan, device=device, dtype=torch.float32),
        dpsum=torch.full((B * H, S), math.nan, device=device, dtype=torch.float32),
        dq_accum=torch.full((B * H * S * D,), math.nan, device=device, dtype=torch.float32),
        B=B,
        H=H,
        S=S,
        D=D,
    )


def pre_outputs(t: types.SimpleNamespace, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """``dPsum``, ``LSE_log2`` and the cleared ``dQ_accum`` in ``dtype``, from the FP32 ``lse`` as stored."""
    out, dout = _heads(t.out, dtype), _heads(t.dout, dtype)
    lse = stored(t.lse.flatten(0, 1), dtype)
    return {
        "dPsum": (out * dout).sum(-1),
        "LSE_log2": torch.where(lse.isneginf(), 0.0, lse * LOG2_E).to(dtype),
        "dQ_accum": torch.zeros(t.dq_accum.shape, device=t.dq_accum.device, dtype=dtype),
    }


def post_inputs(case: Case, D: int, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """The postprocess's input (module docstring), ``softmax_scale = D^-0.5``, and its ``dq`` filled with NaN."""
    B, H, S = case.B, case.H, case.S
    generator = torch.Generator(device=device).manual_seed(seed)
    return types.SimpleNamespace(
        dq_accum=torch.randn(B * H * S * D, device=device, generator=generator).mul_(0.1),
        dq=torch.full((B, S, H, D), math.nan, device=device, dtype=torch.bfloat16),
        softmax_scale=D**-0.5,
        B=B,
        H=H,
        S=S,
        D=D,
    )


def post_outputs(t: types.SimpleNamespace, dtype: torch.dtype, *, pair: bool = False) -> dict[str, torch.Tensor]:
    """``dq = dQ_accum * softmax_scale`` in ``[B, Sq, H, D]`` in ``dtype``, the FP32 accumulation (in the reduction
    layout, or with ``pair`` in the pair-row layout: :func:`accumulator`) and scale (the kernel's FP32 argument) as
    stored."""
    tiles = from_accumulator(t.dq_accum, t.B, t.H, t.S, t.D, pair=pair)
    scale = torch.tensor(t.softmax_scale, dtype=torch.float32, device=t.dq_accum.device)
    return {"dq": (stored(tiles, dtype) * stored(scale, dtype)).to(dtype).permute(0, 2, 1, 3)}


# ----------------------------------------------------------------------------------------------------------------
# Contract and judge


def assert_backward_contract(label: str, x: types.SimpleNamespace, got: Mapping[str, torch.Tensor]) -> None:
    """The contract of the outputs ``got`` (any of ``dq``, ``dk``, ``dv``) of one backward call on ``x``: BF16 on the
    operands' device, contiguous, ``dq`` of ``q``'s shape ``[B, Sq, H, D]`` and ``dk`` / ``dv`` of ``k``'s
    ``[B, Skv, H, D]``, every element finite."""
    shapes = {"dq": tuple(x.q.shape), "dk": tuple(x.k.shape), "dv": tuple(x.k.shape)}
    for name, value in got.items():
        found = (tuple(value.shape), value.dtype, value.device)
        assert found == (shapes[name], torch.bfloat16, x.q.device), (label, name, found)
        assert value.is_contiguous(), (label, name, tuple(value.stride()))
        assert bool(torch.isfinite(value).all()), f"{label}: non-finite {name}"


def judged_backward(
    x: types.SimpleNamespace, runs: Mapping[str, Mapping[str, torch.Tensor]]
) -> tuple[dict, dict, dict]:
    """``(outputs, oracle, naive)`` of one draw for :func:`judge_draws`: the backward calls ``runs`` (label ->
    outputs) on ``x``, each held to :func:`assert_backward_contract`, beside the FP64 oracle and the naive tier of
    ``x``.  With several calls each output is named ``"<label> <output>"``."""
    ref = backward_outputs(x, ORACLE_DTYPE)
    low = backward_outputs(x, NAIVE_DTYPE)
    found, oracle, naive = {}, {}, {}
    for label, got in runs.items():
        assert_backward_contract(label, x, got)
        prefix = f"{label} " if len(runs) > 1 else ""
        for name, value in got.items():
            found[prefix + name], oracle[prefix + name], naive[prefix + name] = value, ref[name], low[name]
    return found, oracle, naive


def judged_stage(
    outputs: Callable[[types.SimpleNamespace, torch.dtype], Mapping[str, torch.Tensor]],
    t: types.SimpleNamespace,
    got: Mapping[str, torch.Tensor],
) -> tuple[dict, dict, dict]:
    """``(outputs, oracle, naive)`` of one draw of a stage: ``got`` (name -> the buffer the stage wrote) beside
    ``outputs(t, float64)`` and ``outputs(t, bfloat16)`` of the same names."""
    ref = outputs(t, ORACLE_DTYPE)
    low = outputs(t, NAIVE_DTYPE)
    return dict(got), {n: ref[n] for n in got}, {n: low[n] for n in got}


def judge_draws(record_property: Callable, case: Callable[[int], tuple[dict, dict, dict]], *, seed: int = 0) -> None:
    """Judge one case with :func:`judge_case` over its draws: ``case(s)`` makes the draw's inputs from the seed ``s``,
    launches the package and returns the ``(outputs, oracle, naive)`` of :func:`judged_backward` or
    :func:`judged_stage`.  Every verdict is recorded (``TIER`` lines), and any contract or tier failure fails the test
    with the judge's numbers."""
    held: dict[int, dict] = {}

    def run(j: int, s: int):
        found, oracle, naive = case(s)
        held.clear()
        held[j] = naive
        return found, oracle

    judge, failures = judge_case(run, lambda j, tier: held[j], seed=seed)
    judge.record(record_property)
    assert not failures, "\n".join(failures)
