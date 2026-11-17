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

The accuracy judge of the family's packed-varlen causal GQA training attention (head dimension 256): the FP64 oracle,
the naive BF16 tier and the contract of one call, for the judge of ``tests/cake_naive.py``.

:func:`outputs` evaluates the forward and the backward of a call from the call's own inputs (BF16 ``q``, ``k``, ``v``
and ``dout``, the segment boundaries), every op in one dtype: float64 is the oracle, bfloat16 the naive tier.  Query
row ``i`` of a segment with ``q_len`` rows and ``k_len`` keys sees the keys ``j <= i + k_len - q_len``; query head
``h`` reads key head ``h // (H_q / H_kv)``.  Per segment, with ``S = q k^T * scale`` under the mask: ``lse =
logsumexp(S)``, ``P = exp(S - lse)``, ``out = P v``, ``dS = P (dout v^T - rowsum(dout * out))``, ``dq = dS k * scale``,
``dk = scale * sum over the group of dS^T q`` and ``dv = sum over the group of P^T dout``.  A row that sees no key has
``out = 0``, ``lse = -inf`` and ``dq = 0``; a key that no query sees has ``dk = dv = 0``.  The row-map form adds the
window gradients into FP32 destinations (:func:`row_map_adds`): the oracle in float64, the tier in FP32 (the
destinations as stored) cast to bfloat16.

:func:`judged` holds each call to its contract (:func:`assert_contract`: shapes, dtypes, finite outputs, the exact
zeros, untouched destination rows) and pairs its outputs with the oracle and the tier of the same draw;
:func:`judge_draws` runs the template's :func:`judge_case` over the draws and records every verdict.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch

from tests.cake_naive import TIER_DTYPES, TIERS, judge_case

SCALE = 0.0625
(_TIER,) = TIERS
NAIVE_DTYPE = TIER_DTYPES[_TIER]
_BITS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
# dK / dV dtype of each form: BF16 leaves, the FP32 window, FP32 row-map destinations.
DKV_DTYPE = {"bf16": torch.bfloat16, "fp32": torch.float32, "rowmap": torch.float32}


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b``; a contraction of length 1 (a one-row or one-key segment) as the broadcast product it equals, since
    torch may route that outer product to a Triton kernel, which needs a ptxas that knows the GPU."""
    return a * b if a.shape[-1] == 1 else a @ b


def outputs(x: Mapping, dtype: torch.dtype, scale: float = SCALE) -> dict[str, torch.Tensor]:
    """``out [T_q, H_q, 256]``, ``lse [H_q, T_q]``, ``dq``, ``dk`` and ``dv`` of the inputs ``x`` (``q``, ``k``, ``v``,
    ``dout`` and the host boundaries ``cq`` / ``ck``), every op in ``dtype`` on the inputs' device."""
    q_all, k_all, v_all, do_all = x["q"], x["k"], x["v"], x["dout"]
    device = q_all.device
    t_q, h_q, d = q_all.shape
    t_k, h_kv, _ = k_all.shape
    group = h_q // h_kv
    on = dict(device=device, dtype=dtype)
    out = torch.zeros(t_q, h_q, d, **on)
    lse = torch.full((h_q, t_q), -math.inf, **on)
    dq = torch.zeros(t_q, h_q, d, **on)
    dk = torch.zeros(t_k, h_kv, d, **on)
    dv = torch.zeros(t_k, h_kv, d, **on)
    cq, ck = x["cq"], x["ck"]
    for b in range(len(cq) - 1):
        q0, q1, k0, k1 = cq[b], cq[b + 1], ck[b], ck[b + 1]
        q_len, k_len = q1 - q0, k1 - k0
        if q_len == 0 or k_len == 0:
            continue
        q = q_all[q0:q1].to(dtype).transpose(0, 1)  # [H_q, q_len, 256]
        do = do_all[q0:q1].to(dtype).transpose(0, 1)
        k = k_all[k0:k1].to(dtype).transpose(0, 1).repeat_interleave(group, 0)  # [H_q, k_len, 256]
        v = v_all[k0:k1].to(dtype).transpose(0, 1).repeat_interleave(group, 0)
        s = _mm(q, k.transpose(1, 2)) * scale
        rows = torch.arange(q_len, device=device).view(-1, 1)
        cols = torch.arange(k_len, device=device).view(1, -1)
        s.masked_fill_(cols > rows + (k_len - q_len), -math.inf)
        row_lse = torch.logsumexp(s, -1)  # -inf where a row sees no key
        p = torch.exp(s - row_lse.unsqueeze(-1)).nan_to_num_(0.0)
        o = _mm(p, v)
        ds = p * (_mm(do, v.transpose(1, 2)) - (do * o).sum(-1).unsqueeze(-1))
        out[q0:q1] = o.transpose(0, 1)
        lse[:, q0:q1] = row_lse
        dq[q0:q1] = (_mm(ds, k) * scale).transpose(0, 1)
        dk[k0:k1] = (_mm(ds.transpose(1, 2), q).unflatten(0, (h_kv, group)).sum(1) * scale).transpose(0, 1)
        dv[k0:k1] = _mm(p.transpose(1, 2), do).unflatten(0, (h_kv, group)).sum(1).transpose(0, 1)
    return {"out": out, "lse": lse, "dq": dq, "dk": dk, "dv": dv}


def row_map_adds(init: torch.Tensor, row_map: torch.Tensor, window: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The destinations ``init`` with every window row ``j`` added into row ``row_map[j]``, in ``dtype``: ``init``
    (FP32) as the dtype reads it (widened to float64, kept FP32 for bfloat16), the window in that dtype."""
    base = init.to(dtype) if init.dtype.itemsize <= dtype.itemsize else init
    return base.index_add(0, row_map.long(), window.to(base.dtype)).to(dtype)


def no_key_rows(x: Mapping) -> torch.Tensor:
    """Mask ``[T_q]`` of the query rows that see no key (``i + k_len - q_len < 0``)."""
    rows = []
    for b in range(len(x["cq"]) - 1):
        q_len, k_len = x["cq"][b + 1] - x["cq"][b], x["ck"][b + 1] - x["ck"][b]
        rows += [i + k_len - q_len < 0 for i in range(q_len)]
    return torch.tensor(rows, dtype=torch.bool, device=x["q"].device)


def unseen_keys(x: Mapping) -> torch.Tensor:
    """Mask ``[T_k]`` of the keys that no query sees: the keys of the segments without queries."""
    keys = []
    for b in range(len(x["cq"]) - 1):
        keys += [x["cq"][b + 1] == x["cq"][b]] * (x["ck"][b + 1] - x["ck"][b])
    return torch.tensor(keys, dtype=torch.bool, device=x["q"].device)


def assert_contract(
    label: str,
    x: Mapping,
    form: str,
    got: Mapping[str, torch.Tensor],
    dst_init: tuple[torch.Tensor, torch.Tensor] | None = None,
    row_map: torch.Tensor | None = None,
) -> None:
    """The contract of the outputs ``got`` (any of ``out``, ``lse``, ``dq``, ``dk``, ``dv``) of one call on ``x`` in
    the dK / dV ``form``: each output has its shape and dtype and is finite, except ``lse``, which is exactly ``-inf``
    on the rows that see no key; ``out`` and ``dq`` are exactly 0 on those rows; ``dk`` and ``dv`` are exactly 0 on
    the keys no query sees (BF16 and FP32 window forms) and keep the bits of ``dst_init`` on the destination rows no
    window row maps to (row-map form)."""
    q, k = x["q"], x["k"]
    t_q, h_q, _ = q.shape
    dkv_shape = tuple(dst_init[0].shape) if form == "rowmap" else tuple(k.shape)
    shapes = {"out": tuple(q.shape), "lse": (h_q, t_q), "dq": tuple(q.shape), "dk": dkv_shape, "dv": dkv_shape}
    dtypes = {"out": torch.bfloat16, "lse": torch.float32, "dq": torch.bfloat16}
    dtypes["dk"] = dtypes["dv"] = DKV_DTYPE[form]
    no_key = no_key_rows(x)
    for t, value in got.items():
        assert (tuple(value.shape), value.dtype) == (shapes[t], dtypes[t]), (label, t, tuple(value.shape), value.dtype)
        if t == "lse":
            assert bool((value[:, no_key] == -math.inf).all()), f"{label}: lse of a row that sees no key is not -inf"
            assert bool(torch.isfinite(value[:, ~no_key]).all()), f"{label}: non-finite lse"
        else:
            assert bool(torch.isfinite(value).all()), f"{label}: non-finite {t}"
    for t in ("out", "dq"):
        if t in got:
            rows = got[t][no_key]
            assert torch.equal(rows, torch.zeros_like(rows)), f"{label}: {t} of a row that sees no key is not 0"
    if form == "rowmap":
        touched = torch.zeros(dst_init[0].shape[0], dtype=torch.bool, device=q.device)
        touched[row_map.long()] = True
        for t, init in (("dk", dst_init[0]), ("dv", dst_init[1])):
            if t in got:
                kept, want = got[t][~touched], init[~touched]
                assert torch.equal(kept.view(_BITS[kept.dtype]), want.view(_BITS[want.dtype])), (
                    f"{label}: an untouched {t} destination row changed"
                )
    else:
        unseen = unseen_keys(x)
        for t in ("dk", "dv"):
            if t in got:
                rows = got[t][unseen]
                assert torch.equal(rows, torch.zeros_like(rows)), f"{label}: {t} of a key no query sees is not 0"


def judged(
    x: Mapping,
    runs: Mapping[str, tuple[str, Mapping[str, torch.Tensor]]],
    dst_init: tuple[torch.Tensor, torch.Tensor] | None = None,
    row_map: torch.Tensor | None = None,
    scale: float = SCALE,
) -> tuple[dict, dict, dict]:
    """``(outputs, oracle, naive)`` of one draw for :func:`judge_draws`: the calls ``runs`` (label -> ``(form,
    outputs)``) on the inputs ``x``, each held to :func:`assert_contract`, beside the FP64 oracle and the naive tier
    of ``x`` (in the row-map form, of the destinations ``dst_init`` through ``row_map``).  With several calls each
    output is named ``"<label> <output>"``."""
    ref = outputs(x, torch.float64, scale)
    low = outputs(x, NAIVE_DTYPE, scale)
    found, oracle, naive = {}, {}, {}
    for label, (form, got) in runs.items():
        assert_contract(label, x, form, got, dst_init, row_map)
        want, tier = ref, low
        if form == "rowmap":
            want, tier = dict(ref), dict(low)
            for t, init in (("dk", dst_init[0]), ("dv", dst_init[1])):
                want[t] = row_map_adds(init, row_map, ref[t], torch.float64)
                tier[t] = row_map_adds(init, row_map, low[t], NAIVE_DTYPE)
        prefix = f"{label} " if len(runs) > 1 else ""
        for t, value in got.items():
            found[prefix + t], oracle[prefix + t], naive[prefix + t] = value, want[t], tier[t]
    return found, oracle, naive


def judge_draws(record_property: Callable, case: Callable[[int], tuple[dict, dict, dict]], *, seed: int = 0) -> None:
    """Judge one case with :func:`judge_case` over its draws: ``case(s)`` makes the draw's inputs from the seed ``s``,
    launches the package and returns :func:`judged` of its calls.  Every verdict is recorded (``TIER`` lines), and
    any contract or tier failure fails the test with the judge's numbers."""
    held: dict[int, dict] = {}

    def run(j: int, s: int):
        found, oracle, naive = case(s)
        held.clear()
        held[j] = naive
        return found, oracle

    judge, failures = judge_case(run, lambda j, tier: held[j], seed=seed)
    judge.record(record_property)
    assert not failures, "\n".join(failures)
