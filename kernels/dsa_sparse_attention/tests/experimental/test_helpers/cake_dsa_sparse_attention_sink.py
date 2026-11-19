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

The trainable per-head sink of the NoPE form in both packages' suites: the acceptance bounds beside the judge (one
draw, a call with the sink and the same call without it), the cases of the flat entries, and the packed (varlen)
inputs of the sink's model -- each document's key rows its own token rows, then its compressed rows -- with the
routes of the packed entries.
"""

import math
from dataclasses import dataclass

import torch

from tests.experimental.test_helpers import cake_dsa_sparse_attention_naive as nv
from tests.experimental.test_helpers import (
    cake_dsa_sparse_attention_nondet_train_reference as nope_ref,
)

NUM_HEADS = nope_ref.NUM_HEADS
D_LATENT = nope_ref.D_LATENT
SINK_SCALE = nope_ref.SINK_SCALE
# The acceptance bounds of the sink (beside the judge), per draw.  out: e, its rel-L2 over the rel-L2 of the FP64
# reference rounded to BF16, at most SINK_REL_L2_FACTOR x the e of the same call without a sink (the issue's
# definition: no BF16 output is closer to the reference than its rounding, and the sink moves that floor -- without a
# sink, a row with one valid slot is a row of kv exactly).  dq and the accumulator's increment: rel-L2 at most
# SINK_REL_L2_FACTOR x the same call's without a sink.  Every finite lse within SINK_LSE_FACTOR x the largest lse error
# without a sink plus SINK_LSE_REL x |lse_ref|; dsink within SINK_DSINK_REL_L2 rel-L2 and every head within
# SINK_DSINK_HEAD x max |dsink_ref|.  Each call also reports the raw out ratio (rel-L2 with the sink over rel-L2
# without it) and the same ratio over the rows with more than SINK_MANY_KEYS valid slots.
SINK_REL_L2_FACTOR = 1.1
SINK_MANY_KEYS = 16
SINK_LSE_FACTOR = 4.0
SINK_LSE_REL = 1e-6
SINK_DSINK_REL_L2 = 5e-3
SINK_DSINK_HEAD = 5e-3

# name -> (T, index row width, make_sink_inputs keywords, key_passes): the widths 640 and 128 of the model's layers
# at T 1,024 and 4,096, and at T 1,024 width 640 the variants of the acceptance: q x 5, a head-major dout, the
# all-compressed layout (ratio 1), and one and four key passes.  Every case has one-token documents, holes in the
# middle of the rows, a row without a valid slot and a destination map with repeated rows.
SINK_CASES = {
    "w640_t1024": (1024, 640, {}, None),
    "w640_t4096": (4096, 640, {}, None),
    "w128_t1024": (1024, 128, {}, None),
    "w128_t4096": (4096, 128, {}, None),
    "w640_t1024_q_x5": (1024, 640, {"q_scale": 5.0}, None),
    "w640_t1024_head_major_dout": (1024, 640, {"dout_layout": "head_major"}, None),
    "w640_t1024_ratio1": (1024, 640, {"ratio": 1}, None),
    "w640_t1024_key_passes1": (1024, 640, {}, 1),
    "w640_t1024_key_passes4": (1024, 640, {}, 4),
}

# name -> (T, index row width) of the packed cases, and the routes of the packed entries: ``mask`` (the forward
# converts the rows itself, position-preserving), ``index_prep`` (the index-preparation stage compacts them first) and
# ``kernel`` (the forward converts and compacts them itself; explicit only).
SINK_VARLEN_CASES = {
    "w640_t1024": (1024, 640),
    "w640_t4096": (4096, 640),
    "w128_t1024": (1024, 128),
    "w128_t4096": (4096, 128),
}
SINK_VARLEN_ROUTES = ("mask", "index_prep", "kernel")
# keys of the zero-length query segment (a document with key rows and no query)
SINK_ZERO_Q_KEYS = 5


def sink_partial_rows(num_queries: int) -> int:
    """Rows of the sink-gradient stage's partials: one per 64-token chunk."""
    return -(-int(num_queries) // 64)


def sink_problem(inp, *, sink=True, indices=None) -> nv.Problem:
    """The judged problem of a sink draw: the draw's global rows (``indices`` when given) at the sink scale, with its
    sink (``sink=False``: the same problem without one)."""
    return nv.Problem(
        q_latent=inp.q,
        kv_latent=inp.kv,
        indices=inp.indices if indices is None else indices,
        dout=inp.dout,
        scale=SINK_SCALE,
        sink=inp.sink if sink else None,
    )


def rel_l2(value: torch.Tensor, ref: torch.Tensor) -> float:
    """``||value - ref|| / ||ref||`` in FP64 over the elements where ``ref`` is finite."""
    ref = ref.double()
    keep = torch.isfinite(ref)
    ref = torch.where(keep, ref, 0.0)
    diff = torch.where(keep, value.double() - ref, 0.0)
    return float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref))


def max_abs_error(value: torch.Tensor, ref: torch.Tensor) -> float:
    """The largest ``|value - ref|`` in FP64 over the elements where ``ref`` is finite."""
    ref = ref.double()
    keep = torch.isfinite(ref)
    return float(
        torch.where(
            keep, (value.double() - torch.where(keep, ref, 0.0)).abs(), 0.0
        ).max()
    )


def sink_out_ratios(found, oracle, form: str, many_keys: torch.Tensor) -> dict:
    """The ``out`` ratios of one call with the sink to the same call without it (outputs ``"sink <form> out"`` and
    ``"no sink <form> out"``): ``raw``, of their rel-L2 errors; ``floor``, of the rel-L2 errors of their FP64
    references rounded to BF16; ``e = raw / floor``, of their e (a call's rel-L2 over its floor), the bounded one;
    and ``many_keys``, the raw ratio over the rows ``many_keys`` selects (``[T]``, more than :data:`SINK_MANY_KEYS`
    valid slots)."""
    errors = []
    for side in ("sink", "no sink"):
        got, ref = found[f"{side} {form} out"], oracle[f"{side} {form} out"]
        errors.append(
            (
                rel_l2(got, ref),
                rel_l2(ref.to(torch.bfloat16), ref),
                rel_l2(got[many_keys], ref[many_keys]),
            )
        )
    (raw_s, floor_s, many_s), (raw_n, floor_n, many_n) = errors
    raw, floor = raw_s / raw_n, floor_s / floor_n
    return dict(raw=raw, floor=floor, e=raw / floor, many_keys=many_s / many_n)


def assert_sink_bounds(found, oracle, out_ratios: dict) -> None:
    """The acceptance bounds of one draw beside the judge, for each form of ``out_ratios`` (form ->
    :func:`sink_out_ratios`; outputs named ``"sink <form> <name>"`` with the sink and ``"no sink <form> <name>"`` for
    the same call without it): the ``e`` ratio of ``out`` at most :data:`SINK_REL_L2_FACTOR`; the rel-L2 of ``dq``
    and the accumulator's increment at most :data:`SINK_REL_L2_FACTOR` x the call's without a sink; every finite
    ``lse`` within :data:`SINK_LSE_FACTOR` x the largest ``lse`` error without a sink plus :data:`SINK_LSE_REL` x
    ``|lse_ref|``, and ``-inf`` exactly where the reference is; ``dsink`` within :data:`SINK_DSINK_REL_L2` rel-L2 and
    every head within :data:`SINK_DSINK_HEAD` x ``max |dsink_ref|``; on the dominant head (about 100 above the
    scores) ``lse`` is the sink to FP32 rounding on every row."""
    dominant = max(nope_ref.SINK_SPECIAL_HEADS, key=nope_ref.SINK_SPECIAL_HEADS.get)
    top = nope_ref.SINK_SPECIAL_HEADS[dominant]
    half_ulp = 0.5 * torch.finfo(torch.float32).eps * 2.0 ** math.floor(math.log2(top))
    for form, ratios in out_ratios.items():
        with_sink, without = f"sink {form} ", f"no sink {form} "
        assert ratios["e"] <= SINK_REL_L2_FACTOR, (
            f"{form} out: e (rel-L2 over its BF16 floor) {ratios['e']:.4f} x the call's without a sink "
            f"(raw {ratios['raw']:.4f} x, floor {ratios['floor']:.4f} x; at most {SINK_REL_L2_FACTOR} x)"
        )
        for name in ("dq", "dkv_acc"):
            base = rel_l2(found[without + name], oracle[without + name])
            got = rel_l2(found[with_sink + name], oracle[with_sink + name])
            assert got <= SINK_REL_L2_FACTOR * base, (
                f"{form} {name}: rel-L2 {got:.4e} with the sink, {base:.4e} without "
                f"(at most {SINK_REL_L2_FACTOR} x)"
            )
        base_lse = max_abs_error(found[without + "lse"], oracle[without + "lse"])
        lse = found[with_sink + "lse"].double()
        ref = oracle[with_sink + "lse"].double()
        finite = torch.isfinite(ref)
        assert torch.equal(torch.isfinite(lse), finite), (
            f"{form} lse: non-finite where the reference is finite"
        )
        assert torch.equal(lse[~finite], ref[~finite]), (
            f"{form} lse: not -inf where the reference is"
        )
        ref0 = torch.where(finite, ref, 0.0)
        err = torch.where(finite, (lse - ref0).abs(), 0.0)
        excess = float(
            (err - SINK_LSE_FACTOR * base_lse - SINK_LSE_REL * ref0.abs()).max()
        )
        assert excess <= 0.0, (
            f"{form} lse: an error exceeds {SINK_LSE_FACTOR} x {base_lse:.3e} + {SINK_LSE_REL} |lse_ref| "
            f"by {excess:.3e}"
        )
        worst = float((lse[:, dominant] - ref[:, dominant]).abs().max())
        assert worst <= half_ulp, (
            f"{form} lse of the dominant head: {worst:.3e} from the sink"
        )
        dsink = found[with_sink + "dsink"].double()
        want = oracle[with_sink + "dsink"].double()
        rel = float(
            torch.linalg.vector_norm(dsink - want) / torch.linalg.vector_norm(want)
        )
        assert rel <= SINK_DSINK_REL_L2, f"{form} dsink: rel-L2 {rel:.3e}"
        head = float((dsink - want).abs().max())
        limit = SINK_DSINK_HEAD * float(want.abs().max())
        assert head <= limit, f"{form} dsink: a head's error {head:.3e} > {limit:.3e}"


def make_sink_window_inputs(
    total_q: int, *, seed: int, device="cuda"
) -> "nope_ref.SinkInputs":
    """One document of ``total_q`` tokens (any count, one included) on the window-only layout (width 128, ``ratio``
    0) of the sink's model, without planted holes or emptied rows: the short calls :func:`nope_ref.make_sink_inputs`
    (more than 40 tokens) does not make."""
    gen = torch.Generator().manual_seed(seed)
    rows, num_kv = nope_ref.sink_index_rows([total_q], 0, gen)
    dgen = torch.Generator(device=device).manual_seed(seed + 1)
    q = torch.randn(total_q, NUM_HEADS, D_LATENT, device=device, generator=dgen).to(
        torch.bfloat16
    )
    return nope_ref.SinkInputs(
        q=q,
        kv=nope_ref._randn_bf16((num_kv, D_LATENT), dgen, device),
        indices=rows.to(device).contiguous(),
        sink=nope_ref.sink_vector(gen, device),
        dout=nope_ref._randn_dout(total_q, "contiguous", dgen, device),
        ratio=0,
        empty_row=-1,
    )


@dataclass
class SinkVarlenInputs:
    """One packed NoPE problem with a sink: BF16 ``q [T, 64, 512]``, ``kv [S, 512]`` and ``dout``, the per-document
    int32 ``local [T, width]`` rows with the prefix sums ``cu_seqlens_q`` / ``cu_seqlens_k`` (a zero-length query
    segment among the documents), their global rows ``indices`` (the judged problem's), the FP32 ``sink [64]`` and the
    row without a valid slot ``empty_row``."""

    q: torch.Tensor
    kv: torch.Tensor
    local: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    indices: torch.Tensor
    sink: torch.Tensor
    dout: torch.Tensor
    empty_row: int

    @property
    def total_q(self) -> int:
        return int(self.q.shape[0])

    @property
    def total_k(self) -> int:
        return int(self.kv.shape[0])

    @property
    def num_segments(self) -> int:
        return int(self.cu_seqlens_k.numel()) - 1


def make_sink_varlen_inputs(
    total_q: int, width: int, *, seed: int, device="cuda"
) -> SinkVarlenInputs:
    """The packed inputs of the sink's model: the documents of :func:`nope_ref.sink_doc_lens` (two one-token
    documents, a short and a long one) with a zero-length query segment of :data:`SINK_ZERO_Q_KEYS` keys after the
    second; each document's key rows are its own token rows, then its compressed rows (``ratio`` 2 for width 640, 0
    for width 128), and its index rows the per-document rows of :func:`nope_ref.sink_index_rows`.  A hole is planted
    in the middle of every row with three valid slots or more, and the row ``T // 2`` is emptied, as in
    :func:`nope_ref.make_sink_inputs`.  ``indices`` are the global rows (``-1`` for every invalid slot)."""
    ratio = 2 if width == nope_ref.SINK_WINDOW + nope_ref.SINK_TOPK else 0
    if (nope_ref.SINK_WINDOW + (nope_ref.SINK_TOPK if ratio else 0)) != width:
        raise ValueError(f"width {width} is not a layer width of the sink's model")
    gen = torch.Generator().manual_seed(seed)
    lens = nope_ref.sink_doc_lens(total_q)
    docs_q = lens[:2] + [0] + lens[2:]
    rows, seq_k = [], []
    for n in docs_q:
        if n == 0:
            seq_k.append(SINK_ZERO_Q_KEYS)
            continue
        r, s = nope_ref.sink_index_rows([n], ratio, gen)
        rows.append(r)
        seq_k.append(s)
    local = torch.cat(rows)
    valid = local >= 0
    count = valid.sum(1)
    slot = torch.arange(width)
    first = torch.where(valid, slot, width).min(1).values
    last = torch.where(valid, slot, -1).max(1).values
    mid = (first + last) // 2
    plant = (count >= 3) & valid[torch.arange(total_q), mid.clamp(0, width - 1)]
    local[torch.arange(total_q)[plant], mid[plant]] = -1
    empty_row = total_q // 2
    local[empty_row] = -1
    cu_q = torch.tensor([0] + docs_q).cumsum(0)
    cu_k = torch.tensor([0] + seq_k).cumsum(0)
    doc = torch.repeat_interleave(torch.arange(len(docs_q)), torch.tensor(docs_q))
    base, limit = cu_k[doc][:, None], torch.tensor(seq_k)[doc][:, None]
    glob = torch.where((local >= 0) & (local < limit), local + base, -1)
    dgen = torch.Generator(device=device).manual_seed(seed + 1)
    num_kv = int(cu_k[-1])
    q = torch.randn(total_q, NUM_HEADS, D_LATENT, device=device, generator=dgen).to(
        torch.bfloat16
    )
    kv = nope_ref._randn_bf16((num_kv, D_LATENT), dgen, device)
    dout = nope_ref._randn_dout(total_q, "contiguous", dgen, device)
    return SinkVarlenInputs(
        q=q,
        kv=kv,
        local=local.int().to(device).contiguous(),
        cu_seqlens_q=cu_q.int().to(device),
        cu_seqlens_k=cu_k.int().to(device),
        indices=glob.int().to(device).contiguous(),
        sink=nope_ref.sink_vector(gen, device),
        dout=dout,
        empty_row=empty_row,
    )
