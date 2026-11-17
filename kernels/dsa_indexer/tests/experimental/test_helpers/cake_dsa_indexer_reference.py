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

Independent reference for the DSA indexer top-k contract: input generators, an FP64 / FP32 scorer, the
exact global selection with the contract's tie rule, the scores of given keys in a chosen dtype and the
structure checks of a result.  Shared by
``tests/experimental/test_helpers/cake_dsa_indexer_naive.py`` (the training-accuracy judge of the operator tests),
the operator suites under ``tests/experimental/``, ``benchmarks/bench_cake_dsa_indexer_det_train.py`` and
``benchmarks/bench_cake_dsa_indexer_nondet_train.py``.

Nothing here knows how the kernels partition queries or keys.

Input generators.  ``make_integer_inputs`` draws small integers: ``q`` / ``k`` uniform in ``[-2, 2]`` (BF16), ``w``
uniform in ``[-2, 2]`` (FP32) and ``softmax_scale = 1/16`` (optionally ``peaked``); every head logit is then an
integer and every score a multiple of ``1/16`` far below ``2**24`` of them, so every score is exact in FP32 under any
summation order (:func:`exact_inputs`).  ``make_integer_fp8_inputs`` holds the same integers as float8_e4m3fn with
per-key scales ``(1 + m / 2**21) * 2**e`` (``m`` odd, ``e`` in -3..3: never a power of two, below BF16's precision):
the head sum stays an exact integer and the one scale multiply rounds once.  ``make_integer_sink_inputs`` is an
integer analogue of the attention-sink + local-band statistics.  ``make_random_inputs`` / ``make_fp8_inputs`` /
``make_sink_inputs`` draw the Gaussian (and amax-quantized e4m3) inputs of a serving stack, on which the FP32
summation order rounds; ``make_sink_inputs`` holds the recorded attention-sink + local-band statistics
(key 0 boosted along a shared direction, log-spaced sinusoid position features shared by every query and its nearby
keys, positive ``w``), the distribution on which the sampled first threshold of the generated program is
stress-tested.  The packed, tile-boundary and non-finite generators build on the integer inputs.

Scoring.  For every visible (query, key) pair the FP64 score is treated as
exact and an independent FP32 scorer that follows the generated program's
documented reduction model is evaluated beside it.  With
``order="sequential"`` the dot product ``x_h`` is accumulated with elementwise
operations ``(((q0 k0 + q1 k1) + q2 k2) + ...)``: the products of two BF16
values are exact in FP32 and FP64, so each addition is the only rounding and
the bits do not depend on how the operands were chunked.  The head reduction
is the program's: ``r_h = fl(x_h + |x_h|)`` (exactly ``2 relu(x_h)``; ``+0.0``
for every ``x_h <= 0`` including ``-0.0``), ``w'_h = fl(w_h * softmax_scale)``,
four interleaved FMA chains ``c_m = fma(r_h, w'_h, c_m)`` over ``h = m, m + 4,
..., m + 28`` starting from ``+0.0`` (the FMA is emulated as the exact FP64
product-sum rounded once to FP32), then ``s = fl(fl(fl(c_0 + c_2) + fl(c_1 +
c_3)) * 0.5)``.  Because every chain starts from ``+0.0`` and
``fma(+0, w', +0) = +0`` for either sign of ``w'``, a row whose head terms are
all zero scores ``+0.0``: ``-0.0`` cannot arise from this reduction (the
program's ``zero_sign_policy = "positive_accumulator"``).  ``order="matmul"``
uses ``torch.matmul`` (TF32 disabled for FP32) for the dot products: much
faster, but its bits depend on the operand shapes; ``order="auto"`` picks ``sequential`` up to
``SEQUENTIAL_PAIR_LIMIT`` visible pairs.

FP8 form.  Inputs with ``k_scale`` hold float8_e4m3fn
``q`` / ``k`` and score ``s = k_scale[j] * sum_h w_h relu(x_h)`` (``softmax_scale
= 1``).  The product of two e4m3 values is exact and every partial sum of
128 of them is a multiple of ``2**-18`` below ``2**25``, so the FP64 logits
are exact in any order (``order="auto"`` picks ``matmul``); the head sum is
FP64 and ``k_scale[j]`` multiplies it once.  The FP32 scorer rounds that head
sum to FP32 and multiplies by ``k_scale[j]`` (one rounding each); it is not a
model of a program's reduction.

Selection.  :func:`select_reference` ranks every visible key by its FP64 score.  :func:`exact_selection` takes inputs
whose scores are exact in FP32 (:func:`exact_inputs`) and ranks by the FP32 score the contract defines (the exact
score, or for the FP8 form the exact head sum times ``k_scale`` rounded once): the top ``min(top_k, visible)`` keys
by score descending, then by segment-local id descending (``+0 == -0``), in ascending id order, then ``(-1, -inf)``
padding.  On such inputs that selection is the one result every program must return, so the ids and the counts
compare exactly.  :func:`scores_at` scores given keys in FP64 (the oracle) or in BF16 (the naive tier: the einsum in
BF16, the FP32 ``w`` and ``k_scale`` as stored, the result rounded to BF16).

Structure.  :func:`structure_failures` checks shapes, dtypes, padding, uniqueness, ascending order, visibility and
segment membership exactly.  :func:`compare_selection` reports, without a bound, how many rows of a result hold the
reference's ids (benchmarks); on Gaussian inputs a row may differ where the FP32 rounding reorders keys whose FP64
scores are near-tied, so the count is information there, not a verdict.
"""

from __future__ import annotations

import contextlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

NUM_HEADS = 32
HEAD_DIM = 128
DEFAULT_TOP_K = 2048
DEFAULT_SCALE = HEAD_DIM**-0.5
EXACT_SCALE = 2.0**-3
TRAINER_KEY_ROW_STRIDE = 704
PAD_ID = -1
PAD_SCORE = float("-inf")
SEQUENTIAL_PAIR_LIMIT = 1 << 25
CHUNK_BYTES = 1 << 30
INT64_MAX = (1 << 63) - 1
_GEN_ROWS = 1 << 17
# the integer inputs: values uniform in [-INTEGER_RANGE, INTEGER_RANGE], softmax_scale INTEGER_SCALE (a power of two)
INTEGER_RANGE = 2
INTEGER_SCALE = 2.0**-4
# exact_inputs: every partial sum of a score, counted in its smallest step, stays below this (FP32 holds 2**24)
EXACT_SUM_LIMIT = 1 << 23


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def cumulative(lengths) -> list[int]:
    out = [0]
    for n in lengths:
        out.append(out[-1] + int(n))
    return out


def effective_offsets(seg_q_len, seg_k_len, q_causal_offsets, ratio: int) -> list[int]:
    """Per-segment offset: the supplied value, else ``Lk - Lq`` for ``ratio == 1``, else ``0``."""
    if q_causal_offsets is not None:
        if len(q_causal_offsets) != len(seg_q_len):
            raise ValueError("one offset per segment is required")
        return [int(o) for o in q_causal_offsets]
    if int(ratio) == 1:
        return [int(lk) - int(lq) for lq, lk in zip(seg_q_len, seg_k_len, strict=True)]
    return [0 for _ in seg_q_len]


def visible_scalar(offset: int, position: int, ratio: int, num_keys: int) -> int:
    """Visible keys of one query by the contract's rule (Python ``//`` is a true floor)."""
    return max(0, min(int(num_keys), (int(offset) + int(position) + 1) // int(ratio)))


@dataclass
class IndexerInputs:
    """Operator inputs plus the host segment metadata they were built from."""

    q: torch.Tensor
    k: torch.Tensor
    w: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    top_k: int
    softmax_scale: float
    seg_q_len: list[int]
    seg_k_len: list[int]
    q_causal_offsets: Optional[torch.Tensor] = None
    ratio: int = 1
    label: str = ""
    # the FP8 form's per-key FP32 scales [Tkv] (float8_e4m3fn q / k); None for BF16 inputs
    k_scale: Optional[torch.Tensor] = None

    @property
    def fp8(self) -> bool:
        return self.k_scale is not None

    @property
    def num_queries(self) -> int:
        return int(self.q.shape[0])

    @property
    def num_keys(self) -> int:
        return int(self.k.shape[0])

    @property
    def num_segments(self) -> int:
        return len(self.seg_q_len)

    @property
    def device(self) -> torch.device:
        return self.q.device

    def offsets(self) -> list[int]:
        explicit = (
            None if self.q_causal_offsets is None else self.q_causal_offsets.tolist()
        )
        return effective_offsets(self.seg_q_len, self.seg_k_len, explicit, self.ratio)

    def kwargs(self) -> dict[str, Any]:
        if self.fp8:
            # the FP8 form takes softmax_scale=None (every scale but k_scale lives in w)
            return dict(
                top_k=self.top_k,
                softmax_scale=None,
                q_causal_offsets=self.q_causal_offsets,
                ratio=self.ratio,
                k_scale=self.k_scale,
            )
        return dict(
            top_k=self.top_k,
            softmax_scale=self.softmax_scale,
            q_causal_offsets=self.q_causal_offsets,
            ratio=self.ratio,
        )

    def query_starts(self) -> list[int]:
        return cumulative(self.seg_q_len)

    def key_starts(self) -> list[int]:
        return cumulative(self.seg_k_len)


def visible_rows(inputs: IndexerInputs, device=None) -> torch.Tensor:
    """int64 ``[T]`` visible-key count per query row from the host metadata only."""
    device = inputs.device if device is None else device
    out = torch.zeros(inputs.num_queries, dtype=torch.int64, device=device)
    starts = inputs.query_starts()
    for s, off in enumerate(inputs.offsets()):
        lq, lk = inputs.seg_q_len[s], inputs.seg_k_len[s]
        if lq == 0:
            continue
        u = torch.arange(lq, dtype=torch.int64, device=device)
        vis = torch.div(u + (off + 1), inputs.ratio, rounding_mode="floor").clamp_(
            min=0, max=lk
        )
        out[starts[s] : starts[s] + lq] = vis
    return out


def segment_of_rows(inputs: IndexerInputs, device=None) -> torch.Tensor:
    """int64 ``[T]`` segment index of every query row."""
    device = inputs.device if device is None else device
    out = torch.empty(inputs.num_queries, dtype=torch.int64, device=device)
    starts = inputs.query_starts()
    for s, lq in enumerate(inputs.seg_q_len):
        if lq:
            out[starts[s] : starts[s] + lq] = s
    return out


def build_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    seg_q_len,
    seg_k_len,
    *,
    top_k: int,
    softmax_scale: float,
    q_causal_offsets=None,
    ratio: int = 1,
    label: str = "",
    k_scale: Optional[torch.Tensor] = None,
) -> IndexerInputs:
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    if len(seg_q_len) != len(seg_k_len):
        raise ValueError("query and key segment lists must have the same length")
    if sum(seg_q_len) != q.shape[0] or sum(seg_k_len) != k.shape[0]:
        raise ValueError("segment lengths must sum to the tensor row counts")
    if k_scale is not None and float(softmax_scale) != 1.0:
        # the FP8 form takes softmax_scale=None and the references would still apply it
        raise ValueError(
            "the FP8 form folds softmax_scale into w: pass softmax_scale=1.0 with k_scale"
        )
    device = q.device
    offsets = None
    if q_causal_offsets is not None:
        offsets = torch.tensor(
            [int(o) for o in q_causal_offsets], dtype=torch.int64, device=device
        )
    return IndexerInputs(
        q=q,
        k=k,
        w=w,
        cu_seqlens_q=torch.tensor(
            cumulative(seg_q_len), dtype=torch.int32, device=device
        ),
        cu_seqlens_k=torch.tensor(
            cumulative(seg_k_len), dtype=torch.int32, device=device
        ),
        top_k=int(top_k),
        softmax_scale=float(softmax_scale),
        seg_q_len=seg_q_len,
        seg_k_len=seg_k_len,
        q_causal_offsets=offsets,
        ratio=int(ratio),
        label=label,
        k_scale=k_scale,
    )


def _gaussian_bf16(shape, gen, device) -> torch.Tensor:
    out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    for r0 in range(0, shape[0], _GEN_ROWS):
        r1 = min(shape[0], r0 + _GEN_ROWS)
        out[r0:r1].copy_(
            torch.randn(
                (r1 - r0, *shape[1:]), generator=gen, device=device, dtype=torch.float32
            )
        )
    return out


def make_random_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
    device="cuda",
    peaked: bool = False,
    ratio: int = 1,
    q_causal_offsets=None,
    softmax_scale: Optional[float] = None,
    k_row_stride: Optional[int] = None,
    hot_keys_per_segment: int = 16,
    label: str = "",
) -> IndexerInputs:
    """Seeded iid Gaussian BF16 ``q`` / ``k`` (unit RMS) and signed FP32 ``w ~ N(0, 1) * H**-0.5``.

    ``peaked=True`` adds a shared +-1 direction to every query and twice that
    direction to a few keys per segment, so those keys dominate most rows
    (heavy-tailed scores).  ``k_row_stride`` returns ``k`` as the leading
    ``[:, :128]`` view of a wider packed ``[Tkv, k_row_stride]`` tensor.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    T, Tkv = sum(seg_q_len), sum(seg_k_len)
    q = _gaussian_bf16((T, NUM_HEADS, HEAD_DIM), gen, device)
    w = torch.randn(
        (T, NUM_HEADS), generator=gen, device=device, dtype=torch.float32
    ) * (NUM_HEADS**-0.5)
    if k_row_stride is None:
        k = _gaussian_bf16((Tkv, HEAD_DIM), gen, device)
    else:
        if k_row_stride < HEAD_DIM:
            raise ValueError("k_row_stride must be >= 128")
        k = _gaussian_bf16((Tkv, int(k_row_stride)), gen, device)[:, :HEAD_DIM]
    if peaked:
        direction = (
            torch.randint(0, 2, (HEAD_DIM,), generator=gen, device=device) * 2 - 1
        ).to(torch.bfloat16)
        q.add_(direction.view(1, 1, HEAD_DIM))
        k_start = 0
        for lk in seg_k_len:
            n_hot = min(int(hot_keys_per_segment), lk)
            if n_hot:
                hot = torch.randperm(lk, generator=gen, device=device)[:n_hot] + k_start
                k[hot] = (
                    k[hot].float() + 2.0 * direction.float().view(1, HEAD_DIM)
                ).to(torch.bfloat16)
            k_start += lk
    return build_inputs(
        q,
        k,
        w,
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=DEFAULT_SCALE if softmax_scale is None else softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        label=label,
    )


def _integers(shape, gen, device, low: int, high: int) -> torch.Tensor:
    """Uniform integers in ``[low, high]`` as FP32, drawn in row blocks (bounded int64 scratch)."""
    out = torch.empty(shape, dtype=torch.float32, device=device)
    rows = max(1, (1 << 24) // max(1, math.prod(shape[1:])))
    for r0 in range(0, shape[0], rows):
        r1 = min(shape[0], r0 + rows)
        out[r0:r1].copy_(
            torch.randint(low, high + 1, (r1 - r0, *shape[1:]), generator=gen, device=device)
        )
    return out


def make_integer_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
    device="cuda",
    peaked: bool = False,
    ratio: int = 1,
    q_causal_offsets=None,
    k_row_stride: Optional[int] = None,
    hot_keys_per_segment: int = 16,
    label: str = "",
) -> IndexerInputs:
    """Seeded integer inputs whose every score is exact in FP32 under any summation order (:func:`exact_inputs`):
    ``q`` / ``k`` uniform in ``[-2, 2]`` (BF16), signed ``w`` uniform in ``[-2, 2]`` (FP32), ``softmax_scale = 1/16``.
    Equal scores are common, so the id rule of the contract decides many rows.

    ``peaked=True`` adds a shared +-1 direction to every query and twice that direction to a few keys per segment, so
    those keys dominate most rows (heavy-tailed scores).  ``k_row_stride`` returns ``k`` as the leading ``[:, :128]``
    view of a wider packed ``[Tkv, k_row_stride]`` tensor.
    """
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    T, Tkv = sum(seg_q_len), sum(seg_k_len)
    r = INTEGER_RANGE
    q = _integers((T, NUM_HEADS, HEAD_DIM), gen, device, -r, r)
    w = _integers((T, NUM_HEADS), gen, device, -r, r)
    width = HEAD_DIM if k_row_stride is None else int(k_row_stride)
    if width < HEAD_DIM:
        raise ValueError("k_row_stride must be >= 128")
    k = _integers((Tkv, width), gen, device, -r, r)
    if peaked:
        direction = (
            torch.randint(0, 2, (HEAD_DIM,), generator=gen, device=device) * 2 - 1
        ).to(torch.float32)
        q.add_(direction.view(1, 1, HEAD_DIM))
        k_start = 0
        for lk in seg_k_len:
            n_hot = min(int(hot_keys_per_segment), lk)
            if n_hot:
                hot = torch.randperm(lk, generator=gen, device=device)[:n_hot] + k_start
                k[hot, :HEAD_DIM] += 2.0 * direction.view(1, HEAD_DIM)
            k_start += lk
    k = k.to(torch.bfloat16)
    return build_inputs(
        q.to(torch.bfloat16),
        k if k_row_stride is None else k[:, :HEAD_DIM],
        w,
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=INTEGER_SCALE,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        label=label,
    )


def integer_key_scales(n: int, gen, device) -> torch.Tensor:
    """Per-key FP8 scales ``(1 + m / 2**21) * 2**e``, ``m`` odd in ``[1, 2**21)``, ``e`` in -3..3: exact in FP32,
    never a power of two, and below BF16's precision, so two keys of equal integer head sums score differently and the
    one scale multiply of a score rounds."""
    m = torch.randint(0, 2**20, (n,), generator=gen, device=device) * 2 + 1
    e = torch.randint(-3, 4, (n,), generator=gen, device=device)
    return ((1 + m.double() / 2**21) * torch.pow(2.0, e.double())).float()


def make_integer_fp8_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
    device="cuda",
    ratio: int = 1,
    q_causal_offsets=None,
    k_row_stride: Optional[int] = None,
    label: str = "",
) -> IndexerInputs:
    """FP8 inputs exact up to the scale multiply: ``q`` / ``k`` uniform integers in ``[-2, 2]`` as float8_e4m3fn
    (exact), signed ``w`` uniform in ``[-2, 2]`` (FP32), ``softmax_scale = 1`` and per-key scales
    :func:`integer_key_scales`.  Every head sum is an exact integer; the score is that sum times ``k_scale[j]``
    rounded once (:func:`exact_selection`).  ``k_row_stride`` returns ``k`` as the leading ``[:, :128]`` view of a
    wider ``[Tkv, k_row_stride]`` e4m3 tensor."""
    base = make_integer_inputs(
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        seed=seed,
        device=device,
        ratio=ratio,
        q_causal_offsets=q_causal_offsets,
        label=label,
    )
    device = base.device
    gen = torch.Generator(device=device).manual_seed(int(seed) + 7919)
    width = HEAD_DIM if k_row_stride is None else int(k_row_stride)
    if width < HEAD_DIM:
        raise ValueError("k_row_stride must be >= 128")
    packed = torch.zeros((base.num_keys, width), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    packed[:, :HEAD_DIM] = base.k.to(torch.float32).to(torch.float8_e4m3fn)
    return build_inputs(
        base.q.to(torch.float32).to(torch.float8_e4m3fn),
        packed if k_row_stride is None else packed[:, :HEAD_DIM],
        base.w,
        base.seg_q_len,
        base.seg_k_len,
        top_k=top_k,
        softmax_scale=1.0,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        label=label,
        k_scale=integer_key_scales(base.num_keys, gen, device),
    )


E4M3_MAX = 448.0


def _quantize_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last dimension) amax scaling of FP32 ``x`` to float8_e4m3fn: ``(x8, scale)`` with ``x ~ x8 * scale``."""
    scale = (x.abs().amax(dim=-1) / E4M3_MAX).clamp_min(2.0**-60)
    x8 = (x / scale.unsqueeze(-1)).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return x8, scale


def make_fp8_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
    device="cuda",
    ratio: int = 1,
    q_causal_offsets=None,
    softmax_scale: float = DEFAULT_SCALE,
    k_row_stride: Optional[int] = None,
    label: str = "",
) -> IndexerInputs:
    """Seeded FP8 inputs: iid Gaussian ``q`` / ``k`` quantized to float8_e4m3fn with one amax scale per (query, head)
    and one per key row (``k_scale``), and signed ``w ~ N(0, 1) * H**-0.5`` carrying the query scale and
    ``softmax_scale`` (the deployed FP8 indexer's folding; the stored ``softmax_scale`` is 1.0).  ``k_row_stride``
    returns ``k`` as the leading ``[:, :128]`` view of a wider ``[Tkv, k_row_stride]`` e4m3 tensor."""
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    T, Tkv = sum(seg_q_len), sum(seg_k_len)
    q = torch.empty((T, NUM_HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
    q_scale = torch.empty((T, NUM_HEADS), dtype=torch.float32, device=device)
    for r0 in range(0, T, _GEN_ROWS):
        r1 = min(T, r0 + _GEN_ROWS)
        q[r0:r1], q_scale[r0:r1] = _quantize_e4m3(
            torch.randn(
                (r1 - r0, NUM_HEADS, HEAD_DIM),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
        )
    w = (
        torch.randn((T, NUM_HEADS), generator=gen, device=device, dtype=torch.float32)
        * (NUM_HEADS**-0.5)
        * q_scale
        * float(softmax_scale)
    )
    width = HEAD_DIM if k_row_stride is None else int(k_row_stride)
    if width < HEAD_DIM:
        raise ValueError("k_row_stride must be >= 128")
    packed = torch.zeros((Tkv, width), dtype=torch.uint8, device=device).view(
        torch.float8_e4m3fn
    )
    k_scale = torch.empty(Tkv, dtype=torch.float32, device=device)
    for r0 in range(0, Tkv, _GEN_ROWS):
        r1 = min(Tkv, r0 + _GEN_ROWS)
        packed[r0:r1, :HEAD_DIM], k_scale[r0:r1] = _quantize_e4m3(
            torch.randn(
                (r1 - r0, HEAD_DIM), generator=gen, device=device, dtype=torch.float32
            )
        )
    k = packed if k_row_stride is None else packed[:, :HEAD_DIM]
    return build_inputs(
        q,
        k,
        w,
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=1.0,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        label=label,
        k_scale=k_scale,
    )


def dequantized_bf16(inputs: IndexerInputs) -> IndexerInputs:
    """BF16 inputs of the same data as the FP8 ``inputs``: ``q`` exactly (every e4m3 value is a BF16 value), ``k``
    as ``bf16(k * k_scale)``, the same ``w`` and ``softmax_scale = 1``; for timing the BF16 form on the FP8 form's
    data (its scores are close to, not equal to, the FP8 form's)."""
    if not inputs.fp8:
        raise ValueError("dequantized_bf16 takes FP8 inputs")
    k = (inputs.k.to(torch.float32) * inputs.k_scale.unsqueeze(1)).to(torch.bfloat16)
    return IndexerInputs(
        q=inputs.q.to(torch.float32).to(torch.bfloat16),
        k=k,
        w=inputs.w,
        cu_seqlens_q=inputs.cu_seqlens_q,
        cu_seqlens_k=inputs.cu_seqlens_k,
        top_k=inputs.top_k,
        softmax_scale=1.0,
        seg_q_len=list(inputs.seg_q_len),
        seg_k_len=list(inputs.seg_k_len),
        q_causal_offsets=inputs.q_causal_offsets,
        ratio=inputs.ratio,
        label=f"{inputs.label}_bf16" if inputs.label else "bf16",
    )


SINK_NUM_FREQUENCIES = 32
SINK_SEED = 7


def make_sink_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = SINK_SEED,
    device="cuda",
    q_causal_offsets=None,
    noise: float = 0.5,
    local: float = 4.0,
    sink: float = 2.0,
    key0_boost: float = 8.0,
    n_freq: int = SINK_NUM_FREQUENCIES,
    label: str = "",
) -> IndexerInputs:
    """Attention-sink + local-band inputs (the GLM-5.2 layer-6 indexer statistics).

    The recorded generator, generalised to several segments: position features ``z(pos)`` are 32
    random-phase ``(cos, sin)`` pairs at the log-spaced frequencies ``2**-3 .. 2**-17`` over ``sqrt(32)``,
    zero-padded to 128; a unit direction ``s`` is shared by every query (``+ sink * s``) and boosted on key 0 of
    every segment (``+ key0_boost * s``); ``q = noise * N + local * z(q_pos) + sink * s``,
    ``k = noise * N + local * z(k_pos)``, ``w = |N| * H**-0.5`` (positive weights, unlike
    :func:`make_random_inputs`).  Query ``u`` of segment ``s`` sits at document position ``offset_s + u``
    (``offset_s = Lk - Lq`` by default, the context-parallel tail rule), key ``j`` at position ``j``, so the
    nearest keys of every query form a log-decaying band and key 0 is an attention sink.  ``ratio`` is 1.

    The draws follow one seeded generator in a fixed order (``phi``, ``s``, ``q`` noise, ``k`` noise, ``w``), so two
    calls with the same arguments on the same device give the same bits; CUDA Gaussian draws of large tensors
    depend on the device's SM count, so bitwise identity of the inputs holds within one GPU model only.
    """
    device = torch.device(device)
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    if len(seg_q_len) != len(seg_k_len):
        raise ValueError("query and key segment lists must have the same length")
    offsets = effective_offsets(seg_q_len, seg_k_len, q_causal_offsets, 1)
    q_pos = torch.cat(
        [
            torch.arange(off, off + lq, dtype=torch.int64)
            for off, lq in zip(offsets, seg_q_len, strict=True)
        ]
    ).to(device)
    k_pos = torch.cat([torch.arange(0, lk, dtype=torch.int64) for lk in seg_k_len]).to(
        device
    )
    T = int(q_pos.numel())
    gen = torch.Generator(device=device).manual_seed(int(seed))
    omega = torch.logspace(
        -3, -17, n_freq, base=2.0, device=device, dtype=torch.float64
    )
    phi = (
        torch.rand(n_freq, generator=gen, device=device, dtype=torch.float64)
        * 2
        * torch.pi
    )

    def z(pos):
        ang = pos.double()[:, None] * omega + phi
        feat = torch.cat((ang.cos(), ang.sin()), 1).float() / n_freq**0.5
        return torch.nn.functional.pad(feat, (0, HEAD_DIM - 2 * n_freq))

    s = torch.randn(HEAD_DIM, generator=gen, device=device)
    s = s / s.norm()
    q = (
        noise * torch.randn(T, NUM_HEADS, HEAD_DIM, generator=gen, device=device)
        + local * z(q_pos)[:, None, :]
        + sink * s
    )
    k = noise * torch.randn(
        k_pos.numel(), HEAD_DIM, generator=gen, device=device
    ) + local * z(k_pos)
    k = k + (k_pos == 0).float()[:, None] * key0_boost * s
    w = torch.randn(T, NUM_HEADS, generator=gen, device=device).abs() * NUM_HEADS**-0.5
    return build_inputs(
        q.bfloat16(),
        k.bfloat16(),
        w,
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=DEFAULT_SCALE,
        q_causal_offsets=q_causal_offsets,
        ratio=1,
        label=label,
    )


def make_integer_sink_inputs(
    seg_q_len,
    seg_k_len,
    *,
    top_k: int = DEFAULT_TOP_K,
    seed: int = SINK_SEED,
    device="cuda",
    q_causal_offsets=None,
    n_freq: int = SINK_NUM_FREQUENCIES,
    label: str = "",
) -> IndexerInputs:
    """Integer attention-sink + local-band inputs, exact in FP32 (:func:`exact_inputs`).

    Query ``u`` of a segment sits at document position ``offset + u`` (``offset`` the segment's causal offset,
    ``Lk - Lq`` by default), key ``j`` at position ``j``.  Dims 0..63 carry the band: ``round(2 cos)`` and
    ``round(2 sin)`` of ``pos * omega + phi`` for 32 random-phase frequencies ``2**-3 .. 2**-17`` (the position
    features of :func:`make_sink_inputs` quantised to ``-2..2``), identical for a query and the keys at its position,
    decaying with the log distance as the high frequencies decohere.  Dims 64..127 carry the sink: every query gets a
    shared ``+-1`` direction, key 0 of every segment twice that direction.  Both get uniform ``-1..1`` noise; ``w`` is
    1 or 2 (positive, like :func:`make_sink_inputs`).  With ``softmax_scale = 1/16`` every head logit is an integer of
    magnitude at most 2560.  ``ratio`` is 1."""
    device = torch.device(device)
    seg_q_len = [int(x) for x in seg_q_len]
    seg_k_len = [int(x) for x in seg_k_len]
    if len(seg_q_len) != len(seg_k_len):
        raise ValueError("query and key segment lists must have the same length")
    offsets = effective_offsets(seg_q_len, seg_k_len, q_causal_offsets, 1)
    q_pos = torch.cat(
        [
            torch.arange(off, off + lq, dtype=torch.int64)
            for off, lq in zip(offsets, seg_q_len, strict=True)
        ]
    ).to(device)
    k_pos = torch.cat([torch.arange(0, lk, dtype=torch.int64) for lk in seg_k_len]).to(
        device
    )
    T, Tkv = int(q_pos.numel()), int(k_pos.numel())
    gen = torch.Generator(device=device).manual_seed(int(seed))
    omega = torch.logspace(-3, -17, n_freq, base=2.0, device=device, dtype=torch.float64)
    phi = torch.rand(n_freq, generator=gen, device=device, dtype=torch.float64) * 2 * torch.pi
    band_dims = 2 * n_freq
    direction = (
        torch.randint(0, 2, (HEAD_DIM - band_dims,), generator=gen, device=device) * 2 - 1
    ).to(torch.float32)

    def band(pos):
        ang = pos.double()[:, None] * omega + phi
        return torch.round(2 * torch.cat((ang.cos(), ang.sin()), 1)).to(torch.float32)

    q = _integers((T, NUM_HEADS, HEAD_DIM), gen, device, -1, 1)
    q[:, :, :band_dims] += band(q_pos)[:, None, :]
    q[:, :, band_dims:] += direction
    k = _integers((Tkv, HEAD_DIM), gen, device, -1, 1)
    k[:, :band_dims] += band(k_pos)
    k[:, band_dims:] += (k_pos == 0).to(torch.float32)[:, None] * 2 * direction
    w = _integers((T, NUM_HEADS), gen, device, 1, 2)
    return build_inputs(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        w,
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=INTEGER_SCALE,
        q_causal_offsets=q_causal_offsets,
        ratio=1,
        label=label,
    )


# -- exactly representable cases ----------------------------------------------------------------------------------
# Key ``j`` carries two integer levels ``up_j`` and ``down_j`` in ``[0, 255]``: ``up`` is spread over dims 0..7 and
# ``down`` over dims 64..71 as small integers (<= 32, exact in BF16).  Query head 0 holds the amplitude ``a`` on dims
# 0..7 and head 1 holds ``a`` on dims 64..71 (``a`` in {1, 2, 4}); every other head is zero.  With ``softmax_scale =
# 1/8`` the head-0 logit is ``a * up_j`` and the head-1 logit ``a * down_j`` (integers <= 1020), so with weights
# ``w_0``, ``w_1`` in {+-1} the score ``(w_0 * a * up_j + w_1 * a * down_j) / 8`` is exact in FP32 under every
# reduction order and every other head contributes a signed zero only.
#
# Every case has its historical small geometry (the default) and a scaled geometry (``num_queries`` queries at the
# tail of ``num_keys`` keys, one segment) built by the same rule, so a call runs long enough for benchmark timing to
# resolve small differences; the scaled ``distinct`` case separates up to 65535 keys exactly through the head-1
# weight ``2**-8`` (score ``a * (up_j + down_j / 256) / 8``, a multiple of ``2**-11`` below ``2**10``).

UP_DIMS = slice(0, 8)
DOWN_DIMS = slice(64, 72)


def _split_level(level: int, parts: int = 8) -> list[int]:
    base, rem = divmod(int(level), parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def mixed_weights() -> list[float]:
    w = [0.25 if h % 2 else -0.25 for h in range(NUM_HEADS)]
    w[0], w[1] = 1.0, -1.0
    return w


def negative_weights() -> list[float]:
    w = [-0.25] * NUM_HEADS
    w[0], w[1] = -1.0, -1.0
    return w


def distinct_weights() -> list[float]:
    """Mixed weights with head 1 at ``2**-8``: distinct ``(up, down)`` pairs give distinct exact scores."""
    w = mixed_weights()
    w[1] = 2.0**-8
    return w


def _scaled_geometry(num_queries, num_keys) -> Optional[tuple[int, int]]:
    """``(lq, lk)`` of the scaled single-segment geometry, or ``None`` for a case's historical geometry."""
    if num_queries is None and num_keys is None:
        return None
    if num_queries is None or num_keys is None:
        raise ValueError("num_queries and num_keys must be given together")
    lq, lk = int(num_queries), int(num_keys)
    if lq < 1 or lk < 8 or lq > lk:
        raise ValueError(
            "scaled exact cases need 1 <= num_queries <= num_keys and num_keys >= 8"
        )
    return lq, lk


def _scaled_top_k(lk: int) -> int:
    return min(DEFAULT_TOP_K, lk // 4)


def make_exact_inputs(
    segments,
    *,
    top_k: int,
    device="cuda",
    label: str,
    q_causal_offsets=None,
) -> IndexerInputs:
    """Build an exactly representable problem from per-segment specs.

    Each spec: ``lq`` (queries), ``up`` (list of key levels), optional ``down``
    (same length), optional ``weights`` (list of 32 floats or ``row ->
    list``), optional ``amp`` (list per row or ``row -> int``; default cycles
    1, 2, 4).
    """
    device = torch.device(device)
    seg_q_len = [int(s["lq"]) for s in segments]
    seg_k_len = [len(s["up"]) for s in segments]
    cu_q, cu_k = cumulative(seg_q_len), cumulative(seg_k_len)
    q = torch.zeros((cu_q[-1], NUM_HEADS, HEAD_DIM), dtype=torch.float32)
    w = torch.zeros((cu_q[-1], NUM_HEADS), dtype=torch.float32)
    k = torch.zeros((cu_k[-1], HEAD_DIM), dtype=torch.float32)
    for s, spec in enumerate(segments):
        up = [int(v) for v in spec["up"]]
        down = [int(v) for v in spec.get("down", [0] * len(up))]
        if len(down) != len(up):
            raise ValueError("up and down must have the same length")
        for j, (u_level, d_level) in enumerate(zip(up, down, strict=True)):
            if not (0 <= u_level <= 255 and 0 <= d_level <= 255):
                raise ValueError("levels must be integers in [0, 255]")
            k[cu_k[s] + j, UP_DIMS] = torch.tensor(
                _split_level(u_level), dtype=torch.float32
            )
            k[cu_k[s] + j, DOWN_DIMS] = torch.tensor(
                _split_level(d_level), dtype=torch.float32
            )
        weights = spec.get("weights", mixed_weights())
        amp = spec.get("amp")
        for u in range(seg_q_len[s]):
            t = cu_q[s] + u
            a = (
                amp(u)
                if callable(amp)
                else (amp[u] if amp is not None else (1, 2, 4)[u % 3])
            )
            q[t, 0, UP_DIMS] = float(a)
            q[t, 1, DOWN_DIMS] = float(a)
            row_w = weights(u) if callable(weights) else weights
            w[t] = torch.tensor(list(row_w), dtype=torch.float32)
    return build_inputs(
        q.to(torch.bfloat16).to(device),
        k.to(torch.bfloat16).to(device),
        w.to(device),
        seg_q_len,
        seg_k_len,
        top_k=top_k,
        softmax_scale=EXACT_SCALE,
        q_causal_offsets=q_causal_offsets,
        label=label,
    )


def exact_closed_form(inputs: IndexerInputs) -> Callable[[int, int], float]:
    """``score(row, local_key_id)`` of an exact problem as a Python float, from the input tensors alone."""
    q = inputs.q.float().cpu()
    k = inputs.k.float().cpu()
    w = inputs.w.cpu()
    up = k[:, UP_DIMS].sum(dim=1).tolist()
    down = k[:, DOWN_DIMS].sum(dim=1).tolist()
    amp = q[:, 0, 0].tolist()
    w0 = w[:, 0].tolist()
    w1 = w[:, 1].tolist()
    seg = segment_of_rows(inputs, device="cpu").tolist()
    k_starts = inputs.key_starts()
    scale = inputs.softmax_scale

    def score(row: int, local_id: int) -> float:
        j = k_starts[seg[row]] + int(local_id)
        a = amp[row]
        term0 = w0[row] * max(scale * a * up[j], 0.0)
        term1 = w1[row] * max(scale * a * down[j], 0.0)
        return term0 + term1

    return score


def _shuffled_levels(count: int, seed: int) -> list[int]:
    levels = list(range(1, count + 1))
    random.Random(seed).shuffle(levels)
    return levels


def _case_distinct(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    if geometry is None:
        return make_exact_inputs(
            [
                {"lq": 8, "up": _shuffled_levels(96, 1)},
                {"lq": 40, "up": _shuffled_levels(40, 2)},
            ],
            top_k=24,
            device=device,
            label="exact_distinct",
        )
    lq, lk = geometry
    if lk > 65535:
        raise ValueError("the scaled distinct case separates at most 65535 keys")
    # distinct (up, down) pairs: level = up * 256 + down
    levels = _shuffled_levels(lk, 1)
    return make_exact_inputs(
        [
            {
                "lq": lq,
                "up": [level // 256 for level in levels],
                "down": [level % 256 for level in levels],
                "weights": distinct_weights(),
            }
        ],
        top_k=_scaled_top_k(lk),
        device=device,
        label="exact_distinct",
    )


def _case_cutoff_ties(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    if geometry is None:
        up = [j // 3 + 1 for j in range(48)]
        return make_exact_inputs(
            [{"lq": 7, "up": up}, {"lq": 30, "up": up[:30]}],
            top_k=10,
            device=device,
            label="exact_cutoff_ties",
        )
    lq, lk = geometry
    group = -(-lk // 255)  # equal-level groups of this size keep every level <= 255
    top_k = _scaled_top_k(lk)
    if (lk - top_k) % group == 0:
        # the selection boundary of the row seeing every key must fall inside a tie group
        top_k += 1
    return make_exact_inputs(
        [{"lq": lq, "up": [j // group + 1 for j in range(lk)]}],
        top_k=top_k,
        device=device,
        label="exact_cutoff_ties",
    )


def _case_all_equal(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    if geometry is None:
        return make_exact_inputs(
            [{"lq": 6, "up": [5] * 50}, {"lq": 30, "up": [5] * 30}],
            top_k=16,
            device=device,
            label="exact_all_equal",
        )
    lq, lk = geometry
    return make_exact_inputs(
        [{"lq": lq, "up": [5] * lk}],
        top_k=_scaled_top_k(lk),
        device=device,
        label="exact_all_equal",
    )


def _case_signed_zeros(device, *, num_queries=None, num_keys=None):
    """Zero-score keys (six of every eight ids) with the selection boundary inside the zero group of every row.

    Even rows use all-negative head weights (an IEEE sequential head sum of their zero terms would be ``-0.0``;
    the program's positive-accumulator reduction gives ``+0.0``, so the bit check tells the two models apart),
    odd rows mixed-sign weights.  Segment 0 rows see 49..64 keys, segment 1 rows 21..40; with
    ``top_k = 12`` every mixed-weight row selects its <= 8 positive keys plus some but not all zeros, and every
    all-negative row selects 12 of >= 15 zeros, so the id-descending tie rule and the zero bits are exercised on
    every row.  The scaled geometry keeps the pattern (levels cycle through 1..255) with ``top_k = num_keys // 4``
    (at most 2048), which lies between the <= 1/8 positive keys and the >= 3/4 zeros of every row as long as the
    first row sees enough keys; a geometry breaking that is rejected.
    """
    geometry = _scaled_geometry(num_queries, num_keys)
    n = 64 if geometry is None else geometry[1]
    up, down = [0] * n, [0] * n
    for j in range(n):
        if j % 8 == 6:
            down[j] = (j // 8) % 255 + 1
        elif j % 8 == 7:
            up[j] = (j // 8) % 255 + 1

    def weights(u):
        return negative_weights() if u % 2 == 0 else mixed_weights()

    if geometry is None:
        return make_exact_inputs(
            [
                {"lq": 16, "up": up, "down": down, "weights": weights},
                {"lq": 20, "up": up[:40], "down": down[:40], "weights": weights},
            ],
            top_k=12,
            device=device,
            label="exact_signed_zeros",
        )
    lq, lk = geometry
    top_k = _scaled_top_k(lk)
    for visible in (lk - lq + 1, lk):  # the rows seeing the fewest and the most keys
        positive, negative = visible // 8, (visible + 1) // 8
        if not positive < top_k < visible - negative:
            raise ValueError(
                "signed_zeros needs #positive < top_k < #positive + #zeros on every row"
            )
    return make_exact_inputs(
        [{"lq": lq, "up": up, "down": down, "weights": weights}],
        top_k=top_k,
        device=device,
        label="exact_signed_zeros",
    )


def _case_negative(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    n = 100 if geometry is None else geometry[1]
    up, down = [0] * n, [0] * n
    for j in range(n):
        if j % 2 == 0:
            up[j] = (j // 2) % 37 + 1
        else:
            down[j] = (j // 2) % 29 + 1

    def weights(u):
        return negative_weights() if u % 3 == 0 else mixed_weights()

    if geometry is None:
        return make_exact_inputs(
            [
                {"lq": 12, "up": up, "down": down, "weights": weights},
                {"lq": 36, "up": up[:36], "down": down[:36], "weights": weights},
            ],
            top_k=40,
            device=device,
            label="exact_negative",
        )
    lq, lk = geometry
    return make_exact_inputs(
        [{"lq": lq, "up": up, "down": down, "weights": weights}],
        top_k=_scaled_top_k(lk),
        device=device,
        label="exact_negative",
    )


def _case_few_winners_large_tie(device, *, num_queries=None, num_keys=None):
    geometry = _scaled_geometry(num_queries, num_keys)
    lq, lk = (6, 4096) if geometry is None else geometry
    if lk < 128:
        raise ValueError("few_winners_large_tie needs at least 128 keys")
    up = [1] * lk
    # five winners at the relative positions of the 4096-key geometry (ids 3, 1000, 2047, 2048 and 4095 there)
    for j, level in (
        (3, 50),
        (lk // 4 - 24, 40),
        (lk // 2 - 1, 30),
        (lk // 2, 20),
        (lk - 1, 10),
    ):
        up[j] = level
    return make_exact_inputs(
        [{"lq": lq, "up": up}],
        top_k=2048 if geometry is None else _scaled_top_k(lk),
        device=device,
        label="exact_few_winners_large_tie",
    )


EXACT_CASES: dict[str, Callable[..., IndexerInputs]] = {
    "distinct": _case_distinct,
    "cutoff_ties": _case_cutoff_ties,
    "all_equal": _case_all_equal,
    "signed_zeros": _case_signed_zeros,
    "negative": _case_negative,
    "few_winners_large_tie": _case_few_winners_large_tie,
}


def make_exact_case(
    name: str,
    device="cuda",
    *,
    num_queries: Optional[int] = None,
    num_keys: Optional[int] = None,
) -> IndexerInputs:
    """An exactly representable case in its historical geometry or, with both sizes, as one segment of
    ``num_queries`` queries at the tail of ``num_keys`` keys built by the same rule."""
    return EXACT_CASES[name](device, num_queries=num_queries, num_keys=num_keys)


# -- packed causality ---------------------------------------------------------------------------------------------
PACKED_GEOMETRIES: dict[str, dict[str, Any]] = {
    "equal_lengths": dict(
        seg_q_len=[256, 384, 320], seg_k_len=[256, 384, 320], top_k=128
    ),
    "unequal_lengths": dict(
        seg_q_len=[48, 160, 90], seg_k_len=[900, 120, 90], top_k=128
    ),
    "positive_offsets": dict(
        seg_q_len=[100, 70], seg_k_len=[600, 450], top_k=96, q_causal_offsets=[7, 123]
    ),
    "negative_offsets": dict(
        seg_q_len=[100, 70], seg_k_len=[180, 260], top_k=96, q_causal_offsets=[-5, -40]
    ),
    "ratio2": dict(seg_q_len=[480, 251], seg_k_len=[240, 126], top_k=96, ratio=2),
    "ratio2_offsets": dict(
        seg_q_len=[320, 90],
        seg_k_len=[380, 60],
        top_k=96,
        ratio=2,
        q_causal_offsets=[150, -9],
    ),
    "empty_segments": dict(
        seg_q_len=[0, 60, 0, 72], seg_k_len=[50, 0, 0, 220], top_k=64
    ),
    "singletons": dict(seg_q_len=[1, 1, 2], seg_k_len=[1, 6, 1], top_k=16),
    "short_rows": dict(seg_q_len=[40, 12], seg_k_len=[90, 280], top_k=256),
    "k1": dict(seg_q_len=[260, 90], seg_k_len=[260, 800], top_k=1),
    "k4096": dict(seg_q_len=[12, 20], seg_k_len=[4600, 4200], top_k=4096),
}


def make_packed_inputs(
    name: str,
    device="cuda",
    *,
    seed: int = 100,
    peaked: bool = False,
    extra_segments: int = 0,
    extra_segment_queries: int = 1024,
    extra_segment_keys: int = 8192,
) -> IndexerInputs:
    """A packed geometry, optionally followed by ``extra_segments`` iid segments of ``extra_segment_queries``
    queries over ``extra_segment_keys`` keys (same ``top_k`` and ``ratio``; each appended segment sees all of its
    keys from its last query on, i.e. offset ``ratio * keys - queries``) so a call runs long enough for benchmark
    timing to resolve small differences while the geometry's own segments stay exactly as declared.  The values are
    :func:`make_integer_inputs`'s (exact in FP32)."""
    spec = dict(PACKED_GEOMETRIES[name])
    seg_q_len = list(spec.pop("seg_q_len"))
    seg_k_len = list(spec.pop("seg_k_len"))
    extra_segments = int(extra_segments)
    if extra_segments < 0:
        raise ValueError("extra_segments must be >= 0")
    if extra_segments:
        lq, lk = int(extra_segment_queries), int(extra_segment_keys)
        if lq < 1 or lk < 1:
            raise ValueError("extra segments need at least one query and one key")
        ratio = int(spec.get("ratio", 1))
        offsets = spec.get("q_causal_offsets")
        if offsets is not None or ratio != 1:  # the ratio-1 default already is lk - lq
            spec["q_causal_offsets"] = (
                effective_offsets(seg_q_len, seg_k_len, offsets, ratio)
                + [ratio * lk - lq] * extra_segments
            )
        seg_q_len += [lq] * extra_segments
        seg_k_len += [lk] * extra_segments
    return make_integer_inputs(
        seg_q_len,
        seg_k_len,
        device=device,
        seed=seed,
        peaked=peaked,
        label=f"packed_{name}",
        **spec,
    )


def make_boundary_inputs(
    lq: int, lk: int, *, top_k: int, device="cuda", seed: int = 7
) -> IndexerInputs:
    """One segment of ``lq`` queries at the tail of ``lk`` keys (default offset ``lk - lq``), integer values
    (:func:`make_integer_inputs`)."""
    return make_integer_inputs(
        [lq],
        [lk],
        top_k=top_k,
        device=device,
        seed=seed,
        label=f"boundary_{lq}x{lk}_k{top_k}",
    )


def make_nonfinite_inputs(
    device="cuda", *, seed: int = 11, top_k: int = 64
) -> IndexerInputs:
    """Three segments: [0] finite (integer values, :func:`make_integer_inputs`), [1] NaN queries and infinite keys,
    [2] BF16-max values whose products overflow FP32."""
    inputs = make_integer_inputs(
        [200, 64, 40],
        [200, 260, 40],
        top_k=top_k,
        device=device,
        seed=seed,
        label="nonfinite",
    )
    q, k = inputs.q, inputs.k
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    q[q_starts[1] + 3] = float("nan")
    q[q_starts[1] + 7, 5] = float("nan")
    k[k_starts[1] + 5] = float("inf")
    k[k_starts[1] + 9, 17] = float("-inf")
    big = torch.finfo(torch.bfloat16).max
    q[q_starts[2] : q_starts[2] + 8] = big
    k[k_starts[2] : k_starts[2] + 8] = big
    k[k_starts[2] + 8 : k_starts[2] + 12] = -big
    return inputs


def clean_rows(inputs: IndexerInputs) -> torch.Tensor:
    """Rows of the finite segment of :func:`make_nonfinite_inputs`."""
    return torch.arange(0, inputs.seg_q_len[0], dtype=torch.int64, device=inputs.device)


def clean_segment(inputs: IndexerInputs) -> IndexerInputs:
    """The finite first segment of :func:`make_nonfinite_inputs` as a problem of its own: its rows and keys come
    first, so its rows and segment-local ids are the same in both problems."""
    lq, lk = inputs.seg_q_len[0], inputs.seg_k_len[0]
    return build_inputs(
        inputs.q[:lq],
        inputs.k[:lk],
        inputs.w[:lq],
        [lq],
        [lk],
        top_k=inputs.top_k,
        softmax_scale=inputs.softmax_scale,
        q_causal_offsets=None
        if inputs.q_causal_offsets is None
        else [int(inputs.q_causal_offsets[0])],
        ratio=inputs.ratio,
        label=f"{inputs.label}_clean",
    )


def dirty_rows(inputs: IndexerInputs) -> torch.Tensor:
    """Rows of the non-finite segments of :func:`make_nonfinite_inputs`."""
    return torch.arange(
        inputs.seg_q_len[0], inputs.num_queries, dtype=torch.int64, device=inputs.device
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def fp32_ieee_matmul():
    """True-FP32 GEMMs (no TF32, no reduced-precision reductions); restores the previous policy."""
    prev_precision = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    matmul = torch.backends.cuda.matmul
    prev_bf16 = getattr(matmul, "allow_bf16_reduced_precision_reduction", None)
    prev_fp16 = getattr(matmul, "allow_fp16_reduced_precision_reduction", None)
    try:
        torch.set_float32_matmul_precision("highest")
        matmul.allow_tf32 = False
        if prev_bf16 is not None:
            matmul.allow_bf16_reduced_precision_reduction = False
        if prev_fp16 is not None:
            matmul.allow_fp16_reduced_precision_reduction = False
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        matmul.allow_tf32 = prev_tf32
        if prev_bf16 is not None:
            matmul.allow_bf16_reduced_precision_reduction = prev_bf16
        if prev_fp16 is not None:
            matmul.allow_fp16_reduced_precision_reduction = prev_fp16


def _relu_plus_zero(x: torch.Tensor) -> torch.Tensor:
    """``x if x > 0 else +0.0`` (also for ``-0.0``)."""
    return torch.where(x > 0, x, torch.zeros((), dtype=x.dtype, device=x.device))


def _dot_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum_d a[..., d] * b[..., d]`` accumulated in order ``d = 0 .. D-1`` with elementwise ops (broadcasting)."""
    acc = a[..., 0] * b[..., 0]
    for d in range(1, a.shape[-1]):
        acc = acc + a[..., d] * b[..., d]
    return acc


def _heads_sequential(terms: torch.Tensor) -> torch.Tensor:
    """``((t_0 + t_1) + t_2) + ...`` over dimension 1 (IEEE signed zeros)."""
    acc = terms.select(1, 0).clone()
    for h in range(1, terms.shape[1]):
        acc = acc + terms.select(1, h)
    return acc


def _logits(
    q: torch.Tensor, k: torch.Tensor, order: str, *, batched: bool
) -> torch.Tensor:
    """``[c, H, D] x [n, D] -> [c, H, n]`` (``batched=False``) or ``[c, H, D] x [c, n, D] -> [c, H, n]``."""
    if order == "sequential":
        kk = k[:, None, :, :] if batched else k[None, None, :, :]
        return _dot_sequential(q[:, :, None, :], kk)
    if order != "matmul":
        raise ValueError(f"order must be 'sequential' or 'matmul', got {order!r}")
    kt = k.transpose(1, 2) if batched else k.t()
    if q.dtype == torch.float32:
        with fp32_ieee_matmul():
            return torch.matmul(q, kt)
    return torch.matmul(q, kt)


def _fma_fp32(c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``fl32(c + a * b)``: the product of two FP32 values is exact in FP64 and the
    FP64 sum is rounded once to FP32 (a double rounding differs from the fused
    operation only when the FP64 sum lands on an FP32 halfway point; the
    exactly representable cases never do)."""
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(
        torch.float32
    )


def _heads_program_model(relu2: torch.Tensor, w_scaled: torch.Tensor) -> torch.Tensor:
    """The generated program's head reduction (module docstring): four interleaved
    FMA chains over ``h = m, m + 4, ...`` from ``+0.0``, then
    ``fl(fl(fl(c_0 + c_2) + fl(c_1 + c_3)) * 0.5)``.  ``relu2`` is ``[c, H, n]``
    (``fl(x + |x|)``), ``w_scaled`` ``[c, H, 1]`` (``fl(w * scale)``), FP32."""
    num_heads = relu2.shape[1]
    chains = []
    for m in range(4):
        acc = torch.zeros_like(relu2.select(1, 0))  # +0.0
        for h in range(m, num_heads, 4):
            acc = _fma_fp32(acc, relu2.select(1, h), w_scaled.select(1, h))
        chains.append(acc)
    return ((chains[0] + chains[2]) + (chains[1] + chains[3])) * 0.5


def _finish_scores(logits64, logits32, w_rows, scale: float):
    w64 = w_rows.to(torch.float64).unsqueeze(-1)
    s64 = _heads_sequential(w64 * _relu_plus_zero(logits64 * scale))
    relu2 = (
        logits32 + logits32.abs()
    )  # fl(x + |x|): +0.0 for x <= 0 (also -0.0), NaN for NaN and -inf
    w_scaled = (
        w_rows * torch.tensor(scale, dtype=torch.float32, device=w_rows.device)
    ).unsqueeze(-1)
    s32 = _heads_program_model(relu2, w_scaled)
    return s64, s32


def _fp64_values(x: torch.Tensor) -> torch.Tensor:
    """FP64 values of BF16 / e4m3 / FP32 elements (exact)."""
    return x.to(torch.float32).to(torch.float64)


def _rows_of(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """``x.index_select(0, index)``; e4m3 rows are gathered through their uint8 bytes."""
    if x.dtype == torch.float8_e4m3fn:
        return x.view(torch.uint8).index_select(0, index).view(torch.float8_e4m3fn)
    return x.index_select(0, index)


def _fp8_scores(logits64, w_rows, k_scale_rows):
    """FP8 form (module docstring): ``logits64 [c, H, n]`` exact, ``w_rows [c, H]`` fp32, ``k_scale_rows`` fp32
    broadcasting to ``[c, n]``; returns ``(s64, s32)``."""
    w64 = w_rows.to(torch.float64).unsqueeze(-1)
    head64 = _heads_sequential(w64 * _relu_plus_zero(logits64))
    s64 = head64 * k_scale_rows.to(torch.float64)
    s32 = head64.to(torch.float32) * k_scale_rows
    return s64, s32


def score_block(
    q_rows,
    w_rows,
    k_rows,
    scale: float,
    *,
    order: str = "sequential",
    k_scale_rows: Optional[torch.Tensor] = None,
):
    """Scores of every (query, key) pair: ``q_rows [c, H, D]`` bf16, ``w_rows [c, H]`` fp32, ``k_rows [n, D]`` bf16.

    Returns ``(s64 [c, n], s32 [c, n])``.  ``k_scale_rows [n]`` (fp32): the FP8 form, with e4m3 ``q_rows`` /
    ``k_rows``.
    """
    if k_scale_rows is not None:
        q64, k64 = _fp64_values(q_rows), _fp64_values(k_rows)
        return _fp8_scores(_logits(q64, k64, order, batched=False), w_rows, k_scale_rows)
    q64, k64 = q_rows.to(torch.float64), k_rows.to(torch.float64)
    q32, k32 = q_rows.to(torch.float32), k_rows.to(torch.float32)
    return _finish_scores(
        _logits(q64, k64, order, batched=False),
        _logits(q32, k32, order, batched=False),
        w_rows,
        scale,
    )


def score_selected(
    inputs: IndexerInputs, ids: torch.Tensor, *, rows=None, order: str = "sequential"
):
    """Scores at a ``[R, K]`` table of segment-local ids (``-1`` = padding -> ``-inf`` / ``-inf``).

    Returns ``(s64, s32)`` of shape ``[R, K]`` for the rows ``rows`` (default: all rows).
    """
    device = inputs.device
    if rows is None:
        rows = torch.arange(inputs.num_queries, dtype=torch.int64, device=device)
    R, K = int(rows.numel()), int(ids.shape[1])
    s64 = torch.full((R, K), PAD_SCORE, dtype=torch.float64, device=device)
    s32 = torch.full((R, K), PAD_SCORE, dtype=torch.float32, device=device)
    if R == 0 or K == 0 or inputs.num_keys == 0:
        return s64, s32
    per_row = K * (HEAD_DIM * 12 + NUM_HEADS * 40 + 64)
    chunk = max(1, min(R, CHUNK_BYTES // per_row))
    for r0 in range(0, R, chunk):
        r1 = min(R, r0 + chunk)
        rr = rows[r0:r1]
        global_ids, valid = _global_keys(inputs, rr, ids[r0:r1])
        wb = inputs.w.index_select(0, rr)
        if inputs.fp8:
            qb = _rows_of(inputs.q, rr)
            kb = _rows_of(inputs.k, global_ids.reshape(-1)).reshape(
                r1 - r0, K, HEAD_DIM
            )
            ksb = inputs.k_scale.index_select(0, global_ids.reshape(-1)).reshape(
                r1 - r0, K
            )
            q64, k64 = _fp64_values(qb), _fp64_values(kb)
            b64, b32 = _fp8_scores(_logits(q64, k64, order, batched=True), wb, ksb)
        else:
            qb = inputs.q.index_select(0, rr)
            kb = inputs.k.index_select(0, global_ids.reshape(-1)).reshape(
                r1 - r0, K, HEAD_DIM
            )
            l64 = _logits(qb.to(torch.float64), kb.to(torch.float64), order, batched=True)
            l32 = _logits(qb.to(torch.float32), kb.to(torch.float32), order, batched=True)
            b64, b32 = _finish_scores(l64, l32, wb, inputs.softmax_scale)
        s64[r0:r1] = torch.where(valid, b64, s64[r0:r1])
        s32[r0:r1] = torch.where(valid, b32, s32[r0:r1])
    return s64, s32


def _global_keys(inputs: IndexerInputs, rows: torch.Tensor, ids: torch.Tensor):
    """``(global key rows [R, K], valid [R, K])`` of a ``[R, K]`` table of segment-local ids of the query rows
    ``rows``.  A padding id (``-1``) is clamped onto a real key row (``valid`` discards it): one in a trailing empty
    key segment would point one past the last key."""
    k_starts = torch.tensor(
        inputs.key_starts()[:-1], dtype=torch.int64, device=inputs.device
    )
    seg = segment_of_rows(inputs).index_select(0, rows)
    local = ids.to(torch.int64)
    global_ids = (k_starts[seg].unsqueeze(1) + local.clamp_min(0)).clamp_max(
        max(0, inputs.num_keys - 1)
    )
    return global_ids, local >= 0


def score_pairs(
    inputs: IndexerInputs,
    rows: torch.Tensor,
    ids: torch.Tensor,
    *,
    order: str = "sequential",
):
    """Scores of arbitrary ``(row, local id)`` pairs (int64 ``[n]`` each): ``(s64 [n], s32 [n])``."""
    s64, s32 = score_selected(inputs, ids.view(-1, 1), rows=rows, order=order)
    return s64[:, 0], s32[:, 0]


def count_visible_pairs(inputs: IndexerInputs) -> int:
    return int(visible_rows(inputs, device="cpu").sum().item())


def resolve_order(inputs: IndexerInputs, order: str) -> str:
    if order in ("sequential", "matmul"):
        return order
    if order != "auto":
        raise ValueError(
            f"order must be 'auto', 'sequential' or 'matmul', got {order!r}"
        )
    if inputs.fp8:
        # e4m3 logits are exact in FP64 under any order
        return "matmul"
    return (
        "sequential"
        if count_visible_pairs(inputs) <= SEQUENTIAL_PAIR_LIMIT
        else "matmul"
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def structure_failures(
    inputs: IndexerInputs,
    indices: torch.Tensor,
    scores: torch.Tensor,
    *,
    rows=None,
    finite_domain: bool = True,
) -> list[str]:
    """Exact structural checks: shapes, dtypes, padding, uniqueness, ascending order, visibility."""
    failures: list[str] = []
    T, K = inputs.num_queries, inputs.top_k
    if tuple(indices.shape) != (T, K):
        failures.append(f"indices shape {tuple(indices.shape)} != {(T, K)}")
    if tuple(scores.shape) != (T, K):
        failures.append(f"scores shape {tuple(scores.shape)} != {(T, K)}")
    if indices.dtype != torch.int32:
        failures.append(f"indices dtype {indices.dtype} != int32")
    if scores.dtype != torch.float32:
        failures.append(f"scores dtype {scores.dtype} != float32")
    if failures:
        return failures
    device = indices.device
    visible = visible_rows(inputs, device)
    selected = visible.clamp(max=K)
    if rows is not None:
        rows = rows.to(device=device, dtype=torch.int64)
        indices, scores, visible, selected = (
            indices[rows],
            scores[rows],
            visible[rows],
            selected[rows],
        )
    if indices.shape[0] == 0:
        return failures
    slot = torch.arange(K, dtype=torch.int64, device=device).unsqueeze(0)
    valid = slot < selected.unsqueeze(1)
    ids = indices.to(torch.int64)

    def first(mask: torch.Tensor, what: str) -> None:
        if bool(mask.any()):
            r, c = mask.nonzero()[0].tolist()
            row = int(rows[r].item()) if rows is not None else r
            failures.append(
                f"{what} at row {row} slot {c} (id={int(ids[r, c])}, score={float(scores[r, c])!r})"
            )

    first(
        valid & ((ids < 0) | (ids >= visible.unsqueeze(1))),
        "invisible or out-of-segment id in a valid slot",
    )
    first((~valid) & (ids != PAD_ID), "padding slot without id -1")
    first((~valid) & ~torch.isneginf(scores), "padding slot without -inf score")
    if K > 1:
        first(
            valid[:, 1:] & ~(ids[:, 1:] > ids[:, :-1]),
            "ids not strictly ascending (duplicate or unsorted)",
        )
    if finite_domain:
        first(valid & ~torch.isfinite(scores), "non-finite score in a valid slot")
    return failures


# ---------------------------------------------------------------------------
# Selection by the FP64 score
# ---------------------------------------------------------------------------


@dataclass
class ReferenceSelection:
    """Reference output (ids ascending per row; padding ``-1`` / ``-inf``)."""

    inputs: IndexerInputs
    indices: torch.Tensor  # int32 [T, K]
    scores: torch.Tensor  # fp32 [T, K]: the independent FP32 scorer at the selected ids
    scores_fp64: torch.Tensor  # fp64 [T, K]: the exact scores
    visible: torch.Tensor  # int64 [T]
    selected_count: torch.Tensor  # int64 [T] = min(K, visible)
    order: str
    chunks: dict[str, int] = field(default_factory=dict)


def _ranking_key(scores: torch.Tensor) -> torch.Tensor:
    """``-0.0 -> +0.0``; NaN ranks lowest (``-inf``); everything else by value.

    The NaN placement is the reference's own selection convention (finite inputs
    are the contract's domain); it is not a guarantee of the generated program.
    """
    return torch.nan_to_num(
        scores + 0.0, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf")
    )


def _lexicographic_top(
    rank: torch.Tensor, ids: torch.Tensor, keep: int
) -> torch.Tensor:
    """Permutation ``[c, keep]`` selecting the top entries by (rank desc, id desc): two stable sorts."""
    _, by_id = torch.sort(ids, dim=1, descending=True, stable=True)
    _, by_rank = torch.sort(rank.gather(1, by_id), dim=1, descending=True, stable=True)
    return by_id.gather(1, by_rank)[:, :keep]


def plan_chunks(
    inputs: IndexerInputs, chunk_bytes: int = CHUNK_BYTES
) -> tuple[int, int]:
    """(query chunk, key chunk) under a byte budget (fp64 + fp32 logits and terms per pair)."""
    K = inputs.top_k
    max_lk = max(inputs.seg_k_len, default=1)
    key_chunk = max(1, min(max_lk, max(2 * K, 8192)))
    per_pair = NUM_HEADS * 40 + 64
    query_chunk = max(1, min(512, chunk_bytes // max(1, key_chunk * per_pair)))
    return query_chunk, key_chunk


def select_reference(
    inputs: IndexerInputs,
    *,
    order: str = "auto",
    query_chunk: Optional[int] = None,
    key_chunk: Optional[int] = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> ReferenceSelection:
    """Global selection per row by the FP64 score with the contract's tie rule, memory-bounded over keys.

    Ranking uses the FP64 score.  ``query_chunk`` / ``key_chunk`` change only
    the work partition: with ``order="sequential"`` the result is bitwise
    identical for every partition, with ``"matmul"`` only up to the BLAS
    rounding of the dot products.  On inputs whose scores are exact in FP32
    (:func:`exact_inputs`) it is :func:`exact_selection`'s selection.
    """
    order = resolve_order(inputs, order)
    device = inputs.device
    T, K = inputs.num_queries, inputs.top_k
    cq, ck = plan_chunks(inputs, chunk_bytes)
    if query_chunk is not None:
        cq = max(1, int(query_chunk))
    if key_chunk is not None:
        ck = max(1, int(key_chunk))
    out_ids = torch.full((T, K), PAD_ID, dtype=torch.int32, device=device)
    out_s32 = torch.full((T, K), PAD_SCORE, dtype=torch.float32, device=device)
    out_s64 = torch.full((T, K), PAD_SCORE, dtype=torch.float64, device=device)
    visible = visible_rows(inputs)
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    for s in range(inputs.num_segments):
        q0, lq = q_starts[s], inputs.seg_q_len[s]
        k0, lk = k_starts[s], inputs.seg_k_len[s]
        if lq == 0 or lk == 0:
            continue
        keep = min(K, lk)
        for t0 in range(q0, q0 + lq, cq):
            t1 = min(q0 + lq, t0 + cq)
            c = t1 - t0
            vis_rows = visible[t0:t1]
            j_end = int(
                vis_rows.max().item()
            )  # visibility is monotone within a segment: the last row sees most
            if j_end == 0:
                continue
            qb, wb = inputs.q[t0:t1], inputs.w[t0:t1]
            best_rank = torch.empty((c, 0), dtype=torch.float64, device=device)
            best_id = torch.empty((c, 0), dtype=torch.int64, device=device)
            best_s32 = torch.empty((c, 0), dtype=torch.float32, device=device)
            best_s64 = torch.empty((c, 0), dtype=torch.float64, device=device)
            for j0 in range(0, j_end, ck):
                j1 = min(j_end, j0 + ck)
                s64, s32 = score_block(
                    qb,
                    wb,
                    inputs.k[k0 + j0 : k0 + j1],
                    inputs.softmax_scale,
                    order=order,
                    k_scale_rows=None
                    if inputs.k_scale is None
                    else inputs.k_scale[k0 + j0 : k0 + j1],
                )
                ids = (
                    torch.arange(j0, j1, dtype=torch.int64, device=device)
                    .unsqueeze(0)
                    .expand(c, -1)
                )
                seen = ids < vis_rows.unsqueeze(1)
                rank = torch.where(
                    seen, _ranking_key(s64), torch.full_like(s64, float("-inf"))
                )
                ids = torch.where(seen, ids, torch.full_like(ids, -1))
                cand_rank = torch.cat([best_rank, rank], dim=1)
                cand_id = torch.cat([best_id, ids], dim=1)
                perm = _lexicographic_top(
                    cand_rank, cand_id, min(keep, cand_rank.shape[1])
                )
                best_rank = cand_rank.gather(1, perm)
                best_id = cand_id.gather(1, perm)
                best_s32 = torch.cat([best_s32, s32], dim=1).gather(1, perm)
                best_s64 = torch.cat([best_s64, s64], dim=1).gather(1, perm)
            # ascending ids, padding (id -1) last
            key = torch.where(
                best_id >= 0, best_id, torch.full_like(best_id, INT64_MAX)
            )
            key_sorted, perm = torch.sort(key, dim=1)
            pad = key_sorted == INT64_MAX
            n = key_sorted.shape[1]
            out_ids[t0:t1, :n] = torch.where(
                pad, torch.full_like(key_sorted, PAD_ID), key_sorted
            ).to(torch.int32)
            out_s32[t0:t1, :n] = torch.where(
                pad, torch.full_like(best_s32, PAD_SCORE), best_s32.gather(1, perm)
            )
            out_s64[t0:t1, :n] = torch.where(
                pad, torch.full_like(best_s64, PAD_SCORE), best_s64.gather(1, perm)
            )
    return ReferenceSelection(
        inputs=inputs,
        indices=out_ids,
        scores=out_s32,
        scores_fp64=out_s64,
        visible=visible,
        selected_count=visible.clamp(max=K),
        order=order,
        chunks={"query_chunk": cq, "key_chunk": ck},
    )


def bruteforce_rows(inputs: IndexerInputs) -> tuple[list[list[int]], list[list[float]]]:
    """Pure-Python per-row selection (FP64 scores from torch, ranking in Python) for tiny problems."""
    q = _fp64_values(inputs.q).cpu()
    k = _fp64_values(inputs.k).cpu()
    w = inputs.w.to(torch.float64).cpu()
    k_scale = None if inputs.k_scale is None else inputs.k_scale.to(torch.float64).cpu()
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    offsets = inputs.offsets()
    ids_out: list[list[int]] = []
    scores_out: list[list[float]] = []
    for s in range(inputs.num_segments):
        for u in range(inputs.seg_q_len[s]):
            t = q_starts[s] + u
            vis = visible_scalar(offsets[s], u, inputs.ratio, inputs.seg_k_len[s])
            scored = []
            for j in range(vis):
                logit = (q[t] * k[k_starts[s] + j]).sum(dim=1) * inputs.softmax_scale
                terms = w[t] * torch.clamp(logit, min=0.0)
                score = float(terms.sum())
                if k_scale is not None:
                    score *= float(k_scale[k_starts[s] + j])
                scored.append((score + 0.0, j, score))
            scored.sort(key=lambda e: (e[0], e[1]), reverse=True)
            chosen = sorted(scored[: inputs.top_k], key=lambda e: e[1])
            row_ids = [j for _, j, _ in chosen] + [PAD_ID] * (
                inputs.top_k - len(chosen)
            )
            row_scores = [sc for _, _, sc in chosen] + [PAD_SCORE] * (
                inputs.top_k - len(chosen)
            )
            ids_out.append(row_ids)
            scores_out.append(row_scores)
    return ids_out, scores_out


# ---------------------------------------------------------------------------
# Exact selection on inputs exact in FP32, and the scores of given keys
# ---------------------------------------------------------------------------


def _values32(x: torch.Tensor) -> torch.Tensor:
    """FP32 values of BF16 / e4m3 / FP32 elements (exact)."""
    return x.to(torch.float32)


def _dyadic_step(w: torch.Tensor) -> Optional[int]:
    """The smallest ``e`` in ``0 .. 12`` with every ``w * 2**e`` an integer, or None."""
    for e in range(13):
        v = w * float(2**e)
        if bool((v == v.round()).all()):
            return e
    return None


def exact_inputs(inputs: IndexerInputs) -> Optional[str]:
    """Why the scores of ``inputs`` are not exact in FP32 under every summation order, or None when they are.

    They are when ``q`` and ``k`` hold integers, every ``w`` is a multiple of ``2**-e`` (``e <= 12``),
    ``softmax_scale`` is a power of two (the FP8 form: 1) and the head sum of every row, counted in its step
    ``softmax_scale * 2**-e``, stays below :data:`EXACT_SUM_LIMIT` in magnitude for twice the largest head terms (the
    program's ``r_h = 2 relu(x_h)``).  Then every product, partial sum and final score of a (query, key) pair is an
    FP32 value, the FP8 form's one ``k_scale`` multiply aside (rounded once, as the contract defines it)."""
    q, k, w = _values32(inputs.q), _values32(inputs.k), inputs.w.to(torch.float32)
    for name, x in (("q", q), ("k", k), ("w", w)):
        if not bool(torch.isfinite(x).all()):
            return f"{name} holds a non-finite value"
    if not bool((q == q.round()).all()) or not bool((k == k.round()).all()):
        return "q and k must hold integers"
    step = _dyadic_step(w)
    if step is None:
        return "every w must be a multiple of 2**-12"
    mantissa, _ = math.frexp(inputs.softmax_scale)
    if inputs.softmax_scale <= 0 or mantissa != 0.5:
        return f"softmax_scale {inputs.softmax_scale} is not a power of two"
    if inputs.fp8 and not bool(torch.isfinite(inputs.k_scale).all()):
        return "k_scale holds a non-finite value"
    if inputs.num_queries == 0 or inputs.num_keys == 0:
        return None
    k_max = k.abs().amax(dim=0)  # [D]: the largest |k| of every dimension
    logit_max = (q.abs() * k_max).sum(dim=-1)  # [T, H]: a bound on every |x_h| of the row
    head_max = (w.abs() * logit_max).sum(dim=-1).max()  # the largest head sum of any row
    units = float(head_max) * 2 * 2**step
    if units >= EXACT_SUM_LIMIT:
        return f"a head sum reaches {units:.0f} steps of 2**-{step}: not exact in FP32"
    return None


def order_key(scores32: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The contract's order as a non-negative int64: ``key32 << 31 | id`` with ``key32`` the FP32 score's bits mapped
    to an unsigned order (``-0.0`` as ``+0.0``, NaN lowest); a larger key ranks first (score descending, then id
    descending)."""
    m = scores32.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    m = torch.where((m & 0x7FFFFFFF) == 0, torch.zeros_like(m), m)
    key32 = torch.where(m & 0x80000000 != 0, ~m & 0xFFFFFFFF, m | 0x80000000)
    key32 = torch.where((m & 0x7FFFFFFF) > 0x7F800000, torch.zeros_like(key32), key32)
    return (key32 << 31) | ids.to(torch.int64)


def exact_selection(inputs: IndexerInputs, *, chunk_elems: int = 1 << 27) -> ReferenceSelection:
    """The exact top-``min(top_k, visible)`` of every row of inputs whose scores are exact in FP32
    (:func:`exact_inputs`, else ``ValueError``): ranked by the FP32 score the contract defines (score descending,
    segment-local id descending, ``+0 == -0``), ids in ascending order, then ``(-1, -inf)`` padding.  ``scores`` holds
    the FP32 scores of the kept keys and ``scores_fp64`` their FP64 scores (:func:`scores_at`).  The logits run in FP32
    with TF32 off, ``chunk_elems`` logits at a time (exact on these inputs in any order)."""
    reason = exact_inputs(inputs)
    if reason is not None:
        raise ValueError(f"exact_selection needs inputs exact in FP32: {reason}")
    device = inputs.device
    T, K = inputs.num_queries, inputs.top_k
    out_ids = torch.full((T, K), PAD_ID, dtype=torch.int32, device=device)
    out_s32 = torch.full((T, K), PAD_SCORE, dtype=torch.float32, device=device)
    visible = visible_rows(inputs)
    q_starts, k_starts = inputs.query_starts(), inputs.key_starts()
    q32, w32 = _values32(inputs.q), inputs.w.to(torch.float32)
    with fp32_ieee_matmul():
        for s in range(inputs.num_segments):
            q0, lq = q_starts[s], inputs.seg_q_len[s]
            k0, lk = k_starts[s], inputs.seg_k_len[s]
            if lq == 0 or lk == 0:
                continue
            keep = min(K, lk)
            keys = _values32(inputs.k[k0 : k0 + lk])
            scale = (
                inputs.k_scale[k0 : k0 + lk].to(torch.float32)
                if inputs.fp8
                else inputs.softmax_scale
            )
            ids = torch.arange(lk, dtype=torch.int64, device=device).unsqueeze(0)
            step = max(1, min(lq, chunk_elems // (NUM_HEADS * lk)))
            for t0 in range(q0, q0 + lq, step):
                t1 = min(q0 + lq, t0 + step)
                logits = torch.matmul(q32[t0:t1], keys.t())  # [c, H, lk], exact
                head = (logits.clamp_min(0) * w32[t0:t1].unsqueeze(-1)).sum(dim=1)
                s32 = head * scale + 0.0  # exact, or the FP8 form's one rounding; -0.0 -> +0.0
                key = order_key(s32, ids.expand(t1 - t0, -1))
                key = torch.where(ids < visible[t0:t1].unsqueeze(1), key, torch.full_like(key, -1))
                best = key.topk(keep, dim=1)
                chosen = torch.where(best.values >= 0, best.indices, torch.full_like(best.indices, -1))
                order = torch.where(chosen >= 0, chosen, torch.full_like(chosen, INT64_MAX)).argsort(dim=1)
                chosen = chosen.gather(1, order)
                picked = s32.gather(1, chosen.clamp_min(0))
                out_ids[t0:t1, :keep] = chosen.to(torch.int32)
                out_s32[t0:t1, :keep] = torch.where(chosen >= 0, picked, torch.full_like(picked, PAD_SCORE))
    return ReferenceSelection(
        inputs=inputs,
        indices=out_ids,
        scores=out_s32,
        scores_fp64=scores_at(inputs, out_ids, torch.float64),
        visible=visible,
        selected_count=visible.clamp(max=K),
        order="exact",
    )


def _stored(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``x`` as a computation in ``dtype`` reads it: cast to ``dtype`` when ``x`` is stored no wider, else as is (the
    FP32 ``w`` and ``k_scale`` stay FP32 in a BF16 computation)."""
    return x.to(dtype) if x.dtype.itemsize <= dtype.itemsize else x


def scores_at(
    inputs: IndexerInputs,
    ids: torch.Tensor,
    dtype: torch.dtype,
    *,
    rows=None,
    chunk_elems: int = 1 << 25,
) -> torch.Tensor:
    """The scores ``sum_h w_h relu(x_h) * scale`` of a ``[R, K]`` table of segment-local ids of the query rows ``rows``
    (default: all rows) in ``dtype`` (``-inf`` at padding ids): ``q`` and ``k`` in ``dtype`` (exact from BF16 and
    e4m3), the einsum, ReLU and weighted head sum in torch in ``dtype`` with the FP32 ``w`` and ``k_scale`` as stored
    (:func:`_stored`), ``scale`` the softmax scale (the FP8 form: ``k_scale[j]``), the result rounded to ``dtype``.
    ``torch.float64`` gives the oracle; ``torch.bfloat16`` the naive tier."""
    device = inputs.device
    if rows is None:
        rows = torch.arange(inputs.num_queries, dtype=torch.int64, device=device)
    rows = rows.to(device=device, dtype=torch.int64)
    R, K = int(rows.numel()), int(ids.shape[1])
    out = torch.full((R, K), PAD_SCORE, dtype=dtype, device=device)
    if R == 0 or K == 0 or inputs.num_keys == 0:
        return out
    step = max(1, chunk_elems // (K * HEAD_DIM))
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        rr = rows[r0:r1]
        keys, valid = _global_keys(inputs, rr, ids[r0:r1])
        q = _values32(_rows_of(inputs.q, rr)).to(dtype)
        k = _values32(_rows_of(inputs.k, keys.reshape(-1))).to(dtype).reshape(r1 - r0, K, HEAD_DIM)
        logits = torch.einsum("rhd,rkd->rhk", q, k)
        head = (logits.relu() * _stored(inputs.w.index_select(0, rr), dtype).unsqueeze(-1)).sum(dim=1)
        if inputs.fp8:
            scale = _stored(inputs.k_scale.index_select(0, keys.reshape(-1)).reshape(r1 - r0, K), dtype)
        else:
            scale = inputs.softmax_scale
        value = (head * scale).to(dtype)
        out[r0:r1] = torch.where(valid, value, torch.full_like(value, PAD_SCORE))
    return out


def compare_selection(result, reference: ReferenceSelection, *, rows=None) -> tuple[list[str], int, int]:
    """``(structure failures, rows compared, rows whose ids equal the reference's)`` of ``result = (indices,
    scores)`` with rows in ascending id order.  No bound: on inputs exact in FP32 every row must hold the reference's
    ids; on Gaussian inputs the FP32 rounding may reorder keys whose FP64 scores are near-tied, so the count is
    reported as information (benchmarks)."""
    indices, scores = result
    failures = structure_failures(reference.inputs, indices, scores, rows=rows)
    if failures:
        return failures, 0, 0
    if rows is None:
        rows = torch.arange(indices.shape[0], dtype=torch.int64, device=indices.device)
    rows = rows.to(device=indices.device, dtype=torch.int64)
    got = indices.index_select(0, rows)
    want = reference.indices.to(indices.device).index_select(0, rows)
    same = int((got == want).all(dim=1).sum().item())
    return failures, int(rows.numel()), same


def same_bits(a, b) -> bool:
    """Bitwise identity of two ``(indices, scores)`` results."""
    ia, sa = a
    ib, sb = b
    if tuple(ia.shape) != tuple(ib.shape) or tuple(sa.shape) != tuple(sb.shape):
        return False
    return bool(torch.equal(ia, ib)) and bool(
        torch.equal(
            sa.contiguous().view(torch.int32), sb.contiguous().view(torch.int32)
        )
    )


__all__ = [
    "CHUNK_BYTES",
    "DEFAULT_SCALE",
    "DEFAULT_TOP_K",
    "E4M3_MAX",
    "EXACT_CASES",
    "EXACT_SCALE",
    "EXACT_SUM_LIMIT",
    "HEAD_DIM",
    "INTEGER_RANGE",
    "INTEGER_SCALE",
    "NUM_HEADS",
    "PACKED_GEOMETRIES",
    "PAD_ID",
    "PAD_SCORE",
    "SEQUENTIAL_PAIR_LIMIT",
    "SINK_NUM_FREQUENCIES",
    "SINK_SEED",
    "TRAINER_KEY_ROW_STRIDE",
    "IndexerInputs",
    "ReferenceSelection",
    "bruteforce_rows",
    "build_inputs",
    "clean_rows",
    "clean_segment",
    "compare_selection",
    "count_visible_pairs",
    "cumulative",
    "dequantized_bf16",
    "dirty_rows",
    "effective_offsets",
    "exact_closed_form",
    "exact_inputs",
    "exact_selection",
    "fp32_ieee_matmul",
    "integer_key_scales",
    "make_boundary_inputs",
    "make_exact_case",
    "make_exact_inputs",
    "make_fp8_inputs",
    "make_integer_fp8_inputs",
    "make_integer_inputs",
    "make_integer_sink_inputs",
    "make_nonfinite_inputs",
    "make_packed_inputs",
    "make_random_inputs",
    "make_sink_inputs",
    "mixed_weights",
    "negative_weights",
    "order_key",
    "plan_chunks",
    "resolve_order",
    "same_bits",
    "score_block",
    "score_pairs",
    "score_selected",
    "scores_at",
    "segment_of_rows",
    "select_reference",
    "structure_failures",
    "visible_rows",
    "visible_scalar",
]
