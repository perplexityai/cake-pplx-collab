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

Independent single-GPU fixtures for the MoK kernel tests.

The MXFP8 model follows the MoK recipe without importing the package: 32
consecutive K elements share a UE8M0 byte ``round_up(max(amax * 0.002232142857,
1e-12))`` saturated to 254, data is E4M3 ``x * 2**(127 - byte)`` (round to
nearest even, saturating), and 128 x 128 scale tiles store byte
``(row % 32) * 16 + (row // 32) * 4 + k_block`` as ``[R/128, K/128, 32, 16]``.
:func:`mxfp8_quantize_32x32` is the same recipe with one byte per 32 x 32 block.

Random inputs are drawn through :func:`randn`, :func:`randn_like`,
:func:`rand`, :func:`randint`, :func:`normal_` and :func:`uniform_`, so every
GPU gets the same inputs.  torch's CUDA random kernels launch ``ceil(n / 256)``
blocks of 256 threads for ``n`` elements, but at most ``threads per SM / 256``
blocks per SM, and element ``i`` takes the numbers of thread ``i`` modulo the
threads launched: one call of more elements than that has other values on a GPU
with another SM count (148 against 152, for example), and the generator's
offset, so every later draw of it, moves by another step.  The helpers fill a
CUDA tensor in consecutive torch calls of :data:`DRAW_CHUNK` elements each.  A
call of at most 2^17 elements runs one thread per element on every GPU with
2^17 threads or more (64 SMs of 2048 threads or 128 SMs of 1024; the supported
GPUs have 148 to 216 SMs of 1024 or 2048 threads), so its values and the offset
step it takes (4) do not depend on the SM count.  Each call is torch's own: the
generator (``generator``, or the default one that ``torch.manual_seed`` seeds),
the dtype (a BF16 tensor holds float normals rounded to BF16) and the order of
the draws are unchanged, and a draw of at most :data:`DRAW_CHUNK` elements is a
single torch call.  A CPU tensor is drawn in one torch call.
"""

import pytest
import torch
import torch.nn.functional as F

SUPPORTED = ((10, 0), (10, 3), (10, 7))


def require_gpu():
    if not torch.cuda.is_available() or (
        torch.cuda.get_device_capability() not in SUPPORTED
    ):
        pytest.skip("Requires an SM100a, SM103a or SM107a CUDA device")


#: Elements of one torch random call of the draw helpers on a CUDA device.
DRAW_CHUNK = 1 << 17
#: Threads per block of torch's CUDA random kernels.
_RANDOM_BLOCK = 256


def fill_in_chunks(out, fill):
    """``fill(part)`` for the consecutive parts of :data:`DRAW_CHUNK` elements of
    the contiguous ``out`` (the last part the rest), in order.  Returns ``out``."""
    for part in out.view(-1).split(DRAW_CHUNK):
        fill(part)
    return out


def _draw(out, fill):
    """Fills ``out`` with ``fill`` (a torch random call on its argument): a CUDA
    tensor by :func:`fill_in_chunks`, any other tensor in one call."""
    if out.device.type != "cuda":
        fill(out)
        return out
    p = torch.cuda.get_device_properties(out.device)
    per_sm = p.max_threads_per_multi_processor // _RANDOM_BLOCK
    assert p.multi_processor_count * per_sm * _RANDOM_BLOCK >= DRAW_CHUNK, p.name
    return fill_in_chunks(out, fill)


def randn(*size, generator=None, dtype=None, device=None):
    """``torch.randn(*size, generator=generator, dtype=dtype, device=device)``,
    the same values on every GPU (:data:`DRAW_CHUNK`)."""
    out = torch.empty(*size, dtype=dtype, device=device)
    return _draw(out, lambda t: t.normal_(generator=generator))


def randn_like(x, *, generator=None):
    """:func:`randn` of ``x``'s shape, dtype and device."""
    return randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)


def rand(*size, generator=None, dtype=None, device=None):
    """``torch.rand(*size, generator=generator, dtype=dtype, device=device)``,
    the same values on every GPU."""
    out = torch.empty(*size, dtype=dtype, device=device)
    return _draw(out, lambda t: t.uniform_(generator=generator))


def randint(low, high, size, *, generator=None, dtype=torch.int64, device=None):
    """``torch.randint(low, high, size, ...)``, the same values on every GPU."""
    out = torch.empty(size, dtype=dtype, device=device)
    return _draw(out, lambda t: t.random_(low, high, generator=generator))


def normal_(x, mean=0.0, std=1.0, *, generator=None):
    """``x.normal_(mean, std)`` in place (a CUDA ``x`` contiguous), the same
    values on every GPU.  Returns ``x``."""
    return _draw(x, lambda t: t.normal_(mean, std, generator=generator))


def uniform_(x, low=0.0, high=1.0, *, generator=None):
    """``x.uniform_(low, high)`` in place (a CUDA ``x`` contiguous), the same
    values on every GPU.  Returns ``x``."""
    return _draw(x, lambda t: t.uniform_(low, high, generator=generator))


def single_rank_schedule(routes, experts, extra=512):
    """Single-rank padded schedule: per expert its route slots, then -1 padding."""
    slots = [torch.where(routes.flatten() == e)[0] for e in range(experts)]
    count_values = [((s.numel() + 255) // 256) * 256 for s in slots]
    pieces = [
        torch.cat(
            (
                s.int(),
                torch.full(
                    (c - s.numel(),), -1, dtype=torch.int32, device=routes.device
                ),
            )
        )
        for s, c in zip(slots, count_values, strict=True)
    ]
    schedule = torch.cat(pieces)
    actual = schedule.numel()
    schedule = torch.cat(
        (schedule, torch.full((extra,), -1, dtype=torch.int32, device=routes.device))
    )
    peers = torch.where(schedule >= 0, 0, -1).int()
    num_tokens = torch.tensor([actual], device=routes.device, dtype=torch.int32)
    counts = torch.tensor(count_values, device=routes.device, dtype=torch.int32)
    return dict(
        slots=slots,
        count_values=count_values,
        schedule=schedule,
        actual=actual,
        peers=peers,
        num_tokens=num_tokens,
        counts=counts,
    )


def _e8m0_round_up(scale):
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    return (exponent + (mantissa != 0).to(torch.int32)).clamp(max=254).to(torch.uint8)


def _quantize_rows(x):
    *lead, rows, cols = x.shape
    blocks = x.float().reshape(*lead, rows, cols // 32, 32)
    amax = blocks.abs().amax(-1)
    scale = torch.clamp(
        amax * torch.tensor(0.002232142857, dtype=torch.float32, device=x.device),
        min=1e-12,
    )
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return data.reshape(*lead, rows, cols), byte


def _scale_tiles(byte):
    *lead, rows, kb = byte.shape
    tiles = byte.reshape(-1, rows // 128, 4, 32, kb // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return tiles.reshape(-1, kb // 4, 32, 16).contiguous()


def mxfp8_quantize(x, normal=True, transposed=True):
    """``(fp8, scales, fp8_t, scales_t)`` of a BF16 ``[..., R, K]`` tensor."""
    out = [None] * 4
    if normal:
        data, byte = _quantize_rows(x)
        out[0], out[1] = data, _scale_tiles(byte)
    if transposed:
        data, byte = _quantize_rows(x.transpose(-1, -2).contiguous())
        out[2], out[3] = data, _scale_tiles(byte)
    return tuple(out)


def mxfp8_quantize_32x32(x):
    """``(fp8, scales, fp8_t, scales_t, byte)`` of a BF16 ``[..., R, K]`` tensor with one UE8M0 byte per 32 x 32 block:
    ``byte`` ``[..., R/32, K/32]`` is the recipe's byte of the block's largest magnitude and every element of the block
    is E4M3 ``x * 2**(127 - byte)``; ``fp8_t`` is ``fp8`` transposed, and the tiles repeat each block's byte for its 32
    rows (``scales``) or its 32 columns (``scales_t``)."""
    *lead, rows, cols = x.shape
    blocks = x.float().reshape(*lead, rows // 32, 32, cols // 32, 32)
    amax = blocks.abs().amax(dim=(-3, -1))
    scale = torch.clamp(
        amax * torch.tensor(0.002232142857, dtype=torch.float32, device=x.device),
        min=1e-12,
    )
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    data = (blocks * inverse[..., :, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    data = data.reshape(*lead, rows, cols)
    rows_byte = byte.repeat_interleave(32, dim=-2)
    cols_byte = byte.transpose(-1, -2).repeat_interleave(32, dim=-2)
    return data, _scale_tiles(rows_byte), data.transpose(-1, -2).contiguous(), _scale_tiles(cols_byte), byte


def mxfp8_dequantize(data, scales):
    """FP32 values of E4M3 data and its tile-layout UE8M0 scales."""
    *lead, rows, cols = data.shape
    s = scales.reshape(-1, rows // 128, cols // 128, 32, 4, 4).permute(0, 1, 4, 3, 2, 5)
    byte = s.reshape(*lead, rows, cols // 32).to(torch.int32)
    factor = (byte << 23).view(torch.float32)
    values = data.float().reshape(*lead, rows, cols // 32, 32) * factor[..., None]
    return values.reshape(*lead, rows, cols)


def swiglu(gate, up, limit=None):
    """BF16-rounded SwiGLU on FP32-promoted inputs; optional clamp limit."""
    gate, up = gate.float(), up.float()
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (gate * torch.sigmoid(gate) * up).bfloat16()


def production_inputs(tokens, hidden, intermediate, experts, topk, limit, *, device="cuda", seed=0):
    """One draw of the requester's input distribution for the router weight before the down projection.

    ``x``: rows of unit RMS (the post-RMSNorm hidden state) in BF16.  Router: ``sqrt(softplus(logits))`` scores of
    ``randn`` logits over ``experts``, the top ``topk`` distinct experts per token, their scores normalized over the
    top ``topk`` and scaled by 1.5 (FP32; each row sums to 1.5, each weight in [0, 1.5]).  Every 8th token gets its
    smallest weight set to exactly 0 (the others rescaled to sum to 1.5) and token 2 the weights ``(1.5, 0, ..., 0)``.
    Gate and up weights ``randn * 4 / sqrt(hidden)`` (gate and up of standard deviation about 4, so a small fraction
    passes the limit), down weights ``randn / sqrt(intermediate)``, shared and routed alike; ``dy`` ``randn``.  Token 1
    is pinned: its ``x`` row is ``+-1`` and its first expert's gate row 0 and up rows 0 and 1 are ``+-x * limit /
    hidden``, so its gate is exactly ``limit`` and its up exactly ``+limit`` and ``-limit`` there (``limit / hidden``
    must be a BF16 value).  Returns a dict of ``x``, ``dy``, ``ids`` (int64 ``[T, topk]``), ``weights`` (FP32
    ``[T, topk]``), ``shared`` and ``routed`` (BF16, the package's shapes) and ``pinned`` (token, expert).  The draws
    come from a generator of ``device`` seeded with ``seed`` (:func:`randn`)."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def normal(*shape):
        return randn(*shape, generator=generator, device=device)

    x = normal(tokens, hidden)
    x = (x / x.square().mean(-1, keepdim=True).sqrt()).bfloat16()
    scores = F.softplus(normal(tokens, experts)).sqrt()
    top, ids = scores.topk(topk, dim=-1)
    weights = top / top.sum(-1, keepdim=True) * 1.5
    zero = torch.arange(tokens, device=device) % 8 == 7
    weights[zero, topk - 1] = 0.0
    weights[zero] = weights[zero] / weights[zero].sum(-1, keepdim=True) * 1.5
    if tokens > 2:
        weights[2] = 0.0
        weights[2, 0] = 1.5
    gate_up = 4.0 / hidden**0.5
    down = 1.0 / intermediate**0.5
    shared = [
        (normal(intermediate, hidden) * gate_up).bfloat16(),
        (normal(intermediate, hidden) * gate_up).bfloat16(),
        (normal(hidden, intermediate) * down).bfloat16(),
    ]
    routed = [
        (normal(experts, intermediate, hidden) * gate_up).bfloat16(),
        (normal(experts, intermediate, hidden) * gate_up).bfloat16(),
        (normal(experts, hidden, intermediate) * down).bfloat16(),
    ]
    dy = normal(tokens, hidden).bfloat16()
    pinned = None
    step = torch.tensor(limit / hidden).bfloat16().double().item()
    if tokens > 1 and step * hidden == limit:
        signs = torch.where(normal(hidden) < 0, -1.0, 1.0)
        x[1] = signs.bfloat16()
        expert = int(ids[1, 0])
        row = (signs * step).bfloat16()
        routed[0][expert, 0] = row
        routed[1][expert, 0] = row
        routed[1][expert, 1] = -row
        pinned = (1, expert)
    return dict(x=x, dy=dy, ids=ids, weights=weights.float().contiguous(), shared=shared, routed=routed, pinned=pinned)
