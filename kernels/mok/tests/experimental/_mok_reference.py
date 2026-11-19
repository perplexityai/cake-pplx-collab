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
    ``[T, topk]``), ``shared`` and ``routed`` (BF16, the package's shapes) and ``pinned`` (token, expert)."""
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=generator)

    x = randn(tokens, hidden)
    x = (x / x.square().mean(-1, keepdim=True).sqrt()).bfloat16()
    scores = F.softplus(randn(tokens, experts)).sqrt()
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
        (randn(intermediate, hidden) * gate_up).bfloat16(),
        (randn(intermediate, hidden) * gate_up).bfloat16(),
        (randn(hidden, intermediate) * down).bfloat16(),
    ]
    routed = [
        (randn(experts, intermediate, hidden) * gate_up).bfloat16(),
        (randn(experts, intermediate, hidden) * gate_up).bfloat16(),
        (randn(experts, hidden, intermediate) * down).bfloat16(),
    ]
    dy = randn(tokens, hidden).bfloat16()
    pinned = None
    step = torch.tensor(limit / hidden).bfloat16().double().item()
    if tokens > 1 and step * hidden == limit:
        signs = torch.where(randn(hidden) < 0, -1.0, 1.0)
        x[1] = signs.bfloat16()
        expert = int(ids[1, 0])
        row = (signs * step).bfloat16()
        routed[0][expert, 0] = row
        routed[1][expert, 0] = row
        routed[1][expert, 1] = -row
        pinned = (1, expert)
    return dict(x=x, dy=dy, ids=ids, weights=weights.float().contiguous(), shared=shared, routed=routed, pinned=pinned)
