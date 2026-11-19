# Copyright 2026 Cursor Research
# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch MoK MXFP8 quantization for caller-prequantized routed weights.

Recipe (MoK ``mxfp8_quantize``): 32 consecutive elements along the quantized
(K, last) dimension share one UE8M0 scale byte,
``round_up(max(amax * 0.002232142857f, 1e-12f))`` saturated to 254; data is
``cvt.rn.satfinite.e4m3(x * 2**(127 - byte))``. Scale tiles cover 128 rows x
128 K, stored as ``[rows // 128, K // 128, 32, 16]`` bytes with byte
``(row % 32) * 16 + (row // 32) * 4 + k_block`` (tensor-core scale layout);
experts are concatenated along the first dimension.

Every layout is quantized with PyTorch operations (no Triton dependency) on
the BF16 input, without an FP32 copy: the block maximum of ``|x|`` is exact in
BF16, and ``x * 2**(127 - byte)`` is exact in BF16 (a power-of-two factor of a
BF16 value, at most 448 in magnitude; a product below the BF16 normal range
is below the smallest E4M3 value either way), so the bytes equal those of the
FP32 form of the recipe. The transposed layout takes its 32-element blocks
along M of the input as stored and transposes the E4M3 result.
"""

import functools

import torch

_BLOCK = 32
# The FP32 ``1 / 448`` factor of the recipe.
_SCALE_FACTOR = 0.002232142857
# Temporaries of one quantization pass stay below this many bytes (at most four
# per element): experts are quantized together in chunks of this size.
_CHUNK_BYTES = 1 << 30


@functools.cache
def _scale_factor(device):
    """The FP32 ``1 / 448`` factor of the recipe, created once per device."""
    return torch.tensor(_SCALE_FACTOR, dtype=torch.float32, device=device)


def _e8m0_round_up(scale):
    """FP32 -> UE8M0 byte, rounding toward +inf and saturating to 254."""
    bits = scale.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    byte = exponent + (mantissa != 0).to(torch.int32)
    return byte.clamp(max=254).to(torch.uint8)


def _block_scales(blocks, dim):
    """UE8M0 bytes of the 32-element blocks of BF16 ``blocks`` along ``dim`` and
    the BF16 factors ``2**(127 - byte)`` (0 for byte 254, which only a NaN or
    infinite block takes). The block maximum of ``|x|`` comes from the block's
    minimum and maximum; a NaN propagates."""
    low, high = torch.aminmax(blocks, dim=dim)
    amax = torch.maximum(high, low.neg())
    scale = torch.clamp(amax.float() * _scale_factor(blocks.device), min=1e-12)
    byte = _e8m0_round_up(scale)
    inverse = ((254 - byte.to(torch.int32)) << 23).view(torch.float32)
    return byte, inverse.to(torch.bfloat16)


def _scaled_codes(blocks, inverse):
    """E4M3 codes of BF16 ``blocks`` times the broadcast BF16 factors."""
    scaled = blocks * inverse
    scaled.clamp_(-448.0, 448.0)
    return scaled.to(torch.float8_e4m3fn)


def _scale_tiles(byte):
    """[E, R, K/32] bytes -> [E * R/128, K/128, 32, 16] tensor-core scale tiles."""
    experts, rows, kb = byte.shape
    tiles = byte.reshape(
        experts, rows // 128, 4, 32, kb // 4, 4
    )  # [e, rb, row//32, row%32, kt, kb%4]
    return tiles.permute(0, 1, 4, 3, 2, 5).reshape(
        experts * rows // 128, kb // 4, 32, 16
    )


def _quantize_layout_torch(matrices, transposed):
    """One layout of ``[E, M, N]`` matrices with PyTorch operations: experts are
    quantized together in chunks that bound the temporaries to
    ``_CHUNK_BYTES``; the result does not depend on the chunking. The
    transposed layout quantizes 32-element blocks along M of the input as
    stored and transposes the E4M3 codes and the scale bytes."""
    experts, rows, cols = matrices.shape
    chunk = max(1, _CHUNK_BYTES // (rows * cols * 4))
    shape = (experts, cols, rows) if transposed else (experts, rows, cols)
    data = torch.empty(shape, dtype=torch.float8_e4m3fn, device=matrices.device)
    scales = []
    for start in range(0, experts, chunk):
        part = matrices[start : start + chunk]
        count = part.shape[0]
        if transposed:
            blocks = part.reshape(count, rows // _BLOCK, _BLOCK, cols)
            byte, inverse = _block_scales(blocks, 2)
            codes = _scaled_codes(blocks, inverse[:, :, None, :])
            codes = codes.reshape(count, rows, cols)
            data[start : start + chunk] = codes.transpose(1, 2)
            byte = byte.transpose(1, 2).contiguous()
        else:
            blocks = part.reshape(count, rows, cols // _BLOCK, _BLOCK)
            byte, inverse = _block_scales(blocks, 3)
            codes = _scaled_codes(blocks, inverse[..., None])
            data[start : start + chunk] = codes.reshape(count, rows, cols)
        scales.append(_scale_tiles(byte))
    return data, torch.cat(scales)


def mxfp8_quantize(x, return_normal=True, return_transposed=True):
    """MoK ``mxfp8_quantize`` for BF16 ``[M, N]`` or ``[E, M, N]`` matrices.

    Returns ``(fp8, scales, fp8_t, scales_t)``; entries not requested are
    ``None``. ``fp8`` is E4M3 with the input shape and ``scales`` is uint8
    ``[E * M // 128, N // 128, 32, 16]``; the transposed pair quantizes
    ``x.transpose(-1, -2)`` the same way. M and N must be multiples of 128.
    Each requested layout is quantized with PyTorch operations on the BF16
    input, experts in chunks that bound the temporaries; the bytes do not
    depend on the chunking.
    """
    if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16:
        raise ValueError("x must be a BF16 tensor")
    if x.ndim not in (2, 3) or any(size <= 0 for size in x.shape):
        raise ValueError("x must have positive shape (M, N) or (E, M, N)")
    if x.shape[-2] % 128 or x.shape[-1] % 128:
        raise ValueError("x M and N dimensions must be divisible by 128")
    if type(return_normal) is not bool or type(return_transposed) is not bool:
        raise TypeError("return_normal and return_transposed must be booleans")
    if not return_normal and not return_transposed:
        raise ValueError("At least one quantized layout must be requested")
    matrices = x.unsqueeze(0) if x.ndim == 2 else x
    outputs = []
    for transposed in (False, True):
        if not (return_transposed if transposed else return_normal):
            outputs += [None, None]
            continue
        data, scales = _quantize_layout_torch(matrices, transposed)
        outputs += [data[0] if x.ndim == 2 else data, scales]
    return tuple(outputs)


def quantize_routed_weights(gate, up, down):
    """Prequantize BF16 routed expert weights in the MoK MXFP8 recipe.

    ``gate``/``up`` are ``[E, I, H]`` and ``down`` is ``[E, H, I]``. Returns
    ``(forward_weights, backward_weights)``: forward and context recompute use
    ``(data, scales)`` pairs for gate, up and down; backward uses
    ``(data, scales, data_t, scales_t)`` for gate and up and the transposed
    ``(data_t, scales_t)`` pair for down.
    """
    if (
        any(not isinstance(w, torch.Tensor) or w.ndim != 3 for w in (gate, up, down))
        or gate.shape != up.shape
        or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])
    ):
        raise ValueError("Expected gate/up [E, I, H] and down [E, H, I] weights")
    q_gate, q_up, q_down = (mxfp8_quantize(w) for w in (gate, up, down))
    forward = ((q_gate[0], q_gate[1]), (q_up[0], q_up[1]), (q_down[0], q_down[1]))
    backward = (q_gate, q_up, (q_down[2], q_down[3]))
    return forward, backward
