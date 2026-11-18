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

Reference of the Engram row lookup tests and benchmark: the FP64 oracle of the feature request, the table builders and
the bitwise comparisons.  This is the only file of the family's tests that computes in FP64.

* :func:`round_fp64_to_bf16`, :func:`row_lookup_fwd_released` and :func:`row_lookup_fwd_fp64` are the request's CPU
  reference, verbatim: the FP64 product rounded once to BF16, and the release path (gather, FP32 multiply, BF16 cast).
* :func:`pair_bits` is the oracle of every (E4M3 code, E8M0 byte) pair as BF16 bits, from :func:`round_fp64_to_bf16`;
  :func:`oracle_bits` looks the rows of a table up through it on the table's device (no FP64 there), and
  :func:`oracle` runs :func:`row_lookup_fwd_fp64` itself on the CPU copy of the looked-up rows.
* :func:`exhaustive_bytes` is the 256-row table whose row ``b`` holds every E4M3 code ``0 .. 255`` in its columns and
  the E8M0 byte ``b`` in its 8 scales, so ``row_ids = arange(256)`` looks every pair up; :func:`exhaustive_report`
  judges such an output against the request's counts (64,770 non-NaN pairs, 60,198 exact for the scales 2^-117 to
  2^119, 568 FP32-subnormal products, 560 products past the FP32 range).
* :func:`make_table` holds one set of table bytes in either layout: two contiguous tensors, or the two views of one
  packed ``uint8 [R, 264]`` buffer.

NaN: the oracle's NaN and the kernel's NaN may differ in their payload bits, so every comparison takes the NaN
positions to be equal and the bits elsewhere to be equal (:func:`assert_same_bits`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch

BLOCK, ROW_DIM, N_COLS = 32, 256, 24
NUM_GROUPS = ROW_DIM // BLOCK  # 8
# A packed row: its ROW_DIM value bytes, then its NUM_GROUPS scale bytes.
PACKED_ROW_BYTES = ROW_DIM + NUM_GROUPS  # 264
LAYOUTS = ("contiguous", "packed")
# The canonical BF16 NaN of the comparisons.
NAN_BITS = 0x7FC0
# The request's counts of the exhaustive (E4M3 code, E8M0 byte) table.
EXHAUSTIVE_COUNTS = {
    "non_nan_pairs": 64_770,
    "mid_range_pairs": 60_198,
    "fp32_subnormal_pairs": 568,
    "fp32_overflow_pairs": 560,
}
# The E8M0 bytes of the scales 2^-117 .. 2^119 (byte b is 2^(b - 127)).
MID_RANGE_BYTES = (127 - 117, 127 + 119)


# ----------------------------------------------------------------------------------------------------------------
# The request's reference (CPU, FP64), verbatim


def round_fp64_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Single round-to-nearest-even from FP64 to BF16 (avoids double rounding via FP32)."""
    x32 = x.float()
    b = x32.to(torch.bfloat16)
    bits = x32.contiguous().view(torch.int32)
    fix = ((bits & 0xFFFF) == 0x8000) & (x32.double() != x)
    up = x.abs() > x32.abs().double()
    fixed = torch.where(up, (bits + 0x8000) & -65536, bits & -65536).view(torch.float32).to(torch.bfloat16)
    return torch.where(fix, fixed, b)


def row_lookup_fwd_released(weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    vals = weight_fp8[row_ids]
    scls = scale_e8m0[row_ids]
    out = vals.float().unflatten(-1, (NUM_GROUPS, BLOCK)) * scls.float().unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


def row_lookup_fwd_fp64(weight_fp8: torch.Tensor, scale_e8m0: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    vals = weight_fp8[row_ids].to(torch.float64)
    scls = scale_e8m0[row_ids].to(torch.float64)
    return round_fp64_to_bf16((vals.unflatten(-1, (NUM_GROUPS, BLOCK)) * scls.unsqueeze(-1)).flatten(-2))


# ----------------------------------------------------------------------------------------------------------------
# Tables


@dataclass
class Table:
    """One set of table bytes in one layout: ``weight`` ``float8_e4m3fn [R, 256]`` and ``scale`` ``float8_e8m0fnu
    [R, 8]``, two contiguous tensors (``packed`` None) or the two views of the packed ``uint8 [R, 264]`` buffer
    ``packed``."""

    layout: str
    weight: torch.Tensor
    scale: torch.Tensor
    packed: Optional[torch.Tensor] = None

    @property
    def rows(self) -> int:
        return int(self.weight.shape[0])


def make_table(weight_bytes: torch.Tensor, scale_bytes: torch.Tensor, layout: str, device) -> Table:
    """The table of ``weight_bytes`` (``uint8 [R, 256]``) and ``scale_bytes`` (``uint8 [R, 8]``) on ``device`` in
    ``layout`` (one of :data:`LAYOUTS`).  Both layouts hold the same bytes."""
    rows = weight_bytes.shape[0]
    assert tuple(weight_bytes.shape) == (rows, ROW_DIM) and weight_bytes.dtype == torch.uint8
    assert tuple(scale_bytes.shape) == (rows, NUM_GROUPS) and scale_bytes.dtype == torch.uint8
    if layout == "contiguous":
        weight = weight_bytes.to(device).contiguous().view(torch.float8_e4m3fn)
        scale = scale_bytes.to(device).contiguous().view(torch.float8_e8m0fnu)
        return Table(layout, weight, scale)
    if layout != "packed":
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    packed = torch.empty(rows, PACKED_ROW_BYTES, dtype=torch.uint8, device=device)
    packed[:, :ROW_DIM] = weight_bytes.to(device)
    packed[:, ROW_DIM:] = scale_bytes.to(device)
    weight = packed[:, :ROW_DIM].view(torch.float8_e4m3fn)
    scale = packed[:, ROW_DIM:].view(torch.float8_e8m0fnu)
    assert weight.stride() == (PACKED_ROW_BYTES, 1) and scale.stride() == (PACKED_ROW_BYTES, 1)
    return Table(layout, weight, scale, packed)


def empty_table(rows: int, layout: str, device) -> Table:
    """A table of ``rows`` rows in ``layout`` whose bytes are not initialized (for tables of gigabytes: the test
    writes the rows it looks up with :func:`write_rows`)."""
    if layout == "contiguous":
        weight = torch.empty(rows, ROW_DIM, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
        scale = torch.empty(rows, NUM_GROUPS, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)
        return Table(layout, weight, scale)
    if layout != "packed":
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    packed = torch.empty(rows, PACKED_ROW_BYTES, dtype=torch.uint8, device=device)
    return Table(
        layout, packed[:, :ROW_DIM].view(torch.float8_e4m3fn), packed[:, ROW_DIM:].view(torch.float8_e8m0fnu), packed
    )


def write_rows(table: Table, row_ids: torch.Tensor, weight_bytes: torch.Tensor, scale_bytes: torch.Tensor) -> None:
    """Write ``weight_bytes`` (``uint8 [n, 256]``) and ``scale_bytes`` (``uint8 [n, 8]``) into the rows ``row_ids``
    (int64 ``[n]``) of ``table``.  With a repeated id one of its rows is written; the oracle reads the table
    afterwards, so it sees the bytes the table holds."""
    device = table.weight.device
    table.weight.view(torch.uint8)[row_ids] = weight_bytes.to(device)
    table.scale.view(torch.uint8)[row_ids] = scale_bytes.to(device)


def table_bytes(table: Table) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weight_bytes, scale_bytes)``: the table's bytes as contiguous ``uint8 [R, 256]`` and ``uint8 [R, 8]``."""
    return table.weight.view(torch.uint8).contiguous(), table.scale.view(torch.uint8).contiguous()


def exhaustive_bytes() -> tuple[torch.Tensor, torch.Tensor]:
    """The exhaustive table's bytes (CPU): row ``b`` holds the E4M3 codes ``0 .. 255`` in its columns and the E8M0
    byte ``b`` in its 8 scales."""
    codes = torch.arange(256, dtype=torch.uint8)
    weight_bytes = codes.repeat(256, 1)
    scale_bytes = codes[:, None].repeat(1, NUM_GROUPS)
    return weight_bytes, scale_bytes


def random_bytes(rows: int, seed: int, *, nan_every: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random table bytes (CPU) from ``seed``: every E4M3 code (the NaN codes ``0x7F`` and ``0xFF`` included) and every
    finite E8M0 byte ``0 .. 254``.  With ``nan_every``: the rows ``r`` with ``r % nan_every == 0`` hold the NaN scale
    byte ``0xFF`` in group ``(r // nan_every) % 8``, and no other scale byte is ``0xFF``."""
    gen = torch.Generator().manual_seed(seed)
    weight_bytes = torch.randint(0, 256, (rows, ROW_DIM), generator=gen, dtype=torch.uint8)
    scale_bytes = torch.randint(0, 255, (rows, NUM_GROUPS), generator=gen, dtype=torch.uint8)
    if nan_every:
        marked = torch.arange(0, rows, nan_every)
        scale_bytes[marked, (marked // nan_every) % NUM_GROUPS] = 0xFF
    return weight_bytes, scale_bytes


def nan_scale_cells(rows: int, nan_every: int) -> list[tuple[int, int]]:
    """The ``(row, group)`` cells :func:`random_bytes` set to the NaN scale byte for ``nan_every``."""
    return [(r, (r // nan_every) % NUM_GROUPS) for r in range(0, rows, nan_every)]


def random_ids(rows: int, shape: tuple[int, ...], seed: int, device) -> torch.Tensor:
    """Contiguous int64 row ids of ``shape``, uniform over ``[0, rows)``, from ``seed``."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, rows, shape, generator=gen, dtype=torch.int64).to(device)


# ----------------------------------------------------------------------------------------------------------------
# Oracle


@functools.cache
def _pair_bits_cpu() -> torch.Tensor:
    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).to(torch.float64)
    scales = codes.view(torch.float8_e8m0fnu).to(torch.float64)
    bits = round_fp64_to_bf16(scales[:, None] * values[None, :]).view(torch.int16)
    return canonical_bits(bits)


def pair_bits(device=None) -> torch.Tensor:
    """``int16 [256, 256]``: entry ``[b, c]`` is the BF16 bits of E4M3 code ``c`` times E8M0 byte ``b``, the FP64
    product rounded once (:func:`round_fp64_to_bf16`); every NaN as :data:`NAN_BITS`."""
    bits = _pair_bits_cpu()
    return bits if device is None else bits.to(device)


def canonical_bits(bits: torch.Tensor) -> torch.Tensor:
    """BF16 ``bits`` (int16) with every NaN replaced by :data:`NAN_BITS`."""
    return torch.where((bits & 0x7FFF) > 0x7F80, torch.full_like(bits, NAN_BITS), bits)


def oracle_bits(table: Table, row_ids: torch.Tensor, *, chunk_rows: int = 1 << 17) -> torch.Tensor:
    """``int16 [*row_ids.shape, 256]`` on the table's device: the oracle's BF16 bits of the looked-up rows (NaN as
    :data:`NAN_BITS`), through :func:`pair_bits`, ``chunk_rows`` rows at a time."""
    table_pairs = pair_bits(table.weight.device)
    weight_bytes = table.weight.view(torch.uint8)
    scale_bytes = table.scale.view(torch.uint8)
    flat = row_ids.reshape(-1)
    out = torch.empty(flat.numel(), ROW_DIM, dtype=torch.int16, device=table.weight.device)
    for start in range(0, flat.numel(), chunk_rows):
        ids = flat[start : start + chunk_rows]
        codes = weight_bytes[ids].long()
        groups = scale_bytes[ids].long().repeat_interleave(BLOCK, dim=1)
        out[start : start + ids.numel()] = table_pairs[groups, codes]
    return out.view(*row_ids.shape, ROW_DIM)


def oracle(table: Table, row_ids: torch.Tensor) -> torch.Tensor:
    """BF16 ``[*row_ids.shape, 256]`` on the CPU: :func:`row_lookup_fwd_fp64` of the looked-up rows (gathered as
    bytes on the table's device, then copied to the CPU)."""
    flat = row_ids.reshape(-1)
    weight = table.weight.view(torch.uint8)[flat].cpu().view(torch.float8_e4m3fn)
    scale = table.scale.view(torch.uint8)[flat].cpu().view(torch.float8_e8m0fnu)
    local = torch.arange(flat.numel(), dtype=torch.int64).view(row_ids.shape)
    return row_lookup_fwd_fp64(weight, scale, local)


def assert_same_bits(got: torch.Tensor, want: torch.Tensor, label: str = "") -> int:
    """``got`` (BF16) holds ``want`` (BF16, or its int16 bits) bit for bit, NaN payloads aside: the same shape, NaN at
    the same elements and the same bits everywhere else.  Returns the number of non-NaN elements compared."""
    assert got.dtype == torch.bfloat16, (label, got.dtype)
    got_bits = got.view(torch.int16)
    want_bits = want.view(torch.int16) if want.dtype == torch.bfloat16 else want
    want_bits = want_bits.to(got_bits.device)
    assert tuple(got_bits.shape) == tuple(want_bits.shape), (label, tuple(got.shape), tuple(want_bits.shape))
    got_nan = (got_bits & 0x7FFF) > 0x7F80
    want_nan = (want_bits & 0x7FFF) > 0x7F80
    nan_mismatch = int((got_nan != want_nan).sum())
    assert nan_mismatch == 0, f"{label}: {nan_mismatch} elements NaN on one side only"
    differ = (got_bits != want_bits) & ~want_nan
    count = int(differ.sum())
    if count:
        where = differ.nonzero()[:8].tolist()
        examples = [(w, int(got_bits[tuple(w)]) & 0xFFFF, int(want_bits[tuple(w)]) & 0xFFFF) for w in where]
        raise AssertionError(f"{label}: {count} elements differ (index, got bits, want bits): {examples}")
    return int((~want_nan).sum())


# ----------------------------------------------------------------------------------------------------------------
# The exhaustive table


@functools.cache
def _exhaustive_classes() -> dict[str, torch.Tensor]:
    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(torch.float8_e4m3fn).to(torch.float64)
    scales = codes.view(torch.float8_e8m0fnu).to(torch.float64)
    product = scales[:, None] * values[None, :]  # [scale byte, code], exact in FP64
    nan_code = values.isnan()[None, :].expand(256, 256)
    nan_scale = scales.isnan()[:, None].expand(256, 256)
    non_nan = ~nan_code & ~nan_scale
    lo, hi = MID_RANGE_BYTES
    byte = codes.long()[:, None].expand(256, 256)
    classes = {
        "product": product,
        "non_nan": non_nan,
        "nan": ~non_nan,
        "mid_range": non_nan & (byte >= lo) & (byte <= hi),
        "fp32_subnormal": non_nan & (product != 0) & (product.abs() < 2.0**-126),
        "fp32_overflow": non_nan & (product.abs() >= 2.0**128),
    }
    counts = {
        "non_nan_pairs": int(classes["non_nan"].sum()),
        "mid_range_pairs": int(classes["mid_range"].sum()),
        "fp32_subnormal_pairs": int(classes["fp32_subnormal"].sum()),
        "fp32_overflow_pairs": int(classes["fp32_overflow"].sum()),
    }
    assert counts == EXHAUSTIVE_COUNTS, counts
    return classes


def exhaustive_report(out: torch.Tensor) -> dict[str, int]:
    """Judge ``out``, BF16 ``[256, 256]`` (row ``b``, column ``c``: E4M3 code ``c`` times E8M0 byte ``b``, the lookup
    of :func:`exhaustive_bytes` by ``arange(256)``), against the FP64 oracle.  Every count is over the pairs of one
    class; the request's acceptance holds when each ``*_equal`` / ``*_exact`` / ``*_inf`` count equals its class
    count, and ``nan_mismatches`` is 0.

    * ``non_nan_pairs`` / ``non_nan_equal``: the 64,770 pairs of a non-NaN code and a finite byte, and those whose
      bits equal the FP64 product rounded once to BF16;
    * ``mid_range_pairs`` / ``mid_range_exact``: the 60,198 of them with a scale in 2^-117 .. 2^119, and those whose
      value equals the exact product;
    * ``fp32_subnormal_pairs`` / ``fp32_subnormal_equal`` / ``fp32_subnormal_nonzero``: the 568 products below 2^-126
      in magnitude, those with the oracle's bits, and those the oracle rounds to a non-zero BF16 (a flush to zero
      changes their bits);
    * ``fp32_overflow_pairs`` / ``fp32_overflow_inf``: the 560 products of magnitude 2^128 or more, and those that are
      ``inf`` with the product's sign;
    * ``nan_pairs`` / ``nan_mismatches``: the pairs with a NaN code or the NaN byte ``0xFF``, and the elements that
      are NaN on one side only (over all 65,536 pairs).
    """
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (256, 256), (out.dtype, tuple(out.shape))
    classes = _exhaustive_classes()
    got_bits = out.view(torch.int16).cpu()
    want_bits = pair_bits()
    got_nan = (got_bits & 0x7FFF) > 0x7F80
    equal = got_bits == want_bits
    got_value = out.cpu().to(torch.float64)
    product = classes["product"]
    subnormal = classes["fp32_subnormal"]
    overflow = classes["fp32_overflow"]
    return {
        "non_nan_pairs": int(classes["non_nan"].sum()),
        "non_nan_equal": int((equal & classes["non_nan"]).sum()),
        "mid_range_pairs": int(classes["mid_range"].sum()),
        "mid_range_exact": int(((got_value == product) & classes["mid_range"]).sum()),
        "fp32_subnormal_pairs": int(subnormal.sum()),
        "fp32_subnormal_equal": int((equal & subnormal).sum()),
        "fp32_subnormal_nonzero": int(((want_bits & 0x7FFF) != 0)[subnormal].sum()),
        "fp32_overflow_pairs": int(overflow.sum()),
        "fp32_overflow_inf": int(
            (got_value.isinf() & (got_value.sign() == product.sign()) & overflow).sum()
        ),
        "nan_pairs": int(classes["nan"].sum()),
        "nan_mismatches": int((got_nan != classes["nan"]).sum()),
    }
