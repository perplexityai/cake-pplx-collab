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

Checks, launches and autograd nodes of the Cake ratio-2 gated KV compression
(width 512; SM100 / SM103 / SM107).

The registry (:mod:`.cake_jit`) is data plus the host plan: ``MODULES`` holds
one record per program (``pair_table``, ``fwd`` on SM100 / SM103,
``fwd_sm_107a`` on SM107, ``bwd``; the ``build.json`` entries are
``<library>_<arch>``), :func:`.cake_jit.launch_grid` the grids and
:func:`.cake_jit.run_module` the launch of a role: the record serving it on
the device's architecture (:func:`.cake_jit.role_record`).  :mod:`.cake_build` serves a
library from a prebuilt ``aot/`` file when a builder compiled it and otherwise
through the package's JIT build shim (``_cake_pplx_jit.py``) from the sources
and flags of ``build.json``.

A forward call runs ``pair_table`` (the exclusive prefix of the pair counts,
int32 ``[S + 1]``) and ``fwd`` (one output row per pair); a backward call runs
``pair_table`` (unless the caller passes the table as ``pair_offsets``) and
``bwd`` (the ``dC`` / ``dZ`` rows of every pair's two tokens and the zero rows
of the odd-length documents' tail tokens).  Every launch is a plain
stream-ordered launch on the current torch stream of the operands' device.
Nothing here reads an environment variable, reads device memory or
synchronizes: grids and buffer sizes come from tensor shapes and the caller's
pair count ``G``, never from ``cu_seqlens``.

This is the deterministic package of its family, and the family's only
package: every output element is one fixed chain of FP32 operations on its own
pair or token, with no reduction across threads, CTAs or calls and no atomic,
so every output repeats bit for bit.  No entry alerts under
``torch.use_deterministic_algorithms(True)``.
"""

from __future__ import annotations

import operator
from typing import Any, Optional

import torch
from torch.autograd.function import once_differentiable

from . import cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeGatedKvCompressionDetCzFunction",
    "CakeGatedKvCompressionDetFunction",
    "arch_for_capability",
    "cake_gated_kv_compression_det",
    "cake_gated_kv_compression_det_cz",
    "cake_gated_kv_compression_det_train_backward",
    "cake_gated_kv_compression_det_train_forward",
    "cake_gated_kv_compression_det_train_memory_bytes",
    "device_arch",
    "is_cake_gated_kv_compression_det_train_supported",
    "launch_bwd",
    "launch_fwd",
    "launch_pair_table",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 / GB300 (``sm_103a``) and R200
# (``sm_107a``), each compiled with its exact ``-gencode`` flag set.  No other architecture has a build.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)
SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(sorted(ARCH_BY_CAPABILITY))
WIDTH = cake_jit.WIDTH
CZ_WIDTH = cake_jit.CZ_WIDTH

arch_for_capability = cake_jit.arch_for_capability
device_arch = cake_jit.device_arch

_F32 = torch.float32
_BF16 = torch.bfloat16
_I32 = torch.int32
_ALIGN = cake_jit.ALIGNMENT_BYTES
_LD = cake_jit.ROW_STRIDE_MULTIPLE


def is_cake_gated_kv_compression_det_train_supported(device: Any) -> bool:
    """Whether ``device`` is a CUDA device of a delivered architecture (SM100, SM103 or SM107)."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return arch_for_capability(tuple(torch.cuda.get_device_capability(index))) is not None


# ----------------------------------------------------------------------------------------------------------------
# Checks (dtypes, shapes, strides, alignment, devices and overlaps only: nothing reads tensor contents)


def _tensor(t: Any, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
    return t


def _same_device(t: torch.Tensor, name: str, device: torch.device) -> None:
    if t.device != device:
        raise ValueError(f"{name} must be on {device}, the device of the operands, got {t.device}")


def _dtype(t: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if t.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {t.dtype}")


def _aligned(t: torch.Tensor, name: str, key: str) -> None:
    alignment = _ALIGN[key]
    if t.numel() and t.data_ptr() % alignment:
        raise ValueError(f"{name} must start on a {alignment}-byte boundary, got address {t.data_ptr():#x}")


def _row_strided(t: torch.Tensor, name: str, width: int, key: str) -> int:
    """The row stride (elements) of an FP32 ``[rows, width]`` operand the kernels read or write in place: a unit
    feature stride, a row stride that is a multiple of :data:`.cake_jit.ROW_STRIDE_MULTIPLE` and at least ``width``
    (rows do not overlap), a base aligned for the kernels' 256-bit accesses.  ``ValueError`` otherwise; no copy."""
    if t.dim() != 2 or t.shape[1] != width:
        raise ValueError(f"{name} must be [T, {width}], got {tuple(t.shape)}")
    rows = int(t.shape[0])
    if rows == 0:
        return width
    if t.stride(1) != 1:
        raise ValueError(f"{name} must have a unit feature stride, got strides {tuple(t.stride())}")
    ld = int(t.stride(0)) if rows > 1 else width
    if ld < width or ld % _LD:
        raise ValueError(
            f"{name} must have a row stride that is a multiple of {_LD} elements and at least {width}, got {ld}"
        )
    _aligned(t, name, key)
    return ld


def _dense(t: torch.Tensor, name: str, rows: Optional[int], key: str) -> int:
    """The rows of a contiguous BF16 ``[rows, 512]`` operand (``out``, ``dout``); ``ValueError`` otherwise."""
    if t.dim() != 2 or t.shape[1] != WIDTH or (rows is not None and t.shape[0] != rows):
        wanted = f"[{rows}, {WIDTH}]" if rows is not None else f"[G, {WIDTH}]"
        raise ValueError(f"{name} must be {wanted}, got {tuple(t.shape)}")
    if t.numel() and not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(t.stride())} for shape {tuple(t.shape)}")
    _aligned(t, name, key)
    return int(t.shape[0])


def _table(t: torch.Tensor, name: str, entries: Optional[int]) -> int:
    """The entry count of a contiguous int32 1-D table (``cu_seqlens``, ``pair_offsets``); ``ValueError``
    otherwise."""
    if t.dim() != 1 or t.numel() < 1 or (entries is not None and t.numel() != entries):
        wanted = f"[{entries}]" if entries is not None else "[S + 1] with S >= 0"
        raise ValueError(f"{name} must be a 1-D int32 table {wanted}, got shape {tuple(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tuple(t.stride())}")
    _aligned(t, name, "cu_seqlens")
    return int(t.numel())


def _count(value: Any, name: str) -> int:
    """A host integer argument (``num_pairs``): a Python or NumPy integer, never a tensor (reading a device value
    would synchronize)."""
    if isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Python int (a tensor would need a device-to-host read)")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}") from None


def _extent(t: torch.Tensor) -> tuple[int, int]:
    """``[lo, hi)``: the byte range of the elements ``t`` can reach."""
    size = t.element_size()
    lo = t.data_ptr()
    hi = lo + size
    for n, s in zip(t.shape, t.stride()):
        hi += (int(n) - 1) * int(s) * size
    return lo, hi


def _rows(t: torch.Tensor) -> Optional[tuple[int, Optional[int], int]]:
    """``(rows, row stride, width)`` in elements of a view of at most two dimensions whose rows are dense (a unit
    inner stride); the row stride is ``None`` for a single row (any stride describes it).  ``None`` otherwise."""
    if t.dim() == 1:
        return (1, None, int(t.shape[0])) if t.stride(0) == 1 or t.shape[0] == 1 else None
    if t.dim() == 2 and (t.stride(1) == 1 or t.shape[1] == 1):
        rows = int(t.shape[0])
        return rows, (int(t.stride(0)) if rows > 1 else None), int(t.shape[1])
    return None


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` may share a byte.  Exact for two views of dense rows (at most two dimensions, a unit
    inner stride) with one row stride, a single row taking the other's (the two column halves of one ``[T, 1024]``
    buffer share none); otherwise any overlap of their byte ranges counts."""
    if a.numel() == 0 or b.numel() == 0:
        return False
    a0, a1 = _extent(a)
    b0, b1 = _extent(b)
    if a1 <= b0 or b1 <= a0:
        return False
    ra, rb = _rows(a), _rows(b)
    if ra is not None and rb is not None and a.element_size() == b.element_size():
        strides = {stride for stride in (ra[1], rb[1]) if stride is not None}
        size = a.element_size()
        if not strides:
            return True  # two single rows whose byte ranges meet
        if len(strides) == 1:
            row = strides.pop() * size
            wa, wb = ra[2] * size, rb[2] * size
            if row >= max(wa, wb):
                # Row i of a is [a0 + i row, + wa), row k of b is [b0 + k row, + wb): they meet when
                # -wb < (b0 - a0) + (k - i) row < wa for some i < rows(a), k < rows(b).
                d = b0 - a0
                low = max(-(ra[0] - 1), (-wb - d) // row + 1)
                high = min(rb[0] - 1, -((d - wa) // row) - 1)
                return low <= high
    return True


def _disjoint(written: tuple[str, torch.Tensor], others: tuple[tuple[str, Optional[torch.Tensor]], ...]) -> None:
    name, t = written
    for other_name, other in others:
        if other is not None and _overlaps(t, other):
            raise ValueError(f"{name} must not overlap {other_name}")


def _token_operands(
    c: Any, z: Any, cu_seqlens: Any, device: Optional[torch.device]
) -> tuple[torch.device, int, int, int, int]:
    """``(device, T, S, c_ld, z_ld)`` of valid ``c``, ``z`` and ``cu_seqlens``; ``ValueError`` / ``TypeError``
    otherwise."""
    _tensor(c, "c")
    _tensor(z, "z")
    _tensor(cu_seqlens, "cu_seqlens")
    if device is None:
        if c.device.type != "cuda":
            raise ValueError(f"c must be on a CUDA device, got {c.device}")
        device = c.device
    for name, t in (("c", c), ("z", z), ("cu_seqlens", cu_seqlens)):
        _same_device(t, name, device)
    _dtype(c, "c", _F32)
    _dtype(z, "z", _F32)
    _dtype(cu_seqlens, "cu_seqlens", _I32)
    if c.dim() != 2 or z.dim() != 2 or c.shape[0] != z.shape[0]:
        raise ValueError(f"c and z must be [T, {WIDTH}] with one T, got {tuple(c.shape)} and {tuple(z.shape)}")
    c_ld = _row_strided(c, "c", WIDTH, "c")
    z_ld = _row_strided(z, "z", WIDTH, "z")
    T = int(c.shape[0])
    if T > cake_jit.MAX_TOKENS:
        raise ValueError(f"T must be at most {cake_jit.MAX_TOKENS} (int32 cu_seqlens), got {T}")
    S = _table(cu_seqlens, "cu_seqlens", None) - 1
    if S > cake_jit.MAX_DOCS:
        raise ValueError(f"S must be at most {cake_jit.MAX_DOCS}, got {S}")
    return device, T, S, c_ld, z_ld


def _pair_count(num_pairs: Any, out: Optional[torch.Tensor]) -> int:
    """``G``: ``num_pairs``, or the rows of a caller ``out``; both given must agree; neither is a ``ValueError``
    (``G`` comes from the caller, never from ``cu_seqlens``)."""
    G = None if num_pairs is None else _count(num_pairs, "num_pairs")
    if out is not None:
        _tensor(out, "out")
        if out.dim() != 2:
            raise ValueError(f"out must be [G, {WIDTH}], got {tuple(out.shape)}")
        rows = int(out.shape[0])
        if G is not None and G != rows:
            raise ValueError(f"num_pairs = {G} disagrees with the {rows} rows of out")
        G = rows
    if G is None:
        raise ValueError(
            "pass num_pairs (the sum over documents of floor(length / 2)) or a preallocated out: the host never "
            "reads cu_seqlens"
        )
    if not 0 <= G <= cake_jit.MAX_PAIRS:
        raise ValueError(f"num_pairs must be in [0, {cake_jit.MAX_PAIRS}], got {G}")
    return G


# ----------------------------------------------------------------------------------------------------------------
# Launchers: caller-owned buffers, the current stream, no allocation, no check of the operands' contents


def launch_pair_table(*, cu_seqlens: torch.Tensor, cu_pairs: torch.Tensor) -> None:
    """Launch ``pair_table``: ``cu_pairs[0] = 0``, ``cu_pairs[s + 1] = cu_pairs[s] + (cu_seqlens[s + 1] -
    cu_seqlens[s]) // 2`` (int32 ``[S + 1]``, ``S >= 1``)."""
    S = int(cu_seqlens.numel()) - 1
    values = {"cu_seqlens": cu_seqlens, "cu_pairs": cu_pairs, "num_docs": S}
    cake_jit.run_module("pair_table", values, rows=1, device_index=cu_seqlens.device.index)


def launch_fwd(
    *,
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_pairs: torch.Tensor,
    out: torch.Tensor,
    c_ld: int,
    z_ld: int,
) -> None:
    """Launch the ``fwd`` role (record ``fwd`` on SM100 / SM103, ``fwd_sm_107a`` on SM107): ``out [G, 512]`` (BF16)
    from the FP32 ``c`` and ``z`` rows (row strides ``c_ld`` and ``z_ld`` elements), ``cu_seqlens`` and its pair table
    ``cu_pairs`` (``S >= 1``, ``G >= 1``)."""
    S, G = int(cu_seqlens.numel()) - 1, int(out.shape[0])
    values = {
        "c": c,
        "z": z,
        "cu_seqlens": cu_seqlens,
        "cu_pairs": cu_pairs,
        "out": out,
        "num_docs": S,
        "G": G,
        "c_ld": int(c_ld),
        "z_ld": int(z_ld),
    }
    cake_jit.run_module("fwd", values, rows=G, device_index=out.device.index)


def launch_bwd(
    *,
    dout: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_pairs: torch.Tensor,
    dc: torch.Tensor,
    dz: torch.Tensor,
    c_ld: int,
    z_ld: int,
    dc_ld: int,
    dz_ld: int,
) -> None:
    """Launch ``bwd``: every ``dc`` and ``dz`` row (FP32, row strides ``dc_ld`` and ``dz_ld``) of a token below
    ``cu_seqlens[S]`` from ``dout [G, 512]`` (BF16), ``c``, ``z``, ``cu_seqlens`` and ``cu_pairs`` (``S >= 1``,
    ``G >= 1``), on ``G + S`` virtual rows."""
    S, G = int(cu_seqlens.numel()) - 1, int(dout.shape[0])
    values = {
        "dout": dout,
        "c": c,
        "z": z,
        "cu_seqlens": cu_seqlens,
        "cu_pairs": cu_pairs,
        "dc": dc,
        "dz": dz,
        "num_docs": S,
        "G": G,
        "c_ld": int(c_ld),
        "z_ld": int(z_ld),
        "dc_ld": int(dc_ld),
        "dz_ld": int(dz_ld),
    }
    cake_jit.run_module("bwd", values, rows=G + S, device_index=dout.device.index)


# ----------------------------------------------------------------------------------------------------------------
# Forward and backward


def _forward(
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_pairs: Any,
    out: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``(out, cu_pairs)`` of one forward call; ``cu_pairs`` is the pair table the call computed (``None`` when it
    launched nothing)."""
    device, T, S, c_ld, z_ld = _token_operands(c, z, cu_seqlens, None)
    G = _pair_count(num_pairs, out)
    if out is None:
        out = torch.empty((G, WIDTH), dtype=_BF16, device=device)
    else:
        _same_device(out, "out", device)
        _dtype(out, "out", _BF16)
        _dense(out, "out", G, "out")
        _disjoint(("out", out), (("c", c), ("z", z), ("cu_seqlens", cu_seqlens)))
    device_arch(device.index)  # RuntimeError on a device outside ARCHES
    if G == 0:
        return out, None
    if T == 0 or S == 0:
        # No token, or no document: every row is past the pair count.
        out.zero_()
        return out, None
    cu_pairs = torch.empty(S + 1, dtype=_I32, device=device)
    launch_pair_table(cu_seqlens=cu_seqlens, cu_pairs=cu_pairs)
    launch_fwd(c=c, z=z, cu_seqlens=cu_seqlens, cu_pairs=cu_pairs, out=out, c_ld=c_ld, z_ld=z_ld)
    return out, cu_pairs


def cake_gated_kv_compression_det_train_forward(
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_pairs: Optional[int] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 ``out [G, 512]``: the softmax-weighted pair sums; see
    :func:`cake_pplx.cake_gated_kv_compression_det_train.cake_gated_kv_compression_det_train_forward`."""
    return _forward(c, z, cu_seqlens, num_pairs, out)[0]


def cake_gated_kv_compression_det_train_backward(
    dout: torch.Tensor,
    c: torch.Tensor,
    z: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    dc: Optional[torch.Tensor] = None,
    dz: Optional[torch.Tensor] = None,
    pair_offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``(dc, dz)``, ``[T, 512]`` each, every row written; see
    :func:`cake_pplx.cake_gated_kv_compression_det_train.cake_gated_kv_compression_det_train_backward`."""
    _tensor(dout, "dout")
    if dout.device.type != "cuda":
        raise ValueError(f"dout must be on a CUDA device, got {dout.device}")
    device = dout.device
    device, T, S, c_ld, z_ld = _token_operands(c, z, cu_seqlens, device)
    _dtype(dout, "dout", _BF16)
    G = _dense(dout, "dout", None, "dout")
    if G > cake_jit.MAX_PAIRS:
        raise ValueError(f"dout must have at most {cake_jit.MAX_PAIRS} rows, got {G}")
    grads = []
    for name, given in (("dc", dc), ("dz", dz)):
        if given is None:
            given = torch.empty((T, WIDTH), dtype=_F32, device=device)
        else:
            _tensor(given, name)
            _same_device(given, name, device)
            _dtype(given, name, _F32)
            if given.dim() != 2 or given.shape[0] != T:
                raise ValueError(f"{name} must be [{T}, {WIDTH}] (T of c), got {tuple(given.shape)}")
        grads.append((given, _row_strided(given, name, WIDTH, name)))
    (dc, dc_ld), (dz, dz_ld) = grads
    if pair_offsets is not None:
        _tensor(pair_offsets, "pair_offsets")
        _same_device(pair_offsets, "pair_offsets", device)
        _dtype(pair_offsets, "pair_offsets", _I32)
        _table(pair_offsets, "pair_offsets", S + 1)
    reads = (("dout", dout), ("c", c), ("z", z), ("cu_seqlens", cu_seqlens), ("pair_offsets", pair_offsets))
    _disjoint(("dc", dc), (("dz", dz), *reads))
    _disjoint(("dz", dz), reads)
    device_arch(device.index)  # RuntimeError on a device outside ARCHES
    if T == 0:
        return dc, dz
    if G == 0 or S == 0:
        # No pair: every token is a tail token.
        dc.zero_()
        dz.zero_()
        return dc, dz
    cu_pairs = pair_offsets
    if cu_pairs is None:
        cu_pairs = torch.empty(S + 1, dtype=_I32, device=device)
        launch_pair_table(cu_seqlens=cu_seqlens, cu_pairs=cu_pairs)
    launch_bwd(
        dout=dout,
        c=c,
        z=z,
        cu_seqlens=cu_seqlens,
        cu_pairs=cu_pairs,
        dc=dc,
        dz=dz,
        c_ld=c_ld,
        z_ld=z_ld,
        dc_ld=dc_ld,
        dz_ld=dz_ld,
    )
    return dc, dz


def cake_gated_kv_compression_det_train_memory_bytes(num_tokens: int, num_docs: int) -> dict[str, int]:
    """The workspace and the state retained for the backward, in bytes, for ``T = num_tokens`` and ``S = num_docs``;
    see :func:`cake_pplx.cake_gated_kv_compression_det_train.cake_gated_kv_compression_det_train_memory_bytes`."""
    T = _count(num_tokens, "num_tokens")
    S = _count(num_docs, "num_docs")
    if T < 0 or S < 0:
        raise ValueError(f"num_tokens and num_docs must be at least 0, got {T} and {S}")
    table = 4 * (S + 1)
    return {
        "forward_workspace": table,
        "backward_workspace": table,
        "retained_for_backward": table,
        "retained_inputs": 2 * WIDTH * 4 * T + table,
    }


# ----------------------------------------------------------------------------------------------------------------
# Autograd


def _incoming(dout: torch.Tensor) -> torch.Tensor:
    """The incoming gradient as the backward kernel reads it: contiguous and 16-byte aligned (an expanded or offset
    view is copied once)."""
    dout = dout.contiguous()
    if dout.numel() and dout.data_ptr() % _ALIGN["dout"]:
        dout = dout.clone(memory_format=torch.contiguous_format)
    return dout


def _halves(cz: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """``(C, Z)``: the column halves of a valid FP32 ``cz [T, 1024]``; ``ValueError`` / ``TypeError`` otherwise."""
    _tensor(cz, "cz")
    if cz.device.type != "cuda":
        raise ValueError(f"cz must be on a CUDA device, got {cz.device}")
    _dtype(cz, "cz", _F32)
    _row_strided(cz, "cz", CZ_WIDTH, "c")
    return cz[:, :WIDTH], cz[:, WIDTH:]


class CakeGatedKvCompressionDetFunction(torch.autograd.Function):
    """Autograd node of :func:`cake_gated_kv_compression_det`: forward
    :func:`cake_gated_kv_compression_det_train_forward`, backward :func:`cake_gated_kv_compression_det_train_backward`
    (one launch writes ``dC`` and ``dZ``).  The node saves ``c``, ``z``, ``cu_seqlens`` and its forward's pair table
    (int32 ``[S + 1]``) only: no activation and no probability, and the backward reuses the table (no pair-table
    launch).  Under ``torch.utils.checkpoint(..., use_reentrant=False)`` the recompute gives the same bits."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, c: torch.Tensor, z: torch.Tensor, cu_seqlens: torch.Tensor, num_pairs: int
    ) -> torch.Tensor:
        out, cu_pairs = _forward(c, z, cu_seqlens, num_pairs, None)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(c, z, cu_seqlens, cu_pairs)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        if dout is None:
            return None, None, None, None
        c, z, cu_seqlens, cu_pairs = ctx.saved_tensors
        dc, dz = cake_gated_kv_compression_det_train_backward(_incoming(dout), c, z, cu_seqlens, pair_offsets=cu_pairs)
        need_c, need_z = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        return (dc if need_c else None), (dz if need_z else None), None, None


class CakeGatedKvCompressionDetCzFunction(torch.autograd.Function):
    """Autograd node of :func:`cake_gated_kv_compression_det_cz`: ``cz [T, 1024]`` holds C in columns ``[0, 512)``
    and Z in ``[512, 1024)``.  Its backward allocates one FP32 ``[T, 1024]`` gradient and one launch writes ``dC``
    and ``dZ`` into its two halves: no slice-backward zero fill or copy.  The node saves ``cz``, ``cu_seqlens`` and its
    forward's pair table only."""

    @staticmethod
    def forward(ctx: Any, cz: torch.Tensor, cu_seqlens: torch.Tensor, num_pairs: int) -> torch.Tensor:  # type: ignore[override]
        c, z = _halves(cz)
        out, cu_pairs = _forward(c, z, cu_seqlens, num_pairs, None)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(cz, cu_seqlens, cu_pairs)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dout: Optional[torch.Tensor]) -> tuple[Optional[torch.Tensor], ...]:  # type: ignore[override]
        if dout is None:
            return None, None, None
        cz, cu_seqlens, cu_pairs = ctx.saved_tensors
        grad = torch.empty((cz.shape[0], CZ_WIDTH), dtype=_F32, device=cz.device)
        cake_gated_kv_compression_det_train_backward(
            _incoming(dout),
            cz[:, :WIDTH],
            cz[:, WIDTH:],
            cu_seqlens,
            dc=grad[:, :WIDTH],
            dz=grad[:, WIDTH:],
            pair_offsets=cu_pairs,
        )
        return grad, None, None


def cake_gated_kv_compression_det(
    c: torch.Tensor, z: torch.Tensor, cu_seqlens: torch.Tensor, num_pairs: int
) -> torch.Tensor:
    """Autograd-enabled forward on separate ``c`` and ``z``; see
    :func:`cake_pplx.cake_gated_kv_compression_det_train.cake_gated_kv_compression_det`."""
    return CakeGatedKvCompressionDetFunction.apply(c, z, cu_seqlens, num_pairs)


def cake_gated_kv_compression_det_cz(cz: torch.Tensor, cu_seqlens: torch.Tensor, num_pairs: int) -> torch.Tensor:
    """Autograd-enabled forward on one ``cz [T, 1024]``; see
    :func:`cake_pplx.cake_gated_kv_compression_det_train.cake_gated_kv_compression_det_cz`."""
    return CakeGatedKvCompressionDetCzFunction.apply(cz, cu_seqlens, num_pairs)
