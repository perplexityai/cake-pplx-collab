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

Backend of :mod:`cake_pplx.cake_act_qdq_det_train`: the Cake block-scaled
quantize-dequantize of BF16 activations back to BF16 (SM100 / SM103 /
SM107), deterministic, as one self-contained package (these host checks and
launches, the ``cake_jit`` registry, the ``cake_build`` loader and the
generated CUDA sources under ``csrc/``).  The ``@cake_api`` entry points and
the user documentation live in :mod:`cake_pplx.cake_act_qdq_det_train`;
import that module.

Host design (every switch is a keyword; no environment variable):

- ``x`` is a contiguous BF16 CUDA tensor ``[..., D]`` with ``D`` a multiple
  of the group size ``G`` (16 or 32).  Each group of ``G`` consecutive
  elements is quantized with one scale and dequantized back to BF16.
- ``out`` is the output buffer: ``None`` allocates one, ``out=x`` works in
  place, any other buffer must not overlap ``x``.
- The FP32 scales ``[..., D / G]`` are written when ``return_scales`` is
  true or a ``scales`` buffer is given.
- A call runs one program of its route (``cake_jit.selected_module``) on the
  current torch stream: the small program (256-thread CTAs, tiles of 512
  vectors) when ``n_vecs = numel / 8`` lies in its range for the call's kind
  (out of place: 65,537 to 524,288 on sm_100a and sm_103a, to 131,072 for
  MXFP8 without the scales, 1 to 262,144 on sm_107a; in place: 1 to 262,144
  on sm_107a only); for MXFP8 without the scales on sm_100a and sm_103a, the
  wide program (128-thread CTAs, tiles of 512 vectors) above those sizes (out
  of place from 131,073 vectors, in place from 262,145, both to
  2,147,483,136); else the main
  program (64-thread CTAs, tiles of 128 vectors, 256 on sm_107a).  The grid
  is the ``tiles`` rule of ``cake_jit.launch_grid``: one CTA per tile once
  the tiles pass 2.5 waves of resident CTAs (32 per SM, 16 on sm_107a; 8 and
  4 for the small program), else the resident CTAs, at most one per tile,
  walk the tiles; the wide program always launches one CTA per tile.  It does not read device memory on the host and does not
  synchronize.  A call with no elements launches nothing.
- Every output repeats bit for bit for identical inputs: each group is
  quantized by its own threads, and nothing is summed (``README.md``).
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence, Tuple, Union

import torch

from . import cake_jit
from .cake_jit import ALIGNMENT_BYTES, ARCH_BY_CAPABILITY, ARCHES, FORMATS, GROUP_SIZES

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(sorted(ARCH_BY_CAPABILITY))
# The largest element count of one call: ``numel / 8`` vectors, at most
# ``cake_jit.MAX_VECTORS``.
MAX_ELEMENTS = cake_jit.VECTOR * cake_jit.MAX_VECTORS


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    """The architecture of one CUDA device index, queried once per process."""

    return cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def is_cake_act_qdq_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of a supported architecture."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    return _arch_of_index(_device_index(device)) is not None


def _group_size(format: str) -> int:
    if not isinstance(format, str) or format not in GROUP_SIZES:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
    return GROUP_SIZES[format]


def _check_shape(shape: Sequence[int], group: int) -> int:
    """Check one input shape and return its element count."""

    dims = [int(d) for d in shape]
    if not dims:
        raise ValueError("x must have at least one dimension")
    if any(d < 0 for d in dims):
        raise ValueError(f"x has a negative dimension: {tuple(dims)}")
    if dims[-1] % group:
        raise ValueError(
            f"the last dimension of x ({dims[-1]}) must be a multiple of the group size {group}"
        )
    numel = 1
    for d in dims:
        numel *= d
    if numel > MAX_ELEMENTS:
        raise ValueError(
            f"x has {numel} elements; one call takes at most {MAX_ELEMENTS} "
            "(the kernels count 8-element vectors in signed 32 bits)"
        )
    return numel


def _span(t: torch.Tensor) -> tuple[int, int]:
    start = t.data_ptr()
    return start, start + t.numel() * t.element_size()


def _overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    return a0 < b1 and b0 < a1


def _check_buffer(
    t: object,
    name: str,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Check a caller buffer the kernel writes."""

    if not isinstance(t, torch.Tensor) or not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype is not dtype:
        raise TypeError(f"{name} must have {str(dtype).replace('torch.', '')} dtype")
    if t.device != device:
        raise ValueError(f"{name} must be on the device of x ({device})")
    if tuple(t.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(t.shape)}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if torch.is_grad_enabled() and t.requires_grad:
        raise ValueError(
            f"{name} requires grad; the kernel writes it without autograd history"
        )
    return t


def _mark_written(t: torch.Tensor) -> None:
    # The kernel writes a caller buffer outside autograd: advance its version
    # counter, as a PyTorch in-place op does, so a stale saved tensor is caught.
    if not t.is_inference():
        torch.autograd.graph.increment_version(t)


def cake_act_qdq_det_train_forward_workspace_bytes(
    shape: Sequence[int],
    *,
    format: str,
) -> int:
    """The workspace of one forward call of an input of ``shape``: always 0
    bytes.  The call keeps no state between calls either.  Raises
    ``ValueError`` for a format or shape the forward refuses."""

    _check_shape(shape, _group_size(format))
    return 0


def cake_act_qdq_det_train_forward(
    x: torch.Tensor,
    *,
    format: str,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    return_scales: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Quantize ``x`` per group to ``format`` and dequantize it back to BF16.

    Returns ``y``, or ``(y, scales)`` when ``return_scales`` is true or a
    ``scales`` buffer is given.  ``y`` is ``out`` when given (``out=x``
    works in place).
    """

    group = _group_size(format)
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype is not torch.bfloat16:
        raise TypeError("x must have bfloat16 dtype")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    numel = _check_shape(x.shape, group)
    device_index = _device_index(x.device)
    arch = _arch_of_index(device_index)
    if arch is None:
        raise RuntimeError("activation QDQ requires SM100, SM103 or SM107")
    if x.data_ptr() % ALIGNMENT_BYTES:
        raise ValueError(f"x must be {ALIGNMENT_BYTES}-byte aligned")

    shape = tuple(x.shape)
    if out is None:
        y = torch.empty_like(x, memory_format=torch.contiguous_format)
    else:
        y = _check_buffer(out, "out", dtype=torch.bfloat16, shape=shape, device=x.device)
        if y.data_ptr() % ALIGNMENT_BYTES:
            raise ValueError(f"out must be {ALIGNMENT_BYTES}-byte aligned")
        # In place is out=x, or a buffer with x's exact bytes; any other
        # overlap is refused.
        if y.data_ptr() != x.data_ptr() and _overlaps(x, y):
            raise ValueError("out must be x itself (in place) or must not overlap x")

    scale_out = bool(return_scales) or scales is not None
    s = None
    if scale_out:
        scale_shape = (*shape[:-1], shape[-1] // group)
        if scales is None:
            s = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
        else:
            s = _check_buffer(
                scales, "scales", dtype=torch.float32, shape=scale_shape, device=x.device
            )
            if _overlaps(s, x) or _overlaps(s, y):
                raise ValueError("scales must not overlap x or out")

    if numel:
        n_vecs = numel // cake_jit.VECTOR
        name = cake_jit.selected_module(
            arch,
            format,
            scale_out=scale_out,
            n_vecs=n_vecs,
            in_place=y.data_ptr() == x.data_ptr(),
        )
        values = {"x": x, "y": y, "n_vecs": n_vecs}
        if scale_out:
            values["scales"] = s
        cake_jit.run_module(name, values, device_index=device_index)
        if out is not None:
            _mark_written(y)
        if scales is not None:
            _mark_written(s)
    return (y, s) if scale_out else y


__all__ = [
    "ALIGNMENT_BYTES",
    "ARCHES",
    "FORMATS",
    "GROUP_SIZES",
    "MAX_ELEMENTS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "cake_act_qdq_det_train_forward",
    "cake_act_qdq_det_train_forward_workspace_bytes",
    "is_cake_act_qdq_det_train_supported",
]
