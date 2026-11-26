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

Cake activation quantize-dequantize (QDQ) training forward (SM100 / SM103 /
SM107), deterministic.

Each group of ``G`` consecutive elements of the last dimension of a BF16
tensor is quantized to a block-scaled format with one scale and dequantized
back to BF16.  Three formats:

- ``"fp4_e4m3_g16"``: E2M1 values, an E4M3 scale per 16 elements;
- ``"mxfp4_e8m0_g32"``: E2M1 values, an E8M0 (power-of-two) scale per 32
  elements;
- ``"mxfp8_e8m0_g32"``: E4M3 values, an E8M0 scale per 32 elements.

The outputs equal the FP64 rule bit for bit (true division, round to nearest
even, signed zeros, saturation and the amax floors); the backend
``README.md`` gives the rule and why the FP32 and BF16 steps reach it.
Every output repeats bit for bit for identical inputs: each group is
quantized on its own, and nothing is summed.

Naming rules:

1. The mode is part of every name.  Every public name of this module carries
   ``det``; the module constants ``ARCHES``,
   ``SUPPORTED_COMPUTE_CAPABILITIES``, ``FORMATS`` and ``GROUP_SIZES`` are
   exempt.  The backend package is
   ``cake_pplx._backends.cake_act_qdq_det_train``, imported on first use.
2. No entry point takes a ``deterministic`` keyword.  No call alerts under
   ``torch.use_deterministic_algorithms(True)``.
3. The family has no ``nondet`` module: no step of the op has an order to
   relax.  It has no KCP module: the op works on each token alone.

Usage::

    y = cake_act_qdq_det_train_forward(x, format="mxfp8_e8m0_g32")
    y, scales = cake_act_qdq_det_train_forward(
        x, format="fp4_e4m3_g16", return_scales=True
    )
    cake_act_qdq_det_train_forward(x, format="mxfp4_e8m0_g32", out=x)  # in place

On an R200 (SM107), ``locality="domain"`` runs a program whose CTAs work on
their own die's memory, for an out-of-place call of 8,388,608 to 2,147,482,624
vectors (``numel / 8``), whose input and output lie in memory from
:func:`cake_act_qdq_det_train_domain_empty` (2 MB chunks that alternate
between the two dies), at the same offset in their chunks and in chunks of the
same die.  Every other call runs the default program; the bytes are the same
either way, and :func:`cake_act_qdq_det_train_domain_plan` reports which
program a call runs::

    o = torch.randn(16384, 8, 4096, device="cuda", dtype=torch.bfloat16)
    x = cake_act_qdq_det_train_domain_empty(o.shape)
    y = cake_act_qdq_det_train_domain_empty(o.shape)
    x.copy_(o)
    cake_act_qdq_det_train_forward(x, format="mxfp8_e8m0_g32", out=y, locality="domain")
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Sequence, Tuple, Union

import torch

from ._cake_pplx_api_act_qdq import cake_api
from ._cake_pplx_api_act_qdq import cake_act_qdq_det_train_forward_trace_dispatch


@functools.cache
def _backend() -> Any:
    from ._backends import cake_act_qdq_det_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_act_qdq_det_train


# The backend package's constants, resolved on first access by the module
# ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "FORMATS",
    "GROUP_SIZES",
)


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        return getattr(_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@cake_api
def is_cake_act_qdq_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of a supported architecture.

    Parameters
    ----------
    device : torch.device
        The device to check.

    Returns
    -------
    bool
        ``True`` when the package has kernels for ``device`` (SM100, SM103 or
        SM107).
    """

    return _backend().is_cake_act_qdq_det_train_supported(device)


@cake_api(trace=cake_act_qdq_det_train_forward_trace_dispatch)
def cake_act_qdq_det_train_forward(
    x: torch.Tensor,
    *,
    format: str,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    return_scales: bool = False,
    locality: str = "off",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Quantize ``x`` per group to ``format`` and dequantize it back to BF16.

    Per group of ``G`` elements, with ``amax`` the largest magnitude and ``s``
    the format's scale, ``y = bf16(q * s)`` where ``q`` is the element format's
    round-to-nearest-even value of ``clamp(x / s, -Q, Q)``.  The result equals
    the FP64 rule bit for bit.  Not autograd: the outputs carry no history.

    Parameters
    ----------
    x : torch.Tensor
        Contiguous, 16-byte aligned BF16 CUDA tensor ``[..., D]`` with ``D`` a
        multiple of ``G``.  Any number of rows, 0 included (no launch).
    format : str
        ``"fp4_e4m3_g16"`` (``G = 16``), ``"mxfp4_e8m0_g32"`` or
        ``"mxfp8_e8m0_g32"`` (``G = 32``).
    out : torch.Tensor, optional
        Output buffer: ``x`` itself works in place; another buffer is a
        contiguous, 16-byte aligned BF16 tensor of ``x``'s shape that does not
        overlap ``x``.  A new tensor when omitted.
    scales : torch.Tensor, optional
        FP32 buffer ``[..., D / G]`` for the scales; giving it requests the
        scales.
    return_scales : bool, optional
        Whether to return the FP32 scales (a new tensor unless ``scales`` is
        given).
    locality : str, optional
        ``"off"`` (default) or ``"domain"``; checked before every other
        argument (``TypeError`` for a non-string, ``ValueError`` for another
        string).  ``"domain"`` runs the R200 program whose CTAs work on their
        own die's memory when the call is out of place on SM107, ``x`` and
        ``out`` both lie in memory from
        :func:`cake_act_qdq_det_train_domain_empty` at the same offset in
        their 2 MB chunks and in chunks of the same die, and the call is
        large enough; every other call runs the ``"off"`` program.  The
        bytes are the same either way.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        ``y`` (BF16, ``x``'s shape; ``out`` when given), or ``(y, scales)``
        when ``return_scales`` is true or ``scales`` is given.
    """
    return _backend().cake_act_qdq_det_train_forward(
        x, format=format, out=out, scales=scales, return_scales=return_scales, locality=locality
    )


@cake_api
def cake_act_qdq_det_train_domain_empty(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[Union[torch.device, str, int]] = None,
) -> torch.Tensor:
    """An uninitialized tensor in memory whose 2 MB chunks alternate between
    the two dies of an R200 (SM107): the ``x`` and ``out`` of a
    ``locality="domain"`` call.

    Chunk ``k`` of the allocation lies in die ``k & 1``.  The tensor is
    contiguous and starts on a chunk; views of it (slices, reshapes) keep it
    alive.  Freeing the last view synchronizes the device; during a CUDA
    graph capture on any device in this thread, the range is freed after the
    capture, by a later free or a later call of this function.  A device's
    first call also maps the claim counters of the domain program (2 MB in
    each die, kept for the process), zeroes them and synchronizes once.
    Allocate the buffers once, before any CUDA graph capture, and reuse
    them.  A capture begun with ``cudaStreamBeginCapture`` on a stream that is
    not torch's current stream is not detected.  The ``"off"`` programs of
    act_qdq 0.1.6 ran on this memory at 0.95-1.03x of their speed on default
    memory for out-of-place calls of 131,072 to 4,194,304 vectors (GPU time,
    VR200 package benchmark); in-place calls, larger ones and the mid program
    were not measured.
    Needs cuda-python (the ``locality`` extra).

    Parameters
    ----------
    shape : sequence of int
        The tensor shape.
    dtype : torch.dtype, optional
        The element type (BF16 by default).
    device : torch.device, str or int, optional
        The CUDA device; the current one by default.

    Returns
    -------
    torch.Tensor
        A new contiguous tensor; its contents are undefined.  Raises
        ``TypeError`` for a ``dtype`` that is not a ``torch.dtype``,
        ``ValueError`` for a negative dimension or a device that is not
        CUDA, ``torch.cuda.OutOfMemoryError`` when the device has no room
        for the tensor or, on its first call, for the claim counters (after
        torch's cached free memory is returned to the driver once), and
        ``RuntimeError`` during a CUDA graph capture on any device in
        this thread, on a device other than SM107, when the driver reports
        fewer than two locality domains, or without a cuda-python that names
        them.  torch's ``torch.cuda.memory_*`` counters do not count this
        memory or the claim counters.
    """

    return _backend().cake_act_qdq_det_train_domain_empty(shape, dtype=dtype, device=device)


@cake_api
def cake_act_qdq_det_train_domain_plan(
    x: torch.Tensor,
    *,
    format: str,
    out: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    return_scales: bool = False,
) -> dict[str, Any]:
    """Which program ``cake_act_qdq_det_train_forward(..., locality="domain")``
    runs for these arguments, without a launch or an allocation.

    The arguments are checked as the forward checks them.

    Parameters
    ----------
    x : torch.Tensor
        The forward's input: contiguous BF16 CUDA ``[..., D]``.
    format : str
        ``"fp4_e4m3_g16"``, ``"mxfp4_e8m0_g32"`` or ``"mxfp8_e8m0_g32"``.
    out : torch.Tensor, optional
        The forward's output buffer: ``x`` itself (in place) or a buffer of
        ``x``'s shape that does not overlap ``x``.
    scales : torch.Tensor, optional
        The forward's FP32 scale buffer ``[..., D / G]``.
    return_scales : bool, optional
        Whether the forward returns the scales.

    Returns
    -------
    dict
        ``route``: ``"domain"`` or ``"plain"``.  ``reason``: ``"placed"`` on
        the domain route; on the plain route the first that holds of
        ``"empty"`` (no elements, no launch), ``"arch"`` (not SM107),
        ``"in_place"``, ``"no_out"`` (``out`` not given), ``"size"`` (outside
        the domain program's size range), ``"x_memory"`` / ``"out_memory"``
        (``x`` / ``out`` not in memory from
        :func:`cake_act_qdq_det_train_domain_empty`), ``"chunk_offset"``
        (different offsets in their chunks) and ``"die_parity"`` (chunks in
        different dies).  ``program``: the program's registry name (None
        without a launch).  ``first_tile`` and ``first_die``: where the
        input's first tile lies (None on the plain route).  ``grid``: the
        launch grid (None without a launch).
    """

    return _backend().cake_act_qdq_det_train_domain_plan(
        x, format=format, out=out, scales=scales, return_scales=return_scales
    )


@cake_api
def cake_act_qdq_det_train_forward_workspace_bytes(
    shape: Sequence[int],
    *,
    format: str,
) -> int:
    """The workspace of one forward call of an input of ``shape``.

    Always 0: a call takes no caller workspace.  A ``locality="domain"`` call
    keeps its claim counters in the package's own workspace per stream: 128
    bytes in each die's memory, from the device's 15,360 eager workspaces
    (for eager calls) or 1,024 graph workspaces (for CUDA graph captures); a
    stream or capture that finds its kind all taken keeps both counters in
    default memory.

    Parameters
    ----------
    shape : sequence of int
        The input shape ``[..., D]``.
    format : str
        One of ``FORMATS``.

    Returns
    -------
    int
        0.  Raises ``ValueError`` for a format or shape the forward refuses.
    """

    return _backend().cake_act_qdq_det_train_forward_workspace_bytes(shape, format=format)


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "FORMATS",  # noqa: F822
    "GROUP_SIZES",  # noqa: F822
    "cake_act_qdq_det_train_domain_empty",
    "cake_act_qdq_det_train_domain_plan",
    "cake_act_qdq_det_train_forward",
    "cake_act_qdq_det_train_forward_workspace_bytes",
    "is_cake_act_qdq_det_train_supported",
]
