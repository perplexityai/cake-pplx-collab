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
``README.md`` gives the rule and why FP32 steps reach it.  Every output
repeats bit for bit for identical inputs: each group is quantized on its own,
and nothing is summed.

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

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        ``y`` (BF16, ``x``'s shape; ``out`` when given), or ``(y, scales)``
        when ``return_scales`` is true or ``scales`` is given.
    """
    return _backend().cake_act_qdq_det_train_forward(
        x, format=format, out=out, scales=scales, return_scales=return_scales
    )


@cake_api
def cake_act_qdq_det_train_forward_workspace_bytes(
    shape: Sequence[int],
    *,
    format: str,
) -> int:
    """The workspace of one forward call of an input of ``shape``.

    Always 0: a call needs no workspace and keeps no state between calls.

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
    "cake_act_qdq_det_train_forward",
    "cake_act_qdq_det_train_forward_workspace_bytes",
    "is_cake_act_qdq_det_train_supported",
]
