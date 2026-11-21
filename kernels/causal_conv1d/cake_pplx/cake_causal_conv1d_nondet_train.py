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

Cake packed width-4 depthwise causal convolution + SiLU training forward /
backward (SM100 / SM103 / SM107), the non-deterministic package: the stable
entry points over the experimental backend package
``cake_pplx._backends.cake_causal_conv1d_nondet_train``.
The backward is ONE ordinary launch whose CTAs add their ``dweight`` /
``dbias`` rows into zeroed accumulators with unordered global reductions (the
backward of the deterministic package
(:mod:`cake_pplx.cake_causal_conv1d_det_train`) is a main launch plus a
fixed-order reduce launch).  At the rows of the backend's
``cake_jit.SM_WIDE_ROWS`` (the default backward route at C = 520 to 608 from
S = 24576 on sm_100a and from S = 20480 on sm_103a, at C = 520 to 680 from
S = 32768 on sm_107a) that launch is an SM-wide program: one CTA per SM for
the whole launch, each CTA a whole token stream, so a kernel that keeps an SM
busy on another stream can delay it.  On sm_100a and sm_103a, at the backward
rows of ``cake_jit.FALLBACK_ROWS`` (the default backward route at C = 520 to
1016; from release 0.2.7 also the ``bwd_bias``, ``bwd_bias_bf16_taps`` and
``bwd_dt_bf16_taps`` routes at C = 520 to 1016 and small S,
``bwd_dt_bf16_taps`` not at C = 768 on sm_103a; each row up to its own token
bound), that launch runs the route's fallback program, the one the route runs
at its smallest channel counts, in place of the program its channel thresholds
pick.  This module never runs the deterministic package.

Numerical mode (the same rule as every non-deterministic training package):
the module name chooses the package, and no entry point takes a mode keyword.
"torch deterministic mode" is ``torch.use_deterministic_algorithms(True)``,
with or without ``warn_only=True``::

    package                                        torch deterministic mode off   torch deterministic mode on
    the non-deterministic package                  non-deterministic              the forward runs; the backward
      (cake_pplx.cake_causal_conv1d_nondet_                                         raises RuntimeError before it
      train[_kcp], this module)                                                     launches (warn_only=True: a
                                                                                    UserWarning, then it runs); it
                                                                                    never falls back
    the deterministic package                      deterministic                  deterministic; no warning
      (cake_pplx.cake_causal_conv1d_det_
      train[_kcp])

The forward is fixed-order (this package's copies of the deterministic package's programs) and never
alerts; only the backward's unordered ``dweight`` / ``dbias`` reduction does,
and it reads torch's deterministic setting when it runs (the autograd backward
included).  The package is chosen by name, so a checkpoint recompute
(``torch.utils.checkpoint``, reentrant or not) runs the same package.

Numerical contract: the non-deterministic package makes no bit promise (no
output is promised to repeat bit for bit run to run or to match the
deterministic package's bits).  The forward runs this package's copies of the
deterministic package's programs.  ``dx`` uses the deterministic backward's per-thread FP32 arithmetic
and one BF16 rounding.  By default (``dweight_sum="fp32"``) ``dweight`` /
``dbias`` accumulate in plain FP32 words on every route, as the deterministic
reduce sums (the default ``silu_grad_preactivation="bf16"``: within about
3e-5 rel-L2 of the exact FP64 gradient; the ``silu_grad_preactivation="fp32"``
routes spread the adds over sixteen FP32 copies of the accumulators, added in
copy order at the end; no FP64 and no compensated sums; the opt-in
``dweight_sum="fp32_pair"`` keeps FP32 (hi, lo) pairs instead), and are
rounded once into the requested dtype; the accumulation order across CTAs
varies run to run.  The inputs,
output dtypes, masks, layouts, dynamic shapes and parameter-gradient ownership
are the deterministic package's.

The forward programs (each the deterministic package's program of the same row):
at C = 576 to 768, inside the token range of a row of the backend's
``cake_jit.FALLBACK_ROWS`` or ``cake_jit.DIRECT_ROWS`` (by architecture and
route), the static program (a fixed token range per CTA) or the direct
program (the next token blocks per CTA, no shared memory) runs, neither with
``work`` counters: a passed forward ``workspace`` is checked and not passed
to the program, and such a call zeroes no package buffer.  Each row gives its
own token range.  From release 0.2.6, every architecture and route has a row
at each ``C`` = 576 to 768 (0.2.5 had none for sm_100a ``fwd_bias`` at
C = 640, sm_103a ``fwd_bias`` at C = 576 and 712, and sm_103a ``fwd`` at
C = 768).  For most ``C`` the first range starts at S = 1.  The sm_103a
``fwd_bias`` rows at C = 720 to 768 start at S = 13824 (C = 736: 14336) and
have only a static range.  Every other call (outside a row's range, or where an
architecture, route and ``C`` have no row) runs the dynamic program, as in
0.2.4, which takes its token chunks from zeroed int32 counters.  The forward
workspace size is the same in every case.

The backward workspace holds zeroed accumulators and counters that every
launch leaves zero (:func:`cake_causal_conv1d_nondet_train_backward_workspace`
returns it zeroed; size depends on ``C`` and the route, not on ``S``; never
share one between two backward launches that may run at once, e.g. on two
streams); when omitted the package keeps one zeroed buffer per (device,
current stream), or inside a CUDA Graph capture one buffer per capture and
stream, shared by its calls on that stream: a buffer the package zeroed
before the capture, on an eager call that omitted ``workspace=``, so the
graph records no fill node for it (the graph holds it until the graph and
its instantiations are destroyed and their launches have completed; it
returns to the package, zero again, at the package's next refill).  The
replays of one graph must not overlap.  Only eager calls without
``workspace=`` that take counters zero these buffers (a forward inside the
token range of a row of ``FALLBACK_ROWS`` / ``DIRECT_ROWS`` takes none), the
forward's and the backward's separately, sized to the largest need of such a call whose
zeroing succeeded (one that runs out of memory keeps no new buffer and keeps
their size).
When none is ready (no such eager call of the kind before the capture, all
taken by earlier captures, too small, or their zero fills not complete yet:
a ``CUDAGraph.capture_begin`` right after such an eager call with no
synchronization; ``torch.cuda.graph`` synchronizes first) each captured call
zeroes a buffer of its own (one fill node per call), as every captured call
does with capture id 0 (also when ``cuda.bindings`` is not importable).  The
forward workspace
(zeroed int32 token-chunk counters) has the deterministic package's size and
rules; inside a capture the forward's omitted workspace follows the
backward's rules with buffers of its own.  Memory these buffers free stays
cached in the package's own memory pool (``torch.cuda.MemPool``); the package
does not return it to torch's other pools.

Programmatic dependent launch: the single-rank forward and backward programs
(not those of the opt-in pair routes) take it per call, by architecture and
mode (outside / inside a CUDA Graph capture; the backend's
``cake_jit.PDL_EAGER`` / ``PDL_GRAPH``): with it a launch may begin while the
stream's preceding kernel finishes, and every program waits for that kernel
before it reads or writes global memory, so the outputs do not depend on it.
The backward launches without it, in both modes, at the rows of the backend's
``cake_jit.PDL_OFF_ROWS`` (by route, ``C`` and ``S``: the default backward
route at C = 1000 to 1008 from S = 187755 and at C = 1016 from S = 155648 on
sm_100a, and at C = 1000 to 1008 from S = 131073 and at C = 1016 from
S = 125171 on sm_103a), from where the measured training step was not slower
without it; the forward keeps it at every row.

The backward entry points and the backward workspace helpers take the
deterministic package's opt-in ``bwd_grid_mult: int = 1`` (one of
``BWD_GRID_MULTS``: 1, 2, 3 or 4), with its validation (``TypeError`` for a
non-int, ``ValueError`` for another value): the backward launches
``bwd_grid_mult`` times its default untrimmed CTA count, a finer static
partition of the token axis for a backward that overlaps kernels of other
streams.  The CTAs of each channel tile split its tokens among themselves at
any grid size, so ``dx`` is computed the same way and
``dweight`` / ``dbias`` are summed the same way; the workspace size does not depend
on it (one accumulator set and one counter per channel tile, whatever the
grid).  At the rows of ``cake_jit.SM_WIDE_ROWS`` the SM-wide program holds
one CTA per SM, so ``bwd_grid_mult`` times the SM count runs in that many
waves of CTAs.

Name map (the deterministic package -> the non-deterministic package)::

    cake_pplx.cake_causal_conv1d_det_train.             cake_pplx.cake_causal_conv1d_nondet_train.
      cake_causal_conv1d_det                               cake_causal_conv1d_nondet
      cake_causal_conv1d_det_train_forward                 cake_causal_conv1d_nondet_train_forward
      cake_causal_conv1d_det_train_backward                cake_causal_conv1d_nondet_train_backward
      cake_causal_conv1d_det_train_{forward,backward}_     cake_causal_conv1d_nondet_train_{forward,backward}_
        workspace{,_bytes}                                   workspace{,_bytes}
      CakeCausalConv1dDetFunction                          CakeCausalConv1dNondetFunction
      is_cake_causal_conv1d_det_train_supported            is_cake_causal_conv1d_nondet_train_supported
    cake_pplx.cake_causal_conv1d_det_train_kcp          cake_pplx.cake_causal_conv1d_nondet_train_kcp
                                                           (cake_causal_conv1d_nondet_kcp,
                                                           cake_causal_conv1d_nondet_kcp_multi,
                                                           CakeCausalConv1dNondetKcpRecomputeHandle)

Usage::

    from cake_pplx.cake_causal_conv1d_nondet_train import (
        cake_causal_conv1d_nondet,
        cake_causal_conv1d_nondet_train_backward,
        cake_causal_conv1d_nondet_train_forward,
    )

    y = cake_causal_conv1d_nondet(x, weight, cu_seqlens, bias)  # one-launch backward

    y = cake_causal_conv1d_nondet_train_forward(x, weight, cu_seqlens)
    dx, dweight = cake_causal_conv1d_nondet_train_backward(x, weight, cu_seqlens, dy)

    # opt-in finer backward partition (1, 2, 3 or 4; default 1)
    dx, dweight = cake_causal_conv1d_nondet_train_backward(
        x, weight, cu_seqlens, dy, bwd_grid_mult=3
    )

    # opt-in: dweight / dbias accumulated as FP32 (hi, lo) pairs (FP32 SiLU' only)
    dx, dweight = cake_causal_conv1d_nondet_train_backward(
        x, weight, cu_seqlens, dy, silu_grad_preactivation="fp32", dweight_sum="fp32_pair"
    )
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ._cake_pplx_api_causal_conv1d import cake_api

# The backend package lives under cake_pplx._backends; this module imports it on first use only, so importing this
# module loads nothing from the backend.
# Its constants and its autograd Function resolve through the module ``__getattr__`` below.  The entry points are stable
# APIs (``@cake_api``): real ``def``s whose signatures mirror the package's, forwarding each call to it.
_FROM_BACKEND = (
    "ARCHES",
    "BWD_GRID_MULTS",
    "DWEIGHT_SUMS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dNondetFunction",
)


def _backend():
    from ._backends import cake_causal_conv1d_nondet_train  # noqa: PLC0415

    return cake_causal_conv1d_nondet_train


@cake_api
def is_cake_causal_conv1d_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` supports the non-deterministic causal Conv1d package.

    Parameters
    ----------
    device : torch.device
        CUDA device to query.

    Returns
    -------
    bool
        ``True`` when ``device`` runs the package's base forward and backward
        routes and the rendered ``build.json`` is available.
    """

    return _backend().is_cake_causal_conv1d_nondet_train_supported(device)


@cake_api
def cake_causal_conv1d_nondet_train_backward_workspace_bytes(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
) -> int:
    """Return the backward workspace size in bytes.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.
    bwd_grid_mult : int, default=1
        Backward grid multiplier to validate for the selected backend.

    Returns
    -------
    int
        Bytes of the caller-owned backward workspace. The size depends on
        ``C`` and the target device, not on ``S``.
    """

    return _backend().cake_causal_conv1d_nondet_train_backward_workspace_bytes(
        S, C, device, bwd_grid_mult=bwd_grid_mult
    )


@cake_api
def cake_causal_conv1d_nondet_train_backward_workspace(
    S: int,
    C: int,
    device: torch.device,
    *,
    bwd_grid_mult: int = 1,
) -> torch.Tensor:
    """Allocate a zeroed backward workspace tensor.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.
    bwd_grid_mult : int, default=1
        Backward grid multiplier to validate for the selected backend.

    Returns
    -------
    torch.Tensor
        Zeroed caller-owned workspace for reuse on a single stream.
    """

    return _backend().cake_causal_conv1d_nondet_train_backward_workspace(
        S, C, device, bwd_grid_mult=bwd_grid_mult
    )


@cake_api
def cake_causal_conv1d_nondet_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device
) -> int:
    """Return the forward workspace size in bytes.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.

    Returns
    -------
    int
        Bytes of the zeroed int32 forward workspace.
    """

    return _backend().cake_causal_conv1d_nondet_train_forward_workspace_bytes(
        S, C, device
    )


@cake_api
def cake_causal_conv1d_nondet_train_forward_workspace(
    S: int, C: int, device: torch.device
) -> torch.Tensor:
    """Allocate a zeroed forward workspace tensor.

    Parameters
    ----------
    S : int
        Packed token count.
    C : int
        Channel count.
    device : torch.device
        CUDA device that will execute the kernel.

    Returns
    -------
    torch.Tensor
        Zeroed int32 forward workspace for reuse on a single stream.
    """

    return _backend().cake_causal_conv1d_nondet_train_forward_workspace(S, C, device)


@cake_api
def cake_causal_conv1d_nondet_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    The package's copy of the deterministic forward programs; the forward runs (without an alert) under torch's
    deterministic mode.

    The kernels access ``x``, ``weight``, ``bias`` and ``out`` as vectors, so each address (``data_ptr()``) must be
    aligned as stated below; a misaligned one raises ``ValueError`` before anything launches.  A fresh tensor or a
    ``.clone()`` is aligned, and so is a view that starts at a whole row (``x[1:]``, ``weight[1:]``).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.  16-byte aligned.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the
        current token.  FP32 taps are 16-byte aligned, BF16 taps 8-byte aligned.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths; the host reads its metadata only.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias, added after the taps.
        An FP32 bias is 32-byte aligned, a BF16 bias 16-byte aligned.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous, 16-byte aligned.
    workspace : torch.Tensor, optional
        Caller-owned zeroed int32 buffer from
        :func:`cake_causal_conv1d_nondet_train_forward_workspace`.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    return _backend().cake_causal_conv1d_nondet_train_forward(
        x,
        weight,
        cu_seqlens,
        bias=bias,
        out=out,
        workspace=workspace,
    )


@cake_api
def cake_causal_conv1d_nondet_train_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dy: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    silu_grad_preactivation: str = "bf16",
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    dbias: Optional[torch.Tensor] = None,
    bwd_grid_mult: int = 1,
    dweight_sum: str = "fp32",
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward (one launch, unordered
    ``dweight`` / ``dbias`` reductions).

    Under ``torch.use_deterministic_algorithms(True)`` the backward raises PyTorch's ``RuntimeError`` before it
    launches (a ``UserWarning``, then it runs, with ``warn_only=True``); it never falls back to the deterministic
    package.

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs (an FP32 ``bias`` needs only 16-byte alignment
        here); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous, 16-byte aligned.
    silu_grad_preactivation : {"bf16", "fp32"}
        ``"bf16"``: SiLU' at the BF16-rounded recomputed preactivation with an
        FP32 ``weight``, at the unrounded FP32 preactivation with a BF16
        ``weight``.
        ``"fp32"``: at the unrounded FP32 preactivation.  Both accumulate in
        plain FP32 words by default (see ``dweight_sum``).
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_causal_conv1d_nondet_train_backward_workspace`.
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (FP32 or BF16 each; the parameter's dtype when omitted).  ``dx`` must
        be 16-byte aligned, and so must an FP32 ``dweight`` (one 16-byte store
        per channel); ``ValueError`` otherwise.
    bwd_grid_mult : {1, 2, 3, 4}
        Opt-in finer static partition of the token axis (default 1, the
        default grid): the backward launches ``bwd_grid_mult`` times its
        default untrimmed CTA count, so each CTA streams that much fewer tokens
        (for a backward that overlaps kernels of other streams: a CTA that
        starts late trails by a shorter stream).  ``dx`` is computed as by
        the default; more CTAs add into the same accumulators (same sum form,
        same workspace size).  ``TypeError`` for a non-int, ``ValueError`` for
        another value.
    dweight_sum : {"fp32", "fp32_pair"}
        The sum form of ``dweight`` / ``dbias``.  ``"fp32"`` (default): one
        plain FP32 word per sum.  ``"fp32_pair"`` (opt-in): an FP32 (hi, lo)
        pair per sum -- the CTA's fold Kahan-compensated and every accumulator
        word a pair that keeps each add's rounding error, rounded once to the
        output dtype; FP32 operations only, ``dx`` computed as by the default
        and the same workspace (the CTAs' order still varies run to run).  Only with
        ``silu_grad_preactivation="fp32"``; ``ValueError`` otherwise and for
        another ``str``, ``TypeError`` for a non-``str``.

    Returns
    -------
    ``(dx, dweight)``, with a third element ``dbias`` only when ``bias`` is
    given.  ``S == 0`` returns the empty ``dx`` and zero gradients without
    launching.
    """

    return _backend().cake_causal_conv1d_nondet_train_backward(
        x,
        weight,
        cu_seqlens,
        dy,
        bias=bias,
        silu_grad_preactivation=silu_grad_preactivation,
        workspace=workspace,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        bwd_grid_mult=bwd_grid_mult,
        dweight_sum=dweight_sum,
    )


@cake_api
def cake_causal_conv1d_nondet(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
    bwd_grid_mult: int = 1,
    dweight_sum: str = "fp32",
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU, non-deterministic backward.

    The package's autograd Function :class:`CakeCausalConv1dNondetFunction`, whose backward is one launch with
    unordered ``dweight`` / ``dbias`` reductions.  When torch's deterministic mode is on as the backward runs, the
    backward raises a ``RuntimeError`` (a ``UserWarning``, then it runs, with ``warn_only=True``).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations, 16-byte
        aligned.
    weight : torch.Tensor
        FP32 (16-byte aligned) or BF16 (8-byte aligned) ``[C, 4]`` contiguous
        taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.
    bias : torch.Tensor, optional
        FP32 (32-byte aligned) or BF16 (16-byte aligned) ``[C]`` contiguous
        per-channel bias.  A misaligned tensor raises ``ValueError``.
    silu_grad_preactivation : {"bf16", "fp32"}
        The weight gradient's SiLU' policy.
    bwd_grid_mult : {1, 2, 3, 4}
        The backward's opt-in finer partition (default 1; see
        :func:`cake_causal_conv1d_nondet_train_backward`); checked at the forward
        call, used by the backward.
    dweight_sum : {"fp32", "fp32_pair"}
        The sum form of ``dweight`` / ``dbias`` (default ``"fp32"``; the opt-in
        ``"fp32_pair"`` needs ``silu_grad_preactivation="fp32"``, see
        :func:`cake_causal_conv1d_nondet_train_backward`); checked at the
        forward call, used by the backward.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape; the backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns the gradients in the
        parameters' dtypes.
    """

    return _backend().cake_causal_conv1d_nondet(
        x,
        weight,
        cu_seqlens,
        bias,
        silu_grad_preactivation=silu_grad_preactivation,
        bwd_grid_mult=bwd_grid_mult,
        dweight_sum=dweight_sum,
    )


def __getattr__(name: str) -> Any:
    if name in _FROM_BACKEND:
        return getattr(_backend(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_FROM_BACKEND))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "BWD_GRID_MULTS",  # noqa: F822 - resolved by __getattr__
    "DWEIGHT_SUMS",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822 - resolved by __getattr__
    "CakeCausalConv1dNondetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_causal_conv1d_nondet",
    "cake_causal_conv1d_nondet_train_backward",
    "cake_causal_conv1d_nondet_train_backward_workspace",
    "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_forward",
    "cake_causal_conv1d_nondet_train_forward_workspace",
    "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    "is_cake_causal_conv1d_nondet_train_supported",
]
