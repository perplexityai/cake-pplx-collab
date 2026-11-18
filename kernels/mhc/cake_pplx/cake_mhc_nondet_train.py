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

Cake four-stream mHC training forward / backward, ``nondet`` mode
(4 streams; hidden 4096, 5120 and 7168 on SM100 / SM103 / SM107; another
hidden size raises ``ValueError``): every entry point runs the
package ``cake_pplx._backends.cake_mhc_nondet_train``, imported on
the first call or attribute access (importing this module imports no
backend).  The function, the tensor contracts and the autograd wiring are
those of the deterministic module :mod:`cake_pplx.cake_mhc_det_train`
(documented there); the workspaces are described below.

Mode: the mode is part of the names (``cake_mhc_nondet_train_*``,
``CakeMHCNondet*Function``).  No entry point takes a ``deterministic``
keyword (passing one is a ``TypeError``), and nothing falls back to the
deterministic package ``cake_pplx.cake_mhc_det_train``.

Numerical contract: no bit promise.  Every sum of the current programs is a
plain FP32 sum in an order fixed by the token count and the device, but the
package does not promise that an output repeats bit for bit from run to run,
and its outputs are not compared bitwise with the deterministic package's.
``backward_finish`` has no final reduction kernel: the column-slab pass
stores the FP32 ``dfn`` sum of every (column slab, token chunk) unit in its
chunk's slots of the backward workspace, and once every unit has stored its
sums the grid adds the slots (``P <= 32`` token chunks, the deterministic
package's chunks per token count and device) per element in ascending chunk
order as one FP32 sum from +0.0, the caller's ``dfn`` last under
``accumulate=True``: within ``gamma_n (|start| + sum |x_p|)`` of the exact
sum (``x_p`` the partials, ``n`` the additions, ``gamma_n = n u / (1 - n
u)``, ``u = 2^-24``).  The column-slab pass also folds ``dscale`` and
``dbase`` from the token-owner pass's parameter rows in a fixed order.
By default (``accumulation="fp32"``) every sum is one plain FP32 word: no
FP64 accumulation or reduction and no compensation word (a few scalar
epilogues of the RMS scale, a finished FP32 sum times ``1/(4H)`` plus
``rms_eps`` or times the RMS scale, compute in FP64 and round once to
FP32).  The opt-in ``accumulation="fp32x2"`` of ``forward_prepare``, ``forward_stats``,
``backward_finish`` and ``prepare`` runs two-word programs of the mapping
pass, the backward token-owner pass and the column slab with its folds: each
of their sums is a compensated pair of FP32 words (the running sum and its
rounding error), stored as one FP32 word.  It is a compensated FP32 sum, not
FP64.  The other passes keep their one-word programs, and the workspaces and
records are unchanged.  The mode serves hidden 4096 with
``sinkhorn_iters=20`` only (otherwise ``NotImplementedError``).  The forward, the statistics replay and ``backward_prepare`` run copies
of the deterministic package's programs.  Under
``torch.use_deterministic_algorithms(True)`` no entry point raises PyTorch's
alert or warning.

Workspaces: the forward workspace has one layout in both packages.  The
backward workspace has this package's own layout: a 256-byte ticket region
and 512 bytes of column-slab arrival counts first (zero on first use), the
deterministic package's regions without its per-chunk FP32 ``dfn`` partials,
then the column slab's per-chunk FP32 ``dfn`` slots
(``min(32, ceil(T / 64)) x 24 x 4H x 4`` bytes at hidden size ``H``).  A
deterministic backward workspace does not serve this backward: a buffer the
deterministic package's backward used must be zeroed
(:func:`cake_mhc_nondet_train_reset_workspace`) before this package uses it.
The records of either package (``stats``, ``grad_stats``) have one layout and
feed the other's entry points; a prepare handle feeds only the finish of its
own package (``TypeError`` otherwise).

Default workspaces (``workspace=None``): one buffer per device, hidden size and
power-of-two token-count bucket, shared by every stream and every graph replay
on that device; calls that can run concurrently on different streams pass
caller-owned workspaces.  No default buffer is created while the current stream
of the buffer's device is capturing a CUDA graph (``ValueError``): warm up with
the same token count and hidden size before the capture.  Every completed
launch leaves the ticket words and the arrival counts of its workspace at
zero, so a replay needs no reset; :func:`cake_mhc_nondet_train_reset_workspace`
zeroes the buffer it is given.

Name map (the same verbs in both modes)::

    cake_pplx.cake_mhc_det_train                 cake_pplx.cake_mhc_nondet_train
      cake_mhc_det_train_forward_prepare            cake_mhc_nondet_train_forward_prepare
      cake_mhc_det_train_forward_stats              cake_mhc_nondet_train_forward_stats
      cake_mhc_det_train_forward_finish             cake_mhc_nondet_train_forward_finish
      cake_mhc_det_train_backward_prepare           cake_mhc_nondet_train_backward_prepare
      cake_mhc_det_train_backward_finish            cake_mhc_nondet_train_backward_finish
      cake_mhc_det_train_prepare                    cake_mhc_nondet_train_prepare
      cake_mhc_det_train_finish                     cake_mhc_nondet_train_finish
      cake_mhc_det_train_{forward,backward}_        cake_mhc_nondet_train_{forward,backward}_
        workspace{,_bytes}                            workspace{,_bytes}
      cake_mhc_det_train_reset_workspace            cake_mhc_nondet_train_reset_workspace
      is_cake_mhc_det_train_supported               is_cake_mhc_nondet_train_supported
      CakeMHCDetPrepareFunction                     CakeMHCNondetPrepareFunction
      CakeMHCDetFinishFunction                      CakeMHCNondetFinishFunction
      cake_mhc_det_train_route_key                  cake_mhc_nondet_train_route_key
      cake_mhc_det_train_handoff_{forward_prepare,  cake_mhc_nondet_train_handoff_{forward_prepare,
        backward_finish,prepare}                      backward_finish,prepare}
      cake_mhc_det_train_collapse{,_forward,        cake_mhc_nondet_train_collapse{,_forward,
        _backward}                                    _backward}
      is_cake_mhc_det_train_handoff_supported       is_cake_mhc_nondet_train_handoff_supported
      CakeMHCDetHandoffPrepareFunction              CakeMHCNondetHandoffPrepareFunction
      CakeMHCDetCollapseFunction                    CakeMHCNondetCollapseFunction

The model's head is a module of its own in this mode:
``cake_pplx.cake_mhc_head_nondet_train`` (``cake_mhc_head_nondet_train_forward``
/ ``cake_mhc_head_nondet_train_backward`` / ``cake_mhc_head_nondet``).

Usage::

    from cake_pplx.cake_mhc_nondet_train import (
        cake_mhc_nondet_train_backward_finish,
        cake_mhc_nondet_train_backward_workspace,
        cake_mhc_nondet_train_finish,
        cake_mhc_nondet_train_prepare,
    )

    U, handle = cake_mhc_nondet_train_prepare(X, fn, scale, base)
    Y = cake_mhc_nondet_train_finish(X, branch(U), handle)  # backward: folded dfn

    ws_b = cake_mhc_nondet_train_backward_workspace(T, X.device, hidden=X.shape[2])
    dX, dfn, dscale, dbase = cake_mhc_nondet_train_backward_finish(
        X, dY, dU, B, fn, scale, base, stats, grad_stats, workspace=ws_b)

Pre handoff (DeepSeek-V4.1-Flash; hidden 5120 on SM100 / SM103 / SM107,
:func:`is_cake_mhc_nondet_train_handoff_supported`): a sublayer collapses its
streams with the incoming FP32 ``pre_in [T, 4]`` and hands its own ``pre`` on
as ``pre_next``; its backward takes ``dpre_next`` (the gradient of
``pre_next``) for the coefficient VJP and returns ``dpre_in``.  The final
collapse runs alone.  Like every entry point of this module, these make no
run-to-run bit promise::

    U, pre_next, handle = cake_mhc_nondet_train_handoff_prepare(X, pre_in, fn, scale, base, rms_eps=1e-20)
    Y = cake_mhc_nondet_train_finish(X, branch(U), handle)
    head_input = cake_mhc_nondet_train_collapse(Y_last, pre_last)       # the final collapse

    # direct calls: handoff_forward_prepare -> (U, pre_next, stats); forward_finish; backward_prepare;
    # handoff_backward_finish(X, dY, dU, pre_in, dpre_next, B, fn, scale, base, stats, grad_stats)
    #   -> (dX, dpre_in, dfn, dscale, dbase); collapse_forward / collapse_backward
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from ._cake_pplx_api_mhc import cake_api

# The default hidden size of the workspace helpers: the backend's ``cake_jit.HIDDEN``,
# spelled here so that defining the entry points imports no backend (a test checks
# that the two agree).
_HIDDEN = 4096


# The backend package is imported on first use and cached: a call does not repeat the import.
@functools.cache
def _pkg():
    from ._backends import cake_mhc_nondet_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_mhc_nondet_train


# ----------------------------------------------------------------------------
# Entry points (real ``def``s: public-API tooling reads the decorated
# signatures of this module; the bodies live in the backend package)


@cake_api
def is_cake_mhc_nondet_train_supported(
    device: torch.device,
    hidden: Optional[int] = None,
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has every exported route of the backend package.

    Parameters
    ----------
    device : torch.device
        Device to query; a device that is not CUDA, or whose architecture is not
        SM100 / SM103 / SM107, gives ``False``.
    hidden : int, optional
        Hidden size to query: 4096, 5120 or 7168 on SM100 / SM103 / SM107.
        ``None`` (default) queries the architecture only.

    Returns
    -------
    ``True`` when the device (at ``hidden``, when given) has the routes.
    """

    return _pkg().is_cake_mhc_nondet_train_supported(device, hidden)


@cake_api
def cake_mhc_nondet_train_forward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> int:
    """Bytes of the caller-owned forward workspace for up to ``T`` tokens on ``device``.

    The forward workspace has one layout in both packages (see
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_forward_workspace_bytes`).

    Parameters
    ----------
    T : int
        Largest token count the buffer serves (the size is the maximum over
        every token count up to ``T``); ``T == 0`` needs no workspace.
    device : torch.device
        CUDA device of the calls.
    hidden : int
        Hidden size ``H`` of the calls the buffer serves (``X.shape[2]``;
        default ``4096``): 4096, 5120 or 7168 on SM100 / SM103 / SM107 (any
        other value raises ``ValueError``).

    Returns
    -------
    The size of the forward workspace in bytes.
    """

    return _pkg().cake_mhc_nondet_train_forward_workspace_bytes(
        T, device, hidden=hidden
    )


@cake_api
def cake_mhc_nondet_train_backward_workspace_bytes(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> int:
    """Bytes of the caller-owned backward workspace for up to ``T`` tokens on ``device``.

    This package's backward layout: a 256-byte ticket region and 512 bytes of
    column-slab arrival counts first (zero on first use; every completed
    launch leaves them zero), the deterministic package's regions without its
    per-chunk FP32 ``dfn`` partials, then the column slab's per-chunk FP32
    ``dfn`` slots (``min(32, ceil(T / 64)) x 24 x 4H x 4`` bytes at hidden
    size ``H``).  A deterministic backward workspace
    (:func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_backward_workspace`)
    does not serve this backward.

    Parameters
    ----------
    T : int
        Largest token count the buffer serves (the size is the maximum over
        every token count up to ``T``); ``T == 0`` needs no workspace.
    device : torch.device
        CUDA device of the calls.
    hidden : int
        Hidden size of the calls the buffer serves (default ``4096``), as in
        :func:`cake_mhc_nondet_train_forward_workspace_bytes`.

    Returns
    -------
    The size of the backward workspace in bytes.
    """

    return _pkg().cake_mhc_nondet_train_backward_workspace_bytes(
        T, device, hidden=hidden
    )


@cake_api
def cake_mhc_nondet_train_forward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> torch.Tensor:
    """Allocate a zero-initialized forward workspace (one layout in both packages; see
    :func:`cake_mhc_nondet_train_forward_workspace_bytes`).

    Parameters
    ----------
    T, device, hidden : int, torch.device, int
        As in :func:`cake_mhc_nondet_train_forward_workspace_bytes`.

    Returns
    -------
    A zero-filled ``uint8`` tensor of that many bytes on ``device``.
    """

    return _pkg().cake_mhc_nondet_train_forward_workspace(T, device, hidden=hidden)


@cake_api
def cake_mhc_nondet_train_backward_workspace(
    T: int,
    device: torch.device,
    *,
    hidden: int = _HIDDEN,
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace of this package's layout (see
    :func:`cake_mhc_nondet_train_backward_workspace_bytes`).

    Parameters
    ----------
    T, device, hidden : int, torch.device, int
        As in :func:`cake_mhc_nondet_train_backward_workspace_bytes`.

    Returns
    -------
    A zero-filled ``uint8`` tensor of that many bytes on ``device``.
    """

    return _pkg().cake_mhc_nondet_train_backward_workspace(T, device, hidden=hidden)


@cake_api
def cake_mhc_nondet_train_forward_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    out: Optional[torch.Tensor] = None,
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward mapping, coefficients and collapse: ``(U, stats)``.

    The arguments and outputs are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_forward_prepare`.
    The call runs the backend package, whose forward runs copies of the
    deterministic programs, so ``U`` and ``stats`` are the deterministic
    package's bits.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams; ``H = X.shape[2]`` is
        4096, 5120 or 7168 on SM100 / SM103 / SM107.
    fn, scale, base : torch.Tensor
        FP32 ``[24, 4H]`` mapping matrix, ``[3]`` group scales and ``[24]``
        logit offsets.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        Normalization epsilon (default ``1e-5``), coefficient epsilon (default
        ``1e-6``) and Sinkhorn iteration count (1 to 64, default ``20``).
    accumulation : str
        ``"fp32"`` (default): one plain FP32 word per sum.  ``"fp32x2"``
        (opt-in): the mapping pass (and, in the backward, the token-owner pass
        and the column slab with its folds) keeps each sum as a compensated
        pair of FP32 words and stores one FP32 word: a compensated two-word
        FP32 sum, not FP64.  Hidden 4096 and ``sinkhorn_iters=20`` only;
        otherwise ``NotImplementedError``.
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``U``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned buffer from :func:`cake_mhc_nondet_train_forward_workspace`
        (``hidden=X.shape[2]``); a per-device default when omitted.

    Returns
    -------
    ``(U, stats)``: BF16 ``U`` of shape ``[T, H]`` and the FP32 ``[T, 64]``
    per-token record.
    """

    return _pkg().cake_mhc_nondet_train_forward_prepare(
        X,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        out=out,
        stats_out=stats_out,
        workspace=workspace,
    )


@cake_api
def cake_mhc_nondet_train_forward_stats(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Recompute the per-token statistics of ``forward_prepare`` (checkpoint replay).

    The arguments and output are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_forward_stats`;
    the call runs the backend package (the deterministic package's
    bits).

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    accumulation : str
        As in :func:`cake_mhc_nondet_train_forward_prepare`: the mode of the forward
        whose record this replays.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (``hidden=X.shape[2]``); a per-device
        default when omitted.

    Returns
    -------
    FP32 ``[T, 64]`` ``stats``: the record of
    :func:`cake_mhc_nondet_train_forward_prepare` for the same inputs and mode,
    computed by the same programs (no bit promise).
    """

    return _pkg().cake_mhc_nondet_train_forward_stats(
        X,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        stats_out=stats_out,
        workspace=workspace,
    )


@cake_api
def cake_mhc_nondet_train_forward_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    stats: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Residual mixing: ``Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])``.

    The arguments and output are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_forward_finish`;
    the call runs the backend package (the deterministic package's
    bits).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams (the ``forward_prepare`` input).
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous output of the external branch applied to ``U``.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of :func:`cake_mhc_nondet_train_forward_prepare`
        (or :func:`cake_mhc_nondet_train_forward_stats`).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous ``Y``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``.
    """

    return _pkg().cake_mhc_nondet_train_forward_finish(X, B, stats, out=out)


@cake_api
def cake_mhc_nondet_train_backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor] = None,
    grad_stats_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Branch-output gradient and the per-token gradient statistics: ``(dB, grad_stats)``.

    The arguments and outputs are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_backward_prepare`;
    the call runs the backend package (the deterministic package's
    bits: this pass has no cross-token reduction).

    Parameters
    ----------
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    B : torch.Tensor
        BF16 ``[T, H]`` contiguous branch output of the forward.
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward.
    dB : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous output.
    grad_stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 32]`` contiguous output.

    Returns
    -------
    ``(dB, grad_stats)``: BF16 ``dB`` of shape ``[T, H]`` and the FP32
    ``[T, 32]`` record that :func:`cake_mhc_nondet_train_backward_finish` consumes.
    """

    return _pkg().cake_mhc_nondet_train_backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out
    )


@cake_api
def cake_mhc_nondet_train_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    B: Optional[torch.Tensor],
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    accumulate: bool = False,
    workspace: Optional[torch.Tensor] = None,
    dX: Optional[torch.Tensor] = None,
    dfn: Optional[torch.Tensor] = None,
    dscale: Optional[torch.Tensor] = None,
    dbase: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Input gradient and parameter gradients: ``(dX, dfn, dscale, dbase)``.

    The arguments and outputs are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_backward_finish`;
    the call runs the backend package's backward, with no final reduction
    kernel.  Its sums run in a fixed order today, but no output is promised
    to repeat bit for bit from run to run.
    Each element of ``dfn`` adds its ``P <= 32`` per-chunk FP32 partials
    ``x_p`` in ascending chunk order as one FP32 sum from +0.0, the caller's
    ``dfn`` last under ``accumulate=True``: within ``gamma_n (|start| + sum
    |x_p|)`` of the exact sum (``n`` the additions, ``gamma_n = n u / (1 - n
    u)``, ``u = 2^-24``; see the module docstring).  ``dscale`` and ``dbase``
    are fixed-order FP32 sums of the token-owner pass's parameter rows.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams of the forward
        (``H = X.shape[2]``).
    dY : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous gradient of ``Y``.
    dU : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``U`` from the branch backward.
    B : torch.Tensor, optional
        BF16 ``[T, H]`` branch output of the forward; validated when given, not read.
    fn, scale, base : torch.Tensor
        The parameters of the forward (FP32 ``[24, 4H]``, ``[3]``, ``[24]``).
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward (saved, or recomputed with
        :func:`cake_mhc_nondet_train_forward_stats`).
    grad_stats : torch.Tensor
        FP32 ``[T, 32]`` record of :func:`cake_mhc_nondet_train_backward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    accumulation : str
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``"fp32"`` default,
        ``"fp32x2"`` opt-in); pass the mode of the forward.
    accumulate : bool
        ``False`` (default): ``dfn``, ``dscale`` and ``dbase`` are written.
        ``True``: the caller passes all three and the sums are added to their
        current values.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_nondet_train_backward_workspace`;
        a per-device default when omitted.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` contiguous output.
    dfn, dscale, dbase : torch.Tensor, optional
        Preallocated FP32 ``[24, 4H]``, ``[3]``, ``[24]`` contiguous outputs
        (required with ``accumulate=True``).

    Returns
    -------
    ``(dX, dfn, dscale, dbase)``: the BF16 input gradient and the FP32
    parameter gradients.
    """

    return _pkg().cake_mhc_nondet_train_backward_finish(
        X,
        dY,
        dU,
        B,
        fn,
        scale,
        base,
        stats,
        grad_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        accumulate=accumulate,
        workspace=workspace,
        dX=dX,
        dfn=dfn,
        dscale=dscale,
        dbase=dbase,
    )


@cake_api
def cake_mhc_nondet_train_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    save_stats: bool = True,
    workspace: Optional[torch.Tensor] = None,
    backward_workspace: Optional[torch.Tensor] = None,
):
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_nondet_train_finish`.

    The arguments and outputs are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_prepare`; the call
    makes the backend package's prepare node.  Its forward and its checkpoint
    replay run copies of the deterministic package's programs; its backward runs
    ``backward_finish`` (see :func:`cake_mhc_nondet_train_backward_finish`).

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_forward_prepare` (``X`` and the
        parameters may require gradients).
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    accumulation : str
        As in :func:`cake_mhc_nondet_train_forward_prepare`.  The forward, its checkpoint
        replay and the backward of this node run in this mode; the handle
        carries it, so :func:`cake_mhc_nondet_train_finish` takes no mode.
    save_stats : bool
        ``True`` (default): the per-token statistics are kept for the backward.
        ``False``: they are recomputed from ``X`` in the backward (checkpoint
        replay) by the same programs as the saved record (no bit promise).
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (``hidden=X.shape[2]``); a per-device
        default when omitted.
    backward_workspace : torch.Tensor, optional
        Caller-owned backward workspace of :func:`cake_mhc_nondet_train_backward_workspace`;
        a per-device default when omitted.

    Returns
    -------
    ``(U, handle)``: BF16 ``U`` of shape ``[T, H]`` and the handle of this
    prepare node for one :func:`cake_mhc_nondet_train_finish`.
    """

    return _pkg().cake_mhc_nondet_train_prepare(
        X,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        save_stats=save_stats,
        workspace=workspace,
        backward_workspace=backward_workspace,
    )


@cake_api
def cake_mhc_nondet_train_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    handle,
) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    The arguments and output are those of
    :func:`~cake_pplx.cake_mhc_det_train.cake_mhc_det_train_finish`; ``handle``
    is the handle :func:`cake_mhc_nondet_train_prepare` returned (any other
    handle, such as a ``cake_mhc_det_train_prepare`` one, raises ``TypeError``).

    Parameters
    ----------
    X : torch.Tensor
        The BF16 ``[T, 4, H]`` tensor passed to :func:`cake_mhc_nondet_train_prepare`.
    B : torch.Tensor
        BF16 ``[T, H]`` output of the external branch applied to ``U`` (may
        require gradients).
    handle : object
        The handle returned by :func:`cake_mhc_nondet_train_prepare`, for one finish.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, H]``.
    """

    return _pkg().cake_mhc_nondet_train_finish(X, B, handle)


# ----------------------------------------------------------------------------
# Pre handoff and standalone collapse (hidden 5120 on SM100 / SM103 / SM107)


@cake_api
def is_cake_mhc_nondet_train_handoff_supported(
    device: torch.device,
    hidden: int | None = None,
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has the pre-handoff routes and the standalone collapse.

    Parameters
    ----------
    device : torch.device
        Device to query (``False`` for a device that is not CUDA or not an exported architecture).
    hidden : int, optional
        Hidden size to query (5120 on SM100 / SM103); ``None`` queries the architecture only.

    Returns
    -------
    ``True`` when the device (at ``hidden``, when given) has the routes.
    """

    return _pkg().is_cake_mhc_nondet_train_handoff_supported(device, hidden)


@cake_api
def cake_mhc_nondet_train_handoff_forward_prepare(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    out: torch.Tensor | None = None,
    pre_next_out: torch.Tensor | None = None,
    stats_out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Pre-handoff forward of one sublayer: ``(U, pre_next, stats)``.

    The coefficients ``(pre_next, post, C)`` are those of
    :func:`cake_mhc_nondet_train_forward_prepare` on ``X``; the collapse takes the
    incoming ``pre_in`` instead of this sublayer's ``pre``:
    ``U[t, d] = BF16(sum_i pre_in[t, i] X[t, i, d])``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous residual streams; ``H = 5120`` on SM100 /
        SM103 (another size raises ``ValueError``).
    pre_in : torch.Tensor
        FP32 ``[T, 4]`` contiguous ``pre`` of the previous sublayer (a one-hot
        ``[1, 0, 0, 0]`` row per token for the first sublayer).
    fn, scale, base : torch.Tensor
        FP32 ``[24, 4H]``, ``[3]``, ``[24]`` parameters.
    rms_eps, hc_eps, sinkhorn_iters, accumulation : float, float, int, str
        As in :func:`cake_mhc_nondet_train_forward_prepare`.
    out, pre_next_out, stats_out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` ``U``, FP32 ``[T, 4]`` ``pre_next`` and FP32
        ``[T, 64]`` ``stats`` (contiguous).
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (:func:`cake_mhc_nondet_train_forward_workspace`,
        ``hidden=X.shape[2]``); a per-device default when omitted.

    Returns
    -------
    ``(U, pre_next, stats)``: BF16 ``U``, FP32 ``pre_next`` (this sublayer's
    ``pre``, the values of ``stats`` columns 25..28) and the ``[T, 64]`` record
    that :func:`cake_mhc_nondet_train_forward_finish`,
    :func:`cake_mhc_nondet_train_backward_prepare` and
    :func:`cake_mhc_nondet_train_handoff_backward_finish` consume.
    """

    return _pkg().cake_mhc_nondet_train_handoff_forward_prepare(
        X,
        pre_in,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        out=out,
        pre_next_out=pre_next_out,
        stats_out=stats_out,
        workspace=workspace,
    )


@cake_api
def cake_mhc_nondet_train_handoff_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    pre_in: torch.Tensor,
    dpre_next: torch.Tensor,
    B: torch.Tensor | None,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    accumulate: bool = False,
    workspace: torch.Tensor | None = None,
    dX: torch.Tensor | None = None,
    dpre_in: torch.Tensor | None = None,
    dfn: torch.Tensor | None = None,
    dscale: torch.Tensor | None = None,
    dbase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Pre-handoff backward of one sublayer: ``(dX, dpre_in, dfn, dscale, dbase)``.

    ``dX = BF16(dX_coef + pre_in_i dU + sum_j C_ij dY_j)`` (one FP32 sum, one
    rounding); the coefficient VJP takes ``dpre_next`` (with ``dpost`` and
    ``dC`` of ``grad_stats``) and does not depend on ``dU``;
    ``dpre_in[t, i] = sum_d dU[t, d] X[t, i, d]`` (one FP32 sum per element in a
    fixed order).  No run-to-run bit promise.

    Parameters
    ----------
    X, dY, dU, B, fn, scale, base, stats, grad_stats : torch.Tensor
        As in :func:`cake_mhc_nondet_train_backward_finish`.
    pre_in : torch.Tensor
        FP32 ``[T, 4]`` ``pre_in`` of the forward.
    dpre_next : torch.Tensor
        FP32 ``[T, 4]`` contiguous gradient of ``pre_next`` from its consumer.
    rms_eps, hc_eps, sinkhorn_iters, accumulation, accumulate, workspace : as in
        :func:`cake_mhc_nondet_train_backward_finish`.
    dX, dpre_in, dfn, dscale, dbase : torch.Tensor, optional
        Preallocated outputs (BF16 ``[T, 4, H]``, FP32 ``[T, 4]``, ``[24, 4H]``,
        ``[3]``, ``[24]``; the last three required with ``accumulate=True``).

    Returns
    -------
    ``(dX, dpre_in, dfn, dscale, dbase)``.
    """

    return _pkg().cake_mhc_nondet_train_handoff_backward_finish(
        X,
        dY,
        dU,
        pre_in,
        dpre_next,
        B,
        fn,
        scale,
        base,
        stats,
        grad_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        accumulate=accumulate,
        workspace=workspace,
        dX=dX,
        dpre_in=dpre_in,
        dfn=dfn,
        dscale=dscale,
        dbase=dbase,
    )


@cake_api
def cake_mhc_nondet_train_handoff_prepare(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulation: str = "fp32",
    save_stats: bool = True,
    workspace: torch.Tensor | None = None,
    backward_workspace: torch.Tensor | None = None,
):
    """Autograd-enabled ``handoff_forward_prepare``: ``(U, pre_next, handle)``.

    Pass ``handle`` to :func:`cake_mhc_nondet_train_finish` (unchanged) and
    ``pre_next`` to the next sublayer's ``pre_in`` or to
    :func:`cake_mhc_nondet_train_collapse`.  The node's backward runs
    :func:`cake_mhc_nondet_train_handoff_backward_finish` with the gradient
    ``pre_next`` received (zeros when it reached no loss).

    Parameters
    ----------
    X, pre_in, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_nondet_train_handoff_forward_prepare` (``X``,
        ``pre_in`` and the parameters may require gradients).
    rms_eps, hc_eps, sinkhorn_iters, accumulation, save_stats : as in
        :func:`cake_mhc_nondet_train_prepare`.
    workspace, backward_workspace : torch.Tensor, optional
        As in :func:`cake_mhc_nondet_train_prepare`.

    Returns
    -------
    ``(U, pre_next, handle)``.
    """

    return _pkg().cake_mhc_nondet_train_handoff_prepare(
        X,
        pre_in,
        fn,
        scale,
        base,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulation=accumulation,
        save_stats=save_stats,
        workspace=workspace,
        backward_workspace=backward_workspace,
    )


@cake_api
def cake_mhc_nondet_train_collapse_forward(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Standalone collapse: ``U[t, d] = BF16(sum_i pre_in[t, i] X[t, i, d])`` (no parameters, no workspace).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, H]`` contiguous streams (``H = 5120`` on SM100 / SM103).
    pre_in : torch.Tensor
        FP32 ``[T, 4]`` contiguous weights (the last sublayer's ``pre_next``).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, H]`` contiguous ``U``.

    Returns
    -------
    BF16 ``U`` of shape ``[T, H]``.
    """

    return _pkg().cake_mhc_nondet_train_collapse_forward(X, pre_in, out=out)


@cake_api
def cake_mhc_nondet_train_collapse_backward(
    X: torch.Tensor,
    pre_in: torch.Tensor,
    dU: torch.Tensor,
    *,
    dX: torch.Tensor | None = None,
    dpre_in: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""VJP of the standalone collapse: ``(dX, dpre_in)``.

    ``dX[t, i, d] = BF16(pre_in[t, i] dU[t, d])`` and
    ``dpre_in[t, i] = sum_d dU[t, d] X[t, i, d]`` (one FP32 sum per element).

    Parameters
    ----------
    X, pre_in : torch.Tensor
        The inputs of :func:`cake_mhc_nondet_train_collapse_forward`.
    dU : torch.Tensor
        BF16 ``[T, H]`` contiguous gradient of ``U``.
    dX, dpre_in : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, H]`` and FP32 ``[T, 4]`` contiguous outputs.

    Returns
    -------
    ``(dX, dpre_in)``.
    """

    return _pkg().cake_mhc_nondet_train_collapse_backward(
        X, pre_in, dU, dX=dX, dpre_in=dpre_in
    )


@cake_api
def cake_mhc_nondet_train_collapse(
    X: torch.Tensor,
    pre_in: torch.Tensor,
) -> torch.Tensor:
    """Autograd-enabled standalone collapse: ``U``; its backward is
    :func:`cake_mhc_nondet_train_collapse_backward`.

    Parameters
    ----------
    X, pre_in : torch.Tensor
        As in :func:`cake_mhc_nondet_train_collapse_forward` (both may require
        gradients).

    Returns
    -------
    BF16 ``U`` of shape ``[T, H]``.
    """

    return _pkg().cake_mhc_nondet_train_collapse(X, pre_in)


# The backend objects this module re-exports, resolved on first access so that
# importing this module imports no backend.
_LAZY = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeMHCNondetCollapseFunction",
    "CakeMHCNondetFinishFunction",
    "CakeMHCNondetHandoffPrepareFunction",
    "CakeMHCNondetPrepareFunction",
    "cake_mhc_nondet_train_reset_workspace",
    "cake_mhc_nondet_train_route_key",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(_pkg(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822
    "CakeMHCNondetCollapseFunction",  # noqa: F822
    "CakeMHCNondetFinishFunction",  # noqa: F822
    "CakeMHCNondetHandoffPrepareFunction",  # noqa: F822
    "CakeMHCNondetPrepareFunction",  # noqa: F822
    "cake_mhc_nondet_train_backward_finish",
    "cake_mhc_nondet_train_backward_prepare",
    "cake_mhc_nondet_train_backward_workspace",
    "cake_mhc_nondet_train_backward_workspace_bytes",
    "cake_mhc_nondet_train_collapse",
    "cake_mhc_nondet_train_collapse_backward",
    "cake_mhc_nondet_train_collapse_forward",
    "cake_mhc_nondet_train_finish",
    "cake_mhc_nondet_train_forward_finish",
    "cake_mhc_nondet_train_forward_prepare",
    "cake_mhc_nondet_train_forward_stats",
    "cake_mhc_nondet_train_forward_workspace",
    "cake_mhc_nondet_train_forward_workspace_bytes",
    "cake_mhc_nondet_train_handoff_backward_finish",
    "cake_mhc_nondet_train_handoff_forward_prepare",
    "cake_mhc_nondet_train_handoff_prepare",
    "cake_mhc_nondet_train_prepare",
    "cake_mhc_nondet_train_reset_workspace",  # noqa: F822
    "cake_mhc_nondet_train_route_key",  # noqa: F822
    "is_cake_mhc_nondet_train_handoff_supported",
    "is_cake_mhc_nondet_train_supported",
]
