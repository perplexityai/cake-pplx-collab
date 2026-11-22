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


Cake SiLU-gated RMSNorm training forward / backward (SM100 / SM103 / SM107)
of the non-deterministic family (``nondet``).  Every entry point runs only the
kernels of this family; the deterministic package is
:mod:`cake_pplx.cake_rmsnorm_silu_gate_det_train`.

Forward, per row of 128 of ``x`` and the matching row of the separate gate
input ``z`` (``[T, H, 128]`` or ``[T * H, 128]``, BF16; ``weight`` ``[128]``
shared by every token and head)::

    rstd = 1 / sqrt(mean(x^2) + eps)                     # FP32
    y    = BF16(((x * rstd) * weight) * silu(z))         # FP32, one rounding

The backward recomputes the statistics and the gate from ``x`` and ``z`` and
returns BF16 ``dx`` and ``dz`` and the weight gradient summed over every token
and head in FP32 (no FP64; the band list below says how each backward sums
and rounds it), then rounded to BF16 for a BF16 weight gradient.
The autograd function saves only its inputs, which keeps
``torch.utils.checkpoint(..., use_reentrant=False)`` free of FP32 temporaries.
No call synchronizes the host or the device; ``T`` and ``H`` are runtime
values.

``y``, ``dx``, ``dz`` and an FP32 weight gradient repeat bit for bit: the
same inputs on the same device give the same bits on every call, whatever the
order in which CTAs run and the work beside the call on other streams.  A
BF16 weight gradient repeats bit for bit too, except in the fused band
(below: past 512 rows on SM100, past 256 rows on SM103, up to 1000 rows per
SM; on SM107 past 64 up to 256 rows per SM), where its last bits may vary
from run to run.  ``y``, ``dx`` and ``dz`` are the same bits whatever the
grid, and so is the weight gradient but in the repeatable bands (below: a
BF16 weight gradient past 1000 rows per SM on SM100 and SM103, on SM107 past
512 rows up to 64 rows per SM and past 256; an FP32 weight gradient past 512
rows on every architecture but in the static band of SM100 and SM103, past
64 up to 128 rows per SM), where the rows and the SM count fix the grid
and so the weight gradient's bits: the same bits for the same rows and SM
count, possibly other bits on a device with another SM count.  From version
0.2.2 to 0.2.3 a call with an FP32 weight gradient of more than 512 rows
changed its ``dx``, ``dz`` and weight-gradient bits (to the deterministic
package's FP32 backward's bits), except on SM100 and SM103 past 64 and up
to 128 rows per SM, where 0.2.3 runs 0.2.2's static backward (0.2.2 ran it
up to 256 rows per SM there).  Version 0.2.4 changes no bits: a call with
``compute_grad_weight=False`` runs a program built without the weight
gradient, whose ``dx`` and ``dz`` are those of 0.2.3.  Version 0.2.5
changes no bits: in the repeatable band (SM107) packed rows (``grad_out``,
``x`` and ``z`` of head stride 128 and token stride ``H * 128``, more than
one token) run a 128-thread build of the strided program without the
weight gradient (8-row blocks: twice the CTAs of 16-row blocks at a small
row count), other layouts the 0.2.4 256-thread one; this build and the
fused band's strided program read packed rows at their linear offsets, and
a CTA of theirs with one block takes no integer division before its loads.
Version 0.2.6 changes no kernel and no bits: it lowers the host time of a
call (each module's launch arguments and tvm-ffi function are looked up
once, the launch reads the handle of torch's current stream without making a
stream object, and a caller's workspace is checked and viewed once per
tensor); a launch still runs on torch's current stream of its tensors'
device, as in 0.2.5.
Version 0.2.7 changes nothing in this package and no bits: the deterministic
package's backward without the weight gradient runs its 128-thread strided
build on rows of any layout up to 32 rows per SM, and on packed rows up to
128 rows per SM a build of packed rows only (SM100 and SM103), with the same
``dx`` and ``dz`` bits.
Version 0.2.8 changes nothing in this package and no bits: on SM107 the
deterministic package's backward without the weight gradient runs its
128-thread strided build on rows of any layout up to 96 rows per SM, and on
packed rows (and one token) up to 192 rows per SM a build of packed rows
only, with the same ``dx`` and ``dz`` bits.
Version 0.2.9 changes the ``dx`` and ``dz`` bits of some calls with
``compute_grad_weight=False`` (no other call): on SM100 and SM103 packed rows
(and one token of head stride 128, which such a call passes as packed rows)
up to 128 rows per SM run this package's copy of the deterministic package's
build of packed rows only in the place of the band's program without the
weight gradient, so their ``dx`` and ``dz`` are the repeatable backward's
bits; they were the bits of the call that computes a BF16 weight gradient,
from which they now differ in their last bits in the one-cluster bands and,
on SM103, in the fused band.  Every other call, and every call on SM107,
runs the program and gives the bits of 0.2.8.
Version 0.2.10 changes nothing in this package and no bits: the
deterministic package's backward that computes the weight gradient runs, up
to 16 rows, a program of one CTA and, from 17 to 256 rows on a device of at
least 16 SMs that is not a MIG instance, a program of one cluster of 16
CTAs, neither of which uses a workspace, with the same ``dx``, ``dz`` and
``dw`` bits on a device of at least 4 SMs.
Version 0.2.11 changes nothing in this package and no bits: the
deterministic package's forward runs, on short inputs whose rows are
packed (contiguous ``x`` and ``z``, or one token of head stride 128) up to a
cut of rows per SM (on SM107 not from 17 to 32 rows), a program of one row
block per CTA with no loop, with the same ``y`` bits.
Version 0.2.12 keeps the ``y`` bits of every call that 0.2.11 ran
correctly: on SM100 and SM103 the forward of packed rows (contiguous ``x``
and ``z``, or one token of head stride 128) up to 4 rows per SM on SM100
and 12 on SM103 runs a forward with 32 threads to a row (two to each group
of 8 elements, one 4-row block per 128-thread CTA) with the sum of squares
of the forward with 16 threads to a row, so its ``y`` bits; that forward
runs from there up to 36 rows per SM (past these cuts, with the rows in L2,
the forward with 32 threads to a row is the slower), and SM107 runs the
forward of 0.2.11.
A forward of ``x`` or ``z`` broadcast over more than one token (token
stride 0) runs a forward of other layouts where the forward of packed rows
would be one that reads packed rows at their linear offsets: 0.2.11 refused
such a call where it ran the forward with 16 threads to a row of packed rows
(up to 36 rows per SM on SM100 and SM103 and 65 on SM107; that forward
takes contiguous ``x`` and ``z`` only, so a ``ValueError``), and on SM107
past 280 rows per SM its pipelined forward of packed rows read rows past the
broadcast token.  At every other row count such a call runs, as in 0.2.11,
the forward of packed rows, which reads the token stride.
Version 0.2.14 changes the ``y`` bits of some calls: on SM100 the forward
of packed rows from 6501 rows per SM (fewer than ``PACKED_ROWS`` rows) runs the
pipelined forward (the deterministic package's forward program) in the place
of the mid forward, so the deterministic package's ``y`` bits where that
package's gate does not flush to 0, from which the mid forward's ``y``
differed in some elements.  Every other call, and every call on SM103 and
SM107, runs the program and gives the bits of 0.2.13.
A band runs a
backward whose weight gradient may vary only where the band-selection
measurement found the checkpointed training step (2 x forward + backward)
about 3 % or more faster than with the repeatable backward (near a band's
edges the gain can be smaller).  The kernels are those of the standalone package
``cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train``:

- The forward computes ``rstd`` with ``rsqrt.approx`` and the SiLU gate with
  ``ex2.approx`` of ``-z log2(e)`` and ``rcp.approx`` of a scaled ``1 + e``
  (the deterministic forward's split gate: a nonzero sigmoid down to ``z``
  about -88.72, where ``ex2.approx`` overflows); its bits need not be the
  deterministic package's.
  Where ``x`` and ``z`` hold packed rows (row ``n`` at element ``128 n``) a
  forward with 32 threads to a row runs up to 4 rows per SM on SM100 and 12
  on SM103, and a forward with 16 threads to a row from there up to 36 rows
  per SM on SM100 and SM103 and up to 65 on SM107; other layouts run a
  forward with 16 threads to a row up to 12 rows per SM on SM100 and SM103
  (59 on SM107); and past that, in bands of rows per SM, the pipelined
  forwards (builds of the deterministic package's forward program with a
  gate that does not flush), forwards whose CTAs each walk several row
  blocks and, for other layouts past 700 rows per SM on SM100 and SM103 (480
  on SM107), a forward with 8 threads to a row and one CTA per 8-row block.
  An ``x`` or ``z`` broadcast over more than one token (token stride 0)
  counts as packed rows, but where the band of packed rows runs a forward
  that reads rows at their linear offsets (the 32- and 16-thread forwards of
  packed rows, and on SM107 the pipelined forward of packed rows past 280
  rows per SM) it runs the forward of other layouts.
  Each forward program's arithmetic is fixed, but the programs
  associate a row's sum of squares differently, on FP32 FMA chains (the
  lanes of ``.f32x2`` FMA chains where a program runs two or four: the
  16-thread forwards four 2-term chains of 16 lanes added in pairs, then
  folded by XOR shuffles 1, 2, 4, 8, and the 32-thread forward the same
  chains on its 16 lane pairs, then folded by XOR shuffles 2, 4, 8, 16, the
  same additions; the mid forward, the extra-long forward
  but on SM100 and the SM107 pipelined forward four 4-term chains of 8 lanes
  added in pairs, then folded by 1, 2, 4; the plain forward and the
  pipelined forward on SM100 and SM103 two 8-term chains of 8 lanes, the
  even and the odd elements, added, then folded by 1, 2, 4; the long
  forward, and the extra-long forward on SM100, 16-term chains of 8 lanes
  folded by 1, 2, 4; the row count, the SM count and the layout pick the
  program), so ``y``'s bits can
  differ between bands while each program repeats them for the same inputs
  on the same device.
- The backward is one kernel; each backward program repeats ``dx`` and
  ``dz`` bit for bit for the same inputs on the same device, but the
  programs' ``.f32x2`` FMA chains for a row's sums differ by program and by
  architecture, so their last bits can differ between bands (the
  deterministic package's backward need not give these bits; the
  repeatable bands give its bits).  A call of few rows
  runs one thread-block cluster of 16 CTAs, which adds the weight-gradient
  partials of each column in a fixed order in FP32, rounds that sum once
  (to BF16 for a BF16 weight gradient) and uses no workspace.  The short
  backward (256-thread CTAs, every CTA's FP32 partials added in rank order)
  runs a BF16 weight gradient or (built without it)
  ``compute_grad_weight=False`` up to 256 rows;
  its weight gradient repeats its bits, which need not be the deterministic
  package's.  The exact tiny backward (its FP32 weight-gradient terms summed
  in plain FP32 words in an order of its own)
  runs an FP32 weight gradient up to 512 rows on every architecture and a BF16
  one (or, built without it, ``compute_grad_weight=False``) from 257 up to
  512 rows on SM100 and
  SM107.  Past those rows the rows (``T * H``) and the SM count pick the
  kernel by bands of rows per SM.  Past 512 rows an FP32 weight gradient
  runs the repeatable backward on every architecture (below), but on SM100
  and SM103 past 64 and up to 128 rows per SM the static backward, where the
  band-selection measurement found it saving about 2 to 7 % (SM100) and 5 to
  8 % (SM103) of the checkpointed training step.  The static backward runs
  over the chunks of rows of the deterministic backward at 2e989556 (8-row
  blocks, ``max(1, ceil(blocks / 8192))`` blocks a chunk), at most 8 CTAs
  per SM, CTA ``b`` running the chunks ``b``, ``b + grid``, ...: the chunks'
  FP32 weight-gradient partials are added exactly, as signed 64-bit integer
  windows of their FP32 values (an addition of integers does not depend on
  its order); the CTA that arrives last rounds the exact sum once into the
  weight gradient, in integer arithmetic (no FP64).  So that weight gradient
  is the same bits at any grid, and an infinite or ``NaN`` partial gives its
  column the class the exact sum gives it.
- Past the one-cluster backwards, a BF16 weight gradient (or, built without
  it, ``compute_grad_weight=False``) runs the fused backward on every
  architecture: past 512 rows on SM100, past 256 rows on SM103, up to 1000
  rows per SM; on SM107 past 64 up to 256 rows per SM, where it makes the
  step faster than the repeatable backward.  An FP32 weight gradient runs no
  fused backward.  It
  runs the deterministic package's fused backward's schedule and row
  arithmetic: 256-thread CTAs (16 lanes a row, 16-row blocks,
  ``min(ceil(rows / 16), 4 * SMs)`` of them), which claim the blocks in chunks
  when there are more blocks than CTAs, with that backward's approximate gate
  and ``rstd``, so ``dx`` and ``dz`` are the same bits at any grid (a row's
  sums run on FMA chains of its own, set per architecture, so their last
  bits may differ from that package's).  It
  adds each thread's FP32 runs of at most 16 rows into FP32 words and each
  CTA's FP32 column sums into 8 FP32 replicas with FP32 atomic adds; the CTA
  that arrives last sums the replicas in FP32 and rounds the sum to BF16 (no
  FP64).  The order of that sum depends on scheduling, so the BF16 weight
  gradient's last bits may vary from run to run.
- Past 1000 rows per SM a BF16 weight gradient (or, built without it,
  ``compute_grad_weight=False``) runs the repeatable backward on SM100 and
  SM103 (on SM107 past 512 rows up to 64 rows per SM and past 256), this
  package's copy of the deterministic package's BF16
  backward: the fused backward's grid and row arithmetic, its CTAs claiming
  the blocks in chunks whose rows are fixed by the row count and the grid,
  each thread's FP32 runs of at most 16 rows of a chunk added into plain FP32
  words, each CTA adding a column's FP32 word exactly into 64-bit fixed-point
  digits in the workspace and the sum rounded once by the CTA that arrives
  last.  Its
  ``dx``, ``dz`` and weight gradient are the deterministic package's bits,
  the same for the rows and the SM count.  Packed rows (``grad_out``, ``x``
  and ``z`` each with head stride
  128 and token stride ``H * 128``, more than one token) run its copy of
  the deterministic package's look-ahead program, the same build, from
  2560 rows per SM on SM100 and SM103 and from 1536 on SM107, with the same
  bits.
- Past 512 rows an FP32 weight gradient runs, on every architecture but in
  the static band of SM100 and SM103 (above), the repeatable backward of an
  FP32 weight gradient, this package's copy of the
  deterministic package's FP32 backward: the same grid, chunks, FP32 runs
  and fixed-point digits as the BF16 one, the CTA that arrives last
  rounding each column's exact sum once to FP32 (adding the accumulator's
  value first when accumulating), so its ``dx``, ``dz`` and weight gradient
  are the deterministic package's FP32 bits, the same for the rows and the
  SM count; packed rows run its look-ahead program from the same cuts (2560
  rows per SM on SM100 and SM103, 1536 on SM107; 378880 rows on a 148-SM
  B200, 325632 on a 212-SM SM107 device), with the same bits.  An infinite
  or ``NaN`` FP32 word of a column gives that column ``NaN`` where a word is
  ``NaN`` or both infinities occur, else that infinity; so a column whose
  terms are finite but whose FP32 run or chunk word overflows gets an
  infinity (or a ``NaN``) where the exact sum stays finite.  The repeatable
  backward of a BF16 weight gradient has the same property.
- A call with ``compute_grad_weight=False`` takes the BF16 bands and runs the
  band's program built without the weight gradient (all weight-gradient
  work removed at build time), so its ``dx`` and ``dz`` are the bits of the
  BF16 call that computes the weight gradient (but on the packed rows of the
  build of packed rows only, below): in the one-cluster bands that
  cluster program without its column sums and cluster synchronization; a
  strided program (CTAs walking their row blocks with no claim, no ticket
  and no workspace: 256-thread CTAs of 16-row blocks, but 128-thread CTAs of
  8-row blocks on the repeatable band's packed rows, more than one token)
  in the fused band up
  to 512 rows per SM on SM100 and 448 on SM103, and in the fused and the
  repeatable bands on SM107; the
  claimed program on the workspace past those cuts and in the repeatable
  band on SM100 and SM103; where packed rows run the look-ahead program,
  that program without the weight gradient.  On SM100 and SM103, on packed
  rows (and one token of head stride 128, whose token stride is never read:
  the call passes it as ``H * 128``) up to 128 rows per SM, every band runs
  instead the repeatable band's build of packed rows only (128-thread CTAs
  of 8-row blocks with no claim, no ticket and no workspace, no stride
  test, every row read at its linear offset, a row's two reductions across
  its lanes overlapped): its ``dx`` and ``dz`` are the repeatable backward's
  bits, which in the one-cluster bands, and on SM103 in the fused band, can
  differ in their last bits from those of the call that computes the weight
  gradient.

Accuracy: the tests hold every floating output (``y``, ``dx``, ``dz`` and
either weight gradient, accumulated or not) to the PyTorch BF16 naive
implementation of the same operation (every op in BF16, the gradients from
PyTorch autograd), each against a float64 reference computed from the BF16
inputs: over 32 random inputs of a case, in the L2 and in the max norm of the
error, the median and the largest error of the output are each at most 1.005
times the naive implementation's, and 0 on every input where the naive
implementation's is 0.  No output has an error budget of its own, and the
tests compare no output of this package with the deterministic package's.

Mode and dispatch rules:

1. The mode is part of every name.  Every public name of the deterministic
   module :mod:`cake_pplx.cake_rmsnorm_silu_gate_det_train` (backend package
   ``cake_pplx._backends.cake_rmsnorm_silu_gate_det_train``) carries
   ``det``; every public name of the non-deterministic module
   :mod:`cake_pplx.cake_rmsnorm_silu_gate_nondet_train` (backend package
   ``cake_pplx._backends.cake_rmsnorm_silu_gate_nondet_train``)
   carries ``nondet``.  The module constants ``ARCHES`` and
   ``SUPPORTED_COMPUTE_CAPABILITIES`` are exempt: they keep their names,
   without a mode word, in both modules and both backend packages.
   Each module imports its backend package on first use; its autograd class and
   its architecture constants are the backend package's objects.
2. No entry point takes a ``deterministic`` keyword: passing one is a
   ``TypeError``.  A call of one package never runs the other package.
3. Under ``torch.use_deterministic_algorithms(True)`` a backward call of this
   module that computes a BF16 weight gradient in the fused band (above)
   raises PyTorch's alert immediately before that launch (``RuntimeError``;
   with ``warn_only=True`` a ``UserWarning``, then the call runs); no other
   call of this module alerts.  The check runs on the host: a CUDA Graph
   checks it when it is captured, not when it is replayed.
4. Autograd: ``CakeRMSNormSiLUGateDetFunction`` and
   ``CakeRMSNormSiLUGateNondetFunction`` save only ``x``, ``z`` and ``weight``
   and run the backward of their own package.  A backward workspace serves
   only its own package (FP32 for the deterministic package, ``torch.uint8``
   for the non-deterministic one): each backward refuses the other package's
   workspace (``ValueError``) in every call, also with no rows.
5. The former names are gone, with no alias or shim: the former modules fail
   at import, and the former names without a mode word are attributes of
   neither module.

The non-deterministic backward's workspace
(``cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, device)``, a
zero-filled ``torch.uint8`` tensor of 29192 bytes on SM100 and SM103 and 36360
bytes on SM107, whatever ``T * H``: the largest layout of the architecture's
backward kernels) holds, for a chunked backward, the integer windows of the
exact sum, 128 FP32 class words, the arrival ticket and the claim counter,
and, for a row-claims or the fused backward, its FP32 replicas (4 x 128 or
8 x 128 FP32 sums) and the same two words (2056 or 4104 bytes), and, for a
repeatable backward (either weight gradient), its fixed-point digits, a
flag word a column and the same two words (18952 bytes); every launch of the
backward uses one, also a claimed program with ``compute_grad_weight=False``,
but the one-cluster backwards of a call of few
rows and the strided programs without the weight gradient, which read
none (a buffer passed to them is checked and left as it is).  Its size
does not depend on ``compute_grad_weight``.  The kernel leaves it zero:
allocate it once, reuse it
across calls, shapes, both weight-gradient dtypes and CUDA Graph replays
without a reset, and give each stream that may run a backward at the same
time its own buffer.  A buffer that did not come from that factory, or that
anything else wrote into, must be zeroed with
``cake_rmsnorm_silu_gate_nondet_train_reset_workspace(ws)`` before its next
backward.  It is not interchangeable with the deterministic package's FP32
buffer: each backward refuses the other package's buffer (``ValueError``) in
every call, also with no rows.
Calls without a workspace use one buffer per device, stream and backward
kernel that takes one (one for each static, fused, repeatable, row-claims
and claimed backward its architecture routes, and for each claimed program
without the weight gradient; none for the one-cluster backwards and the
strided programs), kept for later calls; the autograd node retains nothing else.  They also keep a reserve of 64 zero-filled buffers per device (about
1.8 MiB on SM100 and SM103, 2.2 MiB on SM107, filled outside any capture): a call during CUDA Graph capture takes
one per capture and capture stream from it and keeps it for the life of the
process, so a replay launches no fill kernel (with none ready, the graph
zero-fills a buffer of its own).

Beside kernels resident on other streams: an SM
serves a CTA only once it can switch to a shared-memory carveout that holds
the CTA's dynamic shared memory, so a kernel on another stream whose CTAs hold
every SM with a small carveout (for example a long-running kernel with little
or no shared memory and the default carveout) can delay the start of a kernel
of this module until it exits.  Setting
``cudaFuncAttributePreferredSharedMemoryCarveout`` to
``cudaSharedmemCarveoutMaxShared`` on such a resident kernel, or giving it a
large dynamic shared-memory allocation, lets the backward start beside it.

Usage::

    y = cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)                 # autograd

    y = cake_rmsnorm_silu_gate_nondet_train_forward(x, z, weight, eps)
    dx, dz, dw = cake_rmsnorm_silu_gate_nondet_train_backward(dy, x, z, weight, eps)

    # FP32 main-grad accumulation and a caller-owned workspace
    ws = cake_rmsnorm_silu_gate_nondet_train_backward_workspace(T * H, x.device)
    cake_rmsnorm_silu_gate_nondet_train_backward(
        dy, x, z, weight, eps, grad_weight=main_grad, accumulate_grad_weight=True,
        workspace=ws,
    )
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

from ._cake_pplx_api_rmsnorm_silu_gate import cake_api


# The backend package is imported on first use.
@functools.cache
def _backend() -> Any:
    from ._backends import cake_rmsnorm_silu_gate_nondet_train  # noqa: PLC0415 - the backend package, imported on first use

    return cake_rmsnorm_silu_gate_nondet_train


# The backend package's own objects (the autograd class and the architecture
# constants), resolved on first access by the module ``__getattr__`` below.
_BACKEND_OBJECTS = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormSiLUGateNondetFunction",
)

# The stable entry points are real ``def``s here (public-API tooling reads the
# decorated signatures of this module); the kernels and host bodies live in the
# self-contained backend package next to its generated CUDA sources.  The
# autograd function is the backend package's class.


@cake_api
def is_cake_rmsnorm_silu_gate_nondet_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device with every route of the
    non-deterministic kernels.

    The deterministic package's query is
    :func:`cake_pplx.cake_rmsnorm_silu_gate_det_train.is_cake_rmsnorm_silu_gate_det_train_supported`.
    The answer does not depend on ``torch.use_deterministic_algorithms``.

    Parameters
    ----------
    device : torch.device
        The device to check.

    Returns
    -------
    bool
        ``True`` when the non-deterministic package has kernels for ``device``.
    """

    return _backend().is_cake_rmsnorm_silu_gate_nondet_train_supported(device)


@cake_api
def cake_rmsnorm_silu_gate_nondet_train_forward(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""SiLU-gated RMSNorm forward: ``y = BF16(((x * rstd) * weight) * silu(z))``.

    ``rstd = 1 / sqrt(mean(x^2) + eps)`` over the last dimension (128), in FP32;
    ``silu(z) = z / (1 + exp(-z))`` in FP32; one rounding to BF16 at the end.
    The non-deterministic package's forward program: ``rstd`` from
    ``rsqrt.approx`` and the gate from ``ex2.approx`` and ``rcp.approx`` (the
    deterministic forward's split gate), every program's arithmetic fixed, so
    ``y`` repeats bit for bit for the same inputs on the same device (its bits
    need not be the deterministic package's; that one is
    :func:`cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_forward`).
    Where ``x`` and ``z`` hold packed rows (row ``n`` at element ``128 n``)
    a forward with 32 threads to a row runs up to 4 rows per SM on SM100 and
    12 on SM103, and one with 16 threads to a row from there up to 36 rows
    per SM on SM100 / SM103 and up to 65 on SM107; one with 16 threads to a
    row runs other layouts up to 12 on SM100 / SM103 (59 on SM107); and past
    that, in bands of rows per SM, the pipelined
    forwards, forwards whose CTAs each walk several row blocks and (other
    layouts past 700 rows per SM on SM100 / SM103, 480 on SM107) a forward of
    one CTA per 8-row block; the programs associate a row's sum of
    squares differently (the row count, the SM count and the layout pick the
    program), so ``y``'s bits can differ between bands.  An ``x`` or ``z``
    broadcast over more than one token counts as packed rows, but where the
    band of packed rows runs a forward that reads rows at their linear
    offsets (the 32- and 16-thread forwards of packed rows, and on SM107 the
    pipelined forward of packed rows past 280 rows per SM) it runs the
    forward of other layouts.  The forward does not alert under
    ``torch.use_deterministic_algorithms(True)``.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[..., 128]`` input (``[T, H, 128]`` or ``[T * H, 128]``).
    z : torch.Tensor
        BF16 gate input of the same shape as ``x`` (its own strides).
    weight : torch.Tensor
        BF16 ``[128]`` weight shared by every token and head.
    eps : float
        Variance epsilon (FP32 inside the kernel).
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape (``out`` when given).
    """

    return _backend().cake_rmsnorm_silu_gate_nondet_train_forward(
        x, z, weight, eps, out=out
    )


@cake_api
def cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
    rows: int, device: torch.device
) -> int:
    """Bytes of the caller-owned workspace of the non-deterministic backward
    for ``rows`` rows (``T * H``) on ``device``: the integer windows of the
    exact weight-gradient sum, 128 FP32 class words, the arrival ticket and the
    claim counter (the fused backward's FP32 replicas and two words and
    the repeatable backwards' fixed-point digits, flag words and two words fit
    within it), whatever ``rows``: about 29 KiB on SM100 and SM103 (29192
    bytes) and 36 KiB on SM107 (36360 bytes).

    The deterministic package's FP32 buffer has
    :func:`cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes`.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.

    Returns
    -------
    int
        The workspace size in bytes.
    """

    return _backend().cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes(
        rows, device
    )


@cake_api
def cake_rmsnorm_silu_gate_nondet_train_backward_workspace(
    rows: int, device: torch.device
) -> torch.Tensor:
    """A caller-owned workspace of the non-deterministic backward for ``rows``
    rows (``T * H``): a zero-filled ``torch.uint8`` tensor of
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes.

    The non-deterministic backward leaves it zero after every call, so it
    serves every later call on its stream (any ``T`` and ``H``, BF16 or FP32
    weight gradient) and every CUDA Graph replay without a reset.  A backward
    that may run at the same time on another stream needs its own workspace.
    It is not the deterministic package's FP32 buffer
    (:func:`cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward_workspace`):
    each backward refuses the other package's buffer in every call, also with
    no rows.

    Parameters
    ----------
    rows : int
        Number of rows of the backward call, ``T * H``.
    device : torch.device
        The CUDA device the backward runs on.

    Returns
    -------
    torch.Tensor
        The zero-filled ``torch.uint8`` workspace on ``device``.
    """

    return _backend().cake_rmsnorm_silu_gate_nondet_train_backward_workspace(
        rows, device
    )


@cake_api
def cake_rmsnorm_silu_gate_nondet_train_reset_workspace(
    workspace: torch.Tensor,
) -> None:
    """Return a caller-owned workspace of the non-deterministic backward to its
    first-use state.

    Zeroes the integer windows, the class words, the FP32 replicas, the
    fixed-point digits, the flag words, the arrival ticket and the claim
    counter (a contiguous ``torch.uint8`` CUDA tensor of at least
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes` bytes
    starting at a 16-byte aligned address; all of it is zeroed) with one fill on the current stream of its device, without a
    host synchronization.  The non-deterministic backward leaves its workspace
    zero, so reuse across calls, shapes, weight-gradient dtypes and CUDA Graph
    replays needs no reset; a buffer that did not come from
    :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, or that anything
    else wrote into (a launch that did not complete, another use of the
    buffer), needs one before its next backward.  Passing the deterministic
    package's FP32 buffer here is a ``ValueError``.  The reset does not depend
    on ``torch.use_deterministic_algorithms`` (a fill has
    no reduction).

    Parameters
    ----------
    workspace : torch.Tensor
        The caller-owned ``torch.uint8`` workspace of the non-deterministic
        backward.
    """

    _backend().cake_rmsnorm_silu_gate_nondet_train_reset_workspace(workspace)


@cake_api
def cake_rmsnorm_silu_gate_nondet_train_backward(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    grad_x: Optional[torch.Tensor] = None,
    grad_z: Optional[torch.Tensor] = None,
    grad_weight: Optional[torch.Tensor] = None,
    accumulate_grad_weight: bool = False,
    compute_grad_weight: bool = True,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r"""SiLU-gated RMSNorm backward: ``dx``, ``dz`` and the weight gradient.

    With ``n = x * rstd``, ``s = sigmoid(z)`` and ``a = dy * silu(z)``::

        dz = BF16(dy * (n * weight) * s * (1 + z * (1 - s)))
        dx = BF16(rstd * a * weight - x * rstd^3 * mean(a * weight * x))
        dw = sum over every token and head of a * n

    in FP32, the statistics and the gate recomputed from ``x`` and ``z``.

    One kernel of the non-deterministic package (the rows, the SM count and
    the dtype of ``dw`` pick it; see the module): each backward program
    repeats ``dx`` and ``dz`` bit for bit, but the programs' ``.f32x2`` FMA
    chains for a row's sums differ by program and by architecture, so their
    last bits can differ between bands (the fused and the repeatable bands
    run the deterministic package's row arithmetic; the repeatable bands give
    its bits).  On SM100 and SM103 an FP32 ``dw`` past 64 and up to 128 rows
    per SM runs the static backward over the chunks of rows of the
    deterministic backward at 2e989556: the chunks' FP32 weight-gradient
    partials are added exactly as 64-bit integer windows; the CTA that
    arrives last rounds the exact sum once into ``dw`` in integer arithmetic
    (adding ``grad_weight``'s value first when accumulating), the same bits
    at any grid.  A call of few
    rows runs one cluster of 16 CTAs that adds the partials of a ``dw``
    column in a fixed order in FP32 and uses no workspace: up to 256 rows
    with a BF16 ``dw`` or (built without it) ``compute_grad_weight=False``
    (the short backward, whose ``dw`` repeats its bits, which need not be the
    deterministic backward's), else up to 512 rows (the exact tiny backward,
    its FP32 terms summed in plain FP32 words in an order of its own; SM103
    runs
    it for an FP32 ``dw`` only).  The fused band (a BF16 ``dw`` or, built
    without it, ``compute_grad_weight=False`` past 512 rows on SM100 and past
    256 rows on SM103 up to 1000 rows per SM, on SM107 past 64 up to 256 rows
    per SM; an FP32 ``dw`` has none) runs
    the fused backward, the deterministic package's fused backward's schedule
    and row arithmetic (``dx`` and ``dz`` the same bits at any grid) with
    FP32 runs of rows added into FP32 words and the
    CTAs' FP32 column sums added into FP32 replicas with FP32 atomic adds:
    the last bits of that ``dw`` may vary from run to run.
    Past 1000 rows per SM (on SM107 past 512 rows up to 64 rows per SM and past
    256) the repeatable backward, this package's copy of the deterministic
    package's backward, adds each thread's FP32 runs of rows into plain FP32
    words and each CTA's FP32 column words exactly into 64-bit fixed-point
    digits in the workspace: ``dx``, ``dz`` and
    ``dw`` the deterministic package's bits, the same for the rows and the SM
    count (packed rows from 2560 rows per SM on SM100 and SM103 and from 1536
    on SM107 run its look-ahead program, with the same bits).  Past 512 rows
    an FP32 ``dw`` runs, on every architecture but in the static band of
    SM100 and SM103, the repeatable backward of an
    FP32 weight gradient, this package's copy of the deterministic package's
    FP32 backward: the same runs, chunks and fixed-point digits, the CTA that
    arrives last rounding each column's exact sum once to FP32 (adding
    ``grad_weight``'s value first when accumulating), so ``dx``, ``dz`` and
    ``dw`` are the deterministic package's FP32 bits, the same for the rows
    and the SM count (packed rows from the same cuts run its look-ahead
    program, with the same bits); a column whose terms are finite but whose
    FP32 run or chunk word overflows gets an infinity (or a ``NaN``) there.
    Every other output repeats bit for bit, whatever the order in which CTAs
    run and, but in the repeatable bands, whatever the grid.
    The deterministic backward is
    :func:`cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det_train_backward`.
    Under ``torch.use_deterministic_algorithms(True)`` a call that computes a
    BF16 ``dw`` in the fused band raises PyTorch's alert (``RuntimeError``;
    a ``UserWarning`` under ``warn_only=True``, then the call runs) after the
    argument checks, immediately before the launch; no other call alerts.

    Parameters
    ----------
    grad_out, x, z : torch.Tensor
        BF16 tensors of one ``[..., 128]`` shape (any strides, see the module).
    weight : torch.Tensor
        BF16 ``[128]``.
    eps : float
        The forward's epsilon.
    grad_x, grad_z : torch.Tensor, optional
        Preallocated BF16 outputs of ``x``'s shape.
    grad_weight : torch.Tensor, optional
        A BF16 or FP32 ``[128]`` tensor that receives the weight gradient
        (the sum rounded to FP32, and to BF16 from there for a BF16
        tensor).  Without it a new BF16 tensor is returned.
    accumulate_grad_weight : bool
        Add the gradient to ``grad_weight``'s current value before the
        rounding (exactly in the static band of an FP32 weight gradient and
        in the repeatable bands of either weight gradient; in FP32 before
        the rounding in the small-row and the fused bands;
        needs ``grad_weight``).
        A call with no rows leaves the accumulator as it is.
    compute_grad_weight : bool
        ``False`` returns ``None`` for the weight gradient and runs the BF16
        band's program built without it, whose ``dx`` and ``dz`` are the bits
        of the BF16 call that computes it; it touches no ``grad_weight``.  On
        SM100 and SM103 packed rows (and one token of head stride 128) up to
        128 rows per SM run instead the repeatable band's build of packed rows
        only, with the repeatable backward's ``dx`` and ``dz`` bits (in the
        one-cluster bands, and on SM103 in the fused band, their last bits can
        differ from the call that computes the weight gradient); it uses no
        workspace.  Otherwise a
        call of up to 512 rows (256 on SM103) runs a one-cluster program, a
        call in the fused band up to 512 rows per SM on SM100 and 448 on
        SM103, or in the fused or the repeatable band on SM107 (but for the
        look-ahead program's packed rows), a strided program; neither uses a
        workspace (a ``workspace`` passed is checked and left as it is).  The
        claimed program (past those cuts, and in the repeatable band on SM100
        and SM103) and the look-ahead one (packed rows of the repeatable band
        from 2560 rows per SM on SM100 and SM103 and 1536 on SM107) take the
        arrival ticket and claim blocks from the claim counter, so they use a
        workspace too (and leave it zero).
    workspace : torch.Tensor, optional
        Caller-owned zero-filled ``torch.uint8`` buffer from
        :func:`cake_rmsnorm_silu_gate_nondet_train_backward_workspace`, which the
        kernel leaves zero (zero it with
        :func:`cake_rmsnorm_silu_gate_nondet_train_reset_workspace` after anything
        else wrote into it); the deterministic package's FP32 buffer is a
        ``ValueError``, also in a call with no rows.  When omitted, the
        backward uses one zero-filled buffer per device, stream and backward
        kernel, kept for later calls (during CUDA Graph capture, one
        zero-filled buffer per capture and capture stream, from a reserve
        those calls keep).  One buffer per backward that
        may run at the same time on another stream.
        The call keeps a weak reference to a caller's workspace and its checked
        buffers (``torch.utils.swap_tensors`` refuses a tensor with a weak
        reference); a workspace rebound to other memory is checked again at its
        next call.

    Returns
    -------
    ``(grad_x, grad_z, grad_weight)``.
    """

    return _backend().cake_rmsnorm_silu_gate_nondet_train_backward(
        grad_out,
        x,
        z,
        weight,
        eps,
        grad_x=grad_x,
        grad_z=grad_z,
        grad_weight=grad_weight,
        accumulate_grad_weight=accumulate_grad_weight,
        compute_grad_weight=compute_grad_weight,
        workspace=workspace,
    )


@cake_api
def cake_rmsnorm_silu_gate_nondet(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable ``RMSNorm(x) * weight * SiLU(z)`` over the last dimension
    (128) on the non-deterministic kernels: a :class:`CakeRMSNormSiLUGateNondetFunction`
    node (the backend package's class), which saves only ``x``, ``z`` and
    ``weight``.

    Every output of the node repeats bit for bit but a BF16 weight gradient in
    the fused band, whose last bits may vary from run to run; under
    ``torch.use_deterministic_algorithms(True)`` the backward alerts
    immediately before computing the weight gradient in such a band, and
    nothing else alerts.  A non-reentrant
    checkpoint replays the forward through this function (the same ``y``).  The deterministic
    one is :func:`cake_pplx.cake_rmsnorm_silu_gate_det_train.cake_rmsnorm_silu_gate_det`.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[..., 128]`` input (``[T, H, 128]`` or ``[T * H, 128]``).
    z : torch.Tensor
        BF16 gate input of the same shape as ``x``.
    weight : torch.Tensor
        BF16 ``[128]`` weight shared by every token and head.
    eps : float
        Variance epsilon (FP32 inside the kernel).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.
    """

    return _backend().cake_rmsnorm_silu_gate_nondet(x, z, weight, eps)


def __getattr__(name: str) -> Any:
    if name in _BACKEND_OBJECTS:
        value = getattr(_backend(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BACKEND_OBJECTS))


__all__ = [
    "ARCHES",  # noqa: F822 - resolved by __getattr__
    "SUPPORTED_COMPUTE_CAPABILITIES",  # noqa: F822 - resolved by __getattr__
    "CakeRMSNormSiLUGateNondetFunction",  # noqa: F822 - resolved by __getattr__
    "cake_rmsnorm_silu_gate_nondet",
    "cake_rmsnorm_silu_gate_nondet_train_backward",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace",
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace_bytes",
    "cake_rmsnorm_silu_gate_nondet_train_forward",
    "cake_rmsnorm_silu_gate_nondet_train_reset_workspace",
    "is_cake_rmsnorm_silu_gate_nondet_train_supported",
]
