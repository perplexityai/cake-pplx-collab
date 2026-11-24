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

References of the route-weight row scaling tests, in PyTorch:

* :func:`scaled`: ``BF16(w * v)``, the FP32 product rounded once, the exact contract of ``y`` (``v = x``) and of
  ``dx`` (``v = dy``).
* :func:`tree_row_sum`: ``dw`` in the order the kernel states (pairs within a load of 8 columns, the loads of a lane
  joined in the order of :func:`pairwise_plan`, the even and odd halves, then five XOR shuffles across the row's 32
  lanes), each addition one FP32 rounding.  The products of two BF16 values are exact in FP32, so this gives ``dw``
  bit for bit.
* :func:`fp64_row_sum` and :func:`fp32_row_sum`: the FP64 reference of ``dw`` and the FP32 row sum
  ``(dy.float() * x.float()).sum(1)`` that ``dw`` is judged beside.  FP64 is used here only, never in the kernels.
* :func:`issue_inputs`: the draws of the CPU reference script of the issue (seed 0), so the FP32 row sum's error
  can be compared with the value the issue prints (8.85e-08 at H 4096, 8.72e-08 at H 3584).

:data:`REPORT` collects the measured lines (accuracy, memory) that ``tests/conftest.py`` prints in the terminal
summary, passing tests included.
"""

from __future__ import annotations

import torch

# Lines printed at the end of the run (``tests/conftest.py``).
REPORT: list[str] = []

# 32 lanes own a row; a lane loads 8 BF16 columns at a time, as 4 pairs (even, odd).
LANES = 32
VEC = 8


def report(line: str) -> None:
    """Keep ``line`` for the terminal summary and print it (shown with ``-s``)."""
    REPORT.append(line)
    print(line)


def bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of a BF16 or FP32 tensor, as an integer view of the same storage (any strides)."""
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` have the same dtype, shape and bits (``+0`` and ``-0`` differ, NaNs compare by bits)."""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


def scaled(w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``BF16(w[:, None] * v)``: the FP32 product of the FP32 weight and the BF16 value, rounded once to BF16."""
    return (w[:, None] * v.float()).bfloat16()


def pairwise_plan(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """The kernel's order of joining a lane's ``n`` loads: the pairwise sum of ``n`` values pushed in order, as a
    binary counter.  Each push and each join appends one value to a list, and a join names its two operands by their
    positions in that list.  Entry ``k`` lists the joins after push ``k``; the last entry also joins what is left,
    newest last.  The sum is the last value, at position ``2 n - 2``."""
    stack: list[tuple[int, int]] = []
    joins: list[tuple[tuple[int, int], ...]] = []
    count = 0
    for k in range(n):
        step = []
        stack.append((0, count))
        count += 1
        while len(stack) >= 2 and (stack[-1][0] == stack[-2][0] or k == n - 1):
            (_, b), (level, a) = stack.pop(), stack.pop()
            step.append((a, b))
            stack.append((level + 1, count))
            count += 1
        joins.append(tuple(step))
    return tuple(joins)


def tree_row_sum(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """``dw`` of the kernel bit for bit: ``sum over c of dy * x`` per row in the kernel's fixed order, FP32.

    Column ``c = (p * 32 + lane) * 8 + 2 k + h`` is load ``p`` of lane ``lane``, pair ``k`` and half ``h`` (even or
    odd).  Per lane, load and half: ``(p1 + p0) + (p3 + p2)`` (the kernel's FMA of the second product onto the first
    rounds as one FP32 addition of the two exact products); the loads joined by :func:`pairwise_plan`; the halves
    added; then the lanes ``l`` and ``l + 16``, ``l + 8``, ``l + 4``, ``l + 2`` and ``l + 1`` added in turn (the XOR
    shuffles; FP32 addition is commutative)."""
    rows, width = x.shape
    pieces = width // (LANES * VEC)
    products = (dy.float() * x.float()).reshape(rows, pieces, LANES, VEC // 2, 2)
    loads = (products[:, :, :, 1] + products[:, :, :, 0]) + (products[:, :, :, 3] + products[:, :, :, 2])
    sums: list[torch.Tensor] = []
    for p, joins in enumerate(pairwise_plan(pieces)):
        sums.append(loads[:, p])
        for a, b in joins:
            sums.append(sums[a] + sums[b])
    lane = sums[2 * pieces - 2]
    total = lane[..., 0] + lane[..., 1]
    for half in (16, 8, 4, 2, 1):
        total = total[:, :half] + total[:, half : 2 * half]
    return total[:, 0].contiguous()


def fp64_row_sum(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """The FP64 reference of ``dw``: ``sum over c of dy * x`` in FP64 (exact products, FP64 additions)."""
    return (dy.double() * x.double()).sum(1)


def fp32_row_sum(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """The FP32 row sum ``(dy.float() * x.float()).sum(1)`` (FP32 products and sums), on the inputs' device."""
    return (dy.float() * x.float()).sum(1)


def rel_l2(a: torch.Tensor, reference: torch.Tensor) -> float:
    """``||a - reference|| / ||reference||`` over the whole vector, in FP64."""
    a64, r64 = a.double(), reference.double().to(a.device)
    return float(((a64 - r64).norm() / r64.norm()).item())


def make_inputs(
    rows: int, width: int, *, seed: int, device: torch.device, weight: str = "rand"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x, dy, w)``: BF16 ``[rows, width]`` unit normals and FP32 ``[rows]`` route weights (``rand``: ``U[0, 1)``,
    the issue's draw; ``signed``: normals, any sign; ``ones``)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, width, device=device, generator=g).bfloat16()
    dy = torch.randn(rows, width, device=device, generator=g).bfloat16()
    if weight == "rand":
        w = torch.rand(rows, device=device, generator=g)
    elif weight == "signed":
        w = torch.randn(rows, device=device, generator=g)
    elif weight == "ones":
        w = torch.ones(rows, device=device)
    else:
        raise ValueError(f"weight is rand, signed or ones, got {weight!r}")
    return x, dy, w


def issue_inputs() -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``{H: (x, dy, w)}`` on the CPU for ``H`` 4096 and 3584, 65536 rows each: the draws of the issue's reference
    script (``torch.manual_seed(0)``, then its small gradient check's draws, then per width ``x``, ``dy`` as BF16 unit
    normals and ``w = rand``), from a generator of their own (the global generator is left alone).  The script's
    ``gradcheck`` draws no random numbers."""
    g = torch.Generator().manual_seed(0)
    torch.randn(7, 16, dtype=torch.float64, generator=g)  # the gradient check's x
    torch.rand(7, dtype=torch.float64, generator=g)  # its w
    torch.randn(7, 16, dtype=torch.float64, generator=g)  # its dy
    found = {}
    for rows, hidden in ((65536, 4096), (65536, 3584)):
        x = torch.randn(rows, hidden, generator=g).bfloat16()
        dy = torch.randn(rows, hidden, generator=g).bfloat16()
        w = torch.rand(rows, generator=g)
        found[hidden] = (x, dy, w)
    return found
