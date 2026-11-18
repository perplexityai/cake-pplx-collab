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

Forward outputs that share memory with an input, in both packages.

A caller's ``out``, ``rstd`` or ``residual_out`` whose bytes meet those of
``x``, ``residual`` or ``w`` (``out=x``, an in-place call, among them) is
written through a temporary, which the call copies into it after the launch.
Such a call therefore returns the bits of the same call with fresh outputs.
The kernels never get aliased pointers: a forward program may read an input
again after it stores an output.  The SM107 H = 2048 programs do so at a row
whose ``rstd`` their fast path leaves open: they read the row's ``x`` again
after storing its ``y``.

The GPU test runs every program window of the H = 2048 forward (BF16 and FP32
weights) with such rows among random ones: ``x = +-1`` in every column and
``eps = h - 1``, so ``mean + eps`` is exactly ``h``, an FP32 value whose
``h^(-1/2)`` lies within ``2^-45`` of a midpoint of two FP32 values; and zero
rows at ``eps = 0``.  It compares each overlapping call with the call on fresh
outputs: ``out=x``, ``out`` one row past ``x`` in one buffer, ``out`` holding
``w`` and, on the residual route, ``residual_out=x`` and ``out=residual``.
The test without a device checks the decision itself on CPU tensors.
"""

from __future__ import annotations

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

HIDDEN = 2048
# FP32 inputs ``h`` in [2, 4) whose h^(-1/2) lies within 2^-45 (relative) of a
# midpoint of two FP32 values (bit patterns); h - 1 is exact in FP32.
NEAR_MIDPOINT = (0x403A18E3, 0x4009F038)
# The stride between the constructed rows (``x = +-1``, or zero at eps 0).
EVERY = 5
# The rows past the lower edge of a route's open last window.
OPEN_WINDOW_EXTRA = 7
SEED = 2048
MODES = ("det", "nondet")
W_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}


def _package(mode: str):
    return cake_package(f"cake_rmsnorm_{mode}_train")


def _loader(mode: str):
    return cake_package(f"cake_rmsnorm_{mode}_train", "cake_jit")


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16 if tensor.element_size() == 2 else torch.int32)


def _same(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Bitwise equality (a NaN equals a NaN of the same bits)."""
    return got.shape == want.shape and torch.equal(_bits(got), _bits(want))


def _through_temporary(mode: str, tensor: torch.Tensor, reads) -> bool:
    """Whether the package's output rule writes ``tensor`` through a temporary
    (alignment aside), checked against the input byte ranges ``reads``."""
    kernel, result, user = _package(mode)._output_buffer(
        tensor,
        "out",
        shape=tuple(tensor.shape),
        device=tensor.device,
        dtype=tensor.dtype,
        alignment=1,
        reads=reads,
    )
    assert result is tensor
    if user is None:
        assert kernel is tensor
        return False
    assert user is tensor and kernel.data_ptr() != tensor.data_ptr()
    return True


@pytest.mark.parametrize("mode", MODES)
def test_the_overlap_decision_without_a_device(mode: str) -> None:
    """An output whose bytes meet an input's is written through a temporary;
    an output beside the inputs in their buffer, in another buffer, an empty
    one and one checked against no input (the backward's) are written
    directly."""
    pkg = _package(mode)
    rows, hidden = 8, 16
    size = rows * hidden
    buf = torch.zeros(3 * size, dtype=torch.bfloat16)
    x = buf[:size].view(rows, hidden)
    w = buf[2 * size : 2 * size + hidden]
    reads = pkg._read_ranges(x, None, w)
    assert len(reads) == 2
    one_row_on = buf[hidden : hidden + size].view(rows, hidden)
    over_w = buf[2 * size - hidden : 3 * size - hidden].view(rows, hidden)
    rstd_in_x = buf[: 2 * rows].view(torch.float32)
    beside = buf[size : 2 * size].view(rows, hidden)
    assert _through_temporary(mode, x, reads)
    assert _through_temporary(mode, one_row_on, reads)
    assert _through_temporary(mode, over_w, reads)
    assert _through_temporary(mode, rstd_in_x, reads)
    assert not _through_temporary(mode, beside, reads)
    fresh = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    assert not _through_temporary(mode, fresh, reads)
    assert not _through_temporary(mode, buf[:0].view(0, hidden), reads)
    assert not _through_temporary(mode, x, ())


def _device_or_skip(mode: str) -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    device = torch.device("cuda:0")
    supported = getattr(_package(mode), f"is_cake_rmsnorm_{mode}_train_supported")
    if not supported(device, HIDDEN):
        pytest.skip(f"H = {HIDDEN} has no exported route on this device")
    return device


def _rows_of_windows(loader, arch: str, w_dtype: str) -> list[int]:
    """One row count in every program window of the plain forward route: the
    window's top row, or a few rows past the open window's lower edge."""
    windows = loader.program_windows(arch, HIDDEN, "fwd", False, w_dtype)
    return [
        low + OPEN_WINDOW_EXTRA if high is None else high for low, high, _ in windows
    ]


def _inputs(rows: int, w_dtype: torch.dtype, eps: float, device: torch.device):
    """Random ``x``, ``u`` and ``w`` with every ``EVERY``-th row constructed:
    ``x = +-1`` and ``u = 0``, or both zero at ``eps = 0``."""
    generator = torch.Generator(device=device).manual_seed(SEED + rows)
    x = torch.randn(rows, HIDDEN, device=device, generator=generator).bfloat16()
    u = torch.randn(rows, HIDDEN, device=device, generator=generator).bfloat16()
    w = 1.0 + 0.1 * torch.randn(HIDDEN, device=device, generator=generator)
    picked = torch.arange(0, rows, EVERY, device=device)
    u[picked] = 0
    if eps == 0.0:
        x[picked] = 0
    else:
        signs = torch.randint(
            0, 2, (picked.numel(), HIDDEN), device=device, generator=generator
        )
        x[picked] = (2 * signs - 1).bfloat16()
    return x, u, w.bfloat16().to(w_dtype)


def _check(mode: str, w_name: str, rows: int, eps: float, residual_route: bool):
    """The failures of the overlapping calls on one problem."""
    device = torch.device("cuda:0")
    forward = getattr(_package(mode), f"cake_rmsnorm_{mode}_train_forward")
    x0, u0, w0 = _inputs(rows, W_DTYPES[w_name], eps, device)
    y_ref, r_ref = forward(x0, w0, eps)
    failures = []

    x = x0.clone()
    y, r = forward(x, w0, eps, out=x)
    if not (y is x and _same(x, y_ref) and _same(r, r_ref)):
        failures.append("out=x")

    buf = torch.empty(rows + 1, HIDDEN, dtype=torch.bfloat16, device=device)
    buf[:rows] = x0
    y, r = forward(buf[:rows], w0, eps, out=buf[1:])
    if not (_same(buf[1:], y_ref) and _same(r, r_ref)):
        failures.append("out one row past x")

    # ``w`` held in the first bytes of ``out``.
    held = torch.empty(rows + 2, HIDDEN, dtype=torch.bfloat16, device=device)
    w_in = held.view(-1)[: HIDDEN * w0.element_size() // 2].view(w0.dtype)
    w_in.copy_(w0)
    out = held[:rows]
    y, r = forward(x0, w_in, eps, out=out)
    if not (_same(out, y_ref) and _same(r, r_ref)):
        failures.append("out holding w")

    if residual_route:
        y_ref, r_ref, h_ref = forward(x0, w0, eps, residual=u0)
        h = x0.clone()
        y, r, h_new = forward(h, w0, eps, residual=u0, residual_out=h)
        if not (h_new is h and _same(h, h_ref) and _same(y, y_ref)):
            failures.append("residual_out=x")
        u = u0.clone()
        y, r, h_new = forward(x0, w0, eps, residual=u, out=u)
        if not (y is u and _same(u, y_ref) and _same(h_new, h_ref)):
            failures.append("out=residual")
    return failures


@pytest.mark.parametrize("w_name", list(W_DTYPES))
@pytest.mark.parametrize("mode", MODES)
def test_outputs_sharing_memory_with_an_input_match_fresh_outputs(
    mode: str, w_name: str
) -> None:
    """Every overlapping call of the module docstring returns, and leaves in the
    caller's buffers, the bits of the call with fresh outputs, in every program
    window of the H = 2048 forward."""
    device = _device_or_skip(mode)
    loader = _loader(mode)
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    if w_name not in loader.weight_dtypes(arch, HIDDEN, "fwd", False):
        pytest.skip(f"no {w_name}-weight forward at H = {HIDDEN} on {arch}")
    residual_route = w_name in loader.weight_dtypes(arch, HIDDEN, "fwd", True)
    near = torch.tensor(NEAR_MIDPOINT, dtype=torch.int32).view(torch.float32)
    eps_values = [h - 1.0 for h in near.tolist()] + [0.0]
    failures = []
    for rows in _rows_of_windows(loader, arch, w_name):
        for eps in eps_values:
            for name in _check(mode, w_name, rows, eps, residual_route):
                failures.append(
                    f"{arch} {mode} w={w_name} rows={rows} eps={eps}: {name}"
                )
    assert not failures, "\n".join(failures)
