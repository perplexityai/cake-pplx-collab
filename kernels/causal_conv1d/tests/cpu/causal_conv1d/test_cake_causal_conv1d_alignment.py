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

CPU tests of the address-alignment checks of both causal convolution training packages.  The programs load and
store their operands as vectors, so each operand's address must be a multiple of the vector size: the token rows
(``x``, ``dy``, ``out``, ``dx``) 16 bytes, one channel's taps 16 bytes (FP32) or 8 (BF16), and the bias 32 bytes
(FP32) or 16 (BF16) where a forward runs, 16 bytes in the standalone backward, and the KCP backward's ``hist``
pool 16 bytes.  A misaligned operand raises a ``ValueError`` that names it, the alignment and the address residue;
an aligned view and an empty tensor pass.  The autograd entries copy a misaligned incoming gradient instead (autograd
hands it over, so the caller cannot replace it).  The checks read host metadata only: the tensors live on the CPU,
the architecture lookup returns a fixed answer, and each refusal comes before any device query, so no GPU is needed
and no kernel runs.
"""

from __future__ import annotations

import re
import types

import pytest
import torch

from cake_pplx._backends import cake_causal_conv1d_det_train as det
from cake_pplx._backends import cake_causal_conv1d_nondet_train as nondet
from cake_pplx._backends.cake_causal_conv1d_det_train import kcp as det_kcp
from cake_pplx._backends.cake_causal_conv1d_nondet_train import kcp as nondet_kcp

CPU = torch.device("cpu")
S, C, WIDTH = 16, 64, 4
F32, BF16 = torch.float32, torch.bfloat16
PACKAGES = {"det": (det, det_kcp), "nondet": (nondet, nondet_kcp)}


def _at(offset_bytes: int, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
    """A zeroed ``shape`` view whose address is ``offset_bytes`` past a 64-byte aligned one."""

    item = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for n in shape:
        numel *= n
    buf = torch.zeros(numel + 128 // item, dtype=dtype)
    k = (-buf.data_ptr() % 64) // item + offset_bytes // item
    view = buf[k : k + numel].view(shape)
    assert numel == 0 or view.data_ptr() % 64 == offset_bytes
    return view


def _refusal(name: str, need: int, residue: int) -> str:
    return re.escape(
        f"{name} must be {need}-byte aligned (the kernels access it as {need}-byte vectors), got "
        f"data_ptr() % {need} == {residue}; pass a fresh tensor or .clone()"
    )


def _fixed_arch(monkeypatch, *modules) -> None:
    """Every module's architecture lookup answers sm_100a, so the operand checks run on CPU tensors."""
    for module in modules:
        monkeypatch.setattr(module, "_device_arch", lambda device: "sm_100a")


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_alignment_rules(package: str) -> None:
    backend, kcp = PACKAGES[package]
    assert backend.TOKEN_ALIGN == 16
    assert backend.TAPS_ALIGN == {F32: 16, BF16: 8}
    assert backend.BIAS_ALIGN == {F32: 32, BF16: 16}
    assert backend.BACKWARD_BIAS_ALIGN == {F32: 16, BF16: 16}
    assert backend.DWEIGHT_FP32_ALIGN == 16
    # the KCP entries run the package's checks and its gradient copy
    for name in ("_aligned", "_tokens", "_weight", "_output", "_grad_tokens"):
        assert getattr(kcp, name) is getattr(backend, name), name
    assert kcp.TOKEN_ALIGN == backend.TOKEN_ALIGN


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_token_tensors_need_16_bytes(package: str) -> None:
    backend, _ = PACKAGES[package]
    for offset in range(0, 64, 2):
        for shape in ((S, C), (1, S, C)):
            t = _at(offset, BF16, shape)
            if offset % 16:
                with pytest.raises(ValueError, match=_refusal("dy", 16, offset % 16)):
                    backend._tokens(t, "dy", device=CPU)
            else:
                assert backend._tokens(t, "dy", device=CPU)[:2] == (S, C)
    # no token, no access: an empty view passes at any address
    assert backend._tokens(_at(2, BF16, (0, C)), "x", device=CPU)[:2] == (0, C)


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_taps_need_16_bytes_fp32_and_8_bytes_bf16(package: str) -> None:
    backend, _ = PACKAGES[package]
    for dtype, need in ((F32, 16), (BF16, 8)):
        item = torch.empty((), dtype=dtype).element_size()
        for offset in range(0, 64, item):
            w = _at(offset, dtype, (C, WIDTH))
            if offset % need:
                match = _refusal("weight", need, offset % need)
                with pytest.raises(ValueError, match=match):
                    backend._weight(w, C, device=CPU, dtypes=(F32, BF16))
            else:
                assert backend._weight(w, C, device=CPU, dtypes=(F32, BF16)) is w


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_bias_needs_32_bytes_fp32_forward_and_16_bytes_otherwise(package: str) -> None:
    backend, _ = PACKAGES[package]
    for dtype, forward_need in ((F32, 32), (BF16, 16)):
        item = torch.empty((), dtype=dtype).element_size()
        for offset in range(0, 64, item):
            b = _at(offset, dtype, (C,))
            for backward, need in ((False, forward_need), (True, 16)):
                if offset % need:
                    match = _refusal("bias", need, offset % need)
                    with pytest.raises(ValueError, match=match):
                        backend._bias(b, C, device=CPU, backward=backward)
                else:
                    assert backend._bias(b, C, device=CPU, backward=backward) is b


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_outputs_take_the_alignment_they_are_given(package: str) -> None:
    backend, _ = PACKAGES[package]
    spec = dict(shape=(S, C), dtype=BF16, device=CPU)
    for offset in range(0, 32, 2):
        t = _at(offset, BF16, (S, C))
        if offset % 16:
            with pytest.raises(ValueError, match=_refusal("dx", 16, offset % 16)):
                backend._output(t, "dx", **spec, align=16)
        else:
            assert backend._output(t, "dx", **spec, align=16) is t
        # without an alignment (an output stored per element) any offset passes
        assert backend._output(t, "dbias", **spec) is t
    fresh = backend._output(None, "dx", **spec, align=16)
    assert tuple(fresh.shape) == (S, C) and fresh.dtype == BF16


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_kcp_hist_pool_needs_16_bytes(package: str) -> None:
    _, kcp = PACKAGES[package]
    for offset in range(0, 64, 2):
        hist = _at(offset, BF16, (2, C))
        if offset % 16:
            with pytest.raises(ValueError, match=_refusal("hist", 16, offset % 16)):
                kcp._hist_pool(hist, C, device=CPU)
        else:
            assert kcp._hist_pool(hist, C, device=CPU) is hist
    # an empty pool is never read: any address passes, and the programs get a fresh
    # one-row pool
    empty = _at(2, BF16, (0, C))
    pool = kcp._hist_pool(empty, C, device=CPU)
    assert pool is not empty and tuple(pool.shape) == (1, C) and pool.dtype == BF16
    # the shape, dtype and layout rules come before the address rule
    msg = "hist must be the forward's contiguous BF16"
    bad = (
        _at(2, BF16, (2, C + 8)),
        _at(0, F32, (2, C)),
        _at(0, BF16, (4, C))[::2],
        _at(0, BF16, (2 * C,)),
    )
    for hist in bad:
        with pytest.raises(ValueError, match=msg):
            kcp._hist_pool(hist, C, device=CPU)


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_autograd_gradient_is_copied_to_an_aligned_address(package: str) -> None:
    backend, _ = PACKAGES[package]
    gen = torch.Generator().manual_seed(11)
    for offset in range(0, 32, 2):
        grad = _at(offset, BF16, (S, C))
        grad.copy_(torch.randn(S, C, generator=gen))
        g = backend._grad_tokens(grad)
        assert g.data_ptr() % 16 == 0 and g.is_contiguous() and g.dtype == BF16
        assert torch.equal(g, grad)
        # an aligned gradient is used in place; a misaligned one is copied
        assert (g is grad) == (offset % 16 == 0)
    # an FP32 or strided gradient becomes a contiguous BF16 tensor
    f32 = torch.randn(S, C, generator=gen)
    assert torch.equal(backend._grad_tokens(f32), f32.to(BF16))
    strided = _at(0, BF16, (C, S)).t()
    strided.copy_(torch.randn(S, C, generator=gen))
    g = backend._grad_tokens(strided)
    assert g.is_contiguous() and g.data_ptr() % 16 == 0 and torch.equal(g, strided)


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_single_rank_entries_refuse_misaligned_operands(
    monkeypatch, package: str
) -> None:
    backend, _ = PACKAGES[package]
    _fixed_arch(monkeypatch, backend)
    forward = getattr(backend, f"cake_causal_conv1d_{package}_train_forward")
    backward = getattr(backend, f"cake_causal_conv1d_{package}_train_backward")
    autograd = getattr(backend, f"cake_causal_conv1d_{package}")
    x, dy = _at(0, BF16, (S, C)), _at(0, BF16, (S, C))
    w = _at(0, F32, (C, WIDTH))
    cu = torch.tensor([0, S], dtype=torch.int32)
    tokens, taps = _at(2, BF16, (S, C)), _at(4, F32, (C, WIDTH))
    bias32, bias16 = _at(16, F32, (C,)), _at(4, F32, (C,))
    cases = [
        ("x", 16, 2, lambda: forward(tokens, w, cu)),
        ("x", 16, 2, lambda: backward(tokens, w, cu, dy)),
        ("x", 16, 2, lambda: autograd(tokens, w, cu)),
        ("dy", 16, 2, lambda: backward(x, w, cu, tokens)),
        ("out", 16, 2, lambda: forward(x, w, cu, out=tokens)),
        ("dx", 16, 2, lambda: backward(x, w, cu, dy, dx=tokens)),
        ("weight", 16, 4, lambda: forward(x, taps, cu)),
        ("weight", 16, 4, lambda: backward(x, taps, cu, dy)),
        ("weight", 16, 4, lambda: autograd(x, taps, cu)),
        # an FP32 bias 16 bytes past a 32-byte boundary: refused where a forward runs;
        # the standalone backward refuses one 4 bytes past a 16-byte boundary
        ("bias", 32, 16, lambda: forward(x, w, cu, bias=bias32)),
        ("bias", 32, 16, lambda: autograd(x, w, cu, bias32)),
        ("bias", 16, 4, lambda: backward(x, w, cu, dy, bias=bias16)),
    ]
    for name, need, residue, call in cases:
        with pytest.raises(ValueError, match=_refusal(name, need, residue)):
            call()


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
@pytest.mark.parametrize("host_trim", [1, 0])
def test_kcp_entries_refuse_misaligned_operands(
    monkeypatch, package: str, host_trim: int
) -> None:
    backend, kcp = PACKAGES[package]
    _fixed_arch(monkeypatch, backend, kcp)
    forward = getattr(kcp, f"cake_causal_conv1d_{package}_train_kcp_forward")
    backward = getattr(kcp, f"cake_causal_conv1d_{package}_train_kcp_backward")
    x, dy = _at(0, BF16, (S, C)), _at(0, BF16, (S, C))
    w = _at(0, F32, (C, WIDTH))
    tokens, taps = _at(2, BF16, (S, C)), _at(4, F32, (C, WIDTH))
    hist = torch.empty((1, C), dtype=BF16)
    ctx = types.SimpleNamespace(cu_seqlens_local_cpu=(0, S))
    cases = [
        ("x_local", 2, lambda: forward(tokens, w, ctx)),
        ("weight", 4, lambda: forward(x, taps, ctx)),
        ("x_local", 2, lambda: backward(tokens, w, ctx, dy, hist)),
        ("weight", 4, lambda: backward(x, taps, ctx, dy, hist)),
        ("dy", 2, lambda: backward(x, w, ctx, tokens, hist)),
    ]
    if host_trim:
        # the trimmed forward checks its output, and the trimmed backward its hist pool,
        # before they read the context
        cases.append(("out", 2, lambda: forward(x, w, ctx, out=tokens)))
        cases.append(("hist", 2, lambda: backward(x, w, ctx, dy, _at(2, BF16, (1, C)))))
    try:
        kcp._set_test_knobs({"host_trim": host_trim})
        for name, residue, call in cases:
            with pytest.raises(ValueError, match=_refusal(name, 16, residue)):
                call()
    finally:
        kcp._set_test_knobs(None)
