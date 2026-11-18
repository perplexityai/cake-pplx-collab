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

CPU tests of the token-count limit of the causal convolution training entries: the kernels split a channel tile's
``S`` tokens over its token streams with the 32-bit ``(S + streams - 1) / streams``, so the token check of both
packages accepts ``S`` up to ``MAX_TOKENS = 2**31 - 2**16`` and raises ``ValueError`` beyond, in the single-rank
forward and backward and in the KCP forward and backward (both host paths).  The token tensors live on the ``meta``
device (no storage) and the architecture lookup returns a fixed answer, so the check runs without a GPU.  No kernel
runs here.
"""

from __future__ import annotations

import types

import pytest
import torch

from cake_pplx._backends import cake_causal_conv1d_det_train as det
from cake_pplx._backends import cake_causal_conv1d_nondet_train as nondet
from cake_pplx._backends.cake_causal_conv1d_det_train import kcp as det_kcp
from cake_pplx._backends.cake_causal_conv1d_nondet_train import kcp as nondet_kcp

LIMIT = 2**31 - 2**16
C = 8
META = torch.device("meta")
PACKAGES = {"det": (det, det_kcp), "nondet": (nondet, nondet_kcp)}


def _tokens(S: int, *, batched: bool = False) -> torch.Tensor:
    """A BF16 ``[S, C]`` (or ``[1, S, C]``) token tensor without storage."""
    return torch.empty((1, S, C) if batched else (S, C), dtype=torch.bfloat16, device=META)


def _refused(name: str, S: int) -> str:
    return f"{name}: S={S} must be at most {LIMIT}"


def _fixed_arch(monkeypatch, *modules) -> None:
    """Every module's architecture lookup answers sm_100a, so the operand checks run on meta tensors."""
    for module in modules:
        monkeypatch.setattr(module, "_device_arch", lambda device: "sm_100a")


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_token_limit(package: str) -> None:
    backend, kcp = PACKAGES[package]
    assert backend.MAX_TOKEN_STREAMS == 2**16
    assert backend.MAX_TOKENS == LIMIT == backend.INT32_MAX - backend.MAX_TOKEN_STREAMS + 1
    # the KCP entries run the package's token check
    assert kcp._tokens is backend._tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
@pytest.mark.parametrize("batched", [False, True], ids=["S_C", "1_S_C"])
def test_token_check_takes_the_limit_and_refuses_one_more(package: str, batched: bool) -> None:
    backend, _ = PACKAGES[package]
    S, channels, flat = backend._tokens(_tokens(LIMIT, batched=batched), "x", device=META)
    assert (S, channels, tuple(flat.shape)) == (LIMIT, C, (LIMIT, C))
    with pytest.raises(ValueError, match=_refused("x", LIMIT + 1)):
        backend._tokens(_tokens(LIMIT + 1, batched=batched), "x", device=META)


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_single_rank_entries_refuse_too_many_tokens(monkeypatch, package: str) -> None:
    backend, _ = PACKAGES[package]
    _fixed_arch(monkeypatch, backend)
    x = _tokens(LIMIT + 1)
    weight = torch.empty((C, 4), dtype=torch.float32, device=META)
    cu_seqlens = torch.empty(2, dtype=torch.int32, device=META)
    forward = getattr(backend, f"cake_causal_conv1d_{package}_train_forward")
    backward = getattr(backend, f"cake_causal_conv1d_{package}_train_backward")
    with pytest.raises(ValueError, match=_refused("x", LIMIT + 1)):
        forward(x, weight, cu_seqlens)
    with pytest.raises(ValueError, match=_refused("x", LIMIT + 1)):
        backward(x, weight, cu_seqlens, x)


@pytest.mark.cpu_only
@pytest.mark.parametrize("package", sorted(PACKAGES))
@pytest.mark.parametrize("host_trim", [1, 0])
def test_kcp_entries_refuse_too_many_tokens(monkeypatch, package: str, host_trim: int) -> None:
    backend, kcp = PACKAGES[package]
    _fixed_arch(monkeypatch, backend, kcp)
    x = _tokens(LIMIT + 1)
    weight = torch.empty((C, 4), dtype=torch.float32, device=META)
    hist = torch.empty((1, C), dtype=torch.bfloat16, device=META)
    ctx = types.SimpleNamespace(cu_seqlens_local_cpu=(0, LIMIT + 1))
    forward = getattr(kcp, f"cake_causal_conv1d_{package}_train_kcp_forward")
    backward = getattr(kcp, f"cake_causal_conv1d_{package}_train_kcp_backward")
    try:
        kcp._set_test_knobs({"host_trim": host_trim})
        with pytest.raises(ValueError, match=_refused("x_local", LIMIT + 1)):
            forward(x, weight, ctx)
        with pytest.raises(ValueError, match=_refused("x_local", LIMIT + 1)):
            backward(x, weight, ctx, x, hist)
    finally:
        kcp._set_test_knobs(None)
