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

Training with ``cake_pplx.cake_mhc_nondet_train`` (family ``mhc``) composed with
the model's deterministic head ``cake_pplx.cake_mhc_head_det_train.cake_mhc_head_det``
(family ``mhc_head``, the stream mean of the last layer): the gradients reach
every leaf through the head, and the head output and every gradient are judged
with ``tests/cake_naive.py`` against the naive BF16 tier over 32 inputs.  The
oracle is the whole chain (two connections, the stream mean, the loss) in
float64 by autograd; the tier is the same chain with PyTorch in BF16 on the
same BF16 inputs.  The non-deterministic package makes no bit promise, so
nothing is compared bit for bit with the deterministic training.

Moved from ``kernels/mhc/tests/experimental/test_cake_mhc_nondet_train_api.py``,
which keeps the single-family checks.  Needs ``cake-pplx-mhc`` and
``cake-pplx-mhc-head`` installed in one environment.
"""

from __future__ import annotations

import pytest
import torch

import cake_pplx.cake_mhc_nondet_train as facade
import cake_pplx._backends.cake_mhc_nondet_train as backend
from cake_pplx._backends.cake_mhc_nondet_train import cake_jit as loader

from ..cake_naive import DRAWS, TIERS, TierJudge, contract, draw_seed, seed_all

head_facade = pytest.importorskip(
    "cake_pplx.cake_mhc_head_det_train",
    reason="the mhc_head family (cake-pplx-mhc-head) is not installed",
)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN_SIZES = (4096, 5120, 7168)
SEED = 2032
STREAMS = 4
RMS_EPS, HC_EPS, SINKHORN_ITERS = 1e-5, 1e-6, 20
BF16, F32, F64 = torch.bfloat16, torch.float32, torch.float64


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and facade.is_cake_mhc_nondet_train_supported(_device())
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="cake_mhc_nondet_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _hidden_sizes() -> tuple[int, ...]:
    arch = _arch()
    return tuple(loader.supported_hidden_sizes(arch)) if arch else ()


@pytest.fixture(params=HIDDEN_SIZES, ids=lambda h: f"H{h}")
def hidden(request) -> int:
    """The hidden size of a device case (skipped where the architecture has no route of it)."""
    if request.param not in _hidden_sizes():
        pytest.skip(
            f"hidden {request.param} has no exported route on {_arch()} "
            f"(exported hidden sizes: {_hidden_sizes()})"
        )
    return request.param


def _inputs(T: int, seed: int = SEED, hidden: int = 4096) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=_device(), dtype=torch.float32
        ).to(dtype)

    return {
        "X": randn(T, 4, hidden, dtype=BF16),
        "fn": (randn(24, 4 * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(24),
        "B": randn(T, hidden, dtype=BF16),
        "dU": randn(T, hidden, dtype=BF16),
    }


def _nd_prepare(*args, **kwargs):
    return facade.cake_mhc_nondet_train_prepare(*args, **kwargs)


def _nd_finish(*args, **kwargs):
    return facade.cake_mhc_nondet_train_finish(*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------
# The model's head; the oracle and the naive tier of the chain


def _head_connections(prepare, finish, i: dict, layers: list, **kwargs) -> tuple:
    """Two connections of ``prepare`` / ``finish`` (``layers``: their ``(fn, scale, base)``; the branch ``B = U +
    B_0``), then the model's head ``cake_mhc_head_det`` (the stream mean of the last connection's ``Y``) under
    the loss ``sum(head * dU)``: ``(head output, [X.grad, *(fn, scale, base).grad per connection])``."""
    X = i["X"].clone().requires_grad_(True)
    params = [[p.clone().requires_grad_(True) for p in layer] for layer in layers]
    x = X
    for fn, scale, base in params:
        U, handle = prepare(x, fn, scale, base, **kwargs)
        x = finish(x, (U.float() + i["B"].float()).to(BF16), handle)
    head = head_facade.cake_mhc_head_det(x)
    (head.float() * i["dU"].float()).sum().backward()
    return head.detach(), [X.grad, *[p.grad for layer in params for p in layer]]


def _sinkhorn(logits: torch.Tensor, hc_eps: float, iters: int) -> torch.Tensor:
    """``softmax`` over j, ``+ hc_eps``, one column normalisation, then ``iters - 1`` x (row, column)."""
    C = torch.softmax(logits, dim=-1) + hc_eps
    C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        C = C / (C.sum(dim=2, keepdim=True) + hc_eps)
        C = C / (C.sum(dim=1, keepdim=True) + hc_eps)
    return C


def _connection(x, fn, scale, base, B0):
    """One residual connection in the dtype of its operands: the RMS-normalised mapping, the pre / post sigmoids,
    the Sinkhorn matrix, ``U = sum_i pre_i x_i``, the branch ``B = U + B_0`` and ``Y = post B + C^T x``."""
    T, S, H = x.shape
    u = x.reshape(T, S * H)
    r = torch.rsqrt(u.square().mean(dim=-1, keepdim=True) + RMS_EPS)
    z = (u * r) @ fn.T
    pre = torch.sigmoid(z[:, 0:4] * scale[0] + base[0:4]) + HC_EPS
    post = 2 * torch.sigmoid(z[:, 4:8] * scale[1] + base[4:8])
    C = _sinkhorn(
        (z[:, 8:] * scale[2] + base[8:]).reshape(T, STREAMS, STREAMS),
        HC_EPS,
        SINKHORN_ITERS,
    )
    U = (pre.unsqueeze(-1) * x).sum(dim=1)
    B = U + B0
    return post.unsqueeze(-1) * B.unsqueeze(1) + torch.einsum("tij,tid->tjd", C, x)


def _chain(i: dict, layers: list, dtype: torch.dtype) -> tuple:
    """The chain of :func:`_head_connections` with PyTorch in ``dtype`` (float64: the oracle; BF16 on the same BF16
    inputs: the naive tier), its backward by autograd: ``(head, [dX, *(dfn, dscale, dbase) per connection])``."""
    with torch.enable_grad():
        X = i["X"].detach().to(dtype).requires_grad_(True)
        params = [
            [p.detach().to(dtype).requires_grad_(True) for p in layer]
            for layer in layers
        ]
        B0 = i["B"].to(dtype)
        x = X
        for fn, scale, base in params:
            x = _connection(x, fn, scale, base, B0)
        head = x.mean(dim=1)
        grads = torch.autograd.grad(
            [head],
            [X, *[p for layer in params for p in layer]],
            grad_outputs=[i["dU"].to(dtype)],
        )
    return head.detach(), list(grads)


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_nd_training_composes_with_the_stream_mean_head(
    T: int, save_stats: bool, hidden: int, record_property
) -> None:
    """Training with this package feeding the model's deterministic head
    (``cake_pplx.cake_mhc_head_det_train.cake_mhc_head_det``, ``mean_i X[:, i, :]`` of the last connection's
    four streams), at every hidden size the device routes: two connections, the head and a loss on its output.  The
    connections' handles are the backend's and the gradients reach every leaf through the head (non-zero).  Over 32
    inputs, the head output and the gradients of ``X`` and of each connection's ``fn``, ``scale`` and ``base`` are
    judged against the naive BF16 tier (contract: shapes, finite values)."""
    if not head_facade.is_cake_mhc_head_det_train_supported(_device(), hidden):
        pytest.skip(f"the head has no program of hidden {hidden} on {_arch()}")
    names = [
        "head",
        "dX",
        *(f"{k}[{c}]" for c in range(2) for k in ("dfn", "dscale", "dbase")),
    ]
    judge = TierJudge(DRAWS)
    failures: list[str] = []
    what = f"T={T} H={hidden} save_stats={save_stats}"
    cuda, state = torch.cuda.get_rng_state_all(), torch.get_rng_state()
    try:
        for j in range(DRAWS):
            s = draw_seed(SEED + 40, j)
            seed_all(s)
            i = _inputs(T, seed=s, hidden=hidden)
            second = _inputs(1, seed=s + 1, hidden=hidden)
            layers = [[k[n] for n in ("fn", "scale", "base")] for k in (i, second)]
            handles = []

            def nd_prepare(*args, **kwargs):
                U, handle = _nd_prepare(*args, **kwargs)
                handles.append(handle)
                return U, handle

            head, grads = _head_connections(
                nd_prepare, _nd_finish, i, layers, save_stats=save_stats
            )
            assert len(handles) == 2
            assert all(isinstance(h, backend._MHCTrainHandle) for h in handles), what
            assert head.shape == (T, hidden), f"head {what}"
            got = [head, *grads]
            for name, g in zip(names[1:], grads, strict=True):
                assert g is not None, f"grad {name} {what}"
                assert bool((g != 0).any()), (
                    f"grad {name} {what}: no gradient reached the leaf"
                )
            want_head, want = _chain(i, layers, F64)
            tier_head, tier = _chain(i, layers, BF16)
            for name, actual, expected, naive in zip(
                names, got, [want_head, *want], [tier_head, *tier], strict=True
            ):
                if message := contract(name, actual, expected):
                    failures.append(f"draw {j}: {message}")
                    if actual.shape != expected.shape:
                        continue
                judge.add(name, j, actual, expected, {t: naive.to(BF16) for t in TIERS})
    finally:
        torch.set_rng_state(state)
        torch.cuda.set_rng_state_all(cuda)
    failures += judge.failures()
    judge.record(record_property)
    assert not failures, "\n".join(failures)
