"""
Copyright (c) 2026 by FlashInfer team.

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
every leaf through the head and equal the deterministic training and head's.

Moved from ``kernels/mhc/tests/experimental/test_cake_mhc_nondet_train_api.py``
(FlashInfer ``c5c39b0e``), which keeps the single-family checks; the helpers
below are that module's.  Needs ``cake-pplx-mhc`` and ``cake-pplx-mhc-head``
installed in one environment.
"""

from __future__ import annotations

import pytest
import torch

import cake_pplx.cake_mhc_det_train as det_facade
import cake_pplx.cake_mhc_nondet_train as facade
import cake_pplx._backends.cake_mhc_nondet_train as backend
from cake_pplx._backends.cake_mhc_det_train import cake_jit as det_loader
from cake_pplx._backends.cake_mhc_nondet_train import cake_jit as loader

head_facade = pytest.importorskip(
    "cake_pplx.cake_mhc_head_det_train",
    reason="the mhc_head family (cake-pplx-mhc-head) is not installed",
)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN_SIZES = (4096, 5120, 7168)
SEED = 2032
BF16 = torch.bfloat16
# The ``dfn`` of this package's backward against the deterministic one (test_cake_mhc_nondet_train_api.py's bound).
DFN_REL_L2_MAX = 1e-6


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and det_facade.is_cake_mhc_det_train_supported(_device())
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
        for spec in [*det_loader.specs_for_arch(arch), *loader.specs_for_arch(arch)]:
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

    i = {
        "X": randn(T, 4, hidden, dtype=BF16),
        "fn": (randn(24, 4 * hidden) * 1e-2).contiguous(),
        "scale": 1.0 + 0.1 * randn(3),
        "base": 0.1 * randn(24),
        "B": randn(T, hidden, dtype=BF16),
        "dY": randn(T, 4, hidden, dtype=BF16),
        "dU": randn(T, hidden, dtype=BF16),
    }
    _U, i["stats"] = det_facade.cake_mhc_det_train_forward_prepare(
        i["X"], i["fn"], i["scale"], i["base"]
    )
    _dB, i["gstats"] = det_facade.cake_mhc_det_train_backward_prepare(
        i["dY"], i["B"], i["X"], i["stats"]
    )
    return i


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm())


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.detach().contiguous().reshape(-1).view(torch.uint8),
        b.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _nd_prepare(*args, **kwargs):
    return facade.cake_mhc_nondet_train_prepare(*args, **kwargs)


def _nd_finish(*args, **kwargs):
    return facade.cake_mhc_nondet_train_finish(*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------
# The model's head


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


@requires_device
@pytest.mark.parametrize("T", (7, 2049))
@pytest.mark.parametrize("save_stats", (True, False))
def test_nd_training_composes_with_the_stream_mean_head(
    T: int, save_stats: bool, hidden: int
) -> None:
    """Training with this package feeding the model's deterministic head
    (``cake_pplx.cake_mhc_head_det_train.cake_mhc_head_det``, ``mean_i X[:, i, :]`` of the last connection's
    four streams), at every hidden size the device routes: two connections, the head and a loss on its output.  The
    gradients reach every leaf through the head (finite, non-zero), the connections' handles are the backend's, and
    the head output and the ``X`` / ``scale`` / ``base`` gradients are bitwise those of the deterministic training
    and the same head; each ``fn`` gradient is within FP32 rounding of the deterministic one (rel-L2 <=
    ``DFN_REL_L2_MAX``)."""
    if not head_facade.is_cake_mhc_head_det_train_supported(_device(), hidden):
        pytest.skip(f"the head has no program of hidden {hidden} on {_arch()}")
    i = _inputs(T, seed=SEED + 40, hidden=hidden)
    second = _inputs(1, seed=SEED + 41, hidden=hidden)
    layers = [[j[k] for k in ("fn", "scale", "base")] for j in (i, second)]
    handles = []

    def nd_prepare(*args, **kwargs):
        U, handle = _nd_prepare(*args, **kwargs)
        handles.append(handle)
        return U, handle

    head, grads = _head_connections(
        nd_prepare, _nd_finish, i, layers, save_stats=save_stats
    )
    want_head, want = _head_connections(
        det_facade.cake_mhc_det_train_prepare,
        det_facade.cake_mhc_det_train_finish,
        i,
        layers,
        save_stats=save_stats,
    )
    what = f"T={T} H={hidden} save_stats={save_stats}"
    assert len(handles) == 2
    assert all(isinstance(h, backend._MHCTrainHandle) for h in handles), what
    assert head.shape == (T, hidden) and _same_bits(head, want_head), f"head {what}"
    names = ["X", *(f"{k}[{c}]" for c in range(2) for k in ("fn", "scale", "base"))]
    for name, g, w in zip(names, grads, want, strict=True):
        assert g is not None and bool(torch.isfinite(g).all()), f"grad {name} {what}"
        assert bool((g != 0).any()), f"grad {name} {what}: no gradient reached the leaf"
        if name.startswith("fn"):
            assert _rel_l2(g, w) <= DFN_REL_L2_MAX, f"grad {name} {what}"
        else:
            assert _same_bits(g, w), f"grad {name} {what}"
