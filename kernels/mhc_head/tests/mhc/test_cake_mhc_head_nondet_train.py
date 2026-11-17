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

Tests of the non-deterministic Cake mHC training head
(``cake_pplx.cake_mhc_head_nondet_train``): the stream mean ``Y = X.mean(dim=1)``
of BF16 ``X`` ``[T, 4, H]`` and its gradient, through the forward / backward
entry points, the accumulating backward, the autograd node, activation
checkpointing and CUDA graph capture, at every hidden size the device's
architecture serves (4096, 5120 and 7168 on SM100 / SM103 / SM107).

Every output is judged with ``tests/cake_naive.py`` against the naive BF16 tier
over 32 inputs: its error against the float64 oracle (the exact mean and
gradient, PyTorch's value where that is not finite;
``tests/cake/test_helpers/cake_mhc_head_naive.py``), in the L2 and the max norm,
median and max, is at most 1.005 times the tier's, and zero where the tier's is
zero (the tier's gradient ``dY / 4`` is exact, so the backward must be too).
The ``nondet`` contract makes no bit promise: repeated calls and the checkpoint
replay are judged like any other launch, and nothing is compared with the
deterministic head.  The head has no order-varying step, so under
``torch.use_deterministic_algorithms(True)`` (with and without ``warn_only``)
no entry point raises or warns.

The hidden sizes a case runs at come from the architecture table below, not
from the module's support query, and a separate case checks that query: a
support query that wrongly says "unsupported" fails instead of skipping every
case.
"""

from __future__ import annotations

import gc
import warnings

import pytest
import torch
import torch.utils.checkpoint

import cake_pplx
import cake_pplx.cake_mhc_head_det_train as det_facade
import cake_pplx.cake_mhc_head_nondet_train as facade
from tests.cake.test_helpers.cake_mhc_head_naive import (
    F64,
    backward_naive,
    backward_oracle,
    forward_naive,
    forward_oracle,
    judge_draws,
)


def get_compute_capability(device: torch.device) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(device))


# The hidden sizes each supported compute capability serves.
HIDDEN_BY_CAPABILITY = {
    (10, 0): (4096, 5120, 7168),
    (10, 3): (4096, 5120, 7168),
    (10, 7): (4096, 5120, 7168),
}
HIDDEN = (4096, 5120, 7168)
TOKENS = (1, 2, 3, 7, 65, 129, 257, 1000, 2049, 4096, 7169)
BF16 = torch.bfloat16
SPECIAL = (
    -0.0,
    0.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    torch.finfo(BF16).max,
    -torch.finfo(BF16).max,
    2.0**-126,
    2.0**24,
    -(2.0**24),
    1.0,
    3.0,
)
SEED = 8068
# Repeated calls: the launches of each entry point on identical inputs, and the token counts (with two long calls).
REPEATS = 5
REPEAT_TOKENS = (1, 257, 14928, 16263)


@pytest.fixture(autouse=True)
def collect_graphs():
    """Free leftover CUDA graphs before and after each case (see the deterministic head's test)."""
    gc.collect()
    yield
    gc.collect()


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return tuple(get_compute_capability(_device()))


def _hidden_sizes() -> tuple[int, ...]:
    """The hidden sizes the device's architecture serves (empty off SM100 / SM103 / SM107)."""
    return HIDDEN_BY_CAPABILITY.get(_capability(), ())


pytestmark = pytest.mark.skipif(
    not _hidden_sizes(),
    reason="the Cake mHC training head needs an SM100 / SM103 / SM107 GPU",
)


def _require(hidden: int) -> None:
    if hidden not in _hidden_sizes():
        pytest.skip(
            f"hidden {hidden} is not served on compute capability {_capability()}"
        )


def _draw(
    shape: tuple[int, ...], kind: str, generator: torch.Generator
) -> torch.Tensor:
    """BF16 values: ``randn``, ``spread`` (``randn`` x ``2^k``, ``k`` in [-8, 8]) or ``special`` (half from SPECIAL)."""
    x = torch.randn(shape, device=_device(), generator=generator)
    if kind == "spread":
        x = x * torch.exp2(
            torch.randint(-8, 9, shape, device=_device(), generator=generator).float()
        )
    elif kind == "special":
        values = torch.tensor(SPECIAL, device=_device())
        pick = values[
            torch.randint(0, len(SPECIAL), shape, device=_device(), generator=generator)
        ]
        x = torch.where(
            torch.rand(shape, device=_device(), generator=generator) < 0.5, pick, x
        )
    return x.to(BF16)


def _complement(tensor: torch.Tensor) -> torch.Tensor:
    """A new BF16 tensor whose every element has the inverted bits of ``tensor``'s: an output buffer in which an
    element the kernel does not write differs from ``tensor``."""
    return (~tensor.contiguous().view(torch.int16)).view(BF16).view(tensor.shape)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device=_device()).manual_seed(seed)


def _block_oracle(X: torch.Tensor, W: torch.Tensor, dOut: torch.Tensor) -> dict:
    """The graph ``head(X * 2) * W`` in float64 by autograd: its head output, its output and the gradients of ``X``
    and ``W`` under the incoming gradient ``dOut`` (finite inputs)."""
    with torch.enable_grad():
        x = X.detach().to(F64).requires_grad_(True)
        w = W.detach().to(F64).requires_grad_(True)
        head = (x * 2).sum(dim=1) * 0.25
        out = head * w
        dX, dW = torch.autograd.grad(out, (x, w), dOut.to(F64))
    return {"head": head.detach(), "output": out.detach(), "dX": dX, "dW": dW}


def _block_naive(X: torch.Tensor, W: torch.Tensor, dOut: torch.Tensor) -> dict:
    """The graph ``head(X * 2) * W`` with PyTorch's ``X.mean(dim=1)`` as the head, in BF16."""
    x = X.detach().requires_grad_(True)
    w = W.detach().requires_grad_(True)
    head = (x * 2).mean(dim=1)
    out = head * w
    out.backward(dOut)
    return {"head": head.detach(), "output": out.detach(), "dX": x.grad, "dW": w.grad}


def test_support_query_follows_the_architecture() -> None:
    """The support query names exactly the hidden sizes of the architecture table, as the deterministic head's does."""
    dev = _device()
    assert facade.is_cake_mhc_head_nondet_train_supported(dev)
    for hidden in (*HIDDEN, 2048, 4000, 6144):
        served = hidden in _hidden_sizes()
        assert facade.is_cake_mhc_head_nondet_train_supported(dev, hidden) == served
        assert det_facade.is_cake_mhc_head_det_train_supported(dev, hidden) == served
    assert tuple(facade.SUPPORTED_COMPUTE_CAPABILITIES) == tuple(
        sorted(HIDDEN_BY_CAPABILITY)
    )
    assert tuple(facade.ARCHES) == ("sm_100a", "sm_103a", "sm_107a")


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", TOKENS)
@pytest.mark.parametrize("kind", ["randn", "spread", "special"])
def test_forward_backward_match_fp64(
    hidden: int, T: int, kind: str, record_property
) -> None:
    """The forward, the backward and the accumulating backward (into the caller's ``dX``) judged over 32 draws."""
    _require(hidden)

    def case(j: int, s: int):
        g = _generator(s)
        X = _draw((T, 4, hidden), kind, g)
        dY = _draw((T, hidden), kind, g)
        dX0 = _draw((T, 4, hidden), kind, g)
        acc = dX0.clone()
        out = facade.cake_mhc_head_nondet_train_backward(dY, dX=acc, accumulate=True)
        assert out is acc
        launch = {
            "Y": facade.cake_mhc_head_nondet_train_forward(X),
            "dX": facade.cake_mhc_head_nondet_train_backward(dY),
            "dX accumulate": acc,
        }
        expected = {
            "Y": forward_oracle(X),
            "dX": backward_oracle(dY),
            "dX accumulate": backward_oracle(dY, dX0),
        }
        naive = {
            "Y": forward_naive(X),
            "dX": backward_naive(dY),
            "dX accumulate": backward_naive(dY, dX0),
        }
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED + T * 31 + hidden)


@pytest.mark.parametrize("hidden", HIDDEN)
def test_preallocated_outputs_and_zero_tokens(hidden: int, record_property) -> None:
    """Caller outputs holding NaN are returned and written (judged over 32 draws); zero tokens give empty outputs,
    and an empty accumulation buffer is returned."""
    _require(hidden)
    Y = torch.empty((301, hidden), dtype=BF16, device=_device())
    dX = torch.empty((301, 4, hidden), dtype=BF16, device=_device())

    def case(j: int, s: int):
        g = _generator(s)
        X = _draw((301, 4, hidden), "randn", g)
        dY = _draw((301, hidden), "randn", g)
        Y.fill_(float("nan"))
        dX.fill_(float("nan"))
        assert facade.cake_mhc_head_nondet_train_forward(X, out=Y) is Y
        assert facade.cake_mhc_head_nondet_train_backward(dY, dX=dX) is dX
        launch = {"Y": Y.clone(), "dX": dX.clone()}
        expected = {"Y": forward_oracle(X), "dX": backward_oracle(dY)}
        naive = {"Y": forward_naive(X), "dX": backward_naive(dY)}
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED)
    empty = torch.empty((0, 4, hidden), dtype=BF16, device=_device())
    assert facade.cake_mhc_head_nondet_train_forward(empty).shape == (0, hidden)
    assert facade.cake_mhc_head_nondet_train_backward(empty[:, 0]).shape == (
        0,
        4,
        hidden,
    )
    assert (
        facade.cake_mhc_head_nondet_train_backward(
            empty[:, 0], dX=empty, accumulate=True
        )
        is empty
    )


def test_argument_checks() -> None:
    X = torch.zeros((8, 4, 4096), dtype=BF16, device=_device())
    with pytest.raises(ValueError, match="hidden"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros((8, 4, 4000), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="float32|bfloat16"):
        facade.cake_mhc_head_nondet_train_forward(X.float())
    with pytest.raises(ValueError, match="contiguous"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros((4, 8, 4096), dtype=BF16, device=_device()).transpose(0, 1)
        )
    with pytest.raises(ValueError, match="aligned"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros(8 * 4 * 4096 + 8, dtype=BF16, device=_device())[8:].view(
                8, 4, 4096
            )
        )
    with pytest.raises(ValueError, match=r"\[T, 4, hidden\]"):
        facade.cake_mhc_head_nondet_train_forward(
            torch.zeros((8, 3, 4096), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="contiguous"):
        facade.cake_mhc_head_nondet_train_forward(
            X, out=torch.empty((8, 4096 * 2), dtype=BF16, device=_device())[:, :4096]
        )
    with pytest.raises(ValueError, match="shape"):
        facade.cake_mhc_head_nondet_train_forward(
            X, out=torch.empty((8, 5120), dtype=BF16, device=_device())
        )
    dY = torch.zeros((8, 4096), dtype=BF16, device=_device())
    with pytest.raises(ValueError, match="accumulate"):
        facade.cake_mhc_head_nondet_train_backward(dY, accumulate=True)
    with pytest.raises(ValueError, match="shape"):
        facade.cake_mhc_head_nondet_train_backward(
            dY, dX=torch.empty((8, 4, 5120), dtype=BF16, device=_device())
        )
    with pytest.raises(ValueError, match="aligned"):
        facade.cake_mhc_head_nondet_train_backward(
            torch.zeros(8 * 4096 + 8, dtype=BF16, device=_device())[8:].view(8, 4096)
        )
    with pytest.raises(ValueError, match="overlap"):
        facade.cake_mhc_head_nondet_train_backward(
            X.view(-1)[: 8 * 4096].view(8, 4096), dX=X
        )
    with pytest.raises(ValueError, match="overlap"):
        facade.cake_mhc_head_nondet_train_forward(
            X, out=X.view(-1)[: 8 * 4096].view(8, 4096)
        )
    # The mode is part of the name: no entry point takes a ``deterministic`` keyword.
    with pytest.raises(TypeError):
        facade.cake_mhc_head_nondet_train_forward(X, deterministic=False)
    with pytest.raises(TypeError):
        facade.cake_mhc_head_nondet_train_backward(dY, deterministic=False)
    with pytest.raises(TypeError):
        facade.cake_mhc_head_nondet(X, deterministic=False)


@pytest.mark.parametrize("hidden", HIDDEN)
def test_autograd_matches_fp64(hidden: int, record_property) -> None:
    """The autograd node over 32 draws: ``Y`` (with a ``grad_fn``), the gradient of a first backward and of a second
    one accumulated into ``X.grad`` (as PyTorch's autograd accumulates: the tier is PyTorch's ``X.mean(dim=1)``
    node), and the gradient of an expanded (non-contiguous) incoming gradient ``v`` of a sum over the tokens, read
    through a copy.  The mode has no name in the ``cake_pplx`` namespace: the module is imported by its full name."""
    _require(hidden)
    T = 513

    def case(j: int, s: int):
        g = _generator(s)
        X = _draw((T, 4, hidden), "spread", g)
        dY = _draw((T, hidden), "spread", g)
        v = _draw((hidden,), "spread", g)
        ours = X.detach().requires_grad_(True)
        ref = X.detach().requires_grad_(True)
        Y = facade.cake_mhc_head_nondet(ours)
        assert Y.grad_fn is not None
        launch = {"Y": Y.detach()}
        naive = {"Y": ref.mean(dim=1).detach()}
        expected = {"Y": forward_oracle(X)}
        for n in range(2):
            facade.cake_mhc_head_nondet(ours).backward(dY)
            ref.mean(dim=1).backward(dY)
            launch[f"dX pass {n}"] = ours.grad.clone()
            naive[f"dX pass {n}"] = ref.grad.clone()
        expected["dX pass 0"] = backward_oracle(dY)
        expected["dX pass 1"] = backward_oracle(dY, backward_naive(dY))
        ours.grad = ref.grad = None
        facade.cake_mhc_head_nondet(ours).float().sum(dim=0).backward(v.float())
        ref.mean(dim=1).float().sum(dim=0).backward(v.float())
        launch["dX expanded"], naive["dX expanded"] = ours.grad, ref.grad
        expected["dX expanded"] = backward_oracle(v.expand(T, hidden))
        launch["Function.apply Y"] = facade.CakeMHCHeadNondetFunction.apply(X)
        expected["Function.apply Y"], naive["Function.apply Y"] = (
            expected["Y"],
            naive["Y"],
        )
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED + 1)
    assert not hasattr(cake_pplx, "cake_mhc_head_nondet")
    assert not hasattr(cake_pplx, "cake_mhc_head_nondet_train_forward")
    assert not hasattr(cake_pplx, "CakeMHCHeadNondetFunction")


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_activation_checkpoint(
    hidden: int, use_reentrant: bool, record_property
) -> None:
    """``head(X * 2) * W`` inside ``torch.utils.checkpoint``: the output and the gradient of ``X`` judged over 32
    draws (the tier: the same graph with PyTorch's ``X.mean(dim=1)``)."""
    _require(hidden)

    def case(j: int, s: int):
        g = _generator(s)
        X = _draw((257, 4, hidden), "randn", g)
        W = torch.randn((hidden,), device=_device(), generator=g).to(BF16)
        dY = _draw((257, hidden), "randn", g)

        def block(x: torch.Tensor, head) -> torch.Tensor:
            return head(x * 2) * W

        ours = X.detach().requires_grad_(True)
        ref = X.detach().requires_grad_(True)
        out = torch.utils.checkpoint.checkpoint(
            block, ours, facade.cake_mhc_head_nondet, use_reentrant=use_reentrant
        )
        want = block(ref, lambda x: x.mean(dim=1))
        out.backward(dY)
        want.backward(dY)
        oracle = _block_oracle(X, W, dY)
        launch = {"output": out.detach(), "dX": ours.grad}
        naive = {"output": want.detach(), "dX": ref.grad}
        return [launch], {k: oracle[k] for k in launch}, naive

    judge_draws(case, record_property, seed=SEED + 2)


def test_cuda_graph_capture(record_property) -> None:
    """The forward, the writing and the accumulating backward captured in one CUDA graph, replayed on 32 draws of
    ``special`` inputs copied in place (the accumulation buffer a new draw before each replay): every output judged."""
    T, hidden = 1000, max(_hidden_sizes())
    g = _generator(SEED + 3)
    X = _draw((T, 4, hidden), "randn", g)
    dY = _draw((T, hidden), "randn", g)
    dXa = _draw((T, 4, hidden), "randn", g)
    Y = torch.empty((T, hidden), dtype=BF16, device=_device())
    dX = torch.empty((T, 4, hidden), dtype=BF16, device=_device())
    acc0 = dXa.clone()

    def step() -> None:
        facade.cake_mhc_head_nondet_train_forward(X, out=Y)
        facade.cake_mhc_head_nondet_train_backward(dY, dX=dX)
        facade.cake_mhc_head_nondet_train_backward(dY, dX=dXa, accumulate=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()  # loads the programs outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    dXa.copy_(acc0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    def case(j: int, s: int):
        gj = _generator(s)
        X.copy_(_draw((T, 4, hidden), "special", gj))
        dY.copy_(_draw((T, hidden), "special", gj))
        dXa.copy_(_draw((T, 4, hidden), "special", gj))
        before = dXa.clone()
        Y.fill_(float("nan"))
        dX.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        launch = {"Y": Y.clone(), "dX": dX.clone(), "dX accumulate": dXa.clone()}
        expected = {
            "Y": forward_oracle(X),
            "dX": backward_oracle(dY),
            "dX accumulate": backward_oracle(dY, before),
        }
        naive = {
            "Y": forward_naive(X),
            "dX": backward_naive(dY),
            "dX accumulate": backward_naive(dY, before),
        }
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED + 3)


@pytest.mark.parametrize("warn_only", [False, True], ids=["alert", "warn_only"])
def test_deterministic_algorithms_raise_and_warn_nothing(
    warn_only: bool, record_property
) -> None:
    """With ``torch.use_deterministic_algorithms(True)`` every entry point -- forward, writing and accumulating
    backward, the autograd node and its checkpointed recompute -- runs without raising PyTorch's alert and, with
    ``warn_only=True``, without a warning (every warning is an error here), on 32 draws whose outputs are judged."""
    hidden = max(_hidden_sizes())
    g = _generator(SEED + 4)
    X = _draw((257, 4, hidden), "spread", g)
    dY = _draw((257, hidden), "spread", g)
    dX0 = _draw((257, 4, hidden), "spread", g)
    # Load the programs before the mode is on, so that a first-use build is not inside the check.
    facade.cake_mhc_head_nondet_train_forward(X)
    facade.cake_mhc_head_nondet_train_backward(dY, dX=dX0.clone(), accumulate=True)
    facade.cake_mhc_head_nondet_train_backward(dY)
    torch.cuda.synchronize()

    def case(j: int, s: int):
        gj = _generator(s)
        X = _draw((257, 4, hidden), "spread", gj)
        dY = _draw((257, hidden), "spread", gj)
        dX0 = _draw((257, 4, hidden), "spread", gj)
        enabled = torch.are_deterministic_algorithms_enabled()
        was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                Y = facade.cake_mhc_head_nondet_train_forward(X)
                dX = facade.cake_mhc_head_nondet_train_backward(dY)
                acc = dX0.clone()
                facade.cake_mhc_head_nondet_train_backward(dY, dX=acc, accumulate=True)
                node = X.detach().requires_grad_(True)
                facade.cake_mhc_head_nondet(node).backward(dY)
                checkpointed = X.detach().requires_grad_(True)
                torch.utils.checkpoint.checkpoint(
                    facade.cake_mhc_head_nondet, checkpointed, use_reentrant=False
                ).backward(dY)
                torch.cuda.synchronize()
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
        launch = {
            "Y": Y,
            "dX": dX,
            "dX accumulate": acc,
            "autograd dX": node.grad,
            "checkpointed dX": checkpointed.grad,
        }
        grad, grad_naive = backward_oracle(dY), backward_naive(dY)
        expected = {
            "Y": forward_oracle(X),
            "dX": grad,
            "dX accumulate": backward_oracle(dY, dX0),
            "autograd dX": grad,
            "checkpointed dX": grad,
        }
        naive = {
            "Y": forward_naive(X),
            "dX": grad_naive,
            "dX accumulate": backward_naive(dY, dX0),
            "autograd dX": grad_naive,
            "checkpointed dX": grad_naive,
        }
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED + 4)


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", REPEAT_TOKENS)
def test_repeats_match_fp64(hidden: int, T: int, record_property) -> None:
    """The forward, the backward and the accumulating backward each run ``REPEATS`` times on identical inputs, on 32
    draws: every run is judged (the ``nondet`` contract promises no run-to-run bits, so the runs are not compared
    with each other).  A run writes into a buffer holding the inverted bits of the first run's result, so an element
    the forward or the writing backward leaves unwritten shows as an error; the accumulating backward reads its
    buffer, so each run adds into a fresh copy of the same ``dX0``."""
    _require(hidden)

    def case(j: int, s: int):
        g = _generator(s)
        X = _draw((T, 4, hidden), "special", g)
        dY = _draw((T, hidden), "special", g)
        dX0 = _draw((T, 4, hidden), "special", g)
        Y = facade.cake_mhc_head_nondet_train_forward(X)
        dX = facade.cake_mhc_head_nondet_train_backward(dY)
        acc = dX0.clone()
        facade.cake_mhc_head_nondet_train_backward(dY, dX=acc, accumulate=True)
        expected = {
            "Y": forward_oracle(X),
            "dX": backward_oracle(dY),
            "dX accumulate": backward_oracle(dY, dX0),
        }
        naive = {
            "Y": forward_naive(X),
            "dX": backward_naive(dY),
            "dX accumulate": backward_naive(dY, dX0),
        }
        yield [{"Y": Y, "dX": dX, "dX accumulate": acc}], expected, naive
        for _run in range(1, REPEATS):
            y = _complement(Y)
            assert facade.cake_mhc_head_nondet_train_forward(X, out=y) is y
            dx = _complement(dX)
            assert facade.cake_mhc_head_nondet_train_backward(dY, dX=dx) is dx
            a = dX0.clone()
            facade.cake_mhc_head_nondet_train_backward(dY, dX=a, accumulate=True)
            yield [{"Y": y, "dX": dx, "dX accumulate": a}], expected, naive
            del y, dx, a

    judge_draws(case, record_property, seed=SEED + 5 + T * 31 + hidden)


@pytest.mark.parametrize("hidden", HIDDEN)
@pytest.mark.parametrize("T", (257, 16263))
def test_checkpoint_replay_matches_fp64(hidden: int, T: int, record_property) -> None:
    """Under ``torch.utils.checkpoint(..., use_reentrant=False)`` around a small graph (``head(X * 2) * W``), over 32
    draws: the head output of the checkpoint's forward and of its recompute in the backward, the output and the
    gradients of ``X`` and ``W`` (``W``'s gradient reads the recomputed head output) are judged against the same
    graph with PyTorch's ``X.mean(dim=1)`` (the ``nondet`` contract promises no bits, so the replay is not compared
    with a run without the checkpoint)."""
    _require(hidden)

    def case(j: int, s: int):
        g = _generator(s)
        X = _draw((T, 4, hidden), "spread", g)
        W = _draw((hidden,), "randn", g)
        dOut = _draw((T, hidden), "randn", g)
        heads: list[torch.Tensor] = []

        def block(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            y = facade.cake_mhc_head_nondet(x * 2)
            heads.append(y.detach().clone())
            return y * w

        x = X.detach().requires_grad_(True)
        w = W.detach().requires_grad_(True)
        out = torch.utils.checkpoint.checkpoint(block, x, w, use_reentrant=False)
        out.backward(dOut)
        assert len(heads) == 2, len(heads)
        oracle = _block_oracle(X, W, dOut)
        tier = _block_naive(X, W, dOut)
        launch = {
            "head forward": heads[0],
            "head recompute": heads[1],
            "output": out.detach(),
            "dX": x.grad,
            "dW": w.grad,
        }
        names = {k: k.split()[0] for k in launch}
        expected = {k: oracle[n] for k, n in names.items()}
        naive = {k: tier[n] for k, n in names.items()}
        return [launch], expected, naive

    judge_draws(case, record_property, seed=SEED + 6 + T * 31 + hidden)
