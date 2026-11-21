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

The random inputs of the MoK tests (:mod:`tests.experimental._mok_reference`, :func:`randn` and the other draw
helpers) are drawn on the CPU from a CPU generator and then moved, so they do not depend on the GPU.  On the CPU: each
helper gives the values of the same torch call on a CPU generator, an explicit one or the default one; a tensor for
another device (``meta`` stands in for a GPU) takes its values from the CPU generator; and no module of
``tests/experimental`` calls torch's random functions itself.  On a GPU: the helpers give the CPU's values, leave the
GPU's generator where it was, and refuse a CUDA generator.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from tests.experimental import _mok_reference as draws

SIZE = (3, 1000)

#: Per helper: ``(draw(generator, device) by the helper, the same draw by torch on the CPU)``.
CASES = {
    "randn": (
        lambda g, device: draws.randn(*SIZE, generator=g, device=device),
        lambda g: torch.randn(*SIZE, generator=g),
    ),
    "randn-bf16": (
        lambda g, device: draws.randn(
            *SIZE, generator=g, dtype=torch.bfloat16, device=device
        ),
        lambda g: torch.randn(*SIZE, generator=g).bfloat16(),
    ),
    "randn_like": (
        lambda g, device: draws.randn_like(
            torch.empty(SIZE, dtype=torch.bfloat16, device=device), generator=g
        ),
        lambda g: torch.randn(*SIZE, generator=g).bfloat16(),
    ),
    "rand": (
        lambda g, device: draws.rand(*SIZE, generator=g, device=device),
        lambda g: torch.rand(*SIZE, generator=g),
    ),
    "randint": (
        lambda g, device: draws.randint(
            -1, 9, SIZE, generator=g, dtype=torch.int32, device=device
        ),
        lambda g: torch.randint(-1, 9, SIZE, generator=g, dtype=torch.int32),
    ),
    "normal_": (
        lambda g, device: draws.normal_(
            torch.empty(SIZE, dtype=torch.bfloat16, device=device),
            std=0.125,
            generator=g,
        ),
        lambda g: torch.empty(SIZE).normal_(std=0.125, generator=g).bfloat16(),
    ),
    "uniform_": (
        lambda g, device: draws.uniform_(
            torch.empty(SIZE, device=device), 0.1, 1.0, generator=g
        ),
        lambda g: torch.empty(SIZE).uniform_(0.1, 1.0, generator=g),
    ),
}

#: A call of one of torch's random functions or in-place random methods.
RANDOM_CALL = re.compile(
    r"\btorch\.(rand|randn|randint|randperm|rand_like|randn_like|randint_like|normal"
    r"|bernoulli|multinomial|poisson)\(|\.(normal_|uniform_|random_|bernoulli_"
    r"|exponential_|cauchy_|geometric_|log_normal_)\("
)


def _own_random_state():
    """Restores the CPU's and every CUDA device's random state on exit."""
    return torch.random.fork_rng(devices=range(torch.cuda.device_count()))


@pytest.mark.parametrize("name", list(CASES))
def test_draws_are_the_cpu_generators_values(name):
    """The same seed gives the values of the same torch call on the CPU: on an explicit generator, and on the default
    generator that ``torch.manual_seed`` seeds."""
    draw, torch_draw = CASES[name]
    got = draw(torch.Generator().manual_seed(7), None)
    want = torch_draw(torch.Generator().manual_seed(7))
    assert got.device.type == "cpu" and got.dtype == want.dtype
    assert torch.equal(got, want)
    with _own_random_state():
        torch.manual_seed(7)
        got = draw(None, None)
        torch.manual_seed(7)
        want = torch_draw(None)
    assert got.dtype == want.dtype and torch.equal(got, want)


@pytest.mark.parametrize("name", list(CASES))
def test_a_draw_for_another_device_comes_from_the_cpu_generator(name):
    """A tensor for another device is drawn on the CPU generator as a CPU tensor of its shape: the generator moves as
    far as for that CPU draw (a draw on the other device would leave it where it was)."""
    draw, torch_draw = CASES[name]
    with _own_random_state():
        torch.manual_seed(7)
        got = draw(None, "meta")
        after_draw = torch.rand(16)
        torch.manual_seed(7)
        want = torch_draw(None)
        after_torch_draw = torch.rand(16)
    assert got.device.type == "meta"
    assert got.dtype == want.dtype and got.shape == want.shape
    assert torch.equal(after_draw, after_torch_draw)


def test_gpu_tests_draw_only_through_the_helpers():
    """No module of ``tests/experimental`` but the helpers' own calls torch's random functions or methods: every random
    input of the GPU tests is drawn on the CPU."""
    root = Path(__file__).resolve().parent / "experimental"
    calls = [
        f"{path.name}:{number}: {line.strip()}"
        for path in sorted(root.glob("*.py"))
        if path.name != "_mok_reference.py"
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if RANDOM_CALL.search(line)
    ]
    assert not calls, "\n".join(calls)


def test_gpu_draws_are_the_cpu_values():
    """On a GPU every helper gives the CPU generator's values and leaves the GPU's generator where it was; a CUDA
    generator is refused."""
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device")
    with _own_random_state():
        for name, (draw, torch_draw) in CASES.items():
            torch.manual_seed(7)
            before = torch.cuda.get_rng_state()
            got = draw(None, "cuda")
            assert torch.equal(torch.cuda.get_rng_state(), before), name
            torch.manual_seed(7)
            want = torch_draw(None)
            assert got.is_cuda and got.dtype == want.dtype, name
            assert torch.equal(got.cpu(), want), name
    with pytest.raises(RuntimeError, match="generator"):
        draws.randn(4, generator=torch.Generator(device="cuda"), device="cuda")
