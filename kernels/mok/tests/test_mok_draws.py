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
helpers) fill a CUDA tensor in consecutive torch calls of ``DRAW_CHUNK`` = 2^18 elements, so they do not depend on the
GPU's SM count.  On the CPU: the chunks of ``n`` elements are ``ceil(n / 2^18)`` consecutive parts of ``2^18`` elements
(the last one the rest), filled in order, and filling them from one generator gives torch's draws of each part in turn;
a CPU tensor is one torch call with torch's values; and no module of ``tests/experimental`` calls torch's random
functions itself.  On a GPU: each helper gives the concatenation of torch's draws of each chunk from the same generator
state (the default generator and an explicit one), each chunk moves the generator's offset by 4, and a draw of at most
2^18 elements is torch's own single call.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch

from tests.experimental import _mok_reference as draws

CHUNK = draws.DRAW_CHUNK
SIZE = (3, 1000)
#: 600,000 elements: two whole chunks and a part of 75,712.
LARGE = (5, 3, 40_000)

#: Per helper: ``(draw(shape, generator, device) by the helper, the same draw by one torch call)``.
CASES = {
    "randn": (
        lambda shape, g, device: draws.randn(*shape, generator=g, device=device),
        lambda shape, g, device: torch.randn(*shape, generator=g, device=device),
    ),
    "randn-bf16": (
        lambda shape, g, device: draws.randn(
            *shape, generator=g, dtype=torch.bfloat16, device=device
        ),
        lambda shape, g, device: torch.randn(
            *shape, generator=g, dtype=torch.bfloat16, device=device
        ),
    ),
    "randn_like": (
        lambda shape, g, device: draws.randn_like(
            torch.empty(shape, dtype=torch.bfloat16, device=device), generator=g
        ),
        lambda shape, g, device: torch.randn(
            *shape, generator=g, dtype=torch.bfloat16, device=device
        ),
    ),
    "rand": (
        lambda shape, g, device: draws.rand(*shape, generator=g, device=device),
        lambda shape, g, device: torch.rand(*shape, generator=g, device=device),
    ),
    "randint": (
        lambda shape, g, device: draws.randint(
            -1, 9, shape, generator=g, dtype=torch.int32, device=device
        ),
        lambda shape, g, device: torch.randint(
            -1, 9, shape, generator=g, dtype=torch.int32, device=device
        ),
    ),
    "normal_": (
        lambda shape, g, device: draws.normal_(
            torch.empty(shape, dtype=torch.bfloat16, device=device),
            std=0.125,
            generator=g,
        ),
        lambda shape, g, device: torch.empty(
            shape, dtype=torch.bfloat16, device=device
        ).normal_(std=0.125, generator=g),
    ),
    "uniform_": (
        lambda shape, g, device: draws.uniform_(
            torch.empty(shape, device=device), 0.1, 1.0, generator=g
        ),
        lambda shape, g, device: torch.empty(shape, device=device).uniform_(
            0.1, 1.0, generator=g
        ),
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


def _chunk_sizes(n):
    """The element counts of the torch calls that draw ``n`` elements on a GPU."""
    return [CHUNK] * (n // CHUNK) + ([n % CHUNK] if n % CHUNK else [])


def _per_chunk(torch_draw, shape, g):
    """torch's draws of each chunk of a GPU draw of ``shape`` from ``g``, one after another, as one tensor."""
    parts = [torch_draw((size,), g, "cuda") for size in _chunk_sizes(math.prod(shape))]
    return torch.cat(parts).view(shape)


@pytest.mark.parametrize(
    "shape",
    [(1,), (CHUNK - 1,), (CHUNK,), (CHUNK + 1,), (3, CHUNK + 5), (2, 3, 65_537)],
    ids=str,
)
def test_chunks_cover_the_tensor_in_order(shape):
    """A tensor of ``n`` elements is filled in ``ceil(n / 2^18)`` calls, on consecutive parts of ``2^18`` elements in
    order (the last part the rest), each a view of the tensor's storage."""
    out = torch.empty(shape, dtype=torch.int64)
    base = out.untyped_storage().data_ptr()
    calls = []

    def fill(part):
        assert part.dim() == 1 and part.untyped_storage().data_ptr() == base
        calls.append((part.storage_offset(), part.numel()))
        part.fill_(len(calls) - 1)

    assert draws.fill_in_chunks(out, fill) is out
    n = out.numel()
    sizes = _chunk_sizes(n)
    assert len(calls) == math.ceil(n / CHUNK) == len(sizes)
    assert calls == [(k * CHUNK, size) for k, size in enumerate(sizes)]
    expected = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    assert torch.equal(out.view(-1), expected)


@pytest.mark.parametrize("name", ["randn", "rand", "randint", "normal_"])
def test_chunks_draw_from_one_generator_in_order(name):
    """Filling the chunks from one generator gives torch's draw of each chunk in turn from that generator, and leaves
    the generator where those draws leave it."""
    fill = {
        "randn": lambda t, g: t.normal_(generator=g),
        "rand": lambda t, g: t.uniform_(generator=g),
        "randint": lambda t, g: t.random_(-1, 9, generator=g),
        "normal_": lambda t, g: t.normal_(0.5, 0.125, generator=g),
    }[name]
    dtype = torch.int32 if name == "randint" else torch.float32
    n = 2 * CHUNK + 1000
    g = torch.Generator().manual_seed(11)
    got = draws.fill_in_chunks(torch.empty(n, dtype=dtype), lambda t: fill(t, g))
    g_parts = torch.Generator().manual_seed(11)
    want = torch.cat(
        [fill(torch.empty(size, dtype=dtype), g_parts) for size in _chunk_sizes(n)]
    )
    assert torch.equal(got, want)
    assert torch.equal(g.get_state(), g_parts.get_state())


def test_a_cpu_draw_is_one_call():
    """A CPU tensor, larger than a chunk too, is filled by one call on the tensor itself."""
    out = torch.empty(3, CHUNK + 5)
    calls = []
    assert draws._draw(out, calls.append) is out
    assert len(calls) == 1 and calls[0] is out


@pytest.mark.parametrize("name", list(CASES))
def test_cpu_draws_are_torchs_own(name):
    """On the CPU each helper gives the values of the same torch call, on an explicit generator and on the default
    generator that ``torch.manual_seed`` seeds, and moves the generator as far."""
    draw, torch_draw = CASES[name]
    g, g_torch = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    got, want = draw(SIZE, g, None), torch_draw(SIZE, g_torch, None)
    assert got.device.type == "cpu" and got.dtype == want.dtype
    assert torch.equal(got, want)
    assert torch.equal(g.get_state(), g_torch.get_state())
    with _own_random_state():
        torch.manual_seed(7)
        got = draw(SIZE, None, None)
        after_draw = torch.get_rng_state()
        torch.manual_seed(7)
        want = torch_draw(SIZE, None, None)
        assert torch.equal(after_draw, torch.get_rng_state())
    assert got.dtype == want.dtype and torch.equal(got, want)


def test_gpu_tests_draw_only_through_the_helpers():
    """No module of ``tests/experimental`` but the helpers' own calls torch's random functions or methods: every random
    input of the GPU tests is drawn in chunks."""
    root = Path(__file__).resolve().parent / "experimental"
    calls = [
        f"{path.name}:{number}: {line.strip()}"
        for path in sorted(root.glob("*.py"))
        if path.name != "_mok_reference.py"
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if RANDOM_CALL.search(line)
    ]
    assert not calls, "\n".join(calls)


def test_gpu_draws_are_torchs_draws_per_chunk():
    """On a GPU each helper gives the concatenation of torch's draws of each chunk from the same generator state (an
    explicit CUDA generator and the default one), each chunk moves the offset by 4, and a draw of at most one chunk is
    torch's own single call."""
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device")
    chunks = len(_chunk_sizes(math.prod(LARGE)))
    assert chunks == 3
    for name, (draw, torch_draw) in CASES.items():
        g = torch.Generator(device="cuda").manual_seed(7)
        g_parts = torch.Generator(device="cuda").manual_seed(7)
        got, want = draw(LARGE, g, "cuda"), _per_chunk(torch_draw, LARGE, g_parts)
        assert got.is_cuda and got.dtype == want.dtype, name
        assert torch.equal(got, want), name
        assert g.get_offset() == g_parts.get_offset() == 4 * chunks, name
        with _own_random_state():
            torch.manual_seed(7)
            got = draw(LARGE, None, "cuda")
            after_draw = torch.cuda.get_rng_state()
            torch.manual_seed(7)
            want = _per_chunk(torch_draw, LARGE, None)
            assert torch.equal(after_draw, torch.cuda.get_rng_state()), name
            assert torch.equal(got, want), name
            torch.manual_seed(7)
            got = draw(SIZE, None, "cuda")
            torch.manual_seed(7)
            want = torch_draw(SIZE, None, "cuda")
            assert torch.equal(got, want), name
