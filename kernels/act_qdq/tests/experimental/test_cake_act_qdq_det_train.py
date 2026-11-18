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

Tests of the Cake activation quantize-dequantize forward (SM100 / SM103 /
SM107).

Every output -- the BF16 values ``y`` and the FP32 scales -- is compared bit
for bit with the FP64 rule
(``tests/experimental/test_helpers/cake_act_qdq_det_train_reference.py``), so a
``-0.0`` against a ``+0.0`` fails; there is no tolerance.  The inputs are the
edge rows of each format, every finite BF16 value under every scale the format
reaches, and random inputs of the model shapes (2-D, 3-D and 4-D, 1 to 4097
rows, widths 16 to 4096, a width of 96; each released call site at 8192
rows).  Each runs out of place and in place,
with and without the scale output, and every program also runs directly on
grids of 1 and 3 CTAs.  The suite also checks: out-of-place calls leave ``x``
unchanged; caller ``out`` and ``scales`` buffers; five repeated calls bit for
bit; argument validation; no launch for empty inputs; CUDA graph replay; no
rebuild on a new shape; ``torch.utils.checkpoint`` through a straight-through
function; steady-state calls without host synchronization; no
deterministic-algorithms alert; the zero workspace; the registry and the
``build.json`` contract; and no FP64 in the CUDA sources.

This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
``cake_pplx``.  The facade is covered by ``test_cake_act_qdq_det_train_api.py``.
GPU tests skip without an SM100 / SM103 / SM107 device; the ``cpu_only`` tests
read only the registry and the package files.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers import cake_act_qdq_det_train_reference as reference

PACKAGE = "cake.cake_act_qdq_det_train"
pkg = cake_package(PACKAGE)
cake_build = cake_package(PACKAGE, "cake_build")
loader = cake_package(PACKAGE, "cake_jit")
forward = pkg.cake_act_qdq_det_train_forward

FORMATS = reference.FORMATS
REPEATS = 5
# The minimum CUDA toolkit of every library, per architecture.
MIN_CUDA = {"sm_100a": "12.9", "sm_103a": "12.9", "sm_107a": "13.5"}


def _supported() -> bool:
    return torch.cuda.is_available() and pkg.is_cake_act_qdq_det_train_supported(
        torch.device("cuda", 0)
    )


gpu = pytest.mark.skipif(
    not _supported(),
    reason="cake_act_qdq_det_train needs an SM100 / SM103 / SM107 device",
)


def _arch() -> str:
    return loader.device_arch(0)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader builds on
    # first use; a prebuilt or cached library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


# --------------------------------------------------------------------------- helpers


def _scale_shape(shape: tuple[int, ...], fmt: str) -> tuple[int, ...]:
    return (*shape[:-1], shape[-1] // reference.GROUP[fmt])


def _assert_bits(what: str, got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape, f"{what}: shape {tuple(got.shape)}, expected {tuple(want.shape)}"
    if torch.equal(got, want):
        return
    bad = (got != want).nonzero()
    first = tuple(bad[0].tolist())
    raise AssertionError(
        f"{what}: {bad.shape[0]} of {got.numel()} values differ; first at {first}: "
        f"bits {int(got[first]) & 0xFFFFFFFF:#x}, expected {int(want[first]) & 0xFFFFFFFF:#x}"
    )


def _check(what: str, want: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor, s: torch.Tensor | None) -> None:
    """``y`` (and ``s``) equal the FP64 rule's ``want`` bit for bit."""
    want_y, want_s = want
    assert y.dtype == torch.bfloat16 and y.is_contiguous(), what
    _assert_bits(f"{what}: y", reference.y_bits(y), reference.y_bits(want_y))
    if s is not None:
        assert s.dtype == torch.float32 and s.is_contiguous(), what
        _assert_bits(f"{what}: scales", reference.scale_bits(s), reference.scale_bits(want_s))


def _call(x0: torch.Tensor, fmt: str, *, inplace: bool, return_scales: bool):
    """One entry call on a copy of ``x0``: ``(y, scales or None)``.  In place, ``y`` is the copy;
    out of place, the copy is left unchanged."""
    x = x0.clone()
    result = forward(x, format=fmt, out=x if inplace else None, return_scales=return_scales)
    y, s = result if return_scales else (result, None)
    if inplace:
        assert y is x
    else:
        assert y is not x
        _assert_bits("x after an out-of-place call", reference.y_bits(x), reference.y_bits(x0))
    if return_scales:
        assert tuple(s.shape) == _scale_shape(tuple(x0.shape), fmt)
    else:
        assert isinstance(result, torch.Tensor)
    return y, s


def _shapes(fmt: str) -> tuple[tuple[int, ...], ...]:
    G = reference.GROUP[fmt]
    return (
        (1, G),
        (17, 96),
        (129, 512),
        (129, 128),
        (65, 32, 128),
        (17, 8, 4096),
        (4097, 512),
        (3, 5, 7, 64),
    )


RANDOM_CASES = [
    pytest.param(fmt, shape, id=f"{fmt}-{'x'.join(map(str, shape))}") for fmt in FORMATS for shape in _shapes(fmt)
]


# --------------------------------------------------------------------------- numerics against the FP64 rule


@gpu
@pytest.mark.parametrize("inplace", (False, True), ids=("out_of_place", "in_place"))
@pytest.mark.parametrize("fmt", FORMATS)
def test_edge_rows_match_the_rule(fmt: str, inplace: bool) -> None:
    x0 = reference.edge_input(fmt)
    want = reference.qdq(x0, fmt)
    for return_scales in (True, False):
        y, s = _call(x0, fmt, inplace=inplace, return_scales=return_scales)
        _check(f"{fmt} edge rows inplace={inplace} scales={return_scales}", want, y, s)


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_finite_bf16_under_every_scale_matches_the_rule(fmt: str) -> None:
    """The exhaustive sweep: every finite BF16 value under every scale the format reaches, through the program
    with the scale output (out of place) and the one without (in place)."""
    x0 = reference.sweep_input(fmt)
    want = reference.qdq(x0, fmt)
    y, s = _call(x0, fmt, inplace=False, return_scales=True)
    _check(f"{fmt} sweep with scales", want, y, s)
    y, _ = _call(x0, fmt, inplace=True, return_scales=False)
    _check(f"{fmt} sweep in place", want, y, None)


@gpu
@pytest.mark.parametrize("fmt,shape", RANDOM_CASES)
def test_random_inputs_match_the_rule(fmt: str, shape: tuple[int, ...]) -> None:
    x0 = reference.randn_input(shape, fmt, seed=sum(shape))
    want = reference.qdq(x0, fmt)
    for inplace in (False, True):
        for return_scales in (True, False):
            y, s = _call(x0, fmt, inplace=inplace, return_scales=return_scales)
            _check(f"{fmt} {shape} inplace={inplace} scales={return_scales}", want, y, s)


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_wide_exponent_range_matches_the_rule(fmt: str) -> None:
    # Group scales from 2^-120 to 2^120: subnormal inputs, the smallest and the largest scales.
    x0 = reference.randn_input((33, 8, 512), fmt, lo=120, hi=120, seed=31)
    want = reference.qdq(x0, fmt)
    for inplace in (False, True):
        y, s = _call(x0, fmt, inplace=inplace, return_scales=True)
        _check(f"{fmt} wide range inplace={inplace}", want, y, s)


# The released call sites of each format at T = 8192 tokens.
MODEL_CASES = [
    pytest.param(fmt, shape, id=f"{fmt}-{'x'.join(map(str, shape))}")
    for fmt, shape in (
        ("fp4_e4m3_g16", (8192, 512)),
        ("mxfp4_e8m0_g32", (8192, 128)),
        ("mxfp4_e8m0_g32", (8192, 32, 128)),
        ("mxfp8_e8m0_g32", (8192, 512)),
        ("mxfp8_e8m0_g32", (8192, 8, 4096)),
    )
]


@gpu
@pytest.mark.parametrize("fmt,shape", MODEL_CASES)
def test_model_shapes_at_full_size_match_the_rule(fmt: str, shape: tuple[int, ...]) -> None:
    """Random inputs at the full size of each released call site, out of place with the scales and in place."""
    x0 = reference.randn_input(shape, fmt, seed=43)
    want = reference.qdq(x0, fmt)
    y, s = _call(x0, fmt, inplace=False, return_scales=True)
    _check(f"{fmt} {shape} with scales", want, y, s)
    del y, s
    y, _ = _call(x0, fmt, inplace=True, return_scales=False)
    _check(f"{fmt} {shape} in place", want, y, None)


@gpu
def test_rank_does_not_change_the_bits() -> None:
    """Groups run along the flat buffer: a 3-D input and its 2-D views give the same bits."""
    for fmt in FORMATS:
        x0 = reference.randn_input((65, 32, 128), fmt, seed=3)
        y3, s3 = forward(x0, format=fmt, return_scales=True)
        y2, s2 = forward(x0.view(-1, 128), format=fmt, return_scales=True)
        _assert_bits(f"{fmt} y", reference.y_bits(y3).view(-1), reference.y_bits(y2).view(-1))
        _assert_bits(f"{fmt} scales", reference.scale_bits(s3).view(-1), reference.scale_bits(s2).view(-1))


@gpu
def test_every_program_runs_on_any_grid() -> None:
    """Every program of the device's architecture, launched directly, on 1 and 3 CTAs (many grid-stride trips)
    and on the host's grid, out of place and in place."""
    arch = _arch()
    for kind, names in loader.ROUTES[arch].items():
        (name,) = names
        record = loader.MODULES[name]
        fmt = record["format"]
        x0 = reference.randn_input((4097, 96), fmt, seed=41)
        want = reference.qdq(x0, fmt)
        n_vecs = x0.numel() // loader.VECTOR
        for grid in ((1, 1, 1), (3, 1, 1), None):
            for inplace in (False, True):
                x = x0.clone()
                y = x if inplace else torch.full_like(x0, float("nan"))
                values = {"x": x, "y": y, "n_vecs": n_vecs}
                s = None
                if record["scale_out"]:
                    s = torch.full(_scale_shape(tuple(x0.shape), fmt), float("nan"), device="cuda")
                    values["scales"] = s
                loader.run_module(name, values, device_index=0, grid=grid)
                _check(f"{kind} {name} grid={grid} inplace={inplace}", want, y, s)


# --------------------------------------------------------------------------- outputs, buffers and repeats


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_out_and_scales_buffers(fmt: str) -> None:
    shape = (129, 512)
    x0 = reference.randn_input(shape, fmt, seed=7)
    want = reference.qdq(x0, fmt)
    out = torch.empty_like(x0)
    sc = torch.empty(_scale_shape(shape, fmt), dtype=torch.float32, device="cuda")
    out_version, sc_version = out._version, sc._version
    # A scales buffer requests the scales, with or without return_scales.
    result = forward(x0, format=fmt, out=out, scales=sc)
    assert isinstance(result, tuple) and result[0] is out and result[1] is sc
    _check(f"{fmt} caller buffers", want, out, sc)
    assert out._version > out_version and sc._version > sc_version
    sc2 = torch.empty_like(sc)
    y, s = forward(x0, format=fmt, scales=sc2, return_scales=True)
    assert s is sc2
    _check(f"{fmt} scales buffer", want, y, s)
    y, s = forward(x0, format=fmt, return_scales=True)
    _check(f"{fmt} new scales", want, y, s)
    y = forward(x0, format=fmt)
    assert isinstance(y, torch.Tensor)
    _check(f"{fmt} y only", want, y, None)
    # In place advances the version counter of x.
    x = x0.clone()
    version = x._version
    assert forward(x, format=fmt, out=x) is x
    assert x._version > version
    _check(f"{fmt} in place", want, x, None)


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_outputs_repeat_bit_for_bit(fmt: str) -> None:
    for shape in ((65, 32, 128), (17, 8, 4096), (4097, 96)):
        x0 = reference.randn_input(shape, fmt, seed=13)
        runs = [forward(x0, format=fmt, return_scales=True) for _ in range(REPEATS)]
        inplace = [_call(x0, fmt, inplace=True, return_scales=True) for _ in range(REPEATS)]
        first_y, first_s = runs[0]
        for call, (y, s) in enumerate(runs[1:] + inplace, start=1):
            _assert_bits(f"{fmt} {shape} call {call} y", reference.y_bits(y), reference.y_bits(first_y))
            _assert_bits(f"{fmt} {shape} call {call} scales", reference.scale_bits(s), reference.scale_bits(first_s))


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_empty_inputs_launch_nothing(fmt: str, monkeypatch) -> None:
    def no_launch(*args, **kwargs):
        raise AssertionError("a call with no elements launched a kernel")

    monkeypatch.setattr(loader, "run_module", no_launch)
    G = reference.GROUP[fmt]
    for shape in ((0, 512), (0, 32, 128), (4, 0), (2, 0, 64), (3, 0, G)):
        x = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        y, s = forward(x, format=fmt, return_scales=True)
        assert y.shape == x.shape and y.dtype == torch.bfloat16
        assert tuple(s.shape) == _scale_shape(shape, fmt) and s.dtype == torch.float32
        assert forward(x, format=fmt, out=x) is x
        assert forward(x, format=fmt).shape == x.shape


# --------------------------------------------------------------------------- argument validation


@gpu
def test_argument_validation() -> None:
    fmt = "mxfp8_e8m0_g32"
    x = reference.randn_input((4, 64), fmt, seed=1)
    with pytest.raises(ValueError, match="format"):
        forward(x, format="nvfp4")
    with pytest.raises(ValueError, match="format"):
        forward(x, format=None)
    with pytest.raises(ValueError, match="CUDA tensor"):
        forward(x.cpu(), format=fmt)
    with pytest.raises(TypeError, match="bfloat16"):
        forward(x.float(), format=fmt)
    with pytest.raises(ValueError, match="contiguous"):
        forward(reference.randn_input((64, 64), fmt).t(), format=fmt)
    with pytest.raises(ValueError, match="multiple of the group size"):
        forward(torch.zeros(4, 48, dtype=torch.bfloat16, device="cuda"), format="mxfp8_e8m0_g32")
    with pytest.raises(ValueError, match="multiple of the group size"):
        forward(torch.zeros(4, 24, dtype=torch.bfloat16, device="cuda"), format="fp4_e4m3_g16")
    with pytest.raises(ValueError, match="at least one dimension"):
        forward(torch.zeros((), dtype=torch.bfloat16, device="cuda"), format=fmt)
    flat = torch.zeros(4 * 64 + 8, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="16-byte aligned"):
        forward(flat[1 : 1 + 256].view(4, 64), format=fmt)

    # out
    with pytest.raises(TypeError, match="bfloat16"):
        forward(x, format=fmt, out=torch.empty(4, 64, device="cuda"))
    with pytest.raises(ValueError, match="shape"):
        forward(x, format=fmt, out=torch.empty(4, 32, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        forward(x, format=fmt, out=torch.empty(64, 4, dtype=torch.bfloat16, device="cuda").t())
    with pytest.raises(ValueError, match="CUDA tensor"):
        forward(x, format=fmt, out=torch.empty(4, 64, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="16-byte aligned"):
        forward(x, format=fmt, out=flat[1 : 1 + 256].view(4, 64))
    shared = torch.zeros(2 * 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        forward(shared[:256].view(4, 64), format=fmt, out=shared[8:264].view(4, 64))

    # scales
    with pytest.raises(TypeError, match="float32"):
        forward(x, format=fmt, scales=torch.empty(4, 2, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="shape"):
        forward(x, format=fmt, scales=torch.empty(4, 3, device="cuda"))
    with pytest.raises(ValueError, match="contiguous"):
        forward(x, format=fmt, scales=torch.empty(2, 4, device="cuda").t())
    raw = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    xs = raw[:512].view(torch.bfloat16).view(4, 64)
    with pytest.raises(ValueError, match="must not overlap"):
        forward(xs, format=fmt, scales=raw[256:288].view(torch.float32).view(4, 2))

    # autograd: a written buffer must not require grad while grad mode is on
    xg = x.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="requires grad"):
        forward(xg, format=fmt, out=xg)
    with pytest.raises(ValueError, match="requires grad"):
        forward(x, format=fmt, out=torch.empty_like(x).requires_grad_(True))
    with torch.no_grad():
        assert forward(xg, format=fmt, out=xg) is xg
    # an input that requires grad is read only: allowed, and the output has no history
    y = forward(x.clone().requires_grad_(True), format=fmt)
    assert not y.requires_grad and y.grad_fn is None


# --------------------------------------------------------------------------- graphs, rebuilds, autograd, sync


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_cuda_graph_replays_match_eager(fmt: str) -> None:
    x0 = reference.randn_input((65, 32, 128), fmt, seed=17)
    eager_y, eager_s = forward(x0, format=fmt, return_scales=True)
    xi = x0.clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            forward(x0, format=fmt, return_scales=True)
            forward(xi, format=fmt, out=xi)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cap_y, cap_s = forward(x0, format=fmt, return_scales=True)
        forward(xi, format=fmt, out=xi)
    for replay in range(3):
        xi.copy_(x0)
        cap_y.fill_(float("nan"))
        cap_s.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_bits(f"{fmt} replay {replay} y", reference.y_bits(cap_y), reference.y_bits(eager_y))
        _assert_bits(f"{fmt} replay {replay} scales", reference.scale_bits(cap_s), reference.scale_bits(eager_s))
        _assert_bits(f"{fmt} replay {replay} in place", reference.y_bits(xi), reference.y_bits(eager_y))


@gpu
def test_changing_shapes_do_not_rebuild() -> None:
    arch = _arch()
    for name in loader.MODULES:
        loader.load(name, arch)
    loaded_before = loader.load.cache_info().misses
    for shape in ((1, 32), (17, 96), (129, 512), (4097, 128), (65, 32, 128), (17, 8, 4096), (3, 5, 7, 64)):
        for fmt in FORMATS:
            x = reference.randn_input(shape, fmt)
            forward(x, format=fmt)
            forward(x, format=fmt, return_scales=True)
            forward(x, format=fmt, out=x)
    torch.cuda.synchronize()
    assert loader.load.cache_info().misses == loaded_before, "a changed shape rebuilt a module"


class _StraightThrough(torch.autograd.Function):
    """QDQ forward, identity backward (a straight-through estimator), local to this test."""

    @staticmethod
    def forward(ctx, x, fmt, calls):
        calls.append(fmt)
        return pkg.cake_act_qdq_det_train_forward(x, format=fmt)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


def _region(x, w, fmt, calls):
    # The product saves the QDQ output, so the checkpointed backward recomputes the QDQ.
    return _StraightThrough.apply(x, fmt, calls) * w


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_checkpoint_matches_plain_autograd(fmt: str) -> None:
    """``torch.utils.checkpoint`` (``use_reentrant=False``) runs the QDQ twice -- once in the step, once replayed
    in the backward -- and gives the outputs and gradients of plain autograd bit for bit."""
    x0 = reference.randn_input((129, 512), fmt, seed=19)
    gen = torch.Generator(device="cuda").manual_seed(23)
    w0 = torch.randn(512, device="cuda", generator=gen).bfloat16()
    grad = torch.randn(129, 512, device="cuda", generator=gen).bfloat16()
    expected = forward(x0, format=fmt) * w0

    def step(checkpointed: bool):
        calls: list[str] = []
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        if checkpointed:
            out = torch.utils.checkpoint.checkpoint(_region, x, w, fmt, calls, use_reentrant=False)
        else:
            out = _region(x, w, fmt, calls)
        assert len(calls) == 1
        out.backward(grad)
        assert len(calls) == (2 if checkpointed else 1), "the checkpointed QDQ was not replayed in the backward"
        return out.detach(), x.grad, w.grad

    plain = step(False)
    ckpt = step(True)
    _assert_bits(f"{fmt} plain output", reference.y_bits(plain[0]), reference.y_bits(expected))
    for what, a, b in zip(("output", "x.grad", "w.grad"), plain, ckpt, strict=True):
        _assert_bits(f"{fmt} checkpoint {what}", reference.y_bits(b), reference.y_bits(a))


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_steady_state_calls_do_not_synchronize(fmt: str) -> None:
    """Once the libraries are loaded and the SM count is known, a call runs under the error sync-debug mode: a
    device synchronization, a blocking copy or a read of device memory would raise."""
    x0 = reference.randn_input((129, 512), fmt, seed=29)
    x = x0.clone()
    out = torch.empty_like(x0)
    sc = torch.empty(_scale_shape((129, 512), fmt), dtype=torch.float32, device="cuda")

    def calls():
        forward(x0, format=fmt)
        forward(x0, format=fmt, return_scales=True)
        forward(x0, format=fmt, out=out, scales=sc)
        forward(x, format=fmt, out=x)

    calls()
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        calls()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_no_deterministic_algorithms_alert(fmt: str) -> None:
    x0 = reference.randn_input((17, 96), fmt, seed=37)
    want = reference.qdq(x0, fmt)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y, s = forward(x0, format=fmt, return_scales=True)
            x = x0.clone()
            forward(x, format=fmt, out=x)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    _check(f"{fmt} under deterministic algorithms", want, y, s)
    _check(f"{fmt} in place under deterministic algorithms", want, x, None)


# --------------------------------------------------------------------------- CPU-only: registry, manifest, sources


@pytest.mark.cpu_only
def test_workspace_is_zero() -> None:
    query = pkg.cake_act_qdq_det_train_forward_workspace_bytes
    for fmt in FORMATS:
        for shape in ((0, 512), (16384, 512), (16384, 32, 128), (16384, 8, 4096)):
            assert query(shape, format=fmt) == 0
    with pytest.raises(ValueError, match="format"):
        query((4, 64), format="nvfp4")
    with pytest.raises(ValueError, match="multiple of the group size"):
        query((4, 48), format="mxfp4_e8m0_g32")


@pytest.mark.cpu_only
def test_element_bound() -> None:
    # n_vecs = numel / 8 is a signed 32-bit kernel argument, and the kernel forms n_vecs + 255 in 32 bits.
    vectors = loader.MAX_VECTORS
    assert vectors == 2**31 - 256 and pkg.MAX_ELEMENTS == 8 * vectors
    query = pkg.cake_act_qdq_det_train_forward_workspace_bytes
    assert query((vectors // 4, 32), format="mxfp8_e8m0_g32") == 0  # exactly MAX_ELEMENTS
    with pytest.raises(ValueError, match="at most"):
        query((vectors // 4 + 1, 32), format="mxfp8_e8m0_g32")


@pytest.mark.cpu_only
def test_constants() -> None:
    assert tuple(pkg.FORMATS) == FORMATS
    assert dict(pkg.GROUP_SIZES) == reference.GROUP
    assert tuple(pkg.ARCHES) == ("sm_100a", "sm_103a", "sm_107a")
    assert tuple(pkg.SUPPORTED_COMPUTE_CAPABILITIES) == ((10, 0), (10, 3), (10, 7))
    assert pkg.ALIGNMENT_BYTES == 16


@pytest.mark.cpu_only
def test_routes_select_one_program_per_format() -> None:
    for capability, arch in ((10, 0), "sm_100a"), ((10, 3), "sm_103a"), ((10, 7), "sm_107a"):
        assert loader.arch_for_capability(capability) == arch
    assert loader.arch_for_capability((9, 0)) is None
    assert loader.arch_for_capability((12, 0)) is None
    for arch in loader.ARCHES:
        chosen = set()
        for fmt in FORMATS:
            for scale_out in (False, True):
                name = loader.selected_module(arch, fmt, scale_out=scale_out)
                record = loader.MODULES[name]
                assert record["format"] == fmt and record["scale_out"] is scale_out, name
                chosen.add(name)
        assert chosen == set(loader.MODULES), arch
    with pytest.raises(ValueError, match="format"):
        loader.selected_module("sm_100a", "nvfp4", scale_out=False)
    with pytest.raises(ValueError, match="route"):
        loader.route_modules("sm_90a", "fp4_e4m3_g16")


@pytest.mark.cpu_only
def test_launch_grid() -> None:
    record = loader.MODULES["mxfp8_e8m0_g32"]
    assert loader.launch_grid(record, n_vecs=1, sm_count=148) == (1, 1, 1)
    assert loader.launch_grid(record, n_vecs=256, sm_count=148) == (1, 1, 1)
    assert loader.launch_grid(record, n_vecs=257, sm_count=148) == (2, 1, 1)
    assert loader.launch_grid(record, n_vecs=8 * 148 * 256, sm_count=148) == (8 * 148, 1, 1)
    assert loader.launch_grid(record, n_vecs=loader.MAX_VECTORS, sm_count=148) == (8 * 148, 1, 1)


@pytest.mark.cpu_only
def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and its host plan; the manifest (schema
    ``cake-build/1``) has exactly one entry per (record, architecture), each with the kernel and binding sources
    under ``csrc/`` and their digests, compiled with IEEE division and without fast math."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "cake_pplx._backends.cake_act_qdq_det_train"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    assert set(loader.ROUTES) == set(loader.ARCHES)
    for arch, routes in loader.ROUTES.items():
        assert set(routes) == set(loader.KINDS), arch
        for kind, names in routes.items():
            assert names == [kind], (arch, kind)
    referenced = set()
    for name, record in loader.MODULES.items():
        assert name == loader.route_name(record["format"], scale_out=record["scale_out"]), name
        assert re.fullmatch(r"cake_act_qdq_[0-9a-f]{20}", record["library"]), name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        assert tuple(record["arches"]) == loader.ARCHES, name
        assert record["ffi_entry"] == "run", name
        assert record["grid_rule"] == {
            "cta_vectors": 256,
            "ctas_per_sm": 8,
            "kind": "grid_stride",
            "vector_elements": loader.VECTOR,
        }, name
        assert record["group"] == loader.GROUP_SIZES[record["format"]], name
        launch = record["launch"]
        assert tuple(launch["block"]) == (256, 1, 1), name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert launch["dynamic_smem_bytes"] == 0, name
        params = [key for kind, key in record["arg_plan"] if kind != "grid"]
        axes = [key for kind, key in record["arg_plan"] if kind == "grid"]
        assert params == (["x", "y", "scales", "n_vecs"] if record["scale_out"] else ["x", "y", "n_vecs"]), name
        assert axes == list(loader.GRID_AXES), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            assert library in libraries, f"{library} is not in build.json"
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                data = (package / source).read_bytes()
                assert hashlib.sha256(data).hexdigest() == entry["sha256"][source], source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            # The three-input max of PTX ISA 8.8 needs CUDA 12.9; sm_107a ships from CUDA 13.5 (docs/API.md).
            assert entry["min_cuda"] == MIN_CUDA[arch], library
            flags = entry["nvcc_flags"]
            assert "--use_fast_math" not in flags, library
            assert "--ftz=true" not in flags, library
            assert "--prec-div=true" in flags, library
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()


# FP64 in CUDA C++ or inline PTX: the type and its vector forms, the double intrinsics and conversions, f64 ops.
_FP64 = re.compile(r"\bdouble\d?\b|__double|2double\b|\.f64\b")
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


@pytest.mark.cpu_only
def test_kernel_sources_hold_no_fp64() -> None:
    """No CUDA source of the package uses FP64 (comments aside): every step of the rule is exact in FP32."""
    package = Path(cake_build.__file__).parent
    sources = sorted((package / "csrc").glob("*.cu"))
    assert sources, "the package has no CUDA sources"
    found = []
    for path in sources:
        code = _COMMENTS.sub("", path.read_text(encoding="utf-8"))
        for number, line in enumerate(code.splitlines(), start=1):
            if _FP64.search(line):
                found.append(f"{path.name}:{number}: {line.strip()[:120]}")
    assert not found, "FP64 in the CUDA sources:\n" + "\n".join(found[:20])
