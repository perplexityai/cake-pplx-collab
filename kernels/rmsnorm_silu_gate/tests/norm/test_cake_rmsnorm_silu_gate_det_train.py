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

Tests of the Cake SiLU-gated RMSNorm training forward / backward
(:mod:`cake_pplx.cake_rmsnorm_silu_gate_det_train`): ``RMSNorm(x) * w * SiLU(z)``.
Every training output (``y``, ``dx``, ``dz``, the BF16 and the FP32 weight
gradient, and both accumulated onto a starting value) is judged against the
PyTorch BF16 naive implementation of the operation, each against a float64
oracle of the BF16 inputs, over 32 random inputs of a case
(:mod:`tests.cake.test_helpers.silu_reference`, the ``TIER`` lines of the
terminal summary): over dynamic token and head counts, irregular tails,
strided layouts, near-zero variance, saturated gates, finite gate inputs at
the edges of the forward's gate, the row counts at the edges of the grids,
the claimed backward far past one wave, the grids of other SM counts and
gradient accumulation.  With infinite and NaN gates, ``y`` and the BF16 and
FP32 weight gradients have the oracle's classes, and ``dx`` and ``dz`` are
finite in the rows whose gates are all finite (where the oracle is finite);
with an infinite upstream gradient the weight gradients have the oracle's
infinities; no test uses a NaN upstream gradient.  ``y``, ``dx``,
``dz`` and the BF16, FP32 and accumulated FP32 weight gradients are the same
bits at every call with the same inputs: at the issue's shapes, small and odd
shapes, the row counts at the edges of the forward's grids (one CTA a row
block below ``persistent_from_rows_per_sm`` rows per SM of the device's
architecture, whole waves from there; on flat and on strided rows) and of
the backward's grid on the device, on every layout (on an
architecture that routes a forward of packed rows, on both forward routes), on
two streams at once, in CUDA Graph replays and under
``torch.use_deterministic_algorithms``.  Packed rows and a single token run
the tiny forward up to ``FWD_TINY_ROWS_PER_SM`` rows per SM, outside the
rows of ``FWD_TINY_GAP_ROWS`` (``forward_module``), and its ``y`` is the bits
of the forwards it replaces
(the routes without it, and ``fwd``) on every data kind, rows of zero ``x``
with a zero and a subnormal epsilon, partial last blocks, one token of 1 to
128 heads, the rows around its cut on this device and on other SM counts,
every layout of ``make``, a side stream, CUDA Graph replays, an ``out`` of
either layout, autograd and checkpointed steps and
``torch.use_deterministic_algorithms``; other layouts, a broadcast and rows
past the cut never run it.  A backward of packed rows runs its
role's long module from the long module's ``min_rows_per_sm`` rows per SM, and
strided rows, a single token and fewer rows run the role's module; the two
give the same ``dx``, ``dz`` and weight-gradient bits on the same inputs and
grid.  A backward with ``compute_grad_weight=False`` runs the program
``nodw_module`` names (strided up to ``NODW_STRIDED_ROWS_PER_SM`` rows per
SM: on packed rows (a single token of head stride 128 among them where
``PACKED_SMALL_ROWS_PER_SM`` has a cut, on an architecture without a strided
cut only up to that cut) its build of packed rows only up to that cut and its
128-thread build past it, on other layouts the
128-thread build up to ``NODW_ANY_LAYOUT_ROWS_PER_SM`` rows per SM and the
256-thread one past it; claimed past the strided cut, strided at every row
count where no claimed program is routed, the long one where packed rows run
the long module, and the long module itself where no long program without
the weight gradient is routed) and gives the ``dx`` and ``dz`` bits of the
call that computes the weight gradient, and the same bits at every call, a
row and an SM's rows around each cut, on packed and strided rows, on every
data kind and at forced grids; the build of packed rows only gives the bits
of the 128-thread build on the same rows and never runs rows that are not
packed; every strided program makes no stream workspace and takes no
reserve workspace, eagerly and in CUDA Graph capture, and leaves a caller
workspace as it is.  A backward that computes the weight gradient on up to
256 rows runs its role's program of one CTA (up to 16 rows) or of one
cluster of 16 CTAs (where ``device_runs_clusters`` holds; ``dw_module``): on
every layout of ``make``, three data kinds and infinite and NaN terms, both
weight-gradient dtypes, fresh and accumulated (onto infinite and NaN starts
too), at every call, on the grids of other SM counts and where the device
check fails (the role's module past 16 rows), it gives the ``dx``, ``dz``
and weight-gradient bits of the role's module on the same calls, its
weight gradient has the FP64 reference's NaN and infinite columns, and it
makes no stream workspace, takes no reserve workspace and leaves a caller
workspace as it is, eagerly and in CUDA Graph capture.  The
tests never compare this package with the non-deterministic one and hold no
output to an error bound of its own.  The backward's workspace is zero
after every call (``compute_grad_weight=False`` and accumulation included),
serves every shape and both weight-gradient
dtypes, is zeroed by ``cake_rmsnorm_silu_gate_det_train_reset_workspace``, and
the non-deterministic package's workspace (``uint8``) and unfit buffers (FP32
ones, FP16, FP64) are refused; a caller workspace's checked buffers serve its
later calls, are checked and made again once its storage was freed in place
and the tensor rebound, and go with the tensor; a call without a workspace uses its stream's and, during
CUDA Graph capture, one of the device's reserve (or one the graph zero-fills).
Without a device, the registry's records and grids are checked at several SM
counts.

This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
``cake_pplx``.  The ``cake_pplx.cake_rmsnorm_silu_gate_det_train`` facade is
covered by
``test_cake_rmsnorm_silu_gate_det_train_api.py``.
"""

import gc
import inspect
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.silu_reference import Draw, judge, silu_draw

pkg = cake_package("cake_rmsnorm_silu_gate_det_train")
cake_build = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_build")
loader = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_jit")
cake_rmsnorm_silu_gate_det = pkg.cake_rmsnorm_silu_gate_det
cake_rmsnorm_silu_gate_det_train_backward = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward
)
cake_rmsnorm_silu_gate_det_train_backward_workspace = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward_workspace
)
cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
)
cake_rmsnorm_silu_gate_det_train_forward = pkg.cake_rmsnorm_silu_gate_det_train_forward
cake_rmsnorm_silu_gate_det_train_reset_workspace = (
    pkg.cake_rmsnorm_silu_gate_det_train_reset_workspace
)
is_cake_rmsnorm_silu_gate_det_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_det_train_supported
)
forward = cake_rmsnorm_silu_gate_det_train_forward
backward = cake_rmsnorm_silu_gate_det_train_backward
workspace_for = cake_rmsnorm_silu_gate_det_train_backward_workspace
workspace_bytes = cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes
reset_workspace = cake_rmsnorm_silu_gate_det_train_reset_workspace

D = 128
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_det_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


def fp64_reference(dy, x, z, w, eps):
    """An independent FP64 forward and closed-form backward from the BF16 inputs:
    ``(y, dx, dz, dw)`` in FP64 (unrounded)."""
    xd, zd, wd, gd = x.double(), z.double(), w.double(), dy.double()
    rstd = 1.0 / torch.sqrt(xd.square().mean(dim=-1, keepdim=True) + eps)
    n = xd * rstd
    s = torch.sigmoid(zd)
    silu = zd * s
    y = n * wd * silu
    a = gd * silu
    dz = gd * n * wd * s * (1.0 + zd * (1.0 - s))
    dn = a * wd
    dx = rstd * dn - xd * rstd**3 * (dn * xd).mean(dim=-1, keepdim=True)
    dw = (a * n).reshape(-1, D).sum(dim=0)
    return y, dx, dz, dw


def make(T, H, *, layout="contiguous", data="normal", seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=g).bfloat16()

    if layout == "contiguous":
        x, z, dy = randn(T, H, D), randn(T, H, D), randn(T, H, D)
    elif layout == "flat":
        x, z, dy = randn(T * H, D), randn(T * H, D), randn(T * H, D)
    elif layout == "projection":
        # z (and dy) as the gate columns of wider projections, read in place.
        x = randn(T, H, D)
        z = randn(T, 2 * H * D + 16)[:, H * D : 2 * H * D].view(T, H, D)
        dy = randn(T, 2 * H * D + 16)[:, : H * D].view(T, H, D)
    elif layout == "head_major":
        x = randn(H, T, D).permute(1, 0, 2)
        z = randn(H, T, D).permute(1, 0, 2)
        dy = randn(T, H, D)
    elif layout == "misaligned":
        # Storage offsets of 8 elements: materialized once by the host.
        x = randn(T * H * D + 8)[8:].view(T, H, D)
        z = randn(T * H * D + 8)[8:].view(T, H, D)
        dy = randn(T * H * D + 8)[8:].view(T, H, D)
    else:
        raise ValueError(layout)
    if data == "near_zero":
        flat = torch.arange(T * H, device=device).view(T, H)
        xv = x.view(T, H, D) if layout != "flat" else x.view(T, H, D)
        xv[flat % 3 == 0] = 0
        xv[flat % 3 == 1] = 2.0**-60
    elif data == "saturated":
        values = torch.tensor(
            [-1e4, -100.0, -90.0, -30.0, -10.0, 10.0, 30.0, 90.0, 100.0, 1e4],
            device=device,
        ).bfloat16()
        pick = torch.randint(0, 10, z.shape, device=device, generator=g)
        z.copy_(
            torch.where(
                torch.rand(z.shape, device=device, generator=g) < 0.5, values[pick], z
            )
        )
    w = (1.0 + 0.5 * torch.randn(D, device=device, generator=g)).bfloat16()
    return x, z, w, dy


# The judged outputs of ``outputs`` and the quantity of each.
JUDGED = {
    "y": "y",
    "dx": "dx",
    "dz": "dz",
    "dw_bf16": "dw",
    "dw_f32": "dw",
    "dw_accumulated": "dw",
    "dw_bf16_accumulated": "dw",
}


def check_all(record_property, inputs, eps=EPS, *, seed=0, label=""):
    """Judge every training output against the PyTorch BF16 naive
    implementation over the draws of ``judge`` (``inputs(s)`` gives the
    draw's ``(x, z, w, dy)`` from its seed ``s``): ``y``, ``dx``, ``dz``, the
    BF16 and the FP32 weight gradient, the FP32 one accumulated onto 0.25 and
    the BF16 one accumulated onto 0.5.  ``label`` prefixes the output names
    (one test judging several cases)."""

    def run(j, s):
        x, z, w, dy = inputs(s)
        got = outputs(x, z, w, dy, eps=eps)
        y, dx, dz, dw = got[:4]
        assert y.dtype == dx.dtype == dz.dtype == dw.dtype == torch.bfloat16
        assert y.shape == dx.shape == dz.shape == x.shape and dw.shape == (D,)
        assert got[4].dtype == got[5].dtype == torch.float32
        acc16 = torch.full((D,), 0.5, dtype=torch.bfloat16, device=x.device)
        backward(dy, x, z, w, eps, grad_weight=acc16, accumulate_grad_weight=True)
        named = dict(zip(OUTPUTS, got, strict=True), dw_bf16_accumulated=acc16)
        starts = {
            "dw_accumulated": torch.full((D,), 0.25, dtype=torch.float32),
            "dw_bf16_accumulated": torch.full((D,), 0.5, dtype=torch.bfloat16),
        }
        return silu_draw(
            x,
            z,
            w,
            dy,
            eps,
            [{label + name: value for name, value in named.items()}],
            {label + name: kind for name, kind in JUDGED.items()},
            starts={label + name: value for name, value in starts.items()},
        )

    judge(record_property, run, seed=seed)


def assert_same_bits(got, want, what):
    assert got.dtype == want.dtype and got.shape == want.shape, what
    view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.view(view), want.view(view)), what


def assert_zero(workspace, what="the workspace"):
    """Every bit of ``workspace`` zero (an FP32 comparison would pass ``-0``)."""
    torch.cuda.synchronize()
    assert not bool(workspace.view(torch.int32).any()), f"{what} is not zero"


def _device():
    """``(arch, SMs)`` of the current device."""
    index = torch.cuda.current_device()
    return loader.device_arch(index), loader.device_sm_count(index)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The module names the registry launches, in order."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        launched.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


def outputs(x, z, w, dy, *, workspace=None, eps=EPS):
    """The forward, then the backward with a BF16, an FP32 and an FP32 weight
    gradient accumulated onto 0.25: ``(y, dx, dz, dw_bf16, dw_f32, dw_acc)``."""
    y = forward(x, z, w, eps)
    dx, dz, dw = backward(dy, x, z, w, eps, workspace=workspace)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    backward(dy, x, z, w, eps, grad_weight=f32, workspace=workspace)
    acc = torch.full((D,), 0.25, dtype=torch.float32, device=x.device)
    backward(
        dy,
        x,
        z,
        w,
        eps,
        grad_weight=acc,
        accumulate_grad_weight=True,
        workspace=workspace,
    )
    return y, dx, dz, dw, f32, acc


OUTPUTS = ("y", "dx", "dz", "dw_bf16", "dw_f32", "dw_accumulated")


def assert_outputs_equal(got, want, what):
    for a, b, name in zip(got, want, OUTPUTS, strict=True):
        assert_same_bits(a, b, f"{what}: {name}")


REPEATS = 5


@pytest.mark.parametrize(
    "T,H",
    [
        (1, 1),
        (1, 48),
        (3, 48),
        (17, 7),
        (255, 2),
        (1031, 48),
        (4097, 16),
        (2, 64),
        (33, 1),
    ],
)
def test_forward_backward_shapes(T, H, record_property):
    check_all(record_property, lambda s: make(T, H, seed=s))


@pytest.mark.parametrize("layout", ["flat", "projection", "head_major", "misaligned"])
def test_layouts(layout, record_property):
    check_all(record_property, lambda s: make(65, 48, layout=layout, seed=s), seed=1)


@pytest.mark.parametrize("layout", ["flat", "projection", "head_major", "misaligned"])
def test_layouts_give_the_contiguous_bits(layout):
    """The strides change no bits: every output of each layout (rows read in
    place with their own strides, or a misaligned input copied once) is the
    bits of the same rows given as contiguous ``[T, H, 128]`` tensors."""
    T, H = 65, 48
    x, z, w, dy = make(T, H, layout=layout, seed=1)
    got = outputs(x, z, w, dy)
    xc, zc, dyc = (t.reshape(T, H, D).contiguous() for t in (x, z, dy))
    want = outputs(xc, zc, w, dyc)
    for a, b, name in zip(got, want, OUTPUTS, strict=True):
        assert_same_bits(a.reshape(b.shape), b, f"{layout} {name}")


@pytest.mark.parametrize("data", ["near_zero", "saturated"])
def test_edge_values(data, record_property):
    check_all(record_property, lambda s: make(37, 48, data=data, seed=s), seed=2)


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_epsilon(eps, record_property):
    check_all(record_property, lambda s: make(19, 48, seed=s), eps, seed=3)


@pytest.mark.parametrize("T", [30193, 32737, 32388])
def test_issue_shapes(T, record_property):
    check_all(record_property, lambda s: make(T, 48, seed=s), seed=4)


# Finite gate inputs at and past the edges of the forward's gate: below about
# -87.3 its reciprocal flushes to zero and below about -88.7 its exponential
# overflows (the gate is -0 in both), subnormal and zero ones, beside
# saturated and ordinary ones.
GATE_EDGE_VALUES = [
    -1e4,
    -100.0,
    -89.0,
    -88.5,
    -88.0,
    -87.5,
    -87.0,
    -86.5,
    -42.0,
    -41.0,
    -40.0,
    -1e-30,
    -1e-39,
    -0.0,
    0.0,
    1e-39,
    1e-30,
    40.0,
    88.0,
    89.0,
    100.0,
    1e4,
]


def _gate_edges(z, seed):
    """``z`` with gate edge values in a quarter of the elements of two of
    every three rows (so some threads of a row see them and others do not)."""
    g = torch.Generator(device=z.device).manual_seed(seed)
    values = torch.tensor(GATE_EDGE_VALUES, device=z.device).bfloat16()
    pick = torch.randint(
        0, len(GATE_EDGE_VALUES), z.shape, device=z.device, generator=g
    )
    edge = torch.rand(z.shape, device=z.device, generator=g) < 0.25
    rows = torch.arange(z.numel() // D, device=z.device).view(*z.shape[:-1], 1)
    edge &= rows % 3 != 0
    z.copy_(torch.where(edge, values[pick], z))


@pytest.mark.parametrize(
    "T,H", [(1, 48), (13, 7), (37, 48), (129, 48), (260, 7), (1031, 48)]
)
def test_gate_edge_values(T, H, record_property):
    """Gate inputs at the edges of the forward's gate (``GATE_EDGE_VALUES``,
    all finite): every output judged against the naive implementation
    (``check_all``)."""

    def inputs(s):
        x, z, w, dy = make(T, H, seed=s)
        _gate_edges(z, s + 1)
        return x, z, w, dy

    check_all(record_property, inputs, seed=23)


def test_infinite_and_nan_gates():
    """Gates of ``+inf``, ``-inf`` and NaN in six rows, placed so that weight
    gradient column 0 sums to ``+inf``, column 4 to ``-inf`` and columns 1, 2
    (``+inf`` and ``-inf`` terms) and 3 (``silu(-inf)`` is NaN) to NaN: ``y``
    is NaN where the FP64 reference is NaN, the same infinity where it is
    infinite and finite elsewhere; ``dx`` and ``dz`` of the rows with finite
    gates are finite where the reference is; the BF16 and the FP32 weight
    gradient are NaN in the FP64 reference's NaN columns, its infinity in its
    infinite ones and finite in the others; the workspace is zero after each
    call and gives the next call the bits of a fresh workspace."""
    T, H = 257, 48
    x, z, w, dy = make(T, H, seed=25)
    # (token, head, column, gate, upstream gradient); x is 1 there, so the
    # term's sign is the gradient's
    cases = (
        (0, 0, 0, float("inf"), 1.0),
        (1, 0, 1, float("nan"), 1.0),
        (2, 0, 2, float("inf"), 1.0),
        (3, 5, 2, float("inf"), -1.0),
        (4, 0, 3, float("-inf"), 1.0),
        (5, 7, 4, float("inf"), -1.0),
    )
    for t, h, c, gate, grad in cases:
        z[t, h, c] = gate
        dy[t, h, c] = grad
        x[t, h, c] = 1.0
    y64, dx64, dz64, dw64 = fp64_reference(dy, x, z, w, EPS)
    assert dw64[0].item() == float("inf") and dw64[4].item() == float("-inf")
    assert bool(dw64[1:4].isnan().all())

    def assert_classes(got, want, what):
        assert torch.equal(got.isnan(), want.isnan()), f"{what}: NaN elements"
        inf = want.isinf()
        assert torch.equal(got.isinf(), inf), f"{what}: infinite elements"
        assert torch.equal(got[inf].double(), want[inf]), f"{what}: infinities"
        return want.isfinite()

    y = forward(x, z, w, EPS)
    ok = assert_classes(y, y64, "y")
    assert bool(y[ok].isfinite().all()), "y of the finite elements"
    ws = workspace_for(T * H, x.device)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    dx, dz, dw = backward(dy, x, z, w, EPS, workspace=ws)
    assert_zero(ws, "the workspace after the BF16 weight gradient")
    backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
    assert_zero(ws, "the workspace after the FP32 weight gradient")
    for got, what in ((dw, "BF16 dw"), (f32, "FP32 dw")):
        ok = assert_classes(got, dw64, what)
        assert bool(got[ok].isfinite().all()), f"{what}, finite columns"
    rows = torch.isfinite(z.float()).all(-1)
    assert not bool(rows.all())
    for got, want, what in ((dx, dx64, "dx"), (dz, dz64, "dz")):
        finite = want[rows].isfinite()
        assert bool(got[rows][finite].isfinite().all()), f"{what} of the rows with finite gates"
    xf, zf, wf, dyf = make(T, H, seed=26)
    got = backward(dyf, xf, zf, wf, EPS, workspace=ws)
    want = backward(dyf, xf, zf, wf, EPS, workspace=workspace_for(T * H, x.device))
    for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
        assert_same_bits(a, b, f"{name} on the workspace after the non-finite call")
    assert_zero(ws)


@pytest.mark.parametrize("T,H", [(1, 1), (257, 47)], ids=["1x1", "257x47"])
def test_infinite_terms_in_the_last_row_of_a_partial_block(T, H):
    """A row count that is not a multiple of the 16-row block, with infinite
    weight-gradient terms in the last row (a gate of ``+inf`` in column 0, an
    upstream gradient of ``+inf`` in column 1): the rows of the last block past
    the last row read that row and add nothing, so both columns are ``+inf`` in
    the BF16 and the FP32 weight gradient, as in the FP64 reference, and the
    other columns are finite; the workspace is zero after each call."""
    x, z, w, dy = make(T, H, seed=27)
    assert (T * H) % 16
    t, h = T - 1, H - 1
    z[t, h, 0], dy[t, h, 0], x[t, h, 0] = float("inf"), 1.0, 1.0
    z[t, h, 1], dy[t, h, 1], x[t, h, 1] = 1.0, float("inf"), 1.0
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    assert dw64[0].item() == dw64[1].item() == float("inf")
    assert bool(dw64[2:].isfinite().all())
    ws = workspace_for(T * H, x.device)
    f32 = torch.empty(D, dtype=torch.float32, device=x.device)
    dw = backward(dy, x, z, w, EPS, workspace=ws)[2]
    assert_zero(ws, "the workspace after the BF16 weight gradient")
    backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
    assert_zero(ws, "the workspace after the FP32 weight gradient")
    for got, what in ((dw, "BF16 dw"), (f32, "FP32 dw")):
        assert got[0].item() == got[1].item() == float("inf"), what
        assert bool(got[2:].isfinite().all()), what


def test_weight_gradient_is_deterministic():
    x, z, w, dy = make(32737, 48, seed=5)
    first = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)[2]
    for _ in range(5):
        again = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)[2]
        assert torch.equal(first.view(torch.int16), again.view(torch.int16))
    fp32 = torch.empty(D, dtype=torch.float32, device="cuda")
    cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS, grad_weight=fp32)
    for _ in range(3):
        again = torch.empty_like(fp32)
        cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS, grad_weight=again)
        assert torch.equal(fp32.view(torch.int32), again.view(torch.int32))


@pytest.mark.parametrize(
    "T,H",
    [
        pytest.param(T, H, id=f"{T}x{H}")
        for T, H in (
            (30193, 48),
            (32388, 48),
            (32737, 48),
            (1, 1),
            (1, 48),
            (3, 48),
            (17, 7),
            (255, 2),
            (33, 1),
            (2, 64),
            (1031, 48),
            (4097, 16),
        )
    ],
)
def test_outputs_are_the_same_bits_at_every_call(T, H):
    """``REPEATS`` calls of the forward and of the backward (BF16, FP32 and
    accumulated FP32 weight gradient), on the stream's workspace and on a
    caller workspace, give the bits of the first call; the caller workspace is
    zero after them."""
    x, z, w, dy = make(T, H, seed=5)
    want = outputs(x, z, w, dy)
    ws = workspace_for(T * H, x.device)
    for i in range(REPEATS):
        assert_outputs_equal(outputs(x, z, w, dy), want, f"{T}x{H} call {i}")
        assert_outputs_equal(
            outputs(x, z, w, dy, workspace=ws),
            want,
            f"{T}x{H} call {i}, caller workspace",
        )
    assert_zero(ws)


# Row counts at the edges of the grids on this device: the forward's
# ``persistent_from_rows_per_sm[arch] * SMs`` (``arch`` the device's; a row
# less: the most rows of one CTA a block; from there whole waves) and its
# fewest whole waves of CTAs that each walk ``max_blocks_per_cta`` row blocks
# at or above that (a row less: a partial last block; a row more: another
# wave), the backward's one wave of
# ``ctas_per_sm * SMs`` row blocks (a row less: a partial last block; a row and
# a block more, and a row more than two waves: CTAs claim blocks), one row and
# one block and a row.
GRID_EDGES = (
    "one_row",
    "one_block+1",
    "fwd_persistent-1",
    "fwd_persistent",
    "fwd_persistent+1",
    "fwd_waves-1",
    "fwd_waves",
    "fwd_waves+1",
    "bwd_one_wave-1",
    "bwd_one_wave",
    "bwd_one_wave+1",
    "bwd_one_wave+block",
    "bwd_two_waves+1",
)


# The forward's cut of rows per SM of rows the kernel reads as packed, and
# that of other rows where its rule has one (the forward of rows of any
# layout; the forward of packed rows only has the first).
CUT = "persistent_from_rows_per_sm"
STRIDED_CUT = "persistent_from_rows_per_sm_strided"


def _cut(rule, arch, packed=True):
    """The forward's cut of rows per SM on ``arch``: its rule's ``CUT``, or,
    for rows not read as packed, its ``STRIDED_CUT`` where it has one (this
    test's own reading of the rule)."""
    if not packed and STRIDED_CUT in rule:
        return rule[STRIDED_CUT][arch]
    return rule[CUT][arch]


def _persistent_waves_rows(rule, arch, sm_count, packed=True):
    """``(start, rows)``: the forward's first row count of whole waves on
    ``arch`` with ``sm_count`` SMs (its cut, ``_cut``, times the SMs) and the
    fewest rows at or above it that give exactly whole waves of
    ``max_blocks_per_cta`` blocks a CTA."""
    start = _cut(rule, arch, packed) * sm_count
    waves_rows = (
        rule["rows_per_block"]
        * rule["ctas_per_sm"]
        * sm_count
        * rule["max_blocks_per_cta"]
    )
    return start, max(1, -(-start // waves_rows)) * waves_rows


def _edge_rows(edge, packed=True):
    """The rows of ``edge`` on this device, the forward's edges at its cut of
    rows read as packed or (``packed`` False) of other rows."""
    arch, sms = _device()
    fwd = loader.MODULES[loader.route_module(arch, "fwd")]["grid_rule"]
    bwd = loader.MODULES[loader.route_module(arch, "bwd_bf16")]["grid_rule"]
    start, waves = _persistent_waves_rows(fwd, arch, sms, packed)
    bwd_wave = bwd["rows_per_block"] * bwd["ctas_per_sm"] * sms
    return {
        "one_row": 1,
        "one_block+1": bwd["rows_per_block"] + 1,
        "fwd_persistent-1": start - 1,
        "fwd_persistent": start,
        "fwd_persistent+1": start + 1,
        "fwd_waves-1": waves - 1,
        "fwd_waves": waves,
        "fwd_waves+1": waves + 1,
        "bwd_one_wave-1": bwd_wave - 1,
        "bwd_one_wave": bwd_wave,
        "bwd_one_wave+1": bwd_wave + 1,
        "bwd_one_wave+block": bwd_wave + bwd["rows_per_block"],
        "bwd_two_waves+1": 2 * bwd_wave + 1,
    }[edge]


@pytest.mark.parametrize("edge", GRID_EDGES)
def test_grid_edges_on_this_device(edge, record_property):
    """At each of ``GRID_EDGES`` (one flat ``[rows, 128]`` input):
    ``check_all``, and ``REPEATS`` calls on a caller workspace give the bits of
    the first call; the workspace is zero after them."""
    rows = _edge_rows(edge)
    check_all(
        record_property, lambda s: make(rows, 1, layout="flat", seed=s), seed=36
    )
    x, z, w, dy = make(rows, 1, layout="flat", seed=36)
    want = outputs(x, z, w, dy)
    ws = workspace_for(rows, x.device)
    for i in range(REPEATS):
        assert_outputs_equal(
            outputs(x, z, w, dy, workspace=ws), want, f"{edge} ({rows} rows) call {i}"
        )
    assert_zero(ws)


# The forward's ``GRID_EDGES`` (around this device's cut of rows per SM and
# its fewest whole waves from there), here at the cut of rows the kernel does
# not read as packed, on rows of one head each: ``z`` and ``dy`` the columns
# of wider projections (``make``'s ``projection`` layout, read in place), or
# ``x`` and ``z`` one token broadcast over the others (token stride 0).
FORWARD_GRID_EDGES = tuple(edge for edge in GRID_EDGES if edge.startswith("fwd_"))


def _strided_edge_inputs(rows, layout, seed):
    """``(x, z, w, dy)`` of ``rows`` tokens of one head in ``layout``
    (``projection`` or ``broadcast``), which the forward kernel does not read
    as packed rows."""
    if layout == "projection":
        return make(rows, 1, layout="projection", seed=seed)
    assert layout == "broadcast", layout
    x, z, w, dy = make(1, 1, seed=seed)
    return x.expand(rows, 1, D), z.expand(rows, 1, D), w, dy.expand(rows, 1, D)


@pytest.mark.parametrize("layout", ["projection", "broadcast"])
@pytest.mark.parametrize("edge", FORWARD_GRID_EDGES)
def test_forward_grid_edges_on_strided_rows(edge, layout, launches, record_property):
    """At each of ``FORWARD_GRID_EDGES`` of the cut of rows not read as packed
    (``_edge_rows`` with ``packed`` False), on rows that are not packed
    (``_strided_edge_inputs``): the forward launches the architecture's
    ``fwd`` module, its ``y`` is the bits of the same rows given as
    contiguous tensors, and ``REPEATS`` calls give the bits of the first
    call; on the projection layout ``check_all`` judges every output against
    the naive implementation."""
    rows = _edge_rows(edge, packed=False)
    arch, _ = _device()
    x, z, w, dy = _strided_edge_inputs(rows, layout, 37)
    launches.clear()
    y = forward(x, z, w, EPS)
    assert launches == [loader.route_module(arch, "fwd")], (edge, rows, launches)
    want = forward(x.contiguous(), z.contiguous(), w, EPS)
    assert_same_bits(y, want, f"{edge} ({rows} {layout} rows): y")
    for i in range(REPEATS):
        got = forward(x, z, w, EPS)
        assert_same_bits(got, y, f"{edge} ({rows} {layout} rows) call {i}: y")
    if layout == "projection":
        check_all(
            record_property,
            lambda s: make(rows, 1, layout="projection", seed=s),
            seed=37,
        )


def _packed_route():
    """``(arch, the packed forward)`` of this device; skips without one."""
    arch, _ = _device()
    name = loader.ROUTES[arch].get("fwd_packed")
    if name is None:
        pytest.skip(f"{arch} routes no forward of packed rows")
    return arch, name


def _tiny_rows(arch, sm_count):
    """The most rows the tiny forward runs on ``arch`` with ``sm_count`` SMs
    (this test's own reading of the routes and the cut table):
    ``FWD_TINY_ROWS_PER_SM[arch] * sm_count`` where ``arch`` routes
    ``fwd_tiny``, else 0."""
    if "fwd_tiny" not in loader.ROUTES[arch]:
        return 0
    return loader.FWD_TINY_ROWS_PER_SM[arch] * sm_count


def _in_tiny_band(arch, rows, sm_count):
    """Whether the tiny forward runs ``rows`` packed rows (or one token) on
    ``arch`` with ``sm_count`` SMs (this test's own reading of the tables):
    from 1 to ``_tiny_rows`` rows, outside the rows ``(lo, hi]`` of
    ``FWD_TINY_GAP_ROWS[arch]``."""
    lo, hi = loader.FWD_TINY_GAP_ROWS[arch]
    return 1 <= rows <= _tiny_rows(arch, sm_count) and not lo < rows <= hi


def _tiny_route():
    """``(arch, SMs, the tiny forward)`` of this device; skips without one."""
    arch, sms = _device()
    if _tiny_rows(arch, sms) < 1:
        pytest.skip(f"{arch} runs no tiny forward")
    return arch, sms, loader.ROUTES[arch]["fwd_tiny"]


def _plans_without_tiny():
    """``_FWD_PLANS`` with no tiny forward on any architecture (each plan's
    packed and ``fwd`` modules kept, no gap): the routes without the tiny
    forward."""
    return {
        arch: (None, 0, *plan[2:4], 0, 0) for arch, plan in loader._FWD_PLANS.items()
    }


def _forward_launch(arch, rows, sm_count, layout, tiny=True):
    """The forward module a call of ``rows`` rows launches on ``arch`` with
    ``sm_count`` SMs (this test's own reading of the routes), ``layout`` the
    strides the kernel reads ``x`` and ``z`` with: ``"packed"`` (head strides
    128, token strides ``H * 128``), ``"one_token"`` (head stride 128, token
    stride 0) or ``"other"``.  The tiny forward within ``_in_tiny_band`` for
    packed rows and one token (where ``tiny``: with ``False``, the routes
    without it), else ``fwd_packed`` for packed rows below ``PACKED_ROWS``
    where routed, else ``fwd``."""
    routes = loader.ROUTES[arch]
    if tiny and layout != "other" and _in_tiny_band(arch, rows, sm_count):
        return routes["fwd_tiny"]
    if layout == "packed" and rows < loader.PACKED_ROWS:
        return routes.get("fwd_packed", routes["fwd"])
    return routes["fwd"]


def _tiny_edge_rows(edge):
    """Rows around the tiny forward's cut on this device (``cut-1``, ``cut``
    and ``cut+1``: ``_tiny_rows`` less one, it and one more)."""
    arch, sms = _device()
    last = _tiny_rows(arch, sms)
    return {"cut-1": last - 1, "cut": last, "cut+1": last + 1}[edge]


# Contiguous shapes of both forward routes: the issue's, small and odd ones.
ROUTE_SHAPES = ((30193, 48), (32737, 48), (2, 64), (3, 48), (17, 7), (255, 2))
ROUTE_SHAPES += ((33, 1), (1031, 48), (4097, 16), (17, 1), (2, 16))
ROUTE_SHAPES += ((1, 17), (1, 32))
ROUTE_DATA = ("normal", "near_zero", "saturated", "gate_edges", "non_finite")


def _route_inputs(shape, data):
    """``(x, z, w, tokens)`` of one ``ROUTE_SHAPES`` shape (``TxH``, contiguous)
    or one ``GRID_EDGES`` row count (``edge:<name>``) or row count around the
    tiny forward's cut (``tiny:<edge>``, ``_tiny_edge_rows``), both flat
    ``[rows, 128]``, with ``data``: ``make``'s, gate edge values, infinite and
    NaN gates and infinite ``x`` in every fifth row, or ``x`` zero in every
    third row (``zero_rows``)."""
    if shape.startswith("edge:"):
        T, H, layout = _edge_rows(shape[len("edge:") :]), 1, "flat"
    elif shape.startswith("tiny:"):
        T, H, layout = _tiny_edge_rows(shape[len("tiny:") :]), 1, "flat"
    else:
        T, H = (int(v) for v in shape.split("x"))
        layout = "contiguous"
    kind = data if data in ("near_zero", "saturated") else "normal"
    x, z, w, _ = make(T, H, layout=layout, data=kind, seed=41)
    if data == "gate_edges":
        _gate_edges(z, 42)
    elif data == "non_finite":
        zr, xr = z.view(-1, D), x.view(-1, D)
        zr[::5, 7] = float("inf")
        zr[1::5, 9] = float("-inf")
        zr[2::5, 11] = float("nan")
        xr[3::5, 13] = float("inf")
    elif data == "zero_rows":
        x.view(-1, D)[::3] = 0
    return x, z, w, T


@pytest.mark.parametrize("data", ROUTE_DATA)
@pytest.mark.parametrize(
    "shape",
    [*(f"{T}x{H}" for T, H in ROUTE_SHAPES), *(f"edge:{e}" for e in GRID_EDGES)],
)
def test_forward_routes_give_the_same_bits(shape, data, launches, monkeypatch):
    """On an architecture that routes a forward of packed rows: ``REPEATS``
    forwards of contiguous inputs run the module ``_forward_launch`` names
    (the tiny forward up to its rows where routed, past them the forward of
    packed rows, a single token ``fwd``); with the tiny forward taken away
    (``_FWD_PLANS`` patched without it) a forward runs the forward of packed
    rows (a single token ``fwd``); all give the bits of the ``fwd`` program on
    the same inputs (``forward_module`` patched to ``fwd``), finite or not."""
    arch, packed = _packed_route()
    _, sms = _device()
    x, z, w, tokens = _route_inputs(shape, data)
    rows = x.numel() // D
    layout = "packed" if tokens > 1 else "one_token"
    strided = loader.ROUTES[arch]["fwd"]
    launches.clear()
    got = [forward(x, z, w, EPS) for _ in range(REPEATS)]
    name = _forward_launch(arch, rows, sms, layout)
    assert launches == [name] * REPEATS, launches
    monkeypatch.setattr(loader, "_FWD_PLANS", _plans_without_tiny())
    launches.clear()
    got.append(forward(x, z, w, EPS))
    assert launches == [packed if tokens > 1 else strided], launches
    monkeypatch.setattr(
        loader,
        "forward_module",
        lambda arch, rows, heads, sm_count, *strides: loader.module_for_rows(
            arch, "fwd", rows, sm_count
        ),
    )
    launches.clear()
    want = forward(x, z, w, EPS)
    assert launches == [strided], launches
    for i, y in enumerate(got):
        assert_same_bits(y, want, f"{shape} {data} call {i}")


# The layouts of ``make`` whose rows the kernel reads as packed rows at any
# token count (misaligned ones once copied); at one token every layout of
# ``make`` has head stride 128, so its rows are packed too.
PACKED_LAYOUTS = ("contiguous", "flat", "misaligned")


@pytest.mark.parametrize(
    "layout,T,route",
    [
        ("contiguous", 65, "fwd_packed"),
        ("flat", 65, "fwd_packed"),
        ("misaligned", 65, "fwd_packed"),
        ("contiguous", 1, "fwd"),
        ("projection", 65, "fwd"),
        ("head_major", 65, "fwd"),
        ("broadcast", 65, "fwd"),
        ("contiguous", "past", "fwd_packed"),
        ("flat", "past", "fwd_packed"),
        ("projection", "past", "fwd"),
        ("broadcast", "past", "fwd"),
    ],
)
def test_forward_route_follows_the_layout(layout, T, route, launches):
    """On an architecture that routes a forward of packed rows, inputs the
    kernel reads as packed rows (contiguous, flat, and misaligned ones once
    copied) run it, and a single token (token stride 0), a gate sliced from a
    projection, head-major rows and a token-stride-0 broadcast run ``fwd``;
    but packed rows and a single token run the tiny forward up to its rows
    (``_tiny_rows``; ``T`` "past": the fewest tokens, at least 2, past
    them).  Each gives the bits of its rows as contiguous tensors."""
    arch, _ = _packed_route()
    _, sms = _device()
    H = 48
    if T == "past":
        T = max(2, _tiny_rows(arch, sms) // H + 1)
    if layout == "broadcast":
        x, z, w, _ = make(1, H, seed=43)
        x, z = x.expand(T, H, D), z.expand(T, H, D)
    else:
        x, z, w, _ = make(T, H, layout=layout, seed=43)
    if (layout in PACKED_LAYOUTS or T == 1) and _in_tiny_band(arch, T * H, sms):
        route = "fwd_tiny"
    launches.clear()
    y = forward(x, z, w, EPS)
    assert launches == [loader.ROUTES[arch][route]], (layout, T, launches)
    xc, zc = (t.reshape(T, H, D).contiguous() for t in (x, z))
    assert_same_bits(y.reshape(T, H, D), forward(xc, zc, w, EPS), layout)


def test_forward_module_runs_packed_rows_only():
    """``forward_module`` (host only), on 148 SMs past the tiny forward's
    rows (``_tiny_rows``): an architecture's ``fwd_packed`` program for head
    strides of 128 and token strides of ``H * 128`` on both ``x`` and ``z``
    and fewer than ``PACKED_ROWS`` rows; ``fwd`` for any other strides, a
    single token (token stride 0), a token-stride-0 broadcast and from
    ``PACKED_ROWS`` rows, and always on an architecture without one.  Up to
    the tiny forward's rows, where routed, packed rows and a single token run
    it, and a broadcast and other strides ``fwd``; in its gap
    (``FWD_TINY_GAP_ROWS``) packed rows and a single token run as past it."""
    assert loader.PACKED_ROLES == ("fwd_packed",)
    assert loader.PACKED_ROWS == 2**31 // D
    H, last, sms = 48, loader.PACKED_ROWS - 1, 148
    row = H * D
    for arch in loader.ARCHES:
        fwd = loader.ROUTES[arch]["fwd"]
        packed = loader.ROUTES[arch].get("fwd_packed", fwd)
        tiny_rows = _tiny_rows(arch, sms)
        # the fewest tokens of H heads, and of one head, past the tiny rows
        T, n = max(65, tiny_rows // H + 1), max(65, tiny_rows + 1)
        cases = [
            (T * H, H, (row, D, row, D), packed),
            (n, 1, (D, D, D, D), packed),
            (last, 1, (D, D, D, D), packed),
            (loader.PACKED_ROWS, 1, (D, D, D, D), fwd),
            (2**31 - 1, 1, (D, D, D, D), fwd),
            (n, n, (0, D, 0, D), fwd),
            (T * H, H, (0, D, 0, D), fwd),
            (T * H, H, (row, D, 2 * row + 16, D), fwd),
            (T * H, H, (2 * row + 16, D, row, D), fwd),
            (T * H, H, (D, T * D, D, T * D), fwd),
            (T * H, H, (row, D, D, T * D), fwd),
        ]
        if tiny_rows:
            tiny = loader.ROUTES[arch]["fwd_tiny"]
            cases += [
                (H, H, (0, D, 0, D), tiny),
                (tiny_rows, 1, (D, D, D, D), tiny),
                (tiny_rows, tiny_rows, (0, D, 0, D), tiny),
                (2 * H, H, (row, D, row, D), tiny),
                (2 * H, H, (0, D, 0, D), fwd),
                (2 * H, H, (row, D, 2 * row + 16, D), fwd),
                (2 * H, H, (D, 2 * D, D, 2 * D), fwd),
            ]
            # the gap's rows (lo, hi] run the routes without the tiny forward
            # (packed rows the forward of packed rows, one token fwd), and
            # its edges lo and hi + 1 the tiny forward
            lo, hi = loader.FWD_TINY_GAP_ROWS[arch]
            if hi > lo:
                for n in (lo + 1, hi):
                    cases += [(n, 1, (D, D, D, D), packed), (n, n, (0, D, 0, D), fwd)]
                for n in (lo, hi + 1):
                    if 1 <= n <= tiny_rows:
                        cases += [
                            (n, 1, (D, D, D, D), tiny),
                            (n, n, (0, D, 0, D), tiny),
                        ]
        for rows, heads, strides, want in cases:
            got = loader.forward_module(arch, rows, heads, sms, *strides)
            assert got == want, (arch, rows, heads, strides)


# Shapes of the tiny forward's bit tests, as ``TxH`` (contiguous) or rows
# around its cut on this device (``tiny:<edge>``, flat): one token of 1, 16,
# 48 and 128 heads, packed rows of a few tokens with partial last blocks of
# 2, 4, 8 and 16 rows (all of them within the tiny forward's rows on every
# product at any cut), and the cut's edges.
TINY_SHAPES = (
    "1x1",
    "1x16",
    "1x48",
    "1x128",
    "3x5",
    "17x7",
    "2x64",
    "3x48",
    "33x1",
    "tiny:cut-1",
    "tiny:cut",
    "tiny:cut+1",
)
# The data kinds and epsilons of the tiny forward's bit tests: every
# ``ROUTE_DATA`` kind at ``EPS``; rows of zero ``x`` with an epsilon of 0
# and a subnormal one; near-zero rows with the subnormal one; and 1e-3.
TINY_DATA = (
    *((data, EPS) for data in ROUTE_DATA),
    ("zero_rows", 0.0),
    ("zero_rows", 2.0**-140),
    ("near_zero", 2.0**-140),
    ("normal", 1e-3),
)


def _fwd_only(arch, rows, heads, sm_count, *strides):
    """``forward_module`` patched to ``fwd`` (``module_for_rows``)."""
    return loader.module_for_rows(arch, "fwd", rows, sm_count)


@pytest.mark.parametrize(
    "data,eps", TINY_DATA, ids=[f"{data}-{eps:g}" for data, eps in TINY_DATA]
)
@pytest.mark.parametrize("shape", TINY_SHAPES)
def test_tiny_forward_gives_the_bits_of_the_forwards_it_replaces(
    shape, data, eps, launches, monkeypatch
):
    """On an architecture that routes the tiny forward: ``REPEATS`` forwards
    of ``TINY_SHAPES`` with ``TINY_DATA`` run the module ``_forward_launch``
    names (the tiny forward up to its rows, a single token too; past them the
    forward of packed rows where routed, else ``fwd``) and give the bits of
    the forward of the routes without the tiny forward (``_FWD_PLANS``
    patched without it: ``fwd_packed`` where routed, a single token
    ``fwd``) and of the ``fwd`` program (``forward_module`` patched to
    ``fwd``) on the same inputs, finite or not."""
    arch, sms, _ = _tiny_route()
    x, z, w, tokens = _route_inputs(shape, data)
    rows = x.numel() // D
    layout = "packed" if tokens > 1 else "one_token"
    launches.clear()
    got = [forward(x, z, w, eps) for _ in range(REPEATS)]
    assert launches == [_forward_launch(arch, rows, sms, layout)] * REPEATS, launches
    monkeypatch.setattr(loader, "_FWD_PLANS", _plans_without_tiny())
    launches.clear()
    got.append(forward(x, z, w, eps))
    name = _forward_launch(arch, rows, sms, layout, tiny=False)
    assert launches == [name], launches
    monkeypatch.setattr(loader, "forward_module", _fwd_only)
    launches.clear()
    want = forward(x, z, w, eps)
    assert launches == [loader.ROUTES[arch]["fwd"]], launches
    for i, y in enumerate(got):
        assert_same_bits(y, want, f"{shape} {data} eps={eps:g} call {i}")


# Layouts of 3 tokens and of one token of 48 heads beside the tiny forward:
# ``(layout, T, runs the tiny forward)``.  Contiguous, flat and misaligned
# rows (copied once) are packed; a gate sliced from a projection, head-major
# rows and a token-stride-0 broadcast are not, but for one token, whose head
# stride is 128 in every one of them.
TINY_LAYOUTS = (
    ("contiguous", 3, True),
    ("flat", 3, True),
    ("misaligned", 3, True),
    ("projection", 3, False),
    ("head_major", 3, False),
    ("broadcast", 3, False),
    ("contiguous", 1, True),
    ("projection", 1, True),
    ("head_major", 1, True),
    ("broadcast", 1, True),
)


@pytest.mark.parametrize(
    "layout,T,tiny",
    TINY_LAYOUTS,
    ids=[f"{layout}-{T}" for layout, T, _ in TINY_LAYOUTS],
)
def test_tiny_forward_follows_the_layout(layout, T, tiny, launches, monkeypatch):
    """``TINY_LAYOUTS`` within the tiny forward's rows: the packed ones run it,
    the others ``fwd``, and each gives the bits of the ``fwd`` program on its
    rows as contiguous tensors (``forward_module`` patched to ``fwd``)."""
    arch, sms, name = _tiny_route()
    H = 48
    if not _in_tiny_band(arch, T * H, sms):
        pytest.skip(f"{T}x{H} is outside the tiny forward's rows on {sms} SMs")
    if layout == "broadcast":
        x, z, w, _ = make(1, H, seed=43)
        x, z = x.expand(T, H, D), z.expand(T, H, D)
    else:
        x, z, w, _ = make(T, H, layout=layout, seed=43)
    launches.clear()
    y = forward(x, z, w, EPS)
    want_name = name if tiny else loader.ROUTES[arch]["fwd"]
    assert launches == [want_name], (layout, T, launches)
    xc, zc = (t.reshape(T, H, D).contiguous() for t in (x, z))
    monkeypatch.setattr(loader, "forward_module", _fwd_only)
    assert_same_bits(y.reshape(T, H, D), forward(xc, zc, w, EPS), layout)


def _tiny_shapes(arch, sms, shapes):
    """The ``(T, H)`` of ``shapes`` within the tiny forward's rows
    (``_in_tiny_band``); skips where none is."""
    inside = [(T, H) for T, H in shapes if _in_tiny_band(arch, T * H, sms)]
    if not inside:
        pytest.skip(f"no shape within the tiny forward's rows on {sms} SMs")
    return inside


def test_tiny_forward_on_a_side_stream_into_outputs_of_any_layout(launches):
    """At 3 x 48 rows and one token of 48 heads, on a side stream with
    ``torch.cuda.set_sync_debug_mode("error")``: the tiny forward into a new
    tensor, into a contiguous ``out`` (written in place) and into a
    head-major ``out`` of rows padded to ``2 * D`` (not contiguous at one
    token either, so a copy after the launch) runs without a host sync and
    gives the bits of the same call on the current stream."""
    arch, sms, name = _tiny_route()
    for T, H in _tiny_shapes(arch, sms, ((3, 48), (1, 48))):
        x, z, w, _ = make(T, H, seed=45)
        want = forward(x, z, w, EPS)
        out = torch.empty_like(x)
        head_major = torch.empty(H, T, 2 * D, dtype=x.dtype, device=x.device)
        head_major = head_major[..., :D].permute(1, 0, 2)
        assert not head_major.is_contiguous(), (T, H)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        launches.clear()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.cuda.stream(stream):
                y = forward(x, z, w, EPS)
                a = forward(x, z, w, EPS, out=out)
                b = forward(x, z, w, EPS, out=head_major)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.current_stream().wait_stream(stream)
        assert launches == [name] * 3, launches
        assert a is out and b is head_major
        for got, what in ((y, "y"), (out, "out"), (head_major, "head-major out")):
            assert_same_bits(got.contiguous(), want, f"{T}x{H} {what}")


def test_tiny_forward_cuda_graph_capture(launches):
    """The tiny forward of 3 x 48 rows into a contiguous and into a
    head-major ``out``, captured once and replayed three times on new inputs
    copied into the captured tensors: each replay gives the bits of eager
    calls on those inputs."""
    arch, sms, name = _tiny_route()
    ((T, H),) = _tiny_shapes(arch, sms, ((3, 48),))
    x, z, w, _ = make(T, H, seed=46)
    xs, zs = x.clone(), z.clone()
    y = torch.empty_like(x)
    head_major = torch.empty(H, T, D, dtype=x.dtype, device=x.device)
    head_major = head_major.permute(1, 0, 2)

    def step():
        forward(xs, zs, w, EPS, out=y)
        forward(xs, zs, w, EPS, out=head_major)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the module before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    launches.clear()
    with torch.cuda.graph(graph):
        step()
    assert launches == [name, name], launches
    for k in range(3):
        xn, zn, _, _ = make(T, H, seed=47 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        y.zero_()
        head_major.zero_()
        graph.replay()
        torch.cuda.synchronize()
        want = forward(xn, zn, w, EPS)
        assert_same_bits(y, want, f"replay {k} y")
        assert_same_bits(head_major.contiguous(), want, f"replay {k} head-major y")


@pytest.mark.parametrize("T,H", [(1, 48), (3, 48)])
def test_tiny_forward_under_autograd_and_checkpoint(T, H, launches):
    """Within the tiny forward's rows, an autograd step gives the bits of the
    functional calls' ``y``, ``dx``, ``dz`` and ``dw``, and so does a
    non-reentrant checkpointed step; the autograd step launches the tiny
    forward and the functional backward's module, the checkpointed one the
    tiny forward twice (the forward and its recomputation) and that module."""
    arch, sms, name = _tiny_route()
    _tiny_shapes(arch, sms, ((T, H),))
    x, z, w, dy = make(T, H, seed=48)
    launches.clear()
    y = forward(x, z, w, EPS)
    dx, dz, dw = backward(dy, x, z, w, EPS)
    assert launches[0] == name and len(launches) == 2, launches
    bwd = launches[1]
    for checkpoint in (False, True):
        launches.clear()
        got = _step(x, z, w, dy, checkpoint=checkpoint)
        want = [name] * (2 if checkpoint else 1) + [bwd]
        assert launches == want, (checkpoint, launches)
        names = ("y", "dx", "dz", "dw")
        for a, b, what in zip(got, (y, dx, dz, dw), names, strict=True):
            assert_same_bits(a, b, f"checkpoint={checkpoint} {what}")


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_tiny_forward_neither_raises_nor_warns_under_deterministic_algorithms(
    warn_only, launches
):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with warnings as errors, within the tiny forward's rows
    (3 x 48 and one token of 48 heads): the forward, an autograd step and a
    non-reentrant checkpointed step raise nothing, warn nothing and give the
    bits of the same calls with the setting off; every forward launches the
    tiny forward."""
    arch, sms, name = _tiny_route()
    shapes = _tiny_shapes(arch, sms, ((3, 48), (1, 48)))
    cases = [make(T, H, seed=49) for T, H in shapes]

    def run():
        got = []
        for x, z, w, dy in cases:
            got.append(forward(x, z, w, EPS))
            got += _step(x, z, w, dy)
            got += _step(x, z, w, dy, checkpoint=True)
        return got

    want = run()
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    launches.clear()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = run()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    forwards = [n for n in launches if loader.MODULES[n]["role"].startswith("fwd")]
    assert forwards == [name] * (4 * len(cases)), launches
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_bits(a, b, f"warn_only={warn_only} output {i}")


# The rows per SM from which each architecture's backward runs the long module
# of its role on packed rows (the long records' ``min_rows_per_sm``).
LONG_MIN_ROWS_PER_SM = {"sm_100a": 2560, "sm_103a": 2560, "sm_107a": 1536}
# The roles of ``_long_calls``' backward calls, in order.
LONG_CALL_ROLES = ("bwd_bf16", "bwd_bf16", "bwd_bf16", "bwd_f32", "bwd_f32")


def _long_calls(dy, x, z, w, ws):
    """The backward calls of the long-module tests, of the roles
    ``LONG_CALL_ROLES``: a BF16 weight gradient, one accumulated onto 0.5 (on
    ``ws``), ``compute_grad_weight=False``, an FP32 one (on ``ws``) and one
    accumulated onto 0.25.  Their ``dx``, ``dz`` and weight gradients."""
    device = x.device
    bf16 = torch.full((D,), 0.5, dtype=torch.bfloat16, device=device)
    f32 = torch.empty(D, dtype=torch.float32, device=device)
    acc = torch.full((D,), 0.25, dtype=torch.float32, device=device)
    got = []
    for kwargs in (
        {},
        {"grad_weight": bf16, "accumulate_grad_weight": True, "workspace": ws},
        {"compute_grad_weight": False},
        {"grad_weight": f32, "workspace": ws},
        {"grad_weight": acc, "accumulate_grad_weight": True},
    ):
        dx, dz, dw = backward(dy, x, z, w, EPS, **kwargs)
        got += [dx, dz] if dw is None else [dx, dz, dw]
    return got


def _route_to(suffix):
    """A ``backward_module`` that names the module of the call's role
    (``suffix`` ``""``) or of its long role (``LONG``) whatever the call."""

    def module(arch, role, *_strides):
        return loader.ROUTES[arch][role + suffix]

    return module


def _keep_module(arch, name, rows, clusters=True):
    """A ``dw_module`` that keeps the module of every call (no program of one
    CTA or of one cluster)."""
    return name


def _long_call_modules(arch, suffix, rows, sm_count, packed):
    """The modules ``_long_calls``' calls launch when ``backward_module``
    names the module of their role plus ``suffix`` for ``rows`` rows on
    ``sm_count`` SMs: that module, and for ``compute_grad_weight=False`` the
    program ``nodw_module`` names for it (``packed``: the kernel reads the
    inputs as packed rows)."""
    names = [loader.ROUTES[arch][role + suffix] for role in LONG_CALL_ROLES]
    names[2] = loader.nodw_module(arch, names[2], rows, sm_count, packed=packed)
    return names


# Layouts around this device's first row count of the long modules: flat
# ``[rows, 128]`` rows a row below, at and a row above it; ``[T, 48, 128]``
# rows at the fewest tokens that reach it (contiguous a token fewer too,
# misaligned ones copied by the host, a gate and ``grad_out`` sliced from
# projections, head-major ``x`` and ``z``, ``grad_out`` alone sliced from a
# projection, a token-stride-0 broadcast of ``x`` and ``z``); a single token
# of that many heads.
LONG_ROUTE_LAYOUTS = (
    "flat-1",
    "flat",
    "flat+1",
    "contiguous-1",
    "contiguous",
    "misaligned",
    "projection",
    "head_major",
    "dy_projection",
    "broadcast",
    "single_token",
)
# The layouts the kernel reads as packed rows, and those of them at or above
# that row count.
LONG_PACKED = ("flat-1", "flat", "flat+1", "contiguous-1", "contiguous", "misaligned")
LONG_ROUTED = ("flat", "flat+1", "contiguous", "misaligned")
# The layouts a call without the weight gradient passes as packed rows: those
# and, where ``_one_token_packed`` holds, a single token (its heads of stride
# 128; the long module's choice still takes its token stride as 0).
NODW_PACKED = (*LONG_PACKED, "single_token")


@pytest.mark.parametrize("layout", LONG_ROUTE_LAYOUTS)
def test_backward_route_follows_the_layout_and_rows(layout, launches):
    """On this device a backward runs its role's long module when the kernel
    reads ``grad_out``, ``x`` and ``z`` as packed rows (``LONG_ROUTED``) from
    ``min_rows_per_sm * SMs`` rows, and its role's module a row or a token
    fewer, on strided, head-major and broadcast inputs and for a single
    token, for a BF16 and an FP32 weight gradient, accumulated or not, and
    for ``compute_grad_weight=False`` the program ``nodw_module`` names for
    that module (told packed rows for ``NODW_PACKED``: a single token of head
    stride 128 too where ``_one_token_packed``), whose ``dx`` and ``dz`` are
    the bits of the first call's (a BF16 weight gradient) on every layout;
    the workspace is zero after the calls, and the long modules give the bits
    of the role's modules on the same calls."""
    arch, sms = _device()
    start = LONG_MIN_ROWS_PER_SM[arch] * sms
    H = 48
    tokens = -(-start // H)
    if layout.startswith("flat"):
        rows = start + {"flat-1": -1, "flat": 0, "flat+1": 1}[layout]
        x, z, w, dy = make(rows, 1, layout="flat", seed=44)
    elif layout == "single_token":
        x, z, w, dy = make(1, start, seed=44)
    elif layout == "broadcast":
        x, z, w, _ = make(1, H, seed=44)
        x, z = x.expand(tokens, H, D), z.expand(tokens, H, D)
        dy = make(tokens, H, seed=45)[3]
    elif layout == "dy_projection":
        x, z, w, _ = make(tokens, H, seed=44)
        dy = make(tokens, H, layout="projection", seed=45)[3]
    else:
        T = tokens - 1 if layout == "contiguous-1" else tokens
        x, z, w, dy = make(T, H, layout=layout.removesuffix("-1"), seed=44)
    long = layout in LONG_ROUTED
    rows = x.numel() // D
    assert (rows >= start) == (long or layout not in ("flat-1", "contiguous-1"))
    ws = workspace_for(rows, x.device)
    launches.clear()
    got = _long_calls(dy, x, z, w, ws)
    suffix = loader.LONG if long else ""
    packed = layout in NODW_PACKED and (
        layout != "single_token" or _one_token_packed(arch, rows, sms)
    )
    want = _long_call_modules(arch, suffix, rows, sms, packed)
    assert launches == want, (layout, launches)
    assert_zero(ws)
    # _long_calls' third call (compute_grad_weight=False, outputs 6 and 7)
    # against its first (a BF16 weight gradient, outputs 0 and 1)
    for i, what in ((6, "dx"), (7, "dz")):
        assert_same_bits(got[i], got[i - 6], f"{layout} ({rows} rows): nodw {what}")
    if not long:
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "backward_module", _route_to(""))
        launches.clear()
        want = _long_calls(dy, x, z, w, ws)
        assert launches == _long_call_modules(arch, "", rows, sms, packed), launches
    assert_zero(ws)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_bits(a, b, f"{layout} ({rows} rows): output {i}")


# Packed shapes on which each long module is compared with its role's module:
# small and odd ones, the issue's (blocks far past the prefetch share of the
# CTAs), and flat rows around this device's first row count of the long
# modules.  Up to ``LONG_BIT_GRID_ROWS`` rows also on the grids of
# ``LONG_BIT_SM_COUNTS`` SMs, whose CTAs claim many blocks each.
LONG_BIT_SHAPES = ("1x1", "33x1", "17x7", "2049x33", "4097x33", "32737x48")
LONG_BIT_SHAPES += ("long-1", "long", "long+1")
LONG_BIT_SM_COUNTS = (1, 16, 66)
LONG_BIT_GRID_ROWS = 4097 * 33


@pytest.mark.parametrize("shape", LONG_BIT_SHAPES)
def test_long_modules_give_the_bits_of_their_roles(shape, launches):
    """Each backward role's long module (forced through ``backward_module``)
    gives the bits of the role's module on the same packed inputs and grid:
    ``dx``, ``dz``, the BF16 and FP32 weight gradients, accumulated or not,
    and ``compute_grad_weight=False`` (the programs ``nodw_module`` names for
    the two modules), on this device's grid and, up to
    ``LONG_BIT_GRID_ROWS`` rows, on the grids of ``LONG_BIT_SM_COUNTS`` SMs
    (``device_sm_count`` patched), with ``dw_module`` patched to keep the
    module (its programs of one CTA and of one cluster have their own test,
    ``test_small_row_programs_give_the_bits_of_their_modules``); the
    workspace is zero after the calls."""
    arch, sms = _device()
    if shape.startswith("long"):
        start = LONG_MIN_ROWS_PER_SM[arch] * sms
        rows = start + {"long-1": -1, "long": 0, "long+1": 1}[shape]
        x, z, w, dy = make(rows, 1, layout="flat", seed=46)
        packed = True
    else:
        T, H = (int(v) for v in shape.split("x"))
        rows = T * H
        x, z, w, dy = make(T, H, seed=46)
        # packed rows; a single token's heads of stride 128 too, which the
        # call without the weight gradient passes as packed rows where
        # _one_token_packed holds
        packed = T > 1 or _one_token_packed(arch, rows, sms)
    ws = workspace_for(rows, x.device)
    counts = [sms, *LONG_BIT_SM_COUNTS] if rows <= LONG_BIT_GRID_ROWS else [sms]
    for sm_count in counts:
        runs = []
        for suffix in ("", loader.LONG):
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(pkg, "backward_module", _route_to(suffix))
                patch.setattr(loader, "dw_module", _keep_module)
                patch.setattr(
                    loader, "device_sm_count", lambda index, count=sm_count: count
                )
                launches.clear()
                runs.append(_long_calls(dy, x, z, w, ws))
                want = _long_call_modules(arch, suffix, rows, sms, packed)
                assert launches == want, (suffix, launches)
            assert_zero(ws)
        what = f"{shape} on the grids of {sm_count} SMs"
        for i, (a, b) in enumerate(zip(runs[1], runs[0], strict=True)):
            assert_same_bits(a, b, f"{what}: output {i}")


def test_claimed_backward_meets_the_naive_tier(record_property):
    """65536 x 48 rows, many times one wave of the backward's row blocks on
    any device, so its CTAs claim chunks of blocks over several rounds:
    ``check_all``; the FP32 and the accumulated FP32 weight gradient the same
    bits at a second call; the workspace zero after every call."""
    arch, sms = _device()
    T, H = 65536, 48
    rule = loader.MODULES[loader.route_module(arch, "bwd_bf16")]["grid_rule"]
    blocks = -(-T * H // rule["rows_per_block"])
    assert blocks >= 16 * rule["ctas_per_sm"] * sms, (blocks, sms)
    check_all(record_property, lambda s: make(T, H, seed=s), seed=37)
    x, z, w, dy = make(T, H, seed=37)
    ws = workspace_for(T * H, x.device)
    runs = []
    for _ in range(2):
        f32 = torch.empty(D, dtype=torch.float32, device=x.device)
        backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
        assert_zero(ws)
        acc = torch.full((D,), 0.25, dtype=torch.float32, device=x.device)
        backward(
            dy, x, z, w, EPS, grad_weight=acc, accumulate_grad_weight=True, workspace=ws
        )
        assert_zero(ws)
        runs.append((f32, acc))
    for a, b, name in zip(runs[1], runs[0], ("dw_f32", "dw_accumulated"), strict=True):
        assert_same_bits(a, b, f"{name} at the second call")


# SM counts whose grids the host computes on this device: small MIG-like
# instances, the products' counts and one larger than any device.
OTHER_SM_COUNTS = (1, 7, 16, 24, 66, 148, 152, 160, 212, 216, 1024)


@pytest.mark.parametrize("sm_count", OTHER_SM_COUNTS)
def test_grids_of_other_sm_counts(sm_count, record_property):
    """The grids computed for ``sm_count`` SMs (``device_sm_count`` patched)
    on this device: ``y``, ``dx`` and ``dz`` are the bits of the device's own
    grids; the BF16, FP32 and accumulated FP32 weight gradients are the same
    bits at a second call; the workspace is zero after every call; and every
    output of each shape on those grids is judged against the naive
    implementation (``check_all``, its outputs named by the shape).  On few
    SMs the larger shapes reach the forward's whole waves."""
    for T, H in ((17, 7), (257, 48), (1031, 48), (4097, 16)):
        x, z, w, dy = make(T, H, seed=38)
        want = outputs(x, z, w, dy)
        ws = workspace_for(T * H, x.device)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loader, "device_sm_count", lambda index: sm_count)
            got = outputs(x, z, w, dy, workspace=ws)
            assert_zero(ws)
            again = outputs(x, z, w, dy, workspace=ws)
            assert_zero(ws)
        what = f"{T}x{H} on the grids of {sm_count} SMs"
        for a, b, name in zip(got[:3], want[:3], OUTPUTS, strict=False):
            assert_same_bits(a, b, f"{what}: {name}")
        assert_outputs_equal(again, got, f"{what}, second call")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loader, "device_sm_count", lambda index: sm_count)
            check_all(
                record_property,
                lambda s, T=T, H=H: make(T, H, seed=s),
                seed=38,
                label=f"{T}x{H} ",
            )


@pytest.mark.parametrize("sm_count", OTHER_SM_COUNTS)
def test_tiny_forward_on_other_sm_counts(sm_count, launches):
    """With the SM count patched to ``sm_count`` (``device_sm_count`` in the
    package and the registry: the host's routing and grids), forwards of flat
    rows a row short of the tiny forward's rows for ``sm_count``
    (``FWD_TINY_ROWS_PER_SM * sm_count``), at them and a row past them, and
    of 3 x 48 rows and one token of 48 heads, run the module
    ``_forward_launch`` names for ``sm_count`` and give the bits of the
    ``fwd`` program on this device (``forward_module`` patched to ``fwd``,
    the device's SM count)."""
    arch, _, _ = _tiny_route()
    last = _tiny_rows(arch, sm_count)
    cases = [(rows, 1, "flat") for rows in (last - 1, last, last + 1) if rows >= 1]
    cases += [(3, 48, "contiguous"), (1, 48, "contiguous")]
    for T, H, layout in cases:
        x, z, w, _ = make(T, H, layout=layout, seed=44)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loader, "forward_module", _fwd_only)
            launches.clear()
            want = forward(x, z, w, EPS)
            assert launches == [loader.ROUTES[arch]["fwd"]], launches
        with pytest.MonkeyPatch.context() as patch:
            for module in (pkg, loader):
                patch.setattr(module, "device_sm_count", lambda index: sm_count)
            launches.clear()
            got = forward(x, z, w, EPS)
        kind = "packed" if T > 1 else "one_token"
        want_name = _forward_launch(arch, T * H, sm_count, kind)
        assert launches == [want_name], (T, H, sm_count, launches)
        assert_same_bits(got, want, f"{T}x{H} on {sm_count} SMs")


def test_weight_gradient_accumulation(record_property):
    """Two backwards of two microbatches accumulated into one FP32 weight
    gradient that starts at 0.25 (and into one BF16 weight gradient that
    starts at 0.5), judged against the naive implementation's BF16 sum of the
    starting value and the two gradients; autograd accumulates ``.grad`` over
    two calls as the sum of the single call's outputs."""

    def run(j, s):
        x, z, w, dy = make(1031, 48, seed=s)
        x2, z2, _, dy2 = make(1031, 48, seed=s + 1)
        got = {}
        for name, dtype, start in (
            ("dw_f32", torch.float32, 0.25),
            ("dw_bf16", torch.bfloat16, 0.5),
        ):
            main = torch.full((D,), start, dtype=dtype, device=x.device)
            for a, b, c, g in ((x, z, w, dy), (x2, z2, w, dy2)):
                cake_rmsnorm_silu_gate_det_train_backward(
                    g, a, b, c, EPS, grad_weight=main, accumulate_grad_weight=True
                )
            got[name] = (main, start)
        first = silu_draw(x, z, w, dy, EPS, [], {"dw": "dw"})
        second = silu_draw(x2, z2, w, dy2, EPS, [], {"dw": "dw"})
        expected = {
            name: start + first.expected["dw"] + second.expected["dw"]
            for name, (_, start) in got.items()
        }

        def naive_values():
            one, two = first.naive()["dw"], second.naive()["dw"]
            return {
                name: (torch.full_like(one, start) + one) + two
                for name, (_, start) in got.items()
            }

        return Draw(
            [{name: main for name, (main, _) in got.items()}], expected, naive_values
        )

    judge(record_property, run, seed=6)
    x, z, w, dy = make(1031, 48, seed=6)
    # autograd accumulation into .grad over two microbatches
    xa = x.clone().requires_grad_(True)
    wa = w.clone().requires_grad_(True)
    za = z.clone().requires_grad_(True)
    cake_rmsnorm_silu_gate_det(xa, za, wa, EPS).backward(dy)
    cake_rmsnorm_silu_gate_det(xa, za, wa, EPS).backward(dy)
    single = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    assert torch.equal(xa.grad, single[0] + single[0])
    assert torch.equal(za.grad, single[1] + single[1])
    assert torch.equal(wa.grad, single[2] + single[2])


def test_autograd_matches_functional_and_checkpoint():
    x, z, w, dy = make(513, 48, seed=8)
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    y = cake_rmsnorm_silu_gate_det(xa, za, wa, EPS)
    y.backward(dy)
    y_ref = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    assert torch.equal(y, y_ref)
    assert (
        torch.equal(xa.grad, dx)
        and torch.equal(za.grad, dz)
        and torch.equal(wa.grad, dw)
    )

    # Saved state: only the inputs, no FP32 temporary.
    saved = []

    def pack(t):
        saved.append((t.dtype, t.numel(), t.data_ptr()))
        return t

    xb, zb, wb = (t.clone().requires_grad_(True) for t in (x, z, w))
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        cake_rmsnorm_silu_gate_det(xb, zb, wb, EPS)
    assert sorted(saved) == sorted(
        [
            (torch.bfloat16, x.numel(), xb.data_ptr()),
            (torch.bfloat16, z.numel(), zb.data_ptr()),
            (torch.bfloat16, D, wb.data_ptr()),
        ]
    )

    # Non-reentrant activation checkpointing replays the forward in the backward.
    xc, zc, wc = (t.clone().requires_grad_(True) for t in (x, z, w))
    yc = torch.utils.checkpoint.checkpoint(
        cake_rmsnorm_silu_gate_det, xc, zc, wc, EPS, use_reentrant=False
    )
    yc.backward(dy)
    assert torch.equal(yc, y_ref)
    assert (
        torch.equal(xc.grad, dx)
        and torch.equal(zc.grad, dz)
        and torch.equal(wc.grad, dw)
    )


def _step(x, z, w, dy, *, checkpoint=False):
    """One autograd step: ``(y, dx, dz, dw)`` (``dw`` BF16), the forward under
    non-reentrant checkpointing when asked."""
    xa, za, wa = (t.clone().requires_grad_(True) for t in (x, z, w))
    if checkpoint:
        y = torch.utils.checkpoint.checkpoint(
            cake_rmsnorm_silu_gate_det, xa, za, wa, EPS, use_reentrant=False
        )
    else:
        y = cake_rmsnorm_silu_gate_det(xa, za, wa, EPS)
    y.backward(dy)
    return y.detach(), xa.grad, za.grad, wa.grad


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_steps_neither_raise_nor_warn_under_deterministic_algorithms(warn_only):
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and
    ``warn_only``) with warnings as errors, the functional calls (on the
    stream's workspace and on a caller workspace, after its reset), an
    autograd step and a non-reentrant checkpointed step raise nothing, warn
    nothing and give the bits of the same calls with the setting off."""
    x, z, w, dy = make(1031, 48, seed=35)
    ws = workspace_for(1031 * 48, x.device)

    def run():
        reset_workspace(ws)
        return (
            *outputs(x, z, w, dy),
            *outputs(x, z, w, dy, workspace=ws),
            *_step(x, z, w, dy),
            *_step(x, z, w, dy, checkpoint=True),
        )

    want = run()
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = run()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_bits(a, b, f"warn_only={warn_only} output {i}")
    assert_zero(ws)


@pytest.mark.parametrize("T,H", [(1, 1), (17, 7), (1031, 48), (32737, 48)])
def test_each_call_launches_one_module_of_its_role(T, H, launches):
    """One module a call: the forward of contiguous inputs runs
    ``forward_module`` of their strides (``_forward_launch``: the tiny forward
    up to its rows where routed, a single token too; else
    ``ROUTES[arch]["fwd_packed"]`` where routed, except for a single token;
    else ``ROUTES[arch]["fwd"]``); a
    backward runs one kernel, ``backward_module`` of its role and strides:
    ``bwd_bf16`` for a BF16 weight gradient, ``bwd_f32`` for an FP32 one,
    accumulating or not, each its long module (``LONG_ROLES``) where these
    packed rows reach the long module's ``min_rows_per_sm`` rows per SM
    (32737 x 48 on every product) and, up to 256 rows, in its place the
    program of one CTA or of one cluster ``dw_module`` names (one CTA for
    1 x 1; one cluster for 17 x 7 where ``device_runs_clusters`` holds), and
    for ``compute_grad_weight=False`` the program ``nodw_module`` names for
    the ``bwd_bf16`` module (packed rows, a single token of head stride 128
    too where ``_one_token_packed``); the reset launches no module."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    for role in loader.ROLES:
        assert loader.module_for_rows(arch, role, T * H, sms) == routes[role]
    tok = H * D if T > 1 else 0
    forward_name = loader.forward_module(arch, T * H, H, sms, tok, D, tok, D)
    layout = "packed" if T > 1 else "one_token"
    assert forward_name == _forward_launch(arch, T * H, sms, layout)
    long = T > 1 and LONG_MIN_ROWS_PER_SM[arch] * sms <= T * H
    clusters = _clusters()
    names = {"fwd": forward_name}
    modules = {}
    for role in ("bwd_bf16", "bwd_f32"):
        modules[role] = routes[role + loader.LONG] if long else routes[role]
        got = loader.backward_module(arch, role, T * H, H, sms, tok, D, tok, D, tok, D)
        assert got == modules[role], (role, got)
        names[role] = loader.dw_module(arch, modules[role], T * H, clusters)
        small = T * H <= 16 or (clusters and T * H <= 256)
        assert (names[role] != modules[role]) == small, (role, names[role])
    names["nodw"] = loader.nodw_module(
        arch,
        modules["bwd_bf16"],
        T * H,
        sms,
        packed=T > 1 or _one_token_packed(arch, T * H, sms),
    )
    x, z, w, dy = make(T, H, seed=39)
    ws = workspace_for(T * H, x.device)
    bf16 = torch.zeros(D, dtype=torch.bfloat16, device=x.device)
    f32 = torch.zeros(D, dtype=torch.float32, device=x.device)
    calls = (
        ("fwd", lambda: forward(x, z, w, EPS)),
        ("bwd_bf16", lambda: backward(dy, x, z, w, EPS)),
        (
            "bwd_bf16",
            lambda: backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=bf16,
                accumulate_grad_weight=True,
                workspace=ws,
            ),
        ),
        (
            "nodw",
            lambda: backward(
                dy, x, z, w, EPS, grad_weight=f32, compute_grad_weight=False
            ),
        ),
        ("bwd_f32", lambda: backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)),
        (
            "bwd_f32",
            lambda: backward(
                dy, x, z, w, EPS, grad_weight=f32, accumulate_grad_weight=True
            ),
        ),
        (None, lambda: reset_workspace(ws)),
    )
    for role, call in calls:
        launches.clear()
        call()
        want = [] if role is None else [names[role]]
        assert launches == want, (role, launches)
    assert_zero(ws)


# The roles of the programs without the weight gradient.
NODW_STRIDED = "bwd_bf16" + loader.NODW + loader.STRIDED
NODW_STRIDED_PACKED = NODW_STRIDED + loader.PACKED
NODW_STRIDED_SMALL = NODW_STRIDED_PACKED + loader.SMALL
NODW_CLAIMED = "bwd_bf16" + loader.NODW
NODW_LONG = "bwd_bf16" + loader.NODW + loader.LONG
# The 128-thread strided builds: the one of any layout and the one of packed
# rows only.
NODW_128 = (NODW_STRIDED_PACKED, NODW_STRIDED_SMALL)
# The grid rule and the CTA threads of each strided program: the 256-thread
# build the BF16 backward's 4 CTAs per SM over 16-row blocks (its CTA
# count), the 128-thread builds 8 CTAs per SM over 8-row blocks.
NODW_STRIDED_GRID_RULES = {
    NODW_STRIDED: {
        "ctas_per_sm": 4,
        "kind": "strided_row_blocks",
        "rows_per_block": 16,
    },
    NODW_STRIDED_PACKED: {
        "ctas_per_sm": 8,
        "kind": "strided_row_blocks",
        "rows_per_block": 8,
    },
    NODW_STRIDED_SMALL: {
        "ctas_per_sm": 8,
        "kind": "strided_row_blocks",
        "rows_per_block": 8,
    },
}
NODW_STRIDED_THREADS = {
    NODW_STRIDED: 256,
    NODW_STRIDED_PACKED: 128,
    NODW_STRIDED_SMALL: 128,
}
# The strided program each layout of the device tests runs: packed rows the
# 128-thread builds, a gate and ``grad_out`` sliced from projections the
# 256-thread one.
STRIDED_LAYOUTS = {
    NODW_STRIDED: "projection",
    NODW_STRIDED_PACKED: "contiguous",
    NODW_STRIDED_SMALL: "contiguous",
}
# The strided programs of the tests' ``program`` parameter.
STRIDED_PROGRAMS = {
    "strided": NODW_STRIDED,
    "strided_packed": NODW_STRIDED_PACKED,
    "strided_small": NODW_STRIDED_SMALL,
}
# The shipped cuts of the small-row programs, rows per SM by architecture and
# role: the build of packed rows only on packed rows
# (``PACKED_SMALL_ROWS_PER_SM``) and the 128-thread build on any layout
# (``NODW_ANY_LAYOUT_ROWS_PER_SM``), each measured on its architecture.
SMALL_ROWS_PER_SM = {
    "sm_100a": {"bwd_bf16": 128},
    "sm_103a": {"bwd_bf16": 128},
    "sm_107a": {"bwd_bf16": 192},
}
ANY_LAYOUT_ROWS_PER_SM = {
    "sm_100a": {"bwd_bf16": 32},
    "sm_103a": {"bwd_bf16": 32},
    "sm_107a": {"bwd_bf16": 96},
}


def _one_token_packed(arch, rows, sm_count):
    """Whether a call without the weight gradient on ``arch`` (``sm_count``
    SMs) passes one token of ``rows`` heads of stride 128 as packed rows
    (token stride ``H * 128``; this test's own reading of the tables): where
    the build of packed rows only has a cut (``PACKED_SMALL_ROWS_PER_SM``), at
    every row count where a strided cut is (a claimed program past it), else
    up to that cut; elsewhere its token stride stays 0 and it runs as other
    layouts do."""
    small = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
    if small <= 0:
        return False
    if loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16") is not None:
        return True
    return rows <= small * sm_count


def _strided_role(arch, rows, sm_count, packed, routes=None):
    """The role of the strided program a call without the weight gradient
    runs for ``rows`` rows on ``arch`` with ``sm_count`` SMs (``routes``:
    ``ROUTES[arch]`` when omitted) where it runs one (this test's own reading
    of the tables): on packed rows the build of packed rows only up to
    ``PACKED_SMALL_ROWS_PER_SM`` rows per SM where routed, else the
    128-thread build where routed; on other layouts the 128-thread build up
    to ``NODW_ANY_LAYOUT_ROWS_PER_SM`` rows per SM where routed; else the
    256-thread one."""
    routes = loader.ROUTES[arch] if routes is None else routes
    small = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0) * sm_count
    any_layout = loader.NODW_ANY_LAYOUT_ROWS_PER_SM[arch].get("bwd_bf16", 0)
    if packed and NODW_STRIDED_SMALL in routes and rows <= small:
        return NODW_STRIDED_SMALL
    if NODW_STRIDED_PACKED in routes and (packed or rows <= any_layout * sm_count):
        return NODW_STRIDED_PACKED
    return NODW_STRIDED


def _nodw_cut(arch, role=NODW_STRIDED):
    """The rows per SM up to which a backward without the weight gradient
    runs the strided program on ``arch``, ``None`` where it runs it at every
    row count (no claimed program routed, no cut); skips where no strided
    program (or no ``role``, one of its 128-thread builds) is routed."""
    for need in {NODW_STRIDED, role}:
        if need not in loader.ROUTES[arch]:
            pytest.skip(f"{arch} routes no {need} program")
    return loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16")


def _claimed_cut(arch):
    """``_nodw_cut`` of an architecture that routes the claimed program too;
    skips where it routes none."""
    cut = _nodw_cut(arch)
    if cut is None:
        pytest.skip(f"{arch} routes no claimed program without the weight gradient")
    return cut


def _routes_without(arch, *roles):
    """``ROUTES[arch]`` without ``roles``."""
    return {r: name for r, name in loader.ROUTES[arch].items() if r not in roles}


def _patch_routes(patch, arch, table):
    """``ROUTES[arch]`` set to ``table`` for the ``patch`` context, with the
    plans :func:`nodw_module` and :func:`forward_module` read for ``arch``
    (``_NODW_PLANS`` and ``_FWD_PLANS``, read once at import) read again
    from it."""
    patch.setitem(loader.ROUTES, arch, table)
    plans = {name: loader._nodw_plan(arch, name) for name in loader.MODULES}
    patch.setitem(loader._NODW_PLANS, arch, plans)
    patch.setitem(loader._FWD_PLANS, arch, loader._fwd_plan(arch))


def _strided_tokens(arch, sms, role, H=48, T=257):
    """Tokens of ``H`` heads, in ``role``'s layout (``STRIDED_LAYOUTS``), whose
    rows a call without the weight gradient runs on ``role``'s strided
    program on this device (``sms`` SMs), below the strided cut: ``T``, but
    for the 128-thread build of packed rows and the 256-thread one at least
    the fewest tokens past the small-row cut of their layout, and for the
    build of packed rows only at most the most tokens up to its cut; skips
    where ``role`` runs no such rows (not routed, or no cut)."""
    cut = _nodw_cut(arch, role)
    if role == NODW_STRIDED_SMALL:
        per_sm = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        T = min(T, per_sm * sms // H)
    elif role == NODW_STRIDED_PACKED:
        per_sm = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        T = max(T, per_sm * sms // H + 1)
    else:
        per_sm = loader.NODW_ANY_LAYOUT_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        T = max(T, per_sm * sms // H + 1)
    packed = STRIDED_LAYOUTS[role] == "contiguous"
    if (
        T < 1
        or (cut is not None and T * H > cut * sms)
        or _strided_role(arch, T * H, sms, packed) != role
    ):
        pytest.skip(f"{arch} runs no {role} program on {H}-head rows here")
    return T


@pytest.mark.parametrize("program", [*STRIDED_PROGRAMS, "claimed"])
def test_compute_grad_weight_false_touches_no_weight_state(program, launches):
    """``compute_grad_weight=False`` returns ``None`` for the weight gradient,
    leaves ``grad_weight`` as it is and gives the ``dx`` and ``dz`` bits of a
    call that computes it.  Rows of 48 heads up to the cut on every product
    that has one (``_strided_tokens``: 257 tokens, or past the small-row
    cut of the layout, or up to the cut of the build of packed rows only)
    run a strided program (the 256-thread one on a gate and ``grad_out``
    sliced from projections, the 128-thread builds on packed rows), which
    makes no stream workspace and leaves a caller workspace zero; a row past
    the cut (where a claimed program is routed) runs the claimed program,
    which takes the arrival ticket and claims row blocks and leaves a caller
    workspace and, without one, the stream's workspace zero.  Autograd with
    a weight that needs no gradient gives the same ``dx`` and ``dz``."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    if program == "claimed":
        T, H = _claimed_cut(arch) * sms + 1, 1
        role, layout, packed = NODW_CLAIMED, "contiguous", True
    else:
        role = STRIDED_PROGRAMS[program]
        H = 48
        T = _strided_tokens(arch, sms, role, H)
        layout = STRIDED_LAYOUTS[role]
        packed = layout == "contiguous"
    want = routes[role]
    got = loader.nodw_module(arch, routes["bwd_bf16"], T * H, sms, packed=packed)
    assert got == want
    x, z, w, dy = make(T, H, layout=layout, seed=12)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(T * H, x.device)
    main = torch.full((D,), 3.0, dtype=torch.float32, device="cuda")
    launches.clear()
    dx, dz, dw = backward(
        dy, x, z, w, EPS, grad_weight=main, compute_grad_weight=False, workspace=ws
    )
    assert launches == [want], launches
    assert dw is None
    assert_same_bits(dx, dx_d, "dx")
    assert_same_bits(dz, dz_d, "dz")
    torch.cuda.synchronize()
    assert torch.equal(main, torch.full_like(main, 3.0))
    assert_zero(ws)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_STREAM_WORKSPACES", {})
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
        assert dw is None
        assert_same_bits(dx, dx_d, "dx without a workspace")
        assert_same_bits(dz, dz_d, "dz without a workspace")
        key = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
        if program != "claimed":
            assert pkg._STREAM_WORKSPACES == {}
        else:
            assert list(pkg._STREAM_WORKSPACES) == [key]
            for view in pkg._STREAM_WORKSPACES[key]:
                assert_zero(view, "the stream's workspace")
    xa, za = (t.clone().requires_grad_(True) for t in (x, z))
    cake_rmsnorm_silu_gate_det(xa, za, w, EPS).backward(dy)
    assert_same_bits(xa.grad, dx_d, "autograd dx")
    assert_same_bits(za.grad, dz_d, "autograd dz")
    assert w.grad is None


@pytest.mark.parametrize("role", list(STRIDED_PROGRAMS.values()))
def test_strided_program_without_the_weight_gradient_uses_no_workspace(
    role, monkeypatch, launches
):
    """Each strided program without the weight gradient (rows of 48 heads,
    ``_strided_tokens``: the 256-thread one on a gate and ``grad_out``
    sliced from projections, the 128-thread builds on packed rows) makes no
    stream workspace and takes no reserve workspace, eagerly and in CUDA
    Graph capture, whose graph holds its one kernel (no fill kernel, memset
    or event wait); a caller workspace filled with 7 is checked and left as
    it is.  Every call and every replay gives the ``dx`` and ``dz`` bits of
    the call that computes the weight gradient."""
    pytest.importorskip("cuda.bindings.runtime")
    arch, sms = _device()
    H = 48
    T = _strided_tokens(arch, sms, role, H)
    strided = loader.ROUTES[arch][role]
    x, z, w, dy = make(T, H, layout=STRIDED_LAYOUTS[role], seed=13)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(T * H, x.device)
    ws.fill_(7.0)
    torch.cuda.synchronize()
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    launches.clear()
    for workspace in (None, ws, None):
        dx, dz, dw = backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=workspace
        )
        assert dw is None
        assert_same_bits(dx, dx_d, f"dx on workspace {workspace is not None}")
        assert_same_bits(dz, dz_d, f"dz on workspace {workspace is not None}")
    assert launches == [strided] * 3, launches
    torch.cuda.synchronize()
    assert torch.equal(ws, torch.full_like(ws, 7.0)), "the caller workspace changed"
    assert pkg._STREAM_WORKSPACES == {} and pkg._RESERVES == {}
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = [
            backward(dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws_)
            for ws_ in (None, ws)
        ]
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 2, types
    assert pkg._STREAM_WORKSPACES == {} and pkg._RESERVES == {}
    assert pkg._CAPTURE_WORKSPACES == {}
    for k in range(3):
        for dx, dz, _ in outs:
            dx.zero_()
            dz.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for i, (dx, dz, dw) in enumerate(outs):
            assert dw is None
            assert_same_bits(dx, dx_d, f"replay {k} call {i} dx")
            assert_same_bits(dz, dz_d, f"replay {k} call {i} dz")
    assert torch.equal(ws, torch.full_like(ws, 7.0)), "the caller workspace changed"


# The SM counts at which the programs without the weight gradient are
# compared with the call that computes it around the strided program's cut
# (``device_sm_count`` patched; ``0`` is the device's own).
NODW_CUT_SM_COUNTS = (0, 16, 66)


@pytest.mark.parametrize("layout", ["flat", "projection"])
@pytest.mark.parametrize("sm_count", NODW_CUT_SM_COUNTS)
def test_programs_without_the_weight_gradient_give_the_bits_at_the_cut(
    sm_count, layout, launches
):
    """At ``cut - 1`` and ``cut`` rows per SM, a row past the cut and ``cut +
    1`` rows per SM (``NODW_STRIDED_ROWS_PER_SM``; ``device_sm_count``
    patched to ``sm_count`` for the choice and the grid), on packed rows and
    on a gate and ``grad_out`` sliced from projections: a call with
    ``compute_grad_weight=False`` runs the strided program up to the cut (on
    packed rows its 128-thread build, on the projections the 256-thread
    one) and the claimed one past it, gives the ``dx`` and ``dz`` bits of the
    BF16 call that computes the weight gradient and leaves a caller
    workspace zero.  An architecture without a cut runs the strided program
    instead at a row per SM fewer than the long modules' first row count and
    at a row fewer than it."""
    arch, sms = _device()
    sm_count = sm_count or sms
    strided_role = NODW_STRIDED_PACKED if layout == "flat" else NODW_STRIDED
    cut = _nodw_cut(arch, strided_role)
    packed = layout == "flat"
    routes = loader.ROUTES[arch]
    if cut is None:
        last = LONG_MIN_ROWS_PER_SM[arch]
        row_counts = ((last - 1) * sm_count, last * sm_count - 1)
    else:
        row_counts = (
            (cut - 1) * sm_count,
            cut * sm_count,
            cut * sm_count + 1,
            (cut + 1) * sm_count,
        )
    with pytest.MonkeyPatch.context() as patch:
        for module in (pkg, loader):
            patch.setattr(module, "device_sm_count", lambda index: sm_count)
        for rows in row_counts:
            what = f"{rows} rows on {sm_count} SMs ({layout})"
            x, z, w, dy = make(rows, 1, layout=layout, seed=14)
            ws = workspace_for(rows, x.device)
            launches.clear()
            dx_d, dz_d, _ = backward(dy, x, z, w, EPS, workspace=ws)
            dx, dz, dw = backward(
                dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
            )
            strided = cut is None or rows <= cut * sm_count
            # past the small-row cuts: the 128-thread build on packed rows,
            # the 256-thread one on the projections
            program = strided_role if strided else NODW_CLAIMED
            assert not strided or (
                _strided_role(arch, rows, sm_count, packed) == strided_role
            ), what
            assert launches == [routes["bwd_bf16"], routes[program]], what
            assert dw is None
            assert_same_bits(dx, dx_d, f"{what}: dx")
            assert_same_bits(dz, dz_d, f"{what}: dz")
            assert_zero(ws, f"the workspace after {what}")


@pytest.mark.parametrize("layout", ["flat", "projection"])
@pytest.mark.parametrize("sm_count", NODW_CUT_SM_COUNTS)
def test_small_row_programs_give_the_bits_at_their_cuts(sm_count, layout, launches):
    """At ``cut - 1`` and ``cut`` rows per SM, a row past the cut and ``cut +
    1`` rows per SM of a small-row cut (``device_sm_count`` patched to
    ``sm_count`` for the choice and the grid): on packed rows (flat;
    ``PACKED_SMALL_ROWS_PER_SM``) a call with ``compute_grad_weight=False``
    runs the build of packed rows only up to the cut and the 128-thread
    build past it; on a gate and ``grad_out`` sliced from projections
    (``NODW_ANY_LAYOUT_ROWS_PER_SM``) the 128-thread build up to the cut and
    the 256-thread one past it.  ``REPEATS`` such calls give the same bits,
    the ``dx`` and ``dz`` bits of the BF16 call that computes the weight
    gradient, and leave a caller workspace zero.  Skips where there is no
    cut."""
    arch, sms = _device()
    sm_count = sm_count or sms
    packed = layout == "flat"
    if packed:
        table = loader.PACKED_SMALL_ROWS_PER_SM
        below, above = NODW_STRIDED_SMALL, NODW_STRIDED_PACKED
    else:
        table = loader.NODW_ANY_LAYOUT_ROWS_PER_SM
        below, above = NODW_STRIDED_PACKED, NODW_STRIDED
    cut = _nodw_cut(arch, below)
    per_sm = table[arch].get("bwd_bf16", 0)
    if per_sm == 0:
        pytest.skip(f"{arch} runs no {below} program on these rows (no cut)")
    routes = loader.ROUTES[arch]
    row_counts = (
        (per_sm - 1) * sm_count,
        per_sm * sm_count,
        per_sm * sm_count + 1,
        (per_sm + 1) * sm_count,
    )
    with pytest.MonkeyPatch.context() as patch:
        for module in (pkg, loader):
            patch.setattr(module, "device_sm_count", lambda index: sm_count)
        for rows in (r for r in row_counts if r >= 1):
            what = f"{rows} rows on {sm_count} SMs ({layout})"
            assert cut is None or rows <= cut * sm_count, what
            role = below if rows <= per_sm * sm_count else above
            assert _strided_role(arch, rows, sm_count, packed) == role, what
            x, z, w, dy = make(rows, 1, layout=layout, seed=17)
            ws = workspace_for(rows, x.device)
            launches.clear()
            dx_d, dz_d, _ = backward(dy, x, z, w, EPS, workspace=ws)
            for i in range(REPEATS):
                dx, dz, dw = backward(
                    dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
                )
                assert dw is None
                assert_same_bits(dx, dx_d, f"{what}, call {i}: dx")
                assert_same_bits(dz, dz_d, f"{what}, call {i}: dz")
            want = [routes["bwd_bf16"], *[routes[role]] * REPEATS]
            assert launches == want, (what, launches)
            assert_zero(ws, f"the workspace after {what}")


# Shapes of packed rows (contiguous ``[T, H, 128]``; one token among them) at
# or below the cut of the build of packed rows only on sm_100a and sm_103a.
SMALL_SHAPES = ((1, 1), (1, 48), (1, 3120), (3, 48), (17, 7), (255, 2), (257, 48))


@pytest.mark.parametrize("data", ROUTE_DATA)
@pytest.mark.parametrize("shape", [f"{T}x{H}" for T, H in SMALL_SHAPES])
def test_program_of_packed_rows_only_gives_the_bits_of_the_strided_builds(
    shape, data, launches
):
    """On small packed rows, finite or not (``_route_inputs``; one token is
    passed as packed rows where ``_one_token_packed`` holds, else as rows of
    other layouts), a call with ``compute_grad_weight=False`` runs the
    program ``_strided_role`` names (the build of packed rows only where its
    cut reaches the rows, else the 128-thread build), ``REPEATS`` such calls
    give the same ``dx`` and ``dz`` bits, and those are the bits of the
    128-thread build and of the 256-thread one on the same rows (``ROUTES``
    patched without the build of packed rows only, then without both
    128-thread builds) and of the BF16 call that computes the weight
    gradient."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    x, z, w, T = _route_inputs(shape, data)
    H = x.shape[1]
    dy = make(T, H, seed=43)[3]
    rows = T * H
    cut = _nodw_cut(arch)
    assert cut is None or rows <= cut * sms
    packed = T > 1 or _one_token_packed(arch, rows, sms)
    role = _strided_role(arch, rows, sms, packed)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    runs = []
    for drop, want in (
        ((), role),
        ((NODW_STRIDED_SMALL,), NODW_STRIDED_PACKED if packed else role),
        (NODW_128, NODW_STRIDED),
    ):
        table = _routes_without(arch, *drop)
        assert _strided_role(arch, rows, sms, packed, table) == want, drop
        with pytest.MonkeyPatch.context() as patch:
            _patch_routes(patch, arch, table)
            launches.clear()
            got = [
                backward(dy, x, z, w, EPS, compute_grad_weight=False)
                for _ in range(REPEATS if not drop else 1)
            ]
            assert launches == [routes[want]] * len(got), (drop, launches)
        runs += [(f"{want} call {i}", dx, dz) for i, (dx, dz, _) in enumerate(got)]
    for what, dx, dz in runs:
        assert_same_bits(dx, dx_d, f"{shape} {data} {what}: dx")
        assert_same_bits(dz, dz_d, f"{shape} {data} {what}: dz")


# Layouts of one token of ``H`` heads: contiguous, a gate and ``grad_out``
# sliced from projections, head-major ``x`` and ``z`` (each with head stride
# 128: the call without the weight gradient passes them as packed rows where
# ``_one_token_packed`` holds) and a gate whose heads lie 256 elements apart
# (not packed rows).
SINGLE_TOKEN_LAYOUTS = ("contiguous", "projection", "head_major", "wide_heads")
SINGLE_TOKEN_HEADS = ("1", "48", "3120", "small", "small+1", "long")


@pytest.mark.parametrize("heads", SINGLE_TOKEN_HEADS)
@pytest.mark.parametrize("layout", SINGLE_TOKEN_LAYOUTS)
def test_single_token_runs_the_programs_of_packed_rows(layout, heads, monkeypatch):
    """One token of ``H`` heads (1, 48, 3120, this device's cut of the build
    of packed rows only and a row past it, and the long modules' first row
    count where no strided cut is; past that cut where no strided cut is,
    one token keeps the token stride 0): a call with
    ``compute_grad_weight=False`` on heads of stride 128 passes the token
    stride ``H * 128`` (packed rows) and runs the program ``_strided_role``
    names for packed rows where ``_one_token_packed`` holds (else, as on the
    gate of wider heads, it passes 0 and runs the program of other layouts);
    the call that computes the weight gradient passes 0 in both (and runs
    the program ``dw_module`` names).  ``dx`` and ``dz`` are the bits of
    that call."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    small = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
    if heads == "long":
        if _nodw_cut(arch) is not None:
            pytest.skip(f"{arch} runs the claimed program past its strided cut")
        H = LONG_MIN_ROWS_PER_SM[arch] * sms
    elif heads == "small":
        if not small:
            pytest.skip(f"{arch} has no cut of the build of packed rows only")
        H = small * sms
    elif heads == "small+1":
        H = small * sms + 1
    else:
        H = int(heads)
    if layout == "wide_heads":
        x, z, w, dy = make(1, H, seed=48)
        z = torch.cat([z, z], dim=-1)[..., :D]
    else:
        x, z, w, dy = make(1, H, layout=layout, seed=48)
    # a single head's stride is passed as 128 whatever the storage
    heads_128 = layout != "wide_heads" or H == 1
    packed = heads_128 and _one_token_packed(arch, H, sms)
    cut = _nodw_cut(arch)
    assert cut is None or H <= cut * sms
    seen = []
    run_module = loader.run_module

    def spy(name, values, *, rows, device_index):
        strides = tuple(values[k] for k in ("dy_tok", "x_tok", "z_tok"))
        seen.append((name, strides, values["z_head"]))
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", spy)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    tok = H * D if packed else 0
    want = routes[_strided_role(arch, H, sms, packed)]
    head = D if heads_128 else 2 * D
    assert seen == [
        (loader.dw_module(arch, routes["bwd_bf16"], H, _clusters()), (0, 0, 0), head),
        (want, (tok, tok, tok), head),
    ], (layout, H, seen)
    assert dw is None
    assert_same_bits(dx, dx_d, f"{layout} 1x{H}: dx")
    assert_same_bits(dz, dz_d, f"{layout} 1x{H}: dz")


# Inputs the backward reads with their own head and token strides (a gate and
# ``grad_out`` sliced from projections, head-major ``x`` and ``z``, ``grad_out``
# alone sliced from a projection, a token-stride-0 broadcast of ``x`` and
# ``z``, one token of many heads) and as packed rows (contiguous).
NODW_INPUT_LAYOUTS = (
    "contiguous",
    "projection",
    "head_major",
    "dy_projection",
    "broadcast",
    "single_token",
)


def _nodw_inputs(layout, tokens, H, seed):
    """``(x, z, w, dy)`` of ``tokens`` tokens of ``H`` heads in the
    ``NODW_INPUT_LAYOUTS`` layout ``layout`` (``single_token``: one token of
    ``tokens * H`` heads)."""
    if layout == "single_token":
        return make(1, tokens * H, seed=seed)
    if layout == "broadcast":
        x, z, w, _ = make(1, H, seed=seed)
        dy = make(tokens, H, seed=seed + 1)[3]
        return x.expand(tokens, H, D), z.expand(tokens, H, D), w, dy
    if layout == "dy_projection":
        x, z, w, _ = make(tokens, H, seed=seed)
        return x, z, w, make(tokens, H, layout="projection", seed=seed + 1)[3]
    return make(tokens, H, layout=layout, seed=seed)


@pytest.mark.parametrize("program", ["strided", "claimed"])
@pytest.mark.parametrize("layout", NODW_INPUT_LAYOUTS)
def test_programs_without_the_weight_gradient_on_strided_inputs(
    layout, program, launches
):
    """On ``NODW_INPUT_LAYOUTS`` inputs of 48 heads, 65 tokens and the fewest
    tokens past the cut of any layout (up to the cut on every product that
    has one) for the strided program and a token past the cut for the
    claimed one (where routed): a call with ``compute_grad_weight=False``
    runs that program (for the strided one the build ``_strided_role``
    names: on the packed rows of contiguous inputs and of a single token
    (where ``_one_token_packed`` holds), the build of packed rows only where
    its cut reaches them, else the 128-thread build; on every other layout
    the 128-thread build where the cut of any layout reaches them, else the
    256-thread one), gives the ``dx`` and ``dz`` bits of the BF16 call that
    computes the weight gradient on the same inputs and leaves a caller
    workspace zero."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    H = 48
    if program == "strided":
        cut = _nodw_cut(arch)
        any_layout = loader.NODW_ANY_LAYOUT_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        # (no second count without that cut: it would be a single token)
        token_counts = sorted({65, any_layout * sms // H + 1} if any_layout else {65})
    else:
        cut = _claimed_cut(arch)
        token_counts = [cut * sms // H + 1]
    for tokens in token_counts:
        x, z, w, dy = _nodw_inputs(layout, tokens, H, seed=46)
        rows = x.numel() // D
        assert cut is None or (rows <= cut * sms) == (program == "strided"), rows
        packed = layout == "contiguous" or (
            layout == "single_token" and _one_token_packed(arch, rows, sms)
        )
        if program == "strided":
            role = _strided_role(arch, rows, sms, packed)
        else:
            role = NODW_CLAIMED
        want = routes[role]
        ws = workspace_for(rows, x.device)
        launches.clear()
        dx_d, dz_d, _ = backward(dy, x, z, w, EPS, workspace=ws)
        dx, dz, dw = backward(
            dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
        )
        what = f"{layout} ({rows} rows, {program}: {role})"
        assert launches == [routes["bwd_bf16"], want], (what, launches)
        assert dw is None
        assert_same_bits(dx, dx_d, f"{what}: dx")
        assert_same_bits(dz, dz_d, f"{what}: dz")
        assert_zero(ws, f"the workspace after {what}")


@pytest.mark.parametrize("ctas", [1, 3, 7])
@pytest.mark.parametrize("program", [*STRIDED_PROGRAMS, "claimed"])
def test_programs_without_the_weight_gradient_at_forced_grids(
    program, ctas, launches
):
    """On grids of ``ctas`` CTAs (``launch_grid`` patched), fewer than the
    row blocks of the rows of 48 heads (``_strided_tokens``; 257 tokens for
    the claimed one), the strided program (on a gate and ``grad_out`` sliced
    from projections: the 256-thread one), its 128-thread build and its
    build of packed rows only (packed rows) and the claimed one (packed
    rows, the strided ones unrouted; where routed) give the ``dx`` and
    ``dz`` bits of the call that computes the weight gradient; the claimed
    one leaves a caller workspace zero after back-to-back calls."""
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    if program == "claimed" and NODW_CLAIMED not in routes:
        pytest.skip(f"{arch} routes no claimed program without the weight gradient")
    H = 48
    if program == "claimed":
        T, layout = 257, "contiguous"
    else:
        role = STRIDED_PROGRAMS[program]
        T, layout = _strided_tokens(arch, sms, role, H), STRIDED_LAYOUTS[role]
    x, z, w, dy = make(T, H, layout=layout, seed=15)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(T * H, x.device)
    with pytest.MonkeyPatch.context() as patch:
        if program == "claimed":
            table = _routes_without(arch, NODW_STRIDED, *NODW_128)
            _patch_routes(patch, arch, table)
            want = routes[NODW_CLAIMED]
        else:
            want = routes[role]
        patch.setattr(loader, "launch_grid", lambda record, **_: (ctas, 1, 1))
        launches.clear()
        for i in range(3):
            dx, dz, dw = backward(
                dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
            )
            assert dw is None
            assert_same_bits(dx, dx_d, f"{program} on {ctas} CTAs, call {i}: dx")
            assert_same_bits(dz, dz_d, f"{program} on {ctas} CTAs, call {i}: dz")
        assert launches == [want] * 3, launches
    assert_zero(ws)


def test_long_rows_without_the_weight_gradient(launches):
    """Packed rows at this device's first row count of the long modules: a
    call with ``compute_grad_weight=False`` runs the long program without the
    weight gradient, and, where none is routed (``ROUTES`` patched), the
    ``bwd_bf16`` long module with ``weight_grad`` 0; both give the ``dx`` and
    ``dz`` bits of the call that computes the weight gradient and leave the
    workspace zero."""
    arch, sms = _device()
    rows = LONG_MIN_ROWS_PER_SM[arch] * sms
    x, z, w, dy = make(rows, 1, layout="flat", seed=16)
    dx_d, dz_d, _ = backward(dy, x, z, w, EPS)
    ws = workspace_for(rows, x.device)
    routes = loader.ROUTES[arch]
    long = routes["bwd_bf16" + loader.LONG]
    for table, want in (
        (dict(routes), routes.get(NODW_LONG, long)),
        (_routes_without(arch, NODW_LONG), long),
    ):
        with pytest.MonkeyPatch.context() as patch:
            _patch_routes(patch, arch, table)
            launches.clear()
            dx, dz, dw = backward(
                dy, x, z, w, EPS, compute_grad_weight=False, workspace=ws
            )
            assert launches == [want], launches
        assert dw is None
        assert_same_bits(dx, dx_d, f"{want}: dx")
        assert_same_bits(dz, dz_d, f"{want}: dz")
        assert_zero(ws, f"the workspace after {want}")


# The CTAs and the most rows of a backward role's programs of one CTA
# (``SOLO``) and of one cluster (``CLUSTER``).
SMALL_DW_RULES = {loader.SOLO: (1, 16), loader.CLUSTER: (16, 256)}
# Shapes of the backward that computes the weight gradient around those
# programs, as ``(T, H)``: 1, 15, 16, 17, 48, 128, 129, 255 and 256 rows, and
# 257 (the role's module, a control).
SMALL_DW_SHAPES = (
    (1, 1),
    (3, 5),
    (1, 16),
    (17, 1),
    (1, 48),
    (2, 64),
    (3, 43),
    (5, 51),
    (16, 16),
    (257, 1),
)
SMALL_DW_LAYOUTS = ("contiguous", "flat", "projection", "head_major", "misaligned")
SMALL_DW_DATA = ("normal", "near_zero", "saturated", "non_finite")
# The roles of ``_dw_calls``' calls, in order, and the names of its outputs.
DW_CALL_ROLES = ("bwd_bf16", "bwd_bf16", "bwd_f32", "bwd_f32")
DW_CALL_OUTPUTS = tuple(
    f"{call} {name}"
    for call in ("BF16", "BF16 accumulated", "FP32", "FP32 accumulated")
    for name in ("dx", "dz", "dw")
)


def _small_suffix(role):
    """The suffix of a role of ``CLUSTER_ROLES``: ``SOLO`` or ``CLUSTER``."""
    return loader.SOLO if role.endswith(loader.SOLO) else loader.CLUSTER


def _clusters():
    """Whether this device runs the cluster programs."""
    return loader.device_runs_clusters(torch.cuda.current_device())


def _small_dw_module(arch, role, rows, clusters):
    """The module a call of backward role ``role`` that computes the weight
    gradient launches on ``rows`` rows (this test's own reading of the
    routes): the role's program of one CTA up to 16 rows, of one cluster up
    to 256 rows where ``clusters`` (the device runs it), else the role's
    module."""
    for suffix, (ctas, most) in SMALL_DW_RULES.items():
        if rows <= most and (ctas == 1 or clusters):
            return loader.ROUTES[arch][role + suffix]
    return loader.ROUTES[arch][role]


def _small_dw_modules(arch, rows, clusters):
    """``_small_dw_module`` of each of ``_dw_calls``' calls."""
    return [_small_dw_module(arch, role, rows, clusters) for role in DW_CALL_ROLES]


# The accumulated weight gradient's start columns of ``non_finite`` data:
# ``+inf``, ``-inf`` and NaN on finite sums (columns 120 to 122), ``-inf`` on
# the ``+inf`` sum of column 0 (NaN) and the ``-inf`` sum of column 36, NaN
# on the ``+inf`` sum of column 45 and ``+inf`` on the NaN sum of column 9.
NON_FINITE_STARTS = (
    (120, float("inf")),
    (121, float("-inf")),
    (122, float("nan")),
    (0, float("-inf")),
    (36, float("-inf")),
    (45, float("nan")),
    (9, float("inf")),
)


def _dw_start(dtype, value, non_finite, device):
    """The accumulated weight gradient's start: ``value`` in every column, with
    ``NON_FINITE_STARTS`` where ``non_finite``."""
    start = torch.full((D,), value, dtype=dtype, device=device)
    if non_finite:
        for column, v in NON_FINITE_STARTS:
            start[column] = v
    return start


def _set_term(t, row, H, column, value):
    """``value`` at ``row`` (``token * H + head``) and ``column`` of a
    ``[T, H, 128]`` or a flat ``[T * H, 128]`` tensor of ``make``, in place
    through any of its views."""
    if t.dim() == 2:
        t[row, column] = value
    else:
        t[row // H, row % H, column] = value


def _small_non_finite(x, z, dy, H):
    """Infinite and NaN weight-gradient terms in ``make``'s tensors of any
    layout, written in place, in columns of different CTAs of a 16-CTA
    cluster (8 columns each): column 0 sums to ``+inf`` (a gate of ``+inf`` in
    the last row), columns 9 (a NaN gate) and 27 (``silu(-inf)`` is NaN) to
    NaN, column 18 to NaN from 2 rows on (a ``+inf`` and a ``-inf`` term, else
    ``+inf``), column 36 to ``-inf``, column 45 to ``+inf`` (an upstream
    gradient of ``+inf``), from 2 rows on column 127 to NaN (a NaN and a
    ``+inf`` term) and, from 4 rows on, column 54 to NaN (an ``x`` of ``+inf``
    in row 1, whose other terms are then 0).  ``x`` is 1 at the gate terms, so
    a term's sign is the gradient's.  The terms sit in rows 0, ``rows // 2``
    and ``rows - 1``.  The other columns stay finite."""
    rows = x.numel() // D
    last, mid = rows - 1, rows // 2
    cases = [
        (last, 0, float("inf"), 1.0),
        (0, 9, float("nan"), 1.0),
        (0, 18, float("inf"), 1.0),
        (mid, 27, float("-inf"), 1.0),
        (mid, 36, float("inf"), -1.0),
        (last, 45, 1.0, float("inf")),
    ]
    if rows >= 2:
        cases.append((last, 18, float("inf"), -1.0))
        cases.append((0, 127, float("nan"), 1.0))
        cases.append((last, 127, float("inf"), 1.0))
    for row, column, gate, grad in cases:
        _set_term(z, row, H, column, gate)
        _set_term(dy, row, H, column, grad)
        _set_term(x, row, H, column, 1.0)
    if rows >= 4:
        _set_term(x, 1, H, 54, float("inf"))


def _dw_calls(dy, x, z, w, workspace=None, non_finite_starts=False):
    """The backward calls that compute the weight gradient, of the roles
    ``DW_CALL_ROLES``: a BF16 weight gradient, one accumulated onto 0.5, an
    FP32 one and one accumulated onto 0.25 (with ``NON_FINITE_STARTS`` where
    ``non_finite_starts``), each on ``workspace``.  Their ``dx``, ``dz`` and
    weight gradients (``DW_CALL_OUTPUTS``)."""
    got = []
    for dtype, start in ((torch.bfloat16, 0.5), (torch.float32, 0.25)):
        fresh = torch.empty(D, dtype=dtype, device=x.device)
        acc = _dw_start(dtype, start, non_finite_starts, x.device)
        for grad_weight, accumulate in ((fresh, False), (acc, True)):
            got += backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=grad_weight,
                accumulate_grad_weight=accumulate,
                workspace=workspace,
            )
    return got


def _assert_dw_calls_equal(got, want, what):
    for a, b, name in zip(got, want, DW_CALL_OUTPUTS, strict=True):
        assert_same_bits(a, b, f"{what}: {name}")


@pytest.mark.parametrize("data", SMALL_DW_DATA)
@pytest.mark.parametrize("layout", SMALL_DW_LAYOUTS)
@pytest.mark.parametrize("shape", [f"{T}x{H}" for T, H in SMALL_DW_SHAPES])
def test_small_row_programs_give_the_bits_of_their_modules(
    shape, layout, data, launches, monkeypatch
):
    """``SMALL_DW_SHAPES`` in every layout of ``make`` (``SMALL_DW_LAYOUTS``)
    with ``SMALL_DW_DATA`` (``non_finite``: ``make``'s normal data with
    ``_small_non_finite``'s terms, accumulated onto ``NON_FINITE_STARTS``):
    ``REPEATS`` rounds of ``_dw_calls`` run the
    programs ``_small_dw_modules`` names (one CTA up to 16 rows, one cluster
    up to 256 rows where ``device_runs_clusters`` holds, else the role's
    module) and give the ``dx``, ``dz`` and weight-gradient bits, fresh and
    accumulated, of the role's module on the same calls (``dw_module``
    patched to keep the module)."""
    arch, _ = _device()
    clusters = _clusters()
    T, H = (int(v) for v in shape.split("x"))
    rows = T * H
    non_finite = data == "non_finite"
    kind = "normal" if non_finite else data
    x, z, w, dy = make(T, H, layout=layout, data=kind, seed=80)
    if non_finite:
        _small_non_finite(x, z, dy, H)
    runs = []
    for i in range(REPEATS):
        launches.clear()
        runs.append(_dw_calls(dy, x, z, w, non_finite_starts=non_finite))
        assert launches == _small_dw_modules(arch, rows, clusters), (i, launches)
    monkeypatch.setattr(loader, "dw_module", _keep_module)
    launches.clear()
    want = _dw_calls(dy, x, z, w, non_finite_starts=non_finite)
    assert launches == [loader.ROUTES[arch][role] for role in DW_CALL_ROLES], launches
    for i, got in enumerate(runs):
        _assert_dw_calls_equal(got, want, f"{shape} {layout} {data} call {i}")


@pytest.mark.parametrize(
    "T,H",
    [(1, 16), (3, 5), (17, 1), (8, 16), (16, 16)],
    ids=["16_rows", "15_rows", "17_rows", "128_rows", "256_rows"],
)
def test_small_row_programs_give_the_classes_of_infinite_and_nan_sums(T, H, launches):
    """``_small_non_finite``'s terms on the rows of the program of one CTA (15
    and 16 rows) and of one cluster (17, 128 and 256 rows, where
    ``device_runs_clusters`` holds), which ``_dw_calls`` launches
    (``_small_dw_modules``): the BF16 and the FP32 weight gradient, fresh and
    accumulated onto ``NON_FINITE_STARTS``, are NaN in the FP64 reference's NaN
    columns, its infinity in its infinite ones and finite in the others."""
    arch, _ = _device()
    x, z, w, dy = make(T, H, seed=82)
    _small_non_finite(x, z, dy, H)
    dw64 = fp64_reference(dy, x, z, w, EPS)[3]
    assert dw64[0].item() == dw64[45].item() == float("inf")
    assert dw64[36].item() == float("-inf")
    assert bool(dw64[[9, 18, 27, 54, 127]].isnan().all())
    finite = torch.ones(D, dtype=torch.bool, device=x.device)
    finite[[0, 9, 18, 27, 36, 45, 54, 127]] = False
    assert bool(dw64[finite].isfinite().all())
    launches.clear()
    got = _dw_calls(dy, x, z, w, non_finite_starts=True)
    assert launches == _small_dw_modules(arch, T * H, _clusters()), launches
    wants = []
    for dtype, start in ((torch.bfloat16, 0.5), (torch.float32, 0.25)):
        begin = _dw_start(dtype, start, True, x.device).double()
        wants += [dw64, begin + dw64]
    assert bool(wants[1][[0, 9, 45, 122]].isnan().all())
    assert wants[1][120].item() == float("inf")
    assert wants[1][121].item() == wants[1][36].item() == float("-inf")
    names = ("BF16", "BF16 accumulated", "FP32", "FP32 accumulated")
    for got_dw, want, what in zip(got[2::3], wants, names, strict=True):
        assert torch.equal(got_dw.isnan(), want.isnan()), f"{what}: NaN columns"
        inf = want.isinf()
        assert torch.equal(got_dw.isinf(), inf), f"{what}: infinite columns"
        assert torch.equal(got_dw[inf].double(), want[inf]), f"{what}: infinities"
        assert bool(got_dw[want.isfinite()].isfinite().all()), f"{what}: finite columns"


@pytest.mark.parametrize("T,H", [(1, 16), (1, 128), (2, 128)])
def test_small_row_programs_meet_the_naive_tier(T, H, record_property):
    """``check_all`` on the rows of the program of one CTA (16) and of one
    cluster (128 and 256, where ``device_runs_clusters`` holds)."""
    check_all(record_property, lambda s: make(T, H, seed=s), seed=81)


@pytest.mark.parametrize("T,H", [(1, 16), (2, 128)], ids=["one_cta", "one_cluster"])
def test_small_row_programs_use_no_workspace(T, H, monkeypatch, launches):
    """The program of one CTA (16 rows) and of one cluster (256 rows, where
    ``device_runs_clusters`` holds) make no stream workspace and take no
    reserve workspace, eagerly and in CUDA Graph capture, whose graph holds
    their kernels only (no fill kernel, memset or event wait); a caller
    workspace filled with 7 is checked and left as it is, and a zero one
    stays zero.  Every call and every replay gives the bits of the same calls
    before the workspace tables were emptied."""
    pytest.importorskip("cuda.bindings.runtime")
    arch, _ = _device()
    clusters = _clusters()
    rows = T * H
    if rows > SMALL_DW_RULES[loader.SOLO][1] and not clusters:
        pytest.skip("this device runs no cluster program")
    x, z, w, dy = make(T, H, seed=82)
    want = _dw_calls(dy, x, z, w)
    filled = workspace_for(rows, x.device).fill_(7.0)
    zero = workspace_for(rows, x.device)
    torch.cuda.synchronize()
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    modules = _small_dw_modules(arch, rows, clusters)
    for i, workspace in enumerate((None, filled, zero, None)):
        launches.clear()
        got = _dw_calls(dy, x, z, w, workspace=workspace)
        assert launches == modules, (i, launches)
        _assert_dw_calls_equal(got, want, f"{rows} rows, call {i}")
    torch.cuda.synchronize()
    assert torch.equal(filled, torch.full_like(filled, 7.0)), "the caller workspace"
    assert_zero(zero, "the zero caller workspace")
    assert pkg._STREAM_WORKSPACES == {} and pkg._RESERVES == {}
    f32s = [torch.empty(D, dtype=torch.float32, device=x.device) for _ in range(2)]
    accs = [torch.empty(D, dtype=torch.float32, device=x.device) for _ in range(2)]
    graph = _kept_graph()
    launches.clear()
    with torch.cuda.graph(graph):
        outs = []
        for ws, f32, acc in zip((None, filled), f32s, accs, strict=True):
            outs.append(backward(dy, x, z, w, EPS, workspace=ws))
            backward(dy, x, z, w, EPS, grad_weight=f32, workspace=ws)
            backward(
                dy,
                x,
                z,
                w,
                EPS,
                grad_weight=acc,
                accumulate_grad_weight=True,
                workspace=ws,
            )
    assert launches == [modules[0], modules[2], modules[3]] * 2, launches
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 6, types
    assert pkg._STREAM_WORKSPACES == {} and pkg._RESERVES == {}
    assert pkg._CAPTURE_WORKSPACES == {}
    for k in range(3):
        for out in outs:
            for t in out:
                t.zero_()
        for f32, acc in zip(f32s, accs, strict=True):
            f32.zero_()
            acc.fill_(0.25)
        graph.replay()
        torch.cuda.synchronize()
        for i, (out, f32, acc) in enumerate(zip(outs, f32s, accs, strict=True)):
            what = f"{rows} rows, replay {k} call {i}"
            # _dw_calls' outputs: 0-2 BF16, 8 FP32 dw, 11 accumulated FP32 dw
            for a, b, name in zip(out, want[:3], ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"{what}: BF16 {name}")
            assert_same_bits(f32, want[8], f"{what}: FP32 dw")
            assert_same_bits(acc, want[11], f"{what}: FP32 accumulated dw")
    assert torch.equal(filled, torch.full_like(filled, 7.0)), "the caller workspace"


@pytest.mark.parametrize("sm_count", OTHER_SM_COUNTS)
def test_small_row_programs_on_other_sm_counts(sm_count, launches):
    """With the SM count patched to ``sm_count`` (``device_sm_count``: the
    host's routing and grids; ``device_runs_clusters`` reads the device
    itself), ``_dw_calls`` on ``SMALL_DW_SHAPES`` up to 256 rows run the
    programs ``_small_dw_modules`` names, whose grids do not depend on the
    SM count, and give the bits of the same calls on this device; from 4 SMs
    the role's module (``dw_module`` patched to keep it), whose grid there
    is one CTA a 16-row block, gives them too."""
    arch, _ = _device()
    clusters = _clusters()
    for T, H in SMALL_DW_SHAPES[:-1]:
        rows = T * H
        x, z, w, dy = make(T, H, seed=83)
        want = _dw_calls(dy, x, z, w)
        with pytest.MonkeyPatch.context() as patch:
            for module in (pkg, loader):
                patch.setattr(module, "device_sm_count", lambda index: sm_count)
            launches.clear()
            got = _dw_calls(dy, x, z, w)
            assert launches == _small_dw_modules(arch, rows, clusters), launches
            _assert_dw_calls_equal(got, want, f"{T}x{H} on {sm_count} SMs")
            if sm_count < 4:
                continue
            patch.setattr(loader, "dw_module", _keep_module)
            launches.clear()
            got = _dw_calls(dy, x, z, w)
            assert launches == [loader.ROUTES[arch][r] for r in DW_CALL_ROLES], launches
            _assert_dw_calls_equal(got, want, f"{T}x{H}, the module on {sm_count} SMs")


def test_devices_without_clusters_run_the_module_past_16_rows(launches):
    """Where ``device_runs_clusters`` fails (patched: a device of fewer than
    16 SMs or a MIG instance), ``_dw_calls`` on ``SMALL_DW_SHAPES`` run the
    program of one CTA up to 16 rows and the role's module past that, with
    the bits of the same calls on this device."""
    arch, _ = _device()
    for T, H in SMALL_DW_SHAPES:
        rows = T * H
        x, z, w, dy = make(T, H, seed=84)
        want = _dw_calls(dy, x, z, w)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(loader, "device_runs_clusters", lambda index: False)
            launches.clear()
            got = _dw_calls(dy, x, z, w)
        assert launches == _small_dw_modules(arch, rows, False), launches
        _assert_dw_calls_equal(got, want, f"{T}x{H} without clusters")


def test_workspace_size_and_layout():
    """The backward's workspace: ``workspace_elements`` FP32 elements (4738,
    18952 bytes) for every backward module (both weight-gradient dtypes and
    their long modules, so one stream workspace serves them all; the
    programs of one CTA and of one cluster have none) and every row
    count, zero-filled, 16-byte aligned; negative rows are a ``ValueError``.
    Its views are the ``uint64`` digit counters (``slots * limbs * 128``), one
    ``uint32`` flag word a column and the two ``uint32`` words of the arrival
    ticket and the claim counter, in that order."""
    device = torch.device("cuda")
    arch, _ = _device()
    names = [
        loader.route_module(arch, role)
        for role in ("bwd_bf16", "bwd_f32", *loader.LONG_ROLES)
    ]
    assert len(set(names)) == 4
    for role in loader.CLUSTER_ROLES:
        assert "workspace" not in loader.MODULES[loader.route_module(arch, role)], role
    elements = loader.workspace_elements(names[0])
    for name in names:
        shape = loader.MODULES[name]["workspace"]
        assert (
            loader.workspace_elements(name)
            == elements
            == 2 * shape["slots"] * shape["limbs"] * D + D + 2
        )
    assert elements == 4738
    for rows in (0, 1, 129 * 48, 10**7):
        ws = workspace_for(rows, device)
        assert ws.dtype == torch.float32 and ws.shape == (elements,), rows
        assert ws.is_contiguous() and ws.data_ptr() % 16 == 0, rows
        assert ws.device.type == "cuda", rows
        assert_zero(ws, f"the workspace for {rows} rows")
        assert workspace_bytes(rows, device) == elements * 4 == 18952, rows
    for entry in (workspace_for, workspace_bytes):
        with pytest.raises(ValueError, match="non-negative"):
            entry(-1, device)
    for name in names:
        shape = loader.MODULES[name]["workspace"]
        acc, flags, done = pkg._workspace_views(ws, name, ws.device)
        assert acc.dtype == torch.uint64
        assert acc.numel() == shape["slots"] * shape["limbs"] * D
        assert flags.dtype == torch.uint32 and flags.numel() == D
        assert done.dtype == torch.uint32 and done.numel() == 2
        assert acc.data_ptr() == ws.data_ptr()
        assert flags.data_ptr() == ws.data_ptr() + 8 * acc.numel()
        assert done.data_ptr() == flags.data_ptr() + 4 * D


def test_workspace_is_zero_after_every_call_and_serves_every_shape():
    """One caller workspace, never reset, over a sequence of shapes, both
    weight-gradient dtypes, accumulation, ``compute_grad_weight=False`` and a
    call with no rows: it is zero after every call and every call gives the
    bits of the same call on the stream's workspace (whose outputs
    ``check_all`` judges)."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    # (T, H, weight-gradient dtype, accumulate, compute the weight gradient)
    cases = (
        (1031, 48, torch.bfloat16, False, True),
        (7, 48, torch.float32, False, True),
        (4097, 16, torch.bfloat16, True, True),
        (257, 48, torch.float32, False, False),
        (1, 1, torch.float32, True, True),
        (0, 48, torch.float32, True, True),
        (30193, 48, torch.bfloat16, False, True),
        (33, 7, torch.float32, True, True),
        (65, 48, torch.bfloat16, False, False),
        (2, 64, torch.bfloat16, True, True),
    )
    for i, (T, H, dtype, accumulate, compute) in enumerate(cases):
        what = f"{T}x{H} {dtype} accumulate={accumulate} compute={compute}"
        x, z, w, dy = make(T, H, seed=20 + i)
        start = 0.5 if accumulate else 0.0

        kwargs = dict(accumulate_grad_weight=accumulate, compute_grad_weight=compute)
        got = backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=torch.full((D,), start, dtype=dtype, device=device),
            workspace=ws,
            **kwargs,
        )
        assert_zero(ws, f"the workspace after {what}")
        want = backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=torch.full((D,), start, dtype=dtype, device=device),
            **kwargs,
        )
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            if b is None:
                assert a is None, (what, name)
            else:
                assert_same_bits(a, b, f"{what}: {name}")
        assert (got[2] is None) == (not compute), what
        if T == 0:
            if compute:
                assert torch.equal(got[2], torch.full_like(got[2], start)), what
            continue
        if compute:
            assert got[2].dtype == dtype


def test_caller_workspace_buffers_follow_the_tensor():
    """A caller workspace is checked and viewed by its first call and its
    buffers serve its later calls; once its storage is freed in place and the
    tensor rebound to a new zero-filled workspace (the allocator may give it
    the old address), the next call checks it and makes new buffers, gives the
    bits of a call on a fresh workspace and leaves the new storage zero.
    Rebound to a buffer one element short, it is refused and keeps no entry;
    an entry goes with its tensor."""
    x, z, w, dy = make(1031, 48, seed=31)
    want = backward(dy, x, z, w, EPS, workspace=workspace_for(1, x.device))
    ws = workspace_for(1, x.device)
    for i in range(2):
        got = backward(dy, x, z, w, EPS, workspace=ws)
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"call {i}: {name}")
    held = pkg._CALLER_WORKSPACES[id(ws)]
    assert held[0]() is ws and held[4].data_ptr() == ws.data_ptr()
    backward(dy, x, z, w, EPS, workspace=ws)
    assert pkg._CALLER_WORKSPACES[id(ws)] is held
    torch.cuda.synchronize()
    ws.untyped_storage().resize_(0)
    ws.set_(workspace_for(1, x.device))
    got = backward(dy, x, z, w, EPS, workspace=ws)
    torch.cuda.synchronize()
    again = pkg._CALLER_WORKSPACES[id(ws)]
    assert again is not held and again[4].data_ptr() == ws.data_ptr()
    for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
        assert_same_bits(a, b, f"rebound: {name}")
    assert_zero(ws, "the rebound workspace")
    del held, again
    elements = ws.numel()
    ws.set_(torch.ones(elements - 1, dtype=torch.float32, device=x.device))
    with pytest.raises(ValueError) as info:
        backward(dy, x, z, w, EPS, workspace=ws)
    assert str(info.value) == unfit(x.device, elements), str(info.value)
    del info  # its traceback holds this frame: no cycle keeping the tensors
    assert id(ws) not in pkg._CALLER_WORKSPACES
    assert bool(ws.eq(1).all())
    ws = workspace_for(1, x.device)
    backward(dy, x, z, w, EPS, workspace=ws)
    key = id(ws)
    assert pkg._CALLER_WORKSPACES[key][0]() is ws
    del ws
    gc.collect()
    assert key not in pkg._CALLER_WORKSPACES

def test_calls_without_workspace_share_one_buffer_per_stream():
    """Calls without a caller workspace (both weight-gradient dtypes) share the
    zero-filled workspace of their (device, stream), left zero; another stream
    gets its own."""
    device = torch.cuda.current_device()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_STREAM_WORKSPACES", {})
        x, z, w, dy = make(65, 7)
        backward(dy, x, z, w, EPS)
        f32 = torch.empty(D, dtype=torch.float32, device=x.device)
        backward(dy, x, z, w, EPS, grad_weight=f32)
        key = (device, torch.cuda.current_stream().cuda_stream)
        assert list(pkg._STREAM_WORKSPACES) == [key]
        for view in pkg._STREAM_WORKSPACES[key]:
            assert_zero(view, "the stream's workspace")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            backward(dy, x, z, w, EPS)
        side.synchronize()
        assert set(pkg._STREAM_WORKSPACES) == {key, (device, side.cuda_stream)}
        for view in pkg._STREAM_WORKSPACES[(device, side.cuda_stream)]:
            assert_zero(view, "the side stream's workspace")


def test_checkpointed_steps_keep_only_the_stream_workspace():
    """A non-reentrant checkpointed forward keeps nothing alive but its output,
    and once the first step has run, every checkpointed forward and backward
    leaves ``torch.cuda.memory_allocated()`` unchanged.  The buffers kept
    across steps are the stream's workspace of the backward without a caller
    workspace (``backward_workspace_bytes`` bytes) and the device's reserve of
    zero-filled workspaces for CUDA Graph captures (one allocation)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_STREAM_WORKSPACES", {})
        patch.setattr(pkg, "_RESERVES", {})
        # Small enough for the caching allocator's small pool, whose blocks are
        # the requested sizes rounded up to 512 bytes.
        x, z, w, dy = make(17, 48, seed=63)
        leaves = [t.clone().requires_grad_(True) for t in (x, z, w)]
        for leaf in leaves:
            leaf.grad = torch.zeros_like(leaf)

        def checkpointed():
            return torch.utils.checkpoint.checkpoint(
                cake_rmsnorm_silu_gate_det, *leaves, EPS, use_reentrant=False
            )

        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        checkpointed().backward(dy)  # loads the modules, allocates the workspace
        torch.cuda.synchronize()
        kept = torch.cuda.memory_allocated() - before
        nbytes = workspace_bytes(1, x.device)
        key = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream)
        assert list(pkg._STREAM_WORKSPACES) == [key]
        acc, _, _ = pkg._STREAM_WORKSPACES[key]
        assert acc.untyped_storage().nbytes() == nbytes
        (block,) = pkg._RESERVES[torch.cuda.current_device()]["blocks"]
        reserved = block.untyped_storage().nbytes()
        # 256-byte aligned workspaces, one after the other
        stride = -(-(nbytes // 4) // 64) * 64
        assert reserved == pkg._RESERVE_SIZE * stride * 4, reserved
        assert nbytes + reserved <= kept < nbytes + reserved + 512, (kept, nbytes)
        base = torch.cuda.memory_allocated()
        for step in range(3):
            y = checkpointed()
            torch.cuda.synchronize()
            retained = (
                torch.cuda.memory_allocated() - base - y.untyped_storage().nbytes()
            )
            assert retained == 0, (step, retained)
            y.backward(dy)
            del y
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == base, step
        assert list(pkg._STREAM_WORKSPACES) == [key]


# The texts of the workspace refusals.
NONDETERMINISTIC_WORKSPACE = (
    "cake_pplx.cake_rmsnorm_silu_gate_nondet_train."
    "cake_rmsnorm_silu_gate_nondet_train_backward_workspace"
)
DETERMINISTIC_WORKSPACE = (
    "cake_pplx.cake_rmsnorm_silu_gate_det_train."
    "cake_rmsnorm_silu_gate_det_train_backward_workspace"
)


def refusal(dtype):
    """The refusal of a workspace of the non-deterministic package."""
    name = {torch.uint8: "uint8"}[dtype]
    return (
        f"workspace is {name}, a workspace of the non-deterministic mode "
        f"({NONDETERMINISTIC_WORKSPACE}); the deterministic backward needs its "
        f"own zero-filled FP32 workspace ({DETERMINISTIC_WORKSPACE})"
    )


def unfit(device, elements):
    """The refusal of another unfit buffer."""
    return (
        "workspace must be a contiguous, 16-byte aligned FP32 tensor on "
        f"{device} with at least {elements} elements, zero before its first use "
        "(cake_rmsnorm_silu_gate_det_train_backward_workspace(rows, device), "
        "or cake_rmsnorm_silu_gate_det_train_reset_workspace(workspace))"
    )


def _unfit_buffers(elements):
    """Buffers of ones the backward and the reset refuse: FP32 ones too small,
    not 16-byte aligned and strided, an FP16 one and an FP64 one."""
    base = torch.ones(2 * elements + 4, dtype=torch.float32, device="cuda")
    return (
        torch.ones(elements - 1, dtype=torch.float32, device="cuda"),
        base[1 : elements + 1],
        base[: 2 * elements : 2],
        torch.ones(elements, dtype=torch.float16, device="cuda"),
        torch.ones(elements, dtype=torch.float64, device="cuda"),
    )


def test_backward_refuses_unfit_workspaces(launches):
    """The backward refuses a ``uint8`` tensor (a workspace of the
    non-deterministic package) naming both modes' workspaces, with rows and
    with none, for both weight-gradient dtypes, and keeps ``grad_weight``;
    another unfit buffer (too small, not 16-byte aligned, strided, FP16, FP64,
    on the host) gets the FP32 requirement.  Nothing is launched and no refused
    buffer is written."""
    x, z, w, dy = make(65, 48, seed=13)
    empty = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    elements = workspace_for(1, x.device).numel()
    for dtype in (torch.uint8,):
        other = torch.ones(elements, dtype=dtype, device="cuda")
        for args in ((dy, x, z), (empty, empty, empty)):
            for gdtype in (torch.bfloat16, torch.float32):
                main = torch.full((D,), 1.5, dtype=gdtype, device="cuda")
                with pytest.raises(ValueError) as info:
                    backward(*args, w, EPS, grad_weight=main, workspace=other)
                assert str(info.value) == refusal(dtype), str(info.value)
                assert torch.equal(main, torch.full_like(main, 1.5))
        assert bool(other.eq(1).all())
    for bad in (
        *_unfit_buffers(elements),
        torch.ones(elements, dtype=torch.float32),
    ):
        with pytest.raises(ValueError) as info:
            backward(dy, x, z, w, EPS, workspace=bad)
        assert str(info.value) == unfit(x.device, elements), str(info.value)
        assert bool(bad.eq(1).all())
    assert launches == []


def test_reset_workspace_clears_nonzero_scratch():
    """A caller workspace whose digit counters, flag words, arrival ticket and
    claim counter hold garbage (written by something other than the backward)
    is zeroed by ``reset_workspace``; backward launches at two shapes and
    weight-gradient dtypes then give the bits of a fresh workspace and leave
    it zero."""
    device = torch.device("cuda")
    ws = workspace_for(1, device)
    g = torch.Generator(device="cuda").manual_seed(60)
    ws.view(torch.int32).copy_(
        torch.randint(
            1, 2**31 - 1, ws.shape, dtype=torch.int32, device=device, generator=g
        )
    )
    assert bool(ws.view(torch.int32).ne(0).all())
    reset_workspace(ws)
    assert_zero(ws)
    for i, (T, H, dtype) in enumerate(
        ((1031, 48, torch.bfloat16), (37, 7, torch.float32))
    ):
        x, z, w, dy = make(T, H, seed=61 + i)

        def call(workspace, x=x, z=z, w=w, dy=dy, dtype=dtype):
            gw = torch.empty(D, dtype=dtype, device=device)
            return backward(dy, x, z, w, EPS, grad_weight=gw, workspace=workspace)

        got = call(ws)
        want = call(workspace_for(T * H, device))
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"{T}x{H} {dtype}: {name}")
        assert_zero(ws)


def test_reset_workspace_of_a_zero_workspace_is_a_no_op(launches):
    """Resetting a zero workspace keeps it zero and in place, launches no module
    and does not synchronize the host; the next backward is unaffected."""
    x, z, w, dy = make(129, 48, seed=62)
    want = backward(dy, x, z, w, EPS)
    ws = workspace_for(129 * 48, x.device)
    ptr = ws.data_ptr()
    launches.clear()
    torch.cuda.set_sync_debug_mode("error")
    try:
        reset_workspace(ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert ws.data_ptr() == ptr
    assert_zero(ws)
    assert launches == []
    got = backward(dy, x, z, w, EPS, workspace=ws)
    for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
        assert_same_bits(a, b, f"{name} after a reset of a zero workspace")
    assert_zero(ws)


def test_reset_workspace_errors():
    """``reset_workspace`` refuses a non-tensor (``TypeError``), the
    non-deterministic package's ``uint8`` workspace naming both modes'
    workspaces, other unfit buffers with the FP32 requirement and a
    host tensor, each left as it is; it takes no mode keyword."""
    device = torch.device("cuda")
    elements = workspace_for(1, device).numel()
    with pytest.raises(TypeError):
        reset_workspace(None)
    for dtype in (torch.uint8,):
        other = torch.ones(elements, dtype=dtype, device="cuda")
        with pytest.raises(ValueError) as info:
            reset_workspace(other)
        assert str(info.value) == refusal(dtype), str(info.value)
        assert bool(other.eq(1).all())
    for bad in _unfit_buffers(elements):
        with pytest.raises(ValueError) as info:
            reset_workspace(bad)
        assert str(info.value) == unfit(bad.device, elements), str(info.value)
        assert bool(bad.eq(1).all())
    with pytest.raises(ValueError, match="CUDA"):
        reset_workspace(torch.zeros(elements, dtype=torch.float32))
    ws = workspace_for(1, device)
    for value in (False, True, None):
        with pytest.raises(TypeError, match="deterministic"):
            pkg.cake_rmsnorm_silu_gate_det_train_reset_workspace(
                ws, deterministic=value
            )


def test_two_streams_with_their_own_workspaces():
    """Two shapes on two side streams at once, five rounds, each stream with
    its own caller workspace and with its stream's workspace: every output is
    the bits of the same calls on the current stream, and each caller
    workspace is zero after."""
    cases = [make(4097, 48, seed=30), make(1031, 16, seed=31)]
    refs = [outputs(x, z, w, dy) for x, z, w, dy in cases]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = [workspace_for(1, torch.device("cuda")) for _ in streams]
    results: list[list] = [[], []]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(5):
        for i, stream in enumerate(streams):
            x, z, w, dy = cases[i]
            with torch.cuda.stream(stream):
                results[i].append(outputs(x, z, w, dy, workspace=workspaces[i]))
                results[i].append(outputs(x, z, w, dy))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for i in range(2):
        for k, got in enumerate(results[i]):
            assert_outputs_equal(got, refs[i], f"stream {i} call {k}")
        assert_zero(workspaces[i], f"the workspace of stream {i}")


def test_side_stream_and_no_host_sync():
    x, z, w, dy = make(2049, 48, seed=9)
    want_y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
    want = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    ws = cake_rmsnorm_silu_gate_det_train_backward_workspace(
        x.shape[0] * x.shape[1], x.device
    )
    assert ws.numel() * 4 == cake_rmsnorm_silu_gate_det_train_backward_workspace_bytes(
        x.shape[0] * x.shape[1], x.device
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            y = cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS)
            got = cake_rmsnorm_silu_gate_det_train_backward(
                dy, x, z, w, EPS, workspace=ws
            )
            fresh = cake_rmsnorm_silu_gate_det_train_backward(dy, x, z, w, EPS)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(y, want_y)
    for out in (got, fresh):
        for a, b in zip(out, want, strict=True):
            assert torch.equal(a, b)
    assert_zero(ws)


def test_cuda_graph_capture():
    """The forward and the backward with a caller workspace, and an FP32
    accumulation without one, captured once and replayed three times on new
    inputs copied into the captured tensors: each replay gives the bits of
    eager calls on those inputs (the accumulated weight gradient the bits of
    the same accumulation run eagerly), and the caller workspace is zero after
    every replay."""
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=10)
    xs, zs, dys = x.clone(), z.clone(), dy.clone()
    y = torch.empty_like(x)
    dx, dz = torch.empty_like(x), torch.empty_like(x)
    dw = torch.empty(D, dtype=torch.bfloat16, device="cuda")
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    ws = workspace_for(T * H, x.device)

    def step():
        forward(xs, zs, w, EPS, out=y)
        backward(
            dys, xs, zs, w, EPS, grad_x=dx, grad_z=dz, grad_weight=dw, workspace=ws
        )
        backward(dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # loads the modules before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    main.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    eager_main = torch.zeros(D, dtype=torch.float32, device="cuda")
    for k in range(3):
        xn, zn, _, dyn = make(T, H, seed=40 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        for out in (y, dx, dz, dw):
            out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_same_bits(y, forward(xn, zn, w, EPS), f"replay {k} y")
        want = backward(dyn, xn, zn, w, EPS)
        for a, b, name in zip((dx, dz, dw), want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"replay {k} {name}")
        backward(
            dyn, xn, zn, w, EPS, grad_weight=eager_main, accumulate_grad_weight=True
        )
        assert_same_bits(main, eager_main, f"replay {k} accumulated dw")
        assert_zero(ws, f"the workspace after replay {k}")


def _kept_graph() -> torch.cuda.CUDAGraph:
    try:
        return torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        pytest.skip("listing graph nodes needs torch.cuda.CUDAGraph(keep_graph=True)")


def _graph_node_types(graph: torch.cuda.CUDAGraph) -> list[str]:
    """The node types of a graph captured with ``keep_graph=True``."""
    driver = pytest.importorskip("cuda.bindings.driver")
    handle = driver.CUgraph(graph.raw_cuda_graph())
    _, _, count = driver.cuGraphGetNodes(handle)
    _, nodes, count = driver.cuGraphGetNodes(handle, count)
    return [
        driver.cuGraphNodeGetType(node)[1].name.replace("CU_GRAPH_NODE_TYPE_", "")
        for node in nodes[:count]
    ]


@pytest.fixture
def reserve(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Fresh reserve and capture tables; the reserve of the current device,
    filled and seen ready by eager calls without a workspace."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    x, z, w, dy = make(17, 48, seed=69)
    backward(dy, x, z, w, EPS)  # loads the module, fills the reserve
    torch.cuda.synchronize()
    backward(dy, x, z, w, EPS)  # sees the fill complete
    torch.cuda.synchronize()
    state = pkg._RESERVES[torch.cuda.current_device()]
    assert len(state["ready"]) == pkg._RESERVE_SIZE and not state["pending"]
    return state


def test_capture_without_workspace_takes_a_reserve_workspace(reserve):
    """Backwards without a caller workspace captured in one graph on one
    stream share one zero-filled workspace taken from the reserve: the graph
    holds the backward kernels only (no fill kernel, memset or event wait), for
    both weight-gradient dtypes; every replay gives the bits of eager calls
    (the accumulated weight gradient those of the same accumulation run
    eagerly) and leaves the workspace zero; eager calls do not touch it."""
    pytest.importorskip("cuda.bindings.runtime")
    T, H = 129, 48
    x, z, w, dy = make(T, H, seed=70)
    want = backward(dy, x, z, w, EPS)
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    torch.cuda.synchronize()
    main.zero_()
    eager_main = torch.zeros_like(main)
    ready = list(reserve["ready"])

    def step():
        outs = [backward(dy, x, z, w, EPS) for _ in range(2)]
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
        return outs

    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    ws = taken[0]
    for k in range(3):
        backward(dy, x, z, w, EPS)  # an eager call between replays
        graph.replay()
        backward(dy, x, z, w, EPS, grad_weight=eager_main, accumulate_grad_weight=True)
        torch.cuda.synchronize()
        for out in outs:
            for a, b, name in zip(out, want, ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"replay {k} {name}")
        assert_same_bits(main, eager_main, f"replay {k} accumulated dw")
        assert_zero(ws, f"the reserve workspace after replay {k}")


def test_capture_runs_long_and_default_modules_on_shared_workspaces(reserve, launches):
    """Backwards of this device's first row count of the long modules
    (``LONG_MIN_ROWS_PER_SM * SMs`` flat packed rows) captured in one graph on
    one stream: a BF16 weight gradient on a caller workspace and an FP32 one
    accumulated without a workspace run their roles' long modules, and a BF16
    one whose ``grad_out`` is strided (read in place, so not packed) runs the
    ``bwd_bf16`` module on the reserve workspace the FP32 call took.  The
    graph holds the three backward kernels only (no fill kernel, memset or
    event wait).  Every replay on new inputs gives the bits of eager calls
    forced to the roles' default modules (the accumulated weight gradient
    those of the same accumulation run eagerly) and leaves the caller
    workspace and the reserve workspace zero."""
    pytest.importorskip("cuda.bindings.runtime")
    arch, sms = _device()
    routes = loader.ROUTES[arch]
    rows = LONG_MIN_ROWS_PER_SM[arch] * sms
    xs, zs, w, dys = make(rows, 1, layout="flat", seed=76)
    # ``grad_out`` read in place with token stride 2 * 128 + 16, so not packed
    wide = torch.zeros(rows, 2 * D + 16, dtype=torch.bfloat16, device=xs.device)
    dy_strided = wide[:, :D]
    dy_strided.copy_(dys)
    main = torch.zeros(D, dtype=torch.float32, device=xs.device)
    ws = workspace_for(rows, xs.device)

    def step():
        return (
            backward(dys, xs, zs, w, EPS, workspace=ws),
            backward(
                dys, xs, zs, w, EPS, grad_weight=main, accumulate_grad_weight=True
            ),
            backward(dy_strided, xs, zs, w, EPS),
        )

    want = [
        routes["bwd_bf16" + loader.LONG],
        routes["bwd_f32" + loader.LONG],
        routes["bwd_bf16"],
    ]
    launches.clear()
    step()  # loads the long modules before the capture
    torch.cuda.synchronize()
    assert launches == want, launches
    assert_zero(ws)
    main.zero_()
    ready = list(reserve["ready"])
    launches.clear()
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        outs = step()
    assert launches == want, launches
    types = _graph_node_types(graph)
    assert types == ["KERNEL"] * 3, types
    taken = [t for t in ready if all(t is not r for r in reserve["ready"])]
    assert len(taken) == 1 and len(reserve["ready"]) == len(ready) - 1
    eager_main = torch.zeros_like(main)
    defaults = [routes["bwd_bf16"], routes["bwd_f32"], routes["bwd_bf16"]]
    for k in range(3):
        xn, zn, _, dyn = make(rows, 1, layout="flat", seed=77 + k)
        xs.copy_(xn)
        zs.copy_(zn)
        dys.copy_(dyn)
        dy_strided.copy_(dyn)
        graph.replay()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "backward_module", _route_to(""))
            launches.clear()
            refs = (
                backward(dyn, xn, zn, w, EPS),
                backward(
                    dyn,
                    xn,
                    zn,
                    w,
                    EPS,
                    grad_weight=eager_main,
                    accumulate_grad_weight=True,
                ),
                backward(dy_strided, xn, zn, w, EPS),
            )
            assert launches == defaults, launches
        torch.cuda.synchronize()
        for i, (out, ref) in enumerate(zip(outs, refs, strict=True)):
            for a, b, name in zip(out, ref, ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"replay {k} call {i} ({want[i]}) {name}")
        assert_zero(ws, f"the caller workspace after replay {k}")
        assert_zero(taken[0], f"the reserve workspace after replay {k}")


def test_graphs_without_workspace_replay_together_on_two_streams(reserve):
    """Two graphs, each capturing a backward without a caller workspace, own
    two reserve workspaces and replay at the same time on two streams, the
    first replays right after the captures with no host synchronization: the
    bits of eager calls, and both workspaces zero after."""
    cases = [make(4097, 48, seed=71), make(1031, 16, seed=72)]
    refs = [backward(dy, x, z, w, EPS) for x, z, w, dy in cases]
    torch.cuda.synchronize()
    graphs, outs, taken = [], [], []
    for x, z, w, dy in cases:
        ready = list(reserve["ready"])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs.append(backward(dy, x, z, w, EPS))
        graphs.append(graph)
        taken += [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 2 and taken[0].data_ptr() != taken[1].data_ptr()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    results: list[list] = [[], []]
    for _ in range(5):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                graphs[i].replay()
                results[i].append(tuple(t.clone() for t in outs[i]))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for i in range(2):
        for k, got in enumerate(results[i]):
            for a, b, name in zip(got, refs[i], ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"graph {i} replay {k} {name}")
        assert_zero(taken[i], f"the workspace of graph {i}")


def test_captured_calls_without_capture_ids_take_one_workspace_each(reserve):
    """Without cuda-python's capture ids, each captured call takes its own
    reserve workspace; replays give the bits of eager calls and leave them
    zero."""
    x, z, w, dy = make(129, 48, seed=74)
    want = backward(dy, x, z, w, EPS)
    torch.cuda.synchronize()
    ready = list(reserve["ready"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "_cudart", lambda: None)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [backward(dy, x, z, w, EPS) for _ in range(3)]
    taken = [ws for ws in ready if all(ws is not r for r in reserve["ready"])]
    assert len(taken) == 3
    for k in range(2):
        graph.replay()
        torch.cuda.synchronize()
        for out in outs:
            for a, b, name in zip(out, want, ("dx", "dz", "dw"), strict=True):
                assert_same_bits(a, b, f"replay {k} {name}")
        for ws in taken:
            assert_zero(ws, f"a reserve workspace after replay {k}")


def test_capture_with_no_ready_workspace_fills_in_the_graph(monkeypatch):
    """With no ready reserve workspace a captured backward without a caller
    workspace zero-fills one from the graph's pool inside the graph (a memset
    or fill kernel before the backward kernel) and gives the bits of an eager
    call at every replay."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_RESERVE_SIZE", 0)
    x, z, w, dy = make(129, 48, seed=73)
    want = backward(dy, x, z, w, EPS)
    torch.cuda.synchronize()
    assert pkg._RESERVES[torch.cuda.current_device()]["blocks"] == []
    graph = _kept_graph()
    with torch.cuda.graph(graph):
        got = backward(dy, x, z, w, EPS)
    types = _graph_node_types(graph)
    assert types in (["MEMSET", "KERNEL"], ["KERNEL", "KERNEL"]), types
    for k in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"replay {k} {name}")


def test_capture_before_any_eager_call_without_workspace(monkeypatch):
    """A capture before any call without a caller workspace (the modules
    loaded by calls with one): the forward and the backward without a caller
    workspace (BF16 and accumulated FP32 weight gradient) capture with no
    stream workspace and no reserve made, each backward on a workspace the
    graph zero-fills; replays give the bits of eager calls, also after eager
    calls without a workspace have made the stream's workspace and the
    reserve."""
    monkeypatch.setattr(pkg, "_RESERVES", {})
    monkeypatch.setattr(pkg, "_CAPTURE_WORKSPACES", {})
    monkeypatch.setattr(pkg, "_STREAM_WORKSPACES", {})
    T, H = 300, 48
    x, z, w, dy = make(T, H, seed=75)
    outputs(x, z, w, dy, workspace=workspace_for(T * H, x.device))
    torch.cuda.synchronize()
    assert pkg._RESERVES == {} and pkg._STREAM_WORKSPACES == {}
    main = torch.zeros(D, dtype=torch.float32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = forward(x, z, w, EPS)
        got = backward(dy, x, z, w, EPS)
        backward(dy, x, z, w, EPS, grad_weight=main, accumulate_grad_weight=True)
    assert pkg._RESERVES == {} and pkg._STREAM_WORKSPACES == {}
    assert pkg._CAPTURE_WORKSPACES == {}
    eager_main = torch.zeros_like(main)
    for k in range(4):
        if k == 2:
            outputs(x, z, w, dy)  # makes the stream's workspace and the reserve
            torch.cuda.synchronize()
            assert pkg._STREAM_WORKSPACES and pkg._RESERVES
        graph.replay()
        backward(dy, x, z, w, EPS, grad_weight=eager_main, accumulate_grad_weight=True)
        torch.cuda.synchronize()
        assert_same_bits(y, forward(x, z, w, EPS), f"replay {k} y")
        want = backward(dy, x, z, w, EPS)
        for a, b, name in zip(got, want, ("dx", "dz", "dw"), strict=True):
            assert_same_bits(a, b, f"replay {k} {name}")
        assert_same_bits(main, eager_main, f"replay {k} accumulated dw")


def test_empty_and_errors():
    x = torch.empty(0, 48, D, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(D, dtype=torch.bfloat16, device="cuda")
    assert cake_rmsnorm_silu_gate_det_train_forward(x, x, w).shape == x.shape
    dx, dz, dw = cake_rmsnorm_silu_gate_det_train_backward(x, x, x, w)
    assert dx.shape == x.shape and torch.equal(dw, torch.zeros_like(dw))
    good = torch.randn(4, 2, D, device="cuda").bfloat16()
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_forward(good.float(), good, w)
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_forward(good, good[:, :1], w)
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_forward(
            torch.randn(4, 64, device="cuda").bfloat16(),
            torch.randn(4, 64, device="cuda").bfloat16(),
            w[:64],
        )
    with pytest.raises(ValueError):
        cake_rmsnorm_silu_gate_det_train_backward(
            good, good, good, w, accumulate_grad_weight=True
        )


def test_registry_records_reference_the_build_manifest():
    """Every call role (``ROLES``: the forward and the backward of each
    weight-gradient dtype) ships one module per architecture, every
    architecture routes the long module of each backward role
    (``LONG_ROLES``) and the programs of one CTA and of one cluster of each
    backward role (``CLUSTER_ROLES``), an architecture may also route a
    forward of packed rows
    (``PACKED_ROLES``), the tiny forward (``TINY_ROLES``; exactly where
    ``FWD_TINY_ROWS_PER_SM``, one int of rows per SM for each architecture,
    has a cut above 0; ``FWD_TINY_GAP_ROWS``, rows ``(lo, hi]`` for each
    architecture, ``(0, 0)`` without the tiny forward) and programs of a
    backward without the weight gradient
    (``NODW_ROLES``; ``NODW_STRIDED_ROWS_PER_SM`` cuts the rows between the
    strided and the claimed one exactly where both are routed; the strided
    program's 128-thread builds only where the strided one is, the build of
    packed rows only where the 128-thread build is; ``PACKED_SMALL_ROWS_PER_SM``
    and ``NODW_ANY_LAYOUT_ROWS_PER_SM`` are the shipped small-row cuts, each
    at or below the strided cut and below the long modules' first row
    count), and no other role is routed; every module
    record
    names its ``build.json`` library (schema ``cake-build/1``), whose entries
    carry the package's kernel and binding sources and the precise-arithmetic
    flags."""
    assert loader.ROLES == ("fwd", "bwd_bf16", "bwd_f32")
    assert loader.LONG == "_long"
    assert loader.LONG_ROLES == ("bwd_bf16_long", "bwd_f32_long")
    assert loader.NODW == "_nodw" and loader.STRIDED == "_strided"
    assert loader.PACKED == "_packed" and loader.SMALL == "_small"
    assert loader.NODW_ROLES == (
        NODW_CLAIMED,
        NODW_STRIDED,
        NODW_STRIDED_PACKED,
        NODW_STRIDED_SMALL,
        NODW_LONG,
    )
    assert loader.SOLO == "_solo" and loader.CLUSTER == "_cluster"
    assert loader.CLUSTER_ROLES == (
        "bwd_bf16" + loader.SOLO,
        "bwd_f32" + loader.SOLO,
        "bwd_bf16" + loader.CLUSTER,
        "bwd_f32" + loader.CLUSTER,
    )
    assert loader.GRID_KINDS == (
        "persistent_row_blocks",
        "claimed_row_blocks",
        "strided_row_blocks",
        "cluster",
        "row_blocks",
    )
    assert loader.TINY_ROLES == ("fwd_tiny",)
    assert set(loader.FWD_TINY_ROWS_PER_SM) == set(loader.ARCHES)
    for arch, per_sm in loader.FWD_TINY_ROWS_PER_SM.items():
        assert isinstance(per_sm, int) and per_sm >= 0, arch
        assert ("fwd_tiny" in loader.ROUTES[arch]) == (per_sm > 0), arch
    # the gap of each architecture: ``(lo, hi]`` rows, two ints with
    # ``0 <= lo <= hi``, ``(0, 0)`` where it keeps none or routes no tiny
    # forward
    assert set(loader.FWD_TINY_GAP_ROWS) == set(loader.ARCHES)
    for arch, gap in loader.FWD_TINY_GAP_ROWS.items():
        assert isinstance(gap, tuple) and len(gap) == 2, arch
        lo, hi = gap
        assert isinstance(lo, int) and isinstance(hi, int) and 0 <= lo <= hi, arch
        if loader.FWD_TINY_ROWS_PER_SM[arch] == 0:
            assert gap == (0, 0), arch
    assert set(loader.NODW_STRIDED_ROWS_PER_SM) == set(loader.ARCHES)
    for arch, cuts in loader.NODW_STRIDED_ROWS_PER_SM.items():
        both = {NODW_STRIDED, NODW_CLAIMED} <= set(loader.ROUTES[arch])
        assert set(cuts) == ({"bwd_bf16"} if both else set()), arch
        if NODW_STRIDED_PACKED in loader.ROUTES[arch]:
            assert NODW_STRIDED in loader.ROUTES[arch], arch
        if NODW_STRIDED_SMALL in loader.ROUTES[arch]:
            assert NODW_STRIDED_PACKED in loader.ROUTES[arch], arch
        assert all(isinstance(v, int) and v >= 1 for v in cuts.values()), arch
    for table, shipped in (
        (loader.PACKED_SMALL_ROWS_PER_SM, SMALL_ROWS_PER_SM),
        (loader.NODW_ANY_LAYOUT_ROWS_PER_SM, ANY_LAYOUT_ROWS_PER_SM),
    ):
        assert table == shipped
        assert set(table) == set(loader.ARCHES)
        for arch, cuts in table.items():
            assert set(cuts) <= {"bwd_bf16"}, arch
            cut = loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16")
            for per_sm in cuts.values():
                assert isinstance(per_sm, int) and per_sm >= 1, arch
                assert cut is None or per_sm <= cut, arch
                assert per_sm < LONG_MIN_ROWS_PER_SM[arch], arch
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert (
        manifest["package"]
        == "cake_pplx._backends.cake_rmsnorm_silu_gate_det_train"
    )
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    for arch in loader.ARCHES:
        roles = set(loader.ROUTES[arch])
        assert {*loader.ROLES, *loader.LONG_ROLES, *loader.CLUSTER_ROLES} <= roles
        assert roles <= {
            *loader.ROLES,
            *loader.LONG_ROLES,
            *loader.PACKED_ROLES,
            *loader.TINY_ROLES,
            *loader.NODW_ROLES,
            *loader.CLUSTER_ROLES,
        }
        for role, name in loader.ROUTES[arch].items():
            record = loader.MODULES[name]
            assert record["role"] == role and arch in record["arches"], (arch, role)
            assert loader.library_name(name, arch) in libraries, (arch, role, name)
    routed = {name for by_role in loader.ROUTES.values() for name in by_role.values()}
    assert set(loader.MODULES) == routed
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["kind"] in loader.GRID_KINDS
        assert record["launch"]["cooperative"] is False
        assert record["launch"]["use_pdl"] is False
        for arch in record["arches"]:
            entry = libraries[loader.library_name(name, arch)]
            for flag in ("--prec-div=true", "--prec-sqrt=true", "--ftz=false"):
                assert flag in entry["nvcc_flags"], (name, arch, flag)
            assert "--use_fast_math" not in entry["nvcc_flags"], (name, arch)
            for source in entry["sources"]:
                assert (package / source).is_file(), (name, source)


# The logical arguments the host passes to each kind of module.
FORWARD_ARGUMENTS = {
    "x",
    "z",
    "w",
    "y",
    "rows",
    "heads",
    "x_tok",
    "x_head",
    "z_tok",
    "z_head",
    "eps",
}
# The tiny forward reads packed rows only: no head count and no strides.
TINY_ARGUMENTS = {"x", "z", "w", "y", "rows", "eps"}
BACKWARD_ARGUMENTS = {
    "dy",
    "x",
    "z",
    "w",
    "dx",
    "dz",
    "acc",
    "flags",
    "done",
    "dw",
    "rows",
    "heads",
    "dy_tok",
    "dy_head",
    "x_tok",
    "x_head",
    "z_tok",
    "z_head",
    "eps",
    "accumulate",
    "weight_grad",
}


def test_records_launch_alone_and_take_the_hosts_arguments():
    """Every record launches without a cooperative launch or programmatic
    dependent launch, and without a cluster but the programs of one cluster,
    and its kernel parameters map exactly onto the logical arguments the
    host passes.  Each forward (``fwd`` and
    ``fwd_packed``): 128 threads, no dynamic shared memory, no workspace, a
    ``persistent_row_blocks`` rule of positive ``ctas_per_sm``,
    ``max_blocks_per_cta`` and ``rows_per_block`` and a positive cut of rows
    per SM (``persistent_from_rows_per_sm``) for each of its architectures,
    keyed by exactly them in order, and the ``fwd`` module (rows of any
    layout) a second such cut, ``persistent_from_rows_per_sm_strided``, which
    the forward of packed rows only has not; an architecture's two forwards
    launch alike (their rules but that second cut) and take the same
    arguments.  The tiny forward (``TINY_ROLES``): no dynamic shared
    memory, no workspace, a ``row_blocks`` rule of a positive
    ``rows_per_cta`` that splits its CTA's threads into rows of 8, 16, 32 or
    64 lanes, the arguments ``TINY_ARGUMENTS``, and the entry, cluster and
    launch flags of its architecture's ``fwd`` module, on some or all of
    that module's architectures.
    Each backward: 256 threads and the ``claimed_row_blocks`` rule of 4 CTAs
    per SM over 16-row blocks, and a workspace of 2 slots of 9 limbs, the same
    for both weight-gradient dtypes; a long module's rule adds its
    ``min_rows_per_sm`` (``LONG_MIN_ROWS_PER_SM``) and ``packed_rows``, and it
    launches and takes its arguments as its role's module does.  Each program
    without the weight gradient launches and takes its arguments as the
    ``bwd_bf16`` module it replaces (the long one for ``bwd_bf16_nodw_long``),
    on some or all of that module's architectures, and keeps its rule and
    workspace, but the strided ones, which have no workspace and whose
    ``strided_row_blocks`` rules (``NODW_STRIDED_GRID_RULES``) count the
    ``bwd_bf16`` module's CTAs (the 256-thread one) or 8 CTAs per SM over
    8-row blocks (the 128-thread builds, which launch 128 threads); the
    build of packed rows only launches and takes its arguments as the
    128-thread build does.  Each program of one CTA or of one cluster
    (``CLUSTER_ROLES``): 256 threads, no workspace, the ``cluster`` rule of
    its CTAs and the most rows it serves (``SMALL_DW_RULES``: 1 CTA up to 16
    rows, 16 CTAs up to 256 rows), launched as one cluster of its CTAs, and
    the arguments, entry and architectures of its role's module."""
    for name, record in loader.MODULES.items():
        launch = record["launch"]
        rule = record["grid_rule"]
        assert tuple(launch["cluster"]) == (rule.get("cluster", 1), 1, 1), name
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        logical = record["logical"]
        plan = record["arg_plan"]
        for kind, key in plan:
            if kind == "grid":
                assert key in ("grid_x", "grid_y", "grid_z"), (name, key)
            else:
                assert key in logical, (name, kind, key)
        assert {key for kind, key in plan if kind != "grid"} == set(logical), name
        if record["role"] in ("fwd", *loader.PACKED_ROLES):
            assert rule["kind"] == "persistent_row_blocks", name
            positive = ("ctas_per_sm", "max_blocks_per_cta", "rows_per_block")
            cuts = (CUT, STRIDED_CUT) if record["role"] == "fwd" else (CUT,)
            assert set(rule) == {"kind", *positive, *cuts}, name
            for key in positive:
                assert isinstance(rule[key], int) and rule[key] >= 1, (name, key)
            for key in cuts:
                assert isinstance(rule[key], dict), (name, key)
                assert list(rule[key]) == sorted(record["arches"]), (name, key)
                for arch, cut in rule[key].items():
                    assert isinstance(cut, int) and cut >= 1, (name, key, arch)
            assert tuple(launch["block"]) == (128, 1, 1), name
            assert launch["dynamic_smem_bytes"] == 0, name
            assert "workspace" not in record, name
            assert set(logical.values()) == FORWARD_ARGUMENTS, name
        elif record["role"] in loader.TINY_ROLES:
            assert set(rule) == {"kind", "rows_per_cta"}, name
            assert rule["kind"] == "row_blocks", name
            per_cta = rule["rows_per_cta"]
            assert isinstance(per_cta, int) and per_cta >= 1, name
            threads = launch["block"][0]
            assert tuple(launch["block"]) == (threads, 1, 1), name
            assert threads % per_cta == 0, name
            assert threads // per_cta in (8, 16, 32, 64), name
            assert launch["dynamic_smem_bytes"] == 0, name
            assert "workspace" not in record, name
            assert set(logical.values()) == TINY_ARGUMENTS, name
        elif record["role"] in NODW_STRIDED_GRID_RULES:
            assert rule == NODW_STRIDED_GRID_RULES[record["role"]], name
            threads = NODW_STRIDED_THREADS[record["role"]]
            assert tuple(launch["block"]) == (threads, 1, 1), name
            assert "workspace" not in record, name
            assert set(logical.values()) == BACKWARD_ARGUMENTS, name
        elif record["role"] in loader.CLUSTER_ROLES:
            ctas, most = SMALL_DW_RULES[_small_suffix(record["role"])]
            assert rule == {"cluster": ctas, "kind": "cluster", "max_rows": most}, name
            assert tuple(launch["block"]) == (256, 1, 1), name
            assert "workspace" not in record, name
            assert set(logical.values()) == BACKWARD_ARGUMENTS, name
        else:
            want = {
                "ctas_per_sm": 4,
                "kind": "claimed_row_blocks",
                "rows_per_block": 16,
            }
            if record["role"].endswith(loader.LONG):
                want["min_rows_per_sm"] = LONG_MIN_ROWS_PER_SM
                want["packed_rows"] = True
            assert rule == want, name
            assert tuple(launch["block"]) == (256, 1, 1), name
            assert record["workspace"] == {"limbs": 9, "slots": 2}, name
            assert set(logical.values()) == BACKWARD_ARGUMENTS, name
    for arch in loader.ARCHES:
        bf16, f32 = (
            loader.MODULES[loader.route_module(arch, role)]
            for role in ("bwd_bf16", "bwd_f32")
        )
        for key in ("grid_rule", "workspace", "logical"):
            assert bf16[key] == f32[key], (arch, key)
        for role in ("bwd_bf16", "bwd_f32"):
            short = loader.MODULES[loader.route_module(arch, role)]
            long = loader.MODULES[loader.route_module(arch, role + loader.LONG)]
            for key in ("workspace", "logical", "arg_plan", "ffi_entry", "arches"):
                assert short[key] == long[key], (arch, role, key)
            for key in ("block", "cluster", "cooperative", "use_pdl"):
                assert short["launch"][key] == long["launch"][key], (arch, role, key)
            rule = dict(long["grid_rule"])
            del rule["min_rows_per_sm"], rule["packed_rows"]
            assert rule == short["grid_rule"], (arch, role)
        if "fwd_packed" in loader.ROUTES[arch]:
            fwd, packed = (
                loader.MODULES[loader.route_module(arch, role)]
                for role in ("fwd", "fwd_packed")
            )
            for key in ("launch", "logical", "arg_plan", "ffi_entry"):
                assert fwd[key] == packed[key], (arch, key)
            rule = {key: v for key, v in fwd["grid_rule"].items() if key != STRIDED_CUT}
            assert rule == packed["grid_rule"], arch
        if "fwd_tiny" in loader.ROUTES[arch]:
            fwd, tiny = (
                loader.MODULES[loader.route_module(arch, role)]
                for role in ("fwd", "fwd_tiny")
            )
            assert tiny["ffi_entry"] == fwd["ffi_entry"], arch
            assert set(tiny["arches"]) <= set(fwd["arches"]), arch
            for key in ("cluster", "cooperative", "use_pdl"):
                assert tiny["launch"][key] == fwd["launch"][key], (arch, key)
        routes = loader.ROUTES[arch]
        for role, twin_role in (
            (NODW_CLAIMED, "bwd_bf16"),
            (NODW_STRIDED, "bwd_bf16"),
            (NODW_STRIDED_PACKED, "bwd_bf16"),
            (NODW_STRIDED_SMALL, "bwd_bf16"),
            (NODW_LONG, "bwd_bf16" + loader.LONG),
        ):
            if role not in routes:
                continue
            nodw, twin = (loader.MODULES[routes[r]] for r in (role, twin_role))
            for key in ("logical", "arg_plan", "ffi_entry"):
                assert nodw[key] == twin[key], (arch, role, key)
            assert set(nodw["arches"]) <= set(twin["arches"]), (arch, role)
            for key in ("block", "cluster", "cooperative", "use_pdl"):
                if key == "block" and role in NODW_128:
                    continue
                assert nodw["launch"][key] == twin["launch"][key], (arch, role, key)
            if role == NODW_STRIDED:
                for key in ("ctas_per_sm", "rows_per_block"):
                    assert nodw["grid_rule"][key] == twin["grid_rule"][key], key
            elif role not in NODW_128:
                for key in ("grid_rule", "workspace"):
                    assert nodw[key] == twin[key], (arch, role, key)
        if NODW_STRIDED_SMALL in routes:
            small, build = (loader.MODULES[routes[r]] for r in NODW_128)
            for key in ("grid_rule", "launch", "logical", "arg_plan", "ffi_entry"):
                assert small[key] == build[key], (arch, key)
        for role in loader.CLUSTER_ROLES:
            twin_role = role.removesuffix(_small_suffix(role))
            small, twin = (loader.MODULES[routes[r]] for r in (role, twin_role))
            for key in ("logical", "arg_plan", "ffi_entry", "arches"):
                assert small[key] == twin[key], (arch, role, key)
            for key in ("block", "cooperative", "use_pdl"):
                assert small["launch"][key] == twin["launch"][key], (arch, role, key)


# Without a device: the grids of both rules at several SM counts (small
# MIG-like instances and the products' counts).
GRID_SM_COUNTS = (1, 2, 7, 14, 16, 24, 33, 66, 148, 152, 160, 212, 216)
# The issue's row counts and the kernels' 32-bit row limit.
ISSUE_ROWS = (30193 * 48, 32388 * 48, 32737 * 48, 65536 * 48, 2**31 - 1)


def _grid_rows(rule, sm_count, arch=None):
    """Row counts at the edges of ``rule``'s grids on ``sm_count`` SMs: a
    row, a block, the rows around one and two waves (of
    ``max_blocks_per_cta`` blocks a CTA for the forward), for the forward on
    ``arch`` also around each of its cuts of rows per SM (``CUT`` and, where
    the rule has one, ``STRIDED_CUT``) and one, two and seven whole waves
    from there, and the issue's."""
    per_block = rule["rows_per_block"]
    wave = rule["ctas_per_sm"] * sm_count
    per_cta = rule.get("max_blocks_per_cta", 1)
    rows = {1, per_block - 1, per_block, per_block + 1, *ISSUE_ROWS}
    edges = [
        wave * per_block,
        (wave + 1) * per_block,
        wave * per_cta * per_block,
        (wave * per_cta + 1) * per_block,
        2 * wave * per_cta * per_block,
        (2 * wave * per_cta + 1) * per_block,
        (7 * wave * per_cta + 3) * per_block,
    ]
    if CUT in rule:
        step = wave * per_cta * per_block
        for packed in (True, False):
            start, waves = _persistent_waves_rows(rule, arch, sm_count, packed)
            edges += [start, waves, waves + step, waves + 6 * step + 3 * per_block]
    for edge in edges:
        rows |= {edge - 1, edge, edge + 1}
    return sorted(r for r in rows if r >= 1)


def test_launch_grid_takes_the_rows_the_sm_count_the_arch_and_the_layout_only():
    """A grid depends on the record, the rows, the SM count, the architecture
    and whether the kernel reads the rows as packed only (``arch`` and
    ``packed`` keywords with the defaults None and True); a rule of another
    kind is a ``ValueError``."""
    parameters = inspect.signature(loader.launch_grid).parameters
    assert list(parameters) == ["record", "rows", "sm_count", "arch", "packed"]
    for key in ("rows", "sm_count", "arch", "packed"):
        assert parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, key
    assert parameters["arch"].default is None
    assert parameters["packed"].default is True
    rule = {"ctas_per_sm": 1, "kind": "row_claims", "rows_per_block": 16}
    with pytest.raises(ValueError, match="row_claims"):
        loader.launch_grid({"grid_rule": rule}, rows=1, sm_count=1)


def _forward_records():
    """``(arch, name)`` of every forward record each architecture routes
    (``fwd`` and, where routed, ``fwd_packed``)."""
    return [
        (arch, loader.route_module(arch, role))
        for arch in loader.ARCHES
        for role in ("fwd", *loader.PACKED_ROLES)
        if role in loader.ROUTES[arch]
    ]


def test_forward_grid_needs_an_architecture_its_cut_keys():
    """A forward record's cuts are keyed by architecture: ``launch_grid``
    without ``arch``, with ``arch`` None, or with an architecture a cut does
    not key (another product's, or an exported one the record does not ship
    on) raises ``ValueError`` naming the record's kernel, the architecture
    and the keys, at any rows (below the cut too) and for rows read as
    packed or not; with each of its architectures it gives a grid.  A rule
    whose cut is a single number serves every architecture and none."""
    for _, name in _forward_records():
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        others = [arch for arch in loader.ARCHES if arch not in record["arches"]]
        for packed in (True, False):
            key = STRIDED_CUT if not packed and STRIDED_CUT in rule else CUT
            keys = str(sorted(rule[key]))
            for rows in (1, 16 * 1024 * 148, 2**31 - 1):
                for arch in (None, "sm_90a", *others):
                    with pytest.raises(ValueError) as raised:
                        loader.launch_grid(
                            record, rows=rows, sm_count=148, arch=arch, packed=packed
                        )
                    message = str(raised.value)
                    for part in (name, key, repr(arch), keys):
                        assert part in message, (name, arch, packed, message)
                with pytest.raises(ValueError, match=name):
                    loader.launch_grid(record, rows=rows, sm_count=148, packed=packed)
                for arch in record["arches"]:
                    got = loader.launch_grid(
                        record, rows=rows, sm_count=148, arch=arch, packed=packed
                    )
                    assert got[1:] == (1, 1) and got[0] >= 1, (name, arch, got)
    # one number: 100 rows per SM on 7 SMs, a CTA a block up to 699 rows
    single = {
        "ctas_per_sm": 6,
        "kind": "persistent_row_blocks",
        "max_blocks_per_cta": 2,
        CUT: 100,
        "rows_per_block": 16,
    }
    for arch in (None, "sm_90a", *loader.ARCHES):
        for packed in (True, False):
            for rows, ctas in ((699, 44), (700, 42)):
                got = loader.launch_grid(
                    {"grid_rule": single},
                    rows=rows,
                    sm_count=7,
                    arch=arch,
                    packed=packed,
                )
                assert got == (ctas, 1, 1), (arch, packed, rows, got)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_forward_grid_is_one_cta_a_block_then_whole_waves(sm_count):
    """The forward's grid, of rows the kernel reads as packed and of other
    rows: below its cut (``_cut``) times the SMs one CTA a row block; from
    there never more CTAs than row blocks, no CTA with more than
    ``max_blocks_per_cta`` blocks, whole waves of ``ctas_per_sm * SMs`` CTAs
    unless one CTA a block, and the fewest such waves (exactly ``k`` waves at
    the fewest rows of ``k`` whole waves of ``max_blocks_per_cta`` blocks a
    CTA from there, ``k + 1`` from a row more), for every forward record on
    each of its architectures.  ``module_for_rows`` names the one ``fwd``
    module whatever the rows."""
    for arch, name in _forward_records():
        record = loader.MODULES[name]
        rule = record["grid_rule"]
        per_block = rule["rows_per_block"]
        most = rule["max_blocks_per_cta"]
        wave = rule["ctas_per_sm"] * sm_count
        for packed in (True, False):
            start, waves_rows = _persistent_waves_rows(rule, arch, sm_count, packed)
            for rows in _grid_rows(rule, sm_count, arch):
                if record["role"] == "fwd":
                    got = loader.module_for_rows(arch, "fwd", rows, sm_count)
                    assert got == name, (arch, rows, got)
                grid = loader.launch_grid(
                    record, rows=rows, sm_count=sm_count, arch=arch, packed=packed
                )
                ctas, blocks = grid[0], -(-rows // per_block)
                what = (arch, packed, rows, grid)
                assert grid[1:] == (1, 1) and 1 <= ctas <= blocks, what
                if rows < start:
                    assert ctas == blocks, what
                    continue
                assert -(-blocks // ctas) <= most, what
                assert ctas == blocks or ctas % wave == 0, what
                assert ctas == blocks or (ctas - wave) * most < blocks, what
            k = waves_rows // (wave * most * per_block)
            for rows, ctas in (
                (start - 1, -(-(start - 1) // per_block)),
                (waves_rows, min(k * wave, -(-waves_rows // per_block))),
                (waves_rows + 1, min((k + 1) * wave, waves_rows // per_block + 1)),
                (per_block, 1),
            ):
                if rows >= 1:
                    got = loader.launch_grid(
                        record, rows=rows, sm_count=sm_count, arch=arch, packed=packed
                    )
                    assert got == (ctas, 1, 1), (arch, packed, rows, got)


# (rows, SMs, architecture, CTAs) of the forward of rows the kernel reads as
# packed (each forward record of the architecture: ``fwd`` and, on sm_107a,
# ``fwd_packed``): one CTA per 16-row block below the architecture's cut of
# rows per SM times the SMs, from there the fewest whole waves of 6 CTAs per
# SM whose CTAs walk at most 2 blocks.  A row less than the cut times the SMs
# and the cut times the SMs on 148, 152 and 212 SMs on every architecture
# (the cuts: 8192 rows per SM on sm_100a and sm_107a, 5600 on sm_103a), and
# rows elsewhere.
FORWARD_GRIDS = [
    (1, 148, "sm_100a", 1),
    (520 * 148 + 1, 148, "sm_100a", 4811),  # 4811 blocks, a CTA a block
    (32737 * 48, 212, "sm_100a", 98211),  # 7412 rows per SM: a CTA a block
    (8192 * 148 - 1, 148, "sm_100a", 75776),  # a row below the cut: a CTA a block
    (8192 * 148, 148, "sm_100a", 38184),  # the cut: 75776 blocks, 43 waves of 888
    (8192 * 152 - 1, 152, "sm_100a", 77824),
    (8192 * 152, 152, "sm_100a", 39216),  # 77824 blocks, 43 waves of 912
    (8192 * 212 - 1, 212, "sm_100a", 108544),
    (8192 * 212, 212, "sm_100a", 54696),  # 108544 blocks, 43 waves of 1272
    (32737 * 48, 148, "sm_100a", 49728),  # 98211 blocks, 56 waves of 888
    (2**31 - 1, 148, "sm_100a", 67109712),  # 2**27 blocks, 75574 waves
    (1, 152, "sm_103a", 1),
    (5600 * 148 - 1, 148, "sm_103a", 51800),
    (5600 * 148, 148, "sm_103a", 26640),  # 51800 blocks, 30 waves of 888
    (5600 * 152 - 1, 152, "sm_103a", 53200),  # a row below the cut: a CTA a block
    (5600 * 152, 152, "sm_103a", 27360),  # the cut: 53200 blocks, 30 waves of 912
    (5600 * 212 - 1, 212, "sm_103a", 74200),
    (5600 * 212, 212, "sm_103a", 38160),  # 74200 blocks, 30 waves of 1272
    (6000 * 152, 152, "sm_103a", 29184),  # 57000 blocks, 32 waves
    (32737 * 48, 152, "sm_103a", 49248),  # 98211 blocks, 54 waves of 912
    (32737 * 48, 212, "sm_103a", 49608),  # 7412 rows per SM: 39 waves of 1272
    (1, 212, "sm_107a", 1),
    (8192 * 148 - 1, 148, "sm_107a", 75776),
    (8192 * 148, 148, "sm_107a", 38184),  # 75776 blocks, 43 waves of 888
    (8192 * 152 - 1, 152, "sm_107a", 77824),
    (8192 * 152, 152, "sm_107a", 39216),  # 77824 blocks, 43 waves of 912
    (32737 * 48, 212, "sm_107a", 98211),  # 7412 rows per SM: a CTA a block
    (8192 * 212 - 1, 212, "sm_107a", 108544),  # a row below the cut: a CTA a block
    (8192 * 212, 212, "sm_107a", 54696),  # the cut: 108544 blocks, 43 waves of 1272
    (65536 * 48, 212, "sm_107a", 99216),  # 196608 blocks, 78 waves of 1272
    (2**31 - 1, 216, "sm_107a", 67109472),  # 2**27 blocks, 51782 waves
]
# (rows, SMs, architecture, CTAs) of the forward of rows of any layout
# (``fwd``) on rows the kernel does not read as packed (strided, broadcast
# over tokens, or one token): its cut of such rows, 500 rows per SM on sm_100a
# and 400 on sm_103a and sm_107a, a row less than it times the SMs and it
# times the SMs on 148, 152 and 212 SMs.  The forward of packed rows only
# (``fwd_packed``) keeps its cut of packed rows on any rows
# (``FORWARD_GRIDS``).
STRIDED_FORWARD_GRIDS = [
    (1, 148, "sm_100a", 1),
    (500 * 148 - 1, 148, "sm_100a", 4625),  # a row below the cut: a CTA a block
    (500 * 148, 148, "sm_100a", 2664),  # the cut: 4625 blocks, 3 waves of 888
    (500 * 152 - 1, 152, "sm_100a", 4750),
    (500 * 152, 152, "sm_100a", 2736),  # 4750 blocks, 3 waves of 912
    (500 * 212 - 1, 212, "sm_100a", 6625),
    (500 * 212, 212, "sm_100a", 3816),  # 6625 blocks, 3 waves of 1272
    (8192 * 148 - 1, 148, "sm_100a", 38184),  # 75776 blocks, 43 waves of 888
    (400 * 148 - 1, 148, "sm_103a", 3700),
    (400 * 148, 148, "sm_103a", 2664),  # 3700 blocks, 3 waves of 888
    (400 * 152 - 1, 152, "sm_103a", 3800),  # a row below the cut: a CTA a block
    (400 * 152, 152, "sm_103a", 2736),  # the cut: 3800 blocks, 3 waves of 912
    (400 * 212 - 1, 212, "sm_103a", 5300),
    (400 * 212, 212, "sm_103a", 3816),  # 5300 blocks, 3 waves of 1272
    (5600 * 152 - 1, 152, "sm_103a", 27360),  # 53200 blocks, 30 waves of 912
    (400 * 148 - 1, 148, "sm_107a", 3700),
    (400 * 148, 148, "sm_107a", 2664),  # 3700 blocks, 3 waves of 888
    (400 * 152 - 1, 152, "sm_107a", 3800),
    (400 * 152, 152, "sm_107a", 2736),  # 3800 blocks, 3 waves of 912
    (400 * 212 - 1, 212, "sm_107a", 5300),  # a row below the cut: a CTA a block
    (400 * 212, 212, "sm_107a", 3816),  # the cut: 5300 blocks, 3 waves of 1272
    (32737 * 48, 212, "sm_107a", 49608),  # 7412 rows per SM: 39 waves of 1272
    (2**31 - 1, 216, "sm_107a", 67109472),
]


@pytest.mark.parametrize(("rows", "sm_count", "arch", "ctas"), FORWARD_GRIDS)
def test_forward_grid_of_packed_rows_at_each_architectures_cut(
    rows, sm_count, arch, ctas
):
    """``FORWARD_GRIDS``: every forward record of ``arch`` (the ``fwd``
    module and, where routed, ``fwd_packed``) launches ``ctas`` CTAs for
    ``rows`` packed rows on ``sm_count`` SMs."""
    names = [name for each, name in _forward_records() if each == arch]
    assert names, arch
    for name in names:
        record = loader.MODULES[name]
        assert arch in record["arches"], (name, arch)
        got = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
        assert got == (ctas, 1, 1), (name, arch, rows, sm_count, got)
        got = loader.launch_grid(
            record, rows=rows, sm_count=sm_count, arch=arch, packed=True
        )
        assert got == (ctas, 1, 1), (name, arch, rows, sm_count, got)


@pytest.mark.parametrize(("rows", "sm_count", "arch", "ctas"), STRIDED_FORWARD_GRIDS)
def test_forward_grid_of_other_rows_at_each_architectures_cut(
    rows, sm_count, arch, ctas
):
    """``STRIDED_FORWARD_GRIDS``: the ``fwd`` module of ``arch`` launches
    ``ctas`` CTAs for ``rows`` rows it does not read as packed on
    ``sm_count`` SMs; the forward of packed rows only (``fwd_packed``, where
    routed) launches on them the grid of packed rows."""
    name = loader.route_module(arch, "fwd")
    record = loader.MODULES[name]
    got = loader.launch_grid(
        record, rows=rows, sm_count=sm_count, arch=arch, packed=False
    )
    assert got == (ctas, 1, 1), (name, arch, rows, sm_count, got)
    for each, other in _forward_records():
        if each != arch or other == name:
            continue
        got = loader.launch_grid(
            loader.MODULES[other], rows=rows, sm_count=sm_count, arch=arch, packed=False
        )
        want = loader.launch_grid(
            loader.MODULES[other], rows=rows, sm_count=sm_count, arch=arch
        )
        assert got == want, (other, arch, rows, got, want)


# Strides (in elements) the host passes the forward kernel for ``x`` and
# ``z`` of 48 heads (``(heads, x_tok, x_head, z_tok, z_head)``), and whether
# the kernel reads them as packed rows: its own test, both head strides 128
# and both token strides ``heads * 128``.  A single token's stride is passed
# as 0 and a single head's as 128 (``_kernel_view``).
FORWARD_PACKED_STRIDES = {
    "contiguous": ((48, 48 * D, D, 48 * D, D), True),
    "one_head": ((1, D, D, D, D), True),
    "one_token": ((48, 0, D, 0, D), False),
    "projection": ((48, 48 * D, D, 2 * 48 * D + 16, D), False),
    "x_projection": ((48, 2 * 48 * D + 16, D, 48 * D, D), False),
    "head_major": ((48, D, 257 * D, D, 257 * D), False),
    "broadcast": ((48, 0, D, 0, D), False),
    "x_broadcast": ((48, 0, D, 48 * D, D), False),
    "z_broadcast": ((48, 48 * D, D, 0, D), False),
    "wide_heads": ((48, 2 * 48 * D, 2 * D, 2 * 48 * D, 2 * D), False),
    "mixed": ((48, 48 * D, D, D, 257 * D), False),
}


def test_forward_packed_is_the_kernels_packed_rows_test():
    """``forward_packed`` (host only) on ``FORWARD_PACKED_STRIDES``: the
    kernel's test of packed rows, both head strides 128 and both token strides
    ``heads * 128``, whatever else; one stride off fails it."""
    for layout, (strides, want) in FORWARD_PACKED_STRIDES.items():
        assert loader.forward_packed(*strides) is want, layout
    for heads in (1, 2, 48, 3120):
        row = heads * D
        assert loader.forward_packed(heads, row, D, row, D) is True, heads
        for slot in range(1, 5):
            for delta in (-16, 16):
                strides = [heads, row, D, row, D]
                strides[slot] += delta
                assert loader.forward_packed(*strides) is False, (heads, strides)


# Inputs of ``T`` tokens of ``H`` heads past every tiny forward's cut and from
# 1000 rows per SM on up to 220 SMs (4600 x 48 rows; one token of 220800
# heads), past the cut of rows not read as packed (500 rows per SM on sm_100a,
# 400 on sm_103a and sm_107a), in each layout, and whether the forward kernel
# reads them as packed rows:
# contiguous ``[T, H, 128]``, flat ``[T * H, 128]`` and one head ``[T, 1,
# 128]`` (both strides as packed rows), a misaligned input (copied once to
# packed rows); not a gate sliced from a projection, head-major rows, ``x``
# and ``z`` broadcast over tokens (token stride 0) or one of them, or one
# token (its token stride passed as 0).
PACKED_TEST_LAYOUTS = {
    "contiguous": True,
    "flat": True,
    "one_head": True,
    "misaligned": True,
    "projection": False,
    "head_major": False,
    "broadcast": False,
    "x_broadcast": False,
    "z_broadcast": False,
    "one_token": False,
}


def _packed_test_inputs(layout, seed):
    """``(x, z, w)`` of ``layout`` (``PACKED_TEST_LAYOUTS``)."""
    if layout in ("contiguous", "flat", "misaligned", "projection", "head_major"):
        x, z, w, _ = make(4600, 48, layout=layout, seed=seed)
    elif layout == "one_head":
        x, z, w, _ = make(4600 * 48, 1, seed=seed)
    elif layout == "one_token":
        x, z, w, _ = make(1, 4600 * 48, seed=seed)
    else:
        x, z, w, _ = make(4600, 48, seed=seed)
        if layout in ("broadcast", "x_broadcast"):
            x = x[:1].expand(x.shape)
        if layout in ("broadcast", "z_broadcast"):
            z = z[:1].expand(z.shape)
    return x, z, w


@pytest.mark.parametrize("layout", PACKED_TEST_LAYOUTS)
def test_the_forward_grid_follows_the_kernels_packed_rows_test(layout):
    """A forward of ``layout`` (``PACKED_TEST_LAYOUTS``): ``forward_packed``
    of the strides the call passes the kernel is the layout's answer, and the
    call passes it to ``launch_grid`` as ``packed`` where the launched
    record's rule has a cut of rows not read as packed (else ``packed`` is
    True: the forward of packed rows only, whose cut does not depend on it),
    with the device's architecture: the grid is that of the cut of the
    kernel's reading of the rows.  ``y`` is the bits of the same rows given
    as contiguous tensors (where the layout is not packed, on another
    grid)."""
    arch, sms = _device()
    x, z, w = _packed_test_inputs(layout, 39)
    want = PACKED_TEST_LAYOUTS[layout]
    calls = []
    run_module, launch_grid = loader.run_module, loader.launch_grid

    def run_spy(name, values, *, rows, device_index):
        calls.append(("run", name, dict(values)))
        return run_module(name, values, rows=rows, device_index=device_index)

    def grid_spy(record, **kwargs):
        grid = launch_grid(record, **kwargs)
        calls.append(("grid", record["kernel"], kwargs, grid))
        return grid

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loader, "run_module", run_spy)
        patch.setattr(loader, "launch_grid", grid_spy)
        y = forward(x, z, w, EPS)
    (_, name, values), (_, kernel, kwargs, grid) = calls
    record = loader.MODULES[name]
    assert kernel == name and record["role"] in ("fwd", *loader.PACKED_ROLES)
    strides = [values[key] for key in ("heads", "x_tok", "x_head", "z_tok", "z_head")]
    assert loader.forward_packed(*strides) is want, (layout, strides)
    packed = want if STRIDED_CUT in record["grid_rule"] else True
    assert kwargs["packed"] is packed, (layout, name, kwargs)
    assert kwargs["arch"] == arch and kwargs["sm_count"] == sms, kwargs
    rows = values["rows"]
    assert grid == launch_grid(
        record, rows=rows, sm_count=sms, arch=arch, packed=packed
    ), (layout, name, grid)
    want_y = forward(x.contiguous(), z.contiguous(), w, EPS)
    assert_same_bits(y.reshape(want_y.shape), want_y, f"{layout}: y")


# Row counts of the tiny forward's host tests below its cut: a row, a few
# rows about a block of 8 and 16 rows, and the edges of sm_107a's gap
# (``FWD_TINY_GAP_ROWS``).
TINY_HOST_ROWS = (1, 7, 8, 9, 15, 16, 17, 32, 33)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_forward_module_runs_the_tiny_forward_up_to_its_cut(sm_count):
    """``forward_module`` (host only), on every architecture that routes the
    tiny forward, at ``TINY_HOST_ROWS``, a row short of its rows
    (``FWD_TINY_ROWS_PER_SM * SMs``) and at them: within ``_in_tiny_band``
    the tiny forward for packed rows (one head of token stride 128, and 48
    heads of token stride 48 * 128) and for one token of head stride 128
    (token stride 0, or one of the two inputs at ``H * 128``); in the gap
    (``FWD_TINY_GAP_ROWS``) and a row past them the module of the routes
    without it (``_forward_launch`` with ``tiny=False``).  At every row count
    ``fwd`` for 2 tokens or more of a gate or an ``x`` sliced from a
    projection, head-major rows, a token-stride-0 broadcast of both inputs or
    of one, heads of stride 256 and mixed strides, and for one token of head
    stride 256.  Its record names the role on the architecture, takes the
    arguments ``TINY_ARGUMENTS`` and launches ``ceil(rows / rows_per_cta)``
    CTAs at every row count."""
    H = 48
    row = H * D
    for arch in loader.ARCHES:
        last = _tiny_rows(arch, sm_count)
        if last == 0:
            assert "fwd_tiny" not in loader.ROUTES[arch], arch
            continue
        routes = loader.ROUTES[arch]
        tiny = routes["fwd_tiny"]
        record = loader.MODULES[tiny]
        assert record["role"] == "fwd_tiny" and arch in record["arches"], arch
        assert set(record["logical"].values()) == TINY_ARGUMENTS, arch
        per_cta = record["grid_rule"]["rows_per_cta"]
        rows_list = sorted({r for r in (*TINY_HOST_ROWS, last - 1, last) if r >= 1})
        for rows in (*rows_list, last + 1):
            inside = _in_tiny_band(arch, rows, sm_count)
            packed = tiny
            if not inside:
                packed = _forward_launch(arch, rows, sm_count, "packed", False)
            one = tiny if inside else routes["fwd"]
            for heads, strides, want in (
                (1, (D, D, D, D), packed),
                (rows, (0, D, 0, D), one),
                (rows, (0, D, rows * D, D), one),
                (rows, (rows * D, D, 0, D), one),
                (rows, (0, 2 * D, 0, 2 * D), routes["fwd"]),
                (rows, (0, D, 0, 2 * D), routes["fwd"]),
            ):
                got = loader.forward_module(arch, rows, heads, sm_count, *strides)
                assert got == want, (arch, sm_count, rows, heads, strides)
            if rows % H == 0:
                got = loader.forward_module(arch, rows, H, sm_count, row, D, row, D)
                assert got == packed, (arch, sm_count, rows, H)
            grid = loader.launch_grid(record, rows=rows, sm_count=sm_count, arch=arch)
            assert grid == (-(-rows // per_cta), 1, 1), (arch, rows, grid)
        for T in sorted({2, 3, max(2, last // H), max(2, last // H + 1)}):
            rows = T * H
            want = (
                tiny
                if _in_tiny_band(arch, rows, sm_count)
                else _forward_launch(arch, rows, sm_count, "packed", False)
            )
            got = loader.forward_module(arch, rows, H, sm_count, row, D, row, D)
            assert got == want, (arch, sm_count, T, "packed")
            for name, strides in (
                ("projection", (row, D, 2 * row + 16, D)),
                ("x_projection", (2 * row + 16, D, row, D)),
                ("head_major", (D, T * D, D, T * D)),
                ("broadcast", (0, D, 0, D)),
                ("x_broadcast", (0, D, row, D)),
                ("z_broadcast", (row, D, 0, D)),
                ("wide_heads", (2 * row, 2 * D, 2 * row, 2 * D)),
                ("mixed", (row, D, D, T * D)),
            ):
                got = loader.forward_module(arch, rows, H, sm_count, *strides)
                assert got == routes["fwd"], (arch, sm_count, T, name, got)


def test_the_tiny_plans_are_those_of_the_routes():
    # forward_module reads the plans built once at import (_FWD_PLANS): of
    # every architecture, the tiny forward and its cut where it routes one
    # with a cut above 0 (else None and 0), the forward of packed rows where
    # routed (else None), the fwd module where no long fwd module is routed
    # (else None) and the gap's rows (lo, hi] where it routes the tiny
    # forward (else 0, 0), so a patch of the cut or gap table alone routes
    # nothing
    want = {}
    for arch, routes in loader.ROUTES.items():
        cut = loader.FWD_TINY_ROWS_PER_SM.get(arch, 0)
        base = None if "fwd" + loader.LONG in routes else routes.get("fwd")
        if "fwd_tiny" in routes and cut > 0:
            lo, hi = loader.FWD_TINY_GAP_ROWS[arch]
            tiny, packed = routes["fwd_tiny"], routes.get("fwd_packed")
            want[arch] = (tiny, cut, packed, base, lo, hi)
        else:
            want[arch] = (None, 0, routes.get("fwd_packed"), base, 0, 0)
    assert loader._FWD_PLANS == want


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_backward_grid_is_at_most_one_wave(sm_count):
    """Each backward's grid (both weight-gradient dtypes and their long
    modules) is ``min(row blocks, ctas_per_sm * SMs)`` CTAs: one CTA a block
    up to one wave, one wave past it (where CTAs claim blocks).
    ``module_for_rows`` names the role's module for rows that are not packed,
    at any row count, and for packed rows the role's long module from its
    ``min_rows_per_sm * SMs`` rows."""
    for arch in loader.ARCHES:
        for role in ("bwd_bf16", "bwd_f32"):
            name = loader.route_module(arch, role)
            long = loader.route_module(arch, role + loader.LONG)
            start = loader.min_rows_per_sm(long, arch) * sm_count
            for module in (name, long):
                record = loader.MODULES[module]
                rule = record["grid_rule"]
                wave = rule["ctas_per_sm"] * sm_count
                edges = {*_grid_rows(rule, sm_count), start - 1, start, start + 1}
                for rows in sorted(edges):
                    got = loader.module_for_rows(arch, role, rows, sm_count)
                    assert got == name, (arch, role, rows, got)
                    got = loader.module_for_rows(arch, role, rows, sm_count, True)
                    assert got == (long if rows >= start else name), (arch, rows)
                    grid = loader.launch_grid(record, rows=rows, sm_count=sm_count)
                    blocks = -(-rows // rule["rows_per_block"])
                    want = blocks if blocks <= wave else wave
                    assert grid == (want, 1, 1), (arch, module, rows, grid)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_backward_module_runs_long_modules_on_packed_rows_only(sm_count):
    """``backward_module`` (host only): a backward role's long module from
    ``min_rows_per_sm * SMs`` rows (``LONG_MIN_ROWS_PER_SM``) when ``dy``,
    ``x`` and ``z`` all have head stride 128 and token stride ``H * 128``
    (also with one head, and at the kernels' row limit); the role's module a
    row fewer, for a single token (token stride 0), a token-stride-0
    broadcast of any one tensor, a gate or a ``dy`` sliced from a
    projection, head-major rows and a head stride other than 128.
    ``module_for_rows`` takes the packed flag; the forward has no long
    module."""
    H = 48
    row = H * D
    packed = (row, D, row, D, row, D)
    one_head = (D, D, D, D, D, D)
    for arch in loader.ARCHES:
        fwd = loader.ROUTES[arch]["fwd"]
        assert loader.module_for_rows(arch, "fwd", 2**31 - 1, sm_count, True) == fwd
        for role in ("bwd_bf16", "bwd_f32"):
            short = loader.ROUTES[arch][role]
            long = loader.ROUTES[arch][role + loader.LONG]
            assert loader.min_rows_per_sm(long, arch) == LONG_MIN_ROWS_PER_SM[arch]
            start = LONG_MIN_ROWS_PER_SM[arch] * sm_count
            for rows, heads, strides, want in (
                (start - 1, H, packed, short),
                (start, H, packed, long),
                (start + 1, H, packed, long),
                (2**31 - 1, H, packed, long),
                (1, 1, one_head, short),
                (start - 1, 1, one_head, short),
                (start, 1, one_head, long),
                (start, start, (0, D, 0, D, 0, D), short),
                (start, H, (0, D, row, D, row, D), short),
                (start, H, (row, D, 0, D, row, D), short),
                (start, H, (row, D, row, D, 0, D), short),
                (start, H, (row, D, row, D, 2 * row + 16, D), short),
                (start, H, (2 * row + 16, D, row, D, row, D), short),
                (start, H, (row, D, D, 65 * D, D, 65 * D), short),
                (start, H, (row, D, row, D, row, 2 * D), short),
            ):
                got = loader.backward_module(
                    arch, role, rows, heads, sm_count, *strides
                )
                assert got == want, (arch, role, rows, heads, strides)
            assert loader.module_for_rows(arch, role, start, sm_count) == short
            assert loader.module_for_rows(arch, role, start, sm_count, True) == long
            got = loader.module_for_rows(arch, role, start - 1, sm_count, True)
            assert got == short, (arch, role)


# The routes of an architecture's programs without the weight gradient taken
# away in turn (``ROUTES`` patched) to reach each step of ``nodw_module``.
NODW_DROPS = (
    (),
    (NODW_LONG,),
    (NODW_STRIDED_SMALL,),
    (NODW_STRIDED_PACKED,),
    NODW_128,
    (NODW_STRIDED,),
    (NODW_CLAIMED,),
    (NODW_STRIDED, NODW_CLAIMED),
)


def test_the_backward_plans_are_those_of_the_tables():
    # nodw_module reads the plans built once at import (_NODW_PLANS): they are
    # the plans the routes and the three cut tables give, for every routed
    # architecture and module
    assert loader._NODW_PLANS == {
        arch: {name: loader._nodw_plan(arch, name) for name in loader.MODULES}
        for arch in loader.ROUTES
    }


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_nodw_module_and_the_strided_grid(sm_count):
    """``nodw_module`` and the strided programs' grids (host only), with each
    of ``NODW_DROPS`` unrouted, for rows that are not packed and for packed
    rows (``packed``; omitted, not packed), a row around each cut: for the
    ``bwd_bf16`` module, on packed rows up to ``PACKED_SMALL_ROWS_PER_SM *
    SMs`` rows the build of packed rows only where it and the strided program
    are routed, else the strided program up to ``NODW_STRIDED_ROWS_PER_SM *
    SMs`` rows where routed (at every row count where the architecture has no
    cut; in its place its 128-thread build where routed, for packed rows and
    for rows of any layout up to ``NODW_ANY_LAYOUT_ROWS_PER_SM * SMs``), else
    the claimed program where routed, else the strided one, else the module
    itself; the build of packed rows only never for rows that are not packed;
    for the long module the long program without the weight gradient where
    routed, else the long module (run with ``weight_grad`` 0).  The
    256-thread strided program launches the ``bwd_bf16`` module's
    ``min(ceil(rows / 16), 4 * SMs)`` CTAs and the 128-thread builds
    ``min(ceil(rows / 8), 8 * SMs)`` CTAs (8-row blocks) at every row
    count."""
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        base, long = routes["bwd_bf16"], routes["bwd_bf16" + loader.LONG]
        per_sm = loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16")
        cut = None if per_sm is None else per_sm * sm_count
        wave = 4 * sm_count
        strided_wave = 8 * sm_count
        edges = {1, 7, 8, 9, 15, 16, 17, 16 * wave, 16 * wave + 1}
        edges |= {8 * strided_wave - 1, 8 * strided_wave, 8 * strided_wave + 1}
        if cut is not None:
            edges |= {cut - 1, cut, cut + 1}
        for small_table in (
            loader.PACKED_SMALL_ROWS_PER_SM,
            loader.NODW_ANY_LAYOUT_ROWS_PER_SM,
        ):
            small_cut = small_table[arch].get("bwd_bf16", 0) * sm_count
            edges |= {small_cut - 1, small_cut, small_cut + 1}
        rows_list = sorted(r for r in {*edges, *ISSUE_ROWS} if r >= 1)
        for drop in NODW_DROPS:
            table = _routes_without(arch, *drop)
            claimed = table.get(NODW_CLAIMED)
            strided = table.get(NODW_STRIDED)
            with pytest.MonkeyPatch.context() as patch:
                _patch_routes(patch, arch, table)
                for packed in (False, True):
                    for rows in rows_list:
                        role = None
                        if strided is not None:
                            role = _strided_role(arch, rows, sm_count, packed, table)
                        if role == NODW_STRIDED_SMALL or (
                            role is not None and cut is not None and rows <= cut
                        ):
                            want = table[role]
                        else:
                            want = claimed or (role and table[role]) or base
                        what = (arch, drop, packed, rows)
                        got = loader.nodw_module(
                            arch, base, rows, sm_count, packed=packed
                        )
                        assert got == want, (*what, got)
                        if not packed:
                            assert got != table.get(NODW_STRIDED_SMALL), what
                            got = loader.nodw_module(arch, base, rows, sm_count)
                            assert got == want, (*what, got)
                        got = loader.nodw_module(
                            arch, long, rows, sm_count, packed=packed
                        )
                        assert got == table.get(NODW_LONG, long), what
        twin = loader.MODULES[base]
        for role, rule in NODW_STRIDED_GRID_RULES.items():
            if role not in routes:
                continue
            record = loader.MODULES[routes[role]]
            per_block, ctas = rule["rows_per_block"], rule["ctas_per_sm"] * sm_count
            for rows in rows_list:
                blocks = -(-rows // per_block)
                got = loader.launch_grid(record, rows=rows, sm_count=sm_count)
                assert got == (min(blocks, ctas), 1, 1), (arch, role, rows, got)
                if role == NODW_STRIDED:
                    want = loader.launch_grid(twin, rows=rows, sm_count=sm_count)
                    assert got == want, (arch, rows, got, want)
        for rows in rows_list:
            got = loader.launch_grid(twin, rows=rows, sm_count=sm_count)
            assert got == (min(-(-rows // 16), wave), 1, 1), (arch, rows, got)


def test_the_small_row_plans_are_those_of_the_routes():
    # dw_module reads the plans built once at import (_DW_PLANS): they are
    # the plans the routes and the small-row programs' rules give, for every
    # routed architecture and module
    assert loader._DW_PLANS == {
        arch: {name: loader._dw_plan(arch, name) for name in loader.MODULES}
        for arch in loader.ROUTES
    }


# Row counts around the cuts of the programs of one CTA and of one cluster,
# and far past them.
SMALL_DW_ROWS = (1, 2, 15, 16, 17, 48, 128, 129, 255, 256, 257, 4096, 2**31 - 1)


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_dw_module_and_the_small_row_grids(sm_count):
    """``dw_module`` and the small-row grids (host only): a backward role's
    module gives way to its program of one CTA up to 16 rows and, with
    ``clusters`` (the default), to its program of one cluster up to 256 rows
    (``_small_dw_module``); every other module is kept at every row count.
    The program of one CTA launches one CTA and that of one cluster its 16
    CTAs at every row count and SM count; from 4 SMs a role's module
    launches one CTA a 16-row block up to 256 rows."""
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        for name, record in loader.MODULES.items():
            role = record["role"]
            for rows in SMALL_DW_ROWS:
                for clusters in (True, False):
                    want = name
                    if role in DW_CALL_ROLES:
                        want = _small_dw_module(arch, role, rows, clusters)
                    got = loader.dw_module(arch, name, rows, clusters)
                    assert got == want, (arch, name, rows, clusters, got)
                got = loader.dw_module(arch, name, rows)
                assert got == loader.dw_module(arch, name, rows, True), (arch, rows)
            if role in loader.CLUSTER_ROLES:
                ctas = SMALL_DW_RULES[_small_suffix(role)][0]
                for rows in SMALL_DW_ROWS:
                    got = loader.launch_grid(record, rows=rows, sm_count=sm_count)
                    assert got == (ctas, 1, 1), (arch, name, rows, got)
        if sm_count < 4:
            continue
        for role in ("bwd_bf16", "bwd_f32"):
            record = loader.MODULES[routes[role]]
            for rows in range(1, 257):
                got = loader.launch_grid(record, rows=rows, sm_count=sm_count)
                assert got == (-(-rows // 16), 1, 1), (arch, role, rows, got)


def test_device_runs_clusters_reads_the_sm_count_and_the_name(monkeypatch):
    """``device_runs_clusters``' rule (the uncached function, the device's
    properties patched): a device of at least ``CLUSTER_MIN_SMS`` (16) SMs
    runs the cluster programs unless it is a MIG instance (``MIG`` in its
    name); a device of fewer SMs does not."""
    assert loader.CLUSTER_MIN_SMS == 16
    rule = loader.device_runs_clusters.__wrapped__
    for count, name, want in (
        (148, "NVIDIA B200", True),
        (16, "NVIDIA B200", True),
        (15, "NVIDIA B200", False),
        (1, "NVIDIA B200", False),
        (16, "NVIDIA B200 MIG 1g.23gb", False),
        (148, "NVIDIA B200 MIG 7g.180gb", False),
    ):
        properties = SimpleNamespace(multi_processor_count=count, name=name)
        monkeypatch.setattr(
            torch.cuda, "get_device_properties", lambda index, p=properties: p
        )
        assert rule(0) is want, (count, name)


# The strides (in elements) of ``dy``, ``x`` and ``z`` of 48 heads the
# backward reads as packed rows (head stride 128, token stride ``48 * 128``),
# and of layouts it reads otherwise: a gate and ``dy`` sliced from
# ``[T, 2 * 48 * 128 + 16]`` projections, ``dy`` alone sliced so, head-major
# ``x`` and ``z``, and a token-stride-0 broadcast of ``x`` and ``z``.
ROUTE_H = 48
PACKED_STRIDES = (ROUTE_H * D, D, ROUTE_H * D, D, ROUTE_H * D, D)
OTHER_STRIDES = {
    "projection": (
        2 * ROUTE_H * D + 16,
        D,
        ROUTE_H * D,
        D,
        2 * ROUTE_H * D + 16,
        D,
    ),
    "dy_projection": (2 * ROUTE_H * D + 16, D, ROUTE_H * D, D, ROUTE_H * D, D),
    "head_major": (ROUTE_H * D, D, D, 65 * D, D, 65 * D),
    "broadcast": (ROUTE_H * D, D, 0, D, 0, D),
}


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_strided_programs_follow_the_layout(sm_count):
    """On every architecture (host only, ``backward_packed``,
    ``backward_module`` and ``nodw_module`` as the backward calls them), a
    call without the weight gradient below the strided cut (where the
    architecture has none, up to a row short of the long modules' first row
    count), a row around each small-row cut, runs on packed rows the build of
    packed rows only (``bwd_bf16_nodw_strided_packed_small``) up to
    ``PACKED_SMALL_ROWS_PER_SM`` rows per SM and the 128-thread build
    (``bwd_bf16_nodw_strided_packed``) past it, and on each of
    ``OTHER_STRIDES`` the 128-thread build up to
    ``NODW_ANY_LAYOUT_ROWS_PER_SM`` rows per SM and the 256-thread one
    (``bwd_bf16_nodw_strided``) past it (no small-row cut: never the
    small-row program); without a strided cut, rows that are not packed run
    the 256-thread one at every row count past the cut of any layout; past a
    strided cut both run the claimed program."""
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        per_sm = loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16")
        start = LONG_MIN_ROWS_PER_SM[arch] * sm_count
        last = start - 1 if per_sm is None else per_sm * sm_count
        small = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0) * sm_count
        any_layout = loader.NODW_ANY_LAYOUT_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        any_layout *= sm_count
        below = {1, 7, 8, 9, 16, 17, ROUTE_H, 65 * ROUTE_H, last // 2, last}
        below |= {small, small + 1, any_layout, any_layout + 1}
        below = {rows for rows in below if 1 <= rows <= last}
        cases = [(PACKED_STRIDES, True)]
        cases += [(strides, False) for strides in OTHER_STRIDES.values()]
        for strides, packed in cases:
            assert loader.backward_packed(ROUTE_H, *strides) is packed, strides
            rows_list = set(below)
            if not packed and per_sm is None:
                rows_list |= {start, start + 1, *ISSUE_ROWS}
            for rows in sorted(rows_list):
                if packed:
                    role = NODW_STRIDED_SMALL if rows <= small else NODW_STRIDED_PACKED
                else:
                    role = NODW_STRIDED_PACKED if rows <= any_layout else NODW_STRIDED
                block = loader.MODULES[routes[role]]["launch"]["block"]
                assert block == (NODW_STRIDED_THREADS[role], 1, 1), (arch, role)
                name = loader.backward_module(
                    arch, "bwd_bf16", rows, ROUTE_H, sm_count, *strides
                )
                assert name == routes["bwd_bf16"], (arch, rows, strides, name)
                got = loader.nodw_module(arch, name, rows, sm_count, packed=packed)
                assert got == routes[role], (arch, rows, strides, got)
            if per_sm is not None:
                rows = per_sm * sm_count + 1
                name = loader.backward_module(
                    arch, "bwd_bf16", rows, ROUTE_H, sm_count, *strides
                )
                got = loader.nodw_module(arch, name, rows, sm_count, packed=packed)
                assert got == routes[NODW_CLAIMED], (arch, rows, strides, got)


def test_one_token_is_packed_rows_only_where_a_small_cut_is():
    """A call without the weight gradient passes one token of head stride 128
    as packed rows (token stride ``H * 128``) on exactly the architectures
    where the build of packed rows only has a cut
    (``PACKED_SMALL_ROWS_PER_SM``): sm_100a, sm_103a and sm_107a (from
    version 0.2.8; in version 0.2.7 sm_100a and sm_103a), at every row count
    where a strided cut is (sm_100a, sm_103a) and up to the cut of that
    build where none is (sm_107a, 192 rows per SM; past it the 256-thread
    program of version 0.2.7)."""
    limits = pkg._ONE_TOKEN_PACKED_ROWS_PER_SM
    assert limits == {"sm_100a": 2**31, "sm_103a": 2**31, "sm_107a": 192}
    for arch in loader.ARCHES:
        small = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        for sm_count in GRID_SM_COUNTS:
            for per_sm in (1, small, small + 1, LONG_MIN_ROWS_PER_SM[arch]):
                for rows in (per_sm * sm_count - 1, per_sm * sm_count):
                    if rows < 1:
                        continue
                    want = _one_token_packed(arch, rows, sm_count)
                    got = rows <= limits.get(arch, 0) * sm_count
                    assert got == want, (arch, sm_count, rows)


def _nodw_program_0_2_6(arch, rows, sm_count, packed):
    """The program version 0.2.6 runs for a call without the weight gradient
    of the ``bwd_bf16`` module (its ``nodw_module``, spelled out): the
    strided program up to the strided cut (on packed rows its 128-thread
    build), else the claimed one where routed, else the strided one."""
    routes = loader.ROUTES[arch]
    strided = routes.get(NODW_STRIDED)
    if packed and strided is not None:
        strided = routes.get(NODW_STRIDED_PACKED, strided)
    cut = loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16")
    if strided is not None and cut is not None and rows <= cut * sm_count:
        return strided
    return routes.get(NODW_CLAIMED) or strided or routes["bwd_bf16"]


@pytest.mark.parametrize("sm_count", GRID_SM_COUNTS)
def test_small_row_cuts_change_only_the_rows_up_to_them(sm_count):
    """``nodw_module`` (host only) names the program of version 0.2.6 for
    every row count past the small-row cut of the layout (packed rows:
    ``PACKED_SMALL_ROWS_PER_SM``; other layouts:
    ``NODW_ANY_LAYOUT_ROWS_PER_SM``), so on an architecture without a cut
    for every row count, and up to the cut the build of packed rows only
    (packed rows) or the 128-thread build (other layouts)."""
    for arch in loader.ARCHES:
        routes = loader.ROUTES[arch]
        base = routes["bwd_bf16"]
        cut = loader.NODW_STRIDED_ROWS_PER_SM[arch].get("bwd_bf16")
        small = loader.PACKED_SMALL_ROWS_PER_SM[arch].get("bwd_bf16", 0) * sm_count
        any_layout = loader.NODW_ANY_LAYOUT_ROWS_PER_SM[arch].get("bwd_bf16", 0)
        any_layout *= sm_count
        rows_list = {1, 7, 8, 9, ROUTE_H, 65 * ROUTE_H, *ISSUE_ROWS}
        rows_list |= {small, small + 1, any_layout, any_layout + 1}
        if cut is not None:
            rows_list |= {cut * sm_count, cut * sm_count + 1}
        for packed in (False, True):
            first = small if packed else any_layout
            for rows in sorted(r for r in rows_list if r >= 1):
                got = loader.nodw_module(arch, base, rows, sm_count, packed=packed)
                if rows > first:
                    want = _nodw_program_0_2_6(arch, rows, sm_count, packed)
                else:
                    want = routes[NODW_STRIDED_SMALL if packed else NODW_STRIDED_PACKED]
                assert got == want, (arch, packed, rows, got)


def test_backward_module_tells_the_layout_as_backward_packed(monkeypatch):
    """``backward_module`` (host only) tells ``module_for_rows`` the rows are
    packed, by its own copy of the kernel's test, exactly when
    ``backward_packed`` does: on packed strides, on each of
    ``OTHER_STRIDES``, on packed strides with any one of the six set to 0,
    moved by a row or set to 256, and on one head."""
    told = []

    def record(arch, role, rows, sm_count, packed=False):
        told.append(packed)
        return loader.ROUTES[arch][role]

    monkeypatch.setattr(loader, "module_for_rows", record)
    row = ROUTE_H * D
    cases = [(ROUTE_H, PACKED_STRIDES)]
    cases += [(ROUTE_H, strides) for strides in OTHER_STRIDES.values()]
    for i, stride in enumerate(PACKED_STRIDES):
        for value in (0, stride + row, 2 * D):
            strides = list(PACKED_STRIDES)
            strides[i] = value
            cases.append((ROUTE_H, tuple(strides)))
    cases += [(1, (D, D, D, D, D, D)), (1, (0, D, 0, D, 0, D))]
    seen = set()
    for heads, strides in cases:
        want = loader.backward_packed(heads, *strides)
        seen.add(want)
        told.clear()
        loader.backward_module("sm_100a", "bwd_bf16", 4096, heads, 148, *strides)
        assert told == [want], (heads, strides, told)
    assert seen == {True, False}


# The layouts of a backward without the weight gradient on 65 tokens of 48
# heads (one token for ``single_token``; ``dy_projection``: ``dy`` alone
# sliced from a projection; ``broadcast``: token-stride-0 ``x`` and ``z``),
# and whether the call passes them as packed rows (misaligned ones once
# copied by the host; one token of head stride 128 with the token stride
# ``H * 128`` where ``_one_token_packed`` holds).
NODW_ENTRY_PACKED = {
    "contiguous": True,
    "flat": True,
    "misaligned": True,
    "single_token": True,
    "projection": False,
    "head_major": False,
    "dy_projection": False,
    "broadcast": False,
}


@pytest.mark.parametrize("layout", NODW_ENTRY_PACKED)
def test_nodw_backward_tells_nodw_module_the_layout(layout, launches):
    """A backward with ``compute_grad_weight=False`` tells ``nodw_module`` the
    rows are packed, by its own copy of the kernel's test, exactly for the
    layouts the kernel reads as packed rows and for one token of head stride
    128 where ``_one_token_packed`` holds, and launches the program
    ``nodw_module`` names; ``dx`` and ``dz`` are the bits of the call that
    computes a BF16 weight gradient."""
    arch, sms = _device()
    H, T = ROUTE_H, 65
    if layout == "single_token":
        x, z, w, dy = make(1, H, seed=46)
    elif layout == "broadcast":
        x, z, w, _ = make(1, H, seed=46)
        x, z = x.expand(T, H, D), z.expand(T, H, D)
        dy = make(T, H, seed=47)[3]
    elif layout == "dy_projection":
        x, z, w, _ = make(T, H, seed=46)
        dy = make(T, H, layout="projection", seed=47)[3]
    else:
        x, z, w, dy = make(T, H, layout=layout, seed=46)
    rows = x.numel() // D
    assert rows < LONG_MIN_ROWS_PER_SM[arch] * sms
    told = []
    nodw_module = pkg.nodw_module

    def record(arch, name, rows, sm_count, packed=False):
        told.append(packed)
        return nodw_module(arch, name, rows, sm_count, packed)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "nodw_module", record)
        launches.clear()
        dx, dz, dw = backward(dy, x, z, w, EPS, compute_grad_weight=False)
    packed = NODW_ENTRY_PACKED[layout] and (
        layout != "single_token" or _one_token_packed(arch, rows, sms)
    )
    assert dw is None
    assert told == [packed], (layout, told)
    name = loader.ROUTES[arch]["bwd_bf16"]
    want = loader.nodw_module(arch, name, rows, sms, packed)
    assert launches == [want], (layout, launches)
    want_dx, want_dz, _ = backward(dy, x, z, w, EPS)
    assert_same_bits(dx, want_dx, f"{layout}: dx")
    assert_same_bits(dz, want_dz, f"{layout}: dz")
