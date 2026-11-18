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

Tests for the Cake fused BF16 RMSNorm training kernels with the
non-deterministic weight gradient (``cake_pplx.cake_rmsnorm_nondet_train``;
SM100 / SM103 / SM107).

Every floating kernel output -- ``y``, ``rstd``, ``h_new``, ``dx``, ``dw`` -- is
judged against the naive PyTorch BF16 implementation, with no tolerance: its
error against an FP64 oracle of the same math, over 32 independent random
inputs of each case, in the L2 and the max norm, median and largest, is at most
1.005 times the naive's, and 0 wherever the naive's is 0
(``tests/norm/test_helpers/cake_rmsnorm_train_reference.py``,
``tests/cake/test_helpers/cake_naive.py``).  ``h_new`` is also the correctly
rounded BF16 sum.  The tests never compare this package with the deterministic
one.

A backward window runs one of two forms (the registry's records for the call's
token count): one kernel or a deterministic pair rendered into this package.
The one kernel is the single kernel (``claim_sum_f32`` with a nonzero
``sum_bytes``), which adds the CTAs' FP32 ``dw`` sums in a
scheduling-dependent order; the part-row single kernel (``claim_part_f32``),
which keeps one FP32 partial row per CTA and folds the rows in a
scheduling-dependent order; or a fixed-order kernel (``claim_sum_f32`` with
``sum_bytes`` 0): the one-cluster kernel or a column-task program.  Every case derives its form from its own window (the records of its
architecture, width, residual flag, weight dtype and token count, read from
``ROUTES`` and the select rules) and checks that the call launched exactly
those records.

The matrix covers token counts from 0 to 65536 (the deterministic contract's
rows, the observed training counts and every program window edge +-1 of the
registry), both eps values where the contract names both, 1e-3 / 1e3 scales,
zero inputs and a row of large values, row-strided inputs, the residual-add
fusion and the FP32-weight routes (also combined), the ``[..., H]`` per-head
views, repeated launches (``dx`` bitwise stable, every launch's ``dw`` judged), a
workspace reused across token counts, windows and weight dtypes with its
documented rest state checked after every call that reuses one, two streams,
CUDA Graph capture and replay, the autograd wrapper (gradient accumulation,
saved tensors, non-reentrant checkpointing, the
``torch.use_deterministic_algorithms`` check made when the backward runs) and
argument validation.  The build manifest's per-architecture CUDA floors and
flags are checked without a device.

This suite exercises the backend packages alone, through
``tests.cake.test_helpers.cake_package``; the facade is
``test_cake_rmsnorm_nondet_train_api.py``'s.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from typing import Optional

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.norm.test_helpers import cake_rmsnorm_train_reference as ref

pkg = cake_package("cake_rmsnorm_nondet_train")
loader = cake_package("cake_rmsnorm_nondet_train", "cake_jit")
cake_build = cake_package("cake_rmsnorm_nondet_train", "cake_build")
forward = pkg.cake_rmsnorm_nondet_train_forward
backward = pkg.cake_rmsnorm_nondet_train_backward
workspace_for = pkg.cake_rmsnorm_nondet_train_backward_workspace
rmsnorm = pkg.cake_rmsnorm_nondet

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-6),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
# Widths whose contract names both eps values: their contract cases run at each.
EPS_VARIANTS = {4096: (1e-5, 1e-6), 1536: (1e-5, 1e-6)}
# The deterministic package's row counts (tile edges, 2^n +- 1, the two contract
# training shapes) and the wider counts of this suite.
ROWS = (1, 2, 7, 63, 64, 65, 127, 128, 129, 255, 257, 1000, 4096, 16172, 16231, 16385)
LARGE_ROWS = (65536,)
# Further counts of the H=1536 / H=4096 contract and the observed training counts.
CONTRACT_ROWS = {
    4096: (1023, 1025, 4097, 14928, 16263, 16384, 32769, 65536),
    1536: (1023, 1025, 4097, 14928, 15566, 16263, 16384, 32769, 65536),
}
OBSERVED_ROWS = (14928, 15288, 15566, 15590, 16114, 16132, 16164, 16263)
SCALES = (1e-3, 1e3)
W_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
REPEATS = 20
SEED = 8066
# The CUDA toolkit floor of every exported architecture (``build.json``
# ``min_cuda``) and the exact code it is compiled for.
MIN_CUDA = {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.5"}
GENCODE = {
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
    "sm_103a": "-gencode=arch=compute_103a,code=sm_103a",
    "sm_107a": "-gencode=arch=compute_107a,code=sm_107a",
}
ACCURACY_FLAGS = ("--prec-div=true", "--prec-sqrt=true", "--ftz=false")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_nondet_train_supported(_device())
    )


def _arch() -> str:
    return loader.arch_for_capability(_capability())


@pytest.fixture(autouse=True)
def device_or_skip(request):
    """Every test needs a supported device except the ``cpu_only`` ones."""
    if request.node.get_closest_marker("cpu_only") is None and not _supported():
        pytest.skip(
            "cake_rmsnorm_nondet_train needs an SM100 / SM103 / SM107 device with exported routes"
        )


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


# Every module launch of this package in the running test (name), and the FP32
# ``dw`` buffer each backward module wrote (the kernel's dw before the autograd
# wrapper casts it to the weight dtype).
_LAUNCHED: list[str] = []
_DW_BUFFERS: list[torch.Tensor] = []


@pytest.fixture(autouse=True)
def record_launches(monkeypatch: pytest.MonkeyPatch):
    run_module = loader.run_module

    def recording(name, values, *, rows, device_index):
        _LAUNCHED.append(name)
        if loader.MODULES[name]["role"].startswith("bwd"):
            _DW_BUFFERS.append(values["dw"])
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", recording)
    _LAUNCHED.clear()
    _DW_BUFFERS.clear()
    yield


# --------------------------------------------------------------------------- the registry, per case


def _w_token(w_dtype: torch.dtype) -> str:
    return "f32" if w_dtype == torch.float32 else "bf16"


def _routes(hidden: int, kind: str, residual: bool, w_dtype: str = "bf16") -> bool:
    return loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype=w_dtype,
    )


def _skip_unless_route(hidden: int, residual: bool = False, w_dtype: str = "bf16"):
    if not all(_routes(hidden, kind, residual, w_dtype) for kind in ("fwd", "bwd")):
        pytest.skip(
            f"hidden={hidden} residual={residual} w={w_dtype} has no exported route here"
        )


def _admits(rule: Optional[dict], rows: int) -> bool:
    if rule is None:
        return True
    assert rule["kind"] == "rows_between", rule
    return rows > int(rule["above"]) and (
        rule["at_most"] is None or rows <= int(rule["at_most"])
    )


def _window_records(
    hidden: int, kind: str, residual: bool, w_dtype: str, rows: int
) -> list[str]:
    """The records one call of ``rows`` tokens launches, in route order, read
    from ``ROUTES`` and each record's select rule (independent of the host's
    ``selected_modules``)."""
    key = loader.route_key(hidden, kind, residual, w_dtype)
    return [
        name
        for name in loader.ROUTES[_arch()][key]
        if _admits(loader.MODULES[name]["select_rule"], rows)
    ]


def _form(names: list[str]) -> str:
    """``single`` (one ``claim_sum_f32`` kernel, or one ``claim_part_f32``
    part-row single kernel) or ``pair`` (a deterministic pair: ``chunk_partials``
    main kernel, then its dw reduce) of one backward window's records."""
    return ref.form(loader, names)


def _run_backward(*args, **kwargs):
    """``((dx, dw), the records this call launched)``."""
    start = len(_LAUNCHED)
    result = backward(*args, **kwargs)
    return result, list(_LAUNCHED[start:])


def _workspace_at_rest(ws: torch.Tensor) -> bool:
    """The documented rest state after every call of any layout: every byte
    before the pair partial rows (``PAIR_ROWS_OFFSET``: the pair counters, the
    single kernel's counters and FP32 sums) is zero, so the buffer serves the
    next call of any layout; the pair partial rows are scratch."""
    torch.cuda.synchronize()
    return not bool(ws[: min(loader.PAIR_ROWS_OFFSET, ws.numel())].any())


def _bytes(rows: int, hidden: int, residual: bool = False) -> int:
    return pkg.cake_rmsnorm_nondet_train_backward_workspace_bytes(
        rows, hidden, _device(), residual=residual
    )


# --------------------------------------------------------------------------- inputs and references


def _seed(*keys) -> int:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return seed


def _generator(*keys, draw: int = 0) -> torch.Generator:
    """The generator of draw ``draw`` of the case keyed ``keys`` (draw 0 is the case's own input)."""
    return torch.Generator(device=_device()).manual_seed(
        ref.draw_seed(_seed(*keys), draw)
    )


def _rows(
    rows: int,
    hidden: int,
    *,
    pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    generator: torch.Generator,
) -> torch.Tensor:
    """BF16 ``[rows, hidden]`` view with ``stride(0) == hidden + pad`` (``pad="2H"``: ``2 * hidden``)."""
    pad = hidden if pad == "2H" else int(pad)
    shape = (max(rows, 1), hidden + pad)
    if zeros:
        base = torch.zeros(shape, dtype=torch.bfloat16, device=_device())
    else:
        base = (
            scale
            * torch.randn(
                shape, dtype=torch.float32, device=_device(), generator=generator
            )
        ).to(torch.bfloat16)
    return base[:rows, :hidden]


def _weight(hidden: int, generator: torch.Generator, dtype=torch.bfloat16):
    """``1 + 0.1 randn`` drawn in BF16; an FP32 weight holds that BF16 draw (the
    naive tier reads it exactly, ``ref.naive_weight``)."""
    w = 1.0 + 0.1 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    return w.to(torch.bfloat16).to(dtype)


def _h_new_is_the_rounded_sum(h_new, x, u, label: str) -> None:
    """``h_new`` is ``x + u`` rounded once to BF16 (the float64 sum of two BF16 values is exact)."""
    assert torch.equal(h_new, (x.double() + u.double()).to(torch.bfloat16)), (
        f"h_new {label} is not the BF16-rounded sum"
    )


@dataclasses.dataclass
class Case:
    x: torch.Tensor
    u: Optional[torch.Tensor]
    g: torch.Tensor
    g_h: Optional[torch.Tensor]
    w: torch.Tensor
    eps: float
    rstd: torch.Tensor
    normalized: torch.Tensor
    y: torch.Tensor
    dx: torch.Tensor
    dw: torch.Tensor
    records: list[str]
    form: Optional[str]


def _case_inputs(
    hidden: int,
    rows: int,
    *,
    residual=False,
    w_dtype=torch.bfloat16,
    pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    spike: bool = False,
    key: int = 0,
    draw: int = 0,
):
    """``(x, u, g, g_h, w)`` of draw ``draw`` of the case (draw 0 is the case's own input)."""
    pad_key = hidden if pad == "2H" else int(pad)
    generator = _generator(
        hidden,
        rows,
        residual,
        pad_key,
        w_dtype == torch.float32,
        int(scale * 1e6) % 9973,
        zeros,
        spike,
        key,
        draw=draw,
    )
    x = _rows(rows, hidden, pad=pad, scale=scale, zeros=zeros, generator=generator)
    if spike and rows > 0:
        row = min(rows - 1, 3)
        x[row] = (x[row].float() * 1e3).to(torch.bfloat16)
        x[row, min(hidden - 1, 17)] = 1e4
    u = (
        _rows(rows, hidden, scale=scale, zeros=zeros, generator=generator)
        if residual
        else None
    )
    g = _rows(rows, hidden, pad=pad, generator=generator)
    g_h = _rows(rows, hidden, generator=generator) if residual else None
    w = _weight(hidden, generator, w_dtype)
    return x, u, g, g_h, w


def _launch(
    hidden: int, rows: int, eps: float, inputs, *, residual, w_token: str, label: str
) -> Case:
    """Forward and backward of one draw's ``inputs``: the shapes, ``h_new`` and the launched records checked."""
    x, u, g, g_h, w = inputs
    start = len(_LAUNCHED)
    outputs = forward(x, w, eps, residual=u)
    fwd_launched = _LAUNCHED[start:]
    y, rstd = outputs[0], outputs[1]
    assert tuple(y.shape) == (rows, hidden) and y.is_contiguous()
    assert rstd.dtype == torch.float32 and tuple(rstd.shape) == (rows,)
    normalized = outputs[2] if residual else x
    if residual:
        _h_new_is_the_rounded_sum(normalized, x, u, label)
    (dx, dw), records = _run_backward(g, normalized, w, rstd, g_residual=g_h)
    assert tuple(dx.shape) == (rows, hidden) and dx.is_contiguous()
    assert dw.dtype == torch.float32 and dw.is_contiguous(), label
    if rows == 0:
        assert fwd_launched == [] and records == []
        assert torch.equal(dw, torch.zeros_like(dw))
        return Case(x, u, g, g_h, w, eps, rstd, normalized, y, dx, dw, [], None)
    assert fwd_launched == _window_records(hidden, "fwd", residual, w_token, rows), (
        f"forward {label} launched {fwd_launched}"
    )
    expected = _window_records(hidden, "bwd", residual, w_token, rows)
    assert records == expected, (
        f"backward {label} launched {records}, window {expected}"
    )
    return Case(
        x, u, g, g_h, w, eps, rstd, normalized, y, dx, dw, records, _form(expected)
    )


def _judge(check: ref.Draws, draw: int, case: Case, *, outputs=None) -> None:
    """Judge ``case``'s outputs (``outputs``: names of ``y``, ``rstd``, ``h_new``,
    ``dx``, ``dw``; default all) as draw ``draw`` of ``check``."""
    names = set(outputs or ("y", "rstd", "h_new", "dx", "dw"))
    forward_outputs = {"y": case.y, "rstd": case.rstd}
    if case.u is not None:
        forward_outputs["h_new"] = case.normalized
    forward_outputs = {k: v for k, v in forward_outputs.items() if k in names}
    if forward_outputs:
        check.forward(draw, forward_outputs, case.x, case.w, case.eps, residual=case.u)
    backward_outputs = {
        k: v for k, v in (("dx", case.dx), ("dw", case.dw)) if k in names
    }
    if backward_outputs:
        check.backward(
            draw,
            backward_outputs,
            case.g,
            case.normalized,
            case.w,
            case.rstd,
            case.eps,
            g_h=case.g_h,
        )


def _check(
    hidden: int,
    rows: int,
    *,
    record,
    tag: str = "",
    residual=False,
    w_dtype=torch.bfloat16,
    pad=0,
    eps: Optional[float] = None,
    scale: float = 1.0,
    zeros: bool = False,
    spike: bool = False,
    key: int = 0,
) -> Case:
    """Forward and backward of one case over its draws: every launch runs exactly
    its window's records, and every output meets the naive BF16 tier
    (``ref.Draws``; verdicts recorded with ``record``, ``tag`` naming the outputs
    of one case among several of a test).  Returns the case's own draw (draw 0)."""
    eps = EPS[hidden] if eps is None else eps
    w_token = _w_token(w_dtype)
    label = (
        f"hidden={hidden} rows={rows} residual={residual} w={w_token} pad={pad} "
        f"eps={eps} scale={scale} zeros={zeros} spike={spike}"
    )
    check = ref.Draws(label, prefix=tag)
    first = None
    for draw in range(check.draws):
        inputs = _case_inputs(
            hidden,
            rows,
            residual=residual,
            w_dtype=w_dtype,
            pad=pad,
            scale=scale,
            zeros=zeros,
            spike=spike,
            key=key,
            draw=draw,
        )
        case = _launch(
            hidden, rows, eps, inputs, residual=residual, w_token=w_token, label=label
        )
        if rows == 0:
            return case
        _judge(check, draw, case)
        if first is None:
            first = case
    check.check(record)
    return first


# --------------------------------------------------------------------------- accuracy matrix


@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", ROWS + LARGE_ROWS)
def test_forward_backward_match_fp64(
    hidden: int, rows: int, w: str, record_property
) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, record=record_property, w_dtype=W_DTYPES[w])


def _contract_cases():
    for hidden in HIDDEN_SIZES:
        counts = sorted(set(CONTRACT_ROWS.get(hidden, ())) | set(OBSERVED_ROWS))
        for eps in EPS_VARIANTS.get(hidden, (EPS[hidden],)):
            for w in W_DTYPES if hidden in EPS_VARIANTS else ("bf16",):
                for rows in counts:
                    yield pytest.param(
                        hidden, rows, eps, w, id=f"h{hidden}-T{rows}-eps{eps:g}-{w}"
                    )


@pytest.mark.parametrize(("hidden", "rows", "eps", "w"), list(_contract_cases()))
def test_contract_and_observed_rows(
    hidden: int, rows: int, eps: float, w: str, record_property
) -> None:
    """The H=1536 / H=4096 contract rows and the observed training token counts
    at both contract eps values (both weight dtypes); the observed counts at
    every other width."""
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, record=record_property, w_dtype=W_DTYPES[w], eps=eps)


@pytest.mark.parametrize(
    ("hidden", "eps"),
    [(hidden, eps) for hidden, variants in EPS_VARIANTS.items() for eps in variants],
)
@pytest.mark.parametrize("rows", (7, 16231))
@pytest.mark.parametrize("scale", (1.0, *SCALES))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_eps_variants(
    hidden: int, eps: float, rows: int, scale: float, w: str, record_property
) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    _check(
        hidden, rows, record=record_property, eps=eps, scale=scale, w_dtype=W_DTYPES[w]
    )


def _route_params():
    for hidden in HIDDEN_SIZES:
        for w in W_DTYPES:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_DTYPES:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


def _edge_rows(hidden: int, residual: bool, w: str) -> list[int]:
    """Both neighbours of every program-window edge of the route's forward and
    backward (``program_windows``), and one row."""
    edges = set()
    for kind in ("fwd", "bwd"):
        for above, at_most, _library in loader.program_windows(
            _arch(), hidden, kind, residual, w
        ):
            edges.update(bound for bound in (above, at_most) if bound)
    return sorted({1, *(row for edge in edges for row in (edge - 1, edge, edge + 1))})


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_route_params()))
def test_program_window_edges(
    hidden: int, residual: bool, w: str, record_property
) -> None:
    """Every window edge +-1 of the route meets the naive BF16 tier, every call
    launches its window's records, and every forward and backward program of the
    route runs for one of those counts."""
    _skip_unless_route(hidden, residual, w)
    launched = set()
    forms = []
    for rows in _edge_rows(hidden, residual, w):
        case = _check(
            hidden,
            rows,
            record=record_property,
            tag=f"T={rows} ",
            residual=residual,
            w_dtype=W_DTYPES[w],
            key=5,
        )
        launched.update(case.records)
        launched.update(_window_records(hidden, "fwd", residual, w, rows))
        forms.append(f"{rows}:{case.form}")
    for kind in ("fwd", "bwd"):
        names = set(loader.ROUTES[_arch()][loader.route_key(hidden, kind, residual, w)])
        assert names <= launched, f"{kind} programs never launched: {names - launched}"
    print(f"EDGES h={hidden} res={residual} w={w} {' '.join(forms)}")


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("pad", (64, "2H"))
@pytest.mark.parametrize("rows", (7, 257, 16385))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_row_strided_inputs(
    hidden: int, pad, rows: int, w: str, record_property
) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    _check(hidden, rows, record=record_property, pad=pad, w_dtype=W_DTYPES[w])


@pytest.mark.parametrize("rows", (1, 7, 129, 257, 4096, 16231, 16385, 65536))
@pytest.mark.parametrize("pad", (0, 64))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_residual_fusion(rows: int, pad, w: str, record_property) -> None:
    _skip_unless_route(6144, residual=True, w_dtype=w)
    _check(
        6144, rows, record=record_property, residual=True, pad=pad, w_dtype=W_DTYPES[w]
    )


def _scale_params():
    for hidden in HIDDEN_SIZES:
        for w in W_DTYPES:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_DTYPES:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_scale_params()))
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("rows", (7, 16231))
def test_input_scales(
    hidden: int, residual: bool, w: str, scale: float, rows: int, record_property
):
    _skip_unless_route(hidden, residual, w)
    _check(
        hidden,
        rows,
        record=record_property,
        residual=residual,
        w_dtype=W_DTYPES[w],
        scale=scale,
    )


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_scale_params()))
@pytest.mark.parametrize("rows", (1, 129, 16231))
def test_zero_input_has_no_nan(
    hidden: int, residual: bool, w: str, rows: int, record_property
):
    """``x = 0`` (and ``u = 0``) with random ``g`` and ``w``: ``y``, ``h_new`` and ``dw`` are exactly 0 and ``rstd``
    is ``eps^(-1/2)``; every output finite and judged over the draws (``g`` and ``w`` change with the seed)."""
    _skip_unless_route(hidden, residual, w)
    _check(
        hidden,
        rows,
        record=record_property,
        residual=residual,
        w_dtype=W_DTYPES[w],
        zeros=True,
    )


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_scale_params()))
@pytest.mark.parametrize("rows", (257, 16385))
def test_extreme_row(
    hidden: int, residual: bool, w: str, rows: int, record_property
) -> None:
    """One row scaled by 1e3 with one element 1e4 (normal BF16 values) among unit-scale rows."""
    _skip_unless_route(hidden, residual, w)
    _check(
        hidden,
        rows,
        record=record_property,
        residual=residual,
        w_dtype=W_DTYPES[w],
        spike=True,
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_zero_rows_launch_nothing(hidden: int, w: str) -> None:
    _skip_unless_route(hidden, w_dtype=w)
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device=_device())
    weight = torch.ones(hidden, dtype=W_DTYPES[w], device=_device())
    y, rstd = forward(x, weight, EPS[hidden])
    dw_buffer = torch.full((hidden,), float("nan"), device=_device())
    dx, dw = backward(x, x, weight, rstd, dw=dw_buffer)
    assert _LAUNCHED == []
    assert y.shape == (0, hidden) and rstd.shape == (0,) and dx.shape == (0, hidden)
    assert dw is dw_buffer and torch.equal(dw, torch.zeros_like(dw))


# --------------------------------------------------------------------------- repeats


def _repeat_params():
    for hidden in HIDDEN_SIZES:
        for rows in (257, 16385, 65536):
            for w in W_DTYPES:
                yield pytest.param(hidden, rows, False, w, id=f"h{hidden}-T{rows}-{w}")
    for rows in (7, 257, 16385, 65536):
        for w in W_DTYPES:
            yield pytest.param(6144, rows, True, w, id=f"res-h6144-T{rows}-{w}")


@pytest.mark.parametrize(("hidden", "rows", "residual", "w"), list(_repeat_params()))
def test_repeated_launches(
    hidden: int, rows: int, residual: bool, w: str, record_property
) -> None:
    """20 more launches of the backward on the case's own input, on one reused
    workspace: the window's records every time, ``dx`` bitwise identical every
    time, the workspace at rest after every call (``dw``'s spread reported).
    The forward, ``dx`` and ``dw`` meet the naive BF16 tier over the case's
    draws, the own input's ``dw`` taken at its largest error over the launches."""
    _skip_unless_route(hidden, residual, w)
    eps = EPS[hidden]
    label = f"repeat h={hidden} T={rows} res={residual} w={w}"
    check = ref.Draws(label)
    ws = workspace_for(rows, hidden, _device(), residual=residual)
    spread, form = 0.0, None
    for draw in range(check.draws):
        inputs = _case_inputs(
            hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=11, draw=draw
        )
        case = _launch(
            hidden, rows, eps, inputs, residual=residual, w_token=w, label=label
        )
        form = case.form
        if draw:
            _judge(check, draw, case)
            continue
        _judge(check, 0, case, outputs=("y", "rstd", "h_new"))
        oracle = ref.oracle_backward(
            case.g, case.normalized, case.w, case.rstd, case.g_h
        )
        naive = ref.naive_backward(
            case.g, case.normalized, case.w, case.rstd, eps, case.g_h
        )
        check.add(0, {"dx": case.dx, "dw": case.dw}, oracle, naive)
        for _ in range(REPEATS):
            (dx, dw), records = _run_backward(
                case.g,
                case.normalized,
                case.w,
                case.rstd,
                g_residual=case.g_h,
                workspace=ws,
            )
            assert records == case.records, f"{label}: launched {records}"
            assert torch.equal(dx, case.dx), f"{label}: dx changed between launches"
            check.add(0, {"dw": dw}, oracle, naive)
            spread = max(spread, (dw - case.dw).abs().max().item())
            assert _workspace_at_rest(ws), f"{label}: workspace not at rest"
    check.check(record_property)
    print(
        f"DW_REPEATS h={hidden} T={rows} res={residual} w={w} form={form} "
        f"repeats={REPEATS} max_abs_spread={spread:.3e}"
    )


@pytest.mark.parametrize("w", tuple(W_DTYPES))
@pytest.mark.parametrize("view", ("query", "key"))
def test_per_head_256_views(view: str, w: str, record_property) -> None:
    """The per-head shapes: a [T, 24, 256] query view of a [T, 24, 512] buffer
    (strides (12288, 512, 1)) and a contiguous [T, 4, 256] key view share one
    w[256]; dw sums over tokens and heads, rstd is [T, heads].  20 launches on
    the case's own input: ``dx`` bitwise stable and equal to the 2-D call on the
    collapsed rows, the workspace at rest; ``y``, ``rstd``, ``dx`` and every
    launch's ``dw`` meet the naive BF16 tier over the draws."""
    _skip_unless_route(256, w_dtype=w)
    rows = 1000
    heads = 24 if view == "query" else 4
    eps = EPS[256]
    expected = _window_records(256, "bwd", False, w, rows * heads)
    form = _form(expected)
    ws = workspace_for(rows * heads, 256, _device())
    check = ref.Draws(f"per-head view={view} w={w}")
    spread = 0.0
    for draw in range(check.draws):
        generator = _generator(256, rows, heads, w == "f32", draw=draw)
        if view == "query":
            buffer = torch.randn(
                rows, 24, 512, device=_device(), generator=generator
            ).to(torch.bfloat16)
            x = buffer[:, :, :256]
            assert x.stride() == (12288, 512, 1)
        else:
            x = torch.randn(rows, 4, 256, device=_device(), generator=generator).to(
                torch.bfloat16
            )
        w_tensor = _weight(256, generator, W_DTYPES[w])
        g = torch.randn(x.shape, device=_device(), generator=generator).to(
            torch.bfloat16
        )
        y, rstd = forward(x, w_tensor, eps)
        assert y.shape == x.shape and y.is_contiguous() and rstd.shape == x.shape[:-1]
        check.forward(draw, {"y": y, "rstd": rstd}, x, w_tensor, eps)
        oracle = ref.oracle_backward(g, x, w_tensor, rstd)
        naive = ref.naive_backward(g, x, w_tensor, rstd, eps)
        dx0 = dw0 = None
        for _ in range(REPEATS if draw == 0 else 1):
            (dx, dw), records = _run_backward(g, x, w_tensor, rstd, workspace=ws)
            assert records == expected, records
            assert dx.shape == x.shape and dx.is_contiguous() and dw.shape == (256,)
            if dx0 is None:
                dx0, dw0 = dx, dw
                check.add(draw, {"dx": dx}, oracle, naive)
            assert torch.equal(dx, dx0)
            check.add(draw, {"dw": dw}, oracle, naive)
            spread = max(spread, (dw - dw0).abs().max().item())
            assert _workspace_at_rest(ws)
        if draw == 0:
            dx2, _dw2 = backward(
                g.reshape(-1, 256), x.reshape(-1, 256), w_tensor, rstd.reshape(-1)
            )
            assert torch.equal(dx0.reshape(-1, 256), dx2)
    check.check(record_property)
    print(
        f"DW_REPEATS per-head view={view} w={w} rows={rows * heads} form={form} "
        f"repeats={REPEATS} max_abs_spread={spread:.3e}"
    )


# --------------------------------------------------------------------------- outputs, workspace, streams, graphs


@pytest.mark.parametrize("hidden", (512, 4096))
@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_garbage_filled_dw_is_overwritten(hidden: int, w: str, record_property) -> None:
    """A caller's ``dw`` filled with NaN, 1e30 or -7 is overwritten: finite, and
    every fill's ``dw`` meets the naive BF16 tier over the draws."""
    _skip_unless_route(hidden, w_dtype=w)
    eps = EPS[hidden]
    fills = (float("nan"), 1e30, -7.0)
    check = ref.Draws(f"garbage dw h={hidden} w={w}")
    for draw in range(check.draws):
        x, _u, g, _g_h, weight = _case_inputs(
            hidden, 1000, w_dtype=W_DTYPES[w], draw=draw
        )
        _y, rstd = forward(x, weight, eps)
        names = [f"dw fill={fill}" for fill in fills]
        oracle = dict.fromkeys(names, ref.oracle_dw(g, x, rstd))
        naive = dict.fromkeys(names, ref.naive_dw(g, x, eps))
        for fill, name in zip(fills, names, strict=True):
            dw = torch.full((hidden,), fill, device=_device())
            _dx2, dw_out = backward(g, x, weight, rstd, dw=dw)
            assert dw_out is dw and torch.isfinite(dw).all()
            check.add(draw, {name: dw}, oracle, naive)
    check.check(record_property)


def _reuse_call(ws, hidden, residual, w, rows, key, draw, check: ref.Draws):
    """One forward and backward of draw ``draw`` on the caller's workspace
    ``ws``, its pair partial rows poisoned (NaN) first: the call rewrites every
    row it reads; ``dx`` and ``dw`` judged in ``check``."""
    eps = EPS[hidden]
    ws[loader.PAIR_ROWS_OFFSET :].fill_(0xFF)
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=key, draw=draw
    )
    outputs = forward(x, weight, eps, residual=u)
    normalized = outputs[2] if residual else x
    rstd = outputs[1]
    (dx, dw), records = _run_backward(
        g, normalized, weight, rstd, g_residual=g_h, workspace=ws
    )
    expected = _window_records(hidden, "bwd", residual, w, rows)
    label = f"h={hidden} res={residual} w={w} T={rows}"
    assert records == expected, f"{label}: launched {records}"
    check.backward(
        draw, {"dx": dx, "dw": dw}, g, normalized, weight, rstd, eps, g_h=g_h
    )
    return label, _form(expected)


@pytest.mark.parametrize("hidden", (512, 2048, 6144))
def test_workspace_reused_across_token_counts_and_weight_dtypes(
    hidden: int, record_property
) -> None:
    """One buffer serves a width's calls of every token count up to its rows,
    every window and both weight dtypes with no host write but NaN poison in its
    pair partial rows before each call; at rest after every call; every call's
    ``dx`` and ``dw`` meet the naive BF16 tier over the draws.  (Reuse across
    widths, residual flags and stale bytes: ``test_cake_rmsnorm_nondet_train_api.py``.)"""
    _skip_unless_route(hidden)
    ws = torch.zeros(_bytes(65536, hidden), dtype=torch.uint8, device=_device())
    w_dtypes = ["bf16"] + (["f32"] if _routes(hidden, "bwd", False, "f32") else [])
    sequence = [
        (rows, w_dtypes[index % len(w_dtypes)])
        for index, rows in enumerate((7, 65536, 1, 257, 16385, 4096, 2))
    ]
    checks = {
        (rows, w): ref.Draws(
            f"reuse h={hidden} T={rows} w={w}", prefix=f"T={rows} {w} "
        )
        for rows, w in sequence
    }
    forms = []
    for draw in range(ref.DRAWS):
        for rows, w in sequence:
            _label, form = _reuse_call(
                ws, hidden, False, w, rows, 29, draw, checks[(rows, w)]
            )
            assert _workspace_at_rest(ws), f"workspace state after h={hidden} T={rows}"
            if draw == 0:
                forms.append(f"{rows}:{w}:{form}")
    ref.check_all(record_property, checks.values())
    print(f"REUSE h={hidden} " + " ".join(forms))


@pytest.mark.parametrize(
    ("hidden", "residual", "w"),
    [
        pytest.param(4096, False, "bf16", id="h4096-bf16"),
        pytest.param(6144, True, "f32", id="res-h6144-f32"),
    ],
)
def test_two_streams_with_their_own_workspaces(
    hidden: int, residual: bool, w: str, record_property
):
    """Two streams, each with its own workspace, run five rounds of backward
    calls of two token counts; every call's ``dx`` and ``dw`` meet the naive
    BF16 tier over the draws, and both workspaces are at rest."""
    _skip_unless_route(hidden, residual, w)
    eps = EPS[hidden]
    counts = (16385, 65536)
    checks = {
        rows: ref.Draws(f"streams h={hidden} T={rows} w={w}", prefix=f"T={rows} ")
        for rows in counts
    }
    streams = [torch.cuda.Stream() for _ in counts]
    workspaces = [
        workspace_for(65536, hidden, _device(), residual=residual) for _ in counts
    ]
    for draw in range(ref.DRAWS):
        cases = []
        for rows in counts:
            x, u, g, g_h, weight = _case_inputs(
                hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=37, draw=draw
            )
            outputs = forward(x, weight, eps, residual=u)
            normalized, rstd = (outputs[2] if residual else x), outputs[1]
            oracle = ref.oracle_backward(g, normalized, weight, rstd, g_h)
            naive = ref.naive_backward(g, normalized, weight, rstd, eps, g_h)
            cases.append((rows, g, g_h, normalized, weight, rstd, oracle, naive))
        torch.cuda.synchronize()
        results = []
        for _round in range(5):
            for stream, ws, case in zip(streams, workspaces, cases, strict=True):
                _count, g, g_h, normalized, weight, rstd = case[:6]
                with torch.cuda.stream(stream):
                    results.append(
                        backward(
                            g, normalized, weight, rstd, g_residual=g_h, workspace=ws
                        )
                    )
        torch.cuda.synchronize()
        for index, (dx, dw) in enumerate(results):
            rows, *_inputs, oracle, naive = cases[index % len(cases)]
            checks[rows].add(draw, {"dx": dx, "dw": dw}, oracle, naive)
        assert all(_workspace_at_rest(ws) for ws in workspaces)
    ref.check_all(record_property, checks.values())


def _graph_params():
    for hidden, residual, w in (
        (512, False, "bf16"),
        (6144, False, "bf16"),
        (6144, True, "bf16"),
        (6144, True, "f32"),
        (4096, False, "f32"),
        (5120, False, "f32"),
        (256, False, "bf16"),
    ):
        for rows in (64, 4096):
            prefix = "res-" if residual else ""
            yield pytest.param(
                hidden, residual, w, rows, id=f"{prefix}h{hidden}-{w}-T{rows}"
            )


@pytest.mark.parametrize(("hidden", "residual", "w", "rows"), list(_graph_params()))
def test_cuda_graph_capture_and_replay(
    hidden: int, residual: bool, w: str, rows: int, record_property
):
    """One captured forward + backward: three replays on the case's own input,
    then one on each further draw copied into the static inputs.  Every replay
    gives the eager call's ``y``, ``rstd``, ``h_new`` and ``dx`` bit for bit on
    the same input, its ``dw`` (and the outputs) meet the naive BF16 tier over the
    draws, and the workspace is at rest."""
    _skip_unless_route(hidden, residual, w)
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=41
    )
    eps = EPS[hidden]
    ws = workspace_for(rows, hidden, _device(), residual=residual)
    outputs = forward(x, weight, eps, residual=u)  # warm the launch path
    normalized, rstd = (outputs[2] if residual else x), outputs[1]
    dx, dw = backward(g, normalized, weight, rstd, g_residual=g_h, workspace=ws)
    out_y, out_rstd = torch.empty_like(outputs[0]), torch.empty_like(rstd)
    out_h = torch.empty_like(normalized) if residual else None
    out_dx, out_dw = torch.empty_like(dx), torch.full_like(dw, float("nan"))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(
            x, weight, eps, residual=u, out=out_y, rstd=out_rstd, residual_out=out_h
        )
        backward(
            g,
            out_h if residual else x,
            weight,
            out_rstd,
            g_residual=g_h,
            workspace=ws,
            dx=out_dx,
            dw=out_dw,
        )
    check = ref.Draws(f"graph h={hidden} res={residual} w={w} T={rows}")
    for draw in range(check.draws):
        if draw:
            fresh = _case_inputs(
                hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=41, draw=draw
            )
            for static, value in zip((x, u, g, g_h, weight), fresh, strict=True):
                if static is not None:
                    static.copy_(value)
            outputs = forward(x, weight, eps, residual=u)
            normalized, rstd = (outputs[2] if residual else x), outputs[1]
            dx, _dw = backward(
                g, normalized, weight, rstd, g_residual=g_h, workspace=ws
            )
            assert _workspace_at_rest(ws)
        for _replay in range(3 if draw == 0 else 1):
            out_dw.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out_y, outputs[0]) and torch.equal(out_rstd, rstd)
            if residual:
                assert torch.equal(out_h, normalized)
            assert torch.equal(out_dx, dx)
            forward_outputs = {"y": out_y, "rstd": out_rstd}
            if residual:
                forward_outputs["h_new"] = out_h
            check.forward(draw, forward_outputs, x, weight, eps, residual=u)
            check.backward(
                draw,
                {"dx": out_dx, "dw": out_dw},
                g,
                normalized,
                weight,
                rstd,
                eps,
                g_h=g_h,
            )
            assert _workspace_at_rest(ws)
    check.check(record_property)


# --------------------------------------------------------------------------- autograd


def _autograd_params():
    for hidden in (512, 4096):
        for w in W_DTYPES:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_DTYPES:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


def _autograd_dw(grad_w, weight, label) -> torch.Tensor:
    """The kernel's FP32 ``dw`` of the last backward this package ran (the last
    backward buffer it wrote); the autograd weight gradient is it cast to the
    weight's dtype bit for bit."""
    assert _DW_BUFFERS, f"{label}: no backward module of this package ran"
    dw32 = _DW_BUFFERS[-1].clone()
    assert grad_w.dtype == weight.dtype, label
    assert torch.equal(grad_w, dw32.to(weight.dtype)), (
        f"{label}: the autograd dw is not the kernel's dw in {weight.dtype}"
    )
    return dw32


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_autograd_matches_fp64(
    hidden: int, residual: bool, w: str, record_property
) -> None:
    """The autograd wrapper's ``y`` (and ``h_new``), input gradient and the
    kernel's FP32 ``dw`` (the weight gradient is it in the weight's dtype) meet
    the naive BF16 tier over the draws, the backward judged at the FP32 ``rstd``
    the forward saved (the functional forward's, same program, same bits)."""
    _skip_unless_route(hidden, residual, w)
    rows = 1000
    eps = EPS[hidden]
    check = ref.Draws(f"autograd h={hidden} res={residual} w={w}")
    for draw in range(check.draws):
        x, u, g, g_h, weight = _case_inputs(
            hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=43, draw=draw
        )
        xa = x.clone().requires_grad_(True)
        wa = weight.clone().requires_grad_(True)
        ua = u.clone().requires_grad_(True) if residual else None
        outputs = rmsnorm(xa, wa, eps, residual=ua)
        if residual:
            y, h_new = outputs
            torch.autograd.backward((y, h_new), (g, g_h))
            assert torch.equal(ua.grad, xa.grad)
            forward_outputs = {"y": y.detach(), "h_new": h_new.detach()}
        else:
            y = outputs
            y.backward(g)
            forward_outputs = {"y": y.detach()}
        dw32 = _autograd_dw(wa.grad, weight, "autograd dw")
        outputs_fwd = forward(x, weight, eps, residual=u)
        normalized, rstd = (outputs_fwd[2] if residual else x), outputs_fwd[1]
        check.forward(draw, forward_outputs, x, weight, eps, residual=u)
        check.backward(
            draw, {"dx": xa.grad, "dw": dw32}, g, normalized, weight, rstd, eps, g_h=g_h
        )
    check.check(record_property)


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_autograd_accumulates_into_existing_grads(
    hidden: int, residual: bool, w: str, record_property
):
    """Gradients add into existing ``.grad`` tensors (two backwards through one
    retained graph, on top of preset gradients) exactly as autograd adds them;
    both kernel ``dw`` meet the naive BF16 tier over the draws; the graph's
    saved tensors are released after the last backward, so a third one raises."""
    _skip_unless_route(hidden, residual, w)
    rows = 4096
    eps = EPS[hidden]
    check = ref.Draws(f"accumulate h={hidden} res={residual} w={w}")
    for draw in range(check.draws):
        x, u, g, g_h, weight = _case_inputs(
            hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=47, draw=draw
        )
        generator = _generator(hidden, rows, 53, draw=draw)
        gx0 = _rows(rows, hidden, generator=generator)
        gw0 = _weight(hidden, generator, W_DTYPES[w]) - 1.0
        xa = x.clone().requires_grad_(True)
        wa = weight.clone().requires_grad_(True)
        ua = u.clone().requires_grad_(True) if residual else None
        xa.grad, wa.grad = gx0.clone(), gw0.clone()
        if residual:
            ua.grad = gx0.clone()
        outputs = rmsnorm(xa, wa, eps, residual=ua)
        tensors = outputs if residual else (outputs,)
        grads = (g, g_h) if residual else (g,)
        outputs_fwd = forward(x, weight, eps, residual=u)
        normalized, rstd = (outputs_fwd[2] if residual else x), outputs_fwd[1]
        dx_k, _dw = backward(g, normalized, weight, rstd, g_residual=g_h)
        oracle = {"dw": ref.oracle_dw(g, normalized, rstd)}
        naive = {"dw": ref.naive_dw(g, normalized, eps)}
        torch.autograd.backward(tensors, grads, retain_graph=True)
        dw1 = _DW_BUFFERS[-1].clone()
        check.add(draw, {"dw": dw1}, oracle, naive)
        expected_x = gx0 + dx_k
        expected_w = gw0 + dw1.to(weight.dtype)
        assert torch.equal(xa.grad, expected_x)
        assert torch.equal(wa.grad, expected_w)
        if residual:
            assert torch.equal(ua.grad, expected_x)
        torch.autograd.backward(tensors, grads)
        dw2 = _DW_BUFFERS[-1].clone()
        check.add(draw, {"dw": dw2}, oracle, naive)
        assert torch.equal(xa.grad, expected_x + dx_k)
        assert torch.equal(wa.grad, expected_w + dw2.to(weight.dtype))
        if residual:
            assert torch.equal(ua.grad, expected_x + dx_k)
        with pytest.raises(
            RuntimeError, match="backward through the graph a second time"
        ):
            torch.autograd.backward(tensors, grads)
    check.check(record_property)


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_autograd_saves_only_its_inputs_and_rstd(hidden: int, residual: bool, w: str):
    """The autograd function saves three tensors -- the normalized input (the
    caller's ``x`` or the returned ``h_new``, not a copy), the weight and the
    FP32 ``[T]`` rstd -- and holds nothing else: after the backward and the
    outputs are dropped, the device memory is the inputs and their gradients."""
    _skip_unless_route(hidden, residual, w)
    rows = 4096
    eps = EPS[hidden]
    x, u, g, g_h, weight = _case_inputs(
        hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=59
    )
    xa = x.clone().requires_grad_(True)
    wa = weight.clone().requires_grad_(True)
    ua = u.clone().requires_grad_(True) if residual else None
    saved: list[torch.Tensor] = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    torch.cuda.synchronize()
    base = ref.requested_bytes(_device())
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        outputs = rmsnorm(xa, wa, eps, residual=ua)
    y = outputs[0] if residual else outputs
    normalized = outputs[1] if residual else xa
    assert len(saved) == 3, [(tuple(t.shape), t.dtype) for t in saved]
    others = [
        t for t in saved if t.data_ptr() not in (normalized.data_ptr(), wa.data_ptr())
    ]
    assert len(others) == 1, [(tuple(t.shape), t.dtype) for t in saved]
    assert others[0].dtype == torch.float32 and tuple(others[0].shape) == (rows,)
    torch.cuda.synchronize()
    held = ref.requested_bytes(_device()) - base
    outputs_bytes = y.numel() * y.element_size() + (
        normalized.numel() * normalized.element_size() if residual else 0
    )
    # exactly the outputs plus the FP32 rstd (requested bytes are not rounded:
    # no slack)
    assert held == outputs_bytes + 4 * rows, (
        f"{held} bytes held after the forward, outputs {outputs_bytes} + "
        f"FP32 rstd {4 * rows}"
    )
    saved.clear()  # drop this test's references to the saved tensors
    others.clear()
    if residual:
        torch.autograd.backward(outputs, (g, g_h))
    else:
        outputs.backward(g)
    del outputs, y, normalized
    _DW_BUFFERS.clear()  # the launch recorder's references to the kernel's dw
    torch.cuda.synchronize()
    grads = [xa.grad, wa.grad] + ([ua.grad] if residual else [])
    storages = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in grads
    }
    grads_bytes = sum(storages.values())
    left = ref.requested_bytes(_device()) - base
    assert left == grads_bytes, (
        f"{left} bytes held after the backward, gradients {grads_bytes}"
    )


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_autograd_params()))
def test_checkpoint_replays_the_same_forward(
    hidden: int, residual: bool, w: str, record_property
):
    """Non-reentrant activation checkpointing: the replayed forward gives the
    same ``y`` and ``dx`` bits as the plain autograd call; ``dx`` and the kernel
    ``dw`` of both (and of a second checkpointed micro-batch, which accumulates
    into the same ``.grad``) meet the naive BF16 tier over the draws."""
    _skip_unless_route(hidden, residual, w)
    rows = 1000
    eps = EPS[hidden]
    check = ref.Draws(f"checkpoint h={hidden} res={residual} w={w}")

    def step(xt, wt, ut):
        out = rmsnorm(xt, wt, eps, residual=ut)
        return out if residual else (out,)

    def run(checkpointed: bool, wt, x, u, g, g_h):
        xt = x.clone().requires_grad_(True)
        ut = u.clone().requires_grad_(True) if residual else None
        if checkpointed:
            outs = torch.utils.checkpoint.checkpoint(
                step, xt, wt, ut, use_reentrant=False
            )
        else:
            outs = step(xt, wt, ut)
        torch.autograd.backward(outs, (g, g_h) if residual else (g,))
        return outs[0].detach(), xt.grad

    for draw in range(check.draws):
        x, u, g, g_h, weight = _case_inputs(
            hidden, rows, residual=residual, w_dtype=W_DTYPES[w], key=61, draw=draw
        )
        outputs_fwd = forward(x, weight, eps, residual=u)
        normalized, rstd = (outputs_fwd[2] if residual else x), outputs_fwd[1]
        oracle = ref.oracle_backward(g, normalized, weight, rstd, g_h)
        naive = ref.naive_backward(g, normalized, weight, rstd, eps, g_h)
        wa = weight.clone().requires_grad_(True)
        y_ckpt, dx_ckpt = run(True, wa, x, u, g, g_h)
        dw_ckpt = _autograd_dw(wa.grad, weight, "checkpoint dw")
        wb = weight.clone().requires_grad_(True)
        y_plain, dx_plain = run(False, wb, x, u, g, g_h)
        dw_plain = _autograd_dw(wb.grad, weight, "plain dw")
        assert torch.equal(y_ckpt, y_plain) and torch.equal(dx_ckpt, dx_plain)
        check.add(draw, {"dx": dx_ckpt, "dw": dw_ckpt}, oracle, naive)
        check.add(draw, {"dw": dw_plain}, oracle, naive)
        first = wa.grad.clone()
        run(True, wa, x, u, g, g_h)  # a second micro-batch accumulates
        dw_second = _DW_BUFFERS[-1].clone()
        check.add(draw, {"dw": dw_second}, oracle, naive)
        assert torch.equal(wa.grad, first + dw_second.to(weight.dtype))
        assert torch.equal(first, dw_ckpt.to(weight.dtype))
    check.check(record_property)


def _backward_launches(fn, *args) -> list[str]:
    start = len(_LAUNCHED)
    fn(*args)
    return [n for n in _LAUNCHED[start:] if loader.MODULES[n]["role"].startswith("bwd")]


@pytest.mark.parametrize("w", tuple(W_DTYPES))
def test_autograd_backward_checks_deterministic_algorithms_when_it_runs(
    w: str, record_property
) -> None:
    """The autograd forward does not consult ``torch.use_deterministic_algorithms``;
    the backward does when it runs.  Strict determinism: the forward runs this
    package's forward with no alert, the backward raises ``RuntimeError`` naming
    the det entry point and launches nothing.  ``warn_only`` at backward time:
    one warning, then this package's backward -- ``dx`` bitwise equal to this
    package's ``dx`` with determinism off, the kernel's ``dw`` meeting the naive
    BF16 tier.  Determinism on only at forward time: the backward runs with no
    alert, ``dx`` and ``dw`` meeting the naive BF16 tier (over the draws)."""
    hidden, rows = 512, 257
    _skip_unless_route(hidden, w_dtype=w)
    eps = EPS[hidden]
    expected = _window_records(hidden, "bwd", False, w, rows)
    warn_only = ref.Draws(f"warn_only w={w}", prefix="warn_only ")
    strict = ref.Draws(f"strict-at-forward w={w}", prefix="strict-at-forward ")

    def package_warnings(caught) -> list[str]:
        return [
            str(item.message)
            for item in caught
            if "cake_pplx.cake_rmsnorm_nondet_train." in str(item.message)
        ]

    def graph(x, weight):
        xa = x.clone().requires_grad_(True)
        wa = weight.clone().requires_grad_(True)
        return xa, wa, rmsnorm(xa, wa, eps)

    for draw in range(ref.DRAWS):
        x, _u, g, _g_h, weight = _case_inputs(
            hidden, rows, w_dtype=W_DTYPES[w], key=67, draw=draw
        )
        rstd = forward(x, weight, eps)[1]
        dx_off, _dw_off = backward(g, x, weight, rstd)
        oracle = ref.oracle_backward(g, x, weight, rstd)
        naive = ref.naive_backward(g, x, weight, rstd, eps)

        # strict determinism: the forward runs, the backward refuses
        torch.use_deterministic_algorithms(True)
        start = len(_LAUNCHED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _xa, _wa, y = graph(x, weight)
        assert package_warnings(caught) == [], "the forward alerted"
        assert _LAUNCHED[start:] == _window_records(hidden, "fwd", False, w, rows)
        start = len(_LAUNCHED)
        with pytest.raises(
            RuntimeError, match=r"cake_rmsnorm_det_train\.cake_rmsnorm_det\b"
        ):
            y.backward(g)
        assert _LAUNCHED[start:] == [], "a refused backward launched a module"
        torch.use_deterministic_algorithms(False)

        # warn_only at backward time: one warning, this package's backward
        xa, wa, y = graph(x, weight)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            launched = _backward_launches(y.backward, g)
        torch.use_deterministic_algorithms(False)
        messages = package_warnings(caught)
        assert len(messages) == 1, messages
        assert re.search(
            r"cake_pplx\.cake_rmsnorm_det_train\.cake_rmsnorm_det\b", messages[0]
        ), messages[0]
        assert launched == expected, launched
        assert torch.equal(xa.grad, dx_off)
        warn_only.add(
            draw, {"dw": _autograd_dw(wa.grad, weight, "warn_only dw")}, oracle, naive
        )

        # determinism on only at forward time: the backward runs with no alert
        torch.use_deterministic_algorithms(True)
        xa, wa, y = graph(x, weight)
        torch.use_deterministic_algorithms(False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            launched = _backward_launches(y.backward, g)
        assert package_warnings(caught) == []
        assert launched == expected, launched
        assert torch.equal(xa.grad, dx_off)
        strict.add(
            draw,
            {
                "dx": xa.grad,
                "dw": _autograd_dw(wa.grad, weight, "strict-at-forward dw"),
            },
            oracle,
            naive,
        )
    ref.check_all(record_property, (warn_only, strict))


# --------------------------------------------------------------------------- argument validation


def _refused(error: type, call, why: str) -> None:
    """``call`` raises ``error`` before launching anything."""
    start = len(_LAUNCHED)
    with pytest.raises(error):
        call()
    assert _LAUNCHED[start:] == [], f"refused call ({why}) launched {_LAUNCHED[start:]}"


@pytest.mark.parametrize("hidden", (512, 6144))
def test_argument_validation(hidden: int) -> None:
    """Wrong dtypes, shapes, devices, layouts, output buffers and workspaces
    (too small for the call, wrong dtype, on the CPU, misaligned, 2-D) are
    refused with ``ValueError`` before any launch (the ``deterministic``
    keyword is Python's ``TypeError``: ``test_cake_rmsnorm_det_train_names.py``)."""
    _skip_unless_route(hidden)
    eps = EPS[hidden]
    rows = 4096
    x, _u, g, _g_h, weight = _case_inputs(hidden, rows, key=71)
    _y, rstd = forward(x, weight, eps)
    device = _device()
    refusals = (
        (lambda: forward(x.float(), weight, eps), "x dtype"),
        (lambda: forward(x, weight.half(), eps), "w dtype"),
        (lambda: forward(x[:, 1:], weight[1:], eps), "no route for hidden - 1"),
        (lambda: forward(x, weight[: hidden // 2], eps), "w shape"),
        (lambda: forward(x.cpu(), weight.cpu(), eps), "cpu"),
        (
            lambda: forward(
                x, weight, eps, out=torch.empty(rows, hidden, device=device)
            ),
            "out dtype",
        ),
        (
            lambda: forward(
                x.view(rows, 4, hidden // 4).transpose(1, 2), weight[:4], eps
            ),
            "non-collapsible N-D",
        ),
        (lambda: backward(g, x, weight, rstd.double()), "rstd dtype"),
        (lambda: backward(g, x, weight, rstd[:-1]), "rstd shape"),
        (lambda: backward(g[:-1], x, weight, rstd), "g shape"),
        (
            lambda: backward(
                g, x, weight, rstd, dx=torch.empty(rows, hidden, device=device)
            ),
            "dx dtype",
        ),
        (
            lambda: backward(
                g, x, weight, rstd, dw=torch.empty(hidden + 1, device=device)
            ),
            "dw shape",
        ),
        (
            lambda: pkg.cake_rmsnorm_nondet_train_backward_workspace_bytes(
                -1, hidden, device
            ),
            "negative rows",
        ),
    )
    for call, why in refusals:
        _refused(ValueError, call, why)
    needed = _bytes(rows, hidden)
    workspaces = [
        (torch.zeros(256, dtype=torch.uint8, device=device), "tiny"),
        (
            torch.zeros(loader.COUNTERS_BYTES, dtype=torch.uint8, device=device),
            "counters only",
        ),
        (torch.zeros(needed // 4, dtype=torch.float32, device=device), "dtype"),
        (torch.zeros(needed, dtype=torch.uint8), "cpu"),
        (torch.zeros(needed + 512, dtype=torch.uint8, device=device)[1:], "unaligned"),
        (torch.zeros(2, needed, dtype=torch.uint8, device=device), "2-D"),
    ]
    if _form(_window_records(hidden, "bwd", False, "bf16", rows)) == "pair":
        # the documented size of a pair window's call is that call's own layout
        workspaces.append(
            (
                torch.zeros(needed - 256, dtype=torch.uint8, device=device),
                "one block short",
            )
        )
    for ws, why in workspaces:
        _refused(
            ValueError, lambda ws=ws: backward(g, x, weight, rstd, workspace=ws), why
        )


# --------------------------------------------------------------------------- build manifest (no device)


@pytest.mark.cpu_only
def test_every_library_carries_its_architecture_floor_and_flags() -> None:
    """Every ``build.json`` library is built for exactly its architecture's code
    (``-gencode=arch=compute_XXXa,code=sm_XXXa``), carries that architecture's
    CUDA floor (sm_100a 12.8, sm_103a 12.9, sm_107a 13.5) and the exact-math
    flags; the package floor is the lowest of them; every architecture ships
    every route (the feature x architecture matrix)."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert set(manifest["arch_flags"]) == set(loader.ARCHES) == set(MIN_CUDA)
    for arch, flags in manifest["arch_flags"].items():
        assert flags == [GENCODE[arch]], (arch, flags)
    requires = manifest["common"]["requires"]["cuda_toolkit"]["min"]
    assert requires == min(
        MIN_CUDA.values(), key=lambda v: tuple(map(int, v.split(".")))
    )
    libraries = cake_build.libraries_by_name()
    per_arch = {arch: 0 for arch in loader.ARCHES}
    for name, entry in libraries.items():
        arch = entry["arch"]
        assert name.endswith(f"_{arch}"), name
        assert entry["min_cuda"] == MIN_CUDA[arch], (name, entry["min_cuda"])
        assert entry["gencode"] == GENCODE[arch], (name, entry["gencode"])
        flags = entry["nvcc_flags"]
        assert all(flag in flags for flag in ACCURACY_FLAGS), (name, flags)
        assert "--use_fast_math" not in flags, name
        per_arch[arch] += 1
    referenced = {
        loader.library_name(name, arch)
        for name, record in loader.MODULES.items()
        for arch in record["arches"]
    }
    assert referenced == set(libraries)
    routes = {arch: set(loader.ROUTES[arch]) for arch in loader.ARCHES}
    first = routes[loader.ARCHES[0]]
    assert all(keys == first for keys in routes.values()), {
        arch: sorted(keys ^ first) for arch, keys in routes.items()
    }
    for key in sorted(first):
        kind, hidden = key.split("_", 1)[0], int(key.rsplit("_h", 1)[1])
        assert hidden in loader.HIDDEN_SIZES and kind in loader.KINDS, key
    print(
        "MANIFEST "
        + " ".join(
            f"{arch}:{MIN_CUDA[arch]}:{per_arch[arch]}libs" for arch in loader.ARCHES
        )
        + f" routes={len(first)}"
    )


@pytest.mark.cpu_only
def test_registry_records_run_on_the_exported_architectures() -> None:
    for arch, by_route in loader.ROUTES.items():
        assert arch in loader.ARCHES
        for route, names in by_route.items():
            for name in names:
                record = loader.MODULES[name]
                assert arch in record["arches"], (arch, route, name)
                assert loader.library_name(name, arch) in cake_build.libraries_by_name()
