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

Tests for the Cake fused BF16 RMSNorm training forward / backward
(SM100 / SM103 / SM107).

Every floating kernel output -- ``y``, ``rstd``, ``h_new``, ``dx``, ``dw`` -- is
judged against the naive PyTorch BF16 implementation, with no tolerance: its
error against an FP64 oracle of the same math, over 32 independent random
inputs of each case, in the L2 and the max norm, median and largest, is at most
1.005 times the naive's, and 0 wherever the naive's is 0
(``tests/norm/test_helpers/cake_rmsnorm_train_reference.py``,
``tests/cake/test_helpers/cake_naive.py``).  ``h_new`` is also the correctly
rounded BF16 sum, and the deterministic outputs are bitwise reproducible run to
run.  The matrix covers small and packed token counts (including 1 and 0), tile
boundaries, row-strided inputs, 1e-3 / 1e3 input scales, zero input and a row of
large values, the residual-add fusion (against the naive add-then-norm training
step for every input gradient), bitwise reproducibility of ``dw``, changing
token counts in one process and the autograd wrapper end to end.

This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
``cake_pplx``, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``cake_pplx.cake_rmsnorm_det_train`` facade is covered by
``test_cake_rmsnorm_det_train_api.py``.
"""

from __future__ import annotations

import pytest
import re
from pathlib import Path

import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.norm.test_helpers import cake_rmsnorm_train_reference as ref

pkg = cake_package("cake_rmsnorm_det_train")
cake_build = cake_package("cake_rmsnorm_det_train", "cake_build")
loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
windows = cake_package("cake_rmsnorm_det_train", "cake_windows")
cake_rmsnorm_det = pkg.cake_rmsnorm_det
cake_rmsnorm_det_train_backward = pkg.cake_rmsnorm_det_train_backward
cake_rmsnorm_det_train_backward_workspace = (
    pkg.cake_rmsnorm_det_train_backward_workspace
)
cake_rmsnorm_det_train_backward_workspace_bytes = (
    pkg.cake_rmsnorm_det_train_backward_workspace_bytes
)
cake_rmsnorm_det_train_forward = pkg.cake_rmsnorm_det_train_forward
is_cake_rmsnorm_det_train_supported = pkg.is_cake_rmsnorm_det_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
# Widths whose contract names both eps values: every case runs at each.
EPS_VARIANTS = {4096: (1e-5, 1e-6), 1536: (1e-5, 1e-6)}
ROWS = (1, 2, 7, 63, 64, 65, 127, 128, 129, 255, 257, 16172, 16231)
# Further row counts of the H=1536 / H=4096 contract: the program-window edges, the 2^n +- 1 counts and the
# representative training shapes (GLM-5.3-Flash CP16 among them).
CONTRACT_ROWS = {
    4096: (1023, 1025, 4097, 14928, 16263, 16384, 32769, 65536),
    1536: (1023, 1025, 4097, 14928, 15566, 16263, 16384, 32769, 65536),
}
STRIDE_PADS = (0, 64, "2H")
# The row count standing in for the open end of a route's last program window.
OPEN_WINDOW_ROWS = 65536
SCALES = (1.0, 1e-3, 1e3)
SEED = 760


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _skip_unless_hidden(hidden: int) -> None:
    if not is_cake_rmsnorm_det_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


def _skip_unless_route(hidden: int, residual: bool) -> None:
    _skip_unless_hidden(hidden)
    if residual and not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="fwd",
        residual=True,
    ):
        pytest.skip(f"hidden={hidden} has no exported residual route on this device")


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
    stride_pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    generator: torch.Generator,
) -> torch.Tensor:
    """BF16 ``[rows, hidden]`` view with ``stride(0) == hidden + pad``."""

    pad = hidden if stride_pad == "2H" else int(stride_pad)
    storage_rows = max(rows, 1)
    if zeros:
        base = torch.zeros(
            storage_rows, hidden + pad, dtype=torch.bfloat16, device=_device()
        )
    else:
        base = (
            scale
            * torch.randn(
                storage_rows,
                hidden + pad,
                dtype=torch.float32,
                device=_device(),
                generator=generator,
            )
        ).to(torch.bfloat16)
    return base[:rows, :hidden]


def _weight(hidden: int, generator: torch.Generator) -> torch.Tensor:
    return (
        1.0
        + 0.1
        * torch.randn(
            hidden, dtype=torch.float32, device=_device(), generator=generator
        )
    ).to(torch.bfloat16)


def _h_new_is_the_rounded_sum(h_new, x, u, label: str) -> None:
    """``h_new`` is ``x + u`` rounded once to BF16 (the float64 sum of two BF16 values is exact)."""
    assert torch.equal(h_new, (x.double() + u.double()).to(torch.bfloat16)), (
        f"h_new {label} is not the BF16-rounded sum"
    )


def _check_case(
    hidden: int,
    rows: int,
    *,
    record,
    tag: str = "",
    x_pad=0,
    g_pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    residual: bool = False,
    spike: bool = False,
    eps: float | None = None,
) -> None:
    """Forward and backward of one case over its draws, each output judged against the naive BF16 tier
    (``ref.Draws``; ``tag`` names the outputs of one case among several of a test); verdicts recorded with
    ``record`` (``record_property``)."""
    eps = EPS[hidden] if eps is None else eps
    pad_key = hidden if x_pad == "2H" else int(x_pad)
    label = f"hidden={hidden} rows={rows} eps={eps} pad={x_pad} scale={scale} zeros={zeros} residual={residual}"
    check = ref.Draws(label, prefix=tag)
    for draw in range(check.draws):
        generator = _generator(
            hidden,
            rows,
            pad_key,
            int(scale * 1e6) % 9973,
            zeros,
            residual,
            spike,
            draw=draw,
        )
        x = _rows(
            rows,
            hidden,
            stride_pad=x_pad,
            scale=scale,
            zeros=zeros,
            generator=generator,
        )
        if spike and rows > 0:
            row = min(rows - 1, 3)
            x[row] = (x[row].float() * 1e3).to(torch.bfloat16)
            x[row, min(hidden - 1, 17)] = 1e4
        w = _weight(hidden, generator)
        u = _rows(rows, hidden, scale=scale, generator=generator) if residual else None
        g = _rows(rows, hidden, stride_pad=g_pad, generator=generator)
        g_h = _rows(rows, hidden, generator=generator) if residual else None

        outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
        y, rstd = outputs[0], outputs[1]
        assert tuple(y.shape) == (rows, hidden) and y.is_contiguous()
        assert rstd.dtype == torch.float32 and tuple(rstd.shape) == (rows,)
        forward = {"y": y, "rstd": rstd}
        if residual:
            normalized = forward["h_new"] = outputs[2]
            _h_new_is_the_rounded_sum(normalized, x, u, label)
        else:
            assert len(outputs) == 2
            normalized = x
        check.forward(draw, forward, x, w, eps, residual=u)

        dx, dw = cake_rmsnorm_det_train_backward(g, normalized, w, rstd, g_residual=g_h)
        assert tuple(dx.shape) == (rows, hidden) and dx.is_contiguous()
        assert dw.dtype == torch.float32 and dw.is_contiguous()
        check.backward(draw, {"dx": dx, "dw": dw}, g, normalized, w, rstd, eps, g_h=g_h)
    check.check(record)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", ROWS)
def test_forward_backward_match_fp64(hidden: int, rows: int, record_property) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, record=record_property)


def _switch_rows(arch: str, hidden: int, kind: str, residual: bool) -> tuple[int, ...]:
    """Row counts on both sides of every program switch point of one route (and a single row)."""

    switches = set()
    for name in loader.route_modules(arch, hidden, kind, residual):
        rule = loader.MODULES[name]["select_rule"]
        if rule is None:
            continue
        switches.update(
            int(bound) for bound in (rule["above"], rule["at_most"]) if bound
        )
    return tuple(sorted({1, *switches, *(rows + 1 for rows in switches)}))


@pytest.mark.parametrize(
    ("hidden", "rows"),
    [(hidden, rows) for hidden, counts in CONTRACT_ROWS.items() for rows in counts],
)
def test_contract_rows_match_fp64(hidden: int, rows: int, record_property) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, record=record_property)


@pytest.mark.parametrize(
    ("hidden", "eps"),
    [(hidden, eps) for hidden, variants in EPS_VARIANTS.items() for eps in variants],
)
@pytest.mark.parametrize("rows", (7, 16231))
@pytest.mark.parametrize("scale", SCALES)
def test_eps_variants(
    hidden: int, eps: float, rows: int, scale: float, record_property
) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, record=record_property, scale=scale, eps=eps)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
def test_forward_programs_switch_on_row_count(
    hidden: int, residual: bool, record_property
) -> None:
    """Exactly one forward program runs per call, every exported program runs for some row count, and each
    meets the naive BF16 tier on both sides of its switch point (over the draws of each row count)."""

    _skip_unless_route(hidden, residual)
    arch = loader.arch_for_capability(_capability())
    names = loader.route_modules(arch, hidden, "fwd", residual)
    eps = EPS[hidden]
    launched = set()
    for rows in _switch_rows(arch, hidden, "fwd", residual):
        selected = loader.selected_modules(arch, hidden, "fwd", residual, rows=rows)
        assert len(selected) == 1, (
            f"hidden={hidden} residual={residual} rows={rows}: {selected}"
        )
        launched.add(selected[0])
        label = f"hidden={hidden} residual={residual} rows={rows}"
        check = ref.Draws(label, prefix=f"T={rows} ")
        for draw in range(check.draws):
            generator = _generator(hidden, rows, 43, residual, draw=draw)
            x = _rows(rows, hidden, generator=generator)
            w = _weight(hidden, generator)
            u = _rows(rows, hidden, generator=generator) if residual else None
            outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
            forward = {"y": outputs[0], "rstd": outputs[1]}
            if residual:
                forward["h_new"] = outputs[2]
                _h_new_is_the_rounded_sum(outputs[2], x, u, label)
            check.forward(draw, forward, x, w, eps, residual=u)
        check.check(record_property)
    assert launched == set(names), (
        f"hidden={hidden} residual={residual}: never launched {set(names) - launched}"
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_backward_programs_switch_on_row_count(hidden: int, record_property) -> None:
    """Exactly one pair -- a main program and its dw reduce program -- or one
    single program (``claim_sum_f32``) runs per call, every exported backward
    program runs for some row count, and forward + backward meet the naive BF16
    tier on both sides of each switch point."""

    _skip_unless_hidden(hidden)
    arch = loader.arch_for_capability(_capability())
    names = loader.route_modules(arch, hidden, "bwd", False)
    launched = set()
    for rows in _switch_rows(arch, hidden, "bwd", False):
        selected = loader.selected_modules(arch, hidden, "bwd", False, rows=rows)
        workspace = loader.MODULES[selected[0]].get("workspace_rule") or {}
        if workspace.get("kind") == "claim_sum_f32":
            assert len(selected) == 1, f"hidden={hidden} rows={rows}: {selected}"
        else:
            assert len(selected) == 2, f"hidden={hidden} rows={rows}: {selected}"
            main, reduce = (loader.MODULES[name] for name in selected)
            assert workspace and reduce["role"] == "bwd_reduce"
        launched.update(selected)
        _check_case(hidden, rows, record=record_property, tag=f"T={rows} ")
    assert launched == set(names), (
        f"hidden={hidden}: never launched {set(names) - launched}"
    )


@pytest.mark.parametrize("hidden", (6144, 5120, 4096, 2048, 1536, 256))
@pytest.mark.parametrize("rows", (1, 129, 4097))
@pytest.mark.parametrize("below", (1, 0))
def test_backward_below_the_measured_sm_count(
    hidden: int, rows: int, below: int, monkeypatch: pytest.MonkeyPatch, record_property
) -> None:
    """On a device with fewer SMs than ``cake_windows.ACCURACY_MIN_SMS`` (emulated: the host's SM count patched to one
    below it, or to 16, which shrinks the chunk table and the grids with it) every call launches one backward
    program: one of the route's compensated programs where the route has an accuracy edge
    (``cake_windows.ACCURACY_EDGES``), else the program of the row count's window, and forward + backward meet the
    naive BF16 tier."""

    _skip_unless_hidden(hidden)
    arch = loader.arch_for_capability(_capability())
    edge = windows.ACCURACY_EDGES.get(arch, {}).get(
        loader.route_key(hidden, "bwd", False)
    )
    programs = loader.program_windows(arch, hidden, "bwd", False)
    if edge is None:
        expected = {
            library
            for low, high, library in programs
            if low < rows and (high is None or rows <= high)
        }
    else:
        expected = {library for low, _high, library in programs if low >= edge}
    sms = windows.ACCURACY_MIN_SMS[arch] - 1 if below else 16
    monkeypatch.setattr(loader, "sm_count", lambda _index: sms)
    launched: list[str] = []
    run_module = loader.run_module

    def record(name, *args, **kwargs):
        launched.append(name)
        return run_module(name, *args, **kwargs)

    monkeypatch.setattr(loader, "run_module", record)
    _check_case(hidden, rows, record=record_property)
    mains = [
        loader.MODULES[name]["library"]
        for name in launched
        if loader.MODULES[name].get("workspace_rule") is not None
    ]
    # one backward call per draw
    assert len(mains) == ref.DRAWS and len(set(mains)) == 1, (sms, rows, set(mains))
    assert mains[0] in expected, (sms, rows, mains[0], expected)


# --------------------------------------------------------------------------- the program windows


def _route_edges(
    arch: str, hidden: int, kind: str, residual: bool
) -> list[tuple[int, str]]:
    """``(edge, class)`` of every window edge a call of the route crosses, the edges from the registry's windows
    (``program_windows``) and the class from the exporter's measurements (``cake_windows.EDGES``); a backward route
    crosses the forward's edges inside its windows as well (the backward consumes the forward's rstd) and takes the
    forward's class there."""

    route = loader.route_key(hidden, kind, residual)
    classes = dict(windows.EDGES[arch][route])
    edges = {}
    for _low, high, _library in loader.program_windows(arch, hidden, kind, residual):
        if high is not None:
            edges[high] = classes[high]
    if kind == "bwd":
        forward = dict(windows.EDGES[arch][loader.route_key(hidden, "fwd", residual)])
        for _low, high, _library in loader.program_windows(
            arch, hidden, "fwd", residual
        ):
            if high is not None:
                edges.setdefault(high, forward[high])
    return sorted(edges.items())


def _window_spans(
    arch: str, hidden: int, kind: str, residual: bool
) -> list[tuple[int, int]]:
    """``(first, last)`` row counts of every window of the route (a backward route's windows cut at the forward's
    edges too); the open last window ends at ``OPEN_WINDOW_ROWS``, or ``OPEN_WINDOW_ROWS`` past the last edge when
    that edge is not below it."""

    edges = [edge for edge, _cls in _route_edges(arch, hidden, kind, residual)]
    top = max(edges, default=0)
    end = OPEN_WINDOW_ROWS if top < OPEN_WINDOW_ROWS else top + OPEN_WINDOW_ROWS
    bounds = [0, *edges, end]
    return [(low + 1, high) for low, high in zip(bounds, bounds[1:], strict=False)]


def _base_inputs(
    hidden: int, rows: int, kind: str, residual: bool
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One random input of ``rows`` rows whose prefixes the window tests slice, and the weight."""

    generator = _generator(hidden, 61, kind == "bwd", residual)
    base = {"x": _rows(rows, hidden, generator=generator)}
    w = _weight(hidden, generator)
    if residual:
        base["u"] = _rows(rows, hidden, generator=generator)
    if kind == "bwd":
        base["g"] = _rows(rows, hidden, generator=generator)
        if residual:
            base["g_h"] = _rows(rows, hidden, generator=generator)
    return base, w


def _route_outputs(
    kind: str,
    residual: bool,
    base: dict[str, torch.Tensor],
    rows: int,
    w: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    """The outputs of one call of the route on the first ``rows`` rows of ``base``: y, rstd (and h_new) of the
    forward, and dx as well for a backward route (dw sums every row and depends on T by definition)."""

    x = base["x"][:rows]
    u = base["u"][:rows] if residual else None
    outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
    result = {"y": outputs[0], "rstd": outputs[1]}
    if residual:
        result["h_new"] = outputs[2]
    if kind == "bwd":
        g_h = base["g_h"][:rows] if residual else None
        dx, _dw = cake_rmsnorm_det_train_backward(
            base["g"][:rows],
            outputs[2] if residual else x,
            w,
            outputs[1],
            g_residual=g_h,
        )
        result["dx"] = dx
    return result


def _differing_rows(a: torch.Tensor, b: torch.Tensor) -> int:
    differs = a != b
    if differs.ndim > 1:
        differs = differs.reshape(differs.shape[0], -1).any(dim=1)
    return int(differs.sum().item())


def _assert_edge_class(
    low: dict[str, torch.Tensor], high: dict[str, torch.Tensor], cls: str, label: str
) -> None:
    """The outputs of T = edge (``low``) and T = edge + 1 (``high``) on their common rows behave as the edge's class
    says: ``same`` -- bitwise identical; ``ulp`` -- rstd differs on some rows, and y and dx differ
    only on rows whose rstd differs; ``dx`` -- y and rstd identical, dx differs on some rows (the base and the
    bulk-copy backward reduce a row in different orders); ``unmeasured`` -- h_new identical.  How far the outputs
    move is not bounded here (no tolerance); accuracy is judged against the naive BF16 tier."""

    rows = low["rstd"].shape[0]
    common = {key: value[:rows] for key, value in high.items()}
    assert set(low) == set(common), label
    if cls == "same":
        for key, value in low.items():
            assert torch.equal(value, common[key]), (
                f"{label}: {key} differs across an edge of class 'same'"
            )
        return
    assert cls in ("ulp", "dx", "unmeasured"), (label, cls)
    if "h_new" in low:
        assert torch.equal(low["h_new"], common["h_new"]), f"{label}: h_new differs"
    same_rstd = low["rstd"] == common["rstd"]
    rstd_rows = rows - int(same_rstd.sum().item())
    if cls == "dx":
        assert rstd_rows == 0 and torch.equal(low["y"], common["y"]), (
            f"{label}: the forward differs across an edge of class 'dx' ({rstd_rows} rstd rows)"
        )
    elif cls == "ulp":
        assert rstd_rows > 0, (
            f"{label}: rstd is identical across an edge of class 'ulp' (re-measure the edge)"
        )
        assert torch.equal(low["y"][same_rstd], common["y"][same_rstd]), (
            f"{label}: y differs on rows whose rstd agrees"
        )
    if "dx" in low:
        if cls == "ulp":
            assert torch.equal(low["dx"][same_rstd], common["dx"][same_rstd]), (
                f"{label}: dx differs on rows whose rstd agrees"
            )
        elif cls == "dx" and rows >= 4096:
            assert _differing_rows(low["dx"], common["dx"]) > 0, (
                f"{label}: dx is identical across an edge of class 'dx' (re-measure the edge)"
            )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("kind", ("fwd", "bwd"))
@pytest.mark.parametrize("residual", (False, True))
def test_outputs_are_prefix_invariant_within_windows(
    hidden: int, kind: str, residual: bool
) -> None:
    """Within one program window a row's outputs do not depend on T: for every window of the route (a backward
    route's windows cut at the forward's edges too) the calls at the window's first and last row count
    (``_window_spans``' end standing for an open window) return bitwise-identical y, rstd, h_new and dx on their
    common rows."""

    _skip_unless_route(hidden, residual)
    arch = loader.arch_for_capability(_capability())
    label = f"{arch} {loader.route_key(hidden, kind, residual)}"
    spans = _window_spans(arch, hidden, kind, residual)
    base, w = _base_inputs(hidden, max(last for _first, last in spans), kind, residual)
    for first, last in spans:
        low = _route_outputs(kind, residual, base, first, w, EPS[hidden])
        high = _route_outputs(kind, residual, base, last, w, EPS[hidden])
        for key, value in low.items():
            assert torch.equal(value, high[key][:first]), (
                f"{label} window [{first}, {last}]: {key} depends on T"
            )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("kind", ("fwd", "bwd"))
@pytest.mark.parametrize("residual", (False, True))
def test_window_edges_match_the_measured_classes(
    hidden: int, kind: str, residual: bool
) -> None:
    """At every window edge of the route the outputs of T = edge and T = edge + 1 behave on their common rows as the
    exporter measured (``cake_windows.EDGES``, see ``_assert_edge_class``); an unmeasured edge is named."""

    _skip_unless_route(hidden, residual)
    arch = loader.arch_for_capability(_capability())
    route = loader.route_key(hidden, kind, residual)
    edges = _route_edges(arch, hidden, kind, residual)
    if not edges:
        pytest.skip(f"{arch} {route} has one program window")
    base, w = _base_inputs(hidden, edges[-1][0] + 1, kind, residual)
    unmeasured = []
    for edge, cls in edges:
        low = _route_outputs(kind, residual, base, edge, w, EPS[hidden])
        high = _route_outputs(kind, residual, base, edge + 1, w, EPS[hidden])
        _assert_edge_class(low, high, cls, f"{arch} {route} edge {edge} (class {cls})")
        if cls == "unmeasured":
            unmeasured.append(edge)
    if unmeasured:
        pytest.skip(f"{arch} {route}: edges {unmeasured} are unmeasured")


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("stride_pad", STRIDE_PADS[1:])
@pytest.mark.parametrize("rows", (7, 129, 16231))
def test_row_strided_inputs(
    hidden: int, stride_pad, rows: int, record_property
) -> None:
    _skip_unless_hidden(hidden)
    _check_case(
        hidden, rows, record=record_property, x_pad=stride_pad, g_pad=stride_pad
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("scale", SCALES[1:])
@pytest.mark.parametrize("rows", (7, 16231))
def test_input_scales(hidden: int, scale: float, rows: int, record_property) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, record=record_property, scale=scale)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (1, 129))
def test_zero_input_has_no_nan(hidden: int, rows: int, record_property) -> None:
    """``x = 0`` with random ``g`` and ``w``: ``y`` and ``dw`` are exactly 0, ``rstd`` is ``eps^(-1/2)`` and ``dx``
    is ``rstd g w``; every output finite and judged over the draws (``g`` and ``w`` change with the seed)."""
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, record=record_property, zeros=True)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_extreme_row(hidden: int, record_property) -> None:
    """One row scaled by 1e3 with one element 1e4 (normal BF16 values) among unit-scale rows."""
    _skip_unless_hidden(hidden)
    _check_case(hidden, 257, record=record_property, spike=True)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("stride_pad", STRIDE_PADS)
@pytest.mark.parametrize("rows", (1, 129, 16231))
def test_residual_fusion(hidden: int, stride_pad, rows: int, record_property) -> None:
    _skip_unless_hidden(hidden)
    if not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="fwd",
        residual=True,
    ):
        pytest.skip(f"hidden={hidden} has no exported residual route on this device")
    _check_case(
        hidden,
        rows,
        record=record_property,
        x_pad=stride_pad,
        g_pad=stride_pad,
        residual=True,
    )


MISALIGNED_LAYOUTS = ("offset1", "stride_h_plus_1")


def _misaligned_rows(
    rows: int, hidden: int, layout: str, *, generator: torch.Generator
) -> torch.Tensor:
    """A ``[rows, hidden]`` BF16 view the kernels cannot read in place.

    ``offset1``: ``buf[:, 1:1 + H]`` of a ``[T, H + 16]`` buffer -- odd column offset
    (storage base 2 bytes past a 16-byte boundary; aligned row stride).
    ``stride_h_plus_1``: ``buf[:, :H]`` of a ``[T, H + 1]`` buffer -- odd row stride.
    """

    storage_rows = max(rows, 1)
    if layout == "offset1":
        base = torch.randn(
            storage_rows,
            hidden + 16,
            dtype=torch.float32,
            device=_device(),
            generator=generator,
        )
        view = base.to(torch.bfloat16)[:rows, 1 : 1 + hidden]
    else:
        base = torch.randn(
            storage_rows,
            hidden + 1,
            dtype=torch.float32,
            device=_device(),
            generator=generator,
        )
        view = base.to(torch.bfloat16)[:rows, :hidden]
    assert view.data_ptr() % 16 != 0 or (view.stride(0) * view.element_size()) % 16 != 0
    return view


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("layout", MISALIGNED_LAYOUTS)
@pytest.mark.parametrize("residual", (False, True))
def test_misaligned_layouts_are_materialized(
    hidden: int, layout: str, residual: bool, record_property
) -> None:
    """Odd column offsets and odd row strides on every ``[T, H]`` input and output (fwd, bwd, residual)."""

    _skip_unless_route(hidden, residual)
    eps = EPS[hidden]
    rows = 129
    label = f"hidden={hidden} layout={layout} residual={residual}"
    check = ref.Draws(label)
    for draw in range(check.draws):
        generator = _generator(
            hidden, rows, 41, layout == "offset1", residual, draw=draw
        )
        x = _misaligned_rows(rows, hidden, layout, generator=generator)
        w = _weight(hidden, generator)
        u = (
            _misaligned_rows(rows, hidden, layout, generator=generator)
            if residual
            else None
        )
        g = _misaligned_rows(rows, hidden, layout, generator=generator)
        g_h = (
            _misaligned_rows(rows, hidden, layout, generator=generator)
            if residual
            else None
        )

        outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
        y, rstd = outputs[0], outputs[1]
        assert y.is_contiguous() and tuple(y.shape) == (rows, hidden)
        forward = {"y": y, "rstd": rstd}
        if residual:
            forward["h_new"] = outputs[2]
            _h_new_is_the_rounded_sum(outputs[2], x, u, label)
        check.forward(draw, forward, x, w, eps, residual=u)
        # Caller-provided outputs with the same awkward layout are honoured by copy-back.
        y_user = _misaligned_rows(rows, hidden, layout, generator=generator)
        outputs_user = cake_rmsnorm_det_train_forward(x, w, eps, residual=u, out=y_user)
        assert outputs_user[0] is y_user and torch.equal(y_user, y), (
            f"out copy-back {label}"
        )

        # Backward: the normalized tensor, both gradients and the dx buffer are misaligned views.
        normalized = _misaligned_rows(rows, hidden, layout, generator=generator)
        normalized.copy_(outputs[2] if residual else x)
        dx, dw = cake_rmsnorm_det_train_backward(g, normalized, w, rstd, g_residual=g_h)
        assert dx.is_contiguous() and tuple(dx.shape) == (rows, hidden)
        check.backward(draw, {"dx": dx, "dw": dw}, g, normalized, w, rstd, eps, g_h=g_h)
        dx_user = _misaligned_rows(rows, hidden, layout, generator=generator)
        dx_again, dw_again = cake_rmsnorm_det_train_backward(
            g, normalized, w, rstd, g_residual=g_h, dx=dx_user
        )
        assert dx_again is dx_user and torch.equal(dx_user, dx), f"dx copy-back {label}"
        assert torch.equal(dw_again, dw), f"dw bitwise across layouts {label}"
    check.check(record_property)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_half_vector_aligned_weight_is_materialized(
    hidden: int, record_property
) -> None:
    """A contiguous ``w`` whose storage base is 16 bytes past a 32-byte boundary.

    The backward kernels load the weight with 16-byte vectors and read this view
    in place; the forward kernels load it with 32-byte vectors, so the forward
    must copy it (an in-place read would be a misaligned vector load).  Both
    meet the naive BF16 tier and give the aligned weight's results bitwise.
    """

    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    rows = 129
    label = f"hidden={hidden} 16-byte aligned w"
    check = ref.Draws(label)
    storage = torch.empty(hidden + 16, dtype=torch.bfloat16, device=_device())
    offset = ((16 - storage.data_ptr() % 32) % 32) // storage.element_size()
    w = storage[offset : offset + hidden]
    for draw in range(check.draws):
        generator = _generator(hidden, rows, 47, draw=draw)
        x = _rows(rows, hidden, generator=generator)
        w_aligned = _weight(hidden, generator)
        w.copy_(w_aligned)
        assert w.is_contiguous() and w.data_ptr() % 32 == 16, "test layout"

        y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
        check.forward(draw, {"y": y, "rstd": rstd}, x, w, eps)
        y_aligned, rstd_aligned = cake_rmsnorm_det_train_forward(x, w_aligned, eps)
        assert torch.equal(y, y_aligned) and torch.equal(rstd, rstd_aligned), label

        g = _rows(rows, hidden, generator=generator)
        dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
        check.backward(draw, {"dx": dx, "dw": dw}, g, x, w, rstd, eps)
        dx_aligned, dw_aligned = cake_rmsnorm_det_train_backward(g, x, w_aligned, rstd)
        assert torch.equal(dx, dx_aligned) and torch.equal(dw, dw_aligned), label
    check.check(record_property)


# --------------------------------------------------------------------------- N-D inputs (the row collapse)


def _nd_rows(
    rows: int, heads: int, hidden: int, *, pitch_pad: int, generator: torch.Generator
) -> torch.Tensor:
    """BF16 ``[rows, heads, hidden]`` view of a ``[rows, heads, hidden + pitch_pad]`` buffer: strides
    ``(heads x (hidden + pad), hidden + pad, 1)``, the issue's layout when ``pad > 0``."""

    base = torch.randn(
        max(rows, 1),
        heads,
        hidden + pitch_pad,
        dtype=torch.float32,
        device=_device(),
        generator=generator,
    ).to(torch.bfloat16)
    return base[:rows, :, :hidden]


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("heads", (1, 3, 24))
@pytest.mark.parametrize("rows", (0, 1, 7, 129))
@pytest.mark.parametrize("pitch_pad", (0, 16, 256))
def test_nd_inputs_collapse_to_rows(
    hidden: int, heads: int, rows: int, pitch_pad: int
) -> None:
    """A ``[T, heads, H]`` input whose leading dimensions are contiguous with one another (strides
    ``(heads x pitch, pitch, 1)``) is the ``[T x heads, H]`` call: y, rstd, dx and dw bitwise equal to the 2-D call
    on the same rows, the outputs in the input's leading shape, a flat rstd accepted by the backward too."""

    _skip_unless_hidden(hidden)
    generator = _generator(hidden, heads, rows, pitch_pad, 71)
    x3 = _nd_rows(rows, heads, hidden, pitch_pad=pitch_pad, generator=generator)
    if (hidden, heads, pitch_pad) == (256, 24, 256):
        assert tuple(x3.stride()) == (12288, 512, 1)  # the issue's per-head view
    w = _weight(hidden, generator)
    g3 = _nd_rows(rows, heads, hidden, pitch_pad=pitch_pad, generator=generator)
    eps = EPS[hidden]
    x2 = x3.reshape(rows * heads, hidden)
    g2 = g3.reshape(rows * heads, hidden)
    y3, rstd3 = cake_rmsnorm_det_train_forward(x3, w, eps)
    y2, rstd2 = cake_rmsnorm_det_train_forward(x2, w, eps)
    assert tuple(y3.shape) == (rows, heads, hidden) and y3.is_contiguous()
    assert tuple(rstd3.shape) == (rows, heads) and rstd3.dtype == torch.float32
    assert torch.equal(y3.reshape(-1, hidden), y2) and torch.equal(
        rstd3.reshape(-1), rstd2
    )
    dx3, dw3 = cake_rmsnorm_det_train_backward(g3, x3, w, rstd3)
    dx2, dw2 = cake_rmsnorm_det_train_backward(g2, x2, w, rstd2)
    assert tuple(dx3.shape) == (rows, heads, hidden) and dx3.is_contiguous()
    assert torch.equal(dx3.reshape(-1, hidden), dx2) and torch.equal(dw3, dw2)
    dx3_flat, dw3_flat = cake_rmsnorm_det_train_backward(g3, x3, w, rstd3.reshape(-1))
    assert torch.equal(dx3_flat, dx3) and torch.equal(dw3_flat, dw3)
    if rows == 0:
        return
    # Preallocated N-D outputs: contiguous ones are written in place, a strided one receives a copy.
    out = torch.empty_like(y3)
    rstd_out = torch.empty(rows, heads, dtype=torch.float32, device=_device())
    outputs = cake_rmsnorm_det_train_forward(x3, w, eps, out=out, rstd=rstd_out)
    assert outputs[0] is out and outputs[1] is rstd_out
    assert torch.equal(out, y3) and torch.equal(rstd_out, rstd3)
    strided = torch.empty(
        rows, heads, hidden + 16, dtype=torch.bfloat16, device=_device()
    )[..., :hidden]
    outputs = cake_rmsnorm_det_train_forward(x3, w, eps, out=strided)
    assert outputs[0] is strided and torch.equal(strided, y3)
    dx_out = torch.empty_like(dx3)
    outputs = cake_rmsnorm_det_train_backward(g3, x3, w, rstd3, dx=dx_out)
    assert outputs[0] is dx_out and torch.equal(dx_out, dx3)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_nd_autograd_matches_the_2d_call(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    rows, heads = 7, 3
    generator = _generator(hidden, rows, heads, 73)
    x3 = _nd_rows(rows, heads, hidden, pitch_pad=16, generator=generator)
    g3 = _nd_rows(rows, heads, hidden, pitch_pad=0, generator=generator)
    w = _weight(hidden, generator)
    x3_leaf = x3.detach().clone().requires_grad_(True)
    x2_leaf = x3.reshape(rows * heads, hidden).detach().clone().requires_grad_(True)
    w3 = w.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    y3 = cake_rmsnorm_det(x3_leaf, w3, EPS[hidden])
    y2 = cake_rmsnorm_det(x2_leaf, w2, EPS[hidden])
    assert tuple(y3.shape) == (rows, heads, hidden)
    assert torch.equal(y3.reshape(-1, hidden), y2)
    y3.backward(g3)
    y2.backward(g3.reshape(-1, hidden))
    assert tuple(x3_leaf.grad.shape) == (rows, heads, hidden)
    assert torch.equal(x3_leaf.grad.reshape(-1, hidden), x2_leaf.grad)
    assert torch.equal(w3.grad, w2.grad)


def test_the_issues_hidden_256_view_collapses_before_the_route_lookup() -> None:
    """The ``[T, 24, 256]`` view with strides ``(12288, 512, 1)`` of a ``[T, 24, 512]`` buffer collapses to
    ``24 T`` rows of 256: the host reaches the route lookup, which answers for hidden 256 with the width-256
    route (a device without it gets the ValueError naming hidden=256, not the shape)."""

    rows = 16
    buffer = torch.zeros(rows, 24, 512, dtype=torch.bfloat16, device=_device())
    x = buffer[..., :256]
    assert tuple(x.stride()) == (12288, 512, 1)
    w = torch.ones(256, dtype=torch.bfloat16, device=_device())
    if is_cake_rmsnorm_det_train_supported(_device(), 256):
        y, rstd = cake_rmsnorm_det_train_forward(x, w, 1e-6)
        assert tuple(y.shape) == (rows, 24, 256) and tuple(rstd.shape) == (rows, 24)
        dx, dw = cake_rmsnorm_det_train_backward(x, x, w, rstd)
        assert tuple(dx.shape) == (rows, 24, 256) and tuple(dw.shape) == (256,)
        return
    with pytest.raises(
        ValueError, match=r"no exported Cake RMSNorm training forward for hidden=256 "
    ):
        cake_rmsnorm_det_train_forward(x, w, 1e-6)
    rstd = torch.ones(rows, 24, dtype=torch.float32, device=_device())
    with pytest.raises(
        ValueError, match=r"no exported Cake RMSNorm training backward for hidden=256 "
    ):
        cake_rmsnorm_det_train_backward(x, x, w, rstd)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES[:1])
def test_nd_inputs_outside_the_collapse_rule_are_refused(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    rows, heads = 8, 4
    eps = EPS[hidden]
    w = _weight(hidden, _generator(hidden, 5))
    # Heads-major storage viewed as [T, heads, H]: stride(0) != stride(1) * heads.
    x = torch.zeros(
        heads, rows, hidden, dtype=torch.bfloat16, device=_device()
    ).transpose(0, 1)
    assert tuple(x.stride()) == (hidden, rows * hidden, 1)
    with pytest.raises(ValueError, match=r"cannot be viewed as \[32, "):
        cake_rmsnorm_det_train_forward(x, w, eps)
    rstd = torch.ones(rows, heads, dtype=torch.float32, device=_device())
    with pytest.raises(ValueError, match=r"cannot be viewed as \[32, "):
        cake_rmsnorm_det_train_backward(x.contiguous(), x, w, rstd)
    # A feature dimension that is not contiguous.
    x = torch.zeros(rows, heads, hidden, 2, dtype=torch.bfloat16, device=_device())[
        ..., 0
    ]
    with pytest.raises(ValueError, match="cannot be viewed as"):
        cake_rmsnorm_det_train_forward(x, w, eps)
    # One dimension is not enough.
    with pytest.raises(ValueError, match=r"must be a \[T, H\] tensor"):
        cake_rmsnorm_det_train_forward(
            torch.zeros(hidden, dtype=torch.bfloat16, device=_device()), w, eps
        )
    # Companions take x's N-D shape.
    x3 = torch.zeros(rows, heads, hidden, dtype=torch.bfloat16, device=_device())
    with pytest.raises(ValueError, match="residual must have shape"):
        cake_rmsnorm_det_train_forward(
            x3, w, eps, residual=x3.reshape(rows * heads, hidden)
        )
    with pytest.raises(ValueError, match="g must have shape"):
        cake_rmsnorm_det_train_backward(x3.reshape(rows * heads, hidden), x3, w, rstd)
    # Dimensions of size one are free: [T, 1, H] slices of a wider buffer collapse to T rows.
    x = torch.zeros(rows, 3, hidden, dtype=torch.bfloat16, device=_device())[:, 1:2, :]
    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    assert tuple(y.shape) == (rows, 1, hidden) and tuple(rstd.shape) == (rows, 1)


# --------------------------------------------------------------------------- FP32 effective weights


def _fp32_route(hidden: int, kind: str, residual: bool) -> bool:
    return loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype="f32",
    )


def _skip_unless_fp32_routes(hidden: int, residual: bool) -> None:
    _skip_unless_route(hidden, residual)
    if not (
        _fp32_route(hidden, "fwd", residual) and _fp32_route(hidden, "bwd", residual)
    ):
        pytest.skip(
            f"hidden={hidden} residual={residual}: no FP32-weight route on this device"
        )


def _fp32_case(
    hidden: int,
    rows: int,
    residual: bool,
    w32: torch.Tensor,
    generator: torch.Generator,
) -> tuple[
    tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    """One forward + backward with the FP32 weight ``w32``: ``(forward outputs, dx, dw, inputs)``."""

    eps = EPS[hidden]
    x = _rows(rows, hidden, generator=generator)
    u = _rows(rows, hidden, generator=generator) if residual else None
    g = _rows(rows, hidden, generator=generator)
    g_h = _rows(rows, hidden, generator=generator) if residual else None
    outputs = cake_rmsnorm_det_train_forward(x, w32, eps, residual=u)
    normalized = outputs[2] if residual else x
    dx, dw = cake_rmsnorm_det_train_backward(
        g, normalized, w32, outputs[1], g_residual=g_h
    )
    return (
        outputs,
        dx,
        dw,
        {"x": x, "u": u, "g": g, "g_h": g_h, "normalized": normalized},
    )


# The backward windows where the FP32-weight route runs a program that the BF16 route does not run over those rows
# (``RECUT_FP32_ROUTES`` of the pairs test): a program of the BF16 route's earlier windows where the BF16 route runs a
# pipelined program that has no FP32-weight build, and at T = 1 another column-task program than the BF16 route's.
# (arch, hidden) -> {FP32-weight window: (the BF16 window, the first 8 hex digits of the BF16 program) whose program
# and dw reduce are the BF16 twins of the FP32-weight window's}.  The twin records share the FP32-weight records' grid
# and workspace rules; a twin window the BF16 route no longer runs keeps its records in the registry under no route.
RECUT_FP32_BACKWARD_TWINS: dict[
    tuple[str, int], dict[tuple[int, int], tuple[tuple[int, int], str]]
] = {
    ("sm_100a", 2048): {
        (768, 1024): ((288, 768), "caf6978e"),
        (1024, 16384): ((1024, 2047), "c73074cf"),
        (16384, 58880): ((22528, 58880), "ceeef043"),
    },
    ("sm_100a", 5120): {(0, 1): ((1, 256), "dd3a3573")},
    ("sm_100a", 6144): {(384, 4096): ((384, 767), "5c804aed")},
    ("sm_103a", 4096): {(0, 1): ((0, 152), "12d53fe5")},
    ("sm_103a", 6144): {
        (0, 1): ((0, 20), "44508df7"),
        (256, 4096): ((256, 767), "5c804aed"),
    },
}

# The backward windows of the swapped FP32-weight routes (``SWAPPED_FP32_ROUTES`` of the pairs test), where the
# FP32-weight route runs, under a record of the BF16 route's host plan, the twin of another program than the BF16
# route's: (arch, hidden, residual) -> {FP32-weight window: the first 8 hex digits of the BF16 program whose record over
# that window (no route selects it) is the BF16 twin of the FP32-weight window's}.  SM100 H = 5120 runs the twin of
# c145ef19 above 6144 rows where the BF16 route runs b482c84d, SM103 H = 4096 the twin of 850eec3a over
# (48640, 51200] where it runs 24643edc, and SM100 H = 6144 with a residual keeps the twin of c1a3be94 above 42496
# rows where the BF16 route runs 84385468.
SWAPPED_FP32_BACKWARD_TWINS: dict[
    tuple[str, int, bool], dict[tuple[int, int | None], str]
] = {
    ("sm_100a", 5120, False): {(6144, 50688): "c145ef19", (50688, None): "c145ef19"},
    ("sm_103a", 4096, False): {(48640, 51200): "850eec3a"},
    ("sm_100a", 6144, True): {(42496, None): "c1a3be94"},
}


def _twin_records(names, twin: tuple[int, int], program: str) -> list[str]:
    """The records among ``names`` whose window is ``twin``: the program ``program`` (the first 8 hex digits of its
    name) and the dw reduces."""

    return [
        name
        for name in names
        if (
            (loader.MODULES[name]["select_rule"] or {}).get("above"),
            (loader.MODULES[name]["select_rule"] or {}).get("at_most"),
        )
        == twin
        and (
            loader.MODULES[name]["role"] == "bwd_reduce"
            or loader.MODULES[name]["library"].startswith(
                f"cake_rmsnorm_det_train_{program}"
            )
        )
    ]


def _swapped_backward_twin(
    arch: str, hidden: int, residual: bool, rows: int
) -> tuple[str, ...]:
    """The BF16 twin modules of the FP32-weight backward program a call of ``rows`` tokens launches in a window of
    ``SWAPPED_FP32_BACKWARD_TWINS``: the BF16 record of the window's program over that window, with the role, grid
    rule and workspace rule of the FP32-weight record, then the dw reduce of the FP32-weight call; empty elsewhere."""

    for (above, at_most), program in SWAPPED_FP32_BACKWARD_TWINS.get(
        (arch, hidden, residual), {}
    ).items():
        if above < rows and (at_most is None or rows <= at_most):
            selected = loader.selected_modules(
                arch, hidden, "bwd", residual, rows=rows, w_dtype="f32"
            )
            reduces = [n for n in selected if loader.MODULES[n]["role"] == "bwd_reduce"]
            (main,) = [n for n in selected if n not in reduces]
            fp32 = loader.MODULES[main]
            twins = [
                name
                for name, record in loader.MODULES.items()
                if record["library"].startswith(f"cake_rmsnorm_det_train_{program}")
                and arch in record["arches"]
                and (record["select_rule"] or {}).get("above") == above
                and (record["select_rule"] or {}).get("at_most") == at_most
                and record["role"] == fp32["role"].replace("_w32", "")
                and record["grid_rule"] == fp32["grid_rule"]
                and record["workspace_rule"] == fp32["workspace_rule"]
            ]
            assert len(twins) == 1, (arch, hidden, residual, program, twins)
            return (twins[0], *reduces)
    return ()


def _fp32_backward_twin(hidden: int, residual: bool, rows: int) -> tuple[str, ...]:
    """The BF16 twin modules (the main program and its dw reduce, or a column-task program alone) of the FP32-weight
    backward program a call of ``rows`` tokens launches in a window of ``RECUT_FP32_BACKWARD_TWINS`` or
    ``SWAPPED_FP32_BACKWARD_TWINS``; empty elsewhere.  The twin records of a re-cut window are the BF16 route's
    where it runs the twin's window, else the registry's records of that window."""

    arch = loader.arch_for_capability(_capability())
    recut = {} if residual else RECUT_FP32_BACKWARD_TWINS.get((arch, hidden), {})
    for (above, at_most), (twin, program) in recut.items():
        if above < rows <= at_most:
            names = _twin_records(
                loader.route_modules(arch, hidden, "bwd", False), twin, program
            ) or _twin_records(
                (
                    name
                    for name, record in loader.MODULES.items()
                    if arch in record["arches"]
                    and int(record["hidden"]) == hidden
                    and "_w32" not in record["role"]
                    and not record["role"].endswith("_residual")
                ),
                twin,
                program,
            )
            mains = [n for n in names if loader.MODULES[n]["role"] != "bwd_reduce"]
            assert len(mains) == 1, (arch, hidden, twin, program, names)
            workspace = loader.MODULES[mains[0]]["workspace_rule"] or {}
            if workspace.get("kind") != "chunk_partials":
                return (mains[0],)  # a column-task program runs alone
            reduces = [n for n in names if loader.MODULES[n]["role"] == "bwd_reduce"]
            assert len(reduces) == 1, (arch, hidden, twin, names)
            return (mains[0], reduces[0])
    return _swapped_backward_twin(arch, hidden, residual, rows)


def _launching(twin: tuple[str, ...]):
    """A ``selected_modules`` under which a BF16 backward call launches the modules ``twin``; every other call
    selects its route's modules."""

    original = loader.selected_modules

    def select(arch, hidden, kind, residual, *, rows, w_dtype="bf16"):
        if kind == "bwd" and w_dtype == "bf16":
            return twin
        return original(arch, hidden, kind, residual, rows=rows, w_dtype=w_dtype)

    return select


def _window_rows(hidden: int, residual: bool) -> tuple[int, ...]:
    """The row counts that visit every window of the route's forward and backward programs:
    both sides of every window edge, the smallest counts and the training shape."""

    arch = loader.arch_for_capability(_capability())
    rows = {1, 2, 129, 16231}
    for kind in ("fwd", "bwd"):
        for above, at_most, _library in loader.program_windows(
            arch, hidden, kind, residual
        ):
            rows.add(above + 1)
            if at_most is not None:
                rows.update((at_most, at_most + 1))
    return tuple(sorted(rows))


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
def test_fp32_weight_equal_to_the_bf16_weight_matches_it_bitwise(
    hidden: int, residual: bool
) -> None:
    """``w32 = float(w16)`` gives bitwise the BF16 path's y, rstd, h_new, dx and dw in every
    window of the route: the FP32-weight programs are the BF16 programs with the weight loaded
    at FP32, selected over the same windows.  In the backward windows of
    ``RECUT_FP32_BACKWARD_TWINS`` the FP32-weight route keeps a program of the BF16 route's
    earlier windows, and in those of ``SWAPPED_FP32_BACKWARD_TWINS`` it runs the twin of a
    program the BF16 route does not run there, so there its dx and dw are bitwise those of a
    BF16 call that launches the BF16 twin of that program (and its dw reduce) over the same
    rows."""

    _skip_unless_fp32_routes(hidden, residual)
    for rows in _window_rows(hidden, residual):
        generator = _generator(hidden, rows, residual, 83)
        w16 = _weight(hidden, generator)
        seed = generator.get_state()
        outputs16, dx16, dw16, _ = _fp32_case(hidden, rows, residual, w16, generator)
        generator.set_state(seed)
        outputs32, dx32, dw32, _ = _fp32_case(
            hidden, rows, residual, w16.float(), generator
        )
        assert len(outputs16) == len(outputs32) == (3 if residual else 2)
        for name, a, b in zip(
            ("y", "rstd", "h_new")[: len(outputs16)], outputs16, outputs32, strict=True
        ):
            assert torch.equal(a, b), (
                f"hidden={hidden} residual={residual} rows={rows}: {name} differs"
            )
        label = f"hidden={hidden} residual={residual} rows={rows}"
        assert torch.equal(dx16, dx32), f"{label}: dx differs"
        twin = _fp32_backward_twin(hidden, residual, rows)
        if twin:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(pkg, "selected_modules", _launching(twin))
                generator.set_state(seed)
                _outputs, dx16, dw16, _ = _fp32_case(
                    hidden, rows, residual, w16, generator
                )
            label = f"{label} (BF16 twin {twin})"
            assert torch.equal(dx16, dx32), f"{label}: dx differs"
        assert torch.equal(dw16, dw32), f"{label}: dw differs"


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
@pytest.mark.parametrize("rows", (7, 16231))
def test_fp32_weight_meets_the_naive_tier(
    hidden: int, residual: bool, rows: int, record_property
) -> None:
    """The FP32-weight routes: y, rstd (h_new), dx and dw meet the naive BF16 tier against the FP64 oracle of the
    FP32 weight.  The weight holds a BF16 draw, so the naive tier reads the same weight as the oracle
    (``ref.naive_weight``)."""

    _skip_unless_fp32_routes(hidden, residual)
    eps = EPS[hidden]
    check = ref.Draws(f"hidden={hidden} residual={residual} rows={rows} fp32 w")
    for draw in range(check.draws):
        generator = _generator(hidden, rows, residual, 89, draw=draw)
        w32 = _weight(hidden, generator).float()
        outputs, dx, dw, inputs = _fp32_case(hidden, rows, residual, w32, generator)
        forward = {"y": outputs[0], "rstd": outputs[1]}
        if residual:
            forward["h_new"] = outputs[2]
        check.forward(draw, forward, inputs["x"], w32, eps, residual=inputs["u"])
        check.backward(
            draw,
            {"dx": dx, "dw": dw},
            inputs["g"],
            inputs["normalized"],
            w32,
            outputs[1],
            eps,
            g_h=inputs["g_h"],
        )
    check.check(record_property)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
@pytest.mark.parametrize("rows", (7, 16231))
def test_fp32_weight_is_applied_at_fp32(hidden: int, residual: bool, rows: int) -> None:
    """An effective weight ``1 + delta`` that BF16 cannot represent is applied as given: rstd is the BF16 path's
    given ``bf16(w32)``, and y and dx differ from it (exact checks; the accuracy of the FP32-weight routes is
    ``test_fp32_weight_meets_the_naive_tier``'s)."""

    _skip_unless_fp32_routes(hidden, residual)
    for draw in range(ref.DRAWS):
        label = f"hidden={hidden} residual={residual} rows={rows} fp32 w draw={draw}"
        generator = _generator(hidden, rows, residual, 89, draw=draw)
        w32 = 1.0 + 1e-3 * torch.randn(
            hidden, dtype=torch.float32, device=_device(), generator=generator
        )
        assert not torch.equal(w32.to(torch.bfloat16).float(), w32)
        seed = generator.get_state()
        outputs, dx, _dw, _ = _fp32_case(hidden, rows, residual, w32, generator)
        generator.set_state(seed)
        outputs16, dx16, _dw16, _ = _fp32_case(
            hidden, rows, residual, w32.to(torch.bfloat16), generator
        )
        assert torch.equal(outputs16[1], outputs[1]), (
            f"{label}: rstd does not depend on w"
        )
        assert not torch.equal(outputs16[0], outputs[0]), (
            f"{label}: y ignores the FP32 weight"
        )
        assert not torch.equal(dx16, dx), f"{label}: dx ignores the FP32 weight"


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_fp32_weight_layouts_outside_the_rule_are_materialized(hidden: int) -> None:
    """An FP32 weight that is not 32-byte aligned, or not contiguous, is copied once: the same results as the
    aligned contiguous weight (and the autograd path returns ``dw`` in FP32 for an FP32 weight)."""

    _skip_unless_fp32_routes(hidden, False)
    rows = 129
    generator = _generator(hidden, rows, 97)
    w32 = 1.0 + 1e-3 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    seed = generator.get_state()
    outputs, dx, dw, _ = _fp32_case(hidden, rows, False, w32, generator)
    storage = torch.empty(hidden + 8, dtype=torch.float32, device=_device())
    half_aligned = storage[4 : 4 + hidden]
    half_aligned.copy_(w32)
    assert half_aligned.data_ptr() % loader.FP32_WEIGHT_ALIGNMENT_BYTES == 16
    strided = torch.empty(2 * hidden, dtype=torch.float32, device=_device())[::2]
    strided.copy_(w32)
    for layout in (half_aligned, strided):
        generator.set_state(seed)
        outputs_l, dx_l, dw_l, _ = _fp32_case(hidden, rows, False, layout, generator)
        assert all(torch.equal(a, b) for a, b in zip(outputs, outputs_l, strict=True))
        assert torch.equal(dx, dx_l) and torch.equal(dw, dw_l)
    x = _rows(rows, hidden, generator=generator).requires_grad_(True)
    w_leaf = w32.clone().requires_grad_(True)
    y = cake_rmsnorm_det(x, w_leaf, EPS[hidden])
    y.backward(torch.ones_like(y))
    assert w_leaf.grad.dtype == torch.float32 and x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_zero_rows(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    generator = _generator(hidden, 0)
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device=_device())
    w = _weight(hidden, generator)
    y, rstd = cake_rmsnorm_det_train_forward(x, w, EPS[hidden])
    assert tuple(y.shape) == (0, hidden) and tuple(rstd.shape) == (0,)
    dx, dw = cake_rmsnorm_det_train_backward(x, x, w, rstd)
    assert tuple(dx.shape) == (0, hidden)
    assert torch.equal(dw, torch.zeros_like(dw))


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (129, 16231))
def test_deterministic_dw_is_bitwise_reproducible(hidden: int, rows: int) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = _generator(hidden, rows, 11)
    x = _rows(rows, hidden, generator=generator)
    w = _weight(hidden, generator)
    g = _rows(rows, hidden, generator=generator)
    _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    workspace = cake_rmsnorm_det_train_backward_workspace(rows, hidden, _device())
    results = []
    for _repeat in range(3):
        dx = torch.full(
            (rows, hidden), float("nan"), dtype=torch.bfloat16, device=_device()
        )
        dw = torch.full((hidden,), float("nan"), dtype=torch.float32, device=_device())
        # Poison the first partial row (after the counters); the kernels must
        # rewrite every chunk they read.
        begin = loader.COUNTERS_BYTES
        workspace[begin : begin + 4 * hidden].view(torch.float32).fill_(float("nan"))
        cake_rmsnorm_det_train_backward(
            g, x, w, rstd, workspace=workspace, dx=dx, dw=dw
        )
        torch.cuda.synchronize()
        results.append((dx.clone(), dw.clone()))
    fresh = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    for dx, dw in results[1:]:
        assert torch.equal(dx, results[0][0])
        assert torch.equal(dw, results[0][1])
    assert torch.equal(fresh[1], results[0][1]), "a fresh workspace changed dw"
    assert torch.isfinite(results[0][1]).all()


def _taper_rows(record: dict, limit: int) -> list[int]:
    """Row counts of the record's window where its taper cuts another chunk
    table than the module constants would on device 0 (the host's slot count
    of the record, ``slot_count``): the first, a middle and the last within
    ``limit`` (none for a program at the default taper)."""

    rule = record["grid_rule"]
    taper = loader.chunk_taper(rule)
    if taper == (loader.TAIL_DIV, loader.MAX_HALVING):
        return []
    select = record["select_rule"]
    above = 0 if select is None else int(select["above"])
    at_most = None if select is None else select["at_most"]
    last = limit if at_most is None else min(limit, int(at_most))
    slots = loader.slot_count(rule, device_index=0)
    rpi = int(rule["rows_per_issue"])
    differing = [
        rows
        for rows in range(above + 1, last + 1)
        if loader.n_chunks(rows, slots, rpi, *taper)
        != loader.n_chunks(rows, slots, rpi)
    ]
    if not differing:
        return []
    return sorted({differing[0], differing[len(differing) // 2], differing[-1]})


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
def test_backward_reuses_a_poisoned_workspace_across_row_counts(
    hidden: int, residual: bool, record_property
) -> None:
    """One caller-owned workspace serves every token count of a training step.
    Its partial rows are poisoned with NaN before each call, so a host that cuts
    another chunk table than the kernel -- a chunk count the main kernel wrote
    no partial rows for -- surfaces in ``dw`` instead of hiding behind a fresh
    zeroed buffer.  The row counts are the route's window edges and, for every
    program built with a taper of its own, counts where its table differs from
    the module constants'; they run largest first, then back up from the
    smallest, on each draw; every call's ``dx`` and ``dw`` meet the naive BF16
    tier over the draws of its row count."""

    _skip_unless_hidden(hidden)
    if residual and not loader.route_applies(
        device_capability=_capability(), hidden=hidden, kind="bwd", residual=True
    ):
        pytest.skip(f"hidden={hidden} has no exported residual backward route")
    arch = loader.arch_for_capability(_capability())
    limit = 32768
    rows_set = {
        rows for rows in _switch_rows(arch, hidden, "bwd", residual) if rows <= limit
    }
    for name in loader.route_modules(arch, hidden, "bwd", residual):
        record = loader.MODULES[name]
        if record.get("workspace_rule") is not None:
            rows_set.update(_taper_rows(record, limit))
    descending = sorted(rows_set, reverse=True)
    order = descending + descending[-2::-1]
    eps = EPS[hidden]
    workspace = cake_rmsnorm_det_train_backward_workspace(
        max(rows_set), hidden, _device(), residual=residual
    )
    checks = {
        rows: ref.Draws(
            f"hidden={hidden} rows={rows} residual={residual} (reused workspace)",
            prefix=f"T={rows} ",
        )
        for rows in rows_set
    }
    for draw in range(ref.DRAWS):
        judged: dict[int, tuple[dict, dict]] = {}
        for rows in order:
            generator = _generator(hidden, rows, 29, residual, draw=draw)
            x = _rows(rows, hidden, generator=generator)
            w = _weight(hidden, generator)
            g = _rows(rows, hidden, generator=generator)
            if residual:
                u = _rows(rows, hidden, generator=generator)
                g_h = _rows(rows, hidden, generator=generator)
                _y, rstd, normalized = cake_rmsnorm_det_train_forward(
                    x, w, eps, residual=u
                )
            else:
                g_h = None
                _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
                normalized = x
            workspace[loader.COUNTERS_BYTES :].view(torch.float32).fill_(float("nan"))
            dx, dw = cake_rmsnorm_det_train_backward(
                g, normalized, w, rstd, g_residual=g_h, workspace=workspace
            )
            # the second call of a row count on a draw has the first one's inputs: its oracle and naive values
            if rows not in judged:
                judged[rows] = (
                    ref.oracle_backward(g, normalized, w, rstd, g_h),
                    ref.naive_backward(g, normalized, w, rstd, eps, g_h),
                )
            checks[rows].add(draw, {"dx": dx, "dw": dw}, *judged[rows])
    ref.check_all(record_property, [checks[rows] for rows in sorted(rows_set)])


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_changing_rows_in_one_process_does_not_rebuild(
    hidden: int, record_property
) -> None:
    """Calls of changing token counts in one process load no further module, and each call meets the naive BF16
    tier over the draws of its token count."""
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    _check_case(hidden, 64, record=record_property, tag="T=64 ")
    loaded_before = loader.load.cache_info().misses
    sequence = (1, 65, 16172, 7, 16231, 255)
    checks = {
        rows: ref.Draws(f"hidden={hidden} rows={rows}", prefix=f"T={rows} ")
        for rows in sequence
    }
    for draw in range(ref.DRAWS):
        generator = _generator(hidden, 17, draw=draw)
        w = _weight(hidden, generator)
        for rows in sequence:
            x = _rows(rows, hidden, generator=generator)
            g = _rows(rows, hidden, generator=generator)
            y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
            dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
            checks[rows].forward(draw, {"y": y, "rstd": rstd}, x, w, eps)
            checks[rows].backward(draw, {"dx": dx, "dw": dw}, g, x, w, rstd, eps)
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed token count rebuilt a module"
    )
    ref.check_all(record_property, [checks[rows] for rows in sequence])


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("view", ("expand", "slice"))
def test_backward_accepts_gradient_views(
    hidden: int, view: str, record_property
) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    rows = 129
    check = ref.Draws(f"hidden={hidden} view={view}")
    for draw in range(check.draws):
        generator = _generator(hidden, rows, 23, draw=draw)
        x = _rows(rows, hidden, generator=generator)
        w = _weight(hidden, generator)
        if view == "expand":
            g = _rows(1, hidden, generator=generator).expand(rows, hidden)
        else:
            g = _rows(rows, hidden, stride_pad="2H", generator=generator)
        _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
        dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
        check.backward(draw, {"dx": dx, "dw": dw}, g, x, w, rstd, eps)
    check.check(record_property)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
@pytest.mark.parametrize("rows", (7, 16231))
def test_autograd_end_to_end(
    hidden: int, residual: bool, rows: int, record_property
) -> None:
    """The autograd wrapper's ``y``, input gradient and weight gradient (in the weight's dtype) against the FP64
    oracle of the whole training step (the residual sum rounded to BF16 as the kernel does, the FP64 ``rstd``),
    judged against the naive BF16 step (``ref.naive_forward`` and ``ref.naive_backward`` at the naive ``rstd``)
    over the draws; both residual-add inputs receive the same gradient."""
    _skip_unless_hidden(hidden)
    if residual and not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="fwd",
        residual=True,
    ):
        pytest.skip(f"hidden={hidden} has no exported residual route on this device")
    eps = EPS[hidden]
    check = ref.Draws(f"autograd hidden={hidden} residual={residual} rows={rows}")
    for draw in range(check.draws):
        generator = _generator(hidden, rows, 31, residual, draw=draw)
        x0 = _rows(rows, hidden, generator=generator)
        w0 = _weight(hidden, generator)
        u0 = _rows(rows, hidden, generator=generator) if residual else None
        grad_y = _rows(rows, hidden, generator=generator)
        grad_h = _rows(rows, hidden, generator=generator) if residual else None

        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        u = u0.detach().clone().requires_grad_(True) if residual else None
        outputs = cake_rmsnorm_det(x, w, eps, residual=u)
        if residual:
            y, h_new = outputs
            torch.autograd.backward([y, h_new], [grad_y, grad_h])
            assert torch.equal(x.grad, u.grad), (
                "both residual-add inputs receive the same gradient"
            )
        else:
            y = outputs
            y.backward(grad_y)
        assert x.grad.dtype == torch.bfloat16
        assert w.grad.dtype == w0.dtype

        oracle = ref.oracle_forward(x0, w0, eps, residual=u0)
        normalized = (x0.double() + u0.double()).to(torch.bfloat16) if residual else x0
        oracle.update(
            ref.oracle_backward(grad_y, normalized, w0, oracle["rstd"], grad_h)
        )
        naive = ref.naive_forward(x0, w0, eps, residual=u0)
        naive.update(
            ref.naive_backward(
                grad_y, naive.get("h_new", x0), w0, naive["rstd"], eps, grad_h
            )
        )
        check.add(draw, {"y": y.detach(), "dx": x.grad, "dw": w.grad}, oracle, naive)
    check.check(record_property)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_workspace_bytes_and_layout(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    small = cake_rmsnorm_det_train_backward_workspace_bytes(1, hidden, _device())
    middle = cake_rmsnorm_det_train_backward_workspace_bytes(129, hidden, _device())
    large = cake_rmsnorm_det_train_backward_workspace_bytes(65536, hidden, _device())
    assert small > 0 and small % 256 == 0 and large % 256 == 0
    assert small <= middle <= large
    workspace = cake_rmsnorm_det_train_backward_workspace(65536, hidden, _device())
    assert workspace.dtype == torch.uint8 and workspace.numel() == large
    assert torch.count_nonzero(workspace) == 0
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward_workspace_bytes(-1, hidden, _device())


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_argument_validation(hidden: int, record_property) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = _generator(hidden, 5)
    x = _rows(8, hidden, generator=generator)
    w = _weight(hidden, generator)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(x.float(), w, eps)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(x, w.double(), eps)
    if not _fp32_route(hidden, "fwd", False):
        with pytest.raises(ValueError, match="weight dtypes exported for this route"):
            cake_rmsnorm_det_train_forward(x, w.float(), eps)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(x[:, 1:], w[1:], eps)
    # An odd row stride is accepted (materialized once), not rejected; y and rstd meet the naive BF16 tier.
    check = ref.Draws(f"hidden={hidden} odd row stride")
    for draw in range(check.draws):
        odd = _rows(
            8, hidden, stride_pad=8, generator=_generator(hidden, 5, 8, draw=draw)
        )
        odd_view = odd.as_strided((8, hidden), (hidden + 4, 1))
        y_odd, rstd_odd = cake_rmsnorm_det_train_forward(odd_view, w, eps)
        check.forward(draw, {"y": y_odd, "rstd": rstd_odd}, odd_view, w, eps)
    check.check(record_property)
    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    # The det package takes no ``deterministic`` keyword.
    with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
        cake_rmsnorm_det_train_backward(y, x, w, rstd, deterministic=False)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward(y, x, w, rstd.double())
    tiny = torch.zeros(256, dtype=torch.uint8, device=_device())
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward(y, x, w, rstd, workspace=tiny)
    cpu = torch.zeros(8, hidden, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(cpu, w.cpu(), eps)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    # build.json names the package's import path; a vendored copy ships the file unchanged under its own name.
    assert manifest["package"] == "cake_pplx._backends.cake_rmsnorm_det_train"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()


# dw at the largest row counts of the long-chunk windows, where an FP32 chunk sum over many rows accumulates the most
# rounding: H = 1536 / 2048 / 4096 / 6144 at 65536 rows; H = 5120 at 40960 and 65536 rows; H = 256 at 81920 rows,
# at the key rows of a 65536-token batch (262144), at 1179649 rows and at the query rows of a 65536-token batch
# (1572864) and one above (1572865).  Three seeds per case, each judged over its draws against the naive BF16 weight
# gradient; BF16 and FP32 weight.
LARGE_ROW_DW_CASES = (
    (1536, 65536),
    (2048, 65536),
    (4096, 65536),
    (5120, 40960),
    (5120, 65536),
    (6144, 65536),
    (256, 81920),
    (256, 262144),
    (256, 1179649),
    (256, 1572864),
    (256, 1572865),
)


@pytest.mark.parametrize(("hidden", "rows"), LARGE_ROW_DW_CASES)
@pytest.mark.parametrize("fp32_weight", (False, True))
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_large_row_dw_meets_the_naive_tier(
    hidden: int, rows: int, fp32_weight: bool, seed: int, record_property
) -> None:
    _check_large_row_dw(hidden, rows, fp32_weight, seed, record_property)


def _check_large_row_dw(
    hidden: int, rows: int, fp32_weight: bool, seed: int, record_property
) -> None:
    if fp32_weight:
        _skip_unless_fp32_routes(hidden, False)
    else:
        _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    check = ref.Draws(
        f"hidden={hidden} rows={rows} fp32_weight={fp32_weight} seed={seed}"
    )
    for draw in range(check.draws):
        generator = _generator(hidden, rows, fp32_weight, seed, 97, draw=draw)
        x = _rows(rows, hidden, generator=generator)
        g = _rows(rows, hidden, generator=generator)
        # an FP32 weight holds a BF16 draw (the naive tier reads it exactly)
        w = _weight(hidden, generator)
        if fp32_weight:
            w = w.float()
        _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
        _dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
        del _y, _dx
        check.add(
            draw,
            {"dw": dw},
            {"dw": ref.oracle_dw(g, x, rstd)},
            {"dw": ref.naive_dw(g, x, eps)},
        )
    check.check(record_property)
