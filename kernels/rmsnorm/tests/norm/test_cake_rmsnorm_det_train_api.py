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

Tests of the public surface of the Cake fused BF16 RMSNorm
training kernels (SM100 / SM103 / SM107): the ``cake_pplx.cake_rmsnorm_det_train``
facade.  The facade's three entry points are pass-throughs to
``cake_pplx._backends.cake_rmsnorm_det_train``; this module checks that identity
(same objects on the facade, same bits as the backend) and leaves the
numerical contract to ``test_cake_rmsnorm_det_train.py``, which imports only the
backend package.  It also checks the bytes an H = 128 backward allocates when it
is given no workspace: only its own call's.
"""

from __future__ import annotations

import re

import pytest
import torch

import cake_pplx.cake_rmsnorm_det_train
import cake_pplx._backends.cake_rmsnorm_det_train as backend
from cake_pplx.cake_rmsnorm_det_train import (
    CakeRMSNormDetFunction,
    cake_rmsnorm_det,
    cake_rmsnorm_det_train_backward,
    cake_rmsnorm_det_train_backward_workspace_bytes,
    cake_rmsnorm_det_train_forward,
    cake_rmsnorm_det_train_program_windows,
    cake_rmsnorm_det_train_route_key,
    cake_rmsnorm_det_train_weight_dtypes,
    is_cake_rmsnorm_det_train_supported,
)
from cake_pplx._backends.cake_rmsnorm_det_train import cake_jit as loader
from tests.cake.test_helpers.cake_jit_warmup import build_jit_specs

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (1280, 1e-20),
    (1024, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
    (128, 1e-20),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
SEED = 760


def _device() -> torch.device:
    return torch.device("cuda:0")


def get_compute_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(get_compute_capability(_device()))
        build_jit_specs(loader.specs_for_arch(arch))
    yield


def _skip_unless_hidden(hidden: int) -> None:
    if not is_cake_rmsnorm_det_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


def _randn(*shape: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        *shape, dtype=torch.float32, device=_device(), generator=generator
    ).to(torch.bfloat16)


def test_function_signature_is_exported() -> None:
    assert cake_pplx.cake_rmsnorm_det_train.cake_rmsnorm_det is cake_rmsnorm_det
    assert cake_pplx.cake_rmsnorm_det_train.cake_rmsnorm_det_train_forward is cake_rmsnorm_det_train_forward
    assert cake_pplx.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward is cake_rmsnorm_det_train_backward
    assert (
        cake_pplx.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward_workspace_bytes
        is cake_rmsnorm_det_train_backward_workspace_bytes
    )
    assert cake_pplx.cake_rmsnorm_det_train.CakeRMSNormDetFunction is CakeRMSNormDetFunction
    assert (
        cake_rmsnorm_det_train_program_windows
        is backend.cake_rmsnorm_det_train_program_windows
        is loader.program_windows
    )
    assert (
        cake_rmsnorm_det_train_route_key
        is backend.cake_rmsnorm_det_train_route_key
        is loader.route_key
    )
    assert (
        cake_rmsnorm_det_train_weight_dtypes
        is backend.cake_rmsnorm_det_train_weight_dtypes
        is loader.weight_dtypes
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_facade_entry_points_match_the_backend(hidden: int) -> None:
    """The facade forwards every argument: its forward, backward and autograd
    entry points return the same bits as the backend package's."""
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = torch.Generator(device=_device()).manual_seed(SEED + hidden)
    x = _randn(129, hidden, generator=generator)
    g = _randn(129, hidden, generator=generator)
    w = _randn(hidden, generator=generator)
    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    y_backend, rstd_backend = backend.cake_rmsnorm_det_train_forward(x, w, eps)
    assert torch.equal(y, y_backend) and torch.equal(rstd, rstd_backend)
    dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    dx_backend, dw_backend = backend.cake_rmsnorm_det_train_backward(g, x, w, rstd)
    assert torch.equal(dx, dx_backend) and torch.equal(dw, dw_backend)
    assert torch.equal(cake_rmsnorm_det(x, w, eps), backend.cake_rmsnorm_det(x, w, eps))


# "Beside kernels resident on other streams" in the facade docstring lists, per
# architecture, the calls whose kernels need a whole SM.  A list item names an
# architecture set ("SM100 / SM103:") and the calls ("none": no call): "the
# residual-fused H <h> backward" (every token count; "... above <t> tokens":
# every program window above ``t``) or "the forward / backward of H <h>" with
# its token ranges, joined by "and": "above <t> tokens" (every program window
# above ``t``) or "above <t> up to <u> tokens" (every program window above
# ``t`` up to ``u``), the last possibly followed by "(except with BF16 / FP32
# weights from <a> to <b> tokens)": the windows of that weight dtype between
# ``a`` and ``b`` start (the stall test's ``WHOLE_SM_STARTS``).
_WHOLE_SM_SPAN = r"above \d+(?: up to \d+)? tokens"
_WHOLE_SM_ITEM = re.compile(
    r"the residual-fused H (\d+) backward(?: above (\d+) tokens)?"
    r"|the (forward|backward) of H (\d+) "
    rf"({_WHOLE_SM_SPAN}(?: and {_WHOLE_SM_SPAN})*)"
    r"(?: \(except with (BF16|FP32) weights from (\d+) to (\d+) tokens\))?"
)
_WHOLE_SM_RANGE = re.compile(r"above (\d+)(?: up to (\d+))? tokens")


def _documented_whole_sm(
    starts: dict[str, set[tuple[str, int, bool, str, int, int]]] | None = None,
) -> dict[str, set[tuple[str, int, bool, int, int | None]]]:
    """``arch -> {(kind, hidden, residual, above, at_most)}`` from the facade
    docstring (``at_most`` None: no upper bound); its "except" clauses go to
    ``starts`` (``arch -> {(kind, hidden, residual, w dtype, above, at_most)}``)."""
    import cake_pplx.cake_rmsnorm_det_train as facade

    out: dict[str, set[tuple[str, int, bool, int, int | None]]] = {}
    for item in re.split(r"\n- ", facade.__doc__ or "")[1:]:
        text = " ".join(item.split("\n\n")[0].split())
        head = re.match(r"((?:SM1\d\d)(?: / SM1\d\d)*):", text)
        if head is None:
            continue
        for sm in re.findall(r"SM1\d\d", head.group(1)):
            entries = out.setdefault(f"sm_{sm[2:]}a", set())
            for (
                residual_hidden,
                residual_above,
                kind,
                hidden,
                spans,
                except_dtype,
                except_above,
                except_at_most,
            ) in _WHOLE_SM_ITEM.findall(text):
                if residual_hidden:
                    entries.add(
                        (
                            "bwd",
                            int(residual_hidden),
                            True,
                            int(residual_above or 0),
                            None,
                        )
                    )
                    continue
                route = ("fwd" if kind == "forward" else "bwd", int(hidden), False)
                for above, at_most in _WHOLE_SM_RANGE.findall(spans):
                    entries.add((*route, int(above), int(at_most) if at_most else None))
                if except_dtype and starts is not None:
                    starts.setdefault(f"sm_{sm[2:]}a", set()).add(
                        (
                            *route,
                            {"BF16": "bf16", "FP32": "f32"}[except_dtype],
                            int(except_above),
                            int(except_at_most),
                        )
                    )
    return out


def test_documented_whole_sm_calls_match_the_program_table() -> None:
    """The facade's whole-SM lists name every architecture of the package, each
    listed call names a shipped route of that architecture whose token
    thresholds are edges of the route's program windows (the lower one a
    window's lower bound, the upper one, where given, a window's upper bound;
    with BF16 and FP32 weights alike), and the lists are the stall test's
    ``WHOLE_SM`` table, which checks them on the device; an "except" clause
    names, for its weight dtype, two edges of the route's program windows, and
    the clauses are the stall test's ``WHOLE_SM_STARTS``."""
    from tests.norm.test_cake_rmsnorm_det_train_stall import WHOLE_SM, WHOLE_SM_STARTS

    starts: dict[str, set[tuple[str, int, bool, str, int, int]]] = {}
    documented = _documented_whole_sm(starts)
    assert starts == {arch: set(entries) for arch, entries in WHOLE_SM_STARTS.items()}
    for arch, entries in starts.items():
        for kind, hidden, residual, w_dtype, low, high in entries:
            edges = {
                edge
                for above, at_most, _library in cake_rmsnorm_det_train_program_windows(
                    arch, hidden, kind, residual, w_dtype
                )
                for edge in (above, at_most)
            }
            assert {low, high} <= edges, (arch, kind, hidden, w_dtype, low, high)
    assert set(documented) == set(loader.ARCHES)
    for arch, entries in documented.items():
        assert entries == set(WHOLE_SM[arch]), arch
        for kind, hidden, residual, above, at_most in entries:
            dtypes = loader.weight_dtypes(arch, hidden, kind, residual)
            assert dtypes, (arch, kind, hidden, residual)
            for w_dtype in dtypes:
                windows = cake_rmsnorm_det_train_program_windows(
                    arch, hidden, kind, residual, w_dtype
                )
                edges = {low for low, _high, _library in windows}
                assert above in edges, (
                    arch,
                    kind,
                    hidden,
                    residual,
                    w_dtype,
                    above,
                    edges,
                )
                if at_most is not None:
                    tops = {high for _low, high, _library in windows}
                    assert at_most in tops, (
                        arch,
                        kind,
                        hidden,
                        residual,
                        w_dtype,
                        at_most,
                        tops,
                    )


def _bwd_windows(
    arch: str, kind: str, hidden: int
) -> list[list[tuple[bool, str, int]]]:
    """For each backward route of ``arch``'s registry that has one, in
    registry order, ``(residual, w dtype token, tokens)`` inside each of its
    windows of width ``hidden`` whose workspace-owning module has the workspace
    rule ``kind`` (``claim_sum_f32``: a column-task kernel; ``chunk_partials``:
    the main kernel of a pair), in the route's order."""
    routes = []
    for route, names in loader.ROUTES.get(arch, {}).items():
        if not route.startswith("bwd"):
            continue
        found = []
        for name in names:
            record = loader.MODULES[name]
            rule = record.get("workspace_rule")
            if rule is None or rule["kind"] != kind or int(record["hidden"]) != hidden:
                continue
            window = record.get("select_rule")
            above = 0 if window is None else int(window["above"])
            at_most = None if window is None else window["at_most"]
            tokens = above + 1 if at_most is None else (above + 1 + int(at_most)) // 2
            residual = "_residual" in route
            token = "f32" if "_w32" in route else "bf16"
            found.append((residual, token, tokens))
        if found:
            routes.append(found)
    return routes


def _bwd_window(arch: str, kind: str, hidden: int) -> tuple[bool, str, int] | None:
    """``(residual, w dtype token, tokens)`` inside a backward window of width
    ``hidden`` of ``arch``'s registry whose workspace-owning module has the
    workspace rule ``kind``, the route's last such window (the column tasks
    above a pair's window, where a route has them) in the first route that has
    one (:func:`_bwd_windows`), or ``None``."""
    routes = _bwd_windows(arch, kind, hidden)
    return routes[0][-1] if routes else None


def _no_workspace_bytes(
    arch: str, kind: str, hidden: int, window: tuple[bool, str, int]
) -> tuple[int, int]:
    """``(own, public)`` of a backward call of width ``hidden`` in ``window`` on
    the device, whose workspace-owning module has the workspace rule ``kind``:
    ``own`` is the ``total_bytes`` of the launched module's layout, with the
    partial capacity of the route's owner that the call binds, ``public`` the
    :func:`cake_rmsnorm_det_train_backward_workspace_bytes` of its token count
    (the size that serves every window up to it)."""
    residual, token, tokens = window
    label = (kind, hidden, residual, token, tokens)
    device_index = backend._device_index(_device())
    owner = backend._workspace_record(arch, hidden, residual, tokens, token)
    selection = backend._backward_selection_rows(
        arch, hidden, residual, tokens, token, device_index
    )
    launched = [
        loader.MODULES[name]
        for name in backend.selected_modules(
            arch, hidden, "bwd", residual, rows=selection, w_dtype=token
        )
        if loader.MODULES[name].get("workspace_rule") is not None
    ]
    assert len(launched) == 1, (label, launched)
    assert launched[0]["workspace_rule"]["kind"] == kind, label
    layout = loader.workspace_layout(
        launched[0],
        tokens,
        device_index=device_index,
        capacity_record=launched[0] if owner is None else owner,
    )
    public = cake_rmsnorm_det_train_backward_workspace_bytes(
        tokens, hidden, _device(), residual=residual
    )
    return int(layout["total_bytes"]), public


def _no_workspace_windows(
    arch: str,
) -> list[tuple[str, int, tuple[bool, str, int], bool]]:
    """``(kind, hidden, window, own < public)`` of the calls of
    ``test_the_backward_given_no_workspace_allocates_only_its_own_bytes``,
    chosen at run time from ``arch``'s registry (``own`` and ``public`` of
    :func:`_no_workspace_bytes`): for the H = 128 column-task kernel and the
    H = 128 pair each, the first window (:func:`_bwd_windows`) whose own bytes
    are strictly fewer than the public size, else :func:`_bwd_window`'s; then
    the first window of another width (pairs before column tasks) whose own
    bytes are fewer than the public size, else :func:`_bwd_window`'s of the
    first other width that has one.  So an H = 128 call of fewer own bytes than
    the public size is among them whenever the registry has one of either
    kind, and so is an other-width call whose two sizes differ whenever one
    exists."""

    def first_smaller(kind: str, hidden: int) -> tuple[bool, str, int] | None:
        for route in _bwd_windows(arch, kind, hidden):
            for window in route:
                own, public = _no_workspace_bytes(arch, kind, hidden, window)
                if own < public:
                    return window
        return None

    found = []
    for kind in ("claim_sum_f32", "chunk_partials"):
        window = first_smaller(kind, 128)
        if window is not None:
            found.append((kind, 128, window, True))
        elif (window := _bwd_window(arch, kind, 128)) is not None:
            found.append((kind, 128, window, False))
    others = [
        (kind, hidden)
        for kind in ("chunk_partials", "claim_sum_f32")
        for hidden in loader.supported_hidden_sizes(arch)
        if hidden != 128
    ]
    other = next(
        (
            (kind, hidden, window, True)
            for kind, hidden in others
            if (window := first_smaller(kind, hidden)) is not None
        ),
        None,
    )
    if other is None:
        other = next(
            (
                (kind, hidden, window, False)
                for kind, hidden in others
                if (window := _bwd_window(arch, kind, hidden)) is not None
            ),
            None,
        )
    if other is not None:
        found.append(other)
    return found


def test_the_backward_given_no_workspace_allocates_only_its_own_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given no workspace, an H = 128 backward (called directly, and through
    autograd, which never passes one) allocates a workspace of exactly the
    ``total_bytes`` of the launched module's layout, with the partial capacity
    of the route's owner that the call binds, never more than
    :func:`cake_rmsnorm_det_train_backward_workspace_bytes` (the size that
    serves every window up to the token count), in a column-task window and a
    pair window; a backward of another width allocates that function's size,
    as before.  The windows are those of :func:`_no_workspace_windows`: where
    the registry has an H = 128 window whose own bytes are fewer than the
    public size (a column-task window above the pair windows), one of them is
    called and the strict inequality asserted, so a backward that allocated
    the public size there fails; where another width has such a window, the
    other-width call is in one, so a backward that allocated that call's own
    bytes there fails.  ``dx`` and ``dw`` are bitwise those of the same call on
    a zeroed buffer of the public size."""
    _skip_unless_hidden(128)
    arch = loader.arch_for_capability(get_compute_capability(_device()))
    cases = _no_workspace_windows(arch)
    assert cases and cases[0][1] == 128, f"no H 128 backward window on {arch}"
    sizes: list[tuple[int, int]] = []
    views = backend._workspace_views

    def recording(workspace: torch.Tensor, record: dict, **kwargs):
        sizes.append((workspace.numel(), workspace.untyped_storage().nbytes()))
        return views(workspace, record, **kwargs)

    monkeypatch.setattr(backend, "_workspace_views", recording)
    generator = torch.Generator(device=_device()).manual_seed(SEED + 61)
    for kind, hidden, window, smaller in cases:
        residual, token, tokens = window
        label = (kind, hidden, residual, token, tokens)
        eps = EPS[hidden]
        w_dtype = torch.float32 if token == "f32" else torch.bfloat16
        x = _randn(tokens, hidden, generator=generator)
        g = _randn(tokens, hidden, generator=generator)
        w = _randn(hidden, generator=generator).to(w_dtype)
        gh = _randn(tokens, hidden, generator=generator) if residual else None
        own, public = _no_workspace_bytes(arch, kind, hidden, window)
        assert own <= public, (label, own, public)
        if smaller:
            assert own < public, (label, own, public)
        if kind == "claim_sum_f32":
            assert own == loader.COUNTERS_BYTES, (label, own)
        want = own if hidden == 128 else public
        _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
        sizes.clear()
        dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd, g_residual=gh)
        assert sizes == [(want, want)], (label, sizes, own, public)
        fresh = torch.zeros(public, dtype=torch.uint8, device=_device())
        dx0, dw0 = cake_rmsnorm_det_train_backward(
            g, x, w, rstd, g_residual=gh, workspace=fresh
        )
        assert torch.equal(dx, dx0) and torch.equal(dw, dw0), label
        xa = x.clone().requires_grad_(True)
        wa = w.clone().requires_grad_(True)
        if residual:
            ra = _randn(tokens, hidden, generator=generator).requires_grad_(True)
            outputs = cake_rmsnorm_det(xa, wa, eps, residual=ra)
            grads = (g, gh)
        else:
            outputs = (cake_rmsnorm_det(xa, wa, eps),)
            grads = (g,)
        sizes.clear()
        torch.autograd.backward(outputs, grads)
        torch.cuda.synchronize()
        assert sizes == [(want, want)], ("autograd", label, sizes, want)
