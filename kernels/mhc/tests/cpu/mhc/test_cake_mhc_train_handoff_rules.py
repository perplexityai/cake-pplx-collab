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

The pre-handoff records and routes of the two mHC training registries, on the host rules alone (no device).

The handoff routes (``HANDOFF_KINDS``) are the standard routes with one swap each, so a handoff call launches the
programs a standard call launches, with the same grids, PDL arguments and workspaces:

* ``fwd_prepare_handoff`` is ``fwd_prepare`` with the collapse ``fwd_collapse__pi_pn`` (it reads ``pre_in`` in place
  of the coefficient pass's ``pre`` and writes ``pre_next``) in place of ``fwd_collapse``;
* ``bwd_finish_handoff`` is ``bwd_finish`` with each token-owner form ``bwd_token__<form>`` replaced by its
  ``bwd_token__<form>_ho`` twin (the base record with ``pre_in``, ``dpre_next`` and ``dpre_in`` after the parameter
  record, before ``T``; its launch, dynamic shared memory included, is the base record's: ``dpre_in`` is one FP32
  word per stream, reduced in the kernel's own order);
* ``collapse_forward`` / ``collapse_backward`` are one record each (``collapse_fwd`` / ``collapse_bwd``) with their own
  library stems and no workspace; the forward takes the per-call PDL attribute on its own edge ``K4->C`` (no band: on),
  the VJP none; the VJP is the coalesced form: its dynamic shared memory stages each warp's slab of the ``dU`` row and
  of the four ``X`` rows (6400 bytes a warp, 8 warps) beside its 256 bytes of partial sums.

The routes exist at hidden 5120 on sm_100a, sm_103a and sm_107a (each token-owner twin on the architectures of its
base form: sm_107a's backward runs the ``dot_ao8_l2l`` form where the others run ``dot_ao8`` and its group-row form);
the nondeterministic records are the deterministic ones under the nondeterministic prefix.
"""

from __future__ import annotations

import copy
import math

import pytest
from tests.cake.test_helpers.cake_package import cake_package

DET = cake_package("cake.cake_mhc_det_train", "cake_jit")
ND = cake_package("cake.cake_mhc_nondet_train", "cake_jit")
REGISTRIES = {"det": DET, "nondet": ND}
PREFIX = {"det": "cake_mhc_det_train_", "nondet": "cake_mhc_train_"}
HIDDEN = 5120
HANDOFF_KINDS = (
    "fwd_prepare_handoff",
    "bwd_finish_handoff",
    "collapse_forward",
    "collapse_backward",
)
HANDOFF_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
BOARDS = (("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 216))
TOKEN_FORMS = (
    "ao22_rae7_pd1_ntr",
    "dot_ao8",
    "dot_ao8_trows_dyn",
    "rt64",
    "dot_ao8_l2l",
)
# The standalone VJP's dynamic shared memory: 8 warps x 6400 bytes of staged dU / X words + 256 bytes of partial sums.
VJP_SMEM = 8 * 6400 + 256
HANDOFF_OPERANDS = (
    ["buffer", "pre_in"],
    ["buffer", "dpre_next"],
    ["buffer", "dpre_in"],
)
ROWS = (
    1,
    2,
    3,
    7,
    64,
    65,
    255,
    256,
    1000,
    1024,
    2049,
    4096,
    4097,
    8192,
    12000,
    16384,
    20001,
    32768,
    65536,
    131072,
)
NAME_KEYS = ("kernel", "kernel_symbol", "library")


def _rec(mode: str, suffix: str) -> dict:
    return REGISTRIES[mode].MODULES[f"{PREFIX[mode]}{suffix}"]


def _strip(record: dict, *keys: str) -> dict:
    return {k: v for k, v in record.items() if k not in (*NAME_KEYS, *keys)}


def _swaps(mode: str) -> dict[str, str]:
    p = PREFIX[mode]
    out = {f"{p}fwd_collapse__h{HIDDEN}": f"{p}fwd_collapse__pi_pn__h{HIDDEN}"}
    out.update(
        {
            f"{p}bwd_token__{f}__h{HIDDEN}": f"{p}bwd_token__{f}_ho__h{HIDDEN}"
            for f in TOKEN_FORMS
        }
    )
    return out


def _selected(
    mode: str, arch: str, kind: str, rows: int, sm: int, iters: int
) -> tuple[str, ...]:
    return REGISTRIES[mode].selected_modules(
        arch, HIDDEN, kind, rows=rows, sm_count=sm, sinkhorn_iters=iters
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
def test_handoff_kinds_keys_and_sizes(mode: str) -> None:
    reg = REGISTRIES[mode]
    assert tuple(reg.HANDOFF_KINDS) == HANDOFF_KINDS
    assert not set(reg.HANDOFF_KINDS) & set(reg.KINDS)
    for kind in HANDOFF_KINDS:
        assert reg.route_key(HIDDEN, kind) == f"{kind}_h{HIDDEN}"
    assert reg.workspace_kind("fwd_prepare_handoff") == reg.workspace_kind(
        "fwd_prepare"
    )
    assert reg.workspace_kind("bwd_finish_handoff") == reg.workspace_kind("bwd_finish")
    assert reg.workspace_kind("collapse_forward") is None
    assert reg.workspace_kind("collapse_backward") is None
    for arch in HANDOFF_ARCHES:
        assert tuple(reg.handoff_hidden_sizes(arch)) == (HIDDEN,)
        assert HIDDEN in reg.supported_hidden_sizes(arch)
    for routes in reg.ROUTES.values():
        for key in routes:
            kind, hidden = key.rsplit("_h", 1)
            assert kind in (*reg.KINDS, *HANDOFF_KINDS), key
            if kind in HANDOFF_KINDS:
                assert int(hidden) == HIDDEN, key


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("form", TOKEN_FORMS)
def test_token_owner_twins_are_the_base_records_with_the_handoff_operands(
    mode: str, form: str
) -> None:
    base = _rec(mode, f"bwd_token__{form}__h{HIDDEN}")
    twin = _rec(mode, f"bwd_token__{form}_ho__h{HIDDEN}")
    name = f"{PREFIX[mode]}bwd_token__{form}_ho__h{HIDDEN}"
    assert (twin["kernel"], twin["library"], twin["kernel_symbol"]) == (
        name,
        name,
        f"kernel_{name}",
    )
    assert _strip(twin, "arches", "arg_plan", "logical", "launch") == _strip(
        base, "arches", "arg_plan", "logical", "launch"
    )
    assert twin["launch"] == base["launch"]
    assert twin["arches"] == [a for a in base["arches"] if a in HANDOFF_ARCHES]
    plan = copy.deepcopy(base["arg_plan"])
    at = plan.index(["buffer", "param_partial"]) + 1
    assert plan[at] == ["parameter", "T"]
    plan[at:at] = [list(op) for op in HANDOFF_OPERANDS]
    assert twin["arg_plan"] == plan
    assert twin["logical"] == {
        **base["logical"],
        "pre_in": "pre_in",
        "dpre_next": "dpre_next",
        "dpre_in": "dpre_in",
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
def test_collapse_records(mode: str) -> None:
    base = _rec(mode, f"fwd_collapse__h{HIDDEN}")
    pi_pn = _rec(mode, f"fwd_collapse__pi_pn__h{HIDDEN}")
    fwd = _rec(mode, f"collapse_fwd__h{HIDDEN}")
    bwd = _rec(mode, f"collapse_bwd__h{HIDDEN}")
    grid = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
    # the sublayer collapse: the base record (same grid, launch facts, PDL edge) with pre_in in and pre_next out
    assert _strip(pi_pn, "arches", "arg_plan", "logical") == _strip(
        base, "arches", "arg_plan", "logical"
    )
    assert pi_pn["arg_plan"] == [
        ["buffer", "x"],
        ["buffer", "pre_in"],
        ["buffer", "stats"],
        ["buffer", "u"],
        ["buffer", "pre_next"],
        ["parameter", "T"],
        ["launch", "enable_pdl"],
        *grid,
    ]
    assert (
        REGISTRIES[mode].pdl_edge(pi_pn["library"])
        == REGISTRIES[mode].pdl_edge(base["library"])
        is not None
    )
    # the standalone collapse: own library stems, no workspace; the forward takes the per-call PDL attribute on its own
    # edge K4->C (after the last sublayer's residual mixing, no band: on), the VJP none
    assert fwd["arg_plan"] == [
        ["buffer", "x"],
        ["buffer", "pre_in"],
        ["buffer", "u"],
        ["parameter", "T"],
        ["launch", "enable_pdl"],
        *grid,
    ]
    assert bwd["arg_plan"] == [
        ["buffer", "x"],
        ["buffer", "pre_in"],
        ["buffer", "du"],
        ["buffer", "dx"],
        ["buffer", "dpre_in"],
        ["parameter", "T"],
        *grid,
    ]
    assert (fwd["role"], bwd["role"]) == ("collapse_fwd", "collapse_bwd")
    assert fwd["launch"]["use_pdl"] == "runtime" and bwd["launch"]["use_pdl"] is False
    assert REGISTRIES[mode].pdl_edge(fwd["library"]) == "K4->C"
    assert REGISTRIES[mode].pdl_edge(bwd["library"]) is None
    for rec, smem in ((fwd, 0), (bwd, VJP_SMEM)):
        assert rec["workspace_rule"] is None and rec["select_rule"] is None
        assert rec["launch"]["dynamic_smem_bytes"] == smem
        assert tuple(rec["launch"]["block"]) == (256, 1, 1)
    assert fwd["grid_rule"] == base["grid_rule"]
    one_token_per_cta = {
        "col_split": 1,
        "grid_limit": 0,
        "grid_per_sm": 0,
        "kind": "rows_per_cta",
        "rows_per_cta": 1,
    }
    assert bwd["grid_rule"] == one_token_per_cta
    for rec in (pi_pn, fwd, bwd):
        assert rec["arches"] == list(HANDOFF_ARCHES)
        assert rec["hidden"] == HIDDEN
        assert set(rec["logical"]) == {
            k for kind, k in rec["arg_plan"] if kind in ("buffer", "parameter")
        }


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize("arch", HANDOFF_ARCHES)
def test_handoff_routes_are_the_standard_routes_with_the_swaps(
    mode: str, arch: str
) -> None:
    reg, p, swaps = REGISTRIES[mode], PREFIX[mode], _swaps(mode)
    routes = reg.ROUTES[arch]
    fwd = routes[f"fwd_prepare_h{HIDDEN}"]
    bwd = routes[f"bwd_finish_h{HIDDEN}"]
    assert routes[f"fwd_prepare_handoff_h{HIDDEN}"] == [swaps.get(n, n) for n in fwd]
    assert routes[f"bwd_finish_handoff_h{HIDDEN}"] == [swaps.get(n, n) for n in bwd]
    # every collapse / token-owner record of the standard routes has its twin (no base form left in a handoff route)
    assert [
        n for n in fwd if REGISTRIES[mode].MODULES[n]["role"] == "fwd_collapse"
    ] == [f"{p}fwd_collapse__h{HIDDEN}"]
    tokens = {n for n in bwd if REGISTRIES[mode].MODULES[n]["role"] == "bwd_token"}
    assert tokens and tokens == {
        f"{p}bwd_token__{f}__h{HIDDEN}"
        for f in TOKEN_FORMS
        if arch in REGISTRIES[mode].MODULES[f"{p}bwd_token__{f}__h{HIDDEN}"]["arches"]
    }
    assert routes[f"collapse_forward_h{HIDDEN}"] == [f"{p}collapse_fwd__h{HIDDEN}"]
    assert routes[f"collapse_backward_h{HIDDEN}"] == [f"{p}collapse_bwd__h{HIDDEN}"]
    for kind in HANDOFF_KINDS:
        for name in reg.route_modules(arch, HIDDEN, kind):
            assert arch in reg.MODULES[name]["arches"], (kind, name)


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
@pytest.mark.parametrize("iters", (20, 7))
def test_a_handoff_call_selects_the_standard_programs_with_the_swaps(
    mode: str, arch: str, sm: int, iters: int
) -> None:
    swaps = _swaps(mode)
    for rows in ROWS:
        for kind, standard in (
            ("fwd_prepare_handoff", "fwd_prepare"),
            ("bwd_finish_handoff", "bwd_finish"),
        ):
            want = tuple(
                swaps.get(n, n)
                for n in _selected(mode, arch, standard, rows, sm, iters)
            )
            assert _selected(mode, arch, kind, rows, sm, iters) == want, (kind, rows)
        for kind, name in (
            ("collapse_forward", "collapse_fwd"),
            ("collapse_backward", "collapse_bwd"),
        ):
            assert _selected(mode, arch, kind, rows, sm, iters) == (
                f"{PREFIX[mode]}{name}__h{HIDDEN}",
            )


@pytest.mark.cpu_only
@pytest.mark.parametrize("mode", tuple(REGISTRIES))
@pytest.mark.parametrize(("arch", "sm"), BOARDS)
def test_handoff_launch_grids_and_pdl_arguments(
    monkeypatch: pytest.MonkeyPatch, mode: str, arch: str, sm: int
) -> None:
    reg, swaps = REGISTRIES[mode], _swaps(mode)
    monkeypatch.setattr(reg, "sm_count", lambda device_index: sm)
    monkeypatch.setattr(reg, "device_arch", lambda device_index: arch)
    fwd = reg.MODULES[f"{PREFIX[mode]}collapse_fwd__h{HIDDEN}"]
    bwd = reg.MODULES[f"{PREFIX[mode]}collapse_bwd__h{HIDDEN}"]
    for rows in ROWS:
        for base, twin in swaps.items():
            assert reg.launch_grid(
                reg.MODULES[twin], rows, device_index=0
            ) == reg.launch_grid(reg.MODULES[base], rows, device_index=0), (twin, rows)
            for path in reg.PDL_PATHS:
                args = (rows, arch, path) if mode == "det" else (rows, arch, path, sm)
                assert reg.enable_pdl(twin, *args) == reg.enable_pdl(base, *args), (
                    twin,
                    rows,
                    path,
                )
        assert reg.launch_grid(fwd, rows, device_index=0) == (
            math.ceil(5 * rows / 2),
            1,
            1,
        )
        assert reg.launch_grid(bwd, rows, device_index=0) == (rows, 1, 1)
        for path in reg.PDL_PATHS:
            for rec in (fwd, bwd):
                args = (rows, arch, path) if mode == "det" else (rows, arch, path, sm)
                assert reg.enable_pdl(rec["kernel"], *args) == 1


@pytest.mark.cpu_only
def test_the_nondet_handoff_records_are_the_det_records_renamed() -> None:
    def renamed(value):
        if isinstance(value, str):
            return value.replace(PREFIX["det"], PREFIX["nondet"])
        if isinstance(value, dict):
            return {renamed(k): renamed(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(renamed(v) for v in value)
        return value

    det_names = {
        name
        for routes in DET.ROUTES.values()
        for key, names in routes.items()
        if key.rsplit("_h", 1)[0] in HANDOFF_KINDS
        for name in names
    }
    new = {
        n
        for n in det_names
        if any(
            s in n for s in ("__pi_pn__", "collapse_fwd__", "collapse_bwd__", "_ho__")
        )
    }
    # the fused collapse, the standalone collapse pair and one twin per token-owner form
    assert len(new) == 3 + len(TOKEN_FORMS), sorted(new)
    for name in new:
        assert ND.MODULES[renamed(name)] == renamed(DET.MODULES[name]), name
    for arch in HANDOFF_ARCHES:
        for kind in ("collapse_forward", "collapse_backward"):
            key = f"{kind}_h{HIDDEN}"
            assert ND.ROUTES[arch][key] == renamed(DET.ROUTES[arch][key])
