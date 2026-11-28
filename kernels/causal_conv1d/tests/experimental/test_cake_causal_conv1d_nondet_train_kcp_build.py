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

The KCP registry (``kcp_jit``) of the non-deterministic causal convolution training package against the
deterministic package's: the forward, history-row and head programs are the deterministic ones (only their names
differ; the history-row program, route ``kcp_fwd_halo`` of the two-launch forward, is a route of its own outside
``KINDS``, and the forward's single-rank main program is the zero-history registry's ``fwd`` route); the main backward
programs keep the deterministic route positions, selection rule, build geometry and arguments, with
the ``partial`` rows replaced by zeroed FP32 accumulators (kernel parameter ``dw_acc``, logical ``acc``) and uint32
``work`` ticket counters, an ordinary occupancy-sized
launch (no cooperative attribute, no grid barrier) and the workspace size of the legacy channel-tile rule the KCP
path plans with.  No device is needed; nothing is built or launched (a host launch is recorded, not run).
"""

from __future__ import annotations

import re

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
cake_jit = cake_package(KEY, "cake_jit")
kcp_jit = cake_package(KEY, "kcp_jit")
det_kcp_jit = cake_package("cake_causal_conv1d_det_train", "kcp_jit")

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
NAME_FIELDS = ("kernel", "kernel_symbol", "library")
PROGRAM_NAME = re.compile(
    r"cake_causal_conv1d_train_(?:TODOEXPORT_[a-z0-9_]+|[0-9a-f]{20})_kcp"
)
CHANNELS = (
    8,
    128,
    256,
    504,
    512,
    768,
    1016,
    1024,
    1160,
    1280,
    1344,
    1536,
    2048,
    4096,
    8192,
)
ACCUMULATORS = {"acc": "float32", "kind": "accumulators", "taps": 4, "work": "uint32"}


def _accumulator_rule(rule: dict) -> dict:
    """``rule`` without its ``slots``, which must name one copy (the KCP programs keep one FP32 accumulator)."""

    rule = dict(rule)
    assert rule.pop("slots", None) == 1, "a KCP accumulators rule names slots 1"
    return rule


def _strip(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _args(record: dict) -> set[str]:
    return {name for kind, name in record["arg_plan"] if kind != "grid"}


def _pairs():
    for arch in ARCHES:
        for kind in kcp_jit.KINDS:
            new, det = kcp_jit.ROUTES[arch][kind], det_kcp_jit.ROUTES[arch][kind]
            assert len(new) == len(det), (arch, kind)
            for name, det_name in zip(new, det, strict=True):
                yield arch, kind, name, det_name


@pytest.mark.cpu_only
def test_routes_mirror_the_deterministic_kcp_registry() -> None:
    assert kcp_jit.KINDS == det_kcp_jit.KINDS == ("kcp_fwd", "kcp_bwd", "kcp_bwd_head")
    assert kcp_jit.ARCHES == det_kcp_jit.ARCHES == ARCHES
    routed = {
        name
        for by_kind in kcp_jit.ROUTES.values()
        for names in by_kind.values()
        for name in names
    }
    assert set(kcp_jit.MODULES) == routed
    assert not set(kcp_jit.MODULES) & set(det_kcp_jit.MODULES)
    assert not set(kcp_jit.MODULES) & set(cake_jit.MODULES), (
        "the KCP programs are not the zero-history ones"
    )
    for name, record in kcp_jit.MODULES.items():
        assert PROGRAM_NAME.fullmatch(name), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["grid_rule"]["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name
    assert (
        *kcp_jit.KINDS,
        "kcp_bwd_dyn",
        "kcp_bwd_dyn_pair",
        "kcp_bwd_halo",
        "kcp_fwd_halo",
    ) == kcp_jit.ROUTE_KINDS
    assert kcp_jit.FWD_HALO_KIND == "kcp_fwd_halo"
    assert kcp_jit.FWD_HALO_KIND not in kcp_jit.KINDS
    for arch in ARCHES:
        assert set(kcp_jit.ROUTES[arch]) == set(kcp_jit.ROUTE_KINDS), arch


@pytest.mark.cpu_only
def test_forward_and_head_records_are_the_deterministic_ones() -> None:
    for arch, kind, name, det_name in _pairs():
        if kind == "kcp_bwd":
            continue
        assert _strip(kcp_jit.MODULES[name], *NAME_FIELDS) == _strip(
            det_kcp_jit.MODULES[det_name], *NAME_FIELDS
        ), (arch, kind, name)


@pytest.mark.cpu_only
def test_history_row_record_is_the_deterministic_one() -> None:
    """The two-launch forward's history-row program (route ``kcp_fwd_halo``, one record on every architecture): the
    det package's record of that route under this package's prefix (the same twenty hex digits), with the head
    program's grid rule."""

    for arch in ARCHES:
        (name,) = kcp_jit.route_modules(arch, kcp_jit.FWD_HALO_KIND)
        (det_name,) = det_kcp_jit.route_modules(arch, "kcp_fwd_halo")
        assert name == det_name.replace(
            "cake_causal_conv1d_det_train_", "cake_causal_conv1d_train_"
        ), (arch, name, det_name)
        assert _strip(kcp_jit.MODULES[name], *NAME_FIELDS) == _strip(
            det_kcp_jit.MODULES[det_name], *NAME_FIELDS
        ), (arch, name)
        assert kcp_jit.selected_modules(arch, kcp_jit.FWD_HALO_KIND, C=8192) == (name,)
        (head,) = kcp_jit.route_modules(arch, "kcp_bwd_head")
        for n_frag in (0, 1, 3, 17):
            for C in CHANNELS:
                assert kcp_jit.head_grid(
                    kcp_jit.MODULES[name], n_frag=n_frag, C=C
                ) == kcp_jit.head_grid(kcp_jit.MODULES[head], n_frag=n_frag, C=C)


@pytest.mark.cpu_only
def test_two_launch_forward_programs() -> None:
    """The two-launch forward's programs (``kcp`` module docstring): the history-row program is the ``kcp_fwd_halo``
    route's one record and takes the arguments the host passes (``kcp._fwd_repair``); the single-rank main program is
    a record of this package's zero-history registry's ``fwd`` route (``kcp.FWD_MAIN_ROUTE``) on every architecture,
    the det package's zero-history record of the same route position, each record of that route takes only ``x``,
    ``w``, ``cu``, ``y``, its ``work`` counters (``tile_counters``; none, and no ``workspace_rule``, for the static,
    the direct and the pair direct program of the small-S rows), the plan scalars and, exactly when its
    ``launch.use_pdl`` is ``"runtime"``, the ``enable_pdl`` launch scalar (``kcp._fwd_main`` passes 0), and launches
    the co-resident grid (no reserved SM; the direct and the pair direct program's capped at their token blocks), so
    the host can launch whichever record ``cake_jit.selected_modules`` picks for the call's ``C`` and token count: at
    C = 640 the pair direct program (two channels per thread) at 1003 tokens, the static program at 3103 and the
    direct program at 12003 on every architecture."""

    kcp = cake_package(KEY, "kcp")
    det_cake_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")
    assert kcp.KCP_FWD_HALO == "kcp_fwd_halo" and kcp.FWD_MAIN_ROUTE == "fwd"
    passed = {"x", "w", "cu", "y", "hist", "hist_row_off", "C", "num_docs"}
    for arch in ARCHES:
        (repair,) = kcp_jit.route_modules(arch, kcp.KCP_FWD_HALO)
        assert set(kcp_jit.MODULES[repair]["logical"].values()) == passed
    main_values = {"x", "w", "cu", "y", "work", *cake_jit.PLAN_SCALARS}
    for arch in ARCHES:
        names = cake_jit.route_modules(arch, kcp.FWD_MAIN_ROUTE)
        assert names, arch
        for name in names:
            record = cake_jit.MODULES[name]
            assert record["role"] == "fwd" and arch in record["arches"], (arch, name)
            runtime_pdl = record["launch"]["use_pdl"] == "runtime"
            logical = set(record["logical"].values())
            assert logical <= main_values | ({"enable_pdl"} if runtime_pdl else set()), (arch, name)
            assert ("enable_pdl" in logical) == runtime_pdl, (arch, name)
            rule = record["workspace_rule"]
            assert rule is None or rule["kind"] == "tile_counters", (arch, name)
            assert (rule is None) == ("work" not in logical), (arch, name)
            assert record["grid_rule"]["kind"] == "co_resident", (arch, name)
            assert "reserve_sms" not in record["grid_rule"], (arch, name)
            assert not record["grid_rule"]["cooperative"], (arch, name)
        # (C, S) -> the selection-rule kind and the geometry ``vec`` of the program the call selects
        small_s = {
            (640, 1003): (cake_jit.ROWS_KIND, 2),
            (640, 3103): (None, 8),
            (640, 12003): (cake_jit.ROWS_KIND, 8),
        }
        for C, S in (
            (256, 1),
            (256, 65536),
            (2048, 1),
            (8192, 32768),
            (10240, 32768),
            *small_s,
        ):
            (name,) = cake_jit.selected_modules(arch, kcp.FWD_MAIN_ROUTE, C=C, S=S)
            assert name in names, (arch, C, S)
            if (C, S) in small_s:
                rule = cake_jit.select_rule(cake_jit.MODULES[name])
                vec = cake_jit.MODULES[name]["grid_rule"]["geometry"]["vec"]
                assert ((rule or {"kind": None})["kind"], vec) == small_s[C, S], (
                    arch,
                    C,
                    S,
                    name,
                )
                assert cake_jit.MODULES[name]["workspace_rule"] is None, (
                    arch,
                    C,
                    S,
                    name,
                )
            # the det package's single-rank main program for the same call: the same program (twenty hex digits)
            (det_name,) = det_cake_jit.selected_modules(
                arch, kcp.FWD_MAIN_ROUTE, C=C, S=S
            )
            assert name.rsplit("_", 1)[-1] == det_name.rsplit("_", 1)[-1], (
                arch,
                C,
                S,
                name,
                det_name,
            )


@pytest.mark.cpu_only
def test_single_rank_forward_main_launches_without_pdl(monkeypatch) -> None:
    """The single-rank main program's launch (``kcp._fwd_main`` of an entry with ``sr``) passes every logical argument
    of each record of the ``fwd`` route on every architecture, and the ``enable_pdl`` launch scalar 0 (no
    programmatic dependent launch, in every mode) with and without token-chunk counters."""

    kcp = cake_package(KEY, "kcp")
    seen: list[dict] = []
    monkeypatch.setattr(
        kcp, "_launch_trim", lambda info, values, device_index: seen.append(dict(values))
    )
    S, C = 16, 64
    entry = {
        "sr": True,
        "cu": torch.tensor([0, S], dtype=torch.int32),
        "scalars": dict.fromkeys(cake_jit.PLAN_SCALARS, 1),
        "launch": None,
        "device_index": 0,
    }
    x = torch.zeros(S * C, dtype=torch.bfloat16)
    w = torch.zeros(C, 4, dtype=torch.float32)
    y = torch.zeros(S, C, dtype=torch.bfloat16)
    for counters in (torch.zeros(8, dtype=torch.int32), None):
        seen.clear()
        kcp._fwd_main(entry, x, w, y, None, counters, S=S, C=C, device=torch.device("cpu"))
        (values,) = seen
        assert values["enable_pdl"] == 0, counters
        if counters is None:
            continue
        for arch in ARCHES:
            for name in cake_jit.route_modules(arch, kcp.FWD_MAIN_ROUTE):
                missing = set(cake_jit.MODULES[name]["logical"].values()) - set(values)
                assert not missing, (arch, name, missing)


@pytest.mark.cpu_only
def test_single_call_forward_single_rank_launch_without_pdl(monkeypatch) -> None:
    """The single-call forward's previous host path (``kcp._kcp_forward`` with the private ``host_trim`` knob 0) launches
    the single-rank main program of the ``sr`` form on every architecture with the ``enable_pdl`` launch scalar 0, as
    the trimmed path's ``kcp._fwd_main`` does, and with every logical argument of that record, with and without
    token-chunk counters.  The device side (tensor checks, context tables, the plan and grid of the program, the launch)
    is stubbed; the form, the scalars and the launch values are the module's."""

    kcp = cake_package(KEY, "kcp")
    S, C = 16, 64
    seen: list[dict] = []
    counters: list = []
    monkeypatch.setitem(kcp._TEST_KNOBS, "host_trim", 0)
    monkeypatch.setattr(kcp, "LAST_CALL", {})
    monkeypatch.setattr(kcp, "_wait_ready", lambda ctx: None)
    monkeypatch.setattr(kcp, "_tokens", lambda x, what, *, device: (S, C, x.view(S, C)))
    monkeypatch.setattr(kcp, "_weight", lambda weight, C, *, device: weight)
    monkeypatch.setattr(
        kcp, "_context", lambda ctx, *, device, T_r: (torch.tensor([0, T_r], dtype=torch.int32), 1)
    )
    monkeypatch.setattr(
        kcp, "_output", lambda out, what, *, shape, dtype, device, align=1: torch.zeros(shape, dtype=dtype)
    )
    monkeypatch.setattr(
        kcp, "_table", lambda ctx, what, *, shape, dtype, device: torch.zeros(shape, dtype=dtype)
    )
    monkeypatch.setattr(kcp, "_device_index", lambda device: 0)
    monkeypatch.setattr(
        kcp,
        "_sr_main",
        lambda arch, *, S, C, n_frag, device_index: (
            cake_jit.selected_modules(arch, kcp.FWD_MAIN_ROUTE, C=C, S=S)[0],
            dict.fromkeys(cake_jit.PLAN_SCALARS, 1),
            (1, 1, 1),
            8,
        ),
    )
    monkeypatch.setattr(
        kcp, "_forward_work", lambda workspace, words, *, device, device_index: counters[0]
    )
    monkeypatch.setattr(kcp, "_zero_history_plan", lambda name, grid, device_index: name)
    monkeypatch.setattr(
        kcp, "_launch_trim", lambda name, values, device_index: seen.append((name, dict(values)))
    )
    ctx = type("Ctx", (), {"halo": type("Halo", (), {"n_hist_rows": 0})()})()
    x = torch.zeros(S, C, dtype=torch.bfloat16)
    w = torch.zeros(C, 4, dtype=torch.float32)
    hist = torch.zeros(0, C, dtype=torch.bfloat16)
    for arch in ARCHES:
        monkeypatch.setattr(kcp, "_device_arch", lambda device, arch=arch: arch)
        assert kcp._fwd_form(arch)[0], arch
        for work in (torch.zeros(8, dtype=torch.int32), None):
            seen.clear()
            counters[:] = [work]
            kcp._kcp_forward(x, w, ctx, _hist=hist)
            assert kcp.LAST_CALL["fwd"]["main"] == "sr", arch
            ((name, values),) = seen
            assert values["enable_pdl"] == 0, (arch, work is None)
            if work is None:
                continue
            missing = set(cake_jit.MODULES[name]["logical"].values()) - set(values)
            assert not missing, (arch, name, missing)


@pytest.mark.cpu_only
def test_main_backward_records_are_ordinary_launches_with_accumulators() -> None:
    vocabulary = set(kcp_jit.LOGICAL_TENSORS) | set(kcp_jit.LOGICAL_SCALARS)
    for arch, kind, name, det_name in _pairs():
        if kind != "kcp_bwd":
            continue
        record, det = kcp_jit.MODULES[name], det_kcp_jit.MODULES[det_name]
        label = (arch, name)
        assert record["role"] == "kcp_bwd" and record["arches"] == det["arches"], label
        assert record["ffi_entry"] == det["ffi_entry"] == "run", label
        assert _args(record) == (_args(det) - {"partial"}) | {"dw_acc", "work"}, label
        assert (
            record["logical"]["dw_acc"] == "acc" and record["logical"]["work"] == "work"
        ), label
        assert [k for k, _ in record["arg_plan"][-3:]] == ["grid", "grid", "grid"], (
            label
        )
        assert set(record["logical"]) == _args(record), label
        assert set(record["logical"].values()) <= vocabulary, label
        grid = record["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det["grid_rule"]["geometry"], label
        # ``grid_mult`` (a rendered leaf): one positive factor per architecture of the record.
        assert set(grid["grid_mult"]) == set(record["arches"]), label
        assert all(isinstance(f, int) and f >= 1 for f in grid["grid_mult"].values()), (
            label
        )
        assert isinstance(grid["ctas_per_sm"], int) and grid["ctas_per_sm"] >= 1, label
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, label
        assert launch["block"] == det["launch"]["block"], label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        assert kcp_jit.select_rule(record) == det_kcp_jit.select_rule(det), label
        assert _accumulator_rule(record["workspace_rule"]) == ACCUMULATORS, label


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_selection_matches_the_deterministic_kcp_registry(arch) -> None:
    for kind in kcp_jit.KINDS:
        new, det = kcp_jit.ROUTES[arch][kind], det_kcp_jit.ROUTES[arch][kind]
        for C in CHANNELS:
            (name,) = kcp_jit.selected_modules(arch, kind, C=C)
            (det_name,) = det_kcp_jit.selected_modules(arch, kind, C=C)
            assert new.index(name) == det.index(det_name), (arch, kind, C)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_backward_workspace(arch) -> None:
    """The KCP backward's ``acc`` / ``work`` bytes follow the zero-history layout under the legacy channel-tile rule
    (``ct_vec = min(C / 8, 128)``): ``align16(align16(16 C) + 4 n_ch_tiles)``; one FP32 owner per route."""

    owners = kcp_jit.workspace_owners(arch)
    assert [o["kernel"] for o in owners] == list(kcp_jit.route_modules(arch, "kcp_bwd"))
    assert (
        _accumulator_rule(kcp_jit.workspace_record(arch)["workspace_rule"])
        == ACCUMULATORS
    )
    for C in CHANNELS:
        n_ch_tiles = -(-(C // 8) // min(C // 8, 128))
        layout = kcp_jit.accumulator_layout(owners[0], C, n_ch_tiles)
        assert layout["acc_dtype"] == "float32" and layout["acc_elems"] == 4 * C
        expected = -(-(-(-16 * C // 16) * 16 + 4 * n_ch_tiles) // 16) * 16
        assert kcp_jit.workspace_bytes(arch, C) == layout["total_bytes"] == expected, (
            arch,
            C,
        )


@pytest.mark.cpu_only
def test_head_grid_is_the_deterministic_rule() -> None:
    for arch in ARCHES:
        (head,) = kcp_jit.ROUTES[arch]["kcp_bwd_head"]
        (det_head,) = det_kcp_jit.ROUTES[arch]["kcp_bwd_head"]
        for n_frag in (0, 1, 3, 17):
            for C in CHANNELS:
                assert kcp_jit.head_grid(
                    kcp_jit.MODULES[head], n_frag=n_frag, C=C
                ) == det_kcp_jit.head_grid(
                    det_kcp_jit.MODULES[det_head], n_frag=n_frag, C=C
                )


@pytest.mark.cpu_only
def test_overlapped_backward_records() -> None:
    """The overlapped backward's two routes: one record each on every architecture.  The main program is an
    ordinary ``co_resident`` launch of one CTA per SM (the host passes the grid) that owns an ``accumulators``
    workspace of ``slots`` FP32 copies of four taps, its logical map names ``acc``, ``tile_tk``, ``claims``,
    ``departs``, the two TMA tensors and its two tail tables (``tail_row_off`` and ``skip_row_off``, the rows it
    leaves to the tail-row program); the tail-row program has the head program's geometry and no workspace."""

    vocabulary = set(kcp_jit.LOGICAL_TENSORS) | set(kcp_jit.LOGICAL_SCALARS)
    for arch in ARCHES:
        (main,) = kcp_jit.route_modules(arch, kcp_jit.DYN_KIND)
        (halo,) = kcp_jit.route_modules(arch, kcp_jit.HALO_KIND)
        (head,) = kcp_jit.route_modules(arch, "kcp_bwd_head")
        for name, role in ((main, "kcp_bwd_dyn"), (halo, "kcp_bwd_halo")):
            record = kcp_jit.MODULES[name]
            label = (arch, name)
            assert record["role"] == role and arch in record["arches"], label
            assert record["ffi_entry"] == "run", label
            assert set(record["logical"]) == _args(record), label
            assert set(record["logical"].values()) <= vocabulary, label
            assert [k for k, _ in record["arg_plan"][-3:]] == ["grid"] * 3, label
            assert record["grid_rule"]["kind"] == "co_resident", label
            assert record["launch"]["cooperative"] is False, label
        record = kcp_jit.MODULES[main]
        logical = record["logical"]
        assert logical["dw_acc"] == "acc" and logical["tile_tk"] == "tile_tk"
        assert logical["claims"] == "claims" and logical["departs"] == "departs"
        assert logical["x_map"] == "x" and logical["dy_map"] == "dy"
        assert logical["tail_row_off"] == "tail_row_off"
        assert logical["skip_row_off"] == "skip_row_off"
        assert {logical[k] for k in ("M", "NL", "P", "L", "n_ctas")} == {
            "M",
            "NL",
            "P",
            "L",
            "n_ctas",
        }
        assert record["grid_rule"]["ctas_per_sm"] == 1
        rule = record["workspace_rule"]
        assert rule["kind"] == "accumulators" and rule["acc"] == "float32"
        assert rule["taps"] == 4 and rule["work"] == "uint32"
        assert isinstance(rule["slots"], int) and rule["slots"] >= 1
        record = kcp_jit.MODULES[halo]
        assert record["workspace_rule"] is None
        assert (
            record["grid_rule"]["geometry"]
            == kcp_jit.MODULES[head]["grid_rule"]["geometry"]
        )
        assert "dpre_tail" in record["logical"] and "tail_row_off" in record["logical"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("sms", [132, 148, 152, 160, 188, 212])
def test_dyn_plan_meets_the_kernel_requirements(sms) -> None:
    """:func:`kcp_jit.dyn_plan` gives every work plan the main program's ``kernel_requires`` admit: ``L`` lanes of
    ``P`` per tile of 2048 channels, at most 256; one CTA per SM and at least one per lane; units of ``M`` tokens (a
    multiple of 8, at least 24); ``NL`` units per lane covering the rank's tokens; the claim words' 16-bit counts.
    The same with each world size of the context-parallel unit table and each architecture of its measured parts."""

    for C in (8, 512, 1024, 2048, 2056, 8192, 10240, 32768, 65536):
        n_tiles = -(-C // kcp_jit.DYN_TILE)
        for S in (1, 7, 64, 1003, 2048, 4096, 8192, 12288, 16384, 18432, 24576, 32768, 1 << 20, 1 << 23):
            for ws in (None, 2, 4, 8):
                for arch in (None, "sm_100a", "sm_103a", "sm_107a"):
                    p = kcp_jit.dyn_plan(S, C, sms, world_size=ws, arch=arch)
                    label = (sms, C, S, ws, arch, p)
                    assert p["n_tiles"] == n_tiles and p["L"] == n_tiles * p["P"], label
                    assert 1 <= p["P"] <= 512 and p["L"] <= kcp_jit.DYN_MAX_LANES, label
                    assert p["n_ctas"] == max(p["L"], sms), label
                    assert p["M"] % 8 == 0 and p["M"] >= 24, label
                    assert 1 <= p["NL"] <= 32768 and p["P"] * p["NL"] * p["M"] >= S, label
                    assert (p["L"] + 1) * p["NL"] + p["n_ctas"] <= 65535, label
                    if sms * p["M"] <= S and n_tiles <= sms:
                        assert p["L"] >= sms - n_tiles + 1, label
    for S, C in ((0, 1024), (16, 0), (16, kcp_jit.DYN_MAX_CHANNELS + 8)):
        with pytest.raises(ValueError):
            kcp_jit.dyn_plan(S, C, 148)
    assert kcp_jit.dyn_applies(kcp_jit.DYN_MAX_CHANNELS, 1)
    assert not kcp_jit.dyn_applies(kcp_jit.DYN_MAX_CHANNELS + 8, 1)
    assert not kcp_jit.dyn_applies(1024, 0)


@pytest.mark.cpu_only
def test_dyn_plan_unit_size() -> None:
    """:func:`kcp_jit.dyn_plan` takes the 64-token plan when it leaves the last lane of each tile without tokens,
    or the 56-token plan (``DYN_IDLE_UNIT``) when that one leaves more idle CTAs (``n_ctas`` less the lanes with
    tokens; the 64-token plan on a tie); otherwise the unit size ``M`` (a multiple of 8 from 64 to 128) whose fitting plan has the least lane cost
    ``NL * M`` tokens plus ``NL`` fixed unit costs plus a tail cost per token of a unit, the smaller ``M`` on a tie;
    above that range it doubles ``M`` from 256 until the claim counts fit."""

    def lane_cost(p):
        unit = 10 * p["M"] + kcp_jit.DYN_UNIT_COST_TENTHS
        return p["NL"] * unit + kcp_jit.DYN_TAIL_COST_TENTHS * p["M"]

    # the customer per-rank shapes.  148-SM B200 (every lane busy; costs in tenths of a token): one tensor of 10240
    # channels, 32768 tokens (P 29: 11 units of 104 tokens, 11 * 1072 + 3 * 104 = 12104, against 13 x 88: 12120
    # and 18 x 64: 12288) and 8192 channels, 16384 tokens (P 37: 7 x 64, 7 * 672 + 192 = 4896, against 4 x 112:
    # 4944).  152 SMs (GB300, GB200) and 212 SMs (VR200): the 64-token lanes of 1152, 448, 832 and 320 tokens fill
    # 29 of 30, 37 of 38, 40 of 42 and 52 of 53 lanes, so M is 64 unless 56-token lanes leave more CTAs idle: they
    # leave 2, 4 (a tie), 2 and 16 idle against 7, 4, 12 and 4, so only the last (VR200, 8192 channels) takes
    # 6 units of 56 tokens (49 of 53 lanes busy).
    for (S, C, sms), want in (
        ((32768, 10240, 148), {"M": 104, "NL": 11, "P": 29, "L": 145, "n_ctas": 148}),
        ((16384, 8192, 148), {"M": 64, "NL": 7, "P": 37, "L": 148, "n_ctas": 148}),
        ((32768, 10240, 152), {"M": 64, "NL": 18, "P": 30, "L": 150, "n_ctas": 152}),
        ((16384, 8192, 152), {"M": 64, "NL": 7, "P": 38, "L": 152, "n_ctas": 152}),
        ((32768, 10240, 212), {"M": 64, "NL": 13, "P": 42, "L": 210, "n_ctas": 212}),
        ((16384, 8192, 212), {"M": 56, "NL": 6, "P": 53, "L": 212, "n_ctas": 212}),
    ):
        p = kcp_jit.dyn_plan(S, C, sms)
        assert {k: p[k] for k in want} == want, (S, C, sms, p)
    branches = {"idle": 0, "idle_unit": 0, "cost": 0, "doubling": 0}
    for sms in (132, 148, 152, 188, 212):
        for C in (8, 2048, 8192, 10240, 65536):
            n_tiles = -(-C // kcp_jit.DYN_TILE)
            for S in (1, 7, 1003, 4096, 16384, 32768, 100000, 1 << 20, 1 << 23):

                def plan_for(M):
                    lanes = min(sms // n_tiles, -(-S // M), 256 // n_tiles)
                    P = max(1, lanes)
                    per_lane = -(-S // P)
                    NL = -(-per_lane // M)
                    L = n_tiles * P
                    if (L + 1) * NL + max(L, sms) > 65535 or NL > 32768:
                        return None
                    n_ctas = max(L, sms)
                    return {"M": M, "NL": NL, "P": P, "L": L, "n_ctas": n_ctas}

                p = kcp_jit.dyn_plan(S, C, sms)
                p = {k: p[k] for k in ("M", "NL", "P", "L", "n_ctas")}
                fits = [q for q in map(plan_for, range(64, 129, 8)) if q is not None]
                label = (sms, C, S, p)

                def idle(q):
                    busy = min(q["P"], -(-S // (q["NL"] * q["M"])))
                    return q["n_ctas"] - n_tiles * busy

                base = plan_for(64)
                if base is not None and -(-S // (base["NL"] * 64)) < base["P"]:
                    small = plan_for(56)
                    if small is not None and idle(small) > idle(base):
                        branches["idle_unit"] += 1
                        assert p == small, label
                    else:
                        branches["idle"] += 1
                        assert p == base, label
                elif fits:
                    branches["cost"] += 1
                    best = min(lane_cost(q) for q in fits)
                    assert lane_cost(p) == best, label
                    tied = [q["M"] for q in fits if lane_cost(q) == best]
                    assert p["M"] == min(tied), label
                else:
                    # the first power of two from 256 whose plan fits (the previous doubling rule)
                    branches["doubling"] += 1
                    assert p["M"] >= 256 and p["M"] & (p["M"] - 1) == 0, label
                    assert p == plan_for(p["M"]), label
                    assert p["M"] == 256 or plan_for(p["M"] // 2) is None, label
    assert all(n > 0 for n in branches.values()), branches
    # a rank too long for any unit up to 128 tokens: the doubling from 256 (32 tiles, 4 lanes each, 256 units)
    assert kcp_jit.dyn_plan(1 << 23, 65536, 148)["M"] == 8192


@pytest.mark.cpu_only
def test_dyn_plan_context_parallel_table() -> None:
    """With ``world_size``, :func:`kcp_jit.dyn_plan` takes the unit of the context-parallel table
    (:func:`kcp_jit.dyn_cp_unit`) for calls of at most 512 channels: 32 tokens up to 4096 tokens on the rank at world
    sizes 2 and 4 and up to 2048 at world size 8; 56 tokens from 16384 tokens at world size 2 when its lanes' runs
    (``NL * M``) are no longer than the 64-token plan's.  Every other call, a table unit whose plan breaks the claim
    counts, and ``world_size=None`` get the plan without the table."""

    dyn, cp_unit = kcp_jit.dyn_plan, kcp_jit.dyn_cp_unit

    def plan_for(S, C, sms, M):
        n_tiles = -(-C // kcp_jit.DYN_TILE)
        P = max(1, min(sms // n_tiles, -(-S // M), kcp_jit.DYN_MAX_LANES // n_tiles))
        NL = -(-(-(-S // P)) // M)
        L = n_tiles * P
        n_ctas = max(L, sms)
        if (L + 1) * NL + n_ctas > 65535 or NL > 32768:
            return None
        return {"M": M, "NL": NL, "P": P, "L": L, "n_ctas": n_ctas, "n_tiles": n_tiles}

    def run(p):
        return p["NL"] * p["M"]

    # the table's classes and edges
    for ws, last in ((2, 4096), (4, 4096), (8, 2048)):
        for S in (1, 7, last // 2, last):
            assert cp_unit(ws, S, 512) == 32, (ws, S)
            assert cp_unit(ws, S, 513) is None, (ws, S)
        assert cp_unit(ws, last + 1, 512) is None, ws
    for S in (16383, 16384, 32768, 1 << 20):
        assert cp_unit(2, S, 512) == (56 if S >= 16384 else None), S
        assert cp_unit(2, S, 513) is None, S
        for ws in (1, 3, 4, 8, 16):
            assert cp_unit(ws, S, 512) is None, (ws, S)
    for ws in (1, 3, 16):
        assert cp_unit(ws, 1, 512) is None, ws
    # the measured rows (C 512): 148-SM B200, 152-SM GB300, 212-SM VR200
    for sms in (148, 152, 212):
        assert dyn(4096, 512, sms, world_size=2) == plan_for(4096, 512, sms, 32)
        assert dyn(4096, 512, sms, world_size=4)["M"] == 32
        assert dyn(2048, 512, sms, world_size=8)["M"] == 32
        assert dyn(4096, 512, sms, world_size=8) == dyn(4096, 512, sms)
        for S in (6144, 8192, 12288):
            assert dyn(S, 512, sms, world_size=2) == dyn(S, 512, sms)
        for S in (16384, 32768):
            assert dyn(S, 512, sms, world_size=2) == plan_for(S, 512, sms, 56)
        # the lane-run check holds on every measured row from 6144 tokens except 12288 tokens on 212 SMs, where
        # 56-token units ran slower (220 units for 212 CTAs: runs of 112 tokens against 64)
        for S in (6144, 8192, 12288, 16384, 32768):
            longer = run(plan_for(S, 512, sms, 56)) > run(plan_for(S, 512, sms, 64))
            assert longer == (sms == 212 and S == 12288), (sms, S)
    # the lane-run check: 18944 tokens on 148 SMs give 56-token runs of 168 tokens against 128, so the plan without the
    # table (64-token units, every lane busy)
    assert run(plan_for(18944, 512, 148, 56)) == 168
    assert run(plan_for(18944, 512, 148, 64)) == 128
    assert dyn(18944, 512, 148, world_size=2) == dyn(18944, 512, 148)
    assert dyn(18944, 512, 148)["M"] == 64
    # the claim-count limit: 56-token units of 2^23 tokens on 148 SMs break it, so the plan without the table
    assert plan_for(1 << 23, 512, 148, 56) is None
    assert dyn(1 << 23, 512, 148, world_size=2) == dyn(1 << 23, 512, 148)
    # every call: the table's plan where it applies (and, for 56 tokens, the lane-run check passes), else the plan
    # without it
    edges = (1, 7, 1003, 2048, 2049, 4096, 4097, 12288, 16383, 16384)
    large = (18944, 24576, 32768, 100000, 1 << 20)
    taken = {32: 0, 56: 0, "longer": 0, "other": 0}
    for sms in (132, 148, 152, 188, 212):
        for C in (8, 256, 512, 513, 2048, 8192):
            for S in (*edges, *large):
                today = dyn(S, C, sms)
                assert dyn(S, C, sms, world_size=None) == today, (sms, C, S)
                for ws in (1, 2, 3, 4, 8, 16):
                    p = dyn(S, C, sms, world_size=ws)
                    label = (sms, C, S, ws, p)
                    unit = cp_unit(ws, S, C)
                    want = None if unit is None else plan_for(S, C, sms, unit)
                    ref = plan_for(S, C, sms, 64)
                    if unit == 56 and want is not None and run(want) > run(ref):
                        taken["longer"] += 1
                        want = None
                    if want is None:
                        taken["other"] += 1
                        assert p == today, label
                    else:
                        taken[unit] += 1
                        assert p == want, label
    assert all(n > 0 for n in taken.values()), taken


@pytest.mark.cpu_only
def test_dyn_plan_unit_table_measured_parts() -> None:
    """With ``world_size`` and ``arch``, a call on the table's measured parts (148-SM sm_100a at world sizes 2, 4 and
    8; 152-SM sm_103a and 212-SM sm_107a at 2 and 4; at most 512 channels and 32768 tokens on the rank) tries
    40-token units on sm_107a at world sizes 2 and 4 from 16384 tokens (at most 2 units per lane), then 56-token
    units at world size 2 from 8192 tokens and on sm_103a at world size 4 from 16384 tokens (both with the lane-run
    check), and without a table unit takes the 56-token idle-unit plan only when its lanes' runs are no longer than
    the 64-token plan's.  Every other call keeps the plan without ``arch``."""

    dyn = kcp_jit.dyn_plan
    applies = kcp_jit.dyn_cp_v2_applies

    def plan_for(S, C, sms, M):
        n_tiles = -(-C // kcp_jit.DYN_TILE)
        P = max(1, min(sms // n_tiles, -(-S // M), kcp_jit.DYN_MAX_LANES // n_tiles))
        NL = -(-(-(-S // P)) // M)
        L = n_tiles * P
        n_ctas = max(L, sms)
        if (L + 1) * NL + n_ctas > 65535 or NL > 32768:
            return None
        return {"M": M, "NL": NL, "P": P, "L": L, "n_ctas": n_ctas, "n_tiles": n_tiles}

    def run(p):
        return p["NL"] * p["M"]

    def mnp(p):
        return (p["M"], p["NL"], p["P"])

    # the parts and their edges
    for arch, sms, worlds in (("sm_100a", 148, (2, 4, 8)), ("sm_103a", 152, (2, 4)), ("sm_107a", 212, (2, 4))):
        for ws in (1, 2, 3, 4, 8, 16):
            assert applies(arch, sms, ws, 16384, 512) == (ws in worlds), (arch, ws)
        assert applies(arch, sms, 2, 32768, 512) and not applies(arch, sms, 2, 32769, 512), arch
        assert applies(arch, sms, 2, 1, 512) and not applies(arch, sms, 2, 0, 512), arch
        assert not applies(arch, sms, 2, 16384, 513), arch
        assert not applies(arch, sms, None, 16384, 512) and not applies(None, sms, 2, 16384, 512), arch
        for other in (132, 148, 152, 160, 188, 212, 216):
            if other != sms:
                assert not applies(arch, other, 2, 16384, 512), (arch, other)
    # the measured rows (C 512, world size 2 unless named): (arch, sms, ws, S) -> (M, NL, P) with arch, and without
    pins = {
        ("sm_100a", 148, 2, 18432): ((64, 2, 148), (56, 3, 148)),
        ("sm_103a", 152, 2, 18432): ((64, 2, 152), (56, 3, 152)),
        ("sm_107a", 212, 2, 24576): ((64, 2, 212), (56, 3, 212)),
        ("sm_107a", 212, 2, 16384): ((40, 2, 212), (56, 2, 212)),
        ("sm_107a", 212, 4, 16384): ((40, 2, 212), (64, 2, 212)),
        ("sm_100a", 148, 2, 12288): ((56, 2, 148), (64, 2, 148)),
        ("sm_103a", 152, 2, 12288): ((56, 2, 152), (64, 2, 152)),
        ("sm_107a", 212, 2, 8192): ((56, 1, 147), (64, 1, 128)),
        ("sm_107a", 212, 2, 12288): ((64, 1, 192), (64, 1, 192)),
        ("sm_103a", 152, 4, 16384): ((56, 2, 152), (64, 2, 152)),
        ("sm_107a", 212, 2, 32768): ((56, 3, 212), (56, 3, 212)),
        ("sm_103a", 152, 8, 18432): ((56, 3, 152), (56, 3, 152)),
        ("sm_107a", 216, 2, 24576): ((56, 3, 216), (56, 3, 216)),
    }
    for (arch, sms, ws, S), (with_arch, without) in pins.items():
        assert mnp(dyn(S, 512, sms, world_size=ws, arch=arch)) == with_arch, (arch, sms, ws, S)
        assert mnp(dyn(S, 512, sms, world_size=ws)) == without, (arch, sms, ws, S)
    # every call: on the parts the units in order, then the guarded idle-unit plan; elsewhere the plan without arch
    edges = (1, 7, 1003, 2048, 2049, 4096, 4097, 8191, 8192, 12288, 14336, 16383, 16384, 16960, 16961, 18432)
    large = (20480, 24576, 26624, 28672, 32768, 32769, 65536, 1 << 20)
    taken = {32: 0, 40: 0, 56: 0, "guard": 0, "other": 0}
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for sms in (132, 148, 152, 160, 188, 212, 216):
            for C in (8, 256, 512, 513, 2048):
                for S in (*edges, *large):
                    today = dyn(S, C, sms)
                    assert dyn(S, C, sms, world_size=None, arch=arch) == today, (arch, sms, C, S)
                    for ws in (1, 2, 3, 4, 8, 16):
                        p = dyn(S, C, sms, world_size=ws, arch=arch)
                        label = (arch, sms, C, S, ws, p)
                        if not applies(arch, sms, ws, S, C):
                            assert p == dyn(S, C, sms, world_size=ws), label
                            continue
                        ref = plan_for(S, C, sms, 64)
                        short = kcp_jit.DYN_CP_SHORT_TOKENS.get(ws)
                        units = []
                        if short is not None and S <= short:
                            units = [32]
                        else:
                            if arch == "sm_107a" and ws in (2, 4) and S >= 16384:
                                units.append(40)
                            if (ws == 2 and S >= 8192) or (arch == "sm_103a" and ws == 4 and S >= 16384):
                                units.append(56)
                        want = None
                        for unit in units:
                            cp = plan_for(S, C, sms, unit)
                            if cp is not None and unit != 32:
                                if ref is None or run(cp) > run(ref) or (unit == 40 and cp["NL"] > 2):
                                    cp = None
                            if cp is not None:
                                want = cp
                                taken[unit] += 1
                                break
                        if want is None:
                            if today["M"] == 56 and ref is not None and run(today) > run(ref):
                                want = ref
                                taken["guard"] += 1
                            else:
                                want = today
                                taken["other"] += 1
                        assert p == want, label
    assert all(n > 0 for n in taken.values()), taken


# The L1o route table's keys per (architecture, SM count): (world size, channels, tokens on the rank).
L1O_KEYS = {
    ("sm_100a", 148): frozenset(
        {
            (2, 256, 32768),
            (2, 512, 4096),
            (2, 512, 8192),
            (2, 512, 12288),
            (2, 512, 16384),
            (2, 512, 18432),
            (2, 512, 20480),
            (2, 512, 24576),
            (2, 512, 26624),
            (2, 512, 28672),
            (2, 512, 32768),
            (2, 768, 16384),
            (2, 768, 32768),
            (2, 1024, 32768),
            (4, 512, 4096),
            (4, 512, 16384),
            (4, 512, 18432),
            (4, 512, 24576),
            (4, 512, 32768),
            (8, 512, 16384),
            (8, 512, 18432),
            (8, 512, 32768),
        }
    ),
    ("sm_103a", 152): frozenset(
        {
            (2, 256, 32768),
            (2, 512, 4096),
            (2, 512, 6144),
            (2, 512, 8192),
            (2, 512, 12288),
            (2, 512, 16384),
            (2, 512, 18432),
            (2, 512, 20480),
            (2, 512, 24576),
            (2, 512, 28672),
            (2, 512, 32768),
            (2, 768, 16384),
            (2, 768, 32768),
            (2, 1024, 32768),
            (4, 512, 4096),
            (4, 512, 16384),
            (4, 512, 18432),
            (4, 512, 24576),
            (4, 512, 32768),
        }
    ),
    ("sm_107a", 212): frozenset(
        {
            (2, 512, 14336),
            (2, 512, 16384),
            (2, 512, 18432),
            (2, 512, 20480),
            (2, 512, 28672),
            (2, 512, 32768),
            (2, 768, 16384),
            (2, 768, 32768),
            (4, 512, 4096),
            (4, 512, 16384),
            (4, 512, 32768),
        }
    ),
}


@pytest.mark.cpu_only
def test_l1o_route_table() -> None:
    """The L1o route (``kcp_jit.l1o_route``) applies exactly on the keys of ``kcp_jit.L1O_ROUTE`` (a measured
    (architecture, SM count) device and a listed (world size, channels, tokens on the rank)) and returns the SMs its
    one-program grid leaves free on that architecture; every other call (no architecture or world size, another SM
    count, world size, channel count or token count) keeps the overlapped pair (``None``)."""

    route = kcp_jit.l1o_route
    table = kcp_jit.L1O_ROUTE
    reserve = kcp_jit.L1O_RESERVE_SMS
    assert {k: len(v) for k, v in table.items()} == {("sm_100a", 148): 22, ("sm_103a", 152): 19, ("sm_107a", 212): 11}
    # the key set of each device, written out (the backend README's table): any moved, added or dropped key fails
    assert table == L1O_KEYS
    assert reserve == {"sm_100a": 3, "sm_103a": 10, "sm_107a": 0}
    for (arch, sms), keys in table.items():
        worlds = {w for w, _, _ in keys}
        assert worlds <= {2, 4, 8}
        for W, C, S in keys:
            label = (arch, sms, W, C, S)
            assert route(arch, sms, W, S, C) == reserve[arch], label
            assert route(None, sms, W, S, C) is None, label
            assert route(arch, sms, None, S, C) is None, label
            for other in (sms - 1, sms + 1, 132, 160, 188, 216):
                if (arch, other) not in table:
                    assert route(arch, other, W, S, C) is None, (label, other)
            for w in (1, 3, 16):
                assert route(arch, sms, w, S, C) is None, (label, w)
            for s in (S - 8, S + 8, 2 * S):
                if (W, C, s) not in keys:
                    assert route(arch, sms, W, s, C) is None, (label, s)
            for c in (C - 1, C + 1, 2048):
                if (W, c, S) not in keys:
                    assert route(arch, sms, W, S, c) is None, (label, c)
        for other_arch in ("sm_100a", "sm_103a", "sm_107a"):
            if (other_arch, sms) not in table:
                assert all(route(other_arch, sms, W, S, C) is None for W, C, S in keys), (other_arch, sms)


@pytest.mark.cpu_only
def test_dyn_host_route_state(monkeypatch) -> None:
    """``kcp._dyn_host`` passes the architecture, the device's SM count and the call's world size to
    ``kcp_jit.l1o_route`` and ``kcp_jit.dyn_plan``.  On every key of ``kcp_jit.L1O_ROUTE`` (sm_107a with its reserve
    0 included) it returns the L1o state (``kcp._l1o_host``): ``"l1o"``, the ``kcp_bwd`` records the call selects,
    each launched over ``kcp._bwd_grid`` with the architecture's reserve (a single-wave grid less ``reserve x
    ctas_per_sm`` CTAs), an all ``-1`` ``no_tail`` table and the tail-row plan only when the rank receives tail rows;
    no work plan is made.  A ``pair`` call, a call without a world size, another SM count or a shape off the keys gets
    the overlapped pair's state (``"main"``, no ``"l1o"``) from ``dyn_plan`` with the architecture.  The device side
    (SM count, architecture, launch grid, module loads) is stubbed."""

    kcp = cake_package(KEY, "kcp")
    device = {"arch": "sm_100a", "sms": 148}
    routed: list[tuple] = []
    planned: list[dict] = []
    real_route, real_plan = kcp.l1o_route, kcp.dyn_plan

    def grid(record, *, device_index, C=None, S=None):
        rule = record["grid_rule"]
        n = int(rule["ctas_per_sm"]) * int(rule["grid_mult"][device["arch"]]) * device["sms"]
        return (n, 1, 1)

    def route(*args):
        routed.append(args)
        return real_route(*args)

    def dyn_plan(*args, **kwargs):
        planned.append(dict(kwargs, args=args))
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(kcp._loader, "sm_count", lambda device_index: device["sms"])
    monkeypatch.setattr(kcp._loader, "device_arch", lambda device_index: device["arch"])
    monkeypatch.setattr(kcp, "launch_grid", grid)
    monkeypatch.setattr(kcp, "_launch_plan", lambda name, grid_, device_index: (name, tuple(grid_)))
    monkeypatch.setattr(kcp, "l1o_route", route)
    monkeypatch.setattr(kcp, "dyn_plan", dyn_plan)
    cpu = torch.device("cpu")

    def host(arch, W, C, S, *, tail_rows=3, pair=False):
        routed.clear()
        planned.clear()
        return kcp._dyn_host(
            arch, S=S, C=C, n_frag=3, tail_rows=tail_rows, device=cpu, device_index=0, pair=pair, world_size=W
        )

    def overlapped_pair(state, arch, W, S, label):
        assert "l1o" not in state and "main" in state, label
        (seen,) = planned
        assert seen["arch"] == arch and seen["world_size"] == W and seen["args"][:3] == (S, 512, device["sms"]), label

    n_keys = 0
    for (arch, sms), keys in kcp_jit.L1O_ROUTE.items():
        device.update(arch=arch, sms=sms)
        reserve = kcp_jit.L1O_RESERVE_SMS[arch]
        for W, C, S in sorted(keys):
            label = (arch, sms, W, C, S)
            for tail_rows in (3, 0):
                state = host(arch, W, C, S, tail_rows=tail_rows)
                assert routed == [(arch, sms, W, S, C)] and not planned, label
                assert state["l1o"] is True, label
                names = kcp_jit.selected_modules(arch, kcp.KCP_BWD, C=C, S=S)
                assert tuple(name for name, _ in state["launches"]) == names, label
                tiles = kcp.plan(
                    S, C, 3, geometry=kcp_jit.MODULES[names[0]]["grid_rule"]["geometry"], legacy=True
                )["n_ch_tiles"]
                for name, launch in state["launches"]:
                    rule = kcp_jit.MODULES[name]["grid_rule"]
                    k = int(rule["ctas_per_sm"])
                    full = grid(kcp_jit.MODULES[name], device_index=0)[0]
                    single_wave = int(rule["grid_mult"][arch]) == 1
                    if reserve and single_wave:
                        want = full - reserve * k  # the L1o reserve
                    elif single_wave and full % tiles >= k:
                        want = full - k  # the trim of every single-wave backward grid
                    else:
                        want = full
                    assert launch == (want, 1, 1), (label, name, launch, want)
                assert state["no_tail"].shape == (3, kcp.HALO), label
                assert bool((state["no_tail"] == -1).all()), label
                assert (state["halo"] is None) == (tail_rows == 0), label
            n_keys += 1
    assert n_keys == 52
    # every other call keeps the overlapped pair, with the architecture in its work plan
    for arch, sms, W, S in (("sm_100a", 148, 2, 4096), ("sm_103a", 152, 4, 4096), ("sm_107a", 212, 4, 4096)):
        device.update(arch=arch, sms=sms)
        assert host(arch, W, 512, S)["l1o"] is True, arch
        overlapped_pair(host(arch, W, 512, S, pair=True), arch, W, S, (arch, "pair"))
        assert not routed, arch  # a pair call never asks the route
        overlapped_pair(host(arch, None, 512, S), arch, None, S, (arch, "no world size"))
        assert routed == [(arch, sms, None, S, 512)], arch
        overlapped_pair(host(arch, W, 512, S + 8), arch, W, S + 8, (arch, "off the keys"))
        overlapped_pair(host(arch, 3, 512, S), arch, 3, S, (arch, "world size 3"))
        device.update(sms=sms + 4)
        overlapped_pair(host(arch, W, 512, S), arch, W, S, (arch, "another SM count"))


@pytest.mark.cpu_only
def test_dyn_workspace_layout() -> None:
    """The main program's workspace: ``slots`` copies of ``[C, 4]`` FP32 accumulators, then eight words per tile,
    ``L`` claim words and one departure word."""

    for arch in ARCHES:
        (main,) = kcp_jit.route_modules(arch, kcp_jit.DYN_KIND)
        record = kcp_jit.MODULES[main]
        slots = record["workspace_rule"]["slots"]
        for C in (8, 2048, 10240, 65536):
            p = kcp_jit.dyn_plan(32768, C, 148)
            layout = kcp_jit.dyn_workspace_layout(record, C, p["n_tiles"], p["L"])
            assert layout["acc_dtype"] == "float32"
            assert layout["acc_elems"] == slots * 4 * C
            assert layout["work_dtype"] == "uint32"
            assert layout["work_words"] == 8 * p["n_tiles"] + p["L"] + 1
            assert layout["work_offset"] % 16 == 0
            assert (
                layout["total_bytes"]
                >= layout["work_offset"] + 4 * layout["work_words"]
            )
