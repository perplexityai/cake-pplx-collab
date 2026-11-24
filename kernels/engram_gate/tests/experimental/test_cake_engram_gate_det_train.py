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

Backend suite of ``cake_pplx._backends.cake_engram_gate_det_train``, the deterministic package of the Engram
context gate (four streams of width 5120).

Without a device: the registry's records, stages, architectures and launch facts; the record of each stage and dot
product summation (two forwards, one backward), and the forward record of each ``dot_sum`` choice; the persistent grid
sizes and the resident cluster table they come from; every stage launcher passes every kernel parameter of its records
in plan order (the FFI entry replaced by a recorder); ``build.json`` covers every record on every architecture and its
source digests; the loader imports standalone; a prebuilt library is preferred; the package reads no environment
variable but the JIT shim's.

On SM100 / SM103 / SM107: every record loads; each forward record writes every element of a NaN-filled ``out`` and
meets the naive BF16 tier against an FP64 oracle (``tests/cake_naive.py``, 32 draws); the backward meets the tier for
``dx``, ``dkey``, ``dvalue``, ``dq_weight`` and ``dk_weight``; ``dw_reduce`` alone meets the tier on random partial
rows, with FP32 and BF16 weights; its BF16 weight gradients hold the bits of torch's ``.to(torch.bfloat16)`` of its
FP32 ones over 2^24 FP32 patterns.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_engram_gate_naive import HIDDEN, STREAMS, judge_draws, make_inputs
from tests.cake.test_helpers.cake_package import cake_package

PACKAGE = "cake.cake_engram_gate_det_train"
pkg = cake_package(PACKAGE)
cake_backend = cake_package(PACKAGE, "cake_backend")
cake_jit = cake_package(PACKAGE, "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
RECORDS = ("bwd_dot2", "bwd_dot2_order", "bwd_dot2_stage", "dw_reduce", "fwd_dot2", "fwd_f32", "fwd_f32_cta1")
# The summations of the dot product each stage ships: the forward both, the backward the compensated one only.
ACC_MODES = {"fwd": ("f32", "dot2"), "bwd": ("dot2",)}
FWD_ACCS = ACC_MODES["fwd"]
# The public ``dot_sum`` choices and the forward record suffix each launches; the first is the default.
DOT_SUMS = {"fp32": "f32", "compensated": "dot2"}
ENTRIES = (
    "cake_engram_gate_det",
    "cake_engram_gate_det_kv",
    "cake_engram_gate_det_train_backward",
    "cake_engram_gate_det_train_forward",
    "cake_engram_gate_det_train_workspace_bytes",
    "is_cake_engram_gate_det_train_supported",
)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if cake_backend.arch_for_capability(capability) is None:
        pytest.skip(f"the package ships {ARCHES}; this device is {capability}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in cake_backend.specs_for_arch(cake_backend.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not rendered)")
    return json.loads(path.read_text())


# ----------------------------------------------------------------------------------------------------------------
# Without a device: the registry and the host plan


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert cake_jit.ARCH_BY_CAPABILITY == {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
    assert cake_jit.STAGES == ("fwd", "bwd", "dw_reduce")
    assert cake_jit.ACC_MODES == ACC_MODES
    assert sorted(cake_jit.MODULES) == sorted(RECORDS)
    for name, rec in cake_jit.MODULES.items():
        assert rec["arches"] == list(ARCHES), name
        assert rec["ffi_entry"] == "run", name
        assert rec["stage"] in cake_jit.STAGES, name
        assert name == rec["stage"] or name.startswith(rec["stage"] + "_"), name
        assert rec["library"].startswith(cake_jit.LIBRARY_PREFIX + "_"), name
        assert rec["kernel_symbol"] == f"kernel_{rec['library']}", name
        launch = rec["launch"]
        assert set(launch) == {"block", "cluster", "cooperative", "dynamic_smem_bytes", "use_pdl"}, name
        assert launch["cooperative"] is False, name
        if rec["stage"] != "dw_reduce":
            # a cta1 forward is one CTA of 640 threads a token, with its cluster program's argument plan
            cta1 = name in cake_jit.CTA1_RECORDS
            assert tuple(launch["cluster"]) == (1 if cta1 else cake_jit.CLUSTER_CTAS, 1, 1), name
            assert cake_jit.unit_ctas(name) == launch["cluster"][0], name
            if cta1:
                assert tuple(launch["block"]) == (640, 1, 1), name
                assert rec["arg_plan"] == cake_jit.MODULES[name.removesuffix("_cta1")]["arg_plan"], name
        kinds = [kind for kind, _ in rec["arg_plan"]]
        assert kinds[-3:] == ["grid", "grid", "grid"], name
        assert cake_backend.parameters(name), name


def test_each_dot_sum_has_a_forward_record_and_one_backward_serves_both() -> None:
    """Each ``dot_sum`` choice names one forward record (``"compensated"``: ``fwd_dot2``, ``"fp32"``: ``fwd_f32``);
    the backward ships ``bwd_dot2`` only (no one-word backward record); another choice or summation is refused."""
    assert cake_backend._DOT_SUMS == DOT_SUMS
    assert sorted(DOT_SUMS.values()) == sorted(FWD_ACCS)
    for choice, acc in DOT_SUMS.items():
        assert cake_backend._dot_acc(choice) == acc
        assert cake_jit.record("fwd", acc) == f"fwd_{acc}"
        assert cake_jit.MODULES[f"fwd_{acc}"]["stage"] == "fwd"
    for bad in ("dot2", "f32", "FP32", "", "compensated "):
        with pytest.raises(ValueError, match=r"dot_sum must be one of \['fp32', 'compensated'\]"):
            cake_backend._dot_acc(bad)
    for bad in (None, 1, b"fp32", ("fp32",)):
        with pytest.raises(TypeError, match=r"dot_sum must be a str, one of \['fp32', 'compensated'\]"):
            cake_backend._dot_acc(bad)
    assert cake_jit.record("bwd", "dot2") == cake_jit.record("bwd") == "bwd_dot2"
    bwd = {cake_jit.MODULES[name]["stage"] for name in ("bwd_dot2", "bwd_dot2_stage", "bwd_dot2_order")}
    assert bwd == {"bwd"}
    # the three backward programs: a call without a token mask launches bwd_dot2_stage from STAGE_BWD_MIN_TOKENS
    # tokens on the architectures where it is the faster one; a call with one launches bwd_dot2_order from the
    # ORDER_BWD_MIN_TOKENS tokens of its architecture; every other call launches bwd_dot2
    assert cake_jit.STAGE_BWD_ARCHES <= set(ARCHES)
    assert set(cake_jit.ORDER_BWD_MIN_TOKENS) <= set(ARCHES)
    assert cake_jit.ORDER_BWD_MIN_TOKENS == {"sm_100a": 8192, "sm_103a": 8192, "sm_107a": 4096}
    least = cake_jit.STAGE_BWD_MIN_TOKENS
    for arch in ARCHES:
        order = cake_jit.ORDER_BWD_MIN_TOKENS.get(arch)
        for T in sorted({1, least - 1, least, 2**31 - 1} | ({order - 1, order} if order else set())):
            ordered = order is not None and T >= order
            assert cake_jit.backward_record(arch, masked=True, T=T) == ("bwd_dot2_order" if ordered else "bwd_dot2")
            staged = arch in cake_jit.STAGE_BWD_ARCHES and T >= least
            want = "bwd_dot2_stage" if staged else "bwd_dot2"
            assert cake_jit.backward_record(arch, masked=False, T=T) == want, (arch, T)
    assert "bwd_f32" not in cake_jit.MODULES
    # fwd_f32's cta1 form (the same bits; fwd_dot2 has none): a call launches it from CTA1_FWD_MIN_TOKENS tokens on
    # the devices (architecture, SM count) and summations that table holds, the cluster program everywhere else
    assert cake_jit.CTA1_RECORDS == {"fwd_f32_cta1"}
    assert "fwd_dot2_cta1" not in cake_jit.MODULES
    assert cake_jit.CTA1_FWD_MIN_TOKENS == {("sm_107a", 212, "f32"): 16384}
    for arch in ARCHES:
        for sms in (148, 152, 160, 212, 216):
            for acc in FWD_ACCS:
                least = cake_jit.CTA1_FWD_MIN_TOKENS.get((arch, sms, acc))
                for T in sorted({1, 4096, 16384, 65536, 2**31 - 1} | ({least - 1, least} if least else set())):
                    want = f"fwd_{acc}_cta1" if least is not None and T >= least else f"fwd_{acc}"
                    assert cake_jit.forward_record(arch, sms, acc, T) == want, (arch, sms, acc, T)
                    assert cake_jit.MODULES[want]["stage"] == "fwd"
    with pytest.raises(ValueError):
        cake_jit.forward_record("sm_107a", 212, "f64", 65536)
    with pytest.raises(ValueError, match=r"unknown acc 'f32' for 'bwd'; it ships \['dot2'\]"):
        cake_jit.record("bwd", "f32")
    for acc in FWD_ACCS:
        assert cake_jit.record("dw_reduce", acc) == "dw_reduce"
    assert cake_jit.record("dw_reduce") == "dw_reduce"
    with pytest.raises(ValueError):
        cake_jit.record("fwd", "f64")
    with pytest.raises(ValueError):
        cake_jit.record("main", "f32")


def test_no_record_is_delivered_outside_the_architectures() -> None:
    for name in cake_jit.MODULES:
        for arch in ARCHES:
            assert cake_backend.library_name(name, arch) == f"{cake_jit.MODULES[name]['library']}_{arch}"
        with pytest.raises(RuntimeError):
            cake_backend.library_name(name, "sm_90a")
    with pytest.raises(KeyError):
        cake_backend.record("bwd_main")


# The measured resident four-CTA clusters of the shipped programs: (architecture, SM count, record) -> clusters.
MEASURED = {
    ("sm_100a", 148, "fwd_dot2"): 142,
    ("sm_100a", 148, "fwd_f32"): 142,
    ("sm_100a", 148, "bwd_dot2"): 104,
    ("sm_100a", 148, "bwd_dot2_stage"): 104,
    ("sm_100a", 148, "bwd_dot2_order"): 104,
    ("sm_100a", 152, "fwd_dot2"): 148,
    ("sm_100a", 152, "fwd_f32"): 148,
    ("sm_100a", 152, "bwd_dot2"): 110,
    ("sm_100a", 152, "bwd_dot2_stage"): 110,
    ("sm_103a", 152, "fwd_dot2"): 148,
    ("sm_103a", 152, "fwd_f32"): 148,
    ("sm_103a", 152, "bwd_dot2"): 110,
    ("sm_103a", 152, "bwd_dot2_stage"): 110,
    ("sm_103a", 152, "bwd_dot2_order"): 110,
    ("sm_107a", 212, "fwd_dot2"): 184,
    ("sm_107a", 212, "fwd_f32"): 184,
    ("sm_107a", 212, "bwd_dot2"): 138,
    ("sm_107a", 212, "bwd_dot2_stage"): 138,
    ("sm_107a", 212, "bwd_dot2_order"): 138,
}
PERSISTENT_RECORDS = ("bwd_dot2", "bwd_dot2_order", "bwd_dot2_stage", "fwd_dot2", "fwd_f32")


def test_the_resident_cluster_table_holds_the_measured_devices() -> None:
    """Every ``fwd`` / ``bwd`` record has its CTAs per SM; the measured devices give their measured clusters; every
    table entry is at most CTAs per SM * SMs / 4 clusters, and the fallback stays below every table entry (a grid
    never exceeds what a measured device holds)."""
    assert sorted(cake_jit.CTAS_PER_SM) == sorted(PERSISTENT_RECORDS)
    assert sorted(name for name, rec in cake_jit.MODULES.items() if rec["stage"] != "dw_reduce") == sorted(
        PERSISTENT_RECORDS + tuple(cake_jit.CTA1_RECORDS)
    )
    for (arch, sms, name), want in MEASURED.items():
        assert cake_jit.resident_clusters(name, arch, sms) == want, (arch, sms, name)
    fraction = cake_jit.UNMEASURED_CLUSTER_FRACTION
    for (arch, sms), table in cake_jit.RESIDENT_CLUSTERS.items():
        assert arch in ARCHES
        for per_sm, held in table.items():
            assert 1 <= held <= per_sm * sms // cake_jit.CLUSTER_CTAS, (arch, sms, per_sm)
            assert int(fraction * per_sm * sms) // cake_jit.CLUSTER_CTAS <= held, (arch, sms, per_sm)
    # A device the table does not hold takes the fallback fraction (a 148-SM sm_103a part: 94 and 125).
    assert cake_jit.resident_clusters("bwd_dot2", "sm_103a", 148) == int(fraction * 3 * 148) // 4 == 94
    assert cake_jit.resident_clusters("fwd_dot2", "sm_103a", 148) == int(fraction * 4 * 148) // 4 == 125
    assert cake_jit.resident_clusters("bwd_dot2", "sm_103a", 160) == int(fraction * 3 * 160) // 4
    assert cake_jit.resident_clusters("fwd_f32", "sm_107a", 1) == 1


@pytest.mark.parametrize("sms", (1, 148, 152, 160, 212))
@pytest.mark.parametrize("T", (1, 2, 3, 129, 2047, 2048, 4096, 16384, 65536, 2**31 - 1))
def test_grids_are_persistent_clusters(T, sms) -> None:
    """``fwd`` and ``bwd`` run ``min(T, resident clusters)`` clusters (at least one) of four CTAs, but a ``bwd`` call of
    at least ``LARGE_BWD_MIN_TOKENS`` tokens on a device of ``LARGE_BWD_CLUSTERS`` runs ``min(T, its entry)``;
    ``dw_reduce`` covers the 20480 weight-gradient columns on any device."""
    for arch in ARCHES:
        for name in PERSISTENT_RECORDS:
            resident = cake_jit.resident_clusters(name, arch, sms)
            large = T >= cake_jit.LARGE_BWD_MIN_TOKENS
            want = resident
            if large and cake_jit.MODULES[name]["stage"] == "bwd":
                want = cake_jit.LARGE_BWD_CLUSTERS.get((arch, sms), resident)
            assert cake_jit.persistent_clusters(name, arch, sms, large) == want
            clusters = cake_jit.clusters(name, T, arch, sms)
            assert 1 <= clusters <= T
            assert clusters == max(1, min(T, want))
            assert resident <= max(1, cake_jit.CTAS_PER_SM[name] * sms // cake_jit.CLUSTER_CTAS)
            assert want <= max(1, cake_jit.CTAS_PER_SM[name] * sms // cake_jit.CLUSTER_CTAS)
            assert cake_jit.grid(name, T, arch, sms) == (cake_jit.CLUSTER_CTAS * clusters, 1, 1)
        assert cake_jit.grid("dw_reduce", T, arch, sms) == cake_jit.dw_reduce_grid()
    assert cake_jit.dw_reduce_grid() == (STREAMS * HIDDEN // cake_jit.DW_REDUCE_COLUMNS_PER_CTA, 1, 1)
    assert STREAMS * HIDDEN % cake_jit.DW_REDUCE_COLUMNS_PER_CTA == 0


def test_b300_backward_of_many_tokens_runs_its_own_grid() -> None:
    """B300 (``sm_103a``, 148 SMs) has no resident entry: below ``LARGE_BWD_MIN_TOKENS`` tokens the backward and the
    forward keep the fallback grids (94 and 125 clusters); from it the backward runs 104 clusters (it holds 104) and
    the forward still 125."""
    assert cake_jit.LARGE_BWD_MIN_TOKENS == 2048
    assert cake_jit.LARGE_BWD_CLUSTERS == {("sm_103a", 148): 104}
    assert ("sm_103a", 148) not in cake_jit.RESIDENT_CLUSTERS
    for name in ("bwd_dot2", "bwd_dot2_stage", "bwd_dot2_order"):
        assert [cake_jit.clusters(name, T, "sm_103a", 148) for T in (1, 94, 129, 2047)] == [1, 94, 94, 94]
        assert [cake_jit.clusters(name, T, "sm_103a", 148) for T in (2048, 4096, 65536, 2**31 - 1)] == [104] * 4
    for name in ("fwd_dot2", "fwd_f32"):
        assert [cake_jit.clusters(name, T, "sm_103a", 148) for T in (129, 2047, 2048, 65536)] == [125] * 4
    for (arch, sms), entry in cake_jit.LARGE_BWD_CLUSTERS.items():
        assert arch in ARCHES
        assert 1 <= entry <= 3 * sms // cake_jit.CLUSTER_CTAS, (arch, sms)


def test_the_launch_grid_and_the_workspace_follow_the_call(monkeypatch) -> None:
    """On a faked device of each measured kind (nothing launched), the launch grid of a call (``_clusters``) and the
    workspace query give ``cake_jit.clusters`` of the call's token count, on both sides of ``LARGE_BWD_MIN_TOKENS``."""
    workspace = cake_backend.cake_engram_gate_det_train_workspace_bytes
    for arch, sms in (("sm_103a", 148), ("sm_100a", 148), ("sm_103a", 152), ("sm_107a", 212)):
        monkeypatch.setattr(cake_backend, "device_arch", lambda index, arch=arch: arch)
        monkeypatch.setattr(cake_backend, "sm_count", lambda index, sms=sms: sms)
        cake_backend._resident.cache_clear()
        try:
            for T in (1, 129, 2047, 2048, 4096, 65536):
                for name in PERSISTENT_RECORDS + tuple(sorted(cake_jit.CTA1_RECORDS)):
                    assert cake_backend._clusters(name, T, 0) == cake_jit.clusters(name, T, arch, sms), (arch, name, T)
                rows = cake_jit.clusters("bwd_dot2", T, arch, sms)
                assert workspace(T, 0) == rows * STREAMS * HIDDEN * 4, (arch, T)
        finally:
            cake_backend._resident.cache_clear()
    monkeypatch.undo()
    cake_backend._resident.cache_clear()


@pytest.mark.parametrize("sms", (1, 148, 152, 160, 212))
@pytest.mark.parametrize("T", (1, 2, 3, 129, 4096, 16384, 65536, 2**31 - 1))
def test_cta1_grids_are_one_cta_an_sm(T, sms) -> None:
    """A ``cta1`` forward runs ``min(T, SMs)`` CTAs (at least one) of 640 threads, one an SM; every other ``fwd`` /
    ``bwd`` record's resident units are its resident four-CTA clusters."""
    for arch in ARCHES:
        for name in sorted(cake_jit.CTA1_RECORDS):
            assert cake_jit.resident_units(name, arch, sms) == sms
            assert cake_jit.clusters(name, T, arch, sms) == max(1, min(T, sms))
            assert cake_jit.grid(name, T, arch, sms) == (max(1, min(T, sms)), 1, 1)
        for name in PERSISTENT_RECORDS:
            assert cake_jit.resident_units(name, arch, sms) == cake_jit.resident_clusters(name, arch, sms)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def entry(self, name: str, arch: str):
        return lambda *args: self.calls.append((name, args))


def _cpu_operands(T: int) -> dict:
    bf16 = dict(dtype=torch.bfloat16)
    return {
        "x": torch.zeros(T, STREAMS, HIDDEN, **bf16),
        "key": torch.zeros(T, STREAMS, HIDDEN, **bf16),
        "value": torch.zeros(T, HIDDEN, **bf16),
        "q_words": torch.zeros(STREAMS, HIDDEN // 2, dtype=torch.int32),
        "k_words": torch.zeros(STREAMS, HIDDEN, dtype=torch.int32),
        "dout": torch.zeros(T, STREAMS, HIDDEN, **bf16),
    }


@pytest.mark.parametrize("masked", (False, True), ids=("no-mask", "mask"))
@pytest.mark.parametrize("acc", FWD_ACCS)
def test_stage_launchers_pass_every_kernel_parameter(acc, masked, monkeypatch) -> None:
    """Each stage launcher passes every parameter of its records' argument plans, in order, then the grid
    (``_entry`` replaced by a recorder; nothing is launched): the forward record of ``acc``, the backward, then
    ``dw_reduce``; the weights as their words (a BF16 ``q_weight``, an FP32 ``k_weight``) with their mode word."""
    recorder = _Recorder()
    monkeypatch.setattr(cake_backend, "_entry", recorder.entry)
    monkeypatch.setattr(cake_backend, "device_arch", lambda index: "sm_100a")
    T, clusters = 5, 2
    ops = _cpu_operands(T)
    mask = torch.ones(T, dtype=torch.bool) if masked else None
    out = torch.empty_like(ops["x"])
    cake_backend.launch_fwd(
        cake_jit.record("fwd", acc),
        x=ops["x"],
        key=ops["key"],
        value=ops["value"],
        q_weight=ops["q_words"],
        k_weight=ops["k_words"],
        wmode=1,
        mask=mask,
        out=out,
        eps=1e-20,
        clamp_value=1e-6,
        clusters=clusters,
    )
    rows = clusters  # one partial row per cluster
    dw = torch.zeros(rows, STREAMS, HIDDEN)
    cake_backend.launch_bwd(
        cake_jit.record("bwd"),
        dout=ops["dout"],
        x=ops["x"],
        key=ops["key"],
        value=ops["value"],
        q_weight=ops["q_words"],
        k_weight=ops["k_words"],
        wmode=1,
        mask=mask,
        dx=torch.empty_like(ops["x"]),
        dkey=torch.empty_like(ops["x"]),
        dvalue=torch.empty_like(ops["value"]),
        dw=dw,
        eps=1e-20,
        clamp_value=1e-6,
        clusters=clusters,
    )
    dq = torch.zeros(STREAMS, HIDDEN // 2, dtype=torch.int32)
    dk = torch.zeros(STREAMS, HIDDEN, dtype=torch.int32)
    cake_backend.launch_dw_reduce(
        "dw_reduce", dw=dw, q_weight=ops["q_words"], k_weight=ops["k_words"], dq=dq, dk=dk, wmode=0b0101
    )
    cta1 = f"fwd_{acc}_cta1"
    if cta1 in cake_jit.CTA1_RECORDS:  # fwd_f32 only: fwd_dot2 has no cta1 form
        cake_backend.launch_fwd(
            cta1,
            x=ops["x"],
            key=ops["key"],
            value=ops["value"],
            q_weight=ops["q_words"],
            k_weight=ops["k_words"],
            wmode=1,
            mask=mask,
            out=out,
            eps=1e-20,
            clamp_value=1e-6,
            clusters=clusters,
        )
        *calls, (cta1_name, cta1_args) = recorder.calls
        recorder.calls = calls
        # the cta1 form takes its cluster program's parameters on ``clusters`` CTAs of 640 threads
        assert cta1_name == cta1 and cta1_args[-3:] == (clusters, 1, 1)
        first = dict(recorder.calls)[f"fwd_{acc}"]
        assert len(cta1_args) == len(first)
        for mine, theirs in zip(cta1_args[:-3], first[:-3], strict=True):
            assert mine is theirs or (not isinstance(mine, torch.Tensor) and mine == theirs)
    assert [name for name, _ in recorder.calls] == [f"fwd_{acc}", "bwd_dot2", "dw_reduce"]
    for name, args in recorder.calls:
        params = cake_backend.parameters(name)
        assert len(args) == len(params) + 3, (name, params)
        grid = args[-3:]
        if name == "dw_reduce":
            assert grid == cake_jit.dw_reduce_grid()
        else:
            assert grid == (cake_jit.CLUSTER_CTAS * clusters, 1, 1)
        named = dict(zip(params, args, strict=False))
        if "T" in named:
            assert named["T"] == T
        if "mask" in named:
            assert named["mask"] is mask
        if name in (f"fwd_{acc}", "bwd_dot2"):
            assert named["q_weight"] is ops["q_words"] and named["k_weight"] is ops["k_words"]
            assert named["wmode"] == 1
        if name == "dw_reduce":
            assert named["dq"] is dq and named["dk"] is dk
            assert (named["wmode"], named["NP"]) == (0b0101, rows)


@pytest.mark.parametrize("T", (0, 1, 2, 3))
@pytest.mark.parametrize("layout", ("contiguous", "wider"))
def test_the_kv_views_are_the_slices(T, layout) -> None:
    """The backend's ``key`` and ``value`` views of a packed ``kv`` address what the slices
    ``kv[:, :20480].unflatten(1, (4, 5120))`` and ``kv[:, 20480:]`` address: the same storage, offsets and sizes and the
    same strides on every dimension with more than one element, for a contiguous ``kv`` and for a view into a wider
    buffer. Both views carry ``kv``'s row stride; a dimension of size 0 or 1 addresses no second element, and torch's
    view rules may give the sliced key another stride there."""
    width = (STREAMS + 1) * HIDDEN
    if layout == "contiguous":
        kv = torch.zeros(T, width, dtype=torch.bfloat16)
    else:
        kv = torch.zeros(T, width + 24, dtype=torch.bfloat16)[:, 8 : 8 + width]
    sliced = (kv[:, : STREAMS * HIDDEN].unflatten(1, (STREAMS, HIDDEN)), kv[:, STREAMS * HIDDEN :])
    made = (*cake_backend._split_kv(kv), *cake_backend._kv_views(kv))
    for got, want in zip(made, sliced * 2, strict=True):
        assert got.shape == want.shape and got.stride(0) == kv.stride(0), (tuple(got.shape), got.stride())
        for dim, size in enumerate(want.shape):
            if size > 1:
                assert got.stride(dim) == want.stride(dim), (dim, got.stride(), want.stride())
        assert got.storage_offset() == want.storage_offset()
        assert got.untyped_storage().data_ptr() == want.untyped_storage().data_ptr()


def test_the_package_exports_the_backend_entries() -> None:
    for name in ENTRIES:
        assert getattr(pkg, name) is getattr(cake_backend, name), name
        assert name in pkg.__all__ and name in cake_backend.__all__, name


def test_manifest_covers_every_record_and_its_sources() -> None:
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]}
    assert set(entries) == wanted
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert name.endswith("_" + entry["arch"]), name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    sm107 = [entry for entry in manifest["libraries"] if entry["arch"] == "sm_107a"]
    assert all(entry["gencode"] == "-gencode=arch=compute_107a,code=sm_107a" for entry in sm107)


def test_the_loader_imports_standalone_by_path() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a fresh process reads the manifest."""
    _manifest_or_skip()
    loader = str(PACKAGE_DIR / "cake_build.py")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {loader!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    libraries = len({f"{rec['library']}_{arch}" for rec in cake_jit.MODULES.values() for arch in rec["arches"]})
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {libraries}"


def test_the_prebuilt_library_is_preferred() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path."""
    _manifest_or_skip()
    cake_build = cake_package(PACKAGE, "cake_build")
    for name, rec in cake_jit.MODULES.items():
        for arch in rec["arches"]:
            library = cake_backend.library_name(name, arch)
            assert cake_build.prebuilt_path(library) == PACKAGE_DIR / "aot" / f"{library}.so"
            assert cake_backend.is_prebuilt(name, arch) == cake_build.prebuilt_path(library).is_file()


# The variables the JIT shim (``_cake_pplx_jit.py``) reads: the toolchain, cache and JIT switches.
JIT_SHIM_ENVIRONMENT_VARIABLES = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CAKE_PPLX_CACHE_DIR",
        "CAKE_PPLX_NVCC",
        "CXX",
        "CAKE_PPLX_JIT_WORKERS",
        "CAKE_PPLX_DISABLE_JIT",
    }
)


def _environment_reads(tree: ast.AST) -> tuple[set[str], list[int]]:
    """``(names, unresolved)``: the variable names ``tree`` reads through ``os.environ.get(...)``,
    ``os.environ[...]``, ``... in os.environ`` or ``os.getenv(...)`` by a string literal or a loop variable over a
    tuple / list of string literals, and the lines of every other ``environ`` / ``getenv`` use."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def keys(expr: ast.AST, at: ast.AST):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.Name):
            node = parents.get(at)
            while node is not None:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == expr.id
                    and isinstance(node.iter, (ast.Tuple, ast.List))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)
                ):
                    return {e.value for e in node.iter.elts}
                node = parents.get(node)
        return None

    names: set[str] = set()
    unresolved: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")):
            continue
        parent = parents.get(node)
        found = None
        if node.attr == "getenv":
            if isinstance(parent, ast.Call) and parent.func is node and parent.args:
                found = keys(parent.args[0], parent)
        elif isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and call.args:
                found = keys(call.args[0], call)
        elif isinstance(parent, ast.Subscript) and parent.value is node:
            found = keys(parent.slice, parent)
        elif (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is node
        ):
            found = keys(parent.left, parent)
        if found is None:
            unresolved.append(node.lineno)
        else:
            names |= found
    return names, unresolved


def test_the_package_reads_no_environment_variable() -> None:
    """No module of the package reads an environment variable, except the JIT shim, which reads exactly
    :data:`JIT_SHIM_ENVIRONMENT_VARIABLES`."""
    offenders = []
    shim = None
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        names, unresolved = _environment_reads(ast.parse(path.read_text()))
        if path.name == "_cake_pplx_jit.py":
            shim = names
            offenders += [f"{path.name}:{line} (variable not resolved)" for line in unresolved]
        else:
            offenders += [f"{path.name}:{line}" for line in unresolved]
            offenders += [f"{path.name}: {name}" for name in sorted(names)]
    assert not offenders, offenders
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107


def test_every_record_loads(device) -> None:
    arch = cake_backend.device_arch(device.index)
    for name in cake_jit.MODULES:
        assert callable(cake_backend.load(name, arch).run), name


# (T, weights, mask, layout): one token, two, a token count past one wave of clusters, an odd count.
CASES = (
    (1, "fp32", "none", "contiguous"),
    (2, "bf16", "some", "kv"),
    (129, "mixed", "some", "kv"),
    (1031, "fp32", "none", "contiguous"),
)
CASE_IDS = [f"T{T}-{w}-{m}-{lay}" for T, w, m, lay in CASES]


def _record_failures(judge, failures, record_property) -> None:
    judge.record(record_property)
    assert not failures, failures


@pytest.mark.parametrize("acc", FWD_ACCS)
@pytest.mark.parametrize(("T", "weights", "mask", "layout"), CASES, ids=CASE_IDS)
def test_the_forward_record_matches_the_reference(acc, T, weights, mask, layout, device, record_property) -> None:
    """The forward record of ``acc`` launched directly writes every element of a NaN-filled ``out`` and meets the
    tier."""
    name, arch = cake_jit.record("fwd", acc), cake_backend.device_arch(device.index)

    def run(ns):
        out = torch.full_like(ns.x, math.nan)
        cake_backend.launch_fwd(
            name,
            x=ns.x,
            key=ns.key,
            value=ns.value,
            q_weight=cake_backend._words(ns.q_weight),
            k_weight=cake_backend._words(ns.k_weight),
            wmode=cake_backend._wmode(ns.q_weight.dtype, ns.k_weight.dtype),
            mask=ns.token_mask,
            out=out,
            eps=1e-20,
            clamp_value=1e-6,
            clusters=cake_jit.clusters(name, T, arch, cake_backend.sm_count(device.index)),
        )
        torch.cuda.synchronize(device)
        return {"out": out}

    def make(seed):
        return make_inputs(T, seed=seed, device=device, weights=weights, mask=mask, layout=layout)

    _record_failures(*judge_draws(run, make, seed=11 + T), record_property)


@pytest.mark.parametrize(("T", "weights", "mask", "layout"), CASES, ids=CASE_IDS)
def test_the_backward_matches_the_reference(T, weights, mask, layout, device, record_property) -> None:
    """The backward (``bwd_dot2`` then ``dw_reduce``), with NaN-filled ``dx``, ``dkey`` and ``dvalue``, meets the
    tier for every gradient."""

    def run(ns):
        dx = torch.full_like(ns.x, math.nan)
        dkey = torch.full_like(ns.x, math.nan)
        dvalue = torch.full_like(ns.value.contiguous(), math.nan)
        got = cake_backend.cake_engram_gate_det_train_backward(
            ns.dout,
            ns.x,
            ns.key,
            ns.value,
            ns.q_weight,
            ns.k_weight,
            ns.token_mask,
            dx=dx,
            dkey=dkey,
            dvalue=dvalue,
        )
        torch.cuda.synchronize(device)
        return dict(zip(("dx", "dkey", "dvalue", "dq", "dk"), got, strict=True))

    def make(seed):
        return make_inputs(T, seed=seed, device=device, weights=weights, mask=mask, layout=layout)

    _record_failures(*judge_draws(run, make, seed=23 + T), record_property)


@pytest.mark.parametrize("weights", ("fp32", "bf16"))
@pytest.mark.parametrize("parts", (1, 7, 296))
def test_dw_reduce_matches_the_reference(parts, weights, device, record_property) -> None:
    """``dw_reduce`` alone: ``dq = (sum_p dw[p]) * k_weight`` and ``dk = (sum_p dw[p]) * q_weight`` (FP32) from random
    partial rows and the weights stored as ``weights`` meet the tier (BF16 tier: the rows, weights and sum in BF16)."""
    from tests.cake_naive import judge_case

    cache = {}
    dtype = torch.bfloat16 if weights == "bf16" else torch.float32

    def run(j, s):
        gen = torch.Generator(device=device).manual_seed(s)
        dw = torch.randn(parts, STREAMS, HIDDEN, device=device, generator=gen)
        qw = (1.0 + 0.1 * torch.randn(STREAMS, HIDDEN, device=device, generator=gen)).to(dtype)
        kw = (1.0 + 0.1 * torch.randn(STREAMS, HIDDEN, device=device, generator=gen)).to(dtype)
        dq = torch.full((STREAMS, HIDDEN), math.nan, device=device)
        dk = torch.full_like(dq, math.nan)
        words = [t.view(torch.int32) for t in (qw, kw, dq, dk)]
        mode = cake_backend._wmode(qw.dtype, kw.dtype)
        cake_backend.launch_dw_reduce(
            "dw_reduce", dw=dw, q_weight=words[0], k_weight=words[1], dq=words[2], dk=words[3], wmode=mode
        )
        torch.cuda.synchronize(device)
        cache[j] = (dw, qw, kw)
        total = dw.double().sum(0)
        return {"dq": dq, "dk": dk}, {"dq": total * kw.double(), "dk": total * qw.double()}

    def naive(j, tier):
        dw, qw, kw = cache.pop(j)
        total = dw.bfloat16().sum(0)
        return {"dq": total * kw.bfloat16(), "dk": total * qw.bfloat16()}

    judge, failures = judge_case(run, naive, seed=31 + parts)
    _record_failures(judge, failures, record_property)


# The FP32 patterns of the rounding test: every upper half (each sign, exponent and leading mantissa: zeros,
# subnormals, the largest finite values, infinities and NaNs), each with these lower halves (zero, the smallest, the
# largest below the tie, the tie, past it, the largest, two more) and random ones up to 256.
_LOWER_FIXED = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0x4000, 0xC000)
_LOWER_HALVES = 256


def _fp32_patterns(device: torch.device, row: int) -> torch.Tensor:
    """The 2^24 FP32 patterns as int32 words, then the first ones again up to a whole number of ``row`` words."""
    generator = torch.Generator().manual_seed(5)
    random = torch.randint(0, 1 << 16, (_LOWER_HALVES - len(_LOWER_FIXED),), generator=generator)
    lower = torch.cat((torch.tensor(_LOWER_FIXED), random)).to(device=device, dtype=torch.int64)
    upper = torch.arange(1 << 16, device=device, dtype=torch.int64)
    bits = ((upper.view(-1, 1) << 16) | lower.view(1, -1)).flatten()
    bits = torch.cat((bits, bits[: -bits.numel() % row]))
    return torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32)


@pytest.mark.parametrize("weights", ("fp32", "bf16", "mixed"))
def test_bf16_weight_gradients_round_as_torch(weights, device) -> None:
    """``dw_reduce`` with unit weights (stored as ``weights`` says) and one partial row of FP32 patterns: its BF16
    ``dq_weight`` and ``dk_weight`` (mode bits 2 and 3, together and alone) hold the bits of its FP32 ones (both bits
    clear) rounded by torch's ``.to(torch.bfloat16)``, the cast the autograd entries applied before, on every
    pattern of 2^24 (ties, NaN, infinities, subnormals, overflow)."""
    q_dtype = torch.float32 if weights == "fp32" else torch.bfloat16
    k_dtype = torch.bfloat16 if weights == "bf16" else torch.float32
    qw = torch.ones(STREAMS, HIDDEN, dtype=q_dtype, device=device).view(torch.int32)
    kw = torch.ones(STREAMS, HIDDEN, dtype=k_dtype, device=device).view(torch.int32)
    mode = cake_backend._wmode(q_dtype, k_dtype)
    dw = torch.empty(1, STREAMS, HIDDEN, device=device)
    row = dw.view(torch.int32).flatten()
    patterns = _fp32_patterns(device, row.numel())
    f32 = torch.empty(2, STREAMS, HIDDEN, device=device)
    bf = torch.empty(2, STREAMS, HIDDEN, dtype=torch.bfloat16, device=device)
    words = {False: f32.view(torch.int32), True: bf.view(torch.int32)}
    mismatches = {grads: torch.zeros((), dtype=torch.int64, device=device) for grads in (0b0100, 0b1000, 0b1100)}
    for at in range(0, patterns.numel(), row.numel()):
        row.copy_(patterns[at : at + row.numel()])
        cake_backend.launch_dw_reduce(
            "dw_reduce", dw=dw, q_weight=qw, k_weight=kw, dq=words[False][0], dk=words[False][1], wmode=mode
        )
        expected = f32.to(torch.bfloat16).view(torch.int16)
        for grads, count in mismatches.items():
            bf.fill_(math.nan)
            dq_bf16, dk_bf16 = bool(grads & 0b0100), bool(grads & 0b1000)
            cake_backend.launch_dw_reduce(
                "dw_reduce",
                dw=dw,
                q_weight=qw,
                k_weight=kw,
                dq=words[dq_bf16][0],
                dk=words[dk_bf16][1],
                wmode=mode | grads,
            )
            got = bf.view(torch.int16)
            if dq_bf16:
                count += (got[0] != expected[0]).sum()
            if dk_bf16:
                count += (got[1] != expected[1]).sum()
    torch.cuda.synchronize(device)
    assert patterns.numel() >= 1 << 24
    assert {grads: int(count) for grads, count in mismatches.items()} == {0b0100: 0, 0b1000: 0, 0b1100: 0}
