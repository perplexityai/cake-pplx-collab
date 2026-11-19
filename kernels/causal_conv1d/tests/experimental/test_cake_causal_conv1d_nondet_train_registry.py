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

The single-rank registry (``cake_jit``) of the non-deterministic causal convolution training package against the
deterministic package's: the forward records are the deterministic forward programs (only their names and the
per-call programmatic dependent launch scalar differ), every backward route keeps the deterministic route's programs
and selection (but where a channel-band program takes the lowest channel counts of its twin, the program of the route
it shares a body with, and at the rows of ``FALLBACK_ROWS``, where the route's fallback runs), every program but the
opt-in pair routes' takes the per-call PDL scalar (its kernels wait for the preceding grid before touching global
memory; the host's policy tables, and the backward rows of ``PDL_OFF_ROWS`` that launch without it, which the
backward entry finds by passing its route, ``C`` and ``S``), each backward program is ONE
ordinary launch (no reduce program) that owns an ``accumulators`` workspace (one FP32 word per sum and accumulator
copy on every route, :data:`ACC_SLOTS` copies; ``[C, 4]`` taps then ``[C]`` bias words; uint32 ticket counters per
channel tile), and the workspace byte
layout and size depend on ``C`` and the route only.  The public names (each carries the mode word ``nondet``; no
entry point takes a mode keyword), and the package's wiring (test-helper key, package data, no ``__init__.py`` at
the ``cake_pplx`` / ``cake_pplx._backends`` namespace levels).
No device is needed; nothing is built or launched.  The ``build.json`` checks skip until the exporter renders the
package.
"""

from __future__ import annotations

import filecmp
import hashlib
import importlib.util
import inspect
import itertools
import os
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

KEY = "cake_causal_conv1d_nondet_train"
pkg = cake_package(KEY)
cake_build = cake_package(KEY, "cake_build")
cake_jit = cake_package(KEY, "cake_jit")
kcp_jit = cake_package(KEY, "kcp_jit")
det_build = cake_package("cake_causal_conv1d_det_train", "cake_build")
det_jit = cake_package("cake_causal_conv1d_det_train", "cake_jit")

PACKAGE_DIR = Path(cake_jit.__file__).parent
REPO = Path(__file__).resolve().parents[2]
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
NAME_FIELDS = ("kernel", "kernel_symbol", "library")
# A placeholder name until the exporter renders the package, then the identity rule's prefix + 20 hex digits.
PROGRAM_NAME = re.compile(
    r"cake_causal_conv1d_train_(?:TODOEXPORT_[a-z0-9_]+|[0-9a-f]{20})"
)
CHANNELS = (
    8,
    128,
    136,
    256,
    264,
    384,
    504,
    512,
    520,
    576,
    640,
    704,
    760,
    768,
    776,
    1016,
    1024,
    1032,
    1160,
    1280,
    1344,
    1536,
    1600,
    1920,
    2048,
    2560,
    4096,
    8192,
)
# 262143 / 262144: both sides of the sm_107a token-floored backward program's floor; S_max and S_max + 1 of every
# ``FALLBACK_ROWS`` key (:data:`FALLBACK_ROWS_OF_RECORD`).
TOKENS = (
    1,
    1000,
    31292,
    31293,
    62585,
    62586,
    65536,
    98304,
    125170,
    125171,
    131072,
    131073,
    147456,
    250340,
    250341,
    262143,
    262144,
    262145,
)
# The rows where a backward route runs its fallback program in place of the program its selection rules pick:
# arch -> (route, C_lo, C_hi) -> (S_min, S_max), every multiple of 8 from C_lo to C_hi.  On sm_100a and sm_103a the
# ``bwd`` route's C 520 to 1016 run the route's fallback (the stage_pair body) up to the last measured token count at
# which no backward, training-step or checkpointed-step row was slower than the program the thresholds pick.
FALLBACK_ROWS_OF_RECORD = {
    "sm_100a": {
        ("bwd", 520, 760): (1, 131072),
        ("bwd", 768, 768): (1, 262144),
        ("bwd", 776, 1016): (1, 62585),
    },
    "sm_103a": {
        ("bwd", 520, 760): (1, 31292),
        ("bwd", 768, 768): (1, 250340),
        ("bwd", 776, 1016): (1, 125170),
    },
}
# The programs whose rows the fallback takes inside :data:`FALLBACK_ROWS_OF_RECORD` (named by program, not by
# route position): the ``bwd`` program the channel thresholds pick at C 520 to 1016 on each architecture.
FALLBACK_REPLACES_OF_RECORD = {
    "sm_100a": frozenset({"cake_causal_conv1d_train_461d358d7090642c65de"}),
    "sm_103a": frozenset(
        {
            "cake_causal_conv1d_train_461d358d7090642c65de",
            "cake_causal_conv1d_train_efd2e6f9ac3d58ab92ab",
        }
    ),
}
# The deterministic package's programs this package does not have: the row-group backward program of route ``bwd``
# on sm_107a (264 <= C < 512) and the SM-wide row-group programs of route ``bwd`` on sm_100a, sm_103a and sm_107a
# (channel bands of 264 <= C < 512, from a token floor), and the copy of route ``bwd``'s fallback program on sm_107a
# (the channel band of 520 <= C < 1024).  Every other record pairs with this package's at the same route position.
# This package carries its own channel-band copy of its route's fallback for the last of them (:data:`OWN_COPIES`).
DET_ONLY = frozenset(
    {
        "cake_causal_conv1d_det_train_50aa818e2b2338fb34a4",
        "cake_causal_conv1d_det_train_e057357c655024d4ff1d",
        "cake_causal_conv1d_det_train_baca4c969c783d07d81f",
        "cake_causal_conv1d_det_train_5ca7515dd2118dd96241",
        "cake_causal_conv1d_det_train_0d99ef0e0ed10b11dfd5",
    }
)
# The deterministic channel-band programs of :data:`DET_ONLY` whose rows this package runs on its own channel-band copy
# of the same route's fallback (the fallback's build keywords with the same ``c_band``, under this package's name):
# deterministic program -> this package's copy.  The copy is listed after the deterministic route's programs, as a
# channel-band program, but its twin (the fallback) keeps no channel threshold to give it.
OWN_COPIES = {
    "cake_causal_conv1d_det_train_0d99ef0e0ed10b11dfd5": "cake_causal_conv1d_train_e5d6d187103ab6e0cbbb",
}
# The route whose ``TILE_OVERRIDES`` / ``NARROW_TILES`` rows each ``_bf16_taps`` route plans by: its base route, whose
# body it runs with SiLU' at the unrounded FP32 preactivation; every other route plans by its own rows.
TILE_ROUTE = {"bwd_dt_bf16_taps": "bwd_dt", "bwd_bias_bf16_taps": "bwd_bias"}


@contextmanager
def warnings_recorded():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield caught


def _strip(record: dict, *fields: str) -> dict:
    return {key: value for key, value in record.items() if key not in fields}


def _args(record: dict) -> set[str]:
    return {name for kind, name in record["arg_plan"] if kind != "grid"}


# The launch fields of a program with the per-call programmatic dependent launch scalar (``launch.use_pdl`` of
# ``"runtime"``): its binding takes ``enable_pdl`` (0 or 1) as its last host parameter, before the grid.
PDL_FIELDS = ("launch", "arg_plan", "logical")
PDL_ARG = ["launch", "enable_pdl"]
# The single-rank programs that take the scalar: every record but the opt-in pair routes' (9 forward, 30 backward).
RUNTIME_PDL_PROGRAMS = 39


def _pdl_program_of(det: dict) -> dict:
    """The launch fields a program with the per-call PDL scalar takes from the deterministic record ``det`` it
    mirrors (launched without PDL): ``use_pdl`` ``"runtime"``, ``("launch", "enable_pdl")`` before the three grid
    entries of the arguments, and the logical name ``enable_pdl``."""

    assert det["launch"]["use_pdl"] is False and PDL_ARG not in det["arg_plan"]
    assert [kind for kind, _ in det["arg_plan"][-3:]] == ["grid", "grid", "grid"]
    return {
        "launch": {**det["launch"], "use_pdl": "runtime"},
        "arg_plan": [*det["arg_plan"][:-3], PDL_ARG, *det["arg_plan"][-3:]],
        "logical": {**det["logical"], "enable_pdl": "enable_pdl"},
    }


def _planning_routes(arch: str, route: str) -> tuple[str, ...]:
    """``route`` and the backward routes of ``arch`` that plan by its tile rows (:data:`TILE_ROUTE`)."""

    return (
        route,
        *(
            twin
            for twin, base in sorted(TILE_ROUTE.items())
            if base == route and twin in cake_jit.backward_kinds(arch)
        ),
    )


def _det_route(arch: str, kind: str) -> list[str]:
    """The deterministic route of ``(arch, kind)`` without its :data:`DET_ONLY` programs."""

    return [name for name in det_jit.ROUTES[arch][kind] if name not in DET_ONLY]


def _pairs(role: str):
    """``(arch, kind, new record name, det record name)`` of every route of ``role``, in route order: the
    deterministic route's programs, position for position (the route's channel-band programs follow them, see
    :func:`_bands`)."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            if cake_jit.route_role(kind) != role:
                continue
            new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
            assert len(new) >= len(det), (arch, kind)
            for name, det_name in zip(new[: len(det)], det, strict=True):
                yield arch, kind, name, det_name


def _channels(registry, name: str, arch: str) -> int | None:
    """The lowest channel threshold of a program's rule on ``arch``: a deterministic program the det-only programs
    split into several channel bands runs here from the first of them."""

    rule = registry.select_rule(registry.MODULES[name])
    value = None if rule is None else rule["channels"].get(arch)
    return value[0] if isinstance(value, list) else value


def _bands():
    """``(arch, kind, band name, twin name, det twin name)`` of every channel-band program: a program a route lists
    after the deterministic route's programs (but this package's copies of deterministic programs,
    :data:`OWN_COPIES`).  Its twin is the program of the route whose channel threshold on ``arch`` is no longer the
    deterministic one: the band program owns the twin's lowest channel counts there, from the deterministic threshold
    up to the twin's new one."""

    for arch in ARCHES:
        for kind in cake_jit.KINDS:
            new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
            moved = [
                (name, det_name)
                for name, det_name in zip(new[: len(det)], det, strict=True)
                if _channels(cake_jit, name, arch) != _channels(det_jit, det_name, arch)
            ]
            for band in new[len(det) :]:
                if band in OWN_COPIES.values():
                    continue
                (twin,) = [
                    pair
                    for pair in moved
                    if _channels(det_jit, pair[1], arch)
                    == _channels(cake_jit, band, arch)
                ]
                moved.remove(twin)
                yield arch, kind, band, *twin
            assert not moved, (arch, kind, moved)


@pytest.mark.cpu_only
def test_loader_is_the_det_loader() -> None:
    """``cake_build.py`` is byte for byte the deterministic package's (one manifest-driven loader)."""

    assert filecmp.cmp(
        Path(cake_build.__file__), Path(det_build.__file__), shallow=False
    )


@pytest.mark.cpu_only
def test_routes_mirror_the_deterministic_package() -> None:
    assert cake_jit.ARCHES == det_jit.ARCHES == ARCHES
    assert cake_jit.KINDS == det_jit.KINDS
    assert cake_jit.REQUIRED_KINDS == det_jit.REQUIRED_KINDS
    assert cake_jit.ROLES == ("fwd", "bwd")
    assert cake_jit.GRID_KINDS == ("co_resident", "occupancy")
    for arch in ARCHES:
        assert set(cake_jit.ROUTES[arch]) == set(det_jit.ROUTES[arch]), arch
    # The host's argument -> route map is the deterministic package's, over every combination.
    for bias, fp32, taps, dweight in itertools.product((False, True), repeat=4):
        args = dict(bias=bias, fp32_grad=fp32, bf16_taps=taps, bf16_dweight=dweight)
        assert cake_jit.backward_route(**args) == det_jit.backward_route(**args), args
    for bias, taps in itertools.product((False, True), repeat=2):
        args = dict(bias=bias, bf16_taps=taps)
        assert cake_jit.forward_route(**args) == det_jit.forward_route(**args), args
    routed = {
        name
        for by_kind in cake_jit.ROUTES.values()
        for names in by_kind.values()
        for name in names
    }
    # No reduce program: every record is a routed forward or backward program.
    assert set(cake_jit.MODULES) == routed
    assert not set(cake_jit.MODULES) & set(det_jit.MODULES)
    for name, record in cake_jit.MODULES.items():
        assert PROGRAM_NAME.fullmatch(name), name
        assert record["library"] == record["kernel"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["role"] in cake_jit.ROLES and "reduce" not in record, name
        assert record["grid_rule"]["kind"] in cake_jit.GRID_KINDS, name
        assert record["grid_rule"]["cooperative"] is False, name
        assert record["launch"]["cooperative"] is False, name


@pytest.mark.cpu_only
def test_forward_records_are_the_deterministic_forward_records() -> None:
    """The forward programs are the deterministic ones (same module, build keywords, plan): ``y`` keeps its bits.
    Only the three name fields and the launch's programmatic dependent launch differ: the deterministic record
    launches without it, this package's takes the per-call ``enable_pdl`` scalar (:func:`_pdl_program_of`)."""

    for arch, kind, name, det_name in _pairs("fwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        label = (arch, kind, name)
        assert _strip(record, *NAME_FIELDS, *PDL_FIELDS) == _strip(det, *NAME_FIELDS, *PDL_FIELDS), label
        assert {field: record[field] for field in PDL_FIELDS} == _pdl_program_of(det), label


@pytest.mark.cpu_only
def test_backward_records_are_one_launch_with_accumulators() -> None:
    """Every backward program replaces the deterministic program + reduce pair of the same route position: the
    deterministic arguments without ``partial``, plus ``dw_acc`` / ``work`` and the gradient outs it now stores
    (``dw``, and ``db`` on the bias routes); the same build geometry, block, cluster, architectures and selection
    rule; an ordinary occupancy-sized launch with the per-call programmatic dependent launch scalar ``enable_pdl``
    (last before the grid; the opt-in pair routes' programs launch without PDL); an ``accumulators`` workspace of the
    route's dtype and tap count."""

    vocabulary = set(cake_jit.LOGICAL_TENSORS) | set(cake_jit.LOGICAL_SCALARS)
    twins = {twin: band_arch for band_arch, _, _, twin, _ in _bands()}
    for arch, kind, name, det_name in _pairs("bwd"):
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        bias = "bias" in kind
        label = (arch, kind, name)
        assert record["role"] == "bwd" and record["arches"] == det["arches"], label
        assert record["ffi_entry"] == det["ffi_entry"], label
        runtime = kind not in cake_jit.PAIR_ROUTES
        expected = (
            (_args(det) - {"partial"})
            | {"dw_acc", "work", "dw"}
            | ({"db"} if bias else set())
            | ({"enable_pdl"} if runtime else set())
        )
        assert _args(record) == expected, label
        assert [k for k, _ in record["arg_plan"][-3:]] == ["grid", "grid", "grid"], (
            label
        )
        assert (record["arg_plan"][-4] == PDL_ARG) is runtime, label
        assert set(record["logical"]) == _args(record), label
        assert set(record["logical"].values()) <= vocabulary, label
        assert (
            record["logical"].get("x_map", "x") == "x"
            and record["logical"].get("dy_map", "dy") == "dy"
        )
        # The global accumulators: kernel parameter ``dw_acc`` (the det fragments' ``acc`` is a register), logical ``acc``.
        assert (
            record["logical"]["dw_acc"] == "acc" and record["logical"]["work"] == "work"
        ), label
        assert ("flags" in _args(record)) is (kind != "bwd"), label
        grid, det_grid = record["grid_rule"], det["grid_rule"]
        assert grid["kind"] == "occupancy" and grid["cooperative"] is False, label
        assert grid["geometry"] == det_grid["geometry"], label
        assert set(grid["grid_mult"]) == set(record["arches"]), label
        # ``ctas_per_sm`` is the rendered program's occupancy (the exporter's leaf).
        assert isinstance(grid["ctas_per_sm"], int) and grid["ctas_per_sm"] >= 1, label
        launch = record["launch"]
        assert launch["cooperative"] is False, label
        assert launch["use_pdl"] == ("runtime" if runtime else False), label
        assert launch["block"] == det["launch"]["block"], label
        assert launch["cluster"] == det["launch"]["cluster"], label
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        # A twin keeps the deterministic rule but on the architecture of its channel-band program.
        det_rule = det_jit.select_rule(det)
        if det_rule is not None:
            channels = {
                a: _channels(det_jit, det_name, a) for a in det_rule["channels"]
            }
            det_rule = {**det_rule, "channels": channels}
        if name in twins:
            det_rule["channels"][twins[name]] = _channels(cake_jit, name, twins[name])
        assert cake_jit.select_rule(record) == det_rule, label
        rule = dict(record["workspace_rule"])
        assert "slots" in rule, label  # required: the package never defaults it
        assert rule == {
            "acc": "float32",
            "slots": ACC_SLOTS[kind],
            "kind": "accumulators",
            "taps": 5 if bias else 4,
            "work": "uint32",
        }, label


@pytest.mark.cpu_only
def test_channel_band_programs_take_their_twins_lowest_channels() -> None:
    """A channel-band program ships for one architecture: its record is its twin's under its own name, architecture
    and selection rule (the rendered occupancy and shared memory aside).  It owns the calls from the deterministic
    program's channel threshold, and the twin those from a higher one, so the route still runs one program per
    ``(S, C)``."""

    bands = list(_bands())
    assert {(arch, kind) for arch, kind, *_ in bands} == {
        ("sm_107a", "bwd"),
        ("sm_107a", "bwd_bias"),
        ("sm_107a", "bwd_fp32g"),
        ("sm_107a", "bwd_bias_bf16_taps"),
    }
    for arch, kind, band, twin, det_twin in bands:
        record, twin_record = cake_jit.MODULES[band], cake_jit.MODULES[twin]
        label = (arch, kind, band)
        assert band not in det_jit.MODULES and record["arches"] == [arch], label
        assert record["library"] == record["kernel"] == band, label
        assert record["kernel_symbol"] == f"kernel_{band}", label
        threshold = _channels(cake_jit, band, arch)
        assert threshold == _channels(det_jit, det_twin, arch), label
        assert threshold < _channels(cake_jit, twin, arch), label
        assert cake_jit.select_rule(record) == {
            "channels": {arch: threshold},
            "kind": "channels_at_least",
        }, label
        grid, twin_grid = record["grid_rule"], twin_record["grid_rule"]
        assert grid["grid_mult"] == {arch: twin_grid["grid_mult"][arch]}, label
        derived = ("grid_mult", "ctas_per_sm")
        assert _strip(grid, *derived) == _strip(twin_grid, *derived), label
        launch = _strip(record["launch"], "dynamic_smem_bytes")
        assert launch == _strip(twin_record["launch"], "dynamic_smem_bytes"), label
        own = ("arches", "select_rule", "grid_rule", "launch", *NAME_FIELDS)
        assert _strip(record, *own) == _strip(twin_record, *own), label


@pytest.mark.cpu_only
def test_own_copies_take_the_deterministic_band_rows() -> None:
    """Each copy of :data:`OWN_COPIES` is a channel-band program of the deterministic program's route and
    architecture: listed after the deterministic route's programs, its record the route's fallback's under its own
    name and selection rule (the rendered occupancy and shared memory aside), which owns the calls from the
    deterministic program's lowest channel threshold."""

    for det_name, copy in OWN_COPIES.items():
        det_record = det_jit.MODULES[det_name]
        (arch,) = det_record["arches"]
        (kind,) = [k for k, names in det_jit.ROUTES[arch].items() if det_name in names]
        new = cake_jit.ROUTES[arch][kind]
        label = (arch, kind, copy)
        assert copy in new[len(_det_route(arch, kind)) :], label
        (fallback,) = [n for n in new if cake_jit.select_rule(cake_jit.MODULES[n]) is None]
        record, twin_record = cake_jit.MODULES[copy], cake_jit.MODULES[fallback]
        assert copy not in det_jit.MODULES and record["arches"] == [arch], label
        assert record["library"] == record["kernel"] == copy, label
        assert record["kernel_symbol"] == f"kernel_{copy}", label
        assert cake_jit.select_rule(twin_record) is None, label
        assert cake_jit.select_rule(record) == {
            "channels": {arch: _channels(det_jit, det_name, arch)},
            "kind": "channels_at_least",
        }, label
        grid, twin_grid = record["grid_rule"], twin_record["grid_rule"]
        derived = ("grid_mult", "ctas_per_sm")
        assert grid["grid_mult"] == {arch: twin_grid["grid_mult"][arch]}, label
        assert _strip(grid, *derived) == _strip(twin_grid, *derived), label
        launch = _strip(record["launch"], "dynamic_smem_bytes")
        assert launch == _strip(twin_record["launch"], "dynamic_smem_bytes"), label
        own = ("arches", "select_rule", "grid_rule", "launch", *NAME_FIELDS)
        assert _strip(record, *own) == _strip(twin_record, *own), label


# The host's programmatic dependent launch policy per architecture: on for every call, outside and inside a CUDA Graph
# capture.
PDL_EAGER_OF_RECORD = {"sm_100a": True, "sm_103a": True, "sm_107a": True}
PDL_GRAPH_OF_RECORD = {"sm_100a": True, "sm_103a": True, "sm_107a": True}
# The backward rows that launch without programmatic dependent launch in both modes, whatever the two tables say
# (``PDL_OFF_ROWS``): arch -> (route, C_lo, C_hi) -> (S_min, S_max), every multiple of 8 from C_lo to C_hi.  On sm_100a
# and sm_103a the ``bwd`` route at its largest channel counts below 1024, each key from the first measured token count
# from which the backward without the attribute is not slower than with it in the training step and the checkpointed
# step, eager and graph, up to every token count the package takes.
PDL_OFF_ROWS_OF_RECORD = {
    "sm_100a": {
        ("bwd", 1000, 1008): (187755, 2147483647),
        ("bwd", 1016, 1016): (155648, 2147483647),
    },
    "sm_103a": {
        ("bwd", 1000, 1008): (131073, 2147483647),
        ("bwd", 1016, 1016): (125171, 2147483647),
    },
}
# The program the calls of the :data:`PDL_OFF_ROWS_OF_RECORD` rows launch (the one the rows were measured with): the
# ``bwd`` program the channel thresholds pick there on both architectures.
PDL_OFF_PROGRAMS_OF_RECORD = {
    "sm_100a": frozenset({"cake_causal_conv1d_train_461d358d7090642c65de"}),
    "sm_103a": frozenset({"cake_causal_conv1d_train_461d358d7090642c65de"}),
}


def _pair_programs() -> set[str]:
    """The programs of the opt-in pair routes on any architecture."""

    return {
        name
        for by_kind in cake_jit.ROUTES.values()
        for kind, names in by_kind.items()
        if kind in cake_jit.PAIR_ROUTES
        for name in names
    }


@pytest.mark.cpu_only
def test_single_rank_programs_take_the_runtime_pdl_scalar() -> None:
    """Every program but the opt-in pair routes' launches with ``launch.use_pdl`` ``"runtime"``: one
    ``("launch", "enable_pdl")`` argument right before the grid, bound to the logical scalar ``enable_pdl`` of
    ``LOGICAL_SCALARS`` (not a plan scalar); the pair routes' programs launch without PDL and take no such
    argument."""

    assert "enable_pdl" in cake_jit.LOGICAL_SCALARS and "enable_pdl" not in cake_jit.PLAN_SCALARS
    pair = _pair_programs()
    assert pair and pair <= set(cake_jit.MODULES)
    runtime = 0
    for name, record in cake_jit.MODULES.items():
        launch_args = [entry for entry in record["arg_plan"] if entry[0] == "launch"]
        if name in pair:
            assert record["launch"]["use_pdl"] is False, name
            assert not launch_args and "enable_pdl" not in record["logical"], name
            continue
        assert record["launch"]["use_pdl"] == "runtime", name
        assert launch_args == [PDL_ARG] and record["arg_plan"][-4] == PDL_ARG, name
        assert record["logical"]["enable_pdl"] == "enable_pdl", name
        runtime += 1
    assert runtime == RUNTIME_PDL_PROGRAMS == len(cake_jit.MODULES) - len(pair)


@pytest.mark.cpu_only
def test_pdl_policy_tables(monkeypatch) -> None:
    """``PDL_EAGER`` and ``PDL_GRAPH`` are the literal policy (:data:`PDL_EAGER_OF_RECORD`,
    :data:`PDL_GRAPH_OF_RECORD`), one bool per architecture of ``ARCHES``; ``pdl_enabled`` reads ``PDL_GRAPH``
    inside a capture and ``PDL_EAGER`` outside one, as the int 0 or 1, and refuses an architecture outside
    ``ARCHES``."""

    assert cake_jit.PDL_EAGER == PDL_EAGER_OF_RECORD and cake_jit.PDL_GRAPH == PDL_GRAPH_OF_RECORD
    for table in (cake_jit.PDL_EAGER, cake_jit.PDL_GRAPH):
        assert tuple(table) == ARCHES and all(type(value) is bool for value in table.values()), table
    for arch in ARCHES:
        for capturing, table in ((False, PDL_EAGER_OF_RECORD), (True, PDL_GRAPH_OF_RECORD)):
            got = cake_jit.pdl_enabled(arch, capturing=capturing)
            assert type(got) is int and got == int(table[arch]), (arch, capturing)
    for arch in ("sm_90a", None):
        with pytest.raises(ValueError):
            cake_jit.pdl_enabled(arch, capturing=False)
    for value in (False, True):
        monkeypatch.setitem(cake_jit.PDL_EAGER, "sm_103a", value)
        monkeypatch.setitem(cake_jit.PDL_GRAPH, "sm_103a", not value)
        assert cake_jit.pdl_enabled("sm_103a", capturing=False) == int(value)
        assert cake_jit.pdl_enabled("sm_103a", capturing=True) == int(not value)


@pytest.mark.cpu_only
def test_pdl_off_rows_are_the_literal_rows() -> None:
    """``PDL_OFF_ROWS`` is the literal :data:`PDL_OFF_ROWS_OF_RECORD`.  Every key names an exported architecture, one
    of its backward routes other than a pair route (whose programs launch without PDL), a closed range of channel
    counts (multiples of 8) and a closed range of token counts from one token on; the keys of one architecture and
    route are disjoint; and at each corner of a key the call selects a program of
    :data:`PDL_OFF_PROGRAMS_OF_RECORD` that takes the per-call PDL scalar."""

    assert cake_jit.PDL_OFF_ROWS == PDL_OFF_ROWS_OF_RECORD
    assert set(cake_jit.PDL_OFF_ROWS) == set(PDL_OFF_PROGRAMS_OF_RECORD)
    for arch, rows in cake_jit.PDL_OFF_ROWS.items():
        assert arch in ARCHES and rows, arch
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.backward_kinds(arch) and route not in cake_jit.PAIR_ROUTES, label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert type(s_min) is int and type(s_max) is int, label
            assert 1 <= s_min <= s_max <= cake_jit.INT32_MAX, label
            for C in (c_lo, c_hi):
                for S in (s_min, s_max):
                    (name,) = cake_jit.selected_modules(arch, route, C=C, S=S)
                    assert name in PDL_OFF_PROGRAMS_OF_RECORD[arch], (label, C, S, name)
                    assert cake_jit.MODULES[name]["launch"]["use_pdl"] == "runtime", (label, name)
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in itertools.pairwise(ranges):
                assert hi < lo, (arch, route, ranges)


def _pdl_off_expected(arch: str, kind: str, C: int, S: int, capturing: bool) -> int:
    """The ``enable_pdl`` value of a call of route ``kind`` with ``C`` channels and ``S`` tokens on ``arch``, from
    :data:`PDL_OFF_ROWS_OF_RECORD` and the live ``PDL_EAGER`` / ``PDL_GRAPH`` (written apart from ``pdl_enabled``)."""

    for (route, c_lo, c_hi), (s_min, s_max) in PDL_OFF_ROWS_OF_RECORD.get(arch, {}).items():
        if route == kind and c_lo <= C <= c_hi and C % 8 == 0 and s_min <= S <= s_max:
            return 0
    return int((cake_jit.PDL_GRAPH if capturing else cake_jit.PDL_EAGER)[arch])


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_pdl_enabled_is_off_inside_the_pdl_off_rows(arch, monkeypatch) -> None:
    """With a call's row (``kind``, ``C`` and ``S``) ``pdl_enabled`` is 0 inside a :data:`PDL_OFF_ROWS_OF_RECORD` key
    (at both channel and both token bounds), eagerly and inside a capture, whatever ``PDL_EAGER`` / ``PDL_GRAPH``
    say; outside every key it is the tables' value, as without a row: one token below a key, a channel count off the
    8-channel grid, every other route (``bwd_bias``, ``bwd_dt`` and the forward routes among them) at a key's row,
    and every row on an architecture without keys (sm_107a); one channel step below and above a key and one token
    above it unless another key covers them.
    ``pdl_off_row_range`` holds the key's token range at its channel counts only.  The row is ``kind``, ``C`` and
    ``S`` together: a ``kind`` outside ``KINDS``, or a part of the row alone, is a ``ValueError``."""

    rows = PDL_OFF_ROWS_OF_RECORD.get(arch, {})
    assert cake_jit.PDL_EAGER[arch] and cake_jit.PDL_GRAPH[arch], arch
    for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
        for capturing in (False, True):
            for C in (c_lo, c_hi):
                for S in (s_min, s_max):
                    assert cake_jit.pdl_enabled(arch, capturing=capturing, kind=route, C=C, S=S) == 0, (C, S)
            # one token below the key, the key's row on another route (forward included), the key's row on sm_107a
            outside = [(route, c_lo, s_min - 1), (route, c_hi, s_min - 1)]
            outside += [
                (kind, C, s_min) for kind in ("bwd_bias", "bwd_dt", "fwd") if kind != route for C in (c_lo, c_hi)
            ]
            for kind, C, S in outside:
                got = cake_jit.pdl_enabled(arch, capturing=capturing, kind=kind, C=C, S=S)
                assert got == 1, (kind, C, S, capturing)
            assert cake_jit.pdl_enabled("sm_107a", capturing=capturing, kind=route, C=c_hi, S=s_min) == 1
            # next to the key (another key of the route may cover these): the tables' value unless a key covers it
            near = [(c_lo - 8, s_min), (c_hi + 8, s_min), (c_lo + 4, s_min), (c_hi, s_max + 1)]
            for C, S in near:
                got = cake_jit.pdl_enabled(arch, capturing=capturing, kind=route, C=C, S=S)
                assert got == _pdl_off_expected(arch, route, C, S, capturing), (route, C, S, capturing)
            assert cake_jit.pdl_enabled(arch, capturing=capturing, kind=route, C=c_lo + 4, S=s_min) == 1
    channels = {512, 1016, 1024}
    tokens = {1, 1000, 125170, 131072, 131073, 187754, 187755, 250340, cake_jit.INT32_MAX}
    for (_, c_lo, c_hi), (s_min, s_max) in rows.items():
        channels |= {c_lo - 8, c_lo, c_lo + 4, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
        tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    turned_off = 0
    for eager_on, graph_on in ((True, True), (False, True), (True, False), (False, False)):
        monkeypatch.setitem(cake_jit.PDL_EAGER, arch, eager_on)
        monkeypatch.setitem(cake_jit.PDL_GRAPH, arch, graph_on)
        for capturing in (False, True):
            table = int(graph_on if capturing else eager_on)
            assert cake_jit.pdl_enabled(arch, capturing=capturing) == table
            for kind in cake_jit.KINDS:
                for C in sorted(c for c in channels if c > 0):
                    keys = [b for (r, lo, hi), b in rows.items() if r == kind and lo <= C <= hi and C % 8 == 0]
                    assert cake_jit.pdl_off_row_range(arch, kind, C) == (keys[0] if keys else None), (kind, C)
                    for S in sorted(s for s in tokens if s > 0):
                        want = _pdl_off_expected(arch, kind, C, S, capturing)
                        got = cake_jit.pdl_enabled(arch, capturing=capturing, kind=kind, C=C, S=S)
                        assert type(got) is int and got == want, (kind, C, S, capturing, eager_on, graph_on)
                        turned_off += want == 0 and table == 1
    assert bool(turned_off) is bool(rows), (arch, turned_off)
    for row in (
        dict(kind="bwd"),
        dict(C=1016),
        dict(S=187755),
        dict(kind="bwd", C=1016),
        dict(kind="bwd", S=187755),
        dict(C=1016, S=187755),
        dict(kind="bwd_unknown", C=1016, S=187755),
    ):
        with pytest.raises(ValueError):
            cake_jit.pdl_enabled(arch, capturing=False, **row)
    with pytest.raises(ValueError):
        cake_jit.pdl_enabled("sm_90a", capturing=False, kind="bwd", C=1016, S=187755)


_KERNEL_SIGNATURE = re.compile(r"__global__[^;{]*?\b(kernel_\w+)\s*\(")


def _kernel_waits_first(text: str, library: str) -> int:
    """Check every ``__global__`` function of a kernel unit: its body executes ``griddepcontrol.wait`` and, before the
    first one, names none of its pointer or tensor-map parameters (the text before it, comments dropped; a member
    access such as ``threadIdx.x`` is not the parameter ``x``).  Returns the number of functions checked."""

    checked = 0
    for match in _KERNEL_SIGNATURE.finditer(text):
        depth, i = 1, match.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        params = text[match.end() : i - 1].split(",")
        guarded = [
            re.findall(r"\w+", param)[-1]
            for param in params
            if "*" in param or "CUtensorMap" in param
        ]
        body_start = text.index("{", i)
        wait = text.find("griddepcontrol.wait", body_start)
        label = (library, match.group(1))
        assert guarded and wait > body_start, label
        before = re.sub(r"//[^\n]*", "", text[body_start:wait])
        named = [name for name in guarded if re.search(rf"(?<!\.)(?<!->)\b{re.escape(name)}\b", before)]
        assert not named, (label, named)
        checked += 1
    return checked


@pytest.mark.cpu_only
def test_runtime_pdl_bindings_and_kernels() -> None:
    """The rendered binding of every library whose records take the per-call PDL scalar takes ``enable_pdl`` as its
    last host parameter, refuses a value other than 0 and 1 and sets the programmatic stream serialization attribute
    only when it is 1; a pair-route library's binding has no PDL attribute.  Every kernel of a PDL library waits for
    the preceding grid (``griddepcontrol.wait``) before it names any pointer or tensor-map parameter, so the
    attribute only lets its launch start early, never read or write before the preceding kernel finishes."""

    pair = _pair_programs()
    libraries = {record["library"]: name in pair for name, record in cake_jit.MODULES.items()}
    csrc = PACKAGE_DIR / "csrc"
    kernels = 0
    for library, is_pair in sorted(libraries.items()):
        binding = (csrc / f"{library}_binding.cu").read_text()
        if is_pair:
            assert "enable_pdl" not in binding and "ProgrammaticStreamSerialization" not in binding, library
            continue
        (run,) = re.findall(r"void Run\(([^)]*)\)", binding)
        params = [param.strip() for param in run.split(",")]
        assert params[-4:] == [
            "int64_t a_enable_pdl",
            "int64_t grid_x",
            "int64_t grid_y",
            "int64_t grid_z",
        ], library
        assert "a_enable_pdl == 0LL || a_enable_pdl == 1LL" in binding, library
        assert binding.count("cudaLaunchAttributeProgrammaticStreamSerialization") == 1, library
        attribute = binding.index("cudaLaunchAttributeProgrammaticStreamSerialization")
        assert binding.rfind("if (a_enable_pdl) {", 0, attribute) > binding.index("void Run("), library
        kernels += _kernel_waits_first((csrc / f"{library}_kernel.cu").read_text(), library)
    assert kernels >= RUNTIME_PDL_PROGRAMS


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_selection_matches_the_deterministic_package(arch, monkeypatch) -> None:
    """A call selects the program at the same route position as the deterministic package for every ``(S, C)``,
    and plans the same scalars (``y`` / ``dx`` keep their per-element arithmetic).  Where the deterministic package
    selects a det-only program, this package selects what the deterministic package selects without its det-only
    records but those this package copies (:data:`OWN_COPIES`), and its copy where that is one of them.  At a row
    of :data:`FALLBACK_ROWS_OF_RECORD` this package selects its route's fallback (the program at the deterministic
    route's first position), where the deterministic package selects the program at the position of one of
    :data:`FALLBACK_REPLACES_OF_RECORD`."""

    twin_of = {
        band: twin for band_arch, _, band, twin, _ in _bands() if band_arch == arch
    }
    det_only_hits = set()
    replaced = set()
    for kind in cake_jit.KINDS:
        new, det = cake_jit.ROUTES[arch][kind], _det_route(arch, kind)
        for C in CHANNELS:
            for S in TOKENS:
                (name,) = cake_jit.selected_modules(arch, kind, C=C, S=S)
                (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                if det_name in DET_ONLY:
                    det_only_hits.add(det_name)
                    copied = [
                        d
                        for d in det_jit.ROUTES[arch][kind]
                        if d not in DET_ONLY or d in OWN_COPIES
                    ]
                    with monkeypatch.context() as m:
                        m.setitem(det_jit.ROUTES[arch], kind, copied)
                        (det_name,) = det_jit.selected_modules(arch, kind, C=C, S=S)
                bounds = cake_jit.fallback_row_range(arch, kind, C)
                if bounds is not None and bounds[0] <= S <= bounds[1]:
                    # The route's fallback, in place of the program at the deterministic pick's position.
                    assert name == new[0] and cake_jit.select_rule(cake_jit.MODULES[name]) is None, (
                        arch, kind, C, S, name
                    )
                    taken = new[det.index(det_name)]
                    assert taken in FALLBACK_REPLACES_OF_RECORD[arch], (arch, kind, C, S, taken)
                    replaced.add(taken)
                elif det_name in OWN_COPIES:
                    # This package's copy of the deterministic channel-band program.
                    assert name == OWN_COPIES[det_name], (arch, kind, C, S, name)
                else:
                    # A channel-band program serves its twin's route position.
                    position = new.index(twin_of.get(name, name))
                    assert position == det.index(det_name), (arch, kind, C, S)
                record, det_record = cake_jit.MODULES[name], det_jit.MODULES[det_name]
                geometry = record["grid_rule"]["geometry"]
                assert cake_jit.plan(
                    S, C, 3, geometry=geometry, role=record["role"]
                ) == det_jit.plan(
                    S,
                    C,
                    3,
                    geometry=det_record["grid_rule"]["geometry"],
                    role=det_record["role"],
                ), (arch, kind, C, S)
    det_only_routed = {
        name for name in DET_ONLY if arch in det_jit.MODULES[name]["arches"]
    }
    assert det_only_hits == det_only_routed, arch
    assert replaced == FALLBACK_REPLACES_OF_RECORD.get(arch, frozenset()), arch


# The FP32 accumulator slots each backward route keeps (one FP32 word per sum and slot): the fp32g routes spread a
# channel tile's CTAs over 16 copies, the others add into one; the opt-in pair routes keep 4 copies of an FP32
# (hi, lo) pair, 8 slots.
ACC_SLOTS = {
    "bwd": 1,
    "bwd_dt": 1,
    "bwd_bias": 1,
    "bwd_fp32g": 16,
    "bwd_bias_fp32g": 16,
    "bwd_dt_bf16_taps": 1,
    "bwd_bias_bf16_taps": 1,
    "bwd_fp32g_pair": 8,
    "bwd_bias_fp32g_pair": 8,
}


def _align16(n: int) -> int:
    return -(-n // 16) * 16


@pytest.mark.cpu_only
def test_accumulator_layout() -> None:
    assert cake_jit.WORKSPACE_ALIGN == 16
    assert cake_jit.ACCUMULATOR_DTYPES == {"float32": 4, "float64": 8}
    assert cake_jit.COUNTER_DTYPES == {"uint32": 4}
    for arch, kind, name, _ in _pairs("bwd"):
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        size = cake_jit.ACCUMULATOR_DTYPES[rule["acc"]]
        for C in CHANNELS:
            _, _, n_ch_tiles = cake_jit.channel_split(
                C, geometry=record["grid_rule"]["geometry"], role="bwd"
            )
            layout = cake_jit.accumulator_layout(record, C, n_ch_tiles)
            words = rule["slots"] * rule["taps"]
            assert cake_jit.accumulator_words_per_channel(record) == words
            acc_bytes = words * C * size
            assert layout == {
                "acc_dtype": rule["acc"],
                "acc_elems": words * C,
                "acc_bytes": acc_bytes,
                "work_dtype": "uint32",
                "work_offset": _align16(acc_bytes),
                "work_words": n_ch_tiles,
                "total_bytes": _align16(_align16(acc_bytes) + 4 * n_ch_tiles),
            }, (arch, kind, C)
            # The views the host takes stay inside the buffer and do not overlap.
            assert layout["acc_bytes"] <= layout["work_offset"]
            assert layout["work_offset"] + 4 * n_ch_tiles <= layout["total_bytes"]
            assert layout["work_offset"] % size == 0
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 0, 1)
        with pytest.raises(ValueError):
            cake_jit.accumulator_layout(record, 256, 0)
    forward = cake_jit.MODULES[cake_jit.ROUTES["sm_100a"]["fwd"][0]]
    with pytest.raises(ValueError, match="no backward accumulator workspace"):
        cake_jit.accumulator_layout(forward, 256, 1)


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ("bwd_fp32g", "bwd_bias_fp32g"))
@pytest.mark.parametrize("C", (512, 8192))
def test_accumulators_are_sized_from_the_rule(kind, C) -> None:
    """The accumulator size comes from the rendered rule, never a fixed ``C x taps``: an fp32g program whose rule
    names ``slots`` FP32 copies gets ``slots x taps x C`` words; a rule without ``slots``, or with a malformed one,
    is refused."""

    import copy

    for arch in ARCHES:
        if kind not in cake_jit.ROUTES[arch]:
            continue
        name = cake_jit.route_modules(arch, kind)[0]
        record = cake_jit.MODULES[name]
        rule = record["workspace_rule"]
        assert rule["acc"] == "float32" and rule["slots"] == ACC_SLOTS[kind]
        _, _, n_ch_tiles = cake_jit.channel_split(
            C, geometry=record["grid_rule"]["geometry"], role="bwd"
        )
        rendered = cake_jit.accumulator_layout(record, C, n_ch_tiles)
        assert rendered["acc_elems"] == rule["slots"] * rule["taps"] * C
        for slots in (1, 4):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            layout = cake_jit.accumulator_layout(variant, C, n_ch_tiles)
            assert layout["acc_elems"] == slots * rule["taps"] * C, (arch, slots)
            assert layout["acc_bytes"] == 4 * slots * rule["taps"] * C
            assert layout["work_offset"] == _align16(layout["acc_bytes"])
            assert layout["total_bytes"] >= layout["work_offset"] + 4 * n_ch_tiles
        for slots in (0, -1, 1.5, True, "4", None):
            variant = copy.deepcopy(record)
            variant["workspace_rule"]["slots"] = slots
            with pytest.raises(ValueError, match="slots"):
                cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        # A rule without the field is refused (no default could size dw_acc for every program).
        variant = copy.deepcopy(record)
        del variant["workspace_rule"]["slots"]
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_layout(variant, C, n_ch_tiles)
        with pytest.raises(ValueError, match="slots"):
            cake_jit.accumulator_words_per_channel(variant)
        with pytest.raises(ValueError, match="slots"):
            pkg._accumulator_layout(variant, C, n_ch_tiles)


def _owner_tiles(record: dict, arch: str, C: int) -> int:
    """The largest channel tile count a backward ``record`` launches with at ``C`` on ``arch``, whatever ``S``:
    the split's, or its route's :data:`cake_jit.TILE_OVERRIDES` or :data:`cake_jit.NARROW_TILES` row's when that
    has more."""

    geometry = record["grid_rule"]["geometry"]
    route = cake_jit.record_route(record, arch)
    _, _, n_ch_tiles = cake_jit.channel_split(C, geometry=geometry, role="bwd")
    entry = cake_jit.TILE_OVERRIDES.get(arch, {}).get((TILE_ROUTE.get(route, route), C))
    if entry is not None:
        n_ch_tiles = max(n_ch_tiles, -(-(C // geometry["vec"]) // entry[0]))
    elif (
        cake_jit.narrow_tile_range(arch, route, C) is not None
        and geometry["threads"] == 128
    ):
        n_ch_tiles = max(n_ch_tiles, -(-(C // geometry["vec"]) // 32))
    return n_ch_tiles


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_workspace_serves_every_backward_program(arch) -> None:
    """The public backward workspace is the largest layout over every backward program of the architecture, each
    at the largest tile count it may launch with (its split, or its route's ``TILE_OVERRIDES`` or ``NARROW_TILES``
    row), so one zeroed buffer serves whichever program, plan and route a call selects; the capacity owner is an
    fp32g bias program (4 bytes x 16 copies x 5 taps per channel, 320).  A narrow row's three tickets fit the
    16-byte alignment of the single tile's, so its size is the split's."""

    owner = cake_jit.workspace_record(arch)
    assert (
        owner["workspace_rule"]["acc"] == "float32"
        and owner["workspace_rule"]["slots"] == ACC_SLOTS["bwd_bias_fp32g"]
        and owner["workspace_rule"]["taps"] == 5
    )
    owners = cake_jit.workspace_owners(arch)
    assert len(owners) == sum(
        len(cake_jit.route_modules(arch, kind))
        for kind in cake_jit.backward_kinds(arch)
    )
    overridden = {C for _, C in cake_jit.TILE_OVERRIDES.get(arch, {})}
    narrow = {
        C for _, c_lo, c_hi in cake_jit.NARROW_TILES.get(arch, {}) for C in (c_lo, c_hi)
    }
    for C in sorted(set(CHANNELS) | overridden | narrow):
        nbytes = cake_jit.backward_workspace_bytes(arch, C)
        layouts = [
            cake_jit.accumulator_layout(record, C, _owner_tiles(record, arch, C))
            for record in owners
        ]
        assert nbytes == max(layout["total_bytes"] for layout in layouts), (arch, C)
        if C not in overridden:
            per_channel = 4 * cake_jit.accumulator_words_per_channel(owner)
            assert nbytes == _align16(_align16(per_channel * C) + 4 * cake_jit.channel_split(
                C, geometry=owner["grid_rule"]["geometry"], role="bwd")[2])  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# The narrower channel tiles of the backward (``TILE_OVERRIDES``).


@pytest.mark.cpu_only
def test_bf16_taps_routes_run_their_base_routes_records() -> None:
    """``TILE_ROUTE`` names each ``_bf16_taps`` route's base route.  On every architecture a ``_bf16_taps`` route
    lists, position for position, its base route's records under their own names (the same selection rule, plan,
    grid, launch, arguments and workspace rule; another program), and the tile tables name base routes only (a
    ``_bf16_taps`` route plans by its base route's rows)."""

    assert cake_jit.TILE_ROUTE == TILE_ROUTE
    for arch in ARCHES:
        for twin, base in TILE_ROUTE.items():
            names, base_names = cake_jit.ROUTES[arch][twin], cake_jit.ROUTES[arch][base]
            assert not set(names) & set(base_names), (arch, twin)
            for name, base_name in zip(names, base_names, strict=True):
                assert _strip(cake_jit.MODULES[name], *NAME_FIELDS) == _strip(
                    cake_jit.MODULES[base_name], *NAME_FIELDS
                ), (arch, twin, name, base_name)
    for table in (cake_jit.TILE_OVERRIDES, cake_jit.NARROW_TILES):
        for rows in table.values():
            for key in rows:
                assert key[0] not in TILE_ROUTE, key


@pytest.mark.cpu_only
def test_tile_overrides_are_backward_rows() -> None:
    """Every row of the rendered table names an exported architecture, one of its backward routes, a channel count
    of the package and a ``ct_vec`` that is a whole number of warps no wider than a CTA, over a closed ``S`` range."""

    assert set(cake_jit.TILE_OVERRIDES) <= set(ARCHES)
    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (route, C), (ct_vec, s_min, s_max) in rows.items():
            label = (arch, route, C)
            assert route in cake_jit.backward_kinds(arch), label
            assert C > 0 and C % cake_jit.CHANNEL_MULTIPLE == 0, label
            assert 0 < s_min <= s_max, label
            for name in cake_jit.route_modules(arch, route):
                threads = cake_jit.MODULES[name]["grid_rule"]["geometry"]["threads"]
                assert ct_vec % 32 == 0 and 0 < ct_vec <= threads, label


@pytest.mark.cpu_only
def test_tile_override_bounds_and_plan() -> None:
    """A row's ``ct_vec`` applies from ``S_min`` through ``S_max`` (both included) and to no other token count,
    route or architecture; ``plan`` of a backward record given its architecture and route then plans ``(ct_vec,
    threads // ct_vec, ceil(C / vec / ct_vec))``, and :func:`cake_jit.channel_split`'s split outside the range, for
    the forward role without a route, under ``legacy`` and without the architecture or the route; the forward role
    refuses a backward route (``ValueError``, as the deterministic plan does)."""

    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (base, C), (ct_vec, s_min, s_max) in rows.items():
            for route in _planning_routes(arch, base):
                label = (arch, route, C)
                assert cake_jit.tile_override(arch, route, C, s_min - 1) is None, label
                assert cake_jit.tile_override(arch, route, C, s_min) == ct_vec, label
                assert cake_jit.tile_override(arch, route, C, s_max) == ct_vec, label
                assert cake_jit.tile_override(arch, route, C, s_max + 1) is None, label
                for other in ARCHES:
                    if (base, C) not in cake_jit.TILE_OVERRIDES.get(other, {}):
                        assert cake_jit.tile_override(other, route, C, s_min) is None
                for name in cake_jit.route_modules(arch, route):
                    geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
                    threads, vec = geometry["threads"], geometry["vec"]
                    split = dict(
                        zip(
                            ("ct_vec", "n_sub", "n_ch_tiles"),
                            cake_jit.channel_split(C, geometry=geometry, role="bwd"),
                            strict=True,
                        )
                    )
                    wanted = {
                        "ct_vec": ct_vec,
                        "n_sub": threads // ct_vec,
                        "n_ch_tiles": -(-(C // vec) // ct_vec),
                    }
                    for S in (s_min, s_max):
                        got = cake_jit.plan(
                            S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                        )
                        assert {k: got[k] for k in wanted} == wanted, (label, S)
                    for S in (s_min - 1, s_max + 1):
                        got = cake_jit.plan(
                            S, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                        )
                        assert {k: got[k] for k in split} == split, (label, S)
                    for kwargs in (
                        {"role": "bwd"},
                        {"role": "bwd", "arch": arch},
                        {"role": "bwd", "route": route},
                        {"role": "fwd", "arch": arch},
                    ):
                        got = cake_jit.plan(s_min, C, 3, geometry=geometry, **kwargs)
                        want = cake_jit.channel_split(
                            C, geometry=geometry, role=kwargs["role"]
                        )
                        assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, (
                            label,
                            kwargs,
                        )
                    for registry in (cake_jit, det_jit):
                        with pytest.raises(ValueError, match="is not a route of role"):
                            registry.plan(
                                s_min, C, 3, geometry=geometry, role="fwd", arch=arch, route=route
                            )
                    got = cake_jit.plan(
                        s_min, C, 3, geometry=geometry, legacy=True, arch=arch, route=route
                    )
                    want = cake_jit.channel_split(C, geometry=geometry, legacy=True)
                    assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, label


@pytest.mark.cpu_only
def test_backward_workspace_covers_every_override_row() -> None:
    """The public workspace of a row's ``C`` holds the layout of every program of the row's route, and of every
    route that plans by its rows, at the row's tile count (one ticket per tile), whatever ``S``."""

    for arch, rows in cake_jit.TILE_OVERRIDES.items():
        for (base, C), (_, s_min, _) in rows.items():
            nbytes = cake_jit.backward_workspace_bytes(arch, C)
            for route in _planning_routes(arch, base):
                for name in cake_jit.route_modules(arch, route):
                    record = cake_jit.MODULES[name]
                    geometry = record["grid_rule"]["geometry"]
                    tiles = cake_jit.plan(
                        s_min, C, 3, geometry=geometry, role="bwd", arch=arch, route=route
                    )["n_ch_tiles"]
                    assert tiles == cake_jit.override_tiles(arch, route, C, geometry)
                    layout = cake_jit.accumulator_layout(record, C, tiles)
                    assert layout["total_bytes"] <= nbytes, (arch, route, C, name)


# The C 768 rows of ``TILE_OVERRIDES`` per architecture, route -> (ct_vec, S_min, S_max): on sm_100a and sm_103a the
# measured rows of the package's 0.2.0 release on every base route but ``bwd`` (whose C 768 runs the 32-vector tiles of
# its ``NARROW_TILES`` key, :data:`C768_NARROW_ROUTES`), on sm_107a the ``bwd_dt`` row of every S
# (``bwd_dt_bf16_taps`` plans by it too).
C768_TILE_OVERRIDES = {
    "sm_100a": {
        "bwd_dt": (32, 62585, 500680),
        "bwd_bias": (32, 62585, 500680),
        "bwd_fp32g": (32, 31292, 500680),
        "bwd_bias_fp32g": (32, 31292, 500680),
    },
    "sm_103a": {
        "bwd_dt": (32, 62585, 500680),
        "bwd_bias": (32, 62585, 500680),
        "bwd_fp32g": (32, 62585, 500680),
        "bwd_bias_fp32g": (32, 31292, 500680),
    },
    "sm_107a": {"bwd_dt": (32, 1, 2147483647)},
}
# The architectures whose C 768 backward rows are only those measured rows on every backward route but the routes of
# :data:`C768_NARROW_ROUTES`: there a ``NARROW_TILES`` key of the route covers C 768 at every S (the ``bwd`` route's
# C 520 to 1016 key), on every other route none does.
C768_MEASURED_ONLY = ("sm_100a", "sm_103a")
C768_NARROW_ROUTES = {"sm_100a": ("bwd",), "sm_103a": ("bwd",)}


@pytest.mark.cpu_only
def test_c768_backward_rows_are_the_literal_rows() -> None:
    """The C 768 rows of the rendered ``TILE_OVERRIDES`` are the literal :data:`C768_TILE_OVERRIDES` on every
    architecture.  On the architectures of :data:`C768_MEASURED_ONLY` the ``NARROW_TILES`` keys covering C 768 are
    exactly one per route of :data:`C768_NARROW_ROUTES`, each over every token count, and no other backward route has
    a narrow range at C 768."""

    assert set(C768_TILE_OVERRIDES) == set(ARCHES)
    for arch in ARCHES:
        rows = {
            route: tuple(int(value) for value in entry)
            for (route, C), entry in cake_jit.TILE_OVERRIDES.get(arch, {}).items()
            if C == 768
        }
        assert rows == C768_TILE_OVERRIDES[arch], arch
    assert set(C768_NARROW_ROUTES) == set(C768_MEASURED_ONLY)
    for arch in C768_MEASURED_ONLY:
        covering = [key for key in cake_jit.NARROW_TILES.get(arch, {}) if key[1] <= 768 <= key[2]]
        assert sorted(key[0] for key in covering) == sorted(C768_NARROW_ROUTES[arch]), (arch, covering)
        for route in cake_jit.backward_kinds(arch):
            want = (1, cake_jit.INT32_MAX) if route in C768_NARROW_ROUTES[arch] else None
            assert cake_jit.narrow_tile_range(arch, route, 768) == want, (arch, route)


# ----------------------------------------------------------------------------------------------------------------
# The rows where a backward route runs its fallback program (``FALLBACK_ROWS``).


def _route_fallback(arch: str, kind: str) -> str:
    """The fallback program of route ``kind`` on ``arch``: its one record without a selection rule."""

    (name,) = [
        name for name in cake_jit.route_modules(arch, kind) if cake_jit.select_rule(cake_jit.MODULES[name]) is None
    ]
    return name


def _rules_pick(arch: str, kind: str, C: int, S: int) -> str:
    """The program the selection rules pick for ``(S, C)`` of route ``kind`` on ``arch`` (no fallback rows, no SM-wide
    rows)."""

    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FALLBACK_ROWS", {})
        m.setattr(cake_jit, "SM_WIDE_ROWS", {})
        (name,) = cake_jit.selected_modules(arch, kind, C=C, S=S)
    return name


def _table_pick(arch: str, kind: str, C: int, S: int) -> str | None:
    """The program the registry's ``SM_WIDE_ROWS`` segment, else its ``FALLBACK_ROWS`` range, holding ``(C, S)`` of
    route ``kind`` on ``arch`` names (read from the two tables, not through the selection), ``None`` outside both."""

    if C % 8:
        return None
    for (route, c_lo, c_hi), segments in cake_jit.SM_WIDE_ROWS.get(arch, {}).items():
        if route == kind and c_lo <= C <= c_hi:
            for s_lo, s_hi, name in segments:
                if s_lo <= S <= s_hi:
                    return name
    for (route, c_lo, c_hi), (s_min, s_max) in cake_jit.FALLBACK_ROWS.get(arch, {}).items():
        if route == kind and c_lo <= C <= c_hi and s_min <= S <= s_max:
            return _route_fallback(arch, kind)
    return None


@pytest.mark.cpu_only
def test_fallback_rows_are_the_literal_rows() -> None:
    """``FALLBACK_ROWS`` is the literal :data:`FALLBACK_ROWS_OF_RECORD`.  Every key names an exported architecture,
    one of its backward routes other than a pair route (an opt-in route selects by its own rules only), a closed
    range of channel counts (multiples of 8) and a closed range of token counts from one token; the keys of one
    architecture and route are disjoint, and the route's fallback program ships for the architecture."""

    assert cake_jit.FALLBACK_ROWS == FALLBACK_ROWS_OF_RECORD
    for arch, rows in cake_jit.FALLBACK_ROWS.items():
        assert arch in ARCHES and rows, arch
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.backward_kinds(arch) and route not in cake_jit.PAIR_ROUTES, label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert type(s_min) is int and type(s_max) is int, label
            assert 1 <= s_min <= s_max <= cake_jit.INT32_MAX, label
            assert arch in cake_jit.MODULES[_route_fallback(arch, route)]["arches"], label
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in itertools.pairwise(ranges):
                assert hi < lo, (arch, route, ranges)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_fallback_rows_select_the_route_fallback(arch, fake_device) -> None:
    """Inside a key (``C_lo <= C <= C_hi``, ``C`` a multiple of 8, ``S_min <= S <= S_max``) a call of the key's route
    selects the route's fallback program, inside an ``SM_WIDE_ROWS`` segment the segment's program, and the program
    the selection rules pick everywhere else: both sides of every channel and token bound, on every backward route
    of the architecture.  ``fallback_row_range`` holds the key's token range at its channel counts only; ``S`` of
    ``None`` is a ``ValueError`` where a key covers ``C``.  ``plan_row`` needs ``at`` there (``ValueError`` without
    it) and plans, at the first token count of every side of ``row_sides``, the program of that side (``None`` on a
    rules side where a token floor decides); at a row no key covers it plans the same row whatever ``at``."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    rows = cake_jit.FALLBACK_ROWS.get(arch, {})
    assert bool(rows) is (arch in FALLBACK_ROWS_OF_RECORD), arch
    channels = {768, 1024}
    tokens = {1, 1000, 1 << 20, cake_jit.INT32_MAX}
    for (_, c_lo, c_hi), (s_min, s_max) in rows.items():
        channels |= {c_lo - 8, c_lo, c_lo + 4, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
        tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    for (_, c_lo, c_hi), segments in cake_jit.SM_WIDE_ROWS.get(arch, {}).items():
        channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
        tokens |= {bound for s_lo, s_hi, _ in segments for bound in (s_lo - 1, s_lo, s_hi, s_hi + 1)}
    inside_calls = 0
    for kind in cake_jit.backward_kinds(arch):
        fallback = _route_fallback(arch, kind)
        for C in sorted(C for C in channels if C > 0):
            keys = [
                bounds
                for (route, c_lo, c_hi), bounds in rows.items()
                if route == kind and c_lo <= C <= c_hi and C % 8 == 0
            ]
            assert len(keys) <= 1, (kind, C)
            bounds = keys[0] if keys else None
            assert cake_jit.fallback_row_range(arch, kind, C) == bounds, (kind, C)
            for S in sorted(S for S in tokens if 0 < S <= cake_jit.INT32_MAX):
                inside = bounds is not None and bounds[0] <= S <= bounds[1]
                want = _table_pick(arch, kind, C, S) or _rules_pick(arch, kind, C, S)
                assert inside is (want == fallback and _table_pick(arch, kind, C, S) is not None), (kind, C, S)
                assert cake_jit.selected_modules(arch, kind, C=C, S=S) == (want,), (kind, C, S)
                inside_calls += inside
            if C % 8:
                continue

            def row(at, kind=kind, C=C):
                return cake_jit.plan_row(arch, kind, C, device_index=FAKE_DEVICE, at=at)

            sides = cake_jit.row_sides(arch, kind, C)
            if len(sides) == 1:
                assert bounds is None and sides == ((0, sys.maxsize),), (kind, C)
                assert row(None) == row(1) == row(cake_jit.INT32_MAX), (kind, C)
                continue
            with pytest.raises(ValueError, match="needs S"):
                cake_jit.selected_modules(arch, kind, C=C)
            with pytest.raises(ValueError, match="at="):
                row(None)
            for s_lo, _ in sides:
                planned, picked = row(s_lo), _table_pick(arch, kind, C, s_lo)
                if picked is None and cake_jit.tokens_select(arch, kind, C):
                    assert planned is None, (kind, C, s_lo)
                    continue
                assert planned.names == (picked or _rules_pick(arch, kind, C, s_lo),), (kind, C, s_lo)
                if picked is None and s_lo > 0:
                    assert planned.names != (fallback,), (kind, C, s_lo)
    assert bool(inside_calls) is bool(rows), arch


# ----------------------------------------------------------------------------------------------------------------
# The 32-vector channel tiles of the narrow single-tile backward rows (``NARROW_TILES``).

# A table that exercises the narrow-row plan on every architecture whatever the rendered table holds: the sm_100a
# rows of the package's 0.2.0 release (route ``bwd`` only, from 31292 to 500680 tokens), plus rows on the other
# architectures (one of them spans a ``TILE_OVERRIDES`` row of its route, C = 768 of route ``bwd_dt`` on sm_103a,
# which keeps its override).
SYNTHETIC_NARROW_TILES = {
    "sm_100a": {
        ("bwd", 224, 224): (31292, 500680),
        ("bwd", 520, 624): (31292, 500680),
        ("bwd", 632, 760): (31292, 500680),
    },
    "sm_103a": {
        ("bwd_dt", 632, 768): (62585, 250340),
        ("bwd_bias", 224, 224): (31292, 125170),
    },
    "sm_107a": {("bwd_bias", 520, 760): (31292, 62585)},
}
NARROW_TABLES = ("rendered", "synthetic")


@pytest.fixture
def narrow_table(request, monkeypatch):
    """The ``NARROW_TILES`` table of a test: the rendered one, or :data:`SYNTHETIC_NARROW_TILES` in its place."""

    if request.param == "synthetic":
        monkeypatch.setattr(cake_jit, "NARROW_TILES", SYNTHETIC_NARROW_TILES)
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    yield cake_jit.NARROW_TILES
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _covering(table: dict, arch: str, route: str, C: int):
    """The ``(S_min, S_max)`` of the first key of ``table[arch]`` on ``route`` covering ``C`` (``None``: none, or
    a ``TILE_OVERRIDES`` row of ``(route, C)``; a ``_bf16_taps`` route reads its base route's keys)."""

    route = TILE_ROUTE.get(route, route)
    if C % 8 or (route, C) in cake_jit.TILE_OVERRIDES.get(arch, {}):
        return None
    for (name, c_lo, c_hi), bounds in table.get(arch, {}).items():
        if name == route and c_lo <= C <= c_hi:
            return bounds
    return None


def _split3(plan: dict) -> tuple[int, int, int]:
    return plan["ct_vec"], plan["n_sub"], plan["n_ch_tiles"]


@pytest.mark.cpu_only
def test_narrow_tiles_are_backward_rows() -> None:
    """Every key of the rendered table names an exported architecture, one of its backward routes, a closed range of
    channel counts (multiples of 8) and a closed ``S`` range; no key of an architecture and route overlaps another,
    and no ``(route, C)`` of the architecture's ``TILE_OVERRIDES`` lies inside a key."""

    assert set(cake_jit.NARROW_TILES) <= set(ARCHES)
    for arch, rows in cake_jit.NARROW_TILES.items():
        spans: dict[str, list[tuple[int, int]]] = {}
        for (route, c_lo, c_hi), (s_min, s_max) in rows.items():
            label = (arch, route, c_lo, c_hi)
            assert route in cake_jit.backward_kinds(arch), label
            assert 0 < c_lo <= c_hi and c_lo % 8 == 0 and c_hi % 8 == 0, label
            assert 0 < s_min <= s_max, label
            for name in cake_jit.route_modules(arch, route):
                geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
                assert geometry["threads"] % 32 == 0, (label, name)
            for override_route, C in cake_jit.TILE_OVERRIDES.get(arch, {}):
                assert not (override_route == route and c_lo <= C <= c_hi), label
            spans.setdefault(route, []).append((c_lo, c_hi))
        for route, ranges in spans.items():
            ranges.sort()
            for (_, hi), (lo, _) in zip(ranges, ranges[1:], strict=False):
                assert hi < lo, (arch, route, ranges)


def _check_narrow_plan(table: dict, arch: str, route: str, C: int, S: int) -> int:
    """One ``(route, C, S)`` call of ``arch`` against ``table``: the narrow-row reading on every architecture, and
    ``plan`` of every program of the route with and without the architecture / route, under ``legacy`` and at
    another thread count.  Returns the number of programs checked."""

    for other in ARCHES:
        bounds = _covering(table, other, route, C)
        inside = bounds is not None and bounds[0] <= S <= bounds[1]
        assert cake_jit.narrow_tile_range(other, route, C) == bounds, (other, route, C)
        assert cake_jit.narrow_tile_row(other, route, C, S) == inside, (other, route, C)
    bounds = _covering(table, arch, route, C)
    inside = bounds is not None and bounds[0] <= S <= bounds[1]
    override = cake_jit.tile_override(arch, route, C, S)
    on_row = {"arch": arch, "route": route}
    for name in cake_jit.route_modules(arch, route):
        geometry = cake_jit.MODULES[name]["grid_rule"]["geometry"]
        threads, vec = geometry["threads"], geometry["vec"]
        label = (arch, route, C, S, name)
        split = cake_jit.channel_split(C, geometry=geometry, role="bwd")
        if override is not None:
            want = (override, threads // override, -(-(C // vec) // override))
        elif inside and threads == 128:
            want = (32, threads // 32, -(-(C // vec) // 32))
        else:
            want = split
        got = cake_jit.plan(S, C, 3, geometry=geometry, role="bwd", **on_row)
        assert _split3(got) == want, label
        for kwargs in ({}, {"arch": arch}, {"route": route}):
            got = cake_jit.plan(S, C, 3, geometry=geometry, role="bwd", **kwargs)
            assert _split3(got) == split, (label, kwargs)
        got = cake_jit.plan(S, C, 3, geometry=geometry, legacy=True, **on_row)
        legacy = cake_jit.channel_split(C, geometry=geometry, legacy=True)
        assert _split3(got) == legacy, label
        wide = {**geometry, "threads": 256}
        got = cake_jit.plan(S, C, 3, geometry=wide, role="bwd", **on_row)
        if override is None:
            wide_split = cake_jit.channel_split(C, geometry=wide, role="bwd")
            assert _split3(got) == wide_split, label
        tiles = cake_jit.narrow_tiles(arch, route, C, geometry)
        narrow = bounds is not None and threads == 128
        assert tiles == (-(-(C // vec) // 32) if narrow else 0), label
    return len(cake_jit.route_modules(arch, route))


@pytest.mark.cpu_only
@pytest.mark.parametrize("narrow_table", NARROW_TABLES, indirect=True)
def test_narrow_tile_bounds_and_plan(narrow_table) -> None:
    """A key covers its channel counts from ``C_lo`` through ``C_hi`` and its token counts from ``S_min`` through
    ``S_max`` (both ends included; never a ``TILE_OVERRIDES`` row) on its architecture and route only.  ``plan`` of a
    backward record given its architecture and route then plans 32-vector tiles at 128 threads, ``(32, threads //
    32, ceil(C / vec / 32))``; :func:`cake_jit.channel_split`'s split (or the override's) everywhere else, at
    another thread count, under ``legacy`` and without the architecture or the route.  ``narrow_tiles`` (the
    workspace bound) holds the key's tile count whatever ``S``."""

    checked = 0
    for arch, rows in narrow_table.items():
        for (base, c_lo, c_hi), (s_min, s_max) in rows.items():
            channels = {c_lo - 8, c_lo, (c_lo + c_hi) // 16 * 8, c_hi, c_hi + 8}
            # The key's route and every route that plans by its rows.
            for route in _planning_routes(arch, base):
                for C in sorted(C for C in channels if C > 0):
                    for S in (s_min - 1, s_min, s_max, s_max + 1):
                        checked += _check_narrow_plan(narrow_table, arch, route, C, S)
    assert checked or not narrow_table


@pytest.mark.cpu_only
@pytest.mark.parametrize("narrow_table", NARROW_TABLES, indirect=True)
def test_backward_workspace_covers_every_narrow_row(narrow_table) -> None:
    """The public workspace of a key's channel counts holds the layout of every program of the key's route, and of
    every route that plans by its rows, at the key's tile count (one ticket per tile), whatever ``S``, and is the
    largest layout over every backward program at the largest tile count it may launch with."""

    for arch, rows in narrow_table.items():
        owners = cake_jit.workspace_owners(arch)
        for (route, c_lo, c_hi), (s_min, _) in rows.items():
            for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                nbytes = cake_jit.backward_workspace_bytes(arch, C)
                layouts = [
                    cake_jit.accumulator_layout(
                        record, C, _owner_tiles(record, arch, C)
                    )
                    for record in owners
                ]
                assert nbytes == max(layout["total_bytes"] for layout in layouts), (
                    arch,
                    route,
                    C,
                )
                for kind in _planning_routes(arch, route):
                    for name in cake_jit.route_modules(arch, kind):
                        record = cake_jit.MODULES[name]
                        geometry = record["grid_rule"]["geometry"]
                        tiles = cake_jit.plan(
                            s_min,
                            C,
                            3,
                            geometry=geometry,
                            role="bwd",
                            arch=arch,
                            route=kind,
                        )["n_ch_tiles"]
                        layout = cake_jit.accumulator_layout(record, C, tiles)
                        assert layout["total_bytes"] <= nbytes, (arch, kind, C, name)


# ----------------------------------------------------------------------------------------------------------------
# The forward plan: the deterministic package's forward rule (route, ``S`` and architecture) over the same programs
# from the row's token floor (``forward_narrow_min_s``: ``FORWARD_NARROW_MIN_S`` or ``FORWARD_NARROW_ROW_MIN_S``);
# below it the single tile of the package's 0.2.0 release.

# The forward's narrow rows at 128 threads x 8 channels: C -> (ct_vec, n_sub, n_ch_tiles) on every forward route
# below the row's token floor (:func:`_fwd_row_floor`; in a plan without ``S`` or the architecture, and on an
# architecture without that floor), from that floor up to the width floor of ``FWD_FLOORS``, and from both floors on
# the call's architecture; a call without a route and the legacy rule keep one tile of C / 8 vectors at every ``S``.
# The rows below 72 vectors (C 520 to 568) run their width (or the 32-vector tiles, C 536 and 568) from their row
# floor (``FWD_ROW_FLOORS``), which is their width floor: they never run the 32-vector tiles below a width.
FWD_ONE_TILE_SPLITS = {
    520: (65, 1, 1),
    528: (66, 1, 1),
    536: (67, 1, 1),
    544: (68, 1, 1),
    552: (69, 1, 1),
    560: (70, 1, 1),
    568: (71, 1, 1),
    576: (72, 1, 1),
    640: (80, 1, 1),
    704: (88, 1, 1),
    712: (89, 1, 1),
    768: (96, 1, 1),
}
FWD_NARROW_SPLITS = {C: (32, 4, 3) for C in FWD_ONE_TILE_SPLITS}
FWD_WIDE_SPLITS = {
    520: (13, 9, 5),
    528: (11, 11, 6),
    536: (32, 4, 3),
    544: (17, 7, 4),
    552: (23, 5, 3),
    560: (14, 9, 5),
    568: (32, 4, 3),
    576: (18, 7, 4),
    640: (40, 3, 2),
    704: (32, 4, 3),
    712: (32, 4, 3),
    768: (32, 4, 3),
}
# The rows below this many vectors take ``FWD_ROW_FLOORS`` (``FORWARD_ROW_FLOOR_VECTORS``).
FWD_ROW_FLOOR_VECTORS = 72
# The token count from which the narrow rows from 72 vectors run narrower tiles than one tile, per architecture.
FWD_MIN_S = {"sm_100a": 20480, "sm_103a": 12288, "sm_107a": 16384}
# The token count from which each narrow row below 72 vectors (C 520 to 568) runs narrower tiles, per architecture:
# arch -> {C / 8: S}, the smallest measured token count from which no measured cell ran slower with them.
FWD_ROW_FLOORS = {
    "sm_100a": {65: 16384, 66: 24576, 67: 16384, 68: 16384, 69: 16384, 70: 24576, 71: 12288},
    "sm_103a": {65: 16384, 66: 24576, 67: 16384, 68: 16384, 69: 16384, 70: 24576, 71: 12288},
    "sm_107a": {65: 24576, 66: 24576, 67: 16384, 68: 20480, 69: 24576, 70: 24576, 71: 16384},
}
# The tap dtypes of the forward row tables.
FWD_TAPS = ("f32", "bf16")
# The rows from 72 vectors that take another token floor: arch -> ((C / 8 from which, S, forward routes, taps), ...).
# On sm_103a the ``fwd_bias`` rows of 89 to 96 vectors (C 712 to 768) take 20480 with FP32 taps and 12288 with BF16
# taps; the ``fwd`` and ``fwd_dt`` rows there, and C 704 (88 vectors) on every route, keep the architecture's 12288.
FWD_ROW_MIN_S = {"sm_103a": ((89, 20480, ("fwd_bias",), ("f32",)), (89, 12288, ("fwd_bias",), ("bf16",)))}
# The rows that keep the single tile in a token range at or above their floor: arch -> ((C / 8 from, C / 8 to,
# forward routes, taps, S_lo, S_hi), ...).  On sm_103a the BF16-taps ``fwd_bias`` rows of 89 to 96 vectors.
FWD_SINGLE_TILE_ROWS = {
    "sm_103a": (
        (89, 89, ("fwd_bias",), ("bf16",), 12961, 16383),
        (90, 95, ("fwd_bias",), ("bf16",), 12289, 16383),
        (96, 96, ("fwd_bias",), ("bf16",), 13313, 16383),
    )
}
# The token count from which each width row runs its width, per architecture: arch -> {C / 8: S}; a row below 72
# vectors at its row floor.
FWD_FLOORS = {
    "sm_100a": {65: 16384, 66: 24576, 68: 16384, 69: 16384, 70: 24576, 72: 32768, 80: 24576},
    "sm_103a": {65: 16384, 66: 24576, 68: 16384, 69: 16384, 70: 24576, 72: 32768, 80: 24576},
    "sm_107a": {65: 24576, 66: 24576, 68: 20480, 69: 24576, 70: 24576, 72: 40960, 80: 36864},
}
# The forward workspace's channel tiles of the narrow rows: the largest tile count of the three plans (C 576: 1, 3,
# then 4; C 528: 1, then 6), whatever the token floors.  C 520 to 568 hold more than one tile from 0.2.4 on.
FWD_NARROW_WORKSPACE_TILES = {
    520: 5,
    528: 6,
    536: 3,
    544: 4,
    552: 3,
    560: 5,
    568: 3,
    576: 4,
    640: 3,
    704: 3,
    712: 3,
    768: 3,
}


def _fwd_taps(route: str, bf16_taps: bool = False) -> str:
    """The taps of a forward call on ``route``: FP32 on ``fwd``, BF16 on ``fwd_dt``, the call's on ``fwd_bias``."""

    if route == "fwd_bias":
        return "bf16" if bf16_taps else "f32"
    return "bf16" if route == "fwd_dt" else "f32"


def _fwd_row_floor(arch: str, route: str, channel_vectors: int, bf16_taps: bool = False) -> int | None:
    """The token floor of the forward row of ``channel_vectors`` vectors on forward route ``route`` (and taps) on
    ``arch`` from the literal tables: below 72 vectors its ``FWD_ROW_FLOORS`` floor (``None`` without one); else the
    last ``FWD_ROW_MIN_S`` entry listing the route and taps that the row reaches, else ``FWD_MIN_S``; ``None`` on an
    architecture without a floor."""

    if channel_vectors < FWD_ROW_FLOOR_VECTORS:
        return FWD_ROW_FLOORS.get(arch, {}).get(channel_vectors)
    floor = FWD_MIN_S.get(arch)
    if floor is None:
        return None
    for cv, S, routes, taps in FWD_ROW_MIN_S.get(arch, ()):
        if channel_vectors >= cv and route in routes and _fwd_taps(route, bf16_taps) in taps:
            floor = S
    return floor


def _fwd_gap(arch: str, route: str, channel_vectors: int, bf16_taps: bool = False) -> tuple[int, int] | None:
    """The ``(S_lo, S_hi)`` single-tile range of the forward row from the literal ``FWD_SINGLE_TILE_ROWS``, else
    ``None``."""

    for cv_lo, cv_hi, routes, taps, s_lo, s_hi in FWD_SINGLE_TILE_ROWS.get(arch, ()):
        if cv_lo <= channel_vectors <= cv_hi and route in routes and _fwd_taps(route, bf16_taps) in taps:
            return s_lo, s_hi
    return None


def _fwd_taps_of(route: str) -> tuple[bool, ...]:
    """The ``bf16_taps`` values a forward call of ``route`` may pass: both on ``fwd_bias`` (which reads them), the
    route's own on ``fwd`` (FP32) and ``fwd_dt`` (BF16)."""

    return (False, True) if route == "fwd_bias" else (route == "fwd_dt",)


def _forward_tokens() -> tuple[int, ...]:
    """Token counts on both sides of every forward token floor and single-tile range of either package and of every
    token floor of the forward selection rules, plus a few others."""

    tokens = {1, 2, 1000, 4096, 8192, 16384, 250340, 262144, 500680, 1 << 20}
    for table in (
        cake_jit.FORWARD_WIDTH_FLOORS,
        det_jit.NARROW_WIDTH_FLOORS["fwd"],
        cake_jit.FORWARD_NARROW_ROW_FLOORS,
        det_jit.NARROW_ROW_FLOORS["fwd"],
    ):
        for floors in table.values():
            for S in floors.values():
                tokens |= {S - 1, S, S + 1}
    for S in cake_jit.FORWARD_NARROW_MIN_S.values():
        tokens |= {S - 1, S, S + 1}
    for rows in cake_jit.FORWARD_NARROW_ROW_MIN_S.values():
        for _, S, _, _ in rows:
            tokens |= {S - 1, S, S + 1}
    for rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.values():
        for *_, s_lo, s_hi in rows:
            tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    for registry in (cake_jit, det_jit):
        for record in registry.MODULES.values():
            if record["role"] != "fwd":
                continue
            rule = registry.select_rule(record)
            for S in (rule or {}).get("tokens", {}).values():
                tokens |= {int(S) - 1, int(S)}
    return tuple(sorted(S for S in tokens if S > 0))


@pytest.mark.cpu_only
def test_forward_narrow_rows_are_the_deterministic_forward_rows() -> None:
    """The forward's narrow tables are the deterministic package's forward rows: every forward route, the thread
    count, the widths, the width floors and the row floors of the rows below 72 vectors on every architecture.  The
    rows below 72 vectors are C 520 to 568, the same on every architecture; a width row among them takes its width
    from its row floor (its width floor is its row floor)."""

    forward = tuple(kind for kind in cake_jit.KINDS if cake_jit.route_role(kind) == "fwd")
    assert cake_jit.FORWARD_NARROW_ROUTES == det_jit.NARROW_TILE_ROUTES["fwd"] == forward
    assert cake_jit.FORWARD_NARROW_THREADS == det_jit.NARROW_TILE_THREADS == 128
    widths = {65: 13, 66: 11, 68: 17, 69: 23, 70: 14, 72: 18, 80: 40}
    assert cake_jit.FORWARD_NARROW_WIDTHS == det_jit.NARROW_TILE_WIDTHS["fwd"] == widths
    assert cake_jit.FORWARD_WIDTH_FLOORS == det_jit.NARROW_WIDTH_FLOORS["fwd"] == FWD_FLOORS
    assert cake_jit.FORWARD_NARROW_ROW_FLOORS == det_jit.NARROW_ROW_FLOORS["fwd"] == FWD_ROW_FLOORS
    assert cake_jit.FORWARD_ROW_FLOOR_VECTORS == FWD_ROW_FLOOR_VECTORS
    assert set(FWD_FLOORS) == set(FWD_ROW_FLOORS) == set(ARCHES)
    for cv, width in widths.items():
        assert cv % width == 0 and 0 < width <= cake_jit.FORWARD_NARROW_THREADS, (cv, width)
    for arch in ARCHES:
        rows = FWD_ROW_FLOORS[arch]
        assert sorted(rows) == list(range(65, FWD_ROW_FLOOR_VECTORS)), arch
        for cv, S in rows.items():
            assert type(S) is int and S > 0, (arch, cv, S)
            if cv in widths:
                assert FWD_FLOORS[arch][cv] == S, (arch, cv)
        assert {cv for cv in FWD_FLOORS[arch] if cv < FWD_ROW_FLOOR_VECTORS} == set(rows) & set(widths), arch


@pytest.mark.cpu_only
def test_forward_narrow_token_floors() -> None:
    """``FORWARD_NARROW_MIN_S``, ``FORWARD_NARROW_ROW_MIN_S`` and ``FORWARD_SINGLE_TILE_ROWS`` are the literal tables:
    one positive token floor per architecture of ``ARCHES``; the ``fwd_bias`` rows of sm_103a from 89 vectors (C 712
    to 768) at 20480 with FP32 taps and 12288 with BF16 taps (each row entry ``(cv, S, routes, taps)``: ``routes`` a
    non-empty tuple of forward routes in ``FORWARD_NARROW_ROUTES`` order, ``taps`` of ``FORWARD_TAPS`` in that
    order, no ``(cv, route, taps)`` twice), and the BF16-taps ``fwd_bias`` single-tile ranges there (each ``(cv_lo,
    cv_hi, routes, taps, S_lo, S_hi)``, no ``(cv, route, taps)`` in two).  ``forward_narrow_min_s`` gives every row
    of every forward route and taps its floor (:func:`_fwd_row_floor`: below 72 vectors ``FORWARD_NARROW_ROW_FLOORS``)
    and ``forward_single_tile_range`` its range (:func:`_fwd_gap`); both refuse a route outside
    ``FORWARD_NARROW_ROUTES``; below that floor and inside the range the narrow rows keep one tile and elsewhere from
    the floor they plan as the deterministic forward; ``channel_split`` reads only the row's floor and range on the
    call's route, taps and architecture, and an architecture without a floor keeps the single tile at every ``S``."""

    routes_all = cake_jit.FORWARD_NARROW_ROUTES
    assert cake_jit.FORWARD_TAPS == FWD_TAPS
    assert cake_jit.FORWARD_NARROW_MIN_S == FWD_MIN_S
    for arch, S in cake_jit.FORWARD_NARROW_MIN_S.items():
        assert arch in ARCHES and type(S) is int and S > 0, (arch, S)
    assert cake_jit.FORWARD_NARROW_ROW_MIN_S == FWD_ROW_MIN_S
    for arch, rows in cake_jit.FORWARD_NARROW_ROW_MIN_S.items():
        assert arch in cake_jit.FORWARD_NARROW_MIN_S and type(rows) is tuple and rows, (arch, rows)
        assert [cv for cv, *_ in rows] == sorted(cv for cv, *_ in rows), (arch, rows)
        keys = [(cv, route, taps) for cv, _, routes, row_taps in rows for route in routes for taps in row_taps]
        assert len(keys) == len(set(keys)), (arch, rows)
        for cv, S, routes, taps in rows:
            assert type(cv) is int and 72 <= cv <= 96 and type(S) is int and S > 0, (arch, cv, S)
            assert type(routes) is tuple and routes, (arch, cv, routes)
            assert routes == tuple(route for route in routes_all if route in routes), (arch, cv, routes)
            assert type(taps) is tuple and taps == tuple(t for t in FWD_TAPS if t in taps), (arch, cv, taps)
    assert cake_jit.FORWARD_SINGLE_TILE_ROWS == FWD_SINGLE_TILE_ROWS
    for arch, rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.items():
        assert arch in cake_jit.FORWARD_NARROW_MIN_S and type(rows) is tuple and rows, (arch, rows)
        keys = [
            (cv, route, taps)
            for cv_lo, cv_hi, routes, row_taps, _, _ in rows
            for cv in range(cv_lo, cv_hi + 1)
            for route in routes
            for taps in row_taps
        ]
        assert len(keys) == len(set(keys)), (arch, rows)
        for cv_lo, cv_hi, routes, taps, s_lo, s_hi in rows:
            label = (arch, cv_lo, cv_hi)
            assert type(cv_lo) is int and type(cv_hi) is int and 72 <= cv_lo <= cv_hi <= 96, label
            assert type(routes) is tuple and routes == tuple(r for r in routes_all if r in routes) and routes, label
            assert type(taps) is tuple and taps == tuple(t for t in FWD_TAPS if t in taps) and taps, label
            assert type(s_lo) is int and type(s_hi) is int and 0 < s_lo <= s_hi < cake_jit.INT32_MAX, label
            for cv in range(cv_lo, cv_hi + 1):
                for route in routes:
                    for bf16 in _fwd_taps_of(route):
                        if _fwd_taps(route, bf16) in taps:
                            # A range lies above the row's floor (else part of it would plan as below the floor).
                            assert _fwd_row_floor(arch, route, cv, bf16) < s_lo, (label, cv, route, bf16)
    for arch in ARCHES:
        for route in routes_all:
            for bf16 in (False, True):
                for cv in range(1, 1025):
                    label = (arch, route, bf16, cv)
                    got = cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16)
                    assert got == _fwd_row_floor(arch, route, cv, bf16), label
                    got = cake_jit.forward_single_tile_range(arch, cv, route=route, bf16_taps=bf16)
                    assert got == _fwd_gap(arch, route, cv, bf16), label
            assert cake_jit.forward_narrow_min_s(arch, 96, route=route) == _fwd_row_floor(arch, route, 96), arch
    for route in routes_all:
        assert cake_jit.forward_narrow_min_s(None, 96, route=route) is None
        assert cake_jit.forward_narrow_min_s("sm_90a", 96, route=route) is None
        assert cake_jit.forward_narrow_min_s("sm_90a", 65, route=route) is None
        assert cake_jit.forward_single_tile_range("sm_90a", 96, route=route, bf16_taps=True) is None
    for route in ("bwd", None, "fwd_pair"):
        with pytest.raises(ValueError, match="route must be one of"):
            cake_jit.forward_narrow_min_s("sm_103a", 96, route=route)
        with pytest.raises(ValueError, match="route must be one of"):
            cake_jit.forward_single_tile_range("sm_103a", 96, route=route)
    # The literal floors of the rows the plans below cross: on sm_103a C 704 at 12288 on every route, C 712 and 768
    # at 20480 on ``fwd_bias`` with FP32 taps and at 12288 with BF16 taps and on ``fwd`` / ``fwd_dt``; the rows of C
    # 520 to 568 at their row floors on every route and taps.
    floors = {
        (route, bf16): [
            cake_jit.forward_narrow_min_s("sm_103a", C // 8, route=route, bf16_taps=bf16)
            for C in (520, 568, 576, 640, 704, 712, 768)
        ]
        for route in routes_all
        for bf16 in (False, True)
    }
    plain = [16384, 12288, 12288, 12288, 12288, 12288, 12288]
    assert floors == {
        ("fwd", False): plain,
        ("fwd", True): plain,
        ("fwd_dt", False): plain,
        ("fwd_dt", True): plain,
        ("fwd_bias", False): [16384, 12288, 12288, 12288, 12288, 20480, 20480],
        ("fwd_bias", True): plain,
    }
    gaps = {
        (route, bf16): [
            cake_jit.forward_single_tile_range("sm_103a", C // 8, route=route, bf16_taps=bf16)
            for C in (704, 712, 720, 760, 768)
        ]
        for route in routes_all
        for bf16 in (False, True)
    }
    none5 = [None] * 5
    assert gaps == {
        ("fwd", False): none5,
        ("fwd", True): none5,
        ("fwd_dt", False): none5,
        ("fwd_dt", True): none5,
        ("fwd_bias", False): none5,
        ("fwd_bias", True): [None, (12961, 16383), (12289, 16383), (12289, 16383), (13313, 16383)],
    }
    for arch in ("sm_100a", "sm_107a"):
        for route in routes_all:
            for bf16 in (False, True):
                got = {cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16) for cv in range(72, 97)}
                assert got == {FWD_MIN_S[arch]}, (arch, route, bf16)
                got = {cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16) for cv in range(65, 72)}
                assert got == set(FWD_ROW_FLOORS[arch].values()), (arch, route, bf16)
    geometry = {"threads": 128, "vec": 8}
    for arch in ARCHES:
        for route in routes_all:
            for bf16 in (False, True):  # ``fwd`` and ``fwd_dt`` ignore it (their taps are the route's)
                for C, one_tile in FWD_ONE_TILE_SPLITS.items():
                    S_min = _fwd_row_floor(arch, route, C // 8, bf16)
                    gap = _fwd_gap(arch, route, C // 8, bf16)
                    steps = (12287, 12288, 12960, 12961, 13312, 13313, 14336, 16383, 16384, 20479, 20480, 24575)
                    for S in (0, 1, 4096, 8192, *steps, 24576, 1 << 20):
                        got = cake_jit.channel_split(
                            C, geometry=geometry, role="fwd", route=route, S=S, arch=arch, bf16_taps=bf16
                        )
                        label = (arch, route, bf16, C, S)
                        if S_min is None or S < S_min or (gap is not None and gap[0] <= S <= gap[1]):
                            assert got == one_tile, label
                        else:
                            assert got == det_jit.channel_split(
                                C, geometry=geometry, role="fwd", route=route, S=S, arch=arch
                            ), label
                            assert got in (FWD_NARROW_SPLITS[C], FWD_WIDE_SPLITS[C]), label
                            if C // 8 < FWD_ROW_FLOOR_VECTORS:
                                assert got == FWD_WIDE_SPLITS[C], label
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FORWARD_NARROW_MIN_S", {})
        m.setattr(cake_jit, "FORWARD_NARROW_ROW_FLOORS", {})
        for arch in ARCHES:
            for C, one_tile in FWD_ONE_TILE_SPLITS.items():
                for S in (1, 1 << 20):
                    got = cake_jit.channel_split(C, geometry=geometry, role="fwd", route="fwd", S=S, arch=arch)
                    assert got == one_tile, (arch, C, S)
    # Without the row table every row from 72 vectors takes its architecture's floor: sm_103a's ``fwd_bias`` C 712 and
    # 768 from 12288 with either taps; without the single-tile table the BF16-taps rows narrow inside the ranges too.
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FORWARD_NARROW_ROW_MIN_S", {})
        m.setattr(cake_jit, "FORWARD_SINGLE_TILE_ROWS", {})
        for C in (712, 768):
            for bf16 in (False, True):
                for S, want in (
                    (12287, FWD_ONE_TILE_SPLITS[C]),
                    (12288, FWD_NARROW_SPLITS[C]),
                    (14336, FWD_NARROW_SPLITS[C]),
                ):
                    got = cake_jit.channel_split(
                        C, geometry=geometry, role="fwd", route="fwd_bias", S=S, arch="sm_103a", bf16_taps=bf16
                    )
                    assert got == want, (C, bf16, S)
    # The single-tile table alone decides the BF16-taps ``fwd_bias`` rows inside the ranges (the floors are 12288).
    for C, inside, outside in ((712, (12961, 16383), (12960, 16384)), (768, (13313, 16383), (13312, 16384))):
        for S in inside:
            got = cake_jit.channel_split(
                C, geometry=geometry, role="fwd", route="fwd_bias", S=S, arch="sm_103a", bf16_taps=True
            )
            assert got == FWD_ONE_TILE_SPLITS[C], (C, S)
        for S in outside:
            got = cake_jit.channel_split(
                C, geometry=geometry, role="fwd", route="fwd_bias", S=S, arch="sm_103a", bf16_taps=True
            )
            assert got == FWD_NARROW_SPLITS[C], (C, S)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_forward_plan_is_the_deterministic_forward_plan(arch) -> None:
    """Every forward record plans what the deterministic record it copies plans from the row's floor on ``arch``
    (``forward_narrow_min_s``) outside the row's single-tile range (``forward_single_tile_range``), and below the floor
    and inside the range (and on an architecture without a floor) the route-less plan of the 0.2.0 release, for every
    multiple of 8 up to 8192 channels, every token count of :func:`_forward_tokens` and both tap dtypes on
    ``fwd_bias``: ``plan`` given the record's architecture, route and taps (as ``_launch`` and the KCP forward's
    single-rank main program call it), ``channel_split`` with and without the route, ``S`` and the architecture, and
    the legacy rule.  At 128 x 8 the narrow rows run one tile of C / 8 vectors below the floor, the 32-vector tiles
    from it and their width from the width floor of ``arch`` (the literal tables; the rows below 72 vectors their
    width from the row floor); without a route, without ``S`` or the architecture and under ``legacy`` one tile of C /
    8 vectors stays; both packages refuse a route of the other role and an unknown architecture."""

    tokens = _forward_tokens()
    literal = 0
    for pair_arch, kind, name, det_name in _pairs("fwd"):
        if pair_arch != arch:
            continue
        record, det = cake_jit.MODULES[name], det_jit.MODULES[det_name]
        geometry, det_geometry = record["grid_rule"]["geometry"], det["grid_rule"]["geometry"]
        route = cake_jit.record_route(record, arch)
        assert route == kind == det_jit.record_route(det, arch), (name, det_name)
        for C in range(8, 8193, 8):
            label = (kind, name, C)
            S_min = cake_jit.forward_narrow_min_s(arch, C // int(geometry["vec"]), route=route)
            # The route-less plan, the 0.2.0 plan of every row: that of each call below the floor.
            route_less = det_jit.channel_split(C, geometry=det_geometry, role="fwd")
            for kwargs in (
                {},
                {"route": route},
                {"route": route, "arch": arch},
                {"route": route, "S": tokens[-1]},
                {"arch": arch, "S": tokens[-1]},
            ):
                got = cake_jit.channel_split(C, geometry=geometry, role="fwd", **kwargs)
                assert got == route_less, (label, kwargs)
                if "route" not in kwargs:
                    want = det_jit.channel_split(C, geometry=det_geometry, role="fwd", **kwargs)
                    assert got == want, (label, kwargs)
            kwargs = {"route": route, "arch": arch, "S": tokens[-1]}
            got = cake_jit.channel_split(C, geometry=geometry, role="fwd", **kwargs)
            want = det_jit.channel_split(C, geometry=det_geometry, role="fwd", **kwargs)
            assert got == (route_less if S_min is None else want), (label, kwargs)
            legacy = det_jit.channel_split(C, geometry=det_geometry, legacy=True)
            assert cake_jit.channel_split(C, geometry=geometry, legacy=True) == legacy, label
            got = cake_jit.plan(
                tokens[-1], C, 3, geometry=geometry, role="fwd", legacy=True, arch=arch, route=route
            )
            assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == legacy, label
            for bf16 in _fwd_taps_of(kind):
                cv = C // int(geometry["vec"])
                S_min = cake_jit.forward_narrow_min_s(arch, cv, route=route, bf16_taps=bf16)
                gap = cake_jit.forward_single_tile_range(arch, cv, route=route, bf16_taps=bf16)
                for S in tokens:
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role=record["role"], arch=arch, route=route, bf16_taps=bf16
                    )
                    if S_min is not None and S >= S_min and not (gap is not None and gap[0] <= S <= gap[1]):
                        want = det_jit.plan(
                            S, C, 3, geometry=det_geometry, role=det["role"], route=route, arch=arch
                        )
                    else:
                        want = det_jit.plan(S, C, 3, geometry=det_geometry, role=det["role"])
                        assert (want["ct_vec"], want["n_sub"], want["n_ch_tiles"]) == route_less
                    assert got == want, (label, bf16, S)
        if (geometry["threads"], geometry["vec"]) != (128, 8):
            continue
        for C, narrow in FWD_NARROW_SPLITS.items():
            one_tile = FWD_ONE_TILE_SPLITS[C]
            assert one_tile == (C // 8, 128 // (C // 8), 1)
            assert cake_jit.channel_split(C, geometry=geometry, role="fwd") == one_tile
            assert cake_jit.channel_split(C, geometry=geometry, legacy=True) == one_tile
            assert cake_jit.channel_split(C, geometry=geometry, role="fwd", route=route) == one_tile
            floor = FWD_FLOORS[arch].get(C // 8)
            for bf16 in _fwd_taps_of(kind):
                S_lo = _fwd_row_floor(arch, kind, C // 8, bf16)
                gap = _fwd_gap(arch, kind, C // 8, bf16)
                steps = [(1, one_tile), (S_lo - 1, one_tile), (1 << 20, FWD_WIDE_SPLITS[C])]
                steps += [(S_lo, narrow if floor is None or S_lo < floor else FWD_WIDE_SPLITS[C])]
                if floor is not None:
                    steps += [(floor - 1, narrow if floor - 1 >= S_lo else one_tile)]
                    steps += [(floor, FWD_WIDE_SPLITS[C])]
                # Both sides of the architecture's floor and of every row floor (sm_103a: C 704 from 12288; C 712 and
                # 768 from 20480 on ``fwd_bias`` with FP32 taps, the single tile at 12288 to 20479, and from 12288
                # with BF16 taps and on ``fwd`` / ``fwd_dt``; C 520 to 568 from their row floors).
                for S in sorted({FWD_MIN_S[arch] - 1, FWD_MIN_S[arch], 16384, 20479, 20480, 24575, 24576}):
                    if S < S_lo:
                        steps += [(S, one_tile)]
                    elif floor is None or S < floor:
                        steps += [(S, narrow)]
                # Both ends of a single-tile range and the token counts around them (sm_103a BF16-taps ``fwd_bias``).
                if gap is not None:
                    steps = [(S, want) for S, want in steps if not gap[0] <= S <= gap[1]]
                    steps += [(gap[0], one_tile), (gap[1], one_tile), (gap[0] - 1, narrow), (gap[1] + 1, narrow)]
                for S, want in steps:
                    got = cake_jit.plan(
                        S, C, 3, geometry=geometry, role="fwd", arch=arch, route=route, bf16_taps=bf16
                    )
                    assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == want, (kind, C, bf16, S)
                    literal += 1
    assert literal, arch
    geometry = {"threads": 128, "vec": 8}
    for registry in (cake_jit, det_jit):
        for kwargs in (
            {"role": "fwd", "route": "bwd"},
            {"role": "bwd", "route": "fwd_dt"},
            {"legacy": True, "route": "fwd"},
            {"role": "fwd", "route": "fwd", "S": 1 << 20, "arch": "sm_90a"},
        ):
            with pytest.raises(ValueError):
                registry.channel_split(576, geometry=geometry, **kwargs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_forward_workspace_is_the_deterministic_forward_workspace(arch) -> None:
    """``forward_workspace_words`` is the deterministic package's for every multiple of 8 up to 8192 channels (C 520
    to 568 included, whose rows hold 3 to 6 tiles from their row floors): two int32 counters per channel tile, the
    most over every forward program and every plan of a row whose tiles change at a token floor
    (``FWD_NARROW_WORKSPACE_TILES``: C 576 holds 4 tiles, C 528 6), whatever ``FORWARD_NARROW_MIN_S``,
    ``FORWARD_NARROW_ROW_MIN_S``, ``FORWARD_NARROW_ROW_FLOORS`` and ``FORWARD_SINGLE_TILE_ROWS`` hold (the sizing
    takes no ``S`` and no taps); every forward program's plan at every token count of :func:`_forward_tokens` and
    both tap dtypes fits it, so a buffer of the public size never trips the ``work`` size check."""

    tokens = _forward_tokens()
    records = [
        (kind, cake_jit.MODULES[name])
        for kind in cake_jit.forward_kinds(arch)
        for name in cake_jit.route_modules(arch, kind)
    ]
    assert records
    for _, record in records:
        assert record["workspace_rule"] == {"kind": "tile_counters", "words_per_tile": 2}
    for C in range(8, 8193, 8):
        words = cake_jit.forward_workspace_words(arch, C)
        assert words == det_jit.forward_workspace_words(arch, C), C
        for kind, record in records:
            for bf16 in _fwd_taps_of(kind):
                for S in tokens:
                    tiles = cake_jit.plan(
                        S,
                        C,
                        1,
                        geometry=record["grid_rule"]["geometry"],
                        role="fwd",
                        arch=arch,
                        route=kind,
                        bf16_taps=bf16,
                    )["n_ch_tiles"]
                    assert cake_jit.counter_words(record, tiles) <= words, (kind, C, bf16, S)
    for C, tiles in FWD_NARROW_WORKSPACE_TILES.items():
        assert cake_jit.forward_workspace_words(arch, C) == 2 * tiles, C
    # The size does not read the token floors or ranges: none, or every narrow row's from one token, sizes the same
    # buffer.
    every = {cv: 1 for cv in range(65, FWD_ROW_FLOOR_VECTORS)}
    gaps = {arch: ((72, 96, cake_jit.FORWARD_NARROW_ROUTES, FWD_TAPS, 1, 1 << 20),)}
    for table, row_floors in (({}, {}), ({arch: 1}, {arch: every})):
        for rows in ({}, {arch: ((72, 1 << 20, cake_jit.FORWARD_NARROW_ROUTES, FWD_TAPS),)}):
            for single in ({}, gaps):
                with pytest.MonkeyPatch.context() as m:
                    m.setattr(cake_jit, "FORWARD_NARROW_MIN_S", table)
                    m.setattr(cake_jit, "FORWARD_NARROW_ROW_FLOORS", row_floors)
                    m.setattr(cake_jit, "FORWARD_NARROW_ROW_MIN_S", rows)
                    m.setattr(cake_jit, "FORWARD_SINGLE_TILE_ROWS", single)
                    for C, tiles in FWD_NARROW_WORKSPACE_TILES.items():
                        assert cake_jit.forward_workspace_words(arch, C) == 2 * tiles, (table, rows, single, C)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_forward_main_program_plans_the_single_rank_forward(arch, fake_device) -> None:
    """The KCP two-launch forward's single-rank main program (``kcp._sr_main``) runs the single-rank ``fwd`` route
    record ``selected_modules`` picks with that record's forward plan (its role, route and architecture: the narrow
    rows' single tile below the row's ``forward_narrow_min_s`` floor, their 32-vector tiles from it, the width from
    the width floor) and its counter words, which the forward workspace rule covers; the KCP forward program keeps the
    legacy plan, one tile of C / 8 vectors at C 576 to 768."""

    kcp = cake_package(KEY, "kcp")
    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    for C in (512, 520, 528, 536, 544, 552, 560, 568, 576, 640, 704, 712, 768, 776, 1024):
        S_min = _fwd_row_floor(arch, "fwd", C // 8)
        for S in _forward_tokens():
            name, scalars, grid, words = kcp._sr_main(
                arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE
            )
            assert (name,) == cake_jit.selected_modules(arch, "fwd", C=C, S=S), (C, S)
            record = cake_jit.MODULES[name]
            want = cake_jit.plan(
                S, C, 3, geometry=record["grid_rule"]["geometry"], role="fwd", arch=arch, route="fwd"
            )
            assert scalars == want, (C, S)
            assert grid[0] >= scalars["n_ch_tiles"], (C, S)
            assert words == cake_jit.counter_words(record, want["n_ch_tiles"]), (C, S)
            assert words <= cake_jit.forward_workspace_words(arch, C), (C, S)
            if C in FWD_NARROW_SPLITS and (
                record["grid_rule"]["geometry"] == {"threads": 128, "vec": 8}
            ):
                floor = FWD_FLOORS[arch].get(C // 8)
                split = (scalars["ct_vec"], scalars["n_sub"], scalars["n_ch_tiles"])
                if S_min is None or S < S_min:
                    assert split == FWD_ONE_TILE_SPLITS[C], (C, S)
                else:
                    wide = floor is not None and S >= floor
                    assert split == (FWD_WIDE_SPLITS[C] if wide else FWD_NARROW_SPLITS[C]), (C, S)
    (name,) = kcp_jit.route_modules(arch, kcp.KCP_FWD)
    geometry = kcp_jit.MODULES[name]["grid_rule"]["geometry"]
    for C in FWD_NARROW_SPLITS:
        cv = C // geometry["vec"]
        got = cake_jit.plan(1 << 20, C, 3, geometry=geometry, legacy=True)
        threads = geometry["threads"]
        assert (got["ct_vec"], got["n_sub"], got["n_ch_tiles"]) == (
            min(cv, threads),
            threads // min(cv, threads),
            -(-cv // min(cv, threads)),
        ), C


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_kcp_forward_plan_ignores_taps(arch, fake_device) -> None:
    """The KCP forward's single-rank main program plans on the ``fwd`` route (FP32 taps by the KCP contract), so the
    tap-keyed rows of ``FORWARD_NARROW_ROW_MIN_S`` and ``FORWARD_SINGLE_TILE_ROWS`` (``fwd_bias`` only) never change
    its plan: with both tables emptied every ``(C, S)`` plans the same."""

    kcp = cake_package(KEY, "kcp")
    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    channels = (520, 568, 576, 704, 712, 736, 760, 768)
    planned = {
        (C, S): kcp._sr_main(arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE)
        for C in channels
        for S in _forward_tokens()
    }
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_jit, "FORWARD_NARROW_ROW_MIN_S", {})
        m.setattr(cake_jit, "FORWARD_SINGLE_TILE_ROWS", {})
        for (C, S), want in planned.items():
            assert kcp._sr_main(arch, S=S, C=C, n_frag=3, device_index=FAKE_DEVICE) == want, (C, S)


# ----------------------------------------------------------------------------------------------------------------
# The host plan of a call row (``_ROW_PLANS``): planned by the row's first call, the same values as a per-call plan.

# The SM counts of the fake devices of the sweep, per architecture (B200 / GB300 / R200 boards first: the counts the
# grid-trim rules were measured at).  The default sweep takes the first; the full sweep every one.
PLAN_SWEEP_SMS = {"sm_100a": (148, 160), "sm_103a": (152, 148), "sm_107a": (212, 216)}
# A device index no real device has: the rows the sweep plans never serve a launch.
FAKE_DEVICE = 1 << 20
# CAKE_CONV1D_FULL_PLAN_SWEEP=1 runs the full sweep: every SM count, five document counts and three token orders.
FULL_PLAN_SWEEP = os.environ.get("CAKE_CONV1D_FULL_PLAN_SWEEP") == "1"


def _plan_sweep_channels() -> tuple[int, ...]:
    """``CHANNELS``, rows off the 32-channel grid, and every selection threshold and ``TILE_OVERRIDES`` channel count
    of the registry with the multiple of 8 below it, both ends of every ``NARROW_TILES`` key with their neighbours
    outside it, both ends of every ``FALLBACK_ROWS`` and ``SM_WIDE_ROWS`` key with their neighbours outside it, and
    the forward's narrow rows (C 520 to 768, two of them off the 64-channel grid, C 712, the first row of sm_103a's
    ``FORWARD_NARROW_ROW_MIN_S`` floors of the ``fwd_bias`` route, and every row of ``FORWARD_NARROW_ROW_FLOORS`` and
    ``FORWARD_SINGLE_TILE_ROWS``) with their neighbours."""

    channels = {*CHANNELS, 16, 24, 248, 520, 640, 1048, 1152, 3072}
    channels |= {512, 568, 600, *FWD_NARROW_SPLITS, 720, 736, 760, 776}
    for rows in cake_jit.FORWARD_NARROW_ROW_FLOORS.values():
        channels |= {8 * cv for cv in rows}
    for rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.values():
        for cv_lo, cv_hi, *_ in rows:
            channels |= {8 * cv_lo - 8, 8 * cv_lo, 8 * cv_hi, 8 * cv_hi + 8}
    for record in cake_jit.MODULES.values():
        rule = cake_jit.select_rule(record)
        for C in (rule or {}).get("channels", {}).values():
            channels |= {int(C) - 8, int(C)}
    for rows in cake_jit.TILE_OVERRIDES.values():
        for _, C in rows:
            channels |= {C - 8, C, C + 8}
    for table in (cake_jit.NARROW_TILES, cake_jit.FALLBACK_ROWS, cake_jit.SM_WIDE_ROWS):
        for rows in table.values():
            for _, c_lo, c_hi in rows:
                channels |= {c_lo - 8, c_lo, c_hi, c_hi + 8}
    return tuple(sorted(C for C in channels if C > 0))


def _plan_sweep_tokens() -> tuple[int, ...]:
    """Token counts on both sides of every bound where ``S`` enters a plan: the ``TILE_OVERRIDES``, ``NARROW_TILES``
    and ``FALLBACK_ROWS`` ranges and the ``SM_WIDE_ROWS`` segments, the forward's ``FORWARD_NARROW_MIN_S``,
    ``FORWARD_NARROW_ROW_MIN_S``, ``FORWARD_NARROW_ROW_FLOORS`` and ``FORWARD_WIDTH_FLOORS`` floors and
    ``FORWARD_SINGLE_TILE_ROWS`` ranges, the grid-trim minimums and the selection rules' token floors, plus a few
    others."""

    tokens = {1, 2, 63, 64, 1000, 250340, 1 << 20}
    for table in (cake_jit.FORWARD_WIDTH_FLOORS, cake_jit.FORWARD_NARROW_ROW_FLOORS):
        for floors in table.values():
            for S in floors.values():
                tokens |= {S - 1, S}
    for S in cake_jit.FORWARD_NARROW_MIN_S.values():
        tokens |= {S - 1, S}
    for rows in cake_jit.FORWARD_NARROW_ROW_MIN_S.values():
        for _, S, _, _ in rows:
            tokens |= {S - 1, S}
    for rows in cake_jit.FORWARD_SINGLE_TILE_ROWS.values():
        for *_, s_lo, s_hi in rows:
            tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    for table in (cake_jit.TILE_OVERRIDES, cake_jit.NARROW_TILES, cake_jit.FALLBACK_ROWS):
        for rows in table.values():
            for value in rows.values():
                s_min, s_max = value[-2:]
                tokens |= {s_min - 1, s_min, s_max, s_max + 1}
    for rows in cake_jit.SM_WIDE_ROWS.values():
        for segments in rows.values():
            for s_lo, s_hi, _ in segments:
                tokens |= {s_lo - 1, s_lo, s_hi, s_hi + 1}
    tokens |= {cake_jit.TILE_ALIGNED_MIN_S - 1, cake_jit.TILE_ALIGNED_MIN_S}
    for classes in cake_jit.TILE_ALIGNED_CLASSES.values():
        for s_min, _ in classes.values():
            tokens |= {s_min - 1, s_min}
    for record in cake_jit.MODULES.values():
        rule = cake_jit.select_rule(record)
        for S in (rule or {}).get("tokens", {}).values():
            tokens |= {int(S) - 1, int(S)}
    return tuple(sorted(S for S in tokens if S > 0))


@pytest.fixture
def fake_device(monkeypatch):
    """Plan on :data:`FAKE_DEVICE` of a given architecture and SM count: ``use(arch, sms)`` points ``cake_jit``'s
    device queries at it and empties ``_ROW_PLANS``; the rows planned are dropped again after the test."""

    def use(arch: str, sms: int) -> None:
        monkeypatch.setattr(cake_jit, "device_arch", lambda device_index: arch)
        monkeypatch.setattr(cake_jit, "sm_count", lambda device_index: sms)
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()

    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    yield use
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()


def _per_call(arch: str, kind: str, S: int, C: int, num_docs: int, gm: int, policy, bf16_taps: bool = False):
    """What a call plans without ``_ROW_PLANS`` (``_route`` then ``_launch``), or the error it raises."""

    try:
        names = pkg._route(arch, kind, needs=pkg._route_needs(policy), C=C, S=S)
        scalars, grid = pkg._launch(
            cake_jit.MODULES[names[0]],
            S,
            C,
            num_docs,
            device_index=FAKE_DEVICE,
            arch=arch,
            bwd_grid_mult=gm,
            bf16_taps=bf16_taps,
        )
        return tuple(names), pkg._scalars(scalars), tuple(grid)
    except (ValueError, RuntimeError) as exc:
        return type(exc).__name__, str(exc)


def _row_planned(arch: str, kind: str, S: int, C: int, num_docs: int, gm: int, policy, bf16_taps: bool = False):
    """What a call plans from its ``_ROW_PLANS`` row (``_call_plan``), or the error it raises."""

    try:
        names, record, scalars, grid, _ = pkg._call_plan(
            arch,
            kind,
            S,
            C,
            num_docs,
            device_index=FAKE_DEVICE,
            policy=policy,
            bwd_grid_mult=gm,
            bf16_taps=bf16_taps,
        )
        assert record is cake_jit.MODULES[names[0]]
        assert tuple(scalars) == cake_jit.PLAN_SCALARS
        assert all(type(value) is int for value in scalars.values())
        return tuple(names), dict(scalars), tuple(grid)
    except (ValueError, RuntimeError) as exc:
        return type(exc).__name__, str(exc)


def _sweep_row_plans(arch: str) -> int:
    """Every route of ``arch`` x the sweep's channel counts, token counts (in each order of the sweep, the row
    planned by the order's first call), document counts, backward grid multiples and (on ``fwd_bias``) both tap
    dtypes: the row plan equals the per-call plan, errors included.  Returns the number of calls checked."""

    channels, tokens = _plan_sweep_channels(), _plan_sweep_tokens()
    orders = [tokens]
    if FULL_PLAN_SWEEP:
        half = len(tokens) // 2
        orders += [tokens[::-1], tokens[half:] + tokens[:half]]
    docs = (1, 2, 4, 5, 1000) if FULL_PLAN_SWEEP else (1, 1000)
    checked = 0
    for order in orders:
        pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
        for kind in cake_jit.ROUTES[arch]:
            backward = cake_jit.route_role(kind) == "bwd"
            policy = "fp32" if kind.endswith("fp32g") else "bf16" if backward else None
            for gm in pkg.BWD_GRID_MULTS if backward else (1,):
                for bf16 in (False,) if backward else _fwd_taps_of(kind):
                    for C in channels:
                        for S in order:
                            for num_docs in docs:
                                want = _per_call(arch, kind, S, C, num_docs, gm, policy, bf16)
                                got = _row_planned(arch, kind, S, C, num_docs, gm, policy, bf16)
                                assert got == want, (arch, kind, C, S, num_docs, gm, bf16)
                                checked += 1
        assert len(pkg._ROW_PLANS) <= pkg._ROW_PLANS_MAX
    return checked


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_match_the_per_call_plan(arch, fake_device, record_property) -> None:
    """A call planned from its ``_ROW_PLANS`` row launches the programs, plan scalars and grid a per-call plan
    (``_route`` + ``_launch``) gives, and raises the same errors, on fake devices of the architecture: every route,
    channel counts at every selection threshold, the forward's narrow rows and ``C % 32`` rows (whose grid
    ``launch_grid`` may trim by ``S``), token counts at every ``TILE_OVERRIDES`` / ``NARROW_TILES`` bound, forward
    token floor and width floor, grid-trim minimum and token floor, document counts and every ``bwd_grid_mult``.  ``CAKE_CONV1D_FULL_PLAN_SWEEP=1`` runs the full sweep."""

    checked = 0
    for sms in PLAN_SWEEP_SMS[arch][: None if FULL_PLAN_SWEEP else 1]:
        fake_device(arch, sms)
        checked += _sweep_row_plans(arch)
    record_property("row_plan_calls_checked", checked)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_override_table(
    arch, fake_device, monkeypatch
) -> None:
    """A row planned before ``TILE_OVERRIDES`` changes keeps its plan until
    :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties the cache; after it, with the rows of every architecture on ``arch``, every row's calls on both
    sides of its token range plan as a per-call plan does, the override's tiles inside the range, on the row's route
    and on every route that plans by its rows."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    shipped = dict(cake_jit.TILE_OVERRIDES.get(arch, {}))
    rows = {
        key: value
        for table in cake_jit.TILE_OVERRIDES.values()
        for key, value in table.items()
    }

    def policy_of(route: str) -> str:
        return "fp32" if route.endswith("fp32g") else "bf16"

    planned = {
        (route, C): _row_planned(arch, route, value[1], C, 3, 1, policy_of(route))
        for (base, C), value in rows.items()
        if base in cake_jit.ROUTES[arch]
        for route in _planning_routes(arch, base)
    }
    monkeypatch.setitem(cake_jit.TILE_OVERRIDES, arch, rows)
    changed = 0
    for (route, C), before in planned.items():
        _, s_min, _ = rows[(TILE_ROUTE.get(route, route), C)]
        now = _per_call(arch, route, s_min, C, 3, 1, policy_of(route))
        assert _row_planned(arch, route, s_min, C, 3, 1, policy_of(route)) == before
        changed += now != before
    assert changed or rows.items() <= shipped.items(), arch
    assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
    checked = 0
    for (base, C), (ct_vec, s_min, s_max) in rows.items():
        if base not in cake_jit.ROUTES[arch]:
            continue
        for route in _planning_routes(arch, base):
            for S in (s_min - 1, s_min, s_max, s_max + 1):
                for gm in pkg.BWD_GRID_MULTS:
                    want = _per_call(arch, route, S, C, 3, gm, policy_of(route))
                    got = _row_planned(arch, route, S, C, 3, gm, policy_of(route))
                    assert got == want, (route, C, S, gm)
                    if s_min <= S <= s_max and isinstance(got[1], dict):
                        assert got[1]["ct_vec"] == ct_vec, (route, C, S, gm)
                    checked += 1
    assert checked


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_narrow_table(
    arch, fake_device, monkeypatch
) -> None:
    """A row planned before ``NARROW_TILES`` changes keeps its plan until
    :func:`cake_causal_conv1d_nondet_train_plan_cache_clear` empties the cache; after it, with :data:`SYNTHETIC_NARROW_TILES` in place, every key's calls at its channel ends and an
    inner channel count, on both sides of its token range and at every ``bwd_grid_mult``, plan as a per-call plan
    does (the 32-vector tiles inside the range, the side-stream and whole-rank grid trims included), on fake
    devices of the architecture's SM counts, on the key's route and on every route that plans by its rows."""

    rows = SYNTHETIC_NARROW_TILES[arch]

    def policy_of(route: str) -> str:
        return "fp32" if route.endswith("fp32g") else "bf16"

    def planned(route: str, S: int, C: int, gm: int = 1):
        return _row_planned(arch, route, S, C, 3, gm, policy_of(route))

    checked = 0
    for sms in PLAN_SWEEP_SMS[arch]:
        fake_device(arch, sms)
        before = {
            (route, value[0], key[1]): planned(route, value[0], key[1])
            for key, value in rows.items()
            for route in _planning_routes(arch, key[0])
        }
        with monkeypatch.context() as m:
            m.setattr(cake_jit, "NARROW_TILES", SYNTHETIC_NARROW_TILES)
            for (route, S, C), plan_before in before.items():
                assert planned(route, S, C) == plan_before, (sms, route, S, C)
            assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
            for (base, c_lo, c_hi), (s_min, s_max) in rows.items():
                for route in _planning_routes(arch, base):
                    for C in sorted({c_lo, (c_lo + c_hi) // 16 * 8, c_hi}):
                        for S in (s_min - 1, s_min, s_max, s_max + 1):
                            ct_vec = cake_jit.tile_override(arch, route, C, S) or 32
                            for gm in pkg.BWD_GRID_MULTS:
                                want = _per_call(arch, route, S, C, 3, gm, policy_of(route))
                                got = planned(route, S, C, gm)
                                assert got == want, (arch, sms, route, C, S, gm)
                                if s_min <= S <= s_max and isinstance(got[1], dict):
                                    assert got[1]["ct_vec"] == ct_vec, (route, C, S, gm)
                                checked += 1
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    assert checked


# Forward token floors the replaced-floor test plans with on every architecture: one token, below, at and between the
# width floors of ``FWD_FLOORS`` and above all of them.
SYNTHETIC_FORWARD_MIN_S = (1, 8192, 24576, 32768, 36864, 65536)
# A row table the replaced-floor test plans with besides the architecture's own: on ``fwd_dt`` and ``fwd_bias`` the
# rows from 80 vectors (C 640 to 768) of either taps from a floor at the lower width floor, and on ``fwd`` and
# ``fwd_bias`` the FP32-taps rows from 89 vectors (C 712 to 768) from a floor between the width floors (the last
# entry of a route and taps wins: FP32-taps ``fwd_bias`` C 712 to 768 from 28672).
SYNTHETIC_FORWARD_ROW_MIN_S = (
    (80, 24576, ("fwd_dt", "fwd_bias"), ("f32", "bf16")),
    (89, 28672, ("fwd", "fwd_bias"), ("f32",)),
)
# A single-tile table the replaced-floor test plans with besides the architecture's own: BF16-taps ``fwd_bias`` and
# ``fwd_dt`` rows of 72 to 89 vectors (C 576 to 712) keep the single tile from 30000 to 40000 tokens.
SYNTHETIC_FORWARD_SINGLE_TILE_ROWS = ((72, 89, ("fwd_dt", "fwd_bias"), ("bf16",), 30000, 40000),)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_row_plans_follow_a_replaced_forward_floor(arch, fake_device, monkeypatch) -> None:
    """A forward row planned before ``FORWARD_NARROW_MIN_S``, ``FORWARD_NARROW_ROW_MIN_S`` or
    ``FORWARD_SINGLE_TILE_ROWS`` changes keeps its plan until :func:`cake_causal_conv1d_nondet_train_plan_cache_clear`
    empties the cache (a row whose program selection reads ``S`` holds no plan and plans every call); after it, with
    each floor of :data:`SYNTHETIC_FORWARD_MIN_S` on ``arch`` and either ``arch``'s row and single-tile tables or the
    synthetic ones, every forward route's calls of the narrow rows and their neighbours, of both tap dtypes, on both
    sides of the row's floor (``forward_narrow_min_s``: the floor, the row table's, or below 72 vectors the row
    floor), of the width floors and of the single-tile ranges plan as a per-call plan does: the deterministic
    forward's tiles from the row's floor outside its range, the route-less single tile below it and inside it.  The
    row cache keeps the two tap dtypes apart."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    channels = (512, 520, 528, 568, 576, 600, 640, 704, 712, 768, 776)
    tables = (
        (cake_jit.FORWARD_NARROW_ROW_MIN_S.get(arch, ()), cake_jit.FORWARD_SINGLE_TILE_ROWS.get(arch, ())),
        (SYNTHETIC_FORWARD_ROW_MIN_S, SYNTHETIC_FORWARD_SINGLE_TILE_ROWS),
    )
    checked = 0
    for S_min in SYNTHETIC_FORWARD_MIN_S:
        for rows, gaps in tables:
            pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
            before = {
                (kind, C, bf16): _row_planned(arch, kind, S_min, C, 3, 1, None, bf16)
                for kind in cake_jit.forward_kinds(arch)
                for C in channels
                for bf16 in _fwd_taps_of(kind)
            }
            with monkeypatch.context() as m:
                m.setitem(cake_jit.FORWARD_NARROW_MIN_S, arch, S_min)
                m.setitem(cake_jit.FORWARD_NARROW_ROW_MIN_S, arch, rows)
                m.setitem(cake_jit.FORWARD_SINGLE_TILE_ROWS, arch, gaps)
                for (kind, C, bf16), plan_before in before.items():
                    sides = pkg._ROW_PLANS.get((arch, kind, C, FAKE_DEVICE, 1, bf16))
                    if sides is None:  # the call raised before planning a row
                        assert isinstance(plan_before[1], str), (kind, C, plan_before)
                        continue
                    (row,) = [row for s_lo, s_hi, row, _ in sides if s_lo <= S_min <= s_hi]
                    if row is None:  # the selection reads ``S``: every call is planned
                        assert cake_jit.tokens_select(arch, kind, C), (kind, C)
                        continue
                    assert _row_planned(arch, kind, S_min, C, 3, 1, None, bf16) == plan_before, (kind, C, bf16)
                assert pkg.cake_causal_conv1d_nondet_train_plan_cache_clear() >= 1
                bounds = {S_min, *FWD_FLOORS[arch].values(), *FWD_ROW_FLOORS[arch].values(), *(S for _, S, *_ in rows)}
                bounds |= {S for *_, s_lo, s_hi in gaps for S in (s_lo, s_hi + 1)}
                tokens = sorted({1, 1 << 20, *(S + d for S in bounds for d in (-1, 0, 1))} - {0})
                for kind in cake_jit.forward_kinds(arch):
                    for bf16 in _fwd_taps_of(kind):
                        taps = _fwd_taps(kind, bf16)
                        for C in channels:
                            for S in tokens:
                                want = _per_call(arch, kind, S, C, 3, 1, None, bf16)
                                got = _row_planned(arch, kind, S, C, 3, 1, None, bf16)
                                label = (S_min, rows, gaps, kind, bf16, C, S)
                                assert got == want, label
                                assert isinstance(got[1], dict), (label, got)
                                geometry = cake_jit.MODULES[got[0][0]]["grid_rule"]["geometry"]
                                cv = C // int(geometry["vec"])
                                if cv < FWD_ROW_FLOOR_VECTORS:
                                    row_floor = FWD_ROW_FLOORS[arch].get(cv)
                                else:
                                    row_floor = S_min
                                    for cv_from, S_row, row_routes, row_taps in rows:
                                        if cv >= cv_from and kind in row_routes and taps in row_taps:
                                            row_floor = S_row
                                assert cake_jit.forward_narrow_min_s(arch, cv, route=kind, bf16_taps=bf16) == (
                                    row_floor
                                ), label
                                gap = next(
                                    (
                                        (s_lo, s_hi)
                                        for cv_lo, cv_hi, row_routes, row_taps, s_lo, s_hi in gaps
                                        if cv_lo <= cv <= cv_hi and kind in row_routes and taps in row_taps
                                    ),
                                    None,
                                )
                                inside = gap is not None and gap[0] <= S <= gap[1]
                                if row_floor is not None and S >= row_floor and not inside:
                                    split = det_jit.channel_split(
                                        C, geometry=geometry, role="fwd", route=kind, S=S, arch=arch
                                    )
                                else:
                                    split = det_jit.channel_split(C, geometry=geometry, role="fwd")
                                assert _split3(got[1]) == split, label
                                checked += 1
    pkg.cake_causal_conv1d_nondet_train_plan_cache_clear()
    assert checked


# ----------------------------------------------------------------------------------------------------------------
# The public names: the mode word in every name, no mode keyword.

PUBLIC = (
    "cake_causal_conv1d_nondet_train_forward",
    "cake_causal_conv1d_nondet_train_backward",
    "cake_causal_conv1d_nondet",
    "cake_causal_conv1d_nondet_train_forward_workspace",
    "cake_causal_conv1d_nondet_train_forward_workspace_bytes",
    "cake_causal_conv1d_nondet_train_backward_workspace",
    "cake_causal_conv1d_nondet_train_backward_workspace_bytes",
)
# The names before the mode word joined them: removed, with no alias.
OLD_NAMES = (
    "plan_cache_clear",
    "cake_causal_conv1d",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "is_cake_causal_conv1d_train_supported",
)


@pytest.mark.cpu_only
def test_public_names_carry_the_mode_and_take_no_mode_keyword() -> None:
    """Every public name of ``__all__`` but the constants carries the mode word ``nondet``; no
    entry point and no parameter of the autograd function's forward is a ``deterministic`` keyword; the names before
    the rename are gone (no alias)."""

    for name in PUBLIC:
        assert name in pkg.__all__, name
        assert (
            "deterministic" not in inspect.signature(getattr(pkg, name)).parameters
        ), name
    assert "CakeCausalConv1dNondetFunction" in pkg.__all__
    assert (
        "deterministic"
        not in inspect.signature(pkg.CakeCausalConv1dNondetFunction.forward).parameters
    )
    for name in pkg.__all__:
        assert name.isupper() or "nondet" in name.lower(), name
    for name in OLD_NAMES:
        assert name not in pkg.__all__ and not hasattr(pkg, name), name


@pytest.mark.cpu_only
def test_a_mode_keyword_is_an_ordinary_type_error() -> None:
    """``deterministic=`` (either value) is Python's ordinary ``TypeError`` for an unexpected keyword at every entry
    point, raised before any argument, device or rendering check, so it needs neither a device nor ``build.json``;
    the autograd function takes no positional mode argument after ``dweight_sum`` (and a mode value in the
    ``dweight_sum`` position is a ``TypeError`` too)."""

    x = torch.empty(8, 16, dtype=torch.bfloat16)
    w = torch.empty(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    cpu = torch.device("cpu")
    calls = {
        "forward": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu, **d),
        "backward": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x, **d),
        "autograd": lambda **d: pkg.cake_causal_conv1d_nondet(x, w, cu, **d),
        "fwd_ws": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward_workspace(8, 16, cpu, **d),
        "fwd_ws_bytes": lambda **d: pkg.cake_causal_conv1d_nondet_train_forward_workspace_bytes(8, 16, cpu, **d),
        "bwd_ws": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward_workspace(8, 16, cpu, **d),
        "bwd_ws_bytes": lambda **d: pkg.cake_causal_conv1d_nondet_train_backward_workspace_bytes(8, 16, cpu, **d),
    }  # fmt: skip
    for name, call in calls.items():
        for value in (True, False):
            with pytest.raises(TypeError) as info:
                call(deterministic=value)
            assert "unexpected keyword argument 'deterministic'" in str(info.value), (
                name
            )
    with pytest.raises(TypeError):
        pkg.CakeCausalConv1dNondetFunction.apply(x, w, cu, None, "bf16", 1, False)
    with pytest.raises(TypeError):
        pkg.CakeCausalConv1dNondetFunction.apply(x, w, cu, None, "bf16", 1, "fp32", False)


@pytest.mark.cpu_only
def test_alert_helper_in_both_torch_modes(monkeypatch) -> None:
    """The backend's own non-determinism alert (no private PyTorch hook: ``torch._C._alert_not_deterministic`` is
    absent from some PyTorch builds, so it is removed here): silent with torch's deterministic algorithms off, a
    ``RuntimeError`` naming the entry and the deterministic package when on, a ``UserWarning`` with the same naming
    under ``warn_only=True``.  The KCP module uses the same helper."""

    conv_kcp = cake_package(KEY, "kcp")

    assert conv_kcp._alert_nondeterministic is pkg._alert_nondeterministic
    monkeypatch.delattr(torch._C, "_alert_not_deterministic", raising=False)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(False)
        with warnings_recorded() as caught:
            pkg._alert_nondeterministic("probe_entry")
        assert caught == []
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError,
            match="probe_entry does not have a deterministic implementation",
        ) as info:
            pkg._alert_nondeterministic("probe_entry")
        assert "cake_pplx.cake_causal_conv1d_det_train" in str(info.value)
        assert "warn_only=True" in str(info.value)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning,
            match="probe_entry does not have a deterministic implementation",
        ) as record:
            pkg._alert_nondeterministic("probe_entry")
        assert any(
            "cake_pplx.cake_causal_conv1d_det_train" in str(w.message) for w in record
        )
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


@pytest.mark.cpu_only
def test_backward_entries_alert_before_any_device_work(monkeypatch) -> None:
    """Under torch's deterministic mode the single-rank backward alerts after argument validation and before any
    allocation or launch, and the forward does not alert.  Checked without a device: zero-token CPU tensors with the
    device / placement checks stubbed (the zero-token path returns before any device query)."""

    monkeypatch.setattr(pkg, "_device_arch", lambda device: "sm_100a")
    monkeypatch.setattr(pkg, "_check_placement", lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(pkg, "run_module", lambda *a, **k: launched.append(a))
    x = torch.empty(0, 16, dtype=torch.bfloat16)
    w = torch.zeros(16, 4, dtype=torch.float32)
    cu = torch.tensor([0, 0], dtype=torch.int32)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(
            RuntimeError, match="cake_causal_conv1d_nondet_train_backward does not have"
        ):
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
        # The forward runs the deterministic forward programs: no alert.
        y = pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu)
        assert tuple(y.shape) == (0, 16)
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(
            UserWarning, match="cake_causal_conv1d_nondet_train_backward does not have"
        ):
            dx, dw = pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
        assert tuple(dx.shape) == (0, 16) and torch.equal(dw, torch.zeros_like(dw))
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    assert launched == []


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ARCHES)
def test_backward_entry_passes_its_row_to_pdl_enabled(arch, fake_device, monkeypatch) -> None:
    """The backward entry passes its route, ``C`` and ``S`` to ``pdl_enabled``: its launch takes ``enable_pdl`` 0 at
    the rows of :data:`PDL_OFF_ROWS_OF_RECORD`, eagerly and inside a capture, and the tables' value elsewhere (one
    token below a key, every row of an architecture without keys); a call of the bias route at a key's row and every
    forward launch at those rows take the tables' value.  Checked without a device: meta tensors, a fake device of
    the architecture, the capture query, the device index, the dweight alignment check and the workspace helpers
    stubbed, and each launch's ``enable_pdl`` recorded at ``run_module`` (nothing launches)."""

    fake_device(arch, PLAN_SWEEP_SMS[arch][0])
    monkeypatch.setattr(pkg, "_device_arch", lambda device: arch)
    monkeypatch.setattr(pkg, "_device_index", lambda device: FAKE_DEVICE)
    monkeypatch.setattr(pkg, "_dweight_aligned", lambda dw: dw)
    monkeypatch.setattr(pkg, "_forward_work", lambda *args, **kwargs: None)
    monkeypatch.setattr(pkg, "_backward_accumulators", lambda *args, **kwargs: (None, None))
    launched = []
    monkeypatch.setattr(
        pkg, "run_module", lambda name, values, **kwargs: launched.append((name, values["enable_pdl"]))
    )
    capture = {"on": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capture["on"])
    meta = torch.device("meta")
    rows = PDL_OFF_ROWS_OF_RECORD.get(arch, {})
    cases = {(250340, 1016)}
    for (route, c_lo, c_hi), (s_min, _s_max) in rows.items():
        assert route == "bwd", route
        cases |= {(S, C) for C in (c_lo, c_hi) for S in (s_min - 1, s_min, max(s_min, 250340))}
    turned_off = 0
    for S, C in sorted(cases):
        x = torch.empty((S, C), dtype=torch.bfloat16, device=meta)
        w = torch.empty((C, 4), dtype=torch.float32, device=meta)
        b = torch.empty((C,), dtype=torch.float32, device=meta)
        cu = torch.empty((2,), dtype=torch.int32, device=meta)
        for capturing in (False, True):
            capture["on"] = capturing
            table = int((cake_jit.PDL_GRAPH if capturing else cake_jit.PDL_EAGER)[arch])
            want = _pdl_off_expected(arch, "bwd", C, S, capturing)
            assert want == cake_jit.pdl_enabled(arch, capturing=capturing, kind="bwd", C=C, S=S)
            label = (S, C, capturing)
            launched.clear()
            pkg.cake_causal_conv1d_nondet_train_forward(x, w, cu)
            forward = list(launched)
            assert forward and all(name in cake_jit.route_modules(arch, "fwd") for name, _ in forward), label
            assert all(value == table for _, value in forward), (label, forward)
            launched.clear()
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x)
            ((name, value),) = launched
            assert name in cake_jit.route_modules(arch, "bwd"), (label, name)
            assert value == want, (label, name, value, want)
            launched.clear()
            pkg.cake_causal_conv1d_nondet_train_backward(x, w, cu, x, bias=b)
            ((name, value),) = launched
            assert name in cake_jit.route_modules(arch, "bwd_bias"), (label, name)
            assert value == table, (label, name, value, table)
            turned_off += want != table
    assert bool(turned_off) is bool(rows), (arch, turned_off)


@pytest.mark.cpu_only
def test_unrendered_package_reports_unsupported() -> None:
    assert cake_jit.is_rendered() is (PACKAGE_DIR / "build.json").is_file()
    if not cake_jit.is_rendered():
        assert (
            pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cpu"))
            is False
        )
        with pytest.raises(RuntimeError):
            cake_build.manifest()


# ----------------------------------------------------------------------------------------------------------------
# Wiring.


@pytest.mark.cpu_only
def test_package_wiring() -> None:
    assert (
        pkg.__name__ == "cake_pplx._backends.cake_causal_conv1d_nondet_train"
    )
    # ``cake_pplx`` and ``cake_pplx._backends`` are PEP 420 namespace levels shared by every family: neither carries
    # an ``__init__.py``.
    assert not (REPO / "cake_pplx" / "__init__.py").exists()
    assert not (REPO / "cake_pplx" / "_backends" / "__init__.py").exists()
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    package_data = data["tool"]["setuptools"]["package-data"]
    # One family-wide rule ships every backend package's ``build.json`` and ``csrc/`` tree.
    assert {"build.json", "csrc/**/*"} <= set(package_data["*"]), package_data


@pytest.mark.cpu_only
def test_build_manifest_lists_every_program() -> None:
    """Once rendered: ``build.json`` names this package, carries exactly one entry per (program, architecture) of
    the single-rank and KCP registries, each with its kernel and binding sources under ``csrc/`` and their digests,
    IEEE division / square root and no fast math; no record is a placeholder any more."""

    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    referenced = set()
    for registry in (cake_jit, kcp_jit):
        for name, record in registry.MODULES.items():
            assert "TODOEXPORT" not in name, name
            assert set(record["arches"]) <= set(manifest["arch_flags"]), name
            for arch in ARCHES:
                library = f"{name}_{arch}"
                if arch not in record["arches"]:
                    assert library not in libraries, library
                    continue
                entry = libraries[registry.library_name(name, arch)]
                assert entry["arch"] == arch and entry["entry"] == "run", library
                assert entry["sources"] == [
                    f"csrc/{name}_kernel.cu",
                    f"csrc/{name}_binding.cu",
                ]
                assert "--use_fast_math" not in entry["nvcc_flags"], library
                assert {"--prec-div=true", "--prec-sqrt=true", "--ftz=false"} <= set(
                    entry["nvcc_flags"]
                )
                for source in entry["sources"]:
                    path = PACKAGE_DIR / source
                    assert path.is_file(), source
                    assert (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        == entry["sha256"][source]
                    ), source
                referenced.add(library)
    assert referenced == set(libraries)


def _manifest_checker():
    """``scripts/check_cake_build_manifests.py`` of this checkout, loaded as a module."""

    path = REPO / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cpu_only
def test_manifest_checker_lists_the_package() -> None:
    """The build-manifest checker lists this package (its ``cake_jit.py`` and the KCP registry ``kcp_jit.py``) and
    its vendored gate finds the package's suite key; once rendered, the in-tree checks report nothing."""

    checker = _manifest_checker()
    try:
        path = str(PACKAGE_DIR.relative_to(REPO))
    except ValueError:
        pytest.skip(
            f"the package is imported from {PACKAGE_DIR}, outside this checkout"
        )
    entries = [p for p in checker.PACKAGES if p.name == pkg.__name__]
    assert len(entries) == 1, entries
    (entry,) = entries
    assert (entry.path, entry.loader) == (path, "cake_jit.py")
    assert entry.refs is checker.refs_with_kcp
    assert checker.suite_key(REPO, entry) == KEY
    if not cake_jit.is_rendered():
        pytest.skip("build.json is not rendered yet")
    assert checker.run_in_tree(REPO, [entry]) == []
