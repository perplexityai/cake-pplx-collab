# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION
"""Device slices of the ragged MoE grouped GEMM packages: the checks the deterministic and the
non-deterministic suites share (``tests/experimental/test_cake_moe_grouped_gemm_{det,nondet}_train_device_slice.py``).

A device slice is a call with an ``m_indptr`` row table of ``E + 1`` entries whose first entry is
above 0: groups ``[e0, e1)`` of a whole row table, ``m_indptr[e0 : e1 + 1]`` (a view of the device
table) with ``w[e0:e1]``, over the whole ``x`` / ``g``.  The kernels read the rows
``[m_indptr[0], m_indptr[-1])`` in absolute row numbers and neither read nor write any other row;
the host binds the lead twin of every route it launches (``cake_jit.lead_route``) and plans the
launch from the shapes alone, ``sum_m = x.shape[0]``.

Each slice is checked against two references (bits as integer words, :func:`assert_same_bits`):

* reference 1: the ``[E]`` call over the slice's rows moved to row 0 of zero tensors of the same
  row count, with the ends ``m_indptr[1:] - m_indptr[0]``: the same ``sum_m``, so the same plan and
  stages (:func:`assert_lead`).  The slice gives its bits, always (the bit gate);
* reference 2: the ``[E]`` call over the slice's rows alone (``sum_m`` the slice's rows).  Where
  its plan and stages are the slice's (:attr:`Slice.plan_equal`) the slice gives its bits too;
  where they differ (a shape over a plan threshold) only reference 1 binds, and the difference is
  recorded.  The whole-empty slice of an accumulating weight gradient is the one exception of
  values: reference 2 launches nothing and leaves a ``-0.0`` of ``out`` as it is, while the slice
  launches over its ``sum_m`` rows (:func:`empty_accumulate_word`).

The rows of ``x`` / ``g`` outside the slice hold a signalling NaN no launch writes
(:data:`SENTINEL`), so a launch that read them would not give the references' bits, and the rows of
``Y`` / ``dX`` outside the slice start as that NaN and must keep it (:func:`assert_untouched`).

:func:`ensure_lead_registry` stands in for the lead twins on a registry rendered before them, for
the CPU checks of the host binding only (the GPU checks need the rendered twins).
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import warnings
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, Optional

import pytest
import torch

from tests.cake.test_helpers.cake_naive import DRAWS, draw_seed
from tests.cake.test_helpers.cake_package import cake_package

OPS = ("fwd", "dgrad", "wgrad")
OUT_DTYPES = (torch.bfloat16, torch.float32)
BIT_VIEWS = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
# A signalling NaN payload: no launch stores it (an arithmetic NaN is quiet), so a row that still
# holds it was not written, and a launch that read it would not give finite bits.
SENTINEL = {torch.bfloat16: 0x7F81, torch.float32: 0x7F800B1D}
NEG_ZERO = {torch.bfloat16: -0x8000, torch.float32: -0x80000000}
POS_ZERO = 0
# The archs and SM counts of the CPU sweeps of the host binding.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
SM_COUNTS = (148, 160, 212)


# ---------------------------------------------------------------------------
# Bits
# ---------------------------------------------------------------------------


def bits(t: torch.Tensor) -> torch.Tensor:
    """The integer words of ``t``'s elements: ``-0.0`` differs from ``+0.0`` and a NaN equals only
    its own payload."""
    return t.view(BIT_VIEWS[t.dtype])


def assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str = "") -> None:
    """``got`` holds exactly the words of ``want``."""
    assert got.dtype == want.dtype and got.shape == want.shape, (
        what,
        got.dtype,
        tuple(got.shape),
        want.dtype,
        tuple(want.shape),
    )
    differ = bits(got) != bits(want)
    assert not bool(differ.any()), f"{what}: {int(differ.sum())} elements differ"


def differing(got: torch.Tensor, want: torch.Tensor) -> int:
    """How many elements of ``got`` and ``want`` (same shape and dtype) differ in their words."""
    return int((bits(got) != bits(want)).sum())


def fill_word(t: torch.Tensor, word: int) -> torch.Tensor:
    """Fill every element of ``t`` with the word ``word``; returns ``t``."""
    bits(t).fill_(word)
    return t


def sentinel(shape: Sequence[int], dtype: torch.dtype, device) -> torch.Tensor:
    """A tensor of ``shape`` whose every element is :data:`SENTINEL`."""
    return fill_word(
        torch.empty(tuple(shape), dtype=dtype, device=device), SENTINEL[dtype]
    )


def assert_untouched(t: torch.Tensor, what: str) -> None:
    """Every element of ``t`` still holds :data:`SENTINEL`."""
    written = bits(t) != SENTINEL[t.dtype]
    assert not bool(written.any()), f"{what}: {int(written.sum())} elements written"


def assert_all_word(t: torch.Tensor, word: int, what: str) -> None:
    """Every element of ``t`` is the word ``word``."""
    other = bits(t) != word
    assert not bool(other.any()), (
        f"{what}: {int(other.sum())} elements are not {word:#x}"
    )


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------


class Package:
    """The modules and entries of one package (a key of ``cake_package.PACKAGES``)."""

    def __init__(self, key: str):
        self.key = key
        self.mode = "nondet" if "_nondet_" in key else "det"
        self.pkg: ModuleType = cake_package(key)
        self.cb: ModuleType = cake_package(key, "cake_backend")
        self.jit: ModuleType = cake_package(key, "cake_jit")
        self.build: ModuleType = cake_package(key, "cake_build")
        self.autograd: ModuleType = cake_package(key, "cake_autograd")
        self.prepare: dict[str, Callable[..., Any]] = {
            "fwd": getattr(self.cb, f"{key}_prepare_forward"),
            "dgrad": getattr(self.cb, f"{key}_prepare_dgrad"),
            "wgrad": getattr(self.cb, f"{key}_prepare_wgrad"),
        }
        self.gemm: Callable[..., torch.Tensor] = getattr(
            self.pkg, f"cake_moe_grouped_gemm_{self.mode}"
        )

    @property
    def nondet(self) -> bool:
        return self.mode == "nondet"

    @property
    def constants(self) -> dict[str, Any]:
        return self.jit.HOST_PLAN_CONSTANTS

    def dtype_name(self, dtype: torch.dtype) -> str:
        return self.cb._DTYPE_NAMES[dtype]


def require(
    P: Package, *ops: str, out_dtype: Optional[torch.dtype] = None
) -> torch.device:
    """The CUDA device, after skipping unless it is a supported architecture whose registry
    serves every op of ``ops`` in both row-table forms (``generated_program_available``: the routes
    and their lead twins)."""
    if not torch.cuda.is_available():
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    device = torch.device("cuda", 0)
    arch = P.cb.SUPPORTED_COMPUTE_CAPABILITIES.get(
        torch.cuda.get_device_capability(device)
    )
    if arch is None:
        pytest.skip("the ragged grouped GEMM programs require an SM100/SM103/SM107 GPU")
    for op in ops:
        dtype = out_dtype if op == "wgrad" else None
        if not P.cb.generated_program_available(device, op, dtype):
            pytest.skip(f"no {op} program with its lead twin registered for {arch}")
    return device


def arch_of(P: Package, device: torch.device) -> str:
    return P.cb.SUPPORTED_COMPUTE_CAPABILITIES[torch.cuda.get_device_capability(device)]


def fresh_stream_workspaces(monkeypatch, P: Package) -> None:
    """No kept stream workspace and no queue probe at the start (both restored after the test)."""
    monkeypatch.setattr(P.cb, "_stream_workspaces", {})
    monkeypatch.setattr(P.cb, "_queue_splits", {})


@contextlib.contextmanager
def quiet():
    """Silence the once-per-process warnings of a forced composed route."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# Layouts and operands
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Layout:
    """A whole row table: groups of ``sizes`` rows, group after group from row 0, then ``tail``
    rows no group holds.  ``row_pad`` extra elements end every operand row (``stride(0) = cols +
    row_pad``; a multiple of 8, so the row pitch stays a multiple of 16 bytes)."""

    sizes: tuple[int, ...]
    n: int
    k: int
    tail: int = 0
    row_pad: int = 0

    @property
    def groups(self) -> int:
        return len(self.sizes)

    @property
    def rows(self) -> int:
        """``R``: every operand's row count, the ``sum_m`` of a slice."""
        return sum(self.sizes) + self.tail

    def table(self) -> list[int]:
        """``m_indptr`` of the whole table (``E + 1`` entries from 0)."""
        out = [0]
        for size in self.sizes:
            out.append(out[-1] + size)
        return out

    def bounds(self, e0: int, e1: int) -> tuple[int, int]:
        """``(m_indptr[e0], m_indptr[e1])``: the rows of groups ``[e0, e1)``."""
        table = self.table()
        return table[e0], table[e1]

    def chunks(self, size: int) -> list[tuple[int, int]]:
        """Consecutive ``[e0, e1)`` of ``size`` groups (the last may be shorter)."""
        return [(e, min(e + size, self.groups)) for e in range(0, self.groups, size)]


def inputs(layout: Layout, device, seed: int = 0):
    """``x [R, K]``, ``w [E, N, K]``, ``g [R, N]`` (bf16, the sibling suites' scaling) and the device
    ``m_indptr [E + 1]`` (int32) of ``layout``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    rows, e, n, k = layout.rows, layout.groups, layout.n, layout.k
    x = torch.randn(rows, k, generator=gen, device=device, dtype=torch.float32)
    w = torch.randn(e, n, k, generator=gen, device=device, dtype=torch.float32)
    g = torch.randn(rows, n, generator=gen, device=device, dtype=torch.float32)
    x = (x / math.sqrt(k)).to(torch.bfloat16)
    w = (w / math.sqrt(k)).to(torch.bfloat16)
    g = g.to(torch.bfloat16)
    table = torch.tensor(layout.table(), dtype=torch.int32, device=device)
    return x, w, g, table


def poisoned(t: torch.Tensor, r0: int, r1: int, row_pad: int = 0) -> torch.Tensor:
    """``t`` with every row outside ``[r0, r1)`` (and every padding element) :data:`SENTINEL`, as
    the leading columns of rows of ``cols + row_pad`` elements."""
    rows, cols = t.shape
    wide = sentinel((rows, cols + row_pad), t.dtype, t.device)
    wide[r0:r1, :cols] = t[r0:r1]
    return wide[:, :cols]


def padded(t: torch.Tensor, r0: int, r1: int) -> torch.Tensor:
    """Rows ``[r0, r1)`` of ``t`` moved to row 0 of a zero tensor of ``t``'s shape (reference 1)."""
    out = torch.zeros(t.shape, dtype=t.dtype, device=t.device)
    out[: r1 - r0] = t[r0:r1]
    return out


def random_like_out(shape, dtype, device, seed: int) -> torch.Tensor:
    """A finite ``out`` to accumulate into (``main_grad``)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        tuple(shape), generator=gen, device=device, dtype=torch.float32
    ).to(dtype)


# ---------------------------------------------------------------------------
# Lead binding
# ---------------------------------------------------------------------------


def assert_lead(P: Package, lead, base, what: str = "") -> None:
    """``lead`` (an ``m_indptr`` call) binds the lead twins of what ``base`` (an ``[E]`` call of the
    same shapes and ``sum_m``) binds: the same route record, plan and stages; every launch on the
    lead twin of ``base``'s route (a lead route); every program the twin of ``base``'s."""
    jit = P.jit
    assert lead.record_name == base.record_name, (
        what,
        lead.record_name,
        base.record_name,
    )
    assert lead.plan == base.plan, (what, lead.plan, base.plan)
    assert lead.stages == base.stages, (what, lead.stages, base.stages)
    assert base.bound_routes, (what, "the [E] call launches nothing")
    assert not any(jit.is_lead_route(r) for r in base.bound_routes), (
        what,
        base.bound_routes,
    )
    assert all(jit.is_lead_route(r) for r in lead.bound_routes), (
        what,
        lead.bound_routes,
    )
    assert lead.bound_routes == tuple(jit.lead_route(r) for r in base.bound_routes), (
        what,
        lead.bound_routes,
        base.bound_routes,
    )
    assert lead.bound_programs == tuple(
        jit.lead_twin(p) for p in base.bound_programs
    ), (
        what,
        lead.bound_programs,
        base.bound_programs,
    )


def lead_route_mirrors(P: Package, name: str) -> None:
    """Route ``name``'s lead twin mirrors it: the same op, architectures, stage names, order
    and specializations, each stage on the twin of the route's program, the twin's record the
    program's with ``m_begin`` after ``offs`` and ``twin_of`` the program."""
    jit = P.jit
    route = jit.ROUTES[name]
    twin_name = jit.lead_route(name)
    assert twin_name in jit.ROUTES, f"route {name} has no lead twin {twin_name}"
    twin = jit.ROUTES[twin_name]
    assert twin["op"] == route["op"] and twin["arches"] == route["arches"], (
        name,
        twin,
        route,
    )
    assert [s["name"] for s in twin["stages"]] == [
        s["name"] for s in route["stages"]
    ], name
    for item, lead_item in zip(route["stages"], twin["stages"], strict=True):
        where = (name, item["name"])
        assert lead_item.get("specializations", {}) == item.get(
            "specializations", {}
        ), where
        program = item["program"]
        assert lead_item["program"] == jit.lead_twin(program), (
            where,
            lead_item["program"],
        )
        base, record = jit.PROGRAMS[program], jit.PROGRAMS[lead_item["program"]]
        assert record["twin_of"] == program and "twin_of" not in base, where
        plan = [list(entry) for entry in base["arg_plan"]]
        plan.insert(plan.index(["buffer", "offs"]) + 1, ["buffer", "m_begin"])
        assert [list(entry) for entry in record["arg_plan"]] == plan, where
        for key in ("ffi_entry", "launch", "arches"):
            assert record[key] == base[key], (where, key)
        assert record["library"] == lead_item["program"], where


def ensure_lead_registry(monkeypatch, P: Package) -> bool:
    """Stand-ins for the lead twins on a registry rendered without them (restored after the test);
    False (nothing changed) when the registry has lead routes.  A stand-in route is its route with
    each stage on ``<program>_lead``, whose record is the program's with ``["buffer", "m_begin"]``
    after ``["buffer", "offs"]`` and ``twin_of`` the program (the rendered form,
    :func:`lead_route_mirrors`).  For the CPU checks of the host binding only: no stand-in builds."""
    jit = P.jit
    if any(jit.is_lead_route(name) for name in jit.ROUTES):
        return False
    for name, route in list(jit.ROUTES.items()):
        stages = []
        for item in route["stages"]:
            program = item["program"]
            twin = program + jit.LEAD_SUFFIX
            if twin not in jit.PROGRAMS:
                record = copy.deepcopy(jit.PROGRAMS[program])
                plan = record["arg_plan"]
                plan.insert(plan.index(["buffer", "offs"]) + 1, ["buffer", "m_begin"])
                record["library"] = twin
                record["twin_of"] = program
                monkeypatch.setitem(jit.PROGRAMS, twin, record)
            stages.append({**item, "program": twin})
        monkeypatch.setitem(
            jit.ROUTES, jit.lead_route(name), {**route, "stages": stages}
        )
    return True


# ---------------------------------------------------------------------------
# A CPU stand-in device (host binding only)
# ---------------------------------------------------------------------------


class NoModule:
    """A loaded program stand-in: every entry is a no-op."""

    def __getattr__(self, name):
        return lambda *args: None


def fake_device(
    monkeypatch, P: Package, arch: str, sms: int, *, loadable: bool = True
) -> None:
    """Run the hosts of the prepare functions on meta tensors: the operands' device stands for an
    ``arch`` device with ``sms`` SMs, programs load as no-op modules, the workspaces are plain bytes,
    every accumulate build counts as loadable (``loadable``), the accumulate checks of ``out`` pass
    (meta tensors have no addresses) and the queue probe is not built."""
    cb = P.cb
    monkeypatch.setattr(
        cb, "_check_device", lambda tensors: next(iter(tensors.values())).device
    )
    monkeypatch.setattr(cb, "arch_for", lambda device: arch)
    monkeypatch.setattr(cb, "sm_count", lambda device: sms)
    monkeypatch.setattr(cb, "load_module", lambda *args, **kwargs: NoModule())
    monkeypatch.setattr(
        cb,
        "_aligned_bytes",
        lambda n, device, what: torch.empty(n, dtype=torch.uint8, device=device),
    )
    monkeypatch.setattr(cb, "_device_index", lambda device: 0)
    monkeypatch.setattr(cb, "_WARNED", set())
    monkeypatch.setattr(cb, "acc_library_loadable", lambda *args: loadable)
    monkeypatch.setattr(cb, "_check_accumulate_out", lambda *args, **kwargs: None)
    monkeypatch.setattr(cb, "_accumulate_aligned", lambda out: True)
    monkeypatch.setattr(cb, "_queue_probe_module", lambda capability: None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0)
    )


def record_binds(monkeypatch, P: Package) -> list[tuple[str, str, dict[str, Any]]]:
    """Record ``(route, stage, kwargs)`` of every ``_bind``."""
    bound: list[tuple[str, str, dict[str, Any]]] = []
    real = P.cb._bind

    def record(name, stage, kwargs, arch, **extra):
        bound.append((name, stage, dict(kwargs)))
        return real(name, stage, kwargs, arch, **extra)

    monkeypatch.setattr(P.cb, "_bind", record)
    return bound


def meta_calls(P: Package, op: str, layout: Layout, e0: int, e1: int, **kwargs):
    """The slice call, reference 1 and reference 2 of groups ``[e0, e1)`` of ``layout``, prepared
    on meta tensors (under :func:`fake_device`): ``(slice, ref1, ref2, chunk)``."""
    meta = torch.device("meta")
    r0, r1 = layout.bounds(e0, e1)
    rows, e = layout.rows, e1 - e0
    table = torch.empty(layout.groups + 1, dtype=torch.int32, device=meta)
    chunk = table[e0 : e1 + 1]
    ends = torch.empty(e, dtype=torch.int32, device=meta)
    w = torch.empty(e, layout.n, layout.k, dtype=torch.bfloat16, device=meta)
    prepare = P.prepare[op]

    def rows_of(count, cols):
        return torch.empty(count, cols, dtype=torch.bfloat16, device=meta)

    if op == "wgrad":
        dtype = kwargs.pop("out_dtype", torch.bfloat16)

        def call(count, offs, **extra):
            out = torch.empty(e, layout.n, layout.k, dtype=dtype, device=meta)
            return prepare(
                rows_of(count, layout.n),
                rows_of(count, layout.k),
                offs,
                out=out,
                **extra,
                **kwargs,
            )

        return (
            call(rows, chunk, num_groups=e),
            call(rows, ends),
            call(r1 - r0, ends),
            chunk,
        )
    cols_in, cols_out = (layout.k, layout.n) if op == "fwd" else (layout.n, layout.k)

    def call_rows(count, offs):
        out = torch.empty(count, cols_out, dtype=torch.bfloat16, device=meta)
        return prepare(rows_of(count, cols_in), w, offs, out=out)

    return (
        call_rows(rows, chunk),
        call_rows(rows, ends),
        call_rows(r1 - r0, ends),
        chunk,
    )


def assert_m_begin_view(binds, chunk: torch.Tensor, what: str = "") -> None:
    """Every recorded bind of a lead route took ``m_begin`` as ``chunk[:1]`` and ``offs`` as
    ``chunk[1:]`` (views of the caller's table: no copy, no host read)."""
    root = chunk if chunk._base is None else chunk._base
    seen = 0
    for name, stage, kwargs in binds:
        m_begin, offs = kwargs["m_begin"], kwargs["offs"]
        assert m_begin.shape == (1,) and offs.shape == (chunk.numel() - 1,), (
            what,
            name,
            stage,
        )
        for view, start in ((m_begin, 0), (offs, 1)):
            assert view._base is root, (what, name, stage)
            assert view.storage_offset() == chunk.storage_offset() + start, (
                what,
                name,
                stage,
            )
        seen += 1
    assert seen, (what, "no bind")


# ---------------------------------------------------------------------------
# Forced stages
# ---------------------------------------------------------------------------

ROW_FORMS = ("main", "main_wide", "main_wide_ef", "main_wide_ts33")
WGRAD_FORMS = (
    "k256_main",
    "k256_main_wave",
    "k512_main",
    "k512_main_wave",
    "k512_main_halves",
    "k512_main_wave_halves",
    "k512_main_halves_k256t",
    "k512_queue",
)


def force_row_form(monkeypatch, P: Package, op: str, form: str, arch: str) -> bool:
    """Host constants under which ``op`` on ``arch`` takes stage ``form`` (:data:`ROW_FORMS`) at any
    shape whose output columns are a multiple of ``wide_block_n``; False where ``arch`` has no such
    form.  (The registry must also serve the stage: ``cake_jit.stage_served``.)"""
    c = P.constants
    if form == "main":
        monkeypatch.setitem(c, f"wide_{op}_archs", [])
        return True
    if arch not in c[f"wide_{op}_archs"]:
        return False
    ef, ts33 = form == "main_wide_ef", form == "main_wide_ts33"
    if op == "dgrad":
        if ef:
            if arch not in c["dgrad_evict_first_archs"]:
                return False
            monkeypatch.setitem(c["dgrad_evict_first_min_rows"], arch, 1)
            monkeypatch.setitem(c, "dgrad_evict_first_min_k", 1)
        else:
            monkeypatch.setitem(c, "dgrad_evict_first_archs", [])
    elif ef:
        if arch in c["fwd_lwh_ef_archs"]:
            monkeypatch.setitem(c, "fwd_lwh_ef_max_k", 1 << 30)
        elif arch in c["fwd_evict_first_archs"]:
            monkeypatch.setitem(c, "fwd_evict_first_min_k", 1)
        else:
            return False
    else:
        monkeypatch.setitem(c, "fwd_lwh_ef_archs", [])
        monkeypatch.setitem(c, "fwd_evict_first_archs", [])
    if ts33:
        if arch not in c[f"{op}_wide_ts33_archs"]:
            return False
        if op == "dgrad":
            monkeypatch.setitem(c["dgrad_wide_ts33_min_rows"], arch, 1)
            monkeypatch.setitem(c, "dgrad_wide_ts33_min_k", 1)
        else:
            monkeypatch.setitem(c, "fwd_wide_ts33_max_k", 1 << 30)
            monkeypatch.setitem(c, "fwd_wide_ts33_max_rows", 1 << 30)
    else:
        monkeypatch.setitem(c, f"{op}_wide_ts33_archs", [])
    return True


def force_wgrad_form(
    monkeypatch, P: Package, form: str, arch: str
) -> Optional[tuple[int, str]]:
    """Host constants under which a weight gradient on ``arch`` takes ``form``
    (:data:`WGRAD_FORMS`: k tile and main stage; ``k512_queue`` the non-deterministic queue walk
    standing in for ``main_halves``) at any rows per group, given a ``K`` of :func:`form_k`;
    ``(tile_k, main stage)``, or None where ``arch`` (or the package) has no such form."""
    c = P.constants
    tile = 512 if form.startswith("k512") else 256
    stage = form.split("_", 1)[1]
    queue = stage == "queue"
    if queue:
        if not P.nondet:
            return None
        stage = "main_halves"
    halves = stage in ("main_halves", "main_wave_halves", "main_halves_k256t")
    wave = stage in ("main_wave", "main_wave_halves")
    k256t = stage == "main_halves_k256t"
    if tile == 256:
        monkeypatch.setitem(c, "k512_archs", [])
    else:
        if arch not in c["k512_archs"]:
            return None
        monkeypatch.setitem(c["k512_min_rows_per_group"], arch, 1)
        if halves and arch not in c["k512_release_archs"]:
            return None
        if not halves:
            monkeypatch.setitem(c, "k512_release_archs", [])
        if k256t:
            if arch not in c["k256_tail_archs"]:
                return None
            monkeypatch.setitem(c, "k256_tail_min_rows", 1)
        else:
            monkeypatch.setitem(c, "k256_tail_archs", [])
    if wave and arch not in c["wave_sync_archs"][tile]:
        return None
    if not wave:
        monkeypatch.setitem(c, "wave_sync_min_k", 1 << 30)
    if P.nondet:
        rules = []
        if queue:
            rules = [
                dict(
                    arch=arch,
                    sm_counts=list(range(1, 1025)),
                    stage="main_halves",
                    outputs=["bfloat16", "float32", "accumulate"],
                    groups=[1, 1 << 30],
                )
            ]
        monkeypatch.setitem(c, "queue_walk", rules)
        if queue:
            return tile, P.cb.QUEUE_STAGE
    return tile, stage


def form_ks(P: Package, form: str) -> tuple[int, ...]:
    """Weight-gradient ``K`` of ``form``, smallest first: at least ``wave_sync_min_k`` for a wave
    form, at most ``k256_tail_max_k`` for the k256 tail, else from 1024."""
    c = P.constants
    if "wave" in form:
        least = int(c["wave_sync_min_k"])
        return (least, 2 * least)
    if form.endswith("k256t"):
        return tuple(k for k in (1024, 2048) if k <= int(c["k256_tail_max_k"]))
    return (1024, 2048, 4096)


def form_k(P: Package, form: str) -> int:
    """The smallest ``K`` of ``form`` (:func:`form_ks`)."""
    return form_ks(P, form)[0]


def wgrad_plan(
    P: Package, arch: str, sms: int, sum_m: int, groups: int, n: int, k: int, out_dtype
):
    """``(main stage, plan)`` a weight gradient of ``sum_m`` rows over ``groups`` groups, ``(N, K)``,
    takes on ``arch`` with ``sms`` SMs under the constants as set (the host's choice, before the
    queue walk)."""
    cb, c = P.cb, P.constants
    tile_k = cb.select_wgrad_tile(k, arch, sum_m, groups, c)
    name = P.jit.record_name("wgrad", out_dtype=P.dtype_name(out_dtype), tile_k=tile_k)
    stage, _wave = cb.select_wgrad_stage(
        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=groups
    )
    kwargs: dict[str, Any] = dict(
        sum_m=sum_m,
        num_groups=groups,
        n=n,
        k=k,
        tile_k=tile_k,
        sm_count=sms,
        reduce_threads=256,
        k256_tail=stage == "main_halves_k256t",
        constants=c,
    )
    if P.nondet:
        kwargs.update(
            unordered_tail=cb.wgrad_unordered_tail(tile_k, arch, c),
            ordered_tail=cb.wgrad_ordered_tail(tile_k, arch, c),
        )
    return stage, cb.plan_wgrad(**kwargs)


def split_tail_n(
    P: Package, arch: str, sms: int, sum_m: int, groups: int, k: int, out_dtype
):
    """The smallest ``N`` (``256 * b``, ``b <= 48``) whose weight-gradient plan splits its tail
    (``tail_splits > 1``) at ``sum_m`` rows over ``groups`` groups, within the 32-bit output span;
    None when none does."""
    span = int(P.constants["int32_elems"])
    for blocks in range(1, 49):
        n = 256 * blocks
        if groups * n * k > span:
            break
        _stage, plan = wgrad_plan(P, arch, sms, sum_m, groups, n, k, out_dtype)
        if plan["tail_splits"] > 1:
            return n
    return None


# ---------------------------------------------------------------------------
# Slice runs
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Slice:
    """A prepared device slice of groups ``[e0, e1)`` and its references (module docstring)."""

    op: str
    layout: Layout
    e0: int
    e1: int
    lead: Any
    ref1: Any
    ref2: Any

    @property
    def rows(self) -> tuple[int, int]:
        return self.layout.bounds(self.e0, self.e1)

    @property
    def plan_equal(self) -> bool:
        """Reference 2 plans the slice's launch: the same plan and stages."""
        return self.ref2.plan == self.lead.plan and self.ref2.stages == self.lead.stages

    def launch_refs(self) -> None:
        self.ref1.launch()
        self.ref2.launch()

    def outputs(self):
        """``(slice, reference 1, reference 2)`` outputs over the slice: the rows ``[r0, r1)`` of
        ``Y`` / ``dX`` (reference 1 from row 0), the whole ``dW``."""
        if self.op == "wgrad":
            return self.lead.out, self.ref1.out, self.ref2.out
        r0, r1 = self.rows
        return self.lead.out[r0:r1], self.ref1.out[: r1 - r0], self.ref2.out

    def check(self, *, ref2: str = "if_plan_equal") -> int:
        """After the launches: the rows outside the slice untouched (``Y`` / ``dX``), the slice's
        bits reference 1's, and reference 2's where ``ref2`` asks: ``"required"`` (its plan must be
        the slice's), ``"if_plan_equal"`` or ``"values"`` (equal values, not bits: the whole-empty
        accumulate).  Returns the elements that differ from reference 2 (0 when its bits hold)."""
        what = f"{self.op} slice [{self.e0}, {self.e1}) rows {self.rows}"
        if self.op != "wgrad":
            r0, r1 = self.rows
            assert_untouched(self.lead.out[:r0], f"{what}: rows before the slice")
            assert_untouched(self.lead.out[r1:], f"{what}: rows after the slice")
        got, want1, want2 = self.outputs()
        assert_same_bits(got, want1, f"{what} vs reference 1")
        if ref2 == "values":
            assert bool((got.float() == want2.float()).all()), (
                f"{what}: values differ from reference 2"
            )
            return differing(got, want2)
        if ref2 == "required":
            assert self.plan_equal, (
                what,
                self.lead.plan,
                self.ref2.plan,
                self.lead.stages,
                self.ref2.stages,
            )
        if self.plan_equal:
            assert_same_bits(got, want2, f"{what} vs reference 2")
            return 0
        return differing(got, want2)


def build_slice(
    P: Package,
    op: str,
    layout: Layout,
    e0: int,
    e1: int,
    device,
    *,
    seed: int = 0,
    out_dtype: torch.dtype = torch.bfloat16,
    accumulate: bool = False,
    init: Optional[torch.Tensor] = None,
    operands=None,
) -> Slice:
    """Prepare the slice of groups ``[e0, e1)`` of ``layout`` (operands of ``inputs(layout, device,
    seed)`` unless ``operands`` gives ``(x, w, g, table)``; the rows outside the slice poisoned)
    and its references, and check the lead binding against reference 1 (:func:`assert_lead`).
    A weight gradient writes into a copy of ``init`` (else :data:`SENTINEL`) per call."""
    x, w, g, table = operands if operands is not None else inputs(layout, device, seed)
    r0, r1 = layout.bounds(e0, e1)
    chunk = table[e0 : e1 + 1]
    ends = table[e0 + 1 : e1 + 1] - table[e0]
    wc = w[e0:e1]
    prepare = P.prepare[op]
    pad = layout.row_pad
    if op == "wgrad":
        groups = e1 - e0

        def out():
            if init is not None:
                return init.clone()
            return sentinel((groups, layout.n, layout.k), out_dtype, device)

        common = dict(out_dtype=out_dtype, accumulate=accumulate)
        lead = prepare(
            poisoned(g, r0, r1, pad),
            poisoned(x, r0, r1, pad),
            chunk,
            out=out(),
            num_groups=groups,
            **common,
        )
        ref1 = prepare(padded(g, r0, r1), padded(x, r0, r1), ends, out=out(), **common)
        ref2 = prepare(g[r0:r1], x[r0:r1], ends, out=out(), **common)
    else:
        a = x if op == "fwd" else g
        cols = layout.n if op == "fwd" else layout.k
        rows = layout.rows
        lead = prepare(
            poisoned(a, r0, r1, pad),
            wc,
            chunk,
            out=sentinel((rows, cols), torch.bfloat16, device),
        )
        ref1 = prepare(
            padded(a, r0, r1),
            wc,
            ends,
            out=sentinel((rows, cols), torch.bfloat16, device),
        )
        ref2 = prepare(
            a[r0:r1], wc, ends, out=sentinel((r1 - r0, cols), torch.bfloat16, device)
        )
    assert_lead(P, lead, ref1, f"{op} slice [{e0}, {e1})")
    return Slice(op, layout, e0, e1, lead, ref1, ref2)


def run_slice(
    P: Package,
    op: str,
    layout: Layout,
    e0: int,
    e1: int,
    device,
    *,
    ref2="if_plan_equal",
    **kwargs,
):
    """:func:`build_slice`, launch the slice and its references, :meth:`Slice.check`; returns
    ``(slice, elements differing from reference 2)``."""
    s = build_slice(P, op, layout, e0, e1, device, **kwargs)
    s.lead.launch()
    s.launch_refs()
    torch.cuda.synchronize(device)
    return s, s.check(ref2=ref2)


def run_from_zero(
    P: Package,
    op: str,
    layout: Layout,
    device,
    *,
    seed: int = 0,
    out_dtype: torch.dtype = torch.bfloat16,
    accumulate: bool = False,
    init: Optional[torch.Tensor] = None,
):
    """The ``m_indptr`` call over the whole table of ``layout`` (first entry 0; the tail rows
    poisoned) against the ``[E]`` call ``m_indptr[1:]`` on the same tensors: the lead twins, the
    same plan, the same bits, the tail rows of ``Y`` / ``dX`` untouched.  Returns both launches."""
    x, w, g, table = inputs(layout, device, seed)
    r1 = sum(layout.sizes)
    pad, e = layout.row_pad, layout.groups
    prepare = P.prepare[op]
    if op == "wgrad":
        a, b = poisoned(g, 0, r1, pad), poisoned(x, 0, r1, pad)

        def out():
            if init is not None:
                return init.clone()
            return sentinel((e, layout.n, layout.k), out_dtype, device)

        common = dict(out_dtype=out_dtype, accumulate=accumulate)
        lead = prepare(a, b, table, out=out(), num_groups=e, **common)
        base = prepare(a, b, table[1:], out=out(), **common)
    else:
        a = poisoned(x if op == "fwd" else g, 0, r1, pad)
        cols = layout.n if op == "fwd" else layout.k
        lead = prepare(
            a, w, table, out=sentinel((layout.rows, cols), torch.bfloat16, device)
        )
        base = prepare(
            a, w, table[1:], out=sentinel((layout.rows, cols), torch.bfloat16, device)
        )
    assert_lead(P, lead, base, f"{op} from row 0")
    lead.launch()
    base.launch()
    torch.cuda.synchronize(device)
    assert_same_bits(lead.out, base.out, f"{op} m_indptr from 0 vs [E]")
    if op != "wgrad":
        assert_untouched(lead.out[r1:], f"{op} from row 0: rows after the last group")
    return lead, base


def empty_accumulate_word(P: Package, route: str, dtype: torch.dtype) -> int:
    """The word a whole-empty slice of an accumulating weight gradient leaves in an ``out`` of
    ``-0.0`` (the documented exception: equal values).  The slice launches over its ``sum_m`` rows:
    the deterministic package adds exact zeros to every empty group on both routes, and so does the
    non-deterministic composed route (``out.add_`` of the scratch), turning ``-0.0`` into ``+0.0``;
    the non-deterministic fused route does not store an empty group, leaving ``-0.0``.  Reference 2
    launches nothing and leaves ``-0.0``."""
    if P.nondet and route == "fused":
        return NEG_ZERO[dtype]
    return POS_ZERO


# ---------------------------------------------------------------------------
# Judged bits
# ---------------------------------------------------------------------------

LEAD_ROW_CANDIDATES = (1, 3, 5, 7, 9, 13, 17, 31, 33, 63, 65, 127, 129)


def judged_cases(test) -> list[tuple]:
    """The ``(sizes, n, k)`` parameter sets of a sibling ``test_*_matches_reference`` (from its
    ``parametrize`` marks)."""
    for mark in getattr(test, "pytestmark", ()):
        if mark.name == "parametrize" and mark.args[0].replace(" ", "") == "sizes,n,k":
            return [tuple(case) for case in mark.args[1]]
    raise AssertionError(f"{test.__name__} has no sizes,n,k parametrization")


def check_judged_bits(
    P: Package, suite_inputs, op: str, sizes, n: int, k: int, out_dtype=torch.bfloat16
) -> int:
    """Accuracy of a slice through bit identity: the inputs of every draw ``j`` of a case the
    sibling suite judges (``suite_inputs(sizes, n, k, device, seed=draw_seed(0, j))``, all
    ``DRAWS``), placed after ``p`` poisoned rows, give the bits of the ``[E]`` call the suite judged,
    ``p`` the first of :data:`LEAD_ROW_CANDIDATES` whose slice plans as the ``[E]`` call does (the
    rows before the groups change the plan's ``sum_m`` only).  Returns ``p``."""
    device = require(P, op, out_dtype=out_dtype)
    s, e = sum(sizes), len(sizes)
    prepare = P.prepare[op]
    cols = n if op == "fwd" else k
    most = max(LEAD_ROW_CANDIDATES)
    # The operand buffers of the slice: [most + s] rows, the rows before the groups poisoned.
    big_a = sentinel((most + s, k if op == "fwd" else n), torch.bfloat16, device)
    big_b = sentinel((most + s, k), torch.bfloat16, device)

    def calls(p, x, w, g, offs):
        table = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), offs]) + p
        if op == "wgrad":
            base = prepare(g, x, offs, out_dtype=out_dtype)
            big_a[p : p + s] = g
            big_b[p : p + s] = x
            out = sentinel((e, n, k), out_dtype, device)
            lead = prepare(
                big_a[: p + s],
                big_b[: p + s],
                table,
                out=out,
                out_dtype=out_dtype,
                num_groups=e,
            )
        else:
            a = x if op == "fwd" else g
            base = prepare(a, w, offs)
            big_a[p : p + s] = a
            out = sentinel((p + s, cols), torch.bfloat16, device)
            lead = prepare(big_a[: p + s], w, table, out=out)
        return base, lead

    first = suite_inputs(sizes, n, k, device, seed=draw_seed(0, 0))
    lead_rows = None
    for p in LEAD_ROW_CANDIDATES:
        base, lead = calls(p, *first)
        if lead.plan == base.plan and lead.stages == base.stages:
            lead_rows = p
            break
    assert lead_rows is not None, (
        f"no lead rows of {LEAD_ROW_CANDIDATES} keep the plan of {op} {sizes}"
    )
    p = lead_rows
    for buffer in (big_a, big_b):  # the search wrote other rows
        fill_word(buffer, SENTINEL[torch.bfloat16])
    for j in range(DRAWS):
        base, lead = calls(p, *suite_inputs(sizes, n, k, device, seed=draw_seed(0, j)))
        assert_lead(P, lead, base, f"{op} {sizes} draw {j}")
        want = base.launch()
        got = lead.launch()
        torch.cuda.synchronize(device)
        if op == "wgrad":
            assert_same_bits(got, want, f"{op} {sizes} draw {j}")
        else:
            assert_untouched(got[:p], f"{op} {sizes} draw {j}: rows before the slice")
            assert_same_bits(got[p:], want, f"{op} {sizes} draw {j}")
    return p


# ---------------------------------------------------------------------------
# Layouts of the checks
# ---------------------------------------------------------------------------

# Two references: 24 ragged groups (empty and one-row groups, odd boundaries) in four chunks of six, every
# chunk over 7000 rows.  At these shapes reference 2 plans every chunk as the slice does on every
# architecture and SM count (check_reference_2_plans): the forward pass and the data gradient fill
# the SM pairs with tiles on both sides and take main_wide (K 4096 / 2048 meets no evict-first,
# lwh_ef or ts33 rule), and the weight gradient (K 256: the 256-wide k tile, no wave) has fewer tiles
# than clusters, so no tail.
TWO_REF_SIZES = (
    (1201, 0, 977, 2049, 1, 2803)
    + (1500, 1499, 0, 0, 2111, 1907)
    + (3333, 1, 1, 1777, 0, 2001)
    + (0, 2500, 2500, 1023, 999, 7)
)
TWO_REF_CHUNK = 6
TWO_REF_LAYOUTS = {
    "fwd": Layout(TWO_REF_SIZES, n=2048, k=4096, tail=37),
    "dgrad": Layout(TWO_REF_SIZES, n=4096, k=2048, tail=37),
    "wgrad": Layout(TWO_REF_SIZES, n=512, k=256, tail=37),
}
# A divergent plan: a weight gradient whose slice takes the 512-wide k tile (its sum_m over three groups averages
# above every k512_min_rows_per_group) where reference 2 (7101 rows) takes the 256-wide one.
DIVERGENT_LAYOUT = Layout((5000, 4100, 0, 3001, 6200, 1), n=512, k=1024, tail=11)
DIVERGENT_CHUNK = (1, 4)
# Small ragged layouts: an empty group, a one-row group, odd boundaries, tail rows.
SMALL_SIZES = (300, 0, 1, 700, 513, 129)
SMALL_CHUNKS = ((1, 4), (3, 5), (5, 6))
SHAPES = {"fwd": (512, 256), "dgrad": (256, 512), "wgrad": (512, 256)}
# Shapes whose output columns are a multiple of wide_block_n (the row forms).
WIDE_SHAPES = {"fwd": (1024, 512), "dgrad": (512, 1024)}
# Edges: whole-empty chunks at row 0, inside and before the tail, a one-row chunk, a chunk that ends
# with the last (empty) group, and the whole table from 0.
EDGE_SIZES = (0, 300, 0, 1, 0, 0, 700, 0)
EDGE_CHUNKS = ((0, 1), (2, 3), (4, 6), (3, 4), (5, 8), (1, 7), (0, 8))
EMPTY_CHUNKS = ((0, 1), (2, 3), (4, 6))
# Graphs: the routings a captured slice of groups [2, 6) replays over (each within the layout's rows).
GRAPH_ROUTINGS = (
    (300, 0, 1, 700, 513, 129, 64, 1000),
    (100, 200, 0, 901, 500, 0, 64, 900),
    (0, 0, 513, 1, 1, 1200, 9, 900),
    (1500, 1000, 0, 0, 0, 0, 7, 100),
)
GRAPH_ROWS = 2747
GRAPH_CHUNK = (2, 6)
# Padded rows: odd first rows and operand rows padded past their columns.
PADDED_SIZES = (333, 1, 0, 2049, 517, 3)
PADDED_CHUNKS = ((1, 5), (3, 6))
PAD = 72
# The one-group slices whose split tail holds group 0: (rows before, rows of the group, rows after).
SPLIT_ROWS = ((777, 5000, 333), (777, 20000, 333), (777, 60000, 333))


def small(
    op: str, *, tail: int = 77, row_pad: int = 0, sizes=SMALL_SIZES, shape=None
) -> Layout:
    n, k = shape or SHAPES[op]
    return Layout(tuple(sizes), n=n, k=k, tail=tail, row_pad=row_pad)


# ---------------------------------------------------------------------------
# CPU checks: registry and host binding
# ---------------------------------------------------------------------------


def check_registry_mirror(P: Package) -> None:
    """Every public route has its lead twin (:func:`lead_route_mirrors`); every lead route
    twins a public route; every ``twin_of`` record twins a program and is bound by a lead route."""
    jit = P.jit
    public = [name for name in jit.ROUTES if not jit.is_lead_route(name)]
    assert public
    for name in public:
        lead_route_mirrors(P, name)
    bound = set()
    for name, route in jit.ROUTES.items():
        if jit.is_lead_route(name):
            assert name.removesuffix(jit.LEAD_SUFFIX) in jit.ROUTES, name
            bound.update(item["program"] for item in route["stages"])
    twins = {p for p, record in jit.PROGRAMS.items() if "twin_of" in record}
    assert twins == bound, (sorted(twins - bound), sorted(bound - twins))
    for twin in twins:
        program = jit.PROGRAMS[twin]["twin_of"]
        assert program in jit.PROGRAMS and "twin_of" not in jit.PROGRAMS[program], twin
        assert jit.lead_twin(program) == twin
    with pytest.raises(ValueError):
        jit.lead_route(jit.lead_route(public[0]))


def check_lead_builds(P: Package) -> None:
    """Every stage binding of a lead route has its ``build.json`` entry, with the ``min_cuda`` of
    the entry of the program it twins."""
    jit = P.jit
    entries = P.build.libraries_by_name()
    seen = 0
    for name, route in jit.ROUTES.items():
        if not jit.is_lead_route(name):
            continue
        base_route = jit.ROUTES[name.removesuffix(jit.LEAD_SUFFIX)]
        for item, base_item in zip(route["stages"], base_route["stages"], strict=True):
            spec = item.get("specializations", {})
            for arch in route["arches"]:
                if arch not in jit.PROGRAMS[item["program"]]["arches"]:
                    continue
                twin = entries[jit.library_name(item["program"], arch, spec)]
                base = entries[jit.library_name(base_item["program"], arch, spec)]
                assert twin["min_cuda"] == base["min_cuda"], (name, item["name"], arch)
                assert twin["arch"] == base["arch"] == arch, (name, item["name"], arch)
                seen += 1
    assert seen, "no lead route binding"


def check_row_table(P: Package) -> None:
    """``_row_table`` decides on the shape alone and returns views: an int32 ``[E + 1]`` table gives
    ``(table[1:], table[:1])``; an ``[E]`` table, another dtype or rank is returned as it is."""
    table = torch.arange(7, dtype=torch.int32)
    ends, m_begin = P.cb._row_table(table, 6)
    assert m_begin is not None and ends._base is table and m_begin._base is table
    assert ends.storage_offset() == 1 and tuple(ends.shape) == (6,)
    assert m_begin.storage_offset() == 0 and tuple(m_begin.shape) == (1,)
    chunk = table[2:6]
    ends, m_begin = P.cb._row_table(chunk, 3)
    assert ends.storage_offset() == 3 and m_begin.storage_offset() == 2
    for other, groups in ((table[1:], 6), (table.long(), 6), (table.view(7, 1), 6)):
        offs, m_begin = P.cb._row_table(other, groups)
        assert offs is other and m_begin is None


def check_bound_route(monkeypatch, P: Package) -> None:
    """``_bound_route``: the route for an ``[E]`` table, its lead twin for an ``m_indptr``, and
    ``NotImplementedError`` where the lead twin does not serve the architecture (no silent
    fallback to group 0 from row 0)."""
    ensure_lead_registry(monkeypatch, P)
    jit, cb = P.jit, P.cb
    m_begin = torch.zeros(1, dtype=torch.int32)
    for arch in ARCHES:
        for op in ("fwd", "dgrad"):
            name = jit.select_module(op, arch)
            assert cb._bound_route(name, None, arch) == name
            assert cb._bound_route(name, m_begin, arch) == jit.lead_route(name)
    name = jit.select_module("fwd", ARCHES[0])
    monkeypatch.delitem(jit.ROUTES, jit.lead_route(name))
    assert cb._bound_route(name, None, ARCHES[0]) == name
    with pytest.raises(NotImplementedError):
        cb._bound_route(name, m_begin, ARCHES[0])


def check_available_needs_lead(monkeypatch, P: Package) -> None:
    """``generated_program_available`` holds only where the lead twin is registered too."""
    ensure_lead_registry(monkeypatch, P)
    jit, cb = P.jit, P.cb
    arch = ARCHES[0]
    fake_device(monkeypatch, P, arch, SM_COUNTS[0])
    meta = torch.device("meta")
    assert cb.generated_program_available(meta, "fwd")
    monkeypatch.delitem(jit.ROUTES, jit.lead_route(jit.select_module("fwd", arch)))
    assert not cb.generated_program_available(meta, "fwd")
    assert cb.generated_program_available(meta, "dgrad")


def check_host_binding(monkeypatch, P: Package, op: str) -> None:
    """CPU sweep of the host binding (architectures, SM counts, chunks; both weight-gradient
    dtypes and the accumulate mode): the slice binds the lead twins of reference 1's routes with its
    plan (:func:`assert_lead`), every launch takes ``m_begin`` / ``offs`` as views of the caller's
    table (:func:`assert_m_begin_view`), and an ``[E]`` call binds no lead route and no
    ``m_begin``."""
    ensure_lead_registry(monkeypatch, P)
    variants: list[dict[str, Any]] = [{}]
    if op == "wgrad":
        variants = [
            dict(out_dtype=dtype, accumulate=acc)
            for dtype in OUT_DTYPES
            for acc in (False, True)
        ]
    layouts = [
        small(op),
        small(op, sizes=TWO_REF_SIZES[:12], shape=WIDE_SHAPES.get(op)),
    ]
    if op == "wgrad":
        layouts.append(Layout((777, 20000, 333), n=4096, k=1024))
    for arch in ARCHES:
        for sms in SM_COUNTS:
            with monkeypatch.context() as scope:
                fake_device(scope, P, arch, sms)
                binds = record_binds(scope, P)
                for layout in layouts:
                    for e0, e1 in layout.chunks(3) + [(1, 2)]:
                        for extra in variants:
                            what = (arch, sms, op, layout.sizes, e0, e1, extra)
                            binds.clear()
                            lead, ref1, _ref2, chunk = meta_calls(
                                P, op, layout, e0, e1, **extra
                            )
                            assert_lead(P, lead, ref1, str(what))
                            lead_binds = list(binds[: len(lead.launches)])
                            assert all(P.jit.is_lead_route(b[0]) for b in lead_binds), (
                                what
                            )
                            assert_m_begin_view(lead_binds, chunk, str(what))
                            for name, _stage, kwargs in binds[len(lead.launches) :]:
                                assert not P.jit.is_lead_route(name), what
                                assert "m_begin" not in kwargs, what


def check_reference_2_plans(monkeypatch, P: Package) -> None:
    """The chunks of :data:`TWO_REF_LAYOUTS` plan as their reference 2 on every architecture and SM count of the sweep
    (both weight-gradient dtypes)."""
    ensure_lead_registry(monkeypatch, P)
    for arch in ARCHES:
        for sms in SM_COUNTS:
            with monkeypatch.context() as scope:
                fake_device(scope, P, arch, sms)
                for op, layout in TWO_REF_LAYOUTS.items():
                    dtypes = OUT_DTYPES if op == "wgrad" else (None,)
                    for dtype in dtypes:
                        extra = {} if dtype is None else {"out_dtype": dtype}
                        for e0, e1 in layout.chunks(TWO_REF_CHUNK):
                            lead, _ref1, ref2, _chunk = meta_calls(
                                P, op, layout, e0, e1, **extra
                            )
                            what = (arch, sms, op, dtype, e0, e1)
                            assert lead.plan == ref2.plan, (what, lead.plan, ref2.plan)
                            assert lead.stages == ref2.stages, (
                                what,
                                lead.stages,
                                ref2.stages,
                            )


def check_bind_mismatch(monkeypatch, P: Package) -> None:
    """``_bind`` refuses ``m_begin`` for a program that does not take it and a lead program
    without it (``KeyError``), so a wrong pairing of route and row table cannot launch."""
    ensure_lead_registry(monkeypatch, P)
    jit, cb = P.jit, P.cb
    arch = ARCHES[0]
    fake_device(monkeypatch, P, arch, SM_COUNTS[0])
    layout = small("fwd")
    real = cb._bound_route
    monkeypatch.setattr(cb, "_bound_route", lambda name, m_begin, arch: name)
    with pytest.raises(KeyError):
        meta_calls(P, "fwd", layout, 1, 4)
    monkeypatch.setattr(
        cb, "_bound_route", lambda name, m_begin, arch: jit.lead_route(name)
    )
    meta = torch.device("meta")
    x = torch.empty(layout.rows, layout.k, dtype=torch.bfloat16, device=meta)
    w = torch.empty(3, layout.n, layout.k, dtype=torch.bfloat16, device=meta)
    with pytest.raises(KeyError):
        P.prepare["fwd"](x, w, torch.empty(3, dtype=torch.int32, device=meta))
    monkeypatch.setattr(cb, "_bound_route", real)
    with pytest.raises(KeyError):  # an argument outside HOST_ARGUMENTS
        name = jit.select_module("fwd", arch)
        cb._bind(name, "main", {"not_an_argument": 1}, arch)


# ---------------------------------------------------------------------------
# GPU checks
# ---------------------------------------------------------------------------


def check_two_references(P: Package, op: str, out_dtype=torch.bfloat16) -> None:
    """Every chunk of :data:`TWO_REF_LAYOUTS` gives reference 1's bits and, its plan being
    reference 2's, reference 2's bits."""
    device = require(P, op, out_dtype=out_dtype)
    layout = TWO_REF_LAYOUTS[op]
    operands = inputs(layout, device, seed=11)
    for e0, e1 in layout.chunks(TWO_REF_CHUNK):
        s = build_slice(
            P, op, layout, e0, e1, device, out_dtype=out_dtype, operands=operands
        )
        s.lead.launch()
        s.launch_refs()
        torch.cuda.synchronize(device)
        s.check(ref2="required")


def check_plan_divergent(P: Package, out_dtype, record_property) -> None:
    """Where reference 2 plans otherwise (the k tile), the slice still gives reference 1's
    bits; how many elements differ from reference 2 is recorded (disclosed, not
    gated)."""
    device = require(P, "wgrad", out_dtype=out_dtype)
    s, differ = run_slice(
        P,
        "wgrad",
        DIVERGENT_LAYOUT,
        *DIVERGENT_CHUNK,
        device,
        seed=12,
        out_dtype=out_dtype,
    )
    assert s.lead.plan["tile_k"] == 512 and s.ref2.plan["tile_k"] == 256, (
        s.lead.plan,
        s.ref2.plan,
    )
    assert not s.plan_equal
    record_property("ref2_plan_equal", False)
    record_property("ref2_differing_elements", differ)
    record_property("ref2_elements", s.lead.out.numel())


def check_row_form(monkeypatch, P: Package, op: str, form: str) -> None:
    """Forward pass and data gradient: under constants that force stage ``form``, slices from an odd first row and
    a one-group slice give reference 1's bits (reference 2's where it plans the same), and the whole
    table from row 0 gives the ``[E]`` bits."""
    device = require(P, op)
    arch = arch_of(P, device)
    if not force_row_form(monkeypatch, P, op, form, arch):
        pytest.skip(f"no {form} {op} form on {arch}")
    if not P.jit.stage_served(P.jit.select_module(op, arch), form, arch):
        pytest.skip(f"the {op} route serves no {form} stage on {arch}")
    layout = small(op, shape=WIDE_SHAPES[op])
    operands = inputs(layout, device, seed=13)
    for e0, e1 in SMALL_CHUNKS:
        s = build_slice(P, op, layout, e0, e1, device, operands=operands)
        assert s.lead.stages == (form,), (s.lead.stages, form)
        s.lead.launch()
        s.launch_refs()
        torch.cuda.synchronize(device)
        s.check()
    lead, _base = run_from_zero(P, op, layout, device, seed=14)
    assert lead.stages == (form,), lead.stages


def split_tail_layout(P: Package, device, arch: str, ks, out_dtype) -> Optional[Layout]:
    """A layout of three groups whose slice of the middle one (``[1, 2)``, from an odd row) splits
    its tail under the constants as set (:data:`SPLIT_ROWS`, ``K`` of ``ks``, :func:`split_tail_n`),
    or None."""
    sms = P.cb.sm_count(device)
    for k in ks:
        for before, rows, after in SPLIT_ROWS:
            n = split_tail_n(P, arch, sms, before + rows + after, 1, k, out_dtype)
            if n is not None:
                return Layout((before, rows, after), n=n, k=k)
    return None


def check_wgrad_form(
    monkeypatch, P: Package, form: str, out_dtype, record_property
) -> None:
    """Weight gradient: under constants that force ``form`` (:data:`WGRAD_FORMS`), a one-group slice
    whose split tail holds group 0 (where a shape splits it), a three-group slice from an odd row
    and the whole table from row 0 take the form's main stage and give the references' bits."""
    device = require(P, "wgrad", out_dtype=out_dtype)
    arch = arch_of(P, device)
    forced = force_wgrad_form(monkeypatch, P, form, arch)
    if forced is None:
        pytest.skip(f"no {form} weight-gradient form on {arch}")
    tile, stage = forced
    route = P.jit.record_name("wgrad", out_dtype=P.dtype_name(out_dtype), tile_k=tile)
    if not P.jit.stage_served(route, stage, arch):
        pytest.skip(f"{route} serves no {stage} stage on {arch}")
    k = form_k(P, form)
    split = split_tail_layout(P, device, arch, form_ks(P, form), out_dtype)
    record_property(
        "split_tail_layout", None if split is None else (split.sizes, split.n, split.k)
    )
    if split is not None:
        s, _ = run_slice(P, "wgrad", split, 1, 2, device, seed=15, out_dtype=out_dtype)
        assert s.lead.stages[0] == stage, (s.lead.stages, stage)
        assert s.lead.plan["tail_splits"] > 1 and s.lead.plan["num_tail"] > 0, (
            s.lead.plan
        )
    layout = small("wgrad", shape=(512, k))
    s, _ = run_slice(P, "wgrad", layout, 1, 4, device, seed=16, out_dtype=out_dtype)
    assert s.lead.stages[0] == stage, (s.lead.stages, stage)
    lead, _base = run_from_zero(
        P, "wgrad", layout, device, seed=17, out_dtype=out_dtype
    )
    assert lead.stages[0] == stage, (lead.stages, stage)


def check_split_tail_holds_group_0(monkeypatch, P: Package, out_dtype) -> None:
    """The 512-wide k tile's split tail with group 0 in it: a one-group slice from an odd row whose
    plan splits the tail (the ordered one-launch tail of the non-deterministic package, the tail
    reduce of the deterministic one, both bound as lead twins) gives reference 1's bits."""
    device = require(P, "wgrad", out_dtype=out_dtype)
    arch = arch_of(P, device)
    form = (
        "k512_main_halves" if arch in P.constants["k512_release_archs"] else "k512_main"
    )
    if force_wgrad_form(monkeypatch, P, form, arch) is None:
        pytest.skip(f"no 512-wide k tile on {arch}")
    layout = split_tail_layout(P, device, arch, form_ks(P, form), out_dtype)
    assert layout is not None, (
        f"no shape of {SPLIT_ROWS} splits the 512-wide k tile's tail on {arch}"
    )
    s, _ = run_slice(P, "wgrad", layout, 1, 2, device, seed=18, out_dtype=out_dtype)
    assert s.lead.plan["tail_splits"] > 1, s.lead.plan
    if P.nondet:
        assert s.lead.plan["ordered_tail"], s.lead.plan
    else:
        assert len(s.lead.bound_routes) == 2, (
            s.lead.bound_routes
        )  # main stage, tail reduce


def check_probed_queue_split(monkeypatch, P: Package) -> None:
    """A data-gradient slice on an architecture of ``domain_split_archs`` takes the probed queue
    split, and gives reference 1's bits before and after its stream's probe completes."""
    device = require(P, "dgrad")
    arch = arch_of(P, device)
    if arch not in P.constants["domain_split_archs"]:
        pytest.skip(f"no probed queue split on {arch}")
    fresh_stream_workspaces(monkeypatch, P)
    layout = small("dgrad", shape=WIDE_SHAPES["dgrad"])
    s = build_slice(P, "dgrad", layout, 1, 4, device, seed=19)
    assert s.lead.plan["queue_split"] == "probed", s.lead.plan
    s.launch_refs()
    for _ in range(3):
        fill_word(s.lead.out, SENTINEL[torch.bfloat16])
        s.lead.launch()
        torch.cuda.synchronize(device)
        s.check()


def check_graph(monkeypatch, P: Package, op: str, out_dtype=torch.bfloat16) -> None:
    """A captured slice (``m_indptr[2:7]`` of a device table) replays over routings copied into
    that table on the device, under ``torch.cuda.set_sync_debug_mode("error")``, and gives each
    routing's reference 1 bits (a whole-empty slice included) without writing outside its rows."""
    device = require(P, op, out_dtype=out_dtype)
    fresh_stream_workspaces(monkeypatch, P)
    e0, e1 = GRAPH_CHUNK
    n, k = SHAPES[op]
    layouts = [Layout(r, n=n, k=k, tail=GRAPH_ROWS - sum(r)) for r in GRAPH_ROUTINGS]
    x, w, g, table = inputs(layouts[0], device, seed=20)
    refs = []
    for layout in layouts:
        routed = torch.tensor(layout.table(), dtype=torch.int32, device=device)
        s = build_slice(
            P,
            op,
            layout,
            e0,
            e1,
            device,
            out_dtype=out_dtype,
            operands=(x, w, g, routed),
        )
        s.launch_refs()
        refs.append((layout, routed, s))
    torch.cuda.synchronize(device)
    buffer = table.clone()
    chunk = buffer[e0 : e1 + 1]
    prepare = P.prepare[op]
    if op == "wgrad":
        out = sentinel((e1 - e0, n, k), out_dtype, device)
        lead = prepare(g, x, chunk, out=out, out_dtype=out_dtype, num_groups=e1 - e0)
    else:
        cols = n if op == "fwd" else k
        out = sentinel((GRAPH_ROWS, cols), torch.bfloat16, device)
        lead = prepare(x if op == "fwd" else g, w[e0:e1], chunk, out=out)
    lead.launch()  # stream workspaces (and the queue probe) before the capture
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        lead.launch()
    for layout, routed, s in refs:
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            buffer.copy_(routed)
            fill_word(lead.out, SENTINEL[lead.out.dtype])
            graph.replay()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize(device)
        r0, r1 = layout.bounds(e0, e1)
        what = f"{op} graph replay over {layout.sizes}"
        if op == "wgrad":
            assert_same_bits(lead.out, s.ref1.out, what)
        else:
            assert_untouched(lead.out[:r0], f"{what}: rows before the slice")
            assert_untouched(lead.out[r1:], f"{what}: rows after the slice")
            assert_same_bits(lead.out[r0:r1], s.ref1.out[: r1 - r0], what)


def check_edges(P: Package, op: str, out_dtype=torch.bfloat16) -> None:
    """Whole-empty chunks (from row 0, inside, before the tail), a one-row chunk, a chunk
    ending with the last (empty) group and the whole table from 0, with the rows outside each
    poisoned: reference 1's bits, reference 2's where it plans the same, nothing written outside;
    an overwriting weight gradient of a whole-empty chunk is exact ``+0.0``."""
    device = require(P, op, out_dtype=out_dtype)
    layout = small(op, sizes=EDGE_SIZES, tail=129)
    operands = inputs(layout, device, seed=21)
    for e0, e1 in EDGE_CHUNKS:
        s = build_slice(
            P, op, layout, e0, e1, device, out_dtype=out_dtype, operands=operands
        )
        s.lead.launch()
        s.launch_refs()
        torch.cuda.synchronize(device)
        s.check()
        if op == "wgrad" and (e0, e1) in EMPTY_CHUNKS:
            assert_all_word(s.lead.out, POS_ZERO, f"whole-empty chunk [{e0}, {e1})")


def accumulate_route(monkeypatch, P: Package, device, out_dtype, route: str) -> None:
    """Constants under which an accumulating weight gradient takes ``route``: the 512-wide k tile's
    ``main_halves`` (whose accumulate build the fused route binds) at any rows per group, and for
    ``composed`` accumulate builds that cannot load; skips where the device has no such route."""
    arch = arch_of(P, device)
    if force_wgrad_form(monkeypatch, P, "k512_main_halves", arch) is None:
        pytest.skip(f"no main_halves weight gradient on {arch}")
    if route == "composed":
        monkeypatch.setattr(P.cb, "acc_library_loadable", lambda *args: False)
    elif not P.cb.generated_program_available(
        device, "wgrad", out_dtype, accumulate=True
    ):
        pytest.skip(
            f"no fused accumulate build of the {out_dtype} weight gradient here"
        )


def check_empty_accumulate(monkeypatch, P: Package, out_dtype, route: str) -> None:
    """A whole-empty chunk accumulating into ``out`` of ``-0.0`` gives
    reference 1's bits and reference 2's values; its words are exactly
    :func:`empty_accumulate_word` (reference 2, which launches nothing, keeps ``-0.0``)."""
    device = require(P, "wgrad", out_dtype=out_dtype)
    accumulate_route(monkeypatch, P, device, out_dtype, route)
    layout = small("wgrad", sizes=EDGE_SIZES, tail=129, shape=(512, 1024))
    operands = inputs(layout, device, seed=22)
    word = empty_accumulate_word(P, route, out_dtype)
    for e0, e1 in EMPTY_CHUNKS:
        init = fill_word(
            torch.empty(e1 - e0, layout.n, layout.k, dtype=out_dtype, device=device),
            NEG_ZERO[out_dtype],
        )
        with quiet():
            s = build_slice(
                P,
                "wgrad",
                layout,
                e0,
                e1,
                device,
                out_dtype=out_dtype,
                accumulate=True,
                init=init,
                operands=operands,
            )
        assert s.lead.plan["accumulate"] == route, s.lead.plan
        s.lead.launch()
        s.launch_refs()
        torch.cuda.synchronize(device)
        s.check(ref2="values")
        assert_all_word(s.ref2.out, NEG_ZERO[out_dtype], "reference 2 (no launch)")
        assert_all_word(
            s.lead.out, word, f"whole-empty chunk [{e0}, {e1}) on the {route} route"
        )


def check_row_padded(
    monkeypatch, P: Package, op: str, form: Optional[str] = None
) -> None:
    """Slices from odd first rows of operands whose rows are padded past their columns
    (``stride(0) > cols``; the weight gradient's 256-wide k tile reads them through the row bases
    on sm_107a) give the references' bits."""
    out_dtype = torch.float32 if op == "wgrad" else None
    device = require(P, op, out_dtype=out_dtype)
    shape = SHAPES[op]
    if form is not None:
        arch = arch_of(P, device)
        forced = force_wgrad_form(monkeypatch, P, form, arch)
        if forced is None:
            pytest.skip(f"no {form} weight-gradient form on {arch}")
        shape = (512, form_k(P, form))
    layout = small(op, sizes=PADDED_SIZES, tail=61, row_pad=PAD, shape=shape)
    operands = inputs(layout, device, seed=23)
    kwargs = {} if out_dtype is None else {"out_dtype": out_dtype}
    for e0, e1 in PADDED_CHUNKS:
        s = build_slice(P, op, layout, e0, e1, device, operands=operands, **kwargs)
        s.lead.launch()
        s.launch_refs()
        torch.cuda.synchronize(device)
        s.check()


def check_two_streams(
    monkeypatch, P: Package, op: str, form: Optional[str] = None
) -> None:
    """Two slices launched concurrently on two streams (each with its own stream workspaces;
    the queue walk where ``form`` forces it) give their references' bits."""
    device = require(P, op)
    fresh_stream_workspaces(monkeypatch, P)
    shape = SHAPES[op]
    if form is not None:
        arch = arch_of(P, device)
        forced = force_wgrad_form(monkeypatch, P, form, arch)
        if forced is None:
            pytest.skip(f"no {form} weight-gradient form on {arch}")
        tile, stage = forced
        route = P.jit.record_name("wgrad", out_dtype="bfloat16", tile_k=tile)
        if not P.jit.stage_served(route, stage, arch):
            pytest.skip(f"{route} serves no {stage} stage on {arch}")
        shape = (512, form_k(P, form))
    layout = small(op, shape=shape)
    operands = inputs(layout, device, seed=24)
    slices = [
        build_slice(P, op, layout, e0, e1, device, operands=operands)
        for e0, e1 in ((1, 4), (3, 6))
    ]
    if form is not None:
        assert all(s.lead.stages[0] == stage for s in slices), [
            s.lead.stages for s in slices
        ]
    for s in slices:
        s.launch_refs()
    current = torch.cuda.current_stream(device)
    streams = [torch.cuda.Stream(device) for _ in slices]
    for _ in range(2):
        for s in slices:
            fill_word(s.lead.out, SENTINEL[s.lead.out.dtype])
        for stream, s in zip(streams, slices, strict=True):
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                s.lead.launch()
        for stream in streams:
            current.wait_stream(stream)
        torch.cuda.synchronize(device)
        for s in slices:
            s.check()


def check_autograd_slice(P: Package) -> None:
    """The autograd function over a slice from an odd row (``x`` and ``grad_y`` poisoned
    outside it) gives reference 1's ``Y`` and ``dX`` rows and ``dW``."""
    device = require(P, "fwd", "dgrad", "wgrad", out_dtype=torch.bfloat16)
    layout = small("fwd", sizes=(300, 0, 513, 1, 700, 129), tail=31)
    x, w, g, table = inputs(layout, device, seed=25)
    e0, e1 = 2, 5
    r0, r1 = layout.bounds(e0, e1)
    xs = poisoned(x, r0, r1).clone().requires_grad_(True)
    wc = w[e0:e1].clone().requires_grad_(True)
    y = P.gemm(xs, wc, table[e0 : e1 + 1])
    y.backward(poisoned(g, r0, r1))
    ends = table[e0 + 1 : e1 + 1] - table[e0]
    y1 = P.prepare["fwd"](padded(x, r0, r1), w[e0:e1], ends).launch()
    dx1 = P.prepare["dgrad"](padded(g, r0, r1), w[e0:e1], ends).launch()
    dw1 = P.prepare["wgrad"](padded(g, r0, r1), padded(x, r0, r1), ends).launch()
    torch.cuda.synchronize(device)
    assert_same_bits(y.detach()[r0:r1], y1[: r1 - r0], "autograd Y")
    assert_same_bits(xs.grad[r0:r1], dx1[: r1 - r0], "autograd dX")
    assert_same_bits(wc.grad, dw1, "autograd dW")


def check_autograd_chunks(P: Package) -> None:
    """Chunks of one weight (views of one leaf) over one ``x`` leaf, one chunk per backward
    (``x.grad`` and ``w.grad`` cleared between them): each chunk's ``dW`` and its rows of ``dX``
    are reference 1's.  Summing the chunks' ``x.grad`` is not supported: the rows outside a chunk
    are left uninitialized (the docstring says so)."""
    device = require(P, "fwd", "dgrad", "wgrad", out_dtype=torch.bfloat16)
    layout = small("fwd", tail=31)
    x, w, g, table = inputs(layout, device, seed=26)
    x_leaf = x.clone().requires_grad_(True)
    w_leaf = w.clone().requires_grad_(True)
    assert "cannot be summed" in " ".join(P.gemm.__doc__.split())
    for e0, e1 in SMALL_CHUNKS:
        r0, r1 = layout.bounds(e0, e1)
        x_leaf.grad = w_leaf.grad = None
        y = P.gemm(x_leaf, w_leaf[e0:e1], table[e0 : e1 + 1])
        y.backward(poisoned(g, r0, r1))
        ends = table[e0 + 1 : e1 + 1] - table[e0]
        dx1 = P.prepare["dgrad"](padded(g, r0, r1), w[e0:e1], ends).launch()
        dw1 = P.prepare["wgrad"](padded(g, r0, r1), padded(x, r0, r1), ends).launch()
        torch.cuda.synchronize(device)
        what = f"autograd chunk [{e0}, {e1})"
        assert_same_bits(x_leaf.grad[r0:r1], dx1[: r1 - r0], f"{what} dX")
        assert_same_bits(w_leaf.grad[e0:e1], dw1, f"{what} dW")
        others = torch.cat([w_leaf.grad[:e0].flatten(), w_leaf.grad[e1:].flatten()])
        assert_all_word(others, POS_ZERO, f"{what}: dW of the other groups")


def check_accumulate(monkeypatch, P: Package, out_dtype, route: str) -> None:
    """A slice accumulating into a finite ``out`` (``main_grad``) on the ``route`` route
    (``composed`` forced by an unloadable accumulate build) gives the references' bits, the split
    tail with group 0 included (the 512-wide k tile)."""
    device = require(P, "wgrad", out_dtype=out_dtype)
    accumulate_route(monkeypatch, P, device, out_dtype, route)
    layout = small("wgrad", shape=(512, 1024))
    operands = inputs(layout, device, seed=27)
    cases = [(layout, e0, e1, operands) for e0, e1 in SMALL_CHUNKS]
    split = split_tail_layout(
        P, device, arch_of(P, device), (1024, 2048, 4096), out_dtype
    )
    if split is not None:
        cases.append((split, 1, 2, inputs(split, device, seed=28)))
    for layout, e0, e1, ops in cases:
        init = random_like_out(
            (e1 - e0, layout.n, layout.k), out_dtype, device, seed=29
        )
        with quiet():
            s = build_slice(
                P,
                "wgrad",
                layout,
                e0,
                e1,
                device,
                out_dtype=out_dtype,
                accumulate=True,
                init=init,
                operands=ops,
            )
        assert s.lead.plan["accumulate"] == route, s.lead.plan
        s.lead.launch()
        s.launch_refs()
        torch.cuda.synchronize(device)
        s.check()


def check_base_host(monkeypatch, P: Package, op: str) -> None:
    """Negative control: the host before row tables bound ``m_indptr[1:]`` to the route's
    own programs, which start group 0 at row 0.  With that binding the same slice reads (the
    weight gradient) or writes (``Y`` / ``dX``) rows before ``m_indptr[0]`` and misses reference 1;
    the lead binding gives its bits (:func:`check_two_references`)."""
    device = require(P, op)
    layout = small(op, sizes=(300, 513, 1, 700), tail=9)
    e0, e1 = 1, 3
    s = build_slice(P, op, layout, e0, e1, device, seed=30)

    def old_row_table(offs, num_groups):
        if (
            offs.dtype == torch.int32
            and offs.ndim == 1
            and offs.numel() == num_groups + 1
        ):
            return offs[1:], None
        return offs, None

    monkeypatch.setattr(P.cb, "_row_table", old_row_table)
    x, w, g, table = inputs(layout, device, seed=30)
    r0, r1 = layout.bounds(e0, e1)
    chunk = table[e0 : e1 + 1]
    if op == "wgrad":
        base = P.prepare[op](
            poisoned(g, r0, r1), poisoned(x, r0, r1), chunk, num_groups=e1 - e0
        )
    else:
        a = x if op == "fwd" else g
        cols = layout.n if op == "fwd" else layout.k
        out = sentinel((layout.rows, cols), torch.bfloat16, device)
        base = P.prepare[op](poisoned(a, r0, r1), w[e0:e1], chunk, out=out)
    assert not any(P.jit.is_lead_route(r) for r in base.bound_routes), base.bound_routes
    s.ref1.launch()
    base.launch()
    torch.cuda.synchronize(device)
    if op == "wgrad":
        assert differing(base.out[0], s.ref1.out[0]) > 0, (
            "the base binding read group 0 right"
        )
    else:
        written = bits(base.out[:r0]) != SENTINEL[torch.bfloat16]
        assert bool(written.any()), "the base binding wrote no row before m_indptr[0]"
