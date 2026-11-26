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

Tests of the ``locality="domain"`` route of the Cake activation
quantize-dequantize forward: the interleaved-domain programs of SM107 (R200),
the memory whose 2 MB chunks alternate between the two dies
(``cake_act_qdq_det_train_domain_empty``) and the route's plan
(``cake_act_qdq_det_train_domain_plan``).

The ``cpu_only`` tests check the pure parts: the interleaved-domain records,
where a byte range lies in such memory, the reasons of the route in their
order, and the launch arguments and grid, against the kernel's own decoding of
its items; and, against a fake driver, the unwind of a mapping that fails (out
of memory, retried once; any other error, not retried) and the error without
cuda-python; the free of a range queued while another call holds the module's
lock; a device's first claim-counter mapping (no room for the counters raises
and records nothing, a failure after they map frees their range once and
reports the first error, and the counters are zeroed and the stream
synchronized before the device's entry is published); the per-thread default
stream's key per thread; the two dies' rows of a claim workspace; and the
read-only launch arguments a placed call keeps.  The
GPU tests check that ``"domain"`` runs the ``"off"`` program on other
architectures and on default memory, that ``"off"`` loads none of the route's
state, that a launch keys its claim workspace by torch's current stream, and
that on the per-thread default stream each thread takes its own claim
workspace.  The
SM107 tests (skipped elsewhere; an SM107 whose driver reports fewer than two
locality domains fails them) check the allocation (dies, registry, zero copy,
the free, the free deferred past a capture, the refusal during capture, and
with two SM107s both during a capture on the other device), the route's bytes
against the ``"off"`` route for every format with and without the scales (and
a caller's scales buffer), out of place above the route's first size and at
offsets inside a chunk, every domain program against the FP64 rule on any grid
and on the plain programs' rule inputs (edge rows, every finite BF16 under
every scale, group scales from 2^-120 to 2^120), each plain reason, the claim
counters left zero, a launch that raises, two streams at once, each die's
claim counters in that die's memory, an eager call with no slot left, CUDA
graph replay with a graph workspace and with none left, and steady-state calls
without host synchronization.

This suite names the backend package once, through
``tests.cake.test_helpers.cake_package``.
"""

from __future__ import annotations

import collections
import enum
import gc
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_helpers import cake_act_qdq_det_train_reference as reference

PACKAGE = "cake.cake_act_qdq_det_train"
pkg = cake_package(PACKAGE)
loader = cake_package(PACKAGE, "cake_jit")
memory = cake_package(PACKAGE, "cake_domain")
forward = pkg.cake_act_qdq_det_train_forward
plan = pkg.cake_act_qdq_det_train_domain_plan
domain_empty = pkg.cake_act_qdq_det_train_domain_empty

FORMATS = reference.FORMATS
KINDS = tuple(loader.KINDS)
ARCH = "sm_107a"
# The chunk of the memory that alternates dies; the bytes of the tile of the interleaved-domain programs, a whole
# number of every program's tiles; and the bytes of each format's tile.
CHUNK = 2097152
TILE_BYTES = 1024 * 16
DOMAIN_TILE_BYTES = {"fp4_e4m3_g16": 1024 * 16, "mxfp4_e8m0_g32": 1024 * 16, "mxfp8_e8m0_g32": 1024 * 16}
BF16 = 2
# The width of the inputs of the size tests: 12 vectors a row, so a size just above a bound ends inside a tile.
WIDTH = 96


def _domain_name(kind: str) -> str:
    return loader.DOMAIN_ROUTES[ARCH][kind]


def _first(kind: str) -> int:
    """The first ``n_vecs`` the interleaved-domain program of route ``kind`` runs."""
    return int(loader.MODULES[_domain_name(kind)]["vectors"][ARCH]["out_of_place"][0])


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _assert_same(what: str, got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype, what
    if not torch.equal(_bits(got), _bits(want)):
        bad = (_bits(got) != _bits(want)).nonzero()
        raise AssertionError(f"{what}: {bad.shape[0]} of {got.numel()} values differ; first at {bad[0].tolist()}")


def _spy(monkeypatch) -> list[str]:
    """The names of the records launched through ``cake_jit.run_module`` from now on."""
    launched: list[str] = []
    run_module = loader.run_module

    def spy(name, values, **kwargs):
        launched.append(name)
        return run_module(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", spy)
    return launched


# --------------------------------------------------------------------------- CPU-only: the pure parts


@pytest.mark.cpu_only
def test_domain_records_serve_their_routes(monkeypatch) -> None:
    for arch in loader.ARCHES:
        for fmt in FORMATS:
            for scale_out in (False, True):
                kind = loader.route_name(fmt, scale_out=scale_out)
                name = loader.domain_module(arch, fmt, scale_out=scale_out)
                assert name == (kind + "_domain" if arch == ARCH else None), (arch, kind, name)
                if name is not None:
                    assert name not in loader.ROUTES[arch][kind]
    assert loader.domain_chunk_bytes(ARCH) == CHUNK
    assert loader.domain_chunk_bytes("sm_100a") is None and loader.domain_chunk_bytes("sm_103a") is None
    with pytest.raises(ValueError, match="format"):
        loader.domain_module(ARCH, "nvfp4", scale_out=False)
    monkeypatch.setitem(loader.DOMAIN_ROUTES[ARCH], "fp4_e4m3_g16", "fp4_e4m3_g16_scales_domain")
    with pytest.raises(RuntimeError, match="does not serve that route"):
        loader.domain_module(ARCH, "fp4_e4m3_g16", scale_out=False)
    monkeypatch.undo()
    # the claims leave unsigned counters zero: no other word type, and no host re-arm
    record = loader.MODULES["mxfp8_e8m0_g32_domain"]
    for change in ({"rearm": "host"}, {"rearm": "never"}, {"dtype": "int32"}, {"words": 0}):
        monkeypatch.setitem(record, "counters", {**record["counters"], **change})
        with pytest.raises(RuntimeError, match="does not serve that route"):
            loader.domain_module(ARCH, "mxfp8_e8m0_g32", scale_out=False)
        monkeypatch.undo()


@pytest.mark.cpu_only
def test_chunk_place() -> None:
    """Offset in its chunk and die of the chunk (chunk k in die k & 1) of a byte range that lies in the buffer."""
    place = loader.chunk_place
    base, size = 1 << 40, 3 * CHUNK
    assert place(base, 16, base, size, CHUNK) == (0, 0)
    assert place(base + CHUNK + 48, 16, base, size, CHUNK) == (48, 1)
    assert place(base + 2 * CHUNK - 16, 32, base, size, CHUNK) == (CHUNK - 16, 1)
    assert place(base + 2 * CHUNK + TILE_BYTES, CHUNK - TILE_BYTES, base, size, CHUNK) == (TILE_BYTES, 0)
    # one byte past the end, a start before the base or at the end: not in the buffer
    assert place(base + 2 * CHUNK + TILE_BYTES, CHUNK - TILE_BYTES + 1, base, size, CHUNK) is None
    assert place(base - 16, 16, base, size, CHUNK) is None
    assert place(base + size, 0, base, size, CHUNK) is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", KINDS)
def test_domain_route_reasons_in_order(kind: str) -> None:
    fmt = kind.removesuffix("_scales")
    scale_out = kind.endswith("_scales")
    name = _domain_name(kind)
    first, last = loader.MODULES[name]["vectors"][ARCH]["out_of_place"]
    assert loader.MODULES[name]["vectors"][ARCH]["in_place"] is None
    assert 1 <= first <= last <= loader.MAX_VECTORS
    good = {
        "scale_out": scale_out,
        "n_vecs": first,
        "in_place": False,
        "out_given": True,
        "x_place": (5 * TILE_BYTES + 48, 1),
        "y_place": (5 * TILE_BYTES + 48, 1),
    }

    def route(arch: str = ARCH, **change):
        return loader.domain_route(arch, fmt, **{**good, **change})

    def split(arch: str = ARCH, **change):
        # the call path: the checks without a placement, then the placement checks of a candidate
        args = {**good, **change}
        x_place, y_place = args.pop("x_place"), args.pop("y_place")
        candidate, reason = loader.domain_candidate(arch, fmt, **args)
        if candidate is None:
            return None, reason
        assert reason == "candidate"
        reason = loader.domain_placement(x_place, y_place)
        return (candidate, reason) if reason == "placed" else (None, reason)

    assert route() == split() == (name, "placed")
    assert route(n_vecs=last) == (name, "placed")
    assert route(x_place=(0, 0), y_place=(0, 0)) == (name, "placed")
    seen = {"placed"}
    for change, reason in (
        ({"arch": "sm_100a"}, "arch"),
        ({"arch": "sm_103a"}, "arch"),
        ({"in_place": True}, "in_place"),
        ({"out_given": False, "y_place": None}, "no_out"),
        ({"n_vecs": first - 1}, "size"),
        ({"n_vecs": last + 1}, "size"),
        ({"x_place": None}, "x_memory"),
        ({"y_place": None}, "out_memory"),
        ({"y_place": (5 * TILE_BYTES + 64, 1)}, "chunk_offset"),
        ({"y_place": (5 * TILE_BYTES + 48, 0)}, "die_parity"),
    ):
        assert route(**change) == split(**change) == (None, reason), change
        seen.add(reason)
    assert seen == set(loader.DOMAIN_REASONS)
    # the first reason that holds
    assert route(in_place=True, n_vecs=first - 1, x_place=None) == (None, "in_place")
    assert route(out_given=False, n_vecs=first - 1, x_place=None) == (None, "no_out")
    assert route(n_vecs=first - 1, x_place=None, y_place=None) == (None, "size")
    assert route(x_place=None, y_place=None) == (None, "x_memory")
    assert route(y_place=(0, 0)) == (None, "chunk_offset")


def _kernel_item(slot: int, first_die: int, shift: int, batch_shift: int) -> tuple[int, int]:
    """``(die, item)`` of chunk slot ``slot``, as the kernel numbers its items: chunk ``k`` lies in die
    ``first_die ^ (k & 1)``, and die ``d``'s items run over its chunks in order, ``2^(shift - batch_shift)`` a chunk."""
    chunk = slot >> shift
    die = first_die ^ (chunk & 1)
    item = ((chunk >> 1) << (shift - batch_shift)) + ((slot & ((1 << shift) - 1)) >> batch_shift)
    return die, item


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("name", "tile", "chunk_tile_shift", "ctas_per_sm"),
    (("fp4_e4m3_g16_domain", 1024, 7, 7), ("mxfp8_e8m0_g32_domain", 1024, 7, 8)),
    ids=("fp4", "mx"),
)
def test_domain_launch_covers_every_tile(name: str, tile: int, chunk_tile_shift: int, ctas_per_sm: int) -> None:
    """``domain_launch``: the first tile rounds down, the chunk holds the format's tiles (128 of 16,384 bytes), an item
    is one of them, and the grid is one CTA per item of the chunks that hold the input, at most 7 (FP4) or 8 (MXFP4,
    MXFP8) a SM.  Every tile of the input lies in an item below its
    die's item count, as the kernel decodes the item back to its slots."""
    record = loader.MODULES[name]
    tile_bytes = DOMAIN_TILE_BYTES[record["format"]]
    chunk_vectors = CHUNK // 16
    sms = 212
    resident = ctas_per_sm * sms

    def launch(n_vecs, offset, die):
        return loader.domain_launch(record, n_vecs=n_vecs, x_place=(offset, die), sm_count=sms, arch=ARCH)

    assert launch(chunk_vectors, 0, 0) == {
        "first_tile": 0,
        "first_die": 0,
        "chunk_tile_shift": chunk_tile_shift,
        "batch_shift": 0,
        "grid": (1 << chunk_tile_shift, 1, 1),
    }
    got = launch(chunk_vectors + 1, 3 * tile_bytes + 48, 1)
    assert (got["first_tile"], got["first_die"], got["grid"]) == (3, 1, (2 << chunk_tile_shift, 1, 1))
    assert launch(record["vectors"][ARCH]["out_of_place"][1], 0, 0)["grid"] == (resident, 1, 1)
    sizes = (1, tile - 1, tile, tile + 1, 4 * tile + 3, chunk_vectors - 1, chunk_vectors, 3 * chunk_vectors + 7)
    for offset in (0, 16, tile_bytes, 3 * tile_bytes + 48, CHUNK - tile_bytes, CHUNK - 16):
        for n_vecs in sizes:
            for first_die in (0, 1):
                got = launch(n_vecs, offset, first_die)
                shift, batch_shift = got["chunk_tile_shift"], got["batch_shift"]
                first_tile = got["first_tile"]
                assert first_tile == offset // tile_bytes and got["first_die"] == first_die
                tiles = -(-n_vecs // tile)
                last_chunk = (first_tile + tiles - 1) >> shift
                items = {die: ((last_chunk - (die ^ first_die) + 2) >> 1) << (shift - batch_shift) for die in (0, 1)}
                assert got["grid"] == (min(items[0] + items[1], resident), 1, 1)
                for c in range(tiles):
                    die, item = _kernel_item(first_tile + c, first_die, shift, batch_shift)
                    assert item < items[die], (offset, n_vecs, first_die, c)
                    # the kernel's decode: the item's chunk and slots hold tile c
                    chunk = (die ^ first_die) + (item >> (shift - batch_shift) << 1)
                    slot = (chunk << shift) + ((item & ((1 << (shift - batch_shift)) - 1)) << batch_shift)
                    assert slot <= first_tile + c < slot + (1 << batch_shift)
    with pytest.raises(ValueError, match="interleaved-domain grid rule"):
        loader.domain_launch(loader.MODULES["mxfp8_e8m0_g32"], n_vecs=1, x_place=(0, 0), sm_count=sms, arch=ARCH)
    with pytest.raises(ValueError, match="unknown"):
        loader.launch_grid(record, n_vecs=1, sm_count=sms, arch=ARCH)


CUresult = enum.Enum("CUresult", ("CUDA_SUCCESS", "CUDA_ERROR_OUT_OF_MEMORY", "CUDA_ERROR_INVALID_VALUE"))
OK = CUresult.CUDA_SUCCESS


class _FakeDriver:
    """The driver calls of ``cake_domain._map`` without a GPU.  ``call`` returns ``status`` at chunk ``k`` of every
    attempt (``cuMemSetAccess`` at any chunk); ``log`` records each reservation, created handle, mapped and unmapped
    chunk, released handle and freed reservation as ``(what, value)``."""

    CUresult = CUresult
    CUdeviceptr = int
    CUmemAllocationType = SimpleNamespace(CU_MEM_ALLOCATION_TYPE_PINNED=1)
    CUmemLocationType = SimpleNamespace(CU_MEM_LOCATION_TYPE_DEVICE=1, CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN=2)
    CUmemAllocationGranularity_flags = SimpleNamespace(CU_MEM_ALLOC_GRANULARITY_MINIMUM=0)
    CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

    def __init__(self, call: str, k: int, status: CUresult) -> None:
        self.call, self.k, self.status = call, k, status
        self.log: list[tuple[str, int]] = []
        self.base = 0
        self.created = 0
        self.handles = 0
        self.contexts = 0

    def _status(self, call: str, k: int) -> CUresult:
        return self.status if call == self.call and (k == self.k or call == "cuMemSetAccess") else OK

    @staticmethod
    def CUmemAllocationProp() -> SimpleNamespace:
        return SimpleNamespace(type=None, location=SimpleNamespace(type=None, id=None))

    @staticmethod
    def CUmemAccessDesc() -> SimpleNamespace:
        return SimpleNamespace(location=SimpleNamespace(type=None, id=None), flags=None)

    def cuInit(self, flags):
        return (OK,)

    def cuDeviceGet(self, index):
        return (OK, index)

    def cuDevicePrimaryCtxRetain(self, device):
        self.contexts += 1
        return (OK, "context")

    def cuDevicePrimaryCtxRelease(self, device):
        self.contexts -= 1
        return (OK,)

    def cuCtxPushCurrent(self, context):
        return (OK,)

    def cuCtxPopCurrent(self):
        return (OK, "context")

    def cuMemGetAllocationGranularity(self, prop, flags):
        return (OK, CHUNK)

    def cuMemAddressReserve(self, size, alignment, address, flags):
        self.base += 1 << 40
        self.created = 0
        self.log.append(("reserve", self.base))
        return (OK, self.base)

    def cuMemCreate(self, size, prop, flags):
        status = self._status("cuMemCreate", self.created)
        self.created += 1
        if status != OK:
            return (status, None)
        self.handles += 1
        self.log.append(("create", self.handles))
        return (OK, self.handles)

    def cuMemMap(self, address, size, offset, handle, flags):
        k = (address - self.base) // CHUNK
        status = self._status("cuMemMap", k)
        if status == OK:
            self.log.append(("map", k))
        return (status,)

    def cuMemRelease(self, handle):
        self.log.append(("release", handle))
        return (OK,)

    def cuMemSetAccess(self, address, size, descriptors, count):
        return (self._status("cuMemSetAccess", 0),)

    def cuMemUnmap(self, address, size):
        self.log.append(("unmap", (address - self.base) // CHUNK))
        return (OK,)

    def cuMemAddressFree(self, address, size):
        self.log.append(("free", address))
        return (OK,)


def _attempts(fake: _FakeDriver) -> list[dict[str, list[int]]]:
    """The fake's log split at each reservation: per attempt, the values of each kind of call in order."""
    attempts: list[dict[str, list[int]]] = []
    for what, value in fake.log:
        if what == "reserve":
            attempts.append({kind: [] for kind in ("reserve", "create", "map", "release", "unmap", "free")})
        attempts[-1][what].append(value)
    return attempts


def _fake_driver(monkeypatch, call: str, k: int, status: CUresult) -> tuple[_FakeDriver, list[int]]:
    """A fake driver as the module's driver, and the list ``torch.cuda.empty_cache`` appends to."""
    fake = _FakeDriver(call, k, status)
    monkeypatch.setattr(memory, "_driver", lambda: fake)
    emptied: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    return fake, emptied


@pytest.mark.cpu_only
def test_out_of_memory_unwinds_each_attempt(monkeypatch) -> None:
    """``cuMemCreate`` out of memory at chunk 3: each of the two attempts unmaps chunks 0 to 2, releases every handle
    it made and frees its reservation once; torch's cache is returned to the driver once between them, then
    ``torch.cuda.OutOfMemoryError``, and the registry is unchanged."""
    k, chunks = 3, 8
    before = memory.registered(0)
    fake, emptied = _fake_driver(monkeypatch, "cuMemCreate", k, CUresult.CUDA_ERROR_OUT_OF_MEMORY)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="no room"):
        memory._map_or_out_of_memory(fake, 0, CHUNK, chunks)
    assert len(emptied) == 1
    attempts = _attempts(fake)
    assert len(attempts) == 2
    for attempt in attempts:
        assert len(attempt["create"]) == k
        assert attempt["map"] == list(range(k)) and attempt["unmap"] == list(range(k))
        assert sorted(attempt["release"]) == attempt["create"]
        assert attempt["free"] == attempt["reserve"]
    assert fake.contexts == 0
    assert memory.registered(0) == before


@pytest.mark.cpu_only
@pytest.mark.parametrize("call", ("cuMemMap", "cuMemSetAccess"))
def test_a_failed_mapping_unwinds_without_retry(call: str, monkeypatch) -> None:
    """A ``cuMemMap`` that fails at chunk 3 still releases that chunk's handle and unmaps chunks 0 to 2; a
    ``cuMemSetAccess`` that fails unmaps every chunk.  Either frees the reservation once and, with a status other than
    out of memory, raises the module's driver error with no second attempt."""
    k, chunks = 3, 8
    fake, emptied = _fake_driver(monkeypatch, call, k, CUresult.CUDA_ERROR_INVALID_VALUE)
    with pytest.raises(memory.DriverError, match=call) as raised:
        memory._map_or_out_of_memory(fake, 0, CHUNK, chunks)
    assert raised.value.status is CUresult.CUDA_ERROR_INVALID_VALUE
    assert emptied == []
    (attempt,) = _attempts(fake)
    mapped = k if call == "cuMemMap" else chunks
    assert len(attempt["create"]) == (k + 1 if call == "cuMemMap" else chunks)
    assert sorted(attempt["release"]) == attempt["create"]
    assert attempt["map"] == list(range(mapped)) and attempt["unmap"] == list(range(mapped))
    assert attempt["free"] == attempt["reserve"]
    assert fake.contexts == 0


@pytest.mark.cpu_only
def test_driver_needs_cuda_python(monkeypatch) -> None:
    """Without cuda-python, or with one that lacks the locality-domain names, the driver module is a
    ``RuntimeError`` (the uncached function, so the module's cache is untouched)."""
    monkeypatch.setitem(sys.modules, "cuda.bindings", None)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", None)
    with pytest.raises(RuntimeError, match="cuda-python"):
        memory._driver.__wrapped__()
    bindings = ModuleType("cuda.bindings")
    bindings.driver = ModuleType("cuda.bindings.driver")
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    with pytest.raises(RuntimeError, match="does not name"):
        memory._driver.__wrapped__()


@pytest.mark.cpu_only
def test_a_range_queued_under_the_lock_is_freed_when_the_lock_is_released(monkeypatch) -> None:
    """A finalizer that finds the module's lock taken only queues its range; the call that holds the lock frees it
    as it releases it: here :func:`domain_empty`, whose mapping fails after the finalizer ran.  A pass that can free
    nothing (during a capture) leaves the range queued for a later call."""
    freed: list = []
    monkeypatch.setattr(memory, "_PENDING", collections.deque())
    monkeypatch.setattr(memory, "_free", freed.append)
    monkeypatch.setattr(memory, "_unregister", lambda index, base: None)
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    monkeypatch.setattr(memory, "_driver", lambda: SimpleNamespace())
    monkeypatch.setattr(memory, "domain_count", lambda index: memory.DIES)
    monkeypatch.setattr(memory, "_make_graph_workspaces", lambda index: None)
    monkeypatch.setattr(memory.cake_jit, "device_arch", lambda index: ARCH)
    entry = memory._Mapped(0, 1 << 30, CHUNK, CHUNK)
    queued: list = []

    def map_after_a_finalizer(driver, index, chunk, chunks):
        memory._release(entry)  # the finalizer of a range whose last view died while domain_empty holds the lock
        queued.append((list(freed), list(memory._PENDING)))
        raise RuntimeError("mapping refused")

    monkeypatch.setattr(memory, "_map_or_out_of_memory", map_after_a_finalizer)
    with pytest.raises(RuntimeError, match="mapping refused"):
        memory.domain_empty((4, WIDTH), device="cuda:0")
    assert queued == [([], [entry])], queued
    assert freed == [entry] and not memory._PENDING
    monkeypatch.setattr(memory, "_capturing", lambda: True)
    memory._release(entry)
    assert freed == [entry] and list(memory._PENDING) == [entry]
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    memory._drain()
    assert freed == [entry, entry] and not memory._PENDING


def _stub_counter_mapping(monkeypatch) -> list:
    """The stubs of a device's first claim-counter mapping on the CPU: no device has its counters yet, the driver is
    a stand-in, and ``_free`` records what it frees (the list returned).  Each test patches ``_map_or_out_of_memory``
    itself."""
    freed: list = []
    monkeypatch.setattr(memory, "_GRAPH_WORKSPACES", {})
    monkeypatch.setattr(memory, "_PENDING", collections.deque())
    monkeypatch.setattr(memory, "_free", freed.append)
    monkeypatch.setattr(memory, "_unregister", lambda index, base: None)
    monkeypatch.setattr(memory, "_capturing", lambda: False)
    monkeypatch.setattr(memory, "_driver", lambda: SimpleNamespace())
    monkeypatch.setattr(memory, "domain_count", lambda index: memory.DIES)
    monkeypatch.setattr(memory.cake_jit, "device_arch", lambda index: ARCH)
    return freed


@pytest.mark.cpu_only
def test_no_room_for_the_claim_counters_records_nothing(monkeypatch) -> None:
    """A device's first :func:`domain_empty` maps its claim counters (one chunk in each die) before the tensor; when
    the device has no room for them, the call raises ``torch.cuda.OutOfMemoryError``, maps no tensor and records no
    counters, so the next call maps them again."""
    mapped: list = []
    freed = _stub_counter_mapping(monkeypatch)

    def no_room(driver, index, chunk, chunks):
        mapped.append((index, chunk, chunks))
        raise torch.cuda.OutOfMemoryError("no room for the claim counters")

    monkeypatch.setattr(memory, "_map_or_out_of_memory", no_room)
    for _ in range(2):
        with pytest.raises(torch.cuda.OutOfMemoryError, match="claim counters"):
            memory.domain_empty((4, WIDTH), device="cuda:0")
    assert mapped == [(0, CHUNK, memory.DIES)] * 2
    assert freed == [] and memory._GRAPH_WORKSPACES == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize("free_fails", (False, True))
def test_a_failure_after_the_claim_counters_map_frees_them(free_fails: bool, monkeypatch) -> None:
    """A step after the claim counters map (here their export as a tensor) that raises frees their range once and
    records nothing; when that free raises too, the first error is the one reported."""
    mapped: list = []
    freed = _stub_counter_mapping(monkeypatch)
    base = 1 << 30

    def counters(driver, index, chunk, chunks):
        mapped.append((index, chunk, chunks))
        return base, chunk * chunks

    def free(entry) -> None:
        freed.append(entry)
        if free_fails:
            raise RuntimeError("free refused")

    def refused(producer):
        raise RuntimeError("export refused")

    monkeypatch.setattr(memory, "_map_or_out_of_memory", counters)
    monkeypatch.setattr(memory, "_free", free)
    monkeypatch.setattr(torch, "from_dlpack", refused)
    with pytest.raises(RuntimeError, match="export refused"):
        memory._make_graph_workspaces(0)
    assert mapped == [(0, CHUNK, memory.DIES)]
    (entry,) = freed
    assert (entry.index, entry.base, entry.size, entry.chunk) == (0, base, memory.DIES * CHUNK, CHUNK)
    assert entry.unmapped == 0
    assert memory._GRAPH_WORKSPACES == {}


@pytest.mark.cpu_only
def test_the_claim_counters_are_zeroed_and_synchronized_before_they_are_published(monkeypatch) -> None:
    """The claim counters' block is zeroed, then the current stream is synchronized, and only then is the device's
    entry published (a reader without the lock finds no entry or a zeroed block); a later call maps nothing."""
    mapped: list = []
    freed = _stub_counter_mapping(monkeypatch)
    events: list = []

    class Block:
        def view(self, *shape):
            events.append(("view", shape))
            return self

        def zero_(self):
            events.append(("zero", 0 in memory._GRAPH_WORKSPACES))
            return self

    block = Block()

    def counters(driver, index, chunk, chunks):
        mapped.append((index, chunk, chunks))
        return 1 << 30, chunk * chunks

    def stream(device):
        assert device == torch.device("cuda", 0)
        return SimpleNamespace(synchronize=lambda: events.append(("sync", 0 in memory._GRAPH_WORKSPACES)))

    monkeypatch.setattr(memory, "_map_or_out_of_memory", counters)
    monkeypatch.setattr(torch, "from_dlpack", lambda producer: block)
    monkeypatch.setattr(torch.cuda, "current_stream", stream)
    memory._make_graph_workspaces(0)
    memory._make_graph_workspaces(0)
    assert events == [("view", (torch.int32,)), ("view", (memory.DIES, CHUNK // 4)), ("zero", False), ("sync", False)]
    assert mapped == [(0, CHUNK, memory.DIES)] and freed == []
    assert memory._GRAPH_WORKSPACES == {
        0: {"block": block, "taken": 0, "eager": memory.GRAPH_WORKSPACES, "slots": CHUNK // (4 * memory.SLOT_WORDS)}
    }


@pytest.mark.cpu_only
def test_the_per_thread_default_stream_has_a_key_per_thread() -> None:
    """The per-thread default stream's handle names another stream in each thread, so its workspaces are keyed by
    the thread's token, which no later thread reuses; any other handle names one stream."""
    keys: list = []
    for _ in range(2):
        thread = threading.Thread(target=lambda: keys.append(memory._stream_key(0, memory.PER_THREAD_STREAM)))
        thread.start()
        thread.join()
    here = memory._stream_key(0, memory.PER_THREAD_STREAM)
    assert here == memory._stream_key(0, memory.PER_THREAD_STREAM)
    assert len({here, *keys}) == 3, (here, keys)
    assert all(key[:2] == (0, memory.PER_THREAD_STREAM) for key in (here, *keys))
    assert memory._stream_key(0, 0) == (0, 0)
    assert memory._stream_key(1, 0x7F00) == (1, 0x7F00)


@pytest.mark.cpu_only
def test_a_workspace_gives_each_die_its_row() -> None:
    """A claim workspace's counters are its two rows, die 0's first: one-dimensional contiguous unsigned views of
    ``words`` words each, as the binding takes them, over a slot of a larger block, kept per count."""
    block = torch.arange(memory.DIES * 4 * memory.SLOT_WORDS, dtype=torch.int32).view(memory.DIES, -1)
    words = block[:, memory.SLOT_WORDS : 2 * memory.SLOT_WORDS]
    workspace = memory._Workspace(words)
    for count in (1, 3):
        counters = workspace.counters(count)
        assert workspace.counters(count) is counters and len(counters) == memory.DIES
        for d, row in enumerate(counters):
            assert row.dtype == torch.uint32 and row.shape == (count,) and row.is_contiguous(), (count, d)
            assert row.data_ptr() == words[d].data_ptr(), (count, d)
            assert row.view(torch.int32).tolist() == words[d, :count].tolist(), (count, d)


@pytest.mark.cpu_only
def test_a_placed_call_keeps_its_launch_arguments_read_only() -> None:
    """A placed call's launch arguments and grid are ``domain_launch``'s, kept per argument set and read only, so the
    calls on one pair of buffers compute them once."""
    name = _domain_name(KINDS[0])
    n_vecs, x_place, sms = _first(KINDS[0]), (3 * TILE_BYTES + 16, 1), 212
    got = pkg._domain_launch(name, n_vecs, x_place, sms, ARCH)
    want = loader.domain_launch(loader.MODULES[name], n_vecs=n_vecs, x_place=x_place, sm_count=sms, arch=ARCH)
    assert dict(got) == want
    assert pkg._domain_launch(name, n_vecs, x_place, sms, ARCH) is got
    with pytest.raises(TypeError):
        got["grid"] = (1, 1, 1)


# --------------------------------------------------------------------------- GPU: every architecture


def _supported() -> bool:
    return torch.cuda.is_available() and pkg.is_cake_act_qdq_det_train_supported(torch.device("cuda", 0))


gpu = pytest.mark.skipif(not _supported(), reason="cake_act_qdq_det_train needs an SM100 / SM103 / SM107 device")


@gpu
@pytest.mark.parametrize("fmt", FORMATS)
def test_domain_on_default_memory_runs_the_off_program(fmt: str, monkeypatch) -> None:
    """On default memory (and on SM100 / SM103 everywhere) ``"domain"`` launches the program ``"off"`` launches, for
    the reason the route gives, with the same bytes."""
    arch = loader.device_arch(0)
    x = reference.randn_input((129, 512), fmt, seed=61)
    n_vecs = x.numel() // loader.VECTOR
    for scale_out in (False, True):
        want = forward(x, format=fmt, return_scales=scale_out)
        program = loader.selected_module(arch, fmt, scale_out=scale_out, n_vecs=n_vecs, in_place=False)
        for out in (None, torch.empty_like(x)):
            name, reason = loader.domain_route(
                arch,
                fmt,
                scale_out=scale_out,
                n_vecs=n_vecs,
                in_place=False,
                out_given=out is not None,
                x_place=None,
                y_place=None,
            )
            assert name is None
            got = plan(x, format=fmt, out=out, return_scales=scale_out)
            assert (got["route"], got["reason"], got["program"]) == ("plain", reason, program), got
            launched = _spy(monkeypatch)
            result = forward(x, format=fmt, out=out, return_scales=scale_out, locality="domain")
            assert launched == [program], launched
            monkeypatch.undo()
            pairs = zip(result, want, strict=True) if scale_out else ((result, want),)
            for what, (a, b) in zip(("y", "scales"), pairs):
                _assert_same(f"{fmt} scale_out={scale_out} out={out is not None} {what}", a, b)
    # in place: the reason is the route's, the program the in-place one
    y = x.clone()
    got = plan(y, format=fmt, out=y)
    assert (got["route"], got["reason"]) == ("plain", "arch" if arch != ARCH else "in_place"), got
    assert got["program"] == loader.selected_module(arch, fmt, scale_out=False, n_vecs=n_vecs, in_place=True)
    _assert_same(f"{fmt} in place", forward(y, format=fmt, out=y, locality="domain"), forward(x, format=fmt))


@gpu
def test_empty_input_plan() -> None:
    x = torch.empty((0, 512), device="cuda", dtype=torch.bfloat16)
    assert plan(x, format="mxfp8_e8m0_g32") == {
        "route": "plain",
        "reason": "empty",
        "program": None,
        "first_tile": None,
        "first_die": None,
        "grid": None,
    }
    assert forward(x, format="mxfp8_e8m0_g32", locality="domain").shape == (0, 512)


@gpu
def test_a_launch_keys_its_workspace_by_torchs_current_stream(monkeypatch) -> None:
    """An eager launch takes the claim workspace of torch's current stream of the device, on the default stream, a
    side stream and the per-thread default stream, with the stream handle read through torch's raw getter or,
    without it, through ``torch.cuda.current_stream``.  The kernel is replaced, so this runs on every
    architecture."""
    monkeypatch.setattr(memory.cake_jit, "run_module", lambda name, values, **kwargs: None)
    name = _domain_name(KINDS[0])
    side = torch.cuda.Stream()
    for getter in (memory._CURRENT_RAW_STREAM, None):
        monkeypatch.setattr(memory, "_CURRENT_RAW_STREAM", getter)
        monkeypatch.setattr(memory, "_STREAM_WORKSPACES", {})
        for stream in (None, side, torch.cuda.ExternalStream(memory.PER_THREAD_STREAM)):
            with torch.cuda.stream(stream):
                handle = torch.cuda.current_stream(0).cuda_stream
                assert memory._current_stream_handle(0) == handle, (getter, stream)
                assert memory.stream_workspace(0, handle) is None, (getter, stream)
                memory.launch(name, {}, device_index=0, grid=(1, 1, 1))
                assert memory.stream_workspace(0, handle) is not None, (getter, stream)
        assert len(memory._STREAM_WORKSPACES) == 3, (getter, list(memory._STREAM_WORKSPACES))


@gpu
def test_the_per_thread_default_stream_gives_each_thread_its_workspace(monkeypatch) -> None:
    """On the per-thread default stream (``cudaStreamPerThread``) an eager launch takes its own thread's claim
    workspace: two threads never share counters, and a thread's later launches reuse its own.  The kernel is
    replaced, so this runs on every architecture."""
    monkeypatch.setattr(memory, "_STREAM_WORKSPACES", {})
    monkeypatch.setattr(memory.cake_jit, "run_module", lambda name, values, **kwargs: None)
    name = _domain_name(KINDS[0])
    seen: dict[int, list[int]] = {}
    errors: list = []

    def calls(k: int) -> None:
        try:
            with torch.cuda.stream(torch.cuda.ExternalStream(memory.PER_THREAD_STREAM)):
                for _ in range(2):
                    memory.launch(name, {}, device_index=0, grid=(1, 1, 1))
                    seen.setdefault(k, []).append(memory.stream_workspace(0, memory.PER_THREAD_STREAM).data_ptr())
                torch.cuda.current_stream().synchronize()
        except BaseException as exc:  # noqa: BLE001 - reported by the test thread
            errors.append(exc)

    threads = [threading.Thread(target=calls, args=(k,)) for k in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert seen[0][0] == seen[0][1] and seen[1][0] == seen[1][1], seen
    assert seen[0][0] != seen[1][0], seen
    assert len(memory._STREAM_WORKSPACES) == 2


@gpu
def test_off_loads_no_domain_state() -> None:
    """In a fresh interpreter, ``"off"`` calls (and ``"domain"`` calls on SM100 / SM103) never import the module of
    the memory that alternates dies and of the claim workspaces."""
    script = (
        "import sys, torch\n"
        f"import {pkg.__name__} as b\n"
        "x = torch.randn(129, 512, device='cuda', dtype=torch.bfloat16)\n"
        "b.cake_act_qdq_det_train_forward(x, format='mxfp8_e8m0_g32')\n"
        "b.cake_act_qdq_det_train_forward(x, format='mxfp8_e8m0_g32', out=torch.empty_like(x), locality='off')\n"
        "if b.cake_jit.device_arch(0) != 'sm_107a':\n"
        "    b.cake_act_qdq_det_train_forward(x, format='mxfp8_e8m0_g32', out=torch.empty_like(x), locality='domain')\n"
        "    b.cake_act_qdq_det_train_domain_plan(x, format='mxfp8_e8m0_g32', out=torch.empty_like(x))\n"
        "torch.cuda.synchronize()\n"
        f"assert {memory.__name__!r} not in sys.modules, sorted(sys.modules)\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, (r.stdout, r.stderr)


@gpu
def test_domain_empty_refuses_other_architectures() -> None:
    if loader.device_arch(0) == ARCH:
        pytest.skip("SM107 is the architecture of the memory that alternates dies")
    with pytest.raises(RuntimeError, match="SM107"):
        domain_empty((4, 96))


# --------------------------------------------------------------------------- SM107 (R200)


@pytest.fixture(scope="module")
def r200() -> int:
    """Device 0, when it is an SM107.  A missing cuda-python, or an SM107 whose driver reports fewer than two
    locality domains, fails here: neither is a reason to skip."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 7):
        pytest.skip("the interleaved-domain route needs an SM107 (R200) device")
    count = memory.domain_count(0)
    if count < memory.DIES:
        pytest.fail(
            f"device 0 is an SM107 but reports {count} locality domain(s); the interleaved-domain route needs "
            f"{memory.DIES}: check the driver"
        )
    return 0


# Byte offsets of an input in its buffer: inside a tile of a die-0 chunk, and at a whole tile of a die-1 chunk.
OFFSETS = (5 * TILE_BYTES + 48, CHUNK + 7 * TILE_BYTES)


class _Big:
    """Two buffers of memory that alternate dies, room for an input just above every route's first size at each of
    :data:`OFFSETS` and one chunk further, and a third one for a second stream's output."""

    def __init__(self) -> None:
        first = max(_first(kind) for kind in KINDS)
        self.rows = first // (WIDTH // 8) + 1
        self.numel = self.rows * WIDTH
        elements = (max(OFFSETS) + CHUNK) // BF16 + self.numel + 64
        self.x_buf = domain_empty((elements,))
        self.y_buf = domain_empty((elements,))
        self.z_buf = domain_empty((elements,))
        self._inputs: dict[str, torch.Tensor] = {}

    def view(self, buf: torch.Tensor, offset: int) -> torch.Tensor:
        start = offset // BF16
        return buf[start : start + self.numel].view(self.rows, WIDTH)

    def input(self, fmt: str) -> torch.Tensor:
        if fmt not in self._inputs:
            self._inputs = {fmt: reference.randn_input((self.rows, WIDTH), fmt, seed=67)}
        return self._inputs[fmt]


@pytest.fixture(scope="module")
def big(r200) -> _Big:
    return _Big()


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32, torch.uint8))
def test_domain_empty_alternates_dies(r200, dtype) -> None:
    shape = (3, CHUNK // 2 + 7)
    t = domain_empty(shape, dtype=dtype)
    nbytes = t.numel() * t.element_size()
    chunks = -(-nbytes // CHUNK)
    assert t.shape == shape and t.dtype == dtype and t.device == torch.device("cuda", 0) and t.is_contiguous()
    base = t.data_ptr()
    assert base % CHUNK == 0
    assert (base, chunks * CHUNK) in memory.registered(0)
    for k in range(chunks):
        for at in (base + k * CHUNK, base + (k + 1) * CHUNK - 1):
            assert memory.domain_of(at) == k & 1, (k, at)
    flat = t.view(-1)
    stride = CHUNK // t.element_size()
    assert memory.place(flat[stride + 3 : stride + 11], CHUNK) == (3 * t.element_size(), 1)
    assert memory.place(flat, CHUNK) == (0, 0)
    # zero copy: the tensor is the registered range itself, and a write through a view is read through another
    t.zero_()
    flat[stride] = 1
    torch.cuda.synchronize()
    assert t.view(-1).nonzero().view(-1).tolist() == [stride]
    # every spelling of the device
    for device in (None, 0, "cuda", "cuda:0", torch.device("cuda", 0)):
        u = domain_empty((4, WIDTH), device=device)
        assert u.device == torch.device("cuda", 0) and u.dtype == torch.bfloat16 and u.shape == (4, WIDTH)
        del u
    # the last view gone, the range is unregistered and freed
    del t, flat
    gc.collect()
    assert all(entry[0] != base for entry in memory.registered(0))


def test_domain_empty_refuses_capture(r200) -> None:
    domain_empty((4, WIDTH))  # the graph workspaces exist before the capture
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="capture"):
        with torch.cuda.graph(graph):
            domain_empty((4, WIDTH))


def test_a_free_during_capture_waits_for_the_next_allocation(r200) -> None:
    """A buffer whose last view goes during a capture stays registered (and mapped) through the capture; the next
    ``domain_empty`` call frees and unregisters it."""
    t = domain_empty((4, WIDTH))
    base = t.data_ptr()
    scratch = torch.zeros(16, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scratch.add_(1)
        del t
        gc.collect()
        assert any(entry[0] == base for entry in memory.registered(0))
    assert any(entry[0] == base for entry in memory.registered(0))
    graph.replay()
    torch.cuda.synchronize()
    assert scratch.tolist() == [1.0] * 16
    domain_empty((4, WIDTH))
    assert all(entry[0] != base for entry in memory.registered(0))


@pytest.fixture(scope="module")
def r200_pair(r200) -> int:
    """Device 1, when devices 0 and 1 are both SM107s whose drivers report two locality domains; skipped otherwise."""
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability(1) != (10, 7):
        pytest.skip("a capture on one device with memory on another needs two SM107 (R200) devices")
    if memory.domain_count(1) < memory.DIES:
        pytest.skip(f"device 1 reports fewer than {memory.DIES} locality domains")
    return 1


def test_a_capture_on_another_device_defers_the_free_and_refuses_allocation(r200_pair) -> None:
    """In torch's default (global) capture mode a capture on cuda:0 forbids this thread the driver calls of a free or
    an allocation on cuda:1.  The last view of a cuda:1 buffer going during the capture leaves its range registered
    (and mapped), an allocation on cuda:1 raises, and the capture ends cleanly; the next allocation frees the range."""
    other = r200_pair
    fmt = "mxfp8_e8m0_g32"
    t = domain_empty((4, WIDTH), device=other)
    base = t.data_ptr()
    x = domain_empty((4, WIDTH), device=other)
    y = domain_empty((4, WIDTH), device=other)
    x.copy_(reference.randn_input((4, WIDTH), fmt, seed=73, device=x.device))
    _assert_same("eager on cuda:1", forward(x, format=fmt, out=y, locality="domain"), forward(x, format=fmt))
    torch.cuda.synchronize(other)
    with torch.cuda.device(0):
        scratch = torch.zeros(16, device="cuda:0")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream(0)):
            scratch.add_(1)
            del t
            gc.collect()
            assert any(entry[0] == base for entry in memory.registered(other))
            with pytest.raises(RuntimeError, match="capture"):
                domain_empty((4, WIDTH), device=other)
        assert any(entry[0] == base for entry in memory.registered(other))
        graph.replay()
        torch.cuda.synchronize(0)
        assert scratch.tolist() == [1.0] * 16
    domain_empty((4, WIDTH), device=other)
    assert all(entry[0] != base for entry in memory.registered(other))


@pytest.mark.parametrize("offset", OFFSETS, ids=("die0_inside_a_tile", "die1_whole_tile"))
@pytest.mark.parametrize("scale_out", (False, True), ids=("y", "y_scales"))
@pytest.mark.parametrize("fmt", FORMATS)
def test_domain_route_gives_the_off_bytes(big, fmt: str, scale_out: bool, offset: int, monkeypatch) -> None:
    """Out of place, just above the route's first size, ``x`` and ``out`` at the same offset in chunks of one die:
    the plan says ``domain``, the call launches the interleaved-domain program once, and its bytes (and scales) are
    the ``"off"`` route's on default memory; ``x`` is unchanged."""
    kind = loader.route_name(fmt, scale_out=scale_out)
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    y.fill_(float("nan"))
    got = plan(x, format=fmt, out=y, return_scales=scale_out)
    assert got["route"] == "domain" and got["reason"] == "placed", got
    assert got["program"] == _domain_name(kind)
    assert (got["first_tile"], got["first_die"]) == ((offset % CHUNK) // DOMAIN_TILE_BYTES[fmt], (offset // CHUNK) & 1)
    launched = _spy(monkeypatch)
    result = forward(x, format=fmt, out=y, return_scales=scale_out, locality="domain")
    assert launched == [_domain_name(kind)], launched
    monkeypatch.undo()
    want = forward(x0, format=fmt, return_scales=scale_out)
    y_got, s_got = result if scale_out else (result, None)
    y_want, s_want = want if scale_out else (want, None)
    assert y_got is y
    _assert_same(f"{kind} offset={offset} y", y_got, y_want)
    if scale_out:
        _assert_same(f"{kind} offset={offset} scales", s_got, s_want)
    _assert_same(f"{kind} offset={offset} x after the call", x, x0)


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_domain_program_runs_on_any_grid(r200, fmt: str) -> None:
    """Every interleaved-domain program, launched directly on grids of 1 and 3 CTAs and on the host's grid, gives
    the FP64 rule's bits: over memory that alternates dies (an input that crosses three chunk boundaries) and over
    default memory, whose placement only changes the speed.  The walk leaves its counters zero."""
    x0 = reference.randn_input((32769, WIDTH), fmt, seed=71)
    want = reference.qdq(x0, fmt)
    n_vecs = x0.numel() // loader.VECTOR
    offset = CHUNK - 3 * TILE_BYTES + 16
    buf_x = domain_empty((offset // BF16 + x0.numel(),))
    buf_y = domain_empty((offset // BF16 + x0.numel(),))
    placed = (
        buf_x[offset // BF16 :].view(x0.shape),
        buf_y[offset // BF16 :].view(x0.shape),
    )
    default = (torch.empty_like(x0), torch.empty_like(x0))
    for scale_out in (False, True):
        name = _domain_name(loader.route_name(fmt, scale_out=scale_out))
        record = loader.MODULES[name]
        words_a_die = record["counters"]["words"]
        for x, y in (placed, default):
            x_place = memory.place(x, CHUNK) or (0, 0)
            launch = loader.domain_launch(
                record, n_vecs=n_vecs, x_place=x_place, sm_count=loader.sm_count(0), arch=ARCH
            )
            for grid in ((1, 1, 1), (3, 1, 1), launch["grid"]):
                x.copy_(x0)
                y.fill_(float("nan"))
                words = torch.zeros(2 * words_a_die, dtype=torch.int32, device="cuda")
                values = {
                    "x": x,
                    "y": y,
                    "n_vecs": n_vecs,
                    "counters0": words.view(torch.uint32)[:words_a_die],
                    "counters1": words.view(torch.uint32)[words_a_die:],
                    **{key: launch[key] for key in ("first_tile", "first_die", "chunk_tile_shift", "batch_shift")},
                }
                s = None
                if scale_out:
                    s = torch.full((x0.shape[0], WIDTH // reference.GROUP[fmt]), float("nan"), device="cuda")
                    values["scales"] = s
                loader.run_module(name, values, device_index=0, grid=grid)
                what = f"{name} grid={grid} placed={x.data_ptr() == placed[0].data_ptr()}"
                _assert_same(f"{what} y", reference.y_bits(y), reference.y_bits(want[0]))
                if scale_out:
                    _assert_same(f"{what} scales", reference.scale_bits(s), reference.scale_bits(want[1]))
                assert int(words.count_nonzero()) == 0, (what, words.tolist())


# The inputs that define the rule's contract, as the plain programs' tests use them.
RULE_INPUTS = {
    "edge": lambda fmt: reference.edge_input(fmt),
    "sweep": lambda fmt: reference.sweep_input(fmt),
    "wide": lambda fmt: reference.randn_input((33, 8, 512), fmt, lo=120, hi=120, seed=31),
}
# Byte offsets of those inputs in memory that alternates dies: 3 tiles (of every format) before the end of a die-0
# chunk (a longer input reaches die 1), and inside the first tile of a die-1 chunk.
RULE_OFFSETS = (CHUNK - 3 * TILE_BYTES + 16, CHUNK + 16)


@pytest.mark.parametrize("inputs", tuple(RULE_INPUTS))
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_domain_program_gives_the_rule_on_its_contract_inputs(r200, fmt: str, inputs: str) -> None:
    """Every interleaved-domain program, launched directly on the host's grid over memory that alternates dies,
    gives the FP64 rule's bits on the inputs of the plain programs' rule tests: the edge rows (midpoints and ties,
    the floors, saturation), every finite BF16 value under every scale, and group scales from 2^-120 to 2^120.  The
    walk leaves its counters zero."""
    x0 = RULE_INPUTS[inputs](fmt)
    want = reference.qdq(x0, fmt)
    G = reference.GROUP[fmt]
    n_vecs = x0.numel() // loader.VECTOR
    buf_x = domain_empty((max(RULE_OFFSETS) // BF16 + x0.numel(),))
    buf_y = domain_empty((max(RULE_OFFSETS) // BF16 + x0.numel(),))
    for offset in RULE_OFFSETS:
        x = buf_x[offset // BF16 : offset // BF16 + x0.numel()].view(x0.shape)
        y = buf_y[offset // BF16 : offset // BF16 + x0.numel()].view(x0.shape)
        x_place = memory.place(x, CHUNK)
        assert x_place == (offset % CHUNK, (offset // CHUNK) & 1), x_place
        for scale_out in (False, True):
            name = _domain_name(loader.route_name(fmt, scale_out=scale_out))
            record = loader.MODULES[name]
            words_a_die = record["counters"]["words"]
            launch = loader.domain_launch(
                record, n_vecs=n_vecs, x_place=x_place, sm_count=loader.sm_count(0), arch=ARCH
            )
            x.copy_(x0)
            y.fill_(float("nan"))
            words = torch.zeros(2 * words_a_die, dtype=torch.int32, device="cuda")
            values = {
                "x": x,
                "y": y,
                "n_vecs": n_vecs,
                "counters0": words.view(torch.uint32)[:words_a_die],
                "counters1": words.view(torch.uint32)[words_a_die:],
                **{key: launch[key] for key in ("first_tile", "first_die", "chunk_tile_shift", "batch_shift")},
            }
            s = None
            if scale_out:
                s = torch.full((*x0.shape[:-1], x0.shape[-1] // G), float("nan"), device="cuda")
                values["scales"] = s
            loader.run_module(name, values, device_index=0, grid=launch["grid"])
            what = f"{name} {inputs} offset={offset}"
            _assert_same(f"{what} y", reference.y_bits(y), reference.y_bits(want[0]))
            if scale_out:
                _assert_same(f"{what} scales", reference.scale_bits(s), reference.scale_bits(want[1]))
            _assert_same(f"{what} x after the call", x, x0)
            assert int(words.count_nonzero()) == 0, (what, words.tolist())


def test_plain_reasons_on_r200(big, monkeypatch) -> None:
    """Each placement the route refuses runs the ``"off"`` program with the same bytes, for the reason the plan
    gives."""
    fmt = "mxfp8_e8m0_g32"
    offset = OFFSETS[0]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want = forward(x0, format=fmt)
    ordinary = torch.empty_like(x0)
    small = slice(0, 4097)
    cases = (
        ("x_memory", x0, y),
        ("out_memory", x, ordinary),
        ("no_out", x, None),
        ("chunk_offset", x, big.view(big.y_buf, offset + 16)),
        ("die_parity", x, big.view(big.y_buf, offset + CHUNK)),
        ("size", x[small], y[small]),
    )
    n_vecs = x0.numel() // loader.VECTOR
    for reason, source, out in cases:
        if out is not None:
            out.fill_(float("nan"))
        got = plan(source, format=fmt, out=out)
        size = source.numel() // loader.VECTOR
        program = loader.selected_module(ARCH, fmt, scale_out=False, n_vecs=size, in_place=False)
        assert (got["route"], got["reason"], got["program"]) == ("plain", reason, program), (reason, got)
        launched = _spy(monkeypatch)
        result = forward(source, format=fmt, out=out, locality="domain")
        assert launched == [program], (reason, launched)
        monkeypatch.undo()
        expected = want[small] if reason == "size" else want
        _assert_same(f"{reason} y", result, expected)
    # in place on memory that alternates dies
    got = plan(x, format=fmt, out=x)
    assert (got["route"], got["reason"]) == ("plain", "in_place"), got
    assert got["program"] == loader.selected_module(ARCH, fmt, scale_out=False, n_vecs=n_vecs, in_place=True)
    _assert_same("in place", forward(x, format=fmt, out=x, locality="domain"), want)


def test_counters_are_zero_after_calls(big) -> None:
    """Every launch leaves the claim counters of a stream zero: the host zeroes them once."""
    offset = OFFSETS[1]
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    for fmt in FORMATS:
        x.copy_(big.input(fmt))
        for scale_out in (False, True):
            for _ in range(3):
                forward(x, format=fmt, out=y, return_scales=scale_out, locality="domain")
    words = memory.stream_workspace(0, torch.cuda.current_stream().cuda_stream)
    assert words is not None
    torch.cuda.synchronize()
    assert int(words.count_nonzero()) == 0, words.tolist()


def test_a_raising_launch_drops_its_workspace(big, monkeypatch) -> None:
    """An eager launch that raises drops its stream's claim workspace, whose slot no later call takes; the next call
    on that stream takes another zeroed one and gives the ``"off"`` bytes."""
    fmt = "mxfp8_e8m0_g32"
    offset = OFFSETS[0]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want = forward(x0, format=fmt)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    def refuse(name, values, **kwargs):
        raise RuntimeError("launch refused")

    with torch.cuda.stream(side):
        forward(x, format=fmt, out=y, locality="domain")
        dropped = memory.stream_workspace(0, side.cuda_stream)
        assert dropped is not None
        monkeypatch.setattr(loader, "run_module", refuse)
        with pytest.raises(RuntimeError, match="launch refused"):
            forward(x, format=fmt, out=y, locality="domain")
        monkeypatch.undo()
        assert memory.stream_workspace(0, side.cuda_stream) is None
        y.fill_(float("nan"))
        forward(x, format=fmt, out=y, locality="domain")
        words = memory.stream_workspace(0, side.cuda_stream)
        assert words is not None and words.data_ptr() != dropped.data_ptr()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _assert_same("after a raising launch", y, want)
    assert int(words.count_nonzero()) == 0, words.tolist()


def test_caller_scales_on_the_domain_route(big, monkeypatch) -> None:
    """A caller's ``scales`` buffer (in default memory) on the domain route: the call launches the scales program
    and returns ``out`` and that buffer with the ``"off"`` bytes."""
    fmt = "mxfp4_e8m0_g32"
    offset = OFFSETS[1]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want_y, want_s = forward(x0, format=fmt, return_scales=True)
    s = torch.full_like(want_s, float("nan"))
    y.fill_(float("nan"))
    launched = _spy(monkeypatch)
    got_y, got_s = forward(x, format=fmt, out=y, scales=s, locality="domain")
    assert launched == [_domain_name(loader.route_name(fmt, scale_out=True))], launched
    monkeypatch.undo()
    assert got_y is y and got_s is s
    _assert_same("y", got_y, want_y)
    _assert_same("scales", got_s, want_s)


def test_two_streams_at_once(big) -> None:
    """Calls on two streams at once, each with its own claim workspace, give the ``"off"`` bytes."""
    fmt = "fp4_e4m3_g16"
    offset = OFFSETS[0]
    x0 = big.input(fmt)
    x = big.view(big.x_buf, offset)
    outs = (big.view(big.y_buf, offset), big.view(big.z_buf, offset))
    x.copy_(x0)
    want = forward(x0, format=fmt)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(3):
        for stream, out in zip(streams, outs, strict=True):
            with torch.cuda.stream(stream):
                out.fill_(float("nan"))
                forward(x, format=fmt, out=out, locality="domain")
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for k, out in enumerate(outs):
        _assert_same(f"stream {k}", out, want)
    words = [memory.stream_workspace(0, stream.cuda_stream) for stream in streams]
    assert words[0] is not None and words[1] is not None and words[0].data_ptr() != words[1].data_ptr()
    assert all(int(w.count_nonzero()) == 0 for w in words)


def test_claim_counters_lie_in_their_dies(big, monkeypatch) -> None:
    """Each die's claim counters lie in that die's memory: the rows of the device's counter chunks, and the counters
    the launch receives for a new stream's eager call (an eager slot) and for a captured call (a graph workspace the
    capture takes); a replay gives the eager bytes."""
    fmt = "mxfp8_e8m0_g32"
    offset = OFFSETS[1]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want = forward(x0, format=fmt)
    dies = list(range(memory.DIES))
    block = memory._GRAPH_WORKSPACES[0]["block"]
    assert block.shape == (memory.DIES, CHUNK // 4) and block.dtype == torch.int32
    for d in dies:
        for at in (block[d].data_ptr(), block[d].data_ptr() + CHUNK - 1):
            assert memory.domain_of(at) == d, (d, at)
    # die 0's row of the first eager slot: the graph workspaces lie below it
    eager = block[0, memory.GRAPH_WORKSPACES * memory.SLOT_WORDS].data_ptr()
    counters: list[tuple[int, int]] = []
    run_module = loader.run_module

    def spy(name, values, **kwargs):
        counters.append((values["counters0"].data_ptr(), values["counters1"].data_ptr()))
        return run_module(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", spy)
    monkeypatch.setattr(memory, "_STREAM_WORKSPACES", {})
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        forward(x, format=fmt, out=y, locality="domain")
    torch.cuda.current_stream().wait_stream(side)
    words = memory.stream_workspace(0, side.cuda_stream)
    assert words.shape == (memory.DIES, memory.SLOT_WORDS)
    assert counters == [tuple(words[d].data_ptr() for d in dies)], counters
    assert [memory.domain_of(at) for at in counters[0]] == dies
    assert counters[0][0] >= eager, (counters[0], eager)
    left = memory.graph_workspaces_left(0)
    assert left > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(x, format=fmt, out=y, locality="domain")
    assert memory.graph_workspaces_left(0) == left - 1
    assert len(counters) == 2, counters
    assert [memory.domain_of(at) for at in counters[1]] == dies
    assert counters[1][0] < eager, (counters[1], eager)
    y.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_same("replay", y, want)


def test_an_eager_call_with_no_slot_left_keeps_its_counters_in_default_memory(big, monkeypatch) -> None:
    """With every eager slot of the device taken, the first call on a stream makes a zeroed workspace outside the
    device's counter chunks, in default memory, and gives the ``"off"`` bytes."""
    fmt = "fp4_e4m3_g16"
    offset = OFFSETS[0]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want = forward(x0, format=fmt)
    held = memory._GRAPH_WORKSPACES[0]
    monkeypatch.setitem(held, "eager", held["slots"])
    monkeypatch.setattr(memory, "_STREAM_WORKSPACES", {})
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        y.fill_(float("nan"))
        forward(x, format=fmt, out=y, locality="domain")
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    words = memory.stream_workspace(0, side.cuda_stream)
    assert words.shape == (memory.DIES, memory.SLOT_WORDS)
    start = held["block"].data_ptr()
    for d in range(memory.DIES):
        assert not start <= words[d].data_ptr() < start + memory.DIES * CHUNK, d
    assert held["eager"] == held["slots"]
    _assert_same("no eager slot left", y, want)
    assert int(words.count_nonzero()) == 0, words.tolist()


@pytest.mark.parametrize("fmt", FORMATS)
def test_cuda_graph_replays_match_eager(big, fmt: str) -> None:
    """After one eager call per program, two captured calls on one capture stream take one zeroed graph workspace,
    and every replay gives the eager bytes."""
    offset = OFFSETS[1]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want_y, want_s = forward(x0, format=fmt, return_scales=True)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # one eager call per program the capture runs, so no library loads during the capture
        forward(x, format=fmt, out=y, locality="domain")
        forward(x, format=fmt, out=y, return_scales=True, locality="domain")
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    left = memory.graph_workspaces_left(0)
    assert left > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(x, format=fmt, out=y, locality="domain")
        _, s = forward(x, format=fmt, out=y, return_scales=True, locality="domain")
    assert memory.graph_workspaces_left(0) == left - 1
    for replay in range(3):
        y.fill_(float("nan"))
        s.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(f"{fmt} replay {replay} y", y, want_y)
        _assert_same(f"{fmt} replay {replay} scales", s, want_s)


def test_cuda_graph_without_graph_workspaces_left(big, monkeypatch) -> None:
    """With every graph workspace taken, each captured call takes a workspace from the graph's pool that the graph
    zeroes at every replay; no capture keeps it, and every replay gives the eager bytes."""
    fmt = "mxfp8_e8m0_g32"
    offset = OFFSETS[0]
    x0 = big.input(fmt)
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(x0)
    want_y, want_s = forward(x0, format=fmt, return_scales=True)
    forward(x, format=fmt, out=y, locality="domain")
    forward(x, format=fmt, out=y, return_scales=True, locality="domain")
    torch.cuda.synchronize()
    monkeypatch.setitem(memory._GRAPH_WORKSPACES[0], "taken", memory.GRAPH_WORKSPACES)
    held = dict(memory._CAPTURE_WORKSPACES)
    launched = _spy(monkeypatch)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward(x, format=fmt, out=y, locality="domain")
        _, s = forward(x, format=fmt, out=y, return_scales=True, locality="domain")
    assert launched == [_domain_name(fmt), _domain_name(fmt + "_scales")], launched
    assert memory.graph_workspaces_left(0) == 0
    assert memory._CAPTURE_WORKSPACES == held
    for replay in range(3):
        y.fill_(float("nan"))
        s.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(f"replay {replay} y", y, want_y)
        _assert_same(f"replay {replay} scales", s, want_s)


@pytest.mark.parametrize("fmt", FORMATS)
def test_steady_state_domain_calls_do_not_synchronize(big, fmt: str) -> None:
    """After the first call on a stream, a ``"domain"`` call and its plan run under the error sync-debug mode."""
    offset = OFFSETS[0]
    x, y = big.view(big.x_buf, offset), big.view(big.y_buf, offset)
    x.copy_(big.input(fmt))
    forward(x, format=fmt, out=y, locality="domain")
    forward(x, format=fmt, out=y, return_scales=True, locality="domain")
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert plan(x, format=fmt, out=y)["route"] == "domain"
        forward(x, format=fmt, out=y, locality="domain")
        forward(x, format=fmt, out=y, return_scales=True, locality="domain")
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
