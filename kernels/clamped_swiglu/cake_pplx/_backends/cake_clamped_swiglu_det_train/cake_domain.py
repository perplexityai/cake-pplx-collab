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

Memory whose chunks alternate between the two dies of an R200 (SM107), the
SM table and the claim workspaces of the locality-domain twin of the Cake
clamped SwiGLU training forward (the backward has none).

An R200 has two dies, each with its own SMs and its own memory (a locality
domain).  Default device memory alternates between the dies every 4 KB, so
about half of the bytes an SM reads come from the other die.
:func:`domain_empty` maps one virtual range of chunks, chunk ``k`` in die
``k & 1``, and returns a BF16 ``[R, 2304]`` tensor over chunks of 2 MB or a
merged ``[R, 4608]`` one over chunks of 4 MB, without a copy.  Either chunk
holds :data:`CHUNK_SLOTS` slots of a ninth of a row (512 or 1,024 bytes), so
row ``r`` of every such tensor of one row count starts at slot ``9 r`` of the
range: all of them change die at the same rows, and row 0 lies at slot 0 of
a die-0 chunk.  The range is registered by device, so :func:`place` finds a
tensor's first slot in its chunk and the die of that chunk from its address
alone, with no driver call.  The driver maps this memory, and the 4 MB of
claim counters below, outside torch's caching allocator, so
``torch.cuda.memory_allocated`` and the other ``torch.cuda.memory_*``
counters do not count it, and it cannot be shared with another process.

The tensor and every view of it keep the range alive.  Once the last one is
gone, its finalizer queues the range and frees it: it removes the range from
the registry, then synchronizes the device, unmaps the chunks and frees the
range, so an address torch reuses later is never taken for this memory.
When another call holds the module's lock (another thread allocating, or
this thread inside a garbage collection that runs during an allocation),
the finalizer only queues the range, and that call frees it as it releases
the lock; a finalizer never waits on the lock.  In torch's default (global)
capture mode a CUDA graph capture forbids its thread these driver calls on
every device, so during a capture on any device in this thread a range
waits for the next free outside a capture (another range's finalizer, the
next :func:`domain_empty` call or :func:`release`), and :func:`domain_empty`
refuses to allocate.  The check reads torch's current stream of every device
with an active primary context: a capture begun with
``cudaStreamBeginCapture`` on a stream that is not torch's current stream is
not detected.  Do not allocate such memory, or drop the last view of it, in
one thread while another thread captures a CUDA graph: the driver calls would
end that capture.

SM table (:func:`sm_table`): a twin reads its CTA's die from one int32 word
per SM id, passed by value.  The device's first :func:`domain_empty` call
probes it once for the process (the package's ``cake_locality`` probe, built
by the package's own JIT; outside a capture) and keeps it, or keeps None
when the probe fails, in which case every call runs the stage's other
programs.  The route never probes.

Claim workspaces (:func:`launch`): a twin launch takes its two dies' claim
counters, ``words`` unsigned words each (the record's ``counters``), from a
workspace that no launch in flight shares: two rows of :data:`SLOT_WORDS`
words, row ``d`` die ``d``'s counters; every launch leaves its counters zero.
The device's first :func:`domain_empty` call maps one 2 MB chunk in each die
for the device's claim counters (chunk ``d`` in die ``d``, 4 MB in all, kept
for the process), zeroes them and synchronizes once, so each die's counters
lie in that die's memory.  Each workspace of those chunks (a slot) is handed
out once: the first :data:`GRAPH_WORKSPACES` to CUDA graph captures, the rest
to eager calls, which need no fill.  Eager calls use one workspace per
(device, stream); on the per-thread default stream (``cudaStreamPerThread``,
one handle that names another stream in each thread) one per (device,
thread), kept for the process.  The twin launches of one stream share its
workspace: they run in order, and each leaves it zero.
The calls of one CUDA graph capture on one capture stream share one graph
workspace, taken for good.  An eager call with no slot left makes a
workspace in default memory and zeroes it, and once every graph workspace is
taken each captured call takes one from the graph's pool, which the graph
zeroes at every replay (die memory cannot be mapped during a capture);
either keeps both dies' counters in default memory, with the same bits.  The
replays of one instantiated graph never overlap, but a capture instantiated
more than once, or added to other graphs as a child graph, shares its
workspace between those copies: run them one at a time.  An eager launch that
raises drops its stream's workspace and its slot, so the next call takes a
zeroed one.  After the first call on a stream, a call reads no device memory
on the host and does not synchronize.

cuda-python is imported on first use of this module's driver calls, and the
names they need are checked then (:data:`DRIVER_NAMES`; the ``locality``
extra requires cuda-bindings :data:`CUDA_BINDINGS_FLOOR` or later).
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import ctypes
import functools
import itertools
import threading
import weakref
from collections.abc import Iterator, Sequence
from typing import Any, Optional, Union

import torch

from . import cake_jit

# The cuda-python (cuda-bindings) release the ``locality`` extra requires: the CUDA 13.5 driver API.
CUDA_BINDINGS_FLOOR = "13.5"
# The driver names this module needs beyond the virtual memory management API, as (enum, member).
DRIVER_NAMES = (
    ("CUmemLocationType", "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN"),
    ("CUdevice_attribute", "CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT"),
)
# The dies of an R200 (its locality domains): chunk ``k`` of a range lies in die ``k & 1``.
DIES = 2
# The slots of a chunk, ``2^CHUNK_SHIFT``: a row is ``cake_jit.DOMAIN_ROW_SLOTS`` (9) slots, so a chunk holds 455.1
# rows of every tensor this module allocates, whatever its width.
CHUNK_SHIFT = 12
CHUNK_SLOTS = 1 << CHUNK_SHIFT
# The row width of each tensor this module allocates, in BF16 elements, and the bytes of its chunk: a [R, 2304] row is
# 9 slots of 512 bytes (2 MB chunks), a merged [R, 4608] row 9 slots of 1,024 bytes (4 MB chunks).
CHUNK_BYTES = {2304: 2097152, 4608: 4194304}
# The bytes of one chunk of the claim counters (one in each die).
COUNTER_CHUNK_BYTES = 2097152
# The words of one die's row of a claim workspace (128 bytes, so no two workspaces share a line in either die's
# chunk); a launch uses the first ``words`` of each row.
SLOT_WORDS = 32
# The claim workspaces each device keeps for CUDA graph captures: the first GRAPH_WORKSPACES slots of its claim
# counter chunks (128 KB of each die's chunk), mapped by its first :func:`domain_empty` call; the other slots serve
# eager calls.
GRAPH_WORKSPACES = 1024
# The words of an SM table and their type, as a twin takes it (``sm_table``).
SM_TABLE_WORDS = 256


class DriverError(RuntimeError):
    """A driver call failed; ``status`` is its ``CUresult``."""

    def __init__(self, what: str, status: Any) -> None:
        super().__init__(f"{what} failed: {getattr(status, 'name', status)}")
        self.status = status


@functools.cache
def _driver() -> Any:
    """cuda-python's driver module, once its names are checked (``RuntimeError`` without them)."""

    try:
        from cuda.bindings import driver
    except ImportError as exc:
        raise RuntimeError(
            f"memory that alternates dies needs cuda-python (cuda-bindings {CUDA_BINDINGS_FLOOR} or later): install "
            "the 'locality' extra of cake-pplx-clamped-swiglu"
        ) from exc
    missing = [f"{enum}.{name}" for enum, name in DRIVER_NAMES if not hasattr(getattr(driver, enum, None), name)]
    if missing:
        raise RuntimeError(
            f"this cuda-python does not name {missing}; memory that alternates dies needs cuda-bindings "
            f"{CUDA_BINDINGS_FLOOR} or later"
        )
    return driver


def _check(result: Any, what: str) -> Any:
    """The values of one driver call's result tuple, or :class:`DriverError` naming its error."""

    driver = _driver()
    status, *values = result if isinstance(result, tuple) else (result,)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise DriverError(what, status)
    return values[0] if len(values) == 1 else tuple(values)


@contextlib.contextmanager
def _primary_context(driver: Any, index: int) -> Iterator[Any]:
    """The device's primary context (torch's), current on this thread inside the block."""

    _check(driver.cuInit(0), "cuInit")
    device = _check(driver.cuDeviceGet(index), "cuDeviceGet")
    context = _check(driver.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain")
    try:
        _check(driver.cuCtxPushCurrent(context), "cuCtxPushCurrent")
        try:
            yield device
        finally:
            driver.cuCtxPopCurrent()
    finally:
        driver.cuDevicePrimaryCtxRelease(device)


def _capturing() -> bool:
    """Whether this thread's current stream of any CUDA device is capturing a graph: in the global capture mode a
    capture on one device ends at a driver call of this thread that may synchronize any device.  A device whose
    primary context is inactive captures nothing and is skipped, so the check makes no context."""

    driver = _driver()
    _check(driver.cuInit(0), "cuInit")
    for index in range(torch.cuda.device_count()):
        device = _check(driver.cuDeviceGet(index), "cuDeviceGet")
        _, active = _check(driver.cuDevicePrimaryCtxGetState(device), "cuDevicePrimaryCtxGetState")
        if not active:
            continue
        with torch.cuda.device(index):
            if torch.cuda.is_current_stream_capturing():
                return True
    return False


def domain_count(index: int) -> int:
    """The locality domains the driver reports for CUDA device ``index``: 1 where it reports none."""

    driver = _driver()
    _check(driver.cuInit(0), "cuInit")
    device = _check(driver.cuDeviceGet(index), "cuDeviceGet")
    status, count = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device
    )
    return max(1, int(count)) if status == driver.CUresult.CUDA_SUCCESS else 1


def domain_of(address: int) -> Optional[int]:
    """The die the driver reports for the memory at ``address``, or None for memory in no single die."""

    driver = _driver()
    attribute = getattr(driver.CUpointer_attribute, "CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL", None)
    if attribute is None:
        return None
    status, ordinal = driver.cuPointerGetAttribute(attribute, driver.CUdeviceptr(address))
    if status != driver.CUresult.CUDA_SUCCESS or int(ordinal) < 0:
        return None
    return int(ordinal)


# --------------------------------------------------------------------------- the registry and the row frame

# Guards the registry, the frees, the SM tables and the graph workspaces.  Code that may run inside a garbage
# collection (a range's finalizer) only tries it, so a thread never waits on a lock it holds.
_LOCK = threading.Lock()
# device index -> (bases, ranges, frames): the registered ranges sorted by base, ``(base, size, chunk_bytes,
# start_die)``, their bases and their :func:`_frame` entries.  Replaced whole under the lock, so :func:`place` reads it
# without the lock.
_RANGES: dict[
    int,
    tuple[
        tuple[int, ...],
        tuple[tuple[int, int, int, int], ...],
        tuple[tuple[int, int, int, int, Optional[int], int, int], ...],
    ],
] = {}
# The registry of a device with no range, as :func:`ranges` returns it.
_NO_RANGES: tuple[tuple[()], tuple[()], tuple[()]] = ((), (), ())


def _frame(entry: tuple[int, int, int, int]) -> tuple[int, int, int, int, Optional[int], int, int]:
    """What :func:`slot_of` reads of the registered range ``entry``: ``(base, end, slot_bytes, row_bytes, die_bits,
    chunk_bytes, start_die)``, ``die_bits`` being ``start_die << CHUNK_SHIFT`` when a chunk is :data:`CHUNK_SLOTS`
    whole slots (then the place of a slot ``n`` slots from the base is ``(n + die_bits) % FRAME_PERIOD``) and None
    otherwise: such a range places no tensor (:data:`LAYOUT`, as :func:`chunk_place`)."""

    base, size, chunk, start_die = entry
    slot = chunk >> CHUNK_SHIFT
    die_bits = start_die << CHUNK_SHIFT if slot << CHUNK_SHIFT == chunk else None
    return base, base + size, slot, cake_jit.DOMAIN_ROW_SLOTS * slot, die_bits, chunk, start_die


def _set_ranges(index: int, entries: tuple[tuple[int, int, int, int], ...]) -> None:
    # under _LOCK
    _RANGES[index] = (tuple(entry[0] for entry in entries), entries, tuple(_frame(entry) for entry in entries))


def _register(index: int, base: int, size: int, chunk: int, start_die: int) -> None:
    # under _LOCK
    _set_ranges(index, tuple(sorted((*_RANGES.get(index, _NO_RANGES)[1], (base, size, chunk, start_die)))))


def _unregister(index: int, base: int) -> None:
    # under _LOCK
    _set_ranges(index, tuple(entry for entry in _RANGES.get(index, _NO_RANGES)[1] if entry[0] != base))


def registered(index: int) -> tuple[tuple[int, int, int, int], ...]:
    """The ranges of CUDA device ``index`` from :func:`domain_empty` that are not freed yet, as ``(base, size,
    chunk_bytes, start_die)`` by base (chunk ``k`` of a range lies in die ``start_die ^ (k & 1)``)."""

    return _RANGES.get(index, _NO_RANGES)[1]


# The results of :func:`place` for a tensor outside every registered range, and for one whose rows do not follow the
# range's slots.
MEMORY = "memory"
LAYOUT = "layout"


def chunk_place(
    address: int, rows: int, cols: int, stride: int, entry: tuple[int, int, int, int]
) -> Union[tuple[int, int], str]:
    """Where a 2-D BF16 tensor of ``rows >= 1`` rows of ``cols`` elements at ``address`` with row stride ``stride``
    (elements) lies in the registered range ``entry`` (``(base, size, chunk_bytes, start_die)``): ``(first_slot,
    first_die)`` (the slot of its first byte in its chunk, a slot being ``chunk_bytes / CHUNK_SLOTS`` bytes, and the
    die of that chunk); :data:`MEMORY` when its bytes do not all lie in the range; :data:`LAYOUT` when the range's
    chunk is not :data:`CHUNK_SLOTS` whole slots (no first slot below :data:`CHUNK_SLOTS` names every row start),
    its first byte is not on a slot or its row pitch (``2 * stride`` bytes, read only for more than one row) is not
    ``cake_jit.DOMAIN_ROW_SLOTS`` slots."""

    base, size, chunk, start_die = entry
    end = address + ((rows - 1) * stride + cols) * 2
    if address < base or end > base + size:
        return MEMORY
    slot = chunk >> CHUNK_SHIFT
    offset = address - base
    if slot << CHUNK_SHIFT != chunk or offset % slot or (rows > 1 and 2 * stride != cake_jit.DOMAIN_ROW_SLOTS * slot):
        return LAYOUT
    k, inner = divmod(offset, chunk)
    return inner // slot, start_die ^ (k & 1)


def place(t: torch.Tensor) -> Union[tuple[int, int], str]:
    """:func:`chunk_place` of the 2-D BF16 CUDA tensor ``t`` (at least one row) in the registered range of its device
    that holds its first byte, or :data:`MEMORY` when none does.  It reads only the registry: no driver call."""

    bases, entries, _ = _RANGES.get(t.get_device(), _NO_RANGES)
    if not bases:
        return MEMORY
    address = t.data_ptr()
    at = bisect.bisect_right(bases, address) - 1
    if at < 0:
        return MEMORY
    rows, cols = t.shape
    return chunk_place(address, rows, cols, t.stride(0), entries[at])


def in_frame(anchor: tuple[int, int], other: tuple[int, int]) -> bool:
    """Whether a tensor placed at ``other`` (:func:`place`) lies in the row frame of an anchor placed at ``anchor``:
    its row 0 starts at most 8 slots after the anchor's (a chunk later only in the other die), so its row ``r`` lies in
    the die of the anchor's row ``r`` but at most at one row of each chunk boundary."""

    period = 2 * CHUNK_SLOTS
    distance = (other[0] + (other[1] << CHUNK_SHIFT) - anchor[0] - (anchor[1] << CHUNK_SHIFT)) % period
    return distance < cake_jit.DOMAIN_ROW_SLOTS


# The period of the row frame (:func:`in_frame`), in slots: two chunks, one in each die.
FRAME_PERIOD = 2 * CHUNK_SLOTS
_bisect_right = bisect.bisect_right


def ranges(
    index: int,
) -> tuple[
    tuple[int, ...],
    tuple[tuple[int, int, int, int], ...],
    tuple[tuple[int, int, int, int, Optional[int], int, int], ...],
]:
    """``(bases, entries, frames)``: the registered ranges of CUDA device ``index`` (:func:`registered`), their bases
    and what :func:`slot_of` reads of each (:func:`_frame`), one read of the registry for the :func:`slot_of` calls of
    one launch."""

    return _RANGES.get(index, _NO_RANGES)


def slot_of(
    bases: tuple[int, ...],
    frames: tuple[tuple[int, int, int, int, Optional[int], int, int], ...],
    address: int,
    rows: int,
    cols: int,
) -> int:
    """:func:`place` of a contiguous 2-D BF16 tensor of ``rows >= 1`` rows of ``cols`` elements (row stride ``cols``) at
    ``address`` in the ranges ``bases``, ``frames`` of its device (:func:`ranges`), as one integer:
    ``first_slot + (first_die << CHUNK_SHIFT)`` when it is placed, -1 for :data:`MEMORY` and -2 for :data:`LAYOUT`.  The
    rules of :func:`chunk_place` in the range that holds the first byte, the tensor's bytes being ``2 * rows * cols``
    from ``address`` (a range whose chunk is not whole slots gives -2).  Two placed tensors ``anchor`` and ``other`` are
    in one row frame (:func:`in_frame`) when ``(other - anchor) % FRAME_PERIOD < cake_jit.DOMAIN_ROW_SLOTS``."""

    at = _bisect_right(bases, address) - 1
    if at < 0:
        return -1
    base, end, slot, row_bytes, die_bits, _, _ = frames[at]
    if address + 2 * rows * cols > end:
        return -1
    offset = address - base
    if die_bits is None or offset % slot or (rows > 1 and 2 * cols != row_bytes):
        return -2
    # chunk k holds slots k * CHUNK_SLOTS on, in die start_die ^ (k & 1): one count of slots modulo two chunks
    return (offset // slot + die_bits) % FRAME_PERIOD


# --------------------------------------------------------------------------- a torch tensor over the range (DLPack)

_DLPACK_CUDA = 2
_DLPACK_UINT = 1
_DLTENSOR = b"dltensor"


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", _DELETER)]


# This module's own bindings of the capsule API (setting ``argtypes`` on ``ctypes.pythonapi``'s shared function
# objects would change them for every other user in the process).
_capsule_new = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)(
    ("PyCapsule_New", ctypes.pythonapi)
)
_capsule_is_valid = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_capsule_pointer = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)
_incref = ctypes.PYFUNCTYPE(None, ctypes.py_object)(("Py_IncRef", ctypes.pythonapi))
# managed-tensor address -> what the exported tensor keeps alive: the struct, its shape and the range's owner
_EXPORTED: dict[int, tuple[Any, ...]] = {}


@_DELETER
def _delete_managed(address: int, exported: dict[int, tuple[Any, ...]] = _EXPORTED) -> None:
    # The consumer's tensor is gone: release the struct and the owner it kept alive.
    exported.pop(address, None)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _destroy_capsule(
    capsule: int,
    exported: dict[int, tuple[Any, ...]] = _EXPORTED,
    valid: Any = _capsule_is_valid,
    pointer: Any = _capsule_pointer,
    name: bytes = _DLTENSOR,
) -> None:
    # A capsule that no consumer took (a consumer renames it): release what it kept alive.
    if valid(capsule, name):
        exported.pop(pointer(capsule, name), None)


# A tensor freed while the interpreter tears this module down still calls the deleter, which reads its struct: one
# reference that is never released keeps the callbacks, the structs and the functions they call for the process.
_incref((_EXPORTED, _delete_managed, _destroy_capsule, _capsule_is_valid, _capsule_pointer))


class _Export:
    """A DLPack producer of the ``size`` bytes from ``base`` on CUDA device ``index`` as one uint8 row, whose
    tensors keep ``owner`` alive.  The memory is ready when the view is made, so no stream is waited on."""

    def __init__(self, base: int, size: int, index: int, owner: object) -> None:
        self.base, self.size, self.index, self.owner = base, size, index, owner

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_DLPACK_CUDA, self.index)

    def __dlpack__(
        self,
        *,
        stream: object = None,
        max_version: Optional[tuple[int, int]] = None,
        dl_device: Optional[tuple[int, int]] = None,
        copy: Optional[bool] = None,
    ) -> object:
        if copy:
            raise BufferError("memory that alternates dies is exported without a copy")
        if dl_device is not None and tuple(dl_device) != self.__dlpack_device__():
            raise BufferError(f"memory on cuda:{self.index} cannot be exported to DLPack device {dl_device}")
        managed = _DLManagedTensor()
        shape = (ctypes.c_int64 * 1)(self.size)
        tensor = managed.dl_tensor
        tensor.data = self.base
        tensor.device = _DLDevice(_DLPACK_CUDA, self.index)
        tensor.ndim = 1
        tensor.dtype = _DLDataType(_DLPACK_UINT, 8, 1)
        tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
        tensor.strides = None
        tensor.byte_offset = 0
        managed.deleter = _delete_managed
        address = ctypes.addressof(managed)
        _EXPORTED[address] = (managed, shape, self.owner)
        return _capsule_new(address, _DLTENSOR, ctypes.cast(_destroy_capsule, ctypes.c_void_p))


# --------------------------------------------------------------------------- the range: map, own, free


class _Range:
    """The owner of one mapped range: every tensor made from it keeps it alive, and its finalizer queues the range
    to be freed."""

    __slots__ = ("__weakref__",)


class _Mapped:
    """One mapped range, from its owner's death until it is freed.  ``unmapped`` counts the chunks unmapped so far,
    so a free that failed part way resumes after them."""

    __slots__ = ("base", "chunk", "index", "size", "unmapped")

    def __init__(self, index: int, base: int, size: int, chunk: int) -> None:
        self.index, self.base, self.size, self.chunk, self.unmapped = index, base, size, chunk, 0


def _map(driver: Any, index: int, chunk: int, chunks: int) -> tuple[int, int]:
    """Reserve ``chunks * chunk`` bytes of address space on device ``index``, aligned to a chunk, back chunk ``k`` with
    memory of die ``k & 1`` and give the device read-write access: ``(base, size)``."""

    size = chunk * chunks
    with _primary_context(driver, index):
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
        prop.location.id = index
        granule = int(
            _check(
                driver.cuMemGetAllocationGranularity(
                    prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        if granule < 1 or chunk % granule:
            raise RuntimeError(
                f"device {index} allocates locality-domain memory in granules of {granule} bytes, which do not divide "
                f"the {chunk}-byte chunk the locality-domain twins assume"
            )
        base = int(_check(driver.cuMemAddressReserve(size, chunk, 0, 0), "cuMemAddressReserve"))
        mapped = 0
        try:
            for k in range(chunks):
                # cuda.h packs {device, locality domain} into the location id.
                prop.location.id = index | (k & 1) << 8
                handle = _check(driver.cuMemCreate(chunk, prop, 0), f"cuMemCreate (chunk {k}, die {k & 1})")
                try:
                    _check(driver.cuMemMap(driver.CUdeviceptr(base + k * chunk), chunk, 0, handle, 0), "cuMemMap")
                finally:
                    # The mapping keeps the memory; unmapping frees it.
                    driver.cuMemRelease(handle)
                mapped += 1
            access = driver.CUmemAccessDesc()
            access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = index
            access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check(driver.cuMemSetAccess(driver.CUdeviceptr(base), size, [access], 1), "cuMemSetAccess")
        except BaseException:
            for k in range(mapped):
                driver.cuMemUnmap(driver.CUdeviceptr(base + k * chunk), chunk)
            driver.cuMemAddressFree(driver.CUdeviceptr(base), size)
            raise
    return base, size


def _map_or_out_of_memory(driver: Any, index: int, chunk: int, chunks: int) -> tuple[int, int]:
    """:func:`_map`; when the driver is out of memory, return torch's cached free blocks to it once and map again,
    then ``torch.cuda.OutOfMemoryError``."""

    out_of_memory = driver.CUresult.CUDA_ERROR_OUT_OF_MEMORY
    try:
        return _map(driver, index, chunk, chunks)
    except DriverError as exc:
        if exc.status != out_of_memory:
            raise
    torch.cuda.empty_cache()
    try:
        return _map(driver, index, chunk, chunks)
    except DriverError as exc:
        if exc.status != out_of_memory:
            raise
        raise torch.cuda.OutOfMemoryError(
            f"memory that alternates dies: device {index} has no room for {chunks} chunks of {chunk} bytes ({exc})"
        ) from exc


def _free(entry: _Mapped) -> None:
    """Wait for the device's work, then unmap the range's chunks not unmapped yet and free its address space."""

    driver = _driver()
    with _primary_context(driver, entry.index):
        _check(driver.cuCtxSynchronize(), "cuCtxSynchronize")
        while entry.unmapped * entry.chunk < entry.size:
            at = driver.CUdeviceptr(entry.base + entry.unmapped * entry.chunk)
            _check(driver.cuMemUnmap(at, entry.chunk), "cuMemUnmap")
            entry.unmapped += 1
        _check(driver.cuMemAddressFree(driver.CUdeviceptr(entry.base), entry.size), "cuMemAddressFree")


# Ranges whose owner is gone and that are not freed yet.  A finalizer only appends (no lock).
_PENDING: collections.deque[_Mapped] = collections.deque()


def _free_pending(only: Optional[int] = None) -> int:
    """Under ``_LOCK``: remove from the registry and free every pending range (of device ``only``, or of every
    device), keeping them all for a later call while this thread captures a CUDA graph on any device, and keeping those
    whose free fails.  The registry entry goes first, so no later call takes an address of the range, which torch may
    reuse once it is freed, for this memory.  Returns how many it freed."""

    if not _PENDING:
        return 0
    try:
        capturing = _capturing()
    except Exception:  # noqa: BLE001 - a finalizer cannot raise; a later call checks again
        capturing = True
    if capturing:
        return 0
    kept, freed = [], 0
    while _PENDING:
        entry = _PENDING.popleft()
        if only is not None and entry.index != only:
            kept.append(entry)
            continue
        _unregister(entry.index, entry.base)
        try:
            _free(entry)
        except Exception:  # noqa: BLE001 - a finalizer cannot raise; a later call retries where this one stopped
            kept.append(entry)
            continue
        freed += 1
    _PENDING.extend(kept)
    return freed


def _drain() -> None:
    """Free the pending ranges unless another call holds ``_LOCK``; every call that holds it runs this after it
    releases it, so a range a finalizer queued meanwhile (it found the lock taken) is not left pending.  Stops when
    a pass finds ranges and frees none: during a capture, or when every free fails, they wait for a later call."""

    while _PENDING and _LOCK.acquire(blocking=False):
        try:
            had = bool(_PENDING)
            freed = _free_pending()
        finally:
            _LOCK.release()
        if had and not freed:
            return


def _release(entry: _Mapped) -> None:
    """The finalizer of a range's owner: queue the range, and free the queue unless another call holds ``_LOCK``
    (another thread, or this one inside a garbage collection); that call frees it as it releases the lock
    (:func:`_drain`).  It never waits on the lock."""

    _PENDING.append(entry)
    _drain()


def pending(index: Optional[int] = None) -> int:
    """The ranges queued to be freed (of device ``index``, or of every device)."""

    return sum(1 for entry in list(_PENDING) if index is None or entry.index == index)


def release(index: int) -> int:
    """Free the ranges of CUDA device ``index`` whose last view is gone and that wait in the queue (outside a CUDA
    graph capture: ``RuntimeError`` during a capture on any device in this thread, when the queue holds a range of the
    device), then, as every call that takes ``_LOCK``, the rest of the queue it can free (:func:`_drain`).  Returns how
    many ranges of the device it freed."""

    if not any(entry.index == index for entry in list(_PENDING)):
        return 0
    if _capturing():
        raise RuntimeError(
            "memory that alternates dies cannot be freed during a CUDA graph capture on any device of this thread: "
            "release it after the capture"
        )
    try:
        with _LOCK:
            return _free_pending(index)
    finally:
        _drain()


def _cuda_index(device: Union[torch.device, str, int, None]) -> int:
    if device is None:
        return torch.cuda.current_device()
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {device}")
    return device.index if device.index is not None else torch.cuda.current_device()


def _arch(index: int) -> str:
    """The architecture of CUDA device ``index`` (the backend package's, read once per device)."""

    from . import device_arch  # noqa: PLC0415 - the backend package, imported before this module

    return device_arch(index)


def _has_twins(arch: str) -> bool:
    """Whether ``arch`` has a locality-domain twin of some stage."""

    return any(cake_jit.PROGRAM_OF[stage + cake_jit.DOMAIN].get(arch) for stage in cake_jit.STAGES)


def domain_empty(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[torch.device, str, int, None] = None,
) -> torch.Tensor:
    """An uninitialized contiguous BF16 tensor of ``shape`` ``(R, 2304)`` (2 MB chunks) or ``(R, 4608)`` (4 MB
    chunks) in one range of chunks that alternate dies (chunk ``k`` in die ``k & 1``; row 0 at slot 0 of a die-0
    chunk), on an SM107 device that reports two locality domains (see the module docstring).

    ``TypeError`` for a ``dtype`` that is not a ``torch.dtype``; ``ValueError`` for another dtype, a shape that is not
    ``(R >= 0, 2304)`` or ``(R, 4608)``, or a device that is not CUDA; ``torch.cuda.OutOfMemoryError`` when the device
    has no room, after torch's cached free memory is returned to the driver once; ``RuntimeError`` on another
    architecture, during a CUDA graph capture on any device in this thread, with fewer than two locality domains,
    without cuda-python's locality-domain names, or when the driver refuses the memory.  The device's first call also
    probes its SM table (:func:`sm_table`), maps its claim counters (one 2 MB chunk in each die, 4 MB in all, kept for
    the process), zeroes them and synchronizes the current stream once; it raises ``torch.cuda.OutOfMemoryError``
    when the device has no room for them.  Each call first frees the ranges whose owners are gone."""

    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
    if dtype != torch.bfloat16:
        raise ValueError(f"memory that alternates dies holds BF16 operands of the twins only, got dtype {dtype}")
    try:
        dims = tuple(int(d) for d in shape)
    except TypeError:
        raise ValueError(f"shape must be (R, 2304) or (R, 4608), got {shape!r}") from None
    if len(dims) != 2 or dims[0] < 0 or dims[1] not in CHUNK_BYTES:
        raise ValueError(f"shape must be (R, 2304) or (R, 4608) with R >= 0, got {dims}")
    index = _cuda_index(device)
    arch = _arch(index)
    if not _has_twins(arch):
        raise RuntimeError(
            f"memory that alternates dies needs an SM107 (R200) device; device {index} is {arch}, which has no "
            "locality-domain twin"
        )
    driver = _driver()
    if _capturing():
        raise RuntimeError(
            "memory that alternates dies cannot be allocated during a CUDA graph capture on any device of this "
            "thread: allocate it before the capture"
        )
    domains = domain_count(index)
    if domains < DIES:
        raise RuntimeError(
            f"device {index} reports {domains} locality domain(s); memory that alternates dies needs {DIES} "
            "(an R200 with a driver that reports its dies)"
        )
    _probe_once(index, arch)
    chunk = CHUNK_BYTES[dims[1]]
    nbytes = dims[0] * dims[1] * 2
    chunks = max(1, -(-nbytes // chunk))
    owner = _Range()
    try:
        with _LOCK:
            _free_pending()
            _make_graph_workspaces(index)
            base, size = _map_or_out_of_memory(driver, index, chunk, chunks)
            finalizer = weakref.finalize(owner, _release, _Mapped(index, base, size, chunk))
            finalizer.atexit = False
            _register(index, base, size, chunk, 0)
    finally:
        _drain()
    whole = torch.from_dlpack(_Export(base, size, index, owner))
    del owner
    return whole[:nbytes].view(torch.bfloat16).view(dims)


# --------------------------------------------------------------------------- the SM table

# device index -> its SM table (a contiguous CPU int32 tensor of SM_TABLE_WORDS words), or None when its probe failed.
# A device not in the table has not been probed: no :func:`domain_empty` call ran on it yet.
_TABLES: dict[int, Optional[torch.Tensor]] = {}


def sm_table(index: int) -> Optional[torch.Tensor]:
    """The SM table of CUDA device ``index`` a twin takes as ``sm_table``, or None (not probed yet, or the probe
    failed): one int32 word per SM id, the SM's die among the fields of the package's ``cake_locality`` table.  It
    never probes."""

    return _TABLES.get(index)


def _probe_table(index: int, arch: str) -> Optional[torch.Tensor]:
    """The SM table of device ``index`` from the package's probe (``cake_locality.probe_table``: one CTA on every SM
    reads its SM id and die, taken again until it covers every SM, at most a few times), as a contiguous CPU int32
    tensor of :data:`SM_TABLE_WORDS` words, or None when the probe gives no table."""

    from . import cake_locality  # noqa: PLC0415 - the probe and its JIT build, needed by the first allocation only

    table = cake_locality.probe_table(index, arch)
    if table is None:
        return None
    entries = table.entries
    if (
        not isinstance(entries, torch.Tensor)
        or entries.device.type != "cpu"
        or entries.dtype != torch.int32
        or tuple(entries.shape) != (SM_TABLE_WORDS,)
        or not entries.is_contiguous()
    ):
        return None
    return entries


def _probe_once(index: int, arch: str) -> None:
    """Probe the SM table of device ``index`` unless it was probed: outside a capture (the caller checked), once per
    process and device; a failed probe is kept as None."""

    if index in _TABLES:
        return
    table = _probe_table(index, arch)
    with _LOCK:
        _TABLES.setdefault(index, table)


# --------------------------------------------------------------------------- claim workspaces


class _Workspace:
    """One claim workspace (int32 words ``[DIES, SLOT_WORDS]``, row ``d`` die ``d``'s counter words: in die ``d``'s
    memory, or both rows in default memory), its unsigned counter views per count, one row each, and those views as
    the FFI entry takes them (:meth:`arguments`)."""

    __slots__ = ("_arguments", "_views", "words")

    def __init__(self, words: torch.Tensor) -> None:
        self.words = words
        self._views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._arguments: dict[int, tuple[Any, Any]] = {}

    def counters(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        views = self._views.get(count)
        if views is None:
            unsigned = self.words.view(torch.uint32)
            views = self._views[count] = (unsigned[0, :count], unsigned[1, :count])
        return views

    def arguments(self, count: int) -> tuple[Any, Any]:
        """:meth:`counters` as ``tvm_ffi`` tensors over the same memory, made once per count (the entry takes a torch
        tensor through DLPack at every call, a ``tvm_ffi`` tensor as it is); each holds its view, so the workspace's
        memory lives as long as they do."""
        arguments = self._arguments.get(count)
        if arguments is None:
            import tvm_ffi  # noqa: PLC0415 - loaded with the package's libraries already

            arguments = self._arguments[count] = tuple(tvm_ffi.from_dlpack(view) for view in self.counters(count))
        return arguments


# device index -> {"block": int32 [DIES, chunk / 4] over its two claim counter chunks (row d in die d), "taken": the
# graph slots taken, "eager": its next eager slot, "slots": the slots of a row}
_GRAPH_WORKSPACES: dict[int, dict[str, Any]] = {}
# Guards each device's next eager slot; a call holds it only to take one.
_SLOTS = threading.Lock()
# _stream_key(device index, stream handle) -> the workspace of the eager calls on that stream
_STREAM_WORKSPACES: dict[tuple[int, ...], _Workspace] = {}
# _stream_key(device index, capture stream handle) -> (capture id, the graph workspace of that capture's calls on
# that stream)
_CAPTURE_WORKSPACES: dict[tuple[int, ...], tuple[int, _Workspace]] = {}
# cudaStreamPerThread: one handle that names the calling thread's own default stream
PER_THREAD_STREAM = 0x2
# Each thread's token for the per-thread default stream: drawn once per thread, never reused (a thread's ident may be
# reused after it exits, while its last launch may still run).
_THREAD = threading.local()
_THREAD_TOKENS = itertools.count(1)
# torch's getters of the raw current-stream handle of a device, of the current device and of whether the current
# device's current stream captures (each without building a ``torch.cuda.Stream`` or a context manager); ``None`` on a
# torch build without them.
_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_GET_DEVICE = getattr(torch._C, "_cuda_getDevice", None)
_IS_CAPTURING = getattr(torch._C, "_cuda_isCurrentStreamCapturing", None)


def _current_stream_handle(device_index: int) -> int:
    """The handle of the device's current CUDA stream: the value of ``torch.cuda.current_stream(device_index)
    .cuda_stream``, read without building a ``torch.cuda.Stream`` object when torch has the raw getter."""

    if _CURRENT_RAW_STREAM is not None:
        return int(_CURRENT_RAW_STREAM(device_index))
    return int(torch.cuda.current_stream(device_index).cuda_stream)


def _stream_capturing(device_index: int) -> bool:
    """Whether the device's current stream is capturing a CUDA graph."""

    if _GET_DEVICE is not None and _IS_CAPTURING is not None and _GET_DEVICE() == device_index:
        return bool(_IS_CAPTURING())
    with torch.cuda.device(device_index):
        return torch.cuda.is_current_stream_capturing()


def _stream_key(index: int, stream: int) -> tuple[int, ...]:
    """The key of the workspaces of the calls on ``stream`` of device ``index``: the per-thread default stream's
    handle names another stream in each thread, so its key also holds the thread's token.  A thread's workspace is
    kept for the process (a launch that raises drops it) and is never handed to another thread."""

    if stream == PER_THREAD_STREAM:
        token = getattr(_THREAD, "token", None)
        if token is None:
            token = _THREAD.token = next(_THREAD_TOKENS)
        return (index, stream, token)
    return (index, stream)


def _make_graph_workspaces(index: int) -> None:
    """Under ``_LOCK``, outside any capture, on a device with :data:`DIES` locality domains: map one chunk in each die
    for the device's claim counters (chunk ``d`` in die ``d``), zero them once and wait for the fill, so a capture on
    any stream may take a graph workspace and an eager call a slot with no fill.  The range is not registered and
    lives for the process.  When a step after the mapping raises, the range is freed (a free that fails too leaves it
    mapped) and nothing is recorded, so the next call maps it again."""

    if index in _GRAPH_WORKSPACES:
        return
    chunk = COUNTER_CHUNK_BYTES
    base, size = _map_or_out_of_memory(_driver(), index, chunk, DIES)
    try:
        block = torch.from_dlpack(_Export(base, size, index, _Range())).view(torch.int32).view(DIES, chunk // 4)
        block.zero_()
        torch.cuda.current_stream(torch.device("cuda", index)).synchronize()
    except BaseException:
        with contextlib.suppress(Exception):  # the first error is the one to report
            _free(_Mapped(index, base, size, chunk))
        raise
    # Published last: a reader without the lock finds no entry or a zeroed block.
    _GRAPH_WORKSPACES[index] = {
        "block": block,
        "taken": 0,
        "eager": GRAPH_WORKSPACES,
        "slots": chunk // (4 * SLOT_WORDS),
    }


def _eager_workspace(index: int) -> _Workspace:
    """A zeroed workspace no launch has used, for the eager calls on one stream of CUDA device ``index``: the device's
    next eager slot, else (before its first :func:`domain_empty` call, or once every slot is taken) one made in
    default memory and zeroed, with both dies' counters there."""

    held = _GRAPH_WORKSPACES.get(index)
    k = None
    if held is not None:
        with _SLOTS:
            if held["eager"] < held["slots"]:
                k = held["eager"]
                held["eager"] = k + 1
    if k is None:
        return _Workspace(torch.zeros((DIES, SLOT_WORDS), dtype=torch.int32, device=torch.device("cuda", index)))
    return _Workspace(held["block"][:, k * SLOT_WORDS : (k + 1) * SLOT_WORDS])


def graph_workspaces_left(index: int) -> int:
    """The graph workspaces of CUDA device ``index`` no capture has taken (0 before its first :func:`domain_empty`
    call)."""

    held = _GRAPH_WORKSPACES.get(index)
    return 0 if held is None else GRAPH_WORKSPACES - held["taken"]


def _capture_id(stream: int) -> Optional[int]:
    """The id of the capture in progress on ``stream``, or None when the driver does not say."""

    driver = _driver()
    result = driver.cuStreamGetCaptureInfo(driver.CUstream(stream))
    if (
        result[0] != driver.CUresult.CUDA_SUCCESS
        or result[1] != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE
    ):
        return None
    return int(result[2])


def _capture_workspace(key: tuple[int, ...]) -> _Workspace:
    """The workspace of a call during a capture on the stream of ``key``: the one this capture's calls on that
    stream already hold, else a graph workspace taken for good (without a capture id, one per call), else, with none
    left, one from the graph's pool that the graph zeroes at every replay, in default memory (die memory cannot be
    mapped during a capture).  Only a graph workspace is kept across calls: a pool workspace is freed when its call
    returns, as the graph's own temporary, so no later capture can reuse its bytes while this graph still writes
    them."""

    capture = _capture_id(key[1])
    if capture is not None:
        held = _CAPTURE_WORKSPACES.get(key)
        if held is not None and held[0] == capture:
            return held[1]
    words = None
    with _LOCK:
        workspaces = _GRAPH_WORKSPACES.get(key[0])
        if workspaces is not None and workspaces["taken"] < GRAPH_WORKSPACES:
            k = workspaces["taken"]
            workspaces["taken"] = k + 1
            words = workspaces["block"][:, k * SLOT_WORDS : (k + 1) * SLOT_WORDS]
    _drain()
    if words is None:
        return _Workspace(torch.zeros((DIES, SLOT_WORDS), dtype=torch.int32, device=torch.device("cuda", key[0])))
    workspace = _Workspace(words)
    if capture is not None:
        _CAPTURE_WORKSPACES[key] = (capture, workspace)
    return workspace


def stream_workspace(index: int, stream: int) -> Optional[torch.Tensor]:
    """The words of the eager calls' workspace on ``stream`` of device ``index`` (of this thread, for the per-thread
    default stream): int32 ``[DIES, SLOT_WORDS]``, row ``d`` die ``d``'s counter words; or None before its first
    call."""

    workspace = _STREAM_WORKSPACES.get(_stream_key(index, stream))
    return None if workspace is None else workspace.words


def claim(index: int, words: int) -> tuple[Any, Any, Optional[tuple[int, ...]]]:
    """``(counters0, counters1, key)``: the claim counters (``words`` words a die) a twin launch on the current stream
    of device ``index`` takes from a workspace no launch in flight shares (see the module docstring), and the key of
    that workspace when it is an eager one (None during a capture).  A launch that raises passes the key to
    :func:`drop`."""

    stream = _current_stream_handle(index)
    # :func:`_stream_key`, inline for every stream but the per-thread default one
    key = (index, stream) if stream != PER_THREAD_STREAM else _stream_key(index, stream)
    if _stream_capturing(index):
        counters0, counters1 = _capture_workspace(key).counters(words)
        return counters0, counters1, None
    workspace = _STREAM_WORKSPACES.get(key)
    if workspace is None:
        workspace = _STREAM_WORKSPACES[key] = _eager_workspace(index)
    arguments = workspace._arguments.get(words)
    if arguments is None:
        arguments = workspace.arguments(words)
    return arguments[0], arguments[1], key


def drop(key: Optional[tuple[int, ...]]) -> None:
    """After an eager launch that raised, drop its stream's workspace (``key`` from :func:`claim`) and its slot, so
    the next call takes a zeroed one; nothing for a capture's (``key`` None)."""

    if key is not None:
        _STREAM_WORKSPACES.pop(key, None)


def launch(entry: Any, words: int, index: int, head: tuple[Any, ...], tail: tuple[Any, ...]) -> None:
    """Run the twin's FFI ``entry`` on the current stream of device ``index`` with ``(*head, counters0, counters1,
    *tail)``: the claim counters of :func:`claim`.  An eager launch that raises drops its stream's workspace."""

    counters0, counters1, key = claim(index, words)
    try:
        entry(*head, counters0, counters1, *tail)
    except BaseException:
        drop(key)
        raise


__all__ = [
    "CHUNK_BYTES",
    "CHUNK_SHIFT",
    "CHUNK_SLOTS",
    "COUNTER_CHUNK_BYTES",
    "CUDA_BINDINGS_FLOOR",
    "DIES",
    "DRIVER_NAMES",
    "FRAME_PERIOD",
    "GRAPH_WORKSPACES",
    "LAYOUT",
    "MEMORY",
    "PER_THREAD_STREAM",
    "SLOT_WORDS",
    "SM_TABLE_WORDS",
    "DriverError",
    "chunk_place",
    "claim",
    "domain_count",
    "domain_empty",
    "domain_of",
    "drop",
    "graph_workspaces_left",
    "in_frame",
    "launch",
    "pending",
    "place",
    "ranges",
    "registered",
    "release",
    "slot_of",
    "sm_table",
    "stream_workspace",
]
