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

Locality-domain routing of one Cake package on a dual-die GPU (sm_107a).

A routed program has two records in the registry: its whole record in ``MODULES``
and its domain twin in ``DOMAIN_MODULES``. The twin is one launch over both dies.
It takes each die's shard of the per-head operands as separate parameters
(``<name>@0``, ``<name>@1`` in its ``logical`` map), each die's head count
(``heads@d``), each die's claim counters (``counters@d``) and, by value, the SM
table (``sm_table``): one int32 word per SM id, with the SM's rank in its die, the
die's SM count, the die and the supplemental flag (``ENTRY_*_SHIFT``).

``LOCALITY_ROUTES`` names, per whole record, its twin, the rule ``form`` and the
route ``cells``: ``[die class, shape class(, modes)]``. :func:`route` returns the
twin's record name for a call in a listed cell, else the whole record's name.

:func:`sm_table` probes the device's SMs once per process and device, outside CUDA
graph capture, through the package's own JIT build of :data:`PROBE_SOURCE`. Every
condition it cannot meet gives ``None``, and ``None`` routes every call to the whole
record (the fallback table of :func:`sm_table`). ``CAKE_PPLX_LOCALITY=0`` turns
routing off. :func:`probe_table` is the same probe and the same fallbacks with no
environment variable read, for a package whose own host routes its twins.

A routed launch is :func:`split_units` (the dies' shares of the Q/K heads, in
proportion to their SM counts), :func:`make_shards` (each die's operands), the
twin's launch over :func:`launch_grid`, then :func:`gather` (the per-die output
buffers copied back into the whole outputs), all on the current stream. The route
is a pure function of the die counts and the call's shape, so it is fixed for a
machine once the probe succeeds. Each twin launch claims every work item, so its
result does not depend on where the hardware places its CTAs.

Imports: the standard library at module level; ``torch``, ``tvm_ffi`` and the
package's ``cake_build`` / ``_cake_pplx_jit`` only inside the probe.
"""

from __future__ import annotations

import dataclasses
import fractions
import hashlib
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "ENTRY_BITS",
    "ENTRY_COUNT_SHIFT",
    "ENTRY_DOMAIN_BITS",
    "ENTRY_DOMAIN_SHIFT",
    "ENTRY_RANK_SHIFT",
    "ENTRY_SUPPLEMENTAL_SHIFT",
    "ENV",
    "FORMS",
    "LOCALITY_ARCH",
    "LOCALITY_CUDA",
    "PROBE_SOURCE",
    "SM_TABLE_LENGTH",
    "SmTable",
    "counter_spec",
    "die_class",
    "gather",
    "launch_grid",
    "make_shards",
    "pack_rows",
    "probe_table",
    "route",
    "shape_class",
    "sm_table",
    "split_units",
    "twin_loaded",
]

# The SM table a twin takes by value: one int32 word per SM id.
SM_TABLE_LENGTH = 256
# One word: the SM's rank among its die's SMs (by SM id) and the die's SM count,
# ENTRY_BITS wide each, the die ENTRY_DOMAIN_BITS wide, the supplemental flag.
ENTRY_RANK_SHIFT = 0
ENTRY_COUNT_SHIFT = 10
ENTRY_DOMAIN_SHIFT = 20
ENTRY_SUPPLEMENTAL_SHIFT = 24
ENTRY_BITS = 10
ENTRY_DOMAIN_BITS = 4
# The architecture whose twins route, and the CUDA release (toolkit and driver)
# whose PTX reads the locality registers.
LOCALITY_ARCH = "sm_107a"
LOCALITY_CUDA = (13, 5)
# "0" never routes; unset or any other value routes where the probe succeeds.
ENV = "CAKE_PPLX_LOCALITY"
# A probe that missed an SM (another context held it) is taken again, at most
# this many times in all.
PROBE_ATTEMPTS = 4
# The rule forms :func:`shape_class` knows: the shape arguments after the dies.
FORMS = ("heads", "heads_transfer")
# Two dies whose SM counts differ by at most this many SMs are "even".
EVEN_GAP = 4
# Token classes: at most SHORT_TOKENS is "short", at most MID_TOKENS "mid".
SHORT_TOKENS = 16384
MID_TOKENS = 65536
# Head classes: the first whose bound holds, else HEADS_TOP.
HEAD_CLASSES = (("hv1", 1), ("hv2-4", 4), ("hv5-16", 16), ("hv17-32", 32))
HEADS_TOP = "hv33+"

# One 1-thread CTA per SM (each asks for an SM's whole opt-in shared memory, so no
# two share an SM) stores its SM's id, locality domain, supplemental flag and the
# SM-id bound at its CTA's index.
PROBE_SOURCE = r"""
#include <cstdint>

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

__global__ void __launch_bounds__(1, 1) cake_locality_probe_kernel(uint4* out) {
  unsigned smid, domain, supplemental, nsmid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u32 %0, %%locality_domain_id;" : "=r"(domain));
  asm volatile("mov.u32 %0, %%locality_domain_supplemental_sm;" : "=r"(supplemental));
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
  out[blockIdx.x] = make_uint4(smid, domain, supplemental, nsmid);
}

void cake_locality_probe(tvm::ffi::TensorView out, int64_t ctas, int64_t smem) {
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) >= ctas && out.size(1) == 4);
  DLDevice device = out.device();
  TVM_FFI_ICHECK(cudaSetDevice(device.device_id) == cudaSuccess);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(cake_locality_probe_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)) == cudaSuccess);
  uint4* rows = reinterpret_cast<uint4*>(static_cast<char*>(out.data_ptr()) +
                                         out.byte_offset());
  cake_locality_probe_kernel<<<static_cast<unsigned>(ctas), 1,
                               static_cast<size_t>(smem), stream>>>(rows);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

int64_t cake_driver_version() {
  int version = 0;
  TVM_FFI_ICHECK(cudaDriverGetVersion(&version) == cudaSuccess);
  return version;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(probe, cake_locality_probe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(driver_version, cake_driver_version);
"""
_PROBE_DIGEST = hashlib.sha256(PROBE_SOURCE.encode("utf-8")).hexdigest()
# The JIT name of the probe library: a changed source never reuses a stale build.
PROBE_NAME = f"cake_locality_probe_{_PROBE_DIGEST[:12]}"


@dataclasses.dataclass(frozen=True)
class SmTable:
    """The probed SM table of one device: ``counts`` are the SMs of die 0 and die 1;
    ``entries`` is the CPU int32 tensor of :data:`SM_TABLE_LENGTH` words a twin takes
    as its ``sm_table`` (contiguous; never written)."""

    device_index: int
    counts: tuple[int, int]
    entries: Any


def die_class(counts: Sequence[int]) -> str:
    """``"even"`` when the two dies' SM counts differ by at most :data:`EVEN_GAP`,
    else ``"uneven"``."""
    return "even" if abs(int(counts[0]) - int(counts[1])) <= EVEN_GAP else "uneven"


def _documents(lengths: Sequence[int], heads: int) -> str:
    """``tok/docs/hv`` of documents of ``lengths`` tokens and ``heads`` heads."""
    lengths = [int(length) for length in lengths]
    t = sum(lengths)
    tok = "short" if t <= SHORT_TOKENS else "mid" if t <= MID_TOKENS else "long"
    if tok != "long":
        docs = "any"
    elif len(lengths) == 1:
        docs = "one"
    elif 5 * max(lengths) >= 3 * t:
        docs = "skew"
    else:
        docs = "multi"
    hv = next((c for c, top in HEAD_CLASSES if int(heads) <= top), HEADS_TOP)
    return f"{tok}/{docs}/{hv}"


def shape_class(form: str, *args: Any) -> tuple[Any, ...]:
    """The route key after the die class of a call of rule ``form``.

    ``"heads"`` takes ``(lengths, heads)``: the documents' token counts and the head
    count the rule reads; the key is ``(tok/docs/hv,)``. ``"heads_transfer"`` takes
    ``(lengths, heads, init_mode, use_u, write_transposed)``; the key adds the three
    mode flags as one tuple."""
    if form == "heads":
        lengths, heads = args
        return (_documents(lengths, heads),)
    if form == "heads_transfer":
        lengths, heads, init_mode, use_u, write_transposed = args
        modes = (int(init_mode), int(use_u), int(write_transposed))
        return (_documents(lengths, heads), modes)
    raise ValueError(f"unknown locality rule form {form!r}; known: {FORMS}")


def _listed(value: Any) -> Any:
    """``value`` with every tuple spelled as a list, as the route table spells it."""
    if isinstance(value, tuple | list):
        return [_listed(item) for item in value]
    return value


def route(
    routes: Mapping[str, Mapping[str, Any]],
    record: str,
    table: SmTable | None,
    *args: Any,
) -> str:
    """The record to launch for a call of whole record ``record``: its twin when the
    SM table is known and the call's ``[die class, *shape_class(...)]`` is one of the
    record's cells in ``routes`` (``LOCALITY_ROUTES``), else ``record``. ``args`` are
    the rule form's shape arguments (:func:`shape_class`)."""
    if table is None:
        return record
    row = routes.get(record)
    if row is None:
        return record
    cell = [die_class(table.counts), *_listed(shape_class(row["form"], *args))]
    return row["twin"] if cell in row["cells"] else record


def split_units(units: int, counts: Sequence[int]) -> tuple[int, int]:
    """``units`` whole units shared by the two dies in proportion to their SM counts:
    each die gets the floor of its exact share, and the units left over go one each to
    the larger remainders; a remainder tie goes to the die with more SMs, then to die
    0. 7 over (100, 112) is (3, 4), over (106, 106) (4, 3); 1 over (110, 102) is
    (1, 0), over (102, 110) (0, 1)."""
    weights = [int(count) for count in counts]
    if int(units) < 0 or len(weights) != 2 or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(f"{units} units are not shared by the SM counts {counts}")
    total = sum(weights)
    shares = [fractions.Fraction(int(units) * weight, total) for weight in weights]
    parts = [share.numerator // share.denominator for share in shares]
    order = sorted(range(2), key=lambda i: (parts[i] - shares[i], -weights[i], i))
    for index in order[: int(units) - sum(parts)]:
        parts[index] += 1
    return (parts[0], parts[1])


def pack_rows(
    rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], list[int]] | None:
    """``(counts, words)`` of the probe rows ``(smid, domain, supplemental, nsmid)``,
    one row per CTA: ``words`` holds :data:`SM_TABLE_LENGTH` entries, 0 at an SM id no
    row names. ``None`` when an SM id is not below :data:`SM_TABLE_LENGTH`, an
    ``nsmid`` exceeds it, two rows name one SM id, the domains are not exactly 0 and 1,
    or a die's count does not fit :data:`ENTRY_BITS`."""
    by_sm: dict[int, tuple[int, bool]] = {}
    for smid, domain, supplemental, nsmid in rows:
        if not 0 <= int(smid) < SM_TABLE_LENGTH or int(nsmid) > SM_TABLE_LENGTH:
            return None
        if int(smid) in by_sm:
            return None
        by_sm[int(smid)] = (int(domain), bool(int(supplemental)))
    if {domain for domain, _ in by_sm.values()} != {0, 1}:
        return None
    counts = [0, 0]
    ranks: dict[int, int] = {}
    for smid in sorted(by_sm):
        domain = by_sm[smid][0]
        ranks[smid] = counts[domain]
        counts[domain] += 1
    if max(counts) >= 1 << ENTRY_BITS:
        return None
    words = [0] * SM_TABLE_LENGTH
    for smid, (domain, supplemental) in by_sm.items():
        words[smid] = (
            ranks[smid] << ENTRY_RANK_SHIFT
            | counts[domain] << ENTRY_COUNT_SHIFT
            | domain << ENTRY_DOMAIN_SHIFT
            | int(supplemental) << ENTRY_SUPPLEMENTAL_SHIFT
        )
    return (counts[0], counts[1]), words


# --------------------------------------------------------------------------- probe

_LOCK = threading.Lock()
_TABLES: dict[tuple[int, str], SmTable | None] = {}
_PROBED: dict[tuple[int, str], SmTable | None] = {}  # probe_table's: no switch
_PROBE_MODULE: list[Any] = []  # the loaded probe library, or None, once tried
_FAILED_TWINS: set[str] = set()
_LOADED_TWINS: set[str] = set()


def sm_table(device_index: int, arch: str) -> SmTable | None:
    """The SM table of device ``device_index``, probed once per process and device, or
    ``None``; a caller routes nothing on ``None``.

    ======================================================  =================
    condition                                               result (cached)
    ======================================================  =================
    ``arch`` is not :data:`LOCALITY_ARCH`                   None, no probe
    ``CAKE_PPLX_LOCALITY=0``                                None (yes)
    toolkit before :data:`LOCALITY_CUDA`, or the probe
    library does not build or load                          None (yes)
    driver before :data:`LOCALITY_CUDA`                     None (yes)
    the first call is under CUDA graph capture, or torch
    cannot tell whether it is                               None (no)
    the probe misses an SM :data:`PROBE_ATTEMPTS` times,
    the dies are not exactly 0 and 1, or a row is invalid  None (yes)
    ======================================================  =================

    The probe synchronizes the device once and runs on a private stream; call this
    (or the package's context set-up) eagerly before a graph capture, so a captured
    call and an eager call see the same table."""
    key = (int(device_index), str(arch))
    with _LOCK:
        if key in _TABLES:
            return _TABLES[key]
    if key[1] != LOCALITY_ARCH:
        return None
    if os.environ.get(ENV) == "0":
        return _remember(_TABLES, key, None)
    return _probe_eager(_TABLES, key)


def probe_table(device_index: int, arch: str) -> SmTable | None:
    """The SM table of device ``device_index`` from :func:`sm_table`'s probe, with
    no environment variable read: ``None`` off :data:`LOCALITY_ARCH` (no probe);
    ``None``, cached, for a toolkit or driver before :data:`LOCALITY_CUDA`, a probe
    library that does not build or load, or a probe that misses an SM
    :data:`PROBE_ATTEMPTS` times or does not find exactly dies 0 and 1; ``None``,
    not cached, under CUDA graph capture or when torch cannot tell. Its cache holds
    one result per process and device, apart from :func:`sm_table`'s."""
    key = (int(device_index), str(arch))
    with _LOCK:
        if key in _PROBED:
            return _PROBED[key]
    if key[1] != LOCALITY_ARCH:
        return None
    return _probe_eager(_PROBED, key)


def _probe_eager(
    cache: dict[tuple[int, str], SmTable | None],
    key: tuple[int, str],
) -> SmTable | None:
    """The probe of ``key``'s device into ``cache``, outside CUDA graph capture
    (``None`` uncached under capture or when torch cannot ask)."""
    import torch

    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - no CUDA runtime to ask: nothing to probe yet
        return None
    if capturing:
        return None
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    probe = disable(_probe) if disable is not None else _probe
    return _remember(cache, key, probe(key[0]))


def _remember(
    cache: dict[tuple[int, str], SmTable | None],
    key: tuple[int, str],
    table: SmTable | None,
) -> SmTable | None:
    with _LOCK:
        return cache.setdefault(key, table)


def _probe_library() -> Any:
    """The loaded probe library (built once through the package's JIT), or ``None``
    when the toolkit predates :data:`LOCALITY_CUDA` or the build or load fails."""
    with _LOCK:
        if _PROBE_MODULE:
            return _PROBE_MODULE[0]
    module = None
    try:
        from . import _cake_pplx_jit, cake_build

        version = _cake_pplx_jit.get_cuda_version()
        if (int(version.major), int(version.minor)) >= LOCALITY_CUDA:
            document = cake_build.manifest()
            root = os.environ.get("CAKE_PPLX_CACHE_DIR", "~/.cache/cake_pplx")
            source = Path(root).expanduser() / "locality" / f"{PROBE_NAME}.cu"
            _write_once(source, PROBE_SOURCE)
            module = _cake_pplx_jit.gen_jit_spec(
                PROBE_NAME,
                [source],
                extra_cuda_cflags=list(document["arch_flags"][LOCALITY_ARCH]),
                extra_ldflags=list(document["common"]["link_flags"]),
                use_fast_math=False,
            ).build_and_load()
    except Exception:  # noqa: BLE001 - no probe library: every call stays whole
        module = None
    with _LOCK:
        if not _PROBE_MODULE:
            _PROBE_MODULE.append(module)
        return _PROBE_MODULE[0]


def _write_once(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` unless the file already holds it (atomic replace)."""
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}")
    partial.write_bytes(data)
    os.replace(partial, path)


def _probe(device_index: int) -> SmTable | None:
    """Run the probe on device ``device_index`` (see :func:`sm_table`)."""
    import torch

    module = _probe_library()
    if module is None:
        return None
    try:
        driver = int(module.driver_version())
        if (driver // 1000, driver % 1000 // 10) < LOCALITY_CUDA:
            return None
        import tvm_ffi

        properties = torch.cuda.get_device_properties(device_index)
        ctas = int(properties.multi_processor_count)
        smem = int(properties.shared_memory_per_block_optin)
        out = torch.zeros((ctas, 4), dtype=torch.int32, device=f"cuda:{device_index}")
        torch.cuda.synchronize(device_index)
        stream = torch.cuda.Stream(device=device_index)
        rows: list[list[int]] = []
        for _ in range(PROBE_ATTEMPTS):
            with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
                module.probe(out, ctas, smem)
            stream.synchronize()
            rows = out.cpu().tolist()
            if len({row[0] for row in rows}) == ctas:
                break
        else:
            return None
    except Exception:  # noqa: BLE001 - a probe that cannot run routes nothing
        return None
    packed = pack_rows(rows)
    if packed is None:
        return None
    counts, words = packed
    entries = torch.tensor(words, dtype=torch.int32)
    return SmTable(device_index=device_index, counts=counts, entries=entries)


def _force_table(
    counts: Sequence[int],
    device_index: int = 0,
    arch: str = LOCALITY_ARCH,
) -> SmTable:
    """Test only: cache a table of two dies holding ``counts`` SMs (SM ids
    ``0 .. counts[0] - 1`` on die 0, the rest on die 1) for ``(device_index, arch)``,
    as the probe would, for :func:`sm_table` and :func:`probe_table`."""
    import torch

    n0, n1 = int(counts[0]), int(counts[1])
    rows = [(sm, 0 if sm < n0 else 1, 0, n0 + n1) for sm in range(n0 + n1)]
    packed = pack_rows(rows)
    if packed is None:
        raise ValueError(f"SM counts {tuple(counts)} make no SM table")
    words = torch.tensor(packed[1], dtype=torch.int32)
    table = SmTable(device_index=int(device_index), counts=packed[0], entries=words)
    with _LOCK:
        _TABLES[(int(device_index), str(arch))] = table
        _PROBED[(int(device_index), str(arch))] = table
    return table


def _forget() -> None:
    """Test only: drop every cached table, the probe library and the twin loads."""
    with _LOCK:
        _TABLES.clear()
        _PROBED.clear()
        _PROBE_MODULE.clear()
        _FAILED_TWINS.clear()
        _LOADED_TWINS.clear()


def twin_loaded(name: str, load: Callable[[str], Any]) -> bool:
    """Whether twin record ``name`` can launch: ``load(name)`` once per process; a
    load that raises removes the twin for the process, and its calls run the whole
    record."""
    with _LOCK:
        if name in _LOADED_TWINS:
            return True
        if name in _FAILED_TWINS:
            return False
    try:
        load(name)
    except Exception:  # noqa: BLE001 - a twin that does not load runs whole
        with _LOCK:
            _FAILED_TWINS.add(name)
        return False
    with _LOCK:
        _LOADED_TWINS.add(name)
    return True


# --------------------------------------------------------------------------- launch


def _per_die(logical: str) -> tuple[str, int | None]:
    """``("k", 1)`` for ``"k@1"``; ``(logical, None)`` for a name without a die."""
    name, sep, die = logical.rpartition("@")
    if not sep:
        return logical, None
    return name, int(die)


def counter_spec(record: Mapping[str, Any]) -> tuple[int, str]:
    """``(words, dtype)`` of each die's claim counters of twin ``record``: zero before
    the first launch, left zero by every launch; two launches in flight never share
    one."""
    counters = record["counters"]
    return int(counters["words"]), str(counters["dtype"])


def launch_grid(
    record: Mapping[str, Any],
    values: Mapping[str, Any],
    sm_count: int,
) -> tuple[int, int, int]:
    """The grid of twin ``record``: ``domain_items`` launches the product of its
    factors (logical scalars of ``values``, or ints), one CTA an item;
    ``domain_persistent`` that product capped at ``sm_count``; ``domain_sms`` one
    CTA per SM. ``sm_count`` is the device's own SM count. An empty problem has no
    launch (``ValueError``)."""
    rule = record["grid_rule"]
    kind = rule["kind"]
    if kind == "domain_sms":
        return (int(sm_count), 1, 1)
    product = 1
    for factor in rule["factors"]:
        product *= int(values[factor]) if isinstance(factor, str) else int(factor)
    if product < 1:
        raise ValueError(f"grid rule {dict(rule)} has no item to launch")
    if kind == "domain_items":
        return (product, 1, 1)
    if kind == "domain_persistent":
        return (max(1, min(product, int(sm_count))), 1, 1)
    raise ValueError(f"unknown locality grid rule {kind!r}")


def _workspace(logical: str, shape: Sequence[int], dtype: Any, device: Any) -> Any:
    import torch

    return torch.empty(tuple(shape), dtype=dtype, device=device)


def make_shards(
    record: Mapping[str, Any],
    values: Mapping[str, Any],
    split: Sequence[int],
    *,
    group: int,
    counters: Sequence[Any],
    alloc: Callable[[str, Sequence[int], Any, Any], Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[Any, Any]]]:
    """The logical values of one launch of twin ``record`` and the outputs to gather.

    ``values`` are the whole call's logical values (and the glue's ``sm_table``,
    ``qk_group``, ... where the twin takes them). ``split`` is ``(u0, u1)``: die 0
    holds Q/K heads ``[0, u0)``, die 1 ``[u0, u0 + u1)``; a ``"value"`` shard holds
    ``group`` value heads per Q/K head. ``counters`` are the two dies' counter
    tensors (:func:`counter_spec`). ``alloc(logical, shape, dtype, device)`` gives a
    contiguous workspace buffer (``torch.empty`` without it).

    Per ``<name>@d``: ``heads`` is ``u_d`` (``heads_param`` ``"qk"``) or ``u_d *
    group``; ``counters`` is ``counters[d]``; a shard of ``record["shards"]`` is made
    by its kind -- ``view`` the narrowed whole tensor, ``in`` a copy of it, ``out`` a
    buffer gathered back after the launch, ``inout`` both, ``replicated`` the whole
    tensor, ``placeholder`` a one-unit slice; any other name is the whole call's
    scalar. A die without heads gets one-unit slices and gathers nothing."""
    units = (int(split[0]), int(split[1]))
    shards = record["shards"]
    heads_param = record["heads_param"]
    found: dict[str, Any] = {}
    pending: list[tuple[Any, Any]] = []
    for logical in sorted(set(record["logical"].values())):
        name, die = _per_die(logical)
        if die is None:
            found[logical] = values[logical]
            continue
        count = units[die]
        if name == "heads":
            found[logical] = count if heads_param == "qk" else count * int(group)
            continue
        if name == "counters":
            found[logical] = counters[die]
            continue
        spec = shards.get(name)
        if spec is None:
            found[logical] = values[name]
            continue
        whole = values[name]
        kind = spec[0]
        if kind == "replicated":
            found[logical] = whole
            continue
        unit, axis = spec[1], int(spec[2])
        scale = 1 if unit == "qk" else int(group)
        if kind == "placeholder" or count == 0:
            found[logical] = whole.narrow(axis, 0, 1)
            continue
        part = whole.narrow(axis, (units[0] if die else 0) * scale, count * scale)
        if kind == "view":
            found[logical] = part
            continue
        make = alloc if alloc is not None else _workspace
        buffer = make(logical, tuple(part.shape), part.dtype, part.device)
        if kind in ("in", "inout"):
            buffer.copy_(part)
        if kind in ("out", "inout"):
            pending.append((part, buffer))
        found[logical] = buffer
    return found, pending


def gather(pending: Sequence[tuple[Any, Any]]) -> None:
    """Copy each die's output buffer into its slice of the whole output, on the
    current stream (after the twin's launch)."""
    for part, buffer in pending:
        part.copy_(buffer)
