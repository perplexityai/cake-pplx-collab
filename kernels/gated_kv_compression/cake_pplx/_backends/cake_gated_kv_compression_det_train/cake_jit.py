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

Registry and loader glue of the Cake ratio-2 gated KV compression training
kernels: document-bounded pooling of token pairs, forward and backward, width
512 (SM100 / SM103 / SM107).  The checks, the launches and the autograd nodes
are :mod:`.cake_backend`.

Every record of ``MODULES`` is one mechanically generated program: a CUDA
device translation unit plus a tvm-ffi binding, compiled once per
architecture.  A record is keyed by its role name (or ``<role>_<arch>`` for a
role that ships a different program per architecture) and carries the host
plan: the ``role`` it serves, the architectures it is delivered for, the
``run`` entry and its ``grid_rule``.  Every role is served on every
architecture of ``ARCHES`` by exactly one record; :func:`role_record` names
it, and :func:`run_module` launches the record of the device's
architecture.  The derived leaves (``library``, ``kernel_symbol``,
``launch``, ``arg_plan``) are written by the generator from the rendered
binding; ``library`` is the program identity
(``cake_gated_kv_compression_det_train_<20 hex>``), so a kernel swap renames
it.  ``MODULES`` is a literal table (the generator and the manifest checker
read it with ``ast.literal_eval``).  What to compile -- sources, digests, nvcc
flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through the package's
JIT shim from the same sources and flags.

The roles, in launch order:

* ``pair_table``: the exclusive prefix of the pair counts,
  ``cu_pairs[0] = 0`` and ``cu_pairs[s + 1] = cu_pairs[s] + (cu_seqlens[s + 1]
  - cu_seqlens[s]) // 2`` (int32 ``[S + 1]``), by one CTA that walks the
  documents in passes with a carried running total.  A forward launches it
  only for ``S > fuse_docs`` (:func:`fuse_docs`); an explicit backward without
  a table launches it for every ``S``.
* ``fwd``: one row of 512 columns per output row ``r < G``: the document ``s``
  of ``r`` by a binary search of ``cu_pairs``, the tokens
  ``t0 = cu_seqlens[s] + 2 (r - cu_pairs[s])`` and ``t1 = t0 + 1``, and
  ``out[r] = BF16(softmax-weighted sum of C[t0], C[t1])`` in FP32.  A row
  ``r >= cu_pairs[S]`` (a caller ``G`` above the pair count) is stored as
  zeros and reads nothing.  A launch of ``S <= fuse_docs`` documents (the
  record's ``fuse_docs`` leaf) builds the pair table itself: every CTA scans
  ``cu_seqlens`` into shared memory and searches it there, it reads no entry
  of ``cu_pairs``, and CTA 0 writes all ``S + 1`` entries, the bytes
  ``pair_table`` writes, for the backward.  A launch of more documents reads
  the table ``pair_table`` wrote.  The role has two records, one kernel with
  two schedules, each the faster on its architectures: ``fwd`` on ``sm_100a``
  and ``sm_103a`` reads ``C`` and ``Z`` in 128-bit lanes (4 FP32 channels,
  128 lanes a row); ``fwd_sm_107a`` on ``sm_107a`` reads them in 256-bit
  lanes (8 channels, 64 lanes a row), with the same FP32 operations per
  element.
* ``bwd``: virtual rows ``v < G + S``.  Row ``v < G`` is pair ``v``: it writes
  the ``dC`` and ``dZ`` rows of ``t0`` and ``t1`` (nothing for a pair
  ``v >= cu_pairs[S]``).  Row ``v >= G`` is document ``s = v - G``: when its
  length is odd it stores +0.0 into the ``dC`` and ``dZ`` rows of its last
  token.  Every token row below ``cu_seqlens[S]`` is written exactly once.

Every output element depends on its own pair (or token) only: no reduction
across threads, CTAs or calls, no atomic, so every output repeats bit for bit.
Grids come from host scalars only (``G``, ``S``): the host never reads
``cu_seqlens`` or ``cu_pairs``.  :func:`launch_grid` sizes them from the
record's ``grid_rule`` and its launch block: ``one_cta`` is ``(1, 1, 1)``;
``rows`` covers ``rows`` rows with ``block_x // lanes_per_row *
rows_per_lane_group`` rows per CTA (``lanes_per_row`` threads share one
512-wide row, ``rows_per_lane_group`` rows per group of them).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
PACKAGE = "cake_pplx._backends.cake_gated_kv_compression_det_train"
LIBRARY_PREFIX = "cake_gated_kv_compression_det_train"
MANIFEST = "build.json"

# The roles of the package's programs, in launch order (the pair table first, then the forward or the backward).  Every
# role is served on every architecture by exactly one record of MODULES (``role_record``).
ROLES = ("pair_table", "fwd", "bwd")
GRID_KINDS = ("one_cta", "rows")

# Columns of C, Z, dC, dZ, out and dout; ``cz`` holds C in columns [0, 512) and Z in [512, 1024).
WIDTH = 512
CZ_WIDTH = 2 * WIDTH
# Row strides of the strided FP32 operands (C, Z, dC, dZ) are multiples of this many elements (32 bytes), so every
# row starts on the 32-byte boundary of the 256-bit accesses of the backward and of the sm_107a forward.  The sm_100a /
# sm_103a forward reads C and Z as 128-bit vectors (a multiple of 4 elements would do there); the host keeps one rule on
# every architecture, so an input one record takes is one every record takes.
ROW_STRIDE_MULTIPLE = 8
# Base alignment of each operand (bytes).  The backward reads C and Z and writes dC and dZ as 256-bit vectors and reads
# dout as 128-bit vectors; the forward reads C and Z as 128-bit vectors and writes out as 64-bit vectors on sm_100a /
# sm_103a, and reads 256-bit and writes 128-bit vectors on sm_107a (16 bytes for out, as for dout); the int32 tables
# are read as words.
ALIGNMENT_BYTES: dict[str, int] = {
    "c": 32,
    "z": 32,
    "dc": 32,
    "dz": 32,
    "out": 16,
    "dout": 16,
    "cu_seqlens": 4,
    "cu_pairs": 4,
}
# The kernels take G, S, the token rows and the pair-table entries as int32 and form every element offset in 64 bits
# (row times row stride).  Their launch preconditions bound S by 2^24 and G by 2^26 (the backward's G + S virtual rows
# stay int32); T is bounded by the int32 cu_seqlens.
INT32_MAX = 2**31 - 1
MAX_PAIRS = 1 << 26
MAX_DOCS = 1 << 24
MAX_TOKENS = INT32_MAX

# The derived leaves of every record (``library``, ``kernel_symbol``, ``launch.block``, ``launch.cluster``,
# ``launch.cooperative``, ``launch.dynamic_smem_bytes``, ``launch.use_pdl`` and ``arg_plan``) and the ``fuse_docs`` of
# the forward records (the most documents a forward launch builds the pair table for) are rewritten by the generator
# from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "pair_table": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "cu_seqlens"],
            ["buffer", "cu_pairs"],
            ["parameter", "num_docs"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "one_cta"},
        "kernel_symbol": "kernel_cake_gated_kv_compression_det_train_8997a924b74a8200ddac",
        "launch": {
            "block": (1024, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 128,
            "use_pdl": "runtime",
        },
        "library": "cake_gated_kv_compression_det_train_8997a924b74a8200ddac",
        "role": "pair_table",
    },
    "fwd": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [
            ["buffer", "c"],
            ["buffer", "z"],
            ["buffer", "cu_seqlens"],
            ["buffer", "cu_pairs"],
            ["buffer", "out"],
            ["parameter", "num_docs"],
            ["parameter", "G"],
            ["parameter", "c_ld"],
            ["parameter", "z_ld"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "fuse_docs": 256,
        "grid_rule": {"kind": "rows", "lanes_per_row": 128, "rows_per_lane_group": 1},
        "kernel_symbol": "kernel_cake_gated_kv_compression_det_train_748a06110130c8ab8ec4",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2176,
            "use_pdl": "runtime",
        },
        "library": "cake_gated_kv_compression_det_train_748a06110130c8ab8ec4",
        "role": "fwd",
    },
    "fwd_sm_107a": {
        "arches": ["sm_107a"],
        "arg_plan": [
            ["buffer", "c"],
            ["buffer", "z"],
            ["buffer", "cu_seqlens"],
            ["buffer", "cu_pairs"],
            ["buffer", "out"],
            ["parameter", "num_docs"],
            ["parameter", "G"],
            ["parameter", "c_ld"],
            ["parameter", "z_ld"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "fuse_docs": 256,
        "grid_rule": {"kind": "rows", "lanes_per_row": 64, "rows_per_lane_group": 1},
        "kernel_symbol": "kernel_cake_gated_kv_compression_det_train_3bf8a60e0ac13a63d496",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 2176,
            "use_pdl": "runtime",
        },
        "library": "cake_gated_kv_compression_det_train_3bf8a60e0ac13a63d496",
        "role": "fwd",
    },
    "bwd": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [
            ["buffer", "dout"],
            ["buffer", "c"],
            ["buffer", "z"],
            ["buffer", "cu_seqlens"],
            ["buffer", "cu_pairs"],
            ["buffer", "dc"],
            ["buffer", "dz"],
            ["parameter", "num_docs"],
            ["parameter", "G"],
            ["parameter", "c_ld"],
            ["parameter", "z_ld"],
            ["parameter", "dc_ld"],
            ["parameter", "dz_ld"],
            ["launch", "enable_pdl"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows", "lanes_per_row": 64, "rows_per_lane_group": 1},
        "kernel_symbol": "kernel_cake_gated_kv_compression_det_train_0a0e2df45d6eb8316daf",
        "launch": {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": "runtime",
        },
        "library": "cake_gated_kv_compression_det_train_0a0e2df45d6eb8316daf",
        "role": "bwd",
    },
}


def arch_for_capability(capability: Sequence[int]) -> str | None:
    """The delivered architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Delivered architecture of one CUDA device (``RuntimeError`` when the device is not a delivered target); cached
    per device index."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not a delivered Cake gated KV compression target; "
            f"the delivered architectures are {ARCHES}"
        )
    return arch


@functools.cache
def role_record(role: str, arch: str) -> str:
    """The record of ``MODULES`` serving ``role`` on ``arch`` (``KeyError`` when none does); cached per pair."""
    for name, found in MODULES.items():
        if found["role"] == role and arch in found["arches"]:
            return name
    raise KeyError(f"no record of the Cake gated KV compression registry serves role {role!r} on {arch}")


@functools.cache
def fuse_docs(arch: str) -> int:
    """The most documents a forward launch on ``arch`` builds the pair table for itself (the ``fuse_docs`` leaf of the
    ``fwd`` record serving ``arch``; 0 when the record has none): the host launches ``pair_table`` before the forward
    only for more documents.  Cached per architecture."""
    return int(record(role_record("fwd", arch)).get("fuse_docs", 0))


def record(name: str) -> dict[str, Any]:
    """The registry record ``name`` (a key of ``MODULES``)."""
    found = MODULES.get(name)
    if found is None:
        raise KeyError(f"{name!r} is not a record of the Cake gated KV compression registry")
    return found


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake gated KV compression record {name} is not delivered for {arch}; its architectures: {found['arches']}"
        )
    return f"{found['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a builder
    compiled it, else the JIT build of the same sources and flags (``cake_build.load_library``).  Loaded once per
    process: later calls of any shape reuse it."""
    from . import cake_build  # noqa: PLC0415 - the generated loader, read on first use

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of ``build.json`` without a prebuilt file (for ahead-of-time
    warm-up builds, before a CUDA graph capture)."""
    from . import cake_build  # noqa: PLC0415

    return cake_build.jit_specs(arch)


@functools.cache
def parameters(name: str) -> tuple[str, ...]:
    """The kernel parameter names of record ``name`` in ``run`` order (the argument plan without the launch options,
    such as ``enable_pdl``, and the grid)."""
    names: list[str] = []
    axes: list[str] = []
    for kind, key in record(name)["arg_plan"]:
        if kind == "grid":
            axes.append(key)
        elif kind == "launch":
            if axes:
                raise ValueError(f"record {name}: launch option {key!r} follows the grid arguments")
        else:
            if axes:
                raise ValueError(f"record {name}: argument {key!r} follows the grid arguments")
            names.append(key)
    if axes != ["grid_x", "grid_y", "grid_z"]:
        raise ValueError(f"record {name}: the argument plan must end with grid_x, grid_y, grid_z; got {axes}")
    return tuple(names)


def rows_per_cta(name: str) -> int:
    """Rows one CTA of record ``name`` covers (a ``rows`` grid rule): ``block_x // lanes_per_row *
    rows_per_lane_group``."""
    found = record(name)
    rule = found["grid_rule"]
    if rule["kind"] != "rows":
        raise ValueError(f"record {name} has a {rule['kind']!r} grid rule, not 'rows'")
    rows = int(found["launch"]["block"][0]) // int(rule["lanes_per_row"]) * int(rule["rows_per_lane_group"])
    if rows < 1:
        raise ValueError(f"record {name}: its block {found['launch']['block']} covers no row ({rule})")
    return rows


def launch_grid(name: str, rows: int) -> tuple[int, int, int]:
    """The grid of record ``name``: ``(1, 1, 1)`` for a ``one_cta`` rule; for a ``rows`` rule, one CTA per
    :func:`rows_per_cta` rows of ``rows`` (``G`` for the forward, ``G + S`` for the backward), at least one."""
    rule = record(name)["grid_rule"]
    kind = rule["kind"]
    if kind == "one_cta":
        return (1, 1, 1)
    if kind == "rows":
        per = rows_per_cta(name)
        return (max(1, -(-int(rows) // per)), 1, 1)
    raise ValueError(f"unknown Cake gated KV compression grid rule {kind!r} (known: {GRID_KINDS})")


_TVM_FFI: Any = None
_TVM_DEVICES: dict[int, Any] = {}


def _tvm_ffi() -> Any:
    global _TVM_FFI
    if _TVM_FFI is None:
        import tvm_ffi  # noqa: PLC0415 - imported on the first launch

        _TVM_FFI = tvm_ffi
    return _TVM_FFI


def _tvm_device(device_index: int) -> Any:
    found = _TVM_DEVICES.get(device_index)
    if found is None:
        found = _TVM_DEVICES[device_index] = _tvm_ffi().device("cuda", device_index)
    return found


@functools.cache
def _entry(name: str, arch: str) -> Any:
    """The FFI entry of record ``name`` in its ``arch`` library (looked up once per process)."""
    return getattr(load(name, arch), record(name)["ffi_entry"])


def run_module(role: str, values: Mapping[str, Any], *, rows: int, device_index: int) -> None:
    """Launch the record serving ``role`` on the architecture of device ``device_index`` (:func:`role_record`) with
    its kernel arguments named in ``values`` on the grid :func:`launch_grid` gives ``rows``, on the current torch
    stream of that device.

    ``values`` maps the kernel parameter names of the record (through its ``logical`` map when the record has one) to
    tensors and Python scalars.  A ``["launch", "enable_pdl"]`` entry of the argument plan (a binding whose PDL launch
    is chosen per call, an int64 host parameter) gets ``1`` unless ``values`` sets it.  Nothing here reads device
    memory or synchronizes."""
    import torch  # noqa: PLC0415

    arch = device_arch(device_index)
    name = role_record(role, arch)
    found = record(name)
    logical = found.get("logical", {})
    grid = launch_grid(name, rows)
    axes = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in found["arg_plan"]:
        if kind == "grid":
            args.append(axes[key])
        elif kind == "launch":
            args.append(int(bool(values.get(key, True))))
        else:
            want = logical.get(key, key)
            if want not in values:
                raise ValueError(f"record {name} needs {want!r} (kernel parameter {key!r})")
            args.append(values[want])
    run = _entry(name, arch)
    stream = torch.cuda.current_stream(device_index).cuda_stream
    with _tvm_ffi().use_raw_stream(_tvm_device(device_index), stream):
        run(*args)


__all__ = [
    "ALIGNMENT_BYTES",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CZ_WIDTH",
    "GRID_KINDS",
    "INT32_MAX",
    "LIBRARY_PREFIX",
    "MANIFEST",
    "MAX_DOCS",
    "MAX_PAIRS",
    "MAX_TOKENS",
    "MODULES",
    "PACKAGE",
    "ROLES",
    "ROW_STRIDE_MULTIPLE",
    "WIDTH",
    "arch_for_capability",
    "device_arch",
    "fuse_docs",
    "launch_grid",
    "library_name",
    "load",
    "parameters",
    "record",
    "role_record",
    "rows_per_cta",
    "run_module",
    "specs_for_arch",
]
