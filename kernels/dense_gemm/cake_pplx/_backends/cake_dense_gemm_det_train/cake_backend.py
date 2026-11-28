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

Cake backend: the GLM-5.2 dense projection GEMMs for training and the FP32
router-gate GEMM on SM100 / SM103 / SM107.

K1 -- dense projection GEMM (BF16 operands, FP32 accumulation)
--------------------------------------------------------------
``out[l, m, n] = sum_k A[l, m, k] * B[l, k, n]`` for strided *views*:

* ``A`` BF16 ``[L, M, K]`` (or ``[M, K]``) with unit stride on ``m`` or ``k``;
  ``B`` BF16 ``[L, K, N]`` (or ``[K, N]``) with unit stride on ``k`` or ``n``.
  The other matrix stride and the batch stride are multiples of 8 elements
  (16 bytes); a view starts at a 16-byte aligned address.
* ``out`` BF16 or FP32 ``[L, M, N]`` / ``[M, N]`` with unit stride on ``n``,
  16-byte aligned rows and batches, or -- with ``transposed_out`` --
  ``[L, N, M]`` / ``[N, M]`` with unit stride on ``m`` (element ``(m, n)`` is
  stored at ``out[l, n, m]``, so a swapped small-``N`` weight gradient lands
  directly in ``dW[N, K]``).
* ``M``, ``K`` >= 1 and ``N`` a positive multiple of 8 are runtime values (no
  padding); the batch dimension ``L`` and its strides are arbitrary, so the
  token-major head views of the MLA projections (``[T, H, D]`` permuted to
  ``[H, T, D]``) run without copies.

The training layouts map onto the operator without copies (``T`` tokens):
forward ``X[T, K] @ W[N, K].T`` (``A = X`` K-major, ``B = W.T`` K-major),
input gradient ``G[T, N] @ W[N, K]`` (``B = W`` MN-major), weight gradient
``G[T, N].T @ X[T, K]`` (both MN-major, ``K = T`` ragged); a weight gradient
with fewer than 256 output rows runs the swapped ``X.T @ G`` with the
transposed store (:func:`cake_dense_gemm_det_train_projection_wgrad`).

The host selects one traced kernel instance per (A layout, B layout, tile
width, tile height, output kind, epilogue path, raster group, TMA L2 eviction
hints) exactly as the Cake production launcher does (:func:`instance_symbol`; the raster group
and the hints follow the device's CTA-pair count and L2 size through
:func:`default_group_m` and the wave working-set gate of :func:`default_hints`),
plans the stream-K split from the device's SM count (:func:`stream_k_plan`
with ``sk="auto"``: a
two-part K-aligned split of every tile only when the whole problem is one
partial wave whose halves fit the CTA pairs and K has at least 16 steps;
multi-wave problems stay data-parallel) and binds the argument plan of the
registered module (``cake_jit.MODULES`` / ``cake_jit.KERNELS``).  The result
is bitwise identical to the Cake launcher's on the same device.
The convenience path runs a batched forward of at most 256 tokens on the
swapped program instead (the weight rows in the MMA's M, at most two waves of
single-CTA tiles over the caller's strided views; :func:`swap_ab_template`).
Every K1 program launches as a programmatic dependent launch under
``enable_pdl`` (default: the device supports it, which every architecture the
package serves does): each CTA lets the next kernel launch at entry, and the
planner's programs wait for the preceding kernel before their first global
access (operands, output and workspace), so their prologue overlaps its tail.

K2 -- FP32 router GEMM through split-BF16x3 emulation
----------------------------------------------------
``out[m, n] = sum_k A[m, k] * B[k, n]`` for FP32 2-D views (``A`` with unit
stride on ``m`` or ``k``, ``B`` with unit stride on ``k`` or ``n``, ``out``
FP32 ``[M, N]`` with unit inner stride and 16-byte rows, ``N`` a multiple of
8).  The streamed operand ``A`` is split into three BF16 parts inside the
kernel; the second operand ``B`` is split on the host into a retained
``[3, outer, inner]`` BF16 stack in its own layout every launch (exact:
``b = b1 + b2 + b3``); the products with ``i + j <= 3`` are accumulated with
chunked FP32 promotion and, for the forward and the weight gradient, split-K
partials are reduced on the host in a fixed order.  Splits, chunk boundaries
and the reduction order are fixed: results are bit-exact run to run.  The
operand layouts select the instance and the production split count
(``kk`` forward 4, ``kn`` input gradient 1, ``nn`` -> the swapped
transposed-store ``nn_t`` weight gradient 6).

Two host paths share one planner and one binding:

* the prepared path -- :func:`cake_dense_gemm_det_train_projection_prepare` /
  :func:`cake_dense_gemm_det_train_router_prepare` plan once and own their workspace; the
  returned launch replays with no CUDA allocation and no host synchronization
  for new values written into the bound tensors.  Every ``launch()`` is
  launch-only: the generated binding encodes the operand tensor maps by value
  from the bound views on each call (no descriptor workspace), so a prepared
  launch can be captured into a CUDA Graph from its first call;
* the convenience path -- :func:`cake_dense_gemm_det_train_projection`,
  :func:`cake_dense_gemm_det_train_projection_forward` / :func:`cake_dense_gemm_det_train_projection_dgrad` / :func:`cake_dense_gemm_det_train_projection_wgrad`,
  :func:`cake_dense_gemm_det_train_router`, :func:`cake_dense_gemm_det_train_router_forward` / :func:`cake_dense_gemm_det_train_router_dgrad` /
  :func:`cake_dense_gemm_det_train_router_wgrad` -- keeps a bounded per-process cache of planned launches
  keyed by the device, the stream and the geometry of the views (shapes,
  strides, dtypes, storage offsets, alignment, flags); a call that hits the
  cache rebinds the caller's views and launches without planning, device
  queries or allocation (``out`` is allocated only when the caller passes
  none).  The cached launches of one device and stream share one workspace
  arena (:class:`_Arena`: one growable buffer per workspace kind) instead of
  owning a workspace each, so the memory the convenience path holds is bounded
  by the largest geometry seen, not by the number of geometries (a ragged
  token count).  See ``CONVENIENCE_CACHE_SIZE`` and
  :func:`cake_dense_gemm_det_train_convenience_cache_clear`.

The device facts a plan depends on (architecture, SM count, L2 size) are read
once per device index (:func:`device_facts`).
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Tuple, Union, cast

import torch
import tvm_ffi

from .cake_jit import (
    KERNELS,
    MODULES,
    arg_plan_of,
    load_cake_dense_projection_det_gemm_module,
    select_module,
)

SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}

# ---------------------------------------------------------------------------
# K1 tile configuration (mirror of the Cake kernel module's constants; the
# Cake kernel host is the K1 kernel module of the Cake kernel library,
# cited by line below)
# ---------------------------------------------------------------------------

BLOCK_M = 128  # A rows per MMA instruction per CTA (256 per CTA pair)  [Cake L48]
BLOCK_K = 64  # K elements per stage (128-byte swizzle rows)  [Cake L58]
PANEL_BYTES = (
    BLOCK_K * 128
)  # one MN-major B panel: 64 K rows x 128 B = 8192  [Cake L60]
B_SWZ_CHOICES = (
    128,
    64,
    32,
)  # round 13 (Cake L35): MN-major B panel width in bytes; 128 = the 64-column panels of rounds 1-12
# CTAs per work item: 2 = the pair tile (cta_group::2 MMA over 2 x 128 rows, cluster (2, 1, 1)); 1 = the round-19
# (Cake W3) single-CTA form ``cta1`` (one 128 x BLOCK_N tile per CTA, cluster (1, 1, 1))  [Cake L59]
CTA_GROUP = 2
EPI_WARPS = 8  # [Cake L67]
WORK_STAGES = 4  # cluster-launch-control work-ring depth  [Cake L68]
GROUP_M = 16  # CTA row tiles per raster group (even: the two CTAs of a pair are the row halves of one 256-row tile)  [Cake L69]
SLOT_BYTES = 32 * 128  # one epilogue staging slot (TMA-store epilogue)  [Cake L95]
SK_MIN_ITERS = 8  # stream-K: fewest K steps per unit  [Cake L72]
# stream-K: workspace slabs per tail tile of a linearised plan on the programs that key the workspace by tile (slab =
# tail tile * WS_PARTS + the unit's part on the tile; those programs require 2 * iters_per_unit >= k_blocks, so a
# tile holds at most WS_PARTS units - ``tile_keyed_workspace``)  [Cake WS_PARTS]
WS_PARTS = 3
K_TMA_BF16 = 1024  # bf16 rows with K at or below this: TMA-store epilogue  [Cake L103]
K_TWO_SLOTS = 0  # K at or below this: two staging slots per warp (off)  [Cake L104]
CH_T_COLS = 32  # output rows n per transposed TMA-store box / staging chunk (round 14)  [Cake L111]
# Stream-K slice counters: the real counters of the tail tiles use [0, tail_tiles * 16); lanes 1..31
# of a fixup warp fetch-add 0 to private dummy counters at SK_DUMMY_BASE + slice * 32 + lane (slice <
# 16), so the host asserts tail_tiles * 16 <= SK_DUMMY_BASE and allocates at least
# SK_DUMMY_BASE + 16 * 32 = 4608 counters (the Cake host allocates max(8192, tail_tiles * 16)).  [Cake L116, L122]
SK_DUMMY_BASE = 4096
SK_COUNTERS_MIN = (
    8192  # ``_sk_counters``: zeros of max(8192, needed) u32  [Cake L1395-L1402]
)
SMEM_OPT_IN = 232448  # 227 KiB dynamic shared memory opt-in per CTA (every architecture)  [Cake L117]
# CUDA 13.4 oversized shared-memory mode (launch attribute ALLOW_OVERSIZED; the Cake runtime's host shim enables
# it when a kernel exceeds the standard opt-in): per-architecture ceiling (R200: 334848 B) minus 512 B of headroom.
# Round 9: deeper pipelines on sm_107a.  [Cake L121]
SMEM_OVERSIZED = {"sm_107a": 334848 - 512}
L2_PROMOS = (
    "none",
    "l2_64b",
    "l2_128b",
    "l2_256b",
)  # TMA descriptor L2 promotion  [Cake L772]
W8_FORMS = (
    "w8r",
    "w8rs",
    "w8",
    "w8rel",
    "w8relf",
)  # the eight-warp CTA form of the input-gradient GEMM (instance_key field 37; ``_w8r`` / ``_w8rs`` / ``_w8`` /
# ``_w8rel`` / ``_w8relf`` symbols; ``w8`` = the knob-free program, ``w8rel`` = the ``w8r`` program with the start path,
# ``w8relf`` = the same with the epilogue's TMEM free; ``w8relf`` also in 128-column tiles, and there in 32-element K
# steps)
# the eight-warp forms with the start path: no batch raster and no store hint
_W8_START_FORMS = ("w8rel", "w8relf")
# the eight-warp forms that also run the 128-column pair tile (each CTA one 64-column B panel per K step; every output
# element sums its K in one FP32 accumulator in K order, the bits of the 256-column program): the full start path only
_W8_N128_FORMS = ("w8relf",)
# the mainloop ring of every eight-warp program at either tile width and K step: six stages of 32 KiB
W8_STAGES = 6
# the K step: 64 elements (one 128-byte swizzle row of BF16), or 32 for the 128-column eight-warp program (64-byte rows)
BLOCK_K_CHOICES = (64, 32)
# the instance-key fields every key has; field 40, the K step, follows only where it is not BLOCK_K
INSTANCE_KEY_FIELDS = 39


def instance_block_k(key: tuple) -> int:
    """The K step of instance key ``key``: its optional field 40, else :data:`BLOCK_K`."""
    return int(key[INSTANCE_KEY_FIELDS]) if len(key) > INSTANCE_KEY_FIELDS else BLOCK_K


PD_CHOICES = (
    0,
    1,
    2,
    3,
)  # round 15 (Cake W1): pipelined-drain forms of the 128-row single-pass epilogue (0 = the blocking load)
SK_SLAB_CHOICES = (
    0,
    1,
    2,
    3,
)  # round 19 (Cake W2): slab-path forms of the synchronised stream-K fixup (0 off, 1 early arrival, 2 + bulk slab read, 3 + fp32 output as the slab)
L2_HINTS = (
    "none",
    "evict_normal",
    "evict_first",
    "evict_last",
)  # TMA load L2 eviction policy  [Cake L773]
EPI_MODES = ("reg", "tma")  # [Cake L785]
# Round 9 adds BLOCK_N = 160 / 224 (wave-quantization fits: BLOCK_N / 2 must be a multiple of 16 so every warp
# slice is whole 16-column store chunks) and the 64-row Layout-B tile family (tcgen05 M=128 cta_group::2 pair
# tiles, four TMEM buffers; symbol suffix ``_m64``).  [Cake L775-L776]
BLOCK_N_CHOICES = (128, 160, 192, 224, 256)
CTA_ROWS_CHOICES = (64, 128, 256)

# ---------------------------------------------------------------------------
# K2 tile configuration (mirror of the Cake router kernel module's constants)
# ---------------------------------------------------------------------------

ROUTER_BLOCK_M = 128
ROUTER_BLOCK_K = 32  # fp32 elements split per stage
ROUTER_BLOCK_N = 128  # output columns per CTA pair
# Default split-K counts per instance layout (cake_dense_gemm_det_train_router_forward / cake_dense_gemm_det_train_router_dgrad / cake_dense_gemm_det_train_router_wgrad), where
# ROUTER_SPLIT_RULES names no count for the device and the row.
ROUTER_SPLITS = {"kk": 4, "kn": 1, "nn_t": 6}
# Measured split-K counts by device and row: (arch, SM count, layout, H, N) -> ((T_lo, T_hi, splits), ...), H the hidden
# size (K of the forward ``kk``, M of the swapped weight gradient ``nn_t``), N the experts (the output width) and T the
# token count (M of ``kk``, K of ``nn_t``) in the inclusive window [T_lo, T_hi].  At N = 256 the window T 16129-16384
# holds every token count with the forward's 128 M tiles (505-512 K blocks of the weight gradient); each of its counts
# was measured faster than the default one on GB200 / GB300 (152 SMs) and on B200 / B300 (148 SMs).  Since version
# 0.3.28 the key holds N: at N = 512 and H 4096 (T 32736) the forward runs one split and the weight gradient eight on
# B200 (148-SM sm_100a) and GB300 (152-SM sm_103a), and at N = 896 and H 7168 (T 16380) the forward runs one split on
# GB300; each window is the one measured token count, where the count was faster than the default one in eager
# launches and CUDA graph replays (21 paired rounds; paired medians 1.10 to 1.17).  Version 0.3.29 adds the same counts
# on GB200 (152-SM sm_100a) at N = 512 and H 4096 (T 32736): the forward runs one split and the weight gradient eight
# (paired medians 1.135 and 1.090).  Each count is timed for the whole router launch (the BF16x3 split, the program and
# the sum of the partials); other SM counts, N and token counts keep the default.  A single split stores the forward's
# 16-byte vectors straight into ``out``, so there ``out`` must start at a 16-byte aligned address (the planner refuses
# another).  A different split count is a different K partition: its outputs are bit-exact run to run but not bitwise
# equal to the default count's.
ROUTER_SPLIT_RULES: dict[
    tuple[str, int, str, int, int], tuple[tuple[int, int, int], ...]
] = {
    ("sm_100a", 148, "kk", 4096, 512): ((32736, 32736, 1),),
    ("sm_100a", 148, "nn_t", 4096, 512): ((32736, 32736, 8),),
    ("sm_100a", 148, "nn_t", 5120, 256): ((16129, 16384, 9),),
    ("sm_100a", 148, "nn_t", 6144, 256): ((16129, 16384, 3),),
    ("sm_100a", 152, "kk", 4096, 512): ((32736, 32736, 1),),
    ("sm_100a", 152, "nn_t", 4096, 512): ((32736, 32736, 8),),
    ("sm_100a", 152, "nn_t", 5120, 256): ((16129, 16384, 9),),
    ("sm_100a", 152, "nn_t", 6144, 256): ((16129, 16384, 3),),
    ("sm_103a", 148, "nn_t", 5120, 256): ((16129, 16384, 9),),
    ("sm_103a", 148, "nn_t", 6144, 256): ((16129, 16384, 3),),
    ("sm_103a", 152, "kk", 4096, 512): ((32736, 32736, 1),),
    ("sm_103a", 152, "kk", 7168, 896): ((16380, 16380, 1),),
    ("sm_103a", 152, "nn_t", 4096, 512): ((32736, 32736, 8),),
    ("sm_103a", 152, "nn_t", 5120, 256): ((16129, 16384, 9),),
    ("sm_103a", 152, "nn_t", 6144, 256): ((16129, 16384, 3),),
}


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceFacts:
    """The facts of one CUDA device the host path depends on, read once per device index: the architecture
    tag (``None`` on an unsupported device), the SM count (CTA pairs: stream-K split, raster group, wave working
    set) and the L2 size (the TMA eviction-hint working-set gate)."""

    index: int
    arch: Optional[str]
    sm_count: int
    l2_bytes: int


_DEVICE_FACTS: dict[int, DeviceFacts] = {}


def _cuda_index(device: Any = None) -> int:
    """Ordinal of a CUDA device (a ``torch.device``, an index or a ``"cuda[:n]"`` string; ``None`` or a bare
    ``cuda`` is the current device).  ``torch.device`` objects of one device are not interchangeable as dict
    keys (``cuda`` versus ``cuda:0``), so every per-device cache is keyed by this ordinal."""
    if device is None:
        return int(torch.cuda.current_device())
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device}")
    if device.index is None:
        return int(torch.cuda.current_device())
    return int(device.index)


def _driver_l2_bytes(index: int) -> int:
    from cuda.bindings import driver as cuda

    err, l2 = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, index
    )
    if err != cuda.CUresult.CUDA_SUCCESS or not l2:
        raise RuntimeError(
            f"cake_dense_gemm_det_train_projection: cannot resolve the L2 size of cuda:{index} ({err})"
        )
    return int(l2)


def device_facts(device: Any = None) -> DeviceFacts:
    """The cached :class:`DeviceFacts` of ``device``: one ``torch.cuda.get_device_properties`` query per
    device index for the lifetime of the process.  The L2 size is ``L2_cache_size`` of the properties,
    falling back to ``cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)`` -- the same source as the
    Cake host's ``_l2_bytes`` (no per-SKU table).  [Cake ``_l2_bytes`` L1377-L1392]"""
    index = _cuda_index(device)
    facts = _DEVICE_FACTS.get(index)
    if facts is None:
        props = torch.cuda.get_device_properties(index)
        l2 = getattr(props, "L2_cache_size", None) or _driver_l2_bytes(index)
        facts = _DEVICE_FACTS[index] = DeviceFacts(
            index=index,
            arch=SUPPORTED_COMPUTE_CAPABILITIES.get(
                (int(props.major), int(props.minor))
            ),
            sm_count=int(props.multi_processor_count),
            l2_bytes=int(l2),
        )
    return facts


def pdl_enabled(enable_pdl: Optional[bool], device: Any = None) -> bool:
    """``enable_pdl`` resolved for a launch on ``device``: as given, or, when ``None``, whether the device supports
    programmatic dependent launch (compute capability 9.0 or newer: every architecture the package serves, so a
    device :func:`device_facts` resolves an architecture for)."""
    if enable_pdl is not None:
        return bool(enable_pdl)
    return device_facts(device).arch is not None


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is not None and torch.device(device).type != "cuda":
        return None
    return device_facts(device).arch


def require_arch(device: torch.device) -> str:
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "the Cake dense projection GEMMs require compute capability 10.0, 10.3 or 10.7"
        )
    return arch


def is_cake_dense_gemm_det_train_supported(device: torch.device) -> bool:
    """Whether ``device`` is a CUDA device of compute capability 10.0, 10.3 or 10.7 (``sm_100a`` / ``sm_103a`` /
    ``sm_107a``).  Shape-level support is checked per call: an unsupported view, layout or size raises
    ``ValueError``, and a geometry without a registered program raises ``NotImplementedError``."""
    return arch_for(device) is not None


def generated_program_available(
    device: Optional[torch.device] = None, template: Optional[str] = None
) -> bool:
    """True when this checkout registers a program for ``device`` (serving ``template`` when given)."""
    arch = arch_for(device)
    if arch is None:
        return False
    table = KERNELS.get(arch, {})
    if template is None:
        return bool(table)
    return template in table


def tile_keyed_workspace(arch: str, template: str) -> bool:
    """Whether the program ``KERNELS[arch][template]`` keys its stream-K workspace by tail tile: the programs rendered
    from the current kernel source (a registry record whose ``build.module`` is the K1 kernel module,
    ``*.dense_projection_det_gemm``) store the partial of the unit that holds part ``j`` of tail tile ``t`` (``j``
    counted from the unit that holds the tile's first K step) in slab ``t * WS_PARTS + j`` of a linearised plan,
    ``t * p + j`` of the exact ``p``-way split and ``t`` of the synchronised one, and their bindings require
    ``2 * iters_per_unit >= k_iters`` of a linearised plan.  The carried programs (no ``build`` record) key it by
    unit (slab ``unit + t``) and take any linearised plan.  ``False`` for a template ``arch`` does not register."""
    module = KERNELS.get(arch, {}).get(template)
    if module is None:
        return False
    generator = MODULES[module].get("build", {}).get("module", "")
    return generator.endswith(".dense_projection_det_gemm")


def device_sm_count(device: torch.device) -> int:
    """SM count of ``device`` (cached per device index, see :func:`device_facts`)."""
    return device_facts(device).sm_count


def sm_pairs(sm_count: int) -> int:
    """CTA pairs the persistent grid can hold at once (one pair per two SMs).  [Cake ``_sm_pairs`` L1365-L1371]"""
    return max(1, int(sm_count) // CTA_GROUP)


EXACT_WAVE_BLOCK_N = 224  # the wave-quantisation tile width the exact-wave rules select (``exact_waves``)


def exact_waves(pair_tiles: int, units: int) -> bool:
    """Whether ``pair_tiles`` work items fill the device's ``units`` concurrent work items (CTA pairs, or SMs under
    ``cta1``) in whole waves - ``((m_tiles // 2) * n_tiles * L) % sm_pairs == 0`` in the pair form.  The BLOCK_N 224
    weight-gradient rules were chosen for this fit on the 76-pair measuring devices (152 SMs: the 4096 x 4096 row's
    152 tall tiles are two waves, the 4096 x 1024 row's 76 tiles one); the exact-wave gate of
    :func:`cake_dense_gemm_det_train_projection_plan` reads it for the running device."""
    return pair_tiles > 0 and pair_tiles % units == 0


def sm_units(sm_count: int, cta_group: int = CTA_GROUP) -> int:
    """Concurrent work items of the launch: CTA pairs (``cta_group`` 2) or SMs (the single-CTA form ``cta1``,
    round 19).  [Cake ``_sm_units``]"""
    if int(cta_group) == CTA_GROUP:
        return sm_pairs(sm_count)
    return max(1, int(sm_count))


def device_l2_bytes(device: torch.device) -> int:
    """L2 size of ``device`` in bytes (cached per device index, see :func:`device_facts`): resolves the hint
    working-set gate exactly as the Cake host's ``_l2_bytes`` does."""
    return device_facts(device).l2_bytes


# ---------------------------------------------------------------------------
# K1 instance selection (pure functions; mirror of the Cake launcher's)
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def epi_cols(block_n: int, cta_rows: int = 128) -> int:
    """Accumulator columns per epilogue warp slice: BLOCK_N / 2 (two warps per lane quadrant split
    the columns), or BLOCK_N / 4 for the 64-row Layout-B family (a quadrant holds one column half of
    32 rows; its two warps split that half).  [Cake ``epi_cols`` L779-L782]"""
    return block_n // 4 if cta_rows == 64 else block_n // 2


def epi_mode(
    out_f32: bool,
    out_t: bool,
    K: Optional[int] = None,
    epi: Optional[str] = None,
    block_n: int = 256,
    cta_rows: int = 128,
) -> str:
    """Epilogue of an instance: transposed output -> scalar register stores (``"reg"``);
    row-major fp32 -> per-warp TMA stores (``"tma"``) when the warp slice is whole 32-column
    chunks (the float4 register-store path is an opt-in ``epi="reg"``, measured 1-10 % slower
    on every fp32 row; a BLOCK_N = 160 / 224 or 64-row BLOCK_N = 192 slice has no TMA-store
    path); row-major bf16 -> TMA stores for short K (epilogue-bound tiles) when the slice is
    whole 64-column chunks, 32-byte register stores otherwise.  [Cake ``epi_mode`` L788-L805]"""
    if epi is not None:
        if epi not in EPI_MODES:
            raise ValueError(f"epi must be one of {EPI_MODES}, got {epi!r}")
        if out_t and epi != "reg" and epi_cols(block_n, cta_rows) % CH_T_COLS:
            # round 14: the transposed TMA-store epilogue stages CH_T_COLS-column chunks of the warp slice
            raise ValueError(
                "the transposed TMA-store epilogue needs warp slices of whole "
                f"{CH_T_COLS}-column chunks; BLOCK_N={block_n} CTA_ROWS={cta_rows} gives "
                f"{epi_cols(block_n, cta_rows)}"
            )
        return epi
    if out_t:
        return "reg"  # the transposed TMA store (round 14) is opt-in per row (ROW_RULES epi='tma')
    if out_f32:
        return "tma" if epi_cols(block_n, cta_rows) % 32 == 0 else "reg"
    # bf16 chunks are 64 columns: a 96-column (BLOCK_N = 192) or 32-column (Layout B, BLOCK_N = 128) warp slice has
    # no whole-chunk TMA-store path
    return (
        "tma"
        if (K is not None and K <= K_TMA_BF16 and epi_cols(block_n, cta_rows) % 64 == 0)
        else "reg"
    )


def epi_slots(
    epi: str,
    out_f32: bool,
    block_n: int,
    K: Optional[int] = None,
    slots: Optional[int] = None,
    cta_rows: int = 128,
    out_t: bool = False,
) -> int:
    """Staging slots per epilogue warp (0 without the TMA-store epilogue).  The chunk count
    per tile must be a multiple of the slot count so the round-robin rotation restarts at
    slot 0 on every tile.  Row-major chunks are 128-byte column groups (64 bf16 / 32 fp32);
    the transposed box (round 14, ``out_t``) stages ``CH_T_COLS`` columns per chunk.
    [Cake ``epi_slots`` L1151-L1164]"""
    if epi != "tma":
        return 0
    chunks = epi_cols(block_n, cta_rows) // (
        CH_T_COLS if out_t else (32 if out_f32 else 64)
    )
    if slots is None:
        slots = 2 if (K is None or K <= K_TWO_SLOTS) else 1
        if chunks % slots:
            slots = 1
    slots = int(slots)
    if slots not in (1, 2) or chunks % slots:
        raise ValueError(
            f"slots must be 1 or 2 and divide the {chunks} chunks per tile, got {slots}"
        )
    return slots


def staging_bytes(slots: int) -> int:
    # [Cake ``staging_bytes`` L824-L825]
    return EPI_WARPS * slots * SLOT_BYTES


def stt_bytes(stt: bool) -> int:
    """Dedicated staging bytes of the transposed stmatrix / TMA store (round 27, Cake lane 7, knob ``stt``): 0 - the
    SMEM plan stages the per-warp EPI_COLS x 64 B slots in the DEAD mainloop stage ring of the CTA's last work item, so
    no mainloop stage is traded; a dedicated-slot plan would be entered here and flow into ``default_stages`` and the
    ``instance_key`` SMEM check together.  [Cake ``stt_bytes``]"""
    return 0


def b_stage_bytes(
    b_mn: bool, block_n: int, b_swz: int = 128, cta_group: int = CTA_GROUP
) -> int:
    """Bytes of one B stage per CTA: K-major = ``BLOCK_N / cta_group`` 128-byte rows; MN-major = whole panels of
    ``b_swz / 2`` columns (BLOCK_N = 192 with 128-byte panels loads two 64-column panels per stage and the MMA
    reads 1.5 of them; with 64-byte panels it loads exactly the three 32-column panels it reads - round 13).
    The single-CTA form (``cta1``, round 19) streams the whole BLOCK_N columns per CTA.  [Cake ``b_stage_bytes``]"""
    n_half = block_n // cta_group
    if not b_mn:
        return n_half * BLOCK_K * 2
    cols = int(b_swz) // 2
    return (-(-n_half // cols)) * BLOCK_K * int(b_swz)


def a_halves_of(cta_rows: int) -> int:
    """128-row A halves per CTA tile (1 for the 64 / 128-row families, 2 for the tall family).  [Cake ``A_HALVES_OF``]"""
    return max(1, int(cta_rows) // BLOCK_M)


def smem_limit_for(arch: Optional[str]) -> int:
    """Dynamic SMEM an instance may request on ``arch``: the 227 KiB opt-in, or the architecture's
    oversized ceiling (``SMEM_OVERSIZED``).  ``None`` = the largest limit of any architecture
    (instance keys rebuilt from an exported plan, which was validated with its own architecture's
    limit).  [Cake ``smem_limit_for`` L835-L841]"""
    if arch is None:
        return max(SMEM_OPT_IN, *SMEM_OVERSIZED.values())
    return SMEM_OVERSIZED.get(str(arch), SMEM_OPT_IN)


def default_stages(
    slots: int,
    cta_rows: int = 128,
    block_n: int = 256,
    b_mn: bool = False,
    b_swz: int = 128,
    cta_group: int = CTA_GROUP,
    stt: bool = False,
) -> int:
    """Mainloop stages that fit the 227 KiB opt-in with the epilogue staging: 32 KiB stages
    for 128-row tiles (24 KiB at BLOCK_N = 128, where the streaming-bound small-N rows are
    still latency-bound at 7 stages: 9 / 8 / 6 for 0 / 1 / 2 slots), 48 KiB stages for tall
    (256-row) tiles.  Layout-B tiles (64 rows: 24 KiB stages at BLOCK_N = 256, 16 KiB at 128)
    take the deepest pipeline that fits beside the staging, at most 12 stages (round 9;
    ``b_mn`` sizes the MN-major B panels).  Narrow-panel instances (``b_swz`` 64 / 32, round 13) have smaller
    stages than their 128-byte-panel siblings and likewise take the deepest pipeline that fits the opt-in (at most
    12).  The single-CTA form (``cta1``, round 19: 48 KiB stages at BLOCK_N 256, 32 KiB at 128) takes the deepest
    fit the same way.  ``stt`` (round 27, Cake lane 7) subtracts the transposed stmatrix store's dedicated staging bytes
    (``stt_bytes``: 0 under the dead-ring plan) beside the slots'.  [Cake ``default_stages``]"""
    if cta_rows == 64 or int(b_swz) != 128 or int(cta_group) == 1:
        stage = min(cta_rows, 128) * a_halves_of(
            cta_rows
        ) * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cta_group)
        return max(
            2,
            min(
                12,
                (SMEM_OPT_IN - WORK_STAGES * 16 - staging_bytes(slots) - stt_bytes(stt))
                // stage,
            ),
        )
    if cta_rows != 128:
        return (4, 4, 3)[slots]
    if block_n == 128:
        return (9, 8, 6)[slots]
    return (7, 6, 5)[slots]


def box_rows_of(
    a_mn: bool,
    b_mn: bool,
    block_n: int,
    box_rows: Optional[int] = None,
    cta_rows: int = 128,
    cta_group: int = CTA_GROUP,
) -> tuple[int, int]:
    """TMA box heights of the A and B operands: full height unless capped by ``box_rows``.
    [Cake ``box_rows_of`` L860-L870]"""
    a_full = BLOCK_K if a_mn else cta_rows
    b_full = BLOCK_K if b_mn else block_n // cta_group
    if box_rows is None:
        return a_full, b_full
    box_rows = int(box_rows)
    a_box, b_box = min(a_full, box_rows), min(b_full, box_rows)
    if box_rows < 8 or a_full % a_box or b_full % b_box:
        raise ValueError(
            f"box_rows={box_rows} must divide the operand heights {a_full} / {b_full}"
        )
    return a_box, b_box


def instance_key(
    *,
    a_mn: bool,
    b_mn: bool,
    out_f32: bool = False,
    out_t: bool = False,
    block_n: int = 256,
    stages: Optional[int] = None,
    diag: tuple = (),
    epi: Optional[str] = None,
    slots: Optional[int] = None,
    box_rows: Optional[int] = None,
    cta_rows: int = 128,
    pf: int = 0,
    hints: tuple = ("none", "none"),
    f32_v8: bool = False,
    quad_store: bool = False,
    park: bool = False,
    ovl: bool = False,
    htail: bool = False,
    smem_limit: Optional[int] = None,
    b_swz: int = 128,
    sk_exact: bool = False,
    batch_group: int = 0,
    store_ef: bool = False,
    store_hint: str = "none",
    pd: int = 0,
    sh: int = 0,
    sk_sync: bool = False,
    sk_slab: int = 0,
    cta1: bool = False,
    cgrp: bool = False,
    ovr: bool = False,
    stt: bool = False,
    pro: int = 0,
    sk_first: bool = False,
    lw: bool = False,
    mg: bool = False,
    sw: int = 0,
    form: str = "",
    kser: int = 0,
    promote_blocks: int = 0,
    block_k: int = BLOCK_K,
) -> tuple:
    """The instance tuple the Cake kernel module traces one program per (validation included):
    ``(a_mn, b_mn, out_f32, out_t, block_n, stages, diag, epi, slots, box_rows, cta_rows, pf,
    hints, f32_v8, quad_store, park, ovl, htail, b_swz, sk_exact)`` (round 13: ``park`` = parked tall epilogue, ``ovl`` =
    overlapped single-TMEM-buffer tall epilogue, ``htail`` = deterministic half-height tail wave, ``b_swz`` = MN-major B
    panel width in bytes (128 / 64 / 32; K-major B instances always keep 128), ``sk_exact`` = exact p-way stream-K
    split, ``_skx`` symbols; round 18 (Cake W2): ``sk_sync`` = synchronised stream-K tail, field 26, ``_sks`` symbols;
    round 19: ``sk_slab`` (Cake W2, field 27, ``_sb{n}``) and ``cta1`` (Cake W3, field 28, ``_c1``); round 21: ``cgrp`` (Cake W3, field 29, ``_cg``); round 24: ``ovr`` (the round-20 Cake W3 knob ported in round 23 by W4, field 30 = LAST, ``_or``), see below); the raster group width (``group_m``) and the TMA L2 promotion (``promo_code``) are launch parameters since round 11 (under ``cgrp`` the group counts column tiles).  ``smem_limit`` (bytes; default = the largest
    architecture limit, ``smem_limit_for(None)``) only bounds the stage count - it is not part of
    the key, so an instance has one symbol on every architecture (the planner passes
    ``smem_limit_for(arch)`` like the Cake launcher).  Diagnostic (attribution) instances are not
    exported.  [Cake ``instance_key`` L873-L905]  ``batch_group`` (round 13, L38b) = the batch-entry raster of the
    batched MLA rows: 0 = off, 1 = every entry (``_bf``), G > 1 = G adjacent entries interleaved across the CTA pairs
    (``_bg{G}``); the planner keeps 0 for a batch of one.  ``store_ef`` (round 13, Cake W4) = st.global.L1::no_allocate.L2::evict_first on the fp32 v8
    register stores (``_ef``); only that store form carries the hint, so the field is False for every other epilogue.
    ``store_hint`` (round 15, Cake W3) = the L2 eviction policy operand of the TMA-store epilogue's bulk tensor stores
    (field 22, ``_so{f|l|n}``); only the TMA-store epilogue carries the operand, so the field is "none" for every other
    epilogue.  ``pd`` (round 15, Cake W1) = the software-pipelined chunked TMEM drain of the 128-row single-pass family
    (field 23, ``_pd{n}``: 1 = in-tile chunk pipelining, 2 = cross-tile prefetch, 3 = both chunks in flight) and ``sh``
    (field 24, ``_sh{n}``) = the PTX suspendTimeHint in ns of the free-running waits; both 0 when off.
    ``sk_slab`` (round 19, Cake W2, field 27, symbol ``_sb{n}``) selects the slab path of the synchronised stream-K fixup
    (1 early arrival; 2 + the bulk slab read through the dead mainloop stages; 3 + the fp32 output as the slab: fp32
    outputs through the transposed register path or the TMA-store epilogue); it is forced to 0 without ``sk_sync`` so
    every other instance keeps one key.  ``cta1`` (round 19, Cake W3, field 28, symbol ``_c1``) selects the
    single-CTA form of the 128-row single-pass family: one 128 x BLOCK_N tile per CTA through the cta_group::1 MMA into
    private TMEM, every barrier local, CLC per CTA (cluster dims 1) - no pair handoff; it excludes the pair-only tail
    policies / probes (``htail`` / ``sk_exact`` / ``sk_sync`` / ``a_mcast``), the tall and 64-row families (``ovl`` /
    ``park``) and the batched raster knob.  Its B stage streams the whole BLOCK_N columns per CTA, so the stage
    geometry (``b_stage_bytes`` / ``default_stages`` / ``box_rows_of``) takes the CTA group.
    ``cgrp`` (round 21, Cake W3, field 29, symbol ``_cg``) selects the column-grouped raster: the ``group_m``
    launch parameter counts COLUMN tiles per raster group (any count >= 1) and consecutive pairs sweep a group's
    columns of one pair row before the next pair row (a group's B panels stay L2-resident, A streams once per column
    group); every other field is unchanged and the OFF form renders byte-identical.
    ``ovr`` (round 24; the round-20 Cake W3 knob ported in round 23 by W4, field 30 = LAST, symbol ``_or`` after
    ``_cg``) selects the overlapped tall epilogue released by row half: the eight epilogue warps drain TMEM buffer 0
    (rows [0, 128), 32 rows x 128 columns per warp) and release it to the next tile's half-0 MMAs while buffer 1 is
    drained; the B stage keeps the plain tall layout (one N = 256 MMA per row half per K step - half the tcgen05.mma
    issues and 3/4 of the tensor-core SMEM operand traffic of ``ovl``).  Tall 256-column tiles only; it excludes
    ``ovl`` / ``park`` and the pf / box_rows / a_mcast probes, and (like ``ovl``) the synchronised stream-K plan and
    the single-CTA form; every other field is unchanged and the OFF form renders byte-identical.
    ``stt`` (round 27, Cake lane 7 / W-3 variant A1, field 31 = LAST after ``ovr`` = 30, symbol ``_stt`` directly after
    ``_sks``) selects the transposed stmatrix / TMA-store staging of the bf16 transposed REGISTER epilogue of the 128-row
    single-pass pair family: each epilogue warp writes its 32 x EPI_COLS bf16 slice with ``stmatrix.trans`` into a
    64 B-swizzled slot of the dead mainloop stage ring of the CTA's last work item and stores it with one bulk tensor
    store (``OUT16`` = the real [L, N, M] view, box (32, EPI_COLS, 1)); every other slice keeps the register stores
    through ``out``.  bf16 transposed outputs only (stmatrix is .b16), whole 32-column warp slices up to 256 columns,
    no ovl / ovr / park / htail / pd / cta1, ``sk_slab`` 0 / 1, and the pool must fit the 227 KiB opt-in; every other
    field is unchanged and the OFF form renders byte-identical.
    ``pro`` (round 33, Cake unit 2d / 2e, field 32 = LAST after ``stt`` = 31, symbol ``_pro{n}`` directly after ``_or`` =
    the last knob suffix) is the bitmask of the pair program's prologue / cleanup sync form: 1 = no peer-startup
    cluster rendezvous before the collective tcgen05.alloc, 2 = relaxed arrive on the post-init cluster sync, 4 = the
    MMA warp allocates TMEM, 8 = CTA-local exit (no cleanup barrier.cluster: role-drained rings, a leader -> follower
    tmem_free handoff and the collective dealloc on warp 1).  Pair programs only (the single-CTA form ``cta1`` refuses
    it), 0..15, and bit 8 admits no diag probe but ``trace`` (the planner never emits diag); 0 renders the round-27
    program byte-identical.  The production value is the merged row rule's (``row_rule`` lays ``ARCH_RULE_DEFAULTS``
    under every row rule: 7 on sm_107a and sm_100a; a row rule may pin another value), zeroed on the ``cta1`` rows.
    ``sk_first`` (the round-2 shapes tranche, field 33, symbol ``_skf`` directly after ``_skx``) is the tail-first
    persistent order of the exact p-way split: pair work items ``[0, sk_units)`` are the K parts of the tail tiles and
    item ``sk_units + w`` is whole tile ``w``; the plan arithmetic, the counters, the slabs and the launch contract are
    ``sk_exact``'s - only the program differs.  It needs ``sk_exact`` and keeps the plain item walk (no ``pd`` 2,
    ``sk_slab`` 2, ``stt``, ``sk_sync`` or ``htail``).  ``lw`` (the round-2 shapes tranche, field 34 = LAST, symbol
    ``_lw`` after ``_or`` and before ``_pro``) lets the load warp leave the pair prelude before the post-allocation CTA
    rendezvous (the first TMA loads issue once the cluster rendezvous and the barrier init are done); pair programs
    whose MMA warp allocates TMEM only (``pro`` bit 4, never ``cta1``).  ``mg`` (field 35 = LAST, symbol ``_mg``
    directly after ``_lw``) selects the whole-tile tile resolve: the load warp and the MMA warp take the whole-tile
    constant arm of the work-item walk (one segment per item, no division) and the load warp resolves the raster's
    three quotients through an fp32 reciprocal with an exact +-1 correction (four reciprocals set up once per CTA);
    the epilogue keeps the plain path.  Whole tiles only (the kernel requires a plan without stream-K units and
    fewer than 2**21 cluster tiles), so it excludes the batch raster, the column-grouped raster, the half-height
    tail and every stream-K form (``sk_exact`` / ``sk_first`` / ``sk_sync`` / ``sk_slab`` / ``stt``); every other
    field is unchanged and the OFF form renders byte-identical.  ``sw`` (field 36, symbol ``_sw<n>``
    directly after ``_mg`` and before ``_pro``; 0 = off) paces the output stores of the bf16 TMA-store pd-2 drain:
    each epilogue warp delays its bulk tensor store issue, after the staging write and its fences, by a per-warp
    number of steps of ``sw`` dependent u32 multiply-adds seeded by a clock sample (even chunks (9 - warp) steps,
    odd chunks (warp - 2) steps), so each per-item eight-store burst becomes a seven-step ramp; the stores and
    bytes are unchanged.  A step count >= 0 that needs that drain (``slots`` > 0, ``pd`` 2, no fp32 output);
    every other field is unchanged and the OFF form renders byte-identical.  ``form`` (field 37 = LAST, symbol
    ``_w8r`` / ``_w8rs`` / ``_w8`` / ``_w8rel`` / ``_w8relf`` directly after ``_pro<n>`` and before the diag tokens;
    "" = the ten-warp pipeline
    of every other program) selects the eight-warp CTA form of the same GEMM: ``w8r`` = four epilogue warps, a
    loader, a scheduler and the MMA warp on CTA-pair 256 x 256 tiles over six 32 KiB stages, whole tiles through
    cluster-launch-control items, the BF16 output through the TMA store and the raster quotients resolved by four
    fast divisors the host passes (``fd_tiles_l`` / ``fd_tiles_g`` / ``fd_group`` / ``fd_last``), with the
    accumulator released by every epilogue warp after its last read; ``w8rs`` = the same with the loader's elected
    lane issuing the cluster-launch-control queries (no scheduler warp); ``w8`` = the knob-free program of the form
    (sm_100a): the scheduler warp issues the queries and the accumulator is released once per tile; ``w8rel`` = the
    ``w8r`` program with the start path (sm_100a and sm_103a): the loader's elected lane prefetches the first item's
    first K-step boxes into L2 in the prologue, the pair's landing barriers are armed and waited on at cluster scope,
    each role waits on the cluster rendezvous just before its first operation that needs the peer CTA and the MMA
    warp allocates the pair's TMEM itself at the constant base 0; ``w8relf`` = the same with the epilogue freeing the
    TMEM under the last item's store drain (both keep every ring, item, MMA and summation order of ``w8r``: its
    bits); ``w8relf`` also runs the 128 x 128 pair tile (``block_n`` 128: each CTA one 64-column B panel per K step,
    the same ring, item walk and summation order, so the bits of its 256-column program).  A form key's default stage
    count is the ring's six (:data:`W8_STAGES`).  It is refused
    with every field it cannot
    carry: a layout other than ``kn``, an fp32 or transposed output, the single-CTA form, a tile other than the
    128 x 256 pair tile (or, for ``w8relf``, the 128 x 128 one), an epilogue other than the one-slot TMA store, a
    narrow B panel, a box cap, any tail or
    raster form (``sk_exact`` / ``sk_first`` / ``sk_sync`` / ``sk_slab`` / ``htail`` / ``cgrp`` / ``batch_group`` /
    ``stt`` / ``ovl`` / ``ovr`` / ``park``) and any pair-program knob off its plain value (``pd`` / ``pro`` / ``lw`` /
    ``mg`` / ``sw`` / a stage count other than six / ``hints`` / ``pf`` / ``sh`` / ``store_hint`` / ``store_ef`` /
    ``quad_store`` / ``f32_v8``); the head-interleaved batch raster G > 1 and, with it, the ``evict_last`` store hint
    (the batched MLA programs' knobs) are admitted with the other forms and refused with ``w8rel`` / ``w8relf``;
    the OFF form renders byte-identical.  ``kser`` (field 38 = LAST, symbol ``_ks<P>``
    directly after ``_sw<n>`` and before ``_pro<n>``; 0 = off) is the serpentine K walk of the pair programs: the load
    warp walks the K blocks of a whole-K item (every K block of its tile in one item) whose item index lies in an odd
    block of P items descending, so a wave of P pairs starts on the K end the previous wave left in L2; the MMA warp is
    unchanged.  The direction is a pure function of the item and P: run-to-run deterministic, another FP32 summation
    order than ``kser`` 0.  A pair count >= 0 on the plain item walk: no ``cta1``, ``sk_exact`` / ``sk_first`` /
    ``sk_sync`` / ``stt``, ``mg``, ``batch_group`` or ``form``; every other field is unchanged and the OFF form renders
    byte-identical.  ``promote_blocks`` (field 39 = LAST, symbol ``_kc<KC>`` directly after ``_ks<P>`` and before
    ``_pro<n>``; 0 = off) is the promoted chunk of the nn pair programs (:data:`K1_PROMOTE_BLOCKS`): each segment's K
    blocks are cut into balanced contiguous chunks of at most KC blocks, each summed in an accumulator of its own and
    added in chunk order with one FP32 round-to-nearest addition per element (run-to-run deterministic; one chunk gives
    the bits of the program without it).  The kernel takes it where :func:`promote_refusal` admits the key; every other
    field is unchanged and the OFF form renders byte-identical.  ``block_k`` (field 40, present only when not 64;
    symbol ``_k<K>`` directly after ``_n<N>``) is the K step: 64 elements, or 32 for ``w8relf`` on the 128-column tile
    (a CTA's A box of 128 rows by 32 columns in the 64-byte swizzle, two K-16 MMAs per stage in the same six 32 KiB
    stages; every output element sums its K in one FP32 accumulator in K order, the bits of the 64-element program).
    Every other field is unchanged and the 64-element key is byte-identical (39 fields, :func:`instance_block_k`).
    [Cake ``instance_key``]"""
    a_mn, b_mn, out_f32, out_t = bool(a_mn), bool(b_mn), bool(out_f32), bool(out_t)
    block_n, cta_rows, pf = int(block_n), int(cta_rows), int(pf)
    cta1 = bool(cta1)
    cgrp = bool(cgrp)
    ovr = bool(ovr)
    stt = bool(stt)
    cg = 1 if cta1 else CTA_GROUP
    hints = (str(hints[0]), str(hints[1]))
    if any(h not in L2_HINTS for h in hints):
        raise ValueError(f"hints must be in {L2_HINTS}, got {hints!r}")
    if pf < 0 or pf > 16:
        raise ValueError(f"prefetch distance must be in 0..16 K steps, got {pf}")
    if cta_rows not in CTA_ROWS_CHOICES:
        raise ValueError(f"cta_rows must be one of {CTA_ROWS_CHOICES}, got {cta_rows}")
    box_rows = 0 if box_rows is None else int(box_rows)
    box_rows_of(a_mn, b_mn, block_n, box_rows or None, cta_rows, cg)
    if block_n not in BLOCK_N_CHOICES:
        raise ValueError(f"BLOCK_N must be one of {BLOCK_N_CHOICES}, got {block_n}")
    cols = epi_cols(block_n, cta_rows)
    if cols % 16:
        raise ValueError(
            f"BLOCK_N={block_n} with CTA_ROWS={cta_rows} gives {cols}-column warp slices; "
            "slices must be whole 16-column chunks"
        )
    b_swz = int(b_swz) if b_mn else 128
    if b_swz not in B_SWZ_CHOICES:
        raise ValueError(f"b_swz must be one of {B_SWZ_CHOICES}, got {b_swz}")
    batch_group = int(batch_group)
    if batch_group < 0:
        raise ValueError(
            f"batch_group must be 0 (off), 1 (every batch entry) or a group size G > 1, got {batch_group}"
        )
    epi = epi_mode(out_f32, out_t, None, epi, block_n, cta_rows)
    # a transposed box stages CH_T_COLS output rows per chunk (round 14); row-major chunks are 128 bytes  [Cake L1260]
    if epi == "tma" and cols % (CH_T_COLS if out_t else (32 if out_f32 else 64)):
        raise ValueError(
            f"the TMA-store epilogue needs whole 128-byte column chunks per warp; BLOCK_N={block_n} "
            f"CTA_ROWS={cta_rows} {'fp32' if out_f32 else 'bf16'} output needs epi='reg'"
        )
    slots = epi_slots(epi, out_f32, block_n, None, slots, cta_rows, out_t)
    # the 128-column eight-warp program keeps the form's six-stage ring (``default_stages`` gives the 256-column form
    # its six already; ``form`` is normalised below, with its refusal)
    stages = (
        (
            W8_STAGES
            if str(form or "") in _W8_N128_FORMS and block_n == 128
            else default_stages(slots, cta_rows, block_n, b_mn, b_swz, cg, stt)
        )
        if stages is None
        else int(stages)
    )
    diag = tuple(sorted(set(diag)))
    if diag:
        raise ValueError(f"diagnostic instances are not exported: {diag}")
    limit = smem_limit_for(None) if smem_limit is None else int(smem_limit)
    stage_bytes = cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg)
    if (
        stages < 2
        or stages * stage_bytes
        + staging_bytes(slots)
        + stt_bytes(stt)
        + WORK_STAGES * 16
        > limit
    ):
        raise ValueError(
            f"{stages} stages at BLOCK_N={block_n}, CTA_ROWS={cta_rows} exceed the {limit} B dynamic SMEM limit"
        )
    f32_v8 = (
        bool(f32_v8) and out_f32 and epi == "reg" and not out_t
    )  # only the row-major fp32 register epilogue has the knob
    quad_store = (
        bool(quad_store)
        and (not out_f32)
        and epi == "reg"
        and not out_t
        # bf16 row-major register epilogue with whole 64-column groups per warp slice: quad-transposed
        # 32-byte row segments (round 7)
        and cols % 64 == 0
    )
    # parked tall epilogue: bf16 row-major 256-row tiles only (Cake round 10 knob, measured per row)
    park = bool(park) and cta_rows == 256 and (not out_f32) and (not out_t)
    ovl = bool(ovl)
    if ovl and (cta_rows != 256 or block_n != 256 or park or pf or box_rows):
        # round 13: the overlapped epilogue's chunks are the two 128-column passes of the tall 256-column tile; it
        # owns the B load coordinates (no prefetch / box cap) and excludes the parked epilogue  [Cake instance_key]
        raise ValueError(
            f"ovl needs cta_rows=256, block_n=256 and no park / pf / box_rows (got cta_rows={cta_rows}, "
            f"block_n={block_n}, park={park}, pf={pf}, box_rows={box_rows})"
        )
    if ovl and b_swz != 128:
        raise ValueError(
            f"ovl owns the 256-column B load coordinates and needs the 128-byte B swizzle (got b_swz={b_swz})"
        )
    if ovr and (
        cta_rows != 256
        or block_n != 256
        or ovl
        or park
        or pf
        or box_rows
        or "a_mcast" in diag
    ):
        # round 20 (Cake W3, ported in round 23 by W4): the row-half release is the overlapped epilogue of the tall
        # 256-column tile over the plain B stage (one N = 256 MMA per row half per K step); it excludes the column-chunk
        # release, the parked epilogue and the prefetch / box cap / A-multicast probe (the planner never emits diag, so
        # the a_mcast term is inert; kept for message parity)  [Cake instance_key]
        raise ValueError(
            f"ovr needs cta_rows=256, block_n=256 and no ovl / park / pf / box_rows / a_mcast (got cta_rows={cta_rows}, "
            f"block_n={block_n}, ovl={ovl}, park={park}, pf={pf}, box_rows={box_rows}, diag={diag})"
        )
    htail = bool(htail)
    if htail and (cta_rows not in (128, 256) or pf or box_rows or "a_mcast" in diag):
        raise ValueError(
            f"htail (half-height tail wave) needs cta_rows=256 or 128 and no pf / box_rows / a_mcast (got cta_rows={cta_rows}, "
            f"pf={pf}, box_rows={box_rows})"
        )
    if (
        htail
        and cta_rows == 128
        and (block_n != 256 or out_t or int(pd) or "no_tmem" in diag)
    ):
        # round 16 (Cake W2): the half-height tail wave on the standard family (M=128 cta_group::2 half items, TMEM Layout B)
        # needs the 256-column row-major tile with the plain drain  [Cake instance_key]
        raise ValueError(
            f"htail on the 128-row family needs BLOCK_N=256, a row-major output and pd=0 without the no_tmem probe "
            f"(got block_n={block_n}, out_t={out_t}, pd={pd}, diag={diag})"
        )
    if htail and sk_exact:
        raise ValueError(
            "htail (half-height tail wave) and sk_exact (exact p-way tail split) are two tail policies - choose one per instance"
        )
    sk_sync = bool(sk_sync)
    if sk_sync and (htail or sk_exact):
        raise ValueError(
            "sk_sync (synchronised stream-K) excludes htail and sk_exact - one tail policy per instance"
        )
    if sk_sync and (ovl or ovr or park or int(pd) or "a_mcast" in diag):
        # round 18 (Cake W2): the synchronised plan's two-partial fixup closes a unit's single-pass or serialised tall
        # epilogue; the overlapped (column-chunk ``ovl`` / row-half ``ovr``) / parked tall forms, the pipelined drain
        # and the A multicast are excluded  [Cake instance_key]
        raise ValueError(
            f"sk_sync needs the single-pass or the serialised tall epilogue (no ovl / ovr / park / pd / a_mcast; got ovl={ovl}, "
            f"ovr={ovr}, park={park}, pd={pd}, diag={diag})"
        )
    # L1::no_allocate.L2::evict_first on the fp32 v8 stores: only that store form carries the hint  [Cake instance_key]
    store_ef = bool(store_ef) and f32_v8
    store_hint = str(store_hint)
    if store_hint not in L2_HINTS:
        raise ValueError(f"store_hint must be in {L2_HINTS}, got {store_hint!r}")
    # round 15 (Cake W3): only the TMA-store epilogue carries the L2 eviction-policy operand  [Cake instance_key]
    store_hint = store_hint if epi == "tma" else "none"
    pd, sh = int(pd), int(sh)
    if pd not in PD_CHOICES:
        raise ValueError(
            f"pd must be one of {PD_CHOICES} (0 off, 1 chunk-pipelined drain, 2 cross-tile pipelined drain, 3 both chunks in flight), got {pd}"
        )
    if pd and (
        cta_rows != 128
        or out_t
        or cols not in (64, 128)
        or "no_tmem" in diag
        or box_rows
        or pf
        or "a_mcast" in diag
    ):
        # round 15 (Cake W1): the pipelined drain is built for the 128-row single-pass family's row-major epilogues
        # (64- / 128-column warp slices = one / two 64-column TMEM chunks)  [Cake instance_key]
        raise ValueError(
            f"pd needs cta_rows=128, a row-major output, BLOCK_N 128 / 256 and no no_tmem / a_mcast / pf / box_rows "
            f"(got cta_rows={cta_rows}, out_t={out_t}, block_n={block_n}, diag={diag}, pf={pf}, box_rows={box_rows})"
        )
    if pd in (1, 3) and cols != 128:
        raise ValueError(
            f"pd={pd} overlaps the two chunks of a 128-column warp slice; BLOCK_N={block_n} has one chunk per slice (use pd=2)"
        )
    if sh < 0 or sh > 4294967295:
        raise ValueError(f"sh (suspendTimeHint ns) must fit a u32, got {sh}")
    # round 19 (Cake W2): the slab-path forms exist only on the synchronised stream-K programs  [Cake instance_key]
    sk_slab = int(sk_slab) if sk_sync else 0
    if sk_slab not in SK_SLAB_CHOICES:
        raise ValueError(
            f"sk_slab must be one of {SK_SLAB_CHOICES} (0 off, 1 early arrival, 2 + bulk slab read, 3 + fp32 output as the slab), got {sk_slab}"
        )
    if sk_slab == 3 and (not out_f32 or (epi == "reg" and not out_t)):
        raise ValueError(
            f"sk_slab=3 (the fp32 output as the slab) needs an fp32 output through the transposed register path or the "
            f"TMA-store epilogue (got out_f32={out_f32}, out_t={out_t}, epi={epi!r})"
        )
    # the eight warp slabs of the bulk slab read stage in the mainloop stage ring (the pair form's stage: the slab
    # forms exist only on the synchronised programs, which the single-CTA form excludes)  [Cake instance_key]
    slab_bytes = EPI_WARPS * cols * 128
    if sk_slab == 2 and slab_bytes > stages * (
        cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz)
    ):
        raise ValueError(
            f"sk_slab=2 stages the eight warp slabs ({slab_bytes} B) in the mainloop stage ring ({stages} stages); too small"
        )
    if stt and (
        out_f32
        or not out_t
        or epi != "reg"
        or cta_rows != 128
        or cta1
        or ovl
        or ovr
        or park
        or htail
        or int(pd)
        or cols % 32
        or cols > 256
        or sk_slab not in (0, 1)
    ):
        # round 27 (Cake lane 7, W-3 A1): the transposed stmatrix / TMA-store staging is built for the bf16 transposed
        # REGISTER epilogue of the 128-row single-pass pair family (stmatrix is .b16: the fp32 rows are excluded; the
        # 16x256b drain covers whole 32-column blocks; the box outer extent is <= 256; the ovl / ovr / park / htail / pd
        # forms drain and store through their own chunk paths; the bulk slab read (sk_slab 2) copies a column-group
        # prefix that the 16x256b order no longer is; sk_slab 3 is fp32-only; the single-CTA form's handoff is
        # unmeasured)  [Cake instance_key]
        raise ValueError(
            f"stt (transposed stmatrix / TMA-store staging) needs a bf16 transposed output through the single-pass "
            f"register epilogue of the 128-row pair family (no ovl / ovr / park / htail / pd) with whole 32-column warp "
            f"slices of at most 256 columns and sk_slab 0 / 1 (got out_f32={out_f32}, out_t={out_t}, epi={epi!r}, "
            f"cta_rows={cta_rows}, cta1={cta1}, ovl={ovl}, ovr={ovr}, park={park}, htail={htail}, pd={pd}, "
            f"cols={cols}, sk_slab={sk_slab})"
        )
    stt_slots = EPI_WARPS * cols * 64
    if stt and stt_slots > stages * (
        cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg)
    ):
        raise ValueError(
            f"stt stages the eight warp slots ({stt_slots} B) in the dead mainloop stage ring ({stages} stages); too small"
        )
    stt_pool = (
        stages * (cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg))
        + staging_bytes(slots)
        + stt_bytes(stt)
        + WORK_STAGES * 16
    )
    if stt and stt_pool > SMEM_OPT_IN:
        # round 27 (Cake lane 7): an stt program must fit the 227 KiB opt-in - the oversized shared-memory mode is not
        # shippable  [Cake instance_key]
        raise ValueError(
            f"stt needs the dynamic SMEM pool inside the {SMEM_OPT_IN} B opt-in (got {stt_pool} B at {stages} stages)"
        )
    if cta1 and (
        cta_rows != 128
        or ovl
        or ovr
        or park
        or htail
        or sk_exact
        or sk_sync
        or batch_group
        or "a_mcast" in diag
    ):
        # round 19 (Cake W3): the single-CTA form is the 128-row single-pass family without the pair-level tail
        # policies / probes (and without the tall family's overlapped epilogues ``ovl`` / ``ovr``)  [Cake instance_key]
        raise ValueError(
            f"cta1 needs cta_rows=128 and no ovl / ovr / park / htail / sk_exact / sk_sync / batch_group / a_mcast (got "
            f"cta_rows={cta_rows}, ovl={ovl}, ovr={ovr}, park={park}, htail={htail}, sk_exact={sk_exact}, sk_sync={sk_sync}, "
            f"batch_group={batch_group}, diag={diag})"
        )
    pro = int(pro)
    if pro not in range(16):
        raise ValueError(
            f"pro must be a bitmask in 0..15 (1 = no startup rendezvous before the collective TMEM alloc, 2 = relaxed "
            f"arrive on the post-init cluster sync, 4 = the MMA warp allocates TMEM, 8 = CTA-local exit: no cleanup "
            f"barrier.cluster, role-drained rings + leader->follower tmem_free handoff + warp-1 collective dealloc); got {pro}"
        )
    if pro and cta1:
        # round 33 (Cake unit 2d): the pair prologue / cleanup sync form needs the cluster pair  [Cake instance_key]
        raise ValueError(
            "pro (pair prologue / cleanup sync form) needs the cluster pair (cta1=False)"
        )
    if pro & 8 and any(d != "trace" for d in diag):
        # round 33 (Cake unit 2e): the CTA-local exit's remote-op inventory covers the production protocol (+ the trace
        # probe's local stamps only); the planner never emits diag, so this term is inert - kept for message parity
        # [Cake instance_key]
        raise ValueError(
            f"pro bit 8 (CTA-local exit) admits no diag probe except 'trace' (got diag={diag})"
        )
    sk_first = bool(sk_first)
    if sk_first and not sk_exact:
        raise ValueError(
            "sk_first (tail-first persistent order) needs the exact p-way split (sk_exact)"
        )
    if sk_first and (pd == 2 or sk_slab == 2 or stt or sk_sync or htail):
        raise ValueError(
            f"sk_first keeps the plain item walk - no pd 2, sk_slab 2, stt, sk_sync or htail (got pd={pd}, "
            f"sk_slab={sk_slab}, stt={stt}, sk_sync={sk_sync}, htail={htail})"
        )
    lw = bool(lw)
    if lw and (cta1 or not (pro & 4)):
        raise ValueError(
            f"lw (the load warp out of the pair prelude) needs the cluster pair with the MMA-warp allocator (pro bit 4); "
            f"got cta1={cta1}, pro={pro}"
        )
    mg = bool(mg)
    if mg and (
        batch_group
        or cgrp
        or htail
        or sk_exact
        or sk_first
        or sk_sync
        or sk_slab
        or stt
    ):
        raise ValueError(
            "mg (whole-tile tile resolve) is written for the plain row-grouped raster and the whole-tile walk: no "
            f"batch_group, cgrp, htail or stream-K form (got batch_group={batch_group}, cgrp={cgrp}, htail={htail}, "
            f"sk_exact={bool(sk_exact)}, sk_first={sk_first}, sk_sync={sk_sync}, sk_slab={sk_slab}, stt={stt})"
        )
    sw = int(sw)
    if sw < 0 or (sw and (epi != "tma" or not slots or int(pd) != 2 or out_f32)):
        # the paced output stores are written for the bf16 TMA-store pd-2 drain (the kernel's refusal)
        raise ValueError(
            f"sw (paced output stores) is a step count >= 0 and needs the bf16 TMA-store pd 2 drain (slots > 0, pd 2, "
            f"no out_f32); got sw={sw}, epi={epi!r}, slots={slots}, pd={pd}, out_f32={out_f32}"
        )
    form = str(form or "")
    if form and (
        form not in W8_FORMS
        or a_mn
        or not b_mn
        or out_f32
        or out_t
        or cta1
        or (block_n != 256 and not (form in _W8_N128_FORMS and block_n == 128))
        or cta_rows != 128
        or epi != "tma"
        or slots != 1
        or b_swz != 128
        or box_rows
        or sk_exact
        or sk_first
        or sk_sync
        or sk_slab
        or htail
        or cgrp
        or batch_group == 1
        or stt
        or ovl
        or ovr
        or park
        or pd
        or pro
        or lw
        or mg
        or sw
        or stages != W8_STAGES
        or hints != ("none", "none")
        or pf
        or sh
        or store_hint not in ("none", "evict_last")
        or (store_hint != "none" and batch_group <= 1)
        or (form in _W8_START_FORMS and (batch_group or store_hint != "none"))
        or store_ef
        or quad_store
        or f32_v8
    ):
        # the eight-warp CTA form is written for the kn bf16 row-major 128 x 256 pair tile (``w8relf`` also 128 x 128)
        # with the one-slot TMA-store epilogue, the form's six stages, whole tiles and none of the pair programs'
        # pipeline knobs (the kernel's frozen configurations); the head-interleaved batch raster (``batch_group``
        # G > 1) and, with it, the ``evict_last`` store hint are the kernel's own knobs (the batched MLA programs),
        # which the start path of ``w8rel`` / ``w8relf`` excludes
        raise ValueError(
            f"form (the eight-warp CTA form) must be one of {W8_FORMS} on the kn bf16 row-major 128 x 256 pair tile "
            f"(also 128 x 128 for {_W8_N128_FORMS}) "
            f"with the one-slot TMA-store epilogue, the form's six stages, plain hints and no tail, raster (but a "
            f"batch group G > 1 outside {_W8_START_FORMS}) or pipeline knob (but the evict_last store hint with a "
            f"batch group); got "
            f"form={form!r}, a_mn={a_mn}, b_mn={b_mn}, out_f32={out_f32}, out_t={out_t}, "
            f"cta1={cta1}, block_n={block_n}, cta_rows={cta_rows}, epi={epi!r}, slots={slots}, b_swz={b_swz}, "
            f"box_rows={box_rows}, stages={stages}, hints={hints}, pd={pd}, pro={pro}, lw={lw}, mg={mg}, sw={sw}, "
            f"sk_exact={bool(sk_exact)}, sk_first={sk_first}, sk_sync={sk_sync}, sk_slab={sk_slab}, htail={htail}, "
            f"cgrp={cgrp}, batch_group={batch_group}, stt={stt}, ovl={ovl}, ovr={ovr}, park={park}, pf={pf}, "
            f"sh={sh}, store_hint={store_hint!r}, store_ef={store_ef}, quad_store={quad_store}, f32_v8={f32_v8}"
        )
    block_k = int(block_k)
    if block_k not in BLOCK_K_CHOICES or (
        block_k != BLOCK_K and not (form in _W8_N128_FORMS and block_n == 128)
    ):
        # the 32-element K step is the 128-column start-path program's alone (the kernel's frozen configurations)
        raise ValueError(
            f"block_k (the K step) is one of {BLOCK_K_CHOICES}; {BLOCK_K_CHOICES[1]} needs the eight-warp form "
            f"{_W8_N128_FORMS} on the 128-column tile; got block_k={block_k}, form={form!r}, block_n={block_n}"
        )
    kser = int(kser)
    if kser < 0 or (
        kser
        and (
            cta1 or sk_exact or sk_first or sk_sync or stt or mg or batch_group or form
        )
    ):
        # the serpentine K walk is written for the pair programs' plain item walk (the kernel's refusal)
        raise ValueError(
            f"kser (serpentine K walk) is a pair count >= 0 and needs the cluster pair on the plain item walk (no cta1, "
            f"sk_exact, sk_first, sk_sync, stt, mg, batch_group or form); got kser={kser}, cta1={cta1}, "
            f"sk_exact={bool(sk_exact)}, sk_first={sk_first}, sk_sync={sk_sync}, stt={stt}, mg={mg}, "
            f"batch_group={batch_group}, form={form!r}"
        )
    promote_blocks = int(promote_blocks)
    if promote_blocks < 0:
        raise ValueError(
            f"promote_blocks is 0 (no promotion) or the K blocks of a promoted chunk, got {promote_blocks}"
        )
    key = (
        a_mn,
        b_mn,
        out_f32,
        out_t,
        block_n,
        stages,
        diag,
        epi,
        slots,
        box_rows,
        cta_rows,
        pf,
        hints,
        f32_v8,
        quad_store,
        park,
        ovl,
        htail,
        b_swz,
        bool(sk_exact),
        batch_group,
        store_ef,
        store_hint,
        pd,
        sh,
        sk_sync,
        sk_slab,
        cta1,
        cgrp,
        ovr,
        stt,
        pro,
        sk_first,
        lw,
        mg,
        sw,
        form,
        kser,
        promote_blocks,
    )
    if block_k != BLOCK_K:
        key += (block_k,)  # field 40: only a key with another K step carries it
    if promote_blocks:
        refusal = promote_refusal(key)
        if refusal:
            raise ValueError(
                f"promote_blocks (promoted chunks of {promote_blocks} K blocks) needs an nn CTA-pair pro program the "
                f"kernel admits; got {refusal}"
            )
    return key


def promote_refusal(key: tuple) -> str:
    """Why the program of instance key ``key`` takes no promoted chunks (``promote_blocks``; the kernel's refusal):
    a K-major operand, the tall 256-row tile, a tile width other than 128 / 192 / 256, no pair prologue form (``pro``
    0), a non-default stage count, the single-CTA form, the ``ovl`` / ``ovr`` / ``htail`` / ``pd`` drains, the
    load-warp prelude, the whole-tile tile resolve, the quad-transposed stores, a batch or column-grouped raster, the
    serpentine K walk, the tail-first order, or a slab path other than ``sk_slab`` 2 on the 128-row 192-wide
    64-byte-panel tile and on the 256-wide tile of an FP32 or row-major store; ``""`` where it takes them.  These
    drains read only the thread's own accumulator columns of one buffer per segment, which the chunk sum is written
    back over."""
    (
        a_mn,
        b_mn,
        out_f32,
        out_t,
        block_n,
        stages,
        _diag,
        _epi,
        slots,
        _box_rows,
        cta_rows,
        _pf,
        _hints,
        _f32_v8,
        quad_store,
        _park,
        ovl,
        htail,
        b_swz,
        _sk_exact,
        batch_group,
        _store_ef,
        _store_hint,
        pd,
        _sh,
        _sk_sync,
        sk_slab,
        cta1,
        cgrp,
        ovr,
        stt,
        pro,
        sk_first,
        lw,
        mg,
        _sw,
        _form,
        kser,
        _promote_blocks,
    ) = key[:INSTANCE_KEY_FIELDS]
    cg = 1 if cta1 else CTA_GROUP
    if not (a_mn and b_mn):
        return "a K-major operand (the promoted programs are the nn weight gradients')"
    if cta_rows == 256 or block_n not in (128, 192, 256):
        return f"cta_rows {cta_rows}, block_n {block_n} (64- or 128-row tiles of 128, 192 or 256 columns)"
    if not pro or stages != default_stages(
        slots, cta_rows, block_n, b_mn, b_swz, cg, stt
    ):
        return f"pro {pro}, {stages} stages (a pair prologue form at the default stage count)"
    if (
        cta1
        or ovl
        or ovr
        or htail
        or pd
        or lw
        or mg
        or quad_store
        or batch_group
        or cgrp
        or kser
        or sk_first
    ):
        return (
            f"cta1={cta1}, ovl={ovl}, ovr={ovr}, htail={htail}, pd={pd}, lw={lw}, mg={mg}, quad_store={quad_store}, "
            f"batch_group={batch_group}, cgrp={cgrp}, kser={kser}, sk_first={sk_first} (none of these forms)"
        )
    if sk_slab not in (0, 2) or (
        sk_slab == 2
        and (
            cta_rows != 128
            or (block_n, b_swz) not in ((192, 64), (256, 128))
            or (block_n == 256 and out_t and not out_f32)
        )
    ):
        return (
            f"sk_slab {sk_slab} on cta_rows {cta_rows}, block_n {block_n}, b_swz {b_swz} (sk_slab 2 only on the 128-row "
            "192-wide 64-byte-panel tile and the 256-wide one of an FP32 or row-major store)"
        )
    return ""


def instance_symbol(key: tuple) -> str:
    """Kernel symbol / registry template of an instance key (``dense_proj_det_gemm_<a><b>_n<N>``
    followed by ``_m256`` for tall tiles / ``_m64`` for the 64-row Layout-B family, ``_pf<n>`` for a
    prefetch distance, ``_h<a><b>`` for non-default (A, B) eviction hints (first letters, e.g.
    ``_hen`` = A evict_first / B none), ``_f32``, ``_v8`` for the 256-bit fp32 register stores, ``_t``,
    ``_bz<bytes>`` right after the tile family for a narrow MN-major B panel (round 13), ``_pk`` / ``_ov`` / ``_ht`` for
    the parked / overlapped / half-height-tail tall epilogues (round 13), ``_<epi><slots>`` for the TMA-store
    epilogue, ``_s<stages>`` for a non-default stage count, ``_box<rows>`` and a trailing ``_skx`` for the exact
    p-way stream-K split (round 13), ``_sks`` for the synchronised stream-K tail (round 18) followed by ``_sb<n>`` for
    its slab path (round 19), ``_so<f|l|n>`` after the batch-raster term for the TMA-store L2 eviction policy,
    then ``_pd<n>`` / ``_sh<n>`` for the pipelined TMEM drain / suspend-time hint (round 15), ``_c1`` for
    the single-CTA form (round 19; its default stage count is the single-CTA fit), ``_cg`` for the column-grouped
    raster (round 21) and ``_or`` for the tall epilogue released by row half (round 24); ``_stt`` directly
    after ``_sks`` for the transposed stmatrix / TMA-store staging (round 27); a trailing ``_pro<n>`` after ``_or`` for
    the pair prologue / cleanup sync form (round 33; absent when 0 - the single-CTA ``_c1`` programs never carry it);
    the round-2 shapes tranche adds ``_skf`` directly after ``_skx`` (``sk_first``) and ``_lw`` between ``_or`` and
    ``_pro`` (``lw``); ``_mg`` directly after ``_lw`` for the whole-tile tile resolve (``mg``); ``_sw<n>`` directly
    after ``_mg`` for the paced output stores (``sw``, absent when 0); ``_ks<P>`` directly after ``_sw<n>`` for the
    serpentine K walk (``kser``, absent when 0); ``_kc<KC>`` directly after ``_ks<P>`` and before ``_pro<n>`` for the
    promoted chunks (``promote_blocks``, absent when 0); ``_w8r`` / ``_w8rs`` / ``_w8`` / ``_w8rel`` / ``_w8relf``
    directly after ``_pro<n>`` and before the diag tokens for the eight-warp CTA form (``form``, absent when "";
    a form key's default stage count is the ring's six, :data:`W8_STAGES`); ``_k<K>`` directly after ``_n<N>`` for a
    K step other than 64 (``block_k``, field 40: the 128-column ``w8relf`` program's 32-element steps).
    [Cake ``instance_symbol``]"""
    (
        a_mn,
        b_mn,
        out_f32,
        out_t,
        block_n,
        stages,
        diag,
        epi,
        slots,
        box_rows,
        cta_rows,
        pf,
        hints,
        f32_v8,
        quad_store,
        park,
        ovl,
        htail,
        b_swz,
        sk_exact,
        batch_group,
        store_ef,
        store_hint,
        pd,
        sh,
        sk_sync,
        sk_slab,
        cta1,
        cgrp,
        ovr,
        stt,
        pro,
        sk_first,
        lw,
        mg,
        sw,
        form,
        kser,
        promote_blocks,
    ) = key[:INSTANCE_KEY_FIELDS]
    block_k = instance_block_k(key)
    cg = 1 if cta1 else CTA_GROUP
    so = {
        "evict_first": "f",
        "evict_last": "l",
        "evict_normal": "n",
        "default": "d",
    }.get(store_hint, "")
    return (
        "dense_proj_det_gemm_"
        + ("n" if a_mn else "k")
        + ("n" if b_mn else "k")
        + f"_n{block_n}"
        + (f"_k{block_k}" if block_k != BLOCK_K else "")
        + ("_m256" if cta_rows == 256 else "_m64" if cta_rows == 64 else "")
        + (f"_bz{b_swz}" if b_swz != 128 else "")
        + (f"_pf{pf}" if pf else "")
        + (f"_h{hints[0][0]}{hints[1][0]}" if hints != ("none", "none") else "")
        + ("_f32" if out_f32 else "")
        + ("_v8" if f32_v8 else "")
        + ("_ef" if store_ef else "")
        + ("_q" if quad_store else "")
        + ("_pk" if park else "")
        + ("_ov" if ovl else "")
        + ("_ht" if htail else "")
        + ("_t" if out_t else "")
        + (f"_{epi}{slots}" if epi != "reg" else "")
        + (
            f"_s{stages}"
            if stages
            != (
                W8_STAGES
                if form
                else default_stages(slots, cta_rows, block_n, b_mn, b_swz, cg, stt)
            )
            else ""
        )
        + (f"_box{box_rows}" if box_rows else "")
        + ("_skx" if sk_exact else "")
        + ("_skf" if sk_first else "")
        + ("_sks" if sk_sync else "")
        + ("_stt" if stt else "")
        + (f"_sb{sk_slab}" if sk_slab else "")
        + (
            "_bf"
            if batch_group == 1
            else f"_bg{batch_group}"
            if batch_group > 1
            else ""
        )
        + (f"_so{so}" if so else "")
        + (f"_pd{pd}" if pd else "")
        + (f"_sh{sh}" if sh else "")
        + ("_c1" if cta1 else "")
        + ("_cg" if cgrp else "")
        + ("_or" if ovr else "")
        + ("_lw" if lw else "")
        + ("_mg" if mg else "")
        + (f"_sw{sw}" if sw else "")
        + (f"_ks{kser}" if kser else "")
        + (f"_kc{promote_blocks}" if promote_blocks else "")
        + (f"_pro{pro}" if pro else "")
        + (f"_{form}" if form else "")
        + "".join(f"_{d}" for d in diag)
    )


def swap_small_m(L: int, M: int, N: int, transposed_out: bool) -> bool:
    """Batched rows with M <= 256 and N >= 2 M (the MLA weight gradients: heads = batch, M = head
    dim, N = latent dim) are computed as the transposed GEMM ``out^T[l] = B[l]^T A[l]^T`` with the
    transposed store.  [Cake ``swap_small_m`` L1164-L1167]"""
    return L > 1 and not transposed_out and M <= 256 and N >= 2 * M


#: Package-only: the architectures on which an unbatched weight gradient with a narrow output (``N <= 64`` columns
#: against ``4096 <= M <= 8192`` rows, ``K >= 256``) and no measured rule was measured faster swapped (see
#: :func:`_single_wgrad_swap`).
_NARROW_N_SWAP_ARCHES = ("sm_100a", "sm_103a")

#: Package-only: the smallest token count ``K`` at which a keyed unbatched weight gradient runs swapped, by
#: ``(arch, M, N, out_f32)`` of the generic call (see :func:`_single_wgrad_swap`).  On sm_107a (212 SMs) the 6144 -> 576
#: bf16 weight gradient ran the swapped plan 0.91x at T 2048 and 4096 and 1.18x to 1.46x from T 6144 to 40961.  The
#: four sm_107a bf16 weight gradients that version 0.3.22 keys (4096 -> 16 / 96, 6144 -> 48 / 64, ``ROW_RULES``)
#: swapped from T 8193 on in versions 0.3.22 and 0.3.23 (an entry of T 8193 each: their rules' token floor in version
#: 0.3.22, where the swapped call was measured).  Since version 0.3.24 they have no entry: the generic call swaps where
#: the swapped key's measured rule at ``K`` on the part (``ROW_RULE_MIN_T``, ``ROW_RULE_RANGES_BY_SM``) is not empty,
#: and so runs the weight-gradient entry's plan, program and bits there; where that rule is empty it kept the direct
#: plan without a rule in versions 0.3.24 - 0.3.25 (below T 960 in version 0.3.25, below T 1024 in version 0.3.24; and
#: 4096 -> 16 at T 3073 - 4095 off the 212-SM part in version 0.3.25, at T 2049 - 4095 (T 2049 - 3071 on the 212-SM
#: part) in version 0.3.24); since version 0.3.26 it follows the weight-gradient entry there too, from the row's
#: ``_SINGLE_WGRAD_FOLLOW_MIN_K`` token count on.  Post-ship medians of the direct plan over the swapped one at the
#: timed changed cells of version 0.3.24 were 2.012 to 3.352 (eager launches) and 2.004 to 3.357 (CUDA graph replays)
#: on the 212-SM part and 1.785 to 3.205 and 1.772 to 3.131 on the 216-SM part.
_SINGLE_WGRAD_SWAP_MIN_K = {
    ("sm_107a", 576, 6144, False): 6144,
    # no entry for the four sm_107a bf16 rows of version 0.3.22 (4096 -> 16 / 96, 6144 -> 48 / 64): the post-ship A/B
    # floor of each was T 1024 on both parts (version 0.3.24); since version 0.3.25 the call swaps from T 960 on the
    # 64-row whole tiles (post-ship A/B of the generic call: medians of version 0.3.24 over version 0.3.25 at the timed
    # changed cells were 1.037 to 4.206 (eager launches) and 1.038 to 4.256 (CUDA graph replays) on the 212-SM part
    # (2.058 to 4.206 and 1.998 to 4.256 at the 13 cells where it now swaps); on the 216-SM part the A/B did not
    # complete (cells whose plan is the same in both versions read faster in version 0.3.25 in their paired comparison,
    # at one cell in the first timing and at four in a re-time (one of them moved between two timing states), so the
    # control check fixed in advance voided the result on that part), and its observed medians were 1.021 to 4.033 and
    # 1.024 to 3.972 (1.953 to 4.033 and 1.911 to 3.972 at the 14 cells where it now swaps), with no changed cell
    # slower); since version 0.3.26 also where the rule is empty, from the row's ``_SINGLE_WGRAD_FOLLOW_MIN_K``
    # entry on
}

#: Package-only: the smallest token count ``K`` from which the generic call of a row that ``ROW_RULES`` keys in the
#: swapped orientation alone swaps at every token count, also where the swapped key's measured rule at ``K`` on the
#: part is empty (the weight-gradient entry then plans with the architecture defaults alone), by ``(arch, M, N,
#: out_f32)`` of the generic call (see :func:`_single_wgrad_swap`; an entry of ``_SINGLE_WGRAD_SWAP_MIN_K`` still
#: applies).  Since version 0.3.26 the four sm_107a bf16 weight gradients of version 0.3.22 (4096 -> 16 / 96,
#: 6144 -> 48 / 64) follow the entry from T 1 on: below T 960 on every sm_107a part (below T 128 since version 0.3.27,
#: whose ``ROW_RULE_MIN_T`` tiers give these rows a rule from T 128 on) and, off the 212-SM part, 4096 -> 16 at
#: T 3073 - 4095, where the entry runs ``dense_proj_det_gemm_nn_n128_t_pro7`` (whole 128-row tiles below T 960 in
#: version 0.3.26, below T 128 since version 0.3.27; a two-way K split at T 3073 - 4095).  Timed against version 0.3.25
#: (the direct plan) on sm_107a, medians of 21 paired rounds with L2 flushed; speedup of the entry's plan over the
#: direct plan at the timed changed cells (15 token counts per row below T 960; on the 216-SM part also 4096 -> 16 at
#: T 3073, 3328, 3584, 3840, 4095): 1.19 - 2.15 (eager), 1.19 - 2.13 (CUDA graph) on the 212-SM part, 1.20 - 2.49
#: (eager), 1.21 - 2.70 (CUDA graph) on the 216-SM part.
_SINGLE_WGRAD_FOLLOW_MIN_K = {
    ("sm_107a", 16, 4096, False): 1,
    ("sm_107a", 96, 4096, False): 1,
    ("sm_107a", 48, 6144, False): 1,
    ("sm_107a", 64, 6144, False): 1,
}


def _single_wgrad_swap(
    arch: str,
    M: int,
    N: int,
    K: int,
    a_mn: bool,
    b_mn: bool,
    out_f32: bool,
    *,
    sm_count: Optional[int] = None,
) -> bool:
    """Package-only: an unbatched (``L == 1``) call with both operands MN-major and no transposed store is the
    weight gradient ``G.T @ X`` of :func:`cake_dense_gemm_det_train_projection_wgrad` with ``G = A.T`` (``M`` output
    rows) and ``X = B`` (``N`` columns).  It runs the swapped ``X.T @ G`` with the transposed store

    * where that entry swaps by a measured rule: ``ROW_RULES`` keys the row in the swapped orientation alone
      (:func:`_orientation_rules`; the narrow 16- to 128-row, 512-row and 576-row weight gradients), ``K`` is at
      least the row's ``_SINGLE_WGRAD_SWAP_MIN_K`` entry, if any, and (since version 0.3.24) the row's measured rule
      at ``K`` on the part is not empty (:func:`_measured_row_rule` of the swapped key with ``sm_count``: the token
      tiers of ``ROW_RULE_MIN_T`` and the part ranges of ``ROW_RULE_RANGES_BY_SM``; without ``sm_count`` the part
      tables are not read, the planner always passes its part) or, since version 0.3.26, ``K`` is at least the row's
      ``_SINGLE_WGRAD_FOLLOW_MIN_K`` entry (from it on at every token count: where the rule is empty the entry plans
      with the architecture defaults alone).  The plan, its program and its bits are then the weight-gradient
      entry's;
    * where no rule keys the row in either orientation, the output is narrow (``N <= 64``: the ``[d_in, d_out]``
      weight gradient ``X.T @ G`` of a narrow projection), ``4096 <= M <= 8192`` and ``K >= 256``, on
      ``_NARROW_N_SWAP_ARCHES``: the direct plan pads the ``N`` columns into one 224- / 256-column tile, the swapped
      plan puts them in a 64-row tile.  Above 8192 rows the swapped plan measured slower at most token counts, and
      below 256 tokens the call is launch-bound (flat to 2 % slower swapped).

    Every other unbatched call keeps the caller's orientation."""
    if not (a_mn and b_mn):
        return False
    swapped_rule, direct_rule = _orientation_rules(arch, M, N, out_f32)
    if swapped_rule or direct_rule:
        min_k = _SINGLE_WGRAD_SWAP_MIN_K.get((arch, M, N, out_f32), 0)
        follow_k = _SINGLE_WGRAD_FOLLOW_MIN_K.get((arch, M, N, out_f32))
        # the row's follow floor, else the weight-gradient entry's measured rule at this token count on this part
        # (tiers, part ranges): the swapped GEMM has N = M (d_out), K = K (T) and M = N (d_in)
        return (
            swapped_rule
            and not direct_rule
            and min_k <= K
            and (
                (follow_k is not None and follow_k <= K)
                or bool(
                    _measured_row_rule(
                        arch, True, True, out_f32, True, False, M, K, N, sm_count=sm_count
                    )
                )
            )
        )
    return arch in _NARROW_N_SWAP_ARCHES and N <= 64 and 4096 <= M <= 8192 and K >= 256


# fmt: off
# Per-architecture, per-row knob overrides measured in Cake round 2, keyed by the static row identity
# (arch, A MN-major, B MN-major, fp32 output, transposed output, batched, N, K, M); a ragged rule leaves the
# token count out (M = None for forward / input gradients, K = None for weight gradients), and an exact key names
# one token count (the 216-SM tranche's sm_107a T-exact weight-gradient rules at K = 16231 and 16384; the GB
# T-exact forward / input-gradient rules of sm_100a and sm_103a at M = 16172, 16231 or 16384; the GB T-exact
# weight-gradient rules of sm_100a and sm_103a at K = 16172, 16231 or 16384).
# The round-2 to round-33 rules mirror the former launcher's row table; the shapes tranche below them (sm_100a and
# sm_103a) was measured with this package's planner and extends it per architecture (three round-33 sm_100a keys were
# re-measured and re-valued there).  An architecture-wide default (``ARCH_RULE_DEFAULTS``, round 33) sits under every rule
# of that architecture (``row_rule`` merges it; a row rule overrides).  Three sm_103a input-gradient rows (4096 -> 512,
# 5120 -> 512 and the kv_a 6144 -> 576) carry ``mg`` with ``sk: False``: the whole-tile tile resolve (``_mg`` programs)
# measured faster than the row's program above the pair spread at both T on GB300, and those programs run whole tiles
# only (the planner drops a rule's ``mg`` where the plan carries stream-K units and refuses a caller-forced one there).
# Two sm_103a input-gradient rows (5120 -> 512 and 5120 -> 1024) carry ``sw`` 25: the paced output stores (``_sw25``
# programs) measured faster than the row's program above the pair spread at both T on GB300; the knob needs the bf16
# TMA-store pd-2 drain (the planner drops a rule's ``sw`` where the planned program is not that drain or a caller forces
# the single-CTA form, and refuses a caller-forced one there).  Six sm_103a input-gradient rows (4096 -> 48,
# 4096 -> 512, 4096 -> 1024, 5120 -> 512, 5120 -> 1024 and the kv_a 6144 -> 576) carry ``form``: the eight-warp CTA form
# of the same GEMM (``_w8r`` / ``_w8rs`` programs) measured faster than the row's program above the pair spread at both
# T on GB300; the knobs after it are the row's program where the form yields (the 4096 -> 48 rule carries none: the
# architecture default program runs there), and the planner drops a rule's form where the plan carries stream-K units,
# the single-CTA form, another tile or output class, the register epilogue or a caller-forced pipeline knob, and refuses
# a caller-forced form there.  Eight sm_100a input-gradient rows carry ``form``: 4096 -> 1024 and 5120 -> 1024 the
# knob-free eight-warp program (``_w8``), 4096 -> 512, the kv_a 6144 -> 576 and the batched MLA value-projection row
# (N 512, K 256) the eight-warp program with the early accumulator release and the loader-issued scheduling queries
# (``_w8rs``), and 5120 -> 512, 4096 -> 64 and 5120 -> 64
# the one with the early accumulator release (``_w8r``).  Each measured faster than the row's program above the pair
# spread at both T on GB200, except 4096 -> 64, 5120 -> 64 and the kv_a, where the form measured at least as fast as the
# row's program there; the knobs after the form are the row's program where the form yields, and the raster group is the
# row's measured launch parameter.  The two sm_100a 4096 -> 48 input-gradient T-exact keys (T 16231 and 16384, the GB
# T-exact rules) carry ``w8r`` too: the eight-warp program with the early accumulator release, whole tiles through the
# one-slot TMA store (the ragged 4096 -> 48 rule keeps its half-tail program at every other T).  The batched MLA
# absorbed-query forward (N 512, K 192) carries ``w8rs`` too.  Both MLA rows add the head-interleaved batch raster
# (``_bg<G>``): the forward over all 64 heads (``_bg64``), the value-projection row over 8 heads with the evict_last
# output stores (``_bg8_sol``), raster group 2 for the forward; each measured faster than cuBLASLt's batched kernel at
# the three T on GB200 and GB300.  sm_103a reads the forward's rule through ``RULE_PARENT``; its value-projection key
# follows below (raster group 2).  The batched MLA value-projection forward (N 256, K 512) adds the same raster over 8 heads
# (``_bg8``) to its quad-store register program at the training T alone (T-exact keys at T 16172 / 16231 / 16384, the
# ragged rule's knobs + ``batch_group``): the same output bits, measured faster than cuBLASLt's batched kernel and the
# program without the raster at those T on GB200 and GB300 (sm_103a through ``RULE_PARENT``); from T 1 to 4095 the
# raster measured slower in the sustained frame, so the ragged rule keeps the program without it there.
ROW_RULES: dict[tuple, dict] = {
    ('sm_100a', False, False, False, False, False, 32, 6144, None): {"cta1": True},
    ('sm_100a', False, False, False, False, False, 128, 6144, None): {"cta1": True},
    ('sm_100a', False, False, False, False, False, 576, 6144, None): {"hints": ('evict_first', 'evict_last'), "stages": 8, "pro": 15},
    ('sm_100a', False, False, False, False, False, 2048, 6144, None): {"group_m": 8, "cgrp": True},
    ('sm_100a', False, False, False, False, False, 6144, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True},
    ('sm_100a', False, False, False, False, False, 6144, 16384, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    ('sm_100a', False, False, False, False, False, 16384, 2048, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_100a', False, False, False, False, True, 256, 512, None): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b'},
    ('sm_100a', False, False, False, False, True, 256, 512, 16172): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b', "batch_group": 8},
    ('sm_100a', False, False, False, False, True, 256, 512, 16231): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b', "batch_group": 8},
    ('sm_100a', False, False, False, False, True, 256, 512, 16384): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b', "batch_group": 8},
    ('sm_100a', False, True, False, False, False, 2048, 6144, None): {"group_m": 8, "pro": 0},
    ('sm_100a', False, True, False, False, False, 2048, 16384, None): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True},
    ('sm_100a', False, True, False, False, False, 6144, 32, None): {"slots": 2, "pd": 2},
    ('sm_100a', False, False, False, False, False, 4096, 2048, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 2048, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 6144, 128, None): {"slots": 2, "pd": 2, "pro": 15},
    # the kv_a input gradient (6144 -> 576): the eight-warp program with the early accumulator release and the
    # loader-issued scheduling queries, the row's ten-warp knobs kept as its program where the form yields; the raster
    # group of 12 row tiles is the row's measured launch parameter
    ('sm_100a', False, True, False, False, False, 6144, 576, None): {"form": 'w8rs', "group_m": 12, "slots": 1, "epi": 'tma', "pd": 2, "pro": 15, "sk": False},
    ('sm_100a', False, True, False, False, False, 6144, 2048, None): {"group_m": 8},
    ('sm_100a', False, True, False, False, False, 6144, 12288, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    # the MLA absorbed-query forward (batched 512 -> 192): the eight-warp program with the early accumulator release and
    # the loader-issued scheduling queries, the head-interleaved raster over all 64 heads, raster group 2
    ('sm_100a', False, True, False, False, True, 512, 192, None): {"form": 'w8rs', "batch_group": 64, "group_m": 2},
    # the MLA value-projection input gradient (batched 512 -> 256): the same program with the head-interleaved raster of
    # 8 heads, the evict_last output stores and 256-byte L2 promotion, raster group 4
    ('sm_100a', False, True, False, False, True, 512, 256, None): {"form": 'w8rs', "batch_group": 8, "store_hint": "evict_last", "promo": 'l2_256b', "group_m": 4},
    ('sm_100a', False, True, True, False, False, 2048, 4096, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 2048, 6144, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 2048, 16384, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 6144, 32, None): {"block_n": 128, "slots": 2, "stages": 4, "cta1": True},
    ('sm_100a', False, True, True, False, False, 6144, 128, None): {"block_n": 128, "group_m": 2},
    ('sm_100a', False, True, True, False, False, 6144, 576, None): {"group_m": 8, "promo": 'l2_256b'},
    ('sm_100a', False, True, True, False, False, 6144, 2048, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 6144, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 12288, 6144, None): {"epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 16384, 6144, None): {"epi": 'reg', "f32_v8": True},
    ('sm_100a', True, True, False, False, False, 2048, None, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 2048, None, 6144): {"group_m": 4, "sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 2048, None, 16384): {"cta_rows": 256, "ovl": True, "htail": True},
    ('sm_100a', True, True, False, False, False, 6144, None, 2048): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 16384, None, 6144): {"cta_rows": 256, "group_m": 8, "htail": True},
    ('sm_100a', True, True, False, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, False, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, False, True, False, 576, None, 6144): {"hints": ('evict_first', 'evict_first'), "epi": 'tma'},
    ('sm_100a', True, True, False, True, True, 192, None, 512): {"cta_rows": 256, "epi": 'tma'},
    ('sm_100a', True, True, False, True, True, 256, None, 512): {"cta_rows": 256, "epi": 'tma'},
    ('sm_100a', True, True, True, False, False, 2048, None, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 3},
    ('sm_100a', True, True, True, False, False, 2048, None, 6144): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, True, False, False, 2048, None, 16384): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_100a', True, True, True, False, False, 6144, None, 2048): {"group_m": 32, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_100a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_100a', True, True, True, False, False, 16384, None, 6144): {"cta_rows": 256, "htail": True},
    ('sm_100a', True, True, True, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, True, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, True, True, False, 576, None, 6144): {"hints": ('evict_first', 'evict_first'), "epi": 'tma'},
    ('sm_107a', False, False, False, False, False, 32, 6144, None): {"promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, False, 576, 6144, None): {"cta_rows": 256, "hints": ('evict_first', 'none'), "stages": 5, "pro": 15},
    ('sm_107a', False, False, False, False, False, 6144, 2048, None): {"htail": True, "pro": 15},
    ('sm_107a', False, False, False, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3, "ovr": True, "cgrp": True, "group_m": 12},
    ('sm_107a', False, False, False, False, False, 12288, 6144, None): {"cta_rows": 256, "htail": True, "ovr": True, "cgrp": True, "group_m": 12},
    ('sm_107a', False, False, False, False, False, 6144, 16384, None): {"cta_rows": 256, "htail": True, "ovr": True, "cgrp": True, "group_m": 8},
    ('sm_107a', False, False, False, False, False, 16384, 2048, None): {"group_m": 32, "pro": 15},
    ('sm_107a', False, False, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, True, 256, 512, None): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, True, 512, 256, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, False, 6144, 32, None): {"cta_rows": 128, "slots": 2, "pd": 3, "group_m": 2, "cta1": True},
    ('sm_107a', False, True, False, False, False, 6144, 128, None): {"cta_rows": 128, "slots": 2, "pd": 2, "pro": 15},
    ('sm_107a', False, True, False, False, False, 6144, 576, None): {"group_m": 8, "epi": 'reg', "quad_store": True, "pd": 1},
    ('sm_107a', False, True, False, False, False, 6144, 2048, None): {"htail": True, "pro": 15},
    ('sm_107a', False, True, False, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3, "ovr": True, "cgrp": True, "group_m": 12},
    ('sm_107a', False, True, False, False, False, 12288, 6144, None): {"hints": ('none', 'evict_first')},
    ('sm_107a', False, True, False, False, False, 16384, 6144, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True, "pro": 15},
    ('sm_107a', False, True, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, True, 256, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, True, 512, 192, None): {"batch_group": 8, "store_hint": "evict_last"},
    ('sm_107a', False, True, False, False, True, 512, 256, None): {"promo": 'l2_256b', "batch_group": 8},
    ('sm_107a', False, True, True, False, False, 2048, 4096, None): {"epi": 'reg', "f32_v8": True, "stages": 9, "store_ef": True, "pro": 15},
    ('sm_107a', False, True, True, False, False, 2048, 6144, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 2048, 16384, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 6144, 32, None): {"block_n": 128, "slots": 1, "group_m": 2, "cta1": True},
    ('sm_107a', False, True, True, False, False, 6144, 128, None): {"block_n": 128, "cta1": True, "group_m": 2, "pd": 2},
    ('sm_107a', False, True, True, False, False, 6144, 576, None): {"group_m": 8, "promo": 'l2_256b', "pro": 15},
    ('sm_107a', False, True, True, False, False, 6144, 2048, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "htail": True},
    ('sm_107a', False, True, True, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', False, True, True, False, False, 12288, 6144, None): {"cta_rows": 256},
    ('sm_107a', False, True, True, False, False, 16384, 6144, None): {"cta_rows": 256},
    ('sm_107a', True, True, False, False, False, 2048, None, 4096): {"block_n": 160, "cta_rows": 256, "group_m": 4, "sk_parts": 2, "b_swz": 64, "pro": 15},
    ('sm_107a', True, True, False, False, False, 2048, None, 6144): {"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2},
    ('sm_107a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "sk_parts": 2},
    ('sm_107a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "htail": True},
    ('sm_107a', True, True, False, False, False, 16384, None, 6144): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', True, True, False, False, False, 6144, None, 2048): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    # the indexer_hw BF16 weight gradient: since version 0.3.22 the 64-row exact two-way split of the tail tiles (one of
    # the five rows of that version, below), with the L2 hint ``('evict_first', 'none')`` since version 0.3.23; its
    # ``ROW_RULE_MIN_T`` tiers keep the rule it had before, ``{"block_n": 128, "sk_exact": 3}``, below T 128 (T 960 in
    # versions 0.3.25 - 0.3.26, T 2048 in versions 0.3.23 - 0.3.24, T 8193 in version 0.3.22) and, since version
    # 0.3.25, plan the 64-row whole tiles at T 128 - 3584 (T 960 - 3584 in versions 0.3.25 - 0.3.26; the split from
    # T 3585 on); the 216-SM part plans the whole tiles at T 128 - 3584 (T 960 - 3584 in versions 0.3.25 - 0.3.26), the
    # split at T 3585 - 15360 and that rule from T 15361 on (``ROW_RULE_RANGES_BY_SM``)
    ('sm_107a', True, True, False, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', True, True, False, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', True, True, False, True, False, 576, None, 6144): {"group_m": 8, "hints": ('evict_first', 'evict_first'), "b_swz": 64, "sk_sync": True, "sk_sync_m": 38, "stt": True, "pro": 15},
    ('sm_107a', True, True, False, True, True, 192, None, 512): {"cta_rows": 256, "hints": ('evict_first', 'evict_first'), "epi": 'tma', "b_swz": 64, "stages": 6},
    ('sm_107a', True, True, False, True, True, 256, None, 512): {"cta_rows": 256, "hints": ('evict_first', 'evict_first'), "epi": 'tma', "stages": 6},
    ('sm_107a', True, True, True, False, False, 2048, None, 4096): {"block_n": 160, "cta_rows": 256, "sk_parts": 3, "b_swz": 64, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', True, True, True, False, False, 2048, None, 6144): {"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_107a', True, True, True, False, False, 2048, None, 16384): {"group_m": 8, "epi": 'reg', "f32_v8": True, "pro": 15},
    ('sm_107a', True, True, True, False, False, 6144, None, 2048): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_107a', False, False, False, False, False, 2048, 6144, None): {"group_m": 8, "cgrp": True, "pro": 15},
    ('sm_107a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256, "htail": True},
    ('sm_107a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_107a', True, True, True, False, False, 16384, None, 6144): {"cta_rows": 256},
    ('sm_107a', True, True, True, True, False, 32, None, 6144): {"block_n": 128, "sk_exact": 3},
    ('sm_107a', True, True, True, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', False, False, False, False, False, 128, 6144, None): {"pro": 15},
    ('sm_107a', False, False, False, False, False, 4096, 2048, None): {"pro": 15},
    ('sm_107a', False, True, False, False, False, 2048, 4096, None): {"pro": 15},
    ('sm_107a', True, True, False, False, False, 2048, None, 16384): {"pro": 15},
    ('sm_107a', True, True, True, True, False, 576, None, 6144): {"group_m": 8, "hints": ('evict_first', 'evict_first'), "b_swz": 64, "sk_sync": True, "sk_sync_m": 38, "sk_slab": 2},
    ('sm_107a', True, True, False, True, False, 128, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15},
    ('sm_107a', True, True, True, True, False, 128, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15},
    ('sm_107a', True, True, False, True, False, 128, None, 5120): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', True, True, True, True, False, 128, None, 5120): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', True, True, False, True, False, 128, None, 8192): {"block_n": 128, "sk_sync": True},
    ('sm_107a', True, True, True, True, False, 128, None, 8192): {"block_n": 128, "sk_sync": True},
    ('sm_107a', False, True, False, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_107a', False, True, False, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_107a', False, True, False, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "cta1": True, "pd": 3},
    ('sm_107a', False, True, False, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_107a', False, True, False, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_107a', False, True, False, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True},
    ('sm_107a', False, True, False, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "cta1": True, "pd": 3},
    ('sm_107a', False, True, False, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_107a', False, True, False, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_107a', True, True, False, True, False, 32, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15},
    ('sm_107a', True, True, False, True, False, 48, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15},
    ('sm_107a', True, True, False, True, False, 64, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15},
    ('sm_107a', True, True, False, True, False, 32, None, 5120): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', True, True, False, True, False, 48, None, 5120): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', True, True, False, True, False, 64, None, 5120): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    # ---- version 0.3.22: five BF16 weight gradients of sm_107a take the 64-row exact split of the tail tiles from
    #      T 8193 on (their ``ROW_RULE_MIN_T`` tiers keep the plan before below it): 4096 -> 16 and 4096 -> 96 the rule
    #      of the 4096 -> 32 / 48 / 64 rows above (three parts), 6144 -> 48, 6144 -> 64 and the indexer_hw 6144 -> 32
    #      (its key above) the rule of 6144 -> 128 (two parts).  The plan: 128 x 128 pair tiles of 64-row CTAs, every
    #      one of the 32 (4096 rows) or 48 (6144 rows) tiles a tail tile split into its parts of ceil(k_blocks / parts)
    #      K blocks, 96 stream-K units in one wave of the CTA pairs;
    #      ``dense_proj_det_gemm_nn_n128_m64_hen_t_skx_pro15``, or its promoted twin ``..._kc64_pro15`` where a part
    #      holds more than 64 K blocks (4096 rows from T 12289, 6144 rows from T 8193).  The plan before: no rule (the
    #      128-row two-way split of 16 / 24 tiles, 32 / 48 units: ``..._nn_n128_hen_t_kc64_pro15`` through the registry
    #      fallback from the 128-row hint gate on, below it ``..._nn_n128_t_pro7``), and on indexer_hw the 128-row
    #      three-way split of 24 tiles (72 units, ``..._nn_n128_hen_t_skx(_kc64)_pro7`` from T 10322, below it the
    #      no-rule plan through the rule fallback).  Measured on VR200 (212 SMs; one node, driver 620.43) in both
    #      packages against the plan before and a second prepared copy of it in the same 21 paired rounds per cell and
    #      mode, with cold L2 in eager launches and CUDA graph replays, at 14 token counts per row from T 8193 to 131072
    #      (both sides of T 12288 / 12289, of 10321 / 10322 on the 6144 rows and of 15179 / 15180 on the 4096 rows) and
    #      at T 8192, where the plan stays: faster at every changed token count in both modes and with one bit pattern
    #      on every run, and within the naive BF16 tier on 21 seeds at each changed token count; medians of the plan
    #      before over the new plan 1.171 to 1.523 on 4096 -> 16, 1.200 to 1.631 on 4096 -> 96, 1.064 to 1.517 on
    #      6144 -> 48, 1.071 to 1.504 on 6144 -> 64 and 1.025 to 1.440 on indexer_hw.  The bits against
    #      the plan before: the same on 6144 -> 48 and 6144 -> 64 at an even k_blocks = ceil(T / 64) from T 10322 on,
    #      where the plan before gives every tile the same two K halves (its auto two-way split: unit u is tile u // 2,
    #      K half u % 2) in the same promoted chunks of 64 K blocks (``..._nn_n128_hen_t_kc64_pro15`` from the 128-row
    #      hint gate, T 10322 on a 126-MiB L2); other bits at an odd k_blocks (a unit of the auto split crosses into
    #      the next tile), at T 8193 - 10321 (``..._nn_n128_t_pro7``, no registered promoted twin: one accumulator per
    #      K half) and on 4096 -> 16 / 96 and indexer_hw at every token count from T 8193 on (three parts against two
    #      on the 4096 rows, two against three on indexer_hw from T 10322, the no-rule plan without promoted chunks
    #      below it).  The 216-SM part (108 pairs) plans the same programs and units on 4096 -> 16 / 96 and
    #      6144 -> 48 / 64: measured the same way on one R200 board (driver 620.70.06), faster at every changed token
    #      count in both packages and modes, medians of the plan before over the new plan 1.204 to 1.481 on 4096 -> 16,
    #      1.197 to 1.530 on 4096 -> 96, 1.074 to 1.329 on 6144 -> 48 and 1.095 to 1.352 on 6144 -> 64; there indexer_hw
    #      keeps its plan before, and its bits, at every token count (measured slower than that plan at T 16172, 32768
    #      and 65536: ``ROW_RULE_RANGES_BY_SM``).
    #      The 208- and 224-SM parts and the token counts above 131072 are not measured and plan these rules (one wave
    #      of 96 units on 104 and 112 pairs; longer parts in more promoted chunks).  The unbatched generic call of
    #      4096 -> 16 / 96 and 6144 -> 48 / 64 with both operands MN-major runs swapped from T 8193 on since this
    #      version (``_single_wgrad_swap``: a key in the swapped orientation alone, ``_SINGLE_WGRAD_SWAP_MIN_K``), the
    #      plan of the weight-gradient entry; below T 8193 it keeps its orientation ----
    # ---- version 0.3.23: the five rows of version 0.3.22 split their tail tiles exactly on 64-row tiles from lower
    #      token counts (``ROW_RULE_MIN_T``): 4096 -> 16 the two-way split ``{"block_n": 128, "cta_rows": 64,
    #      "sk_exact": 2, "pro": 15}`` at T 1024 - 2048, the plan without a rule at T 2049 - 4095 and its rule (three
    #      parts) from T 4096 on (on the 212-SM part from T 3072: ``ROW_RULE_RANGES_BY_SM``); 4096 -> 96 the two-way
    #      split at T 1024 - 7168 and its rule from T 7169 on; 6144 -> 48 and 6144 -> 64 their rule from T 1024 on;
    #      indexer_hw its rule from T 2048 on (on the 216-SM part at T 1024 - 15360).  Below T 1024 every row plans as
    #      before.  These rules carry the L2 hint ``('evict_first', 'none')``: the 64-row exact-split programs are
    #      registered with it alone, and below the 64-row hint gate (T 7820 on the 4096 rows, T 5267 on the 6144 rows on
    #      a 126-MiB L2) a rule without it names an unregistered program and falls back to the plan before; from the
    #      gate on the hint changes no plan.  A caller that forces another tile height (``cta_rows``) drops the rule's
    #      family knobs (``block_n``, ``pro``, ...) but keeps its ``sk_exact`` and its hint in the plan it names; below
    #      the 128-row gate (T 15180 / 10322) a caller-forced 128-row call names the unregistered
    #      ``dense_proj_det_gemm_nn_n128_hen_t_skx`` (``pro`` 0) and falls back to the plan without a rule, the plan and
    #      bits of version 0.3.22.  Measured in this package on VR200
    #      (212 SMs) and one R200 board (216 SMs) against the plan of version 0.3.22 and a second prepared copy of it,
    #      cold L2, eager launches and CUDA graph replays, 21 paired rounds per cell and mode, at T 1024, 2048, 3072,
    #      4096, 4097, 5120, 6144, 7168, 8065 and 8192 (indexer_hw on the 216-SM part also at T 8193 - 16171): every new
    #      plan was faster than the plan of version 0.3.22 beyond the A/A band in both modes at each measured token
    #      count of its band, on each measured part that plans it, within the naive BF16 tier on five seeds and with one
    #      bit pattern run to run.  The two-way split takes 4096 -> 16 at T 1024 - 2048, where the rule is not admitted
    #      (T 1024: parts of 6 K blocks) or not faster on both parts (T 2048: flat in CUDA graph replays on the 212-SM
    #      part); there the medians over the 21 paired rounds of the rule's time over the split's (eager / CUDA graph)
    #      were 1.023 / 1.026 on the 212-SM part and 0.993 / 0.997 on the 216-SM part.  It takes 4096 -> 96 at
    #      T 1024 - 7168 (the rule is not admitted at T 1024): those medians of the rule over the split were
    #      1.068 / 1.052, 1.092 / 1.087, 1.000 / 0.997, 1.031 / 1.034, 0.993 / 0.991, 1.006 / 1.004 and 0.984 / 0.984 on
    #      the 212-SM part and 1.041 / 1.041, 1.099 / 1.097, 1.012 / 1.012, 1.022 / 1.022, 1.020 / 1.018, 1.059 / 1.066
    #      and 1.004 / 1.001 on the 216-SM part at T 2048, 3072, 4096, 4097, 5120, 6144 and 7168, and over the band
    #      (T 1024 included) the geometric mean of the medians of the plan before over the split was 1.4 % (212 SMs) and
    #      2.6 % (216 SMs) above that over the rule; at T 8065 the medians of the rule over the split were 0.971 / 0.967
    #      (212 SMs) and 0.991 / 0.990 (216 SMs), so the row plans its rule from T 7169.  The 212-SM range of 4096 -> 16
    #      rests on T 3072, where the rule was faster than the plan before on that part; on the 216-SM part it was flat
    #      in eager launches and slower in CUDA graph replays there.
    #      Post-ship medians of the plan before over the new plan at the timed changed cells were 1.008 to 1.720 (eager
    #      launches) and 1.008 to 1.723 (CUDA graph replays) on the 212-SM part and 1.018 to 1.497 and 1.018 to 1.615 on
    #      the 216-SM part.  Against version 0.3.22 the bits are the same where the two-way split runs at an even number
    #      of 64-token K blocks up to T 8192 (the plan before split every 128-row tile into the same two K halves,
    #      without promoted chunks: 65 of the 110 measured changed cells had the digest of the plan before) and differ
    #      elsewhere (an odd number of K blocks, three or four parts, promoted chunks), as the post-ship A/B found on
    #      every timed cell of both parts.  The 208- and 224-SM parts are not measured and plan these tiers.  The
    #      generic call of 4096 -> 16 / 96 and 6144 -> 48 / 64 keeps T 8193 as its swap floor
    #      (``_SINGLE_WGRAD_SWAP_MIN_K``); the indexer_hw generic call swaps at every token count and runs these
    #      plans ----
    # ---- version 0.3.24: the generic call of 4096 -> 16 / 96 and 6144 -> 48 / 64 has no ``_SINGLE_WGRAD_SWAP_MIN_K``
    #      entry (the post-ship A/B floor of each was T 1024 on both parts): it swaps, and runs the plan, program and
    #      bits of the weight-gradient entry, wherever these rules and tiers give the swapped key a rule on the part
    #      (``_single_wgrad_swap``), and keeps the direct plan without a rule where they give none.  Post-ship medians
    #      of the direct plan over the swapped one at the timed changed cells were 2.012 to 3.352 (eager launches) and
    #      2.004 to 3.357 (CUDA graph replays) on the 212-SM part and 1.785 to 3.205 and 1.772 to 3.131 on the 216-SM
    #      part ----
    # ---- version 0.3.25: the five rows of version 0.3.22 plan whole 128 x 128 pair tiles of 64-row CTAs without a K
    #      split, ``{"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}``
    #      (``dense_proj_det_gemm_nn_n128_m64_hen_t_pro15``, and its promoted twin ``..._m64_hen_t_kc64_pro15`` where a
    #      tile holds more than 64 K blocks: 6144 -> 64 at T 4097 - 5120), as a middle tier of ``ROW_RULE_MIN_T``:
    #      4096 -> 16 at T 960 - 3072, 4096 -> 96, 6144 -> 48 and indexer_hw at T 960 - 3584 and 6144 -> 64 at
    #      T 960 - 5120, on every sm_107a part (the 212-SM range of 4096 -> 16 starts at T 3073 and the 216-SM
    #      indexer_hw ranges at T 960: ``ROW_RULE_RANGES_BY_SM``).  Below T 960 and above these bands every row plans as
    #      in version 0.3.24.  The rule carries the L2 hint: the 64-row whole-tile programs are registered with it
    #      alone, and every band ends below the 64-row hint gate (T 7820 on the 4096 rows, T 5267 on the 6144 rows),
    #      where a rule without it would name an unregistered program.  Inside the bands the rule's ``sk`` holds for a
    #      caller that leaves ``sk`` at ``"auto"``: a caller-forced ``sk_exact``, ``sk_parts`` or ``sk_sync`` plans the
    #      whole tiles there, as on the other rules that pin ``sk: False`` (a caller split there needs the rule arm of
    #      its own tables).
    #      Measured before shipping in this package on VR200 (212 SMs) and one R200 board (216 SMs) against the plan of
    #      version 0.3.24 and a second prepared copy of it, cold L2, eager launches and CUDA graph replays, 21 paired
    #      rounds per cell and mode, at T 960, 1024, 1536, 2048, 3072 and 4096 and (with the promoted twin) at
    #      T 5120 - 8192: at every measured token count of each band as first set (to T 4096 on 4096 -> 96, 6144 -> 48
    #      and indexer_hw) the new plan was faster than the plan of version 0.3.24 beyond the A/A band in both modes on
    #      both parts, with one bit pattern run to run and within the naive BF16 tier on five seeds; medians of that
    #      plan over the new one were 1.038 to 1.361 / 1.043 to 1.354 (eager / CUDA graph) on the 212-SM part and 1.028
    #      to 1.362 / 1.031 to 1.369 on the 216-SM part.  The new plan was not timed at T 961 - 1023 (T 1023 was timed
    #      with the plan of version 0.3.24 and the 128-row whole tiles alone); these token counts lie between the
    #      measured T 960 and 1024, where it was faster on both parts.  Each band as first set ended at the last
    #      measured token count before one where the new plan was not faster in both modes on both parts (medians eager
    #      / CUDA graph, 212-SM part; 216-SM part): 4096 -> 16 at T 4096 (0.997 / 0.997, flat; 0.978 slower / 0.990
    #      flat), 4096 -> 96 at T 5120 (0.991 / 0.996, flat; 0.981 / 0.986, flat), 6144 -> 48 at T 5120 (0.979 / 0.978,
    #      slower; 0.991 / 0.991, flat), indexer_hw at T 5120 (0.999 / 1.001, flat; 0.970 slower / 0.980 flat) and
    #      6144 -> 64 at T 6144 (1.023 faster / 1.018 flat; 0.989 / 0.997, flat).  In the post-ship A/B on the 212-SM
    #      part, at T 4096 on 4096 -> 96, 6144 -> 48 and indexer_hw an entry was not faster than the plan of version
    #      0.3.24 in either launch mode (the weight-gradient entry of both packages and the generic call of the
    #      non-deterministic package on 4096 -> 96, the weight-gradient entry of the non-deterministic package on
    #      6144 -> 48 and of this package on indexer_hw), while this package's entries were faster at every measured
    #      token count of T 960 - 3584, so these three bands end at T 3584; the 216-SM part, where the whole tiles were
    #      faster at T 4096 before shipping, uses the same edge (one edge on every sm_107a part;
    #      its post-ship result: the A/B there did not complete, since cells whose plan is the same in both versions
    #      read faster in version 0.3.25 in their paired comparison, at one cell in the first timing and at four in a
    #      re-time (one of them moved between two timing states) and the control check fixed in advance voided the
    #      result on that part; the medians it observed, given below, show no changed cell slower than version 0.3.24).
    #      At T 3585 - 4096 these rows plan as in version 0.3.24, with its bits, and their generic call runs that plan.
    #      Post-ship medians of the plan before over the new plan at the timed changed cells: 1.035 to 1.571 (eager
    #      launches) and 1.035 to 1.582 (CUDA graph replays) on the 212-SM part and 1.023 to 1.552 and 1.020 to 1.570
    #      (observed in the incomplete A/B at the last timing of each cell, not judged; no changed cell was slower in
    #      any timing) on the 216-SM part.  Against version 0.3.24 the bits are the same at T 960
    #      (both plans sum a tile's K in one chain) and on the twin at an even number of K blocks (its two chunks are
    #      the two halves of the exact split before), except for an output whose two halves both sum to -0.0 (stored
    #      -0.0 now, +0.0 before), and differ elsewhere (one chain over a tile's K blocks instead of two or three
    #      parts).  The unbatched generic call of the five rows swaps at every token count of the bands
    #      (``_single_wgrad_swap``: the rule is not empty) and runs these plans there, now also at T 960 - 1023 and on
    #      4096 -> 16 at T 2049 - 3072 (T 2049 - 3071 on the 212-SM part), where its bits may change against version
    #      0.3.24 (it ran the direct plan there) ----
    # ---- version 0.3.26: these tables do not change.  The unbatched generic call of 4096 -> 16, 4096 -> 96,
    #      6144 -> 48 and 6144 -> 64 swaps at every token count (from T 1 on, ``_SINGLE_WGRAD_FOLLOW_MIN_K``), also
    #      where these rules and tiers give the swapped key no rule on the part (below T 960 on every sm_107a part,
    #      and 4096 -> 16 at T 3073 - 4095 off the 212-SM part), and there runs the
    #      weight-gradient entry's plan of the architecture defaults alone, ``dense_proj_det_gemm_nn_n128_t_pro7``
    #      (whole 128-row tiles below T 960, a two-way K split at T 3073 - 4095), and its bits ----
    # ---- version 0.3.27: these rules do not change.  Thirteen sm_107a weight gradients plan whole 128 x 128 pair tiles
    #      of 64-row CTAs without a K split from T 128 on, the rule of version 0.3.25 ``{"block_n": 128, "cta_rows":
    #      64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}``:
    #      * the five BF16 rows of version 0.3.25 (4096 -> 16, 4096 -> 96, 6144 -> 48, 6144 -> 64 and indexer_hw
    #        6144 -> 32) at T 128 - 959: the first bound of their ``ROW_RULE_MIN_T`` tiers moves from T 960 to T 128, so
    #        the rule runs at T 128 - 3072, 128 - 3584 (4096 -> 96, 6144 -> 48, indexer_hw) and 128 - 5120, one band
    #        with that of version 0.3.25 (the 216-SM indexer_hw range starts at T 128: ``ROW_RULE_RANGES_BY_SM``);
    #      * the BF16 rows 4096 -> 32 / 48 / 64 / 128, 5120 -> 48 / 64 / 128 and 8192 -> 128 at T 128 - 4096 (new
    #        ``ROW_RULE_MIN_T`` keys: below T 128 and from T 4097 on the row's rule above, as before);
    #      * the FP32 outputs of the same thirteen rows at T 128 - 4096 (4096 -> 16: T 128 - 3072) on the 212- and
    #        216-SM parts alone (``ROW_RULE_RANGES_BY_SM``: nine of these rows have no token-free (ragged) key here, and
    #        such a key would change the orientation of their generic call), where the rule names the FP32 program
    #        ``dense_proj_det_gemm_nn_n128_m64_hen_f32_t_pro15``.
    #      The BF16 program is ``dense_proj_det_gemm_nn_n128_m64_hen_t_pro15``.  At the token counts this version adds
    #      (T 128 - 959 on the five rows of version 0.3.25, T 128 - 4096 on the others) a tile holds at most 64 K blocks
    #      (no promoted chunk), and the 32 to 64 pair tiles of a row run in one wave.  The BF16 tiers apply on every
    #      sm_107a part (the 208- and 224-SM parts are not measured, as for every sm_107a tier since version 0.3.22);
    #      the 208- and 224-SM parts keep the FP32 plans of version 0.3.26.  Below T 128 and above these bands every row
    #      plans as in version 0.3.26.
    #      Measured before shipping in this package on VR200 (212 SMs) and one R200 board (216 SMs;
    #      its memory clock held 3186 MHz under load, below its 4752 MHz maximum, before and after shipping) against the
    #      plan of version 0.3.26 (the plan of version 0.3.25 at these cells) and a second prepared copy of it, cold L2,
    #      eager launches and CUDA graph replays, 21 paired rounds per cell and mode, at T 128, 256, 512, 768 and 959
    #      (the five BF16 rows of version 0.3.25) and at T 128, 256, 512, 768, 960, 1024, 1536, 2048, 3072, 3584 and
    #      4096 (the other BF16 rows and every FP32 output): at every measured token count of each band (254 cells per
    #      part) the new plan was faster than the plan of version 0.3.26 beyond the A/A band in both modes on both
    #      parts, with one bit pattern run to run and within the naive BF16 tier on five seeds; medians of that plan
    #      over the new one were 1.071 to 1.701 / 1.064 to 1.694 (eager / CUDA graph) on the 212-SM part and 1.100 to
    #      1.675 / 1.105 to 1.694 on the 216-SM part.  The FP32 band of 4096 -> 16 ends at T 3072: at T 3584 the new
    #      plan was flat on the 212-SM part (paired medians 0.992 / 0.994 against the second copy of the plan before,
    #      within the A/A band; not timed on the 216-SM part), and T 4096, faster on both parts, is a single measured
    #      point above it; T 3073 - 4096 keep the plan of version 0.3.26 there.  Post-ship medians of the plan before
    #      over the new plan at the timed changed cells: 1.070 to 1.715 / 1.064 to 1.709 (eager / CUDA graph; 482 cells,
    #      observed in an A/B that did not complete: cells whose plan is the same in both versions read up to 13 %
    #      (eager) and 14 % (CUDA graph) apart in either direction in their paired comparison, the most at T 4097, in
    #      the first timing and again in the one re-time, so the control check fixed in advance left the result
    #      unjudged; none of these cells was slower) on the 212-SM part and 1.087 to 1.644 / 1.082 to 1.651 (189 of its
    #      482 changed cells, observed in an A/B that was stopped part way, before any control cell was timed, so not
    #      judged; none was slower) on the 216-SM part.  Against version 0.3.26 the bits are the same where both plans
    #      sum each tile's K in one chain (at the measured cells T 128 - 960: the plan before ran whole 128-row tiles
    #      there) and differ where the plan before split each tile's K in two (at the measured cells from T 1024 on)
    #      (post-ship A/B: the same at 234 of the 234 timed changed cells at T 128 - 960 and different at 248 of the 248
    #      from T 961 on, on the 212-SM part; the same at 124 of the 124 timed changed cells at T 128 - 960 and
    #      different at 65 of the 65 from T 961 on, on the 216-SM part); run-to-run bits stay identical.  The unbatched
    #      generic call of the BF16 rows and of the four FP32 rows keyed here in the swapped orientation (4096 -> 128,
    #      5120 -> 128, 8192 -> 128 and indexer_hw) swaps at every token count of these bands (``_single_wgrad_swap``:
    #      the rule is not empty) and runs these plans; that of the nine other FP32 rows keeps its direct plan of
    #      version 0.3.26 (no token-free key in either orientation) ----
    ('sm_107a', True, True, False, True, False, 16, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', True, True, False, True, False, 96, None, 4096): {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', True, True, False, True, False, 48, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', True, True, False, True, False, 64, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', False, False, False, False, False, 512, 4096, None): {"htail": True, "cgrp": True, "group_m": 8, "pro": 15},
    ('sm_107a', False, False, False, False, False, 1024, 4096, None): {"htail": True, "pro": 15},
    ('sm_107a', False, False, False, False, False, 512, 5120, None): {"htail": True, "cgrp": True, "group_m": 8, "pro": 15},
    ('sm_107a', False, False, False, False, False, 1024, 5120, None): {"htail": True, "pro": 15},
    ('sm_107a', True, True, False, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', True, True, False, False, False, 4096, None, 1024): {"sk_sync": True, "sk_slab": 2},
    ('sm_107a', True, True, False, True, False, 512, None, 5120): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_107a', False, True, False, False, False, 4096, 512, None): {"epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "lw": True, "promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, False, 4096, 1024, None): {"epi": 'reg', "pro": 15},
    ('sm_107a', False, True, False, False, False, 5120, 512, None): {"epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "lw": True, "promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, False, 5120, 1024, None): {"epi": 'reg', "pro": 15},
    # ---- sm_107a: the 216-SM tranche - rules measured on sm_107a (216 SMs) against the plan the row had without them, every
    #      planned output bitwise equal to that plan's; the T-exact weight-gradient keys (K = 16231 or 16384 tokens) apply at
    #      that token count only, and each names the orientation the row plans without it (the orientation reads the
    #      ragged keys alone) ----
    ('sm_107a', False, True, True, False, False, 4096, 32, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, False, False, False, False, 32, 4096, None): {"block_n": 128, "cta_rows": 128, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', False, True, True, False, False, 4096, 48, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, False, False, False, False, 48, 4096, None): {"block_n": 128, "cta_rows": 128, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', False, True, True, False, False, 4096, 64, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, False, False, False, False, 64, 4096, None): {"block_n": 128, "cta_rows": 128, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', False, True, True, False, False, 4096, 128, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, False, False, False, False, 128, 4096, None): {"block_n": 128, "cta_rows": 128, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, False, False, False, 4096, None, 4096): {"block_n": 160, "cta_rows": 256, "b_swz": 64, "pro": 15},
    ('sm_107a', True, True, True, False, False, 4096, None, 4096): {"block_n": 160, "cta_rows": 256, "b_swz": 64, "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', False, True, True, False, False, 5120, 32, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, True, True, False, False, 5120, 48, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, True, True, False, False, 5120, 64, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, True, True, False, False, 5120, 128, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', False, False, False, False, False, 128, 5120, None): {"block_n": 128, "cta_rows": 128, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', False, True, True, False, False, 8192, 128, None): {"block_n": 128, "cta_rows": 128, "epi": 'tma', "slots": 1, "cta1": True},
    ('sm_107a', True, True, True, True, False, 32, 16231, 4096): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 48, 16231, 4096): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 64, 16231, 4096): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, False, False, 4096, 16231, 512): {"block_n": 192, "cta_rows": 128, "b_swz": 64, "hints": ('none', 'evict_first'), "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "pro": 15, "epi": 'reg', "pd": 0, "pf": 0},
    ('sm_107a', True, True, True, False, False, 4096, 16231, 1024): {"block_n": 160, "cta_rows": 256, "b_swz": 64, "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', True, True, True, True, False, 32, 16231, 5120): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 48, 16231, 5120): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 64, 16231, 5120): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, False, False, 5120, 16231, 512): {"block_n": 224, "cta_rows": 128, "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', True, True, True, True, False, 32, 16384, 4096): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 48, 16384, 4096): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 64, 16384, 4096): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, False, False, 4096, 16384, 512): {"block_n": 192, "cta_rows": 128, "b_swz": 64, "hints": ('none', 'evict_first'), "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "pro": 15, "epi": 'reg', "pd": 0, "pf": 0},
    ('sm_107a', True, True, True, False, False, 4096, 16384, 1024): {"block_n": 160, "cta_rows": 256, "b_swz": 64, "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', True, True, True, True, False, 32, 16384, 5120): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 48, 16384, 5120): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, True, False, 64, 16384, 5120): {"block_n": 128, "cta_rows": 64, "hints": ('evict_first', 'none'), "pro": 15},
    ('sm_107a', True, True, True, False, False, 5120, 16384, 512): {"block_n": 224, "cta_rows": 128, "f32_v8": True, "store_ef": True, "pro": 15},
    # ragged sm_107a weight-gradient keys (every token count without an exact key): the 4096 -> 512 / 4096 -> 1024 fp32
    # rows carry the rule of their T 16231 keys, the 5120 -> 1024 bf16 row the 224-column tile with the evict-first
    # B hint and the column-grouped raster
    ('sm_107a', True, True, True, False, False, 4096, None, 512): {"block_n": 192, "cta_rows": 128, "b_swz": 64, "hints": ('none', 'evict_first'), "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "pro": 15, "epi": 'reg', "pd": 0, "pf": 0},
    ('sm_107a', True, True, True, False, False, 4096, None, 1024): {"block_n": 160, "cta_rows": 256, "b_swz": 64, "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', True, True, False, False, False, 5120, None, 1024): {"block_n": 224, "cta_rows": 128, "hints": ('none', 'evict_first'), "cgrp": True, "pro": 15},
    # ---- sm_107a: the 212-SM tranche - ragged rules measured on R200 (212 SMs) at T 16172 - 16384 against the plan the
    #      row had without them, both in the cold-L2 frame and under sustained back-to-back load; the half-tail and
    #      column-grouped forward / input-gradient rows keep every output bitwise equal to that plan's; eight of the keys
    #      apply from a token floor on (``ROW_RULE_MIN_T``) ----
    # the 12288 -> 4096 input gradient: the half-tail program with the column-grouped raster (faster)
    ('sm_107a', False, True, False, False, False, 12288, 4096, None): {"cgrp": True, "htail": True, "pro": 15},
    # the 4096 -> 12288 input gradient: the half-tail program with the column-grouped raster (at least as fast)
    ('sm_107a', False, True, False, False, False, 4096, 12288, None): {"cgrp": True, "htail": True, "pro": 15},
    # the 4096 -> 4096 input gradient: the half-tail program with the column-grouped raster (faster)
    ('sm_107a', False, True, False, False, False, 4096, 4096, None): {"cgrp": True, "htail": True, "pro": 15},
    # the 5120 -> 5120 input gradient: the column-grouped raster in 12-column-tile groups (at least as fast)
    ('sm_107a', False, True, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    # the q_b 2048 x 16384 input gradient: the half-tail program with the column-grouped raster in the default groups,
    # in place of 8-column-tile groups (faster)
    ('sm_107a', False, True, False, False, False, 2048, 16384, None): {"cgrp": True, "htail": True, "pro": 15},
    # the shared_down 2048 -> 6144 input gradient: the half-tail program with the column-grouped raster (faster)
    ('sm_107a', False, True, False, False, False, 2048, 6144, None): {"cgrp": True, "htail": True, "pro": 15},
    # the 4096 -> 12288 forward: the half-tail program with the column-grouped raster (at least as fast)
    ('sm_107a', False, False, False, False, False, 12288, 4096, None): {"cgrp": True, "htail": True, "pro": 15},
    # the 12288 -> 4096 weight gradient (direct orientation): the tail tiles split into 3 K parts (faster)
    ('sm_107a', True, True, False, False, False, 12288, None, 4096): {"sk_parts": 3, "pro": 15},
    # the 4096 -> 12288 weight gradient (direct orientation): the tail tiles split into 3 K parts, 16-row raster
    # groups (faster)
    ('sm_107a', True, True, False, False, False, 4096, None, 12288): {"group_m": 16, "sk_parts": 3, "pro": 15},
    # the 5120 -> 5120 weight gradient (direct orientation): the synchronised stream-K program with the slab path and
    # a 26-step collector margin (faster)
    ('sm_107a', True, True, False, False, False, 5120, None, 5120): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    # the 5120 -> 5120 fp32 weight gradient (direct orientation): the synchronised stream-K program with the slab path,
    # a 38-step collector margin and the evict-first fp32 v8 register stores, in place of the whole-tile fp32 v8
    # program (faster)
    ('sm_107a', True, True, True, False, False, 5120, None, 5120): {"epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_sync_m": 38, "sk_slab": 2},
    # ---- sm_100a: the shapes tranche - forward / input gradient / weight gradient rows at K 4096 - 17408 and N 32 - 17408,
    #      measured with this package's planner against cuBLAS on GB200 at T 16172 - 16384 (see the test table
    #      MEASURED_SHAPE_ROWS); a row whose measured choice equals the planner's default carries no rule ----
    ('sm_100a', False, False, False, False, False, 32, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 32, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 48, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 48, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 64, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 64, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 128, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 128, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 128, 8192, None): {"pro": 15},
    ('sm_100a', False, False, False, False, False, 1024, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 1024, 5120, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, False, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_100a', False, False, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, False, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_100a', False, True, False, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 4096, 64, None): {"form": 'w8r', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 8, "sk": False},
    ('sm_100a', False, True, False, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_100a', False, True, False, False, False, 4096, 512, None): {"form": 'w8rs', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 12, "sk": False},
    ('sm_100a', False, True, False, False, False, 4096, 1024, None): {"form": 'w8', "pd": 2, "pro": 15, "group_m": 8, "sk": False},
    ('sm_100a', False, True, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_100a', False, True, False, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 5120, 64, None): {"form": 'w8r', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 8, "sk": False},
    ('sm_100a', False, True, False, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True},
    ('sm_100a', False, True, False, False, False, 5120, 512, None): {"form": 'w8r', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 8, "sk": False},
    ('sm_100a', False, True, False, False, False, 5120, 1024, None): {"form": 'w8', "pd": 2, "pro": 15, "group_m": 8, "sk": False},
    ('sm_100a', False, True, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_100a', False, True, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovr": True, "htail": True},
    ('sm_100a', False, True, False, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, True, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', True, True, False, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 32, None, 5120): {"pro": 15},
    ('sm_100a', True, True, False, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 48, None, 5120): {"pro": 15},
    ('sm_100a', True, True, False, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    # the 5120 -> 64 / 128 weight gradients from T 32768 on: the synchronised stream-K program of the sm_103a rule of
    # the rows (its promoted twin at these T), measured on B200 and GB200; below T 32768 the row keeps the 64-row
    # program (ROW_RULE_MIN_T), its promoted twin where a segment holds more than K1_PROMOTE_MIN_SEGMENT K blocks
    ('sm_100a', True, True, False, True, False, 64, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_100a', True, True, False, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 128, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_100a', True, True, False, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 32, None, 5120): {"pro": 15},
    ('sm_100a', True, True, True, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 48, None, 5120): {"pro": 15},
    ('sm_100a', True, True, True, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 64, None, 5120): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 128, None, 5120): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    # ``wave_yield``: the knobs the 4096 x 4096 weight gradient plans with where the exact-wave gate drops its 224-column
    # tile (148 SMs), from T 8192 on (ROW_RULE_MIN_T); ``sk_exact_fit``: the CTA pair count on which an exact split
    # whose units exceed the pairs takes max(2, pairs // tail) parts (see the planner); both measured on B200 and B300
    # at T 8192 - 32768
    ('sm_100a', True, True, False, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "block_n": 224, "wave_yield": {"sk_sync": True, "sk_slab": 2, "sk_sync_m": 32}},
    ('sm_100a', True, True, False, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True, "sk_exact_fit": 74},
    ('sm_100a', True, True, False, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 39, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_100a', True, True, False, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True, "group_m": 8},
    ('sm_100a', True, True, False, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 32},
    ('sm_100a', True, True, False, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True, "sk_exact_fit": 74},
    ('sm_100a', True, True, False, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_100a', True, True, True, False, False, 4096, None, 1024): {"slots": 1, "epi": 'tma', "pro": 15, "cgrp": True},
    ('sm_100a', True, True, True, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_100a', True, True, True, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_100a', True, True, True, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 26, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_100a', True, True, True, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True, "group_m": 8},
    ('sm_100a', True, True, True, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_100a', True, True, True, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_100a', True, True, True, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    # ---- sm_100a: the round-2 rules - the rows of the shapes tranche whose kernel-level round 2 selected another
    #      program or launch parameter, measured on GB200 at T 16231 / 16384 against cuBLAS (rows not listed keep the
    #      tranche's rule above) ----
    ('sm_100a', True, True, False, False, False, 4096, None, 1024): {"block_n": 224, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True},
    # ---- sm_100a: the GB T-exact rules - forward / input-gradient rows whose program measured faster at one token count
    #      (M = T), each key applying at that T alone with the per-tile K partitions of the row's plan without it (every
    #      output bitwise equal to that plan's), measured on GB200 against that plan: the 4096 -> 48 input gradient (the
    #      eight-warp program with the early accumulator release; the ragged rule's half-tail program elsewhere), the
    #      4096 -> 512 forward, the 6144 x 2048 input gradient (q_a / shared_gate_up) and the 2048 -> 6144 shared_down
    #      forward (the half-tail column-grouped program) ----
    ('sm_100a', False, True, False, False, False, 4096, 48, 16231): {"block_n": 256, "cta_rows": 128, "epi": 'tma', "form": 'w8r', "pro": 0, "slots": 1},
    ('sm_100a', False, True, False, False, False, 4096, 48, 16384): {"block_n": 256, "cta_rows": 128, "epi": 'tma', "form": 'w8r', "pro": 0, "slots": 1},
    ('sm_100a', False, False, False, False, False, 512, 4096, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_100a', False, False, False, False, False, 512, 4096, 16384): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_100a', False, False, False, False, False, 6144, 2048, 16172): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_100a', False, False, False, False, False, 6144, 2048, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_100a', False, True, False, False, False, 6144, 2048, 16172): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15, "group_m": 16},
    ('sm_100a', False, True, False, False, False, 6144, 2048, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15, "group_m": 16},
    # ---- sm_100a: the GB T-exact rules of round 2 - the q_b 2048 x 16384 input gradient (the ragged rule's overlapped
    #      half-tail program with the column-grouped raster: 8-column-tile groups at T 16172, 16-column-tile groups at T 16231) and the
    #      17408 -> 5120 weight gradient in its direct orientation (the half-tail program through the one-slot TMA store,
    #      12-row raster groups, at T 16231 / 16384), each with the per-tile K partitions of the row's plan without it,
    #      measured on GB200 against that plan ----
    ('sm_100a', False, True, False, False, False, 2048, 16384, 16172): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, True, False, False, False, 2048, 16384, 16231): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_100a', True, True, False, False, False, 17408, 16231, 5120): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_100a', True, True, False, False, False, 17408, 16384, 5120): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    # ---- sm_100a: the GB T-exact rules of round 3 - the indexer_q 2048 -> 4096 weight gradient at T 16172 / 16231
    #      in its direct orientation: the ragged rule's synchronised two-partial stream-K program with its
    #      bf16 tiles through the one-slot TMA store, with the per-tile K partitions of the row's plan without it,
    #      measured on GB200 against that plan ----
    ('sm_100a', True, True, False, False, False, 2048, 16172, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2, "epi": 'tma', "slots": 1},
    ('sm_100a', True, True, False, False, False, 2048, 16231, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2, "epi": 'tma', "slots": 1},
    # ---- sm_100a: the GB T-exact rules of round 4 - the 5120 -> 17408 input gradient at T 16231 / 16384 (the
    #      overlapped half-tail program with the column-grouped raster in 12-column-tile groups, in place of the ragged
    #      rule's row-half program), with the bf16 tiles through the one-slot TMA store and the per-tile K partitions of
    #      the row's plan without it, measured on GB200 against that plan ----
    ('sm_100a', False, True, False, False, False, 5120, 17408, 16231): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 12, "epi": 'tma', "slots": 1},
    ('sm_100a', False, True, False, False, False, 5120, 17408, 16384): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 12, "epi": 'tma', "slots": 1},
    # ---- sm_100a: the GB T-exact rules of round 5 - the 12288 -> 4096 forward at T 16231 / 16384 (the ragged rule's
    #      overlapped half-tail program with the column-grouped raster in 8-column-tile groups), with the bf16 tiles
    #      through the one-slot TMA store and the per-tile K partitions of the row's plan without them, measured on
    #      GB200 against that plan ----
    ('sm_100a', False, False, False, False, False, 4096, 12288, 16231): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8, "epi": 'tma', "slots": 1},
    ('sm_100a', False, False, False, False, False, 4096, 12288, 16384): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8, "epi": 'tma', "slots": 1},
    # ---- sm_100a: the GB T-exact rules of round 6 - the 17408 -> 5120 forward at T 16231 / 16384 (the ragged rule's
    #      overlapped half-tail program with the column-grouped raster in 12-column-tile groups, in place of the ragged
    #      rule's 8; round 8 at T 16231 / 16384: the tall epilogue released by row half in place of the column-chunk
    #      one, measured against the round-6 program), with the per-tile K partitions of the row's plan without them,
    #      measured on GB200 against that plan ----
    ('sm_100a', False, False, False, False, False, 5120, 17408, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 12},
    ('sm_100a', False, False, False, False, False, 5120, 17408, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 12},
    # ---- sm_100a: the GB T-exact rules of round 7 - the 5120 -> 17408 weight gradient at T 16231 / 16384 in its direct
    #      orientation (the half-tail program through the one-slot TMA store, 12-row raster groups: the sm_103a
    #      keys' rule of the row), with the per-tile K partitions of the row's plan without them, measured on
    #      GB200 against that plan ----
    ('sm_100a', True, True, False, False, False, 5120, 16231, 17408): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_100a', True, True, False, False, False, 5120, 16384, 17408): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    # ---- sm_100a: the GB T-exact rules of round 8 - the 16384 -> 6144 forward at T 16231 (the half-tail program with
    #      the tall epilogue released by row half and the column-grouped raster in 8-column-tile groups, in place of the
    #      ragged rule's overlapped half-tail program), with the per-tile K partitions of the row's plan without it,
    #      measured on GB200 against that plan ----
    ('sm_100a', False, False, False, False, False, 6144, 16384, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    # ---- sm_100a: the GB T-exact rules of round 10 - the 4096 -> 12288 input gradient at T 16384 (the half-tail
    #      program with the tall epilogue released by row half and the column-grouped raster in 16-column-tile groups,
    #      in place of the ragged rule's overlapped half-tail program in the same groups), with the per-tile K
    #      partitions of the row's plan without it, measured on GB200 against that plan ----
    ('sm_100a', False, True, False, False, False, 4096, 12288, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 16},
    # ---- sm_100a: the GB T-exact rules of round 11 - measured T-exact rules moved to neighbouring T: the 16384 -> 6144
    #      forward at T 16172 / 16384 (round 8's rule), the 12288 -> 4096 forward at T 16172 (the sm_103a round-9 rule:
    #      the half-tail program with the tall epilogue released by row half and the column-grouped raster in
    #      8-column-tile groups), the 17408 -> 5120 forward at T 16172 (round 8's rule), the 5120 -> 17408 input gradient
    #      at T 16172 (round 4's rule) and the 4096 -> 12288 input gradient at T 16231 (round 10's rule), each with the
    #      per-tile K partitions of the row's plan without it (bitwise equal to that plan), measured on GB200 against
    #      that plan on two nodes ----
    ('sm_100a', False, False, False, False, False, 6144, 16384, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, False, False, False, False, 6144, 16384, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, False, False, False, False, 4096, 12288, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, True, False, False, False, 5120, 17408, 16172): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 12, "epi": 'tma', "slots": 1},
    ('sm_100a', False, False, False, False, False, 5120, 17408, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 12},
    ('sm_100a', False, True, False, False, False, 4096, 12288, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 16},
    # ---- sm_100a: the GB T-exact rule of round 12 - the 4096 -> 12288 weight gradient at T 16231 (the ragged rule's
    #      exact 16-way stream-K program with the tail-first order, in 2-row raster groups in place of the default 16),
    #      measured on GB200 against the plan without it on two nodes; the raster changes which four output tiles take
    #      the stream-K tail, so the outputs of those tiles differ from that plan's (each plan is deterministic) ----
    ('sm_100a', True, True, False, False, False, 4096, 16231, 12288): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True, "sk_exact_fit": 74, "group_m": 2},
    # ---- sm_103a: the shapes tranche - forward / input gradient / weight gradient rows at K 4096 - 17408 and N 32 - 17408,
    #      measured with this package's planner against cuBLAS on GB300 at T 16172 - 16384 (see the test table
    #      MEASURED_SHAPE_ROWS); a row whose measured choice equals the planner's default carries no rule ----
    ('sm_103a', False, False, False, False, False, 32, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 32, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 48, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 48, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 64, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 64, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 128, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 128, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 128, 8192, None): {"pro": 15},
    ('sm_103a', False, False, False, False, False, 1024, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 1024, 5120, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 4096, 2048, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_103a', False, False, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_103a', False, False, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_103a', False, False, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 16384, 2048, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_103a', False, True, False, False, False, 2048, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, False, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    # the 4096 -> 48 input gradient (one K step): the eight-warp program runs whole tiles through the one-slot store;
    # the row's load-warp half-tail program (two staging slots) cannot carry the form, so the rule names no ten-warp
    # knobs and the architecture default program runs where the form yields; a caller that forces ``epi="tma",
    # slots=2, htail=True, lw=True, pro=15`` plans the row's previous program
    ('sm_103a', False, True, False, False, False, 4096, 48, None): {"form": 'w8r', "sk": False},
    ('sm_103a', False, True, False, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_103a', False, True, False, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_103a', False, True, False, False, False, 4096, 512, None): {"form": 'w8rs', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "lw": True, "promo": 'l2_256b', "mg": True, "sk": False},
    # the raster group of 8 row tiles is the row's measured launch parameter
    ('sm_103a', False, True, False, False, False, 4096, 1024, None): {"form": 'w8rs', "pd": 2, "pro": 15, "group_m": 8, "sk": False},
    ('sm_103a', False, True, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_103a', False, True, False, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_103a', False, True, False, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_103a', False, True, False, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "cta1": True, "pd": 3},
    ('sm_103a', False, True, False, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_103a', False, True, False, False, False, 5120, 512, None): {"form": 'w8r', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 12, "promo": 'l2_256b', "mg": True, "sk": False, "sw": 25},
    # the raster group of 12 row tiles is the row's measured launch parameter
    ('sm_103a', False, True, False, False, False, 5120, 1024, None): {"form": 'w8r', "pd": 2, "pro": 15, "sw": 25, "group_m": 12, "sk": False},
    ('sm_103a', False, True, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_103a', False, True, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovr": True, "htail": True},
    # the kv_a input gradient (6144 -> 576): the inherited sm_100a knobs restated so that this row keeps its program
    # where the eight-warp form yields and when it takes the whole-tile tile resolve (an sm_103a rule stops the parent
    # lookup); the raster group of 12 row tiles is the row's measured launch parameter
    ('sm_103a', False, True, False, False, False, 6144, 576, None): {"form": 'w8rs', "group_m": 12, "slots": 1, "epi": 'tma', "pd": 2, "pro": 15, "mg": True, "sk": False},
    ('sm_103a', False, True, False, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 16, "pro": 15},
    # the batched MLA value-projection input gradient (512 -> 256) on GB300: sm_100a's program, raster group 2
    ('sm_103a', False, True, False, False, True, 512, 256, None): {"form": 'w8rs', "batch_group": 8, "store_hint": "evict_last", "promo": 'l2_256b', "group_m": 2},
    ('sm_103a', False, True, True, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', True, True, False, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 32, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 48, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 64, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 128, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 32, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 48, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 64, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 128, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, False, False, 4096, None, 1024): {"block_n": 224, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True},
    # ``wave_yield`` / ``sk_exact_fit``: see the sm_100a rules of these rows (measured on B300)
    ('sm_103a', True, True, False, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "block_n": 224, "wave_yield": {"sk_sync": True, "sk_slab": 2, "sk_sync_m": 32}},
    ('sm_103a', True, True, False, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True, "sk_exact_fit": 74},
    ('sm_103a', True, True, False, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 39, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, False, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, False, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 32},
    ('sm_103a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_103a', True, True, False, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True, "sk_exact_fit": 74},
    ('sm_103a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_103a', True, True, False, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_103a', True, True, True, False, False, 4096, None, 1024): {"block_n": 224, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_103a', True, True, True, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_103a', True, True, True, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_103a', True, True, True, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 26, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, True, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, True, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_103a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_103a', True, True, True, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_103a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_103a', True, True, True, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    # ---- sm_103a: the GB T-exact rules - the 4096 -> 512 and 5120 -> 512 forwards (the half-tail column-grouped
    #      program), measured on GB300 at T 16231 / 16384 like the sm_100a GB T-exact rules above, and the sm_100a
    #      6144 x 2048 keys restated (T 16172 / 16231, measured on GB300 through the parent lookup) so that these rows
    #      keep them without it ----
    ('sm_103a', False, False, False, False, False, 512, 4096, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_103a', False, False, False, False, False, 512, 4096, 16384): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_103a', False, False, False, False, False, 512, 5120, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_103a', False, False, False, False, False, 512, 5120, 16384): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_103a', False, False, False, False, False, 6144, 2048, 16172): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_103a', False, False, False, False, False, 6144, 2048, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15},
    ('sm_103a', False, True, False, False, False, 6144, 2048, 16172): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15, "group_m": 16},
    ('sm_103a', False, True, False, False, False, 6144, 2048, 16231): {"block_n": 256, "cta_rows": 128, "htail": True, "cgrp": True, "pro": 15, "group_m": 16},
    # ---- sm_103a: the GB T-exact rules of round 2 - the q_b 2048 x 16384 input gradient (the column-grouped raster
    #      in 16-column-tile groups at T 16172 / 16231) and the 5120 -> 17408 / 17408 -> 5120 weight gradients at
    #      T 16231 / 16384 (the half-tail program through the one-slot TMA store, 12-row raster groups: the sm_100a key's
    #      program on the 17408 -> 5120 row, restated here), measured on GB300 like the sm_100a rules above ----
    ('sm_103a', False, True, False, False, False, 2048, 16384, 16172): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_103a', False, True, False, False, False, 2048, 16384, 16231): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_103a', True, True, False, False, False, 5120, 16231, 17408): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_103a', True, True, False, False, False, 5120, 16384, 17408): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_103a', True, True, False, False, False, 17408, 16231, 5120): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_103a', True, True, False, False, False, 17408, 16384, 5120): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12},
    # ---- sm_103a: the GB T-exact rules of round 3 - the indexer_q 2048 -> 4096 weight gradient at T 16172 / 16231
    #      in its direct orientation: the synchronised two-partial stream-K program of the sm_100a ragged rule
    #      (which this row inherits) with its bf16 tiles through the one-slot TMA store, with the per-tile K
    #      partitions of the row's plan without it, measured on GB300 against that plan ----
    ('sm_103a', True, True, False, False, False, 2048, 16172, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2, "epi": 'tma', "slots": 1},
    ('sm_103a', True, True, False, False, False, 2048, 16231, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2, "epi": 'tma', "slots": 1},
    # ---- sm_103a: the GB T-exact rules of round 4 - the 5120 -> 17408 input gradient at T 16231 / 16384 (the
    #      overlapped half-tail program with the column-grouped raster in 12-column-tile groups, in place of the ragged
    #      rule's row-half program), with the bf16 tiles through the one-slot TMA store and the per-tile K partitions of
    #      the row's plan without it, measured on GB300 against that plan ----
    ('sm_103a', False, True, False, False, False, 5120, 17408, 16231): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 12, "epi": 'tma', "slots": 1},
    ('sm_103a', False, True, False, False, False, 5120, 17408, 16384): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 12, "epi": 'tma', "slots": 1},
    # ---- sm_103a: the GB T-exact rules of round 6 - the 17408 -> 5120 forward at T 16231 / 16384 (the ragged rule's
    #      overlapped half-tail program with the column-grouped raster in 12-column-tile groups, in place of the ragged
    #      rule's 8; round 9 at T 16231 / 16384: the tall epilogue released by row half in place of the column-chunk
    #      one, the sm_100a keys' rule, measured against the round-6 program), with the per-tile K partitions of the
    #      row's plan without them, measured on GB300 against that plan ----
    ('sm_103a', False, False, False, False, False, 5120, 17408, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 12},
    ('sm_103a', False, False, False, False, False, 5120, 17408, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 12},
    # ---- sm_103a: the GB T-exact rules of round 8 - the 16384 -> 6144 forward at T 16231 (the sm_100a key's program:
    #      the half-tail program with the tall epilogue released by row half and the column-grouped raster in
    #      8-column-tile groups, in place of the inherited sm_100a ragged rule's overlapped half-tail program), with the
    #      per-tile K partitions of the row's plan without it, measured on GB300 against that plan ----
    ('sm_103a', False, False, False, False, False, 6144, 16384, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    # ---- sm_103a: the GB T-exact rules of round 9 - the 12288 -> 4096 forward at T 16231 / 16384 (the half-tail
    #      program with the tall epilogue released by row half and the column-grouped raster in 8-column-tile groups,
    #      in place of the ragged rule's overlapped half-tail program in 16-column-tile groups), with the per-tile K
    #      partitions of the row's plan without them, measured on GB300 against that plan; at T 16384 the key also
    #      walks K serpentine over blocks of 76 items (``kser`` 76, see the T 16172 keys),
    #      in 16-column-tile groups at T 16384 ----
    ('sm_103a', False, False, False, False, False, 4096, 12288, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_103a', False, False, False, False, False, 4096, 12288, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 16, "kser": 76},
    # ---- sm_103a: the GB T-exact rules of round 10 - the 4096 -> 12288 input gradient at T 16231 / 16384 (the sm_100a
    #      key's program: the half-tail program with the tall epilogue released by row half and the column-grouped
    #      raster in 16-column-tile groups, in place of the ragged rule's overlapped half-tail program), with the
    #      per-tile K partitions of the row's plan without them, measured on GB300 against that plan; at T 16231 / 16384
    #      the key also walks K serpentine over blocks of 76 items (``kser`` 76, see the T 16172 keys) ----
    ('sm_103a', False, True, False, False, False, 4096, 12288, 16231): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 16, "kser": 76},
    ('sm_103a', False, True, False, False, False, 4096, 12288, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 16, "kser": 76},
    # ---- sm_103a: the GB T-exact rules of round 11 - measured T-exact rules moved to neighbouring T: the 16384 -> 6144
    #      forward at T 16172 / 16384 (round 8's rule), the 12288 -> 4096 forward at T 16172 (round 9's rule), the
    #      17408 -> 5120 forward at T 16172 (round 9's rule), the 5120 -> 17408 input gradient at T 16172 (round 4's
    #      rule), the 4096 -> 12288 input gradient at T 16172 (round 10's rule) and the 5120 -> 17408 / 17408 -> 5120
    #      weight gradients at T 16172 (the shapes tranche's rule of their T 16231 / 16384 keys: the half-tail program
    #      with the bf16 tiles through the one-slot TMA store in 12-row-tile groups), each with the per-tile K partitions
    #      of the row's plan without it (bitwise equal to that plan), measured on GB300 against that plan on two nodes ----
    ('sm_103a', False, False, False, False, False, 6144, 16384, 16384): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_103a', False, False, False, False, False, 6144, 16384, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8},
    # the 12288 -> 4096 forward and the 4096 -> 12288 input gradient at T 16172 also walk K serpentine over blocks
    # of 76 items (``kser`` 76, the CTA pairs of the 152-SM part; the wave guard drops it on the parts with another
    # pair count), measured on GB300 against the best library GEMM on two nodes
    ('sm_103a', False, False, False, False, False, 4096, 12288, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 8, "kser": 76},
    ('sm_103a', False, False, False, False, False, 5120, 17408, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 12},
    ('sm_103a', False, True, False, False, False, 5120, 17408, 16172): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 12, "epi": 'tma', "slots": 1},
    # the serpentine K walk over blocks of 76 items (``kser`` 76: see the 12288 -> 4096 forward at T 16172)
    ('sm_103a', False, True, False, False, False, 4096, 12288, 16172): {"cta_rows": 256, "ovr": True, "htail": True, "cgrp": True, "group_m": 16, "kser": 76},
    #      (the two weight gradients carry ht_min_tail 16, a package-only planner refinement: their 680-tile rows keep
    #      a tail under 16 tiles whole - 148 SMs, 14 tiles, measured slower as 28 half items than whole - and plan as
    #      before on 152 SMs (no half-item fit) and 160 SMs (40 tiles, 80 half items))
    ('sm_103a', True, True, False, False, False, 5120, 16172, 17408): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12, "ht_min_tail": 16},
    ('sm_103a', True, True, False, False, False, 17408, 16172, 5120): {"cta_rows": 256, "htail": True, "epi": 'tma', "slots": 1, "group_m": 12, "ht_min_tail": 16},
    # ---- sm_103a: the GB T-exact rules of round 13 - the 17408 -> 5120 fp32 weight gradient at T 16231 / 16384 in
    #      12-row raster groups (the program the row runs without them: its own 128-row fp32 program is not generated
    #      on sm_103a, so the registry's knob fallback runs the 256-row tiles through the one-slot TMA store; the rule
    #      names that program so it plans it directly) and the indexer_q 2048 -> 4096 weight gradient at T 16384 with
    #      the rule of its T 16172 / 16231 keys (round 3), each with the per-tile K partitions of the row's plan without
    #      it (bitwise equal to that plan), measured on GB300 against that plan on two nodes ----
    ('sm_103a', True, True, True, False, False, 17408, 16231, 5120): {"cta_rows": 256, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_103a', True, True, True, False, False, 17408, 16384, 5120): {"cta_rows": 256, "epi": 'tma', "slots": 1, "group_m": 12},
    ('sm_103a', True, True, False, False, False, 2048, 16384, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2, "epi": 'tma', "slots": 1},
}
# fmt: on

# Per-architecture defaults under every row rule of that architecture (round 33, Cake unit 2d): a knob measured as a
# win on every pair-program row of the architecture is set once here, and a row rule overrides it (``row_rule`` merges
# exactly as the Cake launcher does).  ``pro`` 7 = the pair prologue / cleanup sync form (no peer-startup rendezvous
# before the collective tcgen05.alloc, relaxed arrive on the post-init cluster sync, the MMA warp as allocator); the
# single-CTA programs (``cta1`` rows) take no ``pro`` (``instance_key`` refuses it): the planner zeroes it there like
# the launcher.  The sm_100a / sm_107a values are byte-identical to Cake ``ARCH_RULE_DEFAULTS``; sm_103a (compute
# capability 10.3) takes the same pair form: without it a pair template plans the round-27 form (``pro`` 0) there, which
# no measured sm_103a rule names.
ARCH_RULE_DEFAULTS: dict[str, dict] = {
    "sm_107a": {"pro": 7},
    "sm_100a": {"pro": 7},
    "sm_103a": {"pro": 7},
}


# Rule inheritance: an architecture whose own rule table covers only the rows its lanes measured plans every other
# row with the parent's rule - the same instance-key knobs, so the same program, which the registry fallback keeps
# only where the architecture registers it.  sm_103a (GB300) registers every sm_100a program and its lanes measured
# the shapes tranche alone; its Cake rows ran the architecture defaults (the generic programs) before, slower than
# the sm_100a programs the GB300 benchmark planned through the sm_100a table.  A rule the architecture measured itself
# always wins (either weight-gradient orientation); sm_100a and sm_107a inherit nothing.
RULE_PARENT: dict[str, str] = {"sm_103a": "sm_100a"}

# Package-only token tiers of ragged ``ROW_RULES`` keys: ``key -> ((bound, rule), ...)`` in ascending ``bound``.  At a
# token count T (``M`` of a forward / input-gradient key, ``K`` of a weight-gradient key) below a tier's ``bound``, the
# first such tier's ``rule`` plans the row in place of the key's; the last ``bound`` is the key's floor, from which on
# the key's own rule applies.  A tier rule is the rule the row planned with before the key's rule changed (``{}``:
# none), or a rule measured for the token counts below its bound (the BF16 K-major rows of version 0.3.7 below).  The
# sm_107a keys are the 212-SM tranche (measured at T 16172 - 16384); the fp32 5120 -> 5120 weight gradient
# plans with no rule up to T 257.  Their counts are the R200 A/B of every plan the tranche changed against the plan
# before it (212 SMs, CUPTI cold L2, 2026-11-08).  The five sm_107a BF16 weight-gradient keys of version 0.3.22 (the
# 64-row exact split of 4096 -> 16 / 96, 6144 -> 48 / 64 and indexer_hw 6144 -> 32) floored their rule at T 8193 in that
# version.  Version 0.3.23 moves each floor to the lowest measured token count from which the rule was faster than the
# plan before at every measured token count up to T 8192 on both measured sm_107a parts (VR200, 212 SMs; an R200
# board, 216 SMs): T 4096 on 4096 -> 16, T 1024 on 6144 -> 48 / 64, T 2048 on indexer_hw.  A middle tier runs the
# two-way 64-row split where it was faster than the plan before at every measured token count of its band on both parts
# and the rule was not (4096 -> 16 at T 1024 - 2048), or where its geometric mean over the band was more than 0.5 %
# above the rule's on both parts (4096 -> 96 at T 1024 - 7168; from T 7169 on that row plans its rule, which was faster
# than the split at T 8065 on both parts).  Below T 1024 the rows plan as before (indexer_hw with its three-way 128-row
# rule below T 2048, the others without a rule).  Version 0.3.25 adds a middle tier of whole 64-row tiles without a K
# split (``ROW_RULES``, version 0.3.25) at T 960 - 3072 on 4096 -> 16, T 960 - 3584 on 4096 -> 96, 6144 -> 48 and
# indexer_hw and T 960 - 5120 on 6144 -> 64 (on 4096 -> 16 in place of the two-way split, which 4096 -> 96 keeps at
# T 3585 - 7168), where it was faster than the plan of version 0.3.24 at every measured token count of each band as
# first set (to T 4096 on 4096 -> 96, 6144 -> 48 and indexer_hw) on both parts before shipping (medians of that plan
# over the tier 1.038 to 1.361 / 1.043 to 1.354, eager / CUDA graph, on the 212-SM part and 1.028 to 1.362 / 1.031 to
# 1.369 on the 216-SM part); the post-ship A/B on the 212-SM part did not find it faster in every entry at T 4096 on
# those three rows, whose bands end at T 3584 (``ROW_RULES``, version 0.3.25).
# Below T 960 the rows plan as before.  Version 0.3.27 moves the first bound of these five keys from T 960 to T 128,
# so the whole 64-row tiles run from T 128 on (one band with that of version 0.3.25), and adds tiers to eight more
# sm_107a BF16 weight-gradient keys (4096 -> 32 / 48 / 64 / 128, 5120 -> 48 / 64 / 128 and 8192 -> 128): the key's
# rule below T 128, the whole 64-row tiles at T 128 - 4096 and the key's rule from T 4097 on; each was faster than the
# plan of version 0.3.26 at every measured token count of its band on both measured sm_107a parts (``ROW_RULES``,
# version 0.3.27).  Below T 128 these thirteen rows plan as before.  The sm_100a
# 5120 -> 64 / 128 bf16 weight-gradient keys floor the rule's synchronised split at T 32768.  On B200 and GB200 (148 and
# 152 SMs) that split was 1 to 3 % slower at T 16231 and faster at T 32768 (1.04 to 1.06 times), 65535 and 131072.  Its
# margin of 16 K steps makes the main parts 264 K blocks long at T 32768, so they run the promoted twin, within the
# naive tier there with the 128-block chunks of the twins before version 0.3.21.  Under the promotion threshold of
# versions up to 0.3.5 (256 K blocks) the same split with margin 0 (parts of 256 K blocks, no promoted chunks) missed
# the naive tier on 5120 -> 64 at T 32768, and from T 31744 down the rule's own parts held at most 256 K blocks,
# without promoted chunks.  Below T 32768 these rows keep their 64-row program, which runs its promoted twin where the
# longest segment holds more than ``K1_PROMOTE_MIN_SEGMENT`` (64) K blocks.  The 4096 x 4096 bf16 weight-gradient keys
# of sm_100a and sm_103a floor the rule's ``wave_yield`` at T 8192, the shortest token count it was measured at on B200
# and B300 (148 SMs, T 8192 - 32768); below it the rows plan without it, as before.  An exact (N, K, M) key has no
# tiers.
# Version 0.3.7 adds tiers to the ragged BF16 forward (both operands K-major) and input-gradient keys with K > 8192 of
# sm_100a and to the four such keys of sm_103a; sm_103a reads the sm_100a tiers through ``RULE_PARENT`` on the rows it
# has no ragged rule of its own for (the o_proj and dense_down forwards, the q_b and dense_gate_up input gradients).  At
# small token counts the rules of these rows (256-row tiles with the half-height tail wave) sum a tile's whole K, 192 to
# 272 K blocks, in one FP32 accumulator, and that plan missed the naive BF16 tier on some cells (a cell: a row at one
# token count on one seed of 32 draws; errors up to 1.0089 times the tier's, the tests allow 1.005): 49 of the 840 cells
# at T 1 - 2049 (8 rows x 21 token counts x 5 seeds) on B200 (sm_100a, 148 SMs), 47 of 840 on B300 (sm_103a, 148 SMs),
# 35 of 840 on GB300 (sm_103a, 152 SMs) and the same 35 cells on GB200 (sm_100a, 152 SMs), all at T 1 - 513; later
# passes judged that plan at more token counts and found no other miss.  That plan met the tier on every judged cell of
# the 12288 -> 4096 and dense_down forwards and of the 4096 -> 12288 input gradient, and on each other row above the
# token counts listed with its key: there the tiers are taken for speed alone, in another summation order.  Up to T 256
# a tier splits the tail's K steps into three units per tail tile (four up to T 512 on the q_b input gradient): with K
# 12288 or 16384 each tail tile sums three (four) partials of 64 K blocks; with K 17408 the units, of 91 K blocks, are
# laid in one line over the tail's K steps, so a tile of 272 K blocks sums three or four partials of 1 to 91 K blocks.
# Above that, and on the o_proj forward from T 1, it plans the 128-row program without the row's rule (whole tiles, or
# the two-way K split of a single partial wave), with the column-grouped raster of the row's rule on the two forwards
# whose rule carries it.  Wherever the plan before missed the tier, the tier's plan holds segments of at most 128 K
# blocks.  Every tier plan met the tier on every judged seed and ran faster than the row's rule at every measured T of
# its band; the ratios below are the row's rule's time over the tier plan's (geometric mean of eager launches and CUDA
# graph replays, cold L2).  A key's floor (its last bound) is one above the highest measured T of its band: at the next
# measured T (769, 1024, 1536 or 3072) the tier plan was no longer faster than the row's rule on every measured part,
# and the unmeasured T from the floor up to that T keep the row's rule.  The bounds come from B200, B300 and GB300: the
# sm_100a keys of the 12288 -> 4096 and 17408 -> 5120 forwards and the 4096 -> 12288 and 5120 -> 17408 input gradients
# were timed on B200 alone and the sm_103a keys of these rows on B300 and GB300; both keys of a row take the same
# bounds, the lowest over the three parts.  GB200 (sm_100a, 152 SMs), measured after the bounds were set, planned every
# tier plan as B200 did; each met the tier on every judged seed, ran 1.04 to 2.34 times as fast as the row's rule at
# every measured T of its band and slower at the next measured T above the key's floor.  160-SM sm_103a parts are not
# measured; they take the tiers of their architecture.  The tier plans that split K sum in another order than the plans
# before them; every tier plan gives the same bits on every run.
# fmt: off
ROW_RULE_MIN_T: dict[tuple, tuple[tuple[int, dict], ...]] = {
    # q_b 2048 x 16384 input gradient: slower at 11, flat at 3, faster at 1 of the 15 measured T 1 - 1025; faster at
    # 12, flat at 3 of the 15 from 2048 on
    ('sm_107a', False, True, False, False, False, 2048, 16384, None): ((2048, {"cgrp": True, "group_m": 8}),),
    # 5120 -> 5120 input gradient: slower at the 10 measured T 1 - 257; faster at 15, flat at 6 of the 21 from 959 on
    ('sm_107a', False, True, False, False, False, 5120, 5120, None): ((959, {}),),
    # 5120 -> 5120 fp32 weight gradient: faster at the 10 measured T 1 - 257 with no rule (the plan the rule's
    # registry fallback ran there); slower at 10, flat at 1 of the 11 measured T 959 - 8192; faster at the 10 from
    # 16129 on
    ('sm_107a', True, True, True, False, False, 5120, None, 5120): ((258, {}), (16129, {"block_n": 256, "cta_rows": 128, "f32_v8": True, "pro": 15, "hints": ('none', 'none'), "epi": 'reg', "b_swz": 128, "pd": 0, "pf": 0})),
    # 12288 -> 4096 weight gradient: slower at 5, flat at 16 of the 21 measured T 1 - 8192; faster at the 10 from
    # 16129 on
    ('sm_107a', True, True, False, False, False, 12288, None, 4096): ((16129, {}),),
    # 4096 -> 12288 weight gradient: slower at 5, flat at 14, faster at 2 of the 21 measured T 1 - 8192; faster at the
    # 10 from 16129 on
    ('sm_107a', True, True, False, False, False, 4096, None, 12288): ((16129, {}),),
    # 12288 -> 4096 input gradient: slower at T 2 and 7, faster at T 1; faster at 26, flat at 2 of the 28 measured T
    # from 33 on
    ('sm_107a', False, True, False, False, False, 12288, 4096, None): ((33, {}),),
    # 4096 -> 12288 forward: slower at T 2 and 7, faster at T 1; faster at 25, flat at 3 of the 28 measured T from 33 on
    ('sm_107a', False, False, False, False, False, 12288, 4096, None): ((33, {}),),
    # 5120 -> 5120 weight gradient: slower at 2, flat at 18, faster at 1 of the 21 measured T 1 - 8192; faster at the
    # 10 from 16129 on
    ('sm_107a', True, True, False, False, False, 5120, None, 5120): ((16129, {}),),
    # version 0.3.27 (0.3.25 - 0.3.26: first bound T 960; 0.3.23 - 0.3.24: floors 4096 / 7169 / 1024 / 1024 / 2048, the
    # two-way split of 4096 -> 16 at T 1024 - 2048; 0.3.22: T 8193), the five BF16 weight gradients of the 64-row exact
    # split (``ROW_RULES``): below T 128 the plan before (no rule; indexer_hw 6144 -> 32 the three-way 128-row rule),
    # then the 64-row whole tiles without a K split as a middle tier (version 0.3.25 from T 960, version 0.3.27 from
    # T 128: 4096 -> 16 at T 128 - 3072, then the plan before at T 3073 - 4095 and its rule from T 4096; 4096 -> 96 at
    # T 128 - 3584, then the two-way split at T 3585 - 7168 and its rule from T 7169; 6144 -> 48 and indexer_hw at
    # T 128 - 3584 and 6144 -> 64 at T 128 - 5120, then their rule)
    ('sm_107a', True, True, False, True, False, 16, None, 4096): ((128, {}), (3073, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}), (4096, {})),
    ('sm_107a', True, True, False, True, False, 96, None, 4096): ((128, {}), (3585, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}), (7169, {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 48, None, 6144): ((128, {}), (3585, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 64, None, 6144): ((128, {}), (5121, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 32, None, 6144): ((128, {"block_n": 128, "sk_exact": 3}), (3585, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    # version 0.3.27, eight more BF16 weight gradients of sm_107a: their rule (``ROW_RULES``) below T 128, the 64-row
    # whole tiles without a K split at T 128 - 4096, their rule from T 4097 on
    ('sm_107a', True, True, False, True, False, 32, None, 4096): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 48, None, 4096): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 64, None, 4096): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 128, None, 4096): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 48, None, 5120): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 64, None, 5120): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 128, None, 5120): ((128, {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    ('sm_107a', True, True, False, True, False, 128, None, 8192): ((128, {"block_n": 128, "sk_sync": True}), (4097, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')})),
    # 5120 -> 64 / 128 weight gradient: the 64-row program below T 32768
    ('sm_100a', True, True, False, True, False, 64, None, 5120): ((32768, {"cta_rows": 64, "pro": 15}),),
    ('sm_100a', True, True, False, True, False, 128, None, 5120): ((32768, {"cta_rows": 64, "pro": 15}),),
    # 4096 x 4096 weight gradient: the rule without its ``wave_yield`` below T 8192
    ('sm_100a', True, True, False, False, False, 4096, None, 4096): ((8192, {"cta_rows": 256, "pro": 7, "block_n": 224}),),
    ('sm_103a', True, True, False, False, False, 4096, None, 4096): ((8192, {"cta_rows": 256, "pro": 7, "block_n": 224}),),
    # 12288 -> 4096 forward, on B200: a three-way K split of the tail tiles up to T 256 (64-block segments), 1.87 to
    # 1.95 times faster; up to T 1024 the 128-row program with the rule's column-grouped raster (8-column-tile groups),
    # 1.66 to 1.68 times faster up to T 512 (a two-way K split of the single partial wave, 96-block segments) and 1.22
    # to 1.25 from T 513 (whole tiles)
    ('sm_100a', False, False, False, False, False, 4096, 12288, None): ((257, {"pro": 7, "sk_parts": 3}), (1025, {"cgrp": True, "pro": 7, "group_m": 8})),
    # 17408 -> 5120 forward, on B200: up to T 256 the tail's K steps split into three units per tail tile of 91 K blocks
    # each (laid in one line over the tail: a tile sums three or four partials), 1.57 to 1.87 times faster; up to T 768
    # whole tiles of the 128-row program with the rule's column-grouped raster (8-column-tile groups), 1.24 to 1.25
    # times faster up to T 512 and 1.79 at T 640 and 768.  The plan before missed the tier at T 33, 64, 97 and 192
    ('sm_100a', False, False, False, False, False, 5120, 17408, None): ((257, {"pro": 7, "sk_parts": 3}), (769, {"cgrp": True, "pro": 7, "group_m": 8})),
    # o_proj 16384 -> 6144 forward, also sm_103a: the 128-row program with no rule up to T 768: a two-way K split of the
    # single partial wave up to T 256 (128-block segments), 1.60 to 1.81 times faster; whole tiles from T 257, 1.23 to
    # 1.27 up to T 512 and 1.67 to 1.76 at T 640 and 768 (B200, B300, GB300).  The plan before missed the tier at T 64,
    # 97, 128 and 129 on B200 and B300, at T 64 on GB300
    ('sm_100a', False, False, False, False, False, 6144, 16384, None): ((769, {"pro": 7}),),
    # dense_down 12288 -> 6144 forward, also sm_103a: a three-way K split of the tail tiles up to T 256 (64-block
    # segments), 1.54 to 1.73 times faster; whole tiles of the 128-row program with no rule up to T 768, 1.23 to 1.29 up
    # to T 512 and 1.65 to 1.73 at T 768 (B200, B300, GB300)
    ('sm_100a', False, False, False, False, False, 6144, 12288, None): ((257, {"pro": 7, "sk_parts": 3}), (769, {"pro": 7})),
    # q_b 2048 x 16384 input gradient, also sm_103a: a four-way K split of the tail tiles up to T 512 (64-block
    # segments), 2.05 to 2.53 times faster; the 128-row program with no rule up to T 2048, 1.64 to 1.83 up to T 1024 (a
    # two-way K split of the single partial wave, 128-block segments) and 1.12 to 1.20 at T 1536 and 2048 (whole tiles;
    # B200, B300, GB300).  The plan before missed the tier at every measured T up to 384 and at 513 (12 T) on B200 and
    # B300, at the same T except 129 on GB300
    ('sm_100a', False, True, False, False, False, 2048, 16384, None): ((513, {"pro": 7, "sk_parts": 4}), (2049, {"pro": 7})),
    # 4096 -> 12288 input gradient, on B200: a three-way K split of the tail tiles up to T 256 (64-block segments), 1.82
    # to 1.93 times faster; the 128-row program with no rule up to T 1024, 1.59 to 1.63 up to T 512 (a two-way K split
    # of the single partial wave, 96-block segments) and 1.16 to 1.18 from T 513 (whole tiles)
    ('sm_100a', False, True, False, False, False, 4096, 12288, None): ((257, {"pro": 7, "sk_parts": 3}), (1025, {"pro": 7})),
    # 5120 -> 17408 input gradient, on B200: up to T 256 the tail's K steps split into three units per tail tile of
    # 91 K blocks each (laid in one line over the tail: a tile sums three or four partials), 1.50 to 1.84 times faster;
    # whole tiles of the 128-row program with no rule up to T 768, 1.19 to 1.21 up to T 512 and 1.81 to 1.82 at T 640
    # and 768.  The plan before missed the tier at T 1, 16, 64, 97, 128, 129 and 192
    ('sm_100a', False, True, False, False, False, 5120, 17408, None): ((257, {"pro": 7, "sk_parts": 3}), (769, {"pro": 7})),
    # dense_gate_up 6144 -> 12288 input gradient, also sm_103a: a three-way K split of the tail tiles up to T 256
    # (64-block segments), 1.55 to 1.75 times faster; whole tiles of the 128-row program with no rule up to T 768, 1.20
    # to 1.28 up to T 512 and 1.69 to 1.77 at T 768 (B200, B300, GB300).  The plan before missed the tier at T 1 on all
    # three
    ('sm_100a', False, True, False, False, False, 6144, 12288, None): ((257, {"pro": 7, "sk_parts": 3}), (769, {"pro": 7})),
    # 12288 -> 4096 forward, on B300 and GB300: a three-way K split of the tail tiles up to T 256 (64-block segments),
    # 1.77 to 1.95 times faster; up to T 1024 the 128-row program with the rule's column-grouped raster (16-column-tile
    # groups), 1.59 to 1.65 up to T 512 (a two-way K split of the single partial wave, 96-block segments) and 1.24 to
    # 1.31 from T 513 (whole tiles)
    ('sm_103a', False, False, False, False, False, 4096, 12288, None): ((257, {"pro": 7, "sk_parts": 3}), (1025, {"cgrp": True, "pro": 7, "group_m": 16})),
    # 17408 -> 5120 forward, on B300 and GB300: up to T 256 the tail's K steps split into three units per tail tile of
    # 91 K blocks each (laid in one line over the tail: a tile sums three or four partials), 1.55 to 1.76 times faster;
    # up to T 768 whole tiles of the 128-row program with the rule's column-grouped raster (8-column-tile groups), 1.25
    # to 1.29 up to T 512 and 1.78 to 1.86 at T 640 and 768.  The plan before missed the tier at T 33, 64, 97 and 192 on
    # B300, at T 1, 33, 64, 128 and 129 on GB300
    ('sm_103a', False, False, False, False, False, 5120, 17408, None): ((257, {"pro": 7, "sk_parts": 3}), (769, {"cgrp": True, "pro": 7, "group_m": 8})),
    # 4096 -> 12288 input gradient, on B300 and GB300: a three-way K split of the tail tiles up to T 256 (64-block
    # segments), 1.72 to 1.91 times faster; the 128-row program with no rule up to T 1024, 1.55 to 1.63 up to T 512 (a
    # two-way K split of the single partial wave, 96-block segments) and 1.18 to 1.26 from T 513 (whole tiles)
    ('sm_103a', False, True, False, False, False, 4096, 12288, None): ((257, {"pro": 7, "sk_parts": 3}), (1025, {"pro": 7})),
    # 5120 -> 17408 input gradient, on B300 and GB300: up to T 256 the tail's K steps split into three units per tail
    # tile of 91 K blocks each (laid in one line over the tail: a tile sums three or four partials), 1.49 to 1.73 times
    # faster; whole tiles of the 128-row program with no rule up to T 768, 1.21 to 1.25 up to T 512 and 1.78 to 1.86 at
    # T 640 and 768.  The plan before missed the tier at T 1, 16, 64, 97, 128, 129 and 192 on B300, at T 1 and 16 on
    # GB300
    ('sm_103a', False, True, False, False, False, 5120, 17408, None): ((257, {"pro": 7, "sk_parts": 3}), (769, {"pro": 7})),
}
# fmt: on

# Package-only rules of exact (N, K, M) ``ROW_RULES`` keys that apply on a part of exactly one SM count,
# ``(arch, sm_count) -> {key: rule}`` (keys as ``ROW_RULES``).  Such a rule replaces the row's rule on that part alone:
# the architecture's other parts, its child architectures (``RULE_PARENT``) and every other token count keep the row's
# ``ROW_RULES`` entry.  The 148-SM sm_100a entries are the fp32 input gradients 4096 -> 32 / 48 / 64 / 128,
# 5120 -> 32 / 48 / 64 / 128 and the fp32 6144 -> 128 input gradient, at the token counts measured on B200: the
# single-CTA program with 128-column tiles, two TMA-store slots and four stages, over the raster group of the row's
# rule (16; 2 on 6144 -> 128).  Against the row's plan on two B200 parts: 1.02 to 1.06 times faster, the output equal
# to that plan's bit for bit at every measured key.  Since version 0.3.8 the 148- and 152-SM sm_103a parts and the
# 152-SM sm_100a part run the same rule on these rows (on 148-SM sm_103a parts all but 5120 -> 128 and 8192 -> 128, on
# 152-SM sm_100a parts all but 8192 -> 128) over token ranges (``ROW_RULE_RANGES_BY_SM`` below); since version 0.3.9
# the 152-SM sm_103a part runs it in raster groups of 2 on 8192 -> 128 at T 16231 - 16384.  Version 0.3.10 runs it in
# small raster groups over token ranges of the 148-SM parts (below): on 148-SM sm_103a parts also on 8192 -> 128; on
# 148-SM sm_100a parts on 8192 -> 128 instead of its two keys here (T 16231 and 16384, raster groups of 16) and on
# 5120 -> 128 and 6144 -> 128 at T 16384 - 32768, beyond the token counts of their keys here.  Version 0.3.11 runs it
# in small raster groups over more token ranges of the 152-SM sm_100a part (below): on 8192 -> 128 at T 16172 and
# 16231 - 32768, on 5120 -> 128 at T 16231 (between its two ranges of version 0.3.8) and on 6144 -> 128 at
# T 16231 - 16384 (after its range of version 0.3.8 at T 16172).
# Version 0.3.28 adds 41 exact keys to the 148-SM sm_100a part (B200) and a 152-SM sm_103a part (GB300) of 57 exact
# keys: forwards, input gradients and weight gradients (BF16 and FP32 outputs) at T 16380, 32736 and 32768, each named
# with its measured program in the comment above it.  Each rule is the full knob set that names that program (the
# inverse of its symbol, without the promotion flag: the planner promotes the chunk by its own segment rule, so a
# ``_kc64`` twin may run).  Each was measured on one part of the SM count against the plan of version 0.3.27 in 21
# paired rounds with cold L2, in eager launches and CUDA graph replays: faster beyond the A/A band in both, the paired
# median in the comment, with one bit pattern run to run and within the naive BF16 tier on five seeds.  The batched
# keys (96-head MLA projections) hold no head count or head stride, so they apply at every head count above one; only
# 96 heads were measured.  Every other part, token count and row keeps the plan of version 0.3.27.
# Version 0.3.29 adds a 152-SM sm_100a part (GB200) of 33 exact keys: forwards, input gradients and weight gradients
# (BF16 and FP32 outputs) at T 16380, 32736 and 32768, each named with its measured program in the comment above it;
# each rule is the full knob set that names that program, as in version 0.3.28.  Each was measured on one part of the SM
# count against the plan of version 0.3.28 in 21 paired rounds with cold L2, in eager launches and CUDA graph replays:
# faster beyond the A/A band in both, the paired median in the comment, with one bit pattern run to run and within the
# naive BF16 tier on five seeds.  Its batched keys (96-head MLA projections) apply at every head count above one; only
# 96 heads were measured.  Every other part, token count and row keeps the plan of version 0.3.28.
# Version 0.3.30 adds 13 exact keys to the 152-SM sm_103a part (GB300) and 5 exact keys to the 148-SM sm_100a part
# (B200): forwards, input gradients and weight gradients (BF16 and FP32 outputs) at T 16380, 32736 and 32768, each named
# with its measured program in the comment above it; each rule is the full knob set that names that program, as in
# version 0.3.28.  Each was measured on one part of its SM count against the plan of version 0.3.29 in 21 paired rounds
# with cold L2, in eager launches and CUDA graph replays: faster beyond the A/A band in both, the paired median in the
# comment, with one bit pattern run to run and within the naive BF16 tier on five seeds.  Every other part, token count
# and row keeps the plan of version 0.3.29.
# fmt: off
_SM100A_148_DGRAD_F32 = {"block_n": 128, "epi": 'tma', "slots": 2, "stages": 4, "cta1": True}
ROW_RULES_BY_SM: dict[tuple[str, int], dict[tuple, dict]] = {
    ("sm_100a", 148): {
        **{
            ('sm_100a', False, True, True, False, False, N, K, T): dict(_SM100A_148_DGRAD_F32)
            for N, K, Ts in (
                (4096, 32, (16231, 16384)),
                (4096, 48, (16231, 16384)),
                (4096, 64, (16231, 16384)),
                (4096, 128, (16231, 16384)),
                (5120, 32, (16231, 16384)),
                (5120, 48, (16231, 16384)),
                (5120, 64, (16231, 16384)),
                (5120, 128, (16231,)),
            )
            for T in Ts
        },
        **{
            ('sm_100a', False, True, True, False, False, 6144, 128, T): {**_SM100A_148_DGRAD_F32, "group_m": 2}
            for T in (16172, 16231)
        },
        # 4096 -> 512 forward, BF16 output, T 32736: kk_n256_ht_cg_pro15, paired median 1.101
        ('sm_100a', False, False, False, False, False, 512, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 512 forward, BF16 output, T 32768: kk_n256_ht_cg_pro15, paired median 1.098
        ('sm_100a', False, False, False, False, False, 512, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 576 forward, BF16 output, T 16380: kk_n192_hee_s8_pro15, paired median 1.054
        ('sm_100a', False, False, False, False, False, 576, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "evict_first"), "htail": False,
            "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15,
            "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "stages": 8,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.121
        ('sm_100a', False, False, False, False, False, 896, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 forward, BF16 output, T 16380: kk_n192_hee_s8_pro15, paired median 1.094
        ('sm_100a', False, False, False, False, False, 1536, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "evict_first"), "htail": False,
            "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15,
            "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "stages": 8,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 4096 -> 2048 forward, BF16 output, T 32736: kk_n256_ht_cg_pro15, paired median 1.011
        ('sm_100a', False, False, False, False, False, 2048, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 forward, BF16 output, T 32736: kk_n256_q_pro7, paired median 1.016
        ('sm_100a', False, False, False, False, False, 4096, 1024, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 forward, BF16 output, T 32768: kk_n256_q_pro7, paired median 1.013
        ('sm_100a', False, False, False, False, False, 4096, 1024, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 forward, BF16 output, T 16380: kk_n256_m256_ov_ht_tma1_cg_pro7, paired median 1.599
        ('sm_100a', False, False, False, False, False, 12288, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 512 -> 24576 forward, BF16 output, T 16380: kk_n256_q_pro7, paired median 1.105
        ('sm_100a', False, False, False, False, False, 24576, 512, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 input gradient, BF16 output, T 16380: kn_n128_tma1_w8relf, paired median 1.058
        ('sm_100a', False, True, False, False, False, 128, 12288, 16380): {
            "b_swz": 128, "batch_group": 0, "block_k": 64, "block_n": 128, "cgrp": False, "cta1": False,
            "cta_rows": 128, "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"),
            "htail": False, "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0,
            "pro": 0, "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 96 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.150
        ('sm_100a', False, True, False, False, False, 7168, 96, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 128 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.170
        ('sm_100a', False, True, False, False, False, 7168, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 576 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.110
        ('sm_100a', False, True, False, False, False, 7168, 576, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.090
        ('sm_100a', False, True, False, False, False, 7168, 896, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 forward + 512 -> 128 input gradient, BF16 output, T 16380: kn_n256_tma1_w8rs, paired median 1.012 / 1.015
        ('sm_100a', False, True, False, False, True, 512, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8rs", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 128 -> 12288 weight gradient, BF16 output, T 16380: nn_n256_sks_sb2_pro7, paired median 1.121
        ('sm_100a', True, True, False, False, False, 128, 16380, 12288): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 512 -> 24576 weight gradient, BF16 output, T 16380: nn_n256_sks_sb2_pro7, paired median 1.337
        ('sm_100a', True, True, False, False, False, 512, 16380, 24576): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, BF16 output, T 32736: nn_n256_sks_sb2_pro7, paired median 1.089
        ('sm_100a', True, True, False, False, False, 1024, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, BF16 output, T 32768: nn_n256_sks_sb2_pro7, paired median 1.099
        ('sm_100a', True, True, False, False, False, 1024, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 3584 -> 7168 weight gradient, BF16 output, T 16380: nn_n256_m256_sks_sb2_pro7, paired median 1.069
        ('sm_100a', True, True, False, False, False, 3584, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 1024 weight gradient, BF16 output, T 32736: nn_n256_sks_sb2_pro7, paired median 1.097
        ('sm_100a', True, True, False, False, False, 4096, 32736, 1024): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 2048 weight gradient, BF16 output, T 32736: nn_n256_sks_sb2_pro7, paired median 1.026
        ('sm_100a', True, True, False, False, False, 4096, 32736, 2048): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 1024 weight gradient, BF16 output, T 32768: nn_n256_sks_sb2_pro7, paired median 1.095
        ('sm_100a', True, True, False, False, False, 4096, 32768, 1024): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 896 weight gradient, BF16 output, T 16380: nn_n224_m256_pro7, paired median 1.094
        ('sm_100a', True, True, False, False, False, 7168, 16380, 896): {
            "b_swz": 128, "batch_group": 0, "block_n": 224, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 weight gradient, BF16 output, T 16380: nn_n160_m256_bz64_hne_pro15, paired median 1.480
        ('sm_100a', True, True, False, False, False, 7168, 16380, 1536): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 3584 weight gradient, BF16 output, T 16380: nn_n256_m256_sks_sb2_pro7, paired median 1.057
        ('sm_100a', True, True, False, False, False, 7168, 16380, 3584): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 576 weight gradient, BF16 output, T 16380: nn_n192_m256_t_tma1_pro7, paired median 1.128
        ('sm_100a', True, True, False, True, False, 576, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 2112 weight gradient, BF16 output, T 16380: nn_n256_m64_hen_t_pro15, paired median 1.344
        ('sm_100a', True, True, False, True, False, 2112, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 weight gradient + 512 -> 128 weight gradient, BF16 output, T 16380: nn_n128_hen_t_pro15, paired median 1.043 / 1.050
        ('sm_100a', True, True, False, True, True, 128, 16380, 512): {
            "b_swz": 128, "batch_group": 0, "block_n": 128, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.210
        ('sm_100a', True, True, True, False, False, 128, 16380, 12288): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 512 -> 24576 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.330
        ('sm_100a', True, True, True, False, False, 512, 16380, 24576): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, FP32 output, T 32736: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.104
        ('sm_100a', True, True, True, False, False, 1024, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, FP32 output, T 32768: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.104
        ('sm_100a', True, True, True, False, False, 1024, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 2048 weight gradient, FP32 output, T 32736: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.010
        ('sm_100a', True, True, True, False, False, 4096, 32736, 2048): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 896 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.075
        ('sm_100a', True, True, True, False, False, 7168, 16380, 896): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 weight gradient, FP32 output, T 16380: nn_n160_m256_bz64_hne_f32_v8_ef_pro15, paired median 1.471
        ('sm_100a', True, True, True, False, False, 7168, 16380, 1536): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 3584 weight gradient, FP32 output, T 16380: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.055
        ('sm_100a', True, True, True, False, False, 7168, 16380, 3584): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 576 weight gradient, FP32 output, T 16380: nn_n256_m64_hen_f32_t_pro15, paired median 1.101
        ('sm_100a', True, True, True, True, False, 576, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 2112 weight gradient, FP32 output, T 16380: nn_n192_hee_f32_t_tma1_pro7, paired median 1.318
        ('sm_100a', True, True, True, True, False, 2112, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("evict_first", "evict_first"), "htail": False,
            "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7,
            "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 weight gradient + 512 -> 128 weight gradient, FP32 output, T 16380: nn_n128_hen_f32_t_pro15, paired median 1.040 / 1.052
        ('sm_100a', True, True, True, True, True, 128, 16380, 512): {
            "b_swz": 128, "batch_group": 0, "block_n": 128, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1536 -> 4096 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.008
        ('sm_100a', False, False, False, False, False, 4096, 1536, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 input gradient, BF16 output, T 32736: kn_n256_ht_cg_pro15, paired median 1.012
        ('sm_100a', False, True, False, False, False, 1024, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 input gradient, BF16 output, T 32768: kn_n256_ht_cg_pro15, paired median 1.011
        ('sm_100a', False, True, False, False, False, 1024, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 1024 input gradient, BF16 output, T 32736: kn_n256_tma1_w8rs, paired median 1.005
        ('sm_100a', False, True, False, False, False, 4096, 1024, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8rs", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 1536 input gradient, BF16 output, T 16380: kn_n256_ht_cg_pro15, paired median 1.019
        ('sm_100a', False, True, False, False, False, 7168, 1536, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
    },
    ("sm_100a", 152): {
        # 4096 -> 512 forward, BF16 output, T 32736: kk_n256_ht_cg_pro15, paired median 1.106
        ('sm_100a', False, False, False, False, False, 512, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 512 forward, BF16 output, T 32768: kk_n256_ht_cg_pro15, paired median 1.113
        ('sm_100a', False, False, False, False, False, 512, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 896 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.099
        ('sm_100a', False, False, False, False, False, 896, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.081
        ('sm_100a', False, False, False, False, False, 1536, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 forward, BF16 output, T 32736: kk_n256_m256_ht_cg_or_pro7, paired median 1.046
        ('sm_100a', False, False, False, False, False, 4096, 8192, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 forward, BF16 output, T 32768: kk_n256_m256_ht_cg_or_pro7, paired median 1.044
        ('sm_100a', False, False, False, False, False, 4096, 8192, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 forward, BF16 output, T 16380: kk_n256_m256_ov_ht_tma1_cg_pro7, paired median 1.437
        ('sm_100a', False, False, False, False, False, 12288, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 512 -> 24576 forward, BF16 output, T 16380: kk_n256_q_pro7, paired median 1.089
        ('sm_100a', False, False, False, False, False, 24576, 512, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 input gradient, BF16 output, T 16380: kn_n128_tma1_w8relf, paired median 1.109
        ('sm_100a', False, True, False, False, False, 128, 12288, 16380): {
            "b_swz": 128, "batch_group": 0, "block_k": 64, "block_n": 128, "cgrp": False, "cta1": False,
            "cta_rows": 128, "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"),
            "htail": False, "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0,
            "pro": 0, "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 96 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.149
        ('sm_100a', False, True, False, False, False, 7168, 96, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 128 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.163
        ('sm_100a', False, True, False, False, False, 7168, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 576 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.070
        ('sm_100a', False, True, False, False, False, 7168, 576, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 input gradient, BF16 output, T 16380: kn_n256_tma1_w8r, paired median 1.057
        ('sm_100a', False, True, False, False, False, 7168, 896, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8r", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 128 -> 12288 weight gradient, BF16 output, T 16380: nn_n160_m256_bz64_hne_cg_pro15, paired median 1.107
        ('sm_100a', True, True, False, False, False, 128, 16380, 12288): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": True, "cta1": False, "cta_rows": 256, "epi": "reg",
            "f32_v8": False, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False, "mg": False,
            "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False, "sh": 0,
            "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none", "stt": False,
            "sw": 0,
        },
        # 512 -> 24576 weight gradient, BF16 output, T 16380: nn_n256_sks_sb2_pro7, paired median 1.214
        ('sm_100a', True, True, False, False, False, 512, 16380, 24576): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 weight gradient, BF16 output, T 16380: nn_n160_m256_bz64_hne_pro15, paired median 1.372
        ('sm_100a', True, True, False, False, False, 7168, 16380, 1536): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, BF16 output, T 32736: nn_n256_m256_sks_sb2_pro7, paired median 1.081
        ('sm_100a', True, True, False, False, False, 8192, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, BF16 output, T 32768: nn_n256_m256_sks_sb2_pro7, paired median 1.074
        ('sm_100a', True, True, False, False, False, 8192, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 576 weight gradient, BF16 output, T 16380: nn_n192_m256_t_tma1_pro7, paired median 1.226
        ('sm_100a', True, True, False, True, False, 576, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 2112 weight gradient, BF16 output, T 16380: nn_n256_m64_hen_t_pro15, paired median 1.228
        ('sm_100a', True, True, False, True, False, 2112, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 weight gradient + 512 -> 128 weight gradient, BF16 output, T 16380: nn_n128_hen_t_pro15, paired median 1.063 / 1.062
        ('sm_100a', True, True, False, True, True, 128, 16380, 512): {
            "b_swz": 128, "batch_group": 0, "block_n": 128, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 weight gradient, FP32 output, T 16380: nn_n160_m256_bz64_hne_f32_v8_ef_pro15, paired median 1.203
        ('sm_100a', True, True, True, False, False, 128, 16380, 12288): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 512 -> 24576 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.213
        ('sm_100a', True, True, True, False, False, 512, 16380, 24576): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, FP32 output, T 32736: nn_n256_f32_tma1_cg_pro15, paired median 1.062
        ('sm_100a', True, True, True, False, False, 1024, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, FP32 output, T 32768: nn_n256_f32_tma1_cg_pro15, paired median 1.068
        ('sm_100a', True, True, True, False, False, 1024, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 4096 -> 8192 weight gradient, FP32 output, T 32736: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.082
        ('sm_100a', True, True, True, False, False, 4096, 32736, 8192): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 4096 -> 8192 weight gradient, FP32 output, T 32768: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.070
        ('sm_100a', True, True, True, False, False, 4096, 32768, 8192): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.047
        ('sm_100a', True, True, True, False, False, 7168, 16380, 896): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.355
        ('sm_100a', True, True, True, False, False, 7168, 16380, 1536): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, FP32 output, T 32768: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.080
        ('sm_100a', True, True, True, False, False, 8192, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 576 weight gradient, FP32 output, T 16380: nn_n256_m64_hen_f32_t_pro15, paired median 1.038
        ('sm_100a', True, True, True, True, False, 576, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 2112 weight gradient, FP32 output, T 16380: nn_n256_m64_hen_f32_t_pro15, paired median 1.228
        ('sm_100a', True, True, True, True, False, 2112, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 weight gradient + 512 -> 128 weight gradient, FP32 output, T 16380: nn_n128_hen_f32_t_pro15, paired median 1.050 / 1.060
        ('sm_100a', True, True, True, True, True, 128, 16380, 512): {
            "b_swz": 128, "batch_group": 0, "block_n": 128, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
    },
    ("sm_103a", 152): {
        # 4096 -> 512 forward, BF16 output, T 32736: kk_n256_ht_cg_pro15, paired median 1.113
        ('sm_103a', False, False, False, False, False, 512, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 512 forward, BF16 output, T 32768: kk_n256_ht_cg_pro15, paired median 1.111
        ('sm_103a', False, False, False, False, False, 512, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 576 forward, BF16 output, T 16380: kk_n192_hee_s8_pro15, paired median 1.112
        ('sm_103a', False, False, False, False, False, 576, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "evict_first"), "htail": False,
            "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15,
            "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "stages": 8,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.133
        ('sm_103a', False, False, False, False, False, 896, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.097
        ('sm_103a', False, False, False, False, False, 1536, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 2048 forward, BF16 output, T 32736: kk_n256_ht_cg_pro15, paired median 1.029
        ('sm_103a', False, False, False, False, False, 2048, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 forward, BF16 output, T 32736: kk_n256_q_pro7, paired median 1.012
        ('sm_103a', False, False, False, False, False, 4096, 1024, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 forward, BF16 output, T 32736: kk_n256_m256_ht_cg_or_pro7, paired median 1.041
        ('sm_103a', False, False, False, False, False, 4096, 8192, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 forward, BF16 output, T 32768: kk_n256_m256_ht_cg_or_pro7, paired median 1.047
        ('sm_103a', False, False, False, False, False, 4096, 8192, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 forward, BF16 output, T 16380: kk_n256_q_pro7, paired median 1.474
        ('sm_103a', False, False, False, False, False, 12288, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 12288 forward, BF16 output, T 16380: kk_n256_m256_ht_cg_or_pro7, paired median 1.011
        ('sm_103a', False, False, False, False, False, 12288, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 512 -> 24576 forward, BF16 output, T 16380: kk_n256_q_pro7, paired median 1.083
        ('sm_103a', False, False, False, False, False, 24576, 512, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": True,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 input gradient, BF16 output, T 16380: kn_n128_tma1_w8relf, paired median 1.052
        ('sm_103a', False, True, False, False, False, 128, 12288, 16380): {
            "b_swz": 128, "batch_group": 0, "block_k": 64, "block_n": 128, "cgrp": False, "cta1": False,
            "cta_rows": 128, "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"),
            "htail": False, "lw": False, "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0,
            "pro": 0, "quad_store": False, "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1,
            "store_ef": False, "store_hint": "none", "stt": False, "sw": 0,
        },
        # 4096 -> 8192 input gradient, BF16 output, T 32736: kn_n256_m256_ov_ht_cg_pro7, paired median 1.028
        ('sm_103a', False, True, False, False, False, 4096, 8192, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 8192 input gradient, BF16 output, T 32768: kn_n256_m256_ov_ht_cg_pro7, paired median 1.031
        ('sm_103a', False, True, False, False, False, 4096, 8192, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 16384 input gradient, BF16 output, T 32736: kn_n256_m256_ov_ht_cg_pro7, paired median 1.039
        ('sm_103a', False, True, False, False, False, 4096, 16384, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 16384 input gradient, BF16 output, T 32768: kn_n256_m256_ov_ht_cg_pro7, paired median 1.035
        ('sm_103a', False, True, False, False, False, 4096, 16384, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 17408 input gradient, BF16 output, T 32736: kn_n256_m256_ov_ht_cg_pro7, paired median 1.052
        ('sm_103a', False, True, False, False, False, 4096, 17408, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 20480 input gradient, BF16 output, T 32736: kn_n256_m256_ov_ht_cg_pro7, paired median 1.053
        ('sm_103a', False, True, False, False, False, 4096, 20480, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 20608 input gradient, BF16 output, T 32736: kn_n256_m256_ov_ht_cg_pro7, paired median 1.055
        ('sm_103a', False, True, False, False, False, 4096, 20608, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": True, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 96 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.136
        ('sm_103a', False, True, False, False, False, 7168, 96, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 128 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.155
        ('sm_103a', False, True, False, False, False, 7168, 128, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 576 input gradient, BF16 output, T 16380: kn_n256_tma1_w8relf, paired median 1.119
        ('sm_103a', False, True, False, False, False, 7168, 576, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8relf", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 input gradient, BF16 output, T 16380: kn_n256_tma1_w8rs, paired median 1.084
        ('sm_103a', False, True, False, False, False, 7168, 896, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "w8rs", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 0, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 128 -> 12288 weight gradient, BF16 output, T 16380: nn_n256_sks_sb2_pro7, paired median 1.080
        ('sm_103a', True, True, False, False, False, 128, 16380, 12288): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 512 -> 24576 weight gradient, BF16 output, T 16380: nn_n256_sks_sb2_pro7, paired median 1.313
        ('sm_103a', True, True, False, False, False, 512, 16380, 24576): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, BF16 output, T 32736: nn_n256_sks_sb2_pro7, paired median 1.102
        ('sm_103a', True, True, False, False, False, 1024, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, BF16 output, T 32768: nn_n256_sks_sb2_pro7, paired median 1.086
        ('sm_103a', True, True, False, False, False, 1024, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 3584 -> 7168 weight gradient, BF16 output, T 16380: nn_n224_m256_pro7, paired median 1.082
        ('sm_103a', True, True, False, False, False, 3584, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 224, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 2048 weight gradient, BF16 output, T 32736: nn_n224_m256_pro7, paired median 1.055
        ('sm_103a', True, True, False, False, False, 4096, 32736, 2048): {
            "b_swz": 128, "batch_group": 0, "block_n": 224, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 8192 weight gradient, BF16 output, T 32736: nn_n256_m256_sks_sb2_pro7, paired median 1.101
        ('sm_103a', True, True, False, False, False, 4096, 32736, 8192): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 8192 weight gradient, BF16 output, T 32768: nn_n256_m256_sks_sb2_pro7, paired median 1.082
        ('sm_103a', True, True, False, False, False, 4096, 32768, 8192): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 6144 -> 7168 weight gradient, BF16 output, T 16380: nn_n256_m256_sks_sb2_pro7, paired median 1.064
        ('sm_103a', True, True, False, False, False, 6144, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 896 weight gradient, BF16 output, T 16380: nn_n256_sks_sb2_pro7, paired median 1.128
        ('sm_103a', True, True, False, False, False, 7168, 16380, 896): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 weight gradient, BF16 output, T 16380: nn_n160_m256_bz64_hne_pro15, paired median 1.502
        ('sm_103a', True, True, False, False, False, 7168, 16380, 1536): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 3584 weight gradient, BF16 output, T 16380: nn_n224_m256_pro7, paired median 1.082
        ('sm_103a', True, True, False, False, False, 7168, 16380, 3584): {
            "b_swz": 128, "batch_group": 0, "block_n": 224, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 6144 weight gradient, BF16 output, T 16380: nn_n256_m256_sks_sb2_pro7, paired median 1.073
        ('sm_103a', True, True, False, False, False, 7168, 16380, 6144): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, BF16 output, T 32736: nn_n256_m256_sks_sb2_pro7, paired median 1.104
        ('sm_103a', True, True, False, False, False, 8192, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, BF16 output, T 32768: nn_n256_m256_sks_sb2_pro7, paired median 1.103
        ('sm_103a', True, True, False, False, False, 8192, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 576 weight gradient, BF16 output, T 16380: nn_n192_m256_t_tma1_pro7, paired median 1.162
        ('sm_103a', True, True, False, True, False, 576, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 192, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 2112 weight gradient, BF16 output, T 16380: nn_n256_m64_hen_t_pro15, paired median 1.287
        ('sm_103a', True, True, False, True, False, 2112, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 weight gradient + 512 -> 128 weight gradient, BF16 output, T 16380: nn_n128_hen_t_pro15, paired median 1.053 / 1.058
        ('sm_103a', True, True, False, True, True, 128, 16380, 512): {
            "b_swz": 128, "batch_group": 0, "block_n": 128, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 128 -> 12288 weight gradient, FP32 output, T 16380: nn_n224_f32_v8_ef_pro15, paired median 1.142
        ('sm_103a', True, True, True, False, False, 128, 16380, 12288): {
            "b_swz": 128, "batch_group": 0, "block_n": 224, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 512 -> 24576 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.320
        ('sm_103a', True, True, True, False, False, 512, 16380, 24576): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, FP32 output, T 32736: nn_n256_f32_tma1_cg_pro15, paired median 1.089
        ('sm_103a', True, True, True, False, False, 1024, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 1024 -> 4096 weight gradient, FP32 output, T 32768: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.108
        ('sm_103a', True, True, True, False, False, 1024, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 3584 -> 7168 weight gradient, FP32 output, T 16380: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.065
        ('sm_103a', True, True, True, False, False, 3584, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 4096 -> 8192 weight gradient, FP32 output, T 32736: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.105
        ('sm_103a', True, True, True, False, False, 4096, 32736, 8192): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 4096 -> 8192 weight gradient, FP32 output, T 32768: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.091
        ('sm_103a', True, True, True, False, False, 4096, 32768, 8192): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 896 weight gradient, FP32 output, T 16380: nn_n256_f32_v8_ef_sks_sb2_pro7, paired median 1.146
        ('sm_103a', True, True, True, False, False, 7168, 16380, 896): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 2, "sk_sync": True, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 1536 weight gradient, FP32 output, T 16380: nn_n160_m256_bz64_hne_f32_v8_ef_pro15, paired median 1.488
        ('sm_103a', True, True, True, False, False, 7168, 16380, 1536): {
            "b_swz": 64, "batch_group": 0, "block_n": 160, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": True, "form": "", "hints": ("none", "evict_first"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": True, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 3584 weight gradient, FP32 output, T 16380: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.061
        ('sm_103a', True, True, True, False, False, 7168, 16380, 3584): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, FP32 output, T 32736: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.092
        ('sm_103a', True, True, True, False, False, 8192, 32736, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 8192 -> 4096 weight gradient, FP32 output, T 32768: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.107
        ('sm_103a', True, True, True, False, False, 8192, 32768, 4096): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 576 weight gradient, FP32 output, T 16380: nn_n256_m64_hen_f32_t_pro15, paired median 1.102
        ('sm_103a', True, True, True, True, False, 576, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 2112 weight gradient, FP32 output, T 16380: nn_n256_m64_hen_f32_t_pro15, paired median 1.289
        ('sm_103a', True, True, True, True, False, 2112, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 64,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 96-head MLA 128 -> 512 weight gradient + 512 -> 128 weight gradient, FP32 output, T 16380: nn_n128_hen_f32_t_pro15, paired median 1.061 / 1.055
        ('sm_103a', True, True, True, True, True, 128, 16380, 512): {
            "b_swz": 128, "batch_group": 0, "block_n": 128, "cgrp": False, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("evict_first", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1536 -> 4096 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.041
        ('sm_103a', False, False, False, False, False, 4096, 1536, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 8192 forward, BF16 output, T 32736: kk_n256_ht_cg_pro15, paired median 1.029
        ('sm_103a', False, False, False, False, False, 8192, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 4096 -> 8192 forward, BF16 output, T 32768: kk_n256_ht_cg_pro15, paired median 1.024
        ('sm_103a', False, False, False, False, False, 8192, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1536 -> 18432 forward, BF16 output, T 16380: kk_n256_ht_cg_pro15, paired median 1.022
        ('sm_103a', False, False, False, False, False, 18432, 1536, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 input gradient, BF16 output, T 32736: kn_n256_ht_cg_pro15, paired median 1.017
        ('sm_103a', False, True, False, False, False, 1024, 4096, 32736): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1024 -> 4096 input gradient, BF16 output, T 32768: kn_n256_ht_cg_pro15, paired median 1.015
        ('sm_103a', False, True, False, False, False, 1024, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 1536 -> 18432 input gradient, BF16 output, T 16380: kn_n256_ht_cg_pro15, paired median 1.087
        ('sm_103a', False, True, False, False, False, 1536, 18432, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 3584 -> 7168 input gradient, BF16 output, T 16380: kn_n256_m256_ht_cg_or_pro7, paired median 1.038
        ('sm_103a', False, True, False, False, False, 3584, 7168, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 3584 input gradient, BF16 output, T 16380: kn_n256_ht_cg_pro15, paired median 1.022
        ('sm_103a', False, True, False, False, False, 7168, 3584, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 7168 -> 6144 input gradient, BF16 output, T 16380: kn_n256_m256_ht_cg_or_pro7, paired median 1.015
        ('sm_103a', False, True, False, False, False, 7168, 6144, 16380): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 256,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": True, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 8192 -> 4096 input gradient, BF16 output, T 32768: kn_n256_ht_cg_pro15, paired median 1.020
        ('sm_103a', False, True, False, False, False, 8192, 4096, 32768): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": True, "cta1": False, "cta_rows": 128,
            "epi": "reg", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": True, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 15, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 0, "sk_sync": False, "store_ef": False, "store_hint": "none",
            "stt": False, "sw": 0,
        },
        # 6144 -> 7168 weight gradient, FP32 output, T 16380: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.064
        ('sm_103a', True, True, True, False, False, 6144, 16380, 7168): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
        # 7168 -> 6144 weight gradient, FP32 output, T 16380: nn_n256_m256_f32_tma1_sks_sb3_pro7, paired median 1.065
        ('sm_103a', True, True, True, False, False, 7168, 16380, 6144): {
            "b_swz": 128, "batch_group": 0, "block_n": 256, "cgrp": False, "cta1": False, "cta_rows": 256,
            "epi": "tma", "f32_v8": False, "form": "", "hints": ("none", "none"), "htail": False, "lw": False,
            "mg": False, "ovl": False, "ovr": False, "park": False, "pd": 0, "pf": 0, "pro": 7, "quad_store": False,
            "sh": 0, "sk_first": False, "sk_slab": 3, "sk_sync": True, "slots": 1, "store_ef": False,
            "store_hint": "none", "stt": False, "sw": 0,
        },
    },
}
# fmt: on

# Package-only token ranges of ragged ``ROW_RULES`` keys that apply on a part of exactly one SM count,
# ``(arch, sm_count) -> {key: ((lo, hi, rule), ...)}`` (a ragged key as ``ROW_RULES``: ``K = None`` of a weight
# gradient, ``M = None`` of a forward / input gradient; the ranges of a key are sorted and disjoint).  At a token count
# T (``K`` of a weight-gradient key, ``M`` of a forward / input-gradient key) with ``lo <= T <= hi`` the range's rule
# replaces the row's rule on that part, as an exact ``ROW_RULES`` key at that T would (``row_rule`` still merges
# ``ARCH_RULE_DEFAULTS`` under it).  The architecture's other parts, its child architectures (``RULE_PARENT``) and
# every other token count keep the row's ``ROW_RULES`` entry and its ``ROW_RULE_MIN_T`` tiers, and the weight-gradient
# orientation reads ``ROW_RULES`` alone.  The 212-SM sm_107a entries serve every 212-SM part of the architecture.  The
# FP32 entries of version 0.3.5 were measured on two VR200 nodes against the plan the row had without them, at T 4096,
# 8192, 16172, 16231 and 16384, in eager launches and CUDA graph replays.  Each range ran from the lowest to the highest
# measured T at which the rule was faster on both nodes in both modes (the plan has the same form at the T between),
# within the naive BF16 tier and with the same bits run to run.  Version 0.3.8 extends four of these ranges to measured
# token counts next to them, with the same rule, and adds three ranges at T 12288 (measured as below).  The FP32
# ranges:
#   * the 12288 -> 4096 and 4096 -> 12288 weight gradients at T 4096 - 16384 (4096 -> 12288: T 8192 - 16231 up to
#     version 0.3.7): the synchronised split of the 26-tile tail (26 collectors, margin 38, slab path 2) through the
#     FP32 v8 evict-first register epilogue (``nn_n256_f32_v8_ef_sks_sb2_pro7``), 1.05 to 1.09 times faster in the
#     measurement of version 0.3.5, another summation order than the plan before;
#   * the 17408 -> 5120, 5120 -> 17408, dense_gate_up 6144 -> 12288 and dense_down 12288 -> 6144 weight gradients at
#     T 4096 - 8192 (5120 -> 17408 and 12288 -> 6144: T 8192 up to version 0.3.7): whole 128-row tiles through the FP32
#     v8 register epilogue (``nn_n256_f32_v8_pro15``), 1.02 to 1.17 times faster in the measurement of version 0.3.5,
#     the bits of the plan before;
#   * the o_proj 16384 -> 6144 weight gradient at T 4096 - 8192 (T 4096 and 8192 alone up to version 0.3.7): the same
#     program with a two-way K split of its 52 tail tiles, units of ceil(k / 2) K blocks laid in one line over the tail
#     (k K blocks per tile): at an even k 104 units, each the whole K half of a tile; at an odd k a unit runs on into
#     the next tail tile, 103 units at k 65 - 103 and 104 at k 105 - 127.  1.06 to 1.14 times faster at T 4096 and 8192
#     in the measurement of version 0.3.5; another summation order than the plan before;
#   * the 17408 -> 5120 and 5120 -> 17408 weight gradients at T 12288 (version 0.3.8): the half-height tail wave of the
#     256-row tiles (``nn_n256_m256_f32_ht_tma1_pro7``, 636 whole tiles and 88 half items of the 44-tile tail), the bits
#     of the plan before;
#   * the o_proj 16384 -> 6144 weight gradient at T 12288 (version 0.3.8): a three-way K split of the 26 tail tiles of
#     the 256-row tiles (``nn_n256_m256_f32_tma1_pro7``, 78 units of 64 K blocks), another summation order than the plan
#     before;
#   * the o_proj 16384 -> 6144 input gradient at T 8192 only: the half-height tail wave of the 128-row tiles
#     (``kn_n256_f32_v8_ht_pro7``, 68 half items of the 34-tile tail), 1.02 to 1.06 times faster, the bits of the
#     plan before.
# The FP32 range ends and ranges of version 0.3.8 were measured on two VR200 nodes against the plan of version 0.3.7,
# 11 paired rounds per node in eager launches and in CUDA graph replays.  A token count joins a range where the rule
# was faster in at least 9 of the 11 rounds with a median ratio of at least 1.015 on both nodes in both modes, gave the
# same output on both nodes and the same bits run to run, and, where it sums in another order, met the naive BF16 tier
# on five seeds of 32 draws.  The 4096 -> 12288 synchronised split: 1.090 to 1.137 times faster at T 4096 and 1.058 to
# 1.069 at T 16384; the whole tiles of 5120 -> 17408 and dense_down 12288 -> 6144 at T 4096: 1.121 to 1.160 and 1.112
# to 1.114; the o_proj two-way split at T 6208 and 8128 (k 97 and 127, the two odd-k unit counts): 1.031 to 1.082; at
# T 12288 the half-height tail wave 1.037 to 1.045 times faster and the o_proj three-way split 1.045 to 1.052.  At
# T 16172, 16231, 16384, 24576 and 32768 the rules of the T 12288 ranges did not meet that bar on these rows, which keep
# their plan there.
# The 148- and 152-SM sm_103a entries and the 152-SM sm_100a entries (version 0.3.8) are the FP32 input gradients of
# the 148-SM sm_100a part rules (``ROW_RULES_BY_SM`` above) with their rule: the single-CTA program with 128-column
# tiles, two TMA-store slots and four stages (``kn_n128_f32_tma2_s4_c1``), in the default raster group (16; 2 on
# 6144 -> 128, the group of the row's rule).  The plan before ran ``kn_n256_f32_tma2_pd2_pro15`` (6144 -> 128:
# ``kn_n128_f32_tma1_pro7``); both sum each output element's K in the same order, and the outputs were equal bit for
# bit at every measured token count.  Measured on two nodes of each part against that plan with the paired rounds
# above, each range covers one run of neighbouring measured T at which the rule met that bar, from the lowest to the
# highest of them (a range never spans a measured T at which the rule missed the bar or was not timed; K holds at most
# two K blocks, so the plan has the same form at every T):
#   * B300 (148-SM sm_103a), measured at T 8192, 16231, 16384 and 32768 (6144 -> 128: 8192, 16172, 16231 and 32768):
#     4096 -> 32 / 48 / 64, 5120 -> 64 and 6144 -> 128 at T 8192 - 32768, 4096 -> 128 and 5120 -> 32 at
#     T 8192 - 16384, 5120 -> 48 at T 16231 - 16384; 1.015 to 1.100 times faster.  5120 -> 128 and 8192 -> 128 met
#     the bar at no measured token count and keep their plan;
#   * GB300 (152-SM sm_103a), measured at T 16172, 16231 and 16384: 4096 -> 32 / 48 / 64 / 128,
#     5120 -> 32 / 48 / 64 / 128, 8192 -> 128 and 6144 -> 128 at T 16172 - 16384, 1.015 to 1.064 times faster;
#   * GB200 (152-SM sm_100a), measured at T 16172, 16231 and 16384 (the rule was not timed at T 16231 on
#     5120 -> 128 and 8192 -> 128): 4096 -> 32 / 48 / 64 / 128 and 5120 -> 32 / 48 / 64 at T 16172 - 16384,
#     5120 -> 128 at T 16172 and at T 16384 (two ranges of one token count each), 6144 -> 128 at T 16172 alone;
#     1.021 to 1.063 times faster.  6144 -> 128 missed the bar at T 16231 and 16384 (it met it in three of the four
#     node and mode pairs) and 8192 -> 128 at T 16172 and 16384: they kept their plan there up to version 0.3.10
#     (version 0.3.11 runs both in small raster groups there: below).
# Version 0.3.9 runs two GB300 (152-SM sm_103a) FP32 input gradients of that program in raster groups of 2 row tiles,
# measured on two GB300 nodes against the plan of version 0.3.8 (the same program in raster groups of 16) with the
# paired rounds above: the indexer_hw input gradient 6144 -> 32 (whose plan before is its ``ROW_RULES`` rule of
# sm_100a, read through ``RULE_PARENT``) at T 16172 - 32768, 1.016 to 1.022 times faster (measured at T 8192, 16172,
# 16231, 16384 and 32768; at T 8192 the group met the bar in three of the four node and mode pairs and the row keeps its
# plan there), and 8192 -> 128 at T 16231 - 16384, 1.063 to 1.071 times faster (at T 16172 the plan before was more
# than 1% off a second copy of itself in three of the four pairs, so T 16172 - 16230 keep the group of 16).  The raster
# group changes which tiles run side by side, not the sum of any tile: the outputs were equal bit for bit.
# Version 0.3.10 runs that program in small raster groups on B300 (148-SM sm_103a) and B200 (148-SM sm_100a) FP32 input
# gradients, measured at T 8192, 16231, 16384 and 32768 on two nodes of each part against the plan of version 0.3.9
# with the paired rounds above (groups of 2 and 4, of 4 and 8 on the B300 indexer_hw row; of two groups that met the
# bar the faster, the smaller within 0.005); the outputs were equal to that plan's bit for bit at every measured T:
#   * B300: 8192 -> 128 in groups of 2 at T 8192 - 32768, 1.027 to 1.052 times faster than the plan before
#     (``kn_n256_f32_tma2_pd2_pro15``); the indexer_hw input gradient 6144 -> 32 in groups of 4 at T 8192 alone, 1.015
#     to 1.018 times faster than the same program in groups of 16 (at T 16231 - 32768 the groups missed the bar).
#     5120 -> 128 keeps its plan: at T 8192 both groups met the bar, but the plan before was more than 1% off a second
#     copy of itself in one of the four node and mode pairs, and at T 16231 - 32768 they missed it;
#   * B200: 8192 -> 128 in groups of 2 at T 8192 - 32768 (instead of its part rules of groups of 16 at T 16231 and
#     16384 and ``kn_n256_f32_tma2_pd2_pro15`` elsewhere), 1.038 to 1.078 times faster; 5120 -> 128 and the indexer_k
#     input gradient 6144 -> 128 in groups of 2 at T 16384 - 32768 (plans before ``kn_n256_f32_tma2_pd2_pro15`` and
#     ``kn_n128_f32_tma1_pro7``), 1.020 to 1.027 and 1.015 to 1.025 times faster (at T 8192 and 16231 the groups
#     missed the bar, or on 6144 -> 128 at T 16231 are its part rule already); indexer_hw 6144 -> 32 in groups of 4 at
#     T 16231 and of 2 at T 32768, 1.015 to 1.019 and 1.019 to 1.020 times faster (at T 8192 and 16384 it missed it).
# Version 0.3.11 runs that program in small raster groups on GB200 (152-SM sm_100a) FP32 input gradients, measured at
# T 16172, 16231, 16384 and 32768 on two GB200 nodes against the plan of version 0.3.10 with the paired rounds and the
# bar of version 0.3.10 (groups of 2 and 4; of two groups that met the bar the one with the higher lowest node and
# mode median, the smaller within 0.005); the outputs were equal to that plan's bit for bit at every measured T on both
# nodes, also where the program changes (as in version 0.3.8, the plan before sums each output element's K in the
# same order):
#   * the indexer_hw input gradient 6144 -> 32 in groups of 2 at T 16172 - 32768, 1.030 to 1.042 times faster than
#     the same program in the default group of 16 (the row's ``ROW_RULES`` rule);
#   * 8192 -> 128 in groups of 4 at T 16172 and of 2 at T 16231 - 32768, 1.037 to 1.070 times faster than
#     ``kn_n256_f32_tma2_pd2_pro15``;
#   * 5120 -> 128 in groups of 2 at T 16231 alone, 1.055 to 1.064 times faster than ``kn_n256_f32_tma2_pd2_pro15``.
#     At T 16172 and 16384 (its ranges of version 0.3.8, the group of 16) the groups missed the bar (medians 1.006 to
#     1.019), and at T 32768 too (0.983 to 1.006): these keep their plan;
#   * the indexer_k input gradient 6144 -> 128 in groups of 2 at T 16231 - 16384, 1.017 to 1.024 times faster than
#     ``kn_n128_f32_tma1_pro7``.  At T 16172 its range of version 0.3.8 runs the group of 2 already; at T 32768 the
#     groups missed the bar (medians 1.012 to 1.016, not every node and mode pair a win) and the row keeps its plan.
# Version 0.3.12 runs the start-path eight-warp programs on the BF16 input gradients 4096 -> 32 / 48 / 64 / 128,
# 5120 -> 32 / 48 / 64 / 128, 8192 -> 128, indexer_hw 6144 -> 32 and indexer_k 6144 -> 128 on 148- and 152-SM sm_100a
# and sm_103a parts: ``kn_n256_tma1_w8relf`` (``_W8RELF_DGRAD_BF16``), on GB200 5120 -> 48 at T 32768
# ``kn_n256_tma1_w8rel`` (``_W8REL_DGRAD_BF16``).  They were measured at T 16172, 16231, 16384 and 32768 on two nodes of
# each part (B200, GB200, B300 and GB300) against the plan of version 0.3.11 with the paired rounds and the bar of
# version 0.3.10 (of two forms that met the bar the one with the higher lowest node and mode median, ``w8relf`` within
# 0.005); the outputs were equal to that plan's bit for bit at every measured T on both nodes.  A range runs from one
# measured T to another and holds no measured T that missed the bar; its plans were 1.021 to 1.091 times faster than
# the plan before on B200, 1.020 to 1.095 on GB200, 1.017 to 1.094 on B300 and 1.015 to 1.077 on GB300 (the lowest and
# highest node and mode medians).  On sm_100a a range replaces the GB T-exact rule of 4096 -> 48 (``_tma1_w8r``) at
# T 16231 on both parts and at T 16384 on 148 SMs (since version 0.3.20 on 152 SMs too: below).  A range's rule holds
# for its form: a caller that forces another form (the ten-warp program included) plans with the row's rule without
# the part tables, as in version 0.3.11.  Neither form met the bar at these, which keep their plan:
#   * B200: 4096 -> 64 at T 16172 and 16231 (medians 1.048 to 1.077, but a paired win in one or two of the four node
#     and mode pairs; version 0.3.14 runs T 16172 and version 0.3.15 T 16231 in the 128-column program: below), and
#     5120 -> 128, 8192 -> 128 and indexer_k at T 32768 (0.999 to 1.018);
#   * GB200: 4096 -> 48 and 5120 -> 128 at T 16384 (1.019 to 1.061, a paired win in three of the four pairs; version
#     0.3.20 runs 4096 -> 48 at T 16232 - 32767, T 16384 among them, and 5120 -> 128 at T 16384 in
#     ``kn_n256_tma1_w8relf``);
#   * B300: 4096 -> 48, 8192 -> 128 and indexer_k at T 32768 (1.000 to 1.013);
#   * GB300: 5120 -> 64 and 8192 -> 128 at every measured T (0.995 to 1.026), and 4096 -> 32 and indexer_k at T 32768
#     (1.001 to 1.017); 5120 -> 64 at T 16172 and 4096 -> 32 at T 32768 kept their plan up to version 0.3.12 (version
#     0.3.13 ran both in raster groups of 32 there), 5120 -> 64 at the other measured T up to version 0.3.13 (version
#     0.3.14 runs 5120 -> 64 at every measured T and 4096 -> 32 at T 32768 in the 128-column programs: below).
# Version 0.3.13 runs the start-path program ``kn_n256_tma1_w8relf`` in raster groups of 32 or 2 row tiles on three BF16
# input gradients of the 152-SM parts, each range at one measured T: the rule of version 0.3.12 (``_W8RELF_DGRAD_BF16``)
# with ``group_m`` (up to version 0.3.13: version 0.3.14 runs the 128-column programs at all four, below).  Measured at
# T 16172, 16231, 16384 and 32768 on two nodes of each part against the plan of version 0.3.12 with the paired rounds
# and the bar of version 0.3.10 (groups of 32 and 64 on GB300, of 2 and 4 on GB200; of two groups that met the bar the
# one with the higher lowest node and mode median, the smaller within 0.005); the outputs were equal to that plan's bit
# for bit at every measured T on both nodes, also where the program changes:
#   * GB300 (152-SM sm_103a): 4096 -> 32 in groups of 32 at T 16172, 1.019 to 1.029 times faster than the same program
#     in the default group of 16 (its range of version 0.3.12; T 16173 - 16384 keep it up to version 0.3.13,
#     T 16173 - 16383 in version 0.3.14, T 16232 - 16383 since), and at T 32768, 1.040 to 1.050 times faster than the
#     plan of version 0.3.11 (the row's ``ROW_RULES`` rule, a kept cell of version 0.3.12);
#     5120 -> 64 in groups of 32 at T 16172, 1.018 to 1.023 times faster than the plan of version 0.3.11 (the row had no
#     range of version 0.3.12 on this part).  4096 -> 32 at T 16231 and 16384 and 5120 -> 64 at T 16231, 16384 and
#     32768 keep their plan (all but 4096 -> 32 at T 16231 up to version 0.3.13, that one up to version 0.3.14:
#     below): the group of 32 had medians of 1.002 to 1.029 but a paired win in two or three of the four node and mode
#     pairs, the group of 64 medians of 0.979 to 1.020 and a paired win in at most one;
#   * GB200 (152-SM sm_100a): 5120 -> 32 in groups of 2 at T 32768, 1.015 to 1.021 times faster than the same program
#     in the default group of 16 (its range of version 0.3.12 keeps T 16172 - 32767 up to version 0.3.13,
#     T 16385 - 32767 since).  4096 -> 32 and the indexer_hw input gradient 6144 -> 32 at every measured T and
#     5120 -> 32 at T 16172, 16231 and 16384 keep their plan (all but 4096 -> 32 at T 16231 up to version 0.3.13,
#     that one up to version 0.3.14: below): the groups had medians of 0.999 to 1.023 and a paired win in at most three
#     of the four pairs.
# Version 0.3.14 runs the start-path program on its 128-column tile on BF16 input gradients of the 148- and 152-SM
# parts: ``kn_n128_tma1_w8relf`` (``_W8RELF_N128_DGRAD_BF16``, the 128 x 128 pair tile in 64-element K steps) on
# 4096 -> 64 and 5120 -> 64, ``kn_n128_k32_tma1_w8relf`` (``_W8RELF_N128_K32_DGRAD_BF16``, the same tile in 32-element K
# steps, one per item) on 4096 -> 32, 5120 -> 32 and the indexer_hw input gradient 6144 -> 32, each range in the raster
# group it was measured in (``group_m`` 2, 16 or 32, written in every range).  Both programs sum each output element's
# K in one FP32 accumulator in K order, as ``kn_n256_tma1_w8relf`` does.  Measured at T 16172, 16231, 16384 and 32768
# on two nodes of each part against the plan of version 0.3.13 with the paired rounds and the bar of version 0.3.10
# (the 64-element program on the -> 64 rows and the 32-element one on the -> 32 rows, in groups of 16 and 32 on GB300
# and B300, of 2 and 16 on GB200 and B200, on the B200 -> 64 rows of 8 too; of two groups that met the bar the one
# with the higher lowest node and mode median, the smaller within 0.005); the outputs were equal to that plan's bit
# for bit at every measured T on both nodes, also where the plan before runs another program.  A range runs over
# neighbouring measured T with the same group, from the lowest to the highest of them, and holds no measured T that
# missed the bar; the other token counts of these rows keep their rule of version 0.3.13.  The four ranges of version
# 0.3.13 are among them (GB300 5120 -> 64 at T 16172 in groups of 16 instead of 32).  The ranges, with the lowest and
# highest node and mode medians against the plan before:
#   * GB300 (152-SM sm_103a): 4096 -> 32 in groups of 32 at T 16172 (1.032 to 1.043) and 32768 (1.026 to 1.030) and
#     of 16 at T 16384 (1.057 to 1.067); 4096 -> 64 in groups of 32 at T 16172 (1.032 to 1.046) and 32768 (1.018 to
#     1.026); in groups of 16 at T 16172 - 32768 5120 -> 32 (1.023 to 1.042), 5120 -> 64 (1.028 to 1.062; above
#     T 16172 the plan before is the row's ``ROW_RULES`` rule, the single-CTA program) and indexer_hw (1.022 to 1.044);
#   * GB200 (152-SM sm_100a): 4096 -> 32 in groups of 2 at T 16172 (1.051 to 1.060) and 16384 - 32768 (1.042 to
#     1.065); 4096 -> 64 in groups of 16 at T 16172 (1.020 to 1.028) and 32768 (1.025 to 1.031); 5120 -> 32 in groups
#     of 16 at T 16172 - 16384 (1.037 to 1.053) and of 2 at T 32768 (1.023 to 1.024); 5120 -> 64 in groups of 2 at
#     T 16172 (1.017 to 1.022) and of 16 at T 16231 - 16384 (1.025 to 1.044); indexer_hw in groups of 16 at
#     T 16172 - 32768 (1.022 to 1.048);
#   * B300 (148-SM sm_103a): in groups of 16 4096 -> 32 at T 16384 - 32768 (1.020 to 1.027), 5120 -> 32 at
#     T 16172 - 32768 (1.018 to 1.026) and indexer_hw at T 16231 - 32768 (1.018 to 1.028);
#   * B200 (148-SM sm_100a): 4096 -> 32 in groups of 16 at T 32768 (1.020 to 1.024); 4096 -> 64 in groups of 2 at
#     T 16172 (1.044 to 1.057; the plan before is the row's ``ROW_RULES`` rule, the ``w8r`` program in groups of 8);
#     in groups of 16 5120 -> 32 at T 16172 - 32768 (1.018 to 1.031) and indexer_hw at T 16172 (1.017 to 1.019) and
#     T 16384 - 32768 (1.017 to 1.027).
# No group met the bar at these, which keep their plan of version 0.3.13 (six of them up to version 0.3.14, one more up
# to version 0.3.18: below):
#   * GB300: 4096 -> 32 at T 16231 (medians 1.053 to 1.063, a paired win in two of the four node and mode pairs) and
#     4096 -> 64 at T 16231 and 16384 (1.029 to 1.044, a paired win in two or three; version 0.3.15 runs all three in
#     the 128-column programs);
#   * GB200: 4096 -> 32 at T 16231 (1.046 to 1.064, a paired win in two), 4096 -> 64 at T 16231 and 16384 (1.000 to
#     1.040, in at most three) and 5120 -> 64 at T 32768 (1.007 to 1.019, in at most two; version 0.3.15 runs
#     4096 -> 32 at T 16231 and 4096 -> 64 at T 16384 in the 128-column programs, version 0.3.19 4096 -> 64 at
#     T 16231);
#   * B300: 4096 -> 32 at T 16172 and 16231 and indexer_hw at T 16172 (1.009 to 1.020, a paired win in at most two),
#     and 4096 -> 64 and 5120 -> 64 at every measured T (0.956 to 1.009, no paired win);
#   * B200: 4096 -> 32 at T 16172, 16231 and 16384 (0.996 to 1.022, a paired win in at most two), indexer_hw at
#     T 16231 (0.989 to 1.019, in at most two), 4096 -> 64 at T 16231 (1.050 to 1.065, in two; version 0.3.15 runs it in
#     the 128-column program) and at T 16384 and 32768 (0.966 to 1.005, no paired win), and 5120 -> 64 at every
#     measured T (0.958 to 1.006, no paired win).
# Version 0.3.15 runs the 128-column programs at six of the cells version 0.3.14 kept, each in the raster group it was
# confirmed in (``group_m``, written in every range), and the unmeasured token counts between such a cell and its
# neighbouring range of version 0.3.14 with them.  A confirmation round on two nodes of each part, in eager launches
# and CUDA graph replays, timed each candidate in 21 paired rounds against the plan of version 0.3.14 and against a
# second prepared copy of that plan in the same rounds; a cell met the bar with at least 17 of the 21 paired wins
# against each and both medians at least 1.015 in every node and mode pair, and its outputs were equal to that plan's
# bit for bit on both nodes and the same on every run.  The cells, with the lowest and highest node and mode medians
# (each the lower of the two against the plan before and its copy):
#   * GB300 (152-SM sm_103a): 4096 -> 32 in 32-element K steps in groups of 32 at T 16231 (1.042 to 1.047); 4096 -> 64
#     in 64-element K steps in groups of 32 at T 16231 (1.036 to 1.041) and 16384 (1.023 to 1.034);
#   * GB200 (152-SM sm_100a): 4096 -> 32 in groups of 2 at T 16231 (1.028 to 1.051; the group of 16, 1.030, was within
#     0.005: the smaller group); 4096 -> 64 in groups of 16 at T 16384 (1.023 to 1.029);
#   * B200 (148-SM sm_100a): 4096 -> 64 in groups of 2 at T 16231 (1.032 to 1.054; the groups of 16 and 8, 1.032 and
#     1.030, were within 0.005: the smallest group; the plan before is the row's ``ROW_RULES`` rule, the ``w8r``
#     program in groups of 8).
# A cell joins the range of version 0.3.14 of its row and part where the neighbouring measured T runs the same program
# in the same group, else it is a range of its own; all six join, and the unmeasured T inside a joined range run its
# program.  The joined ranges: GB300 4096 -> 32 in groups of 32 at T 16172 - 16231 and 4096 -> 64 in groups of 32 at
# T 16172 - 32768; GB200 4096 -> 32 in groups of 2 at T 16172 - 32768 and 4096 -> 64 in groups of 16 at
# T 16384 - 32768; B200 4096 -> 64 in groups of 2 at T 16172 - 16231.  The other token counts of these rows plan as in
# version 0.3.14: GB300 4096 -> 32 at T 16232 - 16383 and GB200 4096 -> 64 at T 16173 - 16383 the 256-column program
# (the GB200 row up to version 0.3.18: below), B200 4096 -> 64 at T 16232 - 16383 the row's ``ROW_RULES`` rule.
# Version 0.3.17 runs the 128-column start-path programs on five BF16 input gradients of the 212-SM sm_107a part
# (VR200), which planned the ten-warp programs before (``kn_n256_ht_tma2_lw_pro15`` on 4096 -> 32 and 5120 -> 32,
# ``kn_n256_tma2_pd3_c1`` on the indexer_hw input gradient 6144 -> 32, 4096 -> 64 and 5120 -> 64): the 32-element
# program on the -> 32 rows and the 64-element one on the -> 64 rows, each range in the raster group it was measured in.
# The four start-path programs serve sm_107a with the builds they ship on sm_100a and sm_103a.  Measured at T 16172,
# 16231 and 16384 on two VR200 nodes with the confirmation round and the bar of version 0.3.15 against the plan of
# version 0.3.16 (``kn_n256_tma1_w8relf`` in groups of 16, the 128-column program in groups of 16 and 32; of the
# programs that met the bar the one with the higher lowest node and mode median, ``kn_n256_tma1_w8relf`` and then the
# smaller group within 0.005); every candidate met the bar at every measured T, and the outputs were equal to that
# plan's bit for bit on both nodes.  The ranges, with the lowest and highest node and mode medians against the plan
# before:
#   * in groups of 16 at T 16172 - 16384: 4096 -> 32 (1.132 to 1.161), 5120 -> 32 (1.107 to 1.146) and indexer_hw (1.075
#     to 1.135) in 32-element K steps, and 4096 -> 64 (1.084 to 1.117; at T 16231 the group of 32, 1.100, was within
#     0.005: the smaller group) in 64-element K steps;
#   * 5120 -> 64 in 64-element K steps in groups of 16 at T 16172 - 16231 (1.087 to 1.105) and of 32 at T 16384 (1.090
#     to 1.098).
# The unmeasured token counts inside a range run its program; 5120 -> 64 at T 16232 - 16383 and the other token counts
# of these rows keep their plan of version 0.3.16.
# Version 0.3.19 runs the 128-column program in 64-element K steps in raster groups of 16 on the GB200 (152-SM sm_100a)
# BF16 input gradient 4096 -> 64 at T 16173 - 16383 too, where version 0.3.18 ran the 256-column program
# ``kn_n256_tma1_w8relf`` in the default group of 16 (the rule of version 0.3.12): the row's three ranges become one,
# T 16172 - 32768 in groups of 16.  The two programs give the same bits (every output element sums its K in one FP32
# accumulator in K order).  Measured at T 16231 with cold L2 in eager launches and CUDA graph replays, 21 paired
# rounds per node and mode against the plan of version 0.3.18 and a second prepared copy of it in the same rounds
# (driver 580.126.20, CUDA 13.5, PyTorch 2.14.1+cu130): on two GB200 nodes the 128-column program was faster than the
# plan in all 21 rounds of each of the four node and mode pairs (than its copy in 21, 20, 21 and 21), medians 1.011 to
# 1.028; a confirmation round on two other GB200 nodes, with its bar fixed before timing, found it faster in all 21
# rounds of every node and mode pair against each copy, medians 1.029 to 1.039 against the plan and 1.026 to 1.032
# against its copy, and its outputs equal to that plan's bit for bit.  Median times there: the plan of version 0.3.18
# 23.06 to 23.33 us, the 128-column program 22.38 to 22.56 us, cuBLASLt 22.80 to 22.93 us and the cuDNN frontend 23.36
# to 23.47 us (the faster library takes 1.016 to 1.019 times the time of the new plan, 0.980 to 0.989 times that of
# the plan before).  T 16172 and T 16384 ran this rule already (versions 0.3.14 and 0.3.15); the unmeasured
# T 16173 - 16230 and 16232 - 16383 run it with T 16231.
# Version 0.3.20 runs the start-path programs on the other six BF16 input gradients of the 212-SM sm_107a part (VR200),
# which planned the ten-warp programs of their ``ROW_RULES`` rules up to version 0.3.19 (``kn_n256_ht_tma2_lw_pro15`` on
# 4096 -> 48, ``kn_n256_tma2_pd2_lw_pro15`` on 5120 -> 48, ``kn_n256_tma2_pd2_pro15`` on 4096 -> 128, 5120 -> 128,
# 8192 -> 128 and the indexer_k input gradient 6144 -> 128), each row in one range at T 16172 - 16384 in raster groups
# of 16: ``kn_n256_tma1_w8relf`` on 4096 -> 48, 4096 -> 128, 5120 -> 48, 8192 -> 128 and indexer_k, and the 128-column
# program in 64-element K steps (``kn_n128_tma1_w8relf``) on 5120 -> 128.
# Measured at T 16172, 16231 and 16384 with cold L2 in eager launches and CUDA graph replays, 21 paired rounds per node
# and mode against the plan of version 0.3.19 and a second prepared copy of it in the same rounds, on two VR200 nodes
# (drivers 620.43 and 620.70.06, CUDA 13.5): ``kn_n256_tma1_w8relf`` in groups of 16 on every row, the 128-column
# program in 64-element K steps in groups of 16 and 32 on the -> 128 rows and in 32-element K steps in groups of 16 on
# the -> 48 rows.  A candidate met the bar at a token count with at least 17 of the 21 paired wins against each copy and
# both medians at least 1.0 in every node and mode pair (the two copies within 0.01 of each other), its outputs within
# the naive BF16 tier on five seeds and of one bit pattern on every run; a row takes, over its three measured token
# counts, the candidate that met the bar at all three with the higher lowest node and mode median (each the lower of
# the two against the plan before and its copy), within 0.005 ``kn_n256_tma1_w8relf`` and then the smaller group.  The
# outputs are not held to the bits of the plan before; at the measured token counts they were equal to its bits on
# both nodes.  The ranges, with the lowest and highest node and mode medians against the plan before (each new plan
# faster in all 21 rounds of every node and mode pair against both copies):
#   * ``kn_n256_tma1_w8relf`` in groups of 16: 4096 -> 48 (1.091 to 1.119), 4096 -> 128 (1.072 to 1.082), 5120 -> 48
#     (1.063 to 1.085), 8192 -> 128 (1.030 to 1.053) and indexer_k (1.047 to 1.059);
#   * the 128-column program in groups of 16: 5120 -> 128 (1.053 to 1.077; the group of 32 had the higher lowest
#     median, 1.055 against 1.051, within 0.005: the smaller group).
# Median times at T 16231 in us (the plan before, the new plan, cuBLASLt and the cuDNN frontend, the lowest and highest
# node and mode medians), and the fastest library's time over the new plan's:
#   * 4096 -> 48: 18.14 to 18.27, 16.21 to 16.48, 16.21 to 16.64 and 16.24 to 16.75 (1.000 to 1.010);
#   * 4096 -> 128: 18.02 to 21.73, 16.70 to 20.24, 16.90 to 20.69 and 16.85 to 20.69 (1.009 to 1.022);
#   * 5120 -> 48: 20.37 to 20.59, 18.81 to 19.25, 18.88 to 19.46 and 18.91 to 19.42 (1.003 to 1.009);
#   * 5120 -> 128: 21.38 to 22.98, 20.30 to 21.66, 20.10 to 21.95 and 20.18 to 22.02 (0.990 to 1.013);
#   * indexer_k: 26.54 to 26.69, 25.17 to 25.40, 25.95 to 26.21 and 25.52 to 26.12 (1.014 to 1.028);
#   * 8192 -> 128: 29.86 to 34.00, 28.72 to 32.88, 29.97 to 34.74 and 29.63 to 34.75 (1.032 to 1.056).
# Over the three measured token counts the fastest library took 0.957 to 1.010 times the time of the new plan on
# 4096 -> 48, 0.995 to 1.027 on 4096 -> 128, 0.991 to 1.009 on 5120 -> 48 and 0.990 to 1.025 on 5120 -> 128 (on these
# four rows faster than the new plan at some token count and node and mode pair), 1.014 to 1.042 on indexer_k and 1.019
# to 1.059 on 8192 -> 128.  The unmeasured token counts inside a range run its program; the other token counts of these
# rows keep their plan of version 0.3.19.  The non-deterministic package keeps its entries of these rows at T 16172,
# 16231 and 16384 (its program on the plan of ``kn_n256_tma1_w8r``, a form these ranges do not name; the round timed it
# beside the new plan, its median time 0.987 to 1.020 times the new plan's); at the token counts between them it runs
# this plan.
# Version 0.3.20 also runs ``kn_n256_tma1_w8relf`` in the default raster group of 16 (the rule of version 0.3.12) on the
# GB200 (152-SM sm_100a) BF16 input gradient 4096 -> 48 at T 16232 - 32767, where version 0.3.19 ran the row's
# ``ROW_RULES`` rule, the half-tail program ``kn_n256_ht_tma2_lw_pro15``, and at T 16384 its GB T-exact rule, the
# eight-warp program ``kn_n256_tma1_w8r`` (a kept cell of version 0.3.12): the row's two ranges become one,
# T 16172 - 32768.  Measured at T 16300, 16384 and 24576 with cold L2 (CUPTI) in eager launches and CUDA graph replays,
# 21 paired rounds per node and mode against the plan of version 0.3.19 on two GB200 nodes (driver 580.126.20): medians
# 1.053 to 1.061 at T 16300, 1.027 to 1.038 at T 16384 (faster in all 21 rounds of each of the four node and mode pairs)
# and 1.035 to 1.038 at T 24576, and the outputs equal to that plan's bit for bit at the three token counts on both
# nodes.  Median times (the plan before, the new plan, the cuDNN frontend and cuBLASLt): 24.18 to 24.59, 22.78 to 23.34,
# 23.44 to 23.84 and 23.50 to 24.00 us at T 16300; 23.42 to 23.98, 22.78 to 23.20, 23.47 to 23.84 and 23.50 to 23.97 us
# at T 16384; 34.13 to 34.86, 32.99 to 33.73, 35.18 to 35.71 and 34.11 to 34.59 us at T 24576.  The fastest library (the
# cuDNN frontend at T 16300 and 16384, cuBLASLt at T 24576) takes 1.021 to 1.029, 1.021 to 1.033 and 1.026 to 1.034
# times the time of the new plan at the three token counts, against 0.966 to 0.970, 0.993 to 1.002 and 0.992 to 1.000
# times that of the plan before.  The unmeasured token counts of T 16232 - 32767 run the range's program.  The
# ``ROW_RULES`` key at T 16384 stays: sm_100a parts without a range of this row (SM counts other than 148 and 152) plan
# it, and so does a caller that forces the ``w8r`` form.  The non-deterministic package holds no entry of this row at
# T 16232 - 32767, so its call there runs this plan.  A second pair of GB200 nodes repeated this measurement: medians
# 1.054 to 1.059 at T 16300, 1.024 to 1.028 at T 16384 and 1.032 to 1.044 at T 24576, faster in all 21 rounds of every
# node and mode pair, with the bits of the plan before.
# Version 0.3.20 also runs ``kn_n256_tma1_w8relf`` in the default raster group of 16 on the GB200 BF16 input gradient
# 5120 -> 128 at T 16384 alone, a range of one token count between the row's ranges of version 0.3.12 (T 16172 - 16231
# and 32768), where version 0.3.19 ran the row's ``ROW_RULES`` rule, the ten-warp program
# ``kn_n256_tma2_pd2_lw_pro15`` (a kept cell of version 0.3.12).  Measured at T 16384 with cold L2 (CUPTI) in eager
# launches and CUDA graph replays, 21 paired rounds per node and mode against the plan of version 0.3.19 and a second
# prepared copy of it in the same rounds, with the bar fixed before timing (at least 17 of the 21 paired wins against
# each copy, both medians at least 1.0 and the two copies within 0.01 of each other in every node and mode pair): on
# two GB200 nodes (driver 580.126.20) the new plan was faster in all 21 rounds of each of the four node and mode pairs
# against each copy, medians 1.050 to 1.056 against the plan and 1.043 to 1.049 against its copy, and its outputs met
# the naive BF16 tier on five seeds and were equal to that plan's bit for bit on both nodes.  An earlier pair of GB200
# nodes measured 1.053 to 1.067 against the plan and 1.048 to 1.055 against its copy, but on one of them the two copies
# of the plan were more than 0.01 apart, which the bar does not accept; the second pair decided.  Median times (the plan
# before, the new plan, the cuDNN frontend and cuBLASLt): 30.83 to 31.18, 29.42 to 29.58, 31.41 to 31.55 and 31.36 to
# 31.46 us; the fastest library takes 1.063 to 1.067 times the time of the new plan, against 1.007 to 1.019 times that
# of the plan before.  The other token counts of the row keep their plan of version 0.3.19.  The non-deterministic
# package drops its entry at this key (this package's program ``kn_n256_tma1_w8r``, 30.59 to 30.78 us, 1.039 to 1.041
# times the time of the new plan and slower in all 21 rounds of every node and mode pair), so its call there runs this
# plan.
# The BF16 entries (version 0.3.7) are K-major forward and input-gradient rows with K > 8192 at small token counts,
# judged on VR200 on five seeds of 32 draws against the naive BF16 tier and timed against the plan the row had without
# them (geometric mean of eager launches and CUDA graph replays, cold L2).  That plan sums 136 to 256 K blocks of a tile
# in one FP32 accumulator there and missed the tier on 4 of the 560 cells of the first pass at T 1 - 3072 (a cell: a row
# at one token count on one seed; later passes found no other miss): the o_proj forward at T 16 and 33 and the
# 4096 -> 12288 input gradient at T 1 and 64, errors up to 1.0058 times the tier's (the tests allow 1.005).  Every range
# rule met the tier on every judged seed and gives the same bits on every run; a range plan that splits K where the plan
# before did not, or in more parts, sums in another order:
#   * the o_proj 16384 -> 6144 forward: a four-way K split of the tail tiles at T 1 - 256 (64-block segments), 1.81 to
#     2.25 times faster; at T 257 - 1024 the 128-row program with the column-grouped raster of the row's rule
#     (8-column-tile groups), 1.59 to 1.67 times faster up to T 512 (a two-way K split of the single partial wave,
#     128-block segments) and 1.17 to 1.22 from T 513 (whole tiles);
#   * the 17408 -> 5120 forward and the 5120 -> 17408 input gradient (rows without a ``ROW_RULES`` entry on sm_107a; the
#     range key is the ragged key all the same): a four-way K split of the tail tiles at T 16 - 256 (68-block segments),
#     1.08 to 1.13 and 1.14 to 1.21 times faster; at T 1 the split was slower (0.94 and 0.97 times as fast), so the rows
#     keep their plan at T 1 and at the unmeasured T 2 - 15 between it and the range.  The plan before met the tier on
#     every judged cell of these two rows: their ranges are taken for speed alone;
#   * the 4096 -> 12288 input gradient: a four-way K split of the tail tiles at T 1 - 256 (48-block segments), 1.05 to
#     1.17 times faster from T 16 and 0.89 times as fast at T 1, where the plan before missed the tier.
# The 208-, 216- and 224-SM sm_107a parts, the sm_103a parts of other SM counts and the sm_100a parts of other SM
# counts than 148 and 152 are not measured and plan these rows as before.
# fmt: off
_SM107A_212_F32_SKS = {"epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_sync_m": 38, "sk_slab": 2}
_SM107A_212_F32_V8 = {"epi": 'reg', "f32_v8": True, "pro": 15}
_SM107A_212_F32_V8_SK2 = {"epi": 'reg', "f32_v8": True, "sk_parts": 2, "pro": 15}
_SM107A_212_F32_M256_HT = {"cta_rows": 256, "htail": True}
_SM107A_212_F32_M256_SK3 = {"cta_rows": 256, "sk_parts": 3}
_W8REL_DGRAD_BF16 = {"form": 'w8rel', "epi": 'tma', "slots": 1, "sk": False}
_W8RELF_DGRAD_BF16 = {"form": 'w8relf', "epi": 'tma', "slots": 1, "sk": False}
# the same start-path program in 128-column pair tiles (64-element K steps), and in 128-column tiles with 32-element K
# steps: the bits of ``_W8RELF_DGRAD_BF16`` (every output element sums its K in one FP32 accumulator in K order); each
# range of version 0.3.14 writes the raster group it was measured in, ``{**rule, "group_m": g}`` (16 too)
_W8RELF_N128_DGRAD_BF16 = {**_W8RELF_DGRAD_BF16, "block_n": 128}
_W8RELF_N128_K32_DGRAD_BF16 = {**_W8RELF_N128_DGRAD_BF16, "block_k": 32}
ROW_RULE_RANGES_BY_SM: dict[tuple[str, int], dict[tuple, tuple[tuple[int, int, dict], ...]]] = {
    ("sm_107a", 212): {
        ('sm_107a', True, True, True, False, False, 12288, None, 4096): ((4096, 16384, _SM107A_212_F32_SKS),),
        ('sm_107a', True, True, True, False, False, 4096, None, 12288): ((4096, 16384, _SM107A_212_F32_SKS),),
        ('sm_107a', True, True, True, False, False, 17408, None, 5120): ((4096, 8192, _SM107A_212_F32_V8), (12288, 12288, _SM107A_212_F32_M256_HT)),
        ('sm_107a', True, True, True, False, False, 5120, None, 17408): ((4096, 8192, _SM107A_212_F32_V8), (12288, 12288, _SM107A_212_F32_M256_HT)),
        ('sm_107a', True, True, True, False, False, 16384, None, 6144): ((4096, 8192, _SM107A_212_F32_V8_SK2), (12288, 12288, _SM107A_212_F32_M256_SK3)),
        ('sm_107a', True, True, True, False, False, 6144, None, 12288): ((4096, 8192, _SM107A_212_F32_V8),),
        ('sm_107a', True, True, True, False, False, 12288, None, 6144): ((4096, 8192, _SM107A_212_F32_V8),),
        ('sm_107a', False, True, True, False, False, 16384, 6144, None): ((8192, 8192, {"epi": 'reg', "f32_v8": True, "htail": True, "pro": 7}),),
        ('sm_107a', False, False, False, False, False, 6144, 16384, None): ((1, 256, {"pro": 15, "sk_parts": 4}), (257, 1024, {"cgrp": True, "pro": 15, "group_m": 8})),
        ('sm_107a', False, False, False, False, False, 5120, 17408, None): ((16, 256, {"pro": 15, "sk_parts": 4}),),
        ('sm_107a', False, True, False, False, False, 5120, 17408, None): ((16, 256, {"pro": 7, "sk_parts": 4}),),
        ('sm_107a', False, True, False, False, False, 4096, 12288, None): ((1, 256, {"pro": 7, "sk_parts": 4}),),
        ('sm_107a', False, True, False, False, False, 4096, 32, None): ((16172, 16384, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 5120, 32, None): ((16172, 16384, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 6144, 32, None): ((16172, 16384, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 4096, 64, None): ((16172, 16384, {**_W8RELF_N128_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 5120, 64, None): (
            (16172, 16231, {**_W8RELF_N128_DGRAD_BF16, "group_m": 16}),
            (16384, 16384, {**_W8RELF_N128_DGRAD_BF16, "group_m": 32}),
        ),
        ('sm_107a', False, True, False, False, False, 4096, 48, None): ((16172, 16384, {**_W8RELF_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 4096, 128, None): ((16172, 16384, {**_W8RELF_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 5120, 48, None): ((16172, 16384, {**_W8RELF_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 5120, 128, None): ((16172, 16384, {**_W8RELF_N128_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 6144, 128, None): ((16172, 16384, {**_W8RELF_DGRAD_BF16, "group_m": 16}),),
        ('sm_107a', False, True, False, False, False, 8192, 128, None): ((16172, 16384, {**_W8RELF_DGRAD_BF16, "group_m": 16}),),
        # version 0.3.23: the 4096 -> 16 BF16 weight gradient plans its three-way 64-row rule (``ROW_RULES``) below the
        # key's floor (T 4096) on this part, from T 3073 on since version 0.3.25 (T 3072 in versions 0.3.23 - 0.3.24,
        # where the 64-row whole tiles of version 0.3.25 now run): measured faster than the plan of version 0.3.22 at
        # T 3072 here (post-ship medians of that plan over the rule 1.021 / 1.024, eager / CUDA graph); on the 216-SM
        # part it was flat in eager launches and slower in CUDA graph replays.  The generic call swaps here in every
        # version since 0.3.24; off this part it follows the weight-gradient entry at T 3073 - 4095 since version 0.3.26
        # (``_SINGLE_WGRAD_FOLLOW_MIN_K``)
        ('sm_107a', True, True, False, True, False, 16, None, 4096): ((3073, 4095, {"block_n": 128, "cta_rows": 64, "sk_exact": 3, "pro": 15, "hints": ('evict_first', 'none')}),),
        # version 0.3.27: the FP32 outputs of the thirteen weight gradients of version 0.3.27 (``ROW_RULES``; the BF16
        # outputs plan their ``ROW_RULE_MIN_T`` tiers) plan the 64-row whole tiles without a K split at T 128 - 4096
        # (4096 -> 16: T 128 - 3072) on this part and on the 216-SM part, the FP32 program
        # ``dense_proj_det_gemm_nn_n128_m64_hen_f32_t_pro15`` (the plan of version 0.3.26: ``..._nn_n128_f32_t_pro7``,
        # whole 128-row tiles up to 15 K blocks and a two-way K split from 16 on).  Nine of these rows have no
        # token-free (ragged) ``ROW_RULES`` key (such a key would change the orientation of their generic call), so
        # these ranges hold the band on the two measured parts alone; the 208- and 224-SM parts keep the plan of
        # version 0.3.26
        ('sm_107a', True, True, True, True, False, 16, None, 4096): ((128, 3072, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 32, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 48, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 64, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 96, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 128, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 48, None, 5120): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 64, None, 5120): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 128, None, 5120): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 32, None, 6144): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 48, None, 6144): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 64, None, 6144): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 128, None, 8192): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
    },
    # The 216-SM part (version 0.3.23).
    # - The indexer_hw BF16 weight gradient plans the 64-row whole tiles at T 128 - 3584 (versions 0.3.25 - 0.3.26:
    #   T 960 - 3584; its ``ROW_RULE_MIN_T`` middle tier, here as a range: in version 0.3.25 either start, T 960 or
    #   1024, gave the same plan on this part, and T 960 made the range state the whole band; since version 0.3.27 the
    #   tier and the range both start at T 128) and its 64-row exact two-way split (``ROW_RULES``) at
    #   T 3585 - 15360 (T 1024 - 15360 in versions 0.3.23 - 0.3.24).  There the split measured faster than its plan of
    #   version 0.3.21 at every measured token count (T 1024 - 8192 and 8193 - 15360, medians of that plan over the
    #   split 1.03 to 1.33 at T 8193 - 15360 before shipping; post-ship 1.018 to 1.332 / 1.018 to 1.374 (eager / CUDA
    #   graph) at the timed T 1024 - 15360), and the whole tiles faster than the plan of version 0.3.24 at every
    #   measured token count of T 960 - 3584 before shipping (version 0.3.25, ``ROW_RULES``, where the post-ship A/B
    #   on the 212-SM part ends the band at T 3584 on every sm_107a part), and at every measured token count of
    #   T 128 - 959 before shipping (version 0.3.27, ``ROW_RULES``).  From T 15361 on it keeps its
    #   plan of version 0.3.21, the tier rule ``{"block_n": 128, "sk_exact": 3}``: the split measured slower at T 16171,
    #   16172, 32768 and 65536.  The unmeasured T 15361 - 16170, between the last measured win and the first measured
    #   loss, keep that rule too.
    # - The 6144 -> 48 BF16 weight gradient plans the four-way exact split of its 24 128-row tiles at T 65536 - 131072
    #   (``cta_rows`` 128 and ``pro`` 7 by default, the hint from the 128-row gate:
    #   ``dense_proj_det_gemm_nn_n128_hen_t_skx_kc64_pro7``): it measured faster than its two-way 64-row rule at T 65536,
    #   98304 and 131072 here (post-ship medians of that rule over the four-way split 1.023 / 1.022, 1.036 / 1.035 and
    #   1.054 / 1.055 at these T, eager / CUDA graph) and slower on the 212-SM part.  Token counts above 131072 keep the
    #   rule.
    # - The generic call follows both: it swaps at every token count on indexer_hw (no ``_SINGLE_WGRAD_SWAP_MIN_K``
    #   entry) and on 6144 -> 48 where the rule is not empty (from T 960 on since version 0.3.25, from T 1024 on in
    #   version 0.3.24; from T 8193 on in version 0.3.23), and since version 0.3.26 at every token count (from T 1
    #   on, ``_SINGLE_WGRAD_FOLLOW_MIN_K``).
    # - The other three rows of version 0.3.22 plan their tiers on this part.
    # - The FP32 outputs of the thirteen weight gradients of version 0.3.27 plan the 64-row whole tiles at
    #   T 128 - 4096 (4096 -> 16: T 128 - 3072), as on the 212-SM part (``ROW_RULES``, version 0.3.27).
    # - Deleting this table puts the tiers back on this part (and the FP32 plans of version 0.3.26 on those thirteen
    #   rows); the tests that pin the table change with it.
    ("sm_107a", 216): {
        ('sm_107a', True, True, False, True, False, 32, None, 6144): ((128, 3584, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}), (3585, 15360, {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15, "hints": ('evict_first', 'none')}), (15361, 2**31 - 1, {"block_n": 128, "sk_exact": 3})),
        ('sm_107a', True, True, False, True, False, 48, None, 6144): ((65536, 131072, {"block_n": 128, "sk_exact": 4}),),
        ('sm_107a', True, True, True, True, False, 16, None, 4096): ((128, 3072, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 32, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 48, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 64, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 96, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 128, None, 4096): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 48, None, 5120): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 64, None, 5120): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 128, None, 5120): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 32, None, 6144): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 48, None, 6144): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 64, None, 6144): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
        ('sm_107a', True, True, True, True, False, 128, None, 8192): ((128, 4096, {"block_n": 128, "cta_rows": 64, "sk": False, "pro": 15, "hints": ('evict_first', 'none')}),),
    },
    ("sm_103a", 148): {
        **{
            ('sm_103a', False, True, True, False, False, N, K, None): ((lo, hi, dict(_SM100A_148_DGRAD_F32)),)
            for N, K, lo, hi in (
                (4096, 32, 8192, 32768),
                (4096, 48, 8192, 32768),
                (4096, 64, 8192, 32768),
                (4096, 128, 8192, 16384),
                (5120, 32, 8192, 16384),
                (5120, 48, 16231, 16384),
                (5120, 64, 8192, 32768),
            )
        },
        ('sm_103a', False, True, True, False, False, 6144, 128, None): ((8192, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        ('sm_103a', False, True, True, False, False, 8192, 128, None): ((8192, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        ('sm_103a', False, True, True, False, False, 6144, 32, None): ((8192, 8192, {**_SM100A_148_DGRAD_F32, "group_m": 4}),),
        **{
            ('sm_103a', False, True, False, False, False, N, K, None): ((lo, hi, dict(_W8RELF_DGRAD_BF16)),)
            for N, K, lo, hi in (
                (4096, 48, 16172, 16384),
                (4096, 64, 16172, 32768),
                (4096, 128, 16172, 32768),
                (5120, 48, 16172, 32768),
                (5120, 64, 16172, 32768),
                (5120, 128, 16172, 32768),
                (6144, 128, 16172, 16384),
                (8192, 128, 16172, 16384),
            )
        },
        ('sm_103a', False, True, False, False, False, 4096, 32, None): (
            (16172, 16383, dict(_W8RELF_DGRAD_BF16)),
            (16384, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
        ),
        ('sm_103a', False, True, False, False, False, 5120, 32, None): ((16172, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_103a', False, True, False, False, False, 6144, 32, None): (
            (16172, 16230, dict(_W8RELF_DGRAD_BF16)),
            (16231, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
        ),
    },
    ("sm_103a", 152): {
        **{
            ('sm_103a', False, True, True, False, False, N, K, None): ((16172, 16384, dict(_SM100A_148_DGRAD_F32)),)
            for N, K in (
                (4096, 32),
                (4096, 48),
                (4096, 64),
                (4096, 128),
                (5120, 32),
                (5120, 48),
                (5120, 64),
                (5120, 128),
            )
        },
        ('sm_103a', False, True, True, False, False, 8192, 128, None): (
            (16172, 16230, dict(_SM100A_148_DGRAD_F32)),
            (16231, 16384, {**_SM100A_148_DGRAD_F32, "group_m": 2}),
        ),
        ('sm_103a', False, True, True, False, False, 6144, 128, None): ((16172, 16384, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        ('sm_103a', False, True, True, False, False, 6144, 32, None): ((16172, 32768, {"block_n": 128, "slots": 2, "stages": 4, "cta1": True, "group_m": 2}),),
        ('sm_103a', False, True, False, False, False, 4096, 32, None): (
            (16172, 16231, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 32}),
            (16232, 16383, dict(_W8RELF_DGRAD_BF16)),
            (16384, 16384, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
            (32768, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 32}),
        ),
        ('sm_103a', False, True, False, False, False, 4096, 64, None): ((16172, 32768, {**_W8RELF_N128_DGRAD_BF16, "group_m": 32}),),
        ('sm_103a', False, True, False, False, False, 5120, 32, None): ((16172, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_103a', False, True, False, False, False, 5120, 64, None): ((16172, 32768, {**_W8RELF_N128_DGRAD_BF16, "group_m": 16}),),
        ('sm_103a', False, True, False, False, False, 6144, 32, None): ((16172, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        **{
            ('sm_103a', False, True, False, False, False, N, K, None): ((lo, hi, dict(_W8RELF_DGRAD_BF16)),)
            for N, K, lo, hi in (
                (4096, 48, 16172, 32768),
                (4096, 128, 16172, 32768),
                (5120, 48, 16172, 32768),
                (5120, 128, 16172, 32768),
                (6144, 128, 16172, 16384),
            )
        },
        # version 0.3.27: the FP32 4096 -> 12288 weight gradient (the direct key: M 12288 output rows over N 4096
        # columns) plans the 224-column program with the FP32 v8 evict-first register epilogue on whole tiles at
        # T 512 - 4096, ``dense_proj_det_gemm_nn_n224_f32_v8_ef_pro15`` (the program of the sm_103a FP32 4096 -> 1024
        # rule; 912 pair tiles, 12 whole waves on 76 pairs, no K split), in place of its ``ROW_RULES`` rule (256-row
        # tiles with a four-way K split of the tail, ``dense_proj_det_gemm_nn_n256_m256_f32_tma1_pro7``), which the
        # other sm_103a parts and the other token counts keep.  Measured before shipping in this package on one GB300
        # node (152 SMs) against the plan of version 0.3.26 (the plan of version 0.3.25 at this key) and a second
        # prepared copy of it, cold L2, eager launches and CUDA graph replays, 21 paired rounds per cell and mode, at
        # T 512, 1024, 1984, 2048 and 4096: faster beyond the
        # A/A band in both modes at each, with one bit pattern run to run and within the naive BF16 tier on five seeds;
        # medians of that plan over the new one 1.107 / 1.106, 1.134 / 1.135, 1.125 / 1.125, 1.178 / 1.188 and
        # 1.052 / 1.052 (eager / CUDA graph).  From T 8192 on (measured to T 32768) it was slower, 0.920 to 0.981.  The
        # bits are those of the plan before at the measured T 512 - 1984 (whole tiles in both plans) and differ at
        # T 2048 and 4096 (the plan before splits the tail's K there).  Post-ship: medians of the plan before over the
        # new plan of 1.064 to 1.154 (eager) and 1.066 to 1.155 (CUDA graph) at the 9 timed changed cells, each faster
        # beyond the A/A band.  The generic call plans the same direct key (no swap) and runs this plan
        ('sm_103a', True, True, True, False, False, 4096, None, 12288): ((512, 4096, {"block_n": 224, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15}),),
    },
    ("sm_100a", 152): {
        **{
            ('sm_100a', False, True, True, False, False, N, K, None): ((16172, 16384, dict(_SM100A_148_DGRAD_F32)),)
            for N, K in (
                (4096, 32),
                (4096, 48),
                (4096, 64),
                (4096, 128),
                (5120, 32),
                (5120, 48),
                (5120, 64),
            )
        },
        ('sm_100a', False, True, True, False, False, 5120, 128, None): (
            (16172, 16172, dict(_SM100A_148_DGRAD_F32)),
            (16231, 16231, {**_SM100A_148_DGRAD_F32, "group_m": 2}),
            (16384, 16384, dict(_SM100A_148_DGRAD_F32)),
        ),
        ('sm_100a', False, True, True, False, False, 6144, 128, None): (
            (16172, 16172, {**_SM100A_148_DGRAD_F32, "group_m": 2}),
            (16231, 16384, {**_SM100A_148_DGRAD_F32, "group_m": 2}),
        ),
        ('sm_100a', False, True, True, False, False, 8192, 128, None): (
            (16172, 16172, {**_SM100A_148_DGRAD_F32, "group_m": 4}),
            (16231, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),
        ),
        ('sm_100a', False, True, True, False, False, 6144, 32, None): ((16172, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        **{
            ('sm_100a', False, True, False, False, False, N, K, None): ((16172, 32768, dict(_W8RELF_DGRAD_BF16)),)
            for N, K in (
                (4096, 128),
                (6144, 128),
                (8192, 128),
            )
        },
        ('sm_100a', False, True, False, False, False, 4096, 32, None): ((16172, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 2}),),
        ('sm_100a', False, True, False, False, False, 4096, 64, None): ((16172, 32768, {**_W8RELF_N128_DGRAD_BF16, "group_m": 16}),),
        ('sm_100a', False, True, False, False, False, 5120, 32, None): (
            (16172, 16384, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
            (16385, 32767, dict(_W8RELF_DGRAD_BF16)),
            (32768, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 2}),
        ),
        ('sm_100a', False, True, False, False, False, 5120, 64, None): (
            (16172, 16172, {**_W8RELF_N128_DGRAD_BF16, "group_m": 2}),
            (16173, 16230, dict(_W8RELF_DGRAD_BF16)),
            (16231, 16384, {**_W8RELF_N128_DGRAD_BF16, "group_m": 16}),
            (16385, 32768, dict(_W8RELF_DGRAD_BF16)),
        ),
        ('sm_100a', False, True, False, False, False, 6144, 32, None): ((16172, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_100a', False, True, False, False, False, 4096, 48, None): ((16172, 32768, dict(_W8RELF_DGRAD_BF16)),),
        ('sm_100a', False, True, False, False, False, 5120, 48, None): (
            (16172, 16384, dict(_W8RELF_DGRAD_BF16)),
            (32768, 32768, dict(_W8REL_DGRAD_BF16)),
        ),
        ('sm_100a', False, True, False, False, False, 5120, 128, None): (
            (16172, 16231, dict(_W8RELF_DGRAD_BF16)),
            (16384, 16384, dict(_W8RELF_DGRAD_BF16)),
            (32768, 32768, dict(_W8RELF_DGRAD_BF16)),
        ),
    },
    ("sm_100a", 148): {
        ('sm_100a', False, True, True, False, False, 8192, 128, None): ((8192, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        ('sm_100a', False, True, True, False, False, 5120, 128, None): ((16384, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        ('sm_100a', False, True, True, False, False, 6144, 128, None): ((16384, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),),
        ('sm_100a', False, True, True, False, False, 6144, 32, None): (
            (16231, 16231, {**_SM100A_148_DGRAD_F32, "group_m": 4}),
            (32768, 32768, {**_SM100A_148_DGRAD_F32, "group_m": 2}),
        ),
        **{
            ('sm_100a', False, True, False, False, False, N, K, None): ((lo, hi, dict(_W8RELF_DGRAD_BF16)),)
            for N, K, lo, hi in (
                (4096, 48, 16172, 32768),
                (4096, 128, 16172, 32768),
                (5120, 48, 16172, 32768),
                (5120, 64, 16172, 32768),
                (5120, 128, 16172, 16384),
                (6144, 128, 16172, 16384),
                (8192, 128, 16172, 16384),
            )
        },
        ('sm_100a', False, True, False, False, False, 4096, 32, None): (
            (16172, 32767, dict(_W8RELF_DGRAD_BF16)),
            (32768, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
        ),
        ('sm_100a', False, True, False, False, False, 4096, 64, None): (
            (16172, 16231, {**_W8RELF_N128_DGRAD_BF16, "group_m": 2}),
            (16384, 32768, dict(_W8RELF_DGRAD_BF16)),
        ),
        ('sm_100a', False, True, False, False, False, 5120, 32, None): ((16172, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),),
        ('sm_100a', False, True, False, False, False, 6144, 32, None): (
            (16172, 16172, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
            (16173, 16383, dict(_W8RELF_DGRAD_BF16)),
            (16384, 32768, {**_W8RELF_N128_K32_DGRAD_BF16, "group_m": 16}),
        ),
    },
}
# fmt: on

#: Package-only: the K blocks of one promoted chunk of the weight-gradient programs (``promote_blocks``, the twin
#: templates ``..._kc64_pro<n>`` of :func:`instance_symbol`).  One FP32 accumulator that adds a long token range is
#: less accurate than the reference, so the promoted program cuts each segment of ``n`` K blocks (a whole tile's K
#: range, a stream-K unit's part of a tile, a part of the exact split, the main or a collector part of the synchronised
#: split) into ``ceil(n / K1_PROMOTE_BLOCKS)`` balanced contiguous chunks, sums each in an accumulator of its own and
#: adds the chunk sums in chunk order, one FP32 round-to-nearest addition per element: the same bits on every run.  The
#: loads, the plan, the workspace and the launch arguments are the program's without promotion; at one chunk
#: (``n <= 64``) its bits are that program's.  The planner runs the twin where :func:`k1_promote_blocks` names it (a
#: segment of more than :data:`K1_PROMOTE_MIN_SEGMENT` K blocks), except on the rows of :data:`K1_PROMOTE_EXEMPT_ROWS`.
#: Versions 0.3.2 to 0.3.20 shipped the twins of 128-block chunks (``..._kc128_pro<n>``); version 0.3.21 ships the
#: twins of 64-block chunks in their place, so every plan that runs a twin with a segment of more than 64 K blocks sums
#: in other chunks than version 0.3.20 did, and its bits can differ from that version's.
K1_PROMOTE_BLOCKS = 64
#: The longest segment (:func:`segment_blocks`) above which :func:`k1_promote_blocks` names the promoted twin, in K
#: blocks of 64 tokens: the twin runs where a segment holds more than this many, 64 (4096 tokens in one accumulator,
#: one chunk of :data:`K1_PROMOTE_BLOCKS`).  So, where the architecture registers the twin, the plan holds the twin's
#: stream-K contract (:func:`check_stream_k_plan`) and the row is no row of :data:`K1_PROMOTE_EXEMPT_ROWS`, a plan runs
#: the twin wherever a segment holds more than one chunk.  A plan whose program has no registered twin keeps that
#: program at any length.  Up to version 0.3.5 the threshold was 256 K blocks, and plans whose longest segment held 129
#: to 256 K blocks without promoted chunks missed the naive BF16 tier on some BF16 weight gradients (errors up to 1.0096
#: times the tier's; the tests allow 1.005 times).  Versions 0.3.6 to 0.3.20 set the threshold to 128 K blocks, the
#: chunk of the twins since version 0.3.2.  A cell is a row at one token count on one seed (32 draws against an FP64
#: reference; the L2 and the largest-element norms of the error against the naive BF16 tier's).
#: The full re-judge of version 0.3.20 (5 seeds) found misses on two sm_107a parts: 3 cells on VR200 (212 SMs), 1.00504
#: to 1.00683 times the tier's largest-element error, on segments of 127 to 128 K blocks without promotion (up to 8192
#: tokens in one accumulator), and one cell on the 216-SM R200 board, 1.00540, on the 128-block chunks of a promoted
#: plan.  A pre-registered screen then judged the thin BF16 weight gradients (rows of at most 128 output features, and
#: the cells of the tests' missed-cell tables) on 21 seeds per row and token count, in the deterministic and the
#: non-deterministic package, with the plans of version 0.3.20 and with a chunk and threshold of 64 and of 32 K blocks;
#: its rule takes the largest chunk that misses no cell on any part.
#: Per part: cells; misses (deterministic / non-deterministic) and the worst largest-element ratio of the 0.3.20 plans,
#: then of chunks of 64:
#:   * B200 (sm_100a, 148 SMs): 9534 cells; 0.3.20 1 / 3, 1.00556; chunks of 64 0 / 0, 1.00257;
#:   * GB200 (sm_100a, 152 SMs): 9387 cells; 0.3.20 0 / 0, 1.00478; chunks of 64 0 / 0, 1.00321;
#:   * B300 (sm_103a, 148 SMs): 10122 cells; 0.3.20 1 / 3, 1.00556; chunks of 64 0 / 0, 1.00257;
#:   * GB300 (sm_103a, 152 SMs): 9912 cells; 0.3.20 0 / 0, 1.00478; chunks of 64 0 / 0, 1.00321;
#:   * VR200 (sm_107a, 212 SMs): 9072 cells; 0.3.20 5 / 5, 1.00683; chunks of 64 0 / 0, 1.00377;
#:   * R200 board (sm_107a, 216 SMs): 8631 cells; 0.3.20 5 / 5, 1.00540; chunks of 64 0 / 0, 1.00370.
#: Every 0.3.20 miss lay on a segment of at most 128 K blocks without promotion or on the chunks (of at most 128 K
#: blocks) of a promoted plan.  The L2 ratios stayed at or below 1.000022 (version 0.3.20) and 1.000019 (chunks of 64)
#: on every part.  Chunks of 32 missed no cell either where screened (every part but B300); the rule takes 64.  The
#: wider rows whose plans change with chunks of 64, judged the same way, missed no cell with chunks of 64: 4096 -> 512,
#: 5120 -> 512, the kv_a 6144 -> 576 and the batched MLA value-projection row on B200 (3591 cells) and B300 (3360); the
#: first three on GB200 (2688), GB300 (2562), VR200 (3360) and the R200 board (1974).  The full re-judge of version
#: 0.3.21 (5 seeds, every part above), per part the cells, the misses (deterministic / non-deterministic) and the
#: largest ratio to the tier (either norm, median or largest error): B200 3928, 0 / 0, 1.002558; GB200 3860, 0 / 0,
#: 1.002239; B300 4024, 0 / 0, 1.002558; GB300 4010, 0 / 0, 1.002343; VR200 3617, 0 / 0, 1.002261; the R200 board 3524,
#: 0 / 0, 1.003311.  Cost: the kernel time of the screen's programs with chunks of 64 over that of the 0.3.20 plan, the
#: median over the cells whose plan changes, eager launches on B200, GB200, B300, GB300, VR200 and the R200 board:
#: 1.0004 (GB300) to 1.0019 (the R200 board), the largest single cell 1.0353 (VR200); the package A/B of the plans that
#: change on every part above (CUPTI, cold L2, eager launches and CUDA graph replays), per part the cells, the geomean
#: of the 0.3.20 time over the 0.3.21 time (above 1: 0.3.21 faster) and the cells slower than 0.3.20 beyond the spread
#: of a second 0.3.20 arm, eager / graph: B200 197, 1.000 / 1.000, 12 / 13; GB200 195, 1.000 / 1.000, 2 / 2; B300 218,
#: 1.001 / 1.000, 3 / 3; GB300 208, 1.000 / 1.000, 5 / 4; VR200 166, 0.998 / 0.999, 11 / 6; the R200 board 167, 0.996 /
#: 0.997, 14 / 6; a slower cell took at most 1.0258 times the 0.3.20 time (paired median).
K1_PROMOTE_MIN_SEGMENT = 64
#: Rows whose plans keep the program without promotion where :func:`k1_promote_blocks` names the twin, keyed as
#: ``ROW_RULES`` (the planned orientation): an exact ``(N, K, M)`` key holds at that token count ``K`` only, a ragged
#: ``(N, None, M)`` key at every token count.  ``promote=True`` runs the twin on these rows too.
K1_PROMOTE_EXEMPT_ROWS: frozenset[tuple] = frozenset()


def _measured_row_rule(
    arch: str,
    a_mn: bool,
    b_mn: bool,
    out_f32: bool,
    out_t: bool,
    batched: bool,
    N: int,
    K: int,
    M: int,
    *,
    sm_count: Optional[int] = None,
) -> dict:
    """The row's own ``ROW_RULES`` entry without the architecture defaults: the exact (N, K, M) rule, else the
    ragged-M (N, K, None) rule, else the ragged-K (N, None, M) rule, else the parent architecture's entry
    (``RULE_PARENT``: sm_103a reads the sm_100a table for the rows its lanes did not measure), else empty (the lookup
    of Cake ``row_rule``; package-only split so the registry fallback can drop the MEASURED rule alone, see
    ``cake_dense_gemm_det_train_projection_plan``).  A ragged key with token tiers (``ROW_RULE_MIN_T``) yields the
    first tier whose bound exceeds the token count (``M`` of a ragged-M key, ``K`` of a ragged-K key).  Package-only:
    with ``sm_count``, the part's tables for ``(arch, sm_count)`` come first - an exact key of ``ROW_RULES_BY_SM``, then
    the range of ``ROW_RULE_RANGES_BY_SM`` that holds the token count (the ragged-M key, then the ragged-K key) - for
    the architecture's own tables only (the parent lookup does not read them)."""
    ident = (arch, bool(a_mn), bool(b_mn), bool(out_f32), bool(out_t), bool(batched))
    if sm_count is not None:
        part = (arch, int(sm_count))
        part_rule = ROW_RULES_BY_SM.get(part, {}).get(ident + (int(N), int(K), int(M)))
        if part_rule is not None:
            return dict(part_rule)
        ranges = ROW_RULE_RANGES_BY_SM.get(part, {})
        for tail, tokens in (
            ((int(N), int(K), None), int(M)),
            ((int(N), None, int(M)), int(K)),
        ):
            for lo, hi, rule in ranges.get(ident + tail, ()):
                if lo <= tokens <= hi:
                    return dict(rule)  # the part's rule at this token count
    for tail in (
        (int(N), int(K), int(M)),
        (int(N), int(K), None),
        (int(N), None, int(M)),
    ):
        rule = ROW_RULES.get(ident + tail)
        if rule is not None:
            tokens = int(M) if tail[2] is None else int(K)
            for bound, below in ROW_RULE_MIN_T.get(ident + tail, ()):
                if tokens < bound:
                    return dict(below)  # under the key's token floor: the tier's rule
            return dict(rule)
    parent = RULE_PARENT.get(arch)
    if parent is not None:
        return _measured_row_rule(parent, a_mn, b_mn, out_f32, out_t, batched, N, K, M)
    return {}


def row_rule(
    arch: str,
    a_mn: bool,
    b_mn: bool,
    out_f32: bool,
    out_t: bool,
    batched: bool,
    N: int,
    K: int,
    M: int,
    *,
    sm_count: Optional[int] = None,
) -> dict:
    """The measured knob overrides of one row identity: the exact (N, K, M) rule, else the
    ragged-M (N, K, None) rule, else the ragged-K (N, None, M) rule, else the parent architecture's rule
    (``RULE_PARENT``), else empty (the class defaults apply) - each merged over the architecture's
    ``ARCH_RULE_DEFAULTS`` (round 33: a row rule overrides an architecture default).  Package-only: ``sm_count`` reads
    the part's exact keys of ``ROW_RULES_BY_SM`` and token ranges of ``ROW_RULE_RANGES_BY_SM`` first
    (``_measured_row_rule``).  [Cake ``row_rule``]"""
    merged = dict(ARCH_RULE_DEFAULTS.get(arch, {}))
    merged.update(
        _measured_row_rule(
            arch, a_mn, b_mn, out_f32, out_t, batched, N, K, M, sm_count=sm_count
        )
    )
    return merged


def default_block_n(N: int, b_mn: bool) -> int:
    """256 columns per CTA pair unless the whole output fits a 128-column tile; 192 when N is a
    multiple of 192 but not of 256 (N = 576: three exact 192-column tiles instead of
    256 + 256 + 64; N = 192: one exact tile).  [Cake ``default_block_n`` L1170-L1176]"""
    if N <= 128:
        return 128
    return 192 if (N % 192 == 0 and N % 256 != 0) else 256


def default_pf(M: int, N: int, K: int) -> int:
    """Rolling L2 prefetch distance in K steps (0 = off; selected per row class by measurement).
    [Cake ``default_pf`` L1179-L1181]"""
    return 0


def default_promo(m_tiles: int, n_tiles: int) -> str:
    """TMA descriptor L2 promotion of the operands (``"none"``; selected by measurement).
    [Cake ``default_promo`` L1184-L1186]"""
    return "none"


def wave_working_set(
    m_tiles: int, n_tiles: int, k_len: int, group_m: int, pairs: int, elt_bytes: int = 2
) -> int:
    """Bytes of A and B panels touched by one wave of CTA pairs: with ``group_m`` row tiles per
    raster group a wave covers ``rows`` pair-row panels (256 x K) and ``cols`` column panels
    (K x 256) of one batch entry.  [Cake ``wave_working_set`` L1189-L1201]"""
    m_pairs = max(1, m_tiles // CTA_GROUP)
    g = max(1, group_m // CTA_GROUP)
    band = g * n_tiles  # pair tiles per raster group
    if band >= pairs:
        rows = min(m_pairs, g)
        cols = min(n_tiles, _ceil_div(pairs, rows))
    else:
        rows = min(m_pairs, g * _ceil_div(pairs, band))
        cols = n_tiles
    return (rows + cols) * 256 * k_len * elt_bytes


def default_hints(
    a_mn: bool,
    b_mn: bool,
    m_tiles: int,
    n_tiles: int,
    k_len: int,
    group_m: int,
    pairs: int,
    l2_bytes: int,
) -> tuple[str, str]:
    """TMA load L2 eviction policy of (A, B).  No hint while one wave's operand panels fit the
    L2 (a hint on a resident operand costs 3..23 percent).  Otherwise only a panel used by
    exactly one tile (A when there is one column tile) streams as ``evict_first``.  Reuse-ratio
    hints on the weight-gradient class measured 0..-4 percent in the production stream-K
    configuration and are not applied; ``a_mn`` / ``b_mn`` stay in the signature so a
    layout-class rule can be re-added without touching the call site.
    [Cake ``default_hints`` L1204-L1216]"""
    if wave_working_set(m_tiles, n_tiles, k_len, group_m, pairs) <= l2_bytes:
        return ("none", "none")
    if n_tiles == 1:
        return ("evict_first", "none")
    return ("none", "none")


def wave_working_set_cgrp(
    m_tiles: int, n_tiles: int, k_len: int, group_n: int, pairs: int, elt_bytes: int = 2
) -> int:
    """Round 21 (Cake W3): bytes of A and B panels touched by one wave of CTA pairs under the column-grouped
    raster (``cgrp``): a group of ``group_n`` column panels (K x 256) is swept by consecutive pairs of one pair
    row, so a wave covers ``cols`` column panels and ``rows`` pair-row panels (256 x K) of one batch entry.
    [Cake ``wave_working_set_cgrp`` L2515-L2528]"""
    m_pairs = max(1, m_tiles // CTA_GROUP)
    g = max(1, min(group_n, n_tiles))
    band = g * m_pairs  # pair tiles per column group
    if band >= pairs:
        cols = g
        rows = min(m_pairs, _ceil_div(pairs, g))
    else:
        cols = min(n_tiles, g * _ceil_div(pairs, band))
        rows = m_pairs
    return (rows + cols) * 256 * k_len * elt_bytes


def default_hints_cgrp(
    a_mn: bool,
    b_mn: bool,
    m_tiles: int,
    n_tiles: int,
    k_len: int,
    group_n: int,
    pairs: int,
    l2_bytes: int,
) -> tuple[str, str]:
    """``default_hints`` for the column-grouped raster (round 21, Cake W3): the same policy over the column
    raster's wave working set.  [Cake ``default_hints_cgrp`` L2531-L2537]"""
    if wave_working_set_cgrp(m_tiles, n_tiles, k_len, group_n, pairs) <= l2_bytes:
        return ("none", "none")
    if n_tiles == 1:
        return ("evict_first", "none")
    return ("none", "none")


def default_group_m(
    a_mn: bool, b_mn: bool, m_tiles: int, pair_tiles: int, pairs: int
) -> int:
    """CTA row tiles per raster group (even): 16 on every row (16 is fastest or within 1 percent
    on every projection row; small groups cost 3..30 percent on the many-wave rows).  A 4-row
    group for one-to-two-wave weight-gradient rows measured 0..+2 percent in the production
    configuration and is not applied.  The knob stays per instance.
    [Cake ``default_group_m`` L1219-L1224]"""
    return 16


def default_cta_rows(M: int, N: int, K: int) -> int:
    """Output rows per CTA: 128 (double-buffered TMEM tiles) by default; 256 (two MMAs per K step
    sharing the B stage) and 64 (round 9: 128 x BLOCK_N pair tiles through the M=128 cta_group::2
    MMA, four TMEM buffers, 24 KiB stages) are selected per row by measurement.
    [Cake ``default_cta_rows`` L1227-L1231]"""
    return 128


def sk_parts_plan(
    pair_tiles: int, k_blocks: int, pairs: int, parts: int
) -> Tuple[Union[bool, str], Optional[int]]:
    """``(sk, sk_max_units)`` for a measured ``sk_parts`` row rule: the tail tiles (the whole
    problem when it is one partial wave) are split ``parts`` ways when ``parts * tail`` units
    fit the CTA pairs and every unit keeps at least ``SK_MIN_ITERS`` K steps; otherwise the row
    keeps the ``auto`` policy.  [Cake ``sk_parts_plan`` L1405-L1412]"""
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    if (
        parts < 2
        or tail == 0
        or parts * tail > pairs
        or k_blocks < parts * SK_MIN_ITERS
    ):
        return "auto", None
    return True, parts * tail


def sk_exact_plan(
    pair_tiles: int, k_blocks: int, pairs: int, parts: int
) -> Optional[tuple[int, int, int, int]]:
    """(num_full, tail_tiles, sk_units, iters_per_unit) of the exact ``parts``-way K split of the tail tiles
    (Cake round 13, L36): the tiles that do not fill a whole wave of CTA pairs (all of them when there are fewer
    tiles than pairs) each become ``parts`` units of ``ceil(k_blocks / parts)`` contiguous K steps - identical K
    ranges on every tile, exactly ``parts`` partials per tile, no drift when ``parts`` does not divide
    ``k_blocks``.  The unit count may exceed the pairs.  ``None`` when the split is not admissible: fewer than two
    parts, no tail, a part under ``SK_MIN_ITERS`` steps, or an empty last part.  [Cake ``sk_exact_plan``]"""
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    parts = int(parts)
    if parts < 2 or tail == 0:
        return None
    ipu = _ceil_div(k_blocks, parts)
    if ipu < SK_MIN_ITERS or (parts - 1) * ipu >= k_blocks:
        return None
    return pair_tiles - tail, tail, tail * parts, ipu


def stream_k_plan(
    pair_tiles: int,
    k_blocks: int,
    pairs: int,
    sk: bool | str = True,
    max_units: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """``(num_full, tail_tiles, sk_units, iters_per_unit)``: the tiles that do not fill a
    whole wave of CTA pairs (all of them when there are fewer tiles than pairs) are the tail.

    * ``sk=False``: every work item is a whole tile.
    * ``sk="auto"`` (the launcher default): a two-way split of a single partial wave -- the
      tail's ``tail * k_blocks`` K steps, linearised, cut into units of ``ceil(k_blocks / 2)``
      steps.  On an even ``k_blocks`` unit ``u`` is tile ``u // 2``, K half ``u % 2``; on an odd
      one a unit may cross into the next tile.  The units are the ``ceil(tail * k_blocks /
      ceil(k_blocks / 2))`` that hold a K step (one fewer than ``2 * tail`` when ``k_blocks`` is
      odd and ``2 * tail > k_blocks``): a unit without a K step never finishes in the programs
      that hand off the work in an item's last segment (``pd`` 2, ``sk_slab`` 2, ``stt``).  Only
      when the whole problem is one partial wave whose halves fit the pairs (``pair_tiles <=
      pairs`` and ``2 * tail <= pairs``) and K has at least ``2 * SK_MIN_ITERS`` steps;
      multi-wave problems keep their tail data-parallel.
    * ``sk=True``: the tail tiles are linearised over their K steps and shared by up to
      ``pairs`` (``max_units``) stream-K units of at least ``SK_MIN_ITERS`` contiguous K
      steps; no stream-K when that cannot create more units than tail tiles.
    * ``sk="tiles"``: diagnostic -- the tail tiles go through the unit path as whole tiles.

    [Cake ``stream_k_plan`` L1415-L1444]"""
    if not sk or pair_tiles == 0:
        return pair_tiles, 0, 0, k_blocks
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    if tail == 0:
        return pair_tiles, 0, 0, k_blocks
    if (
        sk == "tiles"
    ):  # diagnostic: the tail tiles go through the unit path as whole tiles (no partials, no fixup)
        return pair_tiles - tail, tail, tail, k_blocks
    if sk == "auto":
        if pair_tiles > pairs or 2 * tail > pairs or k_blocks < 2 * SK_MIN_ITERS:
            return pair_tiles, 0, 0, k_blocks
        half = _ceil_div(k_blocks, 2)
        return pair_tiles - tail, tail, _ceil_div(tail * k_blocks, half), half
    total = tail * k_blocks
    units = min(pairs, total // SK_MIN_ITERS, max_units or pairs)
    if units <= tail:
        return pair_tiles, 0, 0, k_blocks
    iters_per_unit = _ceil_div(total, units)
    units = _ceil_div(total, iters_per_unit)
    if units <= tail:
        return pair_tiles, 0, 0, k_blocks
    return pair_tiles - tail, tail, units, iters_per_unit


def check_stream_k_plan(
    plan: "CakeDenseGemmDetProjectionPlan", arch: Optional[str] = None
) -> None:
    """Refuse a stream-K plan outside the program's launch contract (the Cake kernel's ``requires``) before it
    launches.  The programs that hand off the work inside an item's last segment (``pd`` 2, ``sk_slab`` 2,
    ``stt``) never finish an item that holds no K step: the linearised plan needs ``iters_per_unit >= 1`` and
    ``(sk_units - 1) * iters_per_unit < sk_iters``, the synchronised one at most ``tail_tiles`` collectors.  Every
    synchronised plan needs a collector for the K steps ``[s, k_blocks)`` of its tail tiles.  With ``arch``, a
    linearised plan of a program that keys its workspace by tail tile (``tile_keyed_workspace``) needs
    ``2 * iters_per_unit >= k_blocks``: a tile then holds at most ``WS_PARTS`` units, one slab each.  A plan without
    stream-K units holds every condition."""
    if not plan.sk_units:
        return
    handoff = plan.pd == 2 or plan.sk_slab == 2 or plan.stt
    if plan.sk_sync:
        collectors = plan.sk_collectors
        if collectors < 1 or (handoff and collectors > plan.tail_tiles):
            raise ValueError(
                f"cake_dense_gemm_det_train_projection: {plan.template} needs 1 <= collectors"
                f"{' <= tail_tiles' if handoff else ''} (got {collectors} collectors, {plan.tail_tiles} tail tiles)"
            )
    elif handoff and not plan.sk_exact:
        if (
            plan.iters_per_unit < 1
            or (plan.sk_units - 1) * plan.iters_per_unit >= plan.sk_iters
        ):
            raise ValueError(
                f"cake_dense_gemm_det_train_projection: {plan.template} needs every stream-K unit to hold a K step (got "
                f"{plan.sk_units} units of {plan.iters_per_unit} over {plan.sk_iters} K steps)"
            )
    if (
        arch is not None
        and not (plan.sk_sync or plan.sk_exact or plan.htail_plan)
        and 2 * plan.iters_per_unit < plan.k_blocks
        and tile_keyed_workspace(arch, plan.template)
    ):
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: {plan.template} keys its stream-K workspace by tail tile and needs "
            f"2 * iters_per_unit >= k_blocks, at most {WS_PARTS} units on a tile (got {plan.sk_units} units of "
            f"{plan.iters_per_unit} K steps over {plan.tail_tiles} tail tiles of {plan.k_blocks} K steps)"
        )


def sk_sync_plan(
    pair_tiles: int,
    k_blocks: int,
    pairs: int,
    s_override: Optional[int] = None,
    margin: int = 0,
) -> Optional[tuple[int, int, int, int, int]]:
    """``(num_full, tail_tiles, sk_units, s, collectors)`` of the synchronised stream-K plan (round 18, Cake W2) - pure
    integer arithmetic mirrored from the Cake launcher's ``sk_sync_plan``.  The tail tiles (all of them when there are
    fewer tiles than pairs) each get one MAIN unit over K steps [0, s) and ``collectors`` = min(pairs - tail, tail)
    COLLECTOR units own the leftovers [s, k_blocks) of tail tiles c, c + collectors, c + 2 collectors, ... (one segment
    per leftover, at most q = ceil(tail / collectors) per collector).  With one atomic segment per leftover the makespan
    is max(s, q (k_blocks - s)), balanced at s = ceil(q k_blocks / (q + 1)); ``margin`` m (K steps, rule key
    ``sk_sync_m``) lengthens the mains to s = ceil((q k_blocks + m) / (q + 1)) so the collectors finish m steps early;
    ``s_override`` replaces s (sweeps only).  ``sk_units`` = tail + collectors.  ``None`` when not admissible: no tail,
    no collector, or a part under ``SK_MIN_ITERS`` (the row keeps its other policy)."""
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    if tail == 0:
        return None
    collectors = min(pairs - tail, tail)
    if collectors < 1:
        return None
    q = _ceil_div(tail, collectors)
    s = (
        _ceil_div(q * k_blocks + int(margin), q + 1)
        if s_override is None
        else int(s_override)
    )
    if s < SK_MIN_ITERS or k_blocks - s < SK_MIN_ITERS:
        return None
    return pair_tiles - tail, tail, tail + collectors, s, collectors


# ---------------------------------------------------------------------------
# K1 view validation and planning
# ---------------------------------------------------------------------------


def as_batched(t: torch.Tensor, name: str) -> torch.Tensor:
    # [Cake ``_as_batched`` L1019-L1024]
    if t.dim() == 2:
        return t.unsqueeze(0)
    if t.dim() == 3:
        return t
    raise ValueError(f"{name} must be a 2-D or 3-D view, got {t.dim()}-D")


def operand_view(
    t: torch.Tensor, name: str, *, k_axis: int
) -> tuple[bool, torch.Tensor]:
    """Classify a ``[L, rows, cols]`` matrix view as K-major (contraction axis contiguous) or
    MN-major and return ``(mn_major, desc)`` with ``desc`` the ``[L, outer, inner]`` view
    (inner stride 1) the TMA descriptor spans.  [Cake ``_operand_view`` L1027-L1043]"""
    mn_axis = 3 - k_axis  # the other matrix axis (1 or 2)
    if t.stride(k_axis) == 1:
        mn_major = False
        desc = t if k_axis == 2 else t.transpose(1, 2)
    elif t.stride(mn_axis) == 1:
        mn_major = True
        desc = t if mn_axis == 2 else t.transpose(1, 2)
    else:
        raise ValueError(
            f"{name} needs unit stride on its contraction or its M/N axis; strides {tuple(t.stride())}"
        )
    if (desc.shape[1] > 1 and desc.stride(1) % 8 != 0) or (
        desc.shape[0] > 1 and desc.stride(0) % 8 != 0
    ):
        raise ValueError(
            f"{name}: row and batch strides must be multiples of 8 elements (16 bytes), got {tuple(desc.stride())}"
        )
    if t.data_ptr() % 16 != 0:
        raise ValueError(f"{name}: the view must start at a 16-byte aligned address")
    return mn_major, desc


@dataclass(frozen=True)
class CakeDenseGemmDetProjectionPlan:
    """The host plan of one dense projection GEMM launch (device-independent except ``sm_pairs``: the concurrent
    work items of the device - CTA pairs, or SMs under the single-CTA form ``cta1``)."""

    L: int
    M: int
    N: int
    K: int
    a_mn: bool
    b_mn: bool
    out_f32: bool
    transposed_out: bool
    block_n: int
    cta_rows: int
    stages: int
    epi: str
    slots: int
    pf: int
    promo: str
    hints: tuple[str, str]
    group_m: int
    m_tiles: int
    n_tiles: int
    k_blocks: int
    pair_tiles: int
    sm_pairs: int
    l2_bytes: int
    num_full: int
    tail_tiles: int
    sk_units: int
    iters_per_unit: int
    template: str
    # Package-only (see ``cake_dense_gemm_det_train_projection_plan``): the batched small-M swap was undone / the row's
    # measured rule was dropped because the planned instance is not a generated program of this architecture
    # (the Cake host JIT-compiles it)
    swap_fallback: bool = False
    rule_fallback: bool = False
    # ... and the plan landed on the nearest generated program of the row's layout / output class (another TMA L2
    # policy, the single-CTA form, another tile family, epilogue path or BLOCK_N, or the knobs a registered symbol
    # decodes to) because no generated program serves the row's default tail-T instance either
    knob_fallback: bool = False
    # round-13 tall-family knobs of the planned instance (instance_key fields 15 / 16 / 17)
    park: bool = False
    ovl: bool = False
    htail: bool = False
    # ... and whether the half-height tail PLAN is in effect: the ``_ht`` program's tail tiles run as 2 x tail whole-K
    # half items only when there is a tail and its half items fit the device's pairs (``0 < 2 * tail <= pairs``);
    # otherwise that program runs the row's plain / stream-K plan - the fit selects the plan, never the program a
    # rule or a caller named  [Cake launcher]
    htail_plan: bool = False
    # round-13 W3 knobs: MN-major B panel width (instance_key field 18) and the exact p-way tail split (field 19 is
    # the flag; ``sk_exact`` here is the part count p, 0 = off)
    b_swz: int = 128
    sk_exact: int = 0
    # round 13 (Cake W3, L38b): batch-entry raster of the batched MLA rows (instance_key field 20; 0 = row-major)
    batch_group: int = 0
    # round 13 (Cake W4): fp32 v8 register stores carry L1::no_allocate.L2::evict_first (instance_key field 21)
    store_ef: bool = False
    # round 15 (Cake W3): L2 eviction policy of the TMA-store epilogue's bulk tensor stores (instance_key field 22)
    store_hint: str = "none"
    # round 15 (Cake W1): pipelined TMEM drain form (field 23) and suspend-time hint in ns (field 24) of the planned instance
    pd: int = 0
    sh: int = 0
    # round 18 (Cake W2): synchronised stream-K tail of the planned instance (instance_key field 25 = the 26th field,
    # ``_sks`` symbols): ``tail_tiles`` main units over K steps [0, iters_per_unit) + ``sk_units - tail_tiles``
    # collector units over the leftovers, two partials and one slab per tail tile, deterministic two-addend fixup
    sk_sync: bool = False
    # round 19 (Cake W2): the slab path of that fixup (instance_key field 26 = the 27th field, ``_sb{n}`` symbols): 0 off,
    # 1 early arrival, 2 + the bulk slab read through the dead mainloop stages, 3 + the fp32 output as the slab; always 0
    # without ``sk_sync`` (the same synchronised plan arithmetic: ``ws_f32_elems`` keeps one slab per tail tile)
    sk_slab: int = 0
    # round 19 (Cake W3): the single-CTA form (instance_key field 27 = the 28th field, ``_c1`` symbols): one
    # 128 x BLOCK_N tile per CTA with cluster dims (1, 1, 1), so ``m_tiles`` is not rounded to pairs, ``pair_tiles``
    # counts CTA tiles, ``sm_pairs`` holds the SM count and ``grid`` = ``num_cluster_tiles`` CTAs
    cta1: bool = False
    # round 21 (Cake W3): the column-grouped raster (instance_key field 28 = the 29th field, ``_cg`` symbols):
    # ``group_m`` counts COLUMN tiles per raster group (any count >= 1; the pair invariant is unchanged), consecutive
    # pairs sweep a group's columns of one pair row before the next pair row, so a group's B panels stay L2-resident
    # and A streams once per column group; the plan arithmetic (tiles, units, tail policy, grid) is the pair form's
    cgrp: bool = False
    # round 24 (the round-20 Cake W3 knob ported in round 23 by W4): the overlapped tall epilogue released by ROW HALF
    # (instance_key field 29 = the 30th and LAST field, ``_or`` symbols, after ``_cg``): the eight epilogue warps drain
    # TMEM buffer 0 and release it to the next tile's half-0 MMAs while buffer 1 drains; the B stage keeps the plain
    # tall layout (one N = 256 MMA per row half per K step).  Tall 256-column tiles only; it replaces ``ovl`` on a row
    # (never both), and the plan arithmetic (tiles, units, tail policy, grid, hint gate) is the tall pair form's
    ovr: bool = False
    # round 27 (Cake lane 7, W-3 variant A1): the transposed stmatrix / TMA-store staging of the bf16 transposed register
    # epilogue (instance_key field 30 = the 31st and LAST field, ``_stt`` symbols, directly after ``_sks``): the last work
    # item's warp slices go out through one bulk tensor store per warp from the dead mainloop stage ring (``OUT16`` = the
    # real output view), every other slice through the register stores; the plan arithmetic is the pair form's
    stt: bool = False
    # round 33 (Cake unit 2d / 2e): the pair program's prologue / cleanup sync form (instance_key field 31 = the 32nd
    # and LAST field, ``_pro{n}`` symbols after ``_or``): a bitmask (1 no startup rendezvous before the collective TMEM
    # alloc, 2 relaxed arrive on the post-init cluster sync, 4 the MMA warp allocates TMEM, 8 CTA-local exit) from the
    # merged row rule (``ARCH_RULE_DEFAULTS`` 7 under every rule of sm_100a / sm_107a; a row rule may pin another
    # value), always 0 on the single-CTA form; it changes the program only - no launch argument, no plan arithmetic
    pro: int = 0
    # the round-2 shapes tranche: the tail-first order of the exact split (instance_key field 33, ``_skf``; the plan
    # arithmetic is ``sk_exact``'s) and the load warp leaving the pair prelude early (field 34, ``_lw``); programs only,
    # no launch argument
    sk_first: bool = False
    lw: bool = False
    # the whole-tile tile resolve (instance_key field 35, ``_mg``): the load warp and the MMA warp walk whole tiles with
    # the raster quotients resolved through corrected fp32 reciprocals; a program knob of plans without stream-K units
    # (``sk_units`` 0, the kernel's launch contract) - no launch argument, no plan arithmetic
    mg: bool = False
    # the paced output stores (instance_key field 36, ``_sw<n>``): the per-warp step count by which each epilogue warp of
    # the bf16 TMA-store pd-2 drain delays its bulk tensor store issue - a program knob, no launch argument, no plan
    # arithmetic (0 = off)
    sw: int = 0
    # the eight-warp CTA form (instance_key field 37, ``_w8r`` / ``_w8rs`` / ``_w8`` / ``_w8rel`` / ``_w8relf``; "" =
    # the ten-warp pipeline): the same GEMM through four epilogue warps, a loader, a scheduler (``w8r``; ``w8`` = the
    # knob-free program without the early accumulator release; ``w8rel`` / ``w8relf`` = ``w8r`` with the start path,
    # ``w8relf`` with the epilogue's TMEM free) or the loader's elected lane in its place (``w8rs``) and the MMA warp,
    # whole kn bf16 TMA-store tiles only; a program knob that adds the raster's four fast divisors to the launch
    # (``_k1_values``); with ``block_n`` 128 (``w8relf``) ``n_tiles`` / ``pair_tiles`` and the fast divisors follow the
    # tile, and with ``block_k`` 32 ``k_blocks`` counts 32-element steps
    form: str = ""
    # the serpentine K walk (instance_key field 38, ``_ks<P>``): the load warp walks the K blocks of a whole-K item in
    # an odd block of P items descending (P = ``sm_pairs``: the wave guard keeps the rule's value on devices with P
    # pairs only) - a program knob, no launch argument, no plan arithmetic (0 = off)
    kser: int = 0
    # Package-only: the promoted chunk of the planned instance (instance_key field 39, ``_kc<KC>``; 0 = off): each
    # segment's K blocks summed in chunks of at most this many, added in chunk order (``K1_PROMOTE_BLOCKS``,
    # ``k1_promote_blocks``) - a program knob, no launch argument, no plan arithmetic
    promote_blocks: int = 0
    # Package-only: the rule's exact-wave BLOCK_N (224) yielded to the default tile because the device's CTA pairs do
    # not fit its tiles in whole waves and the default tile needs fewer waves (``exact_waves``)
    wave_fallback: bool = False
    # the K step of the planned instance (instance_key field 40; 32 = the 128-column ``w8relf`` program's 32-element
    # steps): ``k_blocks`` and ``iters_per_unit`` count steps of this many elements
    block_k: int = BLOCK_K

    @property
    def num_cluster_tiles(self) -> int:
        return self.num_full + self.sk_units

    @property
    def wave_working_set_bytes(self) -> int:
        """Operand bytes one wave of CTA pairs touches (the hint gate compares it with ``l2_bytes``): the column
        raster's working set under ``cgrp`` (``wave_working_set_cgrp``), the row raster's otherwise."""
        return (wave_working_set_cgrp if self.cgrp else wave_working_set)(
            self.m_tiles, self.n_tiles, self.K, self.group_m, self.sm_pairs
        )

    @property
    def grid(self) -> tuple[int, int, int]:
        """CTAs of the launch: ``num_cluster_tiles`` pairs x 2, or one CTA per work item under ``cta1``."""
        return (self.num_cluster_tiles * (1 if self.cta1 else CTA_GROUP), 1, 1)

    @property
    def ws_f32_elems(self) -> int:
        """fp32 elements of the stream-K partial slabs; 0 without stream-K and under the half-height tail plan (its items
        are whole-K tiles: no partial slabs, no fixup).  A slab is ``2 * cta_rows * block_n`` elements."""
        if not self.sk_units or self.htail_plan:
            return 0
        if self.sk_sync:
            # one slab per tail tile (the first arriver of a tile stores its partial, the second adds it)  [Cake launcher]
            return self.tail_tiles * 2 * self.cta_rows * self.block_n
        # The unit-keyed programs reach slab sk_units + tail_tiles - 1, the tile-keyed ones (``tile_keyed_workspace``)
        # WS_PARTS * tail_tiles - 1 on a linearised plan (sk_units - 1 on the exact split): room for either mapping.
        slabs = max(self.sk_units + self.tail_tiles, WS_PARTS * self.tail_tiles)
        return slabs * 2 * self.cta_rows * self.block_n

    @property
    def counters_u32(self) -> int:
        """Stream-K slice counters in use: 16 per tail tile (one per epilogue warp of the pair)."""
        return self.tail_tiles * 16

    @property
    def counters_alloc_u32(self) -> int:
        """Slice counters allocated: ``max(8192, tail_tiles * 16)`` like the Cake host's
        ``_sk_counters`` -- the dummy counters at ``SK_DUMMY_BASE + slice * 32 + lane`` (up to
        index 4607) must be addressable even when few slice counters are in use."""
        return max(SK_COUNTERS_MIN, self.counters_u32)

    @property
    def sk_iters(self) -> int:
        """The launcher's linearised stream-K span; under the half-tail plan every half item is one whole-K segment; the
        exact split (``_skx`` instances) passes the part count p in this slot."""
        if self.sk_exact:
            return int(self.sk_exact)
        if self.sk_sync:
            # the ``_sks`` instance reads the main-unit count n_t here (s in ``iters_per_unit``; collectors =
            # num_cluster_tiles - num_full - n_t)  [Cake launcher]
            return int(self.tail_tiles)
        return (self.sk_units if self.htail_plan else self.tail_tiles) * self.k_blocks

    @property
    def sk_collectors(self) -> int:
        """Collector units of the synchronised stream-K plan (0 for every other plan)."""
        return self.sk_units - self.tail_tiles if self.sk_sync else 0

    @property
    def tma_out(self) -> bool:
        # round 27: the stt program stores its last item's slices through OUT16 (the real view) and every other slice
        # through ``out``  [Cake launcher]
        return self.epi == "tma" or self.stt


# ---------------------------------------------------------------------------
# K1 weight-gradient chain plans: one split, several accumulation chains
# ---------------------------------------------------------------------------
#
# The one-split K1 program of several accumulation chains (``chain_det_gemm_nn_n<block_n>_f32_t_c<chains>``): the
# transposed-store GEMM ``out[l, n, m] = sum_k A[l, k, m] B[l, k, n]`` over the launch's ``[L, K, M]`` /
# ``[L, K, N]`` MN-major operand views, ``256 x block_n`` tiles walked persistently by at most one CTA pair per two
# SMs.  Each tile's K blocks are cut into ``chains`` contiguous ranges (the ``j``-th of ``c`` takes blocks
# ``[j kb / c, (j + 1) kb / c)``), each summed in a TMEM accumulator of its own from an ``init`` MMA; the epilogue
# adds them once, in chain order, and stores the FP32 result.  No workspace, no atomics and no order that depends
# on scheduling: a fixed input, plan and device give the same bits on every run.  The bits differ from the
# deterministic programs' (another summation order), so a row moved onto a chain plan changes its bits across
# versions and plans; its FP64 error is below the deterministic program's.

#: Token ranges at which an FP32 weight-gradient row runs the chain program on the parts of exactly one SM count,
#: ``(arch, sm_count) -> {key: ((min_tokens, max_tokens, (block_n, chains), over_exact), ...)}``, keyed by the row's
#: ``ROW_RULES`` identity without the reduction length (``(arch, a_mn, b_mn, out_f32, out_t, batched, N, M)``, the
#: direct orientation of :func:`wgrad_views`).  A token count of a range runs that range's program in place of the
#: planner's, except a token count with an exact ``ROW_RULES`` key of the row (``(..., N, K, M)``), which keeps its
#: key unless the range says ``over_exact``.  Every other SM count (the 208-SM parts) keeps the planner's routing.
# fmt: off
CHAIN_RANGES_BY_SM: dict[
    tuple[str, int],
    dict[tuple[str, bool, bool, bool, bool, bool, int, int], tuple[tuple[int, int, tuple[int, int], bool], ...]],
] = {
    ("sm_107a", 212): {
        ("sm_107a", True, True, True, False, False, 5120, 1024): ((960, 40961, (256, 3), False),),
        ("sm_107a", True, True, True, False, False, 5120, 512): ((960, 40961, (128, 3), True),),
    },
    ("sm_107a", 216): {
        ("sm_107a", True, True, True, False, False, 5120, 1024): ((960, 40961, (256, 3), False),),
        ("sm_107a", True, True, True, False, False, 5120, 512): ((960, 40961, (128, 3), True),),
    },
}
# fmt: on
CHAIN_TEMPLATE_PREFIX = (
    "chain_det_gemm_"  # the registry templates of the chain programs
)
CHAIN_PAIR_ROWS = 256  # rows of a chain program's tile (a CTA pair, 2 x 128)
CHAIN_BLOCK_K = 64  # K elements per K block
_CHAIN_INT_MAX = 1 << 30


def chain_template(block_n: int, chains: int) -> str:
    """The registry template of the chain program of ``block_n`` columns and ``chains`` accumulation chains."""
    return f"{CHAIN_TEMPLATE_PREFIX}nn_n{int(block_n)}_f32_t_c{int(chains)}"


def chains_fit(chains: int, K: int) -> bool:
    """Whether ``chains`` accumulation chains each get at least one 64-element K block of ``K``."""
    return (
        int(chains) >= 1 and int(chains) * CHAIN_BLOCK_K <= int(K) + CHAIN_BLOCK_K - 1
    )


@dataclass(frozen=True)
class ChainGemmPlan:
    """The host plan of one chain-program launch.  ``M`` / ``N`` / ``K`` / ``transposed_out`` are the caller's (as in
    :class:`CakeDenseGemmDetProjectionPlan`); the launch runs the transposed-store program over ``launch_m`` (the inner extent of the
    stored output) and ``launch_n`` (its rows), with the caller's operands exchanged (``swapped``) when the caller's
    output is not transposed."""

    L: int
    M: int
    N: int
    K: int
    transposed_out: bool
    launch_m: int
    launch_n: int
    block_n: int
    chains: int
    m_tiles: int
    n_tiles: int
    sm_pairs: int
    template: str
    swapped: bool
    a_mn: bool = True
    b_mn: bool = True
    out_f32: bool = True
    sk_units: int = 0
    tail_tiles: int = 0
    htail_plan: bool = False
    swap_fallback: bool = False
    rule_fallback: bool = False
    knob_fallback: bool = False
    wave_fallback: bool = False

    @property
    def k_blocks(self) -> int:
        return _ceil_div(self.K, CHAIN_BLOCK_K)

    @property
    def tiles(self) -> int:
        return self.L * self.m_tiles * self.n_tiles

    @property
    def num_full(self) -> int:
        return self.tiles

    @property
    def chain_bounds(self) -> tuple[int, ...]:
        """The K-block boundaries of the chains: ``(0, kb / c, 2 kb / c, ..., kb)`` (integer division)."""
        kb, c = self.k_blocks, self.chains
        return tuple(j * kb // c for j in range(c + 1))

    @property
    def grid(self) -> tuple[int, int, int]:
        return (2 * min(self.tiles, self.sm_pairs), 1, 1)

    @property
    def ws_f32_elems(self) -> int:
        return 0

    @property
    def counters_alloc_u32(self) -> int:
        """The program binds a counter buffer it does not touch at one split: a small zeroed one."""
        return 16


def chain_plan_of(
    arch: str, sm_count: int, N: int, K: int, M: int, *, transposed_out: bool
) -> Optional[tuple[int, int]]:
    """``(block_n, chains)`` of the FP32 nn weight-gradient row ``(N, K, M)`` (:data:`CHAIN_RANGES_BY_SM`) on a part of
    exactly ``sm_count`` SMs, or ``None``: outside the ranges, at an exact ``ROW_RULES`` key the range does not
    override, or at a ``K`` too short for the chains."""
    part = CHAIN_RANGES_BY_SM.get((arch, int(sm_count)))
    if not part:
        return None
    ident = (arch, True, True, True, bool(transposed_out), False)
    for lo, hi, (block_n, chains), over_exact in part.get(ident + (int(N), int(M)), ()):
        if lo <= int(K) <= hi:
            if not over_exact and ident + (int(N), int(K), int(M)) in ROW_RULES:
                return None
            return (int(block_n), int(chains)) if chains_fit(chains, K) else None
    return None


def _chain_route(
    A3: torch.Tensor,
    B3: torch.Tensor,
    O3: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    arch: str,
    sm_count: int,
) -> Optional[tuple[ChainGemmPlan, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The chain plan of a validated binding (:func:`cake_dense_gemm_det_train_projection_plan` with no caller knob), or ``None``
    (the planner's routing): an FP32 output of a single nn matrix whose row has a range in :data:`CHAIN_RANGES_BY_SM`
    for this part, a registered program and views the program takes."""
    if out.dtype != torch.float32 or int(A3.shape[0]) != 1:
        return None
    L, M, K = (int(s) for s in A3.shape)
    N = int(B3.shape[2])
    try:
        a_mn, a_desc = operand_view(A3, "A", k_axis=2)
        b_mn, b_desc = operand_view(B3, "B", k_axis=1)
    except ValueError:
        return None
    if not (a_mn and b_mn):
        return None
    choice = chain_plan_of(arch, sm_count, N, K, M, transposed_out=transposed_out)
    if choice is None:
        return None
    block_n, chains = choice
    template = chain_template(block_n, chains)
    if template not in KERNELS.get(arch, {}):
        return None
    swapped = not transposed_out
    launch_m, launch_n = (N, M) if swapped else (M, N)
    ldo, out_l = _out_strides(O3)
    if (
        launch_m % 8
        or launch_n % 8
        or max(launch_m, launch_n, K) > _CHAIN_INT_MAX
        or O3.stride(2) != 1
        or ldo % 8
        or out_l % 8
        or ldo < launch_m
        or out.data_ptr() % 16
    ):
        return None
    plan = ChainGemmPlan(
        L=L,
        M=M,
        N=N,
        K=K,
        transposed_out=bool(transposed_out),
        launch_m=launch_m,
        launch_n=launch_n,
        block_n=block_n,
        chains=chains,
        m_tiles=_ceil_div(launch_m, CHAIN_PAIR_ROWS),
        n_tiles=_ceil_div(launch_n, block_n),
        sm_pairs=max(1, int(sm_count) // 2),
        template=template,
        swapped=swapped,
    )
    a_launch, b_launch = (b_desc, a_desc) if swapped else (a_desc, b_desc)
    return plan, a_launch, b_launch, O3


def _chain_values(
    plan: ChainGemmPlan,
    *,
    A: Any,
    B: Any,
    o_alias: Any,
    ws: Any,
    counters: Any,
    ldo: int,
    out_l: int,
    enable_pdl: Any,
) -> dict[str, Any]:
    """The keyword arguments of a chain-program launch (:func:`bind_launch` orders them by the program's plan)."""
    return dict(
        A=A,
        B=B,
        out=o_alias,
        ws=ws,
        counters=counters,
        M=plan.launch_m,
        N=plan.launch_n,
        K=plan.K,
        num_l=plan.L,
        num_splits=1,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=enable_pdl,
    )


def cake_dense_gemm_det_train_projection_plan(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    sm_count: int,
    l2_bytes: int,
    transposed_out: bool = False,
    sk: bool | str = "auto",
    block_n: Optional[int] = None,
    stages: Optional[int] = None,
    epi: Optional[str] = None,
    slots: Optional[int] = None,
    cta_rows: Optional[int] = None,
    pf: Optional[int] = None,
    sk_max_units: Optional[int] = None,
    sk_parts: Optional[int] = None,
    promo: Optional[str] = None,
    hints: Optional[tuple] = None,
    group_m: Optional[int] = None,
    f32_v8: Optional[bool] = None,
    quad_store: Optional[bool] = None,
    park: Optional[bool] = None,
    ovl: Optional[bool] = None,
    htail: Optional[bool] = None,
    b_swz: Optional[int] = None,
    sk_exact: Optional[int] = None,
    batch_group: Optional[int] = None,
    store_ef: Optional[bool] = None,
    store_hint: Optional[str] = None,
    pd: Optional[int] = None,
    sh: Optional[int] = None,
    sk_sync: Optional[bool] = None,
    sk_sync_m: Optional[int] = None,
    sk_slab: Optional[int] = None,
    cta1: Optional[bool] = None,
    cgrp: Optional[bool] = None,
    ovr: Optional[bool] = None,
    stt: Optional[bool] = None,
    pro: Optional[int] = None,
    sk_first: Optional[bool] = None,
    lw: Optional[bool] = None,
    mg: Optional[bool] = None,
    sw: Optional[int] = None,
    form: Optional[str] = None,
    promote: Optional[bool] = None,
    block_k: Optional[int] = None,
    arch: str = "sm_100a",
    _fallback: bool = True,
    _allow_swap: bool = True,
    _use_rules: bool = True,
) -> tuple[CakeDenseGemmDetProjectionPlan, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the views exactly as the Cake launcher does and plan the launch for a device
    with ``sm_count`` SMs and an L2 of ``l2_bytes`` (the two device facts the plan depends on:
    stream-K split and raster group through the CTA-pair count, the hint working-set gate
    through the L2).  Returns ``(plan, a_desc, b_desc, out3)`` with the ``[L, outer, inner]``
    operand views the TMA descriptors span and the batched output view.  The knob defaults
    (``sk="auto"``, ``pf`` / ``promo`` / ``group_m`` / ``hints`` from ``default_*``, resolved in
    the launcher's order) are the launcher's; a caller-forced ``cta_rows`` outside the row's
    rule (or default) family drops the rule's ``block_n`` / ``block_k`` / ``stages`` / ``slots`` / ``epi``,
    ``sk_parts`` is a caller knob over the rule's, and the stage count is bounded by ``smem_limit_for(arch)``.
    Round 19: ``cta1`` (Cake W3) selects the single-CTA form - the tile count, the concurrent work items
    (``sm_units``: SMs instead of CTA pairs) and the grid follow the CTA group, and the form carries no tail
    policy; ``sk_slab`` (Cake W2) selects the slab path of a synchronised stream-K plan (the same plan arithmetic),
    with the launcher's yield rules for a rule-derived value.  Round 21: ``cgrp`` (Cake W3) selects the
    column-grouped raster program (``_cg``): ``group_m`` counts column tiles per raster group (any count >= 1) and
    the hint gate takes the column raster's wave working set (``default_hints_cgrp``); the plan arithmetic is unchanged.
    Round 24: ``ovr`` (the round-20 Cake W3 knob ported in round 23 by W4) selects the tall epilogue released by row
    half (``_or``, instance-key field 30): a rule-derived ``ovr`` yields to a caller-forced ``ovl``, a caller-forced
    ``ovr`` replaces a rule-derived ``ovl`` (forcing both raises in ``instance_key``), it is dropped at BLOCK_N != 256,
    and it leaves the raster, the hint gate and the plan arithmetic unchanged.  Round 27: ``stt`` (Cake lane 7) selects the
    transposed stmatrix / TMA-store staging of the bf16 transposed register epilogue (``_stt``, instance-key field 31):
    the row's rule value is taken as is and ``instance_key`` admits or refuses it; the plan arithmetic is unchanged and
    the launch hands the real output view to ``OUT16`` next to the register-store alias.  Round 33: ``pro`` (Cake
    unit 2d / 2e) selects the pair program's prologue / cleanup sync form (``_pro{n}``, instance-key field 32 = LAST):
    read from the merged rule (``row_rule`` lays ``ARCH_RULE_DEFAULTS`` - 7 on sm_100a / sm_107a - under the row's
    rule) exactly as the launcher does and zeroed on the single-CTA programs (``instance_key`` refuses ``pro`` with
    ``cta1``); a caller-forced tile family drops the rule's ``pro`` with the rule's other family-bound knobs, a
    caller-forced ``pro`` is admitted or refused by ``instance_key``; no launch argument and no plan arithmetic change.
    The round-2 shapes tranche: ``sk_first`` (``_skf``) is the tail-first order of the row's exact split - taken from the
    rule with its ``sk_exact`` and dropped with it where the split is not admitted; ``lw`` (``_lw``) is the row's
    load-warp prelude form - taken from the rule and dropped where the planned program cannot carry it (the single-CTA
    form, a ``pro`` without bit 4, a caller-forced tile family); a caller-forced value of either is admitted or refused by
    ``instance_key``.  Neither changes a launch argument or the plan arithmetic.  ``mg`` (``_mg``) is the row's
    whole-tile tile resolve - taken from the rule (the three sm_103a input-gradient rules that carry it also pin
    ``sk: False``) and dropped where the planned program cannot carry it: a plan with stream-K units or over the
    kernel's cluster-tile bound (a caller ``sk`` / ``sk_max_units`` / ``sk_parts`` / exact or synchronised split), a
    caller-forced raster or tail form the program excludes (``batch_group`` / ``cgrp`` / ``htail`` / ``sk_first`` /
    ``stt``), the single-CTA form or a caller-forced tile family; a caller-forced ``mg`` is admitted or refused by
    ``instance_key`` and refused on a plan with stream-K units (the kernel's launch contract: ``sk_units`` is 0 on
    every ``_mg`` program).  It changes neither a launch argument nor the plan arithmetic.  ``sw`` (``_sw<n>``) is the
    row's paced output stores - taken from the rule (the two sm_103a input-gradient rules that carry it, 5120 -> 512
    and 5120 -> 1024, name bf16 TMA-store pd-2 rows) and dropped where the planned program is not the bf16 TMA-store
    pd-2 drain (a caller-forced register epilogue or ``slots`` 0, another ``pd``, the fp32 output) or is the
    single-CTA form; a caller-forced ``sw`` is admitted or refused by ``instance_key`` (the kernel's refusal: a step
    count >= 0 on that drain only).  It changes neither a launch argument nor the plan arithmetic.  ``form``
    (``_w8r`` / ``_w8rs`` / ``_w8`` / ``_w8rel`` / ``_w8relf``) is the row's eight-warp CTA form - taken from the
    rule (the six sm_103a
    input-gradient rules that carry it, 4096 -> 48, 4096 -> 512, 4096 -> 1024, 5120 -> 512, 5120 -> 1024 and the
    kv_a 6144 -> 576, and the five sm_100a input-gradient rules, 4096 -> 512, 4096 -> 1024, 5120 -> 512,
    5120 -> 1024 and the kv_a 6144 -> 576, name whole-tile kn bf16 TMA-store rows; the start-path forms ``w8rel`` /
    ``w8relf`` are named by the token ranges of version 0.3.12, ``w8relf`` on its 128 x 128 tile by those of version
    0.3.14) and dropped where the plan is not
    whole-tile or a caller forces a pipeline,
    tile, output, epilogue, tail or raster
    knob, or where a start-path form meets the rule's batch raster (the row's program above the form runs the plan
    asked for); where it survives, the rule's pair-program knobs
    (``pd`` / ``pro`` / ``lw`` / ``mg`` / ``sw`` / ``stages`` / ``hints`` / ``pf`` / ``sh`` / ``store_hint`` /
    ``store_ef`` / ``quad_store`` / ``f32_v8``) are cleared before the key: the eight-warp programs carry none of them.
    A caller-forced ``form`` is admitted or refused by ``instance_key`` and refused on a plan with stream-K units.  On
    the 128 x 256 tile it changes no plan arithmetic; the launch adds the raster's four fast divisors (``fd_tiles_l`` /
    ``fd_tiles_g`` / ``fd_group`` / ``fd_last``), which the ten-warp programs' argument plans do not name.  A rule
    that names ``w8relf`` may also name its 128 x 128 tile (``block_n`` 128) and, there, the 32-element K step
    (``block_k`` 32; ``_k32`` directly after ``_n128``): ``n_tiles`` / ``pair_tiles`` and the fast divisors follow
    the tile, and with ``block_k`` 32 ``k_blocks`` and ``iters_per_unit`` count 32-element steps (a whole-tile plan
    only); every output element still sums its K in one FP32 accumulator in K order, the bits of the 256-column
    program.  Where the form survives, a caller-forced ``block_n`` 128 keeps that program and a caller-forced
    ``block_n`` 256 runs the 256-column program in 64-element steps; a caller-forced ``block_k`` reaches
    ``instance_key`` as given (32 is refused outside ``w8relf`` on the 128-column tile, 64 names the 64-element
    program).  Where a caller that does not force ``form`` forces the single-CTA form, a split (``sk`` other than
    ``"auto"`` or ``False``) or a knob the form yields to, the rule's ``block_n`` and ``block_k`` yield with it: the
    rule's 256-column program runs the plan asked for.  A caller-forced ``form`` ``w8relf`` keeps the rule's tile and
    K step (with ``pro`` 0, the rule's own program; ``instance_key`` admits or refuses the other knobs).  ``kser``
    (``_ks<P>``) is the
    row's serpentine K walk - taken from the rule (the sm_103a T-exact rules of the 12288 -> 4096 forward and the
    4096 -> 12288 input gradient at T 16172 carry ``kser`` 76) and dropped where the planned program cannot carry it
    (the single-CTA form, an exact, tail-first or synchronised split, the transposed staging, the whole-tile tile
    resolve, a batch raster, the eight-warp form) or where the device's CTA pairs are not P (74 pairs of a 148-SM part,
    80 of a 160-SM part: the reversed blocks would not be the device's waves).  ``kser`` is a rule knob only, not a
    planner keyword (the registry fallback never names a ``_ks<P>`` program).  It changes neither a launch argument
    nor the plan arithmetic; the output bits are those of another FP32 summation order than the row's program without
    it, fixed run to run.
    [Cake K1 kernel host L1234-L1358]

    One package-only deviation: the Cake host applies ``swap_small_m`` unconditionally because it compiles
    the swapped instance on demand, while this package can only launch the generated programs registered for
    ``arch``.  When the plan resolves to an unregistered instance (a batched forward / input gradient with a
    tiny T that engages the swap, or a per-row rule whose template was only generated for the contract's T),
    the plan is redone without the swap and / or without the rule (``plan.swap_fallback`` /
    ``plan.rule_fallback``) so the call still runs; the result is identical, only the tile walk differs.  When
    that is still unregistered, the nearest generated program of the row's layout / output class is planned
    instead (``plan.knob_fallback``): the hand-ordered knob variants first (``_FALLBACK_KNOB_VARIANTS``: the plain
    TMA L2 policy, the single-CTA form, the 64-row pair family, the tall tile, the register epilogue, BLOCK_N 128,
    each in the three ``pro`` forms), then every generated program of the class, nearest symbol first, with the
    knobs its symbol decodes to (``_registry_variants``); a caller-forced knob wins over a variant's.  Only a class
    without any generated program on ``arch`` keeps an unregistered plan (the launch binding raises
    ``NotImplementedError`` naming it).  ``_fallback=False`` returns the pure mirror (unit tests of the planner).

    A second package-only deviation, the exact-wave gate (``plan.wave_fallback``): a rule's BLOCK_N 224
    (``EXACT_WAVE_BLOCK_N``) was measured for the whole-wave fit of its tiles on the 76-pair devices
    (:func:`exact_waves`); on a device whose pairs do not fit them in whole waves and where the default tile needs fewer
    waves (74 pairs of a 148-SM part: the 4096 x 4096 weight gradient's 152 tiles are 2.05 waves against 1.73, the
    4096 x 1024 row's 76 tiles 1.03 against 0.86) the rule's BLOCK_N yields to the default tile; the rule's other knobs
    stay, a caller-forced ``block_n`` is never touched, and the plan is the launcher's wherever the fit is exact or both
    tiles need the same waves (152 and 160 SMs).  A rule's ``wave_yield`` names the knobs the row joins where its
    BLOCK_N yields (the 4096 x 4096 BF16 weight gradient on 148 SMs: the synchronised split of its 54-tile tail, slab
    path 2, margin 32, measured on B200 and B300 at T 8192 - 32768; below T 8192 its ``ROW_RULE_MIN_T`` tier plans the
    rule without it).

    A third package-only deviation, the synchronised TMA-store gate (:func:`_serves`): the synchronised stream-K
    programs that store their bf16 tiles through the TMA store (``_tma<slots>_sks``) were generated for measured
    rows whose rule names ``epi="tma"``; a ragged synchronised row whose short K (``K_TMA_BF16``) turns the bf16
    epilogue default to the TMA store plans the same symbol unmeasured, so the registry does not serve it there and
    the row runs the registry fallback above (as before those programs were generated); a caller-forced ``epi``
    is served, and the fp32-output programs (``_f32_tma<slots>_sks``: the TMA store by default at every K) are
    served as before.

    A fourth package-only deviation, the unbatched weight-gradient swap (:func:`_single_wgrad_swap`): an ``L == 1``
    call with both operands MN-major and no transposed store is the weight gradient ``G.T @ X``; where the rule table
    keys that row in the swapped orientation alone and its measured rule at that token count on the part is not empty
    (since version 0.3.24) or the token count is at least the row's ``_SINGLE_WGRAD_FOLLOW_MIN_K`` entry (since
    version 0.3.26), the call is planned swapped like the batched small-M swap (and under the same registry
    fallback), so it runs the plan of :func:`cake_dense_gemm_det_train_projection_wgrad` for the same math instead of
    padding its few output rows into a tall tile; a row without a rule whose output has at most 64 columns against
    4096 to 8192 rows and ``K >= 256`` is planned swapped too on ``_NARROW_N_SWAP_ARCHES`` (its columns go to the
    64-row tile of the transposed store instead of one padded 224- / 256-column tile).

    A fifth package-only deviation, the promoted chunks (``promote``, ``plan.promote_blocks``): once the program and the
    plan are settled, a plan whose longest segment holds more than :data:`K1_PROMOTE_MIN_SEGMENT` K blocks (64:
    4096 tokens; :func:`segment_blocks`) runs the promoted twin of its program (``_kc64``, :data:`K1_PROMOTE_BLOCKS`)
    where the architecture registers it (:func:`k1_promote_blocks`), except on the rows of
    :data:`K1_PROMOTE_EXEMPT_ROWS`: the same plan, each segment summed in chunks of at most 64 K blocks added in a fixed
    order, so the bits stay the same on every run.  A plan whose segments hold at most 64 K blocks (one chunk) keeps its
    program (versions 0.3.6 to 0.3.20: 128 K blocks, 8192 tokens, ``_kc128``).  ``promote=False`` never runs the twin,
    ``promote=True`` runs it at any length (on the exempt rows too) and raises ``ValueError`` where the program has no
    registered twin (:func:`promote_refusal`); neither changes the program or the plan.

    A sixth package-only deviation, the exact-split fit: a rule's ``sk_exact_fit`` names the CTA pair count on which
    the rule's exact split, where admitted and its ``parts * tail`` units exceed the pairs, takes ``max(2, pairs //
    tail)`` parts instead (one wave of units; the rule's ``sk_first`` and other knobs stay, a caller-forced
    ``sk_exact`` keeps its parts, and the rule's split stays where the fitted one is not admitted).  The 12288 -> 4096
    and 4096 -> 12288 BF16 weight gradients of sm_100a and sm_103a name 74 pairs (148 SMs): their 16-way split of a
    14-tile tail (224 units) becomes a 5-way split (70 units), measured on B200 and B300 at T 8192 - 32768; their
    plans on every other pair count are the launcher's.

    A seventh, the per-part rules: an exact key of :data:`ROW_RULES_BY_SM` for the device's architecture and SM count
    replaces the row's rule on that part alone, and so does a token range of :data:`ROW_RULE_RANGES_BY_SM` (a ragged
    key) at the token counts of the range, as an exact key at that token count would.  The 212-SM sm_107a ranges plan
    FP32 weight gradients of 4096 to 17408 features at T 4096 - 16384 (the synchronised split of the tail, or whole
    128-row tiles, or a two-way K split of the tail; since version 0.3.8 at T 12288 also the half-height tail wave or a
    three-way K split of the tail of the 256-row tiles) and the o_proj FP32 input gradient at T 8192 (the half-height
    tail wave), each measured faster on two VR200 nodes, and since version 0.3.7 four BF16 K-major rows at small token
    counts, measured on VR200: the o_proj forward (a four-way K split of the tail tiles at T 1 - 256, the 128-row
    program with the row's column-grouped raster at T 257 - 1024), the 17408 -> 5120 forward and the 5120 -> 17408 input
    gradient (a four-way split at T 16 - 256) and the 4096 -> 12288 input gradient (a four-way split at T 1 - 256; 0.89
    times as fast at T 1, where the plan before missed the naive tier).  Since version 0.3.8 the 148- and 152-SM sm_103a
    ranges and the 152-SM sm_100a ranges plan the FP32 input gradients 4096 -> 32 / 48 / 64 / 128,
    5120 -> 32 / 48 / 64 / 128, 8192 -> 128 and 6144 -> 128 (the rows of the 148-SM sm_100a part rules; on 148-SM
    sm_103a parts all but 5120 -> 128 and 8192 -> 128, at T 8192 - 32768 or narrower per row; on 152-SM sm_103a parts
    at T 16172 - 16384; on 152-SM sm_100a parts all but 8192 -> 128, at T 16172 - 16384, 5120 -> 128 at T 16172 and
    16384 alone and 6144 -> 128 at T 16172 alone) with the single-CTA program of those part rules, the bits of the plan
    before, measured faster on two B300, two GB300 and two GB200 nodes; on every other part and token count these rows
    plan as before.  Since version 0.3.9 the 152-SM sm_103a ranges run that program in raster groups of 2 on 8192 -> 128
    at T 16231 - 16384 and on the indexer_hw input gradient 6144 -> 32 at T 16172 - 32768 (the same bits, measured
    faster on two GB300 nodes).  Since version 0.3.10 the 148-SM sm_103a ranges also run it in raster groups of 2 on
    8192 -> 128 at T 8192 - 32768 and of 4 on indexer_hw 6144 -> 32 at T 8192, and the 148-SM sm_100a ranges in groups
    of 2 on 8192 -> 128 at T 8192 - 32768, 5120 -> 128 and 6144 -> 128 at T 16384 - 32768 and 6144 -> 32 at T 32768,
    and of 4 on 6144 -> 32 at T 16231 (the same bits, measured faster on two B300 and two B200 nodes).  Since version
    0.3.11 the 152-SM sm_100a ranges also run it in raster groups of 2 on the indexer_hw input gradient 6144 -> 32 at
    T 16172 - 32768, on 8192 -> 128 at T 16231 - 32768, on 5120 -> 128 at T 16231 and on 6144 -> 128 at
    T 16231 - 16384, and of 4 on 8192 -> 128 at T 16172 (the same bits, measured faster on two GB200 nodes).  Since
    version 0.3.12 the 148- and 152-SM sm_100a and sm_103a ranges also plan the BF16 input gradients
    4096 -> 32 / 48 / 64 / 128, 5120 -> 32 / 48 / 64 / 128, 8192 -> 128 and 6144 -> 32 / 128 with the start-path
    eight-warp program ``kn_n256_tma1_w8relf`` (on 152-SM sm_100a parts 5120 -> 48 at T 32768 with
    ``kn_n256_tma1_w8rel``; on 152-SM sm_103a parts all but 5120 -> 64 and 8192 -> 128 up to version 0.3.12), at
    T 16172 - 32768 or narrower per row and part (the same bits, measured faster on two nodes of each part); a caller
    that forces another ``form`` there plans with the row's rule without the part tables.  Since version 0.3.13 the
    152-SM sm_103a ranges run that program in raster groups of 32 on 4096 -> 32 at T 16172 and 32768 and on
    5120 -> 64 at T 16172, and the 152-SM sm_100a ranges in raster groups of 2 on 5120 -> 32 at T 32768 (the same bits,
    measured faster on two GB300 and two GB200 nodes; up to version 0.3.13).  Since version 0.3.14 the 148- and 152-SM
    sm_100a and sm_103a ranges run the start-path program ``w8relf`` on its 128 x 128 tile, ``kn_n128_tma1_w8relf`` on
    4096 -> 64 and 5120 -> 64 and ``kn_n128_k32_tma1_w8relf`` (32-element K steps) on 4096 -> 32, 5120 -> 32 and
    indexer_hw 6144 -> 32, in the raster group each range names (2, 16 or 32), at T 16172 - 32768 or narrower per row
    and part (on 148-SM parts not on 5120 -> 64, on 148-SM sm_103a parts not on 4096 -> 64 either; the same bits,
    measured faster on two nodes of each part); the other token counts of these rows plan as in version 0.3.13 (on
    4096 -> 32 and 4096 -> 64 of the 152-SM parts and 4096 -> 64 of 148-SM sm_100a parts, at some token counts, up to
    version 0.3.14).  Since version 0.3.15 the 128-column ranges of these five row and part pairs take in six cells
    version 0.3.14 kept, and the unmeasured token counts between them, each in the raster group of the range next to
    it (the same bits, measured faster than the plan of version 0.3.14 on two nodes of each part): on 152-SM sm_103a
    parts 4096 -> 32 at T 16172 - 16231 and 4096 -> 64 at T 16172 - 32768 in groups of 32, on 152-SM sm_100a parts
    4096 -> 32 at T 16172 - 32768 in groups of 2 and 4096 -> 64 at T 16384 - 32768 in groups of 16, and on 148-SM
    sm_100a parts 4096 -> 64 at T 16172 - 16231 in groups of 2.  Since version 0.3.19 the 152-SM sm_100a range of
    4096 -> 64 runs the 128-column program in groups of 16 at T 16173 - 16383 too, one range at T 16172 - 32768 (the
    same bits, measured faster than the plan of version 0.3.18 at T 16231 on four GB200 nodes).  Since version 0.3.20
    the 212-SM sm_107a ranges also plan the BF16 input gradients 4096 -> 48, 4096 -> 128, 5120 -> 48, 5120 -> 128,
    8192 -> 128 and 6144 -> 128 at T 16172 - 16384 with a start-path program in groups of 16, ``kn_n256_tma1_w8relf``
    (on 5120 -> 128 ``kn_n128_tma1_w8relf``), where the rows' rules plan ten-warp programs (measured faster than the
    plan of version 0.3.19 on two VR200 nodes, within the naive BF16 tier, the same bits
    on every run), and the 152-SM sm_100a range of 4096 -> 48 runs ``kn_n256_tma1_w8relf`` at T 16232 - 32767 too,
    one range at T 16172 - 32768 (the same bits, measured faster than the plan of version 0.3.19 at T 16300, 16384 and
    24576 on two GB200 nodes), and the 152-SM sm_100a ranges of 5120 -> 128 run ``kn_n256_tma1_w8relf`` at T 16384 too,
    a range of one token count (the same bits, measured faster than the plan of version 0.3.19 there on two GB200
    nodes)."""
    _caller_knobs: dict[str, Any] = dict(
        transposed_out=transposed_out,
        sk=sk,
        block_n=block_n,
        stages=stages,
        epi=epi,
        slots=slots,
        cta_rows=cta_rows,
        pf=pf,
        sk_max_units=sk_max_units,
        sk_parts=sk_parts,
        promo=promo,
        hints=hints,
        group_m=group_m,
        f32_v8=f32_v8,
        quad_store=quad_store,
        park=park,
        ovl=ovl,
        htail=htail,
        b_swz=b_swz,
        sk_exact=sk_exact,
        batch_group=batch_group,
        store_ef=store_ef,
        store_hint=store_hint,
        pd=pd,
        sh=sh,
        sk_sync=sk_sync,
        sk_sync_m=sk_sync_m,
        sk_slab=sk_slab,
        cta1=cta1,
        cgrp=cgrp,
        ovr=ovr,
        stt=stt,
        pro=pro,
        sk_first=sk_first,
        lw=lw,
        mg=mg,
        sw=sw,
        form=form,
        promote=promote,
        block_k=block_k,
        arch=arch,
    )
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError("cake_dense_gemm_det_train_projection: A and B must be bf16")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            "cake_dense_gemm_det_train_projection: out must be bf16 or fp32"
        )
    A3, B3, O3 = as_batched(A, "A"), as_batched(B, "B"), as_batched(out, "out")
    L, M, K = (int(s) for s in A3.shape)
    if tuple(B3.shape) != (L, K, int(B3.shape[2])):
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: B must be [L={L}, K={K}, N], got {tuple(B3.shape)}"
        )
    N = int(B3.shape[2])
    expect = (L, N, M) if transposed_out else (L, M, N)
    if tuple(O3.shape) != expect:
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: out must be {expect}, got {tuple(O3.shape)}"
        )
    if N % 8 != 0 or M < 1 or K < 1 or N < 8:
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: N must be a positive multiple of 8 and M, K >= 1 (M={M}, N={N}, K={K})"
        )
    if (
        O3.stride(2) != 1
        or (O3.shape[1] > 1 and O3.stride(1) % 8 != 0)
        or (L > 1 and O3.stride(0) % 8 != 0)
        or out.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "cake_dense_gemm_det_train_projection: out needs unit inner stride, 16-byte aligned rows / batches "
            f"and base; strides {tuple(O3.stride())}"
        )
    if (
        _use_rules
        and _fallback
        and _allow_swap
        and sk == "auto"
        # ``promote`` None or False chooses no program (the chain programs have no promoted chunks)
        and not promote
        and all(
            v is None
            for k, v in _caller_knobs.items()
            if k not in ("transposed_out", "sk", "arch", "promote")
        )
    ):
        # the FP32 weight-gradient rows of ``CHAIN_RANGES_BY_SM`` on this part: the chain program
        chained = _chain_route(
            A3, B3, O3, out, transposed_out=transposed_out, arch=arch, sm_count=sm_count
        )
        if chained is not None:
            # a ChainGemmPlan carries every CakeDenseGemmDetProjectionPlan field the launch path reads (template, grid, workspace sizes)
            return cast(
                "tuple[CakeDenseGemmDetProjectionPlan, torch.Tensor, torch.Tensor, torch.Tensor]",
                chained,
            )
    swap_eligible = swap_small_m(L, M, N, transposed_out) or (
        L == 1
        and not transposed_out
        and _single_wgrad_swap(
            arch,
            M,
            N,
            K,
            operand_view(A3, "A", k_axis=2)[0],
            operand_view(B3, "B", k_axis=1)[0],
            out.dtype == torch.float32,
            sm_count=sm_count,
        )
    )
    swapped = _allow_swap and swap_eligible
    if swapped:
        A3, B3 = B3.transpose(1, 2), A3.transpose(1, 2)
        M, N = N, M
        transposed_out = True
    a_mn, a_desc = operand_view(A3, "A", k_axis=2)
    b_mn, b_desc = operand_view(B3, "B", k_axis=1)
    # Round 33: ``row_rule`` lays the row's MEASURED rule over the architecture's ``ARCH_RULE_DEFAULTS`` (the
    # launcher's merge).  The package-only rule fallback (``_use_rules=False``) drops the measured rule alone and
    # keeps the architecture default: it is what a row without a rule plans with, and the generated pair programs of
    # the architecture carry the default's ``_pro{n}`` form (the pro-0 pair programs are not registered since round 33).
    rule_dropped = (
        {}
        if _use_rules
        else _measured_row_rule(
            arch,
            a_mn,
            b_mn,
            out.dtype == torch.float32,
            transposed_out,
            L > 1,
            N,
            K,
            M,
            sm_count=sm_count,
        )
    )
    rule = (
        row_rule(
            arch,
            a_mn,
            b_mn,
            out.dtype == torch.float32,
            transposed_out,
            L > 1,
            N,
            K,
            M,
            sm_count=sm_count,
        )
        if _use_rules
        else dict(ARCH_RULE_DEFAULTS.get(arch, {}))
    )
    if (
        _use_rules
        and form is not None
        and str(form or "") != rule.get("form", str(form or ""))
    ):
        # package-only: a part rule or token range that names a CTA form (the start-path ranges of version 0.3.12)
        # holds for that form alone; a caller that forces another form (``""``, the ten-warp program, included) plans
        # with the row's rule without the part tables, the rule its knobs were chosen against
        outside = row_rule(
            arch, a_mn, b_mn, out.dtype == torch.float32, transposed_out, L > 1, N, K, M
        )
        if outside.get("form") != rule["form"]:
            rule = outside
    if cta_rows is None:
        cta_rows = rule.get("cta_rows", default_cta_rows(M, N, K))
    elif int(cta_rows) != rule.get("cta_rows", default_cta_rows(M, N, K)):
        # A caller-forced tile family (sweeps, the registrations of the tall / 64-row families): the rule's knobs
        # that belong to its own family (tile width, stage count, staging slots, epilogue mode) do not carry over
        # (a 160-column rule of the 256-row family is not a valid 64-row tile); the synchronised stream-K plan (round 18) is
        # admitted per family too (its unit arithmetic counts the family's pair tiles), so it does not carry over either.
        # [Cake launcher]
        rule = {
            k: v
            for k, v in rule.items()
            if k
            not in (
                "block_n",
                "block_k",
                "stages",
                "slots",
                "epi",
                "ovl",
                "htail",
                "b_swz",
                "pd",
                "sk_sync",
                "sk_sync_m",
                "sk_slab",
                "cta1",
                "ovr",
                "stt",
                "pro",
                "lw",
                "mg",
                "sw",
                "kser",
                # package-only: the exact-split fit and the exact-wave yield were measured on the rule's own family
                "sk_exact_fit",
                "wave_yield",
            )
        }
    if cta1 is None:
        # round 19 (Cake W3): the single-CTA form of the 128-row family (no pair handoff)  [Cake launcher]
        cta1 = bool(rule.get("cta1", False))
    cta1 = bool(cta1)
    if cgrp is None:
        # round 21 (Cake W3): column-grouped raster (group_m = column tiles per group; B-panel L2 residency)  [Cake launcher]
        cgrp = rule.get("cgrp", False)
    cgrp = bool(cgrp)
    # CTAs per work item: the launch grid, the tile count and the scheduler units follow it  [Cake launcher]
    cg = 1 if cta1 else CTA_GROUP
    if cta1 and (
        sk is True
        or sk_exact
        or sk_sync
        or htail
        or sk_parts
        or sk_max_units
        or (batch_group and L > 1)
    ):
        # the single-CTA form carries no tail policy (its one-K-step rows never split) and no pair-only knob; a caller
        # forcing one with cta1 asked for two incompatible forms  [Cake launcher]
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: cta1 excludes sk / sk_exact / sk_sync / htail / sk_parts / sk_max_units / batch_group "
            f"(got sk={sk}, sk_exact={sk_exact}, sk_sync={sk_sync}, htail={htail}, sk_parts={sk_parts}, "
            f"sk_max_units={sk_max_units}, batch_group={batch_group})"
        )
    if cta1:
        # the row rule's tail policy / raster knob does not carry over to the single-CTA form  [Cake launcher]
        sk, sk_exact, sk_sync, htail, batch_group = False, None, False, False, 0
    if cta1 != bool(rule.get("cta1", False)):
        # a caller-forced switch between the pair and the single-CTA form: the rule's stage count belongs to its own
        # form (a 48 KiB cta1 stage at BLOCK_N 256 fits 3-4 deep where the 32 KiB pair stage fits 5-7); the instance
        # default of the forced form applies  [Cake launcher]
        rule = {k: v for k, v in rule.items() if k != "stages"}
    if (
        form is None
        and rule.get("form") in _W8_N128_FORMS
        and (rule.get("block_n") == 128 or rule.get("block_k", BLOCK_K) != BLOCK_K)
        and (
            cta1
            or sk not in ("auto", False)
            or any(_caller_knobs.get(k) is not None for k in _W8_FIXED)
        )
    ):
        # package-only: the 128-column tile and its K step belong to the eight-warp program; where a caller-forced knob
        # makes the form yield (the whole-tile guard before instance_key) or asks for a split, they yield with it, and
        # the rule's 256-column program runs the plan asked for
        rule = {k: v for k, v in rule.items() if k not in ("block_n", "block_k")}
    wave_fallback = False
    if block_n is None and rule.get("block_n") == EXACT_WAVE_BLOCK_N:
        # Package-only exact-wave gate: the rule's 224-column tile fills the measuring device's CTA pairs in whole
        # waves (``exact_waves``); where the running device's pairs do not, and the default tile needs fewer waves, the
        # narrower tile would cost a nearly empty extra wave, so the rule's BLOCK_N yields (its other knobs stay).  A
        # rule's ``wave_yield`` names the knobs the row was measured with on such a device and joins them (the 4096 x
        # 4096 BF16 weight gradient on 148 SMs: the synchronised split of its 54-tile tail)
        m_tiles_g = _ceil_div(M, int(cta_rows))
        m_tiles_g += m_tiles_g % cg
        units_g = sm_units(sm_count, cg)
        tiles_rule = L * (m_tiles_g // cg) * _ceil_div(N, EXACT_WAVE_BLOCK_N)
        tiles_default = L * (m_tiles_g // cg) * _ceil_div(N, default_block_n(N, b_mn))
        if not exact_waves(tiles_rule, units_g) and _ceil_div(
            tiles_rule, units_g
        ) > _ceil_div(tiles_default, units_g):
            yielded = dict(rule.get("wave_yield", {}))
            rule = {k: v for k, v in rule.items() if k not in ("block_n", "wave_yield")}
            rule.update(yielded)
            wave_fallback = True
    # the rule's 128-column eight-warp tile (a caller-forced 128 keeps it; the whole-tile guard of ``form`` admits it)
    rule_n128 = (
        rule.get("form") in _W8_N128_FORMS
        and rule.get("block_n") == 128
        and (block_n is None or int(block_n) == 128)
    )
    if block_n is None:
        block_n = rule.get("block_n", default_block_n(N, b_mn))
    if stages is None:
        stages = rule.get("stages")
    if epi is None:
        epi = rule.get("epi")
    if slots is None:
        slots = rule.get("slots")
    if sk == "auto" and "sk" in rule:
        sk = rule["sk"]
    if f32_v8 is None:
        f32_v8 = rule.get("f32_v8", False)
    if quad_store is None:
        quad_store = rule.get("quad_store", False)
    if park is None:
        park = rule.get("park", False)
    rule_ovl = ovl is None
    if ovl is None:
        ovl = rule.get("ovl", False)
    if ovr is None:
        # round 20 (Cake W3) / round 24: the overlapped tall epilogue released by row half, from the row's rule  [Cake launcher]
        ovr = rule.get("ovr", False)
        if ovr and ovl and not rule_ovl:
            # round 24: a caller-forced column-chunk release replaces the row's row-half release (the registrations of
            # the ``_ov`` programs at rows whose rule now carries ovr)  [Cake launcher]
            ovr = False
    elif ovr and rule_ovl:
        # a caller-forced row-half release replaces the row's column-chunk release (sweeps / registrations of the
        # ``_or`` programs); forcing both raises (instance_key)  [Cake launcher]
        ovl = False
    if ovr and int(block_n) != 256:
        # like ovl: built for the 256-column tall tile  [Cake launcher]
        ovr = False
    rule_stt = stt is None
    if stt is None:
        # round 27 (Cake lane 7, W-3 A1): the transposed stmatrix / TMA-store staging of the bf16 transposed register
        # epilogue, from the row's rule; a rule-derived value yields to a caller-forced form below, a caller-forced one is admitted or refused by instance_key  [Cake launcher]
        stt = rule.get("stt", False)
    stt = bool(stt)
    if pro is None:
        # round 33 (Cake unit 2d): pair prologue / cleanup sync form (bitmask; 0 = the round-27 form); the rule value is
        # the row's override of ARCH_RULE_DEFAULTS (7 on sm_107a / sm_100a); the single-CTA programs take none
        # (instance_key refuses pro with cta1)  [Cake launcher]
        pro = 0 if cta1 else rule.get("pro", 0)
    pro = int(pro)
    rule_lw = lw is None
    if lw is None:
        # the round-2 shapes tranche: the row's load-warp prelude form (the load warp leaves the pair prelude before the
        # post-allocation CTA rendezvous), from the rule
        lw = rule.get("lw", False)
    lw = bool(lw)
    if lw and rule_lw and (cta1 or not (pro & 4)):
        # the form belongs to the pair programs whose MMA warp allocates TMEM (``pro`` bit 4): a caller-forced single-CTA
        # form or another ``pro`` on the row keeps that form without it (a caller-forced ``lw`` is admitted or refused by
        # ``instance_key``)
        lw = False
    rule_mg = mg is None
    if mg is None:
        # the whole-tile tile resolve of the row's rule (the ``_mg`` programs): the load warp and the MMA warp walk
        # whole tiles with the raster quotients resolved through corrected fp32 reciprocals; the rules that carry it
        # pin ``sk: False`` (read below), and the whole-tile guard before ``instance_key`` drops it from a plan that
        # still carries stream-K units
        mg = rule.get("mg", False)
    mg = bool(mg)
    if mg and rule_mg and cta1:
        # the measured ``_mg`` programs are pair programs: a caller-forced single-CTA form keeps the row's single-CTA
        # program without it (a caller-forced ``mg`` is admitted or refused by ``instance_key``)
        mg = False
    rule_sw = sw is None
    if sw is None:
        # the paced output stores of the row's rule (the ``_sw<n>`` programs): each epilogue warp of the bf16
        # TMA-store pd-2 drain delays its bulk tensor store issue by a per-warp number of dependent multiply-add
        # steps; the drain guard before ``instance_key`` drops it from a plan that is not that drain
        sw = rule.get("sw", 0)
    sw = int(sw)
    if sw and rule_sw and cta1:
        # the measured ``_sw<n>`` programs are pair programs: a caller-forced single-CTA form keeps the row's
        # single-CTA program without it (a caller-forced ``sw`` is admitted or refused by ``instance_key``)
        sw = 0
    # the serpentine K walk of the row's rule (the ``_ks<P>`` programs): a rule knob only, not a planner keyword; the
    # wave guard before ``instance_key`` drops it from a plan that cannot carry it or from a device with another pair count
    kser = int(rule.get("kser", 0))
    rule_form = form is None
    if form is None:
        # the eight-warp CTA form of the row's rule (the ``_w8r`` / ``_w8rs`` / ``_w8`` / ``_w8rel`` / ``_w8relf``
        # programs): the whole-tile guard before ``instance_key`` drops it from a plan that is not whole-tile or that a
        # caller shaped with a tile, output, epilogue, tail, raster or pipeline knob
        form = rule.get("form", "")
    form = str(form or "")
    if form not in ("", *W8_FORMS):
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: form must be one of {W8_FORMS} (the eight-warp CTA form) or '', got {form!r}"
        )
    if htail is None:
        htail = rule.get("htail", False)
    if b_swz is None:
        b_swz = rule.get("b_swz", 128)
    rule_sk_exact = sk_exact is None
    if sk_exact is None:
        sk_exact = rule.get("sk_exact")
    rule_sk_first = sk_first is None
    if sk_first is None:
        # the round-2 shapes tranche: the tail-first order of the row's exact split, from the rule
        sk_first = rule.get("sk_first", False)
    sk_first = bool(sk_first)
    rule_sk_sync = sk_sync is None
    if sk_sync is None:
        # round 18 (Cake W2): synchronised stream-K tail (main units + collectors) of the row's rule
        sk_sync = rule.get("sk_sync", False)
    elif sk_sync:
        # a caller-forced synchronised plan replaces the row's other tail policies (the registrations of the ``_sks``
        # programs)  [Cake launcher]
        sk_exact, htail = None, False
    if sk_sync_m is None:
        # round 18 (Cake W2): main-unit margin in K steps (the collectors finish early; rule key ``sk_sync_m``)
        sk_sync_m = int(rule.get("sk_sync_m", 0))
    rule_sk_slab = sk_slab is None
    if sk_slab is None:
        # round 19 (Cake W2): slab path of the synchronised fixup (0 off, 1 early arrival, 2 + bulk slab read, 3 + fp32
        # output as the slab)  [Cake launcher]
        sk_slab = int(rule.get("sk_slab", 0))
    rule_batch_group = batch_group is None
    if batch_group is None:
        batch_group = rule.get("batch_group", 0)
    # the raster knob only exists for batched rows (one batch entry: identical raster)  [Cake launcher]
    batch_group = int(batch_group) if L > 1 else 0
    if rule_batch_group and batch_group > 1 and L % batch_group:
        # a rule's raster group that does not divide this call's batch count yields: the row's program without it
        batch_group = 0
    if batch_group < 0 or (batch_group > 1 and L % batch_group):
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: batch_group must be 0, 1 or a divisor of the batch size {L}, got {batch_group}"
        )
    if store_ef is None:
        store_ef = rule.get("store_ef", False)
    if store_hint is None:
        # round 15 (Cake W3): L2 eviction policy of the TMA-store epilogue's stores  [Cake launcher]
        store_hint = rule.get("store_hint", "none")
    if pd is None:
        pd = rule.get(
            "pd", 0
        )  # round 15 (Cake W1): pipelined TMEM drain of the 128-row family  [Cake launcher]
    if sh is None:
        sh = rule.get("sh", 0)
    if ovl and int(block_n) != 256:
        # the overlapped epilogue is built for the 256-column tall tile; a narrower tall tile keeps the serialized
        # tall epilogue  [Cake launcher]
        ovl = False
    if (
        sk_sync
        and rule_sk_sync
        and (ovl or ovr or park or int(pd) or htail or sk_exact)
    ):
        # The row's synchronised plan needs the single-pass or the serialised tall epilogue and is the row's only tail
        # policy (instance_key); a caller that forces another form on the row gets that form with the plain plan - the
        # rule's tail policy does not carry over (the Cake launcher also yields to its ``a_mcast`` attribution probe, which
        # the planner does not expose).  A caller-forced ``sk_sync`` keeps raising on such a conflict.  [Cake launcher]
        sk_sync = False
    m_tiles = _ceil_div(M, cta_rows)
    m_tiles += m_tiles % cg
    n_tiles = _ceil_div(N, block_n)
    k_blocks = _ceil_div(K, BLOCK_K)
    # work items = tiles per CTA group (pair tiles, or CTA tiles under cta1)  [Cake launcher]
    pair_tiles = L * (m_tiles // cg) * n_tiles
    # concurrent work items: CTA pairs, or SMs under cta1 (``_sm_units``)  [Cake launcher]
    units = sm_units(sm_count, cg)
    # Round 13 (Cake W3): a ``sk_exact`` rule / kwarg is the row's tail policy (exact p-way K split of every tail
    # tile, one unit per part); it pre-empts ``sk_parts`` and ``htail`` exactly as the Cake launcher orders them.
    exact_plan = (
        sk_exact_plan(pair_tiles, k_blocks, units, int(sk_exact))
        if (sk_exact and sk == "auto" and sk_max_units is None)
        else None
    )
    if (
        exact_plan is not None
        and rule_sk_exact
        and units == int(rule.get("sk_exact_fit", 0))
        and exact_plan[2] > units
    ):
        # Package-only exact-split fit: on the CTA pair count the rule names (``sk_exact_fit``), a rule split whose
        # units exceed the pairs (a second partial wave of units) takes ``max(2, pairs // tail)`` parts instead - one
        # wave of units, the rule's tail-first order and other knobs kept; the rule's split stays where the fitted one
        # is not admitted.  A caller-forced ``sk_exact`` keeps its parts.
        fit_parts = max(2, units // exact_plan[1])
        fitted = sk_exact_plan(pair_tiles, k_blocks, units, fit_parts)
        if fitted is not None:
            sk_exact, exact_plan = fit_parts, fitted
    # Round 18 (Cake W2): synchronised stream-K - n_t main units (K steps [0, s) of every tail tile) + C collector units
    # (the leftovers [s, k_blocks) of tail tiles c, c + C, ...), two partials and one slab per tail tile; admitted when
    # the plan is (``sk_sync_plan``, host arithmetic); it disables ``sk_parts`` and ``htail`` and yields to ``sk_exact``.
    sync_plan = (
        sk_sync_plan(pair_tiles, k_blocks, units, None, int(sk_sync_m))
        if (sk_sync and sk == "auto" and sk_max_units is None and exact_plan is None)
        else None
    )
    if sk_first and rule_sk_first and exact_plan is None:
        # the row's tail-first order is its exact split's: where the split is not admitted (no tail, a part under
        # ``SK_MIN_ITERS`` steps, a caller ``sk`` / ``sk_max_units`` / single-CTA form) the plain plan runs without it;
        # a caller-forced ``sk_first`` is admitted or refused by ``instance_key``
        sk_first = False
    parts = rule.get("sk_parts") if sk_parts is None else int(sk_parts)
    if sync_plan is not None:
        parts = None
    if sk == "auto" and sk_max_units is None and parts and exact_plan is None:
        # Measured per-row p-way split of the tail wave (Cake round 4, L20); a caller ``sk_parts`` wins over the
        # rule's.  [Cake L1296-L1301]
        sk, sk_max_units = sk_parts_plan(pair_tiles, k_blocks, units, int(parts))
    # Round 13 (Cake W1): deterministic half-height tail wave of the tall family - the tail tiles become 2 x tail
    # standard-geometry items of full K (no partial slabs, no fixup).  The knob is the row's (its rule or the caller:
    # the ``_ht`` program, ``instance_key`` field 17); the half-tail PLAN applies only when there is a tail and its
    # half items fit the CTA pairs, otherwise the ``_ht`` program runs the plain plan / stream-K policy of the row (the
    # round-2 shapes tranche measured the K 5120 thin input gradients that way: 1280 tiles on 76 pairs, tail 64).  The
    # fit never changes the program a rule or a caller named.  [Cake launcher]
    htail = (
        bool(htail)
        and int(cta_rows) in (128, 256)
        and exact_plan is None
        and sync_plan is None
    )
    if htail and int(cta_rows) == 128:
        # round 16 (Cake W2): the half-height tail wave on the standard family needs the 256-column row-major tile without the
        # pipelined drain / prefetch; the planner never emits box_rows or diag probes, so the launcher's remaining conditions
        # hold by construction  [Cake launcher]
        pf_eff = pf if pf is not None else rule.get("pf", default_pf(M, N, K))
        htail = (
            int(block_n) == 256
            and not transposed_out
            and int(pd) == 0
            and not int(pf_eff)
        )
    htail_plan = False
    if htail:
        tail_h = pair_tiles % units if pair_tiles > units else pair_tiles
        # Package-only refinement (like ROW_RULE_MIN_T): a rule's ``ht_min_tail`` keeps a tail of fewer tiles whole
        # (the ``_ht`` program then runs its whole-tile plan); no field means any tail that fits
        htail_plan = (
            bool(tail_h)
            and 2 * tail_h <= units
            and tail_h >= int(rule.get("ht_min_tail", 1))
        )
    if exact_plan is not None:
        num_full, tail_tiles, sk_units, iters_per_unit = exact_plan
    elif sync_plan is not None:
        # the ``_sks`` instance reads s in iters_per_unit and n_t in sk_iters; collectors = num_cluster_tiles - num_full - n_t
        num_full, tail_tiles, sk_units, iters_per_unit, _collectors = sync_plan
    elif htail_plan:
        num_full, tail_tiles, sk_units, iters_per_unit = (
            pair_tiles - tail_h,
            tail_h,
            2 * tail_h,
            k_blocks,
        )
    elif htail:
        # The ``_ht`` program reads every unit past ``num_full`` as a half-height item of full K (the half-tail
        # mapping above); it cannot run linearised stream-K units.  Where the half items do not fit, the program
        # runs the plain whole-tile plan, whatever ``sk`` the row or the caller asked for.
        num_full, tail_tiles, sk_units, iters_per_unit = pair_tiles, 0, 0, k_blocks
    else:
        num_full, tail_tiles, sk_units, iters_per_unit = stream_k_plan(
            pair_tiles, k_blocks, units, sk, sk_max_units
        )
    if tail_tiles * 16 > SK_DUMMY_BASE:
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: {tail_tiles} stream-K tail tiles exceed the {SK_DUMMY_BASE // 16} slice-counter budget"
        )
    out_f32 = out.dtype == torch.float32
    mode = epi_mode(out_f32, transposed_out, K, epi, block_n, cta_rows)
    nslots = epi_slots(mode, out_f32, block_n, K, slots, cta_rows, transposed_out)
    if (
        sync_plan is not None
        and rule_sk_slab
        and int(sk_slab) == 3
        and (not out_f32 or (mode == "reg" and not transposed_out))
    ):
        # Round 19 (Cake W2): the row's fp32-output-as-slab form needs the fp32 TMA-store or transposed register
        # epilogue; a caller that forces another epilogue on the row (the registrations / sweeps of the row-major f32_v8
        # programs) keeps the synchronised plan with the bulk slab read.  A caller-forced ``sk_slab`` keeps raising on
        # such a conflict (``instance_key``).  [Cake launcher]
        sk_slab = 2
    if sync_plan is not None and rule_sk_slab and int(sk_slab) == 2:
        stages_eff = (
            default_stages(nslots, cta_rows, block_n, b_mn, b_swz)
            if stages is None
            else int(stages)
        )
        if EPI_WARPS * epi_cols(block_n, cta_rows) * 128 > stages_eff * (
            cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz)
        ):
            # the row's bulk slab read does not fit a caller-forced narrower pipeline: the plain synchronised program
            # [Cake launcher]
            sk_slab = 0
    if stt and rule_stt:
        # Round 27 (Cake lane 7, W-3 A1): the row's staging form belongs to the bf16 transposed single-pass register epilogue of
        # the 128-row pair family and must stage its eight warp slots in the dead stage ring inside the opt-in (instance_key's
        # admission).  A caller that forces another form on the row (the e2e registrations / sweeps of the _t_tma1, _sks_sb2, pd,
        # cta1 and tall programs at the kv_a weight gradient row, a forced narrower pipeline) keeps that form WITHOUT the
        # staging: the rule's stt does not carry over, like the rule's sk_sync / sk_slab above.  A caller-forced stt keeps
        # raising on such a conflict (instance_key).  [Cake launcher]
        cols_stt = epi_cols(block_n, cta_rows)
        stage_b = int(cta_rows) * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg)
        stages_stt = (
            default_stages(nslots, cta_rows, block_n, b_mn, b_swz, cg, True)
            if stages is None
            else int(stages)
        )
        slab_eff = int(sk_slab) if sync_plan is not None else 0
        if (
            out_f32
            or not transposed_out
            or mode != "reg"
            or int(cta_rows) != 128
            or cta1
            or ovl
            or ovr
            or park
            or htail
            or int(pd)
            or cols_stt % 32
            or cols_stt > 256
            or slab_eff not in (0, 1)
            or EPI_WARPS * cols_stt * 64 > stages_stt * stage_b
            or stages_stt * stage_b
            + staging_bytes(nslots)
            + stt_bytes(True)
            + WORK_STAGES * 16
            > SMEM_OPT_IN
        ):
            stt = False
    if pf is None:
        pf = rule.get("pf", default_pf(M, N, K))
    if promo is None:
        promo = rule.get("promo", default_promo(m_tiles, n_tiles))
    promo = str(promo)
    if promo not in L2_PROMOS:
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: promo must be one of {L2_PROMOS}, got {promo!r}"
        )
    if group_m is None:
        group_m = rule.get(
            "group_m", default_group_m(a_mn, b_mn, m_tiles, pair_tiles, units)
        )
    group_m = int(group_m)
    if group_m < 1 or (cg == 2 and not cgrp and (group_m < 2 or group_m % 2)):
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: group_m must be an even number >= 2 (CTA pairs are adjacent row tiles; any count >= 1 under cta1, or under cgrp where it counts column tiles), got {group_m}"
        )
    if hints is None:
        # round 21 (Cake W3): the hint gate over the column raster's wave working set under cgrp  [Cake launcher]
        hints = rule.get(
            "hints",
            (default_hints_cgrp if cgrp else default_hints)(
                a_mn, b_mn, m_tiles, n_tiles, K, group_m, units, int(l2_bytes)
            ),
        )
    # The whole-tile guard of ``mg``: the ``_mg`` programs require a plan without stream-K units and fewer than 2**21
    # cluster tiles (the kernel's ``requires``), and exclude the raster / tail forms ``instance_key`` refuses with it.
    # A rule's ``mg`` yields where the plan or a caller-forced form breaks that (a caller ``sk`` / ``sk_max_units`` /
    # ``sk_parts``, an exact or synchronised split, ``htail``, ``cgrp``, ``stt``, ``sk_first``, a batch raster): the
    # plain program runs the plan asked for.  A caller-forced ``mg`` is refused on such a plan (the forms by
    # ``instance_key``, the stream-K units and the tile bound here).
    whole_tiles = sk_units == 0 and (num_full + sk_units) * CTA_GROUP < 2**22
    if mg and rule_mg:
        if (
            not whole_tiles
            or htail
            or cgrp
            or stt
            or sk_first
            or exact_plan is not None
            or sync_plan is not None
            or batch_group
        ):
            mg = False
    elif mg and not whole_tiles:
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: mg (whole-tile tile resolve) needs a whole-tile plan under 2**21 cluster tiles "
            f"(got {sk_units} stream-K units over {num_full + sk_units} cluster tiles)"
        )
    # The drain guard of ``sw``: the ``_sw<n>`` programs pace the bf16 TMA-store pd-2 drain and the kernel refuses the
    # knob elsewhere.  A rule's ``sw`` yields where the planned program is not that drain (a caller-forced register
    # epilogue or ``slots`` 0, another ``pd``, the fp32 output): the row's unpaced program runs the plan asked for.  A
    # caller-forced ``sw`` is admitted or refused by ``instance_key``.
    if sw and rule_sw and (mode != "tma" or nslots == 0 or int(pd) != 2 or out_f32):
        sw = 0
    # The whole-tile guard of ``form``: the eight-warp programs run whole kn bf16 128 x 256 pair tiles (``w8relf`` also
    # the 128 x 128 tile of its rule) through the
    # one-slot TMA-store epilogue under the kernel's cluster-tile bound, with none of the pair programs' pipeline knobs.
    # A rule's ``form`` yields where the plan or a caller-forced knob breaks that (stream-K units, the single-CTA form,
    # another tile, output class or epilogue, a tail or raster form, a forced pipeline knob - ``_W8_FIXED`` - or, for
    # the start-path forms ``w8rel`` / ``w8relf``, the rule's batch raster): the row's program above the form runs the
    # plan asked for.  Where it survives, the rule-derived pair-program knobs are cleared: the eight-warp programs carry
    # none of them.  A caller-forced ``form`` reaches ``instance_key`` with the caller's knobs and is refused there
    # (the stream-K units here).
    if form and rule_form:
        if (
            not whole_tiles
            or cta1
            or out_f32
            or transposed_out
            or a_mn
            or not b_mn
            or (int(block_n) != 256 and not (rule_n128 and int(block_n) == 128))
            or int(cta_rows) != 128
            or mode != "tma"
            or nslots != 1
            or htail
            or cgrp
            or stt
            or sk_first
            or exact_plan is not None
            or sync_plan is not None
            or batch_group == 1
            or (batch_group and form in _W8_START_FORMS)
            or any(_caller_knobs.get(k) is not None for k in _W8_FIXED)
        ):
            form = ""
        else:
            lw, mg, sw, pd, pro, stages = False, False, 0, 0, 0, None
            hints, pf, sh = _PLAIN_HINTS, 0, 0
            # the batched MLA programs: the head-interleaved raster G > 1 with the rule's evict_last store hint
            store_hint = (
                store_hint if batch_group > 1 and store_hint == "evict_last" else "none"
            )
            store_ef, quad_store, f32_v8 = False, False, False
    elif form and not whole_tiles:
        raise ValueError(
            f"cake_dense_gemm_det_train_projection: form (the eight-warp CTA form) needs a whole-tile kn bf16 plan on 128 x 256 pair "
            f"tiles ({_W8_N128_FORMS} also 128 x 128) under 2**21 cluster tiles (got {sk_units} stream-K units over "
            f"{num_full + sk_units} cluster tiles)"
        )
    if block_k is None:
        # the rule's K step holds for the 128-column w8relf program alone (a rule whose form or tile yielded drops it)
        block_k = (
            rule.get("block_k", BLOCK_K)
            if form in _W8_N128_FORMS and int(block_n) == 128
            else BLOCK_K
        )
    block_k = int(block_k)
    # The wave guard of ``kser``: the ``_ks<P>`` programs reverse the K walk of the whole-K items in every odd block of
    # P items, measured with P = the 76 CTA pairs of the 152-SM parts.  A rule's ``kser`` yields where the planned
    # program cannot carry it (the single-CTA form, an exact / tail-first / synchronised split, the transposed staging,
    # the whole-tile tile resolve, a batch raster, the eight-warp form) or where the device's pairs are not P.
    if kser and (
        cta1
        or exact_plan is not None
        or sk_first
        or sync_plan is not None
        or stt
        or mg
        or batch_group
        or form
        or units != kser
    ):
        kser = 0
    key = instance_key(
        a_mn=a_mn,
        b_mn=b_mn,
        out_f32=out_f32,
        out_t=transposed_out,
        block_n=block_n,
        stages=stages,
        diag=(),
        epi=mode,
        slots=nslots,
        box_rows=None,
        cta_rows=cta_rows,
        pf=pf,
        hints=hints,
        f32_v8=f32_v8,
        quad_store=quad_store,
        park=park,
        ovl=ovl,
        htail=htail,
        smem_limit=smem_limit_for(arch),
        b_swz=int(b_swz),
        sk_exact=exact_plan is not None,
        batch_group=batch_group,
        store_ef=bool(store_ef),
        store_hint=str(store_hint),
        pd=int(pd),
        sh=int(sh),
        sk_sync=sync_plan is not None,
        sk_slab=int(sk_slab) if sync_plan is not None else 0,
        cta1=cta1,
        cgrp=cgrp,
        ovr=ovr,
        stt=stt,
        pro=pro,
        sk_first=sk_first,
        lw=lw,
        mg=mg,
        sw=sw,
        form=form,
        kser=kser,
        block_k=block_k,
    )
    k_step = instance_block_k(key)
    if k_step != BLOCK_K:
        # the 128-column program's 32-element K steps: its plans are whole-tile (the form's guard above), so every item
        # walks the whole reduction in ceil(K / 32) steps, the count the program's A map requires
        if num_full != pair_tiles or sk_units or tail_tiles:
            raise ValueError(
                f"cake_dense_gemm_det_train_projection: block_k {k_step} needs a whole-tile plan "
                f"(got {sk_units} stream-K units over {num_full + sk_units} cluster tiles)"
            )
        k_blocks = iters_per_unit = _ceil_div(K, k_step)
    plan = CakeDenseGemmDetProjectionPlan(
        L=L,
        M=M,
        N=N,
        K=K,
        a_mn=a_mn,
        b_mn=b_mn,
        out_f32=out_f32,
        transposed_out=bool(transposed_out),
        block_n=int(block_n),
        cta_rows=int(cta_rows),
        stages=int(key[5]),
        epi=mode,
        slots=int(nslots),
        pf=int(pf),
        promo=str(promo),
        hints=tuple(key[12]),
        group_m=int(group_m),
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        pair_tiles=pair_tiles,
        sm_pairs=units,
        l2_bytes=int(l2_bytes),
        num_full=num_full,
        tail_tiles=tail_tiles,
        sk_units=sk_units,
        iters_per_unit=iters_per_unit,
        template=instance_symbol(key),
        swap_fallback=swap_eligible and not _allow_swap,
        rule_fallback=bool(rule_dropped),
        park=bool(key[15]),
        ovl=bool(key[16]),
        htail=bool(key[17]),
        htail_plan=htail_plan,
        b_swz=int(key[18]),
        sk_exact=int(sk_exact) if exact_plan is not None else 0,
        batch_group=int(key[20]),
        store_ef=bool(key[21]),
        store_hint=str(key[22]),
        pd=int(key[23]),
        sh=int(key[24]),
        sk_sync=bool(key[25]),
        sk_slab=int(key[26]),
        cta1=bool(key[27]),
        cgrp=bool(key[28]),
        ovr=bool(key[29]),
        stt=bool(key[30]),
        pro=int(key[31]),
        sk_first=bool(key[32]),
        lw=bool(key[33]),
        mg=bool(key[34]),
        sw=int(key[35]),
        form=str(key[36]),
        kser=int(key[37]),
        wave_fallback=wave_fallback,
        block_k=k_step,
    )
    check_stream_k_plan(plan, arch)
    if promote is None or promote:
        plan = _promoted_plan(plan, key, arch, forced=bool(promote))
    if _fallback and not _serves(
        arch, plan, _caller_knobs["epi"], _use_rules, sm_count=sm_count
    ):
        # nearest registered plan: drop the swap first (keeps the measured rule), then the rule, then both
        for allow_swap, use_rules in ((False, True), (True, False), (False, False)):
            if (allow_swap, use_rules) == (_allow_swap, _use_rules):
                continue  # the plan just made
            if (not allow_swap and not swapped) or (not use_rules and not rule):
                continue  # dropping a swap not taken / an empty rule changes nothing
            candidate = cake_dense_gemm_det_train_projection_plan(
                A,
                B,
                out,
                sm_count=sm_count,
                l2_bytes=l2_bytes,
                _fallback=False,
                _allow_swap=allow_swap,
                _use_rules=use_rules,
                **_caller_knobs,
            )
            if _serves(
                arch, candidate[0], _caller_knobs["epi"], use_rules, sm_count=sm_count
            ):
                return candidate
        # last resort: the nearest registered knob variant of the same layout / output kind (every knob set
        # computes the same GEMM; only the tile walk and the store path differ): the hand-ordered variants first
        # (``_FALLBACK_KNOB_VARIANTS``), then every generated program of the plan's layout / output class, nearest
        # by symbol first (``_registry_variants``), so a class the registry serves never fails to plan
        tried: set[tuple] = set()
        for allow_swap in (True, False) if swapped else (False,):
            try:
                base = cake_dense_gemm_det_train_projection_plan(
                    A,
                    B,
                    out,
                    sm_count=sm_count,
                    l2_bytes=l2_bytes,
                    _fallback=False,
                    _allow_swap=allow_swap,
                    _use_rules=False,
                    **_caller_knobs,
                )
            except ValueError:
                continue
            variants = (
                *_FALLBACK_KNOB_VARIANTS,
                *_registry_variants(
                    base[0].template, arch, L > 1, sk_exact_parts=rule.get("sk_exact")
                ),
            )
            for knobs in variants:
                merged = _merge_fallback_knobs(_caller_knobs, knobs)
                frozen = _frozen_knobs(merged)
                if merged == _caller_knobs or frozen in tried:
                    continue
                tried.add(frozen)
                try:
                    candidate = cake_dense_gemm_det_train_projection_plan(
                        A,
                        B,
                        out,
                        sm_count=sm_count,
                        l2_bytes=l2_bytes,
                        _fallback=False,
                        _allow_swap=allow_swap,
                        _use_rules=False,
                        **merged,
                    )
                except ValueError:
                    continue  # a knob the row cannot take (e.g. epi="tma" with a transposed store)
                if _serves(
                    arch, candidate[0], merged.get("epi"), False, sm_count=sm_count
                ):
                    return (
                        dataclasses.replace(candidate[0], knob_fallback=True),
                        *candidate[1:],
                    )
    return plan, a_desc, b_desc, O3


def _serves(
    arch: str,
    plan,
    caller_epi: Optional[str],
    use_rules: bool,
    *,
    sm_count: Optional[int] = None,
) -> bool:
    """Whether the registry of ``arch`` serves ``plan`` as planned (package-only, see
    ``cake_dense_gemm_det_train_projection_plan``): its program is generated, and a synchronised stream-K plan that stores its
    bf16 tiles through the TMA store (``_tma<slots>_sks``) has its epilogue named by the caller (``caller_epi``) or by
    the row's rule (``use_rules``: the merged row rule on a part of ``sm_count`` SMs; else the architecture default).
    Those bf16 programs were generated for the measured T-exact rows whose rule names ``epi="tma"``; a ragged row whose short K
    (``K_TMA_BF16``) turns the bf16 epilogue default to the TMA store plans the same symbol unmeasured and keeps the
    registry fallback.  The fp32-output programs (``_f32_tma<slots>_sks``) take the TMA store by default at every K
    and are served as before."""
    if plan.template not in KERNELS.get(arch, {}):
        return False
    if (
        not (plan.sk_sync and plan.epi == "tma")
        or plan.out_f32
        or caller_epi is not None
    ):
        return True
    rule = (
        row_rule(
            arch,
            plan.a_mn,
            plan.b_mn,
            plan.out_f32,
            plan.transposed_out,
            plan.L > 1,
            plan.N,
            plan.K,
            plan.M,
            sm_count=sm_count,
        )
        if use_rules
        else ARCH_RULE_DEFAULTS.get(arch, {})
    )
    return "epi" in rule


def segment_blocks(plan) -> int:
    """The K blocks of the longest segment one accumulator of ``plan`` sums: a whole tile's ``k_blocks`` where the plan
    stores whole tiles (every tile without stream-K units, the full waves beside them, the half-height tail's whole-K
    items), a stream-K unit's part of a tile (at most ``iters_per_unit``), a part of the exact split
    (``iters_per_unit``), the main part ``[0, s)`` or a collector's part ``[s, k_blocks)`` of the synchronised split
    (``s`` in ``iters_per_unit``)."""
    k_blocks = int(plan.k_blocks)
    longest = k_blocks if plan.num_full or not plan.sk_units else 0
    if plan.sk_units:
        part = min(int(plan.iters_per_unit), k_blocks)
        longest = max(longest, part, k_blocks - part if plan.sk_sync else 0)
    return longest


def _promote_twin_refusal(arch: str, key: tuple) -> str:
    """Why the program of instance key ``key`` (without promotion) has no promoted twin on ``arch``: the kernel's
    refusal (:func:`promote_refusal`) or a twin this architecture's registry does not hold; ``""`` when it has one."""
    twin = (
        key[: INSTANCE_KEY_FIELDS - 1] + (K1_PROMOTE_BLOCKS,) + key[INSTANCE_KEY_FIELDS:]
    )
    refusal = promote_refusal(twin)
    if refusal:
        return refusal
    template = instance_symbol(twin)
    if template not in KERNELS.get(arch, {}):
        return f"{template} is not registered for {arch}"
    return ""


def k1_promote_blocks(arch: str, key: tuple, segment: int) -> int:
    """The promoted chunk of a plan of instance key ``key`` (without promotion) on ``arch`` whose longest segment
    holds ``segment`` K blocks (:func:`segment_blocks`): :data:`K1_PROMOTE_BLOCKS` where the segment holds more than
    :data:`K1_PROMOTE_MIN_SEGMENT` blocks (64) and the architecture registers the program's promoted twin; else ``0``
    (the program without promotion).  The rule reads the segment's length and the registry only; the planner keeps the
    rows of :data:`K1_PROMOTE_EXEMPT_ROWS` without it."""
    if int(segment) <= K1_PROMOTE_MIN_SEGMENT or _promote_twin_refusal(arch, key):
        return 0
    return K1_PROMOTE_BLOCKS


def _promote_exempt(arch: str, plan) -> bool:
    """Whether ``plan``'s row is one of :data:`K1_PROMOTE_EXEMPT_ROWS`: its exact ``(N, K, M)`` key or its ragged
    ``(N, None, M)`` key."""
    ident = (
        arch,
        plan.a_mn,
        plan.b_mn,
        plan.out_f32,
        plan.transposed_out,
        plan.L > 1,
        plan.N,
    )
    return (
        ident + (plan.K, plan.M) in K1_PROMOTE_EXEMPT_ROWS
        or ident + (None, plan.M) in K1_PROMOTE_EXEMPT_ROWS
    )


def _promoted_plan(plan, key: tuple, arch: str, *, forced: bool):
    """``plan`` (of instance key ``key``) on the promoted twin of its program where the automatic rule names it
    (:func:`k1_promote_blocks`, not on a row of :data:`K1_PROMOTE_EXEMPT_ROWS`) or the caller forces it, else ``plan``.
    The twin keeps the plan: only the template and ``promote_blocks`` change.  A twin whose stream-K workspace contract
    the plan does not hold (:func:`check_stream_k_plan`) is not run.  ``forced`` raises ``ValueError`` where the program
    has no registered twin."""
    if forced:
        refusal = _promote_twin_refusal(arch, key)
        if refusal:
            raise ValueError(
                f"cake_dense_gemm_det_train_projection: promote=True, but {plan.template} has no promoted twin: "
                f"{refusal}"
            )
    elif not k1_promote_blocks(arch, key, segment_blocks(plan)) or _promote_exempt(
        arch, plan
    ):
        return plan
    promoted = dataclasses.replace(
        plan,
        template=instance_symbol(
            key[: INSTANCE_KEY_FIELDS - 1]
            + (K1_PROMOTE_BLOCKS,)
            + key[INSTANCE_KEY_FIELDS:]
        ),
        promote_blocks=K1_PROMOTE_BLOCKS,
    )
    try:
        check_stream_k_plan(promoted, arch)
    except ValueError:
        if forced:
            raise
        return plan
    return promoted


# Knob variants the registry fallback tries, nearest first (see ``cake_dense_gemm_det_train_projection_plan``): the plain TMA L2
# policy (the same tiles and plan; only the eviction hints of the loads differ), the single-CTA 128-row form, the
# 64-row pair family, the tall tile, the register epilogue (the canonical-T templates), then BLOCK_N = 128 with the same
# forms.  The TMA L2 promotion and the raster group width are launch parameters since round 11: they select no program
# and are not variants.
# Round 33: a row planned at a T outside the export contract takes the architecture default ``pro`` form (7), but
# the generated pair programs of a row may carry only the row's adopted form (15 = + the CTA-local exit) or the
# pinned round-27 form (0).  ``pro`` changes the pair prologue / cleanup sync only - never the GEMM, the tile walk or
# the store path - so a ``pro`` twin is the nearest registered program: the two other forms of the plan's own tile
# walk come first, then every tile variant in the three forms.  A single-CTA variant takes no pro (``instance_key``
# refuses it): ``_merge_fallback_knobs`` clears it with the tail knobs, so its three pro forms collapse into one try.
# A caller-forced tile family (``cta_rows`` away from the rule's) drops the rule's ``pro`` with the other family knobs
# [Cake launcher], so the forced-family variants must name the form: the architecture default 7 comes first (the form
# of most generated pair programs), then 15 and 0.
_FALLBACK_PRO_FORMS = (dict(pro=7), dict(pro=15), dict(pro=0))
_PLAIN_HINTS = ("none", "none")
_FALLBACK_TILE_VARIANTS = (
    dict(hints=_PLAIN_HINTS),
    dict(cta1=True),
    dict(cta1=True, hints=_PLAIN_HINTS),
    dict(cta_rows=64),
    dict(cta_rows=64, hints=_PLAIN_HINTS),
    dict(cta_rows=256),
    dict(cta_rows=256, hints=_PLAIN_HINTS),
    dict(epi="reg"),
    dict(epi="reg", f32_v8=True),
    dict(epi="reg", cta_rows=256),
    dict(block_n=128),
    dict(block_n=128, epi="reg"),
    dict(block_n=128, hints=_PLAIN_HINTS),
    dict(block_n=128, cta1=True),
    dict(block_n=128, cta_rows=64),
    dict(block_n=128, cta_rows=64, hints=_PLAIN_HINTS),
)
_FALLBACK_KNOB_VARIANTS = _FALLBACK_PRO_FORMS + tuple(
    dict(tile, **pro)
    for tile in _FALLBACK_TILE_VARIANTS
    for pro in ({}, *_FALLBACK_PRO_FORMS)
)
# The knobs the single-CTA form excludes (the planner raises on a forced one): a ``cta1`` variant clears them so the
# ``_c1`` programs stay reachable from a caller that forced a tail policy or a pair-only form on the row.
_CTA1_CLEARED = (
    "sk_exact",
    "sk_sync",
    "sk_sync_m",
    "sk_slab",
    "htail",
    "sk_parts",
    "sk_max_units",
    "batch_group",
    "pro",
    "sk_first",
    "lw",
)
# The caller knobs the eight-warp CTA form yields to (``cake_dense_gemm_det_train_projection_plan``): a value a caller forces for any
# of them shapes the pipeline, tile, epilogue, tail or raster the form cannot carry, so the row's ten-warp program runs
# the plan asked for.  Every name is a planner keyword whose default is None (a default call never yields).
_W8_FIXED = (
    "stages",
    "epi",
    "slots",
    "pf",
    "hints",
    "f32_v8",
    "quad_store",
    "park",
    "ovl",
    "htail",
    "b_swz",
    "sk_exact",
    "batch_group",
    "store_ef",
    "store_hint",
    "pd",
    "sh",
    "sk_sync",
    "sk_sync_m",
    "sk_slab",
    "cgrp",
    "ovr",
    "stt",
    "pro",
    "sk_first",
    "lw",
    "mg",
    "sw",
    "sk_max_units",
    "sk_parts",
)
_TEMPLATE_PREFIX = "dense_proj_det_gemm_"
_HINT_OF_LETTER = {"e": "evict_first", "l": "evict_last", "n": "none"}
_STORE_HINT_OF_LETTER = {
    "f": "evict_first",
    "l": "evict_last",
    "n": "evict_normal",
    "d": "default",
}
_HINT_TOKEN = re.compile(r"h[eln][eln]")


def _merge_fallback_knobs(
    caller: dict[str, Any], knobs: dict[str, Any]
) -> dict[str, Any]:
    """The caller's knobs over a fallback variant's (a caller-forced knob wins, as the plan it replaces); a ``cta1``
    variant clears the knobs the single-CTA form excludes (``_CTA1_CLEARED``) and resets ``sk`` to the launcher's
    automatic policy, which the planner zeroes under ``cta1``."""
    merged = dict(caller)
    for name, value in knobs.items():
        if merged.get(name) is None:
            merged[name] = value
    if knobs.get("cta1"):
        for name in _CTA1_CLEARED:
            merged[name] = None
        merged["sk"] = "auto"
    return merged


def _frozen_knobs(knobs: dict[str, Any]) -> tuple:
    """A hashable identity of a knob set (sequence-valued knobs as tuples)."""
    return tuple(
        (name, tuple(value) if isinstance(value, (list, tuple)) else value)
        for name, value in sorted(knobs.items())
    )


def _template_tokens(template: str) -> Optional[tuple[str, ...]]:
    """The symbol tokens after the family prefix (``kk_n256_hen_pro7`` -> ``('kk', 'n256', 'hen', 'pro7')``);
    ``None`` outside the family."""
    if not template.startswith(_TEMPLATE_PREFIX):
        return None
    return tuple(template[len(_TEMPLATE_PREFIX) :].split("_"))


def _template_class(tokens: tuple[str, ...]) -> tuple[str, bool, bool]:
    """``(layout, fp32 output, transposed store)`` of a symbol: the tokens no knob can change."""
    return tokens[0], "f32" in tokens, "t" in tokens


def _template_form(tokens: tuple[str, ...]) -> str:
    """The eight-warp CTA form token of a symbol (``w8r`` / ``w8rs`` / ``w8`` / ``w8rel`` / ``w8relf``), "" for the
    ten-warp programs: a class of its own for the registry fallback (a ten-warp row never falls onto an eight-warp
    program, nor the reverse)."""
    return next((token for token in tokens if token in W8_FORMS), "")


def _template_knobs(
    template: str, *, sk_exact_parts: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """The planner knobs that name ``template`` (the inverse of :func:`instance_symbol`).  Every knob the symbol can
    carry is set - an absent token to the value the symbol omits (plain hints, 128 rows, the register epilogue, no
    tail policy, no raster form, ``pro`` 0 on a pair program) - so a row of the template's class planned with them
    under ``_use_rules=False`` names the template unless the row cannot take a knob (the planner raises) or its
    arithmetic drops a tail policy (the plan names the template without it).  Launch parameters (promo, group_m) are
    not in the symbol and stay unset; ``_skx`` carries no part count, so ``sk_exact`` takes ``sk_exact_parts`` (the
    row's rule) or the kernel tests' 3 parts on 128-row / 2 on 64-row tiles (``_skf`` = its tail-first order,
    ``_lw`` = the load-warp prelude form, ``_mg`` = the whole-tile tile resolve, ``_sw<n>`` = the paced output stores,
    ``_ks<P>`` = the serpentine K walk - an ``instance_key`` field the planner reads from the rule only,
    ``_kc64`` = ``promote=True``, the promoted twin, and ``promote=False`` without the token,
    ``_w8r`` / ``_w8rs`` / ``_w8`` / ``_w8rel`` / ``_w8relf`` = the eight-warp CTA form, ``_k<K>`` = ``block_k``:
    set where the token is present, and to 64 on the 128-column ``w8relf`` program without it, on no other symbol); a
    single-CTA program leaves
    ``pro`` to the planner (``instance_key`` refuses a forced one).  ``None`` for a symbol outside the family or
    carrying a token the planner never emits (``_box<rows>``, a promoted chunk other than :data:`K1_PROMOTE_BLOCKS`,
    diagnostics)."""
    tokens = _template_tokens(template)
    if tokens is None or len(tokens) < 2:
        return None
    knobs: dict[str, Any] = dict(
        block_n=None,
        cta_rows=128,
        b_swz=128,
        pf=0,
        hints=_PLAIN_HINTS,
        f32_v8=False,
        store_ef=False,
        quad_store=False,
        park=False,
        ovl=False,
        htail=False,
        epi="reg",
        slots=None,
        stages=None,
        sk_exact=None,
        sk_sync=False,
        stt=False,
        sk_slab=0,
        batch_group=0,
        store_hint="none",
        pd=0,
        sh=0,
        cta1=False,
        cgrp=False,
        ovr=False,
        pro=0,
        sk_first=False,
        lw=False,
        mg=False,
        sw=0,
        form="",
        kser=0,
        promote=False,
    )
    for token in tokens[1:]:
        if token in ("f32", "t"):
            continue  # the class
        if m := re.fullmatch(r"n(\d+)", token):
            knobs["block_n"] = int(m[1])
        elif m := re.fullmatch(r"k(\d+)", token):
            knobs["block_k"] = int(m[1])
        elif token in ("m256", "m64"):
            knobs["cta_rows"] = int(token[1:])
        elif m := re.fullmatch(r"bz(\d+)", token):
            knobs["b_swz"] = int(m[1])
        elif m := re.fullmatch(r"pf(\d+)", token):
            knobs["pf"] = int(m[1])
        elif m := re.fullmatch(r"h([eln])([eln])", token):
            knobs["hints"] = (_HINT_OF_LETTER[m[1]], _HINT_OF_LETTER[m[2]])
        elif token == "v8":
            knobs["f32_v8"] = True
        elif token == "ef":
            knobs["store_ef"] = True
        elif token == "q":
            knobs["quad_store"] = True
        elif token == "pk":
            knobs["park"] = True
        elif token == "ov":
            knobs["ovl"] = True
        elif token == "ht":
            knobs["htail"] = True
        elif m := re.fullmatch(r"tma(\d+)", token):
            knobs["epi"], knobs["slots"] = "tma", int(m[1])
        elif m := re.fullmatch(r"s(\d+)", token):
            knobs["stages"] = int(m[1])
        elif token == "skx":
            knobs["sk_exact"] = int(
                sk_exact_parts or (2 if knobs["cta_rows"] == 64 else 3)
            )
        elif token == "skf":
            knobs["sk_first"] = True
        elif token == "sks":
            knobs["sk_sync"] = True
        elif token == "stt":
            knobs["stt"] = True
        elif m := re.fullmatch(r"sb(\d)", token):
            knobs["sk_slab"] = int(m[1])
        elif token == "bf":
            knobs["batch_group"] = 1
        elif m := re.fullmatch(r"bg(\d+)", token):
            knobs["batch_group"] = int(m[1])
        elif m := re.fullmatch(r"so([flnd])", token):
            knobs["store_hint"] = _STORE_HINT_OF_LETTER[m[1]]
        elif m := re.fullmatch(r"pd(\d+)", token):
            knobs["pd"] = int(m[1])
        elif m := re.fullmatch(r"sh(\d+)", token):
            knobs["sh"] = int(m[1])
        elif token == "c1":
            knobs["cta1"] = True
        elif token == "cg":
            knobs["cgrp"] = True
        elif token == "or":
            knobs["ovr"] = True
        elif token == "lw":
            knobs["lw"] = True
        elif token == "mg":
            knobs["mg"] = True
        elif m := re.fullmatch(r"sw(\d+)", token):
            knobs["sw"] = int(m[1])
        elif m := re.fullmatch(r"ks(\d+)", token):
            knobs["kser"] = int(m[1])
        elif token == f"kc{K1_PROMOTE_BLOCKS}":
            knobs["promote"] = True
        elif m := re.fullmatch(r"pro(\d+)", token):
            knobs["pro"] = int(m[1])
        elif token in W8_FORMS:
            knobs["form"] = token
        else:
            return None  # _box<rows> / a diagnostic: not a planner knob
    if knobs["block_n"] is None:
        return None
    if knobs["form"] in _W8_N128_FORMS and knobs["block_n"] == 128:
        # the 128-column eight-warp program names its K step: without ``_k<K>`` the 64-element one (a rule's 32-element
        # step would otherwise name the other program)
        knobs.setdefault("block_k", BLOCK_K)
    if knobs["cta1"]:
        knobs["pro"] = None
    return knobs


def _template_distance(wanted: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    """Tokens one symbol carries and the other does not; a hint token counts half (``_hen`` <-> plain is the
    nearest edit: the same tiles and plan)."""
    return sum(
        0.5 if _HINT_TOKEN.fullmatch(token) else 1.0
        for token in set(wanted) ^ set(candidate)
    )


def _registry_variants(
    template: str, arch: str, batched: bool, *, sk_exact_parts: Optional[int] = None
) -> tuple[dict[str, Any], ...]:
    """The knob sets of the generated programs of ``arch`` in the layout / output class and the CTA form of
    ``template``, nearest symbol first (:func:`_template_distance`, then the name); the batch-raster programs (``_bf``
    / ``_bg<n>``) only for a batched row (a batch of one plans no batch raster).  A promoted twin (``_kc64``) is no
    variant: the planner reaches it from its program (:func:`k1_promote_blocks`), so the variants leave ``promote`` to
    the caller."""
    wanted = _template_tokens(template)
    if wanted is None:
        return ()
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for candidate in KERNELS.get(arch, {}):
        tokens = _template_tokens(candidate)
        if tokens is None or candidate == template:
            continue
        if _template_class(tokens) != _template_class(wanted):
            continue
        if bool(_template_form(tokens)) != bool(_template_form(wanted)):
            # the eight-warp CTA form is a class of its own (its programs fall back onto each other only).  Only a
            # caller-forced form plans in it (the rule-free base names no form), and the caller's form wins over a
            # variant's (``_merge_fallback_knobs``): a variant lends its other knobs, never its form
            continue
        knobs = _template_knobs(candidate, sk_exact_parts=sk_exact_parts)
        if (
            knobs is None
            or (knobs["batch_group"] and not batched)
            or knobs.pop("kser")
            or knobs.pop("promote")
        ):
            # a ``_ks<P>`` program is planned from its rule only (``kser`` is not a planner keyword), a ``_kc64`` twin
            # from its program
            continue
        ranked.append((_template_distance(wanted, tokens), candidate, knobs))
    ranked.sort(key=lambda item: item[:2])
    return tuple(knobs for _, _, knobs in ranked)


# ---------------------------------------------------------------------------
# Launch binding (shared by K1 and K2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.entry(*self.arguments)


def _grid_fits_cluster(
    module_name: str, record: dict[str, Any], grid: tuple[int, int, int]
) -> None:
    cluster = record.get("launch", {}).get("cluster")
    if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"{module_name}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
        )


def _order_arguments(
    module_name: str, plan, values: dict[str, Any], grid: tuple[int, int, int]
) -> list:
    """``values`` in the order of the argument plan ``plan`` (``[kind, name]`` pairs); the ``grid`` entries come
    from ``grid``.  Fails closed: a keyword the kernel expects that the host does not provide raises ``KeyError``
    naming both sides."""
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in plan:
        if kind == "grid":
            arguments.append(grid_values[name])
        elif name in values and values[name] is not None:
            arguments.append(values[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    return arguments


def _bind(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> tuple[Callable[..., Any], tuple]:
    """``(entry, arguments)`` of ``module_name``: the loaded FFI entry and ``values`` ordered by the module's
    argument plan (``cake_jit.arg_plan_of``: a key into ``ARG_PLANS`` or an inline list)."""
    record = MODULES[module_name]
    _grid_fits_cluster(module_name, record, grid)
    arguments = _order_arguments(module_name, arg_plan_of(record), values, grid)
    module = load_cake_dense_projection_det_gemm_module(module_name, arch)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


def bind_launch(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``module_name`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide raises
    ``KeyError`` naming both sides.  The grid must be a multiple of the cluster shape baked
    into the module.
    """
    entry, arguments = _bind(module_name, values, grid, arch)
    return _Launch(module_name, entry, arguments, grid)


_FFI_DEVICES: dict[int, Any] = {}


_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_raw_stream(index: int) -> int:
    """torch's current stream on device ``index`` as a raw ``cudaStream_t`` value."""
    if _RAW_STREAM is not None:
        return int(_RAW_STREAM(index))
    return int(torch.cuda.current_stream(index).cuda_stream)


def _ffi_stream_context(index: int, raw: Optional[int] = None):
    """tvm-ffi environment-stream context for torch's current stream on device ``index`` (``raw`` when the
    caller already read it): the caller's downstream ops are stream-ordered after the kernel."""
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    if raw is None:
        raw = _current_raw_stream(index)
    return tvm_ffi.use_raw_stream(device, raw)


def _device_index(device: torch.device) -> int:
    return _cuda_index(device)


# Small tensors an instance never dereferences (the operands of the epilogue path it does not
# take), cached per device like the Cake launcher's: a 16-element buffer per pointer dtype and a
# ``[1, 32, 64]`` tensor covering both output boxes for the unused output descriptor.
_DUMMIES: dict[tuple, torch.Tensor] = {}


def _dummy_key(device: torch.device, dtype, tag: Optional[str] = None) -> tuple:
    """Per-device-ordinal key of a dummy (``cuda`` and ``cuda:0`` are one device; cheaper than ``str(device)``
    on the planning path)."""
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return (index, dtype, tag)


def _dummy(device: torch.device, dtype) -> torch.Tensor:
    key = _dummy_key(device, dtype)
    dummy = _DUMMIES.get(key)
    if dummy is None:
        dummy = _DUMMIES[key] = torch.zeros(16, device=device, dtype=dtype)
    return dummy


def _dummy_map(device: torch.device, dtype) -> torch.Tensor:
    """A ``[1, 256, 64]`` tensor for an output descriptor the instance never touches: it covers every output box (inner
    <= 64 elements, outer <= 256 rows - OUT_BOX_OUTER is EPI_COLS <= 256 on the stt programs (round 27), 32 / CH_T_COLS
    otherwise), so the dummy map encodes on every program.  [Cake ``_dummy_map``]"""
    key = _dummy_key(device, dtype, "map")
    dummy = _DUMMIES.get(key)
    if dummy is None:
        dummy = _DUMMIES[key] = torch.zeros((1, 256, 64), device=device, dtype=dtype)
    return dummy


def flat_alias(t: torch.Tensor) -> torch.Tensor:
    """Contiguous 1-D view over the storage span of ``t`` (same data pointer), for pointer
    parameters: the kernel addresses rows / batches through ``ldo`` / ``out_l``."""
    span = 1 + sum(
        (int(s) - 1) * int(st)
        for s, st in zip(t.shape, t.stride(), strict=True)
        if int(s) > 0
    )
    return torch.as_strided(t, (span,), (1,))


# ---------------------------------------------------------------------------
# Convenience-path cache (shared by K1 and K2)
# ---------------------------------------------------------------------------

# The convenience entry points keep the planned launches of a process per kind (K1 / K2) in a least-recently-used
# cache of this many entries.  An entry holds its plan, the resolved module entry and an argument template whose
# per-call operands -- the caller's views and the workspace tensors -- are ``_Slot`` markers, never a caller's
# tensor: the operand views are rebound from the caller's tensors on every call.  The workspace of a plan (K1:
# the stream-K partial slabs and the slice counters; K2: the BF16x3 stack, its fp32 scratch and the split-K
# partials) is not owned by the entry but carved at launch time out of the workspace arena of the entry's
# (device index, raw stream) -- one growable buffer per kind shared by every entry of that device and stream
# (:class:`_Arena`) -- so the memory held is bounded by the largest requirement seen per kind, however many
# geometries (token counts) the cache has planned; an arena is released when its last entry is evicted and by
# :func:`cake_dense_gemm_det_train_convenience_cache_clear`.  The key holds the raw stream, so launches on different streams never share
# a workspace, and launches of one device and stream -- the only users of its arena -- are ordered by that
# stream.  Threads: planning and the arena bookkeeping (creation, growth, reference counts, insertion, eviction,
# ``cake_dense_gemm_det_train_convenience_cache_clear``) are serialised by the module lock ``_PLAN_LOCK`` (a miss re-checks the cache
# under it, so a key is planned once); the K2 hit path holds its arena's lock from the carve of the workspace
# views through the split-K reduction, so two threads issuing router launches of different geometries on one
# stream cannot interleave their split / kernel / reduction on the shared BF16x3 stack and partials; the K1 hit
# path and the cache lookup are lock-free (one self-contained kernel per call; a hit whose entry another thread
# evicts between the lookup and the recency refresh is a miss).  The caller still owns the stream: a stream is
# assumed to have one issuer at a time in the sense of CUDA stream semantics.  The bound is read on every
# insertion, so a process may set it after the import.  A CUDA Graph should capture a prepared launch, not the
# convenience path: a growth or an eviction after capture would replace or free the workspace the graph replays.
CONVENIENCE_CACHE_SIZE = 64


@dataclass(frozen=True)
class _Slot:
    """A per-call argument of a cached launch (a view of the caller's tensors), filled in at launch time."""

    name: str


def _view_key(t: torch.Tensor) -> tuple:
    """The geometry of a view that fixes its plan and its binding: shape, strides, dtype, storage offset and
    the 16-byte alignment residue of its first element (the planner's alignment checks); never its address.
    (``shape`` and ``stride()`` are tuples of ints already; this is on the hit path.)"""
    return (
        tuple(t.shape),
        t.stride(),
        t.dtype,
        t.storage_offset(),
        t.data_ptr() & 15,
    )


def convenience_key(
    device_index: int, stream: int, flags: tuple, *views: torch.Tensor
) -> tuple:
    """Cache key of a convenience-path launch: the device, the raw stream, the entry point's flags / knobs and
    the geometry of every view (:func:`_view_key`)."""
    return (
        int(device_index),
        int(stream),
        tuple(flags),
        *(_view_key(t) for t in views),
    )


def _cache_get(cache: "OrderedDict[tuple, Any]", key: tuple) -> Any:
    """The cached entry of ``key`` (refreshed as the most recent) or ``None``: a miss, or a hit whose entry
    another thread evicted between the lookup and the refresh (the caller plans the key again)."""
    entry = cache.get(key)
    if entry is not None:
        try:
            cache.move_to_end(key)
        except KeyError:
            return None
    return entry


def _cache_put(
    cache: "OrderedDict[tuple, Any]",
    key: tuple,
    entry: Any,
    bound: Optional[int] = None,
    on_evict: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Insert ``entry`` as the most recent and evict the least recent entries beyond ``bound``
    (``CONVENIENCE_CACHE_SIZE`` at call time when ``None``); ``on_evict`` sees every entry that leaves the
    cache -- the evicted ones and an earlier entry of ``key`` that ``entry`` replaces."""
    if bound is None:
        bound = CONVENIENCE_CACHE_SIZE
    replaced = cache.get(key) if on_evict is not None else None
    cache[key] = entry
    cache.move_to_end(key)
    if replaced is not None and replaced is not entry:
        on_evict(replaced)
    while len(cache) > bound:
        _, evicted = cache.popitem(last=False)
        if on_evict is not None:
            on_evict(evicted)
    return entry


def _resolve_arguments(template: tuple, views: dict[str, torch.Tensor]) -> list:
    """The argument list of a cached launch: ``template`` with every ``_Slot`` replaced by the caller's view."""
    return [views[item.name] if type(item) is _Slot else item for item in template]


def _same_cuda_device(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor
) -> torch.device:
    device = out.device
    if (
        not (A.is_cuda and B.is_cuda and out.is_cuda)
        or A.device != device
        or B.device != device
    ):
        raise ValueError("Expected A, B and out on one CUDA device")
    return device


# ---------------------------------------------------------------------------
# Convenience-path workspace arena (one per device index and raw stream)
# ---------------------------------------------------------------------------

# The workspace kinds an arena carves: ``(dtype, zero-initialised)``.  The kinds are separate buffers -- a K1
# launch uses ``k1_ws`` + ``k1_counters``, a K2 launch ``k2_parts`` + ``k2_partials`` (+ ``k2_resid``, the fp32
# scratch of the torch-chain split, when the fused split program is not registered for the device) -- so no
# launch depends on the carving of another kind and the slice-counter zero contract has a buffer of its own.
_ARENA_KINDS: dict[str, tuple[torch.dtype, bool]] = {
    "k1_ws": (
        torch.float32,
        False,
    ),  # stream-K partial slabs (``CakeDenseGemmDetProjectionPlan.ws_f32_elems``)
    "k1_counters": (
        torch.uint32,
        True,
    ),  # stream-K slice counters (``CakeDenseGemmDetProjectionPlan.counters_alloc_u32``)
    "k2_parts": (
        torch.bfloat16,
        False,
    ),  # the BF16x3 stack of the split operand ``[3, *split_src.shape]``
    "k2_resid": (
        torch.float32,
        False,
    ),  # the fp32 scratch of the split ``split_src.shape``
    "k2_partials": (
        torch.float32,
        False,
    ),  # the split-K partials ``[splits, rows, cols]``
}
_ARENA_MIN_ELEMS = (
    16  # a pointer parameter the instance never dereferences still needs a buffer
)


@dataclass
class _Arena:
    """The workspace of every cached convenience-path launch of one (device index, raw stream): one buffer per
    kind of ``_ARENA_KINDS``, grown to the largest requirement a planned entry has asked for and never shrunk;
    an entry's workspace tensors are views of these buffers, resolved at launch time (:func:`_entry_views`).

    Why sharing one buffer per kind between every entry of the device and stream is safe:

    * every launch that reads or writes the arena is issued on this stream, and the hit path allocates nothing,
      so two uses of a buffer are always ordered by the stream;
    * growing replaces a buffer while an earlier launch on this stream may still be in flight on the old one.
      The old tensor stays referenced by the arguments of that launch until the FFI call has enqueued it, and
      once freed its block returns to PyTorch's caching allocator, which is stream-ordered: a block is handed
      out again only to a later allocation on the stream it was freed on (whose kernels run after the launch
      in flight), or after a ``cudaFree`` that synchronises the device.  The new buffer is allocated on this
      stream (the current stream of the calling thread is the arena's stream);
    * ``k1_counters``: the stream-K zero contract -- the kernel resets every counter it uses (the last arriver
      of a tail tile's slice writes zero after the fixup) and the lane-spread dummy counters above
      ``SK_DUMMY_BASE`` only ever receive a fetch-add of 0 -- holds for a shared buffer because the launches
      are serialised by the stream and each one leaves the whole buffer zero again (the buffer is
      zero-initialised when allocated, by a memset on the same stream).  This is the contract the Cake
      launcher's per-device counters rely on;
    * ``k2_parts`` / ``k2_resid``: the split rewrites the stack (and, under the torch chain, the scratch) in full
      before every launch (same stream);
    * ``k2_partials``: written by the kernel and consumed by ``torch.sum(..., out=out)`` right after the
      launch on the same stream; the next launch overwrites them only after that reduction has read them;
    * ``k1_ws``: the stream-K slabs carry no state between launches (the first arriver of a slice stores, the
      others add).

    ``version`` counts the growths: an entry memoises the views it carved against the version it read before
    carving and carves them again after a growth (:func:`_entry_views`); a growth also forgets every memo of
    the arena, so no cached entry keeps a replaced buffer alive.  ``refs`` counts the cached entries of the
    arena; the last eviction releases it (:func:`_release_entry`).

    Threads.  The stream-ordering argument above assumes the launches of one stream are issued in the order
    their host work happened, which one issuing thread per stream gives for free.  With several threads on one
    stream: ``require`` / ``refs`` / the registry are only touched under ``_PLAN_LOCK`` (planning is
    serialised; ``require`` is also monotone -- it keeps the larger of its own allocation and a concurrently
    stored buffer), and the K2 hit path holds ``lock`` from the carve of its workspace views through the
    split-K reduction (:func:`_launch_cached_router`), because a K2 call is three stream operations (the
    split, the kernel, the reduction) on the shared BF16x3 stack and partials, which two threads must not
    interleave.  A K1 call is one self-contained kernel (the slabs carry no state, the counters self-reset),
    so the K1 hit path takes no lock.  A memo stored by a thread that carved before a concurrent growth is
    stamped with the pre-growth version and is carved again on its next use.
    """

    device: torch.device
    key: tuple[int, int]  # (device index, raw stream) -- the key of ``_ARENAS``
    buffers: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    version: int = 0
    refs: int = 0
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def require(self, kind: str, numel: int) -> torch.Tensor:
        """The buffer of ``kind`` with at least ``numel`` elements, grown (replaced by a larger allocation on the
        current stream) when the present one is smaller; called when an entry is planned (under ``_PLAN_LOCK``),
        never on a hit.  Monotone: if a larger buffer was stored meanwhile, that one is kept."""
        buf = self.buffers.get(kind)
        numel = max(int(numel), _ARENA_MIN_ELEMS)
        if buf is None or buf.numel() < numel:
            dtype, zeroed = _ARENA_KINDS[kind]
            new = (torch.zeros if zeroed else torch.empty)(
                numel, device=self.device, dtype=dtype
            )
            buf = self.buffers.get(kind)  # re-read: keep the larger buffer
            if buf is None or buf.numel() < new.numel():
                buf = self.buffers[kind] = new
            self.version += 1
            _forget_arena_views(self)
        return buf

    def nbytes(self) -> int:
        """Bytes the arena holds (every kind)."""
        return sum(int(b.numel()) * b.element_size() for b in self.buffers.values())


class _CachedLaunch(Protocol):
    """What the arena bookkeeping needs of a cached convenience-path entry (K1 :class:`_CachedGemm`, K2
    :class:`_CachedRouter`): its arena, the memoised workspace views with the arena version they were carved
    at, and the carve itself."""

    arena: _Arena
    views: Optional[dict[str, torch.Tensor]]
    views_version: int

    def carve(self, arena: _Arena) -> dict[str, torch.Tensor]: ...


_ARENAS: dict[tuple[int, int], _Arena] = {}
# Serialises planning and every mutation of the arenas and the caches (``_plan_entry``, ``cake_dense_gemm_det_train_convenience_cache_clear``);
# the hit paths (``_cache_get``, ``_entry_views``) do not take it.
_PLAN_LOCK = threading.Lock()


def _arena_for(index: int, stream: int, device: torch.device) -> _Arena:
    """The arena of (device ``index``, raw ``stream``), created on first use (under ``_PLAN_LOCK``)."""
    key = (int(index), int(stream))
    arena = _ARENAS.get(key)
    if arena is None:
        arena = _ARENAS[key] = _Arena(device, key)
    return arena


def _release_entry(entry: _CachedLaunch) -> None:
    """``on_evict`` of the convenience caches: the arena of an evicted entry loses a reference and is released
    (its buffers freed, stream-ordered) when no cached entry uses it any more."""
    arena = entry.arena
    arena.refs -= 1
    if arena.refs <= 0 and _ARENAS.get(arena.key) is arena:
        del _ARENAS[arena.key]


def _plan_entry(
    cache: "OrderedDict[tuple, Any]",
    key: tuple,
    index: int,
    stream: int,
    device: torch.device,
    prepare: Callable[[_Arena], Any],
) -> Any:
    """Plan the entry of ``key`` with ``prepare(arena)`` -- which returns ``(entry, views)``: the entry and the
    per-call operand views its fail-closed check already built for this first call -- against the arena of
    (``index``, ``stream``), cache the entry (``_release_entry`` on eviction) and return both; all of it under
    ``_PLAN_LOCK``.  The cache is re-checked under the lock: when another thread planned ``key`` meanwhile its
    entry is returned with ``views = None`` (the caller rebuilds the views).  A failed plan leaves no arena
    behind that nothing references, and a shared arena it had grown is rolled back to its earlier buffers."""
    with _PLAN_LOCK:
        entry = _cache_get(cache, key)
        if entry is not None:
            return entry, None
        arena = _arena_for(index, stream, device)
        buffers, version = dict(arena.buffers), arena.version
        try:
            entry, views = prepare(arena)
        except BaseException:
            if arena.refs <= 0 and _ARENAS.get(arena.key) is arena:
                del _ARENAS[arena.key]
            elif arena.version != version:
                arena.buffers = buffers
                arena.version += 1
                _forget_arena_views(arena)
            raise
        _cache_put(cache, key, entry, on_evict=_release_entry)
        return entry, views


def _forget_arena_views(arena: _Arena) -> None:
    """Drop the memoised workspace views of every cached entry and the converted workspace of every binding
    (:class:`_BoundGemm`) of ``arena`` (a buffer was replaced)."""
    entries: list[_CachedLaunch] = [*_K1_CACHE.values(), *_K2_CACHE.values()]
    for entry in entries:
        if entry.arena is arena:
            entry.views = None
    for bound in [*_K1_BOUND_CACHE.values()]:
        if type(bound) is _BoundGemm and bound.arena is arena:
            bound.state = (-1, None)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _carve(buf: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """The contiguous ``shape`` view over the first elements of the 1-D buffer ``buf`` (one strided view, no
    slice-then-reshape pair)."""
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return buf.as_strided(shape, tuple(reversed(strides)))


def _entry_views(entry: _CachedLaunch) -> dict[str, torch.Tensor]:
    """The workspace views of a cached entry, carved out of its arena's buffers by ``entry.carve(arena)``
    (K1: the slab and counter buffers themselves -- pointer parameters; K2: the ``[3, *split]`` parts, the
    ``split`` scratch of the torch-chain split and the ``[splits, rows, cols]`` partials) and memoised until the
    arena grows."""
    arena = entry.arena
    views = entry.views
    if views is None or entry.views_version != arena.version:
        version = (
            arena.version
        )  # read before carving: a growth that interleaves leaves the memo stale
        views = entry.carve(arena)
        entry.views_version = version
        entry.views = views
    return views


# ---------------------------------------------------------------------------
# K1 prepared launch
# ---------------------------------------------------------------------------


@dataclass
class CakeDenseGemmDetProjectionLaunch:
    """One prepared dense projection GEMM launch.

    ``launch()`` writes ``out`` and returns it; it allocates nothing and never synchronizes.
    The object owns its stream-K workspace and its slice counters; prepare a new one when a shape, dtype, stride or tensor binding changes
    (values may change freely).
    """

    module_name: str
    arch: str
    plan: CakeDenseGemmDetProjectionPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_Launch] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        with _ffi_stream_context(self.device_index):
            self._launch()
        return self.out

    __call__ = launch


def _k1_plan(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    knobs: dict[str, Any],
) -> tuple[
    CakeDenseGemmDetProjectionPlan, torch.Tensor, torch.Tensor, torch.Tensor, str, str
]:
    """Plan one K1 binding for its device: ``(plan, a_desc, b_desc, out3, arch, module_name)`` (the device facts
    come from the per-device cache, the module from the registry; see :func:`cake_dense_gemm_det_train_projection_plan`)."""
    device = _same_cuda_device(A, B, out)
    facts = device_facts(device)
    arch = require_arch(device)
    plan, a_desc, b_desc, O3 = cake_dense_gemm_det_train_projection_plan(
        A,
        B,
        out,
        sm_count=facts.sm_count,
        l2_bytes=facts.l2_bytes,
        transposed_out=transposed_out,
        arch=arch,
        **knobs,
    )
    return plan, a_desc, b_desc, O3, arch, select_module(arch, plan.template)


def _k1_workspace(
    plan: CakeDenseGemmDetProjectionPlan, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stream-K partial slabs and the slice counters of a plan (the only allocations of the K1 backend).
    The counters are zero-initialised once: the kernel resets every counter it uses (the last arriver of a tail
    tile's slice writes zero after the fixup), so a buffer is zero again when its launch completes and is reused
    across launches without a memset, as the Cake launcher's per-device counters are.  Sized like the Cake host's
    ``_sk_counters`` so the lane-spread dummy counters above SK_DUMMY_BASE are addressable."""
    if plan.ws_f32_elems:
        ws = torch.empty(plan.ws_f32_elems, device=device, dtype=torch.float32)
    else:
        ws = _dummy(device, torch.float32)
    counters = torch.zeros(plan.counters_alloc_u32, device=device, dtype=torch.uint32)
    return ws, counters


def _out_strides(out3: torch.Tensor) -> tuple[int, int]:
    """``(ldo, out_l)`` of the batched output view: its row stride and its batch stride.  The kernel declares its
    tile grid against the output's extents, so it requires ``out_l >= rows * ldo`` for every batch size (rows = M, or
    N under the transposed store).  A batch of one never dereferences its batch stride: its ``out_l`` is the view's
    batch stride when that spans the matrix (a 2-D output unsqueezed to ``[1, rows, cols]``) and the matrix span
    ``rows * ldo`` otherwise (an expanded or ``as_strided`` batch axis).  A real batch passes its stride as is."""
    ldo = int(out3.stride(1))
    batch = int(out3.stride(0))
    span = int(out3.shape[1]) * ldo
    if int(out3.shape[0]) == 1 and batch < span:
        batch = span
    return ldo, batch


def _k1_values(
    plan: CakeDenseGemmDetProjectionPlan,
    device: torch.device,
    *,
    A: Any,
    B: Any,
    out3: Any,
    o_alias: Any,
    ws: Any,
    counters: Any,
    ldo: int,
    out_l: int,
    enable_pdl: Any,
) -> dict[str, Any]:
    """The keyword arguments of a K1 launch (:func:`bind_launch` orders them by the module's plan).  ``A`` /
    ``B`` / ``out3`` / ``o_alias`` are the operand views and ``ws`` / ``counters`` the workspace -- or
    :class:`_Slot` markers for a cached launch that rebinds the views on every call and resolves the workspace
    out of its arena; the plan decides which of the operands the instance dereferences.  ``enable_pdl`` is the
    launch's PDL flag (0 or 1), or a :class:`_Slot` a cached launch fills per call."""
    if isinstance(plan, ChainGemmPlan):
        return _chain_values(
            plan,
            A=A,
            B=B,
            o_alias=o_alias,
            ws=ws,
            counters=counters,
            ldo=ldo,
            out_l=out_l,
            enable_pdl=enable_pdl,
        )
    dummy16, dummy32 = _dummy(device, torch.bfloat16), _dummy(device, torch.float32)
    tma_out, out_f32 = plan.tma_out, plan.out_f32
    return dict(
        A=A,
        B=B,
        OUT32=out3 if (tma_out and out_f32) else _dummy_map(device, torch.float32),
        OUT16=out3 if (tma_out and not out_f32) else _dummy_map(device, torch.bfloat16),
        # round 27: the stt program keeps the register-store alias next to the OUT16 descriptor  [Cake launcher]
        out=o_alias if (not out_f32 and (not tma_out or plan.stt)) else dummy16,
        out32=o_alias if (out_f32 and not tma_out) else dummy32,
        ws=ws,
        counters=counters,
        M=plan.M,
        N=plan.N,
        m_tiles=plan.m_tiles,
        n_tiles=plan.n_tiles,
        group_m=plan.group_m,
        promo_code=L2_PROMOS.index(plan.promo),
        k_iters=plan.k_blocks,
        ldo=ldo,
        out_l=out_l,
        num_cluster_tiles=plan.num_cluster_tiles,
        num_l=plan.L,
        num_full=plan.num_full,
        iters_per_unit=plan.iters_per_unit,
        sk_iters=plan.sk_iters,
        # the eight-warp programs' raster fast divisors: tiles per batch entry, tiles per raster group, the group's row
        # tiles and the last group's (the ten-warp programs' argument plans do not name them)
        fd_tiles_l=plan.m_tiles * plan.n_tiles,
        fd_tiles_g=plan.group_m * plan.n_tiles,
        fd_group=plan.group_m,
        fd_last=plan.m_tiles % plan.group_m or plan.group_m,
        enable_pdl=enable_pdl,
    )


def cake_dense_gemm_det_train_projection_prepare(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    sk: bool | str = "auto",
    block_n: Optional[int] = None,
    stages: Optional[int] = None,
    epi: Optional[str] = None,
    slots: Optional[int] = None,
    cta_rows: Optional[int] = None,
    pf: Optional[int] = None,
    sk_max_units: Optional[int] = None,
    sk_parts: Optional[int] = None,
    promo: Optional[str] = None,
    hints: Optional[tuple] = None,
    group_m: Optional[int] = None,
    f32_v8: Optional[bool] = None,
    quad_store: Optional[bool] = None,
    park: Optional[bool] = None,
    ovl: Optional[bool] = None,
    htail: Optional[bool] = None,
    b_swz: Optional[int] = None,
    sk_exact: Optional[int] = None,
    batch_group: Optional[int] = None,
    store_ef: Optional[bool] = None,
    store_hint: Optional[str] = None,
    pd: Optional[int] = None,
    sh: Optional[int] = None,
    sk_sync: Optional[bool] = None,
    sk_sync_m: Optional[int] = None,
    sk_slab: Optional[int] = None,
    cta1: Optional[bool] = None,
    cgrp: Optional[bool] = None,
    ovr: Optional[bool] = None,
    stt: Optional[bool] = None,
    pro: Optional[int] = None,
    form: Optional[str] = None,
    promote: Optional[bool] = None,
    enable_pdl: Optional[bool] = None,
) -> CakeDenseGemmDetProjectionLaunch:
    """Validate one binding, plan it for the device and prepare its launch (the only
    allocations of the K1 backend: the stream-K partial slabs and the slice counters).  See the module docstring for the view contract; the keyword
    knobs default to the Cake launcher's choices (``promote``: the planner's promoted chunks).  ``enable_pdl`` (default: the device supports it) makes every
    ``launch()`` a programmatic dependent launch: the program waits for the preceding kernel before its first global
    access and lets the next one launch at its entry."""
    knobs: dict[str, Any] = dict(
        sk=sk,
        block_n=block_n,
        stages=stages,
        epi=epi,
        slots=slots,
        cta_rows=cta_rows,
        pf=pf,
        sk_max_units=sk_max_units,
        sk_parts=sk_parts,
        promo=promo,
        hints=hints,
        group_m=group_m,
        f32_v8=f32_v8,
        quad_store=quad_store,
        park=park,
        ovl=ovl,
        htail=htail,
        b_swz=b_swz,
        sk_exact=sk_exact,
        batch_group=batch_group,
        store_ef=store_ef,
        store_hint=store_hint,
        pd=pd,
        sh=sh,
        sk_sync=sk_sync,
        sk_sync_m=sk_sync_m,
        sk_slab=sk_slab,
        cta1=cta1,
        cgrp=cgrp,
        ovr=ovr,
        stt=stt,
        pro=pro,
        form=form,
        promote=promote,
    )
    plan, a_desc, b_desc, O3, arch, module_name = _k1_plan(
        A, B, out, transposed_out=transposed_out, knobs=knobs
    )
    device = out.device
    ws, counters = _k1_workspace(plan, device)
    o_alias = flat_alias(O3)
    ldo, out_l = _out_strides(O3)
    tensors = dict(
        A=a_desc, B=b_desc, out3=O3, ws=ws, counters=counters, o_alias=o_alias
    )
    values = _k1_values(
        plan,
        device,
        A=a_desc,
        B=b_desc,
        out3=O3,
        o_alias=o_alias,
        ws=ws,
        counters=counters,
        ldo=ldo,
        out_l=out_l,
        enable_pdl=int(pdl_enabled(enable_pdl, device)),
    )
    launch = bind_launch(module_name, values, plan.grid, arch)
    return CakeDenseGemmDetProjectionLaunch(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=tensors,
        device_index=_device_index(device),
        _launch=launch,
    )


# ---------------------------------------------------------------------------
# K1 convenience path (cached planned launches)
# ---------------------------------------------------------------------------


def _k1_swapped(
    plan: CakeDenseGemmDetProjectionPlan,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    transposed_out: bool,
) -> bool:
    """Did the planner run the swapped GEMM (``B.T @ A.T`` with the transposed store) for these views?  A chain plan
    records it (``plan.swapped``).  Otherwise a swap (the launcher's batched small-M rule, :func:`swap_small_m`, or the
    unbatched weight-gradient rule, :func:`_single_wgrad_swap`) turns a call without the transposed store into a plan
    with it; the registry fallback that undid a swap (``plan.swap_fallback``) planned the caller's orientation."""
    if isinstance(plan, ChainGemmPlan):
        return plan.swapped
    return bool(plan.transposed_out) and not transposed_out


def _k1_views(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, swapped: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The operand views of a K1 launch for the caller's tensors -- ``(a_desc, b_desc, out3)`` exactly as
    :func:`cake_dense_gemm_det_train_projection_plan` derives them (batched, the operands exchanged and transposed when the plan
    swapped, the ``[L, outer, inner]`` TMA spans of :func:`operand_view`) -- without planning.  ``operand_view``
    re-checks the stride and alignment contract of the views."""
    A3, B3, O3 = as_batched(A, "A"), as_batched(B, "B"), as_batched(out, "out")
    if swapped:
        A3, B3 = B3.transpose(1, 2), A3.transpose(1, 2)
    _, a_desc = operand_view(A3, "A", k_axis=2)
    _, b_desc = operand_view(B3, "B", k_axis=1)
    return a_desc, b_desc, O3


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


@dataclass
class _CachedGemm:
    """A planned K1 launch of the convenience path: the plan, the resolved module entry, the argument template
    (constants and :class:`_Slot` markers in the module's plan order) and the arena its workspace is carved
    from.  The plan holds the workspace requirement (``ws_f32_elems`` slabs, ``counters_alloc_u32`` counters)
    the arena was grown to when the entry was planned."""

    module_name: str
    arch: str
    plan: CakeDenseGemmDetProjectionPlan
    swapped: bool
    entry: Callable[..., Any] = field(repr=False)
    template: tuple = field(repr=False)
    arena: _Arena = field(repr=False)
    views: Optional[dict[str, torch.Tensor]] = field(default=None, repr=False)
    views_version: int = -1

    def carve(self, arena: _Arena) -> dict[str, torch.Tensor]:
        """``ws`` and ``counters`` are pointer parameters (the binding checks dtype, device and contiguity
        only): the arena buffers themselves, no slicing."""
        return dict(ws=arena.buffers["k1_ws"], counters=arena.buffers["k1_counters"])


_K1_CACHE: "OrderedDict[tuple, _CachedGemm]" = OrderedDict()


def _prepare_cached_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    knobs: dict[str, Any],
    arena: _Arena,
) -> tuple[_CachedGemm, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(entry, (a_desc, b_desc, out3))``: the planned entry and the per-call views of this first call (the
    rebuild the fail-closed check validated against the planner's views)."""
    plan, a_desc, b_desc, O3, arch, module_name = _k1_plan(
        A, B, out, transposed_out=transposed_out, knobs=knobs
    )
    device = out.device
    swapped = _k1_swapped(plan, A, B, transposed_out=transposed_out)
    rebuilt = _k1_views(A, B, out, swapped=swapped)
    if not all(
        _same_view(got, want)
        for got, want in zip(rebuilt, (a_desc, b_desc, O3), strict=True)
    ):
        raise RuntimeError(
            "cake_dense_gemm_det_train_projection: the per-call operand views differ from the planner's; "
            "the convenience path cannot cache this binding (use cake_dense_gemm_det_train_projection_prepare)"
        )
    ldo, out_l = _out_strides(O3)
    values = _k1_values(
        plan,
        device,
        A=_Slot("A"),
        B=_Slot("B"),
        out3=_Slot("out3"),
        o_alias=_Slot("o_alias"),
        ws=_Slot("ws"),
        counters=_Slot("counters"),
        ldo=ldo,
        out_l=out_l,
        enable_pdl=_Slot("enable_pdl"),
    )
    entry, template = _bind(module_name, values, plan.grid, arch)
    # the workspace is a slot: grow the arena only once the module and the grid are known to be good
    arena.require("k1_ws", plan.ws_f32_elems)
    arena.require("k1_counters", plan.counters_alloc_u32)
    arena.refs += 1
    return _CachedGemm(
        module_name, arch, plan, swapped, entry, template, arena
    ), rebuilt


def _cached_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    transposed_out: bool,
    knobs: dict[str, Any],
    index: int,
    stream: int,
    device: torch.device,
) -> tuple[_CachedGemm, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(entry, (a_desc, b_desc, out3))``: the cached planned launch of the K1 convenience call ``(A, B, out,
    transposed_out, **knobs)`` on the raw ``stream`` of device ``index`` -- planned on the first call of its key --
    and the operand views of this call."""
    key = convenience_key(
        index, stream, (bool(transposed_out), *sorted(knobs.items())), A, B, out
    )
    cached = _cache_get(_K1_CACHE, key)
    operands = None
    if cached is None:
        cached, operands = _plan_entry(
            _K1_CACHE,
            key,
            index,
            stream,
            device,
            lambda arena: _prepare_cached_gemm(
                A, B, out, transposed_out=transposed_out, knobs=knobs, arena=arena
            ),
        )
    if operands is None:
        operands = _k1_views(A, B, out, swapped=cached.swapped)
    return cached, operands


def _launch_cached_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    enable_pdl: Optional[bool] = None,
    **knobs: Any,
) -> torch.Tensor:
    """The convenience path of K1: plan on the first call for a (device, stream, geometry, knobs) key; rebind
    the caller's views, resolve the workspace out of the arena of the device and stream and launch on every
    call (no planning, no device query, no allocation), as a programmatic dependent launch under ``enable_pdl``
    (default: the device supports it)."""
    device = _same_cuda_device(A, B, out)
    index = _cuda_index(device)
    stream = _current_raw_stream(index)
    cached, (a_desc, b_desc, O3) = _cached_gemm(
        A, B, out, transposed_out, knobs, index, stream, device
    )
    views = dict(
        A=a_desc,
        B=b_desc,
        out3=O3,
        o_alias=flat_alias(O3),
        enable_pdl=int(pdl_enabled(enable_pdl, index)),
    )
    views.update(_entry_views(cached))
    arguments = _resolve_arguments(cached.template, views)
    with _ffi_stream_context(index, stream):
        cached.entry(*arguments)
    return out


# --- the swapped small-token forward -------------------------------------------------------
#
# A batched forward row the planner would swap (:func:`swap_small_m`: few tokens against many weight rows) runs on a
# program that puts the weight rows in the MMA's M and the tokens in its N (``swap_ab_<tile>`` templates): the
# weights stream through a deep TMA ring, at most two waves of single-CTA tiles, and every output element is the one FP32 sum
# of its whole K, rounded once (no split, so the bits do not depend on the tile).  The program reads ``A`` and the
# K-major ``B`` through tensor maps over the caller's views and stores into ``out`` at
# ``l * out_l + m * ldo + n``, so token-major views run without a copy.  Under ``enable_pdl`` the default programs
# wait for the predecessor grid before any operand load; with ``weights_static=True`` (the caller states that no
# kernel still in flight on the stream writes ``B``) the ``_we`` programs run instead, whose weight loads start ahead
# of the wait (only the token loads wait), so the weight stream overlaps the predecessor's tail.
SWAP_TILE_T = 64  # tokens per CTA of the swapped programs
#: Weight rows per CTA -> the template serving them, in preference order: the narrow tile while its CTAs fit one
#: wave (more CTAs streaming the weights), the wide one otherwise.
SWAP_TEMPLATES: Tuple[Tuple[int, str], ...] = (
    (64, "swap_ab_w64_t64_s12"),
    (128, "swap_ab_w128_t64_s8"),
)
#: The same tiles whose weight loads run ahead of the PDL wait (``weights_static=True``).
SWAP_TEMPLATES_WEIGHTS_STATIC: Tuple[Tuple[int, str], ...] = tuple(
    (tile_w, f"{template}_we") for tile_w, template in SWAP_TEMPLATES
)
#: The most waves of single-CTA tiles a swapped forward runs in.  Its CTAs each stream their weight rows once; past
#: two waves the planner's pair tiles stream them faster (GB300: 8 x 4096 x 1024 at 64 tokens, 256 CTAs, 17.0 us
#: against 20.5 us; at 128 tokens, 512 CTAs, 25.1 us against 21.7 us).
SWAP_MAX_WAVES = 2


@dataclass(frozen=True)
class _SwapLaunch:
    """A routed swapped forward: the module entry, its grid and the scalars of its argument plan."""

    module_name: str
    entry: Callable[..., Any] = field(repr=False)
    grid: Tuple[int, int, int]
    M: int
    N: int
    K: int
    L: int
    ldo: int
    out_l: int


def _aligned_view(t: torch.Tensor) -> bool:
    """A ``[L, rows, K]`` K-major view the swapped program's tensor maps take: unit K stride, row and batch strides
    in whole 16-byte units, a 16-byte aligned start."""
    return (
        t.stride(2) == 1
        and t.stride(1) % 8 == 0
        and t.stride(0) % 8 == 0
        and t.data_ptr() % 16 == 0
    )


def swap_ab_template(
    L: int, M: int, N: int, sm_count: int, arch: str, weights_static: bool = False
) -> Optional[str]:
    """The ``swap_ab_<tile>`` template a swapped forward of ``L`` batches of ``M`` tokens against ``N`` weight rows
    runs on ``arch``: the first of :data:`SWAP_TEMPLATES` (:data:`SWAP_TEMPLATES_WEIGHTS_STATIC` with
    ``weights_static``) whose CTAs fit one wave of the device's SMs, else the widest; ``None`` when that one needs more than :data:`SWAP_MAX_WAVES` waves (the planner's program is faster
    there) or ``arch`` registers none of them."""
    table = KERNELS.get(arch, {})
    templates = SWAP_TEMPLATES_WEIGHTS_STATIC if weights_static else SWAP_TEMPLATES
    registered = [(w, t) for w, t in templates if t in table]
    if not registered:
        return None
    m_tiles = -(-M // SWAP_TILE_T)
    for tile_w, template in registered:
        if -(-N // tile_w) * m_tiles * L <= sm_count:
            return template
    tile_w, template = registered[-1]
    if -(-N // tile_w) * m_tiles * L > SWAP_MAX_WAVES * sm_count:
        return None
    return template


def _swap_ab_launch(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    transposed_out: bool,
    weights_static: bool = False,
) -> Optional[_SwapLaunch]:
    """The swapped launch of a K1 call, or ``None`` when the call stays on the planner: a batched forward
    (``A`` K-major, ``B`` K-major, BF16 out, no transposed store) that :func:`swap_small_m` selects, with ``K`` a
    multiple of 64, the views' strides and starts in 16-byte units and a registered swapped program."""
    if (
        transposed_out
        or out.dtype != torch.bfloat16
        or A.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or A.dim() != 3
        or B.dim() != 3
        or out.dim() != 3
    ):
        return None
    L, M, K = (int(s) for s in A.shape)
    N = int(B.shape[2])
    if (
        tuple(B.shape) != (L, K, N)
        or tuple(out.shape) != (L, M, N)
        or not swap_small_m(L, M, N, transposed_out)
        or K % BLOCK_K != 0
        or N % 8 != 0
    ):
        return None
    W = B.transpose(1, 2)
    if not (_aligned_view(A) and _aligned_view(W)):
        return None
    ldo = int(out.stride(1)) if M > 1 else N
    out_l = int(out.stride(0))
    if (
        out.stride(2) != 1
        or ldo % 8 != 0
        or ldo < N
        or out_l % 8 != 0
        or out.data_ptr() % 16 != 0
        or max(ldo, out_l) > 2**31 - 1
    ):
        return None
    device = _same_cuda_device(A, B, out)
    facts = device_facts(device)
    if facts.arch is None:
        return None
    template = swap_ab_template(L, M, N, facts.sm_count, facts.arch, weights_static)
    if template is None:
        return None
    module_name = select_module(facts.arch, template)
    tile_w = dict((t, w) for w, t in SWAP_TEMPLATES + SWAP_TEMPLATES_WEIGHTS_STATIC)[
        template
    ]
    grid = (-(-N // tile_w), -(-M // SWAP_TILE_T), L)
    module = load_cake_dense_projection_det_gemm_module(module_name, facts.arch)
    entry = getattr(module, MODULES[module_name]["ffi_entry"])
    return _SwapLaunch(module_name, entry, grid, M, N, K, L, ldo, out_l)


_SWAP_CACHE: "OrderedDict[tuple, Union[_SwapLaunch, bool]]" = OrderedDict()


def _swap_ab_cached(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    transposed_out: bool,
    weights_static: bool = False,
) -> Optional[_SwapLaunch]:
    """:func:`_swap_ab_launch` cached by the device, ``weights_static`` and the geometry of the views (a routed
    launch, or ``False`` for a row that stays on the planner) in a least-recently-used cache of
    ``CONVENIENCE_CACHE_SIZE`` entries."""
    if not (A.is_cuda and B.is_cuda and out.is_cuda):
        return None
    key = convenience_key(
        _cuda_index(out.device),
        0,
        (bool(transposed_out), bool(weights_static)),
        A,
        B,
        out,
    )
    hit = _cache_get(_SWAP_CACHE, key)
    if hit is None:
        launch = _swap_ab_launch(A, B, out, transposed_out, weights_static)
        with _PLAN_LOCK:
            hit = _cache_put(_SWAP_CACHE, key, launch if launch is not None else False)
    return hit or None


def _run_swap_ab(
    launch: _SwapLaunch,
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    index = _cuda_index(out.device)
    with _ffi_stream_context(index):
        launch.entry(
            B.transpose(1, 2),
            A,
            flat_alias(out).view(torch.uint8),
            launch.M,
            launch.N,
            launch.K,
            launch.L,
            launch.ldo,
            launch.out_l,
            int(bool(enable_pdl)),
            *launch.grid,
        )
    return out


def cake_dense_gemm_det_train_projection(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    enable_pdl: Optional[bool] = None,
    weights_static: bool = False,
) -> torch.Tensor:
    """``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` for strided views (the cached convenience path:
    planned once per device, stream and view geometry, then rebound and launched).

    A batched forward row of few tokens (:func:`swap_small_m`) runs the swapped program (:func:`swap_ab_template`),
    every other row the planner's program.  ``enable_pdl`` (default: the device supports it) launches either as a
    programmatic dependent launch: every CTA lets the next kernel launch at its entry and waits for the preceding
    kernel before its first global access.  ``weights_static=True`` states that no kernel still in flight on the
    stream writes ``B`` (a weight of the model between updates): the swapped program then starts its weight loads
    before the wait, so the weight stream overlaps the preceding kernel's tail, and only its token loads wait.  The
    default never reads ``B`` before the preceding kernel completes."""
    if _k1_entry(
        ("gemm", bool(transposed_out), bool(weights_static)),
        (A, B, out),
        enable_pdl,
        lambda: (A, B, out, transposed_out, {}, bool(weights_static)),
    ):
        return out
    pdl = pdl_enabled(enable_pdl, out.device)
    swap = _swap_ab_cached(A, B, out, transposed_out, bool(weights_static) and pdl)
    if swap is not None:
        return _run_swap_ab(swap, A, B, out, pdl)
    return _launch_cached_gemm(
        A, B, out, transposed_out=transposed_out, enable_pdl=enable_pdl
    )


# --- the training operations as thin wrappers (the Cake entry points) ---------------------


def cake_dense_gemm_det_train_projection_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` (bf16 out by default); ``enable_pdl`` as for
    :func:`cake_dense_gemm_det_train_projection`."""
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.bfloat16)
    if _k1_entry(
        _FWD_HEAD, (X, W, out), enable_pdl, lambda: (X, W.t(), out, False, {}, False)
    ):
        return out
    return cake_dense_gemm_det_train_projection(X, W.t(), out, enable_pdl=enable_pdl)


def cake_dense_gemm_det_train_projection_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]``; ``enable_pdl`` as for :func:`cake_dense_gemm_det_train_projection`."""
    if out is None:
        out = torch.empty(
            G.shape[0], W.shape[1], device=G.device, dtype=out_dtype or torch.bfloat16
        )
    if _k1_entry(
        _DGRAD_HEAD, (G, W, out), enable_pdl, lambda: (G, W, out, False, {}, False)
    ):
        return out
    return cake_dense_gemm_det_train_projection(G, W, out, enable_pdl=enable_pdl)


def wgrad_views(
    G: torch.Tensor,
    X: torch.Tensor,
    *,
    arch: Optional[str] = None,
    out_f32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """``(A, B, transposed_out)`` of the weight gradient ``G[T, N].T @ X[T, K] -> dW[N, K]``:
    when ``wgrad_swapped`` the swapped GEMM ``X.T @ G`` runs with the transposed store (the
    result lands directly in ``[N, K]``), otherwise ``G.T @ X``.  ``arch`` / ``out_f32`` let a
    measured rule of that architecture and output dtype decide the orientation (see ``wgrad_swapped``).
    [Cake K1 kernel host weight-gradient entry L1507-L1518 / ``wgrad_swapped`` L1496-L1504]"""
    N, K = int(G.shape[1]), int(X.shape[1])
    if wgrad_swapped(N, K, arch=arch, out_f32=out_f32):
        return X.t(), G, True
    return G.t(), X, False


def _orientation_rules(arch: str, N: int, K: int, out_f32: bool) -> tuple[bool, bool]:
    """Whether ``ROW_RULES`` keys the weight gradient ``[N, K]`` of ``arch`` in the swapped and in the direct
    orientation; an architecture keyed in neither reads its parent's keys (``RULE_PARENT``)."""
    a: Optional[str] = arch
    while a is not None:
        swapped = (a, True, True, bool(out_f32), True, False, N, None, K) in ROW_RULES
        direct = (a, True, True, bool(out_f32), False, False, K, None, N) in ROW_RULES
        if swapped or direct:
            return swapped, direct
        a = RULE_PARENT.get(a)
    return False, False


def wgrad_swapped(
    N: int, K: int, *, arch: Optional[str] = None, out_f32: bool = False
) -> bool:
    """Orientation of the weight gradient.  A measured rule decides first: the row's ``ROW_RULES`` key
    names the orientation it was measured in - the swapped key ``(arch, nn, out_f32, True, False, N, None, K)``
    (the swapped GEMM ``X.T @ G`` has M = K and N = N with the transposed store) or the direct key
    ``(arch, nn, out_f32, False, False, K, None, N)`` (``G.T @ X``: M = N, N = K) - and a row keyed in exactly one
    orientation takes it (the 4096 -> 512 rows swap on sm_100a / sm_103a, the 4096 -> 1024 bf16 row swaps on
    sm_100a; the launcher's rule for the small-N rows below agrees).  Without a rule, or without ``arch``, the
    padded-tile-work heuristic of the Cake launcher: swapped when 256-row tiles over K and column tiles padded
    from N move less than 256-row tiles padded from N rows and ``default_block_n`` column tiles over K
    (N < 256 and N = 576 swap; a tie keeps the direct route).  [Cake ``wgrad_swapped`` L1496-L1504]"""
    if arch is not None:
        # the weight-gradient rules are keyed ragged in K (= T): the exact orientation keys, not ``_measured_row_rule``;
        # an architecture without a rule in either orientation reads its parent's keys (``RULE_PARENT``)
        swapped_rule, direct_rule = _orientation_rules(arch, N, K, out_f32)
        if swapped_rule != direct_rule:
            return swapped_rule
    bk, bn = default_block_n(K, True), default_block_n(N, True)
    direct = _ceil_div(N, 256) * 256 * _ceil_div(K, bk) * bk
    swapped = _ceil_div(K, 256) * 256 * _ceil_div(N, bn) * bn
    return swapped < direct


def cake_dense_gemm_det_train_projection_wgrad_prepare(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    *,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> CakeDenseGemmDetProjectionLaunch:
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]``; the orientation follows the row's measured rule, else the
    small-``N`` swap rule of the Cake launcher (``wgrad_swapped``); ``enable_pdl`` as for
    :func:`cake_dense_gemm_det_train_projection_prepare`."""
    A, B, transposed = wgrad_views(
        G, X, arch=arch_for(out.device), out_f32=out.dtype == torch.float32
    )
    return cake_dense_gemm_det_train_projection_prepare(
        A,
        B,
        out,
        transposed_out=transposed,
        sk=sk,
        cta_rows=cta_rows,
        enable_pdl=enable_pdl,
    )


def cake_dense_gemm_det_train_projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` (ragged reduction length ``T``); see :func:`wgrad_views`;
    ``enable_pdl`` as for :func:`cake_dense_gemm_det_train_projection`."""
    N, K = int(G.shape[1]), int(X.shape[1])
    if out is None:
        out = torch.empty(N, K, device=G.device, dtype=out_dtype or torch.bfloat16)

    o = out

    def call() -> tuple:
        A, B, transposed = wgrad_views(
            G, X, arch=arch_for(o.device), out_f32=o.dtype == torch.float32
        )
        return A, B, o, transposed, dict(sk=sk, cta_rows=cta_rows), None

    if _k1_entry(("wgrad", sk, cta_rows), (G, X, o), enable_pdl, call):
        return o
    A, B, transposed = wgrad_views(
        G, X, arch=arch_for(out.device), out_f32=out.dtype == torch.float32
    )
    return _launch_cached_gemm(
        A,
        B,
        out,
        transposed_out=transposed,
        enable_pdl=enable_pdl,
        sk=sk,
        cta_rows=cta_rows,
    )


# ---------------------------------------------------------------------------
# K1 entry points: the bound launch of a key
# ---------------------------------------------------------------------------

# The K1 entry points (``cake_dense_gemm_det_train_projection``, ``cake_dense_gemm_det_train_projection_forward`` / ``cake_dense_gemm_det_train_projection_dgrad`` /
# ``cake_dense_gemm_det_train_projection_wgrad``) bind their launch once per key of their own tensors: the device, the raw stream, the entry
# point with its flags and knobs, the PDL flag and the geometry of the entry point's tensors (as ``convenience_key``).
# The binding (:class:`_BoundGemm`) holds the cached planned launch of the call (the ``_K1_CACHE`` entry: the same plan
# and program, so the same bits), its argument list in the program's order with the PDL flag bound and the constant
# tensors converted once, the arena's workspace converted once per arena growth, and how each per-call view is rebuilt
# from the entry point's tensors: the tensor itself, its ``unsqueeze(0)`` or one strided view, each checked against the
# planner's view when bound.  A call whose key is bound rebuilds only those views and launches in its thread's stream
# context of the (device, stream), made once: no planning, no view derivation, no orientation lookup, no allocation.
# A row on the swapped small-token forward keeps its routed launch (``_SwapLaunch``) under the key of
# ``cake_dense_gemm_det_train_projection``.  A key whose views the entry point's tensors do not rebuild (tensors that start at one
# data pointer; the key does not record aliasing) is kept unbound and takes the convenience path above, as does a call
# whose tensors are not on one CUDA device.  A binding holds a reference on its arena like a cached entry (released on
# eviction); the bindings live in a least-recently-used cache of ``CONVENIENCE_CACHE_SIZE`` keys that
# ``cake_dense_gemm_det_train_convenience_cache_clear`` drops with the stream contexts.

#: How a per-call view of a bound launch is rebuilt from one of the entry point's tensors (:class:`_BoundGemm`): the
#: tensor itself, its ``unsqueeze(0)``, or its ``as_strided`` view of the bound shape and strides.
_VIEW_SELF, _VIEW_UNSQUEEZE, _VIEW_STRIDED = 0, 1, 2
#: The arena kinds of the workspace slots of a K1 launch.
_K1_WORKSPACE_KINDS = {"ws": "k1_ws", "counters": "k1_counters"}
#: The leading key flags of the forward and the input gradient.
_FWD_HEAD = ("fwd",)
_DGRAD_HEAD = ("dgrad",)


@dataclass
class _BoundGemm:
    """The bound launch of a K1 entry point's key: the cached planned launch it runs (``gemm``), its argument list in
    the program's order with the constants converted and the PDL flag bound (``template``), the positions of the
    arena's workspace buffers (``workspace``: ``(position, arena kind)``), the per-call views (``recipes``:
    ``(position, source, recipe, shape, strides)``, rebuilt from the entry point's tensor ``source``), and ``state``:
    the argument list with the workspace converted and the arena version it was converted at."""

    gemm: _CachedGemm
    template: list = field(repr=False)
    recipes: tuple = field(repr=False)
    workspace: tuple = field(repr=False)
    state: tuple = field(default=(-1, None), repr=False)

    @property
    def arena(self) -> _Arena:
        return self.gemm.arena


_K1_BOUND_CACHE: "OrderedDict[tuple, Union[_BoundGemm, _SwapLaunch, bool]]" = (
    OrderedDict()
)
#: The tvm-ffi stream context of every (thread, device index, raw stream) a bound launch ran on, made by the first
#: launch: a context keeps the stream it replaced while it is entered, so a thread enters only its own.
_STREAM_CONTEXTS: dict[tuple[int, int, int], Any] = {}


def _release_bound(entry: Union[_BoundGemm, _SwapLaunch, bool]) -> None:
    """``on_evict`` of ``_K1_BOUND_CACHE``: a binding releases its reference on the arena (as :func:`_release_entry`)."""
    if type(entry) is _BoundGemm:
        arena = entry.arena
        arena.refs -= 1
        if arena.refs <= 0 and _ARENAS.get(arena.key) is arena:
            del _ARENAS[arena.key]


def _stream_context(index: int, stream: int) -> Any:
    """The calling thread's tvm-ffi stream context of (device ``index``, raw ``stream``)."""
    key = (threading.get_ident(), index, stream)
    context = _STREAM_CONTEXTS.get(key)
    if context is None:
        context = _STREAM_CONTEXTS[key] = _ffi_stream_context(index, stream)
    return context


def _entry_index(sources: tuple) -> int:
    """The CUDA device index of an entry point's tensors (the last one's), or ``-1`` when they are not all on that
    CUDA device."""
    index = sources[-1].get_device()
    if index < 0:
        return -1
    for t in sources[:-1]:
        if t.get_device() != index:
            return -1
    return index


def _view_recipe(
    view: torch.Tensor, sources: tuple, *, pointer: bool
) -> Optional[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    """``(source, recipe, shape, strides)`` rebuilding the launch view ``view`` from one of the entry point's tensors
    ``sources``, checked against ``view`` (its data pointer, shape, strides and dtype), or ``None`` when none does.  A
    ``pointer`` argument (the program reads its data pointer; the binding checks its dtype, device and contiguity)
    takes a contiguous source of the same data pointer and dtype as is.  The source is the one tensor that starts at
    the view's data pointer: with several (tensors that alias, which the key does not record) there is no recipe."""
    ptr = view.data_ptr()
    found = [s for s, t in enumerate(sources) if t.data_ptr() == ptr]
    if len(found) != 1:
        return None
    s = found[0]
    t = sources[s]
    if t.dtype != view.dtype:
        return None
    shape, strides = tuple(view.shape), tuple(view.stride())
    if (pointer and t.is_contiguous()) or _same_view(t, view):
        return s, _VIEW_SELF, shape, strides
    if t.dim() == 2 and _same_view(t.unsqueeze(0), view):
        return s, _VIEW_UNSQUEEZE, shape, strides
    try:
        strided = t.as_strided(shape, strides)
    except RuntimeError:  # the view leaves this tensor's storage
        return None
    if _same_view(strided, view):
        return s, _VIEW_STRIDED, shape, strides
    return None


def _bind_gemm(
    gemm: _CachedGemm,
    operands: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    sources: tuple,
    pdl: int,
) -> Union[_BoundGemm, bool]:
    """The :class:`_BoundGemm` of the cached launch ``gemm`` for an entry point's tensors ``sources``
    (``operands``: the launch's ``(a_desc, b_desc, out3)`` for them; ``pdl``: the PDL flag, 0 or 1), or ``False``
    when a launch view is not rebuilt from ``sources`` (:func:`_view_recipe`)."""
    a_desc, b_desc, O3 = operands
    views = dict(
        A=(a_desc, False),
        B=(b_desc, False),
        out3=(O3, False),
        o_alias=(flat_alias(O3), True),
    )
    template, recipes, workspace = list(gemm.template), [], []
    for i, item in enumerate(template):
        if isinstance(item, torch.Tensor):
            template[i] = tvm_ffi.from_dlpack(item, require_contiguous=False)
        elif type(item) is _Slot:
            if item.name in _K1_WORKSPACE_KINDS:
                workspace.append((i, _K1_WORKSPACE_KINDS[item.name]))
            elif item.name == "enable_pdl":
                template[i] = pdl
            else:
                view, pointer = views[item.name]
                recipe = _view_recipe(view, sources, pointer=pointer)
                if recipe is None:
                    return False
                recipes.append((i, *recipe))
    return _BoundGemm(gemm, template, tuple(recipes), tuple(workspace))


def _bind_k1_entry(
    key: tuple,
    sources: tuple,
    call: Callable[[], tuple],
    index: int,
    stream: int,
    pdl: int,
) -> Union[_BoundGemm, _SwapLaunch, bool]:
    """Bind the key of an entry point's call: ``call()`` gives the convenience call ``(A, B, out, transposed_out,
    knobs, weights_static)`` the entry point makes (``weights_static`` ``None``: the call never runs the swapped
    forward); its planned launch comes from ``_K1_CACHE`` (planned there on a miss, the planner's errors raised), its
    routed swapped forward from ``_SWAP_CACHE``.  The binding is cached under ``_PLAN_LOCK`` (a key bound meanwhile
    by another thread is returned)."""
    A, B, out, transposed_out, knobs, weights_static = call()
    bound: Union[_BoundGemm, _SwapLaunch, bool] = False
    swap = None
    if weights_static is not None:
        swap = _swap_ab_cached(
            A, B, out, transposed_out, bool(weights_static) and bool(pdl)
        )
    if swap is not None:
        # the routed launch takes the call's own tensors: kept only when they are the entry point's
        if all(a is b for a, b in zip((A, B, out), sources, strict=True)):
            bound = swap
    else:
        gemm, operands = _cached_gemm(
            A, B, out, transposed_out, knobs, index, stream, out.device
        )
        bound = _bind_gemm(gemm, operands, sources, pdl)
    with _PLAN_LOCK:
        hit = _cache_get(_K1_BOUND_CACHE, key)
        if hit is not None:
            return hit
        if type(bound) is _BoundGemm:
            bound.arena.refs += 1
        _cache_put(_K1_BOUND_CACHE, key, bound, on_evict=_release_bound)
    return bound


def _launch_bound(bound: _BoundGemm, sources: tuple, index: int, stream: int) -> None:
    """Launch a bound K1 program with the views of the entry point's tensors ``sources`` in place (rebuilt here;
    torch tensors, each converted in the FFI call), after converting the arena's workspace again if the arena grew
    (under the arena's lock), in the thread's stream context of (device ``index``, raw ``stream``)."""
    arena = bound.gemm.arena
    version, args = bound.state
    if version != arena.version:
        with arena.lock:
            version, args = bound.state
            if version != arena.version:
                # the version is read before converting: a growth that interleaves leaves the state stale
                version = arena.version
                args = bound.template.copy()
                for i, kind in bound.workspace:
                    args[i] = tvm_ffi.from_dlpack(
                        arena.buffers[kind], require_contiguous=False
                    )
                bound.state = (version, args)
    args = args.copy()
    for i, s, recipe, shape, strides in bound.recipes:
        t = sources[s]
        if recipe == _VIEW_UNSQUEEZE:
            t = t.unsqueeze(0)
        elif recipe == _VIEW_STRIDED:
            t = t.as_strided(shape, strides)
        args[i] = t
    with _stream_context(index, stream):
        bound.gemm.entry(*args)


def _k1_entry(
    head: tuple,
    sources: tuple,
    enable_pdl: Optional[bool],
    call: Callable[[], tuple],
) -> bool:
    """Launch a K1 entry point's call through the binding of its key -- ``head`` (the entry point, its flags and
    knobs), the PDL flag (``enable_pdl`` resolved for the device) and the geometry of the entry point's tensors
    ``sources`` (the last is ``out``) on the device and raw stream -- binding the key on its first call
    (:func:`_bind_k1_entry`, ``call`` as there).  ``False`` (nothing launched) when the tensors are not on one CUDA
    device or the key is kept unbound: the entry point then makes the convenience call."""
    index = _entry_index(sources)
    if index < 0:
        return False
    stream = _current_raw_stream(index)
    pdl = int(pdl_enabled(enable_pdl, index))
    A, B, out = sources
    key = (
        index,
        stream,
        head,
        pdl,
        (A.shape, A.stride(), A.dtype, A.storage_offset(), A.data_ptr() & 15),
        (B.shape, B.stride(), B.dtype, B.storage_offset(), B.data_ptr() & 15),
        (out.shape, out.stride(), out.dtype, out.storage_offset(), out.data_ptr() & 15),
    )
    bound = _cache_get(_K1_BOUND_CACHE, key)
    if bound is None:
        bound = _bind_k1_entry(key, sources, call, index, stream, pdl)
    if type(bound) is _BoundGemm:
        _launch_bound(bound, sources, index, stream)
        return True
    if type(bound) is _SwapLaunch:
        _run_swap_ab(bound, A, B, out, bool(pdl))
        return True
    return False


# ---------------------------------------------------------------------------
# K2 planning and prepared launch
# ---------------------------------------------------------------------------


def router_symbol(a_mn: bool, b_mn: bool, out_t: bool) -> str:
    """Kernel symbol / registry template of a router instance (``router_fp32_det_gemm_<a><b>[_t]``)."""
    return (
        "router_fp32_det_gemm_"
        + ("n" if a_mn else "k")
        + ("n" if b_mn else "k")
        + ("_t" if out_t else "")
    )


def router_layout_class(a_mn: bool, b_mn: bool) -> str:
    """Instance layout class of the operand layouts: ``kk`` (forward), ``kn`` (input
    gradient) or ``nn_t`` (both MN-major: the swapped transposed-store weight gradient)."""
    if not a_mn and not b_mn:
        return "kk"
    if not a_mn and b_mn:
        return "kn"
    if a_mn and b_mn:
        return "nn_t"
    raise NotImplementedError(
        "cake_dense_gemm_det_train_router: an MN-major A with a K-major B has no registered instance"
    )


@dataclass(frozen=True)
class CakeDenseGemmDetRouterPlan:
    M: int
    N: int
    K: int
    a_mn: bool
    b_mn: bool
    transposed_out: bool
    swapped: bool  # the caller's (A, B) were swapped into the transposed-store form
    layout: str
    splits: int
    m_tiles: int
    n_tiles: int
    k_blocks: int
    k_iters_split: int
    template: str

    @property
    def num_cluster_tiles(self) -> int:
        return self.splits * (self.m_tiles // CTA_GROUP) * self.n_tiles

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.num_cluster_tiles * CTA_GROUP, 1, 1)


def _router_a_layout(A: torch.Tensor) -> bool:
    if A.stride(1) == 1:
        return False
    if A.stride(0) == 1:
        return True
    raise ValueError(
        "cake_dense_gemm_det_train_router: A needs a unit stride on m or k"
    )


def _router_b_layout(B: torch.Tensor) -> bool:
    if B.stride(0) == 1:  # k contiguous -> K-major parts [3, N, K]
        return False
    if B.stride(1) == 1:  # n contiguous -> MN-major parts [3, K, N]
        return True
    raise ValueError(
        "cake_dense_gemm_det_train_router: B needs a unit stride on k or n"
    )


def router_splits(
    layout: str,
    M: int,
    N: int,
    K: int,
    arch: Optional[str] = None,
    sm_count: Optional[int] = None,
) -> int:
    """The split-K count of a router plan without a caller's ``splits``: the :data:`ROUTER_SPLIT_RULES` count of
    ``(arch, sm_count, layout, H, N)`` whose window holds the token count (``kk``: ``H = K``, ``T = M``; ``nn_t``, the
    swapped weight gradient's ``M`` / ``K``: ``H = M``, ``T = K``; ``N`` the experts of the row), else
    :data:`ROUTER_SPLITS` of the layout (also for an unknown device: ``arch`` or ``sm_count`` ``None``)."""
    if arch is not None and sm_count is not None:
        H, T = (int(K), int(M)) if layout == "kk" else (int(M), int(K))
        for lo, hi, splits in ROUTER_SPLIT_RULES.get(
            (arch, int(sm_count), layout, H, int(N)), ()
        ):
            if lo <= T <= hi:
                return splits
    return ROUTER_SPLITS[layout]


def cake_dense_gemm_det_train_router_plan(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
) -> tuple[CakeDenseGemmDetRouterPlan, torch.Tensor, torch.Tensor]:
    """Classify the fp32 views, apply the swap rule of the weight gradient and plan the launch.

    Returns ``(plan, A_eff, B_eff)``: the operands of the launched (possibly swapped) GEMM;
    ``B_eff`` is the operand the host splits into BF16x3 parts.  ``splits`` overrides the
    split-K count :func:`router_splits` resolves for the device of ``out`` (a CUDA tensor; on another device the
    layout's :data:`ROUTER_SPLITS` count)."""
    if (
        A.dtype != torch.float32
        or B.dtype != torch.float32
        or out.dtype != torch.float32
    ):
        raise ValueError("cake_dense_gemm_det_train_router: A, B and out must be fp32")
    if A.dim() != 2 or B.dim() != 2 or out.dim() != 2:
        raise ValueError(
            "cake_dense_gemm_det_train_router: A, B and out must be 2-D views"
        )
    M, K = (int(s) for s in A.shape)
    Kb, N = (int(s) for s in B.shape)
    if Kb != K:
        raise ValueError(
            f"cake_dense_gemm_det_train_router: A is [M={M}, K={K}] but B is [{Kb}, {N}]; K must agree"
        )
    if tuple(out.shape) != (M, N):
        raise ValueError(
            f"cake_dense_gemm_det_train_router: out must be [{M}, {N}], got {tuple(out.shape)}"
        )
    if M < 1 or K < 1 or N < 1:
        # An empty problem (a ragged batch with T = 0 reaches here as M = 0 for the
        # forward / input gradient or K = 0 for the weight gradient) would otherwise
        # produce a zero-sized grid or a TMA descriptor with a zero extent.
        raise ValueError(
            f"cake_dense_gemm_det_train_router: M, K and N must be >= 1 (M={M}, N={N}, K={K})"
        )
    a_mn, b_mn = _router_a_layout(A), _router_b_layout(B)
    layout = router_layout_class(a_mn, b_mn)
    swapped = layout == "nn_t"
    if swapped:
        # out.T[n, m] = sum_k B.T[n, k] A.T[k, m]: A' = B.T (MN-major), B' = A.T (MN-major), transposed store
        A_eff, B_eff, transposed = B.t(), A.t(), True
        M_eff, N_eff = N, M
    else:
        A_eff, B_eff, transposed = A, B, False
        M_eff, N_eff = M, N
    a_eff_mn, b_eff_mn = _router_a_layout(A_eff), _router_b_layout(B_eff)
    lda = int(A_eff.stride(1)) if a_eff_mn else int(A_eff.stride(0))
    if lda % 4 != 0 or A_eff.data_ptr() % 16 != 0:
        raise ValueError(
            "cake_dense_gemm_det_train_router: A rows must be 16-byte aligned"
        )
    if B_eff.data_ptr() % 16 != 0:
        raise ValueError(
            "cake_dense_gemm_det_train_router: B must start at a 16-byte aligned address"
        )
    # the kernel's output view: [M_eff, N_eff] row-major, or [N_eff, M_eff] with the transposed store = out
    if out.stride(1) != 1 or out.stride(0) % 4 != 0 or N_eff % 8 != 0:
        raise ValueError(
            "cake_dense_gemm_det_train_router: out must have unit inner stride and 16-byte rows; N a multiple of 8"
        )
    if splits is None:
        facts = device_facts(out.device) if out.is_cuda else None
        splits = router_splits(
            layout,
            M_eff,
            N_eff,
            K,
            None if facts is None else facts.arch,
            None if facts is None else facts.sm_count,
        )
    splits = int(splits)
    if splits < 1:
        raise ValueError("cake_dense_gemm_det_train_router: splits must be >= 1")
    # one split stores straight into ``out``: the row-major store (kk / kn) writes 16-byte vectors, the
    # transposed store (nn_t) one fp32 element at a time; more splits store into the workspace and reduce into ``out``
    if splits == 1 and not transposed and out.data_ptr() % 16 != 0:
        raise ValueError(
            "cake_dense_gemm_det_train_router: out must start at a 16-byte aligned address "
            f"(the single-split {layout} launch stores 16-byte vectors into it)"
        )
    m_tiles = _ceil_div(M_eff, ROUTER_BLOCK_M)
    m_tiles += m_tiles % CTA_GROUP
    n_tiles = _ceil_div(N_eff, ROUTER_BLOCK_N)
    k_blocks = _ceil_div(K, ROUTER_BLOCK_K)
    plan = CakeDenseGemmDetRouterPlan(
        M=M_eff,
        N=N_eff,
        K=K,
        a_mn=a_eff_mn,
        b_mn=b_eff_mn,
        transposed_out=transposed,
        swapped=swapped,
        layout=layout,
        splits=splits,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        k_iters_split=_ceil_div(k_blocks, splits),
        template=router_symbol(a_eff_mn, b_eff_mn, transposed),
    )
    return plan, A_eff, B_eff


def split_fp32_to_bf16x3_det_(
    src: torch.Tensor, parts: torch.Tensor, resid: torch.Tensor
) -> torch.Tensor:
    """Exact three-way BF16 split ``src = p1 + p2 + p3`` (``p1 = RN(src)``, ``p2 = RN(src - p1)``,
    ``p3 = RN(src - p1 - p2)``) into the retained ``parts [3, *src.shape]`` BF16 stack with the
    fp32 scratch ``resid`` -- no allocation; the same values as the Cake host's
    ``split_fp32_to_bf16x3``."""
    p1, p2, p3 = parts[0], parts[1], parts[2]
    p1.copy_(src)
    torch.sub(src, p1, out=resid)  # exact in fp32
    p2.copy_(resid)
    torch.sub(resid, p2, out=resid)  # exact in fp32
    p3.copy_(resid)
    return parts


# The fused split program: one launch reads the fp32 source once and writes the three BF16 parts (the residual
# stays in registers, so it needs no fp32 scratch).  The K2 paths take it when the registry holds it for the
# device's architecture and the source is a contiguous view, and split with the torch chain
# :func:`split_fp32_to_bf16x3_det_` otherwise -- the same bits either way for finite inputs (a NaN differs in its
# payload: the program writes the canonical BF16 NaN).  Its launch is ``exe(src, parts, n)`` over
# ``grid = ceil(n / SPLIT_BLOCK_ELEMS)`` blocks of 256 threads, 8 elements per thread.
SPLIT_TEMPLATE = "split_fp32_det_bf16x3"
SPLIT_BLOCK_ELEMS = 2048
_SPLIT_MAX_N = 2**31 - 1  # ``n`` is an int32 launch parameter


def fused_split_module(arch: str) -> Optional[str]:
    """The registered module serving the fused split program on ``arch``, or ``None`` (the torch chain splits)."""
    if SPLIT_TEMPLATE not in KERNELS.get(arch, {}):
        return None
    return select_module(arch, SPLIT_TEMPLATE)


def _fused_split_for(arch: str, split_src: torch.Tensor) -> Optional[str]:
    """The fused split's module when it is registered for ``arch`` and can split ``split_src``: a contiguous
    fp32 view (the program reads the elements in order; a storage offset is fine) of at most int32 elements.
    ``None`` selects the torch chain."""
    if not (split_src.is_contiguous() and split_src.numel() <= _SPLIT_MAX_N):
        return None
    return fused_split_module(arch)


def _split_values(src: Any, parts: Any, n: int, enable_pdl: Any = 0) -> dict[str, Any]:
    """The keyword arguments of a fused split launch -- the fp32 source, the ``[3, *src.shape]`` BF16 stack,
    the element count and the PDL flag -- ordered by the module's argument plan in :func:`_bind`.  The one place
    the program's binding names live."""
    return dict(src=src, parts=parts, n=int(n), enable_pdl=enable_pdl)


def _split_grid(n: int) -> tuple[int, int, int]:
    """The grid of a fused split launch over ``n`` elements."""
    return (_ceil_div(int(n), SPLIT_BLOCK_ELEMS), 1, 1)


@dataclass
class CakeDenseGemmDetRouterLaunch:
    """One prepared FP32 router GEMM: the retained BF16x3 stack of the second operand (re-split
    from the bound fp32 view on every ``launch()`` -- one launch of the fused split program when
    this checkout registers it for the device, else the torch chain through the fp32 scratch
    ``resid``), the split-K workspace and the fixed-order host reduction.  ``launch()`` writes
    ``out`` and returns it without allocating."""

    module_name: str
    arch: str
    plan: CakeDenseGemmDetRouterPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_Launch] = field(default=None, repr=False)
    _split: Optional[_Launch] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    @property
    def splits(self) -> int:
        return self.plan.splits

    @property
    def fused_split(self) -> bool:
        """Whether ``launch()`` splits with the fused program (else the torch chain)."""
        return self._split is not None

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        t = self.tensors
        if self._split is None:
            split_fp32_to_bf16x3_det_(t["split_src"], t["parts"], t["resid"])
        with _ffi_stream_context(self.device_index):
            if self._split is not None:
                self._split()
            self._launch()
        if self.plan.splits > 1:
            torch.sum(
                t["workspace"], dim=0, out=self.out
            )  # fixed-order reduction of the partials
        return self.out

    __call__ = launch


def _k2_plan(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, splits: Optional[int]
) -> tuple[CakeDenseGemmDetRouterPlan, torch.Tensor, torch.Tensor, str, str]:
    """Plan one K2 binding for its device: ``(plan, A_eff, B_eff, arch, module_name)``."""
    device = _same_cuda_device(A, B, out)
    arch = require_arch(device)
    plan, A_eff, B_eff = cake_dense_gemm_det_train_router_plan(A, B, out, splits=splits)
    return plan, A_eff, B_eff, arch, select_module(arch, plan.template)


def _k2_views(
    A: torch.Tensor, B: torch.Tensor, plan: CakeDenseGemmDetRouterPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a_view, split_src)`` of a router launch for the caller's operands: the streamed operand as the
    ``[1, M, K]`` / ``[1, K, M]`` TMA span and the second operand in its own layout -- the source of the BF16x3
    split (K-major parts ``[3, N, K]`` come from ``B.T``, MN-major parts ``[3, K, N]`` from ``B``) -- after the
    weight gradient's swap when the plan swapped (``out.T = B.T A.T``)."""
    A_eff, B_eff = (B.t(), A.t()) if plan.swapped else (A, B)
    split_src = B_eff if plan.b_mn else B_eff.t()
    a_view = A_eff.t().unsqueeze(0) if plan.a_mn else A_eff.unsqueeze(0)
    return a_view, split_src


def _k2_checked_views(
    A: torch.Tensor,
    B: torch.Tensor,
    A_eff: torch.Tensor,
    B_eff: torch.Tensor,
    plan: CakeDenseGemmDetRouterPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_k2_views` of the caller's operands, checked against the planner's operands ``(A_eff, B_eff)``
    of :func:`cake_dense_gemm_det_train_router_plan` in the launch form (the streamed ``[1, M, K]`` / ``[1, K, M]`` span under
    ``plan.a_mn``, the split source under ``plan.b_mn``).  The convenience path rebuilds the views from the
    caller's tensors on every call, so a rebuild that is not the planner's view fails closed here (the K1 check
    of :func:`_prepare_cached_gemm`) instead of launching on a view the plan never validated."""
    a_view, split_src = _k2_views(A, B, plan)
    want_a = A_eff.t().unsqueeze(0) if plan.a_mn else A_eff.unsqueeze(0)
    want_split = B_eff if plan.b_mn else B_eff.t()
    if not (_same_view(a_view, want_a) and _same_view(split_src, want_split)):
        raise RuntimeError(
            "cake_dense_gemm_det_train_router: the per-call operand views differ from the planner's; "
            "the convenience path cannot cache this binding (use cake_dense_gemm_det_train_router_prepare)"
        )
    return a_view, split_src


def _k2_workspace(
    plan: CakeDenseGemmDetRouterPlan,
    split_src: torch.Tensor,
    device: torch.device,
    *,
    scratch: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The retained BF16x3 stack of the second operand, the fp32 residual scratch of the torch-chain split
    (``None`` when the fused split program splits: ``scratch=False``) and the split-K workspace (``None`` for a
    single split: the kernel writes ``out`` directly)."""
    parts = torch.empty((3, *split_src.shape), device=device, dtype=torch.bfloat16)
    resid = None
    if scratch:
        resid = torch.empty(split_src.shape, device=device, dtype=torch.float32)
    workspace = None
    if plan.splits > 1:
        expect = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
        workspace = torch.empty(
            (plan.splits, *expect), device=device, dtype=torch.float32
        )
    return parts, resid, workspace


def _k2_workspace_strides(workspace: torch.Tensor) -> tuple[int, int]:
    """``(ldo, out_l)`` of the ``[splits, rows, cols]`` split-K workspace: one row-major partial per split."""
    return int(workspace.shape[2]), int(workspace.shape[1] * workspace.shape[2])


def _k2_out32(out: torch.Tensor) -> torch.Tensor:
    """The pointer argument for a single-split launch that writes ``out`` directly."""
    return out if out.is_contiguous() else flat_alias(out)


def _k2_values(
    plan: CakeDenseGemmDetRouterPlan,
    *,
    A: Any,
    parts: Any,
    out32: Any,
    ldo: int,
    out_l: int,
    enable_pdl: Any,
) -> dict[str, Any]:
    """The keyword arguments of a K2 launch; ``A`` / ``parts`` / ``out32`` are views or :class:`_Slot`
    markers, ``enable_pdl`` the launch's PDL flag (0 or 1) or a :class:`_Slot`."""
    return dict(
        A=A,
        B=parts,
        out32=out32,
        M=plan.M,
        N=plan.N,
        m_tiles=plan.m_tiles,
        n_tiles=plan.n_tiles,
        k_iters_split=plan.k_iters_split,
        ldo=ldo,
        out_l=out_l,
        num_cluster_tiles=plan.num_cluster_tiles,
        enable_pdl=enable_pdl,
    )


def cake_dense_gemm_det_train_router_prepare(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> CakeDenseGemmDetRouterLaunch:
    """Validate one fp32 binding, plan it and prepare its launch (the only allocations of the
    K2 backend: the BF16x3 stack, the fp32 residual scratch of the torch-chain split and the
    split-K workspace).  ``enable_pdl`` (default: the device supports it) launches the fused split and the
    router as programmatic dependent launches, each waiting for the preceding kernel before its first global
    access."""
    plan, _, _, arch, module_name = _k2_plan(A, B, out, splits=splits)
    device = out.device
    a_view, split_src = _k2_views(A, B, plan)
    split_module = _fused_split_for(arch, split_src)
    parts, resid, workspace = _k2_workspace(
        plan, split_src, device, scratch=split_module is None
    )
    tensors = dict(A=a_view, split_src=split_src, parts=parts)
    if resid is not None:
        tensors["resid"] = resid
    if workspace is not None:
        tensors["workspace"] = workspace
        out32, ldo, out_l = workspace, *_k2_workspace_strides(workspace)
    else:
        out32, ldo, out_l = _k2_out32(out), int(out.stride(0)), 0
    pdl = int(pdl_enabled(enable_pdl, device))
    values = _k2_values(
        plan, A=a_view, parts=parts, out32=out32, ldo=ldo, out_l=out_l, enable_pdl=pdl
    )
    launch = bind_launch(module_name, values, plan.grid, arch)
    split_launch = None
    if split_module is not None:
        n = split_src.numel()
        split_launch = bind_launch(
            split_module, _split_values(split_src, parts, n, pdl), _split_grid(n), arch
        )
    return CakeDenseGemmDetRouterLaunch(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=tensors,
        device_index=_device_index(device),
        _launch=launch,
        _split=split_launch,
    )


@dataclass
class _CachedRouter:
    """A planned K2 launch of the convenience path: the plan, the resolved module entry, the argument template,
    the fused split's ``(entry, template)`` when the program is registered for the device (``None``: the torch
    chain splits) and the shapes of the workspace it carves out of its arena (the BF16x3 stack
    ``[3, *split_shape]``, under the torch chain the fp32 scratch ``split_shape`` and, for more than one split,
    the partials ``partials_shape``)."""

    module_name: str
    arch: str
    plan: CakeDenseGemmDetRouterPlan
    entry: Callable[..., Any] = field(repr=False)
    template: tuple = field(repr=False)
    arena: _Arena = field(repr=False)
    split_shape: tuple[int, ...]
    partials_shape: Optional[tuple[int, ...]] = None
    views: Optional[dict[str, torch.Tensor]] = field(default=None, repr=False)
    views_version: int = -1
    split: Optional[tuple[Callable[..., Any], tuple]] = field(default=None, repr=False)

    def carve(self, arena: _Arena) -> dict[str, torch.Tensor]:
        """The ``parts`` stack (the TMA operand ``B``: exact shape), the ``resid`` scratch (torch chain only) and
        the split-K ``workspace`` as contiguous views over the first elements of the arena buffers."""
        bufs = arena.buffers
        views = dict(parts=_carve(bufs["k2_parts"], (3, *self.split_shape)))
        if self.split is None:
            views["resid"] = _carve(bufs["k2_resid"], self.split_shape)
        if self.partials_shape is not None:
            views["workspace"] = _carve(bufs["k2_partials"], self.partials_shape)
        return views


_K2_CACHE: "OrderedDict[tuple, _CachedRouter]" = OrderedDict()


def cake_dense_gemm_det_train_convenience_cache_clear() -> None:
    """Drop every cached convenience-path launch (K1 and K2), every K1 entry-point binding with the stream contexts
    the bindings made, and release every workspace arena; a launch in
    flight keeps a reference to the buffers it was bound with until its FFI call has enqueued it, and the
    freed blocks are reused stream-ordered (see :class:`_Arena`)."""
    with _PLAN_LOCK:
        _K1_CACHE.clear()
        _K2_CACHE.clear()
        _SWAP_CACHE.clear()
        _K1_BOUND_CACHE.clear()
        _STREAM_CONTEXTS.clear()
        _ARENAS.clear()


def _prepare_cached_router(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
    arena: _Arena,
) -> tuple[_CachedRouter, tuple[torch.Tensor, torch.Tensor]]:
    """``(entry, (a_view, split_src))``: the planned entry and the per-call views of this first call (the
    rebuild the fail-closed check validated against the planner's operands)."""
    plan, A_eff, B_eff, arch, module_name = _k2_plan(A, B, out, splits=splits)
    a_view, split_src = _k2_checked_views(A, B, A_eff, B_eff, plan)
    split_shape = tuple(int(s) for s in split_src.shape)
    n = split_src.numel()
    split_module = _fused_split_for(arch, split_src)
    partials_shape = None
    if plan.splits > 1:
        rows, cols = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
        partials_shape = (plan.splits, rows, cols)
        ldo, out_l = cols, rows * cols
    else:
        ldo, out_l = int(out.stride(0)), 0
    values = _k2_values(
        plan,
        A=_Slot("A"),
        parts=_Slot("parts"),
        out32=_Slot("out32"),
        ldo=ldo,
        out_l=out_l,
        enable_pdl=_Slot("enable_pdl"),
    )
    entry, template = _bind(module_name, values, plan.grid, arch)
    split = None
    if split_module is not None:
        split = _bind(
            split_module,
            _split_values(_Slot("src"), _Slot("parts"), n, _Slot("enable_pdl")),
            _split_grid(n),
            arch,
        )
    # the workspace is a slot: grow the arena only once the modules and the grids are known to be good
    arena.require("k2_parts", 3 * n)
    if split is None:
        arena.require("k2_resid", n)
    if partials_shape is not None:
        arena.require("k2_partials", _numel(partials_shape))
    arena.refs += 1
    return _CachedRouter(
        module_name,
        arch,
        plan,
        entry,
        template,
        arena,
        split_shape,
        partials_shape,
        split=split,
    ), (a_view, split_src)


def _launch_cached_router(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """The convenience path of K2: plan on the first call for a (device, stream, geometry, splits) key; on every
    call re-split the second operand into the BF16x3 stack carved out of the arena of the device and stream (one
    launch of the fused split program when it is registered for the device, else the torch chain through the
    arena's fp32 scratch), launch and reduce the split-K partials into ``out`` in the fixed order."""
    device = _same_cuda_device(A, B, out)
    index = _cuda_index(device)
    stream = _current_raw_stream(index)
    key = convenience_key(
        index, stream, (None if splits is None else int(splits),), A, B, out
    )
    cached = _cache_get(_K2_CACHE, key)
    operands = None
    if cached is None:
        cached, operands = _plan_entry(
            _K2_CACHE,
            key,
            index,
            stream,
            device,
            lambda arena: _prepare_cached_router(A, B, out, splits=splits, arena=arena),
        )
    if operands is None:
        operands = _k2_views(A, B, cached.plan)
    a_view, split_src = operands
    # One K2 call is three stream operations on the shared workspace (the split into ``parts`` / ``resid``,
    # the kernel, the reduction of the partials): hold the arena's lock across them so another thread issuing
    # on this stream cannot interleave its own split or reduction (enqueue-only critical section).
    with cached.arena.lock:
        ws = _entry_views(cached)
        if cached.split is None:
            split_fp32_to_bf16x3_det_(split_src, ws["parts"], ws["resid"])
        workspace = ws.get("workspace")
        views = dict(
            A=a_view,
            parts=ws["parts"],
            out32=_k2_out32(out) if workspace is None else workspace,
            src=split_src,
            enable_pdl=int(pdl_enabled(enable_pdl, index)),
        )
        with _ffi_stream_context(index, stream):
            if cached.split is not None:
                split_entry, split_template = cached.split
                split_entry(*_resolve_arguments(split_template, views))
            cached.entry(*_resolve_arguments(cached.template, views))
        if workspace is not None:
            torch.sum(
                workspace, dim=0, out=out
            )  # fixed-order reduction of the partials
    return out


def cake_dense_gemm_det_train_router(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[m, n] = sum_k A[m, k] B[k, n]`` for fp32 views through the emulation (the cached convenience
    path: planned once per device, stream, view geometry and split count, then re-split and launched).
    ``enable_pdl`` as for :func:`cake_dense_gemm_det_train_router_prepare`."""
    return _launch_cached_router(A, B, out, splits=splits, enable_pdl=enable_pdl)


def cake_dense_gemm_det_train_router_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` in FP32 through the emulation; ``splits`` defaults to
    :func:`router_splits` of the device and the row."""
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.float32)
    return cake_dense_gemm_det_train_router(
        X, W.t(), out, splits=splits, enable_pdl=enable_pdl
    )


def cake_dense_gemm_det_train_router_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]`` in FP32 through the emulation."""
    if out is None:
        out = torch.empty(G.shape[0], W.shape[1], device=G.device, dtype=torch.float32)
    return cake_dense_gemm_det_train_router(G, W, out, splits=1, enable_pdl=enable_pdl)


def cake_dense_gemm_det_train_router_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: Optional[int] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` in FP32: the swapped GEMM ``X.T @ G`` with a transposed store; ``splits``
    defaults to :func:`router_splits` of the device and the row."""
    if out is None:
        out = torch.empty(G.shape[1], X.shape[1], device=G.device, dtype=torch.float32)
    return cake_dense_gemm_det_train_router(
        G.t(), X, out, splits=splits, enable_pdl=enable_pdl
    )
