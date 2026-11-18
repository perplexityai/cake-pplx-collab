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

Benchmark of the Cake Engram row lookup (``cake_pplx.cake_engram_lookup_det_train``) against the PyTorch release path
of the feature request, eager and compiled, on one GPU, single rank.

Arms (``--arms``; default every arm but ``torch_eager_u8``):

* ``cake``: ``cake_engram_lookup_det_train_forward(weight, scale, row_ids)``, the public entry, its output allocation
  included.
* ``cake_aa``: the same call as a second arm (the A/A copy): its ratio to ``cake`` shows the method's noise.
* ``torch_eager``: ``row_lookup_fwd_released(weight, scale, row_ids)`` of the feature request, verbatim
  (``tests/cake/test_helpers/cake_engram_lookup_ref.py``): gather, FP32 multiply, BF16 cast; six kernels and FP32
  intermediates.  ``torch_eager_u8`` is the same formula with the two gathers done on ``uint8`` views (for a torch
  without CUDA indexing of ``float8_e8m0fnu``; reported as such).
* ``torch_compile`` / ``torch_compile_max_autotune``: the same function under ``torch.compile(fullgraph=True,
  dynamic=False)``, modes ``default`` and ``max-autotune-no-cudagraphs``, compiled for each row and layout before
  timing.  On sm_107a, Triton needs the CUDA 13.5 ``ptxas``: point ``TRITON_PTXAS_BLACKWELL_PATH`` at it.
* ``copy``: a device-to-device copy of the bytes the lookup moves (``P * 784`` bytes read and written for ``P``
  looked-up rows: 264 table bytes, 8 id bytes, 512 output bytes; a copy of ``P * 392`` bytes), the same-node roof.
* ``cake_det_mode``: ``cake`` under ``torch.use_deterministic_algorithms(True)``, where PyTorch fills each new
  output with NaN first (information: the cost of that fill for a caller that runs in the mode).

Bit gate (before any timing; ``--no-gate`` skips it): every arm but ``copy`` looks up the exhaustive table (all 65,536
E4M3 code / E8M0 byte pairs) in both layouts through ``[M]`` and ``[T, 24]`` ids, random tables with NaN scale bytes,
one row, repeated ids and empty ids, compared bit for bit with the FP64 reference (NaN at the same elements).  A
compiled arm that fails any case is reported "not equal" and is not timed; the other arms are timed and carry their
gate result.

Rows (``--rows``; R = ``--shard-rows``, default 6,000,000; every row in both layouts with the same bytes and ids):

* ``U1`` / ``U2``: ids ``[16384, 24]`` / ``[65536, 24]`` uniform over R (393,216 / 1,572,864 looked-up rows);
* ``D1`` / ``D2``: ids ``[393216]`` / ``[1572864]``, unique and sorted (deduplicated rows);
* ``H1`` / ``H2``: as ``U1`` / ``U2`` over a hot set of 65,536 rows (repeated ids, L2 hits);
* ``S1`` / ``S1m``: ids ``[1, 24]`` / ``[1]`` (launch-bound);
* ``R64``: ids ``[16384, 24]`` over a 64-row table (with ``U1``: time and memory follow the looked-up rows, not R);
* ``F1``: a fetched buffer of 393,216 rows, ids a permutation of it (rows fetched from other ranks).

Table bytes: E4M3 codes uniform over the 254 non-NaN codes, E8M0 bytes uniform over 118 .. 136 (scales 2^-9 ..
2^9), from ``--seed``.

Timing (``cake_engram_lookup_timer.py``): per iteration an L2 flush, then the CUPTI span from the first kernel start to
the last kernel end of the iteration's own launches (CUDA events without ``cupti-python``); ``eager`` mode puts a spin
kernel ahead of each iteration so its host time stays outside the span, ``graph`` mode replays a CUDA graph of one
call.  Per row, layout and mode: ``--groups`` groups; group ``g`` runs every arm (``--warmup`` untimed then
``--iters`` timed iterations) in the arm list rotated left by ``g mod n`` and reversed when ``g // n`` is odd.  An
arm's time is the median over the groups of its group medians; its spread is (max - min) / median of its group
medians.  A ratio (speedup = baseline / Cake; roof fraction = copy / Cake) is the median over the groups of the ratio
of the group medians, with its [min, max].  The noise band is the largest of 0.5 %, the Cake arm's spread and the A/A
deviation; a row where an admitted baseline is faster than Cake by more than the band is timed again with
``--retime-groups`` groups.  Extra memory per arm: the peak allocation of one call above the allocation before it
(the output included).

Output: one JSON (``--json``: identity, settings, gate, rows) and a Markdown table (``--markdown``) with the GPU,
architecture, SM count, driver, clocks, CUDA, torch and Triton versions, absolute times, speedups, effective TB/s
(``P * 784`` bytes over the time), the fraction of the copy roof and the extra memory per arm.

    python benchmarks/bench_cake_engram_lookup_det_train.py --json engram.json --markdown engram.md
    python benchmarks/bench_cake_engram_lookup_det_train.py --rows U1,U2 --layouts packed --modes graph
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from cake_engram_lookup_timer import GpuTimer, capture_graph  # noqa: E402

import cake_pplx.cake_engram_lookup_det_train as api  # noqa: E402
from cake_pplx._backends.cake_engram_lookup_det_train import cake_backend, cake_build  # noqa: E402
from tests.cake.test_helpers.cake_engram_lookup_ref import (  # noqa: E402
    LAYOUTS,
    N_COLS,
    PACKED_ROW_BYTES,
    ROW_DIM,
    Table,
    assert_same_bits,
    exhaustive_bytes,
    make_table,
    oracle_bits,
    random_bytes,
    random_ids,
    row_lookup_fwd_released,
)

ARMS = (
    "cake",
    "cake_aa",
    "torch_eager",
    "torch_eager_u8",
    "torch_compile",
    "torch_compile_max_autotune",
    "copy",
    "cake_det_mode",
)
DEFAULT_ARMS = tuple(arm for arm in ARMS if arm != "torch_eager_u8")
COMPILE_MODES = {"torch_compile": "default", "torch_compile_max_autotune": "max-autotune-no-cudagraphs"}
BASELINES = ("torch_eager", "torch_eager_u8", "torch_compile", "torch_compile_max_autotune")
MODES = ("eager", "graph")
# Bytes one looked-up row moves: its 264 table bytes and 8 id bytes read, its 512 output bytes written.
BYTES_PER_ROW = PACKED_ROW_BYTES + 8 + 2 * ROW_DIM
HOT_ROWS = 65_536
FETCHED_ROWS = 393_216
SMALL_ROWS = 64
# row -> (table, id kind, id shape)
ROWS: dict[str, tuple[str, str, tuple[int, ...]]] = {
    "U1": ("shard", "uniform", (16_384, N_COLS)),
    "U2": ("shard", "uniform", (65_536, N_COLS)),
    "D1": ("shard", "unique_sorted", (393_216,)),
    "D2": ("shard", "unique_sorted", (1_572_864,)),
    "H1": ("shard", "hot", (16_384, N_COLS)),
    "H2": ("shard", "hot", (65_536, N_COLS)),
    "S1": ("shard", "uniform", (1, N_COLS)),
    "S1m": ("shard", "uniform", (1,)),
    "R64": ("small", "uniform", (16_384, N_COLS)),
    "F1": ("fetched", "permutation", (FETCHED_ROWS,)),
}
MIN_NOISE = 0.005


# ----------------------------------------------------------------------------------------------------------------
# Arms


def released_u8(weight: torch.Tensor, scale: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    """``row_lookup_fwd_released`` with the gathers on ``uint8`` views: the same formula."""
    vals = weight.view(torch.uint8)[row_ids].view(torch.float8_e4m3fn)
    scls = scale.view(torch.uint8)[row_ids].view(torch.float8_e8m0fnu)
    out = vals.float().unflatten(-1, (8, 32)) * scls.float().unsqueeze(-1)
    return out.flatten(-2).to(torch.bfloat16)


def _raise_compile_limits() -> None:
    # One specialization per row, layout and gate case: keep every one compiled (no fallback to eager).
    config = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(config, name):
            setattr(config, name, 4096)


@contextlib.contextmanager
def _deterministic_algorithms():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def make_lookups(names: list[str]) -> dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]]:
    """The lookup callable ``(weight, scale, row_ids) -> out`` of every arm but ``copy``."""
    forward = api.cake_engram_lookup_det_train_forward

    def det_mode(weight, scale, row_ids):
        with _deterministic_algorithms():
            return forward(weight, scale, row_ids)

    lookups: dict[str, Callable] = {}
    for name in names:
        if name in ("cake", "cake_aa"):
            lookups[name] = forward
        elif name == "cake_det_mode":
            lookups[name] = det_mode
        elif name == "torch_eager":
            lookups[name] = row_lookup_fwd_released
        elif name == "torch_eager_u8":
            lookups[name] = released_u8
        elif name in COMPILE_MODES:
            _raise_compile_limits()
            lookups[name] = torch.compile(
                row_lookup_fwd_released, fullgraph=True, dynamic=False, mode=COMPILE_MODES[name]
            )
        elif name != "copy":
            raise ValueError(f"unknown arm {name!r}; the arms are {ARMS}")
    return lookups


# ----------------------------------------------------------------------------------------------------------------
# Bit gate


def gate_cases(device: torch.device) -> list[tuple[str, Table, torch.Tensor]]:
    cases = []
    weight_bytes, scale_bytes = exhaustive_bytes()
    nan_weight, nan_scale = random_bytes(512, seed=7, nan_every=5)
    for layout in LAYOUTS:
        table = make_table(weight_bytes, scale_bytes, layout, device)
        cases.append((f"exhaustive-{layout}-M", table, torch.arange(256, device=device)))
        cases.append(
            (f"exhaustive-{layout}-T24", table, (torch.arange(11 * N_COLS, device=device) % 256).view(11, N_COLS))
        )
        rand = make_table(nan_weight, nan_scale, layout, device)
        cases.append((f"random-{layout}-T24", rand, random_ids(512, (64, N_COLS), seed=8, device=device)))
        cases.append((f"random-{layout}-M1", rand, random_ids(512, (1,), seed=9, device=device)))
        cases.append((f"random-{layout}-repeats", rand, torch.full((4096,), 5, dtype=torch.int64, device=device)))
        cases.append((f"random-{layout}-T0", rand, torch.empty(0, N_COLS, dtype=torch.int64, device=device)))
        cases.append((f"random-{layout}-M0", rand, torch.empty(0, dtype=torch.int64, device=device)))
    return cases


def run_gate(lookup: Callable, cases) -> dict[str, Any]:
    results = {}
    for label, table, ids in cases:
        try:
            out = lookup(table.weight, table.scale, ids)
            torch.cuda.synchronize()
            assert_same_bits(out, oracle_bits(table, ids), label)
            results[label] = "equal"
        except AssertionError as e:
            results[label] = f"differs: {str(e)[:400]}"
        except Exception as e:  # noqa: BLE001 -- an arm that cannot run a case fails the gate with the reason
            results[label] = f"error: {type(e).__name__}: {str(e)[:400]}"
    return {"passed": all(v == "equal" for v in results.values()), "cases": results}


# ----------------------------------------------------------------------------------------------------------------
# Tables and ids


def _random_table_bytes(rows: int, gen: torch.Generator, device: torch.device, chunk: int = 1 << 20):
    """Yield ``(start, weight_bytes, scale_bytes)`` chunks of a random table: non-NaN E4M3 codes, E8M0 bytes 118 ..
    136."""
    for start in range(0, rows, chunk):
        n = min(chunk, rows - start)
        weight = torch.randint(0, 256, (n, ROW_DIM), generator=gen, device=device, dtype=torch.uint8)
        weight = torch.where((weight & 0x7F) == 0x7F, weight - 1, weight)
        scale = torch.randint(118, 137, (n, 8), generator=gen, device=device, dtype=torch.uint8)
        yield start, weight, scale


def build_tables(rows: int, seed: int, device: torch.device) -> dict[str, Table]:
    """The same random bytes as a contiguous table and as a packed one."""
    gen = torch.Generator(device=device).manual_seed(seed)
    weight = torch.empty(rows, ROW_DIM, dtype=torch.uint8, device=device)
    scale = torch.empty(rows, 8, dtype=torch.uint8, device=device)
    packed = torch.empty(rows, PACKED_ROW_BYTES, dtype=torch.uint8, device=device)
    for start, w, s in _random_table_bytes(rows, gen, device):
        stop = start + w.shape[0]
        weight[start:stop] = w
        scale[start:stop] = s
        packed[start:stop, :ROW_DIM] = w
        packed[start:stop, ROW_DIM:] = s
    contiguous = Table("contiguous", weight.view(torch.float8_e4m3fn), scale.view(torch.float8_e8m0fnu))
    w_view, s_view = api.cake_engram_lookup_det_train_split_packed(packed)
    return {"contiguous": contiguous, "packed": Table("packed", w_view, s_view, packed)}


def build_ids(kind: str, shape: tuple[int, ...], rows: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    count = 1
    for n in shape:
        count *= n
    if kind == "uniform":
        ids = torch.randint(0, rows, (count,), generator=gen, device=device)
    elif kind == "unique_sorted":
        ids = torch.randperm(rows, generator=gen, device=device)[:count].sort().values
    elif kind == "hot":
        hot = torch.randperm(rows, generator=gen, device=device)[:HOT_ROWS]
        ids = hot[torch.randint(0, HOT_ROWS, (count,), generator=gen, device=device)]
    elif kind == "permutation":
        ids = torch.randperm(rows, generator=gen, device=device)[:count]
    else:
        raise ValueError(f"unknown id kind {kind!r}")
    return ids.to(torch.int64).view(shape).contiguous()


# ----------------------------------------------------------------------------------------------------------------
# Timing


def extra_memory(fn: Callable[[], Any], device: torch.device) -> int:
    """Peak allocation of one call above the allocation before it (its output included)."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    out = fn()
    torch.cuda.synchronize(device)
    extra = torch.cuda.max_memory_allocated(device) - before
    del out
    return int(extra)


def time_groups(
    timer: GpuTimer, fns: dict[str, Callable], spins: dict[str, Optional[float]], groups: int, args
) -> dict:
    names = list(fns)
    n = len(names)
    out: dict[str, Any] = {"method": None, "groups": groups, "arms": {}}
    if not names:
        return out
    medians: dict[str, list[float]] = {name: [] for name in names}
    late = {name: 0 for name in names}
    for g in range(groups):
        order = names[g % n :] + names[: g % n]
        if (g // n) % 2:
            order = order[::-1]
        for name in order:
            if args.warmup:
                timer.times(fns[name], args.warmup, spins[name])
            result = timer.times(fns[name], args.iters, spins[name])
            medians[name].append(statistics.median(result["ms"]))
            late[name] += result["late"]
            out["method"] = result["method"]
    for name in names:
        meds = medians[name]
        value = statistics.median(meds)
        out["arms"][name] = {
            "us": value * 1e3,
            "group_us": [m * 1e3 for m in meds],
            "spread": (max(meds) - min(meds)) / value,
            "late": late[name],
        }
    if "cake" in medians:
        for name in names:
            if name == "cake":
                continue
            ratios = [other / cake for other, cake in zip(medians[name], medians["cake"], strict=True)]
            out["arms"][name]["ratio_to_cake"] = {
                "median": statistics.median(ratios),
                "min": min(ratios),
                "max": max(ratios),
            }
    return out


def noise_band(timed: dict) -> float:
    arms = timed["arms"]
    band = max(MIN_NOISE, arms.get("cake", {}).get("spread", 0.0))
    if "cake_aa" in arms and "ratio_to_cake" in arms["cake_aa"]:
        band = max(band, abs(1.0 - arms["cake_aa"]["ratio_to_cake"]["median"]))
    return band


def time_row(
    row: str, layout: str, table: Table, ids: torch.Tensor, lookups: dict, use: list[str], args, timer, device
) -> dict:
    positions = ids.numel()
    copy_bytes = -(-positions * BYTES_PER_ROW // 2 // 16) * 16
    copy_src = torch.empty(copy_bytes, dtype=torch.uint8, device=device) if "copy" in use else None
    copy_dst = torch.empty_like(copy_src) if copy_src is not None else None
    calls: dict[str, Callable[[], Any]] = {}
    for name in use:
        if name == "copy":
            calls[name] = lambda: copy_dst.copy_(copy_src)
        else:
            lookup = lookups[name]
            calls[name] = lambda lookup=lookup: lookup(table.weight, table.scale, ids)
    result: dict[str, Any] = {"row": row, "layout": layout, "positions": positions, "rows": table.rows, "arms": {}}
    ready = []
    for name, call in calls.items():
        try:
            call()  # compiles / loads outside the timing
            torch.cuda.synchronize(device)
            result["arms"][name] = {"extra_bytes": extra_memory(call, device)}
            ready.append(name)
        except Exception as e:  # noqa: BLE001 -- an arm that refuses a row is reported, not timed
            result["arms"][name] = {"status": "unsupported", "reason": f"{type(e).__name__}: {str(e)[:500]}"}
    result["modes"] = {}
    for mode in args.modes:
        fns: dict[str, Callable] = {}
        spins: dict[str, Optional[float]] = {}
        graphs = []
        for name in ready:
            try:
                if mode == "graph":
                    graph = capture_graph(calls[name], torch.cuda.Stream(device))
                    graphs.append(graph)
                    fns[name] = graph.replay
                    spins[name] = None
                else:
                    fns[name] = calls[name]
                    spins[name] = timer.spin_ms(calls[name])
            except Exception as e:  # noqa: BLE001 -- e.g. a call that cannot be captured
                result["arms"][name].setdefault("mode_errors", {})[mode] = f"{type(e).__name__}: {str(e)[:500]}"
        timed = time_groups(timer, fns, spins, args.groups, args)
        band = noise_band(timed)
        faster = [
            name
            for name in BASELINES
            if "ratio_to_cake" in timed["arms"].get(name, {})
            and timed["arms"][name]["ratio_to_cake"]["median"] < 1.0 - band
        ]
        if faster and args.retime_groups > args.groups:
            first = timed
            timed = time_groups(timer, fns, spins, args.retime_groups, args)
            band = noise_band(timed)
            timed["first_pass"] = {"groups": first["groups"], "faster_baselines": faster, "arms": first["arms"]}
        timed["noise_band"] = band
        cake_us = timed["arms"].get("cake", {}).get("us")
        if cake_us:
            timed["cake_tb_per_s"] = positions * BYTES_PER_ROW / (cake_us * 1e-6) / 1e12
        result["modes"][mode] = timed
        del graphs, fns
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------------------------------------------
# Identity and report


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def identity(device: torch.device, timer: GpuTimer) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(device)
    arch = cake_backend.arch_for_capability(torch.cuda.get_device_capability(device))
    out: dict[str, Any] = {
        "gpu": p.name,
        "uuid": f"GPU-{p.uuid}",
        "capability": list(torch.cuda.get_device_capability(device)),
        "arch": arch,
        "sms": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "memory_bytes": p.total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "timing": timer.method,
        "versions": {
            d: _version(d) for d in ("cake-pplx-engram-lookup", "apache-tvm-ffi", "triton", "cupti-python", "numpy")
        },
    }
    with contextlib.suppress(Exception):
        out["jit_cuda"] = ".".join(map(str, cake_build.record_cuda_version(arch)))
    query = "driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit,temperature.gpu"
    with contextlib.suppress(Exception):
        q = subprocess.run(
            ["nvidia-smi", f"--id=GPU-{p.uuid}", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        out["nvidia_smi"] = dict(zip(query.split(","), (v.strip() for v in q.stdout.strip().split(",")), strict=False))
    return out


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(report: dict) -> str:
    ident = report["identity"]
    smi = ident.get("nvidia_smi", {})
    lines = [
        f"## {ident['gpu']} ({ident['arch']}, {ident['sms']} SMs)",
        "",
        f"Driver {smi.get('driver_version', '?')}, SM clock {smi.get('clocks.sm', '?')} (max "
        f"{smi.get('clocks.max.sm', '?')}), memory clock {smi.get('clocks.mem', '?')}, power limit "
        f"{smi.get('power.limit', '?')}; L2 {ident['l2_bytes'] / 2**20:.0f} MiB; CUDA {ident['torch_cuda']} (torch), "
        f"{ident.get('jit_cuda', '?')} (JIT build); torch {ident['torch']}; Triton {ident['versions'].get('triton')}; "
        f"apache-tvm-ffi {ident['versions'].get('apache-tvm-ffi')}; timing {ident['timing']}.",
        "",
        "Bit gate: "
        + (
            ", ".join(f"`{name}` {'equal' if g['passed'] else 'NOT EQUAL'}" for name, g in report["gate"].items())
            or "not run"
        ),
        "",
        "| Row | ids | R | Layout | Mode | Cake (us) | Eager (us) | Eager / Cake | Compile (us) | Compile / Cake "
        "| Cake TB/s | Copy roof (us) | Roof / Cake | A/A | Noise |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        for mode, timed in row["modes"].items():
            arms = timed["arms"]

            def us(name, arms=arms):
                return arms.get(name, {}).get("us")

            def ratio(name, arms=arms):
                return arms.get(name, {}).get("ratio_to_cake", {}).get("median")

            compiled = [n for n in ("torch_compile", "torch_compile_max_autotune") if us(n) is not None]
            best = min(compiled, key=us) if compiled else None
            label = {None: "", "torch_compile": " (default)", "torch_compile_max_autotune": " (max-autotune)"}[best]
            lines.append(
                f"| {row['row']} | `{list(ROWS[row['row']][2])}` | {row['rows']:,} | {row['layout']} | {mode} "
                f"| {_fmt(us('cake'))} | {_fmt(us('torch_eager'))} | {_fmt(ratio('torch_eager'))} "
                f"| {_fmt(us(best) if best else None)}{label} | {_fmt(ratio(best) if best else None)} "
                f"| {_fmt(timed.get('cake_tb_per_s'))} | {_fmt(us('copy'))} | {_fmt(ratio('copy'))} "
                f"| {_fmt(ratio('cake_aa'), 3)} | {timed['noise_band'] * 100:.1f} % |"
            )
    lines += ["", "| Row | Layout | " + " | ".join(f"{n} extra MiB" for n in ARMS) + " |"]
    lines.append("|---|---|" + "---|" * len(ARMS))
    for row in report["rows"]:
        cells = []
        for name in ARMS:
            extra = row["arms"].get(name, {}).get("extra_bytes")
            cells.append("-" if extra is None else f"{extra / 2**20:.1f}")
        lines.append(f"| {row['row']} | {row['layout']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------------
# Main


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark of the Cake Engram row lookup.")
    ap.add_argument("--rows", default=",".join(ROWS))
    ap.add_argument("--layouts", default=",".join(LAYOUTS))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"comma list of {', '.join(ARMS)}")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--retime-groups", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--shard-rows", type=int, default=6_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-cupti", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()
    args.modes = [m for m in args.modes.split(",") if m]
    for mode in args.modes:
        if mode not in MODES:
            ap.error(f"unknown mode {mode!r}; the modes are {MODES}")
    rows = [r for r in args.rows.split(",") if r]
    for row in rows:
        if row not in ROWS:
            ap.error(f"unknown row {row!r}; the rows are {list(ROWS)}")
    layouts = [lay for lay in args.layouts.split(",") if lay]
    names = [a for a in args.arms.split(",") if a]
    for name in names:
        if name not in ARMS:
            ap.error(f"unknown arm {name!r}; the arms are {ARMS}")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", torch.cuda.current_device())
    timer = GpuTimer(device, use_cupti=not args.no_cupti)
    lookups = make_lookups(names)
    ident = identity(device, timer)
    print(json.dumps({"identity": ident, "arms": names}), flush=True)

    gate: dict[str, Any] = {}
    if not args.no_gate:
        cases = gate_cases(device)
        for name, lookup in lookups.items():
            t0 = time.perf_counter()
            gate[name] = run_gate(lookup, cases)
            gate[name]["seconds"] = time.perf_counter() - t0
            print(json.dumps({"gate": name, "passed": gate[name]["passed"]}), flush=True)
        del cases
        torch.cuda.empty_cache()
    timed_names = [n for n in names if not (n in COMPILE_MODES and n in gate and not gate[n]["passed"])]

    report: dict[str, Any] = {"identity": ident, "settings": vars(args) | {"rows": rows, "arms": names}, "gate": gate}
    report["rows"] = []
    tables: dict[str, dict[str, Table]] = {}
    needed = {ROWS[row][0] for row in rows}
    sizes = {"shard": args.shard_rows, "small": SMALL_ROWS, "fetched": FETCHED_ROWS}
    for kind in ("shard", "small", "fetched"):
        if kind in needed:
            tables[kind] = build_tables(sizes[kind], args.seed + len(kind), device)
    for i, row in enumerate(rows):
        kind, id_kind, shape = ROWS[row]
        ids = build_ids(id_kind, shape, sizes[kind], args.seed + 1000 + i, device)
        for layout in layouts:
            try:
                result = time_row(row, layout, tables[kind][layout], ids, lookups, timed_names, args, timer, device)
            except Exception as e:  # noqa: BLE001 -- one failing row is reported, the others still run
                result = {"row": row, "layout": layout, "error": traceback.format_exc()[-2000:], "modes": {}}
                print(f"row {row} {layout} failed: {e}", file=sys.stderr, flush=True)
            report["rows"].append(result)
            brief = {
                mode: {n: round(a["us"], 2) for n, a in t["arms"].items()} for mode, t in result["modes"].items()
            }
            print(json.dumps({"row": row, "layout": layout, "us": brief}), flush=True)
        del ids
        torch.cuda.empty_cache()
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    text = markdown({**report, "rows": [r for r in report["rows"] if "error" not in r]})
    if args.markdown:
        Path(args.markdown).write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
