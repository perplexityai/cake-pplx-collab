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

Benchmark of the Cake gated KV compression training entries (det) against ``torch.compile`` of the same pairwise
math, on one GPU, single rank.

* rows (``--rows``): ``perf``, T = 16384 tokens in 27 documents and T = 65536 in 68 (seeded log-normal lengths, the
  packs of the family's tests); ``sensitivity``, T = 16384 as one document, as the 27 documents made even (no tail
  token) and as documents of 1, 2, 3, 1, 2, 3, ... tokens (about 8192 documents); ``sweep``, T = 2048 to 262144 with
  one document per 600 tokens; ``--pack T:S[:SEED]`` adds a log-normal pack.  Each row records its lengths, ``S``,
  ``G`` and the number of odd documents, and runs in every layout of ``--layouts``: ``halves`` (``C`` and ``Z`` the
  column halves of one FP32 ``[T, 1024]`` tensor, the gradient one ``[T, 1024]``; the primary layout) and
  ``contiguous`` (two FP32 ``[T, 512]`` tensors).
* arms (``--arms``; ``cake_gated_kv_compression_baselines.py``): ``cake`` (``cake_gated_kv_compression_det_cz`` in
  the ``halves`` layout, ``cake_gated_kv_compression_det`` in ``contiguous``), ``cake_aa`` (the same entry again: the
  A-A noise band), ``compile_default`` and ``compile_max_autotune`` (``torch.compile`` of the same pairwise math, a
  custom ``autograd.Function`` with a compiled forward and a compiled token-major backward, on the same operands:
  device ``cu_seqlens``, the caller's ``G``), ``eager`` (the same functions uncompiled: the reference),
  ``compile_autograd`` (the reference forward with autograd's softmax backward: informational) and ``copy`` (a
  device-to-device copy of the bytes each phase must move: the bandwidth context).
* phases (``--phases``): ``fwd`` (the training forward, its graph recorded), ``bwd`` (the backward of a recorded
  forward), ``step`` (both) and ``ckpt`` (``torch.utils.checkpoint(..., use_reentrant=False)`` around the entry,
  then its backward: two forwards and one backward).
* modes (``--modes``): ``graph`` (the replay of a CUDA graph of one call: the kernel-level comparison, primary) and
  ``eager`` (each iteration behind a spin kernel that hides its host time, secondary).
* timing (``cake_gated_kv_compression_timer.py``, the causal conv1d benchmarks' timer): per iteration an L2 flush
  (cold L2), then the CUPTI span from the first kernel start to the last kernel end of the iteration's own launches
  (CUDA events without ``cupti-python``).  Per phase and mode, ``--groups`` groups; group ``g`` times every arm for
  its iteration count (about 25 ms of GPU time, 10 to 100 iterations) in the arm list rotated left by ``g mod n``
  (reversed when ``g // n`` is odd).  An arm's time is the median over the groups of the group's median; a ratio is
  the median over the groups of the per-group ratio, with its range.  The noise band is the largest ``|cake_aa /
  cake - 1|`` over the groups; a speedup beyond it reads ``faster`` or ``slower``, within it ``parity``.
* accuracy gate (before any timing): every computing arm, Cake's included, on draws of the row's pack (``--draws``
  plain draws, close pairs ``C[t1] = C[t0] (1 + 2^-12)`` and ``(1 + 2^-16)``, gate extremes) and once per layout on
  small edge packs: the feature request's acceptance criteria against a float64 oracle (``out`` within one BF16
  value of the float64 result rounded once, ``dC`` and ``dZ`` within relative L2 ``1e-6``, finite, tail gradient
  rows +0.0) and a second call bitwise equal to the first.  The speed comparison takes only eligible baselines: those
  that meet the criteria, or that are finite, zero on tail rows, repeatable and no farther from the oracle than Cake
  on every draw (the rule used is recorded per arm); an ineligible baseline is not timed unless
  ``--time-ineligible``.  ``cake_aa`` shares ``cake``'s verdict; a Cake arm that misses the criteria is still timed
  and is flagged.
* comparison per phase and mode: the best baseline (the fastest eligible ``compile_*`` arm), its time and Cake's,
  ``speedup = baseline / Cake`` (above 1: Cake faster) with its range, ``eager / Cake``, the copy's time and
  ``copy / Cake`` (the fraction of the copy speed Cake reaches), and the achieved TB/s of every arm for the phase's
  minimal bytes (``fwd`` ``4096 T + 1024 G``, ``bwd`` ``8192 T + 1024 G``).  Each arm's saved-for-backward bytes
  (``saved_tensors_hooks``: all, and those beyond the operands) and Cake's own memory report are recorded per row.

Output: ``--json PATH``, rewritten after every row: the environment (GPU, SMs, L2, driver, torch, Triton, CUDA,
``tvm_ffi``, the package's version and a digest of its files, Dynamo settings), the arguments, the edge-pack gate,
and per row its pack, arms, gates and timing cells.  ``--summarize PATH`` prints the comparison table of a finished
file.  ``torch.compile`` on ``sm_107a`` needs a ``ptxas`` that knows it (``TRITON_PTXAS_PATH``; recorded).

Example (on a GPU node, from ``kernels/gated_kv_compression``)::

    python benchmarks/bench_cake_gated_kv_compression_det_train.py --json gkv-b200.json
    python benchmarks/bench_cake_gated_kv_compression_det_train.py --rows perf --layouts halves --json quick.json
    python benchmarks/bench_cake_gated_kv_compression_det_train.py --summarize gkv-b200.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_pplx.cake_gated_kv_compression_det_train as cake  # noqa: E402

import cake_gated_kv_compression_baselines as baselines  # noqa: E402
import cake_gated_kv_compression_timer as timer_lib  # noqa: E402

SCHEMA = "cake-gated-kv-compression-det-train-bench/1"
CAKE_DISTRIBUTION = "cake-pplx-gated-kv-compression"
DISTRIBUTIONS = (
    CAKE_DISTRIBUTION,
    "torch",
    "triton",
    "apache-tvm-ffi",
    "cupti-python",
    "numpy",
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-cccl",
)
PHASES = ("fwd", "bwd", "step", "ckpt")
MODES = ("graph", "eager")
ARMS = ("cake", "cake_aa", "compile_default", "compile_max_autotune", "eager", "compile_autograd", "copy")
KINDS = {"cake": baselines.CAKE, "cake_aa": baselines.CAKE, **baselines.BASELINE_ARMS, "copy": baselines.CONTEXT}
ROW_SETS = ("perf", "sensitivity", "sweep")
# Each arm's iteration count: about TARGET_MS of GPU time per group, within [MIN_ITERS, MAX_ITERS].
TARGET_MS = 25.0
MIN_ITERS = 10
MAX_ITERS = 100
SWEEP_TOKENS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
SWEEP_TOKENS_PER_DOCUMENT = 600


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {message}", file=sys.stderr, flush=True)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(list(values)))


# ----------------------------------------------------------------------------------------------------------------
# Rows


def perf_packs() -> list[baselines.Pack]:
    return [
        baselines.Pack(baselines.lognormal_lengths(16384, 27, 16384), "T16384-S27"),
        baselines.Pack(baselines.lognormal_lengths(65536, 68, 65536), "T65536-S68"),
    ]


def sensitivity_packs() -> list[baselines.Pack]:
    base = baselines.lognormal_lengths(16384, 27, 16384)
    return [
        baselines.Pack((16384,), "T16384-S1"),
        baselines.Pack(baselines.even_lengths(base), "T16384-S27-even"),
        baselines.Pack(baselines.tiny_lengths(16384), "T16384-tiny"),
    ]


def sweep_packs() -> list[baselines.Pack]:
    found = []
    for T in SWEEP_TOKENS:
        S = max(1, round(T / SWEEP_TOKENS_PER_DOCUMENT))
        found.append(baselines.Pack(baselines.lognormal_lengths(T, S, T), f"sweep-T{T}-S{S}"))
    return found


def custom_pack(spec: str) -> baselines.Pack:
    parts = [int(p) for p in spec.split(":")]
    if len(parts) not in (2, 3) or parts[0] < 1 or parts[1] < 1:
        raise ValueError(f"--pack takes T:S[:SEED] with T, S >= 1, got {spec!r}")
    T, S = parts[0], parts[1]
    seed = parts[2] if len(parts) == 3 else T
    return baselines.Pack(baselines.lognormal_lengths(T, S, seed), f"T{T}-S{S}-seed{seed}")


def select_packs(rows: Sequence[str], extra: Sequence[str]) -> list[baselines.Pack]:
    found: list[baselines.Pack] = []
    makers = {"perf": perf_packs, "sensitivity": sensitivity_packs, "sweep": sweep_packs}
    for name in rows:
        found += makers[name]()
    found += [custom_pack(spec) for spec in extra]
    unique: dict[tuple[int, ...], baselines.Pack] = {}
    for pack in found:
        unique.setdefault(pack.lengths, pack)
    return list(unique.values())


# ----------------------------------------------------------------------------------------------------------------
# Arms and their phase calls


class Calls(dict):
    """``{phase: fn}`` of one arm on one row, plus ``record`` (the forward ``bwd`` replays the backward of), and the
    attributes ``release()`` (drop the recorded forward and the gradients) and ``fresh()`` (the same arm on new leaves
    of the same storage, for a graph capture)."""


def autograd_calls(layer: Callable[..., torch.Tensor], leaves: Sequence[torch.Tensor], dout: torch.Tensor) -> Calls:
    state: dict[str, torch.Tensor] = {}

    def reset() -> None:
        for leaf in leaves:
            leaf.grad = None

    def fwd():
        return layer(*leaves)

    def record():
        state["y"] = layer(*leaves)
        return state["y"]

    def bwd():
        reset()
        torch.autograd.backward([state["y"]], [dout], retain_graph=True)

    def step():
        reset()
        torch.autograd.backward([layer(*leaves)], [dout])

    def ckpt():
        reset()
        # No arm draws random numbers: the recompute needs no RNG state (and a graph capture forbids reading it).
        y = checkpoint(layer, *leaves, use_reentrant=False, preserve_rng_state=False)
        torch.autograd.backward([y], [dout])

    def release() -> None:
        state.clear()
        reset()

    calls = Calls(fwd=fwd, record=record, bwd=bwd, step=step, ckpt=ckpt)
    calls.release = release
    calls.fresh = lambda: autograd_calls(layer, [leaf.detach().requires_grad_(True) for leaf in leaves], dout)
    return calls


def copy_calls(nbytes: dict[str, int], device: torch.device) -> Calls:
    """One device-to-device copy per phase of half the phase's bytes (a copy reads and writes each byte)."""
    n = max(nbytes.values()) // 2
    src = torch.ones(n, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)

    def phase(key: str) -> Callable[[], torch.Tensor]:
        m = nbytes[key] // 2
        return lambda: dst[:m].copy_(src[:m])

    calls = Calls({key: phase(key) for key in PHASES}, record=lambda: None)
    calls.release = lambda: None
    calls.fresh = lambda: calls
    return calls


def cake_layer(layout: str, cu: torch.Tensor, rows: int) -> Callable[..., torch.Tensor]:
    if layout == "halves":
        return lambda cz: cake.cake_gated_kv_compression_det_cz(cz, cu, rows)
    return lambda c, z: cake.cake_gated_kv_compression_det(c, z, cu, rows)


def layer_maker(name: str, layout: str, args) -> Callable[[torch.Tensor, int], Callable[..., torch.Tensor]]:
    """``make(cu, G) -> layer`` of a computing arm."""
    if name in ("cake", "cake_aa"):
        return lambda cu, rows: cake_layer(layout, cu, rows)
    return lambda cu, rows: baselines.make_layer(name, layout=layout, cu=cu, rows=rows, dynamic=args.compile_dynamic)


def describe(name: str, layout: str, args) -> dict:
    if name in ("cake", "cake_aa"):
        entry = "cake_gated_kv_compression_det_cz" if layout == "halves" else "cake_gated_kv_compression_det"
        return {"implementation": f"Cake det autograd entry {entry}"}
    if name == "copy":
        return {"implementation": "device-to-device copy (torch copy_) of half of each phase's minimal bytes"}
    return baselines.describe(name, args.compile_dynamic)


# ----------------------------------------------------------------------------------------------------------------
# Timing


def group_order(arms: list[str], g: int) -> list[str]:
    n = len(arms)
    if not n:
        return []
    shift = g % n
    order = arms[shift:] + arms[:shift]
    if (g // n) % 2:
        order.reverse()
    return order


def capture(calls: Calls, phase: str, side: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    """A CUDA graph of one call of ``phase`` on new leaves whose every call runs on ``side`` (the forward ``bwd``
    replays is recorded there first).  The caller drops it after timing (one graph alive at a time)."""
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fresh = calls.fresh()
        if phase == "bwd":
            fresh["record"]()
    graph = timer_lib.capture_graph(fresh[phase], side)
    graph.calls = fresh
    return graph


def iteration_count(reply: dict) -> int:
    estimate = max(_median(reply["ms"]), 1e-3)
    return int(min(MAX_ITERS, max(MIN_ITERS, TARGET_MS / estimate)))


def time_arm(name, calls, phase, mode, spin, iters, timer, side) -> dict:
    if mode == "eager":
        fn = calls[phase]
        if name not in iters:
            iters[name] = iteration_count(timer.times(fn, 3, spin))
        return timer.times(fn, iters[name], spin)
    graph = capture(calls, phase, side)
    try:
        if name not in iters:
            iters[name] = iteration_count(timer.times(graph.replay, 3, None))
        return timer.times(graph.replay, iters[name], None)
    finally:
        del graph
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def time_cell(calls: dict[str, Calls], names: list[str], phase: str, mode: str, args, timer, side) -> dict:
    """Time ``names`` in one phase and mode, in ``--groups`` paired groups (module docstring)."""
    cell: dict = {"primary": mode == "graph", "arms": {}, "skipped": {}, "group_orders": [], "method": timer.method}
    spins: dict[str, Optional[float]] = {}
    timed: list[str] = []
    for name in names:
        try:
            if mode == "eager":
                if phase == "bwd":
                    calls[name]["record"]()
                # One untimed call first: a first call may compile (the spin length is sized from host times).
                calls[name][phase]()
                torch.cuda.synchronize()
                spins[name] = timer.spin_ms(calls[name][phase])
            else:
                graph = capture(calls[name], phase, side)
                graph.replay()
                torch.cuda.synchronize()
                del graph
            timed.append(name)
        except Exception as exc:  # noqa: BLE001 -- this arm has no such phase or mode; the others go on
            torch.cuda.synchronize()
            cell["skipped"][name] = f"{type(exc).__name__}: {str(exc)[:400]}"
        torch.cuda.empty_cache()
    iters: dict[str, int] = {}
    groups: dict[str, list[float]] = {name: [] for name in timed}
    late: dict[str, int] = {name: 0 for name in timed}
    for g in range(args.groups):
        order = group_order(timed, g)
        cell["group_orders"].append(order)
        for name in order:
            try:
                reply = time_arm(name, calls[name], phase, mode, spins.get(name), iters, timer, side)
            except Exception as exc:  # noqa: BLE001 -- the arm leaves this cell; the others go on
                torch.cuda.synchronize()
                cell["skipped"][name] = f"failed in group {g}: {type(exc).__name__}: {str(exc)[:400]}"
                timed.remove(name)
                groups.pop(name)
                continue
            groups[name].append(_median(reply["ms"]) * 1e3)
            late[name] += reply["late"]
            cell["method"] = reply["method"]
    for name in timed:
        values = groups[name]
        cell["arms"][name] = {
            "us": _median(values),
            "us_range": [min(values), max(values)],
            "groups_us": values,
            "iters": iters[name],
            "late": late[name],
            "spin_ms": spins.get(name),
        }
    for name in names:
        calls[name].release()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return cell


def paired(cell: dict, num: str, den: str) -> Optional[dict]:
    """The median over the groups of ``num / den`` (above 1: ``den`` faster) with its range."""
    a = cell["arms"].get(num, {}).get("groups_us")
    b = cell["arms"].get(den, {}).get("groups_us")
    if not a or not b or len(a) != len(b):
        return None
    ratios = [x / y for x, y in zip(a, b, strict=True)]
    return {"median": _median(ratios), "range": [min(ratios), max(ratios)], "groups": ratios}


def compare(cell: dict, row: dict, phase: str) -> dict:
    us = {name: entry["us"] for name, entry in cell["arms"].items()}
    nbytes = row["bytes"][phase]
    found: dict = {
        "bytes": nbytes,
        "tb_per_s": {name: nbytes / (value * 1e-6) / 1e12 for name, value in us.items() if value > 0},
    }
    band = None
    aa = paired(cell, "cake_aa", "cake")
    if aa is not None:
        band = max(abs(r - 1.0) for r in aa["groups"])
        found["a_a"] = {"median": aa["median"], "range": aa["range"], "noise_band": band}
    kinds = (baselines.BASELINE, baselines.INFORMATIONAL)
    candidates = [n for n in us if row["arms"][n]["kind"] in kinds and row["arms"][n].get("eligible")]
    found["ineligible_timed"] = [
        n for n in us if row["arms"][n]["kind"] in kinds and not row["arms"][n].get("eligible")
    ]
    if "cake" in us:
        found["cake_us"] = us["cake"]
        verdict = row["arms"]["cake"].get("gate")
        found["cake_gate_pass"] = verdict["pass"] if isinstance(verdict, dict) else None
        if candidates:
            best = min(candidates, key=us.__getitem__)
            speedup = paired(cell, best, "cake")
            found.update(best_baseline=best, baseline_us=us[best], speedup=speedup)
            if speedup is not None and band is not None:
                if speedup["median"] > 1.0 + band:
                    found["verdict"] = "faster"
                elif speedup["median"] < 1.0 - band:
                    found["verdict"] = "slower"
                else:
                    found["verdict"] = "parity"
        else:
            found["best_baseline"] = None
        if "eager" in us:
            found["eager_us"] = us["eager"]
            found["eager_over_cake"] = paired(cell, "eager", "cake")
        if "copy" in us:
            found["copy_us"] = us["copy"]
            found["copy_over_cake"] = paired(cell, "copy", "cake")
    return found


# ----------------------------------------------------------------------------------------------------------------
# Rows


def bench_row(pack: baselines.Pack, layout: str, args, device, timer, side, edge: dict) -> dict:
    t_row = time.time()
    T, S, G = pack.T, pack.S, pack.G
    row: dict = {
        "id": pack.label,
        "layout": layout,
        "T": T,
        "S": S,
        "G": G,
        "odd_documents": pack.odd,
        "lengths": list(pack.lengths),
        "bytes": baselines.phase_bytes(T, G),
        "arms": {},
        "not_timed": {},
        "timing": {},
    }
    supported = bool(cake.is_cake_gated_kv_compression_det_train_supported(device))
    if supported:
        row["cake_memory_report"] = cake.cake_gated_kv_compression_det_train_memory_bytes(T, S)
    x = baselines.draw(pack.lengths, seed=args.seed, device=device, layout=layout)
    leaves = baselines.leaves_of(x)
    inputs = [*leaves, x.cz, x.cu]
    calls: dict[str, Calls] = {}
    for name in args.arms:
        entry: dict = {"kind": KINDS[name], **describe(name, layout, args)}
        row["arms"][name] = entry
        try:
            if name == "copy":
                entry.update(gate="not applicable: moves the bytes, computes nothing", eligible=None)
                calls[name] = copy_calls(row["bytes"], device)
                continue
            if name in ("cake", "cake_aa") and not supported:
                raise baselines.Unsupported("the Cake package has no delivered route on this device")
            layer = layer_maker(name, layout, args)(x.cu, G)
            if name == "cake_aa":
                entry.update(
                    gate="the cake arm's verdict (the same entry)", eligible=row["arms"]["cake"].get("eligible")
                )
            elif args.draws > 0:
                t0 = time.time()
                entry["gate"] = baselines.gate(
                    layer,
                    pack.lengths,
                    layout=layout,
                    device=device,
                    draws=baselines.variants(args.draws, args.gate_seed),
                )
                entry["gate_s"] = time.time() - t0
                # Baselines are judged by the criteria, or against Cake's own measures on the same draws.
                reference = row["arms"].get("cake", {}).get("gate") if name != "cake" else None
                ok, entry["eligibility"] = baselines.eligibility(
                    entry["gate"], reference if isinstance(reference, dict) else None
                )
                edge_ok = None
                edge_verdict = edge.get(layout, {}).get(name)
                if isinstance(edge_verdict, dict) and "packs" in edge_verdict:
                    edge_cake = edge.get(layout, {}).get("cake") if name != "cake" else None
                    edge_ok, entry["edge_eligibility"] = baselines.edge_eligibility(
                        edge_verdict, edge_cake if isinstance(edge_cake, dict) else None
                    )
                entry["eligible"] = bool(ok and edge_ok is not False)
            else:
                entry.update(gate="not run (--draws 0)", eligible=False)
            entry["saved"] = baselines.saved_bytes(layer, leaves, inputs)
            calls[name] = autograd_calls(layer, [leaf.detach().requires_grad_(True) for leaf in leaves], x.dout)
        except Exception as exc:  # noqa: BLE001 -- the arm is reported, the row goes on
            entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            traceback.print_exc()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    row["gate_s"] = time.time() - t_row
    always = (baselines.CAKE, baselines.REFERENCE, baselines.CONTEXT)
    names = []
    for name in calls:
        if KINDS[name] in always or row["arms"][name].get("eligible") or args.time_ineligible:
            names.append(name)
        else:
            row["not_timed"][name] = "failed the accuracy gate (--time-ineligible times it anyway)"
    t_timing = time.time()
    for mode in args.modes:
        for phase in args.phases:
            log(f"{pack.label} {layout} {phase}/{mode}: {', '.join(names)}")
            try:
                cell = time_cell(calls, names, phase, mode, args, timer, side)
                cell["comparison"] = compare(cell, row, phase)
            except Exception as exc:  # noqa: BLE001 -- keep the other cells
                torch.cuda.synchronize(device)
                cell = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
                traceback.print_exc()
            row["timing"][f"{phase}/{mode}"] = cell
    row["timing_s"] = time.time() - t_timing
    calls.clear()
    del x, leaves, inputs
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return row


def run_edge_gate(args, device) -> dict:
    """The edge-pack gate of every computing arm in every layout (once; it does not depend on the row)."""
    found: dict = {}
    supported = bool(cake.is_cake_gated_kv_compression_det_train_supported(device))
    for layout in args.layouts:
        for name in args.arms:
            if name in ("cake_aa", "copy"):
                continue
            if name == "cake" and not supported:
                found.setdefault(layout, {})[name] = {"pass": None, "unsupported": "no delivered Cake route"}
                continue
            log(f"edge-pack gate: {name} {layout}")
            t0 = time.time()
            verdict = baselines.edge_gate(layer_maker(name, layout, args), layout=layout, device=device)
            verdict["seconds"] = time.time() - t0
            found.setdefault(layout, {})[name] = verdict
    return found


# ----------------------------------------------------------------------------------------------------------------
# Environment and output


def _version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def package_identity() -> dict:
    """The imported gated KV compression package: its location, its files and a digest of them."""
    root = Path(cake.__file__).resolve().parent
    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and any("gated_kv_compression" in part for part in p.relative_to(root).parts)
    )
    digest = hashlib.sha256()
    for p in files:
        digest.update(str(p.relative_to(root)).encode())
        digest.update(hashlib.sha256(p.read_bytes()).digest())
    return {
        "distribution": CAKE_DISTRIBUTION,
        "version": _version(CAKE_DISTRIBUTION),
        "location": str(root),
        "files": len(files),
        "sha256": digest.hexdigest(),
    }


def driver_version() -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {type(exc).__name__}"
    values = sorted({line.strip() for line in done.stdout.splitlines() if line.strip()})
    return ",".join(values) if values else f"unavailable: nvidia-smi exit {done.returncode}"


def environment(device: torch.device, timer, dynamo: dict) -> dict:
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": [major, minor],
        "arch": f"sm_{major}{minor}a",
        "sm_count": props.multi_processor_count,
        "l2_bytes": props.L2_cache_size,
        "memory_bytes": props.total_memory,
        "driver": driver_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_ptxas_path": os.environ.get("TRITON_PTXAS_PATH"),
        "distributions": {name: _version(name) for name in DISTRIBUTIONS},
        "cake_supported": bool(cake.is_cake_gated_kv_compression_det_train_supported(device)),
        "cake_package": package_identity(),
        "gpu_timer": timer.method,
        "dynamo": dynamo,
    }


def write_json(path: str, result: dict) -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def _us(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _ratio(r: Optional[dict]) -> str:
    return "-" if not r else f"{r['median']:.3f} [{r['range'][0]:.3f}, {r['range'][1]:.3f}]"


def table(result: dict) -> list[str]:
    """The comparison table of every finished row (graph rows first)."""
    lines = [
        "| row | layout | phase | mode | Cake us | best baseline | baseline us | baseline / Cake | verdict "
        "| noise band | eager us | copy us | copy / Cake | Cake TB/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result.get("rows", []):
        if "error" in row:
            lines.append(f"| {row['id']} | {row['layout']} | error: {row['error'][:80]} |")
            continue
        cells = sorted(row["timing"].items(), key=lambda kv: (kv[0].split("/")[1] != "graph", kv[0]))
        for key, cell in cells:
            phase, mode = key.split("/")
            c = cell.get("comparison")
            if c is None:
                lines.append(f"| {row['id']} | {row['layout']} | {phase} | {mode} | {cell.get('error', '-')[:80]} |")
                continue
            band = c.get("a_a", {}).get("noise_band")
            gate_note = " (Cake FAILED the gate)" if c.get("cake_gate_pass") is False else ""
            tbps = c["tb_per_s"].get("cake")
            lines.append(
                f"| {row['id']} | {row['layout']} | {phase} | {mode} | {_us(c.get('cake_us'))}{gate_note} "
                f"| {c.get('best_baseline') or '-'} | {_us(c.get('baseline_us'))} | {_ratio(c.get('speedup'))} "
                f"| {c.get('verdict', '-')} | {'-' if band is None else f'{band:.3f}'} | {_us(c.get('eager_us'))} "
                f"| {_us(c.get('copy_us'))} | {_ratio(c.get('copy_over_cake'))} "
                f"| {'-' if tbps is None else f'{tbps:.2f}'} |"
            )
    return lines


def _list(value: str, known: Sequence[str], what: str) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in items if v not in known]
    if unknown or not items:
        raise argparse.ArgumentTypeError(f"{what}: unknown {unknown} (known: {', '.join(known)})")
    return [k for k in known if k in items]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cake gated KV compression det training benchmark (see the module docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json", help="the output file, rewritten after every row")
    parser.add_argument("--summarize", nargs="+", metavar="JSON", help="print the table of finished files and exit")
    parser.add_argument("--rows", default="perf,sensitivity", help=f"comma list of {', '.join(ROW_SETS)}")
    parser.add_argument("--pack", action="append", default=[], metavar="T:S[:SEED]", help="an extra log-normal pack")
    parser.add_argument("--layouts", default="halves,contiguous", help="comma list of halves, contiguous")
    parser.add_argument("--phases", default="fwd,bwd,ckpt", help=f"comma list of {', '.join(PHASES)}")
    parser.add_argument("--modes", default="graph,eager", help="comma list of graph, eager")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"comma list of {', '.join(ARMS)}")
    parser.add_argument("--groups", type=int, default=7, help="paired timing groups per phase and mode")
    parser.add_argument("--draws", type=int, default=2, help="plain gate draws per row (0: no gate, no baseline)")
    parser.add_argument("--gate-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="the seed of the timed inputs")
    parser.add_argument("--no-edge-gate", action="store_true", help="skip the edge-pack gate")
    parser.add_argument("--time-ineligible", action="store_true", help="also time baselines that fail the gate")
    parser.add_argument("--compile-dynamic", action="store_true", help="torch.compile(dynamic=True)")
    parser.add_argument("--no-cupti", action="store_true", help="time with CUDA events")
    args = parser.parse_args(argv)

    if args.summarize:
        for path in args.summarize:
            result = json.loads(Path(path).read_text())
            env = result.get("env", {})
            print(f"# {path}: {env.get('device')} ({env.get('arch')}, {env.get('sm_count')} SMs), torch "
                  f"{env.get('torch')}, CUDA {env.get('torch_cuda')}, timer {env.get('gpu_timer')}")  # fmt: skip
            print("\n".join(table(result)))
        return 0
    if not args.json:
        parser.error("--json is required unless --summarize")
    try:
        args.rows = _list(args.rows, ROW_SETS, "--rows") if args.rows else []
        args.layouts = _list(args.layouts, baselines.LAYOUTS, "--layouts")
        args.phases = _list(args.phases, PHASES, "--phases")
        args.modes = _list(args.modes, MODES, "--modes")
        args.arms = _list(args.arms, ARMS, "--arms")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if "cake_aa" in args.arms and "cake" not in args.arms:
        parser.error("--arms cake_aa needs cake")
    if args.groups < 1:
        parser.error("--groups must be at least 1")
    if not torch.cuda.is_available():
        print("this benchmark needs a CUDA device", file=sys.stderr)
        return 2

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    dynamo = baselines.configure_dynamo()
    timer = timer_lib.GpuTimer(device, use_cupti=not args.no_cupti)
    side = torch.cuda.Stream(device)
    packs = select_packs(args.rows, args.pack)
    result: dict = {
        "schema": SCHEMA,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "args": vars(args),
        "env": environment(device, timer, dynamo),
        "packs": [{"id": p.label, "T": p.T, "S": p.S, "G": p.G, "odd_documents": p.odd} for p in packs],
        "edge_gate": {},
        "rows": [],
    }
    write_json(args.json, result)
    log(f"{result['env']['device']} {result['env']['arch']}: {len(packs)} packs x {len(args.layouts)} layouts")
    if args.draws > 0 and not args.no_edge_gate:
        result["edge_gate"] = run_edge_gate(args, device)
        write_json(args.json, result)
    for pack in packs:
        for layout in args.layouts:
            log(f"row {pack.label} ({layout}): T {pack.T}, S {pack.S}, G {pack.G}, odd documents {pack.odd}")
            try:
                row = bench_row(pack, layout, args, device, timer, side, result["edge_gate"])
            except Exception as exc:  # noqa: BLE001 -- e.g. out of memory: the next rows go on
                traceback.print_exc()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                row = {"id": pack.label, "layout": layout, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            result["rows"].append(row)
            write_json(args.json, result)
            print("\n".join(table({"rows": [row]})[2:]), flush=True)
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.json, result)
    print("\n".join(table(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
