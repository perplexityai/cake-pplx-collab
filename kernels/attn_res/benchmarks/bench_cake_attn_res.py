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

Benchmark of the Cake attention residual (``cake_pplx.cake_attn_res_{det,nondet}_train``, issue #262) against the
same-contract baselines, with the issue's accuracy bar per arm.

Arms (each called as ``fn(values, w)`` through autograd: ``values`` ``n`` separate BF16 ``[T, 7168]`` leaves, ``w``
an FP32 ``[7168]`` leaf):

* ``eager``: the issue's math as released (``released``: the values stacked, ``rsqrt`` of the mean square, the
  softmax over the scores, a batched matmul) in eager PyTorch FP32 on the BF16 values (``v.float()``), one BF16
  rounding of ``out``; its backward is PyTorch autograd (the value gradients rounded once to BF16 by the casts).
* ``torch_compile``: ``torch.compile(fullgraph=True, dynamic=False)`` of the same function.
* ``cake_nondet`` / ``cake_det``: ``cake_pplx.cake_attn_res_<mode>_train.cake_attn_res_<mode>``.

The cuDNN frontend (NVIDIA/cudnn-frontend main) has no operation with these semantics (a softmax over per-value RMS-
scaled dot products followed by a weighted sum of the separate values), so the baselines are the two PyTorch arms.

Timing boundary (identical for all arms): ``fwd`` = one training forward ``fn(values, w)`` (output and saved-state
allocation, autograd graph recorded and dropped); ``bwd`` = ``torch.autograd.backward`` of a recorded forward
(``retain_graph=True``; the leaves' gradients reset to ``None`` first, so gradient allocation is inside); ``ckpt`` =
``torch.utils.checkpoint`` (non-reentrant) around the forward, then its backward: two forwards and one backward, the
issue's training step.  Each phase runs eagerly (``eager`` mode: CUPTI span from the first kernel start to the last
kernel end of the iteration, host launch gaps included) and as one CUDA graph replay (``graph`` mode).  Cold L2 (a
2 x L2 buffer zeroed outside the span before every iteration), ``--dry`` untimed iterations, ``--repeat`` timed
iterations per group, ``--groups`` groups with the arm order rotated per group and reversed every other group; the
latency is the median of the group medians, the spread the min / max of all samples and the p10 / p90.

Bytes moved (the issue's yardstick traffic, BF16 ``[T, 7168]`` tensors only): ``fwd`` reads the ``n`` values and
writes ``out``, ``(n + 1) T H 2``; ``bwd`` reads ``dout`` and the values and writes every ``dv_j``, ``(2 n + 1) T H 2``;
``ckpt`` both, ``(3 n + 2) T H 2`` (the ``[T, n]`` state and ``w`` add under 0.1%).  Achieved GB/s = bytes / time
(decimal), and the HBM utilization divides it by the device's NVML memory bandwidth (``get_gpu_memory_bandwidth``:
``nvmlDeviceGetMemoryBusWidth`` times ``nvmlDeviceGetClockInfo(NVML_CLOCK_MEM)`` times 2 / 8 / 1000 GB/s, cached per
device), the NVML handle found by the device's UUID; the run records the ``CUDA_VISIBLE_DEVICES`` to NVML index map.

Accuracy bar (the issue's, as the tests apply it): on ``--draws`` draws per ``n`` (``T = 2048``, values ``randn`` times
``linspace(0.5, 4, n)``, ``w = 0.05 randn``), the error of every arm's ``out``, ``dv_j`` and ``dw`` against an FP64
evaluation in the L2 and the max norm; ``out`` and ``dv_j`` meet the bar when the median and the largest over the draws
are at most 1.005 times those of the naive BF16 evaluation, ``dw`` when they are at most 1.005 times those of the
issue's FP32 sum on the CPU.  A baseline that misses the bar is reported as a diagnostic, not as a speedup reference.

Subcommands: ``run`` (all rows on one GPU, JSON) and ``report`` (JSON to markdown, with the speedup of each Cake arm
over the faster baseline arm that meets the bar).

Example (one GPU)::

    python bench_cake_attn_res.py run --json gb300.json
    python bench_cake_attn_res.py report --json gb300.json --md gb300.md --title "GB300"
"""

from __future__ import annotations

import argparse
import bisect
import collections
import functools
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import statistics
import sys
import time
import traceback
import types
import warnings
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

H = 7168
EPS = 1e-5
CAKE_ARMS = ("cake_nondet", "cake_det")
BASELINE_ARMS = ("torch_compile", "eager")
DEFAULT_ARMS = CAKE_ARMS + BASELINE_ARMS
PHASES = ("fwd", "bwd", "ckpt")
MODES = ("eager", "graph")
NEAR_TIE = 1.005


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def _tail(exc: BaseException, n: int = 400) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= n else text[: n // 2] + " ... " + text[-n // 2 :]


def _dist(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class Unsupported(Exception):
    """The arm cannot run the row / phase; the message is the concrete reason."""


# ---------------------------------------------------------------------------------------------------------------------
# GPU timer (kernels/engram_gate/benchmarks/bench_cake_engram_gate.py, unchanged)
# ---------------------------------------------------------------------------------------------------------------------


def bench_gpu_time(
    fn,
    dry_run_iters=3,
    repeat_iters=20,
    enable_cupti=True,
    cold_l2_cache=True,
    use_cuda_graph=False,
    capture_stream=None,
):
    """Per-iteration GPU times (ms) of ``fn``: the CUPTI span from the first start to the last end of the kernels,
    memsets and copies the iteration launches (matched by correlation id); cold L2 by zeroing 2 x L2 before every
    iteration (outside the span); ``use_cuda_graph`` captures ``fn`` once on ``capture_stream`` and times replays.
    Without CUPTI it warns and falls back to CUDA events."""
    device = torch.device("cuda", torch.cuda.current_device())
    flush = None
    if cold_l2_cache:
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        flush = torch.empty(max(2 * l2_bytes, 1), dtype=torch.int8, device=device)
    run = fn
    graph = None
    if use_cuda_graph:
        side = capture_stream if capture_stream is not None else torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            fn()
        run = graph.replay
    try:
        for _ in range(dry_run_iters):
            if flush is not None:
                flush.zero_()
            run()
        torch.cuda.synchronize(device)
        if enable_cupti:
            try:
                from cupti import cupti
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"CUPTI is unavailable ({type(exc).__name__}: {exc}). Falling back to CUDA events.", stacklevel=2
                )
            else:
                return _cupti_times(cupti, run, flush, repeat_iters, device)
        return _event_times(run, flush, repeat_iters, device)
    finally:
        del run
        if graph is not None:
            graph.reset()
            del graph


def _cupti_times(cupti, run, flush, iters, device):
    launches = []
    spans = collections.defaultdict(list)
    gpu_kinds = (cupti.ActivityKind.CONCURRENT_KERNEL, cupti.ActivityKind.MEMSET, cupti.ActivityKind.MEMCPY)
    api_kinds = (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER)

    def requested():
        return 8 * 1024 * 1024, 0

    def completed(activities):
        for activity in activities:
            if activity.kind in gpu_kinds:
                spans[activity.correlation_id].append((activity.start, activity.end))
            elif activity.kind in api_kinds:
                launches.append((activity.start, activity.correlation_id))

    kinds = api_kinds + gpu_kinds
    for kind in kinds:
        cupti.activity_enable(kind)
    cupti.activity_register_callbacks(requested, completed)
    stamps = []
    try:
        for _ in range(iters):
            if flush is not None:
                flush.zero_()
            torch.cuda.synchronize(device)
            begin = cupti.get_timestamp()
            run()
            end = cupti.get_timestamp()
            stamps.append((begin, end))
        torch.cuda.synchronize(device)
        cupti.activity_flush_all(0)
    finally:
        for kind in kinds:
            cupti.activity_disable(kind)
        cupti.finalize()
    launches.sort()
    starts = [start for start, _ in launches]
    times = []
    for begin, end in stamps:
        lo, hi = bisect.bisect_left(starts, begin), bisect.bisect_right(starts, end)
        correlation = {corr for _, corr in launches[lo:hi]}
        found = [span for corr in correlation for span in spans.get(corr, ())]
        if not found:
            raise RuntimeError("CUPTI recorded no GPU activity for a timed iteration")
        times.append((max(e for _, e in found) - min(s for s, _ in found)) / 1e6)
    return times


def _event_times(run, flush, iters, device):
    pairs = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        stop.record()
        pairs.append((start, stop))
    torch.cuda.synchronize(device)
    return [start.elapsed_time(stop) for start, stop in pairs]


def timer_kind() -> str:
    try:
        from cupti import cupti  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return f"cuda_events (cupti unavailable: {type(exc).__name__}: {exc})"
    return f"cupti (cupti-python {_dist('cupti-python')})"


# ---------------------------------------------------------------------------------------------------------------------
# HBM bandwidth (NVML)
# ---------------------------------------------------------------------------------------------------------------------


def _nvml_handle(pynvml, device: torch.device):
    """The NVML handle of a CUDA device, found by its UUID (``CUDA_VISIBLE_DEVICES`` reorders the CUDA indices)."""
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    return pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")


@functools.cache
def get_gpu_memory_bandwidth(device: torch.device) -> float:
    """The device's HBM bandwidth in decimal GB/s from NVML: memory clock (MHz) times bus width (bits) times 2 (double
    data rate) / 8 / 1000.  Cached per device; a device that is not a CUDA device is a ``ValueError``."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the HBM bandwidth needs a CUDA device, got {device}")
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = _nvml_handle(pynvml, torch.device("cuda", device.index or 0))
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        return (mem_clock * bus_width * 2) / 8 / 1000
    finally:
        pynvml.nvmlShutdown()


def nvml_map() -> dict:
    """``CUDA_VISIBLE_DEVICES`` and, per CUDA index, the GPU UUID, the NVML index and PCI bus id, the memory clock,
    the bus width and the NVML bandwidth."""
    found = {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "devices": []}
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for i in range(torch.cuda.device_count()):
                handle = _nvml_handle(pynvml, torch.device("cuda", i))
                found["devices"].append(
                    {
                        "cuda_index": i,
                        "uuid": pynvml.nvmlDeviceGetUUID(handle),
                        "nvml_index": pynvml.nvmlDeviceGetIndex(handle),
                        "pci_bus_id": pynvml.nvmlDeviceGetPciInfo(handle).busId,
                        "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                        "bus_width_bits": pynvml.nvmlDeviceGetMemoryBusWidth(handle),
                        "nvml_bandwidth_gbs": get_gpu_memory_bandwidth(torch.device("cuda", i)),
                    }
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001
        found["error"] = _tail(exc, 600)
    return found


def bytes_moved(phase: str, T: int, n: int) -> int:
    """The issue's yardstick traffic of a phase (module docstring)."""
    per = T * H * 2
    return {"fwd": (n + 1) * per, "bwd": (2 * n + 1) * per, "ckpt": (3 * n + 2) * per}[phase]


# ---------------------------------------------------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------------------------------------------------


def released(values, norm_w, proj_w, eps: float = EPS):
    """_apply_attn_res with the blocks and the running sum stacked, as released (the issue's reference)."""
    v = torch.stack(values, dim=1)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    p = (k * (norm_w * proj_w)).sum(-1).softmax(-1)
    return torch.matmul(p.unsqueeze(1), v).squeeze(1)


def attn_res_fp32(values, w):
    """The issue's math in FP32 on the BF16 values, one BF16 rounding of ``out``."""
    return released([v.float() for v in values], w, torch.ones_like(w)).bfloat16()


def _fresh_function(fn, name):
    """A copy of ``fn`` with its own code object (dynamo keeps its compile cache per code object)."""
    code = fn.__code__.replace(co_name=name)
    return types.FunctionType(code, fn.__globals__, name, fn.__defaults__, fn.__closure__)


class Arm:
    name = ""

    def library(self) -> str:
        return f"torch {torch.__version__}"

    def layer(self):
        raise NotImplementedError


class EagerArm(Arm):
    name = "eager"

    def layer(self):
        return attn_res_fp32


class CompileArm(Arm):
    name = "torch_compile"

    def __init__(self):
        self.compiled = torch.compile(
            _fresh_function(attn_res_fp32, "attn_res_compiled"), fullgraph=True, dynamic=False
        )

    def library(self) -> str:
        import triton

        return f"torch {torch.__version__} (inductor, triton {triton.__version__}), fullgraph=True, dynamic=False"

    def layer(self):
        return self.compiled


class CakeArm(Arm):
    def __init__(self, mode):
        self.name = f"cake_{mode}"
        module = f"cake_pplx.cake_attn_res_{mode}_train"
        try:
            self.mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            raise Unsupported(f"{module} not importable: {type(exc).__name__}: {exc}") from exc
        self.entry = getattr(self.mod, f"cake_attn_res_{mode}")
        if torch.cuda.is_available() and not getattr(self.mod, f"is_cake_attn_res_{mode}_train_supported")():
            raise Unsupported(f"{module} does not serve cc {torch.cuda.get_device_capability()}")

    def library(self) -> str:
        return f"cake-pplx-attn-res {_dist('cake-pplx-attn-res')} ({self.mod.__name__})"

    def layer(self):
        entry = self.entry

        def fn(values, w):
            return entry(values, w, eps=EPS)

        return fn


def make_arm(name):
    if name == "eager":
        return EagerArm()
    if name == "torch_compile":
        return CompileArm()
    if name in CAKE_ARMS:
        return CakeArm(name.split("_", 1)[1])
    raise ValueError(f"unknown arm {name}")


def _bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


# ---------------------------------------------------------------------------------------------------------------------
# Accuracy bar
# ---------------------------------------------------------------------------------------------------------------------


def _outputs(fn, values, w, dout):
    leaves = [v.detach().clone().requires_grad_(True) for v in values]
    wl = w.detach().clone().requires_grad_(True)
    out = fn(leaves, wl)
    out.backward(dout)
    return {"out": out.detach(), **{f"dv{j}": v.grad for j, v in enumerate(leaves)}, "dw": wl.grad}


def _reference(values, w, dout, dtype):
    """out, dv_j and dw of the issue's math in ``dtype`` (float64: the FP64 evaluation; bfloat16: the naive BF16
    evaluation, the FP32 ``w`` kept FP32); each returned in ``dtype``."""
    vs = [v.to(dtype) for v in values]
    wd = w if dtype == torch.bfloat16 else w.to(dtype)
    r = torch.stack([torch.rsqrt(v.square().mean(-1) + EPS) for v in vs], 1)
    d = torch.stack([(v * wd).sum(-1) for v in vs], 1)
    p = (r * d).softmax(1)
    g = dout.to(dtype)
    dp = torch.stack([(g * v).sum(-1) for v in vs], 1)
    ds = p * (dp - (p * dp).sum(1, keepdim=True))
    a, b = ds * r, ds * d * r.pow(3) / H
    found = {"out": sum(p[:, j : j + 1] * v for j, v in enumerate(vs)).to(dtype)}
    for j, v in enumerate(vs):
        found[f"dv{j}"] = (p[:, j : j + 1] * g + a[:, j : j + 1] * wd - b[:, j : j + 1] * v).to(dtype)
    found["dw"] = sum((a[:, j : j + 1] * v).sum(0) for j, v in enumerate(vs)).to(dtype)
    return found


def _issue_fp32_dw(values, w, dout):
    """The issue's FP32 ``dw`` (its written backward in FP32 on the CPU)."""
    vs = [v.cpu().float() for v in values]
    w32, g = w.cpu().float(), dout.cpu().float()
    r = torch.stack([torch.rsqrt(v.pow(2).mean(-1) + EPS) for v in vs], 1)
    dot = torch.stack([v @ w32 for v in vs], 1)
    p = (r * dot).softmax(1)
    dp = torch.stack([(g * v).sum(-1) for v in vs], 1)
    ds = p * (dp - (p * dp).sum(1, keepdim=True))
    return sum(v.t() @ (ds * r)[:, j] for j, v in enumerate(vs))


def _errors(actual, expected):
    diff = (actual.to(expected.device, torch.float64) - expected).abs()
    return {"l2": diff.norm().item(), "max": diff.max().item(), "ref_l2": expected.norm().item()}


def accuracy_unit(arm_names, n, draws, device):
    """Per arm and output: the median and the largest error over the draws in the L2 and the max norm, beside the bar
    (the naive BF16 evaluation for ``out`` and ``dv_j``, the issue's FP32 sum for ``dw``), and the verdict."""
    T = 2048
    arms = {}
    for name in arm_names:
        try:
            arms[name] = make_arm(name).layer()
        except Unsupported as exc:
            log(f"accuracy {name}: unsupported: {exc}")
    errs = {name: collections.defaultdict(list) for name in [*arms, "bar"]}
    for draw in range(draws):
        g = torch.Generator(device=device).manual_seed(1000 * n + draw)
        scales = torch.linspace(0.5, 4.0, n).tolist()
        values = [(torch.randn(T, H, generator=g, device=device) * c).bfloat16() for c in scales]
        w = 0.05 * torch.randn(H, generator=g, device=device)
        dout = torch.randn(T, H, generator=g, device=device).bfloat16()
        exact = _reference(values, w, dout, torch.float64)
        tier = _reference(values, w, dout, torch.bfloat16)
        tier["dw"] = _issue_fp32_dw(values, w, dout)
        for key, t in tier.items():
            for norm, e in _errors(t.float() if key != "dw" else t, exact[key]).items():
                errs["bar"][(key, norm)].append(e)
        for name, fn in arms.items():
            got = _outputs(fn, values, w, dout)
            for key, t in got.items():
                for norm, e in _errors(t, exact[key]).items():
                    errs[name][(key, norm)].append(e)
    records = []
    keys = ["out", *(f"dv{j}" for j in range(n)), "dw"]
    for name in arms:
        rows, ok = {}, True
        for key in keys:
            row = {}
            for norm in ("l2", "max"):
                ours, bar = errs[name][(key, norm)], errs["bar"][(key, norm)]
                row[norm] = {"ours": [statistics.median(ours), max(ours)], "bar": [statistics.median(bar), max(bar)]}
                ok &= row[norm]["ours"][0] <= NEAR_TIE * row[norm]["bar"][0]
                ok &= row[norm]["ours"][1] <= NEAR_TIE * row[norm]["bar"][1]
            refs = errs[name][(key, "ref_l2")]
            row["rel_l2_median"] = statistics.median(e / r for e, r in zip(errs[name][(key, "l2")], refs, strict=True))
            rows[key] = row
        records.append({"kind": "accuracy", "arm": name, "n": n, "T": T, "draws": draws, "meets_bar": ok, "rows": rows})
        log(f"accuracy n={n} {name}: meets_bar={ok}")
    return records


# ---------------------------------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------------------------------


def make_inputs(T, n, device, seed=1234):
    g = torch.Generator(device=device).manual_seed(seed + T + 100 * n)
    scales = torch.linspace(0.5, 4.0, n).tolist()
    values = [(torch.randn(T, H, generator=g, device=device) * c).bfloat16().requires_grad_(True) for c in scales]
    w = (0.05 * torch.randn(H, generator=g, device=device)).requires_grad_(True)
    dout = torch.randn(T, H, generator=g, device=device).bfloat16()
    return values, w, dout


def _reset(leaves):
    for leaf in leaves:
        leaf.grad = None


def phase_fn(lay, values, w, dout, phase, record_stream):
    """(fn, release) of a timed phase.  ``bwd`` records its forward on ``record_stream`` (the capture stream in graph
    mode, so the backward runs on it)."""
    leaves = [*values, w]
    state = {}
    if phase == "fwd":
        return (lambda: lay(values, w)), state.clear
    if phase == "bwd":
        if record_stream is not None:
            record_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(record_stream):
                state["y"] = lay(values, w)
            torch.cuda.current_stream().wait_stream(record_stream)
        else:
            state["y"] = lay(values, w)
        torch.cuda.synchronize()

        def bwd():
            _reset(leaves)
            torch.autograd.backward([state["y"]], [dout], retain_graph=True)

        return bwd, state.clear
    if phase == "ckpt":

        def ckpt():
            _reset(leaves)
            y = checkpoint(lay, values, w, use_reentrant=False, preserve_rng_state=False)
            torch.autograd.backward([y], [dout])

        return ckpt, state.clear
    raise ValueError(phase)


def timing_unit(arm_names, T, n, args, device, cap_stream):
    row = {"T": T, "n": n}
    log(f"row T={T} n={n}: inputs")
    values, w, dout = make_inputs(T, n, device)
    leaves = [*values, w]
    torch.cuda.synchronize()
    records, lays, broken, meta = [], {}, {}, {}
    for name in arm_names:
        base = {"kind": "timing", "arm": name, **row}
        try:
            arm = make_arm(name)
            base["library"] = arm.library()
            lay = arm.layer()
            t0 = time.time()
            _reset(leaves)
            torch.cuda.synchronize()
            m0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = lay(values, w)
            torch.cuda.synchronize()
            m1 = torch.cuda.memory_allocated()
            out_bytes = y.numel() * y.element_size()
            mem = {"out_bytes": out_bytes, "fwd_retained_bytes": m1 - m0 - out_bytes}
            mem["fwd_peak_extra_bytes"] = torch.cuda.max_memory_allocated() - m0
            torch.cuda.reset_peak_memory_stats()
            torch.autograd.backward([y], [dout])
            torch.cuda.synchronize()
            grads = sum(leaf.grad.numel() * leaf.grad.element_size() for leaf in leaves)
            mem["grads_bytes"] = grads
            mem["bwd_peak_extra_bytes"] = torch.cuda.max_memory_allocated() - m1 - grads
            del y
            if args.r2r:
                outs = []
                for _ in range(2):
                    _reset(leaves)
                    y = lay(values, w)
                    torch.autograd.backward([y], [dout])
                    outs.append([y.detach()] + [leaf.grad for leaf in leaves])
                    del y
                torch.cuda.synchronize()
                base["run_to_run_bitwise"] = all(_bitwise_equal(a, b) for a, b in zip(*outs, strict=True))
                del outs
            base["warm_s"] = round(time.time() - t0, 1)
            base["memory"] = mem
            _reset(leaves)
            torch.cuda.empty_cache()
            lays[name] = lay
            log(f"  {name}: warm {base['warm_s']}s mem {mem} r2r={base.get('run_to_run_bitwise')}")
        except Unsupported as exc:
            broken[name] = ("unsupported", str(exc))
            log(f"  {name}: unsupported: {exc}")
        except Exception as exc:  # noqa: BLE001
            broken[name] = ("error", _tail(exc, 1200))
            base["traceback"] = traceback.format_exc()[-3000:]
            log(f"  {name}: ERROR at warm: {_tail(exc)}")
            _reset(leaves)
            torch.cuda.empty_cache()
        meta[name] = base
    bandwidth = get_gpu_memory_bandwidth(device)
    for mode in args.modes:
        for phase in args.phases:
            samples = {n_: [] for n_ in arm_names}
            gmeds = {n_: [] for n_ in arm_names}
            failed = dict(broken)
            for gi in range(args.groups):
                order = [a for a in arm_names if a not in failed]
                if order:
                    k = gi % len(order)
                    order = order[k:] + order[:k]
                    if gi % 2:
                        order = order[::-1]
                for name in order:
                    if name in failed:
                        continue
                    fn, release = None, None
                    try:
                        record = cap_stream if mode == "graph" else None
                        fn, release = phase_fn(lays[name], values, w, dout, phase, record)
                        times = bench_gpu_time(
                            fn,
                            dry_run_iters=args.dry,
                            repeat_iters=args.repeat,
                            use_cuda_graph=(mode == "graph"),
                            capture_stream=cap_stream,
                        )
                        samples[name].extend(times)
                        gmeds[name].append(statistics.median(times))
                    except Exception as exc:  # noqa: BLE001
                        failed[name] = ("error", _tail(exc, 1200))
                        log(f"  {name} {mode}/{phase}: ERROR {_tail(exc)}")
                        try:
                            torch.cuda.synchronize()
                        except Exception as exc2:  # noqa: BLE001
                            log(f"  sync after error failed: {_tail(exc2)}")
                    finally:
                        if release is not None:
                            release()
                        _reset(leaves)
                        del fn
                        torch.cuda.empty_cache()
            for name in arm_names:
                rec = {"kind": "timing", "arm": name, **row, "mode": mode, "phase": phase}
                rec["library"] = meta[name].get("library")
                if name in failed:
                    rec[failed[name][0]] = failed[name][1]
                elif samples[name]:
                    xs = sorted(samples[name])

                    def q(p, xs=xs):
                        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]

                    median_ms = statistics.median(gmeds[name])
                    moved = bytes_moved(phase, T, n)
                    rec.update(
                        median_us=1e3 * median_ms,
                        min_us=1e3 * xs[0],
                        max_us=1e3 * xs[-1],
                        p10_us=1e3 * q(0.10),
                        p90_us=1e3 * q(0.90),
                        samples=len(xs),
                        group_medians_us=[round(1e3 * v, 3) for v in gmeds[name]],
                        bytes=moved,
                        gbs=moved / (median_ms * 1e-3) / 1e9,
                        nvml_bandwidth_gbs=bandwidth,
                    )
                    rec["hbm_utilization"] = rec["gbs"] / bandwidth
                    for key in ("memory", "run_to_run_bitwise", "warm_s"):
                        if key in meta[name]:
                            rec[key] = meta[name][key]
                records.append(rec)
                if "median_us" in rec:
                    log(
                        f"  {mode}/{phase} {name}: {rec['median_us']:.1f} us ({rec['gbs']:.0f} GB/s, "
                        f"{100 * rec['hbm_utilization']:.1f}% of NVML) p10 {rec['p10_us']:.1f} p90 {rec['p90_us']:.1f}"
                    )
    del values, w, dout, lays
    torch.cuda.empty_cache()
    return records


# ---------------------------------------------------------------------------------------------------------------------
# Environment, run, report
# ---------------------------------------------------------------------------------------------------------------------


def environment():
    env = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "timer": timer_kind(),
    }
    for dist in ("triton", "apache-tvm-ffi", "cupti-python", "nvidia-ml-py", "cake-pplx-attn-res"):
        env[f"dist:{dist}"] = _dist(dist)
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        env.update(
            gpu=p.name,
            capability=f"{p.major}.{p.minor}",
            sm_count=p.multi_processor_count,
            l2_bytes=p.L2_cache_size,
            gpu_mem_gb=round(p.total_memory / 2**30, 1),
            driver=torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        )
        env["nvml"] = nvml_map()
    return env


def cmd_run(args):
    import torch._dynamo
    import torch._functorch.config

    # The timed ``bwd`` phase replays the backward of one recorded forward (retain_graph=True); AOTAutograd donated
    # buffers forbid that, so they are off for the compiled arm.
    torch._functorch.config.donated_buffer = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for attr in ("cache_size_limit", "recompile_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 4096)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cap_stream = torch.cuda.Stream(device)
    result = {"env_start": environment(), "args": vars(args), "records": []}
    path = Path(args.json)

    def flush():
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(path)

    flush()
    units = [("acc", n) for n in args.acc_n] + [("row", T, n) for T in args.T for n in args.n]
    for u in units:
        t0 = time.time()
        try:
            if u[0] == "acc":
                result["records"].extend(accuracy_unit(args.arms, u[1], args.draws, device))
            else:
                result["records"].extend(timing_unit(args.arms, u[1], u[2], args, device, cap_stream))
        except Exception as exc:  # noqa: BLE001
            result["records"].append(
                {
                    "kind": "unit_error",
                    "unit": list(u),
                    "error": _tail(exc, 1500),
                    "traceback": traceback.format_exc()[-4000:],
                }
            )
            log(f"UNIT ERROR {u}: {_tail(exc)}")
        log(f"unit {u} done in {time.time() - t0:.0f}s")
        result["env_end"] = environment()
        flush()
    log("RUN_DONE")


def cmd_report(args):
    data = json.loads(Path(args.json).read_text())
    recs = data["records"]
    env = data.get("env_start", {})
    acc = {(r["arm"], r["n"]): r for r in recs if r.get("kind") == "accuracy"}
    timing = [r for r in recs if r.get("kind") == "timing" and "mode" in r]
    lines = [f"## {args.title}", ""]
    nvml = env.get("nvml", {})
    dev0 = (nvml.get("devices") or [{}])[0]
    lines.append(
        f"GPU {env.get('gpu')} (cc {env.get('capability')}, {env.get('sm_count')} SMs), torch {env.get('torch')} "
        f"(CUDA {env.get('torch_cuda')}), timer {env.get('timer')}; "
        f"NVML bandwidth {dev0.get('nvml_bandwidth_gbs')} GB/s "
        f"(mem clock {dev0.get('mem_clock_mhz')} MHz x bus {dev0.get('bus_width_bits')} bit; CUDA_VISIBLE_DEVICES="
        f"{nvml.get('CUDA_VISIBLE_DEVICES')}, cuda:0 = NVML {dev0.get('nvml_index')} {dev0.get('uuid')})."
    )
    lines.append("")
    if acc:
        lines += ["| arm | n | meets the bar | out rel-L2 (median) | worst dv rel-L2 (median) | dw rel-L2 (median) |"]
        lines += ["|---|---|---|---|---|---|"]
        for (arm, n), r in sorted(acc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            rows = r["rows"]
            dv = max(v["rel_l2_median"] for k, v in rows.items() if k.startswith("dv"))
            lines.append(
                f"| {arm} | {n} | {'yes' if r['meets_bar'] else 'no'} | {rows['out']['rel_l2_median']:.2e} | "
                f"{dv:.2e} | {rows['dw']['rel_l2_median']:.2e} |"
            )
        lines.append("")
    meets = {key: r["meets_bar"] for key, r in acc.items()}
    by = collections.defaultdict(dict)
    for r in timing:
        by[(r["mode"], r["phase"], r["T"], r["n"])][r["arm"]] = r
    for mode in MODES:
        for phase in PHASES:
            keys = sorted(k for k in by if k[0] == mode and k[1] == phase)
            if not keys:
                continue
            lines += [f"### {phase}, {mode}", ""]
            lines += [
                "| T | n | bytes | baseline (us) | cake_nondet (us) | speedup | GB/s | HBM util | cake_det (us) | "
                "speedup | GB/s | HBM util |"
            ]
            lines += ["|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for key in keys:
                arms = by[key]
                T, n = key[2], key[3]
                base = [
                    arms[a] for a in BASELINE_ARMS if a in arms and "median_us" in arms[a] and meets.get((a, n), True)
                ]
                best = min(base, key=lambda r: r["median_us"]) if base else None
                cells = [str(T), str(n), f"{bytes_moved(phase, T, n) / 1e6:.0f} MB"]
                cells.append(f"{best['median_us']:.1f} ({best['arm']})" if best else "n/a")
                for arm in CAKE_ARMS:
                    r = arms.get(arm)
                    if r is None or "median_us" not in r:
                        cells += ["n/a", "", "", ""]
                        continue
                    speed = f"{best['median_us'] / r['median_us']:.2f}x" if best else "n/a"
                    cells += [f"{r['median_us']:.1f}", speed, f"{r['gbs']:.0f}", f"{100 * r['hbm_utilization']:.1f}%"]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    text = "\n".join(lines) + "\n"
    if args.md:
        Path(args.md).write_text(text)
    print(text)


def csv_list(conv):
    return lambda s: [conv(x) for x in s.split(",") if x]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--json", required=True)
    run.add_argument("--arms", type=csv_list(str), default=list(DEFAULT_ARMS))
    run.add_argument("--T", type=csv_list(int), default=[16380])
    run.add_argument("--n", type=csv_list(int), default=list(range(2, 10)))
    run.add_argument("--acc-n", type=csv_list(int), default=[2, 5, 9])
    run.add_argument("--draws", type=int, default=8)
    run.add_argument("--modes", type=csv_list(str), default=list(MODES))
    run.add_argument("--phases", type=csv_list(str), default=list(PHASES))
    run.add_argument("--dry", type=int, default=5)
    run.add_argument("--repeat", type=int, default=20)
    run.add_argument("--groups", type=int, default=3)
    run.add_argument("--r2r", action="store_true", help="check run-to-run bitwise equality per arm")
    report = sub.add_parser("report")
    report.add_argument("--json", required=True)
    report.add_argument("--md")
    report.add_argument("--title", default="AttnRes")
    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
