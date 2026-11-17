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

The shared driver of the mHC benchmarks (``bench_cake_mhc_det_train.py``, ``bench_cake_mhc_nondet_train.py``): the
Cake arms, the GPU timer, the interleaved groups and the JSON record.

Rows.  The default table has 27 shapes: hidden 4096 at the 13 token counts 1024, 4096, 8192, 14928, 15288, 15566,
15590, 16114, 16132, 16164, 16263, 32768 and 65536, and hidden 5120 and 7168 at 1024, 4096, 14928, 15288, 16263,
32768 and 65536 (``--shapes`` / ``--hidden`` / ``--tokens`` select others).  The inputs of a row are one draw of the
contract's distributions (``cake_mhc_baselines.make_inputs``), the same for every arm.

Operations (``--ops``):

* ``fwd``: Cake ``forward_prepare`` + ``forward_finish`` (``U``, the per-token statistics and ``Y``); a non-Cake arm's
  training forward (``U`` and ``Y``, the autograd graph recorded);
* ``bwd``: Cake ``backward_prepare`` + ``backward_finish`` from the saved statistics (``dB`` and the gradient
  statistics, then ``dX``, ``dfn``, ``dscale``, ``dbase``); a non-Cake arm's backward of a recorded forward;
* ``ckpt``: the checkpointed step: Cake ``forward_prepare`` + ``forward_finish``, the statistics replay
  ``forward_stats`` and the backward; a non-Cake arm's forward without the graph, its forward with the graph and the
  backward;
* ``bwd_acc`` (not in the default ``--ops``): ``bwd`` with ``accumulate=True`` (the parameter gradients added into
  the caller's buffers); a non-Cake arm's backward into its existing ``.grad`` tensors.

``--sinkhorn-iters`` sets the Sinkhorn count of every arm and of the references (default 20; the packages run any
other count on their run-time-count programs).

``cake_det_fp32x2`` / ``cake_nondet_fp32x2`` run the packages' opt-in ``accumulation="fp32x2"`` (a compensated pair
of FP32 words in the mapping, token-owner, column-slab and reduction passes; hidden 4096 and the default Sinkhorn count
only): beside the default arm of the same package, their ratio is the cost of the two-word mode.

Every Cake call gets caller-owned outputs and workspaces from the package's own factories.  ``--aa`` adds a second
instance of each Cake arm (``<arm>~aa``, its own buffers), timed like any other arm: its ratio to the first instance
is the noise band of the row.

Comparisons.  The non-Cake arm of each comparison is fixed (``cake_mhc_baselines.COMPARISONS``): ``fwd-A``
``tk_cublas``, ``fwd-B`` ``tk_deepgemm``, ``bwd`` / ``ckpt`` ``compile_max_autotune`` and the ``bwd-liger`` /
``ckpt-liger`` columns ``liger``, the same arm on every architecture.  Each cell of a row records the arm, its time,
the first Cake arm's time and their ratio, and a status: ``ok``, ``unsupported: <why>`` (the arm cannot run on this
device or stack), ``different contract`` (the arm missed the judge), or ``error: <message>``.  No cell ever falls
back to another arm, and no "fastest arm" is chosen per cell.  ``check_baselines.py`` verifies across the records of
several architectures that each comparison used the same arm and the same versions everywhere.

Determinism class.  The JSON records each arm's declared class (``cake_mhc_baselines.Arm.determinism``) and, per
row, the measured one (``--det-runs``, 5: the same call on the row's inputs repeated, every output compared bit for
bit with the first run).  The kernel names one eager call launches are recorded per arm, row and operation (with
``cupti-python``): they show which programs ran (e.g. the run-time-count programs, or whether ``accumulate=True``
selects other programs than the default).

Timing.  Each call runs on an idle GPU after a cold L2 (a buffer of twice the L2 size is zeroed, then the device is
synchronized; neither is timed).  The GPU activity records of the call (kernels, memsets and copies) come from CUPTI:
``cupti-python``, the records matched to the call's own runtime / driver API calls by correlation id, or, without
it, ``torch.profiler`` with one-cycle spin kernels marking the calls.  Per call three times are recorded: ``active``
(the union of the records' intervals: GPU busy time, the statistic of the tables), ``span`` (first start to last
end, the gaps between launches included) and ``sum`` (the records' durations added).  Modes (``--modes``): ``eager``
(the call launched each time) and ``graph`` (the call captured once in a CUDA graph on a side stream and replayed).
Per row, operation and mode, every arm is warmed up (``--warmup`` calls, which also compile), then timed in
``--groups`` groups of ``--calls`` calls (``--dry`` untimed calls first in each group); the arm order is rotated per
group and reversed every other group.  An arm's time is the median of its group medians, with the range of the group
medians, the 5th / 95th percentiles of all calls and, against the first Cake arm, the per-group ratios (median and
range; above 1 the Cake arm is faster).

Accuracy before timing.  ``--judge-draws`` (32) draws of the case ``(--judge-tokens, H)`` for each hidden size of the
run hold every arm to the naive-tier judge (``cake_mhc_baselines.judge_arms``); an arm that misses it is recorded as a
different contract: its comparison cells say so.  On every row each arm's outputs on the row's inputs are compared
with the float64 oracle once (``check``: the contract and the error of each output beside the naive tier's on that
draw); an arm that breaks the contract there is not timed on the row.

The JSON (``--json``) records the device (name, SM count, clocks, driver), the software (Python, torch, CUDA,
triton, the non-Cake libraries, ``cupti-python``, the loaded cuBLAS / NVRTC / CUPTI libraries), the image
(``CAKE_IMAGE_TAG`` / ``CAKE_IMAGE_SHA256``), each arm's description (library, composite, declared determinism) and
versions, each Cake package's identity (distribution version, module path, the backend manifests' generator record
and SHA-256), the comparisons, the protocol and every row.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import gc
import hashlib
import importlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_mhc_baselines as base  # noqa: E402

T_H4096 = (1024, 4096, 8192, 14928, 15288, 15566, 15590, 16114, 16132, 16164, 16263, 32768, 65536)
T_WIDE = (1024, 4096, 14928, 15288, 16263, 32768, 65536)
DEFAULT_SHAPES = tuple((4096, T) for T in T_H4096) + tuple((H, T) for H in (5120, 7168) for T in T_WIDE)
CAKE_MODES = ("det", "nondet")
# The packages' accumulation modes: "fp32" (default: one plain FP32 word per sum) and the opt-in "fp32x2" (a
# compensated pair of FP32 words in the mapping, token-owner, column-slab and reduction passes), hidden 4096 only.
CAKE_ACCUMULATIONS = ("fp32", "fp32x2")
PAIR_HIDDEN = 4096
CAKE_ARMS = {"cake_det": ("det", "fp32"), "cake_nondet": ("nondet", "fp32"),
             "cake_det_fp32x2": ("det", "fp32x2"), "cake_nondet_fp32x2": ("nondet", "fp32x2")}


# ----------------------------------------------------------------------------------------------------------------
# GPU timer


def _interval_stats(intervals: list) -> dict:
    """``active`` (union), ``span`` and ``sum`` in microseconds of ``(start_ns, end_ns)`` intervals."""
    if not intervals:
        raise RuntimeError("no GPU activity was recorded for a timed call")
    intervals = sorted(intervals)
    total = sum(e - s for s, e in intervals)
    active = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            active += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    active += cur_e - cur_s
    span = max(e for _, e in intervals) - intervals[0][0]
    return {"active_us": active / 1e3, "span_us": span / 1e3, "sum_us": total / 1e3, "records": len(intervals)}


class Timer:
    """Per-call GPU times of a callable (see the module docstring): ``backend`` ``cupti`` (cupti-python), ``profiler``
    (torch.profiler) or ``auto`` (cupti-python when importable)."""

    def __init__(self, device: torch.device, backend: str = "auto", calls: int = 32, dry: int = 2) -> None:
        self.device = device
        self.default_calls = calls
        self.dry = dry
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        self.flush = torch.empty(max(2 * l2, 1 << 20), dtype=torch.int8, device=device)
        self.flush_bytes = self.flush.numel()
        if backend == "auto":
            try:
                from cupti import cupti  # noqa: F401

                backend = "cupti"
            except Exception:  # noqa: BLE001
                backend = "profiler"
        if backend not in ("cupti", "profiler"):
            raise ValueError(backend)
        self.backend = backend

    def times(self, run: Callable[[], object], calls: Optional[int] = None, dry: Optional[int] = None) -> list:
        calls = calls or self.default_calls
        for _ in range(self.dry if dry is None else dry):
            self.flush.zero_()
            run()
        torch.cuda.synchronize(self.device)
        if self.backend == "cupti":
            return self._cupti(run, calls)
        return self._profiler(run, calls)

    def _cupti(self, run, calls: int) -> list:
        from cupti import cupti

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
            for _ in range(calls):
                self.flush.zero_()
                torch.cuda.synchronize(self.device)
                begin = cupti.get_timestamp()
                run()
                end = cupti.get_timestamp()
                stamps.append((begin, end))
            torch.cuda.synchronize(self.device)
            cupti.activity_flush_all(0)
        finally:
            for kind in kinds:
                cupti.activity_disable(kind)
            cupti.finalize()
        launches.sort()
        starts = [s for s, _ in launches]
        out = []
        for begin, end in stamps:
            lo, hi = bisect.bisect_left(starts, begin), bisect.bisect_right(starts, end)
            corr = {c for _, c in launches[lo:hi]}
            out.append(_interval_stats([iv for c in corr for iv in spans.get(c, ())]))
        return out

    def kernel_names(self, run: Callable[[], object], limit: int = 256) -> Optional[list]:
        """The names of the kernels one call of ``run`` launches, in start order (``cupti-python`` only; else
        None)."""
        if self.backend != "cupti":
            return None
        from cupti import cupti

        found = []
        kind = cupti.ActivityKind.CONCURRENT_KERNEL

        def requested():
            return 8 * 1024 * 1024, 0

        def completed(activities):
            for activity in activities:
                if activity.kind == kind:
                    found.append((activity.start, str(getattr(activity, "name", "?"))))

        torch.cuda.synchronize(self.device)
        cupti.activity_enable(kind)
        cupti.activity_register_callbacks(requested, completed)
        try:
            run()
            torch.cuda.synchronize(self.device)
            cupti.activity_flush_all(0)
        finally:
            cupti.activity_disable(kind)
            cupti.finalize()
        names = [n for _, n in sorted(found)]
        return names[:limit] + ([f"... {len(names) - limit} more"] if len(names) > limit else [])

    def _profiler(self, run, calls: int) -> list:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                torch.cuda._sleep(1)
                self.flush.zero_()
                torch.cuda.synchronize(self.device)
                run()
            torch.cuda._sleep(1)
            torch.cuda.synchronize(self.device)
        events = sorted(
            (e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda e: e.time_range.start,
        )
        out, current = [], None
        for e in events:
            if "spin_kernel" in e.name:
                if current is not None:
                    out.append(_interval_stats([(int(x.time_range.start * 1e3), int(x.time_range.end * 1e3))
                                                for x in current[1:]]))
                current = []
            elif current is not None:
                current.append(e)
        if len(out) != calls:
            raise RuntimeError(f"the profiler returned {len(out)} complete calls of {calls}")
        return out


# ----------------------------------------------------------------------------------------------------------------
# Cake arms


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class CakeArm(base.Arm):
    """One Cake package (``det`` or ``nondet``) through its explicit entry points."""

    cake = True

    def __init__(self, mode: str, label: Optional[str] = None, accumulation: str = "fp32") -> None:
        if mode not in CAKE_MODES:
            raise ValueError(mode)
        if accumulation not in CAKE_ACCUMULATIONS:
            raise ValueError(accumulation)
        self.mode = mode
        self.accumulation = accumulation
        self.name = label or (f"cake_{mode}" if accumulation == "fp32" else f"cake_{mode}_{accumulation}")
        self.library = f"cake_pplx.cake_mhc_{mode}_train"
        self.deterministic = mode == "det"
        self.determinism = (
            "deterministic by contract: run-to-run bitwise on one device" if mode == "det"
            else "non-deterministic mode: the contract promises no run-to-run bits"
        )
        self.mod = importlib.import_module(f"cake_pplx.cake_mhc_{mode}_train")

    def f(self, name: str):
        return getattr(self.mod, f"cake_mhc_{self.mode}_train_{name}")

    def available(self, device, hidden):
        supported = getattr(self.mod, f"is_cake_mhc_{self.mode}_train_supported")
        ok = bool(supported(device, hidden))
        if not ok:
            return False, f"the package exports no route for hidden {hidden} on this device"
        if self.accumulation != "fp32" and hidden != PAIR_HIDDEN:
            return False, f"accumulation={self.accumulation!r} serves hidden {PAIR_HIDDEN} only"
        return True, ""

    def backend_dir(self) -> Path:
        backend = importlib.import_module(f"cake_pplx._backends.cake_mhc_{self.mode}_train")
        return Path(backend.__file__).resolve().parent

    def identity(self) -> dict:
        d = self.backend_dir()
        out = {
            "distribution": "cake-pplx-mhc",
            "version": base.dist_version("cake-pplx-mhc"),
            "module": str(Path(self.mod.__file__).resolve()),
            "backend": str(d),
            "build_json_sha256": _sha256(d / "build.json"),
            "cake_jit_sha256": _sha256(d / "cake_jit.py"),
            "accumulation": self.accumulation,
        }
        try:
            out["generator"] = json.loads((d / "build.json").read_text()).get("generator")
        except (OSError, ValueError) as exc:
            out["generator"] = f"unreadable: {exc}"
        loader = importlib.import_module(f"cake_pplx._backends.cake_mhc_{self.mode}_train.cake_jit")
        out["PARAM_ROW_COLS"] = getattr(loader, "PARAM_ROW_COLS", None)
        return out

    def versions(self) -> dict:
        return self.identity()

    def prebuild(self, device, workers: int = 1) -> dict:
        """Build every library of the device's architecture that has no prebuilt file (the package's warm-up),
        ``workers`` libraries at a time, then load each."""
        from concurrent.futures import ThreadPoolExecutor

        loader = importlib.import_module(f"cake_pplx._backends.cake_mhc_{self.mode}_train.cake_jit")
        arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
        t0 = time.time()
        specs = loader.specs_for_arch(arch)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(lambda spec: spec.build(), specs))
        for spec in specs:
            spec.build_and_load()
        return {"arch": arch, "specs": len(specs), "workers": workers, "seconds": round(time.time() - t0, 1)}

    def _buffers(self, i: dict) -> dict:
        T, H, dev = i["T"], i["hidden"], i["X"].device
        f32 = torch.float32
        return {
            "U": torch.empty(T, H, dtype=torch.bfloat16, device=dev),
            "S": torch.empty(T, 64, dtype=f32, device=dev),
            "S2": torch.empty(T, 64, dtype=f32, device=dev),
            "Y": torch.empty(T, base.STREAMS, H, dtype=torch.bfloat16, device=dev),
            "dB": torch.empty(T, H, dtype=torch.bfloat16, device=dev),
            "G": torch.empty(T, 32, dtype=f32, device=dev),
            "dX": torch.empty(T, base.STREAMS, H, dtype=torch.bfloat16, device=dev),
            "dfn": torch.empty(base.MAPPING_ROWS, base.STREAMS * H, dtype=f32, device=dev),
            "dscale": torch.empty(3, dtype=f32, device=dev),
            "dbase": torch.empty(base.MAPPING_ROWS, dtype=f32, device=dev),
            "fws": self.f("forward_workspace")(T, dev, hidden=H),
            "bws": self.f("backward_workspace")(T, dev, hidden=H),
        }

    def _calls(self, i: dict, b: dict):
        X, fn, scale, bse, B, dY, dU = (i[k] for k in ("X", "fn", "scale", "base", "B", "dY", "dU"))
        fp, ff, fs = self.f("forward_prepare"), self.f("forward_finish"), self.f("forward_stats")
        bp, bf = self.f("backward_prepare"), self.f("backward_finish")
        # The default count is not passed: the call is the package's default call.
        kw = {} if i["iters"] == base.SINKHORN_ITERS else {"sinkhorn_iters": i["iters"]}
        if self.accumulation != "fp32":  # the opt-in two-word mode (the default call passes no mode)
            kw["accumulation"] = self.accumulation

        def forward(stats) -> None:
            fp(X, fn, scale, bse, out=b["U"], stats_out=stats, workspace=b["fws"], **kw)
            ff(X, B, stats, out=b["Y"])

        def backward(stats, accumulate: bool = False) -> None:
            bp(dY, B, X, stats, dB=b["dB"], grad_stats_out=b["G"])
            bf(X, dY, dU, B, fn, scale, bse, stats, b["G"], workspace=b["bws"], dX=b["dX"], dfn=b["dfn"],
               dscale=b["dscale"], dbase=b["dbase"], **kw, **({"accumulate": True} if accumulate else {}))

        def replay(stats) -> None:
            fs(X, fn, scale, bse, stats_out=stats, workspace=b["fws"], **kw)

        return forward, backward, replay

    def run_once(self, i: dict) -> dict:
        b = self._buffers(i)
        forward, backward, _ = self._calls(i, b)
        forward(b["S"])
        backward(b["S"])
        return {k: b[k] for k in base.OUTPUTS}

    def make(self, i: dict, op: str) -> Callable[[], None]:
        b = self._buffers(i)
        forward, backward, replay = self._calls(i, b)
        if op == "fwd":
            return lambda: forward(b["S"])
        if op == "bwd":
            forward(b["S"])

            return lambda: backward(b["S"])
        if op == "bwd_acc":
            forward(b["S"])
            backward(b["S"])  # dfn, dscale, dbase hold a gradient before the first accumulation

            return lambda: backward(b["S"], accumulate=True)
        if op == "ckpt":

            def call() -> None:
                forward(b["S"])
                replay(b["S2"])
                backward(b["S2"])

            return call
        raise ValueError(op)


# ----------------------------------------------------------------------------------------------------------------
# Environment record


def _run(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


def device_record(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    uuid = getattr(props, "uuid", None)
    rec = {
        "name": props.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": props.multi_processor_count,
        "total_memory": props.total_memory,
        "l2_bytes": props.L2_cache_size,
        "uuid": f"GPU-{uuid}" if uuid is not None else None,
    }
    fields = "uuid,name,serial,pci.bus_id,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit"
    smi = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    for line in smi.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if rec["uuid"] and cols and cols[0] == rec["uuid"]:
            rec.update(zip(fields.split(",")[1:], cols[1:]))
    return rec


def loaded_libraries() -> dict:
    """The CUDA libraries mapped into this process (path per library name: cuBLAS, cuBLASLt, NVRTC, the runtime,
    CUPTI, cuDNN)."""
    found: dict = {}
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                parts = line.split()
                path = parts[-1] if len(parts) >= 6 else ""
                for lib in ("libcublas.so", "libcublasLt.so", "libnvrtc.so", "libnvrtc-builtins", "libcudart.so",
                            "libcupti.so", "libcudnn.so", "libnvJitLink.so"):
                    if f"/{lib}" in path:
                        found.setdefault(lib, path)
    except OSError as exc:
        found["error"] = str(exc)
    return found


def software_record(timer: Timer) -> dict:
    # A small GEMM first, so that the cuBLAS library is loaded and its version can be read.
    a = torch.ones(64, 64, device="cuda")
    torch.mm(a, a)
    torch.cuda.synchronize()
    rec = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nvcc": _run(["nvcc", "--version"]).splitlines()[-1:] or None,
        "timer": timer.backend,
    }
    for dist in ("triton", "pytorch-triton", "cupti-python", "apache-tvm-ffi", "numpy", "liger-kernel", "tilelang",
                 "tile-kernels", "deep_gemm", "deep-gemm", "nvidia-cuda-nvrtc", "nvidia-cuda-runtime"):
        version = base.dist_version(dist)
        if version is not None:
            rec[dist] = version
    try:
        import deep_gemm

        rec["deep_gemm.__version__"] = getattr(deep_gemm, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    rec["cublas"] = base.cublas_version()
    rec["libraries"] = loaded_libraries()
    rec["env"] = {k: os.environ.get(k) for k in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRITON_F32_DEFAULT",
        "CAKE_PPLX_CACHE_DIR", "CAKE_PPLX_JIT_WORKERS", "CAKE_IMAGE_TAG", "CAKE_IMAGE_SHA256", "CUDA_HOME",
        "LD_PRELOAD", "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "CAKE_BENCH_STACK")}
    rec["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32,
                   "float32_matmul_precision": torch.get_float32_matmul_precision()}
    return rec


# ----------------------------------------------------------------------------------------------------------------
# Rows


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _err(exc: BaseException, where: str = "") -> str:
    """The recorded message of a caught failure; with ``where``, the traceback also goes to stderr."""
    if where:
        print(f"FAILED {where}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(limit=8, file=sys.stderr)
    return f"{type(exc).__name__}: {str(exc)[:500]}"


def _summary(samples: list, key: str) -> dict:
    groups = [statistics.median(s[key] for s in g) for g in samples]
    flat = sorted(s[key] for g in samples for s in g)

    def pct(p: float) -> float:
        return flat[min(len(flat) - 1, max(0, int(round(p * (len(flat) - 1)))))]

    return {"median": statistics.median(groups), "groups": groups, "range": [min(groups), max(groups)],
            "p05": pct(0.05), "p95": pct(0.95), "calls": len(flat)}


def _build(arm: base.Arm, i: dict, op: str, mode: str, warmup: int, device) -> Callable[[], None]:
    """The timed callable of ``arm``'s ``op`` in ``mode`` (warmed up; captured in a CUDA graph for ``graph``)."""
    if mode == "eager":
        call = arm.make(i, op)
        for _ in range(warmup):
            call()
        torch.cuda.synchronize(device)
        return call
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        call = arm.make(i, op)
        for _ in range(warmup):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.synchronize(device)

    # The graph replays into the buffers ``call`` owns: they must live as long as the graph (capturing the next
    # graph empties the allocator's cache, so a freed buffer would be unmapped under a later replay).
    keep = (graph, call)

    def run() -> None:
        keep[0].replay()

    run()
    torch.cuda.synchronize(device)
    return run


def time_op(arms: list, i: dict, op: str, mode: str, args, timer: Timer, device) -> dict:
    """Interleaved groups of every arm's ``op`` in ``mode`` on the inputs ``i``."""
    result: dict = {}
    runs: dict = {}
    for arm in arms:
        try:
            runs[arm.name] = _build(arm, i, op, mode, args.warmup, device)
        except torch.cuda.OutOfMemoryError as exc:
            result[arm.name] = {"error": "out of memory: " + _err(exc, f"{arm.name} {op} {mode} build")}
            _free()
        except Exception as exc:  # noqa: BLE001 - recorded per arm; the other arms continue
            result[arm.name] = {"error": _err(exc, f"{arm.name} {op} {mode} build"),
                                "trace": traceback.format_exc(limit=4)[-1500:]}
            _free()
    names = list(runs)
    if mode == "eager" and getattr(args, "kernel_names", True):
        for n in names:
            try:
                result.setdefault(n, {})["kernels"] = timer.kernel_names(runs[n])
            except Exception as exc:  # noqa: BLE001 - the names are a record, not a measurement
                result.setdefault(n, {})["kernels_error"] = _err(exc)
    samples = {n: [] for n in names}
    for g in range(args.groups):
        if not names:
            break
        k = g % len(names)
        order = names[k:] + names[:k]
        if g % 2:
            order.reverse()
        for n in order:
            try:
                samples[n].append(timer.times(runs[n], args.calls, args.dry))
            except Exception as exc:  # noqa: BLE001
                result.setdefault(n, {}).update({"error": f"group {g}: " + _err(exc, f"{n} {op} {mode} group {g}"),
                                                 "trace": traceback.format_exc(limit=4)[-1500:]})
                runs.pop(n, None)
        names = [n for n in names if n in runs]
    for n, groups in samples.items():
        if "error" in result.get(n, {}) or not groups:
            continue
        result.setdefault(n, {}).update({
            "active_us": _summary(groups, "active_us"), "span_us": _summary(groups, "span_us"),
            "sum_us": _summary(groups, "sum_us"),
            "records_per_call": statistics.median(s["records"] for g in groups for s in g)})
    cake = next((a.name for a in arms if a.cake and a.name in samples and "active_us" in result.get(a.name, {})), None)
    if cake:
        ref = [statistics.median(s["active_us"] for s in g) for g in samples[cake]]
        for n, groups in samples.items():
            if n == cake or "active_us" not in result.get(n, {}):
                continue
            ratios = [statistics.median(s["active_us"] for s in g) / r for g, r in zip(groups, ref)]
            result[n][f"over_{cake}"] = {"median": statistics.median(ratios), "range": [min(ratios), max(ratios)],
                                         "groups_slower": sum(x > 1 for x in ratios), "groups": len(ratios)}
    del runs
    _free()
    return result


def run_row(hidden: int, tokens: int, arms: list, args, timer: Timer, judge, device, eligible: dict) -> dict:
    row: dict = {"hidden": hidden, "tokens": tokens, "seed": args.seed + 7919 * hidden + tokens,
                 "iters": args.sinkhorn_iters, "arms": {}, "ops": {}}
    t0 = time.time()
    i = base.make_inputs(tokens, hidden, seed=row["seed"], device=device, iters=args.sinkhorn_iters)
    live = []
    for arm in arms:
        ok, note = arm.available(device, hidden)
        rec = row["arms"][arm.name] = {"available": ok, "note": note}
        if not ok:
            print(f"H={hidden} T={tokens} {arm.name}: unsupported: {note}", flush=True)
            continue
        try:
            rec["prepare"] = arm.prepare_shape(i, timer)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "prepare: " + _err(exc, f"{arm.name} prepare H={hidden} T={tokens}")
            _free()
            continue
        live.append(arm)
    if args.check and live:
        outputs, failed = {}, set()
        for arm in live:
            try:
                outputs[arm.name] = arm.run_once(i)
            except Exception as exc:  # noqa: BLE001
                row["arms"][arm.name]["error"] = "check run: " + _err(exc, f"{arm.name} check run H={hidden} T={tokens}")
                failed.add(arm.name)
                _free()
        try:
            report = base.check_outputs(i, outputs, judge)
        except torch.cuda.OutOfMemoryError as exc:
            report = {n: {"error": "check: out of memory: " + _err(exc)} for n in outputs}
        del outputs
        _free()
        for name, rep in report.items():
            row["arms"][name]["check"] = rep
            if any(isinstance(v, dict) and v.get("contract") for v in rep.values()):
                failed.add(name)
        live = [a for a in live if a.name not in failed]
    if args.det_runs > 1:
        for arm in live:
            if arm.name.endswith("~aa"):
                continue
            try:
                row["arms"][arm.name]["determinism"] = base.determinism_runs(arm, i, args.det_runs)
            except Exception as exc:  # noqa: BLE001 - recorded; the timing continues
                row["arms"][arm.name]["determinism"] = {"error": _err(exc, f"{arm.name} determinism")}
            _free()
    for op in args.ops:
        participants = [a for a in live if op in a.ops]
        for mode in args.modes:
            res = time_op(participants, i, op, mode, args, timer, device)
            res["comparisons"] = comparisons(arms, row, res, op, hidden, eligible)
            row["ops"].setdefault(op, {})[mode] = res
            _print_line(hidden, tokens, op, mode, participants, res)
    row["seconds"] = round(time.time() - t0, 1)
    del i
    _free()
    return row


def comparisons(arms: list, row: dict, res: dict, op: str, hidden: int, eligible: dict) -> dict:
    """The fixed comparisons of ``op`` on a row (see the module docstring): per comparison the arm, its status, its
    time, the first Cake arm's time and ``speedup`` (the arm's time over the Cake arm's: above 1 the Cake arm is
    faster) with the median per-group ratio ``paired``.  Empty when the run has no non-Cake arm."""
    if not any(not a.cake for a in arms):
        return {}
    by_name = {a.name: a for a in arms}
    cake = next((a.name for a in arms if a.cake and "active_us" in res.get(a.name, {})), None)
    out = {}
    for cname, aname in base.comparisons_for(op):
        cell: dict = {"arm": aname, "label": by_name[aname].label() if aname in by_name else aname, "reference": cake,
                      "judge": eligible.get((hidden, aname))}
        rec = row["arms"].get(aname)
        r = res.get(aname, {})
        if aname not in by_name:
            cell["status"] = "not run: the arm was not selected"
        elif not rec or not rec.get("available"):
            cell["status"] = "unsupported: " + ((rec or {}).get("note") or "not available")
        elif rec.get("error"):
            cell["status"] = "error: " + rec["error"]
        elif any(isinstance(v, dict) and v.get("contract") for v in (rec.get("check") or {}).values()):
            cell["status"] = "different contract: the arm breaks the contract on this row's inputs"
        elif "active_us" not in r:
            cell["status"] = "error: " + str(r.get("error") or "not timed")
        elif eligible.get((hidden, aname)) is False:
            cell["status"] = "different contract: the arm misses the naive-tier judge"
        else:
            cell["status"] = "ok"
        if "active_us" in r and cake:
            arm_us, cake_us = r["active_us"]["median"], res[cake]["active_us"]["median"]
            cell.update({"arm_us": arm_us, "cake_us": cake_us, "speedup": arm_us / cake_us,
                         "paired": (r.get(f"over_{cake}") or {}).get("median")})
        out[cname] = cell
    return out


def _print_line(hidden, tokens, op, mode, arms, res) -> None:
    cells = []
    for arm in arms:
        r = res.get(arm.name, {})
        if "active_us" in r:
            cells.append(f"{arm.name} {r['active_us']['median']:.2f}")
        else:
            cells.append(f"{arm.name} ERR")
    comp = "; ".join(f"{k} {v['arm']} {v['status'][:40]}" + (f" x{v['speedup']:.2f}" if "speedup" in v else "")
                     for k, v in res.get("comparisons", {}).items())
    print(f"H={hidden} T={tokens} {op} {mode} | " + " | ".join(cells) + (f" || {comp}" if comp else ""), flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Command line


def parse_shapes(args) -> list:
    if args.shapes:
        shapes = []
        for item in args.shapes:
            for part in item.split(","):
                if part:
                    h, t = part.lower().split("x")
                    shapes.append((int(h), int(t)))
        return shapes
    return [(h, t) for h, t in DEFAULT_SHAPES
            if (not args.hidden or h in args.hidden) and (not args.tokens or t in args.tokens)]


def build_parser(description: str, default_arms: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default=default_arms,
                   help="comma-separated arms: cake_det, cake_nondet, their opt-in two-word modes cake_det_fp32x2 / "
                   "cake_nondet_fp32x2 (hidden 4096) and " + ", ".join(base.NON_CAKE_ARMS))
    p.add_argument("--aa", action="store_true", help="add a second instance of each Cake arm (the noise band)")
    p.add_argument("--shapes", nargs="*", default=None, help="HxT items, e.g. 4096x1024,7168x65536")
    p.add_argument("--hidden", type=int, nargs="*", default=None)
    p.add_argument("--tokens", type=int, nargs="*", default=None)
    p.add_argument("--ops", default="fwd,bwd,ckpt", help="comma-separated: fwd, bwd, ckpt, bwd_acc")
    p.add_argument("--sinkhorn-iters", type=int, default=base.SINKHORN_ITERS,
                   help="Sinkhorn count of every arm and reference (default 20; 1..64 for the Cake packages)")
    p.add_argument("--det-runs", type=int, default=5,
                   help="runs of each arm per row compared bit for bit with the first (0: no determinism check)")
    p.add_argument("--no-kernel-names", dest="kernel_names", action="store_false",
                   help="do not record the kernel names of one eager call per arm, row and operation")
    p.add_argument("--modes", default="eager,graph")
    p.add_argument("--groups", type=int, default=8)
    p.add_argument("--calls", type=int, default=32)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--dry", type=int, default=2)
    p.add_argument("--no-check", dest="check", action="store_false",
                   help="skip the per-row comparison with the float64 oracle")
    p.add_argument("--judge-draws", type=int, default=base.JUDGE_DRAWS,
                   help="draws of the naive-tier judge per hidden size (0: no judge)")
    p.add_argument("--judge-tokens", type=int, default=base.JUDGE_TOKENS)
    p.add_argument("--judge-only", action="store_true")
    p.add_argument("--compile-trials", type=int, default=1)
    p.add_argument("--tk-splits", type=int, default=16)
    p.add_argument("--prebuild", action="store_true", help="build the Cake libraries of this architecture first")
    p.add_argument("--prebuild-workers", type=int, default=8, help="libraries built at a time by --prebuild")
    p.add_argument("--timer", default="auto", choices=("auto", "cupti", "profiler"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="", help="free text recorded in the JSON (e.g. the package under test)")
    p.add_argument("--json", default=None)
    return p


def make_arms(names: list, args) -> list:
    arms: list = []
    for name in names:
        if name in CAKE_ARMS:
            mode, accumulation = CAKE_ARMS[name]
            arms.append(CakeArm(mode, accumulation=accumulation))
            if args.aa:
                arms.append(CakeArm(mode, f"{name}~aa", accumulation=accumulation))
    others = [n for n in names if n not in CAKE_ARMS]
    arms.extend(base.non_cake_arms(others, compile_trials=args.compile_trials, tk_splits=args.tk_splits))
    return arms


def main(argv: Optional[list] = None, *, description: str, default_arms: str) -> int:
    args = build_parser(description, default_arms).parse_args(argv)
    args.ops = [o for o in args.ops.split(",") if o]
    args.modes = [m for m in args.modes.split(",") if m]
    if "liger" in args.arms and "TRITON_F32_DEFAULT" not in os.environ:
        os.environ["TRITON_F32_DEFAULT"] = "tf32x3"
    base.set_ieee_fp32()
    device = torch.device("cuda", torch.cuda.current_device())
    timer = Timer(device, args.timer, args.calls, args.dry)
    try:  # a timer that cannot record (e.g. a CUPTI too old for the device) fails the run here, not per arm
        timer.times(lambda: timer.flush.zero_(), calls=2, dry=0)
    except Exception as exc:  # noqa: BLE001
        print(f"TIMER SELF-TEST FAILED ({timer.backend}): {_err(exc)}", file=sys.stderr, flush=True)
        return 4
    judge = base.load_naive_judge()
    arms = make_arms([a for a in args.arms.split(",") if a], args)
    shapes = parse_shapes(args)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    record: dict = {
        "label": args.label,
        "argv": sys.argv,
        "host": socket.gethostname(),
        "started": started,
        "device": device_record(device),
        "software": software_record(timer),
        "protocol": {"groups": args.groups, "calls": args.calls, "warmup": args.warmup, "dry": args.dry,
                     "ops": args.ops, "modes": args.modes, "flush_bytes": timer.flush_bytes,
                     "statistic": "median of group medians of per-call CUPTI active time (us)",
                     "judge_draws": args.judge_draws, "judge_tokens": args.judge_tokens, "check": args.check,
                     "sinkhorn_iters": args.sinkhorn_iters, "det_runs": args.det_runs},
        "comparisons": [{"name": n, "ops": list(o), "arm": a} for n, o, a in base.COMPARISONS],
        "arms": {a.name: a.describe() for a in arms},
        "versions": {},
        "judge": {},
        "rows": [],
    }
    print(f"device {record['device']}", flush=True)
    print(f"software {record['software']}", flush=True)
    for arm in arms:
        try:
            record["versions"][arm.name] = arm.versions()
        except Exception as exc:  # noqa: BLE001
            record["versions"][arm.name] = {"error": _err(exc)}
        print(f"arm {arm.name}: {arm.library} {record['versions'][arm.name]}", flush=True)
    if args.prebuild:
        for arm in arms:
            if isinstance(arm, CakeArm) and not arm.name.endswith("~aa"):
                record.setdefault("prebuild", {})[arm.name] = arm.prebuild(device, args.prebuild_workers)
                print(f"prebuild {arm.name}: {record['prebuild'][arm.name]}", flush=True)

    def dump() -> None:
        if args.json:
            record["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            tmp = args.json + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(record, handle, indent=1, default=str)
            os.replace(tmp, args.json)

    eligible: dict = {}
    if args.judge_draws > 0:
        for hidden in sorted({h for h, _ in shapes}):
            candidates = {}
            for arm in arms:
                if arm.name.endswith("~aa"):
                    continue
                ok, note = arm.available(device, hidden)
                if ok:
                    try:
                        arm.prepare_shape(base.make_inputs(args.judge_tokens, hidden, seed=args.seed, device=device,
                                                           iters=args.sinkhorn_iters))
                        candidates[arm.name] = arm
                    except Exception as exc:  # noqa: BLE001
                        record["judge"].setdefault(str(hidden), {})[arm.name] = {"error": "prepare: " + _err(exc),
                                                                                 "meets_contract": False}
                else:
                    record["judge"].setdefault(str(hidden), {})[arm.name] = {"unavailable": note}
                    print(f"judge H={hidden} {arm.name}: unsupported: {note}", flush=True)
            verdicts = base.judge_arms(candidates, hidden, device=device, judge=judge, draws=args.judge_draws,
                                       tokens=args.judge_tokens, seed=args.seed, iters=args.sinkhorn_iters)
            record["judge"].setdefault(str(hidden), {}).update(verdicts)
            for name, v in verdicts.items():
                eligible[(hidden, name)] = bool(v.get("meets_contract"))
                print(f"judge H={hidden} T={args.judge_tokens} {name}: "
                      f"{'meets the contract' if v.get('meets_contract') else 'DIFFERENT CONTRACT'} "
                      f"{v.get('error') or ''} {'; '.join(v.get('failures', [])[:3])}", flush=True)
                for line in v.get("tier_lines", []):
                    print("  " + line, flush=True)
            _free()
            dump()
    if not args.judge_only:
        for hidden, tokens in shapes:
            try:
                record["rows"].append(run_row(hidden, tokens, arms, args, timer, judge, device, eligible))
            except Exception as exc:  # noqa: BLE001 - e.g. a sticky CUDA error: the process cannot continue
                record["fatal"] = {"hidden": hidden, "tokens": tokens, "error": _err(exc, f"row H={hidden} T={tokens}"),
                                   "trace": traceback.format_exc(limit=8)[-3000:]}
                dump()
                return 3
            dump()
    dump()
    return 0
