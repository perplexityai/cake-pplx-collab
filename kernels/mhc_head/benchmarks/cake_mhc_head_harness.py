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

The timer, the interleaved timing groups, the environment record and the determinism helpers of the mHC training-head
benchmark (``bench_cake_mhc_head_train.py``).  They are copies of the pieces of the mhc family's benchmark harness
that the head benchmark uses (the mhc family's ``benchmarks/cake_mhc_harness.py``: the GPU timer, the CUDA graph
build, the interleaved groups, the device and software records and the default shape table; its
``benchmarks/cake_mhc_baselines.py``: the arm base class, the version and bit helpers and the determinism runs),
kept in this family because a family's benchmarks import only their own modules.  Their behaviour is the mhc
harness's: per call the CUPTI ``active`` (union of the GPU activity intervals), ``span`` and ``sum`` times after a
cold L2, eager launches or CUDA graph replays, the arm order rotated per group and reversed every other group, and an
arm's time the median of its group medians.
"""

from __future__ import annotations

import bisect
import collections
import gc
import importlib.metadata
import os
import platform
import statistics
import subprocess
import sys
import traceback
import types
from typing import Callable, Optional

import torch

#: The operations of the head benchmark (the default operations of an arm).
OPS = ("fwd", "bwd", "bwd_acc")
#: The default rows of :func:`parse_shapes` (without ``--shapes``): the mHC benchmarks' table.
T_H4096 = (1024, 4096, 8192, 14928, 15288, 15566, 15590, 16114, 16132, 16164, 16263, 32768, 65536)
T_WIDE = (1024, 4096, 14928, 15288, 16263, 32768, 65536)
DEFAULT_SHAPES = tuple((4096, T) for T in T_H4096) + tuple((H, T) for H in (5120, 7168) for T in T_WIDE)


# ----------------------------------------------------------------------------------------------------------------
# Versions


def dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cublas_version() -> Optional[str]:
    """``major.minor.patch (path)`` of the cuBLAS library loaded in this process (``cublasGetProperty``), or None
    when none is loaded yet."""
    import ctypes

    path = None
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                name = line.split()[-1] if line.split() else ""
                if "/libcublas.so" in name:
                    path = name
                    break
    except OSError:
        return None
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(path)
        parts = []
        for prop in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
            value = ctypes.c_int()
            if lib.cublasGetProperty(prop, ctypes.byref(value)) != 0:
                return None
            parts.append(str(value.value))
        return ".".join(parts) + f" ({path})"
    except (OSError, AttributeError):
        return None


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
# Arms and determinism


class Arm:
    """One implementation of the function.  ``make(i, op)`` returns the timed callable of an operation on the
    inputs ``i`` (its buffers and state allocated once, outside the timing); ``run_once(i)`` returns every output it
    computes on ``i``.

    ``composite``: None for a library's (or the PyTorch reference's) own code path, else what this harness composed.
    ``determinism``: what the implementation declares about its run-to-run bits (the benchmarks also measure it)."""

    name = "base"
    library = ""
    ops: tuple = OPS
    cake = False
    deterministic: Optional[bool] = None
    composite: Optional[str] = None
    determinism = "not declared (measured by the benchmarks)"

    def available(self, device: torch.device, hidden: int) -> tuple[bool, str]:
        return True, ""

    def versions(self) -> dict:
        return {}

    def label(self) -> str:
        """The arm's name in reports: composites carry the suffix ``(composite)``."""
        return f"{self.name} (composite)" if self.composite else self.name

    def describe(self) -> dict:
        return {"library": self.library, "cake": self.cake, "deterministic": self.deterministic,
                "determinism_declared": self.determinism, "composite": self.composite, "label": self.label(),
                "ops": list(self.ops)}

    def prepare_shape(self, i: dict, timer=None) -> dict:
        """Per-shape setup before the check and the timing (e.g. compile trials); returns what to record."""
        return {}

    def run_once(self, i: dict) -> dict:
        raise NotImplementedError

    def make(self, i: dict, op: str) -> Callable[[], None]:
        raise NotImplementedError


def _fresh_function(fn: Callable, tag: str) -> Callable:
    """A copy of ``fn`` with its own code object, so that ``torch.compile`` compiles it anew."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return types.FunctionType(code, fn.__globals__, f"{fn.__name__}_{tag}", fn.__defaults__, fn.__closure__)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """``t``'s storage as integers of the same width (a bit-for-bit comparison: NaN payloads and signed zeros
    included)."""
    width = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(width)


def determinism_runs(arm: "Arm", i: dict, runs: int) -> dict:
    """Run ``arm.run_once(i)`` ``runs`` times on the same inputs and compare every output bit for bit with the first
    run's: ``{"runs", "bitwise", "differs": {output: [runs that differ]}}``."""
    first = {k: v.detach().clone() for k, v in arm.run_once(i).items()}
    differs: dict = {}
    for r in range(1, runs):
        outs = arm.run_once(i)
        for k, v in outs.items():
            if not torch.equal(_bits(v), _bits(first[k])):
                differs.setdefault(k, []).append(r)
        del outs
    return {"runs": runs, "bitwise": not differs, "differs": differs}


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
        version = dist_version(dist)
        if version is not None:
            rec[dist] = version
    try:
        import deep_gemm

        rec["deep_gemm.__version__"] = getattr(deep_gemm, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    rec["cublas"] = cublas_version()
    rec["libraries"] = loaded_libraries()
    rec["env"] = {k: os.environ.get(k) for k in (
        "CUDA_VISIBLE_DEVICES", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TRITON_F32_DEFAULT",
        "CAKE_PPLX_CACHE_DIR", "CAKE_PPLX_JIT_WORKERS", "CAKE_IMAGE_TAG", "CAKE_IMAGE_SHA256", "CUDA_HOME",
        "LD_PRELOAD", "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "CAKE_BENCH_STACK")}
    rec["tf32"] = {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32,
                   "float32_matmul_precision": torch.get_float32_matmul_precision()}
    return rec


# ----------------------------------------------------------------------------------------------------------------
# Timing


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


def _build(arm: Arm, i: dict, op: str, mode: str, warmup: int, device) -> Callable[[], None]:
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
