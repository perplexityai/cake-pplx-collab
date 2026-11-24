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

Benchmark of the Cake route-weight row scaling training kernels (``cake_pplx.cake_route_weight_scale_det_train``:
``y = bf16(w x)``, and in the backward ``dx = bf16(w dy)`` and the FP32 ``dw = sum over c of dy x``) against the
comparators of the same operation, each when it runs.

Arms (``--arms``; default: every arm; ``cake`` always runs first, since the other arms' outputs are compared with its
outputs):

* ``cake``: ``cake_route_weight_scale_det_train_forward`` / ``_backward`` (one kernel each).
* ``eager``: the PyTorch formula: ``y = (w[:, None] * x.float()).bfloat16()``, ``dx`` the same with ``dy``, and
  ``dw = (dy.float() * x.float()).sum(1)``, which materializes the FP32 ``[rows, H]`` operands and product.
* ``torch_compile``: ``torch.compile`` (Inductor, ``fullgraph=True``, ``dynamic=False``) of the same forward and
  backward formulas, so Inductor fuses them as it chooses.  Each shape calls ``torch._dynamo.reset()`` and compiles
  anew; with ``torch._dynamo.config.fail_on_recompile_limit_hit`` set (when this torch has it), a call that would
  fall back to eager raises instead and prints as unavailable.
* ``cudnn``: a cuDNN frontend graph, when ``cudnn`` (the nvidia-cudnn-frontend Python package) imports.  The forward
  is one pointwise ``mul`` of ``x [1, rows, H]`` (BF16) by ``w [1, rows, 1]`` (FP32, broadcast over ``H``), FP32
  compute, BF16 output.  The backward is first built as one graph: the pointwise ``mul`` of ``dy`` by ``w`` (BF16
  ``dx``) and a ``reduction`` (ADD over ``H``) of the pointwise ``mul`` of ``dy`` by ``x`` (FP32 ``dw [1, rows,
  1]``).  If that graph is refused, two graphs (the ``dx`` graph and the ``dw`` graph, which together read ``dy``
  twice).  Each graph takes the plans of the heuristics modes A and FALLBACK and runs the first plan that builds.
  A graph no plan builds prints as unavailable with the frontend's error, stage by stage.

Every arm runs on identical inputs: ``x`` and ``dy`` BF16 unit normals, ``w`` FP32 ``U[0, 1)`` (the issue's draw),
and allocates its outputs inside the timed call.  Compilation and plan building happen before timing.  Each arm's
outputs are compared with Cake's: whether ``y`` and ``dx`` have the same bits, and the rel-L2 difference of ``dw``.

Stages: ``fwd`` (one forward) and ``bwd`` (one backward from ``x``, ``dy`` and ``w``).

Timers (``--timers``):

* ``cupti``: CUPTI kernel activity (``torch.profiler``) with a cold L2 (``flush.fill_`` of twice the L2 before every
  call, not counted), the summed GPU time of the call's kernels, copies and fills; median of 100 calls after 10
  warm-up calls.
* ``graph``: ``--loop`` calls captured in one CUDA graph and replayed, timed with CUDA events, warm L2; median of
  ``--rounds`` samples after ``--warmup`` samples.

The speedup columns are baseline / Cake per timer (above 1: Cake is faster).  The bandwidth columns are the bytes the
operation must move, each input read once and each output written once (forward ``4 rows H + 4 rows``: ``x`` read,
``y`` written, ``w`` read; backward ``6 rows H + 8 rows``: ``x`` and ``dy`` read, ``dx`` written, ``w`` read, ``dw``
written), over the time, and their share of the device's peak DRAM bandwidth from NVML (:func:`nvml_peak_gbs`:
memory clock x bus width x 2 / 8).  ``peak extra`` is the device memory an arm allocates beyond its outputs during one
call (``torch.cuda.max_memory_allocated``).  The header prints the GPU, ``CUDA_VISIBLE_DEVICES``, the torch device's
UUID beside the UUID of the NVML index the peak is read from, the NVML index that holds the torch device's UUID, and
the library versions; ``--json`` writes everything.

Example::

    python benchmarks/bench_cake_route_weight_scale_det_train.py --json rws.json
    python benchmarks/bench_cake_route_weight_scale_det_train.py --shapes 327680x4096 262144x3584 --arms cake eager
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import functools
import importlib.metadata as md
import json
import os
import statistics
import types
from collections.abc import Callable
from typing import Any, Optional

import torch

from cake_pplx.cake_route_weight_scale_det_train import (
    cake_route_weight_scale_det_train_backward,
    cake_route_weight_scale_det_train_forward,
)

ARMS = ("cake", "eager", "torch_compile", "cudnn")
STAGES = ("fwd", "bwd")
TIMERS = ("cupti", "graph")
# The issue's shapes (327,680 x 4096 and 262,144 x 3584), its receive bounds (655,360 x 4096, 524,160 x 3584) and
# smaller row counts.
DEFAULT_SHAPES = (
    "327680x4096",
    "262144x3584",
    "4097x4096",
    "65536x4096",
    "655360x4096",
    "4097x3584",
    "65536x3584",
    "524160x3584",
)


class Unavailable(Exception):
    """An arm or a stage that cannot run on this shape, with the reason."""


# ----------------------------------------------------------------------------------------------------------------
# NVML: the peak DRAM bandwidth and the GPU identity


class _CtypesNvml:
    """The few NVML calls this benchmark makes, through ``libnvidia-ml.so.1`` with ctypes, under pynvml's names (used
    when pynvml does not import)."""

    NVML_CLOCK_MEM = 2  # nvmlClockType_t NVML_CLOCK_MEM

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")

    def _check(self, status: int, what: str) -> None:
        if status != 0:
            raise RuntimeError(f"NVML {what} returned {status}")

    def nvmlInit(self) -> None:
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")

    def nvmlShutdown(self) -> None:
        self._check(self.lib.nvmlShutdown(), "nvmlShutdown")

    def nvmlDeviceGetCount(self) -> int:
        count = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
        return count.value

    def nvmlDeviceGetHandleByIndex(self, index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(
            self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)),
            "nvmlDeviceGetHandleByIndex_v2",
        )
        return handle

    def nvmlDeviceGetMemoryBusWidth(self, handle: ctypes.c_void_p) -> int:
        width = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetMemoryBusWidth(handle, ctypes.byref(width)), "nvmlDeviceGetMemoryBusWidth")
        return width.value

    def nvmlDeviceGetClockInfo(self, handle: ctypes.c_void_p, clock: int) -> int:
        mhz = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetClockInfo(handle, ctypes.c_int(clock), ctypes.byref(mhz)), "nvmlDeviceGetClockInfo")
        return mhz.value

    def nvmlDeviceGetUUID(self, handle: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(96)
        self._check(self.lib.nvmlDeviceGetUUID(handle, buffer, ctypes.c_uint(96)), "nvmlDeviceGetUUID")
        return buffer.value.decode()


@functools.cache
def _nvml() -> Any:
    """pynvml when it imports and has every call used here, else the ctypes calls of :class:`_CtypesNvml`."""
    try:
        import pynvml  # noqa: PLC0415 - optional
    except ImportError:
        return _CtypesNvml()
    needed = [name for name in vars(_CtypesNvml) if name.startswith("nvml")] + ["NVML_CLOCK_MEM"]
    return pynvml if all(hasattr(pynvml, name) for name in needed) else _CtypesNvml()


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@functools.cache
def nvml_peak_gbs(device: torch.device) -> float:
    """Peak DRAM bandwidth of ``device`` in GB/s from NVML: ``(memory clock MHz x bus width bits x 2) / 8 / 1000``, read
    from the NVML device of index ``device.index or 0``.  A non-CUDA device is refused."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"the NVML peak bandwidth needs a CUDA device, got {device}")
    nvml = _nvml()
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device.index or 0)
        bus_width = nvml.nvmlDeviceGetMemoryBusWidth(handle)
        mem_clock = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
        return (mem_clock * bus_width * 2) / 8 / 1000
    finally:
        nvml.nvmlShutdown()


def nvml_identity(device: torch.device) -> dict[str, Any]:
    """The NVML facts that identify the physical GPU: the index :func:`nvml_peak_gbs` reads, its UUID, bus width and
    memory clock, the torch device's UUID, whether they match, and the NVML index holding the torch device's UUID."""
    device = torch.device(device)
    found: dict[str, Any] = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    torch_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    found["torch_device"] = str(device)
    found["torch_uuid"] = torch_uuid
    try:
        nvml = _nvml()
        nvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        found["nvml_error"] = f"{type(exc).__name__}: {exc}"
        return found
    try:
        index = device.index or 0
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        found["nvml_index"] = index
        found["nvml_uuid"] = _text(nvml.nvmlDeviceGetUUID(handle))
        found["bus_width_bits"] = nvml.nvmlDeviceGetMemoryBusWidth(handle)
        found["mem_clock_mhz"] = nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM)
        found["uuid_match"] = found["nvml_uuid"] == torch_uuid
        found["nvml_index_of_torch_uuid"] = None
        for other in range(nvml.nvmlDeviceGetCount()):
            if _text(nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(other))) == torch_uuid:
                found["nvml_index_of_torch_uuid"] = other
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        found["nvml_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()
    return found


# ----------------------------------------------------------------------------------------------------------------
# Timers


def _cupti_call_ms(fn: Callable[[], object], *, dry_run_iters: int = 10, repeat_iters: int = 100) -> list[float]:
    """Per-call GPU time of ``fn`` in milliseconds, the L2 cache flushed before every call: the summed duration of the
    GPU activities (kernels, copies, fills) that CUPTI records through ``torch.profiler`` between two sentinel kernels.

    Every timed call runs between two launches of ATen's one-cycle spin kernel; the first record after a sentinel is
    the L2 flush and is not counted.  Device records are attributed by position, not by the host operation that
    launched them, so kernels launched through the driver API count too."""
    l2_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    flush = torch.empty(max(2 * l2_bytes, 1 << 20), dtype=torch.uint8, device="cuda")
    for _ in range(dry_run_iters):
        flush.fill_(1)
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        torch.cuda._sleep(1)
        for _ in range(repeat_iters):
            flush.fill_(1)
            fn()
            torch.cuda._sleep(1)
        torch.cuda.synchronize()
    records = sorted(
        (event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA),
        key=lambda event: event.time_range.start,
    )
    call_us: list[float] = []
    current = None
    for event in records:
        if "::spin_kernel(" in event.name:
            if current is not None:
                call_us.append(sum(e.time_range.end - e.time_range.start for e in current[1:]))
            current = []
        elif current is not None:
            current.append(event)
    if len(call_us) != repeat_iters:
        raise RuntimeError(
            f"the profiler returned {len(call_us)} complete calls of {repeat_iters}; CUPTI lost device records"
        )
    if any(us <= 0 for us in call_us):
        raise RuntimeError("CUPTI recorded no GPU activity for a timed call")
    return [us / 1000.0 for us in call_us]


def _events_ms(run: Callable[[], object], loop: int, warmup: int, rounds: int) -> float:
    """Median per-call milliseconds of ``run`` (``loop`` calls), timed with CUDA events, warm L2."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / loop)
    return float(statistics.median(samples))


def _graph_ms(fn: Callable[[], object], loop: int, warmup: int, rounds: int) -> float:
    """Median per-call milliseconds of ``loop`` calls of ``fn`` captured in one CUDA graph and replayed, timed with CUDA
    events, warm L2."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()  # libraries, plans and allocator pools before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(loop):
            fn()
    try:
        return _events_ms(graph.replay, loop, warmup, rounds)
    finally:
        del graph
        torch.cuda.synchronize()


def time_stage(fn: Callable[[], object], timers: tuple[str, ...], args: argparse.Namespace) -> dict[str, Any]:
    """``{timer: milliseconds or the reason it did not run}`` of one stage call ``fn``."""
    found: dict[str, Any] = {}
    for timer in timers:
        try:
            if timer == "cupti":
                found[timer] = float(statistics.median(_cupti_call_ms(fn)))
            elif timer == "graph":
                found[timer] = _graph_ms(fn, args.loop, args.warmup, args.rounds)
        except Exception as exc:  # noqa: BLE001 - a timer that cannot run this call is reported, not fatal
            if isinstance(exc, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
            found[timer] = f"unavailable: {type(exc).__name__}: {str(exc)[:240]}"
    return found


def peak_extra_bytes(fn: Callable[[], object]) -> int:
    """Device memory one call of ``fn`` allocates beyond what it returns (its outputs), at its peak."""
    torch.cuda.synchronize()
    start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - start
    extra = torch.cuda.max_memory_allocated() - start - kept
    del out
    return int(extra)


# ----------------------------------------------------------------------------------------------------------------
# Inputs and bytes


def make_inputs(rows: int, width: int, seed: int) -> types.SimpleNamespace:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, generator=gen)
    dy = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.rand(rows, device="cuda", generator=gen)
    return types.SimpleNamespace(R=rows, H=width, x=x, dy=dy, w=w)


def dram_bytes(rows: int, width: int, stage: str) -> int:
    """The bytes a stage must move when each input is read once and each output written once."""
    return {"fwd": 4 * rows * width + 4 * rows, "bwd": 6 * rows * width + 8 * rows}[stage]


# ----------------------------------------------------------------------------------------------------------------
# Arms: ``{"fwd": () -> y, "bwd": () -> (dx, dw), "version": str, "notes": dict}``


def _formula_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (w[:, None] * x.float()).bfloat16()


def _formula_backward(x: torch.Tensor, dy: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (w[:, None] * dy.float()).bfloat16(), (dy.float() * x.float()).sum(1)


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def cake_arm(t: types.SimpleNamespace) -> dict[str, Any]:
    return {
        "fwd": lambda: cake_route_weight_scale_det_train_forward(t.x, t.w),
        "bwd": lambda: cake_route_weight_scale_det_train_backward(t.x, t.dy, t.w),
        "version": f"cake-pplx-route-weight-scale {_version('cake-pplx-route-weight-scale')}",
        "notes": {},
    }


def eager_arm(t: types.SimpleNamespace) -> dict[str, Any]:
    return {
        "fwd": lambda: _formula_forward(t.x, t.w),
        "bwd": lambda: _formula_backward(t.x, t.dy, t.w),
        "version": f"torch {torch.__version__}",
        "notes": {},
    }


def compile_arm(t: types.SimpleNamespace) -> dict[str, Any]:
    import torch._dynamo  # noqa: PLC0415

    torch._dynamo.reset()
    config = torch._dynamo.config
    guarded = hasattr(config, "fail_on_recompile_limit_hit")
    if guarded:
        config.fail_on_recompile_limit_hit = True
    fwd = torch.compile(_formula_forward, fullgraph=True, dynamic=False)
    bwd = torch.compile(_formula_backward, fullgraph=True, dynamic=False)

    def guard(call: Callable[[], Any], stage: str) -> Callable[[], Any]:
        def run():
            try:
                return call()
            except Exception as exc:
                if type(exc).__module__.startswith(("torch._dynamo", "torch._inductor")):
                    raise Unavailable(f"torch.compile does not run {stage}: {type(exc).__name__}: {str(exc)[:240]}") from None
                raise

        return run

    return {
        "fwd": guard(lambda: fwd(t.x, t.w), "fwd"),
        "bwd": guard(lambda: bwd(t.x, t.dy, t.w), "bwd"),
        "version": f"torch {torch.__version__}, triton {_version('triton')}",
        "notes": {
            "fullgraph": True,
            "dynamic": False,
            "dynamo_reset": True,
            "fail_on_recompile_limit_hit": guarded,
        },
    }


class CudnnGraphs:
    """The cuDNN frontend graphs of one shape: the forward, and the backward as one graph or as a two-graph composite,
    each run with the first plan that builds (heuristics modes A, then FALLBACK)."""

    def __init__(self, t: types.SimpleNamespace) -> None:
        try:
            import cudnn  # noqa: PLC0415 - optional
        except Exception as exc:  # noqa: BLE001
            raise Unavailable(f"the cuDNN frontend does not import: {type(exc).__name__}: {exc}") from None
        self.c = cudnn
        self.t = t
        self.handle = cudnn.create_handle()
        self.version = f"cudnn frontend {getattr(cudnn, '__version__', None)}, backend {cudnn.backend_version()}"
        self.errors: dict[str, str] = {}
        self.forward_graph = self._try("fwd", self._forward_ops)
        self.backward_kind: Optional[str] = None
        self.backward_graphs: Optional[list] = None
        one = self._try("bwd", self._backward_ops(dx=True, dw=True))
        if one is not None:
            self.backward_graphs, self.backward_kind = [one], "one graph: dx pointwise and the dw reduction"
        else:
            parts = [self._try("bwd dx", self._backward_ops(dx=True, dw=False))]
            parts.append(self._try("bwd dw", self._backward_ops(dx=False, dw=True)))
            if all(part is not None for part in parts):
                self.backward_graphs, self.backward_kind = parts, "two graphs: dx pointwise, dw pointwise + reduction"

    def _matrix(self, g, name: str) -> Any:
        R, H = self.t.R, self.t.H
        return g.tensor(name=name, dim=[1, R, H], stride=[R * H, H, 1], data_type=self.c.data_type.BFLOAT16)

    def _rows(self, g, name: str) -> Any:
        R = self.t.R
        return g.tensor(name=name, dim=[1, R, 1], stride=[R, 1, 1], data_type=self.c.data_type.FLOAT)

    def _bf16_output(self, value) -> Any:
        R, H = self.t.R, self.t.H
        return value.set_output(True).set_data_type(self.c.data_type.BFLOAT16).set_dim([1, R, H]).set_stride([R * H, H, 1])

    def _forward_ops(self, g) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = {"x": self._matrix(g, "x"), "w": self._rows(g, "w")}
        y = g.mul(inputs["x"], inputs["w"])
        return inputs, {"y": self._bf16_output(y)}

    def _backward_ops(self, *, dx: bool, dw: bool):
        def ops(g) -> tuple[dict[str, Any], dict[str, Any]]:
            c, R = self.c, self.t.R
            inputs = {"dy": self._matrix(g, "dy")}
            outputs = {}
            if dx:
                inputs["w"] = self._rows(g, "w")
                outputs["dx"] = self._bf16_output(g.mul(inputs["dy"], inputs["w"]))
            if dw:
                inputs["x"] = self._matrix(g, "x")
                total = g.reduction(g.mul(inputs["dy"], inputs["x"]), mode=c.reduction_mode.ADD)
                outputs["dw"] = total.set_output(True).set_data_type(c.data_type.FLOAT).set_dim([1, R, 1]).set_stride([R, 1, 1])
            return inputs, outputs

        return ops

    def _try(self, what: str, ops) -> Optional[types.SimpleNamespace]:
        """The graph of ``ops`` built with its first plan, or None with the frontend's error recorded under ``what``."""
        c = self.c
        try:
            g = c.pygraph(
                io_data_type=c.data_type.BFLOAT16,
                intermediate_data_type=c.data_type.FLOAT,
                compute_data_type=c.data_type.FLOAT,
                handle=self.handle,
            )
            inputs, outputs = ops(g)
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([c.heur_mode.A, c.heur_mode.FALLBACK])
            g.check_support()
            g.build_plans()
            workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
        except Exception as exc:  # noqa: BLE001 - a graph no plan builds is reported with its error
            self.errors[what] = f"{type(exc).__name__}: {str(exc)[:400]}"
            return None
        return types.SimpleNamespace(graph=g, inputs=inputs, outputs=outputs, workspace=workspace)

    def _execute(self, built: types.SimpleNamespace, outputs: dict[str, torch.Tensor]) -> None:
        values = {"x": self.t.x, "dy": self.t.dy, "w": self.t.w}
        pack = {tensor: values[name] for name, tensor in built.inputs.items()}
        pack.update({tensor: outputs[name] for name, tensor in built.outputs.items()})
        self.c.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        built.graph.execute(pack, built.workspace, handle=self.handle)

    def forward(self) -> torch.Tensor:
        if self.forward_graph is None:
            raise Unavailable(f"no cuDNN forward graph builds: {self.errors.get('fwd')}")
        y = torch.empty(self.t.R, self.t.H, dtype=torch.bfloat16, device="cuda")
        self._execute(self.forward_graph, {"y": y})
        return y

    def backward(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.backward_graphs is None:
            raise Unavailable(f"no cuDNN backward graph builds: {self.errors}")
        outputs = {
            "dx": torch.empty(self.t.R, self.t.H, dtype=torch.bfloat16, device="cuda"),
            "dw": torch.empty(self.t.R, dtype=torch.float32, device="cuda"),
        }
        for built in self.backward_graphs:
            self._execute(built, outputs)
        return outputs["dx"], outputs["dw"]


def cudnn_arm(t: types.SimpleNamespace) -> dict[str, Any]:
    graphs = CudnnGraphs(t)
    return {
        "fwd": graphs.forward,
        "bwd": graphs.backward,
        "version": graphs.version,
        "notes": {"backward": graphs.backward_kind, "errors": graphs.errors},
    }


MAKERS = {"cake": cake_arm, "eager": eager_arm, "torch_compile": compile_arm, "cudnn": cudnn_arm}


# ----------------------------------------------------------------------------------------------------------------
# One shape


def _check(stage: str, found: Any, cake: Any) -> dict[str, Any]:
    """How an arm's outputs compare with Cake's: ``y`` / ``dx`` bit for bit, ``dw`` as a rel-L2 difference."""
    if stage == "fwd":
        return {"y_same_bits": bool(torch.equal(found.view(torch.int16), cake.view(torch.int16)))}
    (dx, dw), (cake_dx, cake_dw) = found, cake
    dw = dw.reshape(-1)
    norm = cake_dw.double().norm()
    return {
        "dx_same_bits": bool(torch.equal(dx.view(torch.int16), cake_dx.view(torch.int16))),
        "dw_rel_l2_vs_cake": float(((dw.double() - cake_dw.double()).norm() / norm).item()) if norm > 0 else None,
    }


def bench_shape(rows: int, width: int, args: argparse.Namespace, peak_gbs: Optional[float]) -> dict[str, Any]:
    t = make_inputs(rows, width, args.seed)
    result: dict[str, Any] = {"rows": rows, "H": width, "arms": {}}
    cake_out: dict[str, Any] = {}
    for name in args.arms:
        record: dict[str, Any] = {}
        result["arms"][name] = record
        try:
            arm = MAKERS[name](t)
        except Unavailable as exc:
            record["unavailable"] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 - a comparator that cannot be set up is reported, not fatal
            if name == "cake":
                raise
            record["unavailable"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            continue
        record["version"], record["notes"] = arm["version"], arm["notes"]
        for stage in args.stages:
            entry: dict[str, Any] = {"bytes": dram_bytes(rows, width, stage)}
            record[stage] = entry
            try:
                out = arm[stage]()  # compilation, plans, libraries
                torch.cuda.synchronize()
                if name == "cake":
                    cake_out[stage] = out
                elif stage in cake_out:
                    entry["check"] = _check(stage, out, cake_out[stage])
                del out
                entry["peak_extra_bytes"] = peak_extra_bytes(arm[stage])
                entry["ms"] = time_stage(arm[stage], tuple(args.timers), args)
            except Unavailable as exc:
                entry["unavailable"] = str(exc)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                entry["unavailable"] = f"out of memory: {str(exc)[:200]}"
            except Exception as exc:  # noqa: BLE001 - a comparator that fails at run time is reported, not fatal
                if name == "cake":
                    raise
                entry["unavailable"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            for timer, ms in entry.get("ms", {}).items():
                if isinstance(ms, float):
                    gbs = entry["bytes"] / ms / 1e6
                    entry.setdefault("gbs", {})[timer] = gbs
                    if peak_gbs:
                        entry.setdefault("bw_util", {})[timer] = gbs / peak_gbs
        del arm
        torch.cuda.empty_cache()
    for stage in args.stages:
        base = result["arms"].get("cake", {}).get(stage, {}).get("ms", {})
        for name, record in result["arms"].items():
            entry = record.get(stage)
            if name == "cake" or not isinstance(entry, dict):
                continue
            for timer, ms in entry.get("ms", {}).items():
                if isinstance(ms, float) and isinstance(base.get(timer), float):
                    entry.setdefault("speedup", {})[timer] = ms / base[timer]
    del t, cake_out
    torch.cuda.empty_cache()
    return result


def format_shape(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    lines = []
    timers = tuple(args.timers)
    head = "".join(f"{timer + ' us':>13}{'vs cake':>9}{'GB/s':>8}{'BW%':>7}" for timer in timers)
    for stage in args.stages:
        nbytes = dram_bytes(result["rows"], result["H"], stage)
        lines.append(f"## {result['rows']} x {result['H']} {stage} ({nbytes / 1e9:.3f} GB moved)")
        lines.append(f"{'arm':<14}{head}{'peak extra MiB':>16}  check")
        for name, record in result["arms"].items():
            if "unavailable" in record:
                lines.append(f"{name:<14}unavailable: {record['unavailable']}")
                continue
            entry = record.get(stage, {})
            if "unavailable" in entry:
                lines.append(f"{name:<14}unavailable: {entry['unavailable']}")
                continue
            cells = ""
            for timer in timers:
                ms = entry.get("ms", {}).get(timer)
                if not isinstance(ms, float):
                    cells += f"{'n/a':>13}{'':>9}{'':>8}{'':>7}"
                    continue
                speedup = entry.get("speedup", {}).get(timer)
                util = entry.get("bw_util", {}).get(timer)
                cells += (
                    f"{ms * 1e3:>13.2f}{(f'{speedup:.2f}x' if speedup else '-'):>9}"
                    f"{entry['gbs'][timer]:>8.0f}{(f'{util * 100:.1f}' if util else '-'):>7}"
                )
            extra = entry.get("peak_extra_bytes")
            check = json.dumps(entry.get("check", {}))
            lines.append(f"{name:<14}{cells}{(f'{extra / 2**20:.1f}' if extra is not None else '-'):>16}  {check}")
            for timer, ms in entry.get("ms", {}).items():
                if not isinstance(ms, float):
                    lines.append(f"{'':<14}{timer}: {ms}")
        for name, record in result["arms"].items():
            errors = record.get("notes", {}).get("errors")
            if errors:
                lines.append(f"{'':<14}{name} graph errors: {json.dumps(errors)}")
    return lines


# ----------------------------------------------------------------------------------------------------------------
# Identity and main


def identity(device: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(device)
    dists = ("cake-pplx-route-weight-scale", "apache-tvm-ffi", "torch", "triton", "nvidia-cudnn-frontend", "pynvml", "nvidia-ml-py")
    return {
        "gpu": p.name,
        "cc": list(torch.cuda.get_device_capability(device)),
        "sms": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "versions": {d: _version(d) for d in dists},
        "nvml": nvml_identity(device),
    }


def _shape(text: str) -> tuple[int, int]:
    rows, _, width = text.lower().partition("x")
    return int(rows), int(width)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Times the Cake route-weight row scaling training kernels and the comparators of the same operation."
    )
    parser.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES), help="ROWSxH, H 4096 or 3584")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    parser.add_argument("--timers", nargs="+", default=list(TIMERS), choices=list(TIMERS))
    parser.add_argument("--loop", type=int, default=20, help="calls per graph sample")
    parser.add_argument("--warmup", type=int, default=5, help="graph samples before timing")
    parser.add_argument("--rounds", type=int, default=15, help="graph samples timed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    args.arms = list(dict.fromkeys(["cake", *args.arms]))  # cake first: the other arms are compared with its outputs
    device = torch.device("cuda", torch.cuda.current_device())
    ident = identity(device)
    try:
        peak = nvml_peak_gbs(device)
    except Exception as exc:  # noqa: BLE001 - the bandwidth share is left out, with the reason
        peak = None
        ident["nvml_peak_error"] = f"{type(exc).__name__}: {exc}"
    ident["nvml_peak_gbs"] = peak
    nvml = ident["nvml"]
    print(
        f"# {ident['gpu']} (sm_{ident['cc'][0]}{ident['cc'][1]}, {ident['sms']} SMs), torch {ident['torch']} "
        f"(CUDA {ident['torch_cuda']}); versions {json.dumps(ident['versions'])}"
    )
    print(
        f"# CUDA_VISIBLE_DEVICES={nvml.get('cuda_visible_devices')}; torch {nvml.get('torch_device')} = "
        f"{nvml.get('torch_uuid')}; NVML index {nvml.get('nvml_index')} = {nvml.get('nvml_uuid')} "
        f"(match {nvml.get('uuid_match')}); NVML index of the torch device: {nvml.get('nvml_index_of_torch_uuid')}"
        + (f"; NVML error: {nvml['nvml_error']}" if "nvml_error" in nvml else "")
    )
    if nvml.get("uuid_match") is False:
        print("# WARNING: the NVML index the peak is read from is another GPU than the torch device")
    print(
        f"# NVML peak DRAM bandwidth: {peak if peak is None else round(peak, 1)} GB/s (memory clock "
        f"{nvml.get('mem_clock_mhz')} MHz x bus {nvml.get('bus_width_bits')} bits x 2 / 8)"
        + (f"; error: {ident['nvml_peak_error']}" if "nvml_peak_error" in ident else "")
    )
    print(
        "# times in us per call; vs cake = baseline / cake (above 1: cake is faster); cupti = cold L2 GPU time, median "
        f"of 100 calls; graph = warm L2, {args.loop} calls per CUDA graph replay, median of {args.rounds}; GB/s = "
        "bytes moved / time; BW% = GB/s / NVML peak"
    )
    results = []
    for text in args.shapes:
        rows, width = _shape(text)
        result = bench_shape(rows, width, args, peak)
        results.append(result)
        for line in format_shape(result, args):
            print(line, flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"identity": ident, "args": vars(args), "shapes": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
