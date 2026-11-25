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

Benchmark of the Cake SiTU-GLU training kernels (``cake_pplx.cake_situ_glu_det_train``,
``h = bf16(beta tanh(g / beta) sigmoid(g) lb tanh(u / lb))`` and its backward) against the comparators of the same
operation, each when it imports.

Arms (``--arms``; default: every arm; ``cake`` always runs first, since the other arms' outputs are compared with its
outputs):

* ``cake``: ``cake_situ_glu_det_train_forward`` / ``_backward`` (one kernel each), and the autograd entry
  ``cake_situ_glu_det`` for the ``ckpt`` stage.
* ``cudnn``: cuDNN frontend pointwise graphs of the SiTU activation, op for op as the frontend's gated MLP builds its
  ``situ`` activation and its derivative: ``gate_act = beta tanh(g / beta) sigmoid(g)``, ``up_act = lb tanh(u / lb)``,
  ``h = gate_act up_act``; ``d_gate = dh up_act ((1 - tg^2) s + beta tg s (1 - s))`` and
  ``d_up = dh gate_act (1 - tu^2)`` in one graph.  BF16 inputs and outputs, FP32 intermediates and compute; ``beta``,
  ``lb`` and 1 are runtime scalars (pass-by-value tensors on a frontend without runtime scalars).  Every tensor is
  declared with its row stride, so the merged layout is read in place and its gradient written into the halves of one
  ``[R, 2I]`` tensor.  Each graph takes the plans of the heuristics modes A and FALLBACK, builds all of them
  (``build_plan_policy.ALL``) and keeps the fastest: each plan's ``--plan-calls`` calls captured in one CUDA graph and
  replayed (CUDA events, warm L2), or timed eagerly when the capture fails.  ``--cudnn-first-plan`` builds only the
  first plan that builds and runs it (when building every plan takes too long).  The arm's ``plans`` record holds the
  candidates, each one's time or error, the chosen plan and the build time.  An autograd function of the two graphs
  serves ``ckpt``.
* ``torch_compile``: ``torch.compile`` (Inductor, ``fullgraph=True``, ``dynamic=False``) of the ``eager`` arm's forward
  and backward formulas.  Every row resets Dynamo and compiles fresh code objects, one per calling context (the direct
  forward and backward, and the forward and backward inside the autograd function), with
  ``torch._dynamo.config.fail_on_recompile_limit_hit`` set where it exists, so that a call that would run eagerly
  raises and its stage prints as unavailable.  A stage times only when each code object it runs holds a compiled
  Dynamo cache entry after its untimed call (when this torch can tell); the graphs Dynamo compiles while the timers run
  are recorded.
* ``eager``: the issue's FP32 formulas in eager PyTorch (``1 - tg^2`` and ``1 - s`` as written); in the merged layout
  the backward returns the ``torch.cat`` of the two gradient halves (a separate copy).

Every arm runs on identical inputs: ``gate`` and ``up`` ``2 x`` unit normals and ``dh`` a unit normal (BF16), the
scale of the issue's yardstick, and the scalars ``--beta`` and ``--lb`` (default 4.0 and 25.0, the model's).
``--points`` lists ``R:I`` pairs (default: the issue's 262,144 x 3072, 16,380 x 6144 and 16,380 x 33,792, and three
more row counts at 3072); ``--layouts``: ``merged`` (one ``[R, 2I]`` input and one ``[R, 2I]`` gradient) and
``split`` (two ``[R, I]`` inputs and gradients).  Every arm allocates its outputs inside the timed call; compilation,
graph and plan building happen before timing.  Each arm's outputs are compared with Cake's (the largest absolute
difference and the share of elements whose bits differ), and each arm's outputs on the first ``--acc-rows`` rows with a
float64 oracle (relative L2 error).

Stages: ``fwd`` (one forward), ``bwd`` (one backward from ``gate``, ``up`` and ``dh``; the activation is recomputed),
and ``ckpt`` (an autograd step under ``torch.utils.checkpoint(..., use_reentrant=False, preserve_rng_state=False)``:
the forward, then the backward, which recomputes the forward first: two forwards and one backward, the issue's update
under activation checkpointing).  Every stage runs once untimed before its timers.

Timers (``--timers``):

* ``graph``: ``--calls`` calls captured in one CUDA graph (after three warm-up calls on a side stream) and replayed
  ``--trials`` times after ``--warmup`` replays, each replay timed with CUDA events (warm L2).  The per-call time is
  the trimmed mean of the replays (``--trim`` of the samples dropped at each end); the median, the smallest and the
  largest are recorded too.
* ``cupti``: CUPTI kernel activity (``torch.profiler``) with a cold L2 (``flush.fill_`` of twice the L2 before every
  call, not counted): the summed GPU time of the call's kernels, copies and fills; median of 100 calls after 10
  warm-up calls.

Bandwidth: each input read once and each output written once, over ``R I`` elements: ``fwd`` 6 bytes per element
(``gate``, ``up``, ``h``), ``bwd`` 10 (``gate``, ``up``, ``dh``, ``d_gate``, ``d_up``) and ``ckpt`` 22 (two forwards
and a backward).  ``GB/s`` is these bytes over each timer's time, and ``util`` that over the device's peak DRAM
bandwidth from NVML (:func:`peak_gbps`: memory clock x bus width x 2 / 8 / 1000, read once per device).  The header
prints ``CUDA_VISIBLE_DEVICES``, the torch device's UUID and the UUID of the NVML device at the same index (the one
the peak reads), and whether they match.  The ratio columns are baseline / Cake (above 1: Cake is faster) per timer.
``--json`` writes everything.

Example::

    python benchmarks/bench_cake_situ_glu_det_train.py --json situ_glu.json
    python benchmarks/bench_cake_situ_glu_det_train.py --points 262144:3072 --layouts merged --arms cake cudnn
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.metadata as md
import json
import math
import os
import statistics
import subprocess
import time
import types
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.utils.checkpoint

from cake_pplx.cake_situ_glu_det_train import (
    cake_situ_glu_det,
    cake_situ_glu_det_train_backward,
    cake_situ_glu_det_train_forward,
)

ARMS = ("cake", "cudnn", "torch_compile", "eager")
STAGES = ("fwd", "bwd", "ckpt")
TIMERS = ("graph", "cupti")
POINTS = ("262144:3072", "16380:6144", "16380:33792", "16380:3072", "65536:3072", "1024:3072")
# DRAM bytes per element of R x I: each input read once, each output written once (BF16).
BYTES_PER_ELEMENT = {"fwd": 6, "bwd": 10, "ckpt": 22}
# cuDNN plan selection: warm-up replays and timed replays of every candidate plan.
PLAN_WARMUP, PLAN_ROUNDS = 1, 5


class Unavailable(Exception):
    """An arm or a stage that cannot run on this row, with the reason."""


# ----------------------------------------------------------------------------------------------------------------
# NVML


@functools.cache
def peak_gbps(device: torch.device) -> float:
    """Peak DRAM bandwidth of ``device`` in GB/s from NVML: memory clock (MHz) x bus width (bits) x 2 / 8 / 1000, of the
    NVML device with the torch device's UUID (so a ``CUDA_VISIBLE_DEVICES`` remap reads the same GPU)."""
    if device.type != "cuda":
        raise ValueError(f"the NVML peak bandwidth needs a CUDA device, got {device}")
    import pynvml  # noqa: PLC0415 - optional, imported on use

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(_uuid(torch.cuda.get_device_properties(device).uuid))
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        return (mem_clock * bus_width * 2) / 8 / 1000
    finally:
        pynvml.nvmlShutdown()


def _uuid(value: Any) -> str:
    text = value.decode() if isinstance(value, bytes) else str(value)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def nvml_mapping(device: torch.device) -> dict[str, Any]:
    """``CUDA_VISIBLE_DEVICES``, the torch device's index and UUID, every NVML device's index and UUID, and the NVML
    device with the torch device's UUID (the one :func:`peak_gbps` reads): its index, UUID, bus width and memory clock,
    and whether its UUID is the torch device's."""
    found: dict[str, Any] = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_index": device.index,
        "torch_uuid": _uuid(torch.cuda.get_device_properties(device).uuid),
    }
    try:
        import pynvml  # noqa: PLC0415 - optional, imported on use
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        found["error"] = f"pynvml does not import: {type(exc).__name__}: {exc}"
        return found
    pynvml.nvmlInit()
    try:
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            try:
                devices.append([index, _uuid(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index)))])
            except Exception as exc:  # noqa: BLE001 - a device this process cannot open
                devices.append([index, f"unavailable: {type(exc).__name__}"])
        found["nvml_devices"] = devices
        handle = pynvml.nvmlDeviceGetHandleByUUID(found["torch_uuid"])
        found["nvml_index"] = int(pynvml.nvmlDeviceGetIndex(handle))
        found["nvml_uuid"] = _uuid(pynvml.nvmlDeviceGetUUID(handle))
        found["bus_width_bits"] = int(pynvml.nvmlDeviceGetMemoryBusWidth(handle))
        found["mem_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    finally:
        pynvml.nvmlShutdown()
    found["uuid_match"] = found["nvml_uuid"] == found["torch_uuid"]
    return found


# ----------------------------------------------------------------------------------------------------------------
# Timers


def trimmed_mean(samples: list[float], trim: float) -> float:
    """The mean of ``samples`` without the ``trim`` share of the smallest and of the largest."""
    ordered = sorted(samples)
    cut = int(len(ordered) * trim)
    kept = ordered[cut : len(ordered) - cut] or ordered
    return float(statistics.fmean(kept))


def _replay_samples_ms(fn: Callable[[], object], calls: int, warmup: int, trials: int) -> list[float]:
    """Per-call milliseconds of ``calls`` calls of ``fn`` captured in one CUDA graph, one sample per replay."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()  # libraries, plans and allocator pools before the capture
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    del graph
    return samples


def _events_ms(fn: Callable[[], object], calls: int, warmup: int, trials: int) -> list[float]:
    """Per-call milliseconds of ``calls`` back-to-back eager calls of ``fn``, one sample per batch (warm L2)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / calls)
    return samples


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


def time_stage(fn: Callable[[], object], args: argparse.Namespace) -> dict[str, Any]:
    """``{timer: {"us": ..., ...} or the reason it did not run}`` of one stage call ``fn``."""
    found: dict[str, Any] = {}
    for timer in args.timers:
        try:
            if timer == "graph":
                samples = _replay_samples_ms(fn, args.calls, args.warmup, args.trials)
                found[timer] = {
                    "us": trimmed_mean(samples, args.trim) * 1e3,
                    "statistic": f"trimmed mean ({args.trim:g} of each end) of {len(samples)} replays",
                    "median_us": statistics.median(samples) * 1e3,
                    "min_us": min(samples) * 1e3,
                    "max_us": max(samples) * 1e3,
                    "calls_per_replay": args.calls,
                }
            elif timer == "cupti":
                samples = _cupti_call_ms(fn)
                found[timer] = {"us": statistics.median(samples) * 1e3, "statistic": f"median of {len(samples)}"}
        except Unavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a timer that cannot run this call is reported, not fatal
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            found[timer] = f"unavailable: {type(exc).__name__}: {str(exc)[:240]}"
    return found


# ----------------------------------------------------------------------------------------------------------------
# Inputs and the oracle


def make_inputs(rows: int, width: int, layout: str, beta: float, lb: float, seed: int) -> types.SimpleNamespace:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = torch.randn(rows, 2 * width, device="cuda", dtype=torch.bfloat16, generator=gen).mul_(2.0)
    dh = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, generator=gen)
    x = types.SimpleNamespace(R=rows, I=width, layout=layout, dh=dh)
    x.beta = float(torch.tensor(beta, dtype=torch.float32))
    x.lb = float(torch.tensor(lb, dtype=torch.float32))
    if layout == "merged":
        x.gate_up = gate_up
        x.gate, x.up = gate_up[:, :width], gate_up[:, width:]
        x.first, x.second = gate_up, None
    elif layout == "split":
        x.gate, x.up = gate_up[:, :width].contiguous(), gate_up[:, width:].contiguous()
        x.first, x.second = x.gate, x.up
        del gate_up
    else:
        raise ValueError(f"layout is merged or split, got {layout!r}")
    return x


def oracle(x: types.SimpleNamespace, rows: int) -> dict[str, torch.Tensor]:
    """The float64 ``h``, ``d_gate`` and ``d_up`` of the first ``rows`` rows of ``x``."""
    g, u, d = x.gate[:rows].double(), x.up[:rows].double(), x.dh[:rows].double()
    s = torch.sigmoid(g)
    a = x.beta * torch.tanh(g / x.beta) * s
    up = x.lb * torch.tanh(u / x.lb)
    return {
        "h": a * up,
        "d_gate": d * up * (s / torch.cosh(g / x.beta) ** 2 + a * torch.sigmoid(-g)),
        "d_up": d * a / torch.cosh(u / x.lb) ** 2,
    }


def _halves(first: torch.Tensor, second: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if second is None:
        width = first.shape[1] // 2
        return first[:, :width], first[:, width:]
    return first, second


def _outputs(h: Optional[torch.Tensor], grads: Optional[tuple]) -> dict[str, torch.Tensor]:
    """``h``, ``d_gate`` and ``d_up`` from an arm's results, the ones it has."""
    found = {}
    if h is not None:
        found["h"] = h
    if grads is not None:
        found["d_gate"], found["d_up"] = _halves(*grads)
    return found


# ----------------------------------------------------------------------------------------------------------------
# Arms: ``forward(x) -> h``, ``backward(x) -> (d_first, d_up or None)`` (a merged gradient as ``d_first``), and
# ``autograd(first, second) -> h`` or None


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def cake_arm(x: types.SimpleNamespace) -> dict[str, Any]:
    return {
        "forward": lambda x: cake_situ_glu_det_train_forward(x.first, x.second, beta=x.beta, lb=x.lb),
        "backward": lambda x: cake_situ_glu_det_train_backward(x.dh, x.first, x.second, beta=x.beta, lb=x.lb),
        "autograd": lambda first, second, beta=x.beta, lb=x.lb: cake_situ_glu_det(first, second, beta=beta, lb=lb),
        "version": f"cake-pplx-situ-glu {_version('cake-pplx-situ-glu')}",
        "workspace_bytes": 0,
    }


def _formula_forward(g: torch.Tensor, u: torch.Tensor, beta: float, lb: float) -> torch.Tensor:
    """``h = bf16(beta tanh(g / beta) sigmoid(g) lb tanh(u / lb))`` in FP32, one rounding."""
    gf, uf = g.float(), u.float()
    return (beta * torch.tanh(gf / beta) * torch.sigmoid(gf) * (lb * torch.tanh(uf / lb))).to(torch.bfloat16)


def _formula_backward(dh: torch.Tensor, g: torch.Tensor, u: torch.Tensor, beta: float, lb: float, merged: bool):
    """``(d_first, d_up or None)`` in FP32 as the issue writes it, each BF16 output rounded once; a merged layout's
    gradient as one ``[R, 2I]`` tensor, the ``torch.cat`` of the two halves."""
    gf, uf, d = g.float(), u.float(), dh.float()
    s, tg, tu = torch.sigmoid(gf), torch.tanh(gf / beta), torch.tanh(uf / lb)
    a = beta * tg * s
    d_gate = (d * (lb * tu) * ((1 - tg * tg) * s + a * (1 - s))).to(torch.bfloat16)
    d_up = (d * a * (1 - tu * tu)).to(torch.bfloat16)
    if merged:
        return torch.cat([d_gate, d_up], dim=1), None
    return d_gate, d_up


def _fresh_function(fn: Callable, tag: str) -> Callable:
    """A copy of ``fn`` with its own code object, so that ``torch.compile`` compiles it anew (Dynamo keeps its compiled
    code and guards per code object)."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)


def _reset_dynamo() -> dict[str, Any]:
    """Clears Dynamo's compiled code and makes a call past the recompile limit raise instead of running eagerly;
    returns the settings for the arm's record."""
    import torch._dynamo  # noqa: PLC0415

    torch._dynamo.reset()
    config = torch._dynamo.config
    raises = hasattr(config, "fail_on_recompile_limit_hit")
    if raises:
        config.fail_on_recompile_limit_hit = True
    return {
        "code": "fresh code objects per row and calling context (fwd, bwd, autograd fwd, autograd bwd)",
        "fullgraph": True,
        "dynamic": False,
        "fail_on_recompile_limit_hit": True if raises else "unavailable: torch._dynamo.config has no such setting",
    }


def _compiled(fn: Callable, tag: str) -> Callable:
    """``torch.compile`` (``fullgraph=True``, ``dynamic=False``) of a fresh copy of ``fn``, with that copy's code object
    as ``.code``.  A Dynamo or Inductor error raises :class:`Unavailable`, so its stage prints as unavailable."""
    fresh = _fresh_function(fn, tag)
    compiled = torch.compile(fresh, fullgraph=True, dynamic=False)

    def call(*args):
        try:
            return compiled(*args)
        except Exception as exc:
            if type(exc).__module__.startswith(("torch._dynamo", "torch._inductor")):
                raise Unavailable(f"torch.compile does not run {tag}: {type(exc).__name__}: {str(exc)[:240]}") from None
            raise

    call.code = fresh.__code__
    return call


def _cache_entries(code: types.CodeType) -> Optional[int]:
    """The number of compiled entries Dynamo holds for ``code``, or None when this torch has no lookup."""
    import torch._dynamo.eval_frame  # noqa: PLC0415

    lookup = getattr(torch._dynamo.eval_frame, "_debug_get_cache_entry_list", None)
    return None if lookup is None else len(lookup(code))


def _dynamo_graphs() -> int:
    """The number of graphs Dynamo has compiled in this process."""
    import torch._dynamo.utils  # noqa: PLC0415

    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def formula_arm(x: types.SimpleNamespace, compiled: bool) -> dict[str, Any]:
    fwd = autograd_fwd = _formula_forward
    bwd = autograd_bwd = _formula_backward
    found: dict[str, Any] = {"notes": {}}
    if compiled:
        found["notes"]["compile"] = _reset_dynamo()
        row = f"R{x.R}_I{x.I}_{x.layout}"
        fwd = _compiled(_formula_forward, f"{row}_fwd")
        bwd = _compiled(_formula_backward, f"{row}_bwd")
        autograd_fwd = _compiled(_formula_forward, f"{row}_autograd_fwd")
        autograd_bwd = _compiled(_formula_backward, f"{row}_autograd_bwd")
        found["codes"] = {
            "fwd": fwd.code,
            "bwd": bwd.code,
            "autograd fwd": autograd_fwd.code,
            "autograd bwd": autograd_bwd.code,
        }
    if x.second is None:
        where = "inside the compiled backward graph" if compiled else "a separate copy"
        found["notes"]["merged_gradient"] = f"torch.cat of d_gate and d_up ({where})"

    class Formula(torch.autograd.Function):
        @staticmethod
        def forward(ctx, first, second, beta, lb):
            ctx.save_for_backward(first, second)
            ctx.beta, ctx.lb = beta, lb
            return autograd_fwd(*_halves(first, second), beta, lb)

        @staticmethod
        def backward(ctx, dh):
            first, second = ctx.saved_tensors
            d_first, d_up = autograd_bwd(dh, *_halves(first, second), ctx.beta, ctx.lb, second is None)
            return d_first, d_up, None, None

    found.update(
        forward=lambda x: fwd(x.gate, x.up, x.beta, x.lb),
        backward=lambda x: bwd(x.dh, x.gate, x.up, x.beta, x.lb, x.second is None),
        autograd=lambda first, second, beta=x.beta, lb=x.lb: Formula.apply(first, second, beta, lb),
        version=f"torch {torch.__version__}" + (f", triton {_version('triton')}" if compiled else ""),
        compiled=compiled,
        workspace_bytes=None,
    )
    return found


class CudnnGraphs:
    """The cuDNN frontend graphs of one row, the forward and the backward, each run with its fastest plan (or its
    first plan with ``first_plan``)."""

    def __init__(self, x: types.SimpleNamespace, first_plan: bool) -> None:
        try:
            import cudnn  # noqa: PLC0415 - optional, imported on use
        except Exception as exc:  # noqa: BLE001
            raise Unavailable(f"the cuDNN frontend does not import: {type(exc).__name__}: {exc}") from None
        self.c = cudnn
        self.x = x
        self.first_plan = first_plan
        self.handle = cudnn.create_handle()
        self.version = f"cudnn frontend {getattr(cudnn, '__version__', None)}, backend {cudnn.backend_version()}"
        self.runtime_scalars = hasattr(cudnn, "scalar_type") and hasattr(cudnn.scalar_type, "RUNTIME_PARAM")
        self.errors: dict[str, str] = {}
        self.plans: dict[str, dict[str, Any]] = {}
        self.forward_graph = self._try_build("fwd", self._forward_ops)
        self.backward_graph = self._try_build("bwd", self._backward_ops)

    # -- graph construction

    def _tensor(self, g, name: str, view: torch.Tensor) -> Any:
        rows, cols = view.shape
        s = view.stride(0)
        return g.tensor(name=name, dim=[1, rows, cols], stride=[rows * s, s, 1], data_type=self.c.data_type.BFLOAT16)

    def _scalar(self, g, built: types.SimpleNamespace, name: str, value: float) -> Any:
        """A runtime scalar of the graph, or a pass-by-value FP32 tensor (fed at every call) without one."""
        if self.runtime_scalars:
            return g.tensor_scalar(float(value), self.c.scalar_type.RUNTIME_PARAM, name=name)
        tensor = g.tensor(
            name=name, dim=[1, 1, 1], stride=[1, 1, 1], data_type=self.c.data_type.FLOAT, is_pass_by_value=True
        )
        built.inputs[tensor] = torch.full((1, 1, 1), float(value), dtype=torch.float32)
        return tensor

    def _output(self, value, rows: int, cols: int, row_stride: int) -> Any:
        return (
            value.set_output(True)
            .set_data_type(self.c.data_type.BFLOAT16)
            .set_dim([1, rows, cols])
            .set_stride([rows * row_stride, row_stride, 1])
        )

    def _forward_ops(self, g, built: types.SimpleNamespace) -> None:
        x = self.x
        G, U = self._tensor(g, "gate", x.gate), self._tensor(g, "up", x.up)
        built.inputs.update({G: "gate", U: "up"})
        beta, lb = self._scalar(g, built, "situ_beta", x.beta), self._scalar(g, built, "situ_linear_beta", x.lb)
        gate_act = g.mul(a=g.mul(a=beta, b=g.tanh(input=g.div(a=G, b=beta))), b=g.sigmoid(input=G))
        up_act = g.mul(a=lb, b=g.tanh(input=g.div(a=U, b=lb)))
        built.outputs = {"h": self._output(g.mul(a=gate_act, b=up_act), x.R, x.I, x.I)}

    def _backward_ops(self, g, built: types.SimpleNamespace) -> None:
        x = self.x
        DH, G, U = self._tensor(g, "dh", x.dh), self._tensor(g, "gate", x.gate), self._tensor(g, "up", x.up)
        built.inputs.update({DH: "dh", G: "gate", U: "up"})
        one = self._scalar(g, built, "one", 1.0)
        beta, lb = self._scalar(g, built, "situ_beta", x.beta), self._scalar(g, built, "situ_linear_beta", x.lb)
        s = g.sigmoid(input=G)
        gate_tanh = g.tanh(input=g.div(a=G, b=beta))
        up_tanh = g.tanh(input=g.div(a=U, b=lb))
        gate_act = g.mul(a=g.mul(a=beta, b=gate_tanh), b=s)
        up_act = g.mul(a=lb, b=up_tanh)
        gate_grad = g.add(
            a=g.mul(a=g.sub(a=one, b=g.mul(a=gate_tanh, b=gate_tanh)), b=s),
            b=g.mul(a=g.mul(a=beta, b=gate_tanh), b=g.mul(a=s, b=g.sub(a=one, b=s))),
        )
        up_grad = g.sub(a=one, b=g.mul(a=up_tanh, b=up_tanh))
        d_up = g.mul(a=g.mul(a=DH, b=gate_act), b=up_grad)
        d_gate = g.mul(a=g.mul(a=DH, b=up_act), b=gate_grad)
        out_stride = 2 * x.I if x.second is None else x.I
        built.outputs = {
            "d_gate": self._output(d_gate, x.R, x.I, out_stride),
            "d_up": self._output(d_up, x.R, x.I, out_stride),
        }

    def _try_build(self, what: str, ops) -> Optional[types.SimpleNamespace]:
        """The graph of ``ops`` with its plans built (all of them, else the first that builds), or None with the error
        recorded under ``what``."""
        c = self.c
        record: dict[str, Any] = {"heuristics": ["A", "FALLBACK"]}
        start = time.perf_counter()
        built = types.SimpleNamespace(inputs={}, outputs={}, index=None)
        try:
            g = c.pygraph(
                io_data_type=c.data_type.BFLOAT16,
                intermediate_data_type=c.data_type.FLOAT,
                compute_data_type=c.data_type.FLOAT,
                handle=self.handle,
            )
            ops(g, built)
            g.validate()
            g.build_operation_graph()
            g.create_execution_plans([c.heur_mode.A, c.heur_mode.FALLBACK])
            g.check_support()
            all_plans = getattr(getattr(c, "build_plan_policy", None), "ALL", None)
            if self.first_plan or all_plans is None:
                g.build_plans()
                record["policy"] = "first plan that builds"
            else:
                try:
                    g.build_plans(all_plans)
                    record["policy"] = "ALL"
                except Exception as exc:  # noqa: BLE001 - building every plan refused: the first that builds
                    record["all_refused"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    g.build_plans()
                    record["policy"] = "first plan that builds (ALL refused)"
        except Exception as exc:  # noqa: BLE001 - a graph no engine builds is reported with its error
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            self.errors[what] = f"{type(exc).__name__}: {str(exc)[:400]}"
            return None
        built.graph = g
        built.workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
        record["build_s"] = time.perf_counter() - start
        if record["policy"] == "ALL":
            record.update(self._select_plan(built))
        self.plans[what] = record
        return built

    # -- plans and execution

    def _new_outputs(self, built: types.SimpleNamespace) -> tuple[Any, dict[str, torch.Tensor]]:
        """Fresh outputs of ``built``: ``(result, {graph output name: tensor})``, ``result`` the arm's return value."""
        x = self.x
        if "h" in built.outputs:
            h = torch.empty(x.R, x.I, dtype=torch.bfloat16, device="cuda")
            return h, {"h": h}
        if x.second is None:
            d_first = torch.empty(x.R, 2 * x.I, dtype=torch.bfloat16, device="cuda")
            return (d_first, None), {"d_gate": d_first[:, : x.I], "d_up": d_first[:, x.I :]}
        d_gate = torch.empty(x.R, x.I, dtype=torch.bfloat16, device="cuda")
        d_up = torch.empty(x.R, x.I, dtype=torch.bfloat16, device="cuda")
        return (d_gate, d_up), {"d_gate": d_gate, "d_up": d_up}

    def _execute(
        self,
        built: types.SimpleNamespace,
        values: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        *,
        index: Optional[int] = None,
        workspace: Optional[torch.Tensor] = None,
    ) -> None:
        pack = {
            tensor: (values[source] if isinstance(source, str) else source) for tensor, source in built.inputs.items()
        }
        pack.update({tensor: outputs[name] for name, tensor in built.outputs.items()})
        self.c.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        index = built.index if index is None else index
        workspace = built.workspace if workspace is None else workspace
        if index is None:
            built.graph.execute(pack, workspace, handle=self.handle)
        else:
            built.graph.execute_plan_at_index(pack, workspace, index, handle=self.handle)

    def _select_plan(self, built: types.SimpleNamespace) -> dict[str, Any]:
        """Times every plan of ``built`` and sets ``built.index`` and ``built.workspace`` to the fastest one's."""
        g = built.graph
        calls = ("get_execution_plan_count", "get_workspace_size_plan_at_index", "execute_plan_at_index")
        missing = [name for name in calls if not hasattr(g, name)]
        if missing:
            return {"selection": f"none: the frontend has no {', '.join(missing)}; the default plan runs"}
        values = self._values(self.x)
        _, outputs = self._new_outputs(built)
        count = int(g.get_execution_plan_count())
        candidates = []
        for index in range(count):
            entry: dict[str, Any] = {"index": index}
            with contextlib.suppress(Exception):
                entry["name"] = str(g.get_plan_name_at_index(index))
            try:
                size = int(g.get_workspace_size_plan_at_index(index))
                workspace = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
                entry["workspace_bytes"] = size

                def call(index=index, workspace=workspace):
                    self._execute(built, values, outputs, index=index, workspace=workspace)

                try:
                    samples = _replay_samples_ms(call, self.x.plan_calls, PLAN_WARMUP, PLAN_ROUNDS)
                    entry["timer"] = "graph"
                except Exception as exc:  # noqa: BLE001 - a plan the capture refuses is timed eagerly
                    entry["graph_unavailable"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    samples = _events_ms(call, self.x.plan_calls, PLAN_WARMUP, PLAN_ROUNDS)
                    entry["timer"] = "eager"
                entry["us"] = statistics.median(samples) * 1e3
            except Exception as exc:  # noqa: BLE001 - a plan that does not build or run is recorded and skipped
                if isinstance(exc, torch.OutOfMemoryError):
                    raise
                entry["unavailable"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            candidates.append(entry)
        timed = [entry for entry in candidates if "us" in entry]
        record: dict[str, Any] = {"candidates": count, "timed": len(timed), "candidate_us": candidates}
        if not timed:
            record["selection"] = "none: no candidate plan ran; the default plan runs"
            return record
        captured = [entry for entry in timed if entry["timer"] == "graph"]
        best = min(captured or timed, key=lambda entry: entry["us"])
        built.index = best["index"]
        built.workspace = torch.empty(max(best["workspace_bytes"], 1), dtype=torch.uint8, device="cuda")
        record.update(chosen_index=best["index"], chosen_name=best.get("name"), chosen_us=best["us"])
        return record

    @staticmethod
    def _values(x: types.SimpleNamespace) -> dict[str, torch.Tensor]:
        """The graph inputs ``x`` holds by name (the forward reads ``gate`` and ``up``, the backward ``dh`` too)."""
        return {name: getattr(x, name) for name in ("gate", "up", "dh") if hasattr(x, name)}

    def forward(self, x: types.SimpleNamespace) -> torch.Tensor:
        if self.forward_graph is None:
            raise Unavailable(f"no cuDNN forward graph builds: {self.errors.get('fwd')}")
        h, outputs = self._new_outputs(self.forward_graph)
        self._execute(self.forward_graph, self._values(x), outputs)
        return h

    def backward(self, x: types.SimpleNamespace) -> tuple:
        if self.backward_graph is None:
            raise Unavailable(f"no cuDNN backward graph builds: {self.errors.get('bwd')}")
        result, outputs = self._new_outputs(self.backward_graph)
        self._execute(self.backward_graph, self._values(x), outputs)
        return result


def cudnn_arm(x: types.SimpleNamespace, first_plan: bool) -> dict[str, Any]:
    graphs = CudnnGraphs(x, first_plan)

    class Graphs(torch.autograd.Function):
        """The two graphs as one autograd function (the backward reads the saved inputs through the graphs' strides)."""

        @staticmethod
        def forward(ctx, first, second):
            ctx.save_for_backward(first, second)
            gate, up = _halves(first, second)
            return graphs.forward(types.SimpleNamespace(gate=gate, up=up))

        @staticmethod
        def backward(ctx, dh):
            first, second = ctx.saved_tensors
            gate, up = _halves(first, second)
            return graphs.backward(types.SimpleNamespace(gate=gate, up=up, dh=dh.contiguous()))

    return {
        "forward": graphs.forward,
        "backward": graphs.backward,
        "autograd": lambda first, second: Graphs.apply(first, second),
        "version": graphs.version,
        "workspace_bytes": {
            what: int(built.workspace.numel()) if built is not None else None
            for what, built in (("fwd", graphs.forward_graph), ("bwd", graphs.backward_graph))
        },
        "notes": {
            "graphs": "fwd: one pointwise graph; bwd: one pointwise graph with d_gate and d_up",
            "scalars": "runtime scalars" if graphs.runtime_scalars else "pass-by-value FP32 tensors",
            "errors": graphs.errors,
            "plans": graphs.plans,
        },
    }


def make_arm(name: str, x: types.SimpleNamespace, args: argparse.Namespace) -> dict[str, Any]:
    if name == "cake":
        return cake_arm(x)
    if name in ("torch_compile", "eager"):
        return formula_arm(x, compiled=name == "torch_compile")
    if name == "cudnn":
        return cudnn_arm(x, args.cudnn_first_plan)
    raise ValueError(f"unknown arm {name!r} (one of {', '.join(ARMS)})")


# ----------------------------------------------------------------------------------------------------------------
# Rows


def _stage_calls(arm: dict[str, Any], x: types.SimpleNamespace) -> dict[str, Callable[[], object]]:
    calls: dict[str, Callable[[], object]] = {
        "fwd": lambda: arm["forward"](x),
        "bwd": lambda: arm["backward"](x),
    }
    leaves = [None if t is None else t.detach().clone().requires_grad_(True) for t in (x.first, x.second)]
    wants = [t for t in leaves if t is not None]

    def ckpt():
        h = torch.utils.checkpoint.checkpoint(arm["autograd"], *leaves, use_reentrant=False, preserve_rng_state=False)
        return torch.autograd.grad(h, wants, x.dh)

    calls["ckpt"] = ckpt
    return calls


def _compare(found: dict[str, torch.Tensor], cake: dict[str, torch.Tensor]) -> dict[str, Any]:
    out = {}
    for name, want in cake.items():
        got = found.get(name)
        if got is None:
            out[name] = "missing"
            continue
        if got.shape != want.shape or got.dtype != want.dtype:
            out[name] = f"{tuple(got.shape)} {got.dtype} != {tuple(want.shape)} {want.dtype}"
            continue
        same_bits = got.contiguous().view(torch.int16) == want.contiguous().view(torch.int16)
        n = want.numel()
        out[name] = {
            "max_abs_diff": float((got.float() - want.float()).abs().max()) if n else 0.0,
            "bits_differ": int(torch.count_nonzero(~same_bits)) / n if n else 0.0,
        }
    return out


def _accuracy(found: dict[str, torch.Tensor], want: dict[str, torch.Tensor], rows: int) -> dict[str, float]:
    """The relative L2 error against the float64 oracle of each output on its first ``rows`` rows."""
    out = {}
    for name, value in found.items():
        ref = want[name]
        norm = float(ref.norm())
        out[name] = float((value[:rows].double() - ref).norm()) / norm if norm else math.nan
    return out


def _first(fn: Callable[[], Any], missing: dict[str, str], what: str) -> Any:
    """``fn()``, or None with the reason it did not run under ``missing[what]`` (an out-of-memory error raises)."""
    try:
        return fn()
    except Unavailable as exc:
        missing[what] = str(exc)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, torch.OutOfMemoryError):
            raise
        missing[what] = f"first call: {type(exc).__name__}: {str(exc)[:300]}"
    return None


# The code objects of a compiled arm each stage runs: a stage times only when each holds a compiled Dynamo entry.
CODES = {"fwd": ("fwd",), "bwd": ("bwd",), "ckpt": ("autograd fwd", "autograd bwd")}


def _time_call(
    call: Callable[[], object],
    stage: str,
    arm: dict[str, Any],
    x: types.SimpleNamespace,
    args: argparse.Namespace,
    peak: Optional[float],
) -> dict[str, Any]:
    """The entry of one stage call: the timers, the DRAM bytes and per timer the bandwidth and its share of the NVML
    peak, or the reason it is unavailable.  One untimed call precedes the timers."""
    missing: dict[str, str] = {}
    _first(lambda: (call(), torch.cuda.synchronize()), missing, "call")
    if missing:
        return {"unavailable": missing["call"]}
    entry: dict[str, Any] = {}
    codes = arm.get("codes") or {}
    if codes:
        entries = {name: _cache_entries(codes[name]) for name in CODES[stage]}
        entry["dynamo_cache_entries"] = entries
        eager = [name for name, count in entries.items() if count == 0]
        if eager:
            return {"unavailable": f"torch.compile ran {', '.join(eager)} without a compiled graph (no cache entry)"}
        before = _dynamo_graphs()
    try:
        times = time_stage(call, args)
    except Unavailable as exc:
        return {"unavailable": str(exc)}
    if codes:
        entry["dynamo_graphs_compiled_while_timing"] = _dynamo_graphs() - before
    moved = BYTES_PER_ELEMENT[stage] * x.R * x.I
    entry.update(timers=times, dram_bytes=moved)
    for timer in times.values():
        if isinstance(timer, dict):
            timer["gbs"] = moved / (timer["us"] * 1e3)
            timer["util"] = timer["gbs"] / peak if peak else None
    return entry


def bench_row(rows: int, width: int, layout: str, args: argparse.Namespace, peak: Optional[float]) -> dict[str, Any]:
    x = make_inputs(rows, width, layout, args.beta, args.lb, args.seed)
    x.plan_calls = args.plan_calls
    result: dict[str, Any] = {"R": rows, "I": width, "layout": layout, "beta": x.beta, "lb": x.lb, "arms": {}}
    acc_rows = min(rows, args.acc_rows)
    want = oracle(x, acc_rows)
    reference = None
    for name in args.arms:
        row: dict[str, Any] = {}
        result["arms"][name] = row
        try:
            arm = make_arm(name, x, args)
        except Unavailable as exc:
            row["unavailable"] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            row["unavailable"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            continue
        row["version"] = arm["version"]
        row["workspace_bytes"] = arm.get("workspace_bytes")
        row.update(arm.get("notes", {}))
        missing: dict[str, str] = {}
        h = _first(lambda: arm["forward"](x), missing, "fwd")
        grads = _first(lambda: arm["backward"](x), missing, "bwd")
        outputs = _outputs(h, grads)
        row["rel_l2_vs_fp64"] = _accuracy(outputs, want, acc_rows)
        if name == "cake":
            reference = outputs
        elif reference is not None:
            row["vs_cake"] = _compare(outputs, reference)
        del h, grads, outputs
        row["stages"] = {}
        for stage, call in _stage_calls(arm, x).items():
            needs = ("fwd", "bwd")
            reasons = [missing[part] for part in needs if part in missing and (stage == "ckpt" or part == stage)]
            if reasons:
                row["stages"][stage] = {"unavailable": "; ".join(reasons)}
                continue
            row["stages"][stage] = _time_call(call, stage, arm, x, args, peak)
            torch.cuda.empty_cache()
    cake = result["arms"].get("cake", {}).get("stages", {})
    for row in result["arms"].values():
        for stage, entry in row.get("stages", {}).items():
            base = cake.get(stage, {}).get("timers", {})
            for timer, value in entry.get("timers", {}).items():
                if isinstance(value, dict) and isinstance(base.get(timer), dict):
                    value["vs_cake"] = value["us"] / base[timer]["us"]
    return result


def _cells(entry: dict[str, Any], timers: tuple[str, ...]) -> str:
    cells = []
    for timer in timers:
        value = entry.get("timers", {}).get(timer)
        if not isinstance(value, dict):
            cells += ["unavailable" if value is not None else "-", "-", "-", "-"]
            continue
        util = value.get("util")
        cells += [
            f"{value['us']:.1f}",
            f"{value['gbs']:.0f}",
            f"{100 * util:.1f}%" if util is not None else "-",
            f"{value['vs_cake']:.3f}" if "vs_cake" in value else "-",
        ]
    return " | ".join(cells)


def format_row(result: dict[str, Any], timers: tuple[str, ...]) -> list[str]:
    head = f"R={result['R']} I={result['I']} layout={result['layout']} beta={result['beta']:g} lb={result['lb']:g}"
    lines = [f"### {head}", ""]
    cols = " | ".join(f"{t} us | {t} GB/s | {t} util | {t} base/cake" for t in timers)
    lines.append(f"| arm | stage | {cols} |")
    lines.append("|---|---|" + "---|" * (4 * len(timers)))
    notes = []
    for name, row in result["arms"].items():
        if "unavailable" in row:
            lines.append(f"| {name} | all | " + " | ".join("-" for _ in range(4 * len(timers))) + " |")
            notes.append(f"- {name}: unavailable: {row['unavailable']}")
            continue
        for stage, entry in row["stages"].items():
            if "unavailable" in entry:
                lines.append(f"| {name} | {stage} | " + " | ".join("-" for _ in range(4 * len(timers))) + " |")
                notes.append(f"- {name} {stage}: unavailable: {entry['unavailable']}")
                continue
            lines.append(f"| {name} | {stage} | {_cells(entry, timers)} |")
            for timer, value in entry["timers"].items():
                if isinstance(value, str):
                    notes.append(f"- {name} {stage} {timer}: {value}")
            if entry.get("dynamo_graphs_compiled_while_timing"):
                notes.append(f"- {name} {stage}: Dynamo compiled {entry['dynamo_graphs_compiled_while_timing']} graphs")
        accuracy = ", ".join(f"{k} {v:.3e}" for k, v in row.get("rel_l2_vs_fp64", {}).items())
        notes.append(f"- {name} rel-L2 vs FP64 (first rows): {accuracy}")
        if "vs_cake" in row:
            notes.append(f"- {name} outputs vs cake: {json.dumps(row['vs_cake'])}")
        for key in ("compile", "merged_gradient", "graphs", "scalars", "errors"):
            if row.get(key):
                notes.append(f"- {name} {key}: {row[key]}")
        for graph, plan in row.get("plans", {}).items():
            short = {k: v for k, v in plan.items() if k != "candidate_us"}
            notes.append(f"- {name} plan {graph}: {json.dumps(short)}")
        notes.append(f"- {name} workspace bytes: {row.get('workspace_bytes')}; version: {row.get('version')}")
    return lines + [""] + notes + [""]


# ----------------------------------------------------------------------------------------------------------------
# Identity and main


def identity(device: torch.device) -> dict[str, Any]:
    p = torch.cuda.get_device_properties(device)
    dists = (
        "cake-pplx-situ-glu",
        "apache-tvm-ffi",
        "torch",
        "triton",
        "nvidia-cudnn-frontend",
        "nvidia-cudnn-cu12",
        "nvidia-cudnn-cu13",
        "nvidia-ml-py",
    )
    out: dict[str, Any] = {
        "gpu": p.name,
        "cc": list(torch.cuda.get_device_capability(device)),
        "sms": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "versions": {d: _version(d) for d in dists},
        "nvml": nvml_mapping(device),
    }
    query = ["nvidia-smi", f"--id={out['nvml']['torch_uuid']}", "--query-gpu=driver_version,power.limit,clocks.max.sm"]
    with contextlib.suppress(Exception):
        q = subprocess.run([*query, "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
        out["driver_power_maxclock"] = q.stdout.strip()
    try:
        out["nvml_peak_gbps"] = peak_gbps(device)
    except Exception as exc:  # noqa: BLE001 - no peak: the util columns stay empty
        out["nvml_peak_gbps"] = None
        out["nvml_peak_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _point(text: str) -> tuple[int, int]:
    rows, _, width = text.partition(":")
    try:
        return int(rows), int(width)
    except ValueError:
        raise argparse.ArgumentTypeError(f"a point is R:I, got {text!r}") from None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Times the Cake SiTU-GLU training kernels and the comparators of the same operation."
    )
    parser.add_argument("--points", type=_point, nargs="+", default=[_point(p) for p in POINTS], help="R:I pairs")
    parser.add_argument("--layouts", nargs="+", default=["merged", "split"], choices=["merged", "split"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--timers", nargs="+", default=list(TIMERS), choices=list(TIMERS))
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--lb", type=float, default=25.0)
    parser.add_argument("--calls", type=int, default=20, help="calls captured in one CUDA graph (graph timer)")
    parser.add_argument("--warmup", type=int, default=3, help="untimed replays before the timed ones (graph timer)")
    parser.add_argument("--trials", type=int, default=30, help="timed replays (graph timer)")
    parser.add_argument("--trim", type=float, default=0.1, help="share of the replays dropped at each end")
    parser.add_argument("--plan-calls", type=int, default=10, help="calls per replay when ranking cuDNN plans")
    parser.add_argument(
        "--cudnn-first-plan", action="store_true", help="build and run only the first cuDNN plan that builds"
    )
    parser.add_argument("--acc-rows", type=int, default=1024, help="rows compared with the float64 oracle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if not 0 <= args.trim < 0.5:
        parser.error("--trim is in [0, 0.5)")
    args.arms = list(dict.fromkeys(["cake", *args.arms]))  # cake first: the other arms are compared with its outputs
    device = torch.device("cuda", torch.cuda.current_device())
    ident = identity(device)
    peak = ident.get("nvml_peak_gbps")
    nvml = ident["nvml"]
    print(
        f"# {ident['gpu']} (sm_{ident['cc'][0]}{ident['cc'][1]}, {ident['sms']} SMs), torch {ident['torch']} "
        f"(CUDA {ident['torch_cuda']}), driver/power/max clock {ident.get('driver_power_maxclock')}"
    )
    print(
        f"# CUDA_VISIBLE_DEVICES={nvml['CUDA_VISIBLE_DEVICES']}; torch cuda:{nvml['torch_index']} = "
        f"{nvml['torch_uuid']}; NVML index {nvml.get('nvml_index')} = {nvml.get('nvml_uuid')} "
        f"(match: {nvml.get('uuid_match')}); NVML devices {nvml.get('nvml_devices')}"
    )
    print(
        f"# NVML peak {peak} GB/s = mem clock {nvml.get('mem_clock_mhz')} MHz x bus {nvml.get('bus_width_bits')} bits "
        f"x 2 / 8 / 1000{'' if peak else ' (' + str(ident.get('nvml_peak_error')) + ')'}"
    )
    print(f"# versions: {json.dumps(ident['versions'])}")
    print(
        "# times in us per call; base/cake above 1: cake is faster; graph = trimmed mean of CUDA-graph replays of "
        f"{args.calls} calls (warm L2), cupti = median cold-L2 kernel time; GB/s from 6 / 10 / 22 bytes per element "
        "(fwd / bwd / ckpt); util = GB/s / NVML peak"
    )
    results = []
    for rows, width in args.points:
        for layout in args.layouts:
            result = bench_row(rows, width, layout, args, peak)
            results.append(result)
            for line in format_row(result, tuple(args.timers)):
                print(line, flush=True)
            torch.cuda.empty_cache()
    if args.json:
        record = {"identity": ident, "args": {**vars(args), "points": [list(p) for p in args.points]}, "rows": results}
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
