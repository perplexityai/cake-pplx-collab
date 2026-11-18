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

The GPU timer of the Engram row lookup benchmark (``bench_cake_engram_lookup_det_train.py``), a copy of the causal
conv1d family's timer (families share no code).

Per timed iteration (:meth:`GpuTimer.times`):

1. an L2 flush: ``zero_()`` of an int8 buffer of twice the L2 size;
2. ``eager`` mode only: a one-thread spin kernel (``torch.cuda._sleep``) long enough for the host to enqueue the whole
   iteration behind it, so the iteration's host time (autograd, Python, launch overhead) does not enter its GPU
   span (``late`` counts the iterations whose enqueue outlasted the spin: their span may hold host gaps);
3. the iteration (``fn()``; ``graph`` mode: the replay of a CUDA graph captured from ``fn``).

The iteration's time is its GPU span from CUPTI activity records (``cupti-python``): from the start of the first to
the end of the last kernel, memset or copy launched by the API calls the host made between the iteration's own two
CUPTI timestamps (matched by correlation id, so the flush and the spin are outside it).  Without ``cupti-python`` it
falls back to CUDA events recorded after the spin and after the iteration (``method`` says which).
"""

from __future__ import annotations

import bisect
import collections
import statistics
import time
from typing import Callable, Optional

import torch

# The spin ahead of an eager iteration lasts at least SPIN_MIN_MS and SPIN_FACTOR times the slowest enqueue seen.
SPIN_MIN_MS = 0.5
SPIN_FACTOR = 4.0
SPIN_MAX_MS = 200.0


def _cupti():
    try:
        from cupti import cupti  # type: ignore

        return cupti
    except Exception:  # noqa: BLE001 -- no cupti-python: CUDA events
        return None


class GpuTimer:
    """Cold-L2 GPU times of a callable on the current device (module docstring)."""

    def __init__(self, device: torch.device, use_cupti: bool = True) -> None:
        self.device = device
        l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
        self.flush = torch.empty(max(2 * l2_bytes, 1 << 20), dtype=torch.int8, device=device)
        self.cupti = _cupti() if use_cupti else None
        self.method = "cupti" if self.cupti is not None else "cuda_events"
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        probe, spans = 1_000_000, []
        for _ in range(3):
            torch.cuda.synchronize(device)
            start.record()
            torch.cuda._sleep(probe)
            end.record()
            torch.cuda.synchronize(device)
            spans.append(start.elapsed_time(end))
        self.cycles_per_ms = probe / statistics.median(spans)

    def spin_ms(self, fn: Callable[[], object], samples: int = 3) -> float:
        """The spin length for ``fn``: :data:`SPIN_FACTOR` x its slowest enqueue over ``samples`` calls."""

        slowest = 0.0
        for _ in range(samples):
            torch.cuda.synchronize(self.device)
            self.flush.zero_()
            torch.cuda._sleep(int(SPIN_MIN_MS * self.cycles_per_ms))
            t0 = time.perf_counter()
            fn()
            slowest = max(slowest, (time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize(self.device)
        return min(SPIN_MAX_MS, max(SPIN_MIN_MS, SPIN_FACTOR * slowest))

    def times(self, fn: Callable[[], object], iters: int, spin_ms: Optional[float]) -> dict:
        """``{"ms": [...], "enqueue_ms": [...], "late": n, "method": ...}`` of ``iters`` iterations of ``fn``;
        ``spin_ms`` None: no spin (a graph replay enqueues in microseconds)."""

        cycles = 0 if spin_ms is None else int(spin_ms * self.cycles_per_ms)
        if self.cupti is not None:
            ms, enqueue = self._cupti_times(fn, iters, cycles)
        else:
            ms, enqueue = self._event_times(fn, iters, cycles)
        late = 0 if spin_ms is None else sum(e >= spin_ms for e in enqueue)
        return {"ms": ms, "enqueue_ms": enqueue, "late": late, "method": self.method, "spin_ms": spin_ms}

    def _iteration_prologue(self, cycles: int) -> None:
        torch.cuda.synchronize(self.device)
        self.flush.zero_()
        if cycles:
            torch.cuda._sleep(cycles)

    def _cupti_times(self, fn, iters, cycles):
        cupti = self.cupti
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
        stamps, enqueue = [], []
        try:
            for _ in range(iters):
                self._iteration_prologue(cycles)
                t0 = time.perf_counter()
                begin = cupti.get_timestamp()
                fn()
                end = cupti.get_timestamp()
                enqueue.append((time.perf_counter() - t0) * 1e3)
                stamps.append((begin, end))
            torch.cuda.synchronize(self.device)
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
        return times, enqueue

    def _event_times(self, fn, iters, cycles):
        times, enqueue = [], []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            self._iteration_prologue(cycles)
            start.record()
            t0 = time.perf_counter()
            fn()
            enqueue.append((time.perf_counter() - t0) * 1e3)
            end.record()
            torch.cuda.synchronize(self.device)
            times.append(start.elapsed_time(end))
        return times, enqueue


def capture_graph(fn: Callable[[], object], stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    """A CUDA graph of one call of ``fn`` captured on ``stream`` (after one warm-up call on it); the caller ran any
    autograd forward whose backward ``fn`` replays on the same stream."""

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    return graph
