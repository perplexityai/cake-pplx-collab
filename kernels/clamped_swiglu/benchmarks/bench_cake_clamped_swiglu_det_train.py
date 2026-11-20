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

Benchmark of the Cake routed-expert clamped SwiGLU training kernels (``cake_pplx.cake_clamped_swiglu_det_train``,
``h = bf16(w silu(min(g, L)) clamp(u, -L, L))`` and its backward, ``I = 2304``) against the comparators of the same
operation, each when it imports.

Arms (``--arms``; default: every arm; ``cake`` always runs first, since the other arms' outputs are compared with its
outputs):

* ``cake``: ``cake_clamped_swiglu_det_train_forward`` / ``_backward`` (one kernel each).
* ``te``: Transformer Engine on the merged ``[R, 2I]`` layout (the gate is the first half, the up the second half)
  with ``limit=L``, ``alpha=1`` and ``glu_linear_offset=0``: ``silu(min(g, L)) clamp(u, -L, L)`` with the same
  inclusive masks.  With a route weight it runs ``transformer_engine_torch.scaled_clamped_swiglu`` /
  ``scaled_clamped_dswiglu`` (``glu_interleave_size=0``): one BF16 rounding of ``w a`` and the FP32
  ``d_weight = sum dh a`` from the unweighted ``a``.  Without one (``--unweighted``) it runs ``clamped_swiglu`` /
  ``clamped_dswiglu`` with no quantizer (BF16 outputs).  Split rows print it as unavailable.
* ``torch_compile``: ``torch.compile`` (Inductor, ``fullgraph=True``, ``dynamic=False``) of the forward and backward
  formulas of the ``eager`` arm, so the backward fuses ``d_gate``, ``d_up`` and ``d_weight`` as Inductor chooses; in
  the merged layout the compiled backward graph ends in the ``torch.cat`` of the two gradient halves.  Every row calls
  ``torch._dynamo.reset()`` and compiles fresh code objects, one per calling context (the direct forward and
  backward, and the forward and backward inside the autograd function), with
  ``torch._dynamo.config.fail_on_recompile_limit_hit`` set: a call that would fall back to eager raises instead, and
  its stage prints as unavailable with the reason.  A stage times only when every code object it runs holds a
  compiled Dynamo cache entry after its untimed call (the counts are in the stage and in the arm's ``compile``
  record); if Dynamo compiles a graph while the timers run, the stage is timed again, and prints as unavailable if
  it compiles again.  Each guard fails closed: a torch without ``fail_on_recompile_limit_hit`` makes the arm
  unavailable, and one without Dynamo's cache-entry lookup, or whose graph counter counts no graph after a compiled
  call, makes the stage unavailable, instead of timing a call that could have run eagerly.
* ``eager``: the explicit FP32 formulas in eager PyTorch: ``torch.clamp`` for ``min(g, L)`` and ``clamp(u, -L, L)``,
  ``torch.where`` for the inclusive gradient masks, and in the merged layout ``torch.cat`` (a separate copy) of the
  two gradient halves.
* ``cudnn``: a cuDNN frontend pointwise graph (``min``, ``max``, ``swish``, ``mul``; FP32 compute, BF16 output) for the
  forward.  For the backward the benchmark first builds one graph (pointwise ``d_gate`` / ``d_up`` with the masks as
  ``binary_select`` on BOOLEAN comparisons, and with a route weight the per-row ``reduction`` (ADD) of ``d_weight``),
  then, if a weighted one is refused, a pointwise graph for ``d_gate`` / ``d_up`` plus a pointwise + ``reduction`` graph
  for ``d_weight`` (a composite that reads the inputs twice).  Each graph takes the plans of the heuristics modes A, B
  and FALLBACK (if refused: A and FALLBACK), deselects the runtime-compiled engines (behavior note
  ``RUNTIME_COMPILATION``; ``--cudnn-runtime-compiled`` keeps them: with them, building every plan did not finish in 45
  minutes on B200 and GB300, using up to 230 GB of host memory) and builds all the others (``build_plan_policy.ALL``);
  each plan's 10 calls are captured in one CUDA graph and replayed, timed with CUDA events (warm L2, the median of 5
  samples after one warm-up sample), and the fastest plan is kept.  A plan the capture refuses is timed over 10 eager
  calls instead (which include the host's issue time) and is kept only when no plan captures.  The arm's ``plans``
  record holds the number of candidates, each one's time, timer or error, the timer that ranked them, the chosen plan's
  index and name, each plan's behavior notes, and whether runtime-compiled engines were searched.  If every build of all
  plans is refused, the graph runs the first plan of the modes A and FALLBACK that builds, runtime-compiled engines
  not filtered (its record says ``runtime_compiled_engines: not filtered``): when every plan of a graph carries
  ``RUNTIME_COMPILATION``, deselecting them refuses every build, and this fallback is what runs.  If the frontend
  has no indexed plan calls or no candidate runs, it runs the frontend's selected plan; the record says why.  The
  composite's two graphs are also timed one by one (``parts`` of its ``bwd`` stage).  A stage no graph builds prints
  as unavailable with the error.

Every arm runs on identical inputs: ``gate`` and ``up`` ``6 x`` unit normals, ``dh`` a unit normal (BF16) and the FP32
route weight ``1.5 x U[0, 1)`` (``--unweighted``: none).  ``--deep-gate`` sets the gates at columns 0, 768 and 1536
of every row to ``-88`` (BF16): every group of columns that Cake's kernels vote over holds one, so every warp of
Cake's kernels takes its deep-gate sigmoid (the scaled reciprocal).  ``--layouts``: ``merged`` (one ``[R, 2I]`` tensor
and one ``[R, 2I]`` gradient) and ``split`` (two ``[R, I]`` tensors and gradients).  Every arm allocates its outputs
inside the timed call; compilation, graph and plan building happen before timing.  Each arm's outputs are compared
with Cake's: the largest absolute difference, the share of elements whose values differ (``values_differ``, with
``+0 == -0`` and ``NaN == NaN``), the share that differ only in the sign of a zero (``zero_sign_only``), and the
share whose bits differ (``bits_differ``).

Stages: ``fwd`` (one forward), ``bwd`` (one backward from ``gate``, ``up``, ``weight`` and ``dh``), ``fwd+bwd`` (the
two in sequence), and ``ckpt`` (an autograd step under ``torch.utils.checkpoint(..., use_reentrant=False,
preserve_rng_state=False)``: the forward, then the backward that replays it; nothing here draws random numbers), for
the arms with an autograd form.  Every stage runs once untimed (compilation, plans, caches) before its timers; for
``torch_compile`` the record also holds the Dynamo cache entries of the stage's code objects and the graphs Dynamo
compiled while the kept timers ran (0; a stage timed again says how many the first timing compiled).

Timers (``--timers``):

* ``cupti``: CUPTI kernel activity (``torch.profiler``) with a cold L2 (``flush.fill_`` of twice the L2 before every
  call, not counted), the summed GPU time of the call's kernels, copies and fills; median of 100 calls after 10
  warm-up calls.  The bandwidth column is the operation's DRAM bytes (each input read once, each output written once)
  over this time.
* ``eager``: CUDA events around ``--loop`` back-to-back calls with a warm L2, per call; median of ``--rounds`` samples
  after ``--warmup`` samples.  It includes the host's issue time when the call is host-bound.
* ``graph``: the same ``--loop`` calls captured in one CUDA graph and replayed, timed the same way.

The ratio columns are baseline / Cake (above 1: Cake is faster) per timer.  ``--copy-bw`` adds the bandwidth of a
2 GiB device-to-device copy under the ``cupti`` timer, the reference for the bandwidth column.  The header prints the
GPU, the SM count, the CUDA version of torch and every library version; ``--json`` writes everything, with the
interpreter, the CPU model, the page size, the CPUs the process may run on and whether ``torch.Tensor`` has
``__dlpack_c_exchange_api__`` after ``import tvm_ffi`` (the C DLPack exchange tvm-ffi uses for tensor arguments when
it is present).

``--deterministic`` runs everything under ``torch.use_deterministic_algorithms(True)`` with
``torch.utils.deterministic.fill_uninitialized_memory`` set, so every arm pays PyTorch's NaN fill of each tensor it
allocates through PyTorch; ``args`` and ``identity.determinism`` record it.

``--host`` times only the host path of Cake's entry points instead of the arms: the forward step, the backward step
and one autograd step (the forward, then ``torch.autograd.grad`` with ``dh``), each merged and split, weighted and
unweighted, at ``--host-rows`` rows (default 1 and 4097).  Each round makes ``--host-calls`` back-to-back eager calls
(default 1000, after ``--host-warmup`` untimed calls): ``time.perf_counter`` around them gives the host's issue time
per call (the calls return before their kernels finish), and CUDA events recorded before the first and after the
last call give the device wall time per call.  The record holds the median, the lowest and the highest of
``--host-rounds`` rounds (default 9).

Example::

    python benchmarks/bench_cake_clamped_swiglu_det_train.py --rows 98304 196608 4097 --copy-bw --json rows.json
    python benchmarks/bench_cake_clamped_swiglu_det_train.py --host --json host.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import types
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.utils.checkpoint

from cake_pplx.cake_clamped_swiglu_det_train import (
    cake_clamped_swiglu_det,
    cake_clamped_swiglu_det_train_backward,
    cake_clamped_swiglu_det_train_forward,
)

I = 2304
ARMS = ("cake", "te", "torch_compile", "eager", "cudnn")
STAGES = ("fwd", "bwd", "fwd+bwd", "ckpt")
TIMERS = ("cupti", "eager", "graph")
# ``--deep-gate``: one gate below -87 in each group of columns the Cake kernels vote over (the forward votes over the
# whole row, the backward over columns [768k, 768k + 768)).
DEEP_GATE_COLUMNS = (0, 768, 1536)
DEEP_GATE = -88.0
# cuDNN plan selection: warm-up samples, timed samples and calls per sample of every candidate plan.
PLAN_WARMUP, PLAN_ROUNDS, PLAN_LOOP = 1, 5, 10
# Whether the cuDNN arm's plan search includes runtime-compiled engines (``--cudnn-runtime-compiled``).
CUDNN_OPTIONS = {"runtime_compiled": False}


class Unavailable(Exception):
    """An arm or a stage that cannot run on this row, with the reason."""


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
    return _events_ms(graph.replay, loop, warmup, rounds)


def time_stage(fn: Callable[[], object], timers: tuple[str, ...], args: argparse.Namespace) -> dict[str, Any]:
    """``{timer: milliseconds or the reason it did not run}`` of one stage call ``fn``."""
    found: dict[str, Any] = {}
    for timer in timers:
        try:
            if timer == "cupti":
                found[timer] = float(statistics.median(_cupti_call_ms(fn)))
            elif timer == "eager":

                def run(fn=fn):
                    for _ in range(args.loop):
                        fn()

                found[timer] = _events_ms(run, args.loop, args.warmup, args.rounds)
            elif timer == "graph":
                found[timer] = _graph_ms(fn, args.loop, args.warmup, args.rounds)
        except Unavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a timer that cannot run this call is reported, not fatal
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            found[timer] = f"unavailable: {type(exc).__name__}: {str(exc)[:240]}"
    return found


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def make_inputs(
    rows: int, layout: str, weighted: bool, limit: float, seed: int, deep_gate: bool = False
) -> types.SimpleNamespace:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (6.0 * torch.randn(rows, 2 * I, device="cuda", generator=gen)).bfloat16()
    if deep_gate:
        gate_up[:, list(DEEP_GATE_COLUMNS)] = DEEP_GATE  # gate columns: the gate is the first half
    weight = 1.5 * torch.rand(rows, device="cuda", generator=gen) if weighted else None
    dh = torch.randn(rows, I, device="cuda", generator=gen).bfloat16()
    x = types.SimpleNamespace(R=rows, layout=layout, L=float(torch.tensor(limit, dtype=torch.float32)), weight=weight)
    x.dh = dh
    if layout == "merged":
        x.gate_up = gate_up
        x.gate, x.up = gate_up[:, :I], gate_up[:, I:]
        x.first, x.second = gate_up, None
    elif layout == "split":
        x.gate, x.up = gate_up[:, :I].contiguous(), gate_up[:, I:].contiguous()
        x.first, x.second = x.gate, x.up
        del gate_up
    else:
        raise ValueError(f"layout is merged or split, got {layout!r}")
    return x


def dram_bytes(x: types.SimpleNamespace, stage: str, *, writes_weight_grad: bool) -> int:
    """The bytes a stage (or one graph of the cuDNN composite backward) moves to and from DRAM when each input is read
    once and each output written once."""
    bf16_rows = x.R * I * 2
    weight = x.R * 4 if x.weight is not None else 0
    forward = 3 * bf16_rows + weight  # gate, up, weight in; h out
    backward = 5 * bf16_rows + weight + (x.R * 4 if writes_weight_grad else 0)  # gate, up, dh, weight in; grads out
    return {
        "fwd": forward,
        "bwd": backward,
        "fwd+bwd": forward + backward,
        "ckpt": 2 * forward + backward,
        "bwd pointwise": 5 * bf16_rows + weight,  # gate, up, dh, weight in; d_gate, d_up out
        "bwd reduction": 3 * bf16_rows + x.R * 4,  # gate, up, dh in; d_weight out
    }[stage]


def _outputs(x: types.SimpleNamespace, h: Optional[torch.Tensor], grads: Optional[tuple]) -> dict[str, torch.Tensor]:
    """``h``, ``d_gate``, ``d_up`` and ``d_weight`` (when there is a weight) from an arm's results, the ones it has."""
    found = {}
    if h is not None:
        found["h"] = h
    if grads is not None:
        first, second, d_weight = grads
        found["d_gate"], found["d_up"] = (first[:, :I], first[:, I:]) if second is None else (first, second)
        if x.weight is not None:
            found["d_weight"] = d_weight
    return found


# ----------------------------------------------------------------------------------------------------------------
# Arms: ``forward(x) -> h``, ``backward(x) -> (d_first, d_up or None, d_weight or None)`` (a merged gradient as
# ``d_first``), and ``autograd(first, second, weight) -> h`` or None


def cake_arm(x: types.SimpleNamespace) -> dict[str, Any]:
    return {
        "forward": lambda x: cake_clamped_swiglu_det_train_forward(x.first, x.second, x.weight, limit=x.L),
        "backward": lambda x: cake_clamped_swiglu_det_train_backward(x.dh, x.first, x.second, x.weight, limit=x.L),
        "autograd": lambda first, second, weight, L=x.L: cake_clamped_swiglu_det(first, second, weight, limit=L),
        "writes_weight_grad": True,  # the kernel writes d_weight on every call
        "version": _version("cake-pplx-clamped-swiglu"),
    }


def _formula_forward(g: torch.Tensor, u: torch.Tensor, w: Optional[torch.Tensor], L: float) -> torch.Tensor:
    """``h = bf16(w silu(min(g, L)) clamp(u, -L, L))`` in FP32, one rounding."""
    gf, uf = g.float(), u.float()
    gc = torch.clamp(gf, max=L)
    uc = torch.clamp(uf, -L, L)
    a = (gc * torch.sigmoid(gc)) * uc
    if w is not None:
        a = w[:, None] * a
    return a.to(torch.bfloat16)


def _formula_backward(dh, g, u, w, L: float, merged: bool):
    """``(d_first, d_up or None, d_weight or None)`` in FP32 with the inclusive masks, each BF16 output rounded once;
    a merged layout's gradient as one ``[R, 2I]`` tensor, the ``torch.cat`` of the two halves."""
    gf, uf, d = g.float(), u.float(), dh.float()
    gc = torch.clamp(gf, max=L)
    uc = torch.clamp(uf, -L, L)
    s = torch.sigmoid(gc)
    silu = gc * s
    dws = d if w is None else d * w[:, None]
    d_gate = torch.where(gf <= L, (dws * uc) * (silu * (1.0 - s) + s), 0.0).to(torch.bfloat16)
    d_up = torch.where((uf >= -L) & (uf <= L), dws * silu, 0.0).to(torch.bfloat16)
    d_weight = None if w is None else (d * (silu * uc)).sum(dim=-1)
    if merged:
        return torch.cat([d_gate, d_up], dim=1), None, d_weight
    return d_gate, d_up, d_weight


def _halves(first: torch.Tensor, second: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return (first[:, :I], first[:, I:]) if second is None else (first, second)


def _fresh_function(fn: Callable, tag: str) -> Callable:
    """A copy of ``fn`` with its own code object, so that ``torch.compile`` compiles it anew (Dynamo keeps its compiled
    code and guards per code object)."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag}")
    return types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__, fn.__closure__)


def _reset_dynamo() -> dict[str, Any]:
    """Clears Dynamo's compiled code and makes a call past the recompile limit raise instead of running eagerly;
    returns the settings for the arm's record."""
    import torch._dynamo

    torch._dynamo.reset()
    config = torch._dynamo.config
    raises = hasattr(config, "fail_on_recompile_limit_hit")
    if raises:
        config.fail_on_recompile_limit_hit = True
    return {
        "code": "fresh code objects per row and calling context (fwd, bwd, autograd fwd, autograd bwd)",
        "fullgraph": True,
        "dynamic": False,
        "dynamo_reset": True,
        "fail_on_recompile_limit_hit": True if raises else "unavailable: torch._dynamo.config has no such setting",
        "recompile_limit": getattr(config, "recompile_limit", None),
        "suppress_errors": getattr(config, "suppress_errors", None),
    }


def _compiled(fn: Callable, tag: str) -> Callable:
    """``torch.compile`` (``fullgraph=True``, ``dynamic=False``) of a fresh copy of ``fn``, with that copy's code object
    as ``.code``.  A Dynamo or Inductor error (a refused recompile, a failed compile) raises :class:`Unavailable`, so
    its stage prints as unavailable."""
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


def _dynamo_graphs() -> int:
    """The number of graphs Dynamo has compiled in this process."""
    return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])


def _cache_entries(code: types.CodeType) -> Optional[int]:
    """The number of compiled entries Dynamo holds for ``code`` (0: it never compiled it), or None when this torch has
    no lookup."""
    import torch._dynamo.eval_frame

    lookup = getattr(torch._dynamo.eval_frame, "_debug_get_cache_entry_list", None)
    return None if lookup is None else len(lookup(code))


def _formula_arm(x: types.SimpleNamespace, compiled: bool) -> dict[str, Any]:
    fwd = autograd_fwd = _formula_forward
    bwd = autograd_bwd = _formula_backward
    notes: dict[str, Any] = {}
    codes: dict[str, types.CodeType] = {}
    if compiled:
        notes["compile"] = _reset_dynamo()
        if notes["compile"]["fail_on_recompile_limit_hit"] is not True:
            raise Unavailable(
                "torch.compile: torch._dynamo.config has no fail_on_recompile_limit_hit, so a call past the recompile "
                "limit could run eagerly and be timed as compiled"
            )
        row = f"R{x.R}_{x.layout}_{'weighted' if x.weight is not None else 'unweighted'}"
        fwd = _compiled(_formula_forward, f"{row}_fwd")
        bwd = _compiled(_formula_backward, f"{row}_bwd")
        autograd_fwd = _compiled(_formula_forward, f"{row}_autograd_fwd")
        autograd_bwd = _compiled(_formula_backward, f"{row}_autograd_bwd")
        functions = {"fwd": fwd, "bwd": bwd, "autograd fwd": autograd_fwd, "autograd bwd": autograd_bwd}
        codes = {name: function.code for name, function in functions.items()}
    if x.second is None:
        where = "inside the compiled backward graph" if compiled else "a separate copy"
        notes["merged_gradient"] = f"torch.cat of d_gate and d_up ({where})"

    class Formula(torch.autograd.Function):
        @staticmethod
        def forward(ctx, first, second, weight, L):
            ctx.save_for_backward(first, second, weight)
            ctx.L = L
            return autograd_fwd(*_halves(first, second), weight, L)

        @staticmethod
        def backward(ctx, dh):
            first, second, weight = ctx.saved_tensors
            d_first, d_up, d_weight = autograd_bwd(dh, *_halves(first, second), weight, ctx.L, second is None)
            return d_first, d_up, d_weight, None

    def backward(x):
        return bwd(x.dh, x.gate, x.up, x.weight, x.L, x.second is None)

    return {
        "forward": lambda x: fwd(x.gate, x.up, x.weight, x.L),
        "backward": backward,
        "autograd": lambda first, second, weight, L=x.L: Formula.apply(first, second, weight, L),
        "writes_weight_grad": x.weight is not None,
        "version": f"torch {torch.__version__}" + (f", triton {_version('triton')}" if compiled else ""),
        "compiled": compiled,
        "codes": codes,
        "notes": notes,
    }


def te_arm(x: types.SimpleNamespace) -> dict[str, Any]:
    if x.layout != "merged":
        raise Unavailable("TE's clamped SwiGLU ops take the merged [R, 2I] layout only (a split call needs a copy)")
    try:
        import transformer_engine
        import transformer_engine.pytorch  # noqa: F401  (loads the transformer_engine_torch extension)
        import transformer_engine_torch as tex
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(f"transformer_engine_torch does not import: {type(exc).__name__}: {exc}") from None
    version = getattr(transformer_engine, "__version__", None) or _version("transformer-engine")
    if x.weight is None:
        return _te_unweighted(tex, version, x)
    if not hasattr(tex, "scaled_clamped_swiglu") or not hasattr(tex, "scaled_clamped_dswiglu"):
        raise Unavailable(f"Transformer Engine {version} has no scaled_clamped_swiglu (added in 2.18)")

    # (input, scales, quantizer, limit, alpha, glu_linear_offset, glu_interleave_size) and the backward's
    # (grad, input, scales, quantizer, limit, alpha, glu_linear_offset, glu_interleave_size, compute_scale_grad).
    def forward(x):
        return tex.scaled_clamped_swiglu(x.gate_up, x.weight, None, x.L, 1.0, 0.0, 0)

    def backward(x):
        d_gate_up, d_weight = tex.scaled_clamped_dswiglu(x.dh, x.gate_up, x.weight, None, x.L, 1.0, 0.0, 0, True)
        return d_gate_up, None, d_weight

    class Scaled(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate_up, weight, L):
            ctx.save_for_backward(gate_up, weight)
            ctx.L = L
            return tex.scaled_clamped_swiglu(gate_up, weight, None, L, 1.0, 0.0, 0)

        @staticmethod
        def backward(ctx, dh):
            gate_up, weight = ctx.saved_tensors
            d_gate_up, d_weight = tex.scaled_clamped_dswiglu(
                dh.contiguous(), gate_up, weight, None, ctx.L, 1.0, 0.0, 0, True
            )
            return d_gate_up, d_weight, None

    return {
        "forward": forward,
        "backward": backward,
        "autograd": lambda first, second, weight, L=x.L: Scaled.apply(first, weight, L),
        "writes_weight_grad": True,
        "version": f"transformer_engine {version}",
        "notes": {
            "ops": "transformer_engine_torch.scaled_clamped_swiglu / scaled_clamped_dswiglu (limit=L, alpha=1, "
            "glu_linear_offset=0, glu_interleave_size=0, compute_scale_grad=True, no quantizer)"
        },
    }


def _te_unweighted(tex: Any, version: Optional[str], x: types.SimpleNamespace) -> dict[str, Any]:
    """The ``te`` arm without a route weight: ``clamped_swiglu`` / ``clamped_dswiglu`` on the merged layout."""
    if not hasattr(tex, "clamped_swiglu") or not hasattr(tex, "clamped_dswiglu"):
        raise Unavailable(f"Transformer Engine {version} has no clamped_swiglu")

    # (input, quantizer, limit, alpha, glu_linear_offset) and the backward's
    # (grad, fwd_input, quantizer, limit, alpha, glu_linear_offset); no quantizer: outputs of the input's type.
    def forward(x):
        return tex.clamped_swiglu(x.gate_up, None, x.L, 1.0, 0.0)

    def backward(x):
        return tex.clamped_dswiglu(x.dh, x.gate_up, None, x.L, 1.0, 0.0), None, None

    class Clamped(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate_up, L):
            ctx.save_for_backward(gate_up)
            ctx.L = L
            return tex.clamped_swiglu(gate_up, None, L, 1.0, 0.0)

        @staticmethod
        def backward(ctx, dh):
            (gate_up,) = ctx.saved_tensors
            return tex.clamped_dswiglu(dh.contiguous(), gate_up, None, ctx.L, 1.0, 0.0), None

    return {
        "forward": forward,
        "backward": backward,
        "autograd": lambda first, second, weight, L=x.L: Clamped.apply(first, L),
        "writes_weight_grad": False,
        "version": f"transformer_engine {version}",
        "notes": {
            "ops": "transformer_engine_torch.clamped_swiglu / clamped_dswiglu (limit=L, alpha=1, glu_linear_offset=0, "
            "no quantizer)"
        },
    }


class _Declared(dict):
    """The input tensors of one cuDNN graph by name, each declared on first use (a graph holds only what it reads)."""

    def __init__(self, declare: Callable[[str], Any]) -> None:
        super().__init__()
        self.declare = declare

    def __missing__(self, name: str) -> Any:
        self[name] = value = self.declare(name)
        return value


class CudnnGraphs:
    """The cuDNN frontend graphs of one row: the forward, and the backward as one graph or as a two-graph
    composite, each run with the fastest of its plans."""

    def __init__(self, x: types.SimpleNamespace) -> None:
        try:
            import cudnn
        except Exception as exc:  # noqa: BLE001
            raise Unavailable(f"the cuDNN frontend does not import: {type(exc).__name__}: {exc}") from None
        self.c = cudnn
        self.x = x
        self.weighted = x.weight is not None
        self.handle = cudnn.create_handle()
        self.version = f"cudnn frontend {getattr(cudnn, '__version__', None)}, backend {cudnn.backend_version()}"
        scalar = {"L": x.L, "negL": -x.L, "one": 1.0, "zero": 0.0}
        self.values: dict[str, Any] = {"gate": x.gate, "up": x.up, "dh": x.dh, "w": x.weight}
        self.values.update({name: torch.full((1, 1, 1), v, dtype=torch.float32) for name, v in scalar.items()})
        self.errors: dict[str, str] = {}
        self.plans: dict[str, dict[str, Any]] = {}
        self.forward_graph = self._try_build("fwd", self._forward_ops)
        self.backward_graphs: Optional[list] = None
        self.backward_kind: Optional[str] = None
        one = self._try_build("bwd", self._backward_ops(pointwise=True, reduce=self.weighted))
        if one is not None:
            self.backward_graphs = [one]
            self.backward_kind = "one graph" + (" with the d_weight reduction" if self.weighted else "")
        elif self.weighted:
            parts = [
                self._try_build("bwd pointwise", self._backward_ops(pointwise=True, reduce=False)),
                self._try_build("bwd reduction", self._backward_ops(pointwise=False, reduce=True)),
            ]
            if all(part is not None for part in parts):
                self.backward_graphs = parts
                self.backward_kind = "two-graph composite (reads gate, up and dh twice)"

    def _declare(self, g, name: str) -> Any:
        c, R = self.c, self.x.R
        if name in ("gate", "up"):
            s = self.x.gate.stride(0)  # the up's row stride is the gate's in both layouts
            return g.tensor(name=name, dim=[1, R, I], stride=[R * s, s, 1], data_type=c.data_type.BFLOAT16)
        if name == "dh":
            return g.tensor(name=name, dim=[1, R, I], stride=[R * I, I, 1], data_type=c.data_type.BFLOAT16)
        if name == "w":
            return g.tensor(name=name, dim=[1, R, 1], stride=[R, 1, 1], data_type=c.data_type.FLOAT)
        scalar = c.data_type.FLOAT
        return g.tensor(name=name, dim=[1, 1, 1], stride=[1, 1, 1], data_type=scalar, is_pass_by_value=True)

    def _bf16_output(self, value, row_stride: int) -> Any:
        R = self.x.R
        return (
            value.set_output(True)
            .set_data_type(self.c.data_type.BFLOAT16)
            .set_dim([1, R, I])
            .set_stride([R * row_stride, row_stride, 1])
        )

    def _mask(self, value) -> Any:
        return value.set_data_type(self.c.data_type.BOOLEAN)

    def _clamped(self, g, t):
        gc = g.min(t["gate"], t["L"])
        uc = g.max(g.min(t["up"], t["L"]), t["negL"])
        return gc, uc

    def _forward_ops(self, g, t) -> dict[str, Any]:
        gc, uc = self._clamped(g, t)
        a = g.mul(g.swish(gc), uc)
        h = g.mul(a, t["w"]) if self.weighted else a
        return {"h": self._bf16_output(h, I)}

    def _backward_ops(self, *, pointwise: bool, reduce: bool):
        def ops(g, t) -> dict[str, Any]:
            c, R = self.c, self.x.R
            gc, uc = self._clamped(g, t)
            s = g.sigmoid(gc)
            silu = g.mul(gc, s)
            found = {}
            if pointwise:
                out_stride = 2 * I if self.x.second is None else I
                dws = g.mul(t["dh"], t["w"]) if self.weighted else t["dh"]
                dsilu = g.add(g.mul(silu, g.sub(t["one"], s)), s)
                # binary_select takes a BOOLEAN mask; the graph's FP32 intermediate type would otherwise apply to it
                # (CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE).  The up mask is two nested selects, one per bound.
                gate_in = self._mask(g.cmp_le(t["gate"], t["L"]))
                d_gate = g.binary_select(g.mul(g.mul(dws, uc), dsilu), t["zero"], gate_in)
                d_up = g.binary_select(
                    g.binary_select(g.mul(dws, silu), t["zero"], self._mask(g.cmp_le(t["up"], t["L"]))),
                    t["zero"],
                    self._mask(g.cmp_ge(t["up"], t["negL"])),
                )
                found["d_gate"] = self._bf16_output(d_gate, out_stride)
                found["d_up"] = self._bf16_output(d_up, out_stride)
            if reduce:
                d_weight = g.reduction(g.mul(t["dh"], g.mul(silu, uc)), mode=c.reduction_mode.ADD)
                found["d_weight"] = (
                    d_weight.set_output(True).set_data_type(c.data_type.FLOAT).set_dim([1, R, 1]).set_stride([R, 1, 1])
                )
            return found

        return ops

    def _try_build(self, what: str, ops) -> Optional[types.SimpleNamespace]:
        """The graph of ``ops`` built and set to its fastest plan, or None with the error recorded under ``what``.

        The first attempt builds every plan of the heuristics modes A, B and FALLBACK and times them, the second (if
        the first is refused) every plan of the modes A and FALLBACK; both deselect the runtime-compiled engines unless
        ``--cudnn-runtime-compiled`` keeps them.  The last builds the first plan of the modes A and FALLBACK that builds
        (the frontend's default policy) and deselects nothing: it runs that plan whatever its behavior notes, a
        runtime-compiled engine included.  The record under ``self.plans[what]`` holds ``heuristics``, ``policy``
        (``ALL`` or ``default``), ``runtime_compiled_engines`` (``included``, ``excluded`` or ``not filtered``), the
        ``refused`` attempts, and the selection: the :meth:`_select_plan` record, or under ``selection`` why there
        was none."""
        c = self.c
        all_plans = getattr(getattr(c, "build_plan_policy", None), "ALL", None)
        attempts = [(("A", "FALLBACK"), None)]
        if all_plans is not None:
            attempts.insert(0, (("A", "FALLBACK"), all_plans))
            if hasattr(c.heur_mode, "B"):
                attempts.insert(0, (("A", "B", "FALLBACK"), all_plans))
        refused = []
        for modes, policy in attempts:
            try:
                built = self._build(ops, modes, policy)
            except Exception as exc:  # noqa: BLE001 - a graph no engine builds is reported with its error
                how = "default" if policy is None else "ALL"
                refused.append(f"modes {'+'.join(modes)}, policy {how}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            record: dict[str, Any] = {"heuristics": list(modes), "policy": "default" if policy is None else "ALL"}
            record["runtime_compiled_engines"] = (
                "not filtered (the first plan that builds)"
                if policy is None
                else ("included" if CUDNN_OPTIONS["runtime_compiled"] else "excluded")
            )
            if refused:
                record["refused"] = refused
            if policy is None:
                record["selection"] = "none: " + (
                    "building every plan was refused" if refused else "the frontend has no build_plan_policy.ALL"
                )
            else:
                record.update(self._select_plan(built))
            self.plans[what] = record
            return built
        self.errors[what] = "; ".join(refused)
        return None

    def _build(self, ops, modes: tuple[str, ...], policy: Any) -> types.SimpleNamespace:
        """The graph of ``ops`` with the plans of the heuristics ``modes`` built under ``policy`` (None: the frontend's
        default, the first plan that builds)."""
        c = self.c
        g = c.pygraph(
            io_data_type=c.data_type.BFLOAT16,
            intermediate_data_type=c.data_type.FLOAT,
            compute_data_type=c.data_type.FLOAT,
            handle=self.handle,
        )
        t = _Declared(lambda name: self._declare(g, name))
        outs = ops(g, t)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([getattr(c.heur_mode, mode) for mode in modes])
        if policy is not None and not CUDNN_OPTIONS["runtime_compiled"]:
            # Deselect right after create_execution_plans, before check_support and build_plans, so that neither
            # builds a runtime-compiled engine.
            note = getattr(getattr(c, "behavior_note", None), "RUNTIME_COMPILATION", None)
            if note is None or not hasattr(g, "deselect_behavior_notes"):
                raise RuntimeError("the frontend cannot deselect the runtime-compiled engines")
            g.deselect_behavior_notes([note])
        g.check_support()
        if policy is None:
            g.build_plans()
        else:
            g.build_plans(policy)
        workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
        return types.SimpleNamespace(graph=g, inputs=t, outputs=outs, workspace=workspace, index=None)

    def _select_plan(self, built: types.SimpleNamespace) -> dict[str, Any]:
        """Times every plan of ``built`` and sets ``built.index`` and ``built.workspace`` to the fastest one's; returns
        the record: ``candidates`` (the number of plans), ``timed`` (how many ran), ``candidate_us`` (per plan its
        ``index``, ``name``, ``behavior_notes``, ``workspace_bytes``, ``us`` and ``timer``, or ``unavailable`` with
        the error; ``graph_unavailable`` when its capture failed and eager calls timed it), ``ranked_by``,
        ``chosen_index``, ``chosen_name`` and ``chosen_us``.  The plans that run in a CUDA graph are ranked by their
        replay time (GPU time); a plan that does not is ranked by its eager time only when no plan does.  When the
        selection cannot run, ``built.index`` stays None (the frontend's selected plan runs) and ``selection`` says
        why."""
        g = built.graph
        calls = ("get_execution_plan_count", "get_workspace_size_plan_at_index", "execute_plan_at_index")
        missing = [name for name in calls if not hasattr(g, name)]
        if missing:
            return {"selection": f"none: the frontend has no {', '.join(missing)}; the default plan runs"}
        try:
            outputs = (
                {"h": torch.empty(self.x.R, I, dtype=torch.bfloat16, device="cuda")} if "h" in built.outputs else {}
            )
            outputs.update(self._gradients(self.x, "d_gate" in built.outputs, "d_weight" in built.outputs)[3])
            count = int(g.get_execution_plan_count())
        except Exception as exc:  # noqa: BLE001 - the default plan runs, with the reason recorded
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            return {"selection": f"none: {type(exc).__name__}: {str(exc)[:200]}; the default plan runs"}
        candidates = []
        for index in range(count):
            entry: dict[str, Any] = {"index": index}
            with contextlib.suppress(Exception):
                entry["name"] = str(g.get_plan_name_at_index(index))
            with contextlib.suppress(Exception):
                entry["behavior_notes"] = [str(note) for note in g.get_behavior_notes_for_plan_at_index(index)]
            try:
                entry.update(self._time_plan(built, outputs, index))
            except Exception as exc:  # noqa: BLE001 - a plan that does not build or run is recorded and skipped
                entry["unavailable"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            candidates.append(entry)
        timed = [entry for entry in candidates if "us" in entry]
        captured = [entry for entry in timed if entry["timer"] == "graph"]
        record: dict[str, Any] = {"candidates": count, "timed": len(timed), "candidate_us": candidates}
        if not timed:
            record["selection"] = "none: no candidate plan ran; the default plan runs"
            return record
        record["ranked_by"] = (
            f"CUDA graph replay of {PLAN_LOOP} calls (GPU time)"
            if captured
            else f"{PLAN_LOOP} eager calls (no plan runs in a CUDA graph; includes the host's issue time)"
        )
        best = min(captured or timed, key=lambda entry: entry["us"])
        built.index = best["index"]
        built.workspace = torch.empty(max(best["workspace_bytes"], 1), dtype=torch.uint8, device="cuda")
        record.update(chosen_index=best["index"], chosen_name=best.get("name"), chosen_us=best["us"])
        return record

    def _time_plan(self, built: types.SimpleNamespace, outputs: dict[str, torch.Tensor], index: int) -> dict[str, Any]:
        """The workspace bytes and warm per-call microseconds of the plan ``index``: ``PLAN_LOOP`` calls captured in
        one CUDA graph and replayed (``timer: graph``), or, if the capture fails, ``PLAN_LOOP`` eager calls (``timer:
        eager``, with the capture's error under ``graph_unavailable``)."""
        size = int(built.graph.get_workspace_size_plan_at_index(index))
        workspace = torch.empty(max(size, 1), dtype=torch.uint8, device="cuda")
        found: dict[str, Any] = {"workspace_bytes": size}

        def call():
            self._execute(built, outputs, index=index, workspace=workspace)

        try:
            found.update(us=_graph_ms(call, PLAN_LOOP, PLAN_WARMUP, PLAN_ROUNDS) * 1e3, timer="graph")
        except Exception as exc:  # noqa: BLE001 - a plan the capture refuses is timed over eager calls
            found["graph_unavailable"] = f"{type(exc).__name__}: {str(exc)[:200]}"

            def run():
                for _ in range(PLAN_LOOP):
                    call()

            found.update(us=_events_ms(run, PLAN_LOOP, PLAN_WARMUP, PLAN_ROUNDS) * 1e3, timer="eager")
        return found

    def _execute(
        self,
        built: types.SimpleNamespace,
        outputs: dict[str, torch.Tensor],
        *,
        index: Optional[int] = None,
        workspace: Optional[torch.Tensor] = None,
    ) -> None:
        """Runs ``built`` with the plan ``index`` (default: the chosen one; none chosen: the graph's default plan)."""
        pack = {tensor: self.values[name] for name, tensor in built.inputs.items()}
        pack.update({tensor: outputs[name] for name, tensor in built.outputs.items()})
        self.c.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        index = built.index if index is None else index
        workspace = built.workspace if workspace is None else workspace
        if index is None:
            built.graph.execute(pack, workspace, handle=self.handle)
        else:
            built.graph.execute_plan_at_index(pack, workspace, index, handle=self.handle)

    def _gradients(self, x, pointwise: bool, reduce: bool) -> tuple:
        """Fresh ``(d_first, d_up, d_weight, {graph output name: tensor})``: ``d_gate`` / ``d_up`` when ``pointwise``
        (views of one ``[R, 2I]`` ``d_first`` in the merged layout), ``d_weight`` when ``reduce``."""
        d_first = d_up = d_weight = None
        outputs = {}
        if pointwise:
            merged = x.second is None
            d_first = torch.empty(x.R, 2 * I if merged else I, dtype=torch.bfloat16, device="cuda")
            d_up = None if merged else torch.empty(x.R, I, dtype=torch.bfloat16, device="cuda")
            outputs["d_gate"], outputs["d_up"] = (d_first[:, :I], d_first[:, I:]) if merged else (d_first, d_up)
        if reduce:
            d_weight = outputs["d_weight"] = torch.empty(x.R, dtype=torch.float32, device="cuda")
        return d_first, d_up, d_weight, outputs

    def forward(self, x) -> torch.Tensor:
        if self.forward_graph is None:
            raise Unavailable(f"no cuDNN forward graph builds: {self.errors.get('fwd')}")
        h = torch.empty(x.R, I, dtype=torch.bfloat16, device="cuda")
        self._execute(self.forward_graph, {"h": h})
        return h

    def backward(self, x):
        if self.backward_graphs is None:
            raise Unavailable(f"no cuDNN backward graph builds: {self.errors}")
        d_first, d_up, d_weight, outputs = self._gradients(x, True, self.weighted)
        for built in self.backward_graphs:
            self._execute(built, outputs)
        return d_first, d_up, d_weight

    def backward_parts(self) -> dict[str, Callable]:
        """The graphs of a composite backward as separate calls (``pointwise``: ``d_gate`` and ``d_up``;
        ``reduction``: ``d_weight``), each allocating its outputs; empty for a one-graph backward."""
        if self.backward_graphs is None or len(self.backward_graphs) < 2:
            return {}
        pointwise, reduction = self.backward_graphs

        def run_pointwise(x):
            d_first, d_up, _, outputs = self._gradients(x, True, False)
            self._execute(pointwise, outputs)
            return d_first, d_up

        def run_reduction(x):
            _, _, d_weight, outputs = self._gradients(x, False, True)
            self._execute(reduction, outputs)
            return d_weight

        return {"pointwise": run_pointwise, "reduction": run_reduction}


def cudnn_arm(x: types.SimpleNamespace) -> dict[str, Any]:
    graphs = CudnnGraphs(x)
    return {
        "forward": graphs.forward,
        "backward": graphs.backward,
        "autograd": None,
        "writes_weight_grad": x.weight is not None,
        "version": graphs.version,
        "bwd_parts": graphs.backward_parts(),
        "notes": {"backward": graphs.backward_kind, "errors": graphs.errors, "plans": graphs.plans},
    }


def make_arm(name: str, x: types.SimpleNamespace) -> dict[str, Any]:
    if name == "cake":
        return cake_arm(x)
    if name == "te":
        return te_arm(x)
    if name in ("torch_compile", "eager"):
        return _formula_arm(x, compiled=name == "torch_compile")
    if name == "cudnn":
        return cudnn_arm(x)
    raise ValueError(f"unknown arm {name!r} (one of {', '.join(ARMS)})")


# ----------------------------------------------------------------------------------------------------------------
# Rows


def _stage_calls(arm: dict[str, Any], x: types.SimpleNamespace) -> dict[str, Callable[[], object]]:
    calls: dict[str, Callable[[], object]] = {
        "fwd": lambda: arm["forward"](x),
        "bwd": lambda: arm["backward"](x),
        "fwd+bwd": lambda: (arm["forward"](x), arm["backward"](x)),
    }
    if arm["autograd"] is not None:
        leaves = [None if t is None else t.detach().clone().requires_grad_(True) for t in (x.first, x.second, x.weight)]
        wants = [t for t in leaves if t is not None]

        def ckpt():
            h = torch.utils.checkpoint.checkpoint(
                arm["autograd"], *leaves, use_reentrant=False, preserve_rng_state=False
            )
            return torch.autograd.grad(h, wants, x.dh)

        calls["ckpt"] = ckpt
    return calls


def _saved_bytes(arm: dict[str, Any], x: types.SimpleNamespace) -> Optional[int]:
    """The bytes the arm's autograd form saves for its backward (its retained state), or None without one."""
    if arm["autograd"] is None:
        return None
    seen: dict[int, int] = {}

    def pack(t):
        seen[t.data_ptr()] = max(seen.get(t.data_ptr(), 0), t.numel() * t.element_size())
        return t

    leaves = [None if t is None else t.detach().clone().requires_grad_(True) for t in (x.first, x.second, x.weight)]
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        arm["autograd"](*leaves)
    return sum(seen.values())


def _compare(found: dict[str, torch.Tensor], cake: dict[str, torch.Tensor]) -> dict[str, Any]:
    out = {}
    for name, want in cake.items():
        got = found.get(name)
        if got is None:
            out[name] = "missing"
            continue
        if got.shape != want.shape:
            out[name] = f"shape {tuple(got.shape)} != {tuple(want.shape)}"
            continue
        diff = (got.float() - want.float()).abs()
        got = got.to(want.dtype)
        view = torch.int16 if want.dtype == torch.bfloat16 else torch.int32
        same_bits = got.contiguous().view(view) == want.contiguous().view(view)
        same_value = (got == want) | (torch.isnan(got) & torch.isnan(want))  # +0 == -0, NaN == NaN
        zero_sign = (got == 0) & (want == 0) & ~same_bits
        n = want.numel()
        out[name] = {
            "max_abs_diff": float(diff.max()) if n else 0.0,
            "values_differ": int(torch.count_nonzero(~same_value)) / n if n else 0.0,
            "zero_sign_only": int(torch.count_nonzero(zero_sign)) / n if n else 0.0,
            "bits_differ": int(torch.count_nonzero(~same_bits)) / n if n else 0.0,
        }
    return out


# The first calls a stage needs: a stage times only when each of them ran once (``autograd``: the autograd forward
# that measures the saved bytes).
NEEDS = {"fwd": ("fwd",), "bwd": ("bwd",), "fwd+bwd": ("fwd", "bwd"), "ckpt": ("fwd", "bwd", "autograd")}
# The code objects of a compiled arm each stage runs: a stage times only when each holds a compiled Dynamo entry.
CODES = {"fwd": ("fwd",), "bwd": ("bwd",), "fwd+bwd": ("fwd", "bwd"), "ckpt": ("autograd fwd", "autograd bwd")}


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


def _time_call(
    call: Callable[[], object], stage: str, arm: dict[str, Any], x: types.SimpleNamespace, args: argparse.Namespace
) -> dict[str, Any]:
    """The entry of one stage call: ``ms`` per timer, the DRAM bytes and the ``cupti`` bandwidth, or the reason it is
    unavailable.  One untimed call (compilation, plans, caches) precedes the timers.  For a compiled arm the stage
    times only when each code object it runs holds a compiled Dynamo entry after that call (the counts are in the
    entry); if Dynamo compiles a graph while the timers run, the stage is timed again, and is unavailable if it
    compiles again."""
    missing: dict[str, str] = {}
    _first(lambda: (call(), torch.cuda.synchronize()), missing, "call")
    if missing:
        return {"unavailable": missing["call"]}
    codes = arm.get("codes") or {}
    entries = {name: _cache_entries(codes[name]) for name in CODES.get(stage, ()) if name in codes}
    unknown = [name for name, count in entries.items() if count is None]
    if unknown:
        return {
            "unavailable": "torch.compile: this torch has no Dynamo cache-entry lookup "
            f"(torch._dynamo.eval_frame._debug_get_cache_entry_list), so {', '.join(unknown)} cannot be shown to run "
            "a compiled graph"
        }
    eager = [name for name, count in entries.items() if count == 0]
    if eager:
        return {"unavailable": f"torch.compile ran {', '.join(eager)} without a compiled graph (no Dynamo cache entry)"}
    if arm.get("compiled") and _dynamo_graphs() == 0:
        return {
            "unavailable": "torch.compile: Dynamo's graph counter (counters['stats']['unique_graphs']) counts no graph "
            "after a compiled call, so a graph compiled while the timers run could not be seen"
        }
    graphs = _dynamo_graphs if arm.get("compiled") else lambda: 0
    compiled_while: list[int] = []  # the graphs Dynamo compiled during each timing that is repeated
    for _ in range(2):
        before = graphs()
        try:
            times = time_stage(call, tuple(args.timers), args)
        except Unavailable as exc:
            return {"unavailable": str(exc)}
        if graphs() == before:
            break
        compiled_while.append(graphs() - before)
    else:
        counts = " and ".join(str(count) for count in compiled_while)
        return {"unavailable": f"Dynamo compiled {counts} graph(s) while the timers ran, in both timings"}
    moved = dram_bytes(x, stage, writes_weight_grad=arm["writes_weight_grad"])
    entry: dict[str, Any] = {"ms": times, "dram_bytes": moved}
    if isinstance(times.get("cupti"), float):
        entry["cupti_gbs"] = moved / times["cupti"] / 1e6
    if arm.get("compiled"):
        entry["dynamo_graphs_compiled_while_timing"] = 0
        entry["dynamo_cache_entries"] = entries
        if compiled_while:
            entry["retimed_after_dynamo_compiled"] = compiled_while[0]
    return entry


def bench_row(rows: int, layout: str, args: argparse.Namespace) -> dict[str, Any]:
    x = make_inputs(rows, layout, not args.unweighted, args.limit, args.seed, args.deep_gate)
    result: dict[str, Any] = {
        "R": rows,
        "I": I,
        "layout": layout,
        "weighted": x.weight is not None,
        "L": x.L,
        "deep_gate": args.deep_gate,
        "arms": {},
    }
    reference = None
    for name in args.arms:
        row: dict[str, Any] = {}
        result["arms"][name] = row
        try:
            arm = make_arm(name, x)
        except Unavailable as exc:
            row["unavailable"] = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            row["unavailable"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            continue
        row["version"] = arm["version"]
        row.update(arm.get("notes", {}))
        first: dict[str, Any] = {}
        missing: dict[str, str] = {}
        for stage, call in (("fwd", arm["forward"]), ("bwd", arm["backward"])):
            first[stage] = _first(lambda call=call: call(x), missing, stage)
        outputs = _outputs(x, first.get("fwd"), first.get("bwd"))
        if name == "cake":
            reference = outputs
        elif reference is not None:
            row["vs_cake"] = _compare(outputs, reference)
        del first
        row["saved_bytes"] = _first(lambda: _saved_bytes(arm, x), missing, "autograd") if not missing else None
        row["workspace_bytes"] = 0 if name == "cake" else None
        row["stages"] = {}
        for stage, call in _stage_calls(arm, x).items():
            reasons = [missing[part] for part in NEEDS[stage] if part in missing]
            if reasons:
                row["stages"][stage] = {"unavailable": "; ".join(reasons)}
                continue
            entry = row["stages"][stage] = _time_call(call, stage, arm, x, args)
            if stage == "bwd" and "ms" in entry and arm.get("bwd_parts"):
                entry["parts"] = {
                    part: _time_call(lambda fn=fn: fn(x), f"bwd {part}", arm, x, args)
                    for part, fn in arm["bwd_parts"].items()
                }
        if "ckpt" not in row["stages"]:
            row["stages"]["ckpt"] = {"unavailable": "no autograd form"}
        if arm.get("codes"):
            row["compile"]["cache_entries"] = {name: _cache_entries(code) for name, code in arm["codes"].items()}
    cake = result["arms"].get("cake", {}).get("stages", {})
    for row in result["arms"].values():
        for stage, entry in row.get("stages", {}).items():
            base = cake.get(stage, {}).get("ms", {})
            ratios = {}
            for timer, value in entry.get("ms", {}).items():
                if isinstance(value, float) and isinstance(base.get(timer), float):
                    ratios[timer] = value / base[timer]
            if ratios:
                entry["vs_cake"] = ratios
    return result


def _cell(entry: dict[str, Any], timer: str) -> str:
    value = entry.get("ms", {}).get(timer)
    if isinstance(value, float):
        ratio = entry.get("vs_cake", {}).get(timer)
        return f"{value * 1e3:9.1f}" + (f" ({ratio:5.3f})" if ratio is not None else "")
    return "unavailable" if value is not None else "-"


def format_row(result: dict[str, Any], timers: tuple[str, ...]) -> list[str]:
    head = (
        f"R={result['R']} I={result['I']} layout={result['layout']} "
        f"{'weighted' if result['weighted'] else 'unweighted'} L={result['L']:g}"
        + (" deep gate" if result.get("deep_gate") else "")
    )
    lines = [f"### {head}", ""]
    cols = " | ".join(f"{timer} us (baseline/cake)" for timer in timers)
    lines.append(f"| arm | stage | {cols} | cupti GB/s | saved MiB |")
    lines.append("|---|---|" + "---|" * len(timers) + "---|---|")
    notes = []
    for name, row in result["arms"].items():
        if "unavailable" in row:
            lines.append(f"| {name} | all | " + " | ".join("unavailable" for _ in timers) + " | - | - |")
            notes.append(f"- {name}: unavailable: {row['unavailable']}")
            continue
        saved = row.get("saved_bytes")
        saved_text = f"{saved / 2**20:.1f}" if isinstance(saved, int) else "-"
        for stage, entry in row["stages"].items():
            parts = [(f"{stage} {part} graph", part_entry) for part, part_entry in entry.get("parts", {}).items()]
            for label, item in [(stage, entry), *parts]:
                if "unavailable" in item:
                    lines.append(f"| {name} | {label} | " + " | ".join("unavailable" for _ in timers) + " | - | - |")
                    notes.append(f"- {name} {label}: unavailable: {item['unavailable']}")
                    continue
                gbs = item.get("cupti_gbs")
                gbs_text = f"{gbs:.0f}" if gbs is not None else "-"
                cells = " | ".join(_cell(item, timer) for timer in timers)
                lines.append(f"| {name} | {label} | {cells} | {gbs_text} | {saved_text if item is entry else '-'} |")
                for timer, value in item["ms"].items():
                    if isinstance(value, str):
                        notes.append(f"- {name} {label} {timer}: {value}")
                if item.get("retimed_after_dynamo_compiled"):
                    count = item["retimed_after_dynamo_compiled"]
                    notes.append(f"- {name} {label}: Dynamo compiled {count} graph(s) while timing; timed again")
        if "vs_cake" in row:
            notes.append(f"- {name} outputs vs cake: {json.dumps(row['vs_cake'])}")
        for key in ("ops", "compile", "merged_gradient", "backward", "errors"):
            if row.get(key):
                notes.append(f"- {name} {key}: {row[key]}")
        for graph, plan in row.get("plans", {}).items():
            notes.append(f"- {name} plan {graph}: {_plan_text(plan)}")
        notes.append(f"- {name} version: {row.get('version')}")
    return lines + [""] + notes + [""]


def _plan_text(plan: dict[str, Any]) -> str:
    """One line of a cuDNN graph's plan record."""
    text = f"heuristics {'+'.join(plan['heuristics'])}, policy {plan['policy']}"
    if "chosen_index" in plan:
        text += (
            f", {plan['timed']} of {plan['candidates']} candidates timed, ranked by {plan.get('ranked_by')}, chosen "
            f"[{plan['chosen_index']}] {plan.get('chosen_name')} ({plan['chosen_us']:.1f} us warm)"
        )
    if "selection" in plan:
        text += f", selection {plan['selection']}"
    if "refused" in plan:
        text += f"; refused: {plan['refused']}"
    return text


# ----------------------------------------------------------------------------------------------------------------
# Host path (``--host``)

# The entry points ``--host`` times: the forward step, the backward step and one autograd step.
HOST_ENTRIES = ("fwd", "bwd", "autograd fwd+bwd")


def _host_calls(x: types.SimpleNamespace) -> dict[str, Callable[[], object]]:
    """Cake's entry points on the inputs ``x``, by :data:`HOST_ENTRIES` name: the forward step, the backward step,
    and the autograd forward followed by ``torch.autograd.grad`` with ``dh`` (fresh leaves that require grad)."""
    leaves = [None if t is None else t.detach().clone().requires_grad_(True) for t in (x.first, x.second, x.weight)]
    wants = [t for t in leaves if t is not None]

    def autograd_step():
        h = cake_clamped_swiglu_det(*leaves, limit=x.L)
        return torch.autograd.grad(h, wants, x.dh)

    return {
        "fwd": lambda: cake_clamped_swiglu_det_train_forward(x.first, x.second, x.weight, limit=x.L),
        "bwd": lambda: cake_clamped_swiglu_det_train_backward(x.dh, x.first, x.second, x.weight, limit=x.L),
        "autograd fwd+bwd": autograd_step,
    }


def _spread(samples: list[float]) -> dict[str, float]:
    return {"median": float(statistics.median(samples)), "min": min(samples), "max": max(samples)}


def _host_times(fn: Callable[[], object], calls: int, rounds: int, warmup: int) -> dict[str, Any]:
    """Per-call microseconds of ``calls`` back-to-back eager calls of ``fn``, ``rounds`` times after ``warmup`` untimed
    calls: ``host_us`` is ``time.perf_counter`` around the calls (the host's issue time: a call returns before its
    kernels finish), ``device_us`` the CUDA events recorded before the first and after the last call (the device wall
    time, which is the issue time when the host is the bottleneck); each as the median, lowest and highest round."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    host, device = [], []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        t1 = time.perf_counter()
        end.record()
        torch.cuda.synchronize()
        host.append((t1 - t0) / calls * 1e6)
        device.append(start.elapsed_time(end) / calls * 1e3)
    return {"calls": calls, "rounds": rounds, "host_us": _spread(host), "device_us": _spread(device)}


def bench_host(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The ``--host`` records: every entry point of :data:`HOST_ENTRIES`, merged and split, weighted and unweighted,
    at each of ``--host-rows`` rows, all in this process."""
    records = []
    print("| entry | R | layout | weight | host us per call (median [min..max]) | device us per call |")
    print("|---|---|---|---|---|---|")
    for rows in args.host_rows:
        for layout in args.layouts:
            for weighted in (True, False):
                x = make_inputs(rows, layout, weighted, args.limit, args.seed, args.deep_gate)
                for entry, fn in _host_calls(x).items():
                    record = {"entry": entry, "R": rows, "layout": layout, "weighted": weighted}
                    record.update(_host_times(fn, args.host_calls, args.host_rounds, args.host_warmup))
                    records.append(record)
                    host, device = record["host_us"], record["device_us"]
                    print(
                        f"| {entry} | {rows} | {layout} | {'weighted' if weighted else 'unweighted'} | "
                        f"{host['median']:.2f} [{host['min']:.2f}..{host['max']:.2f}] | "
                        f"{device['median']:.2f} [{device['min']:.2f}..{device['max']:.2f}] |",
                        flush=True,
                    )
                del x
                torch.cuda.empty_cache()
    return records


# ----------------------------------------------------------------------------------------------------------------
# Identity and main


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _cpu_model() -> Optional[str]:
    """The CPU model of ``/proc/cpuinfo``: its first ``model name``, else (Arm) its implementer and part."""
    try:
        with open("/proc/cpuinfo") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    fields: dict[str, str] = {}
    for line in lines:
        key, sep, value = line.partition(":")
        if sep:
            fields.setdefault(key.strip(), value.strip())
    if "model name" in fields:
        return fields["model name"]
    parts = [f"{key} {fields[key]}" for key in ("CPU implementer", "CPU part") if key in fields]
    return ", ".join(parts) or None


def _host_identity() -> dict[str, Any]:
    """The interpreter and the host CPU the timings ran on."""
    found: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "page_bytes": None,
        "affinity_cpus": None,
    }
    with contextlib.suppress(AttributeError, OSError, ValueError):
        found["page_bytes"] = os.sysconf("SC_PAGE_SIZE")
    with contextlib.suppress(AttributeError, OSError):
        found["affinity_cpus"] = len(os.sched_getaffinity(0))
    return found


def identity() -> dict[str, Any]:
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    dists = (
        "cake-pplx-clamped-swiglu",
        "apache-tvm-ffi",
        "torch-c-dlpack-ext",
        "torch",
        "triton",
        "transformer-engine",
        "transformer-engine-cu13",
        "transformer-engine-torch",
        "nvidia-cudnn-frontend",
        "nvidia-cudnn-cu13",
    )
    out: dict[str, Any] = {
        "gpu": p.name,
        "uuid": f"GPU-{p.uuid}",
        "cc": list(torch.cuda.get_device_capability()),
        "sms": p.multi_processor_count,
        "l2_bytes": p.L2_cache_size,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "versions": {d: _version(d) for d in dists},
        "host": _host_identity(),
        "determinism": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "fill_uninitialized_memory": None,
        },
    }
    with contextlib.suppress(ImportError, AttributeError):
        deterministic = importlib.import_module("torch.utils.deterministic")
        out["determinism"]["fill_uninitialized_memory"] = deterministic.fill_uninitialized_memory
    # Cake's tensor arguments go through tvm-ffi, whose import may give torch.Tensor the C DLPack exchange API.
    try:
        importlib.import_module("tvm_ffi")
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        out["tvm_ffi_import"] = f"{type(exc).__name__}: {exc}"
    out["torch_dlpack_c_exchange_api"] = hasattr(torch.Tensor, "__dlpack_c_exchange_api__")
    query = ["nvidia-smi", f"--id=GPU-{p.uuid}", "--query-gpu=driver_version,power.limit,clocks.max.sm"]
    with contextlib.suppress(Exception):
        q = subprocess.run([*query, "--format=csv,noheader"], capture_output=True, text=True, timeout=20, check=False)
        out["driver_power_maxclock"] = q.stdout.strip()
    return out


def copy_bandwidth() -> dict[str, float]:
    """A 2 GiB device-to-device copy under the ``cupti`` timer: bytes read plus written over the median time."""
    n = 1 << 31
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    ms = float(statistics.median(_cupti_call_ms(lambda: dst.copy_(src), repeat_iters=20)))
    return {"bytes": 2 * n, "ms": ms, "gbs": 2 * n / ms / 1e6}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Times the Cake clamped SwiGLU training kernels and the comparators of the same operation."
    )
    parser.add_argument("--rows", type=int, nargs="+", default=[98304, 196608, 4097])
    parser.add_argument("--layouts", nargs="+", default=["merged", "split"], choices=["merged", "split"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--timers", nargs="+", default=list(TIMERS), choices=list(TIMERS))
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--unweighted", action="store_true", help="no route weight (a shared expert)")
    parser.add_argument(
        "--deep-gate",
        action="store_true",
        help="gates at columns 0, 768 and 1536 of every row are -88 (every warp takes Cake's deep-gate sigmoid)",
    )
    parser.add_argument(
        "--cudnn-runtime-compiled",
        action="store_true",
        help="the cuDNN plan search also builds runtime-compiled engines (default: excluded)",
    )
    parser.add_argument("--loop", type=int, default=20, help="calls per eager / graph sample")
    parser.add_argument("--warmup", type=int, default=5, help="eager / graph samples before timing")
    parser.add_argument("--rounds", type=int, default=15, help="eager / graph samples timed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-bw", action="store_true", help="also time a 2 GiB device-to-device copy")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="run every arm under torch.use_deterministic_algorithms(True) with "
        "torch.utils.deterministic.fill_uninitialized_memory (each tensor PyTorch allocates is NaN-filled)",
    )
    parser.add_argument(
        "--host",
        action="store_true",
        help="time the host path of Cake's entry points (perf_counter and CUDA events) instead of the arms",
    )
    parser.add_argument("--host-rows", type=int, nargs="+", default=[1, 4097], help="rows of the --host calls")
    parser.add_argument("--host-calls", type=int, default=1000, help="back-to-back calls per --host round")
    parser.add_argument("--host-rounds", type=int, default=9, help="--host rounds per entry point")
    parser.add_argument("--host-warmup", type=int, default=100, help="untimed calls before the --host rounds")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    args.arms = list(dict.fromkeys(["cake", *args.arms]))  # cake first: the other arms are compared with its outputs
    CUDNN_OPTIONS["runtime_compiled"] = args.cudnn_runtime_compiled
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        importlib.import_module("torch.utils.deterministic").fill_uninitialized_memory = True
    ident = identity()
    ident["deep_gate"] = {"columns": list(DEEP_GATE_COLUMNS), "value": DEEP_GATE} if args.deep_gate else None
    print(
        f"# {ident['gpu']} (sm_{ident['cc'][0]}{ident['cc'][1]}, {ident['sms']} SMs), torch {ident['torch']} "
        f"(CUDA {ident['torch_cuda']}), driver/power/max clock {ident.get('driver_power_maxclock')}"
    )
    print(f"# versions: {json.dumps(ident['versions'])}")
    print(
        f"# host: {json.dumps(ident['host'])}; torch.Tensor.__dlpack_c_exchange_api__: "
        f"{ident['torch_dlpack_c_exchange_api']}; determinism: {json.dumps(ident['determinism'])}"
    )
    if args.host:
        records = bench_host(args)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump({"identity": ident, "args": vars(args), "host": records}, fh, indent=1)
        return 0
    if args.copy_bw:
        ident["copy"] = copy_bandwidth()
        copy = ident["copy"]
        print(f"# device-to-device copy: {copy['gbs']:.0f} GB/s ({copy['ms'] * 1e3:.1f} us for 4 GiB moved)")
    print(
        "# times in us per call; (baseline/cake) above 1: cake is faster; cupti = cold L2 kernel time, "
        f"eager / graph = warm L2 wall time per call over {args.loop} calls"
    )
    results = []
    for rows in args.rows:
        for layout in args.layouts:
            result = bench_row(rows, layout, args)
            results.append(result)
            for line in format_row(result, tuple(args.timers)):
                print(line, flush=True)
            torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"identity": ident, "args": vars(args), "rows": results}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
