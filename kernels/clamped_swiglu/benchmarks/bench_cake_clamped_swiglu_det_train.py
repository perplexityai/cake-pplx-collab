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

Arms (``--arms``; default: every arm):

* ``cake``: ``cake_clamped_swiglu_det_train_forward`` / ``_backward`` (one kernel each).
* ``te``: Transformer Engine ``transformer_engine_torch.scaled_clamped_swiglu`` / ``scaled_clamped_dswiglu`` with
  ``limit=L``, ``alpha=1``, ``glu_linear_offset=0`` and ``glu_interleave_size=0`` (halves): the same masks, one BF16
  rounding of ``w a`` and the FP32 ``d_weight = sum dh a`` from the unweighted ``a``.  It takes the merged ``[R, 2I]``
  layout and a route weight only; other rows print it as unavailable.
* ``torch_compile``: ``torch.compile`` (Inductor, ``dynamic=False``) of the explicit FP32 forward and backward formulas
  with explicit inclusive masks (``torch.where``), one compiled function each, so the backward fuses ``d_gate``,
  ``d_up`` and ``d_weight`` as Inductor chooses.
* ``eager``: the same formulas in eager PyTorch.
* ``cudnn``: a cuDNN frontend pointwise graph (``min``, ``max``, ``swish``, ``mul``; FP32 compute, BF16 output) for
  the forward.  For the backward the benchmark first builds one graph (pointwise ``d_gate`` / ``d_up`` with the masks
  as ``binary_select`` on BOOLEAN comparisons, and with a route weight the per-row ``reduction`` (ADD) of
  ``d_weight``), then, if a weighted one is refused, a pointwise graph for ``d_gate`` / ``d_up`` plus a pointwise +
  ``reduction`` graph for ``d_weight`` (a composite that reads the inputs twice).  A stage no graph builds prints as
  unavailable with the error.

Every arm runs on identical inputs: ``gate`` and ``up`` ``6 x`` unit normals, ``dh`` a unit normal (BF16) and the FP32
route weight ``1.5 x U[0, 1)`` (``--unweighted``: none).  ``--layouts``: ``merged`` (one ``[R, 2I]`` tensor and one
``[R, 2I]`` gradient) and ``split`` (two ``[R, I]`` tensors and gradients).  Every arm allocates its outputs inside the
timed call; compilation, graph and plan building happen before timing.  Each arm's outputs are compared with Cake's
(largest absolute difference and the share of elements whose bits differ).

Stages: ``fwd`` (one forward), ``bwd`` (one backward from ``gate``, ``up``, ``weight`` and ``dh``), ``fwd+bwd`` (the
two in sequence), and ``ckpt`` (an autograd step under ``torch.utils.checkpoint(..., use_reentrant=False,
preserve_rng_state=False)``: the forward, then the backward that replays it; nothing here draws random numbers), for
the arms with an autograd form.

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
GPU, the SM count, the CUDA version of torch and every library version; ``--json`` writes everything.

Example::

    python benchmarks/bench_cake_clamped_swiglu_det_train.py --rows 98304 196608 4097 --copy-bw --json rows.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as md
import json
import statistics
import subprocess
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
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        fn()  # libraries, plans and allocator pools before the capture
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.loop):
                        fn()
                found[timer] = _events_ms(graph.replay, args.loop, args.warmup, args.rounds)
                del graph
        except Unavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a timer that cannot run this call is reported, not fatal
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            found[timer] = f"unavailable: {type(exc).__name__}: {str(exc)[:240]}"
    return found


# ----------------------------------------------------------------------------------------------------------------
# Inputs


def make_inputs(rows: int, layout: str, weighted: bool, limit: float, seed: int) -> types.SimpleNamespace:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (6.0 * torch.randn(rows, 2 * I, device="cuda", generator=gen)).bfloat16()
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
    """The bytes a stage moves to and from DRAM when each input is read once and each output written once."""
    bf16_rows = x.R * I * 2
    weight = x.R * 4 if x.weight is not None else 0
    forward = 3 * bf16_rows + weight  # gate, up, weight in; h out
    backward = 5 * bf16_rows + weight + (x.R * 4 if writes_weight_grad else 0)  # gate, up, dh, weight in; grads out
    return {"fwd": forward, "bwd": backward, "fwd+bwd": forward + backward, "ckpt": 2 * forward + backward}[stage]


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
    gc = torch.where(gf <= L, gf, L)
    uc = torch.where(uf > L, L, torch.where(uf < -L, -L, uf))
    a = (gc * torch.sigmoid(gc)) * uc
    if w is not None:
        a = w[:, None] * a
    return a.to(torch.bfloat16)


def _formula_backward(dh, g, u, w, L: float, merged: bool):
    """``(d_first, d_up or None, d_weight or None)`` in FP32 with the inclusive masks, each BF16 output rounded once;
    a merged layout's gradient as one ``[R, 2I]`` tensor."""
    gf, uf, d = g.float(), u.float(), dh.float()
    gc = torch.where(gf <= L, gf, L)
    uc = torch.where(uf > L, L, torch.where(uf < -L, -L, uf))
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


def _formula_arm(x: types.SimpleNamespace, compiled: bool) -> dict[str, Any]:
    fwd, bwd = _formula_forward, _formula_backward
    if compiled:
        for knob in ("cache_size_limit", "recompile_limit"):  # one compile per row count, layout and weight flag
            if hasattr(torch._dynamo.config, knob):
                setattr(torch._dynamo.config, knob, max(getattr(torch._dynamo.config, knob), 64))
        fwd = torch.compile(_formula_forward, dynamic=False)
        bwd = torch.compile(_formula_backward, dynamic=False)

    class Formula(torch.autograd.Function):
        @staticmethod
        def forward(ctx, first, second, weight, L):
            ctx.save_for_backward(first, second, weight)
            ctx.L = L
            return fwd(*_halves(first, second), weight, L)

        @staticmethod
        def backward(ctx, dh):
            first, second, weight = ctx.saved_tensors
            d_first, d_up, d_weight = bwd(dh, *_halves(first, second), weight, ctx.L, second is None)
            return d_first, d_up, d_weight, None

    def backward(x):
        return bwd(x.dh, x.gate, x.up, x.weight, x.L, x.second is None)

    return {
        "forward": lambda x: fwd(x.gate, x.up, x.weight, x.L),
        "backward": backward,
        "autograd": lambda first, second, weight, L=x.L: Formula.apply(first, second, weight, L),
        "writes_weight_grad": x.weight is not None,
        "version": f"torch {torch.__version__}" + (f", triton {_version('triton')}" if compiled else ""),
    }


def te_arm(x: types.SimpleNamespace) -> dict[str, Any]:
    if x.layout != "merged":
        raise Unavailable("TE's scaled clamped SwiGLU takes the merged [R, 2I] layout only (a split call needs a copy)")
    if x.weight is None:
        raise Unavailable("TE's scaled clamped SwiGLU needs a route weight (its unweighted form is another op)")
    try:
        import transformer_engine
        import transformer_engine.pytorch  # noqa: F401  (loads the transformer_engine_torch extension)
        import transformer_engine_torch as tex
    except Exception as exc:  # noqa: BLE001
        raise Unavailable(f"transformer_engine_torch does not import: {type(exc).__name__}: {exc}") from None
    version = getattr(transformer_engine, "__version__", None) or _version("transformer-engine")
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
    composite."""

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

    def _try_build(self, what: str, ops) -> Optional[tuple]:
        """``(graph, inputs, outputs, workspace)``, or None with the error recorded under ``what``."""
        c = self.c
        try:
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
            g.create_execution_plans([c.heur_mode.A, c.heur_mode.FALLBACK])
            g.check_support()
            g.build_plans()
            workspace = torch.empty(max(int(g.get_workspace_size()), 1), dtype=torch.uint8, device="cuda")
        except Exception as exc:  # noqa: BLE001 - a graph no engine builds is reported with its error
            self.errors[what] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return None
        return g, t, outs, workspace

    def _execute(self, built, outputs: dict[str, torch.Tensor]) -> None:
        g, t, outs, workspace = built
        pack = {tensor: self.values[name] for name, tensor in t.items()}
        pack.update({tensor: outputs[name] for name, tensor in outs.items()})
        self.c.set_stream(handle=self.handle, stream=torch.cuda.current_stream().cuda_stream)
        g.execute(pack, workspace, handle=self.handle)

    def forward(self, x) -> torch.Tensor:
        if self.forward_graph is None:
            raise Unavailable(f"no cuDNN forward graph builds: {self.errors.get('fwd')}")
        h = torch.empty(x.R, I, dtype=torch.bfloat16, device="cuda")
        self._execute(self.forward_graph, {"h": h})
        return h

    def backward(self, x):
        if self.backward_graphs is None:
            raise Unavailable(f"no cuDNN backward graph builds: {self.errors}")
        merged = x.second is None
        d_first = torch.empty(x.R, 2 * I if merged else I, dtype=torch.bfloat16, device="cuda")
        d_up = None if merged else torch.empty(x.R, I, dtype=torch.bfloat16, device="cuda")
        d_gate_view, d_up_view = (d_first[:, :I], d_first[:, I:]) if merged else (d_first, d_up)
        d_weight = torch.empty(x.R, dtype=torch.float32, device="cuda") if self.weighted else None
        outputs = {"d_gate": d_gate_view, "d_up": d_up_view, "d_weight": d_weight}
        for built in self.backward_graphs:
            self._execute(built, outputs)
        return d_first, d_up, d_weight


def cudnn_arm(x: types.SimpleNamespace) -> dict[str, Any]:
    graphs = CudnnGraphs(x)
    return {
        "forward": graphs.forward,
        "backward": graphs.backward,
        "autograd": None,
        "writes_weight_grad": x.weight is not None,
        "version": graphs.version,
        "notes": {"backward": graphs.backward_kind, "errors": graphs.errors},
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
        view = torch.int16 if want.dtype == torch.bfloat16 else torch.int32
        same = got.to(want.dtype).contiguous().view(view) == want.contiguous().view(view)
        out[name] = {
            "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
            "bits_differ": float((~same).float().mean()) if same.numel() else 0.0,
        }
    return out


# The first calls a stage needs: a stage times only when each of them ran once.
NEEDS = {"fwd": ("fwd",), "bwd": ("bwd",), "fwd+bwd": ("fwd", "bwd"), "ckpt": ("fwd", "bwd")}


def bench_row(rows: int, layout: str, args: argparse.Namespace) -> dict[str, Any]:
    x = make_inputs(rows, layout, not args.unweighted, args.limit, args.seed)
    result: dict[str, Any] = {
        "R": rows,
        "I": I,
        "layout": layout,
        "weighted": x.weight is not None,
        "L": x.L,
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
            try:
                first[stage] = call(x)
            except Unavailable as exc:
                missing[stage] = str(exc)
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, torch.OutOfMemoryError):
                    raise
                missing[stage] = f"first call: {type(exc).__name__}: {str(exc)[:300]}"
        outputs = _outputs(x, first.get("fwd"), first.get("bwd"))
        if name == "cake":
            reference = outputs
        elif reference is not None:
            row["vs_cake"] = _compare(outputs, reference)
        del first
        row["saved_bytes"] = _saved_bytes(arm, x) if not missing else None
        row["workspace_bytes"] = 0 if name == "cake" else None
        row["stages"] = {}
        for stage, call in _stage_calls(arm, x).items():
            reasons = [missing[part] for part in NEEDS[stage] if part in missing]
            if reasons:
                row["stages"][stage] = {"unavailable": "; ".join(reasons)}
                continue
            try:
                times = time_stage(call, tuple(args.timers), args)
            except Unavailable as exc:
                row["stages"][stage] = {"unavailable": str(exc)}
                continue
            moved = dram_bytes(x, stage, writes_weight_grad=arm["writes_weight_grad"])
            entry: dict[str, Any] = {"ms": times, "dram_bytes": moved}
            if isinstance(times.get("cupti"), float):
                entry["cupti_gbs"] = moved / times["cupti"] / 1e6
            row["stages"][stage] = entry
        if "ckpt" not in row["stages"]:
            row["stages"]["ckpt"] = {"unavailable": "no autograd form"}
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
            if "unavailable" in entry:
                lines.append(f"| {name} | {stage} | " + " | ".join("unavailable" for _ in timers) + " | - | - |")
                notes.append(f"- {name} {stage}: unavailable: {entry['unavailable']}")
                continue
            gbs = entry.get("cupti_gbs")
            gbs_text = f"{gbs:.0f}" if gbs is not None else "-"
            cells = " | ".join(_cell(entry, timer) for timer in timers)
            lines.append(f"| {name} | {stage} | {cells} | {gbs_text} | {saved_text} |")
            for timer, value in entry["ms"].items():
                if isinstance(value, str):
                    notes.append(f"- {name} {stage} {timer}: {value}")
        if "vs_cake" in row:
            notes.append(f"- {name} outputs vs cake: {json.dumps(row['vs_cake'])}")
        for key in ("backward", "errors"):
            if row.get(key):
                notes.append(f"- {name} {key}: {row[key]}")
        notes.append(f"- {name} version: {row.get('version')}")
    return lines + [""] + notes + [""]


# ----------------------------------------------------------------------------------------------------------------
# Identity and main


def _version(dist: str) -> Optional[str]:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def identity() -> dict[str, Any]:
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    dists = (
        "cake-pplx-clamped-swiglu",
        "apache-tvm-ffi",
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
    }
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
    parser.add_argument("--loop", type=int, default=20, help="calls per eager / graph sample")
    parser.add_argument("--warmup", type=int, default=5, help="eager / graph samples before timing")
    parser.add_argument("--rounds", type=int, default=15, help="eager / graph samples timed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-bw", action="store_true", help="also time a 2 GiB device-to-device copy")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    if "cake" not in args.arms:
        args.arms = ["cake", *args.arms]
    ident = identity()
    print(
        f"# {ident['gpu']} (sm_{ident['cc'][0]}{ident['cc'][1]}, {ident['sms']} SMs), torch {ident['torch']} "
        f"(CUDA {ident['torch_cuda']}), driver/power/max clock {ident.get('driver_power_maxclock')}"
    )
    print(f"# versions: {json.dumps(ident['versions'])}")
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
