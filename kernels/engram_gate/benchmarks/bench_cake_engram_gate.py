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

Benchmark of the Cake Engram four-stream context gate (``cake_pplx.cake_engram_gate_{det,nondet}_train``, issue
#82) against the public baselines, with the issue's accuracy label per arm.

Arms (each called as ``fn(x, kv, q_weight, k_weight, token_mask)`` through autograd on the packed layout: ``x``
``[T, 4, 5120]`` BF16, ``kv`` ``[T, 25600]`` BF16 whose first 20480 columns are K and last 5120 columns are V,
``q_weight`` / ``k_weight`` ``[4, 5120]`` BF16 or FP32 leaves, ``token_mask`` ``[T]`` bool or ``None``):

* ``cudnn_fe``: cuDNN frontend 1.32.2 saved-state Engram gate (``cudnn.EngramGateSavedForward`` /
  ``EngramGateSavedBackward``, backend ``frost``) in an autograd wrapper: ``w = q_weight.float() * k_weight.float()``
  outside the wrapper (autograd applies the product rule), the backward chunked by 8192 tokens with the FP32 weight
  gradient of every later chunk added to the first chunk's in a fixed order; ``token_mask=None`` passes an all-True
  mask built inside the call.  The library fixes ``clamp = 1e-6`` and requires ``T % 64 == 0`` in the backward.
* ``torch_compile``: ``torch.compile(fullgraph=True, dynamic=False, options={"emulate_precision_casts": True})`` of
  the issue formula in FP32 (one BF16 rounding of ``out``).
* ``torch_compile_tuned``: the same with ``max_autotune=True`` and ``triton.cudagraphs=False``.
* ``eager``: the issue formula in eager PyTorch FP32.
* ``torch_compile_fp64dot`` (optional diagnostic): ``torch_compile`` with ``P`` summed in FP64.
* ``cake_nondet`` / ``cake_det``: ``cake_pplx.cake_engram_gate_<mode>_train.cake_engram_gate_<mode>_kv`` with the
  entry's default sum of the dot product (``dot_sum="fp32"``: one FP32 word); ``cake_nondet_compensated`` /
  ``cake_det_compensated``: the same entry with the opt-in ``dot_sum="compensated"`` (a compensated FP32 dot
  product; the backward is the same).  ``cake_nondet_fp32`` / ``cake_det_fp32`` (not run by default) name
  ``dot_sum="fp32"`` explicitly, for comparisons with a package whose default differs.

Timing boundary (identical for all arms): ``fwd`` = one training forward ``fn(x, kv, qw, kw, mask)`` (weight product,
output allocation, autograd graph recorded and dropped); ``bwd`` = ``torch.autograd.backward`` of a recorded forward
(``retain_graph=True``; the leaves' gradients reset to ``None`` first, so gradient allocation, the product rule and
any chunk sums are inside); ``step`` = forward + backward; ``ckpt`` = ``torch.utils.checkpoint`` (non-reentrant)
around the forward, then its backward.  Each phase runs eagerly (``eager`` mode: CUPTI span from the first kernel
start to the last kernel end of the iteration, host launch gaps included) and as one CUDA graph replay (``graph``
mode).  Cold L2 (a 2 x L2 buffer zeroed outside the span before every iteration), ``--dry`` untimed iterations,
``--repeat`` timed iterations per group, ``--groups`` groups with the arm order rotated per group and reversed every
other group; the latency is the median of the group medians, the spread the min / max of all samples and the
p10 / p90.

Accuracy label (the issue's protocol, CPU FP64 oracle ``issue82_reference.py``): the issue script's own 25 random
draws (``torch.manual_seed(7)``, the RNG of its sections 2-3 consumed first, so the draws are the issue's) with stream
scales 0.25/1/2/4 on X and 4/2/1/0.25 on K and ``mask[::4] = False``, and its 25 near-cancel draws (``|dot|`` in
[2e-6, 2e-5], no mask), ``T = 16``.  Relative L2 error of ``out``, ``dX``, ``dK``, ``dV``, ``dq``, ``dk`` against
FP64; median / max over the draws (random: whole tensors; near-cancel: per stream, 100 streams).  Bound = 2 x the
median / max of the FP32 evaluation of the reference (CPU, autograd, the issue's ``yb``) on the random draws, per
output.  ``cudnn_fe`` runs each T = 16 draw padded to 64 tokens with 48 masked all-zero tokens (its backward needs
``T % 64 == 0``; a masked zero token adds exact zeros, so the 16 real tokens see the same arithmetic).

Subcommands: ``probe`` (versions and the cuDNN FE API, no GPU work), ``run`` (one shard: ``--shard i --nshards n``;
a deterministic longest-first split of the accuracy and timing units; run one shard per GPU), ``report`` (merge shard
JSON into markdown, with the speedup of each Cake arm over the best FP32 baseline of the row).

Example (one GPU)::

    python bench_cake_engram_gate.py run --json gb300.json
    python bench_cake_engram_gate.py report --json gb300.json --md gb300.md --title "GB300"
"""

from __future__ import annotations

import argparse
import bisect
import collections
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
import types
import warnings
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

S, H = 4, 5120
KCOLS = S * H
EPS, CLAMP = 1e-20, 1e-6
FE_BWD_CHUNK = 8192
FE_FWD_MAX = 64 * 65535
FE_TOKEN_MULTIPLE = 64
SCALES = (0.25, 1.0, 2.0, 4.0)
# The Cake arms: the entry's default sum of the dot product, then the opt-in compensated forward sum
# (``dot_sum="compensated"``); the arms that name ``dot_sum="fp32"`` (the default) explicitly run on request.
CAKE_ARMS = ("cake_nondet", "cake_det")
CAKE_COMPENSATED_ARMS = ("cake_nondet_compensated", "cake_det_compensated")
CAKE_FP32_ARMS = ("cake_nondet_fp32", "cake_det_fp32")
CAKE_SUM_ARMS = CAKE_COMPENSATED_ARMS + CAKE_FP32_ARMS
DEFAULT_ARMS = CAKE_ARMS + CAKE_COMPENSATED_ARMS + ("cudnn_fe", "torch_compile", "torch_compile_tuned", "eager")
ALL_ARMS = DEFAULT_ARMS + CAKE_FP32_ARMS + ("cudnn_fe_dev", "torch_compile_fp64dot")
DETERMINISTIC_CLAIM = {
    "cudnn_fe": True,
    "cudnn_fe_dev": True,
    "torch_compile": True,
    "torch_compile_tuned": True,
    "eager": True,
    "torch_compile_fp64dot": True,
    "cake_nondet": False,
    "cake_det": True,
    "cake_nondet_compensated": False,
    "cake_det_compensated": True,
    "cake_nondet_fp32": False,
    "cake_det_fp32": True,
}
# Arms that evaluate the issue formula in FP32 (the "best baseline" column picks among these).
FP32_BASELINES = ("cudnn_fe", "cudnn_fe_dev", "torch_compile", "torch_compile_tuned", "eager")
NAMES = ("out", "dX", "dK", "dV", "dq", "dk")
NEAR_NAMES = ("dX", "dK", "dq", "dk")
# The FP32-evaluation numbers the issue prints for its random draws (med, max), for the harness self-check.
ISSUE_FP32_RANDOM = {
    "out": (1.45e-3, 1.46e-3),
    "dX": (1.46e-3, 1.62e-3),
    "dK": (1.66e-3, 1.70e-3),
    "dV": (1.66e-3, 1.67e-3),
    "dq": (1.75e-6, 6.57e-5),
    "dk": (1.75e-6, 6.57e-5),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


class Unsupported(Exception):
    """The arm cannot run the row / phase; the message is the concrete reason."""


# ---------------------------------------------------------------------------------------------------------------------
# GPU timer (kernels/dense_gemm/benchmarks/bench_cake_dense_gemm_det_train.py bench_gpu_time, with an explicit
# capture stream so a backward recorded on that stream is captured on it)
# ---------------------------------------------------------------------------------------------------------------------


def bench_gpu_time(fn, dry_run_iters=3, repeat_iters=20, enable_cupti=True, cold_l2_cache=True, use_cuda_graph=False,
                   capture_stream=None):
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
                warnings.warn(f"CUPTI is unavailable ({type(exc).__name__}: {exc}). Falling back to CUDA events.",
                              stacklevel=2)
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
# Arms
# ---------------------------------------------------------------------------------------------------------------------


def engram_gate_formula(x, kv, q_weight, k_weight, token_mask, eps: float = EPS, clamp: float = CLAMP):
    """The issue formula in FP32 on the packed layout (K and V are views of ``kv``); one BF16 rounding of ``out``."""
    h = x.float()
    k = kv[:, :KCOLS].unflatten(-1, (S, H)).float()
    v = kv[:, KCOLS:].float()
    w = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h * w * k).sum(-1) * rstd * H ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask[:, None], 0.0)
    return (h + gate[..., None] * v[:, None, :]).to(x.dtype)


def engram_gate_formula_fp64dot(x, kv, q_weight, k_weight, token_mask, eps: float = EPS, clamp: float = CLAMP):
    """``engram_gate_formula`` with ``P = sum(h * w * k)`` accumulated in FP64 (diagnostic arm)."""
    h = x.float()
    k = kv[:, :KCOLS].unflatten(-1, (S, H)).float()
    v = kv[:, KCOLS:].float()
    w = q_weight.float() * k_weight.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
    dot = (h.double() * w.double() * k.double()).sum(-1).float() * rstd * H ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask[:, None], 0.0)
    return (h + gate[..., None] * v[:, None, :]).to(x.dtype)


def _fresh_function(fn, name):
    """A copy of ``fn`` with its own code object (dynamo keeps its compile cache per code object)."""
    code = fn.__code__.replace(co_name=name)
    return types.FunctionType(code, fn.__globals__, name, fn.__defaults__, fn.__closure__)


def _dist(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class Arm:
    name = ""
    token_multiple = 1  # the accuracy draws are padded (masked zero tokens) to this multiple

    def library(self) -> str:
        return f"torch {torch.__version__}"

    def layer(self, mask):
        raise NotImplementedError

    def unsupported(self, T: int, phase: str):
        return None


class EagerArm(Arm):
    name = "eager"

    def layer(self, mask):
        return lambda x, kv, qw, kw: engram_gate_formula(x, kv, qw, kw, mask)


class CompileArm(Arm):
    def __init__(self, name, fn, options):
        self.name = name
        self.options = options
        self.compiled = torch.compile(_fresh_function(fn, f"engram_{name}"), fullgraph=True, dynamic=False,
                                      options=options)

    def library(self) -> str:
        import triton
        return f"torch {torch.__version__} (inductor, triton {triton.__version__}), options {self.options}"

    def layer(self, mask):
        c = self.compiled
        return lambda x, kv, qw, kw: c(x, kv, qw, kw, mask)


class _FeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kv, w, mask, arm):
        out = torch.empty_like(x)
        saved = torch.empty((x.shape[0], S, 4), device=x.device, dtype=torch.float32)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        arm.fwd_plan(x, kv, w, mask).execute(x, kv, w, mask, out, saved, current_stream=stream)
        ctx.save_for_backward(x, kv, w, saved)
        ctx.arm = arm
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g):
        x, kv, w, saved = ctx.saved_tensors
        g = g.contiguous()
        T = x.shape[0]
        c = min(T, FE_BWD_CHUNK)
        if T % FE_TOKEN_MULTIPLE or T % c:
            raise RuntimeError(f"cuDNN FE Engram backward: T={T} is not a multiple of 64 split into <= 8192 chunks")
        dx, dkv, dw = torch.empty_like(x), torch.empty_like(kv), torch.empty_like(w)
        part = torch.empty_like(w) if T > c else None
        stream = torch.cuda.current_stream(x.device).cuda_stream
        for a in range(0, T, c):
            sl = slice(a, a + c)
            plan, ws = ctx.arm.bwd_plan(x[sl], kv[sl], w, saved[sl], g[sl])
            target = dw if a == 0 else part
            plan.execute(x[sl], kv[sl], w, saved[sl], g[sl], dx[sl], dkv[sl], target, ws, current_stream=stream)
            if a:
                dw += part  # FP32, chunks in token order
        return dx, dkv, dw, None, None


class FeArm(Arm):
    token_multiple = FE_TOKEN_MULTIPLE

    def __init__(self, name="cudnn_fe"):
        self.name = name
        try:
            self.cudnn = importlib.import_module("cudnn")
        except Exception as exc:  # noqa: BLE001
            raise Unsupported(f"cudnn (nvidia-cudnn-frontend) not importable: {type(exc).__name__}: {exc}") from exc
        for attr in ("EngramGateSavedForward", "EngramGateSavedBackward"):
            if not hasattr(self.cudnn, attr):
                raise Unsupported(f"cudnn {self.cudnn.__version__} has no {attr}")
        self.fwd_plans = {}
        self.bwd_plans = {}
        if torch.cuda.is_available():
            # The library checks the device capability when a plan is built: probe once with a 64-token plan.
            dev = torch.device("cuda", torch.cuda.current_device())
            x = torch.zeros(64, S, H, dtype=torch.bfloat16, device=dev)
            kv = torch.zeros(64, KCOLS + H, dtype=torch.bfloat16, device=dev)
            w = torch.zeros(S, H, dtype=torch.float32, device=dev)
            m = torch.ones(64, dtype=torch.bool, device=dev)
            try:
                plan = self.cudnn.EngramGateSavedForward(x, kv, w, m, eps=EPS, backend="frost")
                if hasattr(plan, "check_support"):
                    plan.check_support()
            except NotImplementedError as exc:
                cc = torch.cuda.get_device_capability(dev)
                raise Unsupported(f"nvidia-cudnn-frontend {_dist('nvidia-cudnn-frontend')} rejects cc {cc}: "
                                  f"NotImplementedError: {exc}") from exc

    def library(self) -> str:
        return (f"nvidia-cudnn-frontend {_dist('nvidia-cudnn-frontend')} (cudnn.__version__ {self.cudnn.__version__}), "
                f"nvidia-cutlass-dsl {_dist('nvidia-cutlass-dsl')}, triton {_dist('triton')}, backend frost")

    def fwd_plan(self, x, kv, w, mask):
        key = (x.device.index, x.shape[0])
        plan = self.fwd_plans.get(key)
        if plan is None:
            plan = self.cudnn.EngramGateSavedForward(x, kv, w, mask, eps=EPS, backend="frost")
            if hasattr(plan, "compile"):
                plan.compile()
            self.fwd_plans[key] = plan
        return plan

    def bwd_plan(self, x, kv, w, saved, g):
        key = (x.device.index, x.shape[0])
        entry = self.bwd_plans.get(key)
        if entry is None:
            plan = self.cudnn.EngramGateSavedBackward(x, kv, w, saved, g, backend="frost")
            if hasattr(plan, "compile"):
                plan.compile()
            entry = (plan, plan.allocate_workspace())
            self.bwd_plans[key] = entry
        return entry

    def unsupported(self, T, phase):
        if T < 1 or T > FE_FWD_MAX:
            return f"cuDNN FE Engram gate takes 1 <= T <= {FE_FWD_MAX} tokens (T={T})"
        if phase != "fwd":
            if T % FE_TOKEN_MULTIPLE:
                return f"cuDNN FE Engram backward requires T % 64 == 0 (T={T})"
            if T > FE_BWD_CHUNK and T % FE_BWD_CHUNK:
                return f"cuDNN FE Engram backward takes T <= 8192 per call; T={T} is not a multiple of 8192"
        return None

    def layer(self, mask):
        arm = self

        def fn(x, kv, qw, kw):
            m = mask if mask is not None else torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            w = qw.float() * kw.float()
            return _FeFunction.apply(x, kv, w, m, arm)

        return fn


class CakeArm(Arm):
    def __init__(self, mode, dot_sum=None):
        self.name = f"cake_{mode}" + (f"_{dot_sum}" if dot_sum else "")
        module = f"cake_pplx.cake_engram_gate_{mode}_train"
        try:
            self.mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            raise Unsupported(f"{module} not importable: {type(exc).__name__}: {exc}") from exc
        entry = f"cake_engram_gate_{mode}_kv"
        self.entry = getattr(self.mod, entry, None)
        if self.entry is None:
            raise Unsupported(f"{module} has no {entry}")
        if torch.cuda.is_available() and not getattr(self.mod, f"is_cake_engram_gate_{mode}_train_supported")():
            raise Unsupported(f"{module} does not serve cc {torch.cuda.get_device_capability()}")
        self.mode = mode
        # The sum of the dot product: the entry's default, or the ``dot_sum`` the arm names.
        default = inspect.signature(inspect.unwrap(self.entry)).parameters["dot_sum"].default
        self.dot_sum = dot_sum or default

    def library(self) -> str:
        return (f"cake-pplx-engram-gate {_dist('cake-pplx-engram-gate')} ({self.mod.__name__}, "
                f"dot_sum={self.dot_sum!r})")

    def layer(self, mask):
        entry, dot_sum = self.entry, self.dot_sum

        def fn(x, kv, qw, kw):
            return entry(x, kv, qw, kw, mask, eps=EPS, clamp_value=CLAMP, dot_sum=dot_sum)

        return fn


def make_arm(name):
    opts = {"emulate_precision_casts": True}
    if name == "eager":
        return EagerArm()
    if name == "torch_compile":
        return CompileArm(name, engram_gate_formula, opts)
    if name == "torch_compile_tuned":
        return CompileArm(name, engram_gate_formula, {**opts, "max_autotune": True, "triton.cudagraphs": False})
    if name == "torch_compile_fp64dot":
        return CompileArm(name, engram_gate_formula_fp64dot, opts)
    if name in ("cudnn_fe", "cudnn_fe_dev"):
        return FeArm(name)
    if name in CAKE_ARMS:
        return CakeArm(name.split("_", 1)[1])
    if name in CAKE_SUM_ARMS:
        _, mode, dot_sum = name.split("_")
        return CakeArm(mode, dot_sum)
    raise ValueError(f"unknown arm {name}")


def _issue_reference():
    """The FP64 reference of issue #82 (CPU): ``engram_gate_ref``, ``engram_gate_vjp_fp64`` and
    ``make_near_cancel_keys``, as the issue states them."""

    def signed_sqrt_gate(dot, clamp_value=CLAMP):
        return torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))

    def engram_gate_ref(x, key, value, q_weight, k_weight, token_mask=None, eps=EPS, clamp_value=CLAMP):
        orig_dtype = x.dtype
        acc = torch.float64 if orig_dtype == torch.float64 else torch.float32
        h, k, v = x.to(acc), key.to(acc), value.to(acc)
        w = q_weight.to(acc) * k_weight.to(acc)
        Hh = h.shape[-1]
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(k.square().mean(-1) + eps)
        dot = (h * w * k).sum(-1) * rstd * (Hh**-0.5)
        gate = signed_sqrt_gate(dot, clamp_value)
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        return (h + gate.unsqueeze(-1) * v.unsqueeze(-2)).to(orig_dtype)

    def engram_gate_vjp_fp64(x, key, value, q_weight, k_weight, dout, token_mask=None, eps=EPS, clamp_value=CLAMP):
        h, k, v = x.double(), key.double(), value.double()
        qw, kw, g = q_weight.double(), k_weight.double(), dout.double()
        T, _, Hh = h.shape
        if T == 0:
            zw = torch.zeros_like(qw)
            return h.clone(), k.clone(), v.clone(), zw, zw.clone()
        w = qw * kw
        rh = torch.rsqrt(h.square().mean(-1) + eps)
        rk = torch.rsqrt(k.square().mean(-1) + eps)
        prod_sum = (h * w * k).sum(-1)
        inv_sqrt_h = Hh**-0.5
        dot = prod_sum * rh * rk * inv_sqrt_h
        gate = signed_sqrt_gate(dot, clamp_value)
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        dv = (gate.unsqueeze(-1) * g).sum(dim=1)
        dgate = (g * v.unsqueeze(1)).sum(dim=-1)
        if token_mask is not None:
            dgate = dgate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        du_ddot = torch.where(dot.abs() > clamp_value, 0.5 / dot.abs().clamp_min(clamp_value).sqrt(), 0.0)
        c = (dgate * gate * (1.0 - gate) * du_ddot * rh * rk * inv_sqrt_h).unsqueeze(-1)
        ps = prod_sum.unsqueeze(-1)
        dx = g + c * (w * k - (ps * rh.square().unsqueeze(-1) / Hh) * h)
        dk = c * (w * h - (ps * rk.square().unsqueeze(-1) / Hh) * k)
        dw = (c * h * k).sum(dim=0)
        return dx, dk, dv, dw * kw, dw * qw

    def make_near_cancel_keys(x, k_init, w):
        Hh = x.shape[-1]
        h, hw = x.double(), x.double() * w.double()
        mag = 2.2e-6 * (8.5 ** torch.rand(k_init.shape[:2], dtype=torch.float64))
        target = torch.where(torch.rand_like(mag) < 0.5, -mag, mag)
        rh = torch.rsqrt(h.square().mean(-1) + EPS)

        def resid(kd):
            return target * (Hh**0.5) / (rh * torch.rsqrt(kd.square().mean(-1) + EPS)) - (hw * kd).sum(-1)

        kd = k_init.double()
        kd = (kd + (resid(kd) / hw.square().sum(-1)).unsqueeze(-1) * hw).bfloat16().double()
        for _ in range(3):
            left = resid(kd).unsqueeze(-1)
            step = left / hw
            upd = (kd + step).bfloat16().double()
            err = torch.where(step.abs() < 2.0, (left - hw * (upd - kd)).abs(), torch.inf)
            idx = err.argmin(-1, keepdim=True)
            kd = kd.scatter(-1, idx, upd.gather(-1, idx))
        dot = ((hw * kd).sum(-1) * rh * torch.rsqrt(kd.square().mean(-1) + EPS) * (Hh**-0.5)).abs()
        assert bool((dot >= 2e-6).all() and (dot <= 2e-5).all())
        return kd.bfloat16()

    return types.SimpleNamespace(
        engram_gate_ref=engram_gate_ref, engram_gate_vjp_fp64=engram_gate_vjp_fp64,
        make_near_cancel_keys=make_near_cancel_keys,
    )


def _tail(exc: BaseException, n: int = 400) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= n else text[: n // 2] + " ... " + text[-n // 2:]


# ---------------------------------------------------------------------------------------------------------------------
# Accuracy (issue protocol)
# ---------------------------------------------------------------------------------------------------------------------


def issue_draws(ref, n=25, T=16):
    """The issue script's random and near-cancel draws, in its RNG order (seed 7; sections 2-3 consumed first)."""
    f64 = torch.float64
    torch.manual_seed(7)
    torch.randn(4, 4, 16, dtype=f64), torch.randn(4, 4, 16, dtype=f64), torch.randn(4, 16, dtype=f64)
    torch.randn(3, 16, dtype=f64), torch.randn(3, 16, dtype=f64)
    torch.randn(5, 4, 16, dtype=f64), torch.randn(5, 80, dtype=f64), torch.randn(4, 16, dtype=f64)
    torch.randn(5, 4, 16, dtype=f64)
    scales = torch.tensor(SCALES).view(1, S, 1)
    out = []
    for _ in range(n):
        X = (torch.randn(T, S, H) * scales).bfloat16()
        K = (torch.randn(T, S, H) * scales.flip(1)).bfloat16()
        V = torch.randn(T, H).bfloat16()
        G = torch.randn(T, S, H).bfloat16()
        qw = (1.0 + 0.1 * torch.randn(S, H)).bfloat16().float()
        kw = (1.0 + 0.1 * torch.randn(S, H)).bfloat16().float()
        mask = torch.ones(T, dtype=torch.bool)
        mask[::4] = False
        Kn = ref.make_near_cancel_keys(X, torch.randn(T, S, H) * scales.flip(1), qw * kw)
        out.append(dict(X=X, K=K, V=V, G=G, qw=qw, kw=kw, mask=mask, Kn=Kn))
    return out


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def errors(res, ref64, per_stream):
    whole = {n: rel(res[n], ref64[n]) for n in NAMES}
    streams = {}
    if per_stream:
        for n in ("out", "dX", "dK"):
            streams[n] = [rel(res[n][:, s], ref64[n][:, s]) for s in range(S)]
        for n in ("dq", "dk"):
            streams[n] = [rel(res[n][s], ref64[n][s]) for s in range(S)]
    return whole, streams


def oracle(ref, X, K, V, G, qw, kw, mask):
    y64 = ref.engram_gate_ref(X.double(), K.double(), V.double(), qw.double(), kw.double(), mask)
    dx, dk, dv, dq, dkw = ref.engram_gate_vjp_fp64(X, K, V, qw, kw, G, mask)
    return dict(out=y64, dX=dx, dK=dk, dV=dv, dq=dq, dk=dkw)


def fp32_eval(ref, X, K, V, G, qw, kw, mask, wdtype):
    leaves = [t.clone().requires_grad_(True) for t in (X, K, V, qw.to(wdtype), kw.to(wdtype))]
    y = ref.engram_gate_ref(*leaves, mask)
    y.backward(G)
    return dict(out=y.detach(), dX=leaves[0].grad, dK=leaves[1].grad, dV=leaves[2].grad, dq=leaves[3].grad,
                dk=leaves[4].grad)


def run_arm_draw(arm, X, K, V, G, qw, kw, mask, wdtype, device):
    T = X.shape[0]
    mult = arm.token_multiple
    Tp = -(-T // mult) * mult
    kv = torch.cat([K.flatten(1), V], dim=1)
    m = mask
    if Tp > T:
        def pad(t):
            return torch.cat([t, t.new_zeros((Tp - T,) + tuple(t.shape[1:]))])
        X, kv, G = pad(X), pad(kv), pad(G)
        m = torch.cat([mask if mask is not None else torch.ones(T, dtype=torch.bool), torch.zeros(Tp - T, dtype=torch.bool)])
    x = X.to(device).requires_grad_(True)
    kvd = kv.to(device).contiguous().requires_grad_(True)
    qwd = qw.to(device=device, dtype=wdtype).requires_grad_(True)
    kwd = kw.to(device=device, dtype=wdtype).requires_grad_(True)
    md = m.to(device) if m is not None else None
    out = arm.layer(md)(x, kvd, qwd, kwd)
    out.backward(G.to(device))
    torch.cuda.synchronize(device)
    dkv = kvd.grad[:T]
    res = dict(out=out.detach()[:T], dX=x.grad[:T], dK=dkv[:, :KCOLS].unflatten(-1, (S, H)), dV=dkv[:, KCOLS:],
               dq=qwd.grad, dk=kwd.grad)
    return {n: t.detach().cpu() for n, t in res.items()}


def med(xs):
    return sorted(xs)[len(xs) // 2]


def summarize(lists):
    return {n: {"med": med(v), "max": max(v), "n": len(v)} for n, v in lists.items() if v}


def accuracy_unit(arm_names, wdtype_name, device, ref, ndraws):
    """Accuracy labels of every arm for one weight dtype; returns a list of records."""
    wdtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[wdtype_name]
    t0 = time.time()
    draws = issue_draws(ref, ndraws)
    log(f"acc[{wdtype_name}] {len(draws)} draws generated in {time.time() - t0:.1f}s")
    # FP64 oracle and FP32 evaluation (the bound) on the CPU.
    rand_ref, near_ref = [], []
    fp32_rand = {n: [] for n in NAMES}
    fp32_near = {n: [] for n in NAMES}
    for d in draws:
        r64 = oracle(ref, d["X"], d["K"], d["V"], d["G"], d["qw"], d["kw"], d["mask"])
        n64 = oracle(ref, d["X"], d["Kn"], d["V"], d["G"], d["qw"], d["kw"], None)
        rand_ref.append(r64)
        near_ref.append(n64)
        f = fp32_eval(ref, d["X"], d["K"], d["V"], d["G"], d["qw"], d["kw"], d["mask"], wdtype)
        w, _ = errors(f, r64, False)
        for n in NAMES:
            fp32_rand[n].append(w[n])
        f = fp32_eval(ref, d["X"], d["Kn"], d["V"], d["G"], d["qw"], d["kw"], None, wdtype)
        _, st = errors(f, n64, True)
        for n, vals in st.items():
            fp32_near[n].extend(vals)
    fr = summarize(fp32_rand)
    bound = {n: {"med": 2 * fr[n]["med"], "max": 2 * fr[n]["max"]} for n in NAMES}
    records = [{
        "kind": "accuracy", "arm": "fp32_eval_cpu", "weights": wdtype_name, "library": f"torch {torch.__version__} CPU",
        "random": fr, "near_per_stream": summarize(fp32_near), "bound": bound,
        "issue_fp32_random": ISSUE_FP32_RANDOM if wdtype_name == "fp32" else None,
    }]
    log(f"acc[{wdtype_name}] oracle + FP32 eval done in {time.time() - t0:.1f}s; FP32 random "
        + " ".join(f"{n}={fr[n]['med']:.2e}/{fr[n]['max']:.2e}" for n in NAMES))
    for name in arm_names:
        rec = {"kind": "accuracy", "arm": name, "weights": wdtype_name, "bound": bound}
        t1 = time.time()
        try:
            arm = make_arm(name)
            rec["library"] = arm.library()
            if arm.token_multiple > 1:
                rec["note"] = (f"T=16 draws padded to {arm.token_multiple} tokens with masked all-zero tokens "
                               f"(backward needs T % {arm.token_multiple} == 0)")
            rand = {n: [] for n in NAMES}
            near_w = {n: [] for n in NAMES}
            near_s = {n: [] for n in NAMES}
            mask_exact, finite, r2r = True, True, None
            for i, d in enumerate(draws):
                res = run_arm_draw(arm, d["X"], d["K"], d["V"], d["G"], d["qw"], d["kw"], d["mask"], wdtype, device)
                if i == 0:
                    again = run_arm_draw(arm, d["X"], d["K"], d["V"], d["G"], d["qw"], d["kw"], d["mask"], wdtype,
                                         device)
                    r2r = all(_bitwise_equal(res[n], again[n]) for n in NAMES)
                finite &= all(bool(torch.isfinite(res[n]).all()) for n in NAMES)
                m = ~d["mask"]
                mask_exact &= (_bitwise_equal(res["out"][m], d["X"][m]) and _bitwise_equal(res["dX"][m], d["G"][m])
                               and bool((res["dK"][m] == 0).all()) and bool((res["dV"][m] == 0).all()))
                w, _ = errors(res, rand_ref[i], False)
                for n in NAMES:
                    rand[n].append(w[n])
                resn = run_arm_draw(arm, d["X"], d["Kn"], d["V"], d["G"], d["qw"], d["kw"], None, wdtype, device)
                finite &= all(bool(torch.isfinite(resn[n]).all()) for n in NAMES)
                w, st = errors(resn, near_ref[i], True)
                for n in NAMES:
                    near_w[n].append(w[n])
                for n, vals in st.items():
                    near_s[n].extend(vals)
            rs, ns = summarize(rand), summarize(near_s)
            rec.update(random=rs, near_whole=summarize(near_w), near_per_stream=ns, mask_exact=mask_exact,
                       finite=finite, run_to_run_bitwise_t16=r2r)
            rec["random_pass"] = {n: rs[n]["med"] <= bound[n]["med"] and rs[n]["max"] <= bound[n]["max"]
                                  for n in NAMES}
            rec["near_pass"] = {n: ns[n]["med"] <= bound[n]["med"] and ns[n]["max"] <= bound[n]["max"]
                                for n in NEAR_NAMES}
            rec["passed"] = all(rec["random_pass"].values()) and all(rec["near_pass"].values()) and mask_exact \
                and finite
            log(f"acc[{wdtype_name}] {name}: passed={rec['passed']} random "
                + " ".join(f"{n}={rs[n]['med']:.2e}/{rs[n]['max']:.2e}" for n in NAMES)
                + " | near/stream " + " ".join(f"{n}={ns[n]['med']:.2e}/{ns[n]['max']:.2e}" for n in NEAR_NAMES)
                + f" | mask_exact={mask_exact} r2r={r2r} ({time.time() - t1:.0f}s)")
        except Unsupported as exc:
            rec["unsupported"] = str(exc)
            log(f"acc[{wdtype_name}] {name}: unsupported: {exc}")
        except Exception as exc:  # noqa: BLE001
            rec["error"] = _tail(exc, 1200)
            rec["traceback"] = traceback.format_exc()[-3000:]
            log(f"acc[{wdtype_name}] {name}: ERROR {_tail(exc)}")
        records.append(rec)
        torch.cuda.empty_cache()
    return records


def _bitwise_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    ints = {2: torch.int16, 4: torch.int32, 8: torch.int64, 1: torch.uint8}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(ints), b.contiguous().view(ints)))


# ---------------------------------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------------------------------


def make_inputs(T, wdtype, mask_every, device, seed=1234):
    g = torch.Generator(device=device).manual_seed(seed + T)
    sc = torch.tensor(SCALES, device=device).view(1, S, 1)
    x = (torch.randn(T, S, H, generator=g, device=device) * sc).bfloat16()
    kv = torch.empty(T, KCOLS + H, dtype=torch.bfloat16, device=device)
    kv[:, :KCOLS] = (torch.randn(T, S, H, generator=g, device=device) * sc.flip(1)).flatten(1)
    kv[:, KCOLS:] = torch.randn(T, H, generator=g, device=device)
    G = torch.randn(T, S, H, generator=g, device=device).bfloat16()
    qw = (1.0 + 0.1 * torch.randn(S, H, generator=g, device=device)).to(wdtype)
    kw = (1.0 + 0.1 * torch.randn(S, H, generator=g, device=device)).to(wdtype)
    mask = None
    if mask_every > 0:
        mask = torch.ones(T, dtype=torch.bool, device=device)
        mask[::mask_every] = False
    leaves = [t.requires_grad_(True) for t in (x, kv, qw, kw)]
    return leaves, G, mask


def _reset(leaves):
    for leaf in leaves:
        leaf.grad = None


def phase_fn(lay, leaves, G, phase, record_stream):
    """(fn, release) of a timed phase.  ``bwd`` records its forward on ``record_stream`` (the capture stream in
    graph mode, so the backward runs on it)."""
    state = {}
    if phase == "fwd":
        return (lambda: lay(*leaves)), state.clear
    if phase == "bwd":
        if record_stream is not None:
            record_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(record_stream):
                state["y"] = lay(*leaves)
            torch.cuda.current_stream().wait_stream(record_stream)
        else:
            state["y"] = lay(*leaves)
        torch.cuda.synchronize()

        def bwd():
            _reset(leaves)
            torch.autograd.backward([state["y"]], [G], retain_graph=True)
        return bwd, state.clear
    if phase == "step":
        def step():
            _reset(leaves)
            torch.autograd.backward([lay(*leaves)], [G])
        return step, state.clear
    if phase == "ckpt":
        def ckpt():
            _reset(leaves)
            y = checkpoint(lay, *leaves, use_reentrant=False, preserve_rng_state=False)
            torch.autograd.backward([y], [G])
        return ckpt, state.clear
    raise ValueError(phase)


def _grads_bytes(leaves):
    return sum(leaf.grad.numel() * leaf.grad.element_size() for leaf in leaves if leaf.grad is not None)


def timing_unit(arm_names, T, wname, mask_every, args, device, cap_stream):
    wdtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[wname]
    row = {"T": T, "weights": wname, "mask": "none" if mask_every <= 0 else f"every_{mask_every}_false"}
    log(f"row T={T} w={wname} mask={row['mask']}: inputs")
    leaves, G, mask = make_inputs(T, wdtype, mask_every, device)
    torch.cuda.synchronize()
    records = []
    lays, arm_objs, broken = {}, {}, {}
    for name in arm_names:
        base = {"kind": "timing", "arm": name, **row}
        try:
            arm = make_arm(name)
            arm_objs[name] = arm
            base["library"] = arm.library()
            lay = arm.layer(mask)
            bwd_ok = arm.unsupported(T, "bwd") is None
            t0 = time.time()
            # warm (compiles), memory, run-to-run bitwise (eager, default stream)
            _reset(leaves)
            torch.cuda.synchronize()
            m0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            y = lay(*leaves)
            torch.cuda.synchronize()
            m1 = torch.cuda.memory_allocated()
            fwd_peak = torch.cuda.max_memory_allocated() - m0
            mem = {"out_bytes": y.numel() * y.element_size(), "fwd_retained_bytes": m1 - m0 - y.numel() * y.element_size(),
                   "fwd_peak_extra_bytes": fwd_peak}
            if bwd_ok:
                torch.cuda.reset_peak_memory_stats()
                torch.autograd.backward([y], [G])
                torch.cuda.synchronize()
                mem["grads_bytes"] = _grads_bytes(leaves)
                mem["bwd_peak_extra_bytes"] = torch.cuda.max_memory_allocated() - m1
            del y
            base["warm_s"] = round(time.time() - t0, 1)
            base["memory"] = mem
            if bwd_ok and args.r2r:
                outs = []
                for _ in range(2):
                    _reset(leaves)
                    y = lay(*leaves)
                    torch.autograd.backward([y], [G])
                    outs.append([y.detach()] + [leaf.grad for leaf in leaves])
                    del y
                torch.cuda.synchronize()
                base["run_to_run_bitwise"] = all(_bitwise_equal(a, b) for a, b in zip(*outs, strict=True))
                del outs
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
        records.append(base)
    meta = {r["arm"]: r for r in records}
    out = []
    for mode in args.modes:
        for phase in args.phases:
            samples = {n: [] for n in arm_names}
            gmeds = {n: [] for n in arm_names}
            failed = dict(broken)
            for name in arm_names:
                if name in lays:
                    why = arm_objs[name].unsupported(T, phase)
                    if why:
                        failed[name] = ("unsupported", why)
            for gi in range(args.groups):
                order = [n for n in arm_names if n not in failed]
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
                        fn, release = phase_fn(lays[name], leaves, G, phase, cap_stream if mode == "graph" else None)
                        times = bench_gpu_time(fn, dry_run_iters=args.dry, repeat_iters=args.repeat,
                                               use_cuda_graph=(mode == "graph"), capture_stream=cap_stream)
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
                rec = {"kind": "timing", "arm": name, **row, "mode": mode, "phase": phase,
                       "library": meta[name].get("library")}
                if name in failed:
                    rec[failed[name][0]] = failed[name][1]
                elif samples[name]:
                    xs = sorted(samples[name])
                    q = lambda p, xs=xs: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
                    rec.update(median_us=1e3 * statistics.median(gmeds[name]), min_us=1e3 * xs[0], max_us=1e3 * xs[-1],
                               p10_us=1e3 * q(0.10), p90_us=1e3 * q(0.90), n=len(xs),
                               group_medians_us=[round(1e3 * v, 3) for v in gmeds[name]])
                    for key in ("memory", "run_to_run_bitwise", "warm_s"):
                        if key in meta[name]:
                            rec[key] = meta[name][key]
                out.append(rec)
                if "median_us" in rec:
                    log(f"  {mode}/{phase} {name}: {rec['median_us']:.1f} us (p10 {rec['p10_us']:.1f} p90 "
                        f"{rec['p90_us']:.1f}, n={rec['n']})")
    del leaves, G, mask, lays
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------------------------------


def _cmd(argv):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<{type(exc).__name__}: {exc}>"


def ptxas_version(path):
    if not path:
        return None
    line = [s for s in _cmd([path, "--version"]).splitlines() if "release" in s]
    return f"{path}: {line[-1].strip() if line else '?'}"


def _donated_buffer():
    """``torch._functorch.config.donated_buffer`` (the module loads on first import), or None without it."""
    try:
        return importlib.import_module("torch._functorch.config").donated_buffer
    except (ImportError, AttributeError):
        return None


def environment():
    env = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_step": os.environ.get("SLURM_STEP_ID"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "TRITON_PTXAS_PATH": os.environ.get("TRITON_PTXAS_PATH"),
        "TRITON_PTXAS_BLACKWELL_PATH": os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
        "timer": timer_kind(),
        "functorch_donated_buffer": _donated_buffer(),
    }
    for dist in ("triton", "nvidia-cudnn-frontend", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base",
                 "apache-tvm-ffi", "cupti-python", "nvidia-cuda-runtime", "nvidia-cuda-nvrtc", "nvidia-cuda-cupti",
                 "nvidia-cuda-nvcc", "cuda-toolkit", "nvidia-cudnn-cu13", "numpy", "cake-pplx-engram-gate"):
        env[f"dist:{dist}"] = _dist(dist)
    env["ptxas_env"] = ptxas_version(os.environ.get("TRITON_PTXAS_PATH"))
    try:
        import triton
        bundled = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
        env["ptxas_triton_bundled"] = ptxas_version(str(bundled)) if bundled.exists() else None
    except Exception as exc:  # noqa: BLE001
        env["ptxas_triton_bundled"] = f"<{exc}>"
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        env.update(gpu=p.name, capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                   l2_bytes=p.L2_cache_size, gpu_mem_gb=round(p.total_memory / 2**30, 1),
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
        env["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.max.sm,"
                                  "clocks.mem,power.limit,temperature.gpu", "--format=csv,noheader",
                                  f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}"])
        env["driver_cuda"] = _cmd(["bash", "-c", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"])
    return env


def probe():
    print(json.dumps(environment(), indent=1))
    try:
        import cudnn
        print("cudnn", cudnn.__version__, cudnn.__file__)
        for cls in ("EngramGateSavedForward", "EngramGateSavedBackward"):
            obj = getattr(cudnn, cls, None)
            print(f"== {cls}: {obj}")
            if obj is None:
                continue
            print("   init", inspect.signature(obj.__init__))
            for meth in ("compile", "execute", "allocate_workspace"):
                f = getattr(obj, meth, None)
                print(f"   {meth}: {inspect.signature(f) if f else None}")
        api = importlib.import_module("cudnn.engram.api")
        src = inspect.getsource(api).splitlines()
        for i, line in enumerate(src, 1):
            if any(s in line for s in ("CCS", "_capability", "current_stream", "def ", "tokens %", "tokens >",
                                       "workspace")):
                print(f"   api.py:{i}: {line.rstrip()}")
    except Exception as exc:  # noqa: BLE001
        print("cudnn probe failed:", _tail(exc, 2000))
        traceback.print_exc()


# ---------------------------------------------------------------------------------------------------------------------
# Sharding, run, report
# ---------------------------------------------------------------------------------------------------------------------


def units(args):
    out = [("acc", w) for w in args.acc_weights] if not args.no_accuracy else []
    if not args.accuracy_only:
        for T in args.T:
            for w in args.weights:
                for m in args.masks:
                    out.append(("row", T, w, m))
    return out


def unit_cost(u):
    if u[0] == "acc":
        return 60000
    return max(u[1], 2048) + 14000


def shard_units(all_units, shard, nshards):
    """Deterministic longest-processing-time split."""
    loads = [0] * nshards
    assigned = [[] for _ in range(nshards)]
    for u in sorted(all_units, key=lambda u: (-unit_cost(u), str(u))):
        k = min(range(nshards), key=lambda i: (loads[i], i))
        loads[k] += unit_cost(u)
        assigned[k].append(u)
    return assigned[shard]


def cmd_run(args):
    import torch._dynamo  # noqa: F401
    import torch._functorch.config
    # The timed `bwd` phase replays the backward of one recorded forward (retain_graph=True); AOTAutograd donated
    # buffers forbid that, so they are off for every compiled arm (they only let the backward reuse saved buffers).
    torch._functorch.config.donated_buffer = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for attr, value in (("cache_size_limit", 4096), ("recompile_limit", 4096), ("accumulated_cache_size_limit", 65536),
                        ("accumulated_recompile_limit", 65536)):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, value)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cap_stream = torch.cuda.Stream(device)
    ref = _issue_reference()
    result = {"env_start": environment(), "args": vars(args), "records": []}
    mine = shard_units(units(args), args.shard, args.nshards)
    log(f"shard {args.shard}/{args.nshards}: units {mine}")
    path = Path(args.json)

    def flush():
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(path)

    flush()
    for u in mine:
        t0 = time.time()
        try:
            if u[0] == "acc":
                result["records"].extend(accuracy_unit(args.arms, u[1], device, ref, args.draws))
            else:
                mask_every = 0 if u[3] == "none" else args.mask_every
                result["records"].extend(timing_unit(args.arms, u[1], u[2], mask_every, args, device, cap_stream))
        except Exception as exc:  # noqa: BLE001
            result["records"].append({"kind": "unit_error", "unit": list(u), "error": _tail(exc, 1500),
                                      "traceback": traceback.format_exc()[-4000:]})
            log(f"UNIT ERROR {u}: {_tail(exc)}")
        log(f"unit {u} done in {time.time() - t0:.0f}s")
        result["env_end"] = environment()
        flush()
    log("SHARD_DONE")


def _fmt_us(r):
    if r is None:
        return "-"
    if "median_us" in r:
        spread = (r["p90_us"] - r["p10_us"]) / 2 / r["median_us"] * 100 if r["median_us"] else 0.0
        return f"{r['median_us']:.1f} ±{spread:.1f}%"
    if "unsupported" in r:
        return "unsupported"
    if "error" in r:
        return "error"
    return "-"


def cmd_report(args):
    shards = [json.loads(Path(p).read_text()) for p in args.json]
    recs = [r for s in shards for r in s["records"]]
    envs = [s.get("env_end") or s.get("env_start") for s in shards]
    arms = []
    for r in recs:
        if r.get("arm") and r["arm"] not in arms and r["arm"] != "fp32_eval_cpu":
            arms.append(r["arm"])
    order = [a for a in ALL_ARMS if a in arms] + [a for a in arms if a not in ALL_ARMS]
    lines = [f"# Engram gate benchmark: {args.title}", ""]
    e = envs[0]
    lines += ["## Environment", "",
              f"- GPU: {e.get('gpu')} (cc {e.get('capability')}, {e.get('sm_count')} SMs, L2 {e.get('l2_bytes')} B, "
              f"{e.get('gpu_mem_gb')} GiB); host {', '.join(sorted({x.get('host') for x in envs}))}; "
              f"job {e.get('slurm_job')}.{e.get('slurm_step')}",
              "- nvidia-smi per shard: " + " | ".join(x.get("nvidia_smi", "") for x in envs),
              f"- driver {e.get('driver_cuda')}; torch {e.get('torch')} (CUDA {e.get('torch_cuda')}); triton "
              f"{e.get('dist:triton')}; nvidia-cudnn-frontend {e.get('dist:nvidia-cudnn-frontend')}; nvidia-cutlass-dsl "
              f"{e.get('dist:nvidia-cutlass-dsl')}; apache-tvm-ffi {e.get('dist:apache-tvm-ffi')}; cupti-python "
              f"{e.get('dist:cupti-python')}; nvidia-cuda-runtime {e.get('dist:nvidia-cuda-runtime')}; "
              f"nvidia-cuda-nvrtc {e.get('dist:nvidia-cuda-nvrtc')}; python {e.get('python')}",
              f"- ptxas: env {e.get('ptxas_env')}; triton bundled {e.get('ptxas_triton_bundled')}",
              f"- timer: {e.get('timer')}; TF32 override {e.get('NVIDIA_TF32_OVERRIDE')}; fp32 matmul precision "
              f"{e.get('float32_matmul_precision')}; torch._functorch.config.donated_buffer {e.get('functorch_donated_buffer')}", ""]
    a0 = shards[0]["args"]
    lines += ["Speedup = the best FP32 baseline's median over the Cake arm's median (above 1: Cake is faster).", ""]
    lines += [f"Timing: CUPTI span per iteration, cold L2 (2 x L2 zeroed outside the span), {a0['dry']} dry + "
              f"{a0['repeat']} timed iterations x {a0['groups']} groups (arm order rotated), latency = median of group "
              f"medians in us, ± = (p90 - p10) / 2 / median. Packed kv [T,25600] with K/V views; mask 'every_"
              f"{a0['mask_every']}_false' = token_mask[::{a0['mask_every']}] = False, 'none' = token_mask None. "
              "fwd = training forward incl. weight product and output allocation; bwd = autograd backward of a "
              "recorded forward incl. gradient allocation, product rule and chunk sums; step = fwd + bwd; ckpt = "
              "checkpoint(use_reentrant=False) forward + backward.", ""]
    tim = [r for r in recs if r.get("kind") == "timing" and "mode" in r]
    keyset = sorted({(r["T"], r["weights"], r["mask"]) for r in tim}, key=lambda k: (k[0], k[1], k[2]))
    for mode in ("graph", "eager"):
        for phase in ("fwd", "bwd", "step", "ckpt"):
            sub = [r for r in tim if r["mode"] == mode and r["phase"] == phase]
            if not sub:
                continue
            cakes = [a for a in order if a.startswith("cake_")]
            lines += [f"### {phase}, {'CUDA graph replay' if mode == 'graph' else 'eager launches'} (us)", "",
                      "| T | weights | mask | " + " | ".join(order) + " | best FP32 baseline | "
                      + " | ".join(f"{a} speedup" for a in cakes) + " |",
                      "| ---: | --- | --- | " + " | ".join("---:" for _ in order) + " | --- | "
                      + " | ".join("---:" for _ in cakes) + " |"]
            for key in keyset:
                cells = {r["arm"]: r for r in sub if (r["T"], r["weights"], r["mask"]) == key}
                if not cells:
                    continue
                timed = [(c["median_us"], a) for a, c in cells.items() if "median_us" in c and a in FP32_BASELINES]
                best = min(timed) if timed else None
                ratios = [
                    f"{best[0] / cells[a]['median_us']:.2f}x" if best and "median_us" in cells.get(a, {}) else "-"
                    for a in cakes
                ]
                lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(_fmt_us(cells.get(a)) for a in order)
                             + f" | {best[1] if best else '-'} | " + " | ".join(ratios) + " |")
            lines.append("")
    unsup = {}
    for r in tim:
        for k in ("unsupported", "error"):
            if k in r:
                unsup.setdefault((r["arm"], k, r[k]), []).append(f"T={r['T']} {r['weights']} {r['mask']} "
                                                                  f"{r['mode']}/{r['phase']}")
    if unsup:
        lines += ["### Unsupported / failed timing cells", "", "| arm | kind | reason | cells |", "| --- | --- | --- | --- |"]
        for (arm, k, why), cells in sorted(unsup.items()):
            shown = "; ".join(cells[:6]) + (f"; ... ({len(cells)} cells)" if len(cells) > 6 else "")
            lines.append(f"| {arm} | {k} | {why.replace('|', '/')[:300]} | {shown} |")
        lines.append("")
    memrows = {}
    for r in tim:
        if "memory" in r:
            memrows[(r["arm"], r["T"], r["weights"], r["mask"])] = (r["memory"], r.get("run_to_run_bitwise"))
    if memrows:
        lines += ["### Memory (eager warm call) and run-to-run bitwise (step, two calls)", "",
                  "| arm | T | weights | mask | retained after fwd (MiB, excl. out) | fwd peak extra (MiB) | "
                  "bwd peak extra (MiB) | run-to-run bitwise |", "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |"]
        for (arm, T, w, m), (mem, r2r) in sorted(memrows.items(), key=lambda kv: (order.index(kv[0][0])
                                                                                if kv[0][0] in order else 99, kv[0][1:])):
            mib = lambda k, mem=mem: f"{mem[k] / 2**20:.1f}" if k in mem else "-"  # noqa: E731
            lines.append(f"| {arm} | {T} | {w} | {m} | {mib('fwd_retained_bytes')} | {mib('fwd_peak_extra_bytes')} | "
                         f"{mib('bwd_peak_extra_bytes')} | {r2r} |")
        lines.append("")
    acc = [r for r in recs if r.get("kind") == "accuracy"]
    for wname in ("fp32", "bf16"):
        sub = [r for r in acc if r["weights"] == wname]
        if not sub:
            continue
        fp = next((r for r in sub if r["arm"] == "fp32_eval_cpu"), None)
        lines += [f"## Accuracy (issue #82 protocol), {wname} q/k weight leaves", ""]
        if fp:
            b = fp["bound"]
            lines += ["Bound = 2 x FP32 evaluation of the reference on the 25 random draws (med / max): "
                      + ", ".join(f"{n} {b[n]['med']:.2e} / {b[n]['max']:.2e}" for n in NAMES) + ".",
                      ("Issue-printed FP32 evaluation (med / max): " + ", ".join(
                          f"{n} {v[0]:.2e} / {v[1]:.2e}" for n, v in ISSUE_FP32_RANDOM.items()) + ".")
                      if wname == "fp32" else "", ""]
        lines += ["Random draws (whole tensors, med / max over 25 draws):", "",
                  "| arm | " + " | ".join(NAMES) + " | random pass |", "| --- | " + " | ".join("---" for _ in NAMES)
                  + " | --- |"]
        for r in ([fp] if fp else []) + [r for a in order for r in sub if r["arm"] == a]:
            if "random" not in r:
                lines.append(f"| {r['arm']} | " + " | ".join("-" for _ in NAMES)
                             + f" | {('unsupported: ' + r['unsupported']) if 'unsupported' in r else 'error: ' + r.get('error', '?')[:200]} |")
                continue
            ok = r.get("random_pass", {})
            lines.append(f"| {r['arm']} | " + " | ".join(
                f"{r['random'][n]['med']:.2e} / {r['random'][n]['max']:.2e}{'' if ok.get(n, True) else ' FAIL'}"
                for n in NAMES) + f" | {all(ok.values()) if ok else 'bound source'} |")
        lines += ["", "Near-cancel draws (per stream, med / max over 100 streams; bound = random-draw bound):", "",
                  "| arm | " + " | ".join(NEAR_NAMES) + " | near pass | mask exact | finite | run-to-run (T=16) | passed |",
                  "| --- | " + " | ".join("---" for _ in NEAR_NAMES) + " | --- | --- | --- | --- | --- |"]
        for r in ([fp] if fp else []) + [r for a in order for r in sub if r["arm"] == a]:
            if "near_per_stream" not in r:
                continue
            ok = r.get("near_pass", {})
            ns = r["near_per_stream"]
            lines.append(f"| {r['arm']} | " + " | ".join(
                f"{ns[n]['med']:.2e} / {ns[n]['max']:.2e}{'' if ok.get(n, True) else ' FAIL'}" for n in NEAR_NAMES)
                + f" | {all(ok.values()) if ok else '-'} | {r.get('mask_exact', '-')} | {r.get('finite', '-')} | "
                f"{r.get('run_to_run_bitwise_t16', '-')} | {r.get('passed', '-')} |")
        notes = [f"- {r['arm']}: {r['note']}" for r in sub if r.get("note")]
        lines += [""] + notes + ([""] if notes else [])
    errs = [r for r in recs if r.get("kind") == "unit_error"]
    if errs:
        lines += ["## Unit errors", ""] + [f"- {r['unit']}: {r['error']}" for r in errs] + [""]
    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.md}: {len(tim)} timing cells, {len(acc)} accuracy records, {len(errs)} unit errors")


def csv_list(conv):
    return lambda s: [conv(x) for x in s.split(",") if x]


def main():
    p = argparse.ArgumentParser(description="Benchmark of the Cake Engram four-stream context gate.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    r = sub.add_parser("run")
    r.add_argument("--arms", type=csv_list(str), default=list(DEFAULT_ARMS))
    r.add_argument("--T", type=csv_list(int), default=[1, 129, 4096, 8192, 16384, 65536])
    r.add_argument("--weights", type=csv_list(str), default=["bf16", "fp32"])
    r.add_argument("--masks", type=csv_list(str), default=["mask", "none"])
    r.add_argument("--mask-every", type=int, default=4)
    r.add_argument("--modes", type=csv_list(str), default=["eager", "graph"])
    r.add_argument("--phases", type=csv_list(str), default=["fwd", "bwd", "step", "ckpt"])
    r.add_argument("--dry", type=int, default=5)
    r.add_argument("--repeat", type=int, default=20)
    r.add_argument("--groups", type=int, default=3)
    r.add_argument("--no-r2r", dest="r2r", action="store_false")
    r.add_argument("--acc-weights", type=csv_list(str), default=["fp32", "bf16"])
    r.add_argument("--draws", type=int, default=25)
    r.add_argument("--no-accuracy", action="store_true")
    r.add_argument("--accuracy-only", action="store_true")
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--nshards", type=int, default=1)
    r.add_argument("--json", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--json", nargs="+", required=True)
    rep.add_argument("--md", required=True)
    rep.add_argument("--title", default="")
    args = p.parse_args()
    if args.cmd == "probe":
        probe()
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
