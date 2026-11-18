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
"""

"""Benchmark the experimental Cake chunked LM-head + loss training kernels (SM100 / SM103 / SM107).

Times the forward, the backward, the whole training step (forward +
backward, wrapper-inclusive) and the checkpointed training step of the
deterministic ``cake_pplx.cake_lm_head_det_train`` and the non-deterministic
``cake_pplx.cake_lm_head_nondet_train`` entries beside fixed non-Cake
baselines, the arms of an invocation in one process on one GPU, on the
GLM-class geometry (``H = 6144``, ``V = 154880``, BF16 ``X`` / ``W``, int64
labels with five percent ignored rows) over irregular token counts: the
loss entry with the cross-entropy and the policy objective at chunks of
2048 / 4096 / 8192 rows (``T`` = 16231, 16172, 4096, 32463), the
log-probability entry (recompute backward) at the same shapes, and the tail
cases ``T`` = 1, 4095, 4097 (informational).

Cake arms: ``cake`` / ``cake_nd``, the deterministic / non-deterministic
public entry of the row (``cake_lm_head_det_train_loss`` /
``cake_lm_head_nondet_train_loss`` on the loss rows, ``..._logprob`` on the
log-probability rows); ``cake_logprob`` / ``cake_nd_logprob`` compose that
mode's log-probability entry with the row's objective in eager PyTorch (the
policy rows: log-probabilities plus an eager policy loss, against the fused
policy loss of the ``cake`` / ``cake_nd`` arm on the same inputs).  Both
modes take the same inputs, switches, temperature and ``memory_mode``
argument; the two packages resolve its default differently (see Memory form).

Baselines: the unchunked PyTorch autograd path (``torch_unchunked``, the
accuracy reference), a chunked PyTorch loop at the same chunk and rounding
boundaries (``torch_chunked``: forward-gradient form for the loss entry,
recompute form for the log-probability entry), and, when importable, Liger's
fused linear cross-entropy and Cut Cross-Entropy.  ``liger``:
``LigerFusedLinearCrossEntropyFunction`` with ``accum_dtype=torch.float32``
(the weight gradient accumulated in FP32 across Liger's token chunks, cast
to BF16 once).  On the cross-entropy rows ``reduction="sum"``: its forward
forms the gradients, and its backward scales the saved BF16 ``dX`` / ``dW``
in place by the upstream gradient ``1 / loss_div`` (one more BF16 rounding;
Liger takes no divisor).  On the policy and log-probability rows its
per-token losses (``reduction="none"``): its forward computes the losses
only; its backward recomputes each chunk's logits, multiplies each row's
BF16 logit gradient by that row's upstream gradient cast to BF16, then runs
the gradient GEMMs (the FP32 ``dW`` accumulation and its cast happen
there).  ``logp = -loss`` on the valid rows (0 on the ignored rows) and the
row's objective in eager PyTorch; its internal token chunk is recorded.
``liger_default``: the same calls at Liger's default ``accum_dtype=None``
(the weight gradient accumulated in BF16 across its token chunks),
informational only, never a gate.  ``cce``: ``impl="cce"`` without gradient
filtering (``filter_eps=None``); the policy and log-probability rows composed
from its per-token losses.  Its backward accumulates ``dX`` and ``dW`` in the
inputs' dtype, BF16 here: each block's partial product is rounded to BF16
and added under a lock in the blocks' arrival order (cut-cross-entropy 25.1.1
has no FP32 accumulation option).  That is a weaker precision contract than
the FP32 accumulation of the PyTorch, ``liger`` and Cake arms: its ``dX``
error is a multiple of the PyTorch path's (``--accuracy``).  Its ``versions``
record it as ``grad_accumulation``, read from its source.  ``cce_default``:
its default gradient filtering (``filter_eps="auto"``; in 25.1.1 the same
threshold as the default ``"high"``), approximate and informational only,
never a gate.

Temperature (``--temperature scalar|constant|hetero``, value
``--temperature-value``): the log-probability rows divide the logits by the
per-token temperature (``scalar``: a Python number; ``constant``: the
``[T]`` tensor of that value; ``hetero``: a log-uniform ``[T]`` tensor in
[0.25, 4] with some rows at 1.0).  On those rows ``torch_unchunked`` is the
non-Cake baseline ``log_softmax(X @ W^T / tau)`` with autograd and
``torch_chunked`` / ``cake`` / ``cake_nd`` take the same temperature; an arm
without a temperature form is reported unsupported.  On the loss rows only
``cake_logprob`` / ``cake_nd_logprob`` take the temperature (the loss entry
has none): the other arms run the same inputs without it.  ``--accuracy`` has
no oracle for the loss-row objective with a temperature (``cake_logprob`` /
``cake_nd_logprob`` there are timing only).

Memory form (``--memory-mode default|fast|budget``): the ``memory_mode`` the
``cake`` / ``cake_logprob`` / ``cake_nd`` / ``cake_nd_logprob`` arms pass
(``default``: none).  The packages resolve none differently: the
deterministic entries take their automatic form (``"auto"``: per call, the
fastest form unless its predicted step peak exceeds the bytes of a head that
materializes the logits, then a budget form), the non-deterministic entries
their fastest form (``"fast"``).  On a row where the automatic form picks a
budget form, ``cake`` and ``cake_nd`` therefore run different memory forms.
Each Cake arm's ``versions`` records ``memory_form``, the resolved mode
(``"auto"``: the benchmark does not observe the per-call choice).
``--memory-mode fast`` runs both modes at their fastest form.  ``--ignore-frac``:
the fraction of ignored rows of every row's inputs (default: the inputs' five
percent).

Ops per arm: ``fwd``, ``bwd`` (the forward's state kept, graph retained),
``step`` (forward + backward, the backward freeing the graph), ``step_miss``
(``step`` with the binding caches of both Cake packages cleared before every
step) and ``ckpt``: ``torch.utils.checkpoint.checkpoint(<the arm's forward of
(X, W) to its loss>, X, W, use_reentrant=False)`` and the ``step`` backward,
timed like ``step`` (the Cake arms call their public entry inside the
checkpoint).  For the PyTorch, Liger and CCE arms that is 2 x forward +
backward.  The Cake entries recognise a non-reentrant checkpoint under their
default gradient placement (``"auto"``): the loss entry's checkpointed
forward computes the loss only and its recomputation forms the gradient
accumulators (one gradient pass per step); the log-probability entry's
recomputation takes the checkpointed call's row results.  A torch whose
checkpoint lacks the hook structure they read keeps their fused forward in
both passes.  Each Cake arm's ``versions`` records ``gradient_placement`` and
``checkpoint_supported``.  An arm without an autograd forward of ``(X, W)``
(``torch_chunked``) reports ``ckpt`` unsupported with the reason.

Timing: the local ``bench_gpu_time`` with CUPTI activity tracing (``torch.profiler``)
and a cold L2 between iterations (per-iteration GPU span); medians over
``--steps`` iterations.  Memory: the peak allocation above the live tensors
during the forward and during the backward, and the bytes the forward leaves
alive for the backward (saved state) -- the timed backward retains the graph --
plus the step peak of a training step whose backward frees the graph (forward
through backward, outputs and gradients included; the entries release their
saved state early there) and its ratio to the bytes of a head that
materializes the BF16 logits of the valid rows, ``B = 2 * T_v * V + 2 * V * H +
2 * T * H``, from the remembered bindings and with the binding caches cleared
first (``step_peak_hit_over_b`` / ``step_peak_miss_over_b``: a new valid-row
count misses the cache, usually every training batch; ``step_peak_over_b`` is
the larger), ``step_miss_ms``, the step time with the caches cleared before
every step, and ``ckpt_ms`` (``ckpt_unsupported``: the reason an arm has
none) with the checkpointed step's memory from the remembered bindings:
``ckpt_saved_gib``, the bytes its checkpointed forward leaves alive for the
backward, and ``ckpt_peak_gib`` / ``ckpt_peak_over_b``, its peak (forward
through backward, outputs and gradients included).  ``--accuracy`` adds the
error report of every arm (its ``fwd`` / ``bwd`` outputs) against the chunked
FP64 oracle and its ratio to the unchunked PyTorch path (``passes_1p05x``:
every ratio at most 1.05; a benchmark record, the tests hold no such gate),
and the same report of its checkpointed step's ``dX`` / ``dW`` (``ckpt``,
with ``same_bits_as_bwd``: whether they equal the ``bwd`` gradients
bitwise).  The JSON carries an ``env`` record (torch, its CUDA, triton,
the cuBLAS distribution, liger-kernel and cut-cross-entropy versions, device
name and compute capability; the driver is recorded outside the benchmark) and
per arm its ``versions`` (Liger: ``accum_dtype``, ``reduction``, its FP32
``addmm(out_dtype)`` gate and the internal token chunk with the source lines
of its formula; CCE: ``impl``, ``filter_eps`` and ``grad_accumulation`` with
its source lines; Cake: the mode, module, stages, switches, ``memory_form``,
``gradient_placement`` and ``checkpoint_supported``).

Usage::

    python benchmarks/bench_cake_lm_head_det_train.py [--rows t16231_c4096_ce ...]
        [--arms torch_unchunked,torch_chunked,liger,liger_default,cce,cce_default,
                cake,cake_logprob,cake_nd,cake_nd_logprob] [--steps 20] [--accuracy]
        [--temperature hetero] [--memory-mode default] [--ignore-frac 0.33] [--json out.json]
"""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import re
import statistics
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import torch
import torch.profiler
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cake_pplx import _cake_lm_head_train_knobs as _knobs  # noqa: E402
from cake_pplx.cake_lm_head_det_train import (
    cake_lm_head_det_train_logprob,
    cake_lm_head_det_train_loss,
)  # noqa: E402
from cake_pplx._backends.cake_lm_head_det_train import cake_backend  # noqa: E402
from tests.experimental.test_helpers.cake_lm_head_loss_reference import (  # noqa: E402
    DEFAULT_H,
    DEFAULT_LOSS_DIV,
    DEFAULT_V,
    DEFAULT_TEMPERATURE,
    IGNORE_INDEX,
    RATIO_CLIP,
    TEMPERATURE_FORMS,
    error_report,
    gate_ratios,
    make_inputs,
    make_weight,
    reference_fp64,
    reference_unchunked,
    rel_l2,
)

# name -> (T, chunk, objective, entry)
ROWS: dict[str, tuple[int, int, str, str]] = {}
_SHAPES = (
    (16231, 4096),
    (16172, 4096),
    (16231, 2048),
    (16231, 8192),
    (4096, 4096),
    (32463, 4096),
)
for _T, _C in _SHAPES:
    ROWS[f"t{_T}_c{_C}_ce"] = (_T, _C, "ce", "loss")
for _T, _C in _SHAPES:
    ROWS[f"t{_T}_c{_C}_policy"] = (_T, _C, "policy", "loss")
for _T, _C in _SHAPES:
    ROWS[f"t{_T}_c{_C}_logprob"] = (_T, _C, "ce", "logprob")
for _T in (1, 4095, 4097):
    ROWS[f"t{_T}_c4096_ce"] = (_T, 4096, "ce", "loss")

SEED_BASE = 761
H, V = DEFAULT_H, DEFAULT_V
LOSS_DIV = DEFAULT_LOSS_DIV


def bench_gpu_time(fn, *, dry_run_iters: int, repeat_iters: int) -> list[float]:
    """Per-iteration GPU span of ``fn`` in ms: the first kernel's start to the last kernel's end of one call, read
    from CUPTI activity records through ``torch.profiler`` (GPU activity only, one profiled call at a time).  Before
    every call the L2 cache is flushed by writing a buffer of at least twice its size, outside the profiled span
    (cold L2); ``dry_run_iters`` untimed calls run first."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    flush = torch.empty(
        max(2 * getattr(props, "L2_cache_size", 0), 256 << 20), dtype=torch.int8, device="cuda"
    )
    for _ in range(dry_run_iters):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat_iters):
        flush.zero_()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        spans = [
            e.time_range for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        if not spans:
            raise RuntimeError("the profiler recorded no GPU activity of the timed call")
        times.append((max(s.end for s in spans) - min(s.start for s in spans)) / 1e3)
    return times


def median_ms(fn, steps):
    times = bench_gpu_time(fn, dry_run_iters=3, repeat_iters=steps)
    return float(statistics.median(times))


def gib(nbytes):
    return nbytes / 2**30


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def _mm_fp32(a, b):
    """``a @ b`` of BF16 operands with an FP32 result and no intermediate rounding."""
    try:
        return torch.mm(a, b, out_dtype=torch.float32)
    except (TypeError, RuntimeError):
        return torch.mm(a.float(), b.float())


def _addmm_fp32(acc, a, b):
    """``acc += a @ b`` of BF16 operands into the FP32 accumulator without intermediate rounding;
    returns the path taken (``addmm(out_dtype)``, ``mm(out_dtype)`` or FP32 operands)."""
    try:
        torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
        return "addmm_out_dtype"
    except (TypeError, RuntimeError):
        pass
    try:
        acc.add_(torch.mm(a, b, out_dtype=torch.float32))
        return "mm_out_dtype"
    except (TypeError, RuntimeError):
        acc.add_(torch.mm(a.float(), b.float()))
        return "float32_operands"


def _liger_internal_chunk(T, H, V, passed):
    """Liger's internal token chunk, computed from the formula in its source (``fused_linear_cross_entropy_forward``
    of ``liger_kernel.ops.fused_linear_cross_entropy``; liger-kernel 0.8.5: ``inc_factor = triton.cdiv(V,
    chunk_mem_const * H)``, ``chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))``, ``chunk_size =
    min(chunk_size, BT)``; older releases ``triton.cdiv(V, H)`` without the cap) and the forward arguments the arm
    passes (``passed``: a name in the ``inc_factor`` product, ``chunk_mem_const``, takes its passed value).  The
    formula's source lines are kept with their line numbers; ``"n/a"`` when the source does not match."""
    try:
        from liger_kernel.ops import fused_linear_cross_entropy as module

        src = inspect.getsource(module)
        lines = [
            f"L{n}: {line.strip()}"
            for n, line in enumerate(src.splitlines(), start=1)
            if re.search(r"\b(inc_factor|chunk_size|num_chunks)\s*=[^=]", line)
            and not line.strip().startswith("#")
        ][:8]
        match = re.search(r"inc_factor\s*=\s*triton\.cdiv\(\s*V\s*,\s*([^)]+)\)", src)
        pow2 = re.search(
            r"chunk_size\s*=\s*triton\.next_power_of_2\(\s*triton\.cdiv\(\s*BT\s*,\s*inc_factor\s*\)\s*\)",
            src,
        )
        if match is None or pow2 is None:
            return dict(chunk="n/a", source=lines)
        k, h_terms = 1, 0
        for term in (t.strip() for t in match.group(1).split("*")):
            value = passed.get(term)
            if term == "H":
                h_terms += 1
            elif term.isdigit():
                k *= int(term)
            elif isinstance(value, int) and not isinstance(value, bool):
                k *= value
            else:
                return dict(chunk="n/a", source=lines, unmatched_term=term)
        if h_terms != 1:
            return dict(chunk="n/a", source=lines)
        capped = re.search(r"chunk_size\s*=\s*min\(\s*chunk_size\s*,\s*BT\s*\)", src) is not None
        inc_factor = -(-V // (k * H))
        chunk = 1 << (max(1, -(-T // inc_factor)) - 1).bit_length()
        if capped:
            chunk = min(chunk, T)
        return dict(
            chunk=chunk,
            num_chunks=-(-T // chunk),
            inc_factor=inc_factor,
            budget_factor=k,
            capped=capped,
            source=lines,
        )
    except Exception as exc:
        return dict(chunk="n/a", error=f"{type(exc).__name__}: {exc}")


def _cce_grad_accumulation(fn):
    """The dtype Cut Cross-Entropy's backward accumulates ``dX`` / ``dW`` in, read from its source
    (``cut_cross_entropy.cce_backward``; cut-cross-entropy 25.1.1: ``de = torch.zeros_like(e)``, ``dc =
    torch.zeros_like(c, dtype = e.dtype)``, each block's partial product rounded to that dtype, ``da_i = tl.dot(do,
    b).to(da_ptrs.dtype.element_ty)``, and added under a lock in the blocks' arrival order, ``tl_lock_add(da_ptrs,
    ...)``: load, add, store): ``"input dtype"`` (BF16 here; across the vocabulary blocks for ``dX``, the token blocks
    for ``dW``) when the source matches, else ``"n/a"``.  The matched source lines are kept with their line numbers,
    with the FP32 / Kahan accumulation options of ``fn``'s signature (``fp32_options``: none in 25.1.1; the arm passes
    none)."""
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []
    options = [p for p in params if re.search(r"fp32|kahan", p)]
    try:
        from cut_cross_entropy import cce_backward as module

        src = inspect.getsource(module)
        lines = [
            f"L{n}: {line.strip()}"
            for n, line in enumerate(src.splitlines(), start=1)
            if re.search(
                r"\b(de|dc)\s*=\s*torch\.zeros_like\(|\.to\(\s*da_ptrs\.dtype\.element_ty\s*\)|tl_lock_add\(\s*da_ptrs",
                line,
            )
        ][:8]
        matched = all(
            re.search(pattern, src)
            for pattern in (
                r"\bde\s*=\s*torch\.zeros_like\(\s*e\s*\)",
                r"\bdc\s*=\s*torch\.zeros_like\(\s*c\s*,\s*dtype\s*=\s*e\.dtype\s*\)",
                r"tl\.dot\(\s*do\s*,\s*b\s*\)\.to\(\s*da_ptrs\.dtype\.element_ty\s*\)",
                r"tl_lock_add\(\s*da_ptrs",
            )
        )
        return dict(
            accumulation="input dtype" if matched else "n/a",
            order="lock-add in arrival order" if matched else "n/a",
            fp32_options=options,
            source=lines,
        )
    except Exception as exc:
        return dict(
            accumulation="n/a",
            fp32_options=options,
            error=f"{type(exc).__name__}: {exc}",
        )


def _probe(fn, *args):
    """``fn(*args)`` for a ``versions`` record, or ``"? (<error>)"`` when it raises (a record never fails the arm)."""
    try:
        return fn(*args)
    except Exception as exc:
        return f"? ({type(exc).__name__}: {exc})"


def _triton_version():
    import triton

    return triton.__version__


def _cublas_version():
    """The cuBLAS wheel the process can see (``"<distribution>==<version>"``), or ``"?"`` when cuBLAS is not a pip
    distribution (e.g. a container's system library)."""
    for dist in ("nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12"):
        version = _version(dist)
        if version != "?":
            return f"{dist}=={version}"
    return "?"


def _downstream_loss(logp, inp):
    """The log-probability entry's downstream loss ``(logp * dlogp)[valid].sum()``."""
    return (logp * inp.dlogp)[inp.valid].sum()


def _objective_loss(logp, inp):
    if inp.objective == "ce":
        return -logp[inp.valid].sum() / inp.loss_div
    ratio = torch.exp(logp - inp.infer_logp)
    return -(inp.loss_weights * torch.clamp_max(ratio, RATIO_CLIP))[inp.valid].sum()


# ---------------------------------------------------------------------------
# Arms: ``forward()`` returns the state (``loss`` FP32 scalar, ``logp`` FP32 [T] plus what the
# backward needs); ``backward(st)`` returns ``dX`` (BF16 [T, H]) and ``dW`` (BF16 [V, H]);
# ``ckpt_step()`` runs the checkpointed step (``ckpt_forward()`` then ``free_backward``; an arm
# without it reports ``ckpt`` unsupported with its ``CKPT_UNSUPPORTED`` reason).
# ``TEMPERATURE_ENTRIES``: the row entries on which an arm applies the inputs' temperature
# (:func:`_arm_or_error` gives the other loss-row arms the inputs without it and reports the
# other log-probability-row arms unsupported).
# ---------------------------------------------------------------------------


class ArmAutograd:
    """Base of the autograd arms: leaves, retained-graph backward; ``_forward_xw(X, W)`` is the arm's forward of the
    two inputs (``forward()`` runs it on the leaves; :meth:`ckpt_step` runs it inside the checkpoint)."""

    TEMPERATURE_ENTRIES: tuple = ()

    def __init__(self, inp, *, entry, chunk):
        self.inp, self.entry, self.chunk = inp, entry, chunk
        self.X = inp.X.detach().requires_grad_()
        self.W = inp.W.detach().requires_grad_()
        # the temperature as the FP32 [T] tensor the PyTorch arms divide by (tensor by tensor); None without one
        self.tau = inp.temperature_rows()

    def forward(self):
        return self._forward_xw(self.X, self.W)

    def ckpt_forward(self):
        """The checkpointed forward: ``torch.utils.checkpoint.checkpoint`` (non-reentrant) of the arm's forward of
        ``(X, W)`` to its loss."""
        loss = torch.utils.checkpoint.checkpoint(
            lambda x, w: self._forward_xw(x, w)["loss"],
            self.X,
            self.W,
            use_reentrant=False,
        )
        return dict(loss=loss)

    def ckpt_step(self):
        """The checkpointed training step: :meth:`ckpt_forward`, then the step's backward (:meth:`free_backward`),
        which recomputes the checkpointed forward -- 2 x forward + backward for an arm that does not recognise the
        checkpoint (the Cake entries do: module docstring, Ops)."""
        return self.free_backward(self.ckpt_forward())

    def backward(self, st):
        dX, dW = torch.autograd.grad(st["loss"], (self.X, self.W), retain_graph=True)
        return dict(dX=dX, dW=dW)

    def free_backward(self, st):
        """The backward of a training step: it frees the graph (no ``retain_graph``), so the entries that release
        their saved state early do (the step-peak probe, :func:`_step_peak`)."""
        dX, dW = torch.autograd.grad(st["loss"], (self.X, self.W))
        return dict(dX=dX, dW=dW)

    def outputs(self, st):
        return dict(loss=st["loss"].detach(), logp=st["logp"].detach())

    def _finish(self, logp):
        loss = (
            _downstream_loss(logp, self.inp)
            if self.entry == "logprob"
            else _objective_loss(logp, self.inp)
        )
        return dict(loss=loss, logp=logp)


class ArmTorchUnchunked(ArmAutograd):
    """The unchunked PyTorch path (accuracy reference): BF16 GEMM promoted to FP32, logsumexp, autograd.  With a
    temperature (log-probability rows) the non-Cake baseline ``log_softmax(X @ W^T / tau)``: the FP32 logits
    divided by the ``[T]`` temperature tensor before the logsumexp."""

    name = "torch_unchunked"
    TEMPERATURE_ENTRIES = ("logprob",)

    def versions(self):
        return dict(torch=torch.__version__, temperature=self.tau is not None)

    def _forward_xw(self, X, W):
        inp = self.inp
        z = (X @ W.t()).float()
        if self.tau is not None:
            z = z / self.tau[:, None]  # tensor by tensor
        lse = torch.logsumexp(z, dim=-1)
        index = torch.where(inp.valid, inp.labels, torch.zeros_like(inp.labels))
        zy = z.gather(1, index[:, None]).squeeze(1)
        logp = torch.where(inp.valid, zy - lse, torch.zeros_like(lse))
        return self._finish(logp)


class ArmTorchChunked:
    """Chunked PyTorch loop at the same chunk and rounding boundaries, without autograd.

    Loss entry: the forward produces the FP32 gradient accumulators chunk by
    chunk (three GEMMs), the backward casts them.  Log-probability entry: the
    forward saves ``lse``; the backward recomputes each chunk's logits.  With a
    temperature (log-probability rows) each chunk's FP32 logits are divided by
    its ``[rows]`` temperature tensor and the row gradient is ``dlogp / tau``.
    """

    name = "torch_chunked"
    TEMPERATURE_ENTRIES = ("logprob",)
    CKPT_UNSUPPORTED = (
        "torch_chunked: no autograd forward of (X, W) to checkpoint (its forward forms the FP32 gradient "
        "accumulators without autograd; its backward casts or recomputes them)"
    )

    def __init__(self, inp, *, entry, chunk):
        self.inp, self.entry, self.chunk = inp, entry, chunk
        self.T = inp.T
        self.tau = inp.temperature_rows()
        self.dx_acc = torch.empty((self.T, H), dtype=torch.float32, device=inp.X.device)
        self.dw_acc = torch.empty((V, H), dtype=torch.float32, device=inp.X.device)

    def versions(self):
        probe = torch.zeros((8, 8), dtype=torch.float32, device=self.inp.X.device)
        a = torch.zeros((8, 8), dtype=torch.bfloat16, device=probe.device)
        return dict(
            torch=torch.__version__,
            dw_accumulate=_addmm_fp32(probe, a.t(), a),
            dx_out=("out_dtype" if _mm_fp32(a, a).dtype == torch.float32 else "?"),
            temperature=self.tau is not None,
        )

    def _logits(self, r0, r1):
        """The chunk's BF16 GEMM output promoted to FP32 (divided by the chunk's temperature rows, tensor by
        tensor, when the row has a temperature)."""
        inp = self.inp
        zf = torch.mm(inp.X[r0:r1], inp.W.t()).float()
        return zf if self.tau is None else zf / self.tau[r0:r1, None]

    def _chunk_stats(self, r0, r1):
        inp = self.inp
        zf = self._logits(r0, r1)
        lse = torch.logsumexp(zf, dim=-1)
        labels = inp.labels[r0:r1]
        valid = labels >= 0
        index = torch.where(valid, labels, torch.zeros_like(labels))
        zy = zf.gather(1, index[:, None]).squeeze(1)
        logp = torch.where(valid, zy - lse, torch.zeros_like(lse))
        return zf, lse, valid, index, logp

    @staticmethod
    def _dlogits(zf, lse, index, d):
        dz = torch.exp(zf - lse[:, None]).neg_()
        dz.scatter_add_(1, index[:, None], torch.ones_like(lse)[:, None])
        return dz.mul_(d[:, None]).to(torch.bfloat16)  # the BF16 dlogits boundary

    def _accumulate(self, r0, r1, dz, first):
        inp = self.inp
        self.dx_acc[r0:r1] = _mm_fp32(dz, inp.W)
        if first:
            self.dw_acc.zero_()
        _addmm_fp32(self.dw_acc, dz.t(), inp.X[r0:r1])  # dW_acc += dz^T @ X_c in FP32

    def forward(self):
        inp = self.inp
        logp_out = torch.empty((self.T,), dtype=torch.float32, device=inp.X.device)
        lse_out = torch.empty_like(logp_out)
        total = torch.zeros((), dtype=torch.float32, device=inp.X.device)
        for i, r0 in enumerate(range(0, self.T, self.chunk)):
            r1 = min(self.T, r0 + self.chunk)
            zf, lse, valid, index, logp = self._chunk_stats(r0, r1)
            logp_out[r0:r1], lse_out[r0:r1] = logp, lse
            zero = torch.zeros_like(lse)
            if self.entry == "logprob":
                total += (logp * inp.dlogp[r0:r1])[valid].sum()
                continue
            if inp.objective == "ce":
                d = torch.where(valid, torch.full_like(lse, -1.0 / inp.loss_div), zero)
                total += logp.sum()
            else:
                ratio = torch.exp(logp - inp.infer_logp[r0:r1])
                w = inp.loss_weights[r0:r1]
                d = torch.where(valid & (ratio <= RATIO_CLIP), -w * ratio, zero)
                total += torch.where(
                    valid, w * torch.clamp_max(ratio, RATIO_CLIP), zero
                ).sum()
            self._accumulate(r0, r1, self._dlogits(zf, lse, index, d), i == 0)
        if self.entry == "logprob":
            loss = total
        else:
            loss = -total / inp.loss_div if inp.objective == "ce" else -total
        return dict(loss=loss, logp=logp_out, lse=lse_out)

    def backward(self, st):
        inp = self.inp
        if self.entry == "logprob":
            for i, r0 in enumerate(range(0, self.T, self.chunk)):
                r1 = min(self.T, r0 + self.chunk)
                zf = self._logits(r0, r1)  # recompute
                labels = inp.labels[r0:r1]
                valid = labels >= 0
                index = torch.where(valid, labels, torch.zeros_like(labels))
                g = inp.dlogp[r0:r1]
                if self.tau is not None:
                    # the 1 / tau chain factor, tensor by tensor
                    g = g / self.tau[r0:r1]
                d = torch.where(valid, g, torch.zeros_like(st["lse"][r0:r1]))
                self._accumulate(
                    r0, r1, self._dlogits(zf, st["lse"][r0:r1], index, d), i == 0
                )
        return dict(
            dX=self.dx_acc.to(torch.bfloat16), dW=self.dw_acc.to(torch.bfloat16)
        )

    def outputs(self, st):
        return dict(loss=st["loss"], logp=st["logp"])


class ArmLiger(ArmAutograd):
    """Liger fused linear cross-entropy (``LigerFusedLinearCrossEntropyFunction``) with FP32 weight-gradient
    accumulation (``accum_dtype=torch.float32``: the ``dW`` accumulator across its token chunks is FP32, cast to the
    weight's dtype once at the end of the pass that forms the gradients).  Cross-entropy loss rows:
    ``reduction="sum"``, ``loss = sum / loss_div``; its forward forms the gradients (the FP32 accumulation and the
    cast there) and its backward scales the saved BF16 ``dX`` / ``dW`` in place by the upstream gradient ``1 /
    loss_div``, one more BF16 rounding (Liger takes no divisor).  Policy and log-probability rows: per-token losses
    (``reduction="none"``); its forward computes the losses only, its backward recomputes each chunk's logits,
    multiplies each row's BF16 logit gradient by that row's upstream gradient cast to BF16, then runs the dX / dW
    GEMMs (the FP32 accumulation and the cast there).  ``logp = -loss`` on the valid rows and 0 on the ignored rows,
    the row's objective in eager PyTorch as :class:`ArmCCE` composes it.  Its internal token chunk is recorded
    (:func:`_liger_internal_chunk`), and its gate of the FP32 ``addmm(out_dtype)`` accumulation path
    (``_ADDMM_SUPPORTS_OUT_DTYPE``, a torch version check; ``"n/a"`` when the release has none)."""

    name = "liger"
    accum_dtype = torch.float32  # passed as ``accum_dtype``; ``None``: the forward's default is passed
    informational = False

    def __init__(self, inp, *, entry, chunk):
        super().__init__(inp, entry=entry, chunk=chunk)
        from liger_kernel.ops import fused_linear_cross_entropy as module
        from liger_kernel.ops.fused_linear_cross_entropy import (
            LigerFusedLinearCrossEntropyFunction,
            fused_linear_cross_entropy_forward,
        )

        self.addmm_out_dtype = getattr(module, "_ADDMM_SUPPORTS_OUT_DTYPE", "n/a")
        self.function = LigerFusedLinearCrossEntropyFunction
        params = list(inspect.signature(self.function.forward).parameters.values())[
            1:
        ]  # drop ctx
        names = [p.name for p in params]
        required = {"ignore_index", "reduction"} | (
            set() if self.accum_dtype is None else {"accum_dtype"}
        )
        if not required <= set(names):
            raise RuntimeError(
                f"unexpected LigerFusedLinearCrossEntropyFunction.forward signature: {names}"
            )
        self.reduction = "sum" if entry == "loss" and inp.objective == "ce" else "none"
        if self.reduction == "none" and "token_grad_output" not in inspect.signature(
            fused_linear_cross_entropy_forward
        ).parameters:
            raise NotImplementedError(
                "liger: this release's reduction='none' backward does not apply the per-token upstream gradient "
                "before its dW sum (no token_grad_output)"
            )
        overrides = {
            names[0]: self.X,
            names[1]: self.W,
            names[2]: inp.labels,
            "ignore_index": IGNORE_INDEX,
            "reduction": self.reduction,
        }
        if self.accum_dtype is not None:
            overrides["accum_dtype"] = self.accum_dtype
        # positions 0 / 1 (X, W): replaced by the call's inputs (:meth:`_forward_xw`)
        self.args = [overrides.get(p.name, p.default) for p in params]
        self.passed = dict(zip(names, self.args))
        self.internal_chunk = _liger_internal_chunk(inp.T, H, V, self.passed)

    def versions(self):
        return dict(
            liger_kernel=_version("liger-kernel"),
            accum_dtype=str(self.passed.get("accum_dtype", "n/a")),
            reduction=self.reduction,
            addmm_out_dtype=self.addmm_out_dtype,
            internal_chunk=self.internal_chunk,
            informational=self.informational,
            torch=torch.__version__,
        )

    def _forward_xw(self, X, W):
        loss = self.function.apply(X, W, *self.args[2:])
        if isinstance(loss, tuple):
            loss = loss[0]
        if self.reduction == "sum":
            return dict(loss=loss / self.inp.loss_div, logp=None)
        logp = torch.where(self.inp.valid, -loss, torch.zeros_like(loss))
        return self._finish(logp)

    def outputs(self, st):
        return dict(
            loss=st["loss"].detach(),
            logp=None if st["logp"] is None else st["logp"].detach(),
        )


class ArmLigerDefault(ArmLiger):
    """Liger fused linear cross-entropy at its default ``accum_dtype=None`` (the ``dW`` accumulator in the weight's
    dtype, BF16 here, across its token chunks), on the rows and in the forms of :class:`ArmLiger`: a weaker
    precision contract than the FP32 accumulation -- informational only, never a gate."""

    name = "liger_default"
    accum_dtype = None
    informational = True


class ArmCCE(ArmAutograd):
    """Cut Cross-Entropy without gradient filtering (``impl="cce"``, ``filter_eps=None``): cross-entropy rows
    directly; policy and log-probability rows composed in PyTorch from its per-token losses (``reduction="none"``).
    Its backward accumulates ``dX`` / ``dW`` in the inputs' dtype (BF16 here, lock-added in the blocks' arrival
    order; cut-cross-entropy 25.1.1 has no FP32 accumulation option): a weaker precision contract than the FP32
    accumulation of the PyTorch, ``liger`` and Cake arms, recorded as ``grad_accumulation``
    (:func:`_cce_grad_accumulation`)."""

    name = "cce"
    filter_eps = None
    informational = False

    def __init__(self, inp, *, entry, chunk):
        super().__init__(inp, entry=entry, chunk=chunk)
        from cut_cross_entropy import linear_cross_entropy

        self.fn = linear_cross_entropy
        self.grad_accumulation = _cce_grad_accumulation(linear_cross_entropy)

    def versions(self):
        return dict(
            cut_cross_entropy=_version("cut-cross-entropy"),
            impl="cce",
            filter_eps=self.filter_eps,
            grad_accumulation=self.grad_accumulation,
            informational=self.informational,
            torch=torch.__version__,
        )

    def _cce(self, X, W, reduction):
        return self.fn(
            X,
            W,
            self.inp.labels,
            ignore_index=IGNORE_INDEX,
            reduction=reduction,
            impl="cce",
            filter_eps=self.filter_eps,
        )

    def _forward_xw(self, X, W):
        inp = self.inp
        if self.entry == "loss" and inp.objective == "ce":
            return dict(loss=self._cce(X, W, "sum") / inp.loss_div, logp=None)
        nll = self._cce(X, W, "none")
        logp = torch.where(inp.valid, -nll, torch.zeros_like(nll))
        return self._finish(logp)

    def outputs(self, st):
        return dict(
            loss=st["loss"].detach(),
            logp=None if st["logp"] is None else st["logp"].detach(),
        )


class ArmCCEDefault(ArmCCE):
    """Cut Cross-Entropy with its default gradient filtering (``impl="cce"``, ``filter_eps="auto"``; in 25.1.1 the
    same threshold as ``linear_cross_entropy``'s default ``"high"``) and the gradient accumulation of :class:`ArmCCE`:
    approximate -- informational only, never a gate."""

    name = "cce_default"
    filter_eps = "auto"
    informational = True


def _cake_api(deterministic):
    """``(backend module, loss entry, log-probability entry)`` of a Cake mode: the deterministic package's, or the
    non-deterministic package's (``cake_pplx.cake_lm_head_nondet_train``, imported on first use)."""
    if deterministic:
        return cake_backend, cake_lm_head_det_train_loss, cake_lm_head_det_train_logprob
    from cake_pplx._backends.cake_lm_head_nondet_train import (
        cake_backend as nondet_backend,
    )
    from cake_pplx.cake_lm_head_nondet_train import (
        cake_lm_head_nondet_train_logprob,
        cake_lm_head_nondet_train_loss,
    )

    return (
        nondet_backend,
        cake_lm_head_nondet_train_loss,
        cake_lm_head_nondet_train_logprob,
    )


class ArmCake(ArmAutograd):
    """The public entry points (``cake_pplx.cake_lm_head_det_train``, ``backend="cake"``); a
    log-probability row with a temperature passes the inputs' temperature argument as is (a Python number for the
    scalar form, the ``[T]`` tensor otherwise)."""

    name = "cake"
    deterministic = True  # the mode: the deterministic or the non-deterministic package (:func:`_cake_api`)
    TEMPERATURE_ENTRIES = ("logprob",)
    program_entry: str | None = None  # the program the arm needs: the row's entry
    # the ``memory_mode`` the arm passes (``None``: none, the entries' default; ``--memory-mode``)
    memory_mode: str | None = None

    def _memory_kw(self):
        return {} if self.memory_mode is None else dict(memory_mode=self.memory_mode)

    def __init__(self, inp, *, entry, chunk):
        super().__init__(inp, entry=entry, chunk=chunk)
        self.backend, self.loss_entry, self.logprob_entry = _cake_api(self.deterministic)
        device = inp.X.device
        hidden, vocab = int(inp.X.shape[1]), int(inp.W.shape[0])
        program = self.program_entry or entry
        if not self.backend.generated_program_available(device, entry=program):
            capability = torch.cuda.get_device_capability(device)
            raise RuntimeError(
                f"generated chunked LM-head program ({program} entry, H {hidden}, V {vocab}) not registered for "
                f"compute capability {capability[0]}.{capability[1]} in this checkout"
            )
        self.module_name, record = self.backend.record_for(device)
        self.abi = self.backend.record_abi(record)

    def _logprob(self, X, W):
        """The mode's log-probability entry of ``X`` / ``W``, with the inputs' temperature when they carry one."""
        kw = self._memory_kw()
        if self.inp.temperature is not None:
            kw["temperature"] = self.inp.temperature
        return self.logprob_entry(
            X, W, self.inp.labels, chunk_size=self.chunk, backend="cake", **kw
        )

    def _memory_form(self):
        """The ``memory_mode`` this arm runs: the one it passes, else its package's default (the opt-in module's
        ``memory_mode_default()``: ``"auto"`` in the deterministic package, ``"fast"`` in the non-deterministic one)."""
        if self.memory_mode is not None:
            return self.memory_mode
        package = "det" if self.deterministic else "nondet"
        module = importlib.import_module(
            f"cake_pplx._backends.cake_lm_head_{package}_train.cake_backend_opt_in"
        )
        return module.memory_mode_default()

    def versions(self):
        # the switches this arm runs with: the public module's resolution of its CAKE_PPLX_LM_HEAD_LOSS_*
        # variables (every keyword left at None), which it passes down to the backend
        switches = _knobs.resolve_knobs(
            compact_rows=None,
            fuse_dw_cast=None,
            dx_finalize=None,
            dw_stream=None,
            hidden_count=None,
            gemm_tuning=None,
        )
        mode = (
            dict(mode="det")
            if self.deterministic
            else dict(
                mode="nondet",
                dx_slice_policy=self.backend.dx_slice_policy(),
                one_pass_rows=sorted(self.backend.one_pass_rows()),
            )
        )
        return dict(
            **mode,
            module=self.module_name,
            abi=self.abi,
            stages=list(
                self.backend.stages_for_entry(
                    self.program_entry or self.entry,
                    fuse_dw_cast=switches["fuse_dw_cast"],
                )
            ),
            temperature=_temperature_form(self.inp.temperature),
            memory_mode=self.memory_mode or "default",
            memory_form=_probe(self._memory_form),
            # the checkpoint handling of the entries (module docstring, Ops): the loss entry's gradient placement and
            # whether this torch's checkpoint has the hook structure the entries read
            gradient_placement=_probe(lambda: self.backend.gradient_placement()),
            checkpoint_supported=_probe(
                lambda: self.backend._checkpoint_supported()
            ),
            **switches,
        )

    def _forward_xw(self, X, W):
        inp = self.inp
        if self.entry == "logprob":
            logp = self._logprob(X, W)
            return dict(loss=_downstream_loss(logp, inp), logp=logp)
        loss, logp = self.loss_entry(
            X,
            W,
            inp.labels,
            objective=inp.objective,
            loss_div=inp.loss_div if inp.objective == "ce" else None,
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=self.chunk,
            return_logp=True,
            backend="cake",
            **self._memory_kw(),
        )
        return dict(loss=loss, logp=logp)


class ArmCakeLogprob(ArmCake):
    """Loss rows through the log-probability entry: ``cake_lm_head_det_train_logprob`` (with the inputs' temperature when
    they carry one) and the row's objective in eager PyTorch on its ``logp`` -- for the policy rows the
    log-probabilities plus an eager policy loss, the matched-input comparison to the fused policy loss of the
    ``cake`` arm."""

    name = "cake_logprob"
    TEMPERATURE_ENTRIES = ("loss",)
    program_entry = "logprob"
    row_arm = "cake"  # the arm of the mode that runs the log-probability rows

    def __init__(self, inp, *, entry, chunk):
        if entry != "loss":
            raise NotImplementedError(
                f"{self.name}: loss rows only (the log-probability rows run it as the {self.row_arm} arm)"
            )
        super().__init__(inp, entry=entry, chunk=chunk)

    def _forward_xw(self, X, W):
        logp = self._logprob(X, W)
        return dict(loss=_objective_loss(logp, self.inp), logp=logp)


class ArmCakeNondet(ArmCake):
    """:class:`ArmCake` through the non-deterministic package's public entries
    (``cake_pplx.cake_lm_head_nondet_train``: ``cake_lm_head_nondet_train_loss`` /
    ``cake_lm_head_nondet_train_logprob``), the same inputs, switches, temperature and ``memory_mode`` argument (whose
    default this package resolves to its fastest form, the deterministic one to its automatic form:
    :meth:`ArmCake._memory_form`)."""

    name = "cake_nd"
    deterministic = False


class ArmCakeNondetLogprob(ArmCakeLogprob):
    """:class:`ArmCakeLogprob` through ``cake_lm_head_nondet_train_logprob``: the loss rows' objective in eager
    PyTorch on the non-deterministic log-probabilities."""

    name = "cake_nd_logprob"
    deterministic = False
    row_arm = "cake_nd"


def _temperature_form(temperature):
    """A JSON description of a temperature argument: ``None``, ``{"scalar": value}`` or ``{"tensor": [min, max,
    rows at 1.0]}``."""
    if temperature is None:
        return None
    if isinstance(temperature, torch.Tensor):
        t = temperature.detach().float()
        return dict(
            tensor=[float(t.min()), float(t.max()), int((t == 1.0).sum())]
            if t.numel()
            else []
        )
    return dict(scalar=float(temperature))


ARMS = {
    cls.name: cls
    for cls in (
        ArmTorchUnchunked,
        ArmTorchChunked,
        ArmLiger,
        ArmLigerDefault,
        ArmCCE,
        ArmCCEDefault,
        ArmCake,
        ArmCakeLogprob,
        ArmCakeNondet,
        ArmCakeNondetLogprob,
    )
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _head_bytes(inp):
    """``B = 2 * T_v * V + 2 * V * H + 2 * T * H``: the bytes of a head that materializes the BF16 logits of the
    ``T_v`` valid rows, with the BF16 ``dW`` and ``dX``."""
    T, Hx = int(inp.X.shape[0]), int(inp.X.shape[1])
    Vw = int(inp.W.shape[0])
    valid = int(inp.valid.sum())
    return 2 * valid * Vw + 2 * Vw * Hx + 2 * T * Hx


def _step_peak(arm):
    """The peak allocation above the live tensors of one training step whose backward frees the graph (forward
    through backward, outputs and gradients included); ``None`` for an arm without such a backward."""
    free = getattr(arm, "free_backward", None)
    if free is None:
        return None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    st = arm.forward()
    grads = free(st)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del grads, st
    return peak


def _ckpt_memory(arm):
    """``(saved, peak)`` of the checkpointed step (:meth:`ArmAutograd.ckpt_step`): the bytes its checkpointed forward
    leaves alive for the backward, and the peak allocation above the live tensors from that forward through its
    backward (outputs and gradients included)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    st = arm.ckpt_forward()
    torch.cuda.synchronize()
    saved = torch.cuda.memory_allocated() - base
    grads = arm.free_backward(st)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del grads, st
    return saved, peak


# the backend modules whose remembered bindings :func:`_clear_binding_caches` drops: each package's default and
# opt-in module (the non-deterministic package has no cache-clear hook of its own; its two ``BINDING_CACHE`` objects
# are the deterministic package's kind and are cleared the same way)
_BINDING_CACHE_MODULES = tuple(
    f"cake_pplx._backends.cake_lm_head_{mode}_train.{module}"
    for mode in ("det", "nondet")
    for module in ("cake_backend", "cake_backend_opt_in")
)


def _clear_binding_caches():
    """Drop the remembered bindings of both Cake packages' backend modules (each once it is imported): the next
    ``cake`` / ``cake_nd`` call misses the binding cache, as a training step with a new valid-row count does."""
    for name in _BINDING_CACHE_MODULES:
        module = sys.modules.get(name)
        if module is not None:
            module.BINDING_CACHE.clear()


def measure_perf(arm, steps):
    """Median forward / backward / step milliseconds (``step_ms`` from the remembered bindings, ``step_miss_ms`` with
    the binding caches cleared before every step: the miss a new valid-row count takes, usually every training
    batch; a step's backward frees the graph where the arm has :meth:`ArmAutograd.free_backward`, as a training step
    does, so the entries that release their saved state early do), the peak allocation above the live tensors during
    the forward and during the backward (the timed backward, which retains the graph), the bytes the forward leaves
    alive for the backward, and the step peak of a
    step whose backward frees the graph (:func:`_step_peak`) from the remembered bindings (``step_peak_hit_*``) and
    after the caches were cleared (``step_peak_miss_*``), each with its ratio to ``B`` (:func:`_head_bytes`);
    ``step_peak_gib`` / ``step_peak_over_b``: the larger of the two.  ``ckpt_ms``: the checkpointed step
    (:meth:`ArmAutograd.ckpt_step`: the checkpointed forward + the step's backward, which recomputes it) timed like
    ``step`` (the forward state of the timed backward still alive, the bindings remembered); ``None`` with the arm's
    ``ckpt_unsupported`` reason when it has none, or with ``ckpt_error`` when it raised.  ``ckpt_saved_gib`` /
    ``ckpt_peak_gib`` / ``ckpt_peak_over_b``: :func:`_ckpt_memory` after the remembered-binding step peak and before
    the caches are cleared for the other one (the checkpointed step's bindings remembered by the timed ``ckpt``)."""
    saved = [
        arm.forward()
    ]  # the forward state the timed backward consumes; released before the memory probes
    torch.cuda.synchronize()
    fwd_ms = median_ms(arm.forward, steps)
    bwd_ms = median_ms(lambda: arm.backward(saved[0]), steps)

    step_backward = getattr(arm, "free_backward", arm.backward)

    def full():
        s = arm.forward()
        step_backward(s)

    step_ms = median_ms(full, steps)

    def full_miss():
        _clear_binding_caches()
        full()

    step_miss_ms = median_ms(full_miss, steps)
    ckpt = getattr(arm, "ckpt_step", None)
    ckpt_ms = ckpt_error = None
    ckpt_unsupported = (
        getattr(arm, "CKPT_UNSUPPORTED", f"{arm.name}: no checkpointed step")
        if ckpt is None
        else None
    )
    if ckpt is not None:
        try:
            ckpt_ms = median_ms(ckpt, steps)
        except Exception as exc:  # reported per op; the arm's other ops stand
            ckpt_error = f"{type(exc).__name__}: {exc}"
    saved.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    st = arm.forward()
    torch.cuda.synchronize()
    fwd_peak = torch.cuda.max_memory_allocated() - base
    saved = torch.cuda.memory_allocated() - base
    torch.cuda.reset_peak_memory_stats()
    base_bwd = torch.cuda.memory_allocated()
    grads = arm.backward(st)
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated() - base_bwd
    del grads, st
    step_peak_hit = _step_peak(arm)
    # the checkpointed step's memory from the remembered bindings: before the caches are cleared (its loss entry's
    # checkpointed forward has a binding of its own, which the step after the clear would not form again)
    ckpt_saved = ckpt_peak = None
    if ckpt is not None and ckpt_error is None:
        try:
            ckpt_saved, ckpt_peak = _ckpt_memory(arm)
        except Exception as exc:  # reported per op; the arm's other ops stand
            ckpt_error = f"memory probe: {type(exc).__name__}: {exc}"
    _clear_binding_caches()
    step_peak_miss = _step_peak(arm)
    step_peak = None if step_peak_hit is None else max(step_peak_hit, step_peak_miss)
    head = _head_bytes(arm.inp)

    def over_b(nbytes):
        return None if nbytes is None else nbytes / head

    def as_gib(nbytes):
        return None if nbytes is None else gib(nbytes)

    return dict(
        fwd_ms=fwd_ms,
        bwd_ms=bwd_ms,
        step_ms=step_ms,
        step_miss_ms=step_miss_ms,
        ckpt_ms=ckpt_ms,
        ckpt_unsupported=ckpt_unsupported,
        ckpt_error=ckpt_error,
        fwd_peak_gib=gib(fwd_peak),
        bwd_peak_gib=gib(bwd_peak),
        saved_gib=gib(saved),
        step_peak_gib=as_gib(step_peak),
        step_peak_hit_gib=as_gib(step_peak_hit),
        step_peak_miss_gib=as_gib(step_peak_miss),
        head_gib=gib(head),
        step_peak_over_b=over_b(step_peak),
        step_peak_hit_over_b=over_b(step_peak_hit),
        step_peak_miss_over_b=over_b(step_peak_miss),
        ckpt_saved_gib=as_gib(ckpt_saved),
        ckpt_peak_gib=as_gib(ckpt_peak),
        ckpt_peak_over_b=over_b(ckpt_peak),
    )


def _passes(ratios, b0_errors):
    return all(r <= 1.05 or b0_errors.get(k, 0) == 0 for k, r in ratios.items())


def _ckpt_accuracy(arm, grads, oracle, labels, b0_errors, informational):
    """The error report of the checkpointed step's ``dX`` / ``dW`` (:meth:`ArmAutograd.ckpt_step`) against the FP64
    oracle, its ratios to the unchunked PyTorch path, ``passes_1p05x`` (``None`` for an informational arm) and
    ``same_bits_as_bwd`` (both gradients bitwise equal to ``grads``, the arm's ``bwd`` gradients); ``unsupported``
    with the reason for an arm without the step, ``error`` when it raised."""
    ckpt = getattr(arm, "ckpt_step", None)
    if ckpt is None:
        return dict(
            unsupported=getattr(arm, "CKPT_UNSUPPORTED", f"{arm.name}: no checkpointed step")
        )
    try:
        got = ckpt()
        torch.cuda.synchronize()
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {exc}")
    valid = labels >= 0
    errors = {
        f"{name}_rel_l2": rel_l2(got[name], oracle[name])
        for name in ("dX", "dW")
        if oracle.get(name) is not None
    }
    errors["dX_ignored_zero"] = bool((got["dX"].detach()[~valid] == 0).all())
    errors["nan"] = not all(bool(torch.isfinite(got[name]).all()) for name in ("dX", "dW"))
    ratios = gate_ratios(errors, b0_errors)
    return dict(
        errors=errors,
        vs_unchunked=ratios,
        passes_1p05x=None if informational else _passes(ratios, b0_errors),
        same_bits_as_bwd=all(torch.equal(got[name], grads[name]) for name in ("dX", "dW")),
    )


def measure_accuracy(arm, inp, oracle, b0_errors):
    """The error report of the arm's ``fwd`` / ``bwd`` outputs (``errors``, ``vs_unchunked``, ``passes_1p05x``) and
    of its checkpointed step's gradients (``ckpt``, :func:`_ckpt_accuracy`; run after the ``bwd`` graph is freed)."""
    st = arm.forward()
    grads = arm.backward(st)
    torch.cuda.synchronize()
    outs = arm.outputs(st)
    del st  # the retained graph: freed before the checkpointed step
    result = dict(loss=outs["loss"], logp=outs["logp"], dX=grads["dX"], dW=grads["dW"])
    if (
        result["logp"] is None
    ):  # arms without per-token output: the row metrics fall away
        result["logp"] = oracle["logp"].float()
    errors = error_report(result, oracle, inp.labels)
    if outs["logp"] is None:
        for key in ("logp_max_abs", "logp_mean_abs", "logp_ignored_zero"):
            errors[key] = None
    ratios = gate_ratios({k: v for k, v in errors.items() if v is not None}, b0_errors)
    informational = bool(getattr(arm, "informational", False))
    passes = None if informational else _passes(ratios, b0_errors)
    return dict(
        errors=errors,
        vs_unchunked=ratios,
        passes_1p05x=passes,
        informational=informational,
        ckpt=_ckpt_accuracy(arm, grads, oracle, inp.labels, b0_errors, informational),
    )


def _arm_inputs(cls, inp, entry):
    """The inputs an arm runs: the row's (with its temperature) when the arm applies the temperature on this entry
    or the row has none; the same inputs without the temperature for the other loss-row arms (the loss entry takes
    none); ``None`` for a log-probability row with a temperature the arm has no form of."""
    if inp.temperature is None or entry in cls.TEMPERATURE_ENTRIES:
        return inp
    if entry == "loss":
        return replace(inp, temperature=None)
    return None


def _arm_or_error(cls, inp, entry, chunk):
    arm_inp = _arm_inputs(cls, inp, entry)
    if arm_inp is None:
        return None, f"NotImplementedError: {cls.name} has no temperature form"
    try:
        return cls(arm_inp, entry=entry, chunk=chunk), None
    except (
        Exception
    ) as exc:  # optional dependency missing or arm unavailable on this device
        return None, f"{type(exc).__name__}: {exc}"


def _row_inputs(row, args, W):
    T, C, objective, entry = ROWS[row]
    ignore = {} if args.ignore_frac is None else dict(ignore_frac=args.ignore_frac)
    return (
        make_inputs(
            T,
            objective=objective,
            seed=SEED_BASE + T,
            device=args.device,
            W=W,
            temperature=args.temperature,
            temperature_value=args.temperature_value,
            **ignore,
        ),
        C,
        entry,
    )


def run_perf(args, results, W):
    fmt = lambda v: "   n/a" if v is None else f"{v:8.3f}"
    for row in args.rows:
        inp, C, entry = _row_inputs(row, args, W)
        record = dict(
            row=row,
            T=inp.T,
            chunk=C,
            objective=inp.objective,
            entry=entry,
            inputs_gib=gib(inp.bytes_inputs()),
            temperature=_temperature_form(inp.temperature),
            arms={},
        )
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp, entry, C)
            if arm is None:
                record["arms"][name] = dict(unsupported=error)
                print(f"{row:22s} {name:16s} unsupported: {error}", flush=True)
                continue
            try:
                perf = measure_perf(arm, max(args.min_steps, args.steps))
                perf["versions"] = arm.versions()
                perf["temperature_applied"] = arm.inp.temperature is not None
            except Exception as exc:
                perf = dict(
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            record["arms"][name] = perf
            if "error" in perf:
                print(f"{row:22s} {name:16s} failed: {perf['error']}", flush=True)
            else:
                step_peak = (
                    "   n/a"
                    if perf["step_peak_gib"] is None
                    else f"hit {perf['step_peak_hit_gib']:6.3f} GiB ({perf['step_peak_hit_over_b']:.4f} B) "
                    f"miss {perf['step_peak_miss_gib']:6.3f} GiB ({perf['step_peak_miss_over_b']:.4f} B)"
                )
                print(
                    f"{row:22s} {name:16s} fwd {fmt(perf['fwd_ms'])} ms  bwd {fmt(perf['bwd_ms'])} ms  step {fmt(perf['step_ms'])} ms  "
                    f"step miss {fmt(perf['step_miss_ms'])} ms  ckpt {fmt(perf['ckpt_ms'])} ms  "
                    f"peak fwd {perf['fwd_peak_gib']:6.2f} GiB  bwd {perf['bwd_peak_gib']:6.2f} GiB  saved {perf['saved_gib']:6.2f} GiB  "
                    f"step peak {step_peak}",
                    flush=True,
                )
                if perf["ckpt_peak_gib"] is not None:
                    print(
                        f"{row:22s} {name:16s} ckpt saved {perf['ckpt_saved_gib']:6.3f} GiB  "
                        f"ckpt peak {perf['ckpt_peak_gib']:6.3f} GiB ({perf['ckpt_peak_over_b']:.4f} B)",
                        flush=True,
                    )
                if perf["ckpt_unsupported"] is not None:
                    print(
                        f"{row:22s} {name:16s} ckpt unsupported: {perf['ckpt_unsupported']}",
                        flush=True,
                    )
                if perf["ckpt_error"] is not None:
                    print(
                        f"{row:22s} {name:16s} ckpt failed: {perf['ckpt_error']}",
                        flush=True,
                    )
            del arm
            torch.cuda.empty_cache()
        results["rows"].append(record)
        del inp
        torch.cuda.empty_cache()


def run_accuracy(args, results, W):
    for row in args.rows:
        inp, C, entry = _row_inputs(row, args, W)
        # the oracle of the row's objective: with the temperature on the log-probability rows; the loss rows' oracle
        # has none (the loss entry takes none), so an arm that applies a temperature there has no oracle
        ref_inp = inp if entry == "logprob" else replace(inp, temperature=None)
        oracle = reference_fp64(ref_inp, entry=entry)
        b0 = reference_unchunked(ref_inp, entry=entry)
        b0_errors = error_report(b0, oracle, inp.labels)
        del b0
        torch.cuda.empty_cache()
        record = dict(
            row=row,
            T=inp.T,
            chunk=C,
            objective=inp.objective,
            entry=entry,
            temperature=_temperature_form(inp.temperature),
            unchunked_errors=b0_errors,
            arms={},
        )
        print(
            f"{row:22s} {'unchunked (B0)':16s} {json.dumps(b0_errors, default=str)}",
            flush=True,
        )
        for name in args.arms:
            arm, error = _arm_or_error(ARMS[name], inp, entry, C)
            if arm is None:
                record["arms"][name] = dict(unsupported=error)
                print(f"{row:22s} {name:16s} unsupported: {error}", flush=True)
                continue
            if entry == "loss" and arm.inp.temperature is not None:
                record["arms"][name] = dict(
                    timing_only="no oracle of the loss-row objective with a temperature"
                )
                print(f"{row:22s} {name:16s} timing only (temperature)", flush=True)
                del arm
                continue
            try:
                record["arms"][name] = measure_accuracy(arm, inp, oracle, b0_errors)
            except Exception as exc:
                record["arms"][name] = dict(
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
            print(
                f"{row:22s} {name:16s} {json.dumps(record['arms'][name], default=str)}",
                flush=True,
            )
            del arm
            torch.cuda.empty_cache()
        results["accuracy"].append(record)
        del inp, oracle
        torch.cuda.empty_cache()


def _nondet_program_available(device):
    """``{loss, logprob}`` program availability of the non-deterministic package on ``device`` (the import error
    when it does not import)."""
    try:
        backend = _cake_api(False)[0]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return dict(
        loss=backend.generated_program_available(device, entry="loss"),
        logprob=backend.generated_program_available(device, entry="logprob"),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    parser.add_argument(
        "--arms",
        default="torch_unchunked,torch_chunked,liger,cce,cake",
        help=f"comma-separated subset of {sorted(ARMS)}",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        choices=TEMPERATURE_FORMS,
        help="temperature form of the log-probability calls (default: no temperature)",
    )
    parser.add_argument(
        "--temperature-value",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="value of the scalar / constant temperature forms",
    )
    parser.add_argument(
        "--memory-mode",
        default="default",
        choices=("default", "fast", "budget"),
        help="memory_mode of the cake arms (default: none passed; the deterministic entries take their automatic "
        "form, the non-deterministic entries their fast form)",
    )
    parser.add_argument(
        "--ignore-frac",
        type=float,
        default=None,
        help="fraction of ignored rows of every row's inputs (default: the inputs' 0.05)",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--accuracy", action="store_true")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.ignore_frac is not None and not 0 <= args.ignore_frac <= 1:
        parser.error("--ignore-frac must be between 0 and 1")
    ArmCake.memory_mode = None if args.memory_mode == "default" else args.memory_mode
    args.arms = [a for a in args.arms.split(",") if a]
    unknown = sorted(set(args.arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arms {unknown}; choose from {sorted(ARMS)}")
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    args.device = device
    env = dict(
        torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        triton=_probe(_triton_version),
        cublas=_cublas_version(),
        liger_kernel=_version("liger-kernel"),
        cut_cross_entropy=_version("cut-cross-entropy"),
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
    )
    results = dict(
        env=env,
        device=env["device"],
        capability=env["capability"],
        torch=torch.__version__,
        geometry=dict(H=H, V=V, loss_div=LOSS_DIV, seed_base=SEED_BASE),
        temperature=dict(form=args.temperature, value=args.temperature_value),
        memory_mode=args.memory_mode,
        ignore_frac=args.ignore_frac,
        program_available=dict(
            loss=cake_backend.generated_program_available(device, entry="loss"),
            logprob=cake_backend.generated_program_available(device, entry="logprob"),
        ),
        program_available_nondet=_nondet_program_available(device),
        rows=[],
        accuracy=[],
    )
    W = make_weight(
        V, H, seed=SEED_BASE + 1000003, device=device
    )  # shared by every row of the process
    if not args.no_perf:
        run_perf(args, results, W)
    if args.accuracy:
        run_accuracy(args, results, W)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
