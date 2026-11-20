"""Benchmark the deterministic Cake DSA sparse-attention training kernels (SM100 / SM103 / SM107).

Times the forward, the backward and the whole training step of
``cake_pplx._backends.cake_dsa_sparse_attention_det_train`` on the rows, inputs and layouts of
``benchmarks/bench_cake_dsa_sparse_attention_nondet_train.py`` (the same row names: one causal document,
context-parallel tails, the packed GLM-5.2 trainer batches with their destination maps and 4096-query backward
chunks, and the ``nope_*`` GLM-5.3-Flash rows), next to the arms of that script: the non-deterministic package
(``nondet`` / ``nondet_nope``) and the external baselines it wraps (``flashmla_cudnn``, ``fa4``, ``fa4_nope``,
``fa4_nope_a2``, ``flashmla_cudnn_nope``) when importable.

Arms ``det`` (RoPE rows) and ``det_nope`` (NoPE rows).  Default layout: the prepared runner
(``prepare_dsa_train``; the ``det_scratch`` arm prepares it with ``o_lo_scratch=True``, the integer accumulator
region in the storage of ``o_lo``).  Trainer layout (the GLM-5.2 rows): the public varlen entry
``cake_dsa_sparse_attention_det_varlen(..., causal=True, dkv_acc=, dkv_dst_map=)`` through autograd, fed the
layout views as they are, with the caller's fp32 ``dkv_acc`` allocated inside the timed forward.  The NoPE arm calls
``cake_dsa_sparse_attention_det_nope_varlen`` through autograd into an fp32 ``dkv_acc [S, 512]`` zero-filled inside
the timed backward, once per step.  The arms ``det_nope_p2`` / ``det_nope_p4`` call that entry with
``key_passes=2`` / ``4`` (the backward's main stage as that many key-range passes; the NoPE policy alone keeps one
pass); run them on the long-key rows, e.g. ``nope_chunk_4096x269564``, ``nope_tail_4096x131072``, ``p8949_nope``.

Columns.  ``fwd_ms``: the training forward.  ``step_ms``: forward + backward.  ``bwd_ms``: ``step_ms - fwd_ms``
(every arm, the same derivation).  ``bwd_again_ms``: a backward over an already-consumed forward (the retained graph
of one forward); the deterministic autograd entries run their backward in the storage of ``o_lo`` and so replay the
forward into the saved tensors first, which this column includes -- the other arms (and a runner prepared without
``--o-lo-scratch``) do not.  ``recompute_step_ms``: two forwards + backward (activation checkpointing).
``peak_above_inputs_gib``: the peak of one step above the memory allocated before the step (a prepared runner's
workspace and outputs excluded).  ``peak_total_gib``: the peak of one step above the memory allocated before the
arm was built (the runner's workspace and outputs included; the column the arms are compared by).
``repeat_bitwise`` (deterministic arms): two further steps give the same bits for ``out``, ``lse`` and every
gradient.

Rows ``p8949_rope`` / ``p8949_nope``: the long-context repro problem (T 16384, S 65536, top-k 2048: key 0,
one key from each of 1023 strata of the older prefix, the 1024 most recent keys; generator seed 0, N(0, 1) BF16)
in the default layout (RoPE) and through the NoPE arms.

Timing: the non-deterministic script's ``median_ms`` (CUPTI activity tracing through ``torch.profiler``, a cold L2
between iterations), medians over ``--steps`` iterations.

Sink rows (``--sink``, in place of the ``--rows`` rows).  The trainable per-head sink on the rows of the
non-deterministic script's ``--sink`` mode: ``T`` 16,384 at index widths 640 and 128 (``sink_w640_t16384``,
``sink_w128_t16384``: a pack with one-token documents, global index rows with ``-1`` holes and a row without a valid
slot) and the packed rows ``sink_varlen_w640_t16384`` / ``sink_varlen_w128_t16384`` (16 documents of 1,024 tokens, the
per-document layout), scale ``512 ** -0.5``, the same inputs and sink vector.  Every arm runs with the row's sink and
without it (``sink=None``; ``--sink-variants`` keeps one of them) under :func:`measure` (the columns above, the
workspace and, for the deterministic arms, ``repeat_bitwise``).  Arms: ``cake_det_nope_sink`` (the flat entry
``cake_dsa_sparse_attention_det_nope`` through autograd: the fp32 ``dkv_acc [S, 512]`` allocated in the forward and
zero-filled inside the backward, ``dq`` and, with the sink, ``dsink`` from one ``autograd.grad``; on a packed row it
takes the global rows), ``cake_det_nope_varlen_sink`` (the packed entry ``cake_dsa_sparse_attention_det_nope_varlen``
with the per-document rows and ``cu_seqlens_q`` / ``cu_seqlens_k``, ``causal=False``; packed rows only),
``cudnn_fe_nope_sink_det`` (the cuDNN frontend's sparse attention with ``attn_sink`` and its deterministic backward,
``deterministic=True``; the non-deterministic script's ``cudnn_fe_nope_sink`` otherwise), and the non-deterministic
script's ``cake_nope_sink``, ``cake_nope_varlen_sink`` and ``cudnn_fe_nope_sink`` under the same protocol.

Usage::

    python benchmarks/bench_cake_dsa_sparse_attention_det_train.py --rows doc_4096 packed_glm_a_dstmap \\
        --arms det,nondet --json det.json
    python benchmarks/bench_cake_dsa_sparse_attention_det_train.py --rows chunk_4096x268757 tail_2123x67923 \\
        nope_glm53_a --arms det,nondet,det_nope,nondet_nope,fa4,fa4_nope
    python benchmarks/bench_cake_dsa_sparse_attention_det_train.py --rows nope_chunk_4096x269564 \\
        nope_tail_4096x131072 p8949_nope --arms det_nope,det_nope_p2,det_nope_p4
    python benchmarks/bench_cake_dsa_sparse_attention_det_train.py --sink --sink-rows sink_w640_t16384 \\
        sink_varlen_w640_t16384 --sink-arms cake_det_nope_sink,cake_det_nope_varlen_sink,cudnn_fe_nope_sink_det \\
        --json det-sink.json
"""

import argparse
import dataclasses
import functools
import json
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_cake_dsa_sparse_attention_nondet_train as nb  # noqa: E402

from cake_pplx.cake_dsa_sparse_attention_det_train import (  # noqa: E402
    cake_dsa_sparse_attention_det_nope,
    cake_dsa_sparse_attention_det_nope_varlen,
    cake_dsa_sparse_attention_det_varlen,
)
from cake_pplx._backends.cake_dsa_sparse_attention_det_train import (  # noqa: E402
    cake_backend,
    cake_jit,
)
from tests.experimental.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (  # noqa: E402
    D_LATENT,
    D_ROPE,
    DEFAULT_NOPE_SCALE,
    DEFAULT_SCALE,
    SINK_SCALE,
)

DEFAULT_ROWS = (
    "doc_4096",
    "doc_16384",
    "doc_16231",
    "tail_2123x67923",
    "packed_glm_a",
    "packed_glm_a_dstmap",
    "chunk_4096x268757",
    "p8949_rope",
    "nope_doc_4096",
    "nope_glm53_a",
    "nope_chunk_4096x269564",
    "p8949_nope",
)
# The long-context repro rows: name -> NoPE.
P8949_ROWS = {"p8949_rope": False, "p8949_nope": True}


def _p8949_indices(t, s, device, gen):
    """The long-context repro's index rows: key 0, one key from each of 1023 strata of the older prefix, the 1024 most
    recent positions; distinct and causal (query ``i`` sits at position ``s - t + i``)."""
    pos = torch.arange(s - t, s, device=device)
    recent = pos[:, None] - torch.arange(1024, device=device)[None, :]
    older = (pos - 1024).double()
    u = torch.rand((t, 1023), device=device, dtype=torch.float64, generator=gen)
    strata = (
        1
        + (
            (torch.arange(1023, device=device)[None, :] + u)
            * (older[:, None] - 1)
            / 1023
        ).long()
    )
    sink = torch.zeros((t, 1), device=device, dtype=torch.long)
    return torch.cat([sink, strata, recent.flip(1)], 1).int().contiguous()


def make_p8949_inputs(nope, device, t=16384, s=65536):
    """The long-context repro's tensors (generator seed 0, N(0, 1) BF16 in the repro's draw order) in the input types of
    the non-deterministic script (every slot valid: ``topk_length = 2048``)."""
    device = torch.device(device)
    gen = torch.Generator(device=device).manual_seed(0)

    def randn(*shape):
        return torch.randn(shape, device=device, generator=gen).bfloat16()

    heads = cake_backend.NUM_HEADS
    q_latent, kv_latent = randn(t, heads, D_LATENT), randn(s, D_LATENT)
    dout = randn(t, heads, D_LATENT)
    lengths = torch.full((t,), 2048, dtype=torch.int32, device=device)
    if nope:
        idx = _p8949_indices(t, s, device, gen)
        base = nb.make_nope_inputs(
            [t], [s], seed=nb.SEED, topk=2048, index="iid", device=device
        )
        return dataclasses.replace(
            base, q=q_latent, kv=kv_latent, dout=dout, idx_local=idx,
            idx_global=idx, topk_length=lengths, index="p8949_repro",
        )  # fmt: skip
    q_rope, k_rope = randn(t, heads, D_ROPE), randn(s, D_ROPE)
    idx = _p8949_indices(t, s, device, gen)
    base = nb.make_inputs([t], [s], seed=nb.SEED, topk=2048, device=device)
    return dataclasses.replace(
        base, q_latent=q_latent, q_rope=q_rope, kv_latent=kv_latent, k_rope=k_rope,
        dout=dout, idx_local=idx, idx_global=idx, topk_length=lengths,
    )  # fmt: skip


class ArmDet(nb.ArmCake):
    """The deterministic package on a RoPE row (the runner in the default layout, with the row's segment count as
    :class:`nb.ArmCake` passes it; the public varlen entry through autograd in the trainer layout); the backward of
    :class:`nb.ArmCake` serves both."""

    name = "det"
    deterministic = True
    o_lo_scratch = False

    def __init__(self, inp, layout):
        self.inp, self.layout = inp, layout
        self.backward_available = cake_backend.generated_program_available(
            inp.q_latent.device, backward=True
        )
        self.runner = None
        if not layout.trainer:
            self.num_segments = int(inp.cu_seqlens_k.numel()) - 1
            self.runner = cake_backend.prepare_dsa_train(
                inp.q_latent,
                inp.q_rope,
                inp.kv_latent,
                inp.k_rope,
                inp.idx_global,
                topk_length=inp.topk_length,
                dout=inp.dout if self.backward_available else None,
                softmax_scale=DEFAULT_SCALE,
                backward=self.backward_available,
                o_lo_scratch=self.o_lo_scratch,
                num_segments=self.num_segments,
            )
            return
        self.q_latent = layout.q_latent.detach().requires_grad_()
        if layout.q256 is not None:
            self.q_buf = layout.q256.detach().requires_grad_()
            self.q_rope = self.q_buf[:, :, 256 - D_ROPE :]
        else:
            self.q_buf = layout.q_rope.detach().requires_grad_()
            self.q_rope = self.q_buf
        if layout.dkv_acc:
            self.kv_leaves = ()
            self.kv_latent, self.k_rope = layout.kv_latent, layout.k_rope
        elif layout.kv_buf is not None:
            kv_buf = layout.kv_buf.detach().requires_grad_()
            self.kv_leaves = (kv_buf,)
            self.kv_latent, self.k_rope = kv_buf[:, :D_LATENT], kv_buf[:, D_LATENT:]
        else:
            self.kv_leaves = (
                layout.kv_latent.detach().requires_grad_(),
                layout.k_rope.detach().requires_grad_(),
            )
            self.kv_latent, self.k_rope = self.kv_leaves

    def versions(self):
        if self.runner is not None:
            return dict(
                module=self.runner.module_name,
                stages=list(self.runner.stages),
                entry="prepare_dsa_train",
                storage=self.runner.storage,
                num_segments=self.num_segments,
                key_passes=self.runner.key_passes,
                backward_launches=len(self.runner.backward_order),
            )
        module_name, _record = cake_backend.record_for(self.inp.q_latent.device)
        return dict(
            module=module_name,
            stages=list(cake_jit.registered_stages()),
            entry="cake_dsa_sparse_attention_det_varlen",
            layout=self.layout.describe(),
        )

    def forward(self):
        if self.runner is not None:
            out, lse, _ = self.runner.forward()
            return dict(out=out, lse=lse)
        inp, layout = self.inp, self.layout
        dkv_acc = layout.new_dkv_acc() if layout.dkv_acc else None
        out, lse = cake_dsa_sparse_attention_det_varlen(
            self.q_latent,
            self.q_rope,
            self.kv_latent,
            self.k_rope,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=True,
            softmax_scale=DEFAULT_SCALE,
            return_lse=True,
            dkv_acc=dkv_acc,
            dkv_dst_map=layout.dst_map,
        )
        return dict(out=out, lse=lse, dkv_acc=dkv_acc)


class ArmDetNope:
    """The deterministic package's native zero-RoPE varlen entry through autograd (the protocol of the
    non-deterministic script's ``cake_nope`` arm: fp32 ``dkv_acc [S, 512]`` zero-filled inside the backward), with
    the entry's ``key_passes`` (``None``: the NoPE key-pass policy)."""

    name = "det_nope"
    deterministic = True
    key_passes = None

    def __init__(self, inp):
        program = cake_jit.NOPE_PROGRAM
        self.backward_available = cake_backend.generated_program_available(
            inp.q.device, backward=True, program=program
        )
        if not cake_backend.generated_program_available(inp.q.device, program=program):
            raise RuntimeError(
                "skipped: the NoPE program is not registered for this device"
            )
        self.program, self.inp = program, inp
        self.q = inp.q.detach().requires_grad_()

    def versions(self):
        return dict(
            module=self.program,
            entry="cake_dsa_sparse_attention_det_nope_varlen",
            key_passes=self.key_passes,
        )

    def forward(self):
        inp = self.inp
        dkv_acc = torch.empty(
            inp.total_k, D_LATENT, dtype=torch.float32, device=self.q.device
        )
        out, lse = cake_dsa_sparse_attention_det_nope_varlen(
            self.q,
            inp.kv,
            inp.idx_local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=True,
            softmax_scale=DEFAULT_NOPE_SCALE,
            return_lse=True,
            key_passes=self.key_passes,
            dkv_acc=dkv_acc,
        )
        return dict(out=out, lse=lse, dkv_acc=dkv_acc)

    def backward(self, st):
        st["dkv_acc"].zero_()
        (dq,) = torch.autograd.grad(
            st["out"], (self.q,), self.inp.dout, retain_graph=True
        )
        return dict(dq=dq, dkv=st["dkv_acc"])


class ArmDetNopeP2(ArmDetNope):
    name = "det_nope_p2"
    key_passes = 2


class ArmDetNopeP4(ArmDetNope):
    name = "det_nope_p4"
    key_passes = 4


class ArmDetScratch(ArmDet):
    name = "det_scratch"
    o_lo_scratch = True


class ArmDetNopeSink(nb.ArmCakeNopeSink):
    """The deterministic package's flat NoPE entry ``cake_dsa_sparse_attention_det_nope`` through autograd on a sink row
    (the protocol of the non-deterministic script's ``cake_nope_sink`` arm, whose forward and backward it runs: scale
    ``512 ** -0.5``, the caller's index rows as given, the fp32 ``dkv_acc [S, 512]`` allocated in the forward and
    zero-filled inside the backward, ``dq`` and, with the sink, ``dsink`` from one ``autograd.grad``), with the row's
    sink (``variant="sink"``) or ``sink=None``."""

    name = "cake_det_nope_sink"
    deterministic = True
    backward_available = True

    def __init__(self, inp, variant):
        program = cake_jit.NOPE_PROGRAM
        device = inp.q.device
        if not cake_backend.generated_program_available(
            device, backward=True, program=program
        ):
            raise RuntimeError(
                "skipped: the NoPE program is not registered for this device"
            )
        if variant == "sink":
            stage = getattr(cake_jit, "NOPE_SINK_FORWARD_STAGE", None)
            arch = cake_backend.arch_for(device)
            # the stage that serves the sink forward on this architecture (its scanner-warp build on sm_107a)
            served = stage and cake_jit.stage_for(stage, arch, program)
            if served is None or arch not in cake_jit.stage_arches(served, program):
                raise RuntimeError(f"skipped: no sink build for {arch}")
        self.fn = cake_dsa_sparse_attention_det_nope
        self.inp, self.variant = inp, variant
        self.width = int(inp.indices.shape[1])
        self.q = inp.q.detach().requires_grad_()
        self.sink = (
            inp.sink.detach().clone().requires_grad_() if variant == "sink" else None
        )

    def versions(self):
        mode = getattr(cake_backend, "nope_flat_index_mode", None)
        return dict(
            module=cake_jit.NOPE_PROGRAM,
            entry="cake_dsa_sparse_attention_det_nope",
            sink=self.sink is not None,
            index_route=mode(self.inp.total_q, self.width) if mode else None,
        )

    def _workspace(self, num_segments):
        inp = self.inp
        return int(
            cake_backend.dsa_nope_train_workspace_size(
                inp.total_q,
                inp.total_k,
                self.width,
                inp.q.device,
                backward=True,
                num_segments=num_segments,
                dkv_acc=True,
                **({"sink": True} if self.sink is not None else {}),
            )
        )

    def workspace(self):
        return self._workspace(1)


class ArmDetNopeVarlenSink(ArmDetNopeSink):
    """The deterministic package's packed NoPE entry ``cake_dsa_sparse_attention_det_nope_varlen`` through autograd on a
    packed sink row (``nb.SINK_VARLEN_ROWS``): the per-document rows with ``cu_seqlens_q`` / ``cu_seqlens_k`` as given
    (``causal=False``: the rows are causal by construction; the entry offsets them on device inside the timed forward),
    otherwise as :class:`ArmDetNopeSink`."""

    name = "cake_det_nope_varlen_sink"

    def __init__(self, inp, variant):
        if not isinstance(inp, nb.SinkVarlenRowInputs):
            raise RuntimeError(
                "skipped: the packed entry runs on the packed rows (sink_varlen_*)"
            )
        super().__init__(inp, variant)
        self.fn = cake_dsa_sparse_attention_det_nope_varlen

    def versions(self):
        inp = self.inp
        mode = getattr(cake_backend, "varlen_index_mode", None)
        return dict(
            module=cake_jit.NOPE_PROGRAM,
            entry="cake_dsa_sparse_attention_det_nope_varlen",
            sink=self.sink is not None,
            index_route=mode(inp.local) if mode else None,
            num_docs=inp.num_segments,
            causal=False,
        )

    def _call(self, dkv_acc):
        inp = self.inp
        return self.fn(
            self.q,
            inp.kv,
            inp.local,
            inp.cu_seqlens_q,
            inp.cu_seqlens_k,
            inp.max_seqlen_q,
            inp.max_seqlen_k,
            causal=False,
            softmax_scale=SINK_SCALE,
            sink=self.sink,
            return_lse=True,
            dkv_acc=dkv_acc,
        )

    def workspace(self):
        return self._workspace(self.inp.num_segments)


class ArmCudnnFeNopeSinkDet(nb.ArmCudnnFeNopeSink):
    """The non-deterministic script's ``cudnn_fe_nope_sink`` arm with the frontend's deterministic backward
    (``deterministic=True``; the same forward call): the external deterministic baseline where the frontend serves it
    (an architecture or head count it refuses raises at the first backward, recorded as the arm's error)."""

    name = "cudnn_fe_nope_sink_det"
    deterministic = True
    backward_available = True
    backward_options = {"deterministic": True}

    def versions(self):
        return dict(super().versions(), backward="deterministic=True")


ROPE_ARMS = {
    ArmDet.name: ArmDet,
    ArmDetScratch.name: ArmDetScratch,
    "nondet": nb.ArmCake,
    "flashmla_cudnn": nb.ArmFlashMLACudnn,
    "fa4": nb.ArmFA4,
}
NOPE_ARMS = {
    ArmDetNope.name: ArmDetNope,
    ArmDetNopeP2.name: ArmDetNopeP2,
    ArmDetNopeP4.name: ArmDetNopeP4,
    "nondet_nope": nb.ArmCakeNope,
    "fa4_nope": nb.ArmFA4Nope,
    "fa4_nope_a2": nb.ArmFA4NopeA2,
    "flashmla_cudnn_nope": nb.ArmFlashMLACudnnNope,
}
# the arms of the sink rows (--sink): name -> class(inputs, variant)
SINK_ARMS = {
    ArmDetNopeSink.name: ArmDetNopeSink,
    ArmDetNopeVarlenSink.name: ArmDetNopeVarlenSink,
    ArmCudnnFeNopeSinkDet.name: ArmCudnnFeNopeSinkDet,
    nb.ArmCakeNopeSink.name: nb.ArmCakeNopeSink,
    nb.ArmCakeNopeVarlenSink.name: nb.ArmCakeNopeVarlenSink,
    nb.ArmCudnnFeNopeSink.name: nb.ArmCudnnFeNopeSink,
}
SINK_ROWS = list(nb.SINK_ROWS) + list(nb.SINK_VARLEN_ROWS)


def _bits(t):
    view = {2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.detach().contiguous().view(view)


def _snapshot(st, grads):
    """Copies of a step's results (forward outputs and gradients), taken before the next step overwrites them."""
    snap = {}
    for name, value in {**st, **grads}.items():
        if isinstance(value, torch.Tensor):
            snap[name] = value.detach().clone()
    return snap


def _repeat_bitwise(arm):
    """Whether two steps of ``arm`` give the same bits for every tensor they return (``out`` and ``lse`` alone
    without a backward: a forward-only step leaves the accumulator it allocates unwritten)."""
    runs = []
    for _ in range(2):
        st = arm.forward()
        if arm.backward_available:
            grads = arm.backward(st)
        else:
            st, grads = {"out": st["out"], "lse": st["lse"]}, {}
        torch.cuda.synchronize()
        runs.append(_snapshot(st, grads))
    first, second = runs
    if first.keys() != second.keys():
        return False
    return all(torch.equal(_bits(first[k]), _bits(second[k])) for k in first)


def measure(arm, steps, before_arm=None):
    """Median milliseconds of one arm (see the module docstring for the columns) plus its peak memory
    (``before_arm``: the bytes allocated before the arm was built)."""
    has_backward = getattr(arm, "backward_available", True)
    st = arm.forward()
    torch.cuda.synchronize()
    perf = dict(fwd_ms=nb.median_ms(arm.forward, steps))
    if has_backward:
        if getattr(getattr(arm, "runner", None), "storage", None) != "o_lo":
            perf["bwd_again_ms"] = nb.median_ms(lambda: arm.backward(st), steps)

        def step():
            arm.backward(arm.forward())

        def recompute_step():
            arm.forward()
            arm.backward(arm.forward())

        perf["step_ms"] = nb.median_ms(step, steps)
        perf["recompute_step_ms"] = nb.median_ms(recompute_step, steps)
        perf["bwd_ms"] = perf["step_ms"] - perf["fwd_ms"]
    st = None  # release the timed forward: its tensors must not count in the peak below
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    s = arm.forward()
    if has_backward:
        arm.backward(s)
    torch.cuda.synchronize()
    perf["peak_above_inputs_gib"] = nb.gib(torch.cuda.max_memory_allocated() - base)
    if before_arm is not None:
        perf["peak_total_gib"] = nb.gib(torch.cuda.max_memory_allocated() - before_arm)
    del s
    if getattr(arm, "deterministic", False):
        perf["repeat_bitwise"] = _repeat_bitwise(arm)
    return perf


def _print(row, name, perf):
    fmt = lambda v: "     n/a" if v is None else f"{v:8.3f}"
    if "error" in perf:
        print(f"{row:24s} {name:19s} {perf['error']}", flush=True)
        return
    rep = perf.get("repeat_bitwise")
    print(
        f"{row:24s} {name:19s} fwd {fmt(perf['fwd_ms'])}  bwd {fmt(perf.get('bwd_ms'))}  "
        f"step {fmt(perf.get('step_ms'))}  recompute {fmt(perf.get('recompute_step_ms'))}  "
        f"bwd_again {fmt(perf.get('bwd_again_ms'))} ms  peak {perf['peak_above_inputs_gib']:.3f} GiB"
        f"  peak_total {perf.get('peak_total_gib', float('nan')):.3f} GiB"
        + ("" if rep is None else f"  repeat_bitwise {rep}"),
        flush=True,
    )


def _run_arm(make, row, name, steps, entry, workspace=False):
    torch.cuda.synchronize()
    before_arm = torch.cuda.memory_allocated()
    try:
        arm = make()
    except Exception as exc:  # noqa: BLE001 -- optional dependency missing or arm unavailable here
        text = str(exc)
        entry["arms"][name] = dict(
            error=text
            if text.startswith("skipped:")
            else f"{type(exc).__name__}: {text}"
        )
        _print(row, name, entry["arms"][name])
        return
    try:
        perf = measure(arm, steps, before_arm)
        if workspace:
            perf["workspace_bytes"] = arm.workspace()
        perf["versions"] = arm.versions()
    except Exception as exc:  # noqa: BLE001 -- record the failure, continue with the next arm
        perf = dict(
            error=f"failed: {type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    entry["arms"][name] = perf
    _print(row, name, perf)
    del arm
    torch.cuda.empty_cache()


def run_row(row, args, results):
    if row in P8949_ROWS and P8949_ROWS[row]:
        inp = make_p8949_inputs(True, args.device)
        arms = [
            (a, lambda cls=NOPE_ARMS[a]: cls(inp)) for a in args.arms if a in NOPE_ARMS
        ]
        entry = dict(
            row=row, total_q=inp.total_q, total_k=inp.total_k, K_slots=inp.topk, arms={}
        )
    elif row in P8949_ROWS:
        inp = make_p8949_inputs(False, args.device)
        layout = nb.Layout(inp, **nb.DEFAULT_LAYOUT)
        arms = [
            (a, lambda cls=ROPE_ARMS[a]: cls(inp, layout))
            for a in args.arms
            if a in ROPE_ARMS
        ]
        entry = dict(
            row=row,
            total_q=inp.total_q,
            total_k=inp.total_k,
            num_docs=1,
            layout=layout.describe(),
            arms={},
        )
    elif row in nb.NOPE_ROWS:
        spec = nb.NOPE_ROWS[row]
        inp = nb.make_nope_inputs(
            spec["seq_q"],
            spec["seq_k"],
            seed=nb.SEED,
            topk=spec["k_slots"],
            index=spec["index"],
            pools=spec["pools"],
            dout_layout=spec["dout_layout"],
            device=args.device,
        )
        if spec.get("holes"):
            inp = nb._plant_holes(inp, spec["holes"])
        if spec.get("twin") == "stable":
            inp = nb.compacted_twin(inp)
        arms = [
            (a, lambda cls=NOPE_ARMS[a]: cls(inp)) for a in args.arms if a in NOPE_ARMS
        ]
        entry = dict(
            row=row, total_q=inp.total_q, total_k=inp.total_k, K_slots=inp.topk, arms={}
        )
    else:
        seq_q, seq_k = nb.ROWS[row]
        inp = nb.make_inputs(
            seq_q,
            seq_k,
            seed=nb.SEED,
            topk=nb.DEFAULT_TOPK,
            dout_layout=nb.ROW_DOUT_LAYOUT.get(row, "contiguous"),
            device=args.device,
        )
        layout = nb.Layout(inp, **nb.ROW_LAYOUTS.get(row, nb.DEFAULT_LAYOUT))
        arms = [
            (a, lambda cls=ROPE_ARMS[a]: cls(inp, layout))
            for a in args.arms
            if a in ROPE_ARMS
        ]
        entry = dict(
            row=row,
            total_q=inp.total_q,
            total_k=inp.total_k,
            num_docs=len(seq_q),
            layout=layout.describe(),
            arms={},
        )
    steps = max(
        args.min_steps,
        args.steps if inp.total_q * inp.total_k < 2**34 else args.steps_large,
    )
    for name, make in arms:
        _run_arm(make, row, name, steps, entry)
    results["rows"].append(entry)


def run_sink(args, results):
    """The sink rows (``--sink``): every ``--sink-arms`` arm with the row's sink and without one
    (``--sink-variants``), each under :func:`measure`, recorded as ``arms[name][variant]``."""
    for row in args.sink_rows:
        inp, width, packed = nb.make_sink_row(row, device=args.device)
        entry = dict(
            row=row,
            total_q=inp.total_q,
            total_k=inp.total_k,
            width=width,
            ratio=inp.ratio,
            valid_keys=int((inp.indices >= 0).sum()),
            softmax_scale=SINK_SCALE,
            dout_layout="contiguous",
            **packed,
            arms={},
        )
        print(
            f"{row:24s} geometry T {inp.total_q} S {inp.total_k} K {width} valid {entry['valid_keys']}",
            flush=True,
        )
        steps = max(
            args.min_steps,
            args.steps if inp.total_q * inp.total_k < 2**34 else args.steps_large,
        )
        for name in args.sink_arms:
            entry["arms"][name] = {}
            for variant in args.sink_variants:
                label, runs = f"{name}/{variant}", dict(arms={})
                make = functools.partial(SINK_ARMS[name], inp, variant)
                _run_arm(make, row, label, steps, runs, workspace=True)
                entry["arms"][name][variant] = runs["arms"][label]
        results["sink_rows"].append(entry)
        del inp
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", nargs="+", default=list(DEFAULT_ROWS))
    parser.add_argument(
        "--arms",
        default="det,nondet,det_nope,nondet_nope",
        help=f"comma-separated, of {sorted(ROPE_ARMS) + sorted(NOPE_ARMS)}",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--steps-large", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--sink",
        action="store_true",
        help="the sink rows in place of the --rows rows (see the module docstring)",
    )
    parser.add_argument(
        "--sink-rows",
        nargs="+",
        default=SINK_ROWS,
        choices=SINK_ROWS,
        help=f"rows for --sink (default {' '.join(SINK_ROWS)})",
    )
    parser.add_argument(
        "--sink-arms",
        default=",".join(SINK_ARMS),
        help=f"comma-separated arms of the sink rows, of {sorted(SINK_ARMS)} (default all)",
    )
    parser.add_argument(
        "--sink-variants",
        default=",".join(nb.SINK_VARIANTS),
        help="comma-separated variants of every sink arm, of sink, no_sink (default both)",
    )
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    args.sink_arms = [a for a in args.sink_arms.split(",") if a]
    args.sink_variants = [v for v in args.sink_variants.split(",") if v]
    unknown_sink = [a for a in args.sink_arms if a not in SINK_ARMS] + [
        v for v in args.sink_variants if v not in ("sink", "no_sink")
    ]
    if unknown_sink:
        parser.error(f"unknown sink arms / variants {unknown_sink}")
    unknown = [a for a in args.arms if a not in ROPE_ARMS and a not in NOPE_ARMS]
    unknown_rows = [
        r
        for r in args.rows
        if r not in nb.ROWS and r not in nb.NOPE_ROWS and r not in P8949_ROWS
    ]
    if unknown or unknown_rows:
        parser.error(f"unknown arms {unknown} / rows {unknown_rows}")
    device = torch.device(args.device)
    # the synchronizations and memory statistics of measure() read the current device
    torch.cuda.set_device(device)
    results = dict(
        device=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        steps=args.steps,
        columns=dict(
            bwd_ms="step_ms - fwd_ms",
            bwd_again_ms="backward over an already-consumed forward (det autograd: includes the forward replay)",
            peak_total_gib="peak of one step above the memory allocated before the arm was built",
        ),
        rows=[],
    )
    if args.sink:
        results["sink_rows"] = []
        run_sink(args, results)
    else:
        for row in args.rows:
            run_row(row, args, results)
            torch.cuda.empty_cache()  # the row's inputs are released with run_row's frame
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
