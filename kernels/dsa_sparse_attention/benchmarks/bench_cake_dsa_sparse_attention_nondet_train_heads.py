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

Benchmark the nondet RoPE form of ``dsa_sparse_attention`` at 96 and 64 query heads against non-Cake baselines.

The rows are the context-parallel geometry of a 1,048,320-token sequence split over 64 ranks (``T = 16,380`` query
rows per rank), top-k 2048, ``softmax_scale = 192 ** -0.5``, BF16 operands and an FP32 ``dkv_acc [S, 576]``:

* ``r1_doc_16380``: rank 0, one causal document (``S = 16,380``; rows near its start have fewer valid keys);
* ``r2_cp_mid_524160``: a middle rank (``S = 524,160``, every slot valid);
* ``r3_cp_last_1048320``: the last rank (``S = 1,048,320``);
* ``r4_packed4_16380``: four causal documents of 4,095 tokens packed into one call (``num_segments = 4``).

Inputs come from the reference helper's ``make_inputs`` (per-document causal random top-k, valid slots first, ``-1``
after them), so every arm sees the same values.  The head count is a run-time option (``--heads 96 64``).

Arms (each times its own whole call, glue included):

* ``cake`` / ``cake_no_olo``: the public flat autograd entry ``cake_dsa_sparse_attention_nondet`` (global index rows,
  ``topk_length``, ``num_segments`` = the row's document count) with ``dkv_acc`` given as a callable, so the zeroed
  FP32 ``[S, 576]`` accumulator is created inside the timed backward; the backward is ``torch.autograd.grad`` of
  ``out`` with respect to ``q_latent`` and ``q_rope``.  ``cake_no_olo`` passes ``with_o_lo=False``.  A package that
  does not serve the head count records the arm as unavailable with its error.
* ``cake_det``: the deterministic flat autograd entry ``cake_dsa_sparse_attention_det`` in the same form (no
  ``with_o_lo`` option).  Its backward runs in the storage of the saved ``o_lo`` and so replays the forward when called
  again on a consumed graph; its ``bwd`` op therefore times each backward on a fresh forward made outside the timed
  span.  A package without a deterministic module for the head count records the arm as unavailable with its error
  (the "vs deterministic Cake" column of such a row is unsupported).
* ``cudnn_fe``: the cuDNN frontend's native DSA forward (``sparse_attention_forward_wrapper``) and its default
  non-deterministic backward (``DSA.sparse_attention_backward_wrapper``) on packed ``q [T, H, 576]`` and ``kv [S, 576]``
  (built inside the timed forward with ``--layout split``, the default; given once with ``--layout packed``).  The
  frontend's ``lse`` is ``+inf`` on a row without a valid slot; inside the timed forward that becomes the Cake
  ``-inf`` and the backward takes the frontend's own ``lse``.  Its BF16 ``dkv`` is upcast-added into the zeroed FP32
  accumulator inside the timed backward.  The installed frontend decides which build runs (the record names its
  version, file and the backward backend it selected).
* ``fa4``: FlashAttention-4's sparse MLA kernels through autograd (``flash_attn.cute.flash_attn_varlen_func`` with
  ``q_rope``, ``k_rope [S, 1, 64]``, ``kv_latent [S, 1, 512]``, ``qv = q_latent``, per-document gather indices,
  ``causal=True``, recompute-P backward; ``--fa4-token-chunk`` runs it in token chunks, default ``0``: one unchunked
  backward, its fastest setting on these rows against 4096- and 8192-token chunks, at the cost of a ``dS`` scratch of
  ``T x H x top-k`` BF16); its BF16 ``dk`` / ``dv`` are upcast-added into the zeroed FP32 accumulator inside the timed
  backward.
* ``fa4_fp32``: the same kernels through their private forward (``_flash_attn_fwd``, training configuration, returning
  ``o_lo``) and sparse-MLA backward (``_flash_attn_bwd_sparse_mla``) with preallocated zeroed FP32 ``dk [S, 1, 64]`` and
  ``dv [S, 1, 512]`` (the kernels' own FP32 accumulation, no BF16 round trip); these two buffers are the arm's
  accumulator.  The faster of ``fa4`` and ``fa4_fp32`` per row is the FlashAttention-4 baseline.

Index glue outside the timed region, for every arm: the frontend takes the valid slots first with ``topk_length``
(``-1`` replaced by 0 past it; rows with holes are compacted stably first), FlashAttention-4 the per-document rows with
``-1`` for every invalid slot, Cake the global rows and ``topk_length``.

Ops: ``fwd``; ``bwd`` (one retained forward, the backward called repeatedly; ``cake_det``: a fresh forward per call,
outside the span); ``step`` (forward + backward);
``recompute`` (forward + forward + backward: the forward runs twice per update under activation checkpointing).

Timing (one arm per process is the protocol; the driver runs fresh processes in rotating rounds): CUPTI activity spans
through ``torch.profiler`` with a cold L2 before every iteration (the timer of
``bench_cake_dsa_sparse_attention_nondet_train.py``, kept here so that a baseline process imports no Cake module),
``--warmup`` warm-up calls, then ``--steps`` timed calls; every per-iteration span is recorded with its median and the
timer that produced it (``cupti``, or ``cuda_events (...)`` with the reason when the profiler recorded no activity).
With ``nvidia-ml-py`` installed each op also records NVML of the timing GPU by UUID: the SM clock (5 ms samples), the
share of samples with a power-cap / slowdown clock event reason, the energy over the op and the enforced power limit.

Memory accounting per arm and row (torch allocator, bytes): ``fwd_workspace`` (peak of the forward above what it
leaves allocated), ``out_lse`` (the returned ``out`` and ``lse``), ``retained`` (what the forward keeps alive for the
backward besides ``out`` and ``lse``; the retained state of the API reference's table counts ``lse`` too, i.e.
``retained`` plus the ``lse`` bytes of ``out_lse``, ``4 H`` per token), ``glue_retained`` (the part of ``retained`` made by the arm's input glue: the
frontend's packed ``q`` and ``kv`` copies in the split layout), ``bwd_workspace`` (peak of the backward above the
gradients it returns) and ``grads`` (``dq`` plus the FP32 accumulator).

Usage::

    python benchmarks/bench_cake_dsa_sparse_attention_nondet_train_heads.py --arms cudnn_fe --heads 96 64 --json fe.json
    python benchmarks/bench_cake_dsa_sparse_attention_nondet_train_heads.py --arms cake --heads 64 \\
        --rows r1_doc_16380 r4_packed4_16380 --steps 20 --json cake64.json
    python benchmarks/bench_cake_dsa_sparse_attention_nondet_train_heads.py --arms fa4,fa4_fp32 --heads 96 \\
        --fa4-token-chunk 8192 --json fa4-8192.json
"""

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The reference helper imports torch only: a baseline process loads no Cake module unless a Cake arm runs.
from tests.experimental.test_helpers.cake_dsa_sparse_attention_nondet_train_reference import (  # noqa: E402, RUF100
    D_LATENT,
    D_QK,
    D_ROPE,
    DEFAULT_TOPK,
    compact_rows,
    make_inputs,
)

#: ``qk_nope_head_dim`` 128 + ``qk_rope_head_dim`` 64 of the model the 96-head form serves.
SCALE = 192**-0.5
T_RANK = 16380
ROWS: dict[str, tuple[list, list]] = {
    "r1_doc_16380": ([T_RANK], [T_RANK]),
    "r2_cp_mid_524160": ([T_RANK], [32 * T_RANK]),
    "r3_cp_last_1048320": ([T_RANK], [64 * T_RANK]),
    "r4_packed4_16380": ([T_RANK // 4] * 4, [T_RANK // 4] * 4),
}
HEADS = (96, 64)
OPS = ("fwd", "bwd", "step", "recompute")
LAYOUTS = ("split", "packed")
SEED = 26300
#: FlashAttention-4 backward token chunk; 0 runs one unchunked backward.
FA4_TOKEN_CHUNK = 0


def _nbytes(*tensors):
    return sum(
        int(t.numel()) * t.element_size()
        for t in tensors
        if isinstance(t, torch.Tensor)
    )


def _local_rows(inp):
    """Per-document index rows with ``-1`` for every invalid slot (``< 0`` or ``>=`` the document's key count)."""
    cu_k = inp.cu_seqlens_k.tolist()
    cu_q = inp.cu_seqlens_q.tolist()
    local = inp.idx_local
    out = torch.empty_like(local)
    for d in range(len(cu_q) - 1):
        rows = slice(cu_q[d], cu_q[d + 1])
        n_k = cu_k[d + 1] - cu_k[d]
        block = local[rows]
        out[rows] = torch.where(
            (block >= 0) & (block < n_k), block, block.new_full((), -1)
        )
    return out.contiguous()


class _Accumulating:
    """The zeroed FP32 ``dkv_acc [S, 576]`` an arm's backward adds into (latent columns ``0:512``, rope ``512:576``)."""

    #: Time the ``bwd`` op on a fresh forward per call (made outside the timed span), not on one retained forward.
    bwd_fresh_forward = False

    def glue_retained(self, st):
        """Bytes of the forward state that the arm's input glue made (part of ``retained``)."""
        return 0

    @staticmethod
    def new_acc(inp):
        return torch.zeros(
            inp.total_k, D_QK, dtype=torch.float32, device=inp.q_latent.device
        )

    @staticmethod
    def grads(dq_latent, dq_rope, acc):
        return {
            "dq_latent": dq_latent,
            "dq_rope": dq_rope,
            "dkv_latent": acc[:, :D_LATENT],
            "dk_rope": acc[:, D_LATENT:D_QK],
        }


class ArmCake(_Accumulating):
    """The public flat autograd entry (see the module docstring)."""

    name = "cake"
    with_o_lo = True

    def __init__(self, inp, *, scale=SCALE, layout="split", topk_length=True):
        self.fn, self.inp, self.scale = self._entry(), inp, float(scale)
        self.layout = layout
        if layout == "packed":
            q = torch.cat([inp.q_latent, inp.q_rope], dim=-1).requires_grad_()
            self.q_leaf = q
            self.q_latent, self.q_rope = q[..., :D_LATENT], q[..., D_LATENT:]
            kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1)
            self.kv_latent, self.k_rope = kv[:, :D_LATENT], kv[:, D_LATENT:]
        else:
            self.q_latent = inp.q_latent.detach().requires_grad_()
            self.q_rope = inp.q_rope.detach().requires_grad_()
            self.q_leaf = None
            self.kv_latent, self.k_rope = inp.kv_latent, inp.k_rope
        self.indices = inp.idx_global
        self.topk_length = inp.topk_length if topk_length else None
        self.num_segments = int(inp.cu_seqlens_k.numel()) - 1
        self.forward()  # validation, binding and the first build, outside the timed region
        torch.cuda.synchronize()

    def _entry(self):
        from cake_pplx.cake_dsa_sparse_attention_nondet_train import (
            cake_dsa_sparse_attention_nondet,
        )

        return cake_dsa_sparse_attention_nondet

    def _forward_kwargs(self):
        return {} if self.with_o_lo else {"with_o_lo": False}

    def versions(self):
        import importlib.metadata as md

        mod = sys.modules.get(self.fn.__module__)
        return {
            "entry": "cake_dsa_sparse_attention_nondet (autograd, dkv_acc callable)",
            "dist": md.version("cake-pplx-dsa-sparse-attention"),
            "module_file": getattr(mod, "__file__", None),
            "with_o_lo": self.with_o_lo,
            "num_segments": self.num_segments,
            "topk_length": self.topk_length is not None,
            "layout": self.layout,
        }

    def forward(self):
        holder = {}
        inp = self.inp

        def acc():
            holder["acc"] = self.new_acc(inp)
            return holder["acc"]

        kw = self._forward_kwargs()
        out, lse = self.fn(
            self.q_latent,
            self.q_rope,
            self.kv_latent,
            self.k_rope,
            self.indices,
            topk_length=self.topk_length,
            softmax_scale=self.scale,
            return_lse=True,
            dkv_acc=acc,
            num_segments=self.num_segments,
            **kw,
        )
        return {"out": out, "lse": lse, "holder": holder}

    def backward(self, st):
        if self.q_leaf is not None:
            (dq,) = torch.autograd.grad(
                st["out"], (self.q_leaf,), self.inp.dout, retain_graph=True
            )
            dq_latent, dq_rope = dq[..., :D_LATENT], dq[..., D_LATENT:]
        else:
            dq_latent, dq_rope = torch.autograd.grad(
                st["out"],
                (self.q_latent, self.q_rope),
                self.inp.dout,
                retain_graph=True,
            )
        return self.grads(dq_latent, dq_rope, st["holder"]["acc"])


class ArmCakeNoOlo(ArmCake):
    """``cake`` with ``with_o_lo=False``."""

    name = "cake_no_olo"
    with_o_lo = False


class ArmCakeDet(ArmCake):
    """The deterministic flat autograd entry ``cake_dsa_sparse_attention_det`` in the form of ``cake`` (see the module
    docstring); its backward is timed on a fresh forward per iteration."""

    name = "cake_det"
    bwd_fresh_forward = True

    def _entry(self):
        from cake_pplx.cake_dsa_sparse_attention_det_train import (
            cake_dsa_sparse_attention_det,
        )

        return cake_dsa_sparse_attention_det

    def _forward_kwargs(self):
        return {}

    def versions(self):
        found = super().versions()
        found.pop("with_o_lo")
        return dict(
            found,
            entry="cake_dsa_sparse_attention_det (autograd, dkv_acc callable)",
        )


class ArmCudnnFe(_Accumulating):
    """The cuDNN frontend's native DSA forward and non-deterministic backward (see the module docstring)."""

    name = "cudnn_fe"

    def __init__(self, inp, *, scale=SCALE, layout="split"):
        import cudnn
        from cudnn import DSA
        from cudnn.deepseek_sparse_attention.sparse_attention_forward.api import (
            sparse_attention_forward_wrapper,
        )

        self.cudnn, self.DSA, self._fwd = cudnn, DSA, sparse_attention_forward_wrapper
        self.inp, self.scale, self.layout = inp, float(scale), layout
        heads = int(inp.q_latent.shape[1])
        rows, counts = compact_rows(inp.idx_global, inp.total_k)
        self.idx = rows.clamp_min(0).contiguous()
        self.topk_length = counts.contiguous()
        self.sink = torch.full(
            (heads,), float("-inf"), dtype=torch.float32, device=inp.q_latent.device
        )
        self.q = self.kv = None
        if layout == "packed":
            self.q = torch.cat([inp.q_latent, inp.q_rope], dim=-1)
            self.kv = torch.cat([inp.kv_latent, inp.k_rope], dim=-1)
        self.rank3 = None
        st = (
            self.forward()
        )  # kv / index rank probe and the forward build, outside the timed region
        self.backward(st)  # the backward plan and build
        torch.cuda.synchronize()

    def _packed(self):
        if self.q is not None:
            return self.q, self.kv
        inp = self.inp
        return torch.cat([inp.q_latent, inp.q_rope], dim=-1), torch.cat(
            [inp.kv_latent, inp.k_rope], dim=-1
        )

    def _call_fwd(self, q, kv, rank3):
        kw = {"topk_length": self.topk_length, "softmax_scale": self.scale}
        return self._fwd(
            q,
            kv.unsqueeze(1) if rank3 else kv,
            self.idx.unsqueeze(1) if rank3 else self.idx,
            **kw,
        )

    def forward(self):
        q, kv = self._packed()
        if self.rank3 is None:
            last = None
            for rank3 in (False, True):
                try:
                    res = self._call_fwd(q, kv, rank3)
                    torch.cuda.synchronize()
                    self.rank3 = rank3
                    break
                except Exception as exc:  # noqa: BLE001 - the other rank may be the accepted one
                    last = exc
            else:
                raise RuntimeError(
                    f"the frontend forward refused both kv / index ranks: {type(last).__name__}: {last}"
                )
        else:
            res = self._call_fwd(q, kv, self.rank3)
        fe_lse = res["lse"]
        lse = fe_lse.masked_fill(torch.isposinf(fe_lse), float("-inf"))
        return {"q": q, "kv": kv, "out": res["out"], "lse": lse, "fe_lse": fe_lse}

    def glue_retained(self, st):
        """The packed ``q [T, H, 576]`` and ``kv [S, 576]`` copies the split layout makes inside the forward."""
        return _nbytes(st["q"], st["kv"]) if self.layout == "split" else 0

    def backward(self, st):
        res = self.DSA.sparse_attention_backward_wrapper(
            st["q"],
            st["kv"],
            st["out"],
            self.inp.dout,
            st["fe_lse"],
            self.sink,
            self.idx,
            softmax_scale=self.scale,
            topk_length=self.topk_length,
        )
        dq, dkv = res["dq"], res["dkv"]
        acc = self.new_acc(self.inp)
        acc.add_(dkv)
        return self.grads(dq[..., :D_LATENT], dq[..., D_LATENT:], acc)

    def backend(self):
        """The backward backend(s) the frontend's cached plans of this arm's query shape selected (e.g.
        ``h96_d576_2cta_m64``); the plan cache is keyed by the input shapes, the query shape third."""
        try:
            from cudnn.deepseek_sparse_attention.sparse_attention_backward import (
                api as bapi,
            )

            cache = getattr(bapi, "_cache_of_SparseAttentionBackwardObjects", {})
            shape = tuple(self.inp.q_latent.shape[:2]) + (D_QK,)
            return sorted(
                {
                    str(getattr(obj, "_backend", None))
                    for key, obj in cache.items()
                    if len(key) > 2 and tuple(key[2]) == shape
                }
            )
        except Exception as exc:  # noqa: BLE001 - informational
            return [f"{type(exc).__name__}: {exc}"]

    def versions(self):
        import importlib.metadata as md

        try:
            dist = md.version("nvidia-cudnn-frontend")
        except Exception:  # noqa: BLE001
            dist = None
        backend = getattr(self.cudnn, "backend_version", None)
        windows = None
        try:
            from cudnn.deepseek_sparse_attention.sparse_attention_forward import (
                _interface_sm100 as fi,
            )

            windows = {
                str(k): list(v)
                for k, v in getattr(fi, "_HEAD64_WINDOW_STARTS", {}).items()
            }
        except Exception as exc:  # noqa: BLE001 - informational
            windows = f"unavailable: {type(exc).__name__}: {exc}"
        return {
            "cudnn_frontend": getattr(self.cudnn, "__version__", None),
            "dist": dist,
            "module_file": getattr(self.cudnn, "__file__", None),
            "cudnn_backend": backend() if callable(backend) else None,
            "forward": "sparse_attention_forward_wrapper",
            "backward": "DSA.sparse_attention_backward_wrapper (default, non-deterministic)",
            "backward_backend": self.backend(),
            "forward_head64_windows": windows,
            "kv_index_rank3": self.rank3,
            "layout": self.layout,
            "cake_vendored_modules_loaded": sorted(
                m for m in sys.modules if "cake" in m.lower() and "cudnn" in m.lower()
            ),
        }


class ArmFA4(_Accumulating):
    """FlashAttention-4 sparse MLA through autograd (see the module docstring)."""

    name = "fa4"

    def __init__(
        self,
        inp,
        *,
        scale=SCALE,
        layout="split",
        causal=True,
        token_chunk=FA4_TOKEN_CHUNK,
    ):
        from flash_attn.cute import flash_attn_varlen_func

        self.fn, self.inp, self.scale, self.layout = (
            flash_attn_varlen_func,
            inp,
            float(scale),
            layout,
        )
        self.causal = bool(causal)
        self.token_chunk = int(token_chunk) or None  # 0: one unchunked backward
        self.idx = _local_rows(inp)
        self.q_rope = inp.q_rope.detach().contiguous().requires_grad_()
        self.q_latent = inp.q_latent.detach().contiguous().requires_grad_()
        self.k_rope = inp.k_rope.detach().unsqueeze(1).contiguous().requires_grad_()
        self.kv_latent = (
            inp.kv_latent.detach().unsqueeze(1).contiguous().requires_grad_()
        )
        st = self.forward()
        self.backward(st)
        torch.cuda.synchronize()

    def _lse(self, lse):
        total_q, heads = (
            int(self.inp.q_latent.shape[0]),
            int(self.inp.q_latent.shape[1]),
        )
        if tuple(lse.shape) == (heads, total_q) and heads != total_q:
            lse = lse.t()
        return lse.masked_fill(torch.isposinf(lse), float("-inf"))

    def forward(self):
        inp = self.inp
        out, lse = self.fn(
            self.q_rope,
            self.k_rope,
            self.kv_latent,
            qv=self.q_latent,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            softmax_scale=self.scale,
            causal=self.causal,
            gather_kv_indices=self.idx,
            pack_gqa=True,
            gather_bwd_recompute_p=True,
            gather_bwd_token_chunk=self.token_chunk,
            return_lse=True,
        )
        return {"out": out, "lse": self._lse(lse), "raw_lse": lse}

    def backward(self, st):
        dq_rope, dk_rope, dkv_latent, dq_latent = torch.autograd.grad(
            st["out"],
            (self.q_rope, self.k_rope, self.kv_latent, self.q_latent),
            self.inp.dout,
            retain_graph=True,
        )
        acc = self.new_acc(self.inp)
        acc[:, :D_LATENT].add_(dkv_latent[:, 0])
        acc[:, D_LATENT:D_QK].add_(dk_rope[:, 0])
        return self.grads(dq_latent, dq_rope, acc)

    def versions(self):
        import importlib.metadata as md

        import cutlass
        import flash_attn.cute as fc

        try:
            dist = md.version("flash-attn-4")
        except Exception:  # noqa: BLE001
            dist = None
        return {
            "entry": "flash_attn.cute.flash_attn_varlen_func (autograd)",
            "dist": dist,
            "module_file": fc.__file__,
            "cutlass_dsl": getattr(cutlass, "__version__", None),
            "causal": self.causal,
            "token_chunk": self.token_chunk,
            "layout": self.layout,
        }


class ArmFA4Fp32(ArmFA4):
    """FlashAttention-4 sparse MLA through its private entries with FP32 ``dk`` / ``dv`` (see the module docstring)."""

    name = "fa4_fp32"

    def __init__(
        self,
        inp,
        *,
        scale=SCALE,
        layout="split",
        causal=True,
        token_chunk=FA4_TOKEN_CHUNK,
    ):
        from flash_attn.cute.interface import (
            _flash_attn_bwd_sparse_mla,
            _flash_attn_fwd,
        )

        self.ffwd, self.fbwd = _flash_attn_fwd, _flash_attn_bwd_sparse_mla
        super().__init__(
            inp, scale=scale, layout=layout, causal=causal, token_chunk=token_chunk
        )

    def forward(self):
        inp = self.inp
        out, lse, _p, _row_max, o_lo = self.ffwd(
            self.q_rope,
            self.k_rope,
            self.kv_latent,
            qv=self.q_latent,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            softmax_scale=self.scale,
            causal=self.causal,
            pack_gqa=True,
            return_lse=True,
            gather_kv_indices=self.idx,
            gather_bwd_recompute_p=True,
        )
        return {"out": out, "lse": self._lse(lse), "raw_lse": lse, "o_lo": o_lo}

    def backward(self, st):
        inp = self.inp
        dev = self.q_latent.device
        dk = torch.zeros(inp.total_k, 1, D_ROPE, dtype=torch.float32, device=dev)
        dv = torch.zeros(inp.total_k, 1, D_LATENT, dtype=torch.float32, device=dev)
        dq_rope, dk, dv, dq_latent, _dsink = self.fbwd(
            self.q_rope.detach(),
            self.k_rope.detach(),
            self.kv_latent.detach(),
            self.q_latent.detach(),
            st["out"],
            inp.dout,
            st["raw_lse"],
            None,
            None,
            self.idx,
            softmax_scale=self.scale,
            causal=self.causal,
            cu_seqlens_q=inp.cu_seqlens_q,
            cu_seqlens_k=inp.cu_seqlens_k,
            max_seqlen_q=inp.max_seqlen_q,
            max_seqlen_k=inp.max_seqlen_k,
            recompute_p=True,
            token_chunk=self.token_chunk,
            o_lo=st["o_lo"],
            dk=dk,
            dv=dv,
        )
        return {
            "dq_latent": dq_latent,
            "dq_rope": dq_rope,
            "dkv_latent": dv[:, 0],
            "dk_rope": dk[:, 0],
        }

    def versions(self):
        return dict(
            super().versions(),
            entry="flash_attn.cute.interface._flash_attn_fwd + _flash_attn_bwd_sparse_mla (FP32 dk / dv, o_lo)",
        )


ARMS = {
    cls.name: cls
    for cls in (ArmCake, ArmCakeNoOlo, ArmCakeDet, ArmCudnnFe, ArmFA4, ArmFA4Fp32)
}
FA4_ARMS = ("fa4", "fa4_fp32")


# ---------------------------------------------------------------------------
# NVML of the timing GPU
# ---------------------------------------------------------------------------

#: clock event reasons counted as a power / thermal cap: SW power cap, HW slowdown, SW thermal, HW thermal, HW power brake
_CAP_BITS = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


class Nvml:
    """SM clock samples, clock event reasons, energy and power limit of one GPU (by UUID), or ``None`` fields."""

    def __init__(self, device):
        self.handle, self.error = None, None
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = "GPU-" + str(torch.cuda.get_device_properties(device).uuid)
            self.nv, self.uuid = pynvml, uuid
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
        except Exception as exc:  # noqa: BLE001 - optional
            self.error = f"{type(exc).__name__}: {exc}"

    def _read(self, name, *args):
        try:
            return getattr(self.nv, name)(self.handle, *args)
        except Exception:  # noqa: BLE001
            return None

    def start(self):
        if self.handle is None:
            return None
        state = {
            "stop": threading.Event(),
            "clocks": [],
            "reasons": [],
            "t0": time.time(),
        }
        state["energy0"] = self._read("nvmlDeviceGetTotalEnergyConsumption")
        nv = self.nv

        def run():
            while not state["stop"].is_set():
                c = self._read("nvmlDeviceGetClockInfo", nv.NVML_CLOCK_SM)
                r = self._read("nvmlDeviceGetCurrentClocksEventReasons") or self._read(
                    "nvmlDeviceGetCurrentClocksThrottleReasons"
                )
                if c is not None:
                    state["clocks"].append(int(c))
                if r is not None:
                    state["reasons"].append(int(r))
                state["stop"].wait(0.005)

        state["thread"] = threading.Thread(target=run, daemon=True)
        state["thread"].start()
        return state

    def stop(self, state):
        if state is None:
            return {"nvml": self.error or "unavailable"}
        state["stop"].set()
        state["thread"].join()
        e1 = self._read("nvmlDeviceGetTotalEnergyConsumption")
        clocks, reasons = state["clocks"], state["reasons"]
        return {
            "gpu_uuid": self.uuid,
            "sm_clock_mhz_median": statistics.median(clocks) if clocks else None,
            "sm_clock_mhz_min": min(clocks) if clocks else None,
            "samples": len(clocks),
            "cap_reason_share": (
                sum(1 for r in reasons if r & _CAP_BITS) / len(reasons)
            )
            if reasons
            else None,
            "reasons_seen": sorted({hex(r) for r in reasons}),
            "energy_mj": (e1 - state["energy0"])
            if e1 is not None and state["energy0"] is not None
            else None,
            "wall_s": time.time() - state["t0"],
            "power_limit_mw": self._read("nvmlDeviceGetEnforcedPowerLimit"),
        }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


_ITERATION_RANGE = "cake_bench_iteration_"
_L2_FLUSH: dict = {}


def _flush_l2(device):
    """Overwrite a buffer twice the size of the device's L2 so that the next call starts with a cold L2."""
    buffer = _L2_FLUSH.get(device)
    if buffer is None:
        l2_bytes = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
        buffer = torch.empty(
            2 * (l2_bytes or (128 << 20)), dtype=torch.uint8, device=device
        )
        _L2_FLUSH[device] = buffer
    buffer.zero_()


def _span_ns(event):
    """``(start, end)`` of a profiler event in ns."""
    if hasattr(event, "start_ns"):
        start = event.start_ns()
        return start, start + event.duration_ns()
    start = event.start_us() * 1000
    return start, start + event.duration_us() * 1000


def _call(fn, prep):
    """One untimed ``prep()`` (when given) for the next timed call: ``fn(prep())`` split at the L2 flush."""
    if prep is None:
        return fn
    state = prep()
    return lambda: fn(state)


def _cupti_times(fn, reps, warmup, prep=None):
    """Per-iteration GPU spans (ms) from the CUPTI activity records of ``torch.profiler``, or None when an iteration
    has no recorded GPU activity.

    Every iteration runs ``prep`` (when given; untimed), flushes the L2, then calls ``fn`` (``fn(prep())`` with
    ``prep``) inside a ``record_function`` range.  The CUDA runtime / driver calls that start inside that range name
    (by CUPTI correlation id) the GPU activities of the iteration; its span runs from the first start to the last end
    of those activities."""
    from torch.profiler import ProfilerActivity, profile, record_function

    device = torch.device("cuda", torch.cuda.current_device())
    for _ in range(warmup):
        call = _call(fn, prep)
        _flush_l2(device)
        call()
        del call
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(reps):
            call = _call(fn, prep)
            _flush_l2(device)
            with record_function(f"{_ITERATION_RANGE}{i}"):
                call()
            del call
        torch.cuda.synchronize()
    events = list(prof.profiler.kineto_results.events())
    cpu, gpu = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    ranges = {}
    launches = []
    activities = {}
    for event in events:
        name = event.name()
        if event.device_type() == cpu:
            if name.startswith(_ITERATION_RANGE):
                ranges[int(name[len(_ITERATION_RANGE) :])] = _span_ns(event)
            elif name.startswith("cu") and event.correlation_id() > 0:
                launches.append((_span_ns(event)[0], event.correlation_id()))
        elif event.device_type() == gpu and event.correlation_id() > 0:
            activities.setdefault(event.correlation_id(), []).append(_span_ns(event))
    times = []
    for i in range(reps):
        if i not in ranges:
            return None
        lo, hi = ranges[i]
        spans = [
            span
            for start, corr in launches
            if lo <= start <= hi
            for span in activities.get(corr, ())
        ]
        if not spans:
            return None
        times.append(
            (max(end for _, end in spans) - min(start for start, _ in spans)) / 1e6
        )
    return times


def _cuda_event_times(fn, reps, warmup, prep=None):
    """Per-iteration CUDA-event spans (ms), ``prep`` (untimed) and an L2 flush before every iteration."""
    device = torch.device("cuda", torch.cuda.current_device())
    for _ in range(warmup):
        call = _call(fn, prep)
        _flush_l2(device)
        call()
        del call
    torch.cuda.synchronize()
    pairs = []
    for _ in range(reps):
        call = _call(fn, prep)
        _flush_l2(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        del call
        pairs.append((start, end))
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in pairs]


def spans_ms(fn, steps, warmup, prep=None):
    """Per-iteration GPU spans (ms) of ``steps`` calls after ``warmup`` calls, a cold L2 before every call: CUPTI
    activity through ``torch.profiler`` (timer ``cupti``), CUDA events where the profiler records no GPU activity
    (timer ``cuda_events (...)``).  With ``prep`` every call is ``fn(prep())``, ``prep`` outside the span."""
    try:
        times, timer = _cupti_times(fn, steps, warmup, prep), "cupti"
    except Exception as exc:  # noqa: BLE001 - any profiler failure selects the CUDA-event fallback
        times, timer = None, f"cupti failed: {type(exc).__name__}: {exc}"
    if times is None:
        times, timer = (
            _cuda_event_times(fn, steps, warmup, prep),
            f"cuda_events ({timer})",
        )
    return times, timer


def _summary(times):
    s = sorted(times)

    def q(f):
        return s[min(len(s) - 1, max(0, round(f * (len(s) - 1))))]

    return {
        "median_ms": statistics.median(s),
        "min_ms": s[0],
        "p25_ms": q(0.25),
        "p75_ms": q(0.75),
        "max_ms": s[-1],
        "n": len(s),
    }


def accounting(arm, device):
    """Allocator bytes of one forward and one backward (see the module docstring)."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    st = arm.forward()
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated(device)
    fwd_peak = torch.cuda.max_memory_allocated(device)
    out_lse = _nbytes(st["out"], st["lse"])
    glue = arm.glue_retained(st)
    torch.cuda.reset_peak_memory_stats(device)
    g = arm.backward(st)
    torch.cuda.synchronize()
    after_bwd = torch.cuda.memory_allocated(device)
    bwd_peak = torch.cuda.max_memory_allocated(device)
    res = {
        "fwd_workspace": fwd_peak - after_fwd,
        "out_lse": out_lse,
        "retained": after_fwd - base - out_lse,
        "glue_retained": glue,
        "bwd_workspace": bwd_peak - after_bwd,
        "grads": after_bwd - after_fwd,
    }
    del st, g
    return res


def measure(arm, ops, steps, warmup, nvml):
    st = arm.forward()
    torch.cuda.synchronize()
    fns = {
        "fwd": (arm.forward, None),
        "bwd": (arm.backward, arm.forward)
        if arm.bwd_fresh_forward
        else (lambda: arm.backward(st), None),
        "step": (lambda: arm.backward(arm.forward()), None),
        "recompute": (lambda: (arm.forward(), arm.backward(arm.forward())), None),
    }
    found = {}
    for op in ops:
        fn, prep = fns[op]
        state = nvml.start()
        times, timer = spans_ms(fn, steps, warmup, prep)
        found[op] = dict(
            _summary(times),
            times_ms=times,
            timer=timer,
            form="fresh forward per call, outside the span"
            if prep is not None
            else ("one retained forward" if op == "bwd" else "whole call"),
            nvml=nvml.stop(state),
        )
    torch.cuda.synchronize()
    return found


def kernel_table(fn, warmup=1):
    """``{kernel name: [launches, total GPU ms]}`` of one call of ``fn`` (CUPTI kernel records), after ``warmup`` calls."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    table = {}
    for event in prof.events():
        if event.device_type.name != "CUDA":
            continue
        entry = table.setdefault(event.name, [0, 0.0])
        entry[0] += 1
        entry[1] += event.time_range.elapsed_us() / 1000.0
    return table


def environment(device):
    import importlib.metadata as md

    def v(name):
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            return None

    props = torch.cuda.get_device_properties(device)
    driver = None
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
    except Exception as exc:  # noqa: BLE001 - informational
        driver = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "host": os.uname().nodename,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": props.name,
        "cc": f"{props.major}.{props.minor}",
        "sms": props.multi_processor_count,
        "gpu_uuid": "GPU-" + str(props.uuid),
        "cudnn_frontend": v("nvidia-cudnn-frontend"),
        "flash_attn_4": v("flash-attn-4"),
        "cutlass_dsl": v("nvidia-cutlass-dsl"),
        "cake_dsa_sparse_attention": v("cake-pplx-dsa-sparse-attention"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "bench_file": str(Path(__file__).resolve()),
    }


def run(args):
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    nvml = Nvml(device)
    results = {"env": environment(device), "args": vars(args), "rows": []}
    print(json.dumps(results["env"]), flush=True)
    for row in args.rows:
        seq_q, seq_k = ROWS[row]
        for heads in args.heads:
            t0 = time.time()
            inp = make_inputs(
                seq_q, seq_k, seed=args.seed, topk=args.topk, heads=heads, device=device
            )
            torch.cuda.synchronize()
            entry = {
                "row": row,
                "heads": heads,
                "total_q": inp.total_q,
                "total_k": inp.total_k,
                "num_docs": len(seq_q),
                "topk": args.topk,
                "softmax_scale": args.softmax_scale,
                "valid_slots": int(inp.topk_length.sum()),
                "inputs_bytes": _nbytes(
                    inp.q_latent,
                    inp.q_rope,
                    inp.kv_latent,
                    inp.k_rope,
                    inp.dout,
                    inp.idx_global,
                ),
                "input_seconds": time.time() - t0,
                "arms": {},
            }
            for name in args.arms:
                kw = {"scale": args.softmax_scale, "layout": args.layout}
                if name in FA4_ARMS:
                    kw["token_chunk"] = args.fa4_token_chunk
                try:
                    arm = ARMS[name](inp, **kw)
                except Exception as exc:  # noqa: BLE001 - unavailable package, refused head count or geometry
                    entry["arms"][name] = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                    print(
                        f"{row:20s} H{heads:<3d} {name:12s} unavailable: {type(exc).__name__}: {str(exc)[:300]}",
                        flush=True,
                    )
                    torch.cuda.empty_cache()
                    continue
                try:
                    perf = {
                        "ops": measure(arm, args.ops, args.steps, args.warmup, nvml)
                    }
                    perf["bytes"] = accounting(arm, device)
                    perf["versions"] = arm.versions()
                    if args.kernels:
                        perf["kernels_step"] = kernel_table(
                            lambda a=arm: a.backward(a.forward())
                        )
                except Exception as exc:  # noqa: BLE001
                    perf = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                entry["arms"][name] = perf
                if "error" in perf:
                    print(
                        f"{row:20s} H{heads:<3d} {name:12s} failed: {perf['error'][:300]}",
                        flush=True,
                    )
                else:
                    ops = " ".join(
                        f"{op} {perf['ops'][op]['median_ms']:9.3f}" for op in args.ops
                    )
                    b = perf["bytes"]
                    print(
                        f"{row:20s} H{heads:<3d} {name:12s} {ops} ms | retained {b['retained'] / 2**20:.0f} MiB "
                        f"(glue {b['glue_retained'] / 2**20:.0f} MiB) fwd_ws {b['fwd_workspace'] / 2**20:.0f} MiB "
                        f"bwd_ws {b['bwd_workspace'] / 2**20:.0f} MiB",
                        flush=True,
                    )
                arm = None
                torch.cuda.empty_cache()
            results["rows"].append(entry)
            del inp
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="*", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--heads", nargs="*", type=int, default=list(HEADS))
    parser.add_argument(
        "--arms",
        default="cake,cudnn_fe,fa4,fa4_fp32",
        help=f"comma list of {', '.join(ARMS)}",
    )
    parser.add_argument(
        "--ops", default=",".join(OPS), help=f"comma list of {', '.join(OPS)}"
    )
    parser.add_argument("--layout", choices=LAYOUTS, default="split")
    parser.add_argument(
        "--fa4-token-chunk",
        type=int,
        default=FA4_TOKEN_CHUNK,
        help="token chunk of the FlashAttention-4 sparse MLA backward (default 0: one unchunked backward)",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--softmax-scale", type=float, default=SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--kernels",
        action="store_true",
        help="also record the kernel table of one step per arm",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    args.ops = [o for o in args.ops.split(",") if o]
    unknown = [a for a in args.arms if a not in ARMS] + [
        o for o in args.ops if o not in OPS
    ]
    if unknown:
        parser.error(f"unknown arms / ops: {unknown}")
    if not (math.isfinite(args.softmax_scale) and args.softmax_scale > 0):
        parser.error("--softmax-scale must be a positive finite float")
    if args.fa4_token_chunk < 0:
        parser.error("--fa4-token-chunk must be >= 0")
    results = run(args)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
