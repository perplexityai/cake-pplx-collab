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

Non-Cake baselines ("arms") of the Cake KDA training benchmarks ``bench_cake_kda_baselines.py`` (one GPU) and
``bench_cake_kda_kcp_baselines.py`` (context parallel).

Single-GPU arms, called through autograd on the benchmark's own tensors (the packed documents as one THD batch, no
copy):

* ``cudnn_bi1``: the cuDNN frontend's ``cudnn.linear_attention.ops.kimi_delta_attention`` with
  ``plan_name="kda_frost"``, ``batch_invariant=True`` (each sequence's results are bitwise independent of the batch
  composition), ``checkpoint_every_n_tokens=16`` and ``output_context=True``: the baseline of the deterministic
  Cake package;
* ``cudnn_bi0``: the same with ``batch_invariant=False``: the baseline of the non-deterministic Cake package;
* ``fla``: flash-linear-attention's ``fla.ops.kda.chunk_kda`` (Triton; ``FLA_FLASH_KDA=0`` and ``FLA_TILELANG=0``
  while the arm runs), a reference column.

Every arm takes the Cake operands as they are: BF16 ``q``, ``k``, ``v`` ``[S, H, 128]``, BF16 ``beta`` after the
sigmoid (``use_beta_sigmoid_in_kernel=False``), scale ``128 ** -0.5``, a zero initial state and no final state,
the int32 ``cu_seqlens`` (int64 for FLA).  The forget gate: the activated FP32 gate goes to cuDNN as
``gate_domain="log"`` and to FLA as ``use_gate_in_kernel=False, safe_gate=True``; the raw gate mode (the BF16 raw
gate, FP32 ``A_log [H]`` and ``dt_bias [H * 128]``, the lower bound) goes to cuDNN as ``safe_gate=True,
a_log=A_log, dt_bias=dt_bias.view(H, 128), gate_lower_bound=lower_bound`` and to FLA as
``use_gate_in_kernel=True, safe_gate=True, A_log, dt_bias, lower_bound``.  Both compute the Cake gate
``lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))``.  The in-kernel Q/K L2 norm is
``use_qk_l2norm_in_kernel`` on every arm.

Context-parallel arms (``cudnn_cp_bi1`` / ``cudnn_cp_bi0``, the ``batch_invariant`` of the single-GPU arm of the
same name): the cuDNN frontend's summary operations on the Cake partition's contiguous token ranges, one rank per
range.  Forward: each rank summarizes its last local sequence (``kimi_delta_attention_summary``, from a zero
state), the summaries are all-gathered, and each rank runs ``kimi_delta_attention`` with the summaries of the
ranks that hold earlier pieces of its first sequence (``preceding_state_update`` / ``preceding_transition``,
folded on the device).  Backward: each rank summarizes its first local sequence
(``kimi_delta_attention_summary_bwd``, a zero outgoing state gradient, with its context), the summaries are
all-gathered, and each rank runs ``torch.ops.cudnn.kimi_delta_attention_bwd`` from the forward's composed state
with the summaries of the ranks that hold later pieces of its last sequence (``following_state_grad`` /
``following_transition``).  ``dA_log`` / ``ddt_bias`` are each rank's partial sums; the benchmark sums them over the
ranks.  This is the call sequence of the cuDNN frontend's context-parallel KDA sample.  Other layouts (zigzag) are
unsupported.

Each arm records its library and version; an arm that cannot run raises :class:`Unsupported` with the reason.
"""

from __future__ import annotations

import bisect
import contextlib
import importlib
import importlib.metadata
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Optional

import torch

D = 128
SCALE = D**-0.5
# The cuDNN arms' state checkpoint cadence and plan.
CUDNN_CHECKPOINT_EVERY_N_TOKENS = 16
CUDNN_PLAN = "kda_frost"
CUDNN_ENTRY = "cudnn.linear_attention.ops.kimi_delta_attention"
FLA_ENTRY = "fla.ops.kda.chunk_kda"
SINGLE_GPU_ARMS = ("cudnn_bi1", "cudnn_bi0", "fla")
CP_ARMS = ("cudnn_cp_bi1", "cudnn_cp_bi0")
# The Cake arm each baseline is paired with in the speedup columns (baseline time / Cake time).
PAIRED_CAKE_ARM = {
    "cudnn_bi1": "cake_det",
    "cudnn_bi0": "cake_nondet",
    "cudnn_cp_bi1": "cake_det_kcp",
    "cudnn_cp_bi0": "cake_nondet_kcp",
}
# While the FLA arm runs: its environment (the Triton chunk kernels, no FlashKDA or TileLang route).
FLA_ENV: dict[str, Optional[str]] = {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}


class Unsupported(Exception):
    """The arm cannot run the row; the message is the concrete reason."""


@contextlib.contextmanager
def environment(values: dict[str, Optional[str]]) -> Iterator[None]:
    """Set (or unset, for ``None``) environment variables for the block, then restore them."""
    saved = {key: os.environ.get(key) for key in values}

    def apply(settings: dict[str, Optional[str]]) -> None:
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    apply(values)
    try:
        yield
    finally:
        apply(saved)


def _distribution(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module(name: str, package: str):
    """The imported module; :class:`Unsupported` naming the import error when it does not import."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 -- any import failure means the arm is absent
        raise Unsupported(
            f"{package} is not importable: {type(exc).__name__}: {exc}"
        ) from exc


def library_versions() -> dict[str, Any]:
    """The versions of the baselines' libraries in this environment (``None`` where absent)."""
    out: dict[str, Any] = {
        name: _distribution(name)
        for name in (
            "nvidia-cudnn-frontend",
            "nvidia-cudnn-cu13",
            "nvidia-cudnn-cu12",
            "nvidia-cutlass-dsl",
            "flash-linear-attention",
            "fla-core",
            "triton",
        )
    }
    try:
        import cudnn

        out["cudnn.__version__"] = getattr(cudnn, "__version__", None)
        try:
            out["cudnn.backend_version()"] = cudnn.backend_version()
        except Exception as exc:  # noqa: BLE001
            out["cudnn.backend_version()"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["cudnn_import_error"] = f"{type(exc).__name__}: {exc}"
    with environment(FLA_ENV):
        try:
            import fla

            out["fla.__version__"] = getattr(fla, "__version__", None)
        except Exception as exc:  # noqa: BLE001
            out["fla_import_error"] = f"{type(exc).__name__}: {exc}"
    return out


def arm_label(arm: str, versions: dict[str, Any]) -> str:
    """The arm's library and version, for the table's legend."""
    if arm.startswith("cudnn"):
        bi = arm.endswith("bi1")
        return (
            f"cuDNN frontend {versions.get('cudnn.__version__') or versions.get('nvidia-cudnn-frontend')} "
            f"(backend {versions.get('cudnn.backend_version()')}), {CUDNN_ENTRY}, plan {CUDNN_PLAN}, "
            f"batch_invariant={bi}, checkpoint_every_n_tokens={CUDNN_CHECKPOINT_EVERY_N_TOKENS}"
            + (", context parallel (summary operations)" if "_cp_" in arm else "")
        )
    if arm == "fla":
        return (
            f"flash-linear-attention {versions.get('flash-linear-attention') or versions.get('fla.__version__')}, "
            f"{FLA_ENTRY}, FLA_FLASH_KDA=0 FLA_TILELANG=0"
        )
    raise ValueError(f"unknown arm {arm!r}")


def _is_raw(inputs: dict) -> bool:
    return "A_log" in inputs


def leaf_names(inputs: dict) -> tuple[str, ...]:
    """The autograd leaves of a row: ``q, k, v, g, beta`` and in the raw gate mode ``A_log, dt_bias``."""
    return ("q", "k", "v", "g", "beta") + (
        ("A_log", "dt_bias") if _is_raw(inputs) else ()
    )


def cudnn_gate_kwargs(
    raw: bool, l2n: bool, batch_invariant: bool, lower_bound: float
) -> dict[str, Any]:
    """The keywords of a cuDNN call that set the row's gate, Q/K norm and beta forms (no tensors)."""
    kw: dict[str, Any] = {
        "use_qk_l2norm_in_kernel": bool(l2n),
        "use_beta_sigmoid_in_kernel": False,
        "batch_invariant": bool(batch_invariant),
    }
    if raw:
        kw.update(safe_gate=True, gate_lower_bound=float(lower_bound))
    else:
        kw.update(gate_domain="log", safe_gate=False)
    return kw


def _cudnn_ops():
    ops = _module(
        "cudnn.linear_attention.ops", "the cuDNN frontend (nvidia-cudnn-frontend)"
    )
    if not hasattr(ops, "kimi_delta_attention"):
        raise Unsupported(f"{CUDNN_ENTRY} is absent from this cuDNN frontend")
    return ops


def _check_device(device: torch.device) -> None:
    major, minor = torch.cuda.get_device_capability(device)
    sm = 10 * major + minor
    if not (100 <= sm <= 103 or sm == 107):
        raise Unsupported(
            f"the {CUDNN_PLAN} plan targets SM 100-103 and 107, not SM {sm}"
        )


def single_gpu_arm(
    arm: str, inputs: dict, lower_bound: float, l2n: bool
) -> tuple[Callable[..., torch.Tensor], dict]:
    """``(call, info)`` of one single-GPU arm: ``call(*leaves)`` (the leaves of :func:`leaf_names`) returns ``out``
    ``[S, H, 128]`` through autograd; ``info`` names the entry and its settings."""
    raw = _is_raw(inputs)
    cu = inputs["cu_seqlens"]
    H = inputs["q"].shape[-2]
    if arm in ("cudnn_bi1", "cudnn_bi0"):
        ops = _cudnn_ops()
        _check_device(cu.device)
        kw = cudnn_gate_kwargs(raw, l2n, arm == "cudnn_bi1", lower_bound)
        kw.update(
            checkpoint_every_n_tokens=CUDNN_CHECKPOINT_EVERY_N_TOKENS,
            plan_name=CUDNN_PLAN,
            output_context=True,
        )
        info = {"entry": CUDNN_ENTRY, **kw}

        def call(q, k, v, g, beta, *params):
            extra = {"a_log": params[0], "dt_bias": params[1].view(H, D)} if raw else {}
            return ops.kimi_delta_attention(q, k, v, g, beta, cu, SCALE, **kw, **extra)[
                0
            ]

        return call, info
    if arm == "fla":
        with environment(FLA_ENV):
            mod = _module("fla.ops.kda", "flash-linear-attention")
        chunk_kda = mod.chunk_kda
        cu_long = cu.long()
        info = {
            "entry": FLA_ENTRY,
            "safe_gate": True,
            "cu_seqlens": "int64",
            "env": dict(FLA_ENV),
        }

        def call(q, k, v, g, beta, *params):
            gate = (
                {
                    "use_gate_in_kernel": True,
                    "A_log": params[0],
                    "dt_bias": params[1],
                    "lower_bound": lower_bound,
                }
                if raw
                else {"use_gate_in_kernel": False}
            )
            with environment(FLA_ENV):
                o, _ = chunk_kda(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    g.unsqueeze(0),
                    beta.unsqueeze(0),
                    scale=SCALE,
                    initial_state=None,
                    output_final_state=False,
                    use_qk_l2norm_in_kernel=bool(l2n),
                    use_beta_sigmoid_in_kernel=False,
                    safe_gate=True,
                    cu_seqlens=cu_long,
                    **gate,
                )
            return o.squeeze(0)

        return call, info
    raise ValueError(f"unknown arm {arm!r}; one of {SINGLE_GPU_ARMS}")


# ---------------------------------------------------------------------------------------------------------------------
# Context parallel
# ---------------------------------------------------------------------------------------------------------------------


class CudnnCpPlan:
    """One rank's view of the contiguous split: its token range, its local ``cu_seqlens`` (every local sequence, the
    first one alone, the last one alone) and the ranks whose summaries its forward and backward fold.

    ``fragments_by_rank``: per rank its ``(doc, offset, length)`` fragments in token order (the Cake partition's
    fragments); the ranks' token ranges must be consecutive and cover the batch in order (the contiguous layout)."""

    def __init__(
        self,
        fragments_by_rank,
        cu: Sequence[int],
        rank: int,
        group,
        device: torch.device,
    ) -> None:
        W = len(fragments_by_rank)
        starts, stops = [], []
        pos = 0
        for r, frags in enumerate(fragments_by_rank):
            live = [f for f in frags if f[2] > 0]
            if not live:
                raise Unsupported(
                    f"rank {r} holds no tokens; the cuDNN context-parallel arm needs every rank to hold some"
                )
            first = cu[live[0][0]] + live[0][1]
            if first != pos:
                raise Unsupported(
                    "the cuDNN context-parallel arm takes the contiguous layout only"
                )
            for doc, off, length in live:
                if cu[doc] + off != pos:
                    raise Unsupported(
                        "the cuDNN context-parallel arm takes the contiguous layout only"
                    )
                pos += length
            starts.append(first)
            stops.append(pos)
        if pos != cu[-1]:
            raise Unsupported("the ranks' token ranges do not cover the batch")
        self.rank, self.world, self.group = rank, W, group
        start, stop = starts[rank], stops[rank]
        bounds = list(cu)
        local = [0] + [b - start for b in bounds if start < b < stop] + [stop - start]
        self.tokens = stop - start
        self.cu_local = torch.tensor(local, dtype=torch.int32, device=device)
        self.cu_first = self.cu_local[:2]
        self.cu_last = self.cu_local[-2:]

        def rank_of(token: int) -> int:
            return bisect.bisect_right(starts, token) - 1

        first_seq = max(i for i in range(len(bounds) - 1) if bounds[i] <= start)
        last_seq = max(i for i in range(len(bounds) - 1) if bounds[i] < stop)
        # the rank holding the first token of this rank's first sequence; one past the rank holding the last token of
        # its last sequence
        self.preceding_start = rank_of(bounds[first_seq])
        self.following_stop = rank_of(bounds[last_seq + 1] - 1) + 1


def _all_gather(x: torch.Tensor, plan: CudnnCpPlan) -> torch.Tensor:
    import torch.distributed as dist

    x = x.contiguous()
    out = torch.empty((plan.world,) + tuple(x.shape), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x, group=plan.group)
    return out


class CudnnCp:
    """The cuDNN context-parallel arm of one rank: :meth:`forward` / :meth:`backward` (explicit) and :meth:`step`
    (an autograd call, for ``torch.utils.checkpoint``)."""

    def __init__(
        self,
        arm: str,
        plan: CudnnCpPlan,
        raw: bool,
        l2n: bool,
        lower_bound: float,
        H: int,
    ) -> None:
        if arm not in CP_ARMS:
            raise ValueError(f"unknown arm {arm!r}; one of {CP_ARMS}")
        self.ops = _cudnn_ops()
        for name in (
            "kimi_delta_attention_summary",
            "kimi_delta_attention_summary_bwd",
        ):
            if not hasattr(self.ops, name):
                raise Unsupported(
                    f"cudnn.linear_attention.ops.{name} is absent from this cuDNN frontend"
                )
        _check_device(plan.cu_local.device)
        self.plan, self.raw, self.H = plan, raw, H
        self.kw = cudnn_gate_kwargs(raw, l2n, arm == "cudnn_cp_bi1", lower_bound)
        self.info = {
            "entry": "kimi_delta_attention_summary + all_gather + kimi_delta_attention(preceding_*); "
            "kimi_delta_attention_summary_bwd + all_gather + kimi_delta_attention_bwd(following_*)",
            "checkpoint_every_n_tokens": CUDNN_CHECKPOINT_EVERY_N_TOKENS,
            "plan_name": CUDNN_PLAN,
            **self.kw,
        }

    def _params(self, params) -> dict[str, torch.Tensor]:
        return (
            {"a_log": params[0], "dt_bias": params[1].view(self.H, D)}
            if self.raw
            else {}
        )

    def forward(self, q, k, v, g, beta, *params) -> tuple[torch.Tensor, tuple]:
        """This rank's ``out`` and the state its backward takes."""
        p, extra = self.plan, self._params(params)
        h, m = self.ops.kimi_delta_attention_summary(
            k, v, g, beta, p.cu_last, plan_name=None, **self.kw, **extra
        )
        hs, ms = _all_gather(h[0], p), _all_gather(m[0], p)
        o, _, ckpts, composed = self.ops.kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            p.cu_local,
            SCALE,
            preceding_state_update=hs[p.preceding_start : p.rank],
            preceding_transition=ms[p.preceding_start : p.rank],
            checkpoint_every_n_tokens=CUDNN_CHECKPOINT_EVERY_N_TOKENS,
            plan_name=CUDNN_PLAN,
            **self.kw,
            **extra,
        )[:4]
        return o, (composed, ckpts)

    def backward(
        self, state, dout, q, k, v, g, beta, *params
    ) -> tuple[torch.Tensor, ...]:
        """This rank's ``(dq, dk, dv, dg, dbeta)`` and in the raw gate mode its partial ``(dA_log, ddt_bias)``."""
        p, extra = self.plan, self._params(params)
        composed, ckpts = state
        gg, tt, context = self.ops.kimi_delta_attention_summary_bwd(
            q,
            k,
            g,
            beta,
            dout,
            p.cu_first,
            scale=SCALE,
            output_transition=True,
            output_context=True,
            checkpoint_every_n_tokens=CUDNN_CHECKPOINT_EVERY_N_TOKENS,
            **self.kw,
            **extra,
        )
        gs, ts = _all_gather(gg[0], p), _all_gather(tt[0], p)
        res = torch.ops.cudnn.kimi_delta_attention_bwd(
            dout,
            q,
            k,
            v,
            g,
            beta,
            p.cu_local,
            SCALE,
            composed_initial_state=composed if composed.numel() else None,
            following_state_grad=gs[p.rank + 1 : p.following_stop],
            following_transition=ts[p.rank + 1 : p.following_stop],
            state_checkpoints=ckpts if ckpts.numel() else None,
            checkpoint_every_n_tokens=CUDNN_CHECKPOINT_EVERY_N_TOKENS,
            context=context,
            plan_name=CUDNN_PLAN,
            **self.kw,
            **extra,
        )
        grads = tuple(res[:5])
        if self.raw:
            grads += (res[6], res[7].reshape(params[1].shape))
        return grads

    def step(self, *leaves) -> torch.Tensor:
        """``out`` through autograd (the backward runs :meth:`backward`)."""
        return _CudnnCpFunction.apply(self, *leaves)


class _CudnnCpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, arm: CudnnCp, *leaves):
        o, state = arm.forward(*leaves)
        ctx.arm, ctx.state = arm, state
        ctx.save_for_backward(*leaves)
        return o

    @staticmethod
    def backward(ctx, dout):
        leaves = ctx.saved_tensors
        grads = ctx.arm.backward(ctx.state, dout.contiguous(), *leaves)
        return (None, *grads)


__all__ = [
    "CP_ARMS",
    "CUDNN_CHECKPOINT_EVERY_N_TOKENS",
    "CUDNN_PLAN",
    "CudnnCp",
    "CudnnCpPlan",
    "FLA_ENV",
    "PAIRED_CAKE_ARM",
    "SINGLE_GPU_ARMS",
    "Unsupported",
    "arm_label",
    "cudnn_gate_kwargs",
    "environment",
    "leaf_names",
    "library_versions",
    "single_gpu_arm",
]
