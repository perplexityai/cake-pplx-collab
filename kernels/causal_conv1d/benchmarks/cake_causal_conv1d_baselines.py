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

Non-Cake baselines ("arms") of the Cake causal conv1d + SiLU training benchmarks
``bench_cake_causal_conv1d_nondet_train.py`` (single rank) and ``bench_cake_causal_conv1d_nondet_train_kcp.py``
(context parallel).

Every arm is the public training entry point of another library, called through autograd on the benchmark's own
tensors without a copy (``[S, C]`` maps to ``[1, S, C]`` or to its channel-last ``[1, C, S]`` view).  Index tensors
and contexts are built once per row, outside the timing.  The phases are the benchmark's: ``fwd`` (the training
forward, its graph recorded and dropped), ``bwd`` (the backward of a recorded forward, ``retain_graph=True``),
``step`` (forward + backward, single rank) and ``ckpt`` (``torch.utils.checkpoint`` around the entry, then its
backward); the leaves' gradients are reset before every backward.

Single-rank arms (the packed documents as one ``B = 1`` batch):

* ``fla``: flash-linear-attention ``fla.modules.causal_conv1d.causal_conv1d`` (``cu_seqlens``), its default Triton
  backend (``FLA_CONV_GLUON`` and ``FLA_GLUON`` unset while the arm runs);
* ``fla_gluon``: the same entry point with its opt-in Gluon backend (``FLA_CONV_GLUON=1``; the Gluon verifiers
  must accept the call, else the arm is unsupported);
* ``dao``: Dao-AILab ``causal_conv1d.causal_conv1d_fn`` on the channel-last view with a per-token ``seq_idx``,
  its default atomic weight gradient (``CAUSAL_CONV1D_DETERMINISTIC`` unset);
* ``dao_det``: the same with ``CAUSAL_CONV1D_DETERMINISTIC=1`` (an ordered weight-gradient reduction);
* ``cudnn_fe``: the cuDNN frontend's ``cudnn.ops.causal_conv1d`` on the channel-last view with ``cu_seqlens`` (its
  default route; with ``torch`` deterministic algorithms off, an atomic weight gradient);
* ``cudnn_fe_det``: the same with ``torch.use_deterministic_algorithms(True)`` held around its calls (bias-free rows
  only: the library's deterministic weight gradient);
* ``torch_compile``: ``torch.compile`` of the document-local formula in FP32 (FP32 taps and accumulation, SiLU' at the
  FP32 preactivation, one BF16 rounding of ``y``), the token shifts masked at the document boundaries.

Context-parallel arms (the benchmark's process group; the halo exchange is part of every phase):

* ``fla_cp_triton`` / ``fla_cp_cuda``: FLA's ``causal_conv1d(..., cp_context=build_cp_context(...),
  backend="triton" | "cuda")`` on FLA's own partition of the pack, padded with one zero document to a multiple of
  ``2 W`` tokens (``contiguous`` and ``zigzag``; ``cuda`` runs Dao's kernel on the ``contiguous`` layout only);
* ``attn_gym_cp``: Attention Gym's ``context_parallel_conv_history`` (the fragment tails all-gathered) feeding
  ``attn_gym.linear.short_conv.causal_conv1d`` (CuTe DSL), on the benchmark's own fragment table (every layout).
  Its taps must have the activations' dtype, so it runs BF16 taps against the FP32 taps of the Cake rows.

Each arm records the library version (``fla.__version__``, ``causal_conv1d.__version__``, ``cudnn.__version__``;
Attention Gym has none, so ``ATTN_GYM_COMMIT`` names its commit), its determinism claim, and its features against the
Cake route of the row: ``exact``; ``more_precise`` / ``other_policy`` (SiLU' at the FP32 preactivation where the Cake
route takes the BF16-rounded one, or the other way round); ``less_precise`` (BF16 taps against FP32 taps: another
operation, never compared).  The accuracy gate (``cake_causal_conv1d_numerics.py``) judges every arm as the training
tests judge a kernel: its own error against a float64 oracle beside the error of PyTorch BF16, over 32 random inputs
(:func:`single_rank_gate`, :func:`kcp_gate`); it never compares an arm with a Cake result.  An arm is ``eligible``
for the best baseline when its taps match the Cake route and it passes the gate; a SiLU' policy that differs from the
Cake route's is a note, since the gate decides whether its precision qualifies.  The gate also records whether the
arm's outputs were the same bytes run to run on the row (``run_to_run_bitwise``); a ``det`` comparison counts an arm
only when it claims and shows that.  A row an arm cannot run is reported with the concrete reason.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cake_causal_conv1d_numerics as numerics  # noqa: E402
import cake_naive  # noqa: E402

WIDTH = 4
SINGLE_RANK_ARMS = ("fla", "fla_gluon", "dao", "dao_det", "cudnn_fe", "cudnn_fe_det", "torch_compile")
KCP_ARMS = ("fla_cp_triton", "fla_cp_cuda", "attn_gym_cp")
# The arms whose library claims an ordered (run-to-run bitwise) weight gradient.
DETERMINISTIC_CLAIM = ("fla", "fla_gluon", "dao_det", "cudnn_fe_det", "fla_cp_triton")
FLA_ENTRY = "fla.modules.causal_conv1d.causal_conv1d"
DAO_ENTRY = "causal_conv1d.causal_conv1d_fn"
CUDNN_ENTRY = "cudnn.ops.causal_conv1d"
ATTN_GYM_ENTRY = (
    "attn_gym.linear.context_parallel.context_parallel_conv_history + "
    "attn_gym.linear.short_conv.causal_conv1d"
)
# While an arm runs: its environment (None: unset).
_FLA_TRITON_ENV: dict[str, Optional[str]] = {"FLA_CONV_GLUON": None, "FLA_GLUON": None}
_FLA_GLUON_ENV: dict[str, Optional[str]] = {"FLA_CONV_GLUON": "1", "FLA_GLUON": None}
_DAO_ENV: dict[str, Optional[str]] = {"CAUSAL_CONV1D_DETERMINISTIC": None}
_DAO_DET_ENV: dict[str, Optional[str]] = {"CAUSAL_CONV1D_DETERMINISTIC": "1"}


class Unsupported(Exception):
    """The arm cannot run the row; the message is the concrete reason."""


# Not an environment variable: while an arm's ``env`` holds this key with a true value, torch deterministic
# algorithms are on (``cudnn_fe_det``).
TORCH_DETERMINISTIC = "torch.use_deterministic_algorithms"


@contextlib.contextmanager
def environment(values: dict[str, Optional[str]]) -> Iterator[None]:
    """Set (or unset, for ``None``) environment variables for the block, then restore them; the
    :data:`TORCH_DETERMINISTIC` key sets torch deterministic algorithms instead."""

    values = dict(values)
    torch_det = values.pop(TORCH_DETERMINISTIC, None)
    saved = {key: os.environ.get(key) for key in values}
    saved_det = torch.are_deterministic_algorithms_enabled()

    def apply(settings: dict[str, Optional[str]]) -> None:
        for key, value in settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    apply(values)
    if torch_det is not None:
        torch.use_deterministic_algorithms(bool(torch_det))
    try:
        yield
    finally:
        apply(saved)
        if torch_det is not None:
            torch.use_deterministic_algorithms(saved_det)


def dtype_name(dtype: Optional[torch.dtype]) -> Optional[str]:
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _library(module: str, package: str) -> str:
    """The library's version; ``Unsupported`` when it is not importable."""

    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 -- any import failure means the arm is absent
        raise Unsupported(
            f"{package} is not importable: {type(exc).__name__}: {exc}"
        ) from exc
    version = getattr(mod, "__version__", None)
    if version is None and module == "attn_gym":
        version = os.environ.get("ATTN_GYM_COMMIT") or _distribution("attn-gym")
    return f"{package} {version or 'unknown'}"


def _distribution(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _fla_distribution() -> str:
    """The installed distribution of ``fla``: ``fla-core`` (the kernels alone) or the ``flash-linear-attention`` one."""

    return "fla-core" if _distribution("fla-core") else "flash-linear-attention"


def feature(
    *,
    preactivation: str,
    taps: Optional[torch.dtype],
    bias: Optional[torch.dtype],
    cake_preactivation: str,
    cake_taps: torch.dtype,
) -> dict:
    """The arm's features on a row and their match against the Cake route of the row."""

    match, notes = "exact", []
    if preactivation != cake_preactivation:
        match = "more_precise" if preactivation == "fp32" else "other_policy"
        notes.append(
            f"SiLU' at the {preactivation} preactivation (Cake route: {cake_preactivation}); the gate decides"
        )
    if taps != cake_taps:
        match = "less_precise"
        notes.append(
            f"{dtype_name(taps)} taps (Cake route: {dtype_name(cake_taps)} taps)"
        )
    return {
        "taps": dtype_name(taps),
        "bias": dtype_name(bias),
        "preactivation": preactivation,
        "match": match,
        "notes": notes,
    }


class Calls(dict):
    """An arm's call table (``{name: fn}``) with the ``leaves`` and the output gradients ``dys`` its calls read
    (the accuracy gate refills them in place with each draw's inputs)."""

    leaves: list[torch.Tensor]
    dys: list[torch.Tensor]
    layer: Callable[..., tuple]
    release: Callable[[], None]


def autograd_calls(
    layer: Callable[..., tuple],
    leaves: Sequence[torch.Tensor],
    dys: Sequence[torch.Tensor],
) -> Calls:
    """``{name: fn}`` of an arm through autograd: the timed phases ``fwd`` / ``bwd`` / ``step`` / ``ckpt`` and
    ``record`` (the forward ``bwd`` replays the backward of)."""

    state: dict[str, tuple] = {}

    def reset() -> None:
        for leaf in leaves:
            leaf.grad = None

    def grads() -> list[Optional[torch.Tensor]]:
        return [leaf.grad for leaf in leaves]

    def fwd():
        return layer(*leaves)

    def record():
        state["ys"] = layer(*leaves)
        return state["ys"]

    def bwd():
        reset()
        torch.autograd.backward(list(state["ys"]), list(dys), retain_graph=True)
        return grads()

    def step():
        reset()
        torch.autograd.backward(list(layer(*leaves)), list(dys))
        return grads()

    def ckpt():
        reset()
        ys = checkpoint(layer, *leaves, use_reentrant=False)
        torch.autograd.backward(list(ys), list(dys))
        return grads()

    def release() -> None:
        state.clear()
        reset()

    calls = Calls(fwd=fwd, record=record, bwd=bwd, step=step, ckpt=ckpt)
    calls.leaves = list(leaves)
    calls.dys = list(dys)
    calls.layer = layer
    calls.release = release
    return calls


def gate_run(layer: Callable[..., tuple], env: dict) -> numerics.ArmRun:
    """The accuracy gate's run of a single-rank arm: its forward and backward on fresh leaves of one draw."""

    def run(x, w, b, dy):
        leaves = [t.detach().clone().requires_grad_(True) for t in (x, w, b) if t is not None]
        with environment(env):
            (y,) = layer(*leaves)
            torch.autograd.backward([y], [dy])
        grads = [leaf.grad for leaf in leaves]
        return {
            "y": y.detach(),
            "dx": grads[0],
            "dweight": grads[1],
            "dbias": grads[2] if b is not None else None,
        }

    return run


def apply_gate(entry: dict, verdict: dict) -> None:
    """Record the gate's ``verdict`` on an arm's ``entry`` and set ``eligible`` (taps match and the gate passed) and
    ``measured_deterministic`` (claimed and run-to-run bitwise on the row)."""

    entry["gate"] = verdict
    entry["eligible"] = (
        "unsupported" not in entry
        and entry.get("feature", {}).get("match") != "less_precise"
        and bool(verdict.get("passed"))
    )
    entry["measured_deterministic"] = bool(
        entry.get("name") in DETERMINISTIC_CLAIM and verdict.get("run_to_run_bitwise")
    )


def _bias(leaves: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    return leaves[2] if len(leaves) > 2 else None


# ---------------------------------------------------------------------------------------------------------------------
# Single rank
# ---------------------------------------------------------------------------------------------------------------------


def _fla_layer(cu: torch.Tensor) -> Callable[..., tuple]:
    from fla.modules.causal_conv1d import causal_conv1d  # type: ignore

    cu_long = cu.long()
    cu_cpu = cu_long.cpu()

    def layer(*t: torch.Tensor) -> tuple:
        y, _ = causal_conv1d(
            t[0].unsqueeze(0),
            t[1],
            bias=_bias(t),
            activation="silu",
            cu_seqlens=cu_long,
            cu_seqlens_cpu=cu_cpu,
        )
        return (y.squeeze(0),)

    return layer


def _gluon_verdict(x: torch.Tensor, w: torch.Tensor, cu: torch.Tensor) -> None:
    """``Unsupported`` unless FLA's dispatch would route this call's forward and backward to Gluon."""

    try:
        from fla.backends import GluonBackend  # type: ignore
        from fla.modules.causal_conv1d.backends import gluon  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise Unsupported(
            f"FLA Gluon backend not importable: {type(exc).__name__}: {exc}"
        ) from exc
    if not GluonBackend.is_available():
        raise Unsupported("FLA GluonBackend.is_available() is False")
    with environment(_FLA_GLUON_ENV):
        if not GluonBackend(env_var="FLA_CONV_GLUON").is_enabled():
            raise Unsupported("FLA Gluon backend not enabled by FLA_CONV_GLUON=1")
    x3 = x.unsqueeze(0)
    for verifier, args in (
        (gluon.causal_conv1d_fwd_verifier, dict(x=x3, weight=w, cu_seqlens=cu)),
        (
            gluon.causal_conv1d_bwd_verifier,
            dict(x=x3, dy=x3, dht=None, weight=w, cu_seqlens=cu),
        ),
    ):
        accepted, reason = verifier(**args)
        if not accepted:
            raise Unsupported(
                f"FLA Gluon {verifier.__name__} rejects the call: {reason}"
            )


def _dao_layer(x: torch.Tensor, lens: Sequence[int]) -> Callable[..., tuple]:
    from causal_conv1d import causal_conv1d_fn  # type: ignore

    # One document id per token (int32 [1, S]), built once: the channel-last kernels take seq_idx, not cu_seqlens.
    seq_idx = (
        torch.repeat_interleave(
            torch.arange(len(lens), dtype=torch.int32, device=x.device),
            torch.tensor(list(lens), device=x.device),
        )
        .unsqueeze(0)
        .contiguous()
    )

    def layer(*t: torch.Tensor) -> tuple:
        y = causal_conv1d_fn(
            t[0].unsqueeze(0).transpose(1, 2),
            t[1],
            _bias(t),
            seq_idx=seq_idx,
            activation="silu",
        )
        return (y.transpose(1, 2).squeeze(0),)

    return layer


def _cudnn_layer(cu: torch.Tensor) -> tuple[Callable[..., tuple], Callable[[], Any]]:
    ops = importlib.import_module("cudnn.ops.causal_conv1d")

    def layer(*t: torch.Tensor) -> tuple:
        y = ops.causal_conv1d(
            t[0].unsqueeze(0).transpose(1, 2), t[1], _bias(t), "silu", cu_seqlens=cu
        )
        return (y.transpose(1, 2).squeeze(0),)

    return layer, ops._get_causal_conv1d_last_route


def _compile_layer(x: torch.Tensor, lens: Sequence[int]) -> Callable[..., tuple]:
    """``torch.compile`` of the document-local formula in FP32 (one BF16 rounding of ``y``)."""

    ids = torch.repeat_interleave(
        torch.arange(len(lens), device=x.device), torch.tensor(list(lens), device=x.device)
    )
    # same[k][t]: token t - k exists and is of token t's document.
    same = [None] + [
        torch.cat([torch.zeros(k, dtype=torch.bool, device=x.device), ids[k:] == ids[:-k]]).unsqueeze(1)
        for k in range(1, WIDTH)
    ]

    def formula(xb, w, b, m1, m2, m3):
        xf = xb.float()
        wf = w.float()
        pre = xf * wf[:, WIDTH - 1]
        for k, m in ((1, m1), (2, m2), (3, m3)):
            shifted = torch.nn.functional.pad(xf[:-k], (0, 0, k, 0)) * m
            pre = pre + shifted * wf[:, WIDTH - 1 - k]
        if b is not None:
            pre = pre + b.float()
        return (pre * torch.sigmoid(pre)).to(xb.dtype)

    compiled = torch.compile(formula, dynamic=False)

    def layer(*t: torch.Tensor) -> tuple:
        return (compiled(t[0], t[1], _bias(t), same[1], same[2], same[3]),)

    return layer


def single_rank_arm(
    name: str,
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    dy: torch.Tensor,
    cu: torch.Tensor,
    lens: Sequence[int],
    cake_preactivation: str,
) -> tuple[dict, Optional[Calls], Optional[numerics.ArmRun]]:
    """``(entry, calls, run)`` of one single-rank arm on a row: ``entry`` describes it (library, version,
    determinism claim, features and ``env``, the environment to hold around its calls; or ``unsupported`` with the
    reason); ``calls`` (``None`` when unsupported) are its :func:`autograd_calls` on the row's tensors, recorded
    once; ``run`` is its accuracy-gate run (:func:`gate_run`).  ``eligible`` is set by :func:`apply_gate`."""

    entry: dict = {"name": name, "eligible": False}
    try:
        env: dict[str, Optional[str]] = {}
        route_of: Optional[Callable[[], Any]] = None
        if name in ("fla", "fla_gluon"):
            entry.update(
                library=_library("fla", _fla_distribution()),
                entry=FLA_ENTRY + (" [Gluon]" if name == "fla_gluon" else " [Triton]"),
                determinism="deterministic (FP32 per-chunk partials, ordered reduce)",
                preactivation="bf16",
            )
            env = _FLA_GLUON_ENV if name == "fla_gluon" else _FLA_TRITON_ENV
            if name == "fla_gluon":
                _gluon_verdict(x, w, cu)
            layer = _fla_layer(cu)
        elif name in ("dao", "dao_det"):
            entry.update(
                library=_library("causal_conv1d", "causal-conv1d"),
                entry=DAO_ENTRY + " (channel-last view, seq_idx)",
                determinism="deterministic (CAUSAL_CONV1D_DETERMINISTIC=1)"
                if name == "dao_det"
                else "non-deterministic (FP32 atomic dweight / dbias)",
                preactivation="fp32",
            )
            if b is not None and b.dtype != w.dtype:
                raise Unsupported(
                    f"causal_conv1d_fn needs the bias dtype to equal the weight dtype "
                    f"({dtype_name(b.dtype)} bias, {dtype_name(w.dtype)} taps)"
                )
            env = _DAO_DET_ENV if name == "dao_det" else _DAO_ENV
            layer = _dao_layer(x, lens)
        elif name in ("cudnn_fe", "cudnn_fe_det"):
            entry.update(
                library=_library("cudnn", "nvidia-cudnn-frontend"),
                cutlass_dsl=_distribution("nvidia-cutlass-dsl"),
                entry=CUDNN_ENTRY + " (channel-last view, cu_seqlens)",
                determinism="deterministic dweight (torch deterministic algorithms on, bias-free)"
                if name == "cudnn_fe_det"
                else "non-deterministic (atomic dweight; torch deterministic algorithms off)",
                preactivation="unverified",
            )
            if name == "cudnn_fe_det":
                if b is not None:
                    raise Unsupported("cuDNN-FE's deterministic weight gradient is bias-free only")
                env = {TORCH_DETERMINISTIC: "1"}
            arch = importlib.import_module("cudnn._causal_conv1d_arch")
            capability = tuple(torch.cuda.get_device_capability(x.device))
            if not arch.is_functional_arch(capability):
                raise Unsupported(
                    f"compute capability {capability[0]}.{capability[1]} is not in cuDNN-FE's causal_conv1d "
                    "FUNCTIONAL_COMPUTE_CAPABILITIES"
                )
            if b is not None and w.dtype != torch.bfloat16:
                raise Unsupported(
                    "cuDNN-FE's packed route takes a bias only with BF16 taps (FP32 taps: bias-free only)"
                )
            if b is not None and b.dtype != torch.bfloat16:
                raise Unsupported("cuDNN-FE's packed route takes a BF16 bias only")
            layer, route_of = _cudnn_layer(cu)
        elif name == "torch_compile":
            entry.update(
                library=f"torch {torch.__version__}",
                entry="torch.compile(document-local conv + SiLU in FP32)",
                determinism="unverified (Inductor kernels; the gate records run-to-run bits)",
                preactivation="fp32",
            )
            layer = _compile_layer(x, lens)
        else:
            raise ValueError(f"unknown single-rank arm {name!r}")
        entry["feature"] = feature(
            preactivation=entry["preactivation"],
            taps=w.dtype,
            bias=None if b is None else b.dtype,
            cake_preactivation=cake_preactivation,
            cake_taps=w.dtype,
        )
        # The leaves share the row's storage: no arm writes its inputs.
        leaves = [t.detach().requires_grad_(True) for t in (x, w, b) if t is not None]
        entry["env"] = env
        calls = autograd_calls(layer, leaves, [dy])
        with environment(env):
            calls["record"]()
            if route_of is not None:
                entry["route"] = route_of()
            calls["bwd"]()
            torch.cuda.synchronize(x.device)
        return entry, calls, gate_run(layer, env)
    except Unsupported as exc:
        entry["unsupported"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
        entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    entry["eligible"] = False
    return entry, None, None


def single_rank_gate(
    runs: dict[str, numerics.ArmRun],
    entries: dict[str, dict],
    *,
    cu: torch.Tensor,
    S: int,
    C: int,
    weight_dtype: torch.dtype,
    bias_dtype: Optional[torch.dtype],
    draws: int = cake_naive.DRAWS,
    seed: int = 0,
) -> None:
    """Run the accuracy gate on every built arm of a row and record it on their entries (:func:`apply_gate`)."""

    verdicts = numerics.gate(
        runs, S=S, C=C, cu=cu, weight_dtype=weight_dtype, bias_dtype=bias_dtype, draws=draws, seed=seed
    )
    for name, verdict in verdicts.items():
        apply_gate(entries[name], verdict)


# ---------------------------------------------------------------------------------------------------------------------
# Context parallel
# ---------------------------------------------------------------------------------------------------------------------


def merged_fragments(
    cu: Sequence[int], fragments: Sequence[Any]
) -> list[tuple[int, int]]:
    """A rank's fragments (``doc`` / ``off`` / ``length``) as half-open global token ranges, adjacent ones merged,
    in span order."""

    out: list[tuple[int, int]] = []
    for f in fragments:
        if f.length <= 0:
            continue
        start = cu[f.doc] + f.off
        stop = start + f.length
        if out and out[-1][1] == start:
            out[-1] = (out[-1][0], stop)
        else:
            out.append((start, stop))
    return out


def _take(t: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``t[rows]`` with zero rows where ``rows`` is past the end of ``t`` (padding)."""

    rows = rows.to(t.device)
    valid = rows < t.shape[0]
    out = t.new_zeros((rows.numel(), *t.shape[1:]))
    out[valid] = t[rows[valid]]
    return out.contiguous()


def kcp_arm(
    name: str,
    *,
    xs: Sequence[torch.Tensor],
    ws: Sequence[torch.Tensor],
    dys: Sequence[torch.Tensor],
    cu: Sequence[int],
    fragments: Sequence[Sequence[tuple[int, int]]],
    layout: str,
    group: Any,
    rank: int,
    world: int,
) -> tuple[dict, Optional[tuple[dict[str, Callable[[], object]], torch.Tensor]]]:
    """``(entry, (calls, rows))`` of one context-parallel arm on a row: ``xs`` / ``dys`` are the whole pack's
    tensors (every rank holds them), ``ws`` the FP32 taps, ``fragments`` every rank's global token ranges in span
    order (the benchmark's partition).  ``rows`` maps the arm's local rows to global tokens (past the pack: FLA's
    padding).  ``entry`` describes the arm or carries ``unsupported`` with the reason; then the second item is
    ``None``."""

    entry: dict = {"name": name}
    device = xs[0].device
    total = int(cu[-1])
    try:
        if name in ("fla_cp_triton", "fla_cp_cuda"):
            backend = name.removeprefix("fla_cp_")
            entry.update(
                library=_library("fla", _fla_distribution()),
                entry=f"{FLA_ENTRY}(cp_context=build_cp_context(...), backend={backend!r})",
                determinism="deterministic (FP32 per-chunk partials, ordered reduce)"
                if backend == "triton"
                else "non-deterministic (Dao kernel, FP32 atomic dweight)",
                preactivation="bf16" if backend == "triton" else "fp32",
                taps_used="float32",
            )
            if layout not in ("contiguous", "zigzag"):
                raise Unsupported(
                    f"FLA CP partitions are contiguous or zigzag; no {layout} form"
                )
            if backend == "cuda":
                if layout != "contiguous":
                    raise Unsupported(
                        "FLA's CUDA CP backend takes the contiguous layout only"
                    )
                entry["dao"] = _library("causal_conv1d", "causal-conv1d")
            from fla.modules.causal_conv1d import causal_conv1d  # type: ignore
            from fla.ops.cp.context import build_cp_context  # type: ignore

            pad = (-total) % (2 * world)
            bounds = list(cu) + ([total + pad] if pad else [])
            cu_cpu = torch.tensor(bounds, dtype=torch.long)
            ctx = build_cp_context(
                cu_cpu.to(device),
                group,
                conv1d_kernel_size=WIDTH,
                cu_seqlens_cpu=cu_cpu,
                layout=layout,
            )
            padded = bounds[-1]
            if layout == "contiguous":
                share = padded // world
                rows = torch.arange(rank * share, (rank + 1) * share)
            else:
                piece = padded // (2 * world)
                rows = torch.cat(
                    [
                        torch.arange(rank * piece, (rank + 1) * piece),
                        torch.arange(
                            (2 * world - 1 - rank) * piece, (2 * world - rank) * piece
                        ),
                    ]
                )
            local_tokens = int(ctx.cu_seqlens_cpu[-1])
            if rows.numel() != local_tokens:
                raise Unsupported(
                    f"FLA's local token count {local_tokens} differs from the {layout} shard {rows.numel()}"
                )
            entry["padding_tokens"] = pad
            leaves = [_take(x, rows).requires_grad_(True) for x in xs] + [
                w.detach().clone().requires_grad_(True) for w in ws
            ]
            local_dys = [_take(dy, rows) for dy in dys]

            def layer(*t: torch.Tensor) -> tuple:
                n = len(t) // 2
                return tuple(
                    causal_conv1d(
                        x.unsqueeze(0),
                        w,
                        activation="silu",
                        cp_context=ctx,
                        backend=backend,
                    )[0].squeeze(0)
                    for x, w in zip(t[:n], t[n:], strict=True)
                )

        elif name == "attn_gym_cp":
            entry.update(
                library=_library("attn_gym", "attention-gym"),
                cutlass_dsl=_distribution("nvidia-cutlass-dsl"),
                entry=ATTN_GYM_ENTRY,
                determinism="deterministic (FP32 partials, torch sum)",
                preactivation="fp32",
                taps_used="bfloat16",
            )
            from attn_gym.linear.context_parallel import (  # type: ignore
                ContextParallelRouting,
                context_parallel_conv_history,
            )
            from attn_gym.linear.short_conv import causal_conv1d as short_conv  # type: ignore

            if any(not owned for owned in fragments):
                raise Unsupported(
                    "a rank owns no tokens (Attention Gym rejects empty ranks)"
                )
            routing = ContextParallelRouting.from_fragments(
                list(cu), fragments, rank, device, conv_history=WIDTH - 1
            )
            rows = torch.cat(
                [torch.arange(start, stop) for start, stop in fragments[rank]]
            )
            leaves = [_take(x, rows).requires_grad_(True) for x in xs] + [
                w.detach().to(torch.bfloat16).requires_grad_(True) for w in ws
            ]
            local_dys = [_take(dy, rows) for dy in dys]

            def layer(*t: torch.Tensor) -> tuple:
                n = len(t) // 2
                out = []
                for x, w in zip(t[:n], t[n:], strict=True):
                    x3 = x.unsqueeze(0)
                    history = context_parallel_conv_history(x3, routing, group)
                    out.append(
                        short_conv(
                            x3,
                            w,
                            activation="silu",
                            cu_seqlens=routing.cu_seqlens,
                            initial_state=history,
                        ).squeeze(0)
                    )
                return tuple(out)

        else:
            raise ValueError(f"unknown KCP arm {name!r}")
        taps = torch.bfloat16 if entry["taps_used"] == "bfloat16" else torch.float32
        # The Cake KCP backward evaluates SiLU' at the FP32 preactivation (its policy B).
        entry["feature"] = feature(
            preactivation=entry["preactivation"],
            taps=taps,
            bias=None,
            cake_preactivation="fp32",
            cake_taps=torch.float32,
        )
        return entry, (autograd_calls(layer, leaves, local_dys), rows)
    except Unsupported as exc:
        entry["unsupported"] = str(exc)
    except Exception as exc:  # noqa: BLE001 -- a failing library is reported, the row goes on
        entry["unsupported"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return entry, None


def kcp_gate(
    arms: dict[str, tuple[Calls, torch.Tensor, dict]],
    *,
    cu: Sequence[int],
    channels: int,
    tensors: int,
    device: torch.device,
    all_reduce: Callable[[torch.Tensor, str], torch.Tensor],
    draws: int = cake_naive.DRAWS,
    seed: int = 0,
) -> dict[str, dict]:
    """The accuracy gate of context-parallel arms (``cake_causal_conv1d_numerics.py``): per draw every rank draws the
    whole pack (``tensors`` convolutions, FP32 taps, no bias) from the same seed, computes the float64 oracle and the
    PyTorch BF16 values of the whole pack, refills each arm's leaves and output gradients in place with its rows of
    the draw (``arms[name] = (calls, rows, env)``; ``rows``: the global token of each local row, past the pack: zero
    padding), and runs its recorded forward and backward.  ``y`` / ``dx`` are judged on the arm's rows (padding
    excluded) with the errors reduced over the ranks, ``dweight`` as the FP32 sum of the ranks' partials.
    ``all_reduce(t, op)`` reduces ``t`` in place over the ranks (``op``: ``"sum"`` or ``"max"``) and returns it;
    every rank must run every arm in the same order.  Returns per arm the gate's verdict (see
    ``cake_causal_conv1d_numerics.gate``)."""

    total = int(cu[-1])
    cu_t = torch.tensor(list(cu), dtype=torch.int32, device=device)
    world = dist_world(all_reduce, device)
    names = list(arms)
    judges = {name: numerics.ReducedJudge(draws) for name in names}
    failed: dict[str, str] = {}
    contract_bad = {name: 0.0 for name in names}
    bitwise = {name: None for name in names}
    outputs = [f"{o}[{i}]" if tensors > 1 else o for i in range(tensors) for o in ("y", "dx", "dweight")]
    for j in range(draws):
        generator = torch.Generator(device=device).manual_seed(cake_naive.draw_seed(seed, j))
        xs, ws, dys = [], [], []
        for _ in range(tensors):
            xs.append(torch.randn(total, channels, device=device, generator=generator, dtype=torch.bfloat16))
            ws.append(torch.randn(channels, WIDTH, device=device, generator=generator) * 0.5)
            dys.append(torch.randn(total, channels, device=device, generator=generator, dtype=torch.bfloat16))
        wants = [numerics.oracle(x, w, cu_t, None, dy) for x, w, dy in zip(xs, ws, dys, strict=True)]
        naives = [numerics.naive_bf16(x, w, cu_t, None, dy) for x, w, dy in zip(xs, ws, dys, strict=True)]
        for name in names:
            if name in failed:
                continue
            calls, rows, env = arms[name]
            rows_d = rows.to(device)
            valid = rows_d < total
            index = rows_d[valid]
            error = None
            stats = torch.zeros(len(outputs), 6, dtype=torch.float64, device=device)
            bad = torch.zeros(1, dtype=torch.float64, device=device)
            same = torch.ones(1, dtype=torch.float64, device=device)
            try:
                with torch.no_grad():
                    for i in range(tensors):
                        calls.leaves[i].copy_(_take(xs[i], rows))
                        calls.leaves[tensors + i].copy_(ws[i])
                        calls.dys[i].copy_(_take(dys[i], rows))
                with environment(env):
                    ys = calls["record"]()
                    grads = calls["bwd"]()
                    if j == 0:
                        first = [t.detach().clone() for t in (*ys, *grads)]
                        for _ in range(numerics.REPEATS):
                            again = [*calls["record"](), *calls["bwd"]()]
                            if not all(torch.equal(a, b) for a, b in zip(first, again, strict=True)):
                                same[0] = 0.0
                        del first
                torch.cuda.synchronize(device)
                for i in range(tensors):
                    y, dx, dw = ys[i].detach()[valid], grads[i][valid], grads[tensors + i]
                    want, naive = wants[i], naives[i]
                    for k, (actual, key) in enumerate(((y, "y"), (dx, "dx"))):
                        expected = want[key].index_select(0, index)
                        bad[0] += float((expected.isfinite() & ~actual.isfinite()).sum())
                        stats[3 * i + k] = numerics.local_norms(actual, expected, naive[key].index_select(0, index))
                    dw_sum = all_reduce(dw.detach().float().clone(), "sum")
                    bad[0] += float((~dw_sum.isfinite()).sum())
                    stats[3 * i + 2] = numerics.local_norms(dw_sum, want["dweight"], naive["dweight"])
                    # Every rank holds the same summed dweight: divide what the reduction below adds up.
                    stats[3 * i + 2, [0, 1, 2, 5]] /= world
            except Exception as exc:  # noqa: BLE001 -- a failing arm is reported, the others go on
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            flag = all_reduce(torch.tensor([1.0 if error else 0.0], dtype=torch.float64, device=device), "max")
            if float(flag[0]):
                failed[name] = error or "failed on another rank"
                continue
            maxima = all_reduce(stats[:, [3, 4]].clone(), "max")
            stats = all_reduce(stats, "sum")
            stats[:, [3, 4]] = maxima
            contract_bad[name] += float(all_reduce(bad, "sum")[0])
            if j == 0:
                bitwise[name] = int(all_reduce(same, "sum")[0]) == world
            for k, out_name in enumerate(outputs):
                judges[name].add_reduced(out_name, j, stats[k].tolist())
        del xs, ws, dys, wants, naives
    results: dict[str, dict] = {}
    for name in names:
        if name in failed:
            results[name] = {"passed": False, "error": failed[name], "draws": draws}
            continue
        found = judges[name].failures()
        if contract_bad[name]:
            found.insert(0, f"{int(contract_bad[name])} elements not finite where the oracle is (all draws)")
        results[name] = {
            "passed": not found,
            "failures": found[:20],
            "draws": draws,
            "outputs": numerics.summarize(judges[name]),
            "run_to_run_bitwise": bitwise[name],
        }
    return results


def dist_world(all_reduce: Callable[[torch.Tensor, str], torch.Tensor], device: torch.device) -> int:
    """The number of ranks ``all_reduce`` spans."""

    return int(all_reduce(torch.ones(1, dtype=torch.float64, device=device), "sum")[0])


def best_baseline(
    arms: dict[str, dict], nondet: dict, keys: Sequence[str]
) -> tuple[dict, dict]:
    """``(best, speedup)``: per ``{phase}_{kind}`` key the fastest eligible arm (``{"arm", "us"}``) and its time
    over the non-deterministic Cake time (above 1: Cake faster).  Every timed arm also gets its own ``speedup``."""

    best: dict = {}
    for key in keys:
        timed = [
            (entry[f"{key}_us"], name)
            for name, entry in arms.items()
            if entry.get("eligible") and f"{key}_us" in entry
        ]
        if timed:
            us, name = min(timed)
            best[key] = {"arm": name, "us": us}
    for entry in arms.values():
        entry["speedup"] = {
            key: entry[f"{key}_us"] / nondet[f"{key}_us"]
            for key in keys
            if f"{key}_us" in entry and f"{key}_us" in nondet
        }
    speedup = {
        key: value["us"] / nondet[f"{key}_us"]
        for key, value in best.items()
        if f"{key}_us" in nondet
    }
    return best, speedup


def describe(entry: dict) -> str:
    """One line per arm: library, determinism, match and numerics, or the unsupported reason."""

    if "unsupported" in entry:
        return f"{entry['name']}: unsupported ({entry['unsupported']})"
    verdict = entry.get("gate", {})
    ratios = " ".join(
        f"{out} {v['norms']['l2']['ours_over_bf16'][1]:.3f}"
        for out, v in verdict.get("outputs", {}).items()
        if v["norms"]["l2"]["ours_over_bf16"]
    )
    route = f" route {entry['route']}" if entry.get("route") else ""
    notes = "; ".join(entry.get("feature", {}).get("notes", []))
    return (
        f"{entry['name']}: {entry.get('library')}, {entry.get('determinism')}, "
        f"match {entry.get('feature', {}).get('match')}"
        + (f" ({notes})" if notes else "")
        + f", gate {'pass' if verdict.get('passed') else 'FAIL'} [largest l2 / bf16: {ratios}]"
        + f", run-to-run bitwise {verdict.get('run_to_run_bitwise')}{route}"
        + ("" if entry.get("eligible") else ", not eligible")
    )


def format_sota(result: dict, head: str, phases: Sequence[str], kind: str) -> list[str]:
    """Per phase every baseline's ``kind`` time (us) and the best eligible one over the non-deterministic time;
    then one line per baseline: library, determinism, match, numerics, or the reason it cannot run the row."""

    arms = result.get("sota")
    if not arms:
        return []
    lines = []
    for phase in phases:
        key = f"{phase}_{kind}"
        cells = " | ".join(
            f"{name} {entry[f'{key}_us']:9.2f}" if f"{key}_us" in entry else f"{name} -"
            for name, entry in arms.items()
        )
        best = result["best_sota"].get(key)
        # No ratio when the non-deterministic mode has no time for the phase (it failed in a timing round).
        ratio = result["speedup_vs_sota"].get(key)
        tail = (
            "best -"
            if best is None
            else f"best {best['arm']} {best['us']:9.2f} -> sota/nondet "
            + ("-" if ratio is None else f"{ratio:5.3f}x")
        )
        lines.append(f"{head} {phase:4s} sota {kind} | {cells} | {tail}")
    lines.extend(f"{head} sota {describe(entry)}" for entry in arms.values())
    return lines


def ineligible_reason(entry: dict) -> Optional[str]:
    """Why an arm is not a candidate for the best baseline of a row (``None``: it is one)."""

    if "unsupported" in entry:
        return entry["unsupported"]
    reasons = []
    if entry.get("feature", {}).get("match") == "less_precise":
        reasons.append(
            f"feature mismatch ({entry['feature']['match']}: "
            + "; ".join(entry["feature"]["notes"])
            + ")"
        )
    verdict = entry.get("gate", {})
    if not verdict.get("passed"):
        reasons.append(
            "accuracy gate not passed: "
            + (verdict.get("error") or "; ".join(verdict.get("failures", [])[:3]) or "not run")
        )
    return "; ".join(reasons) or None


def comparison(result: dict, phases: Sequence[str], column: str, rule: str) -> dict:
    """The two-baseline comparison of a row on one timing ``column`` (``gpu``: the CUPTI cold-L2 span; ``ahead``:
    the GPU time with the host enqueue hidden): per phase the deterministic and non-deterministic Cake times, the
    best eligible baseline (or ``unsupported`` with every arm's reason) and the ratios ``det / nondet`` and
    ``sota / nondet`` (above 1: the non-deterministic Cake mode is faster); every arm is listed with its time and,
    when it cannot be the best, the reason."""

    det, nondet = result["modes"]["det"], result["modes"]["nondet"]
    arms = result.get("sota", {})
    reasons = {name: ineligible_reason(entry) for name, entry in arms.items()}
    excluded = "; ".join(
        f"{name}: {reason}" for name, reason in reasons.items() if reason
    )
    out: dict = {"column": column, "rule": rule, "phases": {}}
    for phase in phases:
        key = f"{phase}_{column}"
        row: dict = {
            "det_us": det.get(f"{key}_us"),
            "nondet_us": nondet.get(f"{key}_us"),
            "arms": {
                name: {
                    "us": entry.get(f"{key}_us"),
                    "eligible": reasons[name] is None,
                    **({"excluded": reasons[name]} if reasons[name] else {}),
                }
                for name, entry in arms.items()
            },
        }
        if row["det_us"] is not None and row["nondet_us"]:
            row["det_over_nondet"] = row["det_us"] / row["nondet_us"]
        best = result.get("best_sota", {}).get(key)
        if best is not None and row["nondet_us"]:
            row["sota"] = {
                "arm": best["arm"],
                "library": arms[best["arm"]].get("library"),
                "determinism": arms[best["arm"]].get("determinism"),
                "match": arms[best["arm"]]["feature"]["match"],
                "us": best["us"],
                "sota_over_nondet": best["us"] / row["nondet_us"],
            }
        else:
            row["sota"] = {"unsupported": excluded or "no non-Cake baseline requested"}
        out["phases"][phase] = row
    return out


def format_comparison(result: dict, head: str) -> list[str]:
    """One line per phase of the row's two-baseline comparison: the timing column, det, nondet, the best eligible
    baseline and the two ratios, or ``unsupported: <reason>``."""

    block = result.get("comparison")
    if not block:
        return []
    lines = []
    for phase, row in block["phases"].items():
        det, nondet = row["det_us"], row["nondet_us"]
        cake = (
            f"det {det:9.2f} nondet {nondet:9.2f} det/nondet {row['det_over_nondet']:5.3f}x"
            if det is not None and nondet
            else "det - nondet -"
        )
        sota = row["sota"]
        best = (
            f"unsupported: {sota['unsupported']}"
            if "unsupported" in sota
            else f"best {sota['arm']} ({sota['match']}) {sota['us']:9.2f} sota/nondet {sota['sota_over_nondet']:5.3f}x"
        )
        lines.append(f"{head} {phase:4s} compare [{block['column']}] {cake} | {best}")
    return lines


def parse_arms(values: Sequence[str], known: Sequence[str]) -> list[str]:
    """``--sota`` values: ``all``, ``none`` or arm names."""

    if not values or list(values) == ["all"]:
        return list(known)
    if list(values) == ["none"]:
        return []
    unknown = [v for v in values if v not in known]
    if unknown:
        raise SystemExit(f"unknown --sota arm(s) {unknown}; known: {list(known)}")
    return list(values)
