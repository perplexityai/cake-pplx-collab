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

The FLA reference of the KDA training accuracy tests: flash-linear-attention (FLA) 0.6.0 ``fla.ops.kda.chunk_kda`` on
the operands of the judged kernels.  It is the judge's second reference (``tests/cake_naive.py``, the KDA rule): an
end-to-end output (``out`` and the gradient of every leaf) passes when it meets the naive BF16 tier or meets FLA under
the same rule, and fails only when it misses both.

FLA runs as the benchmarks' ``fla`` arm runs (``benchmarks/cake_kda_baselines.py``): the Triton ``chunk_kda``
(``FLA_FLASH_KDA=0`` and ``FLA_TILELANG=0`` while it is imported and called), ``safe_gate=True``,
``use_beta_sigmoid_in_kernel=False``, the kernels' scale, int64 ``cu_seqlens``, no initial and no final state.  The
activated FP32 gate goes in with ``use_gate_in_kernel=False``; the raw gate mode with ``use_gate_in_kernel=True``,
``A_log``, ``dt_bias`` and ``lower_bound``; the in-kernel Q/K L2 norm with ``use_qk_l2norm_in_kernel=True``.  FLA's
forward and its own backward give ``out`` and every gradient (:func:`fla_of`), keyed as
``cake_kda_naive.oracle_with_grads``.  A KCP case is judged against FLA on one GPU over the whole packed batch, and a
checkpointed arm against the same FLA step as the plain arm.

FLA is a reference only where it runs the case.  When it does not import, is not version :data:`FLA_VERSION`, the
operands are not on a CUDA device, an activated gate is outside FLA's safe gate range (:data:`SAFE_GATE_RANGE`) or
the call raises, :func:`fla_of` returns the reason (``fla unavailable: ...``) instead of values; the judge then judges
the output against the naive tier only, and a miss of the tier stays a failure.  FLA runs on every draw (not only after
a miss): after its first call in a process its step costs a small part of the FP64 reference and of the naive tier.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Iterator, Mapping

import torch

#: The FLA release the judge reads.
FLA_VERSION = "0.6.0"
FLA_ENTRY = "fla.ops.kda.chunk_kda"
#: FLA's Triton ``chunk_kda``: the environment while FLA is imported and called.
FLA_ENV = {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
#: The activated gates FLA's ``safe_gate=True`` takes.
SAFE_GATE_RANGE = (-5.0, 0.0)
#: The start of every reason FLA gives no values for a draw.
UNAVAILABLE = "fla unavailable"
GRAD_NAMES = ("dq", "dk", "dv", "dg", "dbeta")
RAW_GRAD_NAMES = GRAD_NAMES + ("dA_log", "ddt_bias")
#: The record of the FLA steps of a judge (``record_property``), which the family's ``conftest.py`` prints.
PROPERTY = "fla_reference"

_loaded: tuple | None = None


@contextlib.contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set the environment variables ``values`` for the block, then restore them."""
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load() -> tuple:
    """``(chunk_kda, None)``, or ``(None, reason)`` when FLA :data:`FLA_VERSION` does not import (cached)."""
    global _loaded
    if _loaded is None:
        try:
            with _environment(FLA_ENV):
                import fla
                from fla.ops.kda import chunk_kda
            version = getattr(fla, "__version__", None)
            if version != FLA_VERSION:
                _loaded = (None, f"{UNAVAILABLE}: fla {version} is installed, the reference is fla {FLA_VERSION}")
            else:
                _loaded = (chunk_kda, None)
        except Exception as exc:  # noqa: BLE001 -- any import failure means there is no FLA reference
            _loaded = (None, f"{UNAVAILABLE}: fla does not import ({type(exc).__name__}: {exc})"[:400])
    return _loaded


class FlaReference:
    """FLA's values of one draw's outputs (:func:`fla_of`), or the reason there are none, and the step's seconds."""

    def __init__(self, values: dict[str, torch.Tensor] | None, reason: str | None, seconds: float = 0.0) -> None:
        self.values, self.reason, self.seconds = values, reason, seconds

    def value(self, name: str, actual: torch.Tensor) -> torch.Tensor | str:
        """FLA's value of output ``name`` in the shape of ``actual`` (a reshape of the same elements, as the judge reads
        the reference), or the reason there is none: the ``fla`` of ``TierJudge.add``."""
        if self.values is None:
            return self.reason
        found = self.values.get(name)
        if found is None:
            return f"no FLA counterpart: FLA gives no {name}"
        if found.shape != actual.shape and found.numel() == actual.numel():
            return found.reshape(actual.shape)
        return found


def _check(inputs: Mapping[str, torch.Tensor]) -> str | None:
    """Why FLA does not run these operands, or None."""
    q = inputs["q"]
    if not q.is_cuda:
        return f"{UNAVAILABLE}: the operands are on {q.device}, FLA runs on CUDA devices"
    if q.dim() != 3:
        return f"{UNAVAILABLE}: operands of shape {tuple(q.shape)}, not [S, H, D]"
    if "A_log" not in inputs:
        g = inputs["g"]
        lo, hi = SAFE_GATE_RANGE
        if g.numel() and not bool(((g >= lo) & (g <= hi)).all()):
            return (
                f"{UNAVAILABLE}: the activated gate is outside FLA's safe gate range [{lo}, {hi}] "
                f"(min {float(g.min())}, max {float(g.max())}, finite {bool(g.isfinite().all())})"
            )
    return None


def fla_of(inputs: Mapping[str, torch.Tensor], scale: float, lower_bound: float, l2n: bool) -> FlaReference:
    """FLA's ``out`` and gradients on the operands of a judged draw (as the FP64 reference reads them, without a batch
    axis: ``q, k, v, g, beta, dout, cu_seqlens`` and in the raw gate mode ``A_log``, ``dt_bias``), keyed as
    ``cake_kda_naive.oracle_with_grads``, with ``scale``, the raw gate's ``lower_bound`` and with ``l2n`` the in-kernel
    Q/K L2 norm; or the reason FLA does not run them (:data:`UNAVAILABLE`)."""
    chunk_kda, reason = load()
    if chunk_kda is None:
        return FlaReference(None, reason)
    reason = _check(inputs)
    if reason is not None:
        return FlaReference(None, reason)
    raw = "A_log" in inputs
    names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw else ())
    device = inputs["q"].device
    start = time.perf_counter()
    try:
        with _environment(FLA_ENV), torch.cuda.device(device), torch.enable_grad():
            # dt_bias flat [H * 128]: FLA's gradient comes back in that shape (FlaReference.value reshapes it)
            leaves = [
                (inputs[n].reshape(-1) if n == "dt_bias" else inputs[n]).detach().clone().contiguous()
                for n in names
            ]
            for leaf in leaves:
                leaf.requires_grad_(True)
            kw = {
                "scale": scale,
                "initial_state": None,
                "output_final_state": False,
                "use_qk_l2norm_in_kernel": bool(l2n),
                "use_beta_sigmoid_in_kernel": False,
                "safe_gate": True,
                "cu_seqlens": inputs["cu_seqlens"].long(),
            }
            if raw:
                kw.update(use_gate_in_kernel=True, A_log=leaves[5], dt_bias=leaves[6], lower_bound=lower_bound)
            else:
                kw["use_gate_in_kernel"] = False
            o, _ = chunk_kda(*(x.unsqueeze(0) for x in leaves[:5]), **kw)
            o = o.squeeze(0)
            grads = torch.autograd.grad(o, leaves, inputs["dout"].to(o.dtype).contiguous(), allow_unused=True)
            torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 -- FLA does not run the case: the judge reads the naive tier only
        return FlaReference(None, f"{UNAVAILABLE}: {FLA_ENTRY} raised {type(exc).__name__}: {exc}"[:400])
    values = {"out": o.detach()}
    for name, grad in zip(RAW_GRAD_NAMES if raw else GRAD_NAMES, grads, strict=True):
        if grad is not None:
            values[name] = grad.detach()
    return FlaReference(values, None, time.perf_counter() - start)


class FlaSteps:
    """The FLA steps of one judge: their count, seconds and the reasons of the draws without FLA values, recorded with
    ``record_property`` (:data:`PROPERTY`) for the conftest's summary."""

    def __init__(self) -> None:
        self.calls, self.seconds, self.reasons = 0, 0.0, {}

    def add(self, ref: FlaReference) -> FlaReference:
        self.calls += 1
        self.seconds += ref.seconds
        if ref.values is None:
            self.reasons[ref.reason] = self.reasons.get(ref.reason, 0) + 1
        return ref

    def summary(self) -> dict:
        return {"calls": self.calls, "seconds": round(self.seconds, 3), "reasons": dict(self.reasons)}

    def record(self, record_property) -> None:
        record_property(PROPERTY, self.summary())


def fla_line(test_id: str, steps: Mapping) -> str:
    """``FLA <test> calls=<n> seconds=<s>`` and the reasons of the draws without FLA values."""
    reasons = "; ".join(f"{reason} x{count}" for reason, count in steps.get("reasons", {}).items())
    return f"FLA {test_id} calls={steps['calls']} seconds={steps['seconds']}" + (f" | {reasons}" if reasons else "")


def summarize_fla(terminalreporter) -> None:
    """The terminal summary's ``FLA`` lines (one per test that judged with FLA): call it from the family conftest's
    ``pytest_terminal_summary``."""
    lines = []
    for reports in terminalreporter.stats.values():
        for report in reports:
            if getattr(report, "when", None) != "call":
                continue
            for key, steps in getattr(report, "user_properties", ()):
                if key == PROPERTY:
                    lines.append(fla_line(report.nodeid, steps))
    if lines:
        terminalreporter.section("training accuracy: the FLA reference steps of each judged test")
        for line in lines:
            terminalreporter.write_line(line)


__all__ = [
    "FLA_ENTRY",
    "FLA_ENV",
    "FLA_VERSION",
    "PROPERTY",
    "SAFE_GATE_RANGE",
    "UNAVAILABLE",
    "FlaReference",
    "FlaSteps",
    "fla_line",
    "fla_of",
    "load",
    "summarize_fla",
]
