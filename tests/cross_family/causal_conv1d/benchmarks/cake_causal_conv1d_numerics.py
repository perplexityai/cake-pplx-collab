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

The accuracy gate of the causal conv1d benchmarks: every arm (a Cake mode or a non-Cake baseline) is judged as the
training tests judge a kernel (``cake_naive.py``, the shared judge), never against another arm.

For each row the gate draws :data:`cake_naive.DRAWS` independent inputs (``x``, ``dy`` ``randn`` BF16; taps and bias
``N(0, 0.25)`` in the row's dtypes; draw ``j`` seeded by ``cake_naive.draw_seed(seed, j)``), runs every arm's
forward and backward once on each draw, and judges each output (``y``, ``dx``, ``dweight``, ``dbias``) by its error
against the float64 oracle (:func:`oracle`) beside the error of PyTorch BF16 (:func:`naive_bf16`), in the L2 and the
max norm, median and largest over the draws, with the judge's near tie and zero rule.  An arm passes the gate when
every output meets the BF16 tier and holds its contract on every draw; a baseline counts in a comparison only when it
passes.  On the first draw every arm also runs :data:`REPEATS` more times and the gate records whether its outputs
were the same bytes on every run (``run_to_run_bitwise``): the determinism the arm shows on this row.

The oracle is the function in float64, document-local over ``cu_seqlens``::

    pre[t, c]     = sum_k w[c, k] x[t - 3 + k, c] + b[c]   (x of another document, or before the first token, is 0)
    y[t, c]       = pre * sigmoid(pre)
    dpre[t, c]    = dy[t, c] * s (1 + pre (1 - s)),  s = sigmoid(pre)
    dx[t, c]      = sum_k w[c, k] dpre[t + 3 - k, c]   (dpre of another document, or past the last token, is 0)
    dweight[c, k] = sum_t dpre[t, c] x[t - 3 + k, c],  dbias[c] = sum_t dpre[t, c]

with no rounding, in token blocks of :data:`BLOCK` rows with a three-token halo.  The naive tier is PyTorch in BF16:
every input cast to BF16, each document's rows behind three zero rows, one ``F.conv1d`` (``groups=C``), ``F.silu``
and ``torch.autograd.grad`` against ``dy``, on blocks of channels of at most :data:`NAIVE_BLOCK` padded elements.

The context-parallel benchmarks judge with :class:`ReducedJudge`: each rank measures the errors of its own rows of
``y`` / ``dx`` (:func:`local_norms`), the ranks reduce them (sums of squares added, maxima taken), and the summed
``dweight`` partials are judged whole on every rank.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Optional

import torch
import torch.nn.functional as F

import cake_naive as judge_lib

WIDTH = 4
# The oracle's token block: each float64 temporary of a block is (BLOCK + 6) x C x 8 bytes.
BLOCK = 8192
# The most padded elements (rows x channels) of one block of channels of :func:`naive_bf16`.
NAIVE_BLOCK = 2**27
OUTPUTS = ("y", "dx", "dweight", "dbias")
# Extra runs of every arm on the first draw (the run-to-run bit check).
REPEATS = 2


def doc_ids(S: int, cu: torch.Tensor) -> torch.Tensor:
    """The document of each token: the largest ``d`` with ``cu[d] <= t``."""

    return torch.bucketize(torch.arange(S, device=cu.device), cu[1:].long(), right=True)


def _shift_down(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    """``out[t] = v[t - d]`` where token ``t - d`` exists and is of token ``t``'s document, else 0."""

    if d == 0:
        return v.clone()
    out = torch.zeros_like(v)
    if d < v.shape[0]:
        same = (ids[d:] == ids[:-d]).unsqueeze(1)
        out[d:] = torch.where(same, v[:-d], 0.0)
    return out


def _shift_up(v: torch.Tensor, ids: torch.Tensor, d: int) -> torch.Tensor:
    """``out[t] = v[t + d]`` where token ``t + d`` exists and is of token ``t``'s document, else 0."""

    if d == 0:
        return v.clone()
    out = torch.zeros_like(v)
    if d < v.shape[0]:
        same = (ids[:-d] == ids[d:]).unsqueeze(1)
        out[:-d] = torch.where(same, v[d:], 0.0)
    return out


def oracle(
    x: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    dy: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """The float64 ``y`` and, with ``dy``, ``dx``, ``dweight`` and (with a bias) ``dbias`` of the rows ``x``
    (``[S, C]``) of the documents ``cu``."""

    S, C = x.shape
    ids = doc_ids(S, cu)
    wd = w.double()
    bd = None if b is None else b.double()
    found = {"y": torch.empty(S, C, device=x.device, dtype=torch.float64)}
    if dy is not None:
        found["dx"] = torch.empty(S, C, device=x.device, dtype=torch.float64)
        found["dweight"] = torch.zeros(C, WIDTH, device=x.device, dtype=torch.float64)
        if bd is not None:
            found["dbias"] = torch.zeros(C, device=x.device, dtype=torch.float64)
    for a in range(0, S, BLOCK):
        end = min(a + BLOCK, S)
        lo, hi = max(a - (WIDTH - 1), 0), min(end + (WIDTH - 1), S)
        xs, win = x[lo:hi].double(), ids[lo:hi]
        pre = torch.zeros_like(xs)
        for k in range(WIDTH):
            pre = pre + wd[:, k] * _shift_down(xs, win, WIDTH - 1 - k)
        if bd is not None:
            pre = pre + bd
        s = torch.sigmoid(pre)
        rows = slice(a - lo, end - lo)
        found["y"][a:end] = (pre * s)[rows]
        if dy is None:
            continue
        dpre = dy[lo:hi].double() * (s * (1.0 + pre * (1.0 - s)))
        dxw = torch.zeros_like(dpre)
        for k in range(WIDTH):
            dxw = dxw + wd[:, k] * _shift_up(dpre, win, WIDTH - 1 - k)
        found["dx"][a:end] = dxw[rows]
        dp = dpre[rows]
        for k in range(WIDTH):
            found["dweight"][:, k] += (dp * _shift_down(xs, win, WIDTH - 1 - k)[rows]).sum(dim=0)
        if bd is not None:
            found["dbias"] += dp.sum(dim=0)
    return found


def naive_bf16(
    x: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    b: Optional[torch.Tensor] = None,
    dy: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """The PyTorch BF16 values of the oracle's outputs (BF16 tensors of the oracle's shapes)."""

    S, C = x.shape
    ids = doc_ids(S, cu)
    # Row t of document d sits at t + 3 (d + 1): three zero rows before every document, an empty one's included.
    pos = torch.arange(S, device=x.device) + (WIDTH - 1) * (ids + 1)
    rows = int(pos[-1]) + 1
    step = max(1, NAIVE_BLOCK // rows)
    parts = []
    for c in range(0, C, step):
        cols = slice(c, c + step)
        parts.append(
            _naive_channels(
                x[:, cols], w[cols], None if b is None else b[cols], pos, rows, None if dy is None else dy[:, cols]
            )
        )
    return {
        name: torch.cat([part[name] for part in parts], dim=0 if name in ("dweight", "dbias") else 1)
        for name in parts[0]
    }


def _naive_channels(x, w, b, pos, rows, dy) -> dict[str, torch.Tensor]:
    C = x.shape[1]
    grad = dy is not None
    dtype = torch.bfloat16
    with torch.enable_grad():
        xs = x.detach().to(dtype).requires_grad_(grad)
        ws = w.detach().to(dtype).reshape(C, 1, WIDTH).clone().requires_grad_(grad)
        bs = None if b is None else b.detach().to(dtype).clone().requires_grad_(grad)
        padded = xs.new_zeros(rows, C).index_copy(0, pos, xs)
        pre = F.conv1d(padded.t().unsqueeze(0), ws, bs, groups=C)[0].t()[pos - (WIDTH - 1)]
        y = F.silu(pre)
        if not grad:
            return {"y": y.detach()}
        inputs = (xs, ws) if bs is None else (xs, ws, bs)
        found = torch.autograd.grad(y, inputs, dy.detach().to(dtype))
    out = {"y": y.detach(), "dx": found[0], "dweight": found[1].reshape(C, WIDTH)}
    if bs is not None:
        out["dbias"] = found[2]
    return out


def draw_inputs(
    S: int,
    C: int,
    weight_dtype: torch.dtype,
    bias_dtype: Optional[torch.dtype],
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """``(x, w, b, dy)`` of one draw: ``x`` / ``dy`` ``randn`` BF16 ``[S, C]``, ``w`` ``[C, 4]`` and ``b`` ``[C]``
    ``N(0, 0.25)`` in their dtypes (``b`` None without a bias), from a generator seeded with ``seed``."""

    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(S, C, device=device, generator=generator, dtype=torch.bfloat16)
    w = (torch.randn(C, WIDTH, device=device, generator=generator) * 0.5).to(weight_dtype)
    b = None
    if bias_dtype is not None:
        b = (torch.randn(C, device=device, generator=generator) * 0.5).to(bias_dtype)
    dy = torch.randn(S, C, device=device, generator=generator, dtype=torch.bfloat16)
    return x, w, b, dy


ArmRun = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor], Mapping[str, Optional[torch.Tensor]]
]


def gate(
    arms: Mapping[str, ArmRun],
    *,
    S: int,
    C: int,
    cu: torch.Tensor,
    weight_dtype: torch.dtype,
    bias_dtype: Optional[torch.dtype],
    seed: int = 0,
    draws: int = judge_lib.DRAWS,
    repeats: int = REPEATS,
) -> dict[str, dict]:
    """Judge every arm of a row over ``draws`` inputs (module docstring).  ``arms[name](x, w, b, dy)`` runs the arm's
    forward and backward on one draw and returns its outputs by name (``y``, ``dx``, ``dweight``, ``dbias``; None
    for an output the row has not).  Returns per arm ``{"passed", "failures", "outputs", "draws",
    "run_to_run_bitwise"}`` (``outputs``: each output's verdict, its median and largest error and the tier's, per
    norm), or ``{"passed": False, "error": ...}`` when the arm raised."""

    device = cu.device
    judges = {name: judge_lib.TierJudge(draws) for name in arms}
    failures: dict[str, list[str]] = {name: [] for name in arms}
    errors: dict[str, str] = {}
    bitwise: dict[str, Optional[bool]] = {name: None for name in arms}
    for j in range(draws):
        x, w, b, dy = draw_inputs(S, C, weight_dtype, bias_dtype, judge_lib.draw_seed(seed, j), device)
        want = oracle(x, w, cu, b, dy)
        naive = naive_bf16(x, w, cu, b, dy)
        for name, run in arms.items():
            if name in errors:
                continue
            try:
                outputs = {k: v for k, v in run(x, w, b, dy).items() if v is not None}
                if j == 0:
                    first = {k: v.clone() for k, v in outputs.items()}
                    same = True
                    for _ in range(repeats):
                        again = run(x, w, b, dy)
                        same = same and all(torch.equal(first[k], again[k]) for k in first)
                    bitwise[name] = same
                    del first
                torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001 -- a failing arm is reported, the others go on
                errors[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
                continue
            for out_name, actual in outputs.items():
                expected = want.get(out_name)
                if expected is None:
                    failures[name].append(f"draw {j}: {out_name}: the oracle has no such output")
                    continue
                if message := judge_lib.contract(out_name, actual, expected):
                    failures[name].append(f"draw {j}: {message}")
                    continue
                judges[name].add(out_name, j, actual, expected, {"bf16": naive[out_name]})
            missing = sorted(set(want) - set(outputs))
            if missing:
                failures[name].append(f"draw {j}: no {missing}")
            del outputs
        del x, w, b, dy, want, naive
    results: dict[str, dict] = {}
    for name in arms:
        if name in errors:
            results[name] = {"passed": False, "error": errors[name], "draws": draws}
            continue
        found = failures[name] + judges[name].failures()
        results[name] = {
            "passed": not found,
            "failures": found[:20],
            "draws": draws,
            "outputs": summarize(judges[name]),
            "run_to_run_bitwise": bitwise[name],
        }
    return results


def summarize(judge: judge_lib.TierJudge) -> dict[str, dict]:
    """Each output's verdict in a small JSON form: ``tier`` (``bf16`` when it meets the tier, else None) and per norm
    ``[median, largest]`` of the arm's errors and of the tier's, and their ratios."""

    out = {}
    for name, verdict in judge.verdicts().items():
        norms = {}
        for norm, errors in verdict["norms"].items():
            ours, tier = errors["ours"], errors.get("bf16")
            ratio = None
            if tier is not None and tier[0] > 0 and tier[1] > 0:
                ratio = [ours[0] / tier[0], ours[1] / tier[1]]
            norms[norm] = {"ours": ours, "bf16": tier, "ours_over_bf16": ratio}
        out[name] = {"tier": verdict["tier"], "draws": verdict["draws"], "norms": norms}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Context parallel: the errors of each rank's rows, reduced over the ranks
# ---------------------------------------------------------------------------------------------------------------------


def local_norms(actual: torch.Tensor, expected: torch.Tensor, naive: torch.Tensor) -> torch.Tensor:
    """``[ours^2, tier^2, ref^2, ours_max, tier_max, ref_sum]`` of one rank's rows: the sums of squares and maxima of
    ``|actual - expected|`` and ``|naive - expected|`` (``naive`` cast to ``actual``'s dtype first) over the elements
    where ``expected`` is finite (float64; a non-finite difference counts as inf), and ``expected``'s squared sum and
    sum (the draw's fingerprint)."""

    inf = float("inf")
    out = torch.zeros(6, dtype=torch.float64, device=actual.device)
    for _, (e, a, t) in judge_lib._blocks(expected, actual, naive.to(actual.dtype)):
        e = e.to(device=actual.device, dtype=torch.float64)
        keep = e.isfinite()
        e = torch.where(keep, e, 0.0)
        da = torch.where(keep, (a.double() - e).abs().nan_to_num(nan=inf, posinf=inf), 0.0)
        dt = torch.where(keep, (t.double() - e).abs().nan_to_num(nan=inf, posinf=inf), 0.0)
        if not da.numel():
            continue
        out[0] += da.square().sum()
        out[1] += dt.square().sum()
        out[2] += e.square().sum()
        out[3] = torch.maximum(out[3], da.max())
        out[4] = torch.maximum(out[4], dt.max())
        out[5] += e.sum()
    return out


class ReducedJudge(judge_lib.TierJudge):
    """A :class:`cake_naive.TierJudge` fed with errors the ranks reduced (:func:`local_norms`, sums of squares added
    and maxima taken over the ranks) instead of whole tensors: the same verdicts, near tie and zero rule."""

    def add_reduced(self, name: str, draw: int, reduced: Sequence[float]) -> None:
        """Record draw ``draw`` of output ``name`` from the reduced ``[ours^2, tier^2, ref^2, ours_max, tier_max,
        ref_sum]``."""

        if not 0 <= draw < self.draws:
            raise ValueError(f"{name}: draw {draw} of a judge over {self.draws} draws")
        l2_ours, l2_tier = math.sqrt(reduced[0]), math.sqrt(reduced[1])
        by_norm = {"l2": (l2_ours, l2_tier), "max": (reduced[3], reduced[4])}
        for norm, (ours, tier) in by_norm.items():
            entry = self.errors.setdefault((name, norm), {}).get(draw)
            if entry is None:
                self.errors[(name, norm)][draw] = {"ours": ours, "bf16": tier if tier < float("inf") else None}
            else:
                entry["ours"] = max(entry["ours"], ours)
        self.prints.setdefault(draw, {})[name] = (float(reduced[5]), float(reduced[2]))
